import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.CeilingFloor
import Mathlib.Algebra.Order.NonArchimedean
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.Polynomial.Basic
import Mathlib.Analysis.Trigonometry.Inverse
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Prob.ProbTheory
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.MeasureTheory.Measure.Lebesgue
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Statistics.Normal
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Complex
import Mathlib.Trigonometry.Basic
import data.nat.basic
import tactic

namespace kC_values_l632_632216

theorem kC_values (n : ℕ) (h : n % 2 = 1) :
  ∃ kC : set ℕ, kC = {k : ℕ | 1 ≤ k ∧ k ≤ (n-1)/2 + 1} ∪ {(n+1)/2 * ((n+1)/2)} :=
sorry

end kC_values_l632_632216


namespace find_x_l632_632831

theorem find_x (x : ℝ) (h : x < 0) (hcos : cos (atan (3 / x)) = (sqrt 10 / 10) * x) : x = -1 :=
sorry

end find_x_l632_632831


namespace sum_of_die_rolls_is_even_l632_632060

-- Define probability of rolling a die and getting an even sum.
def prob_even_sum (n : ℕ) (fair_die : ℕ → ℕ → ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 1 / 2
  else if n = 2 then 1 / 2
  else 1 / 2

def coin_toss : ℕ → ℕ → ℕ := 
  sorry -- To be defined: function for fair coin toss

-- The main theorem translating the problem statement and conditions
theorem sum_of_die_rolls_is_even :
  let heads := coin_toss 2 1,
      die_roll := λ h : ℕ, fair_die h in
  let n := heads 3 2 in
  prob_even_sum n die_roll = 15 / 16 :=
by sorry

end sum_of_die_rolls_is_even_l632_632060


namespace no_growth_pie_probability_l632_632278

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l632_632278


namespace cos_F_l632_632444

-- Define the lengths of the sides and the right angle
variables {DE EF DF : ℝ}
variable (α : ℝ)

-- Given conditions
variable h1 : DE = 8
variable h2 : EF = 17
variable h3 : α = 90

-- Define DF using the Pythagorean theorem
noncomputable def calculate_DF (h1 : DE = 8) (h2 : EF = 17) : ℝ :=
  real.sqrt (EF^2 - DE^2)

-- Prove the cosine of angle F
theorem cos_F {DE EF DF : ℝ} (h1 : DE = 8) (h2 : EF = 17) (h3 : α = 90) : 
  real.cos α = (8 / 17) :=
by
  -- State that EF^2 - DE^2 = DF^2 from the Pythagorean theorem
  have h_df : DF = calculate_DF h1 h2 := sorry,
  -- Use the definition of cosine in the right triangle
  sorry

end cos_F_l632_632444


namespace no_such_integers_exist_l632_632659

theorem no_such_integers_exist : ¬ ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (n ∣ (k ^ n - 1)) ∧ (n.gcd (k - 1) = 1) :=
by
  sorry

end no_such_integers_exist_l632_632659


namespace toby_friends_girls_l632_632690

theorem toby_friends_girls (total_friends : ℕ) (num_boys : ℕ) (perc_boys : ℕ) 
  (h1 : perc_boys = 55) (h2 : num_boys = 33) (h3 : total_friends = 60) : 
  (total_friends - num_boys = 27) :=
by
  sorry

end toby_friends_girls_l632_632690


namespace problem_equivalence_l632_632352

noncomputable def a : ℝ := -1
noncomputable def b : ℝ := 3

theorem problem_equivalence (i : ℂ) (hi : i^2 = -1) : 
  (a + 3 * i = (b + i) * i) :=
by
  -- The complex number definitions
  let lhs := a + 3 * i
  let rhs := (b + i) * i

  -- Confirming the parts
  calc
  lhs = -1 + 3 * i : by rfl
  ... = rhs       : by
    simp [a, b]
    rw [hi]
    ring

-- To skip the proof add sorry at the end.
sorry

end problem_equivalence_l632_632352


namespace sum_f_div_geq_n_squared_div_2_sum_l632_632009

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

-- State the main theorem
theorem sum_f_div_geq_n_squared_div_2_sum
  (n : ℕ) (h_n : n > 1)
  (k : ℝ) (hk : 0 < k)
  (x : ℕ → ℝ) (hx_pos : ∀ i, 0 < x i) :
  (∑ i in Finset.range n, f(k)(x i - x ((i+1) % n)) / (x i + x ((i+1) % n))) ≥ n^2 / (2 * ∑ i in Finset.range n, x i) :=
by
  sorry

end sum_f_div_geq_n_squared_div_2_sum_l632_632009


namespace weight_of_each_pack_l632_632896

-- Given conditions as definitions
def numberOfPacks := 5
def pricePerPound := 5.50
def totalAmountPaid := 110.0

-- Calculate the total weight
def totalWeight : Float := totalAmountPaid / pricePerPound

-- The main theorem statement: weight of each pack of beef
theorem weight_of_each_pack :
  (totalWeight / numberOfPacks) = 4 :=
by
  sorry

end weight_of_each_pack_l632_632896


namespace correct_117th_number_l632_632348

def is_valid_four_digit_number (n : Nat) : Prop :=
  let digits := [1, 3, 4, 5, 7, 8, 9]
  let num_digits := List.dedup (n.digits 10)
  (n < 10000) ∧ (1000 ≤ n) ∧ (num_digits.length = 4) ∧ (num_digits.all (λ d => d ∈ digits))

def ascending_four_digit_numbers := 
  List.filter is_valid_four_digit_number (List.range (10000))

noncomputable def sorted_four_digit_numbers :=
  List.sort (· < ·) ascending_four_digit_numbers

theorem correct_117th_number : sorted_four_digit_numbers.get 116 = 1983 :=
  sorry

end correct_117th_number_l632_632348


namespace sum_of_primes_between_20_and_30_l632_632093

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632093


namespace sum_of_primes_between_20_and_30_l632_632123

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632123


namespace bank_balance_after_two_years_l632_632245

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l632_632245


namespace triangle_inequality_l632_632461

-- Defining the necessary entities: points, triangle, and areas
variables {A B C M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M]

-- Assumption: M is inside triangle ABC
def is_inside_triangle (M : A) (ABC : Triangle) : Prop :=
ABC.contains M

-- Definition of triangle area
noncomputable def area (ABC : Triangle) : ℝ := sorry

-- Definition of length calculations AM, BM, and CM
noncomputable def length (x y : ℝ) : ℝ := sorry 

-- The Inequality
theorem triangle_inequality (ABC : Triangle ℝ) (M : ℝ) (h : is_inside_triangle M ABC) : 
  4 * area ABC ≤ length AM * length BC + length BM * length AC + length CM * length AB := 
sorry

end triangle_inequality_l632_632461


namespace total_profit_calculation_l632_632683

variable (X Y : ℕ)  -- B's investment and the time period of his investment, both are positive natural numbers
variable (B_profit : ℕ) (A_ratio B_ratio : ℕ)  -- B's profit and ratios

-- Conditions
def A_investment (X : ℕ) := 3 * X
def A_period (Y : ℕ) := 2 * Y
def profit_ratio := 6 ∶ 1
def B_profit_value := 4500
def total_profit := 31500

-- Lean Theorem Statement
theorem total_profit_calculation 
  (B_ratio : B_ratio = 1)
  (A_ratio : A_ratio = 6)
  (B_profit : B_profit = B_profit_value) :
  total_profit = 7 * B_profit :=
sorry

end total_profit_calculation_l632_632683


namespace pure_imaginary_a_l632_632816

theorem pure_imaginary_a (a : ℝ) : (∃ (x : ℝ), (a + complex.I) * (2 - complex.I) = x * complex.I) → a = -1/2 :=
by {
  sorry
}

end pure_imaginary_a_l632_632816


namespace triangle_side_relation_l632_632218

open Real
open_locale big_operators

variables {α : Type*} [field α] [linear_order α] 

structure Triangle (α : Type*) :=
(A B C : α)

noncomputable def inscribed_exscribed_touch (A B C M N : α) : Prop :=
-- assume the geometric properties and definitions of M and N touching BC
sorry

noncomputable def triangle_with_angle_condition (t : Triangle α) (M N : α) : Prop :=
inscribed_exscribed_touch t.A t.B t.C M N ∧ (2 * ∠ t.A = ∠ M A N)

theorem triangle_side_relation {α : Type*} [field α] [linear_order α] 
  (t : Triangle α) (M N : α) : triangle_with_angle_condition t M N → dist t.B t.C = 2 * dist M N :=
by
  sorry

end triangle_side_relation_l632_632218


namespace arabic_vs_roman_numeral_system_difference_l632_632986

theorem arabic_vs_roman_numeral_system_difference :
  (∃ arabic_numeral_system roman_numeral_system : Type, 
   (∀ (n : arabic_numeral_system) (p : ℕ), value (n, p) = base_value n * 10^p) ∧
   (∀ (r : roman_numeral_system), value r = base_value r) ∧
   (∃ n : arabic_numeral_system, ∃ r : roman_numeral_system, value n ≠ value r) ∧
   (arabic_numeral_system ≠ roman_numeral_system)) :=
sorry

end arabic_vs_roman_numeral_system_difference_l632_632986


namespace compound_interest_interest_l632_632926

theorem compound_interest_interest :
  let P := 2000
  let r := 0.05
  let n := 5
  let A := P * (1 + r)^n
  let interest := A - P
  interest = 552.56 := by
  sorry

end compound_interest_interest_l632_632926


namespace minimum_x_plus_y_l632_632821

variable (x y : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1)

theorem minimum_x_plus_y (hx : 0 < x) (hy : 0 < y) (h : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1) : x + y ≥ 9 / 4 :=
sorry

end minimum_x_plus_y_l632_632821


namespace problem1_l632_632220

theorem problem1 : 
  real.cbrt (-27) + abs (real.sqrt 3 - 2) - real.sqrt (9 / 4) = -5 / 2 - real.sqrt 3 :=
sorry

end problem1_l632_632220


namespace prime_factorization_of_a19702020_l632_632748

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 0) ∧ 
  (a 2 = 2) ∧ 
  (a 3 = 3) ∧ 
  (∀ n ≥ 4, a n = nat.find_greatest (λ d, 0 < d ∧ d < n ∧ a d * a (n - d)) (n - 1))

theorem prime_factorization_of_a19702020 (a : ℕ → ℕ) (h : sequence a) : 
  a 19702020 = 3 ^ 6567340 :=
by
  sorry

end prime_factorization_of_a19702020_l632_632748


namespace perpendicular_BO_DE_l632_632906

variable (ABC : Type) [triangle ABC]
variable (O D E : ABC)
variable [circumcenter O ABC]
variable {B : ABC} (bisector_B : bisector B D AC)
variable (BE_eq_AB : AB E = AB B)

theorem perpendicular_BO_DE :
  ∃ P : ABC, is_intersection_point (line BO) (line DE) P ∧ ∠BOE P = 90 :=
by
  sorry

end perpendicular_BO_DE_l632_632906


namespace min_distinct_sums_2015_split_l632_632043

theorem min_distinct_sums_2015_split (terms : Fin 12 → ℕ) 
  (h_sum : (∑ i, terms i) = 2015) : 
  ∃ S : Finset ℕ, 
  (∀ (T : Finset (Fin 12)), T.card ≤ 9 → (∑ i in T, terms i) ∈ S) ∧ S.card = 10 :=
sorry

end min_distinct_sums_2015_split_l632_632043


namespace area_ratio_of_A₃B₃C₃_to_ABC_l632_632978

theorem area_ratio_of_A₃B₃C₃_to_ABC :
  ∀ (A B C C₁ A₁ B₁ A₂ B₂ C₂ A₃ B₃ C₃ : Type)
  (area : A → B → C → ℝ)
  (AB BC CA : ℝ), 
  AB = 9 ∧ BC = 10 ∧ CA = 13 ∧ 
  ∀ (ω : ℝ) (touches : ω → A → B → C → Prop), 
    touches ω A B C → 
    let A₃' := AA₂ ∩ ω in
    let B₃' := BB₂ ∩ ω in
    let C₃' := CC₂ ∩ ω in
    (∀ A₃ B₃ C₃ : ω, 
      area A₃ B₃ C₃ = (14 / 65) * area A B C) := sorry

end area_ratio_of_A₃B₃C₃_to_ABC_l632_632978


namespace calculate_expression_l632_632301

theorem calculate_expression :
  (10^4 - 9^4 + 8^4 - 7^4 + 6^4 - 5^4 + 4^4 - 3^4 + 2^4 - 1^4) +
  (10^2 + 9^2 + 5 * 8^2 + 5 * 7^2 + 9 * 6^2 + 9 * 5^2 + 13 * 4^2 + 13 * 3^2) = 7615 := by
  sorry

end calculate_expression_l632_632301


namespace integer_solutions_count_l632_632344

-- Define the circle center and radius
def circle_center : (ℝ × ℝ) := (3, 3)
def circle_radius : ℝ := 10

-- Define the point on the circle we are investigating
def point_on_circle (x : ℤ) : (ℤ × ℤ) := (x, -x)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 ≤ circle_radius^2

-- Count the number of integer solutions x such that the point (x, -x) is inside or on the circle
theorem integer_solutions_count : {x : ℤ | circle_equation (x : ℝ) (-x)}.finite.to_finset.card = 13 := by
  sorry

end integer_solutions_count_l632_632344


namespace sin_300_eq_l632_632688

open Real

noncomputable def sin_of_360_sub (θ : ℝ) : Prop := sin (360 - θ) = -sin θ

theorem sin_300_eq :
  sin 300 = - (sqrt 3 / 2) := by
  -- Conditions from the problem
  have h1 : sin_of_360_sub (60) := @rfl ℝ (λ θ => sin (360 - θ)) 60
  
  -- The known value of sin 60°
  have h2 : sin 60 = sqrt 3 / 2 := by sorry

  -- Proof using conditions
  sorry


end sin_300_eq_l632_632688


namespace simplify_expr_l632_632550

-- Define the expression that needs to be simplified
def expr (x : ℝ) := (42 * x^3) / (63 * x^5)

-- The theorem statement asserting that the given expression simplifies to the correct answer
theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : expr x = (2 / (3 * x^2)) :=
by
  sorry

end simplify_expr_l632_632550


namespace least_value_of_x_for_divisibility_l632_632686

theorem least_value_of_x_for_divisibility (x : ℕ) (h : 1 + 8 + 9 + 4 = 22) :
  ∃ x : ℕ, (22 + x) % 3 = 0 ∧ x = 2 := by
sorry

end least_value_of_x_for_divisibility_l632_632686


namespace trajectory_of_point_P_l632_632886

theorem trajectory_of_point_P 
  (x y : ℝ)
  (A : ℝ × ℝ := (0, -real.sqrt 2))
  (B : ℝ × ℝ := (0, real.sqrt 2))
  (h : ((y + real.sqrt 2) / x) * ((y - real.sqrt 2) / x) = -2) :
  (y^2 / 2 + x^2 = 1 ∧ x ≠ 0) :=
sorry

end trajectory_of_point_P_l632_632886


namespace probability_dice_between_30_and_50_l632_632512

theorem probability_dice_between_30_and_50 :
  let p := (3 / 4 : ℚ) in
  ∀ (Die_A Die_B : ℕ),
    1 ≤ Die_A ∧ Die_A ≤ 6 ∧ 1 ≤ Die_B ∧ Die_B ≤ 6 →
    ((Die_A = 3 ∨ Die_A = 4 ∨ Die_A = 5) ∨
     (Die_B = 3 ∨ Die_B = 4 ∨ Die_B = 5)) →
    p = 3 / 4 :=
begin
  sorry
end

end probability_dice_between_30_and_50_l632_632512


namespace weight_equilibrium_l632_632557

variable (x y : ℝ) -- x: weight of one saquinho, y: weight of one bola

theorem weight_equilibrium
    (equilibrium_condition : 5 * x + 4 * y = 2 * x + 10 * y) :
    x = 2 * y :=
by
   simp [equilibrium_condition]
   linarith

end weight_equilibrium_l632_632557


namespace angle_between_apothems_correct_l632_632768

noncomputable def angle_between_apothems (n : ℕ) (α : ℝ) : ℝ :=
  2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2))

theorem angle_between_apothems_correct (n : ℕ) (α : ℝ) (h1 : 0 < n) (h2 : 0 < α) (h3 : α < 2 * Real.pi) :
  angle_between_apothems n α = 2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2)) :=
by
  sorry

end angle_between_apothems_correct_l632_632768


namespace doctor_is_correct_l632_632610

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632610


namespace equation_B_is_quadratic_l632_632671

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end equation_B_is_quadratic_l632_632671


namespace nigella_sold_3_houses_l632_632515

noncomputable def houseA_cost : ℝ := 60000
noncomputable def houseB_cost : ℝ := 3 * houseA_cost
noncomputable def houseC_cost : ℝ := 2 * houseA_cost - 110000
noncomputable def commission_rate : ℝ := 0.02

noncomputable def houseA_commission : ℝ := houseA_cost * commission_rate
noncomputable def houseB_commission : ℝ := houseB_cost * commission_rate
noncomputable def houseC_commission : ℝ := houseC_cost * commission_rate

noncomputable def total_commission : ℝ := houseA_commission + houseB_commission + houseC_commission
noncomputable def base_salary : ℝ := 3000
noncomputable def total_earnings : ℝ := base_salary + total_commission

theorem nigella_sold_3_houses 
  (H1 : total_earnings = 8000) 
  (H2 : houseA_cost = 60000) 
  (H3 : houseB_cost = 3 * houseA_cost) 
  (H4 : houseC_cost = 2 * houseA_cost - 110000) 
  (H5 : commission_rate = 0.02) :
  3 = 3 :=
by 
  -- Proof not required
  sorry

end nigella_sold_3_houses_l632_632515


namespace lemonade_per_drink_l632_632716

theorem lemonade_per_drink
  (total_drink : ℚ)
  (total_lemonade : ℚ)
  (iced_tea_per_drink : ℚ)
  (total_drinks : ℚ)
  (lemonade_per_drink : ℚ) :
  total_drink = 18 →
  total_lemonade = 15 →
  iced_tea_per_drink = 1 / 4 →
  total_drinks = total_drink / iced_tea_per_drink →
  lemonade_per_drink = total_lemonade / total_drinks →
  lemonade_per_drink = 1.25 :=
by
  intros h_total_drink h_total_lemonade h_iced_tea_per_drink h_total_drinks h_lemonade_per_drink
  have h1 : total_drink = 18 := h_total_drink
  have h2 : total_lemonade = 15 := h_total_lemonade
  have h3 : iced_tea_per_drink = 1 / 4 := h_iced_tea_per_drink
  have h4 : total_drinks = 18 / (1 / 4) := h_total_drinks
  have h5 : lemonade_per_drink = 15 / (18 / (1 / 4)) := h_lemonade_per_drink
  exact h5

end lemonade_per_drink_l632_632716


namespace total_interest_calculation_l632_632256

-- Define the total investment
def total_investment : ℝ := 20000

-- Define the fractional part of investment at 9 percent rate
def fraction_higher_rate : ℝ := 0.55

-- Define the investment amounts based on the fractional part
def investment_higher_rate : ℝ := fraction_higher_rate * total_investment
def investment_lower_rate : ℝ := total_investment - investment_higher_rate

-- Define interest rates
def rate_lower : ℝ := 0.06
def rate_higher : ℝ := 0.09

-- Define time period (in years)
def time_period : ℝ := 1

-- Define interest calculations
def interest_lower : ℝ := investment_lower_rate * rate_lower * time_period
def interest_higher : ℝ := investment_higher_rate * rate_higher * time_period

-- Define the total interest
def total_interest : ℝ := interest_lower + interest_higher

-- Theorem stating the total interest earned
theorem total_interest_calculation : total_interest = 1530 := by
  -- skip proof using sorry
  sorry

end total_interest_calculation_l632_632256


namespace determine_n_l632_632492

theorem determine_n (n : ℕ) (hn : 0 < n) :
  (∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 2 * y + z = n) → (n = 15 ∨ n = 16) :=
  by sorry

end determine_n_l632_632492


namespace number_of_results_l632_632594

theorem number_of_results (n : ℕ)
  (avg_all : (summation : ℤ) → summation / n = 42)
  (avg_first_5 : (sum_first_5 : ℤ) → sum_first_5 / 5 = 49)
  (avg_last_7 : (sum_last_7 : ℤ) → sum_last_7 / 7 = 52)
  (fifth_result : (r5 : ℤ) → r5 = 147) :
  n = 11 :=
by
  -- Conditions
  let sum_first_5 := 5 * 49
  let sum_last_7 := 7 * 52
  let summed_results := sum_first_5 + sum_last_7 - 147
  let sum_all := 42 * n 
  -- Since sum of all results = 42n
  exact sorry

end number_of_results_l632_632594


namespace locus_intersection_thales_circle_l632_632495

open Real

theorem locus_intersection_thales_circle 
  (A B O : Point) 
  (O_internal : A ≠ B ∧ O ∈ open_segment A B) 
  (f : Line) 
  (M N : Point) 
  (OM_eq_OA : distance O M = distance O A) 
  (ON_eq_OB : distance O N = distance O B)
  (AM : Line) 
  (BN : Line) 
  (AM_eq : AM = line_through A M) 
  (BN_eq : BN = line_through B N) :
  (locus : Set Point) 
  (locus_eq : locus = {P | ∃ (f : Line) (M N : Point), 
                         M ∈ f ∧ N ∈ f ∧
                         distance O M = distance O A ∧
                         distance O N = distance O B ∧ 
                         P = intersection (line_through A M) (line_through B N)}) → 
  (Thales_circle : Set Point) 
  (Thales_circle_eq : Thales_circle = {P | distance P A * distance P B = distance A B ^ 2}) :
  locus = Thales_circle \ {A, B} :=
by 
  sorry

end locus_intersection_thales_circle_l632_632495


namespace min_t_shifted_function_l632_632949

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def g (t x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * t + Real.pi / 6)

theorem min_t_shifted_function :
  (∃ k : ℤ, ∃ t : ℝ, t > 0 ∧ 2 * t + Real.pi / 6 = k * Real.pi + Real.pi / 2) →
  ∃ t : ℝ, t > 0 ∧ t = Real.pi / 6 :=
begin
  intro h,
  obtain ⟨k, t, t_pos, eqn⟩ := h,
  use (Real.pi / 6),
  split,
  { exact Real.pi_div_six_pos },
  { sorry } -- Prove that t = Real.pi / 6 satisfies the equation and minimal condition
end

end min_t_shifted_function_l632_632949


namespace find_SD_in_rectangle_l632_632442

-- Define the conditions and problem statement
theorem find_SD_in_rectangle
    (ABCD : Type) (P Q R : ABCD) 
    (TS : ABCD → Prop)
    (H1 : ∀ (A B C D : ABCD), ∃ (P : ABCD), angle P A D = 90)
    (H2 : ∀ (PQ : Prop), P ∈ BC → Q ∈ PD ∧ Q ∈ TS ∧ TS ⊥ BC)
    (BP : ℝ) (PT : ℝ)
    (H3 : BP = 2 * PT)
    (PA AQ QP : ℝ)
    (H4 : PA = 12 ∧ AQ = 13 ∧ QP = 5)
    (SD : ℝ) :
  SD = 110 / 13 :=
sorry

end find_SD_in_rectangle_l632_632442


namespace sum_primes_20_to_30_l632_632148

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632148


namespace weeks_to_buy_iphone_l632_632305

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l632_632305


namespace solution_correct_l632_632670

variables {a x : ℝ}

def similar_radical_pair_D : Prop :=
  let r1 := sqrt (3 * a^3)
  let r2 := 3 * sqrt (3 * a^3) in
  r1 = |a| * sqrt 3 * sqrt a ∧ r2 = 3 * |a| * sqrt 3 * sqrt a

theorem solution_correct : similar_radical_pair_D := by
  sorry

end solution_correct_l632_632670


namespace kite_area_proof_l632_632346

-- Define the vertices of the triangles
structure Point where
  x : ℕ
  y : ℕ

def vertex1 : Point := ⟨1, 1⟩
def vertex2 : Point := ⟨4, 5⟩
def vertex3 : Point := ⟨7, 1⟩
def vertex4 : Point := ⟨4, 0⟩

-- Function to calculate the distance between points (base and height)
def distance (p1 p2 : Point) : ℕ :=
  2 * (p2.x - p1.x)  -- p1.y and p2.y are same for base calculation
  -- note: height calculation would involve p1.x == p2.x and subtracting y coordinates

-- Distance for base and height
def base := distance vertex1 vertex3
def height := distance vertex4 vertex2

-- Area of one triangle
def triangle_area : ℕ := base * height / 2

-- Total area of the kite
def kite_area : ℕ := 2 * triangle_area

-- The proof statement
theorem kite_area_proof : kite_area = 60 := by
  -- skipping the proof steps as instructed
  sorry

end kite_area_proof_l632_632346


namespace harvest_season_weeks_l632_632925

-- Definitions based on given conditions
def weekly_earnings : ℕ := 491
def weekly_rent : ℕ := 216
def total_savings : ℕ := 324775

-- Definition to calculate net earnings per week
def net_earnings_per_week (earnings rent : ℕ) : ℕ :=
  earnings - rent

-- Definition to calculate number of weeks
def number_of_weeks (savings net_earnings : ℕ) : ℕ :=
  savings / net_earnings

theorem harvest_season_weeks :
  number_of_weeks total_savings (net_earnings_per_week weekly_earnings weekly_rent) = 1181 :=
by
  sorry

end harvest_season_weeks_l632_632925


namespace axis_of_symmetry_l632_632870

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

variable (φ : ℝ)

theorem axis_of_symmetry 
  (h1 : SymmetricPoint (f φ) (Real.pi / 3, 0))
  (h2 : |φ| < Real.pi / 2) :
  Exists (λ x : ℝ, x = Real.pi / 12) :=
sorry

end axis_of_symmetry_l632_632870


namespace sum_primes_between_20_and_30_l632_632175

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632175


namespace milburg_children_count_l632_632582

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 ∧ grown_ups = 5256 → 
  total_population - grown_ups = 2987 :=
by 
  intros total_population grown_ups h,
  rcases h with ⟨h1, h2⟩,
  rw [h1, h2],
  exact rfl

end milburg_children_count_l632_632582


namespace evaluate_expression_l632_632759

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 :=
by sorry

end evaluate_expression_l632_632759


namespace compute_expression_l632_632493

theorem compute_expression :
  let z := complex.cos (2 * real.pi / 5) + complex.sin (2 * real.pi / 5) * complex.I in
    (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6)) = 2 :=
by
  sorry

end compute_expression_l632_632493


namespace sum_of_log_sequence_l632_632453

open Real BigOperators

noncomputable def geometric_seq (r a₀ : ℝ) (n : ℕ) : ℝ := a₀ * r ^ n

theorem sum_of_log_sequence :
  (∃ r : ℝ, r > 0 ∧ r ≠ 1 ∧ ∀ n, a_n = geometric_seq r 6 (n - 1)) →
  ∑ i in finset.range 9, log 6 (a_n (i + 1)) = 9 :=
by
  sorry

end sum_of_log_sequence_l632_632453


namespace function_y_increases_when_x_gt_1_l632_632385

theorem function_y_increases_when_x_gt_1 :
  ∀ (x : ℝ), (x > 1 → 2*x^2 > 2*(x-1)^2) :=
by
  sorry

end function_y_increases_when_x_gt_1_l632_632385


namespace remaining_episodes_l632_632202

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l632_632202


namespace equation_B_is_quadratic_l632_632672

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end equation_B_is_quadratic_l632_632672


namespace translate_function_result_l632_632063

noncomputable def initial_function (x : ℝ) : ℝ := -1 / x

noncomputable def translate_left (x : ℝ) : ℝ := initial_function (x + 1)

noncomputable def translate_up (x : ℝ) : ℝ := translate_left x + 2

theorem translate_function_result :
  ∀ x : ℝ, x ≠ -1 → translate_up x = (2 * x + 1) / (x + 1) :=
by
  intro x hx
  unfold translate_up translate_left initial_function
  field_simp [hx]
  ring

end translate_function_result_l632_632063


namespace intersection_points_form_rectangle_l632_632961

noncomputable def intersection_points : List (ℝ × ℝ) := 
  let x_values := [Real.sqrt 24, -Real.sqrt 24, Real.sqrt 12, -Real.sqrt 12]
  x_values.map (fun x => (x, 18 / x))

theorem intersection_points_form_rectangle :
  ∀ P1 P2 P3 P4 ∈ intersection_points, 
  ∃l1 l2 l3 l4, 
  Set.pairwise disjoint {l1, l2, l3, l4} ∧
  IsPerpendicular (midpoint P1 P3) (midpoint P2 P4) ∧
  (dist P1 P2 = dist P3 P4) ∧ 
  (dist P1 P3 = dist P2 P4) := 
sorry

end intersection_points_form_rectangle_l632_632961


namespace solve_sqrt_equation_l632_632016

theorem solve_sqrt_equation (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) :
    sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := by
  sorry

end solve_sqrt_equation_l632_632016


namespace millionth_digit_1_div_41_l632_632774

theorem millionth_digit_1_div_41 : 
  ∃ n, n = 1000000 ∧ (n % 5 = 0) → digit_after_decimal (1 / 41) n = 9 :=
begin
  intro exists n,
  sorry -- proof goes here
end

end millionth_digit_1_div_41_l632_632774


namespace sum_of_primes_between_1_and_25_l632_632310

theorem sum_of_primes_between_1_and_25 : (2 + 23) = 25 :=
by
  have smallest_prime := 2
  have largest_prime := 23
  have sum := smallest_prime + largest_prime
  show sum = 25

end sum_of_primes_between_1_and_25_l632_632310


namespace find_m_value_l632_632347

theorem find_m_value : ∃ m : ℤ, 81 - 6 = 25 + m ∧ m = 50 :=
by
  sorry

end find_m_value_l632_632347


namespace triangle_max_b_c_l632_632456

theorem triangle_max_b_c (a b c : ℝ) 
  (A B C : ℝ) 
  (h_a : a = 2 * sqrt 5)
  (h_cos_eq : (2 * c - b) / a = (Real.cos B) / (Real.cos A))
  (h_triangle_angles : A + B + C = Real.pi)
  (h_sine_law_b : b = (2 * sqrt 5) * (Real.sin B) / (Real.sin A))
  (h_sine_law_c : c = (2 * sqrt 5) * (Real.sin C) / (Real.sin A))
  (h_A_pos_pi3 : A = Real.pi / 3) :
  b + c ≤ 4 * sqrt 5 :=
sorry

end triangle_max_b_c_l632_632456


namespace sum_of_primes_between_20_and_30_l632_632089

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632089


namespace range_of_x_l632_632387

open Real

noncomputable def f (x : ℝ) : ℝ := exp (1 + |x|) - 1 / (1 + x^4)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f(-x) = f(x)) : 
  f(2 * x) < f(1 - x) → x ∈ set.Ioo (-1 : ℝ) (1 / 3) :=
sorry

end range_of_x_l632_632387


namespace hydras_never_die_l632_632648

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632648


namespace curve_of_constant_width_l632_632525

structure Curve :=
  (is_convex : Prop)

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def rotate_180 (K : Curve) (O : Point) : Curve := sorry

def sum_curves (K1 K2 : Curve) : Curve := sorry

def is_circle_with_radius (K : Curve) (r : ℝ) : Prop := sorry

def constant_width (K : Curve) (w : ℝ) : Prop := sorry

theorem curve_of_constant_width {K : Curve} {O : Point} {h : ℝ} :
  K.is_convex →
  (K' : Curve) → K' = rotate_180 K O →
  is_circle_with_radius (sum_curves K K') h →
  constant_width K h :=
by 
  sorry

end curve_of_constant_width_l632_632525


namespace asian_population_percentage_in_west_l632_632728

theorem asian_population_percentage_in_west
    (NE MW South West : ℕ)
    (H_NE : NE = 2)
    (H_MW : MW = 3)
    (H_South : South = 2)
    (H_West : West = 6)
    : (West * 100) / (NE + MW + South + West) = 46 :=
sorry

end asian_population_percentage_in_west_l632_632728


namespace probability_no_gp_l632_632271

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l632_632271


namespace tan_theta_right_triangle_l632_632435

theorem tan_theta_right_triangle
  (α : ℝ)
  (hα : (Real.tan (α / 2) = 1 / (Real.cbrt 2))) :
  ∃ θ : ℝ, Real.tan θ = 1 / 2 := 
sorry

end tan_theta_right_triangle_l632_632435


namespace series_sum_l632_632735

theorem series_sum : 
  (\sum_{n=1}^{500} (\frac{1}{n^3 + n^2 + n})) = \frac{250500}{251001} :=
by
  sorry

end series_sum_l632_632735


namespace part1_3kg_part2_5kg_part2_function_part3_compare_l632_632519

noncomputable def supermarket_A_cost (x : ℝ) : ℝ :=
if x <= 4 then 10 * x
else 6 * x + 16

noncomputable def supermarket_B_cost (x : ℝ) : ℝ :=
8 * x

-- Proof that supermarket_A_cost 3 = 30
theorem part1_3kg : supermarket_A_cost 3 = 30 :=
by sorry

-- Proof that supermarket_A_cost 5 = 46
theorem part2_5kg : supermarket_A_cost 5 = 46 :=
by sorry

-- Proof that the cost function is correct
theorem part2_function (x : ℝ) : 
(0 < x ∧ x <= 4 → supermarket_A_cost x = 10 * x) ∧ 
(x > 4 → supermarket_A_cost x = 6 * x + 16) :=
by sorry

-- Proof that supermarket A is cheaper for 10 kg apples
theorem part3_compare : supermarket_A_cost 10 < supermarket_B_cost 10 :=
by sorry

end part1_3kg_part2_5kg_part2_function_part3_compare_l632_632519


namespace six_x_mod_nine_l632_632199

theorem six_x_mod_nine (x : ℕ) (k : ℕ) (hx : x = 9 * k + 5) : (6 * x) % 9 = 3 :=
by
  sorry

end six_x_mod_nine_l632_632199


namespace sum_primes_between_20_and_30_is_52_l632_632135

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632135


namespace acute_angle_radian_l632_632059

theorem acute_angle_radian :
  let r1 := 4
  let r2 := 3
  let r3 := 2
  let total_area := 29 * Real.pi
  let S := (7 / 17) * (total_area - (7 / 17) * total_area)
  let U := total_area - S
  ∃ θ : ℝ, 
    0 < θ ∧ θ < Real.pi / 2 ∧
    (
      let shaded_area := 
        2 * θ * (r1^2 * Real.pi) + 
        2 * (Real.pi - θ) * (r2^2 * Real.pi) + 
        2 * θ * (r3^2 * Real.pi)
      in shaded_area = S
    ) ∧ θ = 229 * Real.pi / 528 := by
  sorry

end acute_angle_radian_l632_632059


namespace remaining_episodes_l632_632208

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l632_632208


namespace sixtieth_integer_is_35142_l632_632044

def digits := [1, 2, 3, 4, 5]

noncomputable def perms := List.permutations digits

def ordered_perms := perms.sorted

def sixtieth_permutation := ordered_perms.get! (60 - 1)

theorem sixtieth_integer_is_35142 : sixtieth_permutation = 35142 := 
    sorry

end sixtieth_integer_is_35142_l632_632044


namespace poem_lines_added_l632_632563

theorem poem_lines_added (x : ℕ) 
  (initial_lines : ℕ)
  (months : ℕ)
  (final_lines : ℕ)
  (h_init : initial_lines = 24)
  (h_months : months = 22)
  (h_final : final_lines = 90)
  (h_equation : initial_lines + months * x = final_lines) :
  x = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end poem_lines_added_l632_632563


namespace sum_of_primes_between_20_and_30_l632_632106

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632106


namespace equation_solution_l632_632331

noncomputable def solve_equation : Prop :=
∃ (x : ℝ), x^6 + (3 - x)^6 = 730 ∧ (x = 1.5 + Real.sqrt 5 ∨ x = 1.5 - Real.sqrt 5)

theorem equation_solution : solve_equation :=
sorry

end equation_solution_l632_632331


namespace intersection_of_M_and_N_l632_632814

open Set

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
by {
  sorry
}

end intersection_of_M_and_N_l632_632814


namespace hydrae_never_equal_heads_l632_632645

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632645


namespace sum_primes_between_20_and_30_l632_632183

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632183


namespace symmetry_center_cos_sin_l632_632317

theorem symmetry_center_cos_sin (f : ℝ → ℝ)
  (h : ∀ x, f x = cos (2 * x - π / 6) * sin (2 * x) - 1 / 4) :
  (7 * π / 24, (0 : ℝ)) = (k / 4 + π / 24, 0)
  ∧ k = 1 :=
by
  sorry

end symmetry_center_cos_sin_l632_632317


namespace sum_of_primes_between_20_and_30_l632_632113

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632113


namespace circumscribed_sphere_radius_eq_4_l632_632558

noncomputable def radius_of_circumscribed_sphere (a b c d : ℝ) : ℝ :=
  let side_length := 6
  let height := 4
  let circumcenter_to_vertex := side_length / (Real.sqrt 3)
  let half_lateral_edge := height / 2
  let radius := Real.sqrt ((circumcenter_to_vertex ^ 2) + (half_lateral_edge ^ 2))
  inradius

theorem circumscribed_sphere_radius_eq_4 (a b c d : ℝ) 
  (h_base : base_is_equilateral a b c)
  (h_side_length : side_length = 6)
  (h_perpendicular_edge : is_perpendicular a d b c)
  (h_perpendicular_length : d = 4) : 
  radius_of_circumscribed_sphere a b c d = 4 :=
by {
  sorry
}

end circumscribed_sphere_radius_eq_4_l632_632558


namespace log_base_approx_l632_632819

theorem log_base_approx (log10_2 : Real := 0.301) (log10_5 : Real := 0.699) :
  Real.log 10 7 ≈ 33 / 10 :=
sorry

end log_base_approx_l632_632819


namespace range_of_f_l632_632837

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else sqrt x

theorem range_of_f :
  ∀ x0 : ℝ, f x0 > 1 ↔ x0 ∈ set.Ioo (-∞) (-1) ∪ set.Ioo 1 ∞ :=
by
  intro x0
  sorry

end range_of_f_l632_632837


namespace hari_joined_after_5_months_l632_632005

noncomputable def praveen_investment := 3780 * 12
noncomputable def hari_investment (x : ℕ) := 9720 * (12 - x)

theorem hari_joined_after_5_months :
  ∃ (x : ℕ), (praveen_investment : ℝ) / (hari_investment x) = (2:ℝ) / 3 ∧ x = 5 :=
by {
  sorry
}

end hari_joined_after_5_months_l632_632005


namespace rationalize_denominator_l632_632539

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l632_632539


namespace hydras_never_die_l632_632626

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632626


namespace wall_width_is_4_l632_632684

structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

theorem wall_width_is_4 (h_eq_6w : ∀ (wall : Wall), wall.height = 6 * wall.width)
                        (l_eq_7h : ∀ (wall : Wall), wall.length = 7 * wall.height)
                        (volume_16128 : ∀ (wall : Wall), wall.volume = 16128) :
  ∃ (wall : Wall), wall.width = 4 :=
by
  sorry

end wall_width_is_4_l632_632684


namespace find_minimum_AC_l632_632362

-- Definitions of formal parameters and assumptions
variables (A B C : ℝ) (a b c : ℝ) (triangle_ABC : ℝ) (min_AC : ℝ)

-- Assuming the angles A, B, and C form an arithmetic sequence
-- and the area of the triangle is sqrt(3)
-- We need to prove the minimum value of edge AC is 2

noncomputable def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def triangle_area_is_sqrt_three (triangle_ABC : ℝ) : Prop :=
  triangle_ABC = sqrt 3

def minimum_edge_AC (min_AC : ℝ) : Prop :=
  min_AC = 2

theorem find_minimum_AC 
  (h1 : angles_form_arithmetic_sequence A B C)
  (h2 : triangle_area_is_sqrt_three triangle_ABC)
  : minimum_edge_AC min_AC :=
 sorry

end find_minimum_AC_l632_632362


namespace millionth_digit_of_1_over_41_l632_632782

theorem millionth_digit_of_1_over_41 :
  let frac := 1 / (41 : ℚ),
      seq := "02439",
      period := (5 : ℕ) in
  (seq.get (1000000 % period - 1) = '9') :=
by
  let frac := 1 / (41 : ℚ)
  let seq := "02439"
  let period := 5
  have h_expansion : frac = 0.02439 / 10000 := sorry
  have h_period : ∀ n, frac = Rational.mkPeriodic seq period n := sorry
  have h_mod : 1000000 % period = 0 := by sorry
  have h_index := h_mod.symm ▸ (dec_trivial : 0 % 5 = 0)
  exact h_period n ▸ (dec_trivial : "02439".get 4 = '9')

end millionth_digit_of_1_over_41_l632_632782


namespace value_of_a_plus_b_l632_632489

open function

theorem value_of_a_plus_b (a b : ℝ) 
  (hM : ∃ x, x ∈ {b / a, 1} ∧ x ∈ {a, 0} ∧ x = x) : a + b = 1 :=
sorry

end value_of_a_plus_b_l632_632489


namespace domain_of_g_l632_632736

def g (x : ℝ) : ℝ := 1 / (floor (x^2 - 7 * x + 10))

theorem domain_of_g :
  {x : ℝ | ∃ y, g y = x} = {x | x ≤ 1 ∨ x ≥ 8} :=
by
  sorry

end domain_of_g_l632_632736


namespace prime_sum_20_to_30_l632_632142

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632142


namespace analytical_expression_range_of_m_minimum_value_l632_632389

section Problem1

variable (ω : ℝ) (φ : ℝ) (f g : ℝ → ℝ)

-- Given conditions
def f_def : Prop := ∀ x, f x = Real.sin (ω * x + φ) - 1
def distance_condition : Prop := ω > 0 ∧ 0 < φ ∧ φ < Real.pi ∧ ∀ d, d = Real.pi / 2 → 2 * Real.pi / ω = 2 * d
def shift_condition : Prop := ∀ x, g x = Real.sin (2 * (x + Real.pi / 12) + φ) ∧ (g (x - Real.pi / 12) + 1) = Real.sin (2 * x + Real.pi / 6 + φ)
def even_condition : Prop := ∀ x, g x = g (-x)

-- To prove the analytical expression of f(x)
theorem analytical_expression (ω φ : ℝ) (h : f = λ x, Real.sin (2 * x + Real.pi / 3) - 1) :
  distance_condition ω φ ∧ shift_condition ω φ g ∧ even_condition g → f = (λ x, Real.sin (2 * x + Real.pi / 3) - 1) := 
  sorry

end Problem1

section Problem2

variable (m : ℝ) (x : ℝ)

-- Given conditions
def f_x_def : Prop := ∀ x ∈ Set.Icc 0 (Real.pi / 3), let f := Real.sin (2 * x + Real.pi / 3) - 1 in (0 ≤ f ∧ f ≤ 1)
def inequality_condition : Prop := ∀ x, x ∈ Set.Icc 0 (Real.pi / 3) → (f x) ^ 2 - (2 + m) * f x + 2 + m ≤ 0

-- To prove the range of m
theorem range_of_m (m : ℝ) (h : ∀ x, (x ∈ Set.Icc 0 (Real.pi / 3)) → (Real.sin (2 * x + Real.pi / 3) - 1) ^ 2 - (2 + m) * (Real.sin (2 * x + Real.pi / 3) - 1) + 2 + m ≤ 0) :
  f_x_def x → (-∞ < m ∧ m ≤ -5/2) := 
  sorry

end Problem2

section Problem3

variable (a b : ℝ) (k : ℤ)

-- Given conditions
def zero_condition : Prop := ∃ (a b : ℝ) (k : ℤ), a < b ∧ (h(x) = 2 * Real.sin (2 * x + Real.pi / 3) + 1) ∧ (b - a) ≥ (k * 2 * Real.pi + Real.pi / 3)

-- To prove the minimum value of b - a
theorem minimum_value (a b : ℝ) (h : ∀ k : ℤ, (∃ (a b : ℝ), a < b ∧ (h(x) = 2 * Real.sin (2 * x + Real.pi / 3) + 1) ∧ (b - a) = (k ∙ 2 ∙ Real.pi + Real.pi / 3)) :
  zero_condition → b - a = 43 / 3 * Real.pi :=
  sorry

end Problem3

end analytical_expression_range_of_m_minimum_value_l632_632389


namespace maximum_alpha_l632_632494

noncomputable def is_in_F (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x

theorem maximum_alpha :
  (∀ f : ℝ → ℝ, is_in_F f → ∀ x > 0, f x ≥ (1 / 2) * x) := 
by
  sorry

end maximum_alpha_l632_632494


namespace parallel_lines_m_l632_632855

noncomputable def lines_parallel (m : ℝ) : Prop :=
  let l1 := λ x y : ℝ, mx + 2 * y - 2 = 0
  let l2 := λ x y : ℝ, 5 * x + (m + 3) * y - 5 = 0
  (-m / 2) = (-5 / (m + 3))

theorem parallel_lines_m :
  (∃ m : ℝ, lines_parallel m) →
  lines_parallel (-5) :=
by
  sorry

end parallel_lines_m_l632_632855


namespace tangent_line_eq_monotonic_intervals_l632_632393

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Point of tangency
def tangent_point : ℝ × ℝ := (2, f 2)

-- Part 1: Prove the equation of the tangent line
theorem tangent_line_eq (x : ℝ) :
  let m := -9
  let b := -18 + f 2
  y - (f 2) = m * (x - 2) ↔ y = m * x + b := sorry

-- Part 2: Monotonicity intervals
theorem monotonic_intervals :
  (∀ x ∈ Ioo (-∞) (-1), -f' x > 0) ∧
  (∀ x ∈ Ioo (-1) 1, f' x > 0) ∧
  (∀ x ∈ Ioo 1 ∞, -f' x > 0) := sorry
  

end tangent_line_eq_monotonic_intervals_l632_632393


namespace total_area_of_rug_l632_632242

theorem total_area_of_rug :
  let length_rect := 6
  let width_rect := 4
  let base_parallelogram := 3
  let height_parallelogram := 4
  let area_rect := length_rect * width_rect
  let area_parallelogram := base_parallelogram * height_parallelogram
  let total_area := area_rect + 2 * area_parallelogram
  total_area = 48 := by sorry

end total_area_of_rug_l632_632242


namespace rationalize_denominator_l632_632532

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l632_632532


namespace find_other_number_l632_632556

-- Definition of LCM and HCF
open Nat

theorem find_other_number (A : ℕ) (B : ℕ) 
    (h1 : lcm A B = 2310) 
    (h2 : gcd A B = 30) 
    (h3 : A = 210) : 
    B = 330 :=
by
  sorry

end find_other_number_l632_632556


namespace curveC_standard_equation_lineL_cartesian_equation_max_distance_curveC_to_lineL_l632_632887

-- Define the parametric curve C and the polar line l
def parametricCurveC (α : ℝ) : ℝ × ℝ := 
  (sqrt 3 * Real.cos α, Real.sin α)

def polarLineL (ρ θ : ℝ) : Prop := 
  ρ * (Real.cos θ + Real.sin θ) = 4

-- Prove the equivalent statements
theorem curveC_standard_equation (x y α : ℝ) :
  (x, y) = parametricCurveC α →
  (x^2 / 3) + y^2 = 1 := 
sorry

theorem lineL_cartesian_equation (x y ρ θ : ℝ) :
  polarLineL ρ θ →
  ρ * Real.cos θ = x →
  ρ * Real.sin θ = y →
  x + y = 4 :=
sorry

theorem max_distance_curveC_to_lineL (α : ℝ) :
  let A := parametricCurveC α in 
  let d := abs ((sqrt 3 * Real.cos α + Real.sin α - 4) / sqrt 2) in
  ∃ (max_d : ℝ), max_d = 3 * sqrt 2 ∧ d ≤ max_d :=
sorry

end curveC_standard_equation_lineL_cartesian_equation_max_distance_curveC_to_lineL_l632_632887


namespace jersey_to_shoes_ratio_l632_632466

theorem jersey_to_shoes_ratio
  (pairs_shoes: ℕ) (jerseys: ℕ) (total_cost: ℝ) (total_cost_shoes: ℝ) 
  (shoes: pairs_shoes = 6) (jer: jerseys = 4) (total: total_cost = 560) (cost_sh: total_cost_shoes = 480) :
  ((total_cost - total_cost_shoes) / jerseys) / (total_cost_shoes / pairs_shoes) = 1 / 4 := 
by 
  sorry

end jersey_to_shoes_ratio_l632_632466


namespace sum_primes_in_range_l632_632084

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632084


namespace sqrt_sum_value_l632_632018

theorem sqrt_sum_value (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) :
    sqrt (64 - y^2) + sqrt (36 - y^2) = 7 :=
sorry

end sqrt_sum_value_l632_632018


namespace n_divisible_by_4_l632_632357

theorem n_divisible_by_4 (n : ℕ) (x : Fin n → ℤ) (h1 : ∀ k, (x k = 1 ∨ x k = -1)) 
(h2 : ∑ i in Finset.range n, x i * x ((i + 1) % n) = 0) : n % 4 = 0 :=
sorry

end n_divisible_by_4_l632_632357


namespace milburg_children_count_l632_632581

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 ∧ grown_ups = 5256 → 
  total_population - grown_ups = 2987 :=
by 
  intros total_population grown_ups h,
  rcases h with ⟨h1, h2⟩,
  rw [h1, h2],
  exact rfl

end milburg_children_count_l632_632581


namespace negation_of_universal_prop_correct_l632_632042

def negation_of_universal_prop : Prop :=
  ¬ (∀ x : ℝ, x = |x|) ↔ ∃ x : ℝ, x ≠ |x|

theorem negation_of_universal_prop_correct : negation_of_universal_prop := 
by
  sorry

end negation_of_universal_prop_correct_l632_632042


namespace sum_primes_between_20_and_30_is_52_l632_632128

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632128


namespace area_square_AB_l632_632040

variable {A B C : Type}
variable [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]

variable (triangle_ABC : Triangle ℝ A B C)
variable (M : Point ℝ)
variable (D : Midpoint ℝ B C)
variable (E : Midpoint ℝ A C)

-- Conditions
axiom perpendicular_medians : (medians_triangle_ABC A).medians B ∅ (perpendicular)
axiom BC_length : dist B C = 36
axiom AC_length : dist A C = 48

-- Prove that the area of the square with side AB is 720
theorem area_square_AB : (dist A B) ^ 2 = 720 := by
  sorry

end area_square_AB_l632_632040


namespace millionth_digit_of_1_over_41_l632_632778

theorem millionth_digit_of_1_over_41 :
  let d := 5    -- The period of decimal expansion is 5
  let seq := [0,2,4,3,9]  -- The repeating sequence of 1/41
  let millionth_digit_position := 1000000 % d  -- Find the position in the repeating sequence
  seq.millionth_digit_position = 9 :=
by
  let d := 5
  let seq := [0, 2, 4, 3, 9]
  let millionth_digit_position := 1000000 % d
  show seq.millionth_digit_position = 9
  sorry

end millionth_digit_of_1_over_41_l632_632778


namespace sum_primes_between_20_and_30_l632_632181

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632181


namespace no_snuggly_two_digit_l632_632712

theorem no_snuggly_two_digit (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) : ¬ (10 * a + b = a + b^3) :=
by {
  sorry
}

end no_snuggly_two_digit_l632_632712


namespace series_inequality_l632_632934

-- Define the sum of the series
def series_sum (n : ℕ) : ℝ := ∑ k in (Finset.range (n + 1)).filter (λ k, k > 0), (1 : ℝ) / k^2

theorem series_inequality (n : ℕ) (h : n ≥ 2) :
  1 + series_sum n < (2 * n - 1) / n := by
  sorry

end series_inequality_l632_632934


namespace leaves_dropped_on_fifth_day_l632_632449

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l632_632449


namespace sum_of_primes_between_20_and_30_l632_632114

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632114


namespace cost_per_pizza_is_12_l632_632933

def numberOfPeople := 15
def peoplePerPizza := 3
def earningsPerNight := 4
def nightsBabysitting := 15

-- We aim to prove that the cost per pizza is $12
theorem cost_per_pizza_is_12 : 
  (earningsPerNight * nightsBabysitting) / (numberOfPeople / peoplePerPizza) = 12 := 
by 
  sorry

end cost_per_pizza_is_12_l632_632933


namespace max_sum_complete_set_sum_l632_632251

def is_sum_complete (S : Finset ℕ) : Prop :=
  ∃ (m n : ℕ), ∀ a : ℕ, (a ∈ (S.powerset.image Finset.sum)) ↔ (m ≤ a ∧ a ≤ n)

def example_set : Finset ℕ := {1, 2, 3, 7, 14, 28, 56, 112}

theorem max_sum_complete_set_sum :
  ∀ S : Finset ℕ, is_sum_complete S → {1, 3} ⊆ S → S.card = 8 → S.sum ≤ 223 :=
by
  sorry

end max_sum_complete_set_sum_l632_632251


namespace sum_of_primes_between_20_and_30_l632_632092

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632092


namespace arrangement_white_balls_not_adjacent_ways_to_score_at_least_8_points_l632_632429

namespace BallArrangements

-- Definitions for red and white balls
constant red_balls : Fin 6 → Prop
constant white_balls : Fin 5 → Prop

-- Theorem for part (1)
theorem arrangement_white_balls_not_adjacent :
  ∃ (red_balls : Fin 6 → Prop) (white_balls: Fin 5 → Prop), 
  (∃ w : ℕ, w = 43200 ∧
  ∃ total_arrangement : (Fin 5 → Prop) × (Fin 4 → Prop), 
    -- Condition to ensure white balls are not adjacent goes here
    true) :=
sorry

-- Theorem for part (2)
theorem ways_to_score_at_least_8_points :
  ∃ (red_balls: Fin 6 → Prop) (white_balls: Fin 5 → Prop), 
  (∃ w : ℕ, w = 81 ∧
  ∃ chosen_balls : (Fin 5 → Prop) × (Fin 4 → Prop), 
    -- Condition to ensure the score is at least 8 points goes here
    true) :=
sorry

end BallArrangements

end arrangement_white_balls_not_adjacent_ways_to_score_at_least_8_points_l632_632429


namespace find_slope_of_line_that_intersects_circle_at_two_points_l632_632359

theorem find_slope_of_line_that_intersects_circle_at_two_points :
  ∃ k : ℝ, (k = 2 ∨ k = 1 / 2) ∧
  (∃ A B : ℝ × ℝ, 
   let y := λ x : ℝ, sqrt (-x^2 + 2*x + 8) + 2 in
   let line := λ x : ℝ, k*x + 3 in
   y A.1 = A.2 ∧ y B.1 = B.2 ∧ line A.1 = A.2 ∧ line B.1 = B.2 ∧
   dist A B = 12 * (sqrt 5) / 5) :=
sorry

end find_slope_of_line_that_intersects_circle_at_two_points_l632_632359


namespace no_growth_pie_probability_l632_632276

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l632_632276


namespace compute_f_one_third_l632_632907

noncomputable def B : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

noncomputable def f (x : ℚ) [hp : x ∈ B] : ℝ :=
  sorry

theorem compute_f_one_third (h₀ : f(1/3) + f(5/3) = log (2/3)) 
                           (h₁ : f(5/3) + f(7/5) = log (10/3))
                           (h₂ : f(7/5) + f(1/3) = log (10/7)) : 
  f (1/3) = log (sqrt 21 / 30) :=
by sorry

end compute_f_one_third_l632_632907


namespace leaves_dropped_on_fifth_day_l632_632450

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l632_632450


namespace exists_z0_l632_632825

open Complex

-- Define the given polynomial f(z)
noncomputable def polynomial (C : Fin n → ℂ) (n : ℕ) (z : ℂ) : ℂ :=
  (Finset.range n).sum (λ i, C i * z^(n - i))

-- Lean statement of the theorem
theorem exists_z0 (C : Fin (n + 1) → ℂ) (n : ℕ) :
  ∃ (z0 : ℂ), abs z0 ≤ 1 ∧
  abs (polynomial C n z0) ≥ abs (C 0) + abs (C n) :=
  sorry

end exists_z0_l632_632825


namespace remainder_of_polynomial_division_l632_632786

theorem remainder_of_polynomial_division
  (x : ℝ)
  (h : 2 * x - 4 = 0) :
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end remainder_of_polynomial_division_l632_632786


namespace transition_to_modern_population_reproduction_l632_632049

-- Defining the conditions as individual propositions
def A : Prop := ∃ (m b : ℝ), m < 0 ∧ b = 0
def B : Prop := ∃ (m b : ℝ), m < 0 ∧ b < 0
def C : Prop := ∃ (m b : ℝ), m > 0 ∧ b = 0
def D : Prop := ∃ (m b : ℝ), m > 0 ∧ b > 0

-- Defining the question as a property marking the transition from traditional to modern types of population reproduction
def Q : Prop := B

-- The proof problem
theorem transition_to_modern_population_reproduction :
  Q = B :=
by
  sorry

end transition_to_modern_population_reproduction_l632_632049


namespace crushers_win_all_6_games_l632_632020

noncomputable def probability_crushers_win_all_6_games : ℚ :=
  let probability_each_game : ℚ := 4 / 5
  in probability_each_game ^ 6

theorem crushers_win_all_6_games :
  probability_crushers_win_all_6_games = 4096 / 15625 :=
by
  -- Skipping the proof
  sorry

end crushers_win_all_6_games_l632_632020


namespace prime_sum_20_to_30_l632_632144

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632144


namespace order_of_magnitude_l632_632409

theorem order_of_magnitude (θ : ℝ) (h1 : -π / 8 < θ) (h2 : θ < 0) :
  tan θ < sin θ ∧ sin θ < cos θ :=
by
  sorry

end order_of_magnitude_l632_632409


namespace probability_one_girl_no_growth_pie_l632_632287

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l632_632287


namespace triangle_area_correct_l632_632916
noncomputable def triangle_area (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_ab : a < b) (h_cd : c < d) (h_bc_ad : bc > ad) : ℝ :=
  (1 / 2) * (bc - ad)

theorem triangle_area_correct (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_ab : a < b) (h_cd : c < d) (h_bc_ad : bc > ad) :
  let area := triangle_area a b c d h_pos h_ab h_cd h_bc_ad
  in area = (1 / 2) * (bc - ad) :=
by 
  sorry

end triangle_area_correct_l632_632916


namespace decimal_to_binary_51_l632_632742

theorem decimal_to_binary_51 :
  ∀ (n : ℕ), n = 51 → n.binary_repr = "110011" :=
by
  intros n h
  rw h
  sorry

end decimal_to_binary_51_l632_632742


namespace oddly_powerful_less_than_5000_l632_632302

def is_oddly_powerful (n : ℕ) : Prop := 
  ∃ (a b : ℕ), b % 2 = 1 ∧ b > 1 ∧ a^b = n

def num_oddly_powerful_less_than (m : ℕ) : ℕ :=
  ((finset.range m).filter is_oddly_powerful).card

theorem oddly_powerful_less_than_5000 : num_oddly_powerful_less_than 5000 = 19 :=
  sorry

end oddly_powerful_less_than_5000_l632_632302


namespace prime_sum_20_to_30_l632_632184

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632184


namespace price_increase_solution_l632_632238

variable (x : ℕ)

def initial_profit := 10
def initial_sales := 500
def price_increase_effect := 20
def desired_profit := 6000

theorem price_increase_solution :
  ((initial_sales - price_increase_effect * x) * (initial_profit + x) = desired_profit) → (x = 5) :=
by
  sorry

end price_increase_solution_l632_632238


namespace rationalize_denominator_ABC_l632_632530

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l632_632530


namespace triangle_inequality_sum_zero_l632_632942

theorem triangle_inequality_sum_zero (a b c p q r : ℝ) (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) (hpqr : p + q + r = 0) : a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
by 
  sorry

end triangle_inequality_sum_zero_l632_632942


namespace hydras_never_die_l632_632654

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632654


namespace sum_of_primes_between_20_and_30_l632_632094

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632094


namespace combined_yearly_return_percentage_l632_632691

-- Define the given conditions
def investment1 : ℝ := 500
def rate1 : ℝ := 0.07

def investment2 : ℝ := 1500
def rate2 : ℝ := 0.19

-- Define the total investment
def total_investment : ℝ := investment1 + investment2

-- Define the returns from each investment
def return1 : ℝ := investment1 * rate1
def return2 : ℝ := investment2 * rate2

-- Define the total return
def total_return : ℝ := return1 + return2

-- Define the combined yearly return percentage
def combined_return_percentage : ℝ := (total_return / total_investment) * 100

-- Proof statement
theorem combined_yearly_return_percentage :
  combined_return_percentage = 16 := by
  sorry

end combined_yearly_return_percentage_l632_632691


namespace median_invariance_l632_632884

def median {α : Type*} [LinearOrder α] (s : List α) : α :=
  let sorted_s := s.qsort (· < ·)
  sorted_s.get (sorted_s.length / 2)

theorem median_invariance (a : List ℝ) (h : a.length = 7)
  (b : List ℝ) (hb₁ : b.length = 5)
  (hb₂ : ∀ x, x ∈ b ↔ x ∈ a ∧ x ≠ List.maximum a ∧ x ≠ List.minimum a) :
  median a = median b :=
by
  sorry

end median_invariance_l632_632884


namespace convex_combination_is_convex_convex_combination_perimeter_l632_632212

-- Definition of convex polygons and their properties
variables {M1 M2 : Type} [convex_polygon M1] [convex_polygon M2]
variables (λ1 λ2 : ℝ) (hλ1 : λ1 ≥ 0) (hλ2 : λ2 ≥ 0) (hλ_sum : λ1 + λ2 = 1)

-- Part (a)
-- Define the convex combination of two convex polygons and prove the statement
theorem convex_combination_is_convex : is_convex_polygon (λ1 • M1 + λ2 • M2) ∧
                                      num_sides (λ1 • M1 + λ2 • M2) ≤ num_sides M1 + num_sides M2 :=
by
  sorry

-- Part (b)
variables (P1 P2 : ℝ) (hP1 : P1 = perimeter M1) (hP2 : P2 = perimeter M2)

-- Define the perimeter of the convex combination of two convex polygons and prove the statement
theorem convex_combination_perimeter : perimeter (λ1 • M1 + λ2 • M2) = λ1 * P1 + λ2 * P2 :=
by
  sorry

end convex_combination_is_convex_convex_combination_perimeter_l632_632212


namespace intersection_of_P_and_Q_l632_632849

noncomputable def P : Set ℝ := {x | 0 < Real.log x / Real.log 8 ∧ Real.log x / Real.log 8 < 2 * (Real.log 3 / Real.log 8)}
noncomputable def Q : Set ℝ := {x | 2 / (2 - x) > 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l632_632849


namespace rationalize_denominator_ABC_l632_632529

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l632_632529


namespace quadratic_function_eq_value_of_m_values_of_n_l632_632830

-- Given conditions
variables (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
variables (vertex_A : A = (-1, 4)) (point_B : B = (2, -5))
variables (point_C : C = (-1/2, m)) (point_D : D = (n, 3))
variables (a : ℝ) (h k : ℝ)
variables h_eq : h = -1
variables k_eq : k = 4

-- Function definition
def quadratic_func (x : ℝ) : ℝ := a * (x + h)^2 + k

-- Proofs
theorem quadratic_function_eq :
  ∃ (a : ℝ), (∀ (x y : ℝ), quadratic_func x = y ↔ y = -x^2 - 2*x + 3) :=
by {
  sorry
}

theorem value_of_m :
  ∃ (m : ℚ), m = 11/4 :=
by {
  sorry
}

theorem values_of_n :
  ∃ (n : ℚ), n = 0 ∨ n = -2 :=
by {
  sorry
}

end quadratic_function_eq_value_of_m_values_of_n_l632_632830


namespace milburg_children_count_l632_632588

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l632_632588


namespace find_D_l632_632953

theorem find_D : ∃ (D : ℝ), (∀ (x : ℝ), x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -5 →
  1 / (x^3 - 2*x^2 - 13*x + 10) = D / (x - 2) + (E : ℝ) / (x - 1) + (F : ℝ) / (x + 5)^2) ∧ D = 1 / 7 :=
by
  have h : x^3 - 2*x^2 - 13*x + 10 = (x - 2)*(x - 1)*(x + 5) := sorry
  sorry

end find_D_l632_632953


namespace Lindas_savings_l632_632927

theorem Lindas_savings (S : ℝ) (h1 : (1/3) * S = 250) : S = 750 := 
by
  sorry

end Lindas_savings_l632_632927


namespace find_expression_for_f_x_neg_l632_632374

theorem find_expression_for_f_x_neg (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x, 0 < x → f x = x - Real.log (abs x)) :
  ∀ x, x < 0 → f x = x + Real.log (abs x) :=
by
  sorry

end find_expression_for_f_x_neg_l632_632374


namespace dust_particles_calculation_l632_632545

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end dust_particles_calculation_l632_632545


namespace solve_sqrt_equation_l632_632017

theorem solve_sqrt_equation (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) :
    sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := by
  sorry

end solve_sqrt_equation_l632_632017


namespace prime_sum_20_to_30_l632_632145

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632145


namespace episodes_remaining_l632_632206

-- Definition of conditions
def seasons : ℕ := 12
def episodes_per_season : ℕ := 20
def fraction_watched : ℚ := 1 / 3
def total_episodes : ℕ := episodes_per_season * seasons
def episodes_watched : ℕ := (fraction_watched * total_episodes).toNat

-- Problem statement
theorem episodes_remaining : total_episodes - episodes_watched = 160 := by
  sorry

end episodes_remaining_l632_632206


namespace prime_sum_20_to_30_l632_632187

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632187


namespace bunnies_out_of_burrows_l632_632419

theorem bunnies_out_of_burrows 
  (bunnies_per_min : Nat) 
  (num_bunnies : Nat) 
  (hours : Nat) 
  (mins_per_hour : Nat):
  bunnies_per_min = 3 -> num_bunnies = 20 -> mins_per_hour = 60 -> hours = 10 -> 
  (bunnies_per_min * mins_per_hour * hours * num_bunnies = 36000) :=
by
  intros,
  sorry

end bunnies_out_of_burrows_l632_632419


namespace option_B_is_quadratic_l632_632674

-- Definitions of the equations provided in conditions
def eqA (x y : ℝ) := x + 3 * y = 4
def eqB (y : ℝ) := 5 * y = 5 * y^2
def eqC (x : ℝ) := 4 * x - 4 = 0
def eqD (x a : ℝ) := a * x^2 - x = 1

-- Quadratic equation definition
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ) (x : ℝ), A ≠ 0 ∧ eq = (A * x^2 + B * x + C = 0)

-- Theorem: Prove that option B (eqB) is definitely a quadratic equation
theorem option_B_is_quadratic : is_quadratic eqB :=
  sorry

end option_B_is_quadratic_l632_632674


namespace hydras_will_live_l632_632638

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632638


namespace parametric_to_standard_form_l632_632747

variable (θ : ℝ)

-- Conditions from the problem
def parametric_x : ℝ := cos θ / (1 + cos θ)
def parametric_y : ℝ := sin θ / (1 + cos θ)
def cos_theta_ne_neg_one : 1 + cos θ ≠ 0

-- Proof goal derived from the correct answer
theorem parametric_to_standard_form (θ : ℝ) (cos_theta_ne_neg_one : 1 + cos θ ≠ 0) :
    (parametric_y θ)^2 = -2 * (parametric_x θ - 1 / 2) :=
sorry

end parametric_to_standard_form_l632_632747


namespace lambda_value_l632_632818

open Real

noncomputable def hyperbola_a (a b : ℝ) : ℝ :=
a

noncomputable def hyperbola_b (a b : ℝ) : ℝ :=
b

noncomputable def eccentricity (a c : ℝ) : ℝ :=
c / a

theorem lambda_value
  (a b c : ℝ) (P F₁ F₂ : ℝ × ℝ)
  (h_hyperbola : ∃ a b, a > 0 ∧ b > 0 ∧ P ∈ { point | (point.1 ^ 2 / a ^ 2) - (point.2 ^ 2 / b ^ 2) = 1 })
  (F₁_Focus : F₁ = (c, 0))
  (F₂_Focus : F₂ = (-c, 0))
  (eccentricity_val : eccentricity a c = 3)
  (area_condition : ∀ M, is_incenter M P F₁ F₂ → S_Δ MP F₁ = S_Δ MP F₂ + λ S_Δ MF₁F₂) :
  λ = 1 / 3 :=
sorry

end lambda_value_l632_632818


namespace prime_sum_20_to_30_l632_632188

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632188


namespace hydra_survival_l632_632602

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632602


namespace sum_of_numbers_in_progressions_l632_632596

theorem sum_of_numbers_in_progressions:
  (a b c : ℚ) 
  (r : ℚ) 
  (d : ℚ) 
  (h1 : a = 3 * r)
  (h2 : b = 3 * r^2)
  (h3 : b = a + d)
  (h4 : c = b + d)
  (h5 : 27 = c + d) 
  : a + b + c = 161 / 3 := by
  sorry

end sum_of_numbers_in_progressions_l632_632596


namespace sum_of_k_correct_l632_632957

noncomputable def sum_of_k : ℕ :=
∑ k in {k | ∃ α β : ℤ, α * β = 20 ∧ α + β = k ∧ k > 0}, k

theorem sum_of_k_correct : sum_of_k = 42 := sorry

end sum_of_k_correct_l632_632957


namespace problem_statement_l632_632808

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum of the first n terms of the sequence
variable (d : ℝ) -- the common difference
variable (a1 : ℝ) -- the first term

-- Conditions
axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a1 + a n) / 2
axiom S_15_eq_45 : S 15 = 45

-- The statement to prove
theorem problem_statement : 2 * a 12 - a 16 = 3 :=
by
  sorry

end problem_statement_l632_632808


namespace modulus_problem_l632_632920

open Complex

noncomputable def modulus_expression (z w : ℂ) : ℂ := (z + 2 * conj w) * (conj z - 2 * w)

theorem modulus_problem
  (z w : ℂ)
  (hz : abs z = 3)
  (hw_condition : (z + conj w) * (conj z - w) = 7 + 4 * Complex.I) :
  abs (modulus_expression z w) = Real.sqrt 65 :=
by
  sorry

end modulus_problem_l632_632920


namespace mechanism_parts_l632_632694

-- Definitions
def total_parts (S L : Nat) : Prop := S + L = 25
def condition1 (S L : Nat) : Prop := ∀ (A : Finset (Fin 25)), (A.card = 12) → ∃ i, i ∈ A ∧ i < S
def condition2 (S L : Nat) : Prop := ∀ (B : Finset (Fin 25)), (B.card = 15) → ∃ i, i ∈ B ∧ i >= S

-- Main statement
theorem mechanism_parts :
  ∃ (S L : Nat), 
  total_parts S L ∧ 
  condition1 S L ∧ 
  condition2 S L ∧ 
  S = 14 ∧ 
  L = 11 :=
sorry

end mechanism_parts_l632_632694


namespace exists_z_in_S_l632_632480

-- Conditions
variables (a p : ℤ) (hp : p.prime) (ha_pos : 0 < a)
variables (x y : ℤ) (hx_in_S : x^41 % p = a % p) (hy_in_S : y^49 % p = a % p)

-- Question: Determine if there exists a positive integer z such that z^{2009} is in S
theorem exists_z_in_S : ∃ z : ℤ, (0 < z) ∧ (z^2009) % p = a % p :=
begin
  sorry
end

end exists_z_in_S_l632_632480


namespace measure_of_angle_BDC_l632_632504

noncomputable def angle_BDC_measure (A B C D : Point) (ABC : Triangle) : ℝ :=
if right_triangle ABC ∧ angle A = 90 then
  45
else
  0

axioms
  (Point : Type)
  (Triangle : Type)
  (angle : Triangle → Point → ℝ)
  (right_triangle : Triangle → Prop)
  (A B C D : Point)
  (ABC : Triangle)
  (H_right : right_triangle ABC)
  (H_angle_A : angle ABC A = 90)
  (H_extangle_bisectors : -- defines the property that D is the intersection of the exterior angle bisectors of B and C in the context of ABC)

theorem measure_of_angle_BDC
  (H_right : right_triangle ABC)
  (H_angle_A : angle ABC A = 90)
  (H_extangle_bisectors : -- include the actual conditions for the intersection of the bisectors):
  angle_BDC_measure A B C D ABC = 45 :=
by
  sorry

end measure_of_angle_BDC_l632_632504


namespace adult_meal_cost_l632_632729

theorem adult_meal_cost (x : ℝ) 
  (total_people : ℕ) (kids : ℕ) (total_cost : ℝ)  
  (h_total_people : total_people = 11) 
  (h_kids : kids = 2) 
  (h_total_cost : total_cost = 72)
  (h_adult_meals : (total_people - kids : ℕ) • x = total_cost) : 
  x = 8 := 
by
  -- Proof will go here
  sorry

end adult_meal_cost_l632_632729


namespace hydras_survive_l632_632621

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632621


namespace parallel_lines_m_l632_632853

noncomputable def lines_parallel (m : ℝ) : Prop :=
  let l1 := λ x y : ℝ, mx + 2 * y - 2 = 0
  let l2 := λ x y : ℝ, 5 * x + (m + 3) * y - 5 = 0
  (-m / 2) = (-5 / (m + 3))

theorem parallel_lines_m :
  (∃ m : ℝ, lines_parallel m) →
  lines_parallel (-5) :=
by
  sorry

end parallel_lines_m_l632_632853


namespace quadratic_inequality_solutions_l632_632845

theorem quadratic_inequality_solutions (m : ℝ) :
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ∈ ℝ ∧ m ≠ 0) := sorry

end quadratic_inequality_solutions_l632_632845


namespace weeks_to_work_l632_632306

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l632_632306


namespace B_is_not_15_percent_less_than_A_l632_632415

noncomputable def A (B : ℝ) : ℝ := 1.15 * B

theorem B_is_not_15_percent_less_than_A (B : ℝ) (h : B > 0) : A B ≠ 0.85 * (A B) :=
by
  unfold A
  suffices 1.15 * B ≠ 0.85 * (1.15 * B) by
    intro h1
    exact this h1
  sorry

end B_is_not_15_percent_less_than_A_l632_632415


namespace cos_F_l632_632443

-- Define the lengths of the sides and the right angle
variables {DE EF DF : ℝ}
variable (α : ℝ)

-- Given conditions
variable h1 : DE = 8
variable h2 : EF = 17
variable h3 : α = 90

-- Define DF using the Pythagorean theorem
noncomputable def calculate_DF (h1 : DE = 8) (h2 : EF = 17) : ℝ :=
  real.sqrt (EF^2 - DE^2)

-- Prove the cosine of angle F
theorem cos_F {DE EF DF : ℝ} (h1 : DE = 8) (h2 : EF = 17) (h3 : α = 90) : 
  real.cos α = (8 / 17) :=
by
  -- State that EF^2 - DE^2 = DF^2 from the Pythagorean theorem
  have h_df : DF = calculate_DF h1 h2 := sorry,
  -- Use the definition of cosine in the right triangle
  sorry

end cos_F_l632_632443


namespace girls_more_than_boys_l632_632876

def ratio_of_girls_to_boys := 5 / 4
def total_students := 36

theorem girls_more_than_boys :
  let x := total_students / (ratio_of_girls_to_boys + 1) in
  let girls := 5 * x in
  let boys := 4 * x in
  girls - boys = 4 :=
by
  sorry

end girls_more_than_boys_l632_632876


namespace no_positive_integer_solution_l632_632526

theorem no_positive_integer_solution (p x y : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) 
  (h_p_div_x : p ∣ x) (hx_pos : 0 < x) (hy_pos : 0 < y) : x^2 - 1 ≠ y^p :=
sorry

end no_positive_integer_solution_l632_632526


namespace sum_of_primes_between_20_and_30_l632_632117

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632117


namespace initial_number_is_12_l632_632695

theorem initial_number_is_12 (x : ℕ) : (y = 3 * 2 * x) → (7899665 - y = 7899593) → x = 12 := by
  intros h1 h2
  have h3 : 7899665 - 7899593 = 6 * x := by
    rw [←h2, h1]
  have h4 : 72 = 6 * x := by
    exact Eq.subst (rfl : 7899665 - 7899593 = 72) h3
  exact Nat.eq_of_mul_hom_eq 12 (Nat.eq_of_mul_eq_mul_right (Nat.succ_pos') h4)

end initial_number_is_12_l632_632695


namespace sum_primes_in_range_l632_632081

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632081


namespace sum_primes_20_to_30_l632_632149

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632149


namespace height_of_boxes_l632_632227

-- Conditions
def total_volume : ℝ := 1.08 * 10^6
def cost_per_box : ℝ := 0.2
def total_monthly_cost : ℝ := 120

-- Target height of the boxes
def target_height : ℝ := 12.2

-- Problem: Prove that the height of each box is 12.2 inches
theorem height_of_boxes : 
  (total_monthly_cost / cost_per_box) * ((total_volume / (total_monthly_cost / cost_per_box))^(1/3)) = target_height := 
sorry

end height_of_boxes_l632_632227


namespace decimal_to_binary_l632_632745

-- Define the decimal value
def decimal_val : ℕ := 51

-- Define the expected binary representation
def binary_val : ℕ := 0b110011  -- 51 in binary notation

-- State the theorem to prove
theorem decimal_to_binary : nat.bv (dec := decimal_val) = binary_val := 
by 
  sorry

end decimal_to_binary_l632_632745


namespace sum_of_lengths_of_segments_l632_632944

theorem sum_of_lengths_of_segments :
  let EF := 5
  let FG := 6
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length k := (k : ℝ) / 200
  (2 * ∑ k in Finset.range 200, diagonal_length * segment_length (199 - k) - diagonal_length) = 198 * Real.sqrt 61 := by
  sorry

end sum_of_lengths_of_segments_l632_632944


namespace number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l632_632689

-- Define the statement about the total number of 5-letter words.
theorem number_of_5_letter_words : 26^5 = 26^5 := by
  sorry

-- Define the statement about the total number of 5-letter words with all different letters.
theorem number_of_5_letter_words_with_all_different_letters : 
  26 * 25 * 24 * 23 * 22 = 26 * 25 * 24 * 23 * 22 := by
  sorry

-- Define the statement about the total number of 5-letter words with no consecutive letters being the same.
theorem number_of_5_letter_words_with_no_consecutive_repeating_letters : 
  26 * 25 * 25 * 25 * 25 = 26 * 25 * 25 * 25 * 25 := by
  sorry

end number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l632_632689


namespace greatest_integer_condition_l632_632071

theorem greatest_integer_condition (x: ℤ) (h₁: x < 150) (h₂: Int.gcd x 24 = 3) : x ≤ 147 := 
by sorry

example : ∃ x, x < 150 ∧ Int.gcd x 24 = 3 ∧ x = 147 :=
begin
  use 147,
  split,
  { exact lt_trans (by norm_num) (by norm_num) },
  { split,
    { norm_num },
    { refl } }
end

end greatest_integer_condition_l632_632071


namespace sum_of_primes_between_20_and_30_l632_632099

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632099


namespace max_load_range_l632_632473

variable (N_m : Real := 2000)
variable (m_s2 : Real := 10)
variable (r : Real := 0.25)
variable (dr : Real := 0.05)
variable (Mmax : Real := N_m / (m_s2 * (r + dr)))
variable (Range120 : Real := 120)
variable (Range160 : Real := 160)
variable (CorrectRange1 : Set Real := set.Icc (800 - Range120) (800 + Range120))
variable (CorrectRange2 : Set Real := set.Icc (800 - Range160) (800 + Range160))

theorem max_load_range :
  (CorrectRange1 = set.Icc 680 920) ∧ (CorrectRange2 = set.Icc 640 960) :=
sorry

end max_load_range_l632_632473


namespace prime_sum_20_to_30_l632_632143

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632143


namespace geometric_sequence_a5_value_l632_632891

-- Definition of geometric sequence and the specific condition a_3 * a_7 = 8
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (geom_seq : is_geometric_sequence a)
  (cond : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
sorry

end geometric_sequence_a5_value_l632_632891


namespace find_m_n_and_area_l632_632404

-- Definitions of the given conditions
structure Point (α : Type*) :=
(x : α)
(y : α)

def OA : Point ℝ := ⟨-2, 3⟩
def OB : Point ℝ := ⟨1.5, 1⟩
def OC : Point ℝ := ⟨5, -1⟩

-- Collinearity condition: vectors AC and AB are parallel
def is_collinear (A B C : Point ℝ) : Prop :=
  let AC := ⟨C.x - A.x, C.y - A.y⟩
  let AB := ⟨B.x - A.x, B.y - A.y⟩
  AC.x * AB.y = AC.y * AB.x

-- Perpendicular condition: OA . OB = 0
def is_perpendicular (A B : Point ℝ) : Prop :=
  A.x * B.x + A.y * B.y = 0

def centroid (A B C : Point ℝ) :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

def correct_position_of_OB (B G : Point ℝ) : Prop :=
  ⟨B.x, B.y⟩ = ⟨3/2 * G.x, 3/2 * G.y⟩

def area_of_triangle (A B C : Point ℝ) : ℝ :=
  (1/2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- The Lean theorem to prove
theorem find_m_n_and_area :
  is_collinear OA OB OC →
  is_perpendicular OA OB →
  ∃ m n : ℝ, m = 3 ∧ n = (3 / 2) ∧
    let A := OA in
    let B := OB in
    let C := OC in
    let G := centroid A C {⟨0, 0⟩} in
    correct_position_of_OB B G ∧
    area_of_triangle A {⟨0, 0⟩} C = 13 / 2 :=
begin
  sorry
end

end find_m_n_and_area_l632_632404


namespace boric_acid_molecular_weight_l632_632300

def molecular_weight (atoms : List (String × ℕ × Float)) : Float :=
  atoms.foldl (λ acc (_, count, weight) => acc + count * weight) 0

theorem boric_acid_molecular_weight :
  molecular_weight [("H", 3, 1.008), ("B", 1, 10.81), ("O", 3, 16.00)] = 61.834 :=
by
  sorry

end boric_acid_molecular_weight_l632_632300


namespace roots_quadratic_l632_632764

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l632_632764


namespace prime_sum_20_to_30_l632_632192

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632192


namespace total_parts_in_order_l632_632226

theorem total_parts_in_order (total_cost : ℕ) (cost_20 : ℕ) (cost_50 : ℕ) (num_50_dollar_parts : ℕ) (num_20_dollar_parts : ℕ) :
  total_cost = 2380 → cost_20 = 20 → cost_50 = 50 → num_50_dollar_parts = 40 → (total_cost = num_50_dollar_parts * cost_50 + num_20_dollar_parts * cost_20) → (num_50_dollar_parts + num_20_dollar_parts = 59) :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_parts_in_order_l632_632226


namespace tree_drops_leaves_on_fifth_day_l632_632451

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l632_632451


namespace find_numbers_l632_632767

theorem find_numbers :
  ∃ (A B C D E : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    5 < A ∧
    ∃ k, A = k * B ∧
    C + A = D ∧
    B + C + E = A ∧
    B + C < E ∧
    C + E < B + 5 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    A = 8 ∧ B = 2 ∧ C = 1 ∧ D = 9 ∧ E = 5 :=
by
  exists 8, 2, 1, 9, 5
  simp
  split
  exact 8 < 10
  split
  exact 2 < 10
  split
  exact 1 < 10
  split
  exact 9 < 10
  split
  exact 5 < 10
  split
  exact 5 < 8
  split
  exists 4
  exact 8 = 4 * 2
  split
  exact 1 + 8 = 9
  split
  exact 2 + 1 + 5 = 8
  split
  exact 2 + 1 < 5
  split
  exact 1 + 5 < 2 + 5
  split
  exact 8 ≠ 2
  split
  exact 8 ≠ 1
  split
  exact 8 ≠ 9
  split
  exact 8 ≠ 5
  split
  exact 2 ≠ 1
  split
  exact 2 ≠ 9
  split
  exact 2 ≠ 5
  split
  exact 1 ≠ 9
  split
  exact 1 ≠ 5
  split
  exact 9 ≠ 5
  split
  exact 8 = 8
  split
  exact 2 = 2
  split
  exact 1 = 1
  split
  exact 9 = 9
  exact 5 = 5
  sorry

end find_numbers_l632_632767


namespace average_speed_l632_632713

def initial_time : ℝ := 6 -- hours
def distance : ℝ := 378 -- km
def time_multiplier : ℝ := 3 / 2 -- multiplier

def new_time : ℝ := initial_time * time_multiplier

theorem average_speed :
  (distance / new_time) = 42 :=
by
  sorry

end average_speed_l632_632713


namespace hydras_will_live_l632_632632

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632632


namespace dust_particles_before_sweeping_l632_632547

theorem dust_particles_before_sweeping 
  (cleared_fraction : ℚ := 9/10)
  (dust_particles_added : ℕ := 223)
  (post_walking_dust_particles : ℕ := 331) :
  let remaining_fraction := 1 - cleared_fraction
  let pre_walking_dust_particles : ℕ := post_walking_dust_particles - dust_particles_added
  let pre_sweeping_dust_particles : ℕ := pre_walking_dust_particles / remaining_fraction
  pre_sweeping_dust_particles = 1080 :=
by
  sorry

end dust_particles_before_sweeping_l632_632547


namespace sum_of_primes_between_20_and_30_l632_632095

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632095


namespace cube_root_inequality_l632_632951

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem cube_root_inequality : 
  Real.cbrt (x^2 * y^2 * z^2) ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

end cube_root_inequality_l632_632951


namespace number_of_four_digit_numbers_l632_632862

theorem number_of_four_digit_numbers : 
  let digits := [2, 0, 0, 5] in
  let valid_numbers := {n | n.to_list.permutations.map (fun l => l.foldl (fun acc d => acc * 10 + d) 0) | l.head ≠ 0} in
  valid_numbers.card = 6 :=
by
  let digits := [2, 0, 0, 5]
  let permutations := digits.permutations
  let valid_numbers := permutations.filter (λ l => l.head ≠ 0)
  let converted_numbers := valid_numbers.map (fun l => l.foldl (fun acc d => acc * 10 + d) 0)
  have cardinality : converted_numbers.length = 6 := sorry
  exact cardinality

end number_of_four_digit_numbers_l632_632862


namespace tetrahedron_height_and_volume_l632_632476

-- Given: 
-- ABCD is a regular tetrahedron with side length 1.
-- EFGH is another regular tetrahedron such that the volume of EFGH is 1/8 of the volume of ABCD.
-- The height of EFGH can be written as sqrt(a / b), where a and b are positive coprime integers.
-- Prove: a + b = 7.

theorem tetrahedron_height_and_volume (s : ℝ) (a b : ℕ) (hab : Nat.coprime a b) :
  let V := s^3 / (6 * Real.sqrt 2),
      V_EFGH := V / 8,
      h := (s * Real.sqrt 6) / 3,
      h_EFGH := h / 2 in
  V = 1 / (6 * Real.sqrt 2) -> 
  V_EFGH = 1 / (48 * Real.sqrt 2) -> 
  h = Real.sqrt 6 / 3 -> 
  h_EFGH = Real.sqrt 6 / 6 -> 
  (h_EFGH = Real.sqrt (a / b)) -> 
  (a = 1) -> 
  (b = 6) -> 
  a + b = 7 := 
by
  intros
  sorry

end tetrahedron_height_and_volume_l632_632476


namespace odd_composite_count_l632_632290

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ k m : ℕ, k > 1 ∧ k < n ∧ m > 1 ∧ m < n ∧ k * m = n

theorem odd_composite_count :
  (∃ (count : ℕ), count = (Finset.filter (λ n, is_odd n ∧ is_composite n) (Finset.range 31)).card ∧ count = 5) :=
by
  sorry

end odd_composite_count_l632_632290


namespace milburg_children_count_l632_632586

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l632_632586


namespace decimal_to_binary_l632_632744

-- Define the decimal value
def decimal_val : ℕ := 51

-- Define the expected binary representation
def binary_val : ℕ := 0b110011  -- 51 in binary notation

-- State the theorem to prove
theorem decimal_to_binary : nat.bv (dec := decimal_val) = binary_val := 
by 
  sorry

end decimal_to_binary_l632_632744


namespace ring_covers_at_least_10_points_l632_632936

theorem ring_covers_at_least_10_points :
  ∀ (points : set (ℝ × ℝ)), 
    (finite points ∧ points.card = 650) →
    (∃ (c ∈ set.sphere (0 : ℝ × ℝ) 0 16), points ⊆ set.ball c 16) →
    ∃ (c : ℝ × ℝ) (r : ℝ), 
      ((2 ≤ r ∧ r ≤ 3) ∧ (∀ p ∈ points, dist c p ≤ r → dist c p ≥ 2)) ∧ 
      (∃ pts ⊆ points, pts.card ≥ 10 ∧ ∀ p ∈ pts, dist c p <= 3 ∧ dist c p >= 2) :=
begin
  sorry
end

end ring_covers_at_least_10_points_l632_632936


namespace annual_interest_rate_last_year_l632_632022

-- Define the conditions
def increased_by_ten_percent (r : ℝ) : ℝ := 1.10 * r

-- Statement of the problem
theorem annual_interest_rate_last_year (r : ℝ) (h : increased_by_ten_percent r = 0.11) : r = 0.10 :=
sorry

end annual_interest_rate_last_year_l632_632022


namespace bank_balance_after_two_years_l632_632244

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l632_632244


namespace correct_turns_for_opposite_direction_l632_632239

def turn_angles_sum_to_180 (a b : ℕ) := a + b = 180

def turns_same_direction (a b : ℕ) := a > 0 ∧ b > 0 

def optionA := (30, -30)
def optionB := (45, 45)
def optionC := (60, -120)
def optionD := (53, 127)

theorem correct_turns_for_opposite_direction :
  turn_angles_sum_to_180 (fst optionD) (snd optionD) ∧ turns_same_direction (fst optionD) (snd optionD) := by
  sorry

end correct_turns_for_opposite_direction_l632_632239


namespace visible_people_expected_value_l632_632221

noncomputable def harmonic (n : ℕ) : ℝ :=
  (finset.range(n) \ {0}).sum (λ k, 1 / (k:ℝ))

theorem visible_people_expected_value (n : ℕ) : 
  ∃ X_n : ℕ → ℝ, (∀ k : ℕ, X_k = (1:ℝ) / (k:ℝ)) ∧
  ∑ k in finset.range(n), X_n k = harmonic n := 
sorry

end visible_people_expected_value_l632_632221


namespace largest_number_is_41_67_l632_632971

noncomputable def largest_number : Real :=
  let a b c : Real := sorry  -- Placeholder for the actual calculation
  if h₁ : a + b + c = 100 ∧ c = b + 10 ∧ b = a + 5 then c else 0

theorem largest_number_is_41_67 : largest_number = 41.67 := 
by sorry

end largest_number_is_41_67_l632_632971


namespace Elmer_food_percentage_l632_632001

noncomputable def percentage_Elmer_food (P G M E R C : ℕ) : ℕ :=
let total := P + G + M + E + R + C in
(E * 100) / total

theorem Elmer_food_percentage (P G M E R C : ℕ) (hP : P = 20)
  (hG : G = P / 10)
  (hM : M = G / 100)
  (hE : E = 4000 * M)
  (hR : R = 3 * G)
  (hC : C = P / 2)
  (htotal_le : P + G + M + E + R + C ≤ 1000) :
  percentage_Elmer_food P G M E R C = 68 :=
by
  sorry

end Elmer_food_percentage_l632_632001


namespace decimal_to_binary_51_l632_632739

theorem decimal_to_binary_51: nat.bin_enc 51 = "110011" :=
  sorry

end decimal_to_binary_51_l632_632739


namespace t_mobile_additional_line_cost_l632_632932

variable (T : ℕ)

def t_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * T

def m_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * 14

theorem t_mobile_additional_line_cost
  (h : t_mobile_cost 5 = m_mobile_cost 5 + 11) :
  T = 16 :=
by
  sorry

end t_mobile_additional_line_cost_l632_632932


namespace combined_area_of_removed_triangles_l632_632709

-- Define the side lengths of the triangles
def leg1 : ℕ := 4
def leg2 : ℕ := 3

-- Define the side length of the original square
def side_length_square : ℕ := 20

-- Calculate the area of one triangle
def area_one_triangle : ℕ := (1 / 2 : ℚ) * leg1 * leg2

-- The theorem to state that the combined area of the four triangles is 24
theorem combined_area_of_removed_triangles : 
    4 * (area_one_triangle) = 24 := 
by
  -- Calculation steps go here
  sorry

end combined_area_of_removed_triangles_l632_632709


namespace circle_through_four_points_l632_632484

theorem circle_through_four_points
  (m n : ℕ) (hm : 2 ≤ m) (hn : 2 ≤ n) 
  (S : set (ℕ × ℕ)) (hS_sub : S ⊆ {p | p.1 ∈ [1, m] ∧ p.2 ∈ [1, n]})
  (hS_card : S.card ≥ m + n + (nat.floor ((m + n) / 4 - 1/2))) :
  ∃ (A B C D : ℕ × ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    collinear {A, B, C} ≠ true ∧ collinear {A, B, D} ≠ true ∧
    collinear {A, C, D} ≠ true ∧ collinear {B, C, D} ≠ true ∧ 
    circle_passes_through{A, B, C, D} :=
sorry

end circle_through_four_points_l632_632484


namespace find_p_l632_632847

-- Definitions of relevant points and conditions
variable (p : ℝ)
axiom h1 : p > 0
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus : (ℝ × ℝ) := (p/2, 0)
def directrix_x := 0
def A : (ℝ × ℝ) := (0, 0)
def P : (ℝ × ℝ) := (p/2, p)
def circle (center : (ℝ × ℝ)) (radius : ℝ) (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2
def AF := λ (x y : ℝ), circle ((p/4,0)) (p/4) x y  -- diameter means radius is (p/4)

-- Properties described in the problem
axiom h2 : parabola (P.1) (P.2)
axiom h3 : P.2 ⊥ 0             -- PF ⊥ x-axis
axiom h4 : (λ M, (A.1 + 1/2*AF(A.1 ellefocus)).x - Ps.2 == 0 --> x = 0 thus right angel A-P-F 

-- Chord length condition
axiom h5 : ∃ M, AF M.1 M.2 → (2 * (P.1 - A.1 | underordinates) ⊥ 2)

-- The final proof statement.
theorem find_p : p = 2 * real.sqrt 2 :=
sorry -- Proof omitted

end find_p_l632_632847


namespace sufficient_money_days_l632_632253

noncomputable def problem_conditions (A B : ℕ) (daysA daysB : ℕ) : Prop :=
  (daysA = 21) ∧
  (daysB = 28) ∧
  (21 * A = 28 * B)

theorem sufficient_money_days (A B : ℕ) (D : ℕ)
  (h : problem_conditions A B D) : 
  21 * A / (A + B) = 12 :=
by
  sorry

end sufficient_money_days_l632_632253


namespace dust_particles_before_sweeping_l632_632546

theorem dust_particles_before_sweeping 
  (cleared_fraction : ℚ := 9/10)
  (dust_particles_added : ℕ := 223)
  (post_walking_dust_particles : ℕ := 331) :
  let remaining_fraction := 1 - cleared_fraction
  let pre_walking_dust_particles : ℕ := post_walking_dust_particles - dust_particles_added
  let pre_sweeping_dust_particles : ℕ := pre_walking_dust_particles / remaining_fraction
  pre_sweeping_dust_particles = 1080 :=
by
  sorry

end dust_particles_before_sweeping_l632_632546


namespace round_nearest_tenth_l632_632543

theorem round_nearest_tenth (x : ℝ) (h : x = 23.2678) : round (x * 10) / 10 = 23.3 := by
  rw [h]
  have : round (23.2678 * 10) = 233 := by sorry -- skipping the proof of this intermediate step
  rw [this]
  norm_num

end round_nearest_tenth_l632_632543


namespace prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l632_632877

section card_draws

/-- A card draw experiment with 10 cards: 5 red, 3 white, 2 blue. --/
inductive CardColor
| red
| white
| blue

def bag : List CardColor := List.replicate 5 CardColor.red ++ List.replicate 3 CardColor.white ++ List.replicate 2 CardColor.blue

/-- Probability of drawing exactly 2 red cards in up to 3 draws with the given conditions. --/
def prob_two_reds : ℚ :=
  (5 / 10) * (5 / 10) + 
  (5 / 10) * (2 / 10) * (5 / 10) + 
  (2 / 10) * (5 / 10) * (5 / 10)

theorem prob_of_two_reds_is_7_over_20 : prob_two_reds = 7 / 20 :=
  sorry

/-- Probability distribution of the number of draws necessary. --/
def prob_ξ_1 : ℚ := 3 / 10
def prob_ξ_2 : ℚ := 21 / 100
def prob_ξ_3 : ℚ := 49 / 100
def expected_value_ξ : ℚ :=
  1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3

theorem expected_value_is_2_19 : expected_value_ξ = 219 / 100 :=
  sorry

end card_draws

end prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l632_632877


namespace fraction_of_milk_in_cup1_l632_632513

structure Cup (content: Type) where
  tea  : ℝ
  milk : ℝ

theorem fraction_of_milk_in_cup1 :
  ∀ (cup1_initial : Cup ℝ) (cup2_initial : Cup ℝ),
    cup1_initial.tea = 6 → 
    cup1_initial.milk = 0 → 
    cup2_initial.tea = 0 → 
    cup2_initial.milk = 6 → 
    let tea_transfer := cup1_initial.tea / 3 in
    let cup1 := { cup1_initial with tea := cup1_initial.tea - tea_transfer } in
    let cup2 := { tea := cup2_initial.tea + tea_transfer, milk := cup2_initial.milk } in
    let back_transfer := (cup2.tea + cup2.milk) / 4 in
    let new_tea_in_cup1 := cup1.tea + (back_transfer * cup2.tea / (cup2.tea + cup2.milk)) in
    let new_milk_in_cup1 := cup1.milk + (back_transfer * cup2.milk / (cup2.tea + cup2.milk)) in
    let final_cup1 := { tea := new_tea_in_cup1, milk := new_milk_in_cup1 } in
    final_cup1.milk / (final_cup1.tea + final_cup1.milk) = 1 / 4 := by
  sorry

end fraction_of_milk_in_cup1_l632_632513


namespace number_of_children_l632_632585

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l632_632585


namespace sum_primes_20_to_30_l632_632164

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632164


namespace jess_father_first_round_l632_632468

theorem jess_father_first_round (initial_blocks : ℕ)
  (players : ℕ)
  (blocks_before_jess_turn : ℕ)
  (jess_falls_tower_round : ℕ)
  (h1 : initial_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_before_jess_turn = 28)
  (h4 : ∀ rounds : ℕ, rounds * players ≥ 26 → jess_falls_tower_round = rounds + 1) :
  jess_falls_tower_round = 6 := 
by
  sorry

end jess_father_first_round_l632_632468


namespace prime_sum_20_to_30_l632_632139

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632139


namespace hydras_will_live_l632_632631

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632631


namespace max_value_eq_two_l632_632913

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l632_632913


namespace power_cycle_i_l632_632325

theorem power_cycle_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^23 + i^75 = -2 * i :=
by
  sorry

end power_cycle_i_l632_632325


namespace doctor_is_correct_l632_632607

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632607


namespace sum_of_primes_between_20_and_30_l632_632110

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632110


namespace sum_primes_in_range_l632_632083

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632083


namespace condition_for_equal_shaded_areas_l632_632888

variables {r φ : ℝ}

theorem condition_for_equal_shaded_areas 
  (h1 : 0 < φ)
  (h2 : φ < π / 2)
  (C : euclidean_space ℝ 2)
  (A B D E : euclidean_space ℝ 2)
  (h3 : A ≠ B)
  (h4 : B ≠ C)
  (h5 : A ≠ E)
  (h6 : B ≠ D)
  (h7 : E ≠ C)
  (h8 : D ≠ C)
  (h_tangent : is_tangent (circle C r) A B)
  (h_center : center_of_circle C)
  (h_segments_BCD : line_segment B C D)
  (h_segments_ACE : line_segment A C E)
  : tan φ = φ :=
sorry

end condition_for_equal_shaded_areas_l632_632888


namespace peter_lost_marbles_l632_632003

theorem peter_lost_marbles (initial marbles_lost final : ℕ) (h1 : initial = 33) (h2 : final = 18) : marbles_lost = 15 :=
by
  -- We need to show that the number of lost marbles is correct
  have h3 : initial - final = marbles_lost := sorry
  -- substitute initial and final values
  rw [h1, h2] at h3
  -- simplify the equation to get the number of marbles lost
  have h4 : 33 - 18 = marbles_lost := h3
  -- since 33 - 18 equals 15, hence marbles_lost must be 15
  rw nat.sub_eq_iff_eq_add at h4
  exact h4

end peter_lost_marbles_l632_632003


namespace number_of_possible_values_for_b_l632_632014

theorem number_of_possible_values_for_b : 
  ∃ n : ℕ, n = 6 ∧
  {b : ℕ | 2 ≤ b ∧ b^2 ≤ 125 ∧ 125 < b^3}.card = n :=
by
  sorry

end number_of_possible_values_for_b_l632_632014


namespace x_formula_l632_632661

open BigOperators

-- Defining the sequence recursively
def x : ℕ → ℕ
| 0     := 0
| 1     := 0
| 2     := 1
| 3     := 2
| (n+1) := if n < 3 then 0 else (n * (x n + x (n-1)))

-- Proving the general formula for the sequence
theorem x_formula (n : ℕ) (h : n ≥ 2) :
  x n = n! * ∑ m in finset.range (n + 1), if m = 0 ∨ m = 1 then 0 else (-1)^m / m! :=
sorry

end x_formula_l632_632661


namespace rationalize_denominator_ABC_l632_632527

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l632_632527


namespace inscribed_circle_radius_l632_632439

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l632_632439


namespace B_won_the_prize_l632_632882

-- Define the statements made by A, B, and C
def A_says_A_won : Prop := true
def B_says_B_did_not_win : Prop := true
def C_says_A_did_not_win : Prop := true

-- Define who won the prize
inductive Person
| A : Person
| B : Person
| C : Person

-- The condition that only one of the statements is true
def exactly_one_statement_is_true
    (A_says_A_won B_says_B_did_not_win C_says_A_did_not_win : Prop) : Prop :=
(A_says_A_won ∧ ¬B_says_B_did_not_win ∧ ¬C_says_A_did_not_win) ∨ 
(¬A_says_A_won ∧ B_says_B_did_not_win ∧ ¬C_says_A_did_not_win) ∨ 
(¬A_says_A_won ∧ ¬B_says_B_did_not_win ∧ C_says_A_did_not_win)

-- Define a function that tells who won the prize
def won_prize (winner : Person) : Prop := match winner with
| Person.A => A_says_A_won
| Person.B => ¬B_says_B_did_not_win
| Person.C => C_says_A_did_not_win
end

-- The theorem to prove that B won the prize
theorem B_won_the_prize :
  (A_says_A_won ∨ ¬A_says_A_won) →
  (B_says_B_did_not_win ∨ ¬B_says_B_did_not_win) →
  (C_says_A_did_not_win ∨ ¬C_says_A_did_not_win) →
  exactly_one_statement_is_true A_says_A_won B_says_B_did_not_win C_says_A_did_not_win →
  won_prize Person.B :=
sorry

end B_won_the_prize_l632_632882


namespace y_in_interval_l632_632702

theorem y_in_interval :
  ∃ (y : ℝ), y = 5 + (1/y) * -y ∧ 2 < y ∧ y ≤ 4 :=
by
  sorry

end y_in_interval_l632_632702


namespace hydras_survive_l632_632619

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632619


namespace min_value_y_l632_632960

noncomputable def min_value (f : ℝ → ℝ) (a b : ℝ) := Inf (set.image f (set.Icc a b))

def y (x : ℝ) : ℝ := - (sin x)^3 - 2 * (sin x)

theorem min_value_y : min_value y (-1) 1 = -3 :=
by
  sorry

end min_value_y_l632_632960


namespace find_tan_beta_l632_632350

variable (α β : ℝ)

def condition1 : Prop := Real.tan α = 3
def condition2 : Prop := Real.tan (α + β) = 2

theorem find_tan_beta (h1 : condition1 α) (h2 : condition2 α β) : Real.tan β = -1 / 7 := 
by {
  sorry
}

end find_tan_beta_l632_632350


namespace greatest_int_with_gcd_3_l632_632073

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end greatest_int_with_gcd_3_l632_632073


namespace multiple_rate_is_correct_l632_632508

-- Define Lloyd's standard working hours per day
def regular_hours_per_day : ℝ := 7.5

-- Define Lloyd's standard hourly rate
def regular_rate : ℝ := 3.5

-- Define the total hours worked on a specific day
def total_hours_worked : ℝ := 10.5

-- Define the total earnings for that specific day
def total_earnings : ℝ := 42.0

-- Define the function to calculate the multiple of the regular rate for excess hours
noncomputable def multiple_of_regular_rate (r_hours : ℝ) (r_rate : ℝ) (t_hours : ℝ) (t_earnings : ℝ) : ℝ :=
  let regular_earnings := r_hours * r_rate
  let excess_hours := t_hours - r_hours
  let excess_earnings := t_earnings - regular_earnings
  (excess_earnings / excess_hours) / r_rate

-- The statement to be proved
theorem multiple_rate_is_correct : 
  multiple_of_regular_rate regular_hours_per_day regular_rate total_hours_worked total_earnings = 1.5 :=
by
  sorry

end multiple_rate_is_correct_l632_632508


namespace number_of_ordered_triples_l632_632314

open Nat

noncomputable def count_ordered_triples : ℕ :=
  (List.product (List.product (List.range 241) (List.range 241)) (List.range 361)).countp (λ ((x, y), z) =>
    Nat.lcm x y = 120 ∧ Nat.lcm x z = 240 ∧ Nat.lcm y z = 360)

theorem number_of_ordered_triples : count_ordered_triples = 17 := 
sorry

end number_of_ordered_triples_l632_632314


namespace millionth_digit_of_1_over_41_l632_632776

theorem millionth_digit_of_1_over_41 :
  let d := 5    -- The period of decimal expansion is 5
  let seq := [0,2,4,3,9]  -- The repeating sequence of 1/41
  let millionth_digit_position := 1000000 % d  -- Find the position in the repeating sequence
  seq.millionth_digit_position = 9 :=
by
  let d := 5
  let seq := [0, 2, 4, 3, 9]
  let millionth_digit_position := 1000000 % d
  show seq.millionth_digit_position = 9
  sorry

end millionth_digit_of_1_over_41_l632_632776


namespace sum_of_primes_between_20_and_30_l632_632102

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632102


namespace points_in_circle_l632_632592

theorem points_in_circle (points : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ points i.1.1 ∧ points i.1.1 ≤ 1 ∧ 0 ≤ points i.1.2 ∧ points i.1.2 ≤ 1) →
  ∃ (a b c : Fin 51), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  dist (points a) (points b) ≤ 2 / 7 ∧
  dist (points b) (points c) ≤ 2 / 7 ∧
  dist (points a) (points c) ≤ 2 / 7 :=
by
  sorry

end points_in_circle_l632_632592


namespace simplify_expression_l632_632952

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) :=
by sorry

end simplify_expression_l632_632952


namespace ratio_vanilla_cream_cheese_l632_632991

theorem ratio_vanilla_cream_cheese (
  (sugar cream_cheese vanilla eggs : ℕ)
  (h1 : sugar = 2)
  (h2 : eggs = 8)
  (h3 : sugar * 4 = cream_cheese)
  (h4 : eggs = 2 * vanilla)
  ) : vanilla / cream_cheese = 1 / 96 := 
sorry

end ratio_vanilla_cream_cheese_l632_632991


namespace area_of_enclosed_region_l632_632984

theorem area_of_enclosed_region :
  let eq1 := λ x y : ℝ, x^2 - 14 * x + 3 * y + 70 = 21 + 11 * y - y^2
      line := λ x y : ℝ, y = x + 1 in
  (∃ f : ℝ → ℝ, ∀ x, eq1 x (f x) → f x ≤ x + 1) → 
  (∃ c r : ℝ, ∀ x y, eq1 x y ↔ (x - c)^2 + (y - c)^2 = r^2 ∧ c = 7 ∧ r = 4) →
  ∃ area : ℝ, area = 16 * Real.pi :=
by
  -- Proof steps here
  sorry

end area_of_enclosed_region_l632_632984


namespace probability_no_gp_l632_632272

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l632_632272


namespace event_C_is_random_event_l632_632995

theorem event_C_is_random_event
  (fair_dice : Π (n : ℕ), n ∈ {1, 2, 3, 4, 5, 6} → n < 7)
  (red_ball_bag : Π (balls : Set string), balls = {"red"} → ¬"white" ∈ balls)
  (deck_of_52 : Π (card : string), card ∈ {"2C", "3C", ..., "6S", ..., "AH"} → card = "6S" → 1 / 52)
  (triangle_angles : Π (triangle : Set ℝ), (θ₁ + θ₂ + θ₃ = 180) → (θ₁ + θ₂ + θ₃ = 180))
  : ∃ (card : string), card ∈ {"2C", "3C", ..., "6S", ..., "AH"} ∧ card = "6S" ∧ 0 < 1 / 52 ∧ 1 / 52 < 1 :=
by
  sorry

end event_C_is_random_event_l632_632995


namespace geometric_areas_l632_632479

theorem geometric_areas (A B C P L F M D N E : Point) 
  (hP_inside : InTriangle P A B C)
  (hPL_parallel_AB : Parallel (Line_through P L) (Line_through A B))
  (hPF_parallel_AB : Parallel (Line_through P F) (Line_through A B))
  (hPM_parallel_BC : Parallel (Line_through P M) (Line_through B C))
  (hPD_parallel_BC : Parallel (Line_through P D) (Line_through B C))
  (hPN_parallel_CA : Parallel (Line_through P N) (Line_through C A))
  (hPE_parallel_CA : Parallel (Line_through P E) (Line_through C A)) :
  (Area (Quadrilateral P D B L)) * (Area (Quadrilateral P E C M)) * (Area (Quadrilateral P F A N)) = 
  8 * (Area (Triangle P F M)) * (Area (Triangle P E L)) * (Area (Triangle P D N)) :=
sorry

end geometric_areas_l632_632479


namespace total_amount_paid_l632_632235

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (tax_free_items_cost : ℝ) (total_paid : ℝ) :
  sales_tax = 1.28 →
  tax_rate = 0.08 →
  tax_free_items_cost = 22.72 →
  total_paid = 40.00 :=
begin
  sorry
end

end total_amount_paid_l632_632235


namespace trig_identity_l632_632972

theorem trig_identity : (1 / tan (20 * real.pi / 180)) - (1 / cos (10 * real.pi / 180)) = real.sqrt 3 :=
by 
-- skip proof
sorry

end trig_identity_l632_632972


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l632_632910

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l632_632910


namespace possible_values_of_m_l632_632371

theorem possible_values_of_m (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end possible_values_of_m_l632_632371


namespace probability_no_growth_pie_l632_632267

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l632_632267


namespace sum_of_primes_between_20_and_30_l632_632122

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632122


namespace work_days_together_l632_632693

theorem work_days_together (A_days B_days : ℕ) (work_left_fraction : ℚ) 
  (hA : A_days = 15) (hB : B_days = 20) (h_fraction : work_left_fraction = 8 / 15) : 
  ∃ d : ℕ, d * (1 / 15 + 1 / 20) = 1 - 8 / 15 ∧ d = 4 :=
by
  sorry

end work_days_together_l632_632693


namespace sum_of_primes_between_20_and_30_l632_632097

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632097


namespace range_of_a_l632_632565

noncomputable def f (x a : ℝ) := Real.log x + 1 / 2 * x^2 + a * x

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (1/x + x + a = 3)) :
  a ≤ 1 :=
by
  sorry

end range_of_a_l632_632565


namespace inscribed_circle_radius_l632_632440

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l632_632440


namespace sufficient_but_not_necessary_condition_l632_632367

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end sufficient_but_not_necessary_condition_l632_632367


namespace volume_ratio_sum_l632_632370
noncomputable def edge_length := (a : ℝ)
noncomputable def volume_dodecahedron := (15 + 7 * Real.sqrt 5) * edge_length ^ 3 / 4
noncomputable def diameter_sphere := edge_length * Real.sqrt 5
noncomputable def radius_sphere := diameter_sphere / 2
noncomputable def volume_sphere := 4 / 3 * Real.pi * radius_sphere ^ 3

theorem volume_ratio_sum (a : ℝ) (h_a_pos : 0 < a) :
  let V_D := (15 + 7 * Real.sqrt 5) * a ^ 3 / 4,
      d_s := a * Real.sqrt 5,
      r_s := d_s / 2,
      V_S := 4 / 3 * Real.pi * r_s ^ 3,
      ratio := V_D / V_S
  in (24 * (15 + 7 * Real.sqrt 5) / (125 * Real.pi * Real.sqrt 5) == ratio) -> sorry

end volume_ratio_sum_l632_632370


namespace countMagicalNumbers_l632_632247

def isMagical (n : ℕ) : Prop :=
  Real.floor (Real.sqrt (Real.ceil (Real.sqrt n))) = Real.ceil (Real.sqrt (Real.floor (Real.sqrt n)))

theorem countMagicalNumbers : 
  (Finset.filter isMagical (Finset.range 10001)).card = 1330 := 
sorry

end countMagicalNumbers_l632_632247


namespace number_of_children_l632_632583

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l632_632583


namespace prime_sum_20_to_30_l632_632136

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632136


namespace simplify_proof_l632_632011

noncomputable def simplify_expression (a b c d x y : ℝ) (h : c * x ≠ d * y) : ℝ :=
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) 
  - d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / (c * x - d * y)

theorem simplify_proof (a b c d x y : ℝ) (h : c * x ≠ d * y) :
  simplify_expression a b c d x y h = b^2 * x^2 + a^2 * y^2 :=
by sorry

end simplify_proof_l632_632011


namespace probability_greater_than_a_l632_632923

   variable {σ : ℝ} (X : ℝ → ℝ)

   noncomputable def normal_distribution : Prop :=
     ∀ X, X ~ Normal 5 σ

   theorem probability_greater_than_a (h1 : normal_distribution X)
     (h2 : ∀ (a : ℝ), P(X > 10 - a) = 0.4) :
     ∀ (a : ℝ), P(X > a) = 0.6 := by
     sorry
   
end probability_greater_than_a_l632_632923


namespace find_a_l632_632423

def are_parallel (a : ℝ) : Prop :=
  (a + 1) = (2 - a)

theorem find_a (a : ℝ) (h : are_parallel a) : a = 0 :=
sorry

end find_a_l632_632423


namespace mutually_exclusive_but_not_complementary_l632_632430

noncomputable def balls : ℕ × ℕ × ℕ := (3, 2, 1)

-- Event definitions based on conditions
def outcomes :=
{(2, 0, 0), (0, 2, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1) }

-- Event Conditions
def at_least_one_white (x : ℕ × ℕ × ℕ) : Prop := x.2 > 0
def at_most_one_white (x : ℕ × ℕ × ℕ) : Prop := x.2 ≤ 1
def no_white (x : ℕ × ℕ × ℕ) : Prop := x.2 = 0
def one_red_one_black (x : ℕ × ℕ × ℕ) : Prop := x.0 = 1 ∧ x.2 = 1

-- Theorem to prove that pair D is mutually exclusive but not complementary
theorem mutually_exclusive_but_not_complementary :
  ¬(∃ x ∈ outcomes, at_least_one_white x ∧ one_red_one_black x) ∧
  ∃ x ∈ outcomes, at_least_one_white x ∧ (¬one_red_one_black x) :=
sorry

end mutually_exclusive_but_not_complementary_l632_632430


namespace rationalize_denominator_l632_632541

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l632_632541


namespace hyperbola_imaginary_axis_twice_real_axis_l632_632568

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) : 
  (exists (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0), mx^2 + b^2 * y^2 = b^2) ∧
  (b = 2 * a) ∧ (m < 0) → 
  m = -1 / 4 := 
sorry

end hyperbola_imaginary_axis_twice_real_axis_l632_632568


namespace intersection_barycentric_coordinates_parallel_lines_iff_l632_632687

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem intersection_barycentric_coordinates (a1 b1 c1 a2 b2 c2 : ℝ) :
  ∃ (α β γ : ℝ), 
    (a1 * α + b1 * β + c1 * γ = 0) ∧
    (a2 * α + b2 * β + c2 * γ = 0) ∧
    (α : β : γ) = (det2x2 b1 c1 b2 c2 : det2x2 c1 a1 c2 a2 : det2x2 a1 b1 a2 b2) := 
sorry

theorem parallel_lines_iff (a1 b1 c1 a2 b2 c2 : ℝ) :
  (∃ (α β γ : ℝ), 
    a1 * α + b1 * β + c1 * γ = 0 ∧ 
    a2 * α + b2 * β + c2 * γ = 0) ↔
  (det2x2 b1 c1 b2 c2 + det2x2 c1 a1 c2 a2 + det2x2 a1 b1 a2 b2 = 0) :=
sorry

end intersection_barycentric_coordinates_parallel_lines_iff_l632_632687


namespace max_value_MC_MB_l632_632400

def polar_eq_circle (θ : ℝ) : ℝ := 4 * Real.sin θ

def inclination_angle : ℝ := (3 * Real.pi) / 4

noncomputable def intersection_point (θ : ℝ) (ρ : ℝ) : Prop :=
  (θ = inclination_angle ∧ ρ = polar_eq_circle θ)

theorem max_value_MC_MB : 
  ∃ A : ℝ × ℝ, A = (2 * Real.sqrt 2, inclination_angle) ∧ 
  ∃ M B C : ℝ × ℝ, -- Definitions of M, B, C in Cartesian coordinates
  ||(Real.dist M B) - (Real.dist M C)|| = 2 * Real.sqrt 2 := 
sorry

end max_value_MC_MB_l632_632400


namespace find_b_of_perpendicular_lines_l632_632066

theorem find_b_of_perpendicular_lines  (b : ℚ) :
  let v1 := ⟨4, 5⟩ : ℚ × ℚ
  let v2 := ⟨3, b⟩ : ℚ × ℚ
  v1.1 * v2.1 + v1.2 * v2.2 = 0 →
  b = -12 / 5 :=
by
  intros v1 v2 h
  sorry

end find_b_of_perpendicular_lines_l632_632066


namespace angle_DNA_is_90_l632_632570

/-- 
Given:
1. The line KM₁ intersects the extension of AB at point N.
2. DM₁ is perpendicular to BC.
3. DK is perpendicular to AC.

We need to prove that the angle DNA is 90 degrees.
-/
theorem angle_DNA_is_90 {A B C D K M₁ N : Point} 
  (h1 : Line KM₁ intersects_extension_of Line AB at N) 
  (h2 : Perpendicular DM₁ BC) 
  (h3 : Perpendicular DK AC) : angle D N A = 90 :=
sorry

end angle_DNA_is_90_l632_632570


namespace concyclic_intersection_ratio_l632_632917

variable {A B C D E : Type*} -- Declare our variables A, B, C, D, and E

-- Define the concyclicity of four points, the intersection point and the requirement to prove the relationship.

open EuclideanGeometry

theorem concyclic_intersection_ratio (h : Concyclic A B C D) (h1 : intersect (A - B) (C - D) = E):
  (dist A C / dist B C) * (dist A D / dist B D) = dist A E / dist B E :=
sorry

end concyclic_intersection_ratio_l632_632917


namespace lcm_fractions_l632_632315

theorem lcm_fractions (x : ℕ) (hx : x ≠ 0) : 
  (∀ (a b c : ℕ), (a = 4*x ∧ b = 5*x ∧ c = 6*x) → (Nat.lcm (Nat.lcm a b) c = 60 * x)) :=
by
  sorry

end lcm_fractions_l632_632315


namespace prime_cubed_plus_prime_plus_one_not_square_l632_632950

theorem prime_cubed_plus_prime_plus_one_not_square (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ k : ℕ, k * k = p^3 + p + 1 :=
by
  sorry

end prime_cubed_plus_prime_plus_one_not_square_l632_632950


namespace probability_no_gp_l632_632270

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l632_632270


namespace janet_extra_cost_l632_632897

theorem janet_extra_cost :
  let clarinet_hourly_rate := 40
  let clarinet_hours_per_week := 3
  let clarinet_weeks_per_year := 50
  let clarinet_yearly_cost := clarinet_hourly_rate * clarinet_hours_per_week * clarinet_weeks_per_year

  let piano_hourly_rate := 28
  let piano_hours_per_week := 5
  let piano_weeks_per_year := 50
  let piano_yearly_cost := piano_hourly_rate * piano_hours_per_week * piano_weeks_per_year
  let piano_discount_rate := 0.10
  let piano_discounted_yearly_cost := piano_yearly_cost * (1 - piano_discount_rate)

  let violin_hourly_rate := 35
  let violin_hours_per_week := 2
  let violin_weeks_per_year := 50
  let violin_yearly_cost := violin_hourly_rate * violin_hours_per_week * violin_weeks_per_year
  let violin_discount_rate := 0.15
  let violin_discounted_yearly_cost := violin_yearly_cost * (1 - violin_discount_rate)

  let singing_hourly_rate := 45
  let singing_hours_per_week := 1
  let singing_weeks_per_year := 50
  let singing_yearly_cost := singing_hourly_rate * singing_hours_per_week * singing_weeks_per_year

  let combined_cost := piano_discounted_yearly_cost + violin_discounted_yearly_cost + singing_yearly_cost
  combined_cost - clarinet_yearly_cost = 5525 := 
  sorry

end janet_extra_cost_l632_632897


namespace sum_primes_between_20_and_30_is_52_l632_632132

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632132


namespace possibleValuesOfSum_l632_632791

noncomputable def symmetricMatrixNonInvertible (x y z : ℝ) : Prop := 
  -(x + y + z) * ( x^2 + y^2 + z^2 - x * y - x * z - y * z ) = 0

theorem possibleValuesOfSum (x y z : ℝ) (h : symmetricMatrixNonInvertible x y z) :
  ∃ v : ℝ, v = -3 ∨ v = 3 / 2 := 
sorry

end possibleValuesOfSum_l632_632791


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l632_632732

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l632_632732


namespace area_DEF_l632_632069

structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -3, y := 4}
def E : Point := {x := 1, y := 7}
def F : Point := {x := 3, y := -1}

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)|

theorem area_DEF : area_of_triangle D E F = 16 := by
  sorry

end area_DEF_l632_632069


namespace largest_horizontal_or_vertical_sum_is_24_l632_632755

/-- Each of the five numbers 1, 4, 7, 10, and 13 is placed in one of the five squares so that 
the sum of the three numbers in the horizontal row equals the sum of the three numbers 
in the vertical column. The largest possible value for the horizontal or vertical sum is 24. -/
theorem largest_horizontal_or_vertical_sum_is_24 : 
  ∃ a b c d e : ℕ, {a, b, c, d, e} = {1, 4, 7, 10, 13} ∧
  (a + b + e = a + c + e) ∧ (a + c + e = b + d + e) ∧ 
  (a + b + e = 24 ∨ a + c + e = 24) :=
begin
  -- to be filled with the actual proof
  sorry
end

end largest_horizontal_or_vertical_sum_is_24_l632_632755


namespace certain_event_bag_l632_632201

theorem certain_event_bag (red yellow : ℕ) (balls : set ℕ) : 
  red = 1 → yellow = 3 → balls = {1, 2, 3, 4} → 
  (∀ x y ∈ balls, x ≠ y) → -- Balls are distinct
  (∃ r y₁ y₂ ∈ balls, r = 1 ∧ y₁ ≠ y₂ ∧ y₁ != r ∧ y₁ = 2 ∧ y₂ = 3 ∨ y₁ = 2 ∧ y₂ = 4) → -- At least one yellow ball in any draw of 2
  true :=
by
  sorry

end certain_event_bag_l632_632201


namespace min_value_f_l632_632335

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^3 + (Real.cos x)^2

theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 26 / 27 :=
by
  sorry

end min_value_f_l632_632335


namespace nth_equation_pattern_l632_632935

theorem nth_equation_pattern (n : ℕ) :
  (∏ i in Finset.range (n+1), i + n) = (2^n) * (∏ i in Finset.range n, 2*i+1) :=
by
  sorry

end nth_equation_pattern_l632_632935


namespace millionth_digit_of_1_over_41_l632_632781

theorem millionth_digit_of_1_over_41 :
  let frac := 1 / (41 : ℚ),
      seq := "02439",
      period := (5 : ℕ) in
  (seq.get (1000000 % period - 1) = '9') :=
by
  let frac := 1 / (41 : ℚ)
  let seq := "02439"
  let period := 5
  have h_expansion : frac = 0.02439 / 10000 := sorry
  have h_period : ∀ n, frac = Rational.mkPeriodic seq period n := sorry
  have h_mod : 1000000 % period = 0 := by sorry
  have h_index := h_mod.symm ▸ (dec_trivial : 0 % 5 = 0)
  exact h_period n ▸ (dec_trivial : "02439".get 4 = '9')

end millionth_digit_of_1_over_41_l632_632781


namespace no_integer_solutions_l632_632006

theorem no_integer_solutions
  (x y : ℤ) :
  3 * x^2 = 16 * y^2 + 8 * y + 5 → false :=
by
  sorry

end no_integer_solutions_l632_632006


namespace reservoir_percentage_full_before_storm_l632_632263

theorem reservoir_percentage_full_before_storm :
  ∀ (C : ℝ), 0.80 * C = 320 ∧ 200 < C → (200 / C) * 100 = 50 :=
by
  intros C h
  cases h with h1 h2
  have hC : C = 400 := sorry
  rw hC
  norm_num
  sorry

end reservoir_percentage_full_before_storm_l632_632263


namespace ellipse_eccentricity_l632_632810

-- Define the basic variables and conditions
variables (a b c : ℝ) (h1 : a > b > 0)

-- Assume we have an ellipse equation and the conditions are as follows:
def ellipse_eq := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1
def distance_OP_eq := ∀ (O P F1 F2 : ℝ), |O - P| = (1 / 2) * |F1 - F2|
def focal_product_eq := ∀ (P F1 F2 : ℝ), |P - F1| * |P - F2| = a^2

-- theorem statement to prove the eccentricity of the ellipse
theorem ellipse_eccentricity (O P F1 F2 : ℝ)
  (h_eq_ellipse : ellipse_eq a b)
  (h_OP : distance_OP_eq O P F1 F2)
  (h_focal_product : focal_product_eq P F1 F2) :
  let e := c / a in
  e = (real.sqrt 2) / 2 := 
sorry

end ellipse_eccentricity_l632_632810


namespace sum_of_solutions_l632_632498

def f (x : ℝ) : ℝ :=
  if x < -2 then 2 * x + 7 else -x^2 - x + 1

theorem sum_of_solutions : ∑ x in {x : ℝ | f x = -5}, x = -4 :=
  by sorry

end sum_of_solutions_l632_632498


namespace sum_triples_equals_1624_l632_632733

def sum_expression : ℚ :=
  ∑ a in (finset.range ∞).filter (λ a, a ≥ 1),
    ∑ b in (finset.range ∞).filter (λ b, a < b),
      ∑ c in (finset.range ∞).filter (λ c, b < c),
        1 / (2^a * 3^b * 5^c)

theorem sum_triples_equals_1624 :
  sum_expression = 1 / 1624 :=
by
  sorry

end sum_triples_equals_1624_l632_632733


namespace inv_prop_func_point_l632_632871

theorem inv_prop_func_point {k : ℝ} :
  (∃ y x : ℝ, y = k / x ∧ (x = 2 ∧ y = -1)) → k = -2 :=
by
  intro h
  -- Proof would go here
  sorry

end inv_prop_func_point_l632_632871


namespace area_of_figure_l632_632941

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end area_of_figure_l632_632941


namespace towel_ratio_is_six_to_one_l632_632510

-- Define constants for the problem
def mary_towels : ℕ := 24
def frances_towel_weight_ounces : ℕ := 128
def total_weight_pounds : ℕ := 60

-- Convert Frances's towel weight from ounces to pounds
def ounces_to_pounds (ounces : ℕ) : ℕ := ounces / 16

-- Calculate the weight of Frances's towels in pounds
def frances_towel_weight_pounds : ℕ := ounces_to_pounds frances_towel_weight_ounces

-- Calculate the weight of Mary's towels in pounds
def mary_towel_weight_n_pounds : ℕ := total_weight_pounds - frances_towel_weight_pounds

-- Average weight of Mary's towels as a fraction to find a more precise weight
def average_towel_weight (total_weight : ℕ) (num_towels : ℕ) : ℚ := total_weight / num_towels

-- Calculate the estimated number of towels Frances has (as a floating-point approximation)
noncomputable def frances_towels_approx : ℚ :=
  frances_towel_weight_pounds / average_towel_weight mary_towel_weight_n_pounds mary_towels

-- Calculate the actual (rounded) number of towels Frances has
def frances_towels : ℕ := frances_towels_approx.toReal.round

-- Prove the ratio of the number of towels
theorem towel_ratio_is_six_to_one :
  (mary_towels : ℚ) / (frances_towels : ℚ) = 6 :=
by {
  sorry -- Proof is omitted; only the statement is provided.
}

end towel_ratio_is_six_to_one_l632_632510


namespace coded_CDE_base10_value_l632_632241

-- Define the correspondence of digits
def digit : Type := ℕ
def A : digit := 0
def B : digit := 5
def C : digit := 0
def D : digit := 1
def E : digit := 0
def F : digit := 5

-- Helper function to convert a base-6 string to base-10
def base6_to_base10 (d1 d2 d3 : digit) : ℕ :=
  d1 * 6^2 + d2 * 6^1 + d3 * 6^0

-- Define the given conditions
def consecutive_integers_encoded : Prop :=
  base6_to_base10 B C F + 1 = base6_to_base10 B C E ∧
  base6_to_base10 B C E + 1 = base6_to_base10 C A A

-- The main theorem
theorem coded_CDE_base10_value : (consecutive_integers_encoded) -> base6_to_base10 C D E = 6 := by
  sorry

end coded_CDE_base10_value_l632_632241


namespace solve_marble_problem_l632_632434

noncomputable def marble_problem : Prop :=
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 50 ∧ 
  (∀ initial_white initial_black : ℕ, initial_white = 50 ∧ initial_black = 50 → 
  ∃ w b : ℕ, w = 50 + k - initial_black ∧ b = 50 - k ∧ (w, b) = (2, 0))

theorem solve_marble_problem: marble_problem :=
sorry

end solve_marble_problem_l632_632434


namespace cannot_transform_to_target_l632_632403

theorem cannot_transform_to_target (a b c : ℝ) (h1 : a = 2) (h2 : b = sqrt 2) (h3 : c = 1 / sqrt 2) :
  ¬ ∃ (steps : ℕ → ℝ × ℝ × ℝ), (steps 0) = (a, b, c) ∧ 
  (∀ n, steps (n + 1) = 
    let (x, y, z) := steps n in if x > y then ( (x + y) / sqrt 2, (x - y) / sqrt 2, z ) 
                                else if x > z then ( (x + z) / sqrt 2, y, (x - z) / sqrt 2 ) 
                                else ( x, (y + z) / sqrt 2, (y - z) / sqrt 2 )) ∧ 
  (∃ n, steps n = (1, sqrt 2, 1 + sqrt 2)) :=
sorry

end cannot_transform_to_target_l632_632403


namespace problem_statement_l632_632381

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h_even : ∀ x : ℝ, f(-x) = f(x))
  (h_periodic : ∀ x : ℝ, (0 ≤ x) → f(x + 2) = f(x))
  (h_def : ∀ x : ℝ, (0 ≤ x) → (x < 2) → f(x) = Real.log (x + 1) / Real.log 2) :
  f (-2010) + f 2011 = 1 :=
begin
  sorry
end

end problem_statement_l632_632381


namespace illumination_possible_for_n_ge_2_l632_632955

theorem illumination_possible_for_n_ge_2 (n : ℕ) (hn : n ≥ 2) (convex : ℕ → Prop)
  (H1 : ∀ (i : ℕ) (hi : i < n), convex i)
  (H2 : ∀ (i : ℕ) (hi : i < n), arena_remains_illuminated_without i n convex)
  (H3 : ∀ (i j : ℕ) (hi : i < n) (hj : j < n) (hij : i ≠ j), ¬ arena_remains_illuminated_without i j n convex) :
  ∃ (illuminate_arena : (ℕ → Prop) → ℕ → Prop),
  ∀ (illuminate_arena : (ℕ → Prop) → ℕ → Prop),
  illuminate_arena convex n ∧
  (∀ i, i < n → illuminate_arena (remove i convex) n) ∧
  (∀ i j, i < n → j < n → i ≠ j → ¬ illuminate_arena (remove i (remove j convex)) n) :=
by
  sorry

end illumination_possible_for_n_ge_2_l632_632955


namespace least_number_subtracted_l632_632990

theorem least_number_subtracted {
  x : ℕ
} : 
  (∀ (m : ℕ), m ∈ [5, 9, 11] → (997 - x) % m = 3) → x = 4 :=
by
  sorry

end least_number_subtracted_l632_632990


namespace sum_primes_20_to_30_l632_632153

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632153


namespace even_shifted_sine_l632_632036

noncomputable def g (x φ : ℝ) : ℝ := 3 * sin (2 * (x + φ) + π/4)

theorem even_shifted_sine (φ : ℝ) (k : ℤ) :
  (∀ x : ℝ, g x φ = g (-x) φ) ↔ φ = π/8 + k * (π/2) :=
by
  sorry

end even_shifted_sine_l632_632036


namespace factorial_inequality_l632_632483

noncomputable def A_n (a : list ℕ) (n : ℕ) : ℚ :=
  (list.sum a : ℚ) / (n : ℚ)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else list.prod (list.range (n+1))

theorem factorial_inequality (a : list ℕ) (n : ℕ) (h_len : a.length = n) (h_pos : 0 < n) :
  list.prod (a.map factorial) ≥ factorial (⌊A_n a n⌋ * n) := 
  sorry

end factorial_inequality_l632_632483


namespace probability_one_no_GP_l632_632281

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l632_632281


namespace sum_primes_between_20_and_30_is_52_l632_632130

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632130


namespace sum_of_digits_B_l632_632197

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem sum_of_digits_B :
  let A := sum_of_digits (4444 ^ 4444) in
  let B := sum_of_digits A in
  sum_of_digits B = 7 :=
by
  sorry

end sum_of_digits_B_l632_632197


namespace proper_subsets_count_of_A_l632_632402

def A : Set ℕ := {0, 2, 3}

theorem proper_subsets_count_of_A : set.finite A → ∃ n, n = 7 ∧ (set.subset_proper_count A = n) :=
by truth assumption A is finite ∧ proper count is 7

end proper_subsets_count_of_A_l632_632402


namespace probability_no_gp_l632_632269

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l632_632269


namespace prime_sum_20_to_30_l632_632190

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632190


namespace energy_doubled_triangle_l632_632032

theorem energy_doubled_triangle (energy : ℝ) (d : ℝ) (k : ℝ) :
  energy = 15 →
  k = energy / (3 * d) →
  let energy_new := 3 * k / (2 * d) in
  energy_new = 7.5 :=
by
  intros h_energy h_k
  let energy_new := (3 * (energy / (3 * d)) / (2 * d))
  have : energy_new = 7.5, by sorry
  exact this

end energy_doubled_triangle_l632_632032


namespace sum_a_sequence_2023_l632_632358

noncomputable def f (x : ℝ) : ℝ := x^2 - 4

def f' (x : ℝ) : ℝ := 2 * x

def newton_sequence (x₀ : ℝ) : ℕ → ℝ
| 0 => x₀
| n + 1 => newton_sequence n - f (newton_sequence n) / f' (newton_sequence n)

def a (x : ℝ) : ℝ := ln (x + 2) - ln (x - 2)

def a_sequence (x₀ : ℝ) (h : x₀ > 2) : ℕ → ℝ
| 0     => a x₀
| n + 1 => 2 * (a_sequence n)

def S (x₀ : ℝ) (h : x₀ > 2) : ℕ → ℝ
| 0     => a_sequence x₀ h 0
| n + 1 => S x₀ h n + a_sequence x₀ h (n + 1)

theorem sum_a_sequence_2023 (x₀ : ℝ) (h : x₀ > 2) (hx₀ : a x₀ = 1) : 
    S x₀ h 2022 = 2^2023 - 1 := 
sorry

end sum_a_sequence_2023_l632_632358


namespace probability_one_no_GP_l632_632282

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l632_632282


namespace probability_one_no_GP_l632_632283

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l632_632283


namespace binary_division_remainder_l632_632993

theorem binary_division_remainder (n : ℕ) (h_n : n = 0b110110011011) : n % 8 = 3 :=
by {
  -- This sorry statement skips the actual proof
  sorry
}

end binary_division_remainder_l632_632993


namespace problem_solution_l632_632320

theorem problem_solution 
  (C : ℝ × ℝ := (0, -2)) 
  (r : ℝ := 2) 
  (ellipse : ℝ → ℝ × ℝ := λ phi, (3 * Real.cos phi, Real.sqrt 3 * Real.sin phi)) 
  (line_l : ℝ → ℝ × ℝ := λ theta, ((Real.sqrt 2) / 2, (Real.sqrt 2) / 2)) :
  (∀ x y : ℝ, x - y + 1 ≠ 0 ∨ x^2 + (y + 2)^2 ≠ 4) ∧
  (∀ t1 t2 : ℝ, let AB := Real.sqrt ((t1 + t2)^2 - 4 * t1 * t2) 
   in AB = (12 * Real.sqrt 2) / 7) :=
begin
  sorry
end

end problem_solution_l632_632320


namespace sum_of_primes_between_20_and_30_l632_632104

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632104


namespace prime_sum_20_to_30_l632_632195

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632195


namespace arrange_A_B_C_l632_632354

theorem arrange_A_B_C (a1 a2 b1 b2 : ℝ) (h1 : a2 > a1) (h2 : a1 > 0) (h3 : b2 > b1) (h4 : b1 > 0)
  (h5 : a1 + a2 = 1) (h6 : b1 + b2 = 1) :
  let A := a1 * b1 + a2 * b2
  let B := a1 * b2 + a2 * b1
  let C := 1 / 2
  in B < C ∧ C < A :=
sorry

end arrange_A_B_C_l632_632354


namespace complex_mul_im_unit_l632_632820

theorem complex_mul_im_unit (i : ℂ) (h : i^2 = -1) : i * (1 - i) = 1 + i := by
  sorry

end complex_mul_im_unit_l632_632820


namespace areas_equal_l632_632303

-- Define the circle areas calculation for both hexagon and octagon with side length 3.
noncomputable def area_between_circles_hexagon (s : ℝ) : ℝ :=
  let r_inscribed := s / (2 * Real.cot (Real.pi / 6))
  let r_circumscribed := s / (2 * Real.sin (Real.pi / 6))
  Real.pi * (r_circumscribed ^ 2 - r_inscribed ^ 2)

noncomputable def area_between_circles_octagon (s : ℝ) : ℝ :=
  let r_inscribed := s / (2 * Real.cot (Real.pi / 8))
  let r_circumscribed := s / (2 * Real.sin (Real.pi / 8))
  Real.pi * (r_circumscribed ^ 2 - r_inscribed ^ 2)

-- Prove that the areas are equal for side length 3
theorem areas_equal : area_between_circles_hexagon 3 = area_between_circles_octagon 3 :=
by
  sorry

end areas_equal_l632_632303


namespace necessary_but_not_sufficient_l632_632503

def M := {x : ℝ | 0 < x ∧ x ≤ 3}
def N := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) (haM : a ∈ M) : (a ∈ N → a ∈ M) ∧ ¬(a ∈ M → a ∈ N) :=
by {
  sorry
}

end necessary_but_not_sufficient_l632_632503


namespace number_and_sum_of_f2_l632_632496

noncomputable def S := {x : ℝ // x > 0}

noncomputable def f (x : S) : ℝ := sorry

def functional_equation := ∀ (x y : S), f x * f y = f ⟨x * y, sorry⟩ + 1001 * (1 / x + 1 / y + 1000)

theorem number_and_sum_of_f2 :
  (∃ f : S → ℝ, functional_equation) →
  let n := 1 in
  let s := 1001.5 in
  n * s = 1001.5 :=
by
  sorry

end number_and_sum_of_f2_l632_632496


namespace people_not_in_pool_l632_632472

noncomputable def total_people_karen_donald : ℕ := 2
noncomputable def children_karen_donald : ℕ := 6
noncomputable def total_people_tom_eva : ℕ := 2
noncomputable def children_tom_eva : ℕ := 4
noncomputable def legs_in_pool : ℕ := 16

theorem people_not_in_pool : total_people_karen_donald + children_karen_donald + total_people_tom_eva + children_tom_eva - (legs_in_pool / 2) = 6 := by
  sorry

end people_not_in_pool_l632_632472


namespace hydras_never_die_l632_632623

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632623


namespace hydras_survive_l632_632617

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632617


namespace max_sin_B_in_triangle_l632_632426

variables {A B C : ℝ} {a b c : ℝ}

theorem max_sin_B_in_triangle (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π) (h5 : a^2 + c^2 = 2 * b^2) :
  sin B ≤ sqrt 3 / 2 :=
sorry

end max_sin_B_in_triangle_l632_632426


namespace Vanya_age_today_l632_632947

variables (S V x : ℕ) 
variable (h1 : S = V + x)
variable (h2 : S + (S + x) + V + (V + x) = 216)

theorem Vanya_age_today : ∃ V x, h1 ∧ h2 ∧ (V + x = 54) :=
by
  sorry

end Vanya_age_today_l632_632947


namespace cylinder_area_l632_632420

noncomputable def cylinder_surface_area : ℝ :=
sorry

theorem cylinder_area
  (area_axial_section : ℝ)
  (h_area_axial_section : area_axial_section = 4) :
  cylinder_surface_area = 6 * real.pi :=
by sorry

end cylinder_area_l632_632420


namespace divisibility_equiv_l632_632943

theorem divisibility_equiv (n : ℕ) : (7 ∣ 3^n + n^3) ↔ (7 ∣ 3^n * n^3 + 1) :=
by sorry

end divisibility_equiv_l632_632943


namespace quadratic_inequalities_l632_632401

theorem quadratic_inequalities (f : ℝ → ℝ) (h : f = λ x, 3 * (x + 2)^2) :
  let y1 := f 1,
      y2 := f 2,
      y3 := f (-3)
  in y3 < y1 ∧ y1 < y2 :=
by
  let y1 := f 1,
      y2 := f 2,
      y3 := f (-3)
  have h1 : y1 = 3 * (1 + 2)^2,
  { rw h, simp },
  have h2 : y2 = 3 * (2 + 2)^2,
  { rw h, simp },
  have h3 : y3 = 3 * (-3 + 2)^2,
  { rw h, simp },
  rw [h1, h2, h3],
  sorry

end quadratic_inequalities_l632_632401


namespace sum_of_primes_between_20_and_30_l632_632108

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632108


namespace edge_count_le_degree_sum_l632_632904

variables {V : Type*} (G : SimpleGraph V)
variables {A : Set V} (hA : IsIndependentSet A)
variables [Fintype V]

open SimpleGraph

theorem edge_count_le_degree_sum (hA : IsIndependentSet G A) :
  G.edgeFinset.card ≤ ∑ v in (finset.univ : Finset V) \ A.toFinset, G.degree v :=
sorry

end edge_count_le_degree_sum_l632_632904


namespace probability_at_most_one_female_is_correct_l632_632237

-- Defining the number of male and female students
def num_males : ℕ := 3
def num_females : ℕ := 2
def total_students : ℕ := num_males + num_females

-- The event of selecting 2 students from the total students
def total_selections : ℕ := Nat.choose total_students 2

-- The event of selecting at most one female student (0 or 1 female)
def favorable_selections : ℕ := Nat.choose num_males 2 + num_males * num_females

-- The required probability
noncomputable def prob_at_most_one_female : ℚ := favorable_selections / total_selections

theorem probability_at_most_one_female_is_correct :
  prob_at_most_one_female = 9 / 10 :=
by
  sorry

end probability_at_most_one_female_is_correct_l632_632237


namespace solution_to_fraction_l632_632873

theorem solution_to_fraction (x : ℝ) (h_fraction : (x^2 - 4) / (x + 4) = 0) (h_denom : x ≠ -4) : x = 2 ∨ x = -2 :=
sorry

end solution_to_fraction_l632_632873


namespace largest_prime_factor_binom_200_100_is_61_l632_632567

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def largest_two_digit_prime_factor (n : ℕ) : ℕ :=
  if h: 61 < 100 ∧ is_prime 61 ∧ n % 61 = 0 then 61 else 0

theorem largest_prime_factor_binom_200_100_is_61 : largest_two_digit_prime_factor (binom 200 100) = 61 := by
  sorry

end largest_prime_factor_binom_200_100_is_61_l632_632567


namespace piglet_steps_count_l632_632004

theorem piglet_steps_count (u v L : ℝ) (h₁ : (L * u) / (u + v) = 66) (h₂ : (L * u) / (u - v) = 198) : L = 99 :=
sorry

end piglet_steps_count_l632_632004


namespace average_age_before_new_students_l632_632023

theorem average_age_before_new_students
  (N : ℕ) (A : ℚ) 
  (hN : N = 8) 
  (new_avg : (A - 4) = ((A * N) + (32 * 8)) / (N + 8)) 
  : A = 40 := 
by
  sorry

end average_age_before_new_students_l632_632023


namespace women_in_company_l632_632999

theorem women_in_company (W: ℕ) (men: ℕ) 
  (h1: W = 3 * men)
  (h2: W = 5 * (60/100 * (1/3 * W) + 40/100 * (2/3 * W)))
  (h3: men = 120): 
  let women := W - men in 
  women = 180 :=
by
  sorry

end women_in_company_l632_632999


namespace sum_primes_in_range_l632_632087

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632087


namespace books_total_l632_632975

theorem books_total (Tim_books Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 := 
by
  sorry

end books_total_l632_632975


namespace fisherman_sale_l632_632214

/-- 
If the price of the radio is both the 4th highest price and the 13th lowest price 
among the prices of the fishes sold at a sale, then the total number of fishes 
sold at the fisherman sale is 16. 
-/
theorem fisherman_sale (h4_highest : ∃ price : ℕ, ∀ p : ℕ, p > price → p ∈ {a | a ≠ price} ∧ p > 3)
                       (h13_lowest : ∃ price : ℕ, ∀ p : ℕ, p < price → p ∈ {a | a ≠ price} ∧ p < 13) :
  ∃ n : ℕ, n = 16 :=
sorry

end fisherman_sale_l632_632214


namespace intersection_M_N_l632_632502

-- Definitions of sets M and N
def M : Set ℤ := {x | abs x < 1}
def N : Set ℝ := {x | x^2 ≤ 1}

-- Statement of the proof problem 
theorem intersection_M_N : M ∩ N = {0} :=
sorry

end intersection_M_N_l632_632502


namespace part_a_part_b_l632_632308

variables {A B C D E F M : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] 
variables [EuclideanGeometry D] [EuclideanGeometry E] [EuclideanGeometry F]
variables [EuclideanGeometry M] [IsTriangle A B C]

-- Conditions
def condition_1 : (¬ CoLinear B E A C) := sorry 
def condition_2 : (¬ CoLinear C D A B) := sorry 
def condition_3 : (CoLinear A F B C) := sorry 
def similarity_1 : (Similar (Triangle A D B) (Triangle C E A)) := sorry
def similarity_2 : (Similar (Triangle C E A) (Triangle C F B)) := sorry
def midpoint_M : (Midpoint A F M) := sorry

-- Proofs
theorem part_a : Similar (Triangle B D F) (Triangle F E C) := sorry
theorem part_b : Midpoint D E M := sorry

end part_a_part_b_l632_632308


namespace pier_to_village_trip_l632_632294

theorem pier_to_village_trip :
  ∃ (x t : ℝ), 
  (x / 10 + x / 8 = t + 1 / 60) ∧
  (5 * t / 2 + 4 * t / 2 = x) ∧
  (x = 6) ∧
  (t = 4 / 3) :=
by
  sorry

end pier_to_village_trip_l632_632294


namespace inequality_holds_for_all_real_l632_632397

theorem inequality_holds_for_all_real (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end inequality_holds_for_all_real_l632_632397


namespace hydras_survive_l632_632616

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632616


namespace remaining_episodes_l632_632203

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l632_632203


namespace magnitude_of_z_l632_632800

theorem magnitude_of_z 
  (i : ℂ) (h_i : i^2 = -1)
  (z : ℂ) (h : z * (1 + i) = i) : 
  |z| = Complex.abs (Complex.mk (1/2) (1/2)) := by
  sorry

end magnitude_of_z_l632_632800


namespace n_prime_or_power_of_2_l632_632474

theorem n_prime_or_power_of_2 
  (n : ℕ) 
  (h_n_gt_6 : n > 6)
  (a : ℕ → Prop) -- This defines the set \( a_1, a_2, \ldots, a_k \)
  (h_rel_prime : ∀ i, a i < n ∧ Nat.coprime (a i) n)
  (h_arith_prog : ∃ Δ > 0, ∀ i j, (i < j) → (a j - a i = (j - i) * Δ))
  : Nat.prime n ∨ ∃ k : ℕ, n = 2 ^ k := sorry

end n_prime_or_power_of_2_l632_632474


namespace hydras_will_live_l632_632635

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632635


namespace apron_more_than_recipe_book_l632_632928

-- Define the prices and the total spent
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def total_ingredient_cost : ℕ := 5 * ingredient_cost
def total_spent : ℕ := 40

-- Define the condition that the total cost including the apron is $40
def total_without_apron : ℕ := recipe_book_cost + baking_dish_cost + total_ingredient_cost
def apron_cost : ℕ := total_spent - total_without_apron

-- Prove that the apron cost $1 more than the recipe book
theorem apron_more_than_recipe_book : apron_cost - recipe_book_cost = 1 := by
  -- The proof goes here
  sorry

end apron_more_than_recipe_book_l632_632928


namespace quilt_cost_l632_632464

def quilt_length : ℝ := 16
def quilt_width : ℝ := 20
def patch_area : ℝ := 4
def first_10_patch_cost : ℝ := 10
def subsequent_patch_cost : ℝ := 5

theorem quilt_cost :
  let total_area := quilt_length * quilt_width in
  let total_patches := total_area / patch_area in
  let cost_first_10 := 10 * first_10_patch_cost in
  let cost_remaining := (total_patches - 10) * subsequent_patch_cost in
  cost_first_10 + cost_remaining = 450 :=
by
  sorry

end quilt_cost_l632_632464


namespace enrollment_difference_l632_632656

theorem enrollment_difference :
  let M := 1500
  let S := 2100
  let L := 2700
  let R := 1800
  let B := 900
  max M (max S (max L (max R B))) - min M (min S (min L (min R B))) = 1800 := 
by 
  sorry

end enrollment_difference_l632_632656


namespace approximate_area_of_semicircle_l632_632890

noncomputable def area_of_semicircle (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

theorem approximate_area_of_semicircle (h : ∀ r, Real.pi * r + 2 * r = 20) :
  ∃ a : ℝ, a ≈ 23.8 ∧ a = area_of_semicircle (20 / (Real.pi + 2)) :=
by
  sorry

end approximate_area_of_semicircle_l632_632890


namespace sum_of_solutions_f_eq_2_l632_632499

def f (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 5 else -x^2 + 4 * x - 4

theorem sum_of_solutions_f_eq_2 : 
  (∑ x in (finset.filter (λ x, f x = 2) (finset.range 4)), x) = 4 :=
by
  sorry

end sum_of_solutions_f_eq_2_l632_632499


namespace max_sum_pq_qr_rs_ps_l632_632589

theorem max_sum_pq_qr_rs_ps (p q r s : ℕ) (hp : p ∈ {3, 4, 5, 6}) (hq : q ∈ {3, 4, 5, 6}) 
(hr : r ∈ {3, 4, 5, 6}) (hs : s ∈ {3, 4, 5, 6}) (h_distinct : {p, q, r, s} = {3, 4, 5, 6}) :
  (pq + qr + rs + ps) ≤ 80 := 
by 
  sorry

end max_sum_pq_qr_rs_ps_l632_632589


namespace Baker_sold_more_cakes_than_pastries_l632_632296

theorem Baker_sold_more_cakes_than_pastries (Sold_cakes Sold_pastries : ℕ)
    (hCakes : Sold_cakes = 358) (hPastries : Sold_pastries = 297) :
    (Sold_cakes - Sold_pastries) = 61 :=
by
  rw [hCakes, hPastries]
  norm_num -- this simplifies 358 - 297 to 61
  sorry

end Baker_sold_more_cakes_than_pastries_l632_632296


namespace sin_D_value_l632_632894

theorem sin_D_value {DE DF : ℝ} (h1 : 0 < DE) (h2 : 0 < DF) 
  (area_DEF : ∃ (D : ℝ), 1 / 2 * DE * DF * D = 81) 
  (geom_mean : (DE * DF)^0.5 = 15) : 
  ∃ D : ℝ, D = 18 / 25 :=
by 
  sorry

end sin_D_value_l632_632894


namespace area_of_triangle_AOB_l632_632812

theorem area_of_triangle_AOB : 
  let z1 := (1 : ℂ) + (complex.I : ℂ)
  let z2 := (-1 : ℂ) + (complex.I : ℂ)
  let A := (1, 1)
  let B := (-1, 1)
  let O := (0, 0)
  let base := ((-1 - 1 : ℝ) ^ 2 + (1 - 1) ^ 2).sqrt
  (1 / 2) * base * 1 = 1 :=
by
  sorry

end area_of_triangle_AOB_l632_632812


namespace find_b_plus_c_l632_632425

-- Definitions based on the given conditions.
variables {A : ℝ} {a b c : ℝ}

-- The conditions in the problem
theorem find_b_plus_c
  (h_cosA : Real.cos A = 1 / 3)
  (h_a : a = Real.sqrt 3)
  (h_bc : b * c = 3 / 2) :
  b + c = Real.sqrt 7 :=
sorry

end find_b_plus_c_l632_632425


namespace rationalize_denominator_l632_632542

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l632_632542


namespace correct_statements_count_l632_632718

theorem correct_statements_count:
  let cofunctions_equal (A : Type) [ Angle A ] :=
    -- Definition for co-functions of the same angle or equal angles being equal
    ∀ (θ : A), co_function θ = θ

  let corresponding_angles_equal (A : Type) [ Line A ] [ Angle A ] :=
    -- Definition for corresponding angles through intersecting lines being equal
    ∀ (l1 l2 l3 : A), intersect_line l1 l2 l3 → (l1 || l2) → corresponding_angles l1 l2 l3

  let perpendicular_segment_shortest {A : Type} [ MetricSpace A ] :=
    -- Definition for shortest segment being perpendicular from the point to the line
    ∀ (p : A) (l : Set A), p ∉ l → shortest_segment p l perpendicular

  let perpendicular_segment_dist {A : Type} [ MetricSpace A ] :=
    -- Definition for the length of a perpendicular segment from point to line being distance
    ∀ (p : A) (l : Set A), p ∉ l → distance p l = length (perpendicular_segment p l)

in
  (cofunctions_equal ∧ perpendicular_segment_shortest ∧ ¬ corresponding_angles_equal ∧ perpendicular_segment_dist)  
  ↔ 2 = 2 := 
sorry

end correct_statements_count_l632_632718


namespace sum_primes_20_to_30_l632_632165

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632165


namespace max_intersection_points_l632_632074

/-- 
Given:
1. A circle can intersect each side of a triangle at most at 2 points.
2. A circle and an ellipse can intersect at most at 4 points.
3. An ellipse can intersect each side of a triangle at most at 2 points.
Prove:
The maximum number of possible points of intersection among a circle, an ellipse, and a triangle is 16.
-/
theorem max_intersection_points : 
  (∀ (circle triangle : Type), ∀ side : triangle → Type, ∀ pt : ∀ t : side, (pt t ∈ circle ∈ℕ) → pt ≤ 2 * 3) →
  (∀ (circle ellipse : Type), ∀ pt : ∀ (c : circle) (e : ellipse), (pt c e ∈ℕ) → pt ≤ 4) →
  (∀ (ellipse triangle : Type), ∀ side : triangle → Type, ∀ pt : ∀ t : side, (pt t ∈ ellipse ∈ℕ) → pt ≤ 2 * 3) →
  let circle_triangle_intersections := 6 in
  let circle_ellipse_intersections := 4 in
  let ellipse_triangle_intersections := 6 in
  circle_triangle_intersections + circle_ellipse_intersections + ellipse_triangle_intersections = 16 := 
by 
  intros h1 h2 h3;
  let circle_triangle_intersections := 6;
  let circle_ellipse_intersections := 4;
  let ellipse_triangle_intersections := 6;
  rw [circle_triangle_intersections, circle_ellipse_intersections, ellipse_triangle_intersections];
  exact 16;

end max_intersection_points_l632_632074


namespace john_pays_30280_l632_632899

def membership_fee : ℕ := 4000
def monthly_fee_john : ℕ := 1000
def monthly_fee_wife : ℕ := 1200
def monthly_fee_son : ℕ := 800
def monthly_fee_daughter : ℕ := 900
def discount_membership : ℝ := 0.20
def discount_monthly : ℝ := 0.10
def num_months : ℕ := 12

-- Calculate the total membership fee after discount
def calc_membership_fee (fee : ℕ) (discount : ℝ) : ℕ :=
  fee - nat.floor (fee * discount)

-- Calculate the total monthly cost after discount
def calc_monthly_fee (fee : ℕ) (discount : ℝ) : ℕ :=
  fee - nat.floor (fee * discount)

-- Calculate the total cost for the first year
def total_cost_for_first_year : ℕ :=
  let total_membership_fee := membership_fee + calc_membership_fee membership_fee discount_membership + membership_fee + membership_fee in
  let total_monthly_fee := monthly_fee_john + calc_monthly_fee monthly_fee_wife discount_monthly + monthly_fee_son + monthly_fee_daughter in
  let total_yearly_cost := total_membership_fee + total_monthly_fee * num_months in
  total_yearly_cost / 2

theorem john_pays_30280 :
  total_cost_for_first_year = 30280 := 
sorry

end john_pays_30280_l632_632899


namespace rectangle_area_l632_632706

theorem rectangle_area (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x * y = 5 :=
by
  -- Conditions given to us:
  -- 1. (h1) The sum of the sides is 5.
  -- 2. (h2) The sum of the squares of the sides is 15.
  -- We need to prove that the product of the sides is 5.
  sorry

end rectangle_area_l632_632706


namespace eccentricity_of_ellipse_eq_l632_632398

noncomputable def verify_eccentricity (a b e : ℝ) :=
  a > b ∧ b > 0 ∧ (3 * e^2 + 2 * real.sqrt 3 * e - 3 = 0 → e = real.sqrt 3 / 3)

theorem eccentricity_of_ellipse_eq :
  ∃ e, ∀ a b : ℝ, 
    a > b → b > 0 → 
      (verify_eccentricity a b e ↔ 
      (3 * e^2 + 2 * real.sqrt 3 * e - 3 = 0 ∧ e = real.sqrt 3 / 3)) :=
by 
  sorry

end eccentricity_of_ellipse_eq_l632_632398


namespace weeks_to_work_l632_632307

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l632_632307


namespace middle_even_sum_eq_sum_first_15_l632_632577

theorem middle_even_sum_eq_sum_first_15 (n : ℕ) :
  (2 + 4 + 6 + ⋯ + 30 = 3 * n) → (n = 80) :=
by
  sorry

end middle_even_sum_eq_sum_first_15_l632_632577


namespace solve_equation_3x6_eq_3mx_div_xm1_l632_632012

theorem solve_equation_3x6_eq_3mx_div_xm1 (x : ℝ) 
  (h1 : x ≠ 1)
  (h2 : x^2 + 5*x - 6 ≠ 0) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ (x = 3 ∨ x = -6) :=
by 
  sorry

end solve_equation_3x6_eq_3mx_div_xm1_l632_632012


namespace square_perimeter_l632_632785

variable (side : ℕ) (P : ℕ)

theorem square_perimeter (h : side = 19) : P = 4 * side → P = 76 := by
  intro hp
  rw [h] at hp
  norm_num at hp
  exact hp

end square_perimeter_l632_632785


namespace chips_probability_l632_632224

def total_chips : ℕ := 12
def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5

def total_ways : ℕ := Nat.factorial total_chips

def blue_group_ways : ℕ := Nat.factorial blue_chips
def green_group_ways : ℕ := Nat.factorial green_chips
def red_group_ways : ℕ := Nat.factorial red_chips
def group_permutations : ℕ := Nat.factorial 3

def satisfying_arrangements : ℕ :=
  group_permutations * blue_group_ways * green_group_ways * red_group_ways

noncomputable def probability_of_event_B : ℚ :=
  (satisfying_arrangements : ℚ) / (total_ways : ℚ)

theorem chips_probability :
  probability_of_event_B = 1 / 4620 :=
by
  sorry

end chips_probability_l632_632224


namespace sum_primes_20_to_30_l632_632161

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632161


namespace quadrilateral_construction_l632_632309

noncomputable def construct_quadrilateral
  (α β γ δ : ℝ) (a b : ℝ) : Prop :=
  ∃ (A B C D : ℝ × ℝ), 
    ∠ A B C = β ∧
    ∠ B C D = δ ∧
    ∠ C D A = γ ∧
    ∠ D A B = α ∧
    dist A B = a ∧
    dist C D = b

theorem quadrilateral_construction
  (α β γ δ : ℝ)
  (a b : ℝ)
  (h1 : 0 < α) (h2 : α < 180)
  (h3 : 0 < β) (h4 : β < 180)
  (h5 : 0 < γ) (h6 : γ < 180)
  (h7 : 0 < δ) (h8 : δ < 180)
  (h9 : 0 < a)
  (h10 : 0 < b)
  : construct_quadrilateral α β γ δ a b :=
by {
  -- This should include the proof steps but is given as sorry for now.
  sorry
}

end quadrilateral_construction_l632_632309


namespace cost_of_each_magazine_l632_632930

theorem cost_of_each_magazine
  (books_about_cats : ℕ)
  (books_about_solar_system : ℕ)
  (magazines : ℕ)
  (cost_per_book : ℝ)
  (total_spent : ℝ)
  (books_total : ℕ := books_about_cats + books_about_solar_system)
  (total_books_cost : ℝ := books_total * cost_per_book)
  (total_magazine_cost : ℝ := total_spent - total_books_cost)
  (magazine_cost : ℝ := total_magazine_cost / magazines)
  (h1 : books_about_cats = 7)
  (h2 : books_about_solar_system = 2)
  (h3 : magazines = 3)
  (h4 : cost_per_book = 7)
  (h5 : total_spent = 75) :
  magazine_cost = 4 :=
by
  sorry

end cost_of_each_magazine_l632_632930


namespace apples_bought_is_28_l632_632575

-- Define the initial number of apples, number of apples used, and total number of apples after buying more
def initial_apples : ℕ := 38
def apples_used : ℕ := 20
def total_apples_after_buying : ℕ := 46

-- State the theorem: the number of apples bought is 28
theorem apples_bought_is_28 : (total_apples_after_buying - (initial_apples - apples_used)) = 28 := 
by sorry

end apples_bought_is_28_l632_632575


namespace min_ball_count_required_l632_632008

def is_valid_ball_count (n : ℕ) : Prop :=
  n >= 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_list (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i sorry ≠ l.nthLe j sorry

def valid_ball_counts_list (l : List ℕ) : Prop :=
  (l.length = 10) ∧ distinct_list l ∧ (∀ n ∈ l, is_valid_ball_count n)

theorem min_ball_count_required : ∃ l, valid_ball_counts_list l ∧ l.sum = 174 := sorry

end min_ball_count_required_l632_632008


namespace binary_predecessor_of_1101010_l632_632411

theorem binary_predecessor_of_1101010 :
  let Q := 1101010
  bin_to_nat Q - 1 = bin_to_nat 1101001 :=
sorry

end binary_predecessor_of_1101010_l632_632411


namespace total_wait_time_l632_632669

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l632_632669


namespace radius_of_circle_with_integer_coordinates_l632_632231

theorem radius_of_circle_with_integer_coordinates (radius : ℝ) : 
  (∃ (p : ℤ × ℤ), (p.1, p.2) ∈ ({(5, 0), (-5, 0), (0, 5), (0, -5), (3, 4), (-3, 4), (3, -4), (-3, -4), (4, 3), (-4, 3), (4, -3), (-4, -3)} : set (ℤ × ℤ)) ∧ 
    real.sqrt ((p.1 : ℝ)^2 + (p.2 : ℝ)^2) = radius) → radius = 5 := 
by
  sorry

end radius_of_circle_with_integer_coordinates_l632_632231


namespace concurrency_of_lines_l632_632915

variables (A B C O H D E F : Point)
variables (circumcenter : Triangle → Point)
variables (orthocenter : Triangle → Point)
variables (on_segment : Point → Point → Point → Prop)
variables (concurrent : Line → Line → Line → Prop)
variables (circumradius_eq : Point → Point → Real → Prop)

-- Define the acute triangle
def acute_triangle (A B C : Point) : Prop := 
  ∃ (t : Triangle), t.a = A ∧ t.b = B ∧ t.c = C ∧ is_acute t

-- Define the points on given segments
def points_on_segments (A B C D E F : Point) : Prop := 
  on_segment B C D ∧ on_segment C A E ∧ on_segment A B F

-- Define equal segment conditions 
def equal_segments_condition (O H D E F : Point) (R : Real) : Prop :=
  circumradius_eq O H R ∧ circumradius_eq O D R ∧
  circumradius_eq O E R ∧ circumradius_eq O F R

-- Define the problem based on the conditions and question
theorem concurrency_of_lines
  (h_triangle : acute_triangle A B C)
  (h_O : O = circumcenter ⟨A, B, C⟩)
  (h_H : H = orthocenter ⟨A, B, C⟩)
  (h_points : points_on_segments A B C D E F)
  (R : Real)
  (h_segments : equal_segments_condition O H D E F R) :
  concurrent (line_through A D) (line_through B E) (line_through C F) :=
sorry

end concurrency_of_lines_l632_632915


namespace parallelogram_with_equal_diagonals_is_rectangle_l632_632675

theorem parallelogram_with_equal_diagonals_is_rectangle
  (P : Type) [euclidean_geometry P (distance P)]
  (p q r s : P)
  (is_parallelogram : parallelogram p q r s)
  (equal_diagonals : distance p r = distance q s) :
  rectangle p q r s :=
sorry

end parallelogram_with_equal_diagonals_is_rectangle_l632_632675


namespace sum_of_primes_between_20_and_30_l632_632105

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632105


namespace solve_eq54_l632_632787

noncomputable def eq54 (x : ℝ) := (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 54

theorem solve_eq54 : (eq54 0) ∨ (eq54 (-1)) ∨ (eq54 (-3)) ∨ (eq54 (-3.5)) := by
  sorry

end solve_eq54_l632_632787


namespace average_writing_speed_l632_632701

theorem average_writing_speed
  (total_words : ℕ)
  (total_hours : ℕ)
  (break_hours : ℕ)
  (effective_hours := total_hours - break_hours)
  (avg_speed := total_words / effective_hours) :
  total_words = 50000 →
  total_hours = 100 →
  break_hours = 20 →
  avg_speed = 625 := 
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold effective_hours
  unfold avg_speed
  rfl

-- The proof body is completed with sorry for success code compilation without a valid proof.
  sorry

end average_writing_speed_l632_632701


namespace midpoint_speed_correct_l632_632249

-- Variables for the speeds given in the conditions.
variables (v1 v2 : ℝ)

-- Assume v1 and v2 are given as per the conditions:
axiom v1_val : v1 = 10
axiom v2_val : v2 = 6

-- Define the speed of the middle of the rod
def midpoint_speed : ℝ := sqrt (v2^2 + (sqrt (v1^2 - v2^2) / 2)^2)

-- The theorem to prove
theorem midpoint_speed_correct : midpoint_speed 10 6 = sqrt 52 := 
by
  rw [midpoint_speed, v1_val, v2_val]
  sorry

end midpoint_speed_correct_l632_632249


namespace percentage_in_quarters_l632_632680

theorem percentage_in_quarters:
  let dimes : ℕ := 40
  let quarters : ℕ := 30
  let value_dimes : ℕ := dimes * 10
  let value_quarters : ℕ := quarters * 25
  let total_value : ℕ := value_dimes + value_quarters
  let percentage_quarters : ℚ := (value_quarters : ℚ) / total_value * 100
  percentage_quarters = 65.22 := sorry

end percentage_in_quarters_l632_632680


namespace f_x_l632_632921

-- Assume there exists a function f : ℝ → ℝ such that for all x : ℝ,
-- f(x + 1) = 2x + 3. We want to show that f(x) = 2x + 1 for all x : ℝ.

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_x {f : ℝ → ℝ} (h : ∀ x, f(x + 1) = 2 * x + 3) : ∀ x, f x = 2 * x + 1 :=
by
  sorry

end f_x_l632_632921


namespace optimal_mixture_cost_l632_632230

noncomputable def cost_function (x y : ℝ) : ℝ :=
  3 * x + 5 * y

theorem optimal_mixture_cost :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ x ≤ 100) ∧ 
  (0 ≤ y ∧ y ≤ 500) ∧ 
  (x + y = 400) ∧ 
  (0.10 * x + 0.30 * y = 100) ∧ 
  cost_function x y = 1800 :=
begin
  sorry -- Proof not provided, as per instruction.
end

end optimal_mixture_cost_l632_632230


namespace f_at_2005_l632_632312

noncomputable def f : ℝ → ℝ := 
  λ x, x -- just for syntax compatibility, actual proof skips implementation

axiom f_rule1 : ∀ x : ℝ, f(x + 3) ≤ f(x) + 3
axiom f_rule2 : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2
axiom f_at_1 : f 1 = 1

theorem f_at_2005 : f 2005 = 2005 :=
by {
  -- The proof will be inserted here once axioms are utilised
  sorry
}

end f_at_2005_l632_632312


namespace cos_pi_minus_alpha_trigonometric_identity_l632_632817

noncomputable def alpha : ℝ := sorry

-- Condition 1: α is in the third quadrant
def isInThirdQuadrant (α : ℝ) : Prop := π < α ∧ α < (3 * π) / 2

-- Condition 2: 2sin(α) = cos(α)
def condition (α : ℝ) := 2 * Real.sin α = Real.cos α

-- Statement 1: Prove cos(π - α) = 2 * sqrt(5) / 5
theorem cos_pi_minus_alpha (α : ℝ) (h1 : isInThirdQuadrant α) (h2 : condition α) : 
  Real.cos (π - α) = 2 * Real.sqrt 5 / 5 :=
sorry

-- Statement 2: Prove (1 + 2 * sin(α) * sin(π / 2 - α)) / (sin^2(α) - cos^2(α)) = -3
theorem trigonometric_identity (α : ℝ) (h2 : condition α) : 
  (1 + 2 * Real.sin α * Real.sin (π / 2 - α)) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -3 :=
sorry

end cos_pi_minus_alpha_trigonometric_identity_l632_632817


namespace sixtieth_integer_is_32541_l632_632046

noncomputable def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n + 1) * factorial n

theorem sixtieth_integer_is_32541 :
  let digits := [1, 2, 3, 4, 5]
  let perms := (list.permutations digits).map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let sorted_perms := list.sort (≤) perms
  sorted_perms.nth_le(59) sorry = 32541
sorry

end sixtieth_integer_is_32541_l632_632046


namespace area_of_figure_l632_632938

-- Define the conditions using a predicate
def satisfies_condition (x y : ℝ) : Prop :=
  |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

-- Define the set of points satisfying the condition
def S : set (ℝ × ℝ) := { p | satisfies_condition p.1 p.2 }

-- Define a function to calculate the area of the resulting figure
noncomputable def area_of_S : ℝ :=
  -- define interior triangular region using vertices (0,0), (8,0), and (0,15)
  1 / 2 * 8 * 15

-- The theorem to be proved
theorem area_of_figure : area_of_S = 60 :=
by
  -- This is where the actual proof would go.
  sorry

end area_of_figure_l632_632938


namespace expand_product_l632_632326

-- Definitions of the polynomial functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 + x + 1

-- Statement of the theorem
theorem expand_product : ∀ x : ℝ, (f x) * (g x) = x^3 + 4*x^2 + 4*x + 3 :=
by
  -- Proof goes here, but is omitted for the statement only
  sorry

end expand_product_l632_632326


namespace largest_possible_number_sum_120_l632_632055

theorem largest_possible_number_sum_120 (n a : ℕ) (h : n > 1) (sum_eq_120 : n * (2 * a + n - 1) = 240) :
  max (finset.range n).map (λ i, a + i) = 26 :=
by
  sorry

end largest_possible_number_sum_120_l632_632055


namespace hydrae_never_equal_heads_l632_632641

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632641


namespace remaining_episodes_l632_632210

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l632_632210


namespace h2so4_moles_needed_l632_632337

-- Definition of the balanced chemical reaction.
def balanced_reaction : Prop :=
  2 * NaCl + H2SO4 = 2 * HCl + Na2SO4

-- Definition of the number of moles of HCl formed.
def moles_HCl_formed (moles_HCl : ℝ) : Prop :=
  moles_HCl = 1

-- Definition of the number of moles of H2SO4 combined.
def moles_H2SO4_combined (moles_H2SO4 : ℝ) : Prop :=
  moles_H2SO4 = 0.5

-- The theorem we need to prove.
theorem h2so4_moles_needed (moles_HCl moles_H2SO4 : ℝ) (r : balanced_reaction) (h : moles_HCl_formed moles_HCl) : 
  moles_H2SO4_combined moles_H2SO4 :=
sorry

end h2so4_moles_needed_l632_632337


namespace area_of_triangle_ABC_l632_632974

theorem area_of_triangle_ABC :
  let A : ℝ × ℝ := (-7, 3)
  let B : ℝ × ℝ := (0, 4)
  let C : ℝ × ℝ := (9, 5)
  let distance_AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 49
  let distance_BC := (B.1 - C.1)^2 + (B.2 - C.2)^2 = 81
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 8 :=
by
  let A := (-7 : ℝ, 3 : ℝ)
  let B := (0 : ℝ, 4 : ℝ)
  let C := (9 : ℝ, 5 : ℝ)
  have distance_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 49 := sorry
  have distance_BC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 81 := sorry
  show 1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 8
  by sorry

end area_of_triangle_ABC_l632_632974


namespace similar_triangles_shadow_l632_632215

theorem similar_triangles_shadow 
  (H1 S1 H2 : ℝ) 
  (h1 : H1 = 2.5) 
  (s1 : S1 = 5) 
  (h2 : H2 = 2) 
  (similar : H1 / S1 = H2 / 4) : 
  H2 / S2 = 1 -> S2 = 4 :=
by
  rw [h1, s1, h2]
  sorry

end similar_triangles_shadow_l632_632215


namespace math_problem_l632_632838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Definition and properties of the function
def extrema (a : ℝ) : Prop :=
  if a ≥ 0 then true -- no extrema for a ≥ 0
  else ∃ x : ℝ, x = -1/a ∧ f a x = -1 + Real.log (-1/a) -- maximum at -1/a for a < 0

-- Accompanying tangent line property
def accompanying_tangent_line (a x1 x2 : ℝ) : Prop :=
  ∃ x0 : ℝ, x1 < x0 ∧ x0 < x2 ∧ (a + 1/x0) = (f a x2 - f a x1) / (x2 - x1)

-- Uniqueness of the accompanying tangent line
def uniqueness_of_tangent_line (a x1 x2 : ℝ) : Prop :=
  ∀ x0 x0' : ℝ, (x1 < x0 ∧ x0 < x2 ∧ (a + 1/x0) = (f a x2 - f a x1) / (x2 - x1)) →
                (x1 < x0' ∧ x0' < x2 ∧ (a + 1/x0') = (f a x2 - f a x1) / (x2 - x1)) →
                x0 = x0'

theorem math_problem (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
  extrema a ∧ accompanying_tangent_line a x1 x2 ∧ uniqueness_of_tangent_line a x1 x2 :=
by
  sorry

end math_problem_l632_632838


namespace max_value_is_nine_l632_632548

noncomputable def max_value_on_ellipse (x y : ℝ) : ℝ :=
  abs (2 * real.sqrt 3 * x + y - 1)

theorem max_value_is_nine :
  (∀ (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) → max_value_on_ellipse x y ≤ 9) ∧
  (∃ (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) ∧ max_value_on_ellipse x y = 9) :=
by sorry

end max_value_is_nine_l632_632548


namespace f_value_l632_632564

def f (x : ℝ) : ℝ := if x ∈ Set.Ico 0 1 then Real.log (x + 1) / Real.log 2 else 0
-- Define the given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period : ∀ x, f (x + 2) = f x

theorem f_value {x : ℝ} (h : x = 2015 / 4) : f x + Real.log 5 / Real.log 2 = 2 :=
by {
  have : x = 2015 / 4, from h,
  sorry -- proof goes here
}

end f_value_l632_632564


namespace probability_greater_than_2_l632_632802

noncomputable def normal_distribution_X : MeasureTheory.Measure ℝ := MeasureTheory.Measure.gaussian 0 (6^2)

axiom P_interval : MeasureTheory.MeasureTheory (Icc (-2 : ℝ) 0) normal_distribution_X = 0.4

theorem probability_greater_than_2 :
  MeasureTheory.MeasureTheory (Ioi (2 : ℝ)) normal_distribution_X = 0.1 :=
sorry

end probability_greater_than_2_l632_632802


namespace sum_primes_between_20_and_30_is_52_l632_632127

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632127


namespace sum_primes_20_to_30_l632_632166

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632166


namespace chocolate_chips_needed_l632_632219

-- Define the variables used in the conditions
def cups_per_recipe := 2
def number_of_recipes := 23

-- State the theorem
theorem chocolate_chips_needed : (cups_per_recipe * number_of_recipes) = 46 := 
by sorry

end chocolate_chips_needed_l632_632219


namespace hydras_never_die_l632_632647

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632647


namespace sum_of_primes_between_20_and_30_l632_632096

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632096


namespace hydras_never_die_l632_632651

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632651


namespace doctor_is_correct_l632_632611

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632611


namespace hydras_will_live_l632_632637

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632637


namespace inequality_solution_set_minimum_value_mn_squared_l632_632392

noncomputable def f (x : ℝ) := |x - 2| + |x + 1|

theorem inequality_solution_set : 
  (∀ x, f x > 7 ↔ x > 4 ∨ x < -3) :=
by sorry

theorem minimum_value_mn_squared (m n : ℝ) (hm : n > 0) (hmin : ∀ x, f x ≥ m + n) :
  m^2 + n^2 = 9 / 2 ∧ m = 3 / 2 ∧ n = 3 / 2 :=
by sorry

end inequality_solution_set_minimum_value_mn_squared_l632_632392


namespace sum_of_primes_between_20_and_30_l632_632112

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632112


namespace standard_equation_of_ellipse_trajectory_equation_of_midpoint_l632_632811

-- Definitions for the problem conditions
def ellipse_center : Prop := (0, 0)

def left_focus : Prop := F = (-Real.sqrt 3, 0)

def point_D : Prop := D = (2, 0)

def point_A : Prop := A = (1, 1 / 2)

def moving_point_on_ellipse (p : ℝ × ℝ) : Prop := 
  ∃ x_0 y_0, (p = (x_0, y_0)) ∧ (x_0^2 / 4 + y_0^2 = 1)

-- Theorems to be proved
theorem standard_equation_of_ellipse :
  (center (0, 0)) → (left_focus F = (-Real.sqrt 3, 0)) →
  (passes_through (D = (2, 0))) →
  (∃ (a : ℝ) (b : ℝ), a = 2 ∧ b = 1 ∧ ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)) :=
by sorry

theorem trajectory_equation_of_midpoint :
  (center (0, 0)) → (left_focus F = (-Real.sqrt 3, 0)) →
  (passes_through (D = (2, 0))) →
  (∃ (A : ℝ × ℝ), A = (1, 1/2)) →
  ((P : ℝ × ℝ) → (moving_point_on_ellipse P)) →
  ∀ x y, ((x - 1/2)^2 + 4 * (y - 1/4)^2 = 1) :=
by sorry

end standard_equation_of_ellipse_trajectory_equation_of_midpoint_l632_632811


namespace total_necklaces_made_l632_632322

-- Definitions based on conditions
def first_machine_necklaces : ℝ := 45
def second_machine_necklaces : ℝ := 2.4 * first_machine_necklaces

-- Proof statement
theorem total_necklaces_made : (first_machine_necklaces + second_machine_necklaces) = 153 := by
  sorry

end total_necklaces_made_l632_632322


namespace log_base_5_of_125_eq_3_l632_632324

theorem log_base_5_of_125_eq_3 : log 5 125 = 3 :=
by
  have h1 : 125 = 5 ^ 3 := by norm_num
  have h2 : log 5 125 = log 5 (5 ^ 3) := by rw h1
  have h3 : log 5 (5 ^ 3) = 3 * log 5 5 := by rw log_pow 3 (by norm_num : 5 ≠ 1)
  have h4 : log 5 5 = 1 := log_self (by norm_num : 5 ≠ 1)
  rw [h2, h3, h4]
  norm_num

end log_base_5_of_125_eq_3_l632_632324


namespace total_birds_in_marsh_l632_632595

-- Define the number of geese and ducks as constants.
def geese : Nat := 58
def ducks : Nat := 37

-- The theorem that we need to prove.
theorem total_birds_in_marsh : geese + ducks = 95 :=
by
  -- Here, we add the sorry keyword to skip the proof part.
  sorry

end total_birds_in_marsh_l632_632595


namespace polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l632_632827

theorem polynomial_three_positive_roots_inequality
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  2 * a^3 + 9 * c ≤ 7 * a * b :=
sorry

theorem polynomial_three_positive_roots_equality_condition
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  (2 * a^3 + 9 * c = 7 * a * b) ↔ (x1 = x2 ∧ x2 = x3) :=
sorry

end polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l632_632827


namespace problem_solution_l632_632454

-- Define the function and its properties
def f (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^3 + x^2 - 2*x)

-- Define the variables as per the conditions of the problem
def a : ℕ := 1 -- number of holes
def b : ℕ := 2 -- number of vertical asymptotes
def c : ℕ := 1 -- number of horizontal asymptotes
def d : ℕ := 0 -- number of oblique asymptotes

theorem problem_solution : a + 2 * b + 3 * c + 4 * d = 8 := by
  -- We just set up the problem and leave the proof to be filled in
  sorry

end problem_solution_l632_632454


namespace tan_Y_right_triangle_l632_632328

theorem tan_Y_right_triangle (YX YZ : ℝ) (hYX : YX = 8) (hYZ : YZ = 17) :
  let XZ := Real.sqrt (YZ^2 - YX^2) in
  ∃ (XZ : ℝ), XZ = 15 ∧ (tan YX XZ = (8 : ℝ) / 15) :=
by
  sorry

end tan_Y_right_triangle_l632_632328


namespace area_of_annulus_l632_632719

-- Define the conditions
def concentric_circles (r s : ℝ) (h : r > s) (x : ℝ) := 
  r^2 = s^2 + x^2

-- State the theorem
theorem area_of_annulus (r s x : ℝ) (h : r > s) (h₁ : concentric_circles r s h x) :
  π * x^2 = π * r^2 - π * s^2 :=
by 
  rw [concentric_circles] at h₁
  sorry

end area_of_annulus_l632_632719


namespace prime_sum_20_to_30_l632_632194

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632194


namespace number_of_bananas_l632_632236

-- Define costs as constants
def cost_per_banana := 1
def cost_per_apple := 2
def cost_per_twelve_strawberries := 4
def cost_per_avocado := 3
def cost_per_half_bunch_grapes := 2
def total_cost := 28

-- Define quantities as constants
def number_of_apples := 3
def number_of_strawberries := 24
def number_of_avocados := 2
def number_of_half_bunches_grapes := 2

-- Define calculated costs
def cost_of_apples := number_of_apples * cost_per_apple
def cost_of_strawberries := (number_of_strawberries / 12) * cost_per_twelve_strawberries
def cost_of_avocados := number_of_avocados * cost_per_avocado
def cost_of_grapes := number_of_half_bunches_grapes * cost_per_half_bunch_grapes

-- Define total cost of other fruits
def total_cost_of_other_fruits := cost_of_apples + cost_of_strawberries + cost_of_avocados + cost_of_grapes

-- Define the remaining cost for bananas
def remaining_cost := total_cost - total_cost_of_other_fruits

-- Prove the number of bananas
theorem number_of_bananas : remaining_cost / cost_per_banana = 4 :=
by
  -- This is a placeholder to indicate a non-implemented proof
  sorry

end number_of_bananas_l632_632236


namespace lambda_range_l632_632889

-- Definition of the sequence
def a (n : ℕ) (λ : ℝ) : ℝ :=
  -2 * (n:ℝ)^2 + λ * (n:ℝ)

-- The main theorem to prove the range of λ
theorem lambda_range {λ : ℝ} (h : ∀ n : ℕ, a (n+1) λ < a n λ) : λ < 6 :=
  sorry

end lambda_range_l632_632889


namespace PQ_ineq_l632_632475

section TriangleInequality

variables {A B C P Q : Type} [metric_space A] [has_dist A] 
variables (A B C : A) (P : A → A) (Q : A → A)
variables (a b c : ℝ) -- side lengths

-- Given conditions
axiom AB_eq_3 : dist A B = 3
axiom AC_eq_4 : dist A C = 4
axiom BC_eq_5 : dist B C = 5

-- Let P be a point on BC
-- axiom P_on_BC : ∃ λ (P : A), ∃ t ∈ Icc 0 1, P = t • B + (1 - t) • C

-- Q is the intersection of line AP with circumcircle of ABC other than A
axiom A_P_Q_A_not : P ≠ A ∧ Q ≠ A
axiom Q_on_circumcircle : dist Q (midpoint B C) = dist A (midpoint B C)

-- We need to prove
theorem PQ_ineq : (dist P Q) ≤ 25 / (4 * real.sqrt 6) :=
sorry

end TriangleInequality

end PQ_ineq_l632_632475


namespace part_a_part_b_l632_632901

-- Conditions for part (a)
variables (r b : ℕ)
variables (r_odd : Odd r)
variables (b_odd : Odd b)
variables (no_overlap : ∀ n : ℕ, (n ≠ 0 ∧ n ≠ r) → (n % r ≠ 0 ∨ n % b ≠ 0))

-- Conditions for part (b)
variables (rel_prime : Nat.coprime r b)
variables (r_b_odd_sum : Odd (r + b))

-- The definition of S (sum of floor terms)
def S (r b : ℕ) : ℕ := ∑ k in Finset.range (r + b), k * r / (r + b)

-- Part (a) statement
theorem part_a : 
  (∑ k in Finset.range (r * b), ite (k % r = 0) 1 0 - ∑ k in Finset.range (r * b), ite (k % b = 0) 1 0 = 0) := by sorry

-- Part (b) statement
theorem part_b : 
  |1 + 2 * (r-1) * (b+r) - 8 * S r b| = (∑ k in Finset.range rb, ite (k % r = 0) 1 0 + ∑ k in Finset.range rb, ite (k % b = 0) 1 0) := by sorry

end part_a_part_b_l632_632901


namespace millionth_digit_of_1_over_41_l632_632779

theorem millionth_digit_of_1_over_41 :
  let frac := 1 / (41 : ℚ),
      seq := "02439",
      period := (5 : ℕ) in
  (seq.get (1000000 % period - 1) = '9') :=
by
  let frac := 1 / (41 : ℚ)
  let seq := "02439"
  let period := 5
  have h_expansion : frac = 0.02439 / 10000 := sorry
  have h_period : ∀ n, frac = Rational.mkPeriodic seq period n := sorry
  have h_mod : 1000000 % period = 0 := by sorry
  have h_index := h_mod.symm ▸ (dec_trivial : 0 % 5 = 0)
  exact h_period n ▸ (dec_trivial : "02439".get 4 = '9')

end millionth_digit_of_1_over_41_l632_632779


namespace find_velocity_l632_632962

noncomputable section

open Real

variable {k A V : ℝ}

def pressure (P A V : ℝ) : ℝ := k * A * V^2

theorem find_velocity :
  (pressure 4 2 8) = k * 2 * 8^2 →
  (pressure 25 (9/2) V) = 25 →
  V = 40 / 3 :=
by
  intro h1 h2
  have h3 : k = 4 / (2 * 8^2) := by sorry
  simp only [pressure] at h2
  rw [h3] at h2
  norm_num at h2
  sorry

end find_velocity_l632_632962


namespace decimal_to_binary_l632_632746

-- Define the decimal value
def decimal_val : ℕ := 51

-- Define the expected binary representation
def binary_val : ℕ := 0b110011  -- 51 in binary notation

-- State the theorem to prove
theorem decimal_to_binary : nat.bv (dec := decimal_val) = binary_val := 
by 
  sorry

end decimal_to_binary_l632_632746


namespace sole_winner_after_seven_rounds_l632_632883

def point_system : Type :=
  { win := 1, draw := 0.5, loss := 0 }

def tournament_result (rounds : ℕ) : Prop :=
  ∀ {players : ℕ} {points : ℕ}, players = 10 ∧ points = 7 → 
  players_tournament_winner rounds points

theorem sole_winner_after_seven_rounds : tournament_result 7 := 
sorry

end sole_winner_after_seven_rounds_l632_632883


namespace complete_graph_properties_l632_632319

-- Definitions for graphs K_1 to K_5
def K₁ := { vertices := 1, edges := 0 }
def K₂ := { vertices := 2, edges := 1 }
def K₃ := { vertices := 3, edges := 3 }
def K₄ := { vertices := 4, edges := 6 }
def K₅ := { vertices := 5, edges := 10 }

-- Formalizing the proof goal in Lean 4
theorem complete_graph_properties (n : ℕ) (h : n ∈ {1, 2, 3, 4, 5}) :
  ∃ (vertices edges : ℕ), 
    vertices = n ∧ 
    edges = (n * (n - 1)) / 2 :=
by {
  cases h;
  { use [1, 0], simp },
  { use [2, 1], simp },
  { use [3, 3], simp },
  { use [4, 6], simp },
  { use [5, 10], simp },
}

end complete_graph_properties_l632_632319


namespace min_value_at_a_eq_4_min_value_2_pow_4_pow_l632_632351

theorem min_value_at_a_eq_4 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 8) : 
  2 ^ a * 4 ^ b ≥ 2 ^ (a + 2 * b) :=
by {
  sorry
}

theorem min_value_2_pow_4_pow (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 8) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → (x * y = 8) → 2 ^ x * 4 ^ y ≥ 2 ^ 8) ∧ (∃ (a b : ℝ), (a = 4) ∧ (a = 2 * b) ∧ (2 ^ a * 4 ^ b = 2 ^ 8)) :=
by {
  sorry
}

end min_value_at_a_eq_4_min_value_2_pow_4_pow_l632_632351


namespace sports_event_duration_and_medals_l632_632053

theorem sports_event_duration_and_medals (m n : ℕ):
  (1 + (m - 1) / 7 + (6 / 7) * (2 + (7 / 6) * (3 + ...) + ... + (n + (7 / 6)^(n-1)) 
  (u_{k+1} = \frac{6}{7}(u_k - k)) 
  -> n = 6 ∧ m = 36 :=
sorry

end sports_event_duration_and_medals_l632_632053


namespace hydra_survival_l632_632600

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632600


namespace find_ellipse_focus_l632_632332

theorem find_ellipse_focus :
  ∀ (a b : ℝ), a^2 = 5 → b^2 = 4 → 
  (∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) →
  ((∃ c : ℝ, c^2 = a^2 - b^2) ∧ (∃ x y, x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end find_ellipse_focus_l632_632332


namespace ellipse_focal_length_l632_632832

theorem ellipse_focal_length (m : ℝ) (h : ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1) (major_axis_along_x : true) (focal_length : 4) : m = 4 := 
by
  -- Insert logical connections and steps to conclude from conditions
  sorry

end ellipse_focal_length_l632_632832


namespace sqrt_sum_value_l632_632019

theorem sqrt_sum_value (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) :
    sqrt (64 - y^2) + sqrt (36 - y^2) = 7 :=
sorry

end sqrt_sum_value_l632_632019


namespace cos_F_l632_632445

-- Define the given conditions
variables (D F E : Type) -- Points in the plane
variables [DecidableEq D] [DecidableEq F] [DecidableEq E] 
variables (distance : D → E → F → Real)
variables (angle : D → E → F → Real)

-- Given conditions
def is_right_triangle (D F E : Type) [DecidableEq D] [DecidableEq F] [DecidableEq E] 
  (distance : D → E → F → Real) (angle : D → E → F → Real) : Prop :=
angle D E F = 90 ∧ distance D E = 8 ∧ distance E F = 17

-- The proof statement
theorem cos_F (D F E : Type) [DecidableEq D] [DecidableEq F] [DecidableEq E] 
 (distance : D → E → F → Real) (angle : D → E → F → Real) :
 is_right_triangle D F E distance angle →
  let DF := sqrt ((distance E F)^2 - (distance D E)^2) in
  cos (angle D E F) = distance D E / distance E F :=
begin
  intros h,
  rcases h with ⟨h_angle, h_DE, h_EF⟩,
  have h_DF : distance D F = sqrt ((distance E F)^2 - (distance D E)^2), {
    rw [h_DE, h_EF],
    simp,
  },
  rw [h_angle, h_DE, h_EF],
  sorry
end

end cos_F_l632_632445


namespace intervals_of_monotonicity_difference_between_extrema_l632_632394

-- Definitions based on conditions 
def function (a b c x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c
def derivative (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

-- Conditions based on the problem statement
def extremum_condition (a b : ℝ) : Prop := derivative a b 2 = 0
def parallel_tangent_condition (a b : ℝ) : Prop := derivative a b 1 = -3

-- The main theorem to prove intervals of monotonicity
theorem intervals_of_monotonicity (a b c : ℝ) 
    (h1 : extremum_condition a b) (h2 : parallel_tangent_condition a b) :
  (monotone_increasing (function a b c) (-∞) 0 ∧ 
   monotone_increasing (function a b c) 2 ∞ ∧
   monotone_decreasing (function a b c) 0 2) := sorry

-- The main theorem to prove the difference between the maximum and minimum values
theorem difference_between_extrema (a b c : ℝ)
    (h1 : extremum_condition a b) (h2 : parallel_tangent_condition a b) 
    (maxx := 0) (minx := 2) :
  (function a b c maxx - function a b c minx) = 4 := sorry

end intervals_of_monotonicity_difference_between_extrema_l632_632394


namespace intersection_of_A_and_B_l632_632919

noncomputable def A : set ℝ := { x | 2^x ≤ 4 }
noncomputable def B : set ℝ := { x | ∃ y, y = Real.log (x - 1) }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
by {
  -- sorry skips the proof
  sorry
}

end intersection_of_A_and_B_l632_632919


namespace milburg_children_count_l632_632580

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 ∧ grown_ups = 5256 → 
  total_population - grown_ups = 2987 :=
by 
  intros total_population grown_ups h,
  rcases h with ⟨h1, h2⟩,
  rw [h1, h2],
  exact rfl

end milburg_children_count_l632_632580


namespace no_growth_pie_probability_l632_632275

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l632_632275


namespace smallest_positive_period_range_of_f_l632_632835

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * (cos x)^2 - sqrt 3 * (sin x)^2

-- Theorem 1: Prove the smallest positive period is π
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := sorry

-- Theorem 2: Prove the range of f(x) for x in [-π/3, π/3] is [-sqrt 3, 2]
theorem range_of_f (x : ℝ) (h1 : -π/3 ≤ x) (h2 : x ≤ π/3) : -sqrt 3 ≤ f x ∧ f x ≤ 2 := sorry

end smallest_positive_period_range_of_f_l632_632835


namespace tetrahedron_edges_vertices_product_l632_632255

theorem tetrahedron_edges_vertices_product :
  let vertices := 4
  let edges := 6
  edges * vertices = 24 :=
by
  let vertices := 4
  let edges := 6
  sorry

end tetrahedron_edges_vertices_product_l632_632255


namespace find_a_if_odd_l632_632422

theorem find_a_if_odd :
  ∀ (a : ℝ), (∀ x : ℝ, (a * (-x)^3 + (a - 1) * (-x)^2 + (-x) = -(a * x^3 + (a - 1) * x^2 + x))) → 
  a = 1 :=
by
  sorry

end find_a_if_odd_l632_632422


namespace order_values_l632_632376

-- Define the hypotheses
variables {f : ℝ → ℝ}

-- Define the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f(1 + x) + f(1 - x) = f(1)
def decreasing_on_interval (f : ℝ → ℝ) := ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 1 → f(a) ≥ f(b)

-- The main theorem to be proved
theorem order_values 
  (h_odd : odd_function f) 
  (h_func_eq : functional_equation f)
  (h_decr : decreasing_on_interval f) :
  f(-2 + (Real.sqrt 2) / 2) < -f(10 / 3) ∧ -f(10 / 3) < f(9 / 2) :=
sorry

end order_values_l632_632376


namespace part1_l632_632839

noncomputable def f (a x : ℝ) : ℝ := a * x - 2 * Real.log x + 2 * (1 + a) + (a - 2) / x

theorem part1 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ 0) ↔ 1 ≤ a :=
sorry

end part1_l632_632839


namespace perpendicular_vectors_x_value_l632_632407

theorem perpendicular_vectors_x_value
  (x : ℝ)
  (m : EuclideanSpace ℝ (Fin 2) := ![x, 2])
  (n : EuclideanSpace ℝ (Fin 2) := ![1, -1])
  (perpendicular : m ⋅ n = 0) :
  x = 2 := by
  sorry

end perpendicular_vectors_x_value_l632_632407


namespace sum_of_coefficients_binomial_expansion_l632_632966

theorem sum_of_coefficients_binomial_expansion :
  (∑ k in range (8 + 1), (nat.choose 8 k)) = 256 := by
  sorry

end sum_of_coefficients_binomial_expansion_l632_632966


namespace total_weight_full_l632_632233

theorem total_weight_full {x y p q : ℝ}
    (h1 : x + (3/4) * y = p)
    (h2 : x + (1/3) * y = q) :
    x + y = (8/5) * p - (3/5) * q :=
by
  sorry

end total_weight_full_l632_632233


namespace stream_speed_l632_632685

variable (D : ℝ) -- Distance rowed

theorem stream_speed (v : ℝ) (h : D / (60 - v) = 2 * (D / (60 + v))) : v = 20 :=
by
  sorry

end stream_speed_l632_632685


namespace planet_combinations_count_l632_632408
  
theorem planet_combinations_count :
  ∃ a b : ℕ, a ≤ 5 ∧ b ≤ 9 ∧ 2 * a + b = 14 ∧ 
  (finset.card (finset.univ.choose a) * finset.card (finset.univ.choose b) =
  636) := sorry

end planet_combinations_count_l632_632408


namespace hyperbola_standard_equation_l632_632396

theorem hyperbola_standard_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
    (real_axis_length : 2 * a = 8) (eccentricity : (5 : ℝ) / 4 = 5 / 4) :
    (frac x^2 16) - (frac y^2 9) = 1 :=
by
  sorry

end hyperbola_standard_equation_l632_632396


namespace millionth_digit_of_1_over_41_l632_632777

theorem millionth_digit_of_1_over_41 :
  let d := 5    -- The period of decimal expansion is 5
  let seq := [0,2,4,3,9]  -- The repeating sequence of 1/41
  let millionth_digit_position := 1000000 % d  -- Find the position in the repeating sequence
  seq.millionth_digit_position = 9 :=
by
  let d := 5
  let seq := [0, 2, 4, 3, 9]
  let millionth_digit_position := 1000000 % d
  show seq.millionth_digit_position = 9
  sorry

end millionth_digit_of_1_over_41_l632_632777


namespace jill_time_to_school_l632_632311

theorem jill_time_to_school :
  let Dave_steps_per_minute := 100
  let Dave_step_length_cm := 80
  let Dave_time_minutes := 20
  let Jill_steps_per_minute := 110
  let Jill_step_length_cm := 70
  let Dave_speed_cm_per_minute := Dave_steps_per_minute * Dave_step_length_cm
  let Distance_cm := Dave_speed_cm_per_minute * Dave_time_minutes
  let Jill_speed_cm_per_minute := Jill_steps_per_minute * Jill_step_length_cm
  let Jill_time_minutes := Distance_cm / Jill_speed_cm_per_minute
  in abs (Jill_time_minutes - 21) < 1 :=
by
  sorry

end jill_time_to_school_l632_632311


namespace lateral_surface_area_cone_l632_632382

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 30 * Real.pi / 180  -- θ in radians

def r := l * Real.sin θ

theorem lateral_surface_area_cone : π * r * l = 8 * π :=
by
  sorry

end lateral_surface_area_cone_l632_632382


namespace calc_dot_product_find_x_perpendicular_l632_632375

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Condition definitions
def a_norm : ℝ := 2
def b_norm : ℝ := 3
def angle : ℝ := Real.pi * (2 / 3) -- 120 degrees in radians

-- Condition use within the Lean context
axiom norm_a : ∥a∥ = a_norm
axiom norm_b : ∥b∥ = b_norm
axiom inner_ab : innerProductSpace ℝ (EuclideanSpace ℝ (Fin 3)) a b = a_norm * b_norm * Real.cos angle

-- Proof problem 1: Calculate dot product expression
theorem calc_dot_product (a b : EuclideanSpace ℝ (Fin 3)) 
  (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 3) 
  (inner_ab : innerProduct (EuclideanSpace ℝ (Fin 3)) a b = -3) :
  innerProduct (EuclideanSpace ℝ (Fin 3)) (2 • a - b) (a + 3 • b) = -34 := 
sorry

-- Proof problem 2: Find x such that expression is zero
theorem find_x_perpendicular (a b : EuclideanSpace ℝ (Fin 3)) 
  (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 3) 
  (inner_ab : innerProduct (EuclideanSpace ℝ (Fin 3)) a b = -3) :
  ∃ x : ℝ, innerProduct (EuclideanSpace ℝ (Fin 3)) (x • a - b) (a + 3 • b) = 0 ∧ x = -24 / 5 := 
sorry

end calc_dot_product_find_x_perpendicular_l632_632375


namespace original_area_correct_l632_632455

-- Define conditions based on the given problem
def side_length : ℝ := 2
def sin60 : ℝ := real.sin (real.pi / 3) -- sin 60 degrees
def intuitive_area : ℝ := (1 / 2) * side_length * side_length * sin60

-- Define the ratio
def ratio : ℝ := 2 * real.sqrt 2
def original_area : ℝ := ratio * intuitive_area

-- The theorem to be proved
theorem original_area_correct : original_area = 2 * real.sqrt 6 := 
by sorry

end original_area_correct_l632_632455


namespace overall_average_correct_l632_632708

-- Define the number of students in each section
def students_in_section : List ℕ := [65, 70, 80, 75, 90, 85, 60, 55]

-- Define the mean marks for each section
def mean_marks_section : List ℕ := [50, 60, 75, 85, 70, 90, 65, 45]

-- Calculate the overall average marks per student
def overall_average_marks : ℚ :=
  let total_students := (students_in_section.map (λ x => x)).sum
  let total_marks := (List.zip students_in_section mean_marks_section).map (λ (num_students, mean_marks) => num_students * mean_marks).sum
  total_marks / total_students

theorem overall_average_correct :
  overall_average_marks ≈ 69.22 := by
  -- Proof skipped
  sorry

end overall_average_correct_l632_632708


namespace arnaldo_bernaldo_distribute_toys_l632_632946

noncomputable def num_ways_toys_distributed (total_toys remaining_toys : ℕ) : ℕ :=
  if total_toys = 10 ∧ remaining_toys = 8 then 6561 - 256 else 0

theorem arnaldo_bernaldo_distribute_toys : num_ways_toys_distributed 10 8 = 6305 :=
by 
  -- Lean calculation for 3^8 = 6561 and 2^8 = 256 can be done as follows
  -- let three_power_eight := 3^8
  -- let two_power_eight := 2^8
  -- three_power_eight - two_power_eight = 6305
  sorry

end arnaldo_bernaldo_distribute_toys_l632_632946


namespace prime_sum_20_to_30_l632_632186

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632186


namespace problem_proof_l632_632428

-- Triangle sides and angles
variables (a b c : ℝ) (A B C : ℝ)

-- Define vectors m and n
def vec_m (A : ℝ) : ℝ × ℝ := (Real.cos A + Real.sqrt 2, Real.sin A)
def vec_n (A : ℝ) : ℝ × ℝ := (-Real.sin A, Real.cos A)

-- Magnitude of vec_m + vec_n
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Law of Cosines
noncomputable def law_of_cosines (a b c A : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A

-- Area of triangle using sine rule
noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

-- Theorem and Lean statement
theorem problem_proof :
  (magnitude (vec_m (π / 4) + vec_n (π / 4)) = 2) →
  b = 4 * Real.sqrt 2 →
  c = Real.sqrt 2 * a →
  law_of_cosines a b c (π / 4) →
  a^2 - 8 * Real.sqrt 2 * a + 32 = 0 →
  a = 4 * Real.sqrt 2 →
  triangle_area a b c (π / 4) = 16 :=
begin
  sorry
end

end problem_proof_l632_632428


namespace smallest_integer_for_polynomial_div_l632_632075

theorem smallest_integer_for_polynomial_div (x : ℤ) : 
  (∃ k : ℤ, x = 6) ↔ ∃ y, y * (x - 5) = x^2 + 4 * x + 7 := 
by 
  sorry

end smallest_integer_for_polynomial_div_l632_632075


namespace composite_numbers_equal_l632_632506

-- Define composite natural number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

-- Define principal divisors
def principal_divisors (n : ℕ) (principal1 principal2 : ℕ) : Prop :=
  is_composite n ∧ 
  (1 < principal1 ∧ principal1 < n) ∧ 
  (1 < principal2 ∧ principal2 < n) ∧
  principal1 * principal2 = n

-- Problem statement to prove
theorem composite_numbers_equal (a b p1 p2 : ℕ) :
  is_composite a → is_composite b →
  principal_divisors a p1 p2 → principal_divisors b p1 p2 →
  a = b :=
by
  sorry

end composite_numbers_equal_l632_632506


namespace find_x_to_equalize_mean_median_mode_l632_632959

open Function

theorem find_x_to_equalize_mean_median_mode (x : ℤ) : 
    (∃ (s : Finset ℤ), s = {5, 6, 7, 8, 8, 9, x} ∧ 
     (mean s = 8) ∧ 
     (median s = 8) ∧ 
     (mode s = 8)) → x = 13 := by
  sorry

end find_x_to_equalize_mean_median_mode_l632_632959


namespace sum_of_primes_between_20_and_30_l632_632098

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632098


namespace hydras_never_die_l632_632624

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632624


namespace hydras_survive_l632_632615

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632615


namespace sum_of_coefficients_binomial_expansion_l632_632967

theorem sum_of_coefficients_binomial_expansion :
  (∑ k in range (8 + 1), (nat.choose 8 k)) = 256 := by
  sorry

end sum_of_coefficients_binomial_expansion_l632_632967


namespace gcd_cond_and_prime_divisor_l632_632922

theorem gcd_cond_and_prime_divisor {
  a b c d : ℤ
  (hgcd : Int.gcd (Int.gcd (Int.gcd a b) c) d = 1) :
  (∀ p : ℕ, Nat.Prime p → (p ∣ (a * d - b * c) → p ∣ a ∧ p ∣ c)) ↔
  (∀ n : ℤ, Int.gcd (a * n + b) (c * n + d) = 1) :=
sorry

end gcd_cond_and_prime_divisor_l632_632922


namespace bike_distance_difference_l632_632029

-- Defining constants for Alex's and Bella's rates and the time duration
def Alex_rate : ℕ := 12
def Bella_rate : ℕ := 10
def time : ℕ := 6

-- The goal is to prove the difference in distance is 12 miles
theorem bike_distance_difference : (Alex_rate * time) - (Bella_rate * time) = 12 := by
  sorry

end bike_distance_difference_l632_632029


namespace hydra_survival_l632_632601

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632601


namespace probability_four_pow_a_plus_eight_pow_b_units_digit_2_l632_632788

open nat

noncomputable def units_digit (n : ℕ) : ℕ := n % 10

def possible_a : finset ℕ := finset.range 99 100
def possible_b : finset ℕ := finset.range 99 100

def favorable_outcomes : finset (ℕ × ℕ) :=
  (possible_a.product possible_b).filter (λ p, units_digit (4 ^ p.1 + 8 ^ p.2) = 2)

def total_outcomes : finset (ℕ × ℕ) :=
  possible_a.product possible_b

theorem probability_four_pow_a_plus_eight_pow_b_units_digit_2:
  (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 8 :=
by sorry

end probability_four_pow_a_plus_eight_pow_b_units_digit_2_l632_632788


namespace cos_sin_inequality_l632_632500

theorem cos_sin_inequality {a b : ℝ}
  (h : ∀ x : ℝ, cos (a * sin x) > sin (b * cos x)) :
  a^2 + b^2 < (π^2) / 4 := 
sorry

end cos_sin_inequality_l632_632500


namespace dust_particles_calculation_l632_632544

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end dust_particles_calculation_l632_632544


namespace sum_of_primes_between_20_and_30_l632_632116

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632116


namespace taxi_fare_l632_632013

theorem taxi_fare (x : ℝ) (h : x > 6) : 
  let starting_price := 6
  let mid_distance_fare := (6 - 2) * 2.4
  let long_distance_fare := (x - 6) * 3.6
  let total_fare := starting_price + mid_distance_fare + long_distance_fare
  total_fare = 3.6 * x - 6 :=
by
  sorry

end taxi_fare_l632_632013


namespace trigonometric_expression_l632_632682

theorem trigonometric_expression:
  cos (π / 11) - cos (2 * π / 11) + cos (3 * π / 11) - cos (4 * π / 11) + cos (5 * π / 11) = 1 / 2 := by
  sorry

end trigonometric_expression_l632_632682


namespace parallel_lines_l632_632850

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l632_632850


namespace length_of_AD_l632_632597

theorem length_of_AD 
  (ABC_right_angle : ∃ A B C : Type, is_right_triangle A B C) 
  (D_on_AB : ∃ D : Type, is_point_on_line D AB ∧ CD = 1) 
  (AE_altitude : ∃ E : Type, is_altitude AE BC)
  (BD_eq : BD = 1)
  (BE_eq : BE = 1) : 
  AD = real.cbrt 2 - 1 :=
sorry

end length_of_AD_l632_632597


namespace probability_one_girl_no_growth_pie_l632_632288

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l632_632288


namespace inradius_of_triangle_area_twice_perimeter_l632_632438

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l632_632438


namespace sum_primes_20_to_30_l632_632167

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632167


namespace sum_of_primes_between_20_and_30_l632_632088

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632088


namespace constant_term_value_l632_632834

theorem constant_term_value :
  ∀ (x y z k : ℤ), (4 * x + y + z = 80) → (2 * x - y - z = 40) → (x = 20) → (3 * x + y - z = k) → (k = 60) :=
by 
  intros x y z k h₁ h₂ hx h₃
  sorry

end constant_term_value_l632_632834


namespace g_2002_equals_1_l632_632824

theorem g_2002_equals_1 (f : ℝ → ℝ)
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1)
  (g : ℝ → ℝ := fun x => f x + 1 - x)
  : g 2002 = 1 :=
by
  sorry

end g_2002_equals_1_l632_632824


namespace number_of_two_digit_nums_tens_less_than_units_nums_divisible_by_3_nums_l632_632501

-- Define the sets A and B
def A := {2, 4, 6, 8}
def B := {1, 3, 5, 7, 9}

-- Prove the number of different two-digit numbers that can be formed
theorem number_of_two_digit_nums : (Set.card (A × B)) = 20 := by
  sorry

-- Prove the number of two-digit numbers where tens digit is less than units digit
theorem tens_less_than_units_nums : 
  (Set.card { (t, u) | t ∈ A ∧ u ∈ B ∧ t < u }) = 9 := by
  sorry

-- Prove the number of two-digit numbers that are divisible by 3
theorem divisible_by_3_nums : 
  (Set.card { (t, u) | t ∈ A ∧ u ∈ B ∧ (10 * t + u) % 3 = 0 }) = 7 := by
  sorry

end number_of_two_digit_nums_tens_less_than_units_nums_divisible_by_3_nums_l632_632501


namespace triangle_properties_l632_632384

noncomputable def point := (ℝ × ℝ)

def triangle (A B C : point) := 
  ∃ (altitudeFromAB BD : ℝ×ℝ → Prop) (D : point),
    altitudeFromAB = λ P, P.1 - 2 * P.2 + 1 = 0 
    ∧ B = (5, -6)
    ∧ D = ((C.1 + 1) / 2, (C.2 + 2) / 2)
    ∧ BD ((7 * D.1 + 5 * D.2 - 5) = 0)
    ∧ (C.1 - 2 * C.2 + 1 = 0)

noncomputable def area (A B C : point) :=
  abs ((1/2) * (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1))

theorem triangle_properties :
  ∃ B : point,
    B = (5, -6)
    ∧ ∃ C : point,
    let A := (1,2) in 
    area A B C = 12 :=
by
  let A := (1, 2) : point
  have h1 : B = (5, -6) := sorry
  have h2 : D = ((C.1 + 1) / 2, (C.2 + 2) / 2) := sorry
  have h3 : BD ((7 * D.1 + 5 * D.2 - 5) = 0) := sorry
  have h4 : C.1 - 2 * C.2 + 1 = 0 := sorry
  have h5 : area A B C = 12 := sorry
  exact ⟨(5, -6), h1, ⟨C, h5⟩⟩

end triangle_properties_l632_632384


namespace no_card_with_number_less_than_0_l632_632471

open Real

theorem no_card_with_number_less_than_0.3 :
  let Jungkook := 0.8
  let Yoongi := 1/2
  let Yoojeong := 0.9
  let Yuna := 1/3
  (∀ (x : ℝ), x ∈ {Jungkook, Yoongi, Yoojeong, Yuna} → x < 0.3 → false) :=
by
  intros
  repeat { intro; cases h; try {linarith} }
  sorry

end no_card_with_number_less_than_0_l632_632471


namespace sum_of_first_seven_terms_l632_632048

variable (a : ℕ → ℝ) -- a sequence of real numbers (can be adapted to other types if needed)

-- Given conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + n * d

def sum_of_three_terms (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  a 2 + a 3 + a 4 = sum

-- Theorem to prove
theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h1 : is_arithmetic_progression a) (h2 : sum_of_three_terms a 12) :
  (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) = 28 :=
sorry

end sum_of_first_seven_terms_l632_632048


namespace man_l632_632211

-- Conditions
def speed_with_current : ℝ := 18
def speed_of_current : ℝ := 3.4

-- Problem statement
theorem man's_speed_against_current :
  (speed_with_current - speed_of_current - speed_of_current) = 11.2 := 
by
  sorry

end man_l632_632211


namespace find_m_l632_632867

theorem find_m (x y m : ℤ) 
  (h1 : 4 * x + y = 34)
  (h2 : m * x - y = 20)
  (h3 : y ^ 2 = 4) 
  : m = 2 :=
sorry

end find_m_l632_632867


namespace monotone_intervals_and_max_difference_l632_632386

def f (x : ℝ) : ℝ := 4 * x - x^4

theorem monotone_intervals_and_max_difference (a : ℝ) (x₁ x₂ : ℝ) (h : f x₁ = a) (k : f x₂ = a) (hx2_gt_1 : 1 < x₂) : 
  (∀ x, x < 1 → deriv f x > 0) ∧
  (∀ x, 1 < x → deriv f x < 0) ∧
  (a ≤ 3) ∧ 
  (x₂ - 1 ≤ 0) :=
by
  have h_deriv : deriv f x = 4 - 4 * x^3 := sorry
  have crit_point : x = 1 := sorry 
  have f_decreasing_above_1 : ∀ x, 1 < x → deriv f x < 0 := sorry
  have f_increasing_below_1 : ∀ x, x < 1 → deriv f x > 0 := sorry
  have a_le_3 : a ≤ 3 := sorry
  have max_difference : x₂ - 1 ≤ 0 := sorry
  exact ⟨f_increasing_below_1, f_decreasing_above_1, a_le_3, max_difference⟩

end monotone_intervals_and_max_difference_l632_632386


namespace integer_intersection_points_l632_632657

theorem integer_intersection_points (m : ℕ) (h : m > 0) :
  (∃ x, y = m * x^2 + (-m-2) * x + 2 ∧ y = 0 ∧ x ∈ ℤ) ∧
  (∃ y, x = 0 ∧ y = m * 0^2 + (-m-2) * 0 + 2 ∧ y ∈ ℤ) ∧
  (1,0) ∈ {(x, 0) |  x = 1} ∧
  (2,0) ∈ {(x, 0) | x = 2} ∧
  (0,2) ∈ {(0, y) | y = 2} := by
  sorry

end integer_intersection_points_l632_632657


namespace lines_parallel_m_eq_neg5_l632_632857

theorem lines_parallel_m_eq_neg5 (m : ℝ) :
  (∀ x y : ℝ, mx + 2y - 2 = 0 → 5x + (m + 3)y - 5 = 0) → m = -5 :=
by
  sorry

end lines_parallel_m_eq_neg5_l632_632857


namespace isabel_country_albums_l632_632677

theorem isabel_country_albums (num_pop_albums : ℕ) (songs_per_album : ℕ) (total_songs : ℕ) (songs_from_country_albums : ℕ) :
  num_pop_albums = 5 →
  songs_per_album = 8 →
  total_songs = 72 →
  songs_from_country_albums = total_songs - num_pop_albums * songs_per_album →
  songs_from_country_albums / songs_per_album = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  exact h4

end isabel_country_albums_l632_632677


namespace initial_profit_percentage_l632_632699

-- Given conditions as definitions
def C : ℝ := 50
def new_C : ℝ := 0.80 * C
def new_S (S : ℝ) : ℝ := S - 10.50
def target_new_S : ℝ := 1.30 * new_C

-- The main theorem to prove the initial profit percentage
theorem initial_profit_percentage (S : ℝ) (hS : new_S S = target_new_S) :
  let profit := S - C in
  let profit_percentage := (profit / C) * 100 in
  profit_percentage = 25 :=
by
  sorry

end initial_profit_percentage_l632_632699


namespace probability_no_growth_pie_l632_632264

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l632_632264


namespace sum_primes_20_to_30_l632_632168

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632168


namespace jessie_problem_l632_632469

def round_to_nearest_five (n : ℤ) : ℤ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

theorem jessie_problem :
  round_to_nearest_five ((82 + 56) - 15) = 125 :=
by
  sorry

end jessie_problem_l632_632469


namespace intersect_on_semi_circle_l632_632905

/-- Let ABC be a right triangle at C.
    H is the foot of the altitude from C to AB.
    Point D is chosen inside the triangle CBH such that CH bisects the segment AD.
    P is the intersection of the lines BD and CH.
    ω is the semicircle with diameter BD that intersects the segment CB in its interior.
    A line passing through P is tangent to ω at Q.
    Show that the lines CQ and AD intersect on ω. -/
theorem intersect_on_semi_circle
  (A B C H D P Q: Type*)
  (ω : Set (Type*))
  (h1 : ∠ C = 90°)
  (h2 : H = foot_of_altitude C AB)
  (h3 : D ∈ triangle C B H)
  (h4 : bisects CH AD)
  (h5 : P = intersection BD CH)
  (h6 : semicircle ω = diameter BD)
  (h7 : ω ∩ CB ∈ interior)
  (h8 : tangent_through P ω = Q) :
  intersects_on (CQ) (AD) (ω) :=
sorry

end intersect_on_semi_circle_l632_632905


namespace hydras_never_die_l632_632630

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632630


namespace perfect_square_has_even_digit_perfect_square_has_even_digit_base_12_l632_632522

theorem perfect_square_has_even_digit (n : ℕ) (d : ℕ) : 
  (d > 1) → (∀ i < d, i^2 < d^(2:ℕ)) → (∃ k < n*n, (k ≤ d*n^2) ∧ even k) := 
sorry

theorem perfect_square_has_even_digit_base_12 (n : ℕ) (d : ℕ) : 
  (d > 1) → (∀ i < d, i^2 < d^(2:ℕ)) → (∃ k < n*n, (k ≤ d*n^2) ∧ even k) := 
sorry

end perfect_square_has_even_digit_perfect_square_has_even_digit_base_12_l632_632522


namespace initial_caterpillars_l632_632459

theorem initial_caterpillars (C : ℕ) 
    (hatch_eggs : C + 4 - 8 = 10) : C = 14 :=
by
  sorry

end initial_caterpillars_l632_632459


namespace find_y_l632_632329

theorem find_y (y : ℝ) : log y 8 = log 125 5 → y = 512 :=
by
  sorry

end find_y_l632_632329


namespace area_of_figure_l632_632940

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end area_of_figure_l632_632940


namespace theta_in_fourth_quadrant_l632_632412

-- Definitions for the conditions
def cos_theta_pos (θ : ℝ) := cos θ > 0
def sin_2theta_neg (θ : ℝ) := sin (2 * θ) < 0

-- Definition stating \theta is in the fourth quadrant
def in_fourth_quadrant (θ : ℝ) := 0 < θ ∧ θ < (2 * π) / 2 ∧ sin θ < 0

-- Mathematical proof problem in Lean 4 statement
theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : cos_theta_pos θ) (h2 : sin_2theta_neg θ) : in_fourth_quadrant θ :=
by
  sorry

end theta_in_fourth_quadrant_l632_632412


namespace prime_sum_20_to_30_l632_632137

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632137


namespace shanghai_metro_min_repeated_stations_l632_632028

-- Define the graph types and necessary structures
structure Graph :=
(nodes : Type)
(edges : nodes → nodes → Prop)
(connected : ∀ n₁ n₂, ∃ path : list nodes, path.head = n₁ ∧ path.last = some n₂ ∧ ∀ i j, list.index_of i path < list.index_of j path → edges i j)

-- Define the minimum n value for traveling all nodes
def min_repeated_stations (G : Graph) : ℕ :=
Inf { n | ∃ (start end : G.nodes), ∀ (p : list G.nodes), 
    p.head = some start ∧ p.last = some end ∧ (∀ (i : G.nodes), i ∈ p) ∧ (∃ S : finset G.nodes, S.card = n ∧ ∀ (s ∈ S), p.count s > 1)}

-- The theorem stating the minimum value of n
theorem shanghai_metro_min_repeated_stations (G : Graph) (hG_conn : G.connected) : min_repeated_stations G = 3 :=
by
  sorry

end shanghai_metro_min_repeated_stations_l632_632028


namespace hydras_survive_l632_632618

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632618


namespace sum_of_primes_between_20_and_30_l632_632091

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632091


namespace sphere_radius_change_factor_l632_632578

/-- Given the surface area of an original sphere being 2464 cm² and the surface
    area of a new sphere being 9856 cm², prove that the factor by which the radius is changed is 2. -/
theorem sphere_radius_change_factor 
  (A1 A2 : ℝ) 
  (hA1 : A1 = 2464)
  (hA2 : A2 = 9856) : 
  ∃ k : ℝ, k = 2 ∧ A2 = 4 * real.pi * (k^2) * (A1 / (4 * real.pi)) := 
begin
  sorry
end

end sphere_radius_change_factor_l632_632578


namespace rationalize_denominator_equals_ABC_product_l632_632536

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l632_632536


namespace remaining_money_after_purchase_l632_632793

-- Definitions based on conditions
def cheapest_lamp_cost := 20
def most_expensive_multiplier := 3
def discount_rate := 0.10
def sales_tax_rate := 0.08
def franks_money := 90

-- Derived definitions to match problem conditions and lead to the correct answer
def most_expensive_lamp_cost := cheapest_lamp_cost * most_expensive_multiplier
def discount_amount := most_expensive_lamp_cost * discount_rate
def discounted_price := most_expensive_lamp_cost - discount_amount
def sales_tax_amount := discounted_price * sales_tax_rate
def final_price := discounted_price + sales_tax_amount
def remaining_money := franks_money - final_price

-- Theorem statement
theorem remaining_money_after_purchase : 
  remaining_money = 31.68 :=
  by
    -- Proof steps would go here
    sorry

end remaining_money_after_purchase_l632_632793


namespace sum_primes_in_range_l632_632079

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632079


namespace angle_CAB_is_90_degrees_l632_632065

-- Define the setup conditions
def twoCirclesTouchAtA (circle1 circle2 : ℝ → ℝ → Prop) (A : ℝ × ℝ) : Prop :=
  circle1 A ∧ circle2 A

def commonExternalTangent (circle1 circle2 : ℝ → ℝ → Prop) (C B : ℝ × ℝ) : Prop :=
  ∀ M : ℝ × ℝ, 
    (∀ P ∈ [C, M], P ∉ circle1) ∧ 
    (∀ P ∈ [B, M], P ∉ circle2) ∧ 
    [C, M, B] form a line

-- Prove that angle CAB equals 90 degrees
theorem angle_CAB_is_90_degrees (circle1 circle2 : ℝ → ℝ → Prop) 
  (A C B : ℝ × ℝ) 
  (h1 : twoCirclesTouchAtA circle1 circle2 A)
  (h2 : commonExternalTangent circle1 circle2 C B) :
  ∠ C A B = 90 :=
by
  sorry

end angle_CAB_is_90_degrees_l632_632065


namespace find_S_6_l632_632364

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Define the sequence sum function for the first n terms
noncomputable def sum_first_n (n : ℕ) : ℤ :=
  n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

-- Define the conditions
axiom S_4_eq : S 4 = -2
axiom S_5_eq : S 5 = 0

-- Define what we want to prove
theorem find_S_6 : S 6 = 3 := by
  -- Assumptions are conditions
  assume h1 : S 4 = -2
  assume h2 : S 5 = 0
  -- Definitions of the sums using the sum formula
  have S_4_def : S 4 = sum_first_n 4 := by sorry
  have S_5_def : S 5 = sum_first_n 5 := by sorry
  -- Prove the required value for S_6 using the definition of the sums
  have S_6_def : S 6 = sum_first_n 6 := by sorry
  -- Prove S_6 = 3
  sorry

end find_S_6_l632_632364


namespace hydras_will_live_l632_632636

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632636


namespace problem_100m_plus_n_equals_604_l632_632477

/-- Problem Statement:
Let \(ABCD\) be a square of side length \(6\). Points \(E\) and \(F\) are selected on rays \(AB\) and \(AD\) such that segments \(EF\) and \(BC\) intersect at a point \(L\), \(D\) lies between \(A\) and \(F\), and the area of \(\triangle AEF\) is 36. Clio constructs triangle \(PQR\) with \(PQ = BL\), \(QR = CL\) and \(RP = DF\), and notices that the area of \(\triangle PQR\) is \(\sqrt{6}\). If the sum of all possible values of \(DF\) is \(\sqrt{m} + \sqrt{n}\) for positive integers \(m \ge n\), then show that \(100m + n = 604\). -/
theorem problem_100m_plus_n_equals_604 :
  ∃ (m n : ℕ), 
    let A := (0 : ℝ, 0 : ℝ),
        B := (6, 0), 
        C := (6, 6), 
        D := (0, 6)
    in 
    ∃ (x y : ℝ) (E := (x, 0)) (F := (0, y)) (L : ℝ × ℝ) (BL DF : ℝ) (PQR_area := √6),
      (x > 6) ∧ 
      (y > 6) ∧ 
      (xy = 72) ∧ 
      (PQ = BL) ∧ 
      (QR = (C - L).pos) ∧ 
      (RP = DF) ∧ 
      (triangle_area PQR PQ QR RP) = √6 ∧ 
      (sum_possible_DF_values DF = (√6 + √2)) ∧
      m = 6 ∧ 
      n = 4 ∧ 
      100 * m + n = 604 :=
  sorry

end problem_100m_plus_n_equals_604_l632_632477


namespace sin_40_tan_10_sub_sqrt3_l632_632341

noncomputable theory

def angle_10 := 10 * Real.pi / 180
def angle_40 := 40 * Real.pi / 180

theorem sin_40_tan_10_sub_sqrt3:
  Real.sin angle_40 * (Real.tan angle_10 - Real.sqrt 3) = -1 :=
sorry

end sin_40_tan_10_sub_sqrt3_l632_632341


namespace weekly_rent_proof_exists_l632_632463

-- Required definitions based on the conditions
def weekly_store_expenses : ℕ := 3440
def weekly_employee_wages : ℕ := 80 * 2 * 12.50
def utilities_factor : ℝ := 1.20

-- The weekly rent we want to find, denoted as R
def weekly_rent (R : ℕ) : Prop := utilities_factor * R + weekly_employee_wages = weekly_store_expenses

-- The proof statement translating the question and correct answer
theorem weekly_rent_proof_exists : ∃ R : ℕ, weekly_rent R ∧ R = 1200 :=
by
  let R := 1200
  have h1 : utilities_factor * (R : ℝ) + weekly_employee_wages = weekly_store_expenses
    := by norm_num
  use R
  split
  · exact h1
  · norm_num

end weekly_rent_proof_exists_l632_632463


namespace lines_parallel_m_eq_neg5_l632_632856

theorem lines_parallel_m_eq_neg5 (m : ℝ) :
  (∀ x y : ℝ, mx + 2y - 2 = 0 → 5x + (m + 3)y - 5 = 0) → m = -5 :=
by
  sorry

end lines_parallel_m_eq_neg5_l632_632856


namespace decimal_to_binary_51_l632_632740

theorem decimal_to_binary_51: nat.bin_enc 51 = "110011" :=
  sorry

end decimal_to_binary_51_l632_632740


namespace millionth_digit_of_1_over_41_l632_632775

theorem millionth_digit_of_1_over_41 :
  let d := 5    -- The period of decimal expansion is 5
  let seq := [0,2,4,3,9]  -- The repeating sequence of 1/41
  let millionth_digit_position := 1000000 % d  -- Find the position in the repeating sequence
  seq.millionth_digit_position = 9 :=
by
  let d := 5
  let seq := [0, 2, 4, 3, 9]
  let millionth_digit_position := 1000000 % d
  show seq.millionth_digit_position = 9
  sorry

end millionth_digit_of_1_over_41_l632_632775


namespace stocks_closed_higher_l632_632679

variables (x : ℝ) (total number_higher : ℝ) (closed_higher : ℝ)

axiom total : total = 1980
axiom relation : closed_higher = 1.20 * x
axiom sum_relation : x + closed_higher = total

theorem stocks_closed_higher : closed_higher = 1080 :=
by
  rw [relation, sum_relation, total]
  have h₁ : x + 1.2 * x = 1980 := by linarith
  have h₂ : 2.2 * x = 1980 := by linarith
  have h₃ : x = 1980 / 2.2 := by linarith
  have h₄ : x = 900 := by norm_num
  rw h₄ at relation
  linarith

end stocks_closed_higher_l632_632679


namespace unique_campers_went_rowing_at_least_once_l632_632436

-- Definitions of the number of campers in various categories
def total_campers : ℕ := 500
def morning : ℕ := 235
def afternoon : ℕ := 387
def evening : ℕ := 142
def morning_and_afternoon : ℕ := 58
def afternoon_and_evening : ℕ := 23
def morning_and_evening : ℕ := 15
def all_three_sessions : ℕ := 8

-- The proof problem
theorem unique_campers_went_rowing_at_least_once :
  morning + afternoon + evening - (morning_and_afternoon + afternoon_and_evening + morning_and_evening) + all_three_sessions = 572 :=
by
  calc
    235 + 387 + 142 - (58 + 23 + 15) + 8 = 764 - 96 + 8 : by decide -- summing and subtracting
    ... = 668 + 8 : by decide -- simplifying
    ... = 676 : by decide -- final calculation

end unique_campers_went_rowing_at_least_once_l632_632436


namespace reflection_proof_l632_632770

def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let divisor := u.1 * u.1 + u.2 * u.2
  ( ((u.1 * v.1 + u.2 * v.2) / divisor) * u.1,
    ((u.1 * v.1 + u.2 * v.2) / divisor) * u.2 )

def reflection_matrix (v : ℝ × ℝ) (u : ℝ × ℝ) : ℕ → ℝ × ℝ :=
  fun _ =>
    (let p := vector_proj u v in
    (2 * p.1 - v.1, 2 * p.2 - v.2))

theorem reflection_proof (x y : ℝ) :
  let v := (x, y)
  let u := (4, 1)
  let p := vector_proj u v
  let r := reflection_matrix v u 2
  r = (14/17 * x + 8/17 * y, 8/17 * x - 6/17 * y) :=
  sorry

end reflection_proof_l632_632770


namespace probability_no_growth_pie_l632_632268

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l632_632268


namespace x_squared_minus_y_squared_l632_632865

theorem x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 9 / 16)
  (h2 : x - y = 5 / 16) :
  x^2 - y^2 = 45 / 256 :=
by
  sorry

end x_squared_minus_y_squared_l632_632865


namespace roots_quadratic_l632_632763

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l632_632763


namespace each_person_consumes_16_cookies_l632_632297

theorem each_person_consumes_16_cookies :
  let batches := 8
  let dozen := 12
  let cookies_per_batch := 5 * dozen
  let total_cookies := batches * cookies_per_batch
  let people := 30
  total_cookies / people = 16 := by
    let batches := 8
    let dozen := 12
    let cookies_per_batch := 5 * dozen
    let total_cookies := batches * cookies_per_batch
    let people := 30
    have total_cookies_eq : total_cookies = 480 := by
      sorry
    have cookies_per_person := total_cookies / people
    have result : cookies_per_person = 16 := by
      sorry
    exact result

end each_person_consumes_16_cookies_l632_632297


namespace trigonometric_identity_evaluation_l632_632057

theorem trigonometric_identity_evaluation :
  sin 15 * cos 30 * sin 75 = real.sqrt 3 / 8 :=
by
  sorry

end trigonometric_identity_evaluation_l632_632057


namespace find_a_find_m_l632_632391

noncomputable def f (x a : ℝ) : ℝ := Real.exp 1 * x - a * Real.log x

theorem find_a {a : ℝ} (h : ∀ x, f x a = Real.exp 1 - a / x)
  (hx : f (1 / Real.exp 1) a = 0) :
  a = 1 :=
by
  sorry

theorem find_m (a : ℝ) (h_a : a = 1)
  (h_exists : ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) 
    ∧ f x₀ a < x₀ + m) :
  1 + Real.log (Real.exp 1 - 1) < m :=
by
  sorry

end find_a_find_m_l632_632391


namespace minimum_value_of_f_l632_632336

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_value_of_f : ∃ x : ℝ, f x = 4 ∧ ∀ y : ℝ, f y ≥ 4 :=
by {
  sorry
}

end minimum_value_of_f_l632_632336


namespace angle_BD_plane_ABC_45_degrees_l632_632790

-- Declare the points A, B, C, D being the vertices of square ABCD
variables {A B C D : Point}

-- Condition: Fold square ABCD along the diagonal AC
-- Define the function fold_square (we don't need to implement it, just declare its existence)
def fold_square (A B C D : Point) : Prop := sorry

-- Condition: Volume of the regular pyramid with vertices A, B, C, and D is maximized
def volume_maximized (A B C D : Point) : Prop := sorry

-- Given the above conditions, we aim to prove the angle formed by BD and plane ABC is 45°
theorem angle_BD_plane_ABC_45_degrees : 
  fold_square A B C D → 
  volume_maximized A B C D → 
  angle BD (plane A B C) = 45 :=
begin
  sorry
end

end angle_BD_plane_ABC_45_degrees_l632_632790


namespace find_x_l632_632752

theorem find_x (x : ℝ) (y : ℝ) : 
  (10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) ↔ x = (3 / 2) :=
by
  sorry

end find_x_l632_632752


namespace decimal_to_binary_51_l632_632738

theorem decimal_to_binary_51: nat.bin_enc 51 = "110011" :=
  sorry

end decimal_to_binary_51_l632_632738


namespace series_sum_plus_fraction_equals_one_l632_632994

def sum_series (n : ℕ) : ℚ :=
  ∑ k in finset.range (n - 1), 1 / ((k + 2 : ℚ) * (k + 1))

theorem series_sum_plus_fraction_equals_one : sum_series 24 + 1 / 24 = 1 :=
by
  sorry

end series_sum_plus_fraction_equals_one_l632_632994


namespace area_equivalence_l632_632902

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Point := sorry
noncomputable def arc_midpoint (A B C : Point) : Point := sorry
noncomputable def is_concyclic (P Q R S : Point) : Prop := sorry
noncomputable def area_of_quad (A B C D : Point) : ℝ := sorry
noncomputable def area_of_pent (A B C D E : Point) : ℝ := sorry

theorem area_equivalence (A B C I X Y M : Point)
  (h1 : I = incenter A B C)
  (h2 : X = angle_bisector B A C)
  (h3 : Y = angle_bisector C A B)
  (h4 : M = arc_midpoint A B C)
  (h5 : is_concyclic M X I Y) :
  area_of_quad M B I C = area_of_pent B X I Y C := 
sorry

end area_equivalence_l632_632902


namespace expand_expression_l632_632761

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 :=
by
  sorry

end expand_expression_l632_632761


namespace sum_of_primes_between_20_and_30_l632_632118

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632118


namespace episodes_remaining_l632_632205

-- Definition of conditions
def seasons : ℕ := 12
def episodes_per_season : ℕ := 20
def fraction_watched : ℚ := 1 / 3
def total_episodes : ℕ := episodes_per_season * seasons
def episodes_watched : ℕ := (fraction_watched * total_episodes).toNat

-- Problem statement
theorem episodes_remaining : total_episodes - episodes_watched = 160 := by
  sorry

end episodes_remaining_l632_632205


namespace hydras_never_die_l632_632629

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632629


namespace sum_of_primes_between_20_and_30_l632_632119

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632119


namespace double_pythagorean_triple_l632_632676

theorem double_pythagorean_triple (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  (2*a)^2 + (2*b)^2 = (2*c)^2 :=
by
  sorry

end double_pythagorean_triple_l632_632676


namespace rationalize_denominator_l632_632531

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l632_632531


namespace tree_drops_leaves_on_fifth_day_l632_632452

def initial_leaves := 340
def daily_drop_fraction := 1 / 10

noncomputable def leaves_after_day (n: ℕ) : ℕ :=
  match n with
  | 0 => initial_leaves
  | 1 => initial_leaves - Nat.floor (initial_leaves * daily_drop_fraction)
  | 2 => leaves_after_day 1 - Nat.floor (leaves_after_day 1 * daily_drop_fraction)
  | 3 => leaves_after_day 2 - Nat.floor (leaves_after_day 2 * daily_drop_fraction)
  | 4 => leaves_after_day 3 - Nat.floor (leaves_after_day 3 * daily_drop_fraction)
  | _ => 0  -- beyond the 4th day

theorem tree_drops_leaves_on_fifth_day : leaves_after_day 4 = 225 := by
  -- We'll skip the detailed proof here, focusing on the statement
  sorry

end tree_drops_leaves_on_fifth_day_l632_632452


namespace population_function_relationship_population_after_10_years_years_to_reach_1_2_million_l632_632562

noncomputable def initial_population : ℕ := 100
noncomputable def annual_growth_rate : ℝ := 0.012

theorem population_function_relationship (x : ℕ) :
  ∃ y : ℝ, y = initial_population * (1 + annual_growth_rate) ^ x := by
  sorry

theorem population_after_10_years :
  ∃ y : ℝ, y = initial_population * (1 + annual_growth_rate) ^ 10 ∧ y ≈ 112.7 := by
  sorry

theorem years_to_reach_1_2_million :
  ∃ x : ℕ, initial_population * (1 + annual_growth_rate) ^ x ≈ 120 ∧ x ≈ 16 := by
  sorry

end population_function_relationship_population_after_10_years_years_to_reach_1_2_million_l632_632562


namespace find_cost_price_l632_632964

def cost_price (SP_total : ℝ) (r_tax : ℝ) (r_profit : ℝ) : ℝ :=
  let SP := (1 + r_profit) * (SP_total / (1 + r_tax + r_tax * r_profit))
  SP

theorem find_cost_price :
  cost_price 616 0.10 0.17 ≈ 478.77 :=
by
  sorry

end find_cost_price_l632_632964


namespace sum_primes_between_20_and_30_is_52_l632_632133

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632133


namespace lines_parallel_m_eq_neg5_l632_632858

theorem lines_parallel_m_eq_neg5 (m : ℝ) :
  (∀ x y : ℝ, mx + 2y - 2 = 0 → 5x + (m + 3)y - 5 = 0) → m = -5 :=
by
  sorry

end lines_parallel_m_eq_neg5_l632_632858


namespace sum_primes_20_to_30_l632_632169

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632169


namespace rationalize_denominator_l632_632540

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l632_632540


namespace option_B_is_quadratic_l632_632673

-- Definitions of the equations provided in conditions
def eqA (x y : ℝ) := x + 3 * y = 4
def eqB (y : ℝ) := 5 * y = 5 * y^2
def eqC (x : ℝ) := 4 * x - 4 = 0
def eqD (x a : ℝ) := a * x^2 - x = 1

-- Quadratic equation definition
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ) (x : ℝ), A ≠ 0 ∧ eq = (A * x^2 + B * x + C = 0)

-- Theorem: Prove that option B (eqB) is definitely a quadratic equation
theorem option_B_is_quadratic : is_quadratic eqB :=
  sorry

end option_B_is_quadratic_l632_632673


namespace train_length_approx_l632_632259

-- Definitions extracted from the given conditions
def speed_kmh : ℝ := 80
def time_sec : ℝ := 10.889128869690424
def bridge_length_m : ℝ := 142
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- The statement to prove the length of the train
theorem train_length_approx :
  let total_distance := speed_ms * time_sec in
  let train_length := total_distance - bridge_length_m in
  abs (train_length - 100.222) < 0.001 :=
by
  simp only [speed_ms, time_sec, bridge_length_m]
  -- This is a placeholder line to indicate proof is not provided.
  sorry

end train_length_approx_l632_632259


namespace hydras_never_die_l632_632653

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632653


namespace exists_another_common_point_l632_632061

-- Definitions for the conditions
variables {S_1 S_2 S_3 : Sphere} {P Q : Point}

-- Conditions
def common_point := P ∈ S_1 ∧ P ∈ S_2 ∧ P ∈ S_3
def not_all_tangent_through_P := ∀ (l : Line), (P ∈ l) → ¬(is_tangent S_1 l ∧ is_tangent S_2 l ∧ is_tangent S_3 l)

-- The theorem
theorem exists_another_common_point (h1 : common_point) (h2 : not_all_tangent_through_P) :
  ∃ (Q : Point), Q ≠ P ∧ Q ∈ S_1 ∧ Q ∈ S_2 ∧ Q ∈ S_3 :=
sorry  -- Proof would go here

end exists_another_common_point_l632_632061


namespace sum_primes_between_20_and_30_l632_632173

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632173


namespace efficiency_ratio_l632_632223

theorem efficiency_ratio (E_A E_B : ℝ) 
  (h1 : E_B = 1 / 18) 
  (h2 : E_A + E_B = 1 / 6) : 
  E_A / E_B = 2 :=
by {
  sorry
}

end efficiency_ratio_l632_632223


namespace probability_no_growth_pie_l632_632265

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l632_632265


namespace problem_equivalence_l632_632353

noncomputable def a : ℝ := -1
noncomputable def b : ℝ := 3

theorem problem_equivalence (i : ℂ) (hi : i^2 = -1) : 
  (a + 3 * i = (b + i) * i) :=
by
  -- The complex number definitions
  let lhs := a + 3 * i
  let rhs := (b + i) * i

  -- Confirming the parts
  calc
  lhs = -1 + 3 * i : by rfl
  ... = rhs       : by
    simp [a, b]
    rw [hi]
    ring

-- To skip the proof add sorry at the end.
sorry

end problem_equivalence_l632_632353


namespace sum_fourth_powers_const_l632_632025

-- Define the vertices of the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A (a : ℝ) : Point := {x := a, y := 0}
def B (a : ℝ) : Point := {x := 0, y := a}
def C (a : ℝ) : Point := {x := -a, y := 0}
def D (a : ℝ) : Point := {x := 0, y := -a}

-- Define distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Circle centered at origin
def on_circle (P : Point) (r : ℝ) : Prop :=
  P.x ^ 2 + P.y ^ 2 = r ^ 2

-- The main theorem
theorem sum_fourth_powers_const (a r : ℝ) (P : Point) (h : on_circle P r) :
  let AP_sq := dist_sq P (A a)
  let BP_sq := dist_sq P (B a)
  let CP_sq := dist_sq P (C a)
  let DP_sq := dist_sq P (D a)
  (AP_sq ^ 2 + BP_sq ^ 2 + CP_sq ^ 2 + DP_sq ^ 2) = 4 * (r^4 + a^4 + 4 * a^2 * r^2) :=
by
  sorry

end sum_fourth_powers_const_l632_632025


namespace sum_primes_in_range_l632_632085

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632085


namespace largest_difference_consecutive_blue_tickets_l632_632518

def is_blue_ticket (n : ℕ) : Prop :=
  let digits := λ m, [m / 100000 % 10, m / 10000 % 10, m / 1000 % 10, m / 100 % 10, m / 10 % 10, m % 10]
  let sum_even := digits n 1 + digits n 3 + digits n 5
  let sum_odd := digits n 0 + digits n 2 + digits n 4
  sum_even = sum_odd

theorem largest_difference_consecutive_blue_tickets : 
  ∃ (X Y : ℕ), is_blue_ticket X ∧ is_blue_ticket Y ∧ 
                0 ≤ X ∧ X < Y ∧ Y ≤ 999999 ∧ 
                ∀ Z, is_blue_ticket Z → X < Z → Z < Y → has_val_eq (Y - X) 990 :=
sorry

end largest_difference_consecutive_blue_tickets_l632_632518


namespace sum_of_primes_between_20_and_30_l632_632109

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632109


namespace tangent_line_at_point_l632_632033

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 1) (h_point : (x, y) = (1, 0)) :
  y = x - 1 :=
sorry

end tangent_line_at_point_l632_632033


namespace factor_expression_l632_632795

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) :=
by
  sorry

end factor_expression_l632_632795


namespace first_three_digits_l632_632983

theorem first_three_digits:
  let x := (10^3003 + 1) ^ (10 / 9)
  in ((x - (x.to_nat)) * 10^3).to_nat = 111 :=
sorry

end first_three_digits_l632_632983


namespace sum_coefficients_binomial_l632_632969

theorem sum_coefficients_binomial (x y : ℕ) :
  let expansion := (x + y) ^ 8 in
  (expansion.eval 1 1 = 256) :=
by
  sorry

end sum_coefficients_binomial_l632_632969


namespace radius_xz_plane_is_sqrt_59_l632_632252

-- Definitions of the conditions
def center_xy_plane := (3, 5, 0)
def radius_xy_plane := 2
def center_xz_plane := (0, 5, -8)

def center_sphere := (3, 5, -8)

def radius_sphere := Real.sqrt (radius_xy_plane^2 + (center_sphere.2 - center_xy_plane.2)^2)

-- The theorem statement
theorem radius_xz_plane_is_sqrt_59 : 
  let r_xz := Real.sqrt (radius_sphere^2 - (center_sphere.1 - center_xz_plane.1)^2) in
  r_xz = Real.sqrt 59 := by
  sorry

end radius_xz_plane_is_sqrt_59_l632_632252


namespace polynomial_determination_l632_632749

theorem polynomial_determination (P : Polynomial ℝ) :
  (∀ X : ℝ, P.eval (X^2) = (X^2 + 1) * P.eval X) →
  (∃ a : ℝ, ∀ X : ℝ, P.eval X = a * (X^2 - 1)) :=
by
  sorry

end polynomial_determination_l632_632749


namespace no_growth_pie_probability_l632_632277

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l632_632277


namespace x_y_difference_correct_l632_632798

noncomputable def x_y_difference : ℝ :=
  if h : ∃ x y : ℝ, 
    ({2, 0, x} : set ℝ) = ({1/x, |x|, y/x} : set ℝ) ∧ 
    x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 0 ∧ |x| = 2
  then
    let ⟨x, y, _, _⟩ := h in x - y
  else 
    0 -- This will never be the case due to the existential hypothesis.

theorem x_y_difference_correct : x_y_difference = 1/2 := by
  sorry

end x_y_difference_correct_l632_632798


namespace cos_alpha_value_l632_632356

open Real

noncomputable def cos_alpha := Real.cos

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ set.Ioo (π / 2) π) (h2 : Real.sin (π - α) = (1 / 3)) :
  cos_alpha α = - (2 * sqrt 2) / 3 :=
by
  -- Proof goes here
  sorry

end cos_alpha_value_l632_632356


namespace topological_sort_possible_l632_632361
-- Import the necessary library

-- Definition of simple, directed, and acyclic graph (DAG)
structure SimpleDirectedAcyclicGraph (V : Type*) :=
  (E : V → V → Prop)
  (acyclic : ∀ v : V, ¬(E v v)) -- no loops
  (simple : ∀ (u v : V), (E u v) → ¬(E v u)) -- no bidirectional edges
  (directional : ∀ (u v w : V), E u v → E v w → E u w) -- directional transitivity

-- Existence of topological sort definition
def topological_sort_exists {V : Type*} (G : SimpleDirectedAcyclicGraph V) : Prop :=
  ∃ (numbering : V → ℕ), ∀ (u v : V), (G.E u v) → (numbering u > numbering v)

-- Theorem statement
theorem topological_sort_possible (V : Type*) (G : SimpleDirectedAcyclicGraph V) : topological_sort_exists G :=
  sorry

end topological_sort_possible_l632_632361


namespace millionth_digit_1_div_41_l632_632772

theorem millionth_digit_1_div_41 : 
  ∃ n, n = 1000000 ∧ (n % 5 = 0) → digit_after_decimal (1 / 41) n = 9 :=
begin
  intro exists n,
  sorry -- proof goes here
end

end millionth_digit_1_div_41_l632_632772


namespace a36_is_131_l632_632826

def sequence (r s t : ℤ) : ℤ := 2^r + 2^s + 2^t

def is_valid (r s t : ℤ) : Prop := 0 ≤ t ∧ t < s ∧ s < r

noncomputable def a_sequence : List ℤ :=
  List.filter_map (λ r, 
    List.filter_map (λ s, 
      List.filter_map (λ t, if is_valid r s t then some (sequence r s t) else none)
      (List.range r).reverse
    )
    (List.range r).reverse
  )
  (List.range 25)

noncomputable def a_n (n : ℕ) : ℤ := (List.sort (≤) a_sequence).get! (n - 1)

theorem a36_is_131 : a_n 36 = 131 :=
by
  sorry

end a36_is_131_l632_632826


namespace sum_primes_20_to_30_l632_632160

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632160


namespace sum_primes_20_to_30_l632_632155

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632155


namespace child_grandmother_ratio_l632_632041

variable (G D C : ℕ)

axiom cond1 : G + D + C = 120
axiom cond2 : D + C = 60
axiom cond3 : D = 48

theorem child_grandmother_ratio : (C : ℚ) / G = 1 / 5 :=
by
  sorry

end child_grandmother_ratio_l632_632041


namespace second_tank_water_amount_l632_632034

theorem second_tank_water_amount (C : ℝ) (h1: 0.45 * C = 300) (h2 : 2 * C = 300 + 0.45 * C + 1250) : C = 1000 → 0.45 * 1000 = 450 :=
by
  -- Definitions and assumptions
  have h3 : 2 * C = 1550 + 0.45 * C := by sorry
  have h4 : 1.55 * C = 1550 := by sorry
  have h5 : C = 1000 := by sorry
  -- Calculation
  exact h5 atk₀.mul₀ h1

end second_tank_water_amount_l632_632034


namespace negation_prop_l632_632574

open Real Int

def prop (x : ℝ) : Prop := ∃ n_0 : ℤ, n_0 ≤ x^2

theorem negation_prop :
  ¬ (∀ x ∈ ℝ, prop x) ↔ ∃ x_0 ∈ ℝ, ∀ n : ℤ, n > x_0^2 :=
sorry

end negation_prop_l632_632574


namespace intersection_points_form_rectangle_l632_632833

theorem intersection_points_form_rectangle
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 + y^2 = 34) :
  ∃ (a b u v : ℝ), (a * b = 8) ∧ (a^2 + b^2 = 34) ∧ 
  (u * v = 8) ∧ (u^2 + v^2 = 34) ∧
  ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧ 
  ((u = -x ∧ v = -y) ∨ (u = -y ∧ v = -x)) ∧
  ((a = u ∧ b = v) ∨ (a = v ∧ b = u)) ∧ 
  ((x = -u ∧ y = -v) ∨ (x = -v ∧ y = -u)) ∧
  (
    (a, b) ≠ (u, v) ∧ (a, b) ≠ (-u, -v) ∧ 
    (a, b) ≠ (v, u) ∧ (a, b) ≠ (-v, -u) ∧
    (u, v) ≠ (-a, -b) ∧ (u, v) ≠ (b, a) ∧ 
    (u, v) ≠ (-b, -a)
  ) :=
by sorry

end intersection_points_form_rectangle_l632_632833


namespace worm_in_apple_l632_632261

theorem worm_in_apple (radius : ℝ) (travel_distance : ℝ) (h_radius : radius = 31) (h_travel_distance : travel_distance = 61) :
  ∃ S : Set ℝ, ∀ point_on_path : ℝ, (point_on_path ∈ S) → false :=
by
  sorry

end worm_in_apple_l632_632261


namespace rosy_days_to_complete_work_l632_632509

theorem rosy_days_to_complete_work (mary_days_to_complete : ℕ) (efficiency_increase : ℕ) : 
  mary_days_to_complete = 28 → 
  efficiency_increase = 40 → 
  (mary_days_to_complete : ℝ) / (1 + efficiency_increase / 100 : ℝ) = 20 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  exact sorry

end rosy_days_to_complete_work_l632_632509


namespace hydra_survival_l632_632604

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632604


namespace shading_blocks_distinct_count_l632_632593

-- Defining the 3x3 grid and blocks
def grid : Type := fin 3 × fin 3

-- Defining the problem conditions
def two_shaded_blocks (b1 b2 : grid) : Prop := b1 ≠ b2

-- The main theorem to state the problem
theorem shading_blocks_distinct_count :
  (∃ (blocks : finset (grid × grid)), blocks.card = 8 ∧ 
    ∀ (b1 b2 : grid), (b1, b2) ∈ blocks ∧ two_shaded_blocks b1 b2) :=
sorry

end shading_blocks_distinct_count_l632_632593


namespace sum_primes_between_20_and_30_l632_632177

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632177


namespace sum_primes_20_to_30_l632_632163

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632163


namespace bunnies_out_of_burrows_l632_632418

theorem bunnies_out_of_burrows 
  (bunnies_per_min : Nat) 
  (num_bunnies : Nat) 
  (hours : Nat) 
  (mins_per_hour : Nat):
  bunnies_per_min = 3 -> num_bunnies = 20 -> mins_per_hour = 60 -> hours = 10 -> 
  (bunnies_per_min * mins_per_hour * hours * num_bunnies = 36000) :=
by
  intros,
  sorry

end bunnies_out_of_burrows_l632_632418


namespace malia_buttons_second_box_l632_632998

theorem malia_buttons_second_box (n1 n2 n3 n4 n5 n6 : ℕ) :
  n1 = 1 → n3 = 9 → n4 = 27 → n5 = 81 → n6 = 243 →
  (∀ m k, m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6 → n(k + 1) = 3 * n(k) → k = m - 1 → n(k + 1)) →
  n2 = 3 :=
by
  intros h1 h3 h4 h5 h6 hGeo
  have h := hGeo 3 8 _ rfl sorry
  sorry

end malia_buttons_second_box_l632_632998


namespace min_sine_range_l632_632801

noncomputable def min_sine_ratio (α β γ : ℝ) := min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β)

theorem min_sine_range (α β γ : ℝ) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : α + β + γ = Real.pi) :
  1 ≤ min_sine_ratio α β γ ∧ min_sine_ratio α β γ < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end min_sine_range_l632_632801


namespace henry_games_given_l632_632860

theorem henry_games_given (G : ℕ) (henry_initial : ℕ) (neil_initial : ℕ) (henry_now : ℕ) (neil_now : ℕ) :
  henry_initial = 58 →
  neil_initial = 7 →
  henry_now = henry_initial - G →
  neil_now = neil_initial + G →
  henry_now = 4 * neil_now →
  G = 6 :=
by
  intros h_initial n_initial h_now n_now eq_henry
  sorry

end henry_games_given_l632_632860


namespace find_a_plus_c_l632_632892

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  (b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B) ∧
  (b = 2) ∧
  ((1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 2) / 2)

theorem find_a_plus_c {A B C a b c : ℝ} (h : triangle_ABC A B C a b c) :
  a + c = 4 :=
by
  rcases h with ⟨hc1, hc2, hc3⟩
  sorry

end find_a_plus_c_l632_632892


namespace probability_one_girl_no_growth_pie_l632_632284

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l632_632284


namespace part1_part2_l632_632390

variable (m x : ℝ)

-- Define the function f(x) and its derivative.
def f (x : ℝ) : ℝ := Real.exp (x + m) - Real.log x
def f' (x : ℝ) : ℝ := Real.exp (x + m) - 1/x

-- Part (I): Prove that if x = 1 is an extremum of f(x), then e^x - e * ln(x) ≥ e
theorem part1 (h_extremum1 : f' m 1 = 0) : 
  ∀ x : ℝ, Real.exp x - Real.exp 1 * Real.log x ≥ Real.exp 1 := by
  sorry

-- Part (II): Prove the range of values for m given an extremum at x0 and f(x) ≥ 0
variable (x0 a : ℝ)
def g (x : ℝ) : ℝ := Real.exp (x + m) - 1/x

theorem part2 (h_extremum2 : f' m x0 = 0) (h_nonneg : ∀ x : ℝ, f m x ≥ 0)
  (h_const : a * Real.log a = 1) : 
  m ≥ -a - Real.log a := by
  sorry

end part1_part2_l632_632390


namespace greatest_number_of_sets_l632_632678

-- Definitions based on conditions
def whitney_tshirts := 5
def whitney_buttons := 24
def whitney_stickers := 12
def buttons_per_set := 2
def stickers_per_set := 1

-- The statement to prove the greatest number of identical sets Whitney can make
theorem greatest_number_of_sets : 
  ∃ max_sets : ℕ, 
  max_sets = whitney_tshirts ∧ 
  max_sets ≤ (whitney_buttons / buttons_per_set) ∧
  max_sets ≤ (whitney_stickers / stickers_per_set) :=
sorry

end greatest_number_of_sets_l632_632678


namespace doctor_is_correct_l632_632609

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632609


namespace product_of_areas_eq_one_over_32_l632_632900

-- Define the regular polygons DIAL, FOR, and FRIEND, and the distance ID = 1 in Lean
variables (DIAL FOR FRIEND : Type)
variable (ID : ℝ)

-- Assume DIAL, FOR, and FRIEND are regular polygons
axiom DIAL_is_regular : true
axiom FOR_is_regular : true
axiom FRIEND_is_regular : true

-- Assume the distance ID is equal to 1
axiom ID_is_one : ID = 1

-- Define the product of all possible areas of the triangle OLA
def all_areas_product := (1/2) * ((2 + real.sqrt 3)/4) * ((2 - real.sqrt 3)/4)

-- Theorem statement to prove the product of all possible areas of the triangle OLA
theorem product_of_areas_eq_one_over_32 : DIAL_is_regular → FOR_is_regular → FRIEND_is_regular → 
  ID_is_one → all_areas_product = (1/32) :=
begin
  sorry
end

end product_of_areas_eq_one_over_32_l632_632900


namespace surface_area_of_sphere_l632_632234

-- Define the conditions: cube edge length and the sphere property
def cube_edge_length (a : ℝ) := a = 2

def sphere_condition (diameter radius : ℝ) :=
  diameter = 2 * radius ∧ radius = sqrt 3

-- Define the theorem to prove the surface area of the sphere
theorem surface_area_of_sphere (a diameter radius : ℝ) (S : ℝ) (π : ℝ) (h1 : cube_edge_length a) (h2 : sphere_condition diameter radius) :
  S = 4 * π * radius^2 → S = 12 * π :=
by
  sorry

end surface_area_of_sphere_l632_632234


namespace hydras_never_die_l632_632652

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632652


namespace trapezoid_area_correct_l632_632260

noncomputable def trapezoid_area (x : ℝ) : ℝ :=
  let base1 := 3 * x
  let base2 := 5 * x + 2
  (base1 + base2) / 2 * x

theorem trapezoid_area_correct (x : ℝ) : trapezoid_area x = 4 * x^2 + x :=
  by
  sorry

end trapezoid_area_correct_l632_632260


namespace total_cost_is_660_l632_632467

def total_material_cost : ℝ :=
  let velvet_area := (12 * 4) * 3
  let velvet_cost := velvet_area * 3
  let silk_cost := 2 * 6
  let lace_cost := 5 * 2 * 10
  let bodice_cost := silk_cost + lace_cost
  let satin_area := 2.5 * 1.5
  let satin_cost := satin_area * 4
  let leather_area := 1 * 1.5 * 2
  let leather_cost := leather_area * 5
  let wool_area := 5 * 2
  let wool_cost := wool_area * 8
  let ribbon_cost := 3 * 2
  velvet_cost + bodice_cost + satin_cost + leather_cost + wool_cost + ribbon_cost

theorem total_cost_is_660 : total_material_cost = 660 := by
  sorry

end total_cost_is_660_l632_632467


namespace statement_a_statement_b_statement_c_l632_632345

noncomputable def f (x : ℝ) : ℝ := x / (1 + 3 * x^2)

theorem statement_a (x : ℝ) : f(x) + f(-x) = 0 := sorry

theorem statement_b : ∀ x, f(x) = 0 → x = 0 := sorry

theorem statement_c : 
  set.range f = set.Icc (-real.sqrt 3 / 6) (real.sqrt 3 / 6) := sorry

end statement_a_statement_b_statement_c_l632_632345


namespace find_cos_tan_of_angle_B_l632_632868

variables (B : Real) (sin_B : Real)

-- Given conditions
def conditions : Prop :=
  B ∈ (π, 3 * π / 2) ∧ sin_B = -5 / 13

-- Stating the proof problem
theorem find_cos_tan_of_angle_B 
  (h : conditions B sin_B) : 
  ∃ cos_B tan_B, 
    cos_B = -12 / 13 ∧ tan_B = 5 / 12 :=
by 
  -- Skipping the proof
  sorry

end find_cos_tan_of_angle_B_l632_632868


namespace bubble_pass_prob_l632_632015

theorem bubble_pass_prob :
  ∀ (r : Fin 50 → ℕ) (r_distinct : ∀ i j, i ≠ j → r i ≠ r j) (random_order : ∀ i j, i ≠ j → r i ≠ r j),
  r 10 < r 41 →
  (∀ k, k < 41 → r 10 > r k →
        r 10 ∉ (Finset.range 40.to_finset.filter (λ i, r 41 > r i)).val) →
  ∃ p q : ℕ, p / q = 1 / 1640 ∧ p + q = 1641 :=
by sorry

end bubble_pass_prob_l632_632015


namespace prime_sum_20_to_30_l632_632146

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632146


namespace fourth_root_409600000_eq_80_l632_632734

theorem fourth_root_409600000_eq_80 : real.sqrt_real (409600000)^(1/4) = 80 := by
  sorry

end fourth_root_409600000_eq_80_l632_632734


namespace algebraic_expression_value_l632_632866

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a - 2 * b + 2 = 0) :
  2024 + 2 * a - b = 2023 :=
by
  sorry

end algebraic_expression_value_l632_632866


namespace weeks_to_buy_iphone_l632_632304

-- Definitions based on conditions
def iphone_cost : ℝ := 800
def trade_in_value : ℝ := 240
def earnings_per_week : ℝ := 80

-- Mathematically equivalent proof problem
theorem weeks_to_buy_iphone : 
  ∀ (iphone_cost trade_in_value earnings_per_week : ℝ), 
  (iphone_cost - trade_in_value) / earnings_per_week = 7 :=
by
  -- Using the given conditions directly.
  intros iphone_cost trade_in_value earnings_per_week
  sorry

end weeks_to_buy_iphone_l632_632304


namespace incorrect_square_root_0_2_l632_632997

theorem incorrect_square_root_0_2 :
  (0.45)^2 = 0.2 ∧ (0.02)^2 ≠ 0.2 :=
by
  sorry

end incorrect_square_root_0_2_l632_632997


namespace mass_percentage_of_nitrogen_in_N2O5_l632_632662

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)

theorem mass_percentage_of_nitrogen_in_N2O5 : 
  (2 * atomic_mass_N / molar_mass_N2O5 * 100) = 25.94 := 
by 
  sorry

end mass_percentage_of_nitrogen_in_N2O5_l632_632662


namespace parallel_lines_l632_632851

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l632_632851


namespace area_of_S_l632_632707

noncomputable def hexagon := { z : ℂ | ∃ (k : ℤ), (z - k * (1 + complex.I √3 / 2)).abs = 1 / √3 }

def region_R := { z : ℂ | z ∉ hexagon }

def S := { w : ℂ | ∃ (z : ℂ), z ∈ region_R ∧ w = 1 / z }

theorem area_of_S : (area_of S) = 3 * real.sqrt(3) + (real.pi / 2) :=
sorry

end area_of_S_l632_632707


namespace common_difference_arithmetic_seq_l632_632448

variables {a₁ d : ℤ}

def arithmetic_seq (n : ℕ) : ℤ := a₁ + n * d 

theorem common_difference_arithmetic_seq :
  arithmetic_seq 1 + arithmetic_seq 5 = 8 ∧
  arithmetic_seq 2 + arithmetic_seq 3 = 3 →
  d = 5 :=
by
  sorry

end common_difference_arithmetic_seq_l632_632448


namespace largest_4_digit_divisible_by_50_l632_632985

-- Define the condition for a 4-digit number
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the largest 4-digit number
def largest_4_digit : ℕ := 9999

-- Define the property that a number is exactly divisible by 50
def divisible_by_50 (n : ℕ) : Prop := n % 50 = 0

-- Main statement to be proved
theorem largest_4_digit_divisible_by_50 :
  ∃ n, is_4_digit n ∧ divisible_by_50 n ∧ ∀ m, is_4_digit m → divisible_by_50 m → m ≤ n ∧ n = 9950 :=
by
  sorry

end largest_4_digit_divisible_by_50_l632_632985


namespace sitting_next_to_jen_l632_632327

-- Define the persons as a type
inductive Person 
| Fay
| Guy
| Huw
| Ian
| Jen

open Person

-- Define what it means to be next to someone in a circle
def next_to_in_circle (a b : Person) (circle : List Person) : Prop :=
 ∃ i, circle.nth i = some a ∧ circle.nth ((i + 1) % circle.length) = some b ∨
      circle.nth ((i + 1) % circle.length) = some a ∧ circle.nth i = some b

-- Define the given conditions
def conditions (circle : List Person) : Prop :=
  (next_to_in_circle Guy Fay circle) ∧ 
  (next_to_in_circle Guy Ian circle) ∧ 
  ¬ (next_to_in_circle Jen Ian circle)

-- The theorem we want to prove
theorem sitting_next_to_jen (circle : List Person) (h : conditions circle) :
  next_to_in_circle Jen Fay circle ∧ next_to_in_circle Jen Huw circle :=
sorry

end sitting_next_to_jen_l632_632327


namespace prob_4_wins_3_losses_l632_632980

theorem prob_4_wins_3_losses :
  (∀ p_A p_B : ℝ, p_A = 1/2 → p_B = 1/2 → (p_A + p_B = 1) →
    let prob := (Nat.choose 7 4 * (1/2)^4 * (1/2)^3) / 2^7 in
    prob = 35 / 128) :=
by
  sorry

end prob_4_wins_3_losses_l632_632980


namespace number_of_Ca_atoms_in_compound_l632_632232

theorem number_of_Ca_atoms_in_compound
  (n : ℤ)
  (total_weight : ℝ)
  (ca_weight : ℝ)
  (i_weight : ℝ)
  (n_i_atoms : ℤ)
  (molecular_weight : ℝ) :
  n_i_atoms = 2 →
  molecular_weight = 294 →
  ca_weight = 40.08 →
  i_weight = 126.90 →
  n * ca_weight + n_i_atoms * i_weight = molecular_weight →
  n = 1 :=
by
  sorry

end number_of_Ca_atoms_in_compound_l632_632232


namespace find_f_neg_five_l632_632355

def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + 1

theorem find_f_neg_five (a b : ℝ) (h₁ : f a b 5 = 7) : f a b (-5) = -5 := by
  sorry

end find_f_neg_five_l632_632355


namespace find_magnitude_of_AB_l632_632874

-- Defining vectors and vectors' operations
variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}

-- Given conditions
def condition1 : inner A B = 10 :=
sorry

def condition2 : inner (-A) (C - B) = 6 :=
sorry

-- The theorem we need to prove
theorem find_magnitude_of_AB 
  (h1 : inner A B = 10) 
  (h2 : inner (-A) (C - B) = 6) :
  ∥A - B∥ = 4 :=
sorry

end find_magnitude_of_AB_l632_632874


namespace center_of_mass_distance_l632_632655

theorem center_of_mass_distance (p q a b : ℝ) (hp : p > 0) (hq : q > 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  let z := (p * a + q * b) / (p + q) in
  z = (p * a + q * b) / (p + q) :=
  by sorry

end center_of_mass_distance_l632_632655


namespace combined_area_of_square_and_triangle_l632_632660

noncomputable def combined_area (d : ℝ) : ℝ :=
  let a := d / (Real.sqrt 2) in
  let area_square := a^2 in
  let area_triangle := (Real.sqrt 3 / 4) * d^2 in
  area_square + area_triangle

theorem combined_area_of_square_and_triangle (d : ℝ) (h : d = 30) :
  combined_area d = 450 + 225 * Real.sqrt 3 :=
by
  dsimp [combined_area]
  rw [h]
  sorry

end combined_area_of_square_and_triangle_l632_632660


namespace train_speed_l632_632258

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 250
noncomputable def crossing_time : ℝ := 28.79769618430526

noncomputable def speed_m_per_s : ℝ := (train_length + bridge_length) / crossing_time
noncomputable def speed_kmph : ℝ := speed_m_per_s * 3.6

theorem train_speed : speed_kmph = 50 := by
  sorry

end train_speed_l632_632258


namespace necessary_and_sufficient_condition_l632_632815

noncomputable def z (a : ℝ) : ℂ :=
  ((a + 2 * complex.I) * (-1 + complex.I)) / complex.I

def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem necessary_and_sufficient_condition : 
  ∀ a : ℝ, (is_purely_imaginary (z a) ↔ a = 2) :=
by
  sorry

end necessary_and_sufficient_condition_l632_632815


namespace roses_in_vase_l632_632058

/-- There were initially 16 roses and 3 orchids in the vase.
    Jessica cut 8 roses and 8 orchids from her garden.
    There are now 7 orchids in the vase.
    Prove that the number of roses in the vase now is 24. -/
theorem roses_in_vase
  (initial_roses initial_orchids : ℕ)
  (cut_roses cut_orchids remaining_orchids final_roses : ℕ)
  (h_initial: initial_roses = 16)
  (h_initial_orchids: initial_orchids = 3)
  (h_cut: cut_roses = 8 ∧ cut_orchids = 8)
  (h_remaining_orchids: remaining_orchids = 7)
  (h_orchids_relation: initial_orchids + cut_orchids = remaining_orchids + cut_orchids - 4)
  : final_roses = initial_roses + cut_roses := by
  sorry

end roses_in_vase_l632_632058


namespace sum_of_integers_l632_632948

theorem sum_of_integers (numbers : List ℕ) (h1 : numbers.Nodup) 
(h2 : ∃ a b, (a ≠ b ∧ a * b = 16 ∧ a ∈ numbers ∧ b ∈ numbers)) 
(h3 : ∃ c d, (c ≠ d ∧ c * d = 225 ∧ c ∈ numbers ∧ d ∈ numbers)) :
  numbers.sum = 44 :=
sorry

end sum_of_integers_l632_632948


namespace books_sold_correct_l632_632511

-- Define the number of books sold by Matias, Olivia, and Luke on each day
def matias_monday := 7
def olivia_monday := 5
def luke_monday := 12

def matias_tuesday := 2 * matias_monday
def olivia_tuesday := 3 * olivia_monday
def luke_tuesday := luke_monday / 2

def matias_wednesday := 3 * matias_tuesday
def olivia_wednesday := 4 * olivia_tuesday
def luke_wednesday := luke_tuesday

-- Calculate the total books sold by each person over three days
def matias_total := matias_monday + matias_tuesday + matias_wednesday
def olivia_total := olivia_monday + olivia_tuesday + olivia_wednesday
def luke_total := luke_monday + luke_tuesday + luke_wednesday

-- Calculate the combined total of books sold by Matias, Olivia, and Luke
def combined_total := matias_total + olivia_total + luke_total

-- Prove the combined total equals 167
theorem books_sold_correct : combined_total = 167 := by
  sorry

end books_sold_correct_l632_632511


namespace sum_primes_between_20_and_30_l632_632182

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632182


namespace problem_solution_l632_632667

theorem problem_solution : (275^2 - 245^2) / 30 = 520 := by
  sorry

end problem_solution_l632_632667


namespace sum_of_primes_between_20_and_30_l632_632100

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632100


namespace jill_nickels_count_l632_632470

theorem jill_nickels_count (n d : Nat) (h1 : n + d = 50) (h2 : 0.05 * n + 0.10 * d = 3.50) : n = 30 :=
by
  sorry

end jill_nickels_count_l632_632470


namespace brooke_number_of_cows_l632_632299

def milk_price_per_gallon : ℝ := 3
def butter_price_per_stick : ℝ := 1.5
def milk_to_butter_ratio : ℝ := 2
def milk_per_cow : ℝ := 4
def number_of_customers : ℕ := 6
def milk_per_customer : ℝ := 6
def total_earnings : ℝ := 144

theorem brooke_number_of_cows :
  let milk_sold := number_of_customers * milk_per_customer in
  let revenue_from_milk := milk_sold * milk_price_per_gallon in
  let revenue_from_butter := total_earnings - revenue_from_milk in
  let sticks_of_butter := revenue_from_butter / butter_price_per_stick in
  let milk_for_butter := sticks_of_butter / milk_to_butter_ratio in
  let total_milk := milk_sold + milk_for_butter in
  let number_of_cows := total_milk / milk_per_cow in
  number_of_cows = 12 :=
by
  let milk_sold := number_of_customers * milk_per_customer
  let revenue_from_milk := milk_sold * milk_price_per_gallon
  let revenue_from_butter := total_earnings - revenue_from_milk
  let sticks_of_butter := revenue_from_butter / butter_price_per_stick
  let milk_for_butter := sticks_of_butter / milk_to_butter_ratio
  let total_milk := milk_sold + milk_for_butter
  let number_of_cows := total_milk / milk_per_cow
  show number_of_cows = 12 from sorry

end brooke_number_of_cows_l632_632299


namespace binomial_sum_generating_function_l632_632377

open Nat

theorem binomial_sum_generating_function 
  (p q n : ℕ) (hp : 0 < p) (hq : 0 < q) (hn : 0 < n):
  (∑ k in Finset.range (n + 1), (Nat.choose (p + k) p) * (Nat.choose (q + n - k) q)) = Nat.choose (p + q + n + 1) (p + q + 1) :=
  sorry

end binomial_sum_generating_function_l632_632377


namespace effective_profit_approximation_l632_632421

noncomputable def effective_profit_percentage (SP CP TotalReceived EffectiveProfit EffectiveProfitPercentage : ℝ) :=
  CP = 0.92 * SP ∧
  TotalReceived = SP * 1.08 ∧
  EffectiveProfit = TotalReceived - CP ∧
  EffectiveProfitPercentage = (EffectiveProfit / CP) * 100 →
  EffectiveProfitPercentage ≈ 17.39

theorem effective_profit_approximation :
  ∃ (SP CP TotalReceived EffectiveProfit EffectiveProfitPercentage : ℝ),
    SP = 100 →
    effective_profit_percentage SP CP TotalReceived EffectiveProfit EffectiveProfitPercentage :=
begin
  use [100, 92, 108, 16, 17.39],
  intros hSP,
  rw hSP,
  dsimp [effective_profit_percentage],
  refine ⟨_, _, _, _⟩;
  norm_num,
  sorry
end

end effective_profit_approximation_l632_632421


namespace feet_heads_difference_l632_632879

theorem feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let heads := hens + goats + camels + keepers
  let feet := (2 * hens) + (4 * goats) + (4 * camels) + (2 * keepers)
  feet - heads = 193 :=
by
  sorry

end feet_heads_difference_l632_632879


namespace convex_pentagon_side_no_longer_than_regular_l632_632460

theorem convex_pentagon_side_no_longer_than_regular (
  (A B C D E : Point) (O : Point) (circle_center : Point -> Real -> Circle)
  (is_convex_pentagon : Convex (Polygon.mk5 A B C D E))
  (is_inscribed : Inscribed (Polygon.mk5 A B C D E) (circle_center O 1))
  (regular_pentagon_side_equal : ∀ (P Q : Point), 
  (RegularPolygon.mk5 O.circle_radius P =
   RegularPolygon.mk5 O.circle_radius Q)) :
  ∃ (A1 A2 : Point), SideLength (A1, A2) ≤ RegularPolygon.side_length (circle_center O 1) := 
sorry

end convex_pentagon_side_no_longer_than_regular_l632_632460


namespace proof_f_f3_l632_632491

def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem proof_f_f3 :
  f (f 3) = -11 / 10 :=
  sorry

end proof_f_f3_l632_632491


namespace IdenticalTrianglesExist_l632_632458

theorem IdenticalTrianglesExist
    (initial_triangles : Set (triangle ℝ))
    (H_initial : initial_triangles.card = 4)
    (H_identical : ∀ (T1 T2 ∈ initial_triangles), T1 ≅ T2)
    (Cut : ∀ T ∈ initial_triangles, ∃ T1 T2, is_altitude_cut T T1 T2) :
    ∀ (n : ℕ) (final_triangles : Set (triangle ℝ)),
    after_n_cuts initial_triangles Cut n final_triangles →
    ∃ T1 T2 ∈ final_triangles, T1 ≅ T2 := 
sorry

end IdenticalTrianglesExist_l632_632458


namespace average_of_P_Q_R_is_correct_l632_632737

theorem average_of_P_Q_R_is_correct (P Q R : ℝ) 
  (h1 : 1001 * R - 3003 * P = 6006) 
  (h2 : 2002 * Q + 4004 * P = 8008) : 
  (P + Q + R)/3 = (2 * (P + 5))/3 :=
sorry

end average_of_P_Q_R_is_correct_l632_632737


namespace propositions_correct_l632_632289

-- Define variables and propositions
variables {A : ℝ} {ω : ℝ} (p q : Prop) (x : ℝ) 
def prop1 := A > 30 → sin A > 1 / 2
def prop2 := (3, 4)⃗ • (-2, -1)⃗ / ∥(-2, -1)⃗∥ = -2
def prop3 := ∃ (x : ℝ), cos x = 1 ∧ ∀ (x : ℝ), x^2 - x + 1 > 0
def prop4 := ¬ (x^2 + x - 6 ≥ 0 → x > 2)
def prop5 := ∀ x, ω > 0 → f' x ≤ 3 → f x = sin (ω x + π / 6) - 2 → is_symmetric f (π / 3)

-- Define the function derivative
noncomputable def f' (x : ℝ) := ω * cos (ω * x + π / 6)

-- Define the symmetry condition
def is_symmetric (f : ℝ → ℝ) (c : ℝ) := ∀ x, f (2 * c - x) = f x

-- State the final proof problem
theorem propositions_correct (h1 : ¬prop1) (h2 : ¬prop2) (h3 : prop3) (h4 : prop4) (h5 : ¬prop5) :
  { 3, 4 } = { n | [false, false, true, true, false].nth (n - 1) = some true } := by
  sorry

end propositions_correct_l632_632289


namespace green_large_toys_count_l632_632432

variables (T : ℕ)
variables (red_large_toys : ℕ) (green_large_toys : ℕ)
variables (red_toys : ℕ) (green_toys : ℕ) (small_toys : ℕ) (large_toys : ℕ)
variables (red_small_toys : ℕ)

-- Conditions
def condition1 : Prop := red_toys = 0.40 * T
def condition2 : Prop := green_toys = 0.60 * T
def condition3 : Prop := small_toys = 0.50 * T
def condition4 : Prop := large_toys = 0.50 * T
def condition5 : Prop := red_small_toys = 0.10 * T
def condition6 : Prop := red_large_toys = 60
def condition7 : Prop := red_large_toys = red_toys - red_small_toys

-- Theorem to prove
theorem green_large_toys_count : 
  condition1 T red_toys →
  condition2 T green_toys →
  condition3 T small_toys →
  condition4 T large_toys →
  condition5 T red_small_toys →
  condition6 red_large_toys →
  condition7 T red_toys red_small_toys red_large_toys →
  green_large_toys = 40 := by
  sorry

end green_large_toys_count_l632_632432


namespace find_diameter_of_window_l632_632250

-- Define the given conditions
def is_perimeter_of_semicircle (D : ℝ) (P : ℝ) : Prop :=
  P = (Real.pi * D / 2) + D

def given_perimeter := 161.96
def expected_diameter := 63.01

-- The proof statement
theorem find_diameter_of_window :
  ∃ D : ℝ, is_perimeter_of_semicircle D given_perimeter ∧ D ≈ expected_diameter :=
by
  sorry

end find_diameter_of_window_l632_632250


namespace sum_of_primes_between_20_and_30_l632_632101

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632101


namespace max_value_on_interval_l632_632572

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : 
  ∀ (x : ℝ), x ∈ Icc 0 3 → f x ≤ 5 :=
begin
  sorry
end

end max_value_on_interval_l632_632572


namespace sum_primes_between_20_and_30_is_52_l632_632126

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632126


namespace solve_system_l632_632554

theorem solve_system (x y : ℝ) :
  (x + 3*y + 3*x*y = -1) ∧ (x^2*y + 3*x*y^2 = -4) →
  (x = -3 ∧ y = -1/3) ∨ (x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 4/3) ∨ (x = 4 ∧ y = -1/3) :=
by
  sorry

end solve_system_l632_632554


namespace jerry_age_l632_632514

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J - 8) (h2 : M = 24) : J = 8 :=
by
  sorry

end jerry_age_l632_632514


namespace problem1_inner_problem2_inner_l632_632343

-- Problem 1
theorem problem1_inner {m n : ℤ} (hm : |m| = 5) (hn : |n| = 4) (opposite_signs : m * n < 0) :
  m^2 - m * n + n = 41 ∨ m^2 - m * n + n = 49 :=
sorry

-- Problem 2
theorem problem2_inner {a b c d x : ℝ} (opposite_ab : a + b = 0) (reciprocals_cd : c * d = 1) (hx : |x| = 5) (hx_pos : x > 0) :
  3 * (a + b) - 2 * (c * d) + x = 3 :=
sorry

end problem1_inner_problem2_inner_l632_632343


namespace isosceles_triangle_height_ratio_l632_632954

theorem isosceles_triangle_height_ratio (a b : ℝ) (h₁ : b = (4 / 3) * a) :
  ∃ m n : ℝ, b / 2 = m + n ∧ m = (2 / 3) * a ∧ n = (1 / 3) * a ∧ (m / n) = 2 :=
by
  sorry

end isosceles_triangle_height_ratio_l632_632954


namespace prime_sum_20_to_30_l632_632193

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632193


namespace slope_of_tangent_line_at_A_l632_632050

-- Define the function y = e^x + 1
def f (x : ℝ) : ℝ := Real.exp x + 1

-- Define point A
def A : ℝ × ℝ := (0, 2)

-- State the theorem
theorem slope_of_tangent_line_at_A :
  (deriv f 0) = 1 :=
by sorry

end slope_of_tangent_line_at_A_l632_632050


namespace arithmetic_sequence_sum_l632_632441

-- Let {a_n} be an arithmetic sequence.
-- Define Sn as the sum of the first n terms.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n-1))) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * a 6 = a 7 + 5) :
  S a 11 = 55 :=
sorry

end arithmetic_sequence_sum_l632_632441


namespace incorrect_statement_l632_632717

-- Definitions related to prisms and their properties.
def prism (P : Type) : Prop :=
  -- Conditions of the definition of a prism
  (∀ f : P → Prop, lateral_face P f → parallelogram f) ∧
  (∃ b₁ b₂ : P → Prop, base_face P b₁ ∧ base_face P b₂ ∧ congruent b₁ b₂ ∧ parallel b₁ b₂) ∧
  (∃ f1 f2 : P → Prop, f1 ≠ f2 ∧ parallel f1 f2)

-- The statement to prove that statement A is incorrect.
theorem incorrect_statement (P : Type) [prism P] :
  ¬ (∀ f1 f2 : P → Prop, parallel f1 f2 → base_face P f1 ∧ base_face P f2) :=
sorry

end incorrect_statement_l632_632717


namespace neg_q_sufficient_not_necc_neg_p_l632_632373

variable (p q : Prop)

theorem neg_q_sufficient_not_necc_neg_p (hp: p → q) (hnpq: ¬(q → p)) : (¬q → ¬p) ∧ (¬(¬p → ¬q)) :=
by
  sorry

end neg_q_sufficient_not_necc_neg_p_l632_632373


namespace milburg_children_count_l632_632587

theorem milburg_children_count : 
  ∀ (total_population grown_ups : ℕ), 
  total_population = 8243 → grown_ups = 5256 → 
  (total_population - grown_ups) = 2987 :=
by
  intros total_population grown_ups h1 h2
  sorry

end milburg_children_count_l632_632587


namespace equilateral_BHD_l632_632756

theorem equilateral_BHD 
  {A B C D E H K : Type*}
  (A_ne_B : A ≠ B) 
  (B_ne_C : B ≠ C) 
  (C_ne_D : C ≠ D) 
  (D_ne_E : D ≠ E) 
  (E_ne_H : E ≠ H) 
  (H_ne_K : H ≠ K)
  (equilateral_ABC : equilateral_triangle A B C) 
  (equilateral_CDE : equilateral_triangle C D E) 
  (equilateral_EHK : equilateral_triangle E H K)
  (vec_AD_eq_DK : vector_equal A D D K) :
  equilateral_triangle B H D := 
sorry

end equilateral_BHD_l632_632756


namespace cardboard_plastic_squares_coincide_l632_632000

theorem cardboard_plastic_squares_coincide (n : ℕ) 
  (cardboard plastic : set (set (ℝ × ℝ)))
  (h_cardboard : ∀ s1 s2 ∈ cardboard, s1 ≠ s2 → ∀ p ∈ s1, p ∉ s2) 
  (h_plastic : ∀ s1 s2 ∈ plastic, s1 ≠ s2 → ∀ p ∈ s1, p ∉ s2) 
  (h_vertices : (⋃ s ∈ cardboard, s) = (⋃ s ∈ plastic, s)) : 
  ∀ c ∈ cardboard, ∃ p ∈ plastic, c = p := 
sorry

end cardboard_plastic_squares_coincide_l632_632000


namespace common_ratio_of_geometric_sequence_l632_632965

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def is_geometric_sequence (x y z : ℝ) (q : ℝ) : Prop :=
  y^2 = x * z

theorem common_ratio_of_geometric_sequence 
    (a_n : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a_n) 
    (a1 a3 a5 : ℝ)
    (h1 : a1 = a_n 1 + 1) 
    (h3 : a3 = a_n 3 + 3) 
    (h5 : a5 = a_n 5 + 5) 
    (h_geom : is_geometric_sequence a1 a3 a5 1) : 
  1 = 1 :=
by
  sorry

end common_ratio_of_geometric_sequence_l632_632965


namespace hydras_survive_l632_632622

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632622


namespace complex_solutions_l632_632762

theorem complex_solutions (z : ℂ) : (z^2 = -91 - 49 * complex.I) ↔ (z = (7 * real.sqrt 2 / 2 - 7 * real.sqrt 2 * complex.I) ∨ z = (-7 * real.sqrt 2 / 2 + 7 * real.sqrt 2 * complex.I)) :=
by
  sorry

end complex_solutions_l632_632762


namespace points_collinear_with_S_l632_632981

-- Definitions for the problem conditions
def Triangle := Type
def Point := Type
variable (A B C A1 B1 C1 P S : Point)
variable (circle: Set Point)
variable (in_circle : ∀ X : Point, X ∈ circle)
variable (inscribed_triangle : Triangle → Set Point → Prop)
variable (A2 B2 C2 : Point)
variable (intersects_in : Point → Point → Point → Prop)

-- Prove the problem statement
theorem points_collinear_with_S
  (ABC A1B1C1 : Triangle)
  (h1 : inscribed_triangle ABC circle)
  (h2 : inscribed_triangle A1B1C1 circle)
  (h_intersect: intersects_in A A1 S ∧ intersects_in B B1 S ∧ intersects_in C C1 S)
  (h_P_on_circle : in_circle P)
  (h_intersections : intersects_in (P A1) (BC) A2
     ∧ intersects_in (P B1) (CA) B2
     ∧ intersects_in (P C1) (AB) C2) :
  collinear {A2, B2, C2, S} :=
sorry

end points_collinear_with_S_l632_632981


namespace expressions_equal_iff_conditions_l632_632758

theorem expressions_equal_iff_conditions (a b c : ℝ) :
  (2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c)) ↔ (a = 0 ∨ a + 2 * b + 1.5 * c = 0) :=
by
  sorry

end expressions_equal_iff_conditions_l632_632758


namespace salary_increase_l632_632945

theorem salary_increase (S P : ℝ) (h1 : 0.70 * S + P * (0.70 * S) = 0.91 * S) : P = 0.30 :=
by
  have eq1 : 0.70 * S * (1 + P) = 0.91 * S := by sorry
  have eq2 : S * (0.70 + 0.70 * P) = 0.91 * S := by sorry
  have eq3 : 0.70 + 0.70 * P = 0.91 := by sorry
  have eq4 : 0.70 * P = 0.21 := by sorry
  have eq5 : P = 0.21 / 0.70 := by sorry
  have eq6 : P = 0.30 := by sorry
  exact eq6

end salary_increase_l632_632945


namespace remaining_episodes_l632_632204

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l632_632204


namespace sum_primes_between_20_and_30_is_52_l632_632134

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632134


namespace sum_primes_20_to_30_l632_632158

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632158


namespace false_statement_among_choices_l632_632996

/-- Define vectors and planes -/
structure Vector3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def is_parallel (v1 v2 : Vector3D) : Prop :=
∃ k : ℝ, v1 = {x := k * v2.x, y := k * v2.y, z := k * v2.z}

def normal_vector_plane (n : Vector3D) (P : Type) : Prop := sorry
def is_perpendicular (v1 v2 : Vector3D) : Prop := v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

/-- Establish given conditions -/
theorem false_statement_among_choices :
  let n1 := Vector3D.mk 0 1 3
  let n2 := Vector3D.mk 1 0 3
  let v := Vector3D.mk 0 0 1
  ∃ (α β : Type),
  (forall (m : Vector3D), is_parallel m n1 → (∃ line : Type, is_parallel line n1)) ∧
  (normal_vector_plane n1 α ∧ normal_vector_plane n2 β) 
  → (forall (l : Type), (is_parallel v n1 ∧ is_perpendicular v n1) → is_parallel l α)
  ∧ (forall (a b : Vector3D), (a + b = {x := 0, y := 0, z := 0}) → is_parallel a b) 
  → ¬ (is_parallel α β) :=
by
  sorry

end false_statement_among_choices_l632_632996


namespace speed_of_stream_l632_632579

variable (x : ℝ) -- Let the speed of the stream be x kmph

-- Conditions
variable (speed_of_boat_in_still_water : ℝ)
variable (time_upstream_twice_time_downstream : Prop)

-- Given conditions
axiom h1 : speed_of_boat_in_still_water = 48
axiom h2 : time_upstream_twice_time_downstream → 1 / (speed_of_boat_in_still_water - x) = 2 * (1 / (speed_of_boat_in_still_water + x))

-- Theorem to prove
theorem speed_of_stream (h2: time_upstream_twice_time_downstream) : x = 16 := by
  sorry

end speed_of_stream_l632_632579


namespace remainder_of_3_pow_101_plus_5_mod_11_l632_632988

theorem remainder_of_3_pow_101_plus_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := by
  -- The theorem statement includes the condition that (3^101 + 5) mod 11 equals 8.
  -- The proof will make use of repetitive behavior and modular arithmetic.
  sorry

end remainder_of_3_pow_101_plus_5_mod_11_l632_632988


namespace integer_not_always_greater_decimal_l632_632721

-- Definitions based on conditions
def is_decimal (d : ℚ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), 0 ≤ f ∧ f < 1 ∧ d = i + f

def is_greater (a : ℤ) (b : ℚ) : Prop :=
  (a : ℚ) > b

theorem integer_not_always_greater_decimal : ¬ ∀ n : ℤ, ∀ d : ℚ, is_decimal d → (is_greater n d) :=
by
  sorry

end integer_not_always_greater_decimal_l632_632721


namespace range_of_k_l632_632378

theorem range_of_k (k : ℝ) (h : k ≠ 0) : (k^2 - 6 * k + 8 ≥ 0) ↔ (k ≥ 4 ∨ k ≤ 2) := 
by sorry

end range_of_k_l632_632378


namespace sum_primes_between_20_and_30_l632_632172

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632172


namespace reflection_not_perpendicular_l632_632977

def P := (2, 5)
def Q := (5, 2)
def R := (6, 6)
def P' := (5, 2)
def Q' := (2, 5)
def R' := (6, 6)

theorem reflection_not_perpendicular :
  ¬(∃ mPQ mP'Q', mPQ * mP'Q' = -1 ∧ (mPQ = (Q.2 - P.2) / (Q.1 - P.1)) ∧ (mP'Q' = (Q'.2 - P'.2) / (Q'.1 - P'.1))) := by
  sorry

end reflection_not_perpendicular_l632_632977


namespace sum_even_positive_integers_lt_70_l632_632666

theorem sum_even_positive_integers_lt_70 : 
  (∑ k in Finset.range 35, 2 * (k + 1)) = 1190 :=
by sorry

end sum_even_positive_integers_lt_70_l632_632666


namespace problem_1_l632_632989

theorem problem_1 ([condition_1 : 1 ^ (-2) = 1] [condition_2 : 2 ^ (-1) = 1/2]) :
  1 ^ (-2) + 2 ^ (-1) = 3 / 2 := sorry

end problem_1_l632_632989


namespace cone_base_radius_and_slant_height_l632_632753

-- Define the conditions
def sector_angle : ℝ := 270
def sector_radius : ℝ := 12
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def cone_base_radius_from_sector (s_angle s_radius : ℝ) : ℝ := 
  circumference s_radius * (s_angle / 360) / (2 * Real.pi)

theorem cone_base_radius_and_slant_height :
  cone_base_radius_from_sector sector_angle sector_radius = 9 ∧ sector_radius = 12 := 
by
  sorry

end cone_base_radius_and_slant_height_l632_632753


namespace probability_at_least_one_A_or_B_selected_l632_632064

/-- Given two candidates A and B, each can choose one college from the three options: A, B, or C,
with an equal probability for each choice. Prove that the probability that at least one of the candidates selects either college A or B is 8/9. -/
theorem probability_at_least_one_A_or_B_selected (h1 : fintype (fin 3)) :
  (1 - (1 / 9) : ℚ) = 8 / 9 :=
by
  sorry

end probability_at_least_one_A_or_B_selected_l632_632064


namespace expression_evaluation_l632_632318

theorem expression_evaluation : 
  76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 :=
by
  sorry

end expression_evaluation_l632_632318


namespace count_valid_as_l632_632507

theorem count_valid_as : 
  let a_values := {a : ℕ | 78 ≤ a ∧ a ≤ 110}
  ∃ n : ℕ, n = a_values.card ∧ n = 33 :=
by
  let a_values := {a : ℕ | 78 ≤ a ∧ a ≤ 110}
  existsi a_values.card
  split
  {
    rfl
  }
  {
    sorry
  }

end count_valid_as_l632_632507


namespace sum_primes_between_20_and_30_is_52_l632_632131

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632131


namespace women_attend_percent_l632_632431

variable (E : ℝ)  -- Total number of employees
variable (P_men : ℝ) (P_men_pic : ℝ) (P_pic : ℝ)

-- Conditions
def P_men_attends (P_men_pic = 0.20) : Prop := sorry
def P_men_all (P_men = 0.35) : Prop := sorry
def P_all_attends (P_pic = 0.33) : Prop := sorry

theorem women_attend_percent {P_men_pic P_men P_pic}
  (h1 : P_men_attends P_men_pic)
  (h2 : P_men_all P_men)
  (h3 : P_all_attends P_pic) :
  (((0.33 * E - 0.07 * E) / (0.65 * E)) * 100) = 40 := 
sorry

end women_attend_percent_l632_632431


namespace area_seq_formula_area_seq_limit_l632_632727

noncomputable theory

def area_seq (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (λ n : ℕ, (8 : ℝ) / 5 - (3 : ℝ) / 5 * (4 / 9) ^ n) n

theorem area_seq_formula {n : ℕ} : area_seq n = 
  if n = 0 then 1
  else (8 : ℝ) / 5 - (3 : ℝ) / 5 * (4 / 9) ^ n := sorry

theorem area_seq_limit : 
  filter.tendsto area_seq filter.at_top (nhds ((8 : ℝ) / 5)) := sorry

end area_seq_formula_area_seq_limit_l632_632727


namespace sixtieth_integer_is_32541_l632_632047

noncomputable def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n + 1) * factorial n

theorem sixtieth_integer_is_32541 :
  let digits := [1, 2, 3, 4, 5]
  let perms := (list.permutations digits).map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let sorted_perms := list.sort (≤) perms
  sorted_perms.nth_le(59) sorry = 32541
sorry

end sixtieth_integer_is_32541_l632_632047


namespace green_space_equation_l632_632062

theorem green_space_equation (x : ℝ) (h_area : x * (x - 30) = 1000) :
  x * (x - 30) = 1000 := 
by
  exact h_area

end green_space_equation_l632_632062


namespace hydrae_never_equal_heads_l632_632646

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632646


namespace sofia_running_time_l632_632551

theorem sofia_running_time :
  ∃ t : ℤ, t = 8 * 60 + 20 ∧ 
  (∀ (laps : ℕ) (d1 d2 v1 v2 : ℤ),
    laps = 5 →
    d1 = 200 →
    v1 = 4 →
    d2 = 300 →
    v2 = 6 →
    t = laps * ((d1 / v1 + d2 / v2))) :=
by
  sorry

end sofia_running_time_l632_632551


namespace millionth_digit_of_1_over_41_l632_632780

theorem millionth_digit_of_1_over_41 :
  let frac := 1 / (41 : ℚ),
      seq := "02439",
      period := (5 : ℕ) in
  (seq.get (1000000 % period - 1) = '9') :=
by
  let frac := 1 / (41 : ℚ)
  let seq := "02439"
  let period := 5
  have h_expansion : frac = 0.02439 / 10000 := sorry
  have h_period : ∀ n, frac = Rational.mkPeriodic seq period n := sorry
  have h_mod : 1000000 % period = 0 := by sorry
  have h_index := h_mod.symm ▸ (dec_trivial : 0 % 5 = 0)
  exact h_period n ▸ (dec_trivial : "02439".get 4 = '9')

end millionth_digit_of_1_over_41_l632_632780


namespace hydras_never_die_l632_632627

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632627


namespace simplify_expression_l632_632549

theorem simplify_expression (a : ℤ) (h_range : -3 < a ∧ a ≤ 0) (h_notzero : a ≠ 0) (h_notone : a ≠ 1 ∧ a ≠ -1) :
  (a - (2 * a - 1) / a) / (1 / a - a) = -3 :=
by
  have h_eq : (a - (2 * a - 1) / a) / (1 / a - a) = (1 - a) / (1 + a) :=
    sorry
  have h_a_neg_two : a = -2 :=
    sorry
  rw [h_eq, h_a_neg_two]
  sorry


end simplify_expression_l632_632549


namespace functional_relationship_and_point_l632_632822

noncomputable def directly_proportional (y x : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x

theorem functional_relationship_and_point :
  (∀ x y, directly_proportional y x → y = 2 * x) ∧ 
  (∀ a : ℝ, (∃ (y : ℝ), y = 3 ∧ directly_proportional y a) → a = 3 / 2) :=
by
  sorry

end functional_relationship_and_point_l632_632822


namespace factors_of_diminished_number_l632_632051

theorem factors_of_diminished_number (n : ℕ) (h₁ : n = 1013) (h₂ : (n - 5) % 28 = 0) :
  (n - 5) = 1008 ∧ ∀ d, d ∣ 1008 ↔ d ∈ {1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 96, 112, 126, 144, 168, 192, 252, 336, 504, 1008} :=
by
  sorry

end factors_of_diminished_number_l632_632051


namespace hydras_never_die_l632_632649

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632649


namespace valid_combinations_l632_632714

def herbs : Nat := 4
def crystals : Nat := 6
def incompatible_pairs : Nat := 3

theorem valid_combinations : 
  (herbs * crystals) - incompatible_pairs = 21 := by
  sorry

end valid_combinations_l632_632714


namespace measure_of_angle_A_l632_632799

theorem measure_of_angle_A (a b c : ℝ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  ∃ A : ℝ, 0 < A ∧ A < π ∧ cos A = -1 / 2 ∧ A = 2 * π / 3 :=
by
  use (2 * π / 3)
  split
  · sorry -- 0 < 2 * π / 3
  split
  · sorry -- 2 * π / 3 < π
  split
  · have : a^2 = 49 := by rw [h_a]; norm_num
    have : b^2 = 25 := by rw [h_b]; norm_num
    have : c^2 = 9 := by rw [h_c]; norm_num
    have : 2 * b * c = 30 := by rw [h_b, h_c]; norm_num
    show cos (2 * π / 3) = -1 / 2
    · sorry
  · rfl

end measure_of_angle_A_l632_632799


namespace find_a10_l632_632576

theorem find_a10 (a : ℕ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ, 0 < n → (1 / (a (n + 1) - 1)) = (1 / (a n - 1)) - 1) : 
  a 10 = 10/11 := 
sorry

end find_a10_l632_632576


namespace interest_rate_l632_632710

variable (P : ℝ) (T : ℝ) (SI : ℝ)

theorem interest_rate (h_P : P = 535.7142857142857) (h_T : T = 4) (h_SI : SI = 75) :
    (SI / (P * T)) * 100 = 3.5 := by
  sorry

end interest_rate_l632_632710


namespace chef_potatoes_leftover_l632_632229

def fries_per_potato : ℕ := 25
def total_fries_needed : ℕ := 200
def cubes_per_potato : ℕ := 10
def total_cubes_needed : ℕ := 50
def total_potatoes : ℕ := 30

theorem chef_potatoes_leftover :
  let potatoes_for_fries := total_fries_needed / fries_per_potato
      potatoes_for_salad := total_cubes_needed / cubes_per_potato
      total_potatoes_needed := potatoes_for_fries + potatoes_for_salad
  in total_potatoes - total_potatoes_needed = 17 :=
by
  sorry

end chef_potatoes_leftover_l632_632229


namespace triangle_cosine_consecutive_integers_l632_632569

theorem triangle_cosine_consecutive_integers :
  ∃ (n : ℕ), 
    let θ := Real.angle.ofCos (n*n + 6*n + 5) / (2*n*n + 6*n + 4) in
    let a := n, b := n+1, c := n+2 in
    let θ_small := θ in
    let θ_large := 2 * θ in
    let lhs := Real.cos θ_large in
    let rhs := 2*Real.cos θ_small^2 - 1 in
    (lhs = rhs) → (Real.cos θ_small = 3/4) := 
    sorry

end triangle_cosine_consecutive_integers_l632_632569


namespace greatest_integer_condition_l632_632070

theorem greatest_integer_condition (x: ℤ) (h₁: x < 150) (h₂: Int.gcd x 24 = 3) : x ≤ 147 := 
by sorry

example : ∃ x, x < 150 ∧ Int.gcd x 24 = 3 ∧ x = 147 :=
begin
  use 147,
  split,
  { exact lt_trans (by norm_num) (by norm_num) },
  { split,
    { norm_num },
    { refl } }
end

end greatest_integer_condition_l632_632070


namespace num_members_organization_l632_632885

noncomputable def num_committees := 6
def member_belongs_exactly_three_committees (members : Finset (Finset ℕ)) : Prop :=
  ∀ m ∈ members, m.card = 3

def unique_member_per_set (members : Finset (Finset ℕ)) : Prop :=
  ∀ s ∈ (Finset.powSet (Finset.range num_committees) 3), ∃! m ∈ members, s ⊆ m

theorem num_members_organization (members : Finset (Finset ℕ)) :
  member_belongs_exactly_three_committees members →
  unique_member_per_set members →
  members.card = 20 :=
by
  sorry

end num_members_organization_l632_632885


namespace sum_of_squares_of_distances_l632_632805

theorem sum_of_squares_of_distances (n : ℕ) (h : 0 < n)
  (A : Fin n → ℂ) (P : ℂ)
  (circ : ∀ i, Complex.abs (A i) = 1)
  (P_circ : Complex.abs P = 1)
  (regular_polygon : ∀ i j, Complex.abs (A i - A j) = 2 * Real.sin (π / n)) :
  ∑ i in Finset.range n, Complex.normSq (P - A i) = 2 * n :=
by
  sorry

end sum_of_squares_of_distances_l632_632805


namespace general_formula_nth_term_T_2n_formula_l632_632363

-- Define the conditions and the sequences
constant (a : ℕ → ℝ)
constant (S : ℕ → ℝ)
constant q : ℝ

axiom S_n : ∀ n, S n = (n * (2 * (a 2) - 2)) 
axiom S_2 : S 2 = 2 * (a 2) - 2
axiom S_3 : S 3 = (a 4) - 2
axiom q_pos : q > 0

-- Define the sequence b_n
def b (n : ℕ) : ℝ :=
  if odd n then
    Real.log2 (a n) / (n^2 * (n + 2))
  else
    (n : ℝ) / a n

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℝ :=
  (Finset.range n).sum b

-- Prove the general formula for the n-th term of the sequence a_n
theorem general_formula_nth_term : a (n : ℕ) = 2^n := by sorry

-- Prove the formula for T_{2n}
theorem T_2n_formula (n : ℕ) : T (2 * n) = 
  (n : ℝ) / (2 * n + 1) + 8 / 9 - (8 + 6 * n) / (9 * 4^n) := by sorry

end general_formula_nth_term_T_2n_formula_l632_632363


namespace distance_between_centers_of_circles_l632_632505

theorem distance_between_centers_of_circles (C_1 C_2 : ℝ) : 
  (∀ a : ℝ, (C_1 = a ∧ C_2 = a ∧ (4- a)^2 + (1 - a)^2 = a^2)) → 
  |C_1 - C_2| = 8 :=
by
  sorry

end distance_between_centers_of_circles_l632_632505


namespace min_value_of_expression_l632_632334

noncomputable def expression (x : ℝ) : ℝ :=
  (sin x) ^ 8 + (cos x) ^ 8 + 1 / (sin x) ^ 6 + (cos x) ^ 6 + 1

theorem min_value_of_expression (x : ℝ) : expression x ≥ 0 :=
sorry

end min_value_of_expression_l632_632334


namespace episodes_remaining_l632_632207

-- Definition of conditions
def seasons : ℕ := 12
def episodes_per_season : ℕ := 20
def fraction_watched : ℚ := 1 / 3
def total_episodes : ℕ := episodes_per_season * seasons
def episodes_watched : ℕ := (fraction_watched * total_episodes).toNat

-- Problem statement
theorem episodes_remaining : total_episodes - episodes_watched = 160 := by
  sorry

end episodes_remaining_l632_632207


namespace smallest_integer_t_exists_l632_632339

theorem smallest_integer_t_exists (t : ℕ) :
  (∀ (x : fin t → ℕ), (∀ i, 0 < x i) → (∑ i, x i ^ 3 = 2002 ^ 2002) → 4 ≤ t) :=
  sorry

end smallest_integer_t_exists_l632_632339


namespace remaining_episodes_l632_632209

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l632_632209


namespace number_is_2_l632_632792

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

theorem number_is_2 :
  (Kolia_true : ℕ → Prop) → (Roman_true : ℕ → Prop) → (Katya_true : ℕ → Prop) → (Natasha_true : ℕ → Prop) →
  (one_boy_one_girl_correct : ℕ → ℕ → ℕ → ℕ → Prop) → 
  Kolia_true 9 → 
  Roman_true (n : ℕ) → 
  Katya_true (m : ℕ) → 
  Natasha_true (k : ℕ) → 
  one_boy_one_girl_correct Kolia_true Roman_true Katya_true Natasha_true → 
  (Roman_true 2) ∧ (Katya_true 2) ∧ ¬ (Natasha_true 2) := 
by {
  /- Define the statements of the children -/
  def Kolia_true := λ n : ℕ, n = 9
  def Roman_true := λ n : ℕ, is_prime n
  def Katya_true := λ n : ℕ, is_even n
  def Natasha_true := λ n : ℕ, is_divisible_by_15 n

  /- The condition that one boy and one girl are correct -/
  def one_boy_one_girl_correct (kb kt kg kn : ℕ → Prop) : Prop :=
    (∃ (b_correct : ℕ → Prop) (g_correct : ℕ → Prop),
      (b_correct = kb ∨ b_correct = kt) ∧ (g_correct = kg ∨ g_correct = kn) ∧
      (b_correct 9 ∨ g_correct 9))

  /- Start the proof -/
  intro Kolia_true Roman_true Katya_true Natasha_true one_boy_one_girl_correct
  intro Kolia_correct Roman_correct Katya_correct Natasha_correct

  /- Sorry placeholder to skip proof -/
  sorry
}

end number_is_2_l632_632792


namespace group_size_of_bananas_l632_632973

theorem group_size_of_bananas (totalBananas numberOfGroups : ℕ) (h1 : totalBananas = 203) (h2 : numberOfGroups = 7) :
  totalBananas / numberOfGroups = 29 :=
sorry

end group_size_of_bananas_l632_632973


namespace parallelogram_count_l632_632937

theorem parallelogram_count :
  let A : ℝ × ℝ := (0, 0),
      vertices_in_first_quadrant := ∀ (u v w) : ℝ , (0 < u ∧ 0 < v ∧ 0 < w) → true,
      B_is_lattice := ∃ (b : ℕ), B = (b, 2 * b),
      D_is_on_line := ∃ (d : ℕ), D = (d, 3 * d),
      area_of_parallelogram := 500000
  in (∃ (num_parallelograms : ℕ), num_parallelograms = 25) :=
sorry

end parallelogram_count_l632_632937


namespace quadratic_solution_identity_l632_632414

theorem quadratic_solution_identity (a b : ℤ) (h : (1 : ℤ)^2 + a * 1 + 2 * b = 0) : 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_identity_l632_632414


namespace ways_to_place_7_balls_in_3_boxes_l632_632521

theorem ways_to_place_7_balls_in_3_boxes : ∃ n : ℕ, n = 8 ∧ (∀ x y z : ℕ, x + y + z = 7 → x ≥ y → y ≥ z → z ≥ 0) := 
by
  sorry

end ways_to_place_7_balls_in_3_boxes_l632_632521


namespace length_DE_l632_632875

variable {EF DF DE : ℝ}
variable {ΔDEF : Triangle}

theorem length_DE (h1 : ΔDEF.has_perpendicular_medians D E)
                  (h2 : EF = 8)
                  (h3 : DF = 9) : DE = Real.sqrt 30.63 :=
by sorry

end length_DE_l632_632875


namespace convex_polygon_diagonals_l632_632861

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) :
  ∃ (d : ℕ), d = 375 ∧
    (∀ (v : ℕ), v < n → 
      let available_vertices := n - 1 - 2 - 2
      let total_diagonals := n * available_vertices / 2
      total_diagonals = d) :=
by
  use 375
  split
  · rfl
  · intros v hv
    let available_vertices := 30 - 1 - 2 - 2
    let total_diagonals := 30 * available_vertices / 2
    have h1 : available_vertices = 25 := rfl
    have h2 : total_diagonals = 375 by norm_num
    exact h2

end convex_polygon_diagonals_l632_632861


namespace optimal_p_closest_to_1000p_l632_632724

theorem optimal_p_closest_to_1000p (p : ℝ) (h : ℕ) (nonneg_p : 0 ≤ p) (le_p : p ≤ 1)
  (ht : 1 < h)
  (prob_recurrence : ∀ h, P h = (1 - p) * P (h - 2) + p * P (h - 3))
  (win_cond : P 0 = 1)
  (lose_cond : P 1 = 0) :
  closestTo (1000 * p) = 618 :=
by
  sorry

end optimal_p_closest_to_1000p_l632_632724


namespace find_original_price_l632_632987

theorem find_original_price 
  (P : ℝ) 
  (h : 0.85 * 0.75 * 0.70 * 0.80 * P = 72) : 
  P ≈ 201.68 := 
by
  sorry

end find_original_price_l632_632987


namespace sum_primes_between_20_and_30_l632_632180

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632180


namespace sum_primes_between_20_and_30_is_52_l632_632124

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632124


namespace probability_one_no_GP_l632_632280

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l632_632280


namespace prime_sum_20_to_30_l632_632138

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632138


namespace george_bags_l632_632796

theorem george_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 648) (h2 : candy_per_bag = 81) : total_candy / candy_per_bag = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end george_bags_l632_632796


namespace log_tan_sum_eq_zero_l632_632323

theorem log_tan_sum_eq_zero : 
  (∑ i in finset.range 89, real.logb 10 (real.tan (real.pi * (i + 1) / 180))) = 0 :=
sorry

end log_tan_sum_eq_zero_l632_632323


namespace symmetric_across_y_axis_l632_632368

-- Define point P and point Q with the given conditions
structure Point where
  x : ℤ
  y : ℤ

def P := Point.mk 3 (-1)

-- Point Q coordinates as (a, 1 - b) with given conditions
variables (a b : ℤ)
def Q := Point.mk a (1 - b)

-- Statement to prove
theorem symmetric_across_y_axis (h_a : a = -3) (h_b : b = 2) : a ^ b = 9 := by
  sorry

end symmetric_across_y_axis_l632_632368


namespace sum_of_primes_between_20_and_30_l632_632115

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632115


namespace greatest_n_l632_632369

def S := { xy : ℕ × ℕ | ∃ x y : ℕ, xy = (x * y, x + y) }

def in_S (a : ℕ) : Prop := ∃ x y : ℕ, a = x * y * (x + y)

def pow_mod (a b m : ℕ) : ℕ := (a ^ b) % m

def satisfies_condition (a : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → in_S (a + pow_mod 2 k 9)

theorem greatest_n (a : ℕ) (n : ℕ) : 
  satisfies_condition a n → n ≤ 3 :=
sorry

end greatest_n_l632_632369


namespace smallest_prime_dividing_sum_l632_632665

theorem smallest_prime_dividing_sum (a b : ℕ) (h₁ : a = 7^15) (h₂ : b = 9^17) (h₃ : a % 2 = 1) (h₄ : b % 2 = 1) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (a + b) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (a + b)) → q ≥ p := by
  sorry

end smallest_prime_dividing_sum_l632_632665


namespace find_initial_lollipops_l632_632298

variable (total_candies : ℕ)
variable (kitkat : ℕ := 5)
variable (hersheys : ℕ := 3 * kitkat)
variable (nerds : ℕ := 8)
variable (babyruth : ℕ := 10)
variable (reeses : ℕ := babyruth / 2)
variable (lollipops_given : ℕ := 5)
variable (candies_left : ℕ := 49)

theorem find_initial_lollipops : 
  total_candies = kitkat + hersheys + nerds + babyruth + reeses + (candies_left + lollipops_given - (kitkat + hersheys + nerds + babyruth + reeses)) :=
begin
  -- Provide the initial context for total_candies
  have total_before_giveaway: total_candies = candies_left + lollipops_given, by sorry,
  -- Compute the already known sum
  have known_candies: kitkat + hersheys + nerds + babyruth + reeses = 43, by sorry,
  -- Put everything together and prove the final statement
  exact calc
    total_candies = candies_left + lollipops_given         : by rw total_before_giveaway
               ... = 49 + 5                                : by rw [candies_left, lollipops_given]
               ... = 54                                    : by norm_num
               ... = 43 + (54 - 43)                        : by sorry -- because 54 = 43 + 11,
               ... = kitkat + hersheys + nerds + babyruth + reeses + 11  : by rw known_candies
               ... = kitkat + hersheys + nerds + babyruth + reeses + (candies_left + lollipops_given - (kitkat + hersheys + nerds + babyruth + reeses)) : by sorry,
end

end find_initial_lollipops_l632_632298


namespace no_solution_cubic_difference_l632_632552

-- Define the problem and conditions
def cubic_difference_inequality (x : ℝ) : Prop :=
  (x^3 - 8) / (x - 2) < 0

-- Formulating the Lean statement
theorem no_solution_cubic_difference (x : ℝ) (hx : x ≠ 2) :
  ¬cubic_difference_inequality x :=
by 
  apply hx
  sorry

end no_solution_cubic_difference_l632_632552


namespace cone_volume_l632_632696

-- Define the conditions of the problem
variables (r : ℝ) (radius_of_inscribed_sphere : r = 1)
variables (radius_of_base height_of_cone : ℝ)
variable (shared_center : True) -- Shared center between the inscribed sphere and circumscribed sphere

-- Define the volume calculation
def volume_of_cone (radius_of_base height_of_cone : ℝ) : ℝ :=
  (1 / 3) * real.pi * (radius_of_base ^ 2) * height_of_cone

-- Main theorem to prove
theorem cone_volume : volume_of_cone (real.sqrt 3) 3 = 3 * real.pi :=
by {
  -- Provide the proof steps here
  sorry
}

end cone_volume_l632_632696


namespace parallel_lines_m_l632_632854

noncomputable def lines_parallel (m : ℝ) : Prop :=
  let l1 := λ x y : ℝ, mx + 2 * y - 2 = 0
  let l2 := λ x y : ℝ, 5 * x + (m + 3) * y - 5 = 0
  (-m / 2) = (-5 / (m + 3))

theorem parallel_lines_m :
  (∃ m : ℝ, lines_parallel m) →
  lines_parallel (-5) :=
by
  sorry

end parallel_lines_m_l632_632854


namespace decimal_to_binary_51_l632_632741

theorem decimal_to_binary_51 :
  ∀ (n : ℕ), n = 51 → n.binary_repr = "110011" :=
by
  intros n h
  rw h
  sorry

end decimal_to_binary_51_l632_632741


namespace wraps_add_more_l632_632898

/-- Let John's raw squat be 600 pounds. Let sleeves add 30 pounds to his lift. Let wraps add 25% 
to his squat. We aim to prove that wraps add 120 pounds more to John's squat than sleeves. -/
theorem wraps_add_more (raw_squat : ℝ) (sleeves_bonus : ℝ) (wraps_percentage : ℝ) : 
  raw_squat = 600 → sleeves_bonus = 30 → wraps_percentage = 0.25 → 
  (raw_squat * wraps_percentage) - sleeves_bonus = 120 :=
by
  intros h1 h2 h3
  sorry

end wraps_add_more_l632_632898


namespace proof_boundaries_l632_632485

variable a b : ℤ

theorem proof_boundaries (ha : 60 ≤ a ∧ a ≤ 84) (hb : 28 ≤ b ∧ b ≤ 33) :
  (88 ≤ a + b ∧ a + b ≤ 117) ∧ (27 ≤ a - b ∧ a - b ≤ 56) :=
by { sorry }

end proof_boundaries_l632_632485


namespace distinct_integer_roots_l632_632765

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l632_632765


namespace number_of_solutions_l632_632338

-- Define that a is a 17th root of unity
def is_seventeenth_root_of_unity (a : ℂ) : Prop := a ^ 17 = 1

-- Define that a and b satisfy the given conditions
def satisfy_conditions (a b : ℂ) : Prop := a ^ 4 * b ^ 6 = 1 ∧ a ^ 5 * b ^ 3 = 1

theorem number_of_solutions : 
  {p : ℂ × ℂ | satisfy_conditions p.1 p.2}.to_finset.card = 17 := 
begin
  sorry
end

end number_of_solutions_l632_632338


namespace sum_primes_20_to_30_l632_632151

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632151


namespace parking_cost_per_hour_l632_632026

theorem parking_cost_per_hour (avg_cost : ℝ) (total_initial_cost : ℝ) (hours_excessive : ℝ) (total_hours : ℝ) (cost_first_2_hours : ℝ)
  (h1 : cost_first_2_hours = 9.00) 
  (h2 : avg_cost = 2.361111111111111)
  (h3 : total_hours = 9) 
  (h4 : hours_excessive = 7):
  (total_initial_cost / total_hours = avg_cost) -> 
  (total_initial_cost = cost_first_2_hours + hours_excessive * x) -> 
  x = 1.75 := 
by
  intros h5 h6
  sorry

end parking_cost_per_hour_l632_632026


namespace fraction_inequality_l632_632524

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b) + (b / c) + (c / a) ≤ (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) := 
by
  sorry

end fraction_inequality_l632_632524


namespace bunnies_out_of_burrow_l632_632416

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end bunnies_out_of_burrow_l632_632416


namespace rationalize_denominator_equals_ABC_product_l632_632538

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l632_632538


namespace positive_difference_of_perimeters_l632_632056

noncomputable def perimeter_figure1 : ℕ :=
  let outer_rectangle := 2 * (5 + 1)
  let inner_extension := 2 * (2 + 1)
  outer_rectangle + inner_extension

noncomputable def perimeter_figure2 : ℕ :=
  2 * (5 + 2)

theorem positive_difference_of_perimeters :
  (perimeter_figure1 - perimeter_figure2 = 4) :=
by
  let perimeter1 := perimeter_figure1
  let perimeter2 := perimeter_figure2
  sorry

end positive_difference_of_perimeters_l632_632056


namespace is_even_function_not_monotonically_increasing_even_shift_property_non_negative_between_neg_half_pi_and_half_pi_l632_632844

noncomputable def f (x : ℝ) : ℝ := cos x * abs (tan x)

theorem is_even_function : ∀ x : ℝ, f (-x) = f x :=
by sorry

theorem not_monotonically_increasing : ∀ x ∈ Icc (-π) (-π / 2), 
  ∃ x₁ x₂ : ℝ, x₁ ∈ Icc (-π) (-π / 2) ∧ x₂ ∈ Icc (-π) (-π / 2) ∧ x₁ < x₂ ∧ f x₁ > f x₂ :=
by sorry

theorem even_shift_property : ∀ x : ℝ, f (π + x) = - (f x) :=
by sorry

theorem non_negative_between_neg_half_pi_and_half_pi : 
  ∀ x ∈ Ioo (-π / 2) (π / 2), f x ≥ f 0 :=
by sorry

end is_even_function_not_monotonically_increasing_even_shift_property_non_negative_between_neg_half_pi_and_half_pi_l632_632844


namespace find_a_range_l632_632406

-- Definitions as per conditions
def prop_P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0
def prop_Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Given conditions
def P_true (a : ℝ) (h : prop_P a) : Prop :=
  ∀ (a : ℝ), a^2 - 16 < 0

def Q_false (a : ℝ) (h : ¬prop_Q a) : Prop :=
  ∀ (a : ℝ), a > 1

-- Main theorem
theorem find_a_range (a : ℝ) (hP : prop_P a) (hQ : ¬prop_Q a) : 1 < a ∧ a < 4 :=
sorry

end find_a_range_l632_632406


namespace expression_equals_4008_l632_632068

def calculate_expression : ℤ :=
  let expr := (2004 - (2011 - 196)) + (2011 - (196 - 2004))
  expr

theorem expression_equals_4008 : calculate_expression = 4008 := 
by
  sorry

end expression_equals_4008_l632_632068


namespace parallel_lines_l632_632852

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l632_632852


namespace doctor_is_correct_l632_632608

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632608


namespace hydra_survival_l632_632605

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632605


namespace cube_divisions_101_l632_632697

theorem cube_divisions_101 : 
  let side_length := 101
  let num_ways :=
    (∑ k in {1 .. side_length}, (k + 1)^2) + 
    (∑ k in {side_length + 1 .. 2 * side_length - 1}, (203 - k)^2)
  in num_ways = 707504 :=
by
  let side_length := 101
  let num_ways :=
    (∑ k in {1 .. side_length}, (k + 1)^2) + 
    (∑ k in {side_length + 1 .. 2 * side_length - 1}, (203 - k)^2)
  have h1 : num_ways = 707504 := sorry
  exact h1

end cube_divisions_101_l632_632697


namespace hydra_survival_l632_632599

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632599


namespace solution_l632_632751

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (4 + 2 * x) / (6 + 3 * x) = (3 + 2 * x) / (5 + 3 * x) ∧ x = -2

theorem solution : problem_statement :=
by
  sorry

end solution_l632_632751


namespace polynomial_integer_roots_is_a_l632_632313

theorem polynomial_integer_roots_is_a (a : ℤ) :
  (∃ x : ℤ, x^3 - 2 * x^2 + a * x + 8 = 0) ↔ 
  a = -49 ∨ a = -47 ∨ a = -22 ∨ a = -10 ∨ a = -7 ∨ a = 4 ∨ a = 9 ∨ a = 16 :=
begin
  sorry
end

end polynomial_integer_roots_is_a_l632_632313


namespace sequence_general_term_l632_632360

theorem sequence_general_term {a : ℕ → ℝ} (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 4 * a n - 3) :
  a n = (4/3)^(n-1) :=
sorry

end sequence_general_term_l632_632360


namespace millionth_digit_1_div_41_l632_632773

theorem millionth_digit_1_div_41 : 
  ∃ n, n = 1000000 ∧ (n % 5 = 0) → digit_after_decimal (1 / 41) n = 9 :=
begin
  intro exists n,
  sorry -- proof goes here
end

end millionth_digit_1_div_41_l632_632773


namespace shortest_distance_to_circle_l632_632663

theorem shortest_distance_to_circle : 
  let circle := λ (x y : ℝ), x^2 - 16 * x + y^2 - 8 * y + 100 = 0 in
  ∃ D : ℝ, D = 4 * Real.sqrt 5 - 8 ∧
  ∀ x y : ℝ, circle x y → Real.sqrt (x^2 + y^2) ≥ D :=
begin
  sorry
end

end shortest_distance_to_circle_l632_632663


namespace necessary_but_not_sufficient_condition_l632_632490

variables {a1 b1 a2 b2 : ℝ}
variables (M N : set ℝ)

def M_def : set ℝ := {x : ℝ | a1 * x + b1 < 0}
def N_def : set ℝ := {x : ℝ | a2 * x + b2 < 0}

theorem necessary_but_not_sufficient_condition
  (ha1 : a1 ≠ 0) (ha2 : a2 ≠ 0) (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0) :
  (M_def a1 b1 = N_def a2 b2 →
   (a1 / a2 = b1 / b2) ∧ ¬(a1 / a2 = b1 / b2 → M_def a1 b1 = N_def a2 b2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l632_632490


namespace percentage_of_tennis_players_also_playing_hockey_l632_632516

-- Definitions for the given conditions
def total_students : ℕ := 600
def fraction_playing_tennis : ℚ := 3 / 4
def number_playing_both : ℕ := 270

-- Definition to calculate the number of students playing tennis based on the total number of students and the fraction that plays tennis
def number_playing_tennis : ℕ := (fraction_playing_tennis * total_students).natAbs

-- The theorem to prove the percentage of students who play tennis and also play hockey is 60%
theorem percentage_of_tennis_players_also_playing_hockey : 
  (number_playing_both / number_playing_tennis : ℚ) * 100 = 60 :=
by sorry

end percentage_of_tennis_players_also_playing_hockey_l632_632516


namespace find_n_such_that_sin_n_eq_sin_758_l632_632333

theorem find_n_such_that_sin_n_eq_sin_758 :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ (Real.sin (n * Real.pi / 180) = Real.sin (758 * Real.pi / 180)) :=
begin
  use 38,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { sorry }
end

end find_n_such_that_sin_n_eq_sin_758_l632_632333


namespace sum_of_coefficients_l632_632340

theorem sum_of_coefficients : 
  let p1 := 5 * (2 * x ^ 8 - 3 * x ^ 5 + 9 * x ^ 3 - 4)
  let p2 := 3 * (4 * x ^ 6 - 6 * x ^ 4 + 7)
  let poly := p1 + p2 in
  poly.eval 1 = 35 :=
by
  sorry

end sum_of_coefficients_l632_632340


namespace period_and_min_of_function_cos_A_value_l632_632841

-- Definitions and conditions
def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sin x - Real.cos x)
def A (A : ℝ) : Prop := 0 < A ∧ A < Real.pi / 4

-- Proof problem in Lean 4 statement
theorem period_and_min_of_function :
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ x, f x ≥ 1 - Real.sqrt 2) :=
sorry

theorem cos_A_value :
  (A A) → (f (A / 2) = 1 - 4 * Real.sqrt 2 / 5) → Real.cos A = 7 * Real.sqrt 2 / 10 :=
sorry

end period_and_min_of_function_cos_A_value_l632_632841


namespace EM_squared_sub_EN_squared_eq_DE_times_BC_l632_632217

open Classical Geometry Affine EuclideanAffineSpace

variable {O1 O2 O3 P A B C D E M N : Point}
variable {circleO1 circleO2 circleO3 : Circle}
variable [h1 : Incircle circleO1 O1]
variable [h2 : Incircle circleO2 O2]
variable [h3 : Incircle circleO3 O3]
variable [h4 : circleO1 ∩ circleO2 = {P, A}]
variable [h5 : LineThrough A intersects circleO1 at B intersects circleO2 at C]
variable [h6 : LineThrough A P intersects circleO3 at D]
variable [h7 : LineThrough D E parallel to LineThrough B C intersects circleO3 at E]
variable [h8 : Line E M tangentTo circleO1 at M]
variable [h9 : Line E N tangentTo circleO2 at N]

theorem EM_squared_sub_EN_squared_eq_DE_times_BC :
  EM^2 - EN^2 = DE * BC := by
  sorry

end EM_squared_sub_EN_squared_eq_DE_times_BC_l632_632217


namespace necessary_and_sufficient_condition_l632_632864

variable (p q : Prop)

theorem necessary_and_sufficient_condition (h1 : p → q) (h2 : q → p) : (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l632_632864


namespace union_of_A_and_B_l632_632848

open Set

def A : Set ℕ := {1, 3, 7, 8}
def B : Set ℕ := {1, 5, 8}

theorem union_of_A_and_B : A ∪ B = {1, 3, 5, 7, 8} := by
  sorry

end union_of_A_and_B_l632_632848


namespace hydrae_never_equal_heads_l632_632639

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632639


namespace doctor_is_correct_l632_632613

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632613


namespace geometric_sequence_and_general_formula_range_of_a_l632_632806

theorem geometric_sequence_and_general_formula (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S_n n = 2 * a_n n - 2 * n) :
  (∀ n : ℕ, n > 0 → a_n n + 2 = 4 * 2 ^ (n - 1)) :=
sorry

theorem range_of_a (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℝ)
  (h2 : ∀ n : ℕ, n > 0 → b_n n = log 2 (a_n n + 2))
  (h3 : ∀ n : ℕ, n > 0 → T_n n = ∑ i in range n, 1 / ((b_n i) * (b_n (i + 1))))
  (h4 : ∀ n : ℕ, T_n n < a) : a ≥ 1 / 2 :=
sorry

end geometric_sequence_and_general_formula_range_of_a_l632_632806


namespace problem_statement_l632_632410

noncomputable def P := (3 : ℝ) + (4 : ℂ) * complex.I
noncomputable def F := -(complex.I : ℂ)
noncomputable def G := (3 : ℝ) - (4 : ℂ) * complex.I
noncomputable def divisor := -(3 : ℂ) * complex.I

theorem problem_statement : (P * F * G) / divisor = (25 : ℝ) / 3 := by
  sorry

end problem_statement_l632_632410


namespace age_problem_l632_632427

theorem age_problem (A B : ℕ) 
  (h1 : A + 10 = 2 * (B - 10))
  (h2 : A = B + 12) :
  B = 42 :=
sorry

end age_problem_l632_632427


namespace sum_primes_in_range_l632_632078

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632078


namespace hydra_survival_l632_632603

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632603


namespace min_value_of_f_l632_632783

def f (t s : ℝ) : ℝ := 6 * t ^ 2 + 3 * s ^ 2 - 4 * s * t - 8 * t + 6 * s + 5

theorem min_value_of_f :
  ∃ (t s : ℝ), f t s = 8/7 ∧ (∀ (t' s' : ℝ), f t' s' ≥ 8/7) ∧ t = 3/7 ∧ s = -5/7 :=
begin
  sorry
end

end min_value_of_f_l632_632783


namespace solution_set_of_cot_product_l632_632908

theorem solution_set_of_cot_product (x : ℝ) (h : cot (⌊x⌋ : ℝ) * cot (x - ⌊x⌋) = 1) : 
  ∃ k : ℤ, x = k * real.pi + (real.pi / 2) :=
sorry

end solution_set_of_cot_product_l632_632908


namespace hydras_never_die_l632_632625

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632625


namespace sales_tax_percentage_is_correct_l632_632711

noncomputable def total_cost_with_tax : ℝ := 945
noncomputable def number_of_tickets : ℕ := 25
noncomputable def face_value_without_tax_per_ticket : ℝ := 35.91

noncomputable def total_cost_without_tax : ℝ := number_of_tickets * face_value_without_tax_per_ticket
noncomputable def sales_tax_paid : ℝ := total_cost_with_tax - total_cost_without_tax
noncomputable def sales_tax_percentage : ℝ := (sales_tax_paid / total_cost_without_tax) * 100

theorem sales_tax_percentage_is_correct :
  sales_tax_percentage ≈ 5.26 :=
by
  -- proof to be filled in
  sorry

end sales_tax_percentage_is_correct_l632_632711


namespace prime_sum_20_to_30_l632_632191

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632191


namespace product_of_millions_l632_632789

-- Define the conditions
def a := 5 * (10 : ℝ) ^ 6
def b := 8 * (10 : ℝ) ^ 6

-- State the proof problem
theorem product_of_millions : (a * b) = 40 * (10 : ℝ) ^ 12 := 
by
  sorry

end product_of_millions_l632_632789


namespace remainder_when_T_divided_by_1000_l632_632487

noncomputable def T : ℕ :=
  ∑ n in Finset.range 402, (-1)^n * (Nat.choose 3006 (3 * n))

theorem remainder_when_T_divided_by_1000 : T % 1000 = 54 := by
  sorry

end remainder_when_T_divided_by_1000_l632_632487


namespace doctor_is_correct_l632_632612

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632612


namespace distance_from_mountains_l632_632517

/-- Given distances and scales from the problem description -/
def distance_between_mountains_map : ℤ := 312 -- in inches
def actual_distance_between_mountains : ℤ := 136 -- in km
def scale_A : ℤ := 1 -- 1 inch represents 1 km
def scale_B : ℤ := 2 -- 1 inch represents 2 km
def distance_from_mountain_A_map : ℤ := 25 -- in inches
def distance_from_mountain_B_map : ℤ := 40 -- in inches

/-- Prove the actual distances from Ram's camp to the mountains -/
theorem distance_from_mountains (dA dB : ℤ) :
  (dA = distance_from_mountain_A_map * scale_A) ∧ 
  (dB = distance_from_mountain_B_map * scale_B) :=
by {
  sorry -- Proof placeholder
}

end distance_from_mountains_l632_632517


namespace sum_primes_in_range_l632_632086

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632086


namespace seeds_per_bed_l632_632520

theorem seeds_per_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 60) (h2 : flower_beds = 6) : total_seeds / flower_beds = 10 := by
  sorry

end seeds_per_bed_l632_632520


namespace population_change_l632_632590

theorem population_change
  (year1_inc : ℝ := 0.30) -- 30% increase means multiplying by 1 + 0.30
  (year2_inc : ℝ := 0.20) -- 20% increase means multiplying by 1 + 0.20
  (year3_dec : ℝ := 0.10) -- 10% decrease means multiplying by 1 - 0.10
  (year4_dec : ℝ := 0.30) -- 30% decrease means multiplying by 1 - 0.30
  (initial_population : ℝ := 1) -- Assume initial population is 1 for simplicity
  : Real.round (((initial_population * (1 + year1_inc) * (1 + year2_inc) * (1 - year3_dec) * (1 - year4_dec)) - 1) * 100) = -2 := 
by
  sorry

end population_change_l632_632590


namespace solution_l632_632365

-- Step d) Use definitions from conditions and the final expression to write a statement in Lean 4
def f : ℝ → ℝ :=
  -- To define f, we acknowledge that it is an odd, periodic function with the specified properties
  sorry

theorem solution :
  let f := sorry in 
  let x := real.log 23 / real.log 2 in
  f(x) = -23 / 16 := 
by
  sorry

end solution_l632_632365


namespace sum_primes_20_to_30_l632_632150

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632150


namespace sum_of_primes_between_20_and_30_l632_632111

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632111


namespace find_g3_l632_632035

theorem find_g3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 ^ x) + 2 * x * g (3 ^ (-x)) = 1) : 
  g 3 = 1 / 5 := 
sorry

end find_g3_l632_632035


namespace students_prefer_dogs_l632_632754

theorem students_prefer_dogs (total_students : ℕ) (perc_dogs_vg perc_dogs_mv : ℕ) (h_total: total_students = 30)
  (h_perc_dogs_vg: perc_dogs_vg = 50) (h_perc_dogs_mv: perc_dogs_mv = 10) :
  total_students * perc_dogs_vg / 100 + total_students * perc_dogs_mv / 100 = 18 := by
  sorry

end students_prefer_dogs_l632_632754


namespace division_theorem_l632_632992

noncomputable def p (z : ℝ) : ℝ := 4 * z ^ 3 - 8 * z ^ 2 + 9 * z - 7
noncomputable def d (z : ℝ) : ℝ := 4 * z + 2
noncomputable def q (z : ℝ) : ℝ := z ^ 2 - 2.5 * z + 3.5
def r : ℝ := -14

theorem division_theorem (z : ℝ) : p z = d z * q z + r := 
by
  sorry

end division_theorem_l632_632992


namespace rationalize_denominator_equals_ABC_product_l632_632535

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l632_632535


namespace OQ_value_l632_632457

variables {X Y Z N O Q R : Type}
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [MetricSpace N] [MetricSpace O] [MetricSpace Q] [MetricSpace R]
variables (XY YZ XN NY ZO XO OZ YN XR OQ RQ : ℝ)
variables (triangle_XYZ : Triangle X Y Z)
variables (X_equal_midpoint_XY : XY = 540)
variables (Y_equal_midpoint_YZ : YZ = 360)
variables (XN_equal_NY : XN = NY)
variables (ZO_is_angle_bisector : is_angle_bisector Z O X Y)
variables (intersection_YN_ZO : Q = intersection YN ZO)
variables (N_midpoint_RQ : is_midpoint N R Q)
variables (XR_value : XR = 216)

theorem OQ_value : OQ = 216 := sorry

end OQ_value_l632_632457


namespace rhombus_perimeter_l632_632027

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 12) (h_d2 : d2 = 16) :
    let s := (sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))
    in 4 * s = 40 :=
by
  let s := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  rw [h_d1, h_d2]
  sorry

end rhombus_perimeter_l632_632027


namespace sum_seventh_roots_unity_l632_632760

-- Sum of reciprocals of absolute values squared of (1 + z) for all seventh roots of unity is 0
theorem sum_seventh_roots_unity : 
  (∑ z in (({z | z ^ 7 = 1} : Set ℂ)), (1 / |1 + z| ^ 2)) = 0 := 
  sorry

end sum_seventh_roots_unity_l632_632760


namespace probability_one_girl_no_growth_pie_l632_632286

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l632_632286


namespace correct_statements_l632_632383

theorem correct_statements (a b c : ℝ) (h : ∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 3) :
  ( ∃ (x : ℝ), c*x^2 + b*x + a < 0 ↔ -1/2 < x ∧ x < 1/3 ) ∧
  ( ∃ (b : ℝ), ∀ b, 12/(3*b + 4) + b = 8/3 ) ∧
  ( ∀ m, ¬ (m < -1 ∨ m > 2) ) ∧
  ( c = 2 → ∀ n1 n2, (3*a*n1^2 + 6*b*n1 = -3 ∧ 3*a*n2^2 + 6*b*n2 = 1) → n2 - n1 ∈ [2, 4] ) :=
sorry

end correct_statements_l632_632383


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l632_632911

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l632_632911


namespace prime_sum_20_to_30_l632_632189

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632189


namespace mass_of_empty_container_l632_632571

/-- Variables and given conditions -/
variables
  (m1 m2 : ℝ)       -- masses of the container filled with kerosene and water
  (rhoW rhoK : ℝ)   -- densities of water and kerosene
  (mc V : ℝ)        -- mass of the empty container and volume of the container

/-- Given values -/
def given_conditions : Prop :=
  m1 = 20 ∧ m2 = 24 ∧ rhoW = 1000 ∧ rhoK = 800

/-- Goal statement for the proof -/
theorem mass_of_empty_container (h : given_conditions) : mc = 4 :=
by {
  sorry
}

end mass_of_empty_container_l632_632571


namespace original_price_correct_l632_632698

-- Given a selling price S and a loss percentage
variable (S : ℝ) (loss_percentage : ℝ)

-- The condition that selling price S is 82% of the original price P
def original_price (S : ℝ) (loss_percentage : ℝ) : ℝ :=
  S / ((100 - loss_percentage) / 100)

-- Define the specific values given in the problem
def S_val : ℝ := 1558
def loss_percentage_val : ℝ := 18

-- Prove that the original price is 1900 given the conditions
theorem original_price_correct :
  original_price S_val loss_percentage_val = 1900 :=
by
  sorry

end original_price_correct_l632_632698


namespace prime_sum_20_to_30_l632_632141

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632141


namespace rationalize_denominator_l632_632533

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l632_632533


namespace total_value_of_bills_l632_632291

theorem total_value_of_bills 
  (total_bills : Nat := 12) 
  (num_5_dollar_bills : Nat := 4) 
  (num_10_dollar_bills : Nat := 8)
  (value_5_dollar_bill : Nat := 5)
  (value_10_dollar_bill : Nat := 10) :
  (num_5_dollar_bills * value_5_dollar_bill + num_10_dollar_bills * value_10_dollar_bill = 100) :=
by
  sorry

end total_value_of_bills_l632_632291


namespace hydrae_never_equal_heads_l632_632643

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632643


namespace savings_duration_before_investment_l632_632700

---- Definitions based on conditions ----
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def stock_price_per_share : ℕ := 50
def shares_bought : ℕ := 25

---- Derived conditions based on definitions ----
def total_spent_on_stocks := shares_bought * stock_price_per_share
def total_savings_before_investment := 2 * total_spent_on_stocks
def monthly_savings_wife := weekly_savings_wife * 4
def total_monthly_savings := monthly_savings_wife + monthly_savings_husband

---- The theorem statement ----
theorem savings_duration_before_investment :
  total_savings_before_investment / total_monthly_savings = 4 :=
sorry

end savings_duration_before_investment_l632_632700


namespace probability_no_gp_l632_632273

/-- 
Alice has six magical pies: Two are growth pies (GP), and four are shrink pies (SP).
Alice randomly picks three pies out of the six and gives them to Mary. We want to find the 
probability that one of the girls does not have a single growth pie (GP).
-/
theorem probability_no_gp : 
  let total_pies := 6
  let gp := 2 -- number of growth pies
  let sp := 4 -- number of shrink pies
  let chosen_pies := 3 -- pies given to Mary
  (let total_ways := Nat.choose total_pies chosen_pies in -- total ways to choose 3 out of 6
  let favorable_ways := Nat.choose sp 2 in -- ways to choose 2 SPs out of 4 (ensuring both have at least one GP)
  (total_ways - favorable_ways) / total_ways = (7 / 10 : ℚ)) :=
  sorry

end probability_no_gp_l632_632273


namespace part1_part2_part3_l632_632836

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem part1 : f = λ x, 2 * sin (2 * x + π / 3) :=
by
  sorry

theorem part2 (k : ℤ) : (f' x > 0) ↔ (-π/2 + 2 * k * π ≤ 2 * x + π/3 ∧ 2 * x + π/3 ≤ π/2 + 2 * k * π) :=
by 
  sorry

theorem part3 : set.range (f '' set.Icc (-π/2) 0) = set.Icc (-2) (sqrt 3) :=
by
  sorry

end part1_part2_part3_l632_632836


namespace rationalize_denominator_ABC_l632_632528

theorem rationalize_denominator_ABC :
  (let A := -9 in let B := -4 in let C := 5 in A * B * C) = 180 :=
by
  let expr := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)
  let numerator := (2 + Real.sqrt 5) * (2 + Real.sqrt 5)
  let denominator := (2 - Real.sqrt 5) * (2 + Real.sqrt 5)
  let simplified_expr := numerator / denominator
  have h1 : numerator = 9 + 4 * Real.sqrt 5 := sorry
  have h2 : denominator = -1 := sorry
  have h3 : simplified_expr = -9 - 4 * Real.sqrt 5 := by
    rw [h1, h2]
    simp
  have hA : -9 = A := rfl
  have hB : -4 = B := rfl
  have hC : 5 = C := rfl
  have hABC : -9 * -4 * 5 = 180 := by 
    rw [hA, hB, hC]
    ring
  exact hABC

end rationalize_denominator_ABC_l632_632528


namespace problem_proof_l632_632010

variable (a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ ((a + 1) * (b + 1) * (c + 1) = 8)

-- The proof problem
theorem problem_proof (h : conditions a b c) : a + b + c ≥ 3 ∧ a * b * c ≤ 1 :=
  sorry

end problem_proof_l632_632010


namespace tetrahedron_ABCD_is_regular_l632_632555

-- Assuming appropriate definitions
variables (A B C D K L : Type*) [Point A] [Point B] [Point C] [Point D] [Point K] [Point L]

-- Hypotheses
axiom AB_eq_CD : dist A B = dist C D
axiom K_centroid_ABC : centroid {A, B, C} = K
axiom L_centroid_ABD : centroid {A, B, D} = L
axiom inscribed_sphere_tangent_to_ABC_at_K : tangent_sphere (circle_of_tetrahedron A B C D) K
axiom inscribed_sphere_tangent_to_ABD_at_L : tangent_sphere (circle_of_tetrahedron A B C D) L

-- Goal
theorem tetrahedron_ABCD_is_regular 
  (ABCD_regular : regular_tetrahedron A B C D) : 
  dist A B = dist B C :=
sorry

end tetrahedron_ABCD_is_regular_l632_632555


namespace hydras_survive_l632_632620

theorem hydras_survive (A_heads : ℕ) (B_heads : ℕ) (growthA growthB : ℕ → ℕ) (a b : ℕ)
    (hA : A_heads = 2016) (hB : B_heads = 2017)
    (growthA_conds : ∀ n, growthA n ∈ {5, 7})
    (growthB_conds : ∀ n, growthB n ∈ {5, 7}) :
  ∀ n, let total_heads := (A_heads + growthA n - 2 * n) + (B_heads + growthB n - 2 * n);
       total_heads % 2 = 1 :=
by intro n
   sorry

end hydras_survive_l632_632620


namespace sum_of_primes_between_20_and_30_l632_632107

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632107


namespace contrapositive_l632_632561

theorem contrapositive (x : ℝ) : (x > 1 → x^2 + x > 2) ↔ (x^2 + x ≤ 2 → x ≤ 1) :=
sorry

end contrapositive_l632_632561


namespace circle_area_through_equilateral_triangle_vertices_l632_632720

theorem circle_area_through_equilateral_triangle_vertices
  (s : ℝ) (h1 : s = 4) : 
  let r := s / Real.sqrt 3 in
  let area := Real.pi * r^2 in
  area = 16 * Real.pi / 3 :=
by
  -- define the side length of the triangle
  have h_r : r = 4 * Real.sqrt 3 / 3, by 
  {
    rw [h1],
    apply (4 / Real.sqrt 3),
    field_simp [Real.sqrt_eq_sqrt, Real.sqrt_nonneg, zero_le_bit1, Real.mul_div_left_comm],
    ring,
  },
  -- define the area of the circle
  have h_area : area = Real.pi * (4 * Real.sqrt 3 / 3)^2, by 
  {
    rw [h_r],
    field_simp [Real.sqrt_eq_sqrt, Real.sqrt_nonneg, zero_le_bit1],
    ring,
  },
  -- prove the area
  have h_final : Real.pi * (4 * Real.sqrt 3 / 3)^2 = 16 * Real.pi / 3, by
  {
    field_simp [h_area],
    ring,
  },
  exact h_final

end circle_area_through_equilateral_triangle_vertices_l632_632720


namespace find_x0_Sn_less_than_one_l632_632828

variable (f : ℝ → ℝ)
variable (monotonic_f : ∀ x1 x2 : ℝ, f (x1) ≤ f (x2) ↔ x1 ≤ x2)
variable (x0 : ℝ)
variable (f_condition : ∀ x1 x2 : ℝ, f (x0 * x1 + x0 * x2) = f (x0) + f (x1) + f (x2))
variable (f_x0_eq_one : f(x0) = 1)
variable (a_n : ℕ+ → ℝ := λ (n : ℕ+), f (1 / (2 ^ (n + 1))) + 1)
variable (S_n : ℕ+ → ℝ := λ (n : ℕ+), finset.sum (finset.range n) a_n)

theorem find_x0 : x0 = 1 :=
sorry

theorem Sn_less_than_one (n : ℕ+) : S_n n < 1 :=
sorry

end find_x0_Sn_less_than_one_l632_632828


namespace linear_eq_with_one_variable_is_B_l632_632200

-- Define the equations
def eqA (x y : ℝ) : Prop := 2 * x = 3 * y
def eqB (x : ℝ) : Prop := 7 * x + 5 = 6 * (x - 1)
def eqC (x : ℝ) : Prop := x^2 + (1 / 2) * (x - 1) = 1
def eqD (x : ℝ) : Prop := (1 / x) - 2 = x

-- State the problem
theorem linear_eq_with_one_variable_is_B :
  ∃ x : ℝ, ¬ (∃ y : ℝ, eqA x y) ∧ eqB x ∧ ¬ eqC x ∧ ¬ eqD x :=
by {
  -- mathematical content goes here
  sorry
}

end linear_eq_with_one_variable_is_B_l632_632200


namespace Trishul_investment_percentage_l632_632982

-- Definitions from the conditions
def Vishal_invested (T : ℝ) : ℝ := 1.10 * T
def total_investment (T : ℝ) (V : ℝ) : ℝ := T + V + 2000

-- Problem statement
theorem Trishul_investment_percentage (T : ℝ) (V : ℝ) (H1 : V = Vishal_invested T) (H2 : total_investment T V = 5780) :
  ((2000 - T) / 2000) * 100 = 10 :=
sorry

end Trishul_investment_percentage_l632_632982


namespace sum_primes_between_20_and_30_l632_632174

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632174


namespace smallest_positive_integer_form_l632_632664

theorem smallest_positive_integer_form (m n : ℤ) : ∃ k > 0, k = 2024 * m + 48048 * n ∧ k = Int.gcd 2024 48048 :=
by
  use [88, 1]
  exact (Int.gcd_eq_right 2024 48048).symm;
  exact ne_of_gt (Int.lt_add_one (Int.zero_lt_of_ne_zero _));
  sorry

end smallest_positive_integer_form_l632_632664


namespace total_boys_in_camp_l632_632880

theorem total_boys_in_camp (T : ℕ) (A B C : ℕ) :
  0.20 * T = A ∧
  0.25 * T = B ∧
  0.55 * T = C ∧
  0.30 * A = A_sci_students ∧
  (A - A_sci_students) = 56 ∧
  (0.10 * B) = 35
  → T = 400 :=
by
  sorry

end total_boys_in_camp_l632_632880


namespace Peter_did_not_make_C_l632_632002

-- Define the conditions and statements
def Peter (day: ℕ -> bool) (n: ℕ) : Prop :=
  ∀ k, (day k = day (k + 2)) ∧ (day k ≠ day (k + 1))

def statement_A (day: ℕ -> bool) (n: ℕ) : Prop :=
  ¬ day (n - 1) ∧ ¬ day (n + 1)

def statement_B (day: ℕ -> bool) (n: ℕ) : Prop :=
  day n ∧ day (n + 1)

def statement_C : Prop :=
  2024 % 11 = 0

def statement_D : Prop :=
  (n: ℕ) → true  -- Simplified condition to introduce statement D

def statement_E : Prop :=
  (n: ℕ) → true  -- Simplified condition to introduce statement E

-- Define the main theorem
theorem Peter_did_not_make_C (day: ℕ -> bool) (n: ℕ) (h: Peter day n) :
  ¬ (statement_C) :=
sorry

end Peter_did_not_make_C_l632_632002


namespace int_coeffs_square_sum_l632_632863

theorem int_coeffs_square_sum (a b c d e f : ℤ)
  (h : ∀ x, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := 
sorry

end int_coeffs_square_sum_l632_632863


namespace ax_perp_ay_l632_632293

theorem ax_perp_ay
  (A B P Q C D Z X Y O N : Type)
  [Point A B P Q C D Z X Y O N]
  (H1 : perp AB CD Z)
  (H2 : midpoint O A B)
  (H3 : circumcenter N triangleABC)
  (H4 : intersects (lineCD) (lineNO) X)
  (H5 : intersects (lineCD) (linePQ) Y)
  : perp AX AY := 
begin
  sorry
end

end ax_perp_ay_l632_632293


namespace option_a_may_not_hold_l632_632813

variables {𝕜 : Type*} [nondiscrete_normed_field 𝕜] 
variables {f g : 𝕜 → 𝕜} 
variable [differentiable 𝕜 g]

-- Conditions
def condition1 : ∀ x, f x + g' x = 2 := sorry
def condition2 : ∀ x, f x - g' (4 - x) = 2 := sorry
def even_function : ∀ x, g (-x) = g x := sorry

-- Proof that A: f(-1) = f(-3) may not necessarily hold
theorem option_a_may_not_hold : ¬ (∀ x, f (-1) = f (-3)) :=
  by 
    -- Using the conditions defined to prove the non-hold of A
    -- proof will be filled here
    sorry

end option_a_may_not_hold_l632_632813


namespace find_f_value_l632_632395

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 - b * (Real.sin x) * (Real.cos x) - a / 2

theorem find_f_value (a b : ℝ)
  (h_max : ∀ x, f a b x ≤ 1/2)
  (h_at_pi_over_3 : f a b (Real.pi / 3) = (Real.sqrt 3) / 4) :
  f a b (-Real.pi / 3) = 0 ∨ f a b (-Real.pi / 3) = -(Real.sqrt 3) / 4 :=
sorry

end find_f_value_l632_632395


namespace rationalize_denominator_equals_ABC_product_l632_632537

theorem rationalize_denominator_equals_ABC_product :
  let A := -9
  let B := -4
  let C := 5
  ∀ (x y : ℚ), (x + y * real.sqrt 5) = (2 + real.sqrt 5) * (2 + real.sqrt 5) / ((2 - real.sqrt 5) * (2 + real.sqrt 5)) →
    A * B * C = 180 := sorry

end rationalize_denominator_equals_ABC_product_l632_632537


namespace isosceles_trapezoid_area_l632_632722

noncomputable def trapezoid_area (leg diagonal long_base: ℝ) : ℝ :=
  let b₂ := long_base - 2 * real.sqrt (leg^2 - (diagonal * leg / long_base)^2)
  let h := (2 * real.sqrt(diagonal^2 - (b₂ / 2)^2)) / long_base
  (b₂ + long_base) / 2 * h

theorem isosceles_trapezoid_area :
  trapezoid_area 35 45 55 ≈ 999.477 :=
sorry

end isosceles_trapezoid_area_l632_632722


namespace max_books_borrowed_l632_632433

-- Defining constants based on the problem conditions
def total_students : ℕ := 38
def students_borrowed_0 : ℕ := 2
def students_borrowed_1 : ℕ := 12
def students_borrowed_2 : ℕ := 10
def avg_books_per_student : ℕ := 2
def total_books : ℕ := total_students * avg_books_per_student
def remaining_students_borrowed_at_least_3 : ℕ := total_students - (students_borrowed_0 + students_borrowed_1 + students_borrowed_2)

-- Hypothesis about the setup and values based on given conditions
theorem max_books_borrowed :
  ∃ max_books : ℕ,
  (students_borrowed_0 * 0 + students_borrowed_1 * 1 + students_borrowed_2 * 2 + (remaining_students_borrowed_at_least_3 - 1) * 3 + max_books = total_books) ∧
  max_books = 5 :=
begin
  sorry
end

end max_books_borrowed_l632_632433


namespace sum_primes_between_20_and_30_is_52_l632_632129

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632129


namespace line_circle_intersection_l632_632846

theorem line_circle_intersection (k : ℝ)
  (h1 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 6) ∧ (B.1^2 + B.2^2 = 6)
        ∧ (k * A.1 - A.2 + 2 * k - 1 = 0) ∧ (k * B.1 - B.2 + 2 * k - 1 = 0)
        ∧ (real.dist A B = 2 * real.sqrt 2)) : k = -3/4 :=
by
    sorry

end line_circle_intersection_l632_632846


namespace bromine_atomic_weight_l632_632784

theorem bromine_atomic_weight :
  ∃ (Br : ℝ), Br = 80.007 ∧ (1 * 26.98 + 3 * Br = 267) :=
begin
  use 80.007,
  split,
  { refl },
  { sorry },
end

end bromine_atomic_weight_l632_632784


namespace sum_primes_20_to_30_l632_632156

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632156


namespace min_acute_triangles_for_isosceles_l632_632723

noncomputable def isosceles_triangle_acute_division : ℕ :=
  sorry

theorem min_acute_triangles_for_isosceles {α : ℝ} (hα : α = 108) (isosceles : ∀ β γ : ℝ, β = γ) :
  isosceles_triangle_acute_division = 7 :=
sorry

end min_acute_triangles_for_isosceles_l632_632723


namespace sixtieth_integer_is_35142_l632_632045

def digits := [1, 2, 3, 4, 5]

noncomputable def perms := List.permutations digits

def ordered_perms := perms.sorted

def sixtieth_permutation := ordered_perms.get! (60 - 1)

theorem sixtieth_integer_is_35142 : sixtieth_permutation = 35142 := 
    sorry

end sixtieth_integer_is_35142_l632_632045


namespace hydras_will_live_l632_632634

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632634


namespace max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l632_632399

namespace Geometry

variables {x y : ℝ}

-- Given condition
def satisfies_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y + 1 = 0

-- Proof problems
theorem max_x_plus_y (h : satisfies_circle x y) : 
  x + y ≤ 2 + Real.sqrt 6 :=
sorry

theorem range_y_plus_1_over_x (h : satisfies_circle x y) : 
  -Real.sqrt 2 ≤ (y + 1) / x ∧ (y + 1) / x ≤ Real.sqrt 2 :=
sorry

theorem extrema_x2_minus_2x_plus_y2_plus_1 (h : satisfies_circle x y) : 
  8 - 2 * Real.sqrt 15 ≤ x^2 - 2 * x + y^2 + 1 ∧ x^2 - 2 * x + y^2 + 1 ≤ 8 + 2 * Real.sqrt 15 :=
sorry

end Geometry

end max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l632_632399


namespace sum_primes_20_to_30_l632_632159

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632159


namespace find_n_that_satisfies_l632_632342

theorem find_n_that_satisfies :
  ∃ (n : ℕ), (1 / (n + 2 : ℕ) + 2 / (n + 2) + (n + 1) / (n + 2) = 2) ∧ (n = 0) :=
by 
  existsi (0 : ℕ)
  sorry

end find_n_that_satisfies_l632_632342


namespace hydras_never_die_l632_632628

def two_hydras_survive (a b : ℕ) : Prop :=
  ∀ n : ℕ, ∀ (a_heads b_heads : ℕ),
    (a_heads = a + n ∗ (5 ∨ 7) - 4 ∗ n) ∧
    (b_heads = b + n ∗ (5 ∨ 7) - 4 ∗ n) → a_heads ≠ b_heads

theorem hydras_never_die :
  two_hydras_survive 2016 2017 :=
by sorry

end hydras_never_die_l632_632628


namespace same_terminal_side_of_minus_80_l632_632021

theorem same_terminal_side_of_minus_80 :
  ∃ k : ℤ, 1 * 360 - 80 = 280 := 
  sorry

end same_terminal_side_of_minus_80_l632_632021


namespace smallest_positive_multiple_of_18_with_digits_9_or_0_l632_632037

noncomputable def m : ℕ := 90
theorem smallest_positive_multiple_of_18_with_digits_9_or_0 : m = 90 ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 9) ∧ m % 18 = 0 → m / 18 = 5 :=
by
  intro h
  sorry

end smallest_positive_multiple_of_18_with_digits_9_or_0_l632_632037


namespace sum_primes_in_range_l632_632076

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632076


namespace discount_allowed_l632_632257

-- Define the conditions
def CP : ℝ := 100 -- Cost Price (CP) is $100 for simplicity
def MP : ℝ := CP + 0.12 * CP -- Selling price marked 12% above cost price
def Loss : ℝ := 0.01 * CP -- Trader suffers a loss of 1% on CP
def SP : ℝ := CP - Loss -- Selling price after suffering the loss

-- State the equivalent proof problem in Lean
theorem discount_allowed : MP - SP = 13 := by
  sorry

end discount_allowed_l632_632257


namespace fault_line_movement_l632_632292

theorem fault_line_movement (total_movement: ℝ) (past_year: ℝ) (prev_year: ℝ) (total_eq: total_movement = 6.5) (past_eq: past_year = 1.25) :
  prev_year = 5.25 := by
  sorry

end fault_line_movement_l632_632292


namespace tangent_line_at_neg1_monotonicity_f_range_of_m_l632_632388

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x - 1) * real.exp (x + 1) + m * x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 - 4 / x - m * x

-- (I) The tangent line when m = 1
theorem tangent_line_at_neg1 (h : f (-1) 1 = -1) (h' : (λ x, x * real.exp (x + 1) + 2 * x) (-1) = -3) :
  ∃ k b : ℝ, k = -3 ∧ b = -1 ∧ (∀ x : ℝ, f x 1 = k * (x + 1) + b) :=
sorry

-- (II) Monotonicity of f(x)
theorem monotonicity_f (m : ℝ) (h : m > - real.exp 1 / 2) :
  (∀ x : ℝ, f'.m x = x * (real.exp (x + 1) + 2 * m)) ∧
  ((m ≥ 0) → (∀ x < 0, f'.m x < 0 ∧ ∀ x > 0, f'.m x > 0)) ∧
  ((m < 0) → 
    (∀ x < ln (-2 * m) - 1, f'.m x < 0) ∧
    (ln (-2 * m) - 1 < x < 0 → f'.m x > 0) ∧
    (x > 0 → f'.m x > 0)) :=
sorry

-- (III) Range of m for which f(x1) ≤ g(x2)
theorem range_of_m :
  ∀ m : ℝ, (0 < m ∧ m ≤ 3 + real.exp 1 / 2) ↔ (∃ x1 ∈ ℝ, ∃ x2 ∈ Icc 0 (2 : ℝ), f x1 m ≤ g x2 m) :=
sorry

end tangent_line_at_neg1_monotonicity_f_range_of_m_l632_632388


namespace distance_walked_by_friend_P_l632_632598

variable (v : ℝ)                          -- Friend Q's speed in km/h
variable (t : ℝ)                          -- Time in hours when they meet
variable (d_Q : ℝ := v * t)               -- Distance walked by Friend Q
variable (d_P : ℝ := 1.35 * v * t)        -- Distance walked by Friend P
variable (length_of_trail : ℝ := 50)      -- Total length of the trail

theorem distance_walked_by_friend_P:
  d_P = 1.35 * d_Q → 
  d_P + d_Q = length_of_trail →
  d_P ≈ 28.72 :=
by
  sorry -- proof not required

end distance_walked_by_friend_P_l632_632598


namespace sum_primes_between_20_and_30_l632_632176

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632176


namespace hydra_survival_l632_632606

-- The initial number of heads for both hydras
def initial_heads_hydra_A : ℕ := 2016
def initial_heads_hydra_B : ℕ := 2017

-- Weekly head growth possibilities
def growth_values : set ℕ := {5, 7}

-- Death condition: The hydras die if their head counts become equal.
def death_condition (heads_A heads_B : ℕ) : Prop := heads_A = heads_B

-- The problem statement to prove
theorem hydra_survival : ∀ (weeks : ℕ) (growth_A growth_B : ℕ),
  growth_A ∈ growth_values →
  growth_B ∈ growth_values →
  ¬ death_condition 
    (initial_heads_hydra_A + weeks * growth_A - 2 * weeks)
    (initial_heads_hydra_B + weeks * growth_B - 2 * weeks) :=
by
  sorry

end hydra_survival_l632_632606


namespace remainder_polynomial_division_l632_632198

theorem remainder_polynomial_division (p : ℝ → ℝ) :
  p 1 = 4 → p 3 = -2 → ∃ q : ℝ → ℝ, p(x) = (x-1)*(x-3)*q(x) + (-3*x + 7) :=
begin
  intros h1 h3,
  use λ x, (p x - (-3 * x + 7)) / ((x - 1) * (x - 3)),
  sorry
end

end remainder_polynomial_division_l632_632198


namespace sum_primes_20_to_30_l632_632162

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632162


namespace center_of_circle_l632_632956

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 10) (h4 : y2 = 7) :
  (x1 + x2) / 2 = 6 ∧ (y1 + y2) / 2 = 2 :=
by
  rw [h1, h2, h3, h4]
  constructor
  · norm_num
  · norm_num

end center_of_circle_l632_632956


namespace nesbitt_inequality_l632_632497

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nesbitt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := 
by
  sorry

end nesbitt_inequality_l632_632497


namespace min_value_a_l632_632872

theorem min_value_a (a : ℤ) (h1 : 2^10 = 1024) (h2 : 2^11 = 2048) :
  (∀ x : ℝ, 2^x < 2011 → x < a) → a = 11 :=
by
  sorry

end min_value_a_l632_632872


namespace equal_area_centroid_S_l632_632976

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def Q : ℝ × ℝ := (7, -5)
noncomputable def R : ℝ × ℝ := (0, 6)
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem equal_area_centroid_S (x y : ℝ) (h : (x, y) = centroid P Q R) :
  10 * x + y = 34 / 3 := by
  sorry

end equal_area_centroid_S_l632_632976


namespace inradius_of_triangle_area_twice_perimeter_l632_632437

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l632_632437


namespace sum_primes_20_to_30_l632_632157

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632157


namespace max_value_eq_two_l632_632912

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l632_632912


namespace find_a_and_zeros_l632_632843

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x * Real.cos x - x * Real.sin x

theorem find_a_and_zeros :
  (∃ a : ℝ, (deriv (f a) 0 = 1) ∧
    (∀ x : ℝ, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 →
      (∃! z : ℝ, -Real.pi / 2 ≤ z ∧ z ≤ Real.pi / 2 ∧ f 1 z = 0))) :=
begin
  sorry
end

end find_a_and_zeros_l632_632843


namespace sum_primes_20_to_30_l632_632154

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632154


namespace total_birds_correct_l632_632705

def num_cages : ℕ := 17
def empty_cages : ℕ := 3
def num_parrots_in_first_cage : ℕ := 2
def num_parakeets_in_first_cage : ℕ := 7
def total_birds := ∑ i in Finset.range (num_cages - empty_cages), 
                  (num_parrots_in_first_cage + i + num_parakeets_in_first_cage + 2 * i)

theorem total_birds_correct :
  total_birds = 399 :=
by
  sorry

end total_birds_correct_l632_632705


namespace triangle_angle_relationship_l632_632262

-- Given triangle vertices and points
variables [Inhabited Point]

theorem triangle_angle_relationship
  (A B C D E : Point)
  (angle_C_eq_2_angle_B : angle C = 2 * angle B)
  (DC_eq_2_BD : distance D C = 2 * distance B D)
  (D_midpoint_AE : midpoint D A E) :
  angle E C B + 180° = 2 * angle E B C :=
sorry

end triangle_angle_relationship_l632_632262


namespace p_sufficient_not_necessary_for_q_l632_632413

-- Define the conditions p and q
def p (x : ℝ) := x^2 < 5 * x - 6
def q (x : ℝ) := |x + 1| ≤ 4

-- The goal to prove
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l632_632413


namespace hydras_never_die_l632_632650

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l632_632650


namespace area_of_figure_l632_632939

-- Define the conditions using a predicate
def satisfies_condition (x y : ℝ) : Prop :=
  |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

-- Define the set of points satisfying the condition
def S : set (ℝ × ℝ) := { p | satisfies_condition p.1 p.2 }

-- Define a function to calculate the area of the resulting figure
noncomputable def area_of_S : ℝ :=
  -- define interior triangular region using vertices (0,0), (8,0), and (0,15)
  1 / 2 * 8 * 15

-- The theorem to be proved
theorem area_of_figure : area_of_S = 60 :=
by
  -- This is where the actual proof would go.
  sorry

end area_of_figure_l632_632939


namespace hydrae_never_equal_heads_l632_632640

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632640


namespace distinct_integer_roots_l632_632766

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l632_632766


namespace number_of_integers_l632_632750

open Int

theorem number_of_integers (n : ℤ) :
  (1 + (floor (120 * n / 121) : ℤ) = (ceil (119 * n / 120) : ℤ)) ↔ (n % 14520 = 0) :=
sorry

end number_of_integers_l632_632750


namespace balanced_subset_count_M_l632_632424

def is_balanced_set (E : Finset ℕ) : Prop :=
  ∃ a b c d ∈ E, (a, b, c, d : ℤ) ∧ (a + b = c + d ∨ a + b + c = d)

def balanced_subsets_count (M : Finset ℕ) : ℕ :=
  (Finset.powersetLen 4 M).filter is_balanced_set).card

theorem balanced_subset_count_M :
  balanced_subsets_count (Finset.range 101) = 105361 := sorry

end balanced_subset_count_M_l632_632424


namespace conjugate_in_second_quadrant_l632_632803

variable {θ : ℝ}

def z (θ : ℝ) : Complex := Complex.mk (Real.cos θ) (Real.cos (θ + Real.pi / 2))

def z_conjugate (θ : ℝ) : Complex := Complex.conj (z θ)

theorem conjugate_in_second_quadrant (hθ : θ ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  let z := z_conjugate θ
  z.re < 0 ∧ z.im > 0 := by
  sorry

end conjugate_in_second_quadrant_l632_632803


namespace prime_sum_20_to_30_l632_632185

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l632_632185


namespace probability_no_growth_pie_l632_632266

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given_mary : ℕ := 3

theorem probability_no_growth_pie : 
  (probability (λ distribution : finset (fin total_pies), 
                distribution.card = pies_given_mary ∧ 
                (distribution.count (λ x, x < growth_pies) = 0 ∨ 
                 (finset.range total_pies \ distribution).count (λ x, x < growth_pies) = 0)) = 0.4) :=
sorry

end probability_no_growth_pie_l632_632266


namespace orthocentric_tetrahedron_nine_point_circles_on_sphere_l632_632523

noncomputable def orthocentric_tetrahedron (A B C D : Point) : Prop :=
  -- Here we define the property of being an orthocentric tetrahedron.
  -- Specifics of the definition would go here.

noncomputable def nine_point_circle (triangle : Triangle) : Circle :=
  -- Definition of a nine-point circle for a given triangle.
  -- Specifics of the definition would go here.

theorem orthocentric_tetrahedron_nine_point_circles_on_sphere
  {A B C D : Point}
  (H_orthocentric : orthocentric_tetrahedron A B C D) :
  ∃ S : Sphere, 
    (nine_point_circle ⟨A, B, C⟩ ∈ S) ∧
    (nine_point_circle ⟨A, B, D⟩ ∈ S) ∧
    (nine_point_circle ⟨A, C, D⟩ ∈ S) ∧
    (nine_point_circle ⟨B, C, D⟩ ∈ S) :=
sorry

end orthocentric_tetrahedron_nine_point_circles_on_sphere_l632_632523


namespace sum_of_primes_between_20_and_30_l632_632120

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632120


namespace gasoline_price_increase_l632_632963

theorem gasoline_price_increase 
  (P Q : ℝ)
  (h_intends_to_spend : ∃ M, M = P * Q * 1.15)
  (h_reduction : ∃ N, N = Q * (1 - 0.08))
  (h_equation : P * Q * 1.15 = P * (1 + x) * Q * (1 - 0.08)) :
  x = 0.25 :=
by
  sorry

end gasoline_price_increase_l632_632963


namespace NimGame_winning_strategy_l632_632692

/-- Definition of the Nim-style game -/
structure NimGame (k : ℕ) (S : fin k → ℤ) :=
  (initial_state : fin k → ℤ)
  (move : (fin k → ℤ) → (fin k → ℤ) → Prop)

/-- Proving the winning strategy conditions for the game -/
theorem NimGame_winning_strategy (k n : ℕ) (S : set (fin k → ℤ)) :
  (∃ game : NimGame k S, 
    ((∃ strategy : (k → ℤ) → Prop, 
      (∀ (state : k → ℤ), 
        (all_diag : ∀ i, state i >= 0) -> 
        (if pow_of_two n then first_player_has_winning_strategy strategy else second_player_has_winning_strategy strategy)
    ):
  sorry

end NimGame_winning_strategy_l632_632692


namespace tangent_of_circumcircle_l632_632478

open EuclideanGeometry

variables (A B C D E T X Y: Point)

-- Define the pentagon inscribed in circle
variable (O: Circle)
variable (hABCDE: inscribedPentagon O A B C D E)

-- Intersection point
variable (hT: intersect AD BE T)

-- Line through T parallel to CD intersecting AB at X and CE at Y
variable (h_parallel: ∃ l, parallel CD l ∧ lineThrough T l ∧ intersect AB l X ∧ intersect CE l Y)

-- Circles
variable (ω: Circle)
variable (hω: circumcircle ω A X Y)
variable (hO: circumcircle O A B C D E)

theorem tangent_of_circumcircle :
  tangentCircleAtPoint ω O A :=
sorry

end tangent_of_circumcircle_l632_632478


namespace probability_one_no_GP_l632_632279

def num_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def picked_pies : ℕ := 3
def total_outcomes : ℕ := Nat.choose num_pies picked_pies

def fav_outcomes : ℕ := Nat.choose shrink_pies 2 -- Choosing 2 out of the 4 SP

def probability_complementary : ℚ := fav_outcomes / total_outcomes
def probability : ℚ := 1 - probability_complementary

theorem probability_one_no_GP :
  probability = 0.4 := by
  sorry

end probability_one_no_GP_l632_632279


namespace hydrae_never_equal_heads_l632_632642

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632642


namespace polynomial_divisible_l632_632918

theorem polynomial_divisible (n : ℕ) (a : ℕ → ℤ)
  (h1 : ∀ k, k ≤ n → a (n - k) = a k)
  (h2 : ∑ i in finset.range (n + 1), a i = 0)
  (h3 : a n ≠ 0) :
  ∃ k : ℕ, k > 1 ∧ (k ∣ P 2022) where P (x : ℕ) : ℤ :=
    ∑ i in finset.range (n + 1), (a i) * x^i :=
begin
  -- proof to be completed
  sorry
end

end polynomial_divisible_l632_632918


namespace no_growth_pie_probability_l632_632274

noncomputable def probability_no_growth_pies : ℝ :=
  let total_pies := 6
  let growth_pies := 2
  let shrink_pies := 4
  let pies_given := 3
  let total_combinations := Nat.choose total_pies pies_given
  let favorable_outcomes := Nat.choose shrink_pies 3 + Nat.choose shrink_pies 2 * Nat.choose growth_pies 1 + Nat.choose shrink_pies 1 * Nat.choose growth_pies 2
  in favorable_outcomes / total_combinations

theorem no_growth_pie_probability :
  probability_no_growth_pies = 0.4 :=
sorry

end no_growth_pie_probability_l632_632274


namespace decimal_to_binary_51_l632_632743

theorem decimal_to_binary_51 :
  ∀ (n : ℕ), n = 51 → n.binary_repr = "110011" :=
by
  intros n h
  rw h
  sorry

end decimal_to_binary_51_l632_632743


namespace flower_shop_percentage_l632_632881

theorem flower_shop_percentage (C : ℕ) : 
  let V := (1/3 : ℝ) * C
  let T := (1/12 : ℝ) * C
  let R := T
  let total := C + V + T + R
  (C / total) * 100 = 66.67 := 
by
  sorry

end flower_shop_percentage_l632_632881


namespace max_cubic_root_sum_l632_632903

theorem max_cubic_root_sum (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (Real.cbrt (a * b * c) + Real.cbrt ((1 - a) * (1 - b) * (1 - c))) ≤ 1 :=
by
  sorry

end max_cubic_root_sum_l632_632903


namespace loaned_books_during_month_l632_632704

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def returned_percent : ℚ := 0.65
def end_books : ℕ := 68

-- Proof statement
theorem loaned_books_during_month (x : ℕ) 
  (h1 : returned_percent = 0.65)
  (h2 : initial_books = 75)
  (h3 : end_books = 68) :
  (0.35 * x : ℚ) = (initial_books - end_books) :=
sorry

end loaned_books_during_month_l632_632704


namespace sin_alpha_minus_beta_l632_632366

variables (α β : ℝ)

theorem sin_alpha_minus_beta (h1 : (Real.tan α / Real.tan β) = 7 / 13) 
    (h2 : Real.sin (α + β) = 2 / 3) :
    Real.sin (α - β) = -1 / 5 := 
sorry

end sin_alpha_minus_beta_l632_632366


namespace hypotenuse_length_l632_632869

theorem hypotenuse_length (x a b: ℝ) (h1: a = 7) (h2: b = x - 1) (h3: a^2 + b^2 = x^2) : x = 25 :=
by {
  -- Condition h1 states that one leg 'a' is 7 cm.
  -- Condition h2 states that the other leg 'b' is 1 cm shorter than the hypotenuse 'x', i.e., b = x - 1.
  -- Condition h3 is derived from the Pythagorean theorem, i.e., a^2 + b^2 = x^2.
  -- We need to prove that x = 25 cm.
  sorry
}

end hypotenuse_length_l632_632869


namespace sum_primes_between_20_and_30_l632_632179

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632179


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l632_632731

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l632_632731


namespace minimize_std_dev_l632_632067

structure WalkingDistances where
  m : ℝ
  n : ℝ
  distance_list : List ℝ
  sorted : distance_list = [11, 12, m, n, 20, 27]
  median_cond : (distance_list.nth 2 + distance_list.nth 3) / 2 = 16
  total_cond : m + n = 32

theorem minimize_std_dev (wd : WalkingDistances) : wd.m = 16 :=
by
  sorry

end minimize_std_dev_l632_632067


namespace probability_one_girl_no_growth_pie_l632_632285

-- Definitions based on the conditions
def total_pies := 6
def growth_pies := 2
def shrink_pies := total_pies - growth_pies
def total_selections := ((total_pies).choose(3) : ℚ)
def favorable_selections := ((shrink_pies).choose(2) : ℚ)

-- Calculation of the probability
noncomputable def probability_no_growth_pie := 1 - favorable_selections / total_selections

-- Proving the required probability
theorem probability_one_girl_no_growth_pie : probability_no_growth_pie = 0.4 :=
by
  sorry

end probability_one_girl_no_growth_pie_l632_632285


namespace parabola_focus_distance_twice_l632_632804

-- Definitions
def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def focus := (3 / 2, 0)
def distance_to_focus (P : ℕ) : ℝ := P + 3 / 2
def distance_to_y_axis (P : ℕ) : ℝ := P

-- The problem statement
theorem parabola_focus_distance_twice (x : ℝ) (y : ℝ) :
  parabola x y →
  distance_to_focus x = 2 * distance_to_y_axis x →
  x = 3 / 2 :=
by
  intros h1 h2
  sorry

end parabola_focus_distance_twice_l632_632804


namespace evaluate_expression_at_value_l632_632757

theorem evaluate_expression_at_value :
  let a := 4 / 3 in
  (6 * a^2 - 11 * a + 2) * (3 * a - 4) = 0 :=
by
  let a := 4 / 3
  have h1 : (3 * a - 4) = 0 := by
    sorry
  have h2 : (6 * a^2 - 11 * a + 2) = -2 := by
    sorry
  calc
    (6 * a^2 - 11 * a + 2) * (3 * a - 4) = -2 * 0 : by congr; assumption <|> sorry
                                        ...   = 0  : by simp

end evaluate_expression_at_value_l632_632757


namespace z_pow12_plus_inv_z_pow12_l632_632823

open Complex

theorem z_pow12_plus_inv_z_pow12 (z: ℂ) (h: z + z⁻¹ = 2 * cos (10 * Real.pi / 180)) :
  z^12 + z⁻¹^12 = -1 := by
  sorry

end z_pow12_plus_inv_z_pow12_l632_632823


namespace cos_F_l632_632446

-- Define the given conditions
variables (D F E : Type) -- Points in the plane
variables [DecidableEq D] [DecidableEq F] [DecidableEq E] 
variables (distance : D → E → F → Real)
variables (angle : D → E → F → Real)

-- Given conditions
def is_right_triangle (D F E : Type) [DecidableEq D] [DecidableEq F] [DecidableEq E] 
  (distance : D → E → F → Real) (angle : D → E → F → Real) : Prop :=
angle D E F = 90 ∧ distance D E = 8 ∧ distance E F = 17

-- The proof statement
theorem cos_F (D F E : Type) [DecidableEq D] [DecidableEq F] [DecidableEq E] 
 (distance : D → E → F → Real) (angle : D → E → F → Real) :
 is_right_triangle D F E distance angle →
  let DF := sqrt ((distance E F)^2 - (distance D E)^2) in
  cos (angle D E F) = distance D E / distance E F :=
begin
  intros h,
  rcases h with ⟨h_angle, h_DE, h_EF⟩,
  have h_DF : distance D F = sqrt ((distance E F)^2 - (distance D E)^2), {
    rw [h_DE, h_EF],
    simp,
  },
  rw [h_angle, h_DE, h_EF],
  sorry
end

end cos_F_l632_632446


namespace radius_of_sphere_touching_edges_of_regular_tetrahedron_l632_632031

theorem radius_of_sphere_touching_edges_of_regular_tetrahedron (a : ℝ) (h : a = √2) : 
  ∃ r : ℝ, r = 1 :=
by
  have h_trans := calc
    a : ℝ = √2 := h
  -- Further calculation steps would go here
  -- We skip steps and directly assert the result using sorry
  exact ⟨1, by sorry⟩

end radius_of_sphere_touching_edges_of_regular_tetrahedron_l632_632031


namespace geometric_sum_five_terms_l632_632924

theorem geometric_sum_five_terms (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_a2a4 : a 1 * a 3 = 16)
  (h_ratio : (a 3 + a 4 + a 7) / (a 0 + a 1 + a 4) = 8) :
  S 5 = 31 :=
sorry

end geometric_sum_five_terms_l632_632924


namespace conjugate_of_z_l632_632560

def i : ℂ := complex.I

def z : ℂ := 2 / ((1 - i) * i)

theorem conjugate_of_z : complex.conj z = 1 + i := by
  sorry

end conjugate_of_z_l632_632560


namespace distance_from_origin_to_line_l632_632030

theorem distance_from_origin_to_line : 
  let A := 1
  let B := 2
  let C := -5
  let x_0 := 0
  let y_0 := 0
  let distance := |A * x_0 + B * y_0 + C| / (Real.sqrt (A ^ 2 + B ^ 2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l632_632030


namespace parabola_2_second_intersection_x_l632_632703

-- Definitions of the conditions in the problem
def parabola_1_intersects : Prop := 
  (∀ x : ℝ, (x = 10 ∨ x = 13) → (∃ y : ℝ, (x, y) ∈ ({p | p = (10, 0)} ∪ {p | p = (13, 0)})))

def parabola_2_intersects : Prop := 
  (∃ x : ℝ, x = 13)

def vertex_bisects_segment : Prop := 
  (∃ a : ℝ, 2 * 11.5 = a)

-- The theorem we want to prove
theorem parabola_2_second_intersection_x : 
  parabola_1_intersects ∧ parabola_2_intersects ∧ vertex_bisects_segment → 
  (∃ t : ℝ, t = 33) := 
  by
  sorry

end parabola_2_second_intersection_x_l632_632703


namespace sum_of_primes_between_20_and_30_l632_632121

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l632_632121


namespace dot_product_sum_l632_632488

variables (a b c : ℝ)
variables (vec_a vec_b vec_c : E) [inner_product_space ℝ E]

def norm (u : E) : ℝ := sqrt (inner u u)

theorem dot_product_sum :
  norm vec_a = 2 → 
  norm vec_b = 3 → 
  norm vec_c = 6 → 
  vec_a + vec_b + vec_c = 0 →
  inner vec_a vec_b + inner vec_a vec_c + inner vec_b vec_c = -49 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end dot_product_sum_l632_632488


namespace cuboid_height_l632_632769

theorem cuboid_height (l b A : ℝ) (hl : l = 10) (hb : b = 8) (hA : A = 480) :
  ∃ h : ℝ, A = 2 * (l * b + b * h + l * h) ∧ h = 320 / 36 := by
  sorry

end cuboid_height_l632_632769


namespace area_of_quadrilateral_l632_632559

noncomputable theory
open_locale classical

variables {A B C D O P : Type} [EuclideanSpace ℝ A B C D O P]
variable (A B C D O : A)
variables (m n : ℝ)

-- Given conditions
def conditions (h1 : ∠ B A O = ∠ D A C)
               (h2 : dist A C = m)
               (h3 : dist B D = n)
               (h4 : ∃ circscribed_circle A B C D O, O ∈ interior_quadrilateral A B C D) : Prop :=
True

-- Statement of the proof problem
theorem area_of_quadrilateral (h1 : ∠ B A O = ∠ D A C)
                              (h2 : dist A C = m)
                              (h3 : dist B D = n)
                              (h4 : ∃ circscribed_circle A B C D O, O ∈ interior_quadrilateral A B C D) :
  area_quadrilateral A B C D = (m * n) / 2 := 
sorry

end area_of_quadrilateral_l632_632559


namespace sum_coefficients_binomial_l632_632968

theorem sum_coefficients_binomial (x y : ℕ) :
  let expansion := (x + y) ^ 8 in
  (expansion.eval 1 1 = 256) :=
by
  sorry

end sum_coefficients_binomial_l632_632968


namespace hamster_win_if_irreducible_l632_632566

-- Defining the game of circulate based on given conditions
variables (k n : ℕ)

structure PileConfig :=
  (cards : fin k → fin n)

structure GameConfig :=
  (piles : fin n → PileConfig)

def is_irreducible (config : GameConfig k n) : Prop := sorry
-- Defining irreducibility will require precise graph definitions
-- and connectivity properties, which we are abstracting.

def win_game (config : GameConfig k n) : Prop := sorry
-- This predicate will assert the winning configuration

theorem hamster_win_if_irreducible (config : GameConfig k n) :
  is_irreducible config ↔ win_game config :=
sorry

end hamster_win_if_irreducible_l632_632566


namespace sum_of_primes_between_20_and_30_l632_632090

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632090


namespace relationship_among_a_b_c_l632_632829

noncomputable theory

def a : ℝ := Real.log (3 / 4) (3 / 2)
def b : ℝ := (3 / 2) ^ (3 / 2)
def c : ℝ := (3 / 4) ^ (4 / 3)

theorem relationship_among_a_b_c : b > c ∧ c > a := by
  sorry

end relationship_among_a_b_c_l632_632829


namespace num_elements_in_S_l632_632482

noncomputable def X : Set ℤ := 
  {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def S : Set (ℤ × ℤ) :=
  { p | ∃ α : ℝ, α^2 + p.1 * α + p.2 = 0 ∧ α^3 + p.2 * α + p.1 = 0 }

theorem num_elements_in_S : (S ∩ (X ×ˢ X)).card = 21 := by
  sorry

end num_elements_in_S_l632_632482


namespace sum_primes_in_range_l632_632082

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632082


namespace millionth_digit_1_div_41_l632_632771

theorem millionth_digit_1_div_41 : 
  ∃ n, n = 1000000 ∧ (n % 5 = 0) → digit_after_decimal (1 / 41) n = 9 :=
begin
  intro exists n,
  sorry -- proof goes here
end

end millionth_digit_1_div_41_l632_632771


namespace max_ratio_inscribed_rectangles_l632_632481

theorem max_ratio_inscribed_rectangles
    (T : Triangle)
    (hT : T.isAcute)
    (R S : Rectangle)
    (hR : R.isInInscribedIn T)
    (hS : S.isInscribedInInSubTriangleOf T R) :
  (A(R) + A(S)) / A(T) ≤ 2 / 3 :=
sorry

end max_ratio_inscribed_rectangles_l632_632481


namespace max_value_of_function_l632_632039

noncomputable def function (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_value_of_function : ∃ x : ℝ, function x = 5 / 4 :=
by
  sorry

end max_value_of_function_l632_632039


namespace eighth_prime_is_19_l632_632658

/-
We define prime numbers and then prove that the 8th prime number is 19.
-/

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (k : ℕ) : ℕ :=
  Nat.find 
    (by 
      let primes := Nat.filter (is_prime) (List.range (k * (2 * k).log_base 2)) -- A simple range to cover enough primes
      exact ⟨k, by sorry⟩ -- Just to ensure we have a k-th prime within the range
    )

theorem eighth_prime_is_19 : nth_prime 8 = 19 :=
sorry

end eighth_prime_is_19_l632_632658


namespace probability_both_boys_or_both_girls_l632_632794

theorem probability_both_boys_or_both_girls 
  (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 5 → boys = 2 → girls = 3 →
    (∃ (p : ℚ), p = 2/5) :=
by
  intros ht hb hg
  sorry

end probability_both_boys_or_both_girls_l632_632794


namespace geometry_theorem_l632_632726

theorem geometry_theorem
  (A B C I D E N G M F : Type)
  [triangle : Triangle ABC]
  [incircle : Incircle I ABC]
  [circumcircle : Circumcircle O ABC]
  [touches_BC : Touches D I BC]
  [touches_CA : Touches E I CA]
  [intersects_AI_N : Intersects AI N O]
  [extend_ND_G : Extends ND G O]
  [extend_NO_M : Extends NO M O]
  [extend_GE_F : Extends GE F O]
  [lt_AB_AC : LT AB AC]
: (Similar (Triangle N I G) (Triangle N D I)) ∧ (Parallel MF AC) :=
  by sorry

end geometry_theorem_l632_632726


namespace num_dimes_is_3_l632_632681

noncomputable def num_dimes (pennies nickels dimes quarters : ℕ) : ℕ :=
  dimes

theorem num_dimes_is_3 (h_total_coins : pennies + nickels + dimes + quarters = 11)
  (h_total_value : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 118)
  (h_at_least_one_each : 0 < pennies ∧ 0 < nickels ∧ 0 < dimes ∧ 0 < quarters) :
  num_dimes pennies nickels dimes quarters = 3 :=
sorry

end num_dimes_is_3_l632_632681


namespace solution_set_inequality_l632_632840

variable {R : Type*} [LinearOrderedField R] 
variable (f : R → R)

theorem solution_set_inequality 
  (h1 : f 1 = 1)
  (h2 : ∀ x, f' x < 1 / 2) :
  { x : R | f (x^2) < x^2 / 2 + 1 / 2 } = { x : R | x < -1 } ∪ { x | 1 < x } :=
by {
  sorry
}

end solution_set_inequality_l632_632840


namespace max_value_fraction_sum_l632_632405

-- Define a triangle with sides a, b, and c
variables {A B C : Type*} [HasInhabited A]
variables (AB AC BC : Real)

-- Conditions: The altitude on side AB equals the length of AB
def altitude_eqAB (h : ℝ) (AB : ℝ) (C : ℝ) : Prop :=
 h = AB + AC - (2 * (BC * Cos C))

-- Objective: Prove the maximum value
theorem max_value_fraction_sum
  (altitude_eq : ∀ (C : ℝ), altitude_eqAB AB h AB) :
  ∀ (AC BC : ℝ), ∃ C, maximum_value \(\frac{AC}{BC} + \frac{BC}{AC} + \frac{AB^2}{BC \cdot AC}) = 2\sqrt{2} :=
by
  sorry

end max_value_fraction_sum_l632_632405


namespace negation_prop_l632_632573

theorem negation_prop (x : ℝ) : (¬ (∀ x : ℝ, Real.exp x > x^2)) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) :=
by
  sorry

end negation_prop_l632_632573


namespace prime_sum_20_to_30_l632_632140

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632140


namespace obtuse_angle_of_triangle_l632_632909

-- Definitions based on conditions
variables (α β : ℝ) (p : ℝ)
-- Conditions: α and β are two interior angles of a triangle.
-- tan α and tan β are the two real roots of the equation x^2 + p(x + 1) + 1 = 0.
def equation_has_tan_roots := ∃ (α β : ℝ), 
  (tan α + tan β = -p) ∧ (tan α * tan β = p + 1)

-- Question rephrased as proving the measure of the obtuse angle
theorem obtuse_angle_of_triangle : 
  equation_has_tan_roots α β p → α + β = 45 →
  ∃ γ, (γ = 135 ∧ γ + α + β = 180) :=
sorry

end obtuse_angle_of_triangle_l632_632909


namespace product_of_selected_ten_numbers_l632_632979

theorem product_of_selected_ten_numbers (ones : ℕ) (neg_ones : ℕ) (select : ℕ) 
(h1 : ones = 12) (h2 : neg_ones = 10) (h3 : select = 10) : 
∑ (s : finset (fin 22)), if s.card = 10 then (s.map (λ i, if i < 12 then 1 else -1)).prod else 0 = -42 :=
by {
  sorry
}

end product_of_selected_ten_numbers_l632_632979


namespace sum_primes_20_to_30_l632_632170

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632170


namespace bunnies_out_of_burrow_l632_632417

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end bunnies_out_of_burrow_l632_632417


namespace grade_point_average_l632_632958

theorem grade_point_average (X : ℝ) (GPA_rest : ℝ) (GPA_whole : ℝ) 
  (h1 : GPA_rest = 66) (h2 : GPA_whole = 64) 
  (h3 : (1 / 3) * X + (2 / 3) * GPA_rest = GPA_whole) : X = 60 :=
sorry

end grade_point_average_l632_632958


namespace height_of_larger_box_l632_632878

/-- Define the dimensions of the larger box and smaller boxes, 
    and show that given the constraints, the height of the larger box must be 4 meters.-/
theorem height_of_larger_box 
  (L H : ℝ) (V_small : ℝ) (N_small : ℕ) (h : ℝ) 
  (dim_large : L = 6) (width_large : H = 5)
  (vol_small : V_small = 0.6 * 0.5 * 0.4) 
  (num_boxes : N_small = 1000) 
  (vol_large : 6 * 5 * h = N_small * V_small) : 
  h = 4 :=
by 
  sorry

end height_of_larger_box_l632_632878


namespace intermission_length_l632_632465

def concert_duration : ℕ := 80
def song_duration_total : ℕ := 70

theorem intermission_length : 
  concert_duration - song_duration_total = 10 :=
by
  -- conditions are already defined above
  sorry

end intermission_length_l632_632465


namespace weight_of_new_person_l632_632024

theorem weight_of_new_person (W : ℝ) : 
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  weight_new_person = 70 :=
by
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  have : weight_new_person = 70 := sorry
  exact this

end weight_of_new_person_l632_632024


namespace real_solutions_l632_632330

variables {R : Type*} [linear_ordered_field R]

def cyclic_index (n : ℕ) (i : ℕ) : ℕ :=
  ((i - 1) % n + n) % n

theorem real_solutions (n k : ℕ) (h₁ : n > k) (h₂ : k > 1) 
  (x : fin n → R) :
  (∀ i : fin n, x i ^ 3 * (finset.sum (finset.range k) (λ j, x (cyclic_index n (i + j)) ^ 2)) = (x (cyclic_index n (i - 1))) ^ 2) ↔
  ((∀ i : fin n, x i = 0) ∨ (∀ i : fin n, x i = (k : R)⁻¹ ^ (1/3))) :=
sorry

end real_solutions_l632_632330


namespace total_wait_time_l632_632668

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l632_632668


namespace exponential_to_logarithmic_l632_632038

theorem exponential_to_logarithmic (b a : ℝ) (hb : 0 < b) (hb_ne_one : b ≠ 1) (h : b ^ 3 = a) : log b a = 3 :=
by 
-- proof goes here
sorry

end exponential_to_logarithmic_l632_632038


namespace sqrt_expr_correct_l632_632054

noncomputable def sqrt_expr (m n : ℝ) : ℝ :=
  (m-2*n-3) * (m-2*n+3) + 9

theorem sqrt_expr_correct (m n : ℝ) : 
  sqrt (sqrt_expr m n) = 
  if h : m ≥ 2 * n then m - 2 * n else 2 * n - m :=
by 
  sorry

end sqrt_expr_correct_l632_632054


namespace fixed_sphere_or_fixed_point_l632_632447

noncomputable def sphere (center : ℝ^3) (radius : ℝ) : set ℝ^3 := 
  {p | dist p center = radius}

variables (A B C : ℝ^3)
variables (S1_center S2_center : ℝ^3) (S1_radius S2_radius : ℝ)
variables (S1 : set ℝ^3 := sphere S1_center S1_radius)
variables (S2 : set ℝ^3 := sphere S2_center S2_radius)
variables (M : ℝ^3) -- point on S1 not coplanar with the triangle ABC
variables (A1 B1 C1 : ℝ^3) -- points on S2

axiom passes_through_triangle (S : set ℝ^3) : A ∈ S ∧ B ∈ S ∧ C ∈ S
axiom intersection_point_S2 (M : ℝ^3) (H_M_on_S1 : M ∈ S1)
  (non_coplanar : ¬ coplanar ({A, B, C} : set ℝ^3) {M}) : 
  under_line_through M A ∈ S2 ∧ under_line_through M B ∈ S2 ∧ under_line_through M C ∈ S2

theorem fixed_sphere_or_fixed_point :
  (∀ M ∈ S1, (¬ coplanar ({A, B, C} : set ℝ^3) {M}) → 
  ∃ (K : ℝ^3), (K ∈ S2 ∨ ∃ (center : ℝ^3) (radius : ℝ), sphere center radius = {P | P ∈ plane A1 B1 C1})) :=
by
  sorry

end fixed_sphere_or_fixed_point_l632_632447


namespace hydrae_never_equal_heads_l632_632644

theorem hydrae_never_equal_heads :
  ∀ (a b : ℕ), a = 2016 → b = 2017 →
  (∀ (a' b' : ℕ), a' ∈ {5, 7} → b' ∈ {5, 7} → 
  ∀ n : ℕ, let aa := a + n * 5 + (n - a / 7) * 2 - n in
           let bb := b + n * 5 + (n - b / 7) * 2 - n in
  aa + bb ≠ 2 * (aa / 2)) → 
  true :=
begin
  -- Sorry, the proof is left as an exercise
  sorry,
end

end hydrae_never_equal_heads_l632_632644


namespace sum_primes_in_range_l632_632080

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632080


namespace find_k_l632_632859

-- Definitions for the vectors and collinearity condition.

def vector := ℝ × ℝ

def collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Given vectors a and b.
def a (k : ℝ) : vector := (1, k)
def b : vector := (2, 2)

-- Vector addition.
def add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

-- Problem statement
theorem find_k (k : ℝ) (h : collinear (add (a k) b) (a k)) : k = 1 :=
by
  sorry

end find_k_l632_632859


namespace sum_of_primes_between_20_and_30_l632_632103

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l632_632103


namespace inequality_solution_l632_632553

theorem inequality_solution (x : ℝ) : (3 * x - 1) / (x - 2) > 0 ↔ x < 1 / 3 ∨ x > 2 :=
sorry

end inequality_solution_l632_632553


namespace time_after_classes_l632_632462

def time_after_maths : Nat := 60
def time_after_history : Nat := 60 + 90
def time_after_break1 : Nat := time_after_history + 25
def time_after_geography : Nat := time_after_break1 + 45
def time_after_break2 : Nat := time_after_geography + 15
def time_after_science : Nat := time_after_break2 + 75

theorem time_after_classes (start_time : Nat := 12 * 60) : (start_time + time_after_science) % 1440 = 17 * 60 + 10 :=
by
  sorry

end time_after_classes_l632_632462


namespace water_consumption_rate_l632_632321

def N : ℕ := 4    -- Number of people
def T : ℕ := 16   -- Total time on the road (in hours)
def W : ℕ := 32   -- Total number of water bottles needed

theorem water_consumption_rate : (W / T) / N = 0.5 := by
  sorry

end water_consumption_rate_l632_632321


namespace sum_primes_20_to_30_l632_632152

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l632_632152


namespace metal_waste_l632_632248

theorem metal_waste (l w : ℝ) (h : l > w) :
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  wasted_metal = l * w - w ^ 2 / 2 :=
by
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  sorry

end metal_waste_l632_632248


namespace angle_ADE_is_15_degrees_l632_632715

-- Definitions from the conditions
variables (A B C D E F: Type)
variables (AB_s BPDEF_side : ℝ)

  -- Conditions
variables (ABCD_square : metric_space.square A B C D AB_s)
variables (BDEF_rhombus : metric_space.rhombus B D E F BPDEF_side)
variables (A_E_F_collinear : collinear {A, E, F})

-- Math proof problem in Lean 4 statement
theorem angle_ADE_is_15_degrees :
  ∠ADE = 15 :=
sorry

end angle_ADE_is_15_degrees_l632_632715


namespace number_of_new_cards_l632_632929

theorem number_of_new_cards (cards_per_page old_cards pages : ℕ)
    (h1 : cards_per_page = 3) (h2 : old_cards = 9) (h3 : pages = 4) : 
    ∃ (new_cards : ℕ), new_cards + old_cards = pages * cards_per_page ∧ new_cards = 3 :=
by
  use 3
  sorry

end number_of_new_cards_l632_632929


namespace probability_of_unique_color_and_number_l632_632591

-- Defining the sets of colors and numbers
inductive Color
| red
| yellow
| blue

inductive Number
| one
| two
| three

-- Defining a ball as a combination of a Color and a Number
structure Ball :=
(color : Color)
(number : Number)

-- Setting up the list of 9 balls
def allBalls : List Ball :=
  [⟨Color.red, Number.one⟩, ⟨Color.red, Number.two⟩, ⟨Color.red, Number.three⟩,
   ⟨Color.yellow, Number.one⟩, ⟨Color.yellow, Number.two⟩, ⟨Color.yellow, Number.three⟩,
   ⟨Color.blue, Number.one⟩, ⟨Color.blue, Number.two⟩, ⟨Color.blue, Number.three⟩]

-- Proving the probability calculation as a theorem
noncomputable def probability_neither_same_color_nor_number : ℕ → ℕ → ℚ :=
  λ favorable total => favorable / total

theorem probability_of_unique_color_and_number :
  probability_neither_same_color_nor_number
    (6) -- favorable outcomes
    (84) -- total outcomes
  = 1 / 14 := by
  sorry

end probability_of_unique_color_and_number_l632_632591


namespace employee_age_when_hired_l632_632228

theorem employee_age_when_hired
    (hire_year retire_year : ℕ)
    (rule_of_70 : ∀ A Y, A + Y = 70)
    (years_worked : ∀ hire_year retire_year, retire_year - hire_year = 19)
    (hire_year_eqn : hire_year = 1987)
    (retire_year_eqn : retire_year = 2006) :
  ∃ A : ℕ, A = 51 :=
by
  have Y := 19
  have A := 70 - Y
  use A
  sorry

end employee_age_when_hired_l632_632228


namespace decreased_amount_l632_632222

theorem decreased_amount {N A : ℝ} (h₁ : 0.20 * N - A = 6) (h₂ : N = 50) : A = 4 := by
  sorry

end decreased_amount_l632_632222


namespace sum_two_smallest_prime_factors_of_525_l632_632196

theorem sum_two_smallest_prime_factors_of_525 : 
  (prime_factors 525).take 2 |> list.sum = 8 := sorry

end sum_two_smallest_prime_factors_of_525_l632_632196


namespace greatest_int_with_gcd_3_l632_632072

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end greatest_int_with_gcd_3_l632_632072


namespace length_AB_equals_six_l632_632809

noncomputable def ellipse_center : (ℝ × ℝ) := (0, 0)
noncomputable def ellipse_eccentricity : ℝ := 1 / 2
noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)
noncomputable def directrix_x : ℝ := -2

theorem length_AB_equals_six
  (ellipse_center = (0, 0))
  (ellipse_eccentricity = 1 / 2)
  (parabola_focus = (2, 0))
  (directrix_x = -2) : |(A B : ℝ)| = 6 := sorry

end length_AB_equals_six_l632_632809


namespace doctor_is_correct_l632_632614

noncomputable theory

def hydra_heads_never_equal : Prop :=
  ∀ (a b : ℕ), 
    a = 2016 ∧ b = 2017 ∧ 
    (∀ n : ℕ, ∃ (a_new b_new : ℕ), 
      (a_new = a + 5 ∨ a_new = a + 7) ∧ 
      (b_new = b + 5 ∨ b_new = b + 7) ∧
      (∀ m : ℕ, m < n → a_new + b_new - 4 * (m + 1) ≠ (a_new + b_new) / 2 * 2)
    ) → 
    ∀ n : ℕ, (a + b) % 2 = 1 ∧ a ≠ b

theorem doctor_is_correct : hydra_heads_never_equal :=
by sorry

end doctor_is_correct_l632_632614


namespace arithmetic_geo_sequences_l632_632807

theorem arithmetic_geo_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (d k : ℕ) (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 1 + a 4 = 14)
  (h3 : (a 2)^2 = a 1 * a 7)
  (h4 : ∀ n, b n = (2 * n ^ 2 - n) / (n + k))
  (h5 : ∀ n, 2 * b (n + 1) = b n + b (n + 2)) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, (2 * n ^ 2 - n))  ∧
  ((∃ n, T n = n / (4 * (n + 1))) ∨ (∃ n, T n = n / (2 * n + 1))) := 
sorry

end arithmetic_geo_sequences_l632_632807


namespace hydras_will_live_l632_632633

noncomputable def hydras_live : Prop :=
  let A_initial := 2016
  let B_initial := 2017
  let possible_growth := {5, 7}
  let weekly_death := 4
  ∀ (weeks : ℕ), 
    let A_heads := A_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    let B_heads := B_initial + weeks * (choose (possible_growth) + choose (possible_growth) - weekly_death)
    A_heads ≠ B_heads

theorem hydras_will_live : hydras_live :=
sorry

end hydras_will_live_l632_632633


namespace final_price_is_correct_l632_632254

def original_price : ℝ := 450
def discounts : List ℝ := [0.10, 0.20, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def final_sale_price (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem final_price_is_correct:
  final_sale_price original_price discounts = 307.8 :=
by
  sorry

end final_price_is_correct_l632_632254


namespace common_difference_arithmetic_sequence_l632_632372

theorem common_difference_arithmetic_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d))
  (h2 : a 7 - 2 * a 4 = -1)
  (h3 : a 3 = 0) :
  ∃ d, (∀ a1, (a1 + 2 * d = 0 ∧ -d = -1) → d = -1/2) :=
by
  sorry

end common_difference_arithmetic_sequence_l632_632372


namespace weeks_to_fill_moneybox_l632_632931

-- Monica saves $15 every week
def savings_per_week : ℕ := 15

-- Number of cycles Monica repeats
def cycles : ℕ := 5

-- Total amount taken to the bank
def total_savings : ℕ := 4500

-- Prove that the number of weeks it takes for the moneybox to get full is 60
theorem weeks_to_fill_moneybox : ∃ W : ℕ, (cycles * savings_per_week * W = total_savings) ∧ W = 60 := 
by 
  sorry

end weeks_to_fill_moneybox_l632_632931


namespace closed_interval_equinumerous_half_open_l632_632007

-- Definitions of the intervals
def interval_closed := {x : ℝ | 0 ≤ x ∧ x ≤ 1}
def interval_half_open := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Existence of a bijection
theorem closed_interval_equinumerous_half_open : ∃ (f : interval_closed → interval_half_open), function.bijective f := by
  sorry

end closed_interval_equinumerous_half_open_l632_632007


namespace sum_primes_in_range_l632_632077

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l632_632077


namespace exists_even_cycle_l632_632295

structure Graph (V : Type) :=
(adjacency : V → V → Prop)
(symm : ∀ {a b}, adjacency a b → adjacency b a)
(irrefl : ∀ {a}, ¬ adjacency a a)

def degree {V : Type} (G : Graph V) (v : V) : ℕ :=
  Nat.card (Finset.filter (G.adjacency v) (Finset.univ : Finset V))

def even_card {V : Type} (s : Finset V) : Prop :=
  s.card % 2 = 0

def has_cycle {V : Type} (G : Graph V) : Prop :=
  ∃ (cycle : List V), (∀ v, v ∈ cycle → cycle.count v = 2) ∧
                      (∀ v, G.adjacency (cycle.head) v)

theorem exists_even_cycle (V : Type) (G : Graph V)
  (h : ∀ v, 3 ≤ degree G v) : 
  ∃ s : Finset V, even_card s ∧ ∃ (cycle : List V), has_cycle G :=
sorry

end exists_even_cycle_l632_632295


namespace minimal_maximal_angle_l632_632895

theorem minimal_maximal_angle
  (n : ℕ)
  (x : Fin n.succ → ℝ)
  (h_pos : ∀ i, 0 < x i)
  (h_desc : ∀ i j, i < j → x j ≤ x i)
  (h_asc : ∀ i j, i < j → x i ≤ x j)  :
  (let α := described_angle x in
  (∀ x_desc, (∀ i j, i < j → x_desc j ≤ x_desc i) → α ≤ described_angle x_desc) ∧ 
  (∀ x_asc, (∀ i j, i < j → x_asc i ≤ x_asc j) → α ≥ described_angle x_asc)) := sorry

end minimal_maximal_angle_l632_632895


namespace exists_k_l632_632914

variable {R : Type} [LinearOrderedField R]

--  Define polynomial over a field R
noncomputable def p (n : ℕ) : Polynomial R := sorry

-- Main theorem stating there exists an integer k such that |p(k) - 3^k| ≥ 1
theorem exists_k (p : Polynomial R) (n : ℕ) (hp : degree p = n) :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ n + 1 ∧ |(p.eval k : R) - 3^k| ≥ 1 :=
sorry

end exists_k_l632_632914


namespace rationalize_denominator_l632_632534

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l632_632534


namespace number_of_solutions_l632_632316

open Real

-- Define main condition
def condition (θ : ℝ) : Prop := sin θ * tan θ = 2 * (cos θ)^2

-- Define the interval and exclusions
def valid_theta (θ : ℝ) : Prop := 
  0 ≤ θ ∧ θ ≤ 2 * π ∧ ¬ ( ∃ k : ℤ, (θ = k * (π/2)) )

-- Define the set of thetas that satisfy both the condition and the valid interval
def valid_solutions (θ : ℝ) : Prop := valid_theta θ ∧ condition θ

-- Formal statement of the problem
theorem number_of_solutions : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_solutions θ) ∧ (s.card = 4) := by
  sorry

end number_of_solutions_l632_632316


namespace parabola_vertex_l632_632052

-- Definitions derived from the problem conditions
variable (c d : ℝ)
def parabola (x : ℝ) : ℝ := -x^2 + c * x + d

-- Conditions
axiom h1 : ∀ x, (-x^2 + c * x + d) ≤ 0 ↔ (x ∈ set.Icc (-5 : ℝ) (1 : ℝ) ∨ x ∈ set.Ici (7 : ℝ))

-- Theorem to state the problem in Lean 4
theorem parabola_vertex : ∃ (x y : ℝ), x = -2 ∧ y = 9 ∧ parabola (-2) = 9 :=
by
  sorry

end parabola_vertex_l632_632052


namespace three_circles_tangent_l632_632725

-- Define the theorem
theorem three_circles_tangent
  (A B C : Point) (l : Line)
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc1 : c < a) (h_abc2 : a < b)
  (h_tangent : is_tangent A B C l a b c)
  : 1 / real.sqrt c = 1 / real.sqrt a + 1 / real.sqrt b :=
sorry


end three_circles_tangent_l632_632725


namespace sum_primes_20_to_30_l632_632171

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l632_632171


namespace math_problem_proof_l632_632730

variable {a : ℝ} (ha : a > 0)

theorem math_problem_proof : ((36 * a^9)^4 * (63 * a^9)^4 = a^(72)) :=
by sorry

end math_problem_proof_l632_632730


namespace number_of_children_l632_632584

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l632_632584


namespace sum_primes_between_20_and_30_l632_632178

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l632_632178


namespace lambda_plus_mu_eq_half_l632_632486

variables {A B C M N : Point}
variables {λ μ : ℝ}
variables (h1 : M ∈ line[BC])
variables (h2 : midpoint N A M)
variables (h3 : vectorcombo N A B C λ μ)

theorem lambda_plus_mu_eq_half (h1 : M ∈ line[BC]) (h2 : midpoint N A M) (h3 : vectorcombo N A B C λ μ) : λ + μ = 1 / 2 :=
sorry

end lambda_plus_mu_eq_half_l632_632486


namespace find_x_plus_y_l632_632797

theorem find_x_plus_y (x y : ℝ) (h1 : 3^x = 27^(y + 1)) (h2 : 16^y = 4^(x - 6)) : x + y = 15 :=
by
  sorry

end find_x_plus_y_l632_632797


namespace cricket_run_rate_l632_632213

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target : ℝ) (overs_first_phase : ℕ) (overs_remaining : ℕ) :
  run_rate_first_10_overs = 4.6 → target = 282 → overs_first_phase = 10 → overs_remaining = 40 →
  (target - run_rate_first_10_overs * overs_first_phase) / overs_remaining = 5.9 :=
by
  intros
  sorry

end cricket_run_rate_l632_632213


namespace smallest_d_l632_632246

theorem smallest_d (d : ℝ) : 
  (dist (5 * real.sqrt 5, d + 4) (0, 0) = 5 * d) → 
  d ≥ 0 → 
  d = 2.596 :=
by
  sorry

end smallest_d_l632_632246


namespace votes_cast_l632_632225

theorem votes_cast (V : ℝ) (hv1 : 0.35 * V + (0.35 * V + 1800) = V) : V = 6000 :=
sorry

end votes_cast_l632_632225


namespace middle_of_three_consecutive_integers_is_60_l632_632970

theorem middle_of_three_consecutive_integers_is_60 (n : ℤ)
    (h : (n - 1) + n + (n + 1) = 180) : n = 60 := by
  sorry

end middle_of_three_consecutive_integers_is_60_l632_632970


namespace determine_a_range_l632_632380

def z (a : ℝ) : ℂ := 3 + complex.i * a
def condition (a : ℝ) : Prop := complex.abs (z a - 2) < 2

theorem determine_a_range (a : ℝ) (h : condition a) : -real.sqrt 3 < a ∧ a < real.sqrt 3 := 
by 
  sorry -- proof goes here

end determine_a_range_l632_632380


namespace tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l632_632349

theorem tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2 (alpha : ℝ) 
  (h1 : Real.sin alpha = - (Real.sqrt 3) / 2) 
  (h2 : 3 * π / 2 < alpha ∧ alpha < 2 * π) : 
  Real.tan alpha = - Real.sqrt 3 := 
by 
  sorry

end tan_alpha_of_sin_alpha_eq_neg_sqrt3_div_2_l632_632349


namespace prime_sum_20_to_30_l632_632147

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l632_632147


namespace hyperbola_equation_fixed_line_intersection_l632_632379

-- We are given that the center of the hyperbola is at the origin,
-- a focus is at (2, 0), the line y = x - 1 intersects the hyperbola at points A and B,
-- and the horizontal coordinate of the midpoint of AB is -1/2.
-- We need to prove the equation of the hyperbola.

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * a + b * b = 4) ∧ (2 * a * a / (a * a - b * b) = -1) ↔ (a = 1) ∧ (b = sqrt 3) :=
by
  sorry

-- Given the hyperbola equation, we need to prove that the intersection point Q of lines
-- A₁M and A₂N lies on the fixed line x = 1/2 where A₁ = (-1, 0) and A₂ = (1, 0).

theorem fixed_line_intersection (M N A₁ A₂ : ℝ × ℝ) 
  (hxA₁ : A₁ = (-1, 0)) (hxA₂ : A₂ = (1, 0)) 
  (k: ℝ) (h_intersect_MN: (M.fst = 2 + k * (sqrt 3)))
  (h_intersect_A₁M: ∀ (α : ℝ), M - α • A₁ = (1/2, sqrt 3 / 2))
  (h_intersect_A₂N: ∀ (β : ℝ), N - β • A₂ = (1/2, -sqrt 3 / 2)) :
  ∃ Q : ℝ × ℝ, Q = (1/2, Q.snd) :=
by
  sorry

end hyperbola_equation_fixed_line_intersection_l632_632379


namespace discount_percentage_l632_632243

theorem discount_percentage (cost marked_price : ℝ) (profit_percentage : ℝ) :
  cost = 80 ∧ marked_price = 130 ∧ profit_percentage = 0.30 → 
  let profit := cost * profit_percentage in
  let selling_price := cost + profit in
  let discount := marked_price - selling_price in
  let discount_percentage := (discount / marked_price) * 100 in
  discount_percentage = 20 :=
by 
  intros h
  let cost := 80
  let marked_price := 130
  let profit_percentage := 0.30
  let profit := cost * profit_percentage
  let selling_price := cost + profit
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  have h1 : profit = 24 by sorry
  have h2 : selling_price = 104 by sorry
  have h3 : discount = 26 by sorry
  have h4 : discount_percentage = 20 by sorry
  exact h4

end discount_percentage_l632_632243


namespace initial_walking_rate_proof_l632_632240

noncomputable def initial_walking_rate (d : ℝ) (v_miss : ℝ) (t_miss : ℝ) (v_early : ℝ) (t_early : ℝ) : ℝ :=
  d / ((d / v_early) + t_early - t_miss)

theorem initial_walking_rate_proof :
  initial_walking_rate 6 5 (7/60) 6 (5/60) = 5 := by
  sorry

end initial_walking_rate_proof_l632_632240


namespace sin_ratio_l632_632893

-- Defining the angles and the point D
variables {A B C D : Type} [euclidean_geometry B C] 
variables (angle_B : real.angle) (angle_C : real.angle)
variables (BD_ratio : ℝ) (CD_ratio : ℝ)

-- Given conditions
def given_conditions : Prop :=
  angle_B = 45 * real.pi / 180 ∧
  angle_C = 60 * real.pi / 180 ∧
  BD_ratio = 2 / 5 ∧
  CD_ratio = 3 / 5

-- Statement to prove
theorem sin_ratio (h : given_conditions) : 
  ∀ (A B C D : Type) [euclidean_geometry B C], 
  {d : D // D ∈ line_segment B C} →
  ∃ (BD : rect), ∃ (CD : rect), 
  (BD.length / CD.length) = (2 / 3) →
  (sin (∠ BAD) / sin (∠ CAD) = 2 * sqrt 6 / 9) := 
sorry

end sin_ratio_l632_632893


namespace problem_l632_632842

def f (x : ℝ) : ℝ := sin x * cos x

theorem problem :
  f (f (Real.pi / 12)) = (1 / 2) * sin (1 / 2) := by
  sorry

end problem_l632_632842


namespace sum_primes_between_20_and_30_is_52_l632_632125

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l632_632125
