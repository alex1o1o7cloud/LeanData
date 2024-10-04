import Mathlib

namespace max_value_a_avoiding_lattice_points_l572_572911

theorem max_value_a_avoiding_lattice_points (a : ℚ) : 
  (∀ m : ℚ, (2/3 < m) ∧ (m < a) → (∀ x : ℤ, (0 < x) ∧ (x ≤ 150) → ((m * x + 3) ∉ set_of (int))) ∧ (a = 152/151)) =
  true :=
sorry

end max_value_a_avoiding_lattice_points_l572_572911


namespace tan_product_l572_572144

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572144


namespace smallest_lambda_l572_572217

variable {n : ℕ} (a b : Fin n → ℝ)

theorem smallest_lambda (n : ℕ) (hn : n ≥ 2) (ha_pos : ∀ i, 0 < a i) (hb_interval : ∀ i, 0 ≤ b i ∧ b i ≤ 1/2)
  (ha_sum : ∑ i, a i = 1) (hb_sum : ∑ i, b i = 1) :
  ∃ λ, λ = 1 / 2 * (1 / (n - 1)) ^ (n - 1) ∧ (∀ a b, (∀ i, 0 < a i) → (∀ i, 0 ≤ b i ∧ b i ≤ 1/2) → (∑ i, a i = 1) → (∑ i, b i = 1) →
  ((∏ i, a i) ≤ λ * ∑ i, a i * b i)) := 
sorry

end smallest_lambda_l572_572217


namespace number_of_parallelograms_l572_572553

-- Problem statement in Lean 4
theorem number_of_parallelograms (n : ℕ) : 
  let k := n + 1 in
  -- Number of parallelograms formed
  3 * (n * (n - 1) / 2) = 3 * nat.choose n 2 :=
by sorry

end number_of_parallelograms_l572_572553


namespace right_triangle_perimeter_l572_572936

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572936


namespace area_and_cost_of_path_l572_572450

-- Define the dimensions of the grass field
def length_field : ℝ := 85
def width_field : ℝ := 55

-- Define the width of the path around the field
def width_path : ℝ := 2.5

-- Define the cost per square meter of constructing the path
def cost_per_sqm : ℝ := 2

-- Define new dimensions including the path
def new_length : ℝ := length_field + 2 * width_path
def new_width : ℝ := width_field + 2 * width_path

-- Define the area of the entire field including the path
def area_with_path : ℝ := new_length * new_width

-- Define the area of the grass field without the path
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_with_path - area_field

-- Define the cost of constructing the path
def cost_constructing_path : ℝ := area_path * cost_per_sqm

-- Theorem to prove the area of the path and cost of constructing it
theorem area_and_cost_of_path :
  area_path = 725 ∧ cost_constructing_path = 1450 :=
by
  -- Skipping the proof as instructed
  sorry

end area_and_cost_of_path_l572_572450


namespace parallel_vectors_l572_572357

-- Definitions and conditions
variables (n : ℝ) (a b c : ℝ × ℝ)
def vec_a : ℝ × ℝ := (n, -1)
def vec_b : ℝ × ℝ := (-1, 1)
def vec_c : ℝ × ℝ := (-1, 2)
def vec_sum := vec_a n + vec_b

-- Statement of the equivalent proof problem
theorem parallel_vectors (h : vec_sum n = n * vec_c) : n = 1 := by
  sorry

end parallel_vectors_l572_572357


namespace inequality_sum_l572_572617

variable {a b c d : ℝ}

theorem inequality_sum (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by {
  sorry
}

end inequality_sum_l572_572617


namespace doughnut_machine_completion_time_l572_572463

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end doughnut_machine_completion_time_l572_572463


namespace graph_asymptote_ratio_l572_572256

theorem graph_asymptote_ratio :
  ∀ (p q : ℝ → ℝ),
  (∃ k : ℝ, p = λ x, k * x) ∧ q = (λ x, (x + 4) * (x - 1))
  ∧ (p 0 = 0) ∧ (p 2 / q 2 = -2)
  ∧ (q 0 ≠ 0) ∧ (q 2 ≠ 0) →
  (p 3 / q 3 = -9/7) :=
by sorry

end graph_asymptote_ratio_l572_572256


namespace sum_of_divisors_of_30_not_perfect_square_l572_572202

theorem sum_of_divisors_of_30_not_perfect_square 
: ∃ s:ℕ, (s = (∑ d in finset.filter (λ d, 30 % d = 0) (finset.range (30 + 1)), d)) ∧ s = 72 ∧ ¬ ∃ n:ℕ, n * n = s :=
by
  sorry

end sum_of_divisors_of_30_not_perfect_square_l572_572202


namespace smallest_d_for_inverse_l572_572345

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 4

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x y ∈ set.Ici d, g x = g y → x = y) ∧ d = 3 :=
begin
  -- Proof would go here
  sorry
end

end smallest_d_for_inverse_l572_572345


namespace perimeter_of_given_triangle_l572_572933

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572933


namespace roots_distribution_l572_572990

noncomputable def polynomial_roots : Polynomial ℝ :=
  Polynomial.Coeff (x^3 + 3x^2 - 4x + 12)
  
theorem roots_distribution : 
  (polynomial_roots.has_one_positive_real_root ∧ polynomial_roots.has_two_negative_real_roots) :=
sorry

end roots_distribution_l572_572990


namespace profit_days_l572_572404

theorem profit_days (total_days : ℕ) (mean_profit_month first_half_days second_half_days : ℕ)
  (mean_profit_first_half mean_profit_second_half : ℕ)
  (h1 : mean_profit_month * total_days = (mean_profit_first_half * first_half_days + mean_profit_second_half * second_half_days))
  (h2 : first_half_days + second_half_days = total_days)
  (h3 : mean_profit_month = 350)
  (h4 : mean_profit_first_half = 225)
  (h5 : mean_profit_second_half = 475)
  (h6 : total_days = 30) : 
  first_half_days = 15 ∧ second_half_days = 15 := 
by 
  sorry

end profit_days_l572_572404


namespace box_volume_increase_l572_572924

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5400)
  (h2 : l * w + w * h + h * l = 960)
  (h3 : l + w + h = 60) :
  let l' := l + 2
  let w' := w + 2
  let h' := h + 2
  V' = (l' * w' * h') = 7568 :=
by 
  let l' := l + 2
  let w' := w + 2
  let h' := h + 2
  calc
    V' = l' * w' * h' : sorry 
    ... = 7568       : sorry

end box_volume_increase_l572_572924


namespace mike_practice_hours_l572_572362

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l572_572362


namespace divisors_not_divisible_by_3_l572_572679

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572679


namespace trader_discount_l572_572890

-- Definitions based on the conditions
def CP : ℝ := 100
def MP : ℝ := CP + (10 / 100 * CP)
def Loss : ℝ := 1 / 100 * CP
def SP : ℝ := CP - Loss
def Discount : ℝ := MP - SP

-- Statement of the proof problem
theorem trader_discount : Discount = 11 := by
  sorry

end trader_discount_l572_572890


namespace incorrect_option_B_l572_572719

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + cos (2 * x + π / 3)

noncomputable def g (x : ℝ) : ℝ := -sqrt 3 * sin (2 * x) + 1

-- Problem to prove Option B is incorrect
theorem incorrect_option_B : ∀ x : ℝ, (x >= π / 6 ∧ x <= π / 2) → g(x) ≠ -1 / 2 :=
by
  intro x h
  sorry

end incorrect_option_B_l572_572719


namespace x_minus_q_eq_3_minus_2q_l572_572705

theorem x_minus_q_eq_3_minus_2q {x q : ℝ} (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_eq_3_minus_2q_l572_572705


namespace monotonicity_of_f_range_of_h_l572_572633

section monotonicity

variable {a x : ℝ}

/-- Define the function f(x) = (a-x)/(x+1) * e^(2x) for x > -1 -/
def f (a x : ℝ) : ℝ := (a - x) / (x + 1) * Real.exp (2 * x)

theorem monotonicity_of_f (a : ℝ) : 
  (-1 ≤ a ∧ a ≤ 1 → ∀ x > -1, ∀ y > -1, x ≤ y → f a x ≤ f a y) ∧ 
  (a < -1 → ∀ x > (a - 1 + Real.sqrt (a^2 - 1)) / 2, ∀ y > (a - 1 + Real.sqrt (a^2 - 1)) / 2, x ≤ y → f a x ≤ f a y ∧
  ∀ x > -1 ∧ x < (a - 1 + Real.sqrt (a^2 - 1)) / 2, ∀ y > -1 ∧ y < (a - 1 + Real.sqrt (a^2 - 1)) / 2, x ≤ y → f a x ≤ f a y) ∧ 
  (a > 1 → ∀ x > -1 ∧ x < (a - 1 - Real.sqrt (a^2 - 1)) / 2, ∀ y > -1 ∧ y < (a - 1 - Real.sqrt (a^2 - 1)) / 2, x ≤ y → f a x ≤ f a y ∧
  ∀ x > (a - 1 + Real.sqrt (a^2 - 1)) / 2, ∀ y > (a - 1 + Real.sqrt (a^2 - 1)) / 2, x ≤ y → f a x ≤ f a y ∧
  ∀ x > (a - 1 - Real.sqrt (a^2 - 1)) / 2 ∧ x < (a - 1 + Real.sqrt (a^2 - 1)) / 2, ∀ y > (a - 1 - Real.sqrt (a^2 - 1)) / 2 ∧ y < (a - 1 + Real.sqrt (a^2 - 1)) / 2, x ≤ y → f a x ≤ f a y) :=
sorry 

end monotonicity

section maximum_of_g

variable {m x : ℝ}

/-- Define the function g(x) = (2mx+m-e^(2x))/x^2 for x > 0 -/
def g (m x : ℝ) : ℝ := (2 * m * x + m - Real.exp (2 * x)) / x^2

/-- Define the function h(x) -/
def h (m : ℝ) (x : ℝ) : ℝ := 
  (m * (2 * x + 1) - Real.exp (2 * x)) / x^2

noncomputable def p (x : ℝ) : ℝ := -2 / (1 + x) * Real.exp (2 * x)

theorem range_of_h (m : ℝ) (hm : 0 ≤ m ∧ m < 1) : 
  ∃ x_a : ℝ, (0 < x_a ∧ x_a ≤ 1 ∧ f 1 x_a = m) ∧ (h m x_a = h m x_a) ∧
  -Real.exp (2 : ℝ) ≤ h m x_a ∧ h m x_a < -2 :=
sorry

end maximum_of_g

end monotonicity_of_f_range_of_h_l572_572633


namespace correct_statements_l572_572216

namespace Geometry

-- Definitions
variable {Point Line Plane : Type}
variable {contains : Plane → Line → Prop}
variable {perpendicular parallel : Line → Line → Prop}
variable {perpendicular_plane parallel_plane : Line → Plane → Prop}

-- Given conditions:
variable {l m : Line}
variable {a : Plane}
variable (not_contained : ¬ contains a l)
variable (contained : contains a m)

-- Statements to be proved:
lemma statement1 (h : perpendicular_plane l a) : perpendicular l m := 
sorry

lemma statement2 (h : parallel_plane l a) : ¬ parallel l m :=
sorry

lemma statement3 (h : perpendicular l m) : ¬ perpendicular_plane l a :=
sorry

lemma statement4 (h : parallel l m) : parallel_plane l a :=
sorry

-- Proving that the correct combination is exactly (statements 1, 4)
theorem correct_statements : 
  (statement1 not_contained contained) ∧ 
  (statement4 not_contained contained) ∧ 
  ¬ (statement2 not_contained contained) ∧ 
  ¬ (statement3 not_contained contained) := 
by sorry

end Geometry

end correct_statements_l572_572216


namespace tan_product_l572_572134

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572134


namespace measure_angle_B_area_of_triangle_l572_572725

-- Definitions
variables (a b c : ℝ) (A B C : ℝ)
variables (triangle : ℝ → ℝ → ℝ → Prop)
variables (cos_ratio : ℝ → ℝ → ℝ → ℝ → Prop)
variables (angle_measure : ℝ → ℝ)

-- Conditions
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  triangle a b c ∧ 
  (cos_ratio B C b (2 * a + c))

-- First Proof: Find the measure of angle B
theorem measure_angle_B (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  B = 2 * Real.pi / 3 :=
sorry

-- Additional Variables and Definitions for Second Part
variables (area : ℝ → ℝ → ℝ → ℝ → Prop)

-- Second Proof: Find the area of the triangle given specific values
theorem area_of_triangle (b a c A B C : ℝ) (h1 : b = Real.sqrt 13)
                                      (h2 : a + c = 4)
                                      (h3 : B = 2 * Real.pi / 3)
                                      : area ABC a b c = 3 * Real.sqrt 3 / 4 :=
sorry

end measure_angle_B_area_of_triangle_l572_572725


namespace sequence_formula_l572_572219

-- Define S_n
def S (n : ℕ) : ℕ := 4 * n^2 - n + 2

-- Define the sequence a_n based on S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 5 else 8 * n - 5

-- Statement of the theorem
theorem sequence_formula (n : ℕ) (n_pos : 0 < n) : 
  (if n = 1 then S n else S n - S (n - 1) = a n) :=
by
  sorry

end sequence_formula_l572_572219


namespace arrangement_of_ABC_l572_572817

theorem arrangement_of_ABC (n k : ℕ) (h₁ : n = 6) (h₂ : k = 3) 
  (h₃ : ¬ (A ≠  A) → True) -- This step ensures A, B, C are non-adjacent 
  : 
  (factorial k) * (choose (n - k + 1) k) = 144
  :=
by
  rw [h₁, h₂]
  -- we would need actual definitions and valid assumptions for A and B,
  -- the sorry just skips to the expected result
  sorry

end arrangement_of_ABC_l572_572817


namespace draw_red_one_more_time_than_white_l572_572283

theorem draw_red_one_more_time_than_white:
  let outcomes := [(ℝ, ℝ, ℝ), (ℝ, ℝ, 𝓌), (ℝ, 𝓌, ℝ), (ℝ, 𝓌, 𝓌), (𝓌, ℝ, ℝ), (𝓌, ℝ, 𝓌), (𝓌, 𝓌, ℝ), (𝓌, 𝓌, 𝓌)] in
  let favorable := [(ℝ, ℝ, 𝓌), (ℝ, 𝓌, ℝ), (𝓌, ℝ, ℝ)] in
  (favorable.length / outcomes.length) = (3 / 8) := 
by
  sorry

end draw_red_one_more_time_than_white_l572_572283


namespace HCF_of_two_numbers_is_20_l572_572823

noncomputable def HCF_LCM_problem : Prop :=
  ∃ (a b : ℕ),
  nat.gcd a b = 20 ∧
  nat.lcm a b = 396 ∧
  (a = 36 ∧ b = 220 ∨ a = 220 ∧ b = 36)

theorem HCF_of_two_numbers_is_20 : HCF_LCM_problem :=
by {
  use [36, 220],
  split,
  { -- Proof that the HCF of 36 and 220 is 20
    exact nat.gcd_eq_gcd 36 220 20 sorry, -- Provides the gcd of 36 and 220 as 20; needs proof
  },
  split,
  { -- Proof that the LCM of 36 and 220 is 396
    exact nat.lcm_eq_lcm 36 220 396 sorry, -- Provides the lcm of 36 and 220 as 396; needs proof
  },
  { -- Ensure both numbers are covered (36, 220) and (220, 36)
    left,
    exact (and.intro rfl rfl),
  }
}

end HCF_of_two_numbers_is_20_l572_572823


namespace sum_of_digits_0_to_2012_l572_572319

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l572_572319


namespace sum_of_numbers_l572_572879

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l572_572879


namespace div_40_iff_pmod5_l572_572340

theorem div_40_iff_pmod5 (p : ℕ) (hp_prime : p.prime) (hp_ge_7 : p ≥ 7) :
  40 ∣ (p^2 - 1) ↔ (p % 5 = 1 ∨ p % 5 = 4) :=
sorry

end div_40_iff_pmod5_l572_572340


namespace total_sum_valid_digits_l572_572359

-- Defining the main condition: a digit is valid if it is one of [1, 2, 4, 5, 7, 8]
def valid_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 8

-- Define a function to check if all digits of a number are valid
def valid_number (n : ℕ) : Prop :=
  let digits := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  in ∀ digit ∈ digits, valid_digit digit

-- Define the task to prove the total sum of the digits of all valid five-digit numbers
theorem total_sum_valid_digits :
  (∑ n in list.range (100000), if valid_number n then (n / 10000 % 10 + n / 1000 % 10 + n / 100 % 10 + n / 10 % 10 + n % 10) else 0) = 174960 :=
by sorry

end total_sum_valid_digits_l572_572359


namespace nth_monomial_in_sequence_l572_572959

variable (a : ℝ)

def sequence (n : ℕ) : ℝ :=
  (-1 : ℝ) ^ n * (n + 1) * a ^ (2 * n)

theorem nth_monomial_in_sequence (n : ℕ) :
  sequence a n = (-1 : ℝ) ^ n * (n + 1) * a ^ (2 * n) :=
by 
  sorry

end nth_monomial_in_sequence_l572_572959


namespace tan_product_pi_nine_l572_572124

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572124


namespace lily_milk_quantity_l572_572358

theorem lily_milk_quantity :
  let init_gallons := (5 : ℝ)
  let given_away := (18 / 4 : ℝ)
  let received_back := (7 / 4 : ℝ)
  init_gallons - given_away + received_back = 2 + 1 / 4 :=
by
  sorry

end lily_milk_quantity_l572_572358


namespace midpoint_C_l572_572379

variables (A B C : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (AC CB : ℝ)

def segment_division (A B C : ℝ × ℝ) (m n : ℝ) : Prop :=
  C = ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n))

theorem midpoint_C :
  A = (-2, 1) →
  B = (4, 9) →
  AC = 2 * CB →
  segment_division A B C 2 1 →
  C = (2, 19 / 3) :=
by
  sorry

end midpoint_C_l572_572379


namespace right_triangle_perimeter_l572_572938

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572938


namespace bug_reaches_opposite_vertex_l572_572030

noncomputable def probability_bug_reaches_opposite_vertex : ℚ :=
  -- Define the structure of the cube and its vertices A, B, C, D, E, F, G, H.
  let vertices := {A, B, C, D, E, F, G, H} in
  -- Define the opposite vertex pairs.
  let opposite : vertices → vertices := λ v, match v with
    | A => G | G => A
    | B => H | H => B
    | C => E | E => C
    | D => F | F => D
  end in
  -- Initial setup of the bug's position at vertex A.
  let start := A in
  -- Define the number of moves.
  let moves := 6 in
  -- Calculate total possible paths.
  let total_paths := (3 : ℚ) ^ moves in
  -- Define the count of valid paths leading to the opposite vertex.
  let valid_paths := 91 in
  -- Probability of reaching the opposite vertex after six moves.
  valid_paths / total_paths

-- The main theorem to prove.
theorem bug_reaches_opposite_vertex :
  probability_bug_reaches_opposite_vertex = 91 / 729 :=
by sorry

end bug_reaches_opposite_vertex_l572_572030


namespace triangle_DEF_area_l572_572836

theorem triangle_DEF_area (a b c : ℝ) (ha : a = 4) (hb : b = 6) (hc : c = 8) :
  let s := (a + b + c) / 2,
      S_ABC := real.sqrt (s * (s - a) * (s - b) * (s - c)),
      AD := (s - a), AF := (s - a),
      BD := (s - b), BE := (s - b),
      CE := (s - c), CF := (s - c),
      x := (s - a), y := (s - b), z := (s - c)
  in S_ABC - (S_ABC * (AD / a) * (AF / c)
              + S_ABC * (BD / a) * (BE / b)
              + S_ABC * (CE / b) * (CF / c)) = (15 * real.sqrt(15)) / 32 :=
by
  sorry

end triangle_DEF_area_l572_572836


namespace sum_squares_of_digits_in_product_l572_572980

theorem sum_squares_of_digits_in_product :
  let nines := list.repeat 9 120
  let ones := list.repeat 1 120
  let product := nines
  let sum_squares := list.sum (product.map (λ d, d * d))
  sum_squares = 9720 :=
sorry

end sum_squares_of_digits_in_product_l572_572980


namespace central_angle_change_l572_572409

theorem central_angle_change {r l : ℝ} (h : r > 0) (l > 0) : 
  let θ := (180 * l) / (π * r) in
  let new_r := 3 * r in
  let x := (60 * l) / (π * r) in
  x = θ / 3 := 
by 
  sorry

end central_angle_change_l572_572409


namespace problem1_problem2_l572_572225

theorem problem1 (a : ℝ) (A B : set ℝ)
  (hA : A = {x : ℝ | x ≤ a + 3})
  (hB : B = {x : ℝ | x < -1 ∨ x > 5})
  (ha : a = -2) :
  A ∩ Bᶜ = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by
  -- the proof will go here
  sorry

theorem problem2 (a : ℝ) (A B : set ℝ)
  (hA : A = {x : ℝ | x ≤ a + 3})
  (hB : B = {x : ℝ | x < -1 ∨ x > 5})
  (hAB : A ⊆ B) :
  a < -4 :=
by
  -- the proof will go here
  sorry

end problem1_problem2_l572_572225


namespace max_elements_of_S_l572_572754

noncomputable def S (a b c : ℕ) : Set ℚ :=
  {r : ℚ | ∃ p q r, (p, q, r) ∈ ([a, b, c].permutations : Finset (ℕ × ℕ × ℕ)) ∧ root (s.mk p q r)}


def root (p : Polynomial ℤ) (x: ℚ) : Prop :=
(p.eval x).normalize = 0

theorem max_elements_of_S : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ # (S a b c) = 8 := sorry

end max_elements_of_S_l572_572754


namespace rounding_proofs_l572_572824

def num : ℝ := 6.4349
def hundredth_rounded_value : ℝ := 6.44
def four_sig_figures_rounded_value : ℝ := 6.435
def thousandth_rounded_value : ℝ := 6.435

theorem rounding_proofs :
  (num.round_to 0.01 = hundredth_rounded_value) ∧
  (num.round_to_significant_figures 4 = four_sig_figures_rounded_value) ∧
  (num.round_to 0.001 = thousandth_rounded_value) :=
by
  sorry

end rounding_proofs_l572_572824


namespace solve_f_log2_20_l572_572237

noncomputable def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then 2^x else 0 -- Placeholder for other values

theorem solve_f_log2_20 :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 4) = f x) →
  (∀ x, -1 ≤ x ∧ x < 0 → f x = 2^x) →
  f (Real.log 20 / Real.log 2) = -4 / 5 :=
by
  sorry

end solve_f_log2_20_l572_572237


namespace free_roofing_correct_l572_572366

open Real

-- Definitions of the given conditions
def totalRoofingFeet : ℝ := 300
def costPerFoot : ℝ := 8
def paidAmount : ℝ := 400

-- Define the amount of roofing paid for
def paidRoofingFeet : ℝ := paidAmount / costPerFoot

-- Define the total roofing bought and to find remaining roofing
def freeRoofingFeet : ℝ := totalRoofingFeet - paidRoofingFeet

-- Theorem stating the amount of free roofing is 250 feet
theorem free_roofing_correct :
  freeRoofingFeet = 250 := by
  sorry

end free_roofing_correct_l572_572366


namespace chimps_moved_l572_572396

theorem chimps_moved (total_chimps : ℕ) (chimps_staying : ℕ) (chimps_moved : ℕ) 
  (h_total : total_chimps = 45)
  (h_staying : chimps_staying = 27) :
  chimps_moved = 18 :=
by
  sorry

end chimps_moved_l572_572396


namespace total_animals_correct_l572_572514

section 
variable 
  (snakes : ℕ)
  (arctic_foxes : ℕ)
  (leopards : ℕ)
  (bee_eaters : ℕ)
  (cheetahs : ℕ)
  (alligators : ℕ)
  (total : ℕ)

-- Define the initial counts
def snakes := 100
def arctic_foxes := 80
def leopards := 20

-- Define the derived counts based on conditions
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

-- Define the total number of animals based on the derived counts
def total := alligators + bee_eaters + snakes + arctic_foxes + leopards + cheetahs

-- Proof objective
theorem total_animals_correct : total = 670 := by 
  sorry
end

end total_animals_correct_l572_572514


namespace number_of_positive_integer_solutions_l572_572019

theorem number_of_positive_integer_solutions :
    ∃ x y z : ℕ, (xy + xz = 255 ∧ xy + yz = 31) ∧
        (card {p : ℕ × ℕ × ℕ | (p.1 * p.2 + p.1 * p.3) = 255 ∧ (p.1 * p.2 + p.2 * p.3) = 31}) = 2 :=
sorry

end number_of_positive_integer_solutions_l572_572019


namespace tan_product_l572_572090

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572090


namespace initial_pencils_on_desk_l572_572419

theorem initial_pencils_on_desk
  (pencils_in_drawer: ℕ)
  (pencils_added: ℕ)
  (total_pencils: ℕ)
  (total_after_adding: ℕ) :
  pencils_in_drawer = 43 →
  pencils_added = 16 →
  total_pencils = 78 →
  total_after_adding = total_pencils - pencils_in_drawer →
  total_after_adding - pencils_added = 19 :=
by
  intro h_pencils_in_drawer h_pencils_added h_total_pencils h_total_after_adding
  rw [h_pencils_in_drawer, h_pencils_added, h_total_pencils] at h_total_after_adding
  simp at h_total_after_adding
  exact h_total_after_adding

end initial_pencils_on_desk_l572_572419


namespace divisors_of_180_not_divisible_by_3_l572_572683

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572683


namespace mean_of_four_numbers_l572_572826

theorem mean_of_four_numbers (a b c d : ℝ) (h : (a + b + c + d + 130) / 5 = 90) : (a + b + c + d) / 4 = 80 := by
  sorry

end mean_of_four_numbers_l572_572826


namespace total_cost_full_units_l572_572031

def total_units : Nat := 12
def cost_1_bedroom : Nat := 360
def cost_2_bedroom : Nat := 450
def num_2_bedroom : Nat := 7
def num_1_bedroom : Nat := total_units - num_2_bedroom

def total_cost : Nat := (num_1_bedroom * cost_1_bedroom) + (num_2_bedroom * cost_2_bedroom)

theorem total_cost_full_units : total_cost = 4950 := by
  -- proof would go here
  sorry

end total_cost_full_units_l572_572031


namespace percent_income_left_l572_572916

-- Define the conditions
def I : ℝ := 100       -- Total income
def f : ℝ := 0.5       -- Percent spent on food
def e : ℝ := 0.15      -- Percent spent on education
def r : ℝ := 0.5       -- Percent spent on house rent of the remaining

-- Define the problem: Prove the final percent left is 17.5%
theorem percent_income_left (I f e r : ℝ) (hI : I = 100) (hf : f = 0.5) (he : e = 0.15) (hr : r = 0.5) :
  let spent_food := f * I,
      spent_education := e * I,
      remaining_after_food_and_education := I - spent_food - spent_education,
      spent_rent := r * remaining_after_food_and_education,
      final_remaining := remaining_after_food_and_education - spent_rent in
  (final_remaining / I) * 100 = 17.5 := by
  sorry

end percent_income_left_l572_572916


namespace competition_total_races_l572_572284

theorem competition_total_races (sprinters : ℕ) (sprinters_with_bye : ℕ) (lanes_preliminary : ℕ) (lanes_subsequent : ℕ) 
  (eliminated_per_race : ℕ) (first_round_advance : ℕ) (second_round_advance : ℕ) (third_round_advance : ℕ) 
  : sprinters = 300 → sprinters_with_bye = 16 → lanes_preliminary = 8 → lanes_subsequent = 6 → 
    eliminated_per_race = 7 → first_round_advance = 36 → second_round_advance = 9 → third_round_advance = 2 
    → first_round_races = 36 → second_round_races = 9 → third_round_races = 2 → final_race = 1
    → first_round_races + second_round_races + third_round_races + final_race = 48 :=
by 
  intros sprinters_eq sprinters_with_bye_eq lanes_preliminary_eq lanes_subsequent_eq eliminated_per_race_eq 
         first_round_advance_eq second_round_advance_eq third_round_advance_eq 
         first_round_races_eq second_round_races_eq third_round_races_eq final_race_eq
  sorry

end competition_total_races_l572_572284


namespace regular_icosahedron_edges_l572_572649

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end regular_icosahedron_edges_l572_572649


namespace percentage_increase_is_20_l572_572448

noncomputable def total_stocks : ℕ := 1980
noncomputable def stocks_higher : ℕ := 1080
noncomputable def stocks_lower : ℕ := total_stocks - stocks_higher

/--
Given that the total number of stocks is 1,980, and 1,080 stocks closed at a higher price today than yesterday.
Furthermore, the number of stocks that closed higher today is greater than the number that closed lower.

Prove that the percentage increase in the number of stocks that closed at a higher price today compared to the number that closed at a lower price is 20%.
-/
theorem percentage_increase_is_20 :
  (stocks_higher - stocks_lower) / stocks_lower * 100 = 20 := by
  sorry

end percentage_increase_is_20_l572_572448


namespace cost_per_yellow_shirt_l572_572397

theorem cost_per_yellow_shirt(
    (num_orange_shirts : ℕ) (cost_per_orange_shirt : ℝ)
    (num_yellow_shirts : ℕ)
    (num_blue_shirts : ℕ) (cost_per_blue_shirt : ℝ)
    (num_green_shirts : ℕ) (cost_per_green_shirt : ℝ)
    (total_spent : ℝ)
    (h1 : num_orange_shirts = 101) (h2 : cost_per_orange_shirt = 5.80)
    (h3 : num_yellow_shirts = 113)
    (h4 : num_blue_shirts = 107) (h5 : cost_per_blue_shirt = 5.60)
    (h6 : num_green_shirts = 108) (h7 : cost_per_green_shirt = 5.25)
    (h8 : total_spent = 2317.00)
    : (total_spent - (num_orange_shirts * cost_per_orange_shirt +
                     num_blue_shirts * cost_per_blue_shirt +
                     num_green_shirts * cost_per_green_shirt)) / num_yellow_shirts = 5.00 := 
by sorry

end cost_per_yellow_shirt_l572_572397


namespace continuous_integral_condition_constant_l572_572324

theorem continuous_integral_condition_constant (f : ℝ → ℝ) (h_cont : ContinuousOn f (Icc 0 1))
  (h_int : ∀ (g : ℝ → ℝ), ContinuousOn g (Icc 0 1) → 
  (∀ x ∈ Icc 0 1, ¬DifferentiableAt ℝ g x) →
  (∫ x in (Icc (0:ℝ) 1), f x * g x = (∫ x in (Icc (0:ℝ) 1), f x) * (∫ x in (Icc (0:ℝ) 1), g x)))
  : ∀ x y ∈ Icc (0:ℝ) 1, f x = f y := by
  sorry

end continuous_integral_condition_constant_l572_572324


namespace calculate_expression_l572_572981

theorem calculate_expression : |(-5 : ℤ)| - (27 : ℤ).cbrt + (-2 : ℤ)^2 + (4 : ℚ) / (2/3 : ℚ) = 12 := by 
  sorry

end calculate_expression_l572_572981


namespace nine_point_circles_tangent_fixed_circle_l572_572456

-- Definitions of points and circles
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Given conditions
variables (P : Point) (Γ : Circle)

-- Function definition to generate the triangle ABC based on B and C on circle Γ
def triangle (A B C : Point) := (A, B, C)

-- The nine-point circle function definition (hypothetical)
-- In reality, we would need to define this in full in Lean
def nine_point_circle (A B C : Point) : Circle := sorry -- Definition based on triangle ABC

-- The statement to be proved, asserting the nine-point circles touch a fixed circle
theorem nine_point_circles_tangent_fixed_circle
  (A : Point)
  (B C : Point)
  (hB : (B.x - Γ.center.x)^2 + (B.y - Γ.center.y)^2 = Γ.radius^2) -- B on Γ
  (hC : (C.x - Γ.center.x)^2 + (C.y - Γ.center.y)^2 = Γ.radius^2) -- C on Γ
  (fixed_circle : Circle) -- The circle that nine-point circles should touch
  (h_tangent : ∀ B C, nine_point_circle A B C = fixed_circle) -- Tangency condition
: ∀ (B C : Point), (∃ k, nine_point_circle A B C = fixed_circle) :=
sorry

end nine_point_circles_tangent_fixed_circle_l572_572456


namespace total_number_of_animals_is_650_l572_572511

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l572_572511


namespace smallest_value_of_expression_l572_572762

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) (h_ω_4 : ω^4 = 1) (h_neq_1 : ω ≠ 1) (h_a_eq : a = 2 * b - c) : 
  ∃ x : ℝ, x ≥ 0 ∧ ∀ y : ℝ, y = complex.abs(a + b * ω + c * ω^3) → y ≥ x :=
sorry

end smallest_value_of_expression_l572_572762


namespace count_divisors_not_divisible_by_3_l572_572701

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572701


namespace tan_product_pi_nine_l572_572122

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572122


namespace correlation_coefficient_line_l572_572733

theorem correlation_coefficient_line (n : ℕ) (x y : ℕ → ℝ) (h1 : ∀ i : ℕ, i < n → y i = 3 * x i + 1) (h2 : 2 ≤ n) (h3 : ¬ ∀ i j : ℕ, i < n → j < n → x i = x j) :
  correlation_coefficient x y = 1 := 
sorry

end correlation_coefficient_line_l572_572733


namespace satisfies_pmod_3_l572_572803

theorem satisfies_pmod_3 (p : ℕ) (hp : p.prime) (hmod : p % 3 = 1) :
  ∃ x : ℕ, (x^2 + x + 1) % p = 0 := 
sorry

end satisfies_pmod_3_l572_572803


namespace find_number_l572_572900

theorem find_number (x : ℝ) (h : 0.26 * x = 93.6) : x = 360 := sorry

end find_number_l572_572900


namespace coplanar_points_l572_572186

theorem coplanar_points (b : ℝ) :
  let v₁ := ![2, b, 0]
  let v₂ := ![0, 2, b]
  let v₃ := ![b, 0, 2]
  matrix.det ↑![v₁, v₂, v₃] = 0 ↔ b = -2 := 
begin
  sorry
end

end coplanar_points_l572_572186


namespace imaginary_part_of_z_l572_572246

def z : ℂ := (i ^ 2017) / (1 - 2 * i)

theorem imaginary_part_of_z : (z.im = 1 / 5) :=  
by
  sorry

end imaginary_part_of_z_l572_572246


namespace doughnut_completion_l572_572464

theorem doughnut_completion :
  let start_time := 8 * 60 + 30 in -- 8:30 AM in minutes
  let one_third_time := 11 * 60 + 10 - start_time in -- Duration from 8:30 AM to 11:10 AM in minutes
  let total_time := 3 * one_third_time in -- Total time to finish the job
  let completion_time := start_time + total_time in -- Completion time in minutes
  completion_time = 16 * 60 + 30 := -- 4:30 PM in minutes
by
  sorry

end doughnut_completion_l572_572464


namespace bob_has_winning_strategy_l572_572496

theorem bob_has_winning_strategy :
  (∀ a b c : ℕ, a ≤ 2018 → b ≤ 2018 → c ≤ 2018 → a ≠ b → b ≠ c → c ≠ a →
  ¬ (2 * b = a + c)) → (Bob wins) :=
sorry

end bob_has_winning_strategy_l572_572496


namespace chord_length_l572_572913

theorem chord_length (x1 x2 y1 y2 : ℝ) (h1 : y1^2 = 12 * x1) (h2 : y2^2 = 12 * x2) 
  (h3 : x1 + x2 = 6) :
  (abs (sqrt ( (x2 - x1)^2 + (y2 - y1)^2))) = 12 :=
sorry

end chord_length_l572_572913


namespace x_minus_q_eq_3_minus_2q_l572_572710

theorem x_minus_q_eq_3_minus_2q (x q : ℝ) (h₁ : |x - 3| = q) (h₂ : x < 3) : x - q = 3 - 2q :=
by
sorry

end x_minus_q_eq_3_minus_2q_l572_572710


namespace divisors_not_divisible_by_3_eq_6_l572_572660

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572660


namespace remainder_when_divided_by_6_l572_572444

theorem remainder_when_divided_by_6 (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 :=
sorry

end remainder_when_divided_by_6_l572_572444


namespace tangent_product_eq_three_l572_572115

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572115


namespace log_2_64_sqrt_2_l572_572562

theorem log_2_64_sqrt_2 : log 2 (64 * real.sqrt 2) = 13 / 2 :=
by
  have h1 : 64 = 2^6 := by norm_num
  have h2 : real.sqrt 2 = 2^(1/2 : ℝ) := by rw real.sqrt_eq_rpow; norm_num
  sorry

end log_2_64_sqrt_2_l572_572562


namespace will_pages_needed_l572_572888

theorem will_pages_needed :
  let new_cards_2020 := 8
  let old_cards := 10
  let duplicates := 2
  let cards_per_page := 3
  let unique_old_cards := old_cards - duplicates
  let pages_needed_for_2020 := (new_cards_2020 + cards_per_page - 1) / cards_per_page -- ceil(new_cards_2020 / cards_per_page)
  let pages_needed_for_old := (unique_old_cards + cards_per_page - 1) / cards_per_page -- ceil(unique_old_cards / cards_per_page)
  let pages_needed := pages_needed_for_2020 + pages_needed_for_old
  pages_needed = 6 :=
by
  sorry

end will_pages_needed_l572_572888


namespace original_price_l572_572384

theorem original_price (P : ℝ) 
  (h : 0.82 * P * 1.05 * 0.90 = 147.60) : P ≈ 200 :=
sorry

end original_price_l572_572384


namespace induction_step_l572_572866

theorem induction_step (k : ℕ) : ((k + 1 + k) * (k + 1 + k + 1) / (k + 1)) = 2 * (2 * k + 1) := by
  sorry

end induction_step_l572_572866


namespace angle_ABC_is_40_l572_572215

noncomputable def circle_with_inscribed_triangle (O A B C : Type*) (h : Circle O) 
  (triangle_ABC : Triangle A B C) 
  (inscribed : ℕ) (angle_AOB : Real) (angle_BOC : Real) : Prop := 
  ∃ (O : Type*) (A B C : Type*),
    inscribed = 1 ∧ 
    angle_AOB = 150 ∧
    angle_BOC = 130

theorem angle_ABC_is_40 (O A B C : Type*) 
  (h : Circle O) (triangle_ABC : Triangle A B C) :
  circle_with_inscribed_triangle O A B C h triangle_ABC 1 150 130 → 
  ∠ABC = 40 := by
  sorry

end angle_ABC_is_40_l572_572215


namespace num_non_divisible_by_3_divisors_l572_572668

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572668


namespace constant_function_l572_572185

theorem constant_function (f : ℕ → ℤ)
  (h1 : ∀ a b : ℕ, a > 0 → b > 0 → a ∣ b → f(a) ≥ f(b))
  (h2 : ∀ a b : ℕ, a > 0 → b > 0 → f(a * b) + f(a^2 + b^2) = f(a) + f(b)) :
  ∃ C : ℤ, ∀ n : ℕ, f(n) = C :=
by
  sorry

end constant_function_l572_572185


namespace amelia_remaining_money_l572_572052

variable {m b n : ℚ}

theorem amelia_remaining_money (h : (1 / 4) * m = (1 / 2) * n * b) : 
  m - n * b = (1 / 2) * m :=
by
  sorry

end amelia_remaining_money_l572_572052


namespace largest_subset_size_l572_572759

variable {T : set ℕ}

theorem largest_subset_size (hT : T ⊆ { n | 1 ≤ n ∧ n ≤ 2021 })
  (h_diff : ∀ a b ∈ T, a ≠ b → a - b ≠ 5 ∧ a - b ≠ 8) :
  T.size ≤ 1089 :=
sorry

end largest_subset_size_l572_572759


namespace inequality_division_l572_572592

variable {a b c : ℝ}

theorem inequality_division (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : 
  (a / (a - c)) > (b / (b - c)) := 
sorry

end inequality_division_l572_572592


namespace matrix_solution_l572_572572

-- Define the 2x2 matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![ ![30 / 7, -13 / 7], 
     ![-6 / 7, -10 / 7] ]

-- Define the vectors
def vec1 : Fin 2 → ℚ := ![2, 3]
def vec2 : Fin 2 → ℚ := ![4, -1]

-- Expected results
def result1 : Fin 2 → ℚ := ![3, -6]
def result2 : Fin 2 → ℚ := ![19, -2]

-- The proof statement
theorem matrix_solution : (N.mulVec vec1 = result1) ∧ (N.mulVec vec2 = result2) :=
  by sorry

end matrix_solution_l572_572572


namespace solution_for_a_l572_572723

theorem solution_for_a :
  ∀ a x : ℝ, (2 - a - x = 0) ∧ (2x + 1 = 3) → a = 1 := 
by
  intros a x h,
  cases h with h1 h2,
  have x_eq := by linarith,
  have a_eq := by linarith,
  exact a_eq

end solution_for_a_l572_572723


namespace max_g_value_l572_572585

def g (x : ℝ) : ℝ := min (min (3 * x + 3) (1 / 3 * x + 2)) (-2 / 3 * x + 8)

theorem max_g_value : ∃ x : ℝ, g x = 4 :=
by
  sorry

end max_g_value_l572_572585


namespace bugs_eat_total_flowers_l572_572367

def num_bugs : ℝ := 2.0
def flowers_per_bug : ℝ := 1.5
def total_flowers_eaten : ℝ := 3.0

theorem bugs_eat_total_flowers : 
  (num_bugs * flowers_per_bug) = total_flowers_eaten := 
  by 
    sorry

end bugs_eat_total_flowers_l572_572367


namespace exists_unique_number_10079_l572_572534
open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  let str := n.to_string.to_list
  str.foldl (λ acc c, acc + c.to_nat - '0'.to_nat) 0

theorem exists_unique_number_10079 : 
  ∃ n, 
  n > 10000 ∧ 
  (∀ d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], n % d = d - 1) ∧
  (sum_of_digits n).is_prime ∧
  ¬∃ m, m * m = n :=
sorry

end exists_unique_number_10079_l572_572534


namespace difference_mean_median_scores_l572_572790

theorem difference_mean_median_scores :
  let mean : ℚ := (0.15 * 60 + 0.25 * 75 + 0.40 * 85 + 0.20 * 95)
  let median : ℚ := 85
  let diff := median - mean
  diff = 4 := 
by
  -- Define individual percentages as variables for clarity
  let p1 : ℚ := 0.15
  let p2 : ℚ := 0.25
  let p3 : ℚ := 0.40
  let p4 : ℚ := 1 - (p1 + p2 + p3)  -- calculate remaining percentage
  -- Define individual scores
  let s1 : ℚ := 60
  let s2 : ℚ := 75
  let s3 : ℚ := 85
  let s4 : ℚ := 95
  -- Calculate the mean score
  let mean := (p1 * s1 + p2 * s2 + p3 * s3 + p4 * s4)
  have : mean = 80.75 := sorry -- Calculation here is provided in the proof
  -- The median score was already provided
  let median := 85
  -- Calculate the difference
  let diff := median - mean
  show diff = 4 from sorry -- Utilize the results to show the difference is 4

end difference_mean_median_scores_l572_572790


namespace slope_of_line_l572_572201

theorem slope_of_line : ∀ (x y : ℝ), (6 * x + 10 * y = 30) → (y = -((3 / 5) * x) + 3) :=
by
  -- Proof needs to be filled out
  sorry

end slope_of_line_l572_572201


namespace find_line_through_point_bisecting_chord_l572_572576

def point := (ℝ × ℝ)

def parabola (p : point) : Prop := p.2^2 = 4 * p.1

def bisects_chord (l : ℝ → ℝ) (P : point) : Prop :=
  P = (3, -1) ∧ 
  ∃ A B : point, parabola A ∧ parabola B ∧
    (A.1 ≠ B.1) ∧
    (A.2 + B.2) / 2 = P.2 ∧
    l(A.1) = A.2 ∧ l(B.1) = B.2

theorem find_line_through_point_bisecting_chord : 
  ∃ (l : ℝ → ℝ), 
  (∀ x, l x = -2 * x + 7) ∧ 
  bisects_chord l (3, -1) := 
sorry

end find_line_through_point_bisecting_chord_l572_572576


namespace total_animals_count_l572_572516

def snakes := 100
def arctic_foxes := 80
def leopards := 20
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

theorem total_animals_count : snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 := by
  calc
    snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators
        = 100 + 80 + 20 + 200 + 50 + 200 : by sorry -- Use the defined values
    ... = 650 : by sorry -- Perform the final addition

end total_animals_count_l572_572516


namespace tan_product_identity_l572_572108

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572108


namespace simplify_and_evaluate_equals_l572_572811

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l572_572811


namespace bake_sale_ratios_and_revenues_l572_572057

-- Definitions from the conditions
def total_items_sold := 250
def cookies_sold := 50
def brownies_sold := 80
def cupcakes_sold := 60
def cookies_price := 1.50
def brownies_price := 2.00
def cupcakes_price := 2.50
def cake_slices_price := 3.00
def cake_slices_sold := total_items_sold - (cookies_sold + brownies_sold + cupcakes_sold)

-- Ratio calculation
def ratio_cookies := cookies_sold / 10
def ratio_brownies := brownies_sold / 10
def ratio_cupcakes := cupcakes_sold / 10
def ratio_cake_slices := cake_slices_sold / 10

-- Revenue calculation
def revenue_cookies := cookies_sold * cookies_price
def revenue_brownies := brownies_sold * brownies_price
def revenue_cupcakes := cupcakes_sold * cupcakes_price
def revenue_cake_slices := cake_slices_sold * cake_slices_price

theorem bake_sale_ratios_and_revenues :
  (ratio_cookies, ratio_brownies, ratio_cupcakes, ratio_cake_slices) = (5, 8, 6, 6) ∧
  (revenue_cookies, revenue_brownies, revenue_cupcakes, revenue_cake_slices) = (75, 160, 150, 180) :=
by
  sorry

end bake_sale_ratios_and_revenues_l572_572057


namespace students_passed_in_dixon_lecture_l572_572280

theorem students_passed_in_dixon_lecture :
  let ratio_collins := 18 / 30
  let students_dixon := 45
  ∃ y, ratio_collins = y / students_dixon ∧ y = 27 :=
by
  sorry

end students_passed_in_dixon_lecture_l572_572280


namespace tan_product_identity_l572_572110

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572110


namespace exists_point_P_l572_572430

variables (P Q1 Q2 A B : Type) [metric_space P] [metric_space Q1] [metric_space Q2]

def on_circle (C : Type) [metric_space C] (A : Point P) (Q : Point C) : Prop :=
∃ r, dist A Q = r

def moving_with_same_speed (A : Point P) (Q1 : Point Q1) (Q2 : Point Q2) : Prop :=
∀ t : ℝ, angle A Q1 = t ∧ angle A Q2 = t

theorem exists_point_P :
  ∀ (P Q1 Q2 A B : Type) [metric_space P] [metric_space Q1] [metric_space Q2],
  (on_circle P A Q1) ∧ (on_circle P A Q2) ∧ (moving_with_same_speed A Q1 Q2) →
  ∃ (P : Point), (dist P Q1 = dist P Q2) :=
sorry

end exists_point_P_l572_572430


namespace find_xyz_l572_572771

open Complex

theorem find_xyz (a b c x y z : ℂ)
(h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0)
(h7 : a = (b + c) / (x - 3)) (h8 : b = (a + c) / (y - 3)) (h9 : c = (a + b) / (z - 3))
(h10 : x * y + x * z + y * z = 10) (h11 : x + y + z = 6) : 
(x * y * z = 15) :=
by
  sorry

end find_xyz_l572_572771


namespace tan_product_equals_three_l572_572165

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572165


namespace circumcircleEquation_incircleEquation_l572_572022

noncomputable def verticesCircumcircle : Prop :=
  let A := (5, 1)
  let B := (7, -3)
  let C := (2, -8)
  ∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ 
           (5 - a)^2 + (1 - b)^2 = r^2 ∧
           (7 - a)^2 + (-3 - b)^2 = r^2 ∧
           (2 - a)^2 + (-8 - b)^2 = r^2 ∧
           (x - 2)^2 + (y + 3)^2 = 25

theorem circumcircleEquation : verticesCircumcircle := 
sorry

noncomputable def verticesIncircle : Prop :=
  let A := (0, 0)
  let B := (5, 0)
  let C := (0, 12)
  ∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ 
           (0 - a)^2 + (0 - b)^2 = r^2 ∧
           (5 - a)^2 + (0 - b)^2 = r^2 ∧
           (0 - a)^2 + (12 - b)^2 = r^2 ∧
           (x - 2)^2 + (y - 2)^2 = 4

theorem incircleEquation : verticesIncircle := 
sorry

end circumcircleEquation_incircleEquation_l572_572022


namespace range_of_a_condition_l572_572268

noncomputable def range_of_a (a : ℝ) : Set ℝ :=
  {x | x ^ 2 ≤ a}

theorem range_of_a_condition (a : ℝ) (h : ∅ ⊂ range_of_a a) : a ∈ set.Ici (0 : ℝ) :=
by
  -- proof steps
  sorry

end range_of_a_condition_l572_572268


namespace cadence_total_earnings_l572_572977

/-- Cadence's total earnings in both companies. -/
def total_earnings (old_salary_per_month new_salary_per_month : ℕ) (old_company_months new_company_months : ℕ) : ℕ :=
  (old_salary_per_month * old_company_months) + (new_salary_per_month * new_company_months)

theorem cadence_total_earnings :
  let old_salary_per_month := 5000
  let old_company_years := 3
  let months_per_year := 12
  let old_company_months := old_company_years * months_per_year
  let new_salary_per_month := old_salary_per_month + (old_salary_per_month * 20 / 100)
  let new_company_extra_months := 5
  let new_company_months := old_company_months + new_company_extra_months
  total_earnings old_salary_per_month new_salary_per_month old_company_months new_company_months = 426000 := by
sorry

end cadence_total_earnings_l572_572977


namespace num_divisors_not_divisible_by_three_l572_572667

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572667


namespace no_solutions_geometric_sequence_sin_l572_572566

-- Define the conditions of the problem
def is_geometric_sequence (a : ℝ) : Prop :=
  (0 < a) ∧ (a < 360) ∧ (sin (2 * a) / sin a = sin (3 * a) / sin (2 * a))

-- State the theorem
theorem no_solutions_geometric_sequence_sin : ¬ ∃ a : ℝ, is_geometric_sequence a :=
by {
  assume h : ∃ a : ℝ, is_geometric_sequence a,
  let ⟨a, ha⟩ := h,
  have h1 : 0 < a := ha.left,
  have h2 : a < 360 := ha.right.left,
  have h3 : sin (2 * a) / sin a = sin (3 * a) / sin (2 * a) := ha.right.right,
  -- Usual proof steps follow (simplifying, contradiction, etc.) which we skip here
  sorry
}

end no_solutions_geometric_sequence_sin_l572_572566


namespace smallest_bisecting_segment_l572_572887

-- Define a structure for a triangle in a plane
structure Triangle (α β γ : Type u) :=
(vertex1 : α) 
(vertex2 : β) 
(vertex3 : γ) 
(area : ℝ)

-- Define a predicate for an excellent line
def is_excellent_line {α β γ : Type u} (T : Triangle α β γ) (A : α) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line excellent here, e.g., dividing area in half
sorry

-- Define a function to get the length of a line segment within the triangle
def length_within_triangle {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : ℝ :=
-- compute the length of the segment within the triangle
sorry

-- Define predicates for triangles with specific properties like medians
def is_median {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line a median
sorry

theorem smallest_bisecting_segment {α β γ : Type u} (T : Triangle α β γ) (A : α) (median : ℝ → ℝ → ℝ) : 
  (∀ line, is_excellent_line T A line → length_within_triangle T line ≥ length_within_triangle T median) →
  median = line  := 
-- show that the median from the vertex opposite the smallest angle has the smallest segment
sorry

end smallest_bisecting_segment_l572_572887


namespace probability_one_red_ball_l572_572901

theorem probability_one_red_ball (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) 
                                (draws : ℕ) (favorable_outcomes : ℕ) (total_outcomes : ℕ) :
  total_balls = 10 → red_balls = 6 → black_balls = 4 → draws = 2 →
  total_outcomes = Nat.choose total_balls draws →
  favorable_outcomes = red_balls * black_balls →
  (favorable_outcomes : ℚ) / total_outcomes = 8 / 15 :=
begin
  -- Convert integers to natural numbers for calculation
  intros h_totals h_redballs h_blackballs h_draws h_totaloutcomes h_favorableoutcomes,
  rw [h_totals, h_redballs, h_blackballs, h_draws, h_totaloutcomes, h_favorableoutcomes],
  norm_num,
  -- Proof goes here
  sorry
end

end probability_one_red_ball_l572_572901


namespace tan_identity_l572_572155

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572155


namespace a_n_formula_S_n_formula_l572_572354

-- Define the sequences and conditions
def a_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a 1 + (finset.range (n + 1)).sum (λ k, 3^k * a (k + 1)) = sorry

def b_seq (b : ℕ → ℝ) : Prop := ∀ n : ℕ, b n = n * 3^n

-- Define the general formula for a_n
theorem a_n_formula (a : ℕ → ℝ) (h : a_seq a) : ∀ n : ℕ, a n = 3 / 2 * 3^(n - 1) := sorry

-- Define the sum of the first n terms of b_seq
theorem S_n_formula (b : ℕ → ℝ) (S : ℕ → ℝ) (h : b_seq b) : ∀ n : ℕ, 
  S n = (finset.range (n + 1)).sum (λ k, b k) :=
  sorry -- Prove that S n = (3/2) * (3^n * (n - 1) + 1)

end a_n_formula_S_n_formula_l572_572354


namespace f_solution_range_f_is_odd_f_monotonicity_intervals_g4_minus_5_f2_g2_g9_minus_5_f3_g3_general_relation_l572_572896

noncomputable def f (x : ℝ) : ℝ := (x^3 + x^(-3)) / 5
noncomputable def g (x : ℝ) : ℝ := (x^3 - x^(-3)) / 5

theorem f_solution_range (m : ℝ) : m ∈ [-2/5, 2/5] → ∃ x : ℝ, f x = m := sorry

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f(x) := sorry

theorem f_monotonicity_intervals : 
  (∀ x : ℝ, (x ≤ -1 ∨ x ≥ 1) → f' x ≥ 0) ∧ 
  (∀ x : ℝ, (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) → f' x ≤ 0) := sorry

theorem g4_minus_5_f2_g2 : g 4 - 5 * f 2 * g 2 = 0 := sorry

theorem g9_minus_5_f3_g3 : g 9 - 5 * f 3 * g 3 = 0 := sorry

theorem general_relation (x : ℝ) (hx : x ≠ 0) : g (x^2) - 5 * f x * g x = 0 := sorry

end f_solution_range_f_is_odd_f_monotonicity_intervals_g4_minus_5_f2_g2_g9_minus_5_f3_g3_general_relation_l572_572896


namespace mike_drive_average_rate_l572_572893

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

end mike_drive_average_rate_l572_572893


namespace rotated_rectangle_shaded_area_l572_572385

-- Defining the necessary entities for the problem
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables {AB AD AC CD: ℝ}
variables {r : ℝ}

-- Stating the proof problem
theorem rotated_rectangle_shaded_area :
  ∃ r, r = (1 / 2) * AB ∧ π * r^2 = (1 / 4) * π * AB^2 :=
begin
  sorry
end

end rotated_rectangle_shaded_area_l572_572385


namespace zuminglish_10_letter_word_mod_l572_572281

def is_zuminglish (word : List Char) : Prop := 
  -- Define the property of a Zuminglish word
  sorry

noncomputable def a (n : ℕ) : ℕ :=
  if n = 2 then 4
  else 2 * (a (n - 1) + c (n - 1))

noncomputable def b (n : ℕ) : ℕ :=
  if n = 2 then 2
  else a (n - 1)

noncomputable def c (n : ℕ) : ℕ :=
  if n = 2 then 2
  else 2 * b (n - 1)

noncomputable def N : ℕ :=
  a 10 + b 10 + c 10

def N_mod_1000 : ℕ := N % 1000

theorem zuminglish_10_letter_word_mod :
  N_mod_1000 = 936 := 
by sorry

end zuminglish_10_letter_word_mod_l572_572281


namespace range_of_a_l572_572636

open Real

theorem range_of_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a - b + c = 3) (h₃ : a + b + c = 1) (h₄ : 0 < c ∧ c < 1) : 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l572_572636


namespace technology_courses_correct_l572_572291

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end technology_courses_correct_l572_572291


namespace cube_and_square_root_eq_l572_572885

theorem cube_and_square_root_eq (x : ℝ) : x^3 = x ∧ sqrt x = x → x = 0 ∨ x = 1 := by
  sorry

end cube_and_square_root_eq_l572_572885


namespace arthur_total_dinner_cost_l572_572854

-- Definitions
def cost_of_appetizer := 8
def cost_of_ribeye := 20
def cost_of_wine_glass := 3
def number_of_wine_glasses := 2
def cost_of_cheesecake := 6
def discount_voucher_percent := 0.5
def tip_percent := 0.2

-- Calculations
def total_cost_before_discount := 
  cost_of_appetizer + cost_of_ribeye + (cost_of_wine_glass * number_of_wine_glasses) + cost_of_cheesecake

def discount_amount := cost_of_ribeye * discount_voucher_percent
def tip_amount := total_cost_before_discount * tip_percent

def total_cost_after_discount_and_tip := 
  total_cost_before_discount - discount_amount + tip_amount

-- Statement to prove
theorem arthur_total_dinner_cost :
  total_cost_after_discount_and_tip = 38 := 
by
  sorry

end arthur_total_dinner_cost_l572_572854


namespace range_of_t_l572_572266

theorem range_of_t (a b c t: ℝ) 
  (h1 : 6 * a = 2 * b - 6)
  (h2 : 6 * a = 3 * c)
  (h3 : b ≥ 0)
  (h4 : c ≤ 2)
  (h5 : t = 2 * a + b - c) : 
  0 ≤ t ∧ t ≤ 6 :=
sorry

end range_of_t_l572_572266


namespace tan_identity_l572_572154

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572154


namespace muffin_banana_ratio_l572_572822

variables {m b : ℝ}

-- Conditions
def susie_cost : ℝ := 4 * m + 3 * b
def calvin_cost_twice_susie : ℝ := 2 * susie_cost
def calvin_cost_direct : ℝ := 2 * m + 16 * b

-- Statement to prove
theorem muffin_banana_ratio (h : calvin_cost_twice_susie = calvin_cost_direct) : m = 5 / 3 * b :=
by
  -- explicitly using the equality derived from conditions
  sorry

end muffin_banana_ratio_l572_572822


namespace inequality_proof_l572_572614

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) +
  (y^5 - y^2) / (y^5 + z^2 + x^2) +
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := 
sorry

end inequality_proof_l572_572614


namespace exists_m_l572_572413

def special_set (n : ℕ) : Prop :=
  ∀ k : ℕ, n + k * (n + 1) ∈ M

theorem exists_m :
  let M := {n | 1 ≤ n ∧ n ≤ 2007}
  (∀ n ∈ M, special_set n) →
  ∃ m, ∀ k, k > m → k ∈ M :=
begin
  let M := {n | 1 ≤ n ∧ n ≤ 2007},
  intro h,
  use 2007,
  intro k,
  intro hk,
  have : k ∈ M, {
    sorry
  },
  exact this,
end

end exists_m_l572_572413


namespace simplify_complex_fraction_l572_572391

-- Define the complex numbers involved
def a : ℂ := 5 - 7 * complex.I
def b : ℂ := 2 - 3 * complex.I
def result : ℂ := (31 / 13) + (1 / 13) * complex.I

-- State the theorem
theorem simplify_complex_fraction : (a / b) = result :=
by sorry

end simplify_complex_fraction_l572_572391


namespace bridge_length_l572_572048

theorem bridge_length (T_train L_train : ℝ) (T_lp T_b : ℝ) (speed_train : T_train = L_train / T_lp) (speed_bridge : T_train = (L_train + L_bridge) / T_b) : L_bridge = 150 :=
by
    let speed := L_train / T_lp
    have : speed = (L_train + L_bridge) / T_b := by rw [←speed_train, ←speed_bridge]
    have : (L_train + L_bridge) = speed * T_b := by rw ←this
    have : 75 + L_bridge = speed * 7.5 := this
    have : 75 + L_bridge = 75 * 3 := by
        have speed_calc : speed = 30 := by
            have speed := L_train / T_lp
            rw [show L_train = 75 by rfl, show T_lp = 2.5 by rfl] at speed
            show speed = 30 from by
                calc
                    75 / 2.5 = 30 from by
                sorry
            sorry
    have L_bridge = 150 :=
        by
            calc
                L_bridge = 225 - 75 from by sorry
                = 150 from by sorry
    show L_bridge = 150 from by sorry

end bridge_length_l572_572048


namespace calculate_expression_l572_572522

theorem calculate_expression : (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := 
by 
  sorry

end calculate_expression_l572_572522


namespace system_unique_solution_l572_572197

theorem system_unique_solution 
  (x y z : ℝ) 
  (h1 : x + y + z = 3 * x * y) 
  (h2 : x^2 + y^2 + z^2 = 3 * x * z) 
  (h3 : x^3 + y^3 + z^3 = 3 * y * z) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) 
  (hz : 0 ≤ z) : 
  (x = 1 ∧ y = 1 ∧ z = 1) := 
sorry

end system_unique_solution_l572_572197


namespace interval_of_decrease_l572_572834

noncomputable def f (x : ℝ) : ℝ := real.logb (1/2) (-x^2 + 5 * x - 6)

theorem interval_of_decrease :
  ∀ x : ℝ, 2 < x ∧ x < 5 / 2 → ∃ y : ℝ, f(y) < f(x) ∧ y > x :=
sorry

end interval_of_decrease_l572_572834


namespace angle_C_measure_l572_572295

-- Given definitions based on the conditions
variable {ABCD : Type} [parallelogram ABCD]
variable (angle : ABCD → ℝ)
variable (angle_B angle_A : ℝ)
variable (C : ℝ)

-- Provide the conditions as hypotheses
hypothesis h1 : ∀ (p q : ABCD), angle_B - angle_A = 40
hypothesis h2 : ∀ (p q : ABCD), angle p + angle (opposite p) = 180

-- Statement of the final proof problem
theorem angle_C_measure : (∀ p : ABCD, angle p = 70) → ∃ q : ABCD, angle q = 70 :=
begin
  intro h,
  use some_p_in_ABCD,
  specialize h some_p_in_ABCD,
  assumption,
end

end angle_C_measure_l572_572295


namespace find_theta_l572_572847

noncomputable def area_of_circle (r : ℝ) : ℝ := π * r * r

def shaded_area_relation (S U : ℝ) : Prop := S = (9 / 17) * U

def total_area (radius1 radius2 radius3 : ℝ) : ℝ := area_of_circle radius1 + area_of_circle radius2 + area_of_circle radius3

theorem find_theta (r1 r2 r3 : ℝ)
    (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
    (h_area_rel : ∃ S U, shaded_area_relation S U ∧ S + U = total_area r1 r2 r3) :
    ∃ θ : ℝ, θ = (36 * π) / 187 :=
by
  have S U : ℝ := sorry
  use (36 * π) / 187
  sorry

end find_theta_l572_572847


namespace erin_tv_hours_l572_572560

def total_minutes_pp : Nat := 6 * 50
def total_minutes_bb : Nat := 62 * 47
def total_minutes_st : Nat := 33 * 51

def total_minutes : Nat := total_minutes_pp + total_minutes_bb + total_minutes_st

def total_hours : Real := total_minutes / 60

theorem erin_tv_hours : Real.floor total_hours = 82 := by 
  sorry

end erin_tv_hours_l572_572560


namespace rubies_correct_l572_572049

theorem rubies_correct (Total_gems : ℕ) (Diamonds : ℕ) (h1: Total_gems = 5155) (h2: Diamonds = 45) : 
  Total_gems - Diamonds = 5110 :=
by
  rw [h1, h2]
  sorry

end rubies_correct_l572_572049


namespace rectangular_solid_surface_area_l572_572557

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end rectangular_solid_surface_area_l572_572557


namespace right_triangle_perimeter_l572_572934

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572934


namespace total_trees_planted_l572_572963

/-- A yard is 255 meters long, with a tree at each end and trees planted at intervals of 15 meters. -/
def yard_length : ℤ := 255

def tree_interval : ℤ := 15

def total_trees : ℤ := 18

theorem total_trees_planted (L : ℤ) (d : ℤ) (n : ℤ) : 
  L = yard_length →
  d = tree_interval →
  n = total_trees →
  n = (L / d) + 1 :=
by
  intros hL hd hn
  rw [hL, hd, hn]
  sorry

end total_trees_planted_l572_572963


namespace find_numbers_l572_572865

theorem find_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
                     (hxy_mul : 2000 ≤ x * y ∧ x * y < 3000) (hxy_add : 100 ≤ x + y ∧ x + y < 1000)
                     (h_digit_relation : x * y = 2000 + x + y) : 
                     (x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30) :=
by
  -- The proof will go here
  sorry

end find_numbers_l572_572865


namespace mascots_arrangement_count_l572_572837

-- Define the entities
def bing_dung_dung_mascots := 4
def xue_rong_rong_mascots := 3

-- Define the conditions
def xue_rong_rong_a_and_b_adjacent := true
def xue_rong_rong_c_not_adjacent_to_ab := true

-- Theorem stating the problem and asserting the answer
theorem mascots_arrangement_count : 
  (xue_rong_rong_a_and_b_adjacent ∧ xue_rong_rong_c_not_adjacent_to_ab) →
  (number_of_arrangements = 960) := by
  sorry

end mascots_arrangement_count_l572_572837


namespace arrangement_of_70616_l572_572290

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end arrangement_of_70616_l572_572290


namespace no_distribution_sums_to_2001_l572_572403

theorem no_distribution_sums_to_2001 :
  ∀ table : fin 3 → fin 3 → ℕ, 
  (∀ i j, 1 ≤ table i j ∧ table i j ≤ 9) →
  (∀ k l, k ≠ l → table k l ≠ table l k) → -- ensuring distinct numbers in the rows and columns
  (∑ i in finRange 3, ∑ j in finRange 3, table i j) = 45 →
  (∑ i in finRange 3, read_as_number (table i 0) (table i 1) (table i 2) + read_as_number (table 0 i) (table 1 i) (table 2 i)) ≠ 2001 := sorry

noncomputable def read_as_number (a b c : ℕ) : ℕ := 
  100 * a + 10 * b + c

#check no_distribution_sums_to_2001

end no_distribution_sums_to_2001_l572_572403


namespace jenna_peeled_potatoes_l572_572646

-- Definitions of constants
def initial_potatoes : ℕ := 60
def homer_rate : ℕ := 4
def jenna_rate : ℕ := 6
def combined_rate : ℕ := homer_rate + jenna_rate
def homer_time : ℕ := 6
def remaining_potatoes : ℕ := initial_potatoes - (homer_rate * homer_time)
def combined_time : ℕ := 4 -- Rounded from 3.6

-- Statement to prove
theorem jenna_peeled_potatoes : remaining_potatoes / combined_rate * jenna_rate = 24 :=
by
  sorry

end jenna_peeled_potatoes_l572_572646


namespace tan_product_eq_three_l572_572076

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572076


namespace correct_interpretation_of_confidence_level_l572_572735

theorem correct_interpretation_of_confidence_level :
  ∀ (smoke lung : Type) (K2 : ℝ), 
  (K2 = 6.635) → 
  (99% confidence that smoke is related to lung) →
  (95% confidence that smoke is related to lung) →
  (correct statement : ℕ → string) →
  correct statement = 3 →
  (∀ (A B : ℕ → string), 
  A 1 = "If the observed value of K^2 is k=6.635, we are 99% confident that smoking is related to lung disease, which means out of 100 smoking individuals, there must be 99 individuals with lung disease." →
  B 2 = "From the independence test, knowing that we are 99% confident that smoking is related to lung disease, we would say that if a person smokes, then there is a 99% chance that they have lung disease.") →
  (correct_statement : ℕ → string) 3 = 
  "If the statistical measure tells us that we are 95% confident that smoking is related to lung disease, it means there is a 5% probability that our inference is incorrect." :=
sorry

end correct_interpretation_of_confidence_level_l572_572735


namespace tan_product_l572_572138

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572138


namespace doughnut_completion_l572_572465

theorem doughnut_completion :
  let start_time := 8 * 60 + 30 in -- 8:30 AM in minutes
  let one_third_time := 11 * 60 + 10 - start_time in -- Duration from 8:30 AM to 11:10 AM in minutes
  let total_time := 3 * one_third_time in -- Total time to finish the job
  let completion_time := start_time + total_time in -- Completion time in minutes
  completion_time = 16 * 60 + 30 := -- 4:30 PM in minutes
by
  sorry

end doughnut_completion_l572_572465


namespace geometric_sequence_solution_l572_572184

noncomputable def frac_part (x : ℝ) : ℝ := x - floor x

theorem geometric_sequence_solution :
  ∃ x : ℝ, x ≠ 0 ∧ frac_part x * floor x + frac_part x ^ 2 = floor x ^ 2 ∧ (frac_part x = (sqrt 5 - 1) / 2 ∧ floor x = 1) ∧ x = sqrt 5 / 2 :=
by
  sorry

end geometric_sequence_solution_l572_572184


namespace proof_ellipse_l572_572222

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
 ∀ x y : ℝ, (y^2 / a^2 + x^2 / b^2 = 1)

noncomputable def ellipse_tangent_to_circle (a : ℝ) : Prop :=
 ∀ x y : ℝ, (x^2 + y^2 = 4) → (x^2 / 2 + y^2 / 4 = 1)

noncomputable def hyperbola_eccentricity : ℝ := 
 ∀ e_h : ℝ, e_h = √2

noncomputable def ellipse_eccentricity (c a : ℝ): ℝ :=
 ∀ e_e : ℝ, e_e = c / a

def solve_ellipse_m (a b c : ℝ) : Prop :=
 ellipse_equation a b ∧ hyperbola_eccentricity (√2) ∧ 
 ellipse_eccentricity (√2 / 2) ∧ 
 ellipse_tangent_to_circle 2

theorem proof_ellipse : 
 solve_ellipse_m 2 (√2) (√2) := 
-- Need Proof Here
sorry

end proof_ellipse_l572_572222


namespace find_a_l572_572631

namespace MyProof

def f (x : ℝ) (a : ℝ) : ℝ := Real.log (1 - a / 2^x)

theorem find_a (a : ℝ) : (∀ x : ℝ, x ∈ Set.Ioo 4 +∞ → f x a = Real.log (1 - a / 2^x)) ∧ (1 - a / 2^4) > 0 → a = 16 :=
by
   sorry

end MyProof

end find_a_l572_572631


namespace parabola_slope_l572_572605

/--
Let C be a parabola defined by y^2 = 4x.
Let M be the point (-1, 0), and let l be a line passing through M with slope k.
Line l intersects the parabola C at points A and B, with A between M and B.
If |AM| = 5/4 |AF|, then k = ± 3/4.
-/
theorem parabola_slope
  (M : ℝ × ℝ)
  (k : ℝ)
  (hM : M = (-1, 0))
  (y : ℝ)
  (A B : ℝ × ℝ)
  (hLine : ∀ (x : ℝ), y = k * (x + 1))
  (hParabola : ∀ (x y : ℝ), y^2 = 4 * x)
  (hIntersects : ∃ A B : ℝ × ℝ, y = k * (fst A + 1) ∧ y^2 = 4 * fst A ∧ y = k * (fst B + 1) ∧ y^2 = 4 * fst B)
  (hOrder : fst M < fst A ∧ fst A < fst B)
  (hDist : dist M A = 5 / 4 * dist A (1, 0)) :
  k = 3 / 4 ∨ k = - 3 / 4 := 
sorry

end parabola_slope_l572_572605


namespace not_collinear_l572_572504

-- Definitions from the problem conditions
def a : Vector ℝ 3 := ⟨[2, 0, -5]⟩
def b : Vector ℝ 3 := ⟨[1, -3, 4]⟩
def c1 : Vector ℝ 3 := 2 • a - 5 • b
def c2 : Vector ℝ 3 := 5 • a - 2 • b

-- Lean statement to prove the non-collinearity
theorem not_collinear : ¬ ∃ γ : ℝ, c1 = γ • c2 := 
by {
    sorry
}

end not_collinear_l572_572504


namespace find_cube_side_length_l572_572842

noncomputable def cube_side_length_eq_sphere : Prop :=
  let radius := 4
  let sphere_surface_area := 4 * Real.pi * (radius ^ 2)
  ∃ (s : ℝ), 6 * (s ^ 2) = sphere_surface_area

theorem find_cube_side_length : cube_side_length_eq_sphere :=
by
  let radius := 4
  let sphere_surface_area := 4 * Real.pi * (radius ^ 2)
  have h : ∃ (s : ℝ), 6 * (s ^ 2) = sphere_surface_area,
  from
    have : 6 * (real.sqrt (32 * Real.pi / 3)) ^ 2 = sphere_surface_area,
    ...
  exact h

end find_cube_side_length_l572_572842


namespace trigonometric_identity_example_l572_572987

theorem trigonometric_identity_example :
  sin (43 * real.pi / 180) * cos (13 * real.pi / 180) - cos (43 * real.pi / 180) * sin (13 * real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_example_l572_572987


namespace Mike_miles_l572_572788

open Real

def Mike_initial_charge := 2.50
def Mike_cost_per_mile := 0.25
def Mike_surcharge := 3.00

def Annie_initial_charge := 2.50
def Annie_cost_per_mile := 0.25
def Annie_miles := 22
def Annie_toll_fees := 2.00 + 1.50 + 1.50

theorem Mike_miles (M : ℝ) : 
  Mike_initial_charge + Mike_cost_per_mile * M + Mike_surcharge = Annie_initial_charge + Annie_cost_per_mile * Annie_miles + Annie_toll_fees → 
  M = 30 := by
  sorry

end Mike_miles_l572_572788


namespace correct_average_l572_572869

-- let's define the numbers as a list
def numbers : List ℕ := [1200, 1300, 1510, 1520, 1530, 1200]

-- the condition given in the problem: the stated average is 1380
def stated_average : ℕ := 1380

-- given the correct calculation of average, let's write the theorem statement
theorem correct_average : (numbers.foldr (· + ·) 0) / numbers.length = 1460 :=
by
  -- we would prove it here
  sorry

end correct_average_l572_572869


namespace compare_neg_fractions_l572_572985

theorem compare_neg_fractions : (- (3 / 2) < -1) :=
by sorry

end compare_neg_fractions_l572_572985


namespace students_no_A_in_any_subject_l572_572729

def total_students : ℕ := 50
def a_in_history : ℕ := 9
def a_in_math : ℕ := 15
def a_in_science : ℕ := 12
def a_in_math_and_history : ℕ := 5
def a_in_history_and_science : ℕ := 3
def a_in_science_and_math : ℕ := 4
def a_in_all_three : ℕ := 1

theorem students_no_A_in_any_subject : 
  (total_students - (a_in_history + a_in_math + a_in_science 
                      - a_in_math_and_history - a_in_history_and_science - a_in_science_and_math 
                      + a_in_all_three)) = 28 := by
  sorry

end students_no_A_in_any_subject_l572_572729


namespace regular_icosahedron_edges_l572_572650

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end regular_icosahedron_edges_l572_572650


namespace triangle_perimeter_l572_572952

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572952


namespace part1_part2_l572_572590

variable (α : ℝ)
variable (h1 : sin α + cos α = -1/5)

-- Part 1
theorem part1 : sin (π/2 + α) * cos (π/2 - α) = -12/25 := by
  sorry

-- Additional hypothesis for part 2
variable (h2 : π/2 < α)
variable (h3 : α < π)

-- Part 2
theorem part2 : 1/sin (π - α) + 1/cos (π - α) = 35/12 := by
  sorry

end part1_part2_l572_572590


namespace isolated_elements_count_l572_572229

open Finset

def is_isolated (A : Finset ℕ) (a : ℕ) : Prop :=
  a ∈ A ∧ a - 1 ∉ A ∧ a + 1 ∉ A

def no_isolated_subset (A : Finset ℕ) : Prop :=
  ∀ (a ∈ A), ¬ is_isolated A a

theorem isolated_elements_count :
  let M := (finset.range 9).image (λ (k : ℕ), k + 1)
  M.card = 9 →
  (finset.filter (λ (A : finset ℕ), A.card = 4 ∧ no_isolated_subset A)
    (finset.powerset_len 4 M)).card = 21 :=
by simp; sorry

end isolated_elements_count_l572_572229


namespace chords_have_common_point_l572_572600

noncomputable def fixed_point_intersection (S : Type) [metric_space S] (O : S)
  (r : ℝ) (h : 0 < r)
  (P : ℝ → S)
  (l : set S)
  (l_linear : ∀ t, P t ∈ l)
  (l_nonintersect : metric.ball O r ∩ l = ∅)
  (A B : S)
  (PA PB : ∀ t, tangent_line S (P t) A ∧ tangent_line S (P t) B)
  (tangent_line : S → S → Prop)
  (chord_AB : ∀ t, S → S → S → S) : Prop :=
∃ X : S, ∀ t, chord_AB (P t) A B S = X

theorem chords_have_common_point (S : Type) [metric_space S] (O : S)
  (r : ℝ) (h : 0 < r)
  (P : ℝ → S)
  (l : set S)
  (l_linear : ∀ t, P t ∈ l)
  (l_nonintersect : metric.ball O r ∩ l = ∅)
  (tangent_line : S → S → Prop)
  (A B : S)
  (PA PB : ∀ t, tangent_line (P t) A ∧ tangent_line (P t) B)
  (chord_AB : ∀ t, S → S → S → S) :
  fixed_point_intersection S O r h P l l_linear l_nonintersect A B PA PB tangent_line chord_AB := sorry

end chords_have_common_point_l572_572600


namespace total_customers_in_line_l572_572542

-- Definition of the number of people standing in front of the last person
def num_people_in_front : Nat := 8

-- Definition of the last person in the line
def last_person : Nat := 1

-- Statement to prove
theorem total_customers_in_line : num_people_in_front + last_person = 9 := by
  sorry

end total_customers_in_line_l572_572542


namespace technology_elective_courses_l572_572293

theorem technology_elective_courses (m : ℕ) :
  let subject_elective := m,
      arts_elective := m + 9,
      technology_elective := 1 / 3 * arts_elective + 5
  in technology_elective = 1 / 3 * m + 8 :=
by
  sorry

end technology_elective_courses_l572_572293


namespace evaluate_product_logs_l572_572182

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

noncomputable def product_logs (n : ℕ) : ℝ :=
  (List.range (n + 1)).map (fun i => log_base (4 + i) (5 + i)).foldl (*) 1

theorem evaluate_product_logs :
  product_logs 35 = (3 / 2) + (Real.log 5 / (2 * Real.log 2)) := sorry

end evaluate_product_logs_l572_572182


namespace A_beats_B_l572_572727

def distance_covered (speed time : ℝ) : ℝ := speed * time

def speed (distance time : ℝ) : ℝ := distance / time

theorem A_beats_B :
  let vB := speed 200 60 in
  let distanceB := distance_covered vB 45 in
  200 - distanceB = 50 :=
by
  let vB := speed 200 60
  let distanceB := distance_covered vB 45
  have : 200 - distanceB = 50
  from sorry
  exact this

end A_beats_B_l572_572727


namespace binary_rep_253_l572_572994

def decimal_to_binary (n : ℕ) : list ℕ := 
  if n = 0 then [] else decimal_to_binary (n / 2) ++ [n % 2]

def count_zeros (binary : list ℕ) : ℕ := binary.count 0
def count_ones (binary : list ℕ) : ℕ := binary.count 1

theorem binary_rep_253 : 
  let n := 253
  let binary_rep := decimal_to_binary n
  let x := count_zeros binary_rep
  let y := count_ones binary_rep
  in y - x = 6 :=
by
  sorry

end binary_rep_253_l572_572994


namespace divisors_not_divisible_by_3_eq_6_l572_572657

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572657


namespace ellipse_foci_distance_l572_572205

noncomputable def distance_between_foci (a b : ℝ) : ℝ :=
let c := real.sqrt (a^2 - b^2) in
2 * c

theorem ellipse_foci_distance :
  distance_between_foci 7 3 = 4 * real.sqrt 10 :=
by
  sorry

end ellipse_foci_distance_l572_572205


namespace sum_smallest_largest_prime_between_1_and_50_l572_572508

def primes_between (a b : ℕ) : List ℕ :=
  List.filter Nat.prime (List.range' a (b + 1 - a))

theorem sum_smallest_largest_prime_between_1_and_50 :
  let primes := primes_between 1 50
  List.minimum primes = some 2 →
  List.maximum primes = some 47 →
  2 + 47 = 49 := by
  assume (h_min : List.minimum primes = some 2) (h_max : List.maximum primes = some 47)
  sorry

end sum_smallest_largest_prime_between_1_and_50_l572_572508


namespace geometric_sequence_common_ratio_l572_572639

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, 0 < a n)
  (h3 : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2) : q = 3 := by
  sorry

end geometric_sequence_common_ratio_l572_572639


namespace mario_pizza_area_l572_572360

theorem mario_pizza_area
  (pizza_area : ℝ)
  (cut_distance : ℝ)
  (largest_piece : ℝ)
  (smallest_piece : ℝ)
  (total_pieces : ℕ)
  (pieces_mario_gets_area : ℝ) :
  pizza_area = 4 →
  cut_distance = 0.5 →
  total_pieces = 4 →
  pieces_mario_gets_area = (pizza_area - (largest_piece + smallest_piece)) / 2 →
  pieces_mario_gets_area = 1.5 :=
sorry

end mario_pizza_area_l572_572360


namespace systematic_sampling_arithmetic_sequence_l572_572489

theorem systematic_sampling_arithmetic_sequence :
  ∃ x : ℕ, x ∈ ({6, 32, 45} : set ℕ) → {6, x, 32, 45}.to_list.sort (≤) = [6, x, 32, 45] ∧ (6 + 45 = x + 32) → x = 19 :=
by
  sorry

end systematic_sampling_arithmetic_sequence_l572_572489


namespace area_of_farm_l572_572041

theorem area_of_farm (W L : ℝ) (hW : W = 30) 
  (hL_fence_cost : 14 * (L + W + Real.sqrt (L^2 + W^2)) = 1680) : 
  W * L = 1200 :=
by
  sorry -- Proof not required

end area_of_farm_l572_572041


namespace cole_cost_l572_572983

def length_of_sides := 15
def length_of_back := 30
def cost_per_foot_side := 4
def cost_per_foot_back := 5
def cole_installation_fee := 50

def neighbor_behind_contribution := (length_of_back * cost_per_foot_back) / 2
def neighbor_left_contribution := (length_of_sides * cost_per_foot_side) / 3

def total_cost := 
  2 * length_of_sides * cost_per_foot_side + 
  length_of_back * cost_per_foot_back

def cole_contribution := 
  total_cost - neighbor_behind_contribution - neighbor_left_contribution + cole_installation_fee

theorem cole_cost (h : cole_contribution = 225) : cole_contribution = 225 := by
  sorry

end cole_cost_l572_572983


namespace final_solution_l572_572782

noncomputable def S : ℝ := ∑' i : ℕ, ∏ j in Finset.range (i + 1), (3 * (j + 1) - 2) / (12 * (j + 1))

theorem final_solution :
  let m := 4
  let n := 3
  ((S + 1) ^ 3 = (m / n) ∧ Nat.coprime m n) → 10 * m + n = 43 :=
by
  sorry

end final_solution_l572_572782


namespace tan_identity_l572_572156

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572156


namespace num_non_divisible_by_3_divisors_l572_572671

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572671


namespace decreasing_interval_l572_572177

noncomputable def f (x : ℝ) := Real.exp (abs (x - 1))

theorem decreasing_interval : ∀ x y : ℝ, x ≤ y → y ≤ 1 → f y ≤ f x :=
by
  sorry

end decreasing_interval_l572_572177


namespace find_a_l572_572270

-- Definitions for the conditions
variables (a b : ℝ)
variable pos_a : 0 < a
variable pos_b : 0 < b
variable pow_eq : a^b = b^a
variable b_eq_4a : b = 4 * a

-- The theorem to be proven
theorem find_a : a = real.cbrt 4 :=
by
  sorry

end find_a_l572_572270


namespace collinear_E_F_C_l572_572300

variable {A B C D E F : Point}
variable {triangle_ABC : Triangle A B C}

-- Definitions not provided in Lean, hypothetical for example purposes
variable (median_AD : isMedian triangle_ABC D)
variable (AD_half_BC : AD < (1 / 2) * B.distance C)
variable (DE_bisects_ADB : angleBisector D E (angle A D B))
variable (DF_bisects_ADC : angleBisector D F (angle A D C))
variable (EA_eq_EB : A.distance E = B.distance E)
variable (FA_eq_FC : A.distance F = C.distance F)
variable (angle_AEB_DCF_eq_90 : angle A E B + angle D C F = 90)

theorem collinear_E_F_C
  (median_AD)
  (AD_half_BC)
  (DE_bisects_ADB)
  (DF_bisects_ADC)
  (EA_eq_EB)
  (FA_eq_FC)
  (angle_AEB_DCF_eq_90) :
  collinear E F C := by
  sorry

end collinear_E_F_C_l572_572300


namespace angle_A_min_a_l572_572279

-- Problem 1: Proving the measure of angle A
theorem angle_A (A B C : ℝ) (h : cos (2 * A) - 3 * cos (B + C) - 1 = 0) :
  A = π / 3 := 
sorry

-- Problem 2: Finding the minimum value of a
theorem min_a (a b c : ℝ) (A : ℝ) (area : ℝ) (hA : A = π / 3)
  (harea : area = sqrt 3 / 2) (hbc : b * c = 2) :
  a ≥ sqrt 2 := 
sorry

end angle_A_min_a_l572_572279


namespace maximum_points_chess_match_l572_572860

theorem maximum_points_chess_match (games : ℕ) (odd_draw : ℕ → Prop) (consecutive_win : ℕ → Prop) :
    games = 24 ∧ 
    (∀ n, ((¬ odd_draw (2*n + 1)) ∧ (odd_draw (2*n + 2) → n < 4))) ∧ 
    (∀ m, (∀ n, consecutive_win n → ¬ consecutive_win (n + 1) → ¬ consecutive_win (n + 2))) → 
    ∃ p, p = 18.5 := 
by
  sorry

end maximum_points_chess_match_l572_572860


namespace find_smaller_number_l572_572431

theorem find_smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : u = 6 :=
by
  sorry

end find_smaller_number_l572_572431


namespace right_triangle_perimeter_l572_572947

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572947


namespace tangent_product_eq_three_l572_572114

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572114


namespace total_animals_count_l572_572518

def snakes := 100
def arctic_foxes := 80
def leopards := 20
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

theorem total_animals_count : snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 := by
  calc
    snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators
        = 100 + 80 + 20 + 200 + 50 + 200 : by sorry -- Use the defined values
    ... = 650 : by sorry -- Perform the final addition

end total_animals_count_l572_572518


namespace equal_segments_BD_CD_DP_l572_572538

-- Define the problem conditions
variables {O A B C P D : Type} [metric_space O]
variables (circ_outer : circle O)
variables (circ_inner : {P : Type} [inner_circle : circle P] inner_circle ⊆ circ_outer)
variables [triangle_abc : triangle ℝ O A B C] (h_A : A ∈ circ_outer) (h_B : B ∈ circ_outer) (h_C : C ∈ circ_outer)
variables [triangle_apd : triangle ℝ O A P D] (h_P : P ∈ circ_inner) (h_tangent1 : tangent P (line_of A B)) (h_tangent2 : tangent P (line_of A C))
variables [segment_ad : segment ℝ A D] (h_D : D ∈ circ_outer) (h_intersect1 : line_intersect A P circ_outer D)

-- Problem statement
theorem equal_segments_BD_CD_DP : segment_length ℝ (segment ℝ B D) = segment_length ℝ (segment ℝ C D) ∧ 
segment_length ℝ (segment ℝ D P) = segment_length ℝ (segment ℝ B D) :=
sorry

end equal_segments_BD_CD_DP_l572_572538


namespace stratified_sampling_third_year_l572_572042

-- The total number of students in the school
def total_students : ℕ := 2000

-- The probability of selecting a female student from the second year
def prob_female_second_year : ℚ := 0.19

-- The number of students to be selected through stratified sampling
def sample_size : ℕ := 100

-- The total number of third-year students
def third_year_students : ℕ := 500

-- The number of students to be selected from the third year in stratified sampling
def third_year_sample (total : ℕ) (third_year : ℕ) (sample : ℕ) : ℕ :=
  sample * third_year / total

-- Lean statement expressing the goal
theorem stratified_sampling_third_year :
  third_year_sample total_students third_year_students sample_size = 25 :=
by
  sorry

end stratified_sampling_third_year_l572_572042


namespace tan_product_eq_three_l572_572074

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572074


namespace tan_product_l572_572087

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572087


namespace problem_statement_l572_572212

noncomputable def a : ℝ := real.sqrt 3 - real.sqrt 2
noncomputable def b : ℝ := real.sqrt 3 + real.sqrt 2

theorem problem_statement : a^2 + 3 * a * b + b^2 - a + b = 13 + 2 * real.sqrt 2 := sorry

end problem_statement_l572_572212


namespace cylindrical_can_increase_l572_572906

theorem cylindrical_can_increase (R H y : ℝ)
  (h₁ : R = 5)
  (h₂ : H = 4)
  (h₃ : π * (R + y)^2 * (H + y) = π * (R + 2*y)^2 * H) :
  y = Real.sqrt 76 - 5 :=
by
  sorry

end cylindrical_can_increase_l572_572906


namespace infinite_series_evaluation_l572_572561

theorem infinite_series_evaluation :
  (∑' m : ℕ, ∑' n : ℕ, 1 / (m * n * (m + n + 2))) = 3 :=
  sorry

end infinite_series_evaluation_l572_572561


namespace sum_of_valid_y_values_l572_572581

theorem sum_of_valid_y_values :
  (∑ y in {y | y > 50 ∧ y < 150 ∧ (sin (2 * y * real.pi / 180))^3 + (cos (6 * y * real.pi / 180))^3 = 9 * (sin (y * real.pi / 180))^3 * (cos (4 * y * real.pi / 180))^3}, y) = 270 :=
sorry

end sum_of_valid_y_values_l572_572581


namespace sin_double_angle_l572_572211

theorem sin_double_angle (θ : ℝ) (h : sin (π / 4 + θ) = 1 / 3) : sin (2 * θ) = -7 / 9 := by
  sorry

end sin_double_angle_l572_572211


namespace trajectory_of_midpoint_l572_572601

-- Define the problem conditions
def X (t : ℝ) : ℝ × ℝ × ℝ :=
  if 0 ≤ t ∧ t ≤ 1 then (t, 0, 0)
  else if 1 < t ∧ t ≤ 2 then (1, t - 1, 0)
  else if 2 < t ∧ t ≤ 3 then (3 - t, 1, 0)
  else if 3 < t ∧ t ≤ 4 then (0, 4 - t, 0)
  else (0, 0, 0)

def Y (t : ℝ) : ℝ × ℝ × ℝ :=
  if 0 ≤ t ∧ t ≤ 1 then (1, t, 1)
  else if 1 < t ∧ t ≤ 2 then (1, 1, 2 - t)
  else if 2 < t ∧ t ≤ 3 then (1, 3 - t, 0)
  else if 3 < t ∧ t ≤ 4 then (1, 0, t - 3)
  else (0, 0, 0)

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the elapsed time variable
def t (t : ℝ) : ℝ := t

-- The proof statement
theorem trajectory_of_midpoint :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 4 → midpoint (X t) (Y t) =
    if t = 0 then (1 / 2, 0, 1 / 2)
    else if t = 1 then (1, 1 / 2, 1 / 2)
    else if t = 2 then (1, 1, 0)
    else if t = 3 then (1 / 2, 1 / 2, 0)
    else if t = 4 then (1 / 2, 0, 1 / 2)
    else midpoint (X t) (Y t) :=     sorry

end trajectory_of_midpoint_l572_572601


namespace number_of_integer_solutions_l572_572551

theorem number_of_integer_solutions :
  {x : ℤ // ((x^2 - 2 * x - 2) ^ (x + 3) = 1)}.subtype.fintype.card = 4 := 
sorry

end number_of_integer_solutions_l572_572551


namespace volume_calculation_l572_572779

noncomputable def enclosedVolume : Real :=
  let f (x y z : Real) : Real := x^2016 + y^2016 + z^2
  let V : Real := 360
  V

theorem volume_calculation : enclosedVolume = 360 :=
by
  sorry

end volume_calculation_l572_572779


namespace tadd_2019th_number_l572_572858

def next_start_point (n : ℕ) : ℕ := 
    1 + (n * (2 * 3 + (n - 1) * 9)) / 2

def block_size (n : ℕ) : ℕ := 
    1 + 3 * (n - 1)

def nth_number_said_by_tadd (n : ℕ) (k : ℕ) : ℕ :=
    let block_n := next_start_point n
    block_n + k - 1

theorem tadd_2019th_number :
    nth_number_said_by_tadd 37 2019 = 5979 := 
sorry

end tadd_2019th_number_l572_572858


namespace max_sum_of_arithmetic_sequence_l572_572349

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
(h1 : 3 * a 8 = 5 * a 13) 
(h2 : a 1 > 0)
(hS : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
S 20 > S 21 ∧ S 20 > S 10 ∧ S 20 > S 11 :=
sorry

end max_sum_of_arithmetic_sequence_l572_572349


namespace largest_set_size_l572_572578

theorem largest_set_size :
  ∃ (S : Set ℕ), (∀ a ∈ S, a ≤ 2002) ∧ 
                 (∀ a b ∈ S, a ≠ b → a^2 ∉ S ∧ a * b ∉ S) ∧ 
                 S.card = 1958 :=
by
  sorry

end largest_set_size_l572_572578


namespace exam_passing_probability_l572_572921

-- Definitions based on the conditions
def num_questions : ℕ := 4
def prob_correct : ℚ := 0.4

def prob_passing_exam : ℚ := (4.choose 3) * prob_correct^3 * (1 - prob_correct) + prob_correct^4

-- The main statement we need to prove
theorem exam_passing_probability : prob_passing_exam = 0.1792 := by
  sorry

end exam_passing_probability_l572_572921


namespace actual_distance_traveled_l572_572012

theorem actual_distance_traveled (D : ℝ) (h : D / 10 = (D + 20) / 20) : D = 20 :=
  sorry

end actual_distance_traveled_l572_572012


namespace probability_crisp_stops_on_dime_l572_572540

noncomputable def crisp_stops_on_dime_probability : ℚ :=
  let a := (2/3 : ℚ)
  let b := (1/3 : ℚ)
  let a1 := (15/31 : ℚ)
  let b1 := (30/31 : ℚ)
  (2 / 3) * a1 + (1 / 3) * b1

theorem probability_crisp_stops_on_dime :
  crisp_stops_on_dime_probability = 20 / 31 :=
by
  sorry

end probability_crisp_stops_on_dime_l572_572540


namespace cos_pi_div_4_plus_alpha_l572_572616

theorem cos_pi_div_4_plus_alpha (α : ℝ) (h : sin (π / 4 - α) = 1 / 5) : cos (π / 4 + α) = 1 / 5 :=
by
  sorry

end cos_pi_div_4_plus_alpha_l572_572616


namespace triangle_perimeter_l572_572957

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572957


namespace ben_paid_supplier_l572_572509

theorem ben_paid_supplier (X : ℝ) (initial_amount : ℝ := 2000) (debtor_payment : ℝ := 800)
                          (maintenance_cost : ℝ := 1200) (remaining_amount : ℝ := 1000) :
    [((initial_amount - X) + debtor_payment) - maintenance_cost = remaining_amount] →
    X = 600 :=
by
  intro h
  sorry

end ben_paid_supplier_l572_572509


namespace find_m_l572_572263

-- Definitions for the given vectors
def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (4, m)

-- The condition that (vector_a + 2 * vector_b) is parallel to (vector_a - vector_b)
def parallel_condition (m : ℝ) : Prop :=
  let left_vec := (vector_a.1 + 2 * 4, vector_a.2 + 2 * m)
  let right_vec := (vector_a.1 - 4, vector_a.2 - m)
  left_vec.1 * right_vec.2 - right_vec.1 * left_vec.2 = 0

-- The main theorem to prove
theorem find_m : ∃ m : ℝ, parallel_condition m ∧ m = -6 := 
sorry

end find_m_l572_572263


namespace tan_product_l572_572136

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572136


namespace model_tower_height_l572_572559

-- Definitions based on conditions
def height_actual_tower : ℝ := 60
def volume_actual_tower : ℝ := 80000
def volume_model_tower : ℝ := 0.5

-- Theorem statement
theorem model_tower_height (h: ℝ) : h = 0.15 :=
by
  sorry

end model_tower_height_l572_572559


namespace tan_identity_l572_572158

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572158


namespace arthur_total_dinner_cost_l572_572852

-- Definitions
def cost_of_appetizer := 8
def cost_of_ribeye := 20
def cost_of_wine_glass := 3
def number_of_wine_glasses := 2
def cost_of_cheesecake := 6
def discount_voucher_percent := 0.5
def tip_percent := 0.2

-- Calculations
def total_cost_before_discount := 
  cost_of_appetizer + cost_of_ribeye + (cost_of_wine_glass * number_of_wine_glasses) + cost_of_cheesecake

def discount_amount := cost_of_ribeye * discount_voucher_percent
def tip_amount := total_cost_before_discount * tip_percent

def total_cost_after_discount_and_tip := 
  total_cost_before_discount - discount_amount + tip_amount

-- Statement to prove
theorem arthur_total_dinner_cost :
  total_cost_after_discount_and_tip = 38 := 
by
  sorry

end arthur_total_dinner_cost_l572_572852


namespace closest_integer_to_sqrt_11_l572_572402

theorem closest_integer_to_sqrt_11 : 
  ∀ (x : ℝ), (3 : ℝ) ≤ x → x ≤ 3.5 → x = 3 :=
by
  intro x hx h3_5
  sorry

end closest_integer_to_sqrt_11_l572_572402


namespace find_sum_of_squares_l572_572792

theorem find_sum_of_squares :
    (17^2 + 19^2 + 23^2 + 29^2 = 2020) :=
begin
  sorry
end

end find_sum_of_squares_l572_572792


namespace math_expression_evaluation_l572_572979

theorem math_expression_evaluation :
  36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 :=
by
  sorry

end math_expression_evaluation_l572_572979


namespace proof1_proof2_l572_572021

variables {α : Real}

-- Condition for first proof
def cond1 : α -> Prop := λ α, cos α = -4/5 ∧ π < α ∧ α < 3*π/2

-- Proof for first condition
theorem proof1 (h : cond1 α) : sin α = -3/5 :=
sorry

-- Condition for second proof
def cond2 : α -> Prop := λ α, tan α = 3

-- Proof for second condition
theorem proof2 (h : cond2 α) : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
sorry

end proof1_proof2_l572_572021


namespace arthur_dinner_cost_l572_572851

theorem arthur_dinner_cost :
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  final_cost = 38 :=
by
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  show final_cost = 38, from sorry

end arthur_dinner_cost_l572_572851


namespace range_of_a_l572_572244

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 3 → true) ∧
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 5 → false) →
  1 < a ∧ a ≤ 7 / 5 :=
by
  sorry

end range_of_a_l572_572244


namespace shaded_areas_sum_l572_572972

theorem shaded_areas_sum (triangle_area : ℕ) (parts : ℕ)
  (h1 : triangle_area = 18)
  (h2 : parts = 9) :
  3 * (triangle_area / parts) = 6 :=
by
  sorry

end shaded_areas_sum_l572_572972


namespace island_coloring_l572_572791

theorem island_coloring (countries : Type) [fintype countries] (adjacent : countries → countries → Prop)
  (triangle_shape : ∀ (c : countries), ∃ (v1 v2 v3 : ℕ), true)
  (adjacency_entire_side : ∀ (c1 c2 : countries), adjacent c1 c2 → ∃ (s : ℕ), true) :
  ∃ (coloring : countries → fin 3), ∀ (c1 c2 : countries), adjacent c1 c2 → coloring c1 ≠ coloring c2 :=
sorry

end island_coloring_l572_572791


namespace candy_count_l572_572796

-- Definitions based on conditions
def cookies : ℕ := 42
def brownies : ℕ := 21
def people : ℕ := 7
def pieces_per_person : ℕ := 18

-- The problem to prove
theorem candy_count : 
  ∃ (candy : ℕ), (people * pieces_per_person) = (cookies + brownies + candy) ∧ candy = 63 :=
by {
  -- Assign the total number of dessert pieces needed
  let total_desserts_needed := people * pieces_per_person,
  
  -- Calculate the total number of desserts Pat already has
  let total_existing_desserts := cookies + brownies,

  -- Candy count calculation
  let required_candy := total_desserts_needed - total_existing_desserts,

  -- Prove the statement
  use required_candy,
  split,
  {
    -- Ensure total count is consistent
    exact eq.refl total_desserts_needed,
  },
  {
    -- Ensure candy count is correct
    exact eq.refl 63,
  }
}

end candy_count_l572_572796


namespace divisors_of_180_not_divisible_by_3_l572_572687

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572687


namespace rectangle_painting_problem_l572_572925

theorem rectangle_painting_problem
  (a b : ℕ)
  (h₁ : a ≥ 5)
  (h₂ : b > a)
  (h₃ : ab = 3 * (a - 4) * (b - 4))
  : ∃ s : finset (ℕ × ℕ), ∀ p ∈ s, (p.1 ≥ 5 ∧ p.2 > p.1) ∧ s.card = 4 := sorry

end rectangle_painting_problem_l572_572925


namespace pyramid_partition_volumes_l572_572607

noncomputable def pyramid_partition_ratios (S A B C D P Q V1 V2 : ℝ) : Prop :=
  let P := ((S + B) / 2 : ℝ)
  let Q := ((S + D) / 2 : ℝ)
  (V1 < V2) → 
  (V2 / V1 = 5)

theorem pyramid_partition_volumes
  (S A B C D P Q : ℝ)
  (V1 V2 : ℝ)
  (hP : P = (S + B) / 2)
  (hQ : Q = (S + D) / 2)
  (hV1 : V1 < V2)
  : V2 / V1 = 5 := 
sorry

end pyramid_partition_volumes_l572_572607


namespace probability_no_shaded_rectangle_l572_572026

-- Definitions
def total_rectangles_per_row : ℕ := (2005 * 2004) / 2
def shaded_rectangles_per_row : ℕ := 1002 * 1002

-- Proposition to prove
theorem probability_no_shaded_rectangle : 
  (1 - (shaded_rectangles_per_row : ℝ) / (total_rectangles_per_row : ℝ)) = (0.25 / 1002.25) := 
sorry

end probability_no_shaded_rectangle_l572_572026


namespace cos2alpha_minus_sin2alpha_l572_572760

theorem cos2alpha_minus_sin2alpha (α : ℝ) (h1 : α ∈ Set.Icc (-π/2) 0) 
  (h2 : (Real.sin (3 * α)) / (Real.sin α) = 13 / 5) :
  Real.cos (2 * α) - Real.sin (2 * α) = (3 + Real.sqrt 91) / 10 :=
sorry

end cos2alpha_minus_sin2alpha_l572_572760


namespace leadership_selection_ways_l572_572488

theorem leadership_selection_ways (M : ℕ) (chiefs : ℕ) (supporting_chiefs : ℕ) (officers_per_supporting_chief : ℕ) 
  (M_eq : M = 15) (chiefs_eq : chiefs = 1) (supporting_chiefs_eq : supporting_chiefs = 2) 
  (officers_eq : officers_per_supporting_chief = 3) : 
  (M * (M - 1) * (M - 2) * (Nat.choose (M - 3) officers_per_supporting_chief) * (Nat.choose (M - 6) officers_per_supporting_chief)) = 3243240 := by
  simp [M_eq, chiefs_eq, supporting_chiefs_eq, officers_eq]
  norm_num
  sorry

end leadership_selection_ways_l572_572488


namespace polynomial_value_l572_572174

theorem polynomial_value
  (p q r s t : ℝ)
  (g : ℝ → ℝ)
  (h₁ : g = λ x, p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h₂ : g (-1) = 4) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 64 := by
sorry

end polynomial_value_l572_572174


namespace base_six_to_ten_l572_572436

theorem base_six_to_ten (n : ℕ) (h : n = 34521) : (3 * 6^4 + 4 * 6^3 + 5 * 6^2 + 2 * 6^1 + 1 * 6^0) = 4945 :=
by
  have h₁ : 3 * 6^4 = 3888 := rfl
  have h₂ : 4 * 6^3 = 864 := rfl
  have h₃ : 5 * 6^2 = 180 := rfl
  have h₄ : 2 * 6^1 = 12 := rfl
  have h₅ : 1 * 6^0 = 1 := rfl
  calc
    3 * 6^4 + 4 * 6^3 + 5 * 6^2 + 2 * 6^1 + 1 * 6^0
        = 3888 + 864 + 180 + 12 + 1 : by rw [h₁, h₂, h₃, h₄, h₅]
    ... = 4945 : by norm_num

end base_six_to_ten_l572_572436


namespace find_a_l572_572248

theorem find_a (a b : ℝ) (h : (deriv (λ x : ℝ, x^3 - a * x^2 + b) 1 = -1)): a = 2 :=
sorry

end find_a_l572_572248


namespace x_minus_q_eq_3_minus_2q_l572_572711

theorem x_minus_q_eq_3_minus_2q (x q : ℝ) (h₁ : |x - 3| = q) (h₂ : x < 3) : x - q = 3 - 2q :=
by
sorry

end x_minus_q_eq_3_minus_2q_l572_572711


namespace necessary_and_sufficient_condition_for_purely_imaginary_l572_572233

theorem necessary_and_sufficient_condition_for_purely_imaginary (x : ℝ) :
  ((x = 1) ↔ (∃ z : ℂ, z = complex.mk 0 (x + 1) ∧ z.im ≠ 0)) :=
by
  sorry

end necessary_and_sufficient_condition_for_purely_imaginary_l572_572233


namespace required_number_of_shirts_l572_572286

/-
In a shop, there is a sale of clothes. Every shirt costs $5, every hat $4, and a pair of jeans $10.
You need to pay $51 for a certain number of shirts, two pairs of jeans, and four hats.
Prove that the number of shirts you need to buy is 3.
-/

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_payment : ℕ := 51
def number_of_jeans : ℕ := 2
def number_of_hats : ℕ := 4

theorem required_number_of_shirts (S : ℕ) (h : 5 * S + 2 * jeans_cost + 4 * hat_cost = total_payment) : S = 3 :=
by
  -- This statement asserts that given the defined conditions, the number of shirts that satisfies the equation is 3.
  sorry

end required_number_of_shirts_l572_572286


namespace x_minus_q_in_terms_of_q_l572_572707

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h : |x - 3| = q) (hx : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_in_terms_of_q_l572_572707


namespace tan_product_identity_l572_572107

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572107


namespace sum_of_odd_positions_l572_572844

-- Given condition: The sum of the first n terms of the sequence a_n is S_n = 2^(n+1) - 1
def S (n : ℕ) : ℕ := 2 ^ (n + 1) - 1

-- Definition of the general terms a_n based on the given condition
def a : ℕ → ℕ
| 1       := 3
| (n + 1) := if n = 0 then 2 else 2 ^ (n + 1)

-- Define the sum of terms at odd positions in the first 2n terms
def sum_odd_positions (n : ℕ) : ℕ :=
  (List.range (2 * n)).filter (fun k => k % 2 = 0).map a).sum

-- The proof problem
theorem sum_of_odd_positions (n : ℕ) : 
  sum_odd_positions n = (1/3:ℝ) * ((2:ℝ)^(2*n + 1) + 1) := 
sorry

end sum_of_odd_positions_l572_572844


namespace power_division_l572_572520

theorem power_division (a b : ℕ) (h : 64 = 8^2) : 8^{15} / (64^3) = 8^{9} :=
by
  -- statement here to prove the objective
  sorry

end power_division_l572_572520


namespace length_of_chord_EF_l572_572988

noncomputable def chord_length (theta_1 theta_2 : ℝ) : ℝ :=
  let x_1 := 2 * Real.cos theta_1
  let y_1 := Real.sin theta_1
  let x_2 := 2 * Real.cos theta_2
  let y_2 := Real.sin theta_2
  Real.sqrt ((x_2 - x_1)^2 + (y_2 - y_1)^2)

theorem length_of_chord_EF :
  ∀ (theta_1 theta_2 : ℝ), 
  (2 * Real.cos theta_1) + (Real.sin theta_1) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_2) + (Real.sin theta_2) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_1)^2 + 4 * (Real.sin theta_1)^2 = 4 →
  (2 * Real.cos theta_2)^2 + 4 * (Real.sin theta_2)^2 = 4 →
  chord_length theta_1 theta_2 = 8 / 5 :=
by
  intros theta_1 theta_2 h1 h2 h3 h4
  sorry

end length_of_chord_EF_l572_572988


namespace general_formula_arithmetic_sequence_l572_572612

theorem general_formula_arithmetic_sequence :
  (∃ (a_n : ℕ → ℕ) (d : ℕ), d ≠ 0 ∧ 
    (a_2 = a_1 + d) ∧ 
    (a_4 = a_1 + 3 * d) ∧ 
    (a_2^2 = a_1 * a_4) ∧
    (a_5 = a_1 + 4 * d) ∧ 
    (a_6 = a_1 + 5 * d) ∧ 
    (a_5 + a_6 = 11) ∧ 
    ∀ n, a_n = a_1 + (n - 1) * d) → 
  ∀ n, a_n = n := 
sorry

end general_formula_arithmetic_sequence_l572_572612


namespace max_area_of_triangle_DEF_l572_572425

noncomputable def max_area_triangle (x : ℝ) (hx1 : 0 < x) (hx2 : x < 12) : ℝ :=
  let s := (12 + 51 * x) / 2
  let a := s * ((s - 12) * (s - 25 * x) * (s - 26 * x))
  real.sqrt(a)

theorem max_area_of_triangle_DEF {x : ℝ} (hx1 : 0 < x) (hx2 : x < 12) : 
  max_area_triangle x hx1 hx2 = 143 / 4 :=
sorry

end max_area_of_triangle_DEF_l572_572425


namespace final_largest_number_l572_572449

def initial_seq : list ℕ := list.repeat [2, 1, 1, 6] 25 |>.join

def sum_of_two_digits (a b : ℕ) : ℕ :=
  if a + b ≤ 9 then a + b else 0

def can_perform_operation (list_digits : list ℕ) : Prop :=
  ∃ (a b : ℕ) (pre post : list ℕ),
    list_digits = pre ++ [a, b] ++ post ∧ a + b ≤ 9

def iterate_op_until_possible (digits : list ℕ) : list ℕ :=
  if can_perform_operation digits then
    -- This assumes we somehow transform the list, hole denotes the implementation
    sorry 
  else
    digits

theorem final_largest_number : 
  let final_number := iterate_op_until_possible initial_seq in
  list.length final_number = 50 ∧ final_number = list.repeat [4, 6] 25 |>.join :=
by
  sorry

end final_largest_number_l572_572449


namespace trader_sold_30_meters_l572_572485

theorem trader_sold_30_meters (x : ℕ) : (x * 150 = 4500) → x = 30 :=
by {
  assume h : x * 150 = 4500,
  sorry
}

end trader_sold_30_meters_l572_572485


namespace arthur_spent_38_l572_572857

noncomputable def arthur_meal_cost : ℝ := 
  let appetizer := 8
  let entree := 20
  let wine := 3 * 2
  let dessert := 6
  let total_no_discount := appetizer + entree + wine + dessert
  let discount := entree / 2
  let total_with_discount := total_no_discount - discount
  let tip := total_no_discount * 0.20
  total_with_discount + tip

theorem arthur_spent_38 :
  arthur_meal_cost = 38 :=
by
  rw [arthur_meal_cost]
  have appetizer : ℝ := 8
  have entree : ℝ := 20
  have wine : ℝ := 3 * 2
  have dessert : ℝ := 6
  have total_no_discount : ℝ := appetizer + entree + wine + dessert
  have discount : ℝ := entree / 2 
  have total_with_discount : ℝ := total_no_discount - discount
  have tip : ℝ := total_no_discount * 0.20
  calc  total_with_discount + tip = (8 + 20 + 6 + 2 * 3) - (20 / 2) + 0.20 * (8 + 20 + 6 + 2 * 3) : by rw [total_no_discount]
                          ... = 38 : by norm_num

end arthur_spent_38_l572_572857


namespace remainder_b22_div_35_l572_572764

def b_n (n : ℕ) : Nat :=
  ((List.range (n + 1)).drop 1).foldl (λ acc k => acc * 10^(Nat.digits 10 k).length + k) 0

theorem remainder_b22_div_35 : (b_n 22) % 35 = 17 :=
  sorry

end remainder_b22_div_35_l572_572764


namespace total_animals_correct_l572_572513

section 
variable 
  (snakes : ℕ)
  (arctic_foxes : ℕ)
  (leopards : ℕ)
  (bee_eaters : ℕ)
  (cheetahs : ℕ)
  (alligators : ℕ)
  (total : ℕ)

-- Define the initial counts
def snakes := 100
def arctic_foxes := 80
def leopards := 20

-- Define the derived counts based on conditions
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

-- Define the total number of animals based on the derived counts
def total := alligators + bee_eaters + snakes + arctic_foxes + leopards + cheetahs

-- Proof objective
theorem total_animals_correct : total = 670 := by 
  sorry
end

end total_animals_correct_l572_572513


namespace number_of_zeros_of_g_l572_572634

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x < 2 then -2^|x-1| + 3 else if x ≥ 2 then (1/2) * f (x-2) else 0

def g (x : ℝ) : ℝ := x * f x - 1

theorem number_of_zeros_of_g : (finset.card {x : ℝ | g x = 0}).to_nat = 6 := 
sorry

end number_of_zeros_of_g_l572_572634


namespace tan_product_pi_nine_l572_572130

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572130


namespace S_range_l572_572234

variables (x : ℝ) (x1 x2 x3 x4 : ℝ)
variable (i : ℕ)
variable (S : ℝ)

-- Conditions
axiom sum_eq : x1 + x2 - x3 + x4 = 1
axiom non_neg : ∀ i, 0 ≤ x

-- Definition of S
def S_def : ℝ := 1 - (x1^4 + x2^4 + x3^4 + x4^4) - 6 * (x1 * x2 + x1 * x3 + x1 * x4 + x2 * x3 + x2 * x4 + x3 * x4)

-- Theorem statement
theorem S_range : 0 ≤ S_def x1 x2 x3 x4 ∧ S_def x1 x2 x3 x4 ≤ 3/4 :=
by
  sorry

end S_range_l572_572234


namespace sum_positive_integer_solutions_l572_572583

def operation (a b : ℝ) : ℝ := b - a + a * b - 2

theorem sum_positive_integer_solutions :
  (∑ k in (finset.filter (λ x, 3 ※ x < 5) (finset.Icc 1 10)).val, k) = 3 := 
by 
  sorry

end sum_positive_integer_solutions_l572_572583


namespace simplify_and_evaluate_equals_l572_572812

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l572_572812


namespace cloud_ratio_l572_572526

theorem cloud_ratio (D Carson Total : ℕ) (h1 : Carson = 6) (h2 : Total = 24) (h3 : Carson + D = Total) :
  (D / Carson) = 3 := by
  sorry

end cloud_ratio_l572_572526


namespace celsius_to_fahrenheit_l572_572015

theorem celsius_to_fahrenheit (temp_celsius : ℝ) (h : temp_celsius = 50) : (temp_celsius * (9/5) + 32) = 122 := by
  rw [h]
  norm_num
  sorry

end celsius_to_fahrenheit_l572_572015


namespace unique_real_root_l572_572400

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x : ℝ, f(f(x + 1)) = x^3 + 1

theorem unique_real_root : ∃! x : ℝ, f x = 0 :=
sorry

end unique_real_root_l572_572400


namespace intervals_of_increase_max_min_on_interval_l572_572632
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem intervals_of_increase :
  ∀ x : ℝ, (x < -1 ∨ x > 3) → deriv f x > 0 :=
by
  intros x h
  sorry

theorem max_min_on_interval :
  is_max_on f (-1 : ℝ) (Icc (-2 : ℝ) 2 : set ℝ) ∧
  is_min_on f (2 : ℝ) (Icc (-2 : ℝ) 2 : set ℝ) :=
by
  sorry

end intervals_of_increase_max_min_on_interval_l572_572632


namespace geometric_sequence_common_ratio_l572_572603

theorem geometric_sequence_common_ratio (a : ℕ → ℤ) (q : ℤ)  
  (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : 4 * a 1 + a 3 = 4 * a 2) : 
  q = 2 := 
by {
  -- Proof is omitted here
  sorry
}

end geometric_sequence_common_ratio_l572_572603


namespace tan_product_l572_572132

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572132


namespace tan_identity_proof_l572_572095

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572095


namespace merchant_profit_l572_572716

theorem merchant_profit (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 :=
by
  let ratio := 3 / 2
  have hS : S = ratio * C := by linarith [h]
  have profit : S - C = (ratio - 1) * C := by linarith [hS]
  have profit_percentage : ((ratio - 1) * C / C) * 100 = 50 := by
    field_simp
    linarith
  exact profit_percentage

end merchant_profit_l572_572716


namespace tan_product_l572_572146

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572146


namespace bob_cleaning_time_is_correct_l572_572051

-- Definitions for conditions
def timeAliceTakes : ℕ := 32
def bobTimeFactor : ℚ := 3 / 4

-- Theorem to prove
theorem bob_cleaning_time_is_correct : (bobTimeFactor * timeAliceTakes : ℚ) = 24 := 
by
  sorry

end bob_cleaning_time_is_correct_l572_572051


namespace neither_sufficient_nor_necessary_l572_572348

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0) ↔ (ab > 0)) := 
sorry

end neither_sufficient_nor_necessary_l572_572348


namespace probability_sum_eq_k_l572_572209

open Polynomial

theorem probability_sum_eq_k (n m k : ℕ) :
  ∃ p : ℚ, p = coeff (m * (X + X^2 + ⋯ + X^n)) k / n^m :=
sorry

end probability_sum_eq_k_l572_572209


namespace rob_has_24_cards_l572_572806

theorem rob_has_24_cards 
  (r : ℕ) -- total number of baseball cards Rob has
  (dr : ℕ) -- number of doubles Rob has
  (hj: dr = 1 / 3 * r) -- one third of Rob's cards are doubles
  (jess_doubles : ℕ) -- number of doubles Jess has
  (hj_mult : jess_doubles = 5 * dr) -- Jess has 5 times as many doubles as Rob
  (jess_doubles_40 : jess_doubles = 40) -- Jess has 40 doubles baseball cards
: r = 24 :=
by
  sorry

end rob_has_24_cards_l572_572806


namespace three_pow_zero_l572_572061

theorem three_pow_zero : 3^0 = 1 :=
by sorry

end three_pow_zero_l572_572061


namespace Barry_reach_l572_572322

noncomputable def Larry_full_height : ℝ := 5
noncomputable def Larry_shoulder_height : ℝ := Larry_full_height - 0.2 * Larry_full_height
noncomputable def combined_reach : ℝ := 9

theorem Barry_reach :
  combined_reach - Larry_shoulder_height = 5 := 
by
  -- Correct answer verification comparing combined reach minus Larry's shoulder height equals 5
  sorry

end Barry_reach_l572_572322


namespace distance_point_to_line_l572_572829

theorem distance_point_to_line :
  let p : ℝ × ℝ := (2, 1)
  let a := 1/2
  let b := -1
  let c := 1
  let line_distance_formula (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
    abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)
  line_distance_formula p a b c = 2 / 5 * Real.sqrt 5 :=
by
  sorry

end distance_point_to_line_l572_572829


namespace probability_of_square_or_circle_is_seven_tenths_l572_572497

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- The number of squares or circles
def num_squares_or_circles : ℕ := num_squares + num_circles

-- The probability of selecting a square or a circle
def probability_square_or_circle : ℚ := num_squares_or_circles / total_figures

-- The theorem stating the required proof
theorem probability_of_square_or_circle_is_seven_tenths :
  probability_square_or_circle = 7/10 :=
sorry -- proof goes here

end probability_of_square_or_circle_is_seven_tenths_l572_572497


namespace selection_group_l572_572389

theorem selection_group (B G : ℕ) (c5_3 : B = 5) (c4_2 : G = 4) :
  (B.choose 3) * (G.choose 2) = (nat.choose 5 3) * (nat.choose 4 2) :=
by
  -- Let B = 5 and G = 4 based on conditions
  rw [c5_3, c4_2]
  -- Simplify the goal
  ring

end selection_group_l572_572389


namespace sum_of_digits_0_to_2012_l572_572315

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l572_572315


namespace negation_proof_l572_572405

-- Definitions based on conditions
def Line : Type := sorry  -- Define a type for lines (using sorry for now)
def Plane : Type := sorry  -- Define a type for planes (using sorry for now)

-- Condition definition
def is_perpendicular (l : Line) (α : Plane) : Prop := sorry  -- Define what it means for a plane to be perpendicular to a line (using sorry for now)

-- Given condition
axiom condition : ∀ (l : Line), ∃ (α : Plane), is_perpendicular l α

-- Statement to prove
theorem negation_proof : (∃ (l : Line), ∀ (α : Plane), ¬is_perpendicular l α) :=
sorry

end negation_proof_l572_572405


namespace matrix_power_100_l572_572070

def matrix_100_pow : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![200, 1]]

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

theorem matrix_power_100 (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A^100 = matrix_100_pow :=
by
  sorry

end matrix_power_100_l572_572070


namespace dot_product_identity_transformation_to_right_triangle_or_equality_l572_572323

section LinearTransformation

variables {R : Type*} [LinearOrderedField R]

structure LinearTransformation :=
  (a b c d : R)
  (f : R × R → R × R := λ u, (a * u.1 + b * u.2, c * u.1 + d * u.2))
  (g : R × R → R × R := λ v, (a * v.1 + c * v.2, b * v.1 + d * v.2))

theorem dot_product_identity (LT : LinearTransformation) (u v : R × R) :
  let f_u := LT.f u,
      g_v := LT.g v in
  f_u.1 * v.1 + f_u.2 * v.2 = u.1 * g_v.1 + u.2 * g_v.2 :=
by sorry

theorem transformation_to_right_triangle_or_equality (LT : LinearTransformation) (k : R) (u : R × R) :
  let P := (k * u.1, k * u.2),
      Q := LT.f P,
      R := LT.g P in
  (Q = R) ∨ ((Q.1 - O.1)^2 + (Q.2 - O.2)^2 + (R.1 - O.1)^2 + (R.2 - O.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2) :=
by sorry

end LinearTransformation

end dot_product_identity_transformation_to_right_triangle_or_equality_l572_572323


namespace x_minus_q_eq_3_minus_2q_l572_572706

theorem x_minus_q_eq_3_minus_2q {x q : ℝ} (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_eq_3_minus_2q_l572_572706


namespace value_of_f_g_pi_l572_572755

noncomputable def f : ℝ → ℤ :=
  λ x, if x > 0 then 1 else if x = 0 then 0 else -1

noncomputable def g : ℝ → ℤ :=
  λ x, if x.is_rational then 1 else 0

theorem value_of_f_g_pi : f (g Real.pi) = 0 :=
  by sorry

end value_of_f_g_pi_l572_572755


namespace ice_cream_tubs_eaten_l572_572428

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end ice_cream_tubs_eaten_l572_572428


namespace expression_value_l572_572230

theorem expression_value (a b c d : ℝ) (h1 : a * b = 1) (h2 : c + d = 0) :
  -((a * b) ^ (1/3)) + (c + d).sqrt + 1 = 0 :=
by sorry

end expression_value_l572_572230


namespace coplanar_points_l572_572187

theorem coplanar_points (b : ℝ) :
  let v₁ := ![2, b, 0]
  let v₂ := ![0, 2, b]
  let v₃ := ![b, 0, 2]
  matrix.det ↑![v₁, v₂, v₃] = 0 ↔ b = -2 := 
begin
  sorry
end

end coplanar_points_l572_572187


namespace arthur_dinner_cost_l572_572850

theorem arthur_dinner_cost :
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  final_cost = 38 :=
by
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  show final_cost = 38, from sorry

end arthur_dinner_cost_l572_572850


namespace sum_of_digits_0_to_2012_l572_572313

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l572_572313


namespace problem_ineq_l572_572595

theorem problem_ineq (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_prod : (∏ i, a i) = 1) :
  (∏ i, (2 + a i)) ≥ 3^n :=
by
  sorry

end problem_ineq_l572_572595


namespace cone_lateral_surface_area_is_12pi_l572_572626

def radius : ℝ := 2
def height : ℝ := 4 * Real.sqrt 2

noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)

noncomputable def lateral_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area_is_12pi : lateral_surface_area radius (slant_height radius height) = 12 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_is_12pi_l572_572626


namespace isosceles_triangle_bisector_properties_l572_572839

theorem isosceles_triangle_bisector_properties:
  ∀ (T : Type) (triangle : T)
  (is_isosceles : Prop) (vertex_angle_bisector_bisects_base : Prop) (vertex_angle_bisector_perpendicular_to_base : Prop),
  is_isosceles 
  → (vertex_angle_bisector_bisects_base ∧ vertex_angle_bisector_perpendicular_to_base) :=
sorry

end isosceles_triangle_bisector_properties_l572_572839


namespace divisors_not_divisible_by_3_l572_572678

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572678


namespace circle_properties_l572_572330

noncomputable def circle_center_and_radius (x y: ℝ) : Prop :=
  (x^2 + 8*x + y^2 - 10*y = 11)

theorem circle_properties :
  (∃ (a b r : ℝ), (a, b) = (-4, 5) ∧ r = 2 * Real.sqrt 13 ∧ circle_center_and_radius x y → a + b + r = 1 + 2 * Real.sqrt 13) :=
  sorry

end circle_properties_l572_572330


namespace tan_product_eq_three_l572_572080

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572080


namespace divisors_not_divisible_by_3_l572_572677

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572677


namespace complement_of_A_in_U_l572_572355

def U : Set ℕ := {1,3,5,7,9}
def A : Set ℕ := {1,9}
def complement_U_A : Set ℕ := {3,5,7}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_in_U_l572_572355


namespace remainder_1000th_S_number_div_1000_l572_572753

def is_valid_S_sequence_number (n : ℕ) : Prop :=
  nat.popcount n = 8

def S : ℕ → ℕ
| 0     => 0    -- There is no 0-th element, 1st element starts from index 1
| (n+1) => Nat.find (λ k, is_valid_S_sequence_number (k + S n))

theorem remainder_1000th_S_number_div_1000 : (S 1000) % 1000 = 32 :=
sorry

end remainder_1000th_S_number_div_1000_l572_572753


namespace candy_fraction_unclaimed_l572_572491

theorem candy_fraction_unclaimed :
  let total_share_al := 2 / 5,
      total_share_bert := 3 / 10,
      total_share_carl := 3 / 50,
      total_share_dana := 3 / 125,
      total_taken := total_share_al + total_share_bert + total_share_carl + total_share_dana in
  total_taken = 98 / 125 →
  let fraction_unclaimed := 1 - total_taken in
  fraction_unclaimed = 27 / 125 :=
sorry

end candy_fraction_unclaimed_l572_572491


namespace fewer_soldiers_on_second_side_l572_572406

theorem fewer_soldiers_on_second_side :
  (∀ (food1 : ℕ) (food2 : ℕ) (total_food : ℕ) (soldiers1 : ℕ) 
  (soldiers2 : ℕ),
    food1 = 10 → 
    food2 = food1 - 2 → 
    total_food = 68000 → 
    soldiers1 = 4000 → 
    total_food = soldiers1 * food1 + soldiers2 * food2 → 
    4000 - soldiers2 = 500) :=
begin
  intros food1 food2 total_food soldiers1 soldiers2 h_food1 h_food2 h_total_food h_soldiers1 h_total_equation,
  -- Proof would go here
  sorry
end

end fewer_soldiers_on_second_side_l572_572406


namespace trains_cross_time_l572_572454

def length_train1 := 140 -- in meters
def length_train2 := 160 -- in meters

def speed_train1_kmph := 60 -- in km/h
def speed_train2_kmph := 48 -- in km/h

def kmph_to_mps (speed : ℕ) := speed * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

def relative_speed_mps := speed_train1_mps + speed_train2_mps

def total_length := length_train1 + length_train2

def time_to_cross := total_length / relative_speed_mps

theorem trains_cross_time : time_to_cross = 10 :=
  by sorry

end trains_cross_time_l572_572454


namespace random_event_proof_l572_572587

-- Definitions for needed conditions
def no_real_solutions_x_squared_plus_one_eq_zero : Prop :=
  ∀ x : ℝ, x^2 + 1 ≠ 0

def random_event_condition : Prop :=
  ∀ x : ℝ, x ≠ 0 → x > (1 / x)

def reciprocal_function_not_increasing : Prop :=
  ∀ x y : ℝ, x < y → y ≠ 0 → x ≠ 0 → (1 / x) ≥ (1 / y)

def zero_sum_of_squares_implies_zero : Prop :=
  ∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem statement
theorem random_event_proof :
  no_real_solutions_x_squared_plus_one_eq_zero →
  random_event_condition →
  reciprocal_function_not_increasing →
  zero_sum_of_squares_implies_zero →
  random_event_condition :=
by {
  intro h1 h2 h3 h4,
  exact h2,
}

end random_event_proof_l572_572587


namespace number_of_oddly_powerful_integers_lt_500_l572_572997

noncomputable def count_oddly_powerful_integers_lt_500 : ℕ :=
  let count_cubes := 7 -- we counted cubes: 1^3, 2^3, 3^3, 4^3, 5^3, 6^3, 7^3
  let count_fifth_powers := 1 -- the additional fifth power not a cube: 3^5
  count_cubes + count_fifth_powers

theorem number_of_oddly_powerful_integers_lt_500 : count_oddly_powerful_integers_lt_500 = 8 :=
  sorry

end number_of_oddly_powerful_integers_lt_500_l572_572997


namespace polynomial_factorization_l572_572447

theorem polynomial_factorization : (∀ x : ℤ, x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1)) := by
  intro x
  sorry

end polynomial_factorization_l572_572447


namespace shaded_percentage_l572_572880

def square_area (side_length : ℕ) : ℕ := side_length ^ 2

def shaded_areas (r1 r2 r3 : ℕ) : ℕ := r1 + r2 + r3

def percent_shaded (total_area shaded_area : ℕ) : ℕ :=
  (shaded_area * 100) / total_area

theorem shaded_percentage (side_length : ℕ) 
  (hs : side_length = 7)
  (h_total_area : square_area side_length = 49)
  (h_r1 : ∀ x y : ℕ, (x = 1) → (y = 1) → r1 = x * y)
  (r1 : ℕ = 1)
  (h_r2 : ∀ x y : ℕ, (x = 3) → (y = 3) → r2 = x * y)
  (r2 : ℕ = 9)
  (h_r3 : ∀ x y : ℕ, (x = 2) → (y = 2) → r3 = x * y)
  (r3 : ℕ = 4)
  (h_shaded_area : shaded_areas r1 r2 r3 = 14) :
  percent_shaded (square_area side_length) (shaded_areas r1 r2 r3) = 28 :=
by
  sorry

end shaded_percentage_l572_572880


namespace tan_product_equals_three_l572_572164

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572164


namespace find_x_l572_572068

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end find_x_l572_572068


namespace analytical_expression_of_f_range_of_m_minimum_s_l572_572635

-- Part (1)
theorem analytical_expression_of_f (a b : ℝ) (h_odd : ∀ x, f x = - f (-x)) (h_at1 : f 1 = 3/2) :
  f = λ x, 6 * x / (x^2 + 3) :=
sorry

-- Part (2)
theorem range_of_m (f : ℝ → ℝ) (h_within_range : ∀ x ∈ set.Icc 1 3, abs (f x - m) ≤ 1) :
  1 / 2 ≤ m ∧ m ≤ 1 + real.sqrt 3 :=
sorry

-- Part (3)
theorem minimum_s (f : ℝ → ℝ) (h_s_inequality : ∀ n ∈ set.Icc 0 1, ∀ t ∈ set.Ioi 0, 
  f t + n * f (t / 3) ≤ s) :
  s = 3 :=
sorry

end analytical_expression_of_f_range_of_m_minimum_s_l572_572635


namespace integer_1000_in_column_F_l572_572962

def column_sequence (n : ℕ) : String :=
  let seq := ["A", "B", "C", "D", "E", "F", "G", "F", "E", "D", "C", "B", "A"]
  seq[(n % 13)]

theorem integer_1000_in_column_F :
  column_sequence (1000 - 2 + 1) = "F" :=
by
  -- This is a placeholder for the proof
  sorry

end integer_1000_in_column_F_l572_572962


namespace tangent_product_eq_three_l572_572118

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572118


namespace find_x_l572_572067

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end find_x_l572_572067


namespace max_leap_years_in_200_years_l572_572506

-- Define what it means for a year to be a leap year in this modified calendrical system
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)

-- Define the 200-year period we are considering
def year_range : List ℕ := List.range' 1 200

-- Count the number of leap years in the given range
def leap_years_in_range (years : List ℕ) : ℕ :=
  years.countp is_leap_year

-- The main theorem stating that the maximum number of leap years in this modified calendrical system
-- over a period of 200 years starting with year 1 is 48
theorem max_leap_years_in_200_years : leap_years_in_range year_range = 48 := by
  sorry

end max_leap_years_in_200_years_l572_572506


namespace quadratic_equation_has_more_than_100_roots_l572_572748

/-- We start with the assumptions and definitions, as gathered from the problem -/
def quadratic_floor_equation_more_than_100_roots(p q : ℤ) : Prop :=
  ∀ (x : ℝ), ∃ (n : ℝ), floor (x^2) + p * x + q = 0 ∧ p ≠ 0 ∧ 100 < n

/-- Now stating the main theorem in Lean -/
theorem quadratic_equation_has_more_than_100_roots
  (p q : ℤ) (h_p : p ≠ 0) (h_q : q = 900) (h_p_val : p = 60) :
  quadratic_floor_equation_more_than_100_roots p q :=
sorry

end quadratic_equation_has_more_than_100_roots_l572_572748


namespace largest_sum_fraction_l572_572982

theorem largest_sum_fraction :
  max 
    ((1/3) + (1/2))
    (max 
      ((1/3) + (1/5))
      (max 
        ((1/3) + (1/6))
        (max 
          ((1/3) + (1/9))
          ((1/3) + (1/10))
        )
      )
    ) = 5/6 :=
by sorry

end largest_sum_fraction_l572_572982


namespace main_l572_572253

noncomputable def f (a k x : ℝ) : ℝ := log a ((1 + k * x) / (1 + x))

theorem main (a k : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : k ≠ 1) :
  (∀ x : ℝ, f a k (-x) = -f a k x → k = -1) ∧ 
  (a > 1 → ∃ t : ℝ, t ∈ Ico 0 1 → f a (-1) t ∈ Icc (1/a - 1/2) (1/2) → a > 2) :=
by
  sorry

end main_l572_572253


namespace tan_product_l572_572142

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572142


namespace diameter_in_given_form_l572_572399

-- Define the original radius of the sphere
def R : ℝ := 7

-- Define the volume function for a sphere given radius r
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of the original sphere
def V : ℝ := volume R

-- Define the volume of the new sphere which is three times the original volume
def V_new : ℝ := 3 * V

-- Define the new radius based on the new volume
def r_new : ℝ := (3 * R^3)^(1/3)

-- Define the diameter of the new sphere
def d_new : ℝ := 2 * r_new

-- Define a and b based on the problem statement
def a : ℝ := 14
def b : ℝ := 3

-- Prove that a + b = 17 given the conditions
theorem diameter_in_given_form : a + b = 17 :=
by {
  -- Placeholder for actual proof
  sorry
}

end diameter_in_given_form_l572_572399


namespace right_angled_triangle_only_B_l572_572002

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l572_572002


namespace tangent_product_eq_three_l572_572117

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572117


namespace triangle_perimeter_l572_572953

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572953


namespace find_a_l572_572251

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem find_a (a : ℝ) (h : f (f a) = f 9 + 1) : a = -1/4 := 
by 
  sorry

end find_a_l572_572251


namespace tan_product_l572_572084

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572084


namespace count_seven_digit_symmetric_with_offset_1100_l572_572446

def is_symmetric (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_symmetric_with_offset (n m : ℕ) : Prop :=
  is_symmetric n ∧ is_symmetric (n + m)

theorem count_seven_digit_symmetric_with_offset_1100 :
  (Finset.filter (λ n => is_symmetric_with_offset n 1100) (Finset.range 10000000)).card = 810 := 
sorry

end count_seven_digit_symmetric_with_offset_1100_l572_572446


namespace num_non_divisible_by_3_divisors_l572_572674

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572674


namespace seating_permutations_l572_572181

theorem seating_permutations (P : Finset ℕ) (hP : P = {2, 3, 4, 5, 6, 7, 8}) :
  ∃ (perms : Finset (Fin n → Fin n)), perms.card = 6 ∧ 
    ∀ π ∈ perms, ∀ i ∈ P, π i ≠ i ∧ π (i + 1) % 8 ≠ π i ∧ π (i - 1 + 8) % 8 ≠ π i :=
begin
  sorry
end

end seating_permutations_l572_572181


namespace probability_all_white_is_zero_l572_572028

noncomputable def probability_all_white_drawn (total_balls white_balls black_balls red_balls drawn_balls : ℕ) : ℝ :=
  if h : white_balls < drawn_balls then 0 else sorry

theorem probability_all_white_is_zero :
  probability_all_white_drawn 16 6 7 3 7 = 0 :=
by
  simp [probability_all_white_drawn]
  sorry

end probability_all_white_is_zero_l572_572028


namespace hexagon_area_l572_572356

open Real

-- Define the lengths and radius provided in the conditions
noncomputable def side_length : ℝ := 14
noncomputable def circumradius : ℝ := 7

-- Prove the area of the specified hexagon
theorem hexagon_area :
  let s := side_length in
  let R := circumradius in
  let equilateral_hexagon_area := 6 * (1 / 2 * (s / 2) * (R * sin (π / 3))) in
  equilateral_hexagon_area = 73.5 * sqrt 3 :=
by
  sorry

end hexagon_area_l572_572356


namespace width_of_rectangular_prism_l572_572271

theorem width_of_rectangular_prism 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ) 
  (hl : l = 5) (hh : h = 15) (hd : d = 17) 
  (h_diag : d = real.sqrt (l^2 + w^2 + h^2)) 
  : w = real.sqrt 39 :=
by
  -- Proof goes here
  sorry

end width_of_rectangular_prism_l572_572271


namespace volume_difference_l572_572967

theorem volume_difference (h_A h_B : ℝ) (c_A c_B : ℝ)
    (h_A_pos : h_A = 12) (c_A_pos : c_A = 9)
    (h_B_pos : h_B = 9) (c_B_pos : c_B = 12) :
    π * (π * (((c_B / (2 * π))^2) * h_B) - π * (((c_A / (2 * π))^2) * h_A)) = 81 :=
by
    -- Define the radii
    let r_A := c_A / (2 * π)
    let r_B := c_B / (2 * π)
    -- Calculate volumes
    let V_A := π * (r_A ^ 2) * h_A
    let V_B := π * (r_B ^ 2) * h_B
    -- Calculate the positive difference times π
    have h : π * (V_B - V_A) = 81
    sorry

end volume_difference_l572_572967


namespace sum_of_m_n_l572_572478

-- Define the setup for the problem
def side_length_of_larger_square := 3
def side_length_of_smaller_square := 1
def side_length_of_given_rectangle_l1 := 1
def side_length_of_given_rectangle_l2 := 3
def total_area_of_larger_square := side_length_of_larger_square * side_length_of_larger_square
def area_of_smaller_square := side_length_of_smaller_square * side_length_of_smaller_square
def area_of_given_rectangle := side_length_of_given_rectangle_l1 * side_length_of_given_rectangle_l2

-- Define the variable for the area of rectangle R
def area_of_R := total_area_of_larger_square - (area_of_smaller_square + area_of_given_rectangle)

-- Given the problem statement, we need to find m and n such that the area of R is m/n.
def m := 5
def n := 1

-- We need to prove that m + n = 6 given these conditions
theorem sum_of_m_n : m + n = 6 := by
  sorry

end sum_of_m_n_l572_572478


namespace tan_identity_proof_l572_572099

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572099


namespace angle_BKO_is_rightangle_l572_572307

open Real EuclideanGeometry

variables {A B C K O : Point}
variables (ω₁ ω₂ : Circle)

-- Circle ω₁ passing through A and B and tangent to BC
def circle_ω₁ := ω₁ ∈ (circleThrough A B ∧ tangent ω₁ (lineBC B C))

-- Circle ω₂ passing through B and C, tangent to AB, intersecting ω₁ at K
def circle_ω₂ := ω₂ ∈ (circleThrough B C ∧ tangent ω₂ (lineBC A B) ∧ intersectingAt ω₂ ω₁ K)

-- O is the circumcenter
def circumcenter_ABC := O ∈ circumcenter (triangleABC A B C)

-- The angle BKO is a right angle
theorem angle_BKO_is_rightangle : circle_ω₁ ω₁ → circle_ω₂ ω₂ → circumcenter_ABC O → angle B K O = 90 := by
  sorry

end angle_BKO_is_rightangle_l572_572307


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_4_l572_572257

-- Define the function representing the quadratic equation
def quadratic_eq (m x : ℝ) := x^2 + (2 - m) * x + 1 - m

-- Part 1
theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
sorry

-- Part 2
theorem find_m_for_roots_difference_4 (m : ℝ) (H : m < 0) :
  (∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 ∧ x1 - x2 = 4) → m = -4 :=
sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_4_l572_572257


namespace determine_C_cards_l572_572422

-- Define the card numbers
def card_numbers : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- Define the card sum each person should have
def card_sum := 26

-- Define person's cards
def A_cards : List ℕ := [10, 12]
def B_cards : List ℕ := [6, 11]

-- Define sum constraints for A and B
def sum_A := A_cards.sum
def sum_B := B_cards.sum

-- Define C's complete set of numbers based on remaining cards and sum constraints
def remaining_cards := card_numbers.diff (A_cards ++ B_cards)
def sum_remaining := remaining_cards.sum

theorem determine_C_cards :
  (sum_A + (26 - sum_A)) = card_sum ∧
  (sum_B + (26 - sum_B)) = card_sum ∧
  (sum_remaining = card_sum) → 
  (remaining_cards = [8, 9]) :=
by
  sorry

end determine_C_cards_l572_572422


namespace area_of_quadrilateral_ABCD_l572_572190

theorem area_of_quadrilateral_ABCD
  (BD : ℝ) (hA : ℝ) (hC : ℝ) (angle_ABD : ℝ) :
  BD = 28 ∧ hA = 8 ∧ hC = 2 ∧ angle_ABD = 60 →
  ∃ (area_ABCD : ℝ), area_ABCD = 140 :=
by
  sorry

end area_of_quadrilateral_ABCD_l572_572190


namespace alex_needs_additional_coins_l572_572492

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end alex_needs_additional_coins_l572_572492


namespace total_number_of_animals_is_650_l572_572512

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l572_572512


namespace mutually_exclusive_not_complementary_l572_572023

namespace BallDrawingProblem

def bag : set ℕ := { red := 2, black := 2 }

def draw (bag : set ℕ) : set ℕ := by
  sorry -- Define the set of possible outcomes when drawing two balls.

def event_one_black (outcome : set ℕ) : Prop :=
  outcome.count (λ b, b = black) = 1

def event_two_black (outcome : set ℕ) : Prop :=
  outcome.count (λ b, b = black) = 2

theorem mutually_exclusive_not_complementary :
  ∀ outcome : set ℕ, event_one_black outcome → ¬ event_two_black outcome :=
by
  sorry -- The actual proof is omitted
end BallDrawingProblem

end mutually_exclusive_not_complementary_l572_572023


namespace min_area_triangle_min_area_acute_triangle_l572_572457

/-- Given a triangle \( T \) with sides \( a, b, c \), if increasing any one side by 1 
still results in a triangle, the minimal possible area of \( T \) is \( \frac{\sqrt{3}}{4} \). -/
theorem min_area_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (h1 : a + 2 > b + c) (h2 : b + 2 > a + c) (h3 : c + 2 > a + b) : 
  ∃ s, a = s ∧ b = s ∧ c = s ∧ (s = 1) ∧ (sqrt 3 / 4 * s^2 = sqrt 3 / 4) :=
by
  sorry

/-- Given a triangle \( T \) with sides \( a, b, c \), if increasing any one side by 1 
results in an acute triangle, the minimal possible area of \( T \) is \( \frac{\sqrt{3}}{4} (3 + 2\sqrt{2}) \). -/
theorem min_area_acute_triangle (a b c : ℝ) (h_equal : a = 1 + sqrt 2 ∧ b = 1 + sqrt 2 ∧ c = 1 + sqrt 2) 
  (h_acute : (a+1)^2 ≤ b^2 + c^2 ∧ (b+1)^2 ≤ a^2 + c^2 ∧ (c+1)^2 ≤ a^2 + b^2) : 
  equivalent sqrt 3 / 4 * (3 + 2 * sqrt 2) :=
by
  sorry

end min_area_triangle_min_area_acute_triangle_l572_572457


namespace total_weight_of_onions_l572_572809

variable (SaraOnions : Nat) (SaraWeightPerOnion : ℝ)
variable (SallyOnions : Nat) (SallyWeightPerOnion : ℝ)
variable (FredOnions : Nat) (FredWeightPerOnion : ℝ)
variable (JackOnions : Nat) (JackWeightPerOnion : ℝ)

#align SaraOnions 4
#align SaraWeightPerOnion 0.5
#align SallyOnions 5
#align SallyWeightPerOnion 0.4
#align FredOnions 9
#align FredWeightPerOnion 0.3
#align JackOnions 7
#align JackWeightPerOnion 0.6

theorem total_weight_of_onions :
  SaraOnions * SaraWeightPerOnion + SallyOnions * SallyWeightPerOnion + FredOnions * FredWeightPerOnion + JackOnions * JackWeightPerOnion = 10.9 := by
sorry

end total_weight_of_onions_l572_572809


namespace solution_set_of_inequality_l572_572765

variables {f : ℝ → ℝ}

def odd_function := ∀ x : ℝ, f (-x) = -f x
def strictly_decreasing_on_pos := ∀ x y : ℝ, (0 < x → x < y → f y < f x)
def f_at_2 := f 2 = 0

theorem solution_set_of_inequality (h1 : odd_function f) (h2 : strictly_decreasing_on_pos f) (h3 : f_at_2) :
  { x : ℝ | x * f x ≤ 0 } = { x : ℝ | x ∈ (-∞, -2] ∪ [2, ∞) } :=
sorry

end solution_set_of_inequality_l572_572765


namespace product_term_identity_l572_572178

theorem product_term_identity :
  (∏ k in finset.range (11) .map (λ i, i + 3), (1 - (1 / (k ^ 2)))) = 13 / 18 :=
  sorry

end product_term_identity_l572_572178


namespace bingo_possibilities_l572_572731

theorem bingo_possibilities:
  let choices := [15, 14, 13, 12, 11]
  in List.prod choices = 360360 :=
by
  simp [List.prod]
  exact Nat.mul_assoc 13 (12 * 11) 14 ▸
        Nat.mul_assoc 14 (12 * 11) 15 ▸
        rfl

end bingo_possibilities_l572_572731


namespace pyramid_base_dimensions_l572_572470

-- Definitions
def radius_cylinder := 2
def height_cylinder := 3
def height_pyramid := 10

def volume_cylinder (r s : ℝ) : ℝ := π * r^2 * s
def volume_pyramid (l w h : ℝ) : ℝ := (1 / 3) * l * w * h

def is_solution (l w : ℝ) : Prop :=
  volume_cylinder radius_cylinder height_cylinder = (1 / 2) * volume_pyramid l w height_pyramid

-- The theorem stating the problem's solution
theorem pyramid_base_dimensions : is_solution (18 * π / 5) 4 :=
  sorry

end pyramid_base_dimensions_l572_572470


namespace power_sum_divisible_by_5_l572_572549

theorem power_sum_divisible_by_5 (n : ℕ) : (2^(4*n + 1) + 3^(4*n + 1)) % 5 = 0 :=
by
  sorry

end power_sum_divisible_by_5_l572_572549


namespace sum_first_11_terms_arithmetic_seq_l572_572736

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α) (d a1 : α)
variable (a_arith_seq : ∀ n : ℕ, a n = a1 + n * d)
variable (h : a 9 = 0.5 * a 12 + 6)

theorem sum_first_11_terms_arithmetic_seq :
  (Σ i in Finset.range 11, a i) = 132 := by
  sorry

end sum_first_11_terms_arithmetic_seq_l572_572736


namespace domain_of_f_l572_572831

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f_l572_572831


namespace achievable_word_l572_572037

theorem achievable_word (n : ℕ) 
    (flights : Π a b : ℕ, list (List (λ x, x = "A" ∨ x = "B")))
    (achievable_len_w : ∀ w : list (λ x, x = 'A' ∨ x = 'B'), w.length = 2^n → achievable w)
    (finite_word : list (λ x, x = 'A' ∨ x = 'B')) :
    achievable finite_word :=
sorry

end achievable_word_l572_572037


namespace divisors_not_divisible_by_3_l572_572675

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572675


namespace general_term_formula_sum_first_20_abs_terms_l572_572220

noncomputable def arithmetic_sequence (n : ℕ) : ℤ := 2 * n - 11

def sum_of_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := (finset.range (n + 1)).sum a

def abs_arithmetic_sequence (n : ℕ) : ℤ := if (arithmetic_sequence n) < 0 then -(arithmetic_sequence n) else (arithmetic_sequence n)

theorem general_term_formula :
  ∃ (a : ℕ → ℤ) (d : ℤ), a 3 = -5 ∧ sum_of_terms a 4 = -24 ∧ ∀ n : ℕ, arithmetic_sequence n = 2 * n - 11 := sorry

theorem sum_first_20_abs_terms :
  ∃ (T₀₂₀ : ℤ), T₀₂₀ = (finset.range 21).sum abs_arithmetic_sequence ∧ T₀₂₀ = 250 := sorry

end general_term_formula_sum_first_20_abs_terms_l572_572220


namespace x_intercept_is_3_l572_572045

-- Define the given points
def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (6, 6)

-- Prove the x-intercept is 3
theorem x_intercept_is_3 (x : ℝ) :
  (∃ m b : ℝ, (∀ x1 y1 x2 y2 : ℝ, (y1 = m * x1 + b) ∧ (x1, y1) = point1 ∧ (x2, y2) = point2) ∧ y = 0 ∧ x = -b / m) → x = 3 :=
sorry

end x_intercept_is_3_l572_572045


namespace find_x_plus_y_l572_572897

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c p : V) (x y : ℝ)

-- Conditions: Definitions as the given problem requires
-- Basis definitions
def basis1 := [a, b, c]
def basis2 := [a + b, a - b, c]

-- Conditions on p
def condition1 : p = 3 • a + b + c := sorry
def condition2 : p = x • (a + b) + y • (a - b) + c := sorry

-- The proof statement
theorem find_x_plus_y (h1 : p = 3 • a + b + c) (h2 : p = x • (a + b) + y • (a - b) + c) :
  x + y = 3 :=
sorry

end find_x_plus_y_l572_572897


namespace total_practice_hours_l572_572364

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l572_572364


namespace find_two_numbers_l572_572432

noncomputable def two_positive_real_numbers_are (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ real.sqrt (a * b) = real.sqrt 5 ∧ 2 / ((1 / a) + (1 / b)) = 5 / 3

theorem find_two_numbers (a b : ℝ) (h : two_positive_real_numbers_are a b) :
  (a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1) :=
sorry

end find_two_numbers_l572_572432


namespace general_formula_for_sequence_l572_572610

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ ∀ n, a (n + 1) = 2 * a n + 2

theorem general_formula_for_sequence :
  ∃ a : ℕ → ℤ, 
    sequence a ∧ ∀ n, a n = 2 ^ (n - 1) - 2 :=
by
  sorry

end general_formula_for_sequence_l572_572610


namespace right_triangle_perimeter_l572_572939

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572939


namespace largest_unique_pair_sum_subset_l572_572873

-- Define the finite set and conditions
def set_1_to_9 : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a function to check if all pair sums in a subset are unique
def unique_pair_sums (S : Set ℕ) : Prop :=
  ∀ x y z w, x ∈ S → y ∈ S → z ∈ S → w ∈ S → x ≠ y → z ≠ w → x + y ≠ z + w

-- State the theorem
theorem largest_unique_pair_sum_subset :
  ∀ S ⊆ set_1_to_9, unique_pair_sums S → S.card ≤ 5 :=
by
  sorry -- Proof goes here

end largest_unique_pair_sum_subset_l572_572873


namespace num_non_divisible_by_3_divisors_l572_572669

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572669


namespace sum_log_diff_correct_l572_572530

noncomputable def sum_log_diff : ℕ :=
  ∑ k in finset.range 1000 + 1, k * (⌈real.log k / real.log (real.sqrt 2)⌉ - ⌊real.log k / real.log (real.sqrt 2)⌋)

theorem sum_log_diff_correct : sum_log_diff = 499477 := 
sorry

end sum_log_diff_correct_l572_572530


namespace right_triangle_perimeter_l572_572941

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572941


namespace max_sigma_for_qualified_rate_l572_572903

-- Define the conditions
variable (σ : ℝ)
variable (μ : ℝ := 50)
variable (qualified : set ℝ := Icc 47 53)

-- Given reference data and distribution properties
def quality_indicator : Prop :=
  ∃ (X : ℝ → Prop), ∀ x, X x ↔ x ∈ normal_space (μ, σ^2)

theorem max_sigma_for_qualified_rate :
  quality_indicator σ →
  ∀ X, normal_space (μ, σ^2).prob (Icc (μ - 3*σ) (μ + 3*σ)) ≈ 0.9974 →
  (∀ x, x ∈ normal_space (μ, σ^2) → qualified x) →
  σ ≤ 1 :=
by
  intros h_quality_indicator h_prob h_qualified
  -- proof will be filled in here
  sorry

end max_sigma_for_qualified_rate_l572_572903


namespace ratio_distances_circumcenter_cos_l572_572288

variable {A B C : Real} (R : Real)
variable {a b c m n p : Real}

-- Assume χtriangle ABC is an acute triangle with sides a, b, c
axiom acute_triangle (h : a^2 + b^2 + c^2 - 2 * (a * b * cos C + b * c * cos A + c * a * cos B) > 0): true

-- Define distances from the circumcenter to the sides a, b, c as m, n, p respectively
def distances_from_circumcenter (R : Real) :=
  (m = R * cos A) ∧ (n = R * cos B) ∧ (p = R * cos C)

-- The statement to prove
theorem ratio_distances_circumcenter_cos (h : acute_triangle _) :
  distances_from_circumcenter R →  (m / R) : (n / R) : (p / R) = cos A : cos B : cos C := 
sorry

end ratio_distances_circumcenter_cos_l572_572288


namespace tan_product_l572_572083

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572083


namespace div_count_of_n_cubed_l572_572475

theorem div_count_of_n_cubed {n : ℕ} (h : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ ∃ p : ℕ, p.prime ∧ n = p^2) :
  (finset.range (6+1)).filter (λ d, d ∣ n^3).card = 7 := 
sorry

end div_count_of_n_cubed_l572_572475


namespace cone_lateral_surface_area_l572_572624

theorem cone_lateral_surface_area
  (r h : ℝ)
  (hr : r = 2)
  (hh : h = 4 * Real.sqrt 2) :
  (let l := Real.sqrt (r^2 + h^2) in
   let S := Real.pi * r * l in
   S = 12 * Real.pi) :=
by
  sorry

end cone_lateral_surface_area_l572_572624


namespace tan_product_equals_three_l572_572163

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572163


namespace tan_product_l572_572140

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572140


namespace tubs_of_ice_cream_guests_ate_l572_572427

def pans : Nat := 2
def pieces_per_pan : Nat := 16
def eaten_percentage : Float := 0.75
def scoops_per_tub : Nat := 8
def guests_not_eating : Nat := 4
def scoops_per_guest : Nat := 2

theorem tubs_of_ice_cream_guests_ate :
  let total_pieces := pans * pieces_per_pan
  let eaten_pieces := pieces_per_pan + Nat.floor (eaten_percentage * pieces_per_pan.toReal)
  let guests_ala_mode := eaten_pieces - guests_not_eating
  let total_scoops_eaten := guests_ala_mode * scoops_per_guest
  let tubs_ice_cream := total_scoops_eaten / scoops_per_tub
  tubs_ice_cream = 6 := by
  sorry

end tubs_of_ice_cream_guests_ate_l572_572427


namespace tan_product_l572_572135

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572135


namespace g_200_x_eq_0_l572_572766

noncomputable def g (n : ℕ) : ℤ → ℤ
| 0     := λ x, x + abs (x - 150) - abs (x + 150)
| (n+1) := λ x, abs (g n x) - 2

def question := ∀ n, n = 200 → ∃! x, g n x = 0

theorem g_200_x_eq_0 : question :=
sorry

end g_200_x_eq_0_l572_572766


namespace measure_angle_XTZ_l572_572303

-- Definition of a triangle with angle measure properties

structure Triangle :=
  (X Y Z: Type)
  [inner_product_space ℝ X]
  [inner_product_space ℝ Y]
  [inner_product_space ℝ Z]
  (angle_XYZ : ℝ)
  (bisection_XT_XZ : ℝ)
  (bisection_TZ_XYZ : ℝ)

noncomputable def measure_angle_XYZ (T: Triangle) : Prop :=
  T.angle_XYZ = 60

noncomputable def bisection_XT_XZ (T: Triangle) : Prop :=
  T.bisection_XT_XZ = T.angle_XYZ / 2

noncomputable def bisection_TZ_XYZ (T: Triangle) : Prop :=
  T.bisection_TZ_XYZ = T.angle_XYZ / 2

-- Proof to find angle XTZ
theorem measure_angle_XTZ (T : Triangle) (h1: measure_angle_XYZ T) (h2: bisection_XT_XZ T) (h3: bisection_TZ_XYZ T) : 
  (T.angle_XYZ / 2 + T.angle_XYZ / 2 + 30) = 90 :=
by {
  sorry
}

end measure_angle_XTZ_l572_572303


namespace dice_probability_five_or_six_l572_572884

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end dice_probability_five_or_six_l572_572884


namespace arthur_total_dinner_cost_l572_572853

-- Definitions
def cost_of_appetizer := 8
def cost_of_ribeye := 20
def cost_of_wine_glass := 3
def number_of_wine_glasses := 2
def cost_of_cheesecake := 6
def discount_voucher_percent := 0.5
def tip_percent := 0.2

-- Calculations
def total_cost_before_discount := 
  cost_of_appetizer + cost_of_ribeye + (cost_of_wine_glass * number_of_wine_glasses) + cost_of_cheesecake

def discount_amount := cost_of_ribeye * discount_voucher_percent
def tip_amount := total_cost_before_discount * tip_percent

def total_cost_after_discount_and_tip := 
  total_cost_before_discount - discount_amount + tip_amount

-- Statement to prove
theorem arthur_total_dinner_cost :
  total_cost_after_discount_and_tip = 38 := 
by
  sorry

end arthur_total_dinner_cost_l572_572853


namespace triangle_area_l572_572996

noncomputable def area_triangle_ACD (t p : ℝ) : ℝ :=
  1 / 2 * p * (t - 2)

theorem triangle_area (t p : ℝ) (ht : 0 < t ∧ t < 12) (hp : 0 < p ∧ p < 12) :
  area_triangle_ACD t p = 1 / 2 * p * (t - 2) :=
sorry

end triangle_area_l572_572996


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572691

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572691


namespace minimum_AP_BP_l572_572329

noncomputable def point := (ℝ × ℝ)

def A : point := (2, 0)
def B : point := (7, 6)

-- Define the parabola y^2 = 8*x in Lean
def on_parabola (P : point) : Prop := P.2 ^ 2 = 8 * P.1

def distance (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def AP (P : point) : ℝ := distance A P
def BP (P : point) : ℝ := distance B P

-- Define the problem statement
theorem minimum_AP_BP :
  ∃ P : point, on_parabola P ∧ (AP P + BP P) = 3 * real.sqrt 13 :=
sorry

end minimum_AP_BP_l572_572329


namespace solution_for_a_l572_572722

theorem solution_for_a :
  ∀ a x : ℝ, (2 - a - x = 0) ∧ (2x + 1 = 3) → a = 1 := 
by
  intros a x h,
  cases h with h1 h2,
  have x_eq := by linarith,
  have a_eq := by linarith,
  exact a_eq

end solution_for_a_l572_572722


namespace binomial_square_eq_l572_572565

theorem binomial_square_eq (a : ℚ) :
  (∃ r s : ℚ, ax^2 + 15x + 4 = (r * x + s) ^ 2 ∧ s ^ 2 = 4 ∧ 2 * r * s = 15)
  ↔ a = 225 / 16 := 
sorry

end binomial_square_eq_l572_572565


namespace correct_option_is_B_l572_572001

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l572_572001


namespace range_of_a_plus_c_l572_572243

noncomputable def geometric_sum_range (a b c : ℝ) : Set ℝ :=
  {x | ∃ r : ℝ, a = r ∧ b = r^2 ∧ c = r^3 ∧ a + b + c = 1 ∧ x = a + c}

theorem range_of_a_plus_c (a b c : ℝ) :
  (a, b, c).1 = a ∧ (a, b, c).2.1 = b ∧ (a, b, c).2.2 = c ∧
  (∀ r : ℝ, a = r ∧ b = r^2 ∧ c = r^3 → a + b + c = 1) →
  geometric_sum_range a b c = [0, ∞) :=
sorry

end range_of_a_plus_c_l572_572243


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572689

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572689


namespace minimize_white_surface_area_l572_572910

structure Cube :=
  (edge_length : ℝ)
  (small_cubes : ℕ)
  (red_cubes : ℕ)
  (white_cubes : ℕ)

def large_cube : Cube := {
  edge_length := 4,
  small_cubes := 64,
  red_cubes := 48,
  white_cubes := 16
}

theorem minimize_white_surface_area (c : Cube) (minimize_exposure : Prop) :
  minimize_exposure → (16 / 96 = 5 / 16) := 
by
  assume minimize_exposure,
  unfold minimize_exposure,
  sorry

end minimize_white_surface_area_l572_572910


namespace arithmetic_sequence_general_term_l572_572335

noncomputable def an (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d
def bn (a_n : ℤ) : ℚ := (1 / 2)^a_n

theorem arithmetic_sequence_general_term
  (a_n : ℕ → ℤ)
  (b_1 b_2 b_3 : ℚ)
  (a_1 d : ℤ)
  (h_seq : ∀ n, a_n n = a_1 + (n - 1) * d)
  (h_b1 : b_1 = (1 / 2)^(a_n 1))
  (h_b2 : b_2 = (1 / 2)^(a_n 2))
  (h_b3 : b_3 = (1 / 2)^(a_n 3))
  (h_sum : b_1 + b_2 + b_3 = 21 / 8)
  (h_prod : b_1 * b_2 * b_3 = 1 / 8)
  : (∀ n, a_n n = 2 * n - 3) ∨ (∀ n, a_n n = 5 - 2 * n) :=
sorry

end arithmetic_sequence_general_term_l572_572335


namespace units_digit_base7_add_mul_l572_572582

theorem units_digit_base7_add_mul :
  let x := 43 : ℕ
  let y := 124 : ℕ
  let z := 15 : ℕ
  let x_base7 := (4 * 7 + 3)
  let y_base7 := (1 * 49 + 2 * 7 + 4)
  let z_base7 := (1 * 7 + 5)
  let sum := x_base7 + y_base7
  let prod := sum * z_base7
  nat.mod prod 7 = 6 := 
sorry

end units_digit_base7_add_mul_l572_572582


namespace x_minus_q_in_terms_of_q_l572_572709

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h : |x - 3| = q) (hx : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_in_terms_of_q_l572_572709


namespace circle_through_A_B_C_has_area_9π_l572_572232

noncomputable def circle_area (l1 l2 : Line) (A B C : Point) : ℝ :=
  if h1 : perpendicular l1 l2 ∧ (B ∈ l1) ∧ (C ∈ l2) 
     ∧ (B ≠ A) ∧ (C ≠ A)
     ∧ (dist B C = 3) then 
     pi * 3^2
  else 
    0

theorem circle_through_A_B_C_has_area_9π 
  (l1 l2 : Line) (A B C : Point) 
  (h : perpendicular l1 l2 ∧ (B ∈ l1) ∧ (C ∈ l2) 
      ∧ (B ≠ A) ∧ (C ≠ A)
      ∧ (dist B C = 3)) : 
  circle_area l1 l2 A B C = 9 * pi :=
by
  sorry

end circle_through_A_B_C_has_area_9π_l572_572232


namespace doughnut_machine_completion_time_l572_572462

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end doughnut_machine_completion_time_l572_572462


namespace triangle_perimeter_PQR_l572_572064

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def triangle_perimeter (P Q R : ℝ × ℝ) : ℝ :=
  distance P Q + distance Q R + distance R P

def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (2, 10)
def R : ℝ × ℝ := (8, 6)

theorem triangle_perimeter_PQR :
  triangle_perimeter P Q R = 7 + 2 * real.sqrt 13 + 3 * real.sqrt 5 :=
by {
  unfold triangle_perimeter distance P Q R,
  -- we can add intermediate steps here if needed, or directly add sorry
  sorry,
}

end triangle_perimeter_PQR_l572_572064


namespace rate_per_square_meter_l572_572835

theorem rate_per_square_meter 
  (L : ℝ) (W : ℝ) (C : ℝ)
  (hL : L = 5.5) 
  (hW : W = 3.75)
  (hC : C = 20625)
  : C / (L * W) = 1000 :=
by
  sorry

end rate_per_square_meter_l572_572835


namespace measure_of_angle_B_l572_572306

theorem measure_of_angle_B 
  (A B C: ℝ)
  (a b c: ℝ)
  (h1: A + B + C = π)
  (h2: B / A = C / B)
  (h3: b^2 - a^2 = a * c) : B = 2 * π / 7 :=
  sorry

end measure_of_angle_B_l572_572306


namespace smallest_prime_factor_in_C_l572_572810

def smallest_prime_factor_def (n : Nat) : Nat :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  sorry /- Define a function to find the smallest prime factor of a number n -/

def is_prime (p : Nat) : Prop :=
  2 ≤ p ∧ ∀ d : Nat, 2 ≤ d → d ∣ p → d = p

def in_set (x : Nat) : Prop :=
  x = 64 ∨ x = 66 ∨ x = 67 ∨ x = 68 ∨ x = 71

theorem smallest_prime_factor_in_C : ∀ x, in_set x → 
  (smallest_prime_factor_def x = 2 ∨ smallest_prime_factor_def x = 67 ∨ smallest_prime_factor_def x = 71) :=
by
  intro x hx
  cases hx with
  | inl hx  => sorry
  | inr hx  => sorry

end smallest_prime_factor_in_C_l572_572810


namespace solve_problem_l572_572353

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) :
  f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f (f y))^2)

theorem solve_problem (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end solve_problem_l572_572353


namespace sum_of_digits_0_to_2012_l572_572314

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l572_572314


namespace area_EFGH_l572_572383

-- Definition: Rhombus Quadrilateral
structure Rhombus (A B C D : Type) :=
  (AB : (A → B) → Prop)
  (BC : (B → C) → Prop)
  (CD : (C → D) → Prop)
  (DA : (D → A) → Prop)
  (side_length : ℝ)
  (height : ℝ)
  (is_rhombus : AB ↔ BC ∧ BC ↔ CD ∧ CD ↔ DA)

-- Points E and G
def E_midpoint_AD (A D E : Type) := 
  ∀(x : A → D), (A → E) ∧ (E → D)

def G_midpoint_BC (B C G : Type) := 
  ∀(x : B → C), (B → G) ∧ (G → C)

-- Intersection Points F and H
def F_intersection_AG_BE (A B E G : Type) := 
  ∃(F : Type), (A → G) ∧ (B → E) → (F)

def H_intersection_CE_DG (C D E G : Type) := 
  ∃(H : Type), (C → E) ∧ (D → G) → (H)

-- Proof statement
theorem area_EFGH (A B C D E F G H : Type) [Rhombus A B C D] [E_midpoint_AD A D E] [G_midpoint_BC B C G]
  [F_intersection_AG_BE A B E G] [H_intersection_CE_DG C D E G] : 
  let area_abcd := 24 in
  ∃(area_efgh : (ℝ)), area_efgh = 6 := by {
  -- The proof would go here
  sorry
}

end area_EFGH_l572_572383


namespace remainder_5310_mod8_l572_572411

theorem remainder_5310_mod8 : (53 ^ 10) % 8 = 1 := 
by 
  sorry

end remainder_5310_mod8_l572_572411


namespace units_digit_13_pow_2003_l572_572442

theorem units_digit_13_pow_2003 : (13 ^ 2003) % 10 = 7 := by
  sorry

end units_digit_13_pow_2003_l572_572442


namespace shaded_area_l572_572825

noncomputable def squareArea (a : ℝ) : ℝ := a * a

theorem shaded_area {s : ℝ} (h1 : squareArea s = 1) (h2 : s / s = 2) : 
  ∃ (shaded : ℝ), shaded = 1 / 3 :=
by
  sorry

end shaded_area_l572_572825


namespace number_of_ways_to_assign_guests_l572_572039

theorem number_of_ways_to_assign_guests (friends rooms : ℕ) (h_friends : friends = 5) (h_rooms : rooms = 5) 
  (h_max_per_room : ∀ r, r ∈ finset.range rooms → r ≤ 2) : 
  (number_of_assignments friends rooms h_max_per_room) = 1620 := sorry

noncomputable def number_of_assignments : ℕ → ℕ → (∀ r : ℕ, r < 5 → r ≤ 2) → ℕ
| 5, 5, h_max_per_room := 120 + 600 + 900
| _, _, _ := 0

end number_of_ways_to_assign_guests_l572_572039


namespace determine_liar_fraction_l572_572486

-- Defining the problem conditions
structure Village (n : ℕ) :=
  (inhabitants : Fin n → Prop) -- Each inhabitant is either a liar or a truth-teller
  (truthful : Fin n → Prop) -- Predicate to check if an inhabitant is truthful
  (lies : Fin n → Prop) -- Predicate to check if an inhabitant is a liar
  (neighbor_right : Fin n → Fin n) -- Each inhabitant's right neighbor
  (reports : ∀ i : Fin n, truthful i ↔ lies (neighbor_right i)) -- Reports about the neighbor to their right

noncomputable def liar_fraction (v : Village n) : ℝ :=
  (Finset.filter (λ i, v.lies i) (Finset.univ : Finset (Fin n))).card / n

-- Main theorem to prove the proportion of liars is 0.5
theorem determine_liar_fraction (n : ℕ) (v : Village n) 
  (h₀ : liar_fraction v = (1 - liar_fraction v)) : liar_fraction v = 0.5 :=
by
  calc liar_fraction v + liar_fraction v = 1 : by { sorry }
     2 * liar_fraction v = 1 : by { sorry }
     liar_fraction v = 0.5 : by { sorry }

end determine_liar_fraction_l572_572486


namespace geom_seq_fraction_l572_572739

theorem geom_seq_fraction (a : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = q * a n)
  (h_sum1 : a 1 + a 2 = 1)
  (h_sum4 : a 4 + a 5 = -8) :
  (a 7 + a 8) / (a 5 + a 6) = -4 :=
sorry

end geom_seq_fraction_l572_572739


namespace cube_largest_ne_sum_others_l572_572381

theorem cube_largest_ne_sum_others (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 :=
by
  sorry

end cube_largest_ne_sum_others_l572_572381


namespace tangent_product_eq_three_l572_572120

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572120


namespace ratio_of_distances_l572_572007

-- Define the given conditions
variables (w x y : ℕ)
variables (h1 : w > 0) -- walking speed must be positive
variables (h2 : x > 0) -- distance from home must be positive
variables (h3 : y > 0) -- distance to stadium must be positive

-- Define the two times:
-- Time taken to walk directly to the stadium
def time_walk (w y : ℕ) := y / w

-- Time taken to walk home, then bike to the stadium
def time_walk_bike (w x y : ℕ) := x / w + (x + y) / (5 * w)

-- Given that both times are equal
def times_equal (w x y : ℕ) := time_walk w y = time_walk_bike w x y

-- We want to prove that the ratio of x to y is 2/3
theorem ratio_of_distances (w x y : ℕ) (h_time_eq : times_equal w x y) : x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l572_572007


namespace intersection_A_B_l572_572784

open Set

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k - 1 }
def B : Set ℕ := { x | ∃ k : ℕ, x = 2 * k + 1 ∧ k < 3 }

theorem intersection_A_B : A ∩ B = {1, 3, 5} := by
  sorry

end intersection_A_B_l572_572784


namespace tangent_product_eq_three_l572_572116

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572116


namespace vector_angle_acute_l572_572264

definition vector (α β : Type) := (α × β)

variables {λ : ℝ}

def vector_a : vector ℝ ℝ := (1, -2)
def vector_b (λ : ℝ) : vector ℝ ℝ := (3, λ)

def dot_product : vector ℝ ℝ -> vector ℝ ℝ -> ℝ
| (x1, y1) (x2, y2) := x1 * x2 + y1 * y2

def acute_condition (v1 v2 : vector ℝ ℝ) : Prop :=
  dot_product v1 v2 > 0

def not_collinear (a b : ℝ) : Prop :=
  b ≠ - (a / 2)

noncomputable def range_of_lambda (λ : ℝ) : Prop :=
  λ < 3 / 2 ∧ λ ≠ -6

theorem vector_angle_acute (λ : ℝ) :
  acute_condition vector_a (vector_b λ) ∧ not_collinear 1 (-2) ↔ range_of_lambda λ :=
by sorry

end vector_angle_acute_l572_572264


namespace grid_divisible_by_L_shapes_l572_572282

theorem grid_divisible_by_L_shapes (n : ℕ) (h : n ≥ 1) : 
  ∀ (r c : ℕ), r = 2^n ∧ c = 2^n → 
  ∃ (cells : ℕ → ℕ × ℕ), 
    (∀ i, i < (r * c - 1) / 3 → 
    is_L_shape (cells i)) ∧
    (∀ i₁ i₂, i₁ ≠ i₂ → 
    disjoint (cells i₁) (cells i₂)) := 
sorry

end grid_divisible_by_L_shapes_l572_572282


namespace total_members_in_sports_club_l572_572287

-- Definitions as per the conditions
def B : ℕ := 20 -- number of members who play badminton
def T : ℕ := 23 -- number of members who play tennis
def Both : ℕ := 7 -- number of members who play both badminton and tennis
def Neither : ℕ := 6 -- number of members who do not play either sport

-- Theorem statement to prove the correct answer
theorem total_members_in_sports_club : B + T - Both + Neither = 42 :=
by
  sorry

end total_members_in_sports_club_l572_572287


namespace tan_identity_proof_l572_572096

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572096


namespace tangent_product_eq_three_l572_572111

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572111


namespace wendy_score_each_treasure_l572_572868

-- Define the conditions
variable (treasuresFound : ℕ) (totalScore : ℕ) (pointsPerTreasure : ℕ)
variable h1 : treasuresFound = 4 + 3
variable h2 : totalScore = 35

-- State the theorem to be proved
theorem wendy_score_each_treasure (h1 : treasuresFound = 4 + 3) (h2 : totalScore = 35) : 
  7 * pointsPerTreasure = totalScore → pointsPerTreasure = 5 := 
by
  intro h
  have ht : treasuresFound = 7 := by rw h1
  have hs : 7 * pointsPerTreasure = 35 := by rw [←ht, h2]
  sorry

end wendy_score_each_treasure_l572_572868


namespace num_divisors_not_divisible_by_three_l572_572666

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572666


namespace solve_for_p_l572_572365

theorem solve_for_p (g t : ℂ) (p : ℂ) (h1 : g = 3) (h2 : t = 3 + 75 * complex.I) (h3 : 3 * g * p - t = 9000) :
  p = 1000 + (1 / 3) + 8 * complex.I + (1 / 3) * complex.I :=
by
  have h4 : 3 * 3 * p - (3 + 75 * complex.I) = 9000 := by rwa [h1, h2] at h3
  have h5 : 9 * p - 3 - 75 * complex.I = 9000 := by rwa [mul_assoc] at h4
  have h6 : 9 * p = 9003 + 75 * complex.I := by linarith
  have h7 : p = (9003 + 75 * complex.I) / 9 := by field_simp[←h6]
  have h8 : p = 1000 + 1/3 + 8 * complex.I + 1/3 * complex.I := by
    sorry
  rwa h8

end solve_for_p_l572_572365


namespace tan_product_identity_l572_572106

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572106


namespace tan_product_l572_572141

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572141


namespace remaining_tickets_divisible_by_1001_l572_572793

theorem remaining_tickets_divisible_by_1001 : 
  (∀ num : Fin 1000000, 
   num.sum_dig_conds(λ (a b c d e f : Fin 10), a * f + b * e + c * d ≠ 100) → 
   sum_over_range (λ (k : Fin 1000000), k) % 1001 = 0) :=
sorry

end remaining_tickets_divisible_by_1001_l572_572793


namespace sum_of_progression_l572_572337

theorem sum_of_progression :
  ∀ (d_c d_d : ℝ)
    (c_n d_n : ℕ → ℝ),
    (c_n 1 = 10) →
    (d_n 1 = 90) →
    (c_n 50 + d_n 50 = 500) →
    (∀ n, c_n n = 10 + (n - 1) * d_c) →
    (∀ n, d_n n = 90 + (n - 1) * d_d) →
    (∑ n in finset.range 50, c_n (n+1) + d_n (n+1) = 15000) 
:=
by {
  intros d_c d_d c_n d_n hc1 hd1 hc50 hc_seq_hd_seq hc_seq hd_seq,
  sorry
}

end sum_of_progression_l572_572337


namespace count_divisors_not_divisible_by_3_l572_572698

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572698


namespace range_of_m_monotonic_range_of_m_above_line_l572_572638

-- Definitions for the function and the intervals
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - (m - 1) * x + m - 7

def is_monotonic_on [OrderClosedTopology ℝ] (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
    (∀ x y ∈ I, x < y → f x ≤ f y) ∨ (∀ x y ∈ I, x < y → f x ≥ f y)

-- Problem statement (part 1)
theorem range_of_m_monotonic (m : ℝ) :
  is_monotonic_on (g _ m) {x | 2 ≤ x ∧ x ≤ 4} → (m ≤ 5 ∨ m ≥ 9) :=
by
  sorry

-- Definitions for the interval and condition for second part
def h (x : ℝ) (m : ℝ) : ℝ := x^2 - (m + 1) * x + m + 2

-- Problem statement (part 2)
theorem range_of_m_above_line (m : ℝ) :
  (∀ x ∈ Set.Icc (-1:ℝ) 1, g x m > 2 * x - 9) → (1 - 2 * Real.sqrt 2 < m) :=
by
  sorry

end range_of_m_monotonic_range_of_m_above_line_l572_572638


namespace incongruent_triangles_count_l572_572653

def num_of_incongruent_triangles (s : Set (ℕ × ℕ × ℕ)) : ℕ :=
  s.count (λ ⟨a, b, c⟩ => 0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b + c = 1994 ∧ a + b > c)

theorem incongruent_triangles_count : num_of_incongruent_triangles {t : ℕ × ℕ × ℕ | true} = 82834 :=
by
  sorry

end incongruent_triangles_count_l572_572653


namespace dakotas_medical_bill_l572_572543

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end dakotas_medical_bill_l572_572543


namespace always_even_l572_572235

theorem always_even (a b c : ℤ) (h : (∃ x y : ℤ, (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0 ∨ 
                                         (a % 2 ≠ 0 ∧ a = x) ∨ (b % 2 ≠ 0 ∧ b = y) ∨ (c % 2 ≠ 0 ∧ c = x) ∨ 
                                         (c % 2 ≠ 0 ∧ c = y) ∨ (b % 2 ≠ 0 ∧ b = x) ∨ (a % 2 ≠ 0 ∧ a = y))) :
  even ((a + 1) * (b + 2) * (c + 3)) :=
sorry

end always_even_l572_572235


namespace part1_solution_part2_solution_l572_572460

-- Part (1)
theorem part1_solution (m n : ℕ) (h : m < n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ m → x i > 0) ∧ (∑ i in finRange m, x i = n) :=
  sorry

-- Part (2)
theorem part2_solution (m n r : ℤ) (h1 : 0 ≤ n) (h2 : n ≥ (m - 2) * r + 1) :
  ∃ (x : ℕ → ℤ), 
    (x 1 ≥ 1) ∧ 
    (∀ i, 2 ≤ i ∧ i ≤ m - 1 → x i ≥ r) ∧ 
    (x m ≥ 0) ∧ 
    (∑ i in finRange m, x i = n) :=
  sorry

end part1_solution_part2_solution_l572_572460


namespace number_of_moles_of_CaCl2_l572_572199

-- Define the conditions
variables (HCl CaCO3 CaCl2 : ℕ)

-- Initial amounts: 4 moles of HCl, 2 moles of CaCO3
def initial_HCl := 4
def initial_CaCO3 := 2

-- The balanced reaction indicates 1 mole of CaCO3 reacts with 2 moles HCl to produce 1 mole of CaCl2.
-- Let's define the function to determine the amount of CaCl2 produced based on the reaction stoichiometry.
def calculate_moles_CaCl2 (moles_HCl : ℕ) (moles_CaCO3 : ℕ) : ℕ :=
    if moles_HCl >= 2 * moles_CaCO3 then moles_CaCO3 else moles_HCl / 2

-- The theorem that encapsulates the problem statement:
theorem number_of_moles_of_CaCl2 :
  calculate_moles_CaCl2 initial_HCl initial_CaCO3 = 2 :=
by {
  -- We can compute it directly
  unfold initial_HCl initial_CaCO3 calculate_moles_CaCl2,
  simp,
  sorry
}

end number_of_moles_of_CaCl2_l572_572199


namespace max_term_of_sequence_l572_572991

noncomputable def a_n (n : ℕ) : ℚ := (n^2 : ℚ) / (2^n : ℚ)

theorem max_term_of_sequence :
  ∃ n : ℕ, (∀ m : ℕ, a_n n ≥ a_n m) ∧ a_n n = 9 / 8 :=
sorry

end max_term_of_sequence_l572_572991


namespace homework_checked_on_friday_l572_572434

-- Define the events
def event_no_homework_checked : Prop := ∀ (d : ℕ), d ∈ {0, 1, 2, 3, 4} → ¬ checked d
def event_homework_checked_friday : Prop := checked 4
def event_homework_not_checked_until_thursday : Prop := ∀ (d : ℕ), d ∈ {0, 1, 2, 3} → ¬ checked d

-- Define the probabilities
def prob_teacher_checks_homework : ℙ := 1 / 2
def prob_teacher_does_not_check_homework : ℙ := 1 / 2
def prob_day_check (d : ℕ) : ℙ := 1 / 5

-- Main statement
theorem homework_checked_on_friday : 
  ℙ (event_homework_checked_friday | event_homework_not_checked_until_thursday) = 1 / 6 := 
begin
  sorry -- Proof to be provided
end

end homework_checked_on_friday_l572_572434


namespace acute_angle_solution_l572_572615

theorem acute_angle_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (π / 2))
    (h3 : cos (2 * α) = sin α) : α = π / 6 := 
sorry

end acute_angle_solution_l572_572615


namespace count_divisors_not_divisible_by_3_l572_572699

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572699


namespace correct_propositions_l572_572964

noncomputable def proposition_1_domain (x : ℝ) : Prop := x > 0
noncomputable def proposition_2_eqn (x : ℝ) : Prop := log 10 (sqrt (x-2)) = log 10 (x-2)
noncomputable def proposition_3_eqn (x : ℝ) : Prop := 3^(1-x) - 2 = 0
noncomputable def proposition_4_ineq (x : ℝ) : Prop := log 10 (x-1) < 1

theorem correct_propositions : 
  ¬proposition_1_domain 0 ∧
  (∀ x : ℝ, proposition_2_eqn x → x = 3) ∧
  (∀ x : ℝ, proposition_3_eqn x → x = 1 - log 10 (2)/log 10 (3)) ∧
  ¬(∀ x : ℝ, proposition_4_ineq x → x < 11) :=
sorry

end correct_propositions_l572_572964


namespace reciprocal_neg_3_div_4_l572_572840

theorem reciprocal_neg_3_div_4 : (- (3 / 4 : ℚ))⁻¹ = -(4 / 3 : ℚ) :=
by
  sorry

end reciprocal_neg_3_div_4_l572_572840


namespace technology_elective_courses_l572_572294

theorem technology_elective_courses (m : ℕ) :
  let subject_elective := m,
      arts_elective := m + 9,
      technology_elective := 1 / 3 * arts_elective + 5
  in technology_elective = 1 / 3 * m + 8 :=
by
  sorry

end technology_elective_courses_l572_572294


namespace tan_identity_proof_l572_572094

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572094


namespace ratio_BD_DF_l572_572374

noncomputable def triangle {α : Type*} [euclidean_geometry α] (A B C D E F : α) : Prop :=
ang_eq B A C 120 ∧ 
dist A D = dist A B ∧ 
dist C E = dist C B ∧ 
right_angle D F B ∧ 
collinear B E F

theorem ratio_BD_DF {α : Type*} [euclidean_geometry α] (A B C D E F : α) 
  (h : triangle A B C D E F) : dist B D / dist D F = 2 :=
sorry

end ratio_BD_DF_l572_572374


namespace speed_of_other_person_l572_572808

theorem speed_of_other_person
    (roja_speed : ℕ)
    (other_person_speed : ℕ)
    (time : ℕ)
    (distance : ℕ) :
    roja_speed = 2 →
    other_person_speed = 3 →
    time = 4 →
    distance = 20 →
    (roja_speed + other_person_speed) * time = distance :=
by
    intros h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    exact rfl

end speed_of_other_person_l572_572808


namespace find_common_divisor_same_remainder_l572_572198

theorem find_common_divisor_same_remainder :
  let a := 480608
  let b := 508811
  let c := 723217
  let d1 := b - a -- 28203
  let d2 := c - b -- 214406
  let d3 := c - a -- 242609
  Int.gcd (Int.gcd d1 d2) d3 = 79 :=
by
  sorry

end find_common_divisor_same_remainder_l572_572198


namespace tan_product_l572_572089

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572089


namespace sum_of_extreme_values_of_g_is_seven_l572_572338

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + x

theorem sum_of_extreme_values_of_g_is_seven :
  let largest_value := 8
  let smallest_value := -1
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 10 → 
  g(x) = 3x - 4 ∨ g(x) = 2
  → largest_value + smallest_value = 7 :=
by
  sorry

end sum_of_extreme_values_of_g_is_seven_l572_572338


namespace line_intersects_curve_l572_572247

-- Definitions and Conditions
def polar_curve₁ : ℝ → ℝ := λ θ, 2 * cos θ - 4 * sin θ
def parametric_line₁ (t : ℝ) : ℝ × ℝ := (1 + t, -1 + t)

-- To Prove: Intersection and Distance
theorem line_intersects_curve : 
  ∃ A B : ℝ × ℝ, 
    A = (2 : ℝ, 0 : ℝ) ∧ B = (-1 : ℝ, -3 : ℝ) ∧
    dist A B = 3 * sqrt 2 := by
  sorry

end line_intersects_curve_l572_572247


namespace limit_f_2_l572_572255

open Real

def f (x : ℝ) : ℝ := x^2 + x

theorem limit_f_2 (h : ∀ Δx : ℝ, Δx ≠ 0 → Δx = 0 → True) :
    tendsto (λ Δx, (f (2 + Δx) - f 2)) (𝓝 0) (𝓝 (5 * 0 + 5)) := 
by
  have h_lim : tendsto (λ Δx, (f (2 + Δx) - f 2) / Δx) (𝓝 0) (𝓝 5),
  calc
    ∀ Δx : ℝ, Δx ≠ 0 → Δx = 0 → True 
  sorry

#check @limit_f_2

end limit_f_2_l572_572255


namespace number_of_pictures_deleted_l572_572445

-- Definitions based on the conditions
def total_files_deleted : ℕ := 17
def songs_deleted : ℕ := 8
def text_files_deleted : ℕ := 7

-- The question rewritten as a Lean theorem statement
theorem number_of_pictures_deleted : 
  (total_files_deleted - songs_deleted - text_files_deleted) = 2 := 
by
  sorry

end number_of_pictures_deleted_l572_572445


namespace shortest_path_paving_grid_l572_572483

theorem shortest_path_paving_grid (b : ℝ) : 
  let d := 28 * b in
  ∃ A, d = 28 * b :=
by
  sorry

end shortest_path_paving_grid_l572_572483


namespace conjugate_of_Z_in_first_quadrant_l572_572999

open Complex

-- Define the complex number Z
def Z : ℂ := (2 - I) / (1 + I)

-- Define the conjugate of Z
def Z_conj : ℂ := conj Z

-- Define what it means for a complex number to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

-- Prove that the conjugate of Z is in the first quadrant
theorem conjugate_of_Z_in_first_quadrant : 
  is_in_first_quadrant Z_conj :=
sorry

end conjugate_of_Z_in_first_quadrant_l572_572999


namespace find_coordinates_c_find_angle_q_l572_572644

section vector_problems

open Real

-- Definition of vectors
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (x, y)
def c : ℝ × ℝ := (u, v)

-- Condition 1: |c| = 3 * sqrt 2
def magnitude_c := sqrt (u^2 + v^2) = 3 * sqrt 2

-- Condition 2: c is parallel to a
def c_parallel_a := u / 1 = v / -1 -- implying u = -v or v = -u

-- First problem: Coordinates of vector c
theorem find_coordinates_c (magnitude_c : magnitude_c) (c_parallel_a : c_parallel_a) :
  c = (-3, 3) ∨ c = (3, -3) :=
sorry

-- Condition 3: |b| = 1
def magnitude_b := sqrt (x^2 + y^2) = 1

-- Condition 4: a is perpendicular to (a - 2b)
def a_perpendicular_a_minus_2b := let (ax, ay) := (1, -1) in ax * (ax - 2 * x) + ay * (ay - 2 * y) = 0

-- Second problem: Angle q between a and b
theorem find_angle_q (magnitude_b : magnitude_b) (a_perpendicular_a_minus_2b : a_perpendicular_a_minus_2b) :
  ∃ q : ℝ, q = π / 4 :=
sorry

end vector_problems

end find_coordinates_c_find_angle_q_l572_572644


namespace num_divisors_not_divisible_by_three_l572_572665

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572665


namespace unique_shapes_foldable_into_open_top_cube_l572_572208

/--
There are exactly 8 different shapes, consisting of 5 connected squares,
that can be cut out from a grid paper and folded into an open-top cube,
where shapes identical up to rotations and reflections are considered the same.
-/
theorem unique_shapes_foldable_into_open_top_cube :
  ∃ S : Finset (Finset (Fin 25)),
    S.card = 8 ∧
    ∀ shape ∈ S, 
      (∀ p q ∈ shape, are_connected p q) ∧ 
      (is_foldable_into_open_top_cube shape) ∧
      (is_unique_up_to_rotation_and_reflection shape) :=
sorry

/-- A helper definition stating that two positions p and q are connected. -/
def are_connected (p q : Fin 25) : Prop :=
  -- Define connectivity logic here
sorry

/-- A helper definition stating that a shape can be folded into an open-top cube. -/
def is_foldable_into_open_top_cube (shape : Finset (Fin 25)) : Prop :=
  -- Define foldability logic here
sorry

/-- A helper definition stating that a shape is unique up to rotations and reflections. -/
def is_unique_up_to_rotation_and_reflection (shape : Finset (Fin 25)) : Prop :=
  -- Define uniqueness logic here
sorry

end unique_shapes_foldable_into_open_top_cube_l572_572208


namespace circumcircles_common_point_l572_572743

theorem circumcircles_common_point
  (ABC : Triangle)
  (w : Circle)
  (B C : Point)
  (E F : Point)
  (B' C' : Point)
  (A' : Point)
  (h1 : w.passes_through B C)
  (h2 : w∩(ABC.side AB) = E)
  (h3 : w∩(ABC.side AC) = F)
  (h4 : (ABC.side BF). intersects_around (ABC.circumcircle) = some B')
  (h5 : (ABC.side CE). intersects_around (ABC.circumcircle) = some C')
  (h6 : A' ∈ (ABC.side BC))
  (h7 : ∠C'A'B = ∠B'A'C)
  :
  ∃ D : Point, ∀ w : Circle, (A'B'C').circumcircle w ∋ D :=
sorry

end circumcircles_common_point_l572_572743


namespace sum_of_digits_0_to_2012_l572_572321

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l572_572321


namespace divisors_of_180_not_divisible_by_3_l572_572684

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572684


namespace tan_product_equals_three_l572_572170

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572170


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572690

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572690


namespace minimum_additional_coins_needed_l572_572495

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem minimum_additional_coins_needed (friends : ℕ) (current_coins : ℕ) :
  friends = 15 → current_coins = 63 → 
  let required_coins := sum_natural_numbers friends in
  let additional_coins := required_coins - current_coins in
  additional_coins = 57 :=
by
  intros h_friends h_coins
  rw [h_friends, h_coins]
  let required_coins := sum_natural_numbers 15
  have h_required_coins : required_coins = 120 := by
    rw [sum_natural_numbers, Nat.mul, Nat.add, /, 2]
    norm_num
  
  let additional_coins := required_coins - 63
  have h_additional_coins : additional_coins = 57 := by
    rw [h_required_coins]
    norm_num
    
  exact h_additional_coins

end minimum_additional_coins_needed_l572_572495


namespace max_value_of_function_l572_572196

open Real

theorem max_value_of_function :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), (λ x, x + 2 * cos x) x ≤ (π / 6 + sqrt 3) :=
sorry

end max_value_of_function_l572_572196


namespace find_a3_l572_572604

-- Define the geometric sequence properties
variables {a : ℕ → ℝ} (q : ℝ)
-- Define the conditions given in the problem
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = a n * q
def condition1 := a 1 + a 2 = 3
def condition2 := a 2 + a 3 = 6

theorem find_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a q) (h2 : condition1) (h3 : condition2) : a 3 = 4 :=
  sorry

end find_a3_l572_572604


namespace sphere_area_approx_l572_572828

noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

theorem sphere_area_approx :
  sphere_surface_area 4 ≈ 201.06 :=
by
  sorry

end sphere_area_approx_l572_572828


namespace h_div_h_l572_572451

def h (n : ℕ) : ℕ :=
  if n % 2 = 0 then (∑ i in (finset.filter (λ i, i % 2 = 0) (finset.Icc 2 n)), i) else 0

theorem h_div_h (h_even_18 : 18 % 2 = 0) (h_even_10 : 10 % 2 = 0) : h 18 / h 10 = 3 :=
  by
  have h18_def : h 18 = 90 := sorry
  have h10_def : h 10 = 30 := sorry
  exact h18_def ▸ h10_def ▸ rfl

end h_div_h_l572_572451


namespace angle_between_a_b_is_60_deg_l572_572261

noncomputable def angle_between_vectors
  (a b : ℝ^2)
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 1)
  (hab : ‖a + b‖ = Real.sqrt 7) : Real :=
let cos_alpha := (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) in
if cos_alpha = 1 / 2 then
  Real.arccos (1 / 2) * 180 / Real.pi
else
  0 -- this serves as a placeholder for any other situation

theorem angle_between_a_b_is_60_deg
  (a b : ℝ^2)
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 1)
  (hab : ‖a + b‖ = Real.sqrt 7) :
  angle_between_vectors a b ha hb hab = 60 := sorry

end angle_between_a_b_is_60_deg_l572_572261


namespace num_divisors_not_divisible_by_three_l572_572662

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572662


namespace max_tiles_on_floor_l572_572805

-- Definitions based on the given conditions
def tile_length1 := 35 -- in cm
def tile_length2 := 30 -- in cm
def floor_length := 1000 -- in cm
def floor_width := 210 -- in cm

-- Lean 4 statement for the proof problem
theorem max_tiles_on_floor : 
  (max ((floor_length / tile_length1) * (floor_width / tile_length2))
       ((floor_length / tile_length2) * (floor_width / tile_length1))) = 198 := by
  sorry

end max_tiles_on_floor_l572_572805


namespace dakotas_medical_bill_l572_572544

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end dakotas_medical_bill_l572_572544


namespace general_formula_sum_first_n_terms_l572_572221

-- Definitions for arithmetic sequence, geometric aspects and sum conditions 
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (S3 S5 : ℕ) : S3 + S5 = 50
axiom common_difference : d ≠ 0
axiom first_term (a1 : ℕ) : a_n 1 = a1
axiom geometric_conditions (a1 a4 a13 : ℕ)
  (h1 : a_n 1 = a1) (h4 : a_n 4 = a4) (h13 : a_n 13 = a13) :
  a4 = a1 + 3 * d ∧ a13 = a1 + 12 * d ∧ (a4 ^ 2 = a1 * a13)

-- Proving the general formula for a_n
theorem general_formula (a_n : ℕ → ℕ)
  (h : ∀ (n : ℕ), a_n n = 2 * n + 1) : 
  a_n n = 2 * n + 1 := 
sorry

-- Proving the sum of the first n terms of sequence {b_n}
theorem sum_first_n_terms (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (h_bn : ∀ (n : ℕ), b_n n = (2 * n + 1) * 2 ^ (n - 1))
  (h_Tn: ∀ (n : ℕ), T_n n = 1 + (2 * n - 1) * 2^n) :
  T_n n = 1 + (2 * n - 1) * 2^n :=
sorry

end general_formula_sum_first_n_terms_l572_572221


namespace ratio_kittens_to_breeding_rabbits_l572_572785

def breeding_rabbits : ℕ := 10
def kittens_first_spring (k : ℕ) : ℕ := k * breeding_rabbits
def adopted_kittens_first_spring (k : ℕ) : ℕ := 5 * k
def returned_kittens : ℕ := 5
def remaining_kittens_first_spring (k : ℕ) : ℕ := (k * breeding_rabbits) / 2 + returned_kittens

def kittens_second_spring : ℕ := 60
def adopted_kittens_second_spring : ℕ := 4
def remaining_kittens_second_spring : ℕ := kittens_second_spring - adopted_kittens_second_spring

def total_rabbits (k : ℕ) : ℕ := 
  breeding_rabbits + remaining_kittens_first_spring k + remaining_kittens_second_spring

theorem ratio_kittens_to_breeding_rabbits (k : ℕ) (h : total_rabbits k = 121) :
  k = 10 :=
sorry

end ratio_kittens_to_breeding_rabbits_l572_572785


namespace tan_product_identity_l572_572105

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572105


namespace cos_beta_of_acute_angles_l572_572236

theorem cos_beta_of_acute_angles (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : cos α = sqrt 5 / 5) (h4 : sin (α + β) = 3 / 5) :
    cos β = 2 * sqrt 5 / 25 :=
by
  sorry

end cos_beta_of_acute_angles_l572_572236


namespace player_A_winning_strategy_l572_572863

theorem player_A_winning_strategy (r R : ℝ) (h : R ≥ r) :
  exists strategy : (ℕ → (ℝ × ℝ)), -- A strategy is a sequence of points (positions of coins)
  (∀ n, within_large_circle (strategy n) R) ∧
  (∀ n m, n ≠ m → non_overlapping_coins (strategy n) (strategy m) r) ∧
  (player_A_wins ∧ efficient_strategy strategy) :=
by sorry

-- Below are the placeholder definitions for the proof to compile
def within_large_circle (pos : (ℝ × ℝ)) (R : ℝ) := sorry -- Defines a valid position within the large circle
def non_overlapping_coins (pos1 pos2 : (ℝ × ℝ)) (r : ℝ) := sorry -- Ensures no overlapping coins
def player_A_wins := sorry -- A condition that denotes Player A's winning strategy
def efficient_strategy (strategy : ℕ → (ℝ × ℝ)) := sorry -- Ensures the strategy is efficient relative to the rules of the game

end player_A_winning_strategy_l572_572863


namespace square_of_binomial_l572_572573

theorem square_of_binomial {a r s : ℚ} 
  (h1 : r^2 = a)
  (h2 : 2 * r * s = 18)
  (h3 : s^2 = 16) : 
  a = 81 / 16 :=
by sorry

end square_of_binomial_l572_572573


namespace dean_insects_l572_572502

-- Definitions for the given problem
variables (J D A : ℕ)

-- Conditions
def condition1 : Prop := A = 75
def condition2 : Prop := J = 2 * A
def condition3 : Prop := J = 5 * D

-- The proof statement
theorem dean_insects (h1 : condition1) (h2 : condition2) (h3 : condition3) : D = 30 := by
  sorry

end dean_insects_l572_572502


namespace tangent_product_eq_three_l572_572112

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572112


namespace lambda_range_l572_572611

variable {R : Type*} [LinearOrderedField R]

def sum_of_first_n_terms (S_n : ℕ → R) (a_n : ℕ → R) : Prop :=
  ∀ n : ℕ, 2 * S_n n = (2 * n + 1) * a_n n - 2 * (n^2) 

def sequence_conditions (a_n b_n : ℕ → R) : Prop :=
  a_n 1 = 2 ∧ b_n 1 = a_n 1 ∧ (∀ n : ℕ, n > 0 → b_n (n + 1) = a_n n * b_n n / n)

def t_n_sum (T_n : ℕ → R) (a_n b_n : ℕ → R) : Prop :=
  ∀ n : ℕ, T_n n = ∑ i in finset.range n, a_n i / b_n i

theorem lambda_range (λ : R) (S_n a_n b_n T_n : ℕ → R) (h_conditions : sequence_conditions a_n b_n) (h_sum : sum_of_first_n_terms S_n a_n) (h_T_sum : t_n_sum T_n a_n b_n) :
  (∀ n : ℕ, n > 0 → λ + (n + 9) / 2^n ≥ 4 - T_n n) → λ ≥ 1 / 64 :=
sorry  -- proof to be filled in

end lambda_range_l572_572611


namespace plane_equation_l572_572577

theorem plane_equation 
  (P Q : ℝ×ℝ×ℝ) (A B : ℝ×ℝ×ℝ)
  (hp : P = (-1, 2, 5))
  (hq : Q = (3, -4, 1))
  (ha : A = (0, -2, -1))
  (hb : B = (3, 2, -1)) :
  ∃ (a b c d : ℝ), (a = 3 ∧ b = 4 ∧ c = 0 ∧ d = 1) ∧ (∀ x y z : ℝ, a * (x - 1) + b * (y + 1) + c * (z - 3) = d) :=
by
  sorry

end plane_equation_l572_572577


namespace tan_identity_l572_572157

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572157


namespace minimum_AP_BP_l572_572328

noncomputable def point := (ℝ × ℝ)

def A : point := (2, 0)
def B : point := (7, 6)

-- Define the parabola y^2 = 8*x in Lean
def on_parabola (P : point) : Prop := P.2 ^ 2 = 8 * P.1

def distance (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def AP (P : point) : ℝ := distance A P
def BP (P : point) : ℝ := distance B P

-- Define the problem statement
theorem minimum_AP_BP :
  ∃ P : point, on_parabola P ∧ (AP P + BP P) = 3 * real.sqrt 13 :=
sorry

end minimum_AP_BP_l572_572328


namespace right_triangle_perimeter_l572_572946

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572946


namespace tan_product_equals_three_l572_572162

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572162


namespace num_non_divisible_by_3_divisors_l572_572673

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572673


namespace intersection_point_locus_is_parallelogram_interior_l572_572606

variable {A B C D K L M N : Type*}

-- Define points on the quadrilateral
variables [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D A]
-- Define segments and conditions
variables [has_dist K B] [has_dist B L] (a b : ℝ)

-- Assume all required conditions
axiom kb_eq_bl : dist K B = dist B L
axiom md_eq_dn : dist M D = dist D N
axiom kl_parallel_mn : parallel (line_through K L) (line_through M N)

noncomputable def intersection_point_locus : set (aff_point ℝ ℝ) :=
{ P : aff_point ℝ ℝ |
  ∃ (a b : ℝ),
  (dist K B = a ∧ dist B L = a ∧ dist M D = b ∧ dist D N = b) ∧
  parallel (line_through K L) (line_through M N) ∧
  is_intersection_point (line_through K L) (line_through M N) P }

theorem intersection_point_locus_is_parallelogram_interior :
  ∀ P ∈ interior (parallelogram A B C D), P ∈ intersection_point_locus a b :=
sorry

end intersection_point_locus_is_parallelogram_interior_l572_572606


namespace cube_difference_divisibility_l572_572008

-- Given conditions
variables {m n : ℤ} (h1 : m % 2 = 1) (h2 : n % 2 = 1) (k : ℕ)

-- The equivalent statement to be proven
theorem cube_difference_divisibility (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) :=
sorry

end cube_difference_divisibility_l572_572008


namespace r_plus_r0_eq_R_l572_572783

variable (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variable (ABC : triangle A B C)
variable (AEF BFD CDE DEF : triangle A B C)
variable (r r0 R : ℝ)

-- Points D, E, F lie on the sides BC, CA, AB of triangle ABC respectively
variable (on_BC : point_velocity D B C)
variable (on_CA : point_velocity E C A)
variable (on_AB : point_velocity F A B)

-- The incircles of triangles AEF, BFD, CDE have the same radius r
variable (incircle_AEF : incircle A E F r)
variable (incircle_BFD : incircle B F D r)
variable (incircle_CDE : incircle C D E r)

-- The radii of the incircles of triangles DEF and ABC are r0 and R, respectively
variable (incircle_DEF : incircle D E F r0)
variable (incircle_ABC : incircle A B C R)

-- Prove that r + r0 = R
theorem r_plus_r0_eq_R :
  r + r0 = R :=
sorry

end r_plus_r0_eq_R_l572_572783


namespace binomial_9_3_l572_572531

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l572_572531


namespace blue_markers_count_l572_572533

-- Definitions based on given conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Main statement to prove
theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l572_572533


namespace classify_events_l572_572395

-- Definitions
def certain_event (P : Prop) : Prop := ∀ ω, P
def impossible_event (P : Prop) : Prop := ∀ ω, ¬ P

-- Statement
theorem classify_events (P Q : Prop) :
  (certain_event P ↔ P = true) ∧ (impossible_event Q ↔ Q = false) := 
sorry

end classify_events_l572_572395


namespace tan_product_pi_nine_l572_572127

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572127


namespace total_number_of_animals_is_650_l572_572510

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l572_572510


namespace proof_identity_l572_572249

theorem proof_identity 
  (a b m n : ℝ)
  (α : ℝ)
  (h1 : a * Real.sin α + b * Real.cos α = m)
  (h2 : b * Real.tan α - n * Real.sec α = a) : 
  a^2 + b^2 = m^2 + n^2 :=
sorry

end proof_identity_l572_572249


namespace smallest_n_l572_572770

theorem smallest_n (n : ℕ) 
  (h1 : ∀ i : ℕ, i < n → 0 ≤ y i) 
  (h2 : (∑ i in Finset.range n, y i) = 1) 
  (h3 : (∑ i in Finset.range n, (y i)^2) ≤ 1 / 50) : 
  n = 50 := 
sorry

end smallest_n_l572_572770


namespace ice_cream_tubs_eaten_l572_572429

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end ice_cream_tubs_eaten_l572_572429


namespace find_x_l572_572914

theorem find_x :
  ∃ x : Real, (√((x - 2)^2 + (5 - 2)^2) = 8) ∧ (x > 0) ∧ (x = 2 + Real.sqrt 55) :=
by
  sorry

end find_x_l572_572914


namespace propP_necessary_but_not_sufficient_l572_572241

open Function Real

variable (f : ℝ → ℝ)

-- Conditions: differentiable function f and the proposition Q
def diff_and_propQ (h_deriv : Differentiable ℝ f) : Prop :=
∀ x : ℝ, abs (deriv f x) < 2018

-- Proposition P
def propP : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018

-- Final statement
theorem propP_necessary_but_not_sufficient (h_deriv : Differentiable ℝ f) (hQ : diff_and_propQ f h_deriv) : 
  (∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) ∧ 
  ¬(∀ x : ℝ, abs (deriv f x) < 2018 ↔ ∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) :=
by
  sorry

end propP_necessary_but_not_sufficient_l572_572241


namespace right_triangle_perimeter_l572_572948

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572948


namespace angle_XIZ_75_l572_572302

theorem angle_XIZ_75
  (XYZ : Type)
  [triangle XYZ]
  (XP YQ ZR angle_bisectors : boolean)
  (I : incenter)
  (a b c : ℝ)
  (XYZ_sum : a + b + 30 = 180) :
  b / 2 + a / 2 = 75 :=
sorry

end angle_XIZ_75_l572_572302


namespace ex3_solutions_abs_eq_l572_572245

theorem ex3_solutions_abs_eq (a : ℝ) : (∃ x1 x2 x3 x4 : ℝ, 
        2 * abs (abs (x1 - 1) - 3) = a ∧ 
        2 * abs (abs (x2 - 1) - 3) = a ∧ 
        2 * abs (abs (x3 - 1) - 3) = a ∧ 
        2 * abs (abs (x4 - 1) - 3) = a ∧ 
        x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ (x1 = x4 ∨ x2 = x4 ∨ x3 = x4)) ↔ a = 6 :=
by
    sorry

end ex3_solutions_abs_eq_l572_572245


namespace interval_of_increase_range_of_a_exists_a_decreasing_increasing_l572_572899
open Real

noncomputable def f (x a : ℝ) : ℝ := exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : (a > 0 → ∀ x, x > ln a → deriv (λ x, f x a) x ≥ 0) ∧ (a ≤ 0 → ∀ x, deriv (λ x, f x a) x ≥ 0) :=
sorry

theorem range_of_a : ∀ (a : ℝ), (∀ x, deriv (λ x, f x a) x ≥ 0) → a ≤ 0 :=
sorry

theorem exists_a_decreasing_increasing : ∃ (a : ℝ), (∀ x, x ≤ 0 → deriv (λ x, f x a) x ≤ 0) ∧ (∀ x, x ≥ 0 → deriv (λ x, f x a) x ≥ 0) ∧ a = 1 :=
sorry

end interval_of_increase_range_of_a_exists_a_decreasing_increasing_l572_572899


namespace select_number_among_49_to_64_l572_572909

-- Definition of the conditions
def total_students := 800
def interval_number := 16
def random_start := 7

-- The question rewritten as a statement in Lean 4 
theorem select_number_among_49_to_64 : ∃ n, n ∈ set.Icc 49 64 ∧ n = 55 :=
by
  use 55
  split
  simp
  sorry

end select_number_among_49_to_64_l572_572909


namespace area_of_triangle_angle_C_l572_572304

-- Definitions for Part (1)
def cos_B := 3 / 5
def dot_product_AB_BC := -21

-- Proof Statement for Part (1)
theorem area_of_triangle (a b c : ℝ) (h_cos_B : cos B = cos_B) (h_dot_AB_BC : a * c * cos B = -dot_product_AB_BC) :
  (1 / 2) * a * c * sqrt(1 - cos_B ^ 2) = 14 := 
sorry

-- Definitions for Part (2)
def side_c := 5

-- Proof Statement for Part (2)
theorem angle_C (a b c : ℝ) (h_cos_B : cos B = cos_B) (h_dot_AB_BC : a * c * cos B = -dot_product_AB_BC) (h_c : c = side_c) :
  acos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 4 := 
sorry

end area_of_triangle_angle_C_l572_572304


namespace sequence_properties_l572_572609

theorem sequence_properties
  (a b : ℤ)
  (a_n : ℕ → ℤ)
  (h1 : a_n 1 = a)
  (h2 : a_n 2 = b)
  (h3 : ∀ n ≥ 2, a_n (n + 1) = a_n n - a_n (n - 1))
  (S_n : ℕ → ℤ)
  (S_def : ∀ n, S_n n = ∑ i in finset.range (n+1), a_n (i+1)) :
  a_n 100 = -a ∧ S_n 100 = 2 * b - a :=
by {
  sorry
}

end sequence_properties_l572_572609


namespace reflection_point_C_l572_572377

theorem reflection_point_C'' (C : ℝ × ℝ) (hx : C = (6, 5)) : 
    (let C' := (C.1, -C.2) in (C'.2, -C'.1)) = (5, -6) := by
  sorry

end reflection_point_C_l572_572377


namespace find_triple_l572_572571

theorem find_triple (x y z : ℕ) (h1 : 100 ≤ x ∧ x < y ∧ y < z ∧ z < 1000)
  (h2 : y - x = z - y)
  (h3 : y^2 = x * (z + 1000))
  (h4 : sqrt (1000 * x) ∈ ℕ) :
  (x = 160 ∧ y = 560 ∧ z = 960) := by
  sorry

end find_triple_l572_572571


namespace sum_of_even_numbers_202_to_300_l572_572415

noncomputable def sum_of_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_of_even_numbers_range (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_of_even_numbers_202_to_300 :
  sum_of_first_n_even_numbers 50 = 2550 →
  sum_of_even_numbers_range 202 300 = 12550 :=
by
  intros h
  sorry

end sum_of_even_numbers_202_to_300_l572_572415


namespace percent_shaded_of_square_l572_572881

theorem percent_shaded_of_square (side_len : ℤ) (first_layer_side : ℤ) 
(second_layer_outer_side : ℤ) (second_layer_inner_side : ℤ)
(third_layer_outer_side : ℤ) (third_layer_inner_side : ℤ)
(h_side : side_len = 7) (h_first : first_layer_side = 2) 
(h_second_outer : second_layer_outer_side = 5) (h_second_inner : second_layer_inner_side = 3) 
(h_third_outer : third_layer_outer_side = 7) (h_third_inner : third_layer_inner_side = 6) : 
  (4 + (25 - 9) + (49 - 36)) / (side_len * side_len : ℝ) = 33 / 49 :=
by
  -- Sorry is used as we are only required to construct the statement, not the proof.
  sorry

end percent_shaded_of_square_l572_572881


namespace find_width_of_second_sheet_l572_572398

-- Define the first rectangular sheet dimensions
def sheet1_length : ℝ := 11
def sheet1_width : ℝ := 9
def sheet1_area := 2 * (sheet1_length * sheet1_width)

-- Define the second rectangular sheet known dimensions and a variable for the unknown width
def sheet2_length : ℝ := 4.5
variable (w : ℝ) -- Unknown width of the second sheet

-- Hypothesis: The combined area difference condition
def condition : Prop :=
  sheet1_area = 100 + 2 * (sheet2_length * w)

-- Prove that w equals to 98/9
theorem find_width_of_second_sheet (h : condition) : w = 98 / 9 :=
by
  sorry

end find_width_of_second_sheet_l572_572398


namespace right_triangle_perimeter_l572_572945

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572945


namespace tan_identity_l572_572152

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572152


namespace max_product_l572_572339

-- Define the functions h and j with their respective ranges
variable (h j : ℝ → ℝ)
variable (x : ℝ)

-- Assume the ranges of the functions
def range_h (x : ℝ) : Prop := h x ∈ Icc (-4 : ℝ) 2
def range_j (x : ℝ) : Prop := j x ∈ Icc (0 : ℝ) 3

-- Problem statement: Prove that the largest possible value of the product h(x) * j(x) is 6.
theorem max_product (x : ℝ) (Hh : range_h h x) (Hj : range_j j x) : ∃ (h j : ℝ), h ∈ Icc (-4 : ℝ) 2 ∧ j ∈ Icc (0 : ℝ) 3 ∧ h * j = 6 :=
sorry

end max_product_l572_572339


namespace prod_sqrt2_plus_x_geq_sqrt2_plus_1_pow_n_l572_572596

theorem prod_sqrt2_plus_x_geq_sqrt2_plus_1_pow_n
  {n : ℕ} (n_pos : 0 < n)
  (x : Fin n → ℝ)
  (h₁ : ∀ i, 0 < x i)
  (h₂ : ∏ i, x i = 1) :
  ∏ i, (Real.sqrt 2 + x i) ≥ (Real.sqrt 2 + 1) ^ n :=
by
  sorry

end prod_sqrt2_plus_x_geq_sqrt2_plus_1_pow_n_l572_572596


namespace perimeter_of_given_triangle_l572_572930

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572930


namespace radius_of_given_perimeter_l572_572410

noncomputable def radius_of_semi_circle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem radius_of_given_perimeter :
  radius_of_semi_circle 24.67964473723101 ≈ 4.7999880734837 :=
begin
  sorry
end

end radius_of_given_perimeter_l572_572410


namespace fixed_point_chord_l572_572597

noncomputable theory
open_locale classical

variables {S : Type*} [euclidean_space S] (O : S) (r : ℝ) (l : affine_subspace ℝ S) (P : S)

-- Define the circle with center O and radius r
def circle (O : S) (r : ℝ) : set S :=
  {X | dist X O = r}

-- Define the line l does not intersect the circle S
def line_disjoint_circle (l : affine_subspace ℝ S) (O : S) (r : ℝ) : Prop :=
  ∃ (M : S), M ∈ l ∧ dist O M > r

-- Define the tangents PA and PB from the point P to the circle O with radius r
def is_tangent (P A : S) (O : S) (r : ℝ) : Prop :=
  dist P A = dist P O ∧ dist A O = r

-- Formalize the theorem
theorem fixed_point_chord {S : Type*} [euclidean_space S] (O : S) (r : ℝ) (l : affine_subspace ℝ S) :
  (line_disjoint_circle l O r) →
  (∀ P ∈ l, ∃ A B : S, is_tangent P A O r ∧ is_tangent P B O r) →
  ∃ X : S, ∀ (P ∈ l) (A B : S), is_tangent P A O r ∧ is_tangent P B O r →
  (let M := classical.some (line_disjoint_circle l O r) in
   collinear {O, M, X} ∧ (line_through A B).intersect (line_through O M) = {X})
  sorry

end fixed_point_chord_l572_572597


namespace diameter_is_longest_chord_l572_572630

/-- Given the statements:
A: Three points determine a circle
B: The diameter in the same circle is the longest chord
C: The inscribed angle is half of the central angle
D: Arcs of equal length are congruent
Prove that the correct statement is: B -/

def correct_statement (A B C D : Prop) : B :=
  ¬ A ∧ (B ∧ ¬ C ∧ ¬ D)

variable (A B C D : Prop)

theorem diameter_is_longest_chord
  (hA : ¬(∀ (p q r : Set Point), p ≠ q ∧ q ≠ r ∧ p ≠ r → ∃ c : Circle, c.Through(p) ∧ c.Through(q) ∧ c.Through(r)))
  (hB : ∀ (c : Circle) (d : Chord) (diameter : d.isDiameter), ∀ (c' : Chord), c'.length ≤ d.length)
  (hC : ¬(∀ (angle : Angle) (inscribed : angle.isInscribed ∧ angle.centralAngle = centralAngle.correspondingTo(inscribed)), angle.measure = 1 / 2 * centralAngle.measure))
  (hD : ¬(∀ (arc1 arc2 : Arc), arc1.length = arc2.length → arc1.isCongruent arc2)) :
  correct_statement A B C D :=
by
  intros
  split
  { exact hA }
  construct
  { exact hB }
  { intro h
    exact hC }
  { intro h
    exact hD }
proof
  apply and.intro,
  exact hA,
  apply and.intro hB,
  apply and.intro hC hD,
sorry

end diameter_is_longest_chord_l572_572630


namespace probability_of_sphere_in_cube_l572_572922

noncomputable def cube_volume : Real :=
  (4 : Real)^3

noncomputable def sphere_volume : Real :=
  (4 / 3) * Real.pi * (2 : Real)^3

noncomputable def probability : Real :=
  sphere_volume / cube_volume

theorem probability_of_sphere_in_cube : probability = Real.pi / 6 := by
  sorry

end probability_of_sphere_in_cube_l572_572922


namespace prob1_prob2_prob3_l572_572183

def star (a b : ℤ) : ℤ :=
  if a = 0 then b^2
  else if b = 0 then a^2
  else if a > 0 ∧ b > 0 then a^2 + b^2
  else if a < 0 ∧ b < 0 then a^2 + b^2
  else -(a^2 + b^2)

theorem prob1 :
  star (-1) (-1) = 2 :=
sorry

theorem prob2 :
  star (-1) (star 0 (-2)) = -17 :=
sorry

theorem prob3 (m n : ℤ) :
  star (m-1) (n+2) = -2 → (m - n = 1 ∨ m - n = 5) :=
sorry

end prob1_prob2_prob3_l572_572183


namespace right_angled_triangle_only_B_l572_572004

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l572_572004


namespace min_value_expression_l572_572336

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = ∑ i in range(3), (if i == 0 then (a + 3 * c) / (a + 2 * b + c)
   else if i == 1 then 4 * b / (a + b + 2 * c)
   else if i == 2 then -(8 * c) / (a + b + 3 * c)
   else 0 ) ∧ x = -17 + 12 * real.sqrt 2 := sorry

end min_value_expression_l572_572336


namespace tan_product_pi_nine_l572_572121

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572121


namespace number_of_ways_to_paint_two_faces_red_l572_572501

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def products_not_perfect_square_pairs (faces : Finset ℕ) : Finset (ℕ × ℕ) :=
  faces.product faces |>.filter (λ p, p.1 ≠ p.2 ∧ ¬ is_perfect_square (p.1 * p.2))

def ways_to_paint_faces_red (faces : Finset ℕ) : ℕ :=
  (products_not_perfect_square_pairs faces).card / 2

theorem number_of_ways_to_paint_two_faces_red : ways_to_paint_faces_red (Finset.range 8) = 16 := by
  sorry

end number_of_ways_to_paint_two_faces_red_l572_572501


namespace teachers_left_after_lunch_l572_572025

def fifty_percent (x : ℕ) := x / 2
def thirty_percent (x : ℕ) := (3 * x) / 10

def teachers_after_lunch (initial_teachers : ℕ) : ℕ :=
  let after_one_hour := initial_teachers - fifty_percent initial_teachers
  let after_lunch := after_one_hour - thirty_percent after_one_hour
  after_lunch

theorem teachers_left_after_lunch (initial_teachers : ℕ) :
  teachers_left_after_lunch 60 = 21 :=
by
  sorry

end teachers_left_after_lunch_l572_572025


namespace matrix_vector_mult_correct_l572_572171

noncomputable def matrixA : matrix (fin 3) (fin 3) ℤ := ![
  ![2, 0, -1],
  ![0, 3, -2],
  ![-2, 3, 2]
]

noncomputable def matrixB : matrix (fin 3) (fin 3) ℤ := ![
  ![1, -1, 1],
  ![2, 1, -2],
  ![3, 0, 0]
]

noncomputable def vectorC : vector ℤ (fin 3) := ![1, -1, 1]

noncomputable def expected_result : vector ℤ (fin 3) := ![1, -3, -2]

theorem matrix_vector_mult_correct : (matrixA ⬝ matrixB) ⬝ vectorC = expected_result :=
by
  sorry

end matrix_vector_mult_correct_l572_572171


namespace quadratic_has_distinct_real_roots_l572_572741

theorem quadratic_has_distinct_real_roots {k : ℝ} (hk : k < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ + k = 0) ∧ (x₂^2 - x₂ + k = 0) :=
by
  -- Proof goes here.
  sorry

end quadratic_has_distinct_real_roots_l572_572741


namespace right_triangle_perimeter_l572_572942

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572942


namespace avg_length_of_song_is_three_l572_572750

def total_amount_per_year : ℝ := 2400
def cost_per_song : ℝ := 0.5
def songs_per_year := total_amount_per_year / cost_per_song
def months_per_year : ℕ := 12
def songs_per_month := songs_per_year / months_per_year
def hours_per_month : ℝ := 20
def minutes_per_hour : ℕ := 60
def minutes_per_month := hours_per_month * minutes_per_hour
def average_length_of_song := minutes_per_month / songs_per_month

theorem avg_length_of_song_is_three : average_length_of_song = 3 := 
by
  sorry

end avg_length_of_song_is_three_l572_572750


namespace count_convex_33_gons_l572_572647

theorem count_convex_33_gons (n k : ℕ) (h₁ : n = 100) (h₂ : k = 33) :
    ∑ i in finset.range (n - k), choose (n - k) i + choose (n - k - 1) (k - 1) = choose 67 33 + choose 66 32 := 
  sorry

end count_convex_33_gons_l572_572647


namespace tan_product_eq_three_l572_572077

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572077


namespace triangle_max_area_l572_572537

variables (AB BC AC : ℝ)
variables (x : ℝ)
noncomputable def s := (13 + 121 * x) / 2

def heron_formula (a b c : ℝ) :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_max_area (h1 : AB = 13)
                          (h2 : BC = 60 * x)
                          (h3 : AC = 61 * x)
                          (h4 : 0 < x)
                          (h5 : x < 13)
                          (h6 : 13 / 121 < x) :
  heron_formula 13 (60 * x) (61 * x) ≤ 3634 :=
begin
  sorry
end

end triangle_max_area_l572_572537


namespace triangle_angle_inequality_l572_572628

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles A, B, and C

-- Assume A / sin A = a, B / sin B = b for the Law of Sines
variable (triangle_abc : ∀ A B C a b c, a / Real.sin A = b / Real.sin B)

-- Given condition
variable (sin_A_gt_sin_B : Real.sin A > Real.sin B)

-- Lean formalization of the problem
theorem triangle_angle_inequality
  (h : ∀ A B C a b c, a / Real.sin A = b / Real.sin B)
  (sin_A_gt_sin_B : Real.sin A > Real.sin B) :
  A > B := 
sorry

end triangle_angle_inequality_l572_572628


namespace min_value_9x_plus_3y_l572_572265

theorem min_value_9x_plus_3y (x y : ℝ) (h : 4 * x + 2 * y = 4) : 9^x + 3^y ≥ 6 :=
by
  -- Proof steps will be added here
  sorry

end min_value_9x_plus_3y_l572_572265


namespace polynomial_characterization_l572_572569

theorem polynomial_characterization (P : ℚ[X]) :
  (∀ r : ℚ, ∃ d : ℚ, P.eval d = r) →
  ∃ a b : ℚ, a ≠ 0 ∧ P = polynomial.C a * polynomial.X + polynomial.C b :=
by sorry

end polynomial_characterization_l572_572569


namespace regular_icosahedron_edges_l572_572652

-- Define what a regular icosahedron is
def is_regular_icosahedron (P : Type) := -- Definition placeholder for a regular icosahedron
  sorry

-- Define the function that counts edges of a polyhedron
def count_edges (P : Type) [is_regular_icosahedron P] : ℕ :=
  sorry

-- The proof statement
theorem regular_icosahedron_edges (P : Type) [h : is_regular_icosahedron P] : count_edges P = 30 :=
  sorry

end regular_icosahedron_edges_l572_572652


namespace number_of_true_propositions_l572_572552

theorem number_of_true_propositions :
  (∀ x : ℝ, 2^x > 0) ∧
  (∃ α β : ℝ, sin (α + β) = sin α + sin β) ∧
  (¬ (∀ (a b c : ℝ), a > b → a * c ^ 2 > b * c ^ 2)) ∧
  (¬ (∀ (A B : ℝ), A > B → sin A > sin B)) →
  3 = 3 :=
by
  intros h
  sorry

end number_of_true_propositions_l572_572552


namespace largest_area_right_angled_quadrilateral_l572_572993

theorem largest_area_right_angled_quadrilateral 
  (r : ℝ)
  (α : ℝ) 
  (h1 : 2 * α = 120 ∨ 2 * α = 240) :
  ∃ T : ℝ, 
    (T = r^2 * (real.sqrt 3) / 3 ∧ 2 * α = 120) ∨ 
    (T = r^2 * (real.sqrt 3) ∧ 2 * α = 240) := by
  sorry

end largest_area_right_angled_quadrilateral_l572_572993


namespace tan_product_l572_572082

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572082


namespace tan_identity_l572_572159

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572159


namespace volume_of_extended_parallelepiped_correct_l572_572992

theorem volume_of_extended_parallelepiped_correct : ∃ m n p : ℕ, ∀ (a b c : ℝ) (r : ℝ), 
  a = 2 → b = 3 → c = 6 → r = 2 →
  let V := a * b * c + 
            2 * (a * b * r + a * c * r + b * c * r) + 
            (8 * (1 / 8) * (4 / 3) * π * r^3) +
            12 * (1 / 4) * π * r^2 * (a + b + c - r)  
  in V = (m + n * π) / p ∧ Nat.gcd n p = 1 ∧ m + n + p = 701 :=
begin
  -- Definitions
  let a := 2,
  let b := 3,
  let c := 6,
  let r := 2,
  -- Volumes calculation
  let V_box := a * b * c,
  let V_ext := 2 * (a * b * r + a * c * r + b * c * r),
  let V_sph := 8 * (1 / 8) * (4 / 3) * π * r^3,
  let V_cyl := 12 * (1 / 4) * π * r^2 * (a + b + c - r),
  let V := V_box + V_ext + V_sph + V_cyl,
  use [540,158,3],
  split,
  { simp [V_box, V_ext, V_sph, V_cyl] at V, 
    sorry
  },
  split,
  { simp, sorry },
  { simp, sorry } 
end

end volume_of_extended_parallelepiped_correct_l572_572992


namespace fixed_amount_per_appliance_l572_572539

variable (F : ℕ)
variable (total_commission total_price_per_week fixed_commission_per_week : ℕ)
variable [Fact (total_commission = 662)] [Fact (total_price_per_week = 3620)] 

theorem fixed_amount_per_appliance 
    (h1 : fixed_commission_per_week = 6 * F) 
    (h2 : 0.10 * total_price_per_week = 362) 
    (h3 : total_commission = fixed_commission_per_week + 362) : 
    F = 50 := 
by 
    sorry

end fixed_amount_per_appliance_l572_572539


namespace count_divisors_not_divisible_by_3_l572_572702

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572702


namespace fourth_guard_distance_l572_572958

theorem fourth_guard_distance 
  (length : ℝ) (width : ℝ)
  (total_distance_three_guards: ℝ)
  (P : ℝ := 2 * (length + width)) 
  (total_distance_four_guards : ℝ := P)
  (total_three : total_distance_three_guards = 850)
  (length_value : length = 300)
  (width_value : width = 200) :
  ∃ distance_fourth_guard : ℝ, distance_fourth_guard = 150 :=
by 
  sorry

end fourth_guard_distance_l572_572958


namespace josef_picks_satisfying_numbers_l572_572312

-- Define a function to compute the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement.
theorem josef_picks_satisfying_numbers :
  { n : ℕ | (1 ≤ n ∧ n ≤ 1080 ∧ 1080 % n = 0 ∧ digit_sum n > 10) }.size = 6 :=
by
  sorry

end josef_picks_satisfying_numbers_l572_572312


namespace final_solution_percentage_X_l572_572392

variables (Y B D : Type)
variables [has_percentage (Y : Type)] [has_percentage (B : Type)] [has_percentage (D : Type)]
variables (SolY_initial : Y) (SolY_added : Y) (SolB_added : B) (SolD_added : D)

-- Given initial conditions
def percentage_combination_Y := (20, 55, 25)  -- percentages (X, water, Z)
def initial_Y_mass := 12
def evaporated_water_Y := 4
def added_mass_Y := 3
def percentage_combination_B := (35, 15, 50)  -- percentages (X, water, Z)
def added_mass_B := 2
def evaporation_percentage := 75
def percentage_combination_D := (15, 60, 25)  -- percentages (X, water, Z)
def added_mass_D := 6

noncomputable def calculate_final_percentage_X : ℚ :=
  let final_mass := 17.8625
  let final_liquid_X := 4.6
  (final_liquid_X / final_mass) * 100

-- Proof problem statement
theorem final_solution_percentage_X :
  calculate_final_percentage_X = 25.75 :=
sorry

end final_solution_percentage_X_l572_572392


namespace arrangement_count_l572_572420

-- Define the students
constants A B C D E F : Type

-- Define the arrangement problem
def arrangement :=
  list (A | B | C | D | E | F)

-- Define the conditions 
def conditions (arr : arrangement) : Prop :=
  -- Condition 1: A is not at the head and B is not at the tail
  (arr.head ≠ A) ∧ (arr.last ≠ B) ∧
  -- Condition 2: A, B, and C are not adjacent to each other
  (∀ i, i < arr.length - 2 → 
    ¬ ((arr.nth i = some A ∧ arr.nth (i+1) = some B ∧ arr.nth (i+2) = some C) ∨
       (arr.nth i = some C ∧ arr.nth (i+1) = some B ∧ arr.nth (i+2)= some A) ∨
       (arr.nth i = some B ∧ arr.nth (i+1) = some A ∧ arr.nth (i+2) = some C) ∨
       (arr.nth i = some B ∧ arr.nth (i+1) = some C ∧ arr.nth (i+2) = some A) ∨
       (arr.nth i = some A ∧ arr.nth (i+1) = some C ∧ arr.nth (i+2) = some B) ∨
       (arr.nth i = some C ∧ arr.nth (i+1) = some A ∧ arr.nth (i+2) = some B))) ∧
  -- Condition 3: A and B are adjacent, and C and D are adjacent
  (∀ i, i < arr.length - 1 →
    ((arr.nth i = some A ∧ arr.nth (i+1) = some B) ∨
    (arr.nth i = some B ∧ arr.nth (i+1) = some A))) ∧
  (∀ j, j < arr.length - 1 → 
    ((arr.nth j = some C ∧ arr.nth (j+1) = some D) ∨
    (arr.nth j = some D ∧ arr.nth (j+1) = some C))) ∧
  -- Condition 4: Neither A nor B is adjacent to C
  (∀ k, k < arr.length - 1 → 
    ¬ ((arr.nth k = some A ∧ arr.nth (k+1) = some C) ∨
       (arr.nth k = some C ∧ arr.nth (k+1) = some A) ∨
       (arr.nth k = some B ∧ arr.nth (k+1) = some C) ∨
       (arr.nth k = some C ∧ arr.nth (k+1) = some B)))

-- The proof statement
theorem arrangement_count : ∃ arr, conditions arr := by
  sorry

end arrangement_count_l572_572420


namespace tan_product_identity_l572_572102

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572102


namespace ratio_of_a_and_b_l572_572892

theorem ratio_of_a_and_b (x y a b : ℝ) (h1 : x / y = 3) (h2 : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end ratio_of_a_and_b_l572_572892


namespace differentiable_limit_l572_572351

theorem differentiable_limit (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) : 
  (∃ l : ℝ, Tendsto (fun (Δx : ℝ) => (f(1) - f(1 + Δx)) / (3 * Δx)) (𝓝 0) (𝓝 l) ∧ l = - (1/3) * f' 1) :=
begin
  sorry
end

end differentiable_limit_l572_572351


namespace divisors_not_divisible_by_3_eq_6_l572_572655

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572655


namespace parabola_problem_exist_line_parallel_l572_572641

noncomputable def parabola_eq (p : ℝ) (C : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (x, y) ↔ y^2 = 2 * p * x

noncomputable def focus_dist (p : ℝ) : ℝ × ℝ → ℝ → Prop :=
  λ (x y : ℝ), dist (x, y) ((p/2), 0) = 3

noncomputable def directrix_eq (p : ℝ) : ℝ → Prop :=
  λ x, x = -p / 2

theorem parabola_problem (p : ℝ) (h_p : p > 0) :
  (∃ p > 0, parabola_eq 4 C ∧ directrix_eq 4) :=
begin
  sorry
end

noncomputable def intersect_parabola_line (k : ℝ) : Prop :=
  -- Define that the line y = k(x + 1) intersects the parabola y^2 = 8x at two points.
  ∃ x1 x2 y1 y2, (y1 = k * (x1 + 1)) ∧ (y2 = k * (x2 + 1)) ∧
                 ((y1^2 = 8$x1) ∧ (y2^2 = 8$x2)) ∧ (x1 ≠ x2)

noncomputable def de_parallel_af (k : ℝ) : Prop :=
  -- Define the condition that DE is parallel to AF given the described Geometric configuration.
  let E := (-4, -3 * k) in
  ∃ x1 x2 y1 y2,
    (x1 + x2 = 7) ∧ (y1 = k * (x1 + 1)) ∧ (y2 = k * (x2 + 1)) ∧
    let A := (x1, y1) in
    let F := (2, 0) in
    let D := (x1, 3 * k + (y2 / (x2 - 2))) in
    slope (D, E) = slope (A, F)

theorem exist_line_parallel (l : ℝ → ℝ) :
  ∃ k ≠ 0, intersect_parabola_line (2*ℝ.sqrt(2) / 3) ∧ de_parallel_af (2*ℝ.sqrt(2) / 3) :=
begin
  sorry
end

end parabola_problem_exist_line_parallel_l572_572641


namespace permutation_probability_contains_9_and_12_l572_572780

theorem permutation_probability_contains_9_and_12 :
  let seq := (1:Finₓ 10.succ)
  let pairs := seq.image (λ i, (i, i + 1))
  let product_9 := (1, 9) ∨ (9, 1) ∨ (3, 3)
  let product_12 := (2, 6) ∨ (6, 2) ∨ (3, 4) ∨ (4, 3)
  let count := Nat.factorial 10
  let favorable := (8 * 7 * Nat.factorial 7)
  favorable / count = 7 / 90 :=
by

end permutation_probability_contains_9_and_12_l572_572780


namespace tan_identity_proof_l572_572097

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572097


namespace chords_have_common_point_l572_572599

noncomputable def fixed_point_intersection (S : Type) [metric_space S] (O : S)
  (r : ℝ) (h : 0 < r)
  (P : ℝ → S)
  (l : set S)
  (l_linear : ∀ t, P t ∈ l)
  (l_nonintersect : metric.ball O r ∩ l = ∅)
  (A B : S)
  (PA PB : ∀ t, tangent_line S (P t) A ∧ tangent_line S (P t) B)
  (tangent_line : S → S → Prop)
  (chord_AB : ∀ t, S → S → S → S) : Prop :=
∃ X : S, ∀ t, chord_AB (P t) A B S = X

theorem chords_have_common_point (S : Type) [metric_space S] (O : S)
  (r : ℝ) (h : 0 < r)
  (P : ℝ → S)
  (l : set S)
  (l_linear : ∀ t, P t ∈ l)
  (l_nonintersect : metric.ball O r ∩ l = ∅)
  (tangent_line : S → S → Prop)
  (A B : S)
  (PA PB : ∀ t, tangent_line (P t) A ∧ tangent_line (P t) B)
  (chord_AB : ∀ t, S → S → S → S) :
  fixed_point_intersection S O r h P l l_linear l_nonintersect A B PA PB tangent_line chord_AB := sorry

end chords_have_common_point_l572_572599


namespace tan_product_l572_572137

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572137


namespace collinear_points_given_b_eq_2a_l572_572588

open Real

noncomputable def point_a (a : ℝ) := (2, 0, a)
noncomputable def point_b (a : ℝ) := (2*a, 2, 0)
noncomputable def point_c (c : ℝ) := (0, c, 1)
noncomputable def point_d (d : ℝ) := (9*d, 9*d, -d)

theorem collinear_points_given_b_eq_2a {a c d : ℝ} :
  let p_a := point_a a,
      p_b := point_b a,
      p_c := point_c c,
      p_d := point_d d in
  (λ t : ℝ × ℝ × ℝ, ∃ k : ℝ, t = (9*d - 2, 9*d*k, -d - a))
  (2*a - 2, 2, -a) →
  d = 1/9 :=
by
  -- The proof is omitted
  sorry

end collinear_points_given_b_eq_2a_l572_572588


namespace sum_of_two_primes_is_multiple_of_six_l572_572781

theorem sum_of_two_primes_is_multiple_of_six
  (p q r : ℕ)
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) (hr_gt_3 : r > 3)
  (h_sum_prime : Nat.Prime (p + q + r)) : 
  (p + q) % 6 = 0 ∨ (p + r) % 6 = 0 ∨ (q + r) % 6 = 0 :=
sorry

end sum_of_two_primes_is_multiple_of_six_l572_572781


namespace intersection_points_of_circle_and_line_l572_572740

theorem intersection_points_of_circle_and_line {θ ρ : ℝ} (hθ : θ ∈ set.Ioo 0 π) :
  ((∃ ρ, ρ = cos θ + sin θ ∧ ρ sin (θ - π / 4) = sqrt 2 / 2) →
  (ρ, θ) = (1, π / 2)) := 
sorry

end intersection_points_of_circle_and_line_l572_572740


namespace derivative_at_2_l572_572252

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_at_2 : (deriv f 2) = 1 / 4 :=
by 
  sorry

end derivative_at_2_l572_572252


namespace x_minus_q_eq_3_minus_2q_l572_572712

theorem x_minus_q_eq_3_minus_2q (x q : ℝ) (h₁ : |x - 3| = q) (h₂ : x < 3) : x - q = 3 - 2q :=
by
sorry

end x_minus_q_eq_3_minus_2q_l572_572712


namespace total_trees_planted_l572_572927

/-- Given a rectangular plot of land 60 meters long and 30 meters wide,
    with trees planted around the perimeter at an interval of 6 meters,
    and one tree already planted at each of the four corners,
    prove that the total number of trees planted is 26. -/
theorem total_trees_planted :
  let length := 60
  let width := 30
  let spacing := 6
  let corners := 4
  (2 * (length + width)) / spacing - corners = 26 :=
by
  let length := 60
  let width := 30
  let spacing := 6
  let corners := 4
  calc
    (2 * (length + width)) / spacing - corners
        = (2 * (60 + 30)) / 6 - 4 : by rfl
    ... = 180 / 6 - 4          : by rfl
    ... = 30 - 4               : by rfl
    ... = 26                   : by rfl

end total_trees_planted_l572_572927


namespace storage_temperature_difference_l572_572033

theorem storage_temperature_difference (recommended_temperature tolerance : ℝ) 
  (h_recommended : recommended_temperature = 20) (h_tolerance : tolerance = 2) : 
  let highest_temperature := recommended_temperature + tolerance,
      lowest_temperature := recommended_temperature - tolerance in
  highest_temperature - lowest_temperature = 4 :=
by
  sorry

end storage_temperature_difference_l572_572033


namespace tan_product_equals_three_l572_572169

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572169


namespace sum_of_cubowat_numbers_l572_572474

-- Define a function to check if a number is a perfect cube
def is_cube (x : ℕ) : Prop :=
  ∃ n : ℕ, n^3 = x

-- Define the predicate for a number being "cubowat"
def is_cubowat (n : ℕ) : Prop :=
  is_cube (n^3 + 13 * n - 273)

-- Define the sum of all cubowat numbers
noncomputable def cubowat_sum : ℕ :=
  ∑ n in finset.filter is_cubowat (finset.range 100), n

-- State the theorem to prove
theorem sum_of_cubowat_numbers :
  cubowat_sum = 29 :=
sorry

end sum_of_cubowat_numbers_l572_572474


namespace path_count_from_origin_to_end_l572_572919

def Point : Type := ℕ × ℕ

def valid_step (p1 p2 : Point) : Prop :=
  match p1, p2 with
  | (a, b), (c, d) =>
    (c = a + 1 ∧ d = b) ∨          -- move right
    (c = a ∧ d = b + 1) ∨          -- move up
    (c = a + 1 ∧ d = b + 1) ∨      -- move diagonal
    (c = a - 1 ∧ d = b + 1)        -- move backward diagonal

def diagonal_step (p1 p2 : Point) : Prop :=
  match p1, p2 with
  | (a, b), (c, d) => 
    (c = a + 1 ∧ d = b + 1) ∨      -- forward diagonal
    (c = a - 1 ∧ d = b + 1)        -- backward diagonal

def no_right_angle (path : List Point) : Prop :=
  ∀ x y z, List.nth path x = some y → List.nth path (x + 1) = some z → 
    (valid_step y z →  
    (diagonal_step y z ∨ 
    ∃ w, List.nth path (x + 2) = some w → valid_step z w → 
    ¬(valid_step y z ∧ valid_step z w)))

def allowed_paths (p1 p2 : Point) (steps : ℕ) : ℕ :=
  if steps > 2 then 0
  else if ¬no_right_angle (list.range 1) then 0
  else sorry  -- Placeholder for path counting logic

theorem path_count_from_origin_to_end : 
  allowed_paths (0, 0) (3, 3) 2 = 62 :=
sorry

end path_count_from_origin_to_end_l572_572919


namespace sum_reciprocals_of_partial_fractions_l572_572773

noncomputable def f (s : ℝ) : ℝ := s^3 - 20 * s^2 + 125 * s - 500

theorem sum_reciprocals_of_partial_fractions :
  ∀ (p q r A B C : ℝ),
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    f p = 0 ∧ f q = 0 ∧ f r = 0 ∧
    (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
      (1 / f s = A / (s - p) + B / (s - q) + C / (s - r))) →
    1 / A + 1 / B + 1 / C = 720 :=
sorry

end sum_reciprocals_of_partial_fractions_l572_572773


namespace vector_subtraction_represents_l572_572334

open Real

def vector (x y : ℝ) : ℝ × ℝ := (x, y)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

def angle (v₁ v₂ : ℝ × ℝ) : ℝ := 
  let dot_product := v₁.1 * v₂.1 + v₁.2 * v₂.2 in
  acos (dot_product / (magnitude v₁ * magnitude v₂))

def a : ℝ × ℝ := vector 10 0
def b : ℝ × ℝ := vector 0 (10 * sqrt 3)
def a_sub_b : ℝ × ℝ := vector (a.1 - b.1) (a.2 - b.2)

theorem vector_subtraction_represents :
  magnitude a_sub_b = 20 ∧ angle a_sub_b b = π / 6 :=
by sorry

end vector_subtraction_represents_l572_572334


namespace circumscribed_radius_l572_572895

noncomputable def find_circumscribed_radius (Z : ℝ) : ℝ :=
  let a := 10
  let r := Z
  let b := 15 -- Derived from solution
  let c := 5 + 2 * Z -- Derived from solution
  (5 + 2 * Z) / 2

theorem circumscribed_radius (Z : ℝ) :
  let R := find_circumscribed_radius Z in
  R = (5 + 2 * Z) / 2 :=
by {
  let a := 10,
  let r := Z,
  let b := 15, -- Derived from solution
  let c := 5 + 2 * Z, -- Derived from solution
  refine eq.refl _,
  sorry
}

end circumscribed_radius_l572_572895


namespace divisors_of_180_not_divisible_by_3_l572_572685

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572685


namespace find_AF_plus_BF_inv_l572_572423

def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def line (x y : ℝ) : Prop := y = (Real.sqrt 3) * (x + 1)
def left_focus : ℝ × ℝ := (-1, 0)
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ line A.1 A.2 ∧ ellipse B.1 B.2 ∧ line B.1 B.2

theorem find_AF_plus_BF_inv (A B : ℝ × ℝ) (h_inter : intersection_points A B) :
  let AF := Real.sqrt ((A.1 + 1)^2 + A.2^2)
  let BF := Real.sqrt ((B.1 + 1)^2 + B.2^2)
  in (1 / AF) + (1 / BF) = 4 / 3 :=
sorry

end find_AF_plus_BF_inv_l572_572423


namespace continuity_at_x_2_l572_572586

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem continuity_at_x_2 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 0 :=
by
  sorry

end continuity_at_x_2_l572_572586


namespace triangle_area_is_10sqrt3_l572_572435

noncomputable def semi_perimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def triangle_area_using_heron (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_is_10sqrt3 :
  triangle_area_using_heron 5 7 8 = 10 * Real.sqrt 3 := by
  sorry

end triangle_area_is_10sqrt3_l572_572435


namespace calories_in_300_grams_of_lemonade_l572_572749

def lemon_juice_mass : ℝ := 200
def sugar_mass : ℝ := 150
def water_mass : ℝ := 650

def lemon_juice_caloric_density : ℝ := 40 / 100
def sugar_caloric_density : ℝ := 400 / 100
def water_caloric_density : ℝ := 0

def lemonade_total_mass : ℝ := lemon_juice_mass + sugar_mass + water_mass
def lemonade_total_calories : ℝ := (lemon_juice_mass * lemon_juice_caloric_density) + 
                                    (sugar_mass * sugar_caloric_density) + 
                                    (water_mass * water_caloric_density)

def caloric_density_lemonade : ℝ := lemonade_total_calories / lemonade_total_mass
def amount_to_evaluate : ℝ := 300

def calories_in_300_grams : ℝ := amount_to_evaluate * caloric_density_lemonade

theorem calories_in_300_grams_of_lemonade : calories_in_300_grams = 204 := by
  sorry

end calories_in_300_grams_of_lemonade_l572_572749


namespace correct_statement_is_A_l572_572005

-- Define planes alpha and beta
variables {Point : Type} {Plane : Type} (p : Point) (α β : Plane) 

-- Definition for planes having only one point in common
def planes_have_one_point_in_common (α β : Plane) : Prop :=
  ∃ p : Point, ∀ q : Point, q ∈ α ∧ q ∈ β ↔ q = p

-- Definition for three pairwise intersecting lines must be coplanar
variables (l1 l2 l3 : Type) [line l1] [line l2] [line l3]
def three_lines_pairwise_intersect (l1 l2 l3 : Type) [line l1] [line l2] [line l3] : Prop :=
  (∃ p1 : Point, p1 ∈ l1 ∧ p1 ∈ l2) ∧ 
  (∃ p2 : Point, p2 ∈ l2 ∧ p2 ∈ l3) ∧ 
  (∃ p3 : Point, p3 ∈ l1 ∧ p3 ∈ l3) →
  ∃ plane : Plane, ∀ l : Type, [line l] → l = l1 ∨ l = l2 ∨ l = l3 → l ∈ plane

-- Definition for four non-coplanar points
variables (a b c d : Point)
def four_non_coplanar_points (a b c d : Point) : Prop :=
  ¬ ∃ plane : Plane, a ∈ plane ∧ b ∈ plane ∧ c ∈ plane ∧ d ∈ plane

-- Definition for any three points being non-collinear
def three_points_not_collinear (a b c : Point) : Prop :=
  ¬ ∃ line : Type, [line line], a ∈ line ∧ b ∈ line ∧ c ∈ line

-- Definition for two planes having three points in common must coincide
def planes_have_three_points_in_common (α β : Plane) : Prop :=
  ∃ (x y z : Point), x ∈ α ∧ x ∈ β ∧ y ∈ α ∧ y ∈ β ∧ z ∈ α ∧ z ∈ β → α = β

-- Theorem statement asserting the correctness
theorem correct_statement_is_A :
  (planes_have_one_point_in_common α β) ∨
  three_lines_pairwise_intersect l1 l2 l3 ∨
  (∀ p1 p2 p3 p4 : Point, four_non_coplanar_points p1 p2 p3 p4 → ¬ three_points_not_collinear p1 p2 p3) ∨
  (planes_have_three_points_in_common α β) :=
sorrry

end correct_statement_is_A_l572_572005


namespace tan_product_equals_three_l572_572161

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572161


namespace maximize_area_partition_l572_572433

noncomputable def optimLengthPartition (material: ℝ) (partitions: ℕ) : ℝ :=
  (material / (4 + partitions))

theorem maximize_area_partition :
  optimLengthPartition 24 (2 * 1) = 3 / 100 :=
by
  sorry

end maximize_area_partition_l572_572433


namespace power_equality_1988_l572_572016

theorem power_equality_1988 (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (1:ℝ / x) + (1 / y) + (1 / z) = 0) : 
    x ^ 1988 + y ^ 1988 + z ^ 1988 = 1 / (x ^ 1988) + 1 / (y ^ 1988) + 1 / (z ^ 1988) := 
by
  sorry

end power_equality_1988_l572_572016


namespace max_value_expression_l572_572768

theorem max_value_expression (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = (x^2 + 3 - sqrt (x^4 + 6*x^2 + 1))/x → y ≤ 2/3) :=
sorry

end max_value_expression_l572_572768


namespace proof_problem_l572_572368

def is_palindrome (n : ℕ) : Prop :=
  let str_n := n.toString in
  str_n = str_n.reverse

def steps_to_palindrome (n : ℕ) : ℕ :=
  let rec helper (count : ℕ) (curr : ℕ) :=
    match count with
    | 4 => curr
    | _ =>
      let rev_curr := (curr.toString.reverse.toNat) in
      if is_palindrome (curr + rev_curr) then count + 1
      else helper (count + 1) (curr + rev_curr)
  helper 0 n

def is_non_palindrome_that_reaches_palindrome_in_4_steps (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 200 ∧ ¬is_palindrome n ∧ steps_to_palindrome n ≤ 4

def problem_statement : Prop :=
  let sum_non_palindrome_4_steps := (Finset.filter is_non_palindrome_that_reaches_palindrome_in_4_steps (Finset.Icc 100 200)).sum (fun n => n)
  sum_non_palindrome_4_steps = 262

-- The following does not need a proof
theorem proof_problem : problem_statement :=
  sorry

end proof_problem_l572_572368


namespace angle_FCG_eq_67_l572_572380

variable (A B C D E F G : Point)
variable (circle : Circle ℝ)
variable (arranged_clockwise : Circle.arrange_clockwise [A, B, C, D, E, F, G] circle)
variable (diam_AE : Circle.diameter circle A E)

variable (angle_ABF : angle_geom A B F = 81)
variable (angle_EDG : angle_geom E D G = 76)

theorem angle_FCG_eq_67 : angle_geom F C G = 67 :=
by
  -- exact proof goes here
  sorry

end angle_FCG_eq_67_l572_572380


namespace min_spend_proof_l572_572883

-- Define the conditions from the problem
def box_length := 20
def box_width := 20
def box_height := 12
def box_cost := 0.80
def pack_material_percent := 0.15
def total_volume := 2160000

noncomputable def box_volume : ℕ := box_length * box_width * box_height

noncomputable def usable_box_volume : ℕ := box_volume - (box_volume * (pack_material_percent * 100).toNat / 100)

noncomputable def num_boxes_needed : ℕ := (total_volume + usable_box_volume - 1) / usable_box_volume -- (total_volume.toNat + usable_box_volume.toNat - 1) div usable_box_volume.toNat

noncomputable def min_spend : ℕ := num_boxes_needed * box_cost

-- Lean statement to prove the minimum spending based on given conditions
theorem min_spend_proof : min_spend = 424 := by sorry

end min_spend_proof_l572_572883


namespace barbara_sells_for_2_l572_572060

-- Definitions of the conditions
def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def trish_price_per_animal : ℝ := 1.5
def total_donation : ℝ := 45

-- Definition of the question
def barbara_price_per_animal (x : ℝ) : Prop := 
  9 * x + (18 * trish_price_per_animal) = total_donation

-- The Lean statement proving the correct answer
theorem barbara_sells_for_2 : barbara_price_per_animal 2 :=
by
  have h1 : 18 * trish_price_per_animal = 27 := by norm_num
  have h2 : 9 * 2 + 27 = 45 := by norm_num
  rw [← h1] at h2
  exact h2

end barbara_sells_for_2_l572_572060


namespace find_x_find_union_l572_572714

variable (x : ℤ)
def A := {x^2, 2*x - 1, -4}
def B := {x - 5, 1 - x, 9}

theorem find_x (h : B ∩ A = {9}) : x = -3 := by
  sorry

theorem find_union (h : x = -3) : 
  A ∪ B = { -8, -7, -4, 4, 9 } := by
  sorry

end find_x_find_union_l572_572714


namespace fraction_of_day_crew_is_5Over7_l572_572058

-- Defining the conditions
variables {D W : ℕ} -- D: Boxes loaded per day worker, W: Workers on the day crew
def boxes_per_night_worker := D / 2 -- Each night worker loads 1/2 as many boxes
def night_workers := (4 / 5 : ℚ) * W -- Number of night workers is 4/5 of day workers

-- Defining the total boxes loaded by both day and night crews
def total_day_boxes := D * W
def total_night_boxes := boxes_per_night_worker * night_workers

-- Defining the fraction of boxes loaded by the day crew
def fraction_day_crew_loaded := total_day_boxes / (total_day_boxes + total_night_boxes)

theorem fraction_of_day_crew_is_5Over7 :
  fraction_day_crew_loaded = (5 / 7 : ℚ) :=
by
  sorry

end fraction_of_day_crew_is_5Over7_l572_572058


namespace shapes_axisymmetric_and_centrally_symmetric_l572_572499

-- Definitions of shapes for identification
inductive Shape
| square
| equilateral_triangle
| rectangle
| rhombus

-- Properties of axisymmetry and central symmetry
def is_axisymmetric : Shape → Prop
| Shape.square := True
| Shape.equilateral_triangle := True
| Shape.rectangle := True
| Shape.rhombus := True

def is_centrally_symmetric : Shape → Prop
| Shape.square := True
| Shape.equilateral_triangle := False
| Shape.rectangle := True
| Shape.rhombus := True

-- The statement of the math proof problem
theorem shapes_axisymmetric_and_centrally_symmetric :
  {s : Shape | is_axisymmetric s ∧ is_centrally_symmetric s} = {Shape.square, Shape.rectangle, Shape.rhombus} := 
sorry

end shapes_axisymmetric_and_centrally_symmetric_l572_572499


namespace expression_bounds_l572_572756

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
                     Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ∧
  (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
   Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ≤ 4 := sorry

end expression_bounds_l572_572756


namespace exists_unique_representation_l572_572801

theorem exists_unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3 * x + y) / 2 :=
sorry

end exists_unique_representation_l572_572801


namespace total_rainfall_2003_to_2005_l572_572726

-- Define the conditions given in the problem
def avg_monthly_rainfall_2003 : ℕ := 50 
def avg_monthly_rainfall_2004 : ℕ := avg_monthly_rainfall_2003 + 3
def avg_monthly_rainfall_2005 : ℕ := avg_monthly_rainfall_2004 + 5

-- Calculate the total rainfall for each year
def total_rainfall_2003 : ℕ := 12 * avg_monthly_rainfall_2003
def total_rainfall_2004 : ℕ := 12 * avg_monthly_rainfall_2004
def total_rainfall_2005 : ℕ := 12 * avg_monthly_rainfall_2005

-- Prove the total rainfall over the three years
theorem total_rainfall_2003_to_2005 : 
  total_rainfall_2003 + total_rainfall_2004 + total_rainfall_2005 = 1932 :=
by
  calc
    total_rainfall_2003 + total_rainfall_2004 + total_rainfall_2005
        = 12 * avg_monthly_rainfall_2003 + 12 * avg_monthly_rainfall_2004 + 12 * avg_monthly_rainfall_2005  : by rw [total_rainfall_2003, total_rainfall_2004, total_rainfall_2005]
    ... = 12 * 50 + 12 * (50 + 3) + 12 * (50 + 3 + 5)                          : by rw [avg_monthly_rainfall_2003, avg_monthly_rainfall_2004, avg_monthly_rainfall_2005]
    ... = 600 + 636 + 696                                                  : by norm_num
    ... = 1932                                                              : by norm_num

end total_rainfall_2003_to_2005_l572_572726


namespace tank_saltwater_solution_l572_572484

theorem tank_saltwater_solution (x : ℝ) :
  let water1 := 0.75 * x
  let water1_evaporated := (1/3) * water1
  let water2 := water1 - water1_evaporated
  let salt2 := 0.25 * x
  let water3 := water2 + 12
  let salt3 := salt2 + 24
  let step2_eq := (salt3 / (water3 + 24)) = 0.4
  let water4 := water3 - (1/4) * water3
  let salt4 := salt3
  let water5 := water4 + 15
  let salt5 := salt4 + 30
  let step4_eq := (salt5 / (water5 + 30)) = 0.5
  step2_eq ∧ step4_eq → x = 192 :=
by
  sorry

end tank_saltwater_solution_l572_572484


namespace asymptotes_of_hyperbola_l572_572191

theorem asymptotes_of_hyperbola (y x : ℝ) (h : y^2 - 2 * x^2 = 8) :
  y = sqrt 2 * x ∨ y = - sqrt 2 * x :=
sorry

end asymptotes_of_hyperbola_l572_572191


namespace tan_product_pi_nine_l572_572128

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572128


namespace divisors_not_divisible_by_3_eq_6_l572_572658

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572658


namespace average_salary_correct_l572_572841

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 16000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 8800 := by
  sorry

end average_salary_correct_l572_572841


namespace reassembleToSquare_l572_572043

-- Define the original grid size
def gridSize : ℕ := 5

-- Define the type for the cells in the grid
structure Cell where
  x : ℕ
  y : ℕ
  deriving DecidableEq, Repr

-- Define the black cells that were removed
def blackCells : List Cell := [{x := 1, y := 2}, {x := 3, y := 4}]

-- Define function to check if a cell is occupied (white)
def isWhite (c : Cell) : Prop :=
  c ∉ blackCells

-- Define the shape after removing two black cells
def validShapeCells : List Cell :=
  [c | c ← List.range (gridSize^2), let rc := {x := c % gridSize, y := c / gridSize}, isWhite rc]

-- Define the target shape: a 5x5 grid with the cells removed
def targetShapeSize : ℕ := gridSize^2 - blackCells.length

-- Statement that the shape can be reassembled into a square
theorem reassembleToSquare : 
  ∃ a b : List Cell, 
  a.length + b.length = targetShapeSize ∧ 
  (∀ c, c ∈ a ∨ c ∈ b ↔ c ∉ blackCells) ∧ 
  -- (proof steps omitted) 
  sorry :=
  sorry

end reassembleToSquare_l572_572043


namespace sum_of_digits_0_to_2012_l572_572320

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l572_572320


namespace pascal_elements_sum_l572_572440

theorem pascal_elements_sum :
  (Nat.choose 20 4 + Nat.choose 20 5) = 20349 :=
by
  sorry

end pascal_elements_sum_l572_572440


namespace math_problem_correct_conclusions_l572_572898

def statement_1 : Prop :=
  ∀ (l1 l2 : ℝ) (h : ¬ ∃ p, l1 = p ∧ l2 = p),
  ¬(l1 = l2)

noncomputable def statement_2 : Prop :=
  ∀ (a : ℝ) (h : ∃ m, m > 0 ∧
    (∃ (x y : ℝ), (x^2 + y^2 - a * x + 2 * a * y + 2 * a + 1 = 0) ∧ 
    (2 * x + 1 = 0))),
  a > -3 ∧ a < 2

def statement_3 : Prop :=
  ∀ (x : ℝ) (h : (sin x + cos x = 1 / 5) ∧ (x > -pi/2) ∧ (x < 0)),
  sin x - cos x = -7 / 5

noncomputable def statement_4 : Prop :=
  ∀ (a : ℝ) (h : 
    ∀ x:ℝ, x > 2/3 → (-x^2 + x + 2 * a ≥ 0)),
  a ≥ -1 / 9

def total_correct_conclusions : ℕ := 3

theorem math_problem_correct_conclusions :
  (statement_1 ∧ statement_3 ∧ statement_4) ∧ ¬statement_2 ↔ total_correct_conclusions = 3 := by
  sorry

end math_problem_correct_conclusions_l572_572898


namespace main_theorem_l572_572602

-- Define the function f : ℝ → ℝ 
noncomputable def f (x m : ℝ) : ℝ := 2^|x - m| - 1

-- Define the values a, b, c
def a (m : ℝ) := f (-2) m
def b (m : ℝ) := f (Math.log 2 5) m
def c (m : ℝ) := f (2*m) m

-- The final theorem statement that should be proved
theorem main_theorem (m : ℝ) (ev_fun : ∀ x : ℝ, f x m = f (-x) m) :
  c 0 < a 0 < b 0 :=
sorry

end main_theorem_l572_572602


namespace cos_eq_implies_isosceles_angle_gt_implies_sin_gt_sides_and_angle_implies_unique_tri_sin_squares_implies_obtuse_l572_572458

-- Given \cos A = \cos B, prove that triangle ABC is isosceles
theorem cos_eq_implies_isosceles (A B C: ℝ) (a b c: ℝ):
  cos A = cos B → a = b :=
by sorry

-- Given A > B, prove that \sin A > \sin B
theorem angle_gt_implies_sin_gt (A B: ℝ) (tri_ABC : triangle):
  A > B → sin A > sin B :=
by sorry

-- Given a = 8, c = 10, B = 60^\circ, prove there are not exactly two triangles that satisfy conditions
theorem sides_and_angle_implies_unique_tri (B: ℝ) (a c: ℝ):
  a = 8 → c = 10 → B = 60 → ¬(∃! b, triangle_exists a b c B) :=
by sorry

-- Given \sin^2 A + \sin^2 B < \sin^2 C, prove that triangle ABC is obtuse
theorem sin_squares_implies_obtuse (A B C: ℝ):
  sin A^2 + sin B^2 < sin C^2 → obtuse_triangle A B C :=
by sorry

-- Definitions to use within Lean (not shown here)
-- Statements about triangles: Isosceles, obtuse, existing triangle, etc.
-- These definitions would need to match axioms, theorems or predicates within Mathlib or custom definitions.

end cos_eq_implies_isosceles_angle_gt_implies_sin_gt_sides_and_angle_implies_unique_tri_sin_squares_implies_obtuse_l572_572458


namespace second_player_wins_when_2003_candies_l572_572418

def game_winning_strategy (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 2

theorem second_player_wins_when_2003_candies :
  game_winning_strategy 2003 = 2 :=
by 
  sorry

end second_player_wins_when_2003_candies_l572_572418


namespace range_of_independent_variable_l572_572297

theorem range_of_independent_variable (x : ℝ) : 
  (y = 3 / (x + 2)) → (x ≠ -2) :=
by
  -- suppose the function y = 3 / (x + 2) is given
  -- we need to prove x ≠ -2 for the function to be defined
  sorry

end range_of_independent_variable_l572_572297


namespace simplify_and_evaluate_expression_l572_572815

open Real

theorem simplify_and_evaluate_expression :
  ∀ (a b : ℝ), (|a - 2| + (b - 1/2)^2 = 0) →
  2 * (a^2 * b - 3 * a * b^2) - [5 * a^2 * b - 3 * (2 * a * b^2 - a^2 * b) - 2] = -10 :=
by
  intros a b h
  have h1 : a = 2 := by 
    sorry -- Proof that |a - 2| = 0 implies a = 2
  have h2 : b = 1 / 2 := by
    sorry -- Proof that (b - 1 / 2)^2 = 0 implies b = 1 / 2
  rw [h1, h2] -- Substitute a and b into the expression
  calc
    2 * (2^2 * (1/2) - 3 * 2 * (1/2)^2) - 
    (5 * 2^2 * (1/2) - 3 * (2 * 2 * (1/2)^2 - 2^2 * (1/2)) - 2)
    = 2 * (4 * (1/2) - 3 * 2 * (1/4)) - (5 * 4 * (1/2) - 3 * (2 * 2 * (1/4) - 4 * (1/2)) - 2)
    := by sorry -- Simplify manually
  -- Continue simplifying step-by-step as represented in the solution above, reaching:
  = -10 := by sorry

end simplify_and_evaluate_expression_l572_572815


namespace number_of_regions_l572_572558

theorem number_of_regions (n : ℕ) (h_parallel : ∀ i j, i ≠ j → ¬parallel (lines i) (lines j))
                           (h_concurrent : ¬∃ p, ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → concurrent p (lines i) (lines j) (lines k)) :
  number_of_regions 8 = 37 := 
sorry

end number_of_regions_l572_572558


namespace SA_l572_572752

noncomputable def circle (O : Point) (r : ℝ) (A : Point) : Prop := dist O A = r

def perpendicular (A B C : Point) : Prop :=
∃ D, ∠ABC = 90 ∧ ∠ABD = 0 ∧ ∠DBC = 90

def symmetric (A B : Point) (l : Line) : Prop := reflection l A = B

def tangent (ℓ : Line) (C : Point) (A : Point) : Prop :=
∀ P, (P ≠ A ∧ C ∈ ℓ) → dist A P = dist P C

def power_of_point (S N A' B' : Point) : Prop :=
dist S N ^ 2 = dist S A' * dist S B'

theorem SA'_SB'_eq_SN_sq (O N S A B A' B' : Point) (r : ℝ) (ℓ : Line) :
  circle O r A ∧ circle O r B ∧
  perpendicular N S O ∧ perpendicular O W E ∧
  tangent ℓ S A ∧ symmetric A B W ∧
  intersects ℓ (NA) A' ∧ intersects ℓ (NB) B' →
  power_of_point S N A' B' :=
by sorry

end SA_l572_572752


namespace sum_of_distances_l572_572757

theorem sum_of_distances (A B C : ℝ × ℝ) (hA : A.2^2 = 8 * A.1) (hB : B.2^2 = 8 * B.1) 
(hC : C.2^2 = 8 * C.1) (h_centroid : (A.1 + B.1 + C.1) / 3 = 2) : 
  dist (2, 0) A + dist (2, 0) B + dist (2, 0) C = 12 := 
sorry

end sum_of_distances_l572_572757


namespace final_score_l572_572006

theorem final_score (Score1 Score2 Score3 : ℕ) (h1 : Score1 = 23) (h2 : Score2 = 34) (h3 : Score3 = 29) : Score1 + Score2 + Score3 = 86 :=
by
  rw [h1, h2, h3]
  rfl

end final_score_l572_572006


namespace inequality_af_a_gt_bf_b_l572_572623

variable (f : ℝ → ℝ)
variable (a b : ℝ)

theorem inequality_af_a_gt_bf_b 
  (h_diff : ∀ x, differentiable_at ℝ f x)
  (h_inequality : ∀ x, x * (deriv f x) > -f x)
  (h_constants : a > b) : a * (f a) > b * (f b) :=
sorry

end inequality_af_a_gt_bf_b_l572_572623


namespace probability_of_two_green_balls_l572_572968

theorem probability_of_two_green_balls :
  ∀ (balls : List Char) (green : Char) (red : Char),
    balls = ['g', 'g', 'r'] →
    (∃ drawn : List Char, drawn = ['g', 'g'] ∧
      (probability (draw 2 balls) (λ drawn, is_two_green drawn)) = 1 / 3) :=
by
  intro balls green red h1
  have h2 : List Char := ['g', 'g', 'r']
  have h3 : ∃ drawn : List Char, drawn = ['g', 'g']
  sorry

end probability_of_two_green_balls_l572_572968


namespace sum_b_l572_572584

def b (p : ℕ) : ℕ :=
  if h : p > 0 then
    let sqrt_p := Real.sqrt p in
    let k := Nat.ceil sqrt_p in
    if Real.abs (k - sqrt_p) < 1 / 3 then k else sorry
  else 0  -- For the case when p = 0 which is not used in our specific setting

theorem sum_b (S : ℕ) (h : S = ∑ p in Finset.range 3000, b (p + 1)) : S = 69045 := sorry

end sum_b_l572_572584


namespace initial_candies_l572_572179

theorem initial_candies (L R : ℕ) (h1 : L + R = 27) (h2 : R - L = 2 * L + 3) : L = 6 ∧ R = 21 :=
by
  sorry

end initial_candies_l572_572179


namespace limit_funct_l572_572986

def funct (x : ℝ) : ℝ := (2^x - 1) / (Real.log (1 + 2 * x))

theorem limit_funct : Filter.Tendsto funct (Filter.nhds 0) (Filter.nhds (Real.log 2 / 2)) :=
  by
  sorry

end limit_funct_l572_572986


namespace childrens_meal_cost_l572_572507

variable (x : ℕ) -- The cost of each children's meal in dollars
variable (A C : ℕ) -- Number of adults and children
variable (cost_adultmeal cost_soda total_bill : ℕ) -- Costs and total bill in dollars

-- Conditions from the problem
def adults := 6
def children := 2
def cost_adult_meal := 6
def cost_soda := 2
def total_bill := 60

-- Hypothesis involving the actual equality to be proved
theorem childrens_meal_cost :
  (adults * cost_adult_meal) + (children * x) + ((adults + children) * cost_soda) = total_bill →
  x = 4 :=
by
  sorry

end childrens_meal_cost_l572_572507


namespace tan_identity_proof_l572_572100

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572100


namespace product_mod_five_remainder_l572_572580

theorem product_mod_five_remainder :
  (114 * 232 * 454 * 454 * 678) % 5 = 4 := by
  sorry

end product_mod_five_remainder_l572_572580


namespace perimeter_of_given_triangle_l572_572928

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572928


namespace calculate_exp_l572_572065

noncomputable def exp1 : ℝ := (9/4)^(1/2)
noncomputable def exp2 : ℝ := (-2016)^0
noncomputable def exp3 : ℝ := (27/8)^(-2/3)
noncomputable def exp4 : ℝ := (3/2)^(-2)

theorem calculate_exp :
  exp1 - exp2 - exp3 + exp4 = 1/2 := sorry

end calculate_exp_l572_572065


namespace dakotas_medical_bill_l572_572546

variable (days_in_hospital : ℕ) (bed_cost_per_day : ℕ) (specialist_cost_per_hour : ℕ) (specialist_time_in_hours : ℚ) (num_specialists : ℕ) (ambulance_cost : ℕ)

theorem dakotas_medical_bill 
  (h1 : days_in_hospital = 3) 
  (h2 : bed_cost_per_day = 900)
  (h3 : specialist_cost_per_hour = 250)
  (h4 : specialist_time_in_hours = 0.25)
  (h5 : num_specialists = 2)
  (h6 : ambulance_cost = 1800) : 

  let bed_total := bed_cost_per_day * days_in_hospital,
      specialists_total := (specialist_cost_per_hour * specialist_time_in_hours * num_specialists).toNat,
      total_cost := bed_total + specialists_total + ambulance_cost
  in 
  total_cost = 4750 := 
by 
  sorry

end dakotas_medical_bill_l572_572546


namespace water_volume_in_second_solution_l572_572386

/-- Samantha needs to prepare two different solutions for her Chemistry class.
Given that the total volume needed for the second solution is 0.75 liters 
and the volume ratio of chemical B to water is 1:2, 
prove that the volume of water used will be 0.5 liters. -/
theorem water_volume_in_second_solution (total_volume : ℝ) (ratio_B_to_water : ℝ)
  (h_total_volume : total_volume = 0.75)
  (h_ratio_B_to_water : ratio_B_to_water = 2) :
  let x := total_volume / (1 + ratio_B_to_water) in
  (ratio_B_to_water * x) = 0.5 := by
  sorry

end water_volume_in_second_solution_l572_572386


namespace simplify_and_evaluate_l572_572816

theorem simplify_and_evaluate (y : ℚ) (hy : y = 1/2) : 
  (y + 1) * (y - 1) + (2 * y - 1) * (2 * y - 1) - 2 * y * (2 * y - 1) = -3/4 := 
by
  rw hy
  -- Sorry to be completed
  sorry

end simplify_and_evaluate_l572_572816


namespace complex_fraction_simplification_l572_572521

open Complex

theorem complex_fraction_simplification : (1 + 2 * I) / (1 - I)^2 = 1 - 1 / 2 * I :=
by
  -- Proof omitted
  sorry

end complex_fraction_simplification_l572_572521


namespace count_divisors_not_divisible_by_3_l572_572697

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572697


namespace tan_product_l572_572131

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572131


namespace inequality_of_sums_equality_condition_l572_572010

theorem inequality_of_sums (n : ℕ) (a b : list ℝ) (h_len : a.length = n) (h_len2 : b.length = n) (h_nonneg_a : ∀ i < n, 0 ≤ a.nth_le i h_len) (h_nonneg_b : ∀ i < n, 0 ≤ b.nth_le i h_len2) : 
(∑ i in finset.range n, real.sqrt ((a.nth_le i (finset.mem_range.mp sorry))^2 + (b.nth_le i (finset.mem_range.mp sorry))^2)) ≥ 
real.sqrt ((∑ i in finset.range n, a.nth_le i (finset.mem_range.mp sorry))^2 + (∑ i in finset.range n, b.nth_le i (finset.mem_range.mp sorry))^2) :=
sorry

theorem equality_condition (n : ℕ) (a b : list ℝ) (h_len : a.length = n) (h_len2 : b.length = n) :
(∀ i < n, a.nth_le i h_len / b.nth_le i h_len2 = a.nth_le 0 h_len / b.nth_le 0 h_len2) ↔
(∑ i in finset.range n, real.sqrt ((a.nth_le i (finset.mem_range.mp sorry))^2 + (b.nth_le i (finset.mem_range.mp sorry))^2)) = 
real.sqrt ((∑ i in finset.range n, a.nth_le i (finset.mem_range.mp sorry))^2 + (∑ i in finset.range n, b.nth_le i (finset.mem_range.mp sorry))^2) :=
sorry

end inequality_of_sums_equality_condition_l572_572010


namespace necessary_but_not_sufficient_condition_for_a_eq_1_l572_572350

theorem necessary_but_not_sufficient_condition_for_a_eq_1 
  (a : ℝ) 
  (A : set ℝ := {x | x ≤ 1}) 
  (B : set ℝ := {x | x ≥ a}) 
  (H : A ∪ B = set.univ) : 
  (a ≤ 1) ∧ ¬((a ≤ 1) → (a = 1)) :=
sorry

end necessary_but_not_sufficient_condition_for_a_eq_1_l572_572350


namespace smallest_possible_positive_value_l572_572524

theorem smallest_possible_positive_value (l w : ℕ) (hl : l > 0) (hw : w > 0) : ∃ x : ℕ, x = w - l + 1 ∧ x = 1 := 
by {
  sorry
}

end smallest_possible_positive_value_l572_572524


namespace circle_tangent_to_line_l572_572798

variable {A B C O D M P : Type}

-- Assuming point structures with necessary properties
variables [Point O] [Point A] [Point B] [Point C] [Point D] [Point M] [Point P]

-- Conditions from the problem
def is_circumcenter (O A B C : Type) : Prop := sorry
def is_angle_bisector_intersection (A B C D : Type) : Prop := sorry
def is_perpendicular_lines (M BC : Type) : Prop := sorry
def is_perpendicular_lines2 (M A AD : Type) : Prop := sorry
def intersect (BM OA P : Type) : Prop := sorry

-- Problem statement
theorem circle_tangent_to_line (h1 : is_circumcenter O A B C)
    (h2 : is_angle_bisector_intersection A B C D)
    (h3 : is_perpendicular_lines M BC)
    (h4 : is_perpendicular_lines2 M A AD)
    (h5 : intersect BM OA P) :
    ∃ (circle : Type), (center circle = P) ∧ (A ∈ circle) ∧ tangent circle BC := sorry

end circle_tangent_to_line_l572_572798


namespace last_two_digits_seq_2007_l572_572548

def seq (n : ℕ) : ℕ :=
  Nat.recOn n 7 (λ n a_n_minus_1, 7 ^ a_n_minus_1)

theorem last_two_digits_seq_2007 : seq 2007 % 100 = 43 := sorry

end last_two_digits_seq_2007_l572_572548


namespace smallest_positive_period_intervals_of_monotonicity_l572_572254
noncomputable def f (x : ℝ) := sin (π - x) * sin (π / 2 - x) + cos x ^ 2

theorem smallest_positive_period : ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π :=
sorry

theorem intervals_of_monotonicity {x : ℝ} (h : -π / 8 ≤ x ∧ x ≤ 3 * π / 8) : 
  ((-π / 8 ≤ x ∧ x ≤ π / 8) → monotone_on f (Icc (-π / 8) (π / 8))) ∧
  ((π / 8 ≤ x ∧ x ≤ 3 * π / 8) → antitone_on f (Icc (π / 8) (3 * π / 8))) :=
sorry

end smallest_positive_period_intervals_of_monotonicity_l572_572254


namespace ellipse_equation_l572_572971

theorem ellipse_equation (a b : ℝ) (e : ℝ) (area : ℝ) 
  (h_major_axis : True) -- condition: major axis along the coordinate axis
  (h_foci_on_x_axis : True) -- condition: foci on the x-axis
  (h_eccentricity : e = (√3) / 2)  -- condition: eccentricity is √3 / 2
  (h_area : area = 8 * π)  -- condition: area is 8π
  : (a = 4) ∧ (b = 2) → (∀ x y : ℝ, ((x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ (x^2 / 16 + y^2 / 4 = 1))) :=
by
  sorry -- The proof is omitted

end ellipse_equation_l572_572971


namespace rationalize_denominator_l572_572804

theorem rationalize_denominator (a b : ℝ) (h1 : a = 2) (h2 : b = sqrt 2) : 
  1 / (a - b) = (a + b) / 2 :=
by
  sorry

end rationalize_denominator_l572_572804


namespace least_number_to_subtract_l572_572011

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (m : ℕ) : n = 62575 ∧ d = 99 → n % d = m → m = 43 :=
by
  intros h₁ h₂
  cases h₁ with hn hd
  rw [hn, hd] at h₂
  exact h₂

end least_number_to_subtract_l572_572011


namespace correct_option_is_B_l572_572000

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l572_572000


namespace find_b_l572_572240

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if h : 0 < x ∧ x < 2
  then log (x^2 - x + b)
  else sorry  -- other definitions based on f(-x) = -f(x) = f(4 - x)

theorem find_b (b : ℝ) (f_zero_points : ℝ → Prop) :
  (∀ x, x ∈ Icc (-2 : ℝ) 2 → f x b = 0 → f_zero_points x) ∧
  (∀ x ∈ Ioo (0 : ℝ) 2, f x b = log (x^2 - x + b)) ∧
  (∀ x, f (-x) b = -f x b = f (4 - x) b) ∧ 
  (∃ z1 z2 z3 z4 z5 ∈ Icc (-2 : ℝ) 2, f_zero_points z1 ∧ f_zero_points z2 ∧ 
    f_zero_points z3 ∧ f_zero_points z4 ∧ f_zero_points z5 ∧ 
    z1 ≠ z2 ∧ z2 ≠ z3 ∧ z3 ≠ z4 ∧ z4 ≠ z5) → 
  (1/4 < b ∧ b ≤ 1) ∨ (b = 5/4) := 
sorry

end find_b_l572_572240


namespace julian_legos_count_l572_572751

def legos_each_airplane : Nat := 240
def additional_legos_needed : Nat := 80
def total_legos_needed_for_two_planes := legos_each_airplane * 2

theorem julian_legos_count :
  ∀ (legos_each_airplane : Nat) (additional_legos_needed : Nat) (total_legos_needed_for_two_planes : Nat),
    legos_each_airplane = 240 → 
    additional_legos_needed = 80 → 
    total_legos_needed_for_two_planes = legos_each_airplane * 2 →
    (total_legos_needed_for_two_planes - additional_legos_needed = 400) :=
by
  intros
  simp_all
  sorry

end julian_legos_count_l572_572751


namespace roses_equal_tulips_l572_572728

variable (C V T R : ℕ)

-- Given conditions
def condition1 : Prop := V = 1 / 3 * C
def condition2 : Prop := T = 1 / 4 * V
def condition3 : Prop := C = 2 / 3 * (C + V + T + R)

-- The theorem to prove
theorem roses_equal_tulips (h1 : condition1) (h2 : condition2) (h3 : condition3) : R = T := by
  sorry

end roses_equal_tulips_l572_572728


namespace words_on_each_page_l572_572027

theorem words_on_each_page (p : ℕ) (h : 150 * p ≡ 198 [MOD 221]) : p = 93 :=
sorry

end words_on_each_page_l572_572027


namespace simplify_expression_l572_572062

theorem simplify_expression (a : ℤ) : 7 * a - 3 * a = 4 * a :=
by
  sorry

end simplify_expression_l572_572062


namespace triangle_sides_solution_l572_572618

noncomputable def solve_triangle_sides (a b c : ℝ) (A : ℝ) (area : ℝ) : Prop :=
  (c = 2) ∧
  (A = 60) ∧
  (area = (Real.sqrt 3) / 2) ∧
  (1 / 2 * b * 2 * Real.sin (A * Real.pi / 180) = area) ∧
  (a * a = b * b + c * c - 2 * b * c * Real.cos (A * Real.pi / 180)) ∧
  (a = Real.sqrt 3) ∧
  (b = 1)

-- Prove that given the conditions, the values of a and b are sqrt(3) and 1, respectively.
theorem triangle_sides_solution : ∀ a b c A area,
  solve_triangle_sides a b c A area :=
begin
  intros,
  sorry -- Proof not required
end

end triangle_sides_solution_l572_572618


namespace right_triangle_perimeter_l572_572950

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572950


namespace greg_books_difference_l572_572787

theorem greg_books_difference (M K G X : ℕ)
  (hM : M = 32)
  (hK : K = M / 4)
  (hG : G = 2 * K + X)
  (htotal : M + K + G = 65) :
  X = 9 :=
by
  sorry

end greg_books_difference_l572_572787


namespace angle_SPC_l572_572859

theorem angle_SPC
  (P Q R S : Type)
  (triangle_PQR_isosceles : ∀ (P Q R : Type), P ≠ Q → P ≠ R → Q ≠ R → isosceles_triangle P Q R)
  (triangle_PRS_isosceles : ∀ (P R S : Type), P ≠ R → P ≠ S → R ≠ S → isosceles_triangle P R S)
  (PQ_eq_QR : PQ = QR)
  (PR_eq_RS : PR = RS)
  (S_inside_triangle_PQR : point_inside_triangle S P Q R)
  (angle_PQR_eq : angle P Q R = 50)
  (angle_PRS_eq : angle P R S = 110)
  : angle S P C = 30 := sorry

end angle_SPC_l572_572859


namespace complex_number_in_second_quadrant_l572_572629

noncomputable theory

open Complex

def complex_number : ℂ := (cos (Real.pi / 4) + sin (Real.pi / 4) * Complex.I) ^ 2 / (1 - Complex.I)

theorem complex_number_in_second_quadrant : 
  Re complex_number < 0 ∧ Im complex_number > 0 := 
sorry

end complex_number_in_second_quadrant_l572_572629


namespace set_C_is_basis_l572_572498

variables (e1 e2 : ℝ × ℝ)

def is_basis_set_C :=
  e1 = (1, -2) ∧ e2 = (2, 3) ∧ 
  (∀ (k : ℝ), e2 ≠ k • e1) ∧ 
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0)

theorem set_C_is_basis (e1 e2 : ℝ × ℝ) : 
  is_basis_set_C e1 e2 :=
by 
  sorry

end set_C_is_basis_l572_572498


namespace matrix_commutative_fraction_zero_l572_572761

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![4, 5]
]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, y],
  ![z, w]
]

-- The main statement for the problem
theorem matrix_commutative_fraction_zero {x y z w : ℝ} (h1 : A.mul (B x y z w) = (B x y z w).mul A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 0 := by
  sorry

end matrix_commutative_fraction_zero_l572_572761


namespace coordinates_of_OA_l572_572223

-- Defining the conditions
def angle_negative_135_degrees : Prop := 
  -- Condition indicating point A is on the terminal side of -135°
  angle_in_standard_position (-135) ⟹ third_quadrant

def magnitude_sqrt_2 (A : ℝ × ℝ) :=
  -- Condition indicating the magnitude of vector OA is sqrt(2)
  ∥A∥ = sqrt(2)

-- Conclusion coordinates of vector OA are (-1, -1)
theorem coordinates_of_OA (A : ℝ × ℝ) (h1 : angle_negative_135_degrees) (h2 : magnitude_sqrt_2 A) :
  A = (-1, -1) :=
sorry

end coordinates_of_OA_l572_572223


namespace veronica_max_area_l572_572594

noncomputable def max_area_garden : ℝ :=
  let l := 105
  let w := 420 - 2 * l
  l * w

theorem veronica_max_area : ∃ (A : ℝ), max_area_garden = 22050 :=
by
  use 22050
  show max_area_garden = 22050
  sorry

end veronica_max_area_l572_572594


namespace count_divisors_not_divisible_by_3_l572_572700

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572700


namespace total_ticket_cost_l572_572960

theorem total_ticket_cost (x y : ℕ) 
  (h1 : x + y = 380) 
  (h2 : y = x + 240) 
  (cost_orchestra : ℕ := 12) 
  (cost_balcony : ℕ := 8): 
  12 * x + 8 * y = 3320 := 
by 
  sorry

end total_ticket_cost_l572_572960


namespace distinct_four_digit_numbers_count_l572_572648

def is_valid_four_digit_number (n : ℕ) : Prop :=
  let digits := [5, 0, 0, 6]
  let num_str := n.toString
  let num_digits := num_str.data.map (λ c => c.to_nat - 48) -- Convert char to corresponding digit
  (num_digits.length = 4) ∧
  (num_digits.head ≠ 0) ∧
  (∀ d ∈ num_digits, d ∈ digits) ∧
  (multiset.card (list.to_multiset num_digits) = multiset.card (list.to_multiset digits))

theorem distinct_four_digit_numbers_count : 
  ∃ (n : ℕ), 6 = n ∧ 
  (∃ s : finset ℕ, s.card = n ∧ (∀ x ∈ s, is_valid_four_digit_number x)) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l572_572648


namespace two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l572_572204

def R (n : ℕ) : ℕ := 
  let remainders := List.range' 2 11 |>.map (λ k => n % k)
  remainders.sum

theorem two_digit_integers_satisfy_R_n_eq_R_n_plus_2 :
  let two_digit_numbers := List.range' 10 89
  (two_digit_numbers.filter (λ n => R n = R (n + 2))).length = 2 := 
by
  sorry

end two_digit_integers_satisfy_R_n_eq_R_n_plus_2_l572_572204


namespace tan_product_l572_572081

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572081


namespace always_space_for_additional_domino_l572_572371

theorem always_space_for_additional_domino
  (board : Finset (Fin 36))
  (placement : Finset (Finset (Fin 36)))
  (h1 : ∀ d ∈ placement, d.card = 2) -- each domino covers exactly 2 squares
  (h2 : placement.card = 11) -- 11 dominos are placed
  (h3 : ∀ d ∈ placement, d ⊆ board) -- each placed domino is within the board
  (h4 : board.card = 36) -- the board has 36 squares
  : ∃ new_d, new_d.card = 2 ∧ new_d ⊆ board ∧ ∀ d ∈ placement, new_d ∩ d = ∅ := 
sorry

end always_space_for_additional_domino_l572_572371


namespace rectangular_solid_surface_area_l572_572556

-- Definitions based on conditions
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rectangular_solid (a b c : ℕ) :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a * b * c = 231

noncomputable def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- Main theorem based on question and answer
theorem rectangular_solid_surface_area :
  ∃ (a b c : ℕ), rectangular_solid a b c ∧ surface_area a b c = 262 := by
  sorry

end rectangular_solid_surface_area_l572_572556


namespace park_area_l572_572926

theorem park_area (w : ℝ) (h1 : 2 * (w + 3 * w) = 72) : w * (3 * w) = 243 :=
by
  sorry

end park_area_l572_572926


namespace minimum_additional_coins_needed_l572_572494

def sum_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem minimum_additional_coins_needed (friends : ℕ) (current_coins : ℕ) :
  friends = 15 → current_coins = 63 → 
  let required_coins := sum_natural_numbers friends in
  let additional_coins := required_coins - current_coins in
  additional_coins = 57 :=
by
  intros h_friends h_coins
  rw [h_friends, h_coins]
  let required_coins := sum_natural_numbers 15
  have h_required_coins : required_coins = 120 := by
    rw [sum_natural_numbers, Nat.mul, Nat.add, /, 2]
    norm_num
  
  let additional_coins := required_coins - 63
  have h_additional_coins : additional_coins = 57 := by
    rw [h_required_coins]
    norm_num
    
  exact h_additional_coins

end minimum_additional_coins_needed_l572_572494


namespace train_a_speed_correct_l572_572864

-- Definitions and conditions
variables (v_A : ℝ) (distanceA distanceB : ℝ)

-- Condition 1: Train A travels at a constant speed v_A
def train_a_speed := v_A

-- Condition 2: Train B travels at a constant speed 80 mph
def train_b_speed := 80

-- Condition 3: Train A has already traveled v_A * 1/2 miles by the time Train B leaves
def distance_travel_a_before_b_leaves := v_A * (1 / 2)

-- Condition 4: In 50 minutes (5/6 hours), the distance covered by Train B: 80 * 5/6 miles
def distance_travel_b_in_50_minutes := (80 * (5 / 6))

-- Proof goal: When Train B overtakes Train A, both have traveled the same distance
theorem train_a_speed_correct (h : distance_travel_a_before_b_leaves + v_A * (5 / 6) = distance_travel_b_in_50_minutes) : v_A = 50 :=
 by sorry
 
end train_a_speed_correct_l572_572864


namespace divisors_not_divisible_by_3_l572_572681

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572681


namespace part_a_part_b_part_c_l572_572017

-- Part (a)
theorem part_a : ∃ a b, a * b = 80 ∧ (a = 8 ∨ a = 4) ∧ (b = 10 ∨ b = 5) :=
by sorry

-- Part (b)
theorem part_b : ∃ a b c, (a * b) / c = 50 ∧ (a = 10 ∨ a = 5) ∧ (b = 10 ∨ b = 5) ∧ (c = 2 ∨ c = 1) :=
by sorry

-- Part (c)
theorem part_c : ∃ n, n = 4 ∧ ∀ a b c, (a + b) / c = 23 :=
by sorry

end part_a_part_b_part_c_l572_572017


namespace least_value_of_z_minus_x_l572_572452

theorem least_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : even x) 
  (hy : odd y) 
  (hz : odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) 
  : z - x = 9 := 
sorry

end least_value_of_z_minus_x_l572_572452


namespace perimeter_of_given_triangle_l572_572931

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572931


namespace valid_a_l572_572774

noncomputable def remainder (k p : ℕ) : ℕ := k % p

theorem valid_a (p : ℕ) (hp : p ≥ 5) (hprime : Prime p) :
  ∀ a, (∀ m ∈ List.range (p - 1), 
    let R := remainder (m * a) p in
    m + R > a) ↔ 
    (a = p - 1 ∨ ∃ s, 1 ≤ s ∧ s ≤ p - 1 ∧ a = p / s) :=
sorry

end valid_a_l572_572774


namespace marble_202_is_green_l572_572047

def marble_pattern := [("red", 6), ("green", 5), ("blue", 4)]
def total_marbles := 202

def marble_color(n : ℕ, pattern : List (String × ℕ)) : String :=
  let cycle_length := pattern.foldl (fun acc x => acc + x.2) 0
  let pos_in_cycle := n % cycle_length
  let rec find_color (pos : ℕ) (p : List (String × ℕ)) :=
    match p with
    | []          => "unknown"
    | (color, count) :: ps => if pos ≤ count then color else find_color (pos - count) ps
  find_color pos_in_cycle pattern

theorem marble_202_is_green : marble_color total_marbles marble_pattern = "green" := by
  sorry

end marble_202_is_green_l572_572047


namespace min_period_cos2_omega_l572_572593

theorem min_period_cos2_omega (ω : ℝ) (hω : ω > 0) (h_period : (∀ x, (cos (2 * ω * x)) = (cos (2 * ω * (x + π / ω / 2))))):
  ω = 2 :=
by
  sorry

end min_period_cos2_omega_l572_572593


namespace max_rational_sums_is_1250_l572_572370

/-- We define a structure to represent the problem's conditions. -/
structure GridConfiguration where
  grid_rows : Nat
  grid_cols : Nat
  total_numbers : Nat
  rational_count : Nat
  irrational_count : Nat
  (h_grid : grid_rows = 50)
  (h_grid_col : grid_cols = 50)
  (h_total_numbers : total_numbers = 100)
  (h_rational_count : rational_count = 50)
  (h_irrational_count : irrational_count = 50)

/-- We define a function to calculate the number of rational sums in the grid. -/
def max_rational_sums (config : GridConfiguration) : Nat :=
  let x := config.rational_count / 2 -- rational numbers to the left
  let ni := 2 * x * x - 100 * x + 2500
  let rational_sums := 2500 - ni
  rational_sums

/-- The theorem stating the maximum number of rational sums is 1250. -/
theorem max_rational_sums_is_1250 (config : GridConfiguration) : max_rational_sums config = 1250 :=
  sorry

end max_rational_sums_is_1250_l572_572370


namespace unit_prices_min_basketballs_purchase_schemes_l572_572035

theorem unit_prices (x : ℝ) (y : ℝ) : 
  let basket_unit_price : ℝ := y + 30 in
  basket_unit_price = 120 ∧ x = 90 :=
by
  have h1 : 360 / x = 480 / y,
    from sorry,
  have h2 : x = 90 ∧ y = 90,
    from sorry,
  exact ⟨h2.2 + 30, h2.1⟩

theorem min_basketballs (m : ℕ) : 
  let soccer_balls_sold := (1/3 : ℝ) * m + 10 in
  let profit := 30 * m + 20 * soccer_balls_sold in
  profit > 1300 → m ≥ 33 :=
by
  intro h,
  sorry

theorem purchase_schemes (n_basketballs : ℕ) (n_soccerballs : ℕ) : 
  let total_cost := 120 * n_basketballs + 90 * n_soccerballs in
  let total_balls := 100 in
  let eq1 := total_cost ≤ 10350 in
  let eq2 := n_basketballs + n_soccerballs = total_balls in
  let schemax := (43:ℕ) ≤ n_basketballs ∧ n_basketballs ≤ (45:ℕ) in
  eq1 ∧ eq2 ∧ schemax →
    ∃ k : ℕ, k = 3 ∧ (n_basketballs, n_soccerballs) = (45, 55) :=
by
  intro h,
  sorry

end unit_prices_min_basketballs_purchase_schemes_l572_572035


namespace remaining_rectangle_area_l572_572505

theorem remaining_rectangle_area (s a b : ℕ) (hs : s = a + b) (total_area_cut : a^2 + b^2 = 40) : s^2 - 40 = 24 :=
by
  sorry

end remaining_rectangle_area_l572_572505


namespace correct_conclusions_sequence_l572_572519

-- Definitions for each condition
def lines_perpendicular_to_same_line_are_parallel : Prop := False
def lines_perpendicular_to_same_plane_are_parallel : Prop := True
def planes_perpendicular_to_same_line_are_parallel : Prop := True
def planes_perpendicular_to_same_plane_are_parallel : Prop := False

-- The final question to prove
theorem correct_conclusions_sequence :
  (lines_perpendicular_to_same_line_are_parallel = False) ∧
  (lines_perpendicular_to_same_plane_are_parallel = True) ∧
  (planes_perpendicular_to_same_line_are_parallel = True) ∧
  (planes_perpendicular_to_same_plane_are_parallel = False) →
  ("Sequence is (2)(3)") :=
by
  intro h
  trivial
  sorry

end correct_conclusions_sequence_l572_572519


namespace tan_product_equals_three_l572_572166

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572166


namespace students_created_one_problem_l572_572461

/- Conditions -/
def n_i : ℕ → ℕ := sorry   -- Function that assigns the number of students per course
def x_i : ℕ → ℕ := sorry   -- Function that assigns the problems per course for a student
def students := 30
def total_problems := 40

/- Definitions used in Lean 4 statement -/
def distinct (α : ℕ → ℕ) : Prop := ∀ i j, i ≠ j → α i ≠ α j

/- Lean 4 theorem stating our proof problem -/
theorem students_created_one_problem :
  (∃ (n : ℕ → ℕ), (∀ i < 5, n i > 0) ∧ (∑ i in finset.range 5, n i * x_i i = 40) ∧ (∑ i in finset.range 5, n i = 30) ∧ distinct x_i) →
  (∃ k, k = 26 ∧ (number of students who created exactly one problem = k)) :=
by
  sorry

end students_created_one_problem_l572_572461


namespace remaining_bananas_l572_572550

def original_bananas : ℕ := 46
def removed_bananas : ℕ := 5

theorem remaining_bananas : original_bananas - removed_bananas = 41 := by
  sorry

end remaining_bananas_l572_572550


namespace probability_single_solution_l572_572424

theorem probability_single_solution :
  let outcomes := (finset.univ : finset (ℕ × ℕ)).filter (λ p, 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6)
  ∃ a b : ℕ, a ∈ finset.range(7) ∧ b ∈ finset.range(7) ∧
    (a ≠ 0) ∧ (b ≠ 0) ∧ (b ≠ 2 * a) →
  (let solution_count := outcomes.filter
        (λ p, (p.1 ≠ 0) ∧ (p.2 ≠ 0) ∧ (p.2 ≠ 2 * p.1)).card in
  solution_count / outcomes.card = (11 / 12 : ℚ)) :=
by sorry

end probability_single_solution_l572_572424


namespace sum_digits_0_to_2012_l572_572316

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_0_to_2012 : ∑ n in Finset.range 2013, sum_of_digits n = 28077 := 
by
  sorry

end sum_digits_0_to_2012_l572_572316


namespace right_angled_triangle_only_B_l572_572003

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l572_572003


namespace tangent_line_at_point_l572_572213

def f (x : ℝ) : ℝ := x * Real.log x

def point : ℝ × ℝ := (1, f 1)

theorem tangent_line_at_point :
  ∃ A B C,
  A ≠ 0 ∧ B ≠ 0 ∧
  (forall x y, (y = f x) → (x - point.1) * A + (y - point.2) * B + C = 0) ∧
  A = 1 ∧ B = -1 ∧ C = -1 :=
begin
  sorry
end

end tangent_line_at_point_l572_572213


namespace sufficient_but_not_necessary_l572_572769

variables {R : Type*} [real R]
variable (f : R → R)
variable (a : R)

theorem sufficient_but_not_necessary (h : ∀ {x : R}, x ≠ a → f x ≠ f a) :
  (∀ x, x ≠ a → f x ≠ f a) ∧ ¬ (∀ x, f x ≠ f a → x ≠ a) :=
by
  sorry

end sufficient_but_not_necessary_l572_572769


namespace inv_25_mod_79_l572_572226

theorem inv_25_mod_79 (h : (5 : ℤ)⁻¹ ≡ 39 [ZMOD 79]) : (25 : ℤ)⁻¹ ≡ 20 [ZMOD 79] :=
sorry

end inv_25_mod_79_l572_572226


namespace day_crew_fraction_correct_l572_572059

-- Given conditions
variables (D W : ℕ)
def night_boxes_per_worker := (5 : ℚ) / 8 * D
def night_workers := (3 : ℚ) / 5 * W

-- Total boxes loaded
def total_day_boxes := D * W
def total_night_boxes := night_boxes_per_worker D * night_workers W

-- Fraction of boxes loaded by day crew
def fraction_loaded_by_day_crew := total_day_boxes D W / (total_day_boxes D W + total_night_boxes D W)

-- Theorem to prove
theorem day_crew_fraction_correct (D W : ℕ) : fraction_loaded_by_day_crew D W = (8 : ℚ) / 11 :=
by
  sorry

end day_crew_fraction_correct_l572_572059


namespace sum_of_digits_of_leading_N_such_that_Q_l572_572333

def Q (N : ℕ) : ℝ := sorry -- Placeholder for the actual probability function Q(N)

noncomputable def least_N_such_that_Q (upper_bound : ℝ) : ℕ :=
  if h : ∃ N, N > 0 ∧ N % 10 = 0 ∧ Q N < upper_bound then
    Nat.find h
  else
    0 -- default case, shouldn't happen under the problem constraints

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_leading_N_such_that_Q :
  sum_of_digits (least_N_such_that_Q (319 / 400)) = 7 :=
sorry

end sum_of_digits_of_leading_N_such_that_Q_l572_572333


namespace sum_digits_0_to_2012_l572_572318

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_0_to_2012 : ∑ n in Finset.range 2013, sum_of_digits n = 28077 := 
by
  sorry

end sum_digits_0_to_2012_l572_572318


namespace symmetric_diff_equality_l572_572267

theorem symmetric_diff_equality (A B K : Set α) (h : A ∆ K = B ∆ K) : A = B := by
sorry

end symmetric_diff_equality_l572_572267


namespace quadrant_of_angle_l572_572591

-- Definitions for conditions
def sin_pos_cos_pos (α : ℝ) : Prop := (Real.sin α) * (Real.cos α) > 0

-- The theorem to prove
theorem quadrant_of_angle (α : ℝ) (h : sin_pos_cos_pos α) : 
  (0 < α ∧ α < π / 2) ∨ (π < α ∧ α < 3 * π / 2) :=
sorry

end quadrant_of_angle_l572_572591


namespace x_minus_q_in_terms_of_q_l572_572708

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h : |x - 3| = q) (hx : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_in_terms_of_q_l572_572708


namespace power_function_decreasing_l572_572882

noncomputable theory

open Real

theorem power_function_decreasing (x : ℝ) (hx : 0 < x) :
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x ^ (-5 * m - 3) < 0) ↔ m = 2 := by
sorry

end power_function_decreasing_l572_572882


namespace real_part_of_z_is_one_l572_572973

-- Define the complex number and the condition
variable {z : ℂ}
constant imaginary_unit : ℂ
@[simp] lemma imaginary_unit_def : imaginary_unit = complex.I := by simp [imaginary_unit_def]

-- Define the condition on z
axiom condition : imaginary_unit * (z + 1) = -3 + 2 * imaginary_unit 

-- State the theorem to be proved
theorem real_part_of_z_is_one : z.re = 1 :=
sorry

end real_part_of_z_is_one_l572_572973


namespace num_divisors_not_divisible_by_three_l572_572663

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572663


namespace ellipse_standard_equation_l572_572239

def ellipse_standard_equation_condition_1 (x y : ℝ) : Prop :=
  (x / 4)^2 + (y / 2)^2 = 1

def ellipse_standard_equation_condition_2 (x y : ℝ) : Prop :=
  (y / 8)^2 + (x / 4)^2 = 1

theorem ellipse_standard_equation (x y : ℝ) :
  ((∃ a b c e : ℝ, a = 4 ∧ e = sqrt 3 / 2 ∧ c = (e * a) ∧ b^2 = a^2 - c^2 ∧ ellipse_standard_equation_condition_1 x y) ∨
  (∃ a b c e : ℝ, b = 4 ∧ e = sqrt 3 / 2 ∧ c = (e * a) ∧ a^2 - c^2 = b^2 ∧ ellipse_standard_equation_condition_2 x y)) :=
sorry

end ellipse_standard_equation_l572_572239


namespace tan_product_l572_572143

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572143


namespace simplify_and_evaluate_expression_l572_572813

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l572_572813


namespace greatest_possible_value_of_a_l572_572535

noncomputable def a (k : ℤ) : ℚ := (-25 * k - 10) / 12

theorem greatest_possible_value_of_a :
    (∃ (a b c k : ℚ),
      b = - 6 * a / 5 ∧
      c = 9 * a / 25 - 1 / 5 ∧
      b + 2 * c = k ∧
      a < 0 ∧ k ∈ ℤ ∧ a = - 5 / 6) :=
by
  use [-(5 / 6), -(6 * (-(5 / 6))) / 5, (9 * (-(5 / 6))) / 25 - 1 / 5, 0]
  sorry

end greatest_possible_value_of_a_l572_572535


namespace AM_GM_Inequality_l572_572778

theorem AM_GM_Inequality (n : ℕ) (x : Fin (n + 1) → ℝ) (hx : ∀ i, 0 < x i) :
  (Finset.univ.sum (λ i, (x i) / (x ((i + 1) % Finset.card Finset.univ))) ^ n) 
  ≥ (Finset.univ.sum (λ i, (x i) / (x ((i + 1) % Finset.card Finset.univ)))) := 
sorry

end AM_GM_Inequality_l572_572778


namespace total_cats_received_l572_572974

-- Defining the constants and conditions
def total_adult_cats := 150
def fraction_female_cats := 2 / 3
def fraction_litters := 2 / 5
def kittens_per_litter := 5

-- Defining the proof problem
theorem total_cats_received :
  let number_female_cats := (fraction_female_cats * total_adult_cats : ℤ)
  let number_litters := (fraction_litters * number_female_cats : ℤ)
  let number_kittens := number_litters * kittens_per_litter
  number_female_cats + number_kittens + (total_adult_cats - number_female_cats) = 350 := 
by
  sorry

end total_cats_received_l572_572974


namespace num_non_divisible_by_3_divisors_l572_572672

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572672


namespace integer_add_results_in_perfect_square_l572_572872

theorem integer_add_results_in_perfect_square (x a b : ℤ) :
  (x + 100 = a^2 ∧ x + 164 = b^2) → (x = 125 ∨ x = -64 ∨ x = -100) :=
by
  intros h
  sorry

end integer_add_results_in_perfect_square_l572_572872


namespace exists_m_l572_572325

open Int

noncomputable def a : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => (⟨2 * (a n) / (a (n + 1))⟩ + ⟨2 * (a (n + 1)) / (a n)⟩) : ℕ

theorem exists_m : ∃ m : ℕ, a m = 4 ∧ (a (m + 1) = 3 ∨ a (m + 1) = 4) :=
sorry

end exists_m_l572_572325


namespace line_BC_eq_l572_572250

def altitude1 (x y : ℝ) : Prop := x + y = 0
def altitude2 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def point_A : ℝ × ℝ := (1, 2)

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_BC_eq (x y : ℝ) :
  (∃ b c : ℝ × ℝ, altitude1 b.1 b.2 ∧ altitude2 c.1 c.2 ∧
                   line_eq 2 3 7 b.1 b.2 ∧ line_eq 2 3 7 c.1 c.2 ∧
                   b ≠ c) → 
    line_eq 2 3 7 x y :=
by sorry

end line_BC_eq_l572_572250


namespace rectangular_solid_surface_area_l572_572555

open Nat

theorem rectangular_solid_surface_area (a b c : ℕ) 
  (h_prime_a : Prime a)
  (h_prime_b : Prime b) 
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 231) :
  2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end rectangular_solid_surface_area_l572_572555


namespace right_triangle_perimeter_l572_572943

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572943


namespace charity_event_fund_raising_l572_572889

theorem charity_event_fund_raising :
  let n := 9
  let I := 2000
  let p := 0.10
  let increased_total := I * (1 + p)
  let amount_per_person := increased_total / n
  amount_per_person = 244.44 := by
  sorry

end charity_event_fund_raising_l572_572889


namespace room_length_exceeds_width_l572_572310

theorem room_length_exceeds_width : ∀ (width length : Real), width = 19.7 → length = 20.25 → length - width = 0.55 :=
by
  intros width length h1 h2
  rw [h1, h2]
  have : 20.25 - 19.7 = 0.55 := by norm_num
  exact this

end room_length_exceeds_width_l572_572310


namespace smaller_circle_radius_l572_572469

open Real

theorem smaller_circle_radius {B1 B3 : ℝ} (h1 : B3 + B1 = 36 * π)
    (h2 : B3 - B1 = 18 * π)
    (h3 : 36 * π = π * 6^2) :
    ∃ r : ℝ, B1 = π * r^2 ∧ r = 3 * sqrt 2 :=
by
    let h : 36 * π = π * 36 := by sorry -- Only necessary for successful statement
    have h4 : B1 = 18 * π := by sorry
    exact ⟨3 * sqrt 2, h4, by sorry⟩

end smaller_circle_radius_l572_572469


namespace arthur_spent_38_l572_572856

noncomputable def arthur_meal_cost : ℝ := 
  let appetizer := 8
  let entree := 20
  let wine := 3 * 2
  let dessert := 6
  let total_no_discount := appetizer + entree + wine + dessert
  let discount := entree / 2
  let total_with_discount := total_no_discount - discount
  let tip := total_no_discount * 0.20
  total_with_discount + tip

theorem arthur_spent_38 :
  arthur_meal_cost = 38 :=
by
  rw [arthur_meal_cost]
  have appetizer : ℝ := 8
  have entree : ℝ := 20
  have wine : ℝ := 3 * 2
  have dessert : ℝ := 6
  have total_no_discount : ℝ := appetizer + entree + wine + dessert
  have discount : ℝ := entree / 2 
  have total_with_discount : ℝ := total_no_discount - discount
  have tip : ℝ := total_no_discount * 0.20
  calc  total_with_discount + tip = (8 + 20 + 6 + 2 * 3) - (20 / 2) + 0.20 * (8 + 20 + 6 + 2 * 3) : by rw [total_no_discount]
                          ... = 38 : by norm_num

end arthur_spent_38_l572_572856


namespace div_neg_cancel_l572_572978

theorem div_neg_cancel (x y : Int) (hx : x ≠ 0) (hy : y ≠ 0) :
  (-x) / (-y) = x / y :=
by
  sorry

example : (-300) / (-25) = 12 :=
by
  have hx : 300 ≠ 0 := by decide
  have hy : 25 ≠ 0 := by decide
  rw [div_neg_cancel 300 25 hx hy]
  norm_num

end div_neg_cancel_l572_572978


namespace constant_term_is_minus_80_l572_572574

noncomputable def constant_term_expansion : ℤ := 
  let c5_3 := (nat.choose 5 3 : ℤ)
  let term_value := c5_3 * (-2)^3
  term_value

theorem constant_term_is_minus_80 : constant_term_expansion = -80 := 
  by
    sorry

end constant_term_is_minus_80_l572_572574


namespace tan_product_eq_three_l572_572072

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572072


namespace div_by_6_for_all_k_l572_572344

def b_n_sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem div_by_6_for_all_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 → (b_n_sum_of_squares k) % 6 = 0 :=
by
  intros k hk
  sorry

end div_by_6_for_all_k_l572_572344


namespace count_true_statements_l572_572172

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement1 := reciprocal 2 + reciprocal 8 = reciprocal 10
def statement2 := reciprocal 9 - reciprocal 3 = reciprocal 6
def statement3 := reciprocal 10 * reciprocal 2 = reciprocal 20
def statement4 := reciprocal 15 / reciprocal 5 = reciprocal 3

theorem count_true_statements :
  (¬statement1 → true) ∧ 
  (¬statement2 → true) ∧ 
  (statement3 → true) ∧ 
  (statement4 → true) →
  sorry -- This part will be used to complete the proof showing that the count is exactly 2

end count_true_statements_l572_572172


namespace length_of_LM_l572_572305

variable (A B C K L M : Type)
variable [HasAngle A B C]
variable [RightAngle A]
variable (angleA: ∠ ABC = 90)
variable (angleB: ∠ BAC = 30)
variable (dAK : AK = 4)
variable (dBL : BL = 31)
variable (dMC : MC = 3)
variable (dKL_KM : KL = KM)

theorem length_of_LM : (LM = 14) :=
  sorry

end length_of_LM_l572_572305


namespace max_n_for_sum_of_squares_l572_572438

theorem max_n_for_sum_of_squares (n : ℕ) (ks : list ℕ) (h1 : ks.nodup) (h2 : ∀ k ∈ ks, k > 0) :
  (ks.map (λ k, k ^ 2)).sum = 2050 ↔ n = 16 :=
by
  sorry

end max_n_for_sum_of_squares_l572_572438


namespace perimeter_of_given_triangle_l572_572932

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572932


namespace total_animals_correct_l572_572515

section 
variable 
  (snakes : ℕ)
  (arctic_foxes : ℕ)
  (leopards : ℕ)
  (bee_eaters : ℕ)
  (cheetahs : ℕ)
  (alligators : ℕ)
  (total : ℕ)

-- Define the initial counts
def snakes := 100
def arctic_foxes := 80
def leopards := 20

-- Define the derived counts based on conditions
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

-- Define the total number of animals based on the derived counts
def total := alligators + bee_eaters + snakes + arctic_foxes + leopards + cheetahs

-- Proof objective
theorem total_animals_correct : total = 670 := by 
  sorry
end

end total_animals_correct_l572_572515


namespace tan_product_l572_572088

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572088


namespace unique_prime_solution_l572_572206

def is_prime (n : ℤ) : Prop :=
  2 ≤ n ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = -1 ∨ m = n ∨ m = -n

def polynomial (n : ℕ) : ℤ :=
  n^3 - 9 * n^2 + 27 * n - 28

theorem unique_prime_solution :
  ({ n : ℕ | is_prime (polynomial n) }).to_finset.card = 1 :=
by
  sorry

end unique_prime_solution_l572_572206


namespace Claudia_solution_l572_572528

noncomputable def Claudia_coins : Prop :=
  ∃ (x y : ℕ), x + y = 12 ∧ 23 - x = 17 ∧ y = 6

theorem Claudia_solution : Claudia_coins :=
by
  existsi 6
  existsi 6
  sorry

end Claudia_solution_l572_572528


namespace find_k_l572_572228

noncomputable section

variables (k λ : ℝ)
variables (e₁ e₂ : ℝ × ℝ × ℝ)

-- e₁ and e₂ are non-coplanar
axiom h₀ : e₁ ≠ (0, 0, 0) ∧ e₂ ≠ (0, 0, 0)

-- Definitions of \overrightarrow {AB}, \overrightarrow {CB}, and \overrightarrow {CD}
def vector_AB := (2 * e₁.1 - k * e₂.1, 2 * e₁.2 - k * e₂.2, 2 * e₁.3 - k * e₂.3)
def vector_CB := (3 * e₁.1 + 3 * e₂.1, 3 * e₁.2 + 3 * e₂.2, 3 * e₁.3 + 3 * e₂.3)
def vector_CD := (k * e₁.1 + e₂.1, k * e₁.2 + e₂.2, k * e₁.3 + e₂.3)

-- Points A, B, and D are collinear, so AB and BD are collinear
def collinear_AB_BD (λ : ℝ) :=
  vector_CD.1 - vector_CB.1 = λ * vector_AB.1 ∧
  vector_CD.2 - vector_CB.2 = λ * vector_AB.2 ∧
  vector_CD.3 - vector_CB.3 = λ * vector_AB.3

-- Question: Find the value of k such that points A, B, and D are collinear
theorem find_k :
  ∃ k : ℝ, 
  (∃ λ : ℝ, collinear_AB_BD k λ) → 
  (k = 4 ∨ k = -1) :=
sorry

end find_k_l572_572228


namespace problem_a_solved_problem_b_solved_l572_572523

-- Definition of the first problem
def expression_a : ℝ :=
  (1 - Real.cos (15 * Real.pi / 180)) * (1 + Real.sin (75 * Real.pi / 180)) +
  Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) * Real.cot (15 * Real.pi / 180)

-- Proof goal for the first problem
theorem problem_a_solved : expression_a = 1 := by
  sorry

-- Definition of the second problem
def expression_b (alpha: ℝ) : ℝ :=
  Real.sin (45 * Real.pi / 180 - alpha) 
  - Real.cos (30 * Real.pi / 180 + alpha)
  + Real.sin (30 * Real.pi / 180)^2
  - Real.cos (45 * Real.pi / 180 + alpha)
  + Real.sin (60 * Real.pi / 180 - alpha)
  + Real.sin (60 * Real.pi / 180)^2

-- Proof goal for the second problem
theorem problem_b_solved (alpha : ℝ) : expression_b alpha = 1 := by
  sorry

end problem_a_solved_problem_b_solved_l572_572523


namespace tubs_of_ice_cream_guests_ate_l572_572426

def pans : Nat := 2
def pieces_per_pan : Nat := 16
def eaten_percentage : Float := 0.75
def scoops_per_tub : Nat := 8
def guests_not_eating : Nat := 4
def scoops_per_guest : Nat := 2

theorem tubs_of_ice_cream_guests_ate :
  let total_pieces := pans * pieces_per_pan
  let eaten_pieces := pieces_per_pan + Nat.floor (eaten_percentage * pieces_per_pan.toReal)
  let guests_ala_mode := eaten_pieces - guests_not_eating
  let total_scoops_eaten := guests_ala_mode * scoops_per_guest
  let tubs_ice_cream := total_scoops_eaten / scoops_per_tub
  tubs_ice_cream = 6 := by
  sorry

end tubs_of_ice_cream_guests_ate_l572_572426


namespace cone_lateral_surface_area_is_12pi_l572_572627

def radius : ℝ := 2
def height : ℝ := 4 * Real.sqrt 2

noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)

noncomputable def lateral_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area_is_12pi : lateral_surface_area radius (slant_height radius height) = 12 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_is_12pi_l572_572627


namespace simplify_and_evaluate_expression_l572_572814

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l572_572814


namespace tan_product_l572_572149

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572149


namespace technology_courses_correct_l572_572292

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end technology_courses_correct_l572_572292


namespace num_divisors_not_divisible_by_three_l572_572664

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572664


namespace ordered_pairs_count_l572_572332

theorem ordered_pairs_count :
  ∃ M : ℕ,
  (∀ (C D : set ℕ), 
      C ∪ D = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
      C ∩ D = ∅ ∧ 
      (|C| ∉ C) ∧ 
      (|D| ∉ D) → 
      M = 37) :=
sorry

end ordered_pairs_count_l572_572332


namespace cos_5theta_zero_l572_572640

-- Conditions of the problem
variable (a : ℝ) (h_pos : a > 0)
variable (x y : ℝ) (h_hyp : x^2 / a^2 - y^2 / a^2 = 1)
variable (P : ℝ × ℝ) (h_on_hyp : P ≠ (a, 0))
variable (A1 A2 : ℝ × ℝ) (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (θ : ℝ) (h_angle_relation : 3 * θ = angle A1 P A2)

theorem cos_5theta_zero : cos (5 * θ) = 0 :=
by
  sorry

end cos_5theta_zero_l572_572640


namespace right_triangle_perimeter_l572_572944

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572944


namespace tan_product_eq_three_l572_572073

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572073


namespace triangle_perimeter_l572_572955

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572955


namespace distribute_students_l572_572845

theorem distribute_students : 
  ∃ (ways : ℕ), ways = 150 ∧ 
    (∃ (students factories : ℕ) (assignment : List ℕ → List ℕ → Prop),
    students = 5 ∧ factories = 3 ∧ 
    (∀ (xs : List ℕ) (ys : List ℕ), assignment xs ys → xs.length = students ∧ ys.length = factories ∧ (∀ y ∈ ys, y > 0)) ∧
    ways = ∑ x in ({3, 3, 1}.permutations.toFinset ∪ {2, 2, 1}.permutations.toFinset), 
               let ⟨a, b, c⟩ := x.pmap (λ i hi => i.succ) (by simp [Finset.mem_insert, Nat.lt_succ_self]) in 
               (Nat.choose 5 a) * (Nat.choose (5 - a) b) * (Fact.fact 3 / (if a = b then 2 else 1)))

end distribute_students_l572_572845


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572692

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572692


namespace tan_product_pi_nine_l572_572123

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572123


namespace solve_for_x_l572_572818

theorem solve_for_x (x : ℝ) : (8^(16^x) = 16^(8^x)) → x = Real.logb 2 (4 / 3) :=
by
  sorry

end solve_for_x_l572_572818


namespace x_minus_q_eq_3_minus_2q_l572_572704

theorem x_minus_q_eq_3_minus_2q {x q : ℝ} (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_eq_3_minus_2q_l572_572704


namespace tan_product_l572_572085

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572085


namespace m_n_sum_eq_three_l572_572717

noncomputable def distance_between_parallel_lines 
(A B C1 C2 : ℝ) : ℝ := 
  abs (C2 - C1) / real.sqrt (A^2 + B^2)

theorem m_n_sum_eq_three (m n : ℝ) 
(hm : m > 0) 
(hparallel : 1 / (-2 : ℝ) = 2 / n) 
(hdist : distance_between_parallel_lines 1 (-2) m (-6) = 2 * real.sqrt 5)
: m + n = 3 :=
sorry

end m_n_sum_eq_three_l572_572717


namespace cone_height_l572_572904

noncomputable def radius : ℝ := 10
noncomputable def arc_length_one_sector : ℝ := (2 * Real.pi * radius) / 4
noncomputable def base_radius_cone : ℝ := arc_length_one_sector / (2 * Real.pi)
noncomputable def slant_height_cone : ℝ := radius
noncomputable def height_cone : ℝ := Real.sqrt(slant_height_cone^2 - base_radius_cone^2)

theorem cone_height (r : ℝ) (a : ℝ) (h_rb : a = (2 * Real.pi * r) / 4)
  (h_br : base_radius_cone = a / (2 * Real.pi))
  (h_sh : slant_height_cone = r) :
  height_cone = 5 * Real.sqrt 3.75 := by
  sorry

end cone_height_l572_572904


namespace reflection_find_R_l572_572375

open Real

def point := (ℝ × ℝ)

def is_reflection {A C B D : point} (k c : ℝ) : Prop :=
  let M := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let N := ((B.1 + D.1) / 2, (B.2 + D.2) / 2)
  M.2 = k * M.1 + c ∧ N.2 = k * N.1 + c

theorem reflection_find_R :
  ∀ (A C B D : point) (k c : ℝ),
  A = (3, 7) → C = (5, 5) → B = (8, 14) → D = (12, 10) →
  is_reflection A C B D k c →
  (k / c) = 1/2 :=
by
  sorry

end reflection_find_R_l572_572375


namespace dakotas_medical_bill_l572_572545

variable (days_in_hospital : ℕ) (bed_cost_per_day : ℕ) (specialist_cost_per_hour : ℕ) (specialist_time_in_hours : ℚ) (num_specialists : ℕ) (ambulance_cost : ℕ)

theorem dakotas_medical_bill 
  (h1 : days_in_hospital = 3) 
  (h2 : bed_cost_per_day = 900)
  (h3 : specialist_cost_per_hour = 250)
  (h4 : specialist_time_in_hours = 0.25)
  (h5 : num_specialists = 2)
  (h6 : ambulance_cost = 1800) : 

  let bed_total := bed_cost_per_day * days_in_hospital,
      specialists_total := (specialist_cost_per_hour * specialist_time_in_hours * num_specialists).toNat,
      total_cost := bed_total + specialists_total + ambulance_cost
  in 
  total_cost = 4750 := 
by 
  sorry

end dakotas_medical_bill_l572_572545


namespace min_PA_PM_l572_572238

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 = 0.5 * P.1^2

def projection_on_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def A : ℝ × ℝ := (6, 17/2)

theorem min_PA_PM (P : ℝ × ℝ) (hP : on_parabola P) :
  let M := projection_on_x_axis P in
  |distance P A + distance P M| = 19/2 :=
sorry

end min_PA_PM_l572_572238


namespace isosceles_triangle_sum_t_l572_572875

theorem isosceles_triangle_sum_t (t : ℝ) : 
  (0 <= t) ∧ (t <= 360) ∧
  let A := (Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180)) in
  let B := (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180)) in
  let C := (Real.cos (t * Real.pi / 180), Real.sin (t * Real.pi / 180)) in
  (dist A B = dist B C ∨ dist A C = dist B C ∨ dist A B = dist A C) →
  ∃ tvals : List ℝ, tvals = [30, 150, 330] ∧ tvals.sum = 330 :=
by
  sorry

end isosceles_triangle_sum_t_l572_572875


namespace rectangular_solid_surface_area_l572_572441

theorem rectangular_solid_surface_area (length width depth : ℝ) (H1 : length = 9) (H2 : width = 8) (H3 : depth = 5) :
    2 * (length * width) + 2 * (width * depth) + 2 * (length * depth) = 314 :=
by
  rw [H1, H2, H3]
  simp
  sorry

end rectangular_solid_surface_area_l572_572441


namespace ways_to_paint_faces_l572_572969

theorem ways_to_paint_faces (die_faces : set ℕ) :
  die_faces = {1, 2, 3, 4, 5, 6} →
  (∃! (subset : set (set ℕ)), 
    (subset.card = 3 ∧ 
      subset ⊆ die_faces ∧ 
      ∑ x in subset, x ≠ 11 ∧
      subset.card = 17)) :=
by
  intro h_faces,
  have h_total : (finset.univ : finset ℕ).card = 3 := sorry,
  have h_sum_restriction : ∑ x in finset.univ, x ≠ 11 := sorry,
  exact h_total, h_sum_restriction, sorry

end ways_to_paint_faces_l572_572969


namespace isosceles_triangle_parallel_ID_AC_l572_572289

open Real EuclideanGeometry

variable {A B C O I D : Point}

-- Definitions of conditions
def is_isosceles (ABC : Triangle) : Prop :=
  AC = BC

def circumcenter (ABC : Triangle) (O : Point) : Prop :=
  ∀ P : Point, (P = O ↔ ∃ Q : Point, (Triangle.has_circumcenter ABC Q ∧ Q = O))

def incenter (ABC : Triangle) (I : Point) : Prop :=
  ∀ R : Point, (R = I ↔ ∃ S : Point, (Triangle.has_incenter ABC S ∧ S = I))

def on_side_BC (D : Point) : Prop :=
  point_on_line D BC

def perpendicular_to_BI (O D I : Point) : Prop :=
  Line.perpendicular (line_through O D) (line_through B I)

-- Statement of the theorem to be proved:
theorem isosceles_triangle_parallel_ID_AC
  (h_isosceles : is_isosceles ABC)
  (h_circumcenter : circumcenter ABC O)
  (h_incenter : incenter ABC I)
  (h_on_BC : on_side_BC D)
  (h_perpendicular : perpendicular_to_BI O D I) :
  Line.parallel (line_through I D) (line_through A C) :=
sorry

end isosceles_triangle_parallel_ID_AC_l572_572289


namespace rectangle_width_l572_572414

theorem rectangle_width (P l: ℕ) (hP : P = 50) (hl : l = 13) : 
  ∃ w : ℕ, 2 * l + 2 * w = P ∧ w = 12 := 
by
  sorry

end rectangle_width_l572_572414


namespace zero_is_multiple_of_every_integer_l572_572443

theorem zero_is_multiple_of_every_integer (x : ℤ) : ∃ n : ℤ, 0 = n * x := by
  use 0
  exact (zero_mul x).symm

end zero_is_multiple_of_every_integer_l572_572443


namespace assignment_count_l572_572541

noncomputable def num_ways_to_assign_students : ℕ :=
  fintype.card {σ : fin 7 → fin 5 // 
    -- Condition 1: Ensuring A and B are not in the same event
    σ 0 ≠ σ 1 ∧ 
    -- Condition 2: Ensuring each event has at least one participant
    ∀ j, ∃ i, σ i = j ∧ 
    -- Condition 3: Each student can only participate in one event
    function.injective σ 
  }

theorem assignment_count :
  num_ways_to_assign_students = 15000 := sorry

end assignment_count_l572_572541


namespace percentage_of_total_money_raised_from_donations_l572_572848

-- Define the conditions
def max_donation := 1200
def num_donors_max := 500
def half_donation := max_donation / 2
def num_donors_half := 3 * num_donors_max
def total_money_raised := 3750000

-- Define the amounts collected from each group
def amount_from_max_donors := num_donors_max * max_donation
def amount_from_half_donors := num_donors_half * half_donation
def total_amount_from_donations := amount_from_max_donors + amount_from_half_donors

-- Define the percentage calculation
def percentage_of_total := (total_amount_from_donations / total_money_raised) * 100

-- State the theorem (but not the proof)
theorem percentage_of_total_money_raised_from_donations : 
  percentage_of_total = 40 := by
  sorry

end percentage_of_total_money_raised_from_donations_l572_572848


namespace sequence_periodic_and_value_l572_572608

noncomputable def sequence (n : ℕ) : ℚ :=
if n = 1 then -1
else 1 - (1 / sequence (n - 1))

theorem sequence_periodic_and_value :
  sequence 2018 = 2 :=
by
  sorry

end sequence_periodic_and_value_l572_572608


namespace moving_point_trajectory_dot_product_condition_range_of_t_for_angle_l572_572242

noncomputable def trajectory (x y t : ℝ) : Prop :=
(1 - t^2) * x^2 + y^2 + 4 * x + 3 = 0

theorem moving_point_trajectory (x y t : ℝ) (hx: t > 0) (hy: t ≠ 1) :
  trajectory x y t :=
sorry

noncomputable def curve_G_eqa (x y : ℝ) t :=
t = sqrt 3 → ((x - 1)^2 / (5 / 2) - y^2 / 5 = 1)

theorem dot_product_condition (x₀ y₀ x m n : ℝ) (h0 : t = sqrt 3) (hy₀ : y₀^2 = 2 * x₀^2 - 5) :
  -- The coordinates of Q are (x₀, y₀), and coordinates of P_1 and P_2 whose values we need to calculate
  let Q := (x₀, y₀),
       P₁ := ( 2 * y₀ + sqrt 2 * x₀) / (3 * sqrt 2),
       P₂ := ( sqrt 2 * x₀ - 2 * y₀) / (3 * sqrt 2) in
  (fst Q - fst P₁) * (fst Q - fst P₂) + (snd Q - snd P₁) * (snd Q - snd P₂) = 5 / 9 :=
sorry

theorem range_of_t_for_angle (t theta : ℝ) (hθ: 0 < θ) (hθπ : θ < π) :
  (0 < t) → t < sqrt ((1 - cos θ) / 2) :=
sorry

end moving_point_trajectory_dot_product_condition_range_of_t_for_angle_l572_572242


namespace sum_of_quartic_and_quadratic_is_quartic_expression_l572_572843

-- Definitions of the given polynomials
def quartic_polynomial (P : ℝ[X]) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ P = a * X^4 + b * X^3 + c * X^2 + d * X + e

def quadratic_polynomial (Q : ℝ[X]) : Prop :=
  ∃ (f g h : ℝ), f ≠ 0 ∧ Q = f * X^2 + g * X + h

-- Theorem statement
theorem sum_of_quartic_and_quadratic_is_quartic_expression
  (P Q : ℝ[X]) 
  (hP : quartic_polynomial P) 
  (hQ : quadratic_polynomial Q) : ∃ (R : ℝ[X]), ∃ n ≤ 4, degree R = n ∧ R = P + Q :=
sorry

end sum_of_quartic_and_quadratic_is_quartic_expression_l572_572843


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572693

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572693


namespace exists_point_P_l572_572487

-- Given definitions based on conditions
variables {A B C P : Type} -- Points A, B, C form a triangle, P is the point to be constructed
variables (circle : Set Type) -- The circle in which triangle ABC is inscribed
variables (is_triangle_inscribed : Triangle ABC ∧ Inscribed circle ABC) -- Condition 1
variables (equilateral_triangle_vertices : Set (Point P)) -- The vertices of an equilateral triangle

-- Main statement: There exists a point P which forms an equilateral triangle
theorem exists_point_P : 
  ∃ P : Type, (Intersection (Line AP circle), Intersection (Line BP circle), Intersection (Line CP circle)) = (equilateral_triangle_vertices) := sorry

end exists_point_P_l572_572487


namespace Bettina_card_value_l572_572311

theorem Bettina_card_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : Real.tan x ≠ 1) (h₃ : Real.sin x ≠ Real.cos x) :
  ∀ {a b c : ℝ}, (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
                  (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
                  (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
                  a ≠ b → b ≠ c → a ≠ c →
                  (b = Real.cos x) → b = Real.sqrt 3 / 2 := 
  sorry

end Bettina_card_value_l572_572311


namespace problem_smallest_l_l572_572218

noncomputable def A (m l : ℕ) : ℕ :=
  ((4 * l + 1) * (4 * l + 2) * ...) * (4 * (5^m + 1) * l)

theorem problem (m : ℕ) (hm : m > 0):
  ∃ (l : ℕ), (l > 0) ∧
  (5^(5 * m * l) ∣ A m l) ∧
  ((5^(5 * m * l + 1) + A m l).denom = 1) :=
sorry

theorem smallest_l (m : ℕ) (hm : m > 0) : 
  ∃ (l : ℕ), (l > 0) ∧
  (5^(5 * m * l) ∣ A m l) ∧
  ((5^(5 * m * l + 1) + A m l).denom = 1) ∧
  l = (5^(m+1) - 1) / 4 :=
sorry

end problem_smallest_l_l572_572218


namespace triangle_area_sum_eq_l572_572745

variables (A B C A1 A2 B1 B2 C1 C2 : Point)
variable (ABC : Triangle A B C)
variables (symBC : symmetric_mirror_midpoint A1 A2 BC)
variables (symCA : symmetric_mirror_midpoint B1 B2 CA)
variables (symAB : symmetric_mirror_midpoint C1 C2 AB)

theorem triangle_area_sum_eq : 
  area (Triangle A1 B1 C1) + area (Triangle A1 B1 C2) + area (Triangle A1 B2 C1) + area (Triangle A2 B1 C1) = area ABC :=
sorry

end triangle_area_sum_eq_l572_572745


namespace number_of_people_disliking_radio_and_music_l572_572376

noncomputable def people_dont_like_radio_and_music (total_people : ℕ) (percentage_dislike_radio : ℝ) (percentage_dislike_both : ℝ) : ℕ :=
  let num_dislike_radio := percentage_dislike_radio * total_people
  let num_dislike_both := percentage_dislike_both * num_dislike_radio
  num_dislike_both.to_int

theorem number_of_people_disliking_radio_and_music :
  ∃ (total_people : ℕ) (percentage_dislike_radio percentage_dislike_both : ℝ),
    total_people = 1500 ∧
    percentage_dislike_radio = 0.25 ∧
    percentage_dislike_both = 0.15 ∧
    people_dont_like_radio_and_music total_people percentage_dislike_radio percentage_dislike_both = 56 :=
by
  use 1500, 0.25, 0.15
  split
  · refl
  split
  · refl
  split
  · refl
  rw [people_dont_like_radio_and_music]
  sorry

end number_of_people_disliking_radio_and_music_l572_572376


namespace determine_a_l572_572720

theorem determine_a (a : ℝ) (A : set ℝ) (hA : A = {x | a * x^2 + 4 * x + 4 = 0}) : 
  (A.card = 1) → (a = 0 ∨ a = 1):=
by
  sorry

end determine_a_l572_572720


namespace divisors_not_divisible_by_3_eq_6_l572_572654

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572654


namespace total_practice_hours_l572_572363

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l572_572363


namespace tan_product_l572_572148

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572148


namespace time_comparison_l572_572920

noncomputable def pedestrian_speed : Real := 6.5
noncomputable def cyclist_speed : Real := 20.0
noncomputable def distance_between_points_B_A : Real := 4 * Real.pi - 6.5
noncomputable def alley_distance : Real := 4 * Real.pi - 6.5
noncomputable def combined_speed_3 : Real := pedestrian_speed + cyclist_speed
noncomputable def combined_speed_2 : Real := 21.5
noncomputable def time_scenario_3 : Real := (4 * Real.pi - 6.5) / combined_speed_3
noncomputable def time_scenario_2 : Real := (10.5 - 2 * Real.pi) / combined_speed_2

theorem time_comparison : time_scenario_2 < time_scenario_3 :=
by
  sorry

end time_comparison_l572_572920


namespace find_function_solution_l572_572192

def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (x y : ℝ), f (f (x * y)) = |x| * f y + 3 * f (x * y)

theorem find_function_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 4 * |x|) ∨ (∀ x : ℝ, f x = -4 * |x|) :=
by
  sorry

end find_function_solution_l572_572192


namespace right_triangle_perimeter_l572_572951

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572951


namespace bicycle_speed_l572_572373

theorem bicycle_speed (d1 d2 v1 v_avg : ℝ)
  (h1 : d1 = 300) 
  (h2 : d1 + d2 = 450) 
  (h3 : v1 = 20) 
  (h4 : v_avg = 18) : 
  (d2 / ((d1 / v1) + d2 / (d2 * v_avg / 450)) = 15) :=
by 
  sorry

end bicycle_speed_l572_572373


namespace length_of_PQ_is_8_l572_572296

-- Define the lengths of the sides and conditions
variables (PQ QR PS SR : ℕ) (perimeter : ℕ)

-- State the conditions
def conditions : Prop :=
  SR = 16 ∧
  perimeter = 40 ∧
  PQ = QR ∧ QR = PS

-- State the goal
theorem length_of_PQ_is_8 (h : conditions PQ QR PS SR perimeter) : PQ = 8 :=
sorry

end length_of_PQ_is_8_l572_572296


namespace boat_travel_time_difference_l572_572902

noncomputable def travel_time_difference (v : ℝ) : ℝ :=
  let d := 90
  let t_downstream := 2.5191640969412834
  let t_upstream := d / (v - 3)
  t_upstream - t_downstream

theorem boat_travel_time_difference :
  ∃ v : ℝ, travel_time_difference v = 0.5088359030587166 := 
by
  sorry

end boat_travel_time_difference_l572_572902


namespace find_largest_angle_l572_572744

variables {A B C : Type*} [Real A] [Real B] [Real C]
variables {a b c : ℝ}
variables {cos : ℝ → ℝ}

-- Define the conditions
def cond1 : Prop := a + a*c + b*c = 2*b
def cond2 : Prop := a - a*c + b*c = 2*c
def cond3 : Prop := a = b + c + 2*b*c * cos(⟦A⟧)

-- Define what we need to prove
def largest_angle_is_A : Prop := ∠A = 120

-- The theorem
theorem find_largest_angle
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3) :
  largest_angle_is_A := 
sorry

end find_largest_angle_l572_572744


namespace fixed_point_chord_l572_572598

noncomputable theory
open_locale classical

variables {S : Type*} [euclidean_space S] (O : S) (r : ℝ) (l : affine_subspace ℝ S) (P : S)

-- Define the circle with center O and radius r
def circle (O : S) (r : ℝ) : set S :=
  {X | dist X O = r}

-- Define the line l does not intersect the circle S
def line_disjoint_circle (l : affine_subspace ℝ S) (O : S) (r : ℝ) : Prop :=
  ∃ (M : S), M ∈ l ∧ dist O M > r

-- Define the tangents PA and PB from the point P to the circle O with radius r
def is_tangent (P A : S) (O : S) (r : ℝ) : Prop :=
  dist P A = dist P O ∧ dist A O = r

-- Formalize the theorem
theorem fixed_point_chord {S : Type*} [euclidean_space S] (O : S) (r : ℝ) (l : affine_subspace ℝ S) :
  (line_disjoint_circle l O r) →
  (∀ P ∈ l, ∃ A B : S, is_tangent P A O r ∧ is_tangent P B O r) →
  ∃ X : S, ∀ (P ∈ l) (A B : S), is_tangent P A O r ∧ is_tangent P B O r →
  (let M := classical.some (line_disjoint_circle l O r) in
   collinear {O, M, X} ∧ (line_through A B).intersect (line_through O M) = {X})
  sorry

end fixed_point_chord_l572_572598


namespace train_passing_platform_l572_572891

theorem train_passing_platform (train_length : ℕ) (pole_passing_time : ℕ) (platform_length : ℕ) : 
  train_length = 240 →
  pole_passing_time = 24 →
  platform_length = 650 →
  (train_length + platform_length) / (train_length / pole_passing_time) = 89 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_passing_platform_l572_572891


namespace tan_identity_proof_l572_572091

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572091


namespace consecutive_integers_greatest_l572_572871

theorem consecutive_integers_greatest (n : ℤ) (h : n + 2 = 8) : 
  (n + 2 = 8) → (max n (max (n + 1) (n + 2)) = 8) :=
by {
  sorry
}

end consecutive_integers_greatest_l572_572871


namespace tan_product_pi_nine_l572_572125

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572125


namespace unoccupied_volume_tank_l572_572388

theorem unoccupied_volume_tank :
  let side_length := 12
  let tank_volume := side_length ^ 3  -- Volume of the cube-shaped tank
  let water_volume := (1 / 3 : ℚ) * tank_volume  -- Volume of the water
  let marble_diameter := 1
  let marble_radius := marble_diameter / 2
  let marble_volume := (4 / 3 : ℚ) * Real.pi * marble_radius^3  -- Volume of one marble
  let number_of_marbles := 15
  let total_marble_volume := number_of_marbles * marble_volume  -- Total volume of all marbles
  let total_occupied_volume := water_volume + total_marble_volume  -- Total occupied volume by water and marbles
  let unoccupied_volume := tank_volume - total_occupied_volume -- Unoccupied volume
  unoccupied_volume = 1152 - (15 * Real.pi / 6) := 
by
  sorry

end unoccupied_volume_tank_l572_572388


namespace cone_lateral_surface_area_l572_572625

theorem cone_lateral_surface_area
  (r h : ℝ)
  (hr : r = 2)
  (hh : h = 4 * Real.sqrt 2) :
  (let l := Real.sqrt (r^2 + h^2) in
   let S := Real.pi * r * l in
   S = 12 * Real.pi) :=
by
  sorry

end cone_lateral_surface_area_l572_572625


namespace num_divisors_not_divisible_by_three_l572_572661

theorem num_divisors_not_divisible_by_three (n : ℕ) (h : n = 180) : 
  ∃ d, d = 6 ∧ ∀ k, k > 0 ∧ k ∣ n → (¬ 3 ∣ k → k ∈ {d | d ∣ n ∧ ¬ 3 ∣ d}) :=
by 
  have h180 : 180 = 2^2 * 3^2 * 5, by norm_num
  rw h at h180
  use 6
  sorry

end num_divisors_not_divisible_by_three_l572_572661


namespace alex_needs_additional_coins_l572_572493

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end alex_needs_additional_coins_l572_572493


namespace sum_underlined_numbers_non_negative_l572_572024

def sum_underlined_numbers (seq : Fin 100 → Int) : Bool :=
  let underlined_indices : List (Fin 100) :=
    List.range 100 |>.filter (λ i =>
      seq i > 0 ∨ (i < 99 ∧ seq i + seq (i + 1) > 0) ∨ (i < 98 ∧ seq i + seq (i + 1) + seq (i + 2) > 0))
  let underlined_sum : Int := underlined_indices.map (λ i => seq i) |>.sum
  underlined_sum ≤ 0

theorem sum_underlined_numbers_non_negative {seq : Fin 100 → Int} :
  ¬ sum_underlined_numbers seq :=
sorry

end sum_underlined_numbers_non_negative_l572_572024


namespace complement_union_eq_l572_572260

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_eq :
  U \ (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_eq_l572_572260


namespace determine_b_coplanar_l572_572188

def coplanar_points (b : ℝ) : Prop :=
  let v1 := ![2, b, 0]
      v2 := ![0, 2, b]
      v3 := ![b, 0, 2]
  in (Matrix.det (Matrix.of ![v1, v2, v3]) = 0)

theorem determine_b_coplanar :
  ∀ (b : ℝ), coplanar_points b ↔ b = 2 ∨ b = -2 := 
by
  sorry

end determine_b_coplanar_l572_572188


namespace rectangle_dimensions_folded_l572_572480

-- Define the conditions
variables {a b : ℝ}

def is_rectangle_folded (a b : ℝ) : Prop :=
  let diag := (a ^ 2 + b ^ 2) ^ (1/2)
  a > 0 ∧ b > 0 ∧
  ∃ (x : ℝ), x > 0 ∧
  (diag = 10 * real.sqrt 3) ∧
  (a = b) ∧
  (b = x)

-- The main statement to prove
theorem rectangle_dimensions_folded :
  is_rectangle_folded (5 * real.sqrt 3) 15 :=
sorry -- proof

end rectangle_dimensions_folded_l572_572480


namespace range_of_m_l572_572259

open Set

variable {α : Type} [LinearOrder α] {x : α} {m : α}

def A := {x : Real | -1 < x ∧ x < 3}
def B := {x : Real | -1 < x ∧ x < m + 1}

theorem range_of_m (h : B ⊆ A) : -2 < m ∧ m < 2 := by
  sorry

end range_of_m_l572_572259


namespace comparison_of_a_b_c_l572_572619

noncomputable def a : ℝ := 2^(-2/3)
noncomputable def b : ℝ := 2^(-4/3)
noncomputable def c : ℝ := 2^(-1/3)

theorem comparison_of_a_b_c : b < a ∧ a < c := by 
  sorry

end comparison_of_a_b_c_l572_572619


namespace tangent_product_eq_three_l572_572119

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572119


namespace sin_alpha_is_one_sixth_l572_572347

theorem sin_alpha_is_one_sixth {α : ℝ} (h1 : α > π / 2 ∧ α < π) (h2 : 3 * sin (2 * α) = cos α) :
  sin α = 1 / 6 :=
by
  sorry

end sin_alpha_is_one_sixth_l572_572347


namespace derivative_at_neg_one_eq_zero_l572_572718

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := x * Real.exp(x) + c * x^2

theorem derivative_at_neg_one_eq_zero 
  (f : ℝ → ℝ)
  (hyp : ∀ x, f x = x * Real.exp(x) + f'(-1) * x^2) : f'(-1) = 0 :=
by
  sorry

end derivative_at_neg_one_eq_zero_l572_572718


namespace exists_two_same_remainder_l572_572390

theorem exists_two_same_remainder (n : ℤ) (a : ℕ → ℤ) :
  ∃ i j : ℕ, i ≠ j ∧ 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n ∧ (a i % n = a j % n) := sorry

end exists_two_same_remainder_l572_572390


namespace factor_x10_minus_1296_l572_572529

theorem factor_x10_minus_1296 (x : ℝ) : (x^10 - 1296) = (x^5 + 36) * (x^5 - 36) :=
  by
  sorry

end factor_x10_minus_1296_l572_572529


namespace tan_identity_proof_l572_572092

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572092


namespace sheepdog_rounded_up_percentage_l572_572066

/-- Carla's sheepdog rounded up a certain percentage of her sheep. We know the remaining 10% of the sheep  wandered off into the hills, which is 9 sheep out in the wilderness. There are 81 sheep in the pen. We need to prove that the sheepdog rounded up 90% of the total number of sheep. -/
theorem sheepdog_rounded_up_percentage (total_sheep pen_sheep wilderness_sheep : ℕ) 
  (h1 : wilderness_sheep = 9) 
  (h2 : pen_sheep = 81) 
  (h3 : wilderness_sheep = total_sheep / 10) :
  (pen_sheep * 100 / total_sheep) = 90 :=
sorry

end sheepdog_rounded_up_percentage_l572_572066


namespace train_crossing_time_l572_572453

namespace TrainProblem

-- Define the given conditions
def trainLength : ℝ := 140 -- in meters
def trainSpeed_kmh : ℝ := 108 -- in km/hr

-- Conversion from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Speed in m/s
def trainSpeed_ms : ℝ := kmh_to_ms trainSpeed_kmh

-- Time taken to cross the electric pole
def time_to_cross_pole (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- The statement we need to prove
theorem train_crossing_time :
  time_to_cross_pole trainLength trainSpeed_ms = 4.67 :=
by
  sorry

end TrainProblem

end train_crossing_time_l572_572453


namespace tan_product_equals_three_l572_572168

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572168


namespace divisors_not_divisible_by_3_eq_6_l572_572656

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572656


namespace unique_function_solution_l572_572567

theorem unique_function_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x^2 + f(y)) = y + (f(x))^2) : f = id :=
begin
  sorry
end

end unique_function_solution_l572_572567


namespace equal_area_division_l572_572173

def triangle : Type :=
  {A B C : (ℝ × ℝ) // 
    A = (1, 3) ∧ 
    B = (1, 1) ∧ 
    C = (10, 1)}

theorem equal_area_division (T : triangle) : 
  ∃ b : ℝ, 
    (∀ y, 
      let area_below := (1 / 2) * 9 * (b - 1)
      in y = b → area_below = 4.5) → b = 2 :=
sorry

end equal_area_division_l572_572173


namespace tan_product_identity_l572_572109

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572109


namespace red_and_purple_probability_l572_572032

def total_balls : ℕ := 120
def white_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 24
def red_balls : ℕ := 20
def blue_balls : ℕ := 10
def purple_balls : ℕ := 5
def orange_balls : ℕ := 4
def gray_balls : ℕ := 2

def probability_red_purple : ℚ := 5 / 357

theorem red_and_purple_probability :
  ((red_balls / total_balls) * (purple_balls / (total_balls - 1)) +
  (purple_balls / total_balls) * (red_balls / (total_balls - 1))) = probability_red_purple :=
by
  sorry

end red_and_purple_probability_l572_572032


namespace minimum_value_inequality_l572_572342

variable {x y z : ℝ}
variable (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)

theorem minimum_value_inequality : (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 :=
sorry

end minimum_value_inequality_l572_572342


namespace num_nat_numbers_satisfying_condition_l572_572200

theorem num_nat_numbers_satisfying_condition :
  ∃ (n : ℕ), n = 2800 ∧ ∀ k : ℕ, k ≤ 353500 ∧ (k^2 + k) % 505 = 0 → k ∈ n :=
sorry

end num_nat_numbers_satisfying_condition_l572_572200


namespace does_not_determine_shape_l572_572789

-- Definition of a function that checks whether given data determine the shape of a triangle
def determines_shape (data : Type) : Prop := sorry

-- Various conditions about data
def ratio_two_angles_included_side : Type := sorry
def ratios_three_angle_bisectors : Type := sorry
def ratios_three_side_lengths : Type := sorry
def ratio_angle_bisector_opposite_side : Type := sorry
def three_angles : Type := sorry

-- The main theorem stating that the ratio of an angle bisector to its corresponding opposite side does not uniquely determine the shape of a triangle.
theorem does_not_determine_shape :
  ¬determines_shape ratio_angle_bisector_opposite_side := sorry

end does_not_determine_shape_l572_572789


namespace loom_weaving_rate_l572_572467

theorem loom_weaving_rate :
  (119.04761904761905 : ℝ) > 0 ∧ (15 : ℝ) > 0 ∧ ∃ rate : ℝ, rate = 15 / 119.04761904761905 → rate = 0.126 :=
by sorry

end loom_weaving_rate_l572_572467


namespace tan_identity_proof_l572_572098

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572098


namespace distance_3D_l572_572870

theorem distance_3D : 
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  d = Real.sqrt 145 :=
by
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  dsimp [A, B, d]
  sorry

end distance_3D_l572_572870


namespace samantha_total_payment_l572_572387

noncomputable def rental_cost (days : ℕ) : ℝ := 30 * days
noncomputable def mileage_cost (miles : ℕ) : ℝ := 0.2 * miles
noncomputable def total_cost (days : ℕ) (miles : ℕ) : ℝ := rental_cost days + mileage_cost miles

theorem samantha_total_payment : total_cost 5 600 = 270 :=
by
  unfold total_cost
  unfold rental_cost
  unfold mileage_cost
  norm_num
  sorry

end samantha_total_payment_l572_572387


namespace tan_product_l572_572147

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572147


namespace trains_crossing_time_l572_572014

theorem trains_crossing_time
  (L : ℕ) (t1 t2 : ℕ)
  (h_length : L = 120)
  (h_t1 : t1 = 10)
  (h_t2 : t2 = 15) :
  let V1 := L / t1
  let V2 := L / t2
  let V_relative := V1 + V2
  let D := L + L
  (D / V_relative) = 12 :=
by
  sorry

end trains_crossing_time_l572_572014


namespace officeEmployees_l572_572734

noncomputable def totalEmployees 
  (averageSalaryAll : ℝ) 
  (averageSalaryOfficers : ℝ) 
  (averageSalaryManagers : ℝ) 
  (averageSalaryWorkers : ℝ) 
  (numOfficers : ℕ) 
  (numManagers : ℕ) 
  (numWorkers : ℕ) : ℕ := 
  if (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
      = (numOfficers + numManagers + numWorkers) * averageSalaryAll 
  then numOfficers + numManagers + numWorkers 
  else 0

theorem officeEmployees
  (averageSalaryAll : ℝ)
  (averageSalaryOfficers : ℝ)
  (averageSalaryManagers : ℝ)
  (averageSalaryWorkers : ℝ)
  (numOfficers : ℕ)
  (numManagers : ℕ)
  (numWorkers : ℕ) :
  averageSalaryAll = 720 →
  averageSalaryOfficers = 1320 →
  averageSalaryManagers = 840 →
  averageSalaryWorkers = 600 →
  numOfficers = 10 →
  numManagers = 20 →
  (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
    = (numOfficers + numManagers + numWorkers) * averageSalaryAll →
  totalEmployees averageSalaryAll averageSalaryOfficers averageSalaryManagers averageSalaryWorkers numOfficers numManagers numWorkers = 100 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6] at h7
  rw [h1]
  simp [totalEmployees, h7]
  sorry

end officeEmployees_l572_572734


namespace average_of_first_two_numbers_is_6_2_l572_572827

theorem average_of_first_two_numbers_is_6_2 
  (numbers : Fin 6 → ℝ)
  (h_average_6 : (∑ i, numbers i) / 6 = 6.40)
  (h_average_2_set1 : (numbers 0 + numbers 1) / 2 = ?A)
  (h_average_2_set2 : (numbers 2 + numbers 3) / 2 = 6.1)
  (h_average_2_set3 : (numbers 4 + numbers 5) / 2 = 6.9) :
  (numbers 0 + numbers 1) / 2 = 6.2 := sorry

end average_of_first_two_numbers_is_6_2_l572_572827


namespace prime_factors_of_M_l572_572998

theorem prime_factors_of_M :
  ∀ (M : ℝ), log 2 (log 3 (log 7 (log 11 M))) = 7 → (∃ n : ℕ, n = 1) :=
by {
sorry
}

end prime_factors_of_M_l572_572998


namespace find_c_l572_572301

-- Definitions for the problem
variables (A B c : ℝ)
variables (a b : ℝ := 1) (b' : ℝ := Real.sqrt 3)

-- Hypotheses
hypothesis B_eq_2A : B = 2 * A
hypothesis a_eq_1 : a = 1
hypothesis b_sqrt3 : b = Real.sqrt 3

-- Lean statement to prove
theorem find_c (A B a b : ℝ) (c : ℝ) 
  (B_eq_2A : B = 2 * A) (a_eq_1 : a = 1) 
  (b_sqrt3 : b = Real.sqrt 3) : c = 2 :=
  sorry -- proof to be filled in later

end find_c_l572_572301


namespace right_triangle_perimeter_l572_572937

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572937


namespace tan_product_l572_572150

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572150


namespace log_lt_x_sq_l572_572269

theorem log_lt_x_sq (x : ℝ) (h : x > 0) : log (1 + x^2) < x^2 :=
by
  sorry

end log_lt_x_sq_l572_572269


namespace min_ap_bp_is_9_l572_572326

open Real

def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def parabola (P : ℝ × ℝ) : Prop :=
  P.2^2 = 8 * P.1

def A : ℝ × ℝ := point 2 0
def B : ℝ × ℝ := point 7 6

noncomputable def min_ap_bp_value : ℝ :=
  Inf { d | ∃ P, parabola P ∧ d = distance A P + distance B P }

theorem min_ap_bp_is_9 : min_ap_bp_value = 9 := by
  sorry

end min_ap_bp_is_9_l572_572326


namespace probability_log_base_three_is_integer_l572_572923

theorem probability_log_base_three_is_integer :
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let numbers_with_integer_log_base_3 := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = 3^k}
  (numbers_with_integer_log_base_3.card : ℚ) / (three_digit_numbers.card) = 1 / 450 :=
by
  sorry

end probability_log_base_three_is_integer_l572_572923


namespace perimeter_of_given_triangle_l572_572929

structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (right_triangle : (a^2 + b^2 = c^2))

def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

def given_triangle := {a := 10, b := 30, c := 10 * real.sqrt 10, right_triangle := by simp [pow_two, mul_self_sqrt, pow_two]}

theorem perimeter_of_given_triangle : perimeter given_triangle = 40 + 10 * real.sqrt 10 := sorry

end perimeter_of_given_triangle_l572_572929


namespace ellipse_equation_segment_length_l572_572054

def ellipse_eq (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2/a^2 + y^2/b^2 = 1)

noncomputable def ellipse_parameters_satisfy (a b : ℝ) : Prop :=
  2 * a * b = 2 * real.sqrt 2 ∧ a^2 + b^2 = 3 ∧ a > b ∧ b > 0

noncomputable def line_passing_Q (k : ℝ) : Prop :=
  ∃ (x y : ℝ), y = k * x - 2

theorem ellipse_equation
  (a b : ℝ) (hp : ellipse_parameters_satisfy a b) :
  ellipse_eq (real.sqrt 2) 1 :=
by
  sorry

theorem segment_length
  (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (hl : line_passing_Q k)
  (hxy1 : (x1^2 / 2) + y1^2 = 1)
  (hxy2 : (x2^2 / 2) + y2^2 = 1)
  (hprod_of_slopes : (y1 * y2) / (x1 * x2) = -1) :
  (real.sqrt (1 + k^2) * real.sqrt ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * real.sqrt 21 / 11 :=
by
  sorry

end ellipse_equation_segment_length_l572_572054


namespace parabolas_intersections_l572_572989
-- Import the necessary library

-- Define the theorem based on the identified tuple
theorem parabolas_intersections :
  let a_values := [-3, -2, -1, 0, 1, 2, 3]
  let b_values := [-2, -1, 0, 1, 2]
  (∑ a1 in a_values, ∑ a2 in a_values, a1 ≠ a2) * 2 + 
  (∑ a in a_values, ∑ b1 in b_values, ∑ b2 in b_values, a ≠ 0 → (b1 - b2) * (b2 - b1) = 0) * 2 = 814 :=
by
  sorry

end parabolas_intersections_l572_572989


namespace max_value_fraction_l572_572579

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 :=
by
  sorry

end max_value_fraction_l572_572579


namespace divisors_of_180_not_divisible_by_3_l572_572682

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572682


namespace regularDodecagonProduct_is_531440_l572_572482

open Complex

noncomputable def regularDodecagonProduct (Q1 Q7 : ℂ) (n : ℕ) : ℂ :=
  if Q1 = Complex.mk 2 0 ∧ Q7 = Complex.mk 4 0 ∧ n = 12 then
    let q := (z : ℂ) -> (z - 3) ^ n - 1
    let roots := Finset.univ.filter (λ z, q z = 0)
    roots.prod id
  else
    0

theorem regularDodecagonProduct_is_531440 :
  regularDodecagonProduct (Complex.mk 2 0) (Complex.mk 4 0) 12 = 531440 :=
by
  sorry

end regularDodecagonProduct_is_531440_l572_572482


namespace chessboard_max_distance_l572_572737

theorem chessboard_max_distance :
  ∀ (a b : Fin 8 → Fin 8), 
  (∀ i j : Fin 8, i ≠ j → a i ≠ a j ∧ b i ≠ b j) →
  (∀ i j : Fin 8, i ≠ j → a i ≠ b j) →
  ∑ i, abs (a i - b i : ℤ) ≤ 32 :=
by
  sorry

end chessboard_max_distance_l572_572737


namespace divisors_not_divisible_by_3_eq_6_l572_572659

theorem divisors_not_divisible_by_3_eq_6 :
  let n := 180 in
  let prime_factorization := (2^2 * 3^2 * 5^1) in
  (∀ d ∣ n, ¬ (3 ∣ d) → (∃! (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ a ≤ 2 ∧ b = 0 ∧ c ≤ 1)) →
  (finset.card {d ∣ n | ¬ (3 ∣ d)} = 6) :=
by
  let n := 180
  let prime_factorization := (2^2 * 3^2 * 5^1)
  have h : prime_factorization = n := by norm_num
  let s := finset.filter (λ d, ¬3 ∣ d) (finset.divisors n)
  suffices : finset.card s = 6
  { exact this }
  sorry

end divisors_not_divisible_by_3_eq_6_l572_572659


namespace tan_product_identity_l572_572104

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572104


namespace sum_of_consecutive_2022_l572_572570

theorem sum_of_consecutive_2022 :
  ∃ (a p : ℕ), (∑ i in range p, (a + i)) = 2022 ∧ 
  ((a = 673 ∧ p = 3) ∨ (a = 504 ∧ p = 4) ∨ (a = 163 ∧ p = 12)) :=
by
  sorry

end sum_of_consecutive_2022_l572_572570


namespace value_of_a2020_l572_572258

def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = Real.sqrt 6 ∧ 
  ∀ n, a (n + 1) = Real.floor (a n) + 1 / (a n - Real.floor (a n))

theorem value_of_a2020 (a : ℕ → ℝ) (h : sequence a) : a 2020 = 6060 + Real.sqrt 6 :=
sorry

end value_of_a2020_l572_572258


namespace problem1_l572_572459

variables (m n : ℝ)

axiom cond1 : 4 * m + n = 90
axiom cond2 : 2 * m - 3 * n = 10

theorem problem1 : (m + 2 * n) ^ 2 - (3 * m - n) ^ 2 = -900 := sorry

end problem1_l572_572459


namespace opposite_of_abs_neg_five_l572_572407

-- Define the absolute value function and negation for better readability
def abs_five : Int := Int.abs (-5)
def neg_abs_five : Int := -abs_five

-- Theorem statement that needs to be proved
theorem opposite_of_abs_neg_five : neg_abs_five = -5 := by
  -- Proof will be written here
  sorry

end opposite_of_abs_neg_five_l572_572407


namespace compute_expression_l572_572069

theorem compute_expression : 42 * 52 + 48 * 42 = 4200 :=
by sorry

end compute_expression_l572_572069


namespace largest_magnitude_u3_v3_l572_572613

variable (u v : ℂ)

theorem largest_magnitude_u3_v3 (h1 : |u + v| = 3) (h2 : |u^2 + v^2| = 10) :
  |u^3 + v^3| ≤ 31.5 :=
sorry

end largest_magnitude_u3_v3_l572_572613


namespace second_metal_gold_percentage_l572_572473

theorem second_metal_gold_percentage (w_final : ℝ) (p_final : ℝ) (w_part : ℝ) (p_part1 : ℝ) (w_part1 : ℝ) (w_part2 : ℝ)
  (h_w_final : w_final = 12.4) (h_p_final : p_final = 0.5) (h_w_part : w_part = 6.2) (h_p_part1 : p_part1 = 0.6)
  (h_w_part1 : w_part1 = 6.2) (h_w_part2 : w_part2 = 6.2) :
  ∃ p_part2 : ℝ, p_part2 = 0.4 :=
by sorry

end second_metal_gold_percentage_l572_572473


namespace sum_of_numbers_l572_572877

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l572_572877


namespace each_half_month_has_15_days_l572_572838

noncomputable def days_in_each_half (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) : ℕ :=
  let first_half_days := total_days / 2
  let second_half_days := total_days - first_half_days
  first_half_days

theorem each_half_month_has_15_days (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) :
  total_days = 30 → mean_profit_total = 350 → mean_profit_first_half = 275 → mean_profit_last_half = 425 → 
  days_in_each_half total_days mean_profit_total mean_profit_first_half mean_profit_last_half = 15 :=
by
  intros h_days h_total h_first h_last
  sorry

end each_half_month_has_15_days_l572_572838


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572695

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572695


namespace f_2017_eq_2018_l572_572547

def f : ℕ → ℕ
| 1 := 2
| n := if n % 2 = 0 then f (n - 1) + 1 else if n > 1 then f (n - 2) + 2 else 0  -- fallback for n=0

theorem f_2017_eq_2018 : f 2017 = 2018 :=
by
  sorry

end f_2017_eq_2018_l572_572547


namespace triangle_perimeter_l572_572956

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572956


namespace pythagorean_triple_divisibility_l572_572777

theorem pythagorean_triple_divisibility (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 = z^2) :
  (∃ a b c : ℕ, a^2 + b^2 = c^2 ∧ ((a % 3 = 0 ∨ b % 3 = 0 ∨ c % 3 = 0) ∧
                                    (a % 4 = 0 ∨ b % 4 = 0 ∨ c % 4 = 0) ∧
                                    (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0))) :=
begin
  sorry
end

end pythagorean_triple_divisibility_l572_572777


namespace trapezium_area_proof_l572_572013

def trapeziumArea (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area_proof :
  let a := 20
  let b := 18
  let h := 14
  trapeziumArea a b h = 266 := by
  sorry

end trapezium_area_proof_l572_572013


namespace total_animals_count_l572_572517

def snakes := 100
def arctic_foxes := 80
def leopards := 20
def bee_eaters := 10 * leopards
def cheetahs := snakes / 2
def alligators := 2 * (arctic_foxes + leopards)

theorem total_animals_count : snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 := by
  calc
    snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators
        = 100 + 80 + 20 + 200 + 50 + 200 : by sorry -- Use the defined values
    ... = 650 : by sorry -- Perform the final addition

end total_animals_count_l572_572517


namespace abs_diff_eq_one_l572_572772

def quadratic (a b c : ℤ) : ℤ → ℤ := λ x, a * x^2 + b * x + c

theorem abs_diff_eq_one (a b c m n : ℤ) (hf : quadratic a b c m - quadratic a b c n = 1) : 
  |m - n| = 1 := sorry

end abs_diff_eq_one_l572_572772


namespace submerged_height_and_weight_of_cone_l572_572044

variable (s m α : ℝ)

theorem submerged_height_and_weight_of_cone (h_s : 0 < s ∧ s < 1) :
  let x := m * (root3 (1 / (1 - s)) - 1),
      Q := (π * m^3 * s / (3 * (1 - s))) * (tan (α / 2))^2
  in x = m * (root3 (1 / (1 - s)) - 1) ∧
     Q = (π * m^3 * s / (3 * (1 - s))) * (tan (α / 2))^2 :=
by sorry

end submerged_height_and_weight_of_cone_l572_572044


namespace tan_identity_l572_572151

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572151


namespace distinct_sums_in_table_l572_572800

theorem distinct_sums_in_table (N : ℕ) (hN : N > 10) (distinct_ints : finset ℕ) (h_card : distinct_ints.card = N^2) : 
  ∃ table : list (list ℕ), (length table = N ∧ (∀ i < N, length (table.nth_le i _) = N) ∧ 
  ∀ i j k l, i < N → j < N → k < N → l < N → (i ≠ k ∨ j ≠ l) → 
  (table.nth_le i _).sum ≠ (table.nth_le k _).sum ∧ (table.nth_le j _).sum ≠ (table.nth_le l _)) := 
begin
  sorry
end

end distinct_sums_in_table_l572_572800


namespace tan_identity_l572_572153

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572153


namespace polynomial_divisibility_l572_572412

noncomputable def P : ℕ → ℂ[X]
| 1 := 2 * X
| 2 := 2 * (X^2 + 1)
| (n + 2) := 2 * X * (P (n + 1)) - (X^2 - 1) * (P n)

theorem polynomial_divisibility (n : ℕ) : 
  (X^2 + 1) ∣ P n ↔ n % 4 = 2 := 
sorry

end polynomial_divisibility_l572_572412


namespace angle_equality_l572_572527

open EuclideanGeometry

-- Definitions of points in the conditions of the problem
variables {Ω Γ δ : Circle}
variables (A B P Q D T K L : Point)
variables (h_intersect_ΩΓ : Ω ∩ Γ = {A, B})
variables (h_on_line_center : P ∈ Ω ∧ Q ∈ Γ ∧ (line_through_center Ω Γ).contains P ∧ (line_through_center Ω Γ).contains Q)
variables (h_same_side : same_side (line_through A B) P Q)
variables (h_closer : dist Q (line_through A B) < dist P (line_through A B))
variables (h_touch_δ : touches δ (segment A B) D ∧ touches δ Γ T)
variables (h_lie_δ : delta_side (line_through A B) δ P ∧ delta_side (line_through A B) δ Q ∧ delta_side (line_through A B) δ D ∧ delta_side (line_through A B) δ T)
variables (h_PD_δΩ : (line_through P D).meets δ K ∧ (line_through P D).meets Ω L)

-- Theorem to prove
theorem angle_equality : ∠ Q T K = ∠ D T L :=
by
  sorry

end angle_equality_l572_572527


namespace cost_of_dozen_chocolate_bars_infinite_solutions_y_z_l572_572298

variables (x y z : ℝ)

-- Given conditions
def condition1 : Prop := 4 * x = 8
def condition2 : Prop := 1 = 1
def condition3 : Prop := 2 * y + 3 * z = 5

-- Prove that 12 * x = 24
theorem cost_of_dozen_chocolate_bars (h1 : condition1) : 12 * x = 24 := by
  have x_eq_2 : x = 2 := by linarith [h1]
  linarith [x_eq_2]

-- The equation 2 * y + 3 * z = 5 has multiple solutions
theorem infinite_solutions_y_z (h3 : condition3) : ∃ (y1 y2 z1 z2 : ℝ), (2 * y1 + 3 * z1 = 5) ∧ (2 * y2 + 3 * z2 = 5) ∧ (y1 ≠ y2 ∨ z1 ≠ z2) := by
  sorry

end cost_of_dozen_chocolate_bars_infinite_solutions_y_z_l572_572298


namespace percentage_first_less_third_l572_572476

variable (A B C : ℝ)

theorem percentage_first_less_third :
  B = 0.58 * C → B = 0.8923076923076923 * A → (100 - (A / C * 100)) = 35 :=
by
  intros h₁ h₂
  sorry

end percentage_first_less_third_l572_572476


namespace sum_of_numbers_l572_572878

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l572_572878


namespace problem_statement_l572_572965

-- Let A and B be non-empty sets
variable (A B : Type) [Nonempty A] [Nonempty B]

-- Let f be a function from A to B
variable (f : A → B)

-- Statement definitions
def statement1 : Prop := ∀ y ∈ set.range f, ∃! x, f x = y
def statement2 : Prop := Nonempty A ∧ Nonempty (set.range f)
def statement3 : Prop := ∀ y, y ∈ set.range f ↔ ∃ x, f x = y
def statement4 : Prop := ∀ x y, (x ≠ y → f x ≠ f y) ∨ (A = ({x} : set A) → set.range f = {f x})

-- Proof statements
theorem problem_statement : ¬ statement1 ∧ statement2 ∧ statement3 ∧ statement4 :=
sorry

end problem_statement_l572_572965


namespace calculate_A_l572_572730

theorem calculate_A (D B E C A : ℝ) :
  D = 2 * 4 →
  B = 2 * D →
  E = 7 * 2 →
  C = 7 * E →
  A^2 = B * C →
  A = 28 * Real.sqrt 2 :=
by
  sorry

end calculate_A_l572_572730


namespace tan_product_eq_three_l572_572071

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572071


namespace fraction_candy_unclaimed_l572_572050

theorem fraction_candy_unclaimed (x : ℝ) :
  let al_share := (4/9) * x,
      bert_share := (1/3) * (5/9) * x,
      carl_share := (2/9) * (10/27) * x,
      remaining := x - al_share - bert_share - carl_share
  in remaining / x = 230 / 243 :=
sorry

end fraction_candy_unclaimed_l572_572050


namespace nonnegative_integer_solutions_l572_572175

theorem nonnegative_integer_solutions (x y : ℕ) :
  3 * x^2 + 2 * 9^y = x * (4^(y+1) - 1) ↔ (x, y) ∈ [(2, 1), (3, 1), (3, 2), (18, 2)] :=
by sorry

end nonnegative_integer_solutions_l572_572175


namespace triangle_ratio_and_angle_l572_572278

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinA sinB sinC : ℝ)

theorem triangle_ratio_and_angle
  (h_triangle : a / sinA = b / sinB ∧ b / sinB = c / sinC)
  (h_sin_ratio : sinA / sinB = 5 / 7 ∧ sinB / sinC = 7 / 8) :
  (a / b = 5 / 7 ∧ b / c = 7 / 8) ∧ B = 60 :=
by
  sorry

end triangle_ratio_and_angle_l572_572278


namespace minimum_sequences_needed_l572_572503

theorem minimum_sequences_needed :
  ∀ (a : Fin 4 → ℕ), ∃ (b₁ b₂ : Fin 4 → ℕ),
  (∃ S₁, S₁ = ∑ i, a i * b₁ i) ∧
  (∃ S₂, S₂ = ∑ i, a i * b₂ i) ∧
  ∀ (b₁ b₂ b₃ : Fin 4 → ℕ),
  (2016 <  ∑ i, a i * b₃ i) → ∃ b₃, false :=
begin
  sorry
end

end minimum_sequences_needed_l572_572503


namespace figure_50_squares_l572_572180

def sequence (n : ℕ) : ℕ := 2*n^2 + 4*n + 2

theorem figure_50_squares : sequence 50 = 5202 := by
  sorry

end figure_50_squares_l572_572180


namespace tangent_product_eq_three_l572_572113

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l572_572113


namespace brownie_leftover_is_zero_l572_572309

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 15

-- Define the dimensions of one piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- The total area of the pan
def pan_area : ℕ := pan_length * pan_width

-- The total area of one piece
def piece_area : ℕ := piece_length * piece_width

-- The number of full pieces that can be cut
def number_of_pieces : ℕ := pan_area / piece_area

-- The total used area when pieces are cut
def used_area : ℕ := number_of_pieces * piece_area

-- The leftover area
def leftover_area : ℕ := pan_area - used_area

theorem brownie_leftover_is_zero (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24) (h2 : pan_width = 15) 
  (h3 : piece_length = 3) (h4 : piece_width = 4) :
  pan_width * pan_length - (pan_width * pan_length / (piece_width * piece_length)) * (piece_width * piece_length) = 0 := 
by sorry

end brownie_leftover_is_zero_l572_572309


namespace matrix_example_l572_572018

/-- Example of a matrix for which the absolute sum diverges and does not satisfy 
the equalities in formula (*) from problem II.6.12 -/
theorem matrix_example (a : ℕ → ℕ → ℝ) :
  (∀ i j, a i j = if i = j then 0 else (i - j : ℕ)⁻³) →
  (∑ i j, |a i j| = ∞) ∧ ¬ (formula_star_II_6_12 a) := 
begin
  intros h,
  sorry,
end

end matrix_example_l572_572018


namespace right_triangle_perimeter_l572_572949

theorem right_triangle_perimeter (a b : ℝ) (h₁ : a = 30) (h₂ : 0 < b) (h₃ : 0.5 * a * b = 150) : 
  let c := Real.sqrt (a^2 + b^2)
  in a + b + c = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572949


namespace determine_q_l572_572821

noncomputable def q (x : ℝ) := x^3 - (82/13)*x^2 + (277/13)*x - (390/13)

theorem determine_q :
  (∀ x : ℝ, q(2 - 3 * complex.I) = 0 ∧ q(0) = -30 ∧ ∀ x : ℝ, polynomial.monic (λ x, q(x))) → 
  (q(x) = x^3 - (82/13)*x^2 + (277/13)*x - (390/13)) := 
by
  sorry

end determine_q_l572_572821


namespace collin_initial_flowers_l572_572984

theorem collin_initial_flowers (ingrid_flowers : ℕ) (flowers_per_petal: ℕ) (collin_final_petals: ℕ) :
  ingrid_flowers = 33 →
  flowers_per_petal = 4 →
  collin_final_petals = 144 →
  let ingrid_flowers_given := ingrid_flowers / 3 in
  let ingrid_petals_given := ingrid_flowers_given * flowers_per_petal in
  let collin_initial_petals := collin_final_petals - ingrid_petals_given in
  let collin_initial_flowers := collin_initial_petals / flowers_per_petal in
  collin_initial_flowers = 25 :=
by {
  intros h1 h2 h3,
  let ingrid_flowers_given := ingrid_flowers / 3,
  let ingrid_petals_given := ingrid_flowers_given * flowers_per_petal,
  let collin_initial_petals := collin_final_petals - ingrid_petals_given,
  let collin_initial_flowers := collin_initial_petals / flowers_per_petal,
  sorry
}

end collin_initial_flowers_l572_572984


namespace arthur_dinner_cost_l572_572849

theorem arthur_dinner_cost :
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  final_cost = 38 :=
by
  let appetizer := 8
  let ribeye := 20
  let wine_per_glass := 3
  let num_glasses := 2
  let cheesecake := 6
  let voucher_discount := 1 / 2
  let tip_percentage := 0.20
  let cost_before_voucher := appetizer + ribeye + wine_per_glass * num_glasses + cheesecake
  let tip := cost_before_voucher * tip_percentage
  let savings := ribeye * voucher_discount
  let final_cost := cost_before_voucher - savings + tip
  show final_cost = 38, from sorry

end arthur_dinner_cost_l572_572849


namespace arthur_spent_38_l572_572855

noncomputable def arthur_meal_cost : ℝ := 
  let appetizer := 8
  let entree := 20
  let wine := 3 * 2
  let dessert := 6
  let total_no_discount := appetizer + entree + wine + dessert
  let discount := entree / 2
  let total_with_discount := total_no_discount - discount
  let tip := total_no_discount * 0.20
  total_with_discount + tip

theorem arthur_spent_38 :
  arthur_meal_cost = 38 :=
by
  rw [arthur_meal_cost]
  have appetizer : ℝ := 8
  have entree : ℝ := 20
  have wine : ℝ := 3 * 2
  have dessert : ℝ := 6
  have total_no_discount : ℝ := appetizer + entree + wine + dessert
  have discount : ℝ := entree / 2 
  have total_with_discount : ℝ := total_no_discount - discount
  have tip : ℝ := total_no_discount * 0.20
  calc  total_with_discount + tip = (8 + 20 + 6 + 2 * 3) - (20 / 2) + 0.20 * (8 + 20 + 6 + 2 * 3) : by rw [total_no_discount]
                          ... = 38 : by norm_num

end arthur_spent_38_l572_572855


namespace probability_both_slate_rocks_l572_572417

noncomputable def totalRocks := 12 + 17 + 8
noncomputable def P_slate_first_draw := 12 / totalRocks.to_rat
noncomputable def remainingRocks_after_first_draw := totalRocks - 1
noncomputable def P_slate_second_draw := 11 / remainingRocks_after_first_draw.to_rat
noncomputable def P_both_slate := P_slate_first_draw * P_slate_second_draw

theorem probability_both_slate_rocks :
  P_both_slate = 132 / 1332 :=
by
  sorry

end probability_both_slate_rocks_l572_572417


namespace tan_product_l572_572133

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572133


namespace binary_representation_three_consecutive_identical_digits_l572_572776

theorem binary_representation_three_consecutive_identical_digits (x : ℕ) (h : x > 2) : 
  ∃ d : Bool, (x^2 - 1).toDigits 2 = [d, d, d] ∨ [d, d, d, d] ∨ [d, d, d, d, d, d, d] := sorry

end binary_representation_three_consecutive_identical_digits_l572_572776


namespace triangle_DAB_angle_l572_572020

-- Definitions from the conditions of the problem
def Point := ℝ × ℝ
def dist (p q : Point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def right_angled (a b c : Point) (B : Point) : Prop :=
  dist a b ^ 2 + dist b c ^ 2 = dist a c ^ 2

-- Problem statement to be proven in Lean
theorem triangle_DAB_angle 
  (A B D C : Point)
  (h_right_angle : right_angled A B D B)
  (h_AC_equals_CD : dist A C = dist C D)
  (h_AB_equals_2BC : dist A B = 2 * dist B C) :
  ∠ D A B = 45 :=
sorry

end triangle_DAB_angle_l572_572020


namespace Carrie_total_money_l572_572525

-- Define the quantities harvested
def tomatoes := 200
def carrots := 350
def eggplants := 120
def cucumbers := 75

-- Define the selling prices
def price_tomato := 1
def price_carrot := 1.5
def price_eggplant := 2.5
def price_cucumber := 1.75

-- Calculate the total money from each vegetable type being sold
def total_tomatoes := tomatoes * price_tomato
def total_carrots := carrots * price_carrot
def total_eggplants := eggplants * price_eggplant
def total_cucumbers := cucumbers * price_cucumber

-- Calculate the total money Carrie can make by selling all vegetables
def total_money := total_tomatoes + total_carrots + total_eggplants + total_cucumbers

-- Prove that the total money equals $1156.25
theorem Carrie_total_money : total_money = 1156.25 :=
by
  sorry

end Carrie_total_money_l572_572525


namespace motion_is_indeterminate_l572_572466

-- Define the stationary point P
constant P : Type

-- Define the point Q describing a circle around P
constant Q : Type

-- Define the radius r
constant r : ℝ

-- Define the distance function between points
constant distance : Q → P → ℝ

-- Given conditions
axiom stationary_point : ∀ (p : P), p = P
axiom circle_motion : ∀ (q : Q), distance q P = r

-- Theorem stating that the motion is indeterminate
theorem motion_is_indeterminate : true := sorry

end motion_is_indeterminate_l572_572466


namespace players_at_least_two_sciences_l572_572056

-- Define the conditions of the problem
def total_players : Nat := 30
def players_biology : Nat := 15
def players_chemistry : Nat := 10
def players_physics : Nat := 5
def players_all_three : Nat := 3

-- Define the main theorem we want to prove
theorem players_at_least_two_sciences :
  (players_biology + players_chemistry + players_physics 
    - players_all_three - total_players) = 9 :=
sorry

end players_at_least_two_sciences_l572_572056


namespace plot_length_l572_572034

-- Define the conditions
def rent_per_acre_per_month : ℝ := 30
def total_rent_per_month : ℝ := 300
def width_feet : ℝ := 1210
def area_acres : ℝ := 10
def square_feet_per_acre : ℝ := 43560

-- Prove that the length of the plot is 360 feet
theorem plot_length (h1 : rent_per_acre_per_month = 30)
                    (h2 : total_rent_per_month = 300)
                    (h3 : width_feet = 1210)
                    (h4 : area_acres = 10)
                    (h5 : square_feet_per_acre = 43560) :
  (area_acres * square_feet_per_acre) / width_feet = 360 := 
by {
  sorry
}

end plot_length_l572_572034


namespace prime_pairs_square_l572_572568

noncomputable def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem prime_pairs_square (a b : ℤ) (ha : is_prime a) (hb : is_prime b) :
  is_square (3 * a^2 * b + 16 * a * b^2) ↔ (a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3) :=
by
  sorry

end prime_pairs_square_l572_572568


namespace Anne_height_l572_572970

-- Define the conditions
variables (S : ℝ)   -- Height of Anne's sister
variables (A : ℝ)   -- Height of Anne
variables (B : ℝ)   -- Height of Bella

-- Define the relations according to the problem's conditions
def condition1 (S : ℝ) := A = 2 * S
def condition2 (S : ℝ) := B = 3 * A
def condition3 (S : ℝ) := B - S = 200

-- Theorem statement to prove Anne's height
theorem Anne_height (S : ℝ) (A : ℝ) (B : ℝ)
(h1 : A = 2 * S) (h2 : B = 3 * A) (h3 : B - S = 200) : A = 80 :=
by sorry

end Anne_height_l572_572970


namespace triangle_perimeter_l572_572954

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 30) 
  (h2 : b = 10) 
  (h3 : c = real.sqrt (a^2 + b^2)) 
  (h4 : (1 / 2) * a * b = 150) : 
  a + b + c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end triangle_perimeter_l572_572954


namespace broker_investment_increase_l572_572029

noncomputable def final_value_stock_A := 
  let initial := 100.0
  let year1 := initial * (1 + 0.80)
  let year2 := year1 * (1 - 0.30)
  year2 * (1 + 0.10)

noncomputable def final_value_stock_B := 
  let initial := 100.0
  let year1 := initial * (1 + 0.50)
  let year2 := year1 * (1 - 0.10)
  year2 * (1 - 0.25)

noncomputable def final_value_stock_C := 
  let initial := 100.0
  let year1 := initial * (1 - 0.30)
  let year2 := year1 * (1 - 0.40)
  year2 * (1 + 0.80)

noncomputable def final_value_stock_D := 
  let initial := 100.0
  let year1 := initial * (1 + 0.40)
  let year2 := year1 * (1 + 0.20)
  year2 * (1 - 0.15)

noncomputable def total_final_value := 
  final_value_stock_A + final_value_stock_B + final_value_stock_C + final_value_stock_D

noncomputable def initial_total_value := 4 * 100.0

noncomputable def net_increase := total_final_value - initial_total_value

noncomputable def net_increase_percentage := (net_increase / initial_total_value) * 100

theorem broker_investment_increase : net_increase_percentage = 14.5625 := 
by
  sorry

end broker_investment_increase_l572_572029


namespace quadrilateral_area_is_correct_l572_572477

def point := (ℝ, ℝ)

def A : point := (1, 2)
def B : point := (1, -1)
def C : point := (3, -1)
def D : point := (7, 8)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def triangle_area (p1 p2 p3 : point) : ℝ :=
  let base := distance p1 p2
  let height := (abs ((p3.1 - p1.1) * (p1.2 - p2.2) - (p1.1 - p2.1) * (p3.2 - p1.2)) / distance p1 p2) in
  0.5 * base * height

noncomputable def quadrilateral_area (A B C D : point) : ℝ :=
  triangle_area A B C + triangle_area A C D

theorem quadrilateral_area_is_correct :
  quadrilateral_area A B C D = 9.364 :=
by
  sorry

end quadrilateral_area_is_correct_l572_572477


namespace smallest_N_for_odd_sum_greater_than_2021_l572_572472

theorem smallest_N_for_odd_sum_greater_than_2021 :
  let a : ℕ → ℚ := λ n, if n = 0 then 3 else if n = 1 then 4 else (a (n - 1) + 1) / a (n - 2) in
  ∃ N : ℕ, (∑ i in finset.range N, a i).natAbs % 2 = 1 ∧ (∑ i in finset.range N, a i) > 2021 ∧ N = 989 :=
sorry

end smallest_N_for_odd_sum_greater_than_2021_l572_572472


namespace man_swim_distance_downstream_l572_572917

noncomputable def DistanceDownstream (Vm : ℝ) (Vupstream : ℝ) (time : ℝ) : ℝ :=
  let Vs := Vm - Vupstream
  let Vdownstream := Vm + Vs
  Vdownstream * time

theorem man_swim_distance_downstream :
  let Vm : ℝ := 3  -- speed of man in still water in km/h
  let time : ℝ := 6 -- time taken in hours
  let d_upstream : ℝ := 12 -- distance swum upstream in km
  let Vupstream : ℝ := d_upstream / time
  DistanceDownstream Vm Vupstream time = 24 := sorry

end man_swim_distance_downstream_l572_572917


namespace solve_system_of_equations_l572_572819

def system_of_equations(x y z: ℝ): Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

theorem solve_system_of_equations :
  ∀ (x y z: ℝ), system_of_equations x y z ↔
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2 ∨ z = -1) ∨
  (x = -3 ∧ y = 2 ∨ z = 1) :=
by
  sorry

end solve_system_of_equations_l572_572819


namespace tan_product_identity_l572_572101

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572101


namespace fraction_product_l572_572063

theorem fraction_product :
  (3 / 7) * (5 / 8) * (9 / 13) * (11 / 17) = 1485 / 12376 := 
by
  sorry

end fraction_product_l572_572063


namespace roots_sum_powers_l572_572775

theorem roots_sum_powers (t : ℕ → ℝ) (b d f : ℝ)
  (ht0 : t 0 = 3)
  (ht1 : t 1 = 6)
  (ht2 : t 2 = 11)
  (hrec : ∀ k ≥ 2, t (k + 1) = b * t k + d * t (k - 1) + f * t (k - 2))
  (hpoly : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0) :
  b + d + f = 13 :=
sorry

end roots_sum_powers_l572_572775


namespace quadratic_real_roots_condition_l572_572274

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0) → m ≤ 1/4 :=
by
  sorry

end quadratic_real_roots_condition_l572_572274


namespace divisors_not_divisible_by_3_l572_572676

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572676


namespace circle_area_circumference_l572_572176

def R : ℝ × ℝ := (2, -1)
def S : ℝ × ℝ := (7, 4)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def radius : ℝ := distance R S

def area (r : ℝ) : ℝ := real.pi * r^2
def circumference (r : ℝ) : ℝ := 2 * real.pi * r

theorem circle_area_circumference :
  area radius = 50 * real.pi ∧ circumference radius = 10 * real.pi * real.sqrt 2 :=
by
  sorry

end circle_area_circumference_l572_572176


namespace f_minus4_plus_f_zero_range_of_a_l572_572622

def f (x : ℝ) : ℝ := if x > 0 then Real.log x / Real.log 2 else if x < 0 then -(Real.log (-x) / Real.log 2) else 0

theorem f_minus4_plus_f_zero : f (-4) + f 0 = -2 := 
by sorry

theorem range_of_a (a : ℝ) : f a > f (-a) ↔ (a > 1 ∨ (-1 < a ∧ a < 0)) := 
by sorry

end f_minus4_plus_f_zero_range_of_a_l572_572622


namespace cylinder_volume_relation_l572_572995

def volume (r h : ℝ) : ℝ := π * r^2 * h

variable (r h : ℝ)

theorem cylinder_volume_relation
  (vol_A : volume r h = 2 * volume h r)
  : ∃ N : ℝ, volume r h = N * π * h^3 ∧ N = 4 := by
  sorry

end cylinder_volume_relation_l572_572995


namespace rahul_spends_10_percent_on_clothes_l572_572589

theorem rahul_spends_10_percent_on_clothes 
    (salary : ℝ) (house_rent_percent : ℝ) (education_percent : ℝ) (remaining_after_expense : ℝ) (expenses : ℝ) (clothes_percent : ℝ) 
    (h_salary : salary = 2125) 
    (h_house_rent_percent : house_rent_percent = 0.20)
    (h_education_percent : education_percent = 0.10)
    (h_remaining_after_expense : remaining_after_expense = 1377)
    (h_expenses : expenses = salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)
    (h_clothes_expense : remaining_after_expense = salary - (salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)) :
    clothes_percent = 0.10 := 
by 
  sorry

end rahul_spends_10_percent_on_clothes_l572_572589


namespace smallest_palindrome_satisfying_conditions_infinitely_many_palindromes_satisfying_conditions_l572_572905

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem smallest_palindrome_satisfying_conditions :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 3 [MOD 4]) ∧ (n ≡ 0 [MOD 5]) ∧ is_palindrome n ∧ 
  (∀ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 3 [MOD 4]) ∧ (m ≡ 0 [MOD 5]) ∧ is_palindrome m → n ≤ m) :=
begin
  -- proof omitted
  sorry,
end

theorem infinitely_many_palindromes_satisfying_conditions :
  ∀ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 3 [MOD 4]) ∧ (n ≡ 0 [MOD 5]) ∧ is_palindrome n → 
  ∃ m : ℕ, m > n ∧ (m ≡ 2 [MOD 3]) ∧ (m ≡ 3 [MOD 4]) ∧ (m ≡ 0 [MOD 5]) ∧ is_palindrome m :=
begin
  -- proof omitted
  sorry,
end

end smallest_palindrome_satisfying_conditions_infinitely_many_palindromes_satisfying_conditions_l572_572905


namespace no_t_for_xyz_equal_l572_572276

theorem no_t_for_xyz_equal (t : ℝ) (x y z : ℝ) : 
  (x = 1 - 3 * t) → 
  (y = 2 * t - 3) → 
  (z = 4 * t^2 - 5 * t + 1) → 
  ¬ (x = y ∧ y = z) := 
by
  intro h1 h2 h3 h4
  have h5 : t = 4 / 5 := 
    by linarith [h1, h2, h4]
  rw [h5] at h3
  sorry

end no_t_for_xyz_equal_l572_572276


namespace evaluate_expression_l572_572563

def a : ℚ := 7/3

theorem evaluate_expression :
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140 / 27 :=
by
  sorry

end evaluate_expression_l572_572563


namespace stella_weeks_l572_572795

-- Define the constants used in the conditions
def rolls_per_bathroom_per_day : ℕ := 1
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def rolls_per_pack : ℕ := 12
def packs_bought : ℕ := 14

-- Define the total number of rolls Stella uses per day and per week
def rolls_per_day := rolls_per_bathroom_per_day * bathrooms
def rolls_per_week := rolls_per_day * days_per_week

-- Calculate the total number of rolls bought
def total_rolls_bought := packs_bought * rolls_per_pack

-- Calculate the number of weeks Stella bought toilet paper for
def weeks := total_rolls_bought / rolls_per_week

theorem stella_weeks : weeks = 4 := by
  sorry

end stella_weeks_l572_572795


namespace prob_factor_120_less_than_9_l572_572439

/-
Problem:
What is the probability that a randomly drawn positive factor of 120 is less than 9?
-/

def num_factors (n : ℕ) : ℕ :=
  (factors n).eraseDuplicates.length

def factors_less_than (n m : ℕ) : List ℕ :=
  (factors n).filter (λ d => d < m)

def prob_factors_less_than (n m : ℕ) : ℚ :=
  (factors_less_than n m).length / (num_factors n)

theorem prob_factor_120_less_than_9 : prob_factors_less_than 120 9 = 7 / 16 := 
sorry

end prob_factor_120_less_than_9_l572_572439


namespace tan_angle_NBM_l572_572346

theorem tan_angle_NBM (A B C D M N : Type) 
  [is_trapezoid A B C D]
  (AD BC : Real := 255)
  (AB : Real := 128)
  (M_midpoint : midpoint M C D)
  (N_foot_perpendicular : foot_perpendicular N A C D)
  (angle_MBC_right : ∠MBC = 90) :
  tan (angle NBM) = 120 / 353 :=
sorry

end tan_angle_NBM_l572_572346


namespace integral_of_one_integral_of_half_x_plus_one_l572_572564

theorem integral_of_one :
  ∫ x in 0..2, (1 : ℝ) = 2 := by
  sorry

theorem integral_of_half_x_plus_one :
  ∫ x in 0..2, ((1/2 : ℝ) * x + 1) = 3 := by
  sorry

end integral_of_one_integral_of_half_x_plus_one_l572_572564


namespace magnitude_z_add_i_l572_572194

noncomputable def z : ℂ := by sorry -- Assume z is a complex number with the given condition

theorem magnitude_z_add_i (z : ℂ) (h : (1 + I) * z = (2 - I)) : 
  |z + I| = (Real.sqrt 2) / 2 := by
sorry

end magnitude_z_add_i_l572_572194


namespace equilateral_centroid_perimeter_l572_572500

theorem equilateral_centroid_perimeter :
  ∃ (T : Triangle) (r : ℝ) (C1 C2 C3 : Point),
    T.is_equilateral ∧
    T.side_length = 18 ∧
    r = 4 ∧
    T.has_inscribed_circles_touching_sides C1 C2 C3 r ∧
    (triangle_perimeter (triangle_of_points C1 C2 C3)) = 30 + 15 * Real.sqrt 3 :=
by
  sorry

end equilateral_centroid_perimeter_l572_572500


namespace time_to_complete_job_together_l572_572915

-- Conditions specified in the problem
def man_rate : ℚ := 1 / 20
def father_rate : ℚ := 1 / 20
def son_rate : ℚ := 1 / 25

-- Definition of their combined rate
def combined_rate : ℚ := man_rate + father_rate + son_rate

-- Definition of the total time to complete the job together
def total_time : ℝ := (1 : ℝ) / (combined_rate : ℚ).toReal

-- Theorem statement
theorem time_to_complete_job_together : total_time ≈ 7.14 := sorry

end time_to_complete_job_together_l572_572915


namespace circles_intersect_l572_572408

noncomputable def C1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2*x + 8*y - 8 = 0
noncomputable def C2 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 4*x - 5 = 0

theorem circles_intersect : ∃ x y : ℝ, (C1 x y ∧ C2 x y) :=
by
  sorry

end circles_intersect_l572_572408


namespace p_plus_q_l572_572401

-- Define the problem conditions
def p (x : ℝ) : ℝ := 4 * (x - 2)
def q (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- Main theorem to prove the answer
theorem p_plus_q (x : ℝ) : p x + q x = x^2 + 4 * x - 12 := 
by
  sorry

end p_plus_q_l572_572401


namespace degree_sum_star_l572_572053

section StarAngles

variables (n : ℕ) (h : n ≥ 5)
variable S : ℝ

-- Definition of degree sum of the interior angles at the n points of the star
def degreeSumOfStar (n : ℕ) : ℝ := 180 * (n - 4)

-- Condition: The polygon sides are non-parallel and n >= 5, prolonged until they meet.
axiom star_properties : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ¬(sides_parallel k (k + 2)) 

-- Theorem: prove the degree-sum equals 180 * (n - 4)
theorem degree_sum_star : S = degreeSumOfStar n :=
  sorry

end StarAngles

end degree_sum_star_l572_572053


namespace rectangular_solid_surface_area_l572_572554

open Nat

theorem rectangular_solid_surface_area (a b c : ℕ) 
  (h_prime_a : Prime a)
  (h_prime_b : Prime b) 
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 231) :
  2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end rectangular_solid_surface_area_l572_572554


namespace revenue_after_fall_is_correct_l572_572490

variable (originalRevenue : ℝ) (percentageDecrease : ℝ)

theorem revenue_after_fall_is_correct :
    originalRevenue = 69 ∧ percentageDecrease = 39.130434782608695 →
    originalRevenue - (originalRevenue * (percentageDecrease / 100)) = 42 := by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end revenue_after_fall_is_correct_l572_572490


namespace fixed_chord_property_l572_572036

theorem fixed_chord_property (c : ℝ) (t : ℝ) (y : ℝ → ℝ) (A B C : (ℝ × ℝ)) :
  (∀ x, y x = 4 * x^2) →
  (C = (0, 2 * c)) →
  let AC := (C.1 - A.1)^2 + (C.2 - A.2)^2 in
  let BC := (C.1 - B.1)^2 + (C.2 - B.2)^2 in
  t = (1 / AC) + (1 / BC) →
  c = 1 →
  t = 0 :=
by
  sorry

end fixed_chord_property_l572_572036


namespace isosceles_triangle_has_largest_area_l572_572369

/-- Of all the triangles with a common base inscribed in a circle, the isosceles triangle has the largest area. -/
def largest_area_triangles (C : Type) [metric_space C] [normed_space ℝ C] (O : C) (r : ℝ) (A B C : C) 
  (h : dist O A = r ∧ dist O B = r ∧ dist O C = r) : Prop :=
  is_isosceles (triangle A B C)

theorem isosceles_triangle_has_largest_area (C : Type) [metric_space C] [normed_space ℝ C] (O : C) (r : ℝ) 
  (A B C : C) (h : dist O A = r ∧ dist O B = r ∧ dist O C = r) :
  largest_area_triangles C O r A B :=
sorry

end isosceles_triangle_has_largest_area_l572_572369


namespace min_total_translation_time_l572_572372

theorem min_total_translation_time (x1 x2 x3 x4 x5 : ℕ) :
  (x1 + x2 + x3 + x4 + x5 = 12) →
  (let S_i := λ i, [0, x1, x2, x3, x4, x5] in
   ∀ i, i ∈ [1, 2, 3, 4, 5] → S_i i = i * x_i) →
  (S := ∑ i in [1, 2, 3, 4, 5], i * x_i) →
  S / 5 = 24 :=
by sorry

end min_total_translation_time_l572_572372


namespace solution_set_inequality_k_l572_572721

theorem solution_set_inequality_k (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) → k = -4/5 :=
by
  sorry

end solution_set_inequality_k_l572_572721


namespace game_starting_player_wins_l572_572862

noncomputable def can_win (n : ℕ) : Prop :=
  ∃ move : ℕ, (0 < move ∧ move < n) ∧ ¬can_win (n + move)

theorem game_starting_player_wins :
  can_win 2 :=
sorry

end game_starting_player_wins_l572_572862


namespace find_int_less_than_neg3_l572_572966

theorem find_int_less_than_neg3 : 
  ∃ x ∈ ({-4, -2, 0, 3} : Set Int), x < -3 ∧ x = -4 := 
by
  -- formal proof goes here
  sorry

end find_int_less_than_neg3_l572_572966


namespace percentageSpentOnGummyBears_is_15_l572_572797

-- Define the problem variables
def hourly_wage : ℝ := 12.50
def tax_rate : ℝ := 0.20
def hours_worked : ℝ := 40
def amount_left : ℝ := 340

-- Gross pay calculation
def gross_pay (hourly_wage : ℝ) (hours_worked : ℝ) : ℝ := hourly_wage * hours_worked

-- Calculate the amount paid in taxes and fees
def tax_amount (gross_pay : ℝ) (tax_rate : ℝ) : ℝ := gross_pay * tax_rate

-- Net pay after taxes and fees
def net_pay (gross_pay : ℝ) (tax_amount : ℝ) : ℝ := gross_pay - tax_amount

-- Amount spent on gummy bears
def amount_spent_on_gummy_bears (net_pay : ℝ) (amount_left : ℝ) : ℝ := net_pay - amount_left

-- Percentage spent on gummy bears
def percentage_spent_on_gummy_bears (amount_spent : ℝ) (net_pay : ℝ) : ℝ := (amount_spent / net_pay) * 100

-- Prove that the percentage spent on gummy bears is 15%
theorem percentageSpentOnGummyBears_is_15 :
  let gross := gross_pay hourly_wage hours_worked,
      tax := tax_amount gross tax_rate,
      net := net_pay gross tax,
      spent := amount_spent_on_gummy_bears net amount_left
  in percentage_spent_on_gummy_bears spent net = 15 := by
  -- Proof would go here
  sorry

end percentageSpentOnGummyBears_is_15_l572_572797


namespace tan_identity_proof_l572_572093

noncomputable def tan_identity : ℝ := Real.tan (π / 9) * Real.tan (2 * (π / 9)) * Real.tan (4 * (π / 9))

theorem tan_identity_proof : tan_identity = 1 := 
by 
  -- Use the polynomial relation: 9t - 84t^3 + 126t^5 - 36t^7 + t^9 = 0
  -- derived from \tan(9θ)
  -- And factor it to get t(t^8 - 84t^6 + 126t^4 - 36t^2 + 1) = 0
  -- For theta = π/9, 2π/9, 4π/9
  -- Use Vieta’s formulas to find the product
  sorry

end tan_identity_proof_l572_572093


namespace max_f_value_l572_572343

noncomputable def f (A B C : ℝ) (Z : ℂ) : ℝ :=
  A * complex.norm_sq Z + 2 * B * Z.re + C

theorem max_f_value (A B C : ℝ) (hA : A < 0) :
  ∃ Z : ℂ, f A B C Z = (AC - B^2)/A ∧ Z = -B/A :=
by
  sorry

end max_f_value_l572_572343


namespace nickel_ate_2_chocolates_l572_572807

def nickels_chocolates (r n : Nat) : Prop :=
r = n + 7

theorem nickel_ate_2_chocolates (r : Nat) (h : r = 9) (h1 : nickels_chocolates r 2) : 2 = 2 :=
by
  sorry

end nickel_ate_2_chocolates_l572_572807


namespace find_other_outlet_rate_l572_572046

open Real

-- Definitions based on conditions
def V : ℝ := 20 * 1728   -- volume of the tank in cubic inches
def r1 : ℝ := 5          -- rate of inlet pipe in cubic inches/min
def r2 : ℝ := 8          -- rate of one outlet pipe in cubic inches/min
def t : ℝ := 2880        -- time in minutes required to empty the tank
 
-- Mathematically equivalent proof statement
theorem find_other_outlet_rate (x : ℝ) : 
  -- Given conditions
  V = 34560 →
  r1 = 5 →
  r2 = 8 →
  t = 2880 →
  -- Statement to prove
  V = (r2 + x - r1) * t → x = 9 :=
by
  intro hV hr1 hr2 ht hEq
  sorry

end find_other_outlet_rate_l572_572046


namespace min_ap_bp_is_9_l572_572327

open Real

def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def parabola (P : ℝ × ℝ) : Prop :=
  P.2^2 = 8 * P.1

def A : ℝ × ℝ := point 2 0
def B : ℝ × ℝ := point 7 6

noncomputable def min_ap_bp_value : ℝ :=
  Inf { d | ∃ P, parabola P ∧ d = distance A P + distance B P }

theorem min_ap_bp_is_9 : min_ap_bp_value = 9 := by
  sorry

end min_ap_bp_is_9_l572_572327


namespace total_pieces_of_candy_limit_l572_572645

theorem total_pieces_of_candy_limit (Maggie_candy : ℕ) (Harper_more : ℚ) (Neil_more : ℚ) (Liam_more : ℚ) (total_limit : ℕ) :
  Maggie_candy = 50 ∧ Harper_more = 0.30 ∧ Neil_more = 0.40 ∧ Liam_more = 0.20 ∧ total_limit = 300 →
  Maggie_candy + ((1 + Harper_more) * Maggie_candy).nat_floor + ((1 + Neil_more) * ((1 + Harper_more) * Maggie_candy)).nat_floor + ((1 + Liam_more) * ((1 + Neil_more) * ((1 + Harper_more) * Maggie_candy))).nat_floor = total_limit :=
by
  sorry

end total_pieces_of_candy_limit_l572_572645


namespace mass_of_hemisphere_is_correct_l572_572195

noncomputable def mass_of_hemisphere (R : ℝ) : ℝ :=
  (π^2 * R^4) / 2

theorem mass_of_hemisphere_is_correct (R : ℝ) : 
  let z := λ (x y : ℝ), sqrt (R^2 - x^2 - y^2) in
  let density := λ (x y z : ℝ), sqrt (x^2 + y^2) in
  let ds := λ (x y : ℝ), R / sqrt (R^2 - x^2 - y^2) in
  ∫∫ (λ (x y : ℝ), density x y (z x y) * ds x y) = (π^2 * R^4) / 2 :=
sorry

end mass_of_hemisphere_is_correct_l572_572195


namespace tan_product_pi_nine_l572_572126

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572126


namespace average_viewing_times_correct_l572_572794

-- Define the viewing times for each family member per week
def Evelyn_week1 : ℕ := 10
def Evelyn_week2 : ℕ := 8
def Evelyn_week3 : ℕ := 6

def Eric_week1 : ℕ := 8
def Eric_week2 : ℕ := 6
def Eric_week3 : ℕ := 5

def Kate_week2_episodes : ℕ := 12
def minutes_per_episode : ℕ := 40
def Kate_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def Kate_week3 : ℕ := 4

def John_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def John_week3 : ℕ := 8

-- Calculate the averages
def average (total : ℚ) (weeks : ℚ) : ℚ := total / weeks

-- Define the total viewing time for each family member
def Evelyn_total : ℕ := Evelyn_week1 + Evelyn_week2 + Evelyn_week3
def Eric_total : ℕ := Eric_week1 + Eric_week2 + Eric_week3
def Kate_total : ℕ := 0 + Kate_week2 + Kate_week3
def John_total : ℕ := 0 + John_week2 + John_week3

-- Define the expected averages
def Evelyn_expected_avg : ℚ := 8
def Eric_expected_avg : ℚ := 19 / 3
def Kate_expected_avg : ℚ := 4
def John_expected_avg : ℚ := 16 / 3

-- The theorem to prove that the calculated averages are correct
theorem average_viewing_times_correct :
  average Evelyn_total 3 = Evelyn_expected_avg ∧
  average Eric_total 3 = Eric_expected_avg ∧
  average Kate_total 3 = Kate_expected_avg ∧
  average John_total 3 = John_expected_avg :=
by sorry

end average_viewing_times_correct_l572_572794


namespace binomial_9_3_l572_572532

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l572_572532


namespace passengers_remaining_after_fourth_stop_l572_572715

theorem passengers_remaining_after_fourth_stop :
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  (initial_passengers * remaining_fraction * remaining_fraction * remaining_fraction * remaining_fraction = 1024 / 81) :=
by
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  have H1 : initial_passengers * remaining_fraction = 128 / 3 := sorry
  have H2 : (128 / 3) * remaining_fraction = 256 / 9 := sorry
  have H3 : (256 / 9) * remaining_fraction = 512 / 27 := sorry
  have H4 : (512 / 27) * remaining_fraction = 1024 / 81 := sorry
  exact H4

end passengers_remaining_after_fourth_stop_l572_572715


namespace right_triangle_perimeter_l572_572935

noncomputable theory

def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter
  (a : ℝ) (area : ℝ) (b : ℝ) :
  area = 150 ∧ a = 30 →
  ∃ x c : ℝ, (1 / 2) * a * x = area ∧
             c^2 = a^2 + x^2 ∧
             perimeter_of_triangle a x c = 40 + 10 * real.sqrt 10 :=
begin
  sorry
end

end right_triangle_perimeter_l572_572935


namespace arrangement_count_5_l572_572846

open Finset

theorem arrangement_count_5 (A B : Fin 5) :
  (card ((finPerm 5).filter (λ σ, σ 0 ≠ A ∧ σ 4 ≠ B))) = 72 := 
sorry

end arrangement_count_5_l572_572846


namespace desired_average_sale_is_5600_l572_572908

-- Define the sales for five consecutive months
def sale1 : ℕ := 5266
def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029

-- Define the required sale for the sixth month
def sale6 : ℕ := 4937

-- Calculate total sales for the first five months
def total_five_months := sale1 + sale2 + sale3 + sale4 + sale5

-- Calculate total sales for six months
def total_six_months := total_five_months + sale6

-- Calculate the desired average sale for six months
def desired_average := total_six_months / 6

-- The theorem statement: desired average sale for the six months
theorem desired_average_sale_is_5600 : desired_average = 5600 :=
by
  sorry

end desired_average_sale_is_5600_l572_572908


namespace tan_product_pi_nine_l572_572129

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end tan_product_pi_nine_l572_572129


namespace tan_product_l572_572139

noncomputable def tan_pi_nine := Real.tan (Real.pi / 9)
noncomputable def tan_two_pi_nine := Real.tan (2 * Real.pi / 9)
noncomputable def tan_four_pi_nine := Real.tan (4 * Real.pi / 9)

-- Key polynomial condition
def poly := λ t : ℝ, t^8 - 36 * t^6 + 126 * t^4 - 84 * t^2 + 9

-- Condition: the polynomial has roots tan(π/9), tan(2π/9), tan(4π/9)
axiom poly_roots: 
  (poly tan_pi_nine = 0 ∧
   poly tan_two_pi_nine = 0 ∧
   poly tan_four_pi_nine = 0)

-- The proof statement
theorem tan_product : 
  tan_pi_nine * 
  tan_two_pi_nine * 
  tan_four_pi_nine = 3 :=
sorry

end tan_product_l572_572139


namespace sum_of_segments_l572_572481

theorem sum_of_segments (k : ℕ) (R : ℝ) (O : Point) (polygon : regular_polygon (4 * k + 2) R O) :
  let segments_sum :=
    (λ idx : fin k, segment_length (O$1$ * (polygon.vertices idx)) (polygon.vertices (2 * k - idx))) in
  list.sum segments_sum = R := 
sorry

end sum_of_segments_l572_572481


namespace pascal_triangle_pth_row_l572_572802

theorem pascal_triangle_pth_row {p : ℕ} (hp : Nat.Prime p) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ p - 1 → p ∣ Nat.choose p k :=
by
  intros k hk
  have hpk := Nat.Prime.dvd_choose_self hp k hk.left (hk.right.trans (Nat.sub_le p 1))
  exact hpk

end pascal_triangle_pth_row_l572_572802


namespace sum_of_numbers_l572_572876

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 :=
by
  sorry

end sum_of_numbers_l572_572876


namespace pradeep_pass_percentage_l572_572799

variable (marks_obtained : ℕ) (marks_short : ℕ) (max_marks : ℝ)

theorem pradeep_pass_percentage (h1 : marks_obtained = 150) (h2 : marks_short = 25) (h3 : max_marks = 500.00000000000006) :
  ((marks_obtained + marks_short) / max_marks) * 100 = 35 := 
by
  sorry

end pradeep_pass_percentage_l572_572799


namespace tan_product_equals_three_l572_572167

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l572_572167


namespace main_l572_572479

def fold_mapped_point_proof (m n : ℝ) : Prop :=
  let midpoint (x1 y1 x2 y2 : ℝ) := (x1 + x2) / 2, (y1 + y2) / 2
  let slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)
  let fold_line_through_midpoint := (x y : ℝ) → y = 2 * x - 3
  let point_after_fold (x y : ℝ) (x' y' : ℝ) := slope x y x' y' = -1/2
  let point_midpoint_on_fold_line (x y x' y' : ℝ) :=
    let mpx, mpy := midpoint x y x' y'
    fold_line_through_midpoint mpx mpy

  (point_after_fold 1 4 5 2 ∧ 
  point_midpoint_on_fold_line 1 4 5 2 ∧
  point_after_fold 9 3 m n ∧
  point_midpoint_on_fold_line 9 3 m n) →
  m + n = 7 

theorem main : ∃ (m n : ℝ), fold_mapped_point_proof m n :=
begin
  unfold fold_mapped_point_proof,
  have h1 : (2:ℝ) * (3:ℝ) - 3 = 3, by norm_num,
  have h2 : (15:ℝ) = (2:ℝ) * (3:ℝ) + (3:ℝ), by norm_num,
  exact ⟨-1, 8, by repeat {exact h1;}⟩,
end

end main_l572_572479


namespace determine_b_coplanar_l572_572189

def coplanar_points (b : ℝ) : Prop :=
  let v1 := ![2, b, 0]
      v2 := ![0, 2, b]
      v3 := ![b, 0, 2]
  in (Matrix.det (Matrix.of ![v1, v2, v3]) = 0)

theorem determine_b_coplanar :
  ∀ (b : ℝ), coplanar_points b ↔ b = 2 ∨ b = -2 := 
by
  sorry

end determine_b_coplanar_l572_572189


namespace divisors_of_180_not_divisible_by_3_l572_572688

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572688


namespace find_a_l572_572621

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, quadratic_function a x = x^2 + 2 * (a - 1) * x + 2) ∧ 
  (∀ x1 x2 : ℝ, x1 ≤ 4 → x2 ≤ 4 → x1 < x2 → quadratic_function a x1 > quadratic_function a x2) ∧ 
  (∀ x1 x2 : ℝ, x1 ≥ 4 → x2 ≥ 4 → x1 < x2 → quadratic_function a x1 < quadratic_function a x2)
  → a = -3 :=
begin
  sorry
end

end find_a_l572_572621


namespace find_line_equation_l572_572575

-- Definitions: Point and Line in 2D
structure Point2D where
  x : ℝ
  y : ℝ

structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Line passes through the point
def line_through_point (L : Line2D) (P : Point2D) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Perpendicular lines condition: if Line L1 and Line L2 are perpendicular.
def perpendicular (L1 L2 : Line2D) : Prop :=
  L1.a * L2.a + L1.b * L2.b = 0

-- Define line1 and line2 as given
def line1 : Line2D := {a := 1, b := -2, c := 0} -- corresponds to x - 2y + m = 0

-- Define point P (-1, 3)
def P : Point2D := {x := -1, y := 3}

-- Required line passing through point P and perpendicular to line1
def required_line : Line2D := {a := 2, b := 1, c := -1}

-- The proof goal
theorem find_line_equation : (line_through_point required_line P) ∧ (perpendicular line1 required_line) :=
by
  sorry

end find_line_equation_l572_572575


namespace time_per_lap_is_correct_l572_572308

-- Given conditions
def total_time_in_minutes (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def total_laps : ℕ := 5
def total_time : ℕ := total_time_in_minutes 1 36

-- Theorem statement
theorem time_per_lap_is_correct : total_time / total_laps = 19.2 :=
by
  sorry -- Proof to be filled in

end time_per_lap_is_correct_l572_572308


namespace mike_practice_hours_l572_572361

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l572_572361


namespace find_lambda_l572_572214

variables (a b : EuclideanVector 3) (λ : ℝ)

-- Given conditions
def mag_a := ∥a∥ = 2
def mag_b := ∥b∥ = Real.sqrt 2
def angle := ∠a b = Real.pi / 4
def perp := (λ • b - a) ⬝ a = 0

theorem find_lambda (h1 : mag_a) (h2 : mag_b) (h3 : angle) (h4 : perp) :
  λ = 2 := 
sorry

end find_lambda_l572_572214


namespace tan_product_identity_l572_572103

noncomputable def tan_pi_over_n (n : ℕ) : ℝ := Real.tan (Real.pi / n)

theorem tan_product_identity :
  tan_pi_over_n 9 * tan_pi_over_n 4.5 * tan_pi_over_n 2.25 = Real.sqrt 3 := 
sorry

end tan_product_identity_l572_572103


namespace calculate_total_area_of_figure_l572_572738

-- Defining the lengths of the segments according to the problem conditions.
def length_1 : ℕ := 8
def length_2 : ℕ := 6
def length_3 : ℕ := 3
def length_4 : ℕ := 5
def length_5 : ℕ := 2
def length_6 : ℕ := 4

-- Using the given lengths to compute the areas of the smaller rectangles
def area_A : ℕ := length_1 * length_2
def area_B : ℕ := length_4 * (10 - 6)
def area_C : ℕ := (6 - 3) * (15 - 10)

-- The total area of the figure is the sum of the areas of the smaller rectangles
def total_area : ℕ := area_A + area_B + area_C

-- The statement to prove
theorem calculate_total_area_of_figure : total_area = 83 := by
  -- Proof goes here
  sorry

end calculate_total_area_of_figure_l572_572738


namespace train_length_l572_572961

theorem train_length (speed_kmph : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_crossed := speed_mps * time_s
  let train_length := distance_crossed - bridge_length_m
  train_length = 250 :=
by
  sorry

end train_length_l572_572961


namespace divisors_not_divisible_by_3_l572_572680

/-!
# Number of divisors of 180 not divisible by 3

In this statement, we prove that the number of positive divisors of 180 that are not divisible by 3 is equal to 6.
-/

def prime_factorization_180 : ℕ → Prop
| 180 := (∀ {p : ℕ}, prime p → p ∣ 180 → p = 2 ∨ p = 3 ∨ p = 5)

theorem divisors_not_divisible_by_3 : 
  { d : ℕ // d ∣ 180 ∧ ( ∀ {p : ℕ}, prime p → p ∣ d → p ≠ 3) } = (6 : ℕ) :=
by {
  sorry
}

end divisors_not_divisible_by_3_l572_572680


namespace equation_of_circle_l572_572040

-- Define constants used in the problem
def line_equation (x y: ℝ) := x + 2 * y + 2 + Real.sqrt 5 
def center_A := (2 : ℝ, 2 : ℝ)
def reflected_line_equation (x y : ℝ) := x - 2 * y + 2 + Real.sqrt 5

-- Define the distance function from point to line
def distance_from_point_to_line (x0 y0 : ℝ) (a b c : ℝ) :=
  abs (a*x0 + b*y0 + c) / (Real.sqrt (a^2 + b^2))

-- Proof that the equation of the circle is correct
theorem equation_of_circle :
  let (x0, y0) := center_A in
  line_equation x0 y0 = 0 ∧
  distance_from_point_to_line 2 2 1 (-2) (2 + Real.sqrt 5) = 1 →
  ∀ x y, (x - 2)^2 + (y - 2)^2 = 1 :=
by
  intros x0 y0 h 
  sorry

end equation_of_circle_l572_572040


namespace quadrilateral_centroid_theorem_l572_572331

open EuclideanGeometry

variable {A B C D P : Point}

-- Definition of the centroid G of quadrilateral ABCD
def centroid (A B C D : Point) : Point := 
  (A.toVec + B.toVec + C.toVec + D.toVec) / 4

-- The statement to be proved
theorem quadrilateral_centroid_theorem (A B C D P : Point) :
  let G := centroid A B C D in
  dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 + dist P D ^ 2 =
  4 * dist P G ^ 2 + dist G A ^ 2 + dist G B ^ 2 + dist G C ^ 2 + dist G D ^ 2 := 
by
  sorry

end quadrilateral_centroid_theorem_l572_572331


namespace max_a_for_common_tangent_line_l572_572273

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (a x : ℝ) : ℝ := 2 * a * Real.log x + 1

theorem max_a_for_common_tangent_line (h : ∃ a : ℝ, ∃ x y : ℝ, deriv (f x) = deriv (g a y) ∧ f x = g a y) :
  upper_bound (λ a, ∃ x y : ℝ, deriv (f x) = deriv (g a y) ∧ f x = g a y ∧ a ≤ e) := sorry

end max_a_for_common_tangent_line_l572_572273


namespace f_2002_is_1001_l572_572637

open Real

noncomputable def f (a : ℝ) : ℝ := sorry

lemma functional_eq (x y : ℝ) : f(x + y)^2 = f(x) + 2 * (f(y)^2) := sorry

lemma f_one_neq_zero : f(1) ≠ 0 := sorry

theorem f_2002_is_1001 : f(2002) = 1001 :=
begin
  -- the proof will be filled here later by one solving the problem
  sorry
end

end f_2002_is_1001_l572_572637


namespace yogurt_combinations_l572_572907

theorem yogurt_combinations 
  (flavors : Finset String) 
  (toppings : Finset String) 
  (h_flavors : flavors.card = 5) 
  (h_toppings : toppings.card = 8) : 
  let no_toppings := 1
      one_topping := (toppings.card).choose 1
      two_toppings := (toppings.card).choose 2 in
  flavors.card * (no_toppings + one_topping + two_toppings) = 185 := 
  by
    have one_topping := h_toppings.choose 1 
    have two_toppings := h_toppings.choose 2 
    calc
      flavors.card * (no_toppings + one_topping + two_toppings) 
        = 5 * (1 + one_topping + two_toppings) : by rw [h_flavors]
    ... 
        = 5 * (1 + 8 + 28) : by rw [one_topping, two_toppings]
    ... 
        = 185 : by norm_num

end yogurt_combinations_l572_572907


namespace exists_lambda_mu_limit_fib_ratio_fib_sum_squares_fib_even_index_fib_divisibility_l572_572832

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def phi_prime : ℝ := (1 - Real.sqrt 5) / 2
noncomputable def Fib : ℕ → ℝ
| 0       := 0
| 1       := 1
| (n + 2) := Fib (n + 1) + Fib n

-- 1. Proof for the existence of λ and μ such that Fib n = λ phi^n + μ phi_prime^n
theorem exists_lambda_mu (n : ℕ) :
  ∃ (λ μ : ℝ), 
    (λ + μ = 0) ∧ 
    (λ * phi + μ * phi_prime = 1) ∧ 
    (Fib n = λ * phi ^ n + μ * phi_prime ^ n) := sorry

-- 2. Proof that the limit of the ratio of the Fibonacci sequence converges to phi
theorem limit_fib_ratio : 
  tendsto (λ n, Fib (n + 1) / Fib n) at_top (nhds phi) := sorry

-- 3. Proof of the sum of squares formula
theorem fib_sum_squares (n : ℕ) :
  (∑ i in Finset.range (n + 1), (Fib i) ^ 2) = Fib n * Fib (n + 1) := sorry

-- 4. Proof of the even-index Fibonacci identity
theorem fib_even_index (n : ℕ) :
  Fib (2 * n - 1) = (Fib n) ^ 2 + (Fib (n + 1)) ^ 2 := sorry

-- 5. Proof of division property
theorem fib_divisibility (m n : ℕ) (h : m ∣ n) : 
  Fib m ∣ Fib n := sorry


end exists_lambda_mu_limit_fib_ratio_fib_sum_squares_fib_even_index_fib_divisibility_l572_572832


namespace common_elements_1700_multiples_of_5_and_9_l572_572758

/-- 
  Let S be the set of the 1700 smallest positive multiples of 5, 
  and let T be the set of the 1700 smallest positive multiples of 9.
  Prove that the number of elements common to both S and T is 188.
-/
theorem common_elements_1700_multiples_of_5_and_9 :
  let S := {n : ℕ | ∃ k, k ≤ 1700 ∧ n = 5 * k},
      T := {n : ℕ | ∃ k, k ≤ 1700 ∧ n = 9 * k} in
  S ∩ T = {n : ℕ | ∃ k, k ≤ 188 ∧ n = 45 * k} :=
by
  let S := {n : ℕ | ∃ k, k ≤ 1700 ∧ n = 5 * k}
  let T := {n : ℕ | ∃ k, k ≤ 1700 ∧ n = 9 * k}
  have : S ∩ T = {n : ℕ | ∃ k, k ≤ 188 ∧ n = 45 * k}, from sorry
  exact this

end common_elements_1700_multiples_of_5_and_9_l572_572758


namespace ellipse_foci_area_tangent_l572_572009

theorem ellipse_foci_area_tangent (
  a b : ℝ,
  h1 : a > b,
  h2 : b > 0,
  P : ℝ × ℝ,
  F1 F2 : ℝ × ℝ,
  h3 : (P.1^2 / a^2 + P.2^2 / b^2 = 1),
  area_triangle_P_F1_F2 : ℝ,
  h4 : area_triangle_P_F1_F2 = 1,
  tan_P_F1_F2 : ℝ,
  h5 : tan_P_F1_F2 = 1/2,
  tan_P_F2_F1 : ℝ,
  h6 : tan_P_F2_F1 = -2
) : a = Real.sqrt 15 / 2 := sorry

end ellipse_foci_area_tangent_l572_572009


namespace sample_variance_is_two_l572_572732

-- Define the sample mean
def sample_mean (a : ℝ) : ℝ := (a + 0 + 1 + 2 + 3) / 5

-- Define the sample variance
def sample_variance (a : ℝ) (mean : ℝ) : ℝ :=
  let sq_diff x := (x - mean) ^ 2 in
  (sq_diff a + sq_diff 0 + sq_diff 1 + sq_diff 2 + sq_diff 3) / 5

-- The main theorem
theorem sample_variance_is_two (a : ℝ) (mean : ℝ) (h_mean : sample_mean a = mean) (h_mean_eq : mean = 1) : 
  sample_variance a mean = 2 :=
by
  sorry

end sample_variance_is_two_l572_572732


namespace prism_max_volume_l572_572285

noncomputable def maximum_volume_prism (s h : ℝ) : ℝ :=
  let area_base := (sqrt 3 / 4) * s^2
  let area_lateral := s * h
  let volume := area_base * h
  if (area_base + 2 * area_lateral = 27) then volume else 0

theorem prism_max_volume
 (s h : ℝ)
 (hsqrt_nonneg : s >= 0)
 (h_nonneg : h >= 0)
 (sum_area_eq_27 : (sqrt 3 / 4) * s^2 + 2 * (s * h) = 27)
 : maximum_volume_prism s s = 15 := 
sorry

end prism_max_volume_l572_572285


namespace common_difference_arithmetic_sequence_l572_572746

theorem common_difference_arithmetic_sequence (a b : ℝ) :
  ∃ d : ℝ, b = a + 6 * d ∧ d = (b - a) / 6 :=
by
  sorry

end common_difference_arithmetic_sequence_l572_572746


namespace optimalBananaBuys_l572_572455

noncomputable def bananaPrices : List ℕ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

def days := List.range 18

def computeOptimalBuys : List ℕ :=
  sorry -- Implement the logic to compute the optimal number of bananas to buy each day.

theorem optimalBananaBuys :
  computeOptimalBuys = [4, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 1] :=
sorry

end optimalBananaBuys_l572_572455


namespace wave_number_probability_l572_572471

-- Define the wave number concept
def is_wave_number (l : List ℕ) : Prop :=
  l.nth 0 < l.nth 1 ∧ l.nth 1 > l.nth 2 ∧ l.nth 2 < l.nth 3 ∧ l.nth 3 > l.nth 4

-- Define the set of digits
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Define the property we want to prove
theorem wave_number_probability :
  (digits.permutations.count is_wave_number : ℚ) / digits.permutations.length = 2 / 15 :=
by
  sorry

end wave_number_probability_l572_572471


namespace correct_statement_l572_572886

theorem correct_statement : (-2 : ℤ) ∈ ℤ := by
  sorry

end correct_statement_l572_572886


namespace relationship_among_abc_l572_572224

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end relationship_among_abc_l572_572224


namespace divisors_of_180_not_divisible_by_3_l572_572686

def is_divisor (n d : ℕ) := d ∣ n

def prime_factors (n : ℕ) :=
  if n = 180 then [2, 3, 5] else []

def divisor_not_divisible_by_3 (n d : ℕ) :=
  is_divisor n d ∧ ¬ (3 ∣ d)

def number_of_divisors_not_divisible_by_3 (n : ℕ) :=
  if n = 180 then 6 else 0

theorem divisors_of_180_not_divisible_by_3 : 
  ∀ n, n = 180 → (∑ d in (List.filter (divisor_not_divisible_by_3 n) (List.range (n+1))), 1) = 6 := by
  sorry

end divisors_of_180_not_divisible_by_3_l572_572686


namespace tan_product_eq_three_l572_572079

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572079


namespace sum_prime_reciprocal_product_lt_one_l572_572713

theorem sum_prime_reciprocal_product_lt_one (N : ℕ) (hN : N ≥ 2) :
  ∑ n in finset.range (N - 1) + 2, 
    (∑ p in (nat.factors (n + 2)).to_finset, 1 / p)  * 
    (∑ p in (nat.factors (n + 1)).to_finset, 1 / p) * 
    ... * 
    (∑ p in (nat.factors 2).to_finset, 1 / p) < 1 := 
sorry

end sum_prime_reciprocal_product_lt_one_l572_572713


namespace right_triangle_perimeter_l572_572940

/-- A right triangle has an area of 150 square units,
and one leg with a length of 30 units. Prove that the
perimeter of the triangle is 40 + 10 * sqrt 10 units. -/
theorem right_triangle_perimeter :
  ∃ (x c : ℝ), (1 / 2) * 30 * x = 150 ∧ c^2 = 30^2 + x^2 ∧ 30 + x + c = 40 + 10 * sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l572_572940


namespace tan_product_l572_572086

theorem tan_product : 
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = 3 :=
by
  sorry

end tan_product_l572_572086


namespace measure_angle_CAB_l572_572833

noncomputable def geometrical_theorem : Prop :=
∀ (A B C K C1 B1 B2 C2 : Type*)
(midpoint_AB : C1 = midpoint A B)
(midpoint_AC : B1 = midpoint A C)
(line_C1K_AC : C1K_meets_AC_at C1 K A C B2)
(line_B1K_AB : B1K_meets_AB_at B1 K A B C2)
(equal_areas : area_triangle AB2C2 = area_triangle ABC),
∠ A B C = 60

axiom midpoint : Type* → Type* → Type*

axiom area_triangle : Type* → Type*

axiom ∠ : Type* → Type* → Type* → Type*

axiom C1K_meets_AC_at : Type* → Type* → Type* → Type* → (Type* → Type*)

axiom B1K_meets_AB_at : Type* → Type* → Type* → Type* → (Type* → Type*)

theorem measure_angle_CAB :
  geometrical_theorem :=
sorry

end measure_angle_CAB_l572_572833


namespace fg_value_correct_l572_572394

def f_table (x : ℕ) : ℕ :=
  if x = 1 then 3
  else if x = 3 then 7
  else if x = 5 then 9
  else if x = 7 then 13
  else if x = 9 then 17
  else 0  -- Default value to handle unexpected inputs

def g_table (x : ℕ) : ℕ :=
  if x = 1 then 54
  else if x = 3 then 9
  else if x = 5 then 25
  else if x = 7 then 19
  else if x = 9 then 44
  else 0  -- Default value to handle unexpected inputs

theorem fg_value_correct : f_table (g_table 3) = 17 := 
by sorry

end fg_value_correct_l572_572394


namespace parabola_equilateral_triangle_point_on_xaxis_l572_572536

theorem parabola_equilateral_triangle (p : ℝ) (A B F : ℝ×ℝ) (side : ℝ) (P0 : p > 0) 
(h1 : A.2^2 = 2*p*A.1) (h2 : B.1 = 0) (h3 : dist A F = side) (h4 : dist B F = side) (h5 : dist A B = side) :
  p = 2 :=
sorry

theorem point_on_xaxis (N : ℝ×ℝ) (t : ℝ) (m : ℝ) (Q R : ℝ×ℝ) (p : ℝ) 
(h1 : p = 2) (h2 : t = 2) (h3 : Q.2^2 = 4*Q.1) (h4 : R.2^2 = 4*R.1) 
(h5 : Q.1 = m*Q.2 + t) (h6 : R.1 = m*R.2 + t) :
  ∃ N : ℝ×ℝ, N = (2, 0) ∧ ∀ Q, ∀ R, 
      (∃ l', line_passing_through l' N ∧ intersects_parabola l' Q R) → 
      (1/((Q.1 - N.1)^2 + Q.2^2) + 1/((R.1 - N.1)^2 + R.2^2) = 1/4) :=
sorry

def line_passing_through (l' : ℝ→ℝ→Prop) (N : ℝ×ℝ) : Prop :=
l' N.1 N.2

def intersects_parabola (l' : ℝ→ℝ→Prop) (Q R : ℝ×ℝ) : Prop :=
l' Q.1 Q.2 ∧ l' R.1 R.2

end parabola_equilateral_triangle_point_on_xaxis_l572_572536


namespace parallel_vectors_implies_product_l572_572210

noncomputable def vec_a (λ : ℝ) : (ℝ × ℝ × ℝ) := (λ + 1, 0, 2 * λ)
noncomputable def vec_b (μ : ℝ) : (ℝ × ℝ × ℝ) := (6, 2 * μ - 1, 2)

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2, k * b.3)

theorem parallel_vectors_implies_product (λ μ : ℝ) (h : are_parallel (vec_a λ) (vec_b μ)) :
  λ * μ = 1 / 10 :=
sorry

end parallel_vectors_implies_product_l572_572210


namespace courier_speeds_correctness_l572_572038

noncomputable def courier_speeds : Prop :=
  ∃ (s1 s2 : ℕ), (s1 * 8 + s2 * 8 = 176) ∧ (s1 = 60 / 5) ∧ (s2 = 60 / 6)

theorem courier_speeds_correctness : courier_speeds :=
by
  sorry

end courier_speeds_correctness_l572_572038


namespace simple_interest_amount_l572_572275

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

theorem simple_interest_amount:
  ∀ (P : ℝ), compound_interest P 5 2 = 51.25 → simple_interest P 5 2 = 50 :=
by
  intros P h
  -- this is where the proof would go
  sorry

end simple_interest_amount_l572_572275


namespace largest_possible_a_plus_b_leq_l572_572193

open Nat

theorem largest_possible_a_plus_b_leq (a b n : ℕ) (h1 : a + b ≤ 2007)
  (h2 : Nat.gcd a b = 1)
  (h3 : (∏ k in range(2,n+1), ((k^3 - 1) / (k^3 + 1))) = (a / b)) : a + b = 1891 :=
sorry

end largest_possible_a_plus_b_leq_l572_572193


namespace sum_of_reciprocal_b_l572_572642

-- Definitions based on the problem conditions
def a (n : ℕ) : ℕ := 2 ^ (b n)
def b (n : ℕ) : ℕ := (n * (n + 1)) / 2
def seq_sum (n : ℕ) : ℕ := (Finset.range n).sum (λ i, 1 / b (i + 1))

-- Main problem statement
theorem sum_of_reciprocal_b {n : ℕ} (hn : n > 0) : seq_sum n = 2 * n / (n + 1) := 
by
  sorry

end sum_of_reciprocal_b_l572_572642


namespace building_height_l572_572437

theorem building_height (h : ℕ) 
  (shadow_building : ℕ) 
  (shadow_pole : ℕ) 
  (height_pole : ℕ) 
  (ratio_proportional : shadow_building * height_pole = shadow_pole * h) 
  (shadow_building_val : shadow_building = 63) 
  (shadow_pole_val : shadow_pole = 32) 
  (height_pole_val : height_pole = 28) : 
  h = 55 := 
by 
  sorry

end building_height_l572_572437


namespace min_value_of_m_l572_572227

theorem min_value_of_m :
  ∃ m, (∀ (x y : ℝ), y = x → m = sqrt ((x - 1) ^ 2 + (y - 2) ^ 2) + sqrt ((x + 2) ^ 2 + (y - 1) ^ 2)) ∧ m ≥ 0 ∧ m = 4 :=
sorry

end min_value_of_m_l572_572227


namespace number_of_ways_pairs_l572_572207

theorem number_of_ways_pairs (n : ℕ) (hc1 : n = 4) :
  let total_ways := (nat.choose 8 4),
      non_paired_ways := 16,
      paired_ways := total_ways - non_paired_ways in
  paired_ways = 54 :=
by
  sorry

end number_of_ways_pairs_l572_572207


namespace gain_amount_l572_572976

theorem gain_amount (S : ℝ) (g : ℝ) (h_S : S = 180) (h_g : g = 0.20) : ∃ G : ℝ, G = 30 :=
by
  let C := S / (1 + g)
  have h_C : C = 150 := by
    rw [h_S, h_g]
    linarith
  let G := g * C
  have h_G : G = 30 := by
    rw [h_C, h_g]
    linarith
  use G
  exact h_G

end gain_amount_l572_572976


namespace ratio_BE_ED_l572_572912

variables {A B C D E F G : Type} [parallelogram A B C D]
variables {BD CD : Type} (a b: ℝ) (x y: ℝ)
variables (BE: ℝ) (ED: ℝ)

-- Given conditions
hypothesis : (line_through_vertex A intersect diagonal BD at E) ∧
             (line_through_vertex A intersect side CD at F) ∧
             (line_through_vertex A intersect line BC at G) ∧
             (ratio FG FE = 4) 

theorem ratio_BE_ED :
  x = a + b ∧ y = a + 5*b →
  BE / ED = sqrt 5 :=
sorry

end ratio_BE_ED_l572_572912


namespace distance_point_to_line_l572_572830

theorem distance_point_to_line : 
  let P := (1, 2)
  let L := λ (x y : ℝ), y = 2 * x + 1
  let d := (λ (x1 y1 a b c : ℝ), |a * x1 + b * y1 + c| / sqrt (a * a + b * b))
  d 1 2 2 (-1) 1 = sqrt 5 / 5 :=
by
  sorry

end distance_point_to_line_l572_572830


namespace cyclic_identity_l572_572382

theorem cyclic_identity (a b c : ℝ) :
  a * (a - c)^2 + b * (b - c)^2 - (a - c) * (b - c) * (a + b - c) =
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) ∧
  b * (b - a)^2 + c * (c - a)^2 - (b - a) * (c - a) * (b + c - a) =
  c * (c - b)^2 + a * (a - b)^2 - (c - b) * (a - b) * (c + a - b) := by
sorry

end cyclic_identity_l572_572382


namespace circumscribed_quadrilateral_angles_l572_572468

theorem circumscribed_quadrilateral_angles
  (α β : ℝ) 
  (hα : α = 50) 
  (hβ : β = 20) : 
  ∠EFG + ∠EHG = 110 :=
by
  sorry

end circumscribed_quadrilateral_angles_l572_572468


namespace displacement_formula_l572_572975

noncomputable def velocity (t : ℝ) : ℝ := 5 * t^2

theorem displacement_formula (t : ℝ) (ht : 0 ≤ t) :
  ∫ (τ : ℝ) in 0..t, velocity τ = (5 * t^3) / 3 :=
by
  sorry

end displacement_formula_l572_572975


namespace count_divisors_not_divisible_by_3_l572_572696

theorem count_divisors_not_divisible_by_3 :
  let n := 180 
  let a_max := 2 
  let b_max := 2 
  let c_max := 1 
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max 
  (∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0) = 6 :=
begin
  let n := 180,
  let a_max := 2,
  let b_max := 2,
  let c_max := 1,
  let condition := λ (a b c : ℕ), a ≤ a_max ∧ b = 0 ∧ c ≤ c_max,
  let valid_combinations := ∑ a in finset.range (a_max + 1), ∑ c in finset.range (c_max + 1), if condition a 0 c then 1 else 0,
  split,
  sorry
end

end count_divisors_not_divisible_by_3_l572_572696


namespace tan_product_eq_three_l572_572075

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572075


namespace find_x_l572_572262

-- Definitions for the vectors and their relationships
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := (a.1 + 2 * (b x).1, a.2 + 2 * (b x).2)
def v (x : ℝ) : ℝ × ℝ := (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

-- Given condition that u is parallel to v
def u_parallel_v (x : ℝ) : Prop := u x = v x

-- Prove that the value of x is 1/2
theorem find_x : ∃ x : ℝ, u_parallel_v x ∧ x = 1 / 2 := 
sorry

end find_x_l572_572262


namespace union_is_real_l572_572643

def M (x : ℝ) : Prop := x^2 > 4

def N (x : ℝ) : Prop := x < 3

theorem union_is_real :
  {x : ℝ | M x} ∪ {x : ℝ | N x} = set.univ := sorry

end union_is_real_l572_572643


namespace relationship_among_a_b_c_l572_572620

noncomputable def a : ℝ := 2 ^ 0.3
noncomputable def b : ℝ := 3 ^ 2
noncomputable def c : ℝ := 2 ^ (-0.3)

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  have h1 : c = 2 ^ (-0.3) := rfl
  have h2 : a = 2 ^ 0.3 := rfl
  have h3 : b = 3 ^ 2 := rfl
  have h4 : 2^(-0.3) < 1 := sorry -- requires calculation and proving
  have h5 : 1 < 2^(0.3) := sorry  -- requires calculation and proving
  have h6 : 2 < 9 := by norm_num -- straightforward numerical comparison
  exact ⟨h4.trans h5, h5.trans h6⟩

end relationship_among_a_b_c_l572_572620


namespace measure_angle_ECD_l572_572724

-- Given conditions
variables (A B C D E : Type) [geometry : euclidean_space (A B C)]
open euclidean_space

-- Definitions according to the given problem
def isosceles (A B C : euclidean_space) : Prop := distance A B = distance B C
def right_angle (D C B : euclidean_space) : Prop := angle D C B = 60
def perpendicular (C D A B : euclidean_space) : Prop := line (C, D) ⊥ line (A, B)

-- Theorem statement
theorem measure_angle_ECD (A B C D E : euclidean_space) 
  (h1 : isosceles A B C)
  (h2 : right_angle D C B)
  (h3 : perpendicular C D A B) : angle E C D = 30 :=
by 
  sorry

end measure_angle_ECD_l572_572724


namespace count_of_odd_divisors_less_than_hundred_l572_572703

-- Definitions based on the conditions in the problem
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def less_than_hundred (n : ℕ) : Prop := n < 100

-- The problem statement translated to Lean 4:
theorem count_of_odd_divisors_less_than_hundred :
  {n : ℕ | less_than_hundred n ∧ is_perfect_square n}.card = 9 :=
sorry

end count_of_odd_divisors_less_than_hundred_l572_572703


namespace triangle_sin_double_angle_l572_572277

open Real

theorem triangle_sin_double_angle (A : ℝ) (h : cos (π / 4 + A) = 5 / 13) : sin (2 * A) = 119 / 169 :=
by
  sorry

end triangle_sin_double_angle_l572_572277


namespace intersection_range_of_m_max_value_OA_plus_OB_l572_572299

-- Definition of the polar coordinates equations as given conditions.
def line_l (ρ θ : ℝ) : Prop := ρ = 2 / (sin θ + cos θ)
def curve_C (ρ θ m : ℝ) : Prop := ρ = 2 * m * cos θ ∧ m > 0

-- Proof problem 1: Intersection implies the given range for m.
theorem intersection_range_of_m (m : ℝ) (h1 : ∃ θ, curve_C (2 * m * cos θ) θ m ∧ line_l (2 / (sin θ + cos θ)) θ) : m ∈ Icc (2 * sqrt 2 - 2) (∝) := 
sorry

-- Proof problem 2: Maximum value of |OA| + |OB| given specific conditions.
theorem max_value_OA_plus_OB (m : ℝ) (θ : ℝ) (h2 : curve_C (2 * m * cos θ) θ m ∧ curve_C (2 * m * cos (θ + π / 4)) (θ + π / 4) m ∧ m > 0) : 
|2 * m * cos θ + 2 * m * cos (θ + π / 4)| = 2 * sqrt (2 + sqrt 2) * m := 
sorry

end intersection_range_of_m_max_value_OA_plus_OB_l572_572299


namespace solve_inequality_l572_572393

-- Defining the inequality
def inequality (x : ℝ) : Prop := 1 / (x - 1) ≤ 1

-- Stating the theorem
theorem solve_inequality :
  { x : ℝ | inequality x } = { x : ℝ | x < 1 } ∪ { x : ℝ | 2 ≤ x } :=
by
  sorry

end solve_inequality_l572_572393


namespace tan_product_l572_572145

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l572_572145


namespace probability_ball_sports_l572_572378

theorem probability_ball_sports (clubs : Finset String)
  (ball_clubs : Finset String)
  (count_clubs : clubs.card = 5)
  (count_ball_clubs : ball_clubs.card = 3)
  (h1 : "basketball" ∈ clubs)
  (h2 : "soccer" ∈ clubs)
  (h3 : "volleyball" ∈ clubs)
  (h4 : "swimming" ∈ clubs)
  (h5 : "gymnastics" ∈ clubs)
  (h6 : "basketball" ∈ ball_clubs)
  (h7 : "soccer" ∈ ball_clubs)
  (h8 : "volleyball" ∈ ball_clubs) :
  (2 / ((5 : ℝ) * (4 : ℝ)) * ((3 : ℝ) * (2 : ℝ)) = (3 / 10)) :=
by
  sorry

end probability_ball_sports_l572_572378


namespace circle_area_ratio_l572_572055

theorem circle_area_ratio : 
  ∀ (s : ℝ), 
  let R := s / Real.sqrt 3 in
  let r := (s * Real.sqrt 3) / 6 in
  let s' := r * Real.sqrt 2 in
  let r' := s' / 2 in
  (π * r'^2) / (π * R^2) = 1 / 8 :=
by 
  intros s
  let R := s / Real.sqrt 3
  let r := (s * Real.sqrt 3) / 6
  let s' := r * Real.sqrt 2
  let r' := s' / 2
  have : (π * r'^2) / (π * R^2) = (r'^2) / (R^2) := by sorry
  have r'2 : r'^2 = ((r * Real.sqrt 2) / 2)^2 := by sorry
  have R2 : R^2 = (s / Real.sqrt 3)^2 := by sorry
  have : ((r * Real.sqrt 2) / 2)^2 = (s^2 * 3 * 2) / (36 * 4) := by sorry
  have : (s / Real.sqrt 3)^2 = s^2 / 3 := by sorry
  have : ((s^2 * 3 * 2) / (36 * 4)) / (s^2 / 3) = 1/8 := by sorry
  exact this

end circle_area_ratio_l572_572055


namespace sum_digits_0_to_2012_l572_572317

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

theorem sum_digits_0_to_2012 : ∑ n in Finset.range 2013, sum_of_digits n = 28077 := 
by
  sorry

end sum_digits_0_to_2012_l572_572317


namespace populations_equal_in_years_l572_572867

-- Definitions
def populationX (n : ℕ) : ℤ := 68000 - 1200 * n
def populationY (n : ℕ) : ℤ := 42000 + 800 * n

-- Statement to prove
theorem populations_equal_in_years : ∃ n : ℕ, populationX n = populationY n ∧ n = 13 :=
sorry

end populations_equal_in_years_l572_572867


namespace unit_diameter_circle_can_be_placed_l572_572747

noncomputable def exists_unit_diameter_circle : Prop :=
  ∃ (circle_center : ℝ × ℝ),
    circle_center ∈ set.Icc (0.5, 0.5) (19.5, 24.5) ∧
    ∀ (square_center : ℝ × ℝ) (square_side : ℝ),
      (square_center.1 - 1 / 2 ≤ circle_center.1) ∧ 
      (square_center.1 + 1 / 2 ≥ circle_center.1) ∧
      (square_center.2 - 1 / 2 ≤ circle_center.2) ∧ 
      (square_center.2 + 1 / 2 ≥ circle_center.2) →
      (dist circle_center (square_center.1, square_center.2) ≥ 0.5)

theorem unit_diameter_circle_can_be_placed (rect_width rect_height : ℝ) (num_squares : ℕ) :
  rect_width = 20 ∧ rect_height = 25 ∧ num_squares = 120 →
  ∃ circle_center, 
    circle_center ∈ set.Icc (0.5, 0.5) (19.5, 24.5) ∧
    ∀ (square_center : ℝ × ℝ) (square_side : ℝ),
      (square_center.1 - 1 / 2 ≤ circle_center.1) ∧ 
      (square_center.1 + 1 / 2 ≥ circle_center.1) ∧
      (square_center.2 - 1 / 2 ≤ circle_center.2) ∧ 
      (square_center.2 + 1 / 2 ≥ circle_center.2) →
      (dist circle_center (square_center.1, square_center.2) ≥ 0.5) :=
sorry

end unit_diameter_circle_can_be_placed_l572_572747


namespace tan_identity_l572_572160

theorem tan_identity :
  tan (π / 9) * tan (2 * π / 9) * tan (4 * π / 9) = 3 :=
sorry

end tan_identity_l572_572160


namespace susan_avg_speed_l572_572894

theorem susan_avg_speed 
  (speed1 : ℕ)
  (distance1 : ℕ)
  (speed2 : ℕ)
  (distance2 : ℕ)
  (no_stops : Prop) 
  (H1 : speed1 = 15)
  (H2 : distance1 = 40)
  (H3 : speed2 = 60)
  (H4 : distance2 = 20)
  (H5 : no_stops) :
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 20 := by
  sorry

end susan_avg_speed_l572_572894


namespace value_of_f_g6_minus_g_f6_l572_572231

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g6_minus_g_f6 : f (g 6) - g (f 6) = 48 :=
by
  sorry

end value_of_f_g6_minus_g_f6_l572_572231


namespace pyramid_cone_radius_l572_572742

open Real

-- Definition of a regular pyramid and the parameters.
variable (a : ℝ) -- Side of the base of the pyramid
variable (BE OE OB r x : ℝ) 

-- Conditions from the problem statement
axiom ratio_condition : OE = 2 / 3 * BE
axiom height_condition : BE = a * sqrt 3 / 2
axiom radius_condition : r = a / 4

-- Statement encompassing the problem's solution
theorem pyramid_cone_radius (a : ℝ) (BE OE OB r x : ℝ) 
  (h1 : OE = 2 / 3 * BE)
  (h2 : BE = a * sqrt 3 / 2)
  (h3 : r = a / 4) :
  r = a / 4 ∧ 
  x = a * sqrt 3 / (2 * (2 * cos (1 / 3 * π / 2))) :=
sorry

end pyramid_cone_radius_l572_572742


namespace tan_product_eq_three_l572_572078

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end tan_product_eq_three_l572_572078


namespace matt_ordered_15_pounds_l572_572786

def total_weight_ounces (num_steaks : ℕ) (ounces_per_steak : ℕ) : ℕ :=
  num_steaks * ounces_per_steak

def total_weight_pounds (weight_ounces : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  weight_ounces / ounces_per_pound

theorem matt_ordered_15_pounds:
  (num_steaks : ℕ) (ounces_per_steak : ℕ) (ounces_per_pound : ℕ):
  num_steaks = 20 → ounces_per_steak = 12 → ounces_per_pound = 16 →
  total_weight_pounds (total_weight_ounces num_steaks ounces_per_steak) ounces_per_pound = 15 :=
by
  sorry

end matt_ordered_15_pounds_l572_572786


namespace sum_of_reciprocals_l572_572416

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) : 
  (1 / x + 1 / y = 1 / 2) :=
by 
  sorry

end sum_of_reciprocals_l572_572416


namespace num_non_divisible_by_3_divisors_l572_572670

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l572_572670


namespace number_of_cherry_pie_days_l572_572820

theorem number_of_cherry_pie_days (A C : ℕ) (h1 : A + C = 7) (h2 : 12 * A = 12 * C + 12) : C = 3 :=
sorry

end number_of_cherry_pie_days_l572_572820


namespace regular_icosahedron_edges_l572_572651

-- Define what a regular icosahedron is
def is_regular_icosahedron (P : Type) := -- Definition placeholder for a regular icosahedron
  sorry

-- Define the function that counts edges of a polyhedron
def count_edges (P : Type) [is_regular_icosahedron P] : ℕ :=
  sorry

-- The proof statement
theorem regular_icosahedron_edges (P : Type) [h : is_regular_icosahedron P] : count_edges P = 30 :=
  sorry

end regular_icosahedron_edges_l572_572651


namespace smallest_k_l572_572874

theorem smallest_k :
  ∃ k : ℤ, k > 1 ∧ k % 13 = 1 ∧ k % 8 = 1 ∧ k % 4 = 1 ∧ k = 105 :=
by
  sorry

end smallest_k_l572_572874


namespace part1_part2_l572_572352

def f (x a : ℝ) := |x - a| + x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  {x : ℝ | f x a ≥ x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} := 
by
  sorry

theorem part2 (a : ℝ) (h : {x : ℝ | f x a ≤ 3 * x} = {x | x ≥ 2}) : 
  a = 6 := 
by
  sorry

end part1_part2_l572_572352


namespace poland_2001_q2_l572_572767

def is_coprime (a n : ℕ) : Prop := Nat.gcd a n = 1

theorem poland_2001_q2 (n : ℕ) (h : n ≥ 3) : 
  n ∣ ∑ k in (Finset.filter (λ a, is_coprime a n) (Finset.range (n+1))), k^3 := 
by
  sorry

end poland_2001_q2_l572_572767


namespace least_time_meeting_l572_572421

open Nat

theorem least_time_meeting (k : ℕ) (h : 1 ≤ k ∧ k ≤ 8) : 
  ∃ T > 0, (∀ n ∈ ({2, 4, 6, 8, 10, 12, 14, 16} : Finset ℕ), k = n → ∃ m, T = n * m) ∧ 
  (Finset.card ({n | ∃ m, T = n * m ∧ n ∈ {2, 4, 6, 8, 10, 12, 14, 16}}) ≥ 4) ∧ 
  (digits 10 T).sum = 6 :=
by
  sorry

end least_time_meeting_l572_572421


namespace meal_total_cost_l572_572203

theorem meal_total_cost (x : ℝ) (h_initial: x/5 - 15 = x/8) : x = 200 :=
by sorry

end meal_total_cost_l572_572203


namespace number_of_positive_divisors_not_divisible_by_3_of_180_l572_572694

theorem number_of_positive_divisors_not_divisible_by_3_of_180 : 
  (finset.card ((finset.range 3).bind (λ a, (finset.range 2).image (λ c, 2^a * 5^c)))) = 6 :=
sorry

end number_of_positive_divisors_not_divisible_by_3_of_180_l572_572694


namespace max_min_sum_on_interval_l572_572272

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem max_min_sum_on_interval : 
  let M := Real.max (f (-3)) (Real.max (f (-1)) (f 0))
  let m := Real.min (f (-3)) (Real.min (f (-1)) (f 0))
  M + m = -14 :=
by
  let M := Real.max (f (-3)) (Real.max (f (-1)) (f 0))
  let m := Real.min (f (-3)) (Real.min (f (-1)) (f 0))
  sorry

end max_min_sum_on_interval_l572_572272


namespace min_value_arithmetic_sequence_l572_572763

theorem min_value_arithmetic_sequence :
  ∃ d : ℤ, ∀ a_1 a_2 a_3 : ℤ, a_1 = 3 → a_2 = a_1 + d → a_3 = a_2 + d → 3 * a_2 + 7 * a_3 = -∞ :=
sorry

end min_value_arithmetic_sequence_l572_572763


namespace midpoint_tracer_minimum_usage_l572_572918

theorem midpoint_tracer_minimum_usage
(starting_distance : ℝ)
(target_distance_min : ℝ)
(target_distance_max : ℝ)
(points : ℕ → ℝ)
(midpoint : ℝ → ℝ → ℝ) 
(h1 : starting_distance = 1)
(h2 : target_distance_min = 1 / 2017)
(h3 : target_distance_max = 1 / 2016)
(h4 : ∀ n, (points n + points (n+1)) = midpoint (points n) (points (n+1)))
: ∃ n, n = 17 ∧ 
  target_distance_min < real.dist (points (n-1)) (points n) ∧
  real.dist (points (n-1)) (points n) < target_distance_max := 
sorry

end midpoint_tracer_minimum_usage_l572_572918


namespace solve_problem_l572_572341

noncomputable def problem_statement : Prop :=
  let α := 3 + Real.sqrt 8
  let β := 3 - Real.sqrt 8
  let x := α ^ 12
  let n := Real.floor x
  let f := x - n
  x * (1 - f) = 1

theorem solve_problem : problem_statement :=
by
  let α := 3 + Real.sqrt 8
  let β := 3 - Real.sqrt 8
  let x := α ^ 12
  let n := Real.floor x
  let f := x - n
  have : x * (1 - f) = 1 := sorry
  exact this

end solve_problem_l572_572341


namespace quadratic_trinomials_unique_root_value_l572_572861

theorem quadratic_trinomials_unique_root_value (p q : ℝ) :
  ∀ x, (x^2 + p * x + q) + (x^2 + q * x + p) = (2 * x^2 + (p + q) * x + (p + q)) →
  (((p + q = 0 ∨ p + q = 8) → (2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 8 ∨ 2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 32))) :=
by
  sorry

end quadratic_trinomials_unique_root_value_l572_572861
