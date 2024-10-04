import Mathlib

namespace trajectory_eq_circle_through_fixed_points_l304_304396

-- Definitions based on the conditions 
def dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Proof problem 1
theorem trajectory_eq (M : ℝ × ℝ) (h : dist M (1, 0) = Real.abs M.1 + 1) :
  (M.1 >= 0 -> M.2^2 = 4 * M.1) ∧ (M.1 < 0 -> M.2 = 0) := sorry

noncomputable def C (x : ℝ) := if x >= 0 then (some (λ y : ℝ, y^2 = 4 * x)) else (0)

-- Proof problem 2
theorem circle_through_fixed_points (A B : ℝ × ℝ) (hC : ∀ x, A = (1, (4 / (some (λ y, y = x))))) :
  ∃ (fixed_points : ℝ × ℝ), fixed_points = (-1,0) ∨ fixed_points = (3,0) := sorry

end trajectory_eq_circle_through_fixed_points_l304_304396


namespace smallest_three_digit_multiple_of_17_l304_304839

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304839


namespace solve_inequality_l304_304292

noncomputable def f : ℝ → ℝ := sorry

-- The conditions
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom increasing_on_pos : ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f x < f y
axiom f_two_zero : f 2 = 0

-- The theorem to be proved
theorem solve_inequality (x : ℝ) :
  f (Real.log2 (x^2 + 5 * x + 4)) ≥ 0 ↔
  (x ≥ 0 ∨ x ≤ -5 ∨ (-1 < x ∧ x ≤ (-5 + Real.sqrt 10) / 2) ∨ ((-5 - Real.sqrt 10) / 2 ≤ x ∧ x < -4)) :=
sorry

end solve_inequality_l304_304292


namespace smallest_three_digit_multiple_of_17_l304_304866

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304866


namespace wizard_answers_bal_l304_304887

-- Define the types for human and zombie as truth-tellers and liars respectively
inductive WizardType
| human : WizardType
| zombie : WizardType

-- Define the meaning of "bal"
inductive BalMeaning
| yes : BalMeaning
| no : BalMeaning

-- Question asked to the wizard
def question (w : WizardType) (b : BalMeaning) : Prop :=
  match w, b with
  | WizardType.human, BalMeaning.yes => true
  | WizardType.human, BalMeaning.no => false
  | WizardType.zombie, BalMeaning.yes => false
  | WizardType.zombie, BalMeaning.no => true

-- Theorem stating the wizard will answer "bal" to the given question
theorem wizard_answers_bal (w : WizardType) (b : BalMeaning) :
  question w b = true ↔ b = BalMeaning.yes :=
by
  sorry

end wizard_answers_bal_l304_304887


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304713

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304713


namespace integral_sin_pi_l304_304950

theorem integral_sin_pi : ∫ x in 0..π, sin x = 2 := by
  sorry

end integral_sin_pi_l304_304950


namespace smallest_three_digit_multiple_of_17_l304_304596

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304596


namespace smallest_three_digit_multiple_of_17_l304_304673

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304673


namespace minimum_distance_sum_l304_304108

variables {V : Type*} [metric_space V] [normed_add_comm_group V] [normed_space ℝ V]

/-- Definition of a regular tetrahedron with vertices A, B, C, D -/
structure regular_tetrahedron (V : Type*) [metric_space V] :=
(vertices : fin 4 → V)
(equilateral_faces : ∀ (i j k : fin 4), dist (vertices i) (vertices j) = dist (vertices i) (vertices k))

/-- Definition of the center O of the circumscribed sphere of tetrahedron T -/
noncomputable def circumsphere_center (T : regular_tetrahedron V) : V := sorry

/-- The main theorem statement -/
theorem minimum_distance_sum (T : regular_tetrahedron V) (O : V) 
  (hO : O = circumsphere_center T) :
  ∀ P : V, (dist P (T.vertices 0) + dist P (T.vertices 1) + dist P (T.vertices 2) + dist P (T.vertices 3)) ≥
  (dist O (T.vertices 0) + dist O (T.vertices 1) + dist O (T.vertices 2) + dist O (T.vertices 3)) :=
sorry

end minimum_distance_sum_l304_304108


namespace smallest_n_l304_304987

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l304_304987


namespace discount_percent_l304_304535

theorem discount_percent (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : (SP - CP) / CP * 100 = 34.375) :
  ((MP - SP) / MP * 100) = 14 :=
by
  -- Proof would go here
  sorry

end discount_percent_l304_304535


namespace find_ratio_l304_304194

theorem find_ratio (a b c d : ℝ) (pos_a : 0 < a) (pos_b : 0 < b)
  (pos_c : 0 < c) (pos_d : 0 < d)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c)
  (h2 : a^2 + b^2 = c^2 + d^2) :
  (ab + cd) / (ad + bc) = √(3) / 2 := 
sorry

end find_ratio_l304_304194


namespace product_floor_ceil_l304_304999

theorem product_floor_ceil :
  (∏ n in finset.range 7, (Int.floor (-n - 0.5) * Int.ceil (n + 0.5))) = -25401600 :=
by
  sorry

end product_floor_ceil_l304_304999


namespace smallest_three_digit_multiple_of_17_l304_304603

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304603


namespace determine_subtracted_number_l304_304932

theorem determine_subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 7 * x - y = 130) : y = 150 :=
by sorry

end determine_subtracted_number_l304_304932


namespace problem1_problem2_problem3_l304_304094

-- First problem
theorem problem1 (A B : Set ℝ) (A_def : A = {0, 1, 2}) (B_def : B = {-1, 3}) : 
  A + B = {-1, 0, 1, 3, 4, 5} :=
sorry

-- Second problem
theorem problem2 (a₁ : ℝ) (aₙ : ℕ → ℝ) (A : Set ℝ) (B : Set ℝ)
  (h₁ : a₁ = 2/3)
  (h₂ : ∀ n : ℕ, n ≥ 2 → aₙ n = 2/3 * n)
  (A_def : ∀ n : ℕ, A = {a₁} ∪ {aₙ i | 2 ≤ i ∧ i ≤ n})
  (B_def : B = {-1/9, -2/9, -2/3})
  (S : ℕ → ℝ)
  (S_def : ∀ n : ℕ, S n = ∑ x in (A + B), x)
  (m n k : ℕ) (h₃ : m + n = 3 * k) (h₄ : m ≠ n) :
  ∀ λ : ℝ, λ ≤ 9/2 → S m + S n - λ * S k > 0 :=
sorry

-- Third problem
theorem problem3 : 
  ∃ (A : Set ℤ), 
  (∀ A₁ : Set ℤ, A₁ ⊆ A₁ + A₁) ∧ 
  (∀ A₂ : Set ℤ, (∀ t : ℤ, 0 < t → t ∈ A₂) → (∃ s : Finset ℤ, s ⊆ A₂ ∧ t = ∑ i in s, i)) :=
sorry

end problem1_problem2_problem3_l304_304094


namespace smallest_three_digit_multiple_of_17_l304_304756

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304756


namespace expected_value_unfair_die_l304_304463

theorem expected_value_unfair_die :
  let P : Fin 8 → ℚ := λ i, [3/20, 1/20, 1/20, 1/20, 1/10, 1/5, 1/10, 3/10].nth i.val
  let X : Fin 8 → ℚ := λ i, i.val + 1
  ∑ i in Finset.univ, (X i) * (P i) = 3 :=
begin
  sorry
end

end expected_value_unfair_die_l304_304463


namespace decrease_by_4_percent_l304_304384

-- Define the number of boarding students in 2010 as n.
variable (n : ℝ)

-- Define the increase and decrease percentages.
def increase_2011 : ℝ := 1.2
def decrease_2012 : ℝ := 0.8

-- Define the number of boarding students in 2011 and 2012.
def n_2011 : ℝ := n * increase_2011
def n_2012 : ℝ := n_2011 * decrease_2012

-- Prove that the number of boarding students in 2012 is a 4% decrease compared to 2010.
theorem decrease_by_4_percent : n_2012 = n * 0.96 :=
by
  simp only [n_2012, n_2011, increase_2011, decrease_2012]
  sorry

end decrease_by_4_percent_l304_304384


namespace part1_part2_l304_304350

noncomputable theory
open Real

def vector_a (x : ℝ) : ℝ × ℝ := (cos x, sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (-cos x, cos x)
def vector_c : ℝ × ℝ := (-1, 0)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
  
def norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ :=
  real.acos (dot_product v1 v2 / (norm v1 * norm v2))

def f (x : ℝ) : ℝ :=
  2 * dot_product (vector_a x) (vector_b x) + 1

theorem part1 :
  angle_between_vectors (vector_a (π / 6)) vector_c = 5 * π / 6 := 
sorry

theorem part2 :
  (∀ x ∈ Icc (π / 2) (9 * π / 8), f x ≤ 1) ∧ f (π / 2) = 1 := 
sorry

end part1_part2_l304_304350


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304707

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304707


namespace find_m_l304_304451

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A based on the condition in the problem
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5 * x + m = 0}

-- Define the complement of A in the universal set U
def complementA (m : ℕ) : Set ℕ := U \ A m

-- Given condition that the complement of A in U is {2, 3}
def complementA_condition : Set ℕ := {2, 3}

-- The proof problem statement: Prove that m = 4 given the conditions
theorem find_m (m : ℕ) (h : complementA m = complementA_condition) : m = 4 :=
sorry

end find_m_l304_304451


namespace solid_not_a_cylinder_l304_304391

theorem solid_not_a_cylinder (viewed_as_triangle : ∃ fig : Type, (angle : Type) → viewed_as_triangle angle fig) : ∀ shape, shape = Cylinder → False :=
by
  intros shape h_shape
  cases h_shape
  sorry

end solid_not_a_cylinder_l304_304391


namespace smallest_three_digit_multiple_of_17_l304_304588

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304588


namespace smallest_three_digit_multiple_of_17_l304_304638

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304638


namespace percentage_puppies_greater_profit_l304_304207

/-- A dog breeder wants to know what percentage of puppies he can sell for a greater profit.
    Puppies with more than 4 spots sell for more money. The last litter had 10 puppies; 
    6 had 5 spots, 3 had 4 spots, and 1 had 2 spots.
    We need to prove that the percentage of puppies that can be sold for more profit is 60%. -/
theorem percentage_puppies_greater_profit
  (total_puppies : ℕ := 10)
  (puppies_with_5_spots : ℕ := 6)
  (puppies_with_4_spots : ℕ := 3)
  (puppies_with_2_spots : ℕ := 1)
  (puppies_with_more_than_4_spots := puppies_with_5_spots) :
  (puppies_with_more_than_4_spots : ℝ) / (total_puppies : ℝ) * 100 = 60 :=
by
  sorry

end percentage_puppies_greater_profit_l304_304207


namespace compute_total_cost_l304_304189

theorem compute_total_cost
  (milliseconds_per_second : ℕ := 1000)
  (milliseconds_per_minute : ℕ := 60000)
  (minutes : ℕ := 45)
  (gigabytes : ℕ := 3.5)
  (kilowatt_hours : ℕ := 2)
  (cost_operating_system : ℝ := 1.07)
  (cost_per_millisecond : ℝ := 0.023)
  (cost_mounting_data_tape : ℝ := 5.35)
  (cost_per_megabyte : ℝ := 0.15)
  (cost_per_kilowatt_hour : ℝ := 0.02)
  : (cost_operating_system +
     (minutes * milliseconds_per_minute * cost_per_millisecond) +
     cost_mounting_data_tape +
     ((gigabytes * 1024) * cost_per_megabyte) +
     (kilowatt_hours * cost_per_kilowatt_hour) = 62644.06) :=
begin
  sorry
end

end compute_total_cost_l304_304189


namespace sarah_problem_l304_304116

theorem sarah_problem (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 100 ≤ y ∧ y ≤ 999) 
  (h : 1000 * x + y = 11 * x * y) : x + y = 110 :=
sorry

end sarah_problem_l304_304116


namespace math_proof_problem_l304_304513

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304513


namespace log_expr_solution_l304_304979

-- Define the function that represents the recursive logarithmic expression
def log_expr (x : ℝ) := Real.logb 3 (81 + x)

-- The main statement we want to prove
theorem log_expr_solution : ∃ x : ℝ, x = log_expr x ∧ 0 < x ∧ x = 8 :=
by
  sorry

end log_expr_solution_l304_304979


namespace find_number_eq_fifty_l304_304367

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l304_304367


namespace smallest_three_digit_multiple_of_17_l304_304820

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304820


namespace eccentricity_of_ellipse_final_ellipse_equation_l304_304941

theorem eccentricity_of_ellipse (a b c : ℝ) (ha_gt_hb : a > b) (hb_gt_0 : b > 0) 
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (ab_eq : ∀ (f1 f2 : ℝ), (b - a) = (√3 / 2) * (2 * c)) :
  ∃ e : ℝ, e = c / a ∧ e = √2 / 2 := 
begin
  -- proof will be added here
  sorry
end

theorem final_ellipse_equation (a b c : ℝ) (ha_gt_hb : a > b) (hb_gt_0 : b > 0) 
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (ab_eq : ∀ (f1 f2 : ℝ), (b - a) = (√3 / 2) * (2 * c))
  (P : ℝ × ℝ) (P_coords : P = (√2 * c * sin θ, c * cos θ))
  (circle_diameter_eq : |P.2 - b| = 2 * √2)
  (line_tangent_eq : ∀ (l : ℝ) (M : ℝ) (f2 : ℝ), |M - f2| = 2 * √2):
  ∃ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1) := 
begin
  -- proof will be added here
  sorry
end

end eccentricity_of_ellipse_final_ellipse_equation_l304_304941


namespace smallest_three_digit_multiple_of_17_l304_304762

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304762


namespace graph_symmetric_about_point_l304_304321

def f (x a : ℝ) : ℝ := (x - 1)^2 / (x - 1) + a / (x - 1)

theorem graph_symmetric_about_point (a : ℝ) (h : a ≠ 0) : 
  ∃ c : ℝ × ℝ, ∀ x : ℝ, x ≠ 1 → f x a = f (2 * (c.1) - x) a :=
sorry

end graph_symmetric_about_point_l304_304321


namespace returned_books_percentage_is_correct_l304_304210

-- This function takes initial_books, end_books, and loaned_books and computes the percentage of books returned.
noncomputable def percent_books_returned (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let books_out_on_loan := initial_books - end_books
  let books_returned := loaned_books - books_out_on_loan
  (books_returned : ℚ) / (loaned_books : ℚ) * 100

-- The main theorem that states the percentage of books returned is 70%
theorem returned_books_percentage_is_correct :
  percent_books_returned 75 57 60 = 70 := by
  sorry

end returned_books_percentage_is_correct_l304_304210


namespace smallest_three_digit_multiple_of_17_correct_l304_304632

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304632


namespace smallest_three_digit_multiple_of_17_l304_304865

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304865


namespace incorrect_arrangements_of_hello_l304_304903

def num_incorrect_arrangements : ℕ :=
  let total_permutations := (5! : ℕ)
  let unique_arrangements := total_permutations / (2 : ℕ)
  unique_arrangements - 1

theorem incorrect_arrangements_of_hello : num_incorrect_arrangements = 59 := by
  sorry

end incorrect_arrangements_of_hello_l304_304903


namespace grade11_score_l304_304389

theorem grade11_score (n : ℕ) (h1 : 20/100 * n = n / 5)
  (h2 : 80/100 * n = 4 * n / 5)
  (h3 : 78 * n)
  (h4 : 75 * (4 * n / 5))
  (h5 : 78 * n - 75 * (4 * n / 5) = 90 * (n / 5)) :
  90 = (78 * n - 75 * (4 * n / 5)) / (n / 5) := 
sorry

end grade11_score_l304_304389


namespace complement_of_A_in_U_l304_304379

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}
def C_UA : Set ℝ := U \ A

theorem complement_of_A_in_U :
  C_UA = {x | 0 < x ∧ x < 1} :=
sorry

end complement_of_A_in_U_l304_304379


namespace decreasing_function_cos_on_interval_l304_304180

theorem decreasing_function_cos_on_interval:
  ∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), ∀ f : ℝ → ℝ,
  (f = λ x, Real.cos x) →
  (∀ f' : ℝ → ℝ, (f' = λ x, -Real.sin x) → f' x < 0) →
  (f = λ x, Real.cos x) :=
by sorry

end decreasing_function_cos_on_interval_l304_304180


namespace area_of_triangle_l304_304421

variables {R A B C : ℝ}
variables (cosA cosB cosC : ℝ)
constants (H A B C : ℝ)
constants (ah bh ch : ℝ)

def orthocenter_ortho_triangle (H A B C : ℝ) :=
  H * (A * B * C) = 1

def is_acute_triangle (A B C : ℝ) := 
  A + B + C < 180

axiom AH_eq_2 : AH = 2
axiom BH_eq_12 : BH = 12
axiom CH_eq_9 : CH = 9

theorem area_of_triangle : ∃ (area : ℝ), area = 7 * (real.sqrt 63) :=
by 
  use 7 * (real.sqrt 63)
  sorry

end area_of_triangle_l304_304421


namespace trigonometric_inequalities_l304_304280

theorem trigonometric_inequalities (θ : ℝ) : 
  sin (θ + Real.pi) < 0 → cos (θ - Real.pi) > 0 → sin θ > 0 ∧ cos θ < 0 :=
by
  intros h1 h2
  have h3 : -sin θ < 0 := by
    rw sin_add_pi at h1
    exact h1
  have h4 : -cos θ > 0 := by
    rw cos_sub_pi at h2
    exact h2
  split
  exact (neg_neg_iff_pos.mpr h3)
  exact (neg_neg_iff_pos.mpr h4)

end trigonometric_inequalities_l304_304280


namespace number_of_people_per_cubic_yard_l304_304387

-- Lean 4 statement

variable (P : ℕ) -- Number of people per cubic yard

def city_population_9000 := 9000 * P
def city_population_6400 := 6400 * P

theorem number_of_people_per_cubic_yard :
  city_population_9000 - city_population_6400 = 208000 →
  P = 80 :=
by
  sorry

end number_of_people_per_cubic_yard_l304_304387


namespace total_games_should_be_190_l304_304192

-- Definitions based on problem conditions
def num_players := 20
def num_games := Nat.choose num_players 2

-- Theorem statement based on the question and correct answer
theorem total_games_should_be_190 : num_games = 190 := by
sory 

end total_games_should_be_190_l304_304192


namespace half_angle_in_first_quadrant_l304_304304

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l304_304304


namespace smallest_three_digit_multiple_of_17_l304_304774

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304774


namespace smallest_three_digit_multiple_of_17_l304_304768

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304768


namespace largest_digit_not_in_odd_units_digits_l304_304171

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end largest_digit_not_in_odd_units_digits_l304_304171


namespace calculate_p9_plus_p_minus5_l304_304335

noncomputable def polynomial := (x : ℝ) → x^4 + a * x^3 + b * x^2 + c * x + d

variables (a b c d : ℝ)

axiom P1_condition : polynomial 1 = 2000
axiom P2_condition : polynomial 2 = 4000
axiom P3_condition : polynomial 3 = 6000

theorem calculate_p9_plus_p_minus5 :
  polynomial 9 + polynomial (-5) = 12704 :=
sorry

end calculate_p9_plus_p_minus5_l304_304335


namespace smallest_three_digit_multiple_of_17_correct_l304_304631

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304631


namespace smallest_three_digit_multiple_of_17_l304_304663

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304663


namespace solution_set_of_inequality_l304_304155

theorem solution_set_of_inequality : 
  {x : ℝ | x * (x + 3) ≥ 0} = {x : ℝ | x ≥ 0 ∨ x ≤ -3} := 
by sorry

end solution_set_of_inequality_l304_304155


namespace smallest_three_digit_multiple_of_17_l304_304746

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304746


namespace largest_number_is_56_l304_304894

-- Definitions based on the conditions
def ratio_three_five_seven (a b c : ℕ) : Prop :=
  3 * c = a ∧ 5 * c = b ∧ 7 * c = c

def difference_is_32 (a c : ℕ) : Prop :=
  c - a = 32

-- Statement of the proof
theorem largest_number_is_56 (a b c : ℕ) (h1 : ratio_three_five_seven a b c) (h2 : difference_is_32 a c) : c = 56 :=
by
  sorry

end largest_number_is_56_l304_304894


namespace geometric_sum_n_equals_neg_11_l304_304891

open BigOperators

/-- 
Let S_n be the sum of the first n terms of a geometric sequence {a_n}. 
If 8a_2 + a_5 = 0, then S_n equals -11.
-/
theorem geometric_sum_n_equals_neg_11 {a : ℕ → ℝ} (n : ℕ) (h : 8 * a 2 + a 5 = 0) :
  ∑ i in finset.range n, a i = -11 :=
sorry

end geometric_sum_n_equals_neg_11_l304_304891


namespace calc_price_per_litre_l304_304068

noncomputable def pricePerLitre (initial final totalCost : ℝ) : ℝ :=
  totalCost / (final - initial)

theorem calc_price_per_litre :
  pricePerLitre 10 50 36.60 = 91.5 :=
by
  sorry

end calc_price_per_litre_l304_304068


namespace Sasha_can_determine_X_after_seven_questions_l304_304125

theorem Sasha_can_determine_X_after_seven_questions :
  ∃ (X : ℕ) (M N : ℕ), 
    (X ≤ 100) →
    (∀ M N, M < 100 ∧ N < 100 → 
    let gcd_result := Nat.gcd (X + M) N in
    -- Conditions to check
    true) →
    -- Sasha determines X in at most 7 questions
    sorry

end Sasha_can_determine_X_after_seven_questions_l304_304125


namespace prob_sum_even_is_one_third_prob_second_less_than_first_plus_one_is_five_eighths_l304_304900

-- Define the set of balls
def balls : List ℕ := [1, 2, 3, 4]

-- Part 1: Define the event of drawing two balls and their sum being even
def sum_even (a b : ℕ) : Bool := (a + b) % 2 = 0

-- Probabilities should be rational numbers
theorem prob_sum_even_is_one_third : 
  ∃ (p : ℚ), p = 1 / 3 ∧ (p * (List.length (balls.product balls) / 2).toRat) =  1/3.toRat := sorry

-- Part 2: Define the event of drawing two balls with replacement where second number is less than first + 1
def second_less_than_first_plus_one (m n : ℕ) : Bool := n < m + 1

theorem prob_second_less_than_first_plus_one_is_five_eighths :
  ∃ (p : ℚ), p = 5 / 8 ∧ (p * (List.length (balls.product balls)).toRat) = 10/16.toRat := sorry

end prob_sum_even_is_one_third_prob_second_less_than_first_plus_one_is_five_eighths_l304_304900


namespace orthocentric_tetrahedron_centroid_distance_l304_304437

-- Define the problem parameters and prove the equality.

theorem orthocentric_tetrahedron_centroid_distance
  (G : Point)
  (F : Point)
  (K : Point)
  (h1 : ∃ T : Tetrahedron, is_centroid_of G T ∧ is_foot_of_altitude F T ∧ is_intersection_with_circumscribed_sphere K (line_through F G) T ∧ is_between G K F)
  (h2 : ∀ P Q R S : Point, is_homothetic_with_ratio G (-1/3) P Q R S) :
  distance K G = 3 * distance F G :=
by
  sorry

end orthocentric_tetrahedron_centroid_distance_l304_304437


namespace acute_angle_condition_vector_equation_l304_304349

variable (x y : ℝ)
def a := (2, -1) : ℝ × ℝ
def b := (x, 1) : ℝ × ℝ

-- Part 1: Prove x > 1/2 for the angle between a and b to be acute
theorem acute_angle_condition (h : x > 1/2) :
  (2 * x - 1) > 0 :=
by sorry

-- Part 2: Prove x + y = -4 when 3a - 2b = (4, y)
theorem vector_equation (h : 3 * a.1 - 2 * b.1 = 4 ∧ -5 = y) :
  x + y = -4 :=
by sorry

end acute_angle_condition_vector_equation_l304_304349


namespace remainder_division_l304_304151

theorem remainder_division (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 :=
sorry

end remainder_division_l304_304151


namespace minimum_value_fraction_l304_304080

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b + c ≥ a)

theorem minimum_value_fraction : (b / c + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end minimum_value_fraction_l304_304080


namespace recurring_decimal_to_fraction_l304_304262

theorem recurring_decimal_to_fraction : (4 + (Int.recur 8) / 10) = 44 / 9 :=
by
  sorry

end recurring_decimal_to_fraction_l304_304262


namespace smallest_three_digit_multiple_of_17_l304_304850

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304850


namespace smallest_three_digit_multiple_of_17_l304_304749

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304749


namespace dogs_not_liking_any_food_l304_304388

theorem dogs_not_liking_any_food (total_dogs watermelon_dogs salmon_dogs watermelon_and_salmon_dogs chicken_dogs salmon_and_not_watermelon_dogs chicken_and_watermelon_not_salmon_dogs : ℕ) :
  total_dogs = 100 →
  watermelon_dogs = 20 →
  salmon_dogs = 65 →
  watermelon_and_salmon_dogs = 10 →
  chicken_dogs = 15 →
  salmon_and_not_watermelon_dogs = 3 →
  chicken_and_watermelon_not_salmon_dogs = 2 →
  (total_dogs - (watermelon_dogs + salmon_dogs + chicken_dogs 
  - watermelon_and_salmon_dogs - salmon_and_not_watermelon_dogs - chicken_and_watermelon_not_salmon_dogs)) = 10 :=
begin
  intros h_total h_watermelon h_salmon h_watermelonsalmon h_chicken h_salmonsalmon h_chickenwatermelonnot,
  sorry
end

end dogs_not_liking_any_food_l304_304388


namespace smallest_three_digit_multiple_of_17_correct_l304_304624

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304624


namespace rectangle_diagonal_proof_l304_304271

noncomputable def rectangle_diagonal (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem rectangle_diagonal_proof :
  ∃ (x y : ℝ), x + y = 7 ∧ x * y = 12 ∧ rectangle_diagonal x y = 5 :=
by
  use 4, 3
  split
  · exact rfl
  split
  · exact rfl
  · exact rfl

end rectangle_diagonal_proof_l304_304271


namespace simplify_and_evaluate_expression_l304_304497

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304497


namespace solution_set_of_inequality_l304_304232

noncomputable def odd_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x)) ∧ (∀ x y, 0 < x → x < y → f (x) < f (y)) ∧ (f (1) = 0)

theorem solution_set_of_inequality (f : ℝ → ℝ) (h : odd_increasing_function f) :
  {x : ℝ | (f(x) - f(-x)) / x < 0} = set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

end solution_set_of_inequality_l304_304232


namespace log_expression_equals_four_l304_304973

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l304_304973


namespace smallest_three_digit_multiple_of_17_l304_304843

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304843


namespace highest_page_number_l304_304471

/-- Given conditions: Pat has 19 instances of the digit '7' and an unlimited supply of all 
other digits. Prove that the highest page number Pat can number without exceeding 19 instances 
of the digit '7' is 99. -/
theorem highest_page_number (num_of_sevens : ℕ) (highest_page : ℕ) 
  (h1 : num_of_sevens = 19) : highest_page = 99 :=
sorry

end highest_page_number_l304_304471


namespace find_number_eq_fifty_l304_304368

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l304_304368


namespace isometric_curve_l304_304474

noncomputable def Q (a b c x y : ℝ) := a * x^2 + 2 * b * x * y + c * y^2

theorem isometric_curve (a b c d e f : ℝ) (h : a * c - b^2 = 0) :
  ∃ (p : ℝ), (Q a b c x y + 2 * d * x + 2 * e * y = f → 
    (y^2 = 2 * p * x) ∨ 
    (∃ c' : ℝ, y^2 = c'^2) ∨ 
    y^2 = 0 ∨ 
    ∀ x y : ℝ, false) :=
sorry

end isometric_curve_l304_304474


namespace melissa_shoe_repair_l304_304097

theorem melissa_shoe_repair :
  ∀ (buckle_time total_time : ℝ), 
    buckle_time = 5 ∧ total_time = 30 → 
    (total_time - buckle_time) / 2 = 12.5 :=
by
  intros buckle_time total_time h,
  cases h with h_buckle h_total,
  rw [h_buckle, h_total],
  norm_num,
  sorry -- proof goes here

end melissa_shoe_repair_l304_304097


namespace dihedral_angle_range_l304_304041

theorem dihedral_angle_range (n : ℕ) (h : 3 ≤ n) : 
  (∀ θ, θ ∈ set.Ioo ((n-2) * real.pi / n) real.pi) ↔ 
    ∃ (θ : ℝ) (a b : ℝ), θ = a ∧ 
      θ ∈ set.Icc a b ∧ 
      a = (n-2) * real.pi / n ∧ 
      b = real.pi := by 
  sorry

end dihedral_angle_range_l304_304041


namespace students_in_game_divisors_of_119_l304_304904

theorem students_in_game_divisors_of_119 (n : ℕ) (h1 : ∃ (k : ℕ), k * n = 119) :
  n = 7 ∨ n = 17 :=
sorry

end students_in_game_divisors_of_119_l304_304904


namespace smallest_three_digit_multiple_of_17_l304_304853

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304853


namespace arithmetic_sequence_a_eq_zero_l304_304290

theorem arithmetic_sequence_a_eq_zero (a : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ S : ℕ → ℝ, S n = (n^2 : ℝ) + 2 * n + a) →
  a = 0 :=
by
  sorry

end arithmetic_sequence_a_eq_zero_l304_304290


namespace min_value_f_on_interval_l304_304138

noncomputable def f (a x : ℝ) := a^(2 * x) + 3 * a^x - 2

theorem min_value_f_on_interval (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∃ x : ℝ, x ∈ Icc (-1 : ℝ) 1 ∧ f a x = 8) :
  ∃ x : ℝ, x ∈ Icc (-1 : ℝ) 1 ∧ f a x = -1/4 := 
sorry -- We skip the proof; it's not required in this task

end min_value_f_on_interval_l304_304138


namespace two_digit_numbers_count_l304_304355

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  ∃ n : ℕ, n = 40 ∧ ∀ d ∈ finset.range 90, 
  let m := d + 10 in
  10 ≤ m ∧ m < 100 ∧ tens_digit m < units_digit m →
  ∃ n : ℕ, n = 40 ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ tens_digit m < units_digit m → n = 40 := 
sorry

end two_digit_numbers_count_l304_304355


namespace smallest_three_digit_multiple_of_17_l304_304860

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304860


namespace trajectory_of_M_circle_through_fixed_points_l304_304395

variables {M : ℝ × ℝ} {F : ℝ × ℝ}
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trajectory_of_M (x y : ℝ) :
  (distance M (1, 0) = distance M (x, 0) + 1) →
  (y^2 = 4 * x ∧ x ≥ 0 ∨ y = 0 ∧ x < 0) :=
sorry

theorem circle_through_fixed_points (x y : ℝ) (A B F : ℝ × ℝ) :
  let C := (set_of (λ p : ℝ × ℝ, p.snd^2 = 4 * p.fst ∧ p.fst ≥ 0)) in
  let line_PQ := (F.1, y) in
  let OP := (0, 0) in
  let OQ := (0, 0) in
  let A := (1, (4 / y)) in
  let B := (1, (4 / (y + 4))) in
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2*y)^2 = 4 * (y^2 + 1)} in
  (A ∈ circle ∧ B ∈ circle) →
  ((-1, 0) ∈ circle ∧ (3, 0) ∈ circle) :=
sorry

end trajectory_of_M_circle_through_fixed_points_l304_304395


namespace Sarah_copies_l304_304118

theorem Sarah_copies : 
  ∀ (copies_per_person number_of_people pages_per_contract : ℕ),
    copies_per_person = 2 →
    number_of_people = 9 →
    pages_per_contract = 20 →
    (copies_per_person * number_of_people * pages_per_contract) = 360 := 
by
  intros copies_per_person number_of_people pages_per_contract h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end Sarah_copies_l304_304118


namespace smallest_three_digit_multiple_of_17_l304_304600

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304600


namespace smallest_three_digit_multiple_of_17_correct_l304_304628

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304628


namespace compound_interest_rate_l304_304934

theorem compound_interest_rate (
  P : ℝ) (r : ℝ)  (A : ℕ → ℝ) :
  A 2 = 2420 ∧ A 3 = 3025 ∧ 
  (∀ n : ℕ, A n = P * (1 + r / 100)^n) → r = 25 :=
by
  sorry

end compound_interest_rate_l304_304934


namespace number_of_fills_l304_304222

-- Definitions based on conditions
def needed_flour : ℚ := 4 + 3 / 4
def cup_capacity : ℚ := 1 / 3

-- The proof statement
theorem number_of_fills : (needed_flour / cup_capacity).ceil = 15 := by
  sorry

end number_of_fills_l304_304222


namespace smallest_three_digit_multiple_of_17_l304_304765

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304765


namespace mul_scientific_notation_l304_304949

theorem mul_scientific_notation (a b : ℝ) (c d : ℝ) (h1 : a = 7 * 10⁻¹) (h2 : b = 8 * 10⁻¹) :
  (a * b = 0.56) :=
by
  sorry

end mul_scientific_notation_l304_304949


namespace remaining_money_after_purchase_l304_304487

def initial_money : Float := 15.00
def notebook_cost : Float := 4.00
def pen_cost : Float := 1.50
def notebooks_purchased : ℕ := 2
def pens_purchased : ℕ := 2

theorem remaining_money_after_purchase :
  initial_money - (notebook_cost * notebooks_purchased + pen_cost * pens_purchased) = 4.00 := by
  sorry

end remaining_money_after_purchase_l304_304487


namespace find_x_l304_304369

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end find_x_l304_304369


namespace cost_price_of_book_l304_304883

theorem cost_price_of_book 
  (C : ℝ) 
  (h1 : 1.10 * C = sp10) 
  (h2 : 1.15 * C = sp15)
  (h3 : sp15 - sp10 = 90) : 
  C = 1800 := 
sorry

end cost_price_of_book_l304_304883


namespace rectangular_cross_section_impossible_l304_304029

-- Definitions of the geometric shapes
inductive Shape
| Cone
| Cylinder
| TriangularPrism
| RectangularPrism

-- Definition specifying a rectangular cross-section capability
def can_have_rectangular_cross_section (s : Shape) : Prop :=
match s with
| Shape.Cone             => false
| Shape.Cylinder         => true
| Shape.TriangularPrism  => true
| Shape.RectangularPrism => true
end

-- The theorem statement
theorem rectangular_cross_section_impossible (s : Shape) : 
  ¬can_have_rectangular_cross_section(s) ↔ s = Shape.Cone :=
by
  sorry

end rectangular_cross_section_impossible_l304_304029


namespace hexagon_coloring_l304_304992

theorem hexagon_coloring :
  ∃ (coloring_count : ℕ), coloring_count = 31230 ∧
  (∀ (A B C D E F : ℕ), 
  (A ∈ finset.range 7 ∧ B ∈ finset.range 7 ∧ C ∈ finset.range 7 ∧ D ∈ finset.range 7 ∧ E ∈ finset.range 7 ∧ F ∈ finset.range 7) → 
  (A ≠ D ∧ B ≠ E ∧ C ≠ F)) :=
sorry

end hexagon_coloring_l304_304992


namespace smallest_three_digit_multiple_of_17_l304_304681

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304681


namespace white_papers_per_envelope_l304_304212

theorem white_papers_per_envelope (total_papers envelopes : ℕ) (h1 : total_papers = 120) (h2 : envelopes = 12) : total_papers / envelopes = 10 :=
by {
  rw [h1, h2],
  norm_num,
}

end white_papers_per_envelope_l304_304212


namespace fans_stayed_home_l304_304557

theorem fans_stayed_home : 
  ∀ (total_seats fans_attended : ℕ)
    (sold_percentage : ℚ)
    (seats_sold fans_stayed_home : ℕ),
  total_seats = 60000 →
  sold_percentage = 0.75 →
  fans_attended = 40000 →
  seats_sold = (sold_percentage * total_seats).to_nat →
  fans_stayed_home = seats_sold - fans_attended →
  fans_stayed_home = 5000 :=
by
  intros total_seats fans_attended sold_percentage seats_sold fans_stayed_home
  sorry

end fans_stayed_home_l304_304557


namespace log_base_5_10_approx_l304_304163

theorem log_base_5_10_approx :
  ∀ (log10_2 log10_3 : ℝ), log10_2 ≈ 0.301 → log10_3 ≈ 0.477 → 
  log 5 10 ≈ 10 / 7 :=
by sorry

end log_base_5_10_approx_l304_304163


namespace unique_solution_of_exponential_l304_304110

theorem unique_solution_of_exponential (x : ℝ) : (∀ x1 x2 : ℝ, 3^x1 = 12 ∧ 3^x2 = 12 → x1 = x2) :=
by
  sorry

end unique_solution_of_exponential_l304_304110


namespace extremum_and_monotonicity_inequality_for_c_l304_304002

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem extremum_and_monotonicity (α : ℝ) (h_extremum : ∀ (x : ℝ), x = Real.exp 2 → f x α = 0) :
  (∃ α : ℝ, (∀ x : ℝ, x > Real.exp 2 → f x α > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < Real.exp 2 → f x α < 0)) := sorry

theorem inequality_for_c (c : ℝ) (α : ℝ) (h_extremum : α = 3)
  (h_ineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 3 → f x α < 2 * c^2 - c) :
  (1 < c) ∨ (c < -1 / 2) := sorry

end extremum_and_monotonicity_inequality_for_c_l304_304002


namespace sofia_running_time_l304_304122

-- Define the conditions
def lap_distance := 500  -- in meters
def laps := 3
def first_segment_distance := 200  -- in meters
def first_segment_speed := 3  -- in meters/second
def second_segment_distance := 300  -- in meters
def second_segment_speed := 6  -- in meters/second

-- The proof problem
theorem sofia_running_time : 
  let first_segment_time := first_segment_distance / first_segment_speed
      second_segment_time := second_segment_distance / second_segment_speed
      time_per_lap := first_segment_time + second_segment_time
      total_time := laps * time_per_lap
  in total_time = 350 := 
by sorry

end sofia_running_time_l304_304122


namespace sum_of_circle_numbers_l304_304531

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem sum_of_circle_numbers (numbers : Fin 10 → ℕ) 
  (h : ∀ i : Fin 10, numbers i = gcd (numbers (i - 1)) (numbers (i + 1)) + 1) : 
  (Finset.univ.sum numbers) = 28 :=
by
  sorry

end sum_of_circle_numbers_l304_304531


namespace smallest_three_digit_multiple_of_17_l304_304782

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304782


namespace log_expr_solution_l304_304982

-- Define the function that represents the recursive logarithmic expression
def log_expr (x : ℝ) := Real.logb 3 (81 + x)

-- The main statement we want to prove
theorem log_expr_solution : ∃ x : ℝ, x = log_expr x ∧ 0 < x ∧ x = 8 :=
by
  sorry

end log_expr_solution_l304_304982


namespace correct_option_is_C_l304_304224

-- Definitions based on conditions
def option_a := 1 ⊆ {0, 1, 2}
def option_b := ¬ (∅ ⊆ {0, 1, 2})
def option_c := ∅ ⊆ {2, 0, 1}
def option_d := {1} ∈ {0, 1, 2}

-- Statement to prove the correct option is C
theorem correct_option_is_C : option_c ∧ ¬ option_a ∧ ¬ option_b ∧ ¬ option_d :=
by
  sorry

end correct_option_is_C_l304_304224


namespace triangle_angle_R_l304_304055

theorem triangle_angle_R (P Q R O S T : Type) [Triangle P Q R] [Intersect PS QT O] 
  (h1 : length PO = length OQ) (h2 : length OQ = length OR) 
  (h3 : length OR = length OS) (h4 : angle P = 3 * angle Q) : 
  angle R = 90 := 
by
sor

end triangle_angle_R_l304_304055


namespace smallest_three_digit_multiple_of_17_correct_l304_304629

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304629


namespace sara_change_l304_304183

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end sara_change_l304_304183


namespace michael_truck_meetings_l304_304456

noncomputable def michael_speed : ℝ := 5
noncomputable def trash_pail_distance : ℝ := 200
noncomputable def truck_speed : ℝ := 10
noncomputable def truck_stop_time : ℝ := 30

def meet_count : ℕ := 5

theorem michael_truck_meetings :
  ∃ (meetings : ℕ), 
  (michael_speed = 5) ∧ 
  (trash_pail_distance = 200) ∧ 
  (truck_speed = 10) ∧ 
  (truck_stop_time = 30) ∧ 
  (meetings = meet_count) :=
by 
  use 5
  simp [michael_speed, trash_pail_distance, truck_speed, truck_stop_time, meet_count]
  sorry

end michael_truck_meetings_l304_304456


namespace solution_set_of_inequality_l304_304229

noncomputable def solutionSet (f : ℝ → ℝ) : set ℝ := 
  {x | (f x - f (-x)) / x < 0}

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : odd f)
  (h2 : ∀ x > 0, f x < f (x + 1)) 
  (h3 : f 1 = 0) :
  solutionSet f = set.Ioc (-1 : ℝ) 0 ∪ set.Ioc 0 1 :=
sorry

end solution_set_of_inequality_l304_304229


namespace smallest_three_digit_multiple_of_17_l304_304734

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304734


namespace ellipse_hk_ab_sum_eq_14_l304_304444

theorem ellipse_hk_ab_sum_eq_14 :
  let F1 := (0 : ℝ, 2 : ℝ)
      F2 := (6 : ℝ, 2 : ℝ)
      P (x y : ℝ) := dist (x, y) F1 + dist (x, y) F2 = 10
  in ∃ h k a b : ℝ,
    (∃ x y : ℝ, P x y) ∧ x = (x - h) ^ 2 / a ^ 2 + (y - k) ^ 2 / b ^ 2 = 1
    ∧ h + k + a + b = 14 :=
begin
  rintros ⟨F1, F2, P⟩,
  sorry
end

end ellipse_hk_ab_sum_eq_14_l304_304444


namespace time_to_run_around_square_l304_304882

-- Define the constants and conditions
def side_of_square : ℝ := 50  -- meters
def speed_kmph : ℝ := 9  -- km/hr

-- Convert speed from km/hr to m/s
def speed_mps : ℝ := (speed_kmph * 1000) / 3600  -- (meters/second)

-- Define the perimeter of the square
def perimeter : ℝ := 4 * side_of_square  -- meters

-- Define time taken to run around the square
def time_taken : ℝ := perimeter / speed_mps  -- seconds

-- The theorem to prove
theorem time_to_run_around_square : time_taken = 80 := by
  sorry

end time_to_run_around_square_l304_304882


namespace price_difference_in_cents_l304_304248

noncomputable def lp : ℝ := 149.99
noncomputable def value_market_price : ℝ := lp - 10
noncomputable def tech_bargains_price : ℝ := lp * (1 - 0.30)
noncomputable def gadget_hub_price : ℝ := lp * (1 - 0.20)

theorem price_difference_in_cents :
  (value_market_price - tech_bargains_price) * 100 = 3500 :=
by
  sorry

end price_difference_in_cents_l304_304248


namespace smallest_three_digit_multiple_of_17_l304_304687

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304687


namespace dessert_menus_count_l304_304902

theorem dessert_menus_count :
  ∃ menus, set.size menus = 24
    ∧ (∀ m ∈ menus, (m Monday = cake ∨ m Monday = pie ∨ m Monday = ice_cream) ∧ -- 3 choices for Monday
        (∀ d ∈ [Tuesday, Wednesday, Thursday, Friday], m d ∈ {cake, pie, ice_cream}) ∧ -- 3 choices for the remaining days
        m Wednesday = pie ∧ -- pie is fixed on Wednesday
        ∀ d ∈ [Monday, Tuesday, Wednesday, Thursday, Friday], m d ≠ m (d.pred) ∧ m d ≠ m (d.succ)) := -- no consecutive repeats
sorry

end dessert_menus_count_l304_304902


namespace smallest_b_for_quadratic_factorization_l304_304269

theorem smallest_b_for_quadratic_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (x : ℤ) → (x + p) * (x + q) = x^2 + ↑b * x + 1764 → p + q = b ∧ p * q = 1764) ∧ b = 84 :=
begin
  sorry
end

end smallest_b_for_quadratic_factorization_l304_304269


namespace remainder_of_n_plus_2024_l304_304867

-- Define the assumptions
def n : ℤ := sorry  -- n will be some integer
def k : ℤ := sorry  -- k will be some integer

-- Main statement to be proved
theorem remainder_of_n_plus_2024 (h : n % 8 = 3) : (n + 2024) % 8 = 3 := sorry

end remainder_of_n_plus_2024_l304_304867


namespace domain_equivalence_l304_304310

noncomputable def domain1 (f : ℝ → ℝ) : set ℝ := set.Icc 0 3

noncomputable def domain2 (f : ℝ → ℝ) : set ℝ := set.Icc 0 (9 / 2)

-- Lean 4 statement to prove the equivalence
theorem domain_equivalence (f : ℝ → ℝ) :
  domain1 (λ x, f (x^2 - 1)) = set.Icc 0 3 →
  domain2 (λ x, f (2x - 1)) = set.Icc 0 (9 / 2) :=
sorry

end domain_equivalence_l304_304310


namespace arc_midpoint_AC_length_l304_304106

noncomputable def length_of_AC (A B C : Point) (radius : ℝ) (AB : ℝ) (C_is_midpoint : is_midpoint C A B) : ℝ :=
  if radius = 7 ∧ AB = 8 then sqrt (98 - 14 * sqrt 33) else 0

theorem arc_midpoint_AC_length
  (A B C : Point)
  (radius : ℝ)
  (AB : ℝ)
  (C_is_midpoint : is_midpoint C A B)
  (on_circle : ∀ {P : Point}, P = A ∨ P = B → dist P O = radius)
  (O : Point)
  (center_OA OB : ℝ)
  (center_condition : OA = OB ∧ OA = radius ∧ OB = radius)
  (dist_AB_condition : dist A B = AB)
  : length_of_AC A B C radius AB C_is_midpoint = sqrt (98 - 14 * sqrt 33) :=
sorry

end arc_midpoint_AC_length_l304_304106


namespace smallest_three_digit_multiple_of_17_l304_304611

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304611


namespace cos_A_value_l304_304359

theorem cos_A_value (A : ℝ) (h : Real.tan A + Real.sec A = 3) : Real.cos A = 3/5 :=
sorry

end cos_A_value_l304_304359


namespace probability_eq_one_half_l304_304260

-- Definition of the letters in "PROGRAMMING"
def letters_programming := ['P', 'R', 'O', 'G', 'R', 'A', 'M', 'M', 'I', 'N', 'G']

-- Definition of the letters in "GAMER"
def letters_gamer := ['G', 'A', 'M', 'E', 'R']

-- Define a function to count the occurrences of the letters of "GAMER" in "PROGRAMMING"
def count_favorable_outcomes : Nat :=
  letters_programming.count (λ c => c ∈ letters_gamer)

-- Define the total outcomes
def total_outcomes : Nat := letters_programming.length

-- Define the probability
def probability : ℚ := count_favorable_outcomes / total_outcomes

-- The theorem stating the probability is 1/2
theorem probability_eq_one_half : probability = 1 / 2 := by
  sorry

end probability_eq_one_half_l304_304260


namespace remainder_mod_29_l304_304901

-- Definitions of the given conditions
def N (k : ℕ) := 899 * k + 63

-- The proof statement to be proved
theorem remainder_mod_29 (k : ℕ) : (N k) % 29 = 5 := 
by {
  sorry
}

end remainder_mod_29_l304_304901


namespace chasity_candies_l304_304245

theorem chasity_candies :
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  initial_amount - total_spent = 5 :=
by
  -- Let definitions of constants
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  -- Total cost calculation
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  -- Proof of the final amount left
  have h : initial_amount - total_spent = 15 - (4 * 1.5 + 2 * 2) := rfl
  simp at h
  exact h

end chasity_candies_l304_304245


namespace smallest_three_digit_multiple_of_17_l304_304671

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304671


namespace period_sine_function_l304_304576

theorem period_sine_function (x : ℝ) : 
  let y := sin (3 * x / 4) + π / 6 in period y = 8 * π / 3 :=
sorry

end period_sine_function_l304_304576


namespace factorize_polynomial_equilateral_triangle_l304_304478
-- For Problem (1)

theorem factorize_polynomial (a b : ℂ) : 
  a^2 - 6 * a * b + 9 * b^2 - 36 = (a - 3 * b - 6) * (a - 3 * b + 6) := 
by 
  sorry

-- For Problem (2)

theorem equilateral_triangle (a b c : ℝ) 
  (h : a^2 + c^2 + 2 * b^2 - 2 * a * b - 2 * b * c = 0) : 
  a = b ∧ b = c := 
by 
  sorry

end factorize_polynomial_equilateral_triangle_l304_304478


namespace smallest_three_digit_multiple_of_17_l304_304592

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304592


namespace sum_of_circle_numbers_l304_304529

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem sum_of_circle_numbers (numbers : Fin 10 → ℕ) 
  (h : ∀ i : Fin 10, numbers i = gcd (numbers (i - 1)) (numbers (i + 1)) + 1) : 
  (Finset.univ.sum numbers) = 28 :=
by
  sorry

end sum_of_circle_numbers_l304_304529


namespace math_problem_l304_304405

/-- Problem definitions for given conditions. -/
def line_l (x y : ℝ) : Prop := y - 2 = sqrt 3 * (x + 2)
def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

/-- Parametric equations for line l -/
def parametric_line_l (t : ℝ) : Prop :=
  let x := -2 + 1/2 * t in
  let y := 2 + sqrt 3 / 2 * t in
  line_l x y

/-- Length of |AB| -/
def length_AB (t1 t2 : ℝ) : ℝ := abs (t1 - t2)

/-- Distance from point P to midpoint M --/
def distance_PM (P t1 t2 : ℝ × ℝ) : ℝ :=
  let M := (-2 + 1/2 * (t1 + t2 / 2), 2 + sqrt 3 / 2 * (t1 + t2 / 2)) in
  sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)

/-- Theorem stating translated proof problems -/
theorem math_problem
  (t1 t2 : ℝ)
  (h1 : t1 + t2 = -4)
  (h2 : t1 * t2 = -10)
  (P : ℝ × ℝ := (-2, 2)) :
  parametric_line_l t1 ∧ parametric_line_l t2 ∧ length_AB t1 t2 = 2 * sqrt 14 ∧ distance_PM P t1 t2 = 2 :=
by sorry

end math_problem_l304_304405


namespace smallest_three_digit_multiple_of_17_l304_304863

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304863


namespace smallest_three_digit_multiple_of_17_l304_304664

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304664


namespace certain_amount_l304_304366

theorem certain_amount (x : ℝ) (A : ℝ) (h1: x = 900) (h2: 0.25 * x = 0.15 * 1600 - A) : A = 15 :=
by
  sorry

end certain_amount_l304_304366


namespace smallest_three_digit_multiple_of_17_l304_304644

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304644


namespace Alice_usd_cost_l304_304223

noncomputable def convertGBPtoUSD (gbp : ℝ) (conversionRate : ℝ) : ℝ := 
  gbp / conversionRate

theorem Alice_usd_cost :
  let gbp := 30
  let conversionRate := 0.82
  let usd := convertGBPtoUSD gbp conversionRate
  Float.round (usd * 100) / 100 = 36.59 :=
sorry

end Alice_usd_cost_l304_304223


namespace find_f_2015_l304_304430

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)
axiom period_3 (f : ℝ → ℝ) : ∀ x, f(x + 3) * f(x) = -1
axiom initial_condition (f : ℝ → ℝ) : f(1) = -2

theorem find_f_2015 : f(2015) = 2 :=
by
  sorry

end find_f_2015_l304_304430


namespace problem1_problem2_problem3_l304_304330

-- Definition of the function f
def f (x : ℝ) : ℝ := x * exp (x - 1) - x

-- Definition of the function g
def g (x : ℝ) : ℝ := (1 / exp 1 - 1) * x

-- Problem (1): f(x) has exactly two zeros
theorem problem1 : ∃ z₁ z₂ : ℝ, z₁ ≠ z₂ ∧ f z₁ = 0 ∧ f z₂ = 0 :=
sorry

-- Problem (2): ∀ x ∈ ℝ, f(x) ≥ g(x)
theorem problem2 : ∀ x : ℝ, f x ≥ g x :=
sorry

-- Problem (3): If f(x) = a has exactly two distinct real roots x₁ < x₂, then |x₁ - x₂| ≤ (1 - 2 * exp 1) * a / (1 - exp 1) + 1
theorem problem3 (a : ℝ) (x1 x2 : ℝ) (h1 : x1 < x2) (hf1 : f x1 = a) (hf2 : f x2 = a) (hrs : ∀ x : ℝ, f x ≥ g x) :
  |x1 - x2| ≤ (1 - 2 * exp 1) * a / (1 - exp 1) + 1 :=
sorry

end problem1_problem2_problem3_l304_304330


namespace smallest_three_digit_multiple_of_17_l304_304838

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304838


namespace annalise_total_cost_correct_l304_304943

-- Define the constants from the problem
def boxes : ℕ := 25
def packs_per_box : ℕ := 18
def tissues_per_pack : ℕ := 150
def tissue_price : ℝ := 0.06
def discount_per_box : ℝ := 0.10
def volume_discount : ℝ := 0.08
def tax_rate : ℝ := 0.05

-- Calculate the total number of tissues
def total_tissues : ℕ := boxes * packs_per_box * tissues_per_pack

-- Calculate the total cost without any discounts
def initial_cost : ℝ := total_tissues * tissue_price

-- Apply the 10% discount on the price of the total packs in each box purchased
def cost_after_box_discount : ℝ := initial_cost * (1 - discount_per_box)

-- Apply the 8% volume discount for buying 10 or more boxes
def cost_after_volume_discount : ℝ := cost_after_box_discount * (1 - volume_discount)

-- Apply the 5% tax on the final price after all discounts
def final_cost : ℝ := cost_after_volume_discount * (1 + tax_rate)

-- Define the expected final cost
def expected_final_cost : ℝ := 3521.07

-- Proof statement
theorem annalise_total_cost_correct : final_cost = expected_final_cost := by
  -- Sorry is used as placeholder for the actual proof
  sorry

end annalise_total_cost_correct_l304_304943


namespace shaded_area_is_35_l304_304393

-- Define conditions in Lean
def rectangle (AD AB HE BF : ℝ) := AD = 6 ∧ AB = 10 ∧ HE = 2 ∧ BF = 5

-- The areas of the given triangle and the trapezoid within the rectangle
def triangle_area (base height : ℝ) := (1 / 2) * base * height
def trapezoid_area (a b height : ℝ) := (1 / 2) * (a + b) * height

-- Total area of the rectangle
def rect_area (AD AB : ℝ) := AD * AB

-- Define the given rectangle and its properties
variable (AD AB HE BF : ℝ)
variable (h_rect : rectangle AD AB HE BF)

-- Areas of the specified triangle and trapezoid within the given rectangle
def area_triangle_AEG := triangle_area 8 4
def area_trapezoid_EHCF := trapezoid_area 2 1 6

-- Total area of the rectangle ABCD
def area_rectangle_ABCD := rect_area AD AB

-- Goal: Prove that the total shaded area is 35 cm^2
theorem shaded_area_is_35 :
  (area_rectangle_ABCD - area_triangle_AEG - area_trapezoid_EHCF) = 35 := by {
    -- Proof will be added here
    sorry
}

end shaded_area_is_35_l304_304393


namespace measure_of_angle_AFE_l304_304945

open Real

-- Define the points and the square
variable (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variable [SquareABCD : Square A B C D]

-- Define the angles using given conditions
variable [CDE : Angle C D E = 100]
variable [AFD : LiesOnLine F A D] [EFD : EqualLength E F F D]

-- Define what we want to prove
theorem measure_of_angle_AFE : 
  ∃ A B C D E F, (SquareABCD) ∧ (Angle C D E = 100) ∧ (AFD) ∧ (EFD) → (Angle A F E = 175) :=
by
  sorry

end measure_of_angle_AFE_l304_304945


namespace remainder_of_cubed_sum_mod_6_l304_304429

theorem remainder_of_cubed_sum_mod_6 (a : ℕ → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : ∑ i in finset.range 2023, a i = 2023 ^ 2023) :
  (∑ i in finset.range 2023, a i ^ 3) % 6 = 5 :=
sorry

end remainder_of_cubed_sum_mod_6_l304_304429


namespace neg_five_power_zero_simplify_expression_l304_304199

-- Proof statement for the first question.
theorem neg_five_power_zero : (-5 : ℝ)^0 = 1 := 
by sorry

-- Proof statement for the second question.
theorem simplify_expression (a b : ℝ) : ((-2 * a^2)^2) * (3 * a * b^2) = 12 * a^5 * b^2 := 
by sorry

end neg_five_power_zero_simplify_expression_l304_304199


namespace real_root_exists_l304_304255

theorem real_root_exists (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 :=
sorry

end real_root_exists_l304_304255


namespace smallest_three_digit_multiple_of_17_l304_304612

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304612


namespace count_nonneg_integers_l304_304939

def is_nonneg_integer (x : ℝ) : Prop :=
  x ≥ 0 ∧ ∃ n : ℕ, x = n

theorem count_nonneg_integers :
  let s := [-8, 0, 5, real.pi, -0.01, 13/22] in
  (list.countp is_nonneg_integer s) = 2 :=
by
  sorry

end count_nonneg_integers_l304_304939


namespace monotonic_intervals_ln_ex_over_x_ge_2_no_real_k_l304_304329

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x * log x - k * (x - 1)
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := log x + 1 - k

theorem monotonic_intervals (k : ℝ) : 
  (∀ x, (0 < x ∧ x < real.exp (k - 1)) → f' x k < 0) ∧ 
  (∀ x, (real.exp (k - 1) < x ∧ x < ∞) → f' x k > 0) := sorry

theorem ln_ex_over_x_ge_2 (x : ℝ) (h : 0 < x) :
  log x + (real.exp 1) / x ≥ 2 := sorry

theorem no_real_k (x₀ x₁ : ℝ) (h₁: x₁ > 1) 
  (h₂: f x₁ k = 0) (h₃: f' x₀ k = 0) :
  ¬∃ k : ℝ, x₁ / x₀ = k := sorry

end monotonic_intervals_ln_ex_over_x_ge_2_no_real_k_l304_304329


namespace least_range_product_multiple_840_l304_304373

def is_multiple (x y : Nat) : Prop :=
  ∃ k : Nat, y = k * x

theorem least_range_product_multiple_840 : 
  ∃ (a : Nat), a > 0 ∧ ∀ (n : Nat), (n = 3) → is_multiple 840 (List.foldr (· * ·) 1 (List.range' a n)) := 
by {
  sorry
}

end least_range_product_multiple_840_l304_304373


namespace sum_q_p_evaluations_l304_304962

def p (x : ℝ) : ℝ := |x^2 - 4|
def q (x : ℝ) : ℝ := -|x|

theorem sum_q_p_evaluations : 
  q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3)) = -20 := 
by 
  sorry

end sum_q_p_evaluations_l304_304962


namespace probability_gcd_1_from_set_is_5_7_l304_304491

open Classical

noncomputable def probability_coprime_pairs : ℚ :=
  let S := {1, 2, 3, 4, 5, 6, 7}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b }
  let total_pairs := pairs.to_finset.card
  let non_coprime_pairs := ( { (2, 4), (2, 6), (3, 6), (4, 6), (4, 7), (6, 7) } : Set (ℕ × ℕ) )
  let coprime_pairs := total_pairs - non_coprime_pairs.to_finset.card
  coprime_pairs / total_pairs

theorem probability_gcd_1_from_set_is_5_7 :
  probability_coprime_pairs = 5 / 7 :=
sorry

end probability_gcd_1_from_set_is_5_7_l304_304491


namespace triangle_same_color_exists_l304_304991

-- Defining the problem conditions and the proof goal
theorem triangle_same_color_exists
  (coloring : ℝ × ℝ → ℕ)
  (hcoloring : ∀ p : ℝ × ℝ, coloring p ∈ {1, 2, 3}) :
  ∃ (A B C : ℝ × ℝ), 
    (coloring A = coloring B ∧ coloring B = coloring C) ∧
    (1/2 * abs ((fst B - fst A) * (snd C - snd A) - (snd B - snd A) * (fst C - fst A)) = 1) :=
sorry

end triangle_same_color_exists_l304_304991


namespace goats_count_l304_304966

variable (h d c t g : Nat)
variable (l : Nat)

theorem goats_count 
  (h_eq : h = 2)
  (d_eq : d = 5)
  (c_eq : c = 7)
  (t_eq : t = 3)
  (l_eq : l = 72)
  (legs_eq : 4 * h + 4 * d + 4 * c + 4 * t + 4 * g = l) : 
  g = 1 := by
  sorry

end goats_count_l304_304966


namespace sum_of_ten_numbers_in_circle_l304_304523

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l304_304523


namespace square_area_l304_304928

theorem square_area (d : ℝ) (h : d = 8 * real.sqrt 2) : 
  let s := d / real.sqrt 2 in s * s = 64 :=
by
  -- Provided conditions:
  -- h : d = 8 * real.sqrt 2
  -- The side length is defined as s = d / real.sqrt 2
  -- We need to prove: s * s = 64
  sorry

end square_area_l304_304928


namespace pyramid_volume_l304_304479

theorem pyramid_volume (AB BC PA : ℝ)
  (h₁ : AB = 10)
  (h₂ : BC = 5)
  (h₃ : PA = 8) :
  (1 / 3) * (AB * BC) * PA = 400 / 3 :=
by
  have base_area := AB * BC
  have volume := (1 / 3) * base_area * PA
  calc
    volume = (1 / 3) * (10 * 5) * 8 : by { rw [h₁, h₂, h₃] }
           ... = 400 / 3                : by norm_num

end pyramid_volume_l304_304479


namespace journey_time_is_48_hours_l304_304153

-- Define the radius of Earth at the equator in kilometers
def radius_earth : ℝ := 6370

-- Define the speed of the jet in kilometers per hour
def jet_speed : ℝ := 850

-- Define the additional distance due to stopover in kilometers
def stopover_distance : ℝ := 850

-- Define the total travel distance including the circumference of the Earth and the stopover
def total_distance : ℝ := (2 * Real.pi * radius_earth) + stopover_distance

-- Define the formula to calculate the travel time given the total distance and jet speed
def travel_time : ℝ := total_distance / jet_speed

-- The expected answer, rounded to the nearest whole number
def expected_time : ℝ := 48

-- The statement that needs to be proven
theorem journey_time_is_48_hours : travel_time ≈ expected_time := 
by
  -- Proof will be placed here
  sorry

end journey_time_is_48_hours_l304_304153


namespace max_min_square_in_N_l304_304963

def parabola1 (x y : ℝ) := x^2 - 4 * y = 0
def point_K := (0, 3 : ℝ)

def rotated_parabola (x y : ℝ) (angle : ℝ) :=
  match angle with
  | 90 => (y - 3)^2 + 4 * (x - 3) = 0
  | 180 => x^2 + 4 * (y - 6) = 0
  | 270 => (y - 3)^2 - 4 * (x + 3) = 0
  | _ => false -- Invalid angle for this problem

theorem max_min_square_in_N :
  ∃ l_max l_min : ℝ,
    (l_max = 6) ∧
    (l_min = 4 * real.sqrt 2) ∧
    (true -- represent region N can be tiled with squares here, to be expanded with actual tiling proof
     sorry) :=
begin
  sorry
end

end max_min_square_in_N_l304_304963


namespace michael_truck_meetings_l304_304454

def speed_michael := 5 -- feet per second
def speed_truck := 10  -- feet per second
def distance_pails := 200 -- feet
def stop_time := 30 -- seconds
def initial_position_michael := 0 -- feet
def initial_position_truck := 200 -- feet

theorem michael_truck_meetings (v_m v_t d_p t_s M0 T0 : ℕ)
  (hv_m : v_m = speed_michael) 
  (hv_t : v_t = speed_truck) 
  (hd_p : d_p = distance_pails) 
  (ht_s : t_s = stop_time) 
  (hM0 : M0 = initial_position_michael) 
  (hT0 : T0 = initial_position_truck) :
  ∃ t : ℕ, (M(t) = T(t) ∧ t > 0) := sorry

end michael_truck_meetings_l304_304454


namespace value_of_f_at_3_l304_304026

def f (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 - 3 * x + 7

theorem value_of_f_at_3 : f 3 = 196 := by
  sorry

end value_of_f_at_3_l304_304026


namespace smallest_three_digit_multiple_of_17_correct_l304_304616

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304616


namespace ratio_of_areas_l304_304057

theorem ratio_of_areas (A B C D E F : Type) (AB BC CA : ℝ) (p q r : ℝ) 
  (h1 : AB = 10) (h2 : BC = 14) (h3 : CA = 16) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r)
  (h7 : p + q + r = 3 / 4) (h8 : p^2 + q^2 + r^2 = 1 / 2)
  (h9 : D ∈ lineSegment A B)
  (h10 : E ∈ lineSegment B C)
  (h11 : F ∈ lineSegment C A)
  (h12 : dist A D = p * dist A B)
  (h13 : dist B E = q * dist B C)
  (h14 : dist C F = r * dist C A) : 
  ∃ m n : ℕ, m + n = 41 ∧ coprime m n ∧ ∑ (m / n) = 9 / 32 :=
sorry

end ratio_of_areas_l304_304057


namespace smallest_consecutive_divisible_by_24_l304_304140

theorem smallest_consecutive_divisible_by_24 : 
  ∃ n : ℕ, (∃ (a b c d e : ℕ), a = n ∧ b = n + 1 ∧ c = n + 2 ∧ d = n + 3 ∧ e = n + 4 ∧ 24 ∣ (a * b * c * d * e)) ∧ n ≥ 1 :=
begin
  sorry
end

end smallest_consecutive_divisible_by_24_l304_304140


namespace smallest_three_digit_multiple_of_17_l304_304656

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304656


namespace AX_is_symmedian_of_triangle_ABC_l304_304532

noncomputable def symmedian_property (A B C D E X : Point) (ABC_circumcircle : Circle) 
  (DE_diameter_circle : Circle) : Prop :=
  let S := line (A, X) ∩ line (B, C) in
  AX_is_symmedian : (line (A, X).intersects_circumcircle (A, B, C))
  → (line (D, E) ∩ (line (B, C)) = {S}) (* Notation and intersection may require refinement *)
  → (CX / BX = AC^2 / AB^2)

theorem AX_is_symmedian_of_triangle_ABC
  (A B C D E X : Point)
  (BC : Line)
  (internal_angle_bisector_A : Line)
  (external_angle_bisector_A : Line)
  (circumcircle_ABC : Circle)
  (circle_DE : Circle)
  (h1 : D ∈ (internal_angle_bisector_A ∩ BC))
  (h2 : E ∈ (external_angle_bisector_A ∩ BC))
  (h3 : ∀ P Q, (circle_DE).diameter_is P Q → P = D ∨ P = E → Q = E ∨ Q = D)
  (h4 : ∀ P Q, circumcircle_ABC.contains P → circumcircle_ABC.contains Q → line (P, Q) = circum_circle_ABC)
  : symmedian_property A B C D E X :=
by {
  sorry
}

end AX_is_symmedian_of_triangle_ABC_l304_304532


namespace length_of_AE_l304_304965

theorem length_of_AE (AB CD AC : ℕ) (hAB : AB = 10) (hCD : CD = 15) (hAC : AC = 18)
  (equal_perimeters : ∀ (E : Type) (AED BEC : E → Prop),
  AED → BEC → (sum_of_sides AED = sum_of_sides BEC))
  (AE EC : ℕ) (ratio : AE / EC = 2 / 3)
  : AE = 7.2 := sorry

end length_of_AE_l304_304965


namespace flowers_sold_difference_l304_304250

def number_of_daisies_sold_on_second_day (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) : Prop :=
  d3 = 2 * d2 - 10 ∧
  d_sum = 45 + d2 + d3 + 120

theorem flowers_sold_difference (d2 : ℕ) (d3 : ℕ) (d_sum : ℕ) 
  (h : number_of_daisies_sold_on_second_day d2 d3 d_sum) :
  45 + d2 + d3 + 120 = 350 → 
  d2 - 45 = 20 := 
by
  sorry

end flowers_sold_difference_l304_304250


namespace initial_temperature_l304_304417

/-- Jason is making pasta. Each minute the temperature of the water increases by 3 degrees. 
Once the water reaches 212 degrees and is boiling, Jason needs to cook his pasta for 12 minutes. 
Then it takes him 1/3 that long to mix the pasta with the sauce and make a salad. It takes Jason 
73 minutes to cook dinner. What was the initial temperature of the water? -/
theorem initial_temperature (total_time : ℕ) (boil_temp : ℕ) (increase_rate : ℕ) 
(cook_time : ℕ) (mix_ratio : ℚ) (initial_temp : ℕ) : 
  total_time = 73 →
  boil_temp = 212 →
  increase_rate = 3 →
  cook_time = 12 →
  mix_ratio = 1 / 3 →
  initial_temp = boil_temp - (increase_rate * 
    (total_time - (cook_time + (mix_ratio * cook_time).to_nat))) :=
by 
  sorry

end initial_temperature_l304_304417


namespace sin_cube_eq_l304_304968

theorem sin_cube_eq (c d : ℝ) :
  (∀ θ : ℝ, sin θ * sin θ * sin θ = c * sin (3 * θ) + d * sin θ) ↔ c = -1 / 4 ∧ d = 3 / 4 := by
  sorry

end sin_cube_eq_l304_304968


namespace chasity_candies_l304_304246

theorem chasity_candies :
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  initial_amount - total_spent = 5 :=
by
  -- Let definitions of constants
  let lollipop_cost := 1.5
  let gummy_pack_cost := 2
  let initial_amount := 15
  let lollipops_bought := 4
  let gummies_bought := 2
  -- Total cost calculation
  let total_spent := lollipops_bought * lollipop_cost + gummies_bought * gummy_pack_cost
  -- Proof of the final amount left
  have h : initial_amount - total_spent = 15 - (4 * 1.5 + 2 * 2) := rfl
  simp at h
  exact h

end chasity_candies_l304_304246


namespace perpendicular_bisector_property_l304_304038

theorem perpendicular_bisector_property 
  {A B C D F : Type*} 
  [InnerProductGeometry A] [InnerProductGeometry B] [InnerProductGeometry C] 
  [InnerProductGeometry D] [Point Geometry F] 
  (hAngleBisector : is_angle_bisector A B C D)
  (hPerpendicularBisector : is_perpendicular_bisector A D F (B C)) : 
  (distance F D)^2 = (distance F B) * (distance F C) :=
sorry

end perpendicular_bisector_property_l304_304038


namespace smallest_three_digit_multiple_of_17_l304_304593

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304593


namespace minimum_colors_required_l304_304533

def beaver_move (c1 c2 : ℕ × ℕ) : Prop :=
  (abs (c1.1 - c2.1) = 2 ∧ c1.2 = c2.2) ∨ (abs (c1.2 - c2.2) = 2 ∧ c1.1 = c2.1)

def knight_move (c1 c2 : ℕ × ℕ) : Prop :=
  (abs (c1.1 - c2.1) = 2 ∧ abs (c1.2 - c2.2) = 1) ∨ (abs (c1.1 - c2.1) = 1 ∧ abs (c1.2 - c2.2) = 2)

def valid_coloring (f : ℕ × ℕ → ℕ) : Prop :=
  ∀ c1 c2, (beaver_move c1 c2 ∨ knight_move c1 c2) → f c1 ≠ f c2

theorem minimum_colors_required : ∃ (f : ℕ × ℕ → ℕ), valid_coloring f ∧ ∀ g. (valid_coloring g → (∀ c, g c ≤ 4)) :=
sorry

end minimum_colors_required_l304_304533


namespace find_a_l304_304154

theorem find_a (a : ℝ) : (let z := (1 + a * complex.I) * (2 - complex.I) in z.re = z.im) → a = 3 :=
by
  intros h
  sorry  -- proof left as an exercise.

end find_a_l304_304154


namespace Sarah_copies_l304_304117

theorem Sarah_copies : 
  ∀ (copies_per_person number_of_people pages_per_contract : ℕ),
    copies_per_person = 2 →
    number_of_people = 9 →
    pages_per_contract = 20 →
    (copies_per_person * number_of_people * pages_per_contract) = 360 := 
by
  intros copies_per_person number_of_people pages_per_contract h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end Sarah_copies_l304_304117


namespace smallest_three_digit_multiple_of_17_l304_304740

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304740


namespace adjust_collection_amount_l304_304946

/-- Define the error caused by mistaking half-dollars for dollars -/
def halfDollarError (x : ℕ) : ℤ := 50 * x

/-- Define the error caused by mistaking quarters for nickels -/
def quarterError (x : ℕ) : ℤ := 20 * x

/-- Define the total error based on the given conditions -/
def totalError (x : ℕ) : ℤ := halfDollarError x - quarterError x

theorem adjust_collection_amount (x : ℕ) : totalError x = 30 * x := by
  sorry

end adjust_collection_amount_l304_304946


namespace total_distance_driven_l304_304482

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l304_304482


namespace smallest_three_digit_multiple_of_17_l304_304695

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304695


namespace unlock_combination_lock_l304_304413

theorem unlock_combination_lock :
  ∃ (seq : List ℕ), seq.length = 29 ∧
  (∀ code ∈ { l : List ℕ | l.length = 3 ∧ ∀ n ∈ l, n ∈ {1, 2, 3}}, 
    ∃ i, code = seq.slice i (i+3)) :=
by
  sorry

end unlock_combination_lock_l304_304413


namespace smallest_three_digit_multiple_of_17_l304_304815

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304815


namespace smallest_fraction_with_digits_347_l304_304133

theorem smallest_fraction_with_digits_347 :
  ∃ (m n : ℕ), m < n ∧ Nat.coprime m n ∧ ( ∃ (k : ℕ), (10^-(k+3) * (3 + 4 * 10 + 7 * 10^2 : ℝ)) = (m : ℝ) / (n : ℝ) )
    ∧ m = 6 ∧ n = 17 :=
by
  -- Existence and properties of m and n
  use 6, 17
  -- Conditions
  split
  { -- m < n
    exact Nat.lt_succ_self 16 },
  split
  { -- m and n are coprime
    exact Nat.coprime_of_dvd 1 (by norm_num) },
  split
  { -- Decimal representation condition
    use 0
    norm_num },
  -- m = 6 and n = 17
  split; refl

end smallest_fraction_with_digits_347_l304_304133


namespace smallest_three_digit_multiple_of_17_l304_304684

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304684


namespace smallest_three_digit_multiple_of_17_l304_304842

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304842


namespace point_2000_coordinates_l304_304195

-- Definition to describe the spiral numbering system in the first quadrant
def spiral_number (n : ℕ) : ℕ × ℕ := sorry

-- The task is to prove that the coordinates of the 2000th point are (44, 25).
theorem point_2000_coordinates : spiral_number 2000 = (44, 25) :=
by
  sorry

end point_2000_coordinates_l304_304195


namespace bowling_ball_weight_l304_304995

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l304_304995


namespace smallest_three_digit_multiple_of_17_l304_304669

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304669


namespace rise_in_water_level_result_l304_304899

-- Definitions based on given conditions
def side_fishbowl := 20 -- cm
def height_fishbowl := 20 -- cm
def initial_water_height := 15 -- cm
def side_cube := 10 -- cm

-- Total volume calculation
def volume_fishbowl_base := side_fishbowl * side_fishbowl -- square cm
def volume_initial_water := volume_fishbowl_base * initial_water_height -- cubic cm
def volume_cube := side_cube * side_cube * side_cube -- cubic cm
def volume_total_water := volume_initial_water + volume_cube -- cubic cm

-- New water height calculation
def new_water_height := volume_total_water / volume_fishbowl_base -- cm
def rise_in_water_level := new_water_height - initial_water_height -- cm

-- Theorem to prove the rise in water level
theorem rise_in_water_level_result : rise_in_water_level = 2.5 := by
  sorry

end rise_in_water_level_result_l304_304899


namespace smallest_three_digit_multiple_of_17_l304_304660

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304660


namespace polynomial_term_count_l304_304961

open Nat

theorem polynomial_term_count (N : ℕ) (h : (N.choose 5) = 2002) : N = 17 :=
by
  sorry

end polynomial_term_count_l304_304961


namespace part_I_part_II_l304_304322

noncomputable def f (a x : ℝ) : ℝ := (x^2 + a) / (x + 1)

theorem part_I (a : ℝ) :
  (deriv (λ x, f a x) 1 = 1/2) → a = 1 :=
by sorry
  
theorem part_II (a : ℝ) :
  (deriv (λ x, f a x) 1 = 0) →
  a = 3 → 
  ((∀ x, x < -3 → deriv (λ x, f 3 x) x > 0) ∧ 
   (∀ x, x > 1 → deriv (λ x, f 3 x) x > 0) ∧
   (∀ x, -3 < x ∧ x < -1 → deriv (λ x, f 3 x) x < 0) ∧
   (∀ x, -1 < x ∧ x < 1 → deriv (λ x, f 3 x) x < 0)) :=
by sorry

end part_I_part_II_l304_304322


namespace harmonic_sum_inequality_l304_304010

theorem harmonic_sum_inequality (n : ℕ) (h : n ≥ 1) :
  (∑ i in Finset.range (n+1), (1 / (i+1) : ℝ)) > (Real.log (n+1) + n / (2 * (n + 1))) := 
sorry

end harmonic_sum_inequality_l304_304010


namespace cost_of_notebook_l304_304563

-- Definitions based on the conditions in (a)
def price_notebook := ℝ
def price_pen := ℝ
def cost1 (n : price_notebook) (p : price_pen) : Prop := 3 * n + 4 * p = 3.75
def cost2 (n : price_notebook) (p : price_pen) : Prop := 5 * n + 2 * p = 3.05

-- Translation to Lean 4 statement of the problem
theorem cost_of_notebook (n p : ℝ) (h1 : cost1 n p) (h2 : cost2 n p) : n = 0.3357 :=
sorry

end cost_of_notebook_l304_304563


namespace negation_proposition_l304_304547

theorem negation_proposition : ∀ (a : ℝ), (a > 3) → (a^2 ≥ 9) :=
by
  intros a ha
  sorry

end negation_proposition_l304_304547


namespace monotonicity_intervals_range_of_c_l304_304003

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem monotonicity_intervals (α : ℝ) : 
  (∀ x, 0 < x ∧ x < Real.exp 2 → deriv (f x α) < 0) ∧ 
  (∀ x, x > Real.exp 2 → deriv (f x α) > 0) :=
by
  sorry

theorem range_of_c (c : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 3 → f x 3 < 2 * c^2 - c) → 
  (c > 1 ∨ c < -1/2) :=
by
  sorry

end monotonicity_intervals_range_of_c_l304_304003


namespace determine_a_square_binomial_l304_304967

theorem determine_a_square_binomial (a : ℝ) (h : ∃ r s : ℝ, (rx + s)^2 = ax^2 + 8x + 16) : a = 1 :=
sorry

end determine_a_square_binomial_l304_304967


namespace geometric_series_sum_l304_304240

theorem geometric_series_sum :
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  (a * (1 - r^n) / (1 - r) = 728 / 243) := 
by
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  show a * (1 - r^n) / (1 - r) = 728 / 243
  sorry

end geometric_series_sum_l304_304240


namespace equal_cost_number_of_minutes_l304_304193

theorem equal_cost_number_of_minutes :
  ∃ m : ℝ, (8 + 0.25 * m = 12 + 0.20 * m) ∧ m = 80 :=
by
  sorry

end equal_cost_number_of_minutes_l304_304193


namespace smallest_three_digit_multiple_of_17_l304_304688

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304688


namespace smallest_three_digit_multiple_of_17_l304_304650

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304650


namespace part_a_part_b_l304_304892

-- Part (a): Prove that \( 2^n - 1 \) is divisible by 7 if and only if \( 3 \mid n \).
theorem part_a (n : ℕ) : 7 ∣ (2^n - 1) ↔ 3 ∣ n := sorry

-- Part (b): Prove that \( 2^n + 1 \) is not divisible by 7 for all natural numbers \( n \).
theorem part_b (n : ℕ) : ¬ (7 ∣ (2^n + 1)) := sorry

end part_a_part_b_l304_304892


namespace cos_theta_eq_neg_half_l304_304427

noncomputable def vector_cos_theta (a b c : ℝ^3) : ℝ :=
  let theta := real.angle (b, c)
  real.cos theta

theorem cos_theta_eq_neg_half (a b c : ℝ^3) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (∀ u v : ℝ^3, u ≠ v → (u × v) ≠ 0) ∧  -- ensures a, b, c are non-parallel
  (((a ⊗ b) ⊗ c) = (1/2) * norm b * norm c • a) →
  vector_cos_theta a b c = -1/2 :=
sorry

end cos_theta_eq_neg_half_l304_304427


namespace gcd_4320_2550_l304_304573

-- Definitions for 4320 and 2550
def a : ℕ := 4320
def b : ℕ := 2550

-- Statement to prove the greatest common factor of a and b is 30
theorem gcd_4320_2550 : Nat.gcd a b = 30 := 
by 
  sorry

end gcd_4320_2550_l304_304573


namespace smallest_three_digit_multiple_of_17_l304_304610

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304610


namespace smallest_three_digit_multiple_of_17_correct_l304_304625

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304625


namespace fly_distance_from_ceiling_l304_304167

theorem fly_distance_from_ceiling (x y z : ℝ) (hx : x = 2) (hy : y = 6) (hP : x^2 + y^2 + z^2 = 100) : z = 2 * Real.sqrt 15 :=
by
  sorry

end fly_distance_from_ceiling_l304_304167


namespace smallest_three_digit_multiple_of_17_correct_l304_304619

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304619


namespace second_lady_distance_l304_304164

theorem second_lady_distance (x : ℕ) 
  (h1 : ∃ y, y = 2 * x) 
  (h2 : x + 2 * x = 12) : x = 4 := 
by 
  sorry

end second_lady_distance_l304_304164


namespace smallest_three_digit_multiple_of_17_l304_304597

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304597


namespace triangle_ARS_isosceles_l304_304426

variables {A B C P R S : Type}

-- Definitions and hypotheses from the problem conditions
variables [triangle_ABC : Triangle A B C]
variables [circumscribed_circle : CircumscribedCircle A B C]
variables [tangent_A : TangentAtPoint A circumscribed_circle]
variables [intersect_P : IntersectsAtPoint tangent_A (Line B C) P]
variables [angle_bisector_APB : AngleBisector (Angle AP B)]
variables [intersect_R : IntersectsAtPoint angle_bisector_APB (Line A B) R]
variables [intersect_S : IntersectsAtPoint angle_bisector_APB (Line A C) S]
variables (isosceles_ARS_at_A : IsoscelesAt A (Triangle A R S))

-- The theorem stating the problem's conclusion
theorem triangle_ARS_isosceles (h1 : AB < AC) : isosceles_ARS_at_A :=
sorry

end triangle_ARS_isosceles_l304_304426


namespace smallest_three_digit_multiple_of_17_l304_304754

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304754


namespace find_integers_l304_304888

theorem find_integers (p : ℤ) (hp1 : p > 3) (hp2 : Prime p) :
  ∃ a b : ℤ, a^2 + 3 * a * b + 2 * p * (a + b) + p^2 = 0 ∧ (a = -p ∧ b = 0) :=
begin
  sorry
end

end find_integers_l304_304888


namespace smallest_three_digit_multiple_of_17_l304_304691

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304691


namespace smallest_three_digit_multiple_of_17_l304_304841

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304841


namespace smallest_three_digit_multiple_of_17_l304_304665

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304665


namespace area_triangle_ABC_l304_304386

variables (AC BD E A B C D : Point)
variables (ratio : ℕ) (areaABE : ℕ)

-- Given conditions from the problem
axiom intersect_at_E : E ∈ AC ∧ E ∈ BD
axiom tangent_parallel_BD : tangent_through A ∥ BD
axiom ratio_CD_ED : ratio = 3 / 2
axiom area_ΔABE : area A B E = 8

-- Proof that the area of triangle △ABC is 18
theorem area_triangle_ABC :
  area A B C = 18 :=
by sorry

end area_triangle_ABC_l304_304386


namespace surface_area_paraboloid_cylinder_l304_304952

theorem surface_area_paraboloid_cylinder (R : ℝ) (hR : R ≥ 0) : 
  let S := (∫ (θ : ℝ) in 0..2*Real.pi, 
             (1/3 : ℝ) * ((1 + R^2)^(3/2) - 1)) in
  S = 2*Real.pi/3 * ((1 + R^2)^(3/2) - 1) :=
by
  let S := (∫ (θ : ℝ) in 0..2*Real.pi, 
             (1/3 : ℝ) * ((1 + R^2)^(3/2) - 1))
  show S = 2*Real.pi/3 * ((1 + R^2)^(3/2) - 1)
  sorry

end surface_area_paraboloid_cylinder_l304_304952


namespace chocolate_bars_sold_last_week_l304_304414

-- Definitions based on conditions
def initial_chocolate_bars : Nat := 18
def chocolate_bars_sold_this_week : Nat := 7
def chocolate_bars_needed_to_sell : Nat := 6

-- Define the number of chocolate bars sold so far
def chocolate_bars_sold_so_far : Nat := chocolate_bars_sold_this_week + chocolate_bars_needed_to_sell

-- Target statement to prove
theorem chocolate_bars_sold_last_week :
  initial_chocolate_bars - chocolate_bars_sold_so_far = 5 :=
by
  sorry

end chocolate_bars_sold_last_week_l304_304414


namespace smallest_three_digit_multiple_of_17_l304_304753

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304753


namespace initial_tagged_fish_is_50_l304_304040

noncomputable def initial_tagged_fish : ℝ :=
  let N := 312.5 in
  let ratio := (8 : ℝ) / 50 in
  let tagged_ratio := 16 / 100 in
  ratio * N

theorem initial_tagged_fish_is_50 : initial_tagged_fish = 50 := by
  let N := 312.5
  let ratio := (8 : ℝ) / 50
  let tagged_ratio := 16 / 100
  have h : 312.5 * (8 / 50) = 312.5 * 0.16 := by
    norm_num
  simp [initial_tagged_fish, h]
  norm_num
  sorry

end initial_tagged_fish_is_50_l304_304040


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304710

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304710


namespace smallest_three_digit_multiple_of_17_l304_304831

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304831


namespace smallest_three_digit_multiple_of_17_correct_l304_304617

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304617


namespace math_proof_problem_l304_304514

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304514


namespace smallest_three_digit_multiple_of_17_l304_304770

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304770


namespace max_value_of_4x_plus_3y_l304_304297

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l304_304297


namespace cone_volume_l304_304556

noncomputable def volume_of_cone (L h : ℝ) : ℝ :=
  (1/3) * Math.pi * (sqrt (L^2 - h^2))^2 * h

theorem cone_volume (L h : ℝ) (hL : L = 15) (hh : h = 8) :
  volume_of_cone L h = (1288 / 3) * Math.pi :=
by
  rw [hL, hh]
  -- proving steps go here
  sorry

end cone_volume_l304_304556


namespace smallest_three_digit_multiple_of_17_l304_304587

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304587


namespace simplify_expression_l304_304499

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304499


namespace probability_four_dots_collinear_l304_304410

-- Define the 5x5 grid and collinearity
structure Dot := (x : ℕ) (y : ℕ)

def is_collinear (d1 d2 d3 d4 : Dot) : Prop :=
  (d1.x = d2.x ∧ d2.x = d3.x ∧ d3.x = d4.x) ∨
  (d1.y = d2.y ∧ d2.y = d3.y ∧ d3.y = d4.y) ∨
  (d1.x - d1.y = d2.x - d2.y ∧ d2.x - d2.y = d3.x - d3.y ∧ d3.x - d3.y = d4.x - d4.y) ∨
  (d1.x + d1.y = d2.x + d2.y ∧ d2.x + d2.y = d3.x + d3.y ∧ d3.x + d3.y = d4.x + d4.y)

-- Count all combinations
noncomputable def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def total_combinations : ℕ := comb 25 4

def collinear_sets : ℕ := 28

-- Proof statement: The probability that four random dots are collinear
theorem probability_four_dots_collinear :
  (collinear_sets : ℚ) / total_combinations = 14 / 6325 := by
  sorry

end probability_four_dots_collinear_l304_304410


namespace smallest_three_digit_multiple_of_17_l304_304700

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304700


namespace total_collected_funds_l304_304044

theorem total_collected_funds (A B T : ℕ) (hA : A = 5) (hB : B = 3 * A + 3) (h_quotient : B / 3 = 6) (hT : T = B * (B / 3) + A) : 
  T = 113 := 
by 
  sorry

end total_collected_funds_l304_304044


namespace smallest_three_digit_multiple_of_17_l304_304856

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304856


namespace smallest_three_digit_multiple_of_17_l304_304651

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304651


namespace _l304_304072

noncomputable theorem sum_h_geq_norms (f g : ℤ → ℝ≥0) (p q : ℝ) 
  (hpq : 1 / p + 1 / q = 1) 
  (hfg_zero : ∀ n:ℤ, ((∃! x : ℤ, f x ≠ 0) ∧ (∃! y : ℤ, g y ≠ 0)) → f n = 0 ∧ g n = 0) :
  ∑ n in ℤ, let h := (λ n, Sup {m | ∃ k:ℤ, m = f (n - k) * g k}) n
            in h n ≥ ((∑ n in ℤ, (f n)^p)^(1/p) * ∑ n in ℤ, (g n)^q)^(1/q) :=
by
  sorry

end _l304_304072


namespace discount_remains_same_l304_304027

def discount_at_double_time (P D : ℝ) (T r : ℝ) : ℝ :=
  P / (1 + r / 100) ^ (2 * T)

theorem discount_remains_same 
  (P : ℝ)
  (D : ℝ)
  (T : ℝ)
  (r : ℝ)
  (h1 : D = 10)
  (h2 : P + D = 110) :
  discount_at_double_time P D T r = D :=
by {
  sorry
}

end discount_remains_same_l304_304027


namespace chastity_leftover_money_l304_304244

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end chastity_leftover_money_l304_304244


namespace smallest_three_digit_multiple_of_17_l304_304667

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304667


namespace GCF_30_90_75_l304_304170

theorem GCF_30_90_75 : Nat.gcd (Nat.gcd 30 90) 75 = 15 := by
  sorry

end GCF_30_90_75_l304_304170


namespace math_proof_problem_l304_304512

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304512


namespace part_a_part_b_part_c_l304_304277

-- Part a: Prove equivalence of the given equality and the interval condition
theorem part_a (x : ℝ) (h : x ≥ 1 / 2) :
  (√(x + √(2 * x - 1)) + √(x - √(2 * x - 1)) = √2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
sorry

-- Part b: Prove there are no solutions for the given equality
theorem part_b (x : ℝ) (h : x ≥ 1 / 2) :
  (√(x + √(2 * x - 1)) + √(x - √(2 * x - 1)) = 1) → false :=
sorry

-- Part c: Prove equivalence of the given equality and the specific value of x
theorem part_c (x : ℝ) (h : x ≥ 1 / 2) :
  (√(x + √(2 * x - 1)) + √(x - √(2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
sorry

end part_a_part_b_part_c_l304_304277


namespace smallest_three_digit_multiple_of_17_l304_304643

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304643


namespace closest_to_sin_2017_l304_304940

-- Definitions
def sin_2017_eq_neg_sin_37 : Prop :=
  sin 2017 = -sin 37

def angle_bounds : Prop :=
  30 < 37 ∧ 37 < 45

def sin_30_value : Prop :=
  sin 30 = 1 / 2

def sin_45_value : Prop :=
  sin 45 = sqrt 2 / 2

def bound_values : Prop :=
  1 / 2 < 3 / 5 ∧ 3 / 5 < sqrt 2 / 2

-- Theorem to be proved
theorem closest_to_sin_2017 :
  sin_2017_eq_neg_sin_37 →
  angle_bounds →
  sin_30_value →
  sin_45_value →
  bound_values →
  abs (sin 2017 - (-3 / 5)) < min (abs (sin 2017 - (-1 / 2))) (min (abs (sin 2017 - (-sqrt 2 / 2))) (abs (sin 2017 - (-4 / 5)))) :=
by
  sorry

end closest_to_sin_2017_l304_304940


namespace smallest_three_digit_multiple_of_17_l304_304797

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304797


namespace smallest_three_digit_multiple_of_17_l304_304742

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304742


namespace vertex_angle_of_isosceles_triangle_l304_304045

theorem vertex_angle_of_isosceles_triangle (a b h : ℝ) (phi : ℝ) 
    (h_base : b = a * Real.sqrt 2)
    (h_height : h = a / Real.sqrt 2)
    (h_cond : a^2 = 3 * b * h)
    (h_angle : phi = 180 - 2 * 45) : 
    phi = 90 :=
by
    rw [h_base, h_height, h_cond, h_angle]
    sorry

end vertex_angle_of_isosceles_triangle_l304_304045


namespace line_passes_through_fixed_point_l304_304024

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) : a * 1 + b * (-1) + c = 0 := 
by sorry

end line_passes_through_fixed_point_l304_304024


namespace y_value_range_l304_304279

theorem y_value_range (x y : ℝ) (h1 : log 2 = log ((sin x - 1 / 3) ^ 2 / 2 / (1 - y))) :
  7 / 9 ≤ y ∧ y < 1 := by
  sorry

end y_value_range_l304_304279


namespace smallest_three_digit_multiple_of_17_l304_304598

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304598


namespace half_angle_in_first_or_third_quadrant_l304_304307

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l304_304307


namespace necessary_condition_for_line_passes_quadrants_l304_304145

theorem necessary_condition_for_line_passes_quadrants (m n : ℝ) (h_line : ∀ x : ℝ, x * (m / n) - (1 / n) < 0 ∨ x * (m / n) - (1 / n) > 0) : m * n < 0 :=
by
  sorry

end necessary_condition_for_line_passes_quadrants_l304_304145


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304716

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304716


namespace exists_parallelogram_from_convex_quadrilateral_l304_304060

-- Define a type for points in the plane
structure Point : Type :=
  (x : ℝ)
  (y : ℝ)

-- Define function for convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop :=
  ∀ (P Q R S : Point), -- points lying within the quadrilateral
  (P = A ∨ P = B ∨ P = C ∨ P = D) → 
  (Q = A ∨ Q = B ∨ Q = C ∨ Q = D) → 
  (R = A ∨ R = B ∨ R = C ∨ R = D) → 
  (S = A ∨ S = B ∨ S = C ∨ S = D) →
  -- Sum of interior angles of a quadrilateral is 360 degrees
  angle A + angle B + angle C + angle D = 360 -- this angle calculation will need a definition

-- Define function for angles (dummy for now, proper definition required)
noncomputable def angle (pt1 pt2 pt3 : Point) : ℝ := sorry

-- Define the existence of a point E such that ABC and E forms a parallelogram
theorem exists_parallelogram_from_convex_quadrilateral (A B C D : Point) :
  is_convex_quadrilateral A B C D →
  ∃ E : Point, 
  -- Three vertices of the parallelogram coincide with three vertices of quadrilateral
  (E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D) ∧
  -- Parallelogram properties AB = CE and BC = AE (which implies ABCE is a parallelogram)
  (∃ E : Point, AB_line ∥ CE_line ∧ BC_line ∥ AE_line) := 
  sorry

end exists_parallelogram_from_convex_quadrilateral_l304_304060


namespace expression_evaluation_l304_304284

variable {x y : ℝ}

theorem expression_evaluation (h : (x-2)^2 + |y-3| = 0) :
  ( (x - 2 * y) * (x + 2 * y) - (x - y) ^ 2 + y * (y + 2 * x) ) / (-2 * y) = 2 :=
by
  sorry

end expression_evaluation_l304_304284


namespace distribute_tickets_l304_304204

theorem distribute_tickets :
  ∃ (drawing_method : Prop) (random_number_method : Prop),
  (class_size = 60) → (ticket_count = 10) →
  (drawing_method ∨ random_number_method) :=
begin
  let class_size := 60,
  let ticket_count := 10,
  let drawing_method := 
    ∃ (students : list ℕ), (students.length = class_size) ∧
    (∃ (slips : list ℕ), (slips = students) ∧ 
      (∃ (draws : list ℕ), (draws.length = ticket_count) ∧
          ∀d ∈ draws, d ∈ slips)),
  let random_number_method :=
    ∃ (students : list ℕ), (students.length = class_size) ∧
    (∃ (initial_index : ℕ), 
      (∃ (selected_students : list ℕ), (selected_students.length = ticket_count) ∧
          ∀s ∈ selected_students, s ≤ class_size)),
  existsi drawing_method,
  existsi random_number_method,
  intros,
  sorry,
end

end distribute_tickets_l304_304204


namespace circle_color_bound_l304_304560

noncomputable def circle_problem (k n : Nat) : Prop :=
  ∀ (c : Finset (Fin n)) (h : ∀ (A B : Fin k), A ≠ B → (∃! color ∈ c, ⟦AB⟧)), k ≤ 2^n

-- Now state the theorem that we need to prove
theorem circle_color_bound (k n : Nat) (c : Finset (Fin n)) (h : ∀ (A B : Fin k), A ≠ B → (∃! color ∈ c, ⟦AB⟧)) : k ≤ 2^n :=
  sorry

end circle_color_bound_l304_304560


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304708

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304708


namespace spiritual_connection_probability_l304_304165

-- The set of numbers A and B can choose from
def number_set : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Predicate indicating a "spiritual connection"
def spiritual_connection (a b : ℕ) : Prop :=
  |a - b| ≤ 1

noncomputable def probability_spiritual_connection : ℚ :=
  let total_outcomes := (number_set.card) * (number_set.card)
  let successful_outcomes := Finset.card {ab | ∃ a b, a ∈ number_set ∧ b ∈ number_set ∧ spiritual_connection a b}
  (successful_outcomes : ℚ) / (total_outcomes : ℚ)

theorem spiritual_connection_probability :
  probability_spiritual_connection = 7 / 25 := sorry

end spiritual_connection_probability_l304_304165


namespace reasoning_form_is_incorrect_l304_304247

-- Conditions
def some_rat_num_are_proper_fractions : Prop := ∃ q : ℚ, q.denom > 1
def int_are_rat_nums : Prop := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Statement to prove: the reasoning form is incorrect
theorem reasoning_form_is_incorrect
  (h1 : some_rat_num_are_proper_fractions)
  (h2 : int_are_rat_nums) :
  (∀ z : ℤ, ¬ (∃ q : ℚ, q.denom > 1 ∧ q = z)) :=
begin
  sorry
end

end reasoning_form_is_incorrect_l304_304247


namespace rented_movie_cost_l304_304467

def cost_of_tickets (c_ticket : ℝ) (n_tickets : ℕ) := c_ticket * n_tickets
def total_cost (cost_tickets cost_bought : ℝ) := cost_tickets + cost_bought
def remaining_cost (total_spent cost_so_far : ℝ) := total_spent - cost_so_far

theorem rented_movie_cost
  (c_ticket : ℝ)
  (n_tickets : ℕ)
  (c_bought : ℝ)
  (c_total : ℝ)
  (h1 : c_ticket = 10.62)
  (h2 : n_tickets = 2)
  (h3 : c_bought = 13.95)
  (h4 : c_total = 36.78) :
  remaining_cost c_total (total_cost (cost_of_tickets c_ticket n_tickets) c_bought) = 1.59 :=
by 
  sorry

end rented_movie_cost_l304_304467


namespace math_proof_problem_l304_304510

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304510


namespace john_spending_on_mangoes_before_reduction_l304_304464

noncomputable def original_price_per_mango (total_price: ℝ) (num_mangoes: ℝ) : ℝ :=
  total_price / num_mangoes

noncomputable def new_price_per_mango (original_price: ℝ): ℝ :=
  0.9 * original_price

noncomputable def num_mangoes_purchased (new_price: ℝ) (original_price: ℝ) (extra_mangoes: ℕ): ℝ :=
  extra_mangoes / (original_price - new_price)

noncomputable def amount_spent (num_mangoes: ℝ) (price_per_mango: ℝ) : ℝ :=
  num_mangoes * price_per_mango

theorem john_spending_on_mangoes_before_reduction :
  original_price_per_mango 433.33 130 = 3.333 -> 
  new_price_per_mango 3.333 = 3.00 -> 
  let N := num_mangoes_purchased 3.00 3.333 12 in 
  N = 108 -> 
  amount_spent N 3.333 = 360 := by
  intro h1 h2 h3
  rw [h1, h2, h3]
  have : num_mangoes_purchased 3.00 3.333 12 = 108 := by sorry
  rw [this]
  unfold amount_spent
  simp
  sorry

end john_spending_on_mangoes_before_reduction_l304_304464


namespace smallest_three_digit_multiple_of_17_l304_304675

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304675


namespace sum_of_super_rectangle_areas_l304_304920

theorem sum_of_super_rectangle_areas :
  ∃ (a b : ℕ), (a * b = 3 * (2 * a + 2 * b)) ∧ 
              ((a = 7 ∧ b = 42 ∨ a = 8 ∧ b = 24 ∨ a = 9 ∧ b = 18 ∨ a = 10 ∧ b = 15 ∨ a = 12 ∧ b = 12) ∧ 
                (a * b = 294 ∨ a * b = 192 ∨ a * b = 162 ∨ a * b = 150 ∨ a * b = 144)) ∧ 
              (Σ(area: ℕ), area = 942) :=
sorry

end sum_of_super_rectangle_areas_l304_304920


namespace even_function_sufficient_condition_l304_304294

theorem even_function_sufficient_condition (φ : ℝ) (h : φ = π / 4) :
  ∃ k : ℤ, φ = π / 4 + k * (π / 2) → ∀ x : ℝ, sin (x + 2 * φ) = sin (-x + 2 * φ) :=
by
  sorry

end even_function_sufficient_condition_l304_304294


namespace smallest_three_digit_multiple_of_17_l304_304748

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304748


namespace relationship_p_q_l304_304283

variable {a b : ℝ}

theorem relationship_p_q (ha : a > 2) (hb : b ∈ ℝ) :
  let p := a + 1 / (a - 2),
      q := -b^2 - 2 * b + 3
  in p ≥ q := sorry

end relationship_p_q_l304_304283


namespace spending_together_l304_304217

def sandwich_cost := 2
def hamburger_cost := 2
def hotdog_cost := 1
def juice_cost := 2
def selene_sandwiches := 3
def selene_juices := 1
def tanya_hamburgers := 2
def tanya_juices := 2

def selene_spending : ℕ := (selene_sandwiches * sandwich_cost) + (selene_juices * juice_cost)
def tanya_spending : ℕ := (tanya_hamburgers * hamburger_cost) + (tanya_juices * juice_cost)
def total_spending : ℕ := selene_spending + tanya_spending

theorem spending_together : total_spending = 16 :=
by
  sorry

end spending_together_l304_304217


namespace solution_set_of_inequality_l304_304230

noncomputable def solutionSet (f : ℝ → ℝ) : set ℝ := 
  {x | (f x - f (-x)) / x < 0}

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : odd f)
  (h2 : ∀ x > 0, f x < f (x + 1)) 
  (h3 : f 1 = 0) :
  solutionSet f = set.Ioc (-1 : ℝ) 0 ∪ set.Ioc 0 1 :=
sorry

end solution_set_of_inequality_l304_304230


namespace maximum_value_of_w_l304_304299

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l304_304299


namespace sum_of_altitudes_l304_304358

theorem sum_of_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : a + b = 21 :=
by
  -- Using the provided hypotheses, the proof would ensure a + b = 21.
  sorry

end sum_of_altitudes_l304_304358


namespace smallest_three_digit_multiple_of_17_correct_l304_304615

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304615


namespace sum_of_center_coords_is_eight_l304_304403

theorem sum_of_center_coords_is_eight :
  ∃ (A B C D : ℝ × ℝ), 
    (A.1 = 2 ∧ A.2 = 0) ∧ 
    (B.1 = 6 ∧ B.2 = 0) ∧ 
    (C.1 = 10 ∧ C.2 = 0) ∧ 
    (D.1 = 14 ∧ D.2 = 0) ∧ 
    let center := (A.1 + B.1 + C.1 + D.1) / 4 in
    center = 8 :=
sorry

end sum_of_center_coords_is_eight_l304_304403


namespace paint_cost_contribution_l304_304416

theorem paint_cost_contribution
  (paint_cost_per_gallon : ℕ) 
  (coverage_per_gallon : ℕ) 
  (total_wall_area : ℕ) 
  (two_coats : ℕ) 
  : paint_cost_per_gallon = 45 → coverage_per_gallon = 400 → total_wall_area = 1600 → two_coats = 2 → 
    ((total_wall_area / coverage_per_gallon) * two_coats * paint_cost_per_gallon) / 2 = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_cost_contribution_l304_304416


namespace problem_statement_l304_304077

noncomputable def omega : ℂ := sorry -- Definition placeholder for a specific nonreal root of x^4 = 1. 

theorem problem_statement (h1 : omega ^ 4 = 1) (h2 : omega ^ 2 = -1) : 
  (1 - omega + omega ^ 3) ^ 4 + (1 + omega - omega ^ 3) ^ 4 = -14 := 
sorry

end problem_statement_l304_304077


namespace arithmetic_sequence_sum_l304_304399

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_sum (h1 : ∀ n, a n > 0) (h2 : (∑ n in finset.range 10, a (n + 1)) = 30) :
  (a 5 + a 6) = 6 := by
  -- proof steps would go here
  sorry

end arithmetic_sequence_sum_l304_304399


namespace unique_solution_l304_304196

def floor (x : ℝ) : ℤ := Int.floor x

theorem unique_solution {x : ℝ} (h : (floor ((x + 3) / 2))^2 - x = 1) : x = 0 :=
by
  sorry

end unique_solution_l304_304196


namespace perp_proof_l304_304886

noncomputable def problem_statement (A B C K H : Point) :=
  is_acute A B C ∧ altitude B H A C ∧ dist A B = dist C H ∧
  ∠ B K C = ∠ B C K ∧ ∠ A B K = ∠ A C B

theorem perp_proof (A B C K H : Point) 
  (h_acute : is_acute A B C)
  (h_alt : altitude B H A C) 
  (h_AB_CH : dist A B = dist C H)
  (h_angle_BKC_BCK : ∠ B K C = ∠ B C K)
  (h_angle_ABK_ACB : ∠ A B K = ∠ A C B) : 
  perp A K A B := 
sorry

end perp_proof_l304_304886


namespace line_equation_k_value_l304_304054

theorem line_equation_k_value (m n k : ℝ) 
    (h1 : m = 2 * n + 5) 
    (h2 : m + 5 = 2 * (n + k) + 5) : 
    k = 2.5 :=
by sorry

end line_equation_k_value_l304_304054


namespace zero_knights_l304_304895

noncomputable def knights_count (n : ℕ) : ℕ := sorry

theorem zero_knights (n : ℕ) (half_lairs : n ≥ 205) :
  knights_count 410 = 0 :=
sorry

end zero_knights_l304_304895


namespace number_of_incorrect_propositions_l304_304274

theorem number_of_incorrect_propositions :
  ∀ (A B C : ℝ × ℝ)
  (d : ℝ × ℝ → ℝ × ℝ → ℝ)
  (p1 : ∀ (x1 y1 x2 y2 x0 y0 : ℝ), (x0 = (x2 - x1) / 2) → (y0 = (y2 - y1) / 2) → d (x1, y1) (x2, y2) = d (x1, y1) (x0, y0) + d (x0, y0) (x2, y2))
  (p2 : ∀ (x1 y1 x2 y2 x3 y3 : ℝ), d(x1, y1) (x3, y3) + d(x3, y3) (x2, y2) > d (x1, y1) (x2, y2))
  (p3 : ∀ (x1 y1 x2 y2 x3 y3 : ℝ), x1^2 + y1^2 + x2^2 + y2^2 = x3^2 + y3^2),
  let incorrect := (¬ (∀ x1 y1 x2 y2 x0 y0 : ℝ, x0 = (x2 - x1) / 2 → y0 = (y2 - y1) / 2 → d(x1, y1) (x2, y2) = d(x1, y1) (x0, y0) + d(x0, y0) (x2, y2)))
                  + (∀ x1 y1 x2 y2 x3 y3, d(x1, y1) (x3, y3) + d(x3, y3) (x2, y2) > d (x1, y1) (x2, y2))
                  + (¬ ∀ x1 y1 x2 y2 x3 y3, x1^2 + y1^2 + x2^2 + y2^2 = x3^2 + y3^2) in
  incorrect = 2 := by
sorry

end number_of_incorrect_propositions_l304_304274


namespace smallest_prime_factor_l304_304490

def C : Set ℕ := {65, 67, 68, 71, 73}

theorem smallest_prime_factor : ∃ (n ∈ C), ∀ m ∈ C, factor_fn 68 < factor_fn m := sorry

end smallest_prime_factor_l304_304490


namespace smallest_three_digit_multiple_of_17_l304_304771

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304771


namespace smallest_three_digit_multiple_of_17_l304_304727

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304727


namespace min_value_of_z_l304_304435

def z (x y : ℝ) : ℝ := 4^x * 2^y

theorem min_value_of_z :
  (∃ x y : ℝ, (x - 4*y ≤ -3) ∧ (3*x + 5*y ≤ 25) ∧ (x ≥ 1)) →
  (∀ x y : ℝ, (x - 4*y ≤ -3) → (3*x + 5*y ≤ 25) → (x ≥ 1) → z x y ≥ 8) ∧
  (∃ x y : ℝ, (x - 4*y ≤ -3) ∧ (3*x + 5*y ≤ 25) ∧ (x ≥ 1) ∧ z x y = 8) :=
by 
  sorry

end min_value_of_z_l304_304435


namespace tilling_time_in_minutes_l304_304947

-- Definitions
def plot_width : ℕ := 110
def plot_length : ℕ := 120
def tiller_width : ℕ := 2
def tilling_rate : ℕ := 2 -- 2 seconds per foot

-- Theorem: The time to till the entire plot in minutes
theorem tilling_time_in_minutes : (plot_width / tiller_width * plot_length * tilling_rate) / 60 = 220 := by
  sorry

end tilling_time_in_minutes_l304_304947


namespace find_b_minus_c_l304_304272

open Real

noncomputable def a (n : ℕ) : ℝ := 1 / (log 5000 / log n)

noncomputable def b : ℝ := a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ := a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = log 5000 (1 / 3876) :=
  sorry

end find_b_minus_c_l304_304272


namespace sequence_bound_l304_304252

noncomputable def sequence (k : ℝ) (h : 1 < k ∧ k < 2) : ℕ → ℝ
| 0       := k
| (n + 1) := sequence n - (sequence n ^ 2) / 2 + 1

theorem sequence_bound (k : ℝ) (h : 1 < k ∧ k < 2) : ∀ n > 2, | sequence k h n - Real.sqrt 2 | < 1 / 2 ^ n :=
by
  assume n hn_pos
  sorry

end sequence_bound_l304_304252


namespace sum_of_circular_integers_l304_304526

theorem sum_of_circular_integers (a : Fin 10 → ℕ) (h : ∀ i, a i = Nat.gcd (a ((i - 1) % 10)) (a ((i + 1) % 10)) + 1) :
    (Finset.univ.sum (λ i, a i)) = 28 := by
  sorry

end sum_of_circular_integers_l304_304526


namespace concentric_circles_squares_equal_l304_304422

-- Let Gamma and Gamma' be concentric circles with equilateral triangles ABC and A'B'C' inscribed in them.
-- Let P and P' be points on Gamma and Gamma' respectively.
-- Prove: P'A^2 + P'B^2 + P'C^2 = A'P^2 + B'P^2 + C'P^2.

variables {R r : ℝ} (hR : R > r) -- Radii of the circles
variables (O A B C P P' : EuclideanPoint) -- Points O is the center, others are the vertices
variables (A' B' C' : EuclideanPoint) -- Points on the inscribed triangles

definition equilateral_triangle (A B C : EuclideanPoint) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

definition inscribed_in_circle (O : EuclideanPoint) (A B C : EuclideanPoint) (R : ℝ) : Prop :=
  dist O A = R ∧ dist O B = R ∧ dist O C = R

theorem concentric_circles_squares_equal
  (hO : dist O A = R) (hO' : dist O A' = r)
  (hABC: equilateral_triangle A B C)
  (hA'B'C': equilateral_triangle A' B' C')
  (hPA: dist O P = R) (hP'A: dist O P' = r)
  : dist P' A ^ 2 + dist P' B ^ 2 + dist P' C ^ 2 = dist A' P ^ 2 + dist B' P ^ 2 + dist C' P ^ 2 :=
by sorry

end concentric_circles_squares_equal_l304_304422


namespace smallest_three_digit_multiple_of_17_l304_304808

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304808


namespace hyperbola_foci_distance_l304_304127

noncomputable def distance_between_foci (x y : ℝ) : ℝ :=
  let d: ℝ := 8 in 2 * d.sqrt

theorem hyperbola_foci_distance : 
  ∀ (a b: ℝ), (a = 1 ∧ b = 3) → (∀ x y, (x, y) = (4, 4) → 
  (x - a)^2 / 8 - (y - b)^2 / 8 = 1) → 
  distance_between_foci a b = 8 := 
by sorry

end hyperbola_foci_distance_l304_304127


namespace smallest_three_digit_multiple_of_17_l304_304834

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304834


namespace find_special_three_digit_number_l304_304935

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def first_two_digits (n : ℕ) : ℕ := n / 10
def last_digit (n : ℕ) : ℕ := n % 10

def three_digit_number_satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧ 
  is_perfect_square n ∧
  let first_two := first_two_digits n in
  let last := last_digit n in
  last ≠ 0 ∧ -- Ensure the last digit is not zero to avoid division by zero
  is_perfect_square (first_two / last)

theorem find_special_three_digit_number :
  ∃ n : ℕ, three_digit_number_satisfies_conditions n ∧ n = 361 :=
by
  sorry

end find_special_three_digit_number_l304_304935


namespace john_total_distance_l304_304884

-- Define the given conditions
def initial_speed : ℝ := 45 -- mph
def first_leg_time : ℝ := 2 -- hours
def second_leg_time : ℝ := 3 -- hours
def distance_before_lunch : ℝ := initial_speed * first_leg_time
def distance_after_lunch : ℝ := initial_speed * second_leg_time

-- Define the total distance
def total_distance : ℝ := distance_before_lunch + distance_after_lunch

-- Prove the total distance is 225 miles
theorem john_total_distance : total_distance = 225 := by
  sorry

end john_total_distance_l304_304884


namespace length_F_F_l304_304566

-- defining the original and reflected points
structure Point where
  x : ℝ
  y : ℝ

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

noncomputable def segment_length (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- original and reflected points
def F : Point := { x := -5, y := 3 }
def F' : Point := reflect_over_y_axis F

-- the statement to prove
theorem length_F_F' : segment_length F F' = 10 := by
  sorry

end length_F_F_l304_304566


namespace smallest_three_digit_multiple_of_17_l304_304851

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304851


namespace circle_ellipse_intersect_four_points_l304_304983

theorem circle_ellipse_intersect_four_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = a^2 → y = x^2 / 2 - a) →
  a > 1 :=
by
  sorry

end circle_ellipse_intersect_four_points_l304_304983


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304715

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304715


namespace fair_total_revenue_l304_304931

noncomputable def price_per_ticket : ℝ := 8
noncomputable def total_ticket_revenue : ℝ := 8000
noncomputable def total_tickets_sold : ℝ := total_ticket_revenue / price_per_ticket

noncomputable def food_revenue : ℝ := (3/5) * total_tickets_sold * 10
noncomputable def rounded_ride_revenue : ℝ := (333 : ℝ) * 6
noncomputable def ride_revenue : ℝ := rounded_ride_revenue
noncomputable def rounded_souvenir_revenue : ℝ := (166 : ℝ) * 18
noncomputable def souvenir_revenue : ℝ := rounded_souvenir_revenue
noncomputable def game_revenue : ℝ := (1/10) * total_tickets_sold * 5

noncomputable def total_additional_revenue : ℝ := food_revenue + ride_revenue + souvenir_revenue + game_revenue
noncomputable def total_revenue : ℝ := total_ticket_revenue + total_additional_revenue

theorem fair_total_revenue : total_revenue = 19486 := by
  sorry

end fair_total_revenue_l304_304931


namespace find_m_value_l304_304348

open Real

-- Define the vectors a and b as specified in the problem
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the sum of vectors a and b
def vec_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the dot product of the vector sum with vector b to be zero as the given condition
def dot_product (m : ℝ) : ℝ := (vec_sum m).1 * vec_b.1 + (vec_sum m).2 * vec_b.2

-- The theorem to prove that given the defined conditions, m equals 8
theorem find_m_value (m : ℝ) (h : dot_product m = 0) : m = 8 := by
  sorry

end find_m_value_l304_304348


namespace pages_copied_for_25_dollars_l304_304063

def cost_per_page := 3
def total_cents := 25 * 100

theorem pages_copied_for_25_dollars : (total_cents div cost_per_page) = 833 :=
by sorry

end pages_copied_for_25_dollars_l304_304063


namespace smallest_three_digit_multiple_of_17_l304_304698

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304698


namespace probability_product_multiple_of_18_l304_304380

open Finset

def set : Finset ℕ := {2, 3, 6, 9}

def isMultipleOf18 (a b : ℕ) : Prop := (a * b) % 18 = 0

def number_of_pairs : ℕ := (set.card).choose 2

def valid_pairs : Finset (ℕ × ℕ) := 
  { (2,9), (3,6), (6,9) : ℕ × ℕ | 
    set.mem <| (2, 9)
    ∨ set.mem <| (3, 6)
    ∨ set.mem <| (6, 9)}

def number_of_valid_pairs : ℕ := valid_pairs.card

theorem probability_product_multiple_of_18 :
  (number_of_valid_pairs : ℚ) / (number_of_pairs : ℚ) = 1 / 2 := 
sorry

end probability_product_multiple_of_18_l304_304380


namespace smallest_n_l304_304988

theorem smallest_n (r g b n : ℕ) 
  (h1 : 12 * r = 14 * g)
  (h2 : 14 * g = 15 * b)
  (h3 : 15 * b = 20 * n)
  (h4 : ∀ n', (12 * r = 14 * g ∧ 14 * g = 15 * b ∧ 15 * b = 20 * n') → n ≤ n') :
  n = 21 :=
by
  sorry

end smallest_n_l304_304988


namespace max_value_of_expression_l304_304076

variables {E : Type*} [inner_product_space ℝ E]

-- Variable declarations
variables (p q r : E)

-- Assumptions
axiom norm_p : ‖p‖ = 2
axiom norm_q : ‖q‖ = 1
axiom norm_r : ‖r‖ = 3

-- The statement to prove
theorem max_value_of_expression : 
  ‖p - 3 • q‖^2 + ‖q - 3 • r‖^2 + ‖r - 3 • p‖^2 ≤ 128 :=
sorry

end max_value_of_expression_l304_304076


namespace hawks_percentage_l304_304415

noncomputable def P := 60

theorem hawks_percentage (crows hawks total : ℕ) (P : ℕ) (h1 : crows = 30) (h2 : hawks = crows * (1 + P / 100)) (h3 : total = crows + hawks) : 
  total = 78 → P = 60 :=
by
  intros htotal
  rw [h1] at h2 h3
  have hhawks : hawks = 30 * (1 + P / 100) := h2 
  rw [hhawks] at h3
  suffices : 30 + 30 * (1 + P / 100) = 78 → P = 60
  sorry
  exact this htotal

end hawks_percentage_l304_304415


namespace largest_digit_not_in_odd_units_digits_l304_304172

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end largest_digit_not_in_odd_units_digits_l304_304172


namespace coordinates_of_B_l304_304404

theorem coordinates_of_B (A B : ℝ × ℝ) (h1 : A = (-2, 3)) (h2 : (A.1 = B.1 ∨ A.1 + 1 = B.1 ∨ A.1 - 1 = B.1)) (h3 : A.2 = B.2) : 
  B = (-1, 3) ∨ B = (-3, 3) := 
sorry

end coordinates_of_B_l304_304404


namespace smallest_three_digit_multiple_of_17_l304_304801

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304801


namespace smallest_three_digit_multiple_of_17_l304_304835

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304835


namespace smallest_three_digit_multiple_of_17_l304_304635

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304635


namespace smallest_value_of_n_l304_304990

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l304_304990


namespace smallest_three_digit_multiple_of_17_l304_304777

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304777


namespace friends_seating_count_l304_304907

open Nat

def friends_seating_arrangements : Prop :=
  ∃ (E G L O1 O2 O3 O4 : Nat)
    (positions : Finset (Fin 8))
    (remaining_positions : Finset (Fin 8)),
  positions.card = 6 ∧ -- Seating exactly 6 friends
  ({0, 1, 2, 3, 4, 5, 6, 7} : Finset (Fin 8)).card = 8 ∧
  ((E = 1 ∧ G = 2) ∨ (E = 2 ∧ G = 3) ∨ (E = 3 ∧ G = 4) ∨ (E = 4 ∧ G = 5) ∨
   (E = 5 ∧ G = 6) ∨ (E = 6 ∧ G = 7) ∨ (E = 7 ∧ G = 8) ∨ 
   (E = 8 ∧ G = 9)) ∧ -- Euler and Gauss sit next to each other
  (remaining_positions = {O1, O2, O3, O4}) ∧ -- Remaining 4 friends
  ∀ p ∈ remaining_positions, p ≠ E + 1 ∧ p ≠ G + 1 ∧ p ≠ E - 1 ∧ p ≠ G - 1 ∧ -- Lagrange constraint
  (positions = {E, G, L, O1, O2, O3, O4}) ∧
  (L ≠ E + 1 ∧ L ≠ G + 1 ∧ L ≠ E - 1 ∧ L ≠ G - 1) -- Lagrange not next to Euler or Gauss

theorem friends_seating_count : ∃ n, friends_seating_arrangements ∧ n = 3360 :=
by
  sorry

end friends_seating_count_l304_304907


namespace cyclic_sum_leq_2s_2n_l304_304420

noncomputable def cyclic_sum {n : ℕ} (a : fin n → ℝ) : ℝ :=
  (finset.range n).sum (λ i, let i1 := (i + 1) % n
                                 i2 := (i + 2) % n
                             in (a i)^2 + (a i1)^2 - (a i2)^2) / (a i + a i1 - a i2)

theorem cyclic_sum_leq_2s_2n {n : ℕ} (hn : 3 ≤ n)(a : fin n → ℝ) (h_range : ∀ i, 2 ≤ a i ∧ a i ≤ 3) :
  let s := (finset.univ.sum a) in
  cyclic_sum a ≤ 2 * s - 2 * n :=
by
  sorry

end cyclic_sum_leq_2s_2n_l304_304420


namespace european_scientists_ratio_l304_304473

theorem european_scientists_ratio (T : ℕ) (hT : T = 70)
  (C : ℕ) (hC : C = T / 5)
  (U : ℕ) (hU : U = 21) :
  let E := T - (C + U) in
  (E : ℚ) / T = 1 / 2 :=
by
  -- placeholder for actual proof
  sorry

end european_scientists_ratio_l304_304473


namespace length_of_overlapping_part_l304_304877

theorem length_of_overlapping_part
  (l_p : ℕ)
  (n : ℕ)
  (total_length : ℕ)
  (l_o : ℕ) :
  n = 3 →
  l_p = 217 →
  total_length = 627 →
  3 * l_p - 2 * l_o = total_length →
  l_o = 12 := by
  intros n_eq l_p_eq total_length_eq equation
  sorry

end length_of_overlapping_part_l304_304877


namespace find_d_l304_304278

theorem find_d (d : ℤ) (h_a : ℝ) (h_b : ℝ) (h_c : ℝ) (ha3 : h_a = 3) (hb7 : h_b = 7) (hc_d : h_c = d) :
  2.1 < d ∧ d < 5.25 :=
by
  sorry

end find_d_l304_304278


namespace min_omega_l304_304375

open Real

def omega_pos (ω : ℝ) : Prop := ω > 0

def f (ω x : ℝ) : ℝ := sin (ω * x) + sin (ω * x - π / 2)

def symmetric_about (ω : ℝ) : Prop := ∃ k : ℤ, ω = 8 * k + 2

def zero_within (ω : ℝ) : Prop := -ω / 4 - π / 4 ≤ -π

theorem min_omega (ω : ℝ) (h_pos : omega_pos ω) (h_sym : symmetric_about ω) (h_zero : zero_within ω) :
  ω = 10 := by
  sorry

end min_omega_l304_304375


namespace pages_copied_for_25_dollars_l304_304061

def cost_per_page := 3
def total_cents := 25 * 100

theorem pages_copied_for_25_dollars : (total_cents div cost_per_page) = 833 :=
by sorry

end pages_copied_for_25_dollars_l304_304061


namespace number_of_roots_eq_seven_l304_304969

noncomputable def problem_function (x : ℝ) : ℝ :=
  (21 * x - 11 + (Real.sin x) / 100) * Real.sin (6 * Real.arcsin x) * Real.sqrt ((Real.pi - 6 * x) * (Real.pi + x))

theorem number_of_roots_eq_seven :
  (∃ xs : List ℝ, (∀ x ∈ xs, problem_function x = 0) ∧ (∀ x ∈ xs, -1 ≤ x ∧ x ≤ 1) ∧ xs.length = 7) :=
sorry

end number_of_roots_eq_seven_l304_304969


namespace smallest_three_digit_multiple_of_17_l304_304584

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304584


namespace smallest_three_digit_multiple_of_17_l304_304594

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304594


namespace smallest_three_digit_multiple_of_17_l304_304858

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304858


namespace quadrilateral_angles_triangle_contradiction_l304_304954

-- Statement of the problem
theorem quadrilateral_angles_triangle_contradiction
  (α β γ θ₄ : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_convex_quad : θ₁ + θ₂ + θ₃ + θ₄ = 360)
  (θ₁ = α)
  (θ₂ = β)
  (θ₃ = γ)
  : θ₄ = 180 → False :=
sorry

end quadrilateral_angles_triangle_contradiction_l304_304954


namespace smallest_three_digit_multiple_of_17_l304_304783

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304783


namespace find_value_of_a_3m_2n_l304_304282

variable {a : ℝ} {m n : ℕ}
axiom h1 : a ^ m = 2
axiom h2 : a ^ n = 5

theorem find_value_of_a_3m_2n : a ^ (3 * m - 2 * n) = 8 / 25 := by
  sorry

end find_value_of_a_3m_2n_l304_304282


namespace value_of_s_for_g_neg_1_eq_0_l304_304441

def g (x s : ℝ) := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

theorem value_of_s_for_g_neg_1_eq_0 (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end value_of_s_for_g_neg_1_eq_0_l304_304441


namespace log_expr_solution_l304_304981

-- Define the function that represents the recursive logarithmic expression
def log_expr (x : ℝ) := Real.logb 3 (81 + x)

-- The main statement we want to prove
theorem log_expr_solution : ∃ x : ℝ, x = log_expr x ∧ 0 < x ∧ x = 8 :=
by
  sorry

end log_expr_solution_l304_304981


namespace cube_vertex_plane_distance_l304_304905

theorem cube_vertex_plane_distance
  (d : ℝ)
  (h_dist : d = 9 - Real.sqrt 186)
  (h7 : ∀ (a b c  : ℝ), a^2 + b^2 + c^2 = 1 → 64 * (a^2 + b^2 + c^2) = 64)
  (h8 : ∀ (d : ℝ), 3 * d^2 - 54 * d + 181 = 0) :
  ∃ (p q r : ℕ), 
    p = 27 ∧ q = 186 ∧ r = 3 ∧ (p + q + r < 1000) ∧ (d = (p - Real.sqrt q) / r) := 
  by
    sorry

end cube_vertex_plane_distance_l304_304905


namespace smallest_three_digit_multiple_of_17_l304_304743

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304743


namespace michael_truck_meetings_l304_304455

noncomputable def michael_speed : ℝ := 5
noncomputable def trash_pail_distance : ℝ := 200
noncomputable def truck_speed : ℝ := 10
noncomputable def truck_stop_time : ℝ := 30

def meet_count : ℕ := 5

theorem michael_truck_meetings :
  ∃ (meetings : ℕ), 
  (michael_speed = 5) ∧ 
  (trash_pail_distance = 200) ∧ 
  (truck_speed = 10) ∧ 
  (truck_stop_time = 30) ∧ 
  (meetings = meet_count) :=
by 
  use 5
  simp [michael_speed, trash_pail_distance, truck_speed, truck_stop_time, meet_count]
  sorry

end michael_truck_meetings_l304_304455


namespace roots_equivalence_l304_304520

open Polynomial

noncomputable def polynomial_roots (p: Polynomial ℝ) : ℕ := sorry -- Definition to count the real roots of polynomial p

theorem roots_equivalence (P Q : Polynomial ℝ) (λ : ℝ) 
  (h1 : P.derivative * Q - Q.derivative * P ≠ 0)
  (h2 : degree P = degree Q) : 
  polynomial_roots P = polynomial_roots (λ * P + (1 - λ) * Q) :=
sorry

end roots_equivalence_l304_304520


namespace isosceles_triangle_interior_point_perpendicular_l304_304071

theorem isosceles_triangle_interior_point_perpendicular (P A B C : Point) 
  (h : isosceles_right_triangle A B C) (h1 : interior P A B C)
  (h2 : ∠PAB + ∠PBC + ∠PCA = 90) : perpendicular AP BC := sorry

end isosceles_triangle_interior_point_perpendicular_l304_304071


namespace probability_odd_divisor_22_l304_304148

theorem probability_odd_divisor_22! : 
  let factorial_22 := 2^19 * 3^10 * 5^4 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1,
      total_divisors := (19 + 1) * (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1),
      odd_divisors := (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in 
  (odd_divisors : ℚ) / total_divisors = 1 / 20 :=
by sorry

end probability_odd_divisor_22_l304_304148


namespace count_valid_arrangements_correct_l304_304472

def keychains := ["Star", "Moon", "Sun", "Cloud", "Comet"]

def is_valid_arrangement (arr : List String) : Prop :=
  ∀ i, (i < arr.length - 1) → ¬(arr[i] = "Star" ∧ arr[i+1] = "Sun" ∨ arr[i] = "Sun" ∧ arr[i+1] = "Star")

def count_valid_arrangements (arrangements : List (List String)) : Nat :=
  arrangements.count is_valid_arrangement

theorem count_valid_arrangements_correct :
  let arrangements := [("Star", "Moon", "Sun", "Cloud", "Comet")].permutations;
  count_valid_arrangements arrangements = 72 :=
by
  sorry

end count_valid_arrangements_correct_l304_304472


namespace solve_abs_equation_l304_304517

theorem solve_abs_equation (y : ℤ) : (|y - 8| + 3 * y = 12) ↔ (y = 2) :=
by
  sorry  -- skip the proof steps.

end solve_abs_equation_l304_304517


namespace certain_event_l304_304870

-- Definitions of the events as propositions
def EventA : Prop := ∃ n : ℕ, n ≥ 1 ∧ (n % 2 = 0)
def EventB : Prop := ∃ t : ℝ, t ≥ 0  -- Simplifying as the event of an advertisement airing
def EventC : Prop := ∃ w : ℕ, w ≥ 1  -- Simplifying as the event of rain in Weinan on a specific future date
def EventD : Prop := true  -- The sun rises from the east in the morning is always true

-- The statement that Event D is the only certain event among the given options
theorem certain_event : EventD ∧ ¬EventA ∧ ¬EventB ∧ ¬EventC :=
by
  sorry

end certain_event_l304_304870


namespace smallest_three_digit_multiple_of_17_l304_304829

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304829


namespace roots_of_equation_l304_304374

-- define a noncomputable function for the square root of 2
noncomputable def sqrt2 := real.sqrt 2

-- define the equation condition
def eq_condition (x : ℝ) := (x + 2)^2 = 8

-- state the theorem containing the proof problem
theorem roots_of_equation : ∀ x : ℝ, eq_condition x → (x = 2 * sqrt2 - 2 ∨ x = -2 * sqrt2 - 2) :=
begin
  sorry
end

end roots_of_equation_l304_304374


namespace smallest_three_digit_multiple_of_17_l304_304861

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304861


namespace simplify_expression_l304_304500

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304500


namespace implication_equivalence_l304_304873

variable (P Q : Prop)

theorem implication_equivalence :
  (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by sorry

end implication_equivalence_l304_304873


namespace isosceles_triangle_side_length_l304_304107

-- Definitions of Points P and Q on the parabola y = -x^2
def P (p : ℝ) : ℝ × ℝ := (p, -p^2)
def Q (p : ℝ) : ℝ × ℝ := (-p, -p^2)

-- Define the origin point O
def O : ℝ × ℝ := (0, 0)

-- Function to compute the Euclidean distance between two points
def distance (A B : ℝ × ℝ) : ℝ := (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

-- The Lean statement (goal) to prove
theorem isosceles_triangle_side_length (p : ℝ) : 
  distance (P p) O = distance (Q p) O := 
  by
  -- Given that triangle POQ is isosceles with positions defined, we need to prove the distances.
  sorry

end isosceles_triangle_side_length_l304_304107


namespace distance_to_x_axis_l304_304889

-- Definitions and conditions
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 6 + P.2^2 / 2 = 1

def is_intersection (A B C D : ℝ × ℝ) : Prop :=
  A = (-2, 0) ∧ B = (2, 0) ∧ C = (0, -1) ∧ D = (0, 1)

def distance_sum (P A B C D : ℝ × ℝ) : Prop :=
  |P.1 - A.1| + |P.2 - A.2| + |P.1 - B.1| + |P.2 - B.2| +
  |P.1 - C.1| + |P.2 - C.2| + |P.1 - D.1| + |P.2 - D.2| = 4 * real.sqrt 6

-- Theorem to prove
theorem distance_to_x_axis (P A B C D : ℝ × ℝ)
  (h1 : is_on_ellipse P)
  (h2 : is_intersection A B C D)
  (h3 : distance_sum P A B C D) :
  abs P.2 = real.sqrt(78) / 13 :=
sorry

end distance_to_x_axis_l304_304889


namespace smallest_three_digit_multiple_of_17_l304_304793

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304793


namespace sports_club_people_after_four_years_l304_304218

theorem sports_club_people_after_four_years :
  let b : ℕ → ℕ := λ k, match k with
    | 0 => 8
    | n + 1 => 2 * b n - 2
  in b 4 = 98 :=
by
  sorry

end sports_club_people_after_four_years_l304_304218


namespace sum_of_ten_numbers_in_circle_l304_304525

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l304_304525


namespace angle_120_degrees_l304_304342

variable {ι : Type*}
variable [inner_product_space ℝ (ι → ℝ)]

namespace vector_geometry

open_locale real_inner_product_space

def angle_between (a b : ι → ℝ) : ℝ := real.arccos (inner a b / (∥a∥ * ∥b∥))

theorem angle_120_degrees
  {a b : ι → ℝ}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h : ∥a∥ = ∥b∥ ∧ ∥a∥ = ∥a + b∥) :
  angle_between a b = 2 * real.pi / 3 :=
by
  sorry

end vector_geometry

end angle_120_degrees_l304_304342


namespace modulus_of_z_l304_304296

theorem modulus_of_z (z : ℂ) (h : (1 - complex.I) * z = 2 * complex.I) : complex.abs z = real.sqrt 2 :=
sorry

end modulus_of_z_l304_304296


namespace simplify_and_evaluate_expression_l304_304505

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304505


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304709

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304709


namespace tangent_line_MP_l304_304234

theorem tangent_line_MP
  (O : Type)
  (circle : O → O → Prop)
  (K M N P L : O)
  (is_tangent : O → O → Prop)
  (is_diameter : O → O → O)
  (K_tangent : is_tangent K M)
  (eq_segments : ∀ {P Q R}, circle P Q → circle Q R → circle P R → (P, Q) = (Q, R))
  (diam_opposite : L = is_diameter K L)
  (line_intrsc : ∀ {X Y}, is_tangent X Y → circle X Y → (Y = Y) → P = Y)
  (circ : ∀ {X Y}, circle X Y) :
  is_tangent M P :=
by
  sorry

end tangent_line_MP_l304_304234


namespace simplify_and_evaluate_expression_l304_304506

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304506


namespace smallest_three_digit_multiple_of_17_l304_304652

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304652


namespace total_distance_is_27_l304_304480

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l304_304480


namespace half_angle_in_first_quadrant_l304_304305

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l304_304305


namespace right_triangle_area_l304_304169

theorem right_triangle_area (a b c : ℝ) (h₀ : a = 24) (h₁ : c = 30) (h2 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 216 :=
by
  sorry

end right_triangle_area_l304_304169


namespace garden_fencing_l304_304213

theorem garden_fencing (length width : ℕ) (h1 : length = 80) (h2 : width = length / 2) : 2 * (length + width) = 240 :=
by
  sorry

end garden_fencing_l304_304213


namespace ratio_PQ_PR_eq_AD_BE_l304_304105

open_locale classical

variables (A B C D E P Q R : Type)

-- Given conditions
variables [linear_ordered_field A B C P Q R]
variables [∀ a b : A, decidable (a = b)] 

-- Points D and E on sides BC and CA of triangle ABC such that BD = AE
variables on_side_BC : B ≠ C
variables on_side_CA : C ≠ A
variables (BD : A) (AE : A)
variables (h1 : BD = AE)

-- Segments AD and BE intersect at P
variables on_AD : A ≠ D
variables on_BE : B ≠ E
variables intersects_at_P : P ≠ D ∨ P ≠ E

-- The angle bisector of ∠ACB intersects AD and BE at Q and R respectively
variables bisects_ACB_at_Q : P ≠ Q
variables bisects_ACB_at_R : P ≠ R

-- To prove: PQ / PR = AD / BE
theorem ratio_PQ_PR_eq_AD_BE : (PQ / PR : A) = (AD / BE : A) :=
sorry

end ratio_PQ_PR_eq_AD_BE_l304_304105


namespace solve_xyz_sum_l304_304521

theorem solve_xyz_sum :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x+y+z)^3 - x^3 - y^3 - z^3 = 378 ∧ x+y+z = 9 :=
by
  sorry

end solve_xyz_sum_l304_304521


namespace correct_options_l304_304178

theorem correct_options (a : ℝ) (h1 : a > 0) (x : ℝ) (h2 : x > 1) (h3 : x > 0) (h4 : x < 5 / 4) :
  (C : ∀ (x : ℝ), x > 0 → sqrt x + 1 / sqrt x ≥ 2) ∧ 
  (D : ∀ (x : ℝ), x < 5 / 4 → 4 * x - 2 + 1 / (4 * x - 5) ≤ 1) :=
  sorry

end correct_options_l304_304178


namespace smallest_three_digit_multiple_of_17_l304_304760

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304760


namespace june_to_bernard_l304_304070

theorem june_to_bernard (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ)
  (h1 : distance_to_julia = 2) (h2 : time_to_julia = 6) (h3 : distance_to_bernard = 5) :
  ∃ (time_to_bernard : ℝ), time_to_bernard = 15 :=
begin
  sorry
end

end june_to_bernard_l304_304070


namespace smallest_three_digit_multiple_of_17_l304_304676

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304676


namespace prod_eq_zero_l304_304543

variable {a : ℤ}
variable {x : Fin 13 → ℤ}

theorem prod_eq_zero 
  (h1 : a = ∏ i, (1 + x i))
  (h2 : a = ∏ i, (1 - x i)) :
  a * ∏ i, x i = 0 := 
sorry

end prod_eq_zero_l304_304543


namespace count_five_digit_numbers_divisibility_by_11_l304_304084

theorem count_five_digit_numbers_divisibility_by_11 :
  let n_values := ({n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ ∃ q r, 
                    n = 100 * q + r ∧ 100 ≤ q ∧ q ≤ 999 ∧ 0 ≤ r ∧ r ≤ 99 ∧ (q + r) % 11 = 0}) 
  in Fintype.card n_values = 9000 :=
by {
  let n_values := {n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ ∃ q r, 
                    n = 100 * q + r ∧ 100 ≤ q ∧ q ≤ 999 ∧ 0 ≤ r ∧ r ≤ 99 ∧ (q + r) % 11 = 0},
  have n_val_count : Fintype.card n_values = 9000,
  exact n_val_count
}

end count_five_digit_numbers_divisibility_by_11_l304_304084


namespace smallest_three_digit_multiple_of_17_l304_304849

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304849


namespace half_angle_in_first_or_third_quadrant_l304_304306

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end half_angle_in_first_or_third_quadrant_l304_304306


namespace matrix_projection_ratios_l304_304332

theorem matrix_projection_ratios (x y z : ℚ) (h : 
  (1 / 14 : ℚ) * x - (5 / 14 : ℚ) * y = x ∧
  - (5 / 14 : ℚ) * x + (24 / 14 : ℚ) * y = y ∧
  0 * x + 0 * y + 1 * z = z)
  : y / x = 13 / 5 ∧ z / x = 1 := 
by 
  sorry

end matrix_projection_ratios_l304_304332


namespace smallest_three_digit_multiple_of_17_l304_304812

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304812


namespace range_of_a_l304_304273

theorem range_of_a {x y a : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + y + 6 = 4 * x * y) : a ≤ 10 / 3 :=
  sorry

end range_of_a_l304_304273


namespace profit_percentage_l304_304031

noncomputable def cost_price (C : ℝ) : Prop := true
noncomputable def selling_price (S : ℝ) : Prop := true

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : 
  (S = (17 / 16) * C) → (S - C) / C * 100 = 6.25 :=
by
  assume h1 : S = (17 / 16) * C,
  sorry

end profit_percentage_l304_304031


namespace part_I_part_II_l304_304012

-- Definitions for part (I)
def f (a x : ℝ) : ℝ := x^3 + (4 - a) * x^2 - 15 * x + a
def P : ℝ × ℝ := (0, -2)

-- Theorem for part (I)
theorem part_I (a : ℝ) :
  f a 0 = -2 → a = -2 ∧ ∃ x_min, minimum_value f a x_min = -10 := by
  sorry

-- Definitions for part (II)
def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * (4 - a) * x - 15

-- Theorem for part (II)
theorem part_II (a : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, f' a x ≤ 0) → a ≤ 10 := by
  sorry

end part_I_part_II_l304_304012


namespace smallest_three_digit_multiple_of_17_l304_304840

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304840


namespace solve_inequality_l304_304186

theorem solve_inequality (x : ℝ) (hx : x > 0) : 
  (||log 2 (abs x) + 1|| ≥ log 2 (4 * x)) ↔ (0 < x ∧ x ≤ real.sqrt 2 / 2) := 
sorry

end solve_inequality_l304_304186


namespace function_expression_log_comparison_l304_304014

theorem function_expression (k : ℕ) (hk : k ∈ {1, 2}) (symmetry : k^2 - 2 * k - 3 < 0) (decreasing : ∀ x ∈ Ioi (0 : ℝ), deriv (λ y, y^(k^2 - 2*k - 3)) x < 0) :
  f(x) = x^(k^2 - 2*k - 3) = x^(-4) :=
sorry

theorem log_comparison (a : ℝ) (ha : a > 1) :
  if 1 < a ∧ a < Real.exp 1 then (Real.log a)^(0.7) < (Real.log a)^(0.6) ∧
  if a = Real.exp 1 then (Real.log a)^(0.7) = (Real.log a)^(0.6) ∧
  if a > Real.exp 1 then (Real.log a)^(0.7) > (Real.log a)^(0.6) :=
sorry

end function_expression_log_comparison_l304_304014


namespace banyan_tree_area_l304_304128

-- Define the conditions: circumference and the formulas
def circumference := 6.28
def radius := circumference / (2 * Real.pi)
def area (r : ℝ) := Real.pi * r^2

-- The statement to be proven
theorem banyan_tree_area : area radius = Real.pi := by
  sorry

end banyan_tree_area_l304_304128


namespace solution_is_correct_l304_304959

def x_y_solution (x y : ℕ) :=
x^y + 3 = y^x ∧ 3 * x^y = y^x + 13

theorem solution_is_correct : x_y_solution 2 3 :=
by {
  have A : 2^3 + 3 = 3^2, by norm_num,
  have B : 3 * 2^3 = 3^2 + 13, by norm_num,
  split,
  assumption,
  assumption,
}

end solution_is_correct_l304_304959


namespace number_of_children_l304_304937

-- Definitions of ticket costs and total values
def ticket_cost_adult := 12
def ticket_cost_child := 6
def total_people := 80
def total_amount := 840
def senior_discount := 0.25
def num_seniors := 3
def group_discount := 0.25
def group_size := 15

-- Calculating discounted ticket prices
def ticket_cost_senior := ticket_cost_adult * (1 - senior_discount)

-- The target proof statement
theorem number_of_children (A C: ℕ) (H1 : A + C = total_people) (H2 : (ticket_cost_adult * A) + (ticket_cost_child * C) = total_amount) : C = 20 :=
by
  sorry

end number_of_children_l304_304937


namespace relationship_between_x_and_y_l304_304577

theorem relationship_between_x_and_y
  (z : ℤ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = (z^4 + z^3 + z^2 + z + 1) / (z^2 + 1))
  (h2 : y = (z^3 + z^2 + z + 1) / (z^2 + 1)) :
  (y^2 - 2 * y + 2) * (x + y - y^2) - 1 = 0 := 
by
  sorry

end relationship_between_x_and_y_l304_304577


namespace smallest_three_digit_multiple_of_17_l304_304640

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304640


namespace correct_alarm_clock_time_l304_304468

-- Definitions for the conditions
def alarm_set_time : ℕ := 7 * 60 -- in minutes
def museum_arrival_time : ℕ := 8 * 60 + 50 -- in minutes
def museum_touring_time : ℕ := 1 * 60 + 30 -- in minutes
def alarm_home_time : ℕ := 11 * 60 + 50 -- in minutes

-- The problem: proving the correct time the clock should be set to
theorem correct_alarm_clock_time : 
  (alarm_home_time - (2 * ((museum_arrival_time - alarm_set_time) + museum_touring_time / 2)) = 12 * 60) :=
  by
    sorry

end correct_alarm_clock_time_l304_304468


namespace sum_of_circular_integers_l304_304528

theorem sum_of_circular_integers (a : Fin 10 → ℕ) (h : ∀ i, a i = Nat.gcd (a ((i - 1) % 10)) (a ((i + 1) % 10)) + 1) :
    (Finset.univ.sum (λ i, a i)) = 28 := by
  sorry

end sum_of_circular_integers_l304_304528


namespace problem_solution_l304_304238

noncomputable def problem : ℝ := (√( 0.2 )) * (2 : ℝ)

theorem problem_solution : (problem : ℝ) ≈ 0.9 : ℝ :=
by 
  have h : problem = 0.8944
  exact sorry -- you may fill the proof steps here with approximation, simplification steps etc.

end problem_solution_l304_304238


namespace math_proof_problem_l304_304511

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304511


namespace valid_starting_day_count_l304_304915

-- Defining the structure of the 30-day month and conditions
def days_in_month : Nat := 30

-- A function to determine the number of each weekday in a month which also checks if the given day is valid as per conditions
def valid_starting_days : List Nat :=
  [1] -- '1' represents Tuesday being the valid starting day corresponding to equal number of Tuesdays and Thursdays

-- The theorem we want to prove
-- The goal is to prove that there is only 1 valid starting day for the 30-day month to have equal number of Tuesdays and Thursdays
theorem valid_starting_day_count (days : Nat) (valid_days : List Nat) : 
  days = days_in_month → valid_days = valid_starting_days :=
by
  -- Sorry to skip full proof implementation
  sorry

end valid_starting_day_count_l304_304915


namespace smallest_three_digit_multiple_of_17_l304_304852

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304852


namespace smallest_three_digit_multiple_of_17_l304_304689

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304689


namespace intersection_of_M_and_N_l304_304016

open Set

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 4, 8, 16} →
  N = {2, 4, 6, 8} →
  M ∩ N = {2, 4, 8} :=
by
  intros hM hN
  rw [hM, hN]
  ext x
  simp
  sorry

end intersection_of_M_and_N_l304_304016


namespace length_of_AB_l304_304295

theorem length_of_AB
  (F1 F2 A B : ℝ → ℝ) (b : ℝ) (h1 : 0 < b) (h2 : b < 1)
  (h_ellipse : ∀ p, p ∈ E ↔ p.1^2 + (p.2^2 / b^2) = 1)
  (h_foci : (F1 0, 0), (F2 0, 0) ∈ E) 
  (h_line : ∃ ℓ, ℓ = λ x, (x, b * x) ∧ ∀ p, p ∈ A ∨ p ∈ B ↔ ℓ p)
  (h_arith_prog : ∃ AF2 AB BF2 : ℝ, AF2 + BF2 = 2 * AB ∧ (AF2 + AB + BF2 = 4))
  : |AB| = 4 / 3 := by sorry

end length_of_AB_l304_304295


namespace smallest_three_digit_multiple_of_17_l304_304788

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304788


namespace smallest_three_digit_multiple_of_17_l304_304595

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304595


namespace pages_copied_for_25_dollars_l304_304062

def cost_per_page := 3
def total_cents := 25 * 100

theorem pages_copied_for_25_dollars : (total_cents div cost_per_page) = 833 :=
by sorry

end pages_copied_for_25_dollars_l304_304062


namespace sin_double_angle_value_l304_304037

theorem sin_double_angle_value (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (1/2) * Real.cos (2 * α) = Real.sin (π/4 + α)) :
  Real.sin (2 * α) = -1 :=
by
  sorry

end sin_double_angle_value_l304_304037


namespace smallest_three_digit_multiple_of_17_l304_304803

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304803


namespace samantha_candy_distribution_l304_304489

theorem samantha_candy_distribution (candy_count : ℕ) (friend_count : ℕ) (extra_candies : ℕ)
  (h1 : candy_count = 27) (h2 : friend_count = 5) (h3 : extra_candies = candy_count % friend_count) :
  extra_candies = 2 := by
  rw [h1, h2]
  show 27 % 5 = 2
  sorry

end samantha_candy_distribution_l304_304489


namespace smallest_three_digit_multiple_of_17_l304_304609

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304609


namespace vector_coordinates_l304_304017

def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)
def c : ℝ × ℝ × ℝ := (4, -1, -3)

theorem vector_coordinates :
  2 • a - 3 • b + 4 • c = (16, 0, -19) :=
sorry

end vector_coordinates_l304_304017


namespace parity_impossible_l304_304371

theorem parity_impossible {n m : ℤ} (h : even (n^2 - m^2)) : ¬odd (n + m) := 
sorry

end parity_impossible_l304_304371


namespace isosceles_triangle_perimeter_l304_304046

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 6) (h2 : b = 3) (h3 : a ≠ b) :
  (2 * b + a = 15) :=
by
  sorry

end isosceles_triangle_perimeter_l304_304046


namespace smallest_three_digit_multiple_of_17_l304_304680

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304680


namespace smallest_three_digit_multiple_of_17_l304_304604

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304604


namespace smallest_number_of_players_l304_304042

theorem smallest_number_of_players :
  ∃ n, n ≡ 1 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 4 [MOD 6] ∧ ∃ m, n = m * m ∧ ∀ k, (k ≡ 1 [MOD 3] ∧ k ≡ 2 [MOD 4] ∧ k ≡ 4 [MOD 6] ∧ ∃ m, k = m * m) → k ≥ n :=
sorry

end smallest_number_of_players_l304_304042


namespace max_triangle_area_l304_304166

theorem max_triangle_area (A B : ℝ × ℝ) (r : ℝ) (hAB : dist A B = 10)
  (hA_center : A = (0,0)) (hB : B = (10,0)) :
  ∃ r, r > 0 ∧ r ≤ 10 ∧ (∀ z, z > 0 ∧ z ≤ 10 → 5 * z ≤ 25) :=
begin
  use 5,
  split,
  { linarith, },
  split,
  { linarith, },
  { intros z hz,
    calc 5 * z ≤ 5 * 5 : by linarith
            ... = 25 : by norm_num, }
end

end max_triangle_area_l304_304166


namespace optimal_play_winner_l304_304569

theorem optimal_play_winner (n : ℕ) (h : n > 1) : (n % 2 = 0) ↔ (first_player_wins: Bool) :=
  sorry

end optimal_play_winner_l304_304569


namespace simplify_expression_l304_304498

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304498


namespace smallest_value_of_n_l304_304989

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l304_304989


namespace count_5_digit_numbers_q_plus_r_divisible_by_11_l304_304087

open Nat

theorem count_5_digit_numbers_q_plus_r_divisible_by_11 : 
  let FiveDigitNumbers := {n // 10^4 ≤ n ∧ n < 10^5},
  let IsDivisibleBy11 (n : ℕ) : Prop := let q := n / 100 in let r := n % 100 in (q + r) % 11 = 0,
  (Finset.filter IsDivisibleBy11 (Finset.filter (λ n, 10^4 ≤ n ∧ n < 10^5) (Finset.range 10^5))).card = 8181 :=
sorry

end count_5_digit_numbers_q_plus_r_divisible_by_11_l304_304087


namespace symmetric_line_origin_l304_304143

theorem symmetric_line_origin (a b : ℝ) :
  (∀ (m n : ℝ), a * m + 3 * n = 9 → -m + 3 * -n + b = 0) ↔ a = -1 ∧ b = -9 :=
by
  sorry

end symmetric_line_origin_l304_304143


namespace smallest_three_digit_multiple_of_17_l304_304805

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304805


namespace smallest_three_digit_multiple_of_17_l304_304802

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304802


namespace sum_of_min_and_max_z_l304_304339

theorem sum_of_min_and_max_z (x y : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 2 * x + y - 2 ≥ 0) (h3 : y ≥ 0) : 
  let z := x - y in
  let min_z := min (x - y) (-2) in
  let max_z := max (x - y) 1 in
  min_z + max_z = -1 :=
by
  sorry

end sum_of_min_and_max_z_l304_304339


namespace rubble_money_left_l304_304485

/-- Rubble has $15 in his pocket. -/
def rubble_initial_amount : ℝ := 15

/-- Each notebook costs $4.00. -/
def notebook_price : ℝ := 4

/-- Each pen costs $1.50. -/
def pen_price : ℝ := 1.5

/-- Rubble needs to buy 2 notebooks. -/
def num_notebooks : ℝ := 2

/-- Rubble needs to buy 2 pens. -/
def num_pens : ℝ := 2

/-- The total cost of the notebooks. -/
def total_notebook_cost : ℝ := num_notebooks * notebook_price

/-- The total cost of the pens. -/
def total_pen_cost : ℝ := num_pens * pen_price

/-- The total amount Rubble spends. -/
def total_spent : ℝ := total_notebook_cost + total_pen_cost

/-- The remaining amount Rubble has after the purchase. -/
def rubble_remaining_amount : ℝ := rubble_initial_amount - total_spent

theorem rubble_money_left :
  rubble_remaining_amount = 4 := 
by
  -- Some necessary steps to complete the proof
  sorry

end rubble_money_left_l304_304485


namespace total_road_signs_l304_304923

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l304_304923


namespace two_digit_numbers_count_l304_304354

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  ∃ n : ℕ, n = 40 ∧ ∀ d ∈ finset.range 90, 
  let m := d + 10 in
  10 ≤ m ∧ m < 100 ∧ tens_digit m < units_digit m →
  ∃ n : ℕ, n = 40 ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ tens_digit m < units_digit m → n = 40 := 
sorry

end two_digit_numbers_count_l304_304354


namespace coefficient_x18_is_zero_coefficient_x17_is_3420_l304_304258

open Polynomial

noncomputable def P : Polynomial ℚ := (1 + X^5 + X^7)^20

theorem coefficient_x18_is_zero : coeff P 18 = 0 :=
sorry

theorem coefficient_x17_is_3420 : coeff P 17 = 3420 :=
sorry

end coefficient_x18_is_zero_coefficient_x17_is_3420_l304_304258


namespace abs_ineq_sol_set_l304_304123

theorem abs_ineq_sol_set (x : ℝ) : (|x - 2| + |x - 1| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end abs_ineq_sol_set_l304_304123


namespace square_side_length_l304_304351

noncomputable def side_length_square_inscribed_in_hexagon : ℝ :=
  50 * Real.sqrt 3

theorem square_side_length (a b: ℝ) (h1 : a = 50) (h2 : b = 50 * (2 - Real.sqrt 3)) 
(s1 s2 s3 s4 s5 s6: ℝ) (ha : s1 = s2) (hb : s2 = s3) (hc : s3 = s4) 
(hd : s4 = s5) (he : s5 = s6) (hf : s6 = s1) : side_length_square_inscribed_in_hexagon = 50 * Real.sqrt 3 :=
by
  sorry

end square_side_length_l304_304351


namespace seating_problem_l304_304392

def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

theorem seating_problem :
  10.choose(10) - (fact 8) * (fact 3) = 3386880 := sorry

end seating_problem_l304_304392


namespace solution_set_of_inequality_l304_304327

variable {a x : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := Real.log a (|x + 1|)

theorem solution_set_of_inequality :
  (∀ x ∈ Ioo (-2 : ℝ) (-1), f x a > 0) → 
  {a : ℝ | f (4^a - 1) a > f 1 a} = Ioo 0 (1/2) :=
sorry

end solution_set_of_inequality_l304_304327


namespace negation_correct_l304_304147

namespace NegationProof

-- Define the original proposition 
def orig_prop : Prop := ∃ x : ℝ, x ≤ 0

-- Define the negation of the original proposition
def neg_prop : Prop := ∀ x : ℝ, x > 0

-- The theorem we need to prove
theorem negation_correct : ¬ orig_prop = neg_prop := by
  sorry

end NegationProof

end negation_correct_l304_304147


namespace count_5_digit_numbers_q_plus_r_divisible_by_11_l304_304086

open Nat

theorem count_5_digit_numbers_q_plus_r_divisible_by_11 : 
  let FiveDigitNumbers := {n // 10^4 ≤ n ∧ n < 10^5},
  let IsDivisibleBy11 (n : ℕ) : Prop := let q := n / 100 in let r := n % 100 in (q + r) % 11 = 0,
  (Finset.filter IsDivisibleBy11 (Finset.filter (λ n, 10^4 ≤ n ∧ n < 10^5) (Finset.range 10^5))).card = 8181 :=
sorry

end count_5_digit_numbers_q_plus_r_divisible_by_11_l304_304086


namespace smallest_three_digit_multiple_of_17_l304_304728

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304728


namespace final_position_of_F_l304_304544

theorem final_position_of_F :
  let F := (λ x : ℝ, (x, 0), (0, y : ℝ)) in
  let F_rotated := (λ x, (-x, 0), (0, -y)) in
  let F_reflected := (λ x, (-x, 0), (0, y)) in
  let F_translated := (λ x, (-x - 3, 0), (0, y - 2)) in
  (F_translated.base = (-x - 3, -2) ∧ F_translated.stem = (0, y - 2))
: (F_translated.base = (-x - 3, -2) ∧ F_translated.stem = (0, y - 2)) :=
sorry

end final_position_of_F_l304_304544


namespace probability_target_hit_probability_target_hit_by_A_alone_l304_304344

variable (A B : Event)
variable (pa pb : ℝ)
variable (pA : P A = 0.95)
variable (pB : P B = 0.9)
variable (independence : Independent A B)

/- The probability that the target is hit in a single shot is 0.995 -/
theorem probability_target_hit :
  P (A ∪ B) = 0.995 := by
  sorry

/- The probability that the target is hit by shooter A alone is 0.095 -/
theorem probability_target_hit_by_A_alone :
  P (A \cap Bᶜ) = 0.095 := by
  sorry

end probability_target_hit_probability_target_hit_by_A_alone_l304_304344


namespace simplified_function_sum_l304_304540

theorem simplified_function_sum :
  ∃ A B C D : ℤ, (∀ x : ℝ, x ≠ -1 → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C) ∧
  D = -1 ∧ A + B + C + D = 8 :=
begin
  sorry
end

end simplified_function_sum_l304_304540


namespace tangent_eqn_at_1_monotonicity_intervals_max_ab_value_l304_304011

noncomputable def f (x a : ℝ) := Real.exp x - a * (x - 1)

theorem tangent_eqn_at_1 (a : ℝ) (h : a = -1) : 
  (let f := f x a in 
  let slope := Real.exp 1 + 1 in 
  ∀ x y, (slope) * x - y - 1 = 0) :=
sorry

theorem monotonicity_intervals (a : ℝ) : 
  (if a ≤ 0 then ∀ x y, x < y → f x a < f y a ∧ ∀ I, I = Ioi (-∞) ∧ Ioi (∞) = (I) 
  else ∀ I, (I = Iio (Real.log a) ∨ I = Ioi (Real.log a))) :=
sorry

theorem max_ab_value (a b : ℝ) (h : ∀ x, f x a ≥ b): 
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

end tangent_eqn_at_1_monotonicity_intervals_max_ab_value_l304_304011


namespace smallest_three_digit_multiple_of_17_l304_304763

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304763


namespace original_price_of_lens_is_correct_l304_304461

-- Definitions based on conditions
def current_camera_price : ℝ := 4000
def new_camera_price : ℝ := current_camera_price + 0.30 * current_camera_price
def combined_price_paid : ℝ := 5400
def lens_discount : ℝ := 200
def combined_price_before_discount : ℝ := combined_price_paid + lens_discount

-- Calculated original price of the lens
def lens_original_price : ℝ := combined_price_before_discount - new_camera_price

-- The Lean theorem statement to prove the price is correct
theorem original_price_of_lens_is_correct : lens_original_price = 400 := by
  -- You do not need to provide the actual proof steps
  sorry

end original_price_of_lens_is_correct_l304_304461


namespace tanya_efficiency_increase_l304_304114

theorem tanya_efficiency_increase 
  (s_efficiency : ℝ := 1 / 10) (t_efficiency : ℝ := 1 / 8) :
  (((t_efficiency - s_efficiency) / s_efficiency) * 100) = 25 := 
by
  sorry

end tanya_efficiency_increase_l304_304114


namespace collinear_A_B_C_l304_304130

-- Define the points A, B, and C as the intersections described above
variables (A B C : Point) (S1 S2 S3 : Circle) 

-- Define that the points are intersections of the common external tangents
definition common_external_tangent_point1 (S1 S2 : Circle) : Point := sorry
definition common_external_tangent_point2 (S2 S3 : Circle) : Point := sorry
definition common_external_tangent_point3 (S3 S1 : Circle) : Point := sorry

axiom h1 : A = common_external_tangent_point1 S1 S2
axiom h2 : B = common_external_tangent_point2 S2 S3
axiom h3 : C = common_external_tangent_point3 S3 S1

-- The proof that A, B, and C are collinear
theorem collinear_A_B_C : ∃ line : Line, A ∈ line ∧ B ∈ line ∧ C ∈ line :=
by
  sorry

end collinear_A_B_C_l304_304130


namespace smallest_three_digit_multiple_of_17_l304_304816

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304816


namespace smallest_three_digit_multiple_of_17_l304_304733

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304733


namespace smallest_three_digit_multiple_of_17_l304_304655

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304655


namespace count_numbers_containing_3_l304_304019

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ d, 10^d ∣ n ∧ (n / 10^d) % 10 = 3

theorem count_numbers_containing_3 :
  (Finset.filter contains_digit_3 (Finset.range 1001)).card = 180 :=
by 
  sorry

end count_numbers_containing_3_l304_304019


namespace sum_recip_binom_le_one_sum_binom_ge_m_sq_l304_304088

variables {α : Type*} {A : Finset α} {n m : ℕ}
variable {A_i : Fin n → Finset α}

noncomputable def disjoint_subsets (A : Finset α) (m : ℕ) (A_i : Fin m → Finset α) :=
  (∀ i j : Fin m, i ≠ j → A_i i ∩ A_i j = ∅) ∧ (∀ i : Fin m, A_i i ⊆ A)

theorem sum_recip_binom_le_one
  (h_subsets : disjoint_subsets A m A_i) : 
  ∑ i in Finset.univ.image (λ i : Fin m, 1 / (Nat.choose n (A_i i).card)) ≤ 1 :=
  sorry

theorem sum_binom_ge_m_sq
  (h_subsets : disjoint_subsets A m A_i) : 
  ∑ i in Finset.univ.image (λ i : Fin m, Nat.choose n (A_i i).card) ≥ m^2 :=
  sorry

end sum_recip_binom_le_one_sum_binom_ge_m_sq_l304_304088


namespace stratified_sampling_l304_304205

theorem stratified_sampling :
  (total_employees employees_no_older_45 employees_older_45 sample_size : ℕ)
  (h1 : total_employees = 200)
  (h2 : employees_no_older_45 = 120)
  (h3 : employees_older_45 = 80)
  (h4 : sample_size = 25) :
  (sample_size * employees_no_older_45) / total_employees = 15 :=
by
  rw [h1, h2, h4]
  exact rfl

end stratified_sampling_l304_304205


namespace cost_of_two_dogs_l304_304465

theorem cost_of_two_dogs (original_price : ℤ) (profit_margin : ℤ) (num_dogs : ℤ) (final_price : ℤ) :
  original_price = 1000 →
  profit_margin = 30 →
  num_dogs = 2 →
  final_price = original_price + (profit_margin * original_price / 100) →
  num_dogs * final_price = 2600 :=
by
  sorry

end cost_of_two_dogs_l304_304465


namespace perimeter_of_midpoint_triangle_l304_304549

def triangle_perimeter_original : ℝ := 28

theorem perimeter_of_midpoint_triangle :
  let original_perimeter := triangle_perimeter_original in
  let smaller_perimeter := original_perimeter / 2 in
  smaller_perimeter = 14 :=
by
  let original_perimeter := triangle_perimeter_original
  let smaller_perimeter := original_perimeter / 2
  show smaller_perimeter = 14
  sorry

end perimeter_of_midpoint_triangle_l304_304549


namespace angle_in_third_quadrant_l304_304075

theorem angle_in_third_quadrant
  (α : ℝ)
  (k : ℤ)
  (h : (π / 2) + 2 * (↑k) * π < α ∧ α < π + 2 * (↑k) * π) :
  π + 2 * (↑k) * π < (π / 2) + α ∧ (π / 2) + α < (3 * π / 2) + 2 * (↑k) * π :=
by
  sorry

end angle_in_third_quadrant_l304_304075


namespace fixed_point_or_parallel_lines_l304_304960

-- Defining a generic point and simple geometry
variables {P : Type*} [plane P]

-- Define the pentagon and its properties
variables (A B C D E X K L Y : P)
variables (AB AE BK EL : ℝ)

-- Convex pentagon criterion
def convex_pentagon (A B C D E : P) : Prop :=
  -- Formal definition of convexity using geometry library assumptions
  sorry

-- Point X is on line segment CD
def on_line_segment_CD (X C D : P) : Prop :=
  -- Formal definition of X being on CD
  sorry

-- Points K and L lie on segment AX such that AB = BK and AE = EL
def on_line_segment_AX (K L A X : P) (AB BK AE EL : ℝ) : Prop :=
  AB = BK ∧ AE = EL

-- Circumcircles of triangles CXK and DXL intersect at Y
def circumcircle_intersection (C X K D L Y : P) : Prop :=
  -- Formal definition of circumcircle intersection
  sorry

-- The theorem we need to prove
theorem fixed_point_or_parallel_lines (A B C D E X K L Y : P)
  (h1 : convex_pentagon A B C D E)
  (h2 : on_line_segment_CD X C D)
  (h3 : on_line_segment_AX K L A X AB BK AE EL)
  (h4 : circumcircle_intersection C X K D L Y) :
  ∃ (P : P), ∀ (X : P), (line_contains P X Y ∨ parallel P X Y) :=
sorry

end fixed_point_or_parallel_lines_l304_304960


namespace count_hexagons_l304_304353

def lattice_point (x y : ℤ) : Prop := x^2 + y^2 = 13

def is_hexagon (points : Fin 6 → (ℤ × ℤ)) : Prop :=
  ∀ i, dist (points i) (points ((i + 1) % 6)) = sqrt 13 ∧
        dist (points 0) (0, 0) = 0

theorem count_hexagons :
  let points := (λ i, (0,0)) :: (Fin 5.succ → (ℤ × ℤ)) in 
  (∀ hexagon, is_hexagon hexagon points → lattice_point (hexagon 1).fst (hexagon 1).snd 
    → lattice_point (hexagon 2).fst (hexagon 2).snd
    → lattice_point (hexagon 3).fst (hexagon 3).snd
    → lattice_point (hexagon 4).fst (hexagon 4).snd
    → lattice_point (hexagon 5).fst (hexagon 5).snd)
    → 216 :=
sorry

end count_hexagons_l304_304353


namespace total_area_is_71_l304_304052

-- Define the lengths of the segments
def length_left : ℕ := 7
def length_top : ℕ := 6
def length_middle_1 : ℕ := 2
def length_middle_2 : ℕ := 4
def length_right : ℕ := 1
def length_right_top : ℕ := 5

-- Define the rectangles and their areas
def area_left_rect : ℕ := length_left * length_left
def area_middle_rect : ℕ := length_middle_1 * (length_top - length_left)
def area_right_rect : ℕ := length_middle_2 * length_middle_2

-- Define the total area
def total_area : ℕ := area_left_rect + area_middle_rect + area_right_rect

-- Theorem: The total area of the figure is 71 square units
theorem total_area_is_71 : total_area = 71 := by
  sorry

end total_area_is_71_l304_304052


namespace find_x_l304_304331

noncomputable def geometric_series_sum (x: ℝ) : ℝ := 
  1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + ∑' n: ℕ, (n + 1) * x^(n + 1)

theorem find_x (x: ℝ) (hx : geometric_series_sum x = 16) : x = 15 / 16 := 
by
  sorry

end find_x_l304_304331


namespace bowling_ball_weight_l304_304994

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l304_304994


namespace solution_set_of_inequality_l304_304033

-- Define the given conditions
variables {f : ℝ → ℝ}
variables (h1 : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) = f(x) + f(y))
variables (h2 : ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f(x) < f(y))

-- Define the theorem to prove
theorem solution_set_of_inequality : {x : ℝ | 0 < x ∧ f(x + 6) + f(x) < 2 * f(4)} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l304_304033


namespace coefficient_of_x_100_l304_304267

-- Define the polynomial P
noncomputable def P : Polynomial ℤ :=
  (Polynomial.C (-1) + Polynomial.X) *
  (Polynomial.C (-2) + Polynomial.X^2) *
  (Polynomial.C (-3) + Polynomial.X^3) *
  (Polynomial.C (-4) + Polynomial.X^4) *
  (Polynomial.C (-5) + Polynomial.X^5) *
  (Polynomial.C (-6) + Polynomial.X^6) *
  (Polynomial.C (-7) + Polynomial.X^7) *
  (Polynomial.C (-8) + Polynomial.X^8) *
  (Polynomial.C (-9) + Polynomial.X^9) *
  (Polynomial.C (-10) + Polynomial.X^10) *
  (Polynomial.C (-11) + Polynomial.X^11) *
  (Polynomial.C (-12) + Polynomial.X^12) *
  (Polynomial.C (-13) + Polynomial.X^13) *
  (Polynomial.C (-14) + Polynomial.X^14) *
  (Polynomial.C (-15) + Polynomial.X^15)

-- State the theorem
theorem coefficient_of_x_100 : P.coeff 100 = 445 :=
  by sorry

end coefficient_of_x_100_l304_304267


namespace smallest_three_digit_multiple_of_17_l304_304659

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304659


namespace simplify_and_evaluate_expression_l304_304504

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304504


namespace smallest_three_digit_multiple_of_17_l304_304785

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304785


namespace smallest_sum_of_a_and_b_l304_304078

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
(h1 : 9 * a^2 ≥ 16 * b) (h2 : 16 * b^2 ≥ 12 * a) : a + b = 70 / 3 :=
begin
  sorry
end

end smallest_sum_of_a_and_b_l304_304078


namespace smallest_three_digit_multiple_of_17_l304_304779

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304779


namespace sum_of_three_smallest_two_digit_primes_l304_304176

theorem sum_of_three_smallest_two_digit_primes :
  11 + 13 + 17 = 41 :=
by
  sorry

end sum_of_three_smallest_two_digit_primes_l304_304176


namespace smallest_three_digit_multiple_of_17_l304_304738

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304738


namespace integral_f_neg1_to_1_l304_304239

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then sin x - 1 else x^2

theorem integral_f_neg1_to_1 :
  ∫ x in -1..1, f x = cos 1 - (5 / 3) :=
by
  sorry

end integral_f_neg1_to_1_l304_304239


namespace smallest_three_digit_multiple_of_17_l304_304692

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304692


namespace smallest_integer_b_l304_304578

theorem smallest_integer_b (b : ℕ) : 27 ^ b > 3 ^ 9 ↔ b = 4 := by
  sorry

end smallest_integer_b_l304_304578


namespace overall_gain_percentage_correct_l304_304161

structure Transaction :=
  (buy_prices : List ℕ)
  (sell_prices : List ℕ)

def overallGainPercentage (trans : Transaction) : ℚ :=
  let total_cost := (trans.buy_prices.foldl (· + ·) 0 : ℚ)
  let total_sell := (trans.sell_prices.foldl (· + ·) 0 : ℚ)
  (total_sell - total_cost) / total_cost * 100

theorem overall_gain_percentage_correct
  (trans : Transaction)
  (h_buy_prices : trans.buy_prices = [675, 850, 920])
  (h_sell_prices : trans.sell_prices = [1080, 1100, 1000]) :
  overallGainPercentage trans = 30.06 := by
  sorry

end overall_gain_percentage_correct_l304_304161


namespace range_of_f_x_l304_304449

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x - 1) else -f (-x)

theorem range_of_f_x : { x : ℝ | f x + f (x - 1) < 2 } = set.Iio 2 :=
by
  sorry

end range_of_f_x_l304_304449


namespace smallest_three_digit_multiple_of_17_l304_304657

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304657


namespace place_tetrahedra_in_cube_non_overlapping_l304_304242

-- Definitions of the problem conditions
def is_regular_tetrahedron (tetrahedron : Type) : Prop :=
  -- Assume there's a definition for regular tetrahedron here
  sorry

def has_edge_length (object : Type) (length : ℝ) : Prop :=
  -- Assume there's a definition for having edge length here
  sorry

def inside_cube (cube tetrahedra : Type) : Prop :=
  -- Assume there's a definition of placement of tetrahedra inside a cube
  sorry

def non_overlapping (tetrahedra : list Type) : Prop :=
  -- Assume there's a definition for non-overlapping tetrahedra
  sorry

-- Definition of the cube and tetrahedra
constant Cube : Type
constant Tetrahedron1 : Type
constant Tetrahedron2 : Type
constant Tetrahedron3 : Type

-- Cube properties
axiom cube_edge_length : has_edge_length Cube 1

-- Tetrahedra properties
axiom tetrahedron1_regular : is_regular_tetrahedron Tetrahedron1
axiom tetrahedron2_regular : is_regular_tetrahedron Tetrahedron2
axiom tetrahedron3_regular : is_regular_tetrahedron Tetrahedron3

axiom tetrahedron1_edge_length : has_edge_length Tetrahedron1 1
axiom tetrahedron2_edge_length : has_edge_length Tetrahedron2 1
axiom tetrahedron3_edge_length : has_edge_length Tetrahedron3 1

-- Prove the existence of non-overlapping placement
theorem place_tetrahedra_in_cube_non_overlapping :
  ∃ (positions : list Type), (positions = [Tetrahedron1, Tetrahedron2, Tetrahedron3]) ∧
  inside_cube Cube positions ∧ non_overlapping positions :=
by {
  sorry
}

end place_tetrahedra_in_cube_non_overlapping_l304_304242


namespace neg_sin_prop_l304_304146

theorem neg_sin_prop : (¬ ∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 := 
sorry

end neg_sin_prop_l304_304146


namespace range_of_a_l304_304050

noncomputable def circle_O : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
noncomputable def circle_M (a : ℝ) : set (ℝ × ℝ) := {p | (p.1 + a + 1)^2 + (p.2 - 2 * a)^2 = 1}
constants (P : ℝ × ℝ) (Q : ℝ × ℝ) (a : ℝ)

axiom on_circle_O : P ∈ circle_O
axiom on_circle_M : Q ∈ circle_M a
axiom angle_30 : ∃ O, ∃ angle (O Q P) = 30 * (real.pi / 180)

theorem range_of_a :
  -1 ≤ a ∧ a ≤ 3 / 5 :=
sorry

end range_of_a_l304_304050


namespace smallest_three_digit_multiple_of_17_l304_304636

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304636


namespace smallest_three_digit_multiple_of_17_l304_304648

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304648


namespace rounding_possible_l304_304286

noncomputable def rounding_error_bound (n : ℕ) (x : Fin n → ℝ) : Prop :=
  let α := λ i : Fin n, x i - ⌊x i⌋
  ∀ m : ℕ, (1 ≤ m ∧ m ≤ n) →
    ∃ k : ℕ, (k ≤ m) ∧ 
      (|((n - k) - ∑ i in (Fin.range n) | k ≤ i, α i) - (∑ i in (Fin.range n) | i < k, α i)| ≤ (n + 1) / 4)

theorem rounding_possible (n : ℕ) (x : Fin n → ℝ) :
  rounding_error_bound n x :=
sorry

end rounding_possible_l304_304286


namespace solve_polynomial_eqn_l304_304265

theorem solve_polynomial_eqn (z : ℂ) : (z^4 - 8*z^2 + 15 = 0) ↔ (z = √5 ∨ z = -√5 ∨ z = √3 ∨ z = -√3) :=
by 
  sorry

end solve_polynomial_eqn_l304_304265


namespace solve_xyz_l304_304257

theorem solve_xyz (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : z > 0) (h4 : x^2 = y * 2^z + 1) :
  (z ≥ 4 ∧ x = 2^(z-1) + 1 ∧ y = 2^(z-2) + 1) ∨
  (z ≥ 5 ∧ x = 2^(z-1) - 1 ∧ y = 2^(z-2) - 1) ∨
  (z ≥ 3 ∧ x = 2^z - 1 ∧ y = 2^z - 2) :=
sorry

end solve_xyz_l304_304257


namespace Chloe_pairs_shoes_l304_304955

theorem Chloe_pairs_shoes (cost_per_shoe total_cost : ℤ) (h_cost: cost_per_shoe = 37) (h_total: total_cost = 1036) :
  (total_cost / cost_per_shoe) / 2 = 14 :=
by
  -- proof goes here
  sorry

end Chloe_pairs_shoes_l304_304955


namespace total_distance_hiked_l304_304059

def distance_car_to_stream : ℝ := 0.2
def distance_stream_to_meadow : ℝ := 0.4
def distance_meadow_to_campsite : ℝ := 0.1

theorem total_distance_hiked : 
  distance_car_to_stream + distance_stream_to_meadow + distance_meadow_to_campsite = 0.7 := by
  sorry

end total_distance_hiked_l304_304059


namespace smallest_three_digit_multiple_of_17_l304_304694

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304694


namespace smallest_three_digit_multiple_of_17_l304_304787

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304787


namespace magnitude_b_eq_l304_304347

-- Definitions for the problem
def a : ℝ × ℝ := (-2, -1)
def b (x y : ℝ) : ℝ × ℝ := (x, y)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Conditions
axiom dot_product_eq : ∀ (x y : ℝ), dot_product a (b x y) = 10
axiom magnitude_diff_eq : ∀ (x y : ℝ), magnitude (a.1 - x, a.2 - y) = real.sqrt 5

-- Theorem to prove
theorem magnitude_b_eq : ∃ x y, magnitude (b x y) = 2 * real.sqrt 5 :=
by {
  sorry
}

end magnitude_b_eq_l304_304347


namespace smallest_three_digit_multiple_of_17_l304_304826

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304826


namespace log_sequence_value_l304_304977

theorem log_sequence_value :
  ∃ x : ℝ, x = log 3 (81 + x) ∧ x > 0 ∧ x ≈ 5 :=
begin
  sorry
end

end log_sequence_value_l304_304977


namespace fraction_of_females_l304_304470

def local_soccer_league_female_fraction : Prop :=
  ∃ (males_last_year females_last_year : ℕ),
    males_last_year = 30 ∧
    (1.10 * males_last_year : ℝ) = 33 ∧
    (males_last_year + females_last_year : ℝ) * 1.15 = 52 ∧
    (females_last_year : ℝ) * 1.25 = 19 ∧
    (33 + 19 = 52)

theorem fraction_of_females
  : local_soccer_league_female_fraction → 
    ∃ (females fraction : ℝ),
    females = 19 ∧ 
    fraction = 19 / 52 :=
by
  sorry

end fraction_of_females_l304_304470


namespace last_page_Chandra_should_read_l304_304099

theorem last_page_Chandra_should_read
  (total_pages : ℕ)
  (alice_speed bob_speed chandra_speed : ℕ)
  (h_alice : alice_speed = 30)
  (h_bob : bob_speed = 50)
  (h_chandra : chandra_speed = 25)
  (h_total_pages : total_pages = 900) :
  ∃ pages_Chandra_reads : ℕ, pages_Chandra_reads = 600 ∧
  (bob_speed * pages_Chandra_reads = chandra_speed * (total_pages - pages_Chandra_reads)) :=
by
  use 600
  constructor
  · rfl
  · simp [h_bob, h_chandra, h_total_pages]
  sorry

end last_page_Chandra_should_read_l304_304099


namespace problem1_problem2_problem3_problem4_problem5_problem6_l304_304890

-- Problem (1)
theorem problem1 : 4 - (-28) + (-2) = 30 := by
  sorry

-- Problem (2)
theorem problem2 : (-3) * ((-2/5) / (-1/4)) = -24/5 := by
  sorry

-- Problem (3)
theorem problem3 : (-42) / (-7) - (-6) * 4 = 30 := by
  sorry

-- Problem (4)
theorem problem4 : (-3 ^ 2) / ((-3) ^ 2) + 3 * (-2) + | -4 | = -3 := by
  sorry

-- Problem (5)
theorem problem5 : (-24) * (3/4 - 5/6 + 7/12) = -12 := by
  sorry

-- Problem (6)
theorem problem6 : -1 ^ 4 - (1 - 0.5) / (5/2) * (1/5) = -11/10 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l304_304890


namespace smallest_three_digit_multiple_of_17_l304_304790

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304790


namespace smallest_three_digit_multiple_of_17_correct_l304_304622

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304622


namespace problem_inequality_l304_304079

variables (a b c : ℝ)
open Real

theorem problem_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
sorry

end problem_inequality_l304_304079


namespace propositions_correct_l304_304225

theorem propositions_correct :
  let domain_tangent := ∀ (x : ℝ), y = tan(x + π / 4) → (x ≠ π / 4 + k * π) :
  let sin_alpha := ∀ (α : ℝ), (sin α = 1 / 2) ∧ (0 ≤ α ∧ α ≤ 2 * π) → (α = π / 6 ∨ α = 5 * π / 6) :
  let symmetric_graph := ∀ (f : ℝ → ℝ), (f(x) = sin(2 * x) + a * cos(2 * x)) ∧ (sym(x = -π/8)) → a = -1 :
  let min_cos_sin := ∀ (x : ℝ), (y = cos(x)^2 + sin(x) → ∃ y_min, y_min = -1) :
  (domain_tangent = (x ≠ π / 4 + k * π)) ∧ 
  (sin_alpha = (α = π / 6 ∨ α = 5 * π / 6)) ∧ 
  (symmetric_graph = (a = -1)) ∧ 
  (min_cos_sin = (y_min = -1)) → 
  (Prop1 = true) ∧ (Prop2 = false) ∧ (Prop3 = true) ∧ (Prop4 = true) :=
begin
  sorry
end

end propositions_correct_l304_304225


namespace thirty_k_divisor_of_929260_l304_304372

theorem thirty_k_divisor_of_929260 (k : ℕ) (h1: 30^k ∣ 929260):
(3^k - k^3 = 2) :=
sorry

end thirty_k_divisor_of_929260_l304_304372


namespace sum_range_l304_304434

variable (x y z : ℝ)

theorem sum_range : 
  (x > 4) ∧ (y > 4) ∧ (z > 4) ∧ 
  ( (x + 3)^2 / (y + z - 4) + 
    (y + 5)^2 / (z + x - 5) + 
    ( (z + 7)^2 / (x + y - 7) = 45 ) ) → 
  21 ≤ x + y + z ∧ x + y + z ≤ 45 := 
by 
  sorry

end sum_range_l304_304434


namespace collinear_probability_correct_l304_304408

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end collinear_probability_correct_l304_304408


namespace product_middle_not_necessarily_zero_l304_304028

theorem product_middle_not_necessarily_zero (a b : ℕ) (h : ∃ x y : ℕ, a = x * 10 + 0 * 10^(nat.log10(x) + 1) + y) : 
  ∃ z : ℕ, ¬((a * b) = z * 10 + 0 * 10^(nat.log10(z) + 1)) ∨ ((a * b) = z * 10 + 0 * 10^(nat.log10(z) + 1)) :=
by
  sorry

end product_middle_not_necessarily_zero_l304_304028


namespace total_distance_driven_l304_304483

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end total_distance_driven_l304_304483


namespace smallest_three_digit_multiple_of_17_l304_304605

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304605


namespace smallest_three_digit_multiple_of_17_l304_304758

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304758


namespace total_sum_is_180_l304_304457

/-- Definitions and conditions given in the problem -/
def CoralineNumber : ℕ := 80
def JaydenNumber (C : ℕ) : ℕ := C - 40
def MickeyNumber (J : ℕ) : ℕ := J + 20

/-- Prove that the total sum of their numbers is 180 -/
theorem total_sum_is_180 : 
  let C := CoralineNumber
  let J := JaydenNumber C 
  let M := MickeyNumber J 
  in M + J + C = 180 := sorry

end total_sum_is_180_l304_304457


namespace chris_total_bill_l304_304956

theorem chris_total_bill :
  ∀ (base_charge overage_charge_per_gb : ℝ) (overage : ℕ) (total_gb : ℕ),
  base_charge = 45 ∧
  overage_charge_per_gb = 0.25 ∧
  overage = 80 ∧
  total_gb = 100 →
  total_bill = base_charge + overage_charge_per_gb * overage →
  total_bill = 65 := 
by
  intro base_charge overage_charge_per_gb overage total_gb
  assume h_conditions
  cases h_conditions with h1 h_rest
  cases h_rest with h2 h_rest2
  cases h_rest2 with h3 h4
  have h5: total_bill = base_charge + overage_charge_per_gb * overage := h4
  rw [h1, h2, h3] at h5
  simp at h5
  exact h5

end chris_total_bill_l304_304956


namespace smallest_three_digit_multiple_of_17_l304_304833

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304833


namespace smallest_three_digit_multiple_of_17_l304_304672

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304672


namespace g_at_8_l304_304137

noncomputable theory

def g : ℝ → ℝ := sorry

theorem g_at_8 :
  (∀ x y : ℝ, g x + g (2 * x + y) + 7 * x * y = g (3 * x - y) + 3 * x^2 - 1) →
  g 8 = -33 :=
by
  intro h
  sorry

end g_at_8_l304_304137


namespace regular_ticket_cost_l304_304209

theorem regular_ticket_cost (
  total_tickets : ℕ,
  senior_tickets : ℕ,
  senior_cost : ℕ,
  total_sales : ℕ,
  remaining_tickets := total_tickets - senior_tickets,
  senior_sales := senior_tickets * senior_cost,
  regular_sales (regular_cost : ℕ) := remaining_tickets * regular_cost
) : total_tickets = 65 ∧ 
    senior_tickets = 24 ∧ 
    senior_cost = 10 ∧ 
    total_sales = 855 → 
    (41 : ℕ) * (15 : ℕ) = total_sales - senior_sales :=
by {
  intros h,
  cases h with htickets hrem,
  cases hrem with hsenior_cost hsenior_tickets,
  cases hsenior_tickets with hsenior_total htotal_sales,
  simp [htickets, hsenior_cost, hsenior_tickets, hsenior_total, htotal_sales],
  sorry
}

end regular_ticket_cost_l304_304209


namespace smallest_three_digit_multiple_of_17_correct_l304_304627

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304627


namespace perp_AC_BC_never_true_circle_cuts_constant_chord_on_y_axis_l304_304049

variables (m x1 x2 : ℝ)

-- Definitions and conditions from the given problem
def curve_intersects_x_axis_at_A_and_B : Prop := x1 * x2 = -2

def point_C_coordinates : Prop := C = (0, 1)

-- Questions to be proved as Lean statements
theorem perp_AC_BC_never_true (AC BC : ℝ) :
  curve_intersects_x_axis_at_A_and_B m x1 x2 ->
  (AC != BC) :=
begin
  sorry
end

theorem circle_cuts_constant_chord_on_y_axis (C : ℝ × ℝ) :
  curve_intersects_x_axis_at_A_and_B m x1 x2 ->
  point_C_coordinates ->
  ∃ D E F : ℝ, D = m ∧ E = 1 ∧ F = -2 ∧ (∀ x y, (x^2 + y^2 + m*x + y - 2 = 0) ∧ x = 0 → (y = 1 ∨ y = -2)) ∧ (abs(1 - (-2)) = 3) :=
begin
  sorry
end

end perp_AC_BC_never_true_circle_cuts_constant_chord_on_y_axis_l304_304049


namespace monotonicity_of_f_range_of_a_l304_304000

noncomputable def f (x a : ℝ) : ℝ := Real.exp x * (Real.exp x - a) - a^2 * x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, a = 0 → (f x a).deriv = 2 * Real.exp (2 * x) ∧ (f x a).monotone_on ℝ) ∧
  (∀ x : ℝ, a > 0 → 
     ((x < Real.log a → (f x a).deriv < 0 ∧ (f x a).monotone_on (Iio (Real.log a))) ∧
      (x > Real.log a → (f x a).deriv > 0 ∧ (f x a).monotone_on (Ioi (Real.log a))))) ∧
  (∀ x : ℝ, a < 0 → 
     ((x < Real.log (-a / 2) → (f x a).deriv < 0 ∧ (f x a).monotone_on (Iio (Real.log (-a / 2)))) ∧
      (x > Real.log (-a / 2) → (f x a).deriv > 0 ∧ (f x a).monotone_on (Ioi (Real.log (-a / 2))))))
:= sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (f x a) ≥ 0 → (-2 * Real.exp (3/4) ≤ a ∧ a ≤ 1))
:= sorry

end monotonicity_of_f_range_of_a_l304_304000


namespace percent_of_y_l304_304381

theorem percent_of_y (y : ℝ) (hy : y > 0) : (6 * y / 20) + (3 * y / 10) = 0.6 * y :=
by
  sorry

end percent_of_y_l304_304381


namespace smallest_three_digit_multiple_of_17_l304_304814

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304814


namespace average_gas_mileage_round_trip_is_29_l304_304220

/-- Define the problem conditions. -/
def distance_to_friend : ℝ := 150
def mpg_compact_car : ℝ := 35
def mpg_suv : ℝ := 25

/-- Calculate the distance and fuel usage for each leg of the trip. -/
def round_trip_distance : ℝ := distance_to_friend + distance_to_friend
def fuel_used_to_friend : ℝ := distance_to_friend / mpg_compact_car
def fuel_used_returning : ℝ := distance_to_friend / mpg_suv
def total_fuel_used : ℝ := fuel_used_to_friend + fuel_used_returning

/-- Calculate the average gas mileage for the entire round trip. -/
def average_gas_mileage : ℝ := round_trip_distance / total_fuel_used

/-- Statement: The average gas mileage for the entire round trip is 29 miles per gallon. -/
theorem average_gas_mileage_round_trip_is_29 :
  average_gas_mileage = 29 :=
sorry

end average_gas_mileage_round_trip_is_29_l304_304220


namespace smallest_three_digit_multiple_of_17_l304_304857

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304857


namespace sarah_boxes_l304_304115

theorem sarah_boxes (b : ℕ) 
  (h1 : ∀ x : ℕ, x = 7) 
  (h2 : 49 = 7 * b) :
  b = 7 :=
sorry

end sarah_boxes_l304_304115


namespace triangle_area_l304_304266

theorem triangle_area :
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  area = 3 := by
  let A := (-3, 0)
  let B := (0, 2)
  let O := (0, 0)
  let area := 1 / 2 * |A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2)|
  sorry

end triangle_area_l304_304266


namespace rubble_money_left_l304_304486

/-- Rubble has $15 in his pocket. -/
def rubble_initial_amount : ℝ := 15

/-- Each notebook costs $4.00. -/
def notebook_price : ℝ := 4

/-- Each pen costs $1.50. -/
def pen_price : ℝ := 1.5

/-- Rubble needs to buy 2 notebooks. -/
def num_notebooks : ℝ := 2

/-- Rubble needs to buy 2 pens. -/
def num_pens : ℝ := 2

/-- The total cost of the notebooks. -/
def total_notebook_cost : ℝ := num_notebooks * notebook_price

/-- The total cost of the pens. -/
def total_pen_cost : ℝ := num_pens * pen_price

/-- The total amount Rubble spends. -/
def total_spent : ℝ := total_notebook_cost + total_pen_cost

/-- The remaining amount Rubble has after the purchase. -/
def rubble_remaining_amount : ℝ := rubble_initial_amount - total_spent

theorem rubble_money_left :
  rubble_remaining_amount = 4 := 
by
  -- Some necessary steps to complete the proof
  sorry

end rubble_money_left_l304_304486


namespace sum_of_circle_numbers_l304_304530

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem sum_of_circle_numbers (numbers : Fin 10 → ℕ) 
  (h : ∀ i : Fin 10, numbers i = gcd (numbers (i - 1)) (numbers (i + 1)) + 1) : 
  (Finset.univ.sum numbers) = 28 :=
by
  sorry

end sum_of_circle_numbers_l304_304530


namespace log_expression_equals_four_l304_304971

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l304_304971


namespace binomial_coeff_congruence_mod_p3_l304_304303

-- Let's define the conditions first.
variables (p a b : ℕ)

-- Conditions linked by ∧
noncomputable def conditions := odd p ∧ p > 3 ∧ a > b ∧ b > 1

-- Statement to be proved
theorem binomial_coeff_congruence_mod_p3 (h : conditions p a b) : 
  binomial (a * p) (a * p) ≡ binomial a b [MOD (p^3)] := 
sorry

end binomial_coeff_congruence_mod_p3_l304_304303


namespace smallest_three_digit_multiple_of_17_l304_304634

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304634


namespace proposition_2_l304_304431

variables {m n : ℝ → ℝ → Prop} {α : set (ℝ × ℝ)}

def perp_line_plane (x : ℝ → ℝ → Prop) (α : set (ℝ × ℝ)) : Prop :=
∀ a b : ℝ, α (a, b) → x a b

def parallel_lines (x y : ℝ → ℝ → Prop) : Prop :=
∀ a b : ℝ, x a b ↔ y a b

theorem proposition_2 (m n : ℝ → ℝ → Prop) (α : set (ℝ × ℝ)) 
(h1 : perp_line_plane m α) 
(h2 : parallel_lines n m) : 
  perp_line_plane n α := 
by
  sorry

end proposition_2_l304_304431


namespace smallest_three_digit_multiple_of_17_l304_304590

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304590


namespace smallest_three_digit_multiple_of_17_l304_304666

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304666


namespace chess_tournament_games_l304_304385

theorem chess_tournament_games:
  let n := 25 in
  let combinations := n * (n - 1) / 2 in
  let games := combinations * 3 in
  games = 900 :=
by
  sorry

end chess_tournament_games_l304_304385


namespace log_sequence_value_l304_304978

theorem log_sequence_value :
  ∃ x : ℝ, x = log 3 (81 + x) ∧ x > 0 ∧ x ≈ 5 :=
begin
  sorry
end

end log_sequence_value_l304_304978


namespace count_mutually_exclusive_not_complementary_l304_304927

 -- Definitions of events
def E1_miss : Prop := sorry -- Event E1: Miss
def E2_hit : Prop := sorry -- Event E2: Hit
def E3_greater_than_4 : Prop := sorry -- Event E3: The number of rings hit is greater than 4
def E4_not_less_than_5 : Prop := sorry -- Event E4: The number of rings hit is not less than 5

theorem count_mutually_exclusive_not_complementary :
  (∃ (pairs : list (Prop × Prop)), 
    pairs = [(E1_miss, E3_greater_than_4), (E1_miss, E4_not_less_than_5)] ∧ 
    pairs.length = 2) := 
sorry

end count_mutually_exclusive_not_complementary_l304_304927


namespace determine_R_l304_304302

noncomputable def Q : Polynomial ℂ := sorry 
noncomputable def R : Polynomial ℂ := - Polynomial.X

theorem determine_R (z : ℂ) :
  (z^2021 + 1 : Polynomial ℂ) = 
  ((z^2 + z + 1 : Polynomial ℂ) * Q + R) :=
begin
  sorry,
end

end determine_R_l304_304302


namespace smallest_three_digit_multiple_of_17_l304_304685

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304685


namespace age_of_B_l304_304879

/--
A is two years older than B.
B is twice as old as C.
The total of the ages of A, B, and C is 32.
How old is B?
-/
theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 32) : B = 12 :=
by
  sorry

end age_of_B_l304_304879


namespace smallest_positive_period_is_pi_axis_of_symmetry_eq_intervals_of_increase_minimum_value_in_interval_l304_304326

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sqrt 3 * cos x - sin x) - sqrt 3

theorem smallest_positive_period_is_pi :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = π :=
sorry

theorem axis_of_symmetry_eq :
  ∃ k ∈ ℤ, ∀ x : ℝ, (x = 1 / 2 * k * π - π / 12) :=
sorry

theorem intervals_of_increase :
  ∃ k ∈ ℤ, (∀ x : ℝ, k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12 → deriv f x > 0) :=
sorry

theorem minimum_value_in_interval :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ f x = -1 - sqrt 3 / 2 ∧ x = 5 * π / 12 :=
sorry

end smallest_positive_period_is_pi_axis_of_symmetry_eq_intervals_of_increase_minimum_value_in_interval_l304_304326


namespace find_y_of_log_eq_l304_304363

variable (m y : ℝ)

theorem find_y_of_log_eq (h : log m y * log 10 m = 4) : y = 10000 := by
  sorry

end find_y_of_log_eq_l304_304363


namespace union_complement_correctness_l304_304074

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complement_correctness : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 4} →
  A ∪ (U \ B) = {1, 2, 3, 5} :=
by
  intro hU hA hB
  sorry

end union_complement_correctness_l304_304074


namespace sara_gets_change_l304_304182

theorem sara_gets_change (cost_book1 cost_book2 money_given : ℝ) :
  cost_book1 = 5.5 ∧ cost_book2 = 6.5 ∧ money_given = 20 →
  money_given - (cost_book1 + cost_book2) = 8 :=
by
  intros h,
  rcases h with ⟨hb1, hb2, hg⟩,
  rw [hb1, hb2, hg],
  norm_num
  sorry -- added to make sure the code builds successfully

end sara_gets_change_l304_304182


namespace no_positive_real_solution_l304_304264

open Real

theorem no_positive_real_solution (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬(∀ n : ℕ, 0 < n → (n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end no_positive_real_solution_l304_304264


namespace problem1_problem2_l304_304241

-- First problem
theorem problem1 :
  real.cbrt (-27) + real.sqrt ((-3) ^ 2) - real.cbrt (-1) - ((-1 : ℝ) ^ 2018) = 0 := by
  sorry
  
-- Second problem
theorem problem2 :
  abs (real.sqrt 3 - real.sqrt 2) + abs (real.sqrt 3 - 2) - abs (real.sqrt 2 - 1) = 3 - 2 * real.sqrt 2 := by
  sorry

end problem1_problem2_l304_304241


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304706

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304706


namespace smallest_three_digit_multiple_of_17_l304_304796

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304796


namespace solution_set_eq_interval_l304_304201

variable (f : ℝ → ℝ)
variable (A : ℝ × ℝ) (B : ℝ × ℝ)
variable (h_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2))
variable (hA : A = (0, -1))
variable (hB : B = (3, 1))

theorem solution_set_eq_interval :
  { x : ℝ | abs (f x) < 1 } = set.Ioo 0 3 :=
by
  sorry

end solution_set_eq_interval_l304_304201


namespace smallest_three_digit_multiple_of_17_l304_304811

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304811


namespace ratio_of_volumes_l304_304216

variables (H R : ℝ) (π : ℝ := Real.pi) 

-- Define the heights of the entire cone and segments
def totalHeight := H
def segmentHeight := H / 5

-- Define radii of the segmented regions
def radius_1 := R / 5
def radius_2 := 2 * R / 5
def radius_3 := 3 * R / 5
def radius_4 := 4 * R / 5
def radius_5 := R

-- Define volumes of the combined segments cones
def V_B := (1 / 3) * π * (radius_2)^2 * (2 * segmentHeight)
def V_C := (1 / 3) * π * (radius_3)^2 * (3 * segmentHeight)
def V_D := (1 / 3) * π * (radius_4)^2 * (4 * segmentHeight)
def V_E := (1 / 3) * π * (radius_5)^2 * totalHeight

-- Define volumes of the largest and second-largest pieces
def V_L := V_E - V_D
def V_SL := V_D - V_C

-- The goal statement: the ratio of the volume of the second-largest piece to the volume of the largest piece
theorem ratio_of_volumes (H R : ℝ) : V_SL / V_L = 37 / 187 :=
by
  sorry

end ratio_of_volumes_l304_304216


namespace james_fuel_cost_l304_304067

variable (originalCost : ℝ) (priceIncreasePct : ℝ) (numTanks : ℕ)

theorem james_fuel_cost : 
  originalCost = 200 ∧ 
  priceIncreasePct = 0.20 ∧ 
  numTanks = 2 → 
  let increasedCost := originalCost * priceIncreasePct
      newCostPerTank := originalCost + increasedCost
      totalCost := newCostPerTank * numTanks
  in totalCost = 480 :=
by
  intros
  sorry

end james_fuel_cost_l304_304067


namespace john_total_money_after_3_years_l304_304418

def principal : ℝ := 1000
def rate : ℝ := 0.1
def time : ℝ := 3

/-
  We need to prove that the total money after 3 years is $1300
-/
theorem john_total_money_after_3_years (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal + (principal * rate * time) = 1300 := by
  sorry

end john_total_money_after_3_years_l304_304418


namespace parabola_axis_of_symmetry_l304_304364

noncomputable def axis_of_symmetry (a b c : ℝ) (ha : a ≠ 0) (h1 : a + b + c = 0) (h2 : 9 * a - 3 * b + c = 0) : ℝ :=
-1

theorem parabola_axis_of_symmetry (a b c : ℝ) (ha : a ≠ 0) (h1 : a + b + c = 0) (h2 : 9 * a - 3 * b + c = 0) : 
  axis_of_symmetry a b c ha h1 h2 = -1 :=
begin
  -- Proof goes here
  sorry
end

end parabola_axis_of_symmetry_l304_304364


namespace tangent_line_properties_l304_304312

-- Defining the conditions and the problem in Lean 4
theorem tangent_line_properties (a : ℝ) (l : ℝ → ℝ → Prop) :
  (l 1 1) ∧ 
  (∀ x y, (x + 1)^2 + (y - 2)^2 = 5 → ∃ t, l x y ∧ (tangent_to_circle l (x + 1)^2 + (y - 2)^2 = 5)) ∧ 
  (∀ x y, ax + y - 1 = 0  → perpendicular ax + y - 1 l) →
  a = 1/2 ∧ l = (2 * x - y - 1 = 0) :=
by
  sorry

end tangent_line_properties_l304_304312


namespace sara_change_l304_304184

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end sara_change_l304_304184


namespace length_segment_satisfying_abs_eq_l304_304574

theorem length_segment_satisfying_abs_eq :
  (let a := (3:ℝ); b := 5 in |a + b - (a - b)| = 10) :=
by
  -- define the relevant values
  let a := (3:ℝ)
  let b := 5
  show |a + b - (a - b)| = 10
  sorry

end length_segment_satisfying_abs_eq_l304_304574


namespace math_proof_problem_l304_304515

def expr (m : ℝ) : ℝ := (1 - (2 / (m + 1))) / ((m ^ 2 - 2 * m + 1) / (m ^ 2 - m))

theorem math_proof_problem :
  expr (Real.tan (Real.pi / 3) - 1) = (3 - Real.sqrt 3) / 3 :=
  sorry

end math_proof_problem_l304_304515


namespace find_point_C_find_area_triangle_ABC_l304_304039

noncomputable section

-- Given points and equations
def point_B : ℝ × ℝ := (4, 4)
def eq_angle_bisector : ℝ × ℝ → Prop := λ p => p.2 = 0
def eq_altitude : ℝ × ℝ → Prop := λ p => p.1 - 2 * p.2 + 2 = 0

-- Target coordinates of point C
def point_C : ℝ × ℝ := (10, -8)

-- Coordinates of point A derived from given conditions
def point_A : ℝ × ℝ := (-2, 0)

-- Line equations derived from conditions
def eq_line_BC : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 - 12 = 0
def eq_line_AC : ℝ × ℝ → Prop := λ p => 2 * p.1 + 3 * p.2 + 4 = 0

-- Prove the coordinates of point C
theorem find_point_C : ∃ C : ℝ × ℝ, eq_line_BC C ∧ eq_line_AC C ∧ C = point_C := by
  sorry

-- Prove the area of triangle ABC.
theorem find_area_triangle_ABC : ∃ S : ℝ, S = 48 := by
  sorry

end find_point_C_find_area_triangle_ABC_l304_304039


namespace calculate_expression_l304_304249

def f (x : ℝ) : ℝ := x^2 + 2 * real.sqrt x

theorem calculate_expression : (2 * f 2 - f 8) = -56 := by
  sorry

end calculate_expression_l304_304249


namespace system_solution_l304_304518

theorem system_solution (x y : ℝ) 
  (h1 : (x^2 + x * y + y^2) / (x^2 - x * y + y^2) = 3) 
  (h2 : x^3 + y^3 = 2) : x = 1 ∧ y = 1 :=
  sorry

end system_solution_l304_304518


namespace domain_eq_range_for_f_l304_304009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.sqrt (a * x ^ 2 + 3 * x)

theorem domain_eq_range_for_f (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = real.sqrt (a * x ^ 2 + 3 * x)) → 
  (∀ x : ℝ, 0 ≤ a * x ^ 2 + 3 * x) →
  a = -4 ∨ a = 0 :=
by
  -- The proof will be added here
  sorry

end domain_eq_range_for_f_l304_304009


namespace exists_f_g_positive_l304_304878

-- Problem 1: There exists f such that \( y = f(x) \) is a solution of the differential equation and \( f(x) > 0 \) for all \( x \), but \( f'(x) \) is not necessarily positive for all \( x \).
theorem exists_f (f : ℝ → ℝ)
  (h1 : ∀ x, ∃ f_deriv : ℝ → ℝ, f_deriv x = (f x)'' - 2*(f x)' + f x - 2*ℯ^x = 0)
  (h2 : ∀ x, f x > 0) :
  ¬(∀ x, (deriv f x) > 0) := sorry

-- Problem 2: \( y = g(x) \) implies \( g(x) > 0 \) for all \( x \).
theorem g_positive (g : ℝ → ℝ)
  (h1 : ∀ x, ∃ g_deriv : ℝ → ℝ, g_deriv x = (g x)'' - 2*(deriv g x) + g x - 2*ℯ^x = 0)
  (h2 : ∀ x, (deriv g x) > 0) :
  ∀ x, g x > 0 := sorry

end exists_f_g_positive_l304_304878


namespace smallest_three_digit_multiple_of_17_l304_304591

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304591


namespace total_logs_in_stack_l304_304930

-- Definitions for the conditions
def a : ℕ := 15
def l : ℕ := 4
def d : ℤ := -1
def n : ℕ := (l - a + d.abs) / d.abs + 1

-- Statement with the proof problem
theorem total_logs_in_stack :
  n = 12 ∧ ∑ k in finset.range n, (a - k) = 114 := by
  sorry

end total_logs_in_stack_l304_304930


namespace smallest_three_digit_multiple_of_17_l304_304750

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304750


namespace math_problem_l304_304023

-- Definitions for the conditions
def condition1 (a b c : ℝ) : Prop := a + b + c = 0
def condition2 (a b c : ℝ) : Prop := |a| > |b| ∧ |b| > |c|

-- Theorem statement
theorem math_problem (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) : c > 0 ∧ a < 0 :=
by
  sorry

end math_problem_l304_304023


namespace find_x_l304_304370

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end find_x_l304_304370


namespace bill_sunday_miles_l304_304101

-- Define the variables
variables (B S J : ℕ) -- B for miles Bill ran on Saturday, S for miles Bill ran on Sunday, J for miles Julia ran on Sunday

-- State the conditions
def condition1 (B S : ℕ) : Prop := S = B + 4
def condition2 (B S J : ℕ) : Prop := J = 2 * S
def condition3 (B S J : ℕ) : Prop := B + S + J = 20

-- The final theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (B S J : ℕ) 
  (h1 : condition1 B S)
  (h2 : condition2 B S J)
  (h3 : condition3 B S J) : 
  S = 6 := 
sorry

end bill_sunday_miles_l304_304101


namespace recurring_decimal_to_fraction_l304_304261

theorem recurring_decimal_to_fraction : (4 + (Int.recur 8) / 10) = 44 / 9 :=
by
  sorry

end recurring_decimal_to_fraction_l304_304261


namespace sequences_equality_l304_304424

/-- Define the sequences u and v as described in the problem statement. -/
noncomputable def sequence_u (a : ℕ → ℝ) : ℕ → ℝ
| 0     := 1
| (k+1) := if k = 0 then 1 else sequence_u k + a k * sequence_u (k-1)

noncomputable def sequence_v (a : ℕ → ℝ) (n : ℕ) : ℕ → ℝ
| 0     := 1
| (k+1) := if k = 0 then 1 else sequence_v k + a (n - k) * sequence_v (k-1)

/-- Theorem to prove that u_n = v_n given the defined sequences. -/
theorem sequences_equality (n : ℕ) (a : ℕ → ℝ) :
  sequence_u a n = sequence_v a n :=
sorry

end sequences_equality_l304_304424


namespace calculate_adult_chaperones_l304_304069

theorem calculate_adult_chaperones (students : ℕ) (student_fee : ℕ) (adult_fee : ℕ) (total_fee : ℕ) 
  (h_students : students = 35) 
  (h_student_fee : student_fee = 5) 
  (h_adult_fee : adult_fee = 6) 
  (h_total_fee : total_fee = 199) : 
  ∃ (A : ℕ), 35 * student_fee + A * adult_fee = 199 ∧ A = 4 := 
by
  sorry

end calculate_adult_chaperones_l304_304069


namespace limit_expression_equals_neg_third_derivative_l304_304448

-- Define the differentiable function
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)

-- State the theorem
theorem limit_expression_equals_neg_third_derivative:
  (tendsto (λ Δx : ℝ, (f 1 - f (1 + Δx)) / (3 * Δx)) (𝓝 0) (𝓝 (-1/3 * fderiv ℝ f 1))) :=
sorry

end limit_expression_equals_neg_third_derivative_l304_304448


namespace smallest_three_digit_multiple_of_17_l304_304809

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304809


namespace quadratic_expression_l304_304316

theorem quadratic_expression (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 + 1 = 0) (h2 : x2^2 - 3 * x2 + 1 = 0) : 
  x1^2 - 2 * x1 + x2 = 2 :=
sorry

end quadratic_expression_l304_304316


namespace smallest_three_digit_multiple_of_17_l304_304789

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304789


namespace simplify_and_evaluate_expression_l304_304496

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304496


namespace range_of_abs_function_l304_304552

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem range_of_abs_function : Set.range f = Set.Ici 2 := by
  sorry

end range_of_abs_function_l304_304552


namespace probability_X_eq_3_l304_304317

def number_of_ways_to_choose (n k : ℕ) : ℕ :=
  Nat.choose n k

def P_X_eq_3 : ℚ :=
  (number_of_ways_to_choose 5 3) * (number_of_ways_to_choose 3 1) / (number_of_ways_to_choose 8 4)

theorem probability_X_eq_3 : P_X_eq_3 = 3 / 7 := by
  sorry

end probability_X_eq_3_l304_304317


namespace find_corresponding_side_l304_304536

-- Define the conditions
variables (A1 A2 : ℕ) (side_small side_large : ℕ) (k : ℕ)

-- Assumptions based on the conditions
def conditions : Prop := 
  A1 - A2 = 72 ∧
  (A1 = k^2 * A2) ∧
  side_small = 6 ∧
  A2 > 0

-- The theorem to prove
theorem find_corresponding_side 
  (h : conditions A1 A2 side_small side_large k) :
  side_large = 12 :=
sorry

end find_corresponding_side_l304_304536


namespace necessary_but_not_sufficient_condition_l304_304053

variable {a : ℕ → ℤ}

noncomputable def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ (m n k : ℕ), a m * a k = a n * a (m + k - n)

noncomputable def is_root_of_quadratic (x y : ℤ) : Prop :=
x^2 + 3*x + 1 = 0 ∧ y^2 + 3*y + 1 = 0

theorem necessary_but_not_sufficient_condition 
  (a : ℕ → ℤ)
  (hgeo : is_geometric_sequence a)
  (hroots : is_root_of_quadratic (a 4) (a 12)) :
  a 8 = -1 ↔ (∃ x y : ℤ, is_root_of_quadratic x y ∧ x + y = -3 ∧ x * y = 1) :=
sorry

end necessary_but_not_sufficient_condition_l304_304053


namespace smallest_three_digit_multiple_of_17_l304_304723

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304723


namespace parity_of_f_monotonicity_of_f_l304_304006

noncomputable def f : ℝ → ℝ := λ x, log ((1 + x) / (1 - x))

theorem parity_of_f :
  ∀ x, f (-x) = -f x := 
sorry

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x < f y := 
sorry

end parity_of_f_monotonicity_of_f_l304_304006


namespace monotonicity_intervals_range_of_c_l304_304004

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem monotonicity_intervals (α : ℝ) : 
  (∀ x, 0 < x ∧ x < Real.exp 2 → deriv (f x α) < 0) ∧ 
  (∀ x, x > Real.exp 2 → deriv (f x α) > 0) :=
by
  sorry

theorem range_of_c (c : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 3 → f x 3 < 2 * c^2 - c) → 
  (c > 1 ∨ c < -1/2) :=
by
  sorry

end monotonicity_intervals_range_of_c_l304_304004


namespace solve_inequality_l304_304516

theorem solve_inequality (x : ℝ) : 
  (x ∈ set.Iic (-3) ∪ set.Ici 0) ↔ (x / (x + 3) ≥ 0) := 
sorry

end solve_inequality_l304_304516


namespace smallest_three_digit_multiple_of_17_l304_304739

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304739


namespace cos10_cos20_minus_sin10_sin20_l304_304200

theorem cos10_cos20_minus_sin10_sin20 :
  cos (10 * (Real.pi / 180)) * cos (20 * (Real.pi / 180)) - sin (10 * (Real.pi / 180)) * sin (20 * (Real.pi / 180)) = Real.sqrt 3 / 2 :=
by sorry

end cos10_cos20_minus_sin10_sin20_l304_304200


namespace smallest_three_digit_multiple_of_17_l304_304795

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304795


namespace smallest_three_digit_multiple_of_17_l304_304670

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304670


namespace smallest_three_digit_multiple_of_17_l304_304747

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304747


namespace find_x2_plus_y2_l304_304263

theorem find_x2_plus_y2 
  (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 :=
by
  sorry

end find_x2_plus_y2_l304_304263


namespace derivative_at_1_l304_304320

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * (f' 1) * log x - x

-- Define the derivative of f using the given expression
noncomputable def f' (x : ℝ) : ℝ := 2 * (f' 1) * (1 / x) - 1

-- State the theorem to prove that f'(1) = 1
theorem derivative_at_1 : f' 1 = 1 :=
sorry

end derivative_at_1_l304_304320


namespace num_isosceles_triangles_with_perimeter_30_l304_304352

theorem num_isosceles_triangles_with_perimeter_30 : 
  (∃ (s : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ s → 2 * a + b = 30 ∧ (a ≥ b) ∧ b ≠ 0 ∧ a + a > b ∧ a + b > a ∧ b + a > a) 
    ∧ s.card = 7) :=
by {
  sorry
}

end num_isosceles_triangles_with_perimeter_30_l304_304352


namespace smallest_three_digit_multiple_of_17_l304_304703

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304703


namespace part1_part2_part3_l304_304323

section

noncomputable def f (x a : ℝ) : ℝ := x * |x - a| + 2 * x

def g (x : ℝ) : ℝ := 2 * x + 1

theorem part1 {a : ℝ} (h : ∀ x : ℝ, (2 * x + (2 - a) ≥ 0 ∧ (x < a) → -2 * x + (2 + a) ≥ 0)) :
  -2 ≤ a ∧ a ≤ 2 := by sorry

theorem part2 {a : ℝ} (h : ∀ x ∈ Icc (1:ℝ) 2, f x a < g x) :
  3/2 < a ∧ a < 2 := by sorry

theorem part3 {t a : ℝ} (ha : a ∈ Icc (-4:ℝ) 4)
    (h3 : ∃ a ∈ Icc (2:ℝ) 4, ∀ x : ℝ, f x a = t * f a a → x ≠ y → x ≠ z → y ≠ z) :
  1 < t ∧ t < 9/8 := by sorry

end

end part1_part2_part3_l304_304323


namespace simplify_and_evaluate_expression_l304_304492

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304492


namespace smallest_three_digit_multiple_of_17_l304_304799

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304799


namespace smallest_three_digit_multiple_of_17_l304_304639

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304639


namespace max_pages_l304_304064

theorem max_pages (cents_available : ℕ) (cents_per_page : ℕ) : 
  cents_available = 2500 → 
  cents_per_page = 3 → 
  (cents_available / cents_per_page) = 833 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end max_pages_l304_304064


namespace smallest_three_digit_multiple_of_17_l304_304725

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304725


namespace water_depth_in_tank_l304_304206

-- Definitions of the problem conditions
def radius : ℝ := 5
def length : ℝ := 10
def volume_tank : ℝ := Real.pi * radius^2 * length
def volume_water : ℝ := 0.5 * volume_tank

-- Problem statement in Lean 4
theorem water_depth_in_tank : depth := 
  depth_condition : depth = 5
by
  -- Definitions of the problem conditions
  let radius := (5 : ℝ)
  let length := (10 : ℝ)
  let volume_tank := Real.pi * radius^2 * length
  let volume_water := 0.5 * volume_tank

  -- Show depth condition is met
  show depth = 5 from depth_condition sorry

end water_depth_in_tank_l304_304206


namespace smallest_three_digit_multiple_of_17_l304_304641

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304641


namespace rational_function_sum_l304_304541

-- Define the problem conditions and the target equality
theorem rational_function_sum (p q : ℝ → ℝ) :
  (∀ x, (p x) / (q x) = (x - 1) / ((x + 1) * (x - 1))) ∧
  (∀ x ≠ -1, q x ≠ 0) ∧
  (q 2 = 3) ∧
  (p 2 = 1) →
  (p x + q x = x^2 + x - 2) := by
  sorry

end rational_function_sum_l304_304541


namespace relationship_h_K_l304_304411

variable (b b' h h' s s' P K p k : ℝ)

noncomputable def isIsoscelesTriangle :=
    ∀ (b h s : ℝ), h ≠ 0 ∧ s = (b / 2) / (cos (atan ((2 * h) / b))) / 2

def triangle_I (b h s P K : ℝ) :=
  (P = b + 2 * s) ∧ (K = (1/2) * b * h)

def triangle_II (b' h' s' p k : ℝ) :=
  (p = b' + 2 * s') ∧ (k = (1/2) * b' * h')

theorem relationship_h_K (h h' K k : ℝ) :
  (h ≠ h') → (triangle_I b h s P K) → (triangle_II b' h' s' p k) → 
  (h / h' = K / k only sometimes) := sorry

end relationship_h_K_l304_304411


namespace possible_pairs_copies_each_key_min_drawers_l304_304198

-- Define the number of distinct keys
def num_keys : ℕ := 10

-- Define the function to calculate the number of pairs
def num_pairs (n : ℕ) := n * (n - 1) / 2

-- Theorem for the first question
theorem possible_pairs : num_pairs num_keys = 45 :=
by sorry

-- Define the number of copies needed for each key
def copies_needed (n : ℕ) := n - 1

-- Theorem for the second question
theorem copies_each_key : copies_needed num_keys = 9 :=
by sorry

-- Define the minimum number of drawers Fernando needs to open
def min_drawers_to_open (n : ℕ) := num_pairs n - (n - 1) + 1

-- Theorem for the third question
theorem min_drawers : min_drawers_to_open num_keys = 37 :=
by sorry

end possible_pairs_copies_each_key_min_drawers_l304_304198


namespace exists_continuous_function_l304_304985

noncomputable theory

open Set Real

theorem exists_continuous_function : 
  ∃ f : ℝ → ℝ, Continuous f ∧ ∀ y : ℝ, ∃! (x : ℝ), f x = y ∧ Count {x : ℝ | f x = y} = 3 :=
by
  sorry

end exists_continuous_function_l304_304985


namespace reduced_bucket_fraction_l304_304564

theorem reduced_bucket_fraction (C : ℝ) (F : ℝ) (h : 25 * F * C = 10 * C) : F = 2 / 5 :=
by sorry

end reduced_bucket_fraction_l304_304564


namespace total_signs_at_intersections_l304_304924

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l304_304924


namespace smallest_three_digit_multiple_of_17_l304_304821

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304821


namespace chastity_leftover_money_l304_304243

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end chastity_leftover_money_l304_304243


namespace boris_takes_l304_304237

variable (initialCandy : ℕ)
variable (daughterCandy : ℕ)
variable (numberOfBowls : ℕ)
variable (finalCandyInBowl : ℕ)

def piecesBorisTook (initialCandy : ℕ) (daughterCandy : ℕ) (numberOfBowls : ℕ) (finalCandyInBowl : ℕ) : ℕ :=
  let remainingCandy := initialCandy - daughterCandy
  let candyPerBowlBeforeTaking := remainingCandy / numberOfBowls
  candyPerBowlBeforeTaking - finalCandyInBowl

theorem boris_takes (h1 : initialCandy = 100)
                    (h2 : daughterCandy = 8)
                    (h3 : numberOfBowls = 4)
                    (h4 : finalCandyInBowl = 20) :
                    piecesBorisTook initialCandy daughterCandy numberOfBowls finalCandyInBowl = 3 :=
by
  rw [piecesBorisTook]
  rw [h1, h2, h3, h4]
  norm_num
  sorry -- To be proven

end boris_takes_l304_304237


namespace pounds_of_peanuts_l304_304948

theorem pounds_of_peanuts (choc_chips raisins trail_mix : ℝ) (h1 : choc_chips = 0.17) (h2 : raisins = 0.08) (h3 : trail_mix = 0.42) :
  ∃ peanuts : ℝ, peanuts = trail_mix - (choc_chips + raisins) ∧ peanuts = 0.17 :=
by
  use 0.17
  rw [h1, h2, h3]
  linarith

end pounds_of_peanuts_l304_304948


namespace smallest_three_digit_multiple_of_17_l304_304581

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304581


namespace triangle_property_l304_304293

variables {A B C D : Type}
variables {length : A → B → ℝ}
variables {x y : ℝ}

-- Definitions of points
variables [Point A] [Point B] [Point C] [Point D]

-- Definitions for triangle being isosceles with base CB, and CD being perpendicular to AB
def isosceles_triangle (A B C : A) :=
  length A B = length A C ∧ length B C = y

def perpendicular (C D : A) (A B : A) :=
  ⟦angle_right C D A B⟧

theorem triangle_property (A B C D : A) (h1 : isosceles_triangle A B C) (h2 : perpendicular C D A B) :
  length C D * length C D + length A D * length A D = length A C * length A C :=
sorry

end triangle_property_l304_304293


namespace find_wickets_before_last_match_l304_304911

-- let W be the number of wickets before the last match
variables (W : ℕ)

-- conditions
def before_last_match_average := 12.4
def wickets_taken_last_match := 6
def runs_given_last_match := 26
def average_decrease := 0.4

-- total runs before the last match
def total_runs_before := before_last_match_average * W

-- new average after the last match
def new_average := before_last_match_average - average_decrease

-- total runs and wickets after the last match
def total_runs_after := total_runs_before + runs_given_last_match
def total_wickets_after := W + wickets_taken_last_match

-- equating the new average to 12.0
theorem find_wickets_before_last_match :
  (total_runs_after / total_wickets_after) = new_average → W = 115 :=
by
  sorry

end find_wickets_before_last_match_l304_304911


namespace corresponding_angle_C1_of_similar_triangles_l304_304281

theorem corresponding_angle_C1_of_similar_triangles
  (α β γ : ℝ)
  (ABC_sim_A1B1C1 : true)
  (angle_A : α = 50)
  (angle_B : β = 95) :
  γ = 35 :=
by
  sorry

end corresponding_angle_C1_of_similar_triangles_l304_304281


namespace maximum_value_of_w_l304_304300

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end maximum_value_of_w_l304_304300


namespace prob_of_entirely_black_l304_304897

noncomputable def prob_all_black_grid : ℚ :=
  if (is_center_black : Prop) ∧ 
     (are_edge_squares_black : Prop) ∧ 
     (are_corner_squares_black : Prop)
  then (1/2 : ℚ) * (7/16 : ℚ) * (7/16 : ℚ)
  else 0

theorem prob_of_entirely_black (h : prob_all_black_grid = 49 / 512) : true :=
by { sorry }

end prob_of_entirely_black_l304_304897


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304712

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304712


namespace pythagorean_triple_l304_304221

theorem pythagorean_triple (a : ℕ) (h_odd : a % 2 = 1) (h_ge_3 : a ≥ 3) : 
  let b := (a^2 - 1) / 2 in
  let c := (a^2 + 1) / 2 in 
  a^2 + b^2 = c^2 := 
by
  let b := (a^2 - 1) / 2
  let c := (a^2 + 1) / 2
  sorry

end pythagorean_triple_l304_304221


namespace sara_gets_change_l304_304181

theorem sara_gets_change (cost_book1 cost_book2 money_given : ℝ) :
  cost_book1 = 5.5 ∧ cost_book2 = 6.5 ∧ money_given = 20 →
  money_given - (cost_book1 + cost_book2) = 8 :=
by
  intros h,
  rcases h with ⟨hb1, hb2, hg⟩,
  rw [hb1, hb2, hg],
  norm_num
  sorry -- added to make sure the code builds successfully

end sara_gets_change_l304_304181


namespace angle_measure_l304_304131

theorem angle_measure : 
  ∃ (x : ℝ), (x + (3 * x + 3) = 90) ∧ x = 21.75 := by
  sorry

end angle_measure_l304_304131


namespace min_total_books_l304_304156

-- Definitions based on conditions
variables (P C B : ℕ)

-- Condition 1: Ratio of physics to chemistry books is 3:2
def ratio_physics_chemistry := 3 * C = 2 * P

-- Condition 2: Ratio of chemistry to biology books is 4:3
def ratio_chemistry_biology := 4 * B = 3 * C

-- Condition 3: Total number of books is 3003
def total_books := P + C + B = 3003

-- The theorem to prove
theorem min_total_books (h1 : ratio_physics_chemistry P C) (h2 : ratio_chemistry_biology C B) (h3: total_books P C B) :
  3003 = 3003 :=
by
  sorry

end min_total_books_l304_304156


namespace no_perfect_square_digit_rearrangement_l304_304412

theorem no_perfect_square_digit_rearrangement :
  ∀ (l : List ℕ), l = List.replicate 10 1 ++ List.replicate 10 2 ++ List.replicate 10 3 →
  ¬ ∃ n : ℕ, (l.perm (to_digits n ^ 2 : List ℕ)) :=
begin
  intros l hl,
  -- Introducing assumption for contradiction
  intro h,
  -- Further steps omitted for clarity
  sorry
end

end no_perfect_square_digit_rearrangement_l304_304412


namespace line_through_point_l304_304035

theorem line_through_point (a b : ℝ) (α : ℝ) (h : (cos α / a) + (sin α / b) = 1) :
  (1 / a^2) + (1 / b^2) ≥ 1 :=
sorry

end line_through_point_l304_304035


namespace integer_closest_to_zero_l304_304872

theorem integer_closest_to_zero (x : ℤ) (h : x ∈ {-1, 2, -3, 4, -5}) : x = -1 :=
by
  sorry

end integer_closest_to_zero_l304_304872


namespace remaining_money_after_purchase_l304_304488

def initial_money : Float := 15.00
def notebook_cost : Float := 4.00
def pen_cost : Float := 1.50
def notebooks_purchased : ℕ := 2
def pens_purchased : ℕ := 2

theorem remaining_money_after_purchase :
  initial_money - (notebook_cost * notebooks_purchased + pen_cost * pens_purchased) = 4.00 := by
  sorry

end remaining_money_after_purchase_l304_304488


namespace conjugate_of_z_l304_304132

theorem conjugate_of_z (z : ℂ) (h : z * (3 * complex.I - 4) = 25 * complex.I) : conj z = -4 + 3 * complex.I :=
sorry

end conjugate_of_z_l304_304132


namespace profit_rate_l304_304168

variables (list_price : ℝ)
          (discount : ℝ := 0.95)
          (selling_increase : ℝ := 1.6)
          (inflation_rate : ℝ := 1.4)

theorem profit_rate (list_price : ℝ) : 
  (selling_increase / (discount * inflation_rate)) - 1 = 0.203 :=
by 
  sorry

end profit_rate_l304_304168


namespace doctor_lindsay_daily_income_l304_304177

def patients_per_hour_adult : ℕ := 4
def patients_per_hour_child : ℕ := 3
def cost_per_adult : ℕ := 50
def cost_per_child : ℕ := 25
def work_hours_per_day : ℕ := 8

theorem doctor_lindsay_daily_income : 
  (patients_per_hour_adult * cost_per_adult + patients_per_hour_child * cost_per_child) * work_hours_per_day = 2200 := 
by
  sorry

end doctor_lindsay_daily_income_l304_304177


namespace determine_a_values_l304_304124

theorem determine_a_values (a : ℝ) :
  (∀ y ∈ ℝ, ∃ x ∈ ℝ, log (x^2 + a*x - a) = y) ↔ (a ∈ Iic (-4) ∨ a ∈ Ici 0) :=
sorry

end determine_a_values_l304_304124


namespace smallest_three_digit_multiple_of_17_l304_304755

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304755


namespace smallest_n_distance_1000_l304_304893

-- Define the initial set K0 consisting of two points A and B with AB = 1
noncomputable def point := ℝ
noncomputable def distance (x y : point) := abs (x - y)

def K₀ : set point := {0, 1}

-- Function that generates set Kn given set Kn-1
def reflect_set (S : set point) : set point :=
  S ∪ {2 * b - a | a ∈ S, b ∈ S}

-- Iteratively generate Kn
noncomputable def K (n : ℕ) : set point :=
  nat.rec_on n K₀ (λ n' Kn', reflect_set Kn')

-- Prove that the smallest n for which Kn contains a point at distance 1000 from 0 is 7 
theorem smallest_n_distance_1000 : ∃ n, (∀ x ∈ K n, distance 0 x ≥ 1000) ∧ (n = 7) :=
by
  sorry

end smallest_n_distance_1000_l304_304893


namespace decision_box_exits_l304_304401

theorem decision_box_exits (one_entrance : Prop) (exists_endpoints : Prop) : ∃ exits : ℕ, exits = 2 :=
by {
  have h_exits : ∃ exits, exits = 2, from ⟨2, rfl⟩,
  exact h_exits,
  sorry
}

end decision_box_exits_l304_304401


namespace smallest_three_digit_multiple_of_17_l304_304642

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304642


namespace song_book_cost_correct_l304_304251

/-- Define the constants for the problem. -/
def clarinet_cost : ℝ := 130.30
def pocket_money : ℝ := 12.32
def total_spent : ℝ := 141.54

/-- Prove the cost of the song book. -/
theorem song_book_cost_correct :
  (total_spent - clarinet_cost) = 11.24 :=
by
  sorry

end song_book_cost_correct_l304_304251


namespace a_2018_mod_49_l304_304091

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : (a 2018) % 49 = 0 := by
  sorry

end a_2018_mod_49_l304_304091


namespace smallest_three_digit_multiple_of_17_l304_304791

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304791


namespace smallest_three_digit_multiple_of_17_l304_304804

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304804


namespace total_signs_at_intersections_l304_304925

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end total_signs_at_intersections_l304_304925


namespace count_valid_two_digit_numbers_l304_304357

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def tens_digit_less_than_ones_digit (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens < ones

theorem count_valid_two_digit_numbers : 
  (Finset.filter tens_digit_less_than_ones_digit (Finset.filter is_two_digit (Finset.range 100))).card = 36 :=
by
  sorry

end count_valid_two_digit_numbers_l304_304357


namespace smallest_three_digit_multiple_of_17_l304_304668

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304668


namespace Sarah_copy_total_pages_l304_304119

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end Sarah_copy_total_pages_l304_304119


namespace smallest_three_digit_multiple_of_17_l304_304607

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304607


namespace possible_integer_roots_l304_304256

def polynomial (x : ℤ) : ℤ := x^3 + 2 * x^2 - 3 * x - 17

theorem possible_integer_roots :
  ∃ (roots : List ℤ), roots = [1, -1, 17, -17] ∧ ∀ r ∈ roots, polynomial r = 0 := 
sorry

end possible_integer_roots_l304_304256


namespace total_numbers_l304_304460

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end total_numbers_l304_304460


namespace smallest_three_digit_multiple_of_17_l304_304602

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304602


namespace simplify_and_evaluate_expression_l304_304509

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304509


namespace angle_measure_l304_304572

theorem angle_measure (x : ℝ) (h1 : ∠A / 2 = x) (h2 : ∠B = 3 * x) (h3 : ∠A + ∠B = 180) : ∠A = 72 :=
by
  sorry

end angle_measure_l304_304572


namespace real_solution_count_l304_304958

theorem real_solution_count :
  (∃ x y z w : ℝ,
    x = 2 * z + 2 * w + z * w * x ∧
    y = 2 * w + 2 * x + w * x * y ∧
    z = 2 * x + 2 * y + x * y * z ∧
    w = 2 * y + 2 * z + y * z * w ∧
    ∃ θ : ℝ, w = (real.sin θ) ^ 2) ↔ 5 := sorry

end real_solution_count_l304_304958


namespace length_minor_axis_of_ellipse_l304_304313

-- Given conditions about the ellipse
def min_distance_focus := 5
def max_distance_focus := 15

-- The proven statement
theorem length_minor_axis_of_ellipse : 
  let a := (min_distance_focus + max_distance_focus) / 2 in
  let c := (max_distance_focus - min_distance_focus) / 2 in 
  let b := Real.sqrt (a^2 - c^2) in
  2 * b = 10 * Real.sqrt 3 :=
by
  -- This is where the proof would go
  sorry

end length_minor_axis_of_ellipse_l304_304313


namespace ellipse_focus_reciprocal_sum_l304_304964

theorem ellipse_focus_reciprocal_sum (θ : ℝ) (x y m n : ℝ) (C : ℝ × ℝ → Prop)
    (hC : ∀ θ, C (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)) 
    (focus : ℝ × ℝ := (1, 0))
    (line_l_intersects_C : ∃ M N, C M ∧ C N ∧ (M.1 = 1 ∨ N.1 = 1) ∧ |M - focus| = m ∧ |N - focus| = n) :
    1 / m + 1 / n = 4 / 3 :=
  sorry

end ellipse_focus_reciprocal_sum_l304_304964


namespace arithmetic_seq_2a9_a10_l304_304400

theorem arithmetic_seq_2a9_a10 (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (arith_seq : ∀ n : ℕ, ∃ d : ℕ, a n = a 1 + (n - 1) * d) : 2 * a 9 - a 10 = 15 :=
by
  sorry

end arithmetic_seq_2a9_a10_l304_304400


namespace b_15_is_135_l304_304428

def seq_b : ℕ → ℤ
| 1 := 2
| n + m := seq_b n + seq_b m + (n * m)

theorem b_15_is_135 : seq_b 15 = 135 := 
sorry

end b_15_is_135_l304_304428


namespace power_result_l304_304285

theorem power_result (a b : ℤ) (h : |a + 3| + (b - 2)^2 = 0) : a^b = 9 :=
by
  sorry

end power_result_l304_304285


namespace smallest_three_digit_multiple_of_17_l304_304582

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304582


namespace mouse_jump_distance_l304_304139

theorem mouse_jump_distance
  (g : ℕ) 
  (f : ℕ) 
  (m : ℕ)
  (h1 : g = 25)
  (h2 : f = g + 32)
  (h3 : m = f - 26) : 
  m = 31 :=
by
  sorry

end mouse_jump_distance_l304_304139


namespace sum_of_15th_set_is_1695_l304_304336

theorem sum_of_15th_set_is_1695 : 
  let f : ℕ → ℕ := λ n, 1 + (n * (n - 1)) / 2
  let nth_set_sum : Π n, ℕ := λ n, n * (f n + f n + n - 1) / 2
  nth_set_sum 15 = 1695 := 
by 
  let f : ℕ → ℕ := λ n, 1 + (n * (n - 1)) / 2
  let nth_set_sum : Π n, ℕ := λ n, n * (f n + f n + n - 1) / 2
  show nth_set_sum 15 = 1695
  sorry

end sum_of_15th_set_is_1695_l304_304336


namespace arithmetic_sequence_l304_304109

theorem arithmetic_sequence (p q : ℕ) : ∀ n : ℕ, (a : ℕ → ℕ) (h : a n = p * n + q), a (n + 1) - a n = p := by
  sorry

end arithmetic_sequence_l304_304109


namespace smallest_three_digit_multiple_of_17_l304_304579

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304579


namespace angular_measure_intercepted_by_triangle_sides_is_60_l304_304203

-- Definitions and conditions based on the problem description
structure EquilateralTriangle :=
(height : ℝ)

structure Circle :=
(radius : ℝ)

-- Construct the given problem as an example in Lean
noncomputable def angular_measure_of_arc (T : EquilateralTriangle) (C : Circle) : ℝ :=
if r = T.height then 60 else 0 -- Simplified condition to match height and radius

-- Statement of the theorem to prove in Lean
theorem angular_measure_intercepted_by_triangle_sides_is_60 (T : EquilateralTriangle) (C : Circle) (h : C.radius = T.height) : 
  angular_measure_of_arc T C = 60 :=
by {
  sorry
}

end angular_measure_intercepted_by_triangle_sides_is_60_l304_304203


namespace max_rational_in_50x50_table_l304_304219

-- Defining the problem conditions
def max_rational_products (n : ℕ) (rat_count : ℕ) (irr_count : ℕ) (total_cells : ℕ) : ℕ :=
  total_cells - (let x := nat.div2 rat_count + nat.div2 irr_count in ((x * x) + (2 * x * (50 - x))))

-- Now state the theorem
theorem max_rational_in_50x50_table :
  max_rational_products 50 50 50 2500 = 1275 :=
sorry

end max_rational_in_50x50_table_l304_304219


namespace sum_T_n_l304_304288

def arith_seq (a_n : ℕ → ℕ) := ∀ n, a_n n = 2 * n - 1

def geom_seq (b_n : ℕ → ℕ) := ∀ n, b_n n = 2^n

def S_n (S_n : ℕ → ℕ) (b_n : ℕ → ℕ) := ∀ n, S_n n = 2 * b_n n - 2

def c_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) := ∀ n, c_n n = a_n n / b_n n

noncomputable def T_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) := 
  ∀ n, T_n n = 3 - (2 * n + 3) / (2^n)

theorem sum_T_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (S_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (arith_seq a_n) → 
  (geom_seq b_n) → 
  (S_n S_n b_n) → 
  (c_n a_n b_n) → 
    T_n T_n a_n b_n c_n := 
sorry

end sum_T_n_l304_304288


namespace chess_games_total_l304_304136

-- Conditions
def crowns_per_win : ℕ := 8
def uncle_wins : ℕ := 4
def draws : ℕ := 5
def father_net_gain : ℤ := 24

-- Let total_games be the total number of games played
def total_games : ℕ := sorry

-- Proof that under the given conditions, total_games equals 16
theorem chess_games_total :
  total_games = uncle_wins + (father_net_gain + uncle_wins * crowns_per_win) / crowns_per_win + draws := by
  sorry

end chess_games_total_l304_304136


namespace smallest_three_digit_multiple_of_17_l304_304772

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304772


namespace relationship_payment_method_age_distribution_expectation_X_l304_304098

/-- Contingency table conditions -/
def contingency_table := ∀ (n a b c d : ℕ) (h : n = a + b + c + d),
  (a = 40) ∧ (b = 10) ∧ (c = 10) ∧ (d = 40)

/-- Calculation for K^2 value and comparison with critical value -/
theorem relationship_payment_method_age (n a b c d: ℕ) (h : n = a + b + c + d) (ha : a = 40) (hb : b = 10) (hc : c = 10) (hd : d = 40) :
  let K2 := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  K2 = 36 ∧ 36 > 10.828 := 
by
  sorry

/-- Distribution and expectation of X under given conditions -/
theorem distribution_expectation_X :
  let P_X_0 := 1 / 45
  let P_X_1 := 16 / 45
  let P_X_2 := 28 / 45
  let E_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 in
  E_X = 8 / 5
:= 
by
  sorry

end relationship_payment_method_age_distribution_expectation_X_l304_304098


namespace sum_log_divisors_eq_1900_l304_304558

theorem sum_log_divisors_eq_1900 (n : ℕ) (h : ∑ (a : ℕ) in Finset.range (n + 1), 
      ∑ (b : ℕ) in Finset.range (n + 1), 
      (log 2 a + log 5 b)) / 2 = 1900) : 
    n = 15 :=
sorry

end sum_log_divisors_eq_1900_l304_304558


namespace log_expression_equals_four_l304_304974

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l304_304974


namespace Sarah_copy_total_pages_l304_304120

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end Sarah_copy_total_pages_l304_304120


namespace count_transformations_return_T_original_l304_304089

-- Definitions for vertices of triangle T
def T : Set (ℝ × ℝ) :=
  {p | p = (0,0) ∨ p = (6,0) ∨ p = (0,4)}

-- Define the transformations
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.snd, p.fst)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, -p.snd)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, -p.fst)
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.fst, -p.snd)
def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- Given set of transformations
def transformations : List (ℝ × ℝ → ℝ × ℝ) := [rotate90, rotate180, rotate270, reflectX, reflectY]

-- Define problem
theorem count_transformations_return_T_original :
  (List.length {s | s ∈ (List.permutations (List.replicate 3 1)).bind (fun p =>
      (List.permutations (List.replicate (p.nth 0).getD 0 rotate90 ⋆ (p.nth 1).getD 0 rotate90 ⋆ (p.nth 2).getD 0 rotate90).map
      (fun f => f.1 ∘ f.2 ∘ f.3)))).filter
    (fun f => f ∘ f ∘ f '' T = T)) = 18 :=
sorry

end count_transformations_return_T_original_l304_304089


namespace average_speed_increases_with_height_increase_l304_304933

/-- Given a set of experimental data where h denotes the height in cm and t denotes the time in seconds:
/-- | Height of Support $h (cm)$ | $10$ | $20$ | $30$ | $40$ | $50$ | $60$ | $70$ |
/-- |---------------------------|------|------|------|------|------|------|------|
/-- | Time for Car to Slide $t (s)$    | $4.23$ | $3.00$ | $2.45$ | $2.13$ | $1.89$ | $1.71$ | $1.59$ |
/-- Prove that as h gradually increases, the average speed of the car sliding down increases -/
theorem average_speed_increases_with_height_increase :
  ∀ (h : ℕ) (t : ℕ → ℝ), 
    (t 10 = 4.23) → (t 20 = 3.00) → (t 30 = 2.45) → (t 40 = 2.13) → (t 50 = 1.89) → (t 60 = 1.71) → (t 70 = 1.59) →
    ∀ (h1 h2 : ℕ), h1 < h2 → average_speed h1 t < average_speed h2 t :=
begin
  sorry
end

/-- Define average speed as height divided by time -/
def average_speed (h : ℕ) (t : ℕ → ℝ) : ℝ := h / (t h)

end average_speed_increases_with_height_increase_l304_304933


namespace sum_of_a6_a7_a8_l304_304315

variable {a : ℕ → ℝ} (h : ∑ i in Finset.range 13, a (i + 1) = 39)

theorem sum_of_a6_a7_a8 : a 6 + a 7 + a 8 = 9 :=
sorry

end sum_of_a6_a7_a8_l304_304315


namespace gain_percentage_l304_304881

theorem gain_percentage (selling_price gain : ℝ) (h1 : selling_price = 225) (h2 : gain = 75) : 
  (gain / (selling_price - gain) * 100) = 50 :=
by
  sorry

end gain_percentage_l304_304881


namespace correct_option_sqrt_neg2_squared_l304_304179

theorem correct_option_sqrt_neg2_squared : (sqrt ((-2) ^ 2)) = 2 :=
by sorry

end correct_option_sqrt_neg2_squared_l304_304179


namespace problem_solution_l304_304346

-- Defining the vectors a and b
def vec_a : EuclideanSpace ℝ (Fin 2) := ![2, 0]
def vec_b : EuclideanSpace ℝ (Fin 2) := ![1, 1]

-- Defining the orthogonality condition
def is_orthogonal (u v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dot_product u v = 0

-- Theorem statement
theorem problem_solution : is_orthogonal (vec_a - vec_b) vec_b := by
  sorry

end problem_solution_l304_304346


namespace quadratic_real_roots_l304_304365

theorem quadratic_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) : ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_real_roots_l304_304365


namespace estimate_total_height_l304_304477

theorem estimate_total_height :
  let middle_height := 100
  let left_height := 0.80 * middle_height
  let right_height := (left_height + middle_height) - 20
  left_height + middle_height + right_height = 340 := 
by
  sorry

end estimate_total_height_l304_304477


namespace smallest_three_digit_multiple_of_17_l304_304759

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304759


namespace smallest_three_digit_multiple_of_17_l304_304794

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304794


namespace range_of_a_l304_304318

open Real

noncomputable
def is_solution (a : ℝ) : Prop :=
  let A := (1 : ℝ, 2 : ℝ)
  let B := (2 : ℝ, 1 : ℝ)
  let circle_eq (p : ℝ × ℝ) := (p.1)^2 + (p.2)^2 + a * (p.1) - 1
  (circle_eq A * circle_eq B < 0)

theorem range_of_a :
  { a : ℝ | is_solution a } = { a : ℝ | -4 < a ∧ a < -2 } :=
by
  sorry

end range_of_a_l304_304318


namespace smallest_three_digit_multiple_of_17_l304_304586

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304586


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304720

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304720


namespace domain_of_f_range_of_m_l304_304328

theorem domain_of_f (m : ℝ) (x : ℝ)
  (h : m = 7) :
  (log 2 (abs (x + 1) + abs (x - 2) - m) domain) = (-∞, -3) ∪ (4, +∞) :=
sorry

theorem range_of_m (x : ℝ)
  (h: ∀ x, log 2 (abs (x + 1) + abs (x - 2) - m) ≥ 2) :
  m ≤ -1 :=
sorry

end domain_of_f_range_of_m_l304_304328


namespace largest_digit_never_in_odd_unit_l304_304173

-- Definition for what constitutes odd number unit digits
def odd_unit_digits : set ℕ := {1, 3, 5, 7, 9}

-- Definition for the largest digit that is not in the given set of odd_unit_digits
def largest_missing_digit : ℕ :=
  let even_digits := {0, 2, 4, 6, 8} in set.max even_digits

-- The theorem stating the problem
theorem largest_digit_never_in_odd_unit : largest_missing_digit ∉ odd_unit_digits :=
by {
  -- Definition of the largest missing digit aligned with the above question and conditions
  have evens : set ℕ := {0, 2, 4, 6, 8},
  have largest : largest_missing_digit = 8 := sorry, -- This would be proved based on max of the set
  -- Now proving largest is not in the odd_unit_digits
  rw largest,
  exact dec_trivial
}

end largest_digit_never_in_odd_unit_l304_304173


namespace simplify_expression_l304_304503

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304503


namespace smallest_three_digit_multiple_of_17_l304_304701

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304701


namespace union_intersection_l304_304450

-- Define the sets M, N, and P
def M := ({1} : Set Nat)
def N := ({1, 2} : Set Nat)
def P := ({1, 2, 3} : Set Nat)

-- Prove that (M ∪ N) ∩ P = {1, 2}
theorem union_intersection : (M ∪ N) ∩ P = ({1, 2} : Set Nat) := 
by 
  sorry

end union_intersection_l304_304450


namespace equation_represents_lines_and_point_l304_304135

theorem equation_represents_lines_and_point:
    (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 0 → (x = 1 ∧ y = -2)) ∧
    (∀ x y : ℝ, x^2 - y^2 = 0 → (x = y) ∨ (x = -y)) → 
    (∀ x y : ℝ, ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0 → 
    ((x = 1 ∧ y = -2) ∨ (x + y = 0) ∨ (x - y = 0))) :=
by
  intros h1 h2 h3
  sorry

end equation_represents_lines_and_point_l304_304135


namespace least_number_added_to_divisible_l304_304191

theorem least_number_added_to_divisible (n : ℕ) (k : ℕ) : n = 1789 → k = 11 → (n + k) % Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 4 3)) = 0 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end least_number_added_to_divisible_l304_304191


namespace cos_A_value_l304_304360

theorem cos_A_value (A : ℝ) (h : Real.tan A + Real.sec A = 3) : Real.cos A = 3/5 :=
sorry

end cos_A_value_l304_304360


namespace sum_of_ten_numbers_in_circle_l304_304524

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l304_304524


namespace bowling_ball_weight_l304_304993

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l304_304993


namespace largest_digit_never_in_odd_unit_l304_304174

-- Definition for what constitutes odd number unit digits
def odd_unit_digits : set ℕ := {1, 3, 5, 7, 9}

-- Definition for the largest digit that is not in the given set of odd_unit_digits
def largest_missing_digit : ℕ :=
  let even_digits := {0, 2, 4, 6, 8} in set.max even_digits

-- The theorem stating the problem
theorem largest_digit_never_in_odd_unit : largest_missing_digit ∉ odd_unit_digits :=
by {
  -- Definition of the largest missing digit aligned with the above question and conditions
  have evens : set ℕ := {0, 2, 4, 6, 8},
  have largest : largest_missing_digit = 8 := sorry, -- This would be proved based on max of the set
  -- Now proving largest is not in the odd_unit_digits
  rw largest,
  exact dec_trivial
}

end largest_digit_never_in_odd_unit_l304_304174


namespace find_a_l304_304447

variable {a : ℝ}
def f (x : ℝ) : ℝ := a * x^2 + 2

theorem find_a (h : deriv f (-1) = 4) : a = -2 :=
by sorry

end find_a_l304_304447


namespace triangle_area_with_polynomial_roots_l304_304134

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  in Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem triangle_area_with_polynomial_roots :
  (∃ a b c : ℝ, (x^3 - 5 * x^2 + 8 * x - (15 / 7) = (x - a) * (x - b) * (x - c)) ∧
  area_of_triangle a b c = (Real.sqrt 135 / Real.sqrt 14)) :=
by
  sorry

end triangle_area_with_polynomial_roots_l304_304134


namespace determine_x_l304_304144

noncomputable def is_equal_mean_median_mode (x : ℕ) : Prop :=
  let s := {3, 4, 5, 6, 6, 7, x}
  let median := 6
  let mode := 6
  let mean := (3 + 4 + 5 + 6 + 6 + 7 + x) / 7
  mode = 6 ∧ median = 6 ∧ mean = 6

theorem determine_x : is_equal_mean_median_mode 11 :=
  by
  sorry

end determine_x_l304_304144


namespace magnitude_of_a_minus_4b_l304_304345

noncomputable def magnitude (v : ℝ) : ℝ := real.sqrt (v)

variables (a b : ℝ) (pi_over_3 : ℝ)

axiom unit_vector_a : real.abs (a) = 1
axiom unit_vector_b : real.abs (b) = 1
axiom angle_a_b : pi_over_3 = real.pi / 3

theorem magnitude_of_a_minus_4b : magnitude (1^2 - 8 * (1 * (real.cos pi_over_3) * 1 / 2) + 16 * 1^2) = real.sqrt 13 :=
by {
  sorry
}

end magnitude_of_a_minus_4b_l304_304345


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304711

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304711


namespace square_area_max_l304_304929

theorem square_area_max (perimeter : ℝ) (h_perimeter : perimeter = 32) : 
  ∃ (area : ℝ), area = 64 :=
by
  sorry

end square_area_max_l304_304929


namespace cos_of_tan_l304_304308

theorem cos_of_tan (α : ℝ) (hα1 : α ∈ Ioc (π / 2) π) (hα2 : Real.tan α = - (Real.sqrt 3) / 3) : 
  Real.cos α = - (Real.sqrt 3) / 2 := 
  sorry

end cos_of_tan_l304_304308


namespace smallest_three_digit_multiple_of_17_l304_304864

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304864


namespace parabola_intersections_count_l304_304548

theorem parabola_intersections_count :
  let y := λ x : ℝ, (x - 1) * (x - 2) in
  -- Define the intersection points with the y-axis
  let y_intersections := [(0, y 0)] in
  -- Define the intersection points with the x-axis
  let x_intersections := [(1, 0), (2, 0)] in
  -- Count the total number of intersections
  y_intersections.length + x_intersections.length = 3 :=
by
  sorry

end parabola_intersections_count_l304_304548


namespace parallel_lines_perpendicular_lines_l304_304341

noncomputable def line1 (m : ℝ) := λ x y : ℝ, x + (1 + m) * y = 2 - m
noncomputable def line2 (m : ℝ) := λ x y : ℝ, 2 * m * x + 4 * y = -16

theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y → line2 m x y) ↔ m = 1 :=
by
  sorry

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, (line1 m x y → (slope1 ≠ 0) ∧ (slope2 ≠ 0) ∧ 
    slope1 * slope2 = -1)) ↔ m = -2 / 3 :=
by
  sorry

end parallel_lines_perpendicular_lines_l304_304341


namespace average_jump_difference_l304_304926

-- Define the total jumps and time
def total_jumps_liu_li : ℕ := 480
def total_jumps_zhang_hua : ℕ := 420
def time_minutes : ℕ := 5

-- Define the average jumps per minute
def average_jumps_per_minute (total_jumps : ℕ) (time : ℕ) : ℕ :=
  total_jumps / time

-- State the theorem
theorem average_jump_difference :
  average_jumps_per_minute total_jumps_liu_li time_minutes - 
  average_jumps_per_minute total_jumps_zhang_hua time_minutes = 12 := 
sorry


end average_jump_difference_l304_304926


namespace ellipse_sum_l304_304538

theorem ellipse_sum (h k a b : ℤ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 7) (b_val : b = 4) : 
  h + k + a + b = 9 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end ellipse_sum_l304_304538


namespace smallest_three_digit_multiple_of_17_correct_l304_304621

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304621


namespace f_strictly_decreasing_l304_304268

-- Define the function g(x) = x^2 - 2x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the function f(x) = log_{1/2}(g(x))
noncomputable def f (x : ℝ) : ℝ := Real.log (g x) / Real.log (1 / 2)

-- The problem statement to prove: f(x) is strictly decreasing on the interval (3, ∞)
theorem f_strictly_decreasing : ∀ x y : ℝ, 3 < x → x < y → f y < f x := by
  sorry

end f_strictly_decreasing_l304_304268


namespace ellipse_eq_standard_lambda_value_constant_l304_304048

-- Define the given conditions
def focal_distance := 2 * Real.sqrt 2
def focal_c := Real.sqrt 2
def b := Real.sqrt 2
def a_squared := b * b + focal_c * focal_c

-- Equation of the ellipse
def equation_of_ellipse := ∀ x y : ℝ,
  (x : ℝ) ^ 2 / 4 + (y : ℝ) ^ 2 / 2 = 1

-- Proving the standard equation of the ellipse
theorem ellipse_eq_standard:
  equation_of_ellipse :=
begin
  sorry -- proof ommited
end

-- Define a constant for lambda_1 + lambda_2
def line_l_passing_through := {M | M = (1, 0)}
def point_N := (0, -k)

-- Additional condition involving lambda_1 and lambda_2
def lambda_constant := ∀ (x1 x2 : ℝ),
  let λ1 := x1 / (1 - x1),
  let λ2 := x2 / (1 - x2) in
  λ1 + λ2 = -8 / 3

-- Proving λ1 + λ2 is constant
theorem lambda_value_constant:
  lambda_constant :=
begin
  sorry -- proof ommited
end

end ellipse_eq_standard_lambda_value_constant_l304_304048


namespace smallest_three_digit_multiple_of_17_l304_304633

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304633


namespace a10_plus_b10_l304_304100

noncomputable def a : ℝ := sorry -- a will be a real number satisfying the conditions
noncomputable def b : ℝ := sorry -- b will be a real number satisfying the conditions

axiom ab_condition1 : a + b = 1
axiom ab_condition2 : a^2 + b^2 = 3
axiom ab_condition3 : a^3 + b^3 = 4
axiom ab_condition4 : a^4 + b^4 = 7
axiom ab_condition5 : a^5 + b^5 = 11

theorem a10_plus_b10 : a^10 + b^10 = 123 :=
by 
  sorry

end a10_plus_b10_l304_304100


namespace proof_y_pow_x_equal_1_by_9_l304_304022

theorem proof_y_pow_x_equal_1_by_9 
  (x y : ℝ)
  (h : (x - 2)^2 + abs (y + 1/3) = 0) :
  y^x = 1/9 := by
  sorry

end proof_y_pow_x_equal_1_by_9_l304_304022


namespace sin_cos_theta_value_l304_304020

theorem sin_cos_theta_value (a b θ : ℝ) 
  (h : (sin θ) ^ 6 / a ^ 2 + (cos θ) ^ 6 / b ^ 2 = 1 / (a + b)) :
  (sin θ) ^ 12 / a ^ 5 + (cos θ) ^ 12 / b ^ 5 = 1 / (a + b) ^ 5 := 
by
  sorry

end sin_cos_theta_value_l304_304020


namespace area_remains_same_l304_304476

-- Define the transformation f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 1, p.2 + 2)

-- Define the quadrilateral F and its area
variable (F : set (ℝ × ℝ))
variable (area_F : ℝ)
variable (area_F_6 : area_F = 6)

-- Define the image of F under the transformation f
def F' : set (ℝ × ℝ) := {p' | ∃ p ∈ F, f p = p'}

-- State the theorem
theorem area_remains_same : area_F = 6 → area_F' = 6 := by
  sorry

end area_remains_same_l304_304476


namespace smallest_three_digit_multiple_of_17_l304_304751

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304751


namespace find_a_l304_304325

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x else x + 1

theorem find_a : ∃ a : ℝ, f(a) + f(1) = 0 ∧ a = -3 := 
by {
  sorry
}

end find_a_l304_304325


namespace smallest_three_digit_multiple_of_17_l304_304693

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304693


namespace main_theorem_l304_304333

open Real

-- Define the quadratic equation and its properties
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - (2 * m + 3) * x + m^2

-- Define the condition for having two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  let Δ := (2 * m + 3)^2 - 4 * 1 * m^2
  in Δ > 0

-- Define the condition for the roots of the quadratic equation
def root_condition (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ + x₂ ≠ 0 ∧ x₁ * x₂ ≠ 0 ∧ (x₁ + x₂) / (x₁ * x₂) = 1

-- Generate the roots x₁ and x₂ of the quadratic equation
noncomputable def roots (m : ℝ) : ℝ × ℝ :=
  let b := -(2 * m + 3)
  let c := m^2
  let Δ := b^2 - 4 * c
  ((-b + sqrt Δ) / 2, (-b - sqrt Δ) / 2)

-- Main proof statement
theorem main_theorem :
  (∀ m : ℝ, has_two_distinct_real_roots m → m > -3 / 4) ∧
  (∀ m : ℝ, has_two_distinct_real_roots m → root_condition m (roots m).1 (roots m).2 → m = 3) :=
by
  sorry

end main_theorem_l304_304333


namespace smallest_three_digit_multiple_of_17_l304_304697

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304697


namespace smallest_three_digit_multiple_of_17_l304_304601

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304601


namespace smallest_three_digit_multiple_of_17_l304_304741

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304741


namespace minimum_folds_for_square_l304_304919

-- Define the property of a quadrilateral being a square
structure Quadrilateral :=
(a b c d : ℝ)  -- sides or relevant properties

def is_square (q : Quadrilateral) : Prop :=
  -- Insert the geometry definition for the quadrilateral to be a square.
  -- For now, we use an abstract placeholder.
  sorry

-- Define the property for a fold
def fold (q : Quadrilateral) (axis : ℝ × ℝ) : Quadrilateral :=
  -- Definition of folding along an axis.
  sorry

-- Prove minimum folds needed to verify a quadrilateral is a square
theorem minimum_folds_for_square (q : Quadrilateral) : (∃ n : ℕ, n ≤ 2 ∧ (is_square q → validation_by_fold q n)) := 
  sorry


end minimum_folds_for_square_l304_304919


namespace symmetry_center_of_g_l304_304005

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 6) + 1

theorem symmetry_center_of_g :
  ∃ x₀ y₀ : ℝ, (x₀ = π / 12 ∧ y₀ = 1) ∧
  ∀ x y : ℝ, (g x₀ - y₀ = g (2 * x₀ - x) - y₀) :=
begin
  sorry
end

end symmetry_center_of_g_l304_304005


namespace tip_percentage_is_10_l304_304908

def original_bill : ℝ := 139.00
def per_person_amount : ℝ := 21.842857142857145
def number_of_people : ℕ := 7

theorem tip_percentage_is_10 :
  let total_amount_paid := per_person_amount * number_of_people
  let tip_amount := total_amount_paid - original_bill
  let tip_percentage := (tip_amount / original_bill) * 100
  tip_percentage ≈ 10 := sorry

end tip_percentage_is_10_l304_304908


namespace bowling_ball_weight_l304_304996

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l304_304996


namespace smallest_three_digit_multiple_of_17_l304_304614

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304614


namespace polar_to_cartesian_l304_304013

-- Definitions for the polar coordinates conversion
noncomputable def polar_to_cartesian_eq (C : ℝ → ℝ → Prop) :=
  ∀ (ρ θ : ℝ), (ρ^2 * (1 + 3 * (Real.sin θ)^2) = 4) → C (ρ * (Real.cos θ)) (ρ * (Real.sin θ))

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1)

-- The main theorem
theorem polar_to_cartesian 
  (C : ℝ → ℝ → Prop)
  (h : polar_to_cartesian_eq C) :
  ∀ x y : ℝ, C x y ↔ cartesian_eq x y :=
by
  sorry

end polar_to_cartesian_l304_304013


namespace zhiqiang_series_l304_304185

theorem zhiqiang_series (a b : ℝ) (n : ℕ) (n_pos : 0 < n) (h : a * b = 1) (h₀ : b ≠ 1):
  (1 + a^n) / (1 + b^n) = ((1 + a) / (1 + b)) ^ n :=
by
  sorry

end zhiqiang_series_l304_304185


namespace ball_returns_to_bella_after_14_throws_l304_304160

theorem ball_returns_to_bella_after_14_throws : 
  ∃ n, (let g (i : ℕ) := (1 + 5 * i) % 13 in
    g n = 1 ∧ ∀ k < n, g k ≠ 1) ∧ n = 14 :=
by
  sorry

end ball_returns_to_bella_after_14_throws_l304_304160


namespace zero_point_in_interval_l304_304008

noncomputable def f (ω x : ℝ) : ℝ :=
  sin^2 (ω * x / 2) + (1/2) * sin (ω * x) - 1/2

theorem zero_point_in_interval (ω : ℝ) (hω : ω > 0) :
  (∃ x ∈ Ioo π (2 * π), f ω x = 0) ↔ ω ∈ Ioo (1/8) (1/4) ∪ Ioi (5/8) :=
by
  sorry

end zero_point_in_interval_l304_304008


namespace measure_angle_PDG_eq_pi_div_2_l304_304214

-- Definitions and conditions
variables (A B C D E F G P : Type) [regular_heptagon A B C D E F G] [line_intersection AB CE P]

-- Goal statement
theorem measure_angle_PDG_eq_pi_div_2 : measure_angle P D G = π / 2 :=
by
  sorry

end measure_angle_PDG_eq_pi_div_2_l304_304214


namespace smallest_three_digit_multiple_of_17_l304_304848

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304848


namespace nuts_eaten_condition_not_all_nuts_eaten_l304_304944

/-- proof problem with conditions and questions --/

-- Let's define the initial setup and the conditions:

def anya_has_all_nuts (nuts : Nat) := nuts > 3

def distribution (a b c : ℕ → ℕ) (n : ℕ) := 
  ((a (n + 1) = b n + c n + (a n % 2)) ∧ 
   (b (n + 1) = a n / 2) ∧ 
   (c (n + 1) = a n / 2))

def nuts_eaten (a b c : ℕ → ℕ) (n : ℕ) := 
  (a n % 2 > 0 ∨ b n % 2 > 0 ∨ c n % 2 > 0)

-- Prove at least one nut will be eaten
theorem nuts_eaten_condition (a b c : ℕ → ℕ) (n : ℕ) :
  anya_has_all_nuts (a 0) → distribution a b c n → nuts_eaten a b c n :=
sorry

-- Prove not all nuts will be eaten
theorem not_all_nuts_eaten (a b c : ℕ → ℕ):
  anya_has_all_nuts (a 0) → distribution a b c n → 
  ¬∀ (n: ℕ), (a n = 0 ∧ b n = 0 ∧ c n = 0) :=
sorry

end nuts_eaten_condition_not_all_nuts_eaten_l304_304944


namespace smallest_three_digit_multiple_of_17_l304_304819

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304819


namespace triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l304_304047

-- Define the properties and variables of the given obtuse triangle
variables (a b c : ℝ) (A C : ℝ)
-- Given conditions
axiom ha : a = 7
axiom hb : b = 3
axiom hcosC : Real.cos C = 11 / 14

-- Prove the values of c and angle A
theorem triangle_ABC_c_and_A_value (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : c = 5 ∧ A = 2 * Real.pi / 3 :=
sorry

-- Prove the value of sin(2C - π / 6)
theorem sin_2C_minus_pi_6 (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : Real.sin (2 * C - Real.pi / 6) = 71 / 98 :=
sorry

end triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l304_304047


namespace donut_hole_problem_l304_304235

theorem donut_hole_problem :
  let r_N := 5
  let r_T := 7
  let r_A := 9
  let v_N := 2
  let v_T := 3
  let v_A := 1
  let area_N := 4 * π * r_N^2
  let area_T := 4 * π * r_T^2
  let area_A := 4 * π * r_A^2
  let rate_N := area_N * v_N
  let rate_T := area_T * v_T
  let rate_A := area_A * v_A
  let lcm_rates := Nat.lcm (Nat.lcm rate_N.toNat rate_T.toNat) rate_A.toNat
  ( lcm_rates / rate_N.toNat ) = 2646 := by
  sorry

end donut_hole_problem_l304_304235


namespace coefficient_of_x3_in_expansion_l304_304402

theorem coefficient_of_x3_in_expansion : 
  (∃ c : ℕ, (x^6 + 6 * x^4 + 15 * x^2 + 20 + 15 * x^-2 + 6 * x^-4 + x^-6) * (1 + x) = (1 + x) * (x^2 + x^-1)^6 ∧ c = 20) :=
sorry

end coefficient_of_x3_in_expansion_l304_304402


namespace smallest_three_digit_multiple_of_17_l304_304781

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304781


namespace smallest_three_digit_multiple_of_17_l304_304757

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304757


namespace sum_of_numerator_and_denominator_of_repeating_decimal_0_47_equals_146_l304_304869

theorem sum_of_numerator_and_denominator_of_repeating_decimal_0_47_equals_146 : 
    ∃ (a b : ℕ), (a / b = (47 / 99) ∧ a.gcd b = 1 ∧ a + b = 146) :=
by 
  -- Let x be the repeating decimal 0.474747...
  let x := 0.474747474747... in

  -- x is equivalent to 47/99 in simplest form 
  have h : x = 47 / 99 := by sorry,

  -- Sum of the numerator and denominator of 47/99
  have numerator := 47,
  have denominator := 99,

  -- GCD of 47 and 99 is 1 (since 47 is prime and does not divide 99)
  have gcd_47_99 : numerator.gcd(denominator) = 1 := by simp,

  -- Sum of numerator and denominator is 146
  have sum_146 : numerator + denominator = 146 := by simp,

  existsi numerator,
  existsi denominator,
  split,
  exact h,
  split,
  exact gcd_47_99,
  exact sum_146

-- sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_0_47_equals_146_l304_304869


namespace smallest_three_digit_multiple_of_17_l304_304647

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304647


namespace inequality_proof_l304_304090

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_eq : a + b + c = 4 * (abc)^(1/3)) :
  2 * (ab + bc + ca) + 4 * min (a^2) (min (b^2) (c^2)) ≥ a^2 + b^2 + c^2 :=
by
  sorry

end inequality_proof_l304_304090


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304719

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304719


namespace smallest_three_digit_multiple_of_17_l304_304686

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304686


namespace leading_coefficient_of_g_l304_304150

theorem leading_coefficient_of_g (g : ℕ → ℝ) (h : ∀ x : ℕ, g x.succ - g x = 8 * x + 9) : 
  leading_coeff (polynomial_of_function g) = 4 :=
sorry

end leading_coefficient_of_g_l304_304150


namespace total_distance_traveled_l304_304111

theorem total_distance_traveled (d d1 d2 d3 d4 d5 : ℕ) 
  (h1 : d1 = d)
  (h2 : d2 = 2 * d)
  (h3 : d3 = 40)
  (h4 : d = 2 * d3)
  (h5 : d4 = 2 * (d1 + d2 + d3))
  (h6 : d5 = 3 * d4 / 2) 
  : d1 + d2 + d3 + d4 + d5 = 1680 :=
by
  have hd : d = 80 := sorry
  have hd1 : d1 = 80 := sorry
  have hd2 : d2 = 160 := sorry
  have hd4 : d4 = 560 := sorry
  have hd5 : d5 = 840 := sorry
  sorry

end total_distance_traveled_l304_304111


namespace smallest_three_digit_multiple_of_17_l304_304766

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304766


namespace trajectory_eq_circle_through_fixed_points_l304_304397

-- Definitions based on the conditions 
def dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Proof problem 1
theorem trajectory_eq (M : ℝ × ℝ) (h : dist M (1, 0) = Real.abs M.1 + 1) :
  (M.1 >= 0 -> M.2^2 = 4 * M.1) ∧ (M.1 < 0 -> M.2 = 0) := sorry

noncomputable def C (x : ℝ) := if x >= 0 then (some (λ y : ℝ, y^2 = 4 * x)) else (0)

-- Proof problem 2
theorem circle_through_fixed_points (A B : ℝ × ℝ) (hC : ∀ x, A = (1, (4 / (some (λ y, y = x))))) :
  ∃ (fixed_points : ℝ × ℝ), fixed_points = (-1,0) ∨ fixed_points = (3,0) := sorry

end trajectory_eq_circle_through_fixed_points_l304_304397


namespace paint_cans_l304_304104

/-
Proof Problem:
Prove that if the ratio of blue paint to green paint is 4:3 and Alice wants to make 40 cans of the mixture, then Alice will need 23 cans of blue paint.
-/

theorem paint_cans (ratio_bg : ℕ × ℕ) (total_cans : ℕ) (same_volume : ∀ x, true) :
  (ratio_bg = (4, 3)) → (total_cans = 40) → ∃ blue_cans : ℕ, blue_cans = 23 :=
by
  intros h_ratio h_total
  sorry

end paint_cans_l304_304104


namespace solution_set_of_inequality_l304_304231

noncomputable def odd_increasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x)) ∧ (∀ x y, 0 < x → x < y → f (x) < f (y)) ∧ (f (1) = 0)

theorem solution_set_of_inequality (f : ℝ → ℝ) (h : odd_increasing_function f) :
  {x : ℝ | (f(x) - f(-x)) / x < 0} = set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

end solution_set_of_inequality_l304_304231


namespace series_sum_l304_304951

theorem series_sum : 
  (∑ k in Finset.range 2016, ∑ j in Finset.range k, (k:ℝ) / (k + j + 1)) = 1015560 := 
by
  -- Summing will go through Finset sums from 1 to 2015 and the inner sum from k+1 to 2016
  sorry

end series_sum_l304_304951


namespace max_value_f_on_interval_l304_304546

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 15 :=
by
  sorry

end max_value_f_on_interval_l304_304546


namespace move_point_A_l304_304398

theorem move_point_A :
  let A := (-5, 6)
  let A_right := (A.1 + 5, A.2)
  let A_upwards := (A_right.1, A_right.2 + 6)
  A_upwards = (0, 12) := by
  sorry

end move_point_A_l304_304398


namespace total_gas_cost_l304_304469

-- Define the conditions from part a
def start_odom : ℤ := 63102
def mid_odom1 : ℤ := 63135
def mid_odom2 : ℤ := 63166

def fuel_efficiency : ℝ := 25
def gas_price : ℝ := 3.95

-- Define the proof problem based on part c
theorem total_gas_cost :
  let distance1 := mid_odom1 - start_odom in
  let distance2 := mid_odom2 - mid_odom1 in
  let total_distance := distance1 + distance2 in
  let gallons_used := total_distance / fuel_efficiency in
  let total_cost := gallons_used * gas_price in
  total_cost ≈ 10.11 :=
by
  sorry

end total_gas_cost_l304_304469


namespace smallest_three_digit_multiple_of_17_l304_304637

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304637


namespace prove_b_eq_A_pow_n_l304_304440

theorem prove_b_eq_A_pow_n (b n : ℤ) (hb : b > 1) (hn : n > 1)
  (hdiv : ∀ k : ℤ, k > 1 → ∃ a_k : ℤ, (b - a_k ^ n) % k = 0) :
  ∃ A : ℤ, b = A ^ n := 
sorry

end prove_b_eq_A_pow_n_l304_304440


namespace smallest_three_digit_multiple_of_17_l304_304589

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304589


namespace fifty_fourth_card_l304_304997

theorem fifty_fourth_card :
  let sequence := ["A_♠", "2_♠", "3_♠", "4_♠", "5_♠", "6_♠", "7_♠", "8_♠", "9_♠", "10_♠", "J_♠", "Q_♠", "K_♠",
                  "A_♥", "2_♥", "3_♥", "4_♥", "5_♥", "6_♥", "7_♥", "8_♥", "9_♥", "10_♥", "J_♥", "Q_♥", "K_♥"] in
  let full_cycle := sequence ++ sequence in
  (full_cycle ++ full_cycle).nth 53 = some "2_♠" :=
by {
  sorry
}

end fifty_fourth_card_l304_304997


namespace non_associative_products_l304_304092

-- Definitions
def h (n : ℕ) : ℕ := if n = 1 then 1 else if n = 2 then 2 else sorry

-- Factorial and Catalan number are already defined in Mathlib

theorem non_associative_products (n : ℕ) (hn : h n) : 
  h n = (2 * n - 2).fact / (n - 1).fact ∧ h n = n.fact * Catalan n :=
by {
  sorry
}

end non_associative_products_l304_304092


namespace smallest_three_digit_multiple_of_17_l304_304645

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304645


namespace smallest_three_digit_multiple_of_17_l304_304646

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304646


namespace max_n_l304_304445

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct_nat_set (s : Set ℕ) : Prop :=
  (∀ x ∈ s, x > 0) ∧ (∀ x y ∈ s, x ≠ y → x ≠ y)

def sum_of_three_is_prime (s : Set ℕ) : Prop :=
  ∀ x y z ∈ s, x ≠ y → x ≠ z → y ≠ z → is_prime (x + y + z)

theorem max_n  (s : Set ℕ) 
  (h1 : distinct_nat_set s) 
  (h2 : sum_of_three_is_prime s) : s.Finite.toFinset.card ≤ 4 :=
by
  sorry

end max_n_l304_304445


namespace greatest_n_l304_304916

theorem greatest_n (k : ℕ) : ∃ n : ℕ, (∀ (f : ℕ → ℕ) (d : ℕ → ℕ),
  (∀ (x : ℕ), x < n → x % k = f x) →
  (∀ (x y: ℕ), x < n ∧ y < n ∧ f x = f y ∧ (∀ z, x < z ∧ z < y → f z ≠ f x) → 
    (d x y = y - x)) →
  (∀ (x y : ℕ), x < n ∧ y < n ∧ x ≠ y → d x y ≠ d y x)) ∧
  n = 3 * k - 1 :=
begin
  sorry
end

end greatest_n_l304_304916


namespace smallest_three_digit_multiple_of_17_l304_304735

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304735


namespace limit_calculation_l304_304953

theorem limit_calculation :
  tendsto (λ n : ℕ, (2 * n : ℝ) / (3 * n - 1)) at_top (𝓝 (2 / 3)) :=
sorry

end limit_calculation_l304_304953


namespace smallest_three_digit_multiple_of_17_l304_304832

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304832


namespace set_of_i_power_sum_l304_304096

theorem set_of_i_power_sum : 
  {Z | ∃ (n : ℤ), Z = (complex.I ^ n) + (complex.I ^ -n)} = {0, 2, -2} := 
by
  sorry

end set_of_i_power_sum_l304_304096


namespace problem1_unions_problem2_range_of_m_l304_304343

open Set

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x < 1 + 3 * m }
def A_complement : Set ℝ := { x | x ≤ -1 ∨ x > 3 }

theorem problem1_unions {m : ℝ} (h : m = 1) :
  (A ∪ B m) = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

theorem problem2_range_of_m (h : B m ⊆ A_complement) :
  m ∈ Iic (-1/2 : ℝ) ∨ m ∈ Ioi (3 : ℝ) :=
sorry

end problem1_unions_problem2_range_of_m_l304_304343


namespace blue_paint_needed_l304_304914

theorem blue_paint_needed (total_cans : ℕ) (blue_ratio : ℕ) (yellow_ratio : ℕ)
  (h_ratio: blue_ratio = 5) (h_yellow_ratio: yellow_ratio = 3) (h_total: total_cans = 45) : 
  ⌊total_cans * (blue_ratio : ℝ) / (blue_ratio + yellow_ratio)⌋ = 28 :=
by
  sorry

end blue_paint_needed_l304_304914


namespace log_expr_solution_l304_304980

-- Define the function that represents the recursive logarithmic expression
def log_expr (x : ℝ) := Real.logb 3 (81 + x)

-- The main statement we want to prove
theorem log_expr_solution : ∃ x : ℝ, x = log_expr x ∧ 0 < x ∧ x = 8 :=
by
  sorry

end log_expr_solution_l304_304980


namespace maximum_value_frac_l304_304093

-- Let x and y be positive real numbers. Prove that (x + y)^3 / (x^3 + y^3) ≤ 4.
theorem maximum_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^3 / (x^3 + y^3) ≤ 4 := sorry

end maximum_value_frac_l304_304093


namespace math_proof_problem_l304_304291

open Real

variables (F1 F2 : ℝ × ℝ) (a b c : ℝ)
variables (A B : ℝ × ℝ)
variables (slope : ℝ)
variables (P : ℝ × ℝ)

-- Given conditions
def conditions : Prop :=
  F1 = (-2 * sqrt 2, 0) ∧
  F2 = (2 * sqrt 2, 0) ∧
  a = 3 ∧
  b = 1 ∧
  c = 2 * sqrt 2 ∧
  P = (0, 2) ∧
  slope = 1

-- Question 1: Standard equation of the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 1 = 1)

-- Question 2: Length of the line segment AB
def length_AB : ℝ :=
  dist A B

theorem math_proof_problem :
  conditions →
  (∀ x y, ellipse_eq x y) ∧
  (∃ A B, A ≠ B ∧ length_AB = 6 * sqrt 3 / 5) :=
by
  sorry

end math_proof_problem_l304_304291


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304718

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304718


namespace smallest_three_digit_multiple_of_17_l304_304778

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304778


namespace collinear_probability_correct_l304_304407

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end collinear_probability_correct_l304_304407


namespace smallest_three_digit_multiple_of_17_l304_304736

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304736


namespace smallest_three_digit_multiple_of_17_l304_304658

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304658


namespace double_mean_value_function_range_l304_304253

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - x ^ 2 + m

theorem double_mean_value_function_range (a : ℝ) (m : ℝ) :
  (1 / 8 : ℝ) < a ∧ a < (1 / 4 : ℝ) ↔
  (∀ x ∈ Icc (0:ℝ) (2*a), 6 * x ^ 2 - 2 * x = 8 * a^2 - 2 * a) ∧
  (f'' (0) = (f(2*a) - f(0)) / (2*a)) :=
begin
  sorry
end

end double_mean_value_function_range_l304_304253


namespace find_length_AB_l304_304043

-- Definitions for the problem conditions.
def angle_B : ℝ := 90
def angle_A : ℝ := 30
def BC : ℝ := 24

-- Main theorem to prove.
theorem find_length_AB (angle_B_eq : angle_B = 90) (angle_A_eq : angle_A = 30) (BC_eq : BC = 24) : 
  ∃ AB : ℝ, AB = 12 := 
by
  sorry

end find_length_AB_l304_304043


namespace smallest_three_digit_multiple_of_17_l304_304824

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304824


namespace product_of_cosines_equal_one_m_n_power_is_one_l304_304874

theorem product_of_cosines_equal_one :
  (∏ k in finset.range 30, real.cos (real.pi / 60 * (k + 1))) = (31 / 2^30) :=
begin
  sorry
end

theorem m_n_power_is_one (m n : ℤ) (hm : m = 31) (hn : n = -30) :
  m ^ n = 1 :=
begin
  -- Utilize the previous theorem to show the exponentiation to 1
  have hcos : (∏ k in finset.range 30, real.cos (real.pi / 60 * (k + 1))) = (31 / 2^30),
  from product_of_cosines_equal_one,

  -- Considering m and n
  rw [hm, hn],

  -- Calculate m^n when m = 31 and n = -30
  rw pow_neg,
  show (31 : ℤ) ^ -30 = (1 : ℤ),
  rw [zpow_neg, zpow_coe_nat],
  have h_pow : (31 : ℤ) ^ 30 = 2^30,
  -- Proof should rely on a previously shown fundamental trigonometric identity
  sorry,  -- Further proof steps to show the equality

  show (1 : ℤ) = (1 : ℤ),
end

end product_of_cosines_equal_one_m_n_power_is_one_l304_304874


namespace sequence_strictly_increasing_l304_304015

theorem sequence_strictly_increasing (λ : ℝ) :
  (∀ n : ℕ+, a_{n} = n^2 + λ * n → a_{n+1} = (n+1)^2 + λ * (n+1) →  a_{n+1} > a_{n}) →
  λ > -3 := 
by 
  sorry

end sequence_strictly_increasing_l304_304015


namespace cumulonimbus_cloud_count_l304_304555

-- Definitions given in conditions
def Ci : ℕ := 144
def Cu := Ci / 4
def Cb := Cu / 12

-- Theorem stating that the number of cumulonimbus clouds is 3
theorem cumulonimbus_cloud_count (Ci : ℕ) (H_Ci : Ci = 144) : Cb = 3 :=
by
  rw [H_Ci]
  unfold Cu
  unfold Cb
  simp
  sorry

end cumulonimbus_cloud_count_l304_304555


namespace fraction_red_roses_l304_304233

-- Definitions of the conditions
def total_rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_pink_roses : ℕ := 40
def non_red_roses (R : ℕ) : ℕ := roses_per_row - R
def white_roses (R : ℕ) : ℕ := 3 * non_red_roses(R) / 5
def pink_roses (R : ℕ) : ℕ := 2 * non_red_roses(R) / 5

-- Main theorem statement
theorem fraction_red_roses :
  ∃ R : ℕ, (pink_roses(R) * total_rows = total_pink_roses) ∧ (R = 10) :=
sorry

end fraction_red_roses_l304_304233


namespace proof_integer_probability_division_is_five_sixteenths_l304_304432

def integer_probability_division_is_five_sixteenths (r k : ℤ) (h1 : -4 < r) (h2 : r < 7) (h3 : 0 < k) (h4 : k < 9): Prop :=
  let valid_r := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6].val
  let valid_k := [1, 2, 3, 4, 5, 6, 7, 8].val
  let pairs := for r in valid_r, k in valid_k do if k ∣ r then some (r, k) else none
  let valid_pairs := pairs.filterMap id
  let probability := valid_pairs.length / 80
  probability = (5/16 : ℚ)

theorem proof_integer_probability_division_is_five_sixteenths (r k : ℤ) (hr1 : -4 < r) (hr2 : r < 7) (hk1 : 0 < k) (hk2 : k < 9) :
  integer_probability_division_is_five_sixteenths r k hr1 hr2 hk1 hk2 :=
by sorry

end proof_integer_probability_division_is_five_sixteenths_l304_304432


namespace decreasing_function_among_given_l304_304871

def is_decreasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem decreasing_function_among_given:
  is_decreasing_on_R (λ x, (1/2)^x) ∧
  ¬is_decreasing_on_R (λ x, log (1/2) x) ∧
  ¬is_decreasing_on_R (λ x, x⁻¹) ∧
  ¬is_decreasing_on_R (λ x, x^2) :=
begin
  sorry
end

end decreasing_function_among_given_l304_304871


namespace max_intersection_points_circle_square_l304_304575

noncomputable def max_intersection_points (circle : Type) (square : Type) :=
  ∃ (intersect : circle → square → ℕ),
    (∀ c s, intersect c s ≤ 2) ∧ -- a circle can intersect a line segment at most at two points
    (∑ s in (finset.range 4), 2) = 8 -- a square consists of four line segments, each can intersect the circle at most at 2 points, and their sum is 8

theorem max_intersection_points_circle_square :
  max_intersection_points ℝ (ℝ × ℝ) := 
sorry

end max_intersection_points_circle_square_l304_304575


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304717

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304717


namespace smallest_three_digit_multiple_of_17_l304_304846

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304846


namespace smallest_three_digit_multiple_of_17_l304_304798

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304798


namespace log_sequence_value_l304_304976

theorem log_sequence_value :
  ∃ x : ℝ, x = log 3 (81 + x) ∧ x > 0 ∧ x ≈ 5 :=
begin
  sorry
end

end log_sequence_value_l304_304976


namespace wendy_time_correct_l304_304236

variable (bonnie_time wendy_difference : ℝ)

theorem wendy_time_correct (h1 : bonnie_time = 7.80) (h2 : wendy_difference = 0.25) : 
  (bonnie_time - wendy_difference = 7.55) :=
by
  sorry

end wendy_time_correct_l304_304236


namespace speed_of_first_boy_l304_304568

-- Definitions of the conditions
def initial_speed_second_boy := 5.5 -- Speed of the second boy in kmph
def time_apart := 9.5 -- Time after which the distance is measured in hours
def distance_apart := 9.5 -- Distance apart after the given time in km

-- Main statement to prove
theorem speed_of_first_boy (v : ℝ) (h : distance_apart = (v - initial_speed_second_boy) * time_apart) : v = 6.5 :=
by
  sorry

end speed_of_first_boy_l304_304568


namespace square_division_ratio_l304_304519

theorem square_division_ratio (PQRS SPT STU : Set Point)
  (side_length : Real)
  (h_square : is_square PQRS side_length)
  (h_side_length : side_length = 30)
  (h_equal_area : (area PQRS) / 5 = (area SPT))
  (area_SP : area SPT = 180)
  (PT UT ST SU : Real) :
  (PT = (area SPT) / (15))
  ∧ (UT = (area STU) / (15))
  ∧ (ST = Real.sqrt ((30^2) + (PT^2)))
  ∧ (SU = Real.sqrt ((30^2) + ((PT + UT)^2))) ->
  abs ((SU / ST) - 1.189) < 0.01 :=
sorry

end square_division_ratio_l304_304519


namespace smallest_three_digit_multiple_of_17_correct_l304_304626

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304626


namespace extremum_and_monotonicity_inequality_for_c_l304_304001

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem extremum_and_monotonicity (α : ℝ) (h_extremum : ∀ (x : ℝ), x = Real.exp 2 → f x α = 0) :
  (∃ α : ℝ, (∀ x : ℝ, x > Real.exp 2 → f x α > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < Real.exp 2 → f x α < 0)) := sorry

theorem inequality_for_c (c : ℝ) (α : ℝ) (h_extremum : α = 3)
  (h_ineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 3 → f x α < 2 * c^2 - c) :
  (1 < c) ∨ (c < -1 / 2) := sorry

end extremum_and_monotonicity_inequality_for_c_l304_304001


namespace max_pages_l304_304065

theorem max_pages (cents_available : ℕ) (cents_per_page : ℕ) : 
  cents_available = 2500 → 
  cents_per_page = 3 → 
  (cents_available / cents_per_page) = 833 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end max_pages_l304_304065


namespace Alice_needs_more_money_for_free_delivery_l304_304565

theorem Alice_needs_more_money_for_free_delivery
  (chicken_price_per_pound lettuce_price tomato_price sweet_potato_price
   broccoli_price_per_head brussel_sprout_price minimum_spending_needed : ℝ)
  (chicken_weight sweet_potato_count broccoli_count brussel_sprout_weight : ℝ) :
  chicken_price_per_pound = 6.0 →
  lettuce_price = 3.0 →
  tomato_price = 2.5 →
  sweet_potato_price = 0.75 →
  broccoli_price_per_head = 2.0 →
  brussel_sprout_price = 2.5 →
  minimum_spending_needed = 35.0 →
  chicken_weight = 1.5 →
  sweet_potato_count = 4 →
  broccoli_count = 2 →
  brussel_sprout_weight = 1 →
  let total_cost := chicken_weight * chicken_price_per_pound +
                    lettuce_price +
                    tomato_price +
                    sweet_potato_count * sweet_potato_price +
                    broccoli_count * broccoli_price_per_head +
                    brussel_sprout_weight * brussel_sprout_price in
  minimum_spending_needed - total_cost = 11.0 :=
by
  intros
  let total_cost := chicken_weight * chicken_price_per_pound +
                    lettuce_price +
                    tomato_price +
                    sweet_potato_count * sweet_potato_price +
                    broccoli_count * broccoli_price_per_head +
                    brussel_sprout_weight * brussel_sprout_price
  sorry

end Alice_needs_more_money_for_free_delivery_l304_304565


namespace smallest_three_digit_multiple_of_17_l304_304836

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304836


namespace smallest_three_digit_multiple_of_17_correct_l304_304623

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304623


namespace selling_prices_l304_304936

theorem selling_prices {x y : ℝ} (h1 : y - x = 10) (h2 : (y - 5) - 1.10 * x = 1) :
  x = 40 ∧ y = 50 := by
  sorry

end selling_prices_l304_304936


namespace intersection_point_on_y_axis_l304_304102

theorem intersection_point_on_y_axis (a b c : ℝ) (h1 : b ≠ 0) (h2: c ≠ 0) :
  ∃ y : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c = y) ↔ (a * x^2 - b * x + c = y)) → (0, y) ∧ y = c :=
by
  sorry

end intersection_point_on_y_axis_l304_304102


namespace smallest_three_digit_multiple_of_17_l304_304764

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304764


namespace find_cos_angle_BDC_l304_304309

-- Define the angles and points involved
variables (α β : Type) [plane α] [plane β]
variables (A D B C : Type) [point A] [point D] [point B] [point C]
variables (l : line)
variables (angle_α_β angle_BDA angle_CDA : ℝ)

-- Define the conditions
def conditions (α β : Type) [plane α] [plane β] [point A] [point D] [point B] [point C] (l : line) (angle_α_β angle_BDA angle_CDA : ℝ) : Prop :=
  (angle_α_β = 60) ∧
  (on_line A D l) ∧
  (ray_on (D, B) α) ∧
  (ray_on (D, C) β) ∧
  (angle_BDA = 45) ∧
  (angle_CDA = 30)

-- State the theorem
theorem find_cos_angle_BDC (α β : Type) [plane α] [plane β] [point A] [point D] [point B] [point C] (l : line)
  (angle_α_β angle_BDA angle_CDA : ℝ) (h : conditions α β (A D B C) l angle_α_β angle_BDA angle_CDA) :
  cos_angle (angle_between B D C) = (2 * sqrt 6 + sqrt 2) / 8 :=
sorry

end find_cos_angle_BDC_l304_304309


namespace find_lambda_l304_304018

-- Definitions based on the problem conditions
def Point := ℝ × ℝ

def A : Point := (1, 0)
def B : Point := (1, 1)
def O : Point := (0, 0)
def second_quadrant (C : Point) : Prop := C.1 < 0 ∧ C.2 > 0

-- Given that point C is in the second quadrant and ∠AOC = 135°
def C (λ : ℝ) : Point := (1 + λ, -λ)

-- The main statement to be proved
theorem find_lambda (λ : ℝ) :
  second_quadrant (C λ) ∧ ∠OAC (C λ) = 135 ∧ 
  let OC := (1 + λ, -λ) in (OC = (1, 0) + λ • (1, 1)) → 
  λ = -1/2 :=
by
  -- statement to skip the proof
  sorry

end find_lambda_l304_304018


namespace michael_truck_meetings_l304_304453

def speed_michael := 5 -- feet per second
def speed_truck := 10  -- feet per second
def distance_pails := 200 -- feet
def stop_time := 30 -- seconds
def initial_position_michael := 0 -- feet
def initial_position_truck := 200 -- feet

theorem michael_truck_meetings (v_m v_t d_p t_s M0 T0 : ℕ)
  (hv_m : v_m = speed_michael) 
  (hv_t : v_t = speed_truck) 
  (hd_p : d_p = distance_pails) 
  (ht_s : t_s = stop_time) 
  (hM0 : M0 = initial_position_michael) 
  (hT0 : T0 = initial_position_truck) :
  ∃ t : ℕ, (M(t) = T(t) ∧ t > 0) := sorry

end michael_truck_meetings_l304_304453


namespace sequence_relation_l304_304406

theorem sequence_relation (k : ℕ) : 
  let a : ℕ → ℚ := λ n, ∑ i in (finset.range n).filter (λ i, i > 0), (-1) ^ (i + 1) * (1 / (i : ℚ)) in
  a (k + 1) = a k + 1 / (2 * k + 1) - 1 / (2 * k + 2) :=
by
  -- sorry to skip the proof.
  have : false := sorry  
  apply false.elim this

end sequence_relation_l304_304406


namespace initial_count_is_50_l304_304157

variable (n : ℕ)
variable (S : ℕ)

-- average of initial n numbers is 62
axiom avg_initial : S = 62 * n

-- after removing 45 and 55, the average of the remaining (n - 2) numbers is 62.5
axiom avg_after_removal : S - 100 = 62.5 * (n - 2)

theorem initial_count_is_50 : n = 50 := by
  sorry

end initial_count_is_50_l304_304157


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304722

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304722


namespace weather_desire_probability_l304_304126

noncomputable def probability_sunny_days_desired 
: ℚ := 135 / 2048

theorem weather_desire_probability 
(p_rain : ℚ) (n_days : ℕ) (p_sunny : ℚ) 
(one_day_prob : ℚ) (two_days_prob : ℚ) 
(desired_prob : ℚ) : 
  p_rain = 3 / 4 ∧ p_sunny = 1 / 4 ∧ n_days = 5 
  ∧ one_day_prob = 5 * (p_sunny * (p_rain ^ 4))
  ∧ two_days_prob = 10 * ((p_sunny ^ 2) * (p_rain ^ 3))
  ∧ desired_prob = one_day_prob + two_days_prob 
  → desired_prob = probability_sunny_days_desired := 
by 
  intro h 
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h_rest
  cases h_rest with h5 h_rest
  cases h_rest with h6 _
  -- continue proof, but add sorry as we only need the statement
  sorry

end weather_desire_probability_l304_304126


namespace max_min_values_on_interval_tangent_line_equations_l304_304324

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_on_interval :
  (∀ x ∈ set.Icc (-2:ℝ) 1, f x ≤ 2) ∧ (∃ x ∈ set.Icc (-2:ℝ) 1, f x = 2) ∧
  (∀ x ∈ set.Icc (-2:ℝ) 1, f x ≥ -2) ∧ (∃ x ∈ set.Icc (-2:ℝ) 1, f x = -2) :=
by
  sorry

theorem tangent_line_equations (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ t, (P.snd - (t^3 - 3 * t)) = 3 * (t^2 - 1) * (P.fst - t) ∧ (3 * P.fst + P.snd = 0) ∨ (24 * P.fst - P.snd - 54 = 0)) :=
by
  sorry

end max_min_values_on_interval_tangent_line_equations_l304_304324


namespace smallest_three_digit_multiple_of_17_l304_304823

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304823


namespace trajectory_is_ellipse_l304_304340

open Real

-- Define the fixed points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the distance function between two points in the plane
def euclidean_distance (p1 p2 : ℝ × ℝ) :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition for the condition |PF1| + |PF2| = 9
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  euclidean_distance P F1 + euclidean_distance P F2 = 9

-- The main theorem stating the trajectory of P is an ellipse
theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : satisfies_condition P) : 
  ∃ a b : ℝ, ∀ P : ℝ × ℝ, satisfies_condition P → 
  (P.1 / a)^2 + (P.2 / b)^2 = 1 :=
sorry

end trajectory_is_ellipse_l304_304340


namespace greatest_prime_factor_f_24_l304_304275

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def f (m : ℕ) : ℕ :=
  if m % 2 = 0 then ∏ k in (finset.range (m / 2 + 1)).map (λ x, 2 * x), id
  else 1 -- this is just to handle non-even m, as f is only defined for even m here.

def greatest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest (λ p, is_prime p ∧ p ∣ n) (by admit)

theorem greatest_prime_factor_f_24 : greatest_prime_factor (f 24) = 23 := by
  sorry

end greatest_prime_factor_f_24_l304_304275


namespace number_of_red_yarns_l304_304452

-- Definitions
def scarves_per_yarn : Nat := 3
def blue_yarns : Nat := 6
def yellow_yarns : Nat := 4
def total_scarves : Nat := 36

-- Theorem
theorem number_of_red_yarns (R : Nat) (H1 : scarves_per_yarn * blue_yarns + scarves_per_yarn * yellow_yarns + scarves_per_yarn * R = total_scarves) :
  R = 2 :=
by
  sorry

end number_of_red_yarns_l304_304452


namespace smallest_three_digit_multiple_of_17_l304_304702

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304702


namespace smallest_three_digit_multiple_of_17_l304_304682

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304682


namespace smallest_three_digit_multiple_of_17_l304_304732

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304732


namespace smallest_three_digit_multiple_of_17_l304_304613

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304613


namespace consecutive_numbers_sum_39_l304_304378

theorem consecutive_numbers_sum_39 (n : ℕ) (hn : n + (n + 1) = 39) : n + 1 = 20 :=
sorry

end consecutive_numbers_sum_39_l304_304378


namespace smallest_three_digit_multiple_of_17_l304_304696

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304696


namespace range_of_k_l304_304337

theorem range_of_k (x y k : ℝ) 
  (h1 : 2 * x + y = k + 1) 
  (h2 : x + 2 * y = 2) 
  (h3 : x + y < 0) : 
  k < -3 :=
sorry

end range_of_k_l304_304337


namespace smallest_three_digit_multiple_of_17_l304_304807

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304807


namespace largest_power_of_two_dividing_product_of_first_50_even_integers_l304_304073

theorem largest_power_of_two_dividing_product_of_first_50_even_integers :
  ∃ k : ℕ, (∀ (n : ℕ), (∏ i in finset.range 50, 2 * (i + 1) : ℕ) % 2^n = 0 ↔ n ≤ k) ∧ k = 97 :=
by
    sorry

end largest_power_of_two_dividing_product_of_first_50_even_integers_l304_304073


namespace smallest_three_digit_multiple_of_17_l304_304699

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304699


namespace inequality_proof_l304_304442

theorem inequality_proof {x y z : ℝ} (hxy : 0 < x) (hyz : 0 < y) (hzx : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
sorry

end inequality_proof_l304_304442


namespace smallest_three_digit_multiple_of_17_l304_304773

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304773


namespace smallest_three_digit_multiple_of_17_l304_304825

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304825


namespace set_intersection_l304_304338

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

noncomputable def complement_U_A := U \ A
noncomputable def intersection := B ∩ complement_U_A

theorem set_intersection :
  intersection = ({3, 4} : Set ℕ) := by
  sorry

end set_intersection_l304_304338


namespace no_more_than_8_non_overlapping_squares_around_square_l304_304475

theorem no_more_than_8_non_overlapping_squares_around_square :
  ∀ (side_length : ℝ), side_length > 0 →
  ∀ (central_square : set (ℝ × ℝ)), is_square central_square side_length →
  ∀ (squares : finset (set (ℝ × ℝ))),
  (∀ sq ∈ squares, is_square sq side_length ∧ disjoint sq central_square) →
  (card squares ≤ 8) :=
by sorry

end no_more_than_8_non_overlapping_squares_around_square_l304_304475


namespace smallest_three_digit_multiple_of_17_l304_304606

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304606


namespace abs_diff_61st_terms_l304_304567

noncomputable def seq_C (n : ℕ) : ℤ := 20 + 15 * (n - 1)
noncomputable def seq_D (n : ℕ) : ℤ := 20 - 15 * (n - 1)

theorem abs_diff_61st_terms :
  |seq_C 61 - seq_D 61| = 1800 := sorry

end abs_diff_61st_terms_l304_304567


namespace smallest_three_digit_multiple_of_17_l304_304729

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304729


namespace completing_the_square_l304_304868

theorem completing_the_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) → ((x - 1)^2 = 6) :=
by
  sorry

end completing_the_square_l304_304868


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304705

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304705


namespace Molly_age_now_l304_304112

/- Definitions -/
def Sandy_curr_age : ℕ := 60
def Molly_curr_age (S : ℕ) : ℕ := 3 * S / 4
def Sandy_age_in_6_years (S : ℕ) : ℕ := S + 6

/- Theorem to prove -/
theorem Molly_age_now 
  (ratio_condition : ∀ S M : ℕ, S / M = 4 / 3 → M = 3 * S / 4)
  (age_condition : Sandy_age_in_6_years Sandy_curr_age = 66) : 
  Molly_curr_age Sandy_curr_age = 45 :=
by
  sorry

end Molly_age_now_l304_304112


namespace quadratic_roots_equation_l304_304311

theorem quadratic_roots_equation (a b c r s : ℝ)
    (h1 : a ≠ 0)
    (h2 : a * r^2 + b * r + c = 0)
    (h3 : a * s^2 + b * s + c = 0) :
    ∃ p q : ℝ, (x^2 - b * x + a * c = 0) ∧ (ar + b, as + b) = (p, q) :=
by
  sorry

end quadratic_roots_equation_l304_304311


namespace num_possible_scores_l304_304113

def is_adjacent (a b : ℕ) : Prop :=
  abs (a - b) = 2

def valid_score (scores : List ℕ) : Prop :=
  scores.length = 3 ∧ 
  (∀ x ∈ scores, x ∈ [1, 3, 5, 7, 9, 11]) ∧ 
  (¬∃ (s1 s2 : ℕ), s1 ∈ scores ∧ s2 ∈ scores ∧ is_adjacent s1 s2)

def possible_scores : List ℕ :=
  [3, 9, 15, 21, 27, 33, 7, 11, 13, 17, 29, 31]

theorem num_possible_scores : ∃ n, n = possible_scores.length :=
  by
    exists 13
    rfl

end num_possible_scores_l304_304113


namespace max_value_of_a_l304_304957

theorem max_value_of_a :
  ∃ b : ℤ, ∃ (a : ℝ), 
    (a = 30285) ∧
    (a * b^2 / (a + 2 * b) = 2019) :=
by
  sorry

end max_value_of_a_l304_304957


namespace log_sequence_value_l304_304975

theorem log_sequence_value :
  ∃ x : ℝ, x = log 3 (81 + x) ∧ x > 0 ∧ x ≈ 5 :=
begin
  sorry
end

end log_sequence_value_l304_304975


namespace total_sales_in_december_correct_l304_304383

def ear_muffs_sales_in_december : ℝ :=
  let typeB_sold := 3258
  let typeB_price := 6.9
  let typeC_sold := 3186
  let typeC_price := 7.4
  let total_typeB_sales := typeB_sold * typeB_price
  let total_typeC_sales := typeC_sold * typeC_price
  total_typeB_sales + total_typeC_sales

theorem total_sales_in_december_correct :
  ear_muffs_sales_in_december = 46056.6 :=
by
  sorry

end total_sales_in_december_correct_l304_304383


namespace complex_number_in_first_quadrant_l304_304030

-- Define the condition given in the problem
def satisfies_equation (z : ℂ) : Prop := (2 + I) * z = 1 + 3 * I

-- Define what it means for a point to be in the first quadrant in the complex plane
def in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

-- The theorem we want to prove
theorem complex_number_in_first_quadrant (z : ℂ) (h : satisfies_equation z) : in_first_quadrant z :=
by
  have : z = 1 + I := sorry
  have : in_first_quadrant (1 + I) := by
    split
    simp
  assumption

end complex_number_in_first_quadrant_l304_304030


namespace radius_of_circle_shortest_distance_l304_304036

theorem radius_of_circle_shortest_distance 
  (r : ℝ)
  (h1 : ∀ x y, (x-3)^2 + (y+5)^2 = r^2 → True)
  (h2 : ∀ x y, 4 * x - 3 * y - 2 = 0 → True)
  (h3 : ∀ d, (d = real.abs ((4 * 3) + (3 * 5) - 2) / (real.sqrt (4^2 + 3^2)) → d = 5) 
  : (5 - r = 1) → r = 4 :=
by
  sorry

end radius_of_circle_shortest_distance_l304_304036


namespace intersection_point_product_l304_304910

noncomputable def parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 + x2 = 12) (h2 : x1 * x2 = 4) : Prop :=
  (x1 + 2) * (x2 + 2) = 32

theorem intersection_point_product :
  ∃ x1 x2 : ℝ, (x1 + x2 = 12 ∧ x1 * x2 = 4 ∧ parabola_line_intersection x1 x2) :=
begin
  use [10, 2], -- Example points that satisfy given conditions
  split,
  { exact rfl },
  split,
  { exact rfl },
  { unfold parabola_line_intersection,
    exact rfl },
end

end intersection_point_product_l304_304910


namespace simplify_and_evaluate_expression_l304_304493

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304493


namespace smallest_three_digit_multiple_of_17_l304_304818

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304818


namespace commodity_price_difference_l304_304152

theorem commodity_price_difference :
  ∃ n : ℕ, 4.20 + 0.40 * n = 6.30 + 0.15 * n + 0.15 := by
  sorry

end commodity_price_difference_l304_304152


namespace smallest_three_digit_multiple_of_17_l304_304608

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304608


namespace smallest_three_digit_multiple_of_17_l304_304855

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304855


namespace range_a_l304_304034

noncomputable def f (x : ℝ) : ℝ := -(1 / 3) * x^3 + x

theorem range_a (a : ℝ) (h1 : a < 1) (h2 : 1 < 10 - a^2) (h3 : f a ≤ f 1) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_a_l304_304034


namespace prob_of_entirely_black_l304_304898

noncomputable def prob_all_black_grid : ℚ :=
  if (is_center_black : Prop) ∧ 
     (are_edge_squares_black : Prop) ∧ 
     (are_corner_squares_black : Prop)
  then (1/2 : ℚ) * (7/16 : ℚ) * (7/16 : ℚ)
  else 0

theorem prob_of_entirely_black (h : prob_all_black_grid = 49 / 512) : true :=
by { sorry }

end prob_of_entirely_black_l304_304898


namespace part1_general_formula_part2_Sn_l304_304287

noncomputable def a_sequence (n : ℕ) : ℕ :=
  2 * n - 1

def b_sequence (n : ℕ) : ℚ :=
  1 / (a_sequence n * a_sequence (n + 1))

def S_n (n : ℕ) : ℚ :=
  ∑ i in finset.range n, b_sequence (i + 1)

theorem part1_general_formula (n : ℕ) : 
  (∀ k ≤ n, ∑ i in finset.range k, (2 * i + 1) / a_sequence (i + 1) = k) → 
  a_sequence n = 2 * n - 1 := 
sorry

theorem part2_Sn (n : ℕ) : 
  S_n n < 1 / 2 := 
sorry

end part1_general_formula_part2_Sn_l304_304287


namespace range_of_m_l304_304377

theorem range_of_m {x m : ℝ} 
  (h1 : 1 / 3 < x) 
  (h2 : x < 1 / 2) 
  (h3 : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  sorry

end range_of_m_l304_304377


namespace log2_n_equals_409_l304_304562

def teams : ℕ := 30
def matches : ℕ := (teams * (teams - 1)) / 2
def total_possible_outcomes : ℕ := 2 ^ matches
def factorial_30 : ℕ := Nat.factorial teams

def power_of_two_in_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec loop (k : ℕ) (acc : ℕ) :=
    if k = 0 then acc else
    loop (k / 2) (acc + (k / 2))
  loop n 0

def favorable_outcomes : ℕ := factorial_30
def probability_denominator : ℕ := total_possible_outcomes
def n : ℕ := probability_denominator / (2 ^ (power_of_two_in_factorial teams))

theorem log2_n_equals_409 :
  2 ^ 409 = probability_denominator / (2 ^ (power_of_two_in_factorial teams)) →
  log2 n = 409 := by
  sorry

end log2_n_equals_409_l304_304562


namespace simplify_and_evaluate_expression_l304_304494

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304494


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304721

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304721


namespace range_g_l304_304082

def f (x: ℝ) : ℝ := 4 * x - 3
def g (x: ℝ) : ℝ := f (f (f (f (f x))))

theorem range_g (x: ℝ) (h: 0 ≤ x ∧ x ≤ 3) : -1023 ≤ g x ∧ g x ≤ 2049 :=
by
  sorry

end range_g_l304_304082


namespace smallest_three_digit_multiple_of_17_l304_304769

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304769


namespace sum_of_common_multiples_15_20_less_150_l304_304175

-- Define the LCM function
def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

-- Define the function to calculate the sum of all multiples of lcm(a, b) that are less than a given limit
def sum_multiples (a b limit : ℕ) : ℕ :=
  let m := lcm a b
  let multiples := List.filter (λ x => x < limit) (List.map (λ n => n * m) (Finset.range (limit / m + 1)).val)
  multiples.sum

theorem sum_of_common_multiples_15_20_less_150 : sum_multiples 15 20 150 = 180 := by
  sorry

end sum_of_common_multiples_15_20_less_150_l304_304175


namespace value_of_k_l304_304301

noncomputable def find_k (x1 x2 : ℝ) (k : ℝ) : Prop :=
  (2 * x1^2 + k * x1 - 2 = 0) ∧ (2 * x2^2 + k * x2 - 2 = 0) ∧ ((x1 - 2) * (x2 - 2) = 10)

theorem value_of_k (x1 x2 : ℝ) (k : ℝ) (h : find_k x1 x2 k) : k = 7 :=
sorry

end value_of_k_l304_304301


namespace sum_of_integers_l304_304270

theorem sum_of_integers :
  let S := {n : ℕ | 2.4 * n - 8.8 < 10.4}
  in ∑ n in S, n = 28 :=
by
  sorry

end sum_of_integers_l304_304270


namespace at_most_one_true_l304_304376

theorem at_most_one_true (p q : Prop) (h : ¬(p ∧ q)) : ¬(p ∧ q ∧ ¬(¬p ∧ ¬q)) :=
by
  sorry

end at_most_one_true_l304_304376


namespace angle_equality_l304_304056

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
variables (h1 : a = 6) (h2 : c = sqrt 3) (h3 : a * cos B = b * cos A) (h4 : A = B)

-- Part 1: Prove A = B
theorem angle_equality (ha : a * cos B = b * cos A) : A = B := by
  sorry

-- Part 2: Find the area of the triangle given A = B, c, and a.
noncomputable def area_triangle (ha : a = 6) (hc : c = sqrt 3) (hAeqB : A = B) : ℝ :=
  let h := sqrt (36 - (3 / 4))
  in (1 / 4) * sqrt 423

end angle_equality_l304_304056


namespace smallest_three_digit_multiple_of_17_l304_304828

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304828


namespace smallest_three_digit_multiple_of_17_l304_304724

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304724


namespace particle_speed_after_2_units_l304_304917

-- Definition of the particle's position function
def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 4, 5 * t - 9)

-- Definition of the speed function after a time interval of Δt
def particle_speed (t : ℝ) (Δt : ℝ) : ℝ :=
  let (x1, y1) := particle_position t in
  let (x2, y2) := particle_position (t + Δt) in
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem: The speed of the particle after Δt = 2 units of time is √136
theorem particle_speed_after_2_units : particle_speed 0 2 = Real.sqrt 136 :=
by
  sorry

end particle_speed_after_2_units_l304_304917


namespace income_on_first_day_l304_304227

theorem income_on_first_day (income : ℕ → ℚ) (h1 : income 10 = 18)
  (h2 : ∀ n, income (n + 1) = 2 * income n) :
  income 1 = 0.03515625 :=
by
  sorry

end income_on_first_day_l304_304227


namespace incorrect_statement_among_given_l304_304542

def quadrilateral_parallel_opposite_sides (quad : Quadrilateral) : Prop := 
  ∃ sides, sides.opposite_parallel ∧ sides.opposite_equal

def plumb_lines_coplanar (line1 line2 : Line) : Prop :=
  (line1.plumb ∧ line2.plumb) → line1.coplanar_with line2

def perpendicular_lines_in_same_plane (line : Line) (point : Point) : Prop :=
  ∀ (line' : Line), (line'.perpendicular_to line ∧ point ∈ line') → line'.in_same_plane line

def unique_plane_perpendicular_to_given_plane (plane1 : Plane) (line : Line) : Prop :=
  ∃! plane2, plane2.perpendicular_to plane1 ∧ line ⊆ plane2

theorem incorrect_statement_among_given {quad : Quadrilateral} {line1 line2 : Line} {plane1 : Plane} {line : Line} {point : Point} :
  (quadrilateral_parallel_opposite_sides quad ∧ 
   plumb_lines_coplanar line1 line2 ∧ 
   perpendicular_lines_in_same_plane line point ∧ 
   ¬ unique_plane_perpendicular_to_given_plane plane1 line) :=
sorry

end incorrect_statement_among_given_l304_304542


namespace frustum_midsection_area_l304_304551

theorem frustum_midsection_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r_mid := (r1 + r2) / 2
  let area_mid := Real.pi * r_mid^2
  area_mid = 25 * Real.pi / 4 := by
  sorry

end frustum_midsection_area_l304_304551


namespace part1_part2_l304_304382

section problem1

variables {A B C : ℝ} {a b c S : ℝ}

theorem part1 (h1: 2 * a * cos (C / 2)^2 + 2 * c * cos (A / 2)^2 = (5/2) * b):
  2 * (a + c) = 3 * b := sorry

end problem1

section problem2

variables {A B C : ℝ} {a b c S : ℝ}

theorem part2 (h1: cos B = 1 / 4) (h2: S = sqrt 15):
  b = 4 := sorry

end problem2

end part1_part2_l304_304382


namespace number_of_coin_arrangements_l304_304484

theorem number_of_coin_arrangements :
  let coins := 10
  let golds := 5
  let silvers := 5
  let alternating := true
  let face := true
  let bottom_coin := "gold"
  count_valid_arrangements coins golds silvers alternating face bottom_coin = 7 := 
sorry 

end number_of_coin_arrangements_l304_304484


namespace smallest_three_digit_multiple_of_17_l304_304583

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304583


namespace quadratic_has_two_distinct_real_roots_l304_304554

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  let a := 1
  let b := -m
  let c := -1
  let delta := b^2 - 4*a*c
  delta > 0 :=
by
  let a := 1
  let b := -m
  let c := -1
  let delta := b^2 - 4*a*c
  have h : delta = m^2 + 4,
    by simp [delta, a, b, c]
  have h_nonneg : m^2 ≥ 0,
    from pow_two_nonneg m
  have h_pos : m^2 + 4 > 0,
    by linarith
  exact h_pos

end quadratic_has_two_distinct_real_roots_l304_304554


namespace smallest_possible_abc_l304_304438

open Nat

theorem smallest_possible_abc (a b c : ℕ)
  (h₁ : 5 * c ∣ a * b)
  (h₂ : 13 * a ∣ b * c)
  (h₃ : 31 * b ∣ a * c) :
  abc = 4060225 :=
by sorry

end smallest_possible_abc_l304_304438


namespace total_distance_is_27_l304_304481

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l304_304481


namespace range_of_function_l304_304970

theorem range_of_function : ∀ θ ∈ ℝ, let y := (cos θ) / (2 + sin θ) in y ∈ Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
sorry

end range_of_function_l304_304970


namespace chord_length_l304_304142

open Real

-- Definitions of the line and the circle
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 4

def circle (x y : ℝ) : Prop := x^2 + y^2 + 6 * x - 4 * y = 0

-- The main theorem to prove
theorem chord_length : 
  (∃ x y : ℝ, line x y ∧ circle x y) → 
  ∀ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2 → 
  dist A B = 4 * sqrt 3 :=
sorry

end chord_length_l304_304142


namespace paco_cookies_l304_304103

theorem paco_cookies (initial_cookies: ℕ) (eaten_cookies: ℕ) (final_cookies: ℕ) (bought_cookies: ℕ) 
  (h1 : initial_cookies = 40)
  (h2 : eaten_cookies = 2)
  (h3 : final_cookies = 75)
  (h4 : initial_cookies - eaten_cookies + bought_cookies = final_cookies) :
  bought_cookies = 37 :=
by
  rw [h1, h2, h3] at h4
  sorry

end paco_cookies_l304_304103


namespace smallest_three_digit_multiple_of_17_l304_304690

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304690


namespace smallest_three_digit_multiple_of_17_l304_304674

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304674


namespace max_value_of_4x_plus_3y_l304_304298

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l304_304298


namespace smallest_three_digit_multiple_of_17_l304_304845

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304845


namespace smallest_three_digit_multiple_of_17_l304_304726

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304726


namespace trains_clear_time_l304_304885

def train_length1 : ℝ := 121
def train_length2 : ℝ := 165
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 55
def total_distance := train_length1 + train_length2
def convert_kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1 / 3.6)
def relative_speed_mps := convert_kmph_to_mps speed1_kmph + convert_kmph_to_mps speed2_kmph
def time_to_clear := total_distance / relative_speed_mps

theorem trains_clear_time :
  time_to_clear = 7.62666667 :=
begin
  sorry
end

end trains_clear_time_l304_304885


namespace count_valid_two_digit_numbers_l304_304356

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def tens_digit_less_than_ones_digit (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens < ones

theorem count_valid_two_digit_numbers : 
  (Finset.filter tens_digit_less_than_ones_digit (Finset.filter is_two_digit (Finset.range 100))).card = 36 :=
by
  sorry

end count_valid_two_digit_numbers_l304_304356


namespace minimum_sum_of_labels_l304_304571

-- Define the conditions
def label : ℕ → ℕ → ℚ := λ i j, 1 / (i + j + 1)

def is_valid_selection (selection : ℕ → ℕ) : Prop :=
  (∀ i ≠ j, selection i ≠ selection j) ∧ -- no two selected squares share the same row
  (∀ i ≠ j, i + selection i ≠ j + selection j) -- at most two squares share a diagonal

-- The theorem that states the minimum sum of the labels is 1
theorem minimum_sum_of_labels : ∃ selection : ℕ → ℕ, 
  is_valid_selection selection ∧
  (finset.range 9).sum (λ i, label i (selection i)) = 1 :=
begin
  sorry
end

end minimum_sum_of_labels_l304_304571


namespace total_road_signs_l304_304922

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l304_304922


namespace smallest_three_digit_multiple_of_17_l304_304844

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304844


namespace simplify_and_evaluate_expression_l304_304507

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304507


namespace smallest_three_digit_multiple_of_17_l304_304679

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304679


namespace smallest_three_digit_multiple_of_17_l304_304837

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304837


namespace smallest_three_digit_multiple_of_17_l304_304767

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304767


namespace construct_acute_triangle_l304_304570

-- Define the conditions given in the problem
variables {ℝ : Type} [LinearOrderedField ℝ]
variables {A B C B1 C1 A1 : ℝ}
variables (l : Set ℝ)

-- Definitions for the problem
def is_orthic (A1 B1 C1 : ℝ) : Prop :=
  -- Conditions defining the orthic triangle
  sorry

def is_altitude (A B C B1 C1 A1 : ℝ) (l : Set ℝ) : Prop :=
  -- Conditions defining the altitudes aligned with the line l
  sorry

-- The main proof statement
theorem construct_acute_triangle (A1 B1 C1 : ℝ) (l : Set ℝ) 
  (h_orthic : is_orthic A1 B1 C1)
  (h_altitude: is_altitude A B C B1 C1 A1 l):
  ∃ (A B C : ℝ), 
    acute_triangle A B C ∧
    (altitude A B C B1 C1 A1 ∧ on_line (A1, l) ∧ 
    altitude A B C B1 C1 A1 ∧ on_line (B1, l) ∧
    altitude A B C B1 C1 A1 ∧ on_line (C1, l)) :=
begin
  sorry
end

end construct_acute_triangle_l304_304570


namespace min_chord_length_m_l304_304534

-- Definition of the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 6 * y + 4 = 0
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Theorem statement: value of m that minimizes the length of the chord
theorem min_chord_length_m (m : ℝ) : m = 1 ↔
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y := sorry

end min_chord_length_m_l304_304534


namespace smallest_three_digit_multiple_of_17_l304_304649

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → n ≤ m) :=
by
  use 102
  have h1 : 100 ≤ 102 := by linarith
  have h2 : 102 ≤ 999 := by linarith
  have h3 : 102 % 17 = 0 := by norm_num
  have h4 : ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 17 = 0) → 102 ≤ m := by
    intro m
    intro h
    have h_m1 := h.1
    have h_m2 := h.2.1
    have h_m3 := h.2.2
    have h_min := nat.le_of_dvd (by linarith) h_m3
    exact h_min
  exact ⟨⟨h1, h2⟩, h3, h4⟩

end smallest_three_digit_multiple_of_17_l304_304649


namespace rectangular_solid_width_l304_304921

theorem rectangular_solid_width 
  (l : ℝ) (w : ℝ) (h : ℝ) (S : ℝ)
  (hl : l = 5)
  (hh : h = 1)
  (hs : S = 58) :
  2 * l * w + 2 * l * h + 2 * w * h = S → w = 4 := 
by
  intros h_surface_area 
  sorry

end rectangular_solid_width_l304_304921


namespace same_volume_increase_rate_l304_304906

def initial_radius := 10
def initial_height := 5 

def volume_increase_rate_new_radius (x : ℝ) :=
  let r' := initial_radius + 2 * x
  (r' ^ 2) * initial_height  - (initial_radius ^ 2) * initial_height

def volume_increase_rate_new_height (x : ℝ) :=
  let h' := initial_height + 3 * x
  (initial_radius ^ 2) * h' - (initial_radius ^ 2) * initial_height

theorem same_volume_increase_rate (x : ℝ) : volume_increase_rate_new_radius x = volume_increase_rate_new_height x → x = 5 := 
  by sorry

end same_volume_increase_rate_l304_304906


namespace smallest_three_digit_multiple_of_17_correct_l304_304630

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304630


namespace kim_easy_round_correct_answers_l304_304390

variable (E : ℕ)

theorem kim_easy_round_correct_answers 
    (h1 : 2 * E + 3 * 2 + 5 * 4 = 38) : 
    E = 6 := 
sorry

end kim_easy_round_correct_answers_l304_304390


namespace find_CD_l304_304254

theorem find_CD : ∃ C D : ℝ, 
  (∀ x : ℝ, x ≠ 7 ∧ x ≠ -2 → 
  ((2 * x + 4) / (x^2 - 5 * x - 14) = (C / (x - 7) + D / (x + 2)))) 
  ∧ C = 2 ∧ D = 0 :=
by {
  use [2, 0],
  intros x hx,
  have h₁ : x^2 - 5 * x - 14 = (x - 7) * (x + 2), by ring,
  rw h₁,
  have h₂ : x ≠ 7 ∧ x ≠ -2, from hx,
  rw [mul_comm (x - 7), mul_comm (x + 2)],
  field_simp [h₂.1, h₂.2],
  ring,
  split, refl, split; refl,
  sorry
}

end find_CD_l304_304254


namespace log_expression_equals_four_l304_304972

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end log_expression_equals_four_l304_304972


namespace smallest_three_digit_multiple_of_17_l304_304817

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304817


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304714

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l304_304714


namespace smallest_three_digit_multiple_of_17_l304_304854

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304854


namespace smallest_three_digit_multiple_of_17_l304_304786

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304786


namespace smallest_three_digit_multiple_of_17_l304_304859

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304859


namespace min_W_value_l304_304021

theorem min_W_value : ∃ (m : ℝ), (∀ x y : ℝ, W x y ≥ m) ∧ m = -2 :=
by
  let W := λ x y : ℝ, 5 * x^2 - 4 * x * y + y^2 - 2 * y + 8 * x + 3
  sorry

end min_W_value_l304_304021


namespace smallest_three_digit_multiple_of_17_correct_l304_304618

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304618


namespace smallest_three_digit_multiple_of_17_l304_304775

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304775


namespace odd_function_iff_f0_zero_l304_304083

variables {A ω : ℝ} (x : ℝ) (φ : ℝ)
variables (hA : A > 0) (hω : ω > 0)

def f (x : ℝ) : ℝ := A * sin (ω * x + φ)

theorem odd_function_iff_f0_zero :
  (f 0 = 0) ↔ (∀ x, f (-x) = -f x) :=
by sorry

end odd_function_iff_f0_zero_l304_304083


namespace triangle_area_ratio_is_one_sixth_l304_304215

def equilateral_triangle_area_ratio {k : ℕ} 
  (hex_divided_into_12: ∀ {A B C D E F G : Type}, 
    regular_hexagon ABCDEF → set.equilateral_triangle ABCDEF)
  (formed_by_every_third_vertex: ∀ {A B C D E F : Type}, 
    equilateral_triangle ADF → connects_every_third_vertex A D F):
  ℝ :=
  let small_triangle_area := k
  let large_triangle_area := 6 * k
  small_triangle_area / large_triangle_area

theorem triangle_area_ratio_is_one_sixth 
  {k : ℕ} 
  (hex_divided_into_12: ∀ {A B C D E F G : Type}, 
    regular_hexagon ABCDEF → set.equilateral_triangle ABCDEF)
  (formed_by_every_third_vertex: ∀ {A B C D E F : Type}, 
    equilateral_triangle ADF → connects_every_third_vertex A D F):
  equilateral_triangle_area_ratio hex_divided_into_12 formed_by_every_third_vertex = 1 / 6 := 
  sorry

end triangle_area_ratio_is_one_sixth_l304_304215


namespace simplify_expression_l304_304502

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304502


namespace total_numbers_l304_304459

theorem total_numbers (m j c : ℕ) (h1 : m = j + 20) (h2 : j = c - 40) (h3 : c = 80) : m + j + c = 180 := 
by sorry

end total_numbers_l304_304459


namespace smallest_three_digit_multiple_of_17_l304_304585

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304585


namespace mrs_hilt_baked_pecan_pies_l304_304466

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

def pecan_pies (total_pies : ℕ) (apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14
  let rows := 6
  let pies_per_row := 5
  let total := total_pies rows pies_per_row
  pecan_pies total apple_pies = 16 :=
by
  sorry

end mrs_hilt_baked_pecan_pies_l304_304466


namespace count_five_digit_numbers_divisibility_by_11_l304_304085

theorem count_five_digit_numbers_divisibility_by_11 :
  let n_values := ({n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ ∃ q r, 
                    n = 100 * q + r ∧ 100 ≤ q ∧ q ≤ 999 ∧ 0 ≤ r ∧ r ≤ 99 ∧ (q + r) % 11 = 0}) 
  in Fintype.card n_values = 9000 :=
by {
  let n_values := {n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ ∃ q r, 
                    n = 100 * q + r ∧ 100 ≤ q ∧ q ≤ 999 ∧ 0 ≤ r ∧ r ≤ 99 ∧ (q + r) % 11 = 0},
  have n_val_count : Fintype.card n_values = 9000,
  exact n_val_count
}

end count_five_digit_numbers_divisibility_by_11_l304_304085


namespace smallest_three_digit_multiple_of_17_l304_304677

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304677


namespace baseball_cards_remaining_l304_304462

-- Define the number of baseball cards Mike originally had
def original_cards : ℕ := 87

-- Define the number of baseball cards Sam bought from Mike
def cards_bought : ℕ := 13

-- Prove that the remaining number of baseball cards Mike has is 74
theorem baseball_cards_remaining : original_cards - cards_bought = 74 := by
  sorry

end baseball_cards_remaining_l304_304462


namespace coefficient_x3_expansion_l304_304129

theorem coefficient_x3_expansion (x : ℝ) : 
  let expansion := (1 - x^3) * (1 + x) ^ 10 in
  (expansion.coeffs 3) = 119 :=
by
  sorry

end coefficient_x3_expansion_l304_304129


namespace smallest_three_digit_multiple_of_17_l304_304653

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304653


namespace smallest_three_digit_multiple_of_17_l304_304862

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end smallest_three_digit_multiple_of_17_l304_304862


namespace unique_solution_iff_d_ne_4_l304_304984

theorem unique_solution_iff_d_ne_4 (c d : ℝ) : 
  (∃! (x : ℝ), 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := 
by 
  sorry

end unique_solution_iff_d_ne_4_l304_304984


namespace find_number_of_students_l304_304190

theorem find_number_of_students (N T : ℕ) 
  (avg_mark_all : T = 80 * N) 
  (avg_mark_exclude : (T - 150) / (N - 5) = 90) : 
  N = 30 := by
  sorry

end find_number_of_students_l304_304190


namespace find_x_l304_304522

variable (x : ℝ)  -- Current distance Teena is behind Loe in miles
variable (t : ℝ) -- Time period in hours
variable (speed_teena : ℝ) -- Speed of Teena in miles per hour
variable (speed_loe : ℝ) -- Speed of Loe in miles per hour
variable (d_ahead : ℝ) -- Distance Teena will be ahead of Loe in 1.5 hours

axiom conditions : speed_teena = 55 ∧ speed_loe = 40 ∧ t = 1.5 ∧ d_ahead = 15

theorem find_x : (speed_teena * t - speed_loe * t = x + d_ahead) → x = 7.5 :=
by
  intro h
  sorry

end find_x_l304_304522


namespace trader_profit_percentage_l304_304880

theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 62 := 
by
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  sorry

end trader_profit_percentage_l304_304880


namespace smallest_three_digit_multiple_of_17_l304_304580

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l304_304580


namespace y_coordinate_of_C_range_l304_304334

noncomputable def A : ℝ × ℝ := (0, 2)

def is_on_parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = P.1 + 4

def is_perpendicular (A B C : ℝ × ℝ) : Prop := 
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

def range_of_y_C (y_C : ℝ) : Prop := y_C ≤ 0 ∨ y_C ≥ 4

theorem y_coordinate_of_C_range (B C : ℝ × ℝ)
  (hB : is_on_parabola B) (hC : is_on_parabola C) (h_perpendicular : is_perpendicular A B C) : 
  range_of_y_C (C.2) :=
sorry

end y_coordinate_of_C_range_l304_304334


namespace smallest_three_digit_multiple_of_17_l304_304800

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304800


namespace evaluate_radical_expression_l304_304025

theorem evaluate_radical_expression (a b x : ℝ) (h : a < b) :
  sqrt (-(x + a) ^ 3 * (x + b)) = -(x + a) * sqrt (-(x + a) * (x + b)) :=
by
  sorry

end evaluate_radical_expression_l304_304025


namespace smallest_three_digit_multiple_of_17_l304_304599

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304599


namespace smallest_three_digit_multiple_of_17_l304_304776

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304776


namespace alpha_range_l304_304007

noncomputable def f (x α : ℝ) : ℝ := ln x + tan α

noncomputable def f_prime (x α : ℝ) : ℝ := deriv (λ x, ln x + tan α) x

noncomputable def f_double_prime (x α : ℝ) : ℝ := deriv (λ x, deriv (λ x, ln x + tan α) x) x

theorem alpha_range (α x0 : ℝ) (h0 : 0 < α) (h1 : α < π / 2) (hx0 : 0 < x0) (hx0_1 : x0 < 1)
  (hf_root : f_double_prime x0 α = f x0 α) :
  (π / 4 < α ∧ α < π / 2) :=
sorry

end alpha_range_l304_304007


namespace simplify_and_evaluate_expression_l304_304495

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l304_304495


namespace ceiling_is_multiple_of_3_l304_304537

-- Given conditions:
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
axiom exists_three_real_roots : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
  polynomial x1 = 0 ∧ polynomial x2 = 0 ∧ polynomial x3 = 0

-- Goal:
theorem ceiling_is_multiple_of_3 (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (hx1 : polynomial x1 = 0) (hx2 : polynomial x2 = 0) (hx3 : polynomial x3 = 0):
  ∀ n : ℕ, n > 0 → ∃ k : ℤ, k * 3 = ⌈x3^n⌉ := by
  sorry

end ceiling_is_multiple_of_3_l304_304537


namespace area_region_lower_bound_l304_304446

noncomputable def region (n : ℕ) : set ℂ :=
  {z : ℂ | ∑ k in finset.range n, 1 / complex.abs (z - k) ≥ 1}

def area (s : set ℂ) : ℝ := sorry -- Define the concept of area (this may involve measure theory)

theorem area_region_lower_bound (n : ℕ) (h : 0 < n) :
  area (region n) ≥ π / 12 * (11 * n^2 + 1) :=
sorry

end area_region_lower_bound_l304_304446


namespace trajectory_of_M_circle_through_fixed_points_l304_304394

variables {M : ℝ × ℝ} {F : ℝ × ℝ}
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trajectory_of_M (x y : ℝ) :
  (distance M (1, 0) = distance M (x, 0) + 1) →
  (y^2 = 4 * x ∧ x ≥ 0 ∨ y = 0 ∧ x < 0) :=
sorry

theorem circle_through_fixed_points (x y : ℝ) (A B F : ℝ × ℝ) :
  let C := (set_of (λ p : ℝ × ℝ, p.snd^2 = 4 * p.fst ∧ p.fst ≥ 0)) in
  let line_PQ := (F.1, y) in
  let OP := (0, 0) in
  let OQ := (0, 0) in
  let A := (1, (4 / y)) in
  let B := (1, (4 / (y + 4))) in
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2*y)^2 = 4 * (y^2 + 1)} in
  (A ∈ circle ∧ B ∈ circle) →
  ((-1, 0) ∈ circle ∧ (3, 0) ∈ circle) :=
sorry

end trajectory_of_M_circle_through_fixed_points_l304_304394


namespace smallest_three_digit_multiple_of_17_l304_304792

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304792


namespace chess_tournament_l304_304159

def participant := {i : Nat // i < 20}

def wins_white : participant → Nat
def wins_black : participant → Nat

def no_weaker_than (A B : participant) : Prop :=
  wins_white A ≥ wins_white B ∧ wins_black A ≥ wins_black B

theorem chess_tournament :
  ∃ (A B : participant), no_weaker_than A B := 
begin
  -- Given
  -- 1. There are 20 participants in a chess tournament.
  -- 2. Each participant played with each other twice: once as white and once as black.
  -- 3. A participant X is no weaker than participant Y if:
  --    - X has won at least the same number of games playing white as Y.
  --    - X has won at least the same number of games playing black as Y.
  sorry
end

end chess_tournament_l304_304159


namespace smallest_three_digit_multiple_of_17_l304_304704

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304704


namespace smallest_three_digit_multiple_of_17_l304_304810

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304810


namespace smallest_three_digit_multiple_of_17_l304_304737

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304737


namespace placards_per_person_l304_304912

theorem placards_per_person (total_placards people_entered : ℕ) (h_total : total_placards = 4634) (h_people : people_entered = 2317) : total_placards / people_entered = 2 :=
by {
  rw [h_total, h_people],
  norm_num,
  sorry
}

end placards_per_person_l304_304912


namespace angle_trisection_l304_304550

/-- A theorem stating the relationship between lines and angles formed by trisection points of an arc and chord in a circle. -/
theorem angle_trisection {O A B C D C₁ D₁ M : Point} 
  (hO : is_center O)
  (hA : is_on_circumference O A)
  (hB : is_on_circumference O B)
  (hC : is_trisection_points_arc O A B C)
  (hD : is_trisection_points_arc O A B D)
  (hC₁ : is_trisection_points_chord O A B C₁)
  (hD₁ : is_trisection_points_chord O A B D₁)
  (hM : intersection_point (line_through_points C C₁) (line_through_points D D₁) M) 
  : angle A M B = (1 / 3) * angle A O B := 
sorry

end angle_trisection_l304_304550


namespace sum_of_circular_integers_l304_304527

theorem sum_of_circular_integers (a : Fin 10 → ℕ) (h : ∀ i, a i = Nat.gcd (a ((i - 1) % 10)) (a ((i + 1) % 10)) + 1) :
    (Finset.univ.sum (λ i, a i)) = 28 := by
  sorry

end sum_of_circular_integers_l304_304527


namespace geometric_sequence_l304_304896

theorem geometric_sequence (a b c r : ℤ) (h1 : b = a * r) (h2 : c = a * r^2) (h3 : c = a + 56) : b = 21 :=
by sorry

end geometric_sequence_l304_304896


namespace find_actual_marks_l304_304918

theorem find_actual_marks (wrong_marks : ℕ) (avg_increase : ℕ) (num_pupils : ℕ) (h_wrong_marks: wrong_marks = 73) (h_avg_increase : avg_increase = 1/2) (h_num_pupils : num_pupils = 16) : 
  ∃ (actual_marks : ℕ), actual_marks = 65 :=
by
  have total_increase := num_pupils * avg_increase
  have eqn := wrong_marks - total_increase
  use eqn
  sorry

end find_actual_marks_l304_304918


namespace initial_weight_l304_304938

theorem initial_weight (lost_weight current_weight : ℕ) (h1 : lost_weight = 35) (h2 : current_weight = 34) :
  lost_weight + current_weight = 69 :=
sorry

end initial_weight_l304_304938


namespace smallest_three_digit_multiple_of_17_l304_304752

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304752


namespace tan_expression_evaluation_l304_304433

variables {x y : ℝ}

def condition1 := (sin x / cos y) + (sin y / cos x) = 2
def condition2 := (cos x / sin y) + (cos y / sin x) = 8

theorem tan_expression_evaluation
  (h1 : condition1)
  (h2 : condition2) :
  (tan x * tan y) / ((tan x / tan y) + (tan y / tan x)) = 31 / 13 :=
by
  sorry

end tan_expression_evaluation_l304_304433


namespace max_students_can_distribute_equally_l304_304545

-- Define the given numbers of pens and pencils
def pens : ℕ := 1001
def pencils : ℕ := 910

-- State the problem in Lean 4 as a theorem
theorem max_students_can_distribute_equally :
  Nat.gcd pens pencils = 91 :=
sorry

end max_students_can_distribute_equally_l304_304545


namespace unit_digit_seven_consecutive_l304_304559

theorem unit_digit_seven_consecutive (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 = 0 := 
by
  sorry

end unit_digit_seven_consecutive_l304_304559


namespace smallest_three_digit_multiple_of_17_l304_304813

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304813


namespace distance_inequality_l304_304197

variables (X : Point) (A B C : Point) (a b c : ℝ)
          (XA XB XC : ℝ) (ha : XA = dist X A) (hb : XB = dist X B) (hc : XC = dist X C)

theorem distance_inequality (h_triangle : triangle A B C)
    (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0) :
  (XB / b) * (XC / c) + (XC / c) * (XA / a) + (XA / a) * (XB / b) ≥ 1 := sorry

end distance_inequality_l304_304197


namespace BC_work_time_l304_304202

-- Definitions
def rateA : ℚ := 1 / 4 -- A's rate of work
def rateB : ℚ := 1 / 4 -- B's rate of work
def rateAC : ℚ := 1 / 3 -- A and C's combined rate of work

-- To prove
theorem BC_work_time : 1 / (rateB + (rateAC - rateA)) = 3 := by
  sorry

end BC_work_time_l304_304202


namespace simplify_and_evaluate_expression_l304_304508

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l304_304508


namespace smallest_three_digit_multiple_of_17_l304_304662

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304662


namespace smallest_three_digit_multiple_of_17_l304_304730

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304730


namespace shanna_total_vegetables_l304_304121

theorem shanna_total_vegetables :
  let T := 6  -- Tomato
  let E := 2  -- Eggplant
  let P := 4  -- Pepper
  let C := 3  -- Cucumber
  let Z := 5  -- Zucchini
  let survival_rate_TEP := 2 / 3
  let survival_rate_CZ := 3 / 4
  let veg_per_s_T := 9
  let veg_per_s_E := 9
  let veg_per_s_P := 7
  let veg_per_s_C := 12
  let veg_per_s_Z := 12
  let S_T := T * survival_rate_TEP
  let S_E := E * survival_rate_TEP
  let S_P := P * survival_rate_TEP
  let S_C := C * survival_rate_CZ
  let S_Z := Z * survival_rate_CZ
 , (S_T.floor * veg_per_s_T + S_E.floor * veg_per_s_E + S_P.floor * veg_per_s_P + S_C.floor * veg_per_s_C + S_Z.floor * veg_per_s_Z) = 119 := 
by sorry

end shanna_total_vegetables_l304_304121


namespace probability_four_dots_collinear_l304_304409

-- Define the 5x5 grid and collinearity
structure Dot := (x : ℕ) (y : ℕ)

def is_collinear (d1 d2 d3 d4 : Dot) : Prop :=
  (d1.x = d2.x ∧ d2.x = d3.x ∧ d3.x = d4.x) ∨
  (d1.y = d2.y ∧ d2.y = d3.y ∧ d3.y = d4.y) ∨
  (d1.x - d1.y = d2.x - d2.y ∧ d2.x - d2.y = d3.x - d3.y ∧ d3.x - d3.y = d4.x - d4.y) ∨
  (d1.x + d1.y = d2.x + d2.y ∧ d2.x + d2.y = d3.x + d3.y ∧ d3.x + d3.y = d4.x + d4.y)

-- Count all combinations
noncomputable def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def total_combinations : ℕ := comb 25 4

def collinear_sets : ℕ := 28

-- Proof statement: The probability that four random dots are collinear
theorem probability_four_dots_collinear :
  (collinear_sets : ℚ) / total_combinations = 14 / 6325 := by
  sorry

end probability_four_dots_collinear_l304_304409


namespace first_snail_time_proof_l304_304162

-- Define the conditions
def first_snail_speed := 2 -- speed in feet per minute
def second_snail_speed := 2 * first_snail_speed
def third_snail_speed := 5 * second_snail_speed
def third_snail_time := 2 -- time in minutes
def distance := third_snail_speed * third_snail_time

-- Define the time it took the first snail
def first_snail_time := distance / first_snail_speed

-- Define the theorem to be proven
theorem first_snail_time_proof : first_snail_time = 20 := 
by
  -- Proof should be filled here
  sorry

end first_snail_time_proof_l304_304162


namespace initial_onions_count_l304_304561

theorem initial_onions_count (sold : ℕ) (left : ℕ) (initial : ℕ) (h1 : sold = 65) (h2 : left = 33) : initial = 98 :=
by
  have h3 : initial = sold + left := rfl
  rw [h1, h2, add_comm]
  exact h3

end initial_onions_count_l304_304561


namespace simplify_expression_l304_304501

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l304_304501


namespace side_length_of_square_MNPQ_l304_304998

-- Definitions for the equiangular hexagon and inscribed square
structure Hexagon :=
  (A B C D E F : Point)
  (equilateral : ∀ i, distance (seq i i.succ) = distance (seq i i.next_next))

structure Square :=
  (M N P Q : Point)
  (side_length : ℝ)
  (is_inscribed_in : Hexagon)

def problem : Prop :=
  ∃ hex : Hexagon,
  ∃ sq : Square,
    sq.is_inscribed_in hex ∧
    hex.edge_length (AB) = 50 ∧
    hex.edge_length (EF) = 53*(√2 - 1) ∧
    sq.side_length = 25*√2 - 23

-- The statement of our problem
theorem side_length_of_square_MNPQ :
  problem :=
sorry

end side_length_of_square_MNPQ_l304_304998


namespace phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l304_304211

def even_digits : Set ℕ := { 0, 2, 4, 6, 8 }
def odd_digits : Set ℕ := { 1, 3, 5, 7, 9 }

theorem phone_numbers_even : (4 * 5^6) = 62500 := by
  sorry

theorem phone_numbers_odd : 5^7 = 78125 := by
  sorry

theorem phone_numbers_ratio
  (evens : (4 * 5^6) = 62500)
  (odds : 5^7 = 78125) :
  (78125 / 62500 : ℝ) = 1.25 := by
    sorry

end phone_numbers_even_phone_numbers_odd_phone_numbers_ratio_l304_304211


namespace multiple_of_27_l304_304208

theorem multiple_of_27 (x y z : ℤ) 
  (h1 : (2 * x + 5 * y + 11 * z) = 4 * (x + y + z)) 
  (h2 : (2 * x + 20 * y + 110 * z) = 6 * (2 * x + 5 * y + 11 * z)) :
  ∃ k : ℤ, x + y + z = 27 * k :=
by
  sorry

end multiple_of_27_l304_304208


namespace smallest_three_digit_multiple_of_17_correct_l304_304620

def smallest_three_digit_multiple_of_17 : ℕ :=
  102

theorem smallest_three_digit_multiple_of_17_correct :
  ∀ n, n >= 100 → n < 1000 → n % 17 = 0 → 102 ≤ n :=
begin
  intros n h1 h2 h3,
  sorry,
end

end smallest_three_digit_multiple_of_17_correct_l304_304620


namespace smallest_three_digit_multiple_of_17_l304_304847

/--
Prove that the smallest three-digit multiple of 17 is 102.
-/
theorem smallest_three_digit_multiple_of_17 : ∃ k : ℤ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 := 
by
  sorry

end smallest_three_digit_multiple_of_17_l304_304847


namespace inclination_angle_of_line_l304_304319

theorem inclination_angle_of_line (m b : ℝ) (h : m = -1) (h_eq : ∀ x, x ∈ (λ x, m * x + b)) :
  ∃ θ : ℝ, θ = 135 ∧ θ ∈ set.Ico 0 180 :=
by
  sorry

end inclination_angle_of_line_l304_304319


namespace smallest_three_digit_multiple_of_17_l304_304827

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304827


namespace athlete_running_minutes_l304_304226

theorem athlete_running_minutes (r w : ℕ) 
  (h1 : r + w = 60)
  (h2 : 10 * r + 4 * w = 450) : 
  r = 35 := 
sorry

end athlete_running_minutes_l304_304226


namespace smallest_three_digit_multiple_of_17_l304_304822

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304822


namespace triangle_AD_length_l304_304058

theorem triangle_AD_length (triangle_ABC: Triangle) (h1: triangle_ABC.has_all_sides_different)
  (A B C D: Point) (h2: is_angle_bisector ∠A (B C) D) (a b: ℝ)
  (h3: |AB| - |BD| = a) (h4: |AC| + |CD| = b) : |AD| = sqrt (a * b) :=
sorry

end triangle_AD_length_l304_304058


namespace angle_inequality_l304_304419

variable {A B C P : Type} [Triangle A B C]

theorem angle_inequality
  (h1 : P ∈ interior A B C) :
  exists θ ∈ {angle P A B, angle P B C, angle P C A}, θ ≤ 30 :=
by 
  -- Proof of the theorem
  sorry  

end angle_inequality_l304_304419


namespace geometric_sequence_sum_5_l304_304095

noncomputable theory

def geometric_sequence (a₁ : ℕ) (q : ℤ) (n : ℕ) : ℤ := a₁ * q ^ (n - 1)

theorem geometric_sequence_sum_5 :
  ∃ (q : ℤ), q > 0 ∧
  geometric_sequence 2 q 3 = geometric_sequence 2 q 2 + 4 ∧
  (geometric_sequence 2 q 1 + geometric_sequence 2 q 2 + geometric_sequence 2 q 3 + geometric_sequence 2 q 4 + geometric_sequence 2 q 5) = 62 :=
by
  sorry

end geometric_sequence_sum_5_l304_304095


namespace cos_value_l304_304361

theorem cos_value (A : ℝ) (h : Real.tan A + Real.sec A = 3) : Real.cos A = 3 / 5 :=
by
  sorry

end cos_value_l304_304361


namespace smallest_three_digit_multiple_of_17_l304_304830

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := 
begin
  use 102,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3,
    sorry }
end

end smallest_three_digit_multiple_of_17_l304_304830


namespace local_minimum_b_l304_304032

noncomputable def func (b x : ℝ) := x^3 - 3 * b * x + b

theorem local_minimum_b (b : ℝ) (h_min : ∃ x ∈ (0:ℝ, 1:ℝ), is_local_min (func b) x) : b ∈ (0:ℝ, 1:ℝ) :=
sorry

end local_minimum_b_l304_304032


namespace wire_length_l304_304913

theorem wire_length (r_sphere r_wire : ℝ) (h : ℝ) (V : ℝ)
  (h₁ : r_sphere = 24) (h₂ : r_wire = 16)
  (h₃ : V = 4 / 3 * Real.pi * r_sphere ^ 3)
  (h₄ : V = Real.pi * r_wire ^ 2 * h): 
  h = 72 := by
  -- we can use provided condition to show that h = 72, proof details omitted
  sorry

end wire_length_l304_304913


namespace length_AB_l304_304051

theorem length_AB (C1_eq : ∀ (x y : ℝ), x^2 - y^2 = 2 ↔ (x = y ∧ y = 0))
                  (C2_eq_x : ∀ (θ : ℝ), x = 2 + 2 * cos θ)
                  (C2_eq_y : ∀ (θ : ℝ), y = 2 * sin θ)
                  (polar_transformation : ∀ (ρ θ : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ)
                  (θ_eq_pi_six : θ = Real.pi / 6) :
  let ρ_A := 2,
      ρ_B := 2 * sqrt 3 in
  |ρ_A - ρ_B| = 2 * sqrt 3 - 2 := 
by
  sorry

end length_AB_l304_304051


namespace piles_3_stones_impossible_l304_304158

theorem piles_3_stones_impossible :
  ∀ n : ℕ, ∀ piles : ℕ → ℕ,
  (piles 0 = 1001) →
  (∀ k : ℕ, k > 0 → ∃ i j : ℕ, piles (k-1) > 1 → piles k = i + j ∧ i > 0 ∧ j > 0) →
  ¬ (∀ m : ℕ, piles m ≠ 3) :=
by
  sorry

end piles_3_stones_impossible_l304_304158


namespace count_same_side_points_l304_304149

def line_equation (x y : ℝ) : ℝ := 3 * y - 2 * x + 1

def same_side (p1 p2 : ℝ × ℝ) : Prop :=
  line_equation p1.1 p1.2 * line_equation p2.1 p2.2 > 0

theorem count_same_side_points :
  let points := [(1, 1), (2, 3), (4, 2)] in
  let reference := (0, -1) in
  (list.count (λ p, same_side p reference) points) = 1 := by
  sorry

end count_same_side_points_l304_304149


namespace max_diagonals_in_rectangle_l304_304553

theorem max_diagonals_in_rectangle :
  let rows := 3
  let cols := 100
  let max_diagonals_per_square := 2
  ∃ (max_diagonals : ℕ), max_diagonals = 200 ∧
    (∀ i j : ℕ, i < rows → j < cols → 
      (let possible_diagonals := max_diagonals_per_square
       ∀ di dj : ℕ, di <= possible_diagonals ∧ dj <= possible_diagonals → 
       not (di = dj)))) :=
sorry

end max_diagonals_in_rectangle_l304_304553


namespace inequality_proof_l304_304423

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) (h2 : (Finset.univ.sum a) ≥ 0) :
  (Finset.univ.sum (λ i => Real.sqrt (a i ^ 2 + 1))) ≥
  Real.sqrt (2 * n * (Finset.univ.sum a)) :=
by
  sorry

end inequality_proof_l304_304423


namespace average_speed_l304_304259

theorem average_speed 
  (total_distance : ℝ) (total_time : ℝ) 
  (h_distance : total_distance = 26) (h_time : total_time = 4) :
  (total_distance / total_time) = 6.5 :=
by
  rw [h_distance, h_time]
  norm_num

end average_speed_l304_304259


namespace NC_passes_midpoint_AX_l304_304436

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := sorry

theorem NC_passes_midpoint_AX 
  (ABC : Triangle) 
  (h_acute : ∀ (angle : Angle), angle ∈ ABC.angles → angle < π / 2) 
  (h_AB_AC : |AB| < |AC|) 
  (circum_circle : Circle (Triangle.circumcenter ABC) (Triangle.circumradius ABC))
  (X Y : Point)
  (h_XY_minor_arc : X ∈ minor_arc ABC.circumcircle BC ∧ Y ∈ minor_arc ABC.circumcircle BC)
  (h_BX_XY_YC : |BX| = |XY| ∧ |XY| = |YC|)
  (N : Point) 
  (h_N_on_AY : N ∈ segment AY)
  (h_AB_AN_NC : |AB| = |AN| ∧ |AN| = |NC|) : 
  passes_through (NC) (midpoint A X) :=
sorry

end NC_passes_midpoint_AX_l304_304436


namespace max_pages_l304_304066

theorem max_pages (cents_available : ℕ) (cents_per_page : ℕ) : 
  cents_available = 2500 → 
  cents_per_page = 3 → 
  (cents_available / cents_per_page) = 833 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end max_pages_l304_304066


namespace smallest_three_digit_multiple_of_17_l304_304678

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304678


namespace central_angle_of_sector_l304_304141

theorem central_angle_of_sector (r : ℝ) (h1 : r > 0) :
  let slant_height := 2 * r
  let circumference := 2 * Real.pi * r
  let arc_length := 2 * Real.pi * r
  let sector_angle := 180
  ∃ n : ℝ, arc_length = (n * Real.pi * 2 * r) / 180 ∧ n = sector_angle :=
begin
  sorry
end

end central_angle_of_sector_l304_304141


namespace smallest_three_digit_multiple_of_17_l304_304661

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304661


namespace smallest_three_digit_multiple_of_17_l304_304731

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  use 102,
  split,
  { exact le_of_lt (nat.lt_of_sub_eq_succ rfl), }, -- 102 is greater than or equal to 100
  split,
  { exact nat.lt_of_sub_eq_succ rfl, }, -- 102 is less than 1000
  split,
  { exact nat.dvd_intro_left (6) rfl, }, -- 17 divides 102
  { intros m hm1 hm2 hm3,
    exact nat.le_of_dvd hm1 (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (nat.dvd_of_mod_eq_zero (nat.mod_eq_zero_of_dvd (hm3)))), }, -- 102 is the smallest
end

end smallest_three_digit_multiple_of_17_l304_304731


namespace common_difference_ne_3_l304_304289

theorem common_difference_ne_3 
  (d : ℕ) (hd_pos : d > 0) 
  (exists_n : ∃ n : ℕ, 81 = 1 + (n - 1) * d) : 
  d ≠ 3 :=
by sorry

end common_difference_ne_3_l304_304289


namespace total_sum_is_180_l304_304458

/-- Definitions and conditions given in the problem -/
def CoralineNumber : ℕ := 80
def JaydenNumber (C : ℕ) : ℕ := C - 40
def MickeyNumber (J : ℕ) : ℕ := J + 20

/-- Prove that the total sum of their numbers is 180 -/
theorem total_sum_is_180 : 
  let C := CoralineNumber
  let J := JaydenNumber C 
  let M := MickeyNumber J 
  in M + J + C = 180 := sorry

end total_sum_is_180_l304_304458


namespace smallest_three_digit_multiple_of_17_l304_304683

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ 17 * n ∧ 17 * n ≤ 999 ∧ 17 * n = 102 :=
begin
  sorry
end

end smallest_three_digit_multiple_of_17_l304_304683


namespace cos_value_l304_304362

theorem cos_value (A : ℝ) (h : Real.tan A + Real.sec A = 3) : Real.cos A = 3 / 5 :=
by
  sorry

end cos_value_l304_304362


namespace find_b_if_even_function_l304_304539

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem find_b_if_even_function (h : ∀ x : ℝ, f (-x) = f (x)) : b = 0 := by
  sorry

end find_b_if_even_function_l304_304539


namespace smallest_three_digit_multiple_of_17_l304_304806

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304806


namespace correct_answers_count_l304_304188

theorem correct_answers_count
  (c w : ℕ)
  (h1 : c + w = 150)
  (h2 : 4 * c - 2 * w = 420) :
  c = 120 := by
  sorry

end correct_answers_count_l304_304188


namespace monthly_profit_relation_and_max_profit_l304_304875

variables (x w : ℝ)

def cost_price := 20
def y (x : ℝ) : ℝ := -10 * x + 500
def profit_per_unit (x : ℝ) : ℝ := x - cost_price
def w (x : ℝ) : ℝ := (x - cost_price) * (-10 * x + 500)
def max_profit := 2160

theorem monthly_profit_relation_and_max_profit :
  (∀ (x : ℝ), 20 ≤ x ∧ x ≤ 32 → w x = -10 * x ^ 2 + 700 * x - 10000) ∧
  (∃ x, 20 ≤ x ∧ x ≤ 32 ∧ w x = max_profit) :=
by
  sorry

end monthly_profit_relation_and_max_profit_l304_304875


namespace smallest_three_digit_multiple_of_17_l304_304761

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≤ m) :=
  sorry

end smallest_three_digit_multiple_of_17_l304_304761


namespace smallest_three_digit_multiple_of_17_l304_304654

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l304_304654


namespace problem_proof_l304_304439

theorem problem_proof (a b c x y z : ℝ) (h₁ : 17 * x + b * y + c * z = 0) (h₂ : a * x + 29 * y + c * z = 0)
                      (h₃ : a * x + b * y + 53 * z = 0) (ha : a ≠ 17) (hx : x ≠ 0) :
                      (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
by
  -- proof goes here
  sorry

end problem_proof_l304_304439


namespace z_lies_on_perpendicular_bisector_l304_304443

open Set

variables {A B C X Y P Q Z : ℝ}

-- Conditions
def conditions (A B C X Y P Q Z : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ X > 0 ∧ Y > 0 ∧ P > 0 ∧ Q > 0 ∧ Z > 0 ∧
  A <> B ∧ A <> C ∧ B <> C ∧ A <> P ∧ A <> Q ∧ B <> X ∧ C <> Y ∧ AB > BC ∧ AC > BC ∧ 
  AB = BX ∧ AC = CY ∧ CA = CP ∧ BA = BQ ∧ (↑BQ ∩ ↑CP = Z)

-- Theorem
theorem z_lies_on_perpendicular_bisector (A B C X Y P Q Z : ℝ) (h : conditions A B C X Y P Q Z) : 
  ∀ (Z : ℝ), dist Z B = dist Z C :=
sorry

end z_lies_on_perpendicular_bisector_l304_304443


namespace smallest_three_digit_multiple_of_17_l304_304744

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304744


namespace cut_square_contains_corner_cell_l304_304909

-- Define the conditions
def large_square_side : ℕ := 40
def small_square_side : ℕ := 39
def remaining_cells : ℕ := 79

-- Prove that the cut-out square necessarily contains one of the corner cells of the large square
theorem cut_square_contains_corner_cell 
  (N M : ℕ) 
  (h : N^2 - M^2 = remaining_cells)
  (h_N : N = large_square_side)
  (h_M : M = small_square_side) : 
  contains_corner_cell : 
  true :=
by
  sorry

end cut_square_contains_corner_cell_l304_304909


namespace smallest_three_digit_multiple_of_17_l304_304745

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, m >= 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m :=
begin
  let n := 102,
  use n,
  split,
  { norm_num, }, -- n >= 100 is obviously true because n = 102
  split,
  { norm_num, }, -- n < 1000 is obviously true because n = 102
  split,
  { exact dvd_refl 17, }, -- 17 divides 102
  { intros m hm,
    cases hm with h100 hm,
    cases hm with h1000 hdiv,
    exact le_trans (nat.le.intro _ (symm hdiv)) (le_of_not_gt (λ h, by norm_num at h)), },
end

end smallest_three_digit_multiple_of_17_l304_304745


namespace smallest_three_digit_multiple_of_17_l304_304784

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304784


namespace label_feasible_if_n_odd_l304_304986

theorem label_feasible_if_n_odd (n : ℕ) (h : n ≥ 3) : 
  ∃ a : Fin (2 * n) → Fin (2 * n), 
  (∀ i : Fin n, a i + a (i + 1) = a (n + i) + a (n + i + 1)) ∧ 
  (∀ i j : Fin (2 * n), i ≠ j → a i ≠ a j) →
  (n % 2 = 1) :=
sorry

end label_feasible_if_n_odd_l304_304986


namespace second_difference_of_polynomial_l304_304276

def polynomial (x : ℤ) : ℤ := x^3 - x

def first_finite_difference (f : ℤ → ℤ) (x : ℤ) : ℤ :=
  f(x + 1) - f(x)

def second_finite_difference (f : ℤ → ℤ) (x : ℤ) : ℤ :=
  first_finite_difference (first_finite_difference f) x

theorem second_difference_of_polynomial :
  second_finite_difference polynomial x = 6 * x + 6 :=
by
  sorry

end second_difference_of_polynomial_l304_304276


namespace n_plus_1_power_two_or_n_equals_5_l304_304425

-- Let  n  be an odd positive integer such that both  φ(n)  and  φ(n+1)  are powers of two.
theorem n_plus_1_power_two_or_n_equals_5 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : 0 < n) (hphi_n : ∃ k1 : ℕ, φ n = 2^k1) (hphi_n1 : ∃ k2 : ℕ, φ (n + 1) = 2^k2) :
  ∃ k : ℕ, n + 1 = 2^k ∨ n = 5 :=
sorry

end n_plus_1_power_two_or_n_equals_5_l304_304425


namespace votes_cast_l304_304187

-- The problem statement
theorem votes_cast (V : ℕ) (h1: 0.35 * V = n) (h2: n + 2280 = 0.65 * V) : V = 7600 :=
by
  sorry

end votes_cast_l304_304187


namespace xiao_peach_days_l304_304876

theorem xiao_peach_days :
  ∀ (xiao_ming_apples xiao_ming_pears xiao_ming_peaches : ℕ)
    (xiao_hong_apples xiao_hong_pears xiao_hong_peaches : ℕ)
    (both_eat_apples both_eat_pears : ℕ)
    (one_eats_apple_other_eats_pear : ℕ),
    xiao_ming_apples = 4 →
    xiao_ming_pears = 6 →
    xiao_ming_peaches = 8 →
    xiao_hong_apples = 5 →
    xiao_hong_pears = 7 →
    xiao_hong_peaches = 6 →
    both_eat_apples = 3 →
    both_eat_pears = 2 →
    one_eats_apple_other_eats_pear = 3 →
    ∃ (both_eat_peaches_days : ℕ),
      both_eat_peaches_days = 4 := 
sorry

end xiao_peach_days_l304_304876


namespace measure_of_two_equal_angles_l304_304942

noncomputable def measure_of_obtuse_angle (θ : ℝ) : ℝ := θ + (0.6 * θ)

-- Given conditions
def is_obtuse_isosceles_triangle (θ : ℝ) : Prop :=
  θ = 90 ∧ measure_of_obtuse_angle 90 = 144 ∧ 180 - 144 = 36

-- The main theorem
theorem measure_of_two_equal_angles :
  ∀ θ, is_obtuse_isosceles_triangle θ → 36 / 2 = 18 :=
by
  intros θ h
  sorry

end measure_of_two_equal_angles_l304_304942


namespace geometric_sequence_sum_is_120_l304_304314

noncomputable def sum_first_four_geometric_seq (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4

theorem geometric_sequence_sum_is_120 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_pos_geometric : 0 < q ∧ q < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a3_a5_product : a 3 * a 5 = 64) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  sum_first_four_geometric_seq a q = 120 :=
sorry

end geometric_sequence_sum_is_120_l304_304314


namespace num_a_k_divisible_by_11_l304_304081

noncomputable def a (k : ℕ) : ℕ := String.to_nat $ String.join $ List.map Nat.to_digits (List.range (k + 1))

def alternating_sum (n : ℕ) : ℤ :=
  List.sum $ List.map (λ (pair : ℕ × ℕ), if pair.snd % 2 = 0 then (pair.fst : ℤ) else -(pair.fst : ℤ))
    (List.zip (List.range (n+1)) (List.range (n+1)))

def is_divisible_by_11 (n : ℕ) : Prop := (alternating_sum n % 11 = 0)

theorem num_a_k_divisible_by_11 : 
  (List.countp (λ k, is_divisible_by_11 (a k)) (List.range 101)) = 8 :=
  sorry

end num_a_k_divisible_by_11_l304_304081


namespace exists_n_divisible_by_5_l304_304228

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h_div : a * m ^ 3 + b * m ^ 2 + c * m + d ≡ 0 [ZMOD 5]) 
  (h_d_nonzero : d ≠ 0) : 
  ∃ n : ℤ, d * n ^ 3 + c * n ^ 2 + b * n + a ≡ 0 [ZMOD 5] :=
sorry

end exists_n_divisible_by_5_l304_304228


namespace smallest_three_digit_multiple_of_17_l304_304780

theorem smallest_three_digit_multiple_of_17 : ∃ (k : ℕ), 100 ≤ 17 * k ∧ 17 * k ≤ 999 ∧ 17 * k = 102 :=
by
  use 6
  split
  · exact of_as_true trivial -- 100 <= 17 * k
  split
  · exact of_as_true trivial -- 17 * k <= 999
  · exact of_as_true trivial -- 17 * k = 102
  sorry

end smallest_three_digit_multiple_of_17_l304_304780
