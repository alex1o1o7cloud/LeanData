import Mathlib

namespace perpendicular_planes_l203_203155
-- Importing the necessary library

-- Defining lines and planes
variables {m n : Type} {α β : Type}
-- Assumptions related to the geometric relations
variables [geometric_relation m α] [geometric_relation m β] 

-- Main theorem statement
theorem perpendicular_planes (m n : Type) (α β : Type) 
  [geometric_relation m α] [geometric_relation m β]
  (h1 : parallel m β) (h2 : perpendicular m α) : 
  perpendicular α β :=
sorry

end perpendicular_planes_l203_203155


namespace two_digit_primes_with_ones_digit_3_l203_203868

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203868


namespace prime_factors_101_103_105_107_l203_203759

theorem prime_factors_101_103_105_107 :
  ∃ (primes : Finset ℕ), primes.card = 6 ∧
    primes = {101, 103, 3, 5, 7, 107} ∧ 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    (∀ a b ∈ primes, a ≠ b → Nat.Coprime a b) :=
by
  sorry

end prime_factors_101_103_105_107_l203_203759


namespace tan_105_l203_203554

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203554


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203987

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203987


namespace part1_part2_l203_203158

-- Define propositions p and q
def p (t : ℝ) : Prop := -2 < t ∧ t < 0
def q (t a : ℝ) : Prop := t^2 - (a + 2) * t + 2 * a < 0

-- Given the conditions, prove the propositions
theorem part1 (t : ℝ) (ht : p t) : -2 < t ∧ t < 0 :=
by {
  apply ht,
}

theorem part2 (t a : ℝ) (suff : ∀ t, p t → q t a) : a ≤ -2 :=
by {
  sorry
}

end part1_part2_l203_203158


namespace find_a_b_c_interval_and_extremes_l203_203026

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 12 * x

-- Conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = - (f x)
def tangent_perpendicular_to_line (f : ℝ → ℝ) (x : ℝ) (line_slope : ℝ) := has_deriv_at f x ⟶ is_tangent_perpendicular
def minimum_value_of_derivative (f' : ℝ → ℝ) (m : ℝ) := ∀ x, f' x ≥ m

-- Correct answers and corresponding questions
-- (1) Finding the values of a, b, c
theorem find_a_b_c (f : ℝ → ℝ) : 
  is_odd f →
  tangent_perpendicular_to_line f 1 (-1/6) →
  minimum_value_of_derivative (deriv f) (-12) →
  ∃ a b c, f = λ x, a * x^3 + b * x + c ∧ a = 2 ∧ b = -12 ∧ c = 0 :=
sorry

-- (2) Intervals of monotonic increase and extents on [-1, 3]
theorem interval_and_extremes (f : ℝ → ℝ)
  (hf: f = λ x, 2 * x^3 - 12 * x) :
  ∃ I, (∀ x ∈ I, deriv f x ≥ 0) ∧
  (∃ (x_max x_min : ℝ), x_max ∈ I ∧ x_min ∈ I ∧ f (x_max) = 18 ∧ f (x_min) = -8) :=
sorry

end find_a_b_c_interval_and_extremes_l203_203026


namespace tan_105_degree_l203_203568

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203568


namespace num_two_digit_primes_with_ones_digit_3_l203_203947

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203947


namespace proof_problem_l203_203229

noncomputable def pointP := (1 / 2 : ℝ, 1 : ℝ)

noncomputable def lineParametric (t : ℝ) : ℝ × ℝ := 
  (1 / 2 + (sqrt 3 / 2) * t, 1 + (1 / 2) * t)

noncomputable def curvePolar (theta : ℝ) : ℝ :=
  sqrt 2 * cos (theta - π / 4)

theorem proof_problem :
  (∀ t : ℝ, ∃ x y : ℝ, lineParametric t = (x, y) → x - sqrt 3 * y - 1 / 2 + sqrt 3 = 0) ∧
  (∀ θ : ℝ, ∃ (ρ : ℝ), curvePolar θ = ρ → (ρ * cos θ, ρ * sin θ) ∈ {(x, y) | (x - 1 / 2)^2 + (y - 1 / 2)^2 = 1 / 2}) ∧
  let intersections : set (ℝ × ℝ) := {p | ∃ t : ℝ, p = lineParametric t ∧ p ∈ {(x, y) | (x - 1 / 2)^2 + (y - 1 / 2)^2 = 1 / 2}} in
  let PA_distance (A : ℝ × ℝ) := dist pointP A in
  ∀ A B ∈ intersections, A ≠ B → PA_distance A * PA_distance B = 1 / 4 :=
sorry

end proof_problem_l203_203229


namespace log_expression_integer_part_l203_203246

theorem log_expression_integer_part {a : ℝ} (h_nonneg : 1 ≤ a) (h_lt : a < 10) : 
  let expr := (2007: ℝ)^(2006: ℝ) * (2006: ℝ)^(2007: ℝ) in 
  let log_expr := Real.log10 expr in
  ∃ (k : ℤ), log_expr = a * 10^k ∧ k = 4 :=
by 
  sorry

end log_expression_integer_part_l203_203246


namespace exists_convex_polyhedron_with_1990_edges_no_triangular_faces_l203_203245

theorem exists_convex_polyhedron_with_1990_edges_no_triangular_faces : 
  ∃ (P : Polyhedron), P.convex ∧ P.edges = 1990 ∧ ¬ ∃ (F : Face), F.triangular :=
sorry

end exists_convex_polyhedron_with_1990_edges_no_triangular_faces_l203_203245


namespace largest_value_among_expressions_l203_203012

def expA : ℕ := 3 + 1 + 2 + 4
def expB : ℕ := 3 * 1 + 2 + 4
def expC : ℕ := 3 + 1 * 2 + 4
def expD : ℕ := 3 + 1 + 2 * 4
def expE : ℕ := 3 * 1 * 2 * 4

theorem largest_value_among_expressions :
  expE > expA ∧ expE > expB ∧ expE > expC ∧ expE > expD :=
by
  -- Proof will go here
  sorry

end largest_value_among_expressions_l203_203012


namespace equation1_solution_equation2_no_solution_l203_203313

theorem equation1_solution :
  ∀ x : ℝ, (2 / x = 3 / (x + 2)) ↔ (x = 4) := sorry

theorem equation2_no_solution :
  ¬∃ x : ℝ, ∀ (x ≠ 2), (1 / (x - 2) = (1 - x) / (2 - x) - 3) := sorry

end equation1_solution_equation2_no_solution_l203_203313


namespace volume_PQRS_l203_203237

noncomputable def volume_of_tetrahedron (P Q R S : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  1 / 6 * abs (det ![
    P.1 - Q.1, P.2 - Q.2, P.3 - Q.3, 1,
    Q.1 - R.1, Q.2 - R.2, Q.3 - R.3, 1,
    R.1 - S.1, R.2 - S.2, R.3 - S.3, 1,
    S.1 - P.1, S.2 - P.2, S.3 - P.3, 1
  ])

theorem volume_PQRS :
  ∀ {K L M N P Q R S : EuclideanSpace ℝ (Fin 3)},
    dist K L = 9 → dist M N = 9 → dist K M = 15 → dist L N = 15 →
    dist K N = 16 → dist L M = 16 →
    P = triangle_incenter K L M → 
    Q = triangle_incenter K L N → 
    R = triangle_incenter K M N → 
    S = triangle_incenter L M N →
    volume_of_tetrahedron P Q R S = 4.85 :=
by
  intros
  -- Proof skipped
  sorry

end volume_PQRS_l203_203237


namespace shirts_needed_for_vacation_l203_203199

def vacation_days := 7
def same_shirt_days := 2
def different_shirts_per_day := 2
def different_shirt_days := vacation_days - same_shirt_days

theorem shirts_needed_for_vacation : different_shirt_days * different_shirts_per_day + same_shirt_days = 11 := by
  sorry

end shirts_needed_for_vacation_l203_203199


namespace shopping_mall_pricing_l203_203033

noncomputable def profit (x : ℝ) : ℝ := (x - 20) * (800 - 20 * (x - 30))

theorem shopping_mall_pricing : 
  ∃ (x : ℝ) (items : ℝ), 
    profit x = 12000 
    ∧ x = 40 
    ∧ items = 1400 - 20 * 40 := 
by {
  let x := 40,
  let items := 1400 - 20 * x,
  use [x, items],
  split,
  {
    show profit x = 12000,
    calc 
    profit x = (x - 20) * (800 - 20 * (x - 30)) : by rfl
    ... = 12000 : by norm_num,
  },
  {
    split,
    { refl },
    { refl }
  }
}

end shopping_mall_pricing_l203_203033


namespace tan_105_l203_203461

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203461


namespace last_score_was_95_l203_203286

noncomputable def last_score (scores : List ℤ) (avg : List ℤ → ℤ) : Prop :=
  ∃ s ∈ scores, 
    scores = [75, 81, 85, 87, s] ∧ 
    ∀ i, 1 ≤ i ∧ i ≤ 5 → avg (List.take i ([75, 81, 85, 87, s])) ∈ ℤ

theorem last_score_was_95 : last_score [75, 81, 85, 87, 95] (λ l, l.sum / l.length) :=
  sorry

end last_score_was_95_l203_203286


namespace range_z_in_parallelogram_l203_203699

noncomputable def point : Type := ℝ × ℝ

def A : point := (-1, 2)
def B : point := (3, 4)
def C : point := (4, -2)
def D : point := (4, 0) -- D calculated using midpoint condition from solution

def z (x y : ℝ) : ℝ := 2 * x - 5 * y

theorem range_z_in_parallelogram :
  ∀ (P : point), (P = A ∨ P = B ∨ P = C ∨ P = D) → (z P.1 P.2 ≥ -14) ∧ (z P.1 P.2 ≤ 18) :=
begin
  sorry
end

end range_z_in_parallelogram_l203_203699


namespace tan_105_eq_neg2_sub_sqrt3_l203_203617

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203617


namespace parabola_vertex_l203_203330

def parabola_eq (a b : ℝ) : (ℝ → ℝ) := λ x, -x^2 + a * x + b

def has_vertex (f : ℝ → ℝ) (vx vy : ℝ) : Prop :=
  ∀ x, f x = f vx - (x - vx) ^ 2

def vertex_of_parabola (a : ℝ) (b : ℝ) : (ℝ × ℝ) := (a / 2, f (a / 2))

theorem parabola_vertex 
  (a b : ℝ)
  (h : ∀ x, parabola_eq a b x ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) :
  has_vertex (parabola_eq a b) 3 16 :=
sorry

end parabola_vertex_l203_203330


namespace gcd_2703_1113_l203_203336

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := 
by 
  sorry

end gcd_2703_1113_l203_203336


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203481

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203481


namespace median_and_range_l203_203352

theorem median_and_range (shots : List ℕ) (h : shots = [6, 10, 5, 3, 4, 8, 4]) :
  List.median shots = 5 ∧ List.range shots = 7 :=
by
  sorry

end median_and_range_l203_203352


namespace leila_total_expenditure_l203_203257

variable (cost_auto cost_market total : ℕ)
variable (h1 : cost_auto = 350)
variable (h2 : cost_auto = 3 * cost_market + 50)

theorem leila_total_expenditure : total = 450 :=
by
  have h3 : cost_market = 100 := by
    calc
      cost_market = (350 - 50) / 3 := by rw [← h2, ← h1]
      ... = 100 : by norm_num
  have h4 : total = cost_auto + cost_market := by norm_num
  calc
    total = 350 + 100 := by rw [h4, h1, h3]
    ... = 450 : by norm_num

end leila_total_expenditure_l203_203257


namespace count_two_digit_primes_with_ones_digit_three_l203_203786

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203786


namespace no_such_quadratic_eqs_l203_203755

-- Definitions of discriminants and quadratic equation properties
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
theorem no_such_quadratic_eqs 
    (d1 d2 d3 : ℝ) 
    (h1 : 2 * quadratic_eq 2 (-b1) c1 d1 = 0) 
    (h2 : 2 * quadratic_eq 2 (-b2) c2 d2 = 0) 
    (h3 : 2 * quadratic_eq 2 (-b3) c3 d3 = 0) 
    (dist_discr : d1 < d2 ∧ d2 < d3)
    (a_eq : ∀ x, 2 * x^2 + (-b1) * x + c1 = 2 * x^2 + (-b2) * x + c2 = 2 * x^2 + (-b3) * x + c3)
    (discr_eq_root : d1 = root (2 * (quadratic_eq 2 (-b2) c2)) d2 ∧ d2 := root (2 * (quadratic_eq 2 (-b3) c3)) d3 ∧ d3 = root (2 * (quadratic_eq 2 (-b1) c1)) d1)
    : false := 
sorry

end no_such_quadratic_eqs_l203_203755


namespace morks_tax_rate_l203_203284

-- Definitions based on conditions
def Mork_income : Type := ℝ
def Mindy_income (M : Mork_income) : Mork_income := 4 * M
def Mork_tax_rate : Type := ℝ
def Mindy_tax_rate : Mork_tax_rate := 0.25
def combined_tax_rate : Mork_tax_rate := 0.29

-- The statement of the theorem
theorem morks_tax_rate (M : Mork_income) (R : Mork_tax_rate) :
  (R * M + Mindy_tax_rate * Mindy_income M) / (M + Mindy_income M) = combined_tax_rate → R = 0.45 :=
by
  sorry

end morks_tax_rate_l203_203284


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203993

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203993


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203985

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203985


namespace two_digit_primes_with_ones_digit_three_count_l203_203768

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203768


namespace number_of_months_l203_203220

noncomputable def probability_survival := (9:ℝ) / 10
noncomputable def initial_population := 400
noncomputable def expected_survivors := 291.6

theorem number_of_months : 
  ∃ (n : ℝ), initial_population * (probability_survival ^ n) ≈ expected_survivors ∧ n ≈ 3 :=
by
  sorry

end number_of_months_l203_203220


namespace tan_105_eq_neg2_sub_sqrt3_l203_203536

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203536


namespace two_digit_primes_ending_in_3_eq_6_l203_203936

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203936


namespace two_digit_primes_with_ones_digit_three_count_l203_203772

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203772


namespace coordinates_of_Q_range_of_g_l203_203228

variable {x : ℝ}

def P := (1/2 : ℝ, Real.sqrt 3 / 2)
def Q_coord (θ : ℝ) : ℝ × ℝ := 
  (1/2 * Real.cos(θ + Real.pi / 3), 1/2 * Real.sin(θ + Real.pi / 3))

def f (θ : ℝ) : ℝ :=
  let (px, py) := P
  let (qx, qy) := Q_coord θ
  px * qx + py * qy

def g (θ : ℝ) : ℝ :=
  f θ * f (θ + Real.pi / 3)

theorem coordinates_of_Q :
  (Q_coord (Real.pi / 4) = ( (Real.sqrt 2 - Real.sqrt 6) / 4, (Real.sqrt 2 + Real.sqrt 6) / 4)) :=
sorry

theorem range_of_g :
  (∀ θ, -1/4 ≤ g θ ∧ g θ ≤ 3/4) :=
sorry

end coordinates_of_Q_range_of_g_l203_203228


namespace solution_set_of_inequality_l203_203345

theorem solution_set_of_inequality :
  { x : ℝ | (x - 3) * (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l203_203345


namespace find_multiple_l203_203433

theorem find_multiple :
  ∀ (total_questions correct_answers score : ℕ) (m : ℕ),
  total_questions = 100 →
  correct_answers = 90 →
  score = 70 →
  score = correct_answers - m * (total_questions - correct_answers) →
  m = 2 :=
by
  intros total_questions correct_answers score m h1 h2 h3 h4
  sorry

end find_multiple_l203_203433


namespace true_propositions_l203_203732

def converse_additive_inverses (x y : ℝ) : Prop :=
  (x + y = 0) → (∀ (x y : ℝ), x = -y)

def negation_congruent_triangle_areas : Prop :=
  ¬ (∀ (T₁ T₂ : Triangle), congruent T₁ T₂ → area T₁ = area T₂)

def converse_quadratic_real_roots (q : ℝ) : Prop :=
  (q ≤ 1) → (∃ x : ℝ, x^2 + 2 * x + q = 0)

def inverse_negation_equilateral_triangle : Prop :=
  (∀ (Δ : Triangle), equilateral Δ → (∀ (a b c : Angle), a = b ∧ b = c))

theorem true_propositions :
  {1, 3, 4} = 
  {i : ℕ | (i = 1 → converse_additive_inverses) ∧ 
                    (i = 2 → ¬ negation_congruent_triangle_areas) ∧
                    (i = 3 → converse_quadratic_real_roots (q : ℝ)) ∧ 
                    (i = 4 → inverse_negation_equilateral_triangle)},
{
  sorry
}

end true_propositions_l203_203732


namespace candle_height_after_burn_l203_203405

theorem candle_height_after_burn (total_time : ℕ) (initial_height : ℕ)
  (time_per_odd : ℕ → ℕ) (time_per_even : ℕ → ℕ)
  (htime_per_odd : ∀ k, k % 2 = 1 → time_per_odd k = 10 * k)
  (htime_per_even : ∀ k, k % 2 = 0 → time_per_even k = 15 * k)
  (total_time_eq : total_time = 80000)
  (initial_height_eq : initial_height = 150):
  let remaining_height := initial_height - (total_time / ((10 * (1 + 3 + ... + m)) + (15 * (2 + 4 + ... + n))))
  in 
  remaining_height = 70 :=
by
  sorry

end candle_height_after_burn_l203_203405


namespace smallest_symmetric_set_size_l203_203426

noncomputable def is_in_symmetry_set (T : Set (ℝ × ℝ)) (p : ℝ × ℝ) :=
  (p ∈ T ∧
   (-p.1, -p.2) ∈ T ∧
   (p.1, -p.2) ∈ T ∧
   (-p.1, p.2) ∈ T ∧
   (p.2, p.1) ∈ T ∧
   (-p.2, -p.1) ∈ T ∧
   (p.2, -p.1) ∈ T ∧
   (-p.2, p.1) ∈ T)

theorem smallest_symmetric_set_size (T : Set (ℝ × ℝ)) (hT : ∃ p : ℝ × ℝ, p = (3, 4) ∧ is_in_symmetry_set T p) :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 8 ∧ ∀ (p ∈ S), is_in_symmetry_set T p :=
begin
  sorry
end

end smallest_symmetric_set_size_l203_203426


namespace tan_105_l203_203500

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203500


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203477

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203477


namespace sum_of_first_three_terms_l203_203333

theorem sum_of_first_three_terms 
  (a d : ℤ) 
  (h1 : a + 4 * d = 15) 
  (h2 : d = 3) : 
  a + (a + d) + (a + 2 * d) = 18 :=
by
  sorry

end sum_of_first_three_terms_l203_203333


namespace tan_105_eq_neg2_sub_sqrt3_l203_203576

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203576


namespace count_two_digit_primes_ending_with_3_l203_203843

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203843


namespace number_of_distinct_trees_7_vertices_l203_203197

theorem number_of_distinct_trees_7_vertices : ∃ (n : ℕ), n = 7 ∧ (Tree.enumeration n).card = 11 :=
by
  sorry

end number_of_distinct_trees_7_vertices_l203_203197


namespace tan_105_degree_l203_203562

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203562


namespace remainder_base12_div_9_l203_203380

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l203_203380


namespace extreme_values_x_axis_l203_203272

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

theorem extreme_values_x_axis (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, f a b c x = x * (a * x^2 + b * x + c))
  (h3 : ∀ x, deriv (f a b c) x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : deriv (f a b c) 1 = 0)
  (h5 : deriv (f a b c) (-1) = 0) :
  b = 0 :=
sorry

end extreme_values_x_axis_l203_203272


namespace min_product_sum_b_l203_203295

theorem min_product_sum_b (a : Fin 7 → ℕ) (b : Fin 7 → ℕ)
  (h₁ : ∀ i, 2 ≤ a i ∧ a i ≤ 166)
  (h₂ : ∀ i, a i ^ b i % 167 = a ((i + 1) % 7) ^ 2 % 167) :
  ∃ (b_min_product : ℕ),
    (∀ b', (∀ i, b' i > 0) → 
            (∀ i, a i ^ b' i % 167 = a ((i + 1) % 7) ^ 2 % 167) →
            (b_min_product ≤ (∏ i, b' i) * (∑ i, b' i))) ∧ 
    b_min_product = 675 :=
sorry

end min_product_sum_b_l203_203295


namespace tan_105_degree_l203_203569

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203569


namespace length_AD_is_twelve_l203_203022

-- Definitions of the conditions in the problem
def bisects (B C A D : Point) : Prop := 
  ∃ M : Point, M = midpoint A D ∧ distance M C = 6

-- Given conditions
variables {A D B C : Point}
variable (M : Point)
axiom midpoint_AD : M = midpoint A D
axiom distance_MC : distance M C = 6
axiom bisect_BC_AD : bisects B C A D

-- The proof we aim to obtain demonstrating the length AD
theorem length_AD_is_twelve (h1 : M = midpoint A D) (h2 : distance M C = 6) (h3 : bisects B C A D) : distance A D = 12 :=
by
  sorry

end length_AD_is_twelve_l203_203022


namespace tan_105_l203_203467

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203467


namespace tan_105_eq_neg2_sub_sqrt3_l203_203530

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203530


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203507

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203507


namespace tan_105_eq_neg2_sub_sqrt3_l203_203540

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203540


namespace sum_partial_fraction_l203_203675

theorem sum_partial_fraction :
  (∑ n in Finset.range 100, 1 / ((3 * (n + 1) - 2) * (3 * (n + 1) + 1))) = 300 / 1505 :=
by 
  sorry

end sum_partial_fraction_l203_203675


namespace solution_set_of_equation_l203_203122

theorem solution_set_of_equation :
  {p : ℝ × ℝ | p.1 * p.2 + 1 = p.1 + p.2} = {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1} :=
by 
  sorry

end solution_set_of_equation_l203_203122


namespace is_not_age_of_child_l203_203285

-- Initial conditions
def mrs_smith_child_ages : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Given number
def n : Nat := 1124

-- Mrs. Smith's age 
noncomputable def mrs_smith_age : Nat := 46

-- Divisibility check
def is_divisible (n k : Nat) : Bool := n % k = 0

-- Prove the statement
theorem is_not_age_of_child (child_age : Nat) : 
  child_age ∈ mrs_smith_child_ages ∧ ¬ is_divisible n child_age → child_age = 3 :=
by
  intros h
  sorry

end is_not_age_of_child_l203_203285


namespace line_equation_parallel_l203_203685

theorem line_equation_parallel (x y r s : ℝ) (h1 : r = 1) (h2 : s = 0) (h3 : x - 2*y - 2 = 0) :
  ∃ A B C : ℝ, A * x + B * y = C ∧ A = 1 ∧ B = -2 ∧ C = 1 :=
by {
  use [1, -2, 1],
  sorry
}

end line_equation_parallel_l203_203685


namespace locus_of_points_is_circle_l203_203703

theorem locus_of_points_is_circle (l w b : ℝ) (h_b: b > l^2 + w^2) :
  ∀ (P : ℝ × ℝ), (let x := P.1 in
                  let y := P.2 in 
                  4*x^2 - 2*x*l + l^2 + 4*y^2 - 2*y*w + w^2 = b) ↔
                  (∃ (u v : ℝ), 
                  P.1 = u + l / 2 ∧ 
                  P.2 = v + w / 2 ∧ 
                  4*u^2 + 4*v^2 = b - l^2 - w^2) :=
begin
  sorry
end

end locus_of_points_is_circle_l203_203703


namespace p_at_1_l203_203688

def p := λ (x r s : ℝ), 2 * (x - r) * (x - s)

theorem p_at_1 (r s k : ℝ) (hr : r ≠ s) (hk : k ≠ 0) 
(h_double_root : ∀ x, (x - r) * (x - s) = r + k → (x = (r + s) / 2)) :
  p 1 r s = -4 :=
by 
  -- Here the provided conditions and requirements would be used in the proof 
  sorry

end p_at_1_l203_203688


namespace remainder_base12_div_9_l203_203381

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l203_203381


namespace distance_between_points_l203_203376

theorem distance_between_points : 
  let x1 := 1
  let y1 := -3
  let x2 := -4 
  let y2 := 5 in
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = Real.sqrt 89 :=
by
  sorry

end distance_between_points_l203_203376


namespace annie_gives_mary_25_crayons_l203_203445

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l203_203445


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203514

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203514


namespace crayons_given_to_mary_l203_203448

theorem crayons_given_to_mary :
  let pack_crayons := 21 in
  let locker_crayons := 36 in
  let bobby_crayons := locker_crayons / 2 in
  let total_crayons := pack_crayons + locker_crayons + bobby_crayons in
  (total_crayons * (1 / 3) = 25) := by
rfl

end crayons_given_to_mary_l203_203448


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203903

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203903


namespace correct_sum_of_students_l203_203051

def sum_of_students (lb : ℕ) (ub : ℕ) : ℕ :=
  let students := [i | i in List.range (ub - lb + 1), (i + lb - 1) % 8 == 0]
  students.foldr (· + ·) 0

theorem correct_sum_of_students :
  sum_of_students 180 250 = 1953 :=
by
  sorry

end correct_sum_of_students_l203_203051


namespace biff_ticket_cost_l203_203452

theorem biff_ticket_cost :
  ∃ T : ℝ,
    let hours := 3 in
    let earnings := 12 * hours in
    let wifi_cost := 2 * hours in
    let snacks_cost := 3 in
    let headphones_cost := 16 in
    earnings = T + snacks_cost + headphones_cost + wifi_cost ∧ T = 11 :=
by
  use 11
  sorry

end biff_ticket_cost_l203_203452


namespace tan_105_eq_neg2_sub_sqrt3_l203_203622

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203622


namespace two_digit_primes_with_ones_digit_three_count_l203_203776

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203776


namespace tan_105_eq_neg2_sub_sqrt3_l203_203534

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203534


namespace tan_105_eq_neg2_sub_sqrt3_l203_203618

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203618


namespace tan_105_eq_neg2_sub_sqrt3_l203_203578

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203578


namespace sum_of_roots_l203_203130

theorem sum_of_roots :
  (∑ x in {x : ℝ | x^2 + 2023*x + 16 = 2040}, x) = -2023 :=
by
  sorry

end sum_of_roots_l203_203130


namespace tan_105_l203_203548

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203548


namespace triangle_altitude_median_intersection_l203_203240

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def altitude (A B C : Triangle) : Line := sorry
noncomputable def median (A B C : Triangle) : Line := sorry

theorem triangle_altitude_median_intersection (A B C : Point) (H : Point) (M : Point) (L : Point) :
  let triangle_ABC := (A, B, C)
  let triangle_BMC := (B, M, C)
  altitude A B C passes_through H →
  median B M C passes_through L →
  H ≠ L → -- additional condition to avoid trivial cases
  midpoint B M = L →
  ∃ H' M' : Point,
    (altitude B M C passes_through H') ∧
    (median B M C passes_through M') ∧
    midpoint B M M' = H' := sorry

end triangle_altitude_median_intersection_l203_203240


namespace tan_105_l203_203491

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203491


namespace tan_105_eq_neg2_sub_sqrt3_l203_203542

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203542


namespace find_n_l203_203711

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + cos x) = (1/2) * (log 10 n - 2)) :
  n = 102 :=
sorry

end find_n_l203_203711


namespace find_angle_B_l203_203283

-- Define the parallel lines and angles
variables (l m : ℝ) -- Representing the lines as real numbers for simplicity
variables (A C B : ℝ) -- Representing the angles as real numbers

-- The conditions
def parallel_lines (l m : ℝ) : Prop := l = m
def angle_A (A : ℝ) : Prop := A = 100
def angle_C (C : ℝ) : Prop := C = 60

-- The theorem stating that, given the conditions, the angle B is 120 degrees
theorem find_angle_B (l m : ℝ) (A C B : ℝ) 
  (h1 : parallel_lines l m) 
  (h2 : angle_A A) 
  (h3 : angle_C C) : B = 120 :=
sorry

end find_angle_B_l203_203283


namespace surface_area_of_cube_l203_203138

open Real

-- Define the vertices of the cube
def A : ℝ × ℝ × ℝ := (1, 4, 2)
def B : ℝ × ℝ × ℝ := (2, 0, -7)
def C : ℝ × ℝ × ℝ := (5, -5, 1)

-- Define a function to calculate the distance between two points
def distance (P Q : ℝ × ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Theorem statement proving the surface area of the cube
theorem surface_area_of_cube : distance A B = √98 ∧ distance A C = √98 ∧ distance B C = √98 →
  let s := √98 / √2 in 6 * s^2 = 294 :=
by
  sorry

end surface_area_of_cube_l203_203138


namespace problem_period_problem_amplitude_problem_initial_phase_problem_axis_of_symmetry_l203_203739

noncomputable def f (x : ℝ) : ℝ := 3 * sin (x / 2 + π / 6) + 3

theorem problem_period : 
  ∃ T : ℝ, T = 4 * π ∧ 
  (∀ x : ℝ, f (x + T) = f x) := sorry

theorem problem_amplitude : 
  ∃ A : ℝ, A = 3 := sorry

theorem problem_initial_phase : 
  ∃ φ : ℝ, φ = π / 6 := sorry

theorem problem_axis_of_symmetry : 
  ∀ k : ℤ, ∃ ν : ℝ, (ν = 2 * k * π + 2 * π / 3) ∧ 
  (∀ x : ℝ, f (ν + x) = f (ν - x)) := sorry

end problem_period_problem_amplitude_problem_initial_phase_problem_axis_of_symmetry_l203_203739


namespace find_x_coordinate_l203_203166

theorem find_x_coordinate 
  (A B : ℝ × ℝ)
  (line : ℝ → ℝ)
  (m : ℝ) 
  (P : ℝ × ℝ) 
  (triangle_right_angled : Prop) :
  A = (-2, 0) →
  B = (4, 0) →
  line = (λ x, 0.5 * x + 2) →
  P = (m, line m) →
  triangle_right_angled ↔ m = -2 ∨ m = 4 ∨ m = 4 * real.sqrt 5 / 5 ∨ m = -4 * real.sqrt 5 / 5 :=
by
  intros
  sorry

end find_x_coordinate_l203_203166


namespace hours_reduction_l203_203016

-- Define the initial conditions
variables (W H : ℝ) -- W: original hourly wage, H: original hours worked

-- Define the new wage and hours
def new_wage := W * 1.10
def new_hours := H / 1.10

-- Define the weekly income condition
def total_income_constant := W * H = new_wage * new_hours

-- Define the percentage reduction in hours
def percentage_reduction := (H - new_hours) / H * 100

-- The theorem to prove the percentage reduction is approximately 9.09%
theorem hours_reduction (W H : ℝ) (h1 : W > 0) (h2 : H > 0) :
  (total_income_constant W H) → percentage_reduction W H ≈ 9.09 :=
by
  sorry

end hours_reduction_l203_203016


namespace sum_a1_to_a8_sum_a1_a3_a5_a7_l203_203710

namespace ProofProblem

-- Definitions
def polynomial_expansion (x : ℝ) : ℝ := 
  (x + 2)^8

def linear_combination (x : ℝ) (a : Fin 9 → ℝ) : ℝ :=
  ∑ i in Finset.range 9, a i * (x + 1)^i

-- The problem states that these polynomials are equal for all x in ℝ
axiom polynomial_relation (a : Fin 9 → ℝ) : 
  ∀ x : ℝ, polynomial_expansion x = linear_combination x a

-- Prove the two statements
theorem sum_a1_to_a8 (a : Fin 9 → ℝ) : 
  polynomial_relation a → (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 255) := 
by sorry

theorem sum_a1_a3_a5_a7 (a : Fin 9 → ℝ) : 
  polynomial_relation a → (a 1 + a 3 + a 5 + a 7 = 128) := 
by sorry

end ProofProblem

end sum_a1_to_a8_sum_a1_a3_a5_a7_l203_203710


namespace count_two_digit_primes_with_ones_digit_3_l203_203798

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203798


namespace hyperbola_eccentricity_l203_203266

open Real

/-- Given a hyperbola x^2/a^2 - y^2/b^2 = 1 with a > 0 and b > 0, and foci F1 and F2,
if |PF2| = |F1F2| and the distance from F2 to the line PF1 equals the length of the real axis,
then the eccentricity e = 5/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (F1 F2 P : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : abs (P - F2) = abs (F1 - F2)) (h4 : abs (F2 - (P + F1) / 2) = 2 * a) :
  let e := F2 / a in
  e = 5 / 3 :=
begin
  sorry
end

end hyperbola_eccentricity_l203_203266


namespace count_two_digit_primes_ending_with_3_l203_203839

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203839


namespace tan_add_tan_105_eq_l203_203630

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203630


namespace count_two_digit_primes_with_ones_digit_three_l203_203780

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203780


namespace count_two_digit_primes_ending_in_3_l203_203821

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203821


namespace count_two_digit_primes_with_ones_3_l203_203884

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203884


namespace two_digit_primes_end_in_3_l203_203914

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203914


namespace number_of_6th_sample_is_245_l203_203111

theorem number_of_6th_sample_is_245:
  (number_of_6th_sample 800 ["84", "42", "17", "53", "31", "57", "24", "55", "06", "88",
   "77", "04", "74", "42", "45", "76", "72", "76", "33", "50", "25", "83", "06", "76",
   "63", "01", "63", "78", "59", "16", "95", "56", "67", "19", "98", "10", "50", "71", 
   "75", "12", "86", "73", "58", "07", "44", "39", "52", "38", "79", "33", "21", "12", 
   "34", "29", "78", "64", "56", "07", "82", "52", "42", "07", "44", "38", "15", "51", 
   "00", "13", "42", "99", "66", "02", "79", "54"]) = 245 := 
sorry

end number_of_6th_sample_is_245_l203_203111


namespace passes_through_point_l203_203335

variables (n : ℤ) (a : ℝ)

noncomputable
def f (x : ℝ) : ℝ := x^n + a^(x-1)

theorem passes_through_point 
  (h_n : n ∈ (ℤ : Type)) 
  (h_a_pos : 0 < a) 
  (h_a_ne_one : a ≠ 1) :
  f n a 1 = 2 :=
by
  sorry

end passes_through_point_l203_203335


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203612

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203612


namespace tan_105_eq_neg2_sub_sqrt3_l203_203584

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203584


namespace num_satisfying_inequality_l203_203760

theorem num_satisfying_inequality : ∃ (s : Finset ℤ), (∀ n ∈ s, (n + 4) * (n - 8) ≤ 0) ∧ s.card = 13 := by
  sorry

end num_satisfying_inequality_l203_203760


namespace angle_between_a_and_b_l203_203194

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
def vector_magnitude_a : ℝ := ∥a∥
def vector_magnitude_b : ℝ := ∥b∥
def vector_c : V := a + b
def vectors_orthogonal : inner_product_space.orthogonal ℝ (a) (c) := by sorry

-- The statement to prove
theorem angle_between_a_and_b (ha : vector_magnitude_a = 1) (hb : vector_magnitude_b = 2)
  (hc : c = vector_c) (h_orth : vectors_orthogonal) : 
  real.angle a b = real.pi * 2 / 3 :=
  sorry

end angle_between_a_and_b_l203_203194


namespace count_two_digit_primes_with_ones_digit_3_l203_203802

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203802


namespace remainder_sum_div_11_l203_203456

theorem remainder_sum_div_11 :
  ((100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007 + 100008 + 100009 + 100010) % 11) = 10 :=
by
  sorry

end remainder_sum_div_11_l203_203456


namespace two_digit_primes_with_ones_digit_three_count_l203_203775

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203775


namespace proof_problem_l203_203217

variable {a b c : ℝ} {A B C : ℝ}

-- Given conditions
def condition1 : Prop := 
  (b / Real.cos B) = ((3 * c - a) / Real.cos A)

noncomputable def question1 : Prop :=
  a = Real.sqrt 2 * Real.sin A → b = 4 / 3

noncomputable def question2 : Prop :=
  b = 3 ∧ (1 / 2) * a * c * Real.sin B = 2 * Real.sqrt 2 → a + c = 5

-- Proof statement combining conditions with the questions
theorem proof_problem : condition1 → question1 ∧ question2 :=
  by
  sorry

end proof_problem_l203_203217


namespace factorize_expression_l203_203676

theorem factorize_expression (a b : ℝ) : 
  a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 := by sorry

end factorize_expression_l203_203676


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203510

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203510


namespace count_two_digit_primes_with_ones_digit_3_l203_203815

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203815


namespace correct_sum_of_students_l203_203052

def sum_of_students (lb : ℕ) (ub : ℕ) : ℕ :=
  let students := [i | i in List.range (ub - lb + 1), (i + lb - 1) % 8 == 0]
  students.foldr (· + ·) 0

theorem correct_sum_of_students :
  sum_of_students 180 250 = 1953 :=
by
  sorry

end correct_sum_of_students_l203_203052


namespace find_a_and_x_l203_203353

theorem find_a_and_x (a x : ℚ) (h1 : 0 < x) (h2 : sqrt x = 2 * a - 3) (h3 : sqrt x = 5 - a) :
  a = 8 / 3 ∧ x = 49 / 9 :=
by sorry

end find_a_and_x_l203_203353


namespace count_valid_m_values_l203_203137

theorem count_valid_m_values : ∃ (count : ℕ), count = 72 ∧
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5000 →
     (⌊Real.sqrt m⌋ = ⌊Real.sqrt (m+125)⌋)) ↔ count = 72 :=
by
  sorry

end count_valid_m_values_l203_203137


namespace polynomial_horner_v4_value_l203_203457

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define Horner's Rule step by step for x = 2
def horner_eval (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  let v4 := v3 * x + 240
  v4

-- Prove that the value of v4 when x = 2 is 80
theorem polynomial_horner_v4_value : horner_eval 2 = 80 := by
  sorry

end polynomial_horner_v4_value_l203_203457


namespace circles_radii_sum_l203_203459

-- Definitions of the radii of the circles.
variables {r r1 r2 : ℝ}

-- Assume the conditions about the circles.
-- O1 touches two sides of the triangle.
-- O2 touches two different sides of the triangle than O1.
-- O1 and O2 touch each other externally.
-- We want to prove that r1 + r2 > r.
theorem circles_radii_sum (r r1 r2 : ℝ) 
  (O1_touches_two_sides : ¬ (O1_touches_two_sides = ∅)) -- O1 touches two sides
  (O2_touches_two_other_sides : ¬ (O2_touches_two_other_sides = ∅)) -- O2 touches two other sides
  (O1_O2_touch_externally : ¬ (O1_O2_touch_externally = ∅)) -- O1 and O2 touch externally
  : r1 + r2 > r :=
sorry

end circles_radii_sum_l203_203459


namespace factorize_expression_l203_203118

theorem factorize_expression (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end factorize_expression_l203_203118


namespace arccos_gt_arcsin_solution_set_l203_203346

theorem arccos_gt_arcsin_solution_set (λ : ℝ) :
  {x : ℝ | ∃ k : ℤ, 2 * (k : ℝ) * Real.pi + Real.pi / 2 < x ∧ x < 2 * ((k + 1) : ℝ) * Real.pi} =
    {x : ℝ | ∃ k : ℤ, Real.arccos (Real.cos λ) > Real.arcsin (Real.sin x)} :=
sorry

end arccos_gt_arcsin_solution_set_l203_203346


namespace two_digit_primes_with_ones_digit_three_count_l203_203774

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203774


namespace min_h10_value_l203_203075

def tenuous_function (h : ℕ → ℤ) : Prop :=
  ∀ (x y : ℕ), x > 0 → y > 0 → h(x) + h(y) > 2 * y * y

theorem min_h10_value :
  ∀ (h : ℕ → ℤ), (tenuous_function h) →
  (∀ n : ℕ, n > 0 → h(1) + h(2) + ⋯ + h(15) = 2201) →
  h(10) = 137 :=
by
  intro h h_tenuous h_sum
  sorry

end min_h10_value_l203_203075


namespace count_two_digit_primes_with_ones_digit_three_l203_203785

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203785


namespace inverse_89_mod_91_l203_203679

theorem inverse_89_mod_91 : ∃ x ∈ set.Icc 0 90, (89 * x) % 91 = 1 :=
by
  use 45
  split
  · exact ⟨le_refl 45, le_of_lt (by norm_num)⟩
  · norm_num; sorry

end inverse_89_mod_91_l203_203679


namespace degrees_minutes_conversion_l203_203082

-- Define conversion rates
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_degrees (m : ℝ) : ℝ := m / 60

-- Define the given problem in Lean 4
theorem degrees_minutes_conversion :
  let d := 18 + minutes_to_degrees 24
  d = 18.4 :=
by
  let d := 18 + minutes_to_degrees 24
  show d = 18.4
  sorry

end degrees_minutes_conversion_l203_203082


namespace values_of_m_l203_203183

theorem values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, mx + 1 = 0}
  B ⊆ A ↔ m ∈ {-1, -1/3, 0} :=
by
  sorry

end values_of_m_l203_203183


namespace tire_lifespan_estimate_l203_203247

noncomputable def estimate_tires_within_range (mu sigma : ℝ) (n : ℕ) : ℕ :=
  let p := 0.9544
  in nat_floor (n * p)

theorem tire_lifespan_estimate :
  let mu := 36203
  let sigma := 4827
  let n := 500
  estimate_tires_within_range mu sigma n = 477 :=
by
  unfold estimate_tires_within_range
  sorry

end tire_lifespan_estimate_l203_203247


namespace find_xyz_l203_203242

variable {Point : Type} [AddCommGroup Point] [VectorSpace ℝ Point]

variables (A B C E F P : Point)

def ratio_AC : ℝ := 3 / 5
def ratio_AB : ℝ := 3 / 4

/-- E divides AC in the ratio 3:2 --/
def E_def : Point := ratio_AC • A + (1 - ratio_AC) • C

/-- F divides AB in the ratio 3:1 --/
def F_def : Point := ratio_AB • A + (1 - ratio_AB) • B

/-- P is the intersection of BE and CF --/
variables (BE CF : Set Point)

axiom P_is_intersection : P ∈ BE ∧ P ∈ CF

theorem find_xyz :
  let x := 8 / 15
      y := 1 / 9
      z := 16 / 45 in
  P = x • A + y • B + z • C ∧ x + y + z = 1 :=
sorry

end find_xyz_l203_203242


namespace tan_105_eq_neg2_sub_sqrt3_l203_203535

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203535


namespace count_two_digit_primes_with_ones_digit_3_l203_203812

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203812


namespace count_two_digit_primes_ending_with_3_l203_203840

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203840


namespace area_PQR_l203_203454

-- Definitions of the points
def P : ℝ × ℝ := (-5, 4)
def Q : ℝ × ℝ := (1, 7)
def R : ℝ × ℝ := (3, -1)

-- Definition of the area formula of a triangle given its vertices
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The proof statement
theorem area_PQR : area_of_triangle P Q R = 27 :=
by
  -- Proof would go here, but we insert "sorry" to skip it
  sorry

end area_PQR_l203_203454


namespace two_digit_primes_end_in_3_l203_203910

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203910


namespace exists_positive_integers_increasing_by_2008_times_l203_203109

theorem exists_positive_integers_increasing_by_2008_times :
  ∃ (x : Fin 14 → ℕ), (∀ i, 0 < x i) ∧ (∏ i, (x i + 1) / x i = 2008) :=
by
  sorry

end exists_positive_integers_increasing_by_2008_times_l203_203109


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203974

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203974


namespace weight_in_kilograms_l203_203029

-- Definitions based on conditions
def weight_of_one_bag : ℕ := 250
def number_of_bags : ℕ := 8

-- Converting grams to kilograms (1000 grams = 1 kilogram)
def grams_to_kilograms (grams : ℕ) : ℕ := grams / 1000

-- Total weight in grams
def total_weight_in_grams : ℕ := weight_of_one_bag * number_of_bags

-- Proof that the total weight in kilograms is 2
theorem weight_in_kilograms : grams_to_kilograms total_weight_in_grams = 2 :=
by
  sorry

end weight_in_kilograms_l203_203029


namespace travel_ways_l203_203363

theorem travel_ways (highways : ℕ) (railways : ℕ) (n : ℕ) :
  highways = 3 → railways = 2 → n = highways + railways → n = 5 :=
by
  intros h_eq r_eq n_eq
  rw [h_eq, r_eq] at n_eq
  exact n_eq

end travel_ways_l203_203363


namespace PH_fixed_point_l203_203148

theorem PH_fixed_point (circle : Type)
  (A B : circle)
  (not_diameter : A ≠ B)
  (C : circle → circle → circle) -- C moves along the large arc AB
  (H : circle × circle × circle → circle) -- orthocenter H of triangle ABC
  (circle_passing_ACH : circle × circle → circle)
  (P : circle × circle → circle) -- P is the re-intersection point with line BC
  :
  ∃ X : circle, ∀ C, (PH : line (circle_passing_ACH (C, H(A, B, C))) (P(C,H(A,B,C)))) passes through X := 
sorry

end PH_fixed_point_l203_203148


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203482

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203482


namespace consecutive_integers_symbols_l203_203361

variables (x1 x2 x3 x4 : ℕ)
variables (s1 s2 s3 s4 : string)

-- Define the symbol representation mappings
def symbol_1 := "1"
def symbol_9 := "9"
def symbol_0 := "0"
def symbol_2 := "2"

-- Given conditions
def condition1 := s1 = "$\\square \\diamond \\diamond$" ∧ x1 = 199
def condition2 := s2 = "$\\vee \\triangle \\Delta$" ∧ x2 = 200
def condition3 := s3 = "$\\vee \\triangle \\square$" ∧ x3 = 201

-- Desired conclusion
def conclusion := s4 = "$\\vee \\triangle \\nabla$" ∧ x4 = 202

theorem consecutive_integers_symbols :
  condition1 →
  condition2 →
  condition3 →
  conclusion :=
sorry

end consecutive_integers_symbols_l203_203361


namespace two_digit_primes_with_ones_digit_3_l203_203865

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203865


namespace find_n_after_folding_l203_203690

theorem find_n_after_folding (n : ℕ) (h : 2 ^ n = 128) : n = 7 := by
  sorry

end find_n_after_folding_l203_203690


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203602

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203602


namespace find_guest_sets_l203_203451

-- Definitions based on conditions
def cost_per_guest_set : ℝ := 32.0
def cost_per_master_set : ℝ := 40.0
def num_master_sets : ℕ := 4
def total_cost : ℝ := 224.0

-- The mathematical problem
theorem find_guest_sets (G : ℕ) (total_cost_eq : total_cost = cost_per_guest_set * G + cost_per_master_set * num_master_sets) : G = 2 :=
by
  sorry

end find_guest_sets_l203_203451


namespace tan_105_l203_203489

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203489


namespace tan_105_l203_203462

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203462


namespace sum_of_digits_base2_345_l203_203008

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end sum_of_digits_base2_345_l203_203008


namespace tan_105_degree_l203_203650

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203650


namespace num_two_digit_primes_with_ones_digit_3_l203_203954

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203954


namespace marble_probability_l203_203695

theorem marble_probability :
  let total_ways := Nat.choose 9 4,
      favorable_ways := 3 * 3 * 3 * Nat.choose 6 1 - 3 * (Nat.choose 3 2 * 3 * 3)
  in (favorable_ways : ℚ) / total_ways = 9 / 14 :=
by
  sorry

end marble_probability_l203_203695


namespace trains_clear_time_l203_203371

noncomputable def length_train1 := 135 -- meters
noncomputable def length_train2 := 165 -- meters
noncomputable def speed_train1_kmh := 80 -- km/h
noncomputable def speed_train2_kmh := 65 -- km/h

noncomputable def total_length := length_train1 + length_train2 -- meters
noncomputable def relative_speed_kmh := speed_train1_kmh + speed_train2_kmh -- km/h
noncomputable def relative_speed_ms := (relative_speed_kmh * 1000) / 3600 -- m/s

noncomputable def time_to_clear := total_length / relative_speed_ms -- seconds

theorem trains_clear_time :
  time_to_clear = 7.448 :=
by
  sorry

end trains_clear_time_l203_203371


namespace two_digit_primes_with_ones_digit_three_count_l203_203770

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203770


namespace tan_105_eq_neg2_sub_sqrt3_l203_203574

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203574


namespace problem_statement_l203_203177

noncomputable theory

def xy_is_perfect_square (x y : ℕ) : Prop :=
  (x * y + 4) = (x + 2) * (x + 2)

theorem problem_statement (x y : ℕ) (h : x > 0 ∧ y > 0) : 
  (1/x + 1/y + 1/(x * y) = 1/(x + 4) + 1/(y - 4) + 1/((x + 4) * (y - 4))) → xy_is_perfect_square x y :=
by
  sorry

end problem_statement_l203_203177


namespace tan_105_l203_203493

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203493


namespace count_two_digit_primes_with_ones_3_l203_203881

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203881


namespace count_two_digit_primes_with_ones_digit_3_l203_203801

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203801


namespace order_of_abc_l203_203142

theorem order_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a^2 + b^2 < a^2 + c^2) (h2 : a^2 + c^2 < b^2 + c^2) : a < b ∧ b < c := 
by
  sorry

end order_of_abc_l203_203142


namespace decimal_to_binary_45_l203_203092

theorem decimal_to_binary_45 :
  (45 : ℕ) = (0b101101 : ℕ) :=
sorry

end decimal_to_binary_45_l203_203092


namespace two_digit_primes_with_ones_digit_3_l203_203862

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203862


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203506

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203506


namespace calculate_BC_l203_203324

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem calculate_BC
  (AB CD : ℝ)        -- lengths of bases
  (h : ℝ)            -- height (altitude)
  (area : ℝ)         -- area of trapezoid
  (total_area : area = 272)
  (h_eq : h = 10)
  (AB_eq : AB = 12)
  (CD_eq : CD = 22) :
  let AE := real.sqrt (AB^2 - h^2),
      FD := real.sqrt (CD^2 - h^2),
      area_AEB := 5 * AE,
      area_DFC := 5 * FD,
      x := (272 - area_AEB - area_DFC) / h in
  x = (272 - 5 * real.sqrt 44 - 5 * real.sqrt 384) / 10 :=
by
  sorry

end calculate_BC_l203_203324


namespace cos_sin_of_triangle_l203_203218

noncomputable def angle_A_equals_90 : Prop := ∃ (A B C : ℝ), A = 0 ∧ B = π / 2

noncomputable def tan_C_equals_2 : Prop := ∃ (A B C : ℝ), tan B = 2

theorem cos_sin_of_triangle (A B C : ℝ) (hA : angle_A_equals_90) (hC : tan_C_equals_2) :
  cos C = Real.sqrt 5 / 5 ∧ sin C = 2 * Real.sqrt 5 / 5 :=
by sorry

end cos_sin_of_triangle_l203_203218


namespace maximum_value_expression_l203_203357

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end maximum_value_expression_l203_203357


namespace tan_add_tan_105_eq_l203_203635

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203635


namespace tan_105_eq_neg2_sub_sqrt3_l203_203533

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203533


namespace problem_solution_l203_203200

noncomputable def number_of_solutions (x : ℝ) : Prop :=
(-19 < x) ∧ (x < 98) ∧ (cos(x)^2 + 2 * sin(x)^2 = 1)

theorem problem_solution : ∃ n : ℕ, n = 38 ∧
  ∀ x : ℝ, number_of_solutions x ↔ ∃ k : ℤ, x = k * Real.pi ∧ -19 < k * Real.pi ∧ k * Real.pi < 98 :=
by sorry

end problem_solution_l203_203200


namespace two_digit_primes_ending_in_3_eq_6_l203_203925

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203925


namespace sin_decreasing_periodic_l203_203070

theorem sin_decreasing_periodic (x : ℝ) (hx : x ∈ set.Icc (Real.pi / 4) (Real.pi / 2)) :
  (∀ x, sin(x + Real.pi) = sin x) ∧ (∀ x, (x ∈ set.Icc (Real.pi / 4) (Real.pi / 2)) → 
  ∀ y, (differentiable_at ℝ (λ x, sin (2 * x + Real.pi / 2)) x) →
  (differentiable_at ℝ sin x) ∧ (((deriv (λ t, sin (2 * t + Real.pi / 2)) x) = 0) → 
  ((cos (2 * x + Real.pi / 2)) = 0) )) sorry

end sin_decreasing_periodic_l203_203070


namespace pq_sum_correct_l203_203271

theorem pq_sum_correct {c : ℝ} (hc : c ∈ Icc (-20 : ℝ) 20) (p q : ℕ) 
  (h_rel_prime : Nat.coprime p q) 
  (h_prob : ((37.3333 / 40) : ℚ) = p / q) : 
  p + q = 29 :=
sorry

end pq_sum_correct_l203_203271


namespace eight_p_plus_one_composite_l203_203018

theorem eight_p_plus_one_composite 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h8p_minus_one : Nat.Prime (8 * p - 1))
  : ¬ (Nat.Prime (8 * p + 1)) :=
sorry

end eight_p_plus_one_composite_l203_203018


namespace two_digit_primes_ending_in_3_eq_6_l203_203928

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203928


namespace tan_add_tan_105_eq_l203_203639

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203639


namespace tan_add_tan_105_eq_l203_203628

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203628


namespace tan_105_eq_neg2_sub_sqrt3_l203_203620

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203620


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203902

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203902


namespace two_digit_primes_end_in_3_l203_203916

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203916


namespace josh_earns_per_hour_l203_203253

theorem josh_earns_per_hour :
  ∃ (J : ℝ), 
    (∃ (h1 : 8 * 5 * 4 = 160), 
     ∃ (h2 : (8 - 2) * 5 * 4 = 120), 
     ∃ (C : ℝ), C = (1 / 2) * J) ∧
    (160 * J + 120 * ((1 / 2) * J) = 1980) ∧ J = 9 :=
begin
  use 9,
  split,
  { use 160,
    use 120,
    use (1 / 2) * 9,
    refl, },
  split,
  { norm_num,
    ring, },
  refl,
end

end josh_earns_per_hour_l203_203253


namespace count_two_digit_primes_with_ones_digit_three_l203_203788

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203788


namespace problem_is_correct_answer_l203_203066

def A : Set ℕ := {1, 2}
def B : Set ℤ := {x | x = 1 ∨ x = 2 ∨ x = 4 ∨ x = -1 ∨ x = -2 ∨ x = -4}
def set1 : Set ℤ := {y | ∃ x, y = 2 * x^2 - 3}
def set2 : Set (ℤ × ℤ) := {(x, y) | y = 2 * x^2 - 3}
def set3 : Set ℚ := {1, 3 / 2, 6 / 4, | - 1 / 2 |, 0.5}
def set4 : Set (ℝ × ℝ) := {(x, y) | x * y ≤ 0}

theorem problem_is_correct_answer : 
  (A ≠ B ∧ set1 ≠ set2 ∧ (set3.card = 3) ∧ set4 ≠ {p : ℝ × ℝ | p.1 ≠ 0 ∧ p.2 ≠ 0 } ) → 0 = 0 := 
by
  sorry

end problem_is_correct_answer_l203_203066


namespace tan_add_tan_105_eq_l203_203632

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203632


namespace tan_105_degree_l203_203565

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203565


namespace find_ellipse_equation_find_area_ratio_range_l203_203707

-- Definitions of the ellipse and its properties
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a b c : ℝ) : Prop :=
  c = sqrt (a^2 - b^2) ∧ (c / a = sqrt 2 / 2)

def area_of_triangle (a b c : ℝ) : Prop := 
  b * c = 1

-- Proof problem 1: Finding the standard equation of the ellipse
theorem find_ellipse_equation (a b c : ℝ) (H_ellipse : ellipse a b 0) (H_eccentricity : eccentricity a b c) (H_area : area_of_triangle a b c) :
  ∃ a b, (a = sqrt 2 ∧ b = 1 ∧ (λ x y, x^2 / 2 + y^2 = 1)) := 
sorry

-- Definitions of the line intersection and area ratios
def line_through_point (m : ℝ) (D : ℝ × ℝ) (x y : ℝ) : Prop := 
  D = (2, 0) ∧ x = m * y + 2

def is_interior (M N : ℝ × ℝ) : Prop := 
  M.1 < N.1 ∧ M.2 < N.2

def ellipse_intersection (a b : ℝ) (m : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 = m * y1 + 2) ∧ (x2 = m * y2 + 2) ∧ (a = sqrt 2 ∨ b = 1 ∨ (x1^2 / 2 + y1^2 = 1)) ∧ (x2^2 / 2 + y2^2 = 1)

-- Proof problem 2: Finding the range of the ratio of triangle areas
theorem find_area_ratio_range (a b : ℝ) (m x1 y1 x2 y2 : ℝ) (H_eq : a = sqrt 2) (H_val : b = 1) (H_intersect : ellipse_intersection a b m x1 y1 x2 y2) :
  3 - 2 * sqrt 2 < (abs y1 / abs y2) ∧ (abs y1 / abs y2) < 1 :=
sorry

end find_ellipse_equation_find_area_ratio_range_l203_203707


namespace count_two_digit_primes_with_ones_3_l203_203879

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203879


namespace tan_105_eq_neg2_sub_sqrt3_l203_203528

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203528


namespace two_digit_primes_with_ones_digit_three_count_l203_203766

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203766


namespace train_length_is_250_l203_203060

def length_of_train (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_mps := (train_speed_kmph - man_speed_kmph) * (1000 / 3600)
  relative_speed_mps * time_s

theorem train_length_is_250 :
  length_of_train 68 8 14.998800095992321 = 250 := by
  sorry

end train_length_is_250_l203_203060


namespace percentage_increase_l203_203418

theorem percentage_increase (a : ℕ) (x : ℝ) (b : ℝ) (r : ℝ) 
    (h1 : a = 1500) 
    (h2 : r = 0.6) 
    (h3 : b = 1080) 
    (h4 : a * (1 + x / 100) * r = b) : 
    x = 20 := 
by 
  sorry

end percentage_increase_l203_203418


namespace count_two_digit_primes_with_ones_digit_three_l203_203797

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203797


namespace two_digit_primes_with_ones_digit_three_count_l203_203767

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203767


namespace sam_more_than_avg_l203_203453

def bridget_count : ℕ := 14
def reginald_count : ℕ := bridget_count - 2
def sam_count : ℕ := reginald_count + 4
def average_count : ℕ := (bridget_count + reginald_count + sam_count) / 3

theorem sam_more_than_avg 
    (h1 : bridget_count = 14) 
    (h2 : reginald_count = bridget_count - 2) 
    (h3 : sam_count = reginald_count + 4) 
    (h4 : average_count = (bridget_count + reginald_count + sam_count) / 3): 
    sam_count - average_count = 2 := 
  sorry

end sam_more_than_avg_l203_203453


namespace max_digit_sum_l203_203412

-- Define the condition for the hours and minutes digits
def is_valid_hour (h : ℕ) := 0 ≤ h ∧ h < 24
def is_valid_minute (m : ℕ) := 0 ≤ m ∧ m < 60

-- Define the function to calculate the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Main statement: Prove that the maximum sum of the digits in the display is 24
theorem max_digit_sum : ∃ h m: ℕ, is_valid_hour h ∧ is_valid_minute m ∧ 
  sum_of_digits h + sum_of_digits m = 24 :=
sorry

end max_digit_sum_l203_203412


namespace hyperbola_eccentricity_l203_203165

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = b) :
  let e := Real.sqrt(1 + (b^2 / a^2))
  in e = Real.sqrt 2 :=
by
  let e := Real.sqrt (1 + (b^2 / a^2))
  sorry

end hyperbola_eccentricity_l203_203165


namespace n_eq_sum_of_digits_plus_9_sum_of_stumps_l203_203135

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def stumps (n : ℕ) : List ℕ :=
  (List.range n.digits.length).tail.map (λ i => n % (10 ^ i))

def sum_of_stumps (n : ℕ) : ℕ :=
  (stumps n).sum

theorem n_eq_sum_of_digits_plus_9_sum_of_stumps (n : ℕ) (h : n > 0) :
  n = sum_of_digits n + 9 * sum_of_stumps n :=
sorry

end n_eq_sum_of_digits_plus_9_sum_of_stumps_l203_203135


namespace final_selling_price_l203_203063

def actual_price : ℝ := 9356.725146198829
def price_after_first_discount (P : ℝ) : ℝ := P * 0.80
def price_after_second_discount (P1 : ℝ) : ℝ := P1 * 0.90
def price_after_third_discount (P2 : ℝ) : ℝ := P2 * 0.95

theorem final_selling_price :
  (price_after_third_discount (price_after_second_discount (price_after_first_discount actual_price))) = 6400 :=
by 
  -- Here we would need to provide the proof, but it is skipped with sorry
  sorry

end final_selling_price_l203_203063


namespace linear_regression_eq_l203_203115

theorem linear_regression_eq (x y : Fin 5 → ℕ)
  (hx : x = ![6, 7, 8, 9, 10])
  (hy : y = ![10, 12, 11, 12, 20]) :
  let x̄ := (6 + 7 + 8 + 9 + 10) / 5
  let ȳ := (10 + 12 + 11 + 12 + 20) / 5
  let sxx := (Array.sum (Array.map (λ i, (i - x̄)^2) x))
  let syy := (Array.sum (Array.map (λ i, (i - ȳ)^2) y))
  let sxy := (Array.sum (Array.map (λ i, (x[i] - x̄) * (y[i] - ȳ)) (Fin 5)))
  let b := sxy / sxx
  let a := ȳ - b * x̄
  in a = -3 ∧ b = 2 :=
sorry

end linear_regression_eq_l203_203115


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203990

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203990


namespace simplify_rationalize_denominator_l203_203309

-- Definitions from the conditions
def fraction_term : ℝ := 1 / (sqrt 5 + 2)
def simplified_term : ℝ := sqrt 5 - 2
def main_expression : ℝ := 1 / (2 + fraction_term)

theorem simplify_rationalize_denominator :
  main_expression = sqrt 5 / 5 := by
  sorry

end simplify_rationalize_denominator_l203_203309


namespace count_two_digit_primes_with_ones_3_l203_203873

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203873


namespace arith_seq_sum_l203_203714

theorem arith_seq_sum 
  (a : ℕ → ℤ)
  (h_arith : ∀ n, a(n+1) - a n = a 1 - a 0)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := by
  sorry

end arith_seq_sum_l203_203714


namespace monotonic_function_identity_l203_203717

theorem monotonic_function_identity (f : ℝ → ℝ) 
  (h_mono : ∀ ⦃x y⦄, 0 < x → 0 < y → x ≤ y → f(x) ≤ f(y))
  (h_eq : ∀ x, 0 < x → f(f(x) - 1/x) = 2) :
  f (1/5) = 6 := 
sorry

end monotonic_function_identity_l203_203717


namespace count_two_digit_primes_ending_with_3_l203_203838

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203838


namespace tangent_line_eqn_l203_203331

noncomputable theory

open Real

-- Define the function y = x * ln x
def f (x : ℝ) : ℝ := x * log x

-- Define the point of tangency
def pt : ℝ × ℝ := (exp 1, exp 1)

-- Prove the equation of the tangent line at the point (e, e)
theorem tangent_line_eqn : 
  let slope := deriv f (exp 1)
  let tangent_eqn := λ x, slope * (x - (exp 1)) + exp 1 in
  tangent_eqn = λ x, 2 * x - exp 1 :=
by
  have deriv_f : (deriv f) (exp 1) = 2 := sorry
  have tan_eqn : ∀ x, (λ x, 2 * (x - exp 1) + exp 1) x = 2 * x - exp 1 := sorry
  show tangent_eqn = λ x, 2 * x - exp 1
  by funext; apply tan_eqn

end tangent_line_eqn_l203_203331


namespace opposite_points_number_line_l203_203417

theorem opposite_points_number_line (a : ℤ) (h : a - 6 = -a) : a = 3 := by
  sorry

end opposite_points_number_line_l203_203417


namespace algebraic_expression_l203_203161

theorem algebraic_expression (m : ℝ) (hm : m^2 + m - 1 = 0) : 
  m^3 + 2 * m^2 + 2014 = 2015 := 
by
  sorry

end algebraic_expression_l203_203161


namespace number_of_valid_permutations_l203_203450

def is_valid_permutation (perm : List Int) : Prop :=
  perm.length = 5 ∧ 
  perm.nodup ∧ 
  perm.perm [1, 2, 3, 4, 5] ∧ 
  perm.get! 4 % 2 = 1 ∧ 
  ∀ i : Int, 0 ≤ i ∧ i ≤ 2 → (perm.get! (i.to_nat)) % perm.get! (i.to_nat) = 0 

-- Theorem: The number of valid permutations satisfying the conditions is exactly 5.
theorem number_of_valid_permutations :
  { p : List Int // is_valid_permutation p }.card = 5 :=
begin
  sorry
end

end number_of_valid_permutations_l203_203450


namespace line_intersects_parabola_l203_203747

theorem line_intersects_parabola (k : ℝ) (hk : k ≠ 0) :
  let C := λ y : ℝ, y^2,
      l := λ x : ℝ, k * x + 1,
      intersection_points := fun (x y : ℝ) => y^2 = x ∧ y = k * x + 1 
  in (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∃ y1 y2 : ℝ, intersection_points x1 y1 ∧ intersection_points x2 y2)) ↔
      (hk ∧ k < 1/4) :=
sorry

end line_intersects_parabola_l203_203747


namespace MISSISSIPPI_arrangement_l203_203080

theorem MISSISSIPPI_arrangement : 
  (factorial 10) / ((factorial 1) * (factorial 4) * (factorial 4) * (factorial 1)) = 6300 := 
by
  sorry

end MISSISSIPPI_arrangement_l203_203080


namespace tan_105_eq_neg2_sub_sqrt3_l203_203531

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203531


namespace polygon_properties_l203_203724

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l203_203724


namespace greatest_large_chips_l203_203030

theorem greatest_large_chips :
  ∃ (l : ℕ), (∃ (s : ℕ), ∃ (p : ℕ), s + l = 70 ∧ s = l + p ∧ Nat.Prime p) ∧ 
  (∀ (l' : ℕ), (∃ (s' : ℕ), ∃ (p' : ℕ), s' + l' = 70 ∧ s' = l' + p' ∧ Nat.Prime p') → l' ≤ 34) :=
sorry

end greatest_large_chips_l203_203030


namespace sum_of_binary_digits_345_l203_203006

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else List.reverse (List.unfold (λ n, if n = 0 then none else some (n % 2, n / 2)) n)

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.foldr (· + ·) 0
  
-- Define the specific example
def digits_of_345 : List ℕ := decimal_to_binary 345

def sum_of_digits_of_345 : ℕ := sum_of_digits digits_of_345

theorem sum_of_binary_digits_345 : sum_of_digits_of_345 = 5 :=
by 
  sorry

end sum_of_binary_digits_345_l203_203006


namespace overall_gain_percentage_l203_203415

noncomputable def gain_percentage (cp_a cp_b cp_c sp_a sp_b sp_c : ℕ) : ℝ :=
  let tcp := cp_a + cp_b + cp_c
  let tsp := sp_a + sp_b + sp_c
  let gain := tsp - tcp
  ((gain : ℝ) / (tcp : ℝ)) * 100

theorem overall_gain_percentage :
  gain_percentage 110 180 230 125 210 280 ≈ 18.27 :=
by 
  -- Here you can provide the proof
  sorry

end overall_gain_percentage_l203_203415


namespace tan_105_eq_neg2_sub_sqrt3_l203_203621

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203621


namespace number_of_true_propositions_is_2_l203_203068

theorem number_of_true_propositions_is_2 :
  let P1 := "From a uniformly moving production line, taking one product every 10 minutes for a certain index test is stratified sampling."
  let P2 := "The stronger the linear correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."
  let P3 := "In a certain measurement, the measurement result ξ follows a normal distribution N(1, σ^2) (σ > 0). If the probability of ξ taking values in (0,1) is 0.4, then the probability of ξ taking values in (0,2) is 0.8."
  let P4 := "For the observed value k of the chi-square variable K^2 of categorical variables X and Y, the smaller k is, the greater the certainty of judging that X and Y are related."
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) = 2 :=
by {
  -- formal proofs are skipped
  sorry
}

end number_of_true_propositions_is_2_l203_203068


namespace remainder_base12_2543_div_9_l203_203385

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l203_203385


namespace annual_interest_earned_l203_203304
noncomputable section

-- Define the total money
def total_money : ℝ := 3200

-- Define the first part of the investment
def P1 : ℝ := 800

-- Define the second part of the investment as total money minus the first part
def P2 : ℝ := total_money - P1

-- Define the interest rates for both parts
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Define the time period (in years)
def time_period : ℝ := 1

-- Define the interest earned from each part
def interest1 : ℝ := P1 * rate1 * time_period
def interest2 : ℝ := P2 * rate2 * time_period

-- The total interest earned from both investments
def total_interest : ℝ := interest1 + interest2

-- The proof statement
theorem annual_interest_earned : total_interest = 144 := by
  sorry

end annual_interest_earned_l203_203304


namespace tan_105_eq_neg2_sub_sqrt3_l203_203625

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203625


namespace maximum_value_expression_l203_203356

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end maximum_value_expression_l203_203356


namespace count_two_digit_primes_with_ones_digit_three_l203_203783

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203783


namespace tan_105_l203_203546

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203546


namespace fill_time_l203_203073

-- Definitions based on conditions:
def length : ℝ := 2
def width : ℝ := 0.6
def height : ℝ := 0.6
def volume : ℝ := length * width * height

def flow_rate_liters_per_minute : ℝ := 3
def flow_rate_cubic_meters_per_second : ℝ := (flow_rate_liters_per_minute * 10⁻³) / 60

-- Proof statement:
theorem fill_time :
  let t : ℝ := volume / flow_rate_cubic_meters_per_second
  in t = 14400 := by
  sorry

end fill_time_l203_203073


namespace at_least_one_angle_not_greater_than_60_l203_203296

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (hSum : A + B + C = 180) : false :=
by
  sorry

end at_least_one_angle_not_greater_than_60_l203_203296


namespace days_in_month_l203_203411

theorem days_in_month
  (monthly_production : ℕ)
  (production_per_half_hour : ℚ)
  (hours_per_day : ℕ)
  (daily_production : ℚ)
  (days_in_month : ℚ) :
  monthly_production = 8400 ∧
  production_per_half_hour = 6.25 ∧
  hours_per_day = 24 ∧
  daily_production = production_per_half_hour * 2 * hours_per_day ∧
  days_in_month = monthly_production / daily_production
  → days_in_month = 28 :=
by
  sorry

end days_in_month_l203_203411


namespace intersection_exists_intersection_exists_parallel_l203_203687

noncomputable def line_of_intersection (P : Point) (projection_axis : Line) (plane1 plane2 : Plane) : Line :=
sorry

theorem intersection_exists (P : Point) (projection_axis : Line) (plane1 plane2 : Plane)
  (h1 : plane1 ∋ projection_axis ∧ plane1 ∋ P)
  (h2 : True) :
  ∃ L : Line, L = line_of_intersection P projection_axis plane1 plane2 :=
sorry

theorem intersection_exists_parallel (P : Point) (projection_axis : Line) (plane1 plane2 : Plane)
  (h1 : plane1 ∋ projection_axis ∧ plane1 ∋ P)
  (h2 : plane2 ∥ second_projection_plane) :
  ∃ L : Line, L = line_of_intersection P projection_axis plane1 plane2 :=
sorry

end intersection_exists_intersection_exists_parallel_l203_203687


namespace school_students_sum_l203_203053

theorem school_students_sum (s : ℕ) (h1 : 180 ≤ s) (h2 : s ≤ 250)
  (h3 : (s - 1) % 8 = 0) : 
  s ∈ ((range (250 - 180 + 1)).filter (λ n, (180 + n - 1) % 8 = 0)) →
  ((range (250 - 180 + 1)).filter (λ n, (180 + n - 1) % 8 = 0)).sum (λ n, 180 + n) = 1953 :=
by
  sorry -- Proof skipped for brevity.

end school_students_sum_l203_203053


namespace triangle_intersection_product_l203_203706

theorem triangle_intersection_product 
  (A B C F D E : Type*)
  [Point A] [Point B] [Point C] [Point F] [Point D] [Point E]
  (l : Line)
  (h_inter1 : A, B ∈ l)
  (h_inter2 : B, C ∈ l)
  (h_inter3 : C, A ∈ l)
  (h1 : line.inter l (segment AB) = Some F)
  (h2 : line.inter l (segment BC) = Some D)
  (h3 : line.inter l (segment CA) = Some E) :
  (AF / FB) * (BD / DC) * (CE / EA) = -1 :=
sorry

end triangle_intersection_product_l203_203706


namespace heptagon_diagonals_l203_203039

theorem heptagon_diagonals : ∀ (n : ℕ), n = 7 → (n * (n - 3)) / 2 = 14 :=
by
  intro n h
  rw h
  norm_num
  sorry

end heptagon_diagonals_l203_203039


namespace distance_between_town_a_and_b_l203_203392

noncomputable def horizontal_distance : ℝ := 8 - 4
noncomputable def vertical_distance : ℝ := 5 + 15

theorem distance_between_town_a_and_b : 
    (horizontal_distance ^ 2 + vertical_distance ^ 2) = (20.4 ^ 2) :=
by
  -- Calculations
  sorry

end distance_between_town_a_and_b_l203_203392


namespace two_digit_primes_ending_in_3_eq_6_l203_203939

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203939


namespace sum_first_2001_terms_l203_203705

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 3 → a n = a (n - 1) - a (n - 2)

noncomputable def sum_upto (a : ℕ → ℤ) (n : ℕ) : ℤ :=
∑ i in finset.range n, a i

theorem sum_first_2001_terms
(a : ℕ → ℤ)
(h_seq : sequence a)
(h_sum_1492 : sum_upto a 1492 = 1985)
(h_sum_1985 : sum_upto a 1985 = 1492) :
sum_upto a 2001 = -13 :=
sorry

end sum_first_2001_terms_l203_203705


namespace cosine_sum_l203_203119

theorem cosine_sum :
  cos (Real.arccos (4 / 5) + Real.arcsin (1 / 2)) = (4 * Real.sqrt 3 - 3) / 10 :=
by
  sorry

end cosine_sum_l203_203119


namespace count_two_digit_primes_with_ones_3_l203_203882

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203882


namespace part1_part2_l203_203697

noncomputable def condition1 (a b c : ℝ) : Prop := 3^a = 4 ∧ 3^b = 5 ∧ 3^c = 8

theorem part1 (a b c : ℝ) (h : condition1 a b c) : 3^(b + c) = 40 := 
by sorry

noncomputable def condition2 (a b : ℝ) : Prop := 3^a = 4 ∧ 3^b = 5

theorem part2 (a b : ℝ) (h : condition2 a b) : 3^(2*a - 3*b) = 16 / 125 := 
by sorry

end part1_part2_l203_203697


namespace log_function_point_l203_203024

theorem log_function_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∃ x y : ℝ, (y = log a (x - 2) + 3) ∧ x = 3 ∧ y = 3 :=
sorry

end log_function_point_l203_203024


namespace student_score_5_hours_focused_l203_203431

def score (time : ℝ) (effectiveness : ℝ) : ℝ := time * effectiveness * 20

theorem student_score_5_hours_focused :
  score 5 1.2 = 100 :=
by
  calc
    score 5 1.2 
        = 5 * 1.2 * 20 : rfl
    ... = 120 : by norm_num
    ... = 100 : by simp

end student_score_5_hours_focused_l203_203431


namespace base12_division_remainder_l203_203387

theorem base12_division_remainder :
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3 in
  n % 9 = 8 :=
by
  let n := 2 * (12^3) + 5 * (12^2) + 4 * 12 + 3
  show n % 9 = 8
  sorry

end base12_division_remainder_l203_203387


namespace expression_value_is_241_l203_203009

noncomputable def expression_value : ℕ :=
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2

theorem expression_value_is_241 : expression_value = 241 := 
by
  sorry

end expression_value_is_241_l203_203009


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203474

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203474


namespace longest_chord_line_eq_l203_203124

/-- Prove that the longest chord intercepted by the circle x^2 + y^2 - 2x + 4y = 0 passes through the point (2,1) and lies on the line 3x - y - 5 = 0. -/
theorem longest_chord_line_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 2*x + 4*y = 0) →
    (3*x - y - 5 = 0) →
    ∃ p : ℝ × ℝ, p = (2, 1) :=
sorry

end longest_chord_line_eq_l203_203124


namespace find_y_eq_l203_203660

def op (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b))^2

theorem find_y_eq (y : ℝ) (h : op 7 y = 64) : y = 43 / 2 :=
by 
  sorry

end find_y_eq_l203_203660


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203905

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203905


namespace polygon_properties_l203_203726

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l203_203726


namespace two_digit_primes_ending_in_3_eq_6_l203_203930

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203930


namespace units_digit_base8_sum_l203_203131

theorem units_digit_base8_sum (a b : ℕ) (ha : a = 3*8 + 5) (hb : b = 4*8 + 7) : 
  let sum := a + b,
      sum_base8 := nat.toDigits 8 sum in
  sum_base8.head = 4 :=
by {
  -- conversion to base 10 and verification omitted
  sorry 
}

end units_digit_base8_sum_l203_203131


namespace count_two_digit_primes_with_ones_digit_3_l203_203814

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203814


namespace count_two_digit_primes_with_ones_3_l203_203872

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203872


namespace tan_105_l203_203464

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203464


namespace count_two_digit_primes_with_ones_digit_three_l203_203791

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203791


namespace range_of_a_l203_203101

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (sin x) ^ 2 + cos x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l203_203101


namespace triangle_area_l203_203232

theorem triangle_area (AB CD : ℝ) (h₁ : 0 < AB) (h₂ : 0 < CD) (h₃ : CD = 3 * AB) :
    let trapezoid_area := 18
    let triangle_ABC_area := trapezoid_area / 4
    triangle_ABC_area = 4.5 := by
  sorry

end triangle_area_l203_203232


namespace sum_of_first_2010_terms_is_1340_l203_203235

theorem sum_of_first_2010_terms_is_1340 (a : ℝ) (h1: a ≤ 1) (h2: a ≠ 0) :
  let b : ℕ → ℝ
    | 0       => 1
    | 1       => a
    | (n + 2) => |b (n + 1) - b n|
  in (b 0 + b 1 + b 2 = 2) →
     3 ≤ 2010 →
     ∀ n ≥ 3, b (n + 3) = b n →
     (∑ i in finset.range 2010, b i) = 1340 :=
by intros a h1 h2 b hS3 hP hPeriod;
   sorry

end sum_of_first_2010_terms_is_1340_l203_203235


namespace class_avg_grade_greater_than_4_l203_203315

theorem class_avg_grade_greater_than_4 {A B C : ℕ} (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (score_A : ℕ → ℝ) (score_B : ℕ → ℝ) (score_C : ℕ → ℝ)
  (h_avg_A : (∑ i in finset.range A, score_A i) / A < 4)
  (h_avg_B : (∑ i in finset.range B, score_B i) / B < 4)
  (h_avg_C : (∑ i in finset.range C, score_C i) / C < 4) :
  ∃ (N : ℕ) (score : ℕ → ℝ), (∑ i in finset.range N, score i) / N > 4 := 
by
  sorry

end class_avg_grade_greater_than_4_l203_203315


namespace radius_of_tangent_circle_l203_203408

-- Definitions for the problem
def isTangentToAxesAndHypotenuse (circle_radius : ℝ) :=
  ∃ (O : ℝ × ℝ) (r : ℝ), r = circle_radius ∧
    -- Circle is tangent to x-axis and y-axis
    O.1 = r ∧ O.2 = r ∧
    -- Circle is tangent to the hypotenuse of a 45-45-90 triangle with hypotenuse length 2
    ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧
    B = (1, 1) ∧
    C = (2, 0) ∧  -- Use coordinates assuming the hypotenuse's length is 2
    let d := λ (p q : ℝ × ℝ), (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 in
    d O (0, r) = r ^ 2 ∧ d O (r, 0) = r ^ 2 ∧
    ∃ (F : ℝ × ℝ), (d O F = r ^ 2) ∧ (d F C + d F B = 2) ∧ (F.1, F.2) ≠ O

theorem radius_of_tangent_circle : isTangentToAxesAndHypotenuse 1 :=
  sorry

end radius_of_tangent_circle_l203_203408


namespace tan_105_eq_neg2_sub_sqrt3_l203_203539

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203539


namespace union_M_N_l203_203184

def M : set ℝ := {y | ∃ x : ℝ, y = 1 - sin x}
def N : set ℝ := {y | ∃ x : ℝ, y = ln (2 - x)}

theorem union_M_N : M ∪ N = { y : ℝ | y ≤ 2 } := by
  sorry

end union_M_N_l203_203184


namespace divisible_by_5_l203_203156

theorem divisible_by_5 (x y : ℕ) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 5 ∣ x := sorry

end divisible_by_5_l203_203156


namespace num_two_digit_primes_with_ones_digit_3_l203_203946

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203946


namespace count_two_digit_primes_with_ones_digit_three_l203_203792

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203792


namespace stamp_problem_solution_l203_203120

theorem stamp_problem_solution : ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m ≥ 2 * n + 2 → ∃ a b : ℕ, m = n * a + (n + 2) * b) ∧ ∀ x : ℕ, 1 < x ∧ (∀ m : ℕ, m ≥ 2 * x + 2 → ∃ a b : ℕ, m = x * a + (x + 2) * b) → x ≥ 3 :=
by
  sorry

end stamp_problem_solution_l203_203120


namespace sin_alpha_l203_203700

def r (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem sin_alpha (P : ℝ × ℝ) (α : ℝ) (hP : P = (1, real.sqrt 3)) (r_eq : r 1 (real.sqrt 3) = 2) :
  real.sin α = real.sqrt 3 / 2 :=
sorry

end sin_alpha_l203_203700


namespace tan_add_tan_105_eq_l203_203637

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203637


namespace two_digit_primes_with_ones_digit_3_l203_203855

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203855


namespace standard_equation_of_parabola_l203_203172

-- Defining the problem's conditions
def directrix : ℝ := 1 / 2

-- Define the parabola with given conditions
noncomputable def parabola (p : ℝ) (y : ℝ) : ℝ := 
  if p = 1 then -2 * y else x

-- Lean 4 statement for the mathematically equivalent proof problem
theorem standard_equation_of_parabola : (parabola 1 y) = x^2 = -2*y :=
by
  sorry

end standard_equation_of_parabola_l203_203172


namespace max_segments_diameter_l203_203671

/-- Given 39 points where at most 72% are on the surface of a sphere, prove that 
    the maximum number of segments that can form diameters is 378. -/
theorem max_segments_diameter (n : ℕ) (p : ℝ) (max_surface_points surface_points : ℕ) :
  n = 39 → p ≤ 0.72 → 
  max_surface_points = (p * n).to_nat → 
  max_surface_points ≤ n → 
  surface_points = 28 → 
  (comb surface_points 2) = 378 :=
by 
  intros h1 h2 h3 h4 h5
  rw [←h5, ←h3, h1]
  rw comb_eq
  sorry

end max_segments_diameter_l203_203671


namespace range_f_l203_203157

noncomputable def f (x y : ℝ) : ℝ :=
(x + y) / (⌊x⌋₊ * ⌊y⌋₊ + ⌊x⌋₊ + ⌊y⌋₊ + 1)

theorem range_f (x y : ℝ) (hxy : x * y = 1) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  Set.range (λ (x y : ℝ), f x y) = ({1/2} : Set ℝ) ∪ Set.Ioc (5/6 : ℝ) (5/4) :=
sorry

end range_f_l203_203157


namespace tan_105_degree_l203_203564

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203564


namespace tan_105_eq_neg2_sub_sqrt3_l203_203572

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203572


namespace sum_a_n_up_to_1499_l203_203136

def a_n (n : ℕ) : ℕ :=
  if (n % 15 = 0) ∧ (n % 10 = 0) then 15
  else if (n % 10 = 0) ∧ (n % 9 = 0) then 10
  else if (n % 9 = 0) ∧ (n % 15 = 0) then 9
  else 0

theorem sum_a_n_up_to_1499 : ∑ n in Finset.range 1500, a_n n = 1192 := by
  sorry

end sum_a_n_up_to_1499_l203_203136


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203613

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203613


namespace count_two_digit_primes_ending_with_3_l203_203842

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203842


namespace crayons_given_to_mary_l203_203449

theorem crayons_given_to_mary :
  let pack_crayons := 21 in
  let locker_crayons := 36 in
  let bobby_crayons := locker_crayons / 2 in
  let total_crayons := pack_crayons + locker_crayons + bobby_crayons in
  (total_crayons * (1 / 3) = 25) := by
rfl

end crayons_given_to_mary_l203_203449


namespace total_viewing_time_amaya_l203_203064

/-- The total viewing time Amaya spent, including rewinding, was 170 minutes. -/
theorem total_viewing_time_amaya 
  (u1 u2 u3 u4 u5 r1 r2 r3 r4 : ℕ)
  (h1 : u1 = 35)
  (h2 : u2 = 45)
  (h3 : u3 = 25)
  (h4 : u4 = 15)
  (h5 : u5 = 20)
  (hr1 : r1 = 5)
  (hr2 : r2 = 7)
  (hr3 : r3 = 10)
  (hr4 : r4 = 8) :
  u1 + u2 + u3 + u4 + u5 + r1 + r2 + r3 + r4 = 170 :=
by
  sorry

end total_viewing_time_amaya_l203_203064


namespace prime_difference_fourth_powers_is_not_prime_l203_203083

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_difference_fourth_powers_is_not_prime (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) : 
  ¬ is_prime (p^4 - q^4) :=
sorry

end prime_difference_fourth_powers_is_not_prime_l203_203083


namespace absolute_value_inequality_solution_set_l203_203349

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l203_203349


namespace tan_105_degree_l203_203599

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203599


namespace coefficient_x2_in_expansion_l203_203327

def general_term (n r : ℕ) (x : ℝ) : ℝ := (-1)^r * choose n r * x^(n - 2*r)

theorem coefficient_x2_in_expansion :
  let term := general_term 6 2 x in
  term = 15 :=
by
  sorry

end coefficient_x2_in_expansion_l203_203327


namespace digit_for_multiple_of_six_l203_203319

theorem digit_for_multiple_of_six (d : ℕ) (h : d ∈ {2, 8}) :
  74630 + 10 * d + 2 \% 6 = 0 :=
sorry

end digit_for_multiple_of_six_l203_203319


namespace tan_105_l203_203466

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203466


namespace a_2021_gt_60_l203_203419

-- Define the sequence and its properties
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + 1 / a n

-- Formulate the proof problem
theorem a_2021_gt_60 (a : ℕ → ℝ)
  (h0 : a 0 = 1)
  (h_rec : ∀ n, a (n + 1) = a n + 1 / a n) :
  a 2021 > 60 :=
sorry

end a_2021_gt_60_l203_203419


namespace max_value_sqrt_expression_l203_203359

theorem max_value_sqrt_expression
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (4 : ℝ)) * (Real.sqrt (2 : ℝ)) ≤ sqrt (4 : ℝ) * sqrt (2 : ℝ ) :=
begin
  sorry,
end

end max_value_sqrt_expression_l203_203359


namespace widgets_per_shipping_box_l203_203668

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l203_203668


namespace tan_105_degree_l203_203590

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203590


namespace two_digit_primes_ending_in_3_eq_6_l203_203927

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203927


namespace count_two_digit_primes_with_ones_digit_3_l203_203810

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203810


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203986

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203986


namespace sum_of_intersections_l203_203698

-- Defining the function and the conditions for the problem
noncomputable def f (x : ℝ) : ℝ :=
sorry -- We'll abstract over the specific definition as it arises from solution steps

theorem sum_of_intersections (f : ℝ → ℝ)
  (H1 : ∀ x : ℝ, f (2 - x) = 2 - f x)  -- Condition from the problem
  (H2 : ∃ (XI : list ℝ), ∀ xi ∈ XI, f xi = xi / (xi - 1))  -- Definition of x_i intersections
  (n : ℕ) (h : n = (H2.some.length)) :  -- Assuming that H2.some represents the list of intersections
  (∑ xi in H2.some, xi + (f xi) = 2 * n) :=
sorry  -- Proof to be filled in

end sum_of_intersections_l203_203698


namespace tan_105_eq_neg2_sub_sqrt3_l203_203626

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203626


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203981

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203981


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203487

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203487


namespace weight_of_10_moles_approx_l203_203201

def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def atomic_mass_O : ℝ := 16.00

def molar_mass_C6H8O6 : ℝ := 
  (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)

def moles : ℝ := 10
def given_total_weight : ℝ := 1760

theorem weight_of_10_moles_approx (ε : ℝ) (hε : ε > 0) :
  abs ((moles * molar_mass_C6H8O6) - given_total_weight) < ε := by
  -- proof will go here.
  sorry

end weight_of_10_moles_approx_l203_203201


namespace count_two_digit_primes_with_ones_3_l203_203885

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203885


namespace two_digit_primes_end_in_3_l203_203906

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203906


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l203_203999

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l203_203999


namespace angle_of_inclination_range_l203_203746

theorem angle_of_inclination_range (a : ℝ) :
  (∃ m : ℝ, ax + (a + 1)*m + 2 = 0 ∧ (m < 0 ∨ m > 1)) ↔ (a < -1/2 ∨ a > 0) := sorry

end angle_of_inclination_range_l203_203746


namespace identify_minor_premise_l203_203071

-- Define the conditions as assumptions
variable (ship_depart_on_time arrive_at_destport_on_time : Prop)
variable (h1 : arrive_at_destport_on_time ↔ ship_depart_on_time)
variable (h2 : arrive_at_destport_on_time)
variable (h3 : ship_depart_on_time)

-- The statement to prove
theorem identify_minor_premise : h3 =
by { sorry }

end identify_minor_premise_l203_203071


namespace function_decreasing_and_extrema_l203_203298

open Set

def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem function_decreasing_and_extrema : 
  (∀ x₁ x₂, 3 ≤ x₁ → x₁ < x₂ → x₂ ≤ 5 → f x₁ > f x₂) ∧
  (f 3 = 3 / 4) ∧
  (f 5 = 1 / 2) := 
by
  sorry

end function_decreasing_and_extrema_l203_203298


namespace angle_AED_eq_66_l203_203076

-- Definitions and conditions based on the problem statement
variables {A B C D F E : Type} [EuclideanGeometry A B C D F E]
open EuclideanGeometry

def ABCD_is_parallelogram (ABCD: Parallelogram A B C D) : Prop := ABCD.IsParallelogram
def angle_ABC (∠ABC : Angle A B C) : Prop := ∠ABC = 72
def AF_perp_BC (AF_perp : Perpendicular A F B C) : Prop := AF_perp IsPerpendicular
def AF_intersects_BD_at_E (intersection : Intersects A F B D E) : Prop := intersection DoesIntersect
def DE_eq_2AB (length : Length D E) (length2AB : Length A B) : Prop := length = 2 * length2AB

-- Question in Lean format as a theorem we need to prove
theorem angle_AED_eq_66 
  (p : Parallelogram A B C D) 
  (h1 : ∠ABC = 72) 
  (h2 : Perpendicular A F B C) 
  (h3 : Intersects A F B D E)
  (h4 : Length D E = 2 * Length A B) :
  ∠AED = 66 :=
sorry

end angle_AED_eq_66_l203_203076


namespace other_questions_points_l203_203013

theorem other_questions_points :
  ∀ (total_points total_questions two_point_question_count two_point_question_value : ℕ)
  (remaining_questions_points : ℕ),
  total_points = 100 →
  total_questions = 40 →
  two_point_question_count = 30 →
  two_point_question_value = 2 →
  remaining_questions_points = 
    (total_points - (two_point_question_count * two_point_question_value)) →
  (total_questions - two_point_question_count) ≠ 0 →
  (remaining_questions_points / (total_questions - two_point_question_count)) = 4 := 
by
  intros total_points total_questions two_point_question_count two_point_question_value remaining_questions_points
  intros h1 h2 h3 h4 h5 h6
  have h7: remaining_questions_points = 40 := by
    rw [h5, h1, h3, h4]
    exact congr_arg (λ x, 100 - x) (nat.mul_comm (two_point_question_count) (two_point_question_value))
  rw h7
  have h8: 10 ≠ 0 := by 
    norm_num
  have h9: (total_questions - two_point_question_count) = 10 := by
    rw [h2, h3]
  rw h9 at h6
  contradiction
  sorry

end other_questions_points_l203_203013


namespace find_t_l203_203191

theorem find_t (t : ℝ) :
  let m := (Real.sqrt 3, 1 : ℝ × ℝ)
      n := (0, -1 : ℝ × ℝ)
      k := (t, Real.sqrt 3 : ℝ × ℝ) in
  ∃ λ : ℝ, λ • (m.1, m.2) - 2 • (n.1, n.2) = k → t = 1 :=
by
  let m := (Real.sqrt 3, 1 : ℝ × ℝ)
  let n := (0, -1 : ℝ × ℝ)
  let k := (t, Real.sqrt 3 : ℝ × ℝ)
  let collinear := ∃ λ : ℝ, λ • (m.1 - 2 * n.1, m.2 - 2 * n.2) = k
  exact collinear → t = 1
  sorry

end find_t_l203_203191


namespace trigonometric_values_of_x_l203_203133

theorem trigonometric_values_of_x:
  ∃ x : ℝ, 
  (sin x = -3/5) ∧ 
  (0 < x) ∧ 
  (x < 3 * real.pi / 2) ∧ 
  (cos x = -4/5) ∧ 
  (tan x = 3/4) ∧ 
  (cot x = 4/3) :=
sorry

end trigonometric_values_of_x_l203_203133


namespace two_digit_primes_with_ones_digit_3_l203_203869

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203869


namespace line_intersects_circle_isosceles_right_triangle_l203_203720

-- Define the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line
def line (x y a : ℝ) : Prop := x - y + a = 0

-- Define the origin
def origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Define the condition for an isosceles right triangle at the origin
def is_isosceles_right_triangle (A B : (ℝ × ℝ)) (O : (ℝ × ℝ)) : Prop :=
  let (ax, ay) := A in
  let (bx, by) := B in
  let (ox, oy) := O in
  (ox = 0 ∧ oy = 0) ∧
  ((ax - ox)^2 + (ay - oy)^2 = (bx - ox)^2 + (by - oy)^2 ∧
   (ax - ox)^2 + (ay - oy)^2 + (ax - bx)^2 + (ay - by)^2 = 2 * ((ax - ox)^2 + (ay - oy)^2))

-- Proven statement: If the line intersects the circle at A and B, then under the given conditions a = ±√2.
theorem line_intersects_circle_isosceles_right_triangle {a : ℝ} :
  (∃ A B : (ℝ × ℝ), line A.1 A.2 a ∧ line B.1 B.2 a ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧ 
   is_isosceles_right_triangle A B (0, 0)) →
  a = sqrt 2 ∨ a = -sqrt 2 :=
by
  sorry

end line_intersects_circle_isosceles_right_triangle_l203_203720


namespace num_students_section2_l203_203050

-- Definitions of given conditions
def num_students_section1 := 65
def num_students_section3 := 45
def num_students_section4 := 42

def mean_marks_section1 := 50
def mean_marks_section2 := 60
def mean_marks_section3 := 55
def mean_marks_section4 := 45

def overall_average_marks := 51.95

-- The goal is to find the number of students in section 2
theorem num_students_section2 : 
  (∀ x : ℕ, ((num_students_section1 * mean_marks_section1 + x * mean_marks_section2 + 
    num_students_section3 * mean_marks_section3 + 
    num_students_section4 * mean_marks_section4) / 
    (num_students_section1 + x + num_students_section3 + 
    num_students_section4)) = overall_average_marks → x = 35) := 
by sorry

end num_students_section2_l203_203050


namespace farey_sequence_consecutive_fractions_l203_203021

theorem farey_sequence_consecutive_fractions
  (n a b c d : ℕ)
  (h_ab_coprime : Nat.gcd a b = 1)
  (h_cd_coprime : Nat.gcd c d = 1)
  (h_consecutive : ¬ ∃ (x y : ℕ), (0 < y ∧ y ≤ n ∧ Nat.gcd x y = 1) ∧ (a * y < b * x ∧ x * d < c * y)) :
  | b * c - a * d | = 1 :=
sorry

end farey_sequence_consecutive_fractions_l203_203021


namespace two_digit_primes_end_in_3_l203_203923

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203923


namespace A_plays_D_third_day_l203_203696

section GoTournament

variables (Player : Type) (A B C D : Player) 

-- Define the condition that each player competes with every other player exactly once.
def each_plays_once (P : Player → Player → Prop) : Prop :=
  ∀ x y, x ≠ y → (P x y ∨ P y x)

-- Define the tournament setup and the play conditions.
variables (P : Player → Player → Prop)
variable [∀ x y, Decidable (P x y)] -- Assuming decidability for the play relation

-- The given conditions of the problem
axiom A_plays_C_first_day : P A C
axiom C_plays_D_second_day : P C D
axiom only_one_match_per_day : ∀ x, ∃! y, P x y

-- We aim to prove that A will play against D on the third day.
theorem A_plays_D_third_day : P A D :=
sorry

end GoTournament

end A_plays_D_third_day_l203_203696


namespace area_AEL_l203_203224

section ProofProblem

variables {A B C L M D E : Type*}
variables [point A] [point B] [point C] [point L] [point M] [point D] [point E]

-- Right triangle with a right angle at B
axiom right_triangle_ABC (hABC : triangle A B C) (hB : right_angle B A C)

-- Bisector BL and median CM, intersecting at D
axiom bisector_BL (hBL : bisector B L A C)
axiom median_CM (hCM : median C M A B)
axiom intersect_D (hIntD : intersection B L C M = D)

-- Line AD intersects BC at point E
axiom line_AD_intersect_E (hADE : intersection A D B C = E)

-- Given length EL = x
parameter (x : ℝ)
axiom length_EL_x (hELx : length E L = x)

-- Prove that the area of triangle AEL is x^2 / 2
theorem area_AEL : area (triangle A E L) = x^2 / 2 := by
  sorry

end ProofProblem

end area_AEL_l203_203224


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203988

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203988


namespace cos_sufficient_sin_l203_203163

theorem cos_sufficient_sin (A : ℝ) (hA : 0 < A ∧ A < π) :
  ((cos A = 1 / 2) → (sin A = sqrt 3 / 2)) ∧
  ((sin A = sqrt 3 / 2) → (cos A = 1 / 2 ∨ cos A = -1 / 2)) :=
by
  sorry

end cos_sufficient_sin_l203_203163


namespace problem_statement_l203_203176

noncomputable theory

def xy_is_perfect_square (x y : ℕ) : Prop :=
  (x * y + 4) = (x + 2) * (x + 2)

theorem problem_statement (x y : ℕ) (h : x > 0 ∧ y > 0) : 
  (1/x + 1/y + 1/(x * y) = 1/(x + 4) + 1/(y - 4) + 1/((x + 4) * (y - 4))) → xy_is_perfect_square x y :=
by
  sorry

end problem_statement_l203_203176


namespace jason_borrowed_amount_l203_203249

theorem jason_borrowed_amount :
  let cycle := [1, 3, 5, 7, 9, 11]
  let total_chores := 48
  let chores_per_cycle := cycle.length
  let earnings_one_cycle := cycle.sum
  let complete_cycles := total_chores / chores_per_cycle
  let total_earnings := complete_cycles * earnings_one_cycle
  total_earnings = 288 :=
by
  sorry

end jason_borrowed_amount_l203_203249


namespace count_two_digit_primes_ending_in_3_l203_203817

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203817


namespace count_two_digit_primes_with_ones_3_l203_203871

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203871


namespace union_A_B_inter_A_B_compl_A_l203_203263

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
def B : Set ℝ := { x | x ≥ 3 }

theorem union_A_B : A ∪ B = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem inter_A_B : A ∩ B = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry

theorem compl_A : Aᶜ = { x : ℝ | x < 2 ∨ x ≥ 4 } :=
by
  sorry

end union_A_B_inter_A_B_compl_A_l203_203263


namespace count_two_digit_primes_with_ones_digit_three_l203_203793

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203793


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203502

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203502


namespace geologists_separation_probability_l203_203366

noncomputable def geologist_probability (n_roads : ℕ) (speed : ℕ) (distance_threshold : ℝ) : ℚ :=
  let possible_distances (n : ℕ) : list ℝ :=
    [0, speed, speed * real.sqrt 3, 2 * speed] -- correspond to the distances based on angles 0°, 60°, 120°, 180°
  let road_pairs (n : ℕ) : list (ℝ × ℝ) :=
    (list.range n).bind (λ r1, (list.range n).map (λ r2, (r1, r2)))
  let favorable_pairs : list (ℝ × ℝ) :=
    road_pairs n |>.filter (λ (p : ℝ × ℝ), prod.fst p ≠ prod.snd p ∧ (possible_distances n).nth_le (abs (prod.fst p - prod.snd p)) 3 ≥ distance_threshold)
  nat.card favorable_pairs /. nat.card road_pairs n

theorem geologists_separation_probability :
  geologist_probability 6 4 6 = 0.5 := sorry

end geologists_separation_probability_l203_203366


namespace math_problem_l203_203317

theorem math_problem : 
  let result := (555.55 - 111.11) * 2 in 
  result = 888.88 :=
by {
  sorry
}

end math_problem_l203_203317


namespace haley_collected_cans_l203_203196

theorem haley_collected_cans :
  ∃ n : ℕ, 9 = 7 * n + 2 :=
by { use 1, norm_num }

end haley_collected_cans_l203_203196


namespace count_two_digit_primes_ending_with_3_l203_203836

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203836


namespace count_two_digit_primes_with_ones_digit_three_l203_203790

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203790


namespace polygon_properties_l203_203725

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l203_203725


namespace polynomials_have_two_roots_l203_203089

noncomputable def countPolynomialsWithTwoRoots
  (polynomial_form : ℕ → ℕ → Bool)
  (integer_partition : ℕ → ℕ → ℕ): ℕ :=
  sorry

theorem polynomials_have_two_roots :
  countPolynomialsWithTwoRoots
    (λ i n => (i ≤ 8) ∧ (i ∈ {0, 1}) ∧ (n = 9) ∧ (cf i = 9, cf i a = 0, cf i a) = 56 :=
    sorry

end polynomials_have_two_roots_l203_203089


namespace tan_105_degree_l203_203592

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203592


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203989

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203989


namespace tan_105_l203_203501

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203501


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203994

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203994


namespace smallest_integer_discussed_l203_203038

theorem smallest_integer_discussed:
  ∃ N : ℕ,
    ∀ i : ℕ, (1 ≤ i ∧ i ≤ 30 ∧ i ≠ 17 ∧ i ≠ 19 → i ∣ N) →
    ¬17 ∣ N ∧ ¬19 ∣ N ∧ N = 122522400 :=
begin
  sorry
end

end smallest_integer_discussed_l203_203038


namespace combined_total_cost_l203_203292

theorem combined_total_cost :
  ∀ (burger_cost soda_cost: ℝ),
    burger_cost = 6 →
    soda_cost = (1/3) * burger_cost →
    let paulo_total := burger_cost + soda_cost in
    let jeremy_total := 2 * (burger_cost + soda_cost) in
    paulo_total + jeremy_total = 24 :=
by
  intros burger_cost soda_cost hburger hsoda
  let paulo_total := burger_cost + soda_cost
  let jeremy_total := 2 * (burger_cost + soda_cost)
  calc
    paulo_total + jeremy_total = (burger_cost + soda_cost) + 2 * (burger_cost + soda_cost) : by rw [paulo_total, jeremy_total]
    ... = 3 * (burger_cost + soda_cost) : by ring
    ... = 3 * (6 + 2) : by rw [hburger, hsoda]; ring
    ... = 24 : by norm_num

end combined_total_cost_l203_203292


namespace max_value_sqrt_expression_l203_203358

theorem max_value_sqrt_expression
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (4 : ℝ)) * (Real.sqrt (2 : ℝ)) ≤ sqrt (4 : ℝ) * sqrt (2 : ℝ ) :=
begin
  sorry,
end

end max_value_sqrt_expression_l203_203358


namespace no_three_distinct_solutions_l203_203208

theorem no_three_distinct_solutions :
  ∀ (a b c : ℝ), (a * (a - 6) = 7) ∧ (b * (b - 6) = 7) ∧ (c * (c - 6) = 7) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) →
  false :=
by
  intro a b c
  intro h
  have ha : a = 7 ∨ a = -1 := sorry
  have hb : b = 7 ∨ b = -1 := sorry
  have hc : c = 7 ∨ c = -1 := sorry
  cases ha with ha1 ha2;
  cases hb with hb1 hb2;
  cases hc with hc1 hc2;
  { contradiction },
  sorry

end no_three_distinct_solutions_l203_203208


namespace tan_105_eq_neg2_sub_sqrt3_l203_203581

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203581


namespace intersection_MN_EF_on_BC_l203_203047

open Set

variables {k : Type*} [Field k]

-- Definitions for the cyclic quadrilateral and its properties
variables (S : k)
variables (A B C D X M N E F K : k)
variables (circle S)
variables (is_cyclic_quad : is_cyclic_quad A B C D S)

-- Definitions of points and intersections
variables (second_intersection_M : second_intersection_circle X A S M)
variables (second_intersection_N : second_intersection_circle X D S N)
variables (intersection_E : intersections (line DC) (line AX) E)
variables (intersection_F : intersections (line AB) (line DX) F)
variables (intersection_K : intersections (line MN) (line EF) K)

-- Theorem
theorem intersection_MN_EF_on_BC :
  ∃ K : k, intersections (line MN) (line EF) K ∧ on_line BC K := 
sorry

end intersection_MN_EF_on_BC_l203_203047


namespace functional_equation_solution_l203_203314

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * real.cos y) : 
  ∀ t : ℝ, f t = f 0 * real.cos t + f (real.pi / 2) * real.sin t :=
sorry

end functional_equation_solution_l203_203314


namespace angle_with_same_terminal_side_l203_203065

-- Given conditions in the problem: angles to choose from
def angles : List ℕ := [60, 70, 100, 130]

-- Definition of the equivalence relation (angles having the same terminal side)
def same_terminal_side (θ α : ℕ) : Prop :=
  ∃ k : ℤ, θ = α + k * 360

-- Proof goal: 420° has the same terminal side as one of the angles in the list
theorem angle_with_same_terminal_side :
  ∃ α ∈ angles, same_terminal_side 420 α :=
sorry  -- proof not required

end angle_with_same_terminal_side_l203_203065


namespace number_of_x_intersections_l203_203096

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x, f x = f (-x)
axiom f_continuous : continuous f
axiom f_monotonic_inc : ∀ {x y}, 0 ≤ x → x ≤ y → f x ≤ f y
axiom f_product_neg : f 1 * f 2 < 0

theorem number_of_x_intersections : 
  ∃ a b, a < b ∧ f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 ↔ x = a ∨ x = b) ∧ (∃! x, x ∈ Ioo 1 2 ∧ f x = 0) ∧ (∃! x, x < 0 ∧ f x = 0) :=
  sorry

end number_of_x_intersections_l203_203096


namespace tan_105_eq_neg2_sub_sqrt3_l203_203519

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203519


namespace count_two_digit_primes_with_ones_digit_three_l203_203787

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203787


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203982

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203982


namespace regular_polygon_properties_l203_203723

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l203_203723


namespace two_digit_primes_ending_in_3_eq_6_l203_203924

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203924


namespace sum_fractions_l203_203117

def partialSum (n : ℕ) : ℚ := ∑ k in Finset.range (n - 1), 1 / (k + 2) / (k + 3)

theorem sum_fractions : partialSum 10 = 9/22 :=
by sorry

end sum_fractions_l203_203117


namespace compute_p_plus_q_l203_203274

theorem compute_p_plus_q {p q : ℝ}
  (hp : p^3 - 18 * p^2 + 27 * p - 72 = 0)
  (hq : 10 * q^3 - 75 * q^2 + 50 * q - 625 = 0) :
  p + q = 2 * real.cbrt 180 + 43 / 3 :=
by 
  sorry

end compute_p_plus_q_l203_203274


namespace tan_105_l203_203552

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203552


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203509

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203509


namespace two_digit_primes_ending_in_3_eq_6_l203_203932

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203932


namespace two_digit_primes_with_ones_digit_3_l203_203863

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203863


namespace count_two_digit_primes_ending_in_3_l203_203816

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203816


namespace count_two_digit_primes_with_ones_digit_three_l203_203796

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203796


namespace combined_rate_l203_203112

theorem combined_rate
  (earl_rate : ℕ)
  (ellen_time : ℚ)
  (total_envelopes : ℕ)
  (total_time : ℕ)
  (combined_total_envelopes : ℕ)
  (combined_total_time : ℕ) :
  earl_rate = 36 →
  ellen_time = 1.5 →
  total_envelopes = 36 →
  total_time = 1 →
  combined_total_envelopes = 180 →
  combined_total_time = 3 →
  (earl_rate + (total_envelopes / ellen_time)) = 60 :=
by
  sorry

end combined_rate_l203_203112


namespace tan_105_eq_neg2_sub_sqrt3_l203_203543

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203543


namespace num_two_digit_primes_with_ones_digit_3_l203_203951

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203951


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203478

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203478


namespace SummitAcademy_Contestants_l203_203078

theorem SummitAcademy_Contestants (s j : ℕ)
  (h1 : s > 0)
  (h2 : j > 0)
  (hs : (1 / 3 : ℚ) * s = (3 / 4 : ℚ) * j) :
  s = (9 / 4 : ℚ) * j :=
sorry

end SummitAcademy_Contestants_l203_203078


namespace smallest_four_digit_number_l203_203037

theorem smallest_four_digit_number : 
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∃ (AB CD : ℕ), 
      n = 1000 * (AB / 10) + 100 * (AB % 10) + CD ∧
      ((AB / 10) * 10 + (AB % 10) + 2) * CD = 100 ∧ 
      n / CD = ((AB / 10) * 10 + (AB % 10) + 1)^2) ∧
    n = 1805 :=
by
  sorry

end smallest_four_digit_number_l203_203037


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203900

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203900


namespace two_digit_primes_end_in_3_l203_203920

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203920


namespace distinct_scores_l203_203403

theorem distinct_scores : 
  (∃ unique_scores : Set ℕ, unique_scores = {P | ∃ x, x ∈ Finset.range 9 ∧ P = 2 * x + 8} ∧ unique_scores.card = 9) :=
by
  sorry

end distinct_scores_l203_203403


namespace count_two_digit_primes_ending_in_3_l203_203819

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203819


namespace annie_gives_mary_25_crayons_l203_203446

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l203_203446


namespace earnings_per_cow_is_800_l203_203301

variable (num_cows : ℕ) (pigs_per_cow : ℕ) (total_earning : ℕ) (earnings_per_pig : ℕ)

-- Given conditions
def condition1 := num_cows = 20
def condition2 := pigs_per_cow = 4
def condition3 := total_earning = 48000
def condition4 := earnings_per_pig = 400

-- The statement we want to prove
theorem earnings_per_cow_is_800 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  ((total_earning - (num_cows * pigs_per_cow) * earnings_per_pig) / num_cows) = 800 :=
by
  sorry

end earnings_per_cow_is_800_l203_203301


namespace two_digit_primes_ending_in_3_eq_6_l203_203935

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203935


namespace gn_divides_gnplus1_l203_203702

theorem gn_divides_gnplus1 (g : ℕ → ℕ)
    (h_start : g 1 = 1)
    (h_recur : ∀ n, g (n + 1) = g n ^ 2 + g n + 1) :
    ∀ n, (g n ^ 2 + 1) ∣ (g (n + 1) ^ 2 + 1) :=
begin
  -- sorry, proof skipped
  sorry
end

end gn_divides_gnplus1_l203_203702


namespace count_two_digit_primes_with_ones_digit_three_l203_203781

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203781


namespace total_flowers_l203_203226

-- Definitions of the conditions
variables {numGreen: ℕ} {numRed: ℕ} {numYellow: ℕ}
variables {numBlue: ℕ} {total: ℕ}

-- Conditions
def condition1 := numGreen = 9
def condition2 := numRed = 3 * numGreen
def condition3 := numBlue = (1 / 2) * total
def condition4 := numYellow = 12

-- Theorem to prove
theorem total_flowers (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : total = 96 :=
sorry

end total_flowers_l203_203226


namespace tied_in_runs_l203_203326

def aaron_runs : List ℕ := [4, 8, 15, 7, 4, 12, 11, 5]
def bonds_runs : List ℕ := [3, 5, 18, 9, 12, 14, 9, 0]

def total_runs (runs : List ℕ) : ℕ := runs.foldl (· + ·) 0

theorem tied_in_runs : total_runs aaron_runs = total_runs bonds_runs := by
  sorry

end tied_in_runs_l203_203326


namespace quadratic_roots_ratio_l203_203276

theorem quadratic_roots_ratio (r1 r2 p q n : ℝ) (h1 : p = r1 * r2) (h2 : q = -(r1 + r2)) (h3 : p ≠ 0) (h4 : q ≠ 0) (h5 : n ≠ 0) (h6 : r1 ≠ 0) (h7 : r2 ≠ 0) (h8 : x^2 + q * x + p = 0) (h9 : x^2 + p * x + n = 0) :
  n / q = -3 :=
by
  sorry

end quadratic_roots_ratio_l203_203276


namespace hyperbola_center_l203_203123

theorem hyperbola_center :
  ∃ (h : ℝ × ℝ), h = (9 / 2, 2) ∧
  (∃ (x y : ℝ), 9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0) :=
  sorry

end hyperbola_center_l203_203123


namespace find_perpendicular_line_l203_203701

-- Define a line in terms of coefficients a, b, c with equation ax + by + c = 0
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Given point P
def P : ℝ × ℝ := (4, -1)

-- Given line equation 3x - 4y + 6 = 0
def line1 : Line := {a := 3, b := -4, c := 6}

-- Condition for perpendicularity: a1 * a2 + b1 * b2 = 0
def is_perpendicular (l1 l2 : Line) : Prop := l1.a * l2.a + l1.b * l2.b = 0

-- Perpendicular line through point P(4, -1), initial form: 4x + 3y + c = 0
def perpendicular_line_to (l : Line) (P : ℝ × ℝ) : Line :=
  let (x, y) := P in
  { a := 4, b := 3, c := - (4 * x + 3 * y) }

-- Define the line perpendicular to the given line and passing through P
def line2 : Line := perpendicular_line_to line1 P

-- Prove the equation of the line passing through P and is perpendicular to line1
theorem find_perpendicular_line :
  is_perpendicular line1 line2 ∧ line2.a * P.fst + line2.b * P.snd + line2.c = 0 :=
by
  sorry

end find_perpendicular_line_l203_203701


namespace two_digit_primes_with_ones_digit_three_count_l203_203773

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203773


namespace tan_105_degree_l203_203570

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203570


namespace two_digit_primes_end_in_3_l203_203909

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203909


namespace hyperbola_focal_length_l203_203337

theorem hyperbola_focal_length (m : ℝ) 
  (h0 : (∀ x y, x^2 / 16 - y^2 / m = 1)) 
  (h1 : (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5)) : 
  m = 4 := 
by sorry

end hyperbola_focal_length_l203_203337


namespace q1_q2_q3_l203_203743

noncomputable def quadratic_function (a x: ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem q1 (a : ℝ) : (∀ {x : ℝ}, quadratic_function a x = 0 → x < 2) ∧ (quadratic_function a 2 > 0) ∧ (2 * a ≠ 0) → a < -1 := 
by 
  sorry

theorem q2 (a : ℝ) : (∀ x : ℝ, quadratic_function a x ≥ -1 - a * x) → -2 ≤ a ∧ a ≤ 6 := 
by 
  sorry
  
theorem q3 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → quadratic_function a x ≤ 4) → a = 2 ∨ a = 2 / 3 := 
by 
  sorry

end q1_q2_q3_l203_203743


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203983

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203983


namespace tan_add_tan_105_eq_l203_203634

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203634


namespace count_two_digit_primes_with_ones_digit_three_l203_203784

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203784


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203898

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203898


namespace min_value_of_f_on_interval_l203_203734

noncomputable def f (x a : ℝ) : ℝ :=
  -x^3 + 3 * x^2 + 9 * x + a

theorem min_value_of_f_on_interval :
  ∃ a : ℝ, (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x a ≤ f 2 (-2)) ∧ (f (-1) (-2) = -7) := 
by
  sorry

end min_value_of_f_on_interval_l203_203734


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203992

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203992


namespace yang_hui_rect_eq_l203_203321

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l203_203321


namespace sum_of_digits_base2_345_l203_203007

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end sum_of_digits_base2_345_l203_203007


namespace tan_105_degree_l203_203655

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203655


namespace operation_to_reduce_to_one_from_64_l203_203428

def isPowerOfTwo (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

noncomputable def operation (n : Nat) : Nat :=
  let powers := List.range n |>.filter isPowerOfTwo
  n - powers.length

def numberOfOperations (start : Nat) (target : Nat) : Nat :=
  if h : start < target then 0
  else
    let rec loop (count : Nat) (current : Nat) : Nat :=
      if current = target then count
      else loop (count + 1) (operation current)
    loop 0 start

theorem operation_to_reduce_to_one_from_64 :
  numberOfOperations 64 1 = 6 :=
sorry

end operation_to_reduce_to_one_from_64_l203_203428


namespace num_two_digit_primes_with_ones_digit_3_l203_203949

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203949


namespace two_digit_primes_with_ones_digit_3_l203_203856

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203856


namespace two_digit_primes_end_in_3_l203_203908

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203908


namespace count_two_digit_primes_ending_with_3_l203_203850

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203850


namespace sqrt_one_over_four_eq_pm_half_l203_203004

theorem sqrt_one_over_four_eq_pm_half : Real.sqrt (1 / 4) = 1 / 2 ∨ Real.sqrt (1 / 4) = - (1 / 2) := by
  sorry

end sqrt_one_over_four_eq_pm_half_l203_203004


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203484

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203484


namespace functional_equation_solution_l203_203312

open Real

theorem functional_equation_solution (f : ℝ → ℝ) (c : ℝ) 
  (h_pos : ∀ x, 0 < x → 0 < f(x))
  (h_eq : ∀ x y, 0 < x → 0 < y →
    f(x) = f(f(f(x)) + y) + f(x * f(y)) * f(x + y)) :
  ∃ c, c > 0 ∧ ∀ x, 0 < x → f(x) = c / x :=
by
  sorry

end functional_equation_solution_l203_203312


namespace tan_105_eq_neg2_sub_sqrt3_l203_203541

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203541


namespace num_two_digit_primes_with_ones_digit_3_l203_203953

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203953


namespace tan_105_l203_203556

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203556


namespace two_digit_primes_with_ones_digit_three_count_l203_203771

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203771


namespace negation_of_proposition_p_l203_203750

-- Define the proposition p
def proposition_p := ∀ x : ℝ, cos x ≤ 1

-- Define the negation of the proposition
def negation_p := ∃ x : ℝ, cos x > 1

-- State the theorem
theorem negation_of_proposition_p : ¬ proposition_p ↔ negation_p :=
by
  -- Proof goes here
  sorry

end negation_of_proposition_p_l203_203750


namespace number_of_correct_propositions_l203_203731

-- Define the conditions
variables (plane1 plane2 : Type) [Plane plane1] [Plane plane2]

-- Define the propositions
def proposition1 (l : Line plane1) (L : Line plane2) : Prop := ¬((line_in_plane plane1 l) ∧ (line_in_plane plane2 L) → perpendicular l L)
def proposition2 (l : Line plane1) : Prop := ∃ L : Line plane2, (line_in_plane plane1 l) ∧ (line_in_plane plane2 L) ∧ (perpendicular l L)
def proposition3 : Prop := ∀ l : Line plane1, perpendicular l plane2
def proposition4 (l : Line plane1) : Prop := ∃ L : Line plane2, (parallel (line_intersection plane1 plane2) L)

-- Prove that there are exactly two correct propositions
theorem number_of_correct_propositions : 2 = 
(ite proposition1 1 0) + (ite proposition2 1 0) + (ite proposition3 1 0) + (ite proposition4 1 0) :=
sorry

end number_of_correct_propositions_l203_203731


namespace two_digit_primes_with_ones_digit_3_l203_203859

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203859


namespace count_two_digit_primes_ending_in_3_l203_203824

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203824


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203991

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203991


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203961

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203961


namespace two_digit_primes_ending_in_3_eq_6_l203_203929

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203929


namespace total_spent_l203_203259

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end total_spent_l203_203259


namespace tan_105_eq_neg2_sub_sqrt3_l203_203527

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203527


namespace part1_l203_203027

theorem part1 (n : ℕ) (m : ℕ) (h_form : m = 2 ^ (n - 2) * 5 ^ n) (h : 6 * 10 ^ n + m = 25 * m) :
  ∃ k : ℕ, 6 * 10 ^ n + m = 625 * 10 ^ (n - 2) :=
by
  sorry

end part1_l203_203027


namespace linear_function_point_l203_203149

theorem linear_function_point (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 :=
by
  sorry

end linear_function_point_l203_203149


namespace exists_polynomial_satisfying_PxPx1_eq_Px2_no_polynomial_satisfies_PxPx1_eq_Px2_plus_1_l203_203666

open Real Polynomial

-- Part (a)
theorem exists_polynomial_satisfying_PxPx1_eq_Px2 :
  ∃ P : Polynomial ℝ, ∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2) := sorry

-- Part (b)
theorem no_polynomial_satisfies_PxPx1_eq_Px2_plus_1 :
  ∀ n : ℕ, ¬∃ P : Polynomial ℝ, P.natDegree = n ∧ (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 + 1)) ∧
         P.roots.card = n := sorry

end exists_polynomial_satisfying_PxPx1_eq_Px2_no_polynomial_satisfies_PxPx1_eq_Px2_plus_1_l203_203666


namespace volume_of_tetrahedron_PQRS_l203_203239

theorem volume_of_tetrahedron_PQRS :
  let K := (0, 0, 0)
  let L := (9, 0, 0)
  let M := some (coordinates such that KM = 15, LM = 16, and form valid tetrahedron KLMN)
  let N := some (coordinates such that KN = 16, LN = 15, MN = 9 and form valid tetrahedron KLMN)
  let P := (some_centroid K L M)
  let Q := (some_centroid K L N)
  let R := (some_centroid K M N)
  let S := (some_centroid L M N)
  volume (P, Q, R, S) = 4.85 :=
sorry

end volume_of_tetrahedron_PQRS_l203_203239


namespace tan_105_degree_l203_203649

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203649


namespace tan_105_degree_l203_203643

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203643


namespace monotonicity_of_g_range_of_a_for_parallel_tangents_l203_203733

noncomputable def f (a x : ℝ) : ℝ := x * real.log x - x + (1/2) * x^2 - (1/3) * a * x^3
noncomputable def g (a x : ℝ) : ℝ := real.log x + x - a * x^2
noncomputable def g' (a x : ℝ) : ℝ := (1/x) + 1 - 2 * a * x

theorem monotonicity_of_g (a : ℝ) :
  (a ≤ 0 ∧ ∀ x > 0, g' a x > 0) ∨
  (a > 0 ∧ (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    g' a x₁ = 0 ∧ g' a x₂ = 0 ∧ 
    ∀ x ∈ (0, x₂), g' a x > 0 ∧ ∀ x ∈ (x₂, ∞), g' a x < 0)) :=
sorry

theorem range_of_a_for_parallel_tangents {a : ℝ} (h₀ : 0 < a) (h₁ : 1 > a) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 :=
sorry

end monotonicity_of_g_range_of_a_for_parallel_tangents_l203_203733


namespace proof_problem1_proof_problem2_l203_203458

noncomputable def problem1 : Prop :=
  sin (6 * real.pi / 180) * sin (42 * real.pi / 180) * sin (66 * real.pi / 180) * sin (78 * real.pi / 180) = 1 / 16

theorem proof_problem1 : problem1 := 
by {
  -- Proof steps would go here, but we'll skip them for now.
  sorry
}

noncomputable def problem2 : Prop := 
  (sin (20 * real.pi / 180))^2 + (cos (50 * real.pi / 180))^2 + sin (20 * real.pi / 180) * cos (50 * real.pi / 180) = 
  (3 / 4) + (1 / 4) * sin (70 * real.pi / 180)

theorem proof_problem2 : problem2 := 
by {
  -- Proof steps would go here, but we'll skip them for now.
  sorry
}

end proof_problem1_proof_problem2_l203_203458


namespace complex_in_fourth_quadrant_l203_203410

open Complex

theorem complex_in_fourth_quadrant (m : ℝ) : ¬(∃ (z : ℂ), z = (m + Complex.i) / (1 - Complex.i) ∧ (Re z) > (Im z)) := 
by
  sorry

end complex_in_fourth_quadrant_l203_203410


namespace solve_for_x_l203_203311

theorem solve_for_x : ∃ x : ℕ, 15 * 2 = 3 + x ∧ x = 27 :=
by
  use 27
  split
  · norm_num
  · rfl

end solve_for_x_l203_203311


namespace amusement_park_l203_203413

theorem amusement_park
  (A : ℕ)
  (adult_ticket_cost : ℕ := 22)
  (child_ticket_cost : ℕ := 7)
  (num_children : ℕ := 2)
  (total_cost : ℕ := 58)
  (cost_eq : adult_ticket_cost * A + child_ticket_cost * num_children = total_cost) :
  A = 2 :=
by {
  sorry
}

end amusement_park_l203_203413


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203610

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203610


namespace count_two_digit_primes_ending_with_3_l203_203848

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203848


namespace carA_arrangements_l203_203673

-- Define the conditions of the problem
def students := {1, 2, 3, 4, 5, 6, 7, 8} -- 8 students
def grades := {1, 2, 3, 4} -- 4 grade levels
def carA := {a // a ∈ students} -- Car A
def carB := {b // b ∈ students} -- Car B

-- Define the freshman twin sisters
def twinSisters := {1, 2}

-- Number of ways to arrange students in Car A to have exactly 2 students from the same grade
def arrangements_in_carA_with_same_grade : ℕ :=
  let twinsInCarA := 3 * 4 -- Case 1
  let twinsNotInCarA := 3 * 4 -- Case 2
  twinsInCarA + twinsNotInCarA

-- Proof statement
theorem carA_arrangements : arrangements_in_carA_with_same_grade = 24 :=
  sorry

end carA_arrangements_l203_203673


namespace count_two_digit_primes_with_ones_digit_3_l203_203807

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203807


namespace sum_of_digits_squared_diff_l203_203355

def x : ℕ := 777777777777777
def y : ℕ := 222222222222223

theorem sum_of_digits_squared_diff : 
  let diff := x^2 - y^2 in 
  (∑ d in diff.digits 10, d) = 74 := sorry

end sum_of_digits_squared_diff_l203_203355


namespace two_digit_primes_ending_in_3_eq_6_l203_203938

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203938


namespace remainder_base12_2543_div_9_l203_203383

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l203_203383


namespace correct_proposition_l203_203735

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition :
  ¬ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  ¬ (∀ h : ℝ, f (-Real.pi / 6 + h) = f (-Real.pi / 6 - h)) ∧
  (∀ h : ℝ, f (-5 * Real.pi / 12 + h) = f (-5 * Real.pi / 12 - h)) :=
by sorry

end correct_proposition_l203_203735


namespace num_two_digit_primes_with_ones_digit_3_l203_203959

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203959


namespace two_digit_primes_with_ones_digit_3_l203_203854

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203854


namespace lucas_factorial_last_two_digits_sum_l203_203105

theorem lucas_factorial_last_two_digits_sum :
  let f2 := Nat.factorial 2,
      f1 := Nat.factorial 1,
      f3 := Nat.factorial 3,
      f4 := Nat.factorial 4,
      f7 := Nat.factorial 7,
      f11 := Nat.factorial 11,
      last_two_digits := (f : Nat) → f % 100,
      sum_last_two_digits := last_two_digits f2 + last_two_digits f1 + last_two_digits f3 + last_two_digits f4 + last_two_digits f7 + last_two_digits f11
  in sum_last_two_digits % 100 = 73 := by
  sorry

end lucas_factorial_last_two_digits_sum_l203_203105


namespace problem_solution_l203_203028

noncomputable theory
open BigOperators

-- Define the various parameters
def n := 800
def p1 := 0.5
def p2 := 0.4
def p3 := 0.3
def trials1 := 200
def trials2 := 400
def trials3 := 200

-- Calculate the average probability
def average_probability : ℝ :=
  (p1 * trials1 + p2 * trials2 + p3 * trials3) / n

-- Calculate the sum of pi * (1 - pi)
def sum_pi_one_minus_pi : ℝ :=
  (p1 * (1 - p1) * trials1) + (p2 * (1 - p2) * trials2) + (p3 * (1 - p3) * trials3)

-- Ε and Poisson's theorem parameters
def epsilon : ℝ := 0.04
def poisson_bound : ℝ := sum_pi_one_minus_pi / (n * n * epsilon * epsilon)

-- The actual probability bound
def probability_bound : ℝ := 1 - poisson_bound

-- State the theorem
theorem problem_solution : probability_bound ≥ 0.817 :=
by
  sorry

end problem_solution_l203_203028


namespace area_BEFC_l203_203400

noncomputable def area_BC_ABC (s : ℝ) : ℝ := 
  (s^2 * Real.sqrt 3) / 4

theorem area_BEFC :
  let s := 3 in 
  let A := (0, 0) : ℝ × ℝ in 
  let B := (s, 0) in 
  let C := (s/2, s * Real.sqrt 3 / 2) in 
  let D := (3 * s / 2, 0) in 
  let E := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
  let F := (2 * s / 3, 0) in
  4 * area_BC_ABC s / 3 = (3 * Real.sqrt 3 / 2) := 
begin 
  -- sorry to skip the proof
  sorry
end

end area_BEFC_l203_203400


namespace tan_105_degree_l203_203586

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203586


namespace max_value_of_f_l203_203341

def f (x : ℝ) : ℝ := sin (3 * Real.pi / 2 + x) * cos (Real.pi / 6 - x)

theorem max_value_of_f : 
  ∃ x : ℝ, f x = (1 / 2 - Real.sqrt 3 / 4) :=
sorry

end max_value_of_f_l203_203341


namespace tan_105_degree_l203_203587

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203587


namespace num_two_digit_primes_with_ones_digit_3_l203_203956

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203956


namespace two_digit_primes_with_ones_digit_3_l203_203860

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203860


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203970

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203970


namespace tan_105_l203_203545

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203545


namespace two_digit_primes_with_ones_digit_3_l203_203857

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203857


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203485

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203485


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203894

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203894


namespace point_P_coordinates_l203_203164

theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 < 0 ∧ abs P.2 = 3 ∧ abs P.1 = 8 ∧ P = (8, -3) :=
sorry

end point_P_coordinates_l203_203164


namespace monotonic_decreasing_intervals_solve_sin_cos_problem_l203_203741

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem monotonic_decreasing_intervals :
  ∀ (k : ℤ), (λ x, f x) isMonotonicDecreasingOn [((3 * Real.pi / 8) + k * Real.pi), ((7 * Real.pi / 8) + k * Real.pi)] := by sorry

theorem solve_sin_cos_problem (α : ℝ) (h₁ : α ∈ Ioo (Real.pi / 2) Real.pi)
  (h₂ : f ((α / 2) + (Real.pi / 4)) = (2 / 3) * Real.cos (α + (Real.pi / 4)) * Real.cos (2 * α)):
  ∃ (x : ℝ), x = (Real.sin α - Real.cos α) ∧ (x = (Real.sqrt 6 / 2) ∨ x = Real.sqrt 2) := by sorry

end monotonic_decreasing_intervals_solve_sin_cos_problem_l203_203741


namespace count_two_digit_primes_ending_with_3_l203_203841

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203841


namespace segment_AX_length_l203_203294

-- Given conditions
variables (A B C D X O : Type)
variables (diameter : line)
variables (angleBAC angleBXC : ℝ)

-- Circle definitions
def is_on_circle (p : Type) (diameter : line) : Prop := sorry
def is_diameter (p1 p2 : Type) (diameter : line) : Prop := true
def is_on_diameter (p : Type) (p1 p2 : Type) (diameter : line) : Prop := true
def angle_eq (theta1 theta2 : ℝ) : Prop := theta1 = theta2 

-- Problem conditions
axiom points_on_circle : is_on_circle A diameter 
axiom points_on_circle2 : is_on_circle B diameter
axiom points_on_circle3 : is_on_circle C diameter
axiom points_on_circle4 : is_on_circle D diameter

axiom BXC_eq_BX_CX : BX = CX
axiom AD_is_diameter : is_diameter A D diameter
axiom X_on_diameter : is_on_diameter X A D diameter
axiom angle_eq1 : angle_eq (4 * angleBAC) 72
axiom angle_eq2 : angle_eq angleBXC 72

-- Prove AX = 2 * sin(18 degree)
theorem segment_AX_length : 
  ∃ AX : ℝ, AX = 2 * Real.sin (Real.pi / 10) := sorry

end segment_AX_length_l203_203294


namespace tan_105_eq_neg2_sub_sqrt3_l203_203615

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203615


namespace count_two_digit_primes_ending_in_3_l203_203825

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203825


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203976

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203976


namespace two_digit_primes_ending_in_3_eq_6_l203_203934

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203934


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l203_203996

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l203_203996


namespace rectangle_area_l203_203322

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l203_203322


namespace hyperbola_slope_range_l203_203280

variables {a b : ℝ}
variables (h_pos_a : a > 0) (h_pos_b : b > 0)
variables (dist_condition : ∀ D BC, dist D BC < a + sqrt (a^2 + b^2))

theorem hyperbola_slope_range
  (hyp : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
  (∀ m, m ∉ (-1, 0) ∪ (0, 1)) ->
  False :=
by
  sorry

end hyperbola_slope_range_l203_203280


namespace tan_105_l203_203494

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203494


namespace least_num_add_divisible_l203_203000

def least_number_to_add (a b : ℕ) : ℕ :=
  let remainder := a % b
  b - remainder

theorem least_num_add_divisible (a b : ℕ) (h : ∃ n, a = n * b + least_number_to_add a b) : 
  (a + least_number_to_add a b) % b = 0 :=
by
  obtain ⟨k, hk⟩ := h
  have : least_number_to_add a b = b - (a % b) := rfl
  rw [least_number_to_add, this, hk, nat.add_sub_cancel_left]
  apply nat.mod_add_div
  sorry

end least_num_add_divisible_l203_203000


namespace part_1_part_2_l203_203216

theorem part_1 (A B C : ℝ) (a b c : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) :
  2 * c = a + b := by
  sorry

theorem part_2 (A B : ℝ) (a b c : ℝ) (h1 : A + B = c) (h2 : b * (1 + Real.cos (Real.pi / 3)) = c * (2 - Real.cos B)) (h3 : 4 * sqrt 3 = (1/2) * a * b * (sqrt 3 / 2)) :
  c = 4 := by
  sorry

end part_1_part_2_l203_203216


namespace area_annulus_l203_203072

noncomputable def area_of_annulus (R r l : ℝ) (hRr : R > r) (radius_condition : R^2 = r^2 + (l / 2)^2) : ℝ :=
  π * (l / 2)^2

theorem area_annulus (R r l : ℝ) (hRr : R > r) (radius_condition : R^2 = r^2 + (l / 2)^2) :
  area_of_annulus R r l hRr radius_condition = π * (R^2 - r^2) :=
begin
  sorry
end

end area_annulus_l203_203072


namespace tan_105_degree_l203_203558

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203558


namespace min_dot_product_on_hyperbola_l203_203745

open Real

theorem min_dot_product_on_hyperbola :
  ∀ (P : ℝ × ℝ), (P.1 ≥ 1 ∧ P.1^2 - (P.2^2) / 3 = 1) →
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  ∃ m : ℝ, m = -2 ∧ PA1.1 * PF2.1 + PA1.2 * PF2.2 = m :=
by
  intros P h
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  use -2
  sorry

end min_dot_product_on_hyperbola_l203_203745


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203480

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203480


namespace distance_of_point_A_l203_203057

noncomputable def square_side_length (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def fold_length (area : ℝ) : ℝ :=
  real.sqrt (2 * area / 3)

noncomputable def distance_traveled (fold_length : ℝ) : ℝ :=
  real.sqrt (fold_length^2 + fold_length^2)

theorem distance_of_point_A
  (area : ℝ) (h_area : area = 18) :
  distance_traveled (fold_length area) = 2 * real.sqrt 6 :=
by
  sorry

end distance_of_point_A_l203_203057


namespace regular_polygon_properties_l203_203721

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l203_203721


namespace area_of_scaled_sum_area_perimeter_inequality_l203_203017

-- Definition for part (a)
variable (M : Type) [ConvexPolygon M]
variable (D : Type) [Circle D]
variable (S P R : ℝ) [Nonnegative S] [Nonnegative P] [Nonnegative R]
variable (λ₁ λ₂ : ℝ) [Nonnegative λ₁] [Nonnegative λ₂]

theorem area_of_scaled_sum
  (hM_area : M → ℝ := S)
  (hM_perimeter : M → ℝ := P)
  (hD_radius : D → ℝ := R)
  : area((λ₁ : ℝ) * M + (λ₂ : ℝ) * D) = λ₁^2 * S + λ₁ * λ₂ * P * R + λ₂^2 * π * R^2 := by
  sorry

-- Definition for part (b)
variable (S P : ℝ) [Nonnegative S] [Nonnegative P]

theorem area_perimeter_inequality 
  (h_isoperimetric : ∀ (A : ℝ) (L : ℝ), 4 * π * A ≤ L^2)
  : S ≤ P^2 / (4 * π) := by
  sorry

end area_of_scaled_sum_area_perimeter_inequality_l203_203017


namespace tan_105_l203_203497

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203497


namespace projectile_highest_point_l203_203046

noncomputable def highest_point (v w_h w_v θ g : ℝ) : ℝ × ℝ :=
  let t := (v * Real.sin θ + w_v) / g
  let x := (v * t + w_h * t) * Real.cos θ
  let y := (v * t + w_v * t) * Real.sin θ - (1/2) * g * t^2
  (x, y)

theorem projectile_highest_point : highest_point 100 10 (-2) (Real.pi / 4) 9.8 = (561.94, 236) :=
  sorry

end projectile_highest_point_l203_203046


namespace count_two_digit_primes_ending_in_3_l203_203823

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203823


namespace min_segments_on_edges_l203_203409

-- Define a structure for a cube
structure Cube :=
(vertices : Fin 8) -- A cube has 8 vertices
(edges : Fin 12 → (Fin 8 × Fin 8)) -- A cube has 12 edges, each connecting two vertices

def is_edge (c : Cube) (u v : Fin 8) : Prop :=
∃ e, c.edges e = (u, v) ∨ c.edges e = (v, u)

def is_diagonal (c : Cube) (u v : Fin 8) : Prop :=
¬(is_edge c u v) ∧ ∃ f, (u, v) ∈ face_diagonals f

-- Define the main theorem
theorem min_segments_on_edges (c : Cube) (polyline : list (Fin 8)) 
(h1 : polyline.head = polyline.last)        -- closed polyline
(h2 : polyline.length = 8 + 1)            -- 8 segments, thus 9 vertices
(h3 : ∀ v ∈ polyline, ∃ i : Fin 8, v = i) -- vertices coincide with the cube vertices
: ∃ segs_on_edges, (∀ ⦃i j : ℕ⦄, i < j → segs_on_edges i = polyline.nth i ∧ segs_on_edges j = polyline.nth j → is_edge c (polyline.nth i) (polyline.nth j)) ∧
  segs_on_edges.length ≥ 2 :=
sorry

end min_segments_on_edges_l203_203409


namespace two_digit_primes_with_ones_digit_3_l203_203861

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203861


namespace two_digit_primes_with_ones_digit_three_count_l203_203779

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203779


namespace geometric_series_m_value_l203_203074

theorem geometric_series_m_value (m : ℝ) : 
    let a : ℝ := 20
    let r₁ : ℝ := 1 / 2  -- Common ratio for the first series
    let S₁ : ℝ := a / (1 - r₁)  -- Sum of the first series
    let b : ℝ := 1 / 2 + m / 20  -- Common ratio for the second series
    let S₂ : ℝ := a / (1 - b)  -- Sum of the second series
    S₁ = 40 ∧ S₂ = 120 → m = 20 / 3 :=
sorry

end geometric_series_m_value_l203_203074


namespace tan_105_degree_l203_203645

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203645


namespace two_digit_primes_ending_in_3_eq_6_l203_203940

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203940


namespace find_k_l203_203742

noncomputable def f (x : ℝ) := Real.sin x + Real.tan x

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (start : ℝ) : Prop :=
  ∀ n, a n = start + n * d

variables (a : ℕ → ℝ) (d : ℝ) (start : ℝ)
  (h_arithmetic_seq : arithmetic_sequence a d start)
  (h_in_domain : ∀ n, a n ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h_non_zero_d : d ≠ 0)
  (h_sum_zero : (Finset.range 31).sum (λ n, f (a n)) = 0)

theorem find_k (k : ℕ) (h_k : k = (31 + 1) / 2) : f (a k) = 0 :=
by {
  sorry
}

end find_k_l203_203742


namespace zeros_in_decimal_rep_of_one_over_30_pow_15_l203_203303

theorem zeros_in_decimal_rep_of_one_over_30_pow_15 :
  ∀ (n : ℕ), (n = 30^15) → (nat.log10 n = 22) :=
by
  intro n h
  rw h
  -- Further steps to complete the proof would go here
  sorry

end zeros_in_decimal_rep_of_one_over_30_pow_15_l203_203303


namespace find_x_equality_l203_203277

-- Define the product of digits function
noncomputable def p (x : ℕ) : ℕ :=
  (x.digits 10).prod

theorem find_x_equality :
  ∀ x : ℕ, x > 0 → p(x) = x^2 - 10 * x - 22 ↔ x = 12 := 
by
  intro x
  intro hx
  sorry

end find_x_equality_l203_203277


namespace appropriate_presentation_length_l203_203368

-- Define the conditions as Lean definitions
def one_third_hour : ℝ := 1 / 3
def two_third_hour : ℝ := 2 / 3
def rate_of_speech : ℝ := 120 -- words per minute

-- Convert hours to minutes
def minutes_in_hour : ℕ := 60

def min_time_minutes : ℕ := (one_third_hour * minutes_in_hour).toNat
def max_time_minutes : ℕ := (two_third_hour * minutes_in_hour).toNat

-- Calculate the word range
def min_words : ℕ := min_time_minutes * rate_of_speech.toNat
def max_words : ℕ := max_time_minutes * rate_of_speech.toNat

-- Define the question in Lean 4
theorem appropriate_presentation_length (words : ℕ) :
    words = 2700 ∨ words = 3900 ∨ words = 4500 ↔ (min_words ≤ words ∧ words ≤ max_words) :=
begin
  sorry -- Proof is not required
end

end appropriate_presentation_length_l203_203368


namespace xy_plus_four_is_square_l203_203175

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end xy_plus_four_is_square_l203_203175


namespace tan_105_l203_203498

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203498


namespace sum_first_3k_plus_2_terms_l203_203425

variable (k : ℕ)

def first_term : ℕ := k^2 + 1

def sum_of_sequence (n : ℕ) : ℕ :=
  let a₁ := first_term k
  let aₙ := a₁ + (n - 1)
  n * (a₁ + aₙ) / 2

theorem sum_first_3k_plus_2_terms :
  sum_of_sequence k (3 * k + 2) = 3 * k^3 + 8 * k^2 + 6 * k + 3 :=
by
  -- Here we define the sequence and compute the sum
  sorry

end sum_first_3k_plus_2_terms_l203_203425


namespace bounded_area_l203_203338

theorem bounded_area (a : ℝ) (h : 0 < a) :
  let region_area := (λ (x y : ℝ), (x + a * y)^2 ≤ 9 * a^2) ∧ (λ (x y : ℝ), (a * x - y)^2 ≤ 4 * a^2) in
  ∃ area : ℝ, area = 24 * a^2 / (1 + a^2) := sorry

end bounded_area_l203_203338


namespace count_two_digit_primes_with_ones_3_l203_203877

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203877


namespace smallest_number_of_beads_l203_203010

theorem smallest_number_of_beads (M : ℕ) (h1 : ∃ d : ℕ, M = 5 * d + 2) (h2 : ∃ e : ℕ, M = 7 * e + 2) (h3 : ∃ f : ℕ, M = 9 * f + 2) (h4 : M > 1) : M = 317 := sorry

end smallest_number_of_beads_l203_203010


namespace regular_polygon_sides_and_interior_angle_l203_203728

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l203_203728


namespace two_digit_primes_ending_in_3_eq_6_l203_203933

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203933


namespace train_speed_is_45_kmh_l203_203434

-- Define the conditions
def train_length : ℕ := 360  -- length of the train in meters
def platform_length : ℕ := 140  -- length of the platform in meters
def time_to_pass : ℕ := 40  -- time to pass the platform in seconds

-- The theorem to prove that the speed of the train is 45 km/hr
theorem train_speed_is_45_kmh : ((train_length + platform_length) / time_to_pass) * 3.6 = 45 := sorry

end train_speed_is_45_kmh_l203_203434


namespace rate_of_stream_l203_203391

theorem rate_of_stream (x : ℝ) (h1 : ∀ (distance : ℝ), (24 : ℝ) > 0) (h2 : ∀ (distance : ℝ), (distance / (24 - x)) = 3 * (distance / (24 + x))) : x = 12 :=
by
  sorry

end rate_of_stream_l203_203391


namespace tan_105_eq_neg2_sub_sqrt3_l203_203580

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203580


namespace cos_A_value_compare_angles_l203_203243

variable (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a = 3) (h2 : b = 2 * Real.sqrt 6) (h3 : B = 2 * A)

-- Problem (I) statement
theorem cos_A_value (hcosA : Real.cos A = Real.sqrt 6 / 3) : 
  Real.cos A = Real.sqrt 6 / 3 :=
by 
  sorry

-- Problem (II) statement
theorem compare_angles (hcosA : Real.cos A = Real.sqrt 6 / 3) (hcosC : Real.cos C = Real.sqrt 6 / 9) :
  B < C :=
by
  sorry

end cos_A_value_compare_angles_l203_203243


namespace inverse_89_mod_91_l203_203680

theorem inverse_89_mod_91 : ∃ x ∈ set.Icc 0 90, (89 * x) % 91 = 1 :=
by
  use 45
  split
  · exact ⟨le_refl 45, le_of_lt (by norm_num)⟩
  · norm_num; sorry

end inverse_89_mod_91_l203_203680


namespace lateral_surface_area_l203_203153

noncomputable def area_of_base := 4 * Real.sqrt 3
noncomputable def lateral_edge_length := 3

-- proof problem statement
theorem lateral_surface_area (a : ℝ) (h_base_area : (Real.sqrt 3 / 4) * a ^ 2 = area_of_base)
  (h_lateral_edge : lateral_edge_length = 3) :
  let lateral_area := 3 * a * lateral_edge_length in
  lateral_area = 36 :=
by
  -- skipping the proof
  sorry

end lateral_surface_area_l203_203153


namespace sum_of_zeros_eq_neg_six_l203_203737

def f (x : ℝ) : ℝ :=
if x ≤ 0 then sin (π * x) + 1 
else log 2 (3 * x ^ 2 - 12 * x + 15)

noncomputable def y (x : ℝ) := f x - 1 

theorem sum_of_zeros_eq_neg_six :
  (∑ x in set.to_finset {x : ℝ | -3 ≤ x ∧ x ≤ 3 ∧ y x = 0}, x) = -6 :=
sorry

end sum_of_zeros_eq_neg_six_l203_203737


namespace tan_105_eq_neg2_sub_sqrt3_l203_203573

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203573


namespace angle_D_measure_l203_203221

theorem angle_D_measure (A B C D E : Type)
  (AB BC CD CE : ℝ)
  (h1 : AB = BC)
  (h2 : BC = CD)
  (h3 : CD = CE)
  (h4 : ∀ (x : ℝ), x = (3 * B)) :
  angle D = 72 := by
  sorry

end angle_D_measure_l203_203221


namespace Tom_age_is_25_l203_203372

noncomputable def TomAge (T : ℕ) : Prop :=
  let jaredCurrentAge := 48
  let jaredAgeTwoYearsAgo := jaredCurrentAge - 2
  jaredAgeTwoYearsAgo = 2 * (T - 2)

theorem Tom_age_is_25 : TomAge 25 :=
  by
    unfold TomAge
    simp
    sorry

end Tom_age_is_25_l203_203372


namespace convex_quadrilaterals_count_l203_203145

theorem convex_quadrilaterals_count (n : ℕ) (h₁ : n > 4) 
  (h₂ : ∀ (x₁ x₂ x₃ : ℝ × ℝ), ¬ collinear {x₁, x₂, x₃}) :
  ∃ (points : Fin n → ℝ × ℝ), number_of_convex_quadrilaterals points ≥ (n - 3) * (n - 4) / 2 :=
sorry

end convex_quadrilaterals_count_l203_203145


namespace p_sufficient_not_necessary_for_q_l203_203713

def p (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 ≤ 2
def q (x y : ℝ) : Prop := y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ ¬(∀ x y : ℝ, p x y → q x y) := by
  sorry

end p_sufficient_not_necessary_for_q_l203_203713


namespace num_two_digit_primes_with_ones_digit_3_l203_203942

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203942


namespace sum_of_variables_l203_203206

variables (a b c d : ℝ)

theorem sum_of_variables :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 → a + b + c + d = 16 :=
by
  intro h
  -- your proof goes here
  sorry

end sum_of_variables_l203_203206


namespace intervals_of_monotonicity_range_of_a_l203_203182

noncomputable def f (x a : ℝ) := (Real.exp x - a) / x
noncomputable def g (x a : ℝ) := a * Real.log x + a
noncomputable def F (x a : ℝ) := f x a - g x a

-- Statement for the first question
theorem intervals_of_monotonicity (x : ℝ) (hx : x > 0) : 
  let F1 := F x 1 in
  (∀ x ≥ 1, ∀ y > x, F1 y ≥ F1 x) ∧ (∀ x ∈ (0, 1), ∀ y < x, F1 y ≤ F1 x) := 
sorry

-- Statement for the second question
theorem range_of_a (a : ℝ) : 
  (∀ x > 1, f x a > g x a) ↔ a ≤ (1 / 2) * Real.exp 1 := 
sorry

end intervals_of_monotonicity_range_of_a_l203_203182


namespace count_two_digit_primes_with_ones_digit_3_l203_203811

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203811


namespace positive_difference_median_mode_l203_203001

noncomputable def stem_and_leaf_data : List Nat :=
  [22, 23, 24, 25, 25, 25, 32, 32, 32, 33, 34, 41, 41, 48, 49, 50, 51, 52, 53,
    61, 62, 68, 69, 69, 69]

noncomputable def mode : Nat :=
  25

noncomputable def median : Nat :=
  69

theorem positive_difference_median_mode :
  abs (median - mode) = 44 :=
by
  sorry

end positive_difference_median_mode_l203_203001


namespace solution_set_inequality_l203_203167

variable (f : ℝ → ℝ)
variable (e : ℝ)

-- Conditions
axiom h1 : ∀ x, f.deriv x.deriv = f x.deriv
axiom h2 : f 1 = Real.exp 1
axiom h3 : ∀ x, 2 * f x - f.deriv x.deriv > 0

-- Proof problem
theorem solution_set_inequality : {x : ℝ | f x / Real.exp x < Real.exp (x - 1)} = {x : ℝ | x > 1} := by
  sorry

end solution_set_inequality_l203_203167


namespace tan_105_degree_l203_203595

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203595


namespace circle_equation_solution_l203_203107

theorem circle_equation_solution (a : ℝ) :
  a^2x^2 + (a+2)y^2 + 2ax + a = 0 → a = 2 :=
sorry

end circle_equation_solution_l203_203107


namespace tangent_line_equation_a_range_for_g_nonnegative_l203_203740

-- Problem 1
def f (x : ℝ) : ℝ := x / Real.exp x + x^2 - x

theorem tangent_line_equation :
  let e := Real.exp 1
  ex - e * f 1 - e + 1 = 0 :=
by sorry

-- Problem 2
def g (f : ℝ → ℝ) (a x : ℝ) : ℝ := -a * Real.log (f x - x^2 + x) - 1 / x - Real.log x - a + 1

theorem a_range_for_g_nonnegative (a : ℝ) :
  (∀ x ≥ 1, g f a x ≥ 0) → a ≥ 1 :=
by sorry

end tangent_line_equation_a_range_for_g_nonnegative_l203_203740


namespace tan_105_l203_203471

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203471


namespace count_two_digit_primes_ending_in_3_l203_203822

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203822


namespace widgets_per_shipping_box_l203_203669

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l203_203669


namespace a_n_formula_geometric_sequence_minimum_sum_value_l203_203151

-- Define the sequences a_n and b_n
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)

-- Given initial conditions and recursive relations
def initial_conditions (a b S : ℕ → ℝ) : Prop :=
  (a 1 = 1 / 4) ∧ (∀ n, S n = S (n - 1) + a n + 1 / 2) ∧ 
  (b 1 = -119 / 4) ∧ (∀ n ≥ 2, 3 * b n - b (n - 1) = n)

-- Question 1: Prove general term of sequence a_n
theorem a_n_formula (a S : ℕ → ℝ) (h : initial_conditions a b S) :
  ∀ n, a n = (1 / 2) * n - (1 / 4) :=
sorry

-- Question 2: Prove that (b_n - a_n) forms a geometric sequence
theorem geometric_sequence (a b : ℕ → ℝ) (h : initial_conditions a b S) :
  ∃ r : ℝ, ∀ n, b n - a n = -30 * r^(n - 1) ∧ r = 1 / 3 :=
sorry

-- Question 3: Find the minimum value of the sum of the first n terms of b_n
theorem minimum_sum_value (b : ℕ → ℝ) (h : initial_conditions a b S) :
  ∃ n, (S n).sum = -41 - (1 / 12) :=
sorry

end a_n_formula_geometric_sequence_minimum_sum_value_l203_203151


namespace perpendicular_line_plane_l203_203144

variables (m n : Type) [LinearOrderedField m] [LinearOrderedField n] 
variables (m_perp_alpha: Prop) (n_subset_alpha: Prop)

theorem perpendicular_line_plane (m_perp_alpha : m → α → Prop) (n_subset_alpha : n → α → Prop) : 
  m_perp_alpha m α → n_subset_alpha n α → m_perp_alpha m n :=
by 
  sorry

end perpendicular_line_plane_l203_203144


namespace simplify_f_value_f_given_condition_l203_203143

def f (alpha : ℝ) : ℝ := (sin (π / 2 - alpha) * sin (-alpha) * tan (π - alpha)) / (tan (-alpha) * sin (π - alpha))

theorem simplify_f (alpha : ℝ) : f(alpha) = cos(alpha) :=
by 
  sorry

theorem value_f_given_condition (alpha : ℝ) (h1 : α > 3 * π / 2 ∨ α < 2 * π) (h2 : cos (3 * π / 2 - α) = 2 / 3) :
  f(α) = sqrt 5 / 3 :=
by
  sorry

end simplify_f_value_f_given_condition_l203_203143


namespace count_two_digit_primes_with_ones_digit_3_l203_203800

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203800


namespace two_digit_primes_with_ones_digit_3_l203_203858

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203858


namespace probability_bob_has_ball_again_after_two_turns_l203_203439

theorem probability_bob_has_ball_again_after_two_turns :
  let P_bob_bob := 3/4 * 3/4 + 3/4 * 1/4 * 2/3 + 1/4 * 1/3 in
  P_bob_bob = 37/48 :=
by
  sorry

end probability_bob_has_ball_again_after_two_turns_l203_203439


namespace simplify_and_rationalize_l203_203307

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end simplify_and_rationalize_l203_203307


namespace percentage_of_men_with_college_degree_l203_203077

theorem percentage_of_men_with_college_degree 
    (total_employees : ℕ) (total_women : ℕ) (total_men : ℕ)
    (men_without_degree : ℕ) (men_with_degree : ℕ) :
    total_women = 48 →
    total_employees = total_women / 0.60 →
    total_men = total_employees * 0.40 →
    men_without_degree = 8 →
    men_with_degree = total_men - men_without_degree →
    (men_with_degree / total_men) * 100 = 75 :=
by
  intros
  sorry

end percentage_of_men_with_college_degree_l203_203077


namespace pebbles_collected_by_tenth_day_l203_203287

-- Define the initial conditions
def a : ℕ := 2
def r : ℕ := 2
def n : ℕ := 10

-- Total pebbles collected by the end of the 10th day
def total_pebbles (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Proof statement
theorem pebbles_collected_by_tenth_day : total_pebbles a r n = 2046 :=
  by sorry

end pebbles_collected_by_tenth_day_l203_203287


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203479

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203479


namespace two_digit_primes_with_ones_digit_3_l203_203866

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203866


namespace tan_105_l203_203555

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203555


namespace tan_105_l203_203463

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203463


namespace exists_chord_through_point_subtending_angle_l203_203373

-- Define the problem conditions and parameters
variables {O K : Point} -- O is the center of the circle, K is a point inside the circle
variables {r : ℝ}       -- r is the radius of the circle
variables {α : Angle}   -- α is the given angle

-- The definition of a circle with center O and radius r
def circle (O : Point) (r : ℝ) := {P : Point | dist O P = r}

-- Definition of a point being inside the circle
def inside_circle (O : Point) (r : ℝ) (K : Point) : Prop :=
  dist O K < r

-- The existence of such a chord
theorem exists_chord_through_point_subtending_angle (hK : inside_circle O r K) :
  ∃ A B : Point, 
    (A ∈ circle O r ∧ B ∈ circle O r) ∧ 
    dist A B < 2 * r ∧ 
    subtends_angle O A B α ∧ 
    lies_on_chord K A B :=
sorry

end exists_chord_through_point_subtending_angle_l203_203373


namespace tan_105_eq_neg2_sub_sqrt3_l203_203582

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203582


namespace area_of_WXYZ_l203_203223

variables (A B C D E G F H I J W X Y Z : Type)

-- Conditions for the rectangle ADEH and points B, C, I quadriving AD, and points G, F, J quadriving HE
variables (quadriveAD : quadrive A B C I D)
variables (quadriveHE : quadrive H J G F E)
variables (lengthAH : ℝ) (lengthAD : ℝ)
variable (isRectangle : rectangle A D E H)

-- The rectangle has side lengths 4
axiom hyp_lengths : lengthAH = 4 ∧ lengthAD = 4

-- Defining the quadrilateral WXYZ within the rectangle
variables (W1 X1 Y1 Z1 : Type) (quadWXYZ : quadrilateral W X Y Z isRectangle)

-- Statement to be proven: The area of quadrilateral WXYZ is 2
theorem area_of_WXYZ : area quadWXYZ = 2 :=
sorry

end area_of_WXYZ_l203_203223


namespace set_A_set_B_union_A_B_range_a_l203_203186

section math_problem

variables (a x : ℝ)

-- Define the sets A, B, C
def A := {x : ℝ | (2 - x) / (3 + x) ≥ 0}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | x^2 - (2 * a + 1) * x + a * (a + 1) < 0}

-- Part I: Prove the sets A, B and A ∪ B are as given.
theorem set_A : A = {x : ℝ | -3 < x ∧ x ≤ 2} := sorry
theorem set_B : B = {x : ℝ | -1 < x ∧ x < 3} := sorry
theorem union_A_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 3} := sorry

-- Part II: If C ⊆ (A ∩ B), prove the range for a
theorem range_a (h : C ⊆ A ∩ B) : -1 ≤ a ∧ a ≤ 1 := sorry

end math_problem

end set_A_set_B_union_A_B_range_a_l203_203186


namespace tan_105_eq_neg2_sub_sqrt3_l203_203624

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203624


namespace find_incorrect_result_l203_203289

theorem find_incorrect_result
  (correct_multiplier : ℕ := 153)
  (correct_result : ℕ := 109395)
  (number : ℕ := 715)
  (incorrect_multiplier : ℕ := 152)
  (incorrect_result : ℕ := 108680) :
  number * correct_multiplier = correct_result →
  number * incorrect_multiplier = incorrect_result :=
by
  intro h
  have h1 : correct_result = number * correct_multiplier, from h
  have h2 : incorrect_result = number * incorrect_multiplier, from sorry
  exact h2

end find_incorrect_result_l203_203289


namespace tan_105_degree_l203_203594

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203594


namespace plane_Q_equation_l203_203657

theorem plane_Q_equation :
  ∃ (A B C D : ℤ) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1),
  let Q := λ x y z, A * x + B * y + C * z + D,
  let plane1 := λ x y z, 2 * x - y + z - 4,
  let plane2 := λ x y z, x + 3 * y - z - 5,
  let d := λ x y z, abs (A * 1 + B * (-2) + C * 0 + D) / sqrt (A * A + B * B + C * C),
  let M := λ x y z, plane1 x y z = 0 ∧ plane2 x y z = 0,
  Q = (λ x y z, 8 * x - 1 * y + 7 * z - 10) ∧
  ∀ x y z, (M x y z → Q x y z = 0) ∧ d 1 -2 0 = 3 / sqrt 5 :=
begin
  sorry
end

end plane_Q_equation_l203_203657


namespace count_two_digit_primes_ending_with_3_l203_203834

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203834


namespace length_of_BO_l203_203360

noncomputable def triangle_ABC_is_right_isosceles (A B C O : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
(is_isosceles_right_triangle A B C) ∧ (circumcenter A B C = O) ∧ (dist A B = 6)

theorem length_of_BO {A B C O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (h : triangle_ABC_is_right_isosceles A B C O) : dist B O = 3 * real.sqrt 2 :=
sorry

end length_of_BO_l203_203360


namespace count_two_digit_primes_ending_with_3_l203_203845

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203845


namespace two_digit_primes_ending_in_3_eq_6_l203_203931

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203931


namespace magnitude_of_complex_fraction_l203_203209

theorem magnitude_of_complex_fraction (b : ℂ)
  (h1 : (1 + b * complex.I) * (2 + complex.I)).re = 0 :
  abs ((2 * b + 3 * complex.I) / (1 + b * complex.I)) = real.sqrt 5 :=
sorry

end magnitude_of_complex_fraction_l203_203209


namespace ratio_of_triangle_and_hexagon_l203_203302

variable {n m : ℝ}

-- Conditions:
def is_regular_hexagon (ABCDEF : Type) : Prop := sorry
def area_of_hexagon (ABCDEF : Type) (n : ℝ) : Prop := sorry
def area_of_triangle_ACE (ABCDEF : Type) (m : ℝ) : Prop := sorry
  
theorem ratio_of_triangle_and_hexagon
  (ABCDEF : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : area_of_hexagon ABCDEF n)
  (H3 : area_of_triangle_ACE ABCDEF m) :
  m / n = 2 / 3 := 
  sorry

end ratio_of_triangle_and_hexagon_l203_203302


namespace prove_p_l203_203019

variables {m n p : ℝ}

/-- Given points (m, n) and (m + p, n + 4) lie on the line 
   x = y / 2 - 2 / 5, prove p = 2.
-/
theorem prove_p (hmn : m = n / 2 - 2 / 5)
                (hmpn4 : m + p = (n + 4) / 2 - 2 / 5) : p = 2 := 
by
  sorry

end prove_p_l203_203019


namespace cos_2x_value_ratio_of_sides_and_sines_l203_203192

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, 1)
noncomputable def vector_n (x : ℝ) : ℝ × ℝ := (sin x, cos x ^ 2 - 1)
noncomputable def f (x : ℝ) : ℝ := (vector_m x).fst * (vector_n x).fst + (vector_m x).snd * (vector_n x).snd + 1 / 2

theorem cos_2x_value (x : ℝ) (h : x ∈ Set.Icc 0 (π / 4)) (hf : f x = sqrt 3 / 3) :
    cos (2 * x) = sqrt 2 / 2 + sqrt 3 / 6 :=
  sorry

variable (a b c A B C : ℝ)
-- Area of triangle is 1/2 * a * c * sin B = sqrt(3)/4
def triangle_area : Prop := (1 / 2) * a * c * sin B = sqrt 3 / 4
def tri_ineq : Prop := 2 * b * cos A ≤ 2 * c - sqrt 3 * a
-- Law of sines
def law_of_sines : Prop := 
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem ratio_of_sides_and_sines (h1 : a = 1) (h2 : triangle_area) (h3 : tri_ineq) (h4 : law_of_sines) :
    (a + c) / (sin A + sin C) = 2 :=
  sorry

end cos_2x_value_ratio_of_sides_and_sines_l203_203192


namespace rectangle_area_l203_203323

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l203_203323


namespace tan_105_eq_neg2_sub_sqrt3_l203_203577

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203577


namespace count_two_digit_primes_with_ones_3_l203_203887

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203887


namespace regular_polygon_properties_l203_203722

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_properties_l203_203722


namespace school_students_sum_l203_203054

theorem school_students_sum (s : ℕ) (h1 : 180 ≤ s) (h2 : s ≤ 250)
  (h3 : (s - 1) % 8 = 0) : 
  s ∈ ((range (250 - 180 + 1)).filter (λ n, (180 + n - 1) % 8 = 0)) →
  ((range (250 - 180 + 1)).filter (λ n, (180 + n - 1) % 8 = 0)).sum (λ n, 180 + n) = 1953 :=
by
  sorry -- Proof skipped for brevity.

end school_students_sum_l203_203054


namespace lines_intersect_ellipse_at_2_or_4_points_l203_203369

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ line x y

def number_of_intersections (line1 line2 : ℝ → ℝ → Prop) (n : ℕ) : Prop :=
  ∃ pts : Finset (ℝ × ℝ), (∀ pt ∈ pts, (line_intersects_ellipse line1 pt.1 pt.2 ∨
                                        line_intersects_ellipse line2 pt.1 pt.2)) ∧
                           pts.card = n ∧ 
                           (∀ pt ∈ pts, line1 pt.1 pt.2 ∨ line2 pt.1 pt.2) ∧
                           (∀ (pt1 pt2 : ℝ × ℝ), pt1 ∈ pts → pt2 ∈ pts → pt1 ≠ pt2 → pt1 ≠ pt2)

theorem lines_intersect_ellipse_at_2_or_4_points 
  (line1 line2 : ℝ → ℝ → Prop)
  (h1 : ∃ x1 y1, line1 x1 y1 ∧ ellipse_eq x1 y1)
  (h2 : ∃ x2 y2, line2 x2 y2 ∧ ellipse_eq x2 y2)
  (h3: ¬ ∀ x y, line1 x y ∧ ellipse_eq x y → false)
  (h4: ¬ ∀ x y, line2 x y ∧ ellipse_eq x y → false) :
  ∃ n : ℕ, (n = 2 ∨ n = 4) ∧ number_of_intersections line1 line2 n := sorry

end lines_intersect_ellipse_at_2_or_4_points_l203_203369


namespace percent_decrease_trouser_correct_percent_decrease_shirt_correct_percent_decrease_shoes_correct_percent_decrease_jacket_correct_percent_decrease_hat_correct_overall_percent_decrease_correct_l203_203252

-- Define original and sale prices
def original_price_trouser : ℝ := 100
def original_price_shirt : ℝ := 50
def original_price_shoes : ℝ := 30
def original_price_jacket : ℝ := 75
def original_price_hat : ℝ := 40

def sale_price_trouser : ℝ := 20
def sale_price_shirt : ℝ := 35
def sale_price_shoes : ℝ := 25
def sale_price_jacket : ℝ := 60
def sale_price_hat : ℝ := 30

-- Define the percent decrease formula
def percent_decrease (original sale : ℝ) : ℝ := ((original - sale) / original) * 100

-- Calculate the individual percent decreases
def percent_decrease_trouser := percent_decrease original_price_trouser sale_price_trouser
def percent_decrease_shirt := percent_decrease original_price_shirt sale_price_shirt
def percent_decrease_shoes := percent_decrease original_price_shoes sale_price_shoes
def percent_decrease_jacket := percent_decrease original_price_jacket sale_price_jacket
def percent_decrease_hat := percent_decrease original_price_hat sale_price_hat

-- Calculate the total costs
def original_total_cost : ℝ :=
  original_price_trouser + original_price_shirt + original_price_shoes + original_price_jacket + original_price_hat

def sale_total_cost : ℝ :=
  sale_price_trouser + sale_price_shirt + sale_price_shoes + sale_price_jacket + sale_price_hat

-- Calculate the overall percent decrease
def overall_percent_decrease : ℝ :=
  percent_decrease original_total_cost sale_total_cost

-- Assert the expected results as the theorem statements
theorem percent_decrease_trouser_correct :
  percent_decrease_trouser = 80 := by sorry

theorem percent_decrease_shirt_correct :
  percent_decrease_shirt = 30 := by sorry

theorem percent_decrease_shoes_correct :
  percent_decrease_shoes = 16.67 := by sorry

theorem percent_decrease_jacket_correct :
  percent_decrease_jacket = 20 := by sorry

theorem percent_decrease_hat_correct :
  percent_decrease_hat = 25 := by sorry

theorem overall_percent_decrease_correct :
  overall_percent_decrease = 42.37 := by sorry

end percent_decrease_trouser_correct_percent_decrease_shirt_correct_percent_decrease_shoes_correct_percent_decrease_jacket_correct_percent_decrease_hat_correct_overall_percent_decrease_correct_l203_203252


namespace cost_of_article_l203_203393

variable (C : ℝ) -- Cost of the article

-- Conditions
variable (G1 : ℝ := 350 - C)
variable (G2 : ℝ := 340 - C)
variable h1 : G1 = 1.04 * G2

theorem cost_of_article : C = 90 :=
by
  -- Conditions used as given facts
  have hG1 : G1 = 350 - C := rfl
  have hG2 : G2 = 340 - C := rfl
  have h_eq : 350 - C = 1.04 * (340 - C) := h1
  
  -- Placeholder for proof steps leading to the conclusion
  sorry

end cost_of_article_l203_203393


namespace S7_minus_S6_mod_1000_l203_203044

def is_m_free (m n : ℕ) : Prop :=
  n ≤ (Nat.factorial m) ∧ ∀ i ∈ Finset.range (m + 1), Nat.gcd i n = 1

def S (k : ℕ) : ℕ :=
  (Finset.range (Nat.factorial k + 1)).filter (is_m_free k).sum (λ x, x^2)

theorem S7_minus_S6_mod_1000 :
  ((S 7) - (S 6)) % 1000 = 80 :=
sorry

end S7_minus_S6_mod_1000_l203_203044


namespace intersection_A_B_l203_203753

-- Define the sets A and B
def set_A : Set ℝ := {y | ∃ x : ℝ, y = -x^2 - 2 * x}
def set_B : Set ℝ := {x | ∃ y : ℝ, y = sqrt (x + 1)}

-- State the theorem for the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {x | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end intersection_A_B_l203_203753


namespace expected_points_52_cards_l203_203062

theorem expected_points_52_cards :
  let score (cards : List (Nat × Nat)) (card : Nat × Nat) :=
    cards.takeWhile (fun p => p.1 = card.1) |>.length in
  let total_score (deck : List (Nat × Nat)) : Nat :=
    deck.foldl (fun acc card => acc + score acc card) 0 in
  (expected (λ deck : List (Nat × Nat), total_score deck) [{suit: s, card: c} | (s, c) ← List.range' 1 5] = 624 / 41) := 
sorry

end expected_points_52_cards_l203_203062


namespace distinct_integers_sum_l203_203275

theorem distinct_integers_sum {p q r s t : ℤ} 
    (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t) 
    (h9 : r ≠ s) (h10 : r ≠ t) (h11 : s ≠ t) : 
  p + q + r + s + t = 35 := 
sorry

end distinct_integers_sum_l203_203275


namespace two_digit_primes_with_ones_digit_three_count_l203_203765

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203765


namespace tan_105_eq_neg2_sub_sqrt3_l203_203614

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203614


namespace find_k_value_l203_203190

open EuclideanSpace

noncomputable def a : Fin 2 → ℝ := ![2, 1]
noncomputable def b (k : ℝ) : Fin 2 → ℝ := ![-1, k - 1]
noncomputable def c (k : ℝ) : Fin 2 → ℝ := ![1, k]

theorem find_k_value : ∀ k : ℝ, (a + b k = c k) ∧ (a ⬝ b k = 0) → k = 3 := by
  sorry

end find_k_value_l203_203190


namespace tan_105_l203_203549

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203549


namespace range_of_x_l203_203202

theorem range_of_x {x : ℝ} : (sqrt ((5 - x) ^ 2) = x - 5) → (x ≥ 5) :=
by
  sorry

end range_of_x_l203_203202


namespace regular_15gon_symmetry_l203_203422

theorem regular_15gon_symmetry :
  ∀ (L R : ℕ),
  (L = 15) →
  (R = 24) →
  L + R = 39 :=
by
  intros L R hL hR
  exact sorry

end regular_15gon_symmetry_l203_203422


namespace sum_of_infinite_squares_areas_l203_203429

theorem sum_of_infinite_squares_areas (side : ℝ) (h : side = 4) : 
  let first_square_area := side ^ 2
  let sum_of_areas := first_square_area / (1 - 1/2)
  sum_of_areas = 32 :=
by
  /- Given the side of the first square is 4 cm -/
  have h₁ : side = 4 := h
  
  /- Calculating the area of the first square -/
  let first_square_area : ℝ := side ^ 2
  have h₂ : first_square_area = 4 ^ 2 := by rw [h₁]

  /- Prove the sum of infinite geometric series -/
  let series_sum : ℝ := first_square_area / (1 - 1 / 2)
  
  /- Assert the sum of the areas is 32 -/
  have h₃ : series_sum = 32 := 
  by
    rw [← h₂]
    sorry
  
  exact h₃

end sum_of_infinite_squares_areas_l203_203429


namespace polynomial_quotient_l203_203126

open Polynomial

noncomputable def dividend : ℤ[X] := 5 * X^4 - 9 * X^3 + 3 * X^2 + 7 * X - 6
noncomputable def divisor : ℤ[X] := X - 1

theorem polynomial_quotient :
  dividend /ₘ divisor = 5 * X^3 - 4 * X^2 + 7 * X + 7 :=
by
  sorry

end polynomial_quotient_l203_203126


namespace toys_left_after_sale_l203_203114

def initial_toys : ℕ := 35
def saturday_fraction : ℚ := 1/2
def sunday_fraction : ℚ := 3/5

theorem toys_left_after_sale :
  let sold_on_saturday := (saturday_fraction * initial_toys : ℚ).toNat,
      remaining_after_saturday := initial_toys - sold_on_saturday,
      sold_on_sunday := (sunday_fraction * remaining_after_saturday : ℚ).toNat,
      remaining_after_sunday := remaining_after_saturday - sold_on_sunday in
  remaining_after_sunday = 8 := by
  -- proof goes here
  sorry

end toys_left_after_sale_l203_203114


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203984

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203984


namespace range_of_x_l203_203204

theorem range_of_x (x : ℝ) : sqrt ((5 - x) ^ 2) = x - 5 → x ≥ 5 :=
by
  sorry

end range_of_x_l203_203204


namespace find_a_and_x_l203_203354

theorem find_a_and_x (a x : ℚ) (h1 : 0 < x) (h2 : sqrt x = 2 * a - 3) (h3 : sqrt x = 5 - a) :
  a = 8 / 3 ∧ x = 49 / 9 :=
by sorry

end find_a_and_x_l203_203354


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203513

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203513


namespace maximum_value_l203_203269

noncomputable def max_expression (a b c d e : ℝ) :=
  ac + 3 * bc + 4 * cd + 8 * ce

theorem maximum_value
  (a b c d e : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_sum : a^2 + b^2 + c^2 + d^2 + e^2 = 2024) :
  let N := max_expression a b c d e in
  N + a + b + c + d + e = 48 + 3028 * Real.sqrt 10 :=
sorry

end maximum_value_l203_203269


namespace count_two_digit_primes_with_ones_digit_three_l203_203789

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203789


namespace xy_plus_four_is_square_l203_203174

theorem xy_plus_four_is_square (x y : ℕ) (h : ((1 / (x : ℝ)) + (1 / (y : ℝ)) + 1 / (x * y : ℝ)) = (1 / (x + 4 : ℝ) + 1 / (y - 4 : ℝ) + 1 / ((x + 4) * (y - 4) : ℝ))) : 
  ∃ (k : ℕ), xy + 4 = k^2 :=
by
  sorry

end xy_plus_four_is_square_l203_203174


namespace sasha_picks_24_leaves_l203_203441

def num_apple_trees := 17
def num_poplar_trees := 20
def starting_apple_tree := 8

theorem sasha_picks_24_leaves :
  ∃ n : ℕ, n = 24 ∧ (num_poplar_trees + (num_apple_trees - starting_apple_tree + 1)) = n :=
begin
  sorry
end

end sasha_picks_24_leaves_l203_203441


namespace min_people_share_birthday_l203_203715

theorem min_people_share_birthday (a : ℕ) (days_in_leap_year : ℕ)
  (h1 : days_in_leap_year = 366)
  (h2 : ∀ group : Finset ℕ, group.card = a → ∃ (b1 b2 : ℕ), b1 ≠ b2 ∧ birthday_of b1 = birthday_of b2) : a = 367 := 
by
  -- Since proof steps are not required, we immediately use 'sorry' to skip the proof.
  sorry

end min_people_share_birthday_l203_203715


namespace tan_105_degree_l203_203571

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203571


namespace num_two_digit_primes_with_ones_digit_3_l203_203952

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203952


namespace volume_PQRS_l203_203236

noncomputable def volume_of_tetrahedron (P Q R S : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  1 / 6 * abs (det ![
    P.1 - Q.1, P.2 - Q.2, P.3 - Q.3, 1,
    Q.1 - R.1, Q.2 - R.2, Q.3 - R.3, 1,
    R.1 - S.1, R.2 - S.2, R.3 - S.3, 1,
    S.1 - P.1, S.2 - P.2, S.3 - P.3, 1
  ])

theorem volume_PQRS :
  ∀ {K L M N P Q R S : EuclideanSpace ℝ (Fin 3)},
    dist K L = 9 → dist M N = 9 → dist K M = 15 → dist L N = 15 →
    dist K N = 16 → dist L M = 16 →
    P = triangle_incenter K L M → 
    Q = triangle_incenter K L N → 
    R = triangle_incenter K M N → 
    S = triangle_incenter L M N →
    volume_of_tetrahedron P Q R S = 4.85 :=
by
  intros
  -- Proof skipped
  sorry

end volume_PQRS_l203_203236


namespace rectangle_quadratic_eq_l203_203048

variable {L W : ℝ}

theorem rectangle_quadratic_eq (h1 : L + W = 15) (h2 : L * W = 2 * W^2) : 
    (∃ x : ℝ, (x - L) * (x - W) = x^2 - 15 * x + 50) :=
by
  sorry

end rectangle_quadratic_eq_l203_203048


namespace solution_l203_203268

noncomputable def length_of_PQ (P Q : ℝ × ℝ) :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def problem (P Q R : ℝ × ℝ) : Prop :=
  R = (8, 6) ∧ 
  P.2 = 15 * P.1 / 8 ∧ 
  Q.2 = 3 * Q.1 / 10 ∧ 
  R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
  length_of_PQ P Q = 60 / 7

theorem solution : ∃ (m n : ℕ), gcd m n = 1 ∧ 60 / 7 = m / n ∧ m + n = 67 :=
by
  use 60
  use 7
  split
  . sorry
  split
  . sorry
  . sorry

end solution_l203_203268


namespace longer_segment_probability_l203_203423

noncomputable def probability_longer_segment_at_least_2x_shorter (x : ℝ) (h : 0 < x) : ℝ :=
  2 / (2 * x + 1)

theorem longer_segment_probability (x : ℝ) (h : 0 < x) :
  probability_longer_segment_at_least_2x_shorter x h = 2 / (2 * x + 1) := 
begin
  sorry
end

end longer_segment_probability_l203_203423


namespace count_two_digit_primes_ending_with_3_l203_203847

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203847


namespace find_n_l203_203222

theorem find_n :
  ∃ n : ℕ, ∀ (a b c : ℕ), a + b + c = 200 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (n = a + b * c) ∧ (n = b + c * a) ∧ (n = c + a * b) → n = 199 :=
by {
  sorry
}

end find_n_l203_203222


namespace find_factor_l203_203430

theorem find_factor (n f : ℕ) (h1 : n = 122) (h2 : n * f - 138 = 106) : f = 2 := 
by {
  sorry // Proof is to be filled in.
}

end find_factor_l203_203430


namespace real_when_k_is_complex_when_k_is_purely_imaginary_when_k_is_zero_when_k_is_l203_203694

noncomputable def complex_number (k : ℝ) : ℂ :=
  (k^2 - 3 * k - 4 : ℝ) + (k^2 - 5 * k - 6 : ℝ) * complex.I

theorem real_when_k_is (k : ℝ) : 
  (k = 6 ∨ k = -1) → (imag_part (complex_number k) = 0) :=
begin
  sorry
end

theorem complex_when_k_is (k : ℝ) :
  (k ≠ 6 ∧ k ≠ -1) → (imag_part (complex_number k) ≠ 0) :=
begin
  sorry
end

theorem purely_imaginary_when_k_is (k : ℝ) : 
  k = 4 → (re (complex_number k) = 0) :=
begin
  sorry
end

theorem zero_when_k_is (k : ℝ) :
  k = -1 → (complex_number k = 0) :=
begin
  sorry
end

end real_when_k_is_complex_when_k_is_purely_imaginary_when_k_is_zero_when_k_is_l203_203694


namespace two_digit_primes_end_in_3_l203_203917

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203917


namespace a1_plus_a9_l203_203704

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a1_plus_a9 : (a 1) + (a 9) = 19 := by
  sorry

end a1_plus_a9_l203_203704


namespace sweater_markup_percentage_l203_203011

variables (W R : ℝ)
variables (h1 : 0.30 * R = 1.40 * W)

theorem sweater_markup_percentage :
  (R = (1.40 / 0.30) * W) →
  (R - W) / W * 100 = 367 := 
by
  intro hR
  sorry

end sweater_markup_percentage_l203_203011


namespace total_spent_l203_203258

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end total_spent_l203_203258


namespace isosceles_triangle_base_length_l203_203342

theorem isosceles_triangle_base_length
  (perimeter_eq_triangle : ℕ)
  (perimeter_isosceles_triangle : ℕ)
  (side_eq_triangle_isosceles : ℕ)
  (side_eq : side_eq_triangle_isosceles = perimeter_eq_triangle / 3)
  (perimeter_eq : perimeter_isosceles_triangle = 2 * side_eq_triangle_isosceles + 15) :
  15 = perimeter_isosceles_triangle - 2 * side_eq_triangle_isosceles :=
sorry

end isosceles_triangle_base_length_l203_203342


namespace shanghai_expo_revenue_l203_203437

noncomputable def expo_revenue (visitors: ℕ) (average_spending: ℕ) : ℝ :=
  (visitors * average_spending : ℝ) / 10^9

theorem shanghai_expo_revenue :
  expo_revenue 70000000 1500 ≈ 110 :=
by
  -- Calculate the revenue step
  have revenue : ℝ := 70000000 * 1500
  -- Convert to billions of yuan
  have revenue_billion : ℝ := revenue / 10^9
  -- Revenue in billions of yuan should be approximately 110 considering significant figures
  have revenue_approx : revenue_billion ≈ 110 := sorry
  exact revenue_approx

end shanghai_expo_revenue_l203_203437


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203606

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203606


namespace count_two_digit_primes_with_ones_digit_3_l203_203804

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203804


namespace length_of_goods_train_l203_203015

theorem length_of_goods_train
  (speed_km_hr : ℕ)
  (platform_length_meters : ℕ)
  (time_seconds : ℕ)
  (speed_m_s : ℕ := speed_km_hr * 1000 / 3600)
  (distance_covered_meters : ℕ := speed_m_s * time_seconds) :
  speed_km_hr = 72 → platform_length_meters = 290 → time_seconds = 26 → 
  distance_covered_meters - platform_length_meters = 230 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [speed_m_s, distance_covered_meters]
  sorry

end length_of_goods_train_l203_203015


namespace find_lambda_l203_203170

noncomputable section

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b : V} (λ : ℝ)

-- Conditions: Vectors a and b are not parallel and the vector (λ • a + b) is parallel to (a + 2 • b)
def vectors_not_parallel (a b : V) : Prop :=
  ¬ (∃ (k : ℝ), k • a = b)

def parallel (u v : V) : Prop :=
  ∃ (μ : ℝ), u = μ • v

def given_conditions (a b : V) (λ : ℝ) : Prop :=
  vectors_not_parallel a b ∧ parallel (λ • a + b) (a + 2 • b)

-- Theorem: Given the conditions, prove that λ = 1/2
theorem find_lambda (a b : V) (h : given_conditions a b λ) : λ = 1 / 2 :=
begin
  sorry
end

end find_lambda_l203_203170


namespace sqrt_inequality_l203_203299

theorem sqrt_inequality : sqrt 3 + sqrt 7 < 2 * sqrt 5 := by
  have h₁ : sqrt 3 + sqrt 7 > 0 := sorry
  have h₂ : 2 * sqrt 5 > 0 := sorry
  have sqr_lhs : (sqrt 3 + sqrt 7) ^ 2 = 3 + 7 + 2 * sqrt 21 :=
    by sorry
  have sqr_rhs : (2 * sqrt 5) ^ 2 = 20 := by sorry
  have h₃ : 10 + 2 * sqrt 21 < 20 := by sorry
  have h₄ : 2 * sqrt 21 < 10 := by sorry
  have h₅ : sqrt 21 < 5 := by sorry
  show sqrt 3 + sqrt 7 < 2 * sqrt 5 from sorry

end sqrt_inequality_l203_203299


namespace two_digit_primes_end_in_3_l203_203922

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203922


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203995

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203995


namespace number_of_ferns_l203_203255

variable (F : Nat) -- Let F be the number of ferns.

-- Conditions
axiom five_palms : Nat := 5
axiom seven_succulents : Nat := 7
axiom total_plants_desired : Nat := 24
axiom plants_needed : Nat := 9

theorem number_of_ferns (h : F + five_palms + seven_succulents = total_plants_desired - plants_needed) : F = 3 :=
sorry

end number_of_ferns_l203_203255


namespace count_two_digit_primes_with_ones_digit_three_l203_203782

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203782


namespace min_degree_for_horizontal_asymptote_l203_203103

-- Define a polynomial with the specified degree
def numerator : Polynomial ℝ := 3 * Polynomial.monomial 7 1 - 5 * Polynomial.monomial 3 1 + 2 * Polynomial.monomial 1 1 - Polynomial.C 4

-- Define the function to compute the degree of a polynomial
def degree (p : Polynomial ℝ) : ℕ :=
  p.natDegree

-- Define the condition that the rational function has a horizontal asymptote
def has_horizontal_asymptote (num denom : Polynomial ℝ) : Prop :=
  degree denom >= degree num

-- The main theorem: The smallest possible degree of p(x) for the function to have a horizontal asymptote is 7
theorem min_degree_for_horizontal_asymptote (p : Polynomial ℝ) (h : has_horizontal_asymptote numerator p) : 
  degree p >= 7 :=
sorry

end min_degree_for_horizontal_asymptote_l203_203103


namespace eliana_steps_ratio_l203_203674

-- Defining the given conditions and what needs to be proved
theorem eliana_steps_ratio :
  let first_day_steps := 200 + 300 in 
  let second_day_steps := 500 in 
  let third_day_steps := second_day_steps + 100 in 
  let total_steps := first_day_steps + second_day_steps + third_day_steps in
  total_steps = 1600 →
  second_day_steps / first_day_steps = 1 :=
by 
  sorry

end eliana_steps_ratio_l203_203674


namespace count_two_digit_primes_ending_with_3_l203_203851

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203851


namespace sufficient_but_not_necessary_condition_for_q_l203_203146

theorem sufficient_but_not_necessary_condition_for_q (k : ℝ) :
  (∀ x : ℝ, x ≥ k → x^2 - x > 2) ∧ (∃ x : ℝ, x < k ∧ x^2 - x > 2) ↔ k > 2 :=
sorry

end sufficient_but_not_necessary_condition_for_q_l203_203146


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203963

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203963


namespace area_of_rectangle_eq_18_l203_203108

-- Geometric definitions and conditions
inductive Point : Type
| A | B | C | D | M | N deriving decidable_eq

def AC (p: Point) : ℝ :=
  match p with
  | Point.B => 2
  | Point.M => 1
  | Point.N => 2
  | Point.D => 1
  | _ => 0  -- Assume 0 for other points for simplicity

def perpendicular_to_diagonal : Point → Prop :=
  λ p, p = Point.M ∨ p = Point.N

noncomputable def length_AC : ℝ :=
  AC Point.B + AC Point.M + AC Point.N + AC Point.D

-- Theorem statement
theorem area_of_rectangle_eq_18 :
  length_AC = 6 → (∀ p, perpendicular_to_diagonal p → p = Point.M ∨ p = Point.N) →
  ∃ area : ℝ, area = 18 :=
by
  intros
  have h1 : length_AC = 6, from ‹length_AC = 6›
  use 18
  sorry

end area_of_rectangle_eq_18_l203_203108


namespace count_two_digit_primes_ending_in_3_l203_203833

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203833


namespace problem_proof_l203_203261

theorem problem_proof (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
  (h2 : a + b + c + d = m^2) 
  (h3 : max (max a b) (max c d) = n^2) : 
  m = 9 ∧ n = 6 :=
by
  sorry

end problem_proof_l203_203261


namespace problem_l203_203159

theorem problem (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : f 1 = f 3) 
  (h2 : f 1 > f 4) 
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c) :
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l203_203159


namespace tan_105_l203_203488

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203488


namespace volume_of_tetrahedron_PQRS_l203_203238

theorem volume_of_tetrahedron_PQRS :
  let K := (0, 0, 0)
  let L := (9, 0, 0)
  let M := some (coordinates such that KM = 15, LM = 16, and form valid tetrahedron KLMN)
  let N := some (coordinates such that KN = 16, LN = 15, MN = 9 and form valid tetrahedron KLMN)
  let P := (some_centroid K L M)
  let Q := (some_centroid K L N)
  let R := (some_centroid K M N)
  let S := (some_centroid L M N)
  volume (P, Q, R, S) = 4.85 :=
sorry

end volume_of_tetrahedron_PQRS_l203_203238


namespace fido_leash_area_fraction_l203_203677

theorem fido_leash_area_fraction (r : ℝ) (s : ℝ) (h : r = s * (Real.sqrt (2 + Real.sqrt 2) / 2)) :
  let area_octagon := 4 * Real.sqrt 2 * r^2,
      area_circle := Real.pi * r^2,
      fraction := area_circle / area_octagon,
      a := 2,
      b := 8
  in fraction = (Real.sqrt a / b) * Real.pi ∧ a * b = 16 :=
by sorry

end fido_leash_area_fraction_l203_203677


namespace simplify_and_rationalize_l203_203306

theorem simplify_and_rationalize : (1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5) :=
by sorry

end simplify_and_rationalize_l203_203306


namespace equal_probabilities_hearts_clubs_l203_203110

/-- Define the total number of cards in a standard deck including two Jokers -/
def total_cards := 52 + 2

/-- Define the counts of specific card types -/
def num_jokers := 2
def num_spades := 13
def num_tens := 4
def num_hearts := 13
def num_clubs := 13

/-- Define the probabilities of drawing specific card types -/
def prob_joker := num_jokers / total_cards
def prob_spade := num_spades / total_cards
def prob_ten := num_tens / total_cards
def prob_heart := num_hearts / total_cards
def prob_club := num_clubs / total_cards

theorem equal_probabilities_hearts_clubs :
  prob_heart = prob_club :=
by
  sorry

end equal_probabilities_hearts_clubs_l203_203110


namespace derivative_at_pi_l203_203179

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_at_pi :
  deriv f π = -1 / (π^2) :=
sorry

end derivative_at_pi_l203_203179


namespace tan_105_eq_neg2_sub_sqrt3_l203_203525

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203525


namespace profit_share_difference_l203_203045

theorem profit_share_difference (total_profit : ℝ) (ratio_X : ℝ) (ratio_Y : ℝ) :
  total_profit = 700 ∧ ratio_X = 1/2 ∧ ratio_Y = 1/3 → 
  let sum_ratio_parts := (3/6) + (2/6),
      value_of_one_part := total_profit / sum_ratio_parts,
      X_share := 3 * value_of_one_part,
      Y_share := 2 * value_of_one_part 
  in X_share - Y_share = 140 := 
by
  intros h,
  cases h with h1 h2,
  cases h2 with hx hy,
  let sum_ratio_parts := (3/6) + (2/6),
  let value_of_one_part := h1 / sum_ratio_parts,
  let X_share := 3 * value_of_one_part,
  let Y_share := 2 * value_of_one_part,
  have h3 : sum_ratio_parts = 5/6 := by sorry,
  have h4 : value_of_one_part = 700 / (5/6) := by sorry,
  have h5 : X_share = 3 * (700 / (5/6)) := by sorry,
  have h6 : Y_share = 2 * (700 / (5/6)) := by sorry,
  have h7 : X_share - Y_share = (3 * (700 / (5/6))) - (2 * (700 / (5/6))) := by sorry,
  have h8 : X_share - Y_share = (3 - 2) * (700 / (5/6)) := by sorry,
  have h9 : X_share - Y_share = 140 := by sorry,
  exact h9

end profit_share_difference_l203_203045


namespace tan_105_degree_l203_203596

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203596


namespace tan_105_eq_neg2_sub_sqrt3_l203_203526

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203526


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203979

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203979


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203899

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203899


namespace find_k_value_l203_203212

noncomputable def solve_for_k (k : ℚ) : Prop :=
  ∃ x : ℚ, (x = 1) ∧ (3 * x + (2 * k - 1) = x - 6 * (3 * k + 2))

theorem find_k_value : solve_for_k (-13 / 20) :=
  sorry

end find_k_value_l203_203212


namespace count_two_digit_primes_ending_with_3_l203_203846

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203846


namespace base12_division_remainder_l203_203386

theorem base12_division_remainder :
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3 in
  n % 9 = 8 :=
by
  let n := 2 * (12^3) + 5 * (12^2) + 4 * 12 + 3
  show n % 9 = 8
  sorry

end base12_division_remainder_l203_203386


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203505

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203505


namespace range_of_a_l203_203334

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a^2-1)^x > (a^2-1)^y)
  ↔ 1 < |a| ∧ |a| < sqrt 2 :=
by
  sorry

end range_of_a_l203_203334


namespace count_two_digit_primes_with_ones_digit_3_l203_203806

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203806


namespace tan_105_eq_neg2_sub_sqrt3_l203_203579

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203579


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203975

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203975


namespace subset_exists_l203_203297

-- Definitions
def subset_of_5_with_divisibility (s : Finset ℕ) :=
  ∃ a b c d e ∈ s, (a ∣ b) ∧ (b ∣ c) ∧ (c ∣ d) ∧ (d ∣ e)

def subset_of_5_without_divisibility (s : Finset ℕ) :=
  ∃ a b c d e ∈ s, (¬(a ∣ b) ∧ ¬(a ∣ c) ∧ ¬(a ∣ d) ∧ ¬(a ∣ e) ∧ 
                     ¬(b ∣ c) ∧ ¬(b ∣ d) ∧ ¬(b ∣ e) ∧ 
                     ¬(c ∣ d) ∧ ¬(c ∣ e) ∧ ¬(d ∣ e))

-- Main statement
theorem subset_exists (s : Finset ℕ) (h : s.card = 17) :
  subset_of_5_with_divisibility s ∨ subset_of_5_without_divisibility s :=
sorry

end subset_exists_l203_203297


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203964

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203964


namespace fraction_multiplication_l203_203207

theorem fraction_multiplication :
  ((2 / 5) * (5 / 7) * (7 / 3) * (3 / 8) = 1 / 4) :=
sorry

end fraction_multiplication_l203_203207


namespace two_digit_primes_end_in_3_l203_203919

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203919


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203508

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203508


namespace part_I_part_II_l203_203744

noncomputable def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

theorem part_I (x : ℝ) : ∃ (s : set ℝ), s = {x | x ≤ -2} ∧ ∀ x, f x ≥ 2 ↔ x ∈ s :=
sorry

theorem part_II (m : ℝ) (M : ℝ) (hM : M = 8) : ∃ r, r = {m | m ≤ 9} ∧ ∀ m, (∃ x, x ^ 2 + 2 * x + m ≤ M) ↔ m ∈ r :=
sorry

end part_I_part_II_l203_203744


namespace tan_105_eq_neg2_sub_sqrt3_l203_203522

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203522


namespace find_a_l203_203188

-- Definitions and conditions from the problem
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}
def intersection_is_three (a : ℝ) : Prop := M a ∩ N a = {3}

-- The theorem we want to prove
theorem find_a (a : ℝ) (h : intersection_is_three a) : a = 4 :=
by
  sorry

end find_a_l203_203188


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203967

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203967


namespace two_digit_primes_end_in_3_l203_203912

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203912


namespace tan_105_l203_203496

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203496


namespace tan_105_degree_l203_203593

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203593


namespace count_two_digit_primes_with_ones_3_l203_203886

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203886


namespace smallest_n_exists_l203_203102

theorem smallest_n_exists : ∃ n : ℕ, (∀ (l : List ℕ), l.length = n → 
  ∃ (subl : List ℕ), subl.length = 18 ∧ (list.sum subl) % 18 = 0) ∧ n = 35 :=
sorry

end smallest_n_exists_l203_203102


namespace sum_max_min_dist_l203_203267

noncomputable def ellipse_condition (x y b : ℝ) := (x^2 / 25) + (y^2 / b^2) = 1

theorem sum_max_min_dist (x y b c m n : ℝ) : 
  ellipse_condition x y b → 
  c = sqrt (25 - b^2) →
  m = 5 + c →
  n = 5 - c →
  m + n = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end sum_max_min_dist_l203_203267


namespace intersection_equiv_l203_203751

def A : Set ℝ := { x : ℝ | x > 1 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def C : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem intersection_equiv : A ∩ B = C :=
by
  sorry

end intersection_equiv_l203_203751


namespace john_average_increase_l203_203250

noncomputable def average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

theorem john_average_increase :
  let scores := [92, 88, 91]
  let initialAverage := average scores
  let newScores := scores ++ [95]
  let newAverage := average newScores
  newAverage - initialAverage = 1.1667 :=
by
  let scores := [92, 88, 91]
  let initialAverage := average scores
  let newScores := scores ++ [95]
  let newAverage := average newScores
  have : initialAverage = 271 / 3 := sorry
  have : newAverage = 366 / 4 := sorry
  exact calc
    newAverage - initialAverage
      = 366 / 4 - 271 / 3 : by rw [<common expression equalities>]
      ... = 1.1667 : sorry

end john_average_increase_l203_203250


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203604

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203604


namespace sum_of_first_50_terms_l203_203152

-- Conditions: Define the sequence as per the given problem
def seq : ℕ → ℕ
| 0 := 1
| n + 1 := if h : n + 1 + 1 ≤ 2 ^ ((n + 1 + 1).nat_root 2) then 2 ^ (nat_root 2 (n + 1) - 1) else seq (n - (2 ^ (nat_root 2 (n + 1) - 1)) + 1)

-- Sum of first n terms of the sequence
def sum_seq (n : ℕ) : ℕ :=
(list.sum (list.range n).map seq)

-- Statement to prove
theorem sum_of_first_50_terms : sum_seq 50 = 1044 := sorry

end sum_of_first_50_terms_l203_203152


namespace count_two_digit_primes_with_ones_3_l203_203883

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203883


namespace two_digit_primes_end_in_3_l203_203913

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203913


namespace positive_integer_solution_l203_203134

theorem positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 = y^2 + 71) :
  x = 6 ∧ y = 35 :=
by
  sorry

end positive_integer_solution_l203_203134


namespace count_two_digit_primes_with_ones_digit_3_l203_203813

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203813


namespace paths_A_to_D_l203_203691

noncomputable def num_paths_from_A_to_D : ℕ := 
  2 * 2 * 2 + 1

theorem paths_A_to_D : num_paths_from_A_to_D = 9 := 
by
  sorry

end paths_A_to_D_l203_203691


namespace mixture_carbonated_water_fraction_l203_203056

theorem mixture_carbonated_water_fraction (V : ℝ) : 
  (0.1999999999999997 * 0.80 + 0.8000000000000003 * 0.55) * V / V * 100 = 60 := 
by
  have h1 : 0.1999999999999997 * 0.80 = 0.15999999999999976 := by norm_num
  have h2 : 0.8000000000000003 * 0.55 = 0.44000000000000017 := by norm_num
  have h3 : 0.15999999999999976 + 0.44000000000000017 = 0.5999999999999999 := by norm_num
  rw [h1, h2, h3]
  have h4 : 0.5999999999999999 * 100 = 59.99999999999999 := by norm_num
  rw [h4]
  linarith

end mixture_carbonated_water_fraction_l203_203056


namespace circle_and_tangent_lines_exist_l203_203147

theorem circle_and_tangent_lines_exist
  (D E : ℝ)
  (h_symm : ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + 3 = 0 ↔ (x + 1) ^ 2 + (y + 2) ^ 2 = 2)
  (h_center_quadrant : ∃ x y : ℝ, -x = D / 2 ∧ -y = E / 2 ∧ 0 < x ∧ y < 0)
  (h_radius : ∃ r : ℝ, r = sqrt 2) :
  (x^2 + y^2 - 4 * x + 2 * y + 3 = 0) ∧ 
    (∃ l : ℝ, l = (λ k : ℝ, if k = 0 then -2 * x + 2 * y + sqrt 10 else ((-2 + sqrt 6) * x / 2) + (2 * y) = 0)) := 
by sorry

end circle_and_tangent_lines_exist_l203_203147


namespace two_digit_primes_with_ones_digit_3_l203_203852

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203852


namespace tan_add_tan_105_eq_l203_203640

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203640


namespace two_digit_primes_ending_in_3_eq_6_l203_203926

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203926


namespace limit_sqrt_tan_l203_203396

open Real

theorem limit_sqrt_tan : 
  tendsto (λ x: ℝ , (sqrt(x^2 - x + 1) - 1) / tan(π * x)) (nhds 1) (nhds (1 / (2 * π))) :=
by sorry

end limit_sqrt_tan_l203_203396


namespace star_perimeter_l203_203264

-- Let ABCDEF be an equiangular convex hexagon with perimeter 2.
-- We define a perimeter function for a hexagon
def perimeter_hexagon (ABCDEF : ℕ) : ℝ := 2

-- Definition of an equiangular convex hexagon
structure EquiangularConvexHexagon :=
  (side_lengths : Fin 6 → ℝ)
  (equiangular : ∀ i, i < 6 → ∠ (1, 1, 1) = 120)
  (convex : ∀ i j k, i < j → i < k → ∠ (1, 1, 1) ≤ 180)
  (perimeter_eq_two : (Fin 6 → ℝ) := perimeter_hexagon)

-- Let s be the perimeter of the star formed by extending the sides of this hexagon.
theorem star_perimeter (h : EquiangularConvexHexagon) : h.perimeter_eq_two * 2 = 4 :=
  sorry

end star_perimeter_l203_203264


namespace remainder_base12_div_9_l203_203382

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l203_203382


namespace max_min_S_l203_203150

noncomputable section

open Real

variables {n : ℕ} (x : Fin n -> ℝ) (t : ℕ)

axiom (n_pos : n ≥ 2)
axiom (x_nonneg : ∀ i, 0 ≤ x i)
axiom (sum_square_eq_one : (∑ i, (x i) ^ 2) = 1)
axiom (t_def : t = floor (sqrt n) ∨ t = floor (sqrt n) + 1)

/-- Definition of maximum value in x -/
def M := ⨆ i, x i

/-- Sum for S calculation -/
def S := n * M ^ 2 + 2 * (∑ i j, if i < j then x i * x j else 0)

theorem max_min_S : 
  (∑ i, (x i) ^ 2 = 1) → 
  n ≥ 2 → 
  M = ⨆ i, x i → 
  t = floor (sqrt n) ∨ t = floor (sqrt n) + 1 → 
  (n / t + t - 1) ≤ S × (S ≤ sqrt n + n - 1) :=
by
  assume n_pos x_nonneg sum_square_eq_one M_def t_def
  sorry

end max_min_S_l203_203150


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203503

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203503


namespace two_digit_primes_with_ones_digit_three_count_l203_203764

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203764


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l203_203997

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l203_203997


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203888

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203888


namespace total_spent_is_13_l203_203658

-- Let cost_cb represent the cost of the candy bar
def cost_cb : ℕ := 7

-- Let cost_ch represent the cost of the chocolate
def cost_ch : ℕ := 6

-- Define the total cost as the sum of cost_cb and cost_ch
def total_cost : ℕ := cost_cb + cost_ch

-- Theorem to prove the total cost equals $13
theorem total_spent_is_13 : total_cost = 13 := by
  sorry

end total_spent_is_13_l203_203658


namespace tan_105_eq_neg2_sub_sqrt3_l203_203623

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203623


namespace isosceles_triangle_l203_203214

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b^2 - 2 * b * c + c^2) = 0) : 
  (a = b) ∨ (b = c) :=
by sorry

end isosceles_triangle_l203_203214


namespace tan_105_eq_neg2_sub_sqrt3_l203_203583

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203583


namespace absolute_value_inequality_solution_set_l203_203348

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l203_203348


namespace tan_105_l203_203551

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203551


namespace sarah_investment_in_real_estate_l203_203305

-- Define the conditions
def total_investment : ℝ := 250000
def investment_ratio : ℝ := 6

-- Define the unknowns
def investment_in_mutual_funds (I_M : ℝ) : ℝ := I_M
def investment_in_real_estate (I_R : ℝ) : ℝ := investment_ratio * investment_in_mutual_funds I_M

-- Define the proof statement
theorem sarah_investment_in_real_estate (I_M : ℝ) (I_R : ℝ) :
  total_investment = I_M + investment_in_real_estate (investment_ratio * I_M) →
  I_R = investment_in_real_estate I_M →
  I_R = 214285.71 :=
by
  sorry

end sarah_investment_in_real_estate_l203_203305


namespace tan_105_l203_203460

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203460


namespace max_rectangles_l203_203058

theorem max_rectangles (k ℓ : ℕ) :
  ∀ (segments : list (ℝ × ℝ) × (ℝ × ℝ)), 
  (∀ xy ∈ segments, (xy.fst.1 = xy.snd.1 ∨ xy.fst.2 = xy.snd.2)) ∧ 
  (∀ xy ∈ segments, (∀ uv ∈ segments, xy ≠ uv → 
    (xy.fst.1 ≠ uv.fst.1 ∨ xy.fst.2 ≠ uv.snd.2 ∨ xy.snd.1 ≠ uv.fst.1 ∨ xy.snd.2 ≠ uv.snd.2))) ∧
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 < x ∧ x < 1) → 
    (∃ cnt : ℕ, (∀ xy ∈ segments, xy.fst.1 = x → cnt > k)) ∧ 
    (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → (0 < y ∧ y < 1) → 
    (∃ cnt : ℕ, (∀ xy ∈ segments, xy.fst.2 = y → cnt > ℓ)))) → 
  ∑ i in (segments.map (λ s, s.fst)).to_finset, 1 + 
  ∑ j in (segments.map (λ s, s.snd)).to_finset, 1 = k * ℓ :=
by sorry

end max_rectangles_l203_203058


namespace tan_105_degree_l203_203598

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203598


namespace checkerboard_area_sum_equal_l203_203672

theorem checkerboard_area_sum_equal
  (ABCD : ConvexQuadrilateral)
  (n : ℕ)
  (h_n : n = 8)
  (div_points : ∀ (s : Side ABCD), Fin n → Point)
  (checkerboard_pattern : ∀ (i j : Fin n), Cell) :
  (sum_area_black checkerboard_pattern div_points = sum_area_white checkerboard_pattern div_points) :=
sorry

end checkerboard_area_sum_equal_l203_203672


namespace question_correctness_l203_203427

variable (x : Fin 11 → ℝ)
variable (sorted_x : ∀ i j : Fin 11, i < j → x i ≤ x j)
variable (mean_x : (∑ i, x i) / 11 = 5)
variable (median_x : x 5 = 5)

theorem question_correctness (new_x : Fin 10 → ℝ)
  (new_sorted_x : ∀ i j : Fin 10, i < j → new_x i ≤ new_x j)
  (new_mean_x : ∑ i, new_x i / 10 = 5)
  (new_median_x : ¬(new_x 4 + new_x 5) / 2 = 5)
  (new_range_x : x 10 - x 0 = new_x 9 - new_x 0) :
  (new_mean_x = 5) ∧ (new_range_x = x 10 - x 0) :=
by
  sorry

end question_correctness_l203_203427


namespace card_arrangement_count_l203_203113

open List

theorem card_arrangement_count :
  let cards := (List.range 8).map (λ x, x + 1) in
  (∃ f : List ℕ → ℕ, 
    (∀ xs, xs.perm cards → 
      (f xs ∈ cards ∧ f xs ≠ 1 ∧ f xs ≠ 8) ∧
      ((remove_all [f xs] xs).sorted (≤) ∨ (remove_all [f xs] xs).sorted (λ a b, a ≥ b))) → 
    #((map_fun f) (perm.choose cards).to_list).nprod) = 60470 := sorry

end card_arrangement_count_l203_203113


namespace loss_per_metre_l203_203055

-- Definitions for given conditions
def TSP : ℕ := 15000           -- Total Selling Price
def CPM : ℕ := 40              -- Cost Price per Metre
def TMS : ℕ := 500             -- Total Metres Sold

-- Definition for the expected Loss Per Metre
def LPM : ℕ := 10

-- Statement to prove that the loss per metre is 10
theorem loss_per_metre :
  (CPM * TMS - TSP) / TMS = LPM :=
by
sorry

end loss_per_metre_l203_203055


namespace Julio_spent_on_limes_l203_203254

theorem Julio_spent_on_limes
  (days : ℕ)
  (lime_cost_per_3 : ℕ)
  (mocktails_per_day : ℕ)
  (lime_juice_per_lime_tbsp : ℕ)
  (lime_juice_per_mocktail_tbsp : ℕ)
  (limes_per_set : ℕ)
  (days_eq_30 : days = 30)
  (lime_cost_per_3_eq_1 : lime_cost_per_3 = 1)
  (mocktails_per_day_eq_1 : mocktails_per_day = 1)
  (lime_juice_per_lime_tbsp_eq_2 : lime_juice_per_lime_tbsp = 2)
  (lime_juice_per_mocktail_tbsp_eq_1 : lime_juice_per_mocktail_tbsp = 1)
  (limes_per_set_eq_3 : limes_per_set = 3) :
  days * mocktails_per_day * lime_juice_per_mocktail_tbsp / lime_juice_per_lime_tbsp / limes_per_set * lime_cost_per_3 = 5 :=
sorry

end Julio_spent_on_limes_l203_203254


namespace find_surface_area_of_sphere_l203_203162

noncomputable def edge_length_base := real.sqrt 3
noncomputable def lateral_edge_length := 2.0
noncomputable def radius_of_circumscribed_sphere := real.sqrt 2
noncomputable def surface_area_sphere (R : ℝ) := 4 * real.pi * R ^ 2

theorem find_surface_area_of_sphere (surface_area : ℝ) 
  (edge_base : ℝ) 
  (lateral_edge : ℝ) 
  (R : ℝ) :
  (∀ (edge_base = edge_length_base) (lateral_edge = lateral_edge_length) (R = radius_of_circumscribed_sphere), 
  surface_area = surface_area_sphere R) :=
by
  intros
  sorry

end find_surface_area_of_sphere_l203_203162


namespace count_two_digit_primes_with_ones_3_l203_203874

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203874


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203892

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203892


namespace problem_inequality_l203_203300

theorem problem_inequality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(y) - f(x) ≤ (y - x)^2) →
  (∀ n : ℕ, n > 0 → ∀ a b : ℝ, |f(b) - f(a)| ≤ (1 / n) * (b - a)^2) :=
by
  intros h n hn a b
  sorry

end problem_inequality_l203_203300


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203890

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203890


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203978

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203978


namespace harmonic_division_property_l203_203091

-- Geometry entities
variables {A B C D M E F N : Type}

-- Points D and M on line segments BC and AD respectively
variables (hD : D ∈ line[BC])
variables (hM : M ∈ line[AD])

-- Lines intercepted at points E and F
variables (hE : is_intersection (BM ∩ AC) E)
variables (hF : is_intersection (CM ∩ AB) F)

-- Line EF meets line AD at point N
variables (hN : is_intersection (EF ∩ AD) N)

-- Proof statement
theorem harmonic_division_property : 
    ∀ (A N D AM DM AN DN : ℝ), 
    hD → hM → hE → hF → hN → 
    (AN / DN = 1/2 * (AM / DM)) :=
by 
  intros A N D AM DM AN DN hD hM hE hF hN
  sorry

end harmonic_division_property_l203_203091


namespace positive_quadratic_expression_l203_203098

theorem positive_quadratic_expression (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + 4 + m > 0) ↔ (- (Real.sqrt 55) / 2 < m ∧ m < (Real.sqrt 55) / 2) := 
sorry

end positive_quadratic_expression_l203_203098


namespace common_tangent_parallel_BD_l203_203399

theorem common_tangent_parallel_BD
  (A B C D : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_angle : ∠BAC = ∠DAC)
  (I1 : Circle)
  (I2 : Circle)
  (h_incircle1 : Incircle I1 (triangle A B D))
  (h_incircle2 : Incircle I2 (triangle A C D)) :
  ∃ t : Line, ExternalTangent t I1 I2 ∧ Parallel t (line B D) :=
by sorry

end common_tangent_parallel_BD_l203_203399


namespace tan_105_degree_l203_203560

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203560


namespace bc_df_ea_eq_ef_ac_bd_l203_203061

variables {A B C D E F : Type}
variables {angle : A → A → B}
variables {length : A → C}
variables {AB BC CD DE EF FA BD AC DF EA : C}

axiom angle_sum : angle A B + angle C D + angle E F = 360
axiom length_product_eq : length AB * length CD * length EF = length BC * length DE * length FA

theorem bc_df_ea_eq_ef_ac_bd (k : C) : 
  length BC * length DF * length EA = length EF * length AC * length BD :=
by 
  -- The proof will be provided here
  sorry

end bc_df_ea_eq_ef_ac_bd_l203_203061


namespace deposit_percentage_correct_l203_203032

-- Define the conditions
def deposit_amount : ℕ := 50
def remaining_amount : ℕ := 950
def total_cost : ℕ := deposit_amount + remaining_amount

-- Define the proof problem statement
theorem deposit_percentage_correct :
  (deposit_amount / total_cost : ℚ) * 100 = 5 := 
by
  -- sorry is used to skip the proof
  sorry

end deposit_percentage_correct_l203_203032


namespace two_digit_primes_end_in_3_l203_203921

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203921


namespace count_two_digit_primes_ending_in_3_l203_203827

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203827


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203969

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203969


namespace crayon_selection_l203_203362

theorem crayon_selection :
  let total_crayons := 15,
      red_crayons := 4,
      selection := 5 in
  ∃ (ways : ℕ), 
      ways = (choose (total_crayons - red_crayons) 5) + (choose red_crayons 1 * choose (total_crayons - red_crayons) 4) ∧ 
      ways = 1782 := by
  -- Definitions and intermediate steps here
  sorry

end crayon_selection_l203_203362


namespace sum_of_squares_of_solutions_l203_203129

theorem sum_of_squares_of_solutions :
  let C := (λ x : ℝ, x^3 - 8 * x^2 + 15 * x)
  let sols := {0, 3, 5}
  ∑ x in sols, x^2 = 34 :=
by
  sorry

end sum_of_squares_of_solutions_l203_203129


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203486

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203486


namespace smaller_angle_at_3_clock_l203_203374

theorem smaller_angle_at_3_clock : 
  let full_circle_degrees := 360
  let hours_on_clock := 12
  let degrees_per_hour := full_circle_degrees / hours_on_clock
in degrees_per_hour * 3 = 90 :=
by sorry

end smaller_angle_at_3_clock_l203_203374


namespace product_of_roots_l203_203002

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 9 * x + 20

-- The main statement for the Lean theorem
theorem product_of_roots : (∃ x₁ x₂ : ℝ, quadratic x₁ = 0 ∧ quadratic x₂ = 0 ∧ x₁ * x₂ = 20) :=
by
  sorry

end product_of_roots_l203_203002


namespace sin_angle_FAC_l203_203265
open Real

noncomputable def coordinates : Type := (ℝ × ℝ × ℝ)

def A : coordinates := (0, 0, 0)
def B : coordinates := (1, 0, 0)
def D : coordinates := (0, 2, 0)
def E : coordinates := (0, 0, 3)
def C : coordinates := (1, 2, 0)
def F : coordinates := (1, 0, 3)

def vector_sub (v1 v2 : coordinates) : coordinates :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : coordinates) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : coordinates) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def vector_AC : coordinates := vector_sub C A
def vector_AF : coordinates := vector_sub F A

def cos_theta : ℝ :=
  dot_product vector_AC vector_AF / (norm vector_AC * norm vector_AF)

def sin_theta : ℝ :=
  sqrt (1 - cos_theta * cos_theta)

theorem sin_angle_FAC : sin_theta = sqrt (49 / 50) := by
  sorry

end sin_angle_FAC_l203_203265


namespace min_value_of_expression_l203_203160

theorem min_value_of_expression (a b : ℝ) (h : 2 * a - 3 * b + 6 = 0) : 
  4^a + (1 / 8^b) = 1 / 4 :=
by sorry

end min_value_of_expression_l203_203160


namespace tan_add_tan_105_eq_l203_203629

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203629


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203609

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203609


namespace two_digit_primes_ending_in_3_eq_6_l203_203941

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203941


namespace interest_rate_borrowed_l203_203043

variables {P : Type} [LinearOrderedField P]

def borrowed_amount : P := 9000
def lent_interest_rate : P := 0.06
def gain_per_year : P := 180
def per_cent : P := 100

theorem interest_rate_borrowed (r : P) (h : borrowed_amount * lent_interest_rate - gain_per_year = borrowed_amount * r) : 
  r = 0.04 :=
by sorry

end interest_rate_borrowed_l203_203043


namespace vec_eq_solution_exists_l203_203682

theorem vec_eq_solution_exists :
  ∃ (u v : ℝ), (1 + u * 8 = 2 + v * (-3)) ∧ (4 + u * (-6) = 5 + v * 4) ∧ u = -1 / 2 ∧ v = 1 :=
by
  use [-1 / 2, 1]
  sorry

end vec_eq_solution_exists_l203_203682


namespace maximum_teams_tied_for_most_wins_l203_203225

/-- In a round-robin tournament with 8 teams, each team plays one game
    against each other team, and each game results in one team winning
    and one team losing. -/
theorem maximum_teams_tied_for_most_wins :
  ∀ (teams games wins : ℕ), 
    teams = 8 → 
    games = (teams * (teams - 1)) / 2 →
    wins = 28 →
    ∃ (max_tied_teams : ℕ), max_tied_teams = 5 :=
by
  sorry

end maximum_teams_tied_for_most_wins_l203_203225


namespace tan_105_degree_l203_203563

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203563


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203965

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203965


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203476

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203476


namespace count_two_digit_primes_ending_with_3_l203_203835

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203835


namespace sum_of_solutions_eq_8_l203_203106

theorem sum_of_solutions_eq_8 :
    let a : ℝ := 1
    let b : ℝ := -8
    let c : ℝ := -26
    ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) →
      x1 + x2 = 8 :=
sorry

end sum_of_solutions_eq_8_l203_203106


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203971

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203971


namespace tan_105_eq_neg2_sub_sqrt3_l203_203537

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203537


namespace tan_105_l203_203465

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203465


namespace general_term_a_l203_203023

def S (n : ℕ) : ℕ := 2 * n ^ 2 - 3 * n + 2

def a (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * n - 5

theorem general_term_a (n : ℕ) : ∑ i in finset.range (n + 1), a i = S n :=
sorry

end general_term_a_l203_203023


namespace tan_105_l203_203492

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203492


namespace range_of_a_l203_203140

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | 5 < x}
  (A ∩ B = ∅) ↔ a ∈ {a : ℝ | a ≤ 2 ∨ a > 3} :=
by
  sorry

end range_of_a_l203_203140


namespace tan_105_l203_203495

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203495


namespace tan_105_degree_l203_203648

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203648


namespace num_two_digit_primes_with_ones_digit_3_l203_203943

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203943


namespace yang_hui_rect_eq_l203_203320

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l203_203320


namespace no_five_coins_sum_to_43_l203_203689

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem no_five_coins_sum_to_43 :
  ¬ ∃ (a b c d e : ℕ), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧ (a + b + c + d + e = 43) :=
sorry

end no_five_coins_sum_to_43_l203_203689


namespace tan_105_degree_l203_203642

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203642


namespace tan_105_degree_l203_203567

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203567


namespace train_crosses_post_in_25_2_seconds_l203_203435

noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

theorem train_crosses_post_in_25_2_seconds :
  train_crossing_time 40 280.0224 = 25.2 :=
by 
  sorry

end train_crosses_post_in_25_2_seconds_l203_203435


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203889

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203889


namespace distance_between_points_is_11_l203_203683

def point1 := (3, 7)
def point2 := (3, -4)

def distance_y (p1 p2 : ℤ × ℤ) : ℤ :=
  abs (p1.2 - p2.2)

theorem distance_between_points_is_11 :
  distance_y point1 point2 = 11 := by
  sorry

end distance_between_points_is_11_l203_203683


namespace count_two_digit_primes_ending_in_3_l203_203826

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203826


namespace abc_sum_l203_203332

theorem abc_sum : ∃ a b c : ℤ, 
  (∀ x : ℤ, x^2 + 13 * x + 30 = (x + a) * (x + b)) ∧ 
  (∀ x : ℤ, x^2 + 5 * x - 50 = (x + b) * (x - c)) ∧
  a + b + c = 18 := by
  sorry

end abc_sum_l203_203332


namespace final_discount_l203_203049

open Real

noncomputable def original_price : ℝ := 1
noncomputable def first_discount : ℝ := 1 / 3
noncomputable def coupon_discount : ℝ := 0.3

theorem final_discount :
  let sale_price := (2 / 3) * original_price in
  let price_after_coupon := (7 / 10) * sale_price in
  let total_discount := 1 - price_after_coupon / original_price in
  total_discount = 0.5333 :=
by
  sorry

end final_discount_l203_203049


namespace danielle_money_for_supplies_l203_203095

-- Define the conditions
def cost_of_molds := 3
def cost_of_sticks_pack := 1
def sticks_in_pack := 100
def cost_of_juice_bottle := 2
def popsicles_per_bottle := 20
def remaining_sticks := 40
def used_sticks := sticks_in_pack - remaining_sticks

-- Define number of juice bottles used
def bottles_of_juice_used : ℕ := used_sticks / popsicles_per_bottle

-- Define the total cost
def total_cost : ℕ := cost_of_molds + cost_of_sticks_pack + bottles_of_juice_used * cost_of_juice_bottle

-- Prove that Danielle had $10 for supplies
theorem danielle_money_for_supplies : total_cost = 10 := by {
  sorry
}

end danielle_money_for_supplies_l203_203095


namespace tan_105_degree_l203_203652

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203652


namespace limit_sqrt_tan_l203_203395

open Real

theorem limit_sqrt_tan : 
  tendsto (λ x: ℝ , (sqrt(x^2 - x + 1) - 1) / tan(π * x)) (nhds 1) (nhds (1 / (2 * π))) :=
by sorry

end limit_sqrt_tan_l203_203395


namespace tan_105_degree_l203_203647

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203647


namespace geometric_mean_geometric_sequence_l203_203090

/-- If the sequence {c_n} is a geometric sequence with positive terms,
then the sequence {d_n} defined by the geometric mean of the first n terms
is also a geometric sequence. -/
theorem geometric_mean_geometric_sequence
  (c : ℕ → ℝ) (c_pos : ∀ n, 0 < c n) (r : ℝ)
  (h : ∀ n, c (n + 1) = r * c n) :
  ∃ r' : ℝ, ∀ n, (∏ i in finset.range(n + 1), c i) ^ (1 / (n + 1)) = r' ^ n :=
sorry

end geometric_mean_geometric_sequence_l203_203090


namespace tan_105_l203_203473

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203473


namespace tan_105_degree_l203_203559

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203559


namespace female_muscovy_ducks_l203_203364

theorem female_muscovy_ducks :
  let total_ducks := 40
  let muscovy_percentage := 0.5
  let female_muscovy_percentage := 0.3
  let muscovy_ducks := total_ducks * muscovy_percentage
  let female_muscovy_ducks := muscovy_ducks * female_muscovy_percentage
  female_muscovy_ducks = 6 :=
by
  sorry

end female_muscovy_ducks_l203_203364


namespace intersection_product_range_l203_203234

theorem intersection_product_range {P Q R : ℝ × ℝ} 
    (hP : P.2^2 = 4 * P.1)
    (hQ : (Q.1 - 4)^2 + Q.2^2 = 8)
    (hR : (R.1 - 4)^2 + R.2^2 = 8)
    (hLine : ∃ k : ℝ, Q.2 = k * Q.1 + 2 * sqrt(P.1) - P.2^2 / 4 ∧ R.2 = k * R.1 + 2 * sqrt(P.1) - P.2^2 / 4)
    (inclination : ∀ x, x = P.1 → Q.2 ≥ √P.1 ∨ Q.2 ≤ -√P.1) :
    ∃ (L U : ℝ), L = 4 ∧ U = 36 ∧ ∀ pq pr, pq = |P - Q| ∧ pr = |P - R| → pq * pr ∈ [L, 8) ∪ (8, U] :=
begin
    sorry
end

end intersection_product_range_l203_203234


namespace two_digit_primes_count_l203_203761

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_two_digit_prime (m t u: ℕ) : Prop :=
  t ∈ {1, 3, 7, 8} ∧ u ∈ {1, 3, 7, 8} ∧ t ≠ u ∧ is_prime (10 * t + u)

theorem two_digit_primes_count : (finset.univ.filter (λ (x : ℕ), ∃ t u, is_valid_two_digit_prime x t u)).card = 7 := 
sorry

end two_digit_primes_count_l203_203761


namespace count_two_digit_primes_ending_in_3_l203_203828

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203828


namespace gym_class_total_students_l203_203438

theorem gym_class_total_students (group1_members group2_members : ℕ) 
  (h1 : group1_members = 34) (h2 : group2_members = 37) :
  group1_members + group2_members = 71 :=
by
  sorry

end gym_class_total_students_l203_203438


namespace tan_add_tan_105_eq_l203_203638

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203638


namespace complex_sum_correct_l203_203086

def complex_sum : ℂ :=
  let z1 : ℂ := -1/2 + (3/4) * Complex.I
  let z2 : ℂ := 7/3 - (5/6) * Complex.I
  z1 + z2

theorem complex_sum_correct : complex_sum = 11/6 - (1/12) * Complex.I :=
by 
  sorry

end complex_sum_correct_l203_203086


namespace range_of_a_l203_203749

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the interval as a closed interval from -1 to 1
def interval : Set ℝ := Set.Icc (-1) (1)

-- State the main proposition
theorem range_of_a (a : ℝ) :
  (∃ x ∈ interval, -x^2 + 3 * x + a > 0) ↔ a > -2 :=
by
  sorry

end range_of_a_l203_203749


namespace honda_day_shift_production_l203_203758

theorem honda_day_shift_production (S : ℕ) (day_shift_production : ℕ)
  (h1 : day_shift_production = 4 * S)
  (h2 : day_shift_production + S = 5500) :
  day_shift_production = 4400 :=
sorry

end honda_day_shift_production_l203_203758


namespace count_valid_numbers_is_31_l203_203198

def is_valid_digit (n : Nat) : Prop := n = 0 ∨ n = 2 ∨ n = 6 ∨ n = 8

def count_valid_numbers : Nat :=
  let valid_digits := [0, 2, 6, 8]
  let one_digit := valid_digits.filter (λ n => n % 4 = 0)
  let two_digits := valid_digits.product valid_digits |>.filter (λ (a, b) => (10*a + b) % 4 = 0)
  let three_digits := valid_digits.product two_digits |>.filter (λ (a, (b, c)) => (100*a + 10*b + c) % 4 = 0)
  one_digit.length + two_digits.length + three_digits.length

theorem count_valid_numbers_is_31 : count_valid_numbers = 31 := by
  sorry

end count_valid_numbers_is_31_l203_203198


namespace num_two_digit_primes_with_ones_digit_3_l203_203948

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203948


namespace sum_of_digits_1_to_5000_l203_203128

def sum_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

def sum_of_digits_up_to (n : ℕ) : ℕ :=
  (list.range (n + 1)).sum (sum_digits)

theorem sum_of_digits_1_to_5000 : sum_of_digits_up_to 5000 = 229450 := 
  sorry

end sum_of_digits_1_to_5000_l203_203128


namespace find_a_b_find_symmetry_monotonic_increase_l203_203316

noncomputable def some_sound_function (a b : ℝ) (x : ℝ) : ℝ :=
  a * sin (x)^2 + sqrt 3 * a * sin (x) * cos (x) - 3 / 2 * a + b

theorem find_a_b (a b : ℝ) (h_pos : a > 0) (h_ampl : ∃ x, some_sound_function a b x = 4) (h_max : ∀ x ∈ Icc (0 : ℝ) (π / 2), some_sound_function a b x ≤ 1) : 
  a = 4 ∧ b = 1 := sorry

theorem find_symmetry_monotonic_increase (a b : ℝ) (h_pos : a > 0) (h_ampl : ∃ x, some_sound_function a b x = 4) (h_max : ∀ x ∈ Icc (0 : ℝ) (π / 2), some_sound_function a b x ≤ 1) (h_a : a = 4) (h_b : b = 1):
  (∀ k : ℤ, ∃ x, some_sound_function a b (x) = some_sound_function a b (x + k * (π / 2))) ∧
  (interval_of_monotonic_increase : (0, π / 3) ∪ (5 * π / 6, π)) := sorry

end find_a_b_find_symmetry_monotonic_increase_l203_203316


namespace problem_statement_l203_203738

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 + x - 2

theorem problem_statement : 
  f (1 / f 2) = 15 / 16 :=
by
  sorry

end problem_statement_l203_203738


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203901

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203901


namespace tan_105_l203_203553

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203553


namespace tan_105_l203_203557

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203557


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203504

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203504


namespace problem_solution_l203_203230

noncomputable 
def polar_to_cartesian_equation : Prop :=
  ∀ (θ ρ : ℝ), (0 ≤ θ ∧ θ < 2*π) → (ρ * (cos θ * (1/2) + sin θ * (√3/2)) = 1) ↔ (∃ x y : ℝ, x + √3 * y = 2)

noncomputable 
def intersection_points_MN (M N : ℝ × ℝ) : Prop :=
  M = (2, 0) ∧ N = (2 * √3 / 3, π / 2)

noncomputable 
def polar_equation_OP : Prop :=
  ∀ ρ : ℝ, θ = π / 6 → (ρ ∈ Ioo (-∞) ∞)

theorem problem_solution : Prop :=
  polar_to_cartesian_equation ∧ 
  (intersection_points_MN (2, 0) (2 * √3 / 3, π / 2)) ∧ 
  polar_equation_OP

end problem_solution_l203_203230


namespace inequality_proof_l203_203270

noncomputable def log4 := Real.log 4
noncomputable def log5 := Real.log 5

def a : ℝ := (Real.log (Real.sqrt 5) / log4)
def b : ℝ := (Real.log 2 / log5)
def c : ℝ := (Real.log 5 / log4)

theorem inequality_proof : b < a ∧ a < c :=
by
  sorry

end inequality_proof_l203_203270


namespace count_valid_triangles_l203_203154

def valid_triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

def different_triangles_count (segments : List ℝ) : ℕ :=
  (segments.combinations 3).countp (λ triangle, 
    match triangle with
    | [a, b, c] => valid_triangle a b c
    | _ => false
  )

theorem count_valid_triangles :
  different_triangles_count [1, 2, 3, 4, 5] = 3 :=
by 
  sorry

end count_valid_triangles_l203_203154


namespace C1_general_eq_C2_cartesian_eq_min_dist_PQ_l203_203231

section
variable (α : ℝ) (x y ρ θ : ℝ)

-- Definition of curve C1 parametric equations.
def C1_parametric : Prop :=
  x = 2 * sqrt 2 * cos α ∧ y = 2 * sin α

-- Definition of curve C2 polar equation.
def C2_polar : Prop :=
  ρ * cos θ - sqrt 2 * ρ * sin θ - 5 = 0

-- Proving the general equation of curve C1.
theorem C1_general_eq (h : C1_parametric α x y) :
  (x^2 / 8) + (y^2 / 4) = 1 :=
sorry

-- Proving the Cartesian coordinate equation of curve C2.
theorem C2_cartesian_eq (h : C2_polar ρ θ) :
  ∃ x y, (x - sqrt 2 * y - 5 = 0) :=
sorry

-- Proving the minimum value of |PQ|.
theorem min_dist_PQ (hC1 : C1_parametric α x y) (hC2 : C2_polar ρ θ) :
  ∃ α, (let d := (2 * sqrt 2 * cos α - 2 * sqrt 2 * sin α - 5) / sqrt (1 + 2) in d = sqrt 3 / 3) :=
sorry

end

end C1_general_eq_C2_cartesian_eq_min_dist_PQ_l203_203231


namespace probability_point_in_region_l203_203293

theorem probability_point_in_region (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 2010) 
  (h2 : 0 ≤ y ∧ y ≤ 2009) 
  (h3 : ∃ (u v : ℝ), (u, v) = (x, y) ∧ x > 2 * y ∧ y > 500) : 
  ∃ p : ℚ, p = 1505 / 4018 := 
sorry

end probability_point_in_region_l203_203293


namespace bottle_caps_per_friend_l203_203084

-- The context where Catherine has 18 bottle caps
def bottle_caps : Nat := 18

-- Catherine distributes these bottle caps among 6 friends
def number_of_friends : Nat := 6

-- We need to prove that each friend gets 3 bottle caps
theorem bottle_caps_per_friend : bottle_caps / number_of_friends = 3 :=
by sorry

end bottle_caps_per_friend_l203_203084


namespace tan_105_eq_neg2_sub_sqrt3_l203_203616

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203616


namespace math_problem_l203_203132

theorem math_problem (a b c : ℝ) (h₁ : a = 85) (h₂ : b = 32) (h₃ : c = 113) :
  (a + b / c) * c = 9637 :=
by
  rw [h₁, h₂, h₃]
  sorry

end math_problem_l203_203132


namespace tan_105_l203_203470

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203470


namespace tangent_line_circle_l203_203719

theorem tangent_line_circle (m : ℝ) (φ : ℝ) (hm : m > 0) : 
  (∃ (x y : ℝ), (x = √m * Real.cos φ) ∧ (y = √m * Real.sin φ) ∧ (x + y = m)) → m = 2 :=
by
  intro h
  sorry

end tangent_line_circle_l203_203719


namespace count_two_digit_primes_with_ones_3_l203_203876

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203876


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203512

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203512


namespace complex_number_real_imaginary_opposite_l203_203211

theorem complex_number_real_imaginary_opposite (a : ℝ) (i : ℂ) (comp : z = (1 - a * i) * i):
  (z.re = -z.im) → a = 1 :=
by 
  sorry

end complex_number_real_imaginary_opposite_l203_203211


namespace tan_add_tan_105_eq_l203_203631

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203631


namespace tan_105_degree_l203_203651

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203651


namespace range_of_x_l203_203203

theorem range_of_x {x : ℝ} : (sqrt ((5 - x) ^ 2) = x - 5) → (x ≥ 5) :=
by
  sorry

end range_of_x_l203_203203


namespace count_two_digit_primes_with_ones_digit_3_l203_203809

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203809


namespace tan_105_eq_neg2_sub_sqrt3_l203_203517

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203517


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203891

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203891


namespace demographers_prediction_basis_l203_203661

theorem demographers_prediction_basis
  (P: Type)
  (time_to_double_mexico: Real)
  (time_to_double_usa: Real)
  (stable_pop_sweden: Prop)
  (decreasing_pop_germany: Prop)
  (main_basis: P → Prop)
  (pb_time_mexico: time_to_double_mexico > 0)
  (usa_pred: time_to_double_usa > 0)
  (sweden_stable: stable_pop_sweden)
  (germany_deci: decreasing_pop_germany) :
  main_basis (λ p, p = "Age composition") :=
by
  sorry

end demographers_prediction_basis_l203_203661


namespace dane_daughters_initial_flowers_l203_203659

theorem dane_daughters_initial_flowers :
  (exists (x y : ℕ), x = y ∧ 5 * 4 = 20 ∧ x + y = 30) →
  (exists f : ℕ, f = 5 ∧ 10 = 30 - 20 + 10 ∧ x = f * 2) :=
by
  -- Lean proof needs to go here
  sorry

end dane_daughters_initial_flowers_l203_203659


namespace solution_to_system_l203_203718

theorem solution_to_system :
  (∀ (x y : ℚ), (y - x - 1 = 0) ∧ (y + x - 2 = 0) ↔ (x = 1/2 ∧ y = 3/2)) :=
by
  sorry

end solution_to_system_l203_203718


namespace two_digit_number_ratio_l203_203248

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b
def swapped_two_digit_number (a b : ℕ) : ℕ := 10 * b + a

theorem two_digit_number_ratio (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h_ratio : 6 * two_digit_number a b = 5 * swapped_two_digit_number a b) : 
  two_digit_number a b = 45 :=
by
  sorry

end two_digit_number_ratio_l203_203248


namespace arithmetic_problem_l203_203085

theorem arithmetic_problem : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end arithmetic_problem_l203_203085


namespace tan_105_eq_neg2_sub_sqrt3_l203_203575

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203575


namespace solution_set_of_inequality_l203_203351

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l203_203351


namespace correct_omega_l203_203389

theorem correct_omega (Ω : ℕ) (h : Ω * Ω = 2 * 2 * 2 * 2 * 3 * 3) : Ω = 2 * 2 * 3 :=
by
  sorry

end correct_omega_l203_203389


namespace tan_105_eq_neg2_sub_sqrt3_l203_203520

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203520


namespace distribute_volunteers_l203_203081

theorem distribute_volunteers (volunteers venues : ℕ) (h_vol : volunteers = 5) (h_venues : venues = 3) :
  ∃ (distributions : ℕ), (∀ v : ℕ, 1 ≤ v → v ≤ venues) ∧ 
  (∑ v in finset.range venues, v) = volunteers ∧ 
  distributions = 150 :=
by
  use 150
  sorry

end distribute_volunteers_l203_203081


namespace count_two_digit_primes_with_ones_digit_3_l203_203805

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203805


namespace count_two_digit_primes_with_ones_digit_3_l203_203803

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203803


namespace two_digit_primes_with_ones_digit_3_l203_203853

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203853


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203603

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203603


namespace area_triangle_HIO_l203_203227

theorem area_triangle_HIO :
  ∀ (HI IJ JM NK : ℝ),
    HI = 8 →
    IJ = 4 →
    JM = 2 →
    NK = 1 →
    let JK := HI in
    ∃ O : Point (ℝ × ℝ),
      let MN := JK - (JM + NK) in
      let height_HIO := (HI / MN) * IJ in
      let area_HIO := (1/2) * HI * height_HIO in
      area_HIO = 25.6 :=
by
  intros HI IJ JM NK HHI HIJ JMN KNK;
  let JK := HI;
  use (0, 0); -- Placeholder for the actual intersection point O
  let MN := JK - (JM + NK);
  let height_HIO := (HI / MN) * IJ;
  let area_HIO := (1/2) * HI * height_HIO;
  sorry

end area_triangle_HIO_l203_203227


namespace min_m_even_g_l203_203168

-- Definitions
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

def g (x m : ℝ) : ℝ := f (x - m)

-- Theorem statement
theorem min_m_even_g (m : ℝ) (h : m > 0) : 
  ∀ x : ℝ, g x m = g (-x) m ↔ m = Real.pi / 3 :=
sorry

end min_m_even_g_l203_203168


namespace problem_I_problem_II_l203_203401

-- Problem (I)
theorem problem_I (a : ℝ) (h : ∀ x : ℝ, x^2 - 3 * a * x + 9 > 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Problem (II)
theorem problem_II (m : ℝ) 
  (h₁ : ∀ x : ℝ, x^2 + 2 * x - 8 < 0 → x - m > 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * x - 8 < 0) : m ≤ -4 :=
sorry

end problem_I_problem_II_l203_203401


namespace two_digit_primes_with_ones_digit_3_count_eq_7_l203_203980

def two_digit_numbers_with_ones_digit_3 : List ℕ :=
  [13, 23, 33, 43, 53, 63, 73, 83, 93]

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_prime_numbers_with_ones_digit_3 : ℕ :=
  (two_digit_numbers_with_ones_digit_3.filter is_prime).length

theorem two_digit_primes_with_ones_digit_3_count_eq_7 : 
  count_prime_numbers_with_ones_digit_3 = 7 := 
  sorry

end two_digit_primes_with_ones_digit_3_count_eq_7_l203_203980


namespace tan_105_eq_neg2_sub_sqrt3_l203_203521

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203521


namespace find_solutions_l203_203681

noncomputable def is_solution (n : ℕ) : Prop :=
  ∀ (a : ℕ), nat.gcd a n = 1 → 2 * n^2 ∣ a^n - 1

theorem find_solutions :
  {n : ℕ | is_solution n} = {2, 6, 42, 1806} := 
sorry

end find_solutions_l203_203681


namespace storage_space_l203_203031

theorem storage_space (length width height number_of_boxes cost_per_box total_cost : ℕ) 
  (box_dims : length = 15 ∧ width = 12 ∧ height = 10)
  (costs : cost_per_box = 0.6 ∧ total_cost = 360)
  (num_boxes : number_of_boxes = total_cost / cost_per_box) 
  (vol_one_box : ℕ) (total_space : ℕ)
  (vol_one_box_def : vol_one_box = length * width * height)
  (total_space_def : total_space = vol_one_box * number_of_boxes) :
  total_space = 1080000 := 
sorry

end storage_space_l203_203031


namespace count_two_digit_primes_with_ones_digit_3_l203_203808

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203808


namespace tan_105_degree_l203_203591

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203591


namespace tan_105_degree_l203_203654

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203654


namespace integral_approximation_l203_203097

variable {a b : ℝ} (ϕ : ℝ → ℝ) (n : ℕ)
variable (r : Fin n → ℝ) [∀ i, r i ∈ set.Icc 0 1]

def x (i : Fin n) : ℝ := a + (b - a) * r i

noncomputable def I1 : ℝ :=
  (b - a) * (Finset.univ.sum (λ i, ϕ (x i)) / n)

theorem integral_approximation :
  ∫ x in a..b, ϕ x = (b - a) * (Finset.univ.sum (λ i, ϕ (x i)) / n) :=
sorry

end integral_approximation_l203_203097


namespace total_pools_l203_203291

theorem total_pools (pools_ark : ℕ) (h_pools_ark : pools_ark = 200)
  (pools_supply : ℕ) (h_pools_supply : pools_supply = 3 * pools_ark) :
  pools_ark + pools_supply = 800 :=
by
  rw [h_pools_ark, h_pools_supply]
  simp
  sorry

end total_pools_l203_203291


namespace storks_difference_l203_203219

def storks_initial := 8
def herons_initial := 4
def sparrows_initial := 5

def storks_flew_away := 3
def herons_flew_away := 2

def sparrows_arrived := 4
def hummingbirds_arrived := 2

def storks_remaining := storks_initial - storks_flew_away
def herons_remaining := herons_initial - herons_flew_away
def sparrows_remaining := sparrows_initial + sparrows_arrived
def hummingbirds_remaining := hummingbirds_arrived

def total_other_species := herons_remaining + sparrows_remaining + hummingbirds_remaining

theorem storks_difference :
  storks_remaining - total_other_species = -8 :=
by
  sorry

end storks_difference_l203_203219


namespace count_two_digit_primes_with_ones_3_l203_203878

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203878


namespace sum_of_valid_two_digit_numbers_l203_203665

theorem sum_of_valid_two_digit_numbers
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (a - b) ∣ (10 * a + b))
  (h4 : (a * b) ∣ (10 * a + b)) :
  (10 * a + b = 21) → (21 = 21) :=
sorry

end sum_of_valid_two_digit_numbers_l203_203665


namespace tan_105_eq_neg2_sub_sqrt3_l203_203627

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203627


namespace count_two_digit_primes_with_ones_3_l203_203870

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203870


namespace num_two_digit_primes_with_ones_digit_3_l203_203955

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203955


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203608

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203608


namespace categorize_numbers_l203_203678

def numbers := {20, -4.8, 0, -2023, +(2/7), -Real.pi, 0.020020002, (0.010010001 : ℝ), (⇑Rat.mkRepeating 1 2)}

def negative_number_set := {-4.8, -2023, -Real.pi}
def fraction_set := {-4.8, +(2/7), 0.020020002, (⇑Rat.mkRepeating 1 2)}
def non_positive_integer_set := {0, -2023}
def irrational_number_set := {-Real.pi, (0.010010001 : ℝ)}

theorem categorize_numbers :
  (∀ x ∈ negative_number_set, x < 0) ∧
  (∀ x ∈ fraction_set, ∃ p q : ℤ, q ≠ 0 ∧ x = p / q) ∧
  (∀ x ∈ non_positive_integer_set, ∃ n : ℤ, x = n ∧ x ≤ 0) ∧
  (∀ x ∈ irrational_number_set, ¬ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q) :=
by
  sorry

end categorize_numbers_l203_203678


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203605

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203605


namespace tan_105_l203_203547

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203547


namespace product_of_numbers_l203_203213

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := 
by 
  sorry

end product_of_numbers_l203_203213


namespace complex_cube_roots_of_unity_l203_203087

noncomputable def x : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem complex_cube_roots_of_unity : x^15 - y^15 = 0 := by
  sorry

end complex_cube_roots_of_unity_l203_203087


namespace nature_of_roots_irrat_l203_203088

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 5*m*x + 3*m^2 + 6

-- Problem statement as a Lean 4 definition
theorem nature_of_roots_irrat (m : ℝ) :
  (∃ x y : ℝ, quadratic_equation x m = 0 ∧ quadratic_equation y m = 0 ∧ x * y = 12) →
  (discriminant (1 : ℝ) (-5 * m) (3 * m^2 + 6) > 0 ∧ ∀ d, discriminant (1 : ℝ) (-5 * m) (3 * m^2 + 6) = d^2 → d = 0) := 
    sorry

-- Helper function to compute the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

end nature_of_roots_irrat_l203_203088


namespace peanut_butter_servings_l203_203040

theorem peanut_butter_servings : 
  let total_peanut_butter := (113 : ℚ) / 3
      serving_size := (5 : ℚ) / 2
  in total_peanut_butter / serving_size = 15 + (1 / 15) :=
by
  -- Proof goes here, skipped with sorry
  sorry

end peanut_butter_servings_l203_203040


namespace count_two_digit_primes_ending_in_3_l203_203831

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203831


namespace tan_105_eq_neg2_sub_sqrt3_l203_203523

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203523


namespace two_digit_primes_with_ones_digit_three_count_l203_203778

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203778


namespace count_two_digit_primes_ending_with_3_l203_203844

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203844


namespace find_z_l203_203279

variable (x y z : ℝ)

-- Define x, y as given in the problem statement
def x_def : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) := by
  sorry

def y_def : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) := by
  sorry

-- Define the equation relating z to x and y
def z_eq : 192 * z = x^4 + y^4 + (x + y)^4 := by 
  sorry

-- Theorem stating the value of z
theorem find_z (h1 : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3))
               (h2 : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3))
               (h3 : 192 * z = x^4 + y^4 + (x + y)^4) :
  z = 6 := by 
  sorry

end find_z_l203_203279


namespace range_of_product_of_zeros_l203_203180

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 1 then real.log x else 1 - x / 2

noncomputable def F (x m : ℝ) : ℝ :=
  f (f x + 1) + m

theorem range_of_product_of_zeros (m : ℝ) (x1 x2 : ℝ) (h1 : F x1 m = 0) (h2 : F x2 m = 0) : 
  x1 * x2 ∈ set.Iio (real.sqrt real.exp 1) :=
sorry

end range_of_product_of_zeros_l203_203180


namespace quadrilateral_area_l203_203436

-- Define the conditions
variable (T : Triangle) -- A triangle T
variable (P Q R : Point) -- Points on the sides of the triangle
variable (O : Point) -- Intersection point within the triangle
variable (area_T1 area_T2 area_T3 : ℝ) -- Areas of the smaller triangles

-- State the areas of the smaller triangles as given
axiom area_T1_def : area (Triangle.mk T.vertex1 P O) = 4
axiom area_T2_def : area (Triangle.mk T.vertex2 Q O) = 8
axiom area_T3_def : area (Triangle.mk T.vertex3 R O) = 10

-- The proof statement
theorem quadrilateral_area : area (Quadrilateral.mk P Q R O) = 70 / 11 :=
  sorry -- Proof to be provided

end quadrilateral_area_l203_203436


namespace problem_solution_l203_203215

open Real

noncomputable def problem_statement (a x y : ℝ) :=
  x * sqrt(a * (x - a)) + y * sqrt(a * (y - a)) = sqrt(log (x - a) - log (a - y))

theorem problem_solution (a x y : ℝ) (h1 : problem_statement a x y) (h2 : x > a) (h3 : a > y) :
  (3 * x ^ 2 + x * y - y ^ 2) / (x ^ 2 - x * y + y ^ 2) = 1 / 3 :=
sorry

end problem_solution_l203_203215


namespace probability_x_gt_3y_l203_203125

-- defining the boundaries of the rectangle
def vertices : set (ℝ × ℝ) := {(0, 0), (2010, 0), (2010, 2011), (0, 2011)}

-- defining the region of interest where x > 3y
def region_of_interest : set (ℝ × ℝ) := {p | p.snd < p.fst / 3}

-- defining the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

-- the area of the rectangle
def rectangle_area (width height : ℝ) : ℝ := width * height

theorem probability_x_gt_3y :
  let rect_width := 2010
  let rect_height := 2011
  let tri_base := 2010
  let tri_height := 670
  let area_triangle := triangle_area tri_base tri_height
  let area_rectangle := rectangle_area rect_width rect_height
  area_triangle / area_rectangle = (67335 / 404511) :=
  by
    let rect_width := 2010
    let rect_height := 2011
    let tri_base := 2010
    let tri_height := 670
    let area_triangle := triangle_area tri_base tri_height
    let area_rectangle := rectangle_area rect_width rect_height
    show area_triangle / area_rectangle = (67335 / 404511)
    sorry

end probability_x_gt_3y_l203_203125


namespace leaves_collected_l203_203442

noncomputable def total_trees := 37
noncomputable def num_apple_trees := 17
noncomputable def num_poplar_trees := 20
noncomputable def start_collecting_from := 8

theorem leaves_collected : num_apple_trees = 17 ∧ num_poplar_trees = 20 ∧ start_collecting_from = 8 → 
                            ((num_apple_trees - (start_collecting_from - 1)) + num_poplar_trees) = 24 :=
begin
  sorry
end

end leaves_collected_l203_203442


namespace tan_105_eq_neg2_sub_sqrt3_l203_203619

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203619


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203972

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203972


namespace triangle_area_is_300_l203_203664

noncomputable def triangle_area : ℝ := 
  let f : ℝ → ℝ := λ x, (x - 5)^2 * (x + 3)
  let x1 : ℝ := -3 -- x-intercept
  let x2 : ℝ := 5  -- x-intercept
  let y_intercept := f 0 -- y-intercept, which is 75
  let base := x2 - x1 -- base of the triangle, which is 8
  let height := y_intercept -- height of the triangle, which is 75
  1 / 2 * base * height -- area of the triangle

theorem triangle_area_is_300 : triangle_area = 300 := by
  sorry

end triangle_area_is_300_l203_203664


namespace coordinates_with_respect_to_origin_l203_203328

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (2, -6)) : (x, y) = (2, -6) :=
by
  sorry

end coordinates_with_respect_to_origin_l203_203328


namespace tan_105_l203_203499

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203499


namespace smallest_prime_greater_than_50_l203_203003

theorem smallest_prime_greater_than_50 : 
  ∃ p : ℕ, prime p ∧ p > 50 ∧ ∀ q : ℕ, prime q ∧ q > 50 → p ≤ q :=
sorry

end smallest_prime_greater_than_50_l203_203003


namespace tan_105_degree_l203_203653

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203653


namespace sum_of_roots_range_l203_203736

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -x^2 - 2*x + 1 else |Real.log x / Real.log 2|

theorem sum_of_roots_range (k : ℝ) (x1 x2 x3 x4 : ℝ) 
  (hk : 0 < k ∧ k < 2) 
  (hx1x2 : x1 + x2 = -2) 
  (hx3x4_prod : x3 * x4 = 1)
  (hx3x4_range : 1 < x4 ∧ x4 ≤ 2) 
  (hroots : ∀ x, f x = k → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  : 0 ≤ x1 + x2 + x3 + x4 ∧ x1 + x2 + x3 + x4 ≤ 1 / 2 :=
sorry

end sum_of_roots_range_l203_203736


namespace two_digit_primes_end_in_3_l203_203915

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203915


namespace number_of_ordered_arrays_l203_203185

theorem number_of_ordered_arrays (a b c d : ℕ) :
  {a, b, c, d} = {1, 2, 3, 4} →
  (a = 1 ∧ (b ≠ 1 ∧ c ≠ 3 ∧ d = 4) → False) →
  (b ≠ 1 ∧ (c ≠ 3 ∧ d ≠ 4) → False) →
  (c = 3 ∧ (a ≠ 1 ∧ b = 1 ∧ d = 4) → False) →
  (d ≠ 4 ∧ (a ≠ 1 ∧ b = 1 ∧ c ≠ 3) → False) →
  (set.to_finset {a, b, c, d}).card = 4 :=
by sorry

end number_of_ordered_arrays_l203_203185


namespace function_solution_unique_l203_203278

theorem function_solution_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → f(x + f(y)) = y * f(x * y + 1)) : ∀ y > 0, f(y) = 1 / y :=
by
  sorry

end function_solution_unique_l203_203278


namespace count_two_digit_primes_with_ones_digit_3_l203_203799

theorem count_two_digit_primes_with_ones_digit_3 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset.card = 6 :=
by
  sorry

end count_two_digit_primes_with_ones_digit_3_l203_203799


namespace count_two_digit_primes_ending_in_3_l203_203832

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203832


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203977

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203977


namespace find_point_P_l203_203756

open Real

def P := (x : ℝ) × (y : ℝ)

def distance (P1 P2 : P) : ℝ :=
  Real.sqrt ((P2.1 - P1.1) ^ 2 + (P2.2 - P1.2) ^ 2)

theorem find_point_P :
  let P1 := (2, -1)
  let P2 := (0, 5)
  let P := (-2, 11)
  distance P1 P = 2 * distance P P2 → P = (-2, 11) :=
by
  sorry

end find_point_P_l203_203756


namespace tan_105_eq_neg2_sub_sqrt3_l203_203516

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203516


namespace distance_1_neg3_neg4_5_l203_203378

def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_1_neg3_neg4_5 :
  distance_between_points 1 (-3) (-4) 5 = Real.sqrt 89 :=
by
  sorry

end distance_1_neg3_neg4_5_l203_203378


namespace function_D_properties_l203_203069

def y1 (x : ℝ) : ℝ := Real.sin (x - Real.pi / 2)
def y2 (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 2)
def y3 (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 2)
def y4 (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2)

theorem function_D_properties : 
  y4 = λ x, -Real.sin (2 * x) ∧ 
  (∀ x, y4 (x + Real.pi) = y4 x) ∧ 
  (∀ x y, x ≤ y → x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → y4 x ≤ y4 y) :=
by
  sorry

end function_D_properties_l203_203069


namespace two_digit_primes_with_ones_digit_three_count_l203_203769

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203769


namespace tan_105_degree_l203_203589

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203589


namespace sasha_picks_24_leaves_l203_203440

def num_apple_trees := 17
def num_poplar_trees := 20
def starting_apple_tree := 8

theorem sasha_picks_24_leaves :
  ∃ n : ℕ, n = 24 ∧ (num_poplar_trees + (num_apple_trees - starting_apple_tree + 1)) = n :=
begin
  sorry
end

end sasha_picks_24_leaves_l203_203440


namespace second_group_members_l203_203407

theorem second_group_members (total first third : ℕ) (h1 : total = 70) (h2 : first = 25) (h3 : third = 15) :
  (total - first - third) = 30 :=
by
  sorry

end second_group_members_l203_203407


namespace two_digit_primes_end_in_3_l203_203918

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203918


namespace two_digit_primes_end_in_3_l203_203911

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203911


namespace find_n_l203_203712

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + cos x) = (1/2) * (log 10 n - 2)) :
  n = 102 :=
sorry

end find_n_l203_203712


namespace sum_even_positive_integers_less_than_62_l203_203379

theorem sum_even_positive_integers_less_than_62 :
  (∑ k in finset.range 31, 2 * k) = 930 := 
begin
  sorry
end

end sum_even_positive_integers_less_than_62_l203_203379


namespace min_distance_point_to_origin_l203_203748

theorem min_distance_point_to_origin :
  (∀ α : ℝ, let x := Real.cos α - 1
                 y := Real.sin α + 1
                 PO := Real.sqrt (x^2 + y^2) in 
                 PO ≥ (Real.sqrt 2 - 1)) :=
begin
  sorry
end

end min_distance_point_to_origin_l203_203748


namespace problem_intersection_l203_203752

open Set

-- Given conditions
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { x | abs x ≤ 1 }

-- The proof problem we aim to solve
theorem problem_intersection : A ∩ B = { -1, 0, 1 } :=
by sorry

end problem_intersection_l203_203752


namespace two_digit_primes_with_ones_digit_three_count_l203_203762

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203762


namespace leaves_collected_l203_203443

noncomputable def total_trees := 37
noncomputable def num_apple_trees := 17
noncomputable def num_poplar_trees := 20
noncomputable def start_collecting_from := 8

theorem leaves_collected : num_apple_trees = 17 ∧ num_poplar_trees = 20 ∧ start_collecting_from = 8 → 
                            ((num_apple_trees - (start_collecting_from - 1)) + num_poplar_trees) = 24 :=
begin
  sorry
end

end leaves_collected_l203_203443


namespace cube_painted_surface_l203_203036

theorem cube_painted_surface (n : ℕ) (hn : n > 2) 
: 6 * (n - 2) ^ 2 = (n - 2) ^ 3 → n = 8 :=
by
  sorry

end cube_painted_surface_l203_203036


namespace probability_sum_even_l203_203139

theorem probability_sum_even :
  let balls := {1, 2, 3, 4}
  let combinations := Lean.List.combinations balls 2
  let even_count := combinations.count (fun (pair : Lean.List ℕ) => (pair.sum % 2 = 0))
  ∑ p in combinations,  pair.sum_even := 
  -- prob_even = count_even / 6
  even_count.to_real / combinations.length.to_real = 1/3 :=
begin
  sorry
end

end probability_sum_even_l203_203139


namespace line_bisects_circle_l203_203340

theorem line_bisects_circle
  (C : Type)
  [MetricSpace C]
  (x y : ℝ)
  (h : ∀ {x y : ℝ}, x^2 + y^2 - 2*x - 4*y + 1 = 0) : 
  x - y + 1 = 0 → True :=
by
  intro h_line
  sorry

end line_bisects_circle_l203_203340


namespace number_of_trees_planted_l203_203034

-- Define the given conditions
def circumference : ℝ := 150
def interval_distance : ℝ := 3

-- State the theorem
theorem number_of_trees_planted :
    ∃ n : ℕ, n = (circumference / interval_distance : ℝ).to_nat :=
by
  -- Proof is not required
  sorry

end number_of_trees_planted_l203_203034


namespace range_of_a_for_quadratic_eq_l203_203210

theorem range_of_a_for_quadratic_eq (a : ℝ) (h : ∀ x : ℝ, ax^2 = (x+1)*(x-1)) : a ≠ 1 :=
by
  sorry

end range_of_a_for_quadratic_eq_l203_203210


namespace ratio_cubed_eq_27_l203_203656

theorem ratio_cubed_eq_27 : (81000^3) / (27000^3) = 27 := 
by
  sorry

end ratio_cubed_eq_27_l203_203656


namespace Gloria_time_race_l203_203195

variable (Greta_time George_time Gloria_time : ℕ)
variable (h1 : Greta_time = 6)
variable (h2 : George_time = Greta_time - 2)
variable (h3 : Gloria_time = George_time * 2)

theorem Gloria_time_race : Gloria_time = 8 :=
by
  -- Using given conditions
  rw [h1, h2, h3]
  norm_num
  sorry

end Gloria_time_race_l203_203195


namespace coefficient_a8_in_expansion_l203_203171

theorem coefficient_a8_in_expansion :
  let a := 2 + x
  in  (a + x)^10 = 
      ∑ (k : ℕ) in Finset.range 11, (nat.choose 10 k) * (2^(10 - k)) * x^k →
  let c := (Finset.range 11).sum (λ k, (10.choose k) * (3^(10 - k)) * (-1)^k * (1 - x)^k)
  in  (Finset.range 11).sum (λ k, (10.choose k) * (3^(10 - k)) * (-1)^k * (1 - x)^k) =
      ∑ (k : ℕ) in Finset.range 11, (45 * 6561) * (1 - x)^8 :=
  c.coeff 8 = 405 := sorry

end coefficient_a8_in_expansion_l203_203171


namespace distance_1_neg3_neg4_5_l203_203377

def distance_between_points (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_1_neg3_neg4_5 :
  distance_between_points 1 (-3) (-4) 5 = Real.sqrt 89 :=
by
  sorry

end distance_1_neg3_neg4_5_l203_203377


namespace luana_top_circle_and_constant_sum_l203_203244

theorem luana_top_circle_and_constant_sum
  (x1 x2 x3 x4 x5 x6 x7 : ℕ)
  (h_distinct : {x1, x2, x3, x4, x5, x6, x7}.card = 7)
  (h_range : ∀ x ∈ {x1, x2, x3, x4, x5, x6, x7}, x ∈ {1, 2, 3, 4, 5, 6, 7})
  (h_sum_28 : x1 + x2 + x3 + x4 + x5 + x6 + x7 = 28)
  (s : ℕ)
  (h_trios : (x1 + x2 + x3 = s) ∧ (x4 + x5 + x6 = s) ∧ (x1 + x5 + x7 = s) ∧ (x2 + x5 + x6 = s) ∧ (x3 + x5 + x7 = s)) :
  x1 = 4 ∧ s = 12 :=
sorry

end luana_top_circle_and_constant_sum_l203_203244


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203483

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203483


namespace tan_105_degree_l203_203644

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203644


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203893

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203893


namespace line_intersects_circle_l203_203344

theorem line_intersects_circle (k : ℝ) :
  let line := λ x y : ℝ, k * x + y + 2 = 0 in
  let circle := λ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 16 in
  ∃ x y : ℝ, line x y ∧ circle x y :=
by
  sorry

end line_intersects_circle_l203_203344


namespace count_two_digit_primes_with_ones_3_l203_203875

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203875


namespace circumcenter_iff_perimeter_condition_l203_203290

structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)

def is_perpendicular (p1 p2 p3 : Point) : Prop := 
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

def perimeter (t : Triangle) : ℝ := 
  let d (p1 p2 : Point) := real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)
  d t.A t.B + d t.B t.C + d t.C t.A

def circumcenter (t : Triangle) (p : Point) : Prop :=
  let d (p1 p2 : Point) := real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)
  d p t.A = d p t.B ∧ d p t.B = d p t.C

noncomputable def perpendicular_meets (t : Triangle) (p d e f : Point) : Prop := 
  is_perpendicular p d t.B ∧ is_perpendicular p d t.C ∧
  is_perpendicular p e t.C ∧ is_perpendicular p e t.A ∧
  is_perpendicular p f t.A ∧ is_perpendicular p f t.B

theorem circumcenter_iff_perimeter_condition (t : Triangle) (p d e f : Point)
  (hPD : perpendicular_meets t p d e f) :
  (circumcenter t p) ↔ 
  (perimeter (Triangle.mk t.A e f) ≤ perimeter (Triangle.mk d e f) ∧
   perimeter (Triangle.mk t.B d f) ≤ perimeter (Triangle.mk d e f) ∧
   perimeter (Triangle.mk t.C d e) ≤ perimeter (Triangle.mk d e f)) :=
sorry

end circumcenter_iff_perimeter_condition_l203_203290


namespace crayons_given_to_mary_l203_203447

theorem crayons_given_to_mary :
  let pack_crayons := 21 in
  let locker_crayons := 36 in
  let bobby_crayons := locker_crayons / 2 in
  let total_crayons := pack_crayons + locker_crayons + bobby_crayons in
  (total_crayons * (1 / 3) = 25) := by
rfl

end crayons_given_to_mary_l203_203447


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203904

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203904


namespace number_of_primes_in_interval_l203_203693

theorem number_of_primes_in_interval (n : ℕ) (h : 2 < n) : 
  ∃ p : ℕ, p.prime ∧ (n! - n) < p ∧ p < n! :=
sorry

end number_of_primes_in_interval_l203_203693


namespace two_digit_primes_ending_in_3_eq_6_l203_203937

open Nat

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def ends_in_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def count_two_digit_primes_ending_in_3 : ℕ :=
  ([13, 23, 33, 43, 53, 63, 73, 83, 93].filter (λ n, is_prime n ∧ is_two_digit n ∧ ends_in_digit_3 n)).length

theorem two_digit_primes_ending_in_3_eq_6 : count_two_digit_primes_ending_in_3 = 6 :=
by
  sorry

end two_digit_primes_ending_in_3_eq_6_l203_203937


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203966

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203966


namespace limit_result_l203_203398

open Real

noncomputable def limit_expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - x + 1) - 1) / (tan (π * x))

theorem limit_result : tendsto limit_expression (𝓝 1) (𝓝 (1 / (2 * π))) :=
by
  sorry

end limit_result_l203_203398


namespace tan_105_l203_203472

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203472


namespace no_real_solutions_l203_203100

theorem no_real_solutions :
  ∀ x : ℝ, (2 * x - 6) ^ 2 + 4 ≠ -(x - 3) :=
by
  intro x
  sorry

end no_real_solutions_l203_203100


namespace tan_105_l203_203544

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203544


namespace optimal_hospital_location_l203_203365

-- Define the coordinates for points A, B, and C
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the distance function
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the statement to be proved: minimizing sum of squares of distances
theorem optimal_hospital_location : ∃ y : ℝ, 
  (∀ (P : ℝ × ℝ), P = (0, y) → (dist_sq P A + dist_sq P B + dist_sq P C) = 146) ∧ y = 4 :=
by sorry

end optimal_hospital_location_l203_203365


namespace tan_105_eq_neg2_sub_sqrt3_l203_203518

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203518


namespace john_total_cost_l203_203251

-- The total cost John incurs to rent a car, buy gas, and drive 320 miles
def total_cost (rental_cost gas_cost_per_gallon cost_per_mile miles driven_gallons : ℝ): ℝ :=
  rental_cost + (gas_cost_per_gallon * driven_gallons) + (cost_per_mile * miles)

theorem john_total_cost :
  let rental_cost := 150
  let gallons := 8
  let gas_cost_per_gallon := 3.50
  let cost_per_mile := 0.50
  let miles := 320
  total_cost rental_cost gas_cost_per_gallon cost_per_mile miles gallons = 338 := 
by
  -- The detailed proof is skipped here
  sorry

end john_total_cost_l203_203251


namespace tan_add_tan_105_eq_l203_203641

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203641


namespace two_digit_primes_with_ones_digit_three_count_l203_203777

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203777


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203968

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203968


namespace tan_105_degree_l203_203597

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203597


namespace base12_division_remainder_l203_203388

theorem base12_division_remainder :
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3 in
  n % 9 = 8 :=
by
  let n := 2 * (12^3) + 5 * (12^2) + 4 * 12 + 3
  show n % 9 = 8
  sorry

end base12_division_remainder_l203_203388


namespace maximum_cars_quotient_l203_203288

theorem maximum_cars_quotient
  (car_length : ℕ) (m_speed : ℕ) (half_hour_distance : ℕ) 
  (unit_length : ℕ) (max_units : ℕ) (N : ℕ) :
  (car_length = 5) →
  (half_hour_distance = 10000) →
  (unit_length = 5 * (m_speed + 1)) →
  (max_units = half_hour_distance / unit_length) →
  (N = max_units) →
  (N / 10 = 200) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end maximum_cars_quotient_l203_203288


namespace count_two_digit_primes_ending_in_3_l203_203829

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203829


namespace functional_eq_l203_203169

variable {ℝ : Type*} [AddCommGroup ℝ] [Module ℝ ℝ]

theorem functional_eq 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
  (h2 : f 2 = 4) : 
  f 1 = 2 :=
sorry

end functional_eq_l203_203169


namespace domain_of_f_l203_203329

def domain_condition1 (x : ℝ) : Prop := 1 - |x - 1| > 0
def domain_condition2 (x : ℝ) : Prop := x - 1 ≠ 0

theorem domain_of_f :
  (∀ x : ℝ, domain_condition1 x ∧ domain_condition2 x → 0 < x ∧ x < 2 ∧ x ≠ 1) ↔
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2)) :=
by
  sorry

end domain_of_f_l203_203329


namespace price_reduction_ensures_profit_l203_203432

-- Define the problem constants:
def average_daily_sale : ℕ := 20
def profit_per_piece : ℕ := 40
def daily_profit_target : ℤ := 1200

-- Define the variable for price reduction and the new quantities:
variable (x : ℝ) -- price reduction in yuan

-- Define equations based on conditions:
def pieces_sold_per_day := average_daily_sale + (2 : ℕ) * x
def profit_per_jacket := profit_per_piece - x

-- Define the equation based on desired profit:
def daily_profit : ℝ := profit_per_jacket * pieces_sold_per_day

-- The statement to prove:
theorem price_reduction_ensures_profit :
  (40 - x) * (20 + 2 * x) = 1200 :=
by
  sorry

end price_reduction_ensures_profit_l203_203432


namespace farm_horses_cows_diff_l203_203020

noncomputable def horses_cows_diff : ℕ :=
  let initial_horses (x : ℕ) := 3 * x in
  let initial_cows (x : ℕ) := x in
  let final_horses (x : ℕ) := initial_horses x - 15 in
  let final_cows (x : ℕ) := initial_cows x + 15 in
  75 -- Given directly from the derived value of 3x - 15 when x = 30

theorem farm_horses_cows_diff (x : ℕ) (h1 : 3 * (3 * x - 15) = 5 * (x + 15))
  : horses_cows_diff = 30 := by
  sorry

end farm_horses_cows_diff_l203_203020


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203511

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203511


namespace tan_105_l203_203490

theorem tan_105 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  -- Definitions
  let tan45 := Real.tan (45 * Real.pi / 180)
  let tan60 := Real.tan (60 * Real.pi / 180)
  have h1 : tan45 = 1 := sorry
  have h2 : tan60 = Real.sqrt 3 := sorry
  have h3 : tan45 + tan60 = 1 + Real.sqrt 3 := sorry
  have h4 : 1 - tan45 * tan60 = 1 - 1 * Real.sqrt 3 := sorry
  
  -- Use tangent addition formula
  have tan_addition : Real.tan (105 * Real.pi / 180) = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) := sorry
  
  -- Simplify and rationalize to prove the final result
  have tan_105_eq : (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = -2 - Real.sqrt 3 := sorry
  
  exact tan_105_eq

end tan_105_l203_203490


namespace count_two_digit_primes_ending_in_3_l203_203818

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203818


namespace num_two_digit_primes_with_ones_digit_3_l203_203958

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203958


namespace range_of_x_l203_203205

theorem range_of_x (x : ℝ) : sqrt ((5 - x) ^ 2) = x - 5 → x ≥ 5 :=
by
  sorry

end range_of_x_l203_203205


namespace tan_105_eq_neg2_sub_sqrt3_l203_203532

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203532


namespace leila_total_expenditure_l203_203256

variable (cost_auto cost_market total : ℕ)
variable (h1 : cost_auto = 350)
variable (h2 : cost_auto = 3 * cost_market + 50)

theorem leila_total_expenditure : total = 450 :=
by
  have h3 : cost_market = 100 := by
    calc
      cost_market = (350 - 50) / 3 := by rw [← h2, ← h1]
      ... = 100 : by norm_num
  have h4 : total = cost_auto + cost_market := by norm_num
  calc
    total = 350 + 100 := by rw [h4, h1, h3]
    ... = 450 : by norm_num

end leila_total_expenditure_l203_203256


namespace population_aging_issue_l203_203094

/-- Conditions of the modern population growth model. -/
variables (low_birth_rates low_death_rates low_natural_growth_rates : Prop)

/-- The main population issue faced by both developed and developing countries in the second half of the 21st century is population aging. -/
theorem population_aging_issue 
  (h1 : low_birth_rates)
  (h2 : low_death_rates)
  (h3 : low_natural_growth_rates) :
  ∃ issue, issue = "population aging" :=
sorry

end population_aging_issue_l203_203094


namespace pages_in_first_chapter_l203_203404

theorem pages_in_first_chapter (x : ℕ) (h1 : x + 43 = 80) : x = 37 :=
by
  sorry

end pages_in_first_chapter_l203_203404


namespace count_two_digit_primes_ending_in_3_l203_203830

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203830


namespace tan_105_eq_neg_2_sub_sqrt_3_l203_203475

-- Definitions
def angle105 : ℝ := 105 * (Math.pi / 180)
def angle45 : ℝ := 45 * (Math.pi / 180)
def angle60 : ℝ := 60 * (Math.pi / 180)

-- Theorem
theorem tan_105_eq_neg_2_sub_sqrt_3 :
  Real.tan angle105 = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg_2_sub_sqrt_3_l203_203475


namespace tan_105_degree_is_neg_sqrt3_minus_2_l203_203515

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l203_203515


namespace infinite_equilateral_triangles_in_M_l203_203402

open Complex

-- Define the set M as complex numbers on the unit circle with rational real parts
def M : Set ℂ := { z | abs z = 1 ∧ (z.re ∈ ℚ) }

-- Statement of the proof problem
theorem infinite_equilateral_triangles_in_M : ∃ (infinitely_many (triangle_vertices : Finset ℂ)), 
  (∀ z ∈ triangle_vertices, z ∈ M) ∧
  (∀ (a b c : ℂ) (h : {a, b, c} = triangle_vertices), 
    ∃ θ : ℝ, a = 1 * exp (θ * I) ∧ b = exp ((θ + 2 * π / 3) * I) ∧ c = exp ((θ + 4 * π / 3) * I)) :=
sorry

end infinite_equilateral_triangles_in_M_l203_203402


namespace part1_part2_l203_203730

open Real

noncomputable def a_seq (n : ℕ) : ℝ :=
  2 ^ n

def b_seq (n : ℕ) : ℝ :=
  1 / (log 2 (a_seq n))^2

noncomputable def T_seq (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), b_seq k

theorem part1 (n : ℕ) : 
  ∑ k in Finset.range (n + 1), a_seq k = 2 * a_seq n - 2 :=
by
  sorry

theorem part2 (n : ℕ) (h : n ≥ 4) : 
  T_seq n < 61 / 36 :=
by
  sorry

end part1_part2_l203_203730


namespace distance_between_points_l203_203375

theorem distance_between_points : 
  let x1 := 1
  let y1 := -3
  let x2 := -4 
  let y2 := 5 in
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = Real.sqrt 89 :=
by
  sorry

end distance_between_points_l203_203375


namespace tan_105_l203_203468

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203468


namespace general_form_of_line_l_l203_203414

-- Define the point
def pointA : ℝ × ℝ := (1, 2)

-- Define the normal vector
def normalVector : ℝ × ℝ := (1, -3)

-- Define the general form equation
def generalFormEq (x y : ℝ) : Prop := x - 3 * y + 5 = 0

-- Statement to prove
theorem general_form_of_line_l (x y : ℝ) (h_pointA : pointA = (1, 2)) (h_normalVector : normalVector = (1, -3)) :
  generalFormEq x y :=
sorry

end general_form_of_line_l_l203_203414


namespace distance_from_center_to_origin_eq_sqrt_5_l203_203173

noncomputable def distanceFromCenterToOrigin : ℝ :=
  let circleEquation := (x : ℝ) (y : ℝ) ↦ x^2 + y^2 - 4 * x + 2 * y + 2 = 0
  let center := (2, -1)
  euclideanDistance center (0, 0)

theorem distance_from_center_to_origin_eq_sqrt_5 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y + 2 = 0 → distanceFromCenterToOrigin = Real.sqrt 5 :=
by
  -- proof here
  sorry

end distance_from_center_to_origin_eq_sqrt_5_l203_203173


namespace tan_105_degree_l203_203588

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  have h_add : ∀ a b : ℝ, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    Real.tan_add

  have h_tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := Real.tan_60
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := Real.tan_45
  
  sorry

end tan_105_degree_l203_203588


namespace area_triangle_ABC_is_6m_l203_203241

-- Define the entities: points and medians
variables {A B C M N O D R : Type} 
variables [PlaneGeometry A B C M N O D R]

-- Conditions from the problem
-- 1. Medians AM and CN intersect at centroid O
-- 2. D is the midpoint of BC
-- 3. AD intersects CN at point R
-- 4. Area of triangle ORD is m

axiom Intersection_at_centroid (triangle_medians_intersect_at_centroid : ∃ O, is_centroid A M C N O)
axiom Midpoint_of_BC (midpoint_D : D = midpoint B C)
axiom Intersection_at_R (intersection_AD_CN : ∃ R, is_intersection A D C N R)
axiom Area_ORD (area_ORD : Real) (m : Real) : area A O D = m

-- Prove the area of triangle ABC is 6m
theorem area_triangle_ABC_is_6m :
  ∃ m, area A B C = 6 * m :=
by
  exact sorry

end area_triangle_ABC_is_6m_l203_203241


namespace circle_constant_ratio_l203_203709

theorem circle_constant_ratio (b : ℝ) :
  (∀ (x y : ℝ), (x + 4)^2 + (y + b)^2 = 16 → 
    ∃ k : ℝ, 
      ∀ P : ℝ × ℝ, 
        P = (x, y) → 
        dist P (-2, 0) / dist P (4, 0) = k)
  → b = 0 :=
by
  intros h
  sorry

end circle_constant_ratio_l203_203709


namespace second_valve_emits_more_l203_203390

noncomputable def V1 : ℝ := 12000 / 120 -- Rate of first valve (100 cubic meters/minute)
noncomputable def V2 : ℝ := 12000 / 48 - V1 -- Rate of second valve

theorem second_valve_emits_more : V2 - V1 = 50 :=
by
  sorry

end second_valve_emits_more_l203_203390


namespace more_chocolate_than_raisin_l203_203757

noncomputable def helen_yesterday_chocolate := 19
noncomputable def helen_morning_chocolate := 237
noncomputable def helen_raisin := 231
noncomputable def helen_oatmeal := 107

noncomputable def giselle_chocolate := 156
noncomputable def giselle_raisin := 89

noncomputable def timmy_chocolate := 135
noncomputable def timmy_oatmeal := 246

noncomputable def total_chocolate_baked := helen_yesterday_chocolate + helen_morning_chocolate + giselle_chocolate + timmy_chocolate
noncomputable def total_raisin_baked := helen_raisin + giselle_raisin

theorem more_chocolate_than_raisin : total_chocolate_baked - total_raisin_baked = 227 := by
  calc
    total_chocolate_baked - total_raisin_baked
        = (19 + 237 + 156 + 135) - (231 + 89) : by
          simp [helen_yesterday_chocolate, helen_morning_chocolate, giselle_chocolate, timmy_chocolate, helen_raisin, giselle_raisin]
        ... = 547 - 320 : by norm_num
        ... = 227 : by norm_num

end more_chocolate_than_raisin_l203_203757


namespace unique_valid_permutation_l203_203662

def is_valid_permutation (perm : Fin 2021 → ℕ) :=
  perm.toList.perm (List.range' 1 2021)

theorem unique_valid_permutation :
  ∀ perm : Fin 2021 → ℕ,
  is_valid_permutation perm →
  (∀ m n : ℕ, 0 < m ∧ 0 < n ∧ m - n > 20^21 →
    (∑ i in Finset.range 2021, Nat.gcd (m + i + 1) (n + perm ⟨i, nat.lt_of_lt_of_le (Finset.mem_range.1 (Finset.mem_range_self i)) 2021⟩)) < 2 * |m - n|) →
  perm = λ i => i + 1
:= by
  sorry

end unique_valid_permutation_l203_203662


namespace tan_105_eq_neg2_sub_sqrt3_l203_203529

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203529


namespace P_inequality_l203_203262

variable {α : Type*} [LinearOrderedField α]

def P (a b c : α) (x : α) : α := a * x^2 + b * x + c

theorem P_inequality (a b c x y : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (P a b c (x * y))^2 ≤ (P a b c (x^2)) * (P a b c (y^2)) :=
sorry

end P_inequality_l203_203262


namespace line_through_point_parallel_l203_203684

theorem line_through_point_parallel (x y : ℝ) (h₁ : 2 * 2 + 4 * 3 + x = 0) (h₂ : x = -16) (h₃ : y = 8) :
  2 * x + 4 * y - 3 = 0 → x + 2 * y - 8 = 0 :=
by
  intro h₄
  sorry

end line_through_point_parallel_l203_203684


namespace lean_proof_problem_l203_203181

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

-- Define the conditions for the monotonicity properties
-- Use all_params for each of the statement to use respectively for the each derivation chain of the inputs.
def monotonicity_properties (a : ℝ) (x : ℝ) : Prop := 
  if a = 0 then 
    x > 0 → (x ∈ Ioi (1/2) → (¬ Real.derivation x ∈ Ioi 0)) ∧ 
    (x ∈ Iio (1/2) → (Real.derivation x ∈ Iio 0) ∧ Real.derivation x ∉ Ioi 0)
  else if a > 0 then 
      (x ∈ Ioi (1/2)) → (Real.derivation x ∈ Ioi 0 ∧ Real.derivation x ∉ Iio 0) ∧ 
      (x ∈ Iio (1/2)) → (Real.derivation x ∈ Iio 0 ∧ Real.derivation x ∉ Ioi 0)
  else if -2 < a < 0 then 
      (x ∈ Ico (1/2, -1/a)) → (Real.derivation x ∈ Ioi 0 ∧ Real.derivation x ∉ Iio 0) ∧ 
      (x ∈ Iio (1/2) ∧ x ∉ Ico (1/2, -1/a)) → (Real.derivation x ∈ Iio 0)
  else if a < -2 then 
      (x ∈ Ico (-1/a, 1/2)) → (Real.derivation x ∈ Ioi 0 ∧ Real.derivation x ∉ Iio 0) ∧ 
      (x ∈ Iio (-1/a) ∧ x ∉ Ico (-1/a, 1/2) ∧ x ∉ Ioi (1/2) )  → (Real.derivation x ∈ Iio 0 ∨ Real.derivation x ∉ Ioi 0)
  else
      (a = -2) → (x ∈ Ioi 0) → (Real.derivation x ∈ Iio 0 ∧ Real.derivation x ∉ Ioi 0)


-- Define the inequality conditions
def inequality_conditions (a : ℝ) (m : ℝ) (x1 x2 : ℝ): Prop :=
  if a ∈ Set.Ioo (-3) (-2) then 
    x1 ∈ Set.Icc 1 3 ∧ x2 ∈ Set.Icc 1 3
    → (m + Real.log 3) * a - 2 * Real.log 3 > Real.dist (f a x1) (f a x2)  

-- Define the main Lean statement
theorem lean_proof_problem (a m x1 x2 : ℝ) :
  (monotonicity_properties a x) → 
  inequality_conditions a m x1 x2
  →  m ≤ -13 / 3 :=
sorry

end lean_proof_problem_l203_203181


namespace largest_among_four_l203_203141

theorem largest_among_four (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  max (max a (max (a + b) (a - b))) (ab) = a - b :=
by {
  sorry
}

end largest_among_four_l203_203141


namespace time_with_walkway_l203_203421

theorem time_with_walkway (v w : ℝ) (t : ℕ) :
  (80 = 120 * (v - w)) → 
  (80 = 60 * v) → 
  t = 80 / (v + w) → 
  t = 40 :=
by
  sorry

end time_with_walkway_l203_203421


namespace count_two_digit_primes_with_ones_digit_three_l203_203794

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203794


namespace five_digit_palindrome_probability_divisible_by_11_l203_203420

-- defining the structure of a five-digit palindrome
def is_palindrome (x : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 10001 * a + 1010 * b + 100 * c

-- defining divisibility by 11
def divisible_by_eleven (x : ℕ) : Prop :=
  x % 11 = 0

-- count and probability calculations as per the proof problem conditions
def five_digit_palindromes : finset ℕ :=
  finset.filter is_palindrome (finset.range 100000).filter (λ n, n ≥ 10000)

def valid_palindromes : finset ℕ :=
  five_digit_palindromes.filter divisible_by_eleven

noncomputable def valid_count : ℕ := valid_palindromes.card
noncomputable def total_count : ℕ := five_digit_palindromes.card
noncomputable def probability : ℚ := valid_count / total_count

theorem five_digit_palindrome_probability_divisible_by_11 :
  probability = 41 / 450 :=
by
  sorry

end five_digit_palindrome_probability_divisible_by_11_l203_203420


namespace two_digit_primes_with_ones_digit_3_l203_203867

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203867


namespace article_use_correctness_l203_203025

theorem article_use_correctness :
  (∀ n : String, 
     n = "sky" ∨ n = "world" → 
     (n = "sky" → "a " ++ "bluer " ++ n = "a bluer sky") ∧ 
     (n = "world" → "a " ++ "less polluted " ++ n = "a less polluted world")) →
  "We can never expect a bluer sky unless we create a less polluted world." :=
by 
  sorry

end article_use_correctness_l203_203025


namespace count_two_digit_primes_with_ones_3_l203_203880

open Nat

/-- Predicate to check if a number is a two-digit prime with ones digit 3. --/
def two_digit_prime_with_ones_3 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n

/-- Prove that there are exactly 6 two-digit primes with ones digit 3. --/
theorem count_two_digit_primes_with_ones_3 : 
  (Finset.filter two_digit_prime_with_ones_3 (Finset.range 100)).card = 6 := 
  by
  sorry

end count_two_digit_primes_with_ones_3_l203_203880


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203960

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203960


namespace total_distance_is_east_fuel_consumption_correct_total_fare_correct_l203_203059

-- Define the problem constants and travel distances
def distances : List Int := [5, -4, 2, -3, 8]
def east_direction : Int := 8
def fuel_rate : Float := 0.2
def total_fuel_consumed : Float := 4.4
def fare_not_exceeding_3 : Int := 7
def fare_per_km_exceeding_3 : Float := 1.5
def total_fare : Float := 47.0

-- Prove that the total distance after all travels
theorem total_distance_is_east : distances.sum = east_direction :=
  sorry

-- Prove that the total fuel consumed during the trips
theorem fuel_consumption_correct :
  distances.map Int.natAbs.sum * fuel_rate = total_fuel_consumed :=
  sorry

-- Prove that the total fare received is the correct amount
theorem total_fare_correct : 
  calc
  (distances.get! 0 - 3).natAbs * fare_per_km_exceeding_3 + fare_not_exceeding_3 +
  (distances.get! 1 - 3).natAbs * fare_per_km_exceeding_3 + fare_not_exceeding_3 +
  fare_not_exceeding_3 +
  fare_not_exceeding_3 +
  (distances.get! 4 - 3).natAbs * fare_per_km_exceeding_3 + fare_not_exceeding_3 = 
  total_fare :=
  sorry

end total_distance_is_east_fuel_consumption_correct_total_fare_correct_l203_203059


namespace tan_105_l203_203550

-- Defining the necessary known values and functions
def tan_addition (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)
def tan_60 : ℝ := Real.sqrt 3
def tan_45 : ℝ := 1

-- Proof goal in Lean 4
theorem tan_105 : Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  let tan_105 := tan_addition (60 * Real.pi / 180) (45 * Real.pi / 180)
  have h1 : Real.tan (60 * Real.pi / 180) = tan_60 := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = tan_45 := by sorry
  show Real.tan (105 * Real.pi / 180) = 2 + Real.sqrt 3 from sorry

end tan_105_l203_203550


namespace ordered_pairs_logarithmic_eq_l203_203663

theorem ordered_pairs_logarithmic_eq :
  let pairs_count : ℕ := 
    (finite_set_enumeration (set.finset (Ioo 0 real_top))  (λ (a : ℝ), ∃ (b : ℕ), b ∈ range 1 51  ∧ (log b a) ^ 3 = log b (a ^ 3))).card
  in pairs_count = 150 :=
sorry

end ordered_pairs_logarithmic_eq_l203_203663


namespace number_of_bags_of_chips_l203_203367

theorem number_of_bags_of_chips (friends : ℕ) (amount_per_friend : ℕ) (cost_per_bag : ℕ) (total_amount : ℕ) (number_of_bags : ℕ) : 
  friends = 3 → amount_per_friend = 5 → cost_per_bag = 3 → total_amount = friends * amount_per_friend → number_of_bags = total_amount / cost_per_bag → number_of_bags = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_bags_of_chips_l203_203367


namespace two_digit_primes_end_in_3_l203_203907

theorem two_digit_primes_end_in_3 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ n % 10 = 3 ∧ Prime n}.card = 6 := 
by
  sorry

end two_digit_primes_end_in_3_l203_203907


namespace tan_105_eq_neg2_sub_sqrt3_l203_203538

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203538


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203973

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203973


namespace limit_sequence_value_l203_203455

noncomputable def limit_of_sequence : Real :=
  limit (fun n : ℕ => ((n + 1) ^ 4 - (n - 1) ^ 4) / ((n + 1) ^ 3 + (n - 1) ^ 3))

theorem limit_sequence_value : limit_of_sequence = 4 := 
by 
  sorry

end limit_sequence_value_l203_203455


namespace rectangle_perimeter_l203_203041

theorem rectangle_perimeter :
  ∃ (a b : ℕ), (a ≠ b) ∧ (a * b = 2 * (a + b) - 4) ∧ (2 * (a + b) = 26) :=
by {
  sorry
}

end rectangle_perimeter_l203_203041


namespace sum_of_primes_no_solution_congruence_l203_203104

theorem sum_of_primes_no_solution_congruence :
  (∀ x : ℤ, ∃ p : ℕ, Prime p ∧ ¬ (5 * (10 * x + 2) ≡ 3 [MOD p])) →
  ∑ x in ({2, 5} : Finset ℕ), x = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l203_203104


namespace num_two_digit_primes_with_ones_digit_3_l203_203950

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203950


namespace mod_pairs_unique_l203_203093

theorem mod_pairs_unique :
  ∀ (x y : ℕ), (1 ≤ x ∧ x ≤ 10) ∧ (1 ≤ y ∧ y ≤ 10) ∧ (x ≠ y) →
  (x % 2, x % 5) ≠ (y % 2, y % 5) :=
by {
  assume x y,
  assume h : (1 ≤ x ∧ x ≤ 10) ∧ (1 ≤ y ∧ y ≤ 10) ∧ (x ≠ y),
  sorry
}

end mod_pairs_unique_l203_203093


namespace count_isosceles_triangle_digits_l203_203281

theorem count_isosceles_triangle_digits :
  let count_isosceles := λ (a b c : ℕ), (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) in
  ∑ (a b c : ℕ) in finset.range 10, if (1 ≤ a) ∧ count_isosceles a b c then 1 else 0 = 165 :=
by sorry

end count_isosceles_triangle_digits_l203_203281


namespace sum_squares_coefficients_equals_1231_l203_203310

def initial_expression := λ (x : ℝ), 3 * (x^2 - 3 * x + 3) - 5 * (x^3 - 2 * x^2 + 4 * x - 1)

theorem sum_squares_coefficients_equals_1231 : 
  let expr := initial_expression in
  let simp_expr := -5 * x^3 + 13 * x^2 - 29 * x + 14 in
  let coeffs := [-5, 13, -29, 14] in
  (coeffs.foldl (λ acc c, acc + c^2) 0) = 1231 := by
  sorry

end sum_squares_coefficients_equals_1231_l203_203310


namespace m_n_units_digit_6_l203_203042

-- Define the sets of numbers m and n
def set_m : Finset ℕ := {12, 14, 16, 18, 20}
def set_n : Finset ℕ := Finset.range (2024 - 2000) |>.map (λ x, x + 2000)

noncomputable def probability_units_digit_6 : ℚ :=
  if h : ∀ m ∈ set_m, ∀ n ∈ set_n, (m % 10) ^ n % 10 = 6 then
    1 / 4
  else
    0

theorem m_n_units_digit_6 :
  probability_units_digit_6 = 1 / 4 :=
by
  unfold probability_units_digit_6
  split
  {
    sorry -- This is where the proof would go
  }
  {
    sorry -- This branch is here because we need a sound proof
  }

end m_n_units_digit_6_l203_203042


namespace main_theorem_l203_203260

variable (O A B C A1 A2 : Point)
variable (R : ℝ)
variable (circle_k : Circle O R)
variable (circle_k1 : Circle A1 (radius1 : ℝ)) (tangent_k1_AB : Tangent circle_k1 (Line A B)) (tangent_k1_AC : Tangent circle_k1 (Line A C)) (internal_tangent_k1 : InternalTangent circle_k1 circle_k)
variable (circle_k2 : Circle A2 (radius2 : ℝ)) (tangent_k2_AB : Tangent circle_k2 (Line A B)) (tangent_k2_AC : Tangent circle_k2 (Line A C)) (external_tangent_k2 : ExternalTangent circle_k2 circle_k)

theorem main_theorem : ((distance O A1 + distance O A2)^2 - (distance A1 A2)^2 = 4 * R^2) :=
sorry

end main_theorem_l203_203260


namespace point_on_line_l203_203189

theorem point_on_line (a : ℝ) :
  let A := (1 : ℝ, -1 : ℝ)
  let B := (3 : ℝ, 3 : ℝ)
  let C := (5 : ℝ, a)
  (C.2 - A.2) / (C.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) -> a = 7 :=
by
  let A := (1, -1) : ℝ × ℝ
  let B := (3, 3) : ℝ × ℝ
  let C := (5, a) : ℝ × ℝ
  have slope_AB : (B.2 - A.2) / (B.1 - A.1) = 2 := by sorry
  have slope_AC_eq_slope_AB : (C.2 - A.2) / (C.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) := by sorry
  have a_eq_7 : a = 7 := by sorry
  exact a_eq_7

end point_on_line_l203_203189


namespace widgets_per_shipping_box_l203_203670

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l203_203670


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203896

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203896


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l203_203998

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l203_203998


namespace num_two_digit_primes_with_ones_digit_3_l203_203944

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203944


namespace tan_add_tan_105_eq_l203_203636

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203636


namespace limit_result_l203_203397

open Real

noncomputable def limit_expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - x + 1) - 1) / (tan (π * x))

theorem limit_result : tendsto limit_expression (𝓝 1) (𝓝 (1 / (2 * π))) :=
by
  sorry

end limit_result_l203_203397


namespace project_completion_time_l203_203014

-- Definitions for conditions
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def combined_rate : ℚ := a_rate + b_rate

-- Total days to complete the project
def total_days (x : ℚ) : Prop :=
  (x - 5) * a_rate + x * b_rate = 1

-- The theorem to be proven
theorem project_completion_time : ∃ (x : ℚ), total_days x ∧ x = 15 := by
  sorry

end project_completion_time_l203_203014


namespace tan_105_eq_neg2_sub_sqrt3_l203_203585

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203585


namespace complement_union_eq_self_l203_203282

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Univ : U = (univ : Set ℝ)]
variable [A_set : A = (Iio (-1) ∪ Ioi 1 : Set ℝ)]
variable [B_set : B = (Icc (-1) ∞ : Set ℝ)]

theorem complement_union_eq_self : ((U \ A) ∪ B) = B := by
  sorry

end complement_union_eq_self_l203_203282


namespace sqrt_sum_of_fractions_is_correct_l203_203116

def evaluate_sqrt_sum_of_fractions : Prop :=
  sqrt (1 / 25 + 1 / 36) = sqrt 61 / 30

theorem sqrt_sum_of_fractions_is_correct : evaluate_sqrt_sum_of_fractions := by
  sorry

end sqrt_sum_of_fractions_is_correct_l203_203116


namespace length_proof_l203_203339

noncomputable def length_of_plot 
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ) -- cost of fencing per meter on flat ground
  (height_rise : ℝ) -- total height rise in meters
  (total_cost: ℝ) -- total cost of fencing
  (length_increase : ℝ) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ) -- scaling factor for cost increase on breadth
  (increased_breadth_cost_rate : ℝ) -- actual increased cost rate per meter for breadth
: ℝ :=
2 * (b + length_increase) * fence_cost_flat + 
2 * b * (fence_cost_flat + fence_cost_flat * (height_rise * cost_increase_rate))

theorem length_proof
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ := 26.50) -- cost of fencing per meter on flat ground
  (height_rise : ℝ := 5) -- total height rise in meters
  (total_cost: ℝ := 5300) -- total cost of fencing
  (length_increase : ℝ := 20) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ := 0.10) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ := fence_cost_flat * 0.5) -- increased cost factor
  (increased_breadth_cost_rate : ℝ := 39.75) -- recalculated cost rate per meter for breadth
  (length: ℝ := b + length_increase)
  (proof_step : total_cost = length_of_plot b fence_cost_flat height_rise total_cost length_increase cost_increase_rate breadth_cost_increase_factor increased_breadth_cost_rate)
: length = 52 :=
by
  sorry -- Proof omitted

end length_proof_l203_203339


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203600

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203600


namespace tan_add_tan_105_eq_l203_203633

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l203_203633


namespace num_two_digit_primes_with_ones_digit_3_l203_203945

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203945


namespace harmonic_odd_numerator_even_denominator_l203_203178

theorem harmonic_odd_numerator_even_denominator (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), nat.coprime a b ∧ is_odd a ∧ is_even b ∧ H n = a / b :=
by sorry

noncomputable def H (n : ℕ) : ℚ :=
∑ k in finset.range (n + 1), 1 / (k + 1)

end harmonic_odd_numerator_even_denominator_l203_203178


namespace avg_speed_ratio_l203_203325

theorem avg_speed_ratio 
  (dist_tractor : ℝ) (time_tractor : ℝ) 
  (dist_car : ℝ) (time_car : ℝ) 
  (speed_factor : ℝ) :
  dist_tractor = 575 -> 
  time_tractor = 23 ->
  dist_car = 450 ->
  time_car = 5 ->
  speed_factor = 2 ->

  (dist_car / time_car) / (speed_factor * (dist_tractor / time_tractor)) = 9/5 := 
by
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end avg_speed_ratio_l203_203325


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203607

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203607


namespace regular_polygon_sides_and_interior_angle_l203_203729

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l203_203729


namespace count_two_digit_primes_with_ones_digit_three_l203_203795

def is_prime (n : ℕ) : Prop := nat.prime n

def ones_digit_three (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem count_two_digit_primes_with_ones_digit_three : 
  {n : ℕ | two_digit_number n ∧ ones_digit_three n ∧ is_prime n}.to_finset.card = 6 :=
sorry

end count_two_digit_primes_with_ones_digit_three_l203_203795


namespace widgets_per_shipping_box_l203_203667

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l203_203667


namespace distinct_possible_lunches_l203_203079

def main_dishes := 3
def beverages := 3
def snacks := 3

theorem distinct_possible_lunches : main_dishes * beverages * snacks = 27 := by
  sorry

end distinct_possible_lunches_l203_203079


namespace line_through_point_with_direction_vector_l203_203716

def point := ℝ × ℝ

def line_equation (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem line_through_point_with_direction_vector (A : point) (d : point) :
  A = (1, 0) → d = (3, -1) → ∃ (m b : ℝ), line_equation m b A.1 A.2 ∧ (∃ (x y : ℝ), line_equation m b x y ∧ x + 3*y - 1 = 0) :=
begin
  intros hA hd,
  use -1/3,
  use 1/3,
  split,
  { rw [line_equation, hA, hd],
    norm_num },
  { use 1,
    use 0,
    split,
    { rw [line_equation, hA],
      norm_num },
    norm_num }
end

end line_through_point_with_direction_vector_l203_203716


namespace tan_105_degree_l203_203566

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203566


namespace units_digit_k_squared_plus_2_k_is_7_l203_203273

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_k_squared_plus_2_k_is_7 : (k^2 + 2^k) % 10 = 7 :=
by sorry

end units_digit_k_squared_plus_2_k_is_7_l203_203273


namespace Ted_has_15_bags_l203_203318

-- Define the parameters
def total_candy_bars : ℕ := 75
def candy_per_bag : ℝ := 5.0

-- Define the assertion to be proved
theorem Ted_has_15_bags : total_candy_bars / candy_per_bag = 15 := 
by
  sorry

end Ted_has_15_bags_l203_203318


namespace max_elements_l203_203708

variables (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3)

noncomputable def S : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, (p.1 ∈ finset.range (m + 1)) ∧ (p.2 ∈ finset.range (n + 1)))

def condition (A : finset (ℕ × ℕ)) : Prop :=
  ∀ x1 x2 x3 y1 y2 y3, x1 < x2 → x2 < x3 → y1 < y2 → y2 < y3 →
    (x1, y2) ∉ A ∨ (x2, y1) ∉ A ∨ (x2, y2) ∉ A ∨ (x2, y3) ∉ A ∨ (x3, y2) ∉ A

theorem max_elements (A : finset (ℕ × ℕ)) (hA : A ⊆ S m n) (h_cond : condition A) : A.card ≤ 2 * m + 2 * n - 4 :=
sorry

end max_elements_l203_203708


namespace real_part_of_complex_pow_l203_203127

open Complex

theorem real_part_of_complex_pow (a b : ℝ) : a = 1 → b = -2 → (realPart ((a : ℂ) + (b : ℂ) * Complex.I)^5) = 41 :=
by
  sorry

end real_part_of_complex_pow_l203_203127


namespace remainder_base12_2543_div_9_l203_203384

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l203_203384


namespace two_digit_primes_with_ones_digit_3_l203_203864

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec f (n : ℕ) : List ℕ :=
    if n = 0 then [] else (n % 10) :: f (n / 10)
  in List.reverse (f n)

def ends_with_3 (n : ℕ) : Prop :=
  digits n = (digits n).init ++ [3]

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_ones_digit_3 :
  (Finset.filter (λ n, is_prime n ∧ ends_with_3 n) (Finset.filter two_digit (Finset.range 100))).card = 6 := by
  sorry

end two_digit_primes_with_ones_digit_3_l203_203864


namespace num_two_digit_primes_with_ones_digit_3_l203_203957

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l203_203957


namespace solution_set_of_inequality_l203_203350

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l203_203350


namespace regular_polygon_sides_and_interior_angle_l203_203727

theorem regular_polygon_sides_and_interior_angle (n : ℕ) (H : (n - 2) * 180 = 3 * 360 + 180) :
  n = 9 ∧ (n - 2) * 180 / n = 140 :=
by
-- This marks the start of the proof, but the proof is omitted.
sorry

end regular_polygon_sides_and_interior_angle_l203_203727


namespace inequality_solution_l203_203347

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := |2 - 3 * x| ≥ 4

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ -2/3 ∨ x ≥ 2

-- The theorem that we need to prove
theorem inequality_solution : {x : ℝ | inequality_condition x} = {x : ℝ | solution_set x} :=
by sorry

end inequality_solution_l203_203347


namespace count_two_digit_primes_ending_with_3_l203_203849

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203849


namespace total_lives_l203_203406

-- Defining the number of lives for each animal according to the given conditions:
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7
def elephant_lives : ℕ := 2 * cat_lives - 5
def fish_lives : ℕ := if (dog_lives + mouse_lives) < (elephant_lives / 2) then (dog_lives + mouse_lives) else elephant_lives / 2

-- The main statement we need to prove:
theorem total_lives :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 :=
by
  sorry

end total_lives_l203_203406


namespace man_walking_speed_percentage_l203_203416

theorem man_walking_speed_percentage (T : ℕ) (d : ℕ) (P : ℚ) : T = 56 → T + d = 80 → P = (56 : ℚ) / 80 :=
by
  intros hT hd
  rw [hT] at hd
  linarith

end man_walking_speed_percentage_l203_203416


namespace tan_105_eq_neg2_sub_sqrt3_l203_203524

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l203_203524


namespace yerema_can_pay_exactly_l203_203692

theorem yerema_can_pay_exactly (t k b m : ℤ) 
    (h_foma : 3 * t + 4 * k + 5 * b = 11 * m) : 
    ∃ n : ℤ, 9 * t + k + 4 * b = 11 * n := 
by 
    sorry

end yerema_can_pay_exactly_l203_203692


namespace sum_of_binary_digits_345_l203_203005

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else List.reverse (List.unfold (λ n, if n = 0 then none else some (n % 2, n / 2)) n)

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.foldr (· + ·) 0
  
-- Define the specific example
def digits_of_345 : List ℕ := decimal_to_binary 345

def sum_of_digits_of_345 : ℕ := sum_of_digits digits_of_345

theorem sum_of_binary_digits_345 : sum_of_digits_of_345 = 5 :=
by 
  sorry

end sum_of_binary_digits_345_l203_203005


namespace reflections_of_candle_l203_203370

noncomputable def number_of_reflections (α : ℝ) : ℕ :=
  ⌊360 / α⌋ - 1

theorem reflections_of_candle (α : ℝ) (hα : 0 < α ∧ α ≤ 360) :
  number_of_reflections α = ⌊360 / α⌋ - 1 :=
by sorry

end reflections_of_candle_l203_203370


namespace distance_AB_l203_203394

theorem distance_AB (AC CR VR : ℝ) (hAC : AC = 4) (hCR : CR = 1) (hVR : VR = 3) : 
  AC / CR * VR = 12 :=
by {
  intro hAC hCR hVR,
  simp [hAC, hCR, hVR],
  sorry -- skip the proof
}

end distance_AB_l203_203394


namespace coefficient_x17_x18_l203_203233

theorem coefficient_x17_x18 (x : ℤ) :
  let f := (1 + x^5 + x^7) ^ 20 in
  (f.coeff 17 = 190) ∧ (f.coeff 18 = 0) := by
  sorry

end coefficient_x17_x18_l203_203233


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203897

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203897


namespace annie_gives_mary_25_crayons_l203_203444

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l203_203444


namespace tan_105_l203_203469

theorem tan_105 :
  tan 105 = -2 - sqrt 3 :=
by sorry

end tan_105_l203_203469


namespace find_k_parallel_l203_203193

theorem find_k_parallel (k : ℝ) : 
  let a := (3, 1)
      b := (1, 3)
      c := (k, 7) in
  -- Condition: (a - c) is parallel to b
  (λ a b, ∃ (λ : ℝ), (a.1 - b.1, a.2 - b.2) = (λ * b.1, λ * b.2)) (a) (a - c) -> k = 5 :=
by
  sorry

end find_k_parallel_l203_203193


namespace simplify_rationalize_denominator_l203_203308

-- Definitions from the conditions
def fraction_term : ℝ := 1 / (sqrt 5 + 2)
def simplified_term : ℝ := sqrt 5 - 2
def main_expression : ℝ := 1 / (2 + fraction_term)

theorem simplify_rationalize_denominator :
  main_expression = sqrt 5 / 5 := by
  sorry

end simplify_rationalize_denominator_l203_203308


namespace angle_between_planes_AC1K_AC1N_l203_203035

-- Definitions and given conditions
variables {A B C D A1 B1 C1 D1 K N : Point}
variables (cube : Cube ABCD A1 B1 C1 D1)
variables (inscribed_sphere : Sphere (center cube))
variables (tangent_plane : Plane)
variables (contains_A : A ∈ tangent_plane)
variables (intersects_K : K ∈ tangent_plane ∧ K ∈ Line A1 B1)
variables (intersects_N : N ∈ tangent_plane ∧ N ∈ Line A1 D1)

-- Theorems and proof
theorem angle_between_planes_AC1K_AC1N :
  angle (Plane.mk A C1 K) (Plane.mk A C1 N) = 120 :=
sorry

end angle_between_planes_AC1K_AC1N_l203_203035


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203611

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203611


namespace tan_105_eq_minus_2_minus_sqrt_3_l203_203601

theorem tan_105_eq_minus_2_minus_sqrt_3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_minus_2_minus_sqrt_3_l203_203601


namespace largest_value_of_x_l203_203686

theorem largest_value_of_x : 
  ∃ x, ( (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ) ∧ x = (19 + Real.sqrt 229) / 22 :=
sorry

end largest_value_of_x_l203_203686


namespace secant_proposal_l203_203424

-- Define the standard types and entities relevant to geometry and incircles
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

-- Definitions of the sides of the triangle and the secant
variables {a b c x p : ℝ}

-- Conditions:
-- 1. p is the semi-perimeter of triangle ABC
-- 2. A secant drawn through vertex C divides the triangle into two such that their inscribed radii are equal.

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Main Statement
theorem secant_proposal 
  (p : ℝ) (a b c : ℝ) (h : semi_perimeter a b c = p)
  (C1 : Type) (r1 r2 : ℝ) -- radii of incircles of the sub-triangles are given as r1 and r2
  (h_radii_eq : r1 = r2)
  (h_secant : triangle_secant_condition a b c x)
  : x = sqrt (p * (p - c)) := 
sorry

end secant_proposal_l203_203424


namespace two_digit_primes_with_ones_digit_three_count_l203_203763

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def number_of_two_digit_primes_with_ones_digit_three : ℕ :=
  6

theorem two_digit_primes_with_ones_digit_three_count :
  number_of_two_digit_primes_with_ones_digit_three =
  (finset.filter (λ n, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n)
                 (finset.range 100)).card :=
by
  sorry

end two_digit_primes_with_ones_digit_three_count_l203_203763


namespace tan_105_degree_l203_203646

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l203_203646


namespace range_of_m_l203_203754

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = m - 1)
  (h2 : x - 3 * y = 2 * m)
  (h3 : x + 2 * y ≥ 0) : 
  m ≤ -1 := 
sorry

end range_of_m_l203_203754


namespace area_triangle_QCA_l203_203099

noncomputable def area_of_triangle_QCA (p : ℝ) : ℝ :=
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let QA := 3
  let QC := 12 - p
  (1/2) * QA * QC

theorem area_triangle_QCA (p : ℝ) : area_of_triangle_QCA p = (3/2) * (12 - p) :=
  sorry

end area_triangle_QCA_l203_203099


namespace count_two_digit_primes_ending_in_3_l203_203820

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_ones_digit_3 (n : ℕ) : Prop := n % 10 = 3
def is_prime (n : ℕ) : Prop := nat.prime n
def two_digit_primes_ending_in_3 (n : ℕ) : Prop :=
  is_two_digit n ∧ has_ones_digit_3 n ∧ is_prime n

theorem count_two_digit_primes_ending_in_3 :
  (nat.card { n : ℕ | two_digit_primes_ending_in_3 n } = 6) :=
sorry

end count_two_digit_primes_ending_in_3_l203_203820


namespace find_c_l203_203343

def vec1 : ℝ × ℝ := (4, c)
def vec2 : ℝ × ℝ := (-3, 2)

def dotProduct (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def normSquared (v : ℝ × ℝ) : ℝ := v.1^2 + v.2^2

theorem find_c (c : ℝ) : (proj_vec2 vec1 = (10 / 13) * vec2) -> c = 11 :=
by
  -- Definitions related to the condition
  let proj_vec2 := (dotProduct vec1 vec2) / (normSquared vec2) * vec2
  sorry

end find_c_l203_203343


namespace number_of_two_digit_primes_with_ones_digit_3_l203_203962

-- Definition of two-digit numbers with a ones digit of 3
def two_digit_numbers_with_ones_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of prime predicate
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Proof statement
theorem number_of_two_digit_primes_with_ones_digit_3 : 
  let primes := (two_digit_numbers_with_ones_digit_3.filter is_prime) in
  primes.length = 7 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_3_l203_203962


namespace find_prime_pairs_l203_203121

def is_prime (n : ℕ) := n ≥ 2 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def has_prime_root (m n : ℕ) : Prop :=
  ∃ (p: ℕ), is_prime p ∧ (p * p - m * p - n = 0)

theorem find_prime_pairs :
  ∀ (m n : ℕ), (is_prime m ∧ is_prime n) → has_prime_root m n → (m, n) = (2, 3) :=
by sorry

end find_prime_pairs_l203_203121


namespace impossible_event_l203_203067

-- Definitions based on conditions
def event_A : Prop := (∃ (red_balls white_balls yellow_balls : ℕ), red_balls = 2 ∧ white_balls = 1 ∧ yellow_balls = 0) ∧ yellow_balls > 0
def event_B : Prop := true -- Since predicting weather can happen, we consider it as always possible
def event_C : Prop := true -- Tossing a fair dice to get 6 is a possible event
def event_D : Prop := true -- The last digit of the license plate can be even

-- The theorem to prove that event A is the impossible event
theorem impossible_event : event_A = false :=
by 
  sorry

end impossible_event_l203_203067


namespace number_of_two_digit_primes_with_ones_digit_three_l203_203895

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l203_203895


namespace intersection_nonempty_implies_nonzero_l203_203187

noncomputable def M (a : ℝ) : Set ℝ := {0, a}
noncomputable def N : Set ℤ := {x ∈ (Set.range (coe : ℤ → ℝ)) | x^2 - 2*x - 3 < 0}

theorem intersection_nonempty_implies_nonzero (a : ℝ) (h : (M a ∩ N.to_real) ≠ ∅) : a ≠ 0 := by
  sorry

end intersection_nonempty_implies_nonzero_l203_203187


namespace tan_105_degree_l203_203561

theorem tan_105_degree : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  have tan_add : ∀ (a b : ℝ), Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
    sorry
  have tan_45 := Real.tan (45 * Real.pi / 180)
  have tan_60 := Real.tan (60 * Real.pi / 180)
  have tan_45_value : tan_45 = 1 := sorry
  have tan_60_value : tan_60 = Real.sqrt 3 := sorry
  sorry

end tan_105_degree_l203_203561


namespace count_two_digit_primes_ending_with_3_l203_203837

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → n % m ≠ 0

def two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def ends_with_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem count_two_digit_primes_ending_with_3 :
  {n : ℕ | two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 := by
sorry

end count_two_digit_primes_ending_with_3_l203_203837
