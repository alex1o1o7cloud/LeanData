import Mathlib

namespace least_k_for_bound_l300_300488

noncomputable def u_seq : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 2.5 * u_seq n - 3 * (u_seq n)^2

def L : ℝ := 2 / 5

def satisfies_bound (k : ℕ) : Prop :=
|u_seq k - L| ≤ 1 / (2^500)

theorem least_k_for_bound : ∃ (k : ℕ), satisfies_bound k ∧ ∀ k' < k, ¬ satisfies_bound k' :=
begin
  use 5,
  split,
  {
    -- Proof for |u_seq 5 - L| ≤ 1 / (2^500)
    sorry
  },
  {
    -- Proof that ∀ k' < 5, ¬ satisfies_bound k'
    sorry
  }
end

end least_k_for_bound_l300_300488


namespace functional_equation_l300_300818

theorem functional_equation (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x^2 + f(y)) = y + (f(x))^2) : 
  ∀ x : ℝ, f(x) = x :=
by
  sorry

end functional_equation_l300_300818


namespace cos_arcsin_l300_300231

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l300_300231


namespace intersection_product_distance_eq_eight_l300_300781

noncomputable def parametricCircle : ℝ → ℝ × ℝ :=
  λ θ => (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def parametricLine : ℝ → ℝ × ℝ :=
  λ t => (2 + (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem intersection_product_distance_eq_eight :
  ∀ θ t,
    let (x1, y1) := parametricCircle θ
    let (x2, y2) := parametricLine t
    (x1^2 + y1^2 = 16) ∧ (x2 = x1 ∧ y2 = y1) →
    ∃ t1 t2,
      x1 = 2 + (1 / 2) * t1 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t1 ∧
      x1 = 2 + (1 / 2) * t2 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t2 ∧
      (t1 * t2 = -8) ∧ (|t1 * t2| = 8) := 
by
  intros θ t
  dsimp only
  intro h
  sorry

end intersection_product_distance_eq_eight_l300_300781


namespace cos_arcsin_l300_300232

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l300_300232


namespace coefficient_x3_eq_14_l300_300862

open BigOperators

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the required expression
def expr (x : ℝ) : ℝ := (1 - 1 / (x^2)) * (1 + x)^6

-- Theorem statement
theorem coefficient_x3_eq_14 : ∀ x : ℝ, (x ≠ 0) → (∃ c : ℝ, c = binom 6 3 - binom 6 5 ∧ c = 14 ∧ expr x = (c * x^3)) :=
by
  sorry

end coefficient_x3_eq_14_l300_300862


namespace simplify_fraction_l300_300922

theorem simplify_fraction : (3^9 / 9^3) = 27 :=
by
  sorry

end simplify_fraction_l300_300922


namespace technicians_count_l300_300858

-- Define the number of workers
def total_workers : ℕ := 21

-- Define the average salaries
def avg_salary_all : ℕ := 8000
def avg_salary_technicians : ℕ := 12000
def avg_salary_rest : ℕ := 6000

-- Define the number of technicians and rest of workers
variable (T R : ℕ)

-- Define the equations based on given conditions
def equation1 := T + R = total_workers
def equation2 := (T * avg_salary_technicians) + (R * avg_salary_rest) = total_workers * avg_salary_all

-- Prove the number of technicians
theorem technicians_count : T = 7 :=
by
  sorry

end technicians_count_l300_300858


namespace sum_of_reciprocals_of_squares_of_roots_l300_300281

noncomputable def reciprocal_squares_sum (p : Polynomial ℝ) : ℝ :=
  let roots := p.roots
  if h : roots.length = 4 then
    let r1, r2, r3, r4 := roots.nth_le 0 sorry, roots.nth_le 1 sorry, roots.nth_le 2 sorry, roots.nth_le 3 sorry
    (1 / (r1 ^ 2)) + (1 / (r2 ^ 2)) + (1 / (r3 ^ 2)) + (1 / (r4 ^ 2))
  else 0

theorem sum_of_reciprocals_of_squares_of_roots :
  let p : Polynomial ℝ := Polynomial.C (1 : ℝ) + Polynomial.X ^ 4 - 2 * Polynomial.C (1 : ℝ) * Polynomial.X ^ 3 + 
                          6 * Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 - 2 * Polynomial.C (1 : ℝ) * Polynomial.X + 1
  reciprocal_squares_sum p = -8 := 
sorry

end sum_of_reciprocals_of_squares_of_roots_l300_300281


namespace count_solutions_g_composition_eq_l300_300820

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the main theorem
theorem count_solutions_g_composition_eq :
  ∃ (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
by
  sorry

end count_solutions_g_composition_eq_l300_300820


namespace intersection_of_A_and_B_l300_300338

def I := {x : ℝ | true}
def A := {x : ℝ | x * (x - 1) ≥ 0}
def B := {x : ℝ | x > 1}
def C := {x : ℝ | x > 1}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l300_300338


namespace minimum_economic_loss_l300_300203

-- Definitions using conditions in a)
def repair_times : List Nat := [12, 17, 8, 18, 23, 30, 14]
def economic_loss_per_minute : Nat := 2
def num_workers : Nat := 3

-- Statement to prove
theorem minimum_economic_loss :  
  let total_repair_time := repair_times.sum,
      distribute_workload := 
        -- Here, you would define a function or logic to distribute the workload optimally (not implemented here)
        sorry,
      waiting_times := 
        -- Here, you would define a function or logic to calculate waiting times based on distribution (not implemented here)
        sorry,
      total_waiting_time := waiting_times.sum,
      total_economic_loss := total_waiting_time * economic_loss_per_minute
  in
  total_economic_loss = 364 := 
sorry

end minimum_economic_loss_l300_300203


namespace TournamentProbability_l300_300891

theorem TournamentProbability (n : ℕ) (hn : n = 30):
  let T := n * (n - 1) / 2,
      favorable_outcomes := factorial n,
      total_outcomes := 2^T,
      probability := favorable_outcomes / total_outcomes
  in 
  (nat.log2 (total_outcomes / gcd favorable_outcomes total_outcomes) = 409) :=
by
  sorry

end TournamentProbability_l300_300891


namespace max_value_of_z_l300_300037

theorem max_value_of_z (k : ℝ) (x y : ℝ)
  (h1 : x + 2 * y - 1 ≥ 0)
  (h2 : x - y ≥ 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ k)
  (h5 : ∀ x y, x + 2 * y - 1 ≥ 0 ∧ x - y ≥ 0 ∧ 0 ≤ x ∧ x ≤ k → x + k * y ≥ -2) :
  ∃ (x y : ℝ), x + k * y = 20 := 
by
  sorry

end max_value_of_z_l300_300037


namespace sum_equals_1000_500_334_l300_300347

theorem sum_equals_1000_500_334 :
  (∑ n in Finset.range (1000 + 1), n * (1001 - n)) = 1000 * 500 * 334 :=
by
  sorry

end sum_equals_1000_500_334_l300_300347


namespace quadratic_inequality_range_l300_300759

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) → a ∈ set.Ioc (-2 : ℝ) 2 := 
begin
  sorry
end

end quadratic_inequality_range_l300_300759


namespace sam_driving_distance_l300_300424

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300424


namespace sam_drove_200_miles_l300_300465

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300465


namespace car_return_point_l300_300190

theorem car_return_point (α : ℝ) (hα1 : 0 < α) (hα2 : α < 180) :
  (∀ n : ℕ, n = 5 → 
    let theta := n * α in ∃ k : ℤ, θ = k * 360) ↔ (α = 72 ∨ α = 144) := 
sorry

end car_return_point_l300_300190


namespace car_returns_to_start_after_5_operations_l300_300188

theorem car_returns_to_start_after_5_operations (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : α = 72 ∨ α = 144 :=
sorry

end car_returns_to_start_after_5_operations_l300_300188


namespace factorize_expression_l300_300671

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300671


namespace count_monomials_in_expansion_l300_300377

theorem count_monomials_in_expansion
  (x y z : ℝ) :
  let expr := (x + y + z) ^ 2030 + (x - y - z) ^ 2030 in
  number_of_monomials_with_nonzero_coeff expr = 1032256 :=
sorry

end count_monomials_in_expansion_l300_300377


namespace calculate_expression_l300_300756

theorem calculate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 3*(x^4 + 2*y^2)/9 = 37 + 2/3 :=
by
  intros x y hx hy
  sorry

end calculate_expression_l300_300756


namespace railway_ticket_count_l300_300111

theorem railway_ticket_count (n : ℕ) (h : n = 25) : (n * (n - 1)) / 2 = 300 :=
by {
  rw h,
  norm_num,
}

end railway_ticket_count_l300_300111


namespace subtract_eq_l300_300915

theorem subtract_eq (x y : ℝ) (h1 : 4 * x - 3 * y = 2) (h2 : 4 * x + y = 10) : 4 * y = 8 :=
by
  sorry

end subtract_eq_l300_300915


namespace sam_driving_distance_l300_300422

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300422


namespace absolute_prime_digits_bound_l300_300825

def is_prime (n : ℕ) : Prop := sorry -- placeholder for the prime number definition

def is_absolute_prime (n : ℕ) : Prop :=
is_prime n ∧ ∀ m : ℕ, m ∈ list.permutations n.digits → is_prime m

theorem absolute_prime_digits_bound (N : ℕ) :
  is_absolute_prime N → (N.digits.to_finset.card ≤ 3) :=
sorry

end absolute_prime_digits_bound_l300_300825


namespace certain_number_unique_l300_300119

-- Define the necessary conditions and statement
def is_certain_number (n : ℕ) : Prop :=
  (∃ k : ℕ, 25 * k = n) ∧ (∃ k : ℕ, 35 * k = n) ∧ 
  (n > 0) ∧ (∃ a b c : ℕ, 1 ≤ a * n ∧ a * n ≤ 1050 ∧ 1 ≤ b * n ∧ b * n ≤ 1050 ∧ 1 ≤ c * n ∧ c * n ≤ 1050 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem certain_number_unique :
  ∃ n : ℕ, is_certain_number n ∧ n = 350 :=
by 
  sorry

end certain_number_unique_l300_300119


namespace icosahedron_faces_l300_300683

theorem icosahedron_faces : 
  ∀ (I : Type) [polyhedron I], (faces I = 20) :=
sorry

end icosahedron_faces_l300_300683


namespace valid_numbers_count_l300_300748

-- Define a predicate that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that counts how many numbers between 100 and 999 are multiples of 13
def count_multiples_of_13 (start finish : ℕ) : ℕ :=
  (finish - start) / 13 + 1

-- Define a function that checks if a permutation of digits of n is a multiple of 13
-- (actual implementation would require digit manipulation, but we assume its existence here)
def is_permutation_of_digits_multiple_of_13 (n : ℕ) : Prop :=
  ∃ (perm : ℕ), is_three_digit perm ∧ perm % 13 = 0

noncomputable def count_valid_permutations (multiples_of_13 : ℕ) : ℕ :=
  multiples_of_13 * 3 -- Assuming on average

-- Problem statement: Prove that there are 207 valid numbers satisfying the condition
theorem valid_numbers_count : (count_valid_permutations (count_multiples_of_13 104 988)) = 207 := 
by {
  -- Place for proof which is omitted here
  sorry
}

end valid_numbers_count_l300_300748


namespace sugar_precipitate_l300_300640

theorem sugar_precipitate {water sugar precipitate : ℝ} :
    (41 / 100 * 220) + precipitate = sugar →
    water = 220 →
    sugar = 280 →
    precipitate = 127 →
    precipitate = 127 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₄]
  exact h₄

#check sugar_precipitate

end sugar_precipitate_l300_300640


namespace Al_atoms_in_compound_l300_300594

noncomputable def compound : Type := sorry

variables (nF : ℕ := 3)
variables (MW_total MW_Al MW_F : ℝ)
variables (num_Al_atoms : ℝ)

def atomic_weights : Prop :=
  MW_total = 84 ∧
  MW_Al = 26.98 ∧
  MW_F = 19.00

theorem Al_atoms_in_compound (h : atomic_weights) : num_Al_atoms = 1 := by
  sorry

end Al_atoms_in_compound_l300_300594


namespace car_return_point_l300_300189

theorem car_return_point (α : ℝ) (hα1 : 0 < α) (hα2 : α < 180) :
  (∀ n : ℕ, n = 5 → 
    let theta := n * α in ∃ k : ℤ, θ = k * 360) ↔ (α = 72 ∨ α = 144) := 
sorry

end car_return_point_l300_300189


namespace perfect_square_divisors_probability_l300_300606

theorem perfect_square_divisors_probability (m n : ℕ) (hrel_prime : Nat.coprime m n) :
  let N := 10.factorial
  let total_divisors := (Nat.divisors N).card
  let perfect_square_divisors := (Nat.divisors N).filter (λ d, Nat.is_square d).card
  (perfect_square_divisors / total_divisors) = (1 / 9)
  → m = 1
  → n = 9
  → m + n = 10 :=
by
  intros m n hrel_prime
  simp only [Nat.factorial, Nat.divisors, Nat.card]
  sorry

end perfect_square_divisors_probability_l300_300606


namespace coeff_x4_in_expansion_l300_300903

theorem coeff_x4_in_expansion (x : ℝ) :
  (coeff_x_n (expand (x + sqrt 5)^8 4)) = 1750 :=
by
  sorry

end coeff_x4_in_expansion_l300_300903


namespace comp_1_sub_i_pow4_l300_300650

theorem comp_1_sub_i_pow4 : (1 - complex.I)^4 = -4 := by
  sorry

end comp_1_sub_i_pow4_l300_300650


namespace avg_speed_additional_hours_l300_300949

/-- Definitions based on the problem conditions -/
def first_leg_speed : ℕ := 30 -- miles per hour
def first_leg_time : ℕ := 6 -- hours
def total_trip_time : ℕ := 8 -- hours
def total_avg_speed : ℕ := 34 -- miles per hour

/-- The theorem that ties everything together -/
theorem avg_speed_additional_hours : 
  ((total_avg_speed * total_trip_time) - (first_leg_speed * first_leg_time)) / (total_trip_time - first_leg_time) = 46 := 
sorry

end avg_speed_additional_hours_l300_300949


namespace triangle_medians_perpendicular_l300_300061

theorem triangle_medians_perpendicular
  (A B C G : Type*) [T : metric_space A] [T : metric_space B] [T : metric_space C]
  (AB AC BC : ℝ)
  (hAB : AB = 15)
  (hAC : AC = 20)
  (hMediansPerpendicular : is_perpendicular (A → B) (A → C))
  (hCentroid : ∀ (A B C G : Type*), is_centroid A B C G) :
  BC = 32 / 3 :=
sorry

end triangle_medians_perpendicular_l300_300061


namespace log_equation_solution_l300_300686

theorem log_equation_solution
  (x : ℝ) 
  (h_cond : log 3 (x - 1) + log (real.sqrt 3) (x ^ 2 - 1) + log (1 / 3) (x - 1) = 3) :
  x = real.sqrt (1 + 3 * real.sqrt 3) :=
sorry

end log_equation_solution_l300_300686


namespace katy_summer_reading_l300_300017

theorem katy_summer_reading :
  let b_June := 8 in
  let b_July := 2 * b_June in
  let b_August := b_July - 3 in
  b_June + b_July + b_August = 37 :=
by
  sorry

end katy_summer_reading_l300_300017


namespace sam_drove_200_miles_l300_300431

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300431


namespace factorize_expression_l300_300670

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300670


namespace melissa_work_hours_l300_300064

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l300_300064


namespace valid_seating_arrangements_l300_300776

def num_people : Nat := 10
def total_arrangements : Nat := Nat.factorial num_people
def restricted_group_arrangements : Nat := Nat.factorial 7 * Nat.factorial 4
def valid_arrangements : Nat := total_arrangements - restricted_group_arrangements

theorem valid_seating_arrangements : valid_arrangements = 3507840 := by
  sorry

end valid_seating_arrangements_l300_300776


namespace sequence_sum_l300_300381

theorem sequence_sum (n : ℕ) (h_pos : ∀ i ≤ n, a i > 0) (h_a1 : a 1 = 2)
  (h_an : ∀ i < n, a (i + 1) = 3 * a i) : 
  ∑ i in Finset.range n, a (i + 1) = 3^n - 1 := sorry

end sequence_sum_l300_300381


namespace necessary_but_not_sufficient_l300_300048

-- Define the sets A and B
def A (x : ℝ) : Prop := x > 2
def B (x : ℝ) : Prop := x > 1

-- Prove that B (necessary condition x > 1) does not suffice for A (x > 2)
theorem necessary_but_not_sufficient (x : ℝ) (h : B x) : A x ∨ ¬A x :=
by
  -- B x is a necessary condition for A x
  have h1 : x > 1 := h
  -- A x is not necessarily implied by B x
  sorry

end necessary_but_not_sufficient_l300_300048


namespace total_acorns_proof_l300_300081

variable (x y : ℝ)

def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := 5.3 * x + y
  let ella := 2 * (4.3 * x + y)
  shawna + sheila + danny + ella

theorem total_acorns_proof (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y :=
by
  unfold total_acorns
  sorry

end total_acorns_proof_l300_300081


namespace average_speed_l300_300205

def total_distance := 225 + 370
def total_time := 3.5 + 5

theorem average_speed : total_distance / total_time = 70 := by
  rw [total_distance, total_time]
  norm_num
  sorry

end average_speed_l300_300205


namespace Aiyanna_has_more_cookies_l300_300208

theorem Aiyanna_has_more_cookies (Alyssa_cookies : ℕ) (Aiyanna_cookies : ℕ) (hAlyssa : Alyssa_cookies = 129) (hAiyanna : Aiyanna_cookies = 140) : Aiyanna_cookies - Alyssa_cookies = 11 := 
by sorry

end Aiyanna_has_more_cookies_l300_300208


namespace meaningful_range_fraction_l300_300761

theorem meaningful_range_fraction (x : ℝ) : 
  ¬ (x = 3) ↔ (∃ y, y = x / (x - 3)) :=
sorry

end meaningful_range_fraction_l300_300761


namespace irreducible_poly_exists_l300_300702

theorem irreducible_poly_exists (n : ℕ) (m : fin n → ℤ) (h_diff : ∀ i j : fin n, i ≠ j → m i ≠ m j) :
  ∃ (f : polynomial ℤ), 
    (∀ i : fin n, f.eval (m i) = -1) ∧ irreducible f := 
sorry

end irreducible_poly_exists_l300_300702


namespace double_sum_evaluation_l300_300265

theorem double_sum_evaluation :
  (∑ m in (Finset.range m), ∑ n in (Finset.range n), (1 : ℝ) / (m * n * (m + n)^2)) = 1 := by
  sorry

end double_sum_evaluation_l300_300265


namespace frog_jump_distance_l300_300504

-- Define the distances jumped by the grasshopper and the frog, and the relationship between them
variable (g f : ℕ)

-- Given conditions
axiom grasshopper_jump : g = 17
axiom frog_jump_additional : f = g + 22

-- Statement to prove
theorem frog_jump_distance : f = 39 := by
  rw [grasshopper_jump, add_comm, add_assoc, add_comm] at frog_jump_additional
  exact frog_jump_additional

end frog_jump_distance_l300_300504


namespace b_minus_a_l300_300517

theorem b_minus_a (a b : ℕ) : (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) :=
by
  sorry

end b_minus_a_l300_300517


namespace delegates_not_wearing_badges_l300_300215

def totalDelegates : ℕ := 45
def prePrintedBadges : ℕ := 16
def takeBreakFraction : ℚ := 1 / 3
def handWrittenFraction : ℚ := 1 / 4

theorem delegates_not_wearing_badges :
  let takeBreak := totalDelegates * takeBreakFraction
  let remainingAfterBreak := totalDelegates - takeBreak.to_nat
  let handWrittenBadges := remainingAfterBreak * handWrittenFraction
  let delegatesWithoutBadges := remainingAfterBreak - handWrittenBadges.to_nat
  let prePrintedNotTakingBreak := prePrintedBadges - takeBreak.to_nat
  delegatesWithoutBadges + prePrintedNotTakingBreak = 24 :=
by {
  -- Proof goes here
  sorry
}

end delegates_not_wearing_badges_l300_300215


namespace solution_of_functional_equation_l300_300040

theorem solution_of_functional_equation
  (R : Type) [linear_ordered_field R] [has_zero R] [has_pow R R]
  (R_plus : set R) 
  (alpha beta : R)
  (f : R → R) (h : ∀ x ∈ R_plus, ∀ y ∈ R_plus, 
  f x * f y = y ^ alpha * f (x / 2) + x ^ beta * f (y / 2)) : 
  (alpha ≠ beta → ∀ x ∈ R_plus, f x = 0) ∧ (alpha = beta → (∀ x ∈ R_plus, f x = 0) ∨ (∃ C, ∀ x ∈ R_plus, f x = C * x ^ alpha)) :=
sorry

end solution_of_functional_equation_l300_300040


namespace bob_sheep_and_ratio_l300_300060

-- Define the initial conditions
def mary_initial_sheep : ℕ := 300
def additional_sheep_bob_has : ℕ := 35
def sheep_mary_buys : ℕ := 266
def fewer_sheep_than_bob : ℕ := 69

-- Define the number of sheep Bob has
def bob_sheep (mary_initial_sheep : ℕ) (additional_sheep_bob_has : ℕ) : ℕ := 
  mary_initial_sheep + additional_sheep_bob_has

-- Define the number of sheep Mary has after buying more sheep
def mary_new_sheep (mary_initial_sheep : ℕ) (sheep_mary_buys : ℕ) : ℕ := 
  mary_initial_sheep + sheep_mary_buys

-- Define the relation between Mary's and Bob's sheep (after Mary buys sheep)
def mary_bob_relation (mary_new_sheep : ℕ) (fewer_sheep_than_bob : ℕ) : Prop :=
  mary_new_sheep + fewer_sheep_than_bob = bob_sheep mary_initial_sheep additional_sheep_bob_has

-- Define the proof problem
theorem bob_sheep_and_ratio : 
  bob_sheep mary_initial_sheep additional_sheep_bob_has = 635 ∧ 
  (bob_sheep mary_initial_sheep additional_sheep_bob_has) * 300 = 635 * mary_initial_sheep := 
by 
  sorry

end bob_sheep_and_ratio_l300_300060


namespace cos_arcsin_l300_300236

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l300_300236


namespace angle_bisector_MN_l300_300044

variable (A B C D P M N Q : Point)
variable (hABCD : Rectangle A B C D)
variable (hP_on_CD : Collinear P C D)
variable (hM_mid_AD : Midpoint M A D)
variable (hN_mid_BC : Midpoint N B C)
variable (hQ_intersect_AC : Line.through P M ∧ Line.through P M Q ∧ Line.through A C Q)

theorem angle_bisector_MN (A B C D P M N Q : Point) 
  (hABCD : Rectangle A B C D)
  (hP_on_CD : Collinear P C D)
  (hM_mid_AD : Midpoint M A D)
  (hN_mid_BC : Midpoint N B C)
  (hQ_intersect_AC : Line.through P M ∧ Line.through P M Q ∧ Line.through A C Q) :
  isAngleBisector MN (angle Q N P) := sorry

end angle_bisector_MN_l300_300044


namespace positive_expressions_l300_300507

-- Define the approximate values for A, B, C, D, and E.
def A := 2.5
def B := -2.1
def C := -0.3
def D := 1.0
def E := -0.7

-- Define the expressions that we need to prove as positive numbers.
def exprA := A + B
def exprB := B * C
def exprD := E / (A * B)

-- The theorem states that expressions (A + B), (B * C), and (E / (A * B)) are positive.
theorem positive_expressions : exprA > 0 ∧ exprB > 0 ∧ exprD > 0 := 
by sorry

end positive_expressions_l300_300507


namespace cos_arcsin_l300_300243

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l300_300243


namespace area_of_trapezoid_l300_300004

variables (AB CD DK BM : ℝ)
variables (K M : ℝ)
variables (D_angle B_angle : ℝ)
variables (perimeter : ℝ)

def is_midpoint (K : ℝ) (AB : ℝ) : Prop := K = AB / 2
def is_midpoint_cd (M : ℝ) (CD : ℝ) : Prop := M = CD / 2
def is_angle_bisector_dk (DK : ℝ) (D_angle : ℝ) : Prop := DK = D_angle / 2
def is_angle_bisector_bm (BM : ℝ) (B_angle : ℝ) : Prop := BM = B_angle / 2
def largest_angle_lower_base := 60
def trapezoid_perimeter := 30

theorem area_of_trapezoid
  (hK : is_midpoint K AB)
  (hM : is_midpoint_cd M CD)
  (hDK : is_angle_bisector_dk DK D_angle)
  (hBM : is_angle_bisector_bm BM B_angle)
  (hLargestAngle : largest_angle_lower_base = 60)
  (hPerimeter : trapezoid_perimeter = 30) :
  area_of_trapezoid = 15 * Real.sqrt 3 := by
  sorry

end area_of_trapezoid_l300_300004


namespace problem_l300_300645

theorem problem : (112^2 - 97^2) / 15 = 209 := by
  sorry

end problem_l300_300645


namespace majority_owner_percentage_l300_300873

theorem majority_owner_percentage (profit total_profit : ℝ)
    (majority_owner_share : ℝ) (partner_share : ℝ) 
    (combined_share : ℝ) 
    (num_partners : ℕ) 
    (total_profit_value : total_profit = 80000) 
    (partner_share_value : partner_share = 0.25 * (1 - majority_owner_share)) 
    (combined_share_value : combined_share = profit)
    (combined_share_amount : combined_share = 50000) 
    (num_partners_value : num_partners = 4) :
  majority_owner_share = 0.25 :=
by
  sorry

end majority_owner_percentage_l300_300873


namespace percentage_difference_l300_300897

theorem percentage_difference (water_yesterday : ℕ) (water_two_days_ago : ℕ) (h1 : water_yesterday = 48) (h2 : water_two_days_ago = 50) : 
  (water_two_days_ago - water_yesterday) / water_two_days_ago * 100 = 4 :=
by
  sorry

end percentage_difference_l300_300897


namespace range_of_a_l300_300723

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → a < x + (1 / x)) → a < 2 :=
by
  sorry

end range_of_a_l300_300723


namespace find_a_minus_b_l300_300091

theorem find_a_minus_b (a b : ℚ) (h_eq : ∀ x : ℚ, (a * (-5 * x + 3) + b) = x - 9) : 
  a - b = 41 / 5 := 
by {
  sorry
}

end find_a_minus_b_l300_300091


namespace sqrt_one_half_eq_sqrt_two_over_two_l300_300936

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two_l300_300936


namespace concyclic_points_l300_300636

theorem concyclic_points
    (O : Type) [MetricSpace O] [EuclideanSpace O]
    (A B C D E F M : O)
    (hCircle : circle O A B C D)
    (hAB_diameter : diameter O A B)
    (hCSide : same_side O A B C D)
    (hTangentC : tangent O A C E)
    (hTangentD : tangent O A D E)
    (hIntersectBC_AD_F : meet O B C A D F)
    (hIntersectBF_AB_M : meet O B F A B M) :
    concyclic_points O E C M D := 
sorry

end concyclic_points_l300_300636


namespace compare_length_l300_300779

universe u

-- Declare the points and their properties
variables {Point : Type u} [metric_space Point] (A B C D M K : Point)
variables (AB BD CD_M_ratio AD_MK_ratio : ℝ)

-- Define conditions
def is_rectangle (A B C D : Point) : Prop := 
  dist A B = 2 ∧ 
  dist (B D) = real.sqrt 7 ∧ 
  (dist C D) ≠ 0 ∧
  (CD_M_ratio = 1 / 3) ∧ 
  (AD_MK_ratio = 1 / 2)

-- Define the problem as proving the length comparison AM > BK
theorem compare_length {A B C D M K : Point} (h : is_rectangle A B C D) :
  dist A M > dist B K := sorry

end compare_length_l300_300779


namespace books_sold_on_wednesday_l300_300797

theorem books_sold_on_wednesday :
  ∀ (total_books : ℕ) (sold_monday : ℕ) (sold_tuesday : ℕ) (sold_thursday : ℕ) (sold_friday : ℕ) (percentage_unsold : ℝ),
    total_books = 1400 →
    sold_monday = 75 →
    sold_tuesday = 50 →
    sold_thursday = 78 →
    sold_friday = 135 →
    percentage_unsold = 71.28571428571429 →
    let unsold_books := total_books * (percentage_unsold / 100) in
    let sold_books := total_books - unsold_books in
    let sold_wednesday := sold_books - (sold_monday + sold_tuesday + sold_thursday + sold_friday) in
    sold_wednesday = 64 :=
by
  intros total_books sold_monday sold_tuesday sold_thursday sold_friday percentage_unsold
  intros ht hm htue htth htfr hpu
  let unsold_books := total_books * (percentage_unsold / 100)
  let sold_books := total_books - unsold_books
  let sold_wednesday := sold_books - (sold_monday + sold_tuesday + sold_thursday + sold_friday)
  sorry

end books_sold_on_wednesday_l300_300797


namespace part1_part2_l300_300939

-- Conditions
variables {a b : ℝ}
variable h1 : a > 0
variable h2 : b > 0
variable h3 : a > b

-- Definitions and functions involved
def x := a / b

-- Proof statements
theorem part1 (ha : a > 0) (hb : b > 0) (hab : a > b) : 
  (a + b) / 2 > (a - b) / (Real.log a - Real.log b) ∧ (a - b) / (Real.log a - Real.log b) > Real.sqrt (a * b) :=
sorry

noncomputable def g (x : ℝ) := Real.log x / x

theorem part2 (ha : a > 0) (hb : b > 0) (g_eq : g a = g b) : a + b > 2 * Real.exp 1 :=
sorry

end part1_part2_l300_300939


namespace triangle_area_inequality_l300_300768

-- Definitions of variables and constants
variable {a b c : ℝ} (λ μ ν : ℝ)

-- Definition of area of triangle
def area_of_triangle (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_inequality (a b c : ℝ) (λ μ ν : ℝ) :
  let Δ := area_of_triangle a b c
  in Δ ≤ (λ * a^2 + μ * b^2 + ν * c^2) * (μ * λ + λ * ν + ν * μ) / (12 * sqrt 3 * μ * ν * λ) :=
sorry

end triangle_area_inequality_l300_300768


namespace possible_rankings_l300_300490

theorem possible_rankings (A B C D E : Type)
  (competes : List (A × B × C × D × E))
  (h1 : ∀ r ∈ competes, r.1 ≠ 1 ∧ r.2 ≠ 1)
  (h2 : ∀ r ∈ competes, r.2 ≠ 5) :
  List.length competes = 54 := 
sorry

end possible_rankings_l300_300490


namespace SamDrove200Miles_l300_300441

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300441


namespace eval_dagger_l300_300268

noncomputable def dagger (m n p q : ℕ) : ℚ := 
  (m * p) * (q / n)

theorem eval_dagger : dagger 5 16 12 5 = 75 / 4 := 
by 
  sorry

end eval_dagger_l300_300268


namespace hyperbola_eccentricity_proof_l300_300303

def hyperbola_eccentricity (a b : ℝ) : Real :=
  sqrt (1 + (b^2) / (a^2))

theorem hyperbola_eccentricity_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ A B : ℝ × ℝ, (A ≠ (0,0) ∧ B ≠ (0,0)) ∧ 
    (A.2 = 4 * a / b ∧ A.1 = 4 * (a^2) / (b^2)) ∧ 
    (B.2 = 4 * a / b ∧ B.1 = 4 * (a^2) / (b^2)) ∧
    (∃ F : ℝ × ℝ, (F = (1,0)) ∧ ∠ A F B = 2 * π / 3))
  : hyperbola_eccentricity a b = sqrt 13 ∨ hyperbola_eccentricity a b = sqrt(21) / 3 :=
sorry

end hyperbola_eccentricity_proof_l300_300303


namespace copies_made_in_half_hour_l300_300595

theorem copies_made_in_half_hour
  (rate1 rate2 : ℕ)  -- rates of the two copy machines
  (time : ℕ)         -- time considered
  (h_rate1 : rate1 = 40)  -- the first machine's rate
  (h_rate2 : rate2 = 55)  -- the second machine's rate
  (h_time : time = 30)    -- time in minutes
  : (rate1 * time + rate2 * time = 2850) := 
sorry

end copies_made_in_half_hour_l300_300595


namespace min_value_expression_l300_300029

theorem min_value_expression (α β : ℝ) :
  ∃ x y, x = 3 * Real.cos α + 6 * Real.sin β ∧
         y = 3 * Real.sin α + 6 * Real.cos β ∧
         (x - 10)^2 + (y - 18)^2 = 121 :=
by
  sorry

end min_value_expression_l300_300029


namespace S_equals_2_l300_300800

noncomputable def problem_S := 
  1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
  1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)

theorem S_equals_2 : problem_S = 2 := by
  sorry

end S_equals_2_l300_300800


namespace find_constant_k_l300_300679

theorem find_constant_k (k : ℝ) :
  (-x^2 - (k + 9) * x - 8 = - (x - 2) * (x - 4)) → k = -15 :=
by 
  sorry

end find_constant_k_l300_300679


namespace roots_cubic_inv_sum_l300_300754

theorem roots_cubic_inv_sum (a b c r s : ℝ) (h_eq : ∃ (r s : ℝ), r^2 * a + b * r - c = 0 ∧ s^2 * a + b * s - c = 0) :
  (1 / r^3) + (1 / s^3) = (b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end roots_cubic_inv_sum_l300_300754


namespace frog_eyes_count_l300_300264

theorem frog_eyes_count (frogs_in_pond : ℕ) (eyes_per_frog : ℕ) (h1 : frogs_in_pond = 4) (h2 : eyes_per_frog = 2) : frogs_in_pond * eyes_per_frog = 8 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end frog_eyes_count_l300_300264


namespace actual_time_before_storm_l300_300103

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end actual_time_before_storm_l300_300103


namespace factory_produces_more_toys_l300_300622

theorem factory_produces_more_toys 
  (total_toys : ℕ) (planned_days : ℕ) (days_ahead : ℕ) 
  (h_toys : total_toys = 10080) 
  (h_planned_days : planned_days = 14)
  (h_days_ahead : days_ahead = 2) :
  let planned_production_per_day := total_toys / planned_days,
      actual_days := planned_days - days_ahead,
      actual_production_per_day := total_toys / actual_days in
  actual_production_per_day - planned_production_per_day = 120 :=
by
  let planned_production_per_day := 10080 / 14,
  actual_days := 14 - 2,
  actual_production_per_day := 10080 / 12 in
  sorry

end factory_produces_more_toys_l300_300622


namespace geometric_series_y_equals_9_l300_300651

theorem geometric_series_y_equals_9 :
  (∑' n : ℕ, ((1 / 3)^n)) * (∑' n : ℕ, (-1 / 3)^n) = (∑' n : ℕ, (1 / (9^n))) →
  y = 9 :=
begin
  intro h,
  have h1 : (∑' n : ℕ, ((1 / 3)^n)) = 3 / 2,
  { sorry },
  have h2 : (∑' n : ℕ, (-1 / 3)^n) = 3 / 4,
  { sorry },
  have h3 : (3 / 2) * (3 / 4) = 9 / 8,
  { sorry },
  have h4 : (∑' n : ℕ, (1 / (y^n))) = 9 / 8,
  { sorry },
  have h5 : 1 - 1 / y = 8 / 9,
  { sorry },
  have h6 : 1 / y = 1 / 9,
  { sorry },
  exact sorry
end

end geometric_series_y_equals_9_l300_300651


namespace min_value_expression_l300_300874

theorem min_value_expression (a b : ℝ) : 
  4 + (a + b)^2 ≥ 4 ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
sorry

end min_value_expression_l300_300874


namespace valid_exponent_rule_l300_300923

theorem valid_exponent_rule (a : ℝ) : (a^3)^2 = a^6 :=
by
  sorry

end valid_exponent_rule_l300_300923


namespace problem_statement_l300_300344

theorem problem_statement :
  (∑ n in Finset.range 1000, (n + 1) * (1001 - (n + 1))) = 1000 * 500 * (667 / 1000) :=
by
  sorry

end problem_statement_l300_300344


namespace hyperbola_eccentricity_a_l300_300736

theorem hyperbola_eccentricity_a (a : ℝ) (ha : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) ∧ (∃ (e : ℝ), e = 2 ∧ e = Real.sqrt (a^2 + 3) / a) → a = 1 :=
by
  sorry

end hyperbola_eccentricity_a_l300_300736


namespace find_RS_length_l300_300403

noncomputable def RS_length (P Q R S : ℝ) (hPQR_right : ∠ PQR = 90) (QR_diameter : diameter QR S) (hPS : S - P = 3) (hQS : S - Q = 5) : ℝ :=
5

theorem find_RS_length (P Q R S : ℝ) (hPQR_right : ∠ PQR = 90) (QR_diameter : diameter QR S) (hPS : S - P = 3) (hQS : S - Q = 5) :
  RS_length P Q R S hPQR_right QR_diameter hPS hQS = 5 := 
sorry 

end find_RS_length_l300_300403


namespace clock_hands_coincide_21st_time_after_1374_55_minutes_l300_300861

theorem clock_hands_coincide_21st_time_after_1374_55_minutes :
  let relative_speed := 11 / 12        -- relative speed of minute hand to hour hand in circles/hour
  let time_one_coincidence := 12 / 11  -- time for one full coincidence in hours
  let time_one_coincidence_minutes := time_one_coincidence * 60  -- time for one coincidence in minutes
  let time_21_coincidences := 21 * time_one_coincidence_minutes  -- time for 21 coincidences in minutes
  (Real.round (time_21_coincidences * 100) / 100) = 1374.55 :=
by
  sorry

end clock_hands_coincide_21st_time_after_1374_55_minutes_l300_300861


namespace k_domain_all_reals_l300_300273

noncomputable def domain_condition (k : ℝ) : Prop :=
  9 + 28 * k < 0

noncomputable def k_values : Set ℝ :=
  {k : ℝ | domain_condition k}

theorem k_domain_all_reals :
  k_values = {k : ℝ | k < -9 / 28} :=
by
  sorry

end k_domain_all_reals_l300_300273


namespace field_properties_l300_300555

open Set Classical

-- Definitions for field conditions
def is_field (F : Set ℝ) : Prop :=
  (∀ a b ∈ F, (a + b) ∈ F ∧ (a - b) ∈ F ∧ (a * b) ∈ F) ∧
  (∀ a b ∈ F, b ≠ 0 → (a / b) ∈ F)

-- Propositions to prove
def prop_1 : Prop := ∀ (F : Set ℝ), is_field F → 0 ∈ F
def prop_4 : Prop := is_field {x : ℚ | True}

-- Main theorem combining the propositions
theorem field_properties : prop_1 ∧ prop_4 := by
  sorry

end field_properties_l300_300555


namespace four_circles_max_parts_l300_300008

theorem four_circles_max_parts (n : ℕ) (h1 : ∀ n, n = 1 ∨ n = 2 ∨ n = 3 → ∃ k, k = 2^n) :
    n = 4 → ∃ k, k = 14 :=
by
  sorry

end four_circles_max_parts_l300_300008


namespace phoenix_number_5841_phoenix_numbers_satisfying_conditions_l300_300691

-- Definition of Phoenix Number
def is_phoenix_number (N : ℕ) : Prop := 
  let d1 := N / 1000,
      d2 := (N / 100) % 10,
      d3 := (N / 10) % 10,
      d4 := N % 10
  in d1 + d3 = 9 ∧ d2 + d4 = 9

-- Part 1: Phoenix Number check and K(N) calculation
theorem phoenix_number_5841 :
  is_phoenix_number 5841 ∧ 5841 / 99 = 59 :=
sorry

-- Part 2: Conditions for solving Phoenix Number N
def K (N : ℕ) : ℕ := N / 99

theorem phoenix_numbers_satisfying_conditions :
  ∀ (N : ℕ),
  is_phoenix_number N /\
  (N % 2 = 0) /\
  (let N' := ((N / 1000) + 9) * 1000 + ((N % 1000 / 100) + 9) * 100 + (N % 1000 / 10 % 10) * 10 + (N % 10) 
      in 3 * K N + 2 * K N' % 9 = 0) /\
  ((N / 1000) >= (N / 100 % 10))
  → (N = 8514 ∨ N = 3168) :=
sorry

end phoenix_number_5841_phoenix_numbers_satisfying_conditions_l300_300691


namespace three_digit_number_is_495_l300_300621

theorem three_digit_number_is_495 :
  ∃ (A : ℕ), (100 ≤ A ∧ A ≤ 999) ∧
             (∃ a b c : ℕ, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a > b ∧ b > c ∧ A = 100 * a + 10 * b + c) ∧
             (A = 100 * (A / 100) + 10 * ((A / 10) % 10) + (A % 10)) ∧
             ((100 * (A / 100) + 10 * ((A / 10) % 10) + (A % 10)) - (100 * (A % 10) + 10 * ((A / 10) % 10) + (A / 100))) = A ∧
             (∃ k : ℕ, A = 99 * k) ∧
             (495 = (100 * (495 / 100) + 10 * ((495 / 10) % 10) + (495 % 10)) - (100 * (495 % 10) + 10 * ((495 / 10) % 10) + (495 / 100))) :=
by
  use 495
  split
    sorry
    split
      sorry
      split
        sorry
        split
          sorry
          exists 5
          sorry

end three_digit_number_is_495_l300_300621


namespace melissa_work_hours_l300_300062

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l300_300062


namespace varying_interest_rates_l300_300135

theorem varying_interest_rates (P1 P2 : ℝ) (r1 r2 r3 r4 r5 : ℝ) :
  P1 * 5 * 8 / 100 = 840 ∧ P1 / P2 = 2 / 3 ∧ P2 * (r1 + r2 + r3 + r4 + r5) / 100 = 840 →
  r1 + r2 + r3 + r4 + r5 = 26.67 :=
begin
  sorry
end

end varying_interest_rates_l300_300135


namespace sam_drove_200_miles_l300_300430

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300430


namespace limit_of_geometric_series_l300_300644

open Filter

theorem limit_of_geometric_series :
  tendsto (λ n, (∑ k in Finset.range (n + 1), (1 / 3) ^ k) / 
                 (∑ k in Finset.range (n + 1), (1 / 2) ^ k)) atTop (𝓝 2) :=
begin
  sorry
end

end limit_of_geometric_series_l300_300644


namespace correct_calculation_l300_300919

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end correct_calculation_l300_300919


namespace Trisha_works_hours_per_week_l300_300532

def calc_hours_per_week (annual_take_home:ℝ) (hourly_rate:ℝ) (weeks_per_year:ℕ) (tax_rate:ℝ) : ℝ :=
  let gross_annual_pay := annual_take_home / (1 - tax_rate)
  let weekly_gross_pay := gross_annual_pay / weeks_per_year
  weekly_gross_pay / hourly_rate

theorem Trisha_works_hours_per_week : 
  let annual_take_home := 24_960
  let hourly_rate := 15
  let weeks_per_year := 52
  let tax_rate := 0.20
  calc_hours_per_week annual_take_home hourly_rate weeks_per_year tax_rate = 40 := 
by
  sorry

end Trisha_works_hours_per_week_l300_300532


namespace expression_value_l300_300938

noncomputable def expression : ℝ :=
  (π - 1)^0 - real.sqrt 9 + 2 * real.cos (real.pi / 4) + (1 / 5)⁻¹

theorem expression_value : expression = 3 + real.sqrt 2 := by
  sorry

end expression_value_l300_300938


namespace real_real_roots_det_eq_zero_l300_300031

theorem real_real_roots_det_eq_zero (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∀ x, 
    Det (Matrix.of ![![x, -c, b], ![c, x, -d], ![-b, d, x]]) = 0 ↔ x = 0 :=
by
  sorry

end real_real_roots_det_eq_zero_l300_300031


namespace hyperbola_equation_correct_l300_300705

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
                                     (eccentricity : Real := (Real.sqrt 6) / 2) 
                                     (distance_focus_asymptote : ℝ := 1) : Prop :=
  (eccentricity = (Real.sqrt 6) / 2) →
  (distance_focus_asymptote = 1) →
  (a / b = Real.sqrt 2) →
  (Eq ((x : ℝ) ^ 2 / a ^ 2 - (y : ℝ) ^ 2 / b ^ 2) 1 → 
  (Eq a (Real.sqrt 2)) → 
  (Eq b 1)) 

theorem hyperbola_equation_correct (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
                                   (eccentricity : Real := (Real.sqrt 6) / 2) 
                                   (distance_focus_asymptote : ℝ := 1) : 
  hyperbola_equation a b a_pos b_pos eccentricity distance_focus_asymptote :=
by
  sorry

end hyperbola_equation_correct_l300_300705


namespace solve_xy_eq_x_plus_y_l300_300848

theorem solve_xy_eq_x_plus_y (x y : ℤ) (h : x * y = x + y) : (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by {
  sorry
}

end solve_xy_eq_x_plus_y_l300_300848


namespace probability_of_yellow_marble_l300_300641

def marbles_prob :=
  let PxW := 4 / 9                -- Probability of drawing a white marble from Bag X
  let PyY := 7 / 10               -- Probability of drawing a yellow marble from Bag Y
  let PxB := 5 / 9                -- Probability of drawing a black marble from Bag X
  let PzY := 1 / 3                -- Probability of drawing a yellow marble from Bag Z
  PxW * PyY + PxB * PzY           -- Total probability

theorem probability_of_yellow_marble :
  marbles_prob = 67 / 135 :=
by
  sorry

end probability_of_yellow_marble_l300_300641


namespace tom_balloons_count_l300_300896

-- Define the number of balloons Tom initially has
def balloons_initial : Nat := 30

-- Define the number of balloons Tom gave away
def balloons_given : Nat := 16

-- Define the number of balloons Tom now has
def balloons_remaining : Nat := balloons_initial - balloons_given

theorem tom_balloons_count :
  balloons_remaining = 14 := by
  sorry

end tom_balloons_count_l300_300896


namespace triangle_angle_inequality_l300_300382

open BigOperators Classical

variables (A B C E F O : Type)
variables [triangle : ∀ (A B C : Type), Prop] 
variables [ge : ∀ (x y : A), Prop]

def AB (A B : Type) : A := sorry
def AC (A C : Type) : A := sorry
def BE (B E : Type) : A := sorry
def CF (C F : Type) : A := sorry
def median (O : Type) (B E C F : Type) : A := sorry
def angle (x y z : Type) : A := sorry
def gt (x y : A) : Prop := sorry

theorem triangle_angle_inequality 
    (ABC : triangle A B C)
    (H1: gt (AB A B) (AC A C) )
    (H2: median O B E C F):
    gt (angle O B C) (angle O C B) := 
sorry

end triangle_angle_inequality_l300_300382


namespace circle_tangent_chords_l300_300898

theorem circle_tangent_chords 
  (O₁ O₂ : Type*) [metric_space O₁] [metric_space O₂]
  (A B C D : O₁) 
  (circle₁ : metric.ball O₁ A) 
  (circle₂ : metric.ball O₂ A)
  (tangent1 : is_tangent B A circle₁)
  (tangent2 : is_tangent C A circle₂)
  (common_point : A = (circle₁ ∩ circle₂).some)
  : (|B - A|^2 / |C - A|^2 = |B - D| / |C - D|) :=
by
  -- Proof omitted
  sorry

end circle_tangent_chords_l300_300898


namespace total_area_of_five_equilateral_triangles_l300_300213

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * side_length^2

def effective_area_covered (n : ℕ) (side_length : ℝ) : ℝ :=
  let single_triangle_area := equilateral_triangle_area side_length
  -- Subtract the overlapping areas:
  -- Each subsequent triangle overlaps with half of the previous triangle's base
  single_triangle_area * (n - 1)

theorem total_area_of_five_equilateral_triangles : effective_area_covered 5 (2 * Real.sqrt 3) = 12 * Real.sqrt 3 :=
by sorry

end total_area_of_five_equilateral_triangles_l300_300213


namespace expected_value_of_winnings_is_3_point_5_l300_300630

noncomputable def expected_value_of_winnings : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probabilities := 1 / (outcomes.length : ℝ)
  let winnings := outcomes.map (λ n, 8 - n)
  let expected_value := probabilities * (winnings.sum)
  expected_value

theorem expected_value_of_winnings_is_3_point_5 : expected_value_of_winnings = 3.5 := by
  sorry

end expected_value_of_winnings_is_3_point_5_l300_300630


namespace average_production_last_5_days_l300_300771

theorem average_production_last_5_days 
  (daily_production_first_25_days : ℕ → ℕ)
  (daily_production_day_26_to_30 : ℕ → ℕ)
  (average_production_first_25_days : ℕ := 50)
  (average_production_month : ℕ := 45)
  (days_first_period : ℕ := 25)
  (days_second_period : ℕ := 5)
  (days_in_month : ℕ := 30) :
  (∑ i in finset.range days_first_period, daily_production_first_25_days i) / days_first_period = average_production_first_25_days →
  (∑ i in finset.range days_in_month, if i < days_first_period then daily_production_first_25_days i else daily_production_day_26_to_30 (i - days_first_period)) / days_in_month = average_production_month →
  (∑ i in finset.range days_second_period, daily_production_day_26_to_30 i) / days_second_period = 20 :=
by
  intros h1 h2
  sorry

end average_production_last_5_days_l300_300771


namespace sum_of_int_values_l300_300519
open Nat

theorem sum_of_int_values (a : ℤ) (a_range : -11 ≤ a ∧ a < -5)
  (h1 : ∀ x : ℤ, a - 2 = (3 * x - (x + 3))) -- Derived from the equation solution (Ensuring x is integer)
  (h2 : ∀ y : ℤ, (1 / 3) * y + 1 ≥ (y + 3) / 2 ∧ (a + y) / 2 < y - 1 → y ≤ -3 ∧ y > a + 2) -- Derived from the inequality system
  : (Set.univ.filter (λ a, -11 ≤ a ∧ a < -5 ∧ is_some (int_to_x a))).sum = -20 :=
sorry

-- auxiliary definition for converting integer a to corresponding x
def int_to_x (a : ℤ) : Option ℤ :=
  let x := (a + 1) / 2
  if (a + 1) % 2 = 0 then some x else none


end sum_of_int_values_l300_300519


namespace wine_cost_increase_l300_300618

noncomputable def additional_cost (initial_price : ℝ) (num_bottles : ℕ) (month1_rate : ℝ) (month2_tariff : ℝ) (month2_discount : ℝ) (month3_tariff : ℝ) (month3_rate : ℝ) : ℝ := 
  let price_month1 := initial_price * (1 + month1_rate) 
  let cost_month1 := num_bottles * price_month1
  let price_month2 := (initial_price * (1 + month2_tariff)) * (1 - month2_discount)
  let cost_month2 := num_bottles * price_month2
  let price_month3 := (initial_price * (1 + month3_tariff)) * (1 - month3_rate)
  let cost_month3 := num_bottles * price_month3
  (cost_month1 + cost_month2 + cost_month3) - (3 * num_bottles * initial_price)

theorem wine_cost_increase : 
  additional_cost 20 5 0.05 0.25 0.15 0.35 0.03 = 42.20 :=
by sorry

end wine_cost_increase_l300_300618


namespace square_area_l300_300069

theorem square_area (x : ℝ) (h1 : x = 60) : x^2 = 1200 :=
by
  sorry

end square_area_l300_300069


namespace max_prob_games_4_choose_best_of_five_l300_300496

-- Definitions of probabilities for Team A and Team B in different game scenarios
def prob_win_deciding_game : ℝ := 0.5
def prob_A_non_deciding : ℝ := 0.6
def prob_B_non_deciding : ℝ := 0.4

-- Definitions of probabilities for different number of games in the series
def prob_xi_3 : ℝ := (prob_A_non_deciding)^3 + (prob_B_non_deciding)^3
def prob_xi_4 : ℝ := 3 * (prob_A_non_deciding^2 * prob_B_non_deciding * prob_A_non_deciding + prob_B_non_deciding^2 * prob_A_non_deciding * prob_B_non_deciding)
def prob_xi_5 : ℝ := 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2) * (2 * prob_win_deciding_game)

-- The statement that a series of 4 games has the highest probability
theorem max_prob_games_4 : prob_xi_4 > prob_xi_5 ∧ prob_xi_4 > prob_xi_3 :=
by {
  sorry
}

-- Definitions of winning probabilities in the series for Team A
def prob_A_win_best_of_3 : ℝ := (prob_A_non_deciding)^2 + 2 * (prob_A_non_deciding * prob_B_non_deciding * prob_win_deciding_game)
def prob_A_win_best_of_5 : ℝ := (prob_A_non_deciding)^3 + 3 * (prob_A_non_deciding^2 * prob_B_non_deciding) + 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2 * prob_win_deciding_game)

-- The statement that Team A has a higher chance of winning in a best-of-five series
theorem choose_best_of_five : prob_A_win_best_of_5 > prob_A_win_best_of_3 :=
by {
  sorry
}

end max_prob_games_4_choose_best_of_five_l300_300496


namespace length_of_train_is_correct_l300_300927

def speed_kmh := 60 -- km/hr
def time_sec := 9 -- seconds
def conversion_factor := 5 / 18 -- to convert km/hr to m/s

def speed_ms := speed_kmh * conversion_factor -- speed in m/s
def length_train := speed_ms * time_sec -- length of the train in meters

theorem length_of_train_is_correct : length_train = 150.03 := by
  sorry

end length_of_train_is_correct_l300_300927


namespace intersection_of_M_and_N_l300_300028

open Set -- to directly use set notation and operations

theorem intersection_of_M_and_N : 
  let U := ℝ
  let M := {-1, 1, 2}
  let N := {x : ℝ | -1 < x ∧ x < 2}
  M ∩ N = {1} :=
by
  let U := ℝ
  let M := {-1, 1, 2}
  let N := {x : ℝ | -1 < x ∧ x < 2}
  sorry

end intersection_of_M_and_N_l300_300028


namespace plane_split_into_regions_l300_300258

theorem plane_split_into_regions :
  let L1 := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, 3*x)},
      L2 := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, x/3)} in
  (L1 ≠ L2) ∧ ∃ r : Finset (Finset (ℝ × ℝ)), 
    (∀ P : ℝ × ℝ, ∃ s ∈ r, P ∈ s) ∧ (∀ s ∈ r, ∀ t ∈ r, s ≠ t → s ∩ t = ∅) ∧ (r.card = 4).

end plane_split_into_regions_l300_300258


namespace sufficient_condition_for_odd_power_function_l300_300336

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

noncomputable def power_function (m n : ℤ) : ℝ → ℝ := 
  λ x, x ^ ((m : ℝ) / (n : ℝ))

theorem sufficient_condition_for_odd_power_function :
  is_odd_function (power_function 1 3) :=
by
  sorry

end sufficient_condition_for_odd_power_function_l300_300336


namespace total_cost_of_dinner_l300_300598

theorem total_cost_of_dinner
  (cost_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (tip_rate : ℝ)
  (sales_tax : ℝ)
  (tip : ℝ)
  (total_amount_paid : ℝ) :
  cost_of_food = 30 →
  sales_tax_rate = 0.095 →
  tip_rate = 0.10 →
  sales_tax = cost_of_food * sales_tax_rate →
  tip = cost_of_food * tip_rate →
  total_amount_paid = cost_of_food + sales_tax + tip →
  total_amount_paid = 35.85 :=
by
  intros cost_of_food_eq food sales_tax_rate_eq tax tip_rate_eq tip_eq sales_tax_eq tip_eq total_amount_paid_eq
  sorry

end total_cost_of_dinner_l300_300598


namespace solve_problem_together_time_l300_300388

theorem solve_problem_together_time :
  let A := 1 / 10 -- Person A's rate in problems per hour
  let B := 0.75 * A -- Person B's rate in problems per hour
  let combined_rate := A + B -- Combined rate of A and B
  let T := 1 / combined_rate -- Time for both to solve together
  T ≈ 5.71 := -- Prove T is approximately 5.71 hours
by
  let A : ℝ := 1 / 10
  let B : ℝ := 0.75 * A
  let combined_rate : ℝ := A + B
  let T : ℝ := 1 / combined_rate
  have h : T ≈ 5.71,
  from sorry
  exact h

end solve_problem_together_time_l300_300388


namespace total_students_l300_300882

theorem total_students 
  (x : ℕ)
  (jonas_marcos : 37)
  (jonas_nair : 3)
  (amanda_marcos : 15)
  (amanda_nair : 201) : 
  x - 33 = 187 → x + 2 = 222 :=
by 
  intros h1
  sorry

end total_students_l300_300882


namespace find_vector_u_l300_300690

def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_vw := v.1 * w.1 + v.2 * w.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  let scalar := dot_vw / dot_vv
  (scalar * v.1, scalar * v.2)

theorem find_vector_u (u : ℝ × ℝ)
  (h₁ : proj ⟨1, 2⟩ u = ⟨2, 4⟩)
  (h₂ : proj ⟨3, 1⟩ u = ⟨6, 2⟩) :
  u = ⟨6, 2⟩ :=
sorry

end find_vector_u_l300_300690


namespace find_divisor_l300_300154

theorem find_divisor
  (d : ℕ) (q : ℕ) (r : ℕ) (v : ℕ)
  (h_dividend : d = 144)
  (h_quotient : q = 13)
  (h_remainder : r = 1)
  (h_formula : d = (v * q) + r) : v = 11 :=
begin
  sorry
end

end find_divisor_l300_300154


namespace intersection_intervals_l300_300804

open Real

theorem intersection_intervals (a b : ℝ)
    (f g : ℝ → ℝ)
    (h₁ : f = λ x, 2*x^4 - a^2*x^2 + b - 1)
    (h₂ : g = λ x, 2*a*x^3 - 1) :
    (b ∈ Set.Ioo (3*a^4 / 128) a^4) ∨ (b < 0) := 
by
  sorry

end intersection_intervals_l300_300804


namespace businessmen_no_drink_l300_300639

theorem businessmen_no_drink 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (juice : ℕ) 
  (coffee_tea : ℕ) 
  (coffee_juice : ℕ) 
  (tea_juice : ℕ) 
  (all_three : ℕ) 
  (S : Finset ℕ) :
  total = 30 → coffee = 15 → tea = 12 → juice = 8 → coffee_tea = 6 → coffee_juice = 4 → tea_juice = 2 → all_three = 1 →
  (total - (coffee + tea + juice - coffee_tea - coffee_juice - tea_juice + all_three)) = 6 := by
  intros h_total h_coffee h_tea h_juice h_coffee_tea h_coffee_juice h_tea_juice h_all_three
  rw [h_total, h_coffee, h_tea, h_juice, h_coffee_tea, h_coffee_juice, h_tea_juice, h_all_three]
  norm_num
  exact eq.refl 6

end businessmen_no_drink_l300_300639


namespace max_and_min_of_z_in_G_l300_300277

def z (x y : ℝ) : ℝ := x^2 + y^2 - 2*x*y - x - 2*y

def G (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

theorem max_and_min_of_z_in_G :
  (∃ (x y : ℝ), G x y ∧ z x y = 12) ∧ (∃ (x y : ℝ), G x y ∧ z x y = -1/4) :=
sorry

end max_and_min_of_z_in_G_l300_300277


namespace equal_segments_pos_l300_300292

open EuclideanGeometry

-- Define the conditions
def symmetric_point (A B : Point) (l : Line) : Point :=
  reflect_over_line A l

theorem equal_segments_pos {A B C M N A' C' : Point} (h1: Triangle A B C) 
  (h2: Line M) (h3: Line N) (h4: Angle A B M = Angle C B N)
  (h5: A' = symmetric_point A B M) (h6: C' = symmetric_point C B N) :
  segment_length (A, C') = segment_length (A', C) := 
sorry

end equal_segments_pos_l300_300292


namespace part1_part2_l300_300711

-- Define the functions f, g, and h
def f (x : ℝ) := Real.log (x + 1)
def g (x : ℝ) := Real.exp x - 1
def h (x : ℝ) := f x - g x + 1

-- Part (1): Number of zeros of h(x).
theorem part1 : ∃! x : ℝ, h x = 0 :=
sorry

-- Part (2): Comparison of expressions.
theorem part2 : g (Real.exp 2 - Real.log 2 - 1) > Real.log (Real.exp 2 - Real.log 2) ∧
                Real.log (Real.exp 2 - Real.log 2) > 2 - f (Real.log 2) :=
sorry

end part1_part2_l300_300711


namespace product_decrease_increase_fifteenfold_l300_300785

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l300_300785


namespace area_common_region_l300_300001

open Real

noncomputable def shared_area_rectangle_circle_triangle : ℝ :=
  let rectangle : set (ℝ × ℝ) := {p | abs p.1 ≤ 5 ∧ abs p.2 ≤ 2}
  let circle : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 16}
  let triangle : set (ℝ × ℝ) :=
    let leg := (3, 0)
    let hypotenuse := (3 / sqrt 2, 3 / sqrt 2)
    {p | p.1 * hypotenuse.1 + p.2 * hypotenuse.2 ≤ 4.5 ∧ abs p.1 ≤ 1.5}
  let intersection := rectangle ∩ circle ∩ triangle
  sorry

theorem area_common_region:
  shared_area_rectangle_circle_triangle = 4.5 :=
  sorry

end area_common_region_l300_300001


namespace find_m_l300_300508

theorem find_m :
  ∃ (m k : ℕ), 
    (1001 * 1002 * ... * 2010 * 2011 = 2 ^ m * (2 * k + 1)) ∧ m = 1008 :=
sorry

end find_m_l300_300508


namespace lattice_points_in_region_l300_300602

theorem lattice_points_in_region : ∃! n : ℕ, n = 14 ∧ ∀ (x y : ℤ), (y = |x| ∨ y = -x^2 + 4) ∧ (-2 ≤ x ∧ x ≤ 1) → 
  (y = -x^2 + 4 ∧ y = |x|) :=
sorry

end lattice_points_in_region_l300_300602


namespace part1_part2_part3_l300_300398

-- Conditions
def condition1 (a : List ℝ) (n : ℕ) : Prop :=
  ∀ i, i < 2 * n → a.get i ∈ {1.0, -1.0}

def condition2 (a : List ℝ) (n : ℕ) : Prop :=
  (a.take (2 * n)).sum = 0

def condition3 (a : List ℝ) (n : ℕ) : Prop :=
  ∀ i, i < 2 * n − 1 → (a.take (i + 1)).sum ≥ 0

-- Part (I): List all A_6 that satisfy the given conditions
theorem part1 : 
  ∃ A6 : List (List ℝ), 
    (∀ a, a ∈ A6 → length a = 6 ∧ condition1 a 3 ∧ condition2 a 3 ∧ condition3 a 3)
    ∧ A6.length = 5 :=
sorry

-- Part (II): Find the set of possible values for a1 + a2 + ... + an
theorem part2 (k : ℕ) (h : k > 0) : 
  let n := 2 * k - 1 in 
  {m : ℤ | ∃ a : List ℝ, a.length = 2 * n ∧ condition1 a n ∧ condition2 a n ∧ condition3 a n ∧ (a.take n).sum = m} = 
  {m | ∃ q, m = 2 * q + 1 ∧ q ∈ {0, 1, ..., k-1}} :=
sorry

-- Part (III): Find the number of A_2n
theorem part3 (n : ℕ) (h : n > 0) : 
  {a : List ℝ | a.length = 2 * n ∧ condition1 a n ∧ condition2 a n ∧ condition3 a n}.size = 
  Nat.choose (2 * n) n / (n + 1) :=
sorry

end part1_part2_part3_l300_300398


namespace sam_distance_traveled_l300_300452

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300452


namespace AL_lt_KL_l300_300401

-- Define the circumcenter of triangle ABC
variable {A B C O K L : Type}
variable [Circumcenter O A B C]

-- Let K be the midpoint of the arc BC not containing A
axiom midpoint_arc_not_A : MidpointArcNotContainingA K B C

-- Let L be an arbitrary point on the circumcircle
axiom point_on_circumcircle : PointOnCircumcircle L A B C

-- Let K lies on the line AL
axiom K_on_line_AL : LiesOnLine K L A

-- Let triangles AHL and KML be similar
axiom similar_triangles_AHL_KML : SimilarTriangles (Triangle.mk A H L) (Triangle.mk K M L)

-- Prove AL < KL
theorem AL_lt_KL : Length (LineSegment.mk A L) < Length (LineSegment.mk K L) := sorry

end AL_lt_KL_l300_300401


namespace triangle_proof_l300_300767

theorem triangle_proof (a b : ℝ) (cosA : ℝ) (ha : a = 6) (hb : b = 5) (hcosA : cosA = -4 / 5) :
  (∃ B : ℝ, B = 30) ∧ (∃ area : ℝ, area = (9 * Real.sqrt 3 - 12) / 2) :=
  by
  sorry

end triangle_proof_l300_300767


namespace inequality_am_gm_l300_300355

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_am_gm_l300_300355


namespace sam_distance_l300_300455

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300455


namespace units_digit_product_odd_integers_l300_300142

theorem units_digit_product_odd_integers {P : ℕ → Prop} (hp : ∀ n, P n ↔ (10 ≤ n ∧ n ≤ 200 ∧ n % 2 = 1)) :
  (∏ n in (finset.filter P (finset.range 201)), n) % 10 = 5 :=
sorry

end units_digit_product_odd_integers_l300_300142


namespace eval_g_at_neg2_l300_300757

def g (x : ℝ) : ℝ := 5 * x + 2

theorem eval_g_at_neg2 : g (-2) = -8 := by
  sorry

end eval_g_at_neg2_l300_300757


namespace eliminated_team_girls_l300_300125

variable {G B : ℕ} -- G is the total number of girls originally, B is the total number of boys originally
variable {n : ℕ} -- n is the number of girls in the eliminated team

def total_team_members := 9 + 15 + 17 + 19 + 21

-- condition that the sum of all team members must be 81
axiom total_members_eq : total_team_members = 81

-- one team of girls has been eliminated
axiom girls_team_eliminated : ∃ g ∈ ({9, 15, 17, 19, 21} : set ℕ), g = n

-- remaining girls is three times the number of boys
axiom girls_remaining_are_three_times_boys : (G - n) = 3 * B

-- After one team of girls is eliminated, the remaining total team members
axiom remaining_members_after_elimination : (total_team_members - n) = G - n + B

-- The relationship between boys and total members after elimination
axiom remaining_member_equation : (total_team_members - n) = 4 * B

theorem eliminated_team_girls : n = 21 :=
by
  sorry

end eliminated_team_girls_l300_300125


namespace same_functions_l300_300988

def f (r : ℝ) (hr : r ≥ 0) : ℝ := π * r^2
def g (x : ℝ) (hx : x ≥ 0) : ℝ := π * x^2

theorem same_functions :
  ∀ (x : ℝ) (hx : x ≥ 0), f x hx = g x hx :=
by sorry

end same_functions_l300_300988


namespace determine_p5_l300_300405

/-- The given conditions -/
variables {a b x y : ℝ}
variables h1 : 2 * a * x + 3 * b * y = 6
variables h2 : 2 * a * x ^ 2 + 3 * b * y ^ 2 = 14
variables h3 : 2 * a * x ^ 3 + 3 * b * y ^ 3 = 33
variables h4 : 2 * a * x ^ 4 + 3 * b * y ^ 4 = 87

/-- The goal to prove -/
theorem determine_p5 : 2 * a * x ^ 5 + 3 * b * y ^ 5 = 528 := 
sorry

end determine_p5_l300_300405


namespace factor_of_polynomial_l300_300675

theorem factor_of_polynomial (t : ℚ) : (8 * t^2 + 17 * t - 10 = 0) ↔ (t = 5/8 ∨ t = -2) :=
by sorry

end factor_of_polynomial_l300_300675


namespace train_pass_time_l300_300625

variable (train_length : Float) -- The length of the train in meters.
variable (train_speed : Float) -- The speed of the train in kmph.
variable (man_speed : Float) -- The speed of the man in kmph.
variable (relative_speed : Float) -- The relative speed between the train and the man in kmph.
variable (relative_speed_m_per_s : Float) -- The relative speed between the train and the man in m/s.
variable (distance : Float) -- The distance to be covered by the train in meters.
variable (time : Float) -- The time taken for the train to pass the man in seconds.

-- Define the conditions and variables
def train_length_def := train_length = 250 -- meters
def train_speed_def := train_speed = 58 -- kmph
def man_speed_def := man_speed = 8 -- kmph
def relative_speed_def := relative_speed = train_speed - man_speed
def relative_speed_m_per_s_def := relative_speed_m_per_s = (relative_speed * 1000) / 3600
def distance_def := distance = train_length
def time_def := time = distance / relative_speed_m_per_s

-- Theorem statement
theorem train_pass_time : 
  train_length = 250 → 
  train_speed = 58 → 
  man_speed = 8 → 
  (train_speed - man_speed) * 1000 / 3600 ≈ 13.8889 → 
  (250 / ((train_speed - man_speed) * 1000 / 3600)) ≈ 18 :=
by
  intros train_length_def train_speed_def man_speed_def relative_speed_m_per_s_def
  rw [train_length_def, train_speed_def, man_speed_def, relative_speed_m_per_s_def]
  sorry

end train_pass_time_l300_300625


namespace minimum_dot_product_l300_300778

open EuclideanGeometry

-- Define the points A, B, D, and P, Q with their conditions
def Rectangle_ABCD (A B C D : Point) : Prop :=
  ∃ (P : Segment), segment_is_rectangle A B C D P ∧ length P = 2 ∧ height P = 1

-- Define conditions for points P and Q
def Point_P_on_DC (P D C : Point) (t : ℝ) : Prop :=
  P = (D.1 + t, D.2) ∧ 0 ≤ t ∧ t ≤ 2

def Point_Q_on_ext_CB (Q B C : Point) (t : ℝ) : Prop :=
  Q = (B.1, B.2 - t)

-- Define the vector representations
def vector_PA (P A : Point) : Vector := ⟨A.1 - P.1, A.2 - P.2⟩
def vector_PQ (P Q : Point) : Vector := ⟨Q.1 - P.1, Q.2 - P.2⟩

def dot_product (v1 v2 : Vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The main theorem to prove
theorem minimum_dot_product {A B C D P Q : Point} (t : ℝ)
  (H1 : Rectangle_ABCD A B C D)
  (H2 : Point_P_on_DC P D C t)
  (H3 : Point_Q_on_ext_CB Q B C t)
  : ∃ t, 0 ≤ t ∧ t ≤ 2 ∧ dot_product (vector_PA P A) (vector_PQ P Q) = 3 / 4 := 
sorry

end minimum_dot_product_l300_300778


namespace profit_percentage_theorem_l300_300983

variables (CP MP SP Profit : ℝ)
variables (discount markup_percentage profit_percentage : ℝ)

-- Conditions
def CP_value : CP = 180 := by sorry
def markup_percentage_value : markup_percentage = 0.4778 := by sorry
def discount_value : discount = 50 := by sorry

-- Derived definitions
def MP_calculation : MP = CP + (markup_percentage * CP) := by sorry
def SP_calculation : SP = MP - discount := by sorry
def Profit_calculation : Profit = SP - CP := by sorry
def profit_percentage_calculation : profit_percentage = (Profit / CP) * 100 := by sorry

-- Proof
theorem profit_percentage_theorem (CP : ℝ) (markup_percentage : ℝ) (discount : ℝ) : 
  (CP = 180) → 
  (markup_percentage = 0.4778) → 
  (discount = 50) → 
  profit_percentage = 20 :=
by
  intros hCP hMarkup hDiscount
  rw [hCP, hMarkup, hDiscount] at *
  unfold MP_calculation SP_calculation Profit_calculation profit_percentage_calculation
  -- Elaborate the proof using the provided calculations steps
  sorry

end profit_percentage_theorem_l300_300983


namespace gcd_repeated_integer_l300_300197

open Nat

theorem gcd_repeated_integer (m : ℕ) (h1 : 100 ≤ m) (h2 : m ≤ 999) : 
  gcd (1001001 * m) (1001001 * (m + 1)) = 1001001 :=
by
  sorry

end gcd_repeated_integer_l300_300197


namespace proof_AC_squared_interval_l300_300969

theorem proof_AC_squared_interval :
  let AB := 10
  let BC := 20
  let theta_min := real.pi / 4 -- 45 degrees in radians
  let theta_max := real.pi / 3 -- 60 degrees in radians
  ∀ θ, θ_min ≤ θ → θ ≤ theta_max →
  let cos_theta := real.cos θ
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_theta
  700 ≤ AC_squared ∧ AC_squared ≤ 800 :=
by
  sorry

end proof_AC_squared_interval_l300_300969


namespace AM_GM_inequality_l300_300482

theorem AM_GM_inequality (n m : ℕ) (a : ℕ → ℝ) 
  (hnm : n > m) (hm0 : m > 0) (hna_pos : ∀ i, i < n → a i > 0) :
  (∑ i in finset.range n, (a i) ^ m) * (∑ i in finset.range n, (a i) ^ (n - m)) 
  ≥ n^2 * (finset.range n).prod (λ i, a i) := 
by
  sorry

end AM_GM_inequality_l300_300482


namespace hexagon_coloring_possible_l300_300663

theorem hexagon_coloring_possible (A B C D E F : Type)
  [fintype A] [fintype B] [fintype C] [fintype D] [fintype E] [fintype F]
  (num_colors : ℕ)
  (h1 : num_colors = 7)
  (h2 : fintype.card A = num_colors)
  (h3 : fintype.card B = num_colors)
  (h4 : fintype.card C = num_colors - 1)
  (h5 : fintype.card D = num_colors - 1)
  (h6 : fintype.card E = num_colors - 1)
  (h7 : fintype.card F = num_colors) :
  fintype.card (A × B × C × D × E × F) = 63504 := by
  sorry

end hexagon_coloring_possible_l300_300663


namespace find_three_digit_number_l300_300676

theorem find_three_digit_number :
  ∃ (Π B Γ : ℕ), Π ≠ B ∧ B ≠ Γ ∧ Π ≠ Γ ∧ Π < 10 ∧ B < 10 ∧ Γ < 10 ∧ 
  (Π * 100 + B * 10 + Γ = (Π + B + Γ) * (Π + B + Γ + 1)) ∧ 
  (Π * 100 + B * 10 + Γ = 156) :=
sorry

end find_three_digit_number_l300_300676


namespace train_pass_time_l300_300624

open Classical

noncomputable def train_problem : ℝ :=
  let L := 110 in
  let V_train := 82 * (5 / 18) in
  let V_man := 6 * (5 / 18) in
  let V_rel := V_train + V_man in
  L / V_rel

theorem train_pass_time {t : ℝ} :
  t = 4.5 :=
  by
    let L := 110
    let V_train := 82 * (5 / 18)
    let V_man := 6 * (5 / 18)
    let V_rel := V_train + V_man
    let t := L / V_rel
    exact sorry

end train_pass_time_l300_300624


namespace arithmetic_expressions_correctness_l300_300629

theorem arithmetic_expressions_correctness :
  ((∀ (a b c : ℚ), (a + b) + c = a + (b + c)) ∧
   (∃ (a b c : ℚ), (a - b) - c ≠ a - (b - c)) ∧
   (∀ (a b c : ℚ), (a * b) * c = a * (b * c)) ∧
   (∃ (a b c : ℚ), a / b / c ≠ a / (b / c))) :=
by
  sorry

end arithmetic_expressions_correctness_l300_300629


namespace polynomial_degree_l300_300658

theorem polynomial_degree
  (b c d e f g h i : ℝ) 
  (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) (hh : h ≠ 0) (hi : i ≠ 0) :
  polynomial.degree ((X^5 + C b * X^8 + C c * X^2 + C d) *
                     (X^4 + C e * X^3 + C f) *
                     (X^2 + C g * X + C h) *
                     (X + C i)) = 15 :=
sorry

end polynomial_degree_l300_300658


namespace largest_divisor_of_three_consecutive_even_integers_product_l300_300409

theorem largest_divisor_of_three_consecutive_even_integers_product :
  ∀ (n : ℕ), ∃ m : ℕ, m = 24 ∧ 24 ∣ 2 * n * (2 * n + 2) * (2 * n + 4) :=
by 
  intros n
  use 24
  split
  . refl
  . sorry

end largest_divisor_of_three_consecutive_even_integers_product_l300_300409


namespace min_ties_to_ensure_pairs_l300_300593

variable (red blue green yellow : Nat)
variable (total_ties : Nat)
variable (pairs_needed : Nat)

-- Define the conditions
def conditions : Prop :=
  red = 120 ∧
  blue = 90 ∧
  green = 70 ∧
  yellow = 50 ∧
  total_ties = 27 ∧
  pairs_needed = 12

-- Define the statement to be proven
theorem min_ties_to_ensure_pairs : conditions red blue green yellow total_ties pairs_needed → total_ties = 27 :=
sorry

end min_ties_to_ensure_pairs_l300_300593


namespace base_6_arithmetic_l300_300278

noncomputable def base_6_to_base_10 (digits : List ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨i, d⟩ => d * 6^i).sum

noncomputable def base_10_to_base_6 (n : ℕ) : List ℕ :=
  let rec decompose_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else decompose_aux (n / 6) (n % 6 :: acc)
  decompose_aux n []

theorem base_6_arithmetic :
  let n1 := base_6_to_base_10 [1, 1, 1]
  let n2 := base_6_to_base_10 [2, 0, 2]
  let prod := 2 * n1
  let sum := prod + n2
  sum = 160 ∧ base_10_to_base_6 sum = [4, 2, 4] :=
by
  -- n1 = 43
  -- n2 = 74
  -- prod = 86
  -- sum = 160
  -- base_10_to_base_6 160 = [4, 2, 4]
  sorry

end base_6_arithmetic_l300_300278


namespace g_symmetric_about_pi_div_12_l300_300360

noncomputable def f (x : ℝ) : ℝ := sin x * (sin x - sqrt 3 * cos x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 12)

theorem g_symmetric_about_pi_div_12 :
  ∀ x : ℝ, g (π / 12 - x) = g (π / 12 + x) :=
sorry

end g_symmetric_about_pi_div_12_l300_300360


namespace lisa_flight_time_l300_300057

theorem lisa_flight_time :
  ∀ (d s : ℕ), (d = 256) → (s = 32) → ((d / s) = 8) :=
by
  intros d s h_d h_s
  sorry

end lisa_flight_time_l300_300057


namespace angle_ODC_eq_angle_OBC_l300_300802

variables {A B C D O : Type}
variables [parallelogram A B C D]

-- Assume O is a point in the interior of parallelogram ABCD
variable (O : A) 

-- Assume the angle condition
variable (angle_AOB_plus_angle_DOC_eq_180 : ∠AOB + ∠DOC = 180)

theorem angle_ODC_eq_angle_OBC 
  (h_parallelogram : parallelogram A B C D)
  (h_AOB_plus_DOC : ∠AOB + ∠DOC = 180) :
  ∠ODC = ∠OBC :=
sorry

end angle_ODC_eq_angle_OBC_l300_300802


namespace ratio_CP_PA_l300_300791

-- Definitions based on the conditions
variables {A B C D M P : Type}
variables (AB AC BD DC BC AD : ℝ)

-- Given Conditions
def condition1 : AB = 24 := sorry
def condition2 : AC = 15 := sorry
def condition3 : AD = 2 * M := sorry
def angleBisector (AB AC BD DC : ℝ) : Prop := AB / AC = BD / DC

-- Statement of the problem
theorem ratio_CP_PA (h1 : angleBisector AB AC BD DC) (h2 : AB = 24) (h3 : AC = 15) (h4 : AD = 2 * M) :
  ∃ m n : ℕ, (m.gcd n = 1) ∧ (CP / PA = 13 / 8) ∧ (m + n = 21) := sorry

end ratio_CP_PA_l300_300791


namespace min_value_y_l300_300908

theorem min_value_y : ∃ x : ℝ, ∀ y (x_val : ℝ), y = 2 * x_val^2 - 8 * x_val + 10 → y = 2 :=
begin
  sorry
end

end min_value_y_l300_300908


namespace probability_distance_greater_b_is_zero_l300_300634

theorem probability_distance_greater_b_is_zero :
  ∀ (a b : ℕ), a ∈ {3, 9, 27, 81, 243, 729} → b ∈ {3, 9, 27, 81, 243, 729} →
  (by : (∀ a b, |a - b| > b → false)) :=
begin
  sorry
end

end probability_distance_greater_b_is_zero_l300_300634


namespace mushrooms_safe_to_eat_l300_300417

theorem mushrooms_safe_to_eat (S : ℕ) (Total_mushrooms Poisonous_mushrooms Uncertain_mushrooms : ℕ)
  (h1: Total_mushrooms = 32)
  (h2: Poisonous_mushrooms = 2 * S)
  (h3: Uncertain_mushrooms = 5)
  (h4: S + Poisonous_mushrooms + Uncertain_mushrooms = Total_mushrooms) :
  S = 9 :=
sorry

end mushrooms_safe_to_eat_l300_300417


namespace complement_of_angle_29_18_l300_300752

def complement (angle_deg : ℕ) (angle_min : ℕ) : ℕ × ℕ :=
  let total_deg := 90
  let total_min := 0
  let result_min := (total_min + 60 - angle_min) % 60
  let result_deg := total_deg - angle_deg - if result_min < angle_min then 1 else 0
  (result_deg, result_min)

theorem complement_of_angle_29_18 : complement 29 18 = (60, 42) :=
by
  unfold complement
  -- Perform computation:
  -- 90 degrees minus 29 degrees 18 minutes
  -- 90 degrees equals to (90 degrees, 0 minutes)
  -- 29 degrees 18 minutes equals (29 degrees, 18 minutes)
  -- result in minutes = (0 + 60 - 18) % 60 = 42 minutes
  -- Adjust degree subtraction if the minute part wrapped around
  let result_deg := 60
  let result_min := 42
  exact rfl

end complement_of_angle_29_18_l300_300752


namespace bananas_left_l300_300212

theorem bananas_left (dozen_bananas : ℕ) (eaten_bananas : ℕ) (h1 : dozen_bananas = 12) (h2 : eaten_bananas = 2) : dozen_bananas - eaten_bananas = 10 :=
sorry

end bananas_left_l300_300212


namespace pages_written_in_a_year_l300_300012

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end pages_written_in_a_year_l300_300012


namespace triangle_min_diff_l300_300531

variable (XY YZ XZ : ℕ) -- Declaring the side lengths as natural numbers

theorem triangle_min_diff (h1 : XY < YZ ∧ YZ ≤ XZ) -- Condition for side length relations
  (h2 : XY + YZ + XZ = 2010) -- Condition for the perimeter
  (h3 : XY + YZ > XZ)
  (h4 : XY + XZ > YZ)
  (h5 : YZ + XZ > XY) :
  (YZ - XY) = 1 := -- Statement that the smallest possible value of YZ - XY is 1
sorry

end triangle_min_diff_l300_300531


namespace average_gas_mileage_correct_l300_300827

-- Define the conditions
def distance_sedan : ℕ := 150
def mileage_sedan : ℕ := 25
def distance_suv : ℕ := 180
def mileage_suv : ℕ := 15

-- Define the total distance and total fuel used
def total_distance : ℕ := distance_sedan + distance_suv
def fuel_used_sedan : ℕ := distance_sedan / mileage_sedan
def fuel_used_suv : ℕ := distance_suv / mileage_suv
def total_fuel_used : ℕ := fuel_used_sedan + fuel_used_suv

-- Define the goal: average gas mileage for the entire journey
def average_gas_mileage : ℝ := total_distance / total_fuel_used.to_real

-- The proof statement
theorem average_gas_mileage_correct :
  average_gas_mileage = 18.333 := 
by
  sorry

end average_gas_mileage_correct_l300_300827


namespace milk_water_mixture_initial_volume_l300_300186

theorem milk_water_mixture_initial_volume
  (M W : ℝ)
  (h1 : 2 * M = 3 * W)
  (h2 : 4 * M = 3 * (W + 58)) :
  M + W = 145 := by
  sorry

end milk_water_mixture_initial_volume_l300_300186


namespace top_card_is_heartsuit_probability_l300_300201

-- Definitions of conditions
def total_ranks : ℕ := 13
def total_suits : ℕ := 4
def total_cards : ℕ := total_ranks * total_suits
def heartsuit_cards : ℕ := total_ranks
def probability_of_heartsuit : ℚ := heartsuit_cards / total_cards

-- Theorem to prove the question equals the answer given the conditions
theorem top_card_is_heartsuit_probability : probability_of_heartsuit = 1 / 4 := by
  -- Proof omitted
  sorry

end top_card_is_heartsuit_probability_l300_300201


namespace sheena_weeks_to_complete_l300_300083

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l300_300083


namespace sum_of_positive_ks_l300_300500

theorem sum_of_positive_ks : 
  (∑ k in {k | ∃ α β : ℤ, k > 0 ∧ α + β = k ∧ α * β = -18}.toFinset, k) = 27 := 
by 
  sorry

end sum_of_positive_ks_l300_300500


namespace domain_of_f_l300_300656

noncomputable def f (x k : ℝ) := (3 * x ^ 2 + 4 * x - 7) / (-7 * x ^ 2 + 4 * x + k)

theorem domain_of_f {x k : ℝ} (h : k < -4/7): ∀ x, -7 * x ^ 2 + 4 * x + k ≠ 0 :=
by 
  intro x
  sorry

end domain_of_f_l300_300656


namespace tennis_balls_Sam_l300_300221

theorem tennis_balls_Sam (L F B S : ℕ) (hL : L = 84)
  (hF : F = L + (35 * L / 100))
  (hB : B = 7 * F / 2)
  (hS_combined : S = 3 * (F + L) / 4) :
  S = 148 :=
by 
  have hL' : L = 84 := by rw hL
  have hF' : F = 84 + (35 * 84 / 100) := by rw [hL, hF]
  have hB' : B = 7 * 119 / 2 := by rw [hF', hB]
  have hS' : S = 3 * (119 + 84) / 4 := by rw [hL, hF', hS_combined]
  linarith

end tennis_balls_Sam_l300_300221


namespace exists_close_pair_in_interval_l300_300840

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l300_300840


namespace postage_for_all_envelopes_l300_300516

def envelope := (length: ℕ) (height: ℕ)

def ratio (e: envelope) : ℚ := e.length / e.height

def extra_postage (r: ℚ) : Prop := r < 1.5 ∨ r > 3.0

def envelopes := [
  envelope.mk 7 5,
  envelope.mk 10 2,
  envelope.mk 5 5,
  envelope.mk 12 3
]

theorem postage_for_all_envelopes : 
  (envelopes.filter (λ e, extra_postage (ratio e))).length = 4 :=
  by
    sorry

end postage_for_all_envelopes_l300_300516


namespace find_angle_in_triangle_l300_300765

theorem find_angle_in_triangle 
  (a b c : ℝ)
  (A : ℝ)
  (h : a^2 = b^2 + real.sqrt 2 * b * c + c^2) : 
  A = 3 * real.pi / 4 :=
sorry

end find_angle_in_triangle_l300_300765


namespace equivalent_problem_l300_300912

variable (a b : ℤ)

def condition1 : Prop :=
  a * (-2)^3 + b * (-2) - 7 = 9

def condition2 : Prop :=
  8 * a + 2 * b - 7 = -23

theorem equivalent_problem (h : condition1 a b) : condition2 a b :=
sorry

end equivalent_problem_l300_300912


namespace probability_ratio_l300_300289

noncomputable def total_slips : ℕ := 40
noncomputable def slips_per_number : ℕ := 5
noncomputable def numbers : ℕ := 8
noncomputable def draw : ℕ := 4

noncomputable def total_combinations : ℕ := (40.choose 4)
noncomputable def p' : ℚ := (8 * (5.choose 4)) / total_combinations
noncomputable def q' : ℚ := (28 * (10 * 10)) / total_combinations

theorem probability_ratio : q' / p' = 70 :=
by
  sorry

end probability_ratio_l300_300289


namespace lines_in_parallel_planes_l300_300940

theorem lines_in_parallel_planes (P1 P2 : Plane) (l1 : Line P1) (l2 : Line P2) 
  (h_parallel_planes : P1 ∥ P2) : Parallel l1 l2 ∨ Skew l1 l2 :=
sorry

end lines_in_parallel_planes_l300_300940


namespace hyperbola_eccentricity_l300_300725

/--
Given a hyperbola with the following properties:
1. Point \( P \) is on the left branch of the hyperbola \( C \): \(\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1\), where \( a > 0 \) and \( b > 0 \).
2. \( F_2 \) is the right focus of the hyperbola.
3. One of the asymptotes of the hyperbola is perpendicular to the line segment \( PF_2 \).

Prove that the eccentricity \( e \) of the hyperbola is \( \sqrt{5} \).
-/
theorem hyperbola_eccentricity (a b e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P_on_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F2_is_focus : True) -- Placeholder for focus-related condition
  (asymptote_perpendicular : True) -- Placeholder for asymptote perpendicular condition
  : e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l300_300725


namespace part_a_part_b_l300_300153

def area_of_triangle_origin (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 / 2 * |x1 * y2 - x2 * y1|

theorem part_a (x1 y1 x2 y2 : ℝ) : 
     area_of_triangle_origin x1 y1 x2 y2 = 1 / 2 * |x1 * y2 - x2 * y1| :=
  sorry

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  1 / 2 * |x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x1 * y3 - x3 * y2|

theorem part_b (x1 y1 x2 y2 x3 y3 : ℝ) : 
     area_of_triangle x1 y1 x2 y2 x3 y3 = 1 / 2 * |x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x1 * y3 - x3 * y2| :=
  sorry

end part_a_part_b_l300_300153


namespace number_of_chinese_l300_300632

theorem number_of_chinese (total americans australians chinese : ℕ) 
    (h_total : total = 49)
    (h_americans : americans = 16)
    (h_australians : australians = 11)
    (h_chinese : chinese = total - americans - australians) :
    chinese = 22 :=
by
    rw [h_total, h_americans, h_australians] at h_chinese
    exact h_chinese

end number_of_chinese_l300_300632


namespace nate_total_distance_l300_300476

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l300_300476


namespace combined_probability_l300_300134

theorem combined_probability (pX pY : ℝ) (hX : pX = 1/5) (hY : pY = 2/7) :
    pX * pY = 2/35 := 
by
  subst hX
  subst hY
  norm_num
  exact (mul_div 1 2 5 7).symm

end combined_probability_l300_300134


namespace solve_inequalities_l300_300852

theorem solve_inequalities (x : ℝ) (h₁ : (x - 1) / 2 < 2 * x + 1) (h₂ : -3 * (1 - x) ≥ -4) : x ≥ -1 / 3 :=
by
  sorry

end solve_inequalities_l300_300852


namespace total_bike_cost_l300_300468

def marions_bike_cost : ℕ := 356
def stephanies_bike_cost : ℕ := 2 * marions_bike_cost

theorem total_bike_cost : marions_bike_cost + stephanies_bike_cost = 1068 := by
  sorry

end total_bike_cost_l300_300468


namespace number_of_snakes_l300_300341

-- Define the variables
variable (S : ℕ) -- Number of snakes

-- Define the cost constants
def cost_per_gecko := 15
def cost_per_iguana := 5
def cost_per_snake := 10

-- Define the number of each pet
def num_geckos := 3
def num_iguanas := 2

-- Define the yearly cost
def yearly_cost := 1140

-- Calculate the total monthly cost
def monthly_cost := num_geckos * cost_per_gecko + num_iguanas * cost_per_iguana + S * cost_per_snake

-- Calculate the total yearly cost
def total_yearly_cost := 12 * monthly_cost

-- Prove the number of snakes
theorem number_of_snakes : total_yearly_cost = yearly_cost → S = 4 := by
  sorry

end number_of_snakes_l300_300341


namespace intersect_segments_inscribed_circles_l300_300832

structure Tetrahedron (α : Type) :=
(A B C D : α)

structure Intersection (α : Type) :=
(E F : α)

theorem intersect_segments_inscribed_circles
    {α : Type} [LinearOrderedField α] (T : Tetrahedron α) (I : Intersection α) 
    (AE_intersect_BF : ∃ P : α, T.A ≠ T.B ∧ T.B ≠ T.C ∧ T.C ≠ T.D ∧ T.D ≠ T.A ∧ 
                           (line_through T.A I.E).intersects (line_through T.B I.F)) :
    ∃ Q : α, 
    (line_through T.C I.E).intersects (line_through T.D I.F) :=
sorry

end intersect_segments_inscribed_circles_l300_300832


namespace tan_alpha_l300_300206

theorem tan_alpha (α : ℝ) (A B C P Q R S : Point)
  (hABC : Triangle A B C)
  (hSides : (dist A B) = 13 ∧ (dist B C) = 14 ∧ (dist A C) = 15)
  (hBisectors : is_bisector_of_perimeter_area A B C P Q R S)
  (hAlpha : α = angle_between_lines P Q R S ∧ α < π/2) :
  tan α = 2 * tan (70 * π / 180) - (tan (70 * π / 180))^2 := by
sorry

end tan_alpha_l300_300206


namespace circumcircle_radius_is_one_l300_300792

-- Define the basic setup for the triangle with given sides and angles
variables {A B C : Real} -- Angles of the triangle
variables {a b c : Real} -- Sides of the triangle opposite these angles
variable (triangle_ABC : a = Real.sqrt 3 ∧ (c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0)) -- Conditions on the sides

-- Define the circumcircle radius
noncomputable def circumcircle_radius (a b c : Real) (A B C : Real) := a / (2 * (Real.sin A))

-- Statement of the problem to be proven
theorem circumcircle_radius_is_one (h : a = Real.sqrt 3)
  (h1 : c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0) :
  circumcircle_radius a b c A B C = 1 :=
sorry

end circumcircle_radius_is_one_l300_300792


namespace cos_arcsin_l300_300251

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l300_300251


namespace probability_cube_selection_l300_300955

/-- Define the structure related to the problem conditions --/
structure UnitCubes :=
  (total : ℕ := 125)
  (three_painted_faces : ℕ := 1)
  (two_painted_faces : ℕ := 9)
  (one_painted_face : ℕ := 9)
  (no_painted_faces : ℕ := 106)

/-- Calculate the probability of selecting one cube with 3 painted faces and one with no painted faces from 125 total cubes. --/
theorem probability_cube_selection :
  let total_ways := Nat.choose 125 2,
      successful_outcomes := 1 * 106
  in (successful_outcomes : ℝ) / (total_ways : ℝ) = 53 / 3875 := 
by 
  -- Sorry is used to skip the proof
  sorry

end probability_cube_selection_l300_300955


namespace last_group_markers_l300_300590

theorem last_group_markers:
  ∀ (total_students group1_students group2_students markers_per_box boxes_of_markers group1_markers group2_markers : ℕ),
    total_students = 30 →
    group1_students = 10 →
    group2_students = 15 →
    markers_per_box = 5 →
    boxes_of_markers = 22 →
    group1_markers = 2 →
    group2_markers = 4 →
    let total_markers := boxes_of_markers * markers_per_box in
    let used_markers1 := group1_students * group1_markers in
    let used_markers2 := group2_students * group2_markers in
    let remaining_students := total_students - group1_students - group2_students in
    let remaining_markers := total_markers - used_markers1 - used_markers2 in
    remaining_students > 0 →
    remaining_markers % remaining_students = 0 →
    remaining_markers / remaining_students = 6 :=
sorry

end last_group_markers_l300_300590


namespace decreasing_intervals_sin_decreasing_intervals_log_cos_l300_300097

theorem decreasing_intervals_sin (k : ℤ) :
  ∀ x : ℝ, 
    ( (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (π / 2 + 2 * k * π < x) ∧ (x < 3 * π / 2 + 2 * k * π)) :=
sorry

theorem decreasing_intervals_log_cos (k : ℤ) :
  ∀ x : ℝ, 
    ( (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π) ) ↔
    (∃ k : ℤ, (2 * k * π < x) ∧ (x < π / 2 + 2 * k * π)) :=
sorry

end decreasing_intervals_sin_decreasing_intervals_log_cos_l300_300097


namespace sam_drove_distance_l300_300433

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300433


namespace find_a_angle_l1_l3_exists_point_satisfies_conditions_l300_300742

-- Given definitions
def l1 (a : ℝ) := { p : ℝ × ℝ | 2 * p.1 - p.2 + a = 0 }
def l2 := { p : ℝ × ℝ | -4 * p.1 + 2 * p.2 + 1 = 0 }
def l3 := { p : ℝ × ℝ | p.1 + p.2 - 1 = 0 }
def point (x y : ℝ) := (x, y)

axiom distance_l1_l2 (a : ℝ) (h : a > 0) : real := 
  |a + 1 / 2| / sqrt (2^2 + (-1)^2)

axiom angle_between_lines (slope1 slope2 : ℝ) : ℝ :=
  arctan (slope1 - slope2) / (1 + slope1 * slope2)

-- Problem statements
theorem find_a (h : distance_l1_l2 3 (by norm_num1) = 7/10 * sqrt 5) : 3 = 3 := sorry

theorem angle_l1_l3 : arctan (3 - -1) / (1 + 3*1) = arctan 3 := sorry

theorem exists_point_satisfies_conditions :
  ∃ (P : ℝ × ℝ), 
    (P.1 > 0 ∧ P.2 > 0) ∧
    (let d1 := (abs (2 * P.1 - P.2 + 3)) / sqrt 5 in
     let d2 := (abs (2 * P.1 - P.2 - 1 / 2)) / sqrt 5 in 
     d1 = d2 / 2) ∧
    (let d1 := (abs (2 * P.1 - P.2 + 3)) / sqrt 5 in
     let d3 := (abs (P.1 + P.2 - 1)) / sqrt 2 in
     d1 / d3 = sqrt 2 / sqrt 5) ∧
    P = (1 / 9, 37 / 18) := sorry

end find_a_angle_l1_l3_exists_point_satisfies_conditions_l300_300742


namespace Ryan_stickers_l300_300079

def Ryan_has_30_stickers (R S T : ℕ) : Prop :=
  S = 3 * R ∧ T = S + 20 ∧ R + S + T = 230 → R = 30

theorem Ryan_stickers : ∃ R S T : ℕ, Ryan_has_30_stickers R S T :=
sorry

end Ryan_stickers_l300_300079


namespace range_of_d_l300_300710

variable {a_1 : ℝ} (d : ℝ) 
variable (S : ℕ → ℝ) 

def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem range_of_d (h1 : a_1 ≠ 0) (h2 : sum_arithmetic_sequence a_1 d 5 * a_1 + 15 = 0) : 
  d ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
sorry

end range_of_d_l300_300710


namespace height_of_smaller_cone_is_18_l300_300957

theorem height_of_smaller_cone_is_18
  (height_frustum : ℝ)
  (area_larger_base : ℝ)
  (area_smaller_base : ℝ) :
  let R := (area_larger_base / π).sqrt
  let r := (area_smaller_base / π).sqrt
  let ratio := r / R
  let H := height_frustum / (1 - ratio)
  let h := ratio * H
  height_frustum = 18 ∧ area_larger_base = 400 * π ∧ area_smaller_base = 100 * π
  → h = 18 := by
  sorry

end height_of_smaller_cone_is_18_l300_300957


namespace min_M_value_l300_300701

noncomputable def max_pq (p q : ℝ) : ℝ := if p ≥ q then p else q

noncomputable def M (x y : ℝ) : ℝ := max_pq (|x^2 + y + 1|) (|y^2 - x + 1|)

theorem min_M_value : (∀ x y : ℝ, M x y ≥ (3 : ℚ) / 4) ∧ (∃ x y : ℝ, M x y = (3 : ℚ) / 4) :=
sorry

end min_M_value_l300_300701


namespace probability_one_class_no_spot_l300_300888

theorem probability_one_class_no_spot :
  let spots := 6
  let classes := 3
  let total_ways := (3 + (5 * 3) + (10))  -- sum of distributions from all scenarios
  let favorable_ways := (5 * 3)  -- ways from the second scenario
  (favorable_ways : ℚ) / total_ways = 15 / 28 :=
by
  let spots := 6
  let classes := 3
  let total_ways := (3 + (5 * 3) + (10))  
  let favorable_ways := (5 * 3)  
  exact 15 / 28
  sorry

end probability_one_class_no_spot_l300_300888


namespace sum_of_solutions_l300_300548

theorem sum_of_solutions : 
  (∑ (x : ℚ) in ({x : ℚ | x = |2 * x - |100 - 2 * x||}.toFinset), x) = 400 / 3 := 
by
  sorry

end sum_of_solutions_l300_300548


namespace semicircle_contains_three_points_hemisphere_contains_four_points_l300_300164

theorem semicircle_contains_three_points (P : Fin 4 → ℝ × ℝ) (h : ∀ i, (P i).1^2 + (P i).2^2 = 1) :
  ∃ (S : ℝ × ℝ → Prop), (∀ x y, S x → S y → (x.1 * y.1 + x.2 * y.2 ≥ 0)) ∧
  (∃ (a b c : Fin 4), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ S (P a) ∧ S (P b) ∧ S (P c)) :=
by
  sorry

theorem hemisphere_contains_four_points (P : Fin 5 → ℝ × ℝ × ℝ) (h : ∀ i, (P i).1^2 + (P i).2^2 + (P i).3^2 = 1) :
  ∃ (H : ℝ × ℝ × ℝ → Prop), (∀ x y, H x → H y → (x.1 * y.1 + x.2 * y.2 + x.3 * y.3 ≥ 0)) ∧
  (∃ (a b c d : Fin 5), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ a ≠ c ∧ b ≠ d ∧ H (P a) ∧ H (P b) ∧ H (P c) ∧ H (P d)) :=
by
  sorry

end semicircle_contains_three_points_hemisphere_contains_four_points_l300_300164


namespace math_competition_l300_300586

theorem math_competition (n : ℕ)
  (p : ℕ → ℕ → ℕ)
  (n_r : ℕ → ℕ)
  (h1 : ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ 6 → p i j > (2 / 5) * n)
  (h2 : ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ 6 → p i j ≥ Nat.ceil ((2 * n) / 5 + 1))
  (h3 : ∑ r in Finset.range 7, n_r r = n)
  (h4 : n_r 6 = 0) : ∃ a b : ℕ, a ≠ b ∧ n_r 5 > 1 :=
by
  sorry

end math_competition_l300_300586


namespace part1_part2_part3_l300_300704

noncomputable def f : ℝ → ℝ := sorry -- Given f is a function on ℝ with domain (0, +∞)

axiom domain_pos (x : ℝ) : 0 < x
axiom pos_condition (x : ℝ) (h : 1 < x) : 0 < f x
axiom functional_eq (x y : ℝ) : f (x * y) = f x + f y
axiom specific_value : f (1/3) = -1

-- (1) Prove: f(1/x) = -f(x)
theorem part1 (x : ℝ) (hx : 0 < x) : f (1 / x) = - f x := sorry

-- (2) Prove: f(x) is an increasing function on its domain
theorem part2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f x1 < f x2 := sorry

-- (3) Prove the range of x for the inequality
theorem part3 (x : ℝ) (hx : 0 < x) (hx2 : 0 < x - 2) : 
  f x - f (1 / (x - 2)) ≥ 2 ↔ 1 + Real.sqrt 10 ≤ x := sorry

end part1_part2_part3_l300_300704


namespace equiv_functions_l300_300986

theorem equiv_functions (x : ℝ) : (λ x, x) = (λ x, (x^3)^(1/3)) :=
by
  sorry

end equiv_functions_l300_300986


namespace cos_arcsin_l300_300248

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l300_300248


namespace problem_statement_l300_300692

def divisors_count (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def f1 (n : ℕ) : ℕ := 3 * divisors_count n

def f_j (j n : ℕ) : ℕ :=
  if j = 1 then f1 n else f1 (f_j (j - 1) n)

noncomputable def count_values (N : ℕ) : ℕ :=
  (Finset.range (N + 1)).filter (λ n, f_j 30 n = 18).card

-- Problem statement
theorem problem_statement : count_values 30 = 2 := by
  sorry

end problem_statement_l300_300692


namespace vasya_correct_l300_300911

theorem vasya_correct (x : ℝ) (h : x^2 + x + 1 = 0) : 
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 :=
by 
  sorry

end vasya_correct_l300_300911


namespace maximum_a_for_increasing_y_l300_300695

theorem maximum_a_for_increasing_y : 
  ∃ a : ℝ, (∀ x : ℝ, x ≤ a → y = -x^2 + 2x - 2 ∧ (y' = ∂ y / ∂ x) →  y' > 0 ) → a = 1 := 
sorry

end maximum_a_for_increasing_y_l300_300695


namespace all_real_possible_values_l300_300811

theorem all_real_possible_values 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 1) : 
  ∃ r : ℝ, r = (a^4 + b^4 + c^4) / (ab + bc + ca) :=
sorry

end all_real_possible_values_l300_300811


namespace determine_finalists_by_median_l300_300528

open Real

-- Definitions
def top_students {n : ℕ} (scores : list ℝ) (k : ℕ) : list ℝ :=
  list.take k (list.sort (≤) scores)

def median_score {n : ℕ} (scores : list ℝ) : ℝ :=
  scores.nth ((scores.length / 2) - 1).get_or_else 0

-- Problem statement in Lean
theorem determine_finalists_by_median (scores : list ℝ) (h_len : scores.length = 20) (h_dist : list.nodup scores) :
  ∀ s ∈ scores, (s ∈ top_students scores 10 ↔ s > median_score scores) :=
sorry

end determine_finalists_by_median_l300_300528


namespace time_to_cross_train_B_l300_300535

-- Constants for the problem
def length_train_A : ℝ := 200
def length_train_B : ℝ := 150

def speed_train_A_kmh : ℝ := 54
def speed_train_B_kmh : ℝ := 36

-- Conversion factors
def km_per_hour_to_m_per_s (speed: ℝ) : ℝ := speed * 1000 / 3600

-- Speeds in m/s
def speed_train_A : ℝ := km_per_hour_to_m_per_s speed_train_A_kmh
def speed_train_B : ℝ := km_per_hour_to_m_per_s speed_train_B_kmh

-- Combined length and relative speed
def combined_length : ℝ := length_train_A + length_train_B
def relative_speed : ℝ := speed_train_A + speed_train_B

-- The time it takes for Arun to completely cross train B
def time_taken : ℝ := combined_length / relative_speed

-- Proof statement
theorem time_to_cross_train_B : time_taken = 14 := by
  sorry

end time_to_cross_train_B_l300_300535


namespace power_function_odd_condition_l300_300335

def sufficient_condition_for_odd_function (m n : ℤ) : Prop :=
  m = 1 ∧ n = 3

theorem power_function_odd_condition (m n : ℤ) :
  (∀ x : ℝ, f x = x^((m : ℝ) / (n : ℝ)) ∧ (∀ x : ℝ, f (-x) = -f x)) ↔ sufficient_condition_for_odd_function m n := 
  by
    sorry

end power_function_odd_condition_l300_300335


namespace tiling_mod_1000_l300_300168

def num_ways_to_tile (n : ℕ) (tile_colors : ℕ) : ℕ := 
  ∑ i in (finset.range (n+1)).filter (λ i, i ≥ 3), 
    nat.choose (n-1) (i-1) * (tile_colors ^ i - 3 * (2 ^ i) + 3)

theorem tiling_mod_1000 :
  (num_ways_to_tile 9 3) % 1000 = 663 :=
by
  -- Proof omitted
  sorry

end tiling_mod_1000_l300_300168


namespace cos_arcsin_l300_300252

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l300_300252


namespace anoop_joined_after_6_months_l300_300929

theorem anoop_joined_after_6_months (arjun_investment : ℕ) (anoop_investment : ℕ) (months_in_year : ℕ)
  (arjun_time : ℕ) (anoop_time : ℕ) :
  arjun_investment * arjun_time = anoop_investment * anoop_time →
  anoop_investment = 2 * arjun_investment →
  arjun_time = months_in_year →
  anoop_time + arjun_time = months_in_year →
  anoop_time = 6 :=
by sorry

end anoop_joined_after_6_months_l300_300929


namespace find_A_l300_300813

variables (A B C : ℝ)
def f (x : ℝ) : ℝ := A * x - 3 * B^2
def g (x : ℝ) : ℝ := B * x
def h (x : ℝ) : ℝ := x + C

theorem find_A : f A B C (g A B C (h A B C 1)) = 0 → A = 3 * B / (1 + C) :=
begin
  sorry
end

end find_A_l300_300813


namespace agnes_flight_cost_l300_300365

theorem agnes_flight_cost
  (booking_fee : ℝ) (cost_per_km : ℝ) (distance_XY : ℝ)
  (h1 : booking_fee = 120)
  (h2 : cost_per_km = 0.12)
  (h3 : distance_XY = 4500) :
  booking_fee + cost_per_km * distance_XY = 660 := 
by
  sorry

end agnes_flight_cost_l300_300365


namespace initial_distance_between_stones_l300_300136

theorem initial_distance_between_stones :
  ∀ (v0 : ℝ) (H : ℝ), H = 40 → 
  (∀ g : ℝ, g = 9.8 →
  (∀ t : ℝ, t = H / v0 →
  ∃ S : ℝ, S = H * (Real.sqrt 2) ∧ H = 40)) :=
begin
  intros v0 H H_eq g g_eq t t_eq,
  use H * Real.sqrt 2,
  split,
  { exact eq.refl (H * Real.sqrt 2) },
  { exact H_eq }
end

end initial_distance_between_stones_l300_300136


namespace number_of_willow_trees_l300_300149

theorem number_of_willow_trees (interval : ℕ) (circumference : ℕ) : interval = 30 → circumference = 1200 → circumference / interval = 40 :=
by
  intros h1 h2
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (nat.succ_pos 39) (by refl)

end number_of_willow_trees_l300_300149


namespace tomato_difference_l300_300956

theorem tomato_difference (tomatoes_before : ℕ) (tomatoes_picked : ℕ) (h_before : tomatoes_before = 17) (h_picked : tomatoes_picked = 9) :
  (tomatoes_before - tomatoes_picked = 8) :=
by
  rw [h_before, h_picked]
  norm_num
  sorry

end tomato_difference_l300_300956


namespace M_inter_N_is_01_l300_300046

variable (x : ℝ)

def M := { x : ℝ | Real.log (1 - x) < 0 }
def N := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem M_inter_N_is_01 : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  -- Proof will go here
  sorry

end M_inter_N_is_01_l300_300046


namespace SamDrove200Miles_l300_300446

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300446


namespace solve_fx_eq_1_l300_300868

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 2^(-x) - 1 else sqrt x

theorem solve_fx_eq_1 (x : ℝ) : f x = 1 ↔ x = 1 ∨ x = -1 :=
by
  sorry

end solve_fx_eq_1_l300_300868


namespace fuel_tank_capacity_l300_300585

theorem fuel_tank_capacity
  (x : ℝ)   -- the capacity of the fuel tank in gallons
  (mileage_pre_mod : ℝ) (mileage_pre_mod = 28)   -- the mileage per gallon before modification
  (fuel_efficiency_ratio : ℝ) (fuel_efficiency_ratio = 0.80)   -- the fuel efficiency ratio after modification (80%)
  (additional_miles : ℝ) (additional_miles = 105) :   -- additional miles the car can travel after modification
  x = 93.75 :=
by
  sorry

end fuel_tank_capacity_l300_300585


namespace SamDrove200Miles_l300_300442

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300442


namespace minimum_cubes_l300_300204

-- Define a structure with front and left side views showing 4 squares each
structure CubeStructure (front_view left_side_view : ℕ) where
  front_view_condition : front_view = 4
  left_side_view_condition : left_side_view = 4

-- Prove that the minimum number of cubes needed is 4
theorem minimum_cubes (C : CubeStructure 4 4) : 4 := by
  sorry

end minimum_cubes_l300_300204


namespace solution_set_of_inequality_l300_300115

theorem solution_set_of_inequality :
  { x : ℝ | |x - 2| > ∫ t in 0..1, 2 * t }
  = { x : ℝ | x < 1 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end solution_set_of_inequality_l300_300115


namespace players_in_physics_class_l300_300638

theorem players_in_physics_class (total players_math players_both : ℕ)
    (h1 : total = 15)
    (h2 : players_math = 9)
    (h3 : players_both = 4) :
    (players_math - players_both) + (total - (players_math - players_both + players_both)) + players_both = 10 :=
by {
  sorry
}

end players_in_physics_class_l300_300638


namespace find_n_l300_300283

variable (n : ℚ)

theorem find_n (h : (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4)) : 
  n = -2 / 3 :=
by
  sorry

end find_n_l300_300283


namespace actual_time_before_storm_l300_300104

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end actual_time_before_storm_l300_300104


namespace solve_equation_sin_cos_l300_300849

theorem solve_equation_sin_cos (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) →
  (sin y ≠ 0) →
  (sin^2 x + 1 / sin^2 x)^3 + (sin^2 y + 1 / sin^2 y)^3 = 16 * cos z →
  x = (π / 2) + π * n ∧ y = (π / 2) + π * k ∧ z = 2 * π * m :=
by
  intro h1 h2 heq
  sorry

end solve_equation_sin_cos_l300_300849


namespace true_or_false_is_true_l300_300318

theorem true_or_false_is_true (p q : Prop) (hp : p = true) (hq : q = false) : p ∨ q = true :=
by
  sorry

end true_or_false_is_true_l300_300318


namespace rotation_combined_translation_l300_300199

noncomputable def parallel_displacement (p1 p2 : Point) : Transformation :=
  translation (p2 - p1)

theorem rotation_combined_translation (F : Figure) (O O1 : Point) (alpha : Angle) :
  (rotate F O alpha) → (rotate F O1 (-alpha)) → (translate (O → O1)) = true :=
  sorry

end rotation_combined_translation_l300_300199


namespace solve_equation_solve_inequality_l300_300576

-- Defining the first problem
theorem solve_equation (x : ℝ) : 3 * (x - 2) - (1 - 2 * x) = 3 ↔ x = 2 := 
by
  sorry

-- Defining the second problem
theorem solve_inequality (x : ℝ) : (2 * x - 1 < 4 * x + 3) ↔ (x > -2) :=
by
  sorry

end solve_equation_solve_inequality_l300_300576


namespace maximum_xyz_l300_300713

theorem maximum_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h: x ^ (Real.log x / Real.log y) * y ^ (Real.log y / Real.log z) * z ^ (Real.log z / Real.log x) = 10) : 
  x * y * z ≤ 10 := 
sorry

end maximum_xyz_l300_300713


namespace prove_expression_value_l300_300351

-- Define the conditions
variables {a b c d m : ℤ}
variable (h1 : a + b = 0)
variable (h2 : |m| = 2)
variable (h3 : c * d = 1)

-- State the theorem
theorem prove_expression_value : (a + b) / (4 * m) + 2 * m ^ 2 - 3 * c * d = 5 :=
by
  -- Proof goes here
  sorry

end prove_expression_value_l300_300351


namespace max_a_value_l300_300739

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a - 1) * x^2 - (a - 1) * x + 2022 ∧ 
                                (a - 1) * x^2 - (a - 1) * x + 2022 ≤ 2022) →
  a = 16177 :=
sorry

end max_a_value_l300_300739


namespace nine_by_one_tiling_l300_300166

theorem nine_by_one_tiling (M : ℕ) 
  (h1 : ∀ T : Finset ℕ,
  (T.card = 9) → 
  (∀ x ∈ T, x ≤ 9) → 
  (∀ t ∈ T, t = m ∧ m ∈ {1,2,3,4,5,6,7,8,9}) → 
  (∀ c ∈ {red, blue, green}, ∃ t ∈ T, t = c) → 
  ∑ t in T, t * t = M) : 
  M % 1000 = 990 := 
sorry

end nine_by_one_tiling_l300_300166


namespace prob_draw_l300_300913

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw_l300_300913


namespace ap_length_l300_300828

theorem ap_length (AB CD AD : ℝ) (P : ℝ) (x : ℝ) 
  (H1 : AB = 10) (H2 : CD = 10) (H3 : AD = 10)
  (H4 : AP = 3 * CP) (H5 : x = CP) 
  (H6 : P = CP) (H7 : AC = 4 * x)
  (H8 : ∠O1 PO2 = 90) : 
  AP = 7.5 * Real.sqrt 2 := 
sorry

end ap_length_l300_300828


namespace sum_of_roots_of_polynomials_l300_300404

theorem sum_of_roots_of_polynomials :
  ∃ (a b : ℝ), (a^4 - 16 * a^3 + 40 * a^2 - 50 * a + 25 = 0) ∧ (b^4 - 24 * b^3 + 216 * b^2 - 720 * b + 625 = 0) ∧ (a + b = 7 ∨ a + b = 3) :=
by 
  sorry

end sum_of_roots_of_polynomials_l300_300404


namespace car_stopping_probability_l300_300997

theorem car_stopping_probability :
  let pG_A := (1 : ℚ) / 3,
      pG_B := (1 : ℚ) / 2,
      pG_C := (2 : ℚ) / 3,
      pR_A := 1 - pG_A,
      pR_B := 1 - pG_B,
      pR_C := 1 - pG_C,
      p_stop_A := pR_A * pG_B * pG_C,
      p_stop_B := pG_A * pR_B * pG_C,
      p_stop_C := pG_A * pG_B * pR_C,
      p_stopping_once := p_stop_A + p_stop_B + p_stop_C
  in p_stopping_once = 7 / 18 := 
by 
  sorry

end car_stopping_probability_l300_300997


namespace correct_average_l300_300569

-- Define the conditions given in the problem
def avg_incorrect : ℕ := 46 -- incorrect average
def n : ℕ := 10 -- number of values
def incorrect_num : ℕ := 25
def correct_num : ℕ := 75
def diff : ℕ := correct_num - incorrect_num

-- Define the total sums
def total_incorrect : ℕ := avg_incorrect * n
def total_correct : ℕ := total_incorrect + diff

-- Define the correct average
def avg_correct : ℕ := total_correct / n

-- Statement in Lean 4
theorem correct_average :
  avg_correct = 51 :=
by
  -- We expect users to fill the proof here
  sorry

end correct_average_l300_300569


namespace range_of_c_l300_300049

-- Define the function
def f (x c : ℝ) : ℝ := x^2 + 2*x - c

-- Proposition p: The domain of y = log(f(x, c)) is ℝ
def proposition_p (c : ℝ) : Prop := ∀ x : ℝ, f(x, c) > 0

-- Proposition q: The range of y = log(f(x, c)) is ℝ
def proposition_q (c : ℝ) : Prop := ∀ t > 0, ∃ x : ℝ, f(x, c) = t

-- The final theorem to prove
theorem range_of_c (c : ℝ) :
  (proposition_p c ∧ ¬ proposition_q c) ∨ (¬ proposition_p c ∧ proposition_q c) ↔ c < -1 := sorry

end range_of_c_l300_300049


namespace cos_arcsin_l300_300234

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l300_300234


namespace grandmother_times_older_l300_300470

variables (M G Gr : ℕ)

-- Conditions
def MilenasAge : Prop := M = 7
def GrandfatherAgeRelation : Prop := Gr = G + 2
def AgeDifferenceRelation : Prop := Gr - M = 58

-- Theorem to prove
theorem grandmother_times_older (h1 : MilenasAge M) (h2 : GrandfatherAgeRelation G Gr) (h3 : AgeDifferenceRelation M Gr) :
  G / M = 9 :=
sorry

end grandmother_times_older_l300_300470


namespace royalty_ratio_decrease_l300_300926

theorem royalty_ratio_decrease :
  let royalties1 := 4.0
  let sales1 := 20.0
  let royalties2 := 9.0
  let sales2 := 108.0
  let ratio1 := royalties1 / sales1
  let ratio2 := royalties2 / sales2
  let pct_decrease := ((ratio1 - ratio2) / ratio1) * 100
  pct_decrease ≈ 58.35 :=
by
  sorry

end royalty_ratio_decrease_l300_300926


namespace largest_two_digit_prime_factor_of_binom_180_90_l300_300542

-- Definitions for the conditions
def binom (n k : ℕ) := n.choose k
def n : ℕ := binom 180 90

-- The prime factor we are considering
def is_prime (p : ℕ) : Prop := Nat.Prime p
def two_digit_prime (p : ℕ) : Prop := 10 ≤ p ∧ p < 100 ∧ is_prime p

-- The statement to be proved
theorem largest_two_digit_prime_factor_of_binom_180_90 :
  ∃ p, two_digit_prime p ∧ p ∣ n ∧ ∀ q, two_digit_prime q ∧ q ∣ n → q ≤ p :=
begin
  sorry
end

end largest_two_digit_prime_factor_of_binom_180_90_l300_300542


namespace months_A_put_oxen_for_grazing_l300_300207

theorem months_A_put_oxen_for_grazing 
    (total_rent : ℝ)
    (cost_A : ℝ)
    (cost_B : ℝ)
    (cost_C : ℝ)
    (share_C : ℝ)
    (months_B : ℝ)
    (months_C : ℝ)
    (oxen_A : ℝ)
    (oxen_B : ℝ)
    (oxen_C : ℝ) : ℝ :=
  
total_rent = 175 ∧
cost_B = oxen_B * months_B ∧
cost_C = oxen_C * months_C ∧
share_C = 45 ∧
share_C = cost_C →
let months_A := ((total_rent - cost_B - cost_C) / oxen_A) in months_A = 7

example : months_A_put_oxen_for_grazing 175 (10 * x) (12 * 5) (15 * 3) 45 5 3 10 12 15 = 7 := sorry

end months_A_put_oxen_for_grazing_l300_300207


namespace fourth_number_in_sequence_l300_300866

noncomputable def fifth_number_in_sequence : ℕ := 78
noncomputable def increment : ℕ := 11
noncomputable def final_number_in_sequence : ℕ := 89

theorem fourth_number_in_sequence : (fifth_number_in_sequence - increment) = 67 := by
  sorry

end fourth_number_in_sequence_l300_300866


namespace merchant_spent_initially_500_rubles_l300_300185

theorem merchant_spent_initially_500_rubles
  (x : ℕ)
  (h1 : x + 100 > x)
  (h2 : x + 220 > x + 100)
  (h3 : x * (x + 220) = (x + 100) * (x + 100))
  : x = 500 := sorry

end merchant_spent_initially_500_rubles_l300_300185


namespace alex_needs_additional_coins_l300_300627

theorem alex_needs_additional_coins : 
  let friends := 12
  let coins := 63
  let total_coins_needed := (friends * (friends + 1)) / 2 
  let additional_coins_needed := total_coins_needed - coins
  additional_coins_needed = 15 :=
by sorry

end alex_needs_additional_coins_l300_300627


namespace sam_drove_distance_l300_300434

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300434


namespace find_ellipse_semi_axes_l300_300506

variables (r r' φ : ℝ)

def ellipse_semi_axes (r r' φ : ℝ) : ℝ × ℝ :=
  let sum   := r^2 + r'^2 + 2 * r * r' * sin φ,
      diff  := r^2 + r'^2 - 2 * r * r' * sin φ in
  ( 0.5 * (sqrt sum + sqrt diff), 0.5 * (sqrt sum - sqrt diff) )

theorem find_ellipse_semi_axes (a b : ℝ) :
  2 * r = 2 * r' → -- Condition that r and r' are the lengths of the semi-diameters
  (a, b) = ellipse_semi_axes r r' φ := by
  sorry

end find_ellipse_semi_axes_l300_300506


namespace cross_shape_perimeter_l300_300619

theorem cross_shape_perimeter
  (total_area : ℝ)
  (num_squares : ℕ)
  (side_length : ℝ)
  (total_area_eq : total_area = 125)
  (num_squares_eq : num_squares = 5)
  (side_length_eq : side_length^2 = total_area / num_squares) :
  let perimeter := 4 * side_length * 4 in
  perimeter = 80 :=
by
  sorry

end cross_shape_perimeter_l300_300619


namespace remainder_division_l300_300198

theorem remainder_division {N : ℤ} (k : ℤ) (h : N = 125 * k + 40) : N % 15 = 10 :=
sorry

end remainder_division_l300_300198


namespace mode_and_median_of_data_set_l300_300325

-- Definitions of the data set
def data_set : List ℕ := [2, 4, 2, 5, 7]

-- Proof statement
theorem mode_and_median_of_data_set :
  (mode data_set = 2) ∧ (median data_set = 4) :=
sorry

end mode_and_median_of_data_set_l300_300325


namespace incorrect_propositions_l300_300042

-- Definitions of the conditions
variables (a b c : ℝ)
def distinct_numbers : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def geometric_sequence : Prop := b^2 = a * c ∧ a * c > 0
def arithmetic_sequence : Prop := ∃ d : ℝ, b = a + d ∧ c = a + 2 * d
def irrational_numbers (x : ℝ) : Prop := ¬ ∃ q : ℚ, x = (q : ℝ)

-- Propositions
def prop1 : Prop := geometric_sequence a b c
def prop2 : Prop :=
  ∀ d : ℝ, arithmetic_sequence a b c → ¬ geometric_sequence a b c
def prop3 : Prop :=
  arithmetic_sequence a b c ∧ ∀ x, (x = a ∨ x = b ∨ x = c) → irrational_numbers x →
  ∀ d : ℝ, (b = a + d ∧ c = a + 2 * d) → irrational_numbers d

-- The theorem statement
theorem incorrect_propositions : distinct_numbers a b c →
  ((¬ prop1 ∨ ¬ prop2 ∨ ¬ prop3) ∧ (¬ prop1 ∧ ¬ prop2 ∧ ¬ prop3)) :=
sorry

end incorrect_propositions_l300_300042


namespace discount_proof_l300_300617

variable (OriginalPrice : ℝ)

-- Definition of the conditions
def SalePrice := 0.6 * OriginalPrice
def FinalCost := 0.7 * SalePrice

theorem discount_proof : FinalCost = 0.42 * OriginalPrice := by
  rw [FinalCost, SalePrice]
  simp
  sorry

end discount_proof_l300_300617


namespace monotonic_intervals_extreme_values_range_of_k_if_increasing_l300_300052

noncomputable def f (x : ℝ) (k : ℝ) := (x - 1) * real.exp x - k * x^2

theorem monotonic_intervals_extreme_values (k : ℝ) (H : k = 1) :
  (∀ x < 0, (x - 1) * real.exp x - x^2 < (0 - 1) * real.exp 0 - 0^2) ∧
  (∀ x > real.log 2, (x - 1) * real.exp x - x^2 > (real.log 2 - 1) * real.exp (real.log 2) - (real.log 2)^2) ∧
  (∃ x₁ x₂, f x₁ 1 = -1 ∧ f x₂ 1 = 2 * (real.log 2 - 1) - (real.log 2)^2) :=
sorry

theorem range_of_k_if_increasing (k : ℝ) :
  (∀ x ≥ 0, (x - 1) * real.exp x - k * x^2 ≥ (0 - 1) * real.exp 0 - k * 0^2) →
  k ≤ 1 / 2 :=
sorry

end monotonic_intervals_extreme_values_range_of_k_if_increasing_l300_300052


namespace plane_split_into_regions_l300_300257

theorem plane_split_into_regions :
  let L1 := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, 3*x)},
      L2 := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, x/3)} in
  (L1 ≠ L2) ∧ ∃ r : Finset (Finset (ℝ × ℝ)), 
    (∀ P : ℝ × ℝ, ∃ s ∈ r, P ∈ s) ∧ (∀ s ∈ r, ∀ t ∈ r, s ≠ t → s ∩ t = ∅) ∧ (r.card = 4).

end plane_split_into_regions_l300_300257


namespace trapezoid_area_correct_l300_300210

noncomputable def trapezoid_area (leg diag base : ℝ) : ℝ :=
  let h := real.sqrt (leg^2 - ((base/2)^2))
  (base + (base - 2 * real.sqrt (leg^2 - h^2))) * h / 2

theorem trapezoid_area_correct :
  let leg := 40
  let diag := 50
  let base := 60
  trapezoid_area leg diag base = 1263 :=
by {
  sorry
}

end trapezoid_area_correct_l300_300210


namespace campers_afternoon_l300_300945

theorem campers_afternoon:
  (A : ℕ) (H1 : 36 + A + 49 = 98) : A = 13 := 
by
  sorry

end campers_afternoon_l300_300945


namespace determinant_eq_l300_300810

-- Definitions based on the conditions in the problem statement:
variables {a b c d r s t : ℝ}

-- Conditions:
-- Suppose a, b, c, d are roots of the polynomial x^4 + rx^2 + sx + t = 0.
def is_root (x : ℝ) := x^4 + r * x^2 + s * x + t = 0
axiom ha : is_root a
axiom hb : is_root b
axiom hc : is_root c
axiom hd : is_root d

-- Define the matrix whose determinant we are interested in.
noncomputable def matrix_det : ℝ := 
  ![
    [1 + a, 1, 1, 1],
    [1, 1+ b, 1, 1],
    [1, 1, 1 + c, 1],
    [1, 1, 1, 1 + d]
  ].det

-- The theorem stating the relationship we want to prove:
theorem determinant_eq : matrix_det = r + s - t :=
sorry

end determinant_eq_l300_300810


namespace markers_last_group_correct_l300_300592

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end markers_last_group_correct_l300_300592


namespace probability_product_multiple_of_4_l300_300495

-- Define the set of integers from 5 to 25 inclusive
def S : Set ℤ := { n | 5 ≤ n ∧ n ≤ 25 }

-- Define the predicate for being a multiple of 4
def multiple_of_4 (n : ℤ) : Prop := n % 4 = 0

-- Define the probability problem
theorem probability_product_multiple_of_4 :
  let total_choices := (S.card choose 2)
  let favorable_choices := ((S.filter multiple_of_4).card choose 2)
  (favorable_choices : ℚ) / total_choices = 1 / 21 :=
  by
  sorry

end probability_product_multiple_of_4_l300_300495


namespace trajectory_midpoint_ellipse_foci_and_eccentricity_l300_300963

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

noncomputable def midpoint (P D M : ℝ × ℝ) : Prop :=
  P.1 = M.1 ∧ P.2 = 2 * M.2 ∧ M.1 = D.1 ∧ D.2 = 0

theorem trajectory_midpoint_ellipse (x₀ y₀ : ℝ) :
  (∃ x y : ℝ, point_on_circle x y ∧ midpoint (x, y) (x, 0) (x₀, y₀)) ↔
    x₀^2 / 4 + y₀^2 = 1 :=
sorry

theorem foci_and_eccentricity :
  (∃ (x₀ y₀ : ℝ) (h: x₀^2 / 4 + y₀^2 = 1), 
    (∃ (a : ℝ) (b : ℝ)
     (hfoci : (a = sqrt 3 ∧ b = 0) ∨ (a = -sqrt 3 ∧ b = 0)), 
    (∃ e : ℝ, e = sqrt 3 / 2))) :=
sorry

end trajectory_midpoint_ellipse_foci_and_eccentricity_l300_300963


namespace sum_a2_to_a20_l300_300308

-- Definitions translated from conditions
def seq_a : ℕ → ℕ 
| 0       := 1
| (n + 1) := if (n % 2) = 0 then 2 * (seq_a n) else (seq_a n) + 1

-- Theorem stating the required sum
theorem sum_a2_to_a20 : (∑ i in (range 10).map (fun i => seq_a (2 * (i + 1)))) = 4072 :=
by
  sorry

end sum_a2_to_a20_l300_300308


namespace jack_received_emails_in_the_morning_l300_300389

theorem jack_received_emails_in_the_morning
  (total_emails : ℕ)
  (afternoon_emails : ℕ)
  (morning_emails : ℕ) 
  (h1 : total_emails = 8)
  (h2 : afternoon_emails = 5)
  (h3 : total_emails = morning_emails + afternoon_emails) :
  morning_emails = 3 :=
  by
    -- proof omitted
    sorry

end jack_received_emails_in_the_morning_l300_300389


namespace population_decrease_is_25_percent_l300_300777

def initial_population : ℕ := 20000
def final_population_first_year : ℕ := initial_population + (initial_population * 25 / 100)
def final_population_second_year : ℕ := 18750

def percentage_decrease (initial final : ℕ) : ℚ :=
  ((initial - final : ℚ) * 100) / initial 

theorem population_decrease_is_25_percent :
  percentage_decrease final_population_first_year final_population_second_year = 25 :=
by
  sorry

end population_decrease_is_25_percent_l300_300777


namespace Jeffrey_steps_l300_300211

theorem Jeffrey_steps
  (Andrew_steps : ℕ) (Jeffrey_steps : ℕ) (h_ratio : Andrew_steps / Jeffrey_steps = 3 / 4)
  (h_Andrew : Andrew_steps = 150) :
  Jeffrey_steps = 200 :=
by
  sorry

end Jeffrey_steps_l300_300211


namespace vector_relation_circumcenter_orthocenter_l300_300039

variables {A B C O H : Type} [point A] [point B] [point C] [point O] [point H]

def is_circumcenter (O : Type) (A B C : Type) : Prop :=
  ∀ (P : Type), point P → (dist O P = dist O A ∧ dist O P = dist O B ∧ dist O P = dist O C)

def is_orthocenter (H : Type) (A B C : Type) : Prop :=
  ∀ (P : Type), point P → (line_perp H A B ∧ line_perp H B C ∧ line_perp H C A)

theorem vector_relation_circumcenter_orthocenter (hO : is_circumcenter O A B C) 
  (hH : is_orthocenter H A B C) : 
  vector (O, H) = vector (O, A) + vector (O, B) + vector (O, C) :=
sorry

end vector_relation_circumcenter_orthocenter_l300_300039


namespace six_times_six_l300_300565

-- Definitions based on the conditions
def pattern (n : ℕ) : ℕ := n * 6

-- Theorem statement to be proved
theorem six_times_six : pattern 6 = 36 :=
by {
  sorry
}

end six_times_six_l300_300565


namespace intersection_A_complementB_l300_300054

universe u

def R : Type := ℝ

def A (x : ℝ) : Prop := 0 < x ∧ x < 2

def B (x : ℝ) : Prop := x ≥ 1

def complement_B (x : ℝ) : Prop := x < 1

theorem intersection_A_complementB : 
  ∀ x : ℝ, (A x ∧ complement_B x) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end intersection_A_complementB_l300_300054


namespace trapezoid_area_l300_300775

variables {A B C D E : Type}

-- Assumptions about areas of triangles
variables (area_ABE area_ADE area_BCE area_trapezoid_ABCD : ℝ)

-- Conditions given in the problem
def given_conditions :=
  (area_ABE = 40) ∧
  (area_ADE = 30) ∧
  (area_BCE = 2 * area_ADE)

-- The main statement to be proven
theorem trapezoid_area (h : given_conditions) : area_trapezoid_ABCD = 160 :=
by
  sorry

end trapezoid_area_l300_300775


namespace bananas_left_after_eating_one_l300_300664

variable (Elias Bananas : Nat)

/-- Elias bought a dozen bananas, i.e., 12 bananas. -/
def dozen_bananas : Elias = 12 := sorry

/-- Elias eats 1 banana. -/
def eats_one_banana (h : Elias = 12) : Bananas = 11 :=
by
  rw [h]
  exact Nat.sub_self 1

theorem bananas_left_after_eating_one : Bananas = 11 := sorry

end bananas_left_after_eating_one_l300_300664


namespace sam_distance_l300_300459

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300459


namespace minimizeCostPerItem_l300_300979

noncomputable def productionCost (x : ℝ) : ℝ :=
  let preparationCost := 800
  let storageCostPerDayPerItem := 1
  let averageStorageTime := x / 8
  preparationCost + x * averageStorageTime * storageCostPerDayPerItem

noncomputable def averageCostPerItem (x : ℝ) : ℝ :=
  productionCost(x) / x

theorem minimizeCostPerItem : averageCostPerItem(80) = 20 := by
  sorry -- This is the spot for the actual proof which we are not required to provide.

end minimizeCostPerItem_l300_300979


namespace last_group_markers_l300_300589

theorem last_group_markers:
  ∀ (total_students group1_students group2_students markers_per_box boxes_of_markers group1_markers group2_markers : ℕ),
    total_students = 30 →
    group1_students = 10 →
    group2_students = 15 →
    markers_per_box = 5 →
    boxes_of_markers = 22 →
    group1_markers = 2 →
    group2_markers = 4 →
    let total_markers := boxes_of_markers * markers_per_box in
    let used_markers1 := group1_students * group1_markers in
    let used_markers2 := group2_students * group2_markers in
    let remaining_students := total_students - group1_students - group2_students in
    let remaining_markers := total_markers - used_markers1 - used_markers2 in
    remaining_students > 0 →
    remaining_markers % remaining_students = 0 →
    remaining_markers / remaining_students = 6 :=
sorry

end last_group_markers_l300_300589


namespace isosceles_triangle_option_a_l300_300864

theorem isosceles_triangle_option_a :
  (∀ (a b c : ℕ), a + b + c = 180) →
  (∀ (A : ℕ), (A = 40 ∧ B = 70 ∧ (∃ C : ℕ, A + B + C = 180) ∧ (B = C) → is_isosceles A B C)) :=
begin
  intro h,
  have h1 : 40 + 70 + 70 = 180 := by norm_num,
  use 70,
  split,
  { apply h1 },
  { exact eq.refl 70 },
end

end isosceles_triangle_option_a_l300_300864


namespace undefined_ratio_of_altitudes_l300_300005

theorem undefined_ratio_of_altitudes (
  (A B C D H : Point)
  (BC AC : ℝ)
  (angle_C : ℝ)
  (altitude_AD_intersects_orthocenter : altitude_intersects_orthocenter A D B C H)
  (BC_eq : BC = 5)
  (AC_eq : AC = 5 * Real.sqrt 2)
  (angle_C_eq : angle_C = 45)
  (AD_calculation : calc_AD A D C ≡ 5)
  (BD_eq : calc_BD B D C ≡ 0)
) : ratio_Undefined AH HD :=
sorry

end undefined_ratio_of_altitudes_l300_300005


namespace inequality_proof_l300_300313

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) :=
by
  sorry

end inequality_proof_l300_300313


namespace angle_A_equals_60_min_value_of_a_l300_300006

-- Define sides opposite the angles
variables {a b c : ℝ}

-- Define the vectors m and n
def m := (2 * b - c, a)
def n := (cos C, cos A)

-- Conditions
variables {A B C : ℝ}
variables (R : ℝ)
variables (h1 : m = (2 * b - c, a))
variables (h2 : n = (cos C, cos A))
variables (h3 : m.1 * n.2 = m.2 * n.1) -- m ∥ n

-- Problem (1): Prove that A = 60 degrees
theorem angle_A_equals_60 : A = 60 := 
sorry

-- New condition for problem (2)
variables (dot_product : ℝ) (h4 : dot_product = 4)

-- Problem (2): Prove that the minimum value of side a is 2 sqrt 2
theorem min_value_of_a : a = 2 * sqrt 2 := 
sorry

end angle_A_equals_60_min_value_of_a_l300_300006


namespace bob_pays_per_muffin_l300_300219

theorem bob_pays_per_muffin :
  let P := (84 * 1.5 - 63) / 84 in
  P = 0.75 := 
by
  sorry

end bob_pays_per_muffin_l300_300219


namespace investment_allocation_l300_300144

-- Define the constants for maximum potential profit and loss rates and constraints
def max_profit_rate_A := 1.0
def max_profit_rate_B := 0.5
def max_loss_rate_A := 0.3
def max_loss_rate_B := 0.1
def max_investment := 100000
def max_loss := 18000

-- Define the optimal investment amounts as provided by the solution
def optimal_investment_A := 40000
def optimal_investment_B := 60000

-- The proof problem
theorem investment_allocation :
  optimal_investment_A + optimal_investment_B ≤ max_investment ∧
  optimal_investment_A * max_loss_rate_A + optimal_investment_B * max_loss_rate_B ≤ max_loss ∧
  optimal_investment_A * max_profit_rate_A + optimal_investment_B * max_profit_rate_B ≥ ∀ x y,
    (x + y ≤ max_investment ∧ x * max_loss_rate_A + y * max_loss_rate_B ≤ max_loss) →
    (x * max_profit_rate_A + y * max_profit_rate_B) :=
by
  sorry

end investment_allocation_l300_300144


namespace chris_money_before_birthday_l300_300223

theorem chris_money_before_birthday :
  ∀ (total amount now : ℤ) (gift_from_grandmother : ℤ) (gift_from_aunt_uncle : ℤ) (gift_from_parents : ℤ),
    (total amount now = 279) →
    (gift_from_grandmother = 25) →
    (gift_from_aunt_uncle = 20) →
    (gift_from_parents = 75) →
    total amount now - (gift_from_grandmother + gift_from_aunt_uncle + gift_from_parents) = 159 :=
by
  intros total_amount_now gift_from_grandmother gift_from_aunt_uncle gift_from_parents
  assume h1 h2 h3 h4
  sorry

end chris_money_before_birthday_l300_300223


namespace balls_in_boxes_no_ball_in_box1_l300_300073

theorem balls_in_boxes_no_ball_in_box1 : 
  let balls := {A, B, C}
  let boxes := {1, 2, 3, 4}
  let valid_boxes := {2, 3, 4}
  (∀ b ∈ balls, ∃ b' ∈ valid_boxes, b = b') → finset.card (finset.product balls valid_boxes) = 27 :=
by
  sorry

end balls_in_boxes_no_ball_in_box1_l300_300073


namespace daxton_refill_percentage_l300_300620

theorem daxton_refill_percentage (capacity init_ratio empty_ratio final_volume : ℝ) 
  (h1: capacity = 8000) 
  (h2: init_ratio = 3 / 4) 
  (h3: empty_ratio = 0.4) 
  (h4: final_volume = 4680) : 
  let init_volume := init_ratio * capacity,
      empty_volume := empty_ratio * init_volume,
      remaining_volume := init_volume - empty_volume,
      added_volume := final_volume - remaining_volume,
      refill_percentage := (added_volume / remaining_volume) * 100 in
  refill_percentage = 30 := 
sorry

end daxton_refill_percentage_l300_300620


namespace final_temperature_is_correct_l300_300795

def initial_temperature : ℝ := 40
def after_jerry_temperature (T : ℝ) : ℝ := 2 * T
def after_dad_temperature (T : ℝ) : ℝ := T - 30
def after_mother_temperature (T : ℝ) : ℝ := T - 0.30 * T
def after_sister_temperature (T : ℝ) : ℝ := T + 24

theorem final_temperature_is_correct :
  after_sister_temperature (after_mother_temperature (after_dad_temperature (after_jerry_temperature initial_temperature))) = 59 :=
sorry

end final_temperature_is_correct_l300_300795


namespace product_of_consecutive_integers_between_sqrt_29_l300_300262

-- Define that \(5 \lt \sqrt{29} \lt 6\)
lemma sqrt_29_bounds : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 :=
sorry

-- Main theorem statement
theorem product_of_consecutive_integers_between_sqrt_29 :
  (∃ (a b : ℤ), 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 ∧ a = 5 ∧ b = 6 ∧ a * b = 30) := 
sorry

end product_of_consecutive_integers_between_sqrt_29_l300_300262


namespace power_comparison_l300_300553

theorem power_comparison :
  2 ^ 16 = 256 * 16 ^ 2 := 
by
  sorry

end power_comparison_l300_300553


namespace find_dot_position_after_transformations_l300_300616

/-- A square piece of paper has a dot in its top right corner. The square
is folded along its diagonal, then rotated 90 degrees clockwise about
its center, and then finally unfolded. Determine the resulting figure. -/
theorem find_dot_position_after_transformations :
  let square := matrix 2 2 (bool)
  let dot_position := (1, 2)
  let folded_position := (2, 1)
  let rotated_position := (1, 1)
  let unfolded_position := (1, 2)
  unfolded_position = (1, 2) := by
sorry

end find_dot_position_after_transformations_l300_300616


namespace find_composite_with_divisors_l300_300574

theorem find_composite_with_divisors : 
  ∃ n : ℕ, 
  (¬ n.prime ∧ (n > 1) ∧ (∀ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n → n - 12 ≥ d ∧ d ≥ n - 20)) 
  ↔ n = 24 :=
by sorry

end find_composite_with_divisors_l300_300574


namespace sum_of_x_coords_l300_300280

theorem sum_of_x_coords (x : ℝ) (y : ℝ) :
  y = abs (x^2 - 6*x + 8) ∧ y = 6 - x → (x = (5 + Real.sqrt 17) / 2 ∨ x = (5 - Real.sqrt 17) / 2 ∨ x = 2)
  →  ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) :=
by
  intros h1 h2
  have H : ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) := sorry
  exact H

end sum_of_x_coords_l300_300280


namespace parabola_circle_perpendicular_l300_300571

-- Define the conditions
def parabola (p : ℝ) : set (ℝ × ℝ) := { pt | ∃ x y, pt = (x, y) ∧ y^2 = 2 * p * x }
def circleP : set (ℝ × ℝ) := { pt | ∃ x y, pt = (x, y) ∧ (x - 3)^2 + y^2 = 8 }

-- Given the point M on the circle
def is_on_circle (M : ℝ × ℝ) : Prop := circleP M

-- Condition for perpendicularity
def perpendicular (A B : ℝ × ℝ) : Prop := (A.1 - focal_x A B) * (B.1 - focal_x A B) + (A.2 - 0) * (B.2 - 0) = -1

def focal_x (A B : ℝ × ℝ) : ℝ := (fst A + fst B) / 2

-- Define the proof problem
theorem parabola_circle_perpendicular (p : ℝ) (h_pos : 0 < p) 
  (E : set (ℝ × ℝ)) (h_E_def : ∀ (pt : ℝ × ℝ), E pt ↔ (∃ x y, pt = (x, y) ∧ y^2 = 2 * p * x)) 
  (P : set (ℝ × ℝ)) (h_P_def : ∀ (pt : ℝ × ℝ), P pt ↔ (∃ x y, pt = (x, y) ∧ (x - 3)^2 + y^2 = 8))
  (M : ℝ × ℝ) (h_M_on_circle : is_on_circle M) 
  (A B : ℝ × ℝ) (h_A_B_on_parabola : (A ∈ E) ∧ (B ∈ E)) 
  (h_l_intersects_E : ∃ l, l = (fst M, fst F) → (fst A ≤ fst M ∧ fst M ≤ fst B)) :
  perpendicular A B :=
by sorry

end parabola_circle_perpendicular_l300_300571


namespace area_light_gray_triangle_l300_300497

theorem area_light_gray_triangle 
  (area_dark_gray : ℝ)
  (segment1 : ℝ)
  (segment2 : ℝ)
  (h_area : area_dark_gray = 35)
  (h_segment1 : segment1 = 14)
  (h_segment2 : segment2 = 10) :
  let total_length := segment1 + segment2,
      height_dark_gray := (2 * area_dark_gray) / segment1,
      height_light_gray := (height_dark_gray / segment1) * total_length,
      area_light_gray := (1 / 2) * total_length * height_light_gray
  in
  area_light_gray = 144 := 
by
  sorry

end area_light_gray_triangle_l300_300497


namespace train_crossing_time_l300_300385

-- Define constants related to the problem
def length_of_train : ℝ := 90
def speed_km_hr : ℝ := 124
def speed_conversion_factor : ℝ := 1000 / 3600

-- Define the converted speed in m/s
def speed_m_s : ℝ := speed_km_hr * speed_conversion_factor

-- Define the time it takes to cross the pole
def crossing_time : ℝ := length_of_train / speed_m_s

-- Theorem stating that the calculated crossing time is approximately 2.61 seconds
theorem train_crossing_time :
  |crossing_time - 2.61| < 0.01 :=
sorry

end train_crossing_time_l300_300385


namespace twelfth_term_of_geometric_sequence_l300_300867

theorem twelfth_term_of_geometric_sequence 
  (a : ℕ → ℕ)
  (h₁ : a 4 = 4)
  (h₂ : a 7 = 32)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 12 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l300_300867


namespace original_cost_of_luxury_bag_l300_300960

theorem original_cost_of_luxury_bag (SP : ℝ) (profit_margin : ℝ) (original_cost : ℝ) 
  (h1 : SP = 3450) (h2 : profit_margin = 0.15) (h3 : SP = original_cost * (1 + profit_margin)) : 
  original_cost = 3000 :=
by
  sorry

end original_cost_of_luxury_bag_l300_300960


namespace number_is_seven_l300_300563

-- We will define the problem conditions and assert the answer
theorem number_is_seven (x : ℤ) (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by 
  -- Proof will be filled in here
  sorry

end number_is_seven_l300_300563


namespace cos_arcsin_l300_300246

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l300_300246


namespace coeff_x4_in_expansion_l300_300902

theorem coeff_x4_in_expansion (x : ℝ) :
  (coeff_x_n (expand (x + sqrt 5)^8 4)) = 1750 :=
by
  sorry

end coeff_x4_in_expansion_l300_300902


namespace sin_squared_minus_sin_cos_l300_300314

noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem sin_squared_minus_sin_cos (α : ℝ) 
  (h : (sin α + cos α) / (sin α - cos α) = 2) : sin α ^ 2 - sin α * cos α = 3 / 5 :=
by
  sorry

end sin_squared_minus_sin_cos_l300_300314


namespace exists_unique_n_digit_number_with_one_l300_300254

def n_digit_number (n : ℕ) : Type := {l : List ℕ // l.length = n ∧ ∀ x ∈ l, x = 1 ∨ x = 2 ∨ x = 3}

theorem exists_unique_n_digit_number_with_one (n : ℕ) (hn : n > 0) :
  ∃ x : n_digit_number n, x.val.count 1 = 1 ∧ ∀ y : n_digit_number n, y ≠ x → x.val.append [1] ≠ y.val.append [1] :=
sorry

end exists_unique_n_digit_number_with_one_l300_300254


namespace sam_drove_200_miles_l300_300432

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300432


namespace sum_of_18th_and_75th_l300_300518

def pattern : List ℕ := [1, 2, 5, 10, 25, 50, 100]

def pattern_at (n : ℕ) : ℕ :=
  pattern[(n % 7)]

theorem sum_of_18th_and_75th :
  pattern_at 18 + pattern_at 75 = 35 :=
by
  sorry

end sum_of_18th_and_75th_l300_300518


namespace average_speed_correct_l300_300216

noncomputable def average_speed (initial_odometer : ℝ) (lunch_odometer : ℝ) (final_odometer : ℝ) (total_time : ℝ) : ℝ :=
  (final_odometer - initial_odometer) / total_time

theorem average_speed_correct :
  average_speed 212.3 372 467.2 6.25 = 40.784 :=
by
  unfold average_speed
  sorry

end average_speed_correct_l300_300216


namespace problem_statement_l300_300697

def is_fibonacci : ℕ → Prop
| 0 => true
| 1 => true
| n => ∃ a b, is_fibonacci a ∧ is_fibonacci b ∧ n = a + b

def fibonacci (n : ℕ) : ℕ := 
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem problem_statement (a b : ℕ) :
  (a^2 + b^2 + 1) % (a * b) = 0 ↔ (a = 1 ∧ b = 1) ∨ (∃ n : ℕ, n ≥ 1 ∧ a = fibonacci (2 * n + 1) ∧ b = fibonacci (2 * n - 1)) :=
sorry

end problem_statement_l300_300697


namespace percent_of_x_l300_300930

-- The mathematical equivalent of the problem statement in Lean.
theorem percent_of_x (x : ℝ) (hx : 0 < x) : (x / 10 + x / 25) = 0.14 * x :=
by
  sorry

end percent_of_x_l300_300930


namespace proof_p_or_q_l300_300034

-- Define the conditions
def p_condition : Prop := (derivative (λ x : ℝ, 3 * x^2 + ln 3)) ≠ (λ x, 6 * x + 3)

def q_condition : Prop := 
  ∀ x, -3 < x ∧ x < 1 ↔ 
  (λ x : ℝ, derivative (λ x : ℝ, (3 - x^2) * exp x) > 0)

-- Prove that $p \lor q$ is true
theorem proof_p_or_q : (p_condition ∨ q_condition) = true :=
by
  -- Skipping the proof
  sorry

end proof_p_or_q_l300_300034


namespace midpoints_of_quadrilateral_form_parallelogram_l300_300834

/- Given a quadrilateral ABCD, with L and K being the midpoints of AD and BC respectively, 
and M and N being the midpoints of AC and BD respectively, 
we aim to prove that the quadrilateral formed by the points L, M, K, and N is a parallelogram. -/
theorem midpoints_of_quadrilateral_form_parallelogram
    (A B C D L K M N : Point) -- Declare points A, B, C, D, L, K, M, N
    (hL : midpoint A D L)     -- L is the midpoint of segment AD
    (hK : midpoint B C K)     -- K is the midpoint of segment BC
    (hM : midpoint A C M)     -- M is the midpoint of segment AC
    (hN : midpoint B D N)     -- N is the midpoint of segment BD :
    parallelogram L M K N :=  -- Conclusion: quadrilateral LMKN is a parallelogram
sorry

end midpoints_of_quadrilateral_form_parallelogram_l300_300834


namespace nate_total_run_l300_300474

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l300_300474


namespace sum_first_60_digits_of_1_div_9999_eq_15_l300_300552

theorem sum_first_60_digits_of_1_div_9999_eq_15 :
  let d := 1 / 9999 in
  let digits := (d.to_decimal 60).take 60 in
  digits.sum = 15 :=
by
  -- Lean code for expressing the decimal representation and summing the digits
  sorry

end sum_first_60_digits_of_1_div_9999_eq_15_l300_300552


namespace new_average_age_l300_300093

theorem new_average_age 
  (num_students : ℕ)
  (student_avg_age : ℚ)
  (teacher_age : ℚ)
  (new_avg_age : ℚ)
  (h_num_students : num_students = 30)
  (h_student_avg_age : student_avg_age = 14)
  (h_teacher_age : teacher_age = 45)
  (h_new_avg_age : new_avg_age = 15) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = new_avg_age :=
begin
  rw [h_num_students, h_student_avg_age, h_teacher_age, h_new_avg_age],
  norm_num,
  sorry
end

end new_average_age_l300_300093


namespace ratio_fifth_terms_l300_300133

-- Define the arithmetic sequences and their sums
variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}

-- Assume conditions of the problem
axiom sum_condition (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_condition2 (n : ℕ) : T n = n * (b 1 + b n) / 2
axiom ratio_condition : ∀ n, S n / T n = (2 * n - 3) / (3 * n - 2)

-- Prove the ratio of fifth terms a_5 / b_5
theorem ratio_fifth_terms : (a 5 : ℚ) / b 5 = 3 / 5 := by
  sorry

end ratio_fifth_terms_l300_300133


namespace solve_custom_operation_l300_300492

theorem solve_custom_operation (x : ℤ) (h : ((4 * 3 - (12 - x)) = 2)) : x = -2 :=
by
  sorry

end solve_custom_operation_l300_300492


namespace find_value_of_c_l300_300564

-- Mathematical proof problem in Lean 4 statement
theorem find_value_of_c (a b c d : ℝ)
  (h1 : a + c = 900)
  (h2 : b + c = 1100)
  (h3 : a + d = 700)
  (h4 : a + b + c + d = 2000) : 
  c = 200 :=
sorry

end find_value_of_c_l300_300564


namespace negation_proposition_l300_300106

theorem negation_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 ≥ 0) := 
by
  sorry

end negation_proposition_l300_300106


namespace line_circle_intersection_probability_l300_300836

theorem line_circle_intersection_probability :
  ∃ (b : ℝ) (h : b ∈ Set.Icc (-3 : ℝ) 3),
    let center : ℝ × ℝ := (0, 1)
    let radius := Real.sqrt 2
    let distance_from_center_to_line := λ b, (| b - 1 | / Real.sqrt 2)
    let intersection_condition := ∀ b, distance_from_center_to_line b ≤ radius
    let probability_interval := Set.Icc (-3 : ℝ) 3
    let intersection_probability := (3 + 1) / (3 + 3)
    intersection_condition b = true → intersection_probability = 2 / 3 :=
sorry

end line_circle_intersection_probability_l300_300836


namespace sequence_sum_mod_10_l300_300655

theorem sequence_sum_mod_10 (a d : ℕ) (n : ℕ) (hn : n < 10) :
  (a = 2) → (d = 5) →
  (∃ k, a + (k - 1) * d = 97 ∧ (2 + 7 + 12 + ... + 97) % 10 = n) :=
by {
  intros h1 h2,
  existsi 20,
  -- Steps to show the sequence sum modulo 10 will result in 0 
  have : (2 + 7 + 12 + ... + 97) % 10 = 0,
  sorry,
}

end sequence_sum_mod_10_l300_300655


namespace integer_root_abs_sum_l300_300694

noncomputable def solve_abs_sum (p q r : ℤ) : ℤ := |p| + |q| + |r|

theorem integer_root_abs_sum (p q r m : ℤ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2024)
  (h3 : ∃ m, ∀ x, x^3 - 2024 * x + m = (x - p) * (x - q) * (x - r)) :
  solve_abs_sum p q r = 104 :=
by sorry

end integer_root_abs_sum_l300_300694


namespace verify_rs_correct_l300_300808

noncomputable def verify_rs : Prop :=
  let N := ![
    ![3, 4],
    ![-2, 1]
  ]
  let I : Matrix (Fin 2) (Fin 2) ℤ := 1
  let N2 := N ⬝ N
  ∃ r s : ℤ, N2 = r • N + s • I ∧ (r = 4 ∧ s = -11)

theorem verify_rs_correct : verify_rs := sorry

end verify_rs_correct_l300_300808


namespace katy_summer_reading_l300_300016

theorem katy_summer_reading :
  let b_June := 8 in
  let b_July := 2 * b_June in
  let b_August := b_July - 3 in
  b_June + b_July + b_August = 37 :=
by
  sorry

end katy_summer_reading_l300_300016


namespace distinct_walls_count_l300_300024

theorem distinct_walls_count (n : ℕ) (h : n > 0) : 
  (number_of_distinct_walls n) = 2^(n-1) :=
sorry

-- additional definition to model "number_of_distinct_walls"
noncomputable def number_of_distinct_walls (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∑ k in finset.range n, nat.choose (n - 1) k

end distinct_walls_count_l300_300024


namespace baker_earnings_in_april_l300_300661

theorem baker_earnings_in_april :
  let price_cakes := 12
  let quantity_cakes := 453
  let price_pies := 7
  let quantity_pies := 126
  let price_bread := 3.5
  let quantity_bread := 95
  let price_cookies := 1.5
  let quantity_cookies := 320
  let discount_pies := 0.10
  let sales_tax := 0.05
  let income_cakes := price_cakes * quantity_cakes
  let income_pies := price_pies * quantity_pies
  let discounted_income_pies := income_pies * (1 - discount_pies)
  let income_bread := price_bread * quantity_bread
  let income_cookies := price_cookies * quantity_cookies
  let total_income_without_tax := income_cakes + discounted_income_pies + income_bread + income_cookies
  let total_income := total_income_without_tax * (1 + sales_tax)
  total_income ≈ 7394.42 :=
by sorry

end baker_earnings_in_april_l300_300661


namespace car_returns_to_start_after_5_operations_l300_300187

theorem car_returns_to_start_after_5_operations (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : α = 72 ∨ α = 144 :=
sorry

end car_returns_to_start_after_5_operations_l300_300187


namespace magic8ball_prob_l300_300014

theorem magic8ball_prob (q p n : ℕ) : 
  (∀ i : ℕ, i ∈ ufin q → (p + n + (q - p - n)) = q) →
  (p = 3) →
  (n = 2) →
  ∃ prob : ℚ, prob = (70 / 243) :=
begin
  assume h1 h2 h3,
  use (70 / 243),
  sorry
end

end magic8ball_prob_l300_300014


namespace intersection_A_B_l300_300740

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | ∃ y : ℝ, y = x^2 + 2}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ ∃ y : ℝ, y = x^2 + 2} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end intersection_A_B_l300_300740


namespace optimal_road_trip_time_l300_300472

theorem optimal_road_trip_time 
  (n : ℕ) (w : ℕ) (car_capacity : ℕ) (car_speed_factor : ℕ) : 
  n = 12 → w = 2 → car_capacity = 4 → car_speed_factor = 15 → 
  ∃ t : ℝ, t = 30.4 :=
by {
  intros,
  sorry
}

end optimal_road_trip_time_l300_300472


namespace circle_C_standard_eq_circle_D_cartesian_eq_circles_externally_tangent_l300_300780

noncomputable def circle_C_parametric (α : ℝ) : (ℝ × ℝ) :=
  (3 + 4 * Real.cos α, -2 + 4 * Real.sin α)

noncomputable def circle_D_polar (ρ θ : ℝ) : Bool :=
  ρ^2 - 12*ρ*Real.cos θ - 4*ρ*Real.sin θ = -39

theorem circle_C_standard_eq : ∀ (x y : ℝ), (∃ α : ℝ, (x, y) = circle_C_parametric α) ↔ (x - 3)^2 + (y + 2)^2 = 16 :=
by
  sorry

theorem circle_D_cartesian_eq : ∀ (x y ρ θ : ℝ),
  ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x ∧ circle_D_polar ρ θ ↔ (x - 6)^2 + (y - 2)^2 = 1 :=
by
  sorry

theorem circles_externally_tangent :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁ - 3)^2 + (y₁ + 2)^2 = 16 ∧ (x₂ - 6)^2 + (y₂ - 2)^2 = 1 →
  Real.sqrt ((3 - 6)^2 + (-2 - 2)^2) = 5 :=
by
  sorry

end circle_C_standard_eq_circle_D_cartesian_eq_circles_externally_tangent_l300_300780


namespace customer_total_payment_l300_300633

structure PaymentData where
  rate : ℕ
  discount1 : ℕ
  lateFee1 : ℕ
  discount2 : ℕ
  lateFee2 : ℕ
  discount3 : ℕ
  lateFee3 : ℕ
  discount4 : ℕ
  lateFee4 : ℕ
  onTime1 : Bool
  onTime2 : Bool
  onTime3 : Bool
  onTime4 : Bool

noncomputable def monthlyPayment (rate discount late_fee : ℕ) (onTime : Bool) : ℕ :=
  if onTime then rate - (rate * discount / 100) else rate + (rate * late_fee / 100)

theorem customer_total_payment (data : PaymentData) : 
  monthlyPayment data.rate data.discount1 data.lateFee1 data.onTime1 +
  monthlyPayment data.rate data.discount2 data.lateFee2 data.onTime2 +
  monthlyPayment data.rate data.discount3 data.lateFee3 data.onTime3 +
  monthlyPayment data.rate data.discount4 data.lateFee4 data.onTime4 = 195 := by
  sorry

end customer_total_payment_l300_300633


namespace find_n_l300_300570

noncomputable def n (n : ℕ) : Prop :=
  lcm n 12 = 42 ∧ gcd n 12 = 6

theorem find_n (n : ℕ) (h : lcm n 12 = 42) (h1 : gcd n 12 = 6) : n = 21 :=
by sorry

end find_n_l300_300570


namespace find_number_of_pourings_l300_300958

-- Define the sequence of remaining water after each pouring
def remaining_water (n : ℕ) : ℚ :=
  (2 : ℚ) / (n + 2)

-- The main theorem statement
theorem find_number_of_pourings :
  ∃ n : ℕ, remaining_water n = 1 / 8 :=
by
  sorry

end find_number_of_pourings_l300_300958


namespace tiling_mod_1000_l300_300169

def num_ways_to_tile (n : ℕ) (tile_colors : ℕ) : ℕ := 
  ∑ i in (finset.range (n+1)).filter (λ i, i ≥ 3), 
    nat.choose (n-1) (i-1) * (tile_colors ^ i - 3 * (2 ^ i) + 3)

theorem tiling_mod_1000 :
  (num_ways_to_tile 9 3) % 1000 = 663 :=
by
  -- Proof omitted
  sorry

end tiling_mod_1000_l300_300169


namespace edges_perpendicular_l300_300075

variables {V : Type*} [InnerProductSpace ℝ V] -- assuming an inner product space over reals
variables (r1 r2 r3 r4 : V)

-- Given conditions translated
def opposite_edges_sum_squares_eq : Prop :=
  (∥r1 - r4∥^2 + ∥r3 - r2∥^2 = ∥r2 - r4∥^2 + ∥r3 - r1∥^2) ∧
  (∥r2 - r4∥^2 + ∥r3 - r1∥^2 = ∥r3 - r4∥^2 + ∥r2 - r1∥^2)

-- The theorem to prove
theorem edges_perpendicular
  (h : opposite_edges_sum_squares_eq r1 r2 r3 r4) :
  ⟪r1, r3⟫ = 0 ∧ ⟪r1, r2⟫ = 0 ∧ ⟪r2, r3⟫ = 0 :=
begin
  sorry
end

end edges_perpendicular_l300_300075


namespace sam_drove_200_miles_l300_300463

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300463


namespace towel_percentage_decrease_l300_300977

theorem towel_percentage_decrease (L B : ℝ) (hL: L > 0) (hB: B > 0) :
  let OriginalArea := L * B
  let NewLength := 0.8 * L
  let NewBreadth := 0.8 * B
  let NewArea := NewLength * NewBreadth
  let PercentageDecrease := ((OriginalArea - NewArea) / OriginalArea) * 100
  PercentageDecrease = 36 :=
by
  sorry

end towel_percentage_decrease_l300_300977


namespace optimal_post_office_location_l300_300973

-- The conditions
variables {n : ℕ} -- Number of houses
variables {x : Fin n → ℝ} -- Coordinates of the houses on a number line

-- The functions to capture the total walking distance of the postman
noncomputable def total_distance (t : ℝ) : ℝ :=
  ∑ i, (abs (x i - t) + abs (x i - t))

-- Proving the optimal location of the post office
theorem optimal_post_office_location (h : ∀ i j, i < j → x i < x j) :
  if odd n then
    ∃ t, t = x (Fin.ofNat (n / 2 + 1)) ∧ ∀ u : ℝ, total_distance t ≤ total_distance u
  else
    ∃ t, t ∈ Icc (x (Fin.ofNat (n / 2 - 1))) (x (Fin.ofNat (n / 2))) ∧ ∀ u : ℝ, total_distance t ≤ total_distance u :=
sorry

end optimal_post_office_location_l300_300973


namespace gcd_repeated_integer_l300_300196

open Nat

theorem gcd_repeated_integer (m : ℕ) (h1 : 100 ≤ m) (h2 : m ≤ 999) : 
  gcd (1001001 * m) (1001001 * (m + 1)) = 1001001 :=
by
  sorry

end gcd_repeated_integer_l300_300196


namespace boat_travel_distance_downstream_l300_300582

def boat_speed : ℝ := 22 -- Speed of boat in still water in km/hr
def stream_speed : ℝ := 5 -- Speed of the stream in km/hr
def time_downstream : ℝ := 7 -- Time taken to travel downstream in hours
def effective_speed_downstream : ℝ := boat_speed + stream_speed -- Effective speed downstream

theorem boat_travel_distance_downstream : effective_speed_downstream * time_downstream = 189 := by
  -- Since effective_speed_downstream = 27 (22 + 5)
  -- Distance = Speed * Time
  -- Hence, Distance = 27 km/hr * 7 hours = 189 km
  sorry

end boat_travel_distance_downstream_l300_300582


namespace length_of_AE_l300_300026

-- Problem conditions and theorem statement
theorem length_of_AE (B C D M A E : Point) (h1 : divides B C D A E) (h2 : midpoint M A E) (h3 : dist M C = 12) :
  dist A E = 48 :=
by sorry

end length_of_AE_l300_300026


namespace frog_jump_paths_l300_300182

noncomputable def φ : ℕ × ℕ → ℕ
| (0, 0) => 1
| (x, y) =>
  let φ_x1 := if x > 1 then φ (x - 1, y) else 0
  let φ_x2 := if x > 1 then φ (x - 2, y) else 0
  let φ_y1 := if y > 1 then φ (x, y - 1) else 0
  let φ_y2 := if y > 1 then φ (x, y - 2) else 0
  φ_x1 + φ_x2 + φ_y1 + φ_y2

theorem frog_jump_paths : φ (4, 4) = 556 := sorry

end frog_jump_paths_l300_300182


namespace find_number_l300_300143

theorem find_number : ∃ n : ℕ, ∃ q : ℕ, ∃ r : ℕ, q = 6 ∧ r = 4 ∧ n = 9 * q + r ∧ n = 58 :=
by
  sorry

end find_number_l300_300143


namespace grandfather_grandchildren_l300_300533

theorem grandfather_grandchildren (children : Finset ℕ) (grandfather_of : ℕ → Finset ℕ) :
  children.card = 20 →
  (∀ x y, x ∈ children → y ∈ children → (grandfather_of x ∩ grandfather_of y).Nonempty) →
  (∃ g, (children.filter (λ c, g ∈ grandfather_of c)).card ≥ 14) :=
by
  intro h_card h_common_grandfather
  sorry

end grandfather_grandchildren_l300_300533


namespace SamDrove200Miles_l300_300445

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300445


namespace tee_shirts_with_60_feet_of_material_l300_300387

def tee_shirts (f t : ℕ) : ℕ := t / f

theorem tee_shirts_with_60_feet_of_material :
  tee_shirts 4 60 = 15 :=
by
  sorry

end tee_shirts_with_60_feet_of_material_l300_300387


namespace sam_distance_traveled_l300_300453

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300453


namespace part1_decreasing_on_pos_part2_t_range_l300_300734

noncomputable def f (x : ℝ) : ℝ := -x + 2 / x

theorem part1_decreasing_on_pos (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) : 
  f x1 > f x2 := by sorry

theorem part2_t_range (t : ℝ) (ht : ∀ x : ℝ, 1 ≤ x → f x ≤ (1 + t * x) / x) : 
  0 ≤ t := by sorry

end part1_decreasing_on_pos_part2_t_range_l300_300734


namespace number_of_connections_l300_300890

theorem number_of_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 :=
by
  sorry

end number_of_connections_l300_300890


namespace side_view_area_l300_300310

-- Given conditions of the triangular prism
def lateral_edge_length : ℝ := 2
def base_side_length : ℝ := 2
def main_view_side_length : ℝ := 2
def equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

-- Conditions described as properties
axiom height_perpendicular_to_base_plane : ∀ (A A1 B1 C1 : ℝ), ∃ (h : ℝ), A1 - A = h ∧ A1 - B1 = 0 ∧ A1 - C1 = 0 ∧ h = 2
axiom main_view_is_square : main_view_side_length = 2

-- Lean 4 statement for the proof problem
theorem side_view_area : 
  let base_triangle := equilateral_triangle base_side_length base_side_length base_side_length 
  in main_view_is_square → lateral_edge_length = 2 → base_side_length = 2 → 
  ∃ (area : ℝ), area = 4 :=
by
  sorry

end side_view_area_l300_300310


namespace reflection_matrix_squared_is_identity_l300_300402

def reflection_matrix (u : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := 
  λ v, let a := u.1
           b := u.2
           c := (a * v.1 + b * v.2) / (a^2 + b^2)
       in (2 * a * c - v.1, 2 * b * c - v.2)

theorem reflection_matrix_squared_is_identity :
  let S := reflection_matrix (4, -1)
  ∀ v : ℝ × ℝ, S (S v) = v := sorry

end reflection_matrix_squared_is_identity_l300_300402


namespace length_of_hypotenuse_l300_300968

/-- Define the problem's parameters -/
def perimeter : ℝ := 34
def area : ℝ := 24
def length_hypotenuse (a b c : ℝ) : Prop := a + b + c = perimeter 
  ∧ (1/2) * a * b = area
  ∧ a^2 + b^2 = c^2

/- Lean statement for the proof problem -/
theorem length_of_hypotenuse (a b c : ℝ) 
  (h1: a + b + c = 34)
  (h2: (1/2) * a * b = 24)
  (h3: a^2 + b^2 = c^2)
  : c = 62 / 4 := sorry

end length_of_hypotenuse_l300_300968


namespace sally_total_revenue_l300_300839

def week1_revenue : ℝ := 20 * 1

def week2_revenue : ℝ := (20 + 0.50 * 20) * 1.25

def week3_revenue : ℝ := (20 + 0.75 * 20) * 1.50

def total_revenue : ℝ := week1_revenue + week2_revenue + week3_revenue

theorem sally_total_revenue : total_revenue = 110 := by
  calc
    total_revenue = week1_revenue + week2_revenue + week3_revenue := rfl
    ... = 20 * 1 + (20 + 0.5 * 20) * 1.25 + (20 + 0.75 * 20) * 1.50 := rfl
    ... = 20 + (20 + 10) * 1.25 + (20 + 15) * 1.50 := rfl
    ... = 20 + 30 * 1.25 + 35 * 1.50 := rfl
    ... = 20 + 37.5 + 52.5 := rfl
    ... = 110 := rfl

end sally_total_revenue_l300_300839


namespace codger_feet_l300_300647

theorem codger_feet (F : ℕ) (h1 : 6 = 2 * (5 - 1) * F) : F = 3 := by
  sorry

end codger_feet_l300_300647


namespace cos_4_theta_l300_300349

noncomputable def sum_infinite_series_cos_squared (θ : ℝ) : ℝ := ∑' n, (cos θ)^(2 * n)

theorem cos_4_theta (θ : ℝ) (h : sum_infinite_series_cos_squared θ = 3) : cos (4 * θ) = -7/9 := 
by 
  sorry

end cos_4_theta_l300_300349


namespace buttermilk_biscuit_cost_l300_300844

-- defining constants and conditions
def quiche_price : ℝ := 15.0
def croissant_price : ℝ := 3.0
def num_quiches : ℕ := 2
def num_croissants : ℕ := 6
def num_biscuits : ℕ := 6
def total_after_discount : ℝ := 54.0
def discount : ℝ := 0.10
def total_cost := ((num_quiches * quiche_price) + (num_croissants * croissant_price) + (num_biscuits * biscuit_price)) * (1 - discount)

-- Prove that each buttermilk biscuit costs $2.00 given the conditions.
theorem buttermilk_biscuit_cost
  (biscuit_price : ℝ)
  (h1 : num_quiches * quiche_price = 30.0)
  (h2 : num_croissants * croissant_price = 18.0)
  (h3 : total_after_discount = 54.0)
  (h4 : ((num_quiches * quiche_price) + (num_croissants * croissant_price) + (num_biscuits * biscuit_price)) * (1 - discount) = total_after_discount) :
  biscuit_price = 2.0 :=
sorry

end buttermilk_biscuit_cost_l300_300844


namespace meters_to_examine_10000_l300_300631

def projection_for_sample (total_meters_examined : ℕ) (rejection_rate : ℝ) (sample_size : ℕ) :=
  total_meters_examined = sample_size

theorem meters_to_examine_10000 : 
  projection_for_sample 10000 0.015 10000 := by
  sorry

end meters_to_examine_10000_l300_300631


namespace gcd_8917_4273_l300_300539

theorem gcd_8917_4273 : Int.gcd 8917 4273 = 1 :=
by
  sorry

end gcd_8917_4273_l300_300539


namespace sufficient_condition_for_odd_power_function_l300_300337

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

noncomputable def power_function (m n : ℤ) : ℝ → ℝ := 
  λ x, x ^ ((m : ℝ) / (n : ℝ))

theorem sufficient_condition_for_odd_power_function :
  is_odd_function (power_function 1 3) :=
by
  sorry

end sufficient_condition_for_odd_power_function_l300_300337


namespace gcd_of_g_and_y_l300_300721

-- Define the function g(y)
def g (y : ℕ) := (3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)

-- Define that y is a multiple of 45678
def isMultipleOf (y divisor : ℕ) : Prop := ∃ k, y = k * divisor

-- Define the proof problem
theorem gcd_of_g_and_y (y : ℕ) (h : isMultipleOf y 45678) : Nat.gcd (g y) y = 1512 :=
by
  sorry

end gcd_of_g_and_y_l300_300721


namespace coins_in_boxes_l300_300373

theorem coins_in_boxes : ∃ n, n = (Nat.choose 7 2) ∧ n = 21 := 
by 
  use Nat.choose 7 2
  split
  · rfl
  · sorry

end coins_in_boxes_l300_300373


namespace percentage_decrease_l300_300948

variables (P : ℝ) (x : ℝ)

theorem percentage_decrease
  (h1 : P > 0)
  (h2 : 1.12 * P = 1.60 * P - (x / 100) * (1.60 * P))
  (h3 : P * 0.12 ≈ 0.12 * P) :
  x = 30 :=
by
  sorry

end percentage_decrease_l300_300948


namespace false_equilateral_triangle_congruence_l300_300924

theorem false_equilateral_triangle_congruence :
  ¬ (∀ (Δ1 Δ2 : Triangle), equilateral Δ1 ∧ equilateral Δ2 → congruent Δ1 Δ2) := by
  sorry

end false_equilateral_triangle_congruence_l300_300924


namespace cos_arcsin_l300_300230

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l300_300230


namespace range_of_a_l300_300053

noncomputable def f (a x : ℝ) : ℝ := x^2 - 3 * x + a

theorem range_of_a {a : ℝ} (h : ∃ x ∈ Ioo (1 : ℝ) 3, f a x = 0) : a ∈ Ioo 0 (9 / 4) ∨ a = 9 / 4 :=
begin
  sorry
end

end range_of_a_l300_300053


namespace S_2001_eq_2S_2000_add_1_l300_300285

noncomputable def sequence_S (n : ℕ) : ℕ :=
  if h : n > 1 then
    let radical_chain : ℝ → ℝ :=
      nat.rec_on (n-1) (λ _, real.sqrt 2)
                  (λ _ f, real.sqrt (2 + f 0)) in
    nat.floor ((2 : ℝ) ^ n * radical_chain 0)
  else 0

theorem S_2001_eq_2S_2000_add_1 : 
  sequence_S 2001 = 2 * sequence_S 2000 + 1 := 
sorry

end S_2001_eq_2S_2000_add_1_l300_300285


namespace cos_arcsin_l300_300228

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l300_300228


namespace solve_equation_sin_cos_l300_300851

theorem solve_equation_sin_cos (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) →
  (sin y ≠ 0) →
  (sin^2 x + 1 / sin^2 x)^3 + (sin^2 y + 1 / sin^2 y)^3 = 16 * cos z →
  x = (π / 2) + π * n ∧ y = (π / 2) + π * k ∧ z = 2 * π * m :=
by
  intro h1 h2 heq
  sorry

end solve_equation_sin_cos_l300_300851


namespace bottom_row_product_is_1232_l300_300674

-- Define a 4x4 matrix of natural numbers from 0 to 15
def table : Matrix (Fin 4) (Fin 4) ℕ := ![
  ![0, 1, 2, 3],
  ![4, 5, 6, 7],
  ![8, 9, 10, 11],
  ![12, 13, 14, 15]
]

-- Define the product of the bottom row elements
def bottom_row_product := table 3 0 * table 3 1 * table 3 2 * table 3 3

theorem bottom_row_product_is_1232 : bottom_row_product = 1232 := by
  have h : bottom_row_product = 12 * 13 * 14 * 15 := by
    simp [bottom_row_product, table]
  calc
    bottom_row_product = 12 * 13 * 14 * 15 := h
    ... = 1232 := by norm_num

end bottom_row_product_is_1232_l300_300674


namespace number_of_basketball_cards_l300_300217

theorem number_of_basketball_cards 
  (B : ℕ) -- Number of basketball cards in each box
  (H1 : 4 * B = 40) -- Given condition from equation 4B = 40
  
  (H2 : 4 * B + 40 - 58 = 22) -- Given condition from the total number of cards

: B = 10 := 
by 
  sorry

end number_of_basketball_cards_l300_300217


namespace sum_of_first_six_terms_arithmetic_seq_l300_300102

variables (a_n : ℕ → ℤ) (a d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (a d : ℤ) : Prop :=
  ∀ n, a_n n = a + n * d

def is_geometric_sequence (a_2 a_3 a_6 : ℤ) : Prop :=
  a_3 * a_3 = a_2 * a_6

theorem sum_of_first_six_terms_arithmetic_seq :
  is_arithmetic_sequence a_n 1 d →
  d ≠ 0 →
  is_geometric_sequence (a_n 1) (a_n 2) (a_n 5) →
  (∑ i in finset.range 6, a_n i) = -24 :=
by
  intro h_arith h_d_ne_zero h_geo
  -- Proof ommited
  sorry

end sum_of_first_six_terms_arithmetic_seq_l300_300102


namespace internal_angle_bisector_length_l300_300573

-- Define the problem in Lean 4
theorem internal_angle_bisector_length (a b c1 c2 f_c : ℝ) 
  (h1 : c1 + c2 = side_length_c) 
  (h2 : f_c = sqrt (a * b - c1 * c2)) : 
  f_c = sqrt (a * b - c1 * c2) :=
by
  sorry

end internal_angle_bisector_length_l300_300573


namespace ratio_a_c_l300_300877

variables (a b c d : ℚ)

axiom ratio_a_b : a / b = 5 / 4
axiom ratio_c_d : c / d = 4 / 3
axiom ratio_d_b : d / b = 1 / 8

theorem ratio_a_c : a / c = 15 / 2 :=
by sorry

end ratio_a_c_l300_300877


namespace perpendicular_tangents_at_x0_l300_300358

noncomputable def x0 := (36 : ℝ)^(1 / 3) / 6

theorem perpendicular_tangents_at_x0 :
  (∃ x0 : ℝ, (∃ f1 f2 : ℝ → ℝ,
    (∀ x, f1 x = x^2 - 1) ∧
    (∀ x, f2 x = 1 - x^3) ∧
    (2 * x0 * (-3 * x0^2) = -1)) ∧
    x0 = (36 : ℝ)^(1 / 3) / 6) := sorry

end perpendicular_tangents_at_x0_l300_300358


namespace rectangle_perimeter_l300_300610

theorem rectangle_perimeter (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ)
  (h1 : a1 + a2 = a3)
  (h2 : a1 + a3 = a4)
  (h3 : a3 + a4 = a5)
  (h4 : a4 + a5 = a6)
  (h5 : a2 + a3 + a5 = a7)
  (h6 : a2 + a7 = a8)
  (h7 : a1 + a4 + a6 = a9)
  (h8 : a6 + a9 = a7 + a8)
  (hp : Nat.gcd a6 a7 = 1)
  : 2 * (a6 + a7) = 260 :=
begin
  sorry
end

end rectangle_perimeter_l300_300610


namespace jade_handled_80_transactions_l300_300155

variable (mabel anthony cal jade : ℕ)

-- Conditions
def mabel_transactions : mabel = 90 :=
by sorry

def anthony_transactions : anthony = mabel + (10 * mabel / 100) :=
by sorry

def cal_transactions : cal = 2 * anthony / 3 :=
by sorry

def jade_transactions : jade = cal + 14 :=
by sorry

-- Proof problem
theorem jade_handled_80_transactions :
  mabel = 90 →
  anthony = mabel + (10 * mabel / 100) →
  cal = 2 * anthony / 3 →
  jade = cal + 14 →
  jade = 80 :=
by
  intros
  subst_vars
  -- The proof steps would normally go here, but we leave it with sorry.
  sorry

end jade_handled_80_transactions_l300_300155


namespace find_ab_l300_300566

theorem find_ab (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 29) : a * b = 10 :=
sorry

end find_ab_l300_300566


namespace product_decrease_increase_fifteenfold_l300_300786

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end product_decrease_increase_fifteenfold_l300_300786


namespace even_increasing_function_inequality_l300_300162

theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ {x₁ x₂ : ℝ}, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_increasing_function_inequality_l300_300162


namespace remainder_when_divided_by_x_minus_xi_is_evaluation_l300_300537

variable {R : Type*} [CommRing R]

-- Definition of a polynomial
variable {f : Polynomial R}

-- Definition for x_i
variable {x_i : R}

theorem remainder_when_divided_by_x_minus_xi_is_evaluation :
  Polynomial.eval x_i f = Polynomial.divByXSubC f x_i := sorry

end remainder_when_divided_by_x_minus_xi_is_evaluation_l300_300537


namespace largest_two_digit_prime_factor_of_binom_180_90_l300_300543

-- Definitions for the conditions
def binom (n k : ℕ) := n.choose k
def n : ℕ := binom 180 90

-- The prime factor we are considering
def is_prime (p : ℕ) : Prop := Nat.Prime p
def two_digit_prime (p : ℕ) : Prop := 10 ≤ p ∧ p < 100 ∧ is_prime p

-- The statement to be proved
theorem largest_two_digit_prime_factor_of_binom_180_90 :
  ∃ p, two_digit_prime p ∧ p ∣ n ∧ ∀ q, two_digit_prime q ∧ q ∣ n → q ≤ p :=
begin
  sorry
end

end largest_two_digit_prime_factor_of_binom_180_90_l300_300543


namespace max_equilateral_triangles_of_six_lines_l300_300544

noncomputable def max_equilateral_triangles (lines: Finset (Set ℝ)) : ℕ :=
  if h : lines.card = 6 
  then 8 
  else 0

theorem max_equilateral_triangles_of_six_lines (lines: Finset (Set ℝ)) (h : lines.card = 6) : 
  max_equilateral_triangles lines = 8 := 
by 
  simp [max_equilateral_triangles, h]
  sorry

end max_equilateral_triangles_of_six_lines_l300_300544


namespace sequence_of_8_numbers_l300_300368

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l300_300368


namespace minimum_a1_a2_sum_l300_300523

theorem minimum_a1_a2_sum (a : ℕ → ℕ)
  (h : ∀ n ≥ 1, a (n + 2) = (a n + 2017) / (1 + a (n + 1)))
  (positive_terms : ∀ n, a n > 0) :
  a 1 + a 2 = 2018 :=
sorry

end minimum_a1_a2_sum_l300_300523


namespace prove_quadrilateral_is_rectangle_l300_300096

def quadrilateral_is_rectangle {z1 z2 z3 z4 : ℂ} (h1 : ∥z1∥ = 1) (h2 : ∥z2∥ = 1) (h3 : ∥z3∥ = 1) (h4 : ∥z4∥ = 1) (h_sum : z1 + z2 + z3 + z4 = 0) : Prop :=
  let vertices := [z1, z2, z3, z4] in
  ∀ (M N : ℂ), M ∈ vertices ∧ N ∈ vertices → is_rectangle {v | v ∈ vertices}

theorem prove_quadrilateral_is_rectangle {z1 z2 z3 z4 : ℂ} 
  (h1 : ∥z1∥ = 1) 
  (h2 : ∥z2∥ = 1) 
  (h3 : ∥z3∥ = 1) 
  (h4 : ∥z4∥ = 1) 
  (h_sum : z1 + z2 + z3 + z4 = 0) : 
  quadrilateral_is_rectangle h1 h2 h3 h4 h_sum :=
sorry -- proof goes here

end prove_quadrilateral_is_rectangle_l300_300096


namespace schur_theorem_l300_300942

theorem schur_theorem {n : ℕ} (P : Fin n → Set ℕ) (h_partition : ∀ x : ℕ, ∃ i : Fin n, x ∈ P i) :
  ∃ (i : Fin n) (x y : ℕ), x ∈ P i ∧ y ∈ P i ∧ x + y ∈ P i :=
sorry

end schur_theorem_l300_300942


namespace gum_pieces_in_each_packet_l300_300471

theorem gum_pieces_in_each_packet
  (packets : ℕ) (chewed_pieces : ℕ) (remaining_pieces : ℕ) (total_pieces : ℕ)
  (h1 : packets = 8) (h2 : chewed_pieces = 54) (h3 : remaining_pieces = 2) (h4 : total_pieces = chewed_pieces + remaining_pieces)
  (h5 : total_pieces = packets * (total_pieces / packets)) :
  total_pieces / packets = 7 :=
by
  sorry

end gum_pieces_in_each_packet_l300_300471


namespace compare_values_l300_300812

def a := Real.exp (1 / 2)
def b := Real.log (1 / 2)
def c := Real.log 2 (Real.sqrt 2)

theorem compare_values : a > c ∧ c > b := by
  sorry

end compare_values_l300_300812


namespace find_circle_o_find_k_l300_300317

noncomputable def circle_c : set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 6*p.2 + 8 = 0 }

def point_m := (0, 2 : ℝ)
def point_n := (2, 0 : ℝ)

def circle_o (x y : ℝ) := x^2 + y^2 = 4

theorem find_circle_o :
  (∀ p ∈ circle_c, p = point_m) →
  (point_n ∈ circle_o) →
  circle_o = { p | p.1^2 + p.2^2 = 4 } 
:= sorry

def line_l (k : ℝ) (x y : ℝ) := y = k * x - (k + 1)

theorem find_k (k : ℝ) :
  (∀ (x y : ℝ), 
    line_l k x y → 
    ∃ arc1 arc2, 
      arc1 / arc2 = 3 / 1) →
  k = 1 
:= sorry

end find_circle_o_find_k_l300_300317


namespace area_proportionality_of_triangles_l300_300801

variables (R : ℝ) (C : ℝ) (O : ℝ) [Nint : NormedField ℝ]

noncomputable theory

def is_diameter (A B : ℝ) (circle_center : ℝ) := B - A = 2 * R

def point_C_condition (A B C : ℝ) := (C - A) / (B - C) = 6 / 7

def perpendicular (D C : ℝ) := C * D = 0

def diameter_through_D (D E : ℝ) := E = -D

def area_ratio (A B D C E : ℝ) : ℝ := 
  let area_ABD := abs (A * B - D * (A + B))
  let area_CDE := abs (C * (D + E) - E * C)
  area_ABD / area_CDE

theorem area_proportionality_of_triangles (A B C D E : ℝ) (h1 : is_diameter A B O) 
  (h2 : point_C_condition A B C) (h3 : perpendicular D C) (h4 : diameter_through_D D E) : 
  area_ratio A B D C E = 13 := 
by
  sorry

end area_proportionality_of_triangles_l300_300801


namespace number_of_non_fictions_equation_number_of_non_fictions_is_6_l300_300887

-- Given conditions
def number_of_fictions : ℕ := 5
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the combinations
def fiction_combinations : ℕ := combination number_of_fictions 2

theorem number_of_non_fictions_equation (N : ℕ) :
  10 * (N * (N - 1) / 2) = 150 :=
  sorry

theorem number_of_non_fictions_is_6 :
  ∃ N : ℕ, 10 * (N * (N - 1) / 2) = 150 ∧ N = 6 :=
begin
  use 6,
  split,
  { sorry },  -- This will be the actual proof that 10 * (6 * (6 - 1) / 2) = 150
  { refl }
end

end number_of_non_fictions_equation_number_of_non_fictions_is_6_l300_300887


namespace sam_distance_traveled_l300_300448

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300448


namespace linear_equation_l300_300145

-- Define each option as a hypothesis
def optionA (y : ℝ) : Prop := 3 * y + 1 = 6
def optionB (x : ℝ) : Prop := x + 3 > 7
def optionC (x : ℝ) : Prop := 4 / (x - 1) = 3 * x
def optionD (a : ℝ) : Prop := 3 * a - 4 = 0 -- Note that we give it an arbitrary equality for definition

-- Prove that option A is a linear equation
theorem linear_equation (y : ℝ) : optionA y → (∃ a b : ℝ, a ≠ 0 ∧ (a * y + b = 0)) :=
by 
  intros h
  use [3, -5]
  constructor
  { exact dec_trivial } -- Proof that 3 ≠ 0
  rw [optionA, sub_eq_add_neg]
  exact h

#check linear_equation

end linear_equation_l300_300145


namespace correct_calculation_l300_300917

theorem correct_calculation : 
(∀ x : ℝ, √ 12 = 3 * √ 2 → false) ∧ 
(∀ x : ℝ, √ 3 + √ 2 = √ 5 → false) ∧ 
(∀ x : ℝ, (√ 3)^2 = 3) := 
by
  split
  sorry  -- proof for first part
  split
  sorry  -- proof for second part
  split 
  sorry  -- proof for correct statement

end correct_calculation_l300_300917


namespace cos_double_angle_l300_300700

-- Defining the condition
def sin_alpha : ℝ := 3 / 5

-- The theorem to be proved
theorem cos_double_angle (h : sin α = sin_alpha) : cos (2 * α) = 7 / 25 :=
  sorry

end cos_double_angle_l300_300700


namespace sam_drove_200_miles_l300_300464

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300464


namespace cubic_polynomial_Q_l300_300817

theorem cubic_polynomial_Q (a b c d k : ℝ)
  (h1 : Q 0 = 3 * k)
  (h2 : Q 1 = 5 * k)
  (h3 : Q (-1) = 7 * k) :
  Q 2 + Q (-2) = 32 * k := by
let Q (x : ℝ) := a * x^3 + b * x^2 + c * x + d
sorry

end cubic_polynomial_Q_l300_300817


namespace Sarah_trucks_l300_300486

-- Define the problem where Sarah's initial trucks are represented by an unknown variable.
def trucks_initial (trucks_given_jeff trucks_given_amy trucks_left trucks_initial : ℕ) :=
  trucks_given_jeff + trucks_given_amy + trucks_left = trucks_initial

theorem Sarah_trucks (trucks_given_jeff trucks_given_amy trucks_left trucks_initial : ℕ) :
  trucks_given_jeff = 13 →
  trucks_given_amy = 21 →
  trucks_left = 38 →
  trucks_initial = 72 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Sarah_trucks_l300_300486


namespace cubes_closed_under_multiplication_l300_300415

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem cubes_closed_under_multiplication :
  ∀ a b : ℕ, is_cube (a^3 * b^3) :=
by
  intros a b
  use a * b
  ring
  sorry

end cubes_closed_under_multiplication_l300_300415


namespace swimmers_meet_22_times_in_15_minutes_l300_300137

theorem swimmers_meet_22_times_in_15_minutes 
  (pool_length : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (duration_minutes : ℕ) 
  (pool_length_eq : pool_length = 120)
  (speed_A_eq : speed_A = 4)
  (speed_B_eq : speed_B = 3)
  (duration_minutes_eq : duration_minutes = 15)
  : 22 = 22 :=
by
  rw [pool_length_eq, speed_A_eq, speed_B_eq, duration_minutes_eq]
  sorry

end swimmers_meet_22_times_in_15_minutes_l300_300137


namespace first_part_trip_distance_l300_300179

noncomputable def total_time (x : ℝ) : ℝ := x / 60 + (50 - x) / 30

def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

theorem first_part_trip_distance (x : ℝ) (h0 : 50 = (x + (50 - x))) (h1 : 40 = average_speed 50 (total_time x)) : x = 25 :=
by
  sorry

end first_part_trip_distance_l300_300179


namespace no_k_m_exists_l300_300286

def num_ones (n : ℕ) : ℕ :=
  nat.bits n tt

def a (n : ℕ) : ℕ :=
  if num_ones n % 2 = 0 then 0 else 1

theorem no_k_m_exists :
  ¬ ∃ k m : ℕ, (k > 0) ∧ (m > 0) ∧ ∀ j : ℕ, (0 ≤ j ∧ j ≤ m - 1) →
    a (k + j) = a (k + m + j) ∧ a (k + m + j) = a (k + 2 * m + j) :=
sorry

end no_k_m_exists_l300_300286


namespace geom_prog_all_integers_l300_300985

theorem geom_prog_all_integers (b : ℕ) (r : ℚ) (a c : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, b * r ^ n = a * n + c) ∧ ∃ b_1 : ℤ, b = b_1 →
  (∀ n : ℕ, ∃ b_n : ℤ, b * r ^ n = b_n) :=
by
  sorry

end geom_prog_all_integers_l300_300985


namespace three_digit_number_ends_same_sequence_l300_300282

theorem three_digit_number_ends_same_sequence (N : ℕ) (a b c : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
  (h2 : N % 10 = c)
  (h3 : (N / 10) % 10 = b)
  (h4 : (N / 100) % 10 = a)
  (h5 : a ≠ 0)
  (h6 : N^2 % 1000 = N) :
  N = 127 :=
by
  sorry

end three_digit_number_ends_same_sequence_l300_300282


namespace f_neg_2_l300_300321

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1/x else 0

theorem f_neg_2 :
  is_odd_function f →
  (∀ x : ℝ, x > 0 → f x = x^2 + 1/x) →
  f (-2) = -9 / 2 :=
by
  sorry

end f_neg_2_l300_300321


namespace simplified_sum_l300_300909

theorem simplified_sum :
  (-2^2003) + (2^2004) + (-2^2005) - (2^2006) = 5 * (2^2003) :=
by
  sorry

end simplified_sum_l300_300909


namespace triangle_side_ratios_l300_300880

theorem triangle_side_ratios (a b c : ℝ) (k l : ℝ) 
    (h_c : c = 280) 
    (h_perm : a + b + c = 720) 
    (h_ratio : a = k * c ∧ b = l * c) : (k + l = 1.5714) := 
by
    have h : a + b = 440 := by sorry
    have h2 : k * c + l * c = 440 := by sorry
    have h3 : k * 280 + l * 280 = 440 := by sorry
    have h4 : k + l = 440 / 280 := by sorry
    show k + l = 1.5714 := by sorry

end triangle_side_ratios_l300_300880


namespace am_gm_inequality_l300_300076

theorem am_gm_inequality {n : ℕ} (x : Fin n → ℝ) (hx : ∀ i, i < n → 0 < x i) :
  (∑ i, x i) / n ≥ (∏ i, x i) ^ (1 / n) := by
  sorry

end am_gm_inequality_l300_300076


namespace smallest_n_arithmetic_progression_l300_300391

theorem smallest_n_arithmetic_progression {a : ℕ → ℝ}
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (ha0_gt_0 : a 0 > 0)
  (h_eq : 5 * a 12 = 6 * a 18) :
  ∃ n, a n < 0 ∧ ∀ k, a k < 0 → k ≥ n :=
begin
  sorry
end

end smallest_n_arithmetic_progression_l300_300391


namespace work_completion_l300_300925

theorem work_completion (W : ℝ) (a b : ℝ) (ha : a = W / 12) (hb : b = W / 6) :
  W / (a + b) = 4 :=
by {
  sorry
}

end work_completion_l300_300925


namespace tan_plus_pi_over_4_l300_300299

variable (θ : ℝ)

-- Define the conditions
def condition_θ_interval : Prop := θ ∈ Set.Ioo (Real.pi / 2) Real.pi
def condition_sin_θ : Prop := Real.sin θ = 3 / 5

-- Define the theorem to be proved
theorem tan_plus_pi_over_4 (h1 : condition_θ_interval θ) (h2 : condition_sin_θ θ) :
  Real.tan (θ + Real.pi / 4) = 7 :=
sorry

end tan_plus_pi_over_4_l300_300299


namespace sam_drove_200_miles_l300_300466

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300466


namespace find_a5_l300_300379

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (a1 : ℝ)

-- Geometric sequence definition
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a1 * q^n

-- Given conditions
def condition1 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 1 + a 3 = 10

def condition2 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 2 + a 4 = -30

-- Theorem to prove
theorem find_a5 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h1 : geometric_sequence a a1 q)
  (h2 : condition1 a a1 q)
  (h3 : condition2 a a1 q) :
  a 5 = 81 := by
  sorry

end find_a5_l300_300379


namespace find_missing_square_l300_300685

-- Defining the sequence as a list of natural numbers' squares
def square_sequence (n: ℕ) : ℕ := n * n

-- Proving the missing element in the given sequence is 36
theorem find_missing_square :
  (square_sequence 0 = 1) ∧ 
  (square_sequence 1 = 4) ∧ 
  (square_sequence 2 = 9) ∧ 
  (square_sequence 3 = 16) ∧ 
  (square_sequence 4 = 25) ∧ 
  (square_sequence 6 = 49) →
  square_sequence 5 = 36 :=
by {
  sorry
}

end find_missing_square_l300_300685


namespace min_cost_19th_element_l300_300560

theorem min_cost_19th_element (f : ℕ × ℕ → ℕ) (h1 : ∀ x y, f(x+1, y) > f(x, y))
    (h2 : ∀ x y, f(x, y+1) > f(x, y)) : 
  O (19 * log 19) :=
sorry

end min_cost_19th_element_l300_300560


namespace extreme_values_of_f_max_min_values_on_interval_l300_300329

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.exp x)

theorem extreme_values_of_f : 
  (∃ x_max : ℝ, f x_max = 2 / Real.exp 1 ∧ ∀ x : ℝ, f x ≤ 2 / Real.exp 1) :=
sorry

theorem max_min_values_on_interval : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 
    (f 1 = 2 / Real.exp 1 ∧ ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → f x ≤ 2 / Real.exp 1)
     ∧ (f 2 = 4 / (Real.exp 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 2, 4 / (Real.exp 2) ≤ f x)) :=
sorry

end extreme_values_of_f_max_min_values_on_interval_l300_300329


namespace sin_alpha_b_point_l300_300323

theorem sin_alpha_b_point (b : ℝ) (α : ℝ) (h1 : sin α = 4 / 5) (h2 : (-b, 4) = point_where_terminal_side_passes α) : b = 3 ∨ b = -3 :=
by
  sorry

end sin_alpha_b_point_l300_300323


namespace sum_of_valid_ks_l300_300760

theorem sum_of_valid_ks : ∑ k in finset.filter (λ (k : ℤ), k < 2 ∧ k ≥ -1) (finset.Icc (-1) 1) = 0 :=
by
  sorry

end sum_of_valid_ks_l300_300760


namespace earnings_per_pig_l300_300484

-- Define the conditions as variables relevant to the problem
variables (numCows : ℕ) (numPigs : ℕ) (earnPerCow : ℕ) (totalEarned : ℕ)

-- Define the hypothesis based on given conditions
def hypotheses := (numCows = 20) ∧ 
                  (numPigs = 4 * numCows) ∧ 
                  (earnPerCow = 800) ∧ 
                  (totalEarned = 48000)

-- Define the theorem stating the amount she would earn for each pig
theorem earnings_per_pig (h : hypotheses) : (totalEarned - numCows * earnPerCow) / numPigs = 400 :=
by {
  -- We assume the proof details here just to complete the statement.
  sorry
}

end earnings_per_pig_l300_300484


namespace cos_arcsin_l300_300247

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l300_300247


namespace only_nonneg_int_solution_l300_300847

theorem only_nonneg_int_solution (x y z : ℕ) (h : x^3 = 3 * y^3 + 9 * z^3) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end only_nonneg_int_solution_l300_300847


namespace interval_length_difference_l300_300654

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

def y_function (x : ℝ) := abs (log_base (1/2) x)

def interval_length (x1 x2 : ℝ) (h : x1 < x2) := x2 - x1

theorem interval_length_difference :
  (∃ a b : ℝ, (∀ x, a ≤ x ∧ x ≤ b → 0 ≤ y_function x ∧ y_function x ≤ 2)) →
  ((interval_length (1/4) 4 (by norm_num) - interval_length (1/4) 1 (by norm_num)) = (3 : ℝ)) :=
by
  intro h
  -- Proof goes here
  sorry

end interval_length_difference_l300_300654


namespace dot_product_a_b_angle_between_a_b_l300_300315

variables (e₁ e₂ : EuclideanSpace ℝ (Fin 2))

-- Conditions
def unit_vector_e1 : Prop := ∥e₁∥ = 1
def unit_vector_e2 : Prop := ∥e₂∥ = 1
def angle_60 : Prop := inner e₁ e₂ = (1 / 2)
def a : EuclideanSpace ℝ (Fin 2) := 2 • e₁ + e₂
def b : EuclideanSpace ℝ (Fin 2) := -3 • e₁ + 2 • e₂

-- Theorem for the dot product
theorem dot_product_a_b (h₁ : unit_vector_e1 e₁)
                        (h₂ : unit_vector_e2 e₂)
                        (h₃ : angle_60 e₁ e₂) :
  inner a b = -7 / 2 := sorry

-- Theorem for the angle between a and b
theorem angle_between_a_b (h₁ : unit_vector_e1 e₁)
                          (h₂ : unit_vector_e2 e₂)
                          (h₃ : angle_60 e₁ e₂) :
  angle a b = 120 := sorry

end dot_product_a_b_angle_between_a_b_l300_300315


namespace leak_empties_cistern_in_24_hours_l300_300176

theorem leak_empties_cistern_in_24_hours (F L : ℝ) (h1: F = 1 / 8) (h2: F - L = 1 / 12) :
  1 / L = 24 := 
by {
  sorry
}

end leak_empties_cistern_in_24_hours_l300_300176


namespace ratio_OM_ON_l300_300392

variables {A B C D M N O : Type} [Field A] [Field B] [Field C] [Field D] [Field M] [Field N] [Field O]
variables (r : ℝ) (xA xB xC xD xM xN xO : ℝ)
variables (A B C D M N O : Type) 
variables [Field O] 

-- Variables for distances from O to A, B, C, and D respectively
variables (OA : ℝ) (OB : ℝ) (OC : ℝ) (OD : ℝ)
variables (OM ON : ℝ)

-- The given conditions
axiom h_OA : OA = 5
axiom h_OB : OB = 6
axiom h_OC : OC = 7
axiom h_OD : OD = 8

-- Midpoints of diagonals
variable h_M : M = midpoint A C
variable h_N : N = midpoint B D

-- Statement to prove
theorem ratio_OM_ON : ∀ (OM ON : ℝ),
  OM / ON = 35 / 48 :=
by { sorry }

end ratio_OM_ON_l300_300392


namespace faith_weekly_earnings_l300_300673

def faith_hourly_rate : ℝ := 13.5
def regular_hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def overtime_hours_per_day : ℝ := 2
def commission_rate : ℝ := 0.1
def total_weekly_sales : ℝ := 3200
def overtime_rate_multiplier : ℝ := 1.5

theorem faith_weekly_earnings :
  let regular_hours_per_week := regular_hours_per_day * days_per_week in
  let regular_earnings := regular_hours_per_week * faith_hourly_rate in
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week in
  let overtime_rate := faith_hourly_rate * overtime_rate_multiplier in
  let overtime_earnings := overtime_hours_per_week * overtime_rate in
  let commission := commission_rate * total_weekly_sales in
  let total_earnings := regular_earnings + overtime_earnings + commission in
  total_earnings = 1062.50 :=
by
  sorry

end faith_weekly_earnings_l300_300673


namespace part1_solution_part2_solution_l300_300305

noncomputable def part1_problem (a b : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, a * x^2 - b * x + 1
  (∀ x, (1/4 < x ∧ x < 1/3) ↔ f x < 0) →
  a = 12 ∧ b = 7

noncomputable def part2_problem (a : ℕ) : Prop :=
  let f : ℝ → ℝ := λ x, a * x^2 - (a + 2) * x + 1
  (∀ x, (0 ≤ x ∧ x ≤ 1) → f x ≥ -1) →
  a = 1 ∨ a = 2

theorem part1_solution (a b : ℝ) (h : part1_problem a b) : a = 12 ∧ b = 7 := sorry

theorem part2_solution (a : ℕ) (h : part2_problem a) : a = 1 ∨ a = 2 := sorry

end part1_solution_part2_solution_l300_300305


namespace value_of_x_squared_plus_y_squared_l300_300354

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h : |x - 1/2| + (2*y + 1)^2 = 0) : 
  x^2 + y^2 = 1/2 :=
sorry

end value_of_x_squared_plus_y_squared_l300_300354


namespace factorize_expression_l300_300669

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300669


namespace stratified_sampling_l300_300946

/-- Given a batch of 98 water heaters with 56 from Factory A and 42 from Factory B,
    and a stratified sample of 14 units is to be drawn, prove that the number 
    of water heaters sampled from Factory A is 8 and from Factory B is 6. --/

theorem stratified_sampling (batch_size A B sample_size : ℕ) 
  (h_batch : batch_size = 98) 
  (h_fact_a : A = 56) 
  (h_fact_b : B = 42) 
  (h_sample : sample_size = 14) : 
  (A * sample_size / batch_size = 8) ∧ (B * sample_size / batch_size = 6) := 
  by
    sorry

end stratified_sampling_l300_300946


namespace square_folding_problem_l300_300502

theorem square_folding_problem
    (A B C D : Point)
    (P Q : Point)
    (mid : Point)
    (dist_AB : dist A B = 24)
    (dist_BC : dist B C = 24)
    (dist_CD : dist C D = 24)
    (dist_DA : dist D A = 24)
    (on_P : P ∈ Segment B C)
    (on_Q : Q ∈ Segment D A)
    (folds : reflecting B P Q = mid)
    (midpoint_CD : mid = midpoint C D) :
    dist P C = 9 ∧ dist A Q = 3 ∧ dist P Q = 12 * sqrt 5 :=
by
  sorry

end square_folding_problem_l300_300502


namespace nate_total_distance_l300_300475

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end nate_total_distance_l300_300475


namespace basketball_count_l300_300118

theorem basketball_count (s b v : ℕ) 
  (h1 : s = b + 23) 
  (h2 : v = s - 18)
  (h3 : v = 40) : b = 35 :=
by sorry

end basketball_count_l300_300118


namespace distance_between_vertices_l300_300399

theorem distance_between_vertices :
  let C := (2, 1)
  let D := (-3, 11)
  dist C D = 5 * Real.sqrt 5 :=
by
  let C := (2, 1)
  let D := (-3, 11)
  calc dist C D = Real.sqrt ((2 - (-3))^2 + (1 - 11)^2) : sorry
               ... = Real.sqrt (5^2 + (-10)^2) : sorry
               ... = Real.sqrt (25 + 100) : sorry
               ... = Real.sqrt 125 : sorry
               ... = 5 * Real.sqrt 5 : sorry

end distance_between_vertices_l300_300399


namespace probability_diff_specialties_l300_300527

def total_students := 50
def art_students := 15
def dance_students := 35

theorem probability_diff_specialties :
  (nat.choose art_students 1 * nat.choose dance_students 1) / nat.choose total_students 2 = 3 / 7 := 
sorry

end probability_diff_specialties_l300_300527


namespace boat_license_combinations_l300_300202

theorem boat_license_combinations : 
  (let letters := 3 in
   let digit_places := 5 in
   let digits := 10 in
   letters * digits^digit_places = 300000) :=
by
  let letters := 3
  let digit_places := 5
  let digits := 10
  have h : letters * digits^digit_places = 3 * 10^5 := by rfl
  rw h
  exact rfl

end boat_license_combinations_l300_300202


namespace problem_remainder_l300_300045

def is_perfect_square (x : ℤ) : Prop := ∃ m : ℤ, m * m = x

noncomputable def T : ℤ :=
  ∑ n in (finset.filter (λ n, 0 < n ∧ is_perfect_square(n^2 + 18 * n - 3000))
                         (finset.range 10000)), n

theorem problem_remainder :
  T % 1000 = 590 :=
sorry

end problem_remainder_l300_300045


namespace gcd_b_c_min_value_l300_300352

open Nat

theorem gcd_b_c_min_value (a b c : ℕ) (h1 : gcd a b = 960) (h2 : gcd a c = 324) : gcd b c = 12 := by
  sorry

end gcd_b_c_min_value_l300_300352


namespace new_fraction_of_red_marbles_is_3_div_7_l300_300364

-- Variables for initial conditions
variables (x : ℝ) (h₀ : x > 0) -- Assume the number of marbles x is positive

-- Conditions: fractions of blue and red marbles
def blue_fraction := 2 / 3
def red_fraction := 1 - blue_fraction

-- Definitions based on initial marbles
def initial_red := red_fraction * x
def initial_blue := blue_fraction * x

-- Changes to the marbles
def new_red := 3 * initial_red
def new_blue := 2 * initial_blue

-- New total number of marbles after changes
def new_total := new_red + new_blue

-- The fraction of red marbles in the new total
def new_red_fraction := new_red / new_total

-- Theorem statement in Lean 4
theorem new_fraction_of_red_marbles_is_3_div_7 : new_red_fraction = 3 / 7 := by
  -- Placeholder for the actual proof
  sorry

end new_fraction_of_red_marbles_is_3_div_7_l300_300364


namespace max_volume_hexagonal_pyramid_l300_300191

theorem max_volume_hexagonal_pyramid (x y : ℝ) (h_cond : x + y = 20) :
  ∃ V, V = 128 * real.sqrt 15 ∧ ∀ V', V' ≤ V := 
begin
  sorry
end

end max_volume_hexagonal_pyramid_l300_300191


namespace sum_of_first_60_digits_l300_300549

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end sum_of_first_60_digits_l300_300549


namespace markers_last_group_correct_l300_300591

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end markers_last_group_correct_l300_300591


namespace probability_one_hit_in_one_round_probability_three_hits_in_three_rounds_l300_300175

-- Define the conditions and probabilities in Lean 4

-- Event that A hits the target with probability 1/2
axiom P_A : ℕ → ℝ
axiom hP_A : ∀ n, P_A n = 1 / 2

-- Event that B hits the target with probability 2/3
axiom P_B : ℕ → ℝ
axiom hP_B : ∀ n, P_B n = 2 / 3

-- Independence of events between A and B in one round
axiom independence_AB : ∀ n, indep (P_A n) (P_B n)

-- Independence of events between different rounds for both A and B
axiom independence_rounds : ∀ m n, m ≠ n → indep (P_A m) (P_A n) ∧ indep (P_B m) (P_B n)

noncomputable def P_one_hit_in_one_round : ℝ :=
  let PA := 1 / 2
  let PB := 2 / 3
  PA * (1 - PB) + (1 - PA) * PB

-- Main theorem for part 1
theorem probability_one_hit_in_one_round : P_one_hit_in_one_round = 1 / 2 :=
by sorry

noncomputable def P_three_hits_in_three_rounds : ℝ :=
  let PA := 1 / 2
  let PB := 2 / 3
  let P_D := [1/8, 3/8, 3/8, 1/8] -- Probabilities for A hitting 0, 1, 2, 3 times
  let P_E := [1/27, 2/9, 4/9, 8/27] -- Probabilities for B hitting 0, 1, 2, 3 times
  (P_D[0] * P_E[3]) + (P_D[1] * P_E[2]) + (P_D[2] * P_E[1]) + (P_D[3] * P_E[0])

-- Main theorem for part 2
theorem probability_three_hits_in_three_rounds : P_three_hits_in_three_rounds = 7 / 24 :=
by sorry

end probability_one_hit_in_one_round_probability_three_hits_in_three_rounds_l300_300175


namespace parabola_equation_l300_300605

theorem parabola_equation (p : ℝ) (h : 2 * p = 8) :
  ∃ (a : ℝ), a = 8 ∧ (y^2 = a * x ∨ y^2 = -a * x) :=
by
  sorry

end parabola_equation_l300_300605


namespace inequality_problem_l300_300722

variables {n : ℕ} {a b : Fin n → ℝ} {m : ℝ}

theorem inequality_problem
  (pos_a : ∀ i, 0 < a i)
  (pos_b : ∀ i, 0 < b i)
  (cond_m : -1 < m ∧ m < 0) :
  (∑ i, (a i) ^ (m + 1) / (b i) ^ m) ≤ (∑ i, a i) ^ (m + 1) / (∑ i, b i) ^ m :=
sorry

end inequality_problem_l300_300722


namespace decreasing_on_interval_l300_300503

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + 2 * m * x + 3

theorem decreasing_on_interval (m : ℝ) (h_even : ∀ x : ℝ, f(m, x) = f(m, -x)) :
  ∀ x y : ℝ, 2 < x → x < y → y < 5 → f(m, x) > f(m, y) :=
by
  sorry

end decreasing_on_interval_l300_300503


namespace regular_pentagon_inscribed_AB_mul_AC_eq_sqrt_5_l300_300980

theorem regular_pentagon_inscribed_AB_mul_AC_eq_sqrt_5 
    (A B C D E : Type) 
    [RegularPentagonInscribedCircle A B C D E (radius := 1)] : 
  length (segment A B) * length (segment A C) = √5 := 
begin 
  sorry 
end

end regular_pentagon_inscribed_AB_mul_AC_eq_sqrt_5_l300_300980


namespace determine_x_l300_300659

theorem determine_x (x : ℝ) :
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 8 * x + 15) / (x^2 - 10 * x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 :=
by
  sorry

end determine_x_l300_300659


namespace min_change_sum_l300_300996

/--
  At the beginning of the school year:
  - 60% of students love math.
  - 40% of students do not love math.
  - 30% of students enjoy math homework.

  At the end of the school year:
  - 80% of students love math.
  - 20% of students do not love math.
  - 50% of students enjoy math homework.

  We need to prove the sum of the minimum possible values of y% (change in students loving math) 
  and z% (change in students enjoying math homework) is 40%.
-/
theorem min_change_sum (initial_love_math : ℝ)
                       (initial_no_love_math : ℝ)
                       (initial_love_homework : ℝ)
                       (end_love_math : ℝ)
                       (end_no_love_math : ℝ)
                       (end_love_homework : ℝ) :
  initial_love_math = 0.60 → initial_no_love_math = 0.40 → 
  initial_love_homework = 0.30 → end_love_math = 0.80 → 
  end_no_love_math = 0.20 → end_love_homework = 0.50 → 
  (20 + 20 = 40 : ℝ) :=
by {
  intros,
  exact rfl
}

end min_change_sum_l300_300996


namespace evaluate_27_x_plus_1_l300_300750

theorem evaluate_27_x_plus_1 (x : ℝ) (h : 3^(2*x) = 13) : 27^(x+1) = 4563 := by
  sorry

end evaluate_27_x_plus_1_l300_300750


namespace cos_arcsin_l300_300244

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l300_300244


namespace sum_of_divisors_of_11_squared_l300_300126

theorem sum_of_divisors_of_11_squared (a b c : ℕ) (h1 : a ∣ 11^2) (h2 : b ∣ 11^2) (h3 : c ∣ 11^2) (h4 : a * b * c = 11^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) :
  a + b + c = 23 :=
sorry

end sum_of_divisors_of_11_squared_l300_300126


namespace integer_expression_iff_divisible_l300_300055

theorem integer_expression_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  let C (n k : ℤ) := (n.factorial / (k.factorial * (n - k).factorial))
  let expr := (n + 2 * k - 3) / (k + 2) * C n k
  expr ∈ ℤ ↔ (k + 2) ∣ n := by
  sorry

end integer_expression_iff_divisible_l300_300055


namespace simplify_trig_expression_l300_300087

theorem simplify_trig_expression :
  (2 - Real.sin 21 * Real.sin 21 - Real.cos 21 * Real.cos 21 + 
  (Real.sin 17 * Real.sin 17) * (Real.sin 17 * Real.sin 17) + 
  (Real.sin 17 * Real.sin 17) * (Real.cos 17 * Real.cos 17) + 
  (Real.cos 17 * Real.cos 17)) = 2 :=
by
  sorry

end simplify_trig_expression_l300_300087


namespace cyclic_quadrilateral_identity_l300_300626

variables {A B C M : Type*} [Field A] (a b c ab bc ac : A)
  (MAB MAC MCA MCB MBA MBC : A)

theorem cyclic_quadrilateral_identity
  (circle : CyclicQuadrilateral A B C (M : PointOnArcBC A B C))
  (segment_BC : Length BC = bc)
  (segment_AB : Length AB = ab)
  (segment_AC : Length AC = ac)
  (segment_BC_squared : (Length BC)^2 = bc^2)
  (MAB_MAC : (MAB)(MAC) = bc^2)
  (MCA_MCB : (MCA)(MCB) = ab^2)
  (MBA_MBC : (MBA)(MBC) = ac^2) :
  bc^2 * MAB * MAC = ab^2 * MCA * MCB + ac^2 * MBA * MBC := by
sorry

end cyclic_quadrilateral_identity_l300_300626


namespace xena_running_speed_l300_300150

theorem xena_running_speed
  (head_start : ℕ)
  (burn_distance : ℕ)
  (dragon_speed : ℕ)
  (xena_time : ℕ)
  (xena_distance_to_cave : ℕ)
  (xena_speed : ℕ)
  (head_start = 600)
  (burn_distance = 120)
  (dragon_speed = 30)
  (xena_time = 32)
  (xena_distance_to_cave = 960 - (head_start - burn_distance))
  (xena_speed = xena_distance_to_cave / xena_time) :
  xena_speed = 15 := by
  sorry

end xena_running_speed_l300_300150


namespace quadratic_inequality_ab_l300_300763

theorem quadratic_inequality_ab (a b : ℝ) (h : Set.Ioo (-∞) (-1/3) ∪ Set.Ioo (1/2) ∞ = {x | ax^2 + bx + 2 < 0}) :
  a - b = -14 := sorry

end quadratic_inequality_ab_l300_300763


namespace total_canoes_built_l300_300218

def boatWorksCanoes (january : ℕ) (feb : ℕ → ℕ) : ℕ :=
  january + feb january + feb (feb january) + feb (feb (feb january))

theorem total_canoes_built :
  boatWorksCanoes 3 (λ n, 3 * n) = 120 :=
by
  -- Insert proof here
  sorry

end total_canoes_built_l300_300218


namespace vector_calculation_l300_300642

theorem vector_calculation :
  2 • (⟨3, -2, 5⟩ : ℝ × ℝ × ℝ + ⟨-1, 6, -7⟩ : ℝ × ℝ × ℝ) = ⟨4, 8, -4⟩ : ℝ × ℝ × ℝ :=
sorry

end vector_calculation_l300_300642


namespace Emily_at_70_percent_l300_300846

-- Define the points: P, Q, R, S, T, U
inductive Point
| P | Q | R | S | T | U
deriving DecidableEq

open Point

-- Define the distance function between consecutive points
def distance (a b : Point) : ℕ :=
  match a, b with
  | P, Q | Q, R | R, S | S, T | T, U | U, T | T, S | S, R | R, Q | Q, P => 1
  | _, _ => 0

-- Define the total distance for a round trip P -> U -> P
def total_distance : ℕ := 10 * distance P Q

-- Define the target distance for 70% of the total distance
def target_distance : ℕ := 7 * distance P Q

-- Prove that after walking 70% of her journey Emily is at Point S
theorem Emily_at_70_percent : 
  let dist : ℕ := distance P Q in 
  let total_dist : ℕ := 10 * dist in
  let target_dist : ℕ := 7 * dist in
  target_dist = 7 :=
by {
  -- Placeholder proof
  sorry
}

end Emily_at_70_percent_l300_300846


namespace speed_boat_25_kmph_l300_300947

noncomputable def speed_of_boat_in_still_water (V_s : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let V_d := distance / time
  V_d - V_s

theorem speed_boat_25_kmph (h_vs : V_s = 5) (h_time : time = 4) (h_distance : distance = 120) :
  speed_of_boat_in_still_water V_s time distance = 25 :=
by
  rw [h_vs, h_time, h_distance]
  unfold speed_of_boat_in_still_water
  simp
  norm_num

end speed_boat_25_kmph_l300_300947


namespace tea_in_pot_l300_300529

theorem tea_in_pot (amount_per_cup : ℕ) (num_cups : ℕ) (total_tea : ℕ) 
  (h1 : amount_per_cup = 65) 
  (h2 : num_cups = 16) 
  (h3 : total_tea = amount_per_cup * num_cups) : 
  total_tea = 1040 :=
by
  rw [h1, h2] at h3
  exact h3.symm

end tea_in_pot_l300_300529


namespace final_image_correct_l300_300524

noncomputable def point_transformation (p : ℝ × ℝ) : ℝ × ℝ :=
  let r := (-p.1, -p.2) -- rotation by 180° clockwise
  in (r.1, -r.2) -- reflection in the x-axis

def transformation_result : String :=
  -- 'stop' -> ('s', 't', 'o', 'p') initial points
  -- After transformations: ('q', 'o', 'n', 's')
  "qons"

theorem final_image_correct :
  transformation_result = "qons" :=
sorry

end final_image_correct_l300_300524


namespace number_values_g1_l300_300815

def g (x : ℝ) : ℝ := sorry

theorem number_values_g1 :
  (∀ x y : ℝ, g ((x - y)^3) = g x^3 - 3 * x^2 * g y + y^3) →
  let n := 2 in
  let s := 1 - real.cbrt 2 in
  n * s = 2 * (1 - real.cbrt 2) :=
by
  sorry

end number_values_g1_l300_300815


namespace union_is_correct_l300_300741

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_is_correct : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_is_correct_l300_300741


namespace exists_close_pair_in_interval_l300_300841

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end exists_close_pair_in_interval_l300_300841


namespace swallow_distance_flew_l300_300580

/-- The TGV departs from Paris at 150 km/h toward Marseille, which is 800 km away, while an intercité departs from Marseille at 50 km/h toward Paris at the same time. A swallow perched on the TGV takes off at that moment, flying at 200 km/h toward Marseille. We aim to prove that the distance flown by the swallow when the two trains meet is 800 km. -/
theorem swallow_distance_flew :
  let distance := 800 -- distance between Paris and Marseille in km
  let speed_TGV := 150 -- speed of TGV in km/h
  let speed_intercite := 50 -- speed of intercité in km/h
  let speed_swallow := 200 -- speed of swallow in km/h
  let combined_speed := speed_TGV + speed_intercite
  let time_to_meet := distance / combined_speed
  let distance_swallow_traveled := speed_swallow * time_to_meet
  distance_swallow_traveled = 800 := 
by
  sorry

end swallow_distance_flew_l300_300580


namespace linear_equation_correct_l300_300921

-- Define what it means for an equation to be linear
def is_linear_eq (eq : String) : Prop :=
  eq = "-x-3=4"

-- Given conditions as definitions
def option_A : String := "1/x - 1 = 2"
def option_B : String := "x^2 + 3 = x + 2"
def option_C : String := "-x - 3 = 4"
def option_D : String := "2y - 3x = 4"

-- Proof problem statement
theorem linear_equation_correct (hA: not (is_linear_eq option_A)) 
                               (hB: not (is_linear_eq option_B)) 
                               (hC: is_linear_eq option_C)
                               (hD: not (is_linear_eq option_D)) : 
  option_C = "-x -3 = 4" :=
by
  sorry

end linear_equation_correct_l300_300921


namespace sequence_geometric_l300_300113

variable (a : ℕ+ → ℝ)

-- sum of the first n terms
def S (n : ℕ+) : ℝ := 3 + 2 * a n

theorem sequence_geometric (n : ℕ+) :
  (S n = 3 + 2 * a n) →
  (∀ n, S n = S (n - 1) + a n) →
  (∀ n, a n = 2 * a (n - 1)) :=
begin
  sorry
end

end sequence_geometric_l300_300113


namespace triangle_altitudes_meet_at_orthocenter_l300_300309

theorem triangle_altitudes_meet_at_orthocenter
  {A B C K H E : Type*} [Nonempty A] [Nonempty B] [Nonempty C]
  [Nonempty K] [Nonempty H] [Nonempty E]
  (altitude_AK : AK) (altitude_BH : BH) (altitude_CE : CE) : 
  ∃ D : Type*, ∀ (a b : Type*), a ≠ b :=
sorry

end triangle_altitudes_meet_at_orthocenter_l300_300309


namespace monotonic_intervals_maximum_value_l300_300718

variables {a b : ℝ}
noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := a * x + b
noncomputable def F (x : ℝ) := f x - g x

-- Theorem 1: Monotonic intervals of F(x) when a=1
theorem monotonic_intervals (h : a = 1) : 
  ∃ (I1 I2 : set ℝ), I1 = set.Iic 0 ∧ I2 = set.Ioi 0 ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x < y) ∧ 
  (∀ x y ∈ I1, x < y → F x > F y) ∧ 
  (∀ x y ∈ I2, x < y → F x < F y) := sorry

-- Theorem 2: Maximum value of a + b given f(x) ≥ g(x) for all x ∈ ℝ
theorem maximum_value (h : ∀ x : ℝ, f x ≥ g x) : 
  ∃ (max_val : ℝ), max_val = Real.exp 1 ∧ (∀ a b : ℝ, (f x ≥ g x) → (a + b ≤ max_val)) := sorry

end monotonic_intervals_maximum_value_l300_300718


namespace pen_price_equation_l300_300972

theorem pen_price_equation
  (x y : ℤ)
  (h1 : 100 * x - y = 100)
  (h2 : 2 * y - 100 * x = 200) : x = 4 :=
by
  sorry

end pen_price_equation_l300_300972


namespace melissa_work_hours_l300_300063

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end melissa_work_hours_l300_300063


namespace solve_system1_solve_system2_l300_300489

theorem solve_system1 (x y : ℚ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) :
  x = 3 / 2 ∧ y = -7 / 2 := 
sorry

theorem solve_system2 (x y : ℚ) (h1 : 3 * x - 2 * y = 1) (h2 : 7 * x + 4 * y = 11) :
  x = 1 ∧ y = 1 := 
sorry

end solve_system1_solve_system2_l300_300489


namespace gcd_9011_4403_l300_300906

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := 
by sorry

end gcd_9011_4403_l300_300906


namespace circle_equation_l300_300680

theorem circle_equation 
  (x y : ℝ)
  (center : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (line1 : ℝ × ℝ → Prop)
  (line2 : ℝ × ℝ → Prop)
  (hx : line1 center)
  (hy : line2 tangent_point)
  (tangent_point_val : tangent_point = (2, -1))
  (line1_def : ∀ (p : ℝ × ℝ), line1 p ↔ 2 * p.1 + p.2 = 0)
  (line2_def : ∀ (p : ℝ × ℝ), line2 p ↔ p.1 + p.2 - 1 = 0) :
  (∃ (x0 y0 r : ℝ), center = (x0, y0) ∧ r > 0 ∧ (x - x0)^2 + (y - y0)^2 = r^2 ∧ 
                        (x - x0)^2 + (y - y0)^2 = (x - 1)^2 + (y + 2)^2 ∧ 
                        (x - 1)^2 + (y + 2)^2 = 2) :=
by {
  sorry
}

end circle_equation_l300_300680


namespace problem_statement_l300_300345

theorem problem_statement :
  (∑ n in Finset.range 1000, (n + 1) * (1001 - (n + 1))) = 1000 * 500 * (667 / 1000) :=
by
  sorry

end problem_statement_l300_300345


namespace part_one_part_two_l300_300003

-- Define the sequence {a_n}
def a : ℕ → ℕ → ℕ
| 0 _ := 0
| 1 _ := 1
| (n + 1) 0 := 0
| (n + 1) (m + 1) := (1 + 1 / n) * a n m + (n + 1) * 2 ^ n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ := a n n / n

-- Define the sum sequence S_n
def S (n : ℕ) : ℕ := ∑ i in range n, a i i

-- The first part of the proof (I)
theorem part_one (n : ℕ) (hn : 0 < n) : b n = 2^n - 1 := sorry

-- The second part of the proof (II)
theorem part_two (n : ℕ) : S n = 2 + (n - 1) * 2^(n + 1) - n * (n + 1) / 2 := sorry

end part_one_part_two_l300_300003


namespace median_moons_per_planet_l300_300907

-- Define the list of number of moons for each planet including Pluto
def moons : List ℕ := [0, 0, 1, 2, 16, 23, 15, 2, 5]

-- Define a function to compute the median of a nonempty list of natural numbers
-- by first sorting the list and then taking the middle element
noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! ((sorted.length - 1) / 2)

theorem median_moons_per_planet : median moons = 2 := by
  sorry

end median_moons_per_planet_l300_300907


namespace line_equation_is_correct_l300_300101

variables {R : Type*} [LinearOrderedField R] [Real] -- Ensure R includes real numbers needed for 45-degree angle and tan.

def line_passing_p (x y : R) : Prop :=
  ∃ k : R, y - 2 = k * (x - 1)

def has_angle_45_degrees (k : R) : Prop :=
  k = Real.tan (Real.pi / 4)

noncomputable def equation_of_line (k : R) (x y : R) : Prop :=
  y - 2 = k * (x - 1)

theorem line_equation_is_correct :
  (line_passing_p 1 2) ∧ 
  (has_angle_45_degrees 1) → 
  (equation_of_line 1 1 2) := 
by
  sorry

end line_equation_is_correct_l300_300101


namespace line_slope_intercept_l300_300959

theorem line_slope_intercept :
  (∀ (x y : ℝ), 3 * (x + 2) - 4 * (y - 8) = 0 → y = (3/4) * x + 9.5) :=
sorry

end line_slope_intercept_l300_300959


namespace penelope_min_games_l300_300481

theorem penelope_min_games (m w l: ℕ) (h1: 25 * w - 13 * l = 2007) (h2: m = w + l) : m = 87 := by
  sorry

end penelope_min_games_l300_300481


namespace find_locus_l300_300773

variable (l1 l2 : set (ℝ × ℝ)) -- Two lines in the plane
variable (d1 d2 : (ℝ × ℝ) → ℝ) -- Distance functions to the lines
variable (a : ℝ) -- Given segment

noncomputable def locus_of_points : set (ℝ × ℝ) :=
  {X | abs (d1 X - d2 X) = a}

theorem find_locus (l1 l2 : set (ℝ × ℝ)) (d1 d2 : (ℝ × ℝ) → ℝ) (a : ℝ) :
  ∃ (M1 M2 M3 M4 : (ℝ × ℝ)), 
    locus_of_points l1 l2 d1 d2 a = 
    set_of (λ X, ∃ (sides : set (ℝ × ℝ)), 
            X ∈ sides ∧ 
            (sides = {M1, M2, M3, M4} ∨
             sides ⊆ {M1, M2, M3, M4} ∧
             (line_through M1 M2) X ∧
             (line_through M3 M4) X)) :=
sorry

noncomputable def line_through (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), p = (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2))}

end find_locus_l300_300773


namespace solve_equation_sin_cos_l300_300850

theorem solve_equation_sin_cos (x y z : ℝ) (n k m : ℤ) :
  (sin x ≠ 0) →
  (sin y ≠ 0) →
  (sin^2 x + 1 / sin^2 x)^3 + (sin^2 y + 1 / sin^2 y)^3 = 16 * cos z →
  x = (π / 2) + π * n ∧ y = (π / 2) + π * k ∧ z = 2 * π * m :=
by
  intro h1 h2 heq
  sorry

end solve_equation_sin_cos_l300_300850


namespace biased_coin_4_heads_probability_l300_300171

theorem biased_coin_4_heads_probability:
  ∀ (h : ℚ) (p q : ℕ),
  {6.choose 2} * h^2 * (1-h)^4 = {6.choose 3} * h^3 * (1-h)^3 →
  h < 1/2 →
  p = 19440 →
  q = 117649 →
  p + q = 137089 :=
begin
  intros h p q,
  intros h_probability h_biased p_val q_val,
  sorry, -- Skipping the proof
end

end biased_coin_4_heads_probability_l300_300171


namespace pages_written_in_a_year_l300_300010

theorem pages_written_in_a_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) :
  pages_per_letter = 3 → friends = 2 → times_per_week = 2 → weeks_per_year = 52 → 
  pages_per_letter * friends * times_per_week * weeks_per_year = 624 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end pages_written_in_a_year_l300_300010


namespace problem_I_problem_II_l300_300735

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem problem_I :
  ∀ x ∈ Set.Icc 0 Real.pi, (f x) ≥ (f (Real.pi / 3) - Real.sqrt 3) ∧ (f x) ≤ f Real.pi :=
sorry

theorem problem_II :
  ∀ a : ℝ, ((∃ x : ℝ, (0 < x ∧ x < Real.pi / 2) ∧ f x < a * x) ↔ a > -1) :=
sorry

end problem_I_problem_II_l300_300735


namespace range_different_l300_300730

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
noncomputable def g (t : ℝ) : ℝ := t^2 - 2 * t + 3

theorem range_different (t : ℝ) : 
  ∃ y : ℝ, y ∉ (set.range (λ x : ℝ, f (g x))) ∧ y ∈ (set.range f) := 
sorry

end range_different_l300_300730


namespace find_angle_C_find_side_c_l300_300384

noncomputable def m_vector (B C : ℝ) : ℝ × ℝ := (Real.cos B, 2 * (Real.cos (C / 2))^2 - 1)
noncomputable def n_vector (c b a : ℝ) : ℝ × ℝ := (c, b - 2 * a)

axiom dot_product_zero (B C c b a : ℝ) : (m_vector B C).fst * (n_vector c b a).fst + (m_vector B C).snd * (n_vector c b a).snd = 0

def question_1_condition (B C : ℝ) : Prop := ∃ a b c : ℝ, dot_product_zero B C c b a
def question_1_answer (C : ℝ) : Prop := C = Real.pi / 3

theorem find_angle_C (B : ℝ) : ∀ (C : ℝ), question_1_condition B C → question_1_answer C :=
begin
  sorry
end

axiom triangle_area (a b C : ℝ) (S : ℝ) : S = 2 * Real.sqrt 3 → 1 / 2 * a * b * Real.sin C = S
axiom sides_sum (a b : ℝ) : a + b = 6

def question_2_condition (a b : ℝ) (C : ℝ) : Prop := ∃ c : ℝ, triangle_area a b C (2 * Real.sqrt (3 : ℝ)) ∧ sides_sum a b
def question_2_answer (c : ℝ) : Prop := c = 2 * Real.sqrt (3 : ℝ)

theorem find_side_c (a b : ℝ) (C : ℝ) : ∀ (c : ℝ), question_2_condition a b C → question_2_answer c :=
begin
  sorry
end

end find_angle_C_find_side_c_l300_300384


namespace cost_to_selling_ratio_l300_300112

theorem cost_to_selling_ratio (cp sp: ℚ) (h: sp = cp * (1 + 0.25)): cp / sp = 4 / 5 :=
by
  sorry

end cost_to_selling_ratio_l300_300112


namespace big_stack_customers_l300_300637

theorem big_stack_customers 
  (big_stack_pancakes short_stack_pancakes total_short_stack_customers total_pancakes : ℕ)
  (h1 : big_stack_pancakes = 5)
  (h2 : short_stack_pancakes = 3)
  (h3 : total_short_stack_customers = 9)
  (h4 : total_pancakes = 57) : 
  let B := (total_pancakes - total_short_stack_customers * short_stack_pancakes) / big_stack_pancakes in
  B = 6 :=
by
  let B := (total_pancakes - total_short_stack_customers * short_stack_pancakes) / big_stack_pancakes
  sorry

end big_stack_customers_l300_300637


namespace sam_drove_200_miles_l300_300462

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300462


namespace cuboid_volume_l300_300854

def cos60 := 1 / 2

theorem cuboid_volume
    (a b c : ℝ)
    (h1 : a / 4 = cos60)
    (h2 : b / 4 = cos60)
    (h3 : c / 4 = cos60)
    (h_diagonal : sqrt (a^2 + b^2 + c^2) = 4)
    :
    a * b * c = 8 :=
by
  sorry

end cuboid_volume_l300_300854


namespace simplified_expression_l300_300557

-- Define the expression to be simplified
def expr := (real.sqrt 5) * (5 ^ (1/2 : ℝ)) + (15 / 3) * 3 - (9 ^ (3/2 : ℝ))

-- State the theorem to verify the simplification
theorem simplified_expression : expr = -7 :=
by
  sorry

end simplified_expression_l300_300557


namespace sin_cos_identity_1_sin_cos_identity_2_l300_300835

variable (α : ℝ)

theorem sin_cos_identity_1 : (sin α)^4 - (cos α)^4 = (sin α)^2 - (cos α)^2 :=
by sorry

theorem sin_cos_identity_2 : (sin α)^4 + (sin α)^2 * (cos α)^2 + (cos α)^2 = 1 :=
by sorry

end sin_cos_identity_1_sin_cos_identity_2_l300_300835


namespace largest_2_digit_prime_factor_of_binom_180_90_l300_300540

theorem largest_2_digit_prime_factor_of_binom_180_90 :
  ∃ (p : ℕ), (nat.prime p) ∧ (10 ≤ p ∧ p < 100) ∧ (3 * p < 180) ∧ 
  (∀ q, nat.prime q ∧ (10 ≤ q ∧ q < 100) ∧ (3 * q < 180) → q ≤ p) ∧ p = 59 :=
sorry

end largest_2_digit_prime_factor_of_binom_180_90_l300_300540


namespace find_y_l300_300751

theorem find_y (y : ℕ) (h : 4 ^ 12 = 64 ^ y) : y = 4 :=
sorry

end find_y_l300_300751


namespace part_one_part_two_l300_300733

-- Definitions for the function f and trigonometric values
def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * sin x * sin x

-- Lean statement for (I)
theorem part_one : f (π / 6) = 0 :=
by sorry

-- Lean statement for (II)
theorem part_two (α : ℝ) (h₁ : α ∈ Ioo 0 π) (h₂ : f (α / 2) = (1/4 - (sqrt 3 / 2))) : sin α = (1 + 3 * sqrt 5) / 8 :=
by sorry

end part_one_part_two_l300_300733


namespace range_of_a_l300_300331

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (x: ℝ) (a: ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x

-- assuming h(x) is a combination of f and g, not explicitly given
def h (x: ℝ) (a: ℝ) : ℝ := f(x) + g(x, a)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc 1 4, deriv (h x a) < 0) ↔ a ∈ Ioi (-1) :=
sorry

end range_of_a_l300_300331


namespace largest_house_number_l300_300879

theorem largest_house_number (phone_digits : List ℕ) (house_digits : List ℕ) 
  (h1 : phone_digits = [3, 4, 6, 2, 8, 9, 0]) 
  (h2 : house_digits.sum = phone_digits.sum)
  (h3 : ∀i j, i ≠ j → house_digits.nth i ≠ house_digits.nth j)
  (h4 : house_digits.length = 4) 
  : house_digits = [9, 8, 7, 6] := 
sorry

end largest_house_number_l300_300879


namespace ratio_traditionalists_progressives_l300_300174

-- Define the given conditions
variables (T P C : ℝ)
variables (h1 : C = P + 4 * T)
variables (h2 : 4 * T = 0.75 * C)

-- State the theorem
theorem ratio_traditionalists_progressives (h1 : C = P + 4 * T) (h2 : 4 * T = 0.75 * C) : T / P = 3 / 4 :=
by {
  sorry
}

end ratio_traditionalists_progressives_l300_300174


namespace cricketer_total_matches_l300_300859

theorem cricketer_total_matches (n : ℕ)
  (avg_total : ℝ) (avg_first_6 : ℝ) (avg_last_4 : ℝ)
  (total_runs_eq : 6 * avg_first_6 + 4 * avg_last_4 = n * avg_total) :
  avg_total = 38.9 ∧ avg_first_6 = 42 ∧ avg_last_4 = 34.25 → n = 10 :=
by
  sorry

end cricketer_total_matches_l300_300859


namespace isosceles_triangle_sides_l300_300991

theorem isosceles_triangle_sides (a b c : ℝ) (hb : b = 3) (hc : a = 3 ∨ c = 3) (hperim : a + b + c = 7) :
  a = 2 ∨ a = 3 ∨ c = 2 ∨ c = 3 :=
by
  sorry

end isosceles_triangle_sides_l300_300991


namespace sam_distance_traveled_l300_300450

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300450


namespace probability_first_die_multiple_of_odd_second_die_l300_300554

theorem probability_first_die_multiple_of_odd_second_die :
  let outcomes := 36
  let favorable_outcomes := 12
  let probability := (favorable_outcomes : ℚ) / outcomes
  probability = 1 / 3 :=
by {
  -- conditions
  let total_outcomes := 6 * 6
  let favorable_outcomes := 12
  let probability := (favorable_outcomes : ℚ) / total_outcomes

  have h1 : total_outcomes = outcomes := rfl,
  have h2 : favorable_outcomes = 12 := rfl,
  have h3 : probability = 1 / 3 := by norm_num [total_outcomes, favorable_outcomes]; exact rfl,  

  -- Proving final probability
  exact h3,
}

end probability_first_die_multiple_of_odd_second_die_l300_300554


namespace expected_winnings_is_correct_l300_300180

noncomputable def expected_value_of_winnings : ℚ :=
  let outcomes := {3, 6}
  let favorable_prob := 2 / 8
  let winning_amounts := {3, 6}
  let expected_winning_amount := (3 / 8 ) + (6 / 8 )
  favorable_prob * (3 + 6)

theorem expected_winnings_is_correct :
  expected_value_of_winnings = 9 / 4 := by
  sorry

end expected_winnings_is_correct_l300_300180


namespace sqrt_sum_comparison_cubic_vs_quadratic_inequality_l300_300648

-- Part 1
theorem sqrt_sum_comparison : 
    sqrt 7 + sqrt 10 > sqrt 3 + sqrt 14 :=
by
  sorry

-- Part 2
theorem cubic_vs_quadratic_inequality (x : ℝ) (h : x > 1) : 
    x^3 > x^2 - x + 1 :=
by
  sorry

end sqrt_sum_comparison_cubic_vs_quadratic_inequality_l300_300648


namespace b_can_finish_work_in_15_days_l300_300584

theorem b_can_finish_work_in_15_days (W : ℕ) (r_A : ℕ) (r_B : ℕ) (h1 : r_A = W / 21) (h2 : 10 * r_B + 7 * r_A / 21 = W) : r_B = W / 15 :=
by sorry

end b_can_finish_work_in_15_days_l300_300584


namespace two_lines_parallel_same_plane_l300_300515

-- Defining the types for lines and planes
variable (Line : Type) (Plane : Type)

-- Defining the relationships similar to the mathematical conditions
variable (parallel_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Defining the non-overlapping relationships between lines (assuming these relations are mutually exclusive)
axiom parallel_or_intersect_or_skew : ∀ (a b: Line), 
  (parallel a b ∨ intersect a b ∨ skew a b)

-- The statement we want to prove
theorem two_lines_parallel_same_plane (a b: Line) (α: Plane) :
  parallel_to_plane a α → parallel_to_plane b α → (parallel a b ∨ intersect a b ∨ skew a b) :=
by
  intro ha hb
  apply parallel_or_intersect_or_skew

end two_lines_parallel_same_plane_l300_300515


namespace total_weight_of_pumpkins_l300_300059

def first_pumpkin_weight : ℝ := 12.6
def second_pumpkin_weight : ℝ := 23.4
def total_weight : ℝ := 36

theorem total_weight_of_pumpkins :
  first_pumpkin_weight + second_pumpkin_weight = total_weight :=
by
  sorry

end total_weight_of_pumpkins_l300_300059


namespace cos_arcsin_l300_300235

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l300_300235


namespace exists_odd_prime_k_l300_300693

variable {a b : ℕ}

-- Define the distance from a real number to the nearest integer
def norm (x : ℝ) : ℝ := abs (x - round x)

theorem exists_odd_prime_k (ha : 0 < a) (hb : 0 < b) :
  ∃ (p : ℕ) (k : ℕ), Prime p ∧ p % 2 = 1 ∧ 0 < k ∧
    norm (a / p^k) + norm (b / p^k) + norm ((a + b) / p^k) = 1 := by
  sorry

end exists_odd_prime_k_l300_300693


namespace min_distance_PQ_triangle_area_h_symmetric_h_range_l300_300806

noncomputable def P (k_1 : ℤ) : ℝ × ℝ := (1/2 + 2 * k_1, 1)
noncomputable def Q (k_2 : ℤ) : ℝ × ℝ := (1 + 2 * k_2, -1)
noncomputable def f (x : ℝ) : ℝ := sin (π * x)
noncomputable def g (x : ℝ) : ℝ := cos (π * x)
noncomputable def h (x : ℝ) : ℝ := cos (π * x)

theorem min_distance_PQ (k : ℤ) : ∀ k_1 k_2, k_1 = k_2 -> dist (P k_1) (Q k_2) = sqrt(17)/2 :=
by sorry

theorem triangle_area : ∃ A B C : ℝ × ℝ, 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  (f A.fst = g A.fst) ∧ 
  (f B.fst = g B.fst) ∧ 
  (f C.fst = g C.fst) ∧ 
  triangle_area A B C = sqrt 2 :=
by sorry

theorem h_symmetric (x : ℝ) : h(x) = cos (π * x) :=
by sorry

theorem h_range : (∀ x ∈ set.Icc (-2 / 3) (1 / 3), h x ≥ -1 / 2 ∧ h x ≤ 1) :=
by sorry

end min_distance_PQ_triangle_area_h_symmetric_h_range_l300_300806


namespace angle_sum_of_octagon_and_triangle_l300_300002

-- Define the problem setup
def is_interior_angle_of_regular_polygon (n : ℕ) (angle : ℝ) : Prop :=
  angle = 180 * (n - 2) / n

def is_regular_octagon_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 8 angle

def is_equilateral_triangle_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 3 angle

-- The statement of the problem
theorem angle_sum_of_octagon_and_triangle :
  ∃ angle_ABC angle_ABD : ℝ,
    is_regular_octagon_angle angle_ABC ∧
    is_equilateral_triangle_angle angle_ABD ∧
    angle_ABC + angle_ABD = 195 :=
sorry

end angle_sum_of_octagon_and_triangle_l300_300002


namespace ratio_of_cream_in_coffees_l300_300895

theorem ratio_of_cream_in_coffees (initial_coffee : ℕ) (tom_drank : ℕ) (cream_added : ℕ):
  let final_tom_cream := (initial_coffee - tom_drank) + cream_added 
  let final_tina_cream := (cream_added * 3 / (initial_coffee + cream_added)) --cream Tina has after drinking
  final_tina_cream to be nat 64/19
  final_ratio := final_tom_cream * 19 / final_tina_cream 
  final_ratio  = nat  19:= sorry

end ratio_of_cream_in_coffees_l300_300895


namespace find_fraction_l300_300332

noncomputable def condition1 := ∀ (a b : ℝ), ∃ (x y : ℝ), ax - by - 3 = 0
noncomputable def condition2 := ∀ (x : ℝ), f(x) = x * exp x
noncomputable def condition3 := ∀ (a b : ℝ), 
  let f' := λ x, exp x + x * exp x in
  let k := f' 1 in -- slope of tangent at P(1, e)
  ax - by - 3 = 0 ∧ ax + by * k = 0 -- perpendicular condition

theorem find_fraction (a b : ℝ) : 
  (condition1 a b ∧ condition2 (1 : ℝ) ∧ condition3 (a) (b)) → 
  (a / b = -1 / (2 * exp 1)) := 
sorry

end find_fraction_l300_300332


namespace angle_between_a_and_b_is_60_l300_300745

variables (a b : ℝ^3)

-- Given conditions as definitions
def cond1 : Prop := (a + 2 • b) ⬝ (a - b) = -6
def cond2 : Prop := ‖a‖ = 1
def cond3 : Prop := ‖b‖ = 2

-- Statement of the problem
theorem angle_between_a_and_b_is_60 (h1 : cond1 a b) (h2 : cond2 a) (h3 : cond3 b) : 
  ∃ θ : ℝ, θ = 60 ∧ θ = real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖)) :=
sorry

end angle_between_a_and_b_is_60_l300_300745


namespace SamDrove200Miles_l300_300440

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300440


namespace minimize_y_l300_300412

noncomputable def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3 * x + 5

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) → x = (2 * a + 2 * b - 3) / 4 := by
  sorry

end minimize_y_l300_300412


namespace perpendicularity_condition_l300_300881

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m + 2) * x + 3 * m * y + 1
def line2 (m : ℝ) (x y : ℝ) : ℝ := (m - 2) * x + (m + 2) * y

-- Define the slopes of the lines when m = 1/2
def slope_line1_for_half : ℝ := - (5 / 3)
def slope_line2_for_half : ℝ := 3 / 5

-- Prove the perpendicularity condition
theorem perpendicularity_condition (m : ℝ) : (line1 m x y) = 0 ∧ (line2 m x y) = 0 → 
  (m = -2 ∨ m = 1 / 2) ∧ (- slope_line1_for_half * slope_line2_for_half = -1) :=
sorry

end perpendicularity_condition_l300_300881


namespace arctan_sum_l300_300809

noncomputable def a : ℚ := 3 / 4
noncomputable def b : ℚ := 2 / 7

theorem arctan_sum:
  (∀ b : ℚ, (a + 1) * (b + 1) = 9 / 4 → arctan a + arctan b = 0.942) :=
begin
  assume b,
  assume h : (a + 1) * (b + 1) = 9 / 4,
  let arctan_sum := arctan a + arctan b,
  sorry
end

end arctan_sum_l300_300809


namespace basketball_team_total_wins_l300_300170

theorem basketball_team_total_wins :
  ∀ (first second third : ℕ),
  first = 40 →
  second = (5 * first / 8) →
  third = first + second →
  first + second + third = 130 :=
by
  intros first second third h1 h2 h3
  rw [h1] at h2
  rw [h1, h2] at h3
  rw [h1, h2, h3]
  sorry

end basketball_team_total_wins_l300_300170


namespace jeff_total_cabinets_l300_300390

/-
  Jeff currently has 3 cabinets.
  He installs twice as many cabinets over 4 different counters, so 2 * 3 * 4 = 24 cabinets.
  Then he installs additional cabinets in the pattern: 3, 5, and 7 over three more counters.
  Finally, he subtracts 2 cabinets he no longer needs.
  We need to prove that the total number of cabinets he has now is 37.
-/

theorem jeff_total_cabinets : 
  let initial_cabinets := 3
  let installed_cabinets := 2 * 3 * 4
  let additional_cabinets := 3 + 5 + 7
  let subtracted_cabinets := 2
  in initial_cabinets + installed_cabinets + additional_cabinets - subtracted_cabinets = 37 := by
    sorry

end jeff_total_cabinets_l300_300390


namespace locally_integrable_implies_l300_300805

-- Conditions: 
-- 1. f is a measurable function from ℝ to ℝ
-- 2. f(x + t) - f(x) is locally integrable for every t

def measurable (f : ℝ → ℝ) : Prop := sorry
def locally_integrable (f : ℝ → ℝ) : Prop := sorry
def locally_integrable_diff (f : ℝ → ℝ) (t : ℝ) : Prop := locally_integrable (λ x, f (x + t) - f x)

theorem locally_integrable_implies (f : ℝ → ℝ) 
  (hf_measurable: measurable f) 
  (hf_li_diff: ∀ t, locally_integrable_diff f t) : locally_integrable f := 
sorry

end locally_integrable_implies_l300_300805


namespace f_at_6_5_l300_300719

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem f_at_6_5:
  (∀ x : ℝ, f (x + 2) = -1 / f x) →
  even_function f →
  specific_values f →
  f 6.5 = -0.5 :=
by
  sorry

end f_at_6_5_l300_300719


namespace complete_square_formula_method_thorough_factorization_x_minus_1_thorough_factorization_x_plus_3_l300_300837
open polynomial -- Open the polynomial namespace if necessary

-- Define the conditions and theorems in Lean 4

theorem complete_square_formula_method (x : ℝ) :
  let y := x^2 - 2 * x in
  (y^2 + 2 * y + 1) = ((y + 1)^2) :=
by
  sorry

theorem thorough_factorization_x_minus_1 (x : ℝ) :
  (x^2 - 2 * x + 1)^2 = (x - 1)^4 :=
by
  sorry

theorem thorough_factorization_x_plus_3 (x : ℝ) :
  let y := x^2 + 6 * x in
  y * (y + 18) + 81 = (x + 3)^4 :=
by
  sorry

end complete_square_formula_method_thorough_factorization_x_minus_1_thorough_factorization_x_plus_3_l300_300837


namespace average_age_of_women_l300_300935

theorem average_age_of_women (A : ℕ) (W1 W2 : ℕ) 
  (h1 : 7 * A - 26 - 30 + W1 + W2 = 7 * (A + 4)) : 
  (W1 + W2) / 2 = 42 := 
by 
  sorry

end average_age_of_women_l300_300935


namespace sam_driving_distance_l300_300419

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300419


namespace mrs_wilsborough_tickets_l300_300067

theorem mrs_wilsborough_tickets :
  ∀ (saved vip_ticket_cost regular_ticket_cost vip_tickets left : ℕ),
    saved = 500 →
    vip_ticket_cost = 100 →
    regular_ticket_cost = 50 →
    vip_tickets = 2 →
    left = 150 →
    (saved - left - (vip_tickets * vip_ticket_cost)) / regular_ticket_cost = 3 :=
by
  intros saved vip_ticket_cost regular_ticket_cost vip_tickets left
  sorry

end mrs_wilsborough_tickets_l300_300067


namespace fractional_part_friends_money_l300_300418

-- Conditions in terms of the Lean statement.
variable {Loki Moe Nick Ott Pam : Type}
variable {money_Loki money_Moe money_Nick money_Ott money_Pam : ℕ}

-- Initial monies
def initial_money (Loki Nick Moe: ℕ) := 12 * money_Moe + 10 * money_Loki + 8 * money_Nick

-- Transfers
def transfer (x : ℕ) := 
  money_Ott + 6 * x = 0 ∧ 
  money_Pam + 6 * x = 0 ∧ 
  money_Moe + 12 * x = 0 ∧ 
  money_Loki + 10 * x = 0 ∧ 
  money_Nick + 8 * x = 0

-- Initial total money
def total_initial_money (Loki Nick Moe : ℕ) := money_Loki + money_Nick + money_Moe

-- Final combined money for Ott and Pam
def combined_money (money_Ott money_Pam : ℕ) := money_Ott + money_Pam

-- Fraction of combined money of Ott and Pam
def fraction (initial_money combined_money : ℕ) := combined_money * 5 = initial_money * 2

theorem fractional_part_friends_money 
  (Loki Nick Moe money_Loki money_Nick money_Moe money_Ott money_Pam x : ℕ)
  (h1 : initial_money = total_initial_money 12 10 8)
  (h2 : transfer x)
  (h3 : combined_money = money_Ott + money_Pam)
  : fraction initial_money combined_money := sorry

end fractional_part_friends_money_l300_300418


namespace sam_distance_traveled_l300_300451

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300451


namespace pages_written_in_a_year_l300_300011

theorem pages_written_in_a_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) :
  pages_per_letter = 3 → friends = 2 → times_per_week = 2 → weeks_per_year = 52 → 
  pages_per_letter * friends * times_per_week * weeks_per_year = 624 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end pages_written_in_a_year_l300_300011


namespace verify_addition_by_subtraction_l300_300536

theorem verify_addition_by_subtraction (a b c : ℤ) (h : a + b = c) : (c - a = b) ∧ (c - b = a) :=
by
  sorry

end verify_addition_by_subtraction_l300_300536


namespace solution_set_f_le_2_l300_300731

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < 2 then 2 - (Real.log 2 (-x + 2) / Real.log 2 2)
  else if -2 < x ∧ x < 0 then 2 - f (-x)
  else 0 -- define a default for domain outside -2 < x < 2

theorem solution_set_f_le_2 :
  ∀ (x : ℝ), f x ≤ 2 ↔ (-2 < x ∧ x ≤ 1) ∨ (x < 0 ∧ x > -2) := 
by
  intros,
  sorry

end solution_set_f_le_2_l300_300731


namespace points_concyclic_l300_300214

open EuclideanGeometry

noncomputable theory

variables {A B C E F M N P Q : Point}
variables (h1 : Altitude E B A C)
variables (h2 : Altitude F C B A)
variables (circle1 : ∃ M N, CircleDiameterIntersectsLine A B F M N)
variables (circle2 : ∃ P Q, CircleDiameterIntersectsLine A C E P Q)

theorem points_concyclic (h1 : Altitude E B A C)
                        (h2 : Altitude F C B A)
                        (circle1 : ∃ M N, CircleDiameterIntersectsLine A B F M N)
                        (circle2 : ∃ P Q, CircleDiameterIntersectsLine A C E P Q) :
  Concyclic M N P Q :=
sorry

end points_concyclic_l300_300214


namespace problem_i_problem_ii_l300_300732

-- (I) Proving the maximum value of the function in a given interval
theorem problem_i (f : ℝ → ℝ) (hx : ∀ x, f x = 2 * cos x * (sin x + cos x) - 1) :
  ∃ x ∈ set.Icc 0 (π / 4), f x = sqrt 2 :=
sorry

-- (II) Proving the range of values for 'b' in the triangle ABC
theorem problem_ii (f : ℝ → ℝ) (B : ℝ) (a c b : ℝ)
  (hf : ∀ x, f x = 2 * cos x * (sin x + cos x) - 1)
  (h_f_eq : f (3 / 4 * B) = 1) (h_a_c : a + c = 2) :
  1 ≤ b ∧ b < 2 :=
sorry

end problem_i_problem_ii_l300_300732


namespace locus_of_A2_l300_300407

variable {ABC : Triangle}
variable {I : Point} [IsIncenter I ABC]
variable {A A1 A2 : Point}
variable (isSymmetricToIncenter : IsSymmetric A A1 I)
variable (isIsotomicallyConjugate : IsIsotomicallyConjugate A1 A2 ABC)

theorem locus_of_A2 : IsOnRadicalAxis A2 (Circumcircle ABC) (Incircle ABC) := by
  sorry

end locus_of_A2_l300_300407


namespace Tim_earnings_correct_l300_300128

def visitors_day := 100
def first_6_days := 6 * visitors_day
def last_day := 2 * first_6_days
def total_visitors := first_6_days + last_day
def earnings_per_visitor := 0.01
def total_earnings := total_visitors * earnings_per_visitor

theorem Tim_earnings_correct :
  total_earnings = 18 := by
  sorry

end Tim_earnings_correct_l300_300128


namespace find_x_minus_y_l300_300511

def rotated_point (x y h k : ℝ) : ℝ × ℝ := (2 * h - x, 2 * k - y)

def reflected_point (x y : ℝ) : ℝ × ℝ := (y, x)

def transformed_point (x y : ℝ) : ℝ × ℝ :=
  reflected_point (rotated_point x y 2 3).1 (rotated_point x y 2 3).2

theorem find_x_minus_y (x y : ℝ) (h1 : transformed_point x y = (4, -1)) : x - y = 3 := 
by 
  sorry

end find_x_minus_y_l300_300511


namespace sum_powers_of_i_l300_300845

theorem sum_powers_of_i : (∑ k in Finset.range 2048, (complex.I^k)) = 0 :=
by
  sorry

end sum_powers_of_i_l300_300845


namespace average_of_six_integers_find_y_min_value_of_four_integers_less_than_100_l300_300163

-- Part (a)
theorem average_of_six_integers :
  let nums := [22, 23, 23, 25, 26, 31]
  (nums.sum / nums.length) = 25 := sorry

-- Part (b)
theorem find_y :
  (let y := Nat
  (y + 7 + 2y - 9 + 8y + 6) / 3 = 27) → y = 7 := sorry

-- Part (c)
theorem min_value_of_four_integers_less_than_100 :
  ∀ a b c d : Nat, a < 100 → b < 100 → c < 100 → d < 100 →
  (a + b + c + d) / 4 = 94 → 
  min a (min b (min c d)) = 79 := sorry

end average_of_six_integers_find_y_min_value_of_four_integers_less_than_100_l300_300163


namespace complex_conjugate_quadrant_l300_300753

open Complex

theorem complex_conjugate_quadrant (z : ℂ) (h : (1 + I) / z = 2 - I) : 
  let z_conj := conj z in (0 < z_conj.re ∧ z_conj.im < 0) :=
sorry

end complex_conjugate_quadrant_l300_300753


namespace solve_for_N_l300_300883

theorem solve_for_N (N : ℤ) (h1 : N < 0) (h2 : 2 * N * N + N = 15) : N = -3 :=
sorry

end solve_for_N_l300_300883


namespace dubblefud_chip_product_l300_300378

theorem dubblefud_chip_product (B G : ℕ) (h1 : B = G) : 
    let yellow_points := 2
    let blue_points := 4
    let green_points := 5
    let num_yellow := 4
    let total_yellow_value := num_yellow * yellow_points
    let total_blue_value := B * blue_points
    let total_green_value := G * green_points
    total_yellow_value * (total_blue_value + total_green_value) = 72 * B :=
by
  -- Definition declarations
  let yellow_points := 2
  let blue_points := 4
  let green_points := 5
  let num_yellow := 4

  -- Computing total values
  let total_yellow_value := num_yellow * yellow_points
  have h2 : total_yellow_value = 8 := by sorry
  let total_blue_value := B * blue_points
  have h3 : total_blue_value = B * 4 := by sorry
  let total_green_value := G * green_points
  have h4 : total_green_value = G * 5 := by sorry

  -- Using condition B = G
  rw h1 at h3 h4
  have h5 : total_yellow_value * (total_blue_value + total_green_value) = 8 * (B * 9) := by sorry
  exact h5.trans (by ring)

end dubblefud_chip_product_l300_300378


namespace sum_a_b_l300_300095

-- Define the variables a, b, and the imaginary unit i in the context of complex numbers.
variables (a b : ℝ) (i : ℂ)
noncomputable def i := complex.I

-- Define the condition given in the problem.
def complex_eq : Prop := (a + 3 * i) + (2 - i) = 5 + b * i

-- State the main theorem to be proven
theorem sum_a_b (h : complex_eq a b) : a + b = 5 :=
sorry

end sum_a_b_l300_300095


namespace fraction_of_5100_l300_300579

theorem fraction_of_5100 (x : ℝ) (h : ((3 / 4) * x * (2 / 5) * 5100 = 765.0000000000001)) : x = 0.5 :=
by
  sorry

end fraction_of_5100_l300_300579


namespace proof_of_competition_results_l300_300225

def scores_team_a : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def scores_team_b : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]

def median (l : List ℝ) : ℝ :=
  let sorted := l.sort
  if l.length % 2 = 0 then
    (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2
  else
    sorted.get! (l.length / 2)

def mode (l : List ℝ) : ℝ :=
  l.foldl (λ acc x => if l.count x > acc.1 then (l.count x, x) else acc) (0, 0).2

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let avg := average l
  (l.map (λ x => (x - avg) ^ 2)).sum / l.length

theorem proof_of_competition_results : median scores_team_a = 9.5 ∧ mode scores_team_b = 10 ∧ average scores_team_b = 9 ∧ variance scores_team_b = 1 ∧ 1 < 1.4 :=
by
  sorry

end proof_of_competition_results_l300_300225


namespace comp_1_sub_i_pow4_l300_300649

theorem comp_1_sub_i_pow4 : (1 - complex.I)^4 = -4 := by
  sorry

end comp_1_sub_i_pow4_l300_300649


namespace sufficient_not_necessary_condition_l300_300158

theorem sufficient_not_necessary_condition (a : ℝ)
  : (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, a ≥ 0 ∨ a * x^2 + x + 1 ≥ 0)
:= sorry

end sufficient_not_necessary_condition_l300_300158


namespace at_least_one_admitted_prob_l300_300132

theorem at_least_one_admitted_prob (pA pB : ℝ) (hA : pA = 0.6) (hB : pB = 0.7) (independent : ∀ (P Q : Prop), P ∧ Q → P ∧ Q):
  (1 - ((1 - pA) * (1 - pB))) = 0.88 :=
by
  rw [hA, hB]
  -- more steps would follow in a complete proof
  sorry

end at_least_one_admitted_prob_l300_300132


namespace contractor_absent_days_l300_300178

theorem contractor_absent_days (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
by
  sorry

end contractor_absent_days_l300_300178


namespace path_of_B_l300_300992

theorem path_of_B (B C : ℝ) (hBC : B + C = 1) (hArc : arc_len := 1 * π) :
  ∃ l : ℝ, l = 2 :=
by
  let total_rotation_radius := 2 * π / 2
  have total_path := 2 
  exact total_path
  sorry

end path_of_B_l300_300992


namespace constant_term_is_correct_l300_300498

noncomputable def constant_term_in_expansion : ℕ :=
  let f := λ x : ℤ, (2 + (1 / x^2)) * (1 - x)^6 in
  -- The constant term equals 17
  17

theorem constant_term_is_correct :
  constant_term_in_expansion = 17 :=
by
  sorry

end constant_term_is_correct_l300_300498


namespace solve_for_x_l300_300413

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def h (x : ℝ) : ℝ := 4 * (Function.inverse f x)

theorem solve_for_x : ∃ (x : ℝ), h x = 20 ∧ x = 30 / 7 := by
  have h_eq : h (30 / 7) = 20 := by
    -- Skipping the evaluation and proof details
    sorry
  use 30 / 7
  split
  · exact h_eq
  · rfl

end solve_for_x_l300_300413


namespace sam_drove_200_miles_l300_300426

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300426


namespace problem_statement_l300_300319

/-
Given conditions: 
a. The area of triangle ABC is S.
b. vector AB dot vector AC = S.
c. ∠B = π/4.
d. The length |vector CA - vector CB| = 6.
We need to prove:
1. sin A = 2√5 / 5
2. cos A = √5 / 5
3. tan 2A = -4 / 3
4. The area S of triangle ABC is 12.
-/

noncomputable def sin_value (A : ℝ) : ℝ := 2 * real.sqrt 5 / 5
noncomputable def cos_value (A : ℝ) : ℝ := real.sqrt 5 / 5
noncomputable def tan_2A_value (A : ℝ) : ℝ := -4 / 3

theorem problem_statement (A B C : ℝ) (S : ℝ) (u v : E) [NormedSpace ℝ E]
  (h1 : ∥u∥ * ∥v∥ * real.sin A = S)
  (h2 : ∥u - v∥ = 6)
  (h3 : B = π / 4) :
  real.sin A = sin_value A ∧
  real.cos A = cos_value A ∧
  real.tan (2 * A) = tan_2A_value A ∧
  S = 12 :=
by
  sorry

end problem_statement_l300_300319


namespace units_digit_of_k_l300_300758

theorem units_digit_of_k (k : ℕ) (hk : k > 1) (α : ℂ) (hα : α^2 - k * α + 1 = 0)
    (hn : ∀ n : ℕ, n > 10 → ((α^n + α^(-2^n)) % 10 = 7)) : 
    (k % 10 = 3) ∨ (k % 10 = 5) ∨ (k % 10 = 7) := 
sorry

end units_digit_of_k_l300_300758


namespace James_has_43_Oreos_l300_300009

variable (J : ℕ)
variable (James_Oreos : ℕ)

-- Conditions
def condition1 : Prop := James_Oreos = 4 * J + 7
def condition2 : Prop := J + James_Oreos = 52

-- The statement to prove: James has 43 Oreos given the conditions
theorem James_has_43_Oreos (h1 : condition1 J James_Oreos) (h2 : condition2 J James_Oreos) : James_Oreos = 43 :=
by
  sorry

end James_has_43_Oreos_l300_300009


namespace range_of_third_side_l300_300727

theorem range_of_third_side (y : ℝ) : (2 < y) ↔ (y < 8) :=
by sorry

end range_of_third_side_l300_300727


namespace large_circle_circumference_l300_300094

-- Define constants for the problem
def C1 : ℝ := 396 -- circumference of the smaller circle
def area_difference : ℝ := 26960.847359767075 -- difference between the areas

-- Define pi
noncomputable def π : ℝ := Real.pi

-- Define r, the radius of the smaller circle
noncomputable def r : ℝ := C1 / (2 * π)

-- Define the property to be proven
theorem large_circle_circumference : 
  let R := Real.sqrt ((area_difference / π) + r^2) in
  let C2 := 2 * π * R in
  C2 ≈ 703.716 := sorry

end large_circle_circumference_l300_300094


namespace find_smallest_n_mod_500_l300_300284

def sum_of_digits_in_base (n : ℕ) (b : ℕ) : ℕ :=
  (n.to_digits b).sum

def f (n : ℕ) : ℕ :=
  sum_of_digits_in_base n 3

def g (n : ℕ) : ℕ :=
  sum_of_digits_in_base (f n) 6

def base_twelve_contains_non_decimal_digit (n : ℕ) : ℕ :=
  if (n.to_digits 12).any (λ d => d ≥ 10) then 1 else 0

theorem find_smallest_n_mod_500 :
  let n := 32
  in base_twelve_contains_non_decimal_digit (g n) = 1 ∧ n % 500 = 32 :=
  by
    sorry

end find_smallest_n_mod_500_l300_300284


namespace who_visited_beijing_l300_300386

-- Definitions of the conditions
axiom A_statement : (A_has_been_to_Shanghai : Prop) ∧ (B_has_been_to_Shanghai : Prop) ∧ (C_has_been_to_Beijing : Prop)
axiom B_statement : (B_has_been_to_Shanghai : Prop) ∧ ¬(A_statement)
axiom C_statement : (C_has_been_to_Beijing : Prop) ∧ B_statement
axiom one_incorrect : (A_incorrect : (¬ A_statement)) ∨ (B_incorrect : (¬ B_statement)) ∨ (C_incorrect : (¬ C_statement)) ∧
  (∀ (a b c : Prop), ¬ (a ∧ b ∧ c))

-- The proof problem
theorem who_visited_beijing (A_has_been_to_Shanghai : Prop) (B_has_been_to_Shanghai : Prop) (C_has_been_to_Beijing : Prop)
  (one_incorrect : (A_incorrect : ¬ (A_has_been_to_Shanghai ∧ B_has_been_to_Shanghai ∧ C_has_been_to_Beijing)) ∨ 
  (B_incorrect : ¬ (B_has_been_to_Shanghai ∧ ¬ (A_has_been_to_Shanghai ∧ B_has_been_to_Shanghai ∧ C_has_been_to_Beijing))) ∨ 
  (C_incorrect : ¬ (C_has_been_to_Beijing ∧ (B_has_been_to_Shanghai ∧ ¬ (A_has_been_to_Shanghai ∧ B_has_been_to_Shanghai ∧ C_has_been_to_Beijing)))) ∧ 
  (∀ (a b c : Prop), ¬ (a ∧ b ∧ c))) : 
  (A_has_been_to_Beijing ∧ C_has_been_to_Beijing) :=
  sorry

end who_visited_beijing_l300_300386


namespace sam_distance_l300_300458

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300458


namespace converse_implication_l300_300863

theorem converse_implication (a : ℝ) : (a^2 = 1 → a = 1) → (a = 1 → a^2 = 1) :=
sorry

end converse_implication_l300_300863


namespace shaded_area_correct_l300_300611

-- Define the dimensions of the rectangle
def length : ℝ := 12
def width : ℝ := 8

-- Define the radius of the quarter circles
def radius : ℝ := 4

-- Calculate the area of the rectangle
def area_rectangle : ℝ := length * width

-- Calculate the area of four quarter circles (which forms a full circle)
def area_circle : ℝ := Real.pi * radius^2

-- Shaded area
def shaded_area : ℝ := area_rectangle - area_circle

-- Statement of the problem: Prove that shaded region area is 96 - 16π cm²
theorem shaded_area_correct : shaded_area = 96 - 16 * Real.pi := by
  sorry

end shaded_area_correct_l300_300611


namespace find_parabola_equation_l300_300886

noncomputable def parabola_equation : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ y^2 = 2 * a * x ∧ (∃ (c : ℝ), 2 * |c| = 8 → c = ± 4)

theorem find_parabola_equation :
  ∃ a : ℝ, a ≠ 0 ∧ y^2 = 2 * a * x ∧
  let f := (a / 2, 0) in
  let chord := abs(a) in
  chord * 2 = 8 →
  y^2 = 8 * x ∨ y^2 = -8 * x :=
sorry

end find_parabola_equation_l300_300886


namespace right_triangle_midpoint_distance_l300_300374

theorem right_triangle_midpoint_distance :
  ∀ (D E F M : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace M] 
    (distance_DE : Real) (distance_DF : Real) (distance_EF : Real)
    (right_triangle_DEF : triangle DEF ∧ right_triangle_at D vertex) -- This formalizes the right triangle condition
    (DE_midpoint_M : is_midpoint M D E),  -- Point M is the midpoint of DE
  distance_DE = 15 → distance_DF = 9 → distance_EF = 12 → distance F M = 7.5 :=
by
  sorry

end right_triangle_midpoint_distance_l300_300374


namespace area_of_hexagon_ABQCDP_l300_300025

variables (A B C D P Q : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space Q]

def trapezoid (A B C D : Type) := (metric_space.is_parallel B D)

def area_of_hexagon (hexagon : Type) := num.real

-----------------------
theorem area_of_hexagon_ABQCDP (A B C D P Q : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] [metric_space Q] 
  (trapezoid_ABCD : trapezoid A B C D)
  (AB_parallel_CD : metric_space.is_parallel A B) 
  (AB_eq_11 : metric_space.is_length_eq A B 11)
  (BC_eq_5 : metric_space.is_length_eq B C 5)
  (CD_eq_19 : metric_space.is_length_eq C D 19)
  (DA_eq_7 : metric_space.is_length_eq D A 7)
  (P_is_bisector_AD : metric_space.bisector_intersection P A C D)
  (Q_is_bisector_BC : metric_space.bisector_intersection Q B C D) :
  area_of_hexagon (A B Q C D P) = 30 * real.sqrt 3 :=
sorry

end area_of_hexagon_ABQCDP_l300_300025


namespace hemisphere_surface_area_l300_300522

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) (hs : ∀ r : ℝ, 4 * Real.pi * r^2 = 4 * Real.pi * r^2) : 
  100 * Real.pi + (4 * Real.pi * r^2 / 2) = 300 * Real.pi :=
by
  rw [h]
  have hs' := hs 10
  rw [mul_pow, hs', pow_two, mul_assoc, mul_div_cancel_left] { occs := occurrences.pos [3] }
  ring
  norm_num
  sorry

end hemisphere_surface_area_l300_300522


namespace elaine_earnings_increase_l300_300799

variable (E : ℝ) (P : ℝ)

theorem elaine_earnings_increase
  (h1 : E > 0) 
  (h2 : 0.30 * E * (1 + P / 100) = 1.80 * 0.20 * E) : 
  P = 20 :=
by
  sorry

end elaine_earnings_increase_l300_300799


namespace probability_green_or_purple_l300_300547

theorem probability_green_or_purple
    (green purple orange : ℕ) 
    (h_green : green = 5) 
    (h_purple : purple = 4) 
    (h_orange : orange = 6) :
    (green + purple) / (green + purple + orange) = 3 / 5 :=
by
  sorry

end probability_green_or_purple_l300_300547


namespace sum_of_valid_a_l300_300764

theorem sum_of_valid_a : 
  ∀ (a : ℤ), 
  (∀ x : ℤ, 5 * x ≥ 3 * (x + 2) ∧ x - (x + 3)/2 ≤ a/16) →
  (∃ x1 x2 : ℤ, (5 * x1 ≥ 3 * (x1 + 2)) ∧ (x1 - (x1 + 3)/2 ≤ a/16) ∧ (5 * x2 ≥ 3 * (x2 + 2)) ∧ (x2 - (x2 + 3)/2 ≤ a/16) ∧ x1 ≠ x2) →
  (∃ y : ℤ, 5 + a * y = 2 * y - 7 ∧ y < 0) → 
  (a = 8 ∨ a = 14) → 
  ∑ a in {8, 14}, a = 22 :=
sorry

end sum_of_valid_a_l300_300764


namespace complex_conjugate_magnitude_l300_300051

theorem complex_conjugate_magnitude (z : ℂ) (h : z * complex.I + 1 = z) : complex.abs (conj z) = real.sqrt 2 / 2 :=
by sorry

end complex_conjugate_magnitude_l300_300051


namespace eval_expression_l300_300267

theorem eval_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end eval_expression_l300_300267


namespace unique_top_field_l300_300900

def valid_labelling (labelling : Fin 9 → ℕ) : Prop :=
  (∀ i j, i ≠ j → labelling i ≠ labelling j) ∧
  ( ∑ i, labelling i = 45) ∧ -- sum of 1 to 9
  -- replace line_sum with the actual conditions based on provided figure
  ( ∀ {lst : List (Fin 9)}, line_sum lst labelling )

theorem unique_top_field (labelling : Fin 9 → ℕ) (h : valid_labelling labelling) :
  labelling 0 = 9 :=
sorry

end unique_top_field_l300_300900


namespace sam_distance_l300_300456

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300456


namespace coefficient_x4_in_expansion_l300_300905

theorem coefficient_x4_in_expansion : 
  (∑ k in finset.range 9, (nat.choose 8 k) * x ^ (8 - k) * (real.sqrt 5) ^ k).coeff 4 = 1750 := 
by {
  sorry -- the proof details are in the solution steps but are not required to be included here
}

end coefficient_x4_in_expansion_l300_300905


namespace find_slope_l300_300056

theorem find_slope 
  (m : ℝ)
  (p_eq : ∀ (x : ℝ), p_eq x = 2 * x + 3)
  (q_eq : ∀ (x : ℝ), q_eq x = m * x + 1)
  (intersection_point : p_eq 1 = 5 ∧ q_eq 1 = 5) :
  m = 4 :=
sorry

end find_slope_l300_300056


namespace function_solution_l300_300021

def real (a : ℝ) (f : ℝ → ℝ) : Prop := ∀ (x : ℝ),
  (∫ f(x)^a dx) = (∫ f(x) dx)^a

-- Prove that the function f satisfying the above condition is
theorem function_solution (a : ℝ) (f : ℝ → ℝ) : (∫ f(x)^a dx = (∫ f(x) dx)^a) → 
  ((a ≠ 1) → ∃ k C : ℝ, ∀ x, f(x) = k * C * exp(k * x)) ∧
  (a = 1 → ∀ x, true) := 
sorry

end function_solution_l300_300021


namespace find_physics_marks_l300_300976

variable (P C M : ℕ)

theorem find_physics_marks
  (h1 : P + C + M = 225)
  (h2 : P + M = 180)
  (h3 : P + C = 140) : 
  P = 95 :=
by
  sorry

end find_physics_marks_l300_300976


namespace inclination_angle_l300_300505

-- Define the line equation
def line_eq (x y : ℝ) : Prop := ∃ c : ℝ, c = 3 ∧ (sqrt 3) * x - y - c = 0

-- Define the inclination angle theorem
theorem inclination_angle {x y : ℝ} (h : line_eq x y) : angle := 
  ∃ θ : ℝ, θ = 60 ∧ θ ∈ set.Ico 0 180 ∧ tan θ = sqrt 3 := sorry

end inclination_angle_l300_300505


namespace five_natural_numbers_increase_15_times_l300_300787

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l300_300787


namespace function_example_l300_300183

-- Define a function f
def is_function_passing_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem function_example (f : ℝ → ℝ) :
  (is_function_passing_through f (1, 3)) ∧ (is_increasing f) → ∃ k b : ℝ, k > 0 ∧ f = λ x, k * x + b :=
sorry

end function_example_l300_300183


namespace set_theory_problem_l300_300941

def U : Set ℤ := {x ∈ Set.univ | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_theory_problem : 
  (A ∩ B = {4}) ∧ 
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (U \ (A ∪ C) = {6, 8, 10}) ∧ 
  ((U \ A) ∩ (U \ B) = {3}) := 
by 
  sorry

end set_theory_problem_l300_300941


namespace arthur_segments_l300_300993

theorem arthur_segments (total_length_cm drawn_segments : ℕ) (h1 : total_length_cm = 400) (h2 : drawn_segments = 7) :
  let n := Nat.floor (Int.sqrt 400 - 1) in
  let remaining_segments := 2 * n + 1 - drawn_segments in
  remaining_segments = 32 :=
by
  sorry

end arthur_segments_l300_300993


namespace rectangle_area_l300_300000

-- Define the conditions
def radius_larger : ℝ := 1
def radius_smaller : ℝ := 1 / 2
def width_rectangle : ℝ := 2
def base_triangle : ℝ := 1
def equal_sides_triangle : ℝ := (3 : ℝ) / 2
def height_triangle : ℝ := Real.sqrt 2
def length_rectangle : ℝ := (3 + 2 * (Real.sqrt 2)) / 2

-- Define the goal
theorem rectangle_area
  (radius_larger : ℝ = 1) 
  (radius_smaller : ℝ = 1 / 2) 
  (width_rectangle : ℝ = 2)
  (base_triangle : ℝ = 1)
  (equal_sides_triangle : ℝ = (3 : ℝ) / 2)
  (height_triangle : ℝ = Real.sqrt 2)
  (length_rectangle : ℝ = (3 + 2 * (Real.sqrt 2)) / 2) :
  width_rectangle * length_rectangle = 3 + 2 * Real.sqrt 2 :=
by 
  sorry

end rectangle_area_l300_300000


namespace find_m_n_f_monotonic_find_k_l300_300301

-- Given function definitions
def f (x : ℝ) (n : ℝ) (m : ℝ) : ℝ := (3 ^ x + n) / (3 ^ (x + 1) + m)

-- Part (1): Proving m = 3 and n = -1 for f to be an odd function
theorem find_m_n (x : ℝ) 
  (hx : ∀ x, f(x, n, m) = -f(-x, n, m)) :
  m = 3 ∧ n = -1 := 
sorry

-- Part (2): Proving that f is monotonic (increasing)
theorem f_monotonic (x1 x2 : ℝ)
  (h1 : x1 < x2) 
  (h2 : ∀ {n m}, n = -1 ∧ m = 3) : 
  f(x1, -1, 3) < f(x2, -1, 3) := 
sorry

-- Part (3): Proving range of k for which f(kx^2) + f(2x - 1) >0 for x in [1/3, 2]
theorem find_k (k x : ℝ)
  (hx : x ∈ Icc (1/3 : ℝ) 2) 
  (h2 : ∀ {n m}, n = -1 ∧ m = 3) : 
  (3 < k) -> f(k * x^2, -1 , 3) + f(2 * x - 1, -1 , 3) > 0 := 
sorry

end find_m_n_f_monotonic_find_k_l300_300301


namespace median_team_a_mode_team_b_avg_team_b_var_team_b_neater_scores_l300_300226

def team_a_scores : list ℕ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : list ℕ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]
def variance_team_a_scores : ℝ := 1.4

theorem median_team_a : 
  (list.median team_a_scores) = 9.5 := 
by sorry

theorem mode_team_b : 
  (list.mode team_b_scores) = 10 := 
by sorry

theorem avg_team_b : 
  (list.sum team_b_scores : ℝ) / (list.length team_b_scores : ℝ) = 9 := 
by sorry

theorem var_team_b : 
  let avg := (list.sum team_b_scores : ℝ) / (list.length team_b_scores : ℝ) in
  (list.sum (list.map (λ x, (x - avg) ^ 2) team_b_scores) / (list.length team_b_scores : ℝ)) = 1 := 
by sorry

theorem neater_scores : 
  variance_team_a_scores > 1 := 
by sorry

end median_team_a_mode_team_b_avg_team_b_var_team_b_neater_scores_l300_300226


namespace total_points_scored_l300_300080

-- Define the points scored by Sam and his friend
def points_scored_by_sam : ℕ := 75
def points_scored_by_friend : ℕ := 12

-- The main theorem stating the total points
theorem total_points_scored : points_scored_by_sam + points_scored_by_friend = 87 := by
  -- Proof goes here
  sorry

end total_points_scored_l300_300080


namespace ice_cream_ratio_l300_300477

theorem ice_cream_ratio
    (T : ℕ)
    (W : ℕ)
    (hT : T = 12000)
    (hMultiple : ∃ k : ℕ, W = k * T)
    (hTotal : T + W = 36000) :
    W / T = 2 :=
by
  -- Proof is omitted, so sorry is used
  sorry

end ice_cream_ratio_l300_300477


namespace water_height_in_cylinder_l300_300209

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def height_of_cylinder (V R : ℝ) : ℝ :=
  V / (real.pi * R^2)

theorem water_height_in_cylinder :
  let r := 15 in
  let h_c := 25 in
  let R := 18 in
  let V := volume_of_cone r h_c in
  height_of_cylinder V R ≈ 5.8 :=
by
  -- Given definitions and assumptions
  let r := 15
  let h_c := 25
  let R := 18
  let V := volume_of_cone r h_c
  show height_of_cylinder V R ≈ 5.8
  sorry

end water_height_in_cylinder_l300_300209


namespace amount_of_H2O_formed_l300_300274

-- Chemical reaction definition
def balanced_reaction (hcl caCO3 h2O : ℕ) : Prop :=
  2 * hcl + caCO3 = CaCO3 + h2O + CO2

-- Given conditions
def hcl_moles : ℕ := 6
def caCO3_moles : ℕ := 3
def required_h2O_moles : ℕ := 3

-- Main proof statement
theorem amount_of_H2O_formed :
  2 * hcl_moles = caCO3_moles →
  balanced_reaction hcl_moles caCO3_moles required_h2O_moles →
  required_h2O_moles = 3 :=
by
  sorry

end amount_of_H2O_formed_l300_300274


namespace almeriense_polynomial_l300_300964

def is_almeriense (p : Polynomial ℝ) : Prop :=
  ∃ a b : ℝ, p = Polynomial.C a * Polynomial.X^3 + Polynomial.C (a * Polynomial.X^2) + Polynomial.C (b * Polynomial.X) + Polynomial.C a

theorem almeriense_polynomial (p : Polynomial ℝ) (h : is_almeriense p) (hzero : p.eval (7/4) = 0) :
  p = Polynomial.C (-21/4) * Polynomial.X^3 + Polynomial.C (73/8) * Polynomial.X^2 - Polynomial.C (21/4) * Polynomial.X 
  ∨ p = Polynomial.C (-291/56) * Polynomial.X^3 + Polynomial.C (14113/1568) * Polynomial.X^2 - Polynomial.C (291/56) * Polynomial.X :=
sorry

end almeriense_polynomial_l300_300964


namespace possible_rankings_l300_300491

theorem possible_rankings (A B C D E : Type)
  (competes : List (A × B × C × D × E))
  (h1 : ∀ r ∈ competes, r.1 ≠ 1 ∧ r.2 ≠ 1)
  (h2 : ∀ r ∈ competes, r.2 ≠ 5) :
  List.length competes = 54 := 
sorry

end possible_rankings_l300_300491


namespace daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l300_300950

-- Definitions based on given conditions
noncomputable def purchase_price : ℝ := 30
noncomputable def max_selling_price : ℝ := 55
noncomputable def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 140

-- Definition of daily profit based on selling price x
noncomputable def daily_profit (x : ℝ) : ℝ := (x - purchase_price) * daily_sales_volume x

-- Lean 4 statements for the proofs
theorem daily_profit_at_35_yuan : daily_profit 35 = 350 := sorry

theorem selling_price_for_600_profit : ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ daily_profit x = 600 ∧ x = 40 := sorry

theorem selling_price_impossible_for_900_profit :
  ∀ x, 30 ≤ x ∧ x ≤ 55 → daily_profit x ≠ 900 := sorry

end daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l300_300950


namespace power_function_odd_condition_l300_300334

def sufficient_condition_for_odd_function (m n : ℤ) : Prop :=
  m = 1 ∧ n = 3

theorem power_function_odd_condition (m n : ℤ) :
  (∀ x : ℝ, f x = x^((m : ℝ) / (n : ℝ)) ∧ (∀ x : ℝ, f (-x) = -f x)) ↔ sufficient_condition_for_odd_function m n := 
  by
    sorry

end power_function_odd_condition_l300_300334


namespace seq_properties_l300_300559

noncomputable def geometric_seq (r : ℝ) (n : ℕ) : ℝ := r^n

theorem seq_properties :
  ∃ (r : ℝ) (a : ℕ → ℝ),
    (∀ n, a n = geometric_seq r n) ∧
    (¬(∀ n, a n ≤ a (n+1)) ∧ ¬(∀ n, a n ≥ a (n+1))) ∧
    (∀ n, abs (a n) ≥ abs (a (n+1))) ∧
    a = λ n, (-1/2)^n :=
by {
  use -1/2,
  use (λ n, geometric_seq (-1/2) n),
  split,
  { intro n,
    refl },
  split,
  { split;
    { intro h,
      specialize h 0,
      linarith }},
  split,
  { intros n,
    simp [geometric_seq, abs_pow],
    norm_num },
  { refl }
}

end seq_properties_l300_300559


namespace cos_arcsin_l300_300237

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l300_300237


namespace sam_distance_l300_300454

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300454


namespace cos_arcsin_l300_300239

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l300_300239


namespace max_area_triangle_PJ1J2_l300_300030

noncomputable def triangle_PQR (PQ QR PR : ℝ) (angle_P angle_Q angle_R : ℝ) : Prop :=
  PQ = 20 ∧ QR = 21 ∧ PR = 29

noncomputable def max_area_PJ1J2 (PQ QR PR angle_P angle_Q angle_R : ℝ) (PJ1 PJ2 : ℝ) : ℝ :=
  PQ * PR * real.sin (angle_P / 2) * real.sin (angle_Q / 2) * real.sin (angle_R / 2)

theorem max_area_triangle_PJ1J2 (PQ QR PR angle_P angle_Q angle_R PJ1 PJ2 : ℝ) (h : triangle_PQR PQ QR PR angle_P angle_Q angle_R) :
  max_area_PJ1J2 PQ QR PR angle_P angle_Q angle_R PJ1 PJ2 = 20 * 29 * real.sin (angle_P / 2) * real.sin (angle_Q / 2) * real.sin (angle_R / 2) :=
sorry

end max_area_triangle_PJ1J2_l300_300030


namespace min_odd_integers_l300_300127

theorem min_odd_integers (a b c d e f g h i : ℤ)
  (h1 : a + b + c = 30)
  (h2 : a + b + c + d + e + f = 48)
  (h3 : a + b + c + d + e + f + g + h + i = 69) :
  ∃ k : ℕ, k = 1 ∧
  (∃ (aa bb cc dd ee ff gg hh ii : ℤ), (fun (x : ℤ) => x % 2 = 1 → k = 1) (aa + bb + cc + dd + ee + ff + gg + hh + ii)) :=
by
  intros
  sorry

end min_odd_integers_l300_300127


namespace median_team_a_mode_team_b_avg_team_b_var_team_b_neater_scores_l300_300227

def team_a_scores : list ℕ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : list ℕ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]
def variance_team_a_scores : ℝ := 1.4

theorem median_team_a : 
  (list.median team_a_scores) = 9.5 := 
by sorry

theorem mode_team_b : 
  (list.mode team_b_scores) = 10 := 
by sorry

theorem avg_team_b : 
  (list.sum team_b_scores : ℝ) / (list.length team_b_scores : ℝ) = 9 := 
by sorry

theorem var_team_b : 
  let avg := (list.sum team_b_scores : ℝ) / (list.length team_b_scores : ℝ) in
  (list.sum (list.map (λ x, (x - avg) ^ 2) team_b_scores) / (list.length team_b_scores : ℝ)) = 1 := 
by sorry

theorem neater_scores : 
  variance_team_a_scores > 1 := 
by sorry

end median_team_a_mode_team_b_avg_team_b_var_team_b_neater_scores_l300_300227


namespace two_numbers_with_difference_less_than_half_l300_300842

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l300_300842


namespace garden_area_increase_l300_300612

-- Given conditions:
def original_length : ℝ := 60
def original_width : ℝ := 20

-- Definitions derived from conditions:
def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)
def new_side_length : ℝ := original_perimeter / 4
def new_area : ℝ := new_side_length * new_side_length
def area_increase : ℝ := new_area - original_area

-- Lean 4 statement for the proof problem:
theorem garden_area_increase : area_increase = 400 := by
  -- Proof steps would be elaborated here
  sorry

end garden_area_increase_l300_300612


namespace non_negative_reals_inequality_l300_300712

theorem non_negative_reals_inequality {n : ℕ} (x : ℕ → ℝ) (s : ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i in range n, x i = s) : 
  ∑ i in range (n - 1), x i * x (i + 1) ≤ s ^ 2 / 4 :=
by {
  sorry
}

end non_negative_reals_inequality_l300_300712


namespace power_of_thirtyfive_l300_300853

theorem power_of_thirtyfive (m n : ℤ) (P Q : ℤ) (hP : P = 5^m) (hQ : Q = 7^n) :
  (35 : ℤ)^(m * n) = P^n * Q^m :=
by
  sorry

end power_of_thirtyfive_l300_300853


namespace cos_arcsin_l300_300233

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l300_300233


namespace problem_statement_l300_300200

def f (x : ℝ) (a : ℝ) : ℝ := a * (x + 2) ^ 2 + 4

theorem problem_statement :
  let a := 1 / 4
  let b := 0
  let c := 4
  4 * a + 4 = 5 ∧ (16 * a + 4 = 5) → a + b + 2 * c = 33 / 4 :=
by
  intros a_eq b_eq c_eq cond1 cond2
  sorry

end problem_statement_l300_300200


namespace total_animals_l300_300772

theorem total_animals (H C2 C1 : ℕ) (humps_eq : 2 * C2 + C1 = 200) (horses_eq : H = C2) :
  H + C2 + C1 = 200 :=
by
  /- Proof steps are not required -/
  sorry

end total_animals_l300_300772


namespace equal_numbers_l300_300333

theorem equal_numbers (x : ℕ → ℝ) (n : ℕ) 
  (h1 : x (n + 1) = x 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 100 * (1 + x i) ≥ 101 * x (i + 1))
  (h3 : (∑ i in Finset.range n, x (i + 1)) ≥ 100 * n) :
  ∀ i, 1 ≤ i ∧ i ≤ n + 1 → x i = 100 :=
by
  sorry

end equal_numbers_l300_300333


namespace isosceles_triangle_base_angle_l300_300356

theorem isosceles_triangle_base_angle (T : Triangle) (a b c : ℝ) 
  (isosceles : T.is_isosceles) (exterior_angle : T.exterior_angle = 70) : T.base_angle = 35 :=
by
  sorry

end isosceles_triangle_base_angle_l300_300356


namespace measure_of_angle_B_l300_300007

noncomputable theory

def triangle_sides (A B C a b c : ℝ) : Prop :=
  c = (2 * real.sqrt 3 / 3) * b * real.sin (A + real.pi / 3) ∧ a + c = 4

theorem measure_of_angle_B (A B C a b c : ℝ) 
    (h1 : c = (2 * real.sqrt 3 / 3) * b * real.sin (A + real.pi / 3))
    (h2 : a + c = 4)
    (h_triangle : triangle_sides A B C a b c)
    : B = real.pi / 3 ∧ (4 < a + b + c ∧ a + b + c ≤ 6) :=
sorry

end measure_of_angle_B_l300_300007


namespace solve_inequality_l300_300089

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem solve_inequality (x : ℝ) : 
  (frac 
    (log3(x^4) * (Real.log x / Real.log (1/3)) + log3(x^2) - (Real.log (x^4) / Real.log (1/3)) + 2)
    ((Real.log(x^2) / Real.log(1/3)) ^ 3 + 64) <= 0) ↔ 
  (x ∈ Set.Icc (-9 : ℝ) (-3) ∪ 
   Set.Iio (0) ∪ 
   Set.Ioo (0) (1/ Real.sqrt (3 : ℝ)^4) ∪ 
   Set.Icc (3 : ℝ) (9))
:= by
  sorry

end solve_inequality_l300_300089


namespace concyclic_points_l300_300821

theorem concyclic_points 
  (A B C B1 C1 B2 C2 : Point) 
  (hABC : scalene_acute_triangle A B C) 
  (hB1 : on_ray A B1 C ∧ dist A B1 = dist B B1) 
  (hC1 : on_ray A C1 B ∧ dist A C1 = dist C C1)
  (hB2 : on_line B C B2 ∧ dist A B2 = dist C B2)
  (hC2 : on_line B C C2 ∧ dist B C2 = dist A C2):
  concyclic B1 C1 B2 C2 :=
by sorry

end concyclic_points_l300_300821


namespace katy_read_books_l300_300018

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end katy_read_books_l300_300018


namespace polynomial_form_for_divisibility_l300_300288

-- Definitions based on conditions
variable {R : Type*} [CommRing R]

-- Define the polynomial P with integer coefficients
def P (n : ℕ) : Polynomial ℤ := sorry

-- Function that assigns a positive integer to every lattice point in ℝ³
def f (x y z : ℤ) : ℤ := sorry

-- Main theorem statement
theorem polynomial_form_for_divisibility (P : ℕ → Polynomial ℤ) :
  (∀ n ≥ 1, ∀ (x y z : ℤ), (P n).eval n ∣ ∑ i in range (n^3), f (x + (i / (n^2))) (y + ((i / n) % n)) (z + (i % n))) ↔
  ∃ (c : ℤ) (k : ℕ), ∀ n, P n = Polynomial.C c * Polynomial.X^k :=
sorry

end polynomial_form_for_divisibility_l300_300288


namespace initial_investment_l300_300604

variable (P1 P2 π1 π2 : ℝ)

-- Given conditions
axiom h1 : π1 = 100
axiom h2 : π2 = 120

-- Revenue relation after the first transaction
axiom h3 : P2 = P1 + π1

-- Consistent profit relationship across transactions
axiom h4 : π2 = 0.2 * P2

-- To be proved
theorem initial_investment (P1 : ℝ) (h1 : π1 = 100) (h2 : π2 = 120) (h3 : P2 = P1 + π1) (h4 : π2 = 0.2 * P2) :
  P1 = 500 :=
sorry

end initial_investment_l300_300604


namespace A_is_editor_l300_300893

-- Define the volunteers and their professions
inductive Profession
| doctor
| teacher
| editor

open Profession

variables (A B C : Profession)
variables (ageA ageB ageC : ℕ)

-- Define the conditions
def condition1 : Prop := A ≠ doctor
def condition2 : Prop := (ageC < max {ageA | A = editor} ageB)
def condition3 : Prop := (ageB < max {ageC | C = doctor} ageB)

-- Prove that A is the editor
theorem A_is_editor : A = editor :=
by 
  have h1 : A ≠ doctor := sorry
  have h2 : C ≠ editor := sorry
  have h3 : (B ≠ doctor ∧ B ≠ editor) ∨ (C ≠ doctor ∧ C ≠ editor) := sorry
  exact editor

end A_is_editor_l300_300893


namespace range_of_ratio_l300_300769

-- Definitions and Assumptions
variable (A B C : Type) [inner_product_space ℝ A]
variable (a b c : ℝ)
variable (angle_A angle_B angle_C : ℝ) 

-- Triangle ABC is a right triangle with ∠C = 90°
def right_triangle (ABC : triangle A B C) : Prop :=
  angle_C = real.pi / 2 ∧ a^2 + b^2 = c^2

-- The claim to prove: 1 < (a + b) / c ≤ sqrt(2)
theorem range_of_ratio {A B C : Type} [inner_product_space ℝ A] 
  (ABC : triangle A B C) (a b c : ℝ) 
  (h : right_triangle ABC) : 
  1 < (a + b) / c ∧ (a + b) / c ≤ real.sqrt 2 :=
sorry

end range_of_ratio_l300_300769


namespace sum_equals_1000_500_334_l300_300346

theorem sum_equals_1000_500_334 :
  (∑ n in Finset.range (1000 + 1), n * (1001 - n)) = 1000 * 500 * 334 :=
by
  sorry

end sum_equals_1000_500_334_l300_300346


namespace P_and_Q_equals_54_l300_300353

noncomputable def P_and_Q (P Q : ℤ) : ℤ :=
  if (x^2 + 3*x + 7 | x^4 + P*x^2 + Q)
  then P + Q
  else 0

theorem P_and_Q_equals_54 (P Q : ℤ) (h : (x^2 + 3*x + 7| x^4 + P*x^2 + Q)) :
  P + Q = 54 := by
  sorry

end P_and_Q_equals_54_l300_300353


namespace monotonicity_intervals_f_above_g_l300_300729

noncomputable def f (x m : ℝ) := (Real.exp x) / (x^2 - m * x + 1)

theorem monotonicity_intervals (m : ℝ) (h : m ∈ Set.Ioo (-2 : ℝ) 2) :
  (m = 0 → ∀ x y : ℝ, x ≤ y → f x m ≤ f y m) ∧ 
  (0 < m ∧ m < 2 → ∀ x : ℝ, (x < 1 → f x m < f (x + 1) m) ∧
    (1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (x > m + 1 → f x m < f (x + 1) m)) ∧
  (-2 < m ∧ m < 0 → ∀ x : ℝ, (x < m + 1 → f x m < f (x + 1) m) ∧
    (m + 1 < x ∧ x < 1 → f x m > f (x + 1) m) ∧
    (x > 1 → f x m < f (x + 1) m)) :=
sorry

theorem f_above_g (m : ℝ) (hm : m ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ)) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (m + 1)) :
  f x m > x :=
sorry

end monotonicity_intervals_f_above_g_l300_300729


namespace sam_drove_distance_l300_300435

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300435


namespace complete_square_solution_l300_300558

theorem complete_square_solution
  (x : ℝ)
  (h : x^2 + 4*x + 2 = 0):
  ∃ c : ℝ, (x + 2)^2 = c ∧ c = 2 :=
by
  sorry

end complete_square_solution_l300_300558


namespace sum_of_ages_is_60_l300_300521

theorem sum_of_ages_is_60 (A B : ℕ) (h1 : A = 2 * B) (h2 : (A + 3) + (B + 3) = 66) : A + B = 60 :=
by sorry

end sum_of_ages_is_60_l300_300521


namespace HCF_of_numbers_l300_300110

theorem HCF_of_numbers (a b : ℕ) (h₁ : a * b = 84942) (h₂ : Nat.lcm a b = 2574) : Nat.gcd a b = 33 :=
by
  sorry

end HCF_of_numbers_l300_300110


namespace prove_p_and_q_false_l300_300414

def p : Prop := ∃ x, x > 0 ∧ ∀ k, y = sin (2 * k * x) → k ≠ 0 → y ≠ sin 2
def q : Prop := ∀ k, y = cos k → k = π

theorem prove_p_and_q_false : (p ∧ q) = False :=
by
  sorry

end prove_p_and_q_false_l300_300414


namespace part1_part2_l300_300708

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 2 else (2 * a (n - 1)) / (a (n - 1) + 2)

theorem part1 (n : ℕ) (h : n ≥ 1) :
  (∀ k : ℕ, k ≥ 1 → (1 / a (k + 1)) = (1 / 2) * (1 / a k + 1 / 2)) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2 / n) :=
sorry

def b (n : ℕ) : ℝ := (2 + a n) / a n

def c (n : ℕ) : ℝ := b n * (1 / 2) ^ n

def T (n : ℕ) : ℝ := ∑ i in Finset.range n, c (i + 1)

theorem part2 (n : ℕ) :
  T n = 3 - (n + 3) / (2 ^ n) :=
sorry

end part1_part2_l300_300708


namespace lateral_surface_area_of_pyramid_inscribed_in_sphere_l300_300967
-- Importing the entire Mathlib library to ensure all necessary definitions and theorems are available.

-- Formulate the problem as a Lean statement.

theorem lateral_surface_area_of_pyramid_inscribed_in_sphere :
  let R := (1 : ℝ)
  let theta := (45 : ℝ) * Real.pi / 180 -- Convert degrees to radians.
  -- Assuming the pyramid is regular and quadrilateral, inscribed in a sphere of radius 1
  ∃ S : ℝ, S = 4 :=
  sorry

end lateral_surface_area_of_pyramid_inscribed_in_sphere_l300_300967


namespace factorize1_factorize2_l300_300672

-- Part 1: Prove the factorization of xy - 1 - x + y
theorem factorize1 (x y : ℝ) : (x * y - 1 - x + y) = (y - 1) * (x + 1) :=
  sorry

-- Part 2: Prove the factorization of (a^2 + b^2)^2 - 4a^2b^2
theorem factorize2 (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
  sorry

end factorize1_factorize2_l300_300672


namespace range_of_a_l300_300295

variable (a : ℝ)

def p (x : ℝ) : Prop := x^2 - 8x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2x + 1 - a^2 ≤ 0
def not_p (x : ℝ) : Prop := x ≤ -2 ∨ x ≥ 10
def not_q (x : ℝ) (a : ℝ) : Prop := x ≤ 1 - a ∨ x ≥ 1 + a

theorem range_of_a (a : ℝ) (h : ∀ x, not_p x → not_q x a) : 9 ≤ a :=
by
  sorry

end range_of_a_l300_300295


namespace fib_mod_3_l300_300276

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib n + fib (n + 1)

theorem fib_mod_3 (n : ℕ) : fib n % 3 = 0 ↔ ∃ k : ℕ, n = 4 * k := 
sorry

end fib_mod_3_l300_300276


namespace largest_2_digit_prime_factor_of_binom_180_90_l300_300541

theorem largest_2_digit_prime_factor_of_binom_180_90 :
  ∃ (p : ℕ), (nat.prime p) ∧ (10 ≤ p ∧ p < 100) ∧ (3 * p < 180) ∧ 
  (∀ q, nat.prime q ∧ (10 ≤ q ∧ q < 100) ∧ (3 * q < 180) → q ≤ p) ∧ p = 59 :=
sorry

end largest_2_digit_prime_factor_of_binom_180_90_l300_300541


namespace factorization_identity_sum_l300_300501

theorem factorization_identity_sum (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 15 * x + 36 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 20 :=
sorry

end factorization_identity_sum_l300_300501


namespace SamDrove200Miles_l300_300443

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300443


namespace sum_of_first_60_terms_l300_300707

/-- Given a sequence {a_n} that satisfies a_{n+1} + (-1)^n * a_n = 3n - 1, 
    prove that the sum of the first 60 terms of {a_n} is 2760. --/
theorem sum_of_first_60_terms (a : ℕ → ℤ) 
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 3 * n - 1) : 
  (∑ k in Finset.range 60, a k) = 2760 := 
by
  sorry

end sum_of_first_60_terms_l300_300707


namespace dealer_gross_profit_l300_300601

theorem dealer_gross_profit (purchase_price : ℝ) (markup_rate : ℝ) (selling_price : ℝ) (gross_profit : ℝ) 
  (purchase_price_cond : purchase_price = 150)
  (markup_rate_cond : markup_rate = 0.25)
  (selling_price_eq : selling_price = purchase_price + (markup_rate * selling_price))
  (gross_profit_eq : gross_profit = selling_price - purchase_price) : 
  gross_profit = 50 :=
by
  sorry

end dealer_gross_profit_l300_300601


namespace inequality_solution_l300_300272

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 4) > 2 / x + 12 / 5 ↔ x < 0 :=
by
  sorry

end inequality_solution_l300_300272


namespace area_square_A_32_l300_300635

-- Define the areas of the squares in Figure B and Figure A and their relationship with the triangle areas
def identical_isosceles_triangles_with_squares (area_square_B : ℝ) (area_triangle_B : ℝ) (area_square_A : ℝ) (area_triangle_A : ℝ) :=
  area_triangle_B = (area_square_B / 2) * 4 ∧
  area_square_A / area_triangle_A = 4 / 9

theorem area_square_A_32 {area_square_B : ℝ} (h : area_square_B = 36) :
  identical_isosceles_triangles_with_squares area_square_B 72 32 72 :=
by
  sorry

end area_square_A_32_l300_300635


namespace triangle_perimeter_l300_300371

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being a root of the equation
    x^2 - 8x + 12 = 0, the perimeter of the triangle is 13. --/
theorem triangle_perimeter
  (a b : ℕ) 
  (ha : a = 2) 
  (hb : b = 5)
  (c : ℕ)
  (h_c_root : c * c - 8 * c + 12 = 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 13 := 
sorry

end triangle_perimeter_l300_300371


namespace SamDrove200Miles_l300_300444

/-- Given conditions -/
def MargueriteDistance : ℝ := 150
def MargueriteTime : ℝ := 3
def SameRateTime : ℝ := 4

/-- Calculate Marguerite's average speed -/
def MargueriteSpeed : ℝ := MargueriteDistance / MargueriteTime

/-- Calculate distance Sam drove -/
def SamDistance : ℝ := MargueriteSpeed * SameRateTime

/-- The theorem statement: Sam drove 200 miles -/
theorem SamDrove200Miles : SamDistance = 200 := by
  sorry

end SamDrove200Miles_l300_300444


namespace parallelogram_either_rectangle_or_rhombus_l300_300038

theorem parallelogram_either_rectangle_or_rhombus
  (A B C D O : Type*)
  [add_comm_group A]
  [module ℝ A]
  [affine_space A B]
  (par : affine_subspace ℝ A)
  (circumcenter_of_triangle : true) -- placeholder for the circumcenter condition
  (lies_on_diagonal : true) -- placeholder for lying on diagonal condition
  : (∃ A B C D : par, par.parallel (B - A) (D - C) ∧ par.parallel (A - B) (C - D) ∧ 
                    ((A = B ∨ C = D) → false) →  -- segment equalities ruling out both segments being vertices.
                    (A = C ∨ B = D) → false) → -- segment equalities ruling out both segments being vertices.
                    is_rectangle par ∨ is_rhombus par :=
sorry

end parallelogram_either_rectangle_or_rhombus_l300_300038


namespace at_most_p_minus_one_divisible_l300_300561

variable {p : ℕ} (hp_prime : Nat.Prime p) (hp_mod_3 : p % 3 = 2) (hp_odd : p % 2 = 1)

theorem at_most_p_minus_one_divisible (S : Set ℤ) :
  S = { m^2 - n^3 - 1 | m n : ℤ ∧ 0 < m ∧ m < p ∧ 0 < n ∧ n < p } →
  (S.filter (λ x, x % p = 0)).toFinset.card ≤ p - 1 :=
sorry

end at_most_p_minus_one_divisible_l300_300561


namespace sunland_more_plates_than_moonland_l300_300416

theorem sunland_more_plates_than_moonland :
  let sunland_plates := 26^5 * 10^2
  let moonland_plates := 26^3 * 10^3
  sunland_plates - moonland_plates = 1170561600 := by
  sorry

end sunland_more_plates_than_moonland_l300_300416


namespace batsman_average_after_12_innings_l300_300581

theorem batsman_average_after_12_innings
  (score_12th: ℕ) (increase_avg: ℕ) (initial_innings: ℕ) (final_innings: ℕ) 
  (initial_avg: ℕ) (final_avg: ℕ) :
  score_12th = 48 ∧ increase_avg = 2 ∧ initial_innings = 11 ∧ final_innings = 12 ∧
  final_avg = initial_avg + increase_avg ∧
  12 * final_avg = initial_innings * initial_avg + score_12th →
  final_avg = 26 :=
by 
  sorry

end batsman_average_after_12_innings_l300_300581


namespace range_fx_l300_300316

noncomputable def f (k x : ℝ) : ℝ := 1 / x^k

theorem range_fx (k : ℝ) (h : k > 0) : 
  set.range (λ x, f k x) (set.Ici 0.5) = set.Ioc 0 2^k := 
sorry

end range_fx_l300_300316


namespace emails_in_afternoon_l300_300793

theorem emails_in_afternoon (emails_morning emails_evening emails_total emails_afternoon : ℕ) 
    (h1 : emails_morning = 6) 
    (h2 : emails_evening = 1) 
    (h3 : emails_total = 10) 
    (h4 : emails_total = emails_morning + emails_afternoon + emails_evening) : 
    emails_afternoon = 4 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  assumption 

end emails_in_afternoon_l300_300793


namespace katy_read_books_l300_300019

theorem katy_read_books (juneBooks : ℕ) (julyBooks : ℕ) (augustBooks : ℕ)
  (H1 : juneBooks = 8)
  (H2 : julyBooks = 2 * juneBooks)
  (H3 : augustBooks = julyBooks - 3) :
  juneBooks + julyBooks + augustBooks = 37 := by
  -- Proof goes here
  sorry

end katy_read_books_l300_300019


namespace line_passing_through_point_is_polar_axis_parallel_l300_300380

-- Define the problem parameters
def P := (2, Real.pi / 6)
def polar_axis_parallel_line (r θ : ℝ) := r * Real.sin θ = 1

-- Statement we want to prove
theorem line_passing_through_point_is_polar_axis_parallel :
  polar_axis_parallel_line 2 (Real.pi / 6) := sorry

end line_passing_through_point_is_polar_axis_parallel_l300_300380


namespace num_4_digit_using_2_and_3_l300_300222

theorem num_4_digit_using_2_and_3 : 
    {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d ∈ [1, 2, 3, 4], (n.digits 10).nth (d - 1) = some 2 ∨ (n.digits 10).nth (d - 1) = some 3 ∧ ∃ d₁ d₂, d₁ ≠ d₂ ∧ ∃ i1, (n.digits 10).nth i1 = some 2 ∧ ∃ i2, (n.digits 10).nth i2 = some 3 }.card = 14 :=
by sorry

end num_4_digit_using_2_and_3_l300_300222


namespace eval_expression_l300_300266

theorem eval_expression :
  2002^3 - 2001 * 2002^2 - 2001^2 * 2002 + 2001^3 + (2002 - 2001)^3 = 4004 := by
  -- Set a and b for ease of use
  let a := 2001
  let b := 2002
  
  -- original expression transformed using a and b
  have h : b^3 - a * b^2 - a^2 * b + a^3 + (b - a)^3 = (b + a) * (b - a)^2 + (b - a)^3 := by sorry

  -- Substitute a and b into the transformed expression
  calc
    2002^3 - 2001 * 2002^2 - 2001^2 * 2002 + 2001^3 + (2002 - 2001)^3
        = (b + a) * (b - a)^2 + (b - a)^3 : by rw h
    ... = (2001 + 2002) * (2002 - 2001)^2 + (2002 - 2001)^3 : by rfl
    ... = 4003 * 1 + 1 : by simp
    ... = 4004 : by norm_num

end eval_expression_l300_300266


namespace max_value_of_f_l300_300361

def f (x : ℝ) : ℝ := (4 - x^2) * (x^2 + 6*x + 5)

theorem max_value_of_f :
  ∃ x : ℝ, f(x) = 36 :=
by
  sorry

end max_value_of_f_l300_300361


namespace symmetric_points_sum_l300_300724

theorem symmetric_points_sum (a b : ℝ) (h₁ : a = -4) (h₂ : b = -2) : a + b = -6 :=
by {
  rw [h₁, h₂],
  norm_num,
  sorry
}

-- Conditions: Point A(a, 2) and B(4, b) are symmetric with respect to the origin.
-- Given these conditions, prove a + b = -6.

end symmetric_points_sum_l300_300724


namespace gcd_factorial_l300_300139

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l300_300139


namespace simplify_expression_l300_300086

variable (b c : ℝ)

theorem simplify_expression :
  3 * b * (3 * b ^ 3 + 2 * b) - 2 * b ^ 2 + c * (3 * b ^ 2 - c) = 9 * b ^ 4 + 4 * b ^ 2 + 3 * b ^ 2 * c - c ^ 2 :=
by
  sorry

end simplify_expression_l300_300086


namespace sum_surface_area_and_volume_l300_300263

structure Point3D where
  x : Int
  y : Int
  z : Int

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def parallelepiped is
  base_v1 (0,0,0)
  base_v2 (3,4,0)
  base_v3 (7,0,0)
  base_v4 (10,4,0)
  height 5

theorem sum_surface_area_and_volume:
  let base := [base_v1,base_v2,base_v3,base_v4]
  let side_lengths := [distance base_v1 base_v2, distance base_v1 base_v3, distance base_v2 base_v4]
  let side_areas := [ (distance base_v1 base_v3) * height, (distance base_v2 base_v4) * height ]
  let area_base := (distance base_v1 base_v3) * (distance base_v1 base_v2)
  let volume := area_base * height
  let surface_area := 2 * area_base + 2 * (distance base_v1 base_v3 * height) + 2 * (distance base_v2 base_v4 * height)
  surface_area + volume = 365 := by
  sorry

end sum_surface_area_and_volume_l300_300263


namespace books_taken_off_l300_300120

def books_initially : ℝ := 38.0
def books_remaining : ℝ := 28.0

theorem books_taken_off : books_initially - books_remaining = 10 := by
  sorry

end books_taken_off_l300_300120


namespace perpendicular_common_chord_l300_300408

open EuclideanGeometry

theorem perpendicular_common_chord 
  (O1 O2 : Point) (AF : Line) (B C D E : Point)
  (on_O1 : B ∈ circle O1)
  (on_O2 : C ∈ circle O2)
  (common_chord : AF.commonChord circle O1 circle O2)
  (AB_eq_AC : distance A B = distance A C)
  (D_on_bisector : ∃ F : Point, F ∈ AF ∧ D ∈ bisector (angle B A F))
  (E_on_bisector : ∃ F : Point, F ∈ AF ∧ E ∈ bisector (angle C A F)) :
  isPerpendicular DE AF := 
sorry

end perpendicular_common_chord_l300_300408


namespace convex_polygon_diagonals_l300_300141

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let diagonals := n * (n - 3) / 2 in
  diagonals = 405 := 
by
  -- Given n = 30, we substitute n into the expression:
  have h1 : diagonals = 30 * (30 - 3) / 2 := by simp [h]
  -- Simplify the expression to find:
  have h2 : diagonals = 30 * 27 / 2 := by simp [h1]
  -- Simplify further:
  have h3 : diagonals = 405 := by norm_num [h2]
  -- Therefore, we conclude:
  exact h3

end convex_polygon_diagonals_l300_300141


namespace fraction_drank_second_day_l300_300181

theorem fraction_drank_second_day 
  (initial_bottles : ℕ) 
  (drank_first_day_fraction : ℚ)
  (remaining_bottles_after_two_days : ℕ)
  (initial_bottles = 24)
  (drank_first_day_fraction = 1/3)
  (remaining_bottles_after_two_days = 8) :
  (initial_bottles - initial_bottles * drank_first_day_fraction - remaining_bottles_after_two_days) / (initial_bottles - initial_bottles * drank_first_day_fraction) = 1 / 2 := 
by 
  sorry

end fraction_drank_second_day_l300_300181


namespace perp_condition_l300_300297

variables {ℝ : Type} [inner_product_space ℝ] (a b : ℝ) (ka : ℝ)

-- Definitions based on given conditions
def magnitude_a := (∥a∥ = 5)
def magnitude_b := (∥b∥ = 4)
def angle_condition := (real.angle a b = real.pi / 3)

-- Lean statement for proof
theorem perp_condition (a b : ℝ) (h₁ : magnitude_a) (h₂ : magnitude_b) (h₃ : angle_condition) :
  let k := (14 / 15) in
  inner (ka - b) (a + 2 * b) = 0 := sorry

end perp_condition_l300_300297


namespace trig_identity_l300_300116

theorem trig_identity :
  (Real.cos (105 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.sin (105 * Real.pi / 180)) = 1 / 2 :=
  sorry

end trig_identity_l300_300116


namespace price_increase_twice_l300_300107

open Real

theorem price_increase_twice (P x : ℝ) (h : (1 + x)^2 = 1.1236) : x = 0.06 :=
by
  have : 1 + x = sqrt 1.1236,
  { sorry },
  have : 1 + x = 1.06,
  { sorry },
  linarith

end price_increase_twice_l300_300107


namespace proof_PQ_squared_l300_300306

-- Definitions of points and lengths in the problem
variable (A B C P Q K L : Type) 
variable [OrderedCommRing Type]
variable (AC AB BC : ℝ)
variable (PK KQ : ℝ)

-- Conditions
def AB : ℝ := 42
def BC : ℝ := 56
axiom right_triangle_ABC : AB^2 + BC^2 = AC^2
axiom circle_intersects_sides : (circle B).intersects_side_AB = P ∧ (circle B).intersects_side_BC = Q ∧ (circle B).intersects_hypotenuse_AC {K, L}
axiom equal_segments : PK = KQ
axiom ratio_property : QL / PL = ¾

-- Required to prove
theorem proof_PQ_squared : 
  ∃ (P Q K L AC : ℝ), 
  (P_to_K = K_to_Q) ∧ 
  (ratio_QK_PK = (3 / 4)) ∧ 
  AC = sqrt(AB^2 + BC^2) ∧ 
  PQ^2 = 1250
by
  sorry

end proof_PQ_squared_l300_300306


namespace find_r_plus_s_l300_300509

/-- The number b = r / s, where r and s are relatively prime positive integers, has the property 
that the sum of all real numbers y satisfying ⌊y⌋ * {y} = b * y ^ 2.5 is 315. 
Find r + s. -/
theorem find_r_plus_s (r s : ℕ) (y : ℝ) (n : ℤ) (d : ℝ) (b : ℝ):
  r.gcd s = 1 ∧ b = (r:ℝ) / (s:ℝ) ∧ (int.floor y : ℝ) * (y - int.floor y) = b * y ^ 2.5 ∧ 
  (∑ y in finset.range 22, (n + d) = 315) → r + s = 277 :=
by
  sorry

end find_r_plus_s_l300_300509


namespace max_airlines_in_country_l300_300770

-- Definition of the problem parameters
variable (N k : ℕ) 

-- Definition of the problem conditions
variable (hN_pos : 0 < N)
variable (hk_pos : 0 < k)
variable (hN_ge_k : k ≤ N)

-- Definition of the function calculating the maximum number of air routes
def max_air_routes (N k : ℕ) : ℕ :=
  Nat.choose N 2 - Nat.choose k 2

-- Theorem stating the maximum number of airlines given the conditions
theorem max_airlines_in_country (N k : ℕ) (hN_pos : 0 < N) (hk_pos : 0 < k) (hN_ge_k : k ≤ N) :
  max_air_routes N k = Nat.choose N 2 - Nat.choose k 2 :=
by sorry

end max_airlines_in_country_l300_300770


namespace ratio_CD_BD_l300_300766

-- Definitions
variables {A B C D E T : Type}
variables [LieGroup B C D E]
variables (h1: lies_on D BC)
variables (h2: lies_on E AC)
variables (h3: LineIntersect AD BE T)
variables (h4: (AT / DT) = 2)
variables (h5: (BT / ET) = 3)

-- Theorem: Prove that CD / BD = 1 / 3 given the conditions.
theorem ratio_CD_BD : CD / BD = 1 / 3 :=
sorry

end ratio_CD_BD_l300_300766


namespace suit_cost_l300_300480

theorem suit_cost :
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  ∃ S, discount_coupon * discount_store * (total_cost + S) = 252 → S = 150 :=
by
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  exists 150
  intro h
  sorry

end suit_cost_l300_300480


namespace red_beads_count_l300_300974

theorem red_beads_count (total_beads : ℕ) (pattern : list string) (red_count_in_pattern : ℕ) (pattern_length : ℕ) 
  (conditions : total_beads = 85 ∧ pattern = ["green", "green", "green", "red", "red", "red", "red", "yellow"] ∧ 
                red_count_in_pattern = 4 ∧ pattern_length = 8) : 
  ∃ red_beads : ℕ, red_beads = 42 :=
by 
  have total_beads := total_beads, from conditions.left,
  have pattern := pattern, from conditions.right.left,
  have red_count_in_pattern := red_count_in_pattern, from conditions.right.right.left,
  have pattern_length := pattern_length, from conditions.right.right.right,
  have total_complete_groups := total_beads / pattern_length,
  have remaining_beads := total_beads % pattern_length,
  have red_beads_in_complete_groups := total_complete_groups * red_count_in_pattern,
  have remaining_pattern_segment := (pattern.take remaining_beads).filter (λ x, x = "red"),
  have red_beads_in_remaining_segment := remaining_pattern_segment.length,
  have red_beads := red_beads_in_complete_groups + red_beads_in_remaining_segment,
  use red_beads,
  have correct_answer := 42,
  sorry

end red_beads_count_l300_300974


namespace arithmetic_sequence_product_l300_300033

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) (h1 : ∀ n m, n < m → b n < b m) 
(h2 : ∀ n, b (n + 1) - b n = d) (h3 : b 3 * b 4 = 18) : b 2 * b 5 = -80 :=
sorry

end arithmetic_sequence_product_l300_300033


namespace transport_capacity_and_cost_l300_300122

variables {a b : ℝ}
variables {x : ℕ}

/-- Definition of the transportation problem --/
theorem transport_capacity_and_cost 
  (h1 : 3 * a + 4 * b = 18)
  (h2 : 2 * a + 6 * b = 17)
  (hb_capacity : b = 1.5)
  (ha_capacity : a = 4)
  (transport_capacity : ∀ x : ℕ, 4 * x + 1.5 * (10 - x) ≥ 33)
  (cost_function : ∀ x : ℕ, 130 * x + 100 * (10 - x)) :
  (∀ x : ℕ, x ≥ 8 → 8 ≤ x ∧ x ≤ 10 → 130 * 8 + 100 * (10 - 8) = 1120) :=
by 
  sorry


end transport_capacity_and_cost_l300_300122


namespace sam_drove_200_miles_l300_300467

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300467


namespace area_of_triangle_maximize_area_angle_condition_area_max_in_equilateral_l300_300020

-- Definitions and conditions used in Lean 4 statement
variables {A B C : Type*} [IsPoint (circle (1 : ℝ)) A] [IsPoint (circle (1 : ℝ)) B] [IsPoint (circle (1 : ℝ)) C]
def radius : ℝ := 1
def angle_sum (α β γ : ℝ) : Prop := α + β + γ = π

-- Problem (a)
theorem area_of_triangle (α β γ : ℝ) (hα : α = angle A B C)
(hβ : β = angle B C A) (hγ : γ = angle C A B) :
Area (triangle A B C) = 1 / 2 * (sin (2*α) + sin (2*β) + sin (2*γ)) := sorry

-- Problem (b)
theorem maximize_area_angle_condition (α β γ : ℝ) (hα : α = angle A B C)
(hβ : β = angle B C A) (hγ : γ = angle C A B) (hfixed : angle_sum α β γ) :
(Areas (triangle A B C)).maximal (when (β = γ)) := sorry

-- Problem (c)
theorem area_max_in_equilateral (α β γ : ℝ) (hα : α = angle A B C)
(hβ : β = angle B C A) (hγ : γ = angle C A B) (heuristic : β = γ) (hsum : angle_sum α β γ) :
(Areas (triangle A B C)).maximal (when (α = β = γ)) := sorry

end area_of_triangle_maximize_area_angle_condition_area_max_in_equilateral_l300_300020


namespace largest_integer_solution_of_abs_eq_and_inequality_l300_300682

theorem largest_integer_solution_of_abs_eq_and_inequality : 
  ∃ x : ℤ, |x - 3| = 15 ∧ x ≤ 20 ∧ (∀ y : ℤ, |y - 3| = 15 ∧ y ≤ 20 → y ≤ x) :=
sorry

end largest_integer_solution_of_abs_eq_and_inequality_l300_300682


namespace at_least_97_l300_300525

-- define the condition: the product of a positive multiple of 5 and an odd number is odd.
def is_product_of_multiple_of_5_and_odd (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ n = 5 * k * m

-- define the main theorem
theorem at_least_97 (x : ℕ) : 
  (∃ l : Finset ℕ, l.card = 10 ∧ (∀ a ∈ l, a < x ∧ is_product_of_multiple_of_5_and_odd a) ∧ l = l.erase_lt x) →
  x ≥ 97 :=
by {
  sorry -- proof omitted
}

end at_least_97_l300_300525


namespace find_thief_l300_300572

def StealSalt : Type :=
| Caterpillar : StealSalt
| LizardBill : StealSalt
| CheshireCat : StealSalt

open StealSalt

-- Conditions
def Caterpillar_statement (thief : StealSalt) : Prop := thief = LizardBill
def LizardBill_statement (thief : StealSalt) : Prop := thief = LizardBill
def CheshireCat_statement (thief : StealSalt) : Prop := thief ≠ CheshireCat

-- Problem conditions
def at_least_one_true (statements : List (Prop)) : Prop :=
  statements.foldr (λ p acc => p ∨ acc) false

def at_least_one_false (statements : List (Prop)) : Prop :=
  statements.foldr (λ p acc => ¬p ∨ acc) false

-- Main theorem
theorem find_thief (thief : StealSalt) :
  (at_least_one_true [
      Caterpillar_statement thief,
      LizardBill_statement thief,
      CheshireCat_statement thief
    ]) ∧
  (at_least_one_false [
      Caterpillar_statement thief,
      LizardBill_statement thief, 
      CheshireCat_statement thief
    ]) 
  → thief = Caterpillar :=
by
  sorry

end find_thief_l300_300572


namespace yuan_conversion_gram_to_kilogram_conversion_kilogram_to_ton_conversion_meter_conversion_l300_300271

-- Definition and proof problems
theorem yuan_conversion (jiao_to_yuan fen_to_yuan : ℤ) (h1 : jiao_to_yuan = 10) (h2 : fen_to_yuan = 100) :
  5 + (4 / jiao_to_yuan) + (8 / fen_to_yuan) = 5.48 :=
by sorry

theorem gram_to_kilogram_conversion (conversion_rate : ℤ) (h : conversion_rate = 1000) :
  80 / conversion_rate = 0.08 :=
by sorry

theorem kilogram_to_ton_conversion (conversion_rate : ℤ) (h : conversion_rate = 1000) :
  73 / conversion_rate = 0.073 :=
by sorry

theorem meter_conversion (conversion_rate : ℤ) (h : conversion_rate = 100) :
  1 + (5 / conversion_rate) = 1.05 :=
by sorry

end yuan_conversion_gram_to_kilogram_conversion_kilogram_to_ton_conversion_meter_conversion_l300_300271


namespace power_of_2_with_half_digits_9_l300_300830

theorem power_of_2_with_half_digits_9 (k : ℕ) (h : k > 1) : 
  ∃ m : ℕ, (nat.digits 10 (2^m)).length ≥ k ∧ 
           (nat.digits 10 (2^m)).take k.count (9) ≥ k / 2 :=
sorry

end power_of_2_with_half_digits_9_l300_300830


namespace projectile_max_height_l300_300608

def height (t : ℝ) : ℝ := -8 * t^2 + 64 * t + 36

theorem projectile_max_height : ∃ t : ℝ, height t = 164 :=
sorry

end projectile_max_height_l300_300608


namespace aziz_parents_move_year_l300_300998

theorem aziz_parents_move_year (current_year aziz_age : ℕ) (years_before_birth : ℕ)
  (h1 : current_year = 2021)
  (h2 : aziz_age = 36)
  (h3 : years_before_birth = 3) :
  let birth_year := current_year - aziz_age in
  let move_year := birth_year - years_before_birth in
  move_year = 1982 := by
  sorry

end aziz_parents_move_year_l300_300998


namespace sam_drove_distance_l300_300439

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300439


namespace area_triangle_F1PF2_l300_300715

noncomputable def F1 := (sorry : ℝ × ℝ)
noncomputable def F2 := (sorry : ℝ × ℝ)
noncomputable def P := (sorry : ℝ × ℝ)

axiom H_P_on_hyperbola : ∃ (x y : ℝ), P = (x, y) ∧ x^2 - y^2 / 24 = 1
axiom H_arithmetic_sequence : abs (dist P F1 + dist P F2) = abs (2 * dist F1 F2)

theorem area_triangle_F1PF2 : 
  let a := dist P F1, b := dist P F2, c := dist F1 F2 in
  c = 10 → a + b = 18 → (2 * b = a + c) →
  abs (triangle_area F1 P F2) = 24 := sorry

end area_triangle_F1PF2_l300_300715


namespace existence_of_infinitely_many_pairs_l300_300831

theorem existence_of_infinitely_many_pairs : 
  ∃∞ (a b : ℤ), 
  ∃ (k1 k2 : ℝ), 
  (k1 ≠ k2) ∧ (k1 * k2 = 1) ∧ (k1^(2012) - a * k1 - b = 0) ∧ (k2^(2012) - a * k2 - b = 0) := sorry

end existence_of_infinitely_many_pairs_l300_300831


namespace fraction_of_men_left_l300_300889

def totalGuests : ℕ := 50
def numberWomen : ℕ := totalGuests / 2
def numberMen : ℕ := 15
def numberChildren : ℕ := totalGuests - (numberWomen + numberMen)
def childrenLeft : ℕ := 4
def peopleStayed : ℕ := 43
def totalPeopleLeft : ℕ := totalGuests - peopleStayed
variable (x : ℚ)

theorem fraction_of_men_left : (4 + x * 15 = 7) → (x = 1 / 5) :=
by
  assume h : 4 + x * 15 = 7
  sorry

end fraction_of_men_left_l300_300889


namespace john_total_distance_l300_300568

theorem john_total_distance :
  let s₁ : ℝ := 45       -- Speed for the first part (mph)
  let t₁ : ℝ := 2        -- Time for the first part (hours)
  let s₂ : ℝ := 50       -- Speed for the second part (mph)
  let t₂ : ℝ := 3        -- Time for the second part (hours)
  let d₁ : ℝ := s₁ * t₁ -- Distance for the first part
  let d₂ : ℝ := s₂ * t₂ -- Distance for the second part
  d₁ + d₂ = 240          -- Total distance
:= by
  sorry

end john_total_distance_l300_300568


namespace seeds_per_can_l300_300794

theorem seeds_per_can (total_seeds : Float) (cans : Float) (h1 : total_seeds = 54.0) (h2 : cans = 9.0) : total_seeds / cans = 6.0 :=
by
  sorry

end seeds_per_can_l300_300794


namespace plane_split_into_regions_l300_300255

theorem plane_split_into_regions (S : set (ℝ × ℝ)) (hx3y : ∀ p ∈ S, p.2 = 3 * p.1 ∨ p.2 = (1 / 3) * p.1) : 
  ∃ (n : ℕ), n = 4 ∧ ∀ x y, (y = 3 * x ∨ y = (1 / 3) * x) → divides_plane_into_regions S n :=
sorry

end plane_split_into_regions_l300_300255


namespace floral_shop_bouquets_total_l300_300599

theorem floral_shop_bouquets_total (sold_monday_rose : ℕ) (sold_monday_lily : ℕ) (sold_monday_orchid : ℕ)
  (price_monday_rose : ℕ) (price_monday_lily : ℕ) (price_monday_orchid : ℕ)
  (sold_tuesday_rose : ℕ) (sold_tuesday_lily : ℕ) (sold_tuesday_orchid : ℕ)
  (price_tuesday_rose : ℕ) (price_tuesday_lily : ℕ) (price_tuesday_orchid : ℕ)
  (sold_wednesday_rose : ℕ) (sold_wednesday_lily : ℕ) (sold_wednesday_orchid : ℕ)
  (price_wednesday_rose : ℕ) (price_wednesday_lily : ℕ) (price_wednesday_orchid : ℕ)
  (H1 : sold_monday_rose = 12) (H2 : sold_monday_lily = 8) (H3 : sold_monday_orchid = 6)
  (H4 : price_monday_rose = 10) (H5 : price_monday_lily = 15) (H6 : price_monday_orchid = 20)
  (H7 : sold_tuesday_rose = 3 * sold_monday_rose) (H8 : sold_tuesday_lily = 2 * sold_monday_lily)
  (H9 : sold_tuesday_orchid = sold_monday_orchid / 2) (H10 : price_tuesday_rose = 12)
  (H11 : price_tuesday_lily = 18) (H12 : price_tuesday_orchid = 22)
  (H13 : sold_wednesday_rose = sold_tuesday_rose / 3) (H14 : sold_wednesday_lily = sold_tuesday_lily / 4)
  (H15 : sold_wednesday_orchid = 2 * sold_tuesday_orchid / 3) (H16 : price_wednesday_rose = 8)
  (H17 : price_wednesday_lily = 12) (H18 : price_wednesday_orchid = 16) :
  (sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose = 60) ∧
  (sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily = 28) ∧
  (sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid = 11) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose) = 648) ∧
  ((sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily) = 456) ∧
  ((sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 218) ∧
  ((sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose + sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily + sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid) = 99) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose + sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily + sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 1322) :=
  by sorry

end floral_shop_bouquets_total_l300_300599


namespace find_sum_invested_l300_300587

variable {P : ℝ}

def simple_interest_18 (P : ℝ) : ℝ := (9 * P) / 25
def simple_interest_12 (P : ℝ) : ℝ := (6 * P) / 25
def interest_difference_condition (P : ℝ) : Prop := simple_interest_18 P - simple_interest_12 P = 480

theorem find_sum_invested (h : interest_difference_condition P) : P = 4000 := sorry

end find_sum_invested_l300_300587


namespace find_angle_l300_300994

-- Define the conditions
def circles_intersect (r : ℝ) : Prop :=
  r > 0

def area_shaded (α r : ℝ) : ℝ :=
  (α * r^2 / 2) - (1 / 2 * r^2 * (Real.sin α))

def angle_condition (α : ℝ) : Prop :=
  α - Real.sin(α) = (4 * Real.pi / 3)

-- Define the constant 2.6053
def alpha_value : ℝ := 2.6053

theorem find_angle (r : ℝ) (hr : circles_intersect r) :
  ∃ α, angle_condition α ∧ (α = alpha_value) :=
by
  sorry

end find_angle_l300_300994


namespace possible_values_of_a_l300_300714

theorem possible_values_of_a
  (a : ℝ)
  (M = {_, _, a^2 - 3a - 1} : set ℝ)
  (N = {_, a, 3} : set ℝ)
  (h : M ∩ N = {3}) : a = 4 :=
sorry

end possible_values_of_a_l300_300714


namespace angle_RPQ_eq_45_l300_300782

  theorem angle_RPQ_eq_45
    (P R T Q : Type) [point P] [segment RT] [point QP] 
    (h1 : QP bisects (∠ TQR)) 
    (h2 : RP = RQ) 
    (h3 : ∠ RTQ = 4 * x)
    (h4 : ∠ RPQ = x) : 
    ∠ RPQ = 45 :=
  begin
    sorry
  end
  
end angle_RPQ_eq_45_l300_300782


namespace number_of_apples_in_shop_l300_300931

-- Definitions derived from conditions
def ratio_mango_orange_apple : ℕ × ℕ × ℕ := (10, 2, 3)
def number_of_mangoes : ℕ := 120

-- The proof statement
theorem number_of_apples_in_shop : 
  let (mango, orange, apple) := ratio_mango_orange_apple in
  number_of_mangoes / mango * apple = 36 :=
by
  sorry

end number_of_apples_in_shop_l300_300931


namespace concert_ticket_revenue_l300_300530

theorem concert_ticket_revenue :
  let original_price := 20
  let first_group_discount := 0.40
  let second_group_discount := 0.15
  let third_group_premium := 0.10
  let first_group_size := 10
  let second_group_size := 20
  let third_group_size := 15
  (first_group_size * (original_price - first_group_discount * original_price)) +
  (second_group_size * (original_price - second_group_discount * original_price)) +
  (third_group_size * (original_price + third_group_premium * original_price)) = 790 :=
by
  simp
  sorry

end concert_ticket_revenue_l300_300530


namespace comfortable_temperature_l300_300981

theorem comfortable_temperature (body_temp golden_ratio : ℝ) 
  (h_body_temp : body_temp = 36)
  (h_golden_ratio : golden_ratio = 0.618) : 
  int.nearest (body_temp * golden_ratio) = 22 :=
by
  sorry

end comfortable_temperature_l300_300981


namespace number_of_rolls_not_random_variable_l300_300556

-- Define the experimental setup for rolling a die twice
def roll_die_twice := (ℕ , ℕ) -- Represents the outcomes of two rolls of a fair die

-- Define the options in the context of random variables
def sum_of_rolls (r1 r2 : ℕ) := r1 + r2
def max_of_rolls (r1 r2 : ℕ) := max r1 r2
def diff_of_rolls (r1 r2 : ℕ) := r1 - r2
def number_of_rolls (r1 r2 : ℕ) := 2

-- Define the concept of a random variable
def is_random_variable (X : ℕ × ℕ → ℕ) := ∀ (r1 r2 : ℕ), r1 ∈ finset.range 1 7 → r2 ∈ finset.range 1 7 → X (r1, r2) ∈ ℕ

-- Prove that the number of times the die is rolled cannot be considered a random variable
theorem number_of_rolls_not_random_variable : ¬ is_random_variable number_of_rolls := by sorry

end number_of_rolls_not_random_variable_l300_300556


namespace maximum_books_l300_300068

theorem maximum_books (dollars : ℝ) (price_per_book : ℝ) (n : ℕ) 
    (h1 : dollars = 12) (h2 : price_per_book = 1.25) : n ≤ 9 :=
    sorry

end maximum_books_l300_300068


namespace max_value_l300_300829

-- Definition of the ellipse and the goal function
def ellipse (x y : ℝ) := 2 * x^2 + 3 * y^2 = 12

-- Definition of the function we want to maximize
def func (x y : ℝ) := x + 2 * y

-- The theorem to prove that the maximum value of x + 2y on the ellipse is √22
theorem max_value (x y : ℝ) (h : ellipse x y) : ∃ θ : ℝ, func x y ≤ Real.sqrt 22 :=
by
  sorry

end max_value_l300_300829


namespace nine_by_one_tiling_l300_300167

theorem nine_by_one_tiling (M : ℕ) 
  (h1 : ∀ T : Finset ℕ,
  (T.card = 9) → 
  (∀ x ∈ T, x ≤ 9) → 
  (∀ t ∈ T, t = m ∧ m ∈ {1,2,3,4,5,6,7,8,9}) → 
  (∀ c ∈ {red, blue, green}, ∃ t ∈ T, t = c) → 
  ∑ t in T, t * t = M) : 
  M % 1000 = 990 := 
sorry

end nine_by_one_tiling_l300_300167


namespace intersection_eq_l300_300394

def setA (x : ℝ) : Prop := (x ≥ 1) ∨ (x ≤ -1)
def setB (y : ℝ) : Prop := (y ≥ 0)
def intersectionAB (x : ℝ) : Prop := (setA x) ∧ (setB (sqrt (x^2 - 1)))

theorem intersection_eq {x : ℝ} : setA x → setB (sqrt (x^2 - 1)) → (x ≥ 1) :=
begin
  intro hx,
  intro hy,
  cases hx,
  { exact hx },
  { exfalso,
    have : sqrt (x^2 - 1) < 0,
    { sorry },
    exact (not_lt_of_ge hy this)
  }
end

end intersection_eq_l300_300394


namespace cos_arcsin_l300_300245

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l300_300245


namespace correct_calculation_l300_300920

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end correct_calculation_l300_300920


namespace area_of_polar_figure_eq_2pi_l300_300856

open Real

/-- Given the polar equation ρ = 2√2cos(π/4 - θ) is 2π. -/
theorem area_of_polar_figure_eq_2pi 
    (polar_eq : ∀ θ, ∃ ρ, ρ = 2 * √2 * cos (π / 4 - θ)) :
    ∃ S, S = 2 * π := 
by 
    sorry

end area_of_polar_figure_eq_2pi_l300_300856


namespace true_discount_is_180_l300_300884

-- Definitions based on the conditions
def face_value : ℝ := 1680
def rate_per_annum : ℝ := 0.16
def time_years : ℝ := 9 / 12

-- Calculation of present value
def present_value (fv : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  fv / (1 + (r * t))

-- Calculation of true discount
def true_discount (fv : ℝ) (pv : ℝ) : ℝ :=
  fv - pv

-- Theorem statement
theorem true_discount_is_180 : true_discount face_value (present_value face_value rate_per_annum time_years) = 180 :=
by
  sorry

end true_discount_is_180_l300_300884


namespace find_red_ball_count_l300_300173

variables (n w g y p : ℕ)
variables (P_not_red_nor_purple : ℝ)

def total_ball_count (n : ℕ) : Prop := n = 60
def white_ball_count (w : ℕ) : Prop := w = 22
def green_ball_count (g : ℕ) : Prop := g = 18
def yellow_ball_count (y : ℕ) : Prop := y = 5
def purple_ball_count (p : ℕ) : Prop := p = 9
def prob_not_red_nor_purple (P_not_red_nor_purple : ℝ) : Prop := P_not_red_nor_purple = 0.75

theorem find_red_ball_count (R : ℕ) (hn : total_ball_count n) (hw : white_ball_count w) 
    (hg : green_ball_count g) (hy : yellow_ball_count y) (hp : purple_ball_count p) 
    (hP : prob_not_red_nor_purple P_not_red_nor_purple) : R = 6 :=
by
  have h_wgy := w + g + y = 45, by sorry
  have h_norp := n * P_not_red_nor_purple = 45, by sorry
  have h_R := n - (45 + p) = 6, by sorry
  exact h_R

end find_red_ball_count_l300_300173


namespace find_q_and_a_n_sum_abs_b_n_l300_300302

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (a b c : ℝ) := 2 * b = a + c

variables (S_3_eq : S 3 = 7) 
          (a1_a2_a3_ari_seq : arithmetic_sequence (a_n 1 + 3) (3 * a_n 2) (a_n 3 + 4))

theorem find_q_and_a_n :
  (∃ q : ℝ, q = 2 ∨ q = 1 / 2) ∧
  (∀ n : ℕ, a_n n = if q = 2 then 2 ^ (n - 1) else if q = 1 / 2 then 2 ^ (3 - n) else 0) :=
sorry

variable (increasing_sequence : ∀ n : ℕ, a_n n < a_n (n + 1))

def log_base_2 (x : ℝ) := real.log x / real.log 2

theorem sum_abs_b_n (n : ℕ) (a_increasing : ∀ k, a_n k < a_n (k + 1)) :
  let b_n := λ k, log_base_2 (a_n (k + 1) / 128),
      sum_b := ∑ k in finset.range n, abs (b_n (k + 1)) in
  sum_b = if n <= 7 then (n * (13 - n)) / 2
          else (n * (n - 13)) / 2 + 42 :=
sorry

end find_q_and_a_n_sum_abs_b_n_l300_300302


namespace largest_whole_number_lt_150_l300_300140

theorem largest_whole_number_lt_150 : 
  ∃ x : ℕ, (9 * x < 150) ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) :=
  sorry

end largest_whole_number_lt_150_l300_300140


namespace range_of_x_l300_300327

noncomputable def f (x : ℝ) : ℝ := log (1 / real.exp 1) (x^2 + 1 / real.exp 1) - abs (x / real.exp 1)

theorem range_of_x (x : ℝ) : 0 < x ∧ x < 2 → f (x + 1) < f (2 * x - 1) :=
begin
  sorry
end

end range_of_x_l300_300327


namespace area_units_ordered_correctly_l300_300857

def area_units :=
  ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"]

theorem area_units_ordered_correctly :
  area_units = ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"] :=
by
  sorry

end area_units_ordered_correctly_l300_300857


namespace sam_driving_distance_l300_300421

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300421


namespace cos_arcsin_l300_300241

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l300_300241


namespace center_of_symmetry_l300_300987

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/--
  Given the function f(x) := sin x - sqrt(3) * cos x,
  prove that (π/3, 0) is the center of symmetry for f.
-/
theorem center_of_symmetry : symmetry_center (fun x => Real.sin x - Real.sqrt 3 * Real.cos x) (Real.pi / 3, 0) :=
by
  sorry

end center_of_symmetry_l300_300987


namespace rotations_needed_to_reach_goal_l300_300340

-- Define the given conditions
def rotations_per_block : ℕ := 200
def blocks_goal : ℕ := 8
def current_rotations : ℕ := 600

-- Define total_rotations_needed and more_rotations_needed
def total_rotations_needed : ℕ := blocks_goal * rotations_per_block
def more_rotations_needed : ℕ := total_rotations_needed - current_rotations

-- Theorem stating the solution
theorem rotations_needed_to_reach_goal : more_rotations_needed = 1000 := by
  -- proof steps are omitted
  sorry

end rotations_needed_to_reach_goal_l300_300340


namespace speed_with_stream_l300_300961

-- Definitions for the conditions in part a
def Vm : ℕ := 8  -- Speed of the man in still water (in km/h)
def Vs : ℕ := Vm - 4  -- Speed of the stream (in km/h), derived from man's speed against the stream

-- The statement to prove the man's speed with the stream
theorem speed_with_stream : Vm + Vs = 12 := by sorry

end speed_with_stream_l300_300961


namespace arithmetic_sequence_15th_term_l300_300910

theorem arithmetic_sequence_15th_term :
  ∀ (a₁ d n : ℕ), a₁ = 4 → d = 4 → n = 15 → (a₁ + (n - 1) * d) = 60 :=
by
  intros a₁ d n ha₁ hd hn
  rw [ha₁, hd, hn]
  norm_num
  reflexivity

end arithmetic_sequence_15th_term_l300_300910


namespace teresa_age_when_michiko_born_l300_300092

theorem teresa_age_when_michiko_born (teresa_current_age morio_current_age morio_age_when_michiko_born : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : morio_age_when_michiko_born = 38) : 
  teresa_current_age - (morio_current_age - morio_age_when_michiko_born) = 26 := 
by 
  sorry

end teresa_age_when_michiko_born_l300_300092


namespace convex_polyhedron_is_tetrahedron_l300_300954

-- Defining concepts related to the problem
variables (V E F : ℕ) -- Number of vertices, edges, faces
variables [ConvexPolyhedron : Type] -- Type representing convex polyhedron

-- Defining the conditions
def polyhedron_no_diagonals (P : ConvexPolyhedron) : Prop :=
  ∀ v₁ v₂ : V, ∃ e ∈ E, connects v₁ v₂ ∧ (¬∃ d ∈ P, diagonal d)

-- Proving that a convex polyhedron with no diagonals is a tetrahedron
theorem convex_polyhedron_is_tetrahedron (P : ConvexPolyhedron)
  (h1 : ∃ (V E F : ℕ), V - E + F = 2)
  (h2 : polyhedron_no_diagonals P)
  : V = 4 :=
sorry

end convex_polyhedron_is_tetrahedron_l300_300954


namespace max_distance_origin_to_curve_point_l300_300716

theorem max_distance_origin_to_curve_point :
  (∃ (θ : ℝ), ∀ (O M : ℝ × ℝ),
    O = (0, 0) ∧
    M = (3 + cos θ, sin θ) →
    sqrt((M.fst - O.fst)^2 + (M.snd - O.snd)^2) ≤ 4) :=
begin
  sorry
end

end max_distance_origin_to_curve_point_l300_300716


namespace prime_not_divisor_ab_cd_l300_300022

theorem prime_not_divisor_ab_cd {a b c d : ℕ} (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) 
  (p : ℕ) (hp : p = a + b + c + d) (hprime : Nat.Prime p) : ¬ p ∣ (a * b - c * d) := 
sorry

end prime_not_divisor_ab_cd_l300_300022


namespace distance_to_top_of_mountain_l300_300646

theorem distance_to_top_of_mountain (initial_speed : ℝ) (mountain_speed_decrease : ℝ) 
  (mountain_speed_increase : ℝ) (distance_down : ℝ) (total_time : ℝ) : ℝ :=
  have ascending_speed : ℝ := initial_speed * mountain_speed_decrease,
  have descending_speed : ℝ := initial_speed * mountain_speed_increase,
  have time_descending : ℝ := distance_down / descending_speed,
  have time_ascending : ℝ := total_time - time_descending,
  ascending_speed * time_ascending

#eval distance_to_top_of_mountain 30 0.5 1.2 72 6 -- Expected output: 60

end distance_to_top_of_mountain_l300_300646


namespace cut_square_into_rectangles_l300_300259

theorem cut_square_into_rectangles :
  ∃ rectangles : list (ℝ × ℝ), 
    (∀ (w h : ℝ), (w, h) ∈ rectangles → w * h > 0) ∧ 
    (∑ (w, h) in rectangles, 2 * (w + h)) = 25 ∧ 
    (∑ (w, h) in rectangles, w * h) = 4 * 4 := 
sorry

end cut_square_into_rectangles_l300_300259


namespace triangle_ABC_equilateral_l300_300790

-- Define the basic elements of the problem
variable {A B C A1 B1 C1 : Type} [IsTriangle A B C] -- A, B, C form a triangle
variable (A1_on_BC : A1 ∈ LineSegment B C)
variable (B1_on_AC : B1 ∈ LineSegment A C)
variable (C1_on_AB : C1 ∈ LineSegment A B)

-- Altitude from A to BC
axiom Altitude_A_A1 : IsAltitude A A1 (LineSegment B C)

-- Median from B to AC
axiom Median_B_B1 : IsMedian B B1 (LineSegment A C)

-- Angle bisector from C to AB
axiom AngleBisector_C_C1 : IsAngleBisector C C1 (LineSegment A B)

-- Equilateral triangle A1B1C1
axiom Equilateral_A1B1C1 : IsEquilateral (Triangle A1 B1 C1)

-- Main theorem to prove: Triangle ABC is equilateral
theorem triangle_ABC_equilateral : IsEquilateral (Triangle A B C) := 
by 
  sorry

end triangle_ABC_equilateral_l300_300790


namespace concyclic_points_l300_300043

theorem concyclic_points {A B C D S E F : Point}
  (h_circ : Circle A B C D)
  (h_midpoint : MidpointArc S ⟨A, B⟩ ¬Contains C D)
  (h_inter_SD_E : Intersects (Line S D) (Line A B) E)
  (h_inter_SC_F : Intersects (Line S C) (Line A B) F) :
  Concyclics C D E F := by
  sorry

end concyclic_points_l300_300043


namespace nature_of_roots_l300_300261

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 - 7 * x^3 - 2 * x + 9

theorem nature_of_roots : 
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P 0 * P x < 0 := 
by {
  sorry
}

end nature_of_roots_l300_300261


namespace log_domain_l300_300100

theorem log_domain :
  { x : ℝ | 2 * x + 1 > 0 } = set.Ioi (-1 / 2) := by
sorry

end log_domain_l300_300100


namespace sam_distance_traveled_l300_300449

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300449


namespace complex_conjugate_magnitude_l300_300050

theorem complex_conjugate_magnitude (z : ℂ) (h : z * complex.I + 1 = z) : complex.abs (conj z) = real.sqrt 2 / 2 :=
by sorry

end complex_conjugate_magnitude_l300_300050


namespace polar_to_cartesian_parabola_l300_300499

theorem polar_to_cartesian_parabola (ρ θ : ℝ) (h : ρ * cos θ ^ 2 = 4 * sin θ) : 
  ∃ x y : ℝ, (x = ρ * cos θ ∧ y = ρ * sin θ) ∧ x^2 = 4 * y :=
by
  sorry

end polar_to_cartesian_parabola_l300_300499


namespace sequence_of_8_numbers_l300_300369

theorem sequence_of_8_numbers :
  ∃ (a b c d e f g h : ℤ), 
    a + b + c = 100 ∧ b + c + d = 100 ∧ c + d + e = 100 ∧ 
    d + e + f = 100 ∧ e + f + g = 100 ∧ f + g + h = 100 ∧ 
    a = 20 ∧ h = 16 ∧ 
    (a, b, c, d, e, f, g, h) = (20, 16, 64, 20, 16, 64, 20, 16) :=
by
  sorry

end sequence_of_8_numbers_l300_300369


namespace sprint_team_total_miles_l300_300117

theorem sprint_team_total_miles (number_of_people : ℝ) (miles_per_person : ℝ) 
  (h1 : number_of_people = 150.0) (h2 : miles_per_person = 5.0) : 
  number_of_people * miles_per_person = 750.0 :=
by
  rw [h1, h2]
  norm_num

end sprint_team_total_miles_l300_300117


namespace triangle_construction_l300_300383

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def is_median (A A₀ D : Point) : Prop := sorry
noncomputable def is_angle_bisector (B A A₀ D : Point) : Prop := sorry
noncomputable def perpend_bisector_intersection (A C : Point) : Point := sorry
noncomputable def triangle_construct (A A₀ D E : Point) : Prop := ∃ B C, midpoint B C = A₀ ∧ is_median A A₀ D ∧ 
                                          is_angle_bisector B A A₀ D ∧ E = perpend_bisector_intersection A C

theorem triangle_construction (A A₀ D E : Point) : triangle_construct A A₀ D E :=
  sorry

end triangle_construction_l300_300383


namespace joe_time_to_friends_house_l300_300148

theorem joe_time_to_friends_house
  (feet_moved : ℕ) (time_taken : ℕ) (remaining_distance : ℕ) (feet_in_yard : ℕ)
  (rate_of_movement : ℕ) (remaining_distance_feet : ℕ) (time_to_cover_remaining_distance : ℕ) :
  feet_moved = 80 →
  time_taken = 40 →
  remaining_distance = 90 →
  feet_in_yard = 3 →
  rate_of_movement = feet_moved / time_taken →
  remaining_distance_feet = remaining_distance * feet_in_yard →
  time_to_cover_remaining_distance = remaining_distance_feet / rate_of_movement →
  time_to_cover_remaining_distance = 135 :=
by
  sorry

end joe_time_to_friends_house_l300_300148


namespace sum_of_x_coordinates_l300_300826

theorem sum_of_x_coordinates : 
  let points := [(3, 10), (6, 20), (12, 35), (18, 40), (20, 50)]
  let line_eq := fun x => 2 * x + 7
  let is_above_line : (ℕ × ℕ) → Prop := λ p, p.snd > line_eq p.fst
  (points.filter is_above_line).map Prod.fst = [6, 12, 20] →
  (points.filter is_above_line).map Prod.fst.sum = 38 :=
by 
  sorry

end sum_of_x_coordinates_l300_300826


namespace S_l300_300807

noncomputable def S' : set ℝ := {x | 0 < x ∧ x < π / 2 ∧ (∃ a b c : ℝ, {a, b, c} = {sin x, cos x, cot x} ∧ a^2 + b^2 = c^2)}

def sum_cot_squared_over_S' : ℝ := ∑ x in S', cot x ^ 2

theorem S'_sum_is_sqrt2 : 
  sum_cot_squared_over_S' = sqrt 2 := 
sorry

end S_l300_300807


namespace distribution_schemes_count_l300_300660

noncomputable def number_of_distribution_schemes 
  (slots : ℕ) (schools : ℕ) (min_slots_A : ℕ) (min_slots_B : ℕ) : ℕ :=
  if slots = 7 ∧ schools = 5 ∧ min_slots_A = 2 ∧ min_slots_B = 2 then 35 else 0

theorem distribution_schemes_count :
  number_of_distribution_schemes 7 5 2 2 = 35 :=
by
  sorry

end distribution_schemes_count_l300_300660


namespace curve_length_integral_l300_300833

noncomputable def curve_length (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
∫ x in a..b, real.sqrt (1 + (deriv f x)^2)

theorem curve_length_integral (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x) :
    ∫ x in a..b, real.sqrt (1 + (deriv f x)^2) = curve_length f a b :=
by
  sorry

end curve_length_integral_l300_300833


namespace sheena_sewing_weeks_l300_300084

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l300_300084


namespace Tim_earnings_correct_l300_300129

def visitors_day := 100
def first_6_days := 6 * visitors_day
def last_day := 2 * first_6_days
def total_visitors := first_6_days + last_day
def earnings_per_visitor := 0.01
def total_earnings := total_visitors * earnings_per_visitor

theorem Tim_earnings_correct :
  total_earnings = 18 := by
  sorry

end Tim_earnings_correct_l300_300129


namespace intersection_A_B_l300_300397

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end intersection_A_B_l300_300397


namespace value_of_2_star_3_l300_300510

def star (a b : ℕ) : ℕ := a * b ^ 3 - b + 2

theorem value_of_2_star_3 : star 2 3 = 53 :=
by
  -- This is where the proof would go
  sorry

end value_of_2_star_3_l300_300510


namespace martha_total_cost_l300_300130

-- Definitions for the conditions
def amount_cheese_needed : ℝ := 1.5 -- in kg
def amount_meat_needed : ℝ := 0.5 -- in kg
def cost_cheese_per_kg : ℝ := 6.0 -- in dollars per kg
def cost_meat_per_kg : ℝ := 8.0 -- in dollars per kg

-- Total cost that needs to be calculated
def total_cost : ℝ :=
  (amount_cheese_needed * cost_cheese_per_kg) +
  (amount_meat_needed * cost_meat_per_kg)

-- Statement of the theorem
theorem martha_total_cost : total_cost = 13 := by
  sorry

end martha_total_cost_l300_300130


namespace valid_parameterizations_l300_300479

-- Define the line y = -3x + 5
def line_eq (p : ℝ × ℝ) : Prop := p.2 = -3 * p.1 + 5

-- Define the validity of the parameterization
def is_valid_param (point dir : ℝ × ℝ) : Prop :=
  (line_eq point) ∧ (dir.2 / dir.1 = -3)

-- Define the points and direction vectors
def P1 := (1, 2)
def D1 := (3, -1)

def P2 := (-3/2, 5)
def D2 := (1, -3)

def P3 := (0, 5)
def D3 := (-1, 3)

def P4 := (2, -1)
def D4 := (2, -6)

def P5 := (-1, 8)
def D5 := (2/3, -2)

-- Create the theorem to prove the valid parameterizations
theorem valid_parameterizations :
  (is_valid_param P3 D3) ∧ (is_valid_param P5 D5) ∧
  ¬(is_valid_param P1 D1) ∧ ¬(is_valid_param P2 D2) ∧ ¬(is_valid_param P4 D4) := 
by
  sorry

end valid_parameterizations_l300_300479


namespace constant_term_in_expansion_l300_300784

-- Define the conditions
def binomial_expansion_flat (x : ℝ) : ℝ := (x / 2 - 1 / (3 * x))

-- Define the problem statement
theorem constant_term_in_expansion (x : ℝ) (n : ℕ)
  (h : n/2 + 1 = 5) :
  ∃ c : ℝ, c = constant_term (binomial_expansion_flat x)^n ∧ c = 7 :=
by { sorry }

end constant_term_in_expansion_l300_300784


namespace b_general_formula_sum_c_sequence_l300_300307

-- Conditions
variable (a b c : ℕ → ℕ) (S : ℕ → ℕ)
variable [Nonempty ℕ]

axiom a_sum_condition : ∀ n, S n = n^2
axiom b_geometric_condition1 : b 1 = a 1
axiom b_geometric_condition2 : 2 * b 3 = b 4
axiom c_definition : ∀ n, c n = a n * b n

-- Proof goals
theorem b_general_formula (n : ℕ) : b n = 2^(n-1) := sorry

theorem sum_c_sequence (T : ℕ → ℕ) (n : ℕ) 
  (hsum : ∀ k, (∑ i in Finset.range k, c i) = T k) : 
  T n = (2*n - 3) * 2^n + 3 := sorry

end b_general_formula_sum_c_sequence_l300_300307


namespace area_of_defined_region_l300_300901

theorem area_of_defined_region : 
  ∃ (A : ℝ), (∀ x y : ℝ, |4 * x - 20| + |3 * y + 9| ≤ 6 → A = 9) :=
sorry

end area_of_defined_region_l300_300901


namespace units_digit_2019_pow_2019_l300_300943

theorem units_digit_2019_pow_2019 : (2019^2019) % 10 = 9 := 
by {
  -- The statement of the problem is proved below
  sorry  -- Solution to be filled in
}

end units_digit_2019_pow_2019_l300_300943


namespace hare_total_distance_l300_300070

-- Define the conditions
def distance_between_trees : ℕ := 5
def number_of_trees : ℕ := 10

-- Define the question to be proved
theorem hare_total_distance : distance_between_trees * (number_of_trees - 1) = 45 :=
by
  sorry

end hare_total_distance_l300_300070


namespace sample_size_is_fifteen_l300_300177

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ)
variable (elderly_employees : ℕ) (young_sample_count : ℕ) (sample_size : ℕ)

theorem sample_size_is_fifteen
  (h1 : total_employees = 750)
  (h2 : young_employees = 350)
  (h3 : middle_aged_employees = 250)
  (h4 : elderly_employees = 150)
  (h5 : 7 = young_sample_count)
  : sample_size = 15 := 
sorry

end sample_size_is_fifteen_l300_300177


namespace smallest_positive_period_value_of_a_l300_300746

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + (Real.pi / 6)) + 3

theorem smallest_positive_period : (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi) ∧
  (∀ k : ℤ, ∃ a b, a = k * Real.pi + (Real.pi / 6) ∧ b = k * Real.pi + (2 * Real.pi / 3) ∧ 
    ∀ x, a ≤ x ∧ x ≤ b → f x = 2 * sin (2 * x + (Real.pi / 6)) + 3 ∧ 
    (∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → x₂ ≤ b → f x₁ ≥ f x₂)) :=
sorry

theorem value_of_a (A : ℝ) (b : ℝ) (area : ℝ) (a : ℝ) 
  (h1 : f A = 4) (h2 : b = 1) (h3 : area = Real.sqrt 3 / 2) : 
  a = Real.sqrt 3 :=
sorry

end smallest_positive_period_value_of_a_l300_300746


namespace intersection_A_B_l300_300396

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end intersection_A_B_l300_300396


namespace solve_for_x_l300_300689

theorem solve_for_x (x : ℝ) (h : (sqrt (8 * x)) / (sqrt (5 * (x - 2))) = 3) : x = 90 / 37 :=
by
  sorry

end solve_for_x_l300_300689


namespace percent_calculation_l300_300578

theorem percent_calculation (Part Whole : ℝ) (h1 : Part = 120) (h2 : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  sorry

end percent_calculation_l300_300578


namespace sam_distance_traveled_l300_300447

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l300_300447


namespace max_distance_PC_l300_300970

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_PC (P : ℝ × ℝ) (u v w : ℝ) 
  (hA : distance P (0, 0) = u)
  (hB : distance P (1, 0) = v)
  (hD : distance P (0, 1) = w)
  (h_condition : u^2 + w^2 = 2 * v^2) :
  ∃ P : ℝ × ℝ, distance P (1, 1) = real.sqrt (0.5) := 
by
  sorry

end max_distance_PC_l300_300970


namespace maximum_area_triangle_l300_300300

theorem maximum_area_triangle (r : ℝ) (P Q R : ℝ × ℝ)
  (h_circle : dist P (0, 0) = r) 
  (h_tangent_point : dist P (0, 0) = r)
  (h_variable_point : dist R (0, 0) = r)
  (h_perpendicular : R.2 = Q.2 ∧ (Q.1 = P.1 ∨ Q.1 = P.1 ∧ Q.2 < P.2)) :
  ∃ (φ : ℝ), 0 ≤ φ ∧ φ ≤ π/2 ∧ 
    (2 * r^2 * (sin φ)^3 * (cos φ) = r^2 * (sqrt 3) / 8) := 
sorry

end maximum_area_triangle_l300_300300


namespace a_equals_b_l300_300041

theorem a_equals_b (a b : ℕ) (h : a^3 + a + 4 * b^2 = 4 * a * b + b + b * a^2) : a = b := 
sorry

end a_equals_b_l300_300041


namespace bankers_discount_l300_300953

noncomputable def principal := 180000
noncomputable def r1 := 0.12
noncomputable def r2 := 0.14
noncomputable def r3 := 0.16
noncomputable def t := 3 / 12

noncomputable def interest_first_3_months := principal * r1 * t
noncomputable def interest_next_3_months := principal * r2 * t
noncomputable def interest_last_3_months := principal * r3 * t

noncomputable def total_interest := interest_first_3_months + interest_next_3_months + interest_last_3_months

theorem bankers_discount : total_interest = 18900 := by
  sorry

end bankers_discount_l300_300953


namespace percentage_hindus_l300_300774

variables (total_boys : ℕ) (percent_muslims percent_sikhs : ℕ) (boys_other_communities : ℕ)

-- Given conditions
def condition1 := total_boys = 850
def condition2 := percent_muslims = 46
def condition3 := percent_sikhs = 10
def condition4 := boys_other_communities = 136

-- Statement to prove
theorem percentage_hindus (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  let number_muslims    := (percent_muslims * total_boys) / 100
      number_sikhs      := (percent_sikhs * total_boys) / 100
      number_non_hindus := number_muslims + number_sikhs + boys_other_communities
      number_hindus     := total_boys - number_non_hindus
      percent_hindus    := (number_hindus * 100) / total_boys
  in percent_hindus = 28 :=
sorry

end percentage_hindus_l300_300774


namespace calculator_reciprocal_l300_300865

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem calculator_reciprocal (initial : ℝ) (target : ℝ) (h_initial : initial = 0.04) (h_target : target = 0.04) :
  reciprocal (reciprocal initial) = target :=
by
  calc
    reciprocal (reciprocal initial)
        = reciprocal (1 / initial) : by rw [reciprocal]
    ... = 1 / (1 / initial) : by rw [reciprocal]
    ... = initial : by field_simp [ne_of_gt (show 0 < initial, by norm_num [h_initial])]
    ... = target : by rw [h_initial, h_target]

end calculator_reciprocal_l300_300865


namespace probability_of_exactly_one_defective_l300_300291

theorem probability_of_exactly_one_defective:
  let total_products := 6
  let genuine_products := 5
  let defective_products := 1
  -- selecting 2 products out of total_products
  let ways_to_choose_two := nat.choose total_products 2
  -- selecting 1 genuine and 1 defective product
  let ways_to_choose_one_genuine_one_defective := genuine_products * defective_products
  -- calculating probability
  (ways_to_choose_one_genuine_one_defective : ℚ) / ways_to_choose_two = 1 / 3 :=
by {
  sorry
}

end probability_of_exactly_one_defective_l300_300291


namespace cos_arcsin_l300_300242

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l300_300242


namespace horizontally_shift_graph_move_graph_right_by_pi_over_5_l300_300894

theorem horizontally_shift_graph (x : ℝ) :
  (3 * sin(2 * x - π / 5)) = (3 * sin(2 * (x - π / 5))) :=
by
  sorry

theorem move_graph_right_by_pi_over_5 :
  ∀ x, 3 * sin(2 * x - π / 5) = 3 * sin(2 * (x - (π / 5))) ∧
  ∀ x, 3 * sin(2 * x + π / 5) = 3 * sin(2 * (x + π / 5)) →
  ∃ t, ∀ x, 3 * sin(2 * (x + t)) = 3 * sin(2 * (x - π / 5)) ∧ t = - π / 5 :=
by
  intros x h
  use - (π / 5)
  sorry

end horizontally_shift_graph_move_graph_right_by_pi_over_5_l300_300894


namespace particles_meeting_angle_l300_300534

theorem particles_meeting_angle :
  ∃ t : ℕ, (295 = (15 * t + (t * (3 * t - 1) / 2))) → 
  ((t : ℝ) * 6 = 60) :=
begin
  sorry
end

end particles_meeting_angle_l300_300534


namespace min_shirts_to_save_money_l300_300982

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), (50 + 9 * (x : ℝ) < 14 * (x : ℝ)) ∧ (x = 11) :=
begin
  sorry
end

end min_shirts_to_save_money_l300_300982


namespace prob_intersects_two_points_l300_300077

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def line_eq (k x : ℝ) : ℝ :=
  k * (x - 2)

def distance_from_center (k : ℝ) : ℝ :=
  |2 * k| / real.sqrt (k^2 + 1)

def intersects_two_points (k : ℝ) : Prop :=
  distance_from_center k < 1

noncomputable def probability (a b : ℝ) (f : ℝ → Prop) : ℝ :=
  let count := ∫ x in a..b, if f x then 1 else 0
  count / (b - a)

theorem prob_intersects_two_points :
  probability (-1) 1 intersects_two_points = real.sqrt(3) / 3 := sorry

end prob_intersects_two_points_l300_300077


namespace problem_conditions_l300_300311

noncomputable def f (x : ℝ) := x^2 - 2 * x * Real.log x
noncomputable def g (x : ℝ) := Real.exp x - (Real.exp 2 * x^2) / 4

theorem problem_conditions :
  (∀ x > 0, deriv f x > 0) ∧ 
  (∃! x, g x = 0) ∧ 
  (∃ x, f x = g x) :=
by
  sorry

end problem_conditions_l300_300311


namespace pages_written_in_a_year_l300_300013

-- Definitions based on conditions
def pages_per_letter : ℕ := 3
def letters_per_week : ℕ := 2
def friends : ℕ := 2
def weeks_per_year : ℕ := 52

-- Definition to calculate total pages written in a week
def weekly_pages (pages_per_letter : ℕ) (letters_per_week : ℕ) (friends : ℕ) : ℕ :=
  pages_per_letter * letters_per_week * friends

-- Definition to calculate total pages written in a year
def yearly_pages (weekly_pages : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weekly_pages * weeks_per_year

-- Theorem to prove the total pages written in a year
theorem pages_written_in_a_year : yearly_pages (weekly_pages pages_per_letter letters_per_week friends) weeks_per_year = 624 :=
by 
  sorry

end pages_written_in_a_year_l300_300013


namespace num_values_of_c_l300_300287

theorem num_values_of_c : 
  let possible_c (c : ℤ) := ∃ x : ℝ, 5 * ⌊x⌋ + 3 * ⌈x⌉ = c
  in (∃ c : ℤ, 0 ≤ c ∧ c ≤ 2000 ∧ possible_c c) = 501 :=
sorry

end num_values_of_c_l300_300287


namespace solution_set_of_inequality_l300_300114

-- Definition of the inequality and its transformation
def inequality (x : ℝ) : Prop :=
  (x - 2) / (x + 1) ≤ 0

noncomputable def transformed_inequality (x : ℝ) : Prop :=
  (x + 1) * (x - 2) ≤ 0 ∧ x + 1 ≠ 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | -1 < x ∧ x ≤ 2} := 
sorry

end solution_set_of_inequality_l300_300114


namespace train_speed_is_72_kmh_l300_300623

-- Define the conditions
def train_length : ℝ := 100  -- in meters
def time_to_pass_pole : ℝ := 5  -- in seconds

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Define the consequent speed in km/hr
def speed_in_kmh : ℝ := (train_length / time_to_pass_pole) * conversion_factor

-- The theorem to prove
theorem train_speed_is_72_kmh : speed_in_kmh = 72 := by
  -- sorry is a placeholder for the actual proof.
  sorry 

end train_speed_is_72_kmh_l300_300623


namespace average_square_feet_is_320000_l300_300513

noncomputable def population : ℕ := 331000000
noncomputable def area_miles : ℕ := 3796742
noncomputable def square_feet_per_mile : ℕ := 5280 * 5280
noncomputable def total_square_feet : ℕ := area_miles * square_feet_per_mile
noncomputable def average_square_feet_per_person : ℕ := total_square_feet / population

theorem average_square_feet_is_320000 : average_square_feet_per_person ≈ 320000 := by
  sorry

end average_square_feet_is_320000_l300_300513


namespace sam_driving_distance_l300_300425

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300425


namespace exponent_division_is_equal_l300_300628

variable (a : ℝ) 

theorem exponent_division_is_equal :
  (a^11) / (a^2) = a^9 := 
sorry

end exponent_division_is_equal_l300_300628


namespace find_difference_l300_300036

noncomputable def expression (x y : ℝ) : ℝ :=
  (|x + y| / (|x| + |y|))^2

theorem find_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0
  let M := 1
  M - m = 1 :=
by
  -- Please note that the proof is omitted and replaced with sorry
  sorry

end find_difference_l300_300036


namespace Q_transformed_correct_l300_300109

variables {x y : ℝ}
def rotated_point (p : ℝ × ℝ) (c : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (cx, cy) := c
  let cosθ := Real.cos θ
  let sinθ := Real.sin θ
  let x' := cosθ * (px - cx) - sinθ * (py - cy) + cx
  let y' := sinθ * (px - cx) + cosθ * (py - cy) + cy
  (x', y')

def scaled_point (p : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (s * px, s * py)

def reflected_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def transformed_point (Q : ℝ × ℝ) : ℝ × ℝ :=
  reflected_point (scaled_point (rotated_point Q (2, 3) (-Real.pi / 4)) 2)

theorem Q_transformed_correct (Q : ℝ × ℝ) 
  (h : transformed_point Q = (14, 2)) :
  let (x, y) := Q
  y - x = 6.66 :=
by sorry

end Q_transformed_correct_l300_300109


namespace correct_mark_l300_300609

theorem correct_mark (wrong_mark : ℕ) 
                     (n_pupils : ℕ) 
                     (avg_increase : ℕ) 
                     (h1 : wrong_mark = 85) 
                     (h2 : n_pupils = 80) 
                     (h3 : avg_increase = 1/2) : 
                     (correct_mark : ℕ) 
                     := 
begin
  have increase_in_marks : ℕ := n_pupils * avg_increase,
  have total_increase : ℕ := wrong_mark - correct_mark,
  have equation : total_increase = increase_in_marks,
  calc correct_mark = 85 - 40 : by { rw [h1, h2, h3, equation], sorry }
                  ... = 45   : by sorry
end

end correct_mark_l300_300609


namespace gcd_factorial_l300_300138

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l300_300138


namespace prime_if_floor_sum_eq_l300_300074

theorem prime_if_floor_sum_eq (N : ℕ) 
  (h : (Finset.range (N + 1)).sum (λ k, ⌊(N : ℚ) / (k + 1)⌋) = 
       2 + (Finset.range N).sum (λ k, ⌊((N - 1) : ℚ) / (k + 1)⌋)) : 
  Nat.Prime N :=
sorry

end prime_if_floor_sum_eq_l300_300074


namespace hexagon_inequality_l300_300816

theorem hexagon_inequality (A B C D E F : Point)
  (h1 : distance A B = distance B C)
  (h2 : distance C D = distance D E)
  (h3 : distance E F = distance F A) :
  (distance B C / distance B E) + (distance D E / distance D A) + (distance F A / distance F C) >= 3 / 2 :=
begin
  sorry
end

end hexagon_inequality_l300_300816


namespace savings_calculation_l300_300971

-- Definitions of the given conditions
def window_price : ℕ := 100
def free_window_offer (purchased : ℕ) : ℕ := purchased / 4

-- Number of windows needed
def dave_needs : ℕ := 7
def doug_needs : ℕ := 8

-- Calculations based on the conditions
def individual_costs : ℕ :=
  (dave_needs - free_window_offer dave_needs) * window_price +
  (doug_needs - free_window_offer doug_needs) * window_price

def together_costs : ℕ :=
  let total_needs := dave_needs + doug_needs
  (total_needs - free_window_offer total_needs) * window_price

def savings : ℕ := individual_costs - together_costs

-- Proof statement
theorem savings_calculation : savings = 100 := by
  sorry

end savings_calculation_l300_300971


namespace todd_initial_gum_l300_300575

theorem todd_initial_gum (g_s g_final g_added : ℕ) :
  g_added = 16 → g_final = 54 → g_final - g_added = 38 :=
by
  intros h1 h2
  rw [h1, h2]
  exact calc
    54 - 16 = 38 : by sorry

end todd_initial_gum_l300_300575


namespace find_three_digit_number_l300_300677

theorem find_three_digit_number :
  ∃ (Π B Γ : ℕ), Π ≠ B ∧ B ≠ Γ ∧ Π ≠ Γ ∧ Π < 10 ∧ B < 10 ∧ Γ < 10 ∧ 
  (Π * 100 + B * 10 + Γ = (Π + B + Γ) * (Π + B + Γ + 1)) ∧ 
  (Π * 100 + B * 10 + Γ = 156) :=
sorry

end find_three_digit_number_l300_300677


namespace AlfredRepairsCost_l300_300984

theorem AlfredRepairsCost 
  (PurchasePrice : ℝ := 4400) 
  (SellingPrice : ℝ := 5800) 
  (GainPercent : ℝ := 0.1154) : 
  let R := (SellingPrice - PurchasePrice * (1 + GainPercent)) / (1 + GainPercent) in
  R = 800 :=
by
  let R := (SellingPrice - PurchasePrice * (1 + GainPercent)) / (1 + GainPercent)
  show R = 800
  sorry

end AlfredRepairsCost_l300_300984


namespace factorize_expression_l300_300270

theorem factorize_expression (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) :=
by sorry

end factorize_expression_l300_300270


namespace three_digit_number_l300_300161

theorem three_digit_number (a b : ℕ) (ha : a < 10) (hb : 10 ≤ b ∧ b < 100) :
  100 * a + b = nat.of_digits 10 [a] + b := by
sorry

end three_digit_number_l300_300161


namespace area_of_triangle_BPQ_l300_300965

theorem area_of_triangle_BPQ :
  ∀ (A B C D P Q R : Type) [has_dist A B C D] [has_measure A B C D]
    (AB AD AC : ℝ) (PQ : ℝ),
  AB = 8 → AD = 6 → AC = 10 → PQ = 2.5 →
  ∃ (BPQ_area : ℝ), BPQ_area = 6 :=
begin
  sorry,
end

end area_of_triangle_BPQ_l300_300965


namespace plane_split_into_regions_l300_300256

theorem plane_split_into_regions (S : set (ℝ × ℝ)) (hx3y : ∀ p ∈ S, p.2 = 3 * p.1 ∨ p.2 = (1 / 3) * p.1) : 
  ∃ (n : ℕ), n = 4 ∧ ∀ x y, (y = 3 * x ∨ y = (1 / 3) * x) → divides_plane_into_regions S n :=
sorry

end plane_split_into_regions_l300_300256


namespace f_decreasing_interval_triangle_abc_l300_300328

noncomputable def f (x : Real) : Real := 2 * (Real.sin x)^2 + Real.cos ((Real.pi) / 3 - 2 * x)

theorem f_decreasing_interval :
  ∃ (a b : Real), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  ∀ x y, (a ≤ x ∧ x < y ∧ y ≤ b) → f y ≤ f x := 
sorry

variables {a b c : Real} (A B C : Real) 

theorem triangle_abc (h1 : A = Real.pi / 3) 
    (h2 : f A = 2)
    (h3 : a = 2 * b)
    (h4 : Real.sin C = 2 * Real.sin B):
  a / b = Real.sqrt 3 := 
sorry

end f_decreasing_interval_triangle_abc_l300_300328


namespace count_valid_numbers_l300_300684

def is_three_digit_ending_in_zero (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 0

def divides (x y : ℕ) : Prop :=
  y % x = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def leaves_remainder_1_when_div_by_3 (n : ℕ) : Prop :=
  n % 3 = 1

def tens_or_hundreds_digit_divisible_by_4 (n : ℕ) : Prop :=
  divides 4 (tens_digit n) ∨ divides 4 (hundreds_digit n)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit_ending_in_zero n ∧ 
  leaves_remainder_1_when_div_by_3 n ∧ 
  tens_or_hundreds_digit_divisible_by_4 n

theorem count_valid_numbers : 
  (finset.card (finset.filter satisfies_conditions (finset.range 1000))) = 15 :=
by
  sorry

end count_valid_numbers_l300_300684


namespace sam_drove_distance_l300_300438

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300438


namespace equal_popularity_l300_300995

theorem equal_popularity :
  let drama := (8 : ℚ) / 24
  let sports := (9 : ℚ) / 27
  let art := (10 : ℚ) / 30
  let music := (7 : ℚ) / 21
  drama = sports ∧ sports = art ∧ art = music :=
by
  let drama := (8 : ℚ) / 24
  let sports := (9 : ℚ) / 27
  let art := (10 : ℚ) / 30
  let music := (7 : ℚ) / 21
  -- simplify the fractions
  have h1 : drama = (90 : ℚ) / 270 := by sorry
  have h2 : sports = (90 : ℚ) / 270 := by sorry
  have h3 : art = (90 : ℚ) / 270 := by sorry
  have h4 : music = (90 : ℚ) / 270 := by sorry
  exact ⟨h1.trans h2, h2.trans h3, h3.trans h4⟩

end equal_popularity_l300_300995


namespace incorrect_assignment_statement_l300_300916

theorem incorrect_assignment_statement :
  ¬ (∀ (N K C A B D : Type) (H1 : N = N + 1) (H2 : K = K * K) 
    (H3 : C = A * (B + D)) (H4 : C = A / B), False) :=
by sorry

end incorrect_assignment_statement_l300_300916


namespace constant_term_expansion_l300_300326

theorem constant_term_expansion :
  (∃ (a : ℝ), ∀ (x : ℝ), (x + 1/x) * (a * x - 1)^5 = 2) →
  (constant_term (expand ((x : ℝ) + 1/x) * (expand ((a * x - 1) ^ 5))) = 10) :=
by
  sorry

end constant_term_expansion_l300_300326


namespace sam_driving_distance_l300_300420

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300420


namespace servings_ratio_l300_300015

theorem servings_ratio
  (C G : ℕ) -- servings per corn and green bean plants respectively
  (carrot_servings : ℕ)
  (num_plants : ℕ)
  (total_servings : ℕ)
  (green_bean_to_corn_ratio : G = C / 2)
  (carrot_per_plant : carrot_servings = 4)
  (plants_per_plot : num_plants = 9)
  (total_servings_all_plots : total_servings = 306) :
  let total_carrot_servings := num_plants * carrot_servings in
  let remaining_servings := total_servings - total_carrot_servings in
  9 * C + 9 * G = remaining_servings →
  C / 4 = 5 :=
by
  -- proof sketch to fill
  sorry

end servings_ratio_l300_300015


namespace sheena_weeks_to_complete_l300_300082

/- Definitions -/
def time_per_dress : ℕ := 12
def number_of_dresses : ℕ := 5
def weekly_sewing_time : ℕ := 4

/- Theorem -/
theorem sheena_weeks_to_complete : (number_of_dresses * time_per_dress) / weekly_sewing_time = 15 := 
by 
  /- Proof is omitted -/
  sorry

end sheena_weeks_to_complete_l300_300082


namespace intersection_eq_l300_300395

def setA (x : ℝ) : Prop := (x ≥ 1) ∨ (x ≤ -1)
def setB (y : ℝ) : Prop := (y ≥ 0)
def intersectionAB (x : ℝ) : Prop := (setA x) ∧ (setB (sqrt (x^2 - 1)))

theorem intersection_eq {x : ℝ} : setA x → setB (sqrt (x^2 - 1)) → (x ≥ 1) :=
begin
  intro hx,
  intro hy,
  cases hx,
  { exact hx },
  { exfalso,
    have : sqrt (x^2 - 1) < 0,
    { sorry },
    exact (not_lt_of_ge hy this)
  }
end

end intersection_eq_l300_300395


namespace garden_length_is_60_l300_300966

noncomputable def garden_length (w l : ℕ) : Prop :=
  l = 2 * w ∧ 2 * w + 2 * l = 180

theorem garden_length_is_60 (w l : ℕ) (h : garden_length w l) : l = 60 :=
by
  sorry

end garden_length_is_60_l300_300966


namespace problem_conditions_part_1_part_2_part_3_l300_300869

noncomputable def f (x : ℝ) : ℝ := 10^(3 * x * (3 - x))

theorem problem_conditions (x : ℝ) (h : 0 < x ∧ x < 3) : 
  log (log (f x)) = log (3 * x) + log (3 - x) := sorry

theorem part_1 (x : ℝ) (h : 0 < x ∧ x < 3) : 
  f x = 10^(3 * x * (3 - x)) := sorry

theorem part_2 : 
  set.range f = (1, 10^(27/4)] := sorry

theorem part_3 :
  ∀ x, (3 / 2 ≤ x ∧ x < 3) → strict_mono_decr_on f (set.Icc (3 / 2) 3) := sorry

end problem_conditions_part_1_part_2_part_3_l300_300869


namespace number_of_solutions_l300_300749

open Nat

theorem number_of_solutions :
  let k_vals := finset.filter (λ k : ℕ, k = floor (Real.sqrt (80 * k - 800))) (finset.range (100)) -- assuming a reasonable upper limit on k
  (finset.sum (finset.filter (λ n : ℕ, (∃ k ∈ k_vals, n = 80 * k - 800 ∧ n > 0)) (finset.range (10000))) 1 = 2 :=
by
  sorry

end number_of_solutions_l300_300749


namespace tangent_line_y_intercept_correct_l300_300952

open Real

-- Define the centers and radii of the circles
def center_circle1 : Point := (3, 0)
def radius_circle1 : ℝ := 3

def center_circle2 : Point := (6, 0)
def radius_circle2 : ℝ := 1

-- The hypothesis that the circles and tangent line are as described in the conditions
def tangent_line_y_intercept (A C : Point) (rA rC : ℝ) : ℝ :=
  if hA : A = (3, 0) ∧ rA = 3 then
    if hC : C = (6, 0) ∧ rC = 1 then
      6 * sqrt 2
    else 0
  else 0

-- The theorem proving the y-intercept of the tangent line
theorem tangent_line_y_intercept_correct :
  tangent_line_y_intercept (3, 0) (6, 0) 3 1 = 6 * sqrt 2 :=
by
  sorry

end tangent_line_y_intercept_correct_l300_300952


namespace smallest_of_five_even_integers_l300_300520

theorem smallest_of_five_even_integers : 
  let even_sum := (2 * (25 * 26)) / 2 in
  let n := even_sum / 5 in
  n - 4 = 126 :=
by
  let even_sum := (2 * (25 * 26)) / 2
  let n := even_sum / 5
  have h1 : n = 130 := by
    norm_num
  show n - 4 = 126
  rw h1
  norm_num
  sorry

end smallest_of_five_even_integers_l300_300520


namespace sum_sequence_2009_terms_l300_300709

theorem sum_sequence_2009_terms (a : ℕ → ℝ) (h : ∀ n, a (n + 1) * a n = 2 * a (n + 1) - 2) :
  (∑ k in Finset.range 2009, a k) = 2008 + a 2009 :=
sorry

end sum_sequence_2009_terms_l300_300709


namespace apples_in_basket_B_l300_300121

-- Problem statement and conditions
variables (A B C : ℕ)
hypothesis h1 : C = 2 * A
hypothesis h2 : A + 12 = C - 24
hypothesis h3 : B = C + 6

-- Theorem proving the number of apples originally in basket B is 90
theorem apples_in_basket_B : B = 90 :=
by
  sorry

end apples_in_basket_B_l300_300121


namespace triangle_determines_plane_l300_300375

theorem triangle_determines_plane (A B C: Type) (x y z: A) :
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → (∃ P: C, some_plane_determined_by_points P x y z) :=
sorry

end triangle_determines_plane_l300_300375


namespace bernardo_wins_at_5_l300_300193

theorem bernardo_wins_at_5 :
  ∃ N : ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 27 * N + 360 < 500 ∧ ∀ M : ℕ, (0 ≤ M ∧ M ≤ 499 ∧ 27 * M + 360 < 500 → N ≤ M) :=
by
  sorry

end bernardo_wins_at_5_l300_300193


namespace coefficient_x4_in_expansion_l300_300904

theorem coefficient_x4_in_expansion : 
  (∑ k in finset.range 9, (nat.choose 8 k) * x ^ (8 - k) * (real.sqrt 5) ^ k).coeff 4 = 1750 := 
by {
  sorry -- the proof details are in the solution steps but are not required to be included here
}

end coefficient_x4_in_expansion_l300_300904


namespace proof_of_competition_results_l300_300224

def scores_team_a : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def scores_team_b : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]

def median (l : List ℝ) : ℝ :=
  let sorted := l.sort
  if l.length % 2 = 0 then
    (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2
  else
    sorted.get! (l.length / 2)

def mode (l : List ℝ) : ℝ :=
  l.foldl (λ acc x => if l.count x > acc.1 then (l.count x, x) else acc) (0, 0).2

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let avg := average l
  (l.map (λ x => (x - avg) ^ 2)).sum / l.length

theorem proof_of_competition_results : median scores_team_a = 9.5 ∧ mode scores_team_b = 10 ∧ average scores_team_b = 9 ∧ variance scores_team_b = 1 ∧ 1 < 1.4 :=
by
  sorry

end proof_of_competition_results_l300_300224


namespace final_digit_is_three_l300_300151

-- Define the initial number
def initial_number := List.replicate 100 8

-- Define the operations allowed
def operation1 (n : List ℕ) : List ℕ := sorry -- Details of operation1 can be implemented/defined here
def operation2 (n : List ℕ) : List ℕ := sorry -- Details of operation2 can be implemented/defined here
def operation3 (n : List ℕ) : List ℕ := sorry -- Details of operation3 can be implemented/defined here

-- Define the final digit remaining
def final_digit (n : List ℕ) : ℕ := sorry -- Simplification to final single-digit

-- Formulate the Lean theorem to prove the equivalence
theorem final_digit_is_three : 
  final_digit (iterate (operation1 ∘ operation2 ∘ operation3) initial_number.length initial_number) = 3 := 
sorry

end final_digit_is_three_l300_300151


namespace common_tangent_lines_count_l300_300699

-- Define the first circle
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 9 = 0

-- Definition for the number of common tangent lines between two circles
def number_of_common_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The theorem stating the number of common tangent lines between the given circles
theorem common_tangent_lines_count : number_of_common_tangent_lines C1 C2 = 2 := by
  sorry

end common_tangent_lines_count_l300_300699


namespace weight_labels_correct_l300_300298

-- Noncomputable because we're dealing with theoretical weight comparisons
noncomputable section

-- Defining the weights and their properties
variables {x1 x2 x3 x4 x5 x6 : ℕ}

-- Given conditions as stated
axiom h1 : x1 + x2 + x3 = 6
axiom h2 : x6 = 6
axiom h3 : x1 + x6 < x3 + x5

theorem weight_labels_correct :
  x1 = 1 ∧ x2 = 2 ∧ x3 = 3 ∧ x4 = 4 ∧ x5 = 5 ∧ x6 = 6 :=
sorry

end weight_labels_correct_l300_300298


namespace bobs_password_probability_l300_300220

theorem bobs_password_probability :
  (5 / 10) * (5 / 10) * 1 * (9 / 10) = 9 / 40 :=
by
  sorry

end bobs_password_probability_l300_300220


namespace two_numbers_with_difference_less_than_half_l300_300843

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l300_300843


namespace digit_1234th_is_four_l300_300035

def sequence_decimal (n : ℕ) : String := 
  String.intercalate "" (List.map toString (List.range (n + 1)))

def get_digit (s : String) (pos : ℕ) : Char := 
  s.get? (pos - 1) |>.getD '0'

theorem digit_1234th_is_four : 
  get_digit (sequence_decimal 500) 1234 = '4' := 
  by sorry

end digit_1234th_is_four_l300_300035


namespace fractional_sum_and_integer_parts_l300_300789

-- Definitions
def is_unique (s : List ℕ) : Prop := s.nodup

def sum_to (s : List ℕ) (n : ℕ) : Prop := s.sum = n

theorem fractional_sum_and_integer_parts {sum : ℕ} (f : List ℕ) (i : List ℕ) 
    (hf : is_unique f) (hi : is_unique i) 
    (sum_f : sum_to f sum) (sum_i : sum_to i (45 - sum)) 
    (carry : sum + (45 - sum) % 10 = 45):
    sum = 27 ∨ sum = 18 :=
by sorry

end fractional_sum_and_integer_parts_l300_300789


namespace correct_calculation_l300_300918

theorem correct_calculation : 
(∀ x : ℝ, √ 12 = 3 * √ 2 → false) ∧ 
(∀ x : ℝ, √ 3 + √ 2 = √ 5 → false) ∧ 
(∀ x : ℝ, (√ 3)^2 = 3) := 
by
  split
  sorry  -- proof for first part
  split
  sorry  -- proof for second part
  split 
  sorry  -- proof for correct statement

end correct_calculation_l300_300918


namespace function_is_zero_l300_300393

variable (n : ℕ) (a : Fin n → ℤ) (f : ℤ → ℝ)

axiom condition : ∀ (k l : ℤ), l ≠ 0 → (Finset.univ.sum (λ i => f (k + a i * l)) = 0)

theorem function_is_zero : ∀ x : ℤ, f x = 0 := by
  sorry

end function_is_zero_l300_300393


namespace find_central_angle_l300_300322

theorem find_central_angle (r θ : ℝ) (h_r : r = 2) (h_area : (1/2) * r^2 * θ = 8) : θ = 4 :=
by
  rw [h_r] at h_area
  norm_num at h_area
  exact h_area

end find_central_angle_l300_300322


namespace five_natural_numbers_increase_15_times_l300_300788

noncomputable def prod_of_decreased_factors_is_15_times_original (a1 a2 a3 a4 a5 : ℕ) : Prop :=
  (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * (a1 * a2 * a3 * a4 * a5)

theorem five_natural_numbers_increase_15_times {a1 a2 a3 a4 a5 : ℕ} :
  a1 * a2 * a3 * a4 * a5 = 48 → prod_of_decreased_factors_is_15_times_original a1 a2 a3 a4 a5 :=
by
  sorry

end five_natural_numbers_increase_15_times_l300_300788


namespace sequence_ge_two_power_l300_300032

theorem sequence_ge_two_power (a : ℕ → ℕ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ i, Nat.gcd (a (i + 1)) (a (i + 2)) > a i) :
  ∀ n, a n ≥ 2 ^ n :=
begin
  sorry
end

end sequence_ge_two_power_l300_300032


namespace perpendicular_bisector_fixed_point_l300_300737

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  - x^2 / 13 + y^2 / 12 = 1

-- Define the points A, B, C on the hyperbola
variables (x1 y1 x2 y2 : ℝ)
def A := (x1, y1)
def B := (sqrt 26, 6)
def C := (x2, y2)

-- Ensure A, B, C lie on the hyperbola
axiom A_on_hyperbola : hyperbola x1 y1
axiom B_on_hyperbola : hyperbola (sqrt 26) 6
axiom C_on_hyperbola : hyperbola x2 y2

-- Definition of foci distance in terms of the points making an arithmetic sequence
def dist (x1 y1 x2 y2 : ℝ) : ℝ := 
  abs (sqrt (x1^2 + (y1 - 5)^2)) -- Placeholder for actual distance calculation
-- Arithmetic sequence condition
axiom distance_arithmetic_sequence : 
  dist x1 y1 0 5 + dist x2 y2 0 5 = 2 * dist (sqrt 26) 6 0 5

-- The fixed point we need to prove the perpendicular bisector passes through
def fixed_point : ℝ × ℝ := (0, 25 / 2)

-- Theorem statement
theorem perpendicular_bisector_fixed_point : 
  ∀ (x1 y1 x2 y2 : ℝ), A_on_hyperbola → B_on_hyperbola → C_on_hyperbola → distance_arithmetic_sequence
  → (true) := -- Needs to be changed to actual proof (Basically placeholder)
sorry

end perpendicular_bisector_fixed_point_l300_300737


namespace number_of_valid_programs_l300_300975

-- Definitions based on the problem conditions
def courses := {'English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science}
def math_courses := {'Algebra, 'Geometry}
def remaining_courses := {'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science}

-- Condition that the program includes English and at least two math courses.
def valid_program (program : Finset Char) : Prop := 
  'English ∈ program ∧ (math_courses ∩ program).card ≥ 2 ∧ program.card = 5

-- The main proof statement
theorem number_of_valid_programs : 
  (finset.univ.filter valid_program).card = 6 :=
sorry

end number_of_valid_programs_l300_300975


namespace sam_drove_distance_l300_300436

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300436


namespace completing_square_l300_300914

theorem completing_square (x : ℝ) : (x^2 - 6 * x + 4 = 0) -> ((x - 3)^2 = 5) := 
by 
  intro h,
  sorry -- the actual proof steps would go here

end completing_square_l300_300914


namespace gcd_of_repeated_three_digit_integers_l300_300194

theorem gcd_of_repeated_three_digit_integers : ∀ (n : ℕ), n ∈ (Set.range (fun k => 100 ≤ k ∧ k < 1000)) →
  (∀ m : ℕ, (∃ k : ℕ, k = 1001001 * m) → gcd(n * 1001001, 1001001 * m) = 1001001) :=
by
  intros n hn m hm
  sorry

end gcd_of_repeated_three_digit_integers_l300_300194


namespace max_value_of_a_l300_300362

theorem max_value_of_a (a : ℝ) : (¬ ∃ x ∈ set.Icc (-2 : ℝ) 1, ax^2 + 2 * a * x + 3 * a > 1) → a ≤ 1 / 6 :=
by
  sorry

end max_value_of_a_l300_300362


namespace solution_correct_l300_300088

noncomputable def solve_system (A1 A2 A3 A4 A5 : ℝ) (x1 x2 x3 x4 x5 : ℝ) :=
  (2 * x1 - 2 * x2 = A1) ∧
  (-x1 + 4 * x2 - 3 * x3 = A2) ∧
  (-2 * x2 + 6 * x3 - 4 * x4 = A3) ∧
  (-3 * x3 + 8 * x4 - 5 * x5 = A4) ∧
  (-4 * x4 + 10 * x5 = A5)

theorem solution_correct {A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 : ℝ} :
  solve_system A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 → 
  x1 = (5 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x2 = (2 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x3 = (A1 + 2 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x4 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 2 * A5) / 12 ∧
  x5 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 5 * A5) / 30 :=
sorry

end solution_correct_l300_300088


namespace sigma_inequality_l300_300822

/-- Define the sum of divisors function -/
def sigma (n : ℕ) : ℕ := 
  ∑ m in (Finset.range (n + 1)), if (m ∣ n) then m else 0

/-- Define the number of distinct prime divisors function -/
def omega (n : ℕ) : ℕ := 
  (Nat.factors n).toFinset.card

/-- The theorem we need to prove -/
theorem sigma_inequality (n : ℕ) (hn : n > 0) : sigma n < n * (omega n + 1) := 
by 
  sorry

end sigma_inequality_l300_300822


namespace josephine_cannot_tile_l300_300798

def color (ℓ c : Nat) : Nat := (ℓ + c) % 4

def check_coloring (board_size : Nat) (piece_length : Nat) : Prop :=
  ∀ pieces : List (List (Nat × Nat)), 
    (∀ piece, piece ∈ pieces → piece.length = piece_length) → 
    (∀ piece1 piece2, piece1 ∈ pieces → piece2 ∈ pieces → piece1 ≠ piece2 → piece1 ∩ piece2 = ∅) → 
    ∃ counts : Fin 4 → Nat, 
      (∀ k : Fin 4, counts k = piece_length * board_size / 4) → 
      (board_color_count color 10 ≠ counts)

def checkerboard_problem := check_coloring 10 4

theorem josephine_cannot_tile : ¬ checkerboard_problem :=
sorry

end josephine_cannot_tile_l300_300798


namespace melissa_work_hours_l300_300065

theorem melissa_work_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (hours_per_dress : ℕ) (total_num_dresses : ℕ) (total_hours : ℕ) 
  (h1 : total_fabric = 56) (h2 : fabric_per_dress = 4) (h3 : hours_per_dress = 3) : 
  total_hours = (total_fabric / fabric_per_dress) * hours_per_dress := by
  sorry

end melissa_work_hours_l300_300065


namespace sam_distance_l300_300460

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300460


namespace factorize_expression_l300_300666

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300666


namespace probability_of_region_C_zero_l300_300978

theorem probability_of_region_C_zero :
  let p_A := 1/5
      p_B := 1/3
      x := (7/30 : ℚ)
      p_C := (0 : ℚ)
  in p_A + p_B + p_C + x + x = 1 :=
by
  let p_A := (1/5 : ℚ)
  let p_B := (1/3 : ℚ)
  let x := (7/30 : ℚ)
  let p_C := (0 : ℚ)
  have h : p_A + p_B + p_C + x + x = 1 :=
    by sorry
  exact h

end probability_of_region_C_zero_l300_300978


namespace starting_number_of_range_l300_300526

theorem starting_number_of_range (N : ℕ) : ∃ (start : ℕ), 
  (∀ n, n ≥ start ∧ n ≤ 200 → ∃ k, 8 * k = n) ∧ -- All numbers between start and 200 inclusive are multiples of 8
  (∃ k, k = (200 / 8) ∧ 25 - k = 13.5) ∧ -- There are 13.5 multiples of 8 in the range
  start = 84 := 
sorry

end starting_number_of_range_l300_300526


namespace bakery_regular_price_l300_300184

theorem bakery_regular_price (y : ℝ) (h₁ : y / 4 * 0.4 = 2) : y = 20 :=
by {
  sorry
}

end bakery_regular_price_l300_300184


namespace propositions_true_false_l300_300738

theorem propositions_true_false :
  (∃ x : ℝ, x ^ 3 < 1) ∧ 
  ¬ (∃ x : ℚ, x ^ 2 = 2) ∧ 
  ¬ (∀ x : ℕ, x ^ 3 > x ^ 2) ∧ 
  (∀ x : ℝ, x ^ 2 + 1 > 0) :=
by
  sorry

end propositions_true_false_l300_300738


namespace prob_at_least_3_speak_l300_300546

-- Define the probability of any baby speaking
def prob_speaking := 1 / 3

-- Define the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of at least 3 out of 6 babies speaking the next day
theorem prob_at_least_3_speak (P: ℝ) (n: ℕ) (k1: ℕ) (k2: ℕ):
  P = prob_speaking ∧ n = 6 ∧ k1 = 3 ∧ k2 = 2 ->
  (1 - ((2/3)^n + binom n 1 * (P)*(2/3)^(n-1) + binom n k2 * (P)^2 * (2/3)^(n-k2))) = 233 / 729 := sorry

end prob_at_least_3_speak_l300_300546


namespace ParallelogramSquareIfTrianglesSimilarAndMidpoints_l300_300803

variables {A B C D O M N : Type*}
  [AddGroup A] [AffineSpace A O]
  [AddGroup B] [AffineSpace B O]
  [AddGroup C] [AffineSpace C O]
  [AddGroup D] [AffineSpace D O]
  [AddGroup O] [AffineSpace O O]
  [AddGroup M] [AffineSpace M O]
  [AddGroup N] [AffineSpace N O]

variables (ABCD : Set {B : Bunct})
  (MidpointBOM : AffineCombination 2 [B O; B])
  (MidpointCDN : AffineCombination 2 [C D; N])
  (SimilarABC_AMN : Triangle ABC ∼ Triangle AMN)

theorem ParallelogramSquareIfTrianglesSimilarAndMidpoints :
  isParallelogram ABCD ∧
  midpoint B O = M ∧
  midpoint C D = N ∧
  triangle ABC ∼ triangle AMN →
  isSquare ABCD :=
sorry

end ParallelogramSquareIfTrianglesSimilarAndMidpoints_l300_300803


namespace max_value_and_period_sin_A_of_right_triangle_l300_300823

def f (x : ℝ) : ℝ := cos (2 * x + π / 3) + sin x ^ 2

theorem max_value_and_period :
  (∃ x : ℝ, f x = (1 + real.sqrt 3) / 2) ∧ (∀ x : ℝ, f (x + π) = f x) := sorry

noncomputable def A (B C : ℝ) : ℝ := π - B - C

theorem sin_A_of_right_triangle (A B C : ℝ)
  (hB : cos B = 1 / 3) (hC : C = π / 2) (hA : A = A B C) :
  sin A = 1 / 3 := sorry

end max_value_and_period_sin_A_of_right_triangle_l300_300823


namespace least_possible_value_of_smallest_integer_l300_300934

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 76 ∧ D = 90 →
  A = 37 :=
by
  sorry

end least_possible_value_of_smallest_integer_l300_300934


namespace tournament_games_count_l300_300577

-- Defining the problem conditions
def num_players : Nat := 12
def plays_twice : Bool := true

-- Theorem statement
theorem tournament_games_count (n : Nat) (plays_twice : Bool) (h : n = num_players ∧ plays_twice = true) :
  (n * (n - 1) * 2) = 264 := by
  sorry

end tournament_games_count_l300_300577


namespace find_x_l300_300720

theorem find_x (y z : ℚ) (h1 : z = 80) (h2 : y = z / 4) (h3 : x = y / 3) : x = 20 / 3 :=
by
  sorry

end find_x_l300_300720


namespace smallest_possible_value_of_other_number_l300_300871

theorem smallest_possible_value_of_other_number (x n : ℕ) (h_pos : x > 0) 
  (h_gcd : Nat.gcd 72 n = x + 6) (h_lcm : Nat.lcm 72 n = x * (x + 6)) : n = 12 := by
  sorry

end smallest_possible_value_of_other_number_l300_300871


namespace expression_value_l300_300755

theorem expression_value (x y : ℤ) (h1 : x = 2) (h2 : y = 5) : 
  (x^4 + 2 * y^2) / 6 = 11 := by
  sorry

end expression_value_l300_300755


namespace tan_abs_period_l300_300876

def period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x

theorem tan_abs_period (n : ℝ) : period (λ x, |tan (n * x)|) π :=
by
  intros x
  have h : tan (n * (x + π)) = tan (n * x + n * π) := by rw [mul_add]
  have h' : tan (n * x + n * π) = tan (n * x) := by sorry  -- Proof that tan(nx + nπ) = tan(nx) needed here
  show |tan (n * (x + π))| = |tan (n * x)| from by rw [h, h']

end tan_abs_period_l300_300876


namespace number_of_monsters_l300_300989

theorem number_of_monsters (U_heads U_legs : Nat) (M_initial_heads M_initial_legs M_new_heads M_new_legs : Nat) (total_heads total_legs : Nat) :
  U_heads = 1 → U_legs = 2 → M_initial_heads = 2 → M_initial_legs = 5 →
  M_new_heads = 1 → M_new_legs = 6 → total_heads = 21 → total_legs = 73 →
  (let remaining_heads := total_heads - U_heads in
   let remaining_legs := total_legs - U_legs in
   let n0 := remaining_heads / M_initial_heads in
   let initial_legs := n0 * M_initial_legs in
   let leg_deficit := remaining_legs - initial_legs in
   let s := leg_deficit / (2 * M_new_legs - M_initial_legs) in
   n0 + s = 13) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8;
  let remaining_heads := total_heads - U_heads;
  let remaining_legs := total_legs - U_legs;
  let n0 := remaining_heads / M_initial_heads;
  let initial_legs := n0 * M_initial_legs;
  let leg_deficit := remaining_legs - initial_legs;
  let s := leg_deficit / (2 * M_new_legs - M_initial_legs);
  sorry

end number_of_monsters_l300_300989


namespace number_of_integers_abs_x_lt_2_sqrt_10_l300_300342

theorem number_of_integers_abs_x_lt_2_sqrt_10 :
  (finset.card ((finset.Icc (-6 : ℤ) 6).filter (λ x, abs x < 2 * real.sqrt 10)))
  = 13 := by
  sorry

end number_of_integers_abs_x_lt_2_sqrt_10_l300_300342


namespace domain_of_f_l300_300099

-- Define the function
def f (x : ℝ) := (sqrt x) / (x - 3)

-- State the conditions
def numerator_condition (x : ℝ) := x ≥ 0
def denominator_condition (x : ℝ) := x ≠ 3

-- State the domain of the function
def domain (x : ℝ) := numerator_condition x ∧ denominator_condition x

-- State the theorem about the domain of the function
theorem domain_of_f : ∀ x : ℝ, domain x ↔ x ≥ 0 ∧ x ≠ 3 :=
by
  sorry

end domain_of_f_l300_300099


namespace ratio_four_l300_300962

variable {x y : ℝ}

theorem ratio_four : y = 0.25 * x → x / y = 4 := by
  sorry

end ratio_four_l300_300962


namespace circle_center_tangent_eq_l300_300588

open Real

theorem circle_center_tangent_eq (x y : ℝ):
  (3 * x - 4 * y = 40) ∧
  (3 * x - 4 * y = 0) ∧
  (x - 2 * y = 0) →
  (x = 20 ∧ y = 10) := 
by
  intro h
  sorry

end circle_center_tangent_eq_l300_300588


namespace robot_steps_difference_zero_l300_300613

/-- Define the robot's position at second n --/
def robot_position (n : ℕ) : ℤ :=
  let cycle_length := 7
  let cycle_steps := 4 - 3
  let full_cycles := n / cycle_length
  let remainder := n % cycle_length
  full_cycles + if remainder = 0 then 0 else
    if remainder ≤ 4 then remainder else 4 - (remainder - 4)

/-- The main theorem to prove x_2007 - x_2011 = 0 --/
theorem robot_steps_difference_zero : 
  robot_position 2007 - robot_position 2011 = 0 :=
by sorry

end robot_steps_difference_zero_l300_300613


namespace fraction_of_lollipops_given_to_emily_is_2_3_l300_300469

-- Given conditions as definitions
def initial_lollipops := 42
def kept_lollipops := 4
def lou_received := 10

-- The fraction of lollipops given to Emily
def fraction_given_to_emily : ℚ :=
  have emily_received : ℚ := initial_lollipops - (kept_lollipops + lou_received)
  have total_lollipops : ℚ := initial_lollipops
  emily_received / total_lollipops

-- The proof statement assert that fraction_given_to_emily is equal to 2/3
theorem fraction_of_lollipops_given_to_emily_is_2_3 : fraction_given_to_emily = 2 / 3 := by
  sorry

end fraction_of_lollipops_given_to_emily_is_2_3_l300_300469


namespace base_b_for_three_digits_l300_300159

theorem base_b_for_three_digits (b : ℕ) : b = 7 ↔ b^2 ≤ 256 ∧ 256 < b^3 := by
  sorry

end base_b_for_three_digits_l300_300159


namespace expression_value_l300_300937

noncomputable def expression : ℝ :=
  (π - 1)^0 - real.sqrt 9 + 2 * real.cos (real.pi / 4) + (1 / 5)⁻¹

theorem expression_value : expression = 3 + real.sqrt 2 := by
  sorry

end expression_value_l300_300937


namespace cos_arcsin_l300_300250

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l300_300250


namespace locus_of_focus_l300_300108

theorem locus_of_focus (x y : ℝ) (t : ℝ) :
  (∀ t, (∃ x y, y = x^2 ∧ x = t ∧ y = -t^2) → focus (x^2) (0, 1/4) = focus (-x^2) (0, -1/4)) →
  locus (focus (x^2) (0, 1/4)) = (y = 1/4) :=
by
  sorry

noncomputable def focus (parabola : ℝ → ℝ) (init_focus : ℝ × ℝ) : ℝ × ℝ :=
  (init_focus.1, 1/4)  -- The focus (for this problem's context) we declare is always on y = 1/4.

noncomputable def locus (focus_fn : (ℝ → ℝ) → ℝ × ℝ → ℝ × ℝ) : ℝ × ℝ -> Prop :=
  λ p, p.2 = 1/4  -- Returns true if the point has y-coordinate 1/4.

end locus_of_focus_l300_300108


namespace count_integers_satisfying_inequality_l300_300747

theorem count_integers_satisfying_inequality :
  (Finset.card ((Finset.filter (λ n : ℤ, (n - 3) * (n + 5) < 0) (Finset.Icc (-4) 2))) = 7) := 
begin
  sorry
end

end count_integers_satisfying_inequality_l300_300747


namespace cos_arcsin_l300_300249

theorem cos_arcsin (h : real.arcsin (3 / 5) = θ) : real.cos θ = 4 / 5 := 
by {
  have h1 : real.sin θ = 3 / 5 := by rwa [real.sin_arcsin],
  have hypo : (4 : real)^2 + (3 : real)^2 = (5 : real)^2 := by norm_num,
  have h2 : abs (real.cos θ) = 4 / 5,
  { rw [real.cos_eq_sqrt_one_sub_sin_sq, h1],
    simp only [sq, pow_two],
    rw [div_pow 3 5],
    norm_num, simp only [real.sqrt_sqr_eq_abs, sqr_pos],
  },
  rw abs_eq_self at h2,
  exact h2,
}

end cos_arcsin_l300_300249


namespace construct_triangle_given_h_a_b_minus_c_and_r_l300_300652

-- Definitions
variables (h_a b c r : ℝ)

-- Hypothesis
def triangle_constructible (h_a b c r : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    let a := (B - C) in
    let s := (a + b + c) / 2 in
    let BR := s - c in
    let BQ := (a + c - b) / 2 in
    let RQ := abs (b - c) in
    let PQ := 2 * r in
    ∃ (A' B' C' : ℝ), 
      PQ * PQ = h_a * h_a + (b - c) * (b - c) ∧  
      (h_a = r * BQ) ∧ 
      (all required conditions to ensure constructing triangle ABC)

-- Main theorem statement
theorem construct_triangle_given_h_a_b_minus_c_and_r (h_a b c r : ℝ) : 
  triangle_constructible h_a b c r :=
sorry

end construct_triangle_given_h_a_b_minus_c_and_r_l300_300652


namespace sam_distance_l300_300457

theorem sam_distance (m_distance m_time s_time : ℝ) (m_distance_eq : m_distance = 150) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  let rate := m_distance / m_time,
      s_distance := rate * s_time
  in s_distance = 200 :=
by
  let rate := m_distance / m_time
  let s_distance := rate * s_time
  sorry

end sam_distance_l300_300457


namespace Sarah_books_in_8_hours_l300_300487

theorem Sarah_books_in_8_hours (pages_per_hour: ℕ) (pages_per_book: ℕ) (hours_available: ℕ) 
  (h_pages_per_hour: pages_per_hour = 120) (h_pages_per_book: pages_per_book = 360) (h_hours_available: hours_available = 8) :
  hours_available * pages_per_hour / pages_per_book = 2 := by
  sorry

end Sarah_books_in_8_hours_l300_300487


namespace factorize_expression_l300_300667

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300667


namespace sally_balloon_count_l300_300796

theorem sally_balloon_count 
  (joan_balloons : Nat)
  (jessica_balloons : Nat)
  (total_balloons : Nat)
  (sally_balloons : Nat)
  (h_joan : joan_balloons = 9)
  (h_jessica : jessica_balloons = 2)
  (h_total : total_balloons = 16)
  (h_eq : total_balloons = joan_balloons + jessica_balloons + sally_balloons) : 
  sally_balloons = 5 :=
by
  sorry

end sally_balloon_count_l300_300796


namespace bound_phi_by_one_l300_300819

noncomputable def real_functions (f : ℝ → ℝ) (φ : ℝ → ℝ) : Prop :=
    ∀ x y, f(x + y) + f(x - y) = 2 * φ(y) * f(x)

theorem bound_phi_by_one (f φ : ℝ → ℝ) (h1 : real_functions f φ)
    (h2 : ∀ x, abs (f x) ≤ 1) 
    (h3 : ∃ x, f x ≠ 0) : ∀ x, abs (φ x) ≤ 1 :=
by
  -- proof will go here
  sorry

end bound_phi_by_one_l300_300819


namespace least_tiles_required_l300_300157

def floor_length : ℕ := 5000
def floor_breadth : ℕ := 1125
def gcd_floor : ℕ := Nat.gcd floor_length floor_breadth
def tile_area : ℕ := gcd_floor ^ 2
def floor_area : ℕ := floor_length * floor_breadth
def tiles_count : ℕ := floor_area / tile_area

theorem least_tiles_required : tiles_count = 360 :=
by
  sorry

end least_tiles_required_l300_300157


namespace sum_first_60_digits_of_1_div_9999_eq_15_l300_300551

theorem sum_first_60_digits_of_1_div_9999_eq_15 :
  let d := 1 / 9999 in
  let digits := (d.to_decimal 60).take 60 in
  digits.sum = 15 :=
by
  -- Lean code for expressing the decimal representation and summing the digits
  sorry

end sum_first_60_digits_of_1_div_9999_eq_15_l300_300551


namespace total_sand_volume_l300_300192

def cone_diameter : ℝ := 12
def cone_height : ℝ := 0.5 * cone_diameter
def cone_radius : ℝ := cone_diameter / 2
def cylinder_height : ℝ := 2
def cylinder_outer_radius : ℝ := cone_radius + 1
def cylinder_inner_radius : ℝ := cone_radius

theorem total_sand_volume : 
  ( (1 / 3 : ℝ) * π * cone_radius^2 * cone_height + π * cylinder_outer_radius^2 * cylinder_height - π * cylinder_inner_radius^2 * cylinder_height) = 98 * π := by
  sorry

end total_sand_volume_l300_300192


namespace problem_l300_300330

def f (x : ℝ) : ℝ := Real.sin ((1/2) * x + Real.pi / 6)

theorem problem (x : ℝ) :
  (∀ x : ℝ, f(x - Real.pi / 3) = Real.sin ((1/2) * x)) → 
  ∃ C, ∀ x, f (x + C) = f (-x - C) :=
by
  intro h
  -- We expect that function f after move by π/3 will be symmetric about origin
  have h_symmetric : ∀ x : ℝ, Real.sin ((1/2) * (x - Real.pi / 3)) = Real.sin ((1/2) * x), from 
    sorry,
  use Real.pi / 3
  intro x
  rw [h_symmetric x]
  exact h_symmetric (-x - Real.pi / 3)

end problem_l300_300330


namespace forty_percent_of_number_l300_300933

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) :
  0.40 * N = 180 :=
by
  sorry

end forty_percent_of_number_l300_300933


namespace power_of_q_in_product_l300_300357

theorem power_of_q_in_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (x : ℕ) 
    (h : Nat.divisors_count (p^4 * q^x) = 30) : x = 5 := by
  sorry

end power_of_q_in_product_l300_300357


namespace inequality_has_solutions_l300_300279

theorem inequality_has_solutions (a : ℝ) :
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3 * a) ↔ (a < -1 ∨ 4 < a) := 
by
  sorry

end inequality_has_solutions_l300_300279


namespace value_of_f_at_sin_pi_over_3_l300_300294

theorem value_of_f_at_sin_pi_over_3 :
  (∀ α : ℝ, f (Real.sin α + Real.cos α) = (1 / 2) * Real.sin (2 * α)) →
  f (Real.sin (Real.pi / 3)) = -1 / 8 :=
by
  intro h
  sorry

end value_of_f_at_sin_pi_over_3_l300_300294


namespace min_Tn_value_l300_300350

theorem min_Tn_value (a : ℕ → ℝ) (T : ℕ → ℝ) (r : ℝ)
  (h1 : 0 < r) 
  (h2 : ∀ n, a n = r^n)
  (h3 : ∀ n, T n = ∏ i in finset.range (n + 1), a i)
  (h4 : T 4 = T 8) :
  ∃ k, T k = T (k + 1) → k = 6 :=
by
  sorry

end min_Tn_value_l300_300350


namespace cos_pi_minus_2alpha_l300_300293

theorem cos_pi_minus_2alpha {α : ℝ} (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l300_300293


namespace modulus_of_z_find_a_and_b_l300_300703

noncomputable def z : ℂ := (1 - 7 * Complex.I) / (1 - Complex.I)

theorem modulus_of_z : Complex.abs z = 5 := 
by sorry

theorem find_a_and_b (a b : ℝ) (h : a * z - Complex.conj z - 2 * b = 4 + 6 * Complex.I) : 
  a = -3 ∧ b = -10 := 
by sorry

end modulus_of_z_find_a_and_b_l300_300703


namespace smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l300_300538

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_and_second_smallest_four_digit_numbers_divisible_by_35 :
  ∃ a b : ℕ, 
    is_four_digit a ∧ 
    is_four_digit b ∧ 
    divisible_by_35 a ∧ 
    divisible_by_35 b ∧ 
    a < b ∧ 
    ∀ c : ℕ, is_four_digit c → divisible_by_35 c → a ≤ c → (c = a ∨ c = b) :=
by
  sorry

end smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l300_300538


namespace nate_total_run_l300_300473

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l300_300473


namespace minimize_piers_when_m_640_l300_300172

def numPiers (m x : ℝ) : ℝ := m / x - 1

def y (m x : ℝ) : ℝ := 2.56 * m / x + m * Real.sqrt x + 2 * m - 2.56

theorem minimize_piers_when_m_640 :
  ∀ m : ℝ, m = 640 → numPiers m 64 = 9 :=
by
  intros m h
  rw [h]
  dsimp [numPiers]
  norm_num
  sorry

end minimize_piers_when_m_640_l300_300172


namespace find_K_l300_300743

theorem find_K (Z K : ℕ)
  (hZ1 : 700 < Z)
  (hZ2 : Z < 1500)
  (hK : K > 1)
  (hZ_eq : Z = K^4)
  (hZ_perfect : ∃ n : ℕ, Z = n^6) :
  K = 3 :=
by
  sorry

end find_K_l300_300743


namespace math_problem_percentile_l300_300483

def percent_elems_greater_than_m (s : Finset ℕ) (m : ℕ) : ℝ :=
(s.filter (λ x, x > m)).card / s.card

def m_position (s : Finset ℕ) (m : ℕ) : ℕ :=
(s.filter (λ x, x ≤ m)).card

def percentile (position total : ℕ) : ℝ :=
(position / total) * 100

theorem math_problem_percentile (s : Finset ℕ)
    (h : s = {2, 3, 4, 5, 6, 7, 8, 9})
    (hm : percent_elems_greater_than_m s 7 = 1 / 4) :
    (percentile (m_position s 7) s.card = 70) :=
by sorry

end math_problem_percentile_l300_300483


namespace number_of_white_balls_l300_300372

theorem number_of_white_balls (total_balls yellow_frequency : ℕ) (h1 : total_balls = 10) (h2 : yellow_frequency = 60) :
  (total_balls - (total_balls * yellow_frequency / 100) = 4) :=
by
  sorry

end number_of_white_balls_l300_300372


namespace area_of_rectangle_inscribed_in_semicircle_l300_300838

noncomputable def semicircle_diameter : ℝ := 44
noncomputable def DA : ℝ := 20
noncomputable def FD_AE : ℝ := 12

theorem area_of_rectangle_inscribed_in_semicircle :
  let CD := real.sqrt(384) in
  let area := DA * CD in
  area = 160 * real.sqrt(6) :=
by
  -- Provided conditions
  let r := semicircle_diameter / 2
  let DE := DA + FD_AE
  let CD := real.sqrt(FD_AE * DE)
  have area := DA * CD
  sorry

end area_of_rectangle_inscribed_in_semicircle_l300_300838


namespace three_digit_numbers_with_1_or_6_l300_300343

theorem three_digit_numbers_with_1_or_6 :
  let total_three_digit_numbers := 999 - 100 + 1 in
  let without_1_or_6 := 7 * 8 * 8 in
  total_three_digit_numbers - without_1_or_6 = 452 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let without_1_or_6 := 7 * 8 * 8
  show total_three_digit_numbers - without_1_or_6 = 452
  -- Proof not required
  sorry

end three_digit_numbers_with_1_or_6_l300_300343


namespace sam_drove_distance_l300_300437

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l300_300437


namespace order_of_a_b_c_l300_300047

noncomputable def a : ℝ := logBase (1 / 3) (2 / 3)
noncomputable def b : ℝ := logBase (1 / 2) (1 / 3)
noncomputable def c : ℝ := (1 / 2) ^ 0.3

theorem order_of_a_b_c : b > c ∧ c > a := 
by
  sorry

end order_of_a_b_c_l300_300047


namespace perpendicular_condition_nec_not_suff_l300_300706

variables (α : Type*) {m n c : α}
noncomputable def line_perpendicular (l1 l2 : α) : Prop := sorry
noncomputable def line_in_plane (l : α) (p : α) : Prop := sorry
noncomputable def plane_perpendicular (l : α) (p : α) : Prop := sorry

theorem perpendicular_condition_nec_not_suff {α : Type*} {m n c : α} (h1 : line_in_plane m α) (h2 : line_in_plane n α) :
  (line_perpendicular c m ∧ line_perpendicular c n) → plane_perpendicular c α ↔ false :=
sorry

end perpendicular_condition_nec_not_suff_l300_300706


namespace cos_arcsin_l300_300240

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l300_300240


namespace area_of_shaded_triangle_l300_300253

theorem area_of_shaded_triangle :
  ∀ (A B : Type) [inst : linear_ordered_field A],
    ∀ (smaller_square larger_square : set (euclidean_space A 2)),
      (smaller_square = {x : euclidean_space A 2 | 0 ≤ x.val ∧ x.val ≤ 4} ∧
       larger_square = {x : euclidean_space A 2 | 0 ≤ x.val ∧ x.val ≤ 12}) →
      (∀ A : euclidean_space A 2, A ∈ larger_square → 
         ∀ C : euclidean_space A 2, C ∈ smaller_square → 
         ∃ D : euclidean_space A 2,
           A = ⟨0, 12⟩ ∧ C = ⟨4, 0⟩ → 
           D = ⟨?, ?⟩) →
      (∀ B D : euclidean_space A 2,
        B = ⟨?, ?⟩ ∧ D = ⟨?, ?⟩ →
        ∀ base height : A,
          base = 4 ∧ height = 4 →
          ∀ area_shaded_triangle : A,
            area_shaded_triangle = 1 / 2 * base * height →
            ∀ area_of_shaded_triangle : A,
              area_of_shaded_triangle = 16 / 3) :=
sorry

end area_of_shaded_triangle_l300_300253


namespace third_side_of_triangle_l300_300023

theorem third_side_of_triangle (h_1 h_2 : ℝ) (h1_pos : h_1 > 0) (h2_pos : h_2 > 0) 
  (ineq : 5 + h_1 ≤ 2 * real.sqrt 6 + h_2) (area_eq : 5 * h_1 = 2 * real.sqrt 6 * h_2) : 
  ∃ c, c = 7 :=
by
  use 7
  sorry

end third_side_of_triangle_l300_300023


namespace mary_maximum_earnings_l300_300932

theorem mary_maximum_earnings :
  ∀ (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℝ) (overtime_rate_multiplier : ℝ),
  max_hours = 45 →
  regular_hours = 20 →
  regular_rate = 8 →
  overtime_rate_multiplier = 0.25 →
  let overtime_hours := max_hours - regular_hours,
      regular_earnings := regular_rate * regular_hours,
      overtime_rate := regular_rate * (1 + overtime_rate_multiplier),
      overtime_earnings := overtime_rate * overtime_hours,
      total_earnings := regular_earnings + overtime_earnings
  in
  total_earnings = 410 :=
begin
  intros max_hours regular_hours regular_rate overtime_rate_multiplier h1 h2 h3 h4,
  let overtime_hours := max_hours - regular_hours,
  let regular_earnings := regular_rate * regular_hours,
  let overtime_rate := regular_rate * (1 + overtime_rate_multiplier),
  let overtime_earnings := overtime_rate * overtime_hours,
  let total_earnings := regular_earnings + overtime_earnings,
  rw [h1, h2, h3, h4],
  rw [show max_hours - regular_hours = 25, by norm_num],
  rw [show regular_rate * regular_hours = 160, by norm_num],
  rw [show regular_rate * (1 + overtime_rate_multiplier) = 10, by norm_num],
  rw [show 10 * 25 = 250, by norm_num],
  rw [show 160 + 250 = 410, by norm_num],
  refl
end

end mary_maximum_earnings_l300_300932


namespace celia_receives_correct_amount_of_aranha_l300_300260

def borboleta_to_tubarao (b : Int) : Int := 3 * b
def tubarao_to_periquito (t : Int) : Int := 2 * t
def periquito_to_aranha (p : Int) : Int := 3 * p
def macaco_to_aranha (m : Int) : Int := 4 * m
def cobra_to_periquito (c : Int) : Int := 3 * c

def celia_stickers_to_aranha (borboleta tubarao cobra periquito macaco : Int) : Int :=
  let borboleta_to_aranha := periquito_to_aranha (tubarao_to_periquito (borboleta_to_tubarao borboleta))
  let tubarao_to_aranha := periquito_to_aranha (tubarao_to_periquito tubarao)
  let cobra_to_aranha := periquito_to_aranha (cobra_to_periquito cobra)
  let periquito_to_aranha := periquito_to_aranha periquito
  let macaco_to_aranha := macaco_to_aranha macaco
  borboleta_to_aranha + tubarao_to_aranha + cobra_to_aranha + periquito_to_aranha + macaco_to_aranha

theorem celia_receives_correct_amount_of_aranha : 
  celia_stickers_to_aranha 4 5 3 6 6 = 171 := 
by
  simp only [celia_stickers_to_aranha, borboleta_to_tubarao, tubarao_to_periquito, periquito_to_aranha, cobra_to_periquito, macaco_to_aranha]
  -- Here we need to perform the arithmetic steps to verify the sum
  sorry -- This is the placeholder for the actual proof

end celia_receives_correct_amount_of_aranha_l300_300260


namespace second_day_sales_l300_300090

def first_day_sales_eq (S : ℝ) := 4 * S + 3 * 9 = 79
def senior_citizen_ticket_price := ∃ S : ℝ, first_day_sales_eq S ∧ S = 13

theorem second_day_sales : senior_citizen_ticket_price → (12 * 13 + 10 * 9 = 246) :=
by
  assume h,
  show 12 * 13 + 10 * 9 = 246, from sorry

end second_day_sales_l300_300090


namespace greatest_b_value_l300_300275

theorem greatest_b_value (b : ℝ) : 
  (-b^3 + b^2 + 7 * b - 10 ≥ 0) ↔ b ≤ 4 + Real.sqrt 6 :=
sorry

end greatest_b_value_l300_300275


namespace OC_is_5_inches_l300_300400

noncomputable 
def isosceles_triangle (A B C : Point) : Prop := 
  dist A B = dist A C

noncomputable 
def midpoint (Q A B : Point) : Prop := 
  dist Q A = dist Q B

noncomputable 
def centroid (O A B C : Point) : Prop :=
  let M := midpoint (some (between A B)) in
  let Q := midpoint (some (between B C)) in
  dist O (some M) / dist (some M) A = 1 / 3 ∧ dist O (some Q) / dist (some Q) B = 1 / 3 ∧
  dist (some O) A / (dist (some O) B / 3) = 2

variables {A B C O Q : Point}

theorem OC_is_5_inches : 
  isosceles_triangle A B C → 
  centroid O A B C →
  midpoint Q A B → 
  dist O Q = 5 → 
  dist O C = 5 := 
by 
  -- skipped proof
  sorry

end OC_is_5_inches_l300_300400


namespace printer_time_calculation_l300_300607

theorem printer_time_calculation (pages_per_minute : ℕ) (total_pages : ℕ) (time : ℕ) :
  pages_per_minute = 25 → total_pages = 340 → time = 14 :=
by
  intros hp_htotal
  have hp : pages_per_minute = 25 := hp_htotal.1
  have htotal : total_pages = 340 := hp_htotal.2
  sorry

end printer_time_calculation_l300_300607


namespace erin_serving_time_correct_l300_300665

noncomputable def time_to_serve_soups (g1 g2 g3 : ℕ) (r1 r2 r3 : ℕ) (b1 b2 b3 : ℕ) (oz_per_gal : ℕ): ℕ :=
let o1 := g1 * oz_per_gal,
    o2 := g2 * oz_per_gal,
    o3 := g3 * oz_per_gal,
    bowls1 := o1 / b1,
    bowls2 := o2 / b2,
    bowls3 := o3 / b3,
    time1 := bowls1 / r1,
    time2 := bowls2 / r2,
    time3 := bowls3 / r3 in
time1 + time2 + time3

theorem erin_serving_time_correct :
  time_to_serve_soups 8 5.5 3.25 5 4 6 10 12 8 128 = 44 :=
sorry

end erin_serving_time_correct_l300_300665


namespace largest_integer_less_than_100_with_remainder_5_when_divided_by_6_l300_300681

theorem largest_integer_less_than_100_with_remainder_5_when_divided_by_6 :
  ∃ n : ℕ, n < 100 ∧ n % 6 = 5 ∧ ∀ m : ℕ, m < 100 → m % 6 = 5 → m ≤ n :=
begin
  use 99,
  split,
  { exact lt_trans (by norm_num) (by norm_num) },
  split,
  { exact mod_eq_of_lt (by norm_num) (by norm_num) },
  { intros m hmlt hmod,
    have : 5 < 6 := by norm_num,
    have hmn : m ≤ 99,
    { rw [← nat.add_one_le_iff, ← plus_eq_add, ← nat.div_eq_mul_add_mod m 6, 
        nat.add_le_add_iff_le_right this, nat.div_le_iff_le_mul_add_pred _ _ this],
      exact le_trans (le_of_lt hmlt) (nat.le_of_eq rfl) },
    exact hmn }
end

end largest_integer_less_than_100_with_remainder_5_when_divided_by_6_l300_300681


namespace min_value_log_condition_l300_300348

theorem min_value_log_condition (x y : ℝ) (h : log 2 x + log 2 y = 3) : 2 * x + y ≥ 8 := 
sorry

end min_value_log_condition_l300_300348


namespace Lesha_can_leave_2_columns_l300_300478

theorem Lesha_can_leave_2_columns (columns rows : ℕ) (colors : ℕ) :
  columns = 25 →
  rows = 300 →
  colors = 3 →
  ∃ k, k = 2 ∧ Lesha_can_guarantee_k_columns columns rows colors k :=
by
  sorry

end Lesha_can_leave_2_columns_l300_300478


namespace count_x_satisfying_equation_l300_300824

-- Define the set S
inductive S : Type
| A0 : S
| A1 : S
| A2 : S
| A3 : S

open S

-- Define the operation ⊕ on the set S
def op : S → S → S
| A0, A0 => A0
| A0, A1 => A1
| A0, A2 => A2
| A0, A3 => A3
| A1, A0 => A1
| A1, A1 => A2
| A1, A2 => A3
| A1, A3 => A0
| A2, A0 => A2
| A2, A1 => A3
| A2, A2 => A0
| A2, A3 => A1
| A3, A0 => A3
| A3, A1 => A0
| A3, A2 => A1
| A3, A3 => A2

-- Define the main statement
theorem count_x_satisfying_equation : 
  {x : S | op (op x x) A2 = A0}.to_finset.card = 2 := 
sorry

end count_x_satisfying_equation_l300_300824


namespace problem_statements_l300_300728

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

def symmetric_about_x (x₀ : ℝ) := ∀ x, f (2 * x₀ - x) = f x
def symmetric_about_point (x₀ y₀ : ℝ) := ∀ x, f (2 * x₀ - x) = 2 * y₀ - f x
def monotonic_in_interval (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def can_be_shifted (g : ℝ → ℝ) := ∀ x, f (x - Real.pi / 3) = g x

theorem problem_statements :
  (symmetric_about_x (11 * Real.pi / 12)) ∧
  (symmetric_about_point (2 * Real.pi / 3) 0) ∧
  ¬(monotonic_in_interval (-Real.pi / 12) (5 * Real.pi / 12)) ∧
  ¬(can_be_shifted (λ x, 3 * Real.sin (2 * x))) := sorry

end problem_statements_l300_300728


namespace three_equal_of_four_l300_300290

theorem three_equal_of_four (a b c d : ℕ) 
  (h1 : (a + b)^2 ∣ c * d) 
  (h2 : (a + c)^2 ∣ b * d) 
  (h3 : (a + d)^2 ∣ b * c) 
  (h4 : (b + c)^2 ∣ a * d) 
  (h5 : (b + d)^2 ∣ a * c) 
  (h6 : (c + d)^2 ∣ a * b) : 
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := 
sorry

end three_equal_of_four_l300_300290


namespace intercepts_sum_l300_300603

theorem intercepts_sum (x y : ℝ) : (y - 6 = -3 * (x - 5)) →
  ((x = 7) → (y = 21) → (7 + 21 = 28)) :=
by {
  intro h₁ h₂ h₃,
  rw [h₂, h₃],
  norm_num
}

end intercepts_sum_l300_300603


namespace total_estate_value_l300_300066

theorem total_estate_value 
  (estate : ℝ)
  (daughter_share son_share wife_share brother_share nanny_share : ℝ)
  (h1 : daughter_share + son_share = (3/5) * estate)
  (h2 : daughter_share = 5 * son_share / 2)
  (h3 : wife_share = 3 * son_share)
  (h4 : brother_share = daughter_share)
  (h5 : nanny_share = 400) :
  estate = 825 := by
  sorry

end total_estate_value_l300_300066


namespace cows_count_l300_300567

theorem cows_count (D C : ℕ) (h1 : 2 * (D + C) + 32 = 2 * D + 4 * C) : C = 16 :=
by
  sorry

end cows_count_l300_300567


namespace tony_saturday_sandwiches_l300_300131

-- Define the conditions
constant slices_per_day : ℕ := 2
constant days_Mon_to_Fri : ℕ := 5
constant total_slices : ℕ := 22
constant slices_left : ℕ := 6

-- Define the statement to prove
theorem tony_saturday_sandwiches : 
  total_slices - (slices_per_day * days_Mon_to_Fri + slices_left) / slices_per_day = 3 :=
by
  sorry

end tony_saturday_sandwiches_l300_300131


namespace area_of_region_bounded_by_tan_cot_l300_300678

open Real Set

theorem area_of_region_bounded_by_tan_cot :
  let region := {p : ℝ × ℝ | ∃ θ, 0 < θ ∧ θ < π/2 ∧ p.1 = tan θ ∧ p.2 = cot θ} in
  ∀ p ∈ region, 0 ≤ p.1 ∧ 0 ≤ p.2 →
  let triangle := {p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 0} ∪ {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0} ∪ {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 1} in
  area_of_triangle region = 1 / 2 := by
  sorry

end area_of_region_bounded_by_tan_cot_l300_300678


namespace sheena_sewing_weeks_l300_300085

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l300_300085


namespace sam_drove_200_miles_l300_300428

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300428


namespace shaded_region_area_l300_300783

-- Definitions based on conditions a)
variables (A B C D P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]

axiom AB_is_10 : dist A B = 10
axiom ABP_area_is_40 : triangle_area A B P = 40
axiom AB_is_parallel_to_CD : parallel A B C D
axiom BC_is_parallel_to_AD : parallel B C A D

theorem shaded_region_area :
  let rectangle_area := dist A B * dist B C in
  let triangle_area := 40 in
  rectangle_area - triangle_area = 40 := by
  sorry

end shaded_region_area_l300_300783


namespace sam_drove_200_miles_l300_300427

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300427


namespace runners_meet_at_same_point_l300_300892

noncomputable def lcm (a b : ℕ) : ℕ := if a * b = 0 then 0 else a * b / (Nat.gcd a b)

theorem runners_meet_at_same_point :
  let v1 := 4.5
  let v2 := 5.0
  let v3 := 5.5
  let track_length := 800
  let t := 28480
  (∃ t, (v1 * t % track_length = 0) ∧ (v2 * t % track_length = 0) ∧ (v3 * t % track_length = 0)) :=
begin
  sorry
end

end runners_meet_at_same_point_l300_300892


namespace cos_arcsin_l300_300238

theorem cos_arcsin (x : ℝ) (h : x = 3/5) : Real.cos (Real.arcsin x) = 4/5 := 
by
  rw h
  sorry

end cos_arcsin_l300_300238


namespace y_real_for_all_x_l300_300657

theorem y_real_for_all_x (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 :=
by
  sorry

end y_real_for_all_x_l300_300657


namespace equivalent_statements_l300_300146

variable (P Q : Prop)

theorem equivalent_statements (h : P → Q) :
  (¬Q → ¬P) ∧ (¬P ∨ Q) :=
by 
  sorry

end equivalent_statements_l300_300146


namespace odd_and_periodic_function_l300_300814

noncomputable def f : ℝ → ℝ := sorry

lemma given_conditions (x : ℝ) : 
  (f (10 + x) = f (10 - x)) ∧ (f (20 - x) = -f (20 + x)) :=
  sorry

theorem odd_and_periodic_function (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
  sorry

end odd_and_periodic_function_l300_300814


namespace graph_symmetry_l300_300870

variable (f : ℝ → ℝ)

theorem graph_symmetry :
  (∀ x y, y = f (x - 1) ↔ ∃ x', x' = 2 - x ∧ y = f (1 - x'))
  ∧ (∀ x' y', y' = f (1 - x') ↔ ∃ x, x = 2 - x' ∧ y' = f (x - 1)) :=
sorry

end graph_symmetry_l300_300870


namespace largest_four_digit_round_l300_300485

theorem largest_four_digit_round (h1: ∀ (a b c d : ℕ), (a ∈ {7, 9, 3, 6}) ∧ (b ≠ a) ∧ (b ∈ {7, 9, 3, 6}) ∧ (c ≠ b) ∧ (c ≠ a) ∧ (c ∈ {7, 9, 3, 6}) ∧ (d ≠ c) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (d ∈ {7, 9, 3, 6}) → 
  (1000*a + 100*b + 10*c + d ≤ 9763) ∧ (∃ (p q r s : ℕ), (p = 9) ∧ (q = 7) ∧ (r = 6) ∧ (s = 3) ∧ (1000*p + 100*q + 10*r + s = 9763))) :
  (∀ n, n ∈ {7, 9, 3, 6}) → 10000 > 9763 ∧ 9763 >= 9700 :=
by
  sorry

end largest_four_digit_round_l300_300485


namespace find_FC_l300_300698

theorem find_FC 
(DC CB AD ED FC : ℝ)
(h1 : DC = 7) 
(h2 : CB = 8) 
(h3 : AB = (1 / 4) * AD)
(h4 : ED = (4 / 5) * AD) : 
FC = 10.4 :=
sorry

end find_FC_l300_300698


namespace problem_statement_l300_300726

theorem problem_statement 
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 :=
sorry

end problem_statement_l300_300726


namespace weighted_avg_marks_correct_l300_300370

def student_data : Type := 
  {students: ℕ, avg_marks: ℝ, class_factor: ℝ}

def class_A := {students := 30, avg_marks := 50, class_factor := 1.2} : student_data
def class_B := {students := 50, avg_marks := 60, class_factor := 1.0} : student_data
def class_C := {students := 40, avg_marks := 55, class_factor := 0.8} : student_data
def class_D := {students := 20, avg_marks := 70, class_factor := 1.5} : student_data

def total_students := class_A.students + class_B.students + class_C.students + class_D.students

def total_weighted_marks := 
  (class_A.students * class_A.avg_marks * class_A.class_factor) + 
  (class_B.students * class_B.avg_marks * class_B.class_factor) + 
  (class_C.students * class_C.avg_marks * class_C.class_factor) + 
  (class_D.students * class_D.avg_marks * class_D.class_factor)

def weighted_avg_marks := total_weighted_marks / total_students

theorem weighted_avg_marks_correct : weighted_avg_marks ≈ 61.857 :=
  by
  sorry

end weighted_avg_marks_correct_l300_300370


namespace conjugate_point_in_complex_plane_l300_300717

theorem conjugate_point_in_complex_plane (z : ℂ) (conj_z : ℂ) 
  (h : conj_z = conj z ∧ z = (1 - 2 * complex.I) / (2 + complex.I) + 2) : 
  (complex.re conj_z, complex.im conj_z) = (2, 1) :=
sorry

end conjugate_point_in_complex_plane_l300_300717


namespace convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l300_300165

theorem convert_sq_meters_to_hectares :
  (123000 / 10000) = 12.3 :=
by
  sorry

theorem convert_hours_to_hours_and_minutes :
  (4 + 0.25 * 60) = 4 * 60 + 15 :=
by
  sorry

end convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l300_300165


namespace number_of_spiders_l300_300999

theorem number_of_spiders (initial_bugs : ℕ) (percentage_reduction : ℝ) 
  (spider_eats : ℕ) (bugs_left : ℕ) (after_spray_bugs : ℕ) (S : ℕ) :
  initial_bugs = 400 →
  percentage_reduction = 0.80 →
  spider_eats = 7 →
  after_spray_bugs = (.80 * initial_bugs) →
  bugs_left = 236 →
  after_spray_bugs - S * spider_eats = bugs_left →
  S = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_spiders_l300_300999


namespace combined_work_rate_time_l300_300928

theorem combined_work_rate_time (h1 : ℝ) (h2 : ℝ) (h1_eq : h1 = 5) (h2_eq : h2 = 8) : (1 / ((1 / h1) + (1 / h2))) = 40 / 13 := 
by 
  -- Given that the first worker's time to load one truck is 5 hours 
  have h1_rate : 1 / h1 = 1 / 5, from sorry,

  -- Given that the second worker's time to load one truck is 8 hours 
  have h2_rate : 1 / h2 = 1 / 8, from sorry,

  -- Combined work rate 
  have combined_rate : (1 / h1) + (1 / h2) = 1 / 5 + 1 / 8, from sorry,

  -- Calculate the combined rate with a common denominator 
  have combined_rate_eq : 1 / 5 + 1 / 8 = 13 / 40, from sorry,

  -- Calculate the time to fill one truck by taking the reciprocal of the combined rate 
  have time_fill_truck : (1 / (1 / 5 + 1 / 8)) = 40 / 13, from sorry,

  exact time_fill_truck

end combined_work_rate_time_l300_300928


namespace max_true_statements_l300_300406

theorem max_true_statements (x : ℝ) :
  let stmt1 := (0 < x^3 ∧ x^3 < 1)
  let stmt2 := (x^3 > 1)
  let stmt3 := (-1 < x ∧ x < 0)
  let stmt4 := (1 < x ∧ x < 2)
  let stmt5 := (0 < 3*x - x^3 ∧ 3*x - x^3 < 2)
  max_true_statements stmt1 stmt2 stmt3 stmt4 stmt5 = 3 := sorry

end max_true_statements_l300_300406


namespace sam_drove_200_miles_l300_300461

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l300_300461


namespace main_theorem_l300_300875

def d_digits (d : ℕ) : Prop :=
  ∃ (d_1 d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 : ℕ),
    d = d_1 * 10^8 + d_2 * 10^7 + d_3 * 10^6 + d_4 * 10^5 + d_5 * 10^4 + d_6 * 10^3 + d_7 * 10^2 + d_8 * 10 + d_9

noncomputable def condition1 (d e : ℕ) (i : ℕ) : Prop :=
  (e - (d / 10^(8 - i) % 10)) * 10^(8 - i) + d ≡ 0 [MOD 7]

noncomputable def condition2 (e f : ℕ) (i : ℕ) : Prop :=
  (f - (e / 10^(8 - i) % 10)) * 10^(8 - i) + e ≡ 0 [MOD 7]

theorem main_theorem
  (d e f : ℕ)
  (h1 : d_digits d)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition1 d e i)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition2 e f i) :
  ∀ i, 1 ≤ i ∧ i ≤ 9 → (d / 10^(8 - i) % 10) ≡ (f / 10^(8 - i) % 10) [MOD 7] := sorry

end main_theorem_l300_300875


namespace cos_arcsin_l300_300229

theorem cos_arcsin (h : real.sin θ = 3 / 5) : real.cos θ = 4 / 5 :=
sorry

end cos_arcsin_l300_300229


namespace steve_speed_back_home_l300_300098

-- Definitions based on conditions
def distance := 20 -- distance from house to work in km
def total_time := 6 -- total time on the road in hours
def speed_to_work (v : ℝ) := v -- speed to work in km/h
def speed_back_home (v : ℝ) := 2 * v -- speed back home in km/h

-- Theorem to assert the proof
theorem steve_speed_back_home (v : ℝ) (h : distance / v + distance / (2 * v) = total_time) :
  speed_back_home v = 10 := by
  -- Proof goes here but we just state sorry to skip it
  sorry

end steve_speed_back_home_l300_300098


namespace lucy_total_packs_l300_300058

-- Define the number of packs of cookies Lucy bought
def packs_of_cookies : ℕ := 12

-- Define the number of packs of noodles Lucy bought
def packs_of_noodles : ℕ := 16

-- Define the total number of packs of groceries Lucy bought
def total_packs_of_groceries : ℕ := packs_of_cookies + packs_of_noodles

-- Proof statement: The total number of packs of groceries Lucy bought is 28
theorem lucy_total_packs : total_packs_of_groceries = 28 := by
  sorry

end lucy_total_packs_l300_300058


namespace original_sales_tax_percentage_l300_300878

theorem original_sales_tax_percentage
  (current_sales_tax : ℝ := 10 / 3) -- 3 1/3% in decimal
  (difference : ℝ := 10.999999999999991) -- Rs. 10.999999999999991
  (market_price : ℝ := 6600) -- Rs. 6600
  (original_sales_tax : ℝ := 3.5) -- Expected original tax
  :  ((original_sales_tax / 100) * market_price = (current_sales_tax / 100) * market_price + difference) 
  := sorry

end original_sales_tax_percentage_l300_300878


namespace bus_distance_after_300_min_l300_300583

noncomputable def distance_covered (t : ℕ) : ℝ :=
  100 * (40 + t).toReal / (100 + t).toReal

theorem bus_distance_after_300_min :
  distance_covered 300 - distance_covered 0 = 45 :=
by
  sorry

end bus_distance_after_300_min_l300_300583


namespace smallest_m_l300_300410

noncomputable def b : ℝ := Real.pi / 2010

def series (m : ℕ) : ℝ :=
  2 * (∑ k in Finset.range (m+1), Real.cos (k^2 * b) * Real.sin (k * b))

theorem smallest_m (m : ℕ) : series m ∈ ℤ ↔ m = 67 := sorry

end smallest_m_l300_300410


namespace population_meets_capacity_l300_300105

open Nat

def acres := 32000
def acres_per_person := 2
def initial_population := 500
def doubling_interval := 30
def maximum_capacity := acres / acres_per_person
def year_initial := 2020
def year_capacity_reached := 2170

theorem population_meets_capacity :
  ∃ n : Nat, year_initial + n * doubling_interval = year_capacity_reached ∧ initial_population * 2^n ≥ maximum_capacity :=
by
  have h1 : maximum_capacity = 16000 := sorry
  have h2 : 500 * 2^5 = 16000 := sorry
  existsi 5
  split
  case inl =>
    calc
      year_initial  + 5 * doubling_interval = 2020 + 5 * 30 := by rfl
                                                  ... = 2170 := by rfl
  case inr =>
    calc
      initial_population * 2^5 = 500 * 32 := by rfl
                       ... = 16000 := sorry

end population_meets_capacity_l300_300105


namespace period_of_f_l300_300696

def has_period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f x = f (x + T)

def f (n : ℤ) (x : ℝ) : ℝ := cos ((n - 1) * x) * cos (15 * x / (2 * n + 1))

theorem period_of_f (n : ℤ) : has_period (f n) (π) ↔ n ∈ {0, -2, 2, -8} :=
sorry

end period_of_f_l300_300696


namespace elective_course_selection_l300_300614

theorem elective_course_selection (TypeA_courses TypeB_courses : ℕ) (total_courses : ℕ)
  (TypeA_courses = 3) (TypeB_courses = 4) (total_courses = 3) :
  (finset.card (finset.powersetLen 1 (finset.range TypeA_courses)) * finset.card (finset.powersetLen 2 (finset.range TypeB_courses)) +
   finset.card (finset.powersetLen 2 (finset.range TypeA_courses)) * finset.card (finset.powersetLen 1 (finset.range TypeB_courses)) = 30) :=
by
  sorry

end elective_course_selection_l300_300614


namespace find_length_of_train_l300_300152

noncomputable def speed_kmhr : ℝ := 30
noncomputable def time_seconds : ℝ := 9
noncomputable def conversion_factor : ℝ := 5 / 18
noncomputable def speed_ms : ℝ := speed_kmhr * conversion_factor
noncomputable def length_train : ℝ := speed_ms * time_seconds

theorem find_length_of_train : length_train = 74.97 := 
by
  sorry

end find_length_of_train_l300_300152


namespace person_B_more_stable_l300_300072

-- Define the variances of Person A and Person B
def variance_A : ℝ := 1.4
def variance_B : ℝ := 0.6

-- State the theorem that person B has more stable shooting performance than person A
theorem person_B_more_stable : variance_A > variance_B → more_stable_performance "B" :=
by
  intro h
  have h_variance : variance_A > variance_B := h
  sorry

end person_B_more_stable_l300_300072


namespace number_is_minus_72_l300_300944

noncomputable def find_number (x : ℝ) : Prop :=
  0.833 * x = -60

theorem number_is_minus_72 : ∃ x : ℝ, find_number x ∧ x = -72 :=
by
  sorry

end number_is_minus_72_l300_300944


namespace dana_pencils_more_than_jayden_l300_300653

theorem dana_pencils_more_than_jayden :
  ∀ (Jayden_has_pencils : ℕ) (Marcus_has_pencils : ℕ) (Dana_has_pencils : ℕ),
    Jayden_has_pencils = 20 →
    Marcus_has_pencils = Jayden_has_pencils / 2 →
    Dana_has_pencils = Marcus_has_pencils + 25 →
    Dana_has_pencils - Jayden_has_pencils = 15 :=
by
  intros Jayden_has_pencils Marcus_has_pencils Dana_has_pencils
  intro h1
  intro h2
  intro h3
  sorry

end dana_pencils_more_than_jayden_l300_300653


namespace add_one_five_times_l300_300071

theorem add_one_five_times (m n : ℕ) (h : n = m + 5) : n - (m + 1) = 4 :=
by
  sorry

end add_one_five_times_l300_300071


namespace probability_divisible_by_5_l300_300662

open ProbabilityTheory

noncomputable def spinner_outcomes : Set ℕ := {1, 3, 5}
noncomputable def spins : ℕ := 4
noncomputable def digits (outcomes : List ℕ) : ℕ :=
  outcomes.drop 1 |> List.take 3 |> Nat.digits 10 |> List.reverse |> List.foldr (fun x acc => x + 10 * acc) 0

theorem probability_divisible_by_5 :
  (∃ (outcomes : List ℕ), List.length outcomes = spins ∧ (spinner_outcomes ⊆ outcomes.toFinset) ∧
  (∃ (digit_seq : List ℕ), digit_seq = outcomes.drop 1 ∧
  digit_seq.length = 3 ∧
  (digits digit_seq) % 5 = 0)) →
  (3/3 * 3/3 * 1/3 = 1/3) :=
sorry

end probability_divisible_by_5_l300_300662


namespace problem_statement_equality_condition_l300_300312

theorem problem_statement (a b n : ℕ) (h_ab_pos : 0 < a ∧ 0 < b ∧ 0 < n) 
  (h_a_gt_b : a > b) 
  (h_ab_eq_n2 : a * b - 1 = n ^ 2) :
  a - b ≥ Int.sqrt (4 * n - 3) := sorry
  
theorem equality_condition (u : ℕ) :
  let a := u^2 + 2*u + 2
  let b := u^2 + 1
  let n := u^2 + u + 1
  (a > b ∧ a*b - 1 = n^2 ∧ a - b = Int.sqrt (4*n - 3)) :=
begin
  assume (h_u_nonneg : 0 ≤ u),
  sorry
end

end problem_statement_equality_condition_l300_300312


namespace problem1_problem2_l300_300269

noncomputable def eval_expr1 : ℝ :=
  log 8 + log 125 - (1 / 7)^(-2) + 16^(3 / 4) + (sqrt 3 - 1)^0

theorem problem1 : eval_expr1 = -37 := by
  sorry

noncomputable def eval_expr2 : ℝ :=
  sin (25 * Real.pi / 6) + cos (25 * Real.pi / 3) + tan (-25 * Real.pi / 4)

theorem problem2 : eval_expr2 = 0 := by
  sorry

end problem1_problem2_l300_300269


namespace probability_prime_on_spinner_l300_300545

open Set

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def spinner_sectors : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 9}
def prime_sectors := spinner_sectors.filter is_prime
def total_sectors := spinner_sectors.card

theorem probability_prime_on_spinner :
  prime_sectors.card.to_rat / total_sectors = 7 / 8 :=
by
  sorry

end probability_prime_on_spinner_l300_300545


namespace sequence_filling_l300_300366

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l300_300366


namespace factorize_expression_l300_300668

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l300_300668


namespace original_price_of_computer_l300_300762

theorem original_price_of_computer (P : ℝ) (h1 : 1.20 * P = 351) (h2 : 2 * P = 585) : P = 292.5 :=
by
  sorry

end original_price_of_computer_l300_300762


namespace sum_of_first_60_digits_l300_300550

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end sum_of_first_60_digits_l300_300550


namespace necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l300_300339

-- Problem 1
theorem necessary_condition_for_q_implies_m_bounds (m : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 20 ≤ 0 → 1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (- Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

-- Problem 2
theorem necessary_but_not_sufficient_condition_for_not_q (m : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - 8 * x - 20 ≤ 0) → ¬ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) → (m ≥ 3 ∨ m ≤ -3) :=
sorry

end necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l300_300339


namespace integral_2x_plus_1_l300_300643

theorem integral_2x_plus_1 :
  (∫ x in 0..1, 2 * x + 1) = 2 :=
by
  sorry

end integral_2x_plus_1_l300_300643


namespace arithmetic_mean_difference_l300_300156

theorem arithmetic_mean_difference :
  let avg1 := (100 + 400) / 2
  let avg2 := (50 + 250) / 2
  avg1 - avg2 = 100 :=
by
  let avg1 := (100 + 400) / 2
  let avg2 := (50 + 250) / 2
  have h1 : avg1 = 250 := by norm_num
  have h2 : avg2 = 150 := by norm_num
  calc avg1 - avg2 = 250 - 150 : by rw [h1, h2]
               ... = 100 : by norm_num

end arithmetic_mean_difference_l300_300156


namespace gcd_of_repeated_three_digit_integers_l300_300195

theorem gcd_of_repeated_three_digit_integers : ∀ (n : ℕ), n ∈ (Set.range (fun k => 100 ≤ k ∧ k < 1000)) →
  (∀ m : ℕ, (∃ k : ℕ, k = 1001001 * m) → gcd(n * 1001001, 1001001 * m) = 1001001) :=
by
  intros n hn m hm
  sorry

end gcd_of_repeated_three_digit_integers_l300_300195


namespace rods_selection_max_rods_selection_l300_300596

theorem rods_selection (n : ℕ)
  (cube : matrix (fin n) (fin n) (fin n) ℕ)
  (rods : fin n → list (fin n × fin n × fin n))
  (∀ i : fin n, ∀ j : fin n, ∀ k: fin n, cube i j k > 0) 
  (∀ r : fin n × fin n × fin n, r ∈ rods l → r.1 = i → r.2 = j) :
  ∃ (selected : list (fin n × fin n × fin n)), (selected.length = n ∧ (∀ i j k l, (selected i j k) ≠ (selected i j l))) :=
begin
  sorry
end

theorem max_rods_selection (n : ℕ)
  (cube : matrix (fin n) (fin n) (fin n) ℕ)
  (rods : fin n → list (fin n × fin n × fin n))
  (∀ i : fin n, ∀ j : fin n, ∀ k: fin n, cube i j k > 0) 
  (∀ r : fin n × fin n × fin n, r ∈ rods l → r.1 = i → r.2 = j) :
  ∃ (selected : list (fin n × fin n × fin n)), (selected.length = 2 * n ∧ (∀ i j k l, (selected i j k) ≠ (selected i j l))) :=
begin
  sorry
end

end rods_selection_max_rods_selection_l300_300596


namespace value_at_pi_div_12_monotonic_intervals_and_range_l300_300320

noncomputable def f (ω x : ℝ) : ℝ := 
  sqrt 3 * (cos (ω * x)) ^ 2 - (sin (ω * x)) * (cos (ω * x)) - sqrt 3 / 2

theorem value_at_pi_div_12 (ω : ℝ) (hω : ω > 0) (h_period : ∃ T > 0, T = π ∧ ∀ x, f ω (x + T) = f ω x) :
  f ω (π / 12) = 1 / 2 :=
by
  sorry

theorem monotonic_intervals_and_range (ω : ℝ) (hω : ω > 0) (h_period : ∃ T > 0, T = π ∧ ∀ x, f ω (x + T) = f ω x) :
  (∀ x ∈ Icc 0 (5 * π / 12), f ω x > f ω (x + π / 6)) ∧
  (∀ x ∈ Icc (5 * π / 12) (7 * π / 12), f ω x < f ω (x + π / 6)) ∧
  (∀ x ∈ Icc 0 (7 * π / 12), f ω x ∈ Icc (-1) (sqrt 3 / 2)) :=
by
  sorry

end value_at_pi_div_12_monotonic_intervals_and_range_l300_300320


namespace parametric_equation_curve_C_general_equation_of_line_l_max_min_PA_length_l300_300324

-- Definitions and conditions
def curve_C (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 9) = 1
def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

-- Parametric equations
def parametric_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 3 * Real.sin θ)
def general_equation_line (x y : ℝ) : Prop := 2 * x + y - 6 = 0

-- Distance formula
def distance_from_P_to_l (θ : ℝ) : ℝ :=
  let (x, y) := (2 * Real.cos θ, 3 * Real.sin θ)
  abs ((4 * Real.cos θ + 3 * Real.sin θ - 6) / Real.sqrt 5)

-- Length PA as a function of θ, considering a 30° angle
def PA_length (θ α : ℝ) : ℝ :=
  let d := distance_from_P_to_l θ
  (2 * Real.sqrt 5 / 5) * abs (5 * Real.sin (θ + α) - 6)

theorem parametric_equation_curve_C :
  ∀ θ : ℝ, parametric_C θ = (2 * Real.cos θ, 3 * Real.sin θ) := sorry

theorem general_equation_of_line_l :
  ∀ x y : ℝ, x = 2 + t → y = 2 - 2 * t → (general_equation_line x y) := sorry

theorem max_min_PA_length :
  ∀ θ α : ℝ, Real.sin (θ + α) = -1 ∨ Real.sin (θ + α) = 1 →
  (PA_length θ α = if Real.sin (θ + α) = -1 then (22 * Real.sqrt 5 / 5) else (2 * Real.sqrt 5 / 5)) := sorry

end parametric_equation_curve_C_general_equation_of_line_l_max_min_PA_length_l300_300324


namespace discount_percentage_correct_l300_300078

-- Define the problem parameters as variables
variables (sale_price marked_price : ℝ) (discount_percentage : ℝ)

-- Provide the conditions from the problem
def conditions : Prop :=
  sale_price = 147.60 ∧ marked_price = 180

-- State the problem: Prove the discount percentage is 18%
theorem discount_percentage_correct (h : conditions sale_price marked_price) : 
  discount_percentage = 18 :=
by
  sorry

end discount_percentage_correct_l300_300078


namespace sam_drove_200_miles_l300_300429

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l300_300429


namespace inequality_part1_inequality_part2_l300_300494

section Proof

variable {x m : ℝ}
def f (x : ℝ) : ℝ := |2 * x + 2| + |2 * x - 3|

-- Part 1: Prove the solution set for the inequality f(x) > 7
theorem inequality_part1 (x : ℝ) :
  f x > 7 ↔ (x < -3 / 2 ∨ x > 2) := 
  sorry

-- Part 2: Prove the range of values for m such that the inequality f(x) ≤ |3m - 2| has a solution
theorem inequality_part2 (m : ℝ) :
  (∃ x, f x ≤ |3 * m - 2|) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
  sorry

end Proof

end inequality_part1_inequality_part2_l300_300494


namespace probability_of_selecting_particular_girl_l300_300899

-- Define the numbers involved
def total_population : ℕ := 60
def num_girls : ℕ := 25
def num_boys : ℕ := 35
def sample_size : ℕ := 5

-- Total number of basic events
def total_combinations : ℕ := Nat.choose total_population sample_size

-- Number of basic events that include a particular girl
def girl_combinations : ℕ := Nat.choose (total_population - 1) (sample_size - 1)

-- Probability of selecting a particular girl
def probability_of_girl_selection : ℚ := girl_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_selecting_particular_girl :
  probability_of_girl_selection = 1 / 12 :=
by sorry

end probability_of_selecting_particular_girl_l300_300899


namespace probability_sum_22_l300_300597

def fairness_of_die1 (n : ℕ) : Prop :=
  n ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19} : set ℕ) ∨ n = 0

def fairness_of_die2 (n : ℕ) : Prop :=
  n ∈ ({1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} : set ℕ) ∨ n = 0

def total_possible_rolls := 400

def valid_ways_sum_22 := 17

theorem probability_sum_22 :
  ∑ n m, (fairness_of_die1 n ∧ fairness_of_die2 m ∧ n + m = 22) =
  valid_ways_sum_22 →
  (valid_ways_sum_22 * 1) / total_possible_rolls = 17 / 400 :=
by
  sorry

end probability_sum_22_l300_300597


namespace AM_eq_NC_l300_300951

theorem AM_eq_NC 
  (ABC : Triangle) 
  (incircle : ∃ (center : Point) (radius : ℝ), True) 
  (M : Point)
  (MK : Line) 
  (N : Point) 
  (tangent_M_AC: IsTangent M AC) 
  (MK_diameter: IsDiameter MK incircle) 
  (BK_intersects_AC: ∃ N', BK.Intersects AC N ∧ N' = N)
  : AM.length = NC.length := 
sorry

end AM_eq_NC_l300_300951


namespace min_A_max_B_l300_300376

-- Part (a): prove A = 15 is the smallest value satisfying the condition
theorem min_A (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : A = 15 := 
sorry

-- Part (b): prove B = 76 is the largest value satisfying the condition
theorem max_B (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : B = 76 := 
sorry

end min_A_max_B_l300_300376


namespace cone_base_divide_ratio_l300_300855

/-- The cone plane division ratio problem -/
theorem cone_base_divide_ratio (α β : ℝ) (h : β < α) :
  let ratio := (Real.arccos (Real.tan β / Real.tan α)) in
  ratio / (Real.pi - ratio) = (Real.arccos (Real.tan β / Real.tan α)) / (Real.pi - Real.arccos (Real.tan β / Real.tan α)) :=
by
  sorry

end cone_base_divide_ratio_l300_300855


namespace triangle_length_bc_l300_300363

noncomputable def cos_rule_length_bc (AB AC : ℝ) (angleA : ℝ) : ℝ :=
  (AB^2 + AC^2 - 2 * AB * AC * Real.cos angleA).sqrt

theorem triangle_length_bc {BC : ℝ}
  (H1 : |AB| = 4)
  (H2 : |AC| = 2)
  (H3 : ∠A = Real.pi / 3) :
  BC = 2 * Real.sqrt 3 :=
by
  sorry

end triangle_length_bc_l300_300363


namespace variance_of_data_set_l300_300885

theorem variance_of_data_set :
  let data := [-2, -1, 0, 3, 5]
  let mean := (data.sum / data.length : ℚ)
  let variance := (data.map (λ x, (x - mean)^2)).sum / data.length
  variance = 34 / 5 := by
  sorry

end variance_of_data_set_l300_300885


namespace sam_driving_distance_l300_300423

-- Definitions based on the conditions
def marguerite_distance : ℝ := 150
def marguerite_time : ℝ := 3
def sam_time : ℝ := 4

-- Desired statement using the given conditions
theorem sam_driving_distance :
  let rate := marguerite_distance / marguerite_time in
  let sam_distance := rate * sam_time in
  sam_distance = 200 :=
by
  sorry

end sam_driving_distance_l300_300423


namespace problem_a_2006_l300_300411

open Nat

def sequence_a (n : ℕ) : ℚ :=
  if n = 1 then 4
  else if n = 2 then 5
  else sequence_a (n-1) / sequence_a (n-2)

theorem problem_a_2006 : sequence_a 2006 = 5 := by
  sorry

end problem_a_2006_l300_300411


namespace shopkeeper_bananas_l300_300615

-- Defining the problem conditions
def shopkeeper_conditions (B : ℕ) :=
  let good_oranges := 510 -- 85% of 600
  let good_bananas := 0.95 * B
  let total_fruits := 600 + B
  let good_fruits := good_oranges + good_bananas
  (good_fruits / total_fruits = 0.89)

-- The proof statement itself, showing that B = 400 satisfies the shopkeeper conditions
theorem shopkeeper_bananas : shopkeeper_conditions 400 :=
by 
  sorry

end shopkeeper_bananas_l300_300615


namespace brock_peanuts_ratio_l300_300124

theorem brock_peanuts_ratio (initial : ℕ) (bonita : ℕ) (remaining : ℕ) (brock : ℕ)
  (h1 : initial = 148) (h2 : bonita = 29) (h3 : remaining = 82) (h4 : brock = 37)
  (h5 : initial - remaining = bonita + brock) :
  (brock : ℚ) / initial = 1 / 4 :=
by {
  sorry
}

end brock_peanuts_ratio_l300_300124


namespace smallest_prime_with_tens_digit_2_and_composite_reverse_l300_300687

def is_two_digit_prime_with_tens_digit_2 (n : ℕ) : Prop :=
  n > 9 ∧ n < 100 ∧ Prime n ∧ (n / 10 = 2)

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_prime_with_tens_digit_2_and_composite_reverse :
  ∃ p, is_two_digit_prime_with_tens_digit_2 p ∧ composite (reverse_digits p) ∧ p = 23 :=
begin
  sorry
end

end smallest_prime_with_tens_digit_2_and_composite_reverse_l300_300687


namespace min_value_frac_l300_300296

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) :
  (1 / x + 1 / (3 * y)) = 4 :=
by
  sorry

end min_value_frac_l300_300296


namespace log_eq_solution_l300_300562

theorem log_eq_solution (a x : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : -4 < x ∧ x < 4)
  (h4 : log a (sqrt (4 + x)) + 3 * log (a^2) (4 - x) - log (a^4) ((16 - x^2)^2) = 2)
  : a ∈ Ioo 0 1 ∨ a ∈ Ioo 1 (2 * Real.sqrt 2) :=
sorry

end log_eq_solution_l300_300562


namespace find_S6_l300_300990

-- Define the increasing geometric sequence with positive terms
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ (q > 1) ∧ (a 0 > 0)

-- Given conditions
def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 + a 3 = 30

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 0 * a 4 = 81

-- Sum of first n terms of the geometric sequence
def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (q ^ n - 1) / (q - 1)

-- The theorem to be proved
theorem find_S6 (a : ℕ → ℝ) (q : ℝ)
  (h_seq : increasing_geometric_sequence a q)
  (h_cond1 : condition_1 a q)
  (h_cond2 : condition_2 a) :
  geometric_sum a q 6 = 364 := sorry

end find_S6_l300_300990


namespace circle_positional_relationship_l300_300514

noncomputable def Circle (h k r : ℝ) := ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2

theorem circle_positional_relationship :
  let C1 := Circle 2 2 1
  let C2 := Circle 2 5 4
  (C1, C2) -> "internally tangent" :=
by
  sorry

end circle_positional_relationship_l300_300514


namespace number_not_equal_54_l300_300123

def initial_number : ℕ := 12
def target_number : ℕ := 54
def total_time : ℕ := 60

theorem number_not_equal_54 (n : ℕ) (time : ℕ) : (time = total_time) → (n = initial_number) → 
  (∀ t : ℕ, t ≤ time → (n = n * 2 ∨ n = n / 2 ∨ n = n * 3 ∨ n = n / 3)) → n ≠ target_number :=
by
  sorry

end number_not_equal_54_l300_300123


namespace right_angled_triangle_l300_300872

-- Define the problem setup, given conditions must be declared as assumptions.
theorem right_angled_triangle 
    (A B C H_3 M_3 : Type) 
    (triangle_ABC : is_triangle A B C) 
    (is_height : is_height CH_3 A B C)
    (is_median : is_median CM_3 A B C)
    (angle_div : angle_divides_three_parts A B C H_3 M_3) : 
    (angle A C B) = 90 := 
begin
  sorry
end

end right_angled_triangle_l300_300872


namespace range_of_a_for_increasing_function_l300_300359

noncomputable def f (a : ℝ) : PiecewiseFunction ℝ ℝ :=
  PiecewiseFunction.of
    (λ x, log a x + a, { x : ℝ // x > 1 })
    (λ x, (2 - (a / 3)) * x + 2, { x : ℝ // x ≤ 1 })

theorem range_of_a_for_increasing_function :
  ∀ (a : ℝ), 
  (∀ x y : ℝ, x > y → f a x > f a y) ↔ (3 ≤ a ∧ a < 6) :=
by
  sorry

end range_of_a_for_increasing_function_l300_300359


namespace correct_statements_l300_300147

-- Define the statements as boolean propositions
def statement1 : Prop := "Correlation is a type of non-deterministic relationship."
def statement2 : Prop := "Any set of data has a regression equation."
def statement3 : Prop := "Scatter plots can intuitively reflect the degree of correlation between data."

-- Define the conditions as stating the correctness of these statements
def is_correct_statement1 : Prop := statement1 = "Correlation is a type of non-deterministic relationship."
def is_correct_statement3 : Prop := statement3 = "Scatter plots can intuitively reflect the degree of correlation between data."

-- Correct answer based on the given conditions
def correct_answer : Prop := "C" = "C"

-- Mathematical proof statement translating the question and its conditions into a Lean 4 statement.
theorem correct_statements : is_correct_statement1 ∧ ¬is_correct_statement2 ∧ is_correct_statement3 → correct_answer := sorry

end correct_statements_l300_300147


namespace e_exp_f_neg2_l300_300493

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
axiom h_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_ln_pos : ∀ x : ℝ, x > 0 → f x = Real.log x

-- Theorem to prove:
theorem e_exp_f_neg2 : Real.exp (f (-2)) = 1 / 2 := by
  sorry

end e_exp_f_neg2_l300_300493


namespace exists_line_m_perpendicular_to_l_l300_300304

-- Definitions (Conditions)
variable {Point : Type}
variable {Line : Type}
variable {Plane : Type}
variable (S : Plane) (A : Point) (l : Line)

-- Assume A is a point in the plane S
axiom A_in_S : A ∈ S

-- Assume l is a line outside the plane S, 
-- representing this using ¬(l ∈ S) (though in Lean 4, we might need a concrete way to state "outside")
axiom l_not_in_S : ¬ (l ∈ S)

-- The proof statement
theorem exists_line_m_perpendicular_to_l :
  ∃ (m : Line), m ∈ S ∧ A ∈ m ∧ is_perpendicular m l := sorry

end exists_line_m_perpendicular_to_l_l300_300304


namespace average_weight_increase_l300_300860

-- Given constants
def weight_old := 75
def weight_new := 95
def total_weight_increase := weight_new - weight_old
def persons := 8

theorem average_weight_increase :
  let x := total_weight_increase / persons in
  x = 2.5 := by
{
  -- Here we will have our proof steps
  sorry
}

end average_weight_increase_l300_300860


namespace find_constants_l300_300027

-- Define constants and the problem
variables (C D Q : Type) [AddCommGroup Q] [Module ℝ Q]
variables (CQ QD : ℝ) (h_ratio : CQ = 3 * QD / 5)

-- Define the conjecture we want to prove
theorem find_constants (t u : ℝ) (h_t : t = 5 / (3 + 5)) (h_u : u = 3 / (3 + 5)) :
  (CQ = 3 * QD / 5) → 
  (t * CQ + u * QD = (5 / 8) * CQ + (3 / 8) * QD) :=
sorry

end find_constants_l300_300027


namespace perimeter_of_PQRS_l300_300160

/-- Define the dimensions given in the problem --/
def PG : ℝ := 12
def GQ : ℝ := 25
def HR : ℝ := 7

def P (pG : ℝ) (gQ : ℝ) := pG + gQ
def S (hR : ℝ) := hR

/-- Mathematically equivalent proof problem statement --/
theorem perimeter_of_PQRS (pG : ℝ) (gQ : ℝ) (hR : ℝ) : 
  pG = 12 → gQ = 25 → hR = 7 → 
  let rectangle_perimeter := 2 * (gQ + (13 + gQ)) in
  rectangle_perimeter = (119.5384) :=
by
  intros
  sorry

end perimeter_of_PQRS_l300_300160


namespace water_pouring_problem_l300_300600

theorem water_pouring_problem :
  ∃ n : ℕ, (∏ k in finset.range n, (k + 2)/(k + 3) = 1/15) ∧ n = 28 :=
by
  sorry

end water_pouring_problem_l300_300600


namespace vector_inequality_l300_300744

open Real

noncomputable theory

variables {a b : ℝ^3} (hab : ‖a + b‖ = ‖b‖) (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0)

theorem vector_inequality :
  2 * ‖b‖ > ‖a + 2 * b‖ :=
sorry

end vector_inequality_l300_300744


namespace sequence_filling_l300_300367

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l300_300367


namespace sum_of_divisors_of_ten_l300_300688

theorem sum_of_divisors_of_ten : 
  let S := {m : ℕ | m ∣ 10 ∧ m > 0} 
  in ∑ m in S, m = 18 :=
by
  sorry

end sum_of_divisors_of_ten_l300_300688


namespace square_area_l300_300512

theorem square_area (p1 p2 : ℝ × ℝ) (h1 : p1 = (1, 3)) (h2 : p2 = (-2, 5))
  (h_adj : ∃ p3 p4 : ℝ × ℝ, square p1 p2 p3 p4) :
  (∃ s : ℝ, s = real.sqrt ((-2 - 1)^2 + (5 - 3)^2)) ∧ 
  (∃ A : ℝ, ∀ s, A = s^2 ) ∧ 
  (A = 13) :=
by
  sorry

end square_area_l300_300512
