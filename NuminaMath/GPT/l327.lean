import Mathlib

namespace total_students_experimental_primary_school_l327_327322

theorem total_students_experimental_primary_school : 
  ∃ (n : ℕ), 
  n = (21 + 11) * 28 ∧ 
  n = 896 := 
by {
  -- Since the proof is not required, we use "sorry"
  sorry
}

end total_students_experimental_primary_school_l327_327322


namespace triangle_perimeter_l327_327683

-- Definitions of the geometric problem conditions
def inscribed_circle_tangent (A B C P : Type) : Prop := sorry
def radius_of_inscribed_circle (r : ℕ) : Prop := r = 24
def segment_lengths (AP PB : ℕ) : Prop := AP = 25 ∧ PB = 29

-- Main theorem to prove the perimeter of the triangle ABC
theorem triangle_perimeter (A B C P : Type) (r AP PB : ℕ)
  (H1 : inscribed_circle_tangent A B C P)
  (H2 : radius_of_inscribed_circle r)
  (H3 : segment_lengths AP PB) :
  2 * (54 + 208.72) = 525.44 :=
  sorry

end triangle_perimeter_l327_327683


namespace collinear_points_find_k_l327_327042

theorem collinear_points_find_k :
  ∀ (k : ℝ),
    let OA := (k, 2)
    let OB := (1, 2k)
    let OC := (1 - k, -1)
    let AB := (1 - k, 2k - 2)
    let BC := (-k, -1 - 2k)
    (AB.1 * BC.2 - AB.2 * BC.1 = 0) →
    OA ≠ OB →
    OA ≠ OC →
    OB ≠ OC →
    k = -1 / 4 := by
  sorry

end collinear_points_find_k_l327_327042


namespace no_prime_sum_10003_l327_327538

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327538


namespace population_after_four_years_l327_327297

def initial_population : ℝ := 20
def yearly_population (a_k : ℝ) : ℝ := 1.9 * a_k - 6.65

theorem population_after_four_years :
  let a_0 := initial_population in
  let a_1 := yearly_population a_0 in
  let a_2 := yearly_population a_1 in
  let a_3 := yearly_population a_2 in
  let a_4 := yearly_population a_3 in
  round a_4 = 172 :=
by
  let a_0 : ℝ := initial_population
  let a_1 : ℝ := yearly_population a_0
  let a_2 : ℝ := yearly_population a_1
  let a_3 : ℝ := yearly_population a_2
  let a_4 : ℝ := yearly_population a_3
  have h : round a_4 = 172 := sorry
  exact h

end population_after_four_years_l327_327297


namespace min_n_1014_dominoes_l327_327931

theorem min_n_1014_dominoes (n : ℕ) :
  (n + 1) ^ 2 ≥ 6084 → n ≥ 77 :=
sorry

end min_n_1014_dominoes_l327_327931


namespace correct_option_C_l327_327394

-- Define points A, B and C given their coordinates and conditions
structure Point (α : Type _) :=
(x : α)
(y : α)

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

variables (a b c d : ℝ)
variable hA : Point ℝ := ⟨a, 2⟩
variable hB : Point ℝ := ⟨b, 6⟩
variable hC : Point ℝ := ⟨c, d⟩
variables (ha_ON_parabola : hA.y = parabola hA.x)
          (hb_ON_parabola : hB.y = parabola hB.x)
          (hc_ON_parabola : hC.y = parabola hC.x)
          (hd_lt_one : d < 1)

theorem correct_option_C (ha_lt_0 : a < 0) (hb_gt_0 : b > 0) : a < c ∧ c < b :=
by
-- Proof will be done here, currently left as sorry just to state the theorem.
sorry

end correct_option_C_l327_327394


namespace pentagon_stack_valid_sizes_l327_327034

def valid_stack_size (n : ℕ) : Prop :=
  ¬ (n = 1) ∧ ¬ (n = 3)

theorem pentagon_stack_valid_sizes (n : ℕ) :
  valid_stack_size n :=
sorry

end pentagon_stack_valid_sizes_l327_327034


namespace quadrilateral_perimeter_l327_327270

theorem quadrilateral_perimeter (A B C D P : Point)
    (PA PB PC PD : ℝ)
    (hPA : PA = 30) (hPB : PB = 40) (hPC : PC = 35) (hPD : PD = 50)
    (H : convex_quadrilateral A B C D)
    (areaABCD : 2500) :
    perimeter A B C D = 222.49 := 
sorry

end quadrilateral_perimeter_l327_327270


namespace sum_of_nth_row_odd_numbers_l327_327117

theorem sum_of_nth_row_odd_numbers (n : ℕ) : 
  let first_term := n^2 - n + 1 in
  let last_term := first_term + 2 * (n - 1) in
  (n * (first_term + last_term) / 2) = n^3 := 
by sorry

end sum_of_nth_row_odd_numbers_l327_327117


namespace intersection_nonempty_implies_a_l327_327396

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0 ∧ x ∈ Int}

theorem intersection_nonempty_implies_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 ∨ a = 2 :=
sorry

end intersection_nonempty_implies_a_l327_327396


namespace intersection_of_lines_l327_327120

noncomputable def line_parametric (p q : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
(p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2), p.3 + t * (q.3 - p.3))

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
⟨((4 : ℕ).out + 10 * (-1 / 15 : ℚ)).out,
 ((-8 : ℕ).out - 10 * (-1 / 15 : ℚ)).out,
 ((8 : ℕ).out + 6 * (-1 / 15 : ℚ)).out⟩

theorem intersection_of_lines
(P Q R S : ℝ × ℝ × ℝ)
(hP : P = (4, -8, 8))
(hQ : Q = (14, -18, 14))
(hR : R = (1, 2, -7))
(hS : S = (3, -6, 9)) :
intersection_point P Q R S = (14 / 3, -22 / 3, 38 / 3) :=
by
  -- proof setup ...
  sorry

end intersection_of_lines_l327_327120


namespace no_prime_sum_10003_l327_327504

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327504


namespace unique_sum_of_two_primes_l327_327555

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327555


namespace fill_time_first_and_fourth_taps_l327_327695

noncomputable def pool_filling_time (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) : ℝ :=
  m / (x + u)

theorem fill_time_first_and_fourth_taps (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) :
  pool_filling_time m x y z u h₁ h₂ h₃ = 12 / 5 :=
sorry

end fill_time_first_and_fourth_taps_l327_327695


namespace shortest_path_between_two_points_l327_327257

/-- The mathematical principle that between two points, the line segment is the shortest --/
theorem shortest_path_between_two_points {P Q : ℝ × ℝ} (h : true) : 
  ∀ R : (ℝ × ℝ) → ℝ, (∀ S : (ℝ × ℝ), dist (P, S) + dist (S, Q) ≥ dist (P, Q)) ∧ (R (P, Q) = dist (P, Q)) := 
by
  sorry

end shortest_path_between_two_points_l327_327257


namespace vector_equation_solution_l327_327073

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := 
sorry

end vector_equation_solution_l327_327073


namespace two_adjacent_numbers_exist_l327_327754

-- Define the condition for 2005 natural numbers arranged in a circle.
def circle (n : ℕ) := fin n → ℕ

-- Statement to be proved
theorem two_adjacent_numbers_exist (n : ℕ) (h : n = 2005) 
  (nums : circle n) :
  ∃ i : fin n, ∀ nums_rem : fin (n - 2) → ℕ,
  (nums_rem = λ j, if j.val < i.val then nums ⟨j.val, by sorry⟩ else nums ⟨j.val + 2, by sorry⟩) →
  ¬∃ (g1 g2 : fin (n - 2) → ℕ), sum g1 = sum g2 :=
sorry

end two_adjacent_numbers_exist_l327_327754


namespace john_borrows_2000_l327_327140

theorem john_borrows_2000
  (initial_amount : ℝ)
  (interest_rate : ℝ)
  (months : ℕ) :
  initial_amount * (1 + interest_rate) ^ months ≥ 3 * initial_amount ↔ months ≥ 17 :=
by
  let initial_amount := 2000
  let interest_rate := 0.06
  let months := ∀ t : ℕ, t
  sorry

end john_borrows_2000_l327_327140


namespace correct_option_l327_327389

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l327_327389


namespace unique_function_l327_327450

noncomputable def f : ℝ → ℝ := sorry

axiom domain_condition : ∀ x : ℝ, x ∈ set.univ

axiom additivity_condition : ∀ x1 x2 : ℝ, x1 + x2 ≠ 0 → f x1 + f x2 = 0

axiom monotonicity_condition : ∀ x t : ℝ, t > 0 → f (x + t) > f x

theorem unique_function : f = λ x : ℝ, x^3 := sorry

end unique_function_l327_327450


namespace eq_fractions_l327_327241

theorem eq_fractions : 
  (1 + 1 / (1 + 1 / (1 + 1 / 2))) = 8 / 5 := 
  sorry

end eq_fractions_l327_327241


namespace magnitude_not_determined_l327_327403

-- Define the approximate numbers A and B
def A : Real := 3.6
def B : Real := 3.60

-- Problem statement: Prove that the exact values of A and B cannot be determined
theorem magnitude_not_determined : ¬∀ ε > 0, ∃ a b : Real, (a = A ∧ b = B) ∧ (0 < |a - b| < ε) := by
  sorry

end magnitude_not_determined_l327_327403


namespace pencils_initial_count_l327_327831

theorem pencils_initial_count (pencils_initially: ℕ) :
  (∀ n, n > 0 → n < 36 → 36 % n = 1) →
  pencils_initially + 30 = 36 → 
  pencils_initially = 6 :=
by
  intro h hn
  sorry

end pencils_initial_count_l327_327831


namespace shorter_side_length_l327_327093

theorem shorter_side_length 
  (L W : ℝ) 
  (h1 : L * W = 117) 
  (h2 : 2 * L + 2 * W = 44) :
  L = 9 ∨ W = 9 :=
by
  sorry

end shorter_side_length_l327_327093


namespace sin_theta_l327_327373

def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta:
  (∀ x, f x ≤ f θ) → Real.sin θ = 3 / 5 :=
by
  sorry

end sin_theta_l327_327373


namespace physics_experiment_kits_l327_327226

theorem physics_experiment_kits (x : ℤ) (y : ℤ) (unit_price_B : ℤ) (unit_price_A : ℤ) : 
  (unit_price_A = 1.2 * unit_price_B) ∧ 
  (9900 / unit_price_A - 7500 / unit_price_B = 5) ∧ 
  (20 - y + y = 20) ∧ 
  (180 * (20 - y) + 150 * y ≤ 3400) -> 
  (unit_price_B = 150) ∧ 
  (unit_price_A = 180) ∧ 
  (y ≥ 7) :=
by
  sorry

end physics_experiment_kits_l327_327226


namespace solution_set_of_inequality_l327_327612

noncomputable def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := x * f (2 * x)

theorem solution_set_of_inequality (h1 : ∀ x < 0, 2 * x * deriv (f (2 * x)) + f (2 * x) < 0)
  (h2 : f (-2) = 0)
  (h3 : ∀ x, f (-x) = -f (x)) :
  {x : ℝ | x * f (2 * x) < 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end solution_set_of_inequality_l327_327612


namespace count_positive_integers_m_l327_327840

theorem count_positive_integers_m (hdiv : ∀ m : ℕ, (m > 0) → 1764 % (m^2 - 4) = 0 → m ∈ {4, 8, 16}) :
    {m : ℕ | m > 0 ∧ 1764 % (m^2 - 4) = 0}.finite.card = 3 :=
by 
  sorry

end count_positive_integers_m_l327_327840


namespace fixed_point_of_function_l327_327169

theorem fixed_point_of_function (a : ℝ) : 
  (a - 1) * 2^1 - 2 * a = -2 := by
  sorry

end fixed_point_of_function_l327_327169


namespace james_writing_time_l327_327596

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l327_327596


namespace solve_for_y_l327_327192

theorem solve_for_y (y : ℕ) : 9^y = 3^12 → y = 6 :=
by
  sorry

end solve_for_y_l327_327192


namespace magnitude_and_projection_l327_327906

noncomputable def vec : Type := ℝ × ℝ × ℝ

namespace vector

def length (v : vec) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def add (v₁ v₂ : vec) : vec :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2, v₁.3 + v₂.3)

def dot (v₁ v₂ : vec) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

def projection (v₁ v₂ : vec) : ℝ :=
  vector.dot v₁ v₂ / vector.length v₂

end vector

open vector

theorem magnitude_and_projection (a b : vec) (h₁ : length a = 3) (h₂ : length (add a b) = real.sqrt 13)
  (h₃ : dot a b = -3) :
  length b = 4 ∧ projection b a = -2 :=
by
  sorry

end magnitude_and_projection_l327_327906


namespace max_alpha_minus_beta_l327_327377

open Real

theorem max_alpha_minus_beta (a b: ℝ) (α β: ℝ) 
  (h1: a > b) (h2: b > 0) 
  (h3: α ∈ Ioo 0 (π / 2)) (h4: β ∈ Ioo 0 (π / 2)) 
  (h5: a * tan β = b * tan α) :
  max (α - β) = arctan ((a - b) / (2 * sqrt (a * b))) :=
sorry

end max_alpha_minus_beta_l327_327377


namespace overall_loss_is_450_l327_327786

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450_l327_327786


namespace sufficient_but_not_necessary_condition_l327_327867

-- Define the conditions as predicates
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

-- Range for m where p is sufficient but not necessary for q
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- The main goal to be proven
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x) ↔ m_range m :=
sorry

end sufficient_but_not_necessary_condition_l327_327867


namespace translated_line_is_correct_l327_327230

def original_line (x : ℝ) : ℝ := (1/2) * x - 2
def translated_line (x : ℝ) : ℝ := original_line x + 3

theorem translated_line_is_correct (x : ℝ) : translated_line x = (1/2) * x + 1 :=
by 
  sorry

end translated_line_is_correct_l327_327230


namespace digits_of_2_pow_100_l327_327353

theorem digits_of_2_pow_100 : 
  (10:ℝ) ^ 60 ≤ (2:ℝ) ^ 200 ∧ (2:ℝ) ^ 200 < (10:ℝ) ^ 61 → 
  ∃ n: ℕ, (n = 31 ∧ (10:ℝ) ^ (n-1 : ℝ) ≤ (2:ℝ) ^ 100 ∧ (2:ℝ) ^ 100 < (10:ℝ)n) :=
by
  sorry

end digits_of_2_pow_100_l327_327353


namespace simplifies_to_4_and_5_l327_327234

noncomputable def triplet_sums (I II III IV : list (ℕ × ℕ)) := 
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 1) I) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 1) I) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 2) I) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 2) I) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 3) I) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 3) I) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 1) II) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 1) II) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 2) II) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 2) II) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 3) II) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 3) II) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 1) III) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 1) III) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 2) III) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 2) III) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 3) III) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 3) III) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 1) IV) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 1) IV) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 2) IV) = list.sum (list.map (λ (n : ℕ × ℕ), n.2 ^ 2) IV) ∧
  list.sum (list.map (λ (n : ℕ × ℕ), n.1 ^ 3) IV) = list.sum (list.map (λ (n: ℕ × ℕ), n.2 ^ 3) IV)

theorem simplifies_to_4_and_5 (I II III IV : list (ℕ × ℕ)) (h : triplet_sums I II III IV) :
  ∃ k : ℤ, 
    (k = 70 ∨ k = 160 ∨ k = 230 ∨ k = -70 ∨ k = -160 ∨ k = -230) ∧
    list.length (filter (λ (n : ℕ), n ≠ 0) (list.map (λ (n : ℕ × ℕ), (n.1 + k))) I) ≤ 4 ∧ 
    list.length (filter (λ (n : ℕ), n ≠ 0) (list.map (λ (n : ℕ × ℕ), (n.1 + k))) II) ≤ 4 ∧
    list.length (filter (λ (n : ℕ), n ≠ 0) (list.map (λ (n : ℕ × ℕ), (n.1 + k))) III) ≤ 5 ∧
    list.length (filter (λ (n : ℕ), n ≠ 0) (list.map (λ (n : ℕ × ℕ), (n.1 + k))) IV) ≤ 5 :=
sorry

end simplifies_to_4_and_5_l327_327234


namespace incorrect_statement_C_l327_327795

def statement_A : Prop :=
  (∀ x : ℝ, (x² - 3 * x - 4 = 0) → (x = 4)) ↔ (∀ x : ℝ, (x ≠ 4) → (x² - 3 * x - 4 ≠ 0))

def statement_B : Prop :=
  (∀ x : ℝ, (x² - 3 * x - 4 = 0) → (x = 4 ∨ x = -1)) ∧ (∀ x : ℝ, (x = 4) → (x² - 3 * x - 4 = 0))

def statement_C : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q))

def statement_D : Prop :=
  (∃ x : ℝ, x² + x + 1 < 0) ↔ (∀ x : ℝ, x² + x + 1 ≥ 0)

theorem incorrect_statement_C : ¬statement_C :=
by
  sorry

end incorrect_statement_C_l327_327795


namespace find_relationship_l327_327366

noncomputable def log_equation (c d : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → 6 * (Real.log (x) / Real.log (c))^2 + 5 * (Real.log (x) / Real.log (d))^2 = 12 * (Real.log (x))^2 / (Real.log (c) * Real.log (d))

theorem find_relationship (c d : ℝ) :
  log_equation c d → 
    (d = c ^ (5 / (6 + Real.sqrt 6)) ∨ d = c ^ (5 / (6 - Real.sqrt 6))) :=
by
  sorry

end find_relationship_l327_327366


namespace trig_equation_cosine_form_l327_327201

theorem trig_equation_cosine_form (a b c : ℕ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  (∀ x : ℝ, sin x ^ 2 + sin (2 * x) ^ 2 + sin (5 * x) ^ 2 + sin (6 * x) ^ 2 = 2 →
   cos (a * x) * cos (b * x) * cos (c * x) = 0) →
  a + b + c = 12 :=
sorry

end trig_equation_cosine_form_l327_327201


namespace angle_B_eq_pi_div_three_range_of_expression_l327_327925

variables {A B C a b c : ℝ}

-- Conditions: 
-- 1. Sides a, b, c opposite to angles A, B, C respectively in ∆ABC
-- 2. Equation: (2c - a) * cos B - b * cos A = 0
axiom side_relation : (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Proof problems:
-- 1. Prove B = π / 3
-- 2. Find the range of values for √3 * sin A + sin (C - π / 6)

theorem angle_B_eq_pi_div_three (h : side_relation) : 
  B = Real.pi / 3 := sorry

theorem range_of_expression (h : side_relation) (h_angle : A ∈ (0 : ℝ, 2 * Real.pi / 3)) :
  1 < √3 * Real.sin A + Real.sin (C - Real.pi / 6) ∧ √3 * Real.sin A + Real.sin (C - Real.pi / 6) ≤ 2 := sorry

end angle_B_eq_pi_div_three_range_of_expression_l327_327925


namespace original_denominator_is_7_point_5_l327_327788

theorem original_denominator_is_7_point_5:
  ∃ d : ℝ, (4 + 3) / (d + 3) = 2 / 3 → d = 7.5 :=
by
  assume h : ∃ d : ℝ, (4 + 3) / (d + 3) = (2 : ℝ) / 3
  obtain ⟨d, hd⟩ := h
  exact_mod_cast sorry

end original_denominator_is_7_point_5_l327_327788


namespace krystiana_monthly_earnings_l327_327958

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l327_327958


namespace geometric_sequence_a3_l327_327129

noncomputable def q_square_root_of_2 : ℝ := Real.sqrt 2

theorem geometric_sequence_a3
  (a : ℕ → ℝ)
  (h1 : a 1 = -2)
  (h5 : a 5 = -4)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 3 = -2 * Real.sqrt 2 :=
by
  obtain ⟨q, hq⟩ := h_geometric
  have hq4 : q^4 = 2, from sorry
  have hq2 : q^2 = Real.sqrt 2, from sorry
  have a3_def : a 3 = a 1 * q^2, from sorry
  rw [h1, hq2, a3_def]
  dsimp [Real.sqrt]
  norm_num

end geometric_sequence_a3_l327_327129


namespace circles_non_intersecting_l327_327824

def circle1_equation (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def circle2_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_non_intersecting :
    (∀ (x y : ℝ), ¬(circle1_equation x y ∧ circle2_equation x y)) :=
by
  sorry

end circles_non_intersecting_l327_327824


namespace complement_of_M_in_U_l327_327163

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l327_327163


namespace probability_sum_is_five_l327_327927

theorem probability_sum_is_five :
  let balls := {1, 2, 3, 4}
  let all_pairs := ({1, 2, 3, 4}.image (λ n, ({n} : set ℕ))).powerset.filter (λ s, s.card = 2)
  let favorable_pairs := all_pairs.filter (λ s, (s.val : set ℕ).sum = 5)
  (↑favorable_pairs.card / ↑all_pairs.card = 1 / 3) := 
by 
  sorry

end probability_sum_is_five_l327_327927


namespace proof_18_to_PQ_l327_327665

theorem proof_18_to_PQ (m n : ℤ) (P Q : ℤ) (hP : P = 3^m) (hQ : Q = 2^n) : 18^(m * n) = P^(2 * n) * Q^m := 
by
  sorry

end proof_18_to_PQ_l327_327665


namespace no_prime_sum_10003_l327_327540

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327540


namespace tomato_puree_energy_cost_l327_327074

theorem tomato_puree_energy_cost:
  (tomato_juice_initial : ℝ) (juice_water_content : ℝ) (puree_water_content : ℝ)
  (evaporation_rate : ℝ) (energy_consumption_per_litre : ℝ) (factory_efficiency : ℝ)
  (working_hours : ℝ) (energy_cost_per_unit : ℝ) (juice_litres : ℝ := 20)
  (juice_water_content := 0.90) (puree_water_content := 0.20)
  (evaporation_rate := 0.30) (energy_consumption_per_litre := 15)
  (factory_efficiency := 0.80) (working_hours := 6) (energy_cost_per_unit := 0.10) :
  ∃ (puree_litres : ℝ) (energy_cost : ℝ), puree_litres = 11.2 ∧ energy_cost = 37.5 := by
  sorry

end tomato_puree_energy_cost_l327_327074


namespace height_squared_eq_four_times_product_of_segments_symmetry_about_circumcenter_l327_327618

variables {V : Type*} [inner_product_space ℝ V]

-- Given variables: height of tetrahedron, segments of height in the triangular face
variables (h h1 h2 : ℝ) (A B C : V) (D : V) [is_regular_tetrahedron A B C D]

-- Definitions
def height_of_tetrahedron : Prop := 
  ∃ DH : V, (|D - DH| = h)

def segments_in_face : Prop := 
  ∃ F : V, (|A - F| = h1) ∧ (|F - foot_of_perpendicular (submodule.span ℝ {B - C}) A| = h2)

-- Theorem 1
theorem height_squared_eq_four_times_product_of_segments (h h1 h2 : ℝ) (A B C D DH F : V) 
  [is_regular_tetrahedron A B C D] 
  (height_tet : height_of_tetrahedron h D DH)
  (height_face : segments_in_face h1 h2 A B C F) : 
  h^2 = 4 * h1 * h2 := 
  sorry

-- Theorem 2
theorem symmetry_about_circumcenter (A B C D DH F O : V) 
  [is_regular_tetrahedron A B C D]
  (height_tet : height_of_tetrahedron h D DH)
  (height_face : segments_in_face h1 h2 A B C F)
  (is_circumcenter : is_circumcenter_of_triangle A B C O) :
  symmetric_point O F (projection_onto_normal O) := 
  sorry

end height_squared_eq_four_times_product_of_segments_symmetry_about_circumcenter_l327_327618


namespace number_of_ways_sum_of_primes_l327_327495

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327495


namespace proof_problem_l327_327410

-- Definitions as per the conditions
variables {f : ℝ → ℝ}

-- Conditions
axiom additivity (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom negative_for_positive (x : ℝ) (hx : 0 < x) : f(x) < 0
axiom value_at_one : f(1) = -2

-- Lean 4 statement for proving the problem
theorem proof_problem :
  (∀ x, f(-x) = -f(x)) ∧ (∀ x y : ℝ, x < y → f(x) > f(y)) ∧
  (∀ (x a : ℝ), (f(a * x^2) - 2 * f(x) < f(x) + 4) → (a > 9/8)) :=
by
  sorry

end proof_problem_l327_327410


namespace krystiana_monthly_earnings_l327_327959

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l327_327959


namespace simplify_product_l327_327661

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l327_327661


namespace probability_of_snow_on_most_3_days_l327_327690

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n : ℕ) (kmax : ℕ) (p : ℝ) : ℝ :=
  finset.sum (finset.range (kmax + 1)) (λ k => binomial_probability n k p)

theorem probability_of_snow_on_most_3_days :
  let p := 1/5 in
  let n := 31 in
  abs (cumulative_binomial_probability n 3 p - 0.257) < 0.001 :=
by
  let p := 1/5
  let n := 31
  -- Here we define the cumulative probability up to 3 days
  let approx := cumulative_binomial_probability n 3 p
  -- We assert that the calculated value should be approximately 0.257
  have h : abs (approx - 0.257) < 0.001
  sorry

end probability_of_snow_on_most_3_days_l327_327690


namespace comparison_of_a_b_c_l327_327025

noncomputable def a := Real.sqrt 2
noncomputable def b := Real.logBase π 3
noncomputable def c := -Real.logBase 2 3

theorem comparison_of_a_b_c 
    (ha : a = Real.sqrt 2)
    (hb : b = Real.logBase π 3)
    (hc : c = -Real.logBase 2 3) :
    c < b ∧ b < a := by
  sorry

end comparison_of_a_b_c_l327_327025


namespace equivalent_region_l327_327427

def satisfies_conditions (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 2 ∧ -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1

def region (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≥ -2*x ∧ x^2 + y^2 ≤ 2

theorem equivalent_region (x y : ℝ) :
  satisfies_conditions x y = region x y := 
sorry

end equivalent_region_l327_327427


namespace hexagon_collinear_rational_l327_327146

theorem hexagon_collinear_rational (A B C D E F M N : Point) (r : ℝ) :
  is_regular_hexagon A B C D E F →
  on_diagonal M A C →
  on_segment N C E →
  collinear B M N →
  (dist A M) / (dist A C) = r →
  (dist C N) / (dist C E) = r →
  r = √3 / 3 ∨ r = - √3 / 3 :=
sorry

end hexagon_collinear_rational_l327_327146


namespace vectors_orthogonal_x_value_l327_327319

theorem vectors_orthogonal_x_value :
  (∀ x : ℝ, (3 * x + 4 * (-7) = 0) → (x = 28 / 3)) := 
by 
  sorry

end vectors_orthogonal_x_value_l327_327319


namespace f_decreasing_f_max_value_interval_l327_327407

/-- Definition of the function f -/
def f (x : ℝ) := 2 / (x + 3)

/-- Prove that f is a decreasing function on (-3, +∞) -/
theorem f_decreasing : ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioi (-3) → x₂ ∈ Set.Ioi (-3) → x₁ < x₂ → f x₁ > f x₂ := by
  intros
  sorry

/-- Find the maximum value of f on the interval [-1, 2] -/
theorem f_max_value_interval : ∃ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Icc (-1 : ℝ) 2, f y ≤ f x ∧ f x = 1 := by
  exists (-1)
  sorry

end f_decreasing_f_max_value_interval_l327_327407


namespace circular_route_exists_l327_327107

-- Define the type for cities and roads
structure City :=
(id : ℕ)

structure Road :=
(city1 city2 : City)

-- Define the main parameters: number of cities and roads
def num_cities : ℕ := 1988
def num_roads : ℕ := 4000

-- Properties: City connection and remote city definition
def connects(c : City) (r : Road) : Prop :=
r.city1 = c ∨ r.city2 = c

def remote_city (c : City) (roads : list Road) : Prop :=
(roads.filter (connects c)).length ≤ 2

-- Theorem statement to be proved
theorem circular_route_exists (cities : list City) (roads : list Road)
  (h_cities : cities.length = num_cities)
  (h_roads : roads.length = num_roads)
  (h_no_remote : ∀ c ∈ cities, ¬remote_city c roads) :
  ∃ route : list City, route.length ≤ 20 ∧ 
    (∀ i < route.length - 1, ∃ r ∈ roads, connects (route.nth_le i sorry) r ∧ connects (route.nth_le (i + 1) sorry) r) ∧
    (∃ r ∈ roads, connects (route.nth_le (route.length - 1) sorry) r ∧ connects (route.nth_le 0 sorry) r) :=
sorry

end circular_route_exists_l327_327107


namespace can_eat_fraction_l327_327912

def total_dishes (vegan_dishes : ℕ) (ratio_vegan : ℚ) : ℕ :=
  vegan_dishes * ratio_vegan.den

def vegan_allergy_free_dishes (vegan_dishes gluten_or_dairy_dishes : ℕ) : ℕ :=
  vegan_dishes - gluten_or_dairy_dishes

def fraction_of_menu (allergy_free_dishes total_dishes : ℕ) : ℚ :=
  allergy_free_dishes / total_dishes

theorem can_eat_fraction (vegan_dishes : ℕ) (ratio_vegan : ℚ) (gluten_or_dairy_dishes : ℕ) 
  (h1 : vegan_dishes = 6)
  (h2 : ratio_vegan = 1/6)
  (h3 : gluten_or_dairy_dishes = 4) :
  fraction_of_menu (vegan_allergy_free_dishes vegan_dishes gluten_or_dairy_dishes) 
    (total_dishes vegan_dishes ratio_vegan) = 1/18 := by
  -- conditions
  have h4 : total_dishes 6 (1/6) = 36 := sorry
  have h5 : vegan_allergy_free_dishes 6 4 = 2 := sorry
  have h6 : fraction_of_menu 2 36 = 1/18 := sorry
  -- conclude
  rw [h4, h5, h6]
  -- complete
  sorry

end can_eat_fraction_l327_327912


namespace smallest_n_divisible_11_remainder1_l327_327736

theorem smallest_n_divisible_11_remainder1 :
  ∃ n, (∀ m ∈ {2, 3, 4, 5, 6, 7, 8}, n % m = 1) ∧ (n % 11 = 0) ∧ (n = 6721) :=
by {
  sorry
}

end smallest_n_divisible_11_remainder1_l327_327736


namespace common_ratio_of_geometric_sequence_l327_327130

variable (a : ℕ → ℝ) -- The geometric sequence {a_n}
variable (q : ℝ)     -- The common ratio

-- Conditions
axiom h1 : a 2 = 18
axiom h2 : a 4 = 8

theorem common_ratio_of_geometric_sequence :
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ q^2 = 4/9 → q = 2/3 ∨ q = -2/3 := by
  sorry

end common_ratio_of_geometric_sequence_l327_327130


namespace solution_set_of_inequality_l327_327356

theorem solution_set_of_inequality :
  { x : ℝ | (1/2)^(x^2 - 3*x) > 4 } = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l327_327356


namespace number_of_tables_l327_327183

-- Define the conditions
def chairs : Nat := 7
def time_per_furniture : Nat := 4
def total_time : Nat := 40

-- Define the main proposition to be proven
theorem number_of_tables : ∃ T : Nat, (7 * 4 + T * 4 = 40 ∧ T = 3) := 
by {
  -- We can skip the proof as per the instructions
  use 3,
  simp,
  sorry
}

end number_of_tables_l327_327183


namespace equal_lengths_PE_PF_l327_327231

/--
Two circles \( O_1 \) and \( O_2 \) intersect at points \( A \) and \( B \). The bisector of the 
outer angle \( \angle O_1AO_2 \) intersects \( O_1 \) at \( C \) and \( O_2 \) at \( D \). Point \( P \)
lies on the circumcircle of \( \triangle BCD \). \( CP \) intersects \( O_1 \) again at \( E \) and
\( DP \) intersects \( O_2 \) again at \( F \). Prove that \( PE = PF \).
-/
theorem equal_lengths_PE_PF 
  (O1 O2 : Circle)
  (A B C D P E F : Point)
  (h1 : intersect_circles O1 O2 A B)
  (h2 : angle_bisector_outer_angle O1 A O2 C D)
  (h3 : on_circumcircle P (triangle B C D))
  (h4 : intersection_line_circle (line_through_points C P) O1 E)
  (h5 : intersection_line_circle (line_through_points D P) O2 F) : 
  length_segment P E = length_segment P F := 
sorry

end equal_lengths_PE_PF_l327_327231


namespace quadrilateral_is_rectangle_l327_327800

variables {A B C D : Type*} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]

/-- Four points in a space that form right angles at each vertex determine a rectangle. -/
theorem quadrilateral_is_rectangle
  (h1 : ∠ A B C = π / 2)
  (h2 : ∠ B C D = π / 2)
  (h3 : ∠ C D A = π / 2)
  (h4 : ∠ D A B = π / 2) :
  ∃ (R : set ℝ), quadrilateral A B C D ∧ rectangle A B C D :=
sorry

end quadrilateral_is_rectangle_l327_327800


namespace simplify_expr_l327_327188

theorem simplify_expr (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x :=
by
  sorry

end simplify_expr_l327_327188


namespace area_of_triangle_ABC_l327_327133

noncomputable def angleA : ℝ := 30
noncomputable def angleC : ℝ := 45
noncomputable def side_a : ℝ := 2
noncomputable def area_triangle_ABC : ℝ := √3 + 1

theorem area_of_triangle_ABC : ∃ (S : ℝ), 
  let A := angleA
  let C := angleC
  let a := side_a
  let B := 180 - A - C in
  let sinA := real.sin (A * real.pi / 180)
  let sinC := real.sin (C * real.pi / 180)
  let sinB := real.sin (B * real.pi / 180) in
  let c := a * sinC / sinA in
  S = 1/2 * a * c * sinB ∧ S = area_triangle_ABC :=
begin
  existsi area_triangle_ABC,
  sorry
end

end area_of_triangle_ABC_l327_327133


namespace person_birth_year_and_age_l327_327238

theorem person_birth_year_and_age (x y: ℕ) (h1: x ≤ 9) (h2: y ≤ 9) (hy: y = (88 - 10 * x) / (x + 1)):
  1988 - (1900 + 10 * x + y) = x * y → 1900 + 10 * x + y = 1964 ∧ 1988 - (1900 + 10 * x + y) = 24 :=
by
  sorry

end person_birth_year_and_age_l327_327238


namespace intersection_point_sum_l327_327839

theorem intersection_point_sum (N : ℕ) :
  (∑ N in {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}, N) = 53 :=
by {
  sorry,
}

end intersection_point_sum_l327_327839


namespace interval_of_monotonic_increase_l327_327161

-- Definition of the function f and its conditions
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(2/3) + m * x + 1

-- The main theorem statement
theorem interval_of_monotonic_increase (m : ℝ) (h_even : ∀ x : ℝ, f(m, x) = f(m, -x)) : 
  [0, +∞) = {x | f(0, x) = x^(2/3) + 1} :=
by sorry

end interval_of_monotonic_increase_l327_327161


namespace bounded_sequence_exists_l327_327748

noncomputable def positive_sequence := ℕ → ℝ

variables {a : positive_sequence}

axiom positive_sequence_pos (n : ℕ) : 0 < a n

axiom sequence_condition (k n m l : ℕ) (h : k + n = m + l) : 
  (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)

theorem bounded_sequence_exists 
  (a : positive_sequence) 
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ (k n m l : ℕ), k + n = m + l → 
              (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ (b c : ℝ), (0 < b) ∧ (0 < c) ∧ (∀ n, b ≤ a n ∧ a n ≤ c) :=
sorry

end bounded_sequence_exists_l327_327748


namespace monthly_earnings_is_correct_l327_327960

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l327_327960


namespace minimum_value_of_f_l327_327005

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x + 1 / x) + 1 / (x^2 + 1 / x^2)

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 3) ∧ (f 1 = 3) :=
by
  sorry

end minimum_value_of_f_l327_327005


namespace find_number_l327_327693

theorem find_number (x : ℤ) (h : 38 + 2 * x = 124) : x = 43 :=
begin
  sorry
end

end find_number_l327_327693


namespace first_term_exceeding_1000_l327_327404

variable (a₁ : Int := 2)
variable (d : Int := 3)

def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

theorem first_term_exceeding_1000 :
  ∃ n : Int, n = 334 ∧ arithmetic_sequence n > 1000 := by
  sorry

end first_term_exceeding_1000_l327_327404


namespace trapezoid_angle_l327_327585

theorem trapezoid_angle
  (EFGH : Type)
  (EF GH : EFGH)
  (parallel : ∃ (EF GH : EFGH), EF ∥ GH)
  (angle_E_eq : ∃ (E H : ℝ), E = 3 * H)
  (angle_G_eq : ∃ (G F : ℝ), G = 2 * F)
  (angle_sum : ∃ (F G : ℝ), F + G = 180) :
  ∃ (F : ℝ), F = 60 :=
by
  sorry

end trapezoid_angle_l327_327585


namespace log_equation_solution_l327_327915

theorem log_equation_solution (m y : ℝ) (hm : 0 < m) (hy : 0 < y) (log_equation : log m y * log 3 m = 4) : y = 81 :=
by
  sorry

end log_equation_solution_l327_327915


namespace sum_angles_acute_l327_327937

open Real

theorem sum_angles_acute (A B C : ℝ) (hA_ac : A < π / 2) (hB_ac : B < π / 2) (hC_ac : C < π / 2)
  (h_angle_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end sum_angles_acute_l327_327937


namespace average_production_last_5_days_l327_327746

/-- In a factory, an average of 70 TVs are produced per day for the first 25 days of the month. Afterward, some workers fell ill for the next 5 days, reducing the daily average for the month to 68 sets per day. Prove that the average production per day for the last 5 days is 58 TVs. -/
theorem average_production_last_5_days :
  (∀ d ≤ 25, avg_production d = 70) →
  (∀ d > 25 ∧ d ≤ 30, ill_workers d) →
  (avg_production_month 68) →
  avg_production_last_5_days = 58 :=
by
  sorry

end average_production_last_5_days_l327_327746


namespace ln_inequality_for_positive_integers_l327_327411

-- Define the function f
def f (a x : ℝ) : ℝ := a / (x + 1) + Real.log x

-- The domain constraint (0, +∞) is inherent to the definition, so no need to state separately

-- The statement to be proved
theorem ln_inequality_for_positive_integers (n : ℕ) (h : n > 0) : 
  Real.log (n + 1) > (Finset.sum (Finset.filter (fun k => k % 2 = 1) (Finset.range (2 * n + 2))) (fun k => 1 / k)) := 
sorry

end ln_inequality_for_positive_integers_l327_327411


namespace geometric_sequence_property_l327_327040

theorem geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h : (a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18 + a 19 + a 20) / 10 
      = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18 + a 19 + a 20 + 
         a 21 + a 22 + a 23 + a 24 + a 25 + a 26 + a 27 + a 28 + a 29 + a 30) / 30) :
    (Real.geomMean (List.map_of_fn b 10 11) = Real.geomMean (List.map_of_fn b 30 1)) :=
sorry

end geometric_sequence_property_l327_327040


namespace polynomial_remainder_l327_327740

theorem polynomial_remainder (p : ℚ[X]) :
  (p.eval 1 = 6) → (p.eval 4 = -8) → (∃ (a b : ℚ), ∀ x, (p x) % ((x - 1) * (x - 4)) = (a * x + b) ∧ a = (-14 / 3) ∧ b = (32 / 3)) :=
by {
  sorry
}

end polynomial_remainder_l327_327740


namespace probability_not_first_class_product_l327_327281

open ProbabilityTheory

variable (Ω : Type*) [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)

/-- Condition: Probability of drawing a first-class product is 0.65 --/
def eventA : Type := {A : Set Ω // P A = 0.65}

/-- Condition: Probability of drawing a second-class product is 0.2 --/
def eventB : Type := {B : Set Ω // P B = 0.2}

/-- Condition: Probability of drawing a third-class product is 0.1 --/
def eventC : Type := {C : Set Ω // P C = 0.1}

theorem probability_not_first_class_product (A B C : Set Ω)
  (hA : P A = 0.65) (hB : P B = 0.2) (hC : P C = 0.1) :
  P (B ∪ C) = 0.3 :=
by sorry

end probability_not_first_class_product_l327_327281


namespace polygon_interior_plus_exterior_l327_327217

theorem polygon_interior_plus_exterior (n : ℕ) 
  (h : (n - 2) * 180 + 60 = 1500) : n = 10 :=
sorry

end polygon_interior_plus_exterior_l327_327217


namespace hens_count_l327_327774

theorem hens_count (H C : ℕ) (heads_eq : H + C = 44) (feet_eq : 2 * H + 4 * C = 140) : H = 18 := by
  sorry

end hens_count_l327_327774


namespace no_person_has_fewer_than_6_cards_l327_327917

-- Definition of the problem and conditions
def cards := 60
def people := 10
def cards_per_person := cards / people

-- Lean statement of the proof problem
theorem no_person_has_fewer_than_6_cards
  (cards_dealt : cards = 60)
  (people_count : people = 10)
  (even_distribution : cards % people = 0) :
  ∀ person, person < people → cards_per_person = 6 ∧ person < people → person = 0 := 
by 
  sorry

end no_person_has_fewer_than_6_cards_l327_327917


namespace johns_total_profit_l327_327952

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end johns_total_profit_l327_327952


namespace four_digit_numbers_divisible_by_5_count_l327_327079

theorem four_digit_numbers_divisible_by_5_count : 
  let a := 1000 in
  let l := 9995 in
  let d := 5 in
  ∃ n, (l = a + (n-1) * d) ∧ n = 1800 :=
by
  existsi (1800 : ℕ)
  simp
  sorry

end four_digit_numbers_divisible_by_5_count_l327_327079


namespace num_ordered_pairs_l327_327354

theorem num_ordered_pairs (N : ℕ) :
  (N = 20) ↔ ∃ (a b : ℕ), 
  (a < b) ∧ (100 ≤ a ∧ a ≤ 1000)
  ∧ (100 ≤ b ∧ b ≤ 1000)
  ∧ (gcd a b * lcm a b = 495 * gcd a b)
  := 
sorry

end num_ordered_pairs_l327_327354


namespace correct_number_of_propositions_is_two_l327_327294

-- Proposition Definitions
def proposition_1 := ∀ (l : Line) (p : Plane), (l ∈ p) → ¬(exists (P : Point), (P ∉ p ∧ P ∈ l))
def proposition_2 := ∀ (p₁ p₂ : Plane), p₁ ≠ p₂ → ¬(exists (P : Point), (P ∈ p₁ ∧ P ∈ p₂) ∧ ∀ (L : Line), ¬(P ∈ L ∧ L ∈ p₁ ∧ L ∈ p₂))
def proposition_3 := ∀ (l₁ l₂ : Line) (p₁ p₂ : Plane), (l₁ ∈ p₁) ∧ (l₂ ∈ p₂) ∧ (p₁ ∩ p₂ ≠ ∅) → (∃ (P : Point), P ∈ l₁ ∧ P ∈ l₂) → exists (L : Line), (P ∈ L ∧ L ∈ p₁ ∧ L ∈ p₂)
def proposition_4 := ∀ (t : Triangle) (l : Line), ∃ (A B : Point), (A ∈ t.sides ∧ B ∈ t.sides ∧ A ∈ l ∧ B ∈ l) → (l ∈ t.plane)

-- Number of True Propositions
def number_of_true_propositions : Nat :=
  [proposition_1, proposition_2, proposition_3, proposition_4].count (λprop, prop = true)

-- Statement
theorem correct_number_of_propositions_is_two : number_of_true_propositions = 2 :=
by {
  sorry,
}

end correct_number_of_propositions_is_two_l327_327294


namespace problem1_problem2_problem3_l327_327864

-- Problem 1: b_m = m for a_n = n^2 and f(m) = m^2
theorem problem1 (m : ℕ) (hm : m > 0) :
  (∀ n, n^2 ≤ m^2 → b_m = m) -> b_1 = 1 ∧ b_2 = 2 ∧ b_3 = 3 :=
sorry

-- Problem 2: Sum of the first m terms S_m for a_n = 2n and f(m) = m
theorem problem2 (m : ℕ) (hm : m > 0) :
  (∀ n, 2 * n ≤ m → ∑ k in finset.range m, b_k = 
    if m % 2 = 1 then (m^2 - 1) / 4 else m^2 / 4) :=
sorry

-- Problem 3: Values of d and A for a_n = 2^n, f(m) = Am^3, b_3 = 10
theorem problem3 {A d : ℕ} (A_pos : A > 0) (d_pos : d > 0) (b_3 : ℕ) (hb3 : b_3 = 10) :
  (∀ n, 2^n ≤ A * m^3 → b_m = 3 * m + log2 A) → d = 3 ∧ (A = 64 ∨ A = 65) :=
sorry

end problem1_problem2_problem3_l327_327864


namespace range_of_m_l327_327874

theorem range_of_m (a b m : ℝ) (h1 : b = 2 * a) (h2 : b = a ^ 2) (h3 : 0 < Real.log (a * b) / Real.log m) (h4 : Real.log (a * b) / Real.log m < 1) : 
  m > 8 := 
by 
  have ha : a = 2 := 
    by 
      sorry -- This follows from the system of equations h1 and h2
  have hb : b = 4 := by 
    sorry -- This follows from ha and h2
  have hab : a * b = 8 := 
    by 
      sorry -- Simplification with ha and hb
  have hlog : ab = 8 := by 
    sorry -- Direct computation
  sorry -- Show m > 8 using the log inequalities h3 and h4 with hab = 8

end range_of_m_l327_327874


namespace product_of_fractions_l327_327237

theorem product_of_fractions :
  (1 / 2) * (3 / 5) * (5 / 6) = 1 / 4 := 
by
  sorry

end product_of_fractions_l327_327237


namespace cost_of_other_parts_l327_327265

theorem cost_of_other_parts (total_cost : ℕ) (total_parts : ℕ) (fixed_cost_parts : ℕ) (fixed_cost : ℕ) 
    (check_amount : ℕ := 2380) (total_parts := 59) (fixed_cost_parts := 40) (fixed_cost := 50) : 
    (total_cost = check_amount - (fixed_cost_parts * fixed_cost)) ∧
    (total_parts = 59) ∧
    (fixed_cost_parts = 40) →
    (total_cost / (total_parts - fixed_cost_parts) = 20) :=
by
  intro h
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [← h1, ← h2, ← h3],
  have h_total_parts : total_parts - fixed_cost_parts = 19 := rfl,
  have h_total_cost : 2380 - 2000 = 380 := rfl,
  rw [h_total_parts, h_total_cost],
  norm_num,
  sorry

end cost_of_other_parts_l327_327265


namespace problem_statement_l327_327176

theorem problem_statement : 
  let n1 := 1 + |(-10: ℝ)| in
  let n2 := -2 - 1 in
  n1 + n2 = 8 := 
by
  let n1 := 1 + |(-10: ℝ)| 
  let n2 := -2 - 1
  show n1 + n2 = 8
  sorry

end problem_statement_l327_327176


namespace janet_total_distance_l327_327949

-- Define the distances covered in each week for each activity
def week1_running := 8 * 5
def week1_cycling := 7 * 3

def week2_running := 10 * 4
def week2_swimming := 2 * 2

def week3_running := 6 * 5
def week3_hiking := 3 * 2

-- Total distances for each activity
def total_running := week1_running + week2_running + week3_running
def total_cycling := week1_cycling
def total_swimming := week2_swimming
def total_hiking := week3_hiking

-- Total distance covered
def total_distance := total_running + total_cycling + total_swimming + total_hiking

-- Prove that the total distance is 141 miles
theorem janet_total_distance : total_distance = 141 := by
  sorry

end janet_total_distance_l327_327949


namespace number_of_prime_pairs_for_10003_l327_327520

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327520


namespace cost_of_each_skirt_l327_327830

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l327_327830


namespace polygon_with_45_deg_exterior_angle_is_eight_gon_l327_327452

theorem polygon_with_45_deg_exterior_angle_is_eight_gon
  (each_exterior_angle : ℝ) (h1 : each_exterior_angle = 45) 
  (sum_exterior_angles : ℝ) (h2 : sum_exterior_angles = 360) :
  ∃ (n : ℕ), n = 8 :=
by
  sorry

end polygon_with_45_deg_exterior_angle_is_eight_gon_l327_327452


namespace a_n_general_term_b_n_general_term_c_n_prefix_sum_l327_327886

-- Definitions of the sequences and their properties
def S (n : ℕ) : ℝ := 2 - 1 / 2^(n-1)
def a (n : ℕ) : ℝ := 1 / 2^(n-1)
def b (n : ℕ) : ℝ := 2 * n - 1
def c (n : ℕ) : ℝ := b n / a n
def T (n : ℕ) : ℝ := (2 * n - 3) * 2^n + 3

-- Conditions
axiom S_condition : ∀ n : ℕ, S n = 2 - 1 / 2^(n-1)
axiom arithmetic_seq_condition : a 1 = b 1 ∧ a 2 * (b 2 - b 1) = a 1

-- Proof statements
theorem a_n_general_term (n : ℕ) : a n = 1 / 2^(n-1) :=
sorry

theorem b_n_general_term (n : ℕ) : b n = 2 * n - 1 :=
sorry

theorem c_n_prefix_sum (n : ℕ) : ∑ i in Finset.range n, c (i+1) = T n :=
sorry

end a_n_general_term_b_n_general_term_c_n_prefix_sum_l327_327886


namespace parallelogram_square_center_quadrilateral_square_l327_327718

-- Define an abstract type for points
variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]

-- Define the terms and conditions
def is_parallelogram (A B C D : Point) : Prop := 
  (B - A = D - C) ∧ (C - B = D - A)

def is_square (S : set Point) : Prop := 
  ∃ (A B C D : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
  (B - A) = (C - B) ∧ (D - C) = (A - D) ∧ 
  (∥B - A∥ = ∥C - B∥) ∧ (∥C - B∥ = ∥D - C∥) ∧ (inner (B - A) (C - B) = 0)

def square_center (A B : Point) := (A + B) / 2

theorem parallelogram_square_center_quadrilateral_square
    (A B C D E F G H I J K L P Q R S : Point) 
    (h_parallelogram : is_parallelogram A B C D)
    (h_square1 : is_square ({A, B, E, F}))
    (h_square2 : is_square ({B, C, G, H}))
    (h_square3 : is_square ({C, D, I, J}))
    (h_square4 : is_square ({D, A, K, L}))
    (h_center1 : P = square_center E F)
    (h_center2 : Q = square_center G H)
    (h_center3 : R = square_center I J)
    (h_center4 : S = square_center K L) :
  is_square ({P, Q, R, S}) :=
sorry

end parallelogram_square_center_quadrilateral_square_l327_327718


namespace first_term_and_common_difference_l327_327036

theorem first_term_and_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) :
  a 1 = 1 ∧ (a 2 - a 1) = 4 :=
by
  sorry

end first_term_and_common_difference_l327_327036


namespace counting_numbers_remainder_7_div_61_l327_327077

def divides (a b : Nat) : Prop := ∃ k, b = k * a

theorem counting_numbers_remainder_7_div_61 : 
    {n : Nat | divides n 54 ∧ n > 7}.card = 4 := 
by
  sorry

end counting_numbers_remainder_7_div_61_l327_327077


namespace no_solution_iff_k_nonnegative_l327_327059

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

theorem no_solution_iff_k_nonnegative (k : ℝ) :
  (¬ ∃ x : ℝ, f k (f k x) = 3 / 2) ↔ k ≥ 0 :=
  sorry

end no_solution_iff_k_nonnegative_l327_327059


namespace water_left_in_bathtub_l327_327946

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end water_left_in_bathtub_l327_327946


namespace cube_octahedron_surface_area_ratio_l327_327768

theorem cube_octahedron_surface_area_ratio (a : ℝ) (h_pos : 0 < a) :
  let cube_surface_area := 6 * a^2 in
  let octahedron_side_length := a * real.sqrt 2 in
  let octahedron_surface_area := 2 * real.sqrt 3 * (octahedron_side_length)^2 in
  cube_surface_area / octahedron_surface_area = real.sqrt 3 / 2 :=
by sorry

end cube_octahedron_surface_area_ratio_l327_327768


namespace difference_of_squares_l327_327901

theorem difference_of_squares (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end difference_of_squares_l327_327901


namespace number_of_ways_sum_of_primes_l327_327497

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327497


namespace petya_cannot_form_figure_four_l327_327648

-- Definitions based on the problem conditions
def rhombus_colored_half_white_half_gray : Prop := true
def rotation_angles : list ℕ := [0, 90, 180, 270]
def not_flipped (r: rhombus_colored_half_white_half_gray) : Prop := true

-- Auxiliary definitions to handle combinations and configurations
def can_form (desired_figure : ℕ) (shape: rhombus_colored_half_white_half_gray) : Prop :=
  -- This is a placeholder for the actual logic to check the formation of a figure
  sorry

-- The actual theorem to be proved
theorem petya_cannot_form_figure_four :
  ∀ (figure: ℕ), figure = 4 → ¬ can_form figure rhombus_colored_half_white_half_gray
:= by
  -- This is a placeholder for the proof.
  intros figure hfigure
  cases hfigure
  apply sorry

end petya_cannot_form_figure_four_l327_327648


namespace _l327_327274

noncomputable theorem mini_sphere_radius_ratio {V_1 V_2 : ℝ} (h1 : V_1 = 512 * real.pi) (h2 : V_2 = V_1 / 4) : 
  ∃ r1 r2 : ℝ, (4/3) * real.pi * r1^3 = V_1 ∧ (4/3) * real.pi * r2^3 = V_2 ∧ (r2 / r1) = 1 / 2 * (real.sqrt 2)^(2 / 3) :=
by {
  sorry
}

end _l327_327274


namespace triangle_inequality_for_f_l327_327852

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3 * x + m

theorem triangle_inequality_for_f (a b c m : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) 
(h₆ : 6 < m) :
  ∃ u v w, u = f a m ∧ v = f b m ∧ w = f c m ∧ u + v > w ∧ u + w > v ∧ v + w > u := 
sorry

end triangle_inequality_for_f_l327_327852


namespace fraction_of_ponies_with_horseshoes_l327_327282

theorem fraction_of_ponies_with_horseshoes 
  (P H : ℕ) 
  (h1 : H = P + 4) 
  (h2 : H + P ≥ 164) 
  (x : ℚ)
  (h3 : ∃ (n : ℕ), n = (5 / 8) * (x * P)) :
  x = 1 / 10 := by
  sorry

end fraction_of_ponies_with_horseshoes_l327_327282


namespace monotonicity_f_l327_327822

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 3) / x

theorem monotonicity_f : 
  (∀ x y : ℝ, x < y ∧ x ∈ set.Iio 0 ∧ y ∈ set.Iio 0 → f x < f y) ∧ 
  (∀ x y : ℝ, x < y ∧ x ∈ set.Ioi 0 ∧ y ∈ set.Ioi 0 → f x < f y) := 
by
  sorry

end monotonicity_f_l327_327822


namespace number_of_permutations_satisfying_condition_l327_327616

open Fintype

def is_permutation (l : List ℕ) : Prop :=
  l ~ List.range' 1 21

def abs_difference_condition (l : List ℕ) : Prop :=
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j < 21 → |l.nthLe (j - 1) (by linarith) - l.nthLe 20 (by decide)| 
  ≥ |l.nthLe (i - 1) (by linarith) - l.nthLe 20 (by decide)|

theorem number_of_permutations_satisfying_condition : 
  ∃ l : List ℕ, is_permutation l ∧ abs_difference_condition l → list.permutations (List.range' 1 21).length = 3070 := 
sorry

end number_of_permutations_satisfying_condition_l327_327616


namespace tangent_line_at_zero_maximum_k_l327_327890

section
variable (x k : ℝ)

def f (x : ℝ) (k : ℝ) := (k - x) * exp(x) - x - 3

theorem tangent_line_at_zero (h : k = 1) : 
  let f_x := f x k in
  ∃ y : ℝ, y = f 0 1 ∧ y = -2 ∧ -exp(x)*x-1 = -1 ∧ (x + y + 2 = 0) :=
  sorry

theorem maximum_k (h : ∀ x > 0, f x k < 0) : 
  k ≤ 2 :=
  sorry
end

end tangent_line_at_zero_maximum_k_l327_327890


namespace find_height_of_cuboid_l327_327350

-- Define the cuboid structure and its surface area formula
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

def surface_area (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Given conditions
def given_cuboid : Cuboid := { length := 12, width := 14, height := 7 }
def given_surface_area : ℝ := 700

-- The theorem to prove
theorem find_height_of_cuboid :
  surface_area given_cuboid = given_surface_area :=
by
  sorry

end find_height_of_cuboid_l327_327350


namespace range_of_lambda_l327_327907

theorem range_of_lambda :
  ∀ (λ : ℝ), (∃ a b : ℝ × ℝ, a = (1, -2) ∧ b = (2, λ) ∧ dot_product a b > 0) ↔ (λ ∈ set.Ioo (-∞) (-4) ∪ set.Ioo (-4) 1) :=
by
  sorry

end range_of_lambda_l327_327907


namespace value_of_each_best_buy_gift_card_l327_327135

theorem value_of_each_best_buy_gift_card
  (B : ℝ) -- value of each Best Buy gift card
  (h1 : 6 * B + 9 * 200) -- total value of gift cards Jack was asked to send
  (h2 : B + 2 * 200) -- value of gift cards Jack actually sent
  (h3 : 3900) -- remaining value of gift cards Jack can return
  : B = 500 :=
by {
  -- Let the remaining value formula
  have h4 : (6 * B + 9 * 200) - (B + 2 * 200) = 3900,
    from sorry,
  calc
    -- Simplify the equation to get the value of B
    6 * B + 1800 - (B + 400) = 3900 : by sorry
    5 * B + 1400 = 3900         : by sorry
    5 * B = 2500                : by sorry
    B = 500                     : by sorry
}

end value_of_each_best_buy_gift_card_l327_327135


namespace GCF_LCM_calculation_l327_327720

theorem GCF_LCM_calculation : 
  GCD (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end GCF_LCM_calculation_l327_327720


namespace number_of_ways_sum_of_primes_l327_327500

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327500


namespace hexagon_area_l327_327590

open Real

def Triangle := {A B C : Type} -- Assuming we have a type 'Type' for points of the triangle
def Segment (A B : Type) := {length : ℝ} -- Segment between two points with some length
def Hexagon := {A B C A' B' C' : Type} -- Hexagon with 6 points 

noncomputable def area_of_hexagon (ABC : Triangle) (p : Segment A B) (perimeter : ℝ) (radius : ℝ) : ℝ := sorry

theorem hexagon_area
  (ABC : Triangle)
  (p : Segment A B := {length := 10})
  (perimeter := 30)
  (radius := 6) :
  area_of_hexagon ABC p perimeter radius = 90 := by
  sorry

end hexagon_area_l327_327590


namespace polynomial_root_theorem_l327_327651

theorem polynomial_root_theorem
  (α β γ δ p q : ℝ)
  (h₁ : α + β = -p)
  (h₂ : α * β = 1)
  (h₃ : γ + δ = -q)
  (h₄ : γ * δ = 1) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
by
  sorry

end polynomial_root_theorem_l327_327651


namespace negation_of_p_negation_of_q_l327_327613

def p (x : ℝ) : Prop := x > 0 → x^2 - 5 * x ≥ -25 / 4

def even (n : ℕ) : Prop := ∃ k, n = 2 * k

def q : Prop := ∃ n, even n ∧ ∃ m, n = 3 * m

theorem negation_of_p : ¬(∀ x : ℝ, x > 0 → x^2 - 5 * x ≥ - 25 / 4) → ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x < - 25 / 4 := 
by sorry

theorem negation_of_q : ¬ (∃ n : ℕ, even n ∧ ∃ m : ℕ, n = 3 * m) → ∀ n : ℕ, even n → ¬ (∃ m : ℕ, n = 3 * m) := 
by sorry

end negation_of_p_negation_of_q_l327_327613


namespace sin_theta_plus_pi_over_six_l327_327019

theorem sin_theta_plus_pi_over_six (θ : ℝ) (h : sin θ + sin (θ + π / 3) = 1) : sin (θ + π / 6) = sqrt 3 / 3 :=
by
  sorry

end sin_theta_plus_pi_over_six_l327_327019


namespace rectangle_ratio_l327_327011

theorem rectangle_ratio (s y x : ℝ) 
  (inner_square_area outer_square_area : ℝ) 
  (h1 : inner_square_area = s^2)
  (h2 : outer_square_area = 9 * inner_square_area)
  (h3 : outer_square_area = (3 * s)^2)
  (h4 : s + 2 * y = 3 * s)
  (h5 : x + y = 3 * s)
  : x / y = 2 := 
by
  -- Proof steps will go here
  sorry

end rectangle_ratio_l327_327011


namespace a_5_eq_61_l327_327418

namespace ProofProblem

def a_seq : ℕ → ℤ
| 1     := 1
| (n+1) := 2 * a_seq n + 3

theorem a_5_eq_61 : a_seq 5 = 61 := 
by
  sorry

end ProofProblem

end a_5_eq_61_l327_327418


namespace problem1_problem2_l327_327565

-- Definitions based on the conditions:
def line (t α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

def curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (2, Real.sqrt 3)

def segmentLength {α : ℝ} (α_eq : α = Real.pi / 3) : ℝ :=
  let t1 := -56 / 13  -- First root of 13t^2 + 56t + 48 = 0
  let t2 := 12 / 13   -- Second root of 13t^2 + 56t + 48 = 0
  let A := line t1 α
  let B := line t2 α
  distance A B

def verifyPA_PB_eq_OP_sq (m : ℝ) (m_eq : m = Real.sqrt 5 / 4) : Prop :=
  let α := Real.atan (Real.sqrt 5 / 4)
  let t1 := ...  -- Expression for root t1
  let t2 := ...  -- Expression for root t2
  let A := line t1 α
  let B := line t2 α
  (distance P A) * (distance P B) = (distance O P) ^ 2

-- Lean 4 statements to prove
theorem problem1 (α := Real.pi / 3) : segmentLength (by rfl) = 8 * Real.sqrt 10 / 13 := sorry

theorem problem2 (m := Real.sqrt 5 / 4) : verifyPA_PB_eq_OP_sq m (by rfl) := sorry

end problem1_problem2_l327_327565


namespace num_correct_statements_is_2_l327_327687

-- Define the statements
def statement1 : Prop := ∀ (a b c d : ℕ) (α : ℝ), (a = b ∧ c = d ∧ α = 60) → false
def statement2 : Prop := ∀ (a b c : ℕ), (a = b ∧ c = a) → false  -- ambiguity
def statement3 : Prop := ∀ (a b c d e f : ℕ), (a = d ∧ b = e ∧ c = f) → (a + b + c = d + e + f)
def statement4 : Prop := ∀ (a b c : ℕ) (α : ℝ), (α = 60) → false
def statement5 : Prop := ∀ (a b c : ℕ), (a = 5 ∧ b = 12 ∧ c = 13) → (a^2 + b^2 = c^2)

-- Assert the statement correctness
def correct_statements : ℕ := (if statement3 then 1 else 0) + (if statement5 then 1 else 0)

-- Prove that the number of correct statements is 2
theorem num_correct_statements_is_2 : correct_statements = 2 := by
  sorry

end num_correct_statements_is_2_l327_327687


namespace math_score_estimate_l327_327766

noncomputable def normal_distribution (μ σ : ℝ) (X : ℝ) : Prop := sorry

variables (n : ℕ) (μ σ : ℝ) (P : ℝ → ℝ → ℝ)

theorem math_score_estimate :
  n = 50 →
  μ = 110 →
  σ = 10 →
  P(100, μ) = 0.34 →
  (∃ k, k = P(100, μ) ∧ P(120, μ) = k) →
  50 * 0.16 = 8 :=
by
  assume (n_eq : n = 50) (μ_eq : μ = 110) (σ_eq : σ = 10)
      (P_eq : P(100, μ) = 0.34) 
      (k_exists : ∃ k, k = P(100, μ) ∧ P(120, μ) = k),
  -- proof steps
  sorry

end math_score_estimate_l327_327766


namespace range_ln_inv_abs_plus_one_l327_327213

theorem range_ln_inv_abs_plus_one : 
  (∀ (x : ℝ), ∃ (y : ℝ), y = ln (1 / (|x| + 1)) → y ≤ 0) :=
by
  sorry

end range_ln_inv_abs_plus_one_l327_327213


namespace students_scoring_130_or_higher_l327_327092

noncomputable def math_exam_students_scoring_130_or_higher : ℕ :=
  let μ := 120
  let σ := 10  -- Since variance is 100, standard deviation σ = sqrt(100) = 10
  let total_students := 40
  -- The given condition to use
  let p_value := 0.1587
  -- Multiply probability by the total number of students
  let number_of_students := total_students * p_value 
  -- Round to the nearest whole number
  in round number_of_students

theorem students_scoring_130_or_higher (μ σ : ℝ) (total_students : ℕ) : 
  μ = 120 ∧ σ = 10 ∧ total_students = 40 → math_exam_students_scoring_130_or_higher = 6 :=
by
  intro h
  rw [math_exam_students_scoring_130_or_higher]
  have μ : ℝ := 120
  have σ : ℝ := 10
  have total_students : ℕ := 40
  have p_value : ℝ := 0.1587
  have number_of_students := total_students * p_value
  finish_round number_of_students = 6
  sorry

end students_scoring_130_or_higher_l327_327092


namespace complete_square_l327_327818

theorem complete_square (x : ℝ) :
    (x^2 - 6 * x + 4 = 0) ↔ ((x - 3)^2 = 5) :=
by
  have h : x^2 - 6 * x + 4 = (x - 3)^2 - 5 := 
    by ring
  rw sub_eq_iff_eq_add.mp h
  exact sorry

end complete_square_l327_327818


namespace sin_A_in_right_triangle_l327_327564

-- Definitions and conditions directly from the problem
variables {A B C : ℝ}  -- angles in argument
variables {a b c : ℝ}  -- sides opposite to A, B, and C respectively
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def tan (A : ℝ) : ℝ := sin A / cos A

theorem sin_A_in_right_triangle
  (h : is_right_triangle a b c)
  (h1 : B = π / 2)
  (h2 : 3 * (tan A) = 4) :
  sin A = 4 / 5 :=
by
  sorry  -- Proof steps are not required as per the instructions

end sin_A_in_right_triangle_l327_327564


namespace no_prime_sum_10003_l327_327509

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327509


namespace find_alpha_plus_beta_l327_327376

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α) (h2 : β < π / 2) 
  (h3 : Real.cos α = (sqrt 5) / 5) (h4 : Real.sin β = (3 * sqrt 10) / 10) : 
  α + β = (3 * π) / 4 := 
sorry

end find_alpha_plus_beta_l327_327376


namespace chord_with_midpoint_l327_327032

-- Define the ellipse and conditions with given point P inside the ellipse
def ellipse (x y: ℝ) : Prop := (x^2 / 16) + (y^2 / 4) = 1

-- Define point P and midpoint property
def P : ℝ × ℝ := (1, 1)

-- Define endpoints A and B such that P is the midpoint
def midpoint (A B: ℝ × ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  (x1 + x2) = 2 * (fst P) ∧ (y1 + y2) = 2 * (snd P)

-- Define the final equation of the chord
def chord_eq (x y: ℝ) : Prop := x + 4 * y - 5 = 0

-- Prove the chord equation given the conditions
theorem chord_with_midpoint : 
  ∀ (A B: ℝ × ℝ), midpoint A B ∧ (ellipse (fst A) (snd A)) ∧ (ellipse (fst B) (snd B)) → chord_eq (fst P) (snd P) :=
by
  intros A B H
  sorry -- Proof omitted

end chord_with_midpoint_l327_327032


namespace no_prime_sum_10003_l327_327464

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327464


namespace length_of_bridge_correct_l327_327789

-- Definition of the conditions
def length_of_train : ℝ := 110
def speed_of_train_kmh : ℝ := 72
def time_to_cross_bridge : ℝ := 12.199024078073753

-- Compute the speed of the train in m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

-- Compute the total distance covered
def total_distance : ℝ := speed_of_train_ms * time_to_cross_bridge

-- The length of the bridge is the total distance minus the length of the train
def length_of_bridge : ℝ := total_distance - length_of_train

-- The theorem statement
theorem length_of_bridge_correct : length_of_bridge = 133.98048156147506 :=
by
  -- Skipping the proof
  sorry

end length_of_bridge_correct_l327_327789


namespace restore_original_multiplication_l327_327762

theorem restore_original_multiplication (A B C D E F G H I: ℕ) 
  (h1 : A ∈ {1, 3, 5, 7, 9}) -- O digit
  (h2 : B ∈ {0, 2, 4, 6, 8}) -- E digit
  (h3 : C ∈ {0, 2, 4, 6, 8}) -- E digit
  (h4 : D ∈ {0, 2, 4, 6, 8}) -- E digit
  (h5 : E ∈ {1, 3, 5, 7, 9}) -- O digit
  (h6 : F ∈ {0, 2, 4, 6, 8}) -- E digit
  (h7 : G ∈ {0, 2, 4, 6, 8}) -- E digit
  (h8 : H ∈ {0, 2, 4, 6, 8}) -- E digit
  (h9 : I ∈ {0, 2, 4, 6, 8}) -- E digit
  (h10 : (100*A + 10*B + C) * (10*D + E) = 1000*F + 100*G + 10*H + I) 
  (h11 : (100*A + 10*B + C) = 346) 
  (h12 : (10*D + E) = 28)
  : (100*A + 10*B + C) * (10*D + E) = 9688 := 
by 
  rw [h11, h12]
  norm_num


end restore_original_multiplication_l327_327762


namespace road_trip_cost_l327_327362

theorem road_trip_cost 
  (x : ℝ)
  (initial_cost_per_person: ℝ) 
  (redistributed_cost_per_person: ℝ)
  (cost_difference: ℝ) :
  initial_cost_per_person = x / 4 →
  redistributed_cost_per_person = x / 7 →
  cost_difference = 8 →
  initial_cost_per_person - redistributed_cost_per_person = cost_difference →
  x = 74.67 :=
by
  intro h1 h2 h3 h4
  -- starting the proof
  rw [h1, h2] at h4
  sorry

end road_trip_cost_l327_327362


namespace general_formula_sum_sn_l327_327865

-- Definitions for the problem conditions.
def T (n : ℕ) := 2^((n * (n - 1)) / 2)

def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else T n / T (n - 1)

-- Definitions for the sum S_n as specified in the problem.
def S (n : ℕ) := (List.range (n + 3)).sumBy (λ i, if i = n + 1 then -(n + 1) * a (n + 3) else (i + 1) * a (i + 1)) - 1

-- Theorems to be proven.
theorem general_formula (n : ℕ) : a n = 2^(n - 1) := sorry

theorem sum_sn (n : ℕ) : S n = 0 := sorry

end general_formula_sum_sn_l327_327865


namespace total_footprints_l327_327739

def pogo_footprints_per_meter : ℕ := 4
def grimzi_footprints_per_6_meters : ℕ := 3
def distance_traveled_meters : ℕ := 6000

theorem total_footprints : (pogo_footprints_per_meter * distance_traveled_meters) + (grimzi_footprints_per_6_meters * (distance_traveled_meters / 6)) = 27000 :=
by
  sorry

end total_footprints_l327_327739


namespace correct_difference_l327_327738

theorem correct_difference 
  (d_wrong_units : ℕ)
  (d_wrong_tens : ℕ)
  (s_wrong_hundreds : ℕ)
  (erroneous_difference : ℕ) :
  d_wrong_units = 5 ∧ 
  d_wrong_tens = 0 ∧ 
  s_wrong_hundreds = 2 ∧ 
  erroneous_difference = 1994 →
  let d_correct_units := d_wrong_units - 2
  let d_correct_tens := d_wrong_tens + 60
  let s_correct_hundreds := s_wrong_hundreds - 500
  (erroneous_difference + s_correct_hundreds + d_correct_tens + d_correct_units) = 1552 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  let d_correct_units := d_wrong_units - 2
  let d_correct_tens := d_wrong_tens + 60
  let s_correct_hundreds := s_wrong_hundreds - 500
  calc 
    erroneous_difference + s_correct_hundreds + d_correct_tens + d_correct_units
    = 1994 + (-500) + 60 + (-2) : by sorry
    = 1552 : by sorry

end correct_difference_l327_327738


namespace problem_equivalent_proof_l327_327399

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def is_arithmetic_sequence (b : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

noncomputable def S (b : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range (2*n + 1), b i

theorem problem_equivalent_proof (a b : ℕ → ℝ)
  (h1 : is_geometric_sequence a)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3)
  (h4 : is_arithmetic_sequence b)
  (h5 : ∀ n : ℕ, S b (n + 1) = b n * b (n + 1)) :
  (∀ n : ℕ, a n = 2^n) ∧ (∀ n : ℕ, ∑ i in Finset.range n, (b i) / (a i) = 5 - (2 * n + 5) / (2^n)) :=
sorry

end problem_equivalent_proof_l327_327399


namespace inequality_proof_l327_327989

def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def a : ℝ := log_base 0.2 0.3
noncomputable def b : ℝ := log_base 2 0.3

theorem inequality_proof : a * b < a + b ∧ a + b < 0 :=
by
  sorry

end inequality_proof_l327_327989


namespace trailing_zeros_base_12_of_33_mul_59_l327_327207

-- We are generating a problem to determine that the number of trailing zeros of 
-- 33 * 59 in base 12 is exactly 4.
theorem trailing_zeros_base_12_of_33_mul_59 :
  let n := 33 * 59
  (nat.trailing_digits n 12) = 4 :=
by
  -- The proof steps are omitted and using sorry for now
  sorry

end trailing_zeros_base_12_of_33_mul_59_l327_327207


namespace find_m_n_sum_l327_327979

theorem find_m_n_sum : 
  ∃ (m n : ℕ), q = m / n ∧ Nat.coprime m n ∧ m + n = 11 :=
begin
  let p_0 := 4 / 7,
  let q := p_0,
  use [4, 7],
  split,
  { exact rfl, },
  split,
  { apply Nat.coprime.intro,
    exact Nat.succ_pos',
    { by_contradiction,
      exact zero_ne_one (eq_zero_of_mod_eq_zero 2 7) },
    exact Nat.gcd_one_right },
  exact rfl,
end

end find_m_n_sum_l327_327979


namespace find_m_l327_327972

variable (A B C D : Type)
variable (is_regular_tetrahedron : regular_tetrahedron A B C D)
variable (edge_length : ∀ (x y : A B C D), x ≠ y → distance x y = 1)
variable (Q : ℕ → ℚ)
variable (initial_condition : Q 0 = 1)
variable (recurrence : ∀ n, Q (n + 1) = (1 / 3) * (1 - Q n))

theorem find_m :
  Q 8 = 547 / 2187 :=
sorry

end find_m_l327_327972


namespace num_elements_divides_l327_327604

open Nat

variable (n : ℕ)
variable (M A B : Set ℕ)
variable (hM : M = {n^3, n^3 + 1, ..., n^3 + n})
variable (hA : A ⊆ M ∧ A ≠ ∅)
variable (hB : B ⊆ M ∧ B ≠ ∅)
variable (h_disjoint : A ∩ B = ∅)
variable (h_sum_div : (∑ a in A, a) ∣ (∑ b in B, b))

theorem num_elements_divides (hM : M = {n^3, n^3 + 1, ..., n^3 + n}) : 
  A ⊆ M ∧ A ≠ ∅ → 
  B ⊆ M ∧ B ≠ ∅ →
  A ∩ B = ∅ →
  (∑ a in A, a) ∣ (∑ b in B, b) →
  (A.card ∣ B.card) :=
sorry

end num_elements_divides_l327_327604


namespace line_AP_bisects_CD_l327_327157

theorem line_AP_bisects_CD (A B C D E P : Type)
  [linear_order A] [linear_order B] [linear_order C] [linear_order D] [linear_order E] [linear_order P]
  (convex_ABCDE : convex_pentagon A B C D E)
  (angles_eq1 : ∠BAC = ∠CAD)
  (angles_eq2 : ∠CAD = ∠DAE)
  (angles_eq3 : ∠CBA = ∠DCA)
  (angles_eq4 : ∠DCA = ∠EDA)
  (P_def : P = BD ∩ CE)
  (M_def : M = line_intersection (AP) (CD))
  (Gamma1_tangent_to_CD : tangent_to CD Gamma1)
  (Gamma2_tangent_to_CD : tangent_to CD Gamma2) :
  midpoint M C D :=
sorry

end line_AP_bisects_CD_l327_327157


namespace min_value_of_M_l327_327870

theorem min_value_of_M (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_sum : a + b + c + d = 1) : 
  ∃ (M : ℝ), M = 3 ∧ M = sqrt(a^2 + 1 / (8 * a)) + sqrt(b^2 + 1 / (8 * b)) + sqrt(c^2 + 1 / (8 * c)) + sqrt(d^2 + 1 / (8 * d)) := 
sorry

end min_value_of_M_l327_327870


namespace B1M_expression_l327_327940

-- Define vectors and their properties as given in the conditions
variables (a b c : ℝ^3)
variable (BC_mid_M : ∀ (BC M : ℝ^3), M = (BC / 2))
variable (A1B1_eq_a : ℝ^3)
variable (A1C1_eq_b : ℝ^3)
variable (A1A_eq_c : ℝ^3)

-- Lean 4 statement for the proof problem
theorem B1M_expression :
  ∃ (B1 M : ℝ^3), B1M = (A1A_eq_c + 1/2*(A1C1_eq_b - A1B1_eq_a)) :=
sorry

end B1M_expression_l327_327940


namespace count_simple_pairs_sum_1492_l327_327775

-- Definitions
def isSimplePair (m n : ℕ) : Prop :=
  let m_digits := Nat.digits 10 m
  let n_digits := Nat.digits 10 n
  (List.length m_digits = List.length n_digits) ∧
  (∀ i, i < List.length m_digits → (m_digits.get i + n_digits.get i) < 10)

-- The statement to prove
theorem count_simple_pairs_sum_1492 : 
  {p : ℕ × ℕ // isSimplePair p.1 p.2 ∧ p.1 + p.2 = 1492}.card = 300 :=
sorry

end count_simple_pairs_sum_1492_l327_327775


namespace difference_is_correct_l327_327206

-- Definitions for the conditions
def number : ℝ := 0.127
def fraction : ℝ := 1 / 8

-- Declare the theorem to be proven
theorem difference_is_correct :
  number - fraction = 0.0020000000000000018 :=
sorry

end difference_is_correct_l327_327206


namespace sum_of_two_primes_unique_l327_327487

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327487


namespace max_non_square_product_set_size_l327_327857

theorem max_non_square_product_set_size :
  ∃ A ⊆ (finset.range 25).map (λ n, n + 1), 
    (∀ a b ∈ A, a ≠ b → ¬ is_square (a * b)) ∧ A.card = 16 ∧ 
    { T : finset ℕ | T ⊆ (finset.range 25).map (λ n, n + 1) ∧ 
     (∀ a b ∈ T, a ≠ b → ¬ is_square (a * b)) ∧ T.card = 16 }.card = 120 := 
sorry

end max_non_square_product_set_size_l327_327857


namespace f_double_a_l327_327368

def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem f_double_a (a : ℝ) (h : f a = 3) : f (2 * a) = 7 :=
by
  sorry

end f_double_a_l327_327368


namespace unique_five_digit_number_l327_327434

theorem unique_five_digit_number (n : ℕ) (h_digits : nat.digits 10 n = [a, a, a, a, a]) (h_sum : a * 5 = 45) : n = 99999 :=
sorry

end unique_five_digit_number_l327_327434


namespace find_d_l327_327449

-- Definitions of the conditions
variables (r s t u d : ℤ)

-- Assume r, s, t, and u are positive integers
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom t_pos : t > 0
axiom u_pos : u > 0

-- Given conditions
axiom h1 : r ^ 5 = s ^ 4
axiom h2 : t ^ 3 = u ^ 2
axiom h3 : t - r = 19
axiom h4 : d = u - s

-- Proof statement
theorem find_d : d = 757 :=
by sorry

end find_d_l327_327449


namespace sum_of_two_primes_unique_l327_327485

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327485


namespace derivative_at_zero_l327_327853

def f (x : ℝ) : ℝ := x^3

theorem derivative_at_zero : deriv f 0 = 0 :=
by
  sorry

end derivative_at_zero_l327_327853


namespace find_x_l327_327094
-- Lean 4 equivalent problem setup

-- Assuming a and b are the tens and units digits respectively.
def number (a b : ℕ) := 10 * a + b
def interchangedNumber (a b : ℕ) := 10 * b + a
def digitsDifference (a b : ℕ) := a - b

-- Given conditions
variable (a b k : ℕ)

def condition1 := number a b = k * digitsDifference a b
def condition2 (x : ℕ) := interchangedNumber a b = x * digitsDifference a b

-- Theorem to prove
theorem find_x (h1 : condition1 a b k) : ∃ x, condition2 a b x ∧ x = k - 9 := 
by sorry

end find_x_l327_327094


namespace triangle_one_solution_l327_327881

open Real

theorem triangle_one_solution (k : ℝ) : 
  (0 < k ∧ k ≤ 12 ∨ k = 8 * sqrt 3) →
  ∃! △ABC : Triangle, 
    angle △ABC B C = 60 * (π / 180) ∧
    side △ABC A C = 12 ∧
    side △ABC B C = k :=
begin
  sorry -- Proof omitted
end

end triangle_one_solution_l327_327881


namespace no_prime_sum_10003_l327_327470

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327470


namespace ellipse_equation_standard_l327_327402

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  (a = 2 * c ∧ a - c = sqrt 3 ∧ b * b = a * a - c * c) →
  (  ∀ x y : ℝ, (x^2) / (a * a) + (y^2) / (b * b) = 1
  ∨ (y^2) / (a * a) + (x^2) / (b * b) = 1 )

theorem ellipse_equation_standard :
  ∃ a b c : ℝ, ellipse_standard_equation a b c := by
  sorry

end ellipse_equation_standard_l327_327402


namespace probability_more_2s_than_5s_l327_327090

theorem probability_more_2s_than_5s (rolls : Fin 5 → Fin 6) : 
  (probability (∑ i, if rolls i = 1 then 1 else 0 > ∑ i, if rolls i = 4 then 1 else 0 )) = 223 / 324 := 
by sorry

end probability_more_2s_than_5s_l327_327090


namespace geom_seq_properties_l327_327873

variable (a_n : ℕ → ℝ)
variable (q : ℝ) (S_n : ℕ → ℝ)
variable (a1 : a_n 1) (a2a4 : a_n 2 * a_n 4)
variable (n : ℕ)

def geom_sequence (a1 : ℝ) (q : ℝ) (n : ℕ) : Prop :=
  a_n n = a1 * q ^ (n - 1)

def sum_geom_sequence (a1 : ℝ) (q : ℝ) (n : ℕ) : Prop :=
  S_n n = a1 * (q ^ n - 1) / (q - 1)

theorem geom_seq_properties
  (h1 : a_n 1 = 3)
  (h2 : a_n 2 * a_n 4 = 144)
  (h3 : q > 0) :
  q = 2 ∧ S_n 10 = 3069 := by
  sorry

end geom_seq_properties_l327_327873


namespace point_M_coordinates_l327_327650

/- Define the conditions -/

def isInFourthQuadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distanceToXAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.2 = d

def distanceToYAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.1 = d

/- Write the Lean theorem statement -/

theorem point_M_coordinates :
  ∀ (M : ℝ × ℝ), isInFourthQuadrant M ∧ distanceToXAxis M 3 ∧ distanceToYAxis M 4 → M = (4, -3) :=
by
  intro M
  sorry

end point_M_coordinates_l327_327650


namespace no_prime_sum_10003_l327_327508

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327508


namespace circumcenter_BCD_on_circumcircle_ABC_l327_327153

variables {S1 S2 : Type*} [circle S1] [circle S2]
variable {B : S1}
variable {A : tangent_line S1 B}
variable {C : Type*}
variable (AC : line A C)
variable {D : point_on_circle_outside_line S2 A C}

theorem circumcenter_BCD_on_circumcircle_ABC
  (h1 : point_on_circle C S1)
  (h2 : intersects_line_segment AC S1)
  (h3 : tangent_circle_point S2 AC C)
  (h4 : tangent_circle_point S2 S1 D)
  (h5 : opposite_sides_line D B AC) :
  exists K,  circumcenter_of_triangle K B C D ∧ on_circumcircle K A B C := by
  sorry

end circumcenter_BCD_on_circumcircle_ABC_l327_327153


namespace sum_of_numbers_l327_327737

theorem sum_of_numbers :
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := 
  by
  sorry

end sum_of_numbers_l327_327737


namespace pizza_slice_division_l327_327707

theorem pizza_slice_division : 
  ∀ (num_coworkers num_pizzas slices_per_pizza : ℕ),
  num_coworkers = 12 →
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  (num_pizzas * slices_per_pizza) / num_coworkers = 2 := 
by
  intros num_coworkers num_pizzas slices_per_pizza h_coworkers h_pizzas h_slices
  rw [h_coworkers, h_pizzas, h_slices]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slice_division_l327_327707


namespace intersection_M_N_l327_327422

-- Definition of set M
def M := {x : ℝ | abs (x - 1) < 1}

-- Definition of set N
def N := {x : ℝ | ∃ y : ℝ, y = log 2 (x^2 + 2*x + 3)}

-- The theorem stating the intersection
theorem intersection_M_N : 
  (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l327_327422


namespace unique_sum_of_two_primes_l327_327563

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327563


namespace correct_option_C_l327_327387

noncomputable theory

def parabola (x : ℝ) : ℝ :=
  (x - 1)^2 - 2

theorem correct_option_C (a b c d : ℝ)
  (ha : a < 0) 
  (hb : b > 0)
  (hc : c = a ∨ c = b ∨ (a < c ∧ c < b))
  (hd : d < 1)
  (hA : parabola a = 2)
  (hB : parabola b = 6)
  (hC : parabola c = d) :
  a < c ∧ c < b := 
sorry

end correct_option_C_l327_327387


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327479

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327479


namespace increasing_sequence_range_l327_327415

theorem increasing_sequence_range (a : ℝ) (f : ℝ → ℝ) (a_n : ℕ+ → ℝ) :
  (∀ n : ℕ+, a_n n = f n) →
  (∀ n m : ℕ+, n < m → a_n n < a_n m) →
  (∀ x : ℝ, f x = if  x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6) ) →
  2 < a ∧ a < 3 :=
by
  sorry

end increasing_sequence_range_l327_327415


namespace algebraic_expression_value_l327_327121

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m * n = 3) 
  (h2 : n = m + 1) : 
  (m - n) ^ 2 * ((1 / n) - (1 / m)) = -1 / 3 :=
by sorry

end algebraic_expression_value_l327_327121


namespace math_problem_l327_327293

theorem math_problem : 
  ( ("If (x + y = 0), then (x are opposite y)".converse) ∧
    (¬ ("Congruent triangles have equal areas")) ∧
    ("If (q ≤ 1), then (x^2 + 2x + q = 0 has real roots)".converse) ∧
    ( ("If (a > b), then (ac^2 > bc^2)".contrapositive) ) = false ) :=
  by sorry

end math_problem_l327_327293


namespace game_ends_after_12_rounds_l327_327113

variable (tokensA tokensB tokensC : ℕ)
variable (rounds : ℕ)

def gameEndsAfterRounds : Prop :=
  tokensA = 14 ∧ tokensB = 13 ∧ tokensC = 12 ∧
  (∀ n, (tokensA, tokensB, tokensC) -- Before nth round state
       -> (tokensA', tokensB', tokensC') -- After nth round state
       -> tokensA' ≥ 0 ∧ tokensB' ≥ 0 ∧ tokensC' ≥ 0) ∧
  (tokensA = 0 ∨ tokensB = 0 ∨ tokensC = 0) -- End condition after 'rounds' moves

theorem game_ends_after_12_rounds :
  tokensA = 14 → tokensB = 13 → tokensC = 12 → gameEndsAfterRounds tokensA tokensB tokensC 12 :=
by {
  intros,
  sorry -- Proof skipped
}

end game_ends_after_12_rounds_l327_327113


namespace jill_shopping_expenses_l327_327633

theorem jill_shopping_expenses (T : ℝ) (h1 : 0 < T)
  (h_clothing : 0.6 * T ≥ 0)
  (h_food : 0.1 * T ≥ 0)
  (h_tax_clothing : 0.04 * 0.6 * T ≥ 0)
  (h_tax_food : 0)
  (h_tax_other : 0.08 * (1 - 0.6 - 0.1) * T)
  (h_total_tax : 0.048 * T = 0.024 * T + 0.08 * ((1 - 0.6 - 0.1) * T)) : 
  (1 - 0.6 - 0.1) = 0.3 :=
  sorry

end jill_shopping_expenses_l327_327633


namespace find_interest_rate_l327_327834

-- Definitions
def SI := 8625
def P := 68800
def T := 0.75
def correct_R := 16.71

-- Statement to be proven
theorem find_interest_rate :
  (SI * 100) / (P * T) = correct_R :=
sorry

end find_interest_rate_l327_327834


namespace number_of_ways_sum_of_primes_l327_327503

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327503


namespace kirin_calculations_l327_327432

theorem kirin_calculations (calculations_per_second : ℝ) (seconds : ℝ) (h1 : calculations_per_second = 10^10) (h2 : seconds = 2022) : 
    calculations_per_second * seconds = 2.022 * 10^13 := 
by
  sorry

end kirin_calculations_l327_327432


namespace sequence_sum_is_65_l327_327807

-- Define the sequence
def sequence_term (n : ℕ) : ℕ := n * (1 - 1 / n)

-- Define the sum of the sequence from n = 3 to n = 12
def sequence_sum : ℕ := (∑ n in Finset.range (12 + 1), if 2 < n ∧ n ≤ 12 then sequence_term n else 0)

-- State the proof problem
theorem sequence_sum_is_65 : sequence_sum = 65 :=
by
  sorry

end sequence_sum_is_65_l327_327807


namespace max_n_is_seven_l327_327220

-- Define the conditions
def distinct_real_numbers (n : ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → (i < n) → (j < n) → card_val i ≠ card_val j

def divided_into_two_piles (card_val : ℕ → ℝ) (pile1 pile2 : List ℕ) : Prop :=
  (pile1.length > 0 ∧ pile2.length > 0) ∧
  ∀ x ∈ pile1, ∀ y ∈ pile2, 
  (card_val x = - card_val y) ∨ 
  (pile1.sum card_val + pile2.sum card_val = 0)

-- Statement of the theorem
theorem max_n_is_seven (n : ℕ) (card_val : ℕ → ℝ) :
  (2 ≤ n) →
  distinct_real_numbers n →
  ∃ pile1 pile2, divided_into_two_piles card_val pile1 pile2 →
  n ≤ 7 :=
by
  sorry

end max_n_is_seven_l327_327220


namespace problem_remainder_1000th_in_S_mod_1000_l327_327607

def S : Nat → Prop :=
  λ n => Nat.bitcount n = 8

def N_1000th := Nat.find (Nat.gt_wf (λ n, S n))

theorem problem_remainder_1000th_in_S_mod_1000 :
  let N := N_1000th 1000
  N % 1000 = 32 :=
by
  sorry

end problem_remainder_1000th_in_S_mod_1000_l327_327607


namespace b_spends_85_percent_of_salary_l327_327756

def combined_salary : ℝ := 7000
def a_salary : ℝ := 5250
def b_salary : ℝ := combined_salary - a_salary
def a_spending_percentage : ℝ := 0.95
def a_savings_percentage : ℝ := 1 - a_spending_percentage
def a_savings : ℝ := a_savings_percentage * a_salary
def b_savings : ℝ := a_savings
def b_spending_percentage : ℝ := 1 - b_savings / b_salary

theorem b_spends_85_percent_of_salary : b_spending_percentage = 0.85 :=
by
  have h1 : b_salary = 7000 - a_salary := rfl
  have h2 : a_savings = 0.05 * 5250 := (by ring)
  have h3 : a_savings = b_savings := rfl
  have h4 : b_savings / b_salary = 262.50 / 1750 := sorry
  have h5 : b_spending_percentage = 1 - (262.50 / 1750) := rfl
  have h6 : 1 - (262.50 / 1750) = 0.85 := sorry
  exact eq.trans h5 h6

end b_spends_85_percent_of_salary_l327_327756


namespace equation_of_circle_equation_of_line_l_l327_327858

-- Definition of the center line
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Definition of point A
def point_A (x y : ℝ) := x = 2 ∧ y = -1

-- Definition of the tangent line
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Definition of internal point P
def point_P (x y : ℝ) := x = 1/2 ∧ y = -3

-- Definition of the circle C
def circle (x y a r : ℝ) := (x - a)^2 + (y + 2*a)^2 = r^2

-- Tangency condition
def tangent_condition (x y a r : ℝ) := |a - 2*a - 1| / real.sqrt 2 = r

-- Midpoint condition for chord AB
def midpoint_condition (x y : ℝ → ℝ → Prop) := ∀ (x1 y1 x2 y2 : ℝ), 
  x (x1, x2) ∧ y (y1, y2) ∧ (x1 + x2) / 2 = 1/2 ∧ (y1 + y2) / 2 = -3

-- Statement for (I)
theorem equation_of_circle : ∃ (a r : ℝ), circle 2 (-1) a r ∧ center_line a (-2*a) ∧ tangent_condition 1 (-1) a r :=
begin
  use [1, real.sqrt 2],
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

-- Statement for (II)
theorem equation_of_line_l : ∃ (k b : ℝ), ∀ x y, midpoint_condition (λ x x1 => x = x1) (λ y y1 => y = y1) → 
  y + 3 = k * (x - 1 / 2) :=
begin
  use [-1/2, 1/2],
  { sorry }
end

end equation_of_circle_equation_of_line_l_l327_327858


namespace last_three_digits_7_pow_105_l327_327351

theorem last_three_digits_7_pow_105 : (7^105) % 1000 = 783 :=
  sorry

end last_three_digits_7_pow_105_l327_327351


namespace sum_of_two_primes_unique_l327_327490

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327490


namespace log_inequalities_l327_327240

theorem log_inequalities :
  ∃ (x y z : ℝ), (real.log x / real.log 3 < real.log y / real.log 2) ∧ (real.log y / real.log 2 < real.log z / real.log 2) :=
by
  use [2, 3, 5]
  sorry

end log_inequalities_l327_327240


namespace min_value_of_expression_l327_327152

theorem min_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 6) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 :=
by
  sorry

end min_value_of_expression_l327_327152


namespace bottle_ratio_l327_327644

theorem bottle_ratio (C1 C2 : ℝ)  
  (h1 : (C1 / 2) + (C2 / 4) = (C1 + C2) / 3) :
  C2 = 2 * C1 :=
sorry

end bottle_ratio_l327_327644


namespace no_prime_sum_10003_l327_327466

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327466


namespace smallest_number_leaving_remainder_1_and_divisible_by_11_l327_327734

theorem smallest_number_leaving_remainder_1_and_divisible_by_11 :
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {2, 3, 4, 5, 6, 7, 8}, n % m = 1) ∧ n = 6721 :=
by
  sorry

end smallest_number_leaving_remainder_1_and_divisible_by_11_l327_327734


namespace cost_of_pizza_l327_327139

variable (P : ℝ)
variable (num_pizzas_park : ℝ := 3)
variable (num_pizzas_building : ℝ := 2)
variable (distance_park : ℝ := 0.1) -- in km
variable (distance_building : ℝ := 2) -- in km
variable (delivery_charge : ℝ := 2)
variable (total_paid : ℝ := 64)

theorem cost_of_pizza : 
  total_paid = num_pizzas_park * P + num_pizzas_building * P + if distance_building > 1 then delivery_charge else 0 → 
  P = 12.4 := 
by 
  sorry

end cost_of_pizza_l327_327139


namespace teaching_experiment_evaluation_l327_327700

-- Definition of the test score ranges and data from the tables
def pre_test_data_class_A := [28, 9, 9, 3, 1]
def pre_test_data_class_B := [25, 10, 8, 2, 1]
def post_test_data_class_A := [14, 16, 12, 6, 2]
def post_test_data_class_B := [6, 8, 11, 18, 3]

-- Calculate the number of students
def num_students_class_A_pre := pre_test_data_class_A.sum
def num_students_class_B_pre := pre_test_data_class_B.sum

-- Mean calculation helper function
def mean (data : List ℕ) (total_students : ℕ) (weights : List ℝ) : ℝ :=
  (data.zip weights).map (λ (x, w), x * w).sum / total_students

-- Weights for mean calculation based on midpoints of score ranges
def weights := [2.5, 7.5, 12.5, 17.5, 22.5]

-- Calculate means for post-test
def mean_class_A_post := mean post_test_data_class_A num_students_class_A_pre weights
def mean_class_B_post := mean post_test_data_class_B num_students_class_B_pre weights

-- Conditions as hypotheses
theorem teaching_experiment_evaluation :
  num_students_class_A_pre = 50 ∧
  num_students_class_B_pre = 46 ∧
  mean_class_A_post = 9.1 ∧
  mean_class_B_post ≈ 12.9 → -- Approximate equality for real number comparison
  ∃ more_effective : Bool, more_effective := true :=
by sorry

end teaching_experiment_evaluation_l327_327700


namespace arithmetic_geometric_seq_common_ratio_one_l327_327287

theorem arithmetic_geometric_seq_common_ratio_one
  (a : ℕ → ℝ)
  (d q : ℝ)
  (h_arith : ∀ n ≥ 1, a (n + 1) - a n = d)
  (h_geom : ∀ n ≥ 1, a (n + 1) = q * a n) :
  q = 1 :=
by
  let n := 2
  have h_common := calc
    (a (n) - a (n - 1)) = d      : h_arith (n - 1) (by norm_num)
    _ = (q - 1) * a (n - 1) : by rw [h_geom (n - 1) (by norm_num), sub_mul, sub_self, zero_mul, add_mul]
   
  sorry

end arithmetic_geometric_seq_common_ratio_one_l327_327287


namespace sqrt_eight_plus_n_eq_nine_l327_327444

theorem sqrt_eight_plus_n_eq_nine (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := by
  sorry

end sqrt_eight_plus_n_eq_nine_l327_327444


namespace binomial_coefficient_x2_l327_327050

theorem binomial_coefficient_x2 (n : ℕ) (h : 2 ^ n = 2 * 64) :
  (binomial n 1 * (-2) * 1 + binomial n 2 * (-2)^2) = 70 :=
by
  sorry

end binomial_coefficient_x2_l327_327050


namespace unique_sum_of_two_primes_l327_327560

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327560


namespace weight_of_new_person_l327_327250

theorem weight_of_new_person (W : ℝ) (average_increase : ℝ) (weight_old : ℝ) (n : ℕ) :
  average_increase = 4.2 → weight_old = 65 → n = 8 → W = weight_old + n * average_increase →
  W = 98.6 :=
by
  intros h_avg h_old h_n h_W
  rw [h_avg, h_old, h_n] at h_W
  assumption

end weight_of_new_person_l327_327250


namespace non_adjacent_choices_12_5_l327_327999

theorem non_adjacent_choices_12_5 :
  ∃ (n : ℕ), n = (Nat.choose 8 5) ∧ n = 56 :=
by
  use Nat.choose 8 5
  split
  . rfl
  . sorry

end non_adjacent_choices_12_5_l327_327999


namespace abs_neg_2023_l327_327670

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l327_327670


namespace weights_problem_l327_327243

theorem weights_problem :
  let weights := {1, 3, 9, 27}
  in max_weighable weights = 40 ∧ count_weighable_combinations weights = 40 :=
by
  -- Definitions
  let weights := {1, 3, 9, 27}

  -- Variables in Statement
  def max_weighable (weights : Set ℕ) : ℕ :=
    Finset.sum weights.toFinset id

  def can_weigh (weights : Set ℕ) (x : ℕ) : Prop :=
    ∃ s : Finset ℕ, (∀ w ∈ s, w ∈ weights) ∧ s.sum id = x

  def count_weighable_combinations (weights : Set ℕ) : ℕ :=
    (Finset.range (max_weighable weights + 1)).filter (can_weigh weights).card

  -- Main Proof (not provided, just a placeholder)
  sorry

end weights_problem_l327_327243


namespace max_norm_sum_leq_224_l327_327164

variables (u v w : ℝ^n)
variables (hu : ∥u∥ = 3) (hv : ∥v∥ = 1) (hw : ∥w∥ = 2)

theorem max_norm_sum_leq_224 : 
  ∥u - 3 • v∥^2 + ∥v - 3 • w∥^2 + ∥w - 3 • u∥^2 ≤ 224 :=
sorry

end max_norm_sum_leq_224_l327_327164


namespace fuel_tank_capacity_l327_327772

theorem fuel_tank_capacity (x : ℝ) 
  (h1 : (5 / 6) * x - (2 / 3) * x = 15) : x = 90 :=
sorry

end fuel_tank_capacity_l327_327772


namespace arithmetic_sequence_terms_l327_327295

-- Define the hypothesis and necessary variables for the arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ → ℕ
| k := a + k * d

-- Assume odd numbered terms and their sum
def sum_odds (a d : ℕ) (n : ℕ) : ℕ :=
∑ i in finset.range (n // 2 + 1), arithmetic_sequence a d (2 * i)

-- Assume even numbered terms and their sum
def sum_evens (a d : ℕ) (n : ℕ) : ℕ :=
∑ i in finset.range (n // 2), arithmetic_sequence a d (2 * i + 1)

-- Given conditions
def condition_sum_odds (a d : ℕ) (n : ℕ) : Prop :=
  sum_odds a d n = 72

def condition_sum_evens (a d : ℕ) (n : ℕ) : Prop :=
  sum_evens a d n = 66

-- Main statement to prove
theorem arithmetic_sequence_terms (a d : ℕ) (h_sum_odds : condition_sum_odds a d 23) (h_sum_evens : condition_sum_evens a d 23) : 
  ∃ n, (∑ i in finset.range (n // 2 + 1), arithmetic_sequence a d (2 * i)) = 72 ∧ 
       (∑ i in finset.range (n // 2), arithmetic_sequence a d (2 * i + 1)) = 66 ∧ 
       n = 23 :=
sorry

end arithmetic_sequence_terms_l327_327295


namespace compare_magnitudes_l327_327982

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_magnitudes : B < C ∧ C < A ∧ A < D :=
by
  sorry

end compare_magnitudes_l327_327982


namespace gcf_of_lcm_9_15_and_10_21_is_5_l327_327729

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l327_327729


namespace acetone_molecular_weight_l327_327806

theorem acetone_molecular_weight :
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  let num_C := 3
  let num_H := 6
  let num_O := 1
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O ≈ 58.078) :=
by
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  let num_C := 3
  let num_H := 6
  let num_O := 1
  have h : num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O = 58.078 := sorry
  exact h

end acetone_molecular_weight_l327_327806


namespace number_of_different_flags_l327_327820

theorem number_of_different_flags : 
  let colors := 3 in
  let stripes := 3 in
  (colors ^ stripes) = 27 :=
by
  sorry

end number_of_different_flags_l327_327820


namespace correct_option_C_l327_327386

noncomputable theory

def parabola (x : ℝ) : ℝ :=
  (x - 1)^2 - 2

theorem correct_option_C (a b c d : ℝ)
  (ha : a < 0) 
  (hb : b > 0)
  (hc : c = a ∨ c = b ∨ (a < c ∧ c < b))
  (hd : d < 1)
  (hA : parabola a = 2)
  (hB : parabola b = 6)
  (hC : parabola c = d) :
  a < c ∧ c < b := 
sorry

end correct_option_C_l327_327386


namespace polynomial_sum_l327_327987

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 7 * x - 6
def h (x : ℝ) : ℝ := 3 * x^2 - 3 * x + 2
def j (x : ℝ) : ℝ := x^2 + x - 1

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end polynomial_sum_l327_327987


namespace emily_can_buy_12_cucumbers_l327_327455

-- Define the equivalence of cost between apples, bananas, and cucumbers
variable (A B C : Type) -- A for apples, B for bananas, C for cucumbers

-- Conditions provided
variable (cost_A cost_B cost_C : ℕ)
variable (h1 : 6 * cost_A = 3 * cost_B)
variable (h2 : 3 * cost_B = 4 * cost_C)

-- Question and answer as a proof statement
theorem emily_can_buy_12_cucumbers : 18 * cost_A = 12 * cost_C :=
by
  -- Declare variables and conditions within the proof context
  have mid_eq_bananas: 18 * cost_A = 9 * cost_B := by sorry
  have mid_eq_cucumbers: 9 * cost_B = 12 * cost_C := by sorry
  exact trans mid_eq_bananas mid_eq_cucumbers

end emily_can_buy_12_cucumbers_l327_327455


namespace tangent_line_at_2_tangent_line_through_origin_l327_327412

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_2 :
  let t : ℝ → ℝ := λ x, 13 * x - 32 in
  ∀ x, (f(2) = -6) → (t 2 = f 2) → ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < abs h ∧ abs h < δ → abs ((f (2 + h) - f 2) / h - 13) < ε :=
by sorry

theorem tangent_line_through_origin :
  let t : ℝ → ℝ := λ x, 13 * x in
  ∃ x_0 : ℝ, x_0 = -2 ∧ f x_0 = -26 ∧ (∀ x, t x = (f' x_0) * (x - x_0) + f x_0) ∧ t 0 = 0 :=
by sorry

end tangent_line_at_2_tangent_line_through_origin_l327_327412


namespace mushroom_problem_l327_327014

variables (x1 x2 x3 x4 : ℕ)

theorem mushroom_problem
  (h1 : x1 + x2 = 6)
  (h2 : x1 + x3 = 7)
  (h3 : x2 + x3 = 9)
  (h4 : x2 + x4 = 11)
  (h5 : x3 + x4 = 12)
  (h6 : x1 + x4 = 9) :
  x1 = 2 ∧ x2 = 4 ∧ x3 = 5 ∧ x4 = 7 := 
  by
    sorry

end mushroom_problem_l327_327014


namespace complement_intersection_in_U_l327_327993

universe u

variables {α : Type u} (U A B : Set α)

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_in_U : U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_intersection_in_U_l327_327993


namespace f_at_3_l327_327884

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end f_at_3_l327_327884


namespace problem_a_l327_327247

theorem problem_a (x : ℝ) (n : ℕ) : 
  x^(2 * n) - 1 = (x^2 - 1) * ∏ k in finset.range (n - 1) + 1, (x^2 - 2 * x * real.cos (k * real.pi / n) + 1) :=
sorry

end problem_a_l327_327247


namespace product_of_w_and_x_l327_327447

variables {w x y : ℝ}

theorem product_of_w_and_x :
  (2 / w + 2 / x = 2 / y) → 
  (w * x = y) → 
  ((w + x) / 2 = 0.5) → 
  (w * x = 0) :=
begin
  intros h1 h2 h3,
  sorry
end

end product_of_w_and_x_l327_327447


namespace monthly_earnings_is_correct_l327_327962

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l327_327962


namespace cost_of_each_skirt_l327_327829

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l327_327829


namespace domain_of_function_l327_327315

noncomputable def function_domain := {x : ℝ | x * (3 - x) ≥ 0 ∧ x - 1 ≥ 0 }

theorem domain_of_function: function_domain = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end domain_of_function_l327_327315


namespace volume_ratio_remainder_520_l327_327780

noncomputable def simplex_ratio_mod : Nat :=
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000

theorem volume_ratio_remainder_520 :
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000 = 520 :=
by 
  sorry

end volume_ratio_remainder_520_l327_327780


namespace one_point_zero_seven_five_billion_in_scientific_notation_l327_327102

-- Given number
def billion := 10^9
def number := 1.075 * billion

-- Conditions in Lean 4
def is_valid_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * (10 ^ n) ∧ (1 ≤ a) ∧ (a < 10)

-- Proof goal
theorem one_point_zero_seven_five_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_valid_scientific_notation a n number ∧ a = 1.075 ∧ n = 9 :=
by {
  -- The details of the proof would go here, but are omitted as specified.
  sorry,
}

end one_point_zero_seven_five_billion_in_scientific_notation_l327_327102


namespace exists_integers_for_prime_l327_327179

theorem exists_integers_for_prime (p : ℕ) (hp : Nat.Prime p) : 
  ∃ x y z w : ℤ, x^2 + y^2 + z^2 = w * p ∧ 0 < w ∧ w < p :=
by 
  sorry

end exists_integers_for_prime_l327_327179


namespace DC_perp_CO2_l327_327938

-- Definitions of points and circles as per the problem conditions
variables {O_1 O_2 A B C D E : Point}
variables {circle1 : Circle} 
variables {circle2 : Circle}

-- Given Conditions
axiom circles_intersect : circle1.center = O_1 ∧ circle2.center = O_2 ∧ circle1 ≠ circle2 ∧ intersects_circle circle1 circle2 = {A, B}
axiom extension_O1A_meets_O2 : meets_extension O_1 A O_2 C circle2
axiom extension_O2A_meets_O1 : meets_extension O_2 A O_1 D circle1
axiom BE_parallel_O2A : parallel BE (line_through O_2 A) ∧ meets_circle BE E circle1 ∧ E ≠ B
axiom DE_parallel_O1A : parallel DE (line_through O_1 A)

-- Proof statement
theorem DC_perp_CO2 : perpendicular (line_through D C) (line_through C O_2) :=
sorry

end DC_perp_CO2_l327_327938


namespace find_ellipse_equation_l327_327038

noncomputable def ellipse (x y : ℝ) (a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def point (x y : ℝ) := (x, y)
def focus_left (a b c : ℝ) := (-c, 0)
def directrix_point (a c : ℝ) := (-a^2 / c, 0)
def perpendicular_condition (p f q : ℝ × ℝ) := (p.1 - f.1) * (q.1 - f.1) + (p.2 - f.2) * (q.2 - f.2) = 0
def segment_ratio (p n q : ℝ × ℝ) (r s : ℝ) := |n.1 - p.1 + n.2 - p.2| * s = |q.1 - n.1 + q.2 - n.2| * r

theorem find_ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a > b) :
  (∀ a b, (∃ (F Q P N : ℝ × ℝ),
    F = focus_left a b c ∧
    Q = directrix_point a c ∧
    P = point 0 3 ∧
    perpendicular_condition P F Q ∧
    segment_ratio P N Q 1 8)) →
  (ellipse x y 3 (sqrt 8)) :=
sorry

end find_ellipse_equation_l327_327038


namespace max_expression_bound_l327_327001

theorem max_expression_bound (a b : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b) : 
  let expr := (|7 * a + 8 * b - a * b| + |2 * a + 8 * b - 6 * a * b|) / (a * real.sqrt (1 + b^2))
  in expr ≤ 9 * real.sqrt 2 :=
  sorry

end max_expression_bound_l327_327001


namespace digits_product_problem_l327_327638

noncomputable theory
open_locale nat

theorem digits_product_problem : 
  ∃ (X Y Z O : ℕ), X ≠ Y ∧ X ≠ Z ∧ X ≠ O ∧ Y ≠ Z ∧ Y ≠ O ∧ Z ≠ O ∧ 
  1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9 ∧ 1 ≤ O ∧ O ≤ 9 ∧ 
  (let n₁ := X * 100 + Y * 10 + Z,
       n₂ := Y * 100 + X * 10 + Z in
  (n₁ * n₂ = 169201 ∨ n₁ * n₂ = 193501) ∧ 
  ((let prod := n₁ * n₂ in 
  (prod / 100000) = (prod % 10) ∧ 
  (prod / 10000) % 10 ≠ (prod / 1000) % 10 ∧ 
  (prod / 10000) % 10 ≠ (prod / 100) % 10 ∧ 
  (prod / 10000) % 10 ≠ (prod / 10) % 10 ∧ 
  (prod / 1000) % 10 ≠ (prod / 100) % 10 ∧ 
  (prod / 1000) % 10 ≠ (prod / 10) % 10 ∧ 
  (prod / 100) % 10 ≠ (prod / 10) % 10))) :=
begin
  sorry
end

end digits_product_problem_l327_327638


namespace number_of_divisors_of_54_greater_than_7_l327_327076

theorem number_of_divisors_of_54_greater_than_7 : 
  (set.filter (λ d, 7 < d) {d | d ∣ 54}).finite.to_finset.card = 4 := 
by
  sorry

end number_of_divisors_of_54_greater_than_7_l327_327076


namespace sqrt_eight_plus_n_eq_nine_l327_327442

theorem sqrt_eight_plus_n_eq_nine (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := by
  sorry

end sqrt_eight_plus_n_eq_nine_l327_327442


namespace num_consecutive_sets_summing_to_90_l327_327208

-- Define the arithmetic sequence sum properties
theorem num_consecutive_sets_summing_to_90 : 
  ∃ n : ℕ, n ≥ 2 ∧
    ∃ (a : ℕ), 2 * a + n - 1 = 180 / n ∧
      (∃ k : ℕ, 
         k ≥ 2 ∧
         ∃ b : ℕ, 2 * b + k - 1 = 180 / k) ∧
      (∃ m : ℕ, 
         m ≥ 2 ∧ 
         ∃ c : ℕ, 2 * c + m - 1 = 180 / m) ∧
      (n = 3 ∨ n = 5 ∨ n = 9) :=
sorry

end num_consecutive_sets_summing_to_90_l327_327208


namespace slices_per_person_l327_327703

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l327_327703


namespace gcf_of_lcm_eq_15_l327_327725

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_of_lcm_eq_15 : gcf (lcm 9 15) (lcm 10 21) = 15 := by
  sorry

end gcf_of_lcm_eq_15_l327_327725


namespace find_a_l327_327409

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 - x) - Real.log (1 + x) + a

theorem find_a 
  (M : ℝ) (N : ℝ) (a : ℝ)
  (h1 : M = f a (-1/2))
  (h2 : N = f a (1/2))
  (h3 : M + N = 1) :
  a = 1 / 2 := 
sorry

end find_a_l327_327409


namespace sum_powers_of_i_l327_327813

theorem sum_powers_of_i :
  let i := Complex.I in  (∑ n in (Finset.range 203).map (λ k, -101 + k) , i^n) = 0 :=
by
  -- proof omitted
  sorry

end sum_powers_of_i_l327_327813


namespace james_selling_price_l327_327594

variable (P : ℝ)  -- Selling price per candy bar

theorem james_selling_price 
  (boxes_sold : ℕ)
  (candy_bars_per_box : ℕ) 
  (cost_price_per_candy_bar : ℝ)
  (total_profit : ℝ)
  (H1 : candy_bars_per_box = 10)
  (H2 : boxes_sold = 5)
  (H3 : cost_price_per_candy_bar = 1)
  (H4 : total_profit = 25)
  (profit_eq : boxes_sold * candy_bars_per_box * (P - cost_price_per_candy_bar) = total_profit)
  : P = 1.5 :=
by 
  sorry

end james_selling_price_l327_327594


namespace prime_sum_10003_l327_327524

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327524


namespace board_has_valid_product_l327_327640

def is_valid_product (XYZ YXZ p : ℕ) : Prop :=
  let digits := [X, Y, Z] in
  let prod_digits := digits ++ [p.digit 0] in     -- middle digits condition
  (1 < X ∧ X < 10 ∧ 1 < Y ∧ Y < 10 ∧ 1 < Z ∧ Z < 10) ∧  -- non-zero and decimal digits
  (∀ d ∈ digits, ∀ e ∈ digits, d ≠ e) ∧   -- distinct digits
  p.digit (p.num_digits-1) = p.digit 0 ∧  -- outer digits are equal
  (∀ d ∈ digits, p.digit (p.num_digits-1) ≠ d) ∧ -- outer digits different from middle ones
  ⟨X, Y, Z⟩ ∈ {(2, 6, 9), (3, 5, 9)} /- numbers pairs -/

theorem board_has_valid_product :
  ∃ XYZ YXZ p, is_valid_product XYZ YXZ p ∧ (p = 169201 ∨ p = 193501) := 
sorry

end board_has_valid_product_l327_327640


namespace complex_vector_PQ_l327_327124

noncomputable def complex_number (z : ℂ) : ℂ := z

theorem complex_vector_PQ :
  let z := 1 - complex.I
  let OP := complex_number z
  let OQ := complex_number (z^2)
  ∃ PQ, PQ = OQ - OP ∧ PQ = -1 - complex.I := 
by
  let z := 1 - complex.I
  let OP := complex_number z
  let OQ := complex_number (z^2)
  use OQ - OP
  split
  case inl =>
    exact rfl
  case inr =>
    sorry

end complex_vector_PQ_l327_327124


namespace jerry_won_47_tickets_l327_327803

open Nat

-- Define the initial number of tickets
def initial_tickets : Nat := 4

-- Define the number of tickets spent on the beanie
def tickets_spent_on_beanie : Nat := 2

-- Define the current total number of tickets Jerry has
def current_tickets : Nat := 49

-- Define the number of tickets Jerry won later
def tickets_won_later : Nat := current_tickets - (initial_tickets - tickets_spent_on_beanie)

-- The theorem to prove
theorem jerry_won_47_tickets :
  tickets_won_later = 47 :=
by sorry

end jerry_won_47_tickets_l327_327803


namespace compressor_stations_distances_l327_327750

theorem compressor_stations_distances 
    (x y z a : ℝ) 
    (h1 : x + y = 2 * z)
    (h2 : z + y = x + a)
    (h3 : x + z = 75)
    (h4 : 0 ≤ x)
    (h5 : 0 ≤ y)
    (h6 : 0 ≤ z)
    (h7 : 0 < a)
    (h8 : a < 100) :
  (a = 15 → x = 42 ∧ y = 24 ∧ z = 33) :=
by 
  intro ha_eq_15
  sorry

end compressor_stations_distances_l327_327750


namespace solve_for_x_l327_327189

theorem solve_for_x (x : ℝ) (h : x ≠ 2) : (7 * x) / (x - 2) - 5 / (x - 2) = 3 / (x - 2) → x = 8 / 7 :=
by
  sorry

end solve_for_x_l327_327189


namespace axis_of_symmetry_l327_327198

theorem axis_of_symmetry (x : ℝ) : 
  ∀ y, y = x^2 - 2 * x - 3 → (∃ k : ℝ, k = 1 ∧ ∀ x₀ : ℝ, y = (x₀ - k)^2 + C) := 
sorry

end axis_of_symmetry_l327_327198


namespace propositions_correctness_l327_327151

variable {a b c d : ℝ}

theorem propositions_correctness (h0 : a > b) (h1 : c > d) (h2 : c > 0) :
  (a > b ∧ c > d → a + c > b + d) ∧ 
  (a > b ∧ c > d → ¬(a - c > b - d)) ∧ 
  (a > b ∧ c > d → ¬(a * c > b * d)) ∧ 
  (a > b ∧ c > 0 → a * c > b * c) :=
by
  sorry

end propositions_correctness_l327_327151


namespace prime_sum_10003_l327_327533

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327533


namespace complement_intersection_l327_327996

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end complement_intersection_l327_327996


namespace find_value_of_k_l327_327941

def line_equation_holds (m n : ℤ) : Prop := m = 2 * n + 5
def second_point_condition (m n k : ℤ) : Prop := m + 4 = 2 * (n + k) + 5

theorem find_value_of_k (m n k : ℤ) 
  (h1 : line_equation_holds m n) 
  (h2 : second_point_condition m n k) : 
  k = 2 :=
by sorry

end find_value_of_k_l327_327941


namespace sum_of_two_primes_unique_l327_327491

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327491


namespace volume_of_larger_cube_l327_327232

theorem volume_of_larger_cube (V_small_cube : ℝ) (V_large_cube : ℝ) (a : ℝ) (h : ℝ) :
  (V_small_cube = 2^3) ∧ ((2 * a) = h) ∧ (V_large_cube = h^3) → V_large_cube = 64 :=
by 
  assume h_conditions,
  sorry

end volume_of_larger_cube_l327_327232


namespace kenny_charges_15_dollars_per_lawn_l327_327601

def charge_per_lawn (x : ℝ) : Prop :=
  let earnings := 35 * x in
  let video_game_cost := 5 * 45 in
  let book_cost := 60 * 5 in
  let total_spent := video_game_cost + book_cost in
  earnings = total_spent

theorem kenny_charges_15_dollars_per_lawn : ∃ x : ℝ, charge_per_lawn x ∧ x = 15 := 
  sorry

end kenny_charges_15_dollars_per_lawn_l327_327601


namespace find_x_l327_327747

theorem find_x (x : ℤ) (h : 3 * x = (26 - x) + 10) : x = 9 :=
by
  -- proof steps would be provided here
  sorry

end find_x_l327_327747


namespace number_of_ways_sum_of_primes_l327_327501

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327501


namespace sqrt_expression_eq_36_l327_327309

theorem sqrt_expression_eq_36 : (Real.sqrt ((3^2 + 3^3)^2)) = 36 := 
by
  sorry

end sqrt_expression_eq_36_l327_327309


namespace prime_sum_10003_l327_327531

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327531


namespace tangent_circles_colinear_points_l327_327675

theorem tangent_circles_colinear_points (n : ℕ) (R1 R2 : Circle) 
(S : Fin n → Circle) (A : Fin (n - 1) → Point) : 
  (∀ i, S i ∈ [tangent R1, tangent R2]) →
  (∀ i, A i ∈ tangent_points (S i) (S (i + 1))) →
  ∃ C : Circle, ∀ i, A i ∈ C :=
by sorry

end tangent_circles_colinear_points_l327_327675


namespace jessie_weight_loss_l327_327291

theorem jessie_weight_loss (current_weight before_weight : ℝ) (h_current : current_weight = 66) (h_before : before_weight = 192) :
  before_weight - current_weight = 126 :=
by
  rw [h_current, h_before]
  exact sub_eq_iff_eq_add.mpr rfl

end jessie_weight_loss_l327_327291


namespace tangent_line_at_point_l327_327202

-- Define the function representing the curve
def curve (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def deriv_curve (x : ℝ) : ℝ := 3*x^2 - 2

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 0)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := x - 1

-- State the theorem to prove that the equation of the tangent line at the given point is as specified
theorem tangent_line_at_point : 
  ∀ (x : ℝ),
  point_of_tangency.2 = curve point_of_tangency.1 →
  tangent_line x = curve point_of_tangency.1 + deriv_curve point_of_tangency.1 * (x - point_of_tangency.1) :=
by {
  intros,
  rw [point_of_tangency, ← curve],
  sorry
}

end tangent_line_at_point_l327_327202


namespace power_function_is_odd_l327_327101

theorem power_function_is_odd (m : ℝ) (x : ℝ) (h : (m^2 - m - 1) * (-x)^m = -(m^2 - m - 1) * x^m) : m = -1 :=
sorry

end power_function_is_odd_l327_327101


namespace lateral_surface_area_of_pyramid_l327_327463

variables (S : ℝ)
variables (α : ℝ) [fact (α = 120)]
variables (base_is_square : Prop)

theorem lateral_surface_area_of_pyramid (h_base_is_square : base_is_square) 
  (h_dihedral_angle : α = 120) (h_diagonal_section_area : Prop) :
  lateral_surface_area S = 4 * S :=
sorry

end lateral_surface_area_of_pyramid_l327_327463


namespace polar_to_rectangular_point_l327_327312

theorem polar_to_rectangular_point (r θ : ℝ) 
  (h_r : r = 6) 
  (h_θ : θ = 5 * Real.pi / 3) 
  (cos_val : Real.cos θ = 1 / 2) 
  (sin_val : Real.sin θ = -(Real.sqrt 3) / 2) : 
    (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
  by
    rw [h_r, h_θ, cos_val, sin_val]
    sorry

end polar_to_rectangular_point_l327_327312


namespace expectation_inequality_l327_327254

open ProbabilityTheory

variables {Ω : Type*} {mΩ : MeasurableSpace Ω}
  {P : MeasureTheory.Measure Ω} [MeasureTheory.ProbabilityMeasure P]
  (X Y Z : Ω → ℝ) [Independent' P (λ ω, X ω)] [Independent' P (λ ω, Y ω)] [Independent' P (λ ω, Z ω)]
  (f g h : ℝ × ℝ → ℝ) (hf : BorelMeasurable f) (hg : BorelMeasurable g) (hh : BorelMeasurable h)
  (bf : BoundedBorelFunction f) (bg : BoundedBorelFunction g) (bh : BoundedBorelFunction h)

noncomputable def expectation := sorry

theorem expectation_inequality :
  |(expectation (λ ω, f (X ω, Y ω) * g (Y ω, Z ω) * h (Z ω, X ω)))| ^ 2
  ≤ (expectation (λ ω, (f (X ω, Y ω)) ^ 2)) *
    (expectation (λ ω, (g (Y ω, Z ω)) ^ 2)) *
    (expectation (λ ω, (h (Z ω, X ω)) ^ 2)) := sorry

end expectation_inequality_l327_327254


namespace toy_cost_l327_327655

theorem toy_cost (initial_amount spent_amount toys : ℕ) 
  (h1 : initial_amount = 68) 
  (h2 : spent_amount = 47) 
  (h3 : toys = 3) 
  (remaining_amount : ℕ := initial_amount - spent_amount): 
  remaining_amount / toys = 7 := 
by
  rw [h1, h2, h3]
  simp
  sorry

end toy_cost_l327_327655


namespace max_true_statements_l327_327610

theorem max_true_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 / a < 1 / b) ∧ (a^2 > b^2) ∧ (a > b) ∧ (a > 0) ∧ (b > 0) :=
by {
  split, 
  sorry,
  split,
  sorry,
  split,
  sorry,
  split,
  sorry,
  exact hb
}

end max_true_statements_l327_327610


namespace sum_of_series_l327_327837

open Complex

theorem sum_of_series :
  ∑' n : ℕ, (-1 : ℂ)^n / (3 * n + 1) = (1 / 3 : ℂ) * (log 2 + π / √3) :=
sorry

end sum_of_series_l327_327837


namespace GCF_LCM_calculation_l327_327721

theorem GCF_LCM_calculation : 
  GCD (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end GCF_LCM_calculation_l327_327721


namespace unique_not_in_range_g_l327_327895

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ :=
  (2 * a * x + b) / (3 * c * x + d)

theorem unique_not_in_range_g (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g a b c d 13 = 13) (h6 : g a b c d 31 = 31)
  (h7 : ∀ x, x ≠ -d / (3 * c) → g a b c d (g a b c d x) = x) : 
  ∃! y, ∀ x, g a b c d x ≠ y := 
begin
  use 4 / 9,
  intros x hx,
  sorry
end

end unique_not_in_range_g_l327_327895


namespace F_at_2_eq_minus_22_l327_327851

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x

def F (x : ℝ) : ℝ := f a b c d x - 6

theorem F_at_2_eq_minus_22 (h : F a b c d (-2) = 10) : F a b c d 2 = -22 :=
by
  sorry

end F_at_2_eq_minus_22_l327_327851


namespace inverse_eval_l327_327684

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1 + Real.logBase a (x + 3)

theorem inverse_eval (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ x : ℝ, f a x = -1 ∧ x = -2 := by
  sorry

end inverse_eval_l327_327684


namespace shaded_area_l327_327283

noncomputable def area_shaded_region : ℝ :=
  let s := 8 in
  let r := 4 in
  let A_hexagon := 6 * (sqrt 3 / 4 * s^2) in
  let A_sector := (1 / 6) * π * r^2 in
  let total_A_sectors := 6 * A_sector in
  A_hexagon - total_A_sectors

theorem shaded_area :
  area_shaded_region = 96 * sqrt 3 - 16 * π :=
by 
  sorry

end shaded_area_l327_327283


namespace arg_range_eq_l327_327048

noncomputable def range_arg_z_minus_b_plus_bi 
  (a b : ℝ) (z : ℂ)
  (h1: 0 < a) 
  (h2: a > b) 
  (h3: b > 0) 
  (h4: complex.arg (z + a + complex.I * a) = real.pi / 4)
  (h5: complex.arg (z - a - complex.I * a) = 5 * real.pi / 4) : 
  Set ℝ :=
  {t | t ∈ Set.Icc (real.arctan ((a + b) / (a - b))) (real.pi + real.arctan ((a - b) / (a + b)))}

theorem arg_range_eq
  (a b : ℝ) (z : ℂ)
  (h1: 0 < a) 
  (h2: a > b) 
  (h3: b > 0) 
  (h4: complex.arg (z + a + complex.I * a) = real.pi / 4)
  (h5: complex.arg (z - a - complex.I * a) = 5 * real.pi / 4) : 
  complex.arg (z - b + complex.I * b) ∈ range_arg_z_minus_b_plus_bi a b z h1 h2 h3 h4 h5 :=
sorry

end arg_range_eq_l327_327048


namespace aces_win_probability_l327_327195

-- Define the probability of Aces winning a single game
def p : ℝ := 7 / 10

-- Define the probability of Kings winning a single game
def q : ℝ := 3 / 10

-- The probability of the Aces winning the series
def P_Aces_win_series : ℝ :=
  (∑ k in Finset.range 5, Nat.choose (4 + k) k * p^5 * q^k) 

-- The theorem to prove
theorem aces_win_probability : P_Aces_win_series = 0.90087 := by
  sorry

end aces_win_probability_l327_327195


namespace woman_away_time_l327_327288

theorem woman_away_time :
  ∃ t : ℝ, 
    (0 < t ∧ t < 60) ∧
    (|210 + 0.5 * t - 6 * t| = 120) ∧
    (|210 + 0.5 * (t + 30) - 6 * (t + 30)| = 120) → 
    t + 30 = 30 :=
begin
  sorry
end

end woman_away_time_l327_327288


namespace probability_highest_six_is_correct_l327_327760

open ProbabilityTheory

noncomputable def probability_highest_six : ℝ :=
  let box := finset.range 7,
      draw := joint (λ _ : { x // x ∈ box }, uniform box) 4
  in (Pr (λ (selection : finset α), 6 ∈ selection ∧ 7 ∉ selection) (prob draw))

theorem probability_highest_six_is_correct : probability_highest_six = 2/7 :=
sorry

end probability_highest_six_is_correct_l327_327760


namespace sum_of_roots_l327_327992

noncomputable def f (x : ℝ) : ℝ := 10^x + x - 7
noncomputable def g (x : ℝ) : ℝ := log x + x - 7

theorem sum_of_roots :
  (∃ x1 : ℝ, f x1 = 0) →
  (∃ x2 : ℝ, g x2 = 0) →
  ∀ x1 x2, f x1 = 0 → g x2 = 0 →  x1 + x2 = 7 :=
by
  intros h1 h2 x1 x2 hf hg
  sorry

end sum_of_roots_l327_327992


namespace parabola_intersection_l327_327416

theorem parabola_intersection (a b : ℝ) (h : a ≠ 0) (hP : (3 - b) * (3 - 1) = 0) :
  ∃ x : ℝ, x ≠ 3 ∧ a * (x - b) * (x - 1) = 0 ∧ x = 1 :=
by
  use 1
  split
  -- x ≠ 3
  sorry
  split
  -- a * (1 - b) * (1 - 1) = 0
  sorry
  -- x = 1
  refl

end parabola_intersection_l327_327416


namespace sum_of_primes_10003_l327_327545

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327545


namespace polar_coordinates_of_point_2_neg2_l327_327313

theorem polar_coordinates_of_point_2_neg2 :
  ∃ r θ, r = 2 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  use 2 * Real.sqrt 2
  use 7 * Real.pi / 4
  split; repeat { split }
  { sorry }
  { sorry }
  { sorry }
  { sorry }

end polar_coordinates_of_point_2_neg2_l327_327313


namespace minimum_value_ineq_l327_327985

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  (1 : ℝ) ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by {
  sorry
}

end minimum_value_ineq_l327_327985


namespace base5_division_l327_327340

theorem base5_division :
  let base5_1324 := 1324 
  let base5_23 := 23 
  (base5_div base5_1324 base5_23) = 31.21 := by
  sorry

end base5_division_l327_327340


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327478

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327478


namespace no_valid_digit_l327_327085

theorem no_valid_digit :
  ∀ (d : ℕ), d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  5 * d + 2 ≠ 9 * d + 7 :=
by {
  assume d hd,
  sorry
}

end no_valid_digit_l327_327085


namespace trapezoid_angle_l327_327584

theorem trapezoid_angle
  (EFGH : Type)
  (EF GH : EFGH)
  (parallel : ∃ (EF GH : EFGH), EF ∥ GH)
  (angle_E_eq : ∃ (E H : ℝ), E = 3 * H)
  (angle_G_eq : ∃ (G F : ℝ), G = 2 * F)
  (angle_sum : ∃ (F G : ℝ), F + G = 180) :
  ∃ (F : ℝ), F = 60 :=
by
  sorry

end trapezoid_angle_l327_327584


namespace kareem_family_ages_l327_327955

/-- 
  Suppose:
  1. Kareem is 3 times as old as his son.
  2. Kareem's daughter is half his son's age.
  3. After 10 years, the sum of Kareem, his son, and his daughter's ages will be 120 years.
  4. Kareem's wife is 8 years younger than him.

  Then:
  - Kareem is currently 60 years old.
  - His son is currently 20 years old.
  - His daughter is currently 10 years old.
  - His wife is currently 52 years old.
-/
theorem kareem_family_ages :
  ∃ (S : ℕ), 
    let Kareem := 3 * S in
    let Daughter := S / 2 in
    (Kareem + 10) + (S + 10) + (Daughter + 10) = 120 →
    Kareem = 60 ∧ S = 20 ∧ Daughter = 10 ∧ (Kareem - 8) = 52 :=
by
  sorry

end kareem_family_ages_l327_327955


namespace smallest_number_leaving_remainder_1_and_divisible_by_11_l327_327733

theorem smallest_number_leaving_remainder_1_and_divisible_by_11 :
  ∃ n : ℕ, (n % 11 = 0) ∧ (∀ m ∈ {2, 3, 4, 5, 6, 7, 8}, n % m = 1) ∧ n = 6721 :=
by
  sorry

end smallest_number_leaving_remainder_1_and_divisible_by_11_l327_327733


namespace remainder_div_l327_327180

theorem remainder_div (N : ℕ) (n : ℕ) : 
  (N % 2^n) = (N % 10^n % 2^n) ∧ (N % 5^n) = (N % 10^n % 5^n) := by
  sorry

end remainder_div_l327_327180


namespace evaluate_e_T_l327_327143

open Real

noncomputable def T : ℝ :=
  ∫ x in 0..(ln 2), (2 * exp (3 * x) + exp (2 * x) - 1) / (exp (3 * x) + exp (2 * x) - exp x + 1)

theorem evaluate_e_T : exp T = 11 / 4 :=
by
  sorry

end evaluate_e_T_l327_327143


namespace find_m_l327_327425

noncomputable def vector_a : ℝ × ℝ := (1, 3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1, 3) + 2 • (-2, m)

theorem find_m : ∀ (m : ℝ), (vector_a.1 * vector_sum m.1) + (vector_a.2 * vector_sum m.2) = 0 → m = -1 :=
by 
  intros m h,
  sorry

end find_m_l327_327425


namespace prob_A_B_path_l327_327614

open ProbabilityTheory

structure Point (α : Type*) :=
(coord : α)

noncomputable def edge_prob (u v : Point ℝ) : ℝ := 1 / 2

noncomputable def prob_A_B_connected (A B C D : Point ℝ) (indep : ∀ u v w x: Point ℝ, u ≠ v → w ≠ x → Prob.indep_event (u, v) (w, x)) : ℝ :=
  3 / 4

theorem prob_A_B_path (A B C D : Point ℝ) (h : ∀ u v w x: Point ℝ, u ≠ v → w ≠ x → Prob.indep_event (u, v) (w, x)) (ncoplanar : ¬ (A.coord, B.coord, C.coord, D.coord).coplanar):
  prob_A_B_connected A B C D h = 3 / 4 :=
sorry

end prob_A_B_path_l327_327614


namespace house_expansion_l327_327224

theorem house_expansion (initial_small_house : ℕ) (initial_large_house : ℕ) (new_total : ℕ) :
  initial_small_house = 5200 →
  initial_large_house = 7300 →
  new_total = 16000 →
  (new_total - (initial_small_house + initial_large_house) = 3500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end house_expansion_l327_327224


namespace max_value_of_z_l327_327835

theorem max_value_of_z (x y z : ℝ) (h_add : x + y + z = 5) (h_mult : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end max_value_of_z_l327_327835


namespace validate_survey_data_l327_327799

-- Frequency distribution in groups
def frequency_data : List ℕ := [4, 10, 46, 16, 20, 4]

-- Frequncy Rate for [20,40)
def frequency_rate_20_40 := 0.1

-- Total students surveyed
def n := 100

-- Frequency for [60,80) and its rate
def f_3 := 16.0 / 100.0

-- Calculated probability
def prob_80_or_above := 6.0 / 10.0

-- Min reading time for "Reading Master"
def min_reading_master_time := 94

-- Prove correct values
theorem validate_survey_data :
  (4 + 10 + 46 + 16 + 20 + 4 = n) ∧
  (10.0 / 100.0 = frequency_rate_20_40) ∧
  (f_3 = 0.16) ∧
  (prob_80_or_above = 0.6) ∧
  (min_reading_master_time = 94) :=
by {
  simp [frequency_data, frequency_rate_20_40, n, f_3, prob_80_or_above, min_reading_master_time],
  split,
  repeat {split},
  all_goals {sorry}
}

end validate_survey_data_l327_327799


namespace circumcenter_MNH_on_OH_l327_327159

-- defining an acute triangle
variables (A B C : Type)

-- defining the points O and H
variable (O H : Type)

-- defining points M and N via intersection properties
variables (M N : Type)

-- defining circumcircle and intersections
variable (circumcircle_AHC : Type -> Type)
variable (circumcircle_AHB : Type -> Type)

-- stating the assumptions based on conditions
variable (ACUTE_TRIANGLE : True)
variable (AB_GT_BC : Prop)
variable (AC_GT_BC : Prop)
variable (O_is_circumcenter : Prop)
variable (H_is_orthocenter : Prop)
variable (M_on_AB_different_from_A : Prop)
variable (N_on_AC_different_from_A : Prop)

-- the theorem statement
theorem circumcenter_MNH_on_OH 
  (AB_GT_BC : AB_GT_BC)
  (AC_GT_BC : AC_GT_BC)
  (O_is_circumcenter : O_is_circumcenter)
  (H_is_orthocenter : H_is_orthocenter)
  (M_on_AB_different_from_A : M_on_AB_different_from_A)
  (N_on_AC_different_from_A : N_on_AC_different_from_A) :
  ∃ (circumcenter_MNH : Type), lies_on (circumcenter_MNH) (line_through O H) := 
  sorry

end circumcenter_MNH_on_OH_l327_327159


namespace burrs_count_l327_327298

variable (B T : ℕ)

theorem burrs_count 
  (h1 : T = 6 * B) 
  (h2 : B + T = 84) : 
  B = 12 := 
by
  sorry

end burrs_count_l327_327298


namespace length_of_goods_train_l327_327277

theorem length_of_goods_train
  (speed_man_train : ℕ) (speed_goods_train : ℕ) (passing_time : ℕ)
  (h1 : speed_man_train = 40)
  (h2 : speed_goods_train = 72)
  (h3 : passing_time = 9) :
  (112 * 1000 / 3600) * passing_time = 280 := 
by
  sorry

end length_of_goods_train_l327_327277


namespace sum_of_primes_10003_l327_327552

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327552


namespace students_going_to_tournament_l327_327631

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end students_going_to_tournament_l327_327631


namespace largest_sample_num_l327_327017

theorem largest_sample_num 
    (N : ℕ) (a1 a2 largest : ℕ)
    (a1_cond : a1 = 7) 
    (a2_cond : a2 = 32)
    (range_cond : N = 500)
    (sampling_interval : ℕ := a2 - a1)
    (size := N / sampling_interval)
    (max_index : ℕ := size)
    (formula : ∀ n, 1 ≤ n ∧ n ≤ max_index → ℕ := λ n, a1 + sampling_interval * (n - 1)) :
    largest = formula max_index := 
by
  sorry

end largest_sample_num_l327_327017


namespace unique_sum_of_two_primes_l327_327554

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327554


namespace problem_circumscribing_sphere_surface_area_l327_327863

noncomputable def surface_area_of_circumscribing_sphere (a b c : ℕ) :=
  let R := (Real.sqrt (a^2 + b^2 + c^2)) / 2
  4 * Real.pi * R^2

theorem problem_circumscribing_sphere_surface_area
  (a b c : ℕ)
  (ha : (1 / 2 : ℝ) * a * b = 4)
  (hb : (1 / 2 : ℝ) * b * c = 6)
  (hc : (1 / 2: ℝ) * a * c = 12) : 
  surface_area_of_circumscribing_sphere a b c = 56 * Real.pi := 
sorry

end problem_circumscribing_sphere_surface_area_l327_327863


namespace circle_area_l327_327456

theorem circle_area 
  (r : ℝ) 
  (h1 : 6 * (1 / (2 * real.pi * r)) = 2 * r) : 
  π * r^2 = 3 / 2 := 
sorry

end circle_area_l327_327456


namespace max_modulus_Z_l327_327620

open Complex

noncomputable def Z : ℂ := sorry

theorem max_modulus_Z : (∃ Z : ℂ, |Z - (3 + 4 * Complex.I)| = 1) → ∃ Z : ℂ, |Z| ≤ 6 := sorry

end max_modulus_Z_l327_327620


namespace find_abs_y_diff_l327_327037

noncomputable def ellipse_minor_axis_length : ℝ := 8
noncomputable def ellipse_eccentricity : ℝ := 3 / 5
def f1 : ℝ × ℝ := (-3, 0)
def f2 : ℝ × ℝ := (3, 0)
noncomputable def circum_inscribed_circle : ℝ := π

theorem find_abs_y_diff
  (e : ℝ := ellipse_eccentricity)
  (minor_axis : ℝ := ellipse_minor_axis_length)
  (a b : ℝ)
  (h1 : 2 * b = minor_axis)
  (h2 : sqrt (1 - b^2 / a^2) = e)
  (h3 : a = 5) :
  |(0 - 0)| = 5 / 3 :=
by
  sorry

end find_abs_y_diff_l327_327037


namespace f_is_odd_g_is_even_l327_327868

def f (x : ℝ) : ℝ := x + 1 / x
def g (x : ℝ) : ℝ := 2^|x|

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := sorry
theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) := sorry

end f_is_odd_g_is_even_l327_327868


namespace alpha_half_l327_327898

theorem alpha_half (α : ℝ) :
  (∀ x : ℝ, x ^ α = f x) ∧ f (1 / 2) = sqrt 2 / 2 → α = 1 / 2 :=
by
  sorry

end alpha_half_l327_327898


namespace sum_to_product_identity_l327_327324

variable (x : ℝ)

theorem sum_to_product_identity : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sum_to_product_identity_l327_327324


namespace part1_l327_327182

theorem part1 (m : ℝ) (a b : ℝ) (h : m > 0) : 
  ( (a + m * b) / (1 + m) )^2 ≤ (a^2 + m * b^2) / (1 + m) :=
sorry

end part1_l327_327182


namespace find_AD_l327_327131

noncomputable def AD_length (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (angle_ABD angle_ACD : ℝ) (angle_BAC : ℝ) (BC : ℝ) : ℝ :=
if (angle_ABD = pi/4 ∧ angle_ACD = pi/4 ∧ angle_BAC = pi/6 ∧ BC = 1) then sqrt 2 else 0

theorem find_AD (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (angle_ABD angle_ACD : ℝ) (angle_BAC : ℝ) (BC : ℝ) :
  angle_ABD = pi/4 → angle_ACD = pi/4 → angle_BAC = pi/6 → BC = 1 → AD_length A B C D angle_ABD angle_ACD angle_BAC BC = sqrt 2 :=
by {
  intros,
  simp [AD_length],
  split_ifs,
  -- proof steps will go here
  sorry
}

end find_AD_l327_327131


namespace number_of_prime_pairs_for_10003_l327_327518

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327518


namespace circle_line_intersection_l327_327689

theorem circle_line_intersection :
  let radius := 2
  let line := λ x y : ℝ, 3 * x - 4 * y - 9
  let center := (0 : ℝ, 0 : ℝ)
  let dist := (|3 * 0 - 4 * 0 - 9| : ℝ) / (Real.sqrt (3 ^ 2 + (-4) ^ 2))
  dist < radius ∧ line center.1 center.2 ≠ 0 :=
by
  let radius := 2
  let line := λ x y : ℝ, 3 * x - 4 * y - 9
  let center := (0 : ℝ, 0 : ℝ)
  let dist := (|3 * 0 - 4 * 0 - 9| : ℝ) / (Real.sqrt (3 ^ 2 + (-4) ^ 2))
  have h_dist : dist = 9 / 5 := sorry
  have h_radius : 9 / 5 < radius := sorry
  have h_center : line 0  0 ≠ 0 := sorry
  exact And.intro h_radius h_center

end circle_line_intersection_l327_327689


namespace minimize_expression_l327_327877

theorem minimize_expression (x : ℝ) (h : 0 < x) : 
  x = 9 ↔ (∀ y : ℝ, 0 < y → x + 81 / x ≤ y + 81 / y) :=
sorry

end minimize_expression_l327_327877


namespace problem_statement_l327_327069

-- Definitions based on the conditions
def P : Prop := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (x / (x - 1) < 0)
def Q : Prop := ∀ (A B : ℝ), (A > B) → (A > 90 ∨ B < 90)

-- The proof problem statement
theorem problem_statement : P ∧ ¬Q := 
by
  sorry

end problem_statement_l327_327069


namespace sin_sum_to_product_identity_l327_327329

theorem sin_sum_to_product_identity (x : ℝ) :
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin_sum_to_product_identity_l327_327329


namespace A_divides_BC_l327_327401

variable (A B C O : Type) [AddCommGroup A] [Module ℝ A]
variables (a b c : A)
variables (m : ℝ)

-- Conditions
variable (h_collinear : ∃ k : ℝ, ∃ l : ℝ, b = k • a ∧ c = l • a)
variable (h_not_collinear : ¬ ∃ k : ℝ, O = k • A)
variable (h_eq_a : a = A)
variable (h_eq_b : b = B)
variable (h_eq_c : c = C)
variable (h_eq_m : m • a - 3 • b - c = 0)

-- Question 
theorem A_divides_BC (h : 3 • b = (3 * (1 / 3 + 1)) • a - (1 / 3) • c): 
  a ∉ OpenSegment ℝ b c  :=
by sorry

end A_divides_BC_l327_327401


namespace no_prime_sum_10003_l327_327541

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327541


namespace regular_21_gon_symmetries_and_angle_sum_l327_327781

theorem regular_21_gon_symmetries_and_angle_sum :
  let L' := 21
  let R' := 360 / 21
  L' + R' = 38.142857 := by
    sorry

end regular_21_gon_symmetries_and_angle_sum_l327_327781


namespace no_prime_sum_10003_l327_327469

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327469


namespace find_positive_value_m_l327_327010

theorem find_positive_value_m (m : ℝ) (h : abs (complex.mk 5 m) = 5 * real.sqrt 26) : m = 25 :=
sorry

end find_positive_value_m_l327_327010


namespace total_amount_earned_l327_327653

theorem total_amount_earned (work_days_rahul : ℕ) (work_days_rajesh : ℕ) (share_rahul : ℝ) : 
  work_days_rahul = 3 ∧ work_days_rajesh = 2 ∧ share_rahul = 68 →
  (let rahul_rate := 1 / work_days_rahul;
       rajesh_rate := 1 / work_days_rajesh;
       combined_rate := rahul_rate + rajesh_rate;
       ratio_rahul := rahul_rate / combined_rate;
       ratio_rajesh := rajesh_rate / combined_rate;
       total_parts := ratio_rahul.den + ratio_rajesh.den;
       part_value := share_rahul / (total_parts * ratio_rahul.den);
       total_amount := total_parts * part_value) in
   total_amount = 170 :=
by
  -- Proof to be filled in
  sorry

end total_amount_earned_l327_327653


namespace d3_location_d4_location_l327_327615

noncomputable def minimal_d3_location (P : RegularHexagon) (A : Point) : Set Point :=
  {A : Point | ∃ (s : ℝ) (M : Point), (P.is_center M) ∧ (P.center_distance A = s / 2) }

noncomputable def minimal_d4_location (P : RegularHexagon) : Point :=
  (P.center : Point)

theorem d3_location (P : RegularHexagon) (A : Point) (hA : A ∈ P.interior ∪ P.boundary) 
  (d : Fin₆ → ℝ) (h : (∀ (i : Fin₆), d i = distance A (P.vertex i)) ∧ 
                    (d 0 ≤ d 1 ≤ d 2 ≤ d 3 ≤ d 4 ≤ d 5)) :
  A ∈ minimal_d3_location P A := 
sorry

theorem d4_location (P : RegularHexagon) (A : Point) (hA : A ∈ P.interior ∪ P.boundary) 
  (d : Fin₆ → ℝ) (h : (∀ (i : Fin₆), d i = distance A (P.vertex i)) ∧ 
                    (d 0 ≤ d 1 ≤ d 2 ≤ d 3 ≤ d 4 ≤ d 5)) :
  A = minimal_d4_location P := 
sorry

end d3_location_d4_location_l327_327615


namespace range_of_a_l327_327062

-- Define the function y
def y (x a : ℝ) : ℝ := x^2 + (a + 1)^2 + |x + a - 1|

-- Define the main problem: proving the range of a such that the minimum value of y is greater than 5
theorem range_of_a (a : ℝ) :
  (∀ x, y x a > 5) ↔ (a < (1 - real.sqrt 14) / 2 ∨ a > real.sqrt 6 / 2) :=
sorry

end range_of_a_l327_327062


namespace sufficient_but_not_necessary_condition_l327_327026

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, ¬q x a ∧ p x) → a ∈ Set.Ici 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l327_327026


namespace farm_problem_l327_327222

theorem farm_problem (D C : ℕ) (h1 : D + C = 15) (h2 : 2 * D + 4 * C = 42) : C = 6 :=
sorry

end farm_problem_l327_327222


namespace range_of_m_l327_327875

def f (x m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m (a b c m : ℝ) (h1 : a ∈ Icc 0 2) (h2 : b ∈ Icc 0 2) (h3 : c ∈ Icc 0 2) 
    (triangle_cond : f a m + f b m > f c m ∧ f b m + f c m > f a m ∧ f c m + f a m > f b m) 
    : m > 6 :=
by
  sorry

end range_of_m_l327_327875


namespace tangent_line_eq_l327_327681

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 - x + 1) (h_point : (x, y) = (0, 1)) : x + y - 1 = 0 := 
sorry

end tangent_line_eq_l327_327681


namespace abs_neg_2023_l327_327672

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l327_327672


namespace sum_of_primes_10003_l327_327547

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327547


namespace driving_speed_Z_to_Y_l327_327236

/-- Given the following conditions:
1. Venki drives 5 hours from town X to town Z at a rate of 80 miles per hour.
2. Town Y is midway between town X and town Z.
3. It takes Venki 4.444444444444445 hours to drive from town Z to town Y.

We need to prove that Venki's driving speed from town Z to town Y is approximately 42.86 miles per hour. -/
theorem driving_speed_Z_to_Y :
  ∀ (d_XZ : ℝ) (d_ZY : ℝ) (time_XZ : ℝ) (time_ZY : ℝ) (speed_XZ : ℝ),
    time_XZ = 5 -> speed_XZ = 80 ->
    d_XZ = speed_XZ * time_XZ -> d_ZY = d_XZ / 2 ->
    time_ZY = 4.444444444444445 ->
    (d_ZY / time_ZY) ≈ 42.86 :=
by 
  intros d_XZ d_ZY time_XZ time_ZY speed_XZ 
  intros H_time_XZ H_speed_XZ H_d_XZ H_d_ZY H_time_ZY
  -- Start the proof
  sorry

end driving_speed_Z_to_Y_l327_327236


namespace range_of_a_l327_327435

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, a*x^2 - 2*a*x + 3 ≤ 0) ↔ (0 ≤ a ∧ a < 3) := 
sorry

end range_of_a_l327_327435


namespace number_of_true_propositions_l327_327369

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem number_of_true_propositions (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, f a b c x ≠ x)
  (p1 : ∀ x:ℝ, a > 0 → f a b c (f a b c x) > x)
  (p3 : ∀ x : ℝ, f a b c (f a b c x) ≠ x) 
  (p4 : (a + b + c = 0) → ∀ x : ℝ, f a b c (f a b c x) < x):
  3 :=
by
  sorry

end number_of_true_propositions_l327_327369


namespace cows_equal_ducks_plus_26_l327_327249

theorem cows_equal_ducks_plus_26 (D C : ℕ) 
    (legs_from_ducks : 2 * D) 
    (legs_from_cows : 4 * C)
    (total_heads : D + C)
    (total_legs : 2 * D + 4 * C = 3 * (D + C) + 26) :
    C = D + 26 :=
    sorry

end cows_equal_ducks_plus_26_l327_327249


namespace correct_option_l327_327390

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l327_327390


namespace krystiana_monthly_income_l327_327963

theorem krystiana_monthly_income :
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  first_floor_income + second_floor_income + third_floor_income = 165 :=
by
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  have h1: first_floor_income = 45 := by simp [first_floor_income]
  have h2: second_floor_income = 60 := by simp [second_floor_income]
  have h3: third_floor_income = 60 := by simp [third_floor_income]
  rw [h1, h2, h3]
  simp
  done

end krystiana_monthly_income_l327_327963


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327474

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327474


namespace CE_squared_plus_DE_squared_eq_108_l327_327609

noncomputable theory
open_locale classical

-- Define the points and elements of the circle
variables {O A B C D E : Type} [metric_space O]
variables (r : ℝ) (radius_6 : r = 6)
variables (diam_AB : dist A B = 2 * r)
variables (chord_CD : dist C D)
variables (intersect_E : E ∈ line_segment A B)
variables (BE_eq_3 : dist B E = 3)
variables (angle_AEC_30 : ∠ A E C = π / 6)

-- Define the problem statement in Lean 4
theorem CE_squared_plus_DE_squared_eq_108
    (radius_6 : r = 6)
    (diam_AB : dist A B = 2 * r)
    (BE_eq_3 : dist B E = 3)
    (angle_AEC_30 : ∠ A E C = π / 6) :
    ∃ CE DE : ℝ, CE^2 + DE^2 = 108 :=
sorry

end CE_squared_plus_DE_squared_eq_108_l327_327609


namespace sqrt_eq_9_implies_n_eq_73_l327_327440

theorem sqrt_eq_9_implies_n_eq_73 (n : ℕ) : sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_eq_9_implies_n_eq_73_l327_327440


namespace largest_square_tile_for_board_l327_327656

theorem largest_square_tile_for_board (length width gcd_val : ℕ) (h1 : length = 16) (h2 : width = 24) 
  (h3 : gcd_val = Int.gcd length width) : gcd_val = 8 := by
  sorry

end largest_square_tile_for_board_l327_327656


namespace ellipse_midpoint_length_l327_327871

noncomputable def length_segment_OM (P F1 F2 M : ℝ × ℝ) (C : ℝ × ℝ → Prop) :=
  let ellipse := ∀ (x y : ℝ), C(x,y) ↔ (x^2 / 36 + y^2 / 27 = 1)
  ∧ F1 = (-3, 0)
  ∧ F2 = (3, 0)
  ∧ (P ∈ {p | C p})
  ∧ dist P F1 = 8
  ∧ M = ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2)
in dist (0, 0) M = 2

theorem ellipse_midpoint_length (F1 F2 P M : ℝ × ℝ) (C : ℝ × ℝ → Prop) :
  let ellipse := ∀ (x y : ℝ), C(x,y) ↔ (x^2 / 36 + y^2 / 27 = 1) in
  ellipse
  ∧ F1 = (-3, 0)
  ∧ F2 = (3, 0)
  ∧ (P ∈ {p | C p})
  ∧ dist P F1 = 8
  → dist (0, 0) ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) = 2
:= by
  sorry

end ellipse_midpoint_length_l327_327871


namespace max_value_of_f_l327_327004

def f (a b : ℝ) : ℝ := (|7 * a + 8 * b - a * b| + |2 * a + 8 * b - 6 * a * b|) / (a * sqrt (1 + b^2))

theorem max_value_of_f :
  (∀ a b : ℝ, a ≥ 1 → b ≥ 1 → f a b ≤ 9 * sqrt 2) ∧ 
  (∃ a b : ℝ, a ≥ 1 ∧ b ≥ 1 ∧ f a b = 9 * sqrt 2) := by
  sorry

end max_value_of_f_l327_327004


namespace James_average_speed_l327_327137

theorem James_average_speed (TotalDistance : ℝ) (BreakTime : ℝ) (TotalTripTime : ℝ) (h1 : TotalDistance = 42) (h2 : BreakTime = 1) (h3 : TotalTripTime = 9) :
  (TotalDistance / (TotalTripTime - BreakTime)) = 5.25 :=
by
  sorry

end James_average_speed_l327_327137


namespace abs_neg_2023_l327_327671

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l327_327671


namespace GCF_LCM_calculation_l327_327722

theorem GCF_LCM_calculation : 
  GCD (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end GCF_LCM_calculation_l327_327722


namespace circumscribed_sphere_radius_l327_327692

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt (6 + Real.sqrt 20)) / 8

theorem circumscribed_sphere_radius (a : ℝ) :
  radius_of_circumscribed_sphere a = a * (Real.sqrt (6 + Real.sqrt 20)) / 8 :=
by
  sorry

end circumscribed_sphere_radius_l327_327692


namespace rectangle_diagonal_property_l327_327043

-- Assuming some preliminary setup for the rectangle and the perpendicularity conditions

variables {A B C D E F : Point}
variables {AB AD AC : ℝ}

-- Assume the rectangle and relevant properties
def RectangleABCDisRect (A B C D : Point) : Prop :=
  -- Define the properties of the rectangle here
  sorry

-- Points E and F are such that CE ⊥ AB and CF ⊥ AD
def PerpendicularCE_AB (C E : Point) (AB : Line) : Prop :=
  -- Define the perpendicularity of CE to AB here
  sorry

def PerpendicularCF_AD (C F : Point) (AD : Line) : Prop :=
  -- Define the perpendicularity of CF to AD here
  sorry

theorem rectangle_diagonal_property
  (h_rect : RectangleABCDisRect A B C D)
  (h_perp_ce_ab : PerpendicularCE_AB C E AB)
  (h_perp_cf_ad : PerpendicularCF_AD C F AD) :
  AB * dist A E + AD * dist A F = (dist A C) ^ 2 :=
sorry

end rectangle_diagonal_property_l327_327043


namespace largest_n_with_f_n_eq_35_l327_327866

def second_largest_divisor (n : ℕ) : ℕ :=
  if n < 2 then 0 else (List.sort (· > ·) (List.filter (· ∣ n) (List.range (n + 1)))).nth! 1

theorem largest_n_with_f_n_eq_35 :
  ∃ (n : ℕ), n ≥ 2 ∧ second_largest_divisor n = 35 ∧ ∀ (m : ℕ), m ≥ 2 ∧ second_largest_divisor m = 35 → n ≥ m :=
begin
  use 175,
  split,
  { -- n ≥ 2
    exact nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 0)) },
  split,
  { -- second_largest_divisor 175 = 35
    sorry },
  { -- ∀ (m : ℕ), m ≥ 2 ∧ second_largest_divisor m = 35 → 175 ≥ m
    sorry }
end

end largest_n_with_f_n_eq_35_l327_327866


namespace number_of_prime_pairs_for_10003_l327_327522

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327522


namespace diamond_computation_l327_327314

def diamond (a b : ℕ) : ℕ := 12 * a - 10 * b

theorem diamond_computation : (((((20 \Diamond 22) \Diamond 22) \Diamond 22) \Diamond 22) = 20 :=
by
  sorry

end diamond_computation_l327_327314


namespace range_of_k_l327_327623

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (h : ∀ x1 x2, 0 < x1 → 0 < x2 → g(x1) / k ≤ f(x2) / (k + 1)) : k ≥ 1 / (2 * Real.exp 1 - 1) :=
sorry

end range_of_k_l327_327623


namespace sqrt_eq_9_implies_n_eq_73_l327_327439

theorem sqrt_eq_9_implies_n_eq_73 (n : ℕ) : sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_eq_9_implies_n_eq_73_l327_327439


namespace sqrt_eight_plus_n_eq_nine_l327_327443

theorem sqrt_eight_plus_n_eq_nine (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := by
  sorry

end sqrt_eight_plus_n_eq_nine_l327_327443


namespace O_c_lies_on_PC_l327_327160

noncomputable theory

variables {P A B C O_a O_b O_c : Type}

def circumcenter_of (X Y Z : Type) : Type := 
sorry -- Placeholder definition for circumcenter

def lies_on (P : Type) (line : Type) : Prop := 
sorry -- Placeholder definition for a point lying on a line

def is_perpendicular (line1 line2 : Type) : Prop := 
sorry -- Placeholder definition for perpendicular lines

axiom circumcenter_property (X Y Z : Type) : 
    sorry -- Placeholder for circumcenter properties theorem

axiom perpendicular_property (P A B O_a O_b O_c : Type) :
  (circumcenter_of P B C = O_a) ∧ (lies_on O_a (P A)) →
  (circumcenter_of P C A = O_b) ∧ (lies_on O_b (P B)) →
  (is_perpendicular (P A) (O_b O_c)) ∧ 
  (is_perpendicular (P B) (O_c O_a)) →
  (lies_on O_c (P C))

theorem O_c_lies_on_PC (P A B C O_a O_b O_c : Type)
  (h1 : circumcenter_of P B C = O_a)
  (h2 : circumcenter_of P C A = O_b)
  (h3 : circumcenter_of P A B = O_c)
  (h4 : lies_on O_a (P A))
  (h5 : lies_on O_b (P B)) :
  lies_on O_c (P C) :=
begin
  sorry
end

end O_c_lies_on_PC_l327_327160


namespace trig_fraction_eq_two_thirds_l327_327071

theorem trig_fraction_eq_two_thirds (α : ℝ) 
  (h : let a := (Real.sin α, -2)
           b := (1, Real.cos α)
       in a.1 * b.1 + a.2 * b.2 = 0) :
  ( Real.sin α / (Real.sin α + Real.cos α) = 2 / 3) :=
sorry

end trig_fraction_eq_two_thirds_l327_327071


namespace find_x_if_orthogonal_l327_327423

-- Define the vectors in terms of Lean structures
def vector_a : Vector3 ℝ := ⟨2, -3, 1⟩
def vector_b (x : ℝ) : Vector3 ℝ := ⟨-4, 2, x⟩

-- Establish orthogonality condition (dot product being zero)
def orthogonal (v w : Vector3 ℝ) : Prop := inner_product v w = 0

theorem find_x_if_orthogonal : ∀ x : ℝ, orthogonal vector_a (vector_b x) → x = 14 :=
begin
  intros x h,
  -- assume inner_product definition is the dot product
  sorry
end

end find_x_if_orthogonal_l327_327423


namespace number_of_prime_pairs_for_10003_l327_327516

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327516


namespace find_C_l327_327289

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 330) : C = 30 := 
sorry

end find_C_l327_327289


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327476

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327476


namespace sum_of_shaded_cells_l327_327688

/-- Given a 3x3 grid with numbers from 1 to 9, and the sum of one diagonal is 7,
   and the sum of the other diagonal is 21, prove that the sum of the numbers
   in the five shaded cells is 25. -/
theorem sum_of_shaded_cells : 
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9) → 
  (grid 0 0 + grid 1 1 + grid 2 2 = 7) → 
  (grid 0 2 + grid 1 1 + grid 2 0 = 21) → 
  (grid 0 1 + grid 0 2 + grid 1 0 + grid 1 2 + grid 2 1 = 25) :=
begin
  intros,
  sorry
end

end sum_of_shaded_cells_l327_327688


namespace sin_theta_value_l327_327370

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end sin_theta_value_l327_327370


namespace functional_relationship_functional_relationship_maximum_daily_profit_l327_327763

-- Definitions for conditions in the problem
def sales_price_1_to_30 (x : ℕ) (h : 1 ≤ x ∧ x ≤ 30) : ℝ := 0.5 * x + 35
def sales_price_31_to_60 (x : ℕ) (h : 31 ≤ x ∧ x ≤ 60) : ℝ := 50
def sales_volume (x : ℕ) (h : 1 ≤ x ∧ x ≤ 60) : ℕ := 124 - 2 * x
def cost_price : ℝ := 30

-- Profit functions
def profit_1_to_30 (x : ℕ) (h : 1 ≤ x ∧ x ≤ 30) : ℝ :=
  let P := sales_price_1_to_30 x h
  let V := sales_volume x ⟨h.left, by linarith⟩
  (P - cost_price) * V

def profit_31_to_60 (x : ℕ) (h : 31 ≤ x ∧ x ≤ 60) : ℝ :=
  let P := sales_price_31_to_60 x h
  let V := sales_volume x ⟨by linarith, h.right⟩
  (P - cost_price) * V

-- Lean proof goals
theorem functional_relationship : 
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 30 → profit_1_to_30 x x.2 = - x*x + 52*x + 620 :=
by sorry

theorem functional_relationship' : 
  ∀ (x : ℕ), 31 ≤ x ∧ x ≤ 60 → profit_31_to_60 x x.2 = - 40*x + 2480 :=
by sorry

theorem maximum_daily_profit :
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 60 ∧ (
    (x ≤ 30 → profit_1_to_30 x x.2 = 1296 ∧ x = 26) ∧ 
    (31 ≤ x → profit_31_to_60 x x.2 = 1240 → x = 31)
  ) :=
by sorry

end functional_relationship_functional_relationship_maximum_daily_profit_l327_327763


namespace total_estate_value_l327_327627

theorem total_estate_value :
  ∃ (E : ℝ), ∀ (x : ℝ),
    (5 * x + 4 * x = (2 / 3) * E) ∧
    (E = 13.5 * x) ∧
    (wife_share = 3 * 4 * x) ∧
    (gardener_share = 600) ∧
    (nephew_share = 1000) →
    E = 2880 := 
by 
  -- Declarations
  let E : ℝ := sorry
  let x : ℝ := sorry
  
  -- Set up conditions
  -- Daughter and son share
  have c1 : 5 * x + 4 * x = (2 / 3) * E := sorry
  
  -- E expressed through x
  have c2 : E = 13.5 * x := sorry
  
  -- Wife's share
  have c3 : wife_share = 3 * (4 * x) := sorry
  
  -- Gardener's share and Nephew's share
  have c4 : gardener_share = 600 := sorry
  have c5 : nephew_share = 1000 := sorry
  
  -- Equate expressions and solve
  have eq1 : E = 21 * x + 1600 := sorry
  have eq2 : E = 2880 := sorry
  use E
  intro x
  -- Prove the equalities under the given conditions
  sorry

end total_estate_value_l327_327627


namespace selection_ways_l327_327364

theorem selection_ways (m f : ℕ) (h_m : m = 5) (h_f : f = 4) (n : ℕ) (h_n : n = 4) :
    let ways := (Nat.choose f 2 * Nat.choose m 2) 
                + (Nat.choose f 3 * Nat.choose m 1)
                + (Nat.choose f 4)
    in ways = 81 := by
    rw [h_m, h_f, h_n]
    simp only [Nat.choose]
    sorry

end selection_ways_l327_327364


namespace find_angle_F_l327_327580

-- Define the given conditions and the goal
variable (EF GH : ℝ) (angleE angleF angleG angleH : ℝ)
variable (h1 : EF ∥ GH) (h2 : angleE = 3 * angleH) (h3 : angleG = 2 * angleF) 

theorem find_angle_F (h_sum : angleF + angleG = 180) : angleF = 60 :=
by sorry

end find_angle_F_l327_327580


namespace parallel_lines_l327_327070

-- Definitions of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

-- Definition of parallel lines: slopes are equal and the lines are not identical
def slopes_equal (m : ℝ) : Prop := -(3 + m) / 4 = -2 / (5 + m)
def not_identical_lines (m : ℝ) : Prop := l1 m ≠ l2 m

-- Theorem stating the given conditions
theorem parallel_lines (m : ℝ) (x y : ℝ) : slopes_equal m → not_identical_lines m → m = -7 := by
  sorry

end parallel_lines_l327_327070


namespace gcf_of_lcm_9_15_and_10_21_is_5_l327_327730

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l327_327730


namespace number_of_prime_pairs_for_10003_l327_327515

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327515


namespace gcd_is_18_l327_327098

-- Define gcdX that represents the greatest common divisor of X and Y.
noncomputable def gcdX (X Y : ℕ) : ℕ := Nat.gcd X Y

-- Given conditions
def cond_lcm (X Y : ℕ) : Prop := Nat.lcm X Y = 180
def cond_ratio (X Y : ℕ) : Prop := ∃ k : ℕ, X = 2 * k ∧ Y = 5 * k

-- Theorem to prove that the gcd of X and Y is 18
theorem gcd_is_18 {X Y : ℕ} (h1 : cond_lcm X Y) (h2 : cond_ratio X Y) : gcdX X Y = 18 :=
by
  sorry

end gcd_is_18_l327_327098


namespace sequence_term_19th_l327_327066

/-- Define the sequence term. -/
def sequence_term (n : ℕ) : ℝ :=
  real.log (4 * n - 1)

/-- Given condition: 2*ln(5) + ln(3) -/
def lhs : ℝ :=
  2 * real.log 5 + real.log 3

/-- Proof statement: Show that lhs is equal to the term (nth term) -/
theorem sequence_term_19th :
  lhs = sequence_term 19 := by 
  sorry

end sequence_term_19th_l327_327066


namespace vectors_parallel_l327_327424

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

theorem vectors_parallel :
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  are_parallel a b :=
by
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  -- Proof omitted
  sorry

end vectors_parallel_l327_327424


namespace largest_possible_value_of_sum_of_products_l327_327694

open Finset

noncomputable def largest_sum_of_products : ℕ :=
  let s : Finset ℕ := {1, 2, 3, 4}
  let products := s.powerset.filter(λ t, t.card = 4).image(λ t,
    let ⟨a, b, c, d⟩ := ⟨t.to_list.nth 0, t.to_list.nth 1, t.to_list.nth 2, t.to_list.nth 3⟩ in
    option.get_or_else (a.feval nat 0 * b.feval nat 0 + b.feval nat 0 * c.feval nat 0 
    + c.feval nat 0 * d.feval nat 0 + d.feval nat 0 * a.feval nat 0) 0)
  products.max'

theorem largest_possible_value_of_sum_of_products (a b c d : ℕ) (h₁ : a ∈ {1, 2, 3, 4})
  (h₂ : b ∈ {1, 2, 3, 4}) (h₃ : c ∈ {1, 2, 3, 4}) (h₄ : d ∈ {1, 2, 3, 4})
  (h₅ : a ≠ b) (h₆ : b ≠ c) (h₇ : c ≠ d) (h₈ : d ≠ a) (h₉ : a ≠ c) (h₁₀ : b ≠ d) :
  ab + bc + cd + da = 25 :=
sorry

end largest_possible_value_of_sum_of_products_l327_327694


namespace number_of_prime_pairs_for_10003_l327_327523

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327523


namespace num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l327_327930

def total_students : ℕ := 800

def percentage_blue_shirts : ℕ := 45
def percentage_red_shirts : ℕ := 23
def percentage_green_shirts : ℕ := 15

def percentage_black_pants : ℕ := 30
def percentage_khaki_pants : ℕ := 25
def percentage_jeans_pants : ℕ := 10

def percentage_white_shoes : ℕ := 40
def percentage_black_shoes : ℕ := 20
def percentage_brown_shoes : ℕ := 15

def students_other_color_shirts : ℕ :=
  total_students * (100 - (percentage_blue_shirts + percentage_red_shirts + percentage_green_shirts)) / 100

def students_other_types_pants : ℕ :=
  total_students * (100 - (percentage_black_pants + percentage_khaki_pants + percentage_jeans_pants)) / 100

def students_other_color_shoes : ℕ :=
  total_students * (100 - (percentage_white_shoes + percentage_black_shoes + percentage_brown_shoes)) / 100

theorem num_students_other_color_shirts : students_other_color_shirts = 136 := by
  sorry

theorem num_students_other_types_pants : students_other_types_pants = 280 := by
  sorry

theorem num_students_other_color_shoes : students_other_color_shoes = 200 := by
  sorry

end num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l327_327930


namespace slices_per_person_l327_327705

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l327_327705


namespace max_children_left_of_xiaoxue_l327_327998

-- Define the set of children dressing numbers
def children := Finset.range 11

-- Define the set of odd and even numbers from 1 to 10
def odd_numbers := {x ∈ children | x % 2 = 1}
def even_numbers := {x ∈ children | x % 2 = 0}

-- Define the conditions
def is_child (x : ℕ) := x ∈ children ∧ 1 ≤ x ∧ x ≤ 10
def is_xiaoxue (x : ℕ) := is_child x

-- Define the proposition
def maximum_children_left_of_xiaoxue (x : ℕ) : Prop :=
  ∀ (l : Finset ℕ), (l ⊆ odd_numbers) ∧ (l.card ≤ x - 1) → l.card ≤ 5

-- Theorem stating the proof problem
theorem max_children_left_of_xiaoxue :
  ∀ x, is_xiaoxue x → maximum_children_left_of_xiaoxue x :=
sorry

end max_children_left_of_xiaoxue_l327_327998


namespace inequality_proof_l327_327155

variable {n : ℕ}
variable {a : ℕ → ℝ}

theorem inequality_proof (h1 : ∀ i, 0 ≤ a i) (h2 : 3 ≤ n) (h3 : (Finset.range n).sum a = 4) :
    ∑ i in Finset.range n, (a i) ^ 3 * a ((i + 1) % n) ≤ 27 := by
  sorry

end inequality_proof_l327_327155


namespace arithmetic_seq_sum_a4_a6_l327_327932

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum_a4_a6 (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_root1 : a 3 ^ 2 - 3 * a 3 + 1 = 0)
  (h_root2 : a 7 ^ 2 - 3 * a 7 + 1 = 0) :
  a 4 + a 6 = 3 :=
sorry

end arithmetic_seq_sum_a4_a6_l327_327932


namespace min_games_to_ensure_played_l327_327939

theorem min_games_to_ensure_played :
  ∀ (teams : Finset ℕ), teams.card = 20 → (∃ (games : ℕ), 
  (∀ (t : Finset ℕ) (h : t.card = 3), (∃ (x y : ℕ) (hx : x ∈ t) (hy : y ∈ t), x ≠ y ∧ played x y)) ∧ games = 90) :=
by
  intro teams h_teams_card
  use 90
  intros t h_t_card
  sorry

end min_games_to_ensure_played_l327_327939


namespace no_prime_sum_10003_l327_327472

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327472


namespace arlo_books_l327_327299

theorem arlo_books (total_items : ℕ) (books_ratio : ℕ) (pens_ratio : ℕ) (notebooks_ratio : ℕ) 
  (ratio_sum : ℕ) (items_per_part : ℕ) (parts_for_books : ℕ) (total_parts : ℕ) :
  total_items = 600 →
  books_ratio = 7 →
  pens_ratio = 3 →
  notebooks_ratio = 2 →
  total_parts = books_ratio + pens_ratio + notebooks_ratio →
  items_per_part = total_items / total_parts →
  parts_for_books = books_ratio →
  parts_for_books * items_per_part = 350 := by
  intros
  sorry

end arlo_books_l327_327299


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327480

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327480


namespace sum_to_product_identity_l327_327325

variable (x : ℝ)

theorem sum_to_product_identity : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sum_to_product_identity_l327_327325


namespace gcf_of_lcm_9_15_and_10_21_is_5_l327_327728

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l327_327728


namespace no_prime_sum_10003_l327_327539

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327539


namespace problem_statement_l327_327343

noncomputable def is_solution (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (n ^ 2 ∣ (nat.factorial (n - 2)))

noncomputable def expected_solutions : set ℕ :=
  {8, 9} ∪ set_of (nat.prime) ∪ { m | ∃ p, nat.prime p ∧ m = 2 * p }

theorem problem_statement :
  { n : ℕ | is_solution n } = expected_solutions := sorry

end problem_statement_l327_327343


namespace find_n_l327_327345

theorem find_n (n : ℕ) (h_pos : 0 < n) :
  967*1024 ≤ n ∧ n < 968*1024 ↔ (∀ k : ℕ, 0 < k → k^2 + ⌊ n / k^2 ⌋ ≠ 1991) := 
sorry

end find_n_l327_327345


namespace rectangle_area_l327_327801

-- Define the rectangle and given conditions
variables (A B C D F E : Type)
variables (AB BC BE CF : ℝ)
variables (x : ℝ)
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq F] [decidable_eq E]

-- Rectangle ABCD with given ratios and lengths
axiom AB_eq_2BC : AB = 2 * BC
axiom CF_eq_6 : CF = 6
axiom CF_eq_3BE : CF = 3 * BE
axiom BE_eq_2 : BE = 2

-- Prove that the area of the rectangle is 8 cm^2
theorem rectangle_area : AB * BC = 8 := by
sorry

end rectangle_area_l327_327801


namespace number_of_elements_in_A_int_set_l327_327067

def A (x : ℝ) : Prop := x^2 < 3 * x + 4

def A_int_set : Set ℤ := {n : ℤ | A n}

theorem number_of_elements_in_A_int_set : Finset.card (A_int_set.to_finset) = 4 := by
  sorry

end number_of_elements_in_A_int_set_l327_327067


namespace bernardo_wins_smallest_sum_l327_327112

theorem bernardo_wins_smallest_sum :
  ∃ M : ℕ, 0 ≤ M ∧ M ≤ 999 ∧ 900 ≤ 72 * M ∧ 72 * M ≤ 999 ∧ (M / 10 + M % 10 = 4) :=
begin
  use 13,
  split,
  { -- Prove 13 is a valid initial number M
    norm_num,
  },
  split,
  { -- Prove M is less than or equal to 999
    norm_num,
  },
  split,
  { -- Prove 900 <= 72 * 13
    norm_num,
  },
  split,
  { -- Prove 72 * 13 <= 999
    norm_num,
  },
  { -- Prove sum of digits of M is 4
    norm_num,
  }
end

end bernardo_wins_smallest_sum_l327_327112


namespace correct_option_C_l327_327392

-- Define points A, B and C given their coordinates and conditions
structure Point (α : Type _) :=
(x : α)
(y : α)

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

variables (a b c d : ℝ)
variable hA : Point ℝ := ⟨a, 2⟩
variable hB : Point ℝ := ⟨b, 6⟩
variable hC : Point ℝ := ⟨c, d⟩
variables (ha_ON_parabola : hA.y = parabola hA.x)
          (hb_ON_parabola : hB.y = parabola hB.x)
          (hc_ON_parabola : hC.y = parabola hC.x)
          (hd_lt_one : d < 1)

theorem correct_option_C (ha_lt_0 : a < 0) (hb_gt_0 : b > 0) : a < c ∧ c < b :=
by
-- Proof will be done here, currently left as sorry just to state the theorem.
sorry

end correct_option_C_l327_327392


namespace area_of_reflected_triangle_leq_original_l327_327973

-- Equilateral triangle with given points and reflections
variables (A B C P C₁ A₁ B₁ : Type) [EquilateralTriangle A B C] 
  (S S' : ℝ) 

-- Conditions
variables (H1 : PointInsideTriangle P A B C)
          (H2 : Reflection P (Line A B) C₁)
          (H3 : Reflection P (Line B C) A₁)
          (H4 : Reflection P (Line C A) B₁)
          (H5 : Area ΔABC = S)
          (H6 : Area ΔA₁B₁C₁ = S')

-- The statement we want to prove
theorem area_of_reflected_triangle_leq_original :
  S' ≤ S :=
sorry

end area_of_reflected_triangle_leq_original_l327_327973


namespace volume_ratio_cone_prism_l327_327783

variables (r h : ℝ) (π : ℝ)

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

def volume_prism (r h : ℝ) : ℝ := 6 * r^2 * h

theorem volume_ratio_cone_prism (r h : ℝ) (π_pos : π > 0) (r_pos : r > 0) (h_pos : h > 0) :
  (volume_cone π r h) / (volume_prism r h) = π / 18 :=
by
  sorry

end volume_ratio_cone_prism_l327_327783


namespace find_A_l327_327132

def point := ℝ × ℝ

def slope (p1 p2 : point) : ℝ := (p2.snd - p1.snd) / (p2.fst - p1.fst)

def midpoint (p1 p2 : point) : point := ((p1.fst + p2.fst) / 2, (p1.snd + p2.snd) / 2)

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

def line_through (p1 p2 : point) (m : ℝ) : Prop := slope p1 p2 = m

def equidistant (p : point) (p1 p2 : point) : Prop :=
  dist p p1 = dist p p2

theorem find_A :
  ∃ A : ℝ,
    let P : point := (0, A),
        Q : point := (12, 8),
        M : point := midpoint P Q,
        SlopePQ := slope P Q,
        PerpendicularSlope := -1 / SlopePQ,
        GivenSlope := 0.5 in
    is_perpendicular SlopePQ GivenSlope ∧
    line_through (4, 4) M PerpendicularSlope ∧
    equidistant (4, 4) P Q ∧
    A = 32 := sorry

end find_A_l327_327132


namespace bridge_length_l327_327685

theorem bridge_length (train_length : ℕ) (train_speed_kmh : ℕ) (time_s : ℕ) (bridge_length : ℕ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  time_s = 30 →
  bridge_length = 245 :=
by
  assume h1 : train_length = 130
  assume h2 : train_speed_kmh = 45
  assume h3 : time_s = 30
  sorry

end bridge_length_l327_327685


namespace distance_from_point_to_y_axis_l327_327869

/-- Proof that the distance from point P(-4, 3) to the y-axis is 4. -/
theorem distance_from_point_to_y_axis {P : ℝ × ℝ} (hP : P = (-4, 3)) : |P.1| = 4 :=
by {
   -- The proof will depend on the properties of absolute value
   -- and the given condition about the coordinates of P.
   sorry
}

end distance_from_point_to_y_axis_l327_327869


namespace line_intercepts_sum_zero_line_triangle_area_sixteen_l327_327030

/-- Given a line l passing through point (2, 3).
    If the sum of the intercepts of line l on the x-axis and y-axis equals 0,
    then the equation of line l is either 3x - 2y = 0 or x - y + 1 = 0. -/
theorem line_intercepts_sum_zero {l : LinearEquations ℝ} :
  pass_through l (2, 3) ∧ intercepts_sum_zero l → 
  equation l = "3x - 2y = 0" ∨ equation l = "x - y + 1 = 0" :=
sorry

/-- Given a line l passing through point (2, 3).
    If the area of the triangle formed by line l and the two coordinate axes 
    in the first quadrant is 16, then the equation of line l is either 
    x + 2y - 8 = 0 or 9x + 2y - 24 = 0. -/
theorem line_triangle_area_sixteen {l : LinearEquations ℝ} :
  pass_through l (2, 3) ∧ triangle_area l = 16 → 
  equation l = "x + 2y - 8 = 0" ∨ equation l = "9x + 2y - 24 = 0" :=
sorry

end line_intercepts_sum_zero_line_triangle_area_sixteen_l327_327030


namespace sin_sum_bound_l327_327145

theorem sin_sum_bound (n : ℕ) (a : ℕ → ℝ) (h₀ : ∑ i in Finset.range n, a i = 0) (h₁ : ∀ i, |a i| ≤ 1) :
  |∑ i in Finset.range n, (i + 1) * a i| ≤ ⌊(n^2 : ℝ) / 4⌋ :=
sorry

end sin_sum_bound_l327_327145


namespace find_a_of_odd_function_l327_327054

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a / (2^x + 1)

theorem find_a_of_odd_function (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = -2 :=
by
  sorry

end find_a_of_odd_function_l327_327054


namespace ap_minus_aq_eq_8_l327_327051

theorem ap_minus_aq_eq_8 (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) (p q : ℕ) 
  (h1 : ∀ n, S_n n = n^2 - 5 * n) 
  (h2 : ∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) 
  (h3 : p - q = 4) :
  a_n p - a_n q = 8 := sorry

end ap_minus_aq_eq_8_l327_327051


namespace solve_equation_l327_327826

theorem solve_equation : ∀ x : ℝ, (10 - x) ^ 2 = 4 * x ^ 2 ↔ x = 10 / 3 ∨ x = -10 :=
by
  intros x
  sorry

end solve_equation_l327_327826


namespace solve_problems_l327_327110

theorem solve_problems (x y : ℕ) (hx : x + y = 14) (hy : 7 * x - 12 * y = 60) : x = 12 :=
sorry

end solve_problems_l327_327110


namespace last_employee_number_l327_327273

theorem last_employee_number (total_employees : ℕ) (selected_employees : ℕ) (first_draw : ℕ) : 
  total_employees = 1000 → 
  selected_employees = 50 → 
  first_draw = 15 → 
  (first_draw + (selected_employees - 1) * (total_employees / selected_employees)) = 995 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end last_employee_number_l327_327273


namespace length_of_chord_in_circle_l327_327031

noncomputable def parabola_focus_distance (p x y : ℝ) : ℝ :=
  ((x - p / 2) ^ 2 + y ^ 2) ^ (1/2)

theorem length_of_chord_in_circle 
  (p a x y : ℝ) (h₀: p > 0) (C_center : (0, 4)) (h₁ : x^2 + (y-4)^2 = a^2)
  (h₂ : a = p / 4 + p / 2) (A : (x, y)) (h₃: y = 2 * sqrt 2 * x) 
  (h₄:  ((x - p / 2) ^ 2 + y ^ 2)^(1/2) = a) :
  2 * sqrt ((p^2 * 3 / 4) - (4 / 3)^2) = (7 * sqrt 2) / 3 :=
sorry

end length_of_chord_in_circle_l327_327031


namespace expression_increase_l327_327571

variable {x y : ℝ}

theorem expression_increase (hx : x > 0) (hy : y > 0) :
  let original_expr := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expr := 3 * new_x ^ 2 * new_y
  (new_expr / original_expr) = 3.456 :=
by
-- original_expr is 3 * x^2 * y
-- new_x = 1.2 * x
-- new_y = 2.4 * y
-- new_expr = 3 * (1.2 * x)^2 * (2.4 * y)
-- (new_expr / original_expr) = (10.368 * x^2 * y) / (3 * x^2 * y)
-- (new_expr / original_expr) = 10.368 / 3
-- (new_expr / original_expr) = 3.456
sorry

end expression_increase_l327_327571


namespace range_of_retained_superiority_after_5_trials_l327_327716

theorem range_of_retained_superiority_after_5_trials : 
  ∀ (original_range : ℝ), 
  (retained_superiority original_range 0.618 5) = original_range * 0.618^5 :=
by
  -- define the function of retained_superiority
  sorry

-- Define the retained_superiority function assuming it reduces the original range by a factor after each trial
noncomputable def retained_superiority (range : ℝ) (factor : ℝ) (trials : ℕ) : ℝ :=
  range * (factor ^ trials)

end range_of_retained_superiority_after_5_trials_l327_327716


namespace sum_of_primes_10003_l327_327553

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327553


namespace wire_length_for_max_area_is_36_l327_327761

-- Define the conditions as a structure
structure ProofConditions where
  r : ℝ
  wire_length : ℝ
  (maximum_area: wire_length = (Real.pi * r + 2 * r))

-- Define the problem as a theorem
theorem wire_length_for_max_area_is_36 (h : ProofConditions) : h.r = 7 → h.wire_length ≈ 36 :=
by
  intro hr_eq_7
  rw [hr_eq_7] at h
  have wire_length_eq := h.maximum_area
  sorry  -- Proof steps would go here, but are omitted.

end wire_length_for_max_area_is_36_l327_327761


namespace paint_amount_third_day_l327_327784

theorem paint_amount_third_day : 
  let initial_paint := 80
  let first_day_usage := initial_paint / 2
  let paint_after_first_day := initial_paint - first_day_usage
  let added_paint := 20
  let new_total_paint := paint_after_first_day + added_paint
  let second_day_usage := new_total_paint / 2
  let paint_after_second_day := new_total_paint - second_day_usage
  paint_after_second_day = 30 :=
by
  sorry

end paint_amount_third_day_l327_327784


namespace maximum_value_expression_l327_327367

theorem maximum_value_expression (a b : ℝ) (h : a^2 + b^2 = 9) : 
  ∃ x, x = 5 ∧ ∀ y, y = ab - b + a → y ≤ x :=
by
  sorry

end maximum_value_expression_l327_327367


namespace quadratic_solutions_l327_327922

theorem quadratic_solutions (a c : ℝ) (h : ∃ (a c : ℝ), a * (-1)^2 - 2 * a * (-1) + c = 0) :
  ∃ x1 x2 : ℝ, (ax^2 - 2ax + c) = 0 ↔ (x1 = -1 ∧ x2 = 3) :=
  by
  sorry

end quadratic_solutions_l327_327922


namespace g_x0_is_2_l327_327018

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + x - 4

-- Define the zero point of the function f
def x0 : ℝ := _
axiom zero_x0 : f x0 = 0

-- Define the floor function
def g (x : ℝ) : ℤ := Int.floor x

-- State the theorem we need to prove
theorem g_x0_is_2 : g x0 = 2 :=
sorry

end g_x0_is_2_l327_327018


namespace total_cases_after_three_days_l327_327172

def initial_cases : ℕ := 2000
def increase_rate : ℝ := 0.20
def recovery_rate : ℝ := 0.02

def day_cases (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_cases
  | n + 1 => 
      let prev_cases := day_cases n
      let new_cases := increase_rate * prev_cases
      let recovered := recovery_rate * prev_cases
      prev_cases + new_cases - recovered

theorem total_cases_after_three_days : day_cases 3 = 3286 := by sorry

end total_cases_after_three_days_l327_327172


namespace larger_angle_decrease_l327_327691

theorem larger_angle_decrease :
  ∀ (α β : ℝ),
  α + β = 90 ∧
  (3 / 2) * β = α ∧
  α * 0.2 + α = α' →
  (1 - (90 - α') / α) * 100 = 13.33 :=
by
  intros α β h1 h2 h3
  sorry

end larger_angle_decrease_l327_327691


namespace determinant_matrix_example_l327_327812

open Matrix

def matrix_example : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -2], ![-3, 6]]

noncomputable def compute_det_and_add_5 : ℤ := (matrix_example.det) + 5

theorem determinant_matrix_example :
  compute_det_and_add_5 = 41 := by
  sorry

end determinant_matrix_example_l327_327812


namespace number_of_prime_pairs_for_10003_l327_327521

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327521


namespace unique_sum_of_two_primes_l327_327562

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327562


namespace cannot_reach_1982_minimum_cost_to_1981_l327_327647

def starts_at_one : ℕ := 1
def multiply_by_3_cost : ℕ := 5
def add_4_cost : ℕ := 2
def target_1981 : ℕ := 1981
def target_1982 : ℕ := 1982

-- Question 1: Prove that 1982 cannot be reached from 1 using the given operations.
theorem cannot_reach_1982 : 
  ¬ (∃ (operations : list (ℕ → ℕ)), 
      (list.foldr (λ op acc, op acc) starts_at_one operations) = target_1982) :=
sorry

-- Question 2: Prove that the cost to obtain 1981 is 47 kopecks.
theorem minimum_cost_to_1981 : 
  (∃ (operations : list (ℕ → ℕ)), 
      (list.foldr (λ op acc, op acc) starts_at_one operations) = target_1981 ∧
      (list.foldr (λ op acc, op(acc)*multiply_by_3_cost + multiply_by_3_cost + add_4_cost) 0 operations) = 47) :=
sorry

end cannot_reach_1982_minimum_cost_to_1981_l327_327647


namespace interest_earned_l327_327966

-- Define the principal, interest rate, and number of years
def principal : ℝ := 1200
def annualInterestRate : ℝ := 0.12
def numberOfYears : ℕ := 4

-- Define the compound interest formula
def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the total interest earned
def totalInterest (P A : ℝ) : ℝ :=
  A - P

-- State the theorem
theorem interest_earned :
  totalInterest principal (compoundInterest principal annualInterestRate numberOfYears) = 688.224 :=
by
  sorry

end interest_earned_l327_327966


namespace count_pairs_l327_327080

theorem count_pairs :
  let pairs := [(a, b) | a b : ℕ+, a + b ≤ 200 ∧ (a : ℝ) + (1 / b : ℝ)) / (1 / (a : ℝ) + (b : ℝ)) = 9]
  pairs.length = 20 :=
by
  sorry

end count_pairs_l327_327080


namespace arithmetic_mean_n_numbers_l327_327311

theorem arithmetic_mean_n_numbers (n : ℕ) (h : n > 1) :
  let nums := (1 + 1 / n) :: List.replicate (n - 1) 1
  let sum_n := nums.foldl (+) 0
  let mean := sum_n / n
  mean = 1 + 1 / n^2 := by
  sorry

end arithmetic_mean_n_numbers_l327_327311


namespace smallest_integer_n_l327_327006

theorem smallest_integer_n (x y z w : ℝ) : 
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ ↑n * (x^4 + y^4 + z^4 + w^4)) ∧ 
    (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ ↑m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) :=
by {
  use 4,
  sorry -- Proof goes here
}

end smallest_integer_n_l327_327006


namespace sequence_has_multiple_forms_l327_327420

def sequence (n : ℕ) : ℤ :=
  if n % 2 = 0 then 1 else -1

theorem sequence_has_multiple_forms :
  (∀ n : ℕ, sequence n = (-1)^(n+1)) ∧ 
  (∀ n : ℕ, sequence n = if n % 2 = 1 then 1 else -1) ∧ 
  (∀ n : ℕ, sequence n = Int.ofNat (Real.toInt (Real.cos (n+1) * Real.pi)))
:= by
  sorry

end sequence_has_multiple_forms_l327_327420


namespace horner_evaluation_l327_327303

def f (x : ℤ) : ℤ := 3 * x ^ 4 + 5 * x ^ 3 + 6 * x ^ 2 + 79 * x - 8

theorem horner_evaluation : 
  let V2 : ℤ := let V0 := 3 in
                let V1 := (-4) * V0 + 5 in
                (-4) * V1 + 6
  in V2 = 34 :=
by
  let V0 := 3
  let V1 := (-4) * V0 + 5
  let V2 := (-4) * V1 + 6
  show V2 = 34
  sorry

end horner_evaluation_l327_327303


namespace sin_sum_to_product_l327_327337

theorem sin_sum_to_product (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by 
  sorry

end sin_sum_to_product_l327_327337


namespace number_of_proper_subsets_of_union_sets_l327_327923

theorem number_of_proper_subsets_of_union_sets : 
  ∀ (M N : Set ℕ), M = {1, 2} → N = {2, 3} → (M ∪ N).powerset.size = 7 + 1 → (M ∪ N).delete (M ∪ N).size = 7 :=
by
  intros M N hM hN hP
  sorry

end number_of_proper_subsets_of_union_sets_l327_327923


namespace johns_total_profit_l327_327951

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end johns_total_profit_l327_327951


namespace find_x_l327_327908

-- Defining vectors and the dot product
def vector (n : ℕ) := fin n → ℤ

def dot_product {n : ℕ} (v1 v2 : vector n) : ℤ := 
  finset.univ.sum (λ i, v1 i * v2 i)

-- Given conditions
def a : vector 2 := ![1, 2]
def b (x : ℤ) : vector 2 := ![-3, x]

-- Definition of the problem
theorem find_x (x : ℤ) (h : dot_product a (a - b x) = 0) : x = 4 :=
  sorry

end find_x_l327_327908


namespace smallest_beams_in_cube_l327_327296

-- Define the type of a beam and its orientations
structure Beam where
  x y z : ℕ
  direction : ℕ -- direction 0 for x-axis, 1 for y-axis, and 2 for z-axis

-- Define the conditions and question as a theorem
theorem smallest_beams_in_cube : 
  (∀ (b1 b2 : Beam), b1 ≠ b2 → 
    ¬ (b1.x = b2.x ∧ b1.y = b2.y ∧ b1.z = b2.z ∧ b1.direction = b2.direction)
   ∧
   (b1.direction = 0 → 
     (b1.z = 0 ∧ b1.z + 2020 = 2020))
   ∧
   (b1.direction = 1 → 
     (b1.y = 0 ∧ b1.y + 2020 = 2020))
   ∧
   (b1.direction = 2 → 
     (b1.x = 0 ∧ b1.x + 2020 = 2020))) →
  (smallest_positive_number_of_beams : ℕ, smallest_positive_number_of_beams = 3030) := 
  sorry

end smallest_beams_in_cube_l327_327296


namespace unique_sum_of_two_primes_l327_327561

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327561


namespace slices_per_person_l327_327710

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l327_327710


namespace isosceles_trapezoid_rotation_l327_327657

-- Definitions
def is_isosceles_trapezoid (t : Type) := sorry
def axis_of_symmetry (t : Type) := sorry
def rotate_around_axis (shape : Type) (axis : Type) := sorry
def frustum (s : Type) := sorry

-- Main theorem
theorem isosceles_trapezoid_rotation (t : Type)
  (h_isosceles: is_isosceles_trapezoid t)
  (h_axis: axis_of_symmetry t) :
  frustum (rotate_around_axis t h_axis) :=
sorry

end isosceles_trapezoid_rotation_l327_327657


namespace hyperbola_eccentricity_l327_327680

-- Let's define the variables and conditions first
variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variable (h_asymptote : b = a)

-- We need to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l327_327680


namespace find_t_l327_327170

-- Given conditions
def hours_worked_me (t : ℝ) := t - 4
def earnings_per_hour_me (t : ℝ) := 3t - 7
def hours_worked_sarah (t : ℝ) := t - 2
def earnings_per_hour_sarah (t : ℝ) := t + 1
def total_earnings_me (t : ℝ) := hours_worked_me t * earnings_per_hour_me t
def total_earnings_sarah (t : ℝ) := hours_worked_sarah t * earnings_per_hour_sarah t

-- Question translated into a proof problem
theorem find_t (t : ℝ) : total_earnings_me t = total_earnings_sarah t → t = 5 := by
  sorry

end find_t_l327_327170


namespace cubic_polynomial_root_l327_327677

theorem cubic_polynomial_root (a b c : ℕ) (h : 27 * x^3 - 9 * x^2 - 9 * x - 3 = 0) : 
  (a + b + c = 11) :=
sorry

end cubic_polynomial_root_l327_327677


namespace proof_problem_l327_327061

open Real -- Open the real numbers namespace

-- Definition of the problem
def function_f (x θ : ℝ) : ℝ := 2 * sin x * cos (θ / 2) ^ 2 + cos x * sin θ - sin x

-- Definitions to encapsulate the given conditions
def condition_1 : Prop := ∀ (θ : ℝ), (0 < θ) ∧ (θ < π)
def condition_2 : Prop := function_f π = (-1)
def condition_3 : Prop := ∀ (A : ℝ), (cos A = sqrt 3 / 2)
def condition_4 : Prop := ∀ (a b : ℝ), (a = 1) ∧ (b = sqrt 2)

-- Main theorem to prove the questions given the conditions
theorem proof_problem (θ : ℝ) (A B C : ℝ) (a b : ℝ)
    (h0 : condition_1 θ)
    (h1 : condition_2 θ)
    (h2 : condition_3 A)
    (h3 : condition_4 a b) :
    (θ = π / 2) ∧ (C = π - A - B ∧ (C = 7 * π / 12 ∨ C = π / 12)) :=
begin
    sorry -- Skip the proof
end

end proof_problem_l327_327061


namespace prime_sum_10003_l327_327526

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327526


namespace balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l327_327013

noncomputable def ways_with_ball_in_box_one : Nat := 369
noncomputable def ways_with_two_empty_boxes : Nat := 360
noncomputable def ways_with_three_empty_boxes : Nat := 140
noncomputable def ways_ball_A_not_less_than_B : Nat := 375

theorem balls_in_boxes_with_one_in_one 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_1 : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_1 = 1 → 
  ∃ ways, ways = ways_with_ball_in_box_one := 
sorry

theorem balls_in_boxes_with_two_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 2 → 
  ∃ ways, ways = ways_with_two_empty_boxes := 
sorry

theorem balls_in_boxes_with_three_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 3 → 
  ∃ ways, ways = ways_with_three_empty_boxes := 
sorry

theorem balls_in_boxes_A_not_less_B 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_A : Nat) (ball_B : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_A ≠ ball_B →
  ∃ ways, ways = ways_ball_A_not_less_than_B := 
sorry

end balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l327_327013


namespace triangle_isosceles_or_right_angle_l327_327573

noncomputable theory

variables {A B C D E : Type} [EuclideanSpace ℝ (E A)]
variables [Inhabited E]

-- assumptions
variables (triangle_ABC : EuclideanTriangle A B C)
variables (midpoint_D : IsMidpoint (LineSegment B C) D)
variables (foot_E : IsFootPerpendicular C (LineSegment A D) E)
variables (equal_angle : Angle A C E = Angle A B C)

-- desired property to prove
theorem triangle_isosceles_or_right_angle :
  IsIsoscelesTriangle A B C ∨ IsRightAngle (Angle A B C) :=
sorry

end triangle_isosceles_or_right_angle_l327_327573


namespace csc_four_pi_over_three_l327_327302

theorem csc_four_pi_over_three : Real.csc (4 * Real.pi / 3) = - (2 * Real.sqrt 3) / 3 :=
by sorry

end csc_four_pi_over_three_l327_327302


namespace sum_of_two_primes_unique_l327_327484

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327484


namespace no_prime_sum_10003_l327_327542

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327542


namespace different_colors_probability_l327_327104

-- Definitions of the chips in the bag
def purple_chips := 7
def green_chips := 6
def orange_chips := 5
def total_chips := purple_chips + green_chips + orange_chips

-- Calculating probabilities for drawing chips of different colors and ensuring the final probability of different colors is correct
def probability_different_colors : ℚ :=
  let P := purple_chips
  let G := green_chips
  let O := orange_chips
  let T := total_chips
  (P / T) * ((G + O) / T) + (G / T) * ((P + O) / T) + (O / T) * ((P + G) / T)

theorem different_colors_probability : probability_different_colors = (107 / 162) := by
  sorry

end different_colors_probability_l327_327104


namespace find_f_expression_find_axis_of_symmetry_find_symmetric_point_l327_327892

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 6)
def M : ℝ × ℝ := (2 * π / 3, -2)
def T := π

-- Conditions
axiom A_pos : A > 0
axiom omega_pos : ω > 0
axiom phi_pos : 0 < φ ∧ φ < π / 2
axiom intersections : ∀ n : ℤ, f (n * π / 2) = 0
axiom lowest_point : f (2 * π / 3) = -2

-- Proof problem statements
theorem find_f_expression : 
  ∃ A ω φ, (A > 0 ∧ ω > 0 ∧ 0 < φ ∧ φ < π / 2 ∧
  f (2 * π / 3) = -2 ∧ 
  ∀ n : ℤ, f (n * π / 2) = 0) →
  f = fun x => 2 * Real.sin (2 * x + π / 6) := 
sorry

theorem find_axis_of_symmetry : 
  ∃ x, x ∈ Icc (π / 6) (2 * π / 3) ∧ g x = 2 * Real.sin (2 * x - π / 6) →
  x = π / 3 := 
sorry

theorem find_symmetric_point : 
  ∃ x, x ∈ Icc (π / 6) (2 * π / 3) ∧ g x = 2 * Real.sin (2 * x - π / 6) →
  (x, g x) = (7 * π / 12, 0) := 
sorry

end find_f_expression_find_axis_of_symmetry_find_symmetric_point_l327_327892


namespace slices_per_person_l327_327709

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l327_327709


namespace find_line_equation1_find_line_equation2_l327_327378

noncomputable def line_through_point (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem find_line_equation1 (a b c : ℝ) :
  line_through_point (-2) 1 a b c →
  (∀ (x1 y1 x2 y2 : ℝ), line_through_point x1 y1 a b c → line_through_point x2 y2 a b c → abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2) = abs (a * x2 + b * y2 + c) / sqrt (a^2 + b^2)) →
  (line_through_point (-5) 4 a b c ∧ line_through_point 3 2 a b c) →
  (a = 1 ∧ b = 4 ∧ c = -2) ∨ (a = 2 ∧ b = -1 ∧ c = 5) :=
by sorry

theorem find_line_equation2 (a b c : ℝ) :
  line_through_point (-2) 1 a b c →
  abs (0.5 * abs (a * b)) = 0.5 →
  (a = -1 ∧ b = -1 ∧ c = -1) ∨ (a = 1 ∧ b = 4 ∧ c = -2) :=
by sorry

end find_line_equation1_find_line_equation2_l327_327378


namespace contrapositive_proof_l327_327676

theorem contrapositive_proof (a b : ℕ) : (a = 1 ∧ b = 2) → (a + b = 3) :=
by {
  sorry
}

end contrapositive_proof_l327_327676


namespace diophantine_solution_l327_327190

theorem diophantine_solution :
  ∃ (x y k : ℤ), 1990 * x - 173 * y = 11 ∧ x = -22 + 173 * k ∧ y = 253 - 1990 * k :=
by {
  sorry
}

end diophantine_solution_l327_327190


namespace div_neg_neg_eq_pos_l327_327810

theorem div_neg_neg_eq_pos (x y : ℕ) (hx : x = 81) (hy : y = 9) : (-81) / (-9) = 9 :=
by
  rw [hx, hy]
  rw [Int.neg_div_neg_eq]
  rw [Int.div_eq, Int.ofNat_add, Int.coe_nat_div, Nat.div_eq, Int.ofNat_one]
  norm_num
  norm_num
  sorry

end div_neg_neg_eq_pos_l327_327810


namespace find_white_balls_l327_327105

theorem find_white_balls (a : ℕ) (h1 : 3 / (a + 3) = 0.20) : a = 12 :=
by
  sorry

end find_white_balls_l327_327105


namespace distance_between_lines_l327_327749

-- Definitions according to the conditions
variable {M : Point} -- Point M
variable {m n : Line} -- Lines m and n
variable {α : Plane} -- Plane passing through lines m and n

-- Given distances from the condition
def distance_M_m := 5
def distance_M_n := 4
def distance_M_plane := 3

-- Hypotheses based on the conditions
variable (h1 : M.distance_to(m) = distance_M_m)
variable (h2 : M.distance_to(n) = distance_M_n)
variable (h3 : M.distance_to(α) = distance_M_plane)
variable (m_parallel_n : m ∥ n)

-- Goal: Prove the distance between lines m and n
theorem distance_between_lines : 
  ∃ d, d = 4 + Real.sqrt 7 ∨ d = 4 - Real.sqrt 7 :=
sorry

end distance_between_lines_l327_327749


namespace max_value_of_f_l327_327003

def f (a b : ℝ) : ℝ := (|7 * a + 8 * b - a * b| + |2 * a + 8 * b - 6 * a * b|) / (a * sqrt (1 + b^2))

theorem max_value_of_f :
  (∀ a b : ℝ, a ≥ 1 → b ≥ 1 → f a b ≤ 9 * sqrt 2) ∧ 
  (∃ a b : ℝ, a ≥ 1 ∧ b ≥ 1 ∧ f a b = 9 * sqrt 2) := by
  sorry

end max_value_of_f_l327_327003


namespace percentage_reduction_is_correct_l327_327083

def percentage_reduction_alcohol_concentration (V_original V_added : ℚ) (C_original : ℚ) : ℚ :=
  let V_total := V_original + V_added
  let Amount_alcohol := V_original * C_original
  let C_new := Amount_alcohol / V_total
  ((C_original - C_new) / C_original) * 100

theorem percentage_reduction_is_correct :
  percentage_reduction_alcohol_concentration 12 28 0.20 = 70 := by
  sorry

end percentage_reduction_is_correct_l327_327083


namespace new_student_weight_l327_327197

theorem new_student_weight :
  ∀ (W : ℝ) (total_weight_19 : ℝ) (total_weight_20 : ℝ),
    total_weight_19 = 19 * 15 →
    total_weight_20 = 20 * 14.8 →
    total_weight_19 + W = total_weight_20 →
    W = 11 :=
by
  intros W total_weight_19 total_weight_20 h1 h2 h3
  -- Skipping the proof as instructed
  sorry

end new_student_weight_l327_327197


namespace height_of_right_prism_l327_327033

theorem height_of_right_prism 
  (h : ℝ) 
  (S₁ S₂ : ℝ) 
  (volume : ℝ)
  (h_S₁ : S₁ = 1)
  (h_S₂ : S₂ = 4)
  (h_volume : volume = 1 / 3)
  (h_volume_formula : volume = (1 / 3) * (S₁ + S₂ + real.sqrt (S₁ * S₂)) * h) : 
  h = 1 / 7 :=
by
  sorry

end height_of_right_prism_l327_327033


namespace alice_minimum_speed_l327_327678

noncomputable def minimum_speed_to_exceed (d t_bob t_alice : ℝ) (v_bob : ℝ) : ℝ :=
  d / t_alice

theorem alice_minimum_speed (d : ℝ) (v_bob : ℝ) (t_lag : ℝ) (v_alice : ℝ) :
  d = 30 → v_bob = 40 → t_lag = 0.5 → v_alice = d / (d / v_bob - t_lag) → v_alice > 60 :=
by
  intros hd hv hb ht
  rw [hd, hv, hb] at ht
  simp at ht
  sorry

end alice_minimum_speed_l327_327678


namespace complement_of_log_set_l327_327454

-- Define the set A based on the logarithmic inequality condition
def A : Set ℝ := { x : ℝ | Real.log x / Real.log (1 / 2) ≥ 2 }

-- Define the complement of A in the real numbers
noncomputable def complement_A : Set ℝ := { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 }

-- The goal is to prove the equivalence
theorem complement_of_log_set :
  complement_A = { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 } :=
by
  sorry

end complement_of_log_set_l327_327454


namespace angle_between_vectors_l327_327878

open Real

variables {a b : ℝ^3}

def magnitude (v : ℝ^3) := sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v w : ℝ^3) := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def angle (v w : ℝ^3) := 
  let cos_theta := dot_product v w / (magnitude v * magnitude w)
  in real.arccos cos_theta * (180 / real.pi)

theorem angle_between_vectors :
  (magnitude a = 2) → 
  (magnitude b = 1) → 
  (dot_product (a - b) b = 0) → 
  angle a b = 60 :=
by
  intros h1 h2 h3
  -- Proof steps would go here.
  -- We insert 'sorry' as instructed to skip the proof.
  sorry

end angle_between_vectors_l327_327878


namespace trig_sum_identity_l327_327406

theorem trig_sum_identity (x : ℝ) (h : sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2) :
  ∃ a b c : ℤ, (cos (a * x) * cos (b * x) * cos (c * x)) = 0 ∧ a + b + c = 14 :=
sorry

end trig_sum_identity_l327_327406


namespace maximoff_monthly_bill_l327_327626

theorem maximoff_monthly_bill 
    (initial_bill : ℝ)
    (electricity_increase : ℝ)
    (internet_increase : ℝ)
    (faster_internet : ℝ)
    (cloud_storage : ℝ)
    (virtual_desktop : ℝ) :
    initial_bill = 60 →
    electricity_increase = 0.45 →
    internet_increase = 0.30 →
    faster_internet = 25 →
    cloud_storage = 15 →
    virtual_desktop = 35 →
    let new_electricity_bill := initial_bill * (1 + electricity_increase) in
    let new_internet_bill := initial_bill * (1 + internet_increase) in
    new_electricity_bill + new_internet_bill + faster_internet + cloud_storage + virtual_desktop = 240 :=
begin
  intros,
  sorry
end

end maximoff_monthly_bill_l327_327626


namespace three_digit_numbers_product_36_l327_327428

/-- Number of 3-digit positive integers with digits whose product equals 36 --/
theorem three_digit_numbers_product_36 :
  {n : ℕ // ∃ a b c : ℕ, (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (a * b * c = 36) ∧ (n = 100 * a + 10 * b + c)}.card = 21 :=
sorry

end three_digit_numbers_product_36_l327_327428


namespace product_real_imaginary_part_eq_neg_two_l327_327888

theorem product_real_imaginary_part_eq_neg_two :
  let z := (2 + complex.I) * complex.I in
  (z.re * z.im = -2) :=
by
  let z := (2 + complex.I) * complex.I
  sorry

end product_real_imaginary_part_eq_neg_two_l327_327888


namespace find_y_l327_327777

theorem find_y (x : ℝ) (h1 : x = 1.3333333333333333) (h2 : (x * y) / 3 = x^2) : y = 4 :=
by 
  sorry

end find_y_l327_327777


namespace Q_value_is_nine_l327_327827

noncomputable def P : ℕ := sorry
noncomputable def Q : ℕ := sorry
noncomputable def R : ℕ := sorry
noncomputable def S : ℕ := sorry
noncomputable def T : ℕ := sorry
noncomputable def U : ℕ := sorry

axiom points_unique_digits : ∀ {x y : ℕ}, x ≠ y → (x ∈ {P, Q, R, S, T, U}) ↔ ¬(y ∈ {P, Q, R, S, T, U})
axiom points_range : ∀ {x : ℕ}, x ∈ {P, Q, R, S, T, U} → 1 ≤ x ∧ x ≤ 9

axiom total_sum_conditions : 
  (P + Q + R) + 
  (P + S + U) + 
  (R + T + U) + 
  (Q + T) + 
  (Q + S) + 
  (S + U) = 
  100

theorem Q_value_is_nine : Q = 9 :=
sorry

end Q_value_is_nine_l327_327827


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327475

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327475


namespace sin_2theta_in_third_quadrant_l327_327880

open Real

variables (θ : ℝ)

/-- \theta is an angle in the third quadrant.
Given that \(\sin^{4}\theta + \cos^{4}\theta = \frac{5}{9}\), 
prove that \(\sin 2\theta = \frac{2\sqrt{2}}{3}\). --/
theorem sin_2theta_in_third_quadrant (h_theta_third_quadrant : π < θ ∧ θ < 3 * π / 2)
(h_cond : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : sin (2 * θ) = 2 * sqrt 2 / 3 :=
sorry

end sin_2theta_in_third_quadrant_l327_327880


namespace f_at_3_l327_327885

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end f_at_3_l327_327885


namespace no_prime_sum_10003_l327_327535

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327535


namespace percent_decrease_correct_l327_327926

-- Define the conversion rate and costs in GBP
def cost1990_GBP : ℝ := 0.75  -- 75 pence converted to GBP
def cost2010_GBP : ℝ := 0.10  -- 10 pence converted to GBP
def exchange_rate : ℝ := 1.5  -- 1 GBP to USD conversion rate

-- Calculate the costs in USD
def cost1990_USD : ℝ := cost1990_GBP * exchange_rate
def cost2010_USD : ℝ := cost2010_GBP * exchange_rate

-- Calculate the percent decrease
def percent_decrease : ℝ := ((cost1990_USD - cost2010_USD) / cost1990_USD) * 100

-- Prove that the percent decrease is approximately 87%
theorem percent_decrease_correct : abs (percent_decrease - 87) < 1 := by
  sorry

end percent_decrease_correct_l327_327926


namespace sin_sum_to_product_l327_327333

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sin_sum_to_product_l327_327333


namespace equilateral_triangles_are_similar_l327_327741

def are_similar (A B : Type) [metric_space A] [metric_space B] : Prop :=
∃(λ f: A → B, bijective f ∧ ∀x y: A, dist (f x) (f y) = dist x y)

def isosceles_triangle : Type := {T : triangle // T.a = T.b ∧ T.c ≠ T.a}
def right_angled_triangle : Type := {T : triangle // T.α = 90 ∨ T.β = 90 ∨ T.γ = 90}
def equilateral_triangle : Type := {T : triangle // T.a = T.b ∧ T.b = T.c}
def rhombus : Type := {R : quadrilateral // R.a = R.b ∧ R.b = R.c ∧ R.c = R.d }

theorem equilateral_triangles_are_similar : 
  ∀(T1 T2 : equilateral_triangle), are_similar T1 T2 :=
begin
  sorry
end

end equilateral_triangles_are_similar_l327_327741


namespace students_going_to_tournament_l327_327630

theorem students_going_to_tournament :
  ∀ (total_students : ℕ) (one_third : ℚ) (half : ℚ),
    total_students = 24 →
    one_third * total_students = 8 →
    half * 8 = 4 →
    4 = 4 :=
by
  intros total_students one_third half h1 h2 h3
  exact h3.symm

end students_going_to_tournament_l327_327630


namespace prob_xi_ge_2_eq_one_third_l327_327991

noncomputable def pmf (c k : ℝ) : ℝ := c / (k * (k + 1))

theorem prob_xi_ge_2_eq_one_third 
  (c : ℝ) 
  (h₁ : pmf c 1 + pmf c 2 + pmf c 3 = 1) :
  pmf c 2 + pmf c 3 = 1 / 3 :=
by
  sorry

end prob_xi_ge_2_eq_one_third_l327_327991


namespace donkey_wins_with_optimal_strategy_l327_327856

theorem donkey_wins_with_optimal_strategy :
  ∀ (points : Finset (Fin (2005))), 
  (∀ (a b c : Fin (2005)), a ≠ b → b ≠ c → a ≠ c → ¬((a, b, c).collinear)) →
  (∀ (a b : Fin (2005)), a ≠ b → (a, b) ∈ points) →
  (∃ (segment_label : (Fin (2005) × Fin (2005)) → Fin 2) 
      (point_label : Fin (2005) → Fin 2), 
    ∃ (a b : Fin (2005)), 
      a ≠ b ∧ segment_label (a, b) = point_label a ∧ point_label a = point_label b) :=
sorry

end donkey_wins_with_optimal_strategy_l327_327856


namespace area_bounded_by_arcsin_cos_l327_327832

noncomputable def function_to_integrate (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ π then (π / 2 - x) else if π < x ∧ x ≤ 2 * π then (x - 3 * π / 2) else 0

theorem area_bounded_by_arcsin_cos :
  (∫ x in 0..(2 * π), function_to_integrate x) = (π^2 / 2) :=
by
  sorry

end area_bounded_by_arcsin_cos_l327_327832


namespace power_addition_rule_l327_327087

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l327_327087


namespace find_angle_F_l327_327587

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l327_327587


namespace book_sale_total_amount_l327_327268

noncomputable def total_amount_received (total_books price_per_book : ℕ → ℝ) : ℝ :=
  price_per_book 80

theorem book_sale_total_amount (B : ℕ)
  (h1 : (1/3 : ℚ) * B = 40)
  (h2 : ∀ (n : ℕ), price_per_book n = 3.50) :
  total_amount_received B price_per_book = 280 := 
by
  sorry

end book_sale_total_amount_l327_327268


namespace gcf_of_lcm_eq_15_l327_327724

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_of_lcm_eq_15 : gcf (lcm 9 15) (lcm 10 21) = 15 := by
  sorry

end gcf_of_lcm_eq_15_l327_327724


namespace no_prime_sum_10003_l327_327506

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327506


namespace sin_sum_to_product_identity_l327_327327

theorem sin_sum_to_product_identity (x : ℝ) :
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin_sum_to_product_identity_l327_327327


namespace find_angle_F_l327_327586

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l327_327586


namespace sum_to_product_identity_l327_327326

variable (x : ℝ)

theorem sum_to_product_identity : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sum_to_product_identity_l327_327326


namespace feeding_solutions_count_l327_327790

def Animal : Type :=
| lion : bool → Animal
| tiger : bool → Animal
| other : ℕ → bool → Animal -- Assuming other species are indexed by ℕ for demonstration

def valid_feeding (feeding_seq : List Animal) : Prop :=
  feeding_seq.head = Animal.lion true ∧
  (∀ (i : ℕ), i < feeding_seq.length - 1 →
    (feeding_seq.nth i).is_some →
    (feeding_seq.nth i.succ).is_some →
    match (feeding_seq.nth i).iget, (feeding_seq.nth i.succ).iget with
    | Animal.lion male, _ =>
      male = true ∧ feeding_seq.nth i.succ ≠ some (Animal.tiger false)
    | Animal.tiger male, _ =>
      male = true ∧ feeding_seq.nth i.succ ≠ some (Animal.lion false)
    | _, _ => true
    end) ∧
  (∀ (i : ℕ), i + 2 < feeding_seq.length →
    feeding_seq.nth i.is_some →
    feeding_seq.nth (i + 2).is_some →
    (feeding_seq.nth i).iget.is_male = (feeding_seq.nth (i + 2)).iget.is_male)

def count_valid_feedings : ℕ :=
  5 * 5 * 4 * 4 * 3 * 3 * 2 * 2 * 1 * 1

theorem feeding_solutions_count :
  (∑ (seq : List Animal),
      if valid_feeding seq then 1 else 0) = 14400 := sorry

end feeding_solutions_count_l327_327790


namespace no_prime_sum_10003_l327_327465

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327465


namespace edric_hourly_rate_l327_327828

def total_earnings_before_deductions (base_salary commission bonus : ℝ) : ℝ :=
  base_salary + commission + bonus

def total_earnings_after_deductions (earnings deductions : ℝ) : ℝ :=
  earnings - deductions

def total_hours_worked (hours_per_day days_per_week weeks_in_month : ℕ) : ℕ :=
  hours_per_day * days_per_week * weeks_in_month

def hourly_rate (total_earnings : ℝ) (total_hours : ℕ) : ℝ :=
  total_earnings / total_hours

theorem edric_hourly_rate :
  let base_salary := 576
  let total_sales := 4000
  let commission := 0.03 * total_sales
  let bonus := 75
  let deductions := 30
  let hours_per_day := 8
  let days_per_week := 6
  let weeks_in_month := 4
  let earnings_before_deductions := total_earnings_before_deductions base_salary commission bonus
  let earnings_after_deductions := total_earnings_after_deductions earnings_before_deductions deductions
  let total_hours := total_hours_worked hours_per_day days_per_week weeks_in_month
  hourly_rate earnings_after_deductions total_hours = 3.86 :=
by
  sorry

end edric_hourly_rate_l327_327828


namespace police_officers_on_duty_l327_327645

theorem police_officers_on_duty (F : ℕ) (hF : F = 300)
  (h1 : ∃ f : ℕ, f = 0.40 * F)
  (h2 : ∃ d : ℕ, d = 2 * (f : ℕ) ∧ f = d / 2) :
  d = 240 :=
by
  sorry

end police_officers_on_duty_l327_327645


namespace students_going_to_tournament_l327_327629

theorem students_going_to_tournament :
  ∀ (total_students : ℕ) (one_third : ℚ) (half : ℚ),
    total_students = 24 →
    one_third * total_students = 8 →
    half * 8 = 4 →
    4 = 4 :=
by
  intros total_students one_third half h1 h2 h3
  exact h3.symm

end students_going_to_tournament_l327_327629


namespace nine_points_configuration_l327_327177

theorem nine_points_configuration (points : Finset (Point ℝ 2)) 
    (h_card : points.card = 9) 
    (h_no_four_collinear : ¬ ∃ (S : Finset (Point ℝ 2)), S ⊆ points ∧ S.card = 4 ∧ Collinear ℝ S) 
    (h_any_six_three_collinear : ∀ (S : Finset (Point ℝ 2)), S ⊆ points ∧ S.card = 6 → ∃ (T : Finset (Point ℝ 2)), 
        T ⊆ S ∧ T.card = 3 ∧ Collinear ℝ T) : 
    True :=
by
  sorry

end nine_points_configuration_l327_327177


namespace pure_imaginary_a_l327_327921

theorem pure_imaginary_a (a : ℝ) :
  (a^2 - 4 = 0) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
by
  sorry

end pure_imaginary_a_l327_327921


namespace minimum_value_fraction_1_x_plus_1_y_l327_327260

theorem minimum_value_fraction_1_x_plus_1_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) :
  1 / x + 1 / y = 1 :=
sorry

end minimum_value_fraction_1_x_plus_1_y_l327_327260


namespace find_f_of_x_l327_327617

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_x :
  (∀ x : ℝ, x ≠ 3 / 2 → f(x) + f((2 * x - 1) / (3 - 2 * x)) = 3 * x) →
  f(3) = 65 / 22 :=
by
  intro h
  sorry

end find_f_of_x_l327_327617


namespace sin_sum_to_product_l327_327338

theorem sin_sum_to_product (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by 
  sorry

end sin_sum_to_product_l327_327338


namespace no_prime_sum_10003_l327_327510

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327510


namespace piecewise_function_value_l327_327893

def f (x : ℝ) : ℝ :=
  if x < 1 then x + 1 else -x + 3

theorem piecewise_function_value :
  f (f (5 / 2)) = 3 / 2 :=
by
  sorry

end piecewise_function_value_l327_327893


namespace find_a_b_l327_327894

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

theorem find_a_b 
  (a b : ℝ) 
  (h_tangent : ∀ x y, y = 2 * x - 5 → y = f 1 a b - 3) 
  : a = -1 ∧ b = -3 :=
by 
{
  sorry
}

end find_a_b_l327_327894


namespace complement_intersection_in_U_l327_327994

universe u

variables {α : Type u} (U A B : Set α)

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_in_U : U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_intersection_in_U_l327_327994


namespace sphere_equiv_l327_327969

variables {A B C D G B' C' D' G_A G_B G_C G_D : Point}
variable (tetra : Tetrahedron A B C D)
variable (midB' : Midpoint B' A B)
variable (midC' : Midpoint C' A C)
variable (midD' : Midpoint D' A D)
variable (baryG_A : Barycenter G_A B C D)
variable (baryG_B : Barycenter G_B A C D)
variable (baryG_C : Barycenter G_C A B D)
variable (baryG_D : Barycenter G_D A B C)
variable (baryG : Barycenter G A B C D)

theorem sphere_equiv (h₁ : Spherical A G G_B G_C G_D) :
  Spherical A G B' C' D' ↔ Spherical A G G_B G_C G_D := sorry

end sphere_equiv_l327_327969


namespace max_expression_bound_l327_327002

theorem max_expression_bound (a b : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b) : 
  let expr := (|7 * a + 8 * b - a * b| + |2 * a + 8 * b - 6 * a * b|) / (a * real.sqrt (1 + b^2))
  in expr ≤ 9 * real.sqrt 2 :=
  sorry

end max_expression_bound_l327_327002


namespace distinct_real_numbers_product_l327_327028

theorem distinct_real_numbers_product (n : ℕ) (a b : Fin n → ℝ)
  (distinct_a : ∀ i1 i2 : Fin n, i1 ≠ i2 → a i1 ≠ a i2)
  (α : ℝ) (hα : ∀ i : Fin n, (∏ j : Fin n, (a i + b j)) = α) :
  ∃ β : ℝ, ∀ j : Fin n, (∏ i : Fin n, (a i + b j)) = β :=
sorry

end distinct_real_numbers_product_l327_327028


namespace find_y_in_terms_of_x_l327_327374

theorem find_y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * (y - 1) + 3) : 
  y = (1 / 4) * x - (1 / 4) := 
by
  sorry

end find_y_in_terms_of_x_l327_327374


namespace all_primes_are_lonely_l327_327305

def is_lonely (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n → (∑ d in (finset.filter (λ x, x ∣ n) (finset.range (n + 1))), (1 : ℚ) / d) ≠
             (∑ d in (finset.filter (λ x, x ∣ m) (finset.range (m + 1))), (1 : ℚ) / d)

theorem all_primes_are_lonely (p : ℕ) (hp : nat.prime p) : is_lonely p :=
begin
  sorry
end

end all_primes_are_lonely_l327_327305


namespace sum_numerator_denominator_of_probability_l327_327015

noncomputable def prime_number := {n : ℕ // nat.prime n ∧ n ∈ set.range (λ i, i + 1)}

theorem sum_numerator_denominator_of_probability :
  ∀ (c1 c2 c3 c4 d1 d2 d3 : ℕ)
  (hc1 : c1 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ nat.prime n})
  (hc2 : c2 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ¬ nat.prime n})
  (hc3 : c3 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ¬ nat.prime n})
  (hc4 : c4 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ¬ nat.prime n})
  (hd1 : d1 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ n ≠ c1 ∧ n ≠ c2 ∧ n ≠ c3 ∧ n ≠ c4 })
  (hd2 : d2 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ n ≠ c1 ∧ n ≠ c2 ∧ n ≠ c3 ∧ n ≠ c4 })
  (hd3 : d3 ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ n ≠ c1 ∧ n ≠ c2 ∧ n ≠ c3 ∧ n ≠ c4 })
  (hc_distinct : c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4)
  (hd_distinct : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3),
sum (numerator false (3 / 20) (3/20)) (denominator false (3 / 20) (3/20)) = 5 :=
by
  sorry

end sum_numerator_denominator_of_probability_l327_327015


namespace minimal_N_for_reverse_order_checkers_l327_327255

theorem minimal_N_for_reverse_order_checkers : ∃ N : ℕ, N = 50 ∧
  (∀ initial_positions : Fin 25 → Fin N,
    all_checkers_reachable initial_positions 25 50) :=
by
  sorry

end minimal_N_for_reverse_order_checkers_l327_327255


namespace Q_proper_subset_P_l327_327421

open Set

def P : Set ℝ := { x | x ≥ 1 }
def Q : Set ℝ := { 2, 3 }

theorem Q_proper_subset_P : Q ⊂ P :=
by
  sorry

end Q_proper_subset_P_l327_327421


namespace prime_sum_10003_l327_327532

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327532


namespace cos_alpha_correct_l327_327397

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  if α ∈ Set.Ioo (π / 2) π ∧ Real.tan α = -Real.sqrt 3 / 3 then -Real.sqrt 3 / 2 else 0

theorem cos_alpha_correct (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.tan α = -Real.sqrt 3 / 3) :
  cos_alpha α = -Real.sqrt 3 / 2 :=
by
  simp [cos_alpha, h1, h2]

end cos_alpha_correct_l327_327397


namespace find_sum_of_a_and_d_l327_327375

theorem find_sum_of_a_and_d 
  {a b c d : ℝ} 
  (h1 : ab + ac + bd + cd = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 :=
sorry

end find_sum_of_a_and_d_l327_327375


namespace root_expression_of_cubic_l327_327977

theorem root_expression_of_cubic :
  ∀ a b c : ℝ, (a^3 - 2*a - 2 = 0) ∧ (b^3 - 2*b - 2 = 0) ∧ (c^3 - 2*c - 2 = 0)
    → a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -6 := 
by 
  sorry

end root_expression_of_cubic_l327_327977


namespace sin_theta_pi_over_6_eq_sqrt3_over3_l327_327021

theorem sin_theta_pi_over_6_eq_sqrt3_over3 (θ : ℝ) (h : sin θ + sin (θ + π / 3) = 1) : 
  sin (θ + π / 6) = sqrt 3 / 3 := 
sorry

end sin_theta_pi_over_6_eq_sqrt3_over3_l327_327021


namespace lines_concurrent_l327_327846

-- Define the given scalene triangle and related constructs.
variable (A B C : Point) (P Q D M N : Point)

-- Suppose ∆ABC is scalene
variable [ScaleneTriangle ABC]

-- Define the incircle touches and the median
variable (IncircleTouch : IncircleTouches AC AB P Q)
variable (TangencyExcircle : AExcircleTangency D (Circumcircle A B C))
variable (MedianAM : Median AM A BC)
variable (SecondIntersection : SecondIntersectionPoint AM (Circumcircle A B C) N)

-- Conclusion to prove: The concurrency of lines PQ, BC, and ND.
theorem lines_concurrent (h1 : IncircleTouches AC AB P Q)
                          (h2 : AExcircleTangency D (Circumcircle A B C))
                          (h3 : Median AM A BC)
                          (h4 : SecondIntersectionPoint AM (Circumcircle A B C) N) :
                          Concurrent PQ BC ND := 
sorry

end lines_concurrent_l327_327846


namespace find_m_l327_327400

theorem find_m (x m : ℤ) (h : x = -1 ∧ x - 2 * m = 9) : m = -5 :=
sorry

end find_m_l327_327400


namespace sum_of_two_primes_unique_l327_327492

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327492


namespace average_age_in_terms_of_Mary_l327_327109

-- We define all conditions and state what needs to be proven
theorem average_age_in_terms_of_Mary (m : ℝ) : 
    let John := 1.5 * m
    let Tonya := 60
    let Sam := 0.8 * Tonya
    let Carol := 2.75 * m
    (John + m + Tonya + Sam + Carol) / 5 = 1.05 * m + 21.6 :=
by
  let John := 1.5 * m
  let Tonya := 60
  let Sam := 0.8 * Tonya
  let Carol := 2.75 * m
  have h1 : John + m + Tonya + Sam + Carol = 1.5 * m + m + 60 + 0.8 * 60 + 2.75 * m := by sorry
  have h2 : John + m + Tonya + Sam + Carol = 5.25 * m + 108 := by sorry
  have h3 : (John + m + Tonya + Sam + Carol) / 5 = (5.25 * m + 108) / 5 := by sorry
  have h4 : (5.25 * m + 108) / 5 = 1.05 * m + 21.6 := by sorry
  exact eq.trans (eq.trans (eq.trans h1 h2) h3) h4

end average_age_in_terms_of_Mary_l327_327109


namespace exponent_properties_l327_327218

theorem exponent_properties :
  (81 : ℝ)^0.25 * (81 : ℝ)^0.2 = 9 :=
by
  -- To be filled with proof steps
  sorry

end exponent_properties_l327_327218


namespace part_a_part_b_l327_327144

-- We define the polynomial and its properties
def polynomial (R : Type) [comm_ring R] (n : ℕ) :=
  {p : R[X] // p.degree ≤ n}

-- Part (a) statement
theorem part_a (d : ℕ) (P : polynomial ℤ d) (α β : ℤ) (hα : P.val.eval α = 1) (hβ : P.val.eval β = -1) : 
  (β - α).nat_abs ∣ 2 := 
sorry

-- Part (b) statement
theorem part_b (d : ℕ) (P : polynomial ℤ d) : 
  ∀ (R : Type) [comm_ring R], finset (R[X]) {x : R | (P.val.eval x)^2 = 1}.card ≤ d + 2 := 
sorry

end part_a_part_b_l327_327144


namespace students_going_to_tournament_l327_327632

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end students_going_to_tournament_l327_327632


namespace loss_per_year_l327_327244

theorem loss_per_year (b : ℝ) (br : ℝ) (lr : ℝ) (t : ℝ) (h : b = 4000) 
(hbr : br = 4) (hlr : lr = 6) (ht : t = 2) :
  ((b * lr * t / 100) - (b * br * t / 100)) / t = -80 :=
by {
  -- Constants
  have h1 : b = 4000 := h,
  have h2 : br = 4 := hbr,
  have h3 : lr = 6 := hlr,
  have h4 : t = 2 := ht,
  -- Substitution and simplification would go here
  sorry
}

end loss_per_year_l327_327244


namespace alien_saturday_sequence_l327_327796

def a_1 : String := "A"
def a_2 : String := "AY"
def a_3 : String := "AYYA"
def a_4 : String := "AYYAYAAY"

noncomputable def a_5 : String := a_4 ++ "YAAYAYYA"
noncomputable def a_6 : String := a_5 ++ "YAAYAYYAAAYAYAAY"

theorem alien_saturday_sequence : 
  a_6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" :=
sorry

end alien_saturday_sequence_l327_327796


namespace unique_a_values_l327_327841

theorem unique_a_values :
  ∃ a_values : Finset ℝ,
    (∀ a ∈ a_values, ∃ r s : ℤ, (r + s = -a) ∧ (r * s = 8 * a)) ∧ a_values.card = 4 :=
by
  sorry

end unique_a_values_l327_327841


namespace strictly_increasing_log_a_l327_327147

theorem strictly_increasing_log_a (a : ℝ) (M : Set ℝ) (h : a ∈ M) : 
  (a > 0) ∧ (a ≠ 1) ∧ (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ x < y ∧ y < 1 → log a (|y - 1|) > log a (|x - 1|)) → 
  M = Ioo 0 (1 / 2) :=
sorry

end strictly_increasing_log_a_l327_327147


namespace number_of_possible_combinations_l327_327276

def four_digit_number_with_first_digit (d: ℕ) (n: ℕ): Prop := (n / 1000 = d) ∧ (1000 ≤ n) ∧ (n < 10000)

def divisible_by (a b: ℕ): Prop := b ∣ a 

theorem number_of_possible_combinations: 
  {n : ℕ // four_digit_number_with_first_digit 4 n ∧ divisible_by n 45}.card = 23 :=
begin
  sorry
end

end number_of_possible_combinations_l327_327276


namespace water_left_in_bathtub_l327_327947

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end water_left_in_bathtub_l327_327947


namespace cos_C_eq_3_5_l327_327460

theorem cos_C_eq_3_5 (A B C : ℝ) (hABC : A^2 + B^2 = C^2) (hRight : B ^ 2 + C ^ 2 = A ^ 2) (hTan : B / C = 4 / 3) : B / A = 3 / 5 :=
by
  sorry

end cos_C_eq_3_5_l327_327460


namespace possible_third_side_of_triangle_l327_327099

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l327_327099


namespace pacific_asian_olympiad_1993_l327_327154

/-- Points \(P_i\) are represented as 2D integer coordinates. -/
structure Point :=
  (x : ℤ)
  (y : ℤ)

def distinct_points (points : List Point) : Prop :=
  points.nodup

def line_segment_free_integers (P Q : Point) : Prop :=
  let dx := Q.x - P.x in
  let dy := Q.y - P.y in
  let gcd := Int.gcd dx dy in
  gcd = 1

noncomputable def exists_odd_Q (points : List Point) : Prop :=
  ∃ i ∈ (Finset.range points.length).val,
    let P := points.get ⟨i, Finset.mem_range.mpr (Nat.lt_of_le_of_ltv _ _)⟩ in
    let Q := points.get ⟨i+1, Finset.mem_range.mpr (Nat.lt_of_le_of_ltv _ _)⟩ in
    ∃ q_x q_y : ℤ,
      q_x * 2 % 2 = 1 ∧
      q_y * 2 % 2 = 1 ∧
      P.x ≤ q_x ∧ q_x ≤ Q.x ∧
      P.y ≤ q_y ∧ q_y ≤ Q.y

theorem pacific_asian_olympiad_1993 (P : List Point)
  (h1 : distinct_points P)
  (h2 : ∀ (i : ℕ), i < P.length → (line_segment_free_integers (P.get ⟨i, _⟩) (P.get ⟨i+1 % P.length, _⟩))) :
  exists_odd_Q P :=
sorry

end pacific_asian_olympiad_1993_l327_327154


namespace sin_sum_to_product_identity_l327_327330

theorem sin_sum_to_product_identity (x : ℝ) :
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin_sum_to_product_identity_l327_327330


namespace negate_proposition_l327_327211

theorem negate_proposition : (∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ ¬ (∃ x : ℝ, x^3 - x^2 + 1 > 1) :=
by
  sorry

end negate_proposition_l327_327211


namespace distance_between_midpoints_l327_327882

theorem distance_between_midpoints 
  {a b : Type} [normed_space ℝ a] [normed_space ℝ b]
  (A B P Q : ℝ^3)
  (m n : ℝ)
  (h_perp : ∥A - B∥ = m)
  (h_skew: is_skew (line_through A P) (line_through B Q))
  (h_PQ_length : ∥P - Q∥ = n) :
  let M := midpoint ℝ A B in
  let N := midpoint ℝ P Q in
  dist M N = (real.sqrt (n^2 - m^2)) / 2 :=
sorry

end distance_between_midpoints_l327_327882


namespace ratio_clara_alice_pens_l327_327791

def alice_pens := 60
def alice_age := 20
def clara_future_age := 61

def clara_age := clara_future_age - 5
def age_difference := clara_age - alice_age

def clara_pens := alice_pens - age_difference

theorem ratio_clara_alice_pens :
  clara_pens / alice_pens = 2 / 5 :=
by
  -- The addressed proof will be here.
  sorry

end ratio_clara_alice_pens_l327_327791


namespace math_problem_l327_327603

variable (a b c : ℝ)
variable (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

theorem math_problem :
  (a + b + c) / 3 - real.cbrt (a * b * c) ≤
  max ((real.sqrt a - real.sqrt b) ^ 2)
      (max ((real.sqrt b - real.sqrt c) ^ 2)
           ((real.sqrt c - real.sqrt a) ^ 2)) :=
sorry

end math_problem_l327_327603


namespace equilateral_isosceles_triangle_CD_eq_DE_l327_327635

theorem equilateral_isosceles_triangle_CD_eq_DE
  (A B C D E : Point) 
  (hABC : EquilateralTriangle A B C) 
  (hADB : IsoscelesTriangle A D B) 
  (hAEB : IsoscelesTriangle A E B) 
  (hAngleADB : ∠ADB = 90) 
  (hAngleAEB : ∠AEB = 150)
  (hD_in_ABC : PointInTriangle D (Triangle.mk A B C))
  (hE_in_ABC : PointInTriangle E (Triangle.mk A B C)) : 
  dist (Point.mk C D) = dist (Point.mk D E) := 
sorry

end equilateral_isosceles_triangle_CD_eq_DE_l327_327635


namespace original_number_proof_l327_327448

theorem original_number_proof (h₁ : 204 / 12.75 = 16) : 
  ∃ x, x / 1.275 = 1.6 ∧ x = 2.04 :=
by
  use 2.04
  split
  { sorry }
  { refl }

end original_number_proof_l327_327448


namespace walkway_area_l327_327779

theorem walkway_area (l w : ℕ) (walkway_width : ℕ) (total_length total_width pool_area walkway_area : ℕ)
  (hl : l = 20) 
  (hw : w = 8)
  (hww : walkway_width = 1)
  (htl : total_length = l + 2 * walkway_width)
  (htw : total_width = w + 2 * walkway_width)
  (hpa : pool_area = l * w)
  (hta : (total_length * total_width) = pool_area + walkway_area) :
  walkway_area = 60 := 
  sorry

end walkway_area_l327_327779


namespace stratified_sampling_male_students_l327_327106

theorem stratified_sampling_male_students (male_students female_students total_sample : ℕ)
  (male_students = 560) 
  (female_students = 420)
  (total_sample = 140) : 
  let total_students := male_students + female_students in
  let probability := (total_sample : ℚ) / total_students in
  let sampled_male_students := (male_students : ℚ) * probability in
  sampled_male_students = 80 := 
by 
  have total_students_eq : total_students = 980 := by sorry
  have probability_eq : probability = (140 : ℚ) / 980 := by sorry
  have sampled_male_students_eq : sampled_male_students = 560 * (1 / 7) := by sorry
  exact sampled_male_students_eq
  sorry

end stratified_sampling_male_students_l327_327106


namespace incorrect_statement_B_l327_327451

def two_times_root_equation (a b c x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x1 = 2 * x2 ∨ x2 = 2 * x1)

theorem incorrect_statement_B (m n : ℝ) (h : (x - 2) * (m * x + n) = 0) :
  ¬(two_times_root_equation 1 (-m+n) (-mn) 2 (-n / m) -> m + n = 0) :=
sorry

end incorrect_statement_B_l327_327451


namespace heaviest_lightest_difference_total_weight_difference_total_earnings_l327_327221

noncomputable def problem_data := 
  { 
    standard_weight : ℝ := 25,
    number_of_baskets : ℕ := 20,
    differences : List ℝ := [-3, -2, -1.5, 0, 1, 2.5],
    basket_counts : List ℕ := [1, 4, 2, 3, 2, 8],
    selling_price : ℝ := 2.6
  }

theorem heaviest_lightest_difference (pd : problem_data) : 
  let heaviest := max 2.5 (-3)
  let lightest := min 2.5 (-3)
  heaviest - lightest = 2.5 - (-3) := by sorry

theorem total_weight_difference (pd : problem_data) : 
  let total_difference := (1 * (-3) + 4 * (-2) + 2 * (-1.5) + 3 * 0 + 2 * 1 + 8 * 2.5)
  total_difference = 8 := by sorry

theorem total_earnings (pd : problem_data) : 
  let total_weight := 25 * 20 + 8
  let earnings := pd.selling_price * total_weight
  earnings = 1321 := by sorry

end heaviest_lightest_difference_total_weight_difference_total_earnings_l327_327221


namespace savings_by_buying_in_bulk_l327_327598

-- Definitions based on conditions
def numMachines := 10
def ballBearingsPerMachine := 30
def normalPricePerBallBearing := 1.0
def salePricePerBallBearing := 0.75
def bulkDiscount := 0.20
def totalBallBearings := numMachines * ballBearingsPerMachine

-- The theorem statement we need to prove
theorem savings_by_buying_in_bulk :
  let normalCost := totalBallBearings * normalPricePerBallBearing
  let saleCostBeforeDiscount := totalBallBearings * salePricePerBallBearing
  let discountAmount := bulkDiscount * saleCostBeforeDiscount
  let saleCostAfterDiscount := saleCostBeforeDiscount - discountAmount in
  normalCost - saleCostAfterDiscount = 120 :=
by
  sorry

end savings_by_buying_in_bulk_l327_327598


namespace ralph_tennis_balls_loaded_l327_327654

theorem ralph_tennis_balls_loaded
  (hit_ratio_1 : ℚ)
  (balls_1 : ℕ)
  (hit_ratio_2 : ℚ)
  (balls_2 : ℕ)
  (not_hit_balls : ℕ)
  (hit_balls_1 : ℕ := (hit_ratio_1 * balls_1).to_nat)
  (hit_balls_2 : ℕ := (hit_ratio_2 * balls_2).to_nat) :
  hit_ratio_1 = 2 / 5 →
  balls_1 = 100 →
  hit_ratio_2 = 1 / 3 →
  balls_2 = 75 →
  not_hit_balls = 110 →
  (balls_1 + balls_2) = 175 :=
by
  intros h1 b1 h2 b2 nh
  sorry

end ralph_tennis_balls_loaded_l327_327654


namespace area_between_concentric_circles_l327_327713

theorem area_between_concentric_circles (R r : ℝ) (hR : R = 10) (hr : r = 4) : 
  let A_L := Real.pi * R^2,
      A_S := Real.pi * r^2,
      A := A_L - A_S
  in A = 84 * Real.pi := by {
  sorry
}

end area_between_concentric_circles_l327_327713


namespace number_of_throwers_l327_327171

theorem number_of_throwers (T N : ℕ) :
  (T + N = 61) ∧ ((2 * N) / 3 = 53 - T) → T = 37 :=
by 
  sorry

end number_of_throwers_l327_327171


namespace solution_set_f_leq_g_range_of_a_l327_327990

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := x + 2

theorem solution_set_f_leq_g (x : ℝ) : f x 1 ≤ g x ↔ (0 ≤ x ∧ x ≤ 2 / 3) := by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ g x) : 2 ≤ a := by
  sorry

end solution_set_f_leq_g_range_of_a_l327_327990


namespace no_two_digit_numbers_form_perfect_cube_sum_l327_327082

theorem no_two_digit_numbers_form_perfect_cube_sum :
  ∀ (N : ℕ), (10 ≤ N ∧ N < 100) →
  ∀ (t u : ℕ), (N = 10 * t + u) →
  let reversed_N := 10 * u + t in
  let sum := N + reversed_N in
  (∃ k : ℕ, sum = k^3) → false :=
by
  sorry

end no_two_digit_numbers_form_perfect_cube_sum_l327_327082


namespace quadratic_pairs_square_diff_exists_l327_327778

open Nat Polynomial

theorem quadratic_pairs_square_diff_exists (P : Polynomial ℤ) (u v w a b n : ℤ) (n_pos : 0 < n)
    (hp : ∃ (u v w : ℤ), P = C u * X ^ 2 + C v * X + C w)
    (h_ab : P.eval a - P.eval b = n^2) : ∃ k > 10^6, ∃ m : ℕ, ∃ c d : ℤ, (c - d = a - b + 2 * k) ∧ 
    (P.eval c - P.eval d = n^2 * m ^ 2) :=
by
  sorry

end quadratic_pairs_square_diff_exists_l327_327778


namespace prime_sum_10003_l327_327525

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327525


namespace equal_areas_ABKD_CEKF_l327_327986

noncomputable section

variables {A B C D E F K : Type}
variables [IsParallelogram A B C D] [PointsOnLine B (Segment A E)] [PointsOnLine D (Segment A F)]
variables [Intersection K (Line E D) (Line F B)]

theorem equal_areas_ABKD_CEKF :
  Area (Quadrilateral A B K D) = Area (Quadrilateral C E K F) := 
sorry

end equal_areas_ABKD_CEKF_l327_327986


namespace sequence_a_100_l327_327899

theorem sequence_a_100 : 
  (∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n, n ≥ 1 → a n = a (n + 1) + 2)) 
  → a 100 = -197 :=
by
  intro h
  sorry

end sequence_a_100_l327_327899


namespace length_PQ_l327_327924

noncomputable theory

open Classical
open Real
open Metric

variables {A B C H D E P Q : Point}
variables {a b c BC AH : ℝ}
variables {P_on_AH Q_on_AH D_on_AC E_on_AB P_on_CE Q_on_BD : Prop}

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def altitude (AH : ℝ) (BC : ℝ) (a b c : ℝ) (A B C H : Point) : Prop :=
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ is_triangle a b c ∧ AH = 2 * sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)) / BC

def angle_bisector (P_on_AH Q_on_AH D_on_AC E_on_AB P_on_CE Q_on_BD : Prop) : Prop :=
  ∃ P Q, P_on_AH ∧ Q_on_AH ∧ D_on_AC ∧ E_on_AB ∧ P_on_CE ∧ Q_on_BD

theorem length_PQ 
  (h1 : AH = 4.8)
  (h2 : BC = 10)
  (h3 : altitude AH BC 6 8 10 A B C H)
  (h4 : angle_bisector P_on_AH Q_on_AH D_on_AC E_on_AB P_on_CE Q_on_BD) : 
  dist P Q = 3.67 :=
sorry

end length_PQ_l327_327924


namespace passengers_remaining_on_bus_l327_327266

theorem passengers_remaining_on_bus
  (initial_passengers : Int)
  (stops : List (Int × Int))
  (h_initial : initial_passengers = 22)
  (h_stops : stops = [(3, -6), (-5, 8), (-4, 2), (1, -8)]) :
  let remaining_passengers :=
    stops.foldl (fun acc (p : Int × Int) => acc + p.1 + p.2) initial_passengers
  in remaining_passengers = 13 := by
  sorry

end passengers_remaining_on_bus_l327_327266


namespace absolute_sum_roots_l327_327608

theorem absolute_sum_roots (m : ℝ) (α β : ℝ) (h_αβ : α ≠ β) (h_root_α : α^2 - 22 * α + m = 0) (h_root_β : β^2 - 22 * β + m = 0) :
  abs(α) + abs(β) = if 0 ≤ m ∧ m ≤ 121 then 22 else real.sqrt (484 - 4 * m) :=
by
  sorry

end absolute_sum_roots_l327_327608


namespace sin_sum_to_product_l327_327335

theorem sin_sum_to_product (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by 
  sorry

end sin_sum_to_product_l327_327335


namespace base_and_exponent_of_neg_two_cubed_l327_327674

def base_of_neg_two_cubed : ℤ := 2
def exponent_of_neg_two_cubed : ℤ := 3

theorem base_and_exponent_of_neg_two_cubed :
  (-2 : ℤ) ^ exponent_of_neg_two_cubed = (- (base_of_neg_two_cubed ^ exponent_of_neg_two_cubed)) :=
begin
  -- base_of_neg_two_cubed = 2
  -- exponent_of_neg_two_cubed = 3
  sorry
end

end base_and_exponent_of_neg_two_cubed_l327_327674


namespace fifth_term_in_arithmetic_sequence_l327_327203

variable (x y : ℝ)

theorem fifth_term_in_arithmetic_sequence :
  let a1 := x + 2 * y,
      a2 := x - 2 * y,
      a3 := x + 2 * y^2,
      a4 := x - 2 * y^2 in
  (a2 - a1 = a3 - a2) → (a3 - a2 = a4 - a3) → 
  (a4 - a3 = -(x - 14 * y) - (x - 10 * y)) → 
  (a5 = x + 42) := by
  sorry

end fifth_term_in_arithmetic_sequence_l327_327203


namespace find_x_plus_y_l327_327916

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 16) (h2 : x + |y| - y = 18) : x + y = 6 := 
sorry

end find_x_plus_y_l327_327916


namespace decreasing_sequence_range_l327_327419

variable (a : ℝ) -- Define the variable a in the set of real numbers
variable (a_n : ℕ+ → ℝ) -- Define the sequence a_n

-- Define the sequences based on the provided conditions
def sequence (n : ℕ+) : ℝ :=
if n <= 6 then (1 - 3 * a) * n + 10 * a else a^(n - 7)

-- State the condition that the sequence is decreasing
def decreasing_sequence : Prop := ∀ n : ℕ+, a_n n > a_n (n + 1)

-- Define the main theorem to be proved
theorem decreasing_sequence_range :
  (decreasing_sequence a_n) ↔ (1 / 3 < a ∧ a < 5 / 8) := by
  sorry -- Proof not included as per the guidelines

end decreasing_sequence_range_l327_327419


namespace no_prime_sum_10003_l327_327507

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327507


namespace find_f_prime_four_l327_327057

noncomputable theory

def f (x : ℝ) : ℝ := x^2 + f' 2 * (Real.log x - x)

theorem find_f_prime_four (f' : ℝ → ℝ) (h : ∀ x, deriv (λ y, y^2 + (f' 2) * (Real.log y - y)) x = f' x) :
  f' 4 = 6 :=
by sorry

end find_f_prime_four_l327_327057


namespace prime_sum_10003_l327_327529

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327529


namespace maggi_ate_5_cupcakes_l327_327625

theorem maggi_ate_5_cupcakes
  (packages : ℕ)
  (cupcakes_per_package : ℕ)
  (left_cupcakes : ℕ)
  (total_cupcakes : ℕ := packages * cupcakes_per_package)
  (eaten_cupcakes : ℕ := total_cupcakes - left_cupcakes)
  (h1 : packages = 3)
  (h2 : cupcakes_per_package = 4)
  (h3 : left_cupcakes = 7) :
  eaten_cupcakes = 5 :=
by
  sorry

end maggi_ate_5_cupcakes_l327_327625


namespace jacoby_lottery_winning_l327_327136

theorem jacoby_lottery_winning :
  let total_needed := 5000
  let job_earning := 20 * 10
  let cookies_earning := 4 * 24
  let total_earnings_before_lottery := job_earning + cookies_earning
  let after_lottery := total_earnings_before_lottery - 10
  let gift_from_sisters := 500 * 2
  let total_earnings_and_gifts := after_lottery + gift_from_sisters
  let total_so_far := total_needed - 3214
  total_so_far - total_earnings_and_gifts = 500 :=
by
  sorry

end jacoby_lottery_winning_l327_327136


namespace problem_statement_l327_327980

variable (n : ℕ) (x : Fin n → ℝ)

noncomputable def x_sum := ∑ i, x i
noncomputable def sum_fractions := ∑ i, (x i / (1 - x i))
noncomputable def sum_squared_fractions := ∑ i, (x i^2 / (1 - x i))

theorem problem_statement
  (h1 : x_sum x = 3)
  (h2 : sum_fractions x = n - 2) :
  sum_squared_fractions x = 5 - n :=
sorry

end problem_statement_l327_327980


namespace tub_emptying_time_l327_327453

variables (x C D T : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)

theorem tub_emptying_time (h1 : 4 * (D - x) = (5 / 7) * C) :
  T = 8 / (5 + (28 * x) / C) :=
by sorry

end tub_emptying_time_l327_327453


namespace textbooks_probability_l327_327950

theorem textbooks_probability (m n : ℕ) (boxed_result : m + n = 1003) :
  let books := 15
  let english_books := 4
  let box1_cap := 3
  let box2_cap := 4
  let box3_cap := 5
  let box4_cap := 3

  ∃ (m n : ℕ), (m + n = 1003) ∧
               (∀ (arrangement : list ℕ), arrangement.length = books →
                  ∃ (box1 box2 box3 box4: list ℕ),
                    box1.length = box1_cap ∧
                    box2.length = box2_cap ∧
                    box3.length = box3_cap ∧
                    box4.length = box4_cap ∧
                    list.perm arrangement (box1 ++ box2 ++ box3 ++ box4) ∧
                    (box3.count english_books = 4))
  sorry

end textbooks_probability_l327_327950


namespace find_lambda_l327_327904

-- Given vectors m and n
def m (λ : ℝ) : ℝ × ℝ := (λ + 1, 1)
def n (λ : ℝ) : ℝ × ℝ := (λ + 2, 2)

-- Prove that λ = -3 given the condition (m + n) is perpendicular to (m - n)
theorem find_lambda (λ : ℝ) 
  (h : let mn := (m λ).1 + (n λ).1, 
            m'n := (m λ).2 + (n λ).2;
       let mmn := (m λ).1 - (n λ).1, 
            m'nn := (m λ).2 - (n λ).2;
       (mn, m'n) = ((2 * λ + 3, 3)) ∧ (mmn, m'nn) = (-1, -1) ∧ 
       ((mn, m'n) ⋅ (mmn, m'nn) = 0)) : λ = -3 := sorry

end find_lambda_l327_327904


namespace semicircle_perimeter_approx_l327_327246

/-- Statement of the math proof problem -/
theorem semicircle_perimeter_approx (r : ℝ) (h : r = 11) (pi_approx : ℝ) (pi_approx_val : pi_approx = 3.14159) :
  ∃ P : ℝ, P ≈ 56.56 ∧ P = pi_approx * r + 2 * r := 
by {
  have radius := (by simp [h] : r = 11),
  have pi_value := (by simp [pi_approx_val] : pi = 3.14159),
  sorry
}

end semicircle_perimeter_approx_l327_327246


namespace problem_1_problem_2_problem_3_l327_327905

#check vector

noncomputable def a (m : ℝ) : vector ℝ 2 := ![m, 1]
noncomputable def b : vector ℝ 2 := ![1/2, real.sqrt(3)/2]

-- Statements
theorem problem_1 (m : ℝ) : (∃ c : ℝ, a m = c • b) → m = -real.sqrt(3)/3 := sorry

theorem problem_2 (m : ℝ) : (a m).dot_product b = 0 → m = -real.sqrt(3) := sorry

theorem problem_3 (k t : ℝ) (h1 : k ≠ 0) (h2 : t ≠ 0) : 
  (let a := a (-real.sqrt(3)); 
           b := b; 
           inner_product := a + (t^2 - 3) • b; 
           other_vec := -k • a + t • b 
  in inner_product ⬝ other_vec = 0) 
  → ∃ t : ℝ, t ≠ 0 ∧ (∀ k : ℝ, k ≠ 0 → k = (t^2 - 3)t / 4) → 
  ∀ t : ℝ, t ≠ 0 → 
            ∃ k : ℝ, k = (t^2 - 3)t / 4 → (k + t^2) / t = (-7 / 4) := sorry

end problem_1_problem_2_problem_3_l327_327905


namespace max_trains_ratio_l327_327997

theorem max_trains_ratio (years : ℕ) 
    (birthday_trains : ℕ) 
    (christmas_trains : ℕ) 
    (total_trains : ℕ)
    (parents_multiple : ℕ) 
    (h_years : years = 5)
    (h_birthday_trains : birthday_trains = 1)
    (h_christmas_trains : christmas_trains = 2)
    (h_total_trains : total_trains = 45)
    (h_parents_multiple : parents_multiple = 2) :
  let trains_received_in_years := years * (birthday_trains + 2 * christmas_trains)
  let trains_given_by_parents := total_trains - trains_received_in_years
  let trains_before_gift := total_trains - trains_given_by_parents
  trains_given_by_parents / trains_before_gift = parents_multiple := by
  sorry

end max_trains_ratio_l327_327997


namespace sum_consecutive_integers_150_l327_327823

theorem sum_consecutive_integers_150 (n : ℕ) (a : ℕ) (hn : n ≥ 3) (hdiv : 300 % n = 0) :
  n * (2 * a + n - 1) = 300 ↔ (a > 0) → n = 3 ∨ n = 5 ∨ n = 15 :=
by sorry

end sum_consecutive_integers_150_l327_327823


namespace problem1_problem2_problem3_problem4_l327_327751

-- Problem 1
theorem problem1 (M : ℝ × ℝ) (hM : M = (-3, 0)) (x y : ℝ) (h : x^2 + (y+2)^2 = 25) (chord_len : ℝ) (h_len : chord_len = 8) : 
  l = "5x - 12y + 15 = 0" ∨ l = "x = -3" :=
sorry

-- Problem 2
theorem problem2 (x y : ℝ) (C1 : x^2 + y^2 + 2*x + 8*y - 8 = 0) (C2 : x^2 + y^2 - 4*x - 4*y - 2 = 0) :
  length_common_chord = 2 * math.sqrt(5) :=
sorry

-- Problem 3
theorem problem3 (A B : ℝ × ℝ) (hA : A = (0, 2)) (hB : B = (-2, 2)) (line : ℝ × ℝ → ℝ) (hline : line = λ P, P.1 - P.2 - 2) :
  circle_equation = "(x+1)^2 + (y+3)^2 = 26" :=
sorry

-- Problem 4
theorem problem4 (x y : ℝ) (circle : (x-2)^2 + y^2 = 1) (P : ℝ × ℝ) (hP : P = (3, 4)) :
  PA_dot_PB = 16 :=
sorry

end problem1_problem2_problem3_problem4_l327_327751


namespace ratio_AD_DC_2_3_l327_327942

variable (A B C D : Type)

def is_triangle (A B C : Type) := Prop

def angle_B_120 (A B C : Type) [is_triangle A B C] := (120 : ℝ)

def sides_ratio (A B C : Type) [is_triangle A B C] := (2 : ℝ) * (BC : ℝ) ≤ (AB : ℝ)

def perpendicular_bisector_intersects (A B C D : Type) [is_triangle A B C] := (intersection_point D : Type) := intersect(perp_bisector(AB), AC)

theorem ratio_AD_DC_2_3 
  (A B C D : Type) 
  [is_triangle A B C] 
  (h1 : angle_B_120 A B C) 
  (h2 : sides_ratio A B C) 
  (h3 : perpendicular_bisector_intersects A B C D) : 
  AD / DC = 2 / 3 := sorry

end ratio_AD_DC_2_3_l327_327942


namespace prime_sum_10003_l327_327530

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327530


namespace goat_max_distance_from_origin_l327_327636

noncomputable def greatest_distance (post_x post_y rope_length : ℝ) : ℝ :=
  let origin_distance := Real.sqrt (post_x^2 + post_y^2)
  in origin_distance + rope_length

theorem goat_max_distance_from_origin :
  greatest_distance 5 1 15 = Real.sqrt 26 + 15 :=
by
  sorry

end goat_max_distance_from_origin_l327_327636


namespace solve_exponential_inequality_l327_327836

theorem solve_exponential_inequality (x : ℝ) : 
  (1 / 4)^(x - 1) > 16 ↔ x < -1 := 
sorry

end solve_exponential_inequality_l327_327836


namespace construct_2n_faced_polyhedron_with_congruent_quadrilateral_faces_l327_327593

theorem construct_2n_faced_polyhedron_with_congruent_quadrilateral_faces (n : ℕ) (h_n : n ≥ 3) : 
  ∃ (P : Polyhedron), 
    P.faces.count = 2 * n ∧ 
    (∀ f ∈ P.faces, is_congruent_quadrilateral f) := 
sorry

end construct_2n_faced_polyhedron_with_congruent_quadrilateral_faces_l327_327593


namespace seating_arrangements_l327_327628

theorem seating_arrangements 
  (mr_lopez : Type) (mrs_lopez : Type) (two_children : Type) (grandparent : Type)
  [finite mr_lopez] [fintype mr_lopez]
  [finite mrs_lopez] [fintype mrs_lopez]
  [finite two_children] [fintype two_children]
  [finite grandparent] [fintype grandparent] :
  let adults := {mr_lopez, mrs_lopez, grandparent}
  let total_people := {mr_lopez, mrs_lopez, grandparent} ∪ (two_children : set Type)
  (|adults| = 3) → 
  (∃ driver ∈ adults, true) →
  (|total_people| = 5) →
  ∃ seating_arrangements : nat, seating_arrangements = 72 :=
by
  sorry

end seating_arrangements_l327_327628


namespace range_of_m_l327_327859

variable (f : ℝ → ℝ)

/- Conditions -/
axiom h_sym : ∀ x : ℝ, f x = f (2 - x)
axiom h_mono : ∀ x : ℝ, x > 1 → f' x < 0

theorem range_of_m 
  (m : ℝ) (h_m_range : 1/3 ≤ m ∧ m ≤ 1) :
  f (m + 1) ≤ f (2 * m) :=
by
  sorry

end range_of_m_l327_327859


namespace total_opponent_scores_is_45_l327_327787

-- Definitions based on the conditions
def games : Fin 10 := Fin.mk 10 sorry

def team_scores : Fin 10 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨6, _⟩ => 7
| ⟨7, _⟩ => 8
| ⟨8, _⟩ => 9
| ⟨9, _⟩ => 10
| _ => 0  -- Placeholder for out-of-bounds, should not be used

def lost_games : Fin 5 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5
| ⟨3, _⟩ => 7
| ⟨4, _⟩ => 9

def opponent_score_lost : ℕ → ℕ := λ s => s + 1

def won_games : Fin 5 → ℕ
| ⟨0, _⟩ => 2
| ⟨1, _⟩ => 4
| ⟨2, _⟩ => 6
| ⟨3, _⟩ => 8
| ⟨4, _⟩ => 10

def opponent_score_won : ℕ → ℕ := λ s => s / 2

-- Main statement to prove total opponent scores
theorem total_opponent_scores_is_45 :
  let total_lost_scores := (lost_games 0 :: lost_games 1 :: lost_games 2 :: lost_games 3 :: lost_games 4 :: []).map opponent_score_lost
  let total_won_scores  := (won_games 0 :: won_games 1 :: won_games 2 :: won_games 3 :: won_games 4 :: []).map opponent_score_won
  total_lost_scores.sum + total_won_scores.sum = 45 :=
by sorry

end total_opponent_scores_is_45_l327_327787


namespace prime_sum_10003_l327_327528

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327528


namespace rectangle_diagonal_property_l327_327044

-- Assuming some preliminary setup for the rectangle and the perpendicularity conditions

variables {A B C D E F : Point}
variables {AB AD AC : ℝ}

-- Assume the rectangle and relevant properties
def RectangleABCDisRect (A B C D : Point) : Prop :=
  -- Define the properties of the rectangle here
  sorry

-- Points E and F are such that CE ⊥ AB and CF ⊥ AD
def PerpendicularCE_AB (C E : Point) (AB : Line) : Prop :=
  -- Define the perpendicularity of CE to AB here
  sorry

def PerpendicularCF_AD (C F : Point) (AD : Line) : Prop :=
  -- Define the perpendicularity of CF to AD here
  sorry

theorem rectangle_diagonal_property
  (h_rect : RectangleABCDisRect A B C D)
  (h_perp_ce_ab : PerpendicularCE_AB C E AB)
  (h_perp_cf_ad : PerpendicularCF_AD C F AD) :
  AB * dist A E + AD * dist A F = (dist A C) ^ 2 :=
sorry

end rectangle_diagonal_property_l327_327044


namespace boy_should_walk_to_next_stop_l327_327264

theorem boy_should_walk_to_next_stop (v : ℝ) :
    ∀ (d_next : ℝ) (d_see : ℝ) (v_boy : ℝ),
    d_next = 1 → d_see = 2 → v_boy = v / 4 →
    (v_boy * (d_see / ((5*v)/4)) <= d_next) ∧ (v_boy * (d_see / ((3*v)/4)) <= d_next) :=
begin
  intros d_next d_see v_boy h1 h2 h3,
  have t1 := d_see / ((5*v)/4),
  have run_back := v_boy * t1,
  have t2 := d_see / ((3*v)/4),
  have reach_stop := v_boy * t2,
  split;
  linarith,
end

end boy_should_walk_to_next_stop_l327_327264


namespace pizza_slice_division_l327_327706

theorem pizza_slice_division : 
  ∀ (num_coworkers num_pizzas slices_per_pizza : ℕ),
  num_coworkers = 12 →
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  (num_pizzas * slices_per_pizza) / num_coworkers = 2 := 
by
  intros num_coworkers num_pizzas slices_per_pizza h_coworkers h_pizzas h_slices
  rw [h_coworkers, h_pizzas, h_slices]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slice_division_l327_327706


namespace GCF_LCM_calculation_l327_327719

theorem GCF_LCM_calculation : 
  GCD (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end GCF_LCM_calculation_l327_327719


namespace find_angle_F_l327_327589

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l327_327589


namespace band_members_minimum_n_l327_327758

theorem band_members_minimum_n 
  (n : ℕ) 
  (h1 : n % 6 = 3) 
  (h2 : n % 8 = 5) 
  (h3 : n % 9 = 7) : 
  n ≥ 165 := 
sorry

end band_members_minimum_n_l327_327758


namespace smallest_prime_with_conditions_l327_327355

theorem smallest_prime_with_conditions :
  ∃ (n : ℕ), 
    prime n ∧ 
    10 ≤ n ∧ n < 100 ∧ 
    (n % 100) / 10 = 2 ∧ 
    (let reversed := (n % 10) * 10 + (n / 10) in ¬ prime reversed ∧ (reversed % 3 = 0 ∨ reversed % 7 = 0)) ∧ 
    ∀ m, prime m → 10 ≤ m → m < 100 → 
        (m % 100) / 10 = 2 → 
        (let reversed := (m % 10) * 10 + (m / 10) in ¬ prime reversed ∧ (reversed % 3 = 0 ∨ reversed % 7 = 0)) → 
        n ≤ m :=
begin
  use 21,
  -- solution steps can be filled in here...
  sorry,
end

end smallest_prime_with_conditions_l327_327355


namespace sum_possible_values_n_l327_327936

theorem sum_possible_values_n : 
  (∑ k in {n | ∃ m : ℕ, m > 0 ∧ n > 0 ∧ 1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 4}, k) = 51 :=
by 
  -- Proof to be filled in 
  sorry

end sum_possible_values_n_l327_327936


namespace unique_sum_of_two_primes_l327_327559

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327559


namespace number_of_divisors_of_54_greater_than_7_l327_327075

theorem number_of_divisors_of_54_greater_than_7 : 
  (set.filter (λ d, 7 < d) {d | d ∣ 54}).finite.to_finset.card = 4 := 
by
  sorry

end number_of_divisors_of_54_greater_than_7_l327_327075


namespace hyperbola_equation_l327_327883

theorem hyperbola_equation (x y : ℝ) (K : ℝ) (c : ℝ) (a b : ℝ) 
  (h_focus : c^2 = 36) 
  (h_asymptotes : ∀ x y : ℝ, y = √2*x ∨ y = -√2*x) 
  (h_smaller : K > 0) 
  (h_sum_distances : 2*K + K = c^2) : 
  x^2 / 24 - y^2 / 12 = 1 :=
by
  sorry

end hyperbola_equation_l327_327883


namespace condition_determine_plane_l327_327261

/-- The condition that can determine a plane in space is a triangle -/
theorem condition_determine_plane (c : Type) [Plane c] :
  (∃ (a b : Line c), True) → 
  (∃ (p : Point c) (q : Line c), True) →
  (∃ (a b c : Point c), True) → 
  (∃ (a b c : Point c), is_triangle a b c) :=
sorry

end condition_determine_plane_l327_327261


namespace solveForN_l327_327436

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l327_327436


namespace isle_of_misfortune_l327_327637

theorem isle_of_misfortune (P L : ℕ) (h1 : P + L = 101)
  (h2 : L ≥ P + 1) : P = 50 ∧ L = 51 :=
begin
  have h3 : L = 101 - P, from (nat.sub_eq_iff_eq_add.mp h1.symm).symm,
  rw h3 at h2,
  have h4 : 101 - P ≥ P + 1, from h2,
  have h5 : 100 ≥ 2 * P, from (nat.le_add_one_iff.mpr $ nat.le_add_of_nonneg_right $ nat.zero_le _).trans h4,
  have h6 : 50 ≥ P, from nat.le_div (nat.zero_le P) h5,
  have h7 : P ≤ 50, from le_of_not_gt h6,
  have h8 : P = 50, from le_antisymm h7 (nat.succ_le_iff.mp $ h6.trans $ h7.ge.trans $ nat.le_two_mul_of_div (by norm_num)),
  rw h8 at h3,
  split; assumption,
end

end isle_of_misfortune_l327_327637


namespace inequality_proof_l327_327383

noncomputable def pos_real := { x : ℝ // x > 0 }

theorem inequality_proof (n : ℕ) (h_n : n ≥ 2) (a : fin n → pos_real) :
  (let s1 := ∑ j in finset.range n, real.root (∏ k in finset.range (j + 1), (a k : ℝ)) (j + 1)
   let s2 := ∑ j in finset.range n, (a j : ℝ)
   (s1 / s2)^(1 / n) + real.root (∏ j in finset.range n, (a j : ℝ)) n / s1 <= (n + 1) / n) :=
by sorry

end inequality_proof_l327_327383


namespace no_prime_sum_10003_l327_327473

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327473


namespace perpendicularity_condition_l327_327072

variables (m : ℝ)

def vector_a := (1, m)
def vector_b := (3, -2)
def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)

theorem perpendicularity_condition : vector_sum m = (4, m - 2) → (vector_sum m).1 * vector_b.1 + (vector_sum m).2 * vector_b.2 = 0 → m = 8 :=
by
  intros _
  sorry

end perpendicularity_condition_l327_327072


namespace correct_judgments_about_f_l327_327821

-- Define the function f with its properties
variable {f : ℝ → ℝ} 

-- f is an even function
axiom even_function : ∀ x, f (-x) = f x

-- f satisfies f(x + 1) = -f(x)
axiom function_property : ∀ x, f (x + 1) = -f x

-- f is increasing on [-1, 0]
axiom increasing_on_interval : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y

theorem correct_judgments_about_f :
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, f x = f (-x + 2)) ∧
  (f 2 = f 0) :=
by 
  sorry

end correct_judgments_about_f_l327_327821


namespace problem_solution_l327_327316

noncomputable def equation_has_2_real_solutions : Prop :=
  ∃ n ∈ ℕ, n = 2 ∧ ∀ x, (6 * x / (x^2 + x + 4) + 8 * x / (x^2 - 8 * x + 4) = 3) → x ∈ ℝ

theorem problem_solution : equation_has_2_real_solutions :=
  sorry

end problem_solution_l327_327316


namespace tangent_line_equation_at_e_l327_327000

noncomputable section

open Real

-- Definitions based on the given conditions
def curve (x : ℝ) : ℝ := x + log x
def derivative (x : ℝ) : ℝ := 1 + 1 / x

-- Lean statement of the problem
theorem tangent_line_equation_at_e :
  let e := exp 1
  let point := (e, e + 1)
  let slope := derivative e 
  ∃ (m : ℝ) (b : ℝ), curve x = slope * x + b ∧ 
    ∀ x, y, (() -> (y = curve x -> y)).appendLS ←
    (e + 1) * x - e * (curve x) = 0 :=
by
  sorry -- proof omitted

end tangent_line_equation_at_e_l327_327000


namespace value_of_a_l327_327095

noncomputable def coefficient_of_x2_term (a : ℝ) : ℝ :=
  a^4 * Nat.choose 8 4

theorem value_of_a (a : ℝ) (h : coefficient_of_x2_term a = 70) : a = 1 ∨ a = -1 := by
  sorry

end value_of_a_l327_327095


namespace scientific_notation_of_12000000000_l327_327666

theorem scientific_notation_of_12000000000 :
  12000000000 = 1.2 * 10^10 :=
by sorry

end scientific_notation_of_12000000000_l327_327666


namespace remainder_div_power10_l327_327659

theorem remainder_div_power10 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, (10^n - 1) % 37 = k^2 := by
  sorry

end remainder_div_power10_l327_327659


namespace monthly_earnings_is_correct_l327_327961

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l327_327961


namespace min_value_of_a_b_l327_327023

noncomputable def min_sum (a b : ℝ) : ℝ := 3 + 2 * Real.sqrt 2

theorem min_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2^a * 4^b = (2^a)^b) :
  a + b = min_sum a b :=
sorry

end min_value_of_a_b_l327_327023


namespace arrangement_of_people_l327_327358

def fiveVolunteers : ℕ := 5
def twoElderly : ℕ := 2
def totalPeople := fiveVolunteers + twoElderly

theorem arrangement_of_people : 
  (∃ (arrangements : ℕ), arrangements = 960 
    ∧ (condition1 : twoElderly must_stand_next_to_each_other) 
    ∧ (condition2 : twoElderly cannot_be_at_either_end))
  :=
by
  sorry

end arrangement_of_people_l327_327358


namespace angle_PIM_eq_78_l327_327591

-- Define the points P, Q, R, and the incenter I
variables {P Q R I : Type}

-- Define the conditions of the triangle and the incenter
-- Condition 1: PQR is a triangle
axiom h_triangle : is_triangle P Q R

-- Condition 2: The angle bisectors intersect at the incenter
axiom h_bisectors : angle_bisectors_intersect_at_incenter P Q R I

-- Condition 3: ∠PRQ = 24°
axiom h_angle_PRQ : angle P R Q = 24

-- Prove that ∠PIM = 78° given the above conditions
theorem angle_PIM_eq_78 : angle P I M = 78 :=
by
  sorry

end angle_PIM_eq_78_l327_327591


namespace unique_sum_of_two_primes_l327_327557

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327557


namespace probability_suitable_joint_given_physique_l327_327847

noncomputable def total_children : ℕ := 20
noncomputable def suitable_physique : ℕ := 4
noncomputable def suitable_joint_structure : ℕ := 5
noncomputable def both_physique_and_joint : ℕ := 2

noncomputable def P (n m : ℕ) : ℚ := n / m

theorem probability_suitable_joint_given_physique :
  P both_physique_and_joint total_children / P suitable_physique total_children = 1 / 2 :=
by
  sorry

end probability_suitable_joint_given_physique_l327_327847


namespace frog_arrangement_count_l327_327696

-- Definitions based on the conditions
def number_of_frogs := 7
def green_frogs := 3
def red_frogs := 3
def blue_frog := 1

def not_sitting_next_to (green: ℕ) (red: ℕ) : Prop := green ≠ red

noncomputable def arrange_frogs (position: Fin number_of_frogs) (colors: List (Fin number_of_frogs -> Char)) : Nat := sorry

-- The mathematical proof problem statement
theorem frog_arrangement_count : 
  ∀ (positions: List (Fin number_of_frogs)) (colors: List (Fin number_of_frogs -> Char)),
  positions.length = number_of_frogs →
  (3 = green_frogs) →
  (3 = red_frogs) →
  (1 = blue_frog) →
  (∀ (i j: Fin number_of_frogs), (colors[i] = 'G' → colors[j] = 'R' → not_sitting_next_to i j) ∧
    (colors[i] = 'B' → (i ≠ 0 ∧ i ≠ (number_of_frogs - 1)))) →
  arrange_frogs positions colors = 360 :=
sorry

end frog_arrangement_count_l327_327696


namespace probability_X_leq_1_uniform_interval_l327_327278

noncomputable def probability_of_X_leq_1 (X : ℝ) (a b : ℝ) (c : ℝ) (d : ℝ): ℝ := 
  if a ≤ X ∧ X ≤ b 
  then (d - c) / (b - a) 
  else 0

theorem probability_X_leq_1_uniform_interval : 
  let X := 1 in
  let a := -2 in
  let b := 3 in
  let c := -2 in
  let d := 1 in
  probability_of_X_leq_1 X a b c d = 3/5 :=
by
  sorry

end probability_X_leq_1_uniform_interval_l327_327278


namespace not_sum_of_squares_or_cubes_in_ap_l327_327347

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a + b * b = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a * a + b * b * b = n

def arithmetic_progression (a d k : ℕ) : ℕ :=
  a + d * k

theorem not_sum_of_squares_or_cubes_in_ap :
  ∀ k : ℕ, ¬ is_sum_of_two_squares (arithmetic_progression 31 36 k) ∧
           ¬ is_sum_of_two_cubes (arithmetic_progression 31 36 k) := by
  sorry

end not_sum_of_squares_or_cubes_in_ap_l327_327347


namespace slices_per_person_l327_327704

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l327_327704


namespace subset_sum_divisible_l327_327658

theorem subset_sum_divisible {a n : ℕ} (h_n : n > 0) :
  ∃ S : finset ℕ, (∀ x ∈ S, ∃ k, x = a + k ∧ k < n) ∧ S.sum id % (n * (n + 1) / 2) = 0 :=
by
  sorry

end subset_sum_divisible_l327_327658


namespace depth_of_right_frustum_l327_327285

-- Definitions
def volume_cm3 := 190000 -- Volume in cubic centimeters (190 liters)
def top_edge := 60 -- Length of the top edge in centimeters
def bottom_edge := 40 -- Length of the bottom edge in centimeters
def expected_depth := 75 -- Expected depth in centimeters

-- The following is the statement of the proof
theorem depth_of_right_frustum 
  (V : ℝ) (A1 A2 : ℝ) (h : ℝ)
  (hV : V = 190 * 1000)
  (hA1 : A1 = top_edge * top_edge)
  (hA2 : A2 = bottom_edge * bottom_edge)
  (h_avg : 2 * A1 / (top_edge + bottom_edge) = 2 * A2 / (top_edge + bottom_edge))
  : h = expected_depth := 
sorry

end depth_of_right_frustum_l327_327285


namespace triangle_circumcircle_area_l327_327461

noncomputable def area_of_circumcircle (a b c : ℝ) (A B C : ℝ) (S : ℝ) :=
  (4 * Real.sqrt 3 * S = b^2 + c^2 - a^2) ∧
  (a = 2) ∧
  (∀ (R : ℝ), R = a / (2 * Real.sin A) → Real.pi * R^2 = 4 * Real.pi)

theorem triangle_circumcircle_area :
  ∀ (a b c A B C S : ℝ),
  area_of_circumcircle a b c A B C S → Real.pi * (2 : ℝ)^2 = 4 * Real.pi :=
by
  intros a b c A B C S h
  cases h with h1 h2
  cases h2 with h3 h4
  exact h4 2 sorry

end triangle_circumcircle_area_l327_327461


namespace q_negative_one_is_minus_one_l327_327300

-- Define the function q and the point on the graph
def q (x : ℝ) : ℝ := sorry

-- The condition: point (-1, -1) lies on the graph of q
axiom point_on_graph : q (-1) = -1

-- The theorem to prove that q(-1) = -1
theorem q_negative_one_is_minus_one : q (-1) = -1 :=
by exact point_on_graph

end q_negative_one_is_minus_one_l327_327300


namespace sum_of_primes_10003_l327_327551

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327551


namespace no_prime_sum_10003_l327_327471

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327471


namespace gcf_of_lcm_eq_15_l327_327726

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_of_lcm_eq_15 : gcf (lcm 9 15) (lcm 10 21) = 15 := by
  sorry

end gcf_of_lcm_eq_15_l327_327726


namespace sum_of_two_primes_unique_l327_327488

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327488


namespace range_of_m_l327_327395

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (x^2 : ℝ) / (2 - m) + (y^2 : ℝ) / (m - 1) = 1 → 2 - m < 0 ∧ m - 1 > 0) →
  (∀ Δ : ℝ, Δ = 16 * (m - 2) ^ 2 - 16 → Δ < 0 → 1 < m ∧ m < 3) →
  (∀ (p q : Prop), p ∨ q ∧ ¬ q → p ∧ ¬ q) →
  m ≥ 3 :=
by
  intros h1 h2 h3
  sorry

end range_of_m_l327_327395


namespace polynomial_inequality_l327_327181

noncomputable def abs_val {R : Type*} [LinearOrderedRing R] (x : R) : R :=
  if x < 0 then -x else x

theorem polynomial_inequality (p : Polynomial ℝ) (n : ℕ) (hn : 100 ≤ n) (deg_cond : p.degree ≤ n - 10 * real.sqrt n) :
  abs_val (p.eval 0) ≤ (1 / 10) * (finset.sum (finset.range (n + 1)) (λ k, (nat.choose n k) * abs_val (p.eval k))) :=
sorry

end polynomial_inequality_l327_327181


namespace max_volume_prism_l327_327114

theorem max_volume_prism (a b h : ℝ) (h_congruent_lateral : a = b) (sum_areas_eq_48 : a * h + b * h + a * b = 48) : 
  ∃ V : ℝ, V = 64 :=
by
  sorry

end max_volume_prism_l327_327114


namespace polynomial_coeffs_correctness_l327_327365

noncomputable def polynomial_coeffs (x : ℝ) : ℕ → ℝ := 
  λ n, (x - real.sqrt 3)^2017.coeff n

theorem polynomial_coeffs_correctness :
  let a : ℕ → ℝ := polynomial_coeffs x in
  ((a 0 + a 2 + ... + a 2016)^2 - (a 1 + a 3 + ... + a 2017)^2 = -2 ^ 2017) :=
sorry

end polynomial_coeffs_correctness_l327_327365


namespace digits_product_problem_l327_327639

noncomputable theory
open_locale nat

theorem digits_product_problem : 
  ∃ (X Y Z O : ℕ), X ≠ Y ∧ X ≠ Z ∧ X ≠ O ∧ Y ≠ Z ∧ Y ≠ O ∧ Z ≠ O ∧ 
  1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9 ∧ 1 ≤ O ∧ O ≤ 9 ∧ 
  (let n₁ := X * 100 + Y * 10 + Z,
       n₂ := Y * 100 + X * 10 + Z in
  (n₁ * n₂ = 169201 ∨ n₁ * n₂ = 193501) ∧ 
  ((let prod := n₁ * n₂ in 
  (prod / 100000) = (prod % 10) ∧ 
  (prod / 10000) % 10 ≠ (prod / 1000) % 10 ∧ 
  (prod / 10000) % 10 ≠ (prod / 100) % 10 ∧ 
  (prod / 10000) % 10 ≠ (prod / 10) % 10 ∧ 
  (prod / 1000) % 10 ≠ (prod / 100) % 10 ∧ 
  (prod / 1000) % 10 ≠ (prod / 10) % 10 ∧ 
  (prod / 100) % 10 ≠ (prod / 10) % 10))) :=
begin
  sorry
end

end digits_product_problem_l327_327639


namespace correct_option_l327_327391

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l327_327391


namespace sum_of_two_primes_unique_l327_327493

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327493


namespace no_prime_sum_10003_l327_327513

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327513


namespace minimum_value_f_l327_327088

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a * x - 1) * Real.exp (x - 1)

theorem minimum_value_f :
  (∀ x : ℝ, 0 ≤ ∇(f x (-1)) * (x + 2)) →
  f 1 (-1) = -1 :=
by
  sorry

end minimum_value_f_l327_327088


namespace chandler_needs_work_29_weeks_l327_327307

theorem chandler_needs_work_29_weeks 
  (fence_cost birthday_money weekly_earnings : ℕ) 
  (h1 : fence_cost = 800) 
  (h2 : birthday_money = 120 + 80 + 20) 
  (h3 : weekly_earnings = 20) :
  ∃ x : ℕ, 220 + 20 * x = fence_cost ∧ x = 29 := 
by
  use 29
  simp [h1, h2, h3]
  sorry

end chandler_needs_work_29_weeks_l327_327307


namespace linear_function_expression_l327_327007

theorem linear_function_expression (k b : ℝ) (h : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 4 → 3 ≤ k * x + b ∧ k * x + b ≤ 6)) :
  (k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7) :=
by
  sorry

end linear_function_expression_l327_327007


namespace counting_numbers_remainder_7_div_61_l327_327078

def divides (a b : Nat) : Prop := ∃ k, b = k * a

theorem counting_numbers_remainder_7_div_61 : 
    {n : Nat | divides n 54 ∧ n > 7}.card = 4 := 
by
  sorry

end counting_numbers_remainder_7_div_61_l327_327078


namespace geom_series_min_q_l327_327984

theorem geom_series_min_q (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h_geom : ∃ k : ℝ, q = p * k ∧ r = q * k)
  (hpqr : p * q * r = 216) : q = 6 :=
sorry

end geom_series_min_q_l327_327984


namespace total_apples_l327_327634

def green_apples : ℕ := 2
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

theorem total_apples : green_apples + red_apples + yellow_apples = 19 :=
by
  -- Placeholder for the proof
  sorry

end total_apples_l327_327634


namespace problem_inequality_l327_327185

theorem problem_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2) : xy + 3 ≥ 2x + 2y :=
sorry

end problem_inequality_l327_327185


namespace fourth_vertex_of_square_l327_327039

theorem fourth_vertex_of_square (z1 z2 z3 z4 : ℂ) (h1 : z1 = 1 + 2 * complex.I)
  (h2 : z2 = (3 + complex.I) / (1 + complex.I)) (h3 : z3 = -1 - 2 * complex.I)
  (h4 : (z1, z2, z3, z4).is_square) : z4 = -2 + complex.I :=
sorry

end fourth_vertex_of_square_l327_327039


namespace find_coordinates_l327_327935

theorem find_coordinates :
  ∃ m n : ℝ, (m + 3)^2 + real.sqrt (4 - n) = 0 ∧ m = -3 ∧ n = 4 :=
by
  use [-3, 4]
  split
  {
    simp,
  },
  {
    ring,
  },
  {
    exact rfl,
  }
sorry

end find_coordinates_l327_327935


namespace count_prime_looking_numbers_below_2000_l327_327304

def is_prime_looking (n : ℕ) : Prop :=
  ¬ prime n ∧ ¬ (∃ k : ℕ, k ∈ {2, 3, 5, 7} ∧ n % k = 0) ∧ (∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n)

def prime_looking_numbers_below (limit : ℕ) :=
  {n | n < limit ∧ is_prime_looking n}

theorem count_prime_looking_numbers_below_2000 : 
  Fintype.card (prime_looking_numbers_below 2000) = 246 :=
sorry

end count_prime_looking_numbers_below_2000_l327_327304


namespace select_team_with_girls_l327_327168

theorem select_team_with_girls 
  (boys girls : ℕ) 
  (team_size min_girls : ℕ) 
  (boys = 7) 
  (girls = 10) 
  (team_size = 5) 
  (min_girls = 2) : 
  (∑ g in finset.range(min_girls, team_size + 1), 
     nat.choose girls g * nat.choose boys (team_size - g)) = 5817 := 
by
  sorry

end select_team_with_girls_l327_327168


namespace root_inequalities_l327_327919

variable {m : ℝ}

def quadratic (m : ℝ) : Type := { f : ℝ → ℝ // (∀ x, f x = x^2 + (m - 1) * x + m^2 - 2) }

theorem root_inequalities (m : ℝ) (f : quadratic m) :
  (f (-1) < 0) ∧ (f 1 < 0) ↔ 0 < m ∧ m < 1 :=
begin
  sorry
end

end root_inequalities_l327_327919


namespace height_to_width_ratio_l327_327682

theorem height_to_width_ratio (w h l : ℝ) (V : ℝ) (x : ℝ) :
  (h = x * w) →
  (l = 7 * h) →
  (V = l * w * h) →
  (V = 129024) →
  (w = 8) →
  (x = 6) :=
by
  intros h_eq_xw l_eq_7h V_eq_lwh V_val w_val
  -- Proof omitted
  sorry

end height_to_width_ratio_l327_327682


namespace number_of_possible_overlap_shapes_l327_327430

-- Define the shapes
inductive Shape
| EquilateralTriangle
| Square
| RegularPentagon
| RegularHexagon

-- Define a predicate that checks if a shape can be the overlap of two triangles
def can_be_overlap (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Square => true
  | Shape.RegularPentagon => true
  | Shape.RegularHexagon => true

-- The main theorem
theorem number_of_possible_overlap_shapes : 
  (Finset.filter can_be_overlap (Finset.univ : Finset Shape)).card = 4 := 
by
  sorry

end number_of_possible_overlap_shapes_l327_327430


namespace sum_of_coprime_numbers_l327_327262

theorem sum_of_coprime_numbers (A B C : ℕ) (h_coprime_AB: Nat.coprime A B) (h_coprime_BC: Nat.coprime B C)
  (h_prod_AB: A * B = 551) (h_prod_BC: B * C = 1073) : A + B + C = 85 :=
by
  sorry

end sum_of_coprime_numbers_l327_327262


namespace find_ellipse_equation_find_range_a_l327_327381

-- Given definitions
def ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def line (x y : ℝ) : Prop :=
  x + 2 * y - 2 = 0

def is_vertex (a x : ℝ) : Prop :=
  x = a

def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt 2) / 2

def segment_intersect (a b x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≤ a

def foci_condition (PF1 PF2 a : ℝ) : Prop :=
  abs (PF1 + PF2) = 2 * a

-- Theorem part (I): Finding the equation of the ellipse
theorem find_ellipse_equation {a b : ℝ} (h1 : a > b) (h2 : b > 0)
  (e : eccentricity a b =(Real.sqrt 2) / 2) (x y : ℝ) :
  (∃ A : ℝ, is_vertex a A → ellipse a b h1 h2 2 0) →   
  ellipse 2 (Real.sqrt 2) h1 h2 x y :=
sorry

-- Theorem part (II): Finding the range of a
theorem find_range_a {a b : ℝ} (h1 : a > b) (h2 : b > 0) (x PF1 PF2 : ℝ) 
  (e : eccentricity a b = (Real.sqrt 2) / 2) :
  (∃ P : ℝ, segment_intersect 2 (Real.sqrt 2) P → foci_condition PF1 PF2 a ∧ line P 0) →
  Real.sqrt (4/3) ≤ a ∧ a ≤ 2 :=
sorry

end find_ellipse_equation_find_range_a_l327_327381


namespace complement_intersection_l327_327995

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end complement_intersection_l327_327995


namespace base_5_division_l327_327342

def base_5_to_nat (n : ℕ) : ℕ :=
n.digits 5.reverse.foldl (λ b a => b * 5 + a) 0

theorem base_5_division :
  let d1 := base_5_to_nat 1324  -- converts 1324_5 to its natural number equivalent
  let d2 := base_5_to_nat 23    -- converts 23_5 to its natural number equivalent
  let q  := base_5_to_nat 41    -- converts 41_5 to its natural number equivalent
  d1 = d2 * q + 1 :=
by
  sorry

end base_5_division_l327_327342


namespace area_ratio_of_circles_l327_327248

theorem area_ratio_of_circles (D_s D_r : ℝ) (h : D_r = 0.5 * D_s) :
  let R_s := D_s / 2,
      R_r := D_r / 2,
      A_s := Real.pi * R_s^2,
      A_r := Real.pi * R_r^2
  in A_r = 0.25 * A_s :=
by
  let R_s := D_s / 2
  let R_r := D_r / 2
  let A_s := Real.pi * R_s^2
  let A_r := Real.pi * R_r^2
  have h1 : R_r = 0.5 * R_s := by
    rw [h]
    simp [R_s, R_r]
  have h2 : A_r = 0.25 * A_s := by
    rw [h1]
    simp [A_s, A_r, Real.pi]
  exact h2

end area_ratio_of_circles_l327_327248


namespace complement_inter_of_A_and_B_l327_327903

open Set

variable (U A B : Set ℕ)

theorem complement_inter_of_A_and_B:
  U = {1, 2, 3, 4, 5}
  ∧ A = {1, 2, 3}
  ∧ B = {2, 3, 4} 
  → U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_inter_of_A_and_B_l327_327903


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327483

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327483


namespace sum_of_ages_53_l327_327301

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53_l327_327301


namespace number_of_type_R_machines_l327_327269

variable (k : ℕ) (R S T : ℕ)

-- Conditions based on the given problem.
def machineR_rate := 1/36
def machineS_rate := 1/24
def machineT_rate := 1/18

def ratioR := 2 * k
def ratioS := 3 * k
def ratioT := 4 * k

def combined_work_rate := ((2 * k) * machineR_rate + (3 * k) * machineS_rate + (4 * k) * machineT_rate)

-- Proof problem to prove that R = 58 given the conditions.
theorem number_of_type_R_machines :=
  combined_work_rate = 1/8 → (29 * k) = 72 → R = 2 * 29 → R = 58 :=
begin
  sorry
end

end number_of_type_R_machines_l327_327269


namespace number_of_prime_pairs_for_10003_l327_327517

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327517


namespace intersection_points_l327_327664

theorem intersection_points (x y : ℝ) (h1 : x^2 - 4 * y^2 = 4) (h2 : x = 3 * y) : 
  (x, y) = (3, 1) ∨ (x, y) = (-3, -1) :=
sorry

end intersection_points_l327_327664


namespace area_of_hexagon_correct_l327_327968

variable (α β γ : ℝ) (S : ℝ) (r R : ℝ)
variable (AB BC AC : ℝ)
variable (A' B' C' : ℝ)

noncomputable def area_of_hexagon (AB BC AC : ℝ) (R : ℝ) (S : ℝ) (r : ℝ) : ℝ :=
  2 * (S / (r * r))

theorem area_of_hexagon_correct
  (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hR : R = 65 / 8) (hS : S = 1344 / 65) :
  area_of_hexagon AB BC AC R S r = 2 * (S / (r * r)) :=
sorry

end area_of_hexagon_correct_l327_327968


namespace distance_PF_l327_327861

-- Conditions as definitions in Lean
def parabola_focus_x (a : ℝ) := a = 1 / 4 * 4
def point_on_parabola (x₀ y₀ : ℝ) := y₀ ^ 2 = 4 * x₀
def distance_to_y_axis (x₀ : ℝ) := x₀ = 2
def focus_point (a : ℝ) := (1, 0)

-- Lean 4 statement
theorem distance_PF (x₀ y₀ : ℝ) (a : ℝ) (h1 : parabola_focus_x a) (h2 : point_on_parabola x₀ y₀) (h3 : distance_to_y_axis x₀) :
  real.dist (x₀, y₀) (1, 0) = 3 := 
by
  sorry

end distance_PF_l327_327861


namespace part_I_a₁_part_I_a₂_part_I_general_part_II_sum_l327_327149

def seq (n : ℕ) : ℕ := 2^(n-1)

def S (n : ℕ) : ℕ :=
  @List.sum ℕ (List.map seq (List.range n))

def seq_na_n (n : ℕ) : ℕ := n * seq n

def T (n : ℕ) : ℕ := 
  @List.sum ℕ (List.map seq_na_n (List.range n))

theorem part_I_a₁ (a1 : ℕ) (h : 2 * a1 - a1 = a1^2) (h_ne : a1 ≠ 0) : a1 = 1 :=
  sorry

theorem part_I_a₂ (a2 : ℕ) (h : 2 * a2 - 1 = 1 + a2) : a2 = 2 :=
  sorry

theorem part_I_general (n : ℕ) : seq n = 2^(n-1) :=
  sorry

theorem part_II_sum (n : ℕ) : T n = 1 + (n-1) * 2^n :=
  sorry

end part_I_a₁_part_I_a₂_part_I_general_part_II_sum_l327_327149


namespace cross_country_winning_scores_l327_327462

theorem cross_country_winning_scores:
  ∀ (scores : Finset ℕ), scores = {n | n ∈ (1:ℕ)..9} → 
  3 * (scores.sum ≤ 15) → 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 6) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 7) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 8) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 9) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 10) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 11) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 12) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 13) ∧ 
  (∃ (S : Finset ℕ), S.card = 3 ∧ S.sum = 14) → 
  9 := 
begin
  -- To be proved
  sorry
end

end cross_country_winning_scores_l327_327462


namespace minimum_S_l327_327879

theorem minimum_S (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  S = (a + 1/a)^2 + (b + 1/b)^2 → S ≥ 8 :=
by
  sorry

end minimum_S_l327_327879


namespace number_of_cannoneers_l327_327715

-- Define the variables for cannoneers, women, and men respectively
variables (C W M : ℕ)

-- Define the conditions as assumptions
def conditions : Prop :=
  W = 2 * C ∧
  M = 2 * W ∧
  M + W = 378

-- Prove that the number of cannoneers is 63
theorem number_of_cannoneers (h : conditions C W M) : C = 63 :=
by sorry

end number_of_cannoneers_l327_327715


namespace a_3_value_l327_327417

def arithmetic_seq (a: ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_value :
  ∃ a : ℕ → ℤ, a 1 = 19 ∧ arithmetic_seq a ∧ a 3 = 13 :=
by
  sorry

end a_3_value_l327_327417


namespace polynomial_factors_l327_327646

theorem polynomial_factors (x : ℝ) : 
  (x^4 - 4*x^2 + 4) = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) :=
by
  sorry

end polynomial_factors_l327_327646


namespace PQCA_cyclic_or_coincide_AC_l327_327385

variables {A B C O : Type} [Triangle A B C] [Circumcenter A B C O]

/-- Points D and E on the angle bisector of ∠ABC such that EA = EB and DB = DC --/
variables (D E : Type) [OnAngleBisector D A B C] [OnAngleBisector E A B C] [Dist EA EB] [Dist DB DC]

/-- P and Q are the circumcenters of (AOE) and (COD), respectively --/
variables (P Q : Type) [Circumcenter A O E P] [Circumcenter C O D Q]

theorem PQCA_cyclic_or_coincide_AC :
  (coincide PQ AC) ∨ Cyclic PQ C A :=
sorry

end PQCA_cyclic_or_coincide_AC_l327_327385


namespace quadratic_two_distinct_real_roots_l327_327845

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (k - 1 ≠ 0 ∧ 8 - 4 * k > 0) ↔ (k < 2 ∧ k ≠ 1) := 
by
  sorry

end quadratic_two_distinct_real_roots_l327_327845


namespace amphibians_count_l327_327433

-- Define the conditions
def frogs : Nat := 7
def salamanders : Nat := 4
def tadpoles : Nat := 30
def newt : Nat := 1

-- Define the total number of amphibians observed by Hunter
def total_amphibians : Nat := frogs + salamanders + tadpoles + newt

-- State the theorem
theorem amphibians_count : total_amphibians = 42 := 
by 
  -- proof goes here
  sorry

end amphibians_count_l327_327433


namespace correct_option_C_l327_327393

-- Define points A, B and C given their coordinates and conditions
structure Point (α : Type _) :=
(x : α)
(y : α)

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

variables (a b c d : ℝ)
variable hA : Point ℝ := ⟨a, 2⟩
variable hB : Point ℝ := ⟨b, 6⟩
variable hC : Point ℝ := ⟨c, d⟩
variables (ha_ON_parabola : hA.y = parabola hA.x)
          (hb_ON_parabola : hB.y = parabola hB.x)
          (hc_ON_parabola : hC.y = parabola hC.x)
          (hd_lt_one : d < 1)

theorem correct_option_C (ha_lt_0 : a < 0) (hb_gt_0 : b > 0) : a < c ∧ c < b :=
by
-- Proof will be done here, currently left as sorry just to state the theorem.
sorry

end correct_option_C_l327_327393


namespace perimeter_ABCD_l327_327125

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

theorem perimeter_ABCD {A B C D E : Type*} 
  (h1 : ∠AEB = 45) (h2 : ∠BEC = 45) (h3 : ∠CED = 45)
  (h4 : right_angle (∠ABE)) (h5 : right_angle (∠BCE)) (h6 : right_angle (∠CDE))
  (h7 : AE = 32)
  (h8 : is_45_45_90_triangle (∆ABE))
  (h9 : is_45_45_90_triangle (∆BCE))
  (h10 : is_45_45_90_triangle (∆CDE)) :
  perimeter ABCD = 32 + 32 * sqrt_2 := sorry

end perimeter_ABCD_l327_327125


namespace probability_without_replacement_probability_with_replacement_l327_327759

-- Definition for without replacement context
def without_replacement_total_outcomes : ℕ := 6
def without_replacement_favorable_outcomes : ℕ := 3
def without_replacement_prob : ℚ :=
  without_replacement_favorable_outcomes / without_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers without replacement is 1/2
theorem probability_without_replacement : 
  without_replacement_prob = 1 / 2 := by
  sorry

-- Definition for with replacement context
def with_replacement_total_outcomes : ℕ := 16
def with_replacement_favorable_outcomes : ℕ := 3
def with_replacement_prob : ℚ :=
  with_replacement_favorable_outcomes / with_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers with replacement is 3/16
theorem probability_with_replacement : 
  with_replacement_prob = 3 / 16 := by
  sorry

end probability_without_replacement_probability_with_replacement_l327_327759


namespace infinite_egyptian_fraction_representation_l327_327187

theorem infinite_egyptian_fraction_representation :
  ∃ (f : ℕ → Finset ℕ), (∀ n, (f n).card > 0 ∧ (∀ (i j : ℕ), i ≠ j → (f i ∩ f j) = ∅)
  ∧ (∀ n, (∑ i in f n, (1 : ℚ) / i) = 1) ∧ ∀ m n, m ≠ n → f m ≠ f n) :=
sorry

end infinite_egyptian_fraction_representation_l327_327187


namespace simplify_expression_l327_327663

theorem simplify_expression (x : ℝ) : (5 * x + 2 * x + 7 * x) = 14 * x :=
by
  sorry

end simplify_expression_l327_327663


namespace right_triangle_in_circle_area_l327_327286

theorem right_triangle_in_circle_area (R α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) :
  let area := (λ R α, R^2 * sin (2 * α) / (sin (2 * α)^2 + 1))
  ∃ (A B C : ℝ), is_right_triangle A B C ∧ inscribed_in_circle_with_radius R A B C ∧ 
  hypotenuse_is_chord A B C ∧ right_angle_on_diameter A B C ∧ 
  area R α =  ∃ (A B C : ℝ), 
sorry

end right_triangle_in_circle_area_l327_327286


namespace height_of_pyramid_l327_327769

-- Definition of the Cube
structure Cube where
  edge_length : ℕ

-- Definition of the Pyramid
structure Pyramid where
  base_edge_length : ℕ
  height : ℚ

-- Equality of volumes condition
theorem height_of_pyramid (c : Cube) (p : Pyramid) (h_eq : ((c.edge_length : ℚ) ^ 3 = (1/3) * (p.base_edge_length : ℚ)^2 * p.height)) : 
  p.height = 3.75 := 
by
  sorry

-- Example usage with the specific values given in the problem
def c : Cube := ⟨5⟩
def p : Pyramid := ⟨10, 3.75⟩

example : height_of_pyramid c p _ := by
  -- You can fill in the proof here
  sorry

end height_of_pyramid_l327_327769


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327477

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327477


namespace base3_minus_base8_5000_l327_327910

-- Conditions involving the number base conversion of 5000
def base3_digits (n : ℕ) : ℕ :=
  let rec aux (k : ℕ) (count : ℕ) : ℕ :=
    if k < 3 then count + 1
    else aux (k / 3) (count + 1)
  in aux n 0

def base8_digits (n : ℕ) : ℕ :=
  let rec aux (k : ℕ) (count : ℕ) : ℕ :=
    if k < 8 then count + 1
    else aux (k / 8) (count + 1)
  in aux n 0

theorem base3_minus_base8_5000 : base3_digits 5000 - base8_digits 5000 = 3 :=
by
  -- skipping the proof steps by adding sorry
  sorry

end base3_minus_base8_5000_l327_327910


namespace ellipse_equation_line_equation_l327_327047

noncomputable def center_xy := (0 : ℝ, 0 : ℝ)
noncomputable def focal_distance := (2 : ℝ)
noncomputable def eccentricity := (1 / 2 : ℝ)
noncomputable def point_M := (0 : ℝ, 1 : ℝ)
noncomputable def distance_AB := (3 * Real.sqrt 5 / 2 : ℝ)

theorem ellipse_equation :
  ∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b), ∃ (eq_ellipse : Prop),
  a = 2 ∧ b = Real.sqrt 3 ∧ eq_ellipse = (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

theorem line_equation :
  ∃ k : ℝ, (k = 1/2 ∨ k = -1/2) ∧ (∀ x y : ℝ, (y = k * x + 1) ↔ (x - 2 * y + 2 = 0 ∨ x + 2 * y - 2 = 0)) :=
sorry

end ellipse_equation_line_equation_l327_327047


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327481

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327481


namespace no_prime_sum_10003_l327_327536

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327536


namespace number_of_ways_sum_of_primes_l327_327494

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327494


namespace krystiana_monthly_income_l327_327965

theorem krystiana_monthly_income :
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  first_floor_income + second_floor_income + third_floor_income = 165 :=
by
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  have h1: first_floor_income = 45 := by simp [first_floor_income]
  have h2: second_floor_income = 60 := by simp [second_floor_income]
  have h3: third_floor_income = 60 := by simp [third_floor_income]
  rw [h1, h2, h3]
  simp
  done

end krystiana_monthly_income_l327_327965


namespace work_day_meeting_percent_l327_327165

open Nat

theorem work_day_meeting_percent :
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 35 := 
by
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  sorry

end work_day_meeting_percent_l327_327165


namespace number_of_subsets_of_A_l327_327850

def A : Set ℕ := {1, 2, 3}

theorem number_of_subsets_of_A : (A.powerset.to_finset.card) = 8 := by
  sorry

end number_of_subsets_of_A_l327_327850


namespace abs_neg_2023_l327_327669

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l327_327669


namespace sum_of_primes_10003_l327_327544

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327544


namespace area_quadrilateral_TUVW_l327_327035

theorem area_quadrilateral_TUVW (PQRS_area : ℝ)
  (h1 : PQRS_area = 120)
  (T_midpoint : ∀ (x y : ℝ), T = (x + y)/2)
  (QU_UR_ratio : ℝ)
  (RV_VS_ratio : ℝ)
  (SW_WP_ratio : ℝ)
  (h2 : QU_UR_ratio = 2/3)
  (h3 : RV_VS_ratio = 3/4)
  (h4 : SW_WP_ratio = 4/5) : 
  67 = area_TUVW PQRS_area T_midpoint QU_UR_ratio RV_VS_ratio SW_WP_ratio := 
sorry

end area_quadrilateral_TUVW_l327_327035


namespace water_left_in_bathtub_l327_327948

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end water_left_in_bathtub_l327_327948


namespace bottle_caps_per_visit_l327_327956

-- Define the given conditions
def total_bottle_caps : ℕ := 25
def number_of_visits : ℕ := 5

-- The statement we want to prove
theorem bottle_caps_per_visit :
  total_bottle_caps / number_of_visits = 5 :=
sorry

end bottle_caps_per_visit_l327_327956


namespace number_of_prime_pairs_for_10003_l327_327519

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327519


namespace meeting_time_proof_l327_327263

def initial_distance : ℝ := 24  -- Initial distance in km
def A_initial_speed : ℝ := 5    -- A's initial speed in kmph
def B_initial_speed : ℝ := 7    -- B's initial speed in kmph
def wind_effect_on_A : ℝ := -1  -- Effect of wind on A's speed in kmph
def wind_effect_on_B : ℝ := 1   -- Effect of wind on B's speed in kmph

def A_effective_speed : ℝ := A_initial_speed + wind_effect_on_A
def B_effective_speed : ℝ := B_initial_speed + wind_effect_on_B

def relative_speed : ℝ := A_effective_speed + B_effective_speed
def travel_time : ℝ := initial_distance / relative_speed  -- Time in hours

-- Start time in hours from 1 pm
def start_time : ℝ := 13  -- 1 pm is 13:00 in 24-hour format

def meeting_time : ℝ := start_time + travel_time

theorem meeting_time_proof : meeting_time = 15 := by
  -- To be filled in with proof steps, if needed
  sorry

end meeting_time_proof_l327_327263


namespace correct_equation_A_incorrect_equation_B_main_theorem_l327_327569

theorem correct_equation_A (x : ℝ) (h : ∀ z, z ≠ 0 → z * 2 = z * (x - 2)) : 
  900 / (x+1) * 2 = 900 / (x-3) :=
by sorry

theorem incorrect_equation_B (y : ℝ) (hy : y ≠ 0) : 
  900 / y - 900 / (2 * y) ≠ 2 :=
by sorry

theorem main_theorem (x y : ℝ) (hx : ∀ z, z ≠ 0 → z * 2 = z * (x - 2)) (hy : y ≠ 0) :
  correct_equation_A x hx ∧ incorrect_equation_B y hy :=
by
  exact ⟨correct_equation_A x hx, incorrect_equation_B y hy⟩

end correct_equation_A_incorrect_equation_B_main_theorem_l327_327569


namespace ball_falls_in_upper_left_pocket_ball_hits_edges_five_times_ball_crosses_twenty_three_squares_l327_327128

variable (grid_size : ℕ := 5)

theorem ball_falls_in_upper_left_pocket :
    (∀ (start_corner : (0, 0)), 
    falls_in_upper_left_pocket (grid_size) start_corner) :=
by
  sorry

theorem ball_hits_edges_five_times :
    (∀ (start_corner : (0, 0)),
    hits_edges_count (grid_size) start_corner = 5) :=
by
  sorry

theorem ball_crosses_twenty_three_squares :
    (∀ (start_corner : (0, 0)), 
    diagonal_squares_crossed (grid_size) start_corner = 23) :=
by
  sorry

end ball_falls_in_upper_left_pocket_ball_hits_edges_five_times_ball_crosses_twenty_three_squares_l327_327128


namespace ways_to_write_10003_as_sum_of_two_primes_l327_327482

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l327_327482


namespace find_a_value_l327_327978

noncomputable def f (a x : ℝ) : ℝ := real.sqrt (a * x ^ 2 - 2 * a * x)

def domain (a : ℝ) : set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem find_a_value (a : ℝ)
  (h_a_neg : a < 0)
  (h_square_region : ∀ m n : ℝ, m ∈ domain a → n ∈ domain a → ∃ s : ℝ, (0 ≤ s) ∧ (f a m, f a n) = (s, s)) :
  a = -4 :=
begin
  sorry
end

end find_a_value_l327_327978


namespace most_likely_number_of_hits_l327_327776

noncomputable def probability_of_hits (k : ℕ) : ℚ :=
  let C := 1 / 19 // or some appropriate constant calculation
  C * (0.8^k) * (0.2^(19-k))

def most_likely_hits : set ℕ := {15, 16}

theorem most_likely_number_of_hits :
  ∃ k ∈ most_likely_hits, ∀ n ∈ finset.range 20, probability_of_hits k ≥ probability_of_hits n :=
begin
  sorry
end

end most_likely_number_of_hits_l327_327776


namespace func4_same_domain_range_as_func1_l327_327793

noncomputable def func1_domain : Set ℝ := {x | 0 < x}
noncomputable def func1_range : Set ℝ := {y | 0 < y}

noncomputable def func4_domain : Set ℝ := {x | 0 < x}
noncomputable def func4_range : Set ℝ := {y | 0 < y}

theorem func4_same_domain_range_as_func1 :
  (func4_domain = func1_domain) ∧ (func4_range = func1_range) :=
sorry

end func4_same_domain_range_as_func1_l327_327793


namespace possible_third_side_of_triangle_l327_327100

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l327_327100


namespace rearrange_rooks_possible_l327_327649

theorem rearrange_rooks_possible (board : Fin 8 × Fin 8 → Prop) (rooks : Fin 8 → Fin 8 × Fin 8) (painted : Fin 8 × Fin 8 → Prop) :
  (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) → -- no two rooks are in the same row or column
  (∃ (unpainted_count : ℕ), (unpainted_count = 64 - 27)) → -- 27 squares are painted red
  (∃ new_rooks : Fin 8 → Fin 8 × Fin 8,
    (∀ i : Fin 8, ¬painted (new_rooks i)) ∧ -- all rooks are on unpainted squares
    (∀ i j : Fin 8, i ≠ j → (new_rooks i).1 ≠ (new_rooks j).1 ∧ (new_rooks i).2 ≠ (new_rooks j).2) ∧ -- no two rooks are in the same row or column
    (∃ i : Fin 8, rooks i ≠ new_rooks i)) -- at least one rook has moved
:=
sorry

end rearrange_rooks_possible_l327_327649


namespace cube_divisors_202_l327_327091

theorem cube_divisors_202 (x : ℕ) (d : ℕ) 
  (hx : ∃ n : ℕ, x = n^3) 
  (hd : d = ∏ p in (prime_factors x).to_finset, (3 * p.2 + 1)) 
  : d = 202 :=
sorry

end cube_divisors_202_l327_327091


namespace find_angle_B_l327_327459

theorem find_angle_B
  (a : ℝ) (c : ℝ) (A B C : ℝ)
  (h1 : a = 5 * Real.sqrt 2)
  (h2 : c = 10)
  (h3 : A = π / 6) -- 30 degrees in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  : B = 7 * π / 12 ∨ B = π / 12 := -- 105 degrees or 15 degrees in radians
sorry

end find_angle_B_l327_327459


namespace number_of_ways_sum_of_primes_l327_327502

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327502


namespace average_salary_excluding_manager_l327_327119

theorem average_salary_excluding_manager (A : ℝ) 
  (num_employees : ℝ := 20)
  (manager_salary : ℝ := 3300)
  (salary_increase : ℝ := 100)
  (total_salary_with_manager : ℝ := 21 * (A + salary_increase)) :
  20 * A + manager_salary = total_salary_with_manager → A = 1200 := 
by
  intro h
  sorry

end average_salary_excluding_manager_l327_327119


namespace range_of_expressions_l327_327084

theorem range_of_expressions (x y : ℝ) (h1 : 30 < x ∧ x < 42) (h2 : 16 < y ∧ y < 24) :
  46 < x + y ∧ x + y < 66 ∧ -18 < x - 2 * y ∧ x - 2 * y < 10 ∧ (5 / 4) < (x / y) ∧ (x / y) < (21 / 8) :=
sorry

end range_of_expressions_l327_327084


namespace no_prime_sum_10003_l327_327534

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327534


namespace estimate_rabbit_population_l327_327227

theorem estimate_rabbit_population :
  ∀ (initially_marked : ℕ) (total_second_capture : ℕ) (marked_second_capture : ℕ),
  initially_marked = 50 →
  total_second_capture = 42 →
  marked_second_capture = 5 →
  (initially_marked * total_second_capture) / marked_second_capture = 420 :=
by
  intros initially_marked total_second_capture marked_second_capture h1 h2 h3
  rw [h1, h2, h3]
  sorry

end estimate_rabbit_population_l327_327227


namespace largest_parallelogram_perimeter_l327_327012

def triangle := {a b c : ℝ // a = 13 ∧ b = 13 ∧ c = 12}

def parallelogram_perimeter (t : triangle) (n : ℕ) : ℝ :=
  if h : n = 4 then
    let perimeter_edges := [13, 13, 13, 13, 12, 12] in
    2 * (perimeter_edges.sum)
  else
    0 -- This case should not happen as per our condition of forming a parallelogram

theorem largest_parallelogram_perimeter :
  ∀ (t : triangle), parallelogram_perimeter t 4 = 76 := 
by
  sorry

end largest_parallelogram_perimeter_l327_327012


namespace find_missing_number_l327_327321

theorem find_missing_number (x : ℤ) : abs (x + 72) - 6 = 73 → x = 7 :=
by
  intro h
  have h1 : abs (x + 72) - 6 + 6 = 73 + 6 := by linarith
  rw [sub_add_cancel, abs_eq_iff] at h1
  rcases h1 with h1 | h1
  { linarith }
  { exfalso; linarith }

end find_missing_number_l327_327321


namespace donation_student_amount_l327_327934

theorem donation_student_amount (a : ℕ) : 
  let total_amount := 3150
  let teachers_count := 5
  let donation_teachers := teachers_count * a 
  let donation_students := total_amount - donation_teachers
  donation_students = 3150 - 5 * a :=
by
  sorry

end donation_student_amount_l327_327934


namespace max_PM_PN_l327_327063

noncomputable def C1 := { p : ℝ × ℝ | (p.1 / 2)^2 + (p.2 / (Real.sqrt 3))^2 = 1 }
noncomputable def C2 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

theorem max_PM_PN (M N : ℝ × ℝ) (P : ℝ × ℝ) 
  (hM : M ∈ C1) (hN : N ∈ C1) (hP : P ∈ C2) :
  ∃ M N, is_vertex_of_C1 M ∧ is_vertex_of_C1 N ∧ 
         ∀ P, P ∈ C2 → PM_PN_criterion M N P (|P - M| + |P - N|) ≤ 2 * Real.sqrt 7 :=
begin
  sorry
end

end max_PM_PN_l327_327063


namespace total_nap_duration_l327_327967

def nap1 : ℚ := 1 / 5
def nap2 : ℚ := 1 / 4
def nap3 : ℚ := 1 / 6
def hour_to_minutes : ℚ := 60

theorem total_nap_duration :
  (nap1 + nap2 + nap3) * hour_to_minutes = 37 := by
  sorry

end total_nap_duration_l327_327967


namespace no_prime_sum_10003_l327_327468

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327468


namespace height_of_cylinder_is_10_l327_327429

-- Definitions and conditions
def cone_radius := 4 -- radius of the cone in meters
def cone_height := 2 -- height of the cone in meters
def cylinder_radius := 12 -- radius of the cylinder in meters
def num_cones := 135 -- number of cones formed from the cylinder

-- Formula for the volume of a cone
def volume_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Formula for the volume of a cylinder
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Mathematical problem statement
theorem height_of_cylinder_is_10 :
  ∃ H : ℝ, volume_cylinder cylinder_radius H = num_cones * volume_cone cone_radius cone_height ∧ H = 10 :=
sorry

end height_of_cylinder_is_10_l327_327429


namespace correct_option_C_l327_327388

noncomputable theory

def parabola (x : ℝ) : ℝ :=
  (x - 1)^2 - 2

theorem correct_option_C (a b c d : ℝ)
  (ha : a < 0) 
  (hb : b > 0)
  (hc : c = a ∨ c = b ∨ (a < c ∧ c < b))
  (hd : d < 1)
  (hA : parabola a = 2)
  (hB : parabola b = 6)
  (hC : parabola c = d) :
  a < c ∧ c < b := 
sorry

end correct_option_C_l327_327388


namespace correct_conclusions_count_l327_327976

theorem correct_conclusions_count (a b c : ℝ) (ha : 3^a = 4^b ∧ 4^b = 6^c) :
    (if (a > 0 ∧ b > 0 ∧ c > 0 → 3 * a < 4 * b ∧ 4 * b < 6 * c) then 1 else 0) +
    (if (a > 0 ∧ b > 0 ∧ c > 0 → 2 / c = 1 / a + 2 / b) then 1 else 0) +
    (if (a < 0 ∧ b < 0 ∧ c < 0 → a < b ∧ b < c) then 1 else 0) = 2 :=
sorry

end correct_conclusions_count_l327_327976


namespace triangle_inequality_l327_327602

-- Let ABC be a triangle
variables (A B C X Y : Point)
variables (triangle_ABC : Triangle A B C)

-- Assume X is the tangency point of the incircle with BC
axiom X_is_tangency_point : incircle_Tangency B C X

-- Assume Y is the second intersection point of segment AX with the incircle
axiom Y_is_second_intersection_point : incircle_SecondIntersection A X Y

-- Aim: Prove that AX + AY + BC > AB + AC given the above assumptions
theorem triangle_inequality :
  AX + AY + BC > AB + AC :=
by {
  sorry
}

end triangle_inequality_l327_327602


namespace find_a_monotonic_intervals_l327_327408

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x ^ 2 - a * Real.log x

-- Step (1): If f(x) attains an extreme value at x = 2, find the value of a
theorem find_a (a : ℝ) (h₁ : ∀ x, Differentiable ℝ (f x a))
  (h₂ : (deriv (λ x, f x a) 2) = 0) : a = 4 := sorry

-- Step (2): Determine the intervals where f(x) is monotonic
theorem monotonic_intervals (a : ℝ) (h₁ : 0 < a) :
  (∀ x, f' x < 0 ↔ 0 < x ∧ x < sqrt a) ∧
  (∀ x, f' x > 0 ↔ sqrt a < x) := sorry

end find_a_monotonic_intervals_l327_327408


namespace francie_remaining_money_l327_327016

noncomputable def total_savings_before_investment : ℝ :=
  (5 * 8) + (6 * 6) + 20

noncomputable def investment_return : ℝ :=
  0.05 * 10

noncomputable def total_savings_after_investment : ℝ :=
  total_savings_before_investment + investment_return

noncomputable def spent_on_clothes : ℝ :=
  total_savings_after_investment / 2

noncomputable def remaining_after_clothes : ℝ :=
  total_savings_after_investment - spent_on_clothes

noncomputable def amount_remaining : ℝ :=
  remaining_after_clothes - 35

theorem francie_remaining_money : amount_remaining = 13.25 := 
  sorry

end francie_remaining_money_l327_327016


namespace set_intersection_l327_327068

theorem set_intersection :
  let A := {x : ℝ | 0 < x}
  let B := {x : ℝ | -1 ≤ x ∧ x < 3}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := 
by
  sorry

end set_intersection_l327_327068


namespace mojave_population_increase_factor_l327_327209

open Real

def mojave_population_factor
  (population_10_years_ago : ℝ) 
  (future_population_5_years : ℝ) 
  (increase_rate_5_years : ℝ) : ℝ :=
  let current_population := (100 / increase_rate_5_years) * future_population_5_years
  current_population / population_10_years_ago

theorem mojave_population_increase_factor
  (population_10_years_ago : ℝ := 4000) 
  (future_population_5_years : ℝ := 16800) 
  (increase_rate_5_years : ℝ := 140) :
  mojave_population_factor population_10_years_ago future_population_5_years increase_rate_5_years = 3 := 
sorry

end mojave_population_increase_factor_l327_327209


namespace area_of_circle_region_l327_327731

-- Definitions of the circle and the line
def circle_eqn (x y : ℝ) : Prop := x^2 - 12*x + y^2 = 28
def line_eqn (x y : ℝ) : Prop := y = x - 4

-- Definition of the correct area
def correct_area : ℝ := 48 * Real.pi

-- Statement to be proved
theorem area_of_circle_region (x y : ℝ) :
  (circle_eqn x y) →
  (y ≥ 0) →
  (line_eqn x y → x ≥ 6) ∨ (x < 6) →
  calc_area x y = correct_area := 
sorry

end area_of_circle_region_l327_327731


namespace sin_theta_plus_pi_over_six_l327_327020

theorem sin_theta_plus_pi_over_six (θ : ℝ) (h : sin θ + sin (θ + π / 3) = 1) : sin (θ + π / 6) = sqrt 3 / 3 :=
by
  sorry

end sin_theta_plus_pi_over_six_l327_327020


namespace root_in_interval_l327_327889

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_in_interval : ∃ x ∈ Ioo (1 : ℝ) 2, f x = 0 :=
by
  sorry

end root_in_interval_l327_327889


namespace base_5_division_l327_327341

def base_5_to_nat (n : ℕ) : ℕ :=
n.digits 5.reverse.foldl (λ b a => b * 5 + a) 0

theorem base_5_division :
  let d1 := base_5_to_nat 1324  -- converts 1324_5 to its natural number equivalent
  let d2 := base_5_to_nat 23    -- converts 23_5 to its natural number equivalent
  let q  := base_5_to_nat 41    -- converts 41_5 to its natural number equivalent
  d1 = d2 * q + 1 :=
by
  sorry

end base_5_division_l327_327341


namespace total_students_in_class_l327_327929

-- Definitions of the conditions
def E : ℕ := 55
def T : ℕ := 85
def N : ℕ := 30
def B : ℕ := 20

-- Statement of the theorem to prove the total number of students
theorem total_students_in_class : (E + T - B) + N = 150 := by
  -- Proof is omitted
  sorry

end total_students_in_class_l327_327929


namespace james_writing_time_l327_327595

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l327_327595


namespace no_prime_sum_10003_l327_327467

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l327_327467


namespace books_leftover_l327_327290

-- Definitions of the conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def books_bought : ℕ := 26

-- The theorem stating the proof problem
theorem books_leftover : (initial_books + books_bought) - (shelves * books_per_shelf) = 2 := by
  sorry

end books_leftover_l327_327290


namespace sum_of_midpoint_coords_l327_327679

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 3) (hy1 : y1 = 5) (hx2 : x2 = 11) (hy2 : y2 = 21) :
  ((x1 + x2) / 2 + (y1 + y2) / 2) = 20 :=
by
  sorry

end sum_of_midpoint_coords_l327_327679


namespace john_profit_l327_327954

theorem john_profit (cost_per_bag selling_price : ℕ) (number_of_bags : ℕ) (profit_per_bag total_profit : ℕ) :
  cost_per_bag = 4 →
  selling_price = 8 →
  number_of_bags = 30 →
  profit_per_bag = selling_price - cost_per_bag →
  total_profit = number_of_bags * profit_per_bag →
  total_profit = 120 :=
by
  intro h_cost h_sell h_num_bags h_profit_per_bag h_total_profit
  rw [h_profit_per_bag, h_cost, h_sell] at h_profit_per_bag
  rw [h_total_profit, h_num_bags, h_profit_per_bag]
  norm_num

end john_profit_l327_327954


namespace f_monotonic_decreasing_interval_l327_327686

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2*x)

theorem f_monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x2 ≤ f x1 := 
sorry

end f_monotonic_decreasing_interval_l327_327686


namespace a_n_is_geometric_sequence_b_n_general_formula_exists_max_k_l327_327817

open Nat

-- Defining the sequence {a_n} and conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ)
axiom h1 : ∀ n : ℕ, (3 - m) * S n + 2 * m * a n = m + 3
axiom h_m_nonzero : m ≠ -3 ∧ m ≠ 0

-- Proving that a_n forms a geometric sequence
theorem a_n_is_geometric_sequence : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n := by
  exists (2 * m / (3 + m))
  sorry

-- Defining the sequence {b_n} and its conditions
variable (b : ℕ → ℝ)
axiom b1 : b 1 = 1
axiom b_rec : ∀ n : ℕ, n ≥ 2 → b n = (3 / 2) * (2 * b (n - 1)) / (b (n - 1) + 3)

-- Proving the general formula for b_n
theorem b_n_general_formula : ∀ n : ℕ, n > 0 → b n = 3 / (n + 2) := by
  intro n h
  sorry

-- Given m = 1, defining T_n
noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), (i + 1) * (a i)

-- Proving existence of maximum k = 7 such that T_n > k / 8 for all n
theorem exists_max_k : ∃ k : ℕ, k = 7 ∧ ( ∀ n : ℕ, T n > k / 8) := by
  sorry

end a_n_is_geometric_sequence_b_n_general_formula_exists_max_k_l327_327817


namespace complex_product_conjugate_l327_327405

noncomputable def z := (√3 + Complex.i) / (1 - √3 * Complex.i) ^ 2
noncomputable def z_conjugate := Complex.conj z

theorem complex_product_conjugate :
  z * z_conjugate = 1 / 4 :=
by
  sorry

end complex_product_conjugate_l327_327405


namespace misha_third_place_l327_327167

-- Define participants and their statements
inductive Participant
| Misha | Anton | Katya | Natasha
deriving DecidableEq

open Participant

-- Define the statements made by each participant
def statement (p : Participant) : Prop :=
  match p with
  | Misha => Misha ≠ first ∧ Misha ≠ last
  | Anton => Anton ≠ last
  | Katya => Katya = first
  | Natasha => Natasha = last

-- Define a function to assert a participant's position
def position (p : Participant) : Nat := sorry

-- Conditions
def truthful (p : Participant) : Prop := sorry
def lied (p : Participant) : Prop := sorry

-- The third place is known to be a boy
def is_boy (p : Participant) : Prop :=
  p = Misha ∨ p = Anton

-- The problem statement
theorem misha_third_place :
  (∃ p, lied p) →
  (∀ p, truthful p → statement p) →
  (∃ p, lied p → ¬ statement p) →
  (position Misha ≠ first) →
  (position Misha ≠ last) →
  (position Anton ≠ last) →
  (position Katya = first) →
  (position Natasha = last) →
  (is_boy Misha ∨ is_boy Anton) →
  position Misha = 3 := 
sorry

end misha_third_place_l327_327167


namespace average_age_of_choir_l327_327928

theorem average_age_of_choir (n_f n_m : ℕ) (A_f A_m : ℕ) (H1 : n_f = 12) (H2 : A_f = 28)
   (H3 : n_m = 18) (H4 : A_m = 40) : 
   (n_f * A_f + n_m * A_m) / (n_f + n_m) = 35.2 :=
by 
   sorry

end average_age_of_choir_l327_327928


namespace sum_of_squares_l327_327108

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hwx : w * x = a^2) 
  (hwy : w * y = b^2) 
  (hwz : w * z = c^2) 
  (hw : w ≠ 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := 
by
  sorry

end sum_of_squares_l327_327108


namespace sum_of_two_primes_unique_l327_327489

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327489


namespace cubic_ineq_solution_l327_327361

theorem cubic_ineq_solution (x : ℝ) :
  (4 < x ∧ x < 4 + 2 * Real.sqrt 3) ∨ (x > 4 + 2 * Real.sqrt 3) → (x^3 - 12 * x^2 + 44 * x - 16 > 0) :=
by
  sorry

end cubic_ineq_solution_l327_327361


namespace arctan_tan_diff_l327_327811

open Real

-- Definitions related to the problem
def tan_75 := tan 75

def tan_20 := tan 20

-- The theorem statement
theorem arctan_tan_diff :
  ∃ θ, 0 ≤ θ ∧ θ ≤ 180 ∧ θ = arctan (tan 75 - 3 * tan 20) := sorry

end arctan_tan_diff_l327_327811


namespace savings_during_sale_l327_327600

-- Definitions for conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def normal_price_per_ball_bearing : ℕ := 1 -- dollar
def sale_price_per_ball_bearing : ℝ := 0.75 -- dollars
def bulk_discount : ℝ := 0.20

-- Statement of the theorem
theorem savings_during_sale :
  let total_ball_bearings := machines * ball_bearings_per_machine in
  let normal_cost := total_ball_bearings * normal_price_per_ball_bearing in
  let sale_cost := total_ball_bearings * sale_price_per_ball_bearing in
  let bulk_discount_amount := sale_cost * bulk_discount in
  let final_price := sale_cost - bulk_discount_amount in
  let savings := normal_cost - final_price in
  savings = 120 := 
  by
    -- proof to be filled in
    sorry

end savings_during_sale_l327_327600


namespace common_vertex_angle_l327_327782

-- Definitions of the internal angles based on given conditions
def equilateral_triangle_internal_angle := 60
def regular_pentagon_internal_angle := 108

-- Statement of the problem in Lean 4
theorem common_vertex_angle (P T Q : Point) (circle : Circle)
  (PTU : Triangle) (PQRST : Pentagon)
  (HPT : PTU.has_vertex P) (HPQ : PQRST.has_vertex P)
  (inscribed_triangle : circle.inscribed PTU)
  (inscribed_pentagon : circle.inscribed PQRST) :
  measure (angle T P Q) = 24 :=
by
  -- Definitions and conditions will be used here
  let triangle_angle := equilateral_triangle_internal_angle
  let pentagon_angle := regular_pentagon_internal_angle
  -- Proof goes here
  sorry

end common_vertex_angle_l327_327782


namespace fred_weekly_allowance_l327_327363

variable (A : ℝ) -- Fred's weekly allowance

-- Condition 1: Fred spent half of his allowance on movie tickets.
def spent_on_tickets (A : ℝ) := A / 2

-- Condition 2: Fred earned 6 dollars for washing the family car.
def earn_car_washing := 6

-- Condition 3: Fred earned 5 dollars for mowing the neighbor's lawn.
def earn_lawn_mowing := 5

-- Condition 4: At the end of the day, Fred counted a total of 20 dollars.
def total_money_at_end := 20

-- Computation to verify the allowance
theorem fred_weekly_allowance : (spent_on_tickets A) + (earn_car_washing + earn_lawn_mowing) = total_money_at_end → A = 18 :=
by
  intro h
  -- Start with the assumption in the hypothesis
  rw [spent_on_tickets, add_assoc, add_comm] at h
  norm_num at h
  sorry

end fred_weekly_allowance_l327_327363


namespace setC_not_basis_l327_327150

-- Definitions based on the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V)
variables (v₁ v₂ : V)

-- Assuming e₁ and e₂ are non-collinear
axiom non_collinear : ¬Collinear ℝ {e₁, e₂}

-- The vectors in the set C
def setC_v1 : V := 3 • e₁ - 2 • e₂
def setC_v2 : V := 4 • e₂ - 6 • e₁

-- The proof problem statement
theorem setC_not_basis : Collinear ℝ {setC_v1 e₁ e₂, setC_v2 e₁ e₂} :=
sorry

end setC_not_basis_l327_327150


namespace ellipse_standard_eq_isosceles_triangle_x_axis_l327_327382

noncomputable theory

variables {x y a b : ℝ} (C : set (ℝ × ℝ))
variables (P : ℝ × ℝ) (A B : ℝ × ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def standard_ellipse (x y : ℝ) := (x ^ 2) / 18 + (y ^ 2) / 8 = 1

def point_P := (3 : ℝ, 2 : ℝ)

def eccentricity (a b : ℝ) := b / a

def line_l (x y t : ℝ) := 2 * x - 3 * y + t = 0

def isosceles_with_x_axis (A B P : ℝ × ℝ) :=
  let slope_AP := (A.2 - P.2) / (A.1 - P.1),
      slope_BP := (B.2 - P.2) / (B.1 - P.1) in
  slope_AP = - slope_BP

theorem ellipse_standard_eq (H : a > b ∧ b > 0 ∧ sqrt 5 / a = sqrt 5 / 3) (H1 : ellipse 3 2 a b) :
  standard_ellipse x y :=
sorry

theorem isosceles_triangle_x_axis (H2 : ∀ t, line_l x y t → line_l a b t) : 
  isosceles_with_x_axis A B P :=
sorry

end ellipse_standard_eq_isosceles_triangle_x_axis_l327_327382


namespace xy_plus_2y_l327_327089

theorem xy_plus_2y (x y : ℝ) (h : x * (x + y) = x^2 + 12) : xy + 2y = 12 + 2y :=
by sorry

end xy_plus_2y_l327_327089


namespace prob_X_less_4_minus_a_l327_327457

open ProbabilityTheory

def X : ProbabilityTheory.Measure ℝ := sorry -- Define the distribution of X later

axiom normal_X : X = Measure.fromDensity (pdf normal 2 (2^2))
axiom P_X_less_a (a : ℝ) : P (X < a) = 0.2

theorem prob_X_less_4_minus_a (a : ℝ) :
  P (X < 4 - a) = 0.8 :=
sorry

end prob_X_less_4_minus_a_l327_327457


namespace base5_division_l327_327339

theorem base5_division :
  let base5_1324 := 1324 
  let base5_23 := 23 
  (base5_div base5_1324 base5_23) = 31.21 := by
  sorry

end base5_division_l327_327339


namespace proof_problem_l327_327065

open Real

-- Let p and q be the given propositions
def p := ∃ x_0 ∈ Iio 0, 2^x_0 < 3^x_0
def q := ∀ x ∈ Ioo 0 (π / 2), sin x < x

-- We need to prove the compound proposition
theorem proof_problem : (¬ p) ∧ q :=
by sorry

end proof_problem_l327_327065


namespace shaded_quadrilateral_area_l327_327698

theorem shaded_quadrilateral_area :
  ∀ (side1 side2 side3 : ℝ), side1 = 3 → side2 = 5 → side3 = 7 →
  let total_length := side1 + side2 + side3 in
  let slope := side3 / total_length in
  let height1 := slope * side1 in
  let height2 := slope * (side1 + side2) in
  let base1 := height1 in
  let base2 := height2 in
  let height := side2 in
  (height * (base1 + base2)) / 2 = 12.8325 :=
by
  intros side1 side2 side3 h1 h2 h3
  let total_length := side1 + side2 + side3
  let slope := side3 / total_length
  let height1 := slope * side1
  let height2 := slope * (side1 + side2)
  let base1 := height1
  let base2 := height2
  let height := side2
  have area := (height * (base1 + base2)) / 2
  show area = 12.8325
  sorry

end shaded_quadrilateral_area_l327_327698


namespace contrapositive_of_similar_triangles_l327_327794

open Classical

theorem contrapositive_of_similar_triangles (ΔABC ΔDEF : Triangle) :
  (∀ (X Y Z : Triangle), Similar X Y → ∀ (a b : ℝ), Angle_ABC X a b → Angle_DEF Y a b → Angle_ABC X a b = Angle_DEF Y a b) ↔
  (∀ (X Y Z : Triangle), ¬ (∀ (a b: ℝ), Angle_ABC X a b = Angle_DEF Y a b) → ¬ (Similar X Y)) := 
sorry

end contrapositive_of_similar_triangles_l327_327794


namespace find_x_plus_y_l327_327855

theorem find_x_plus_y (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x - y > 0) : x + y = 8 ∨ x + y = 2 :=
by
  sorry

end find_x_plus_y_l327_327855


namespace fixed_points_of_f_when_a1_bneg2_range_of_a_for_two_distinct_fixedpoints_l327_327844

noncomputable theory

-- Define the fixed point concept and prove the fixed points for a=1, b=-2
theorem fixed_points_of_f_when_a1_bneg2 : 
  {x : ℝ | (λ x, x^2 - x - 3) x = x} = {3, -1} :=
sorry

-- Define the conditions for the range of a
theorem range_of_a_for_two_distinct_fixedpoints (a b : ℝ) (h: ∀ b : ℝ, (b^2 - 4 * a * (b - 1) > 0)) : 
  0 < a ∧ a < 1 :=
sorry

end fixed_points_of_f_when_a1_bneg2_range_of_a_for_two_distinct_fixedpoints_l327_327844


namespace problem_statement_l327_327058

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then log x / log 0.5 else 1 - 2 ^ x

theorem problem_statement : f (f 2) = 1 / 2 :=
by sorry

end problem_statement_l327_327058


namespace find_coefficients_l327_327458

theorem find_coefficients (k b : ℝ) :
    (∀ x y : ℝ, (y = k * x) → ((x-2)^2 + y^2 = 1) → (2*x + y + b = 0)) →
    ((k = 1/2) ∧ (b = -4)) :=
by
  sorry

end find_coefficients_l327_327458


namespace scooterValue_after_4_years_with_maintenance_l327_327219

noncomputable def scooterDepreciation (initial_value : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((3 : ℝ) / 4) ^ years

theorem scooterValue_after_4_years_with_maintenance (M : ℝ) :
  scooterDepreciation 40000 4 - 4 * M = 12656.25 - 4 * M :=
by
  sorry

end scooterValue_after_4_years_with_maintenance_l327_327219


namespace sum_of_primes_10003_l327_327549

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327549


namespace problem1_problem2_problem3_l327_327891

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Problem 1: Relationship between f(x) and f(1/x)
theorem problem1 (x : ℝ) (hx : x ≠ 0) : f(x) + f(1/x) = 1 :=
sorry

-- Problem 2: Sum of function values
theorem problem2 : 
  f(1) + (Finset.sum (Finset.range 2010) (λ n, f(n+2)) + Finset.sum (Finset.range 2010) (λ n, f(1/(n+2)))) = 2009 + 1/2 :=
sorry

-- Problem 3: Monotonicity of the function
theorem problem3 : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f(x1) < f(x2) :=
sorry

end problem1_problem2_problem3_l327_327891


namespace sin_theta_pi_over_6_eq_sqrt3_over3_l327_327022

theorem sin_theta_pi_over_6_eq_sqrt3_over3 (θ : ℝ) (h : sin θ + sin (θ + π / 3) = 1) : 
  sin (θ + π / 6) = sqrt 3 / 3 := 
sorry

end sin_theta_pi_over_6_eq_sqrt3_over3_l327_327022


namespace translate_line_find_lambda_l327_327229

noncomputable def find_lambda (λ : ℝ) : Prop :=
  let line_translated : ℝ → ℝ → Prop := λ x y, 3 * x - 4 * y + λ + 3 = 0
  let circle : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + (y - 2)^2 = 1
  ∃ x y, circle x y ∧ line_translated 1 2 ∧ ((abs (3 - 8 + λ + 3) / sqrt (9 + 16)) = 1)

theorem translate_line_find_lambda (λ : ℝ) :
  (λ = -3 ∨ λ = 7) ↔ (∃ x y, (circle x y) ∧ (line_translated 1 2) ∧ ((abs (3 - 8 + λ + 3) / sqrt (9 + 16)) = 1)) :=
begin
  sorry
end

end translate_line_find_lambda_l327_327229


namespace sin_sum_to_product_l327_327331

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sin_sum_to_product_l327_327331


namespace energy_fraction_l327_327173

-- Conditions
variables (E : ℝ → ℝ)
variable (x : ℝ)
variable (h : ∀ x, E (x + 1) = 31.6 * E x)

-- The statement to be proven
theorem energy_fraction (x : ℝ) (h : ∀ x, E (x + 1) = 31.6 * E x) : 
  E (x - 1) / E x = 1 / 31.6 :=
by
  sorry

end energy_fraction_l327_327173


namespace find_value_l327_327909

-- Define the variables and given conditions
variables (x y z : ℚ)
variables (h1 : 2 * x - y = 4)
variables (h2 : 3 * x + z = 7)
variables (h3 : y = 2 * z)

-- Define the goal to prove
theorem find_value : 6 * x - 3 * y + 3 * z = 51 / 4 := by 
  sorry

end find_value_l327_327909


namespace smallest_w_l327_327745

theorem smallest_w (w : ℕ) (h1 : Nat.gcd 1452 w = 1) (h2 : 2 ∣ w ∧ 3 ∣ w ∧ 13 ∣ w) :
  (∃ (w : ℕ), 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w ∧ w > 0) ∧
  ∀ (w' : ℕ), (2^4 ∣ 1452 * w' ∧ 3^3 ∣ 1452 * w' ∧ 13^3 ∣ 1452 * w' ∧ w' > 0) → w ≤ w' :=
  sorry

end smallest_w_l327_327745


namespace find_square_digit_l327_327194

-- Define the known sum of the digits 4, 7, 6, and 9
def sum_known_digits := 4 + 7 + 6 + 9

-- Define the condition that the number 47,69square must be divisible by 6
def is_multiple_of_6 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∧ (sum_known_digits + d) % 3 = 0

-- Theorem statement that verifies both the conditions and finds possible values of square
theorem find_square_digit (d : ℕ) (h : is_multiple_of_6 d) : d = 4 ∨ d = 8 :=
by sorry

end find_square_digit_l327_327194


namespace workbooks_needed_l327_327785

theorem workbooks_needed (classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) (total_workbooks : ℕ) :
  classes = 25 → workbooks_per_class = 144 → spare_workbooks = 80 → total_workbooks = 25 * 144 + 80 → 
  total_workbooks = classes * workbooks_per_class + spare_workbooks :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end workbooks_needed_l327_327785


namespace quadratic_inequality_solution_l327_327913

variable (a x : ℝ)

theorem quadratic_inequality_solution (h : 0 < a ∧ a < 1) : (x - a) * (x - (1 / a)) > 0 ↔ (x < a ∨ x > 1 / a) :=
sorry

end quadratic_inequality_solution_l327_327913


namespace Sasha_earnings_proof_l327_327186

def Monday_hours : ℕ := 90  -- 1.5 hours * 60 minutes/hour
def Tuesday_minutes : ℕ := 75  -- 1 hour * 60 minutes/hour + 15 minutes
def Wednesday_minutes : ℕ := 115  -- 11:10 AM - 9:15 AM
def Thursday_minutes : ℕ := 45

def total_minutes_worked : ℕ := Monday_hours + Tuesday_minutes + Wednesday_minutes + Thursday_minutes

def hourly_rate : ℚ := 4.50
def total_hours : ℚ := total_minutes_worked / 60

def weekly_earnings : ℚ := total_hours * hourly_rate

theorem Sasha_earnings_proof : weekly_earnings = 24 := by
  sorry

end Sasha_earnings_proof_l327_327186


namespace S_is_line_l327_327974

open Complex

def is_real (z : ℂ) : Prop := z.im = 0

def S : Set ℂ := {z : ℂ | is_real ((1 + 2i) * z)}

theorem S_is_line : S = {z : ℂ | ∃ (x : ℝ), z = x - 2 * x * I} :=
by
  sorry

end S_is_line_l327_327974


namespace chord_intersects_inner_circle_prob_l327_327712

theorem chord_intersects_inner_circle_prob
    (r_inner r_outer : ℝ)
    (P Q : ℝ × ℝ)
    (h_inner_pos: r_inner > 0)
    (h_outer_gt_inner : r_outer > r_inner)
    (h_P_on_outer : P.1^2 + P.2^2 = r_outer^2)
    (h_P_top : P = (0, r_outer))
    (h_Q_uniform : on.uniform_random Q)
    : probability (chord_intersects_inner_circle r_inner r_outer P Q) = 1 / 3 := sorry

end chord_intersects_inner_circle_prob_l327_327712


namespace find_RS_in_triangle_DEF_l327_327134

noncomputable def triangle_RS : ℝ :=
let DE := 130, DF := 110, EF := 105,
    EU := EF, DV := DF, DE := DE in
let UV := EU + DV - DE in
UV / 2

theorem find_RS_in_triangle_DEF (DE DF EF : ℝ) (EU DV : ℝ) :
  DE = 130 ∧ DF = 110 ∧ EF = 105 ∧
  EU = EF ∧ DV = DF →
  triangle_RS = 42.5 :=
by
  intros h
  rw [h.1, h.2, h.3, h.4, h.5]
  norm_num
  let UV := 105 + 110 - 130
  calc UV / 2 = 85 / 2 : by norm_num
        ... = 42.5 : by norm_num

end find_RS_in_triangle_DEF_l327_327134


namespace matrix_expr_value_l327_327843

-- Given real numbers x, y, and z such that the matrix is not invertible (det A = 0)
-- and x + y + z = 0, prove that the value of the expression is -3.

noncomputable def matrixA (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + y, x, y], ![x, y + z, y], ![y, x, x + z]]

theorem matrix_expr_value (x y z : ℝ) (h1 : det (matrixA x y z) = 0) (h2 : x + y + z = 0) :
  (x / (y + z)) + (y / (x + z)) + (z / (x + y)) = -3 :=
  sorry

end matrix_expr_value_l327_327843


namespace sum_of_primes_10003_l327_327548

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327548


namespace find_y_l327_327570

-- Definitions of the given conditions
def angle_ABC_is_straight_line := true  -- This is to ensure the angle is a straight line.
def angle_ABD_is_exterior_of_triangle_BCD := true -- This is to ensure ABD is an exterior angle.
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Theorem to prove y = 36 given the conditions
theorem find_y (A B C D : Type) (y : ℝ) 
    (h1 : angle_ABC_is_straight_line)
    (h2 : angle_ABD_is_exterior_of_triangle_BCD)
    (h3 : angle_ABD = 118)
    (h4 : angle_BCD = 82) : 
            y = 36 :=
  by
  sorry

end find_y_l327_327570


namespace gcf_of_lcm_eq_15_l327_327723

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_of_lcm_eq_15 : gcf (lcm 9 15) (lcm 10 21) = 15 := by
  sorry

end gcf_of_lcm_eq_15_l327_327723


namespace solve_inequality_l327_327027

theorem solve_inequality (x : ℕ) (hx : x > 0) : 12 * x + 5 < 10 * x + 15 ↔ x ∈ {1, 2, 3, 4} := by
  sorry

end solve_inequality_l327_327027


namespace zoe_jump_vs_cleo_step_l327_327809

theorem zoe_jump_vs_cleo_step 
  (d_gap : ℕ := 30) 
  (total_distance : ℕ := 5280) 
  (cleo_steps_per_gap : ℕ := 36) 
  (zoe_jumps_per_gap : ℕ := 9) :
  let cleo_total_steps := cleo_steps_per_gap * d_gap
  let zoe_total_jumps := zoe_jumps_per_gap * d_gap
  let cleo_step_length := total_distance / cleo_total_steps
  let zoe_jump_length := total_distance / zoe_total_jumps
  zoe_jump_length - cleo_step_length = 14.668 := 
by
  -- Proof skipped
  sorry

end zoe_jump_vs_cleo_step_l327_327809


namespace cos_value_f_range_l327_327426

variables {x A B C a b c : ℝ}

def vec_m := (vector ℝ) := ![√3 * real.sin (x / 4), 1]
def vec_n := (vector ℝ) := ![real.cos (x / 4), (real.cos (x / 4))^2]

-- Condition 1: vec_m ⊥ vec_n
lemma vector_perp : vec_m ⊛ vec_n = 0 :=
sorry

-- Proof of question 1: cosine calculation
theorem cos_value : (vector_perp) → 
    real.cos ((2 * real.pi / 3) - x) = -1 / 2 :=
sorry

-- Function f(x)
def f (x : ℝ) := (√3) * real.sin (x / 4) * real.cos (x / 4) + (real.cos (x / 4))^2

-- Condition 2: in triangle ABC, (2a - c)cos B = bcos C
variables (a b c : ℝ)
lemma triangle_ineq : (2 * a - c) * real.cos B = b * real.cos C :=
sorry

-- Proof of question 2: range of f(A)
theorem f_range (A : ℝ) (h : triangle_ineq) : 1 < f(A) ∧ f(A) < 3 / 2 :=
sorry

end cos_value_f_range_l327_327426


namespace remaining_water_in_bathtub_l327_327945

theorem remaining_water_in_bathtub : 
  ∀ (dripping_rate : ℕ) (evaporation_rate : ℕ) (duration_hr : ℕ) (dumped_out_liters : ℕ), 
    dripping_rate = 40 →
    evaporation_rate = 200 →
    duration_hr = 9 →
    dumped_out_liters = 12 →
    let total_dripped_in_ml := dripping_rate * 60 * duration_hr in
    let total_evaporated_in_ml := evaporation_rate * duration_hr in
    let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml in
    let dumped_out_in_ml := dumped_out_liters * 1000 in
    net_water_in_ml - dumped_out_in_ml = 7800 :=
by
  intros dripping_rate evaporation_rate duration_hr dumped_out_liters
  intros rate_eq evap_eq duration_eq dump_eq
  simp [rate_eq, evap_eq, duration_eq, dump_eq]
  let total_dripped_in_ml := 40 * 60 * 9
  let total_evaporated_in_ml := 200 * 9
  let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml
  let dumped_out_in_ml := 12 * 1000
  simp [net_water_in_ml, dumped_out_in_ml]
  sorry

end remaining_water_in_bathtub_l327_327945


namespace jordan_rectangle_width_l327_327744

theorem jordan_rectangle_width (W : ℝ) :
  (∀ (area_carol area_jordan : ℝ),
    area_carol = 12 * 15 →
    area_jordan = 9 * W →
    area_carol = area_jordan) →
  W = 20 :=
by
  intro h
  let area_carol := 12 * 15
  let area_jordan := 9 * W
  have hc : area_carol = 12 * 15 := rfl
  have hj : area_jordan = 9 * W := rfl
  specialize h area_carol area_jordan hc hj
  sorry

end jordan_rectangle_width_l327_327744


namespace range_of_a_l327_327096

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → deriv (f a) x < 0) ∧
  (∀ x, 6 < x → deriv (f a) x > 0) →
  5 ≤ a ∧ a ≤ 7 :=
sorry

end range_of_a_l327_327096


namespace parabolas_intersect_at_single_point_l327_327652

theorem parabolas_intersect_at_single_point (p q : ℝ) (h : -2 * p + q = 2023) :
  ∃ (x0 y0 : ℝ), (∀ p q : ℝ, y0 = x0^2 + p * x0 + q → -2 * p + q = 2023) ∧ x0 = -2 ∧ y0 = 2027 :=
by
  -- Proof to be filled in
  sorry

end parabolas_intersect_at_single_point_l327_327652


namespace trapezoid_angle_l327_327583

theorem trapezoid_angle
  (EFGH : Type)
  (EF GH : EFGH)
  (parallel : ∃ (EF GH : EFGH), EF ∥ GH)
  (angle_E_eq : ∃ (E H : ℝ), E = 3 * H)
  (angle_G_eq : ∃ (G F : ℝ), G = 2 * F)
  (angle_sum : ∃ (F G : ℝ), F + G = 180) :
  ∃ (F : ℝ), F = 60 :=
by
  sorry

end trapezoid_angle_l327_327583


namespace determine_a_l327_327918

theorem determine_a
  (h : ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a * x - 2) ≥ 0) : 
  a = 1 :=
sorry

end determine_a_l327_327918


namespace distinctPaintedCubeConfigCount_l327_327271

-- Define a painted cube with given face colors
structure PaintedCube where
  blue_face : ℤ
  yellow_faces : Finset ℤ
  red_faces : Finset ℤ
  -- Ensure logical conditions about faces
  face_count : blue_face ∉ yellow_faces ∧ blue_face ∉ red_faces ∧
               yellow_faces ∩ red_faces = ∅ ∧ yellow_faces.card = 2 ∧
               red_faces.card = 3

-- There are no orientation-invariant rotations that change the configuration
def equivPaintedCube (c1 c2 : PaintedCube) : Prop :=
  ∃ (r: ℤ), 
    -- rotate c1 by r to get c2
    true -- placeholder for rotation logic

-- The set of all possible distinct painted cubes under rotation constraints is defined
def possibleConfigurations : Finset PaintedCube :=
  sorry  -- construct this set considering rotations

-- The main proposition
theorem distinctPaintedCubeConfigCount : (possibleConfigurations.card = 4) :=
  sorry

end distinctPaintedCubeConfigCount_l327_327271


namespace min_area_of_triangle_ABC_l327_327605

-- Define points A, B, and C as vectors
def A : ℝ × ℝ × ℝ := (-1, 1, 2)
def B : ℝ × ℝ × ℝ := (1, 2, 3)
def C (t : ℝ) : ℝ × ℝ × ℝ := (t, t, 1)

-- Function to compute the norm of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Function to compute the cross product of two vectors
def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1)

-- Function to compute the area of triangle ABC given t
def area_triangle_ABC (t : ℝ) : ℝ :=
  1 / 2 * norm (cross_product (B.1 - A.1, B.2 - A.2, B.3 - A.3) (C t.1 - A.1, C t.2 - A.2, C t.3 - A.3))

-- Theorem statement: minimum area of triangle ABC is sqrt(29)/(2*sqrt(13))
theorem min_area_of_triangle_ABC : ∃ t : ℝ, area_triangle_ABC t = sqrt(29) / (2 * sqrt(13)) :=
sorry

end min_area_of_triangle_ABC_l327_327605


namespace john_days_off_l327_327141

def streams_per_week (earnings_per_week : ℕ) (rate_per_hour : ℕ) : ℕ := earnings_per_week / rate_per_hour

def streaming_sessions (hours_per_week : ℕ) (hours_per_session : ℕ) : ℕ := hours_per_week / hours_per_session

def days_off_per_week (total_days : ℕ) (streaming_days : ℕ) : ℕ := total_days - streaming_days

theorem john_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (total_days : ℕ) :
  hours_per_session = 4 → 
  hourly_rate = 10 → 
  weekly_earnings = 160 → 
  total_days = 7 → 
  days_off_per_week total_days (streaming_sessions (streams_per_week weekly_earnings hourly_rate) hours_per_session) = 3 := 
by
  intros
  sorry

end john_days_off_l327_327141


namespace inequality_proof_l327_327983

open Complex Real

-- Define the key ratio in the inequality
def max_norm_ratio (a b c d : ℂ) : ℝ :=
  (max (abs (a * c)) (max (abs (a * d + b * c)) (abs (b * d)))) /
  ((max (abs a) (abs b)) * (max (abs c) (abs d)))

-- Define the constant k as (\sqrt{5} - 1) / 2
def k : ℝ := (sqrt 5 - 1) / 2

-- The main theorem statement
theorem inequality_proof (a b c d : ℂ) (ha : a ≠ 0) (hc : c ≠ 0) :
  max_norm_ratio a b c d ≥ k := by
  sorry

end inequality_proof_l327_327983


namespace transformed_sin_graph_l327_327701

noncomputable def transform_sin_graph : ℝ → ℝ :=
  λ x, sin (1/2 * x - π/10)

theorem transformed_sin_graph (x : ℝ) :
  let f := λ x, sin x
  let g := λ x, f (x - π/10)
  let h := λ x, g (2 * x)
  transform_sin_graph x = h x :=
by
  simp [transform_sin_graph, h, g, f]
  sorry

end transformed_sin_graph_l327_327701


namespace sin_sum_to_product_l327_327332

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sin_sum_to_product_l327_327332


namespace max_value_of_f_l327_327204

noncomputable def f (x y : ℝ) : ℝ :=
  real.sqrt (real.cos (4 * x) + 7)
  + real.sqrt (real.cos (4 * y) + 7)
  + real.sqrt (real.cos (4 * x) + real.cos (4 * y) - 8 * real.sin x^2 * real.sin y^2 + 6)

theorem max_value_of_f :
  ∃ (C : ℝ), ∀ x y : ℝ, f x y ≤ C ∧ (∃ x y : ℝ, f x y = C) :=
begin
  use 6 * real.sqrt 2,
  sorry
end

end max_value_of_f_l327_327204


namespace find_angle_F_l327_327574

variable (EF GH : ℝ) -- Lengths of sides EF and GH
variable (angle_E angle_F angle_G angle_H : ℝ) -- Angles at vertices E, F, G, and H

-- Conditions given in the problem
axiom EF_parallel_GH : EF ∥ GH
axiom angle_E_eq_3_angle_H : angle_E = 3 * angle_H
axiom angle_G_eq_2_angle_F : angle_G = 2 * angle_F

-- Target statement to prove
theorem find_angle_F : angle_F = 60 := by
  -- Conditions setup:
  have angle_F_plus_angle_G := 180 - angle_G ; sorry
  -- Solve for angle_F
  have angle_F_eq_60 := 180 / 3; sorry
  sorry

end find_angle_F_l327_327574


namespace minimum_value_l327_327352

open Real

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2) + y^2 / (x - 2)) ≥ 12 :=
sorry

end minimum_value_l327_327352


namespace sin_sum_to_product_identity_l327_327328

theorem sin_sum_to_product_identity (x : ℝ) :
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by sorry

end sin_sum_to_product_identity_l327_327328


namespace parallelogram_slope_l327_327816

theorem parallelogram_slope (a b c d : ℚ) :
    a = 35 + c ∧ b = 125 - c ∧ 875 - 25 * c = 280 + 8 * c ∧ (a, 8) = (b, 25)
    → ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (∃ h : 8 * 33 * a + 595 = 2350, (m, n) = (25, 4)) :=
by
  sorry

end parallelogram_slope_l327_327816


namespace no_prime_sum_10003_l327_327505

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327505


namespace solution_hyperbola_line_l327_327896

noncomputable def hyperbola_eqn (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_M : ℝ × ℝ := (1, -1)

noncomputable def line_eqn (x y : ℝ) (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (y - my) = a * (x - mx)

noncomputable def intersects_hyperbola (line : ℝ × ℝ → Prop) (M A B : ℝ × ℝ) : Prop :=
  line M ∧ hyperbola_eqn A.1 A.2 ∧ hyperbola_eqn B.1 B.2

noncomputable def lambda1_lambda2 (M A N B : ℝ × ℝ) : Prop :=
  ∃ λ1 λ2 : ℝ, λ1 ≠ 0 ∧ λ2 ≠ 0 ∧
    (∃ AN_x AN_y, (N.1 - A.1, N.2 - A.2) = (AN_x, AN_y) ∧ (M.1 - A.1, M.2 - A.2) = λ1 * AN_x) ∧
    (∃ BN_x BN_y, (N.1 - B.1, N.2 - B.2) = (BN_x, BN_y) ∧ (M.1 - B.1, M.2 - B.2) = λ2 * BN_x)

noncomputable def range_value (λ1 λ2 : ℝ) : ℝ := λ1 / λ2 + λ2 / λ1

theorem solution_hyperbola_line (λ1 λ2 : ℝ) (M A N B : ℝ × ℝ)
  (hM : M = point_M) 
  (h1 : intersects_hyperbola (line_eqn M) M A B) 
  (h2 : lambda1_lambda2 M A N B) :
  ∃ r : ℝ, range_value λ1 λ2 = r ∧  r ∈ {val : ℝ | val = 4} :=
sorry

end solution_hyperbola_line_l327_327896


namespace total_students_l327_327743

theorem total_students (r l : ℕ) (hr : r = 13) (hl : l = 8) : (r + l - 1 = 20) :=
by 
  rw [hr, hl]
  admit 
  sorry

end total_students_l327_327743


namespace total_keys_needed_l327_327699

theorem total_keys_needed :
  let apartments := [12, 15, 20, 25]
  let apartments_keys_per_lock := 4
  let shared_facilities := 10
  let shared_facilities_keys_per_lock := 7
  let main_entrance_keys_per_lock := 12
  let individual_houses := 6
  let individual_houses_keys_per_lock := 6
  let commercial_buildings := 3
  let commercial_doors_per_building := 2
  let commercial_keys_per_lock := 5

  let total_apartment_keys := (List.sum apartments) * apartments_keys_per_lock
  let total_shared_facility_keys := shared_facilities * shared_facilities_keys_per_lock
  let total_main_entrance_keys := apartments.length * main_entrance_keys_per_lock
  let total_individual_house_keys := individual_houses * individual_houses_keys_per_lock
  let total_commercial_keys := commercial_buildings * commercial_doors_per_building * commercial_keys_per_lock

  let total_keys := total_apartment_keys + total_shared_facility_keys + total_main_entrance_keys + total_individual_house_keys + total_commercial_keys
  total_keys = 472 := by
  simp [total_apartment_keys, total_shared_facility_keys, total_main_entrance_keys, total_individual_house_keys, total_commercial_keys, apartments, apartments_keys_per_lock, shared_facilities_keys_per_lock, main_entrance_keys_per_lock, individual_houses_keys_per_lock, commercial_doors_per_building, commercial_keys_per_lock]
  rfl

end total_keys_needed_l327_327699


namespace krystiana_monthly_earnings_l327_327957

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l327_327957


namespace wolves_remaining_grassland_l327_327175

open Nat

-- Define the function P(n) as the number of primes less than n.
noncomputable def P (n : ℕ) : ℕ :=
  (List.filter Prime (List.range n)).length

-- Define the main function that returns the number of wolves remaining.
theorem wolves_remaining_grassland : 
  let W := {w ∈ Finset.range 2018 | true}
  let S := {1, 2, 3, 4, 5, 6, 7}
  (W.card - Finset.card (Finset.filter (λ w, ∃ s ∈ S, P w % 7 = s) W)) = 2016 :=
by 
  sorry

end wolves_remaining_grassland_l327_327175


namespace trapezoid_angle_l327_327582

theorem trapezoid_angle
  (EFGH : Type)
  (EF GH : EFGH)
  (parallel : ∃ (EF GH : EFGH), EF ∥ GH)
  (angle_E_eq : ∃ (E H : ℝ), E = 3 * H)
  (angle_G_eq : ∃ (G F : ℝ), G = 2 * F)
  (angle_sum : ∃ (F G : ℝ), F + G = 180) :
  ∃ (F : ℝ), F = 60 :=
by
  sorry

end trapezoid_angle_l327_327582


namespace number_of_triangles_with_positive_area_l327_327911

theorem number_of_triangles_with_positive_area :
  let points := finset.product (finset.range 1 7) (finset.range 1 7) in
  let all_triangles := finset.powerset_len 3 points in
  let collinear_points (p q r : (ℕ × ℕ)) : Prop :=
    (q.1 - p.1) * (r.2 - p.2) = (r.1 - p.1) * (q.2 - p.2) in
  let collinear_triangles := all_triangles.filter (λ t, ∃ p q r, p ∈ t ∧ q ∈ t ∧ r ∈ t ∧ collinear_points p q r) in
  let total_triangles := all_triangles.card in
  let total_collinear := collinear_triangles.card in
  total_triangles - total_collinear = 6778 :=
by sorry

end number_of_triangles_with_positive_area_l327_327911


namespace sin_sum_to_product_l327_327336

theorem sin_sum_to_product (x : ℝ) : 
  sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) :=
by 
  sorry

end sin_sum_to_product_l327_327336


namespace find_angle_F_l327_327575

variable (EF GH : ℝ) -- Lengths of sides EF and GH
variable (angle_E angle_F angle_G angle_H : ℝ) -- Angles at vertices E, F, G, and H

-- Conditions given in the problem
axiom EF_parallel_GH : EF ∥ GH
axiom angle_E_eq_3_angle_H : angle_E = 3 * angle_H
axiom angle_G_eq_2_angle_F : angle_G = 2 * angle_F

-- Target statement to prove
theorem find_angle_F : angle_F = 60 := by
  -- Conditions setup:
  have angle_F_plus_angle_G := 180 - angle_G ; sorry
  -- Solve for angle_F
  have angle_F_eq_60 := 180 / 3; sorry
  sorry

end find_angle_F_l327_327575


namespace general_term_and_diminishing_diff_Sn_range_of_t_for_bn_l327_327009

-- Geometric sequence conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (t : ℝ)

-- Define a geometric sequence with given conditions
def geometric_sequence (q : ℝ) : Prop :=
  a 1 = 1 ∧ S 3 = 7/4 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = q * a n

-- Diminishing difference sequence definition
def diminishing_difference_sequence (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (S n + S (n + 2)) / 2 < S (n + 1)

-- Define b_n sequence
def b_n (n : ℕ) (t : ℝ) :=
  (2 - n * a n) * t + a n

-- Main therem 1: prove the general term and diminishing difference property for S_n
theorem general_term_and_diminishing_diff_Sn (q : ℝ) (h : geometric_sequence a S q) :
  (∀ n > 0, a n = 1 / 2^(n - 1)) ∧ diminishing_difference_sequence S := 
sorry

-- Main theorem 2: prove the range of t for b_n
theorem range_of_t_for_bn (h : diminishing_difference_sequence (b_n t)) :
  t > 1 :=
sorry

end general_term_and_diminishing_diff_Sn_range_of_t_for_bn_l327_327009


namespace oil_bill_for_January_l327_327251

variable {F J : ℕ}

theorem oil_bill_for_January (h1 : 2 * F = 3 * J) (h2 : 3 * (F + 20) = 5 * J) : J = 120 := by
  sorry

end oil_bill_for_January_l327_327251


namespace range_of_a_l327_327622

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) : (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 := by
  intro h
  sorry

end range_of_a_l327_327622


namespace find_total_distance_l327_327245

-- Define the conditions
variables (d : ℝ) (t₁ t₂ t₃ : ℝ) (total_time : ℝ)

-- Assume the speeds and total time
def speed1 := 5 -- km/hr
def speed2 := 10 -- km/hr
def speed3 := 15 -- km/hr
def total_time := (11 : ℝ) / 60 -- hours

-- Define the times for each segment
def time1 := d / speed1
def time2 := d / speed2
def time3 := d / speed3

-- Given the sum of the times equals the total time
axiom sum_of_times : time1 + time2 + time3 = total_time

-- Statement to prove the total distance
theorem find_total_distance : 3 * d = 1.5 := by
  sorry

end find_total_distance_l327_327245


namespace five_digit_even_numbers_l327_327717

-- Definitions and conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

def num_digits : ℕ := 5

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Statement of the problem: 
theorem five_digit_even_numbers (h : ∀ d ∈ digits, 2 ≤ count d (digits.to_list.map fun x => x.to_nat) ≤ 3) : 
  ∃ n : ℕ, (n.to_digits ∈ digits) ∧ (is_even n) ∧ (∑ d in digits, count d n.to_digits = num_digits) ∧ (310 = count (λ k, (∀ (d ∈ digits), 2 ≤ count d k.to_digits ≤ 3) ∧ (is_even k)) (digits.to_list.map (λ x, x.to_nat))) :=
sorry

end five_digit_even_numbers_l327_327717


namespace complement_of_A_l327_327902

-- Define the universal set U
def U : set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def C_UA : set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove that C_UA = {1, 3, 6, 7}
theorem complement_of_A :
  C_UA = {1, 3, 6, 7} :=
sorry

end complement_of_A_l327_327902


namespace find_angle_F_l327_327581

-- Define the given conditions and the goal
variable (EF GH : ℝ) (angleE angleF angleG angleH : ℝ)
variable (h1 : EF ∥ GH) (h2 : angleE = 3 * angleH) (h3 : angleG = 2 * angleF) 

theorem find_angle_F (h_sum : angleF + angleG = 180) : angleF = 60 :=
by sorry

end find_angle_F_l327_327581


namespace kim_branch_marking_l327_327142

theorem kim_branch_marking (L : ℝ) (rem_frac : ℝ) (third_piece : ℝ) (F : ℝ) :
  L = 3 ∧ rem_frac = 0.6 ∧ third_piece = 1 ∧ L * rem_frac = 1.8 → F = 1 / 15 :=
by sorry

end kim_branch_marking_l327_327142


namespace sin_theta_value_l327_327371

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end sin_theta_value_l327_327371


namespace find_f_2017_l327_327860

noncomputable def f : ℕ → ℕ
| 1     := 2
| n     := if n % 2 = 0 then f (n - 1) + 2 else if n > 1 then f (n - 2) + 2 else 2

theorem find_f_2017 : f 2017 = 2018 := sorry

end find_f_2017_l327_327860


namespace apple_distribution_l327_327752

theorem apple_distribution (x : ℕ) (k : ℕ) :
  (k * x + (20 - k) * 3 = 109) → (x = 10 ∨ x = 52) :=
by
  intro h
  have basic := calc
    k * x + (20 - k) * 3 = 109 : h
  sorry

end apple_distribution_l327_327752


namespace probability_area_less_than_circumference_l327_327279

theorem probability_area_less_than_circumference :
  (∑ n in { (1, 1), (1, 2), (2, 1) }, 
    if n.fst + n.snd < 4 ∧ is_prime (n.fst + n.snd) then
      1 / 8^2 else 0) = 3 / 64 :=
  sorry

end probability_area_less_than_circumference_l327_327279


namespace complex_magnitude_problem_l327_327981

theorem complex_magnitude_problem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z - w| = 3) :
  |(2 / z + 1 / w)| = 7 / 8 :=
by sorry

end complex_magnitude_problem_l327_327981


namespace complex_point_eq_l327_327123

-- Definition of the complex number z
def z : ℂ := complex.I * (2 + complex.I)

-- The corresponding point in the complex plane
def point_of_z (z : ℂ) : ℝ × ℝ := (z.re, z.im)

-- The theorem to prove that the point for the complex number z is (-1, 2)
theorem complex_point_eq : point_of_z z = (-1, 2) :=
by
  sorry

end complex_point_eq_l327_327123


namespace problem_solution_l327_327215

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def inequality_solution_set : set ℝ :=
  { x | binomial_coefficient 12 (2 * ⌊x⌋₊ ) < binomial_coefficient 12 (2 * ⌊x⌋₊ - 3) }

theorem problem_solution :
  inequality_solution_set = {4, 4.5, 5, 5.5, 6} :=
by {
  sorry
}

end problem_solution_l327_327215


namespace angle_ABC_45_l327_327643

theorem angle_ABC_45 (ABC : Type) [nonempty ABC]
  (A B C K M N : ABC)
  (h1 : cyclic_quad A K M C)
  (h2 : cyclic_quad K B M N)
  (h3 : same_circumradius (circumradius A K M C) (circumradius K B M N))
  (h4 : N = (segment A M).meet (segment C K)) :
  ∠ B A C = 45 :=
sorry

end angle_ABC_45_l327_327643


namespace circle_properties_l327_327920

theorem circle_properties (m n : ℝ) (x y : ℝ) :
  (m^2 + n^2 = 4) ∧ (∀ x y : ℝ, (x^2 + y^2 = 4) → ((x - m)^2 + (y - n)^2 = 4) → (2*sqrt(3))) → 
  ((m^2 + n^2 = 4) ∧ (mx + ny - 2 = 0)) :=
by
  intros h
  sorry

end circle_properties_l327_327920


namespace range_of_a_l327_327055

def is_range_real (f : ℝ → ℝ) : Prop :=
  ∀ y, ∃ x, f x = y

noncomputable def piecewise_function (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 3 * a else log x / log 2

theorem range_of_a (a : ℝ) :
  is_range_real (piecewise_function a) ↔ -1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l327_327055


namespace find_a_and_b_l327_327053

-- Given conditions
def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_and_b (a b : ℝ) :
  (∀ x, ∀ y, tangent_line x y → y = b ∧ x = 0) ∧
  (∀ x, ∀ y, y = curve x a b) →
  a = 1 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l327_327053


namespace smallest_n_divisible_11_remainder1_l327_327735

theorem smallest_n_divisible_11_remainder1 :
  ∃ n, (∀ m ∈ {2, 3, 4, 5, 6, 7, 8}, n % m = 1) ∧ (n % 11 = 0) ∧ (n = 6721) :=
by {
  sorry
}

end smallest_n_divisible_11_remainder1_l327_327735


namespace rectangle_diagonal_identity_l327_327045

theorem rectangle_diagonal_identity
  (AB AC AD AE AF : ℝ)
  (hRectABCD : AB ∥ AD ∧ AD ∥ BC ∧ AB ⊥ AD)
  (hCE_perp_AB : ∀ E, C = E → E ∈ AB → CE ⊥ AB)
  (hCF_perp_AD : ∀ F, C = F → F ∈ AD → CF ⊥ AD) :
  AB * AE + AD * AF = AC ^ 2 := by
  sorry

end rectangle_diagonal_identity_l327_327045


namespace hexagon_coloring_l327_327815

theorem hexagon_coloring:
  ∃ n : ℕ, (n = 2) ∧
    ∀ (A B : ℕ → ℕ → Prop),
      (∀ {i j : ℕ}, adjacent (i, j) → ¬(color (i, j) = color (i + 1, j + 1))) ∧
      (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 4 → 1 ≤ j ∧ j ≤ 3 → 
        (∀ (color : ℕ → ℕ → Color), 
          ∃ C1 C2 : Color, 
            C1 ≠ Red ∧ C2 ≠ Red ∧ colors_adjacency (C1, C2) ∧
              (color(1, 1) = Red ∧ color_pattern(color, C1, C2, i, j))) :=
sorry

end hexagon_coloring_l327_327815


namespace complex_problem_l327_327854

open Complex

noncomputable def z : ℂ := (1 + I) / Real.sqrt 2

theorem complex_problem :
  1 + z^50 + z^100 = I := 
by
  -- Subproofs or transformations will be here.
  sorry

end complex_problem_l327_327854


namespace no_prime_sum_10003_l327_327543

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327543


namespace board_has_valid_product_l327_327641

def is_valid_product (XYZ YXZ p : ℕ) : Prop :=
  let digits := [X, Y, Z] in
  let prod_digits := digits ++ [p.digit 0] in     -- middle digits condition
  (1 < X ∧ X < 10 ∧ 1 < Y ∧ Y < 10 ∧ 1 < Z ∧ Z < 10) ∧  -- non-zero and decimal digits
  (∀ d ∈ digits, ∀ e ∈ digits, d ≠ e) ∧   -- distinct digits
  p.digit (p.num_digits-1) = p.digit 0 ∧  -- outer digits are equal
  (∀ d ∈ digits, p.digit (p.num_digits-1) ≠ d) ∧ -- outer digits different from middle ones
  ⟨X, Y, Z⟩ ∈ {(2, 6, 9), (3, 5, 9)} /- numbers pairs -/

theorem board_has_valid_product :
  ∃ XYZ YXZ p, is_valid_product XYZ YXZ p ∧ (p = 169201 ∨ p = 193501) := 
sorry

end board_has_valid_product_l327_327641


namespace go_to_yolka_together_l327_327798

noncomputable def anya_will_not_wait : Prop := true
noncomputable def boris_wait_time : ℕ := 10 -- in minutes
noncomputable def vasya_wait_time : ℕ := 15 -- in minutes
noncomputable def meeting_time_window : ℕ := 60 -- total available time in minutes

noncomputable def probability_all_go_together : ℝ :=
  (1 / 3) * (3500 / 3600)

theorem go_to_yolka_together :
  anya_will_not_wait ∧
  boris_wait_time = 10 ∧
  vasya_wait_time = 15 ∧
  meeting_time_window = 60 →
  probability_all_go_together = 0.324 :=
by
  intros
  sorry

end go_to_yolka_together_l327_327798


namespace tendsto_sum_areas_l327_327765

noncomputable def sum_areas_limit (s : ℝ) : ℝ :=
  -- Definitions here should be consistent with the conditions but do not rely on steps
  let r1 := s * Real.sqrt 3 / 6
  let a1 := π * (r1 ^ 2)
  let s2 := 2 * r1 / Real.sqrt 3
  let a2 := (Real.sqrt 3 / 4) * (s2 ^ 2)
  let circles_factor := (Real.sqrt 3 / 3) ^ 2
  let triangles_factor := (2 / 3) ^ 2
  let A := a1 / (1 - triangles_factor) -- Infinite sum of areas for circles and tris
  let S := A
  S

theorem tendsto_sum_areas (s : ℝ) :
  (∀ n, S_n s n) → (∃ l, tendsto (λ n, S_n s n) at_top (𝓝 l)) → l = (3 * π * s ^ 2 / 4) :=
  sorry

end tendsto_sum_areas_l327_327765


namespace chastity_initial_amount_l327_327308

def lollipop_price : ℝ := 1.50
def number_of_lollipops : ℕ := 4
def gummies_price : ℝ := 2
def number_of_gummies : ℕ := 2
def amount_left : ℝ := 5
def total_cost_of_lollipops := number_of_lollipops * lollipop_price
def total_cost_of_gummies := number_of_gummies * gummies_price
def total_cost_of_candies := total_cost_of_lollipops + total_cost_of_gummies

theorem chastity_initial_amount :
  let initial_amount := total_cost_of_candies + amount_left in
  initial_amount = 15 := 
by
  sorry

end chastity_initial_amount_l327_327308


namespace min_value_f_eq_4_ineq_ax2_bx2_l327_327414

noncomputable def f (x : ℝ) : ℝ := abs (x - 5) + abs (x - 1)

theorem min_value_f_eq_4 : ∀ x : ℝ, f x ≥ 4 := 
by 
  intro x 
  have h1 := abs_nonneg (x - 5)
  have h2 := abs_nonneg (x - 1)
  have h3 : f x = abs (x - 5) + abs (x - 1) := rfl
  have h4 : f x ≥ abs (x - 5 + -(x - 1)) := abs_add (x - 5) (-(x - 1))
  rw [neg_sub] at h4
  rw [sub_sub_sub_cancel_right, abs_of_nonneg] at h4
  exact h4

theorem ineq_ax2_bx2 : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a + 1 / b = real.sqrt 6) → (1 / a^2 + 2 / b^2 ≥ 4) :=
by
  intros a b ha hb h
  have h1 : (1 / a^2 + 2 / b^2) * (1^2 + (real.sqrt 2)^2) ≥ (1 / a * 1 + real.sqrt 2 / b * 1 / real.sqrt 2)^2 :=
    youngs_ineq _ _ (1 : ℝ) (real.sqrt 2)
  rw [mul_one, sq_sqrt, mul_div_cancel' _ (sqrt_ne_zero_of_pos _)] at h1
  rw [add_sq, ←h] at h1
  linarith
  exact add_pos_of_pos_of_pos ha hb

end min_value_f_eq_4_ineq_ax2_bx2_l327_327414


namespace unique_sum_of_two_primes_l327_327556

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327556


namespace point_in_second_quadrant_l327_327566

theorem point_in_second_quadrant (x y : ℤ) (hx : x = -3) (hy : y = 4) :
  (x < 0 ∧ y > 0) → (x = -3 ∧ y = 4) := 
by
  intro h
  split
  sorry -- Proof is omitted as per instructions

end point_in_second_quadrant_l327_327566


namespace average_age_decrease_l327_327673

theorem average_age_decrease (A : ℝ) :
  let initial_total_age := 10 * A in
  let decrease_in_age := 42 - 12 in
  let new_total_age := initial_total_age - decrease_in_age in
  let new_average_age := new_total_age / 10 in
  new_average_age = A - 3 := 
by
  sorry

end average_age_decrease_l327_327673


namespace cis_sum_l327_327205

noncomputable def sum_cis_θ (n : ℕ) := Complex.exp (Complex.I * Real.pi * (75 + 8 * n) / 180)

theorem cis_sum :
  let S := ∑ k in Finset.range (9), sum_cis_θ k
  ∃ (r : ℝ), r > 0 ∧ ∃ θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ 
  S = Complex.exp (Complex.I * θ * Real.pi / 180) ↔ θ = 111 :=
by {
  sorry
}

end cis_sum_l327_327205


namespace reservoir_water_last_50_days_l327_327697

noncomputable def num_days (x y z : ℝ) : ℝ :=
  z / (y - 1.2 * x)

theorem reservoir_water_last_50_days (x y z : ℝ) 
  (h1 : 40 * (y - x) = z) 
  (h2 : 40 * (1.1 * y - 1.2 * x) = z) : 
  num_days x y z = 50 :=
begin
  sorry
end

end reservoir_water_last_50_days_l327_327697


namespace min_coins_chessboard_l327_327642

theorem min_coins_chessboard (n : ℕ) (h : n ≥ 4) : 
    (∃ (m : ℕ), 
    (∀ i : fin n, ∃ c : fin n, has_coin (i, c)) ∧
    (∀ j : fin n, ∃ r : fin n, has_coin (r, j)) ∧
    (∀ k : ℤ, k ≠ 0 → ∃ r c : fin n, r - c = k ∧ has_coin (r, c)) ∧ 
    (∀ k : ℤ, k ≠ 0 → ∃ r c : fin n, r + c = k ∧ has_coin (r, c)) ∧
    ((∃ m : ℕ, m = if n % 2 = 0 then 2 * n - 2 else 2 * n - 3))) :=
        sorry

structure has_coin (pos : fin n × fin n) : Prop

end min_coins_chessboard_l327_327642


namespace smallest_product_in_set_l327_327317

theorem smallest_product_in_set (s : Set ℤ) (h : s = {-10, -4, -2, 0, 5}) :
  ∃ x y ∈ s, x ≠ y ∧ x * y = -50 :=
by
  use -10, 5
  split
  all_goals { sorry }

end smallest_product_in_set_l327_327317


namespace bags_sold_in_first_week_l327_327767

def total_bags_sold : ℕ := 100
def bags_sold_week1 (X : ℕ) : ℕ := X
def bags_sold_week2 (X : ℕ) : ℕ := 3 * X
def bags_sold_week3_4 : ℕ := 40

theorem bags_sold_in_first_week (X : ℕ) (h : total_bags_sold = bags_sold_week1 X + bags_sold_week2 X + bags_sold_week3_4) : X = 15 :=
by
  sorry

end bags_sold_in_first_week_l327_327767


namespace max_min_value_of_fraction_l327_327862

theorem max_min_value_of_fraction
  (x y : ℝ)
  (h : x^2 + (y - 1)^2 = 1) :
  ∃ k, k = \frac{y-1}{x-2} ∧ (k = \frac{sqrt 3}{3} ∨ k = -\frac{sqrt 3}{3}) := 
sorry

end max_min_value_of_fraction_l327_327862


namespace complex_conjugate_of_z_l327_327621

theorem complex_conjugate_of_z (z : ℂ) (h : z * (1 - I)^2 = 4 * I) : conj z = -2 :=
sorry

end complex_conjugate_of_z_l327_327621


namespace ralph_did_not_hit_89_balls_l327_327184

-- Definitions based on the conditions
def total_balls : ℕ := 175
def slow_balls : ℕ := 100
def medium_balls : ℕ := 50
def fast_balls : ℕ := 25

def slow_hit_ratio : ℚ := 3 / 5
def medium_hit_ratio : ℚ := 2 / 5
def fast_hit_ratio : ℚ := 1 / 4

-- Number of balls hit at each speed setting
def slow_balls_hit := (slow_hit_ratio * slow_balls).to_nat
def medium_balls_hit := (medium_hit_ratio * medium_balls).to_nat
def fast_balls_hit := (fast_hit_ratio * fast_balls).to_nat

-- Total balls hit
def total_balls_hit := slow_balls_hit + medium_balls_hit + fast_balls_hit

-- Number of balls not hit
def balls_not_hit := total_balls - total_balls_hit

-- The statement to prove
theorem ralph_did_not_hit_89_balls : balls_not_hit = 89 :=
by
  sorry

end ralph_did_not_hit_89_balls_l327_327184


namespace exists_mn_gt_mn_l327_327988

variables (a : ℕ × ℕ → ℕ)

axiom doubly_infinite_array_of_positive_integers :
  ∀ i j, a (i, j) > 0

axiom each_positive_integer_appears_exactly_eight_times :
  ∀ n, (∃! p : ℕ × ℕ, a p = n) ∧ (card (image (λ p, a p) univ) = 8)

theorem exists_mn_gt_mn :
  ∃ m n : ℕ, a (m, n) > m * n :=
sorry

end exists_mn_gt_mn_l327_327988


namespace carlos_time_l327_327808

theorem carlos_time (Diego_half_block_time : ℝ) (average_time_seconds : ℝ) (block_conversion : ℝ)
  (avg_seconds_to_minutes : average_time_seconds / block_conversion = 4) (Diego_full_block_time : Diego_half_block_time * 2 = 5):
  ∀ C : ℝ, (C + Diego_full_block_time) / 2 = 4 → C = 3 :=
by
  intros C h
  calc
    C + 5 = 8 := by linarith
    C = 3 := by linarith

end carlos_time_l327_327808


namespace domain_of_k_l327_327732

noncomputable def k (x : ℝ) := 1 / (x + 3) + 1 / (x^2 + 3) + 1 / (x^3 + 3)

theorem domain_of_k :
  {x : ℝ | x ≠ -3 ∧ x ≠ -real.cbrt 3 } = { x : ℝ | x ∈ (-∞, -3) ∪ (-3, -real.cbrt 3) ∪ (-real.cbrt 3, ∞) } :=
sorry

end domain_of_k_l327_327732


namespace point_coincidence_l327_327838

theorem point_coincidence : 
  (fold : ℝ × ℝ → ℝ × ℝ) 
  (h : fold (2, 0) = (2, 4)) : 
  fold (-4, 1) = (-4, 3) :=
sorry

end point_coincidence_l327_327838


namespace adoption_days_l327_327256

def initial_puppies : ℕ := 2
def additional_puppies : ℕ := 34
def adoption_rate : ℕ := 4
def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days : total_puppies / adoption_rate = 9 := 
by {
  have h1 : total_puppies = 36, by sorry, -- 2 + 34
  have h2 : 36 / adoption_rate = 9, by sorry, -- 36 / 4
  rw h1,
  exact h2,
}

end adoption_days_l327_327256


namespace triangle_area_PQR_l327_327275

noncomputable def lineEquation (slope : ℝ) (point : ℝ × ℝ) : ℝ → ℝ :=
  λ x => slope * (x - point.1) + point.2

noncomputable def xIntercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  -point.2 / slope + point.1

noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((Q.1 - R.1) * (P.2))

theorem triangle_area_PQR :
  let P := (2 : ℝ, 8 : ℝ)
  let Q := (xIntercept 3 P, 0)
  let R := (xIntercept (-1) P, 0)
  Q = ((-2 / 3 : ℝ), 0) →
  R = ((10 : ℝ), 0) →
  triangleArea P Q R = (128 / 3 : ℝ) := 
by
  intros P Q R H_Q H_R
  rw [H_Q, H_R]
  sorry

end triangle_area_PQR_l327_327275


namespace intervals_of_monotonic_increasing_min_value_of_f_in_interval_l327_327060

-- Declare the function f
def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

-- Monotonic intervals
theorem intervals_of_monotonic_increasing :
  (∀ x,  f ' x ≥ 0 → x ≤ -2 ∨ x ≥ 2)

-- Minimum value in the interval [0, 4]
theorem min_value_of_f_in_interval :
  (min_in (λ x, 0 ≤ x ∧ x ≤ 4) f = - (4 / 3))

end intervals_of_monotonic_increasing_min_value_of_f_in_interval_l327_327060


namespace find_first_train_length_l327_327714

namespace TrainProblem

-- Define conditions
def speed_first_train_kmph := 42
def speed_second_train_kmph := 48
def length_second_train_m := 163
def time_clear_s := 12
def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph

-- Convert kmph to m/s
def kmph_to_mps(kmph : ℕ) : ℕ := kmph * 5 / 18
def relative_speed_mps := kmph_to_mps relative_speed_kmph

-- Calculate total distance covered by the trains in meters
def total_distance_m := relative_speed_mps * time_clear_s

-- Define the length of the first train to be proved
def length_first_train_m := 137

-- Theorem statement
theorem find_first_train_length :
  total_distance_m = length_first_train_m + length_second_train_m :=
sorry

end TrainProblem

end find_first_train_length_l327_327714


namespace number_of_ways_sum_of_primes_l327_327499

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327499


namespace sum_of_primes_10003_l327_327546

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327546


namespace prove_polar_equation_and_distance_l327_327572

-- Define the parametric equations for Curve C1 and Curve C2
def curve_C1_x (t : ℝ) : ℝ := 1 + 2 * t
def curve_C1_y (t : ℝ) : ℝ := 2 - 2 * t

def curve_C2_x (θ : ℝ) : ℝ := 2 * Real.cos θ + 2
def curve_C2_y (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the intervals and conditions
def C1_param_interval : Set ℝ := Set.univ
def C2_param_interval : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Given conditions and questions
theorem prove_polar_equation_and_distance:
  ∀ θ ∈ Set.Icc 0 (2 * Real.pi),
  curve_C2_x θ = 2 * Real.cos θ + 2 →
  curve_C2_y θ = 2 * Real.sin θ →
  ∃ ρ : ℝ, ρ = 4 * Real.cos θ ∧
  ∀ t : ℝ,
  curve_C1_x t = 1 + 2 * t →
  curve_C1_y t = 2 - 2 * t →
  ∃ A B : ℝ × ℝ, 
  (A = ⟨curve_C1_x t, curve_C1_y t⟩) ∧ 
  (B = ⟨curve_C2_x θ, curve_C2_y θ⟩)  ∧
  (|A.1 - B.1|^2 + |A.2 - B.2|^2)^0.5 = Real.sqrt 14 :=
sorry

end prove_polar_equation_and_distance_l327_327572


namespace slices_per_person_l327_327711

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l327_327711


namespace no_two_digit_numbers_form_perfect_cube_sum_l327_327081

theorem no_two_digit_numbers_form_perfect_cube_sum :
  ∀ (N : ℕ), (10 ≤ N ∧ N < 100) →
  ∀ (t u : ℕ), (N = 10 * t + u) →
  let reversed_N := 10 * u + t in
  let sum := N + reversed_N in
  (∃ k : ℕ, sum = k^3) → false :=
by
  sorry

end no_two_digit_numbers_form_perfect_cube_sum_l327_327081


namespace geometric_sequence_sum_l327_327162

theorem geometric_sequence_sum (a : ℕ → ℤ)
  (h1 : a 0 = 1)
  (h_q : ∀ n, a (n + 1) = a n * -2) :
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end geometric_sequence_sum_l327_327162


namespace grid_intersect_diff_colors_l327_327103

variable {α : Type} [DecidableEq α]

-- Definition of grid dimensions and colors
def grid_size := 100
def colors : Fin 4 -- There are 4 colors

-- Conditions
variable (coloring : Fin 100 → Fin 100 → Fin 4) -- Function defining the color of each cell in the grid
variable (row_cond : ∀ (r : Fin 100), ∀ (c : Fin 4), #{i | coloring r i = c}.card = 25) -- Each row has 25 of each color
variable (col_cond : ∀ (c : Fin 100), ∀ (r : Fin 4), #{i | coloring i c = r}.card = 25) -- Each column has 25 of each color

-- Theorem statement
theorem grid_intersect_diff_colors (coloring : Fin 100 → Fin 100 → Fin 4) 
  (row_cond : ∀ (r : Fin 100), ∀ (c : Fin 4), #{i | coloring r i = c}.card = 25)
  (col_cond : ∀ (c : Fin 100), ∀ (r : Fin 4), #{i | coloring i c = r}.card = 25) :
  ∃ (r₁ r₂ c₁ c₂ : Fin 100), r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧ 
    coloring r₁ c₁ ≠ coloring r₁ c₂ ∧
    coloring r₁ c₁ ≠ coloring r₂ c₁ ∧
    coloring r₁ c₂ ≠ coloring r₂ c₂ ∧
    coloring r₂ c₁ ≠ coloring r₂ c₂ := 
sorry

end grid_intersect_diff_colors_l327_327103


namespace john_profit_l327_327953

theorem john_profit (cost_per_bag selling_price : ℕ) (number_of_bags : ℕ) (profit_per_bag total_profit : ℕ) :
  cost_per_bag = 4 →
  selling_price = 8 →
  number_of_bags = 30 →
  profit_per_bag = selling_price - cost_per_bag →
  total_profit = number_of_bags * profit_per_bag →
  total_profit = 120 :=
by
  intro h_cost h_sell h_num_bags h_profit_per_bag h_total_profit
  rw [h_profit_per_bag, h_cost, h_sell] at h_profit_per_bag
  rw [h_total_profit, h_num_bags, h_profit_per_bag]
  norm_num

end john_profit_l327_327953


namespace angle_HIO_expression_l327_327567

open Triangle

variables {A B C H I O M D : Point}
variables {angleB angleC angleDAM angleADM : Real}
variables {b c : Real}

-- Assume the conditions of the problem
axioms
  (hB_greater_C : angleB > angleC)
  (triangle_acute : IsAcuteAngledTriangle A B C)
  (orthocenter : IsOrthocenter H A B C)
  (incenter : IsIncenter I A B C)
  (circumcenter : IsCircumcenter O A B C)
  (midpoint_AI : IsMidpoint M A I)
  (perp_AD_BC : IsPerpendicular AD (BC))
  (angle_condition : angle DAM = 2 * angle ADM)
  (B_value : angleB = b)
  (C_value : angleC = c)

-- The statement to be proven
theorem angle_HIO_expression :
  ∃ (angleHIO : Real), angleHIO = 240 - b :=
sorry

end angle_HIO_expression_l327_327567


namespace sum_of_primes_10003_l327_327550

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l327_327550


namespace algebraic_expression_inequality_sqrt_inequality_l327_327259

-- Problem 1: Compare the size of two algebraic expressions
theorem algebraic_expression_inequality (x y : ℝ) : 
  x^2 + y^2 + 1 > 2 * (x + y - 1) :=
sorry

-- Problem 2: Given a > b > 0, c > d > 0, prove \sqrt{a/d} > \sqrt{b/c}
theorem sqrt_inequality (a b c d : ℝ) (hab1 : a > b) (hab2 : b > 0) 
  (hcd1 : c > d) (hcd2 : d > 0) :
    sqrt (a / d) > sqrt (b / c) :=
sorry

end algebraic_expression_inequality_sqrt_inequality_l327_327259


namespace sum_of_distances_to_focus_l327_327897

noncomputable def parabola : set (ℝ × ℝ) := { p | p.1 ^ 2 = 12 * p.2 }

def focus : ℝ × ℝ := (0, -3)

def point_P : ℝ × ℝ := (2, 1)

theorem sum_of_distances_to_focus (A B : ℝ × ℝ) (hA : A ∈ parabola) (hB : B ∈ parabola)
  (h_line : ∃ m b : ℝ, ∀ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + b } → 
    (x, y) = A ∨ (x, y) = B ∨ (x, y) = point_P)
  (h_midpoint : (A.1 + B.1) / 2 = point_P.1 ∧ (A.2 + B.2) / 2 = point_P.2) : 
  dist A focus + dist B focus = 8 := 
by sorry

end sum_of_distances_to_focus_l327_327897


namespace find_amount_after_2_years_l327_327348

noncomputable def compoundInterest (P : ℝ) (R1 : ℝ) (R2 : ℝ) : ℝ :=
let A1 := P + (P * R1 / 100)
in A1 + (A1 * R2 / 100)

theorem find_amount_after_2_years :
  compoundInterest 6552 4 5 = 7154.784 :=
by 
  let A1 := 6552 + (6552 * 4 / 100)
  have hA1 : A1 = 6814.08 := by norm_num
  rw hA1
  show A1 + (A1 * 5 / 100) = 7154.784
  have hA2 : 6814.08 + (6814.08 * 5 / 100) = 7154.784 := by norm_num
  exact hA2


end find_amount_after_2_years_l327_327348


namespace domain_of_f_l327_327876

def f (x : ℝ) : ℝ := 2 / Real.logb (1 / 2) (2 * x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x + 1 > 0 ∧ Real.logb (1 / 2) (2 * x + 1) ≠ 0} =
  {x : ℝ | x > -1/2 ∧ x ≠ 0} :=
by
  ext x
  simp only [set.mem_set_of_eq]
  split
  { intro h
    cases h with h1 h2
    split
    { linarith only [h1] }
    { rw [ne, ←Real.logb_eq_zero_iff] at h2
      simp at h2
      exact h2 }
  }
  { intro h
    cases h with h1 h2
    split
    { linarith only [h1] }
    { intro hlog
      rw Real.logb_eq_zero_iff at hlog
      simp at hlog
      contradiction }
  }

lemma domain_of_f_correct :
  {x : ℝ | x > -1/2 ∧ x ≠ 0} = set.Ioo (-1/2) 0 ∪ set.Ioi 0 :=
by
  ext x
  simp only [set.mem_set_of_eq, set.mem_Ioo, set.mem_Ioi, set.mem_union_eq]
  split
  { rintro ⟨hl,h⟩
    cases (lt_trichotomy x 0) with hlt heqgt
    { left
      exact ⟨hl,hlt⟩ }
    { cases heqgt 
      { exfalso
        exact h heqgt }
      { right
        exact heqgt }
    }
  }
  { intro h
    cases h
    { cases h with h₁ h₂
      exact ⟨h₁, h₂.2⟩ }
    { exact ⟨lt_of_le_of_ne (le_of_lt h) h.symm⟩ }
  }

example : {x : ℝ | 2 * x + 1 > 0 ∧ Real.logb (1 / 2) (2 * x + 1) ≠ 0} =
    set.Ioo (-1/2) 0 ∪ set.Ioi 0 :=
by
  rw [domain_of_f, domain_of_f_correct]
  sorry

end domain_of_f_l327_327876


namespace only_D_is_odd_l327_327239

-- Define the functions
def fA (x : ℝ) := if x ≥ 0 then real.sqrt x else 0
def fB (x : ℝ) := abs (real.sin x)
def fC (x : ℝ) := real.cos x
def fD (x : ℝ) := real.exp x - real.exp (-x)

-- Definitions needed for the statement
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_neither_odd_nor_even (f : ℝ → ℝ) := ¬(is_odd_function f) ∧ ¬(is_even_function f)

-- The statement
theorem only_D_is_odd :
  is_neither_odd_nor_even fA ∧
  is_even_function fB ∧
  is_even_function fC ∧
  is_odd_function fD :=
by sorry

end only_D_is_odd_l327_327239


namespace solveForN_l327_327438

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l327_327438


namespace max_value_expression_l327_327359

noncomputable def expression (y : ℝ) : ℝ := 
  y^6 / (y^12 + 4 * y^9 - 6 * y^6 + 16 * y^3 + 64)

theorem max_value_expression : 
  ∃ y : ℝ, ∀ z : ℝ, expression z ≤ expression (real.cbrt 4) := 
by
  sorry

end max_value_expression_l327_327359


namespace simplify_product_l327_327662

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l327_327662


namespace f_not_factorable_l327_327156

noncomputable def f (n : ℕ) (x : ℕ) : ℕ := x^n + 5 * x^(n - 1) + 3

theorem f_not_factorable (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : ℕ → ℕ, (∀ a b : ℕ, a ≠ 0 ∧ b ≠ 0 → g a * h b = f n a * f n b) ∧ 
    (∀ a b : ℕ, (g a = 0 ∧ h b = 0) → (a = 0 ∧ b = 0)) ∧ 
    (∃ pg qh : ℕ, pg ≥ 1 ∧ qh ≥ 1 ∧ g 1 = 1 ∧ h 1 = 1 ∧ (pg + qh = n)) := 
sorry

end f_not_factorable_l327_327156


namespace problem1_solutions_problem2_solutions_l327_327191

-- Problem 1: Solve x² - 7x + 6 = 0

theorem problem1_solutions (x : ℝ) : 
  x^2 - 7 * x + 6 = 0 ↔ (x = 1 ∨ x = 6) := by
  sorry

-- Problem 2: Solve (2x + 3)² = (x - 3)² 

theorem problem2_solutions (x : ℝ) : 
  (2 * x + 3)^2 = (x - 3)^2 ↔ (x = 0 ∨ x = -6) := by
  sorry

end problem1_solutions_problem2_solutions_l327_327191


namespace find_a1_plus_a2_l327_327849

theorem find_a1_plus_a2 (x : ℝ) (a0 a1 a2 a3 : ℝ) 
  (h : (1 - 2/x)^3 = a0 + a1 * (1/x) + a2 * (1/x)^2 + a3 * (1/x)^3) : 
  a1 + a2 = 6 :=
by
  sorry

end find_a1_plus_a2_l327_327849


namespace prime_sum_10003_l327_327527

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l327_327527


namespace tv_weight_calculations_l327_327804

theorem tv_weight_calculations
    (w1 h1 r1 : ℕ) -- Represents Bill's TV dimensions and weight ratio
    (w2 h2 r2 : ℕ) -- Represents Bob's TV dimensions and weight ratio
    (w3 h3 r3 : ℕ) -- Represents Steve's TV dimensions and weight ratio
    (ounce_to_pound: ℕ) -- Represents the conversion factor from ounces to pounds
    (bill_tv_weight bob_tv_weight steve_tv_weight : ℕ) -- Computed weights in pounds
    (weight_diff: ℕ):
  (w1 * h1 * r1) / ounce_to_pound = bill_tv_weight → -- Bill's TV weight calculation
  (w2 * h2 * r2) / ounce_to_pound = bob_tv_weight → -- Bob's TV weight calculation
  (w3 * h3 * r3) / ounce_to_pound = steve_tv_weight → -- Steve's TV weight calculation
  steve_tv_weight > (bill_tv_weight + bob_tv_weight) → -- Steve's TV is the heaviest
  steve_tv_weight - (bill_tv_weight + bob_tv_weight) = weight_diff → -- weight difference calculation
  True := sorry

end tv_weight_calculations_l327_327804


namespace janet_freelance_income_difference_l327_327138

variable (hours_per_week_current : ℕ := 40)
variable (rate_per_hour_current : ℕ := 30)
variable (weeks_per_month : ℕ := 4)

variable (proj_hours_week_1 : ℕ := 30)
variable (proj_hours_week_2 : ℕ := 35)
variable (proj_hours_week_3 : ℕ := 40)
variable (proj_hours_week_4 : ℕ := 50)

variable (rate_week_1 : ℕ := 45)
variable (rate_week_2 : ℕ := 40)
variable (rate_week_3 : ℕ := 35)
variable (rate_week_4 : ℕ := 38)

variable (fica_per_week : ℕ := 25)
variable (health_per_month : ℕ := 400)
variable (rent_per_month : ℕ := 750)
variable (phone_internet_per_month : ℕ := 150)

theorem janet_freelance_income_difference :
  let current_job_income_per_week := hours_per_week_current * rate_per_hour_current,
      current_job_income_per_month := current_job_income_per_week * weeks_per_month,
      income_week_1 := proj_hours_week_1 * rate_week_1,
      income_week_2 := proj_hours_week_2 * rate_week_2,
      income_week_3 := proj_hours_week_3 * rate_week_3,
      income_week_4 := proj_hours_week_4 * rate_week_4,
      total_freelance_income := income_week_1 + income_week_2 + income_week_3 + income_week_4,
      total_extra_expenses := fica_per_week * weeks_per_month + health_per_month + rent_per_month + phone_internet_per_month,
      net_freelance_income := total_freelance_income - total_extra_expenses
  in net_freelance_income - current_job_income_per_month = -150 :=
by
  sorry

end janet_freelance_income_difference_l327_327138


namespace number_of_ways_sum_of_primes_l327_327498

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327498


namespace last_digits_11_power_l327_327178

theorem last_digits_11_power (n : ℕ) (h : n ≥ 1) : 
  let num := 11 ^ (10 ^ n)
  let str_num := Nat.toDigits 10 num
  let last_n_plus_2_digits := List.take (n + 2) (str_num.reverse)
  last_n_plus_2_digits =  6 :: List.replicate n 0 ++ [1] :=
sorry

end last_digits_11_power_l327_327178


namespace find_8th_term_in_sequence_l327_327212

def sequence (n : ℕ) : ℚ :=
  (-1)^(n+1) * n / 2^n

theorem find_8th_term_in_sequence :
  sequence 8 = -1 / 32 :=
by
  sorry

end find_8th_term_in_sequence_l327_327212


namespace rectangle_diagonal_identity_l327_327046

theorem rectangle_diagonal_identity
  (AB AC AD AE AF : ℝ)
  (hRectABCD : AB ∥ AD ∧ AD ∥ BC ∧ AB ⊥ AD)
  (hCE_perp_AB : ∀ E, C = E → E ∈ AB → CE ⊥ AB)
  (hCF_perp_AD : ∀ F, C = F → F ∈ AD → CF ⊥ AD) :
  AB * AE + AD * AF = AC ^ 2 := by
  sorry

end rectangle_diagonal_identity_l327_327046


namespace sum_series_eq_eight_l327_327814

noncomputable def sum_series : ℝ := ∑' n : ℕ, (3 * (n + 1) + 2) / 2^(n + 1)

theorem sum_series_eq_eight : sum_series = 8 := 
 by
  sorry

end sum_series_eq_eight_l327_327814


namespace convex_pentagon_contains_integer_point_l327_327174

-- Lean code defining the problem statement
def is_integer_point (p : ℝ × ℝ) : Prop :=
  p.1 ∈ ℤ ∧ p.2 ∈ ℤ

def is_on_boundary_or_inside (p : ℝ × ℝ) (P : list (ℝ × ℝ)) : Prop :=
  -- This definition will depend on some function that checks if the point p is inside or on the boundary of the polygon P
  sorry -- Placeholder for the actual definition

theorem convex_pentagon_contains_integer_point (A B C D E : ℝ × ℝ)
  (hA : is_integer_point A) (hB : is_integer_point B)
  (hC : is_integer_point C) (hD : is_integer_point D)
  (hE : is_integer_point E) :
  ∃ (x y : ℤ), is_on_boundary_or_inside (x, y) [A, B, C, D, E] :=
sorry

end convex_pentagon_contains_integer_point_l327_327174


namespace limit_sum_areas_as_n_tends_to_infinity_l327_327764

-- Define the conditions
def side_length (m : ℝ) : ℝ := m
def first_circle_radius (m : ℝ) : ℝ := m / 2
def first_circle_area (m : ℝ) : ℝ := π * (first_circle_radius m) ^ 2

-- The scaling factor for subsequent circles (in area) is 1/2
def scaling_factor_area : ℝ := 1 / 2

-- Define the sum of areas of the first n circles
def sum_areas_first_n_circles (m : ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, first_circle_area m * (scaling_factor_area ^ k))

-- Define the limit of the sum of areas as n approaches infinity
def limit_sum_areas (m : ℝ) : ℝ :=
  first_circle_area m / (1 - scaling_factor_area)

-- The theorem to be proved
theorem limit_sum_areas_as_n_tends_to_infinity (m : ℝ) :
  (∀ n : ℕ, sum_areas_first_n_circles m n < limit_sum_areas m + 1e-10) → 
  limit_sum_areas m = π * m^2 / 2 :=
by 
  sorry

end limit_sum_areas_as_n_tends_to_infinity_l327_327764


namespace seating_arrangements_l327_327702

-- Definitions
def table_size : ℕ := 12
def number_of_couples : ℕ := 6

-- Theorem and proof placeholder
theorem seating_arrangements : 
  let valid_seating (n : ℕ) : Prop :=
    ∃ s : list (ℕ × ℕ), s.length = n ∧
      (∀ i j, s[i].1 ≠ s[j].1 ∧
              s[i].2 ≠ s[j].2 ∧
              s[i].2 = (s[i].1 + 1) % n ∨
              s[i].2 = (s[i].1 - 1) % n ∨
              s[i].2 = (s[i].1 + (n / 2)) % n ∨
              s[i].2 = (s[i].1 - (n / 2)) % n)
   in valid_seating table_size
   → number_of_couples * factorial (number_of_couples - 1) * 2 = 2880 := 
sorry

end seating_arrangements_l327_327702


namespace find_real_pairs_l327_327346

theorem find_real_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end find_real_pairs_l327_327346


namespace find_ratios_sum_l327_327797
noncomputable def Ana_biking_rate : ℝ := 8.6
noncomputable def Bob_biking_rate : ℝ := 6.2
noncomputable def CAO_biking_rate : ℝ := 5

variable (a b c : ℝ)

-- Conditions  
def Ana_distance := 2 * a + b + c = Ana_biking_rate
def Bob_distance := b + c = Bob_biking_rate
def Cao_distance := Real.sqrt (b^2 + c^2) = CAO_biking_rate

-- Main statement
theorem find_ratios_sum : 
  Ana_distance a b c ∧ 
  Bob_distance b c ∧ 
  Cao_distance b c →
  ∃ (p q r : ℕ), p + q + r = 37 ∧ Nat.gcd p q = 1 ∧ ((a / c) = p / r) ∧ ((b / c) = q / r) ∧ ((a / b) = p / q) :=
sorry

end find_ratios_sum_l327_327797


namespace abs_neg_2023_l327_327667

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l327_327667


namespace prove_range_of_a_l327_327056

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a_l327_327056


namespace min_side_length_equilateral_triangle_covering_square_l327_327284

noncomputable def min_side_length_of_covering_triangle : ℝ :=
  1 + 2 / Real.sqrt 3

theorem min_side_length_equilateral_triangle_covering_square (a : ℝ) :
  (∀ (ABC : @square ℝ = Δ EFG ∧ side_length ABC = 1), ∃ efg : eq_triangle ℝ, side_length efg ≥ 1 ∧ side_length_triangle efg = a) 
  -> a = min_side_length_of_covering_triangle :=
begin
  intro H,
  sorry

end min_side_length_equilateral_triangle_covering_square_l327_327284


namespace sin_alpha_is_sqrt3_div_2_l327_327052

variables (α : ℝ) (m : ℝ) (P : ℝ × ℝ)
variable [fact (m ≠ 0)]
variable [fact (P = (real.sqrt 3, m))]
variable [fact (real.cos α = m / 6)]

theorem sin_alpha_is_sqrt3_div_2 :
  real.sin α = real.sqrt 3 / 2 :=
sorry

end sin_alpha_is_sqrt3_div_2_l327_327052


namespace abcd_product_l327_327971

noncomputable def A := (Real.sqrt 3000 + Real.sqrt 3001)
noncomputable def B := (-Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def C := (Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def D := (Real.sqrt 3001 - Real.sqrt 3000)

theorem abcd_product :
  A * B * C * D = -1 :=
by
  sorry

end abcd_product_l327_327971


namespace graph_intersects_x_axis_once_l327_327097

noncomputable def f (m x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3 / 2) * m

theorem graph_intersects_x_axis_once (m : ℝ) :
  (∃ x : ℝ, f m x = 0 ∧ ∀ y : ℝ, f m y = 0 → y = x) ↔ (m = 1 ∨ m = 3 ∨ m = -2) :=
by
  sorry

end graph_intersects_x_axis_once_l327_327097


namespace spot_can_reach_area_l327_327193

theorem spot_can_reach_area (a : ℝ) (r : ℝ) (hex_side_length : a = 1.5) (rope_length : r = 3) :
    let full_reachable_area := π * r^2,
        inaccessible_sector_area := π * r^2 * (60 / 360),
        inaccessible_area := 4 * inaccessible_sector_area
    in full_reachable_area - inaccessible_area = 3 * π := sorry

end spot_can_reach_area_l327_327193


namespace solveForN_l327_327437

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l327_327437


namespace find_angle_F_l327_327579

-- Define the given conditions and the goal
variable (EF GH : ℝ) (angleE angleF angleG angleH : ℝ)
variable (h1 : EF ∥ GH) (h2 : angleE = 3 * angleH) (h3 : angleG = 2 * angleF) 

theorem find_angle_F (h_sum : angleF + angleG = 180) : angleF = 60 :=
by sorry

end find_angle_F_l327_327579


namespace problem_binomial_coefficients_largest_binomial_coefficient_largest_coefficient_l327_327127

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem problem_binomial_coefficients (n k : ℕ) : 
  (1 + 2 : ℚ)^n = ∑ i in finset.range (n + 1), (binomial_coefficient n i) * (2 : ℚ)^i ∧
  (binomial_coefficient n 1) + (binomial_coefficient n 2) * 2 + (binomial_coefficient n 3) * 4 = 201 :=
begin
  sorry
end

theorem largest_binomial_coefficient :
  ∀ n : ℕ, ∃ k : ℕ, k = 6 → 
  @finset.max' ℕ _ (finset.range (n + 1)) (λ k, binomial_coefficient n k) = k :=
begin
  sorry
end

theorem largest_coefficient :
  ∀ n : ℕ, ∃ k : ℕ, k = 7 → 
  @finset.max' ℕ _ (finset.range (n + 1)) (λ k, binomial_coefficient n k * (2 : ℚ)^k) = k :=
begin
  sorry
end

end problem_binomial_coefficients_largest_binomial_coefficient_largest_coefficient_l327_327127


namespace factorial_division_squared_l327_327310

theorem factorial_division_squared:
  (10! / 9!) ^ 2 = 100 := 
  sorry

end factorial_division_squared_l327_327310


namespace remaining_water_in_bathtub_l327_327943

theorem remaining_water_in_bathtub : 
  ∀ (dripping_rate : ℕ) (evaporation_rate : ℕ) (duration_hr : ℕ) (dumped_out_liters : ℕ), 
    dripping_rate = 40 →
    evaporation_rate = 200 →
    duration_hr = 9 →
    dumped_out_liters = 12 →
    let total_dripped_in_ml := dripping_rate * 60 * duration_hr in
    let total_evaporated_in_ml := evaporation_rate * duration_hr in
    let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml in
    let dumped_out_in_ml := dumped_out_liters * 1000 in
    net_water_in_ml - dumped_out_in_ml = 7800 :=
by
  intros dripping_rate evaporation_rate duration_hr dumped_out_liters
  intros rate_eq evap_eq duration_eq dump_eq
  simp [rate_eq, evap_eq, duration_eq, dump_eq]
  let total_dripped_in_ml := 40 * 60 * 9
  let total_evaporated_in_ml := 200 * 9
  let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml
  let dumped_out_in_ml := 12 * 1000
  simp [net_water_in_ml, dumped_out_in_ml]
  sorry

end remaining_water_in_bathtub_l327_327943


namespace find_n_for_prime_digit_number_l327_327344

def is_primedigit_number (n : ℕ) : ℕ :=
  if h : n > 0 then 7 + List.foldr (λ i acc, acc + 10 ^ i) 0 (List.range (n-1)) else 7

theorem find_n_for_prime_digit_number :
  ∀ n : ℕ, (n = 1 ∨ n = 2) ↔ Nat.Prime (is_primedigit_number n) := by
  sorry

end find_n_for_prime_digit_number_l327_327344


namespace car_graph_representation_l327_327306

theorem car_graph_representation :
  ∀ (v t : ℝ), ∃ (graph : string),
  (graph = "Graph A") →
  ((∃ v : ℝ, ∃ t : ℝ, ∃ t_B : ℝ, ∃ v_B : ℝ,
    v_B = 3 * v ∧
    t_B = t / 3 ∧
    (graph = "Graph A"))) :=
by
  intros v t
  use "Graph A"
  intros h
  use v, t, t / 3, 3 * v
  simp [h]
  split
  · refl
  · exact div_eq_div_iff (by norm_num : (3 : ℝ) ≠ 0) rfl

end car_graph_representation_l327_327306


namespace power_addition_rule_l327_327086

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l327_327086


namespace total_possible_ranking_sequences_l327_327118

-- Define the teams participating in the tournament
inductive Team : Type
| A | B | C | D | E | F

-- Total number of teams
def num_teams : ℕ := 6

-- Function to calculate factorial
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Number of ways to arrange 3 teams
def arrangements_per_group : ℕ := factorial 3

-- Number of possible ranking sequences for all teams
def total_ranking_sequences : ℕ := arrangements_per_group * arrangements_per_group

-- Statement of the problem
theorem total_possible_ranking_sequences : total_ranking_sequences = 36 :=
by
  unfold total_ranking_sequences arrangements_per_group factorial
  norm_num
  sorry

end total_possible_ranking_sequences_l327_327118


namespace sum_of_solutions_l327_327825

theorem sum_of_solutions (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ S, S = 9 ∧ ∀ x1 x2, x1 + x2 = S) := by
  intros h
  sorry

end sum_of_solutions_l327_327825


namespace area_of_triangle_QRS_l327_327606

/-- Let PQRS be a tetrahedron such that edges PQ, PR, and PS are mutually perpendicular. 
Let the areas of triangles PQR, PRS, and PQS be denoted by a, b, and c, respectively. 
In terms of a, b, and c, we need to find the area of triangle QRS. 
Our goal is to prove that the area of triangle QRS is sqrt(a^2 + b^2 + c^2). -/
theorem area_of_triangle_QRS (a b c : ℝ) (P Q R S : ℝ × ℝ × ℝ) 
  (hPQ : Q = (fst P + fst Q, snd P, trd P))
  (hPR : R = (fst P, snd P + snd R, trd P))
  (hPS : S = (fst P, snd P, trd P + trd S))
  (h_area_PQR : 1/2 * (fst Q - fst P) * (snd R - snd P) = a)
  (h_area_PRS : 1/2 * (snd R - snd P) * (trd S - trd P) = b)
  (h_area_PQS : 1/2 * (fst Q - fst P) * (trd S - trd P) = c) :
  ∃ K : ℝ, K = sqrt (a^2 + b^2 + c^2) :=
sorry

end area_of_triangle_QRS_l327_327606


namespace deal_or_no_deal_l327_327111

open Nat

theorem deal_or_no_deal {n : ℕ} (boxes : Finset ℕ) (at_least_200k : Finset ℕ) :
  boxes.card = 30 ∧
  at_least_200k.card = 7 ∧
  at_least_200k ⊆ boxes →
  ∃ remove : Finset ℕ, remove.card = 16 ∧ (boxes \ remove).card = 14 ∧
  (boxes \ remove) ∩ at_least_200k = at_least_200k ∧
  (boxes \ remove).card - at_least_200k.card = 7 :=
begin
  sorry
end

end deal_or_no_deal_l327_327111


namespace sum_to_product_identity_l327_327323

variable (x : ℝ)

theorem sum_to_product_identity : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sum_to_product_identity_l327_327323


namespace tetrahedron_edges_perpendicular_l327_327253

variables {Point : Type} [InnerProductSpace ℝ Point]

/-- Given that in the tetrahedron ABCD, edge AB is perpendicular to edge CD,
    and edge BC is perpendicular to edge AD. Prove that edge AC is perpendicular to edge BD. -/
theorem tetrahedron_edges_perpendicular
  {A B C D : Point}
  (hAB_CD : ⟪A - B, C - D⟫ = 0)
  (hBC_AD : ⟪B - C, A - D⟫ = 0) :
  ⟪A - C, B - D⟫ = 0 := 
sorry

end tetrahedron_edges_perpendicular_l327_327253


namespace compare_abc_l327_327024

noncomputable def a : ℝ := 8.1 ^ 0.51
noncomputable def b : ℝ := 8.1 ^ 0.5
noncomputable def c : ℝ := log 3 0.3

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l327_327024


namespace krystiana_monthly_income_l327_327964

theorem krystiana_monthly_income :
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  first_floor_income + second_floor_income + third_floor_income = 165 :=
by
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  have h1: first_floor_income = 45 := by simp [first_floor_income]
  have h2: second_floor_income = 60 := by simp [second_floor_income]
  have h3: third_floor_income = 60 := by simp [third_floor_income]
  rw [h1, h2, h3]
  simp
  done

end krystiana_monthly_income_l327_327964


namespace find_equivalent_to_minus_three_halves_l327_327242

theorem find_equivalent_to_minus_three_halves :
  (-1 - (1 / 2) = - 3 / 2) ∧ (1 / 2 - 1 ≠ - 3 / 2) ∧ (1 - 1 / 3 ≠ - 3 / 2) ∧ (-1 + 1 / 3 ≠ - 3 / 2) :=
by 
  split; norm_num; -- for the comparisons involving fractions
  norm_num
  sorry

end find_equivalent_to_minus_three_halves_l327_327242


namespace sin_sum_to_product_l327_327334

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (7 * x) = 2 * sin (5 * x) * cos (2 * x) := 
sorry

end sin_sum_to_product_l327_327334


namespace min_value_xy_l327_327446

theorem min_value_xy (x y : ℝ) (h : x * y = 1) : x^2 + 4 * y^2 ≥ 4 := by
  sorry

end min_value_xy_l327_327446


namespace abs_neg_2023_l327_327668

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l327_327668


namespace pizza_consumption_order_l327_327357

theorem pizza_consumption_order :
  let e := 1/6
  let s := 1/4
  let n := 1/3
  let o := 1/8
  let j := 1 - e - s - n - o
  (n > s) ∧ (s > e) ∧ (e = j) ∧ (j > o) :=
by
  sorry

end pizza_consumption_order_l327_327357


namespace slope_of_line_l327_327214

-- Define that the line equation is in slope-intercept form
def line_eq (x : ℝ) : ℝ := 2 * x + 1

-- Prove that the slope of the line is 2
theorem slope_of_line : ∃ m b : ℝ, (line_eq = λ x, m * x + b) ∧ m = 2 :=
by
  use 2
  use 1
  sorry

end slope_of_line_l327_327214


namespace initial_bowls_count_l327_327280

variable (x : ℝ)

theorem initial_bowls_count :
  ((0.4830917874396135 / 100) = ((104 * (20 - 18)) / (18 * x))) →
  x = 2393 :=
by
  intro h
  rw [← mul_left_inj' (by norm_num : (18 : ℝ) ≠ 0), ← mul_assoc] at h
  simp only [mul_comm (18 : ℝ), ← div_eq_mul_one_div] at h
  field_simp [ne_of_gt (by norm_num : (18 : ℝ) > 0), ne_of_gt (by norm_num : (0.4830917874396135 : ℝ) > 0)] at h
  norm_num at h
  sorry

end initial_bowls_count_l327_327280


namespace average_speed_entire_trip_l327_327272

-- Define the conditions as given in the problem
def distance1 := 9 -- in km
def speed1 := 12 -- in km/hr
def distance2 := 12 -- in km
def speed2 := 9 -- in km/hr

-- Lean statement to be proved: The average speed for the entire trip
theorem average_speed_entire_trip : 
  let time1 := distance1 / speed1 in
  let time2 := distance2 / speed2 in
  let total_distance := distance1 + distance2 in
  let total_time := time1 + time2 in
  total_distance / total_time = 10.1 := 
by
  -- Proof goes here
  sorry

end average_speed_entire_trip_l327_327272


namespace eggs_in_fridge_l327_327200

theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) (eggs_used : ℕ) (eggs_in_fridge : ℕ)
  (h1 : total_eggs = 60)
  (h2 : eggs_per_cake = 5)
  (h3 : num_cakes = 10)
  (h4 : eggs_used = eggs_per_cake * num_cakes)
  (h5 : eggs_in_fridge = total_eggs - eggs_used) :
  eggs_in_fridge = 10 :=
by
  sorry

end eggs_in_fridge_l327_327200


namespace parabola_with_given_focus_l327_327216

-- Defining the given condition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Defining the focus coordinates
def focus_coords : ℝ × ℝ := (-3, 0)

-- Proving that the standard equation of the parabola with the left focus of the hyperbola as its focus is y^2 = -12x
theorem parabola_with_given_focus :
  ∃ p : ℝ, (∃ focus : ℝ × ℝ, focus = focus_coords) → 
  ∀ y x : ℝ, y^2 = 4 * p * x → y^2 = -12 * x :=
by
  -- placeholder for proof
  sorry

end parabola_with_given_focus_l327_327216


namespace find_angle_F_l327_327576

variable (EF GH : ℝ) -- Lengths of sides EF and GH
variable (angle_E angle_F angle_G angle_H : ℝ) -- Angles at vertices E, F, G, and H

-- Conditions given in the problem
axiom EF_parallel_GH : EF ∥ GH
axiom angle_E_eq_3_angle_H : angle_E = 3 * angle_H
axiom angle_G_eq_2_angle_F : angle_G = 2 * angle_F

-- Target statement to prove
theorem find_angle_F : angle_F = 60 := by
  -- Conditions setup:
  have angle_F_plus_angle_G := 180 - angle_G ; sorry
  -- Solve for angle_F
  have angle_F_eq_60 := 180 / 3; sorry
  sorry

end find_angle_F_l327_327576


namespace find_angle_F_l327_327577

variable (EF GH : ℝ) -- Lengths of sides EF and GH
variable (angle_E angle_F angle_G angle_H : ℝ) -- Angles at vertices E, F, G, and H

-- Conditions given in the problem
axiom EF_parallel_GH : EF ∥ GH
axiom angle_E_eq_3_angle_H : angle_E = 3 * angle_H
axiom angle_G_eq_2_angle_F : angle_G = 2 * angle_F

-- Target statement to prove
theorem find_angle_F : angle_F = 60 := by
  -- Conditions setup:
  have angle_F_plus_angle_G := 180 - angle_G ; sorry
  -- Solve for angle_F
  have angle_F_eq_60 := 180 / 3; sorry
  sorry

end find_angle_F_l327_327577


namespace invalid_inverse_statement_l327_327742

/- Define the statements and their inverses -/

/-- Statement A: Vertical angles are equal. -/
def statement_A : Prop := ∀ {α β : ℝ}, α ≠ β → α = β

/-- Inverse of Statement A: If two angles are equal, then they are vertical angles. -/
def inverse_A : Prop := ∀ {α β : ℝ}, α = β → α ≠ β

/-- Statement B: If |a| = |b|, then a = b. -/
def statement_B (a b : ℝ) : Prop := abs a = abs b → a = b

/-- Inverse of Statement B: If a = b, then |a| = |b|. -/
def inverse_B (a b : ℝ) : Prop := a = b → abs a = abs b

/-- Statement C: If two lines are parallel, then the alternate interior angles are equal. -/
def statement_C (l1 l2 : Prop) : Prop := l1 → l2

/-- Inverse of Statement C: If the alternate interior angles are equal, then the two lines are parallel. -/
def inverse_C (l1 l2 : Prop) : Prop := l2 → l1

/-- Statement D: If a^2 = b^2, then a = b. -/
def statement_D (a b : ℝ) : Prop := a^2 = b^2 → a = b

/-- Inverse of Statement D: If a = b, then a^2 = b^2. -/
def inverse_D (a b : ℝ) : Prop := a = b → a^2 = b^2

/-- The statement that does not have a valid inverse among A, B, C, and D is statement A. -/
theorem invalid_inverse_statement : ¬inverse_A :=
by
sorry

end invalid_inverse_statement_l327_327742


namespace area_GIC_eq_l327_327158

variables {A B C G I : Type} [triangle A B C]
variable [centroid A B C G]
variable [incenter A B C I]
variable [inradius I r]
variables {a b : ℝ} 

theorem area_GIC_eq : 
  triangle_area A B C G I r a b = abs (a - b) * r / 6 := sorry

end area_GIC_eq_l327_327158


namespace no_four_consecutive_sums_of_squares_l327_327320

theorem no_four_consecutive_sums_of_squares :
  ¬ ∃ (n : ℕ), (∃ a b : ℕ, n = a^2 + b^2) ∧
               (∃ a b : ℕ, (n+1) = a^2 + b^2) ∧
               (∃ a b : ℕ, (n+2) = a^2 + b^2) ∧
               (∃ a b : ℕ, (n+3) = a^2 + b^2) :=
begin
  sorry
end

end no_four_consecutive_sums_of_squares_l327_327320


namespace sum_of_absolute_values_l327_327122

variables {a : ℕ → ℤ} {S₁₀ S₁₈ : ℤ} {T₁₈ : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem sum_of_absolute_values 
  (h1 : a 0 > 0) 
  (h2 : a 9 * a 10 < 0) 
  (h3 : sum_n_terms a 9 = 36) 
  (h4 : sum_n_terms a 17 = 12) :
  (sum_n_terms a 9) - (sum_n_terms a 17 - sum_n_terms a 9) = 60 :=
sorry

end sum_of_absolute_values_l327_327122


namespace tangent_circles_condition_l327_327384

noncomputable def geo_condition (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  let ω_O : circle A := circumcircle_right_triangle A B C
  let ω_Q : circle B := tangent_circles AC BC in
  let rad_Q : real := radius_circumcircle ω_Q in
  rad_Q = (AC + BC - AB)

theorem tangent_circles_condition (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  let ω_O : circle A := circumcircle_right_triangle A B C
  let ω_Q : circle B := tangent_circles AC BC in
  let rad_Q : real := radius_circumcircle ω_Q in
  radius_circumcircle ω_Q = (AC + BC - AB) ↔
  tangent_circles rad_Q ω_O sorry

end tangent_circles_condition_l327_327384


namespace area_of_figure_l327_327568

theorem area_of_figure (PQ ST PR QT RS PT : ℝ)
  (PQ_parallel_ST : PQ ∥ ST)
  (PR_parallel_QT : PR ∥ QT)
  (RS_parallel_PT : RS ∥ PT)
  (PQ_length : PQ = 2)
  (QR_length : QR = 1)
  (angle_QPR : ∠QPR = 90) :
  area_of_figure = 4 :=
sorry

end area_of_figure_l327_327568


namespace three_card_deal_probability_l327_327225

theorem three_card_deal_probability :
  (4 / 52) * (4 / 51) * (4 / 50) = 16 / 33150 := 
by 
  sorry

end three_card_deal_probability_l327_327225


namespace value_of_k_l327_327318

theorem value_of_k {k : ℝ} :
  (∀ x : ℝ, (x^2 + k * x + 24 > 0) ↔ (x < -6 ∨ x > 4)) →
  k = 2 :=
by
  sorry

end value_of_k_l327_327318


namespace probability_positive_product_l327_327933

theorem probability_positive_product :
  let labels := [-4, 0, 2, 3]
  let outcomes := list.product labels labels
  let positive_product_outcomes := [(2, 3), (3, 2)]
  let total_outcomes := list.length outcomes - list.countp id (λ p, p.1 = p.2) -- Remove same draws
  let num_positive_product := list.count (λ p, p ∈ positive_product_outcomes) outcomes
  real.to_rat (num_positive_product / (total_outcomes - 4)) = 1 / 6 :=
by
  let labels := [-4, 0, 2, 3]
  let outcomes := list.filter (λ p, p.1 ≠ p.2) (list.product labels labels)
  let positive_product_outcomes := [(2, 3), (3, 2)]
  let num_positive_product := list.count (λ p, p ∈ positive_product_outcomes) outcomes
  let total_outcomes := list.length outcomes
  have : total_outcomes = 12 := sorry
  have : num_positive_product = 2 := sorry
  have : real.to_rat (num_positive_product / (total_outcomes:ℝ)) = (1 / 6:ℝ) := sorry
  exact this

end probability_positive_product_l327_327933


namespace find_smallest_value_l327_327619

open Complex

noncomputable def smallest_possible_value (w : ℂ) : ℝ :=
  if |w - 8| + |w - 3*I| = 15 then |w| else 0

theorem find_smallest_value :
  ∃ w : ℂ, (|w - 8| + |w - 3*I| = 15) ∧ smallest_possible_value w = 8 / 5 :=
sorry

end find_smallest_value_l327_327619


namespace no_equation_rep_same_graph_l327_327258

def eqn1 (x : ℝ) : ℝ := x - 2
def eqn2 (x : ℝ) : ℝ := (x^2 - 4) / (x + 2)
def eqn3 (x : ℝ) : ℝ := (x^2 - 4) / (x + 2)

theorem no_equation_rep_same_graph :
  ∀ x : ℝ, ¬(eqn1 x = eqn2 x ∧ eqn2 x = eqn3 x) :=
by sorry

end no_equation_rep_same_graph_l327_327258


namespace unique_fn_l327_327975

noncomputable def S : Set ℝ := { x | -1 < x }

theorem unique_fn (f : ℝ → ℝ) (hf : ∀ x ∈ S, f x ∈ S)
    (h1 : ∀ x y ∈ S, f (x + f y + x * f y) = y + f x + y * f x)
    (h2 : ∀ x, (-1 < x ∧ x < 0 ∨ x > 0) → StrictMono (λ x, f x / x)) :
  ∀ x ∈ S, f x = -x / (x + 1) := by
  sorry

end unique_fn_l327_327975


namespace number_of_true_propositions_l327_327292

theorem number_of_true_propositions :
  (∀ (A B C D : Type) (h1 : VerticallyOppositeAngles A B) (h2 : PerpendicularLineSegmentsShort h1) 
   (h3 : UniquePerpendicularLineThroughPoint h2) (h4 : ParallelLinesSupplementaryAngles h3), 
     true) → 
  4 = (if h1 && h2 && h3 && h4 then 4 else 0) :=
by
  sorry

end number_of_true_propositions_l327_327292


namespace x_plus_y_l327_327148

-- Variables to represent geometric entities and their properties
variables (P Q R S T U : Type) [convex_pentagon P Q R S T]
          [parallel PQ ST] [parallel QR PS] [parallel PT RS]
          (anglePQR : Float) [PQ_length : Float] [QR_length : Float] [RS_length : Float]
          (ratio_area : Float)

-- Conditions
def angle_condition (anglePQR : Float) : Prop := anglePQR = 120
def lengths_condition (PQ_length QR_length RS_length : Float) : Prop := PQ_length = 4 ∧ QR_length = 6 ∧ RS_length = 18
def area_ratio_condition (ratio_area : Float) : Prop := ratio_area = 16/81

-- Theorem to be proved
theorem x_plus_y (P Q R S T : Type) [convex_pentagon P Q R S T]
  [parallel PQ ST] [parallel QR PS] [parallel PT RS]
  (anglePQR : Float) [PQ_length : Float] [QR_length : Float] [RS_length : Float]
  (ratio_area : Float) : 
  angle_condition anglePQR →
  lengths_condition PQ_length QR_length RS_length →
  area_ratio_condition ratio_area →
  (97 : Nat) := by
    sorry

end x_plus_y_l327_327148


namespace find_angle_F_l327_327578

-- Define the given conditions and the goal
variable (EF GH : ℝ) (angleE angleF angleG angleH : ℝ)
variable (h1 : EF ∥ GH) (h2 : angleE = 3 * angleH) (h3 : angleG = 2 * angleF) 

theorem find_angle_F (h_sum : angleF + angleG = 180) : angleF = 60 :=
by sorry

end find_angle_F_l327_327578


namespace number_of_women_l327_327802

theorem number_of_women (n_men n_women n_dances men_partners women_partners : ℕ) 
  (h_men_partners : men_partners = 4)
  (h_women_partners : women_partners = 3)
  (h_n_men : n_men = 15)
  (h_total_dances : n_dances = n_men * men_partners)
  (h_women_calc : n_women = n_dances / women_partners) :
  n_women = 20 :=
sorry

end number_of_women_l327_327802


namespace conditional_prob_B_given_A_l327_327848

noncomputable def P (n k : ℕ) : ℚ := (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) : ℚ)

def event_A (s : Finset ℕ) : ℚ :=
  (P 5 2 + P 4 2) / P 9 2

def event_B (s : Finset ℕ) : ℚ :=
  P 4 2 / P 9 2

def conditional_probability (A B : ℚ) : ℚ :=
  B / A

theorem conditional_prob_B_given_A :
  conditional_probability (event_A (Finset.range 10 \ {0})) (event_B (Finset.range 10 \ {0})) = 3 / 8 :=
by
  sorry

end conditional_prob_B_given_A_l327_327848


namespace sum_of_two_primes_unique_l327_327486

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l327_327486


namespace probability_odd_or_multiple_of_3_l327_327770

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0
def favorable_outcomes (n : ℕ) : Prop := is_odd n ∨ is_multiple_of_3 n

theorem probability_odd_or_multiple_of_3 : 
  (∃ (s : finset ℕ), s = {1, 2, 3, 4, 5, 6} ∧ 
   (∑ t in s, if favorable_outcomes t then 1 else 0) / ∑ t in s, 1 = 2 / 3) :=
by
  sorry

end probability_odd_or_multiple_of_3_l327_327770


namespace arithmetic_sequence_range_of_k_l327_327887

def T (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 1 - a_n n
def c (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (T a_n n)

theorem arithmetic_sequence (a_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, T a_n n = 1 - a_n n) → (∀ n, c a_n n = n + 1) ∧ 
  (∀ n, a_n n = n / (n + 1)) :=
sorry

def b (n : ℕ) : ℝ := 1 / (2^n)
def T' (n : ℕ) : ℝ := 1 / (n + 1)

theorem range_of_k (k : ℝ) :
  (∀ n, n ∈ ℕ → T' n * (n * b n + n - 2) ≤ k * n) → k ≥ 11 / 96 :=
sorry

end arithmetic_sequence_range_of_k_l327_327887


namespace range_of_x_l327_327008

theorem range_of_x (x : ℝ) (h : floor ((x + 3) / 2) = 3) : 3 ≤ x ∧ x < 5 :=
  sorry

end range_of_x_l327_327008


namespace number_of_real_b_l327_327842

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b_l327_327842


namespace course_last_days_l327_327753

theorem course_last_days
  (n : ℕ) (kn : n = 15)
  (k : ℕ) (kk : k = 3)
  (pairs_unique : ∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! d, (d < n) ∧ (i ≠ d) ∧ (j ≠ d)) :
  ∃ d, d = (Nat.choose n 2) / (Nat.choose k 2) :=
by
  use 35
  have total_pairs : (Nat.choose n 2) = 105 := by 
    rw [kn, Nat.choose_succ_succ, Nat.choose_succ_succ, Nat.choose_self]
    norm_num
  have pairs_per_day : (Nat.choose k 2) = 3 := by 
    rw [kk, Nat.choose_succ_succ, Nat.choose_succ_succ, Nat.choose_self]
    norm_num
  have days_eq : d = 105 / 3 := by
    rw [total_pairs, pairs_per_day]
    norm_num
  rw [days_eq]
  exact rfl
  sorry

end course_last_days_l327_327753


namespace length_PR_of_similar_triangles_l327_327592

noncomputable def length_PR (YZ QR XY : ℝ) : ℝ :=
  XY * QR / YZ

theorem length_PR_of_similar_triangles 
  (YZ QR XY : ℝ) 
  (h_similar : ∀ (a b c d : ℝ), a / b = c / d) 
  (YZ_val : YZ = 35) 
  (QR_val : QR = 14) 
  (XY_val : XY = 20) :
  length_PR YZ QR XY = 8 :=
by simp [length_PR, YZ_val, QR_val, XY_val]; norm_num; sorry

end length_PR_of_similar_triangles_l327_327592


namespace alpha_beta_sum_l327_327872

-- Defining the problem conditions
def is_root (p : Polynomial ℂ) (z : ℂ) : Prop :=
  p.eval z = 0

def alpha_beta_roots (α β : ℂ) : Prop :=
  is_root (Polynomial.Coeff [1, -1, 1]) α ∧ is_root (Polynomial.Coeff [1, -1, 1]) β

-- The Lean statement for the proof problem
theorem alpha_beta_sum (α β : ℂ) (h : alpha_beta_roots α β) : α ^ 2005 + β ^ 2005 = 1 :=
  sorry

end alpha_beta_sum_l327_327872


namespace pizza_slice_division_l327_327708

theorem pizza_slice_division : 
  ∀ (num_coworkers num_pizzas slices_per_pizza : ℕ),
  num_coworkers = 12 →
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  (num_pizzas * slices_per_pizza) / num_coworkers = 2 := 
by
  intros num_coworkers num_pizzas slices_per_pizza h_coworkers h_pizzas h_slices
  rw [h_coworkers, h_pizzas, h_slices]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slice_division_l327_327708


namespace mary_double_counted_sheep_l327_327166

theorem mary_double_counted_sheep :
  ∀ (total_real animals_counted forgotten_pigs double_counted: ℕ),
    total_real = 56 →
    animals_counted = 60 →
    forgotten_pigs = 3 →
    animals_counted = total_real + double_counted - forgotten_pigs →
    double_counted = 7 :=
by
  intros total_real animals_counted forgotten_pigs double_counted
  assume h_real h_counted h_forgotten h_eq
  sorry

end mary_double_counted_sheep_l327_327166


namespace find_m_l327_327914

theorem find_m (m : ℤ) (h : (1, 4, 2*m + 8) ∈ {s : ℕ × ℕ × ℤ | s.1 * s.3 = s.2 * s.2}) : 
  m = 4 :=
sorry

end find_m_l327_327914


namespace savings_by_buying_in_bulk_l327_327597

-- Definitions based on conditions
def numMachines := 10
def ballBearingsPerMachine := 30
def normalPricePerBallBearing := 1.0
def salePricePerBallBearing := 0.75
def bulkDiscount := 0.20
def totalBallBearings := numMachines * ballBearingsPerMachine

-- The theorem statement we need to prove
theorem savings_by_buying_in_bulk :
  let normalCost := totalBallBearings * normalPricePerBallBearing
  let saleCostBeforeDiscount := totalBallBearings * salePricePerBallBearing
  let discountAmount := bulkDiscount * saleCostBeforeDiscount
  let saleCostAfterDiscount := saleCostBeforeDiscount - discountAmount in
  normalCost - saleCostAfterDiscount = 120 :=
by
  sorry

end savings_by_buying_in_bulk_l327_327597


namespace sqrt_eq_9_implies_n_eq_73_l327_327441

theorem sqrt_eq_9_implies_n_eq_73 (n : ℕ) : sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_eq_9_implies_n_eq_73_l327_327441


namespace fabric_cost_equation_l327_327805

theorem fabric_cost_equation (x : ℝ) :
  (3 * x + 5 * (138 - x) = 540) :=
sorry

end fabric_cost_equation_l327_327805


namespace no_prime_sum_10003_l327_327511

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327511


namespace problem_statement_l327_327413

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1/2| + |x + 1/2|

-- Define the set M
def M : Set ℝ := {x | f x < 2}

-- The proof statement
theorem problem_statement (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  M = Set.Ioo (-1 : ℝ) 1 ∧ |a + b| < |1 + a * b| :=
by
  -- Split the proof into two parts
  split
  -- First part: Prove M = (-1, 1)
  { sorry }

  -- Second part: Prove |a + b| < |1 + a * b| for a, b ∈ M
  { sorry }

end problem_statement_l327_327413


namespace sin_theta_l327_327372

def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta:
  (∀ x, f x ≤ f θ) → Real.sin θ = 3 / 5 :=
by
  sorry

end sin_theta_l327_327372


namespace appropriate_investigation_method_l327_327196

theorem appropriate_investigation_method
  (volume_of_investigation_large : Prop)
  (no_need_for_comprehensive_investigation : Prop) :
  (∃ (method : String), method = "sampling investigation") :=
by
  sorry

end appropriate_investigation_method_l327_327196


namespace ratio_HD_HA_zero_l327_327116

theorem ratio_HD_HA_zero (a b c : ℝ) (ha : a = 12) (hb : b = 13) (hc : c = 17) (hA : ∃ hA : ℝ, ∃ hD : ℝ, hD = 0): 
  ∀ (HD HA : ℝ), HD = 0 → HA = hA → HD / HA = 0 := 
by sorry

end ratio_HD_HA_zero_l327_327116


namespace max_min_distance_on_sphere_l327_327360

noncomputable def sphere := {p : ℝ × ℝ × ℝ // ∥p∥ = 1}

def euclidean_distance (p q : ℝ × ℝ × ℝ) : ℝ := ∥p - q∥

def minimum_distance (points : fin 5 → sphere) : ℝ :=
  Finset.univ.image (λ ⟨i, j⟩, euclidean_distance (points i) (points j)).min' sorry

theorem max_min_distance_on_sphere :
  ∀ (points : fin 5 → sphere),
  minimum_distance points ≤ sqrt 2 :=
sorry

end max_min_distance_on_sphere_l327_327360


namespace total_profit_is_64000_l327_327757

-- Definitions for investments and periods
variables (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ)

-- Conditions from the problem
def condition1 := IA = 5 * IB
def condition2 := TA = 3 * TB
def condition3 := Profit_B = 4000
def condition4 := Profit_A / Profit_B = (IA * TA) / (IB * TB)

-- Target statement to be proved
theorem total_profit_is_64000 (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ) :
  condition1 IA IB → condition2 TA TB → condition3 Profit_B → condition4 IA TA IB TB Profit_A Profit_B → 
  Total_Profit = Profit_A + Profit_B → Total_Profit = 64000 :=
by {
  sorry
}

end total_profit_is_64000_l327_327757


namespace negate_proposition_l327_327064

open Classical

variable (x : ℝ)

theorem negate_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0 :=
by
  sorry

end negate_proposition_l327_327064


namespace find_triplets_l327_327624

theorem find_triplets (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a ∣ b + c + 1) (h5 : b ∣ c + a + 1) (h6 : c ∣ a + b + 1) :
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 2) ∨ (a, b, c) = (3, 4, 4) ∨ 
  (a, b, c) = (1, 1, 3) ∨ (a, b, c) = (2, 2, 5) :=
sorry

end find_triplets_l327_327624


namespace parabola_focus_line_ratio_l327_327049

noncomputable def ratio_AF_BF : ℝ := (Real.sqrt 5 + 3) / 2

theorem parabola_focus_line_ratio :
  ∀ (F A B : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (A.2 = 2 * A.1 - 2 ∧ A.2^2 = 4 * A.1 ) ∧ 
    (B.2 = 2 * B.1 - 2 ∧ B.2^2 = 4 * B.1) ∧ 
    A.2 > 0 -> 
  |(A.1 - F.1) / (B.1 - F.1)| = ratio_AF_BF :=
by
  sorry

end parabola_focus_line_ratio_l327_327049


namespace plane_equation_passing_through_point_and_parallel_l327_327833

-- Define the point and the plane parameters
def point : ℝ × ℝ × ℝ := (2, 3, 1)
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

-- Main theorem statement
theorem plane_equation_passing_through_point_and_parallel :
  ∃ D : ℝ, plane 2 (-1) 3 D 2 3 1 ∧ plane 2 (-1) 3 D 0 0 0 :=
sorry

end plane_equation_passing_through_point_and_parallel_l327_327833


namespace SMUG_TWC_minimum_bouts_l327_327228

noncomputable def minimum_bouts (n : ℕ) : ℕ :=
  let total_edges := (n * (n - 1)) / 2
  let turan_edges := (n^2) / 4
  total_edges - turan_edges

theorem SMUG_TWC_minimum_bouts :
  minimum_bouts 2008 = 999000 :=
by
  let total_edges := (2008 * 2007) / 2
  let turan_edges := (2008^2) / 4
  exact total_edges - turan_edges

end SMUG_TWC_minimum_bouts_l327_327228


namespace factorial_divisible_l327_327379

theorem factorial_divisible 
  (P : Polynomial ℤ) 
  (n : ℕ) 
  (p : ℕ) 
  (hP_deg : P.degree = n)
  (hP_lead : P.leadingCoeff = 1) 
  (hP_div : ∀ x : ℤ, p ∣ P.eval x) : 
  p ∣ n! := 
sorry

end factorial_divisible_l327_327379


namespace spinner_probabilities_l327_327773

noncomputable def prob_A : ℚ := 1 / 3
noncomputable def prob_B : ℚ := 1 / 4
noncomputable def prob_C : ℚ := 5 / 18
noncomputable def prob_D : ℚ := 5 / 36

theorem spinner_probabilities :
  prob_A + prob_B + prob_C + prob_D = 1 ∧
  prob_C = 2 * prob_D :=
by {
  -- The statement of the theorem matches the given conditions and the correct answers.
  -- Proof will be provided later.
  sorry
}

end spinner_probabilities_l327_327773


namespace gcf_of_lcm_9_15_and_10_21_is_5_l327_327727

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l327_327727


namespace sniper_B_has_greater_chance_of_winning_l327_327115

-- Define the probabilities for sniper A
def p_A_1 := 0.4
def p_A_2 := 0.1
def p_A_3 := 0.5

-- Define the probabilities for sniper B
def p_B_1 := 0.1
def p_B_2 := 0.6
def p_B_3 := 0.3

-- Define the expected scores for sniper A and B
def E_A := 1 * p_A_1 + 2 * p_A_2 + 3 * p_A_3
def E_B := 1 * p_B_1 + 2 * p_B_2 + 3 * p_B_3

-- The statement we want to prove
theorem sniper_B_has_greater_chance_of_winning : E_B > E_A := by
  simp [E_A, E_B, p_A_1, p_A_2, p_A_3, p_B_1, p_B_2, p_B_3]
  sorry

end sniper_B_has_greater_chance_of_winning_l327_327115


namespace valid_assignment_l327_327235

/-- A function to check if an expression is a valid assignment expression -/
def is_assignment (lhs : String) (rhs : String) : Prop :=
  lhs = "x" ∧ (rhs = "3" ∨ rhs = "x + 1")

theorem valid_assignment :
  (is_assignment "x" "x + 1") ∧
  ¬(is_assignment "3" "x") ∧
  ¬(is_assignment "x" "3") ∧
  ¬(is_assignment "x" "x2 + 1") :=
by
  sorry

end valid_assignment_l327_327235


namespace speed_conversion_l327_327267

theorem speed_conversion (speed_kmph : ℝ) (h : speed_kmph = 18) : speed_kmph * (1000 / 3600) = 5 := by
  sorry

end speed_conversion_l327_327267


namespace sum_g_10_l327_327611

variable (f g : ℕ → ℝ)
variable (h_f : ∀ x : ℝ, f x = x^2 - 5 * x + 12)
variable (h_g : ∀ x : ℝ, g (f x) = 2 * x + 3)

theorem sum_g_10 (x1 x2 : ℝ) (h1 : x1^2 - 5 * x1 + 2 = 0) (h2 : x2^2 - 5 * x2 + 2 = 0) :
    (g 10 + g 10) = 16 :=
by
  sorry

end sum_g_10_l327_327611


namespace combined_bank_discount_is_correct_l327_327223

noncomputable def bank_discount (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / (1 - (R * T))

def P1 : ℝ := 2560
def P2 : ℝ := 3800
def P3 : ℝ := 4500

def R1 : ℝ := 0.05
def R2 : ℝ := 0.07
def R3 : ℝ := 0.08

def T1 : ℝ := 0.5
def T2 : ℝ := 0.75
def T3 : ℝ := 1

def BD1 := bank_discount P1 R1 T1
def BD2 := bank_discount P2 R2 T2
def BD3 := bank_discount P3 R3 T3

def combined_bank_discount : ℝ :=
  BD1 + BD2 + BD3

theorem combined_bank_discount_is_correct : combined_bank_discount ≈ 667.47 :=
  by sorry

end combined_bank_discount_is_correct_l327_327223


namespace simplify_product_l327_327660

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l327_327660


namespace max_fly_path_length_l327_327771

noncomputable def max_path_length_in_box (a b c : ℝ) : ℝ :=
  4 + 4 * real.sqrt 5 + real.sqrt 6

theorem max_fly_path_length :
  max_path_length_in_box 2 1 1 = 4 + 4 * real.sqrt 5 + real.sqrt 6 := 
sorry

end max_fly_path_length_l327_327771


namespace arithmetic_seq_ratio_l327_327398

open Classical

-- Define sequence terms and sums
variable {α : Type*}
variable [LinearOrderedField α]

def a : ℕ → α
def S (n : ℕ) : α := ∑ i in finset.range (n + 1), a i

-- Define conditions
variable {d : α}
variable {a1 a9 a17 : α}
variable (h1 : a 1 = a1)
variable (h9 : a 9 = a1 + 8 * d)
variable (h17 : a 17 = a1 + 16 * d)

-- Relationship between terms given
theorem arithmetic_seq_ratio (h : 3 * a1 + 4 * (a1 + 8 * d) = a1 + 16 * d) :
  S 17 / S 9 = (68 : α) / 9 := by
sorry

end arithmetic_seq_ratio_l327_327398


namespace no_prime_sum_10003_l327_327512

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l327_327512


namespace find_m_set_l327_327900

-- Definitions for the given conditions
def A : set ℝ := {x | x^2 - 5 * x + 6 = 0}
def B (m : ℝ) : set ℝ := {x | (m - 1) * x - 1 = 0}

-- The condition that A ∩ B = B implies B ⊆ A
theorem find_m_set :
  {m : ℝ | B m ⊆ A } = {1, 3/2, 4/3} :=
sorry

end find_m_set_l327_327900


namespace contradiction_proof_a_b_zero_l327_327233

theorem contradiction_proof_a_b_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  by_contradiction h_neg
  have h_not_both_zero : ¬ (a = 0 ∧ b = 0) := h_neg
  sorry

end contradiction_proof_a_b_zero_l327_327233


namespace shaded_area_l327_327755

theorem shaded_area (width height : ℝ) (a b : ℝ) (H1 : width = 10) (H2 : height = 12)
  (H3 : a = 3) (H4 : b = 3) : 
  let area := (1 / 2) * (width / 2) * height 
  in 2 * area = 72 :=
by
  sorry

end shaded_area_l327_327755


namespace number_of_prime_pairs_for_10003_l327_327514

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l327_327514


namespace jimmy_wins_bet_l327_327819

/-- 
Condition: The fan rotates at 50 revolutions per second.
Condition: There are four half-discs (blades) located at equal distances from each other.
Condition: The blades are rotated relative to each other at some angles.
Condition: Jimmy can shoot at any moment and achieve any bullet speed.
Question: Prove that Jimmy will win the bet (shoot through all four blades with one shot).
-/
theorem jimmy_wins_bet (rotation_speed : ℝ) (blade_count : ℕ) (blade_angles : list ℝ) (shoot_time : ℝ → Prop) (bullet_speed : ℝ → ℝ) :
  rotation_speed = 50 → blade_count = 4 → 
  ∃ (trajectory : ℝ → ℝ), ∀ blade : ℕ, blade < blade_count → shoots_through (trajectory, blade)
:= by
  sorry

/- 
Additional necessary definitions for the theorem to make sense:
- shoots_through (trajectory, blade) : Prop -- A predicate indicating that the given trajectory shoots through the specified blade.
-/

end jimmy_wins_bet_l327_327819


namespace solution_is_D_l327_327792

def eqA (x : ℤ) : Prop := x + 1 = 0
def eqB (x : ℤ) : Prop := 3 * x = -3
def eqC (x : ℤ) : Prop := x - 1 = 2
def eqD (x : ℤ) : Prop := 2 * x + 2 = 4

theorem solution_is_D (x : ℤ) (h : x = 1) : eqD x :=
by {
  have : 2 * x + 2 = 4 := sorry,
  exact this,
}

end solution_is_D_l327_327792


namespace fixed_point_intersection_l327_327041

noncomputable def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1}

def point (x y : ℝ) : ℝ × ℝ := (x, y)

-- Given points
def A : ℝ × ℝ := point 2 0
def B : ℝ × ℝ := point (-10/3) (-4/3)

-- Hyperbola parameters
axiom a_pos : 4 > 0
axiom b_pos : 1 > 0

-- Hyperbola definition
def E : Set (ℝ × ℝ) := hyperbola 2 1

-- Verification of given points on the hyperbola
example : A ∈ E ∧ B ∈ E := sorry

-- The condition ∥MP∥ = ∥PQ∥ implies a specific fixed point
theorem fixed_point_intersection (M N P Q : ℝ × ℝ) 
  (l : ℝ → ℝ) :
  (∃ k m : ℝ, l = λ x, k * x + m) →
  (M ∈ E ∧ N ∈ E) →
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    M = (x1, y1) ∧ N = (x2, y2) ∧ 
    P = (x3, 0) ∧ Q = (k * x1 + (m - k * x3), m * P.snd / Q.snd)) →
  vector_sum (vector_from M P) = vector_sum (vector_from P Q) → 
  l 2 = 2 := 
sorry

end fixed_point_intersection_l327_327041


namespace number_of_ordered_pairs_l327_327431

theorem number_of_ordered_pairs :
  {p : ℤ × ℤ // (p.1^2 + p.2^2 < 25) ∧ (p.1^2 + p.2^2 < 10 * p.1) ∧ (p.1^2 + p.2^2 < 10 * p.2)}.toFinset.card = 9 :=
by sorry

end number_of_ordered_pairs_l327_327431


namespace measure_of_XY_l327_327252

variable (XYZ : Type) [IsoscelesRightTriangle XYZ] (XY YZ : ℝ) (area : ℝ)
variable (h1 : IsHypotenuse XY YZ)
variable (h2 : area = 64)

theorem measure_of_XY (area : ℝ) (h2 : area = 64) (h1 : IsHypotenuse XY YZ) : XY = 16 := 
by
  sorry

end measure_of_XY_l327_327252


namespace lambda_range_l327_327380

theorem lambda_range (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (n : ℕ) (h1 : ∀ n, a_n n = 4 * n) 
  (h2 : ∀ n, S_n n = ∑ i in Finset.range n, a_n (i + 1)) 
  (h3 : ∀ n, S_n n + 8 ≥ λ * n) : λ ≤ 10 :=
sorry

end lambda_range_l327_327380


namespace population_growth_l327_327210

theorem population_growth (P_present P_future : ℝ) (r : ℝ) (n : ℕ)
  (h1 : P_present = 7800)
  (h2 : P_future = 10860.72)
  (h3 : n = 2) :
  P_future = P_present * (1 + r / 100)^n → r = 18.03 :=
by sorry

end population_growth_l327_327210


namespace minimum_value_of_f_range_of_a_l327_327029

noncomputable def f(x : ℝ) := Real.exp x - x

theorem minimum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f(y) ≥ f(x)) ∧ f(x) = 1 := sorry

theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc (0 : ℝ) 2, f(x) ≥ a * x} = Set.Iic (Real.exp 1 - 1) := sorry

end minimum_value_of_f_range_of_a_l327_327029


namespace number_of_ways_sum_of_primes_l327_327496

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l327_327496


namespace angle_between_faces_at_AB_length_CD_l327_327199

noncomputable def volume_pyramid := (100 : ℝ) / (3 * real.sqrt 3)
noncomputable def height_D := 4
noncomputable def radius_eq (V S : ℝ) := 3 * V / S

-- Part (a)
theorem angle_between_faces_at_AB 
  (h_volume : volume_pyramid = 100 / (3 * real.sqrt 3))
  (h_height : height_D = 4)
  (h_radius_eq : radius_eq (volume_pyramid / 4) (5 * real.sqrt 3) = radius_eq (volume_pyramid / 4) (5 * real.sqrt 3)): -- condition of equal radii
  ∃ θ ∈ (real.arccos (3/5) ∪ real.arccos (-3/5)), 
    θ = real.arccos (3/5) ∨ θ = real.arccos (-3/5) := 
sorry

-- Part (b)
theorem length_CD 
  (h_perp : ℝ)
  (h_volume : volume_pyramid = 100 / (3 * real.sqrt 3)) -- using the volume of the pyramid again
  (h_height : height_D = 4): 
  ∃ l, 
    l = 8 / real.sqrt 3 ∨ l = 4 * real.sqrt 19 / real.sqrt 3 :=
sorry

end angle_between_faces_at_AB_length_CD_l327_327199


namespace no_point_in_third_quadrant_l327_327445

theorem no_point_in_third_quadrant (m : ℝ) : 
  let x := m^2 + m - 2,
      y := 6 - m - m^2 in
  ¬(x < 0 ∧ y < 0) :=
sorry

end no_point_in_third_quadrant_l327_327445


namespace find_angle_F_l327_327588

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l327_327588


namespace unique_sum_of_two_primes_l327_327558

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l327_327558


namespace savings_during_sale_l327_327599

-- Definitions for conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def normal_price_per_ball_bearing : ℕ := 1 -- dollar
def sale_price_per_ball_bearing : ℝ := 0.75 -- dollars
def bulk_discount : ℝ := 0.20

-- Statement of the theorem
theorem savings_during_sale :
  let total_ball_bearings := machines * ball_bearings_per_machine in
  let normal_cost := total_ball_bearings * normal_price_per_ball_bearing in
  let sale_cost := total_ball_bearings * sale_price_per_ball_bearing in
  let bulk_discount_amount := sale_cost * bulk_discount in
  let final_price := sale_cost - bulk_discount_amount in
  let savings := normal_cost - final_price in
  savings = 120 := 
  by
    -- proof to be filled in
    sorry

end savings_during_sale_l327_327599


namespace number_of_positive_integer_factors_of_G_l327_327970

noncomputable def a : ℕ → ℤ
| 1       => 20
| (n + 1) => a n ^ 2 - b n ^ 2

noncomputable def b : ℕ → ℤ
| 1       => 15
| (n + 1) => 2 * a n * b n - b n ^ 2

noncomputable def G : ℤ :=
  a 10 ^ 2 - a 10 * b 10 + b 10 ^ 2

/-- The number of positive integer factors of G is 525825. -/
theorem number_of_positive_integer_factors_of_G :
  nat.num_divisors G = 525825 := 
sorry

end number_of_positive_integer_factors_of_G_l327_327970


namespace sodium_hypochlorite_formed_l327_327349

-- Define the given conditions
def sodium_hydroxide_moles : ℕ := 4
def chlorine_moles : ℕ := 2

-- Define the stoichiometry of the reaction (2 NaOH + Cl2 → NaOCl + NaCl + H2O)
lemma balanced_reaction (naoh cl2 naocl : ℕ) : 2 * naoh = 2 * naocl ∧ cl2 = naocl :=
  by sorry

-- Prove that the amount of NaOCl formed is 2 moles given the conditions
theorem sodium_hypochlorite_formed :
  sodium_hydroxide_moles = 4 → chlorine_moles = 2 → ∃ (naocl : ℕ), naocl = 2 :=
  by
    intros hnaoh hcl2
    use 2
    have h := balanced_reaction 2 2 2
    sorry

end sodium_hypochlorite_formed_l327_327349


namespace remaining_water_in_bathtub_l327_327944

theorem remaining_water_in_bathtub : 
  ∀ (dripping_rate : ℕ) (evaporation_rate : ℕ) (duration_hr : ℕ) (dumped_out_liters : ℕ), 
    dripping_rate = 40 →
    evaporation_rate = 200 →
    duration_hr = 9 →
    dumped_out_liters = 12 →
    let total_dripped_in_ml := dripping_rate * 60 * duration_hr in
    let total_evaporated_in_ml := evaporation_rate * duration_hr in
    let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml in
    let dumped_out_in_ml := dumped_out_liters * 1000 in
    net_water_in_ml - dumped_out_in_ml = 7800 :=
by
  intros dripping_rate evaporation_rate duration_hr dumped_out_liters
  intros rate_eq evap_eq duration_eq dump_eq
  simp [rate_eq, evap_eq, duration_eq, dump_eq]
  let total_dripped_in_ml := 40 * 60 * 9
  let total_evaporated_in_ml := 200 * 9
  let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml
  let dumped_out_in_ml := 12 * 1000
  simp [net_water_in_ml, dumped_out_in_ml]
  sorry

end remaining_water_in_bathtub_l327_327944


namespace no_prime_sum_10003_l327_327537

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l327_327537


namespace find_k_value_l327_327126

variables (A B D : ℝ × ℝ) (k : ℝ)

-- Defining the coordinates of A and B as per the problem
def A := (-3, 1)
def B := (4, 1)

-- Condition: Rectangle ABCD has an area of 70
def rectangle_area : ℝ := 70

-- Condition: k is positive
def k_positive : Prop := k > 0

-- Compute the length of side AB
def length_AB : ℝ := abs (B.1 - A.1)

-- Compute the width of the rectangle
def width_AD (area : ℝ) (AB_length : ℝ) : ℝ := area / AB_length

-- Deduce k from the coordinates and width
def compute_k (A_y : ℝ) (AD_length : ℝ) : ℝ := A_y + AD_length

-- Theorem stating that given the conditions, k equals 11
theorem find_k_value (h1 : length_AB = 7) (h2 : rectangle_area = 70) (h3 : A.snd = 1) (h4 : k_positive) :
    k = 11 :=
by
  -- Introducing needed definitions
  let AB := length_AB
  let AD := width_AD rectangle_area AB
  let k_val := compute_k A.snd AD
  -- Skipping proof
  sorry

end find_k_value_l327_327126
