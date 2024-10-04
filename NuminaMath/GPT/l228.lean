import Mathlib

namespace find_ab_l228_228235

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x - 1) = 7) ∧ (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x + 1) = 9) →
  (a, b) = (3, -2) := 
by
  sorry

end find_ab_l228_228235


namespace birds_initial_count_l228_228213

theorem birds_initial_count (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end birds_initial_count_l228_228213


namespace correct_expression_l228_228632

theorem correct_expression (a b c d : ℝ) : 
  (\sqrt{36} ≠ ± 6) → 
  (\sqrt{(-3)^2} ≠ -3) → 
  (\sqrt{-4} = complex.I * sqrt 4) → 
  (\sqrt[3]{-8} = -2) → 
  d = -2 :=
by 
  intros h1 h2 h3 h4
  exact h4

end correct_expression_l228_228632


namespace cos_C_eq_sin_B_l228_228148

noncomputable def triangle_ABC := Type

variables (A B C : triangle_ABC) [is_right_triangle : angle A = 90]

def sin_B : ℝ := 3 / 5

theorem cos_C_eq_sin_B : ∀ (A B C : triangle_ABC), 
  (angle A = 90) ∧ (sin (angle B) = 3 / 5) → (cos (angle C) = 3 / 5) :=
by
  sorry

end cos_C_eq_sin_B_l228_228148


namespace large_cube_side_length_l228_228158

theorem large_cube_side_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 6) (h3 : s3 = 8) : 
  ∃ s_large : ℝ, s_large^3 = s1^3 + s2^3 + s3^3 ∧ s_large = 9 := 
by 
  use 9
  rw [h1, h2, h3]
  norm_num

end large_cube_side_length_l228_228158


namespace spiderCanEatAllFlies_l228_228307

-- Define the number of nodes in the grid.
def numNodes := 100

-- Define initial conditions.
def cornerStart := true
def numFlies := 100
def fliesAtNodes (nodes : ℕ) : Prop := nodes = numFlies

-- Define the predicate for whether the spider can eat all flies within a certain number of moves.
def canEatAllFliesWithinMoves (maxMoves : ℕ) : Prop :=
  ∃ (moves : ℕ), moves ≤ maxMoves

-- The theorem we need to prove in Lean 4.
theorem spiderCanEatAllFlies (h1 : cornerStart) (h2 : fliesAtNodes numFlies) : canEatAllFliesWithinMoves 2000 :=
by
  sorry

end spiderCanEatAllFlies_l228_228307


namespace decreasing_function_inequality_l228_228044

theorem decreasing_function_inequality (f : ℝ → ℝ) (hf : ∀ x y, x < y → f(x) > f(y)) (m n : ℝ) (h : f(m) - f(n) > f(-m) - f(-n)) : m - n < 0 :=
sorry

end decreasing_function_inequality_l228_228044


namespace integer_pair_condition_l228_228832

theorem integer_pair_condition (m n : ℤ) (h : (m^2 + m * n + n^2 : ℚ) / (m + 2 * n) = 13 / 3) : m + 2 * n = 9 :=
sorry

end integer_pair_condition_l228_228832


namespace range_of_m_l228_228810

variable {ℝ : Type} [LinearOrderedField ℝ]

def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : ∀ x : ℝ, p x → q x m) :
    9 ≤ m :=
by
  sorry

end range_of_m_l228_228810


namespace pizza_slices_meat_count_l228_228981

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l228_228981


namespace AB_different_groups_probability_l228_228924

-- Define the probability calculation
def probability_AB_different_groups : ℚ :=
  4 / 5

theorem AB_different_groups_probability :
  ∀ (groups : Finset (Finset ℕ)), (groups.card = 3) →
  (∀ g ∈ groups, g.card = 2) →
  (0 ∈ groups.to_finset.join) →
  (1 ∈ groups.to_finset.join) →
  (probability_AB_different_groups = 4 / 5) :=
by
  intros groups h1 h2 h3 h4
  sorry

end AB_different_groups_probability_l228_228924


namespace linear_eq_solution_l228_228058

-- Given problem conditions translated to Lean definitions
variable (a : ℝ) (x : ℝ)
lemma linear_eq_condition : (a - 3) * x ^ (|a| - 2) + 6 = 0
lemma linear_eq_property : |a| - 2 = 1

-- Goal: proving the solution
theorem linear_eq_solution : x = 1 :=
by 
  sorry -- Proof omitted

end linear_eq_solution_l228_228058


namespace sum_1026_is_2008_l228_228086

def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let groups_sum : ℕ := (n * n)
    let extra_2s := (2008 - groups_sum) / 2
    (n * (n + 1)) / 2 + extra_2s

theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 :=
  sorry

end sum_1026_is_2008_l228_228086


namespace angle_BAC_64_l228_228139

open EuclideanGeometry

-- Assumptions
variables {A B C H : Point}
variable  (ABC : Triangle A B C)
variable [AcuteABC : AcuteTriangle ABC]

-- Altitude BH from B to AC
variable (BH : Altitude B (Line A C))

-- Condition: CH = AB + AH
variable (CH_eq : dist (C, H) = dist (A, B) + dist (A, H))

-- Given angle ABC
variable (angle_ABC : ang (B, A, C) = 84)

-- Proving angle BAC=64 degrees
theorem angle_BAC_64 :
  ang (A, B, C) + ang (B, A, C) = 64 := sorry

end angle_BAC_64_l228_228139


namespace multiple_with_digits_l228_228558

theorem multiple_with_digits (n : ℕ) (h : n > 0) :
  ∃ (m : ℕ), (m % n = 0) ∧ (m < 10 ^ n) ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) :=
by
  sorry

end multiple_with_digits_l228_228558


namespace magnitude_of_z_l228_228386

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := (1 + 2 * i) / i

theorem magnitude_of_z : complex.abs z = real.sqrt 5 := 
by {
  sorry
}

end magnitude_of_z_l228_228386


namespace coefficient_of_x3_in_expansion_l228_228124

theorem coefficient_of_x3_in_expansion :
  (∑ k in Finset.filter (λ k => (k % 2 = 1)) (Finset.range (n + 1)), Nat.choose n k) = 16 →
  (n = 5 → 
  (∀ r, (5 - 2 * r = 3) → 
  (∃ k : ℤ, (binomial 5 r) * (-2 : ℤ) ^ r = k ∧ x^(5 - 2*r) ≠ 0)
  → (by existsi (-10 : ℤ), sorry))) :=
sorry

end coefficient_of_x3_in_expansion_l228_228124


namespace problem_conditions_general_formula_a_n_find_m_l228_228526

noncomputable def a_n (n : ℕ) := 6 * n - 5
noncomputable def S_n (n : ℕ) := 3 * n ^ 2 - 2 * n
noncomputable def b_n (n : ℕ) := 3 / (a_n n * a_n (n + 1))
noncomputable def T_n (n : ℕ) := (Finset.range n).sum b_n

theorem problem_conditions (n : ℕ) (hn: n > 0) :
  let S_n' := S_n n in
  (∀ n, (n > 0) → (S_n' = 3 * n^2 - 2 * n) ∧
  (a_n 1 = 1) ∧ 
  (∀ n, n > 0 → a_n n = 6 * n - 5) ∧
  (∀ n > 0, b_n n = 3 / (a_n n * a_n (n + 1))) ∧
  (∀ n, n > 0 → (n, S_n' / n).2 = 3 * n - 2)) :=
by
  sorry

theorem general_formula_a_n (n : ℕ) (hn: n > 0) : a_n n = 6 * n - 5 :=
by 
  sorry

theorem find_m (m : ℕ) (n : ℕ) (hn: n > 0) : T_n n < m / 20 → m = 10 :=
by 
  sorry

end problem_conditions_general_formula_a_n_find_m_l228_228526


namespace find_m_and_M_sum_l228_228522

noncomputable def m_and_M_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) (h5 : x^2 + y^2 + z^2 = 5) : ℝ :=
  let m := 1/3
  let M := 1 
  m + M

theorem find_m_and_M_sum :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 → x^2 + y^2 + z^2 = 5 → 
  m_and_M_sum x y z (by assumption) (by assumption) (by assumption) (by assumption) (by assumption) = 4/3 :=
by sorry

end find_m_and_M_sum_l228_228522


namespace units_digit_of_5_to_4_l228_228261

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_5_to_4 : units_digit (5^4) = 5 := by
  -- The definition ensures that 5^4 = 625 and the units digit is 5
  sorry

end units_digit_of_5_to_4_l228_228261


namespace trapezoid_segment_count_l228_228988

def trapezoid_segment_length (OK RA : ℕ) (h_parallel : OK = 12) :=
  ∃ n : ℕ, ∀ x : ℕ, 0 < x → PQ_length(OK, RA, x) = n

/- Goal: Prove the number of integer segment lengths parallel to OK through the intersection of diagonals is 10 -/
theorem trapezoid_segment_count :
  let OK := 12 in
  let valid_segment_lengths := {l : ℕ | ∃ (RA : ℕ), RA > 0 ∧ PQ_length(OK, RA, l)} in
  valid_segment_lengths.card = 10 :=
sorry

/- Definition of PQ_length used in the above theorem, based on the provided conditions and similarity from the proof -/
def PQ_length (OK RA segment_length : ℕ) : ℕ :=
  2 * (OK * RA) / (OK + RA)

end trapezoid_segment_count_l228_228988


namespace tan_beta_value_l228_228037

noncomputable theory

open Real

theorem tan_beta_value {α β : ℝ} (h1 : cos (α + β) = -1) (h2 : tan α = 2) : tan β = -2 :=
by
  -- Introducing the conditions.
  have h : tan (α + β) = 0 := by
    have h_cos := h1
    -- Use trigonometric identities and given conditions to assert the value of tan β
    sorry
  -- Using the tangent sum formula, solve for tan β given tan α = 2
  sorry

end tan_beta_value_l228_228037


namespace myrtle_hens_l228_228197

/-- Myrtle has some hens that lay 3 eggs a day. She was gone for 7 days and told her neighbor 
    to take as many as they would like. The neighbor took 12 eggs. Once home, Myrtle collected 
    the remaining eggs, dropping 5 on the way into her house. Myrtle has 46 eggs. Prove 
    that Myrtle has 3 hens. -/
theorem myrtle_hens (eggs_per_hen_per_day hens days neighbor_took dropped remaining_hens_eggs : ℕ) 
    (h1 : eggs_per_hen_per_day = 3) 
    (h2 : days = 7) 
    (h3 : neighbor_took = 12) 
    (h4 : dropped = 5) 
    (h5 : remaining_hens_eggs = 46) : 
    hens = 3 := 
by 
  sorry

end myrtle_hens_l228_228197


namespace find_B_intersection_point_l228_228585

theorem find_B_intersection_point (k1 k2 : ℝ) (hA1 : 1 ≠ 0) 
  (hA2 : k1 = -2) (hA3 : k2 = -2) : 
  (-1, 2) ∈ {p : ℝ × ℝ | ∃ k1 k2, p.2 = k1 * p.1 ∧ p.2 = k2 / p.1} :=
sorry

end find_B_intersection_point_l228_228585


namespace find_z_l228_228483

theorem find_z (x y z : ℚ) (h1 : x / (y + 1) = 4 / 5) (h2 : 3 * z = 2 * x + y) (h3 : y = 10) : 
  z = 46 / 5 := 
sorry

end find_z_l228_228483


namespace lucy_fraction_of_edna_distance_l228_228986

-- Defining the conditions
def distance_mary_ran (total_distance : ℝ) : ℝ :=
  (3 / 8) * total_distance

def distance_edna_ran (distance_mary : ℝ) : ℝ :=
  (2 / 3) * distance_mary

def distance_lucy_ran (distance_mary : ℝ) (additional_distance_needed : ℝ) : ℝ :=
  distance_mary - additional_distance_needed

-- The problem to prove
theorem lucy_fraction_of_edna_distance (total_distance : ℝ) (additional_distance_needed : ℝ) :
  let distance_mary := distance_mary_ran total_distance
  let distance_edna := distance_edna_ran distance_mary
  let distance_lucy := distance_lucy_ran distance_mary additional_distance_needed
  additional_distance_needed = 4 → total_distance = 24 → distance_lucy / distance_edna = 5 / 6 :=
by
  sorry

end lucy_fraction_of_edna_distance_l228_228986


namespace mean_inequality_l228_228565

variable (a b : ℝ)

-- Conditions: a and b are distinct and non-zero
axiom h₀ : a ≠ b
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0

theorem mean_inequality (h₀ : a ≠ b) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) :=
sorry -- Proof is not provided, only statement.

end mean_inequality_l228_228565


namespace polygon_sides_eq_eight_l228_228604

theorem polygon_sides_eq_eight (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 :=
by
  intro h
  sorry

end polygon_sides_eq_eight_l228_228604


namespace prove_divisibility_l228_228517

theorem prove_divisibility (a : ℤ) (k : ℕ) (p : ℕ) [Fact (0 < p)] [Fact (Nat.Prime p)]
  (h_pk1 : p > k + 1) 
  (h_div : (Finset.range (k + 1)).sum (λ (i : ℕ), a ^ (k - i)) ≡ 0 [ZMOD p]) :
  ((Finset.range (k + 1)).sum (λ (i : ℕ), a ^ ((k - i) * p)) ≡ 0 [ZMOD (p^2)]) :=
sorry

end prove_divisibility_l228_228517


namespace max_value_y_l228_228408

theorem max_value_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : xy = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  -- Proof steps go here
  sorry

example (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : xy = (x - y) / (x + 3 * y)) : ∃ y_max, y = 1 / 3 :=
by
  -- Proof steps go here
  sorry

end max_value_y_l228_228408


namespace sequences_and_sum_l228_228826

noncomputable def f (x : ℝ) : ℝ := x / (3 * x + 1)

def a_seq (n : ℕ) : ℝ := (1 / 4)^(n - 1)

def b_seq (n : ℕ) : ℝ := 1 / (3 * n)

def c_seq (n : ℕ) : ℝ := 3 * n * (1 / 4)^(n - 1)

def T_seq (n : ℕ) : ℝ := ∑ k in Finset.range n, c_seq (k + 1)

theorem sequences_and_sum (n : ℕ) :
  a_seq n = (1 / 4)^(n - 1) ∧
  b_seq n = 1 / (3 * n) ∧
  T_seq n = (16 / 3) - (3 * n + 4) / (3 * 4^(n - 1)) :=
by
  sorry

end sequences_and_sum_l228_228826


namespace intersection_sum_square_l228_228069

def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * cos theta, rho * sin theta)

def curve_C (x y : ℝ) : Prop := 
  x^2 + y^2 - 2*x - 4*y = 0

def line_l_parametric (t : ℝ) : ℝ × ℝ := 
  (sqrt 3 / 2 * t, 1 + 1 / 2 * t)

def standard_form_line_l (x y : ℝ) : Prop := 
  x - sqrt 3 * y + sqrt 3 = 0

def y_intercept : ℝ × ℝ := (0, 1)

noncomputable def distance_squared (p1 p2 : ℝ × ℝ) : ℝ := 
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_sum_square :
  ∃ A B : ℝ × ℝ, 
  let t1 := (-sqrt 3 - 1 + sqrt ((sqrt 3 + 1)^2 + 4*3)) / 2,
      t2 := (-sqrt 3 - 1 - sqrt ((sqrt 3 + 1)^2 + 4*3)) / 2,
      A := line_l_parametric t1,
      B := line_l_parametric t2,
      MA := distance_squared y_intercept A,
      MB := distance_squared y_intercept B
  in (MA + MB)^2 = 16 + 2 * sqrt 3 := 
sorry

end intersection_sum_square_l228_228069


namespace prove_a_in_S_l228_228088

open Set

variable {α : Type*} [DecidableEq α]

def S : Set α := {1, 2}
def T (a : α) : Set α := {a}

theorem prove_a_in_S (a : α) (h : S ∪ T a = S) : a ∈ S := by
  have h_sub : T a ⊆ S := by
    rw [union_eq_self_of_subset_left h]
  obtain ⟨a_in_S⟩ : a ∈ S := mem_singleton_iff.mp $ h_sub trivial
  exact a_in_S

end prove_a_in_S_l228_228088


namespace Jake_weekly_earnings_l228_228496

theorem Jake_weekly_earnings
  (jacob_weekday_rate : ℕ)
  (jacob_weekend_rate : ℕ)
  (jake_multiplier : ℕ)
  (workdays : ℕ)
  (weekday_hours : ℕ)
  (weekend_days : ℕ)
  (weekend_hours : ℕ)
  (jacob_weekday_rate_eq : jacob_weekday_rate = 6)
  (jacob_weekend_rate_eq : jacob_weekend_rate = 8)
  (jake_multiplier_eq : jake_multiplier = 3)
  (workdays_eq : workdays = 5)
  (weekday_hours_eq : weekday_hours = 8)
  (weekend_days_eq : weekend_days = 2)
  (weekend_hours_eq : weekend_hours = 5)
  : 
  let jake_weekday_rate := jake_multiplier * jacob_weekday_rate
  let jake_weekend_rate := jake_multiplier * jacob_weekend_rate
  let total_weekday_hours := workdays * weekday_hours
  let total_weekend_hours := weekend_days * weekend_hours
  let total_weekday_earnings := total_weekday_hours * jake_weekday_rate
  let total_weekend_earnings := total_weekend_hours * jake_weekend_rate
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  in
  total_earnings = 960 :=
by {
  -- The actual proof would go here
  sorry
}

end Jake_weekly_earnings_l228_228496


namespace area_of_regular_octagon_in_circle_l228_228677

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228677


namespace least_number_to_be_added_l228_228774

theorem least_number_to_be_added (k : ℕ) (h₁ : Nat.Prime 29) (h₂ : Nat.Prime 37) (H : Nat.gcd 29 37 = 1) : 
  (433124 + k) % Nat.lcm 29 37 = 0 → k = 578 :=
by 
  sorry

end least_number_to_be_added_l228_228774


namespace prob_three_rainy_days_is_two_fifths_l228_228987

def sets := [[9, 5, 3, 3], [9, 5, 2, 2], [0, 0, 1, 8], [7, 4, 7, 2], [0, 0, 1, 8], 
             [3, 8, 7, 9], [5, 8, 6, 9], [3, 1, 8, 1], [7, 8, 9, 0], [2, 6, 9, 2],
             [8, 2, 8, 0], [8, 4, 2, 5], [3, 9, 9, 0], [8, 4, 6, 0], [7, 9, 8, 0],
             [2, 4, 3, 6], [5, 9, 8, 7], [3, 8, 8, 2], [0, 7, 5, 3], [8, 9, 3, 5]]

def is_rainy_day (n : ℕ) : Prop := n ≤ 5

def count_rainy_days (numbers : List ℕ) : ℕ :=
  numbers.countp is_rainy_day

def valid_sets (sets : List (List ℕ)) : List (List ℕ) :=
  sets.filter (λ set => count_rainy_days set = 3)

def prob_three_rainy_days (sets : List (List ℕ)) : ℚ :=
  valid_sets sets).length / sets.length

theorem prob_three_rainy_days_is_two_fifths : 
  prob_three_rainy_days sets = 2 / 5 := 
by sorry

end prob_three_rainy_days_is_two_fifths_l228_228987


namespace area_ratio_l228_228055

-- Define points A, B, C, and E as vectors in a vector space over the reals.
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C E : V)

-- Define the conditions given in the problem.
def vector_relation (A B C E : V) : Prop :=
  E = A + (1 / 2) • (B - A) + (1 / 3) • (C - A)

-- Define the areas of triangles ABE and ABC.
def area_triangle (u v : V) : ℝ :=
  (1 / 2) * (∥u∥ * ∥v∥ * real.sin (real.angle u v))

-- State the theorem that needs to be proved.
theorem area_ratio (h : vector_relation A B C E) :
  (area_triangle (B - A) (E - A)) / (area_triangle (B - A) (C - A)) = 1 / 3 :=
sorry

end area_ratio_l228_228055


namespace polar_to_rect_equiv_min_distance_from_ellipse_to_line_l228_228419

noncomputable def polar_to_rect (p θ : ℝ) : ℝ := p * sin (θ - π / 4)

def line_rect_coordinates (x y : ℝ) : Prop := x - y + 4 = 0

def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 9 = 1

noncomputable def distance_from_point_to_line (x y : ℝ) : ℝ :=
  (abs (sqrt 3 * cos (atan2 y x) - 3 * sin (atan2 y x) + 4)) / sqrt 2

theorem polar_to_rect_equiv (p θ x y : ℝ) :
  (polar_to_rect p θ = 2 * sqrt 2) ∧ (x = p * cos θ) ∧ (y = p * sin θ) ↔ line_rect_coordinates x y :=
by sorry

theorem min_distance_from_ellipse_to_line :
  (∀ (x y : ℝ), ellipse x y → ∃ d, d = distance_from_point_to_line x y) →
  ∃ d, d = 2 * sqrt 2 - sqrt 6 :=
by sorry

end polar_to_rect_equiv_min_distance_from_ellipse_to_line_l228_228419


namespace scalene_triangle_angle_bisectors_inequality_l228_228550

theorem scalene_triangle_angle_bisectors_inequality
  (a b c l1 l2 S: ℝ)
  (h_scalene: a > b ∧ b > c ∧ c > 0)
  (h_l1: l1 = max (angle_bisector_length a b c 2 * atan 1) (angle_bisector_length b c a 2 * sin (angle b c a)))
  (h_l2: l2 = min (angle_bisector_length a b c 2 * atan 1) (angle_bisector_length b c a 2 * sin (angle b c a)))
  (h_S: S = 1 / 2 * b * c * sin (2 * atan 1))
  : l1^2 > sqrt 3 * S ∧ sqrt 3 * S > l2^2 :=
by
  sorry

end scalene_triangle_angle_bisectors_inequality_l228_228550


namespace option_A_option_B_option_D_l228_228379

theorem option_A (a b : ℝ) (h : a > b) : a > (a + b) / 2 ∧ (a + b) / 2 > b :=
by
  have h1 : (a + b) / 2 < a := by
    rw [div_lt_iff (by norm_num : (2 : ℝ) > 0)]
    linarith only [h, (by norm_num : (2 : ℝ) = 1 + 1)]
  have h2 : b < (a + b) / 2 := by
    rw [lt_div_iff (by norm_num : (2 : ℝ) > 0)]
    linarith only [h, (by norm_num : (1 + 1 : ℝ) = 2)]
  exact ⟨h1, h2⟩

theorem option_B (a b : ℝ) (h : a > b) (hb : b > 0) : a > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > b :=
by
  have h1 : a > Real.sqrt (a * b) := by
    apply Real.sqrt_ltReal.mpr
    apply mul_lt_mul_of_pos_right h hb
  have h2 : Real.sqrt (a * b) > b := by
    rw [Real.sqrt_lt_iff]
    split
    · exact mul_pos h hb
    · exact mul_lt_mul_of_pos_right h (Real.sqrt_pos.mpr hb)
  exact ⟨h1, h2⟩

theorem option_D (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) : (b + c) / (a + c) > b / a :=
by
  have h4 : (b + c) * a > b * a := by
    apply mul_lt_mul_of_pos_right <| by linarith only [h3]
    exact (by linarith only [h1, h2])
  have h5 : a * (a + c) ≠ 0 := by linarith only [h2, h3]
  have h6 : (b + c) / (a + c) - b / a = (c * (a - b)) / (a * (a + c)) := by
    field_simp only [h5]
    ring
  linarith only [h4, h5]

example : option_A ∧ option_B ∧ option_D := by
  exact ⟨option_A, option_B, option_D⟩

example : ∃ (a b : ℝ), 1 / a > 1 / b → ¬ (a > 0 ∧ b < 0) :=
by
  use [2, 1]
  intros h h'
  linarith only [h']

end option_A_option_B_option_D_l228_228379


namespace sequence_form_l228_228173

theorem sequence_form (c : ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, 0 < n →
    (∃! i : ℕ, 0 < i ∧ a i ≤ a (n + 1) + c)) ↔
  (∀ n : ℕ, 0 < n → a n = n + (c + 1)) :=
by
  sorry

end sequence_form_l228_228173


namespace find_m_l228_228445

variable (m : ℝ)
def a := (5, m)
def b := (2, -2)
def a_plus_b := (7, m - 2)

theorem find_m (h : (7, m - 2) • (2, -2) = 0) : m = 9 := 
  by sorry

end find_m_l228_228445


namespace zeroes_at_end_base_8_of_factorial_15_l228_228885

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228885


namespace range_of_m_l228_228834

theorem range_of_m 
  (m : ℝ)
  (h1 : ∃ a b : ℝ, a ≠ b ∧ (x^2 - (4 * m + 1) * x + (2 * m - 1) = 0) ∧ a > 2 ∧ b < 2)
  (h2 : (2 * m - 1) < -1 / 2) :
  1 / 6 < m ∧ m < 1 / 4 :=
begin
  sorry,
end

end range_of_m_l228_228834


namespace sum_y_coordinates_of_other_vertices_l228_228106

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

theorem sum_y_coordinates_of_other_vertices (P Q : Point) : 
  P = { x := 4, y := 22 } → Q = { x := 12, y := -8 } → 
  2 * (midpoint P Q).y = 14 := 
by
  sorry

end sum_y_coordinates_of_other_vertices_l228_228106


namespace ellipse_equation_and_fixed_point_l228_228138

theorem ellipse_equation_and_fixed_point 
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (ab_ineq : a > b) 
  (ecc : real.sqrt 6 / 3 = (real.sqrt (a^2 - b^2)) / a)
  (M : ℝ × ℝ) (M_coords : M = (0, 1)) :
  (∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 ^ 2 / 3) + (p.2 ^ 2) = 1)) ∧
  (∀ l : ℝ → ℝ, (∀ P Q : ℝ × ℝ, (P.1 ^ 2 / 3) + (P.2 ^ 2) = 1 ∧ (Q.1 ^ 2 / 3) + (Q.2 ^ 2) = 1 ∧ l(P.1) = P.2 ∧ l(Q.1) = Q.2) →
    ∃ fixed_pt : ℝ × ℝ, fixed_pt = (0, -1/2) ∧ l(fixed_pt.1) = fixed_pt.2) :=
sorry

end ellipse_equation_and_fixed_point_l228_228138


namespace num_ways_arrange_l228_228168

open Finset

def valid_combinations : Finset (Finset Nat) :=
  { {2, 5, 11, 3}, {3, 5, 6, 2}, {3, 6, 11, 5}, {5, 6, 11, 2} }

theorem num_ways_arrange : valid_combinations.card = 4 :=
  by
    sorry  -- proof of the statement

end num_ways_arrange_l228_228168


namespace parallel_lines_k_l228_228189

theorem parallel_lines_k : 
  ∀ (k : ℝ), (∀ x y, k * x - y + 1 = 0) ∧ (∀ x y, x - k * y + 1 = 0) →
  (k = -1) :=
begin
  sorry -- proof to be provided
end

end parallel_lines_k_l228_228189


namespace worker_original_daily_wage_l228_228316

-- Given Conditions
def increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]
def new_total_weekly_salary : ℝ := 1457

-- Define the sum of the weekly increases
def total_increase : ℝ := (1 + increases.get! 0) + (1 + increases.get! 1) + (1 + increases.get! 2) + (1 + increases.get! 3) + (1 + increases.get! 4)

-- Main Theorem
theorem worker_original_daily_wage : ∀ (W : ℝ), total_increase * W = new_total_weekly_salary → W = 242.83 :=
by
  intro W h
  sorry

end worker_original_daily_wage_l228_228316


namespace perpendicular_bisector_fixed_point_l228_228169

variable {α : Type*} [NormedLinearOrderedAddCommGroup α] [NormedSpace ℝ α]

/-- Given a triangle ABC, with points M and N on sides AB and AC respectively, such that AM * MB = AN * NC, 
    the perpendicular bisector of MN passes through a fixed point. -/
theorem perpendicular_bisector_fixed_point 
  (A B C : α) 
  (M : α) (hM : M ∈ segment ℝ A B) 
  (N : α) (hN : N ∈ segment ℝ A C) 
  (h_ne_M : M ≠ A ∨ M ≠ B) 
  (h_ne_N : N ≠ A ∨ N ≠ C) 
  (h_eq : dist A M * dist M B = dist A N * dist N C) : 
  ∃ (P : α), ∀ (X : α), perpendicular_bisector ℝ M N X → X = P :=
sorry

end perpendicular_bisector_fixed_point_l228_228169


namespace cos_B_is_2_sqrt_13_over_13_l228_228135

theorem cos_B_is_2_sqrt_13_over_13 (A B C : Type) [RightAngleTriangle A B C] (hA : ∠A = 90)
    (hAB : AB = 16) (hBC : BC = 24) : 
    cos B = 2 * sqrt(13) / 13 :=
by
  sorry

end cos_B_is_2_sqrt_13_over_13_l228_228135


namespace octagon_area_correct_l228_228693

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228693


namespace divide_remaining_square_l228_228794

theorem divide_remaining_square :
  ∃ (parts : list (set (ℝ × ℝ))), 
    let total_area := 16 in 
    let removed_area := 4 in
    let remaining_area := total_area - removed_area in
    ∀ part ∈ parts, 
      ∃ (shape : set (ℝ × ℝ)), 
      measure shape = 3 ∧ 
      (∀ x y : ℝ, (x, y) ∈ shape ↔ true) ∧
      parts.length = 4 ∧ 
      measure (⋃₀ parts) = remaining_area :=
begin
  sorry
end

end divide_remaining_square_l228_228794


namespace max_value_x_plus_2y_l228_228188

variable (x y : ℝ)
variable (h1 : 4 * x + 3 * y ≤ 12)
variable (h2 : 3 * x + 6 * y ≤ 9)

theorem max_value_x_plus_2y : x + 2 * y ≤ 3 := by
  sorry

end max_value_x_plus_2y_l228_228188


namespace bob_bakes_pie_in_6_minutes_l228_228729

theorem bob_bakes_pie_in_6_minutes (x : ℕ) (h_alice : 60 / 5 = 12)
  (h_condition : 12 - 2 = 60 / x) : x = 6 :=
sorry

end bob_bakes_pie_in_6_minutes_l228_228729


namespace regular_octagon_area_l228_228684

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228684


namespace range_of_m_l228_228046

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) :
  (∀ x ∈ Ioo 1 3, x^3 - m * x^2 - 4 > 0) →
  m ≤ -3 :=
by
  sorry

end range_of_m_l228_228046


namespace sector_area_l228_228222

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) : 
    (1/2 * r^2 * θ) = Real.pi :=
by
  sorry

end sector_area_l228_228222


namespace regular_octagon_area_l228_228705

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228705


namespace complement_of_A_l228_228441

noncomputable def U := Set.univ : Set ℝ
noncomputable def A := {x : ℝ | x < 2}

theorem complement_of_A :
  (U \ A) = {x : ℝ | x ≥ 2} :=
by { sorry }

end complement_of_A_l228_228441


namespace product_increased_l228_228916

theorem product_increased (a b c : ℕ) (h1 : a = 1) (h2: b = 1) (h3: c = 676) :
  ((a - 3) * (b - 3) * (c - 3) = a * b * c + 2016) :=
by
  simp [h1, h2, h3]
  sorry

end product_increased_l228_228916


namespace first_day_bacteria_exceeds_200_l228_228914

noncomputable def N : ℕ → ℕ := λ n => 5 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, N n > 200 ∧ ∀ m : ℕ, m < n → N m ≤ 200 :=
by
  sorry

end first_day_bacteria_exceeds_200_l228_228914


namespace num_ways_to_make_20_USD_l228_228004

theorem num_ways_to_make_20_USD (q h : ℕ) (hq : q > 0) (hh : h > 0) :
  25 * q + 50 * h = 2000 ↔ ∃ n, n = 39 :=
begin
  sorry
end

end num_ways_to_make_20_USD_l228_228004


namespace union_M_N_l228_228839

def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} := 
sorry

end union_M_N_l228_228839


namespace diameter_percentage_l228_228901

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.16 * π * (d_S / 2)^2) :
  (d_R / d_S) * 100 = 40 :=
by {
  sorry
}

end diameter_percentage_l228_228901


namespace melissa_total_commission_l228_228532

def sale_price_coupe : ℝ := 30000
def sale_price_suv : ℝ := 2 * sale_price_coupe
def sale_price_luxury_sedan : ℝ := 80000

def commission_rate_coupe_and_suv : ℝ := 0.02
def commission_rate_luxury_sedan : ℝ := 0.03

def commission (rate : ℝ) (price : ℝ) : ℝ := rate * price

def total_commission : ℝ :=
  commission commission_rate_coupe_and_suv sale_price_coupe +
  commission commission_rate_coupe_and_suv sale_price_suv +
  commission commission_rate_luxury_sedan sale_price_luxury_sedan

theorem melissa_total_commission :
  total_commission = 4200 := by
  sorry

end melissa_total_commission_l228_228532


namespace factorial_base_8_zeroes_l228_228855

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228855


namespace regular_octagon_area_l228_228706

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228706


namespace polynomial_divisibility_l228_228205

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (h_pos : 0 < n) :
  ∃ Q : Polynomial ℝ, (P * P + Q * Q) % (X * X + 1)^n = 0 :=
sorry

end polynomial_divisibility_l228_228205


namespace regular_octagon_area_l228_228685

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228685


namespace smallest_n_satisfies_conditions_l228_228239

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
n % 10

def digit_sum : ℕ → ℕ
| 0       := 0
| (n + 1) := (n + 1) % 10 + digit_sum ((n + 1) / 10)

def is_multiple_of (n m : ℕ) : Prop :=
n % m = 0

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), tens_digit n = 4 ∧ units_digit n = 2 ∧ digit_sum n = 42 ∧ is_multiple_of n 42 ∧ n = 2979942 :=
sorry

end smallest_n_satisfies_conditions_l228_228239


namespace hyperbola_asymptotes_l228_228392

/-!
# Hyperbola Asymptotes

Given a hyperbola of the form $\frac{x^2}{a^2} - \frac{y^2}{b^2}=1$ with conditions:
- \(a > 0\)
- \(b > 0\)
- Imaginary axis length is \(2\)
- Focal distance is \(2\sqrt{3}\)

Prove that the equations of the asymptotes of the hyperbola are \(y = \pm \frac{\sqrt{2}}{2}x\).
-/

theorem hyperbola_asymptotes (a b : ℝ) (h0_a : a > 0) (h0_b : b > 0)
  (imaginary_axis_length : 2 * b = 2) (focal_distance : 2 * real.sqrt 3 = 2 * real.sqrt (a^2 + b^2)) :
  ∀ (x y : ℝ), y = x * (1 / (real.sqrt (real.sqrt (3)-1))) :=
by
  sorry

end hyperbola_asymptotes_l228_228392


namespace regular_octagon_area_l228_228686

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228686


namespace b5b9_l228_228930

-- Assuming the sequences are indexed from natural numbers starting at 1
-- a_n is an arithmetic sequence with common difference d
-- b_n is a geometric sequence
-- Given conditions
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry
def d : ℝ := sorry
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) - a n = d
axiom d_nonzero : d ≠ 0
axiom condition_arith : 2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0
axiom geometric_seq : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1
axiom b7_equals_a7 : b 7 = a 7

-- To prove
theorem b5b9 : b 5 * b 9 = 16 :=
by
  sorry

end b5b9_l228_228930


namespace candle_ratio_l228_228990

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end candle_ratio_l228_228990


namespace prob_A_not_losing_is_correct_l228_228546

def prob_A_wins := 0.4
def prob_draw := 0.2
def prob_A_not_losing := 0.6

theorem prob_A_not_losing_is_correct : prob_A_wins + prob_draw = prob_A_not_losing :=
by sorry

end prob_A_not_losing_is_correct_l228_228546


namespace square_area_in_parabola_l228_228725

-- Define the conditions and the proof statement
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21
def is_square_inscribed_in_parabola (s : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    x1 = 5 - s ∧ y1 = 0 ∧
    x2 = 5 + s ∧ y2 = 0 ∧
    parabola (5 + s) = -2 * s

theorem square_area_in_parabola : ∀ (s : ℝ), 
  is_square_inscribed_in_parabola s → 
  (2 * s) ^ 2 = 64 - 16 * real.sqrt 5 :=
by
  intro s inscribed
  sorry

end square_area_in_parabola_l228_228725


namespace geometric_sum_is_513_l228_228336

-- Conditions
def a : ℤ := 3
def r : ℤ := -2
def last_term : ℤ := 768

-- Existence of n such that a * (r ^ (n - 1)) = last_term
def exists_n : Prop := ∃ n : ℕ, a * (r ^ (n - 1)) = last_term

-- Statement of the proof problem
theorem geometric_sum_is_513 (h : exists_n) : ∑ i in (finset.range 9), (a * (r ^ i)) = 513 :=
by sorry

end geometric_sum_is_513_l228_228336


namespace parents_at_park_l228_228241

/-- A theorem that states the number of parents at the park given the conditions -/
theorem parents_at_park :
  ∃ (girls boys groupsize numgroups : ℕ), 
    girls = 14 ∧ 
    boys = 11 ∧ 
    groupsize = 25 ∧ 
    numgroups = 3 ∧ 
    let total_people := numgroups * groupsize in
    let children := girls + boys in
    let parents := total_people - children in
    parents = 50 :=
by
  sorry

end parents_at_park_l228_228241


namespace part1_polar_equation_part2_intersect_value_l228_228072

noncomputable def parametric_curve (a : ℝ) : ℝ × ℝ :=
  (2 * Real.cos a, Real.sqrt 3 * Real.sin a) 

def fixed_point_A : ℝ × ℝ := (0, Real.sqrt 3)

noncomputable def ellipse_focus_F2 : ℝ × ℝ := (1, 0)

theorem part1_polar_equation :
  ∀ (x y : ℝ), (x, y) = fixed_point_A → 
  ∃ (r θ : ℝ), r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) :=
sorry

theorem part2_intersect_value :
  ∃ (M N : ℝ × ℝ) (t₁ t₂ : ℝ), 
  let (x, y) := ellipse_focus_F2, 
  let l := (x - 1 + Real.sqrt 3 / 2 * t, y + 1 / 2 * t),
  let equation := 13 * t^2 - 12 * Real.sqrt 3 * t - 36 := 
  | t₁ + t₂ | = 12 * Real.sqrt 3 / 13
 :=
sorry

end part1_polar_equation_part2_intersect_value_l228_228072


namespace square_new_perimeter_ratio_l228_228815

theorem square_new_perimeter_ratio (s : ℕ) (h : s ≥ 1) : 
  let new_side := s + 1 in 
  let new_perimeter := 4 * new_side in
  new_perimeter / new_side = 4 := by
  sorry

end square_new_perimeter_ratio_l228_228815


namespace circle_equation_of_symmetric_center_l228_228898

open Set

noncomputable def circle_center_symmetric (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem circle_equation_of_symmetric_center :
  ∀ (r : ℝ) (center original_point : ℝ × ℝ),
    r = 1 →
    original_point = (1,0) →
    center = circle_center_symmetric original_point →
    ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ↔ x^2 + (y - 1)^2 = 1 :=
by
  intros r center original_point hr hpoint hcenter x y
  rw [hr, hpoint, hcenter]
  sorry

end circle_equation_of_symmetric_center_l228_228898


namespace triangle_at_most_one_obtuse_angle_l228_228627

theorem triangle_at_most_one_obtuse_angle :
  ∀ (A B C : ℝ), true :=
by
  -- Assuming that there exists at least two obtuse angles
  have h1 : ∃ A B C : ℝ, (A > π/2) ∧ (B > π/2) ∧ (A + B + C = π),
  sorry
  -- Using the contradiction method to prove that the assumption is false,
  -- thus proving that a triangle has at most one obtuse angle

end triangle_at_most_one_obtuse_angle_l228_228627


namespace trader_profit_percentage_l228_228275

-- Definitions for the conditions
def trader_buys_weight (indicated_weight: ℝ) : ℝ :=
  1.10 * indicated_weight

def trader_claimed_weight_to_customer (actual_weight: ℝ) : ℝ :=
  1.30 * actual_weight

-- Main theorem statement
theorem trader_profit_percentage (indicated_weight: ℝ) (actual_weight: ℝ) (claimed_weight: ℝ) :
  trader_buys_weight 1000 = 1100 →
  trader_claimed_weight_to_customer actual_weight = claimed_weight →
  claimed_weight = 1000 →
  (1000 - actual_weight) / actual_weight * 100 = 30 :=
by
  intros h1 h2 h3
  sorry

end trader_profit_percentage_l228_228275


namespace sufficient_but_not_necessary_not_necessary_l228_228645

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (|a| > 0) := by
  sorry

theorem not_necessary (a : ℝ) : |a| > 0 → ¬(a = 0) ∧ (a ≠ 0 → |a| > 0 ∧ (¬(a > 0) → (|a| > 0))) := by
  sorry

end sufficient_but_not_necessary_not_necessary_l228_228645


namespace range_of_b_l228_228245

theorem range_of_b (a b c m : ℝ) (h_ge_seq : c = b * b / a) (h_sum : a + b + c = m) (h_pos_a : a > 0) (h_pos_m : m > 0) : 
  (-m ≤ b ∧ b < 0) ∨ (0 < b ∧ b ≤ m / 3) :=
by
  sorry

end range_of_b_l228_228245


namespace parallelepiped_vectors_l228_228933

theorem parallelepiped_vectors (x y z : ℝ)
  (h1: ∀ (AB BC CC1 AC1 : ℝ), AC1 = AB + BC + CC1)
  (h2: ∀ (AB BC CC1 AC1 : ℝ), AC1 = x * AB + 2 * y * BC + 3 * z * CC1) :
  x + y + z = 11 / 6 :=
by
  -- This is where the proof would go, but as per the instruction we'll add sorry.
  sorry

end parallelepiped_vectors_l228_228933


namespace factorial_ends_with_base_8_zeroes_l228_228878

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228878


namespace find_angle_B_find_sum_a_c_find_max_area_l228_228395

-- Definitions and conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides
variables {area : ℝ} -- Area of the triangle

-- Condition: B is obtuse
def is_obtuse (B : ℝ) : Prop := B > π / 2 ∧ B < π

-- Conditions: Sine Rule and given area of triangle
def sine_rule_condition (a b : ℝ) (A B : ℝ) : Prop :=
  sqrt 3 * a = 2 * b * real.sin A

def area_condition (a c area : ℝ) (B : ℝ) : Prop :=
  area = 1 / 2 * a * c * real.sin B

-- Proof: Find measure of angle B
theorem find_angle_B
  (ha : ∀ A B C a b c, A + B + C = π)
  (h_obtuse : is_obtuse B)
  (h_sine_rule : sine_rule_condition a b A) :
  B = 2 * π / 3 :=
sorry

-- Proof: Find the value of a + c
theorem find_sum_a_c
  (h_area : area_condition a c (15 * sqrt 3 / 4) (2 * π / 3))
  (hb : b = 7)
  (h_cosine : b * b = a * a + c * c - 2 * a * c * real.cos (2 * π / 3)) :
  a + c = 8 :=
sorry

-- Proof: Find the maximum area of the triangle
theorem find_max_area
  (hb : b = 6)
  (h_cosine : ∀ a c, 36 = a * a + c * c + a * c) :
  ∃ (a c : ℝ), a = 2 * sqrt 3 ∧ c = 2 * sqrt 3 ∧ area = 3 * sqrt 3 :=
sorry

end find_angle_B_find_sum_a_c_find_max_area_l228_228395


namespace bags_total_weight_l228_228766

noncomputable def total_weight_of_bags (x y z : ℕ) : ℕ := x + y + z

theorem bags_total_weight (x y z : ℕ) (h1 : x + y = 90) (h2 : y + z = 100) (h3 : z + x = 110) :
  total_weight_of_bags x y z = 150 :=
by
  sorry

end bags_total_weight_l228_228766


namespace range_m_part1_range_m_part2_l228_228428

section Part1

def f (x : ℝ) (a : ℝ) : ℝ := (1/3 : ℝ) * x^3 + a * x^2 + 6 * x
def f_prime (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 6
-- Given f'(3)=0, we find a
def a_cond := ∃ a, f_prime 3 a = 0

theorem range_m_part1 (m : ℝ) (a : ℝ) (h : a_cond) (h1 : ∀ x, f_prime x a > 0 ∨ f_prime x a < 0) :
  m ∈ set.Icc (-∞) 0 ∪ set.Icc 3 ∞ :=
sorry

end Part1


section Part2

-- We use the same f(x) and a from Part 1
def h (x : ℝ) (m : ℝ) (a : ℝ) : ℝ := f x a + m

theorem range_m_part2 (m : ℝ) (a : ℝ) (h : a_cond)
  (h1 : ∀ x, 1 ≤ x ∧ x ≤ 3) :
  m ∈ set.Ioc (-14/3) (-9/2) :=
sorry

end Part2

end range_m_part1_range_m_part2_l228_228428


namespace total_pages_read_l228_228536

theorem total_pages_read
  (hours_per_day : ℕ := 24)
  (fraction_of_day_read : ℝ := 1 / 6)
  (novel_rate : ℕ := 21)
  (graphic_novel_rate : ℕ := 30)
  (comic_book_rate : ℕ := 45)
  (fraction_of_reading_time_per_type : ℝ := 1 / 3) :
  let total_reading_time := (hours_per_day : ℝ) * fraction_of_day_read;
  let reading_time_per_type := total_reading_time * fraction_of_reading_time_per_type;
  let pages_novel := novel_rate * reading_time_per_type;
  let pages_graphic_novel := graphic_novel_rate * reading_time_per_type;
  let pages_comic_book := comic_book_rate * reading_time_per_type;
  pages_novel + pages_graphic_novel + pages_comic_book = 128 :=
by
  let total_reading_time := 24 * (1 / 6)
  let reading_time_per_type := total_reading_time * (1 / 3)
  let pages_novel := 21 * reading_time_per_type
  let pages_graphic_novel := 30 * reading_time_per_type
  let pages_comic_book := 45 * reading_time_per_type
  have h1 : total_reading_time = 4 := by norm_num
  have h2 : reading_time_per_type = 4 / 3 := by norm_num
  have h3 : pages_novel = 28 := by norm_num
  have h4 : pages_graphic_novel = 40 := by norm_num
  have h5 : pages_comic_book = 60 := by norm_num
  calc
    pages_novel + pages_graphic_novel + pages_comic_book
        = 28 + 40 + 60 : by rw [h3, h4, h5]
    ... = 128 : by norm_num

end total_pages_read_l228_228536


namespace find_xz_l228_228895

theorem find_xz (x y z : ℝ) (h1 : 2 * x + z = 15) (h2 : x - 2 * y = 8) : x + z = 15 :=
sorry

end find_xz_l228_228895


namespace zeroes_at_end_base_8_of_factorial_15_l228_228882

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228882


namespace factorial_trailing_zeros_base_8_l228_228869

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228869


namespace probability_guizhou_visit_is_9_div_14_l228_228240

noncomputable def probability_guizhou_visit : ℚ :=
  (Nat.choose 5 1 * Nat.choose 3 1 + Nat.choose 3 2) / Nat.choose 8 2

theorem probability_guizhou_visit_is_9_div_14 :
  probability_guizhou_visit = 9 / 14 :=
by
  -- Skipping the proof steps, but they must show the combinatorial calculations
  sorry

end probability_guizhou_visit_is_9_div_14_l228_228240


namespace length_of_AB_l228_228147

theorem length_of_AB
  (height h : ℝ)
  (AB CD : ℝ)
  (ratio_AB_ADC : (1/2 * AB * h) / (1/2 * CD * h) = 5/4)
  (sum_AB_CD : AB + CD = 300) :
  AB = 166.67 :=
by
  -- The proof goes here.
  sorry

end length_of_AB_l228_228147


namespace integer_pairs_eq_pow_l228_228015

open Int

theorem integer_pairs_eq_pow (a b : ℤ) : a ^ b = b ^ a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := by
  sorry

end integer_pairs_eq_pow_l228_228015


namespace cos_sum_value_l228_228404

theorem cos_sum_value (α : ℝ) (h1: sin α = 3/5) (h2 : α ∈ Ioo (-π/2) (π/2)) :
  cos (α + 5/4 * π) = - (Real.sqrt 2) / 10 :=
by sorry

end cos_sum_value_l228_228404


namespace parallel_lines_solution_l228_228381

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (x + a * y + 6 = 0) → (a - 2) * x + 3 * y + 2 * a = 0) → (a = -1) :=
by
  intro h
  -- Add more formal argument insights if needed
  sorry

end parallel_lines_solution_l228_228381


namespace car_first_hour_speed_l228_228601

theorem car_first_hour_speed
  (x speed2 : ℝ)
  (avgSpeed : ℝ)
  (h_speed2 : speed2 = 60)
  (h_avgSpeed : avgSpeed = 35) :
  (avgSpeed = (x + speed2) / 2) → x = 10 :=
by
  sorry

end car_first_hour_speed_l228_228601


namespace pizza_slices_meat_count_l228_228983

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l228_228983


namespace possible_value_of_f_one_l228_228564

noncomputable def f : ℝ → ℝ := sorry

theorem possible_value_of_f_one (D : set ℝ) (hD : finset D) (h1 : 1 ∈ D) 
  (h_rotate : ∀ x ∈ D, f (cos(π / 6) * x - sin(π / 6) * f x) = sin(π / 6) * x + cos(π / 6) * f x) :
  f 1 = sqrt 3 / 2 :=
sorry

end possible_value_of_f_one_l228_228564


namespace sufficient_balance_after_29_months_l228_228641

noncomputable def accumulated_sum (S0 : ℕ) (D : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  S0 * (1 + r)^n + D * ((1 + r)^n - 1) / r

theorem sufficient_balance_after_29_months :
  let S0 := 300000
  let D := 15000
  let r := (1 / 100 : ℚ) -- interest rate of 1%
  accumulated_sum S0 D r 29 ≥ 900000 :=
by
  sorry -- The proof will be elaborated later

end sufficient_balance_after_29_months_l228_228641


namespace problem1_problem2_l228_228751

-- Problem 1: (x + 2y)^2 - (-2xy^2)^2 / xy^3 = x^2 + 4y^2
theorem problem1 (x y : ℝ) : (x + 2y)^2 - ((-2 * x * y^2)^2 / (x * y^3)) = x^2 + 4y^2 :=
by
  sorry

-- Problem 2: (x-1)/(x-3) * (2 - x + 2/(x-1)) = -x
theorem problem2 (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 1) : (x - 1) / (x - 3) * (2 - x + 2 / (x - 1)) = -x :=
by
  sorry

end problem1_problem2_l228_228751


namespace positive_difference_of_solutions_l228_228253

theorem positive_difference_of_solutions : 
    (∀ x : ℝ, |x + 3| = 15 → (x = 12 ∨ x = -18)) → 
    (abs (12 - (-18)) = 30) :=
begin
  intros,
  sorry
end

end positive_difference_of_solutions_l228_228253


namespace smallest_period_monotonic_interval_max_area_ABC_l228_228396

variables {A B C a b c : ℝ}
variables (m n : ℝ × ℝ)

-- Condition that vectors m and n are perpendicular
def m := (2 * Real.sin B, Real.sqrt 3)
def n := (2 * Real.cos (B / 2) ^ 2 - 1, Real.cos (2 * B))
def perpendicular := m.1 * n.1 + m.2 * n.2 = 0

-- Given an acute triangle ABC and b = 4
def acute_triangle_ABC := A + B + C = Real.pi ∧ (A < Real.pi / 2) ∧ (B < Real.pi / 2) ∧ (C < Real.pi / 2)
def side_b := b = 4

-- (1) Smallest positive period and monotonically increasing interval
def smallest_period_f (f : ℝ → ℝ) : ℝ := sorry
def monotonically_increasing_interval (f : ℝ → ℝ) (k : ℤ) : Set ℝ := sorry
theorem smallest_period_monotonic_interval :
  ∀ B : ℝ, perpendicular → (smallest_period_f (λ x => Real.sin (2 * x - B)) = Real.pi) ∧
    (∃ k:ℤ, monotonically_increasing_interval (λ x => Real.sin (2 * x - B)) k = Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) := sorry

-- (2) Maximum area of triangle ABC
def max_area_triangle (a c : ℝ) : ℝ := (1 / 2) * a * c * (Real.sin B)
theorem max_area_ABC :
  ∀ (a c : ℝ), acute_triangle_ABC → side_b → max_area_triangle a c ≤ 4 * Real.sqrt 3 := sorry

end smallest_period_monotonic_interval_max_area_ABC_l228_228396


namespace greatest_possible_NPMPP_l228_228064

theorem greatest_possible_NPMPP :
  ∃ (M N P PP : ℕ),
    0 ≤ M ∧ M ≤ 9 ∧
    M^2 % 10 = M ∧
    NPMPP = M * (1111 * M) ∧
    NPMPP = 89991 := by
  sorry

end greatest_possible_NPMPP_l228_228064


namespace overall_gain_percentage_l228_228666

noncomputable theory

-- Define conditions
def investment_stock_market := 5000
def investment_artwork := 10000
def investment_crypto := 15000
def returns_stock_market := 6000
def sale_price_artwork := 12000
def sales_tax_artwork := 0.05
def crypto_amount_rub := 17000
def conversion_rate := 1.03
def exchange_fee_crypto := 0.02

-- Define initial investment
def total_initial_investment := investment_stock_market + investment_artwork + investment_crypto

-- Define returns on investments
def net_return_artwork := sale_price_artwork - (sales_tax_artwork * sale_price_artwork)
def gross_return_crypto := crypto_amount_rub * conversion_rate
def net_return_crypto := gross_return_crypto - (exchange_fee_crypto * gross_return_crypto)

-- Define total returns
def total_returns := returns_stock_market + net_return_artwork + net_return_crypto

-- Define overall gain and gain percentage
def overall_gain := total_returns - total_initial_investment
def gain_percentage := (overall_gain / total_initial_investment) * 100

-- Lean statement to prove the overall gain percentage
theorem overall_gain_percentage :
  abs (gain_percentage - 15.20) < 0.01 :=
by
  sorry

end overall_gain_percentage_l228_228666


namespace min_value_g_on_interval_l228_228612

noncomputable def g(x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 6))

theorem min_value_g_on_interval : ∀ x : ℝ, -Real.pi / 3 ≤ x ∧ x ≤ 0 -> g(x) = -1 :=
by
  -- Proof skipped
  sorry

end min_value_g_on_interval_l228_228612


namespace printer_time_ratio_l228_228267

theorem printer_time_ratio
  (X_time : ℝ) (Y_time : ℝ) (Z_time : ℝ)
  (hX : X_time = 15)
  (hY : Y_time = 10)
  (hZ : Z_time = 20) :
  (X_time / (Y_time * Z_time / (Y_time + Z_time))) = 9 / 4 :=
by
  sorry

end printer_time_ratio_l228_228267


namespace equilateral_triangle_combination_l228_228411

-- Given the interior angle of a polygon in degrees
constant interior_angle : ℕ → ℚ
-- Values of interior angles for each polygon mentioned
def interior_angle_quad := 90    -- Regular Quadrilateral
def interior_angle_hex  := 120   -- Regular Hexagon
def interior_angle_oct  := 135   -- Regular Octagon
def interior_angle_tri  := 60    -- Equilateral Triangle
def fixed_angle := 150  -- Given regular polygon
 
-- Define the seamless combination condition
def seamless_combination (a b : ℚ) : Prop := ∃ k l : ℕ, k ≥ 1 ∧ l ≥ 1 ∧ k * a + l * b = 360

theorem equilateral_triangle_combination:
  seamless_combination fixed_angle interior_angle_tri ∧
  ¬ seamless_combination fixed_angle interior_angle_quad ∧
  ¬ seamless_combination fixed_angle interior_angle_hex ∧
  ¬ seamless_combination fixed_angle interior_angle_oct :=
by
  sorry

end equilateral_triangle_combination_l228_228411


namespace find_m_value_l228_228237

theorem find_m_value
  (m : ℝ)
  (M N : Set ℂ)
  (hM : M = {1, 2, complex.mk (m^2 - 3*m - 1) (m^2 - 5*m - 6)})
  (hN : N = {-1, 3})
  (hMI : M ∩ N = {3}) :
  m = -1 :=
by
  sorry

end find_m_value_l228_228237


namespace average_length_l228_228544

def length1 : ℕ := 2
def length2 : ℕ := 3
def length3 : ℕ := 7

theorem average_length : (length1 + length2 + length3) / 3 = 4 :=
by
  sorry

end average_length_l228_228544


namespace probability_of_prime_and_odd_l228_228765

noncomputable def list_of_balls : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def prime_and_odd (n : ℕ) : Prop := is_prime n ∧ is_odd n

theorem probability_of_prime_and_odd :
  let total_balls := list_of_balls.length
  let prime_and_odd_balls := (list_of_balls.filter prime_and_odd).length
  prime_and_odd_balls = 3 ∧ total_balls = 8 →
  (prime_and_odd_balls: ℝ) / (total_balls: ℝ) = 3 / 8 :=
begin
  sorry
end

end probability_of_prime_and_odd_l228_228765


namespace gcd_18_30_45_l228_228363

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l228_228363


namespace always_even_square_l228_228824

theorem always_even_square (x : ℕ) (h_pos : x > 0) : 
  (∀ y, y ∈ [{x ^ 2, 2 * x, |x - 2|, x * (x + 2), (x + 2) ^ 2}] → ¬ (∃ n : ℕ, y = 2*x ∧ ¬even (y^2))) :=
by
  sorry

end always_even_square_l228_228824


namespace perimeter_PQRS_l228_228958

structure Point2D where
  x : ℝ
  y : ℝ

def distance (P Q : Point2D) : ℝ :=
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def P : Point2D := ⟨1, 2⟩
def Q : Point2D := ⟨3, 6⟩
def R : Point2D := ⟨6, 3⟩
def S : Point2D := ⟨8, 1⟩

def perimeter (P Q R S : Point2D) : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

def perimeter_as_sum_of_sqrts : ℝ → ℝ → ℝ :=
  λ x y, x * real.sqrt 2 + y * real.sqrt 5

theorem perimeter_PQRS :
  ∃ x y : ℝ, perimeter P Q R S = perimeter_as_sum_of_sqrts x y ∧ x + y = 12 :=
by
  sorry

end perimeter_PQRS_l228_228958


namespace measure_of_angle_x_l228_228343

-- Given conditions
def angle_ABC : ℝ := 120
def angle_BAD : ℝ := 31
def angle_BDA (x : ℝ) : Prop := x + 60 + 31 = 180 

-- Statement to prove
theorem measure_of_angle_x : 
  ∃ x : ℝ, angle_BDA x → x = 89 :=
by
  sorry

end measure_of_angle_x_l228_228343


namespace rounding_addition_equivalence_l228_228528

theorem rounding_addition_equivalence:
  (round_to_nearest_ten (59 + 28) = 90) :=
by
  -- We define an auxiliary function to round to the nearest ten
  def round_to_nearest_ten (n : ℕ) : ℕ :=
    (n + 5) / 10 * 10

  -- Converting the initial numbers and their sum
  have h1 : 59 + 28 = 87 := by norm_num
  -- Applying the rounding function to the sum
  have h2 : round_to_nearest_ten 87 = 90 := by
    simp [round_to_nearest_ten, h1]

  exact h2

end rounding_addition_equivalence_l228_228528


namespace parallel_relationship_l228_228900

-- Define lines and planes as sets of points
variable {Point : Type}
variable Line : Type := Set Point
variable Plane : Type := Set Point

-- Define the conditions
def parallel_line_plane (a : Line) (α : Plane) : Prop :=
  ∀ (p : Point), p ∈ a → p ∉ α

def line_in_plane (b : Line) (α : Plane) : Prop :=
  ∀ (p : Point), p ∈ b → p ∈ α

-- Define what it means for two lines to be parallel
def parallel_lines (a b : Line) : Prop :=
  ∀ (p : Point), p ∈ a → p ∉ b

-- The main theorem to be proved
theorem parallel_relationship (a b : Line) (α : Plane) :
  parallel_line_plane a α →
  line_in_plane b α →
  parallel_lines a b :=
by sorry

end parallel_relationship_l228_228900


namespace shortest_distance_parabola_line_l228_228416

theorem shortest_distance_parabola_line :
  (∀ M : ℝ × ℝ, (M.2 ^ 2 = -M.1) → 
    let x3 := 1 - 3;
    let d := x3 / Real.sqrt (1 ^ 2 + 2 ^ 2);
    d = 2 / Real.sqrt 5) :=
begin
  sorry
end

end shortest_distance_parabola_line_l228_228416


namespace regular_octagon_area_l228_228707

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228707


namespace ellipse_major_axis_length_l228_228322

theorem ellipse_major_axis_length (F1 F2 : ℝ × ℝ) (y_line : ℝ) 
  (hF1 : F1 = (5, 8)) (hF2 : F2 = (25, 28)) (hy_line : y_line = 1) :
  let reflecting_point := (5, -6 : ℝ×ℝ)
  sqrt ((25 - 5)^2 + (28 - (-6))^2) = 2 * sqrt 389 := by
    sorry

end ellipse_major_axis_length_l228_228322


namespace smallest_integer_x_l228_228973

noncomputable def biasedCoinProbability : ℚ := 3/5

def generatingFunctionFairCoins (x : ℚ) : ℚ :=
(1 + x) ^ 4

def generatingFunctionBiasedCoin (x : ℚ) : ℚ :=
3 + 2 * x

def combinedGeneratingFunction (x : ℚ) : ℚ :=
generatingFunctionFairCoins(x) * generatingFunctionBiasedCoin(x)

def probabilitiesCoefficients : list ℚ := [3, 14, 27, 28, 20, 10, 2]

def probabilitySameHeads : ℚ :=
(probabilitiesCoefficients.map (λ c, c^2)).sum / (probabilitiesCoefficients.sum ^ 2)

def smallestX : ℚ :=
(1 : ℚ) / probabilitySameHeads

theorem smallest_integer_x :
  ∃ (x : ℕ), x = 5 ∧ (1 / x.toRat) < probabilitySameHeads :=
by
  use 5
  sorry

end smallest_integer_x_l228_228973


namespace unique_integer_function_l228_228547

theorem unique_integer_function (f : ℤ → ℤ) :
  (∀ n : ℤ, f(f(n)) = n) →
  (∀ n : ℤ, f(f(n + 2) + 2) = n) →
  f(0) = 1 →
  (∀ n : ℤ, f(n) = 1 - n) :=
by {
  intros h1 h2 h3,
  sorry -- Proof goes here
}

end unique_integer_function_l228_228547


namespace f_domain_l228_228019

def domain_of_f (x : ℝ) : Prop := x > 7

def f (x : ℝ) : ℝ := (2 * x - 3) / Real.sqrt (x - 7)

theorem f_domain : ∀ x : ℝ, (∃ y : ℝ, f y = x) ↔ domain_of_f x := by
  sorry

end f_domain_l228_228019


namespace problem_statement_l228_228518

theorem problem_statement 
  (f : ℝ → ℝ) 
  (non_neg : ∀ x, 0 ≤ x → 0 ≤ f x)
  (f_one : f 1 = 1)
  (f_superadditive : ∀ x y, x ∈ Icc (0 : ℝ) 1 → y ∈ Icc (0 : ℝ) 1 → f (x + y) ≥ f x + f y) :
  (∀ x, x ∈ Icc (0 : ℝ) 1 → f x ≤ 2 * x) ∧ 
  ¬ (∀ x, x ∈ Icc (0 : ℝ) 1 → f x ≤ 1.9 * x) :=
sorry

end problem_statement_l228_228518


namespace distance_from_focus_to_line_l228_228018

-- Definitions for the ellipse and the line
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def line (x y : ℝ) : Prop := y = (√3 / 3) * x

-- Definition for the distance function
def distance (A B C x y: ℝ) : ℝ := abs (A * x + B * y + C) / real.sqrt (A^2 + B^2)

-- Condition for the right focus of the ellipse
def is_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Helper definitions for the line equation
def A : ℝ := 1
def B : ℝ := -√3
def C : ℝ := 0

-- The main theorem to prove
theorem distance_from_focus_to_line :
  ellipse 1 0 ∧ line 1 ((√3 / 3) * 1) → distance A B C 1 0 = 1/2 := 
by sorry

end distance_from_focus_to_line_l228_228018


namespace triangle_shape_max_value_expression_l228_228151

-- Part (1): Prove that the triangle is either right or isosceles
theorem triangle_shape (A B C a b c : ℝ)
(hSides : a = 1 / sin B ∧ b = 1 / sin A)
(hTrig: sin (A - B) * cos C = cos B * sin (A - C)) : 
    (A = π / 2) ∨ (B = C) := 
sorry

-- Part (2): Prove the maximum value of the expression:
theorem max_value_expression (A B : ℝ)
(hAcuteness: A < π / 2 ∧ B < π / 2)
(hSide: a = 1 / sin B):
    ∃ (max_val : ℝ), max_val = 25 / 16 :=
sorry

end triangle_shape_max_value_expression_l228_228151


namespace longer_bus_ride_l228_228203

theorem longer_bus_ride :
  let oscar := 0.75
  let charlie := 0.25
  oscar - charlie = 0.50 :=
by
  sorry

end longer_bus_ride_l228_228203


namespace number_of_foreign_stamps_l228_228482

theorem number_of_foreign_stamps
  (total_stamps : ℕ)
  (old_stamps : ℕ)
  (foreign_and_old_stamps : ℕ)
  (neither_foreign_nor_old_stamps : ℕ)
  (total_eq : total_stamps = 200)
  (old_eq : old_stamps = 50)
  (foreign_and_old_eq : foreign_and_old_stamps = 20)
  (neither_eq : neither_foreign_nor_old_stamps = 80) : 
  ∃ F : ℕ, F = 90 :=
by
  have h1 : total_stamps - neither_foreign_nor_old_stamps = 120 :=
    by rw [total_eq, neither_eq]; norm_num
  have h2 : old_stamps - foreign_and_old_stamps = 30 :=
    by rw [old_eq, foreign_and_old_eq]; norm_num
  have h3 : F = total_stamps - neither_foreign_nor_old_stamps - (old_stamps - foreign_and_old_stamps) :=
    by rw [total_eq, old_eq, foreign_and_old_eq, neither_eq]; norm_num
  use 90
  rw h3
  norm_num

end number_of_foreign_stamps_l228_228482


namespace time_to_cross_pole_is_2_5_l228_228156

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end time_to_cross_pole_is_2_5_l228_228156


namespace partI_partII_l228_228514

-- Part (I)
theorem partI (a : ℝ) (h1 : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → |x - a| ≤ 4) : -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

-- Part (II)
theorem partII (a : ℝ) (h2 : ∃ x : ℝ, |x - a| - |x + a| ≤ 2a - 1) : 1 / 4 ≤ a :=
by
  sorry

end partI_partII_l228_228514


namespace find_d_for_tangency_l228_228371

theorem find_d_for_tangency:
  ∃ d: ℝ, (∀ x y: ℝ, y = 3 * x + d ∧ y^2 = 12 * x → (y - 2)^2 = 0) → d = 1 :=
begin
  sorry
end


end find_d_for_tangency_l228_228371


namespace sum_inferior_numbers_correct_l228_228513

noncomputable def a (n : ℕ) : ℝ := Real.log (n + 2) / Real.log (n + 1 + 1)

def is_inferior_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), (2^k - 2 = n)

def sum_inferior_numbers (upper_bound : ℕ) : ℕ :=
  (2^(Nat.log2 (upper_bound + 1))).foldr (λ k acc, k - 2 + acc) 0

theorem sum_inferior_numbers_correct : sum_inferior_numbers 2004 = 2026 := 
by
  have h1: (2^10 - 2) = 1022 := rfl
  have h2: sum_inferior_numbers 2004 = 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 - 2 * 9 :=
    sorry
  rw h2
  norm_num

end sum_inferior_numbers_correct_l228_228513


namespace exists_smallest_n_f_eq_2010_l228_228516

def f : ℕ → ℤ := sorry

theorem exists_smallest_n_f_eq_2010 :
  ∃ n : ℕ, f(n) = 2010 :=
sorry

end exists_smallest_n_f_eq_2010_l228_228516


namespace miles_reads_128_pages_l228_228535

noncomputable def pages_read (daily_reading_fraction : ℚ) 
  (time_fraction_per_type : ℚ) (pages_per_hour_novels : ℚ) 
  (pages_per_hour_graphic_novels : ℚ) (pages_per_hour_comic_books : ℚ) : ℚ :=
  let total_hours := 24 * daily_reading_fraction in
  let hours_per_type := total_hours * time_fraction_per_type in
  let pages_novels := pages_per_hour_novels * hours_per_type in
  let pages_graphic_novels := pages_per_hour_graphic_novels * hours_per_type in
  let pages_comic_books := pages_per_hour_comic_books * hours_per_type in
  pages_novels + pages_graphic_novels + pages_comic_books

theorem miles_reads_128_pages :
  pages_read (1/6) (1/3) 21 30 45 = 128 := by
  sorry

end miles_reads_128_pages_l228_228535


namespace moles_of_HCl_needed_l228_228101

theorem moles_of_HCl_needed : ∀ (moles_KOH : ℕ), moles_KOH = 2 →
  (moles_HCl : ℕ) → moles_HCl = 2 :=
by
  sorry

end moles_of_HCl_needed_l228_228101


namespace lines_from_intersection_of_parallel_planes_l228_228473

-- Definitions from the conditions
variables {P1 P2 P3 : Plane}
variables {l1 l2 : Line}

-- Given conditions
def planes_parallel (P1 P2 : Plane) : Prop :=
  ∀ (x : Point), x ∈ P1 → x ∈ P2

def plane_intersects_line (P : Plane) (l : Line) : Prop :=
  ∃ (x : Point), x ∈ l ∧ x ∈ P

-- The theorem (proof problem)
theorem lines_from_intersection_of_parallel_planes (h1 : planes_parallel P1 P2)
  (h2 : plane_intersects_line P3 l1)
  (h3 : plane_intersects_line P3 l2)
  : l1.parallel l2 :=
sorry

end lines_from_intersection_of_parallel_planes_l228_228473


namespace ratio_and_lcm_l228_228471

noncomputable def common_factor (a b : ℕ) := ∃ x : ℕ, a = 3 * x ∧ b = 4 * x

theorem ratio_and_lcm (a b : ℕ) (h1 : common_factor a b) (h2 : Nat.lcm a b = 180) (h3 : a = 60) : b = 45 :=
by sorry

end ratio_and_lcm_l228_228471


namespace find_LP_l228_228613

variables (A B C K L P M : Type) 
variables {AC BC AK CK CL AM LP : ℕ}

-- Defining the given conditions
def conditions (AC BC AK CK : ℕ) (AM : ℕ) :=
  AC = 360 ∧ BC = 240 ∧ AK = CK ∧ AK = 180 ∧ AM = 144

-- The theorem statement: proving LP equals 57.6
theorem find_LP (h : conditions 360 240 180 180 144) : LP = 576 / 10 := 
by sorry

end find_LP_l228_228613


namespace moles_NaCH3COO_formed_l228_228780

-- We represent the number of moles of each substance
def moles_CH3COOH : ℝ := 1
def moles_NaOH : ℝ := 1

-- The main statement that needs to be proved
theorem moles_NaCH3COO_formed : 
  ∀ (moles_CH3COOH moles_NaOH : ℝ),
  moles_CH3COOH = 1 ∧ moles_NaOH = 1 → 
  ∃ (moles_NaCH3COO : ℝ), moles_NaCH3COO = 1 :=
by
  intros moles_CH3COOH moles_NaOH h,
  cases h with h_CH3COOH h_NaOH,
  exists 1,
  sorry

end moles_NaCH3COO_formed_l228_228780


namespace find_f1_plus_f1_deriv_l228_228903

noncomputable def f : ℝ → ℝ := sorry

theorem find_f1_plus_f1_deriv
  (h_tangent : ∀ x, (1, f 1) ∈ set_of (λ (p : ℝ) (y : ℝ), y = (1/2) * p + 2))
  (hf_diff : differentiable_at ℝ f 1) :
  f 1 + (deriv f 1) = 3 :=
sorry

end find_f1_plus_f1_deriv_l228_228903


namespace inequality_always_holds_l228_228451

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) : a + c > b + c :=
sorry

end inequality_always_holds_l228_228451


namespace regular_octagon_area_l228_228687

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228687


namespace find_n_l228_228772

theorem find_n :
  ∃ (n : ℤ), (4 ≤ n ∧ n ≤ 8) ∧ (n % 5 = 2) ∧ (n = 7) :=
by
  sorry

end find_n_l228_228772


namespace distance_between_Albany_and_Syracuse_l228_228313

theorem distance_between_Albany_and_Syracuse :
  let v1 := 40
  let v2 := 50
  let t := 5.4 in
  ∃ D : ℝ, (D / v1 + D / v2 = t) ∧ (D = 120) :=
by
  sorry

end distance_between_Albany_and_Syracuse_l228_228313


namespace sum_of_distances_value_of_a_plus_b_l228_228152

-- Define the coordinates of the points D, E, F, and Q
def D := (0, 0)
def E := (8, 0)
def F := (2, 4)
def Q := (3, 1)

-- Distance function between two points
def distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2)

-- The main proposition to prove
theorem sum_of_distances : distance D Q + distance E Q + distance F Q = 2 * Real.sqrt 10 + Real.sqrt 26 :=
by
  sorry

-- A supporting proposition to get the value of a + b
theorem value_of_a_plus_b : 2 + 1 = 3 := by
  rfl

end sum_of_distances_value_of_a_plus_b_l228_228152


namespace nearest_integer_to_expression_l228_228622

theorem nearest_integer_to_expression : 
  Real.floor ((3 + Real.sqrt 3)^4) = 504 := by
  sorry

end nearest_integer_to_expression_l228_228622


namespace initial_percentage_filled_l228_228649

theorem initial_percentage_filled (capacity : ℝ) (added : ℝ) (final_fraction : ℝ) (initial_water : ℝ) :
  capacity = 80 → added = 20 → final_fraction = 3/4 → 
  initial_water = (final_fraction * capacity - added) → 
  100 * (initial_water / capacity) = 50 :=
by
  intros
  sorry

end initial_percentage_filled_l228_228649


namespace pyramid_boxes_l228_228302

theorem pyramid_boxes (a₁ a₂ aₙ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = 15) 
  (h₃ : aₙ = 39) 
  (h₄ : d = 3) 
  (h₅ : a₂ = a₁ + d)
  (h₆ : aₙ = a₁ + (n - 1) * d) 
  (h₇ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 255 :=
by
  sorry

end pyramid_boxes_l228_228302


namespace octagon_area_l228_228673

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228673


namespace solve_x_l228_228562

theorem solve_x :
  (2 / 3 - 1 / 4) = 1 / (12 / 5) :=
by
  sorry

end solve_x_l228_228562


namespace octagon_area_correct_l228_228695

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228695


namespace regular_octagon_area_l228_228689

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228689


namespace identity_function_l228_228182

-- Given conditions as definitions
def f : ℕ → ℕ := sorry

-- The condition: ∀ n ∈ ℕ, f(n + 1) > f(f(n))
axiom condition : ∀ n : ℕ, f(n + 1) > f(f(n))

-- The theorem statement we need to prove
theorem identity_function : ∀ n : ℕ, f(n) = n :=
by
  sorry

end identity_function_l228_228182


namespace place_additional_rook_l228_228200

open Set

-- Definitions of the chessboard and rook placements.
def Chessboard : Type := Fin 10 × Fin 10

variables (black white : Set Chessboard)
variables (initial_rooks : Set Chessboard)

-- Condition that checks no two rooks are on the same row or column
def no_two_rooks_attack (rooks : Set Chessboard) : Prop :=
  ∀ r1 r2 ∈ rooks, r1 ≠ r2 → r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2

-- Condition that rooks are equally distributed on black and white squares
def equal_rooks_on_colors (rooks : Set Chessboard) (black white : Set Chessboard) : Prop :=
  (rooks ∩ black).card = (rooks ∩ white).card

-- Maximum 8 rooks on the board
def max_eight_rooks (rooks : Set Chessboard) : Prop := rooks.card ≤ 8

-- Main theorem statement
theorem place_additional_rook (black white : Set Chessboard)
  (initial_rooks : Set Chessboard)
  (h_no_attack : no_two_rooks_attack initial_rooks)
  (h_equal_colors : equal_rooks_on_colors initial_rooks black white)
  (h_max_rooks : max_eight_rooks initial_rooks) :
  ∃ new_rook : Chessboard, new_rook ∉ initial_rooks ∧
  no_two_rooks_attack (initial_rooks ∪ {new_rook}) :=
sorry

end place_additional_rook_l228_228200


namespace total_tiles_l228_228308

theorem total_tiles (s : ℕ) (H1 : 2 * s - 1 = 57) : s^2 = 841 := by
  sorry

end total_tiles_l228_228308


namespace number_of_parts_divided_by_planes_l228_228942

theorem number_of_parts_divided_by_planes (n : ℕ) : 
  (∀ (p q r : fin n → ℝ), set.inter [p, q, r].count_points = 1) ∧ 
  (∀ (p q r s : fin n → ℝ), set.inter [p, q, r, s].count_points = 0) → 
  ∃ K_n : ℕ, K_n = (1 / 6 : ℚ) * (n^3 + 5 * n + 6) :=
by
  sorry

end number_of_parts_divided_by_planes_l228_228942


namespace water_temp_increase_per_minute_l228_228499

theorem water_temp_increase_per_minute :
  ∀ (initial_temp final_temp total_time pasta_time mixing_ratio : ℝ),
    initial_temp = 41 →
    final_temp = 212 →
    total_time = 73 →
    pasta_time = 12 →
    mixing_ratio = (1 / 3) →
    ((final_temp - initial_temp) / (total_time - pasta_time - (mixing_ratio * pasta_time)) = 3) :=
by
  intros initial_temp final_temp total_time pasta_time mixing_ratio
  sorry

end water_temp_increase_per_minute_l228_228499


namespace problem1_problem2_problem2_zero_problem2_neg_l228_228191

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + a*x + a
def g (a x : ℝ) : ℝ := a*(f a x) - a^2*(x + 1) - 2*x

-- Problem 1
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f a x1 - x1 = 0 ∧ f a x2 - x2 = 0) →
  (0 < a ∧ a < 3 - 2*Real.sqrt 2) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ 
    if a < 1 then a-2 
    else -1/a) :=
sorry

theorem problem2_zero (h2 : a = 0) : 
  g a 1 = -2 :=
sorry

theorem problem2_neg (a : ℝ) (h3 : a < 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ a - 2) :=
sorry

end problem1_problem2_problem2_zero_problem2_neg_l228_228191


namespace probability_12OA_l228_228141

-- Definitions based on conditions
def is_digit (s: Char) : Prop := s.isdigit

-- Vowels in the problem
def is_vowel (ch : Char) : Prop := ch ∈ ['A', 'E', 'I', 'O', 'U']

-- License plate validity given the conditions in the country of Mathlandia
def valid_license_plate (plate : List Char) : Prop :=
  plate.length = 4 ∧ is_digit plate.head ∧ is_digit plate.tail.head ∧
  ∃ p3 p4, p3 ∈ set_of (λ (c: Char), true) ∧ p4 ∈ set_of (λ (c: Char), true) ∧
  (is_vowel p3 ∨ is_vowel p4)

-- Total number of valid plates
def total_plates : ℕ := 21000

-- Specific plate "12OA"
def plate_12OA : List Char := ['1', '2', 'O', 'A']

-- Formal proof problem statement
theorem probability_12OA : 
  valid_license_plate plate_12OA →
  ∀ total_plates : ℕ, total_plates = 21000 →
  \frac{1}{total_plates} = \frac{1}{21000} :=
begin
  sorry
end

end probability_12OA_l228_228141


namespace page_added_twice_l228_228234

theorem page_added_twice (n x : ℕ) (h₁ : ∑ k in finset.range (n + 1), k = (n * (n + 1)) / 2)
  (h₂ : n = 77) (h₃ : (∑ k in finset.range (n + 1), k) + x = 3050) : x = 47 :=
sorry

end page_added_twice_l228_228234


namespace train_length_l228_228637

theorem train_length (L : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = (L + 140) / 15)
  (h2 : v2 = (L + 250) / 20) 
  (h3 : v1 = v2) :
  L = 190 :=
by sorry

end train_length_l228_228637


namespace sine_of_angle_between_line_and_plane_l228_228902

noncomputable def direction_vector : ℝ × ℝ × ℝ := (1, 0, 3)
noncomputable def normal_vector : ℝ × ℝ × ℝ := (-2, 0, 2)

theorem sine_of_angle_between_line_and_plane : 
  sin (angle_between_line_and_plane direction_vector normal_vector) = (real.sqrt 5) / 5 :=
sorry

end sine_of_angle_between_line_and_plane_l228_228902


namespace circle_equation_l228_228367

-- Defining the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Defining the center M of the circle on the x-axis
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Defining the squared distance function between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Statement: Prove that the standard equation of the circle is (x - 2)² + y² = 10
theorem circle_equation : ∃ a : ℝ, (dist_sq (M a) A = dist_sq (M a) B) ∧ ((M a).1 = 2) ∧ (dist_sq (M a) A = 10) :=
sorry

end circle_equation_l228_228367


namespace function_monotonic_increasing_interval_l228_228121

noncomputable def isMonotonicallyIncreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x1 x2 ∈ s, x1 < x2 → f x1 ≤ f x2

theorem function_monotonic_increasing_interval :
  ∀ (k : ℝ), isMonotonicallyIncreasing (λ x => k * x - Real.log x) {x | x > 1 / 2} ↔ k ∈ Set.Ici 2 := by
  sorry

end function_monotonic_increasing_interval_l228_228121


namespace factorial_trailing_zeros_base_8_l228_228865

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228865


namespace Carissa_ran_at_10_feet_per_second_l228_228333

theorem Carissa_ran_at_10_feet_per_second :
  ∀ (n : ℕ), 
  (∃ (a : ℕ), 
    (2 * a + 2 * n^2 * a = 260) ∧ -- Total distance
    (a + n * a = 30)) → -- Total time spent
  (2 * n = 10) :=
by
  intro n
  intro h
  sorry

end Carissa_ran_at_10_feet_per_second_l228_228333


namespace original_ratio_l228_228236

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 :=
by sorry

end original_ratio_l228_228236


namespace form_numbers_from_1_to_39_using_five_threes_l228_228618

def use_five_threes (n : ℕ) : Prop :=
  ∃ (e : ℕ), e = 3 ∧ -- Placeholder condition to represent the use of exactly five 3's
    (∃ a b c d : ℕ, a + b + c + d + e = 5 ∧ -- Ensure exactly five 3's
      -- The expressions using the five 3's form the number n with arithmetic operations and exponentiation
      (n = (some_valid_expression_with_five_3s a b c d e)
       ∨ ... -- Other possible valid expressions
      )
    )

theorem form_numbers_from_1_to_39_using_five_threes :
  ∀ n, 1 ≤ n ∧ n ≤ 39 → use_five_threes n :=
begin
  sorry -- Proof not required
end

end form_numbers_from_1_to_39_using_five_threes_l228_228618


namespace total_dots_not_visible_l228_228034

def total_dots_on_dice (n : ℕ): ℕ := n * 21
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 4 + 5 + 6
def total_dice : ℕ := 4

theorem total_dots_not_visible :
  total_dots_on_dice total_dice - visible_dots = 58 := by
  sorry

end total_dots_not_visible_l228_228034


namespace evaluate_expression_l228_228011

/- The mathematical statement to prove:

Evaluate the expression 2/10 + 4/20 + 6/30, then multiply the result by 3
and show that it equals to 9/5.
-/

theorem evaluate_expression : 
  (2 / 10 + 4 / 20 + 6 / 30) * 3 = 9 / 5 := 
by 
  sorry

end evaluate_expression_l228_228011


namespace sequence_term_number_l228_228837

theorem sequence_term_number (n : ℕ) : (n ≥ 1) → (n + 3 = 17 ∧ n + 1 = 15) → n = 14 := 
by
  intro h1 h2
  sorry

end sequence_term_number_l228_228837


namespace fraction_of_female_participants_this_year_l228_228545

theorem fraction_of_female_participants_this_year :
  ∀ (y : Real) (males_last_year : Real) (increase_males : Real) (increase_females : Real) (increase_total : Real),
  males_last_year = 30 →
  increase_males = 1.10 →
  increase_females = 1.25 →
  increase_total = 1.15 →
  let females_last_year := y in
  let males_this_year := increase_males * males_last_year in
  let females_this_year := increase_females * females_last_year in
  let total_last_year := males_last_year + females_last_year in
  let total_this_year := increase_total * total_last_year in
  total_this_year = males_this_year + females_this_year →
  females_last_year = 15 →
  females_this_year = 19 →
  (19 / (33 + 19) = (19 / 52)) :=
by
  intros y males_last_year increase_males increase_females increase_total males_last_year_def increase_males_def increase_females_def increase_total_def females_last_year_def males_this_year females_this_year total_last_year total_this_year total_this_year_eq females_last_year_val females_this_year_val
  split
  sorry

end fraction_of_female_participants_this_year_l228_228545


namespace negative_correction_is_correct_l228_228654

-- Define the constants given in the problem
def gain_per_day : ℚ := 13 / 4
def set_time : ℚ := 8 -- 8 A.M. on April 10
def end_time : ℚ := 15 -- 3 P.M. on April 19
def days_passed : ℚ := 9

-- Calculate the total time in hours from 8 A.M. on April 10 to 3 P.M. on April 19
def total_hours_passed : ℚ := days_passed * 24 + (end_time - set_time)

-- Calculate the gain in time per hour
def gain_per_hour : ℚ := gain_per_day / 24

-- Calculate the total gained time over the total hours passed
def total_gain : ℚ := total_hours_passed * gain_per_hour

-- The negative correction m to be subtracted
def correction : ℚ := 2899 / 96

theorem negative_correction_is_correct :
  total_gain = correction :=
by
-- skipping the proof
sorry

end negative_correction_is_correct_l228_228654


namespace sum_dihedral_angles_l228_228541

variables {α β γ : ℝ} -- Dihedral angles at edges OA, OB, and OC
variables {A B C O : Type} -- Points on the diameter and circumference

-- A, B diametrically opposite on the base circumference
variable (diam_opp : diametrically_opposite A B)

-- C on the circumference of the other base and not in the plane ABO
variable (C_circum : on_circumference C)
variable (not_in_plane : ¬in_plane A B O C) -- C not in the plane ABO

-- O is the midpoint of the cylinder's axis
variable (midpoint_O : midpoint_axis O)

-- Proof statement
theorem sum_dihedral_angles (diam_opp : diametrically_opposite A B)
    (C_circum : on_circumference C) (not_in_plane : ¬in_plane A B O C)
    (midpoint_O : midpoint_axis O) : α + β + γ = 360 :=
sorry

end sum_dihedral_angles_l228_228541


namespace max_m_value_l228_228051

noncomputable def S (m n : ℕ) : set ℕ :=
  {i | 1 ≤ i ∧ i ≤ m * n}

structure collection (m n : ℕ) :=
  (sets : finset (finset ℕ))
  (size_sets : sets.card = 2 * n)
  (m_element : ∀ s ∈ sets, s.card = m)
  (pairwise_inter_disjoint : ∀ s₁ s₂ ∈ sets, s₁ ≠ s₂ → (s₁ ∩ s₂).card ≤ 1)
  (element_appears_twice : ∀ i ∈ S m n, (finset.filter (λ s, i ∈ s) sets).card = 2)

theorem max_m_value (n : ℕ) (h : 1 < n) : ∃ m, ∀ (c : collection m n), m ≤ 2 * n - 1 :=
begin
  use 2 * n - 1,
  intros c,
  sorry
end

end max_m_value_l228_228051


namespace intersection_of_M_and_P_l228_228461

def f (x : ℝ) : ℝ := real.log (x^2 - 4*x + 3)

def M : set ℝ := {x | x < 1 ∨ x > 3}

def ϕ (x : ℝ) : ℝ := real.sqrt ((4 - x) * (x - 3))

def P : set ℝ := {x | 3 ≤ x ∧ x ≤ 4}

theorem intersection_of_M_and_P : M ∩ P = {x | 3 < x ∧ x ≤ 4} :=
by {
  sorry
}

end intersection_of_M_and_P_l228_228461


namespace hyperbola_sum_l228_228581

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 5)
  (c_eq : c = 7)
  (c_squared_eq : c^2 = a^2 + b^2) :
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  rw [h_eq, k_eq, a_eq, c_eq] at *
  sorry

end hyperbola_sum_l228_228581


namespace pairing_probability_l228_228130

open Classical

noncomputable def prob_pairing_with_friends : ℝ :=
  let n : ℕ := 28 in
  let P_Eva_Tom : ℝ := 1 / (n - 1) in
  let P_June_Leo : ℝ := 1 / (n - 2) in
  P_Eva_Tom * P_June_Leo

theorem pairing_probability : prob_pairing_with_friends = 1 / 702 := by
  sorry

end pairing_probability_l228_228130


namespace simplify_expression_l228_228752

theorem simplify_expression (a b : ℚ) : (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 := 
by 
  sorry

end simplify_expression_l228_228752


namespace integral_eq_sol_l228_228062

theorem integral_eq_sol (t : ℝ) (h : ∫ x in 1..t, (- 1 / x + 2 * x) = 3 - Real.log 2) : t = 2 :=
by
  sorry

end integral_eq_sol_l228_228062


namespace average_weight_increase_l228_228220

-- Define the initial conditions and quantities
def n : Nat := 20
def A : ℝ := sorry -- Initial average weight (requires value type ℝ)
def w1 : ℝ := 40 -- Weight of the oarsman being replaced
def w2 : ℝ := 80 -- Weight of the new oarsman

theorem average_weight_increase :
  let initial_total_weight := n * A;
  let new_total_weight := initial_total_weight - w1 + w2;
  let initial_average_weight := initial_total_weight / n;
  let new_average_weight := new_total_weight / n;
  new_average_weight - initial_average_weight = 2 :=
begin
  sorry
end

end average_weight_increase_l228_228220


namespace decreasing_function_range_l228_228644

theorem decreasing_function_range (f : ℝ → ℝ) (hf : ∀ x, 0 < x → ∀ y, 0 < y → x < y → f y < f x) :
  (∀ a, 0 < a → (f(2 * a^2 + a + 1) < f(3 * a^2 - 4 * a + 1)) → (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5)) :=
begin
  intros a ha h,
  sorry
end

end decreasing_function_range_l228_228644


namespace students_on_bleachers_l228_228991

theorem students_on_bleachers (F B : ℕ) (h1 : F + B = 26) (h2 : F / (F + B) = 11 / 13) : B = 4 :=
by sorry

end students_on_bleachers_l228_228991


namespace max_segments_no_triangle_l228_228201

theorem max_segments_no_triangle (n : ℕ) (h_n : n = 100) :
  ∃ m, m = 2500 ∧
    (∀ (points : Fin n → (ℝ × ℝ)) (segments : Fin n × Fin n → Prop),
      (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (points i) (points j) (points k)) →
      (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ forms_triangle (segments (i, j)) (segments (j, k)) (segments (i, k))) →
      (∃ (segments_count : ℕ), segments_count = m ∧
        ∀ (s : Fin n × Fin n → Prop), (count_segments s = segments_count))) :=
by
  -- proof steps go here
  sorry

end max_segments_no_triangle_l228_228201


namespace three_letter_words_with_A_at_least_once_l228_228848

theorem three_letter_words_with_A_at_least_once :
  let total_words := 4^3
  let words_without_A := 3^3
  total_words - words_without_A = 37 :=
by
  let total_words := 4^3
  let words_without_A := 3^3
  sorry

end three_letter_words_with_A_at_least_once_l228_228848


namespace geometric_sequence_formula_l228_228143

noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)

def b_n (n : ℕ) : ℚ := (n * (n + 1) * (a_n n) + 1) / (n * (n + 1))

noncomputable def S_n (n : ℕ) : ℚ := ∑ k in Finset.range n, b_n (k + 1)

theorem geometric_sequence_formula :
  (∀ n : ℕ, 0 < n → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, 0 < n → S_n n = 2^n - 1/(n+1)) := 
by sorry

end geometric_sequence_formula_l228_228143


namespace cubic_meter_to_cubic_centimeters_l228_228447

theorem cubic_meter_to_cubic_centimeters (h : 1 = 100): (1 : ℝ^3) = (100^3 : ℝ^3) :=
by
  sorry

end cubic_meter_to_cubic_centimeters_l228_228447


namespace identify_negative_number_l228_228579

theorem identify_negative_number :
  ∃ n, n = -1 ∧ n < 0 ∧ (∀ m, m ∈ {-1, 0, 1, 2} → ((m = n) ∨ (m ≥ 0))) :=
by
  sorry

end identify_negative_number_l228_228579


namespace find_side_length_l228_228405

theorem find_side_length
  (a b : ℝ)
  (S : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end find_side_length_l228_228405


namespace minimum_triangle_area_l228_228646

noncomputable def triangle_A : (ℤ × ℤ) := (0, 0)
noncomputable def triangle_B : (ℤ × ℤ) := (48, 18)
noncomputable def area (C: ℤ × ℤ) : ℚ :=
  1 / 2 * abs (48 * C.2 - 18 * C.1)

theorem minimum_triangle_area :
  let C : ℤ × ℤ
  in (∀ C, 1 / 2 * abs (48 * C.2 - 18 * C.1) >= 3) ∧ (∃ C, 1 / 2 * abs (48 * C.2 - 18 * C.1) = 3) :=
begin
  sorry
end

end minimum_triangle_area_l228_228646


namespace only_convex_polygon_with_right_angles_is_rectangle_l228_228557

theorem only_convex_polygon_with_right_angles_is_rectangle (n : ℕ) 
  (h_convex : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_right_angle (internal_angle i)) 
  (h_internal_angle : ∀ (i : ℕ), internal_angle i = 90) 
  (h_polygon : n ≥ 3) :
  n = 4 :=
begin
  sorry
end

end only_convex_polygon_with_right_angles_is_rectangle_l228_228557


namespace tangent_line_eq_monotonicity_max_value_g_l228_228429

-- Define the functions
def f (x : ℝ) (a : ℝ) : ℝ := Math.log (1 + x) - (a * x^2 + x) / ((1 + x)^2)
def g (x : ℝ) : ℝ := ((1 + 1 / x) ^ x) * ((1 + x) ^ (1 / x))

-- Problem statements

-- (I) When a = 1, find the equation of the tangent line to the function f(x) at x = e - 1.
theorem tangent_line_eq (a : ℝ) (x : ℝ) (y : ℝ) : 
  a = 1 → 
  x = Real.exp 1 - 1 → 
  y = (f x 1) + ((x - (Real.exp 1 - 1)) * (x / ((1 + x)^2))) → 
  y = f (Real.exp 1 - 1) 1 + ((x - (Real.exp 1 - 1)) * ((Real.exp 1 - 1) / ((Real.exp 1)^2))) :=
sorry

-- (II) When 3/2 < a ≤ 1, discuss the monotonicity of the function f(x).
theorem monotonicity (a : ℝ) :  
  3/2 < a ∧ a ≤ 1 → 
  ∀ x, 
  (x > -1 ∧ x < 0 → f x a > 0) ∧ 
  (x > 2*a - 3 → f x a > 0) ∧ 
  (x > 0 ∧ x < 2*a - 3 → f x a < 0) :=
sorry

-- (III) If x > 0, find the maximum value of the function g(x).
theorem max_value_g : 
  ∀ x > 0, 
  g x ≤ 4 :=
sorry

end tangent_line_eq_monotonicity_max_value_g_l228_228429


namespace odd_functions_l228_228634

-- Definitions of the functions
def f1 (x : ℝ) := |x|
def f2 (x : ℝ) := 3 * x ^ 3
def f3 (x : ℝ) := 1 / x

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating which functions are odd
theorem odd_functions :
  (is_odd f2) ∧ (is_odd f3) :=
by
  sorry

end odd_functions_l228_228634


namespace range_of_m_l228_228040

variable (m : ℝ)

def p : Prop := (m^2 - 4 > 0) ∧ (m > 0)
def q : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
by
  intro h
  sorry

end range_of_m_l228_228040


namespace factorial_trailing_zeros_base_8_l228_228870

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228870


namespace petya_wins_l228_228539

-- Definition of the initial state of the game
def initial_number : string := "1" ++ "1".repeat(98) ++ "1"

-- Petya wins with optimal play
theorem petya_wins : ∀ n, (n = initial_number) → (∀ play, optimal_play play → result = Petya)
:= sorry

end petya_wins_l228_228539


namespace range_of_f_l228_228782

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.cos x) * (Real.sin x) + (Real.sin x)^4

-- Define the trigonometric identity as a hypothesis
lemma trigonometric_identity (x : ℝ) : (Real.sin x)^2 + (Real.cos x)^2 = 1 :=
Real.sin_sq_add_cos_sq x

-- Prove the range of the function is [0, 5/4]
theorem range_of_f : set.range f = set.Icc 0 (5 / 4) :=
sorry

end range_of_f_l228_228782


namespace probability_nonzero_sum_l228_228510

-- Definitions of Bernoulli random variables and their properties
variables {N : ℕ}
variables (ξ : Fin N → ℤ)
variable (m : ℕ)
hypothesis hξ_indep : ∀ i j, i ≠ j → ξ i ⊥ ξ j
hypothesis hξ_bernoulli : ∀ i, ∃ p : ℝ, p = 1 / 2 ∧ 
  (Probability (ξ i = 1) = p ∧ Probability (ξ i = -1) = p)
  
-- Defining S_m as the sum of ξ from ξ_1 to ξ_m
def S_m (m : ℕ) : ℤ := (Finset.range m).sum ξ

-- Statement of the theorem
theorem probability_nonzero_sum 
  (h2m_le_N : 2 * m ≤ N) : 
  Probability ((λ i, S_m (2 * m)) ≠ 0) = 2^(- (2 * m)) * (Nat.choose (2 * m) m) :=
sorry

end probability_nonzero_sum_l228_228510


namespace regular_octagon_area_l228_228709

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228709


namespace unique_point_X_l228_228501

variables {A B C X : Type} [AffineSpace ℝ X]
variables (A B C : X) (X P Q : X)

axiom non_collinear : ¬ collinear ℝ (set.insert A (set.insert B {C}))

def distance_square (P Q : X) : ℝ := (dist P Q) * (dist P Q)

theorem unique_point_X :
  ∃! X : X, 
    distance_square X A + distance_square X B + distance_square A B =
    distance_square X B + distance_square X C + distance_square B C ∧
    distance_square X C + distance_square X A + distance_square C A :=
sorry

end unique_point_X_l228_228501


namespace distinct_collections_count_l228_228202

def vowel_letters := ['I', 'O', 'Y']
def consonant_letters := ['B', 'L', 'G']

theorem distinct_collections_count :
  let vowels := 2 in
  let consonants := 2 in
  let all_letters := 7 in
  let indistinguishable_o := true in
  (vowels + consonants = 4) →
  (indistinguishable_o = true) →
  (vowel_letters.length = 3) →
  (consonant_letters.length = 3) →
  (vowels = 2) →
  (consonants = 2) →
  -- total distinct possible collections of letters
  12 := 
sorry

end distinct_collections_count_l228_228202


namespace increasing_iff_a_range_l228_228077

noncomputable def f (x a : ℝ) : ℝ := real.sqrt (x^2 - a * x + 3 * a)

theorem increasing_iff_a_range (a : ℝ) : 
  (∀ x Δx : ℝ, x ≥ 2 → Δx > 0 → f (x + Δx) a > f x a) ↔ -4 ≤ a ∧ a ≤ 4 := sorry

end increasing_iff_a_range_l228_228077


namespace compound_interest_period_l228_228360

/-- Given:
    P = Rs. 14800 (principal amount)
    r = 0.135 (annual interest rate)
    n = 1 (compounded annually)
    Interest = Rs. 4265.73 (compound interest accrued)
    We need to prove that the period t (in years) is approximately 2 for which the compound interest is accrued.
-/
theorem compound_interest_period :
  let P := 14800
  let r := 0.135
  let n := 1
  let interest := 4265.73
  let A := P + interest
  t = Real.log (A / P) / (n * Real.log (1 + r / n)) ∧ t ≈ 2 :=
by
  sorry

end compound_interest_period_l228_228360


namespace area_of_regular_octagon_in_circle_l228_228679

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228679


namespace sum_y_equals_2_pow_m_plus_1_l228_228181

def y (m : ℕ) : ℕ → ℕ 
| 0      := 1
| 1      := m
| (k+2)  := (m+1) * y (k+1) - (m - k) * y k / (k + 2)

theorem sum_y_equals_2_pow_m_plus_1 (m : ℕ) (hm : m > 0) : 
  (∑ k in range (m+2), y m k) = 2 ^ (m + 1) :=
sorry

end sum_y_equals_2_pow_m_plus_1_l228_228181


namespace find_integer_n_l228_228897

theorem find_integer_n (n : ℤ) : (⌊(n^2 / 9 : ℝ)⌋ - ⌊(n / 3 : ℝ)⌋ ^ 2 = 5) → n = 14 :=
by
  -- Proof is omitted
  sorry

end find_integer_n_l228_228897


namespace octagon_area_l228_228671

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228671


namespace number_of_trailing_zeroes_base8_l228_228863

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228863


namespace symmetric_point_l228_228285

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1: P = (2, 1)) (h2 : x - y + 1 = 0) :
  (b - 1) = -(a - 2) ∧ (a + 2) / 2 - (b + 1) / 2 + 1 = 0 → (a, b) = (0, 3) := 
sorry

end symmetric_point_l228_228285


namespace principal_amount_l228_228298

theorem principal_amount (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) :
    r = 0.12 → n = 2 → t = 20 →
    ∃ P : ℝ, A = P * (1 + r / n)^(n * t) :=
by
  intros hr hn ht
  have P := A / (1 + r / n)^(n * t)
  use P
  sorry

end principal_amount_l228_228298


namespace robinson_crusoe_sees_multiple_colors_l228_228194

def chameleons_multiple_colors (r b v : ℕ) : Prop :=
  let d1 := (r - b) % 3
  let d2 := (b - v) % 3
  let d3 := (r - v) % 3
  -- Given initial counts and rules.
  (r = 155) ∧ (b = 49) ∧ (v = 96) ∧
  -- Translate specific steps and conditions into properties
  (d1 = 1 % 3) ∧ (d2 = 1 % 3) ∧ (d3 = 2 % 3)

noncomputable def will_see_multiple_colors : Prop :=
  chameleons_multiple_colors 155 49 96 →
  ∃ (r b v : ℕ), r + b + v = 300 ∧
  ((r % 3 = 0 ∧ b % 3 ≠ 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 = 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 ≠ 0 ∧ v % 3 = 0))

theorem robinson_crusoe_sees_multiple_colors : will_see_multiple_colors :=
sorry

end robinson_crusoe_sees_multiple_colors_l228_228194


namespace expected_left_handed_students_l228_228246

/-- Calculate the expected number of left-handed students in a classroom of 32 students
    given that \( \frac{3}{8} \) of the students are left-handed. -/
theorem expected_left_handed_students 
  (proportion : ℚ := 3/8)
  (total_students : ℕ := 32) :
  let expected_left_handed := proportion * total_students
  in expected_left_handed = 12 :=
by
  sorry

end expected_left_handed_students_l228_228246


namespace first_meeting_time_of_boys_l228_228248

theorem first_meeting_time_of_boys 
  (L : ℝ) (v1_kmh : ℝ) (v2_kmh : ℝ) (v1_ms v2_ms : ℝ) (rel_speed : ℝ) (t : ℝ)
  (hv1_km_to_ms : v1_ms = v1_kmh * 1000 / 3600)
  (hv2_km_to_ms : v2_ms = v2_kmh * 1000 / 3600)
  (hrel_speed : rel_speed = v1_ms + v2_ms)
  (hl : L = 4800)
  (hv1 : v1_kmh = 60)
  (hv2 : v2_kmh = 100)
  (ht : t = L / rel_speed) :
  t = 108 := by
  -- we're providing a placeholder for the proof
  sorry

end first_meeting_time_of_boys_l228_228248


namespace negation_of_p_is_correct_l228_228085

open Nat

noncomputable def negation_of_p (p : Prop) : Prop :=
  ∀ n : ℕ, 2^n ≤ 1000

theorem negation_of_p_is_correct :
  let p := ∃ n : ℕ, 2^n > 1000 in
  (¬p) = (∀ n : ℕ, 2^n ≤ 1000) :=
by 
  sorry

end negation_of_p_is_correct_l228_228085


namespace cos_minimum_value_l228_228589

theorem cos_minimum_value :
  ∀ x ∈ Set.Icc (-(Real.pi / 3)) (Real.pi / 6), Real.cos x ≥ 1 / 2 :=
begin
  sorry
end

end cos_minimum_value_l228_228589


namespace correct_average_is_40_point_3_l228_228219

noncomputable def incorrect_average : ℝ := 40.2
noncomputable def incorrect_total_sum : ℝ := incorrect_average * 10
noncomputable def incorrect_first_number_adjustment : ℝ := 17
noncomputable def incorrect_second_number_actual : ℝ := 31
noncomputable def incorrect_second_number_provided : ℝ := 13
noncomputable def correct_total_sum : ℝ := incorrect_total_sum - incorrect_first_number_adjustment + (incorrect_second_number_actual - incorrect_second_number_provided)
noncomputable def number_of_values : ℝ := 10

theorem correct_average_is_40_point_3 :
  correct_total_sum / number_of_values = 40.3 :=
by
  sorry

end correct_average_is_40_point_3_l228_228219


namespace find_t_l228_228177

theorem find_t (t : ℝ) :
  let P := (2 * t - 3, 2) in
  let Q := (-2, 2 * t + 1) in
  let M := ( ((2 * t - 3) - 2) / 2, ((2) + (2 * t + 1)) / 2 ) in
  let d := dist M P in
  d ^ 2 = t ^ 2 + 1 →
  t = 1 + sqrt (3 / 2) ∨ t = 1 - sqrt (3 / 2) := 
by
  sorry

end find_t_l228_228177


namespace even_digit_in_sum_l228_228224

theorem even_digit_in_sum (N : ℤ) (hN : 10^16 ≤ N ∧ N < 10^17) :
  let S := N + (N.digits.reverse.digitsSum) in
  ∃ d : ℕ, d ∈ S.digits ∧ even d :=
by sorry

end even_digit_in_sum_l228_228224


namespace factorial_base_8_zeroes_l228_228854

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228854


namespace calculate_P_X_leq_2_l228_228263

noncomputable def binomialCoefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probDieShowing6 : ℚ := 1 / 6
def probDieNotShowing6 : ℚ := 5 / 6

def P_X_eq_0 : ℚ := probDieNotShowing6 ^ 10
def P_X_eq_1 : ℚ := binomialCoefficient 10 1 * probDieShowing6 * (probDieNotShowing6 ^ 9)
def P_X_eq_2 : ℚ := binomialCoefficient 10 2 * (probDieShowing6 ^ 2) * (probDieNotShowing6 ^ 8)

def P_X_leq_2 : ℚ := P_X_eq_0 + P_X_eq_1 + P_X_eq_2

theorem calculate_P_X_leq_2 : P_X_leq_2 =
  probDieNotShowing6 ^ 10 + 
  binomialCoefficient 10 1 * probDieShowing6 * (probDieNotShowing6 ^ 9) + 
  binomialCoefficient 10 2 * (probDieShowing6 ^ 2) * (probDieNotShowing6 ^ 8) :=
  by
  sorry

end calculate_P_X_leq_2_l228_228263


namespace number_of_dragons_is_one_l228_228912

def Creature : Type := {bob sam ted alice carol : Type}
def is_phoenix (c : Creature) : Prop := sorry
def is_dragon (c : Creature) : Prop := sorry

axiom creature_statements (c : Creature) : 
  (creature_statements.bob → (is_phoenix creature.statements.sam ↔ ¬is_phoenix creature.statements.bob)) ∧
  (creature.statements.sam → (is_phoenix creature.statements.ted ↔ ¬is_phoenix creature.statements.ted)) ∧
  (creature.statements.ted → (is_phoenix creature.statements.alice ↔ ¬is_phoenix creature.statements.alice)) ∧
  (creature.statements.alice → (is_phoenix creature.statements.carol ↔ ¬is_phoenix creature.statements.carol)) ∧
  (creature.statements.carol → (is_dragon creature.statements.bob ∧ is_dragon creature.statements.sam ∧ is_dragon creature.statements.ted ∧ is_dragon creature.statements.alice ∧ is_dragon creature.statements.carol = 3))

theorem number_of_dragons_is_one : 
  ∃ (b : Creature), is_dragon b ∧ 
  is_phoenix creature.statements.bob ∧ 
  is_dragon creature.statements.sam ∧ 
  is_phoenix creature.statements.ted ∧ 
  is_phoenix creature.statements.alice ∧ 
  is_phoenix creature.statements.carol :=
sorry

end number_of_dragons_is_one_l228_228912


namespace minimum_balance_label_l228_228399

-- Definitions of the grid, symmetry, coloring, and k-balance conditions
def grid : Type := array 5 (array 5 (Sum ℤ Unit))
def color_mappings_valid (g : grid) : Prop := sorry -- implementation of color criteria
def balance_valid (g : grid) (k : ℤ) : Prop := sorry -- implementation of balance criteria

-- The theorem statement
theorem minimum_balance_label : ∃ (g : grid) (k : ℤ), color_mappings_valid g ∧ balance_valid g k ∧ k = 5 := 
sorry

end minimum_balance_label_l228_228399


namespace find_k_l228_228098

variables (k : ℝ)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k*1 + (-3), k*2 + 2)
def vector_a_minus_2b : ℝ × ℝ := (1 - 2*(-3), 2 - 2*2)

theorem find_k (h : (vector_k_a_plus_b k).fst * (vector_a_minus_2b).snd = (vector_k_a_plus_b k).snd * (vector_a_minus_2b).fst) : k = -1/2 :=
sorry

end find_k_l228_228098


namespace number_of_ways_to_form_valid_5_digit_number_l228_228793

theorem number_of_ways_to_form_valid_5_digit_number :
  (∑ d in (Finset.univ : Finset (Fin 6)), 
    (if d = 0 ∨ d = 5 then 
      5! 
    else 
      0)) / 10 = 216 :=
sorry

end number_of_ways_to_form_valid_5_digit_number_l228_228793


namespace min_value_of_expression_l228_228523

theorem min_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : 
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 :=
by
  sorry

end min_value_of_expression_l228_228523


namespace m_congruent_n_mod_p_l228_228967

variable {p : ℕ} (p_prime : Nat.Prime p) (hp_odd : p % 2 = 1)

noncomputable def S_q (q : ℕ) : ℚ :=
  ∑ k in Finset.range (q // 2), (1 : ℚ) / (2 * (k + 1) * (k + 2) * (k + 3))

theorem m_congruent_n_mod_p :
  let q := (3 * p - 5) // 2,
  let fraction_form := ((1 : ℚ) / p - 2 * S_q q),
  ∃ (m n : ℤ), fraction_form = (m / n) ∧ (m % p = n % p) :=
by sorry

end m_congruent_n_mod_p_l228_228967


namespace solve_circular_region_l228_228172

def circular_region (x y : ℝ) : Prop := x^2 + y^2 = 36

def vertical_line (x : ℝ) : Prop := x = 4
def horizontal_line (y : ℝ) : Prop := y = 3

def region_areas (R1 R2 R3 R4 : ℝ) (h : R1 > R2 ∧ R2 > R3 ∧ R3 > R4) : Prop :=
  R1 - R2 - R3 + R4 = 0

theorem solve_circular_region :
  ∀ R1 R2 R3 R4 : ℝ,
    (∀ x y : ℝ, circular_region x y → vertical_line x ∨ horizontal_line y) →
    R1 > R2 ∧ R2 > R3 ∧ R3 > R4 →
    region_areas R1 R2 R3 R4 :=
by
  intros R1 R2 R3 R4 h1 h2
  sorry

end solve_circular_region_l228_228172


namespace chess_tournament_games_l228_228911

theorem chess_tournament_games (n : ℕ) (h : 2 * 404 = n * (n - 4)) : False :=
by
  sorry

end chess_tournament_games_l228_228911


namespace find_m_l228_228036

def vector (α : Type) := α × α

noncomputable def dot_product {α} [Add α] [Mul α] (a b : vector α) : α :=
a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (a : vector ℝ) (b : vector ℝ) (h₁ : a = (1, 2)) (h₂ : b = (m, 1)) (h₃ : dot_product a b = 0) : 
m = -2 :=
by
  sorry

end find_m_l228_228036


namespace number_of_divisors_300_l228_228796

theorem number_of_divisors_300 :
  let n := 300 in
  let factorization := [2, 2, 3, 5, 5] in
  let p := [2, 3, 5] in
  let e := [2, 1, 2] in
  n = (2 * 2 * 3 * 5 * 5) →
  (e.sum + e.length = 18) :=
by
  intros
  let n := 300
  let factorization := [2, 2, 3, 5, 5]
  let p := [2, 3, 5]
  let e := [2, 1, 2]
  -- additional proof steps would go here
  sorry

end number_of_divisors_300_l228_228796


namespace smallest_n_inequality_l228_228838

def sequence_a : ℕ → ℝ
| 0       := 7
| (n + 1) := sequence_a n * (sequence_a n + 2)

theorem smallest_n_inequality :
  ∃ n > 0, sequence_a n > 4^2018 :=
begin
  sorry
end

end smallest_n_inequality_l228_228838


namespace correct_option_a_l228_228264

theorem correct_option_a (x y a b : ℝ) : 3 * x - 2 * x = x :=
by sorry

end correct_option_a_l228_228264


namespace oil_leak_while_working_l228_228733

theorem oil_leak_while_working:
  ∀ (before working total : ℕ),
  before = 6522 →
  total = 11687 →
  total - before = 5165 :=
by
  intros before total h_before h_total
  rw [h_before, h_total]
  exact rfl

end oil_leak_while_working_l228_228733


namespace initial_jellybeans_l228_228606

theorem initial_jellybeans (J : ℕ) :
    (∀ x y : ℕ, x = 24 → y = 12 →
    (J - x - y + ((x + y) / 2) = 72) → J = 90) :=
by
  intros x y hx hy h
  rw [hx, hy] at h
  sorry

end initial_jellybeans_l228_228606


namespace trivia_team_points_l228_228726

theorem trivia_team_points (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 9 →
  absent_members = 3 →
  total_points = 12 →
  (total_members - absent_members) > 0 →
  total_points / (total_members - absent_members) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  have h5 : (total_members - absent_members) = 6 := by norm_num [h1, h2]
  rw [h5]
  norm_num
  sorry

end trivia_team_points_l228_228726


namespace total_students_l228_228643

theorem total_students (n1 n2 : ℕ) (h1 : (158 - 140)/(n1 + 1) = 2) (h2 : (158 - 140)/(n2 + 1) = 3) :
  n1 + n2 + 2 = 15 :=
sorry

end total_students_l228_228643


namespace f_prime_zero_eq_pow_2_15_l228_228391

noncomputable def geometric_sequence (n : ℕ) (r : ℝ) : ℕ → ℝ
| 0     := 2
| (n+1) := geometric_sequence n r * r

theorem f_prime_zero_eq_pow_2_15 :
  ∃ r : ℝ, (geometric_sequence 7 r = 4) ∧
  let a_i := λ (i : ℕ), geometric_sequence i r in
  let f (x : ℝ) := x * (x - a_i 0) * (x - a_i 1) * (x - a_i 2) * (x - a_i 3) *
                   (x - a_i 4) * (x - a_i 5) * (x - a_i 6) * (x - a_i 7) in
  deriv f 0 = 2 ^ 15 :=
sorry

end f_prime_zero_eq_pow_2_15_l228_228391


namespace circumradius_of_consecutive_triangle_l228_228943

theorem circumradius_of_consecutive_triangle
  (a b c : ℕ)
  (h : a = b - 1)
  (h1 : c = b + 1)
  (r : ℝ)
  (h2 : r = 4)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)
  : ∃ R : ℝ, R = 65 / 8 :=
by {
  sorry
}

end circumradius_of_consecutive_triangle_l228_228943


namespace A_knit_time_l228_228288

def rate_A (x : ℕ) : ℚ := 1 / x
def rate_B : ℚ := 1 / 6

def combined_rate_two_pairs_in_4_days (x : ℕ) : Prop :=
  rate_A x + rate_B = 1 / 2

theorem A_knit_time : ∃ x : ℕ, combined_rate_two_pairs_in_4_days x ∧ x = 3 :=
by
  existsi 3
  -- (Formal proof would go here)
  sorry

end A_knit_time_l228_228288


namespace parametric_equations_l228_228664

variables (t : ℝ)
def x_velocity : ℝ := 9
def y_velocity : ℝ := 12
def init_x : ℝ := 1
def init_y : ℝ := 1

theorem parametric_equations :
  (x = init_x + x_velocity * t) ∧ (y = init_y + y_velocity * t) :=
sorry

end parametric_equations_l228_228664


namespace train_length_proof_l228_228273

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 5 / 18

theorem train_length_proof (speed_kmph : ℕ) (platform_length_m : ℕ) (crossing_time_s : ℕ) (speed_mps : ℕ) (distance_covered_m : ℕ) (train_length_m : ℕ) :
  speed_kmph = 72 →
  platform_length_m = 270 →
  crossing_time_s = 26 →
  speed_mps = convert_kmph_to_mps speed_kmph →
  distance_covered_m = speed_mps * crossing_time_s →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 250 :=
by
  intros h_speed h_platform h_time h_conv h_dist h_train_length
  sorry

end train_length_proof_l228_228273


namespace factorial_ends_with_base_8_zeroes_l228_228874

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228874


namespace exists_four_distinct_natural_numbers_sum_any_three_prime_l228_228638

theorem exists_four_distinct_natural_numbers_sum_any_three_prime :
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ Prime (a + c + d) ∧ Prime (b + c + d)) :=
sorry

end exists_four_distinct_natural_numbers_sum_any_three_prime_l228_228638


namespace lily_remaining_money_l228_228975

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l228_228975


namespace positive_difference_at_y_equals_20_l228_228096

variable l : LineSegment (0, 6) (4, 0)
variable m : LineSegment (0, 3) (8, 0)
variable y : ℝ := 20

noncomputable def positive_difference_x (l m : Line) (y : ℝ) : ℝ :=
  let x_l := (y - l.b) / l.m
  let x_m := (y - m.b) / m.m
  abs (x_l - x_m)

theorem positive_difference_at_y_equals_20 :
  positive_difference_x l m y = 36 :=
by
  sorry

end positive_difference_at_y_equals_20_l228_228096


namespace octagon_area_correct_l228_228690

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228690


namespace sum_squares_ineq_l228_228187

theorem sum_squares_ineq (n : ℕ) (h : 2 ≤ n) (a : Finₓ n → ℝ) :
    ∃ (ε : Finₓ n → ℤ), (∀ i, ε i = 1 ∨ ε i = -1) ∧
    (∑ i, a i) ^ 2 + (∑ i, ε i * a i) ^ 2 ≤ (n + 1) * (∑ i, a i ^ 2) :=
    by
    classical
    let ε := λ i => if i.val < ⌊n / 2⌋ then 1 else -1
    use ε
    split
    intros i
    simp only [ε]
    split_ifs
    exacts [Or.inl rfl, Or.inr rfl]
    sorry

end sum_squares_ineq_l228_228187


namespace mr_wang_returned_to_first_floor_electricity_consumed_correctly_l228_228195

def start_floor : Int := 1
def moves : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : Int := 3 -- each floor is 3 meters high
def electricity_consumption_per_meter : Float := 0.2 -- elevator consumes 0.2 kWh per meter

def final_floor (start : Int) (moves : List Int) : Int :=
  start + moves.sum

def total_distance (moves : List Int) (floor_height : Int) : Int :=
  floor_height * (moves.map Int.natAbs).sum

def total_electricity (distance : Int) (consumption_rate : Float) : Float :=
  distance.toFloat * consumption_rate

theorem mr_wang_returned_to_first_floor : final_floor start_floor moves = 1 := 
by {
  simp [final_floor, start_floor, moves],
  -- This step evaluates to 1 + (sum of moves) = 1 + 0 = 1.
  norm_num,
  -- Basic arithmetic on whole numbers.
  sorry
}

theorem electricity_consumed_correctly : 
  total_electricity (total_distance moves floor_height) electricity_consumption_per_meter = 33.6 :=
by {
  simp [total_distance, total_electricity, moves, floor_height, electricity_consumption_per_meter],
  -- This step translates the sums and multiplications defined earlier.
  norm_num,
  -- Basic arithmetic on whole numbers and floats to show the final answer is 33.6.
  sorry
}

end mr_wang_returned_to_first_floor_electricity_consumed_correctly_l228_228195


namespace remove_two_vertices_eliminate_all_triangles_l228_228957

theorem remove_two_vertices_eliminate_all_triangles {V : Type*} (G : SimpleGraph V) :
  (¬ ∃ (K5 : set V), (K5.card = 5) ∧ (∀ (u v : V), u ∈ K5 → v ∈ K5 → (u = v ∨ G.Adj u v))) → 
  (∀ {T1 T2 : set V}, T1.card = 3 → T2.card = 3 → (∀ (t1 t2 : V), t1 ∈ T1 → t2 ∈ T2 → T1 ≠ T2 → ∃ v, v ∈ T1 ∧ v ∈ T2)) →
  (∃ (v1 v2 : V), ∀ (T : set V), T.card = 3 → (v1 ∈ T ∨ v2 ∈ T) → ¬ ∃ (u w x : V), {u, w, x} = T ∧ G.Adj u w ∧ G.Adj w x ∧ G.Adj x u) :=
by 
  sorry

end remove_two_vertices_eliminate_all_triangles_l228_228957


namespace correct_expression_l228_228633

theorem correct_expression (a b c d : ℝ) : 
  (\sqrt{36} ≠ ± 6) → 
  (\sqrt{(-3)^2} ≠ -3) → 
  (\sqrt{-4} = complex.I * sqrt 4) → 
  (\sqrt[3]{-8} = -2) → 
  d = -2 :=
by 
  intros h1 h2 h3 h4
  exact h4

end correct_expression_l228_228633


namespace royalty_amount_l228_228590

-- Define the conditions and the question proof.
theorem royalty_amount (x : ℝ) :
  (800 ≤ x ∧ x ≤ 4000 → (x - 800) * 0.14 = 420) ∧
  (x > 4000 → x * 0.11 = 420) ∧
  420 = 420 →
  x = 3800 :=
by
  sorry

end royalty_amount_l228_228590


namespace number_of_correct_statements_is_one_l228_228233

-- Defining what each statement means
def statement1 (θ : ℝ) : Prop := 90 < θ ∧ θ < 180 -> ∃ q, q = 2
def statement2 (θ : ℝ) : Prop := θ < 90 -> (θ > 0 ∧ θ < 90)
def statement3 (θ : ℝ) : Prop := θ > 0 ∧ θ < 90 -> θ ≥ 0
def statement4 (θ1 θ2 : ℝ) : Prop := (90 < θ1 ∧ θ1 < 180) → (0 < θ2 ∧ θ2 < 90) → θ1 > θ2

-- The main theorem stating the number of correct statements
theorem number_of_correct_statements_is_one :
  (∀ (θ : ℝ), statement1 θ) ∧
  ∀ (θ : ℝ), ¬statement2 θ ∧
  ∀ (θ : ℝ), ¬statement3 θ ∧
  ∀ (θ1 θ2 : ℝ), ¬statement4 θ1 θ2 → 
  (¬∀ (θ : ℝ), ¬statement1 θ) ∧
  ∀ (θ : ℝ), statement2 θ ∨ statement3 θ ∨ statement4 θ1 θ2 ->
  1 = 1 :=
begin
  sorry
end

end number_of_correct_statements_is_one_l228_228233


namespace lily_remaining_money_l228_228974

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l228_228974


namespace point_on_inverse_proportion_function_l228_228065

variable (k : ℝ) (x y : ℝ)
variable (A : ℝ × ℝ)

def inverse_proportion_function (k : ℝ) (x : ℝ) : ℝ := k / x

theorem point_on_inverse_proportion_function (hA : A = (2, 4))
  (hk : k ≠ 0) :
  inverse_proportion_function k 4 = 2 :=
by
  sorry

end point_on_inverse_proportion_function_l228_228065


namespace pentagon_angle_problem_l228_228932

-- Define the pentagon and its properties
structure RegularPentagon (A B C D E : Type) : Prop :=
(internal_angle_eq : ∀ a : Type, a ∈ {A, B, C, D, E} → ∠A = 108)

-- Define the isosceles triangle DEF 
structure IsoscelesTriangle (D F E : Type) : Prop :=
(equal_sides : DF = EF)
(angle_DFE : ∠DFE = 70)

-- Define the problem statement
theorem pentagon_angle_problem
  (A B C D E F : Type) 
  [reg_pentagon : RegularPentagon A B C D E]
  [isosceles : IsoscelesTriangle D F E] 
  (F_on_BC : F ∈ line_segment(B, C)) :
  ∠BDF = 140 := 
sorry

end pentagon_angle_problem_l228_228932


namespace simplify_expression_l228_228560

theorem simplify_expression (a : ℝ) : 
  (real.sqrt (real.sqrt (a ^ 16)^ (1/4)) + real.sqrt (real.sqrt (a^16) ^ (1/8))) ^ 2 = 4 * a :=
by
  sorry

end simplify_expression_l228_228560


namespace area_of_polygon_PQRSTU_l228_228570

-- Define the points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

noncomputable def area_of_rectangle (a b : ℝ) : ℝ :=
  a * b

-- Define the main problem
theorem area_of_polygon_PQRSTU :
  ∀ (P Q R S T U V : Point),
    dist P Q = 8 →
    dist Q R = 12 →
    dist T U = 7 →
    dist S P = 10 →
    T.y = U.y → -- TU is perpendicular to QR, indicating they are along the same y-line
    Q.y = R.y → Q.x ≠ R.x → -- QR is a horizontal line
    P.y ≠ Q.y → -- PQ is a vertical line
    P.y ≠ S.y → S.x ≠ P.x → -- SP isn't perfect vertical or horizontal
    let V := ⟨P.x, T.y⟩ in -- The point V
    area_of_rectangle (dist P Q) (dist Q R) - area_of_rectangle (dist V U) (dist V T) = 94 :=
by
  intros P Q R S T U V hPQ hQR hTU hSP hTU_perpend hQR_hor hQR_vert_property hPQ_vert_property hSP_different_orientation,
  sorry

end area_of_polygon_PQRSTU_l228_228570


namespace Lakers_win_probability_l228_228568

theorem Lakers_win_probability : 
  let p_Lakers := 2 / 3,
      p_Celtics := 1 / 3 in
  let prob_Lakers_win := 
    ∑ k in (Finset.range 4), 
      (Nat.choose (3 + k) k) * (p_Lakers ^ 4) * (p_Celtics ^ k) in
  Float.round (prob_Lakers_win * 100) = 83 :=
by
  sorry

end Lakers_win_probability_l228_228568


namespace range_a_l228_228904

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x + 2 / x - a
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 2 / x

theorem range_a (a : ℝ) : (∃ x > 0, f x a = 0) → a ≥ 3 :=
by
sorry

end range_a_l228_228904


namespace identical_solutions_k_zero_l228_228791

theorem identical_solutions_k_zero (k : ℝ) :
  (∀ x : ℝ, x^2 = 3 * x^2 + k) → k = 0 :=
by {
  intro h,
  have eq_zero : ∀ x : ℝ, x^2 = 3 * x^2 + k ↔ x = 0,
  {
    intro x,
    split,
    {
      intro hx,
      have : x^2 - 3 * x^2 = k, { rw [hx] },
      rw [←sub_eq_zero, sub_self] at this,
      exact this,
    },
    {
      intro hx0,
      rw hx0,
      ring,
    }
  },
  specialize h 0,
  rw eq_zero 0 at h,
  exact h,
}

end identical_solutions_k_zero_l228_228791


namespace tan_double_angle_l228_228110

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end tan_double_angle_l228_228110


namespace delta_abc_length_cb_l228_228475

theorem delta_abc_length_cb (CD DA CE CB : ℝ) (h_parallel : ∥DE ∥ ∥AB) 
 (h_cd : CD = 5) (h_da : DA = 15) (h_ce : CE = 10) :
 CB = 40 := by
    sorry

end delta_abc_length_cb_l228_228475


namespace find_point_p_coords_l228_228817

structure Point where
  x : ℝ
  y : ℝ

def midpoint (P1 P2 : Point) : Point :=
  { x := (P1.x + P2.x) / 2, y := (P1.y + P2.y) / 2 }

def P1 := { x := -1, y := 2 }
def P2 := { x := 3, y := 0 }

theorem find_point_p_coords (P : Point) 
  (h : P = midpoint P1 P2) : P = { x := 1, y := 1 } := 
  by sorry

end find_point_p_coords_l228_228817


namespace vector_combination_unique_intersection_l228_228149

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C : V)
def pointOnSegment (P X Y : V) (m n : ℝ) : Prop := P = (m / (m + n)) • X + (n / (m + n)) • Y

theorem vector_combination_unique_intersection 
  (G H Q : V)
  (hG : pointOnSegment G A B 3 2)
  (hH : pointOnSegment H B C 2 3) 
  (hQ : ∃ (u v w : ℝ), u + v + w = 1 ∧ Q = u • A + v • B + w • C) :
  Q = A :=
begin
  sorry
end

end vector_combination_unique_intersection_l228_228149


namespace volume_of_region_l228_228786

noncomputable def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  ∀ x y z : ℝ, f(x, y, z) ≤ 6 → volume_of_region ≤ 18 := 
by
  sorry

end volume_of_region_l228_228786


namespace sum_of_first_2010_terms_l228_228394

noncomputable def sequence (a : ℝ) (n : ℕ) : ℝ :=
  match n % 3 with
  | 0 => 1
  | 1 => a
  | 2 => abs (a - 1)
  | _ => 0  -- this should never happen

theorem sum_of_first_2010_terms (a : ℝ) (h1 : a ≤ 1) (h2 : a ≠ 0) :
    (Finset.range 2010).sum (λ n => sequence a n) = 1340 :=
by
  sorry

end sum_of_first_2010_terms_l228_228394


namespace range_of_a_l228_228470

theorem range_of_a : 
  (∃ a : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (x^2 + a ≤ a*x - 3))) ↔ (a ≥ 7) :=
sorry

end range_of_a_l228_228470


namespace sequence_properties_l228_228048

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℝ := (1 / 2)^n

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (2 * k - 1) * (1 / 2)^k

theorem sequence_properties (n : ℕ) (hn : n ∈ ℕ):
  a_n n = 2 * n - 1 ∧
  b_n n = (1 / 2)^n ∧
  T_n n = 3 - (2 * n + 3) * (1 / 2)^n :=
by
  -- Sequence definition
  sorry

-- To verify imports properly work and coding style
#print axioms sequence_properties

end sequence_properties_l228_228048


namespace triangle_equilateral_if_condition_l228_228150

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition_l228_228150


namespace breaststroke_hours_correct_l228_228926

namespace Swimming

def total_required_hours : ℕ := 1500
def backstroke_hours : ℕ := 50
def butterfly_hours : ℕ := 121
def monthly_freestyle_sidestroke_hours : ℕ := 220
def months : ℕ := 6

def calculated_total_hours : ℕ :=
  backstroke_hours + butterfly_hours + (monthly_freestyle_sidestroke_hours * months)

def remaining_hours_to_breaststroke : ℕ :=
  total_required_hours - calculated_total_hours

theorem breaststroke_hours_correct :
  remaining_hours_to_breaststroke = 9 :=
by
  sorry

end Swimming

end breaststroke_hours_correct_l228_228926


namespace probability_of_correct_match_l228_228296

theorem probability_of_correct_match :
  let n := 3
  let total_arrangements := Nat.factorial n
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = ((1: ℤ) / 6) :=
by
  sorry

end probability_of_correct_match_l228_228296


namespace phosphorus_symbol_l228_228777

def atomic_weight_Al := 26.98
def atomic_weight_P := 30.97
def atomic_weight_O := 16.00
def mol_weight := 122

def symbol_element_P := "P"

theorem phosphorus_symbol (element : String) (al : Float) (p : Float) (o : Float) (mw : Float) : 
  al = atomic_weight_Al → 
  p = atomic_weight_P → 
  o = atomic_weight_O → 
  mw = mol_weight → 
  element = symbol_element_P :=
by
  intros h_al h_p h_o h_mw
  sorry

end phosphorus_symbol_l228_228777


namespace m_range_l228_228841

open Real

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem: m must belong to the interval [-4, 5]
theorem m_range (m : ℝ) : (line_eq A.1 A.2 m) → (line_eq B.1 B.2 m) → -4 ≤ m ∧ m ≤ 5 := 
sorry

end m_range_l228_228841


namespace intersection_M_N_l228_228812

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

-- The theorem stating that the intersection of M and N is {1, 5}
theorem intersection_M_N :
  M ∩ N = {1, 5} :=
  sorry

end intersection_M_N_l228_228812


namespace raise_salary_to_original_l228_228640

/--
The salary of a person was reduced by 25%. By what percent should his reduced salary be raised
so as to bring it at par with his original salary?
-/
theorem raise_salary_to_original (S : ℝ) (h : S > 0) :
  ∃ P : ℝ, 0.75 * S * (1 + P / 100) = S ∧ P = 33.333333333333336 :=
sorry

end raise_salary_to_original_l228_228640


namespace probability_ends_up_multiple_of_4_l228_228944

-- Define the sample space of card numbers and spinner outcomes
def card_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def spinner_outcomes := {move_2_left, move_2_right, move_1_left, move_1_left}

-- Define moves
def move (position : ℤ) (outcome : {move_2_left, move_2_right, move_1_left}) : ℤ :=
  match outcome with
  | move_2_left  => position - 2
  | move_2_right => position + 2
  | move_1_left  => position - 1

-- Define multiple of 4 check
def is_multiple_of_4 (n : ℤ) : Prop := n % 4 = 0

-- The probability function to be proved
theorem probability_ends_up_multiple_of_4 :
  ∀ (starting_point : ℤ),
  starting_point ∈ card_numbers →
  let final_point := move (move starting_point (spinner_outcomes)) (spinner_outcomes) in
  (∃ (p : ℚ), p = 35 / 320 ∧ (is_multiple_of_4 final_point)) :=
by
  sorry

end probability_ends_up_multiple_of_4_l228_228944


namespace other_asymptote_l228_228542

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end other_asymptote_l228_228542


namespace spoonfuls_per_bowl_l228_228533

theorem spoonfuls_per_bowl
  (clusters_per_spoonful : ℕ) 
  (total_clusters : ℕ) 
  (bowls_per_box : ℕ) 
  (clusters_per_spoonful_eq : clusters_per_spoonful = 4) 
  (total_clusters_eq : total_clusters = 500) 
  (bowls_per_box_eq : bowls_per_box = 5) 
  : (total_clusters / bowls_per_box) / clusters_per_spoonful = 25 :=
by
  rw [clusters_per_spoonful_eq, total_clusters_eq, bowls_per_box_eq]
  sorry

end spoonfuls_per_bowl_l228_228533


namespace regression_lines_intersect_at_mean_l228_228611

variables (A_exp B_exp : ℕ) (t₁ t₂ : ℝ → ℝ) (s t : ℝ)

-- Conditions
def conducted_experiments : A_exp = 100 ∧ B_exp = 150 := by sorry
def regression_lines_are (r : ℝ → ℝ) := ∃ data_x data_y : List ℝ, r = fun x => x * (mean (List.zip data_x data_y).map Prod.fst) + t
def average_x_equals : ∀ (data_x : List ℝ), mean data_x = s := by sorry
def average_y_equals : ∀ (data_y : List ℝ), mean data_y = t := by sorry

-- To be proved
theorem regression_lines_intersect_at_mean :
  (regression_lines_are t₁ ∧ regression_lines_are t₂) →
  (average_x_equals (data_x : List ℝ) ∧ average_y_equals (data_y : List ℝ)) →
  t₁ s = t ∧ t₂ s = t → t₁ s = t₂ s :=
by sorry

end regression_lines_intersect_at_mean_l228_228611


namespace part_a_part_b_l228_228969

-- Part (a)
theorem part_a (A : Type) [ring A] (N U : set A) (Z : set A) (h1 : ∀ x : A, x ∈ Z) 
  (h2 : ∀x : A, (∃ n : ℕ, x ^ n = 0) ↔ x ∈ N) (h3 : ∀ x : A, (∃ y : A, x * y = 1) ↔ x ∈ U) :
  N + U = U := 
sorry

-- Part (b)
theorem part_b (A : Type) [ring A] (U : set A) (N : set A) (a : A) (h1 : finite A) 
  (h2 : ∀ x : A, x ∈ U ↔ (∃ y : A, x * y = 1))
  (h3 : ∀ u : A, u + a ∈ U → u ∈ U) : 
  a ∈ N := 
sorry

end part_a_part_b_l228_228969


namespace factorial_base_8_zeroes_l228_228853

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228853


namespace factorial_ends_with_base_8_zeroes_l228_228875

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228875


namespace range_of_m_shift_l228_228583

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem range_of_m_shift (m : ℝ) (h₀ : 0 < m) (h₁ : m < π) :
  isMonotonicIncreasing (λ x, 2 * Real.cos (x - m - π / 3)) (π/6) (5*π/6) → (π/2) ≤ m ∧ m ≤ (5*π/6) :=
sorry

end range_of_m_shift_l228_228583


namespace perpendicular_distance_from_C_to_circle_l228_228540

-- Define the parameters of the problem
def radius : ℝ := 30
def AB : ℝ := 16
def AC : ℝ := AB / 2 -- since C is midpoint of AB

-- Define the theorem statement
theorem perpendicular_distance_from_C_to_circle :
  let R : ℝ := radius
      A := 30
      B := 16
      C := AC
  in (A^2 - C^2) = 836 :=
by
  -- Unfolding the definitions and proving the distance calculation mathematically.
  sorry

end perpendicular_distance_from_C_to_circle_l228_228540


namespace range_of_k_l228_228039

open BigOperators

theorem range_of_k
  {f : ℝ → ℝ}
  (k : ℝ)
  (h : ∀ x : ℝ, f x = 32 * x - (k + 1) * 3^x + 2)
  (H : ∀ x : ℝ, f x > 0) :
  k < 1 /2 := 
sorry

end range_of_k_l228_228039


namespace octagon_area_l228_228699

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228699


namespace range_of_k_l228_228081

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(-k^2 + k + 2)

theorem range_of_k (k : ℝ) : (∃ k, (f 2 k < f 3 k)) ↔ (-1 < k) ∧ (k < 2) :=
by
  sorry

end range_of_k_l228_228081


namespace circle_center_radius_sum_l228_228348

theorem circle_center_radius_sum (u v s : ℝ) (h1 : (x + 4)^2 + (y - 1)^2 = 13)
    (h2 : (u, v) = (-4, 1)) (h3 : s = Real.sqrt 13) : 
    u + v + s = -3 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l228_228348


namespace probability_of_no_shaded_square_l228_228286

noncomputable def rectangles_without_shaded_square_probability : ℚ :=
  let n := 502 * 1003
  let m := 502 ^ 2
  1 - (m : ℚ) / n 

theorem probability_of_no_shaded_square : rectangles_without_shaded_square_probability = 501 / 1003 :=
  sorry

end probability_of_no_shaded_square_l228_228286


namespace trailing_zeroes_in_500_factorial_l228_228746

theorem trailing_zeroes_in_500_factorial : ∀ n = 500, (∑ k in range (nat.log 5 500 + 1), 500 / 5^k) = 124 :=
by
  sorry

end trailing_zeroes_in_500_factorial_l228_228746


namespace minimum_difference_among_sequence_l228_228952

noncomputable def sequence_condition (a : ℕ → ℤ) :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2016 → a (n + 5) + a n > a (n + 2) + a (n + 3)

noncomputable def sequence_max_min_diff (a : ℕ → ℤ) :=
  (list.maximum (list.of_fn (λ i, a (i + 1)))) - (list.minimum (list.of_fn (λ i, a (i + 1))))

theorem minimum_difference_among_sequence (a : ℕ → ℤ) (h : sequence_condition a) :
  sequence_max_min_diff a = 85008 :=
sorry

end minimum_difference_among_sequence_l228_228952


namespace first_prize_ticket_numbers_l228_228723

theorem first_prize_ticket_numbers :
  {n : ℕ | n < 10000 ∧ (n % 1000 = 418)} = {418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by
  sorry

end first_prize_ticket_numbers_l228_228723


namespace ratio_of_perimeters_l228_228283

variables {α : Type*}

open Classical

-- Define the similarity condition
def similar_triangles (ABC DEF : α) (r : ℝ) := 
  ∀ (a b c d e f : ℝ), ABC = (a, b, c) ∧ DEF = (d, e, f) →
    (a/d) = r ∧ (b/e) = r ∧ (c/f) = r

-- Define the triangles and their perimeters
variables (triangle_ABC triangle_DEF : α)
variables (P_ABC P_DEF : ℝ)

-- Now we state the theorem
theorem ratio_of_perimeters (h : similar_triangles triangle_ABC triangle_DEF (1/4))
  (h_PABC : (P_ABC = P_ABC)) (h_PDEF : (P_DEF = P_DEF)) :
  P_ABC / P_DEF = 1/4 := sorry

end ratio_of_perimeters_l228_228283


namespace simplify_expression_l228_228196

-- Define the main theorem
theorem simplify_expression 
  (a b x : ℝ) 
  (hx : x = 1 / a * Real.sqrt ((2 * a - b) / b))
  (hc1 : 0 < b / 2)
  (hc2 : b / 2 < a)
  (hc3 : a < b) : 
  (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1 :=
sorry

end simplify_expression_l228_228196


namespace find_m_plus_n_l228_228493

-- Definitions
structure Triangle (A B C P M N : Type) :=
  (midpoint_AD_P : P)
  (intersection_M_AB : M)
  (intersection_N_AC : N)
  (vec_AB : ℝ)
  (vec_AM : ℝ)
  (vec_AC : ℝ)
  (vec_AN : ℝ)
  (m : ℝ)
  (n : ℝ)
  (AB_eq_AM_mul_m : vec_AB = m * vec_AM)
  (AC_eq_AN_mul_n : vec_AC = n * vec_AN)

-- The theorem to prove
theorem find_m_plus_n (A B C P M N : Type)
  (t : Triangle A B C P M N) :
  t.m + t.n = 4 :=
sorry

end find_m_plus_n_l228_228493


namespace abc_equilateral_l228_228938

variable (A B C D E F S : Type)
variable [triangle ABC : Triangle A B C]
variable [midpoint D BC : Midpoint D B C]
variable [midpoint E CA : Midpoint E C A]
variable [midpoint F AB : Midpoint F A B]
variable [centroid S ABC : Centroid S A B C]
variable [equal_perimeters AFS BDS CES : EqualPerimeters A F S B D S C E S]

theorem abc_equilateral 
  (h1 : Midpoint D B C) 
  (h2 : Midpoint E C A)
  (h3 : Midpoint F A B)
  (h4 : Centroid S A B C) 
  (h5 : EqualPerimeters (Triangle A F S) (Triangle B D S) (Triangle C E S)) : 
  Equilateral ABC :=
sorry

end abc_equilateral_l228_228938


namespace xyz_eq_neg10_l228_228962

noncomputable def complex_numbers := {z : ℂ // z ≠ 0}

variables (a b c x y z : complex_numbers)

def condition1 := a.val = (b.val + c.val) / (x.val - 3)
def condition2 := b.val = (a.val + c.val) / (y.val - 3)
def condition3 := c.val = (a.val + b.val) / (z.val - 3)
def condition4 := x.val * y.val + x.val * z.val + y.val * z.val = 9
def condition5 := x.val + y.val + z.val = 6

theorem xyz_eq_neg10 (a b c x y z : complex_numbers) :
  condition1 a b c x ∧ condition2 a b c y ∧ condition3 a b c z ∧
  condition4 x y z ∧ condition5 x y z → x.val * y.val * z.val = -10 :=
by sorry

end xyz_eq_neg10_l228_228962


namespace sequence_ln_l228_228491

theorem sequence_ln (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 + Real.log n := 
sorry

end sequence_ln_l228_228491


namespace limit_S_n_div_a_n_a_n_plus_1_l228_228818

def a_seq (n : ℕ) (a1 : ℤ) : ℤ := a1 + (n - 1) * 2

def S_n (n : ℕ) (a1 : ℤ) : ℤ := n * (a1 + (n - 1) * 2 / 2)

theorem limit_S_n_div_a_n_a_n_plus_1 (a1 : ℤ) :
  filter.at_top.lim (λ n : ℕ, (S_n n a1 : ℚ) / ((a_seq n a1 : ℚ) * (a_seq (n+1) a1 : ℚ))) = 1 / 4 :=
by 
  sorry

end limit_S_n_div_a_n_a_n_plus_1_l228_228818


namespace video_game_cost_l228_228527

theorem video_game_cost :
  let september_saving : ℕ := 50
  let october_saving : ℕ := 37
  let november_saving : ℕ := 11
  let mom_gift : ℕ := 25
  let remaining_money : ℕ := 36
  let total_savings : ℕ := september_saving + october_saving + november_saving
  let total_with_gift : ℕ := total_savings + mom_gift
  let game_cost : ℕ := total_with_gift - remaining_money
  game_cost = 87 :=
by
  sorry

end video_game_cost_l228_228527


namespace only_negative_is_neg1_l228_228578

theorem only_negative_is_neg1 (a b c d : Int) (h_a : a = -1) (h_b : b = 0) (h_c : c = 1) (h_d : d = 2) :
  (∀ x ∈ {a, b, c, d}, x < 0 ↔ x = a) :=
by
  sorry

end only_negative_is_neg1_l228_228578


namespace number_of_correct_statements_is_one_l228_228731

theorem number_of_correct_statements_is_one :
  (¬ (∀ x, x^2 + x - 2 > 0 → x > 1)) ∧
  (¬ (∀ x ∈ ℝ, sin x ≤ 1) = ∃ x₀ ∈ ℝ, sin x₀ > 1) ∧
  (¬ (tan (π / 4) = 1 → ∀ x, tan x = 1 → x = π / 4)) ∧
  (∀ f : ℝ → ℝ, odd_function f → ¬ (f (log 3 2) + f (log 2 3) = 0)) →
  1 = 1 :=
by
  sorry

end number_of_correct_statements_is_one_l228_228731


namespace jack_total_plates_after_smashing_and_buying_l228_228495

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1
def new_polka_dotted_plates : ℕ := initial_checked_plates * initial_checked_plates

theorem jack_total_plates_after_smashing_and_buying : 
  initial_flower_plates - smashed_flower_plates
  + initial_checked_plates
  + initial_striped_plates - smashed_striped_plates
  + new_polka_dotted_plates = 96 := 
by {
  -- calculation proof here
  sorry
}

end jack_total_plates_after_smashing_and_buying_l228_228495


namespace friend_spent_11_l228_228269

-- Definitions of the conditions
def total_lunch_cost (you friend : ℝ) : Prop := you + friend = 19
def friend_spent_more (you friend : ℝ) : Prop := friend = you + 3

-- The theorem to prove
theorem friend_spent_11 (you friend : ℝ) 
  (h1 : total_lunch_cost you friend) 
  (h2 : friend_spent_more you friend) : 
  friend = 11 := 
by 
  sorry

end friend_spent_11_l228_228269


namespace find_function_and_interval_l228_228432

theorem find_function_and_interval
  (A : ℝ)
  (ω : ℝ)
  (φ : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : |φ| < π / 4)
  (hP : (π / 12, 0) ∈ { p : ℝ × ℝ | p.snd = A * Real.sin (ω * p.fst + φ) })
  (hQ : (π / 3, 5) ∈ { q : ℝ × ℝ | q.snd = A * Real.sin (ω * q.fst + φ) ∧ q.snd = A })
  : ∃ (A ω φ : ℝ), 
    A = 5 ∧ ω = 2 ∧ φ = -π / 6 ∧ 
    (∀ (x : ℝ), 5 * Real.sin (2*x - π / 6) = 5 * Real.sin (2*x - π / 6)) ∧
    (∀ (k : ℤ), 
      (let interval := λ x: ℝ, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π in
      interval ⌊sorry unlikely to be 10⌋)) :=
sorry

end find_function_and_interval_l228_228432


namespace angle_in_second_quadrant_l228_228450

open Real

-- Define the fourth quadrant condition
def isFourthQuadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * π - π / 2 < α ∧ α < 2 * k * π

-- Define the second quadrant condition
def isSecondQuadrant (β : ℝ) (k : ℤ) : Prop :=
  2 * k * π + π / 2 < β ∧ β < 2 * k * π + π

-- The main theorem to prove
theorem angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  isFourthQuadrant α k → isSecondQuadrant (π + α) k :=
sorry

end angle_in_second_quadrant_l228_228450


namespace area_of_AGB_l228_228937

open Real

-- Definitions based on the given conditions
variable (ABC : Triangle)
variable (AD CE : Real)
variable (AB : Real)

-- Conditions given in the problem
axiom h1 : AD = 15
axiom h2 : CE = 24
axiom h3 : AB = 30

-- To prove
theorem area_of_AGB : area (AGB) = 30 * sqrt 21 :=
  sorry

end area_of_AGB_l228_228937


namespace OP_perp_AP_l228_228486

variables {A B C O D E F M N P : Type*}
variables [has_point A] [has_point B] [has_point C] [has_point O] [has_point D] [has_point E] [has_point F] [has_point M] [has_point N] [has_point P]

-- Given conditions
variable (ABC_obtuse : obtuse_triangle A B C)
variable (AB_gt_AC : A.B.dist > A.C.dist)
variable (O_circumcenter : circumcenter O A B C)
variable (midpoint_D : midpoint D B C)
variable (midpoint_E : midpoint E C A)
variable (midpoint_F : midpoint F A B)
variable (median_AD : median A D)
variable (intersection_M : intersection M A D F O)
variable (intersection_N : intersection N A D E O)
variable (meet_BM_CN_P : meet B M C N P)

-- Prove that \( OP \perp AP \)
theorem OP_perp_AP : orthogonal O P A P :=
by
  sorry

end OP_perp_AP_l228_228486


namespace part_I_part_II_l228_228524

noncomputable def f (a x : ℝ) : ℝ :=
(ax^2 - (4 * a + 1) * x + 4 * a + 3) * Real.exp x

def tangent_parallel (a : ℝ) : Prop :=
  deriv (f a) 1 = 0

theorem part_I (a : ℝ) : tangent_parallel a ↔ a = 1 :=
sorry

noncomputable def f' (a x : ℝ) : ℝ :=
((ax^2 - (2 * a + 1) * x + 2) * Real.exp x)

def local_minimum_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, f' a x = 0 → x = 2

theorem part_II (a : ℝ) : local_minimum_condition a ↔ a > 1 / 2 :=
sorry

end part_I_part_II_l228_228524


namespace probability_exactly_M_laws_in_concept_expected_laws_considered_in_concept_l228_228129

-- Define the parameters
variables (K N M : ℕ) (p : ℝ)
noncomputable def q := 1 - (1 - p)^N

-- Theorem (a): Probability that exactly M laws are considered
theorem probability_exactly_M_laws_in_concept (hK: 0 < K) (hN: 0 < N) (hp: 0 < p) :
  let binom_coeff := Nat.choose K M in
  binom_coeff * q ^ M * (1 - q) ^ (K - M) = 
  binom_coeff * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) :=
by sorry

-- Theorem (b): Expected number of laws considered
theorem expected_laws_considered_in_concept (hK: 0 < K) (hN: 0 < N) (hp: 0 < p) :
  K * (1 - (1 - p) ^ N) =
  K * q :=
by sorry

end probability_exactly_M_laws_in_concept_expected_laws_considered_in_concept_l228_228129


namespace parabola_tangent_min_dot_product_l228_228833

theorem parabola_tangent_min_dot_product :
  ∃ (M : Point), ∀ (A B : Point),
  (onParabola A ∧ onParabola B) ∧
  (tangentAt M A C) ∧ (tangentAt M B C) ∧
  isOnNegativeXAxis M →
  dotProduct (vectorFrom M A) (vectorFrom M B) = -1/16 := 
by
  sorry

end parabola_tangent_min_dot_product_l228_228833


namespace factorial_trailing_zeros_base_8_l228_228868

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228868


namespace problem_l228_228099

open Function

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (Prod.fst v ^ 2 + Prod.snd v ^ 2)

def a : ℝ × ℝ := (1, 2)
def b (t : ℝ) : ℝ × ℝ := (-2, t)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem problem (t : ℝ) (h : is_parallel a (b t)) : vector_magnitude (a.1 + 2 * fst (b t), a.2 + 2 * snd (b t)) = 3 * Real.sqrt 5 := by
  have ht : t = -4 := by
    rw [is_parallel] at h
    simp at h
    linarith
  rw [ht]
  simp
  sorry

end problem_l228_228099


namespace probability_three_pairs_six_dice_correct_l228_228010

noncomputable def probability_three_pairs_six_dice : ℚ :=
  let total_outcomes := 46656
  let successful_outcomes := 1800
  successful_outcomes / total_outcomes

theorem probability_three_pairs_six_dice_correct :
  probability_three_pairs_six_dice = 25 / 648 :=
by
  sorry

end probability_three_pairs_six_dice_correct_l228_228010


namespace find_f_prime_at_1_l228_228190

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x then
    Math.log x + x
  else
    0  -- This is just a placeholder as f is noncomputable on x ≤ 0

theorem find_f_prime_at_1 :
  ∀ f : ℝ → ℝ, 
  (∀ x : ℝ, 0 < x → f (Real.exp x) = x + Real.exp x) → 
  differentiable ℝ f → 
  deriv f 1 = 2 :=
by
  sorry

end find_f_prime_at_1_l228_228190


namespace prob_at_least_3_out_of_6_open_eyes_eq_233_div_729_l228_228484

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def prob_at_least_3_babies_open_eyes (p : ℚ) (n k : ℕ) : ℚ :=
  let q := 1 - p
  let prob0 := q^n
  let prob1 := binomial_coeff n 1 * p * q^(n-1)
  let prob2 := binomial_coeff n 2 * p^2 * q^(n-2)
  1 - (prob0 + prob1 + prob2)

theorem prob_at_least_3_out_of_6_open_eyes_eq_233_div_729 :
  prob_at_least_3_babies_open_eyes (1/3) 6 3 = 233/729 :=
by
  sorry

end prob_at_least_3_out_of_6_open_eyes_eq_233_div_729_l228_228484


namespace water_leaked_l228_228947

theorem water_leaked (initial remaining : ℝ) (h_initial : initial = 0.75) (h_remaining : remaining = 0.5) :
  initial - remaining = 0.25 :=
by
  sorry

end water_leaked_l228_228947


namespace algorithm_definiteness_l228_228572

-- Define the condition of the problem
def isUniquelyDetermined (algorithm : Type) : Prop :=
  ∀ (steps : algorithm), steps are uniquely determined ∧ steps are not ambiguous ∧ steps do not have multiple possibilities

-- Define the property of definiteness
def definiteness (algorithm : Type) : Prop :=
  ∀ (steps : algorithm), steps are definitive

-- State the theorem
theorem algorithm_definiteness (alg : Type) (h : isUniquelyDetermined alg) : definiteness alg :=
  sorry

end algorithm_definiteness_l228_228572


namespace area_BDE_l228_228949

-- Definitions based on conditions
def TriangleABC (A B C : ℝ × ℝ) : Prop := 
  dist A B = 3 ∧ dist B C = 4 ∧ dist C A = 5

-- Line through A perpendicular to AC intersects BC at D
def PerpendicularThroughA (A C D : ℝ × ℝ) : Prop := 
  ∃ (l : ℝ), D.2 = l * D.1 ∧ D.1 ≠ 4 ∧ (l * D.1 + 3 = 0)

-- Line through C perpendicular to AC intersects AB at E
def PerpendicularThroughC (A B C E : ℝ × ℝ) : Prop := 
  ∃ (m : ℝ), E.2 = (m * E.1 - 3) ∧ E.1 = 0

-- Area calculation using determinant formula
def area_triangle (B D E : ℝ × ℝ) : ℝ := 
  1/2 * abs (B.1 * (D.2 - E.2) + D.1 * (E.2 - B.2) + E.1 * (B.2 - D.2))

-- Theorem we want to prove
theorem area_BDE (A B C D E : ℝ × ℝ) 
  (hABC : TriangleABC A B C)
  (hPerpA : PerpendicularThroughA A C D)
  (hPerpC : PerpendicularThroughC A B C E) :
  area_triangle B D E = 27 / 8 := by
  sorry

end area_BDE_l228_228949


namespace smallest_n_2008_divides_an_l228_228180

open Nat

noncomputable def a : ℕ → ℤ
  | 1     => 0
  | (n+1) => if 1 ≤ n then ((n - 1) * (a n) - 2 * (n - 1)) / (n + 1) else 0

theorem smallest_n_2008_divides_an (n : ℕ) (h : ∀ n > 0, (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) (h2007 : 2008 ∣ a 2007) : 
  ∃ m : ℕ, m ≥ 2 ∧ 2008 ∣ a m ∧ ∀ k : ℕ, 2 ≤ k < m → ¬ (2008 ∣ a k) := 
by
  sorry

end smallest_n_2008_divides_an_l228_228180


namespace left_vertex_of_ellipse_l228_228050

theorem left_vertex_of_ellipse : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 8 = 0 ∧ x = a - 5) ∧
  2 * b = 8 → left_vertex = (-5, 0) :=
sorry

end left_vertex_of_ellipse_l228_228050


namespace oil_leak_during_work_l228_228735

/-- 
  Given two amounts of oil leakage, one before engineering work started 
  and the total leakage, this theorem calculates the leakage during the work.
-/
def leakage_while_working (initial_leak total_leak : ℕ) : ℕ :=
  total_leak - initial_leak

theorem oil_leak_during_work (initial_leak total_leak expected_leak : ℕ) 
  (h1 : initial_leak = 6522)
  (h2 : total_leak = 11687)
  (h3 : expected_leak = 5165) : 
  leakage_while_work initial_leak total_leak = expected_leak :=
by 
  rw [h1, h2, h3]
  sorry

end oil_leak_during_work_l228_228735


namespace tangent_line_proof_find_point_B_fixed_point_line_AB_l228_228084

-- Define the parabola structure
structure Parabola (p : ℝ) (hp : 0 < p) :=
  (equation : ∀ x y : ℝ, y^2 = 2 * p * x)

-- Definition of point D
structure PointOnDirectrix (p x₀ y₀ : ℝ) :=
  (h₀ : y₀^2 > 2 * p * x₀)

-- Part (1): Proving the Tangency
theorem tangent_line_proof (p : ℝ) (hp : 0 < p) (x₁ y₁ : ℝ) 
  (hₓ : ∀ x y : ℝ, Parabola p hp) : 
  ∀ (x : ℝ) (y : ℝ), y * y₁ = p * (x + x₁) → y^2 = 2 * p * x :=
sorry

-- Part (2): Finding Coordinates of Point B
theorem find_point_B (p : ℝ) (hp : 0 < p) : 
  let A := (4, 4) in
  ∃ (x₂ y₂ : ℝ), x₂ = 1 / 4 ∧ y₂ = -1 :=
sorry

-- Part (3): Fixed Point On Line AB
theorem fixed_point_line_AB (p x₀ y₀ : ℝ) 
  (h_on_directrix : PointOnDirectrix p x₀ y₀) :
  let D := (-p, y₀) in
  ∀ (x₁ y₁ x₂ y₂ : ℝ), y₁ = 4 ∧ x₁ = 4 → 
  ∃ (fixed_pt : (ℝ × ℝ)), fixed_pt = (p, 0) :=
sorry

end tangent_line_proof_find_point_B_fixed_point_line_AB_l228_228084


namespace student_age_is_17_in_1960_l228_228327

noncomputable def student's_age_in_1960 (x y : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) : ℕ := 
  let birth_year : ℕ := 1900 + 10 * x + y
  let age_in_1960 : ℕ := 1960 - birth_year
  age_in_1960

theorem student_age_is_17_in_1960 :
  ∃ x y : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ (1960 - (1900 + 10 * x + y) = 1 + 9 + x + y) ∧ (1960 - (1900 + 10 * x + y) = 17) :=
by {
  sorry -- Proof goes here
}

end student_age_is_17_in_1960_l228_228327


namespace pyramid_surface_area_l228_228571

-- Define parameters for the base isosceles trapezoid
variable (a α β : Real) 

-- Define the main theorem statement
theorem pyramid_surface_area (hα : α > 0) (hβ : β > 0) (ha : a > 0) :
    let S_total := (2 * a^2 * Real.sin α * Real.cos (β / 2)^2) / (Real.cos β)
  in 0 < S_total := 
sorry

end pyramid_surface_area_l228_228571


namespace best_model_is_A_l228_228131

-- Definitions of the models and their R^2 values
def ModelA_R_squared : ℝ := 0.95
def ModelB_R_squared : ℝ := 0.81
def ModelC_R_squared : ℝ := 0.50
def ModelD_R_squared : ℝ := 0.32

-- Definition stating that the best fitting model is the one with the highest R^2 value
def best_fitting_model (R_squared_A R_squared_B R_squared_C R_squared_D: ℝ) : Prop :=
  R_squared_A > R_squared_B ∧ R_squared_A > R_squared_C ∧ R_squared_A > R_squared_D

-- Proof statement
theorem best_model_is_A : best_fitting_model ModelA_R_squared ModelB_R_squared ModelC_R_squared ModelD_R_squared :=
by
  -- Skipping the proof logic
  sorry

end best_model_is_A_l228_228131


namespace true_statements_l228_228345

-- Define the conditions as propositions
def Statement1 : Prop :=
  ∃ p : Prism, ∃ plane : Plane, ¬(ResultingPartsArePrisms (intersect p plane))

def Statement2 : Prop :=
  ∀ s : GeometricSolid, (TwoParallelFaces s ∧ AllOtherFacesParallelograms s) → (IsPrism s)

def Statement3 : Prop :=
  ∀ p : Polyhedron, (HasOnePolygonFace p ∧ AllOtherFacesTriangles p) → (IsAlwaysPyramid p)

def Statement4 : Prop :=
  ∀ s : Sphere, ∀ plane : Plane, (distance (center s) plane < radius s) → (IsIntersectionCircle s plane)

-- The proof problem to determine the true statements
theorem true_statements :
  Statement1 ∧ Statement4 :=
by {
  -- We will use sorry to omit the proof steps
  sorry
}

end true_statements_l228_228345


namespace max_value_AC_plus_BC_squared_l228_228284

theorem max_value_AC_plus_BC_squared {r : ℝ} (h : r > 0) :
  ∀ (A B C : EuclideanSpace ℝ (Fin 2)),
  (dist A B = 2 * r) ∧ (dist A C ≤ r) ∧ (dist B C ≤ r) ∧
  (∃ O : EuclideanSpace ℝ (Fin 2), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = r) →
  (AC : ℝ) (BC : ℝ),
  (AC + BC) ≤ 4 * r :=
begin
  sorry
end

end max_value_AC_plus_BC_squared_l228_228284


namespace geometric_series_sum_equals_3_6_l228_228515

-- Define the series parameters
def a : ℝ := 6
def r : ℝ := -2 / 3

-- Define the expected sum of the geometric series
def expected_sum : ℝ := 3.6

-- The theorem to be proven
theorem geometric_series_sum_equals_3_6 :
  (∃ t : ℝ, t = a / (1 - r) ∧ t = expected_sum) :=
by
  sorry

end geometric_series_sum_equals_3_6_l228_228515


namespace inequality_factorial_power_l228_228174

-- Given conditions
variables (n k : ℕ) (h₁ : 0 < k) (h₂ : k < n)

-- Define the problem statement to be proved
theorem inequality_factorial_power (h₁ : 0 < k) (h₂ : k < n) :
  (1 : ℝ) / (n + 1) * (n:ℝ)^n / ((k:ℝ)^k * (n - k:ℝ)^(n - k)) <
    nat.factorial n / (nat.factorial k * nat.factorial (n - k)) ∧
    nat.factorial n / (nat.factorial k * nat.factorial (n - k)) <
    (n:ℝ)^n / ((k:ℝ)^k * (n - k:ℝ)^(n - k)) :=
by
  sorry

end inequality_factorial_power_l228_228174


namespace volume_is_120_l228_228349

namespace volume_proof

-- Definitions from the given conditions
variables (a b c : ℝ)
axiom ab_relation : a * b = 48
axiom bc_relation : b * c = 20
axiom ca_relation : c * a = 15

-- Goal to prove
theorem volume_is_120 : a * b * c = 120 := by
  sorry

end volume_proof

end volume_is_120_l228_228349


namespace largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l228_228773

theorem largest_integer_less_than_80_with_remainder_3_when_divided_by_5 : 
  ∃ x : ℤ, x < 80 ∧ x % 5 = 3 ∧ (∀ y : ℤ, y < 80 ∧ y % 5 = 3 → y ≤ x) :=
sorry

end largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l228_228773


namespace factorial_ends_with_base_8_zeroes_l228_228872

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228872


namespace impossible_pairs_impossible_triplets_impossible_quadruples_l228_228159

theorem impossible_pairs : ∀ (tickets : list (set ℕ)), (∀ t, t ∈ tickets → t.card = 5) → (∀ x y, x ≠ y → ∃! t, {x, y} ⊆ t) → ¬ (finset.range 90).card % 4 = 0 :=
by
  sorry

theorem impossible_triplets : ∀ (tickets : list (set ℕ)), (∀ t, t ∈ tickets → t.card = 5) → (∀ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z → ∃! t, {x, y, z} ⊆ t) → ¬ (finset.range 90).card % 3 = 0 :=
by
  sorry

theorem impossible_quadruples : ∀ (tickets : list (set ℕ)), (∀ t, t ∈ tickets → t.card = 5) → (∀ x y z w, x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ x ≠ w ∧ x ≠ z ∧ y ≠ w → ∃! t, {x, y, z, w} ⊆ t) → ¬ (finset.range 90).card % 2 = 0 :=
by
  sorry

end impossible_pairs_impossible_triplets_impossible_quadruples_l228_228159


namespace triangular_based_prism_surface_area_l228_228724

theorem triangular_based_prism_surface_area (a b c : ℕ) (h : a = 3 ∧ b = 5 ∧ c = 12) :
  let triangular_prism_surface_area := 36 + 15 + 60 + (3 * 13)
  a = 3 ∧ b = 5 ∧ c = 12 → 
  triangular_prism_surface_area = 150 :=
by
  intro h
  let a := 3
  let b := 5
  let c := 12
  let diagonal := Int.sqrt (5 * 5 + 12 * 12)
  have d_eq_13 : diagonal = 13 := by norm_num
  let triangular_prism_surface_area := 36 + 15 + 60 + 39
  show triangular_prism_surface_area = 150, from
  by
    have areas_sum : 36 + 15 + 60 + 39 = 150 := by norm_num
    exact areas_sum

end triangular_based_prism_surface_area_l228_228724


namespace fraction_sum_is_121_l228_228209

theorem fraction_sum_is_121 :
  let frac := Rat.mk 98 144 in
  let simplified := (frac.num / frac.gcd).nat_abs in
  let den := (frac.den / frac.gcd).nat_abs in
  simplified + den = 121 :=
by
  let frac := Rat.mk 98 144
  let simplified := (frac.num / frac.gcd).nat_abs
  let den := (frac.den / frac.gcd).nat_abs
  have hsim : ∀ (a b : ℤ), Rat.numDenSucc ((Rat.num (a / b)).nat_abs, (Rat.den (a / b)).nat_abs) = (Rat.num (a / b).nat_abs, Rat.den (a / b).nat_abs) / Rat.gcd a b := sorry
  have hnum := (frac.num / frac.gcd).nat_abs
  have hden := (frac.den / frac.gcd).nat_abs
  show hnum + hden = 121
  sorry

end fraction_sum_is_121_l228_228209


namespace candy_days_l228_228027

theorem candy_days (neighbor_candy older_sister_candy candy_per_day : ℝ) 
  (h1 : neighbor_candy = 11.0) 
  (h2 : older_sister_candy = 5.0) 
  (h3 : candy_per_day = 8.0) : 
  ((neighbor_candy + older_sister_candy) / candy_per_day) = 2.0 := 
by 
  sorry

end candy_days_l228_228027


namespace find_AX_l228_228489

-- Given points A, B, and X, with AB = 75
variables {A B X : Point}
variable (d : dist A B = 75)

-- Conditions from the problem: BX = 2 * AX and AB = AX + BX
variables {AX BX : ℝ}
variable (h1 : AX + BX = 75)
variable (h2 : BX = 2 * AX)

-- Prove that AX = 25
theorem find_AX (d : dist A B = 75) (h1 : AX + BX = 75) (h2 : BX = 2 * AX) : AX = 25 := 
sorry

end find_AX_l228_228489


namespace area_of_regular_octagon_in_circle_l228_228678

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228678


namespace lily_account_balance_l228_228976

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l228_228976


namespace area_of_shaded_rectangle_l228_228304

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l228_228304


namespace find_m_l228_228954

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : a + b = 2) : ∃ m, 2^a = m ∧ 5^b = m ∧ a + b = 2 :=
by {
    use m,
    split,
    exact h1,
    split,
    exact h2,
    exact h3,
    sorry
}

end find_m_l228_228954


namespace find_b_l228_228939

noncomputable def given_c := 3
noncomputable def given_C := Real.pi / 3
noncomputable def given_cos_C := 1 / 2
noncomputable def given_a (b : ℝ) := 2 * b

theorem find_b (b : ℝ) (h1 : given_c = 3) (h2 : given_cos_C = Real.cos (given_C)) (h3 : given_a b = 2 * b) : b = Real.sqrt 3 := 
by
  sorry

end find_b_l228_228939


namespace meat_per_slice_is_22_l228_228978

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l228_228978


namespace problem_l228_228318

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem problem :
  let A := 3.14159265
  let B := Real.sqrt 36
  let C := Real.sqrt 7
  let D := 4.1
  is_irrational C := by
  sorry

end problem_l228_228318


namespace compute_iterated_function_l228_228953

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then
    -x^2 - 2 * x
  else
    x + 7

theorem compute_iterated_function : f (f (f (f (f 2)))) = -41 := by
  sorry

end compute_iterated_function_l228_228953


namespace PA_and_notB_l228_228816

noncomputable def A : Event := sorry -- Definition for event A
noncomputable def B : Event := sorry -- Definition for event B

def P (e : Event) : ℝ := sorry -- Probability function

axiom independent (A B : Event) : Prop := sorry -- Independence axiom

axiom PA : P(A) = 0.5 -- Given P(A) = 0.5
axiom PB : P(B) = 0.4 -- Given P(B) = 0.4

theorem PA_and_notB : independent A B → P(A ∩ (set.compl B)) = 0.3 := by
  sorry

end PA_and_notB_l228_228816


namespace octagon_area_l228_228670

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228670


namespace f_odd_function_l228_228380

noncomputable def f (x : ℝ) : ℝ := real.log (1 + x) / real.log (1 - x)

theorem f_odd_function (x : ℝ) (h : -1 < x ∧ x < 1) : f(-x) = -f(x) := 
by 
  have h_dom : -1 < -x ∧ -x < 1 := ⟨by linarith, by linarith⟩
  sorry

end f_odd_function_l228_228380


namespace three_consecutive_multiples_sum_l228_228603

theorem three_consecutive_multiples_sum (h1 : Int) (h2 : h1 % 3 = 0) (h3 : Int) (h4 : h3 = h1 - 3) (h5 : Int) (h6 : h5 = h1 - 6) (h7: h1 = 27) : h1 + h3 + h5 = 72 := 
by 
  -- let numbers be n, n-3, n-6 and n = 27
  -- so n + n-3 + n-6 = 27 + 24 + 21 = 72
  sorry

end three_consecutive_multiples_sum_l228_228603


namespace total_cost_l228_228787

-- Definition of the conditions
def cost_sharing (x : ℝ) : Prop :=
  let initial_cost := x / 5
  let new_cost := x / 7
  initial_cost - 15 = new_cost

-- The statement we need to prove
theorem total_cost (x : ℝ) (h : cost_sharing x) : x = 262.50 := by
  sorry

end total_cost_l228_228787


namespace min_ratio_area_of_incircle_circumcircle_rt_triangle_l228_228908

variables (a b: ℝ)
variables (a' b' c: ℝ)

-- Conditions
def area_of_right_triangle (a b : ℝ) : ℝ := 
    0.5 * a * b

def incircle_radius (a' b' c : ℝ) : ℝ := 
    0.5 * (a' + b' - c)

def circumcircle_radius (c : ℝ) : ℝ := 
    0.5 * c

-- Condition of the problem
def condition (a b a' b' c : ℝ) : Prop :=
    incircle_radius a' b' c = circumcircle_radius c ∧ 
    a' + b' = 2 * c

-- The final proof problem
theorem min_ratio_area_of_incircle_circumcircle_rt_triangle (a b a' b' c : ℝ)
    (h_area_a : a = area_of_right_triangle a' b')
    (h_area_b : b = area_of_right_triangle a b)
    (h_condition : condition a b a' b' c) :
    (a / b ≥ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end min_ratio_area_of_incircle_circumcircle_rt_triangle_l228_228908


namespace simplify_expression_l228_228208

def expr_initial (y : ℝ) := 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2)
def expr_simplified (y : ℝ) := 8*y^2 + 6*y - 5

theorem simplify_expression (y : ℝ) : expr_initial y = expr_simplified y :=
by
  sorry

end simplify_expression_l228_228208


namespace evaluate_function_l228_228828

def f (x : ℝ) : ℝ :=
if x < 1 then 2 * Real.sin (Real.pi * x)
else f (x - 2 / 3)

theorem evaluate_function :
  (f 2 / f (-1 / 6)) = -Real.sqrt 3 :=
by sorry

end evaluate_function_l228_228828


namespace line_equation_l228_228073

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / 4 = 1

noncomputable def is_centroid (A B C centroid : ℝ × ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    A = (x1, y1) ∧ B = (x2, y2) ∧ C = (0, -2) ∧ 
    (centroid.1 = (x1 + x2) / 3 ∧ centroid.2 = (y1 + y2 - 2) / 3)

theorem line_equation (A B : ℝ × ℝ) (C : ℝ × ℝ := (0, -2)) (l_focus centroid : ℝ × ℝ := (-1, 0)) : 
  (ellipse A.1 A.2) ∧ (ellipse B.1 B.2) ∧ (is_centroid A B C centroid) → 
  ∃ (k : ℝ), k = 6 / 5 ∧ (∃ (x y : ℝ), l_focus = (-3 / 2, 1) ∧ (6 * x - 5 * y + 14 = 0)) :=
begin
  sorry
end

end line_equation_l228_228073


namespace trigonometric_identity_l228_228449

theorem trigonometric_identity 
  (θ φ : ℝ)
  (h : (cos θ)^6 / (cos φ)^2 + (sin θ)^6 / (sin φ)^2 = 2) :
  (sin φ)^6 / (sin θ)^2 + (cos φ)^6 / (cos θ)^2 = 1 := 
sorry

end trigonometric_identity_l228_228449


namespace integral_of_3x_plus_4_times_exp_3x_l228_228747

open scoped Real

theorem integral_of_3x_plus_4_times_exp_3x (C : ℝ) :
  ∫ x in 0..Inf Bound (3x + 4) * (exp (3x)) = (x + 1) * exp (3x) + C :=
by
  sorry

end integral_of_3x_plus_4_times_exp_3x_l228_228747


namespace sum_of_cubes_eq_five_l228_228186

noncomputable def root_polynomial (a b c : ℂ) : Prop :=
  (a + b + c = 2) ∧ (a*b + b*c + c*a = 3) ∧ (a*b*c = 5)

theorem sum_of_cubes_eq_five (a b c : ℂ) (h : root_polynomial a b c) :
  a^3 + b^3 + c^3 = 5 :=
sorry

end sum_of_cubes_eq_five_l228_228186


namespace trapezoid_angles_l228_228594

theorem trapezoid_angles (BC AD : ℝ) (k r R : ℝ) (h_par : BC ∥ AD)
  (h_ratio : k = R / r) (h_isosceles : ∀ A B C D, ABCD.isIsoscelesTrapezoid) :
  k > Real.sqrt 2 →
  ∃ α β : ℝ, 
    α = Real.arcsin(Real.sqrt((1 + Real.sqrt(1 + 4 * k^2)) / 2) / k) ∧
    β = Real.pi - Real.arcsin(Real.sqrt((1 + Real.sqrt(1 + 4 * k^2)) / 2) / k) :=
begin
  intro h_k_gt_sqrt2,
  use [Real.arcsin(Real.sqrt((1 + Real.sqrt(1 + 4 * k^2)) / 2) / k),
       Real.pi - Real.arcsin(Real.sqrt((1 + Real.sqrt(1 + 4 * k^2)) / 2) / k)],
  split,
  { sorry }, -- First angle proof
  { sorry }  -- Second angle proof
end

end trapezoid_angles_l228_228594


namespace max_sum_abc_divisible_by_13_l228_228118

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l228_228118


namespace number_of_trailing_zeroes_base8_l228_228858

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228858


namespace combined_selling_price_approx_l228_228948

noncomputable def bond_selling_price (face_value interest_rate percentage_selling_price : ℝ) : ℝ :=
  (interest_rate * face_value) / percentage_selling_price

def bond_A_face_value : ℝ := 5000
def bond_B_face_value : ℝ := 7000
def bond_C_face_value : ℝ := 10000

def bond_A_interest_rate : ℝ := 0.06
def bond_B_interest_rate : ℝ := 0.08
def bond_C_interest_rate : ℝ := 0.05

def bond_A_percentage : ℝ := 0.065
def bond_B_percentage : ℝ := 0.075
def bond_C_percentage : ℝ := 0.045

def bond_A_selling_price : ℝ := bond_selling_price bond_A_face_value bond_A_interest_rate bond_A_percentage
def bond_B_selling_price : ℝ := bond_selling_price bond_B_face_value bond_B_interest_rate bond_B_percentage
def bond_C_selling_price : ℝ := bond_selling_price bond_C_face_value bond_C_interest_rate bond_C_percentage

def combined_selling_price : ℝ := bond_A_selling_price + bond_B_selling_price + bond_C_selling_price
def approx_combined_selling_price : ℝ := 23193.16

theorem combined_selling_price_approx :
  abs (combined_selling_price - approx_combined_selling_price) < 1 :=
begin
  sorry
end

end combined_selling_price_approx_l228_228948


namespace center_and_radius_max_chord_length_l228_228389

-- Definitions and conditions.
def circle_eq (x y a : ℝ) : Prop :=
    x^2 + y^2 + 2 * a * x - 2 * a * y + 2 * a^2 - 4 * a = 0

def line_eq (x y : ℝ) : Prop :=
    y = x + 4

def center_of_circle (a : ℝ) : ℝ × ℝ :=
    (-a, a)

def radius_of_circle (a : ℝ) : ℝ :=
    2 * real.sqrt a

def distance_from_center_to_line (a : ℝ) : ℝ :=
    real.sqrt 2 * abs (2 - a)

def length_of_chord (a : ℝ) : ℝ :=
    2 * real.sqrt (2 * a^2 - (real.sqrt 2 * abs (2 - a))^2)

-- Theorems to be proven.
theorem center_and_radius (a : ℝ) (h : 0 < a ∧ a ≤ 4) : 
    center_of_circle a = (-a, a) ∧ radius_of_circle a = 2 * real.sqrt a :=
sorry

theorem max_chord_length (a : ℝ) (h : 0 < a ∧ a ≤ 4) :
    (∀ a, 0 < a ∧ a ≤ 4 → length_of_chord a ≤ 2 * real.sqrt 10) ∧
    length_of_chord 3 = 2 * real.sqrt 10 :=
sorry

end center_and_radius_max_chord_length_l228_228389


namespace sum_fourth_power_l228_228105

  theorem sum_fourth_power (x y z : ℝ) 
    (h1 : x + y + z = 2) 
    (h2 : x^2 + y^2 + z^2 = 6) 
    (h3 : x^3 + y^3 + z^3 = 8) : 
    x^4 + y^4 + z^4 = 26 := 
  by 
    sorry
  
end sum_fourth_power_l228_228105


namespace intersection_of_sets_l228_228439

open Set

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | x^2 - 4 > 0}
def B : Set ℝ := {x | x + 2 < 0}

-- The proof goal
theorem intersection_of_sets : A ∩ B = {x | x < -2} := 
by sorry

end intersection_of_sets_l228_228439


namespace time_in_137_hours_58_minutes_and_59_seconds_l228_228161

def time_increment (initial_hours initial_minutes initial_seconds incr_hours incr_minutes incr_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := initial_seconds + incr_seconds
  let total_minutes := initial_minutes + incr_minutes + (total_seconds / 60)
  let total_hours := initial_hours + incr_hours + (total_minutes / 60)
  ((total_hours % 24), (total_minutes % 60), (total_seconds % 60))

theorem time_in_137_hours_58_minutes_and_59_seconds
(initial_hours initial_minutes initial_seconds : ℕ)
(incr_hours incr_minutes incr_seconds : ℕ)
(h_initial : initial_hours = 15)
(h_initial_minutes : initial_minutes = 0)
(h_initial_seconds : initial_seconds = 0)
(h_incr_hours : incr_hours = 137)
(h_incr_minutes : incr_minutes = 58)
(h_incr_seconds : incr_seconds = 59) :
let (X, Y, Z) := time_increment initial_hours initial_minutes initial_seconds incr_hours incr_minutes incr_seconds
in X + Y + Z = 125 := by
sorry 

end time_in_137_hours_58_minutes_and_59_seconds_l228_228161


namespace xiao_li_can_buy_l228_228268

def can_buy_pens (x y : ℕ) : Prop :=
  3 * x + y = 11

theorem xiao_li_can_buy : ∃ x, ∃ y, can_buy_pens x y ∧ x ∈ {1, 2, 3} :=
by
  sorry

end xiao_li_can_buy_l228_228268


namespace arithmetic_progression_iff_condition_l228_228185

def is_arithmetic_progression (x : ℕ → ℝ) : Prop :=
∀ n ≥ 1, x (n + 1) - x n = x 2 - x 1

def summation_condition (x : ℕ → ℝ) : Prop :=
∀ n ≥ 2, (Finset.range (n - 1)).sum (λ k, (1 / (x k.succ * x (k + 2)))) = (n - 1) / (x 1 * x n)

theorem arithmetic_progression_iff_condition (x : ℕ → ℝ) (h : ∀ n, x n ≠ 0) :
  is_arithmetic_progression x ↔ summation_condition x := by
  sorry

end arithmetic_progression_iff_condition_l228_228185


namespace amount_given_to_beggar_l228_228653

variable (X : ℕ)
variable (pennies_initial : ℕ := 42)
variable (pennies_to_farmer : ℕ := 22)
variable (pennies_after_farmer : ℕ := 20)

def amount_to_boy (X : ℕ) : ℕ :=
  (20 - X) / 2 + 3

theorem amount_given_to_beggar : 
  (X = 12) →  (pennies_initial - pennies_to_farmer - X) / 2 + 3 + 1 = pennies_initial - pennies_to_farmer - X :=
by
  intro h
  subst h
  sorry

end amount_given_to_beggar_l228_228653


namespace number_of_valid_strings_l228_228778

def count_valid_strings (n : ℕ) : ℕ :=
  4^n - 3 * 3^n + 3 * 2^n - 1

theorem number_of_valid_strings (n : ℕ) :
  count_valid_strings n = 4^n - 3 * 3^n + 3 * 2^n - 1 :=
by sorry

end number_of_valid_strings_l228_228778


namespace gcd_18_30_45_l228_228362

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l228_228362


namespace initial_depth_l228_228647

theorem initial_depth (R : ℝ) : 
  let W1 := 45 * 8 * R in
  let W2 := 75 * 6 * R in
  W2 = 50 → W1 = 40 :=
by
  intro hr
  have h : 75 * 6 * R = 50 := hr
  sorry

end initial_depth_l228_228647


namespace area_of_regular_octagon_in_circle_l228_228682

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228682


namespace range_of_a_for_monotonically_decreasing_l228_228463

noncomputable def f (a x: ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem range_of_a_for_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/x - a*x - 2 < 0)) ↔ (a ∈ Set.Ioi (-1)) := 
sorry

end range_of_a_for_monotonically_decreasing_l228_228463


namespace parallel_lines_condition_l228_228961

theorem parallel_lines_condition (a : ℝ) : 
  (a = 4) ↔ (∀ x y : ℝ, l₁ a x y = 0 → (∀ x y : ℝ, l₂ a x y = 0 → (∃ k : ℝ, k * slope l₁ a = slope l₂ a))) :=
by
  sorry

def l₁ (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 3
def l₂ (a : ℝ) (x y : ℝ) : ℝ := 2 * x + y - a
def slope (l : ℝ → ℝ → ℝ) (a : ℝ) : ℝ := -((a) / 2)

end parallel_lines_condition_l228_228961


namespace find_f_of_3_l228_228798

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^7 + a*x^5 + b*x - 5

theorem find_f_of_3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := by
  sorry

end find_f_of_3_l228_228798


namespace math_score_computation_l228_228290

def comprehensive_score 
  (reg_score : ℕ) (mid_score : ℕ) (fin_score : ℕ) 
  (reg_weight : ℕ) (mid_weight : ℕ) (fin_weight : ℕ) 
  : ℕ :=
  (reg_score * reg_weight + mid_score * mid_weight + fin_score * fin_weight) 
  / (reg_weight + mid_weight + fin_weight)

theorem math_score_computation :
  comprehensive_score 80 80 85 3 3 4 = 82 := by
sorry

end math_score_computation_l228_228290


namespace number_of_dogs_l228_228326

-- Define the total number of dogs
def total_dogs := 150

variables {D : Type} [fintype D]
variable dogs : Finset D
variables sit stay roll_over jump : Finset D

-- Define the given conditions
variables (h1 : sit.card = 60)
          (h2 : (sit ∩ stay).card = 25)
          (h3 : stay.card = 40)
          (h4 : (stay ∩ roll_over).card = 15)
          (h5 : roll_over.card = 45)
          (h6 : (sit ∩ roll_over).card = 20)
          (h7 : jump.card = 50)
          (h8 : (jump ∩ stay).card = 5)
          (h9 : ((sit ∩ stay) ∩ roll_over).card = 10)
          (h10 : (dogs \ (sit ∪ stay ∪ roll_over ∪ jump)).card = 5)

-- Theorem to be proved
theorem number_of_dogs : dogs.card = total_dogs := by
  sorry

end number_of_dogs_l228_228326


namespace petya_time_spent_l228_228994

theorem petya_time_spent :
  (1 / 3) + (1 / 5) + (1 / 6) + (1 / 70) + (1 / 3) > 1 :=
by
  sorry

end petya_time_spent_l228_228994


namespace find_lambda_l228_228097

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (λ : ℝ) : ℝ × ℝ := (2, λ)
def c : ℝ × ℝ := (2, 1)

-- Define the condition that c is parallel to (2a + b)
def parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- The theorem to prove λ = -2
theorem find_lambda (λ : ℝ) : parallel c (2 • a.1, 2 • a.2) + b λ → λ = -2 :=
by
  intros h_parallel
  sorry

end find_lambda_l228_228097


namespace third_person_gets_800_l228_228608

noncomputable def third_person_profit
(x : ℝ) 
(h1 : 3 * x + 3000 = 9000)
(h2 : 1800 : ℝ) : ℝ :=
  let total_investment := 9000 in
  let third_investment := x + 2000 in
  let total_profit := 1800 in
  (third_investment / total_investment) * total_profit

theorem third_person_gets_800
(x : ℝ)
(h1 : 3 * x + 3000 = 9000)
(h2 : x = 2000)
(h3 : third_person_profit x h1 1800 = 800) :
  third_person_profit x h1 1800 = 800 
:= by sorry

end third_person_gets_800_l228_228608


namespace area_of_regular_octagon_in_circle_l228_228676

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228676


namespace banana_to_pear_equiv_l228_228738

/-
Given conditions:
1. 5 bananas cost as much as 3 apples.
2. 9 apples cost the same as 6 pears.
Prove the equivalence between 30 bananas and 12 pears.

We will define the equivalences as constants and prove the cost equivalence.
-/

variable (cost_banana cost_apple cost_pear : ℤ)

noncomputable def cost_equiv : Prop :=
  (5 * cost_banana = 3 * cost_apple) ∧ 
  (9 * cost_apple = 6 * cost_pear) →
  (30 * cost_banana = 12 * cost_pear)

theorem banana_to_pear_equiv :
  cost_equiv cost_banana cost_apple cost_pear :=
by
  sorry

end banana_to_pear_equiv_l228_228738


namespace coeff_a_zero_l228_228459

theorem coeff_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, 0 < p → ∀ (x : ℝ), (a * x^2 + b * x + c + p = 0) → x > 0) :
  a = 0 :=
sorry

end coeff_a_zero_l228_228459


namespace solution_set_ineq_range_of_k_l228_228079

-- Define the function and conditions
def f (x k : ℝ) := k * x / (x^2 + 3 * k)
def m := -2 / 5
def k_fixed := 2

-- Statement for part (1)
theorem solution_set_ineq (x : ℝ) : 
  (-1 < x ∧ x < (3 / 2)) ↔ (5 * m * x^2 + k_fixed / 2 * x + 3 > 0) := sorry

-- Statement for part (2)
theorem range_of_k (k : ℝ) :
  (∃ x > 3, f x k > 1) ↔ (k > 12) := sorry

end solution_set_ineq_range_of_k_l228_228079


namespace area_of_inscribed_octagon_l228_228715

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228715


namespace math_problem_l228_228502

variables (n : ℕ) (x y : Fin n → ℝ)

theorem math_problem (h1 : ∀ i j : Fin n, i < j → x i < x j)
  (h2 : ∀ i, -1 < x i ∧ x i < 1)
  (h3 : ∑ i : Fin n, (x i) ^ 13 = ∑ i : Fin n, x i)
  (h4 : ∀ i j : Fin n, i < j → y i < y j) :
  ∑ i : Fin n, (x i) ^ 13 * y i < ∑ i : Fin n, x i * y i :=
by
  sorry

end math_problem_l228_228502


namespace bottles_per_month_l228_228500

-- Conditions
def discount := 0.3
def discounted_cost : ℝ := 252
def cost_per_bottle : ℝ := 30

-- Question with required proof
theorem bottles_per_month :
  (∀ (B : ℝ), discounted_cost = (1 - discount) * (B * cost_per_bottle) → B = 12) →
  (∀ (B : ℝ), (B = 12) → B / 12 = 1):=
by
  intro h1 h2
  apply h2
  have hb : B = 12 := sorry
  exact hb

end bottles_per_month_l228_228500


namespace solution_l228_228972

noncomputable def problem (b c : ℝ) : Prop :=
  let f := λ x : ℝ, x^2 + b * x + c in
  0 ≤ f 1 ∧ f 1 = f 2 ∧ f 1 ≤ 10 → 2 ≤ c ∧ c ≤ 12

theorem solution (b c : ℝ) (h: problem b c) : 2 ≤ c ∧ c ≤ 12 :=
by
  cases h with h1 h2
  sorry

end solution_l228_228972


namespace every_positive_integer_in_A_or_B_l228_228184

def nth_prime (n : ℕ) : ℕ := sorry -- define the nth prime number function (to be filled in)
def π (n : ℕ) : ℕ := sorry -- define the prime-counting function (to be filled in)

def A : Set ℕ := { n + nth_prime(n) - 1 | n : ℕ, 0 < n }
def B : Set ℕ := { n + π(n) | n : ℕ, 0 < n }

theorem every_positive_integer_in_A_or_B (k : ℕ) (h : 0 < k) : k ∈ A ∪ B ∧ k ∉ A ∩ B :=
by
  sorry

end every_positive_integer_in_A_or_B_l228_228184


namespace angle_B_is_30_degrees_l228_228940

/-- In triangle ABC, the size of angle B is 30 degrees given the conditions -/
theorem angle_B_is_30_degrees
  (A B C : Real)
  (a b c : Real)
  (h1 : b^2 + c^2 - a^2 = b * c)
  (h2 : Real.sin(A) ^ 2 + Real.sin(B) ^ 2 = Real.sin(C) ^ 2)
  (h_triangle : A + B + C = Real.pi) -- Note that we use radians in Lean, π radians is 180 degrees
  : B = Real.pi / 6 :=  -- π/6 radians is 30 degrees
  sorry

end angle_B_is_30_degrees_l228_228940


namespace geometric_sequence_condition_neither_necessary_nor_sufficient_l228_228566

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_condition_neither_necessary_nor_sufficient (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q → ¬( (is_monotonically_increasing a ↔ q > 1) ) :=
by sorry

end geometric_sequence_condition_neither_necessary_nor_sufficient_l228_228566


namespace swap_instruments_readings_change_l228_228279

def U0 : ℝ := 45
def R : ℝ := 50
def r : ℝ := 20

theorem swap_instruments_readings_change :
  let I_total := U0 / (R / 2 + r)
  let U1 := I_total * r
  let I1 := I_total / 2
  let I2 := U0 / R
  let I := U0 / (R + r)
  let U2 := I * r
  let ΔI := I2 - I1
  let ΔU := U1 - U2
  ΔI = 0.4 ∧ ΔU = 7.14 :=
by
  sorry

end swap_instruments_readings_change_l228_228279


namespace zeroes_at_end_base_8_of_factorial_15_l228_228880

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228880


namespace solve_for_n_l228_228031

theorem solve_for_n (n : ℕ) (h : 5 * 8 * 2 * 6 * n = 9!) : n = 756 :=
by {
  sorry
}

end solve_for_n_l228_228031


namespace number_of_true_propositions_l228_228317

theorem number_of_true_propositions :
  (∀ x : ℝ, (x > 0 → x + 1 / x ≥ 2) ∧ (x < 0 → x + 1 / x ≤ -2)) →
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - 2 * x + 1 ≥ 0) ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - 2 * x + 1 < 0) →
  (∀ x y : ℝ, (y = 0.57 * x - 0.448) → (y → x > 0)) →
  (∃ R : ℝ, (R = Real.sqrt 3) ∧ (4 * Real.pi * R^2 = 12 * Real.pi)) →
  3 = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_true_propositions_l228_228317


namespace range_of_a_l228_228971

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h₁ : a ≠ 1) :
  (∃ (x : ℝ), 1 ≤ x ∧ f a x < a / (a - 1)) ↔ (a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1) :=
begin
  sorry
end

end range_of_a_l228_228971


namespace symmetric_point_in_polar_coordinates_l228_228146

theorem symmetric_point_in_polar_coordinates (ρ θ : ℝ) :
  (ρ, θ) = (1, 0) → (ρ, θ + π) = (1, π) :=
by
  intro h
  rw [(Eq.subst h)] at *
  simp
  sorry

end symmetric_point_in_polar_coordinates_l228_228146


namespace problem_statement_l228_228941

variables {A B C O D : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (a b c o d : A)

-- Define the geometric conditions
axiom condition1 : a + 2 • b + 3 • c = 0
axiom condition2 : ∃ (D: A), (∃ (k : ℝ), a = k • d ∧ k ≠ 0) ∧ (∃ (u v : ℝ),  u • b + v • c = d ∧ u + v = 1)

-- Define points
def OA : A := a - o
def OB : A := b - o
def OC : A := c - o
def OD : A := d - o

-- The main statement to prove
theorem problem_statement : 2 • (b - d) + 3 • (c - d) = (0 : A) :=
by
  sorry

end problem_statement_l228_228941


namespace length_GH_l228_228555

-- Definitions for the coordinates
def y_A : ℝ := 14
def y_B : ℝ := 10
def y_C : ℝ := 28

-- Definition for the y-coordinate of centroid G
def y_G : ℝ := (y_A + y_B + y_C) / 3

-- Proving the length of GH
theorem length_GH : ∀ (x : ℝ), x = y_G → x = 52 / 3 :=
by
  intros x h
  rw h
  exact (by norm_num : 52 / 3 = 52 / 3)

end length_GH_l228_228555


namespace pyramid_volume_l228_228330

theorem pyramid_volume (length width height : ℝ) (h_length : length = 4) (h_width : width = 8) (h_height : height = 10) :
  (1 / 3) * (length * width) * height = 106.67 :=
by
  rw [h_length, h_width, h_height]
  norm_num

end pyramid_volume_l228_228330


namespace find_constant_l228_228115

theorem find_constant
  {x : ℕ} (f : ℕ → ℕ)
  (h1 : ∀ x, f x = x^2 + 2*x + c)
  (h2 : f 2 = 12) :
  c = 4 :=
by sorry

end find_constant_l228_228115


namespace extraneous_root_example_l228_228337

theorem extraneous_root_example :
  (∃ x : ℝ, sqrt (x + 15) - 7 / sqrt (x + 15) = 6) ∧ (-15 < -16) ∧ (-16 < -10) :=
by
  -- define u as sqrt (x + 15)
  let u : ℝ := sqrt (-16 + 15)
  -- calculate u
  have h_u : u = sqrt (-1), from rfl
  -- sqrt (-1) is not a real number
  sorry

end extraneous_root_example_l228_228337


namespace sum_coefficients_equals_l228_228453

theorem sum_coefficients_equals :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ), 
  (∀ x : ℤ, (2 * x + 1) ^ 5 = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_0 = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 3^5 - 1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h h0
  sorry

end sum_coefficients_equals_l228_228453


namespace circle_standard_eq_of_symmetric_center_l228_228122

def point := ℝ × ℝ

def symmetric_point (p : point) (line : ℝ → ℝ) : point :=
  (p.2, p.1)

def circle_equation (center : point) (radius : ℝ) (x y : ℝ) : ℝ :=
  (x - center.1)^2 + (y - center.2)^2

theorem circle_standard_eq_of_symmetric_center:
  (radius : ℝ) (p : point)
  (h_radius : radius = 1)
  (h_center_sym : symmetric_point (1, 0) (λ x, x) = p) :
  circle_equation p radius = λ x y => x^2 + (y - 1)^2 := by
  sorry

end circle_standard_eq_of_symmetric_center_l228_228122


namespace smallest_positive_period_and_shift_to_even_function_l228_228426

noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * sqrt 3 * cos x ^ 2 - sqrt 3

theorem smallest_positive_period_and_shift_to_even_function :
  (∃ T > 0, ∀ x, f x = f (x + T) ∧ T = π) ∧
  (∃ φ, (π/2 < φ ∧ φ < π) ∧ ∀ x, f(x - φ) = f(φ - x) ∧ φ = 7 * π / 12) :=
by
  sorry

end smallest_positive_period_and_shift_to_even_function_l228_228426


namespace jerry_total_games_after_birthday_and_trade_l228_228945

def jerry_initial_action_games := 7
def jerry_initial_strategy_games := 5
def jerry_action_games_increase := (30 * jerry_initial_action_games + 50) / 100 -- addition of 50 for rounding
def jerry_strategy_games_increase := (20 * jerry_initial_strategy_games) / 100

def jerry_new_action_games := jerry_initial_action_games + jerry_action_games_increase
def jerry_new_strategy_games := jerry_initial_strategy_games + jerry_strategy_games_increase

def jerry_action_games_after_trade := jerry_new_action_games - 2
def jerry_sports_games_after_trade := 3

def jerry_total_games := jerry_action_games_after_trade + jerry_new_strategy_games + jerry_sports_games_after_trade

theorem jerry_total_games_after_birthday_and_trade : jerry_total_games = 16 :=
by {
  rw [jerry_initial_action_games, jerry_initial_strategy_games, jerry_action_games_increase, jerry_strategy_games_increase, jerry_new_action_games, jerry_new_strategy_games, jerry_action_games_after_trade, jerry_sports_games_after_trade, jerry_total_games],
  norm_num,
  exact trivial,
}

end jerry_total_games_after_birthday_and_trade_l228_228945


namespace wedding_chairs_total_l228_228028

theorem wedding_chairs_total :
  let first_section_rows := 5
  let first_section_chairs_per_row := 10
  let first_section_late_people := 15
  let first_section_extra_chairs_per_late := 2
  
  let second_section_rows := 8
  let second_section_chairs_per_row := 12
  let second_section_late_people := 25
  let second_section_extra_chairs_per_late := 3
  
  let third_section_rows := 4
  let third_section_chairs_per_row := 15
  let third_section_late_people := 8
  let third_section_extra_chairs_per_late := 1

  let fourth_section_rows := 6
  let fourth_section_chairs_per_row := 9
  let fourth_section_late_people := 12
  let fourth_section_extra_chairs_per_late := 1
  
  let total_original_chairs := 
    (first_section_rows * first_section_chairs_per_row) + 
    (second_section_rows * second_section_chairs_per_row) + 
    (third_section_rows * third_section_chairs_per_row) + 
    (fourth_section_rows * fourth_section_chairs_per_row)
  
  let total_extra_chairs :=
    (first_section_late_people * first_section_extra_chairs_per_late) + 
    (second_section_late_people * second_section_extra_chairs_per_late) + 
    (third_section_late_people * third_section_extra_chairs_per_late) + 
    (fourth_section_late_people * fourth_section_extra_chairs_per_late)
  
  total_original_chairs + total_extra_chairs = 385 :=
by
  sorry

end wedding_chairs_total_l228_228028


namespace pizza_slices_meat_count_l228_228982

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l228_228982


namespace line_through_parabola_intersects_vertex_l228_228835

theorem line_through_parabola_intersects_vertex (y x k : ℝ) :
  (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0) ∧ 
  (∃ P Q : ℝ × ℝ, (P.1)^2 = 4 * P.2 ∧ (Q.1)^2 = 4 * Q.2 ∧ 
   (P = (0, 0) ∨ Q = (0, 0)) ∧ 
   (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0)) := sorry

end line_through_parabola_intersects_vertex_l228_228835


namespace geometric_series_sum_l228_228750

theorem geometric_series_sum :
  let a := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  let expected_sum := (3216929751 : ℚ) / 1073741824
  (finset.range n).sum (λ k => a * r ^ k) = expected_sum :=
by
  let a := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  let expected_sum := (3216929751 : ℚ) / 1073741824
  sorry

end geometric_series_sum_l228_228750


namespace a_real_number_a_complex_number_a_purely_imaginary_number_l228_228032

def z (m : ℝ) : ℂ := (m - 4 : ℝ) + (m^2 - 5 * m - 6 : ℂ) * complex.I

theorem a_real_number (m : ℝ) : z m.im = 0 ↔ m = 6 ∨ m = -1 :=
by sorry

theorem a_complex_number (m : ℝ) : z m.im ≠ 0 ↔ m ≠ 6 ∧ m ≠ -1 :=
by sorry

theorem a_purely_imaginary_number (m : ℝ) : z m.re = 0 ∧ z m.im ≠ 0 ↔ m = 4 :=
by sorry

end a_real_number_a_complex_number_a_purely_imaginary_number_l228_228032


namespace evaluate_f_at_neg_3_div_2_l228_228120

def f (x : ℝ) : ℝ :=
  if x < 1 then f (x + 1) else 2 * x - 1

theorem evaluate_f_at_neg_3_div_2 : f (-3/2) = 2 := by sorry

end evaluate_f_at_neg_3_div_2_l228_228120


namespace sum_of_angles_in_triangle_relies_on_parallel_postulate_l228_228008

/-- Does the theorem on the sum of the angles of a triangle (equals 180 degrees) rely on the parallel postulate? -/
theorem sum_of_angles_in_triangle_relies_on_parallel_postulate (E : Type) [euclidean_geometry E] :
  (∀ (T : triangle E), sum_of_angles T = π) → depends_on_parallel_postulate E :=
sorry

end sum_of_angles_in_triangle_relies_on_parallel_postulate_l228_228008


namespace empty_set_negation_l228_228232

open Set

theorem empty_set_negation (α : Type) : ¬ (∀ s : Set α, ∅ ⊆ s) ↔ (∃ s : Set α, ¬(∅ ⊆ s)) :=
by
  sorry

end empty_set_negation_l228_228232


namespace train_crossing_time_l228_228102

variable (Lt : ℝ) -- Length of the train
variable (Lb : ℝ) -- Length of the bridge
variable (S_kmph : ℝ) -- Speed of the train in kmph

def S_mps (S_kmph : ℝ) : ℝ := S_kmph * 1000 / 3600 -- Convert kmph to m/s

theorem train_crossing_time (Lt Lb : ℝ) (S_kmph : ℝ) (S_mps := S_kmph * 1000 / 3600) :
  S_kmph = 18 → Lt = 100 → Lb = 150 → (Lt + Lb) / S_mps = 50 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  have : S_mps = 18 * 1000 / 3600 := rfl
  rw [this]
  norm_num -- Normalize the numeric fraction 18000/3600
  sorry

end train_crossing_time_l228_228102


namespace find_an_l228_228960

def sequence_sum (k : ℝ) (n : ℕ) : ℝ :=
  k * n ^ 2 + n

def term_of_sequence (k : ℝ) (n : ℕ) (S_n : ℝ) (S_nm1 : ℝ) : ℝ :=
  S_n - S_nm1

theorem find_an (k : ℝ) (n : ℕ) (h₁ : n > 0) :
  term_of_sequence k n (sequence_sum k n) (sequence_sum k (n - 1)) = 2 * k * n - k + 1 :=
by
  sorry

end find_an_l228_228960


namespace line_intersects_x_axis_l228_228661

open Real

namespace Proof

structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := ⟨3, 2⟩
def point2 : Point := ⟨6, 5⟩

theorem line_intersects_x_axis :
  ∃ x : ℝ, (x, (0 : ℝ)) = (⟨1, 0⟩ : Point) := by
  sorry

end Proof

end line_intersects_x_axis_l228_228661


namespace distance_from_P_to_BC_l228_228282

noncomputable def problem_conditions := sorry

theorem distance_from_P_to_BC :
  ∀ (A B C P : Type)
  (AB AC BC PA : ℝ)
  (PA_perpendicular : ∀ (x : A), x ∉ A)
  (hAB : AB = 13)
  (hAC : AC = 13)
  (hBC : BC = 10)
  (hPA : PA = 5),
  distance P (line_segment B C) = 13 := 
sorry

end distance_from_P_to_BC_l228_228282


namespace graph_of_equation_is_parabola_and_ellipse_l228_228347

theorem graph_of_equation_is_parabola_and_ellipse :
  let eq := ∀ x y : ℝ, y^4 - 6 * x^4 = 3 * y^2 + 1
  in ∃ (P E : set (ℝ × ℝ)),
       (∀ x y, x ∈ P ↔ ∃ t, (y^2 - 3 / 2)^2 = t) ∧
       (∀ x y, y ∈ E ↔ ∃ s, y^2 = s ∧ s + 6 * x^4 = 5 / 4) ∧
       (∀ x y, eq x y → (x, y) ∈ P ∪ E) :=
by
  let eq := ∀ x y : ℝ, y^4 - 6 * x^4 = 3 * y^2 + 1
  sorry

end graph_of_equation_is_parabola_and_ellipse_l228_228347


namespace johns_second_speed_l228_228166

-- Conditions
def rate1 := 45 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours
def total_distance := 255 -- miles

-- Definition to be proven
theorem johns_second_speed :
  let distance1 := rate1 * time1 in
  let distance2 := total_distance - distance1 in
  let speed2 := distance2 / time2 in
  speed2 = 55 :=
by
  sorry

end johns_second_speed_l228_228166


namespace distinct_values_from_expression_l228_228757

theorem distinct_values_from_expression : 
  let expr := 3^(3+3^3)
  let val1 := 3^(3+27)
  let val2 := 3^((3+3)^3)
  let val3 := (3^(3+3))^3
  let val4 := (3^3)^(3+3)
  ∃ (vals : set ℤ), vals = {val1, val2, val3, val4} ∧ vals.card = 3 := 
by {
  sorry
}

end distinct_values_from_expression_l228_228757


namespace secant_ratio_l228_228206

theorem secant_ratio (A B C D E F P T I IB IC : Point)
  (h1 : InscribedCircle A B C I D E F)
  (h2 : SecantThroughA A B C P)
  (h3 : InscribedCircle AB P IB)
  (h4 : InscribedCircle AC P IC)
  (h5 : TouchPoint IB IC P T) :
  (segment_ratio B P C = segment_ratio AB AC) :=
sorry

end secant_ratio_l228_228206


namespace painted_rooms_l228_228299

def total_rooms : ℕ := 12
def hours_per_room : ℕ := 7
def remaining_hours : ℕ := 49

theorem painted_rooms : total_rooms - (remaining_hours / hours_per_room) = 5 := by
  sorry

end painted_rooms_l228_228299


namespace positive_difference_eq_30_l228_228258

theorem positive_difference_eq_30 : 
  let x1 := 12
      x2 := -18
  in |x1 - x2| = 30 := 
by
  sorry

end positive_difference_eq_30_l228_228258


namespace height_difference_l228_228250

noncomputable def pipe_diameter : ℝ := 12
noncomputable def pipes_per_row_A : ℝ := 8
noncomputable def total_pipes : ℝ := 160

def rows_in_crate_A : ℝ := total_pipes / pipes_per_row_A
def height_crate_A : ℝ := rows_in_crate_A * pipe_diameter

noncomputable def staggered_distance : ℝ := (real.sqrt 3 / 2) * pipe_diameter
noncomputable def row_count_B : ℝ := 20
noncomputable def height_crate_B : ℝ := pipe_diameter + row_count_B * staggered_distance

noncomputable def positive_difference : ℝ := abs (height_crate_A - height_crate_B)

theorem height_difference :
  positive_difference = 20.16 := by
  sorry

end height_difference_l228_228250


namespace seq_a_arithmetic_sum_seq_b_lt_one_l228_228836

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * seq_a (n - 1) + 2^n

-- Prove 1: The sequence {a_n / 2^n} is arithmetic
theorem seq_a_arithmetic : 
  ∀ n : ℕ, ({seq_a n / 2^n} - {seq_a (n - 1) / 2^(n-1)} = 1) :=
sorry

-- Define b_n
def seq_b (n : ℕ) : ℕ :=
(n + 2) / ((n + 1) * seq_a n)

-- Prove 2: Sum of b_n < 1
theorem sum_seq_b_lt_one : 
  ∀ n : ℕ, ∑ i in range n, seq_b i < 1 :=
sorry

end seq_a_arithmetic_sum_seq_b_lt_one_l228_228836


namespace range_of_b_l228_228083

def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem range_of_b (a b : ℝ) (h : f a = g b) : 2 - Real.sqrt 2 ≤ b ∧ b ≤ 2 + Real.sqrt 2 :=
by
  sorry

end range_of_b_l228_228083


namespace largest_divisor_of_n4_minus_n2_l228_228003

theorem largest_divisor_of_n4_minus_n2 :
  ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_of_n4_minus_n2_l228_228003


namespace speedster_convertibles_l228_228272

theorem speedster_convertibles 
  (T : ℕ) 
  (h1 : T > 0)
  (h2 : 30 = (2/3 : ℚ) * T)
  (h3 : ∀ n, n = (1/3 : ℚ) * T → ∃ m, m = (4/5 : ℚ) * n) :
  ∃ m, m = 12 := 
sorry

end speedster_convertibles_l228_228272


namespace scalene_triangle_proof_l228_228549

variable (Triangle : Type)
variable (l1 l2 S : ℝ) -- lengths of longest & shortest bisectors and area respectively
variable [ScaleneTriangle : Triangle]

-- Define properties of a scalene triangle and the respective angle bisectors and area
axiom longest_bisector (T : Triangle) : ℝ
axiom shortest_bisector (T : Triangle) : ℝ
axiom area (T : Triangle) : ℝ

-- Given a scalene triangle
def is_scalene_triangle (T : Triangle) : Prop := True -- placeholder for actual definition

def problem_statement (T : Triangle) : Prop :=
  is_scalene_triangle T → longest_bisector T ^ 2 > sqrt 3 * area T ∧ sqrt 3 * area T > shortest_bisector T ^ 2

-- Main statement to be proved
theorem scalene_triangle_proof (T : Triangle) (h : is_scalene_triangle T) : problem_statement T := sorry

end scalene_triangle_proof_l228_228549


namespace students_on_8th_day_l228_228989

-- Define initial conditions
def initial_students : ℕ := 1 -- Jessica only
def friends_each_day (n : ℕ) : ℕ := 3^n

-- Total number of students knowing the secret on the nth day
noncomputable def total_students (n : ℕ) : ℕ :=
1 + (∑ k in Finset.range (n+1), 3^k)

-- Prove that on the 8th day, a total of 6560 students know the secret
theorem students_on_8th_day : total_students 8 = 6560 :=
by
  have h1 : 6560 = (3^9 - 1) / 2 := 
    by norm_num
  have h2 : total_students 8 = (3^9 - 1) / 2 :=
    by
      rw [total_students]
      rw [add_comm, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ]
      rw [Finset.range_succ, Finset.range_succ, Finset.range_succ]
      sorry
  exact Eq.trans h2 h1

end students_on_8th_day_l228_228989


namespace min_value_PM_PN_l228_228052

noncomputable def point := (ℝ × ℝ)

def C1 (M : point) : Prop := (M.1 - 2)^2 + (M.2 - 3)^2 = 1
def C2 (N : point) : Prop := (N.1 - 3)^2 + (N.2 - 4)^2 = 9
def on_x_axis (P : point) : Prop := P.2 = 0

theorem min_value_PM_PN :
  ∀ (P M N : point), on_x_axis P → C1 M → C2 N → 
  abs ((P.1 - M.1)^2 + (P.2 - M.2)^2)^(1/2) + abs ((P.1 - N.1)^2 + (P.2 - N.2)^2)^(1/2) = 5 * real.sqrt 2 - 4 := sorry

end min_value_PM_PN_l228_228052


namespace tangent_line_at_point_l228_228809

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  x + √3 * y - 4 = 0

theorem tangent_line_at_point : 
  circle_eq 1 (√3) → tangent_line_eq x y :=
begin
  -- proof here
  sorry
end

end tangent_line_at_point_l228_228809


namespace range_of_b_l228_228443

open Real

-- Definitions for the circles and the condition
def C1 (a : ℝ) : set (ℝ × ℝ) := { p | (p.1 - a)^2 + p.2^2 = 4 }
def C2 (b : ℝ) : set (ℝ × ℝ) := { p | p.1^2 + (p.2 - b)^2 = 1 }
def intersecting_points (A B : ℝ × ℝ) := ∃ (a b : ℝ), A ∈ C1 a ∧ B ∈ C1 a ∧ A ∈ C2 b ∧ B ∈ C2 b

theorem range_of_b (a b : ℝ) (A B : ℝ × ℝ) (h1 : intersecting_points A B) (h2 : dist A B = 2) :
  abs b ≤ sqrt 3 :=
begin
  -- We skip the proof using the sorry keyword
  sorry
end

end range_of_b_l228_228443


namespace evaluate_expression_l228_228963

theorem evaluate_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x > y) (hyz : y > z) :
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) :=
by
  sorry

end evaluate_expression_l228_228963


namespace parallelogram_lambda_mu_product_l228_228487

variables {A B C D E F M : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (AB AD BC AC AE AM : ℝ)

-- Parallelogram and vector conditions
def is_parallelogram (ABCD : Prop) : Prop :=
  (B - A = D - C) ∧ (C - A = D - B)

variables {is_parallelogram_ABCD : is_parallelogram}
variables (E_mid_AB : E = (A + B) / 2)
variables (F_trisection_BC : F = (B + 2 * C) / 3)
variables (CE_DF_intersect_M : ∃ t u : ℝ, M = C + t * (E - C) ∧ M = D + u * (F - D))
variables (hAM : AM = λ * AB + μ * AD)
variables (λ μ : ℝ)

-- Conclusion to be proved
theorem parallelogram_lambda_mu_product :
  (is_parallelogram_ABCD) →
  (E_mid_AB) →
  (F_trisection_BC) →
  (CE_DF_intersect_M) →
  (hAM) →
  λ * μ = 3 / 8 :=
sorry

end parallelogram_lambda_mu_product_l228_228487


namespace value_of_f_neg2015_plus_f_2016_l228_228415

noncomputable def f (x: ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 1 then 2^x + 1 else
  if h : x = 0 then 0 else 
  if h : x + 2 = 0 ∨ (x - 2) + 2 ≤ 0 then 
     -f (x + 2) else
     sorry -- Define the exact behavior for all other x values

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

axiom f_odd : is_odd f

axiom f_periodic_2 : ∀ x, 0 < x → f (x + 2) = -f (x)

axiom f_piecewise : ∀ x, 0 < x ∧ x ≤ 1 → f (x) = 2^x + 1

theorem value_of_f_neg2015_plus_f_2016 : f (-2015) + f (2016) = 3 :=
  sorry

end value_of_f_neg2015_plus_f_2016_l228_228415


namespace chosen_number_is_5_l228_228301

theorem chosen_number_is_5 (x : ℕ) (h_pos : x > 0)
  (h_eq : ((10 * x + 5 - x^2) / x) - x = 1) : x = 5 :=
by
  sorry

end chosen_number_is_5_l228_228301


namespace landscape_length_l228_228278

theorem landscape_length (b l : ℕ) (playground_area : ℕ) (total_area : ℕ) 
  (h1 : l = 4 * b) (h2 : playground_area = 1200) (h3 : total_area = 3 * playground_area) (h4 : total_area = l * b) :
  l = 120 := 
by 
  sorry

end landscape_length_l228_228278


namespace relationship_among_a_b_c_l228_228797

noncomputable def a : ℝ := (3/4) ^ (-3/4)
noncomputable def b : ℝ := Real.logb 2 (1/6)
noncomputable def c : ℝ := Real.exp (-1/2)

theorem relationship_among_a_b_c : a > c ∧ c > b :=
by
  -- Proof can be provided here. For now, use sorry.
  sorry

end relationship_among_a_b_c_l228_228797


namespace meat_per_slice_is_22_l228_228979

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l228_228979


namespace equation_solution_unique_l228_228357

theorem equation_solution_unique (x y : ℤ) : 
  x^4 = y^2 + 2*y + 2 ↔ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end equation_solution_unique_l228_228357


namespace max_c_for_log_inequality_l228_228503

theorem max_c_for_log_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / (3 + Real.log b / Real.log a) + 1 / (3 + Real.log a / Real.log b) ≥ c) :=
by
  use 1 / 3
  sorry

end max_c_for_log_inequality_l228_228503


namespace liking_patterns_count_l228_228244

open Set

def Friends := {C : Prop, D : Prop, E : Prop}
def Tracks := Fin 5

-- Define the sets
def CD := {t : Tracks // (Friends.C ∧ Friends.D ∧ ¬Friends.E)}
def DE := {t : Tracks // (Friends.D ∧ Friends.E ∧ ¬Friends.C)}
def EC := {t : Tracks // (Friends.E ∧ Friends.C ∧ ¬Friends.D)}
def C := {t : Tracks // (Friends.C ∧ ¬Friends.D ∧ ¬Friends.E)}
def D := {t : Tracks // (Friends.D ∧ ¬Friends.C ∧ ¬Friends.E)}
def E := {t : Tracks // (Friends.E ∧ ¬Friends.C ∧ ¬Friends.D)}
def N := {t : Tracks // (¬Friends.C ∧ ¬Friends.D ∧ ¬Friends.E)}

theorem liking_patterns_count : 
  ∃ (CD : Fin 5 → Prop) 
    (DE : Fin 5 → Prop) 
    (EC : Fin 5 → Prop) 
    (C : Fin 5 → Prop) 
    (D : Fin 5 → Prop) 
    (E : Fin 5 → Prop) 
    (N : Fin 5 → Prop), 
    (∀ t, ¬(Friends.C t ∧ Friends.D t ∧ Friends.E t)) ∧
    (∃ t, CD t) ∧ (∃ t, DE t) ∧ (∃ t, EC t) ∧
    (fintype.card {t : Tracks // N t} + fintype.card {t : Tracks // C t} + 
     fintype.card {t : Tracks // D t} + fintype.card {t : Tracks // E t} + 
     fintype.card {t : Tracks // CD t} + fintype.card {t : Tracks // DE t} + 
     fintype.card {t : Tracks // EC t} = 5) ∧
    (Σ₀ t : Tracks, CD t ∨ DE t ∨ EC t ∨ C t ∨ D t ∨ E t ∨ N t) = 88 
:= sorry

end liking_patterns_count_l228_228244


namespace simple_interest_years_l228_228600

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end simple_interest_years_l228_228600


namespace distance_P_to_O_l228_228421

noncomputable def distance_from_point_to_planes (P O : ℝ × ℝ × ℝ) : ℝ :=
  let d₁ := abs (P.1 - O.1)
  let d₂ := abs (P.2 - O.2)
  let d₃ := abs (P.3 - O.3)
  real.sqrt (d₁^2 + d₂^2 + d₃^2)

theorem distance_P_to_O {P O : ℝ × ℝ × ℝ} (h₁ : |P.1 - O.1| = 1) 
                                        (h₂ : |P.2 - O.2| = 2) 
                                        (h₃ : |P.3 - O.3| = 3) :
  distance_from_point_to_planes P O = real.sqrt 14 :=
by
  rw [distance_from_point_to_planes]
  rw [h₁, h₂, h₃]
  norm_num
  exact real.sqrt_eq_rfl.symm

end distance_P_to_O_l228_228421


namespace probability_YD_6_half_sqrt3_l228_228247

open ProbabilityTheory

noncomputable theory

def triangle_XYZ : Type := ℝ

variables (X Y Z P D : triangle_XYZ)
variables (angle_XYZ : RealAngle)
variables (angle_YXZ : RealAngle)
variables (XY XZ YZ : ℝ)
variables (R : set triangle_XYZ) -- R is the random variable representing point P inside the triangle

-- Definitions derived from the conditions
def conditions :=
  angle_XYZ = 90 ∧ 
  angle_YXZ = 45 ∧ 
  XY = 12 ∧ 
  XZ = 12 ∧ 
  YZ = 12 * Real.sqrt 2 ∧ 
  D ∈ R

-- Definition representing the probability
def probability_YD_gt_6 : ℝ := do
  let XD' := 6 * Real.sqrt 3
  let XZ := 12
  XD' / XZ

theorem probability_YD_6_half_sqrt3 :
  ∀ P, ∃ (D : triangle_XYZ),
  probability_YD_gt_6 = Real.sqrt 3 / 2 :=
by
  sorry

end probability_YD_6_half_sqrt3_l228_228247


namespace test_takers_answered_both_correctly_l228_228455

variable (P_A : ℕ) (P_B : ℕ) (P_neither : ℕ)

-- Definitions and assumptions
def P_A_correctly := 85
def P_B_correctly := 65
def P_neither_correctly := 5

theorem test_takers_answered_both_correctly :
  P_A_correctly + P_B_correctly - P_A ∩ P_B + P_neither_correctly = 100 → P_A ∩ P_B = 55 :=
by
  sorry

end test_takers_answered_both_correctly_l228_228455


namespace factor_polynomial_l228_228822

theorem factor_polynomial (a b c : ℝ) : 
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
by 
  sorry

end factor_polynomial_l228_228822


namespace prism_is_five_sided_l228_228457

-- Definitions based on problem conditions
def prism_faces (total_faces base_faces : Nat) := total_faces = 7 ∧ base_faces = 2

-- Theorem to prove based on the conditions
theorem prism_is_five_sided (total_faces base_faces : Nat) (h : prism_faces total_faces base_faces) : total_faces - base_faces = 5 :=
sorry

end prism_is_five_sided_l228_228457


namespace bags_not_on_promotion_l228_228667

def total_dog_food : ℕ := 750
def total_cat_food : ℕ := 350
def total_bird_food : ℕ := 200

def percent_on_sale_dog_food : ℕ := 20
def percent_on_sale_cat_food : ℕ := 25
def percent_on_sale_bird_food : ℕ := 15

def bags_on_promotion (total: ℕ) (percent: ℕ) : ℕ :=
  (total * percent) / 100

theorem bags_not_on_promotion : 
  bags_not_on_promotion total_dog_food percent_on_sale_dog_food = 600 ∧
  bags_not_on_promotion total_cat_food percent_on_sale_cat_food = 263 ∧
  bags_not_on_promotion total_bird_food percent_on_sale_bird_food = 170 :=
begin
  sorry
end

end bags_not_on_promotion_l228_228667


namespace parabola_condition_l228_228068

/-- Given the point (3,0) lies on the parabola y = 2x^2 + (k + 2)x - k,
    prove that k = -12. -/
theorem parabola_condition (k : ℝ) (h : 0 = 2 * 3^2 + (k + 2) * 3 - k) : k = -12 :=
by 
  sorry

end parabola_condition_l228_228068


namespace factorial_base8_trailing_zeros_l228_228892

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228892


namespace end_same_digit_l228_228053

theorem end_same_digit
  (a b : ℕ)
  (h : (2 * a + b) % 10 = (2 * b + a) % 10) :
  a % 10 = b % 10 :=
by
  sorry

end end_same_digit_l228_228053


namespace general_term_formula_sum_of_sequence_minimum_lambda_l228_228420

section
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (λ : ℝ)

-- Conditions
hypothesis (h_sum_Sn : ∀ n, S n = n)
hypothesis (h_Sn_an : ∀ n, 2 * real.sqrt (S n) = a n + 1)

-- Question 1: General term formula for a_n
theorem general_term_formula (n : ℕ) : a n = 2 * n - 1 :=
sorry

-- Question 2: Formula for T_n
def b_n (n : ℕ) := (a n + 3) / 2
def T_n (n : ℕ) := ∑ i in finset.range n, 1 / (b i * b (i + 1))

theorem sum_of_sequence (n : ℕ) : T n = n / (2 * (n + 2)) :=
sorry

-- Question 3: Minimum value of λ 
theorem minimum_lambda (h : ∀ n, T n ≤ λ * b (n + 1)) : λ = 1 / 16 :=
sorry
end

end general_term_formula_sum_of_sequence_minimum_lambda_l228_228420


namespace contestant_final_score_l228_228289

theorem contestant_final_score (score_content score_skills score_effects : ℕ) 
                               (weight_content weight_skills weight_effects : ℕ) :
    score_content = 90 →
    score_skills  = 80 →
    score_effects = 90 →
    weight_content = 4 →
    weight_skills  = 2 →
    weight_effects = 4 →
    (score_content * weight_content + score_skills * weight_skills + score_effects * weight_effects) / 
    (weight_content + weight_skills + weight_effects) = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end contestant_final_score_l228_228289


namespace circle_and_line_conditions_l228_228401

noncomputable def circle_equation : Prop :=
  let center : ℝ × ℝ := (3, 4)
  let radius : ℝ := 5
  ∀ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25

noncomputable def line_equation (l : ℝ) : Prop :=
  let P : ℝ × ℝ := (-2, 0)
  l = (-2) ∨ (9 * l + 40 * P.2 + 18 = 0)

theorem circle_and_line_conditions :
  (circle C passes through (0, 0), (6, 0), and (0, 8)) →
  (line l passes through (-2, 0) and is tangent to circle C) →
  circle_equation ∧ ∃ l, line_equation l :=
by
  sorry

end circle_and_line_conditions_l228_228401


namespace intersection_of_A_and_Z_l228_228087

def A : set ℝ := {x | x^2 < 3 * x + 4}

def Z : set ℤ := set.univ

theorem intersection_of_A_and_Z : 
  A ∩ {x : ℝ | x ∈ Z} = {0, 1, 2, 3} :=
sorry

end intersection_of_A_and_Z_l228_228087


namespace domain_ln_x_plus_one_l228_228575

theorem domain_ln_x_plus_one : 
  { x : ℝ | ∃ y : ℝ, y = x + 1 ∧ y > 0 } = { x : ℝ | x > -1 } :=
by
  sorry

end domain_ln_x_plus_one_l228_228575


namespace peter_sold_65_regular_pumpkins_l228_228993

theorem peter_sold_65_regular_pumpkins : 
  ∃ (J R : ℕ), J + R = 80 ∧ 9 * J + 4 * R = 395 ∧ R = 65 :=
by
  exists 15
  exists 65
  simp
  split
  exact rfl
  simp
  split
  exact rfl
  exact rfl

end peter_sold_65_regular_pumpkins_l228_228993


namespace increasing_interval_l228_228759

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * real.log x 

theorem increasing_interval : 
  ∀ x, x > real.sqrt 3 / 3 → 0 < f' x :=
begin
  assume x hx,
  sorry
end

end increasing_interval_l228_228759


namespace factorial_base_8_zeroes_l228_228852

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228852


namespace color_stamps_sold_l228_228207

theorem color_stamps_sold :
    let total_stamps : ℕ := 1102609
    let black_and_white_stamps : ℕ := 523776
    total_stamps - black_and_white_stamps = 578833 := 
by
  sorry

end color_stamps_sold_l228_228207


namespace solution_l228_228340

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions:
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (-x + 1) = f (x + 1)) ∧ f (-1) = 1 :=
sorry

theorem solution : f 2017 = -1 :=
sorry

end solution_l228_228340


namespace positive_difference_of_solutions_l228_228254

theorem positive_difference_of_solutions : 
    (∀ x : ℝ, |x + 3| = 15 → (x = 12 ∨ x = -18)) → 
    (abs (12 - (-18)) = 30) :=
begin
  intros,
  sorry
end

end positive_difference_of_solutions_l228_228254


namespace arithmetic_sequence_sum_l228_228806

variable {α : Type*} [LinearOrderedField α]

def sum_n_terms (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a₁ : α) (h : sum_n_terms a₁ 1 4 = 1) :
  sum_n_terms a₁ 1 8 = 18 := by
  sorry

end arithmetic_sequence_sum_l228_228806


namespace find_base_number_l228_228456

theorem find_base_number (y : ℕ) (base : ℕ) (h : 9^y = base ^ 16) (hy : y = 8) : base = 3 :=
by
  -- We skip the proof steps and insert sorry here
  sorry

end find_base_number_l228_228456


namespace find_f_of_2013_l228_228899

/-- We define f : ℕ → ℕ, with the given conditions:
 - For all n: f(f(n)) + f(n) = 2n + 3
 - f(0) = 1
We aim to prove that f(2013) = 2014.
-/ 
theorem find_f_of_2013
  (f : ℕ → ℕ)
  (h₀ : f(f(n)) + f(n) = 2 * n + 3)
  (h₁ : f(0) = 1) :
  f(2013) = 2014 := 
sorry

end find_f_of_2013_l228_228899


namespace range_of_a_l228_228437

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < 2 → (a+1)*x > 2*a+2) → a < -1 :=
by
  sorry

end range_of_a_l228_228437


namespace chessboard_paradox_l228_228847

theorem chessboard_paradox :
  ∃ (pieces : List (Set (Fin 8 × Fin 8))), 
    (∃ f : Fin 8 × Fin 8 → Fin 9 × Fin 7, 
      (∀ piece ∈ pieces, ∀ (x y : Fin 8), f (x,y) ∈ piece → (x, y) ∈ piece)) → 
    (64 = 63) :=
sorry

end chessboard_paradox_l228_228847


namespace XS_squared_l228_228478

noncomputable def triangle_XYZ (X Y Z : ℝ → ℝ → ℝ) : Prop :=
  X (0, 0) ∧ Y (14, 0) ∧ Z (14 * cos 45, 14 * sin 45)

noncomputable def point_G (X Y Z G : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), G (x, y) ∧ x = 0

noncomputable def point_E (X Y Z E : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), E (x, y) ∧ x = y

noncomputable def point_L (Y Z L : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), L (x / 2)

noncomputable def point_Q (G L Q : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), Q (x, y) ∧ x = y

noncomputable def point_S (X E S : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), S (x, y) ∧ x = XS

theorem XS_squared (X Y Z G E L Q S : ℝ → ℝ → ℝ) 
  (h1 : triangle_XYZ X Y Z)
  (h2 : point_G X Y Z G)
  (h3 : point_E X Y Z E)
  (h4 : point_L Y Z L)
  (h5 : point_Q G L Q)
  (h6 : point_S X E S) : XS^2 = 49 :=
sorry

end XS_squared_l228_228478


namespace f_bounded_by_inverse_l228_228042

theorem f_bounded_by_inverse (f : ℕ → ℝ) (h_pos : ∀ n, 0 < f n) (h_rec : ∀ n, (f n)^2 ≤ f n - f (n + 1)) :
  ∀ n, f n < 1 / (n + 1) :=
by
  sorry

end f_bounded_by_inverse_l228_228042


namespace partition_graph_subsets_l228_228280

open SimpleGraph

theorem partition_graph_subsets (V : Type) [Fintype V] (G : SimpleGraph V) [DecidableRel G.Adj]
  (hconn : G.Connected) (a b c : ℕ) (h_le : 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ Fintype.card V)
  (h_sum : a + b + c = Fintype.card V)
  (hcomp : ∀ (v : V), ∃ (H : G.NeighborSet v).card ≥ a, H.Connected) :
  ∃ (A B C : Finset V), A.card = a ∧ B.card = b ∧ C.card = c ∧
    ∃ (G_A G_B G_C : SimpleGraph V), G_A.InducedSubgraph A ∧ G_B.InducedSubgraph B ∧ G_C.InducedSubgraph C ∧ 
      (G_A.Connected ∧ G_B.Connected ∨ G_B.Connected ∧ G_C.Connected ∨ G_A.Connected ∧ G_C.Connected) :=
sorry

end partition_graph_subsets_l228_228280


namespace eval_math_expr_l228_228354

def sqrt_16_eq_4 : Prop := (sqrt 16) = 4
def inner_expr_value : Prop := (4 + 2)^2 = 36
def bracket_expr_value : Prop := 7 - 36 = -29
def multiplication_value : Prop := -5 * (-29) * 3 = 435
def subtraction_value : Prop := 6 - 435 = -429

theorem eval_math_expr :
  sqrt_16_eq_4 ∧
  inner_expr_value ∧
  bracket_expr_value ∧
  multiplication_value ∧
  subtraction_value → 
  (6 - 5 * (7 - (sqrt 16 + 2)^2) * 3) = -429 :=
by
  intro h
  sorry

end eval_math_expr_l228_228354


namespace octagon_area_correct_l228_228691

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228691


namespace sequence_values_l228_228720

noncomputable def sequence (k : ℕ) : ℕ → ℕ
| 1          := 1
| (n + 1) := (sequence k n + n + 1) % k

def is_power_of_2 (k : ℕ) : Prop :=
∃ n : ℕ, k = 2^n

theorem sequence_values (k : ℕ) : 
  (∀ v ∈ list.range k, ∃ n, sequence k n = v) ↔ is_power_of_2 k :=
sorry

end sequence_values_l228_228720


namespace zeroes_at_end_base_8_of_factorial_15_l228_228879

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228879


namespace count_ordered_pairs_xy_12320_l228_228591

theorem count_ordered_pairs_xy_12320 :
  let n := 12320 in
  let factorization := (2^4 * 5 * 7 * 11) in
  (n = factorization) →
  (∃! (count : ℕ), count = 40 ∧ ∀ (x y : ℕ), (x * y = n) → True) :=
by
  sorry

end count_ordered_pairs_xy_12320_l228_228591


namespace miles_reads_128_pages_l228_228534

noncomputable def pages_read (daily_reading_fraction : ℚ) 
  (time_fraction_per_type : ℚ) (pages_per_hour_novels : ℚ) 
  (pages_per_hour_graphic_novels : ℚ) (pages_per_hour_comic_books : ℚ) : ℚ :=
  let total_hours := 24 * daily_reading_fraction in
  let hours_per_type := total_hours * time_fraction_per_type in
  let pages_novels := pages_per_hour_novels * hours_per_type in
  let pages_graphic_novels := pages_per_hour_graphic_novels * hours_per_type in
  let pages_comic_books := pages_per_hour_comic_books * hours_per_type in
  pages_novels + pages_graphic_novels + pages_comic_books

theorem miles_reads_128_pages :
  pages_read (1/6) (1/3) 21 30 45 = 128 := by
  sorry

end miles_reads_128_pages_l228_228534


namespace rod_length_l228_228384

theorem rod_length (pieces : ℕ) (length_per_piece_cm : ℕ) (total_length_m : ℝ) :
  pieces = 35 → length_per_piece_cm = 85 → total_length_m = 29.75 :=
by
  intros h1 h2
  sorry

end rod_length_l228_228384


namespace solve_equation_l228_228212

theorem solve_equation (x y z : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9) (h_eq : 1 / (x + y + z) = (x * 100 + y * 10 + z) / 1000) :
  x = 1 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end solve_equation_l228_228212


namespace domain_of_f_l228_228576

def domain (f : ℝ → ℝ) (dom : set ℝ) : Prop := ∀ x, x ∈ dom ↔ f x = ∅

def f (x : ℝ) : ℝ := 
  sqrt (x + 1) + (1 / (3 - x))

theorem domain_of_f : setOf(λ x, x ≥ -1 ∧ x ≠ 3) :=
by
  -- Proof statement here
  sorry

end domain_of_f_l228_228576


namespace company_prob_no_software_contract_l228_228592

noncomputable def probability_no_software_contract : ℚ :=
  let P_H := 4 / 5
  let P_H_union_S := 5 / 6
  let P_H_inter_S := 11 / 30
  let P_S := P_H_union_S - P_H + P_H_inter_S
  let P_S_complement := 1 - P_S
  P_S_complement

theorem company_prob_no_software_contract :
  probability_no_software_contract = 3 / 5 :=
by
  -- We start by evaluating the probability of getting a software contract, P(S)
  let P_H := 4 / 5
  let P_H_union_S := 5 / 6
  let P_H_inter_S := 11 / 30
  let P_S := P_H_union_S - P_H + P_H_inter_S
  -- Evaluate P(S)
  have P_S_value : P_S = 2 / 5 := by sorry
  -- Evaluate the complement: P(S')
  have P_S_complement : 1 - P_S = 3 / 5 := by sorry
  -- Conclude the proof
  exact P_S_complement

end company_prob_no_software_contract_l228_228592


namespace correct_option_D_l228_228631

theorem correct_option_D 
  (A : (sqrt 36 = 6 ∨ sqrt 36 = -6) → False)
  (B : sqrt ((-3 : ℤ)^2) = 3)
  (C : (-4 : ℤ) < 0 → (-sqrt (-4 : ℤ) = 2) → False)
  (D : (cbrt (-8 : ℤ) = -2)) : 
  D := by 
  sorry

end correct_option_D_l228_228631


namespace imaginary_part_of_z_l228_228460

theorem imaginary_part_of_z (z : ℂ) (h : (1 + complex.i) * z = complex.abs (1 + complex.i)) :
  complex.im z = - (real.sqrt 2) / 2 := 
sorry

end imaginary_part_of_z_l228_228460


namespace bird_families_flew_away_l228_228266

def initial_families : ℕ := 41
def left_families : ℕ := 14

theorem bird_families_flew_away :
  initial_families - left_families = 27 :=
by
  -- This is a placeholder for the proof
  sorry

end bird_families_flew_away_l228_228266


namespace positive_difference_of_solutions_l228_228748

theorem positive_difference_of_solutions :
  ∃ (x : ℝ), (∃ (y : ℝ), (x = 2 * real.sqrt 6 ∧ y = -2 * real.sqrt 6)
    ∧ (|x - y| = 4 * real.sqrt 6)) := 
sorry

end positive_difference_of_solutions_l228_228748


namespace polynomial_floor_eq_floor_polynomial_eq_linear_term_l228_228002

theorem polynomial_floor_eq_floor_polynomial_eq_linear_term (P : ℝ[X]) :
  (∀ x : ℝ, P.eval ⌊x⌋ = ⌊P.eval x⌋) ↔ ∃ k : ℤ, P = polynomial.C 1 * polynomial.X + polynomial.C (k : ℝ) :=
begin
  sorry
end

end polynomial_floor_eq_floor_polynomial_eq_linear_term_l228_228002


namespace probability_at_least_one_male_and_one_female_l228_228117

theorem probability_at_least_one_male_and_one_female 
  (M F: ℕ) (M_eq : M = 5) (F_eq : F = 2) :
  let total_choices := Nat.choose (M + F) 3,
      event1 := Nat.choose M 1 * Nat.choose F 2,
      event2 := Nat.choose M 2 * Nat.choose F 1,
      desired_event := event1 + event2 in
  (desired_event : ℚ) / total_choices = 5 / 7 :=
by sorry

end probability_at_least_one_male_and_one_female_l228_228117


namespace angle_A_in_triangle_l228_228477

theorem angle_A_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : a^2 + b^2 + c^2) 
  (h_angle_C : C = 60) 
  (h_b : b = Real.sqrt 6) 
  (h_c : c = 3) : 
  A = 75 :=
sorry

end angle_A_in_triangle_l228_228477


namespace tan_2alpha_of_sin_cos_ratio_l228_228108

theorem tan_2alpha_of_sin_cos_ratio (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1/2) : 
  tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_2alpha_of_sin_cos_ratio_l228_228108


namespace sum_of_three_element_subsets_l228_228390

def P : Finset ℕ := (Finset.range 10).image (λ n => 2 * n + 1)

lemma CardP : P.card = 10 := by 
	sorry

lemma three_element_subsets_card : (P.subsetsLen 3).card = Nat.choose 10 3 := by 
  exact Finset.card_subsetsLen 3 (CardP ▸ by simp) 

theorem sum_of_three_element_subsets 
  (P_i : Finset (Finset ℕ))
  (hP_i : P_i = P.subsetsLen 3) :
  ∑ x in P_i, x.card = 3600 :=
by
  have card_eq : P.card = 10 := CardP
  have h_sub_len_3 : (P.subsetsLen 3).card = 120 := by
    rw [three_element_subsets_card]
    exact (nat.choose 10 3).symm
  rw [← hP_i, Finset.sum_const, h_sub_len_3]
  exact (calc 
    ∀ x ∈ P.subsetsLen 3, x.card = 3
    ∑ _ in P.subsetsLen 3, 3 = 3 * (P.subsetsLen 3).card : Finset.sum_const_nat
    3 * 120 = 360 : by norm_num)

end sum_of_three_element_subsets_l228_228390


namespace locus_midpoint_l228_228436

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

def perpendicular_rays (OA OB : ℝ × ℝ) : Prop := (OA.1 * OB.1 + OA.2 * OB.2) = 0 -- Dot product zero for perpendicularity

-- Given the hyperbola and perpendicularity conditions, prove the locus equation
theorem locus_midpoint (x y : ℝ) :
  (∃ A B : ℝ × ℝ, hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2 ∧ perpendicular_rays A B ∧
  x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → 3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) :=
sorry

end locus_midpoint_l228_228436


namespace no_four_consecutive_nines_l228_228262

theorem no_four_consecutive_nines (A : ℤ) : 
  (∃ k : ℕ, ((A * 10^k) % 1989) / 1989 ≥ 0.9999) → false := 
  by
  sorry

end no_four_consecutive_nines_l228_228262


namespace train_cross_time_l228_228154

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  (speed * 1000) / 3600

noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time (length : ℝ) (speed_km_per_hr : ℝ) :
  length = 100 → speed_km_per_hr = 144 → time_to_cross length (speed_km_per_hr_to_m_per_s speed_km_per_hr) = 2.5 :=
by
  intros length_eq speed_eq
  rw [length_eq, speed_eq]
  simp [speed_km_per_hr_to_m_per_s, time_to_cross]
  norm_num
  sorry

end train_cross_time_l228_228154


namespace find_m_l228_228378

theorem find_m (m : ℕ) (h₁ : 1 + 2 + 3 + ... + (m - 1) < 30) (h₂ : 1 + 2 + 3 + ... + m ≥ 30) : m = 8 :=
sorry

end find_m_l228_228378


namespace hands_of_clock_align_twice_l228_228104

theorem hands_of_clock_align_twice :
  ∀ (h m s : ℕ), (∃ (n : ℕ), n = 24) →
  (∃ (t0 t1 : ℕ), t0 = 0 ∧ t1 = 12 ∧ 
  (∀ t ∈ [t0, t0 + 1, t0 + 2, ..., t1 - 1], 
    (h * 3600 + m * 60 + s) % (12 * 3600) ≠ (t * 3600) % (12 * 3600)) ∧ 
  (∀ t ∈ [t1, t1 + 1, t1 + 2, ..., 24 * 3600 - 1], 
    (h * 3600 + m * 60 + s) % (12 * 3600) ≠ (t * 3600) % (12 * 3600))) →
  𝓟 (hands_of_clock_align h m s 24) = 2 :=
begin 
  sorry 
end

end hands_of_clock_align_twice_l228_228104


namespace kitchen_length_l228_228167

-- Define the conditions
def tile_area : ℕ := 6
def kitchen_width : ℕ := 48
def number_of_tiles : ℕ := 96

-- The total area is the number of tiles times the area of each tile
def total_area : ℕ := number_of_tiles * tile_area

-- Statement to prove the length of the kitchen
theorem kitchen_length : (total_area / kitchen_width) = 12 :=
by
  sorry

end kitchen_length_l228_228167


namespace total_squares_in_4x4_grid_l228_228103

-- Define the grid size
def grid_size : ℕ := 4

-- Define a function to count the number of k x k squares in an n x n grid
def count_squares (n k : ℕ) : ℕ :=
  (n - k + 1) * (n - k + 1)

-- Total number of squares in a 4 x 4 grid
def total_squares (n : ℕ) : ℕ :=
  count_squares n 1 + count_squares n 2 + count_squares n 3 + count_squares n 4

-- The main theorem asserting the total number of squares in a 4 x 4 grid is 30
theorem total_squares_in_4x4_grid : total_squares grid_size = 30 := by
  sorry

end total_squares_in_4x4_grid_l228_228103


namespace octagon_area_l228_228700

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228700


namespace minimize_S_n_l228_228508

variable {a : ℕ → ℤ} {S : ℕ → ℤ}
noncomputable def d : ℤ := 2

-- Condition 1: a_4 = -6
def condition1 : Prop := a 4 = -6

-- Condition 2: a_8 = 2
def condition2 : Prop := a 8 = 2

-- General term of the sequence: a_n = 2n - 14
def a_n (n : ℕ) : ℤ := 2 * n - 14

-- Sum of the first n terms of the arithmetic sequence S_n
def S_n (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

-- Proving that n = 6 or n = 7 minimizes S_n
theorem minimize_S_n (n : ℕ) (h1 : condition1) (h2 : condition2) : n = 6 ∨ n = 7 := 
sorry

end minimize_S_n_l228_228508


namespace length_MN_l228_228314

-- Definition of the triangle and its points
variables (A B C M N : Type) [metric_space A]
variables (d_AB d_BC d_AC : ℝ)

/-- Triangle with given side lengths -/
def triangle_lengths (A B C : ℝ) : Prop :=
  d_AB = 50 ∧ d_BC = 20 ∧ d_AC = 40

/-- Point M on AB such that CM is the angle bisector of ∠ACB -/
def angle_bisector (C M A B : Prop) : Prop := sorry

/-- Point N where CN is the altitude to side AB -/
def altitude (C N A B : Prop) : Prop := sorry

/-- Prove the length of MN is 11/3 cm -/
theorem length_MN (A B C M N : Type) [metric_space A]
  (d_AB d_BC d_AC : ℝ) (h_triangle : triangle_lengths A B C)
  (h_M : angle_bisector C M A B) (h_N : altitude C N A B) :
  distance M N = 11 / 3 := sorry

end length_MN_l228_228314


namespace seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l228_228788

theorem seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums
  (a1 a2 a3 a4 a5 a6 a7 : Nat) :
  ¬ ∃ (s : Finset Nat), (s = {a1 + a2, a1 + a3, a1 + a4, a1 + a5, a1 + a6, a1 + a7,
                             a2 + a3, a2 + a4, a2 + a5, a2 + a6, a2 + a7,
                             a3 + a4, a3 + a5, a3 + a6, a3 + a7,
                             a4 + a5, a4 + a6, a4 + a7,
                             a5 + a6, a5 + a7,
                             a6 + a7}) ∧
  (∃ (n : Nat), s = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9}) := 
sorry

end seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l228_228788


namespace pen_count_l228_228270

theorem pen_count (start_pens mike_pens sharons_pens : ℕ) 
  (doubling_factor percentage_increase : ℚ) 
  (start_pens = 25)
  (mike_pens = 22)
  (doubling_factor = 2) 
  (percentage_increase = 0.35) 
  (sharons_pens = 19) :
  let cindy_pens := (start_pens + mike_pens) * doubling_factor
  let increased_cindy_pens := cindy_pens + (cindy_pens * percentage_increase).nat_abs
  let remaining_after_sharon := increased_cindy_pens - sharons_pens
  let alex_pens := remaining_after_sharon / 3
  let final_pens := remaining_after_sharon - alex_pens
  final_pens = 72 :=
by
  sorry

end pen_count_l228_228270


namespace average_first_100_odd_natural_numbers_l228_228359

def odd_sequence (n : ℕ) : ℕ := 2 * n + 1

theorem average_first_100_odd_natural_numbers :
  (∑ i in Finset.range 100, odd_sequence i) / 100 = 100 :=
by
  sorry

end average_first_100_odd_natural_numbers_l228_228359


namespace determine_b_l228_228338

def f (x : ℝ) : ℝ := Real.log x - (1/3) * x + (2/3) * (1/x) - 1
def g (x : ℝ) (b : ℝ) : ℝ := x^2 - 2 * b * x - (5/12)

theorem determine_b :
  (∀ x_1 : ℝ, x_1 ∈ Set.Icc (1:ℝ) 2 →
      ∃ x_2 : ℝ, x_2 ∈ Set.Icc (0:ℝ) 1 ∧ f x_1 ≥ g x_2 (1/2)) →
  (b : ℝ) :
  b ∈ Set.Ici (1/2 : ℝ) := sorry

end determine_b_l228_228338


namespace factorial_base_8_zeroes_l228_228851

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228851


namespace matrix_multiplication_is_correct_l228_228335

def mat_mul (A B : Matrix (Fin 2) (Fin 2) ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  Matrix.mul A B

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-3, 5; 4, 2]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -4; -1, 3]

def result : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-11, 27; 6, -10]

theorem matrix_multiplication_is_correct :
  mat_mul A B = result := by
  sorry

end matrix_multiplication_is_correct_l228_228335


namespace find_divisible_by_3_l228_228605

theorem find_divisible_by_3 (n : ℕ) : 
  (∀ k : ℕ, k ≤ 12 → (3 * k + 12) ≤ n) ∧ 
  (∀ m : ℕ, m ≥ 13 → (3 * m + 12) > n) →
  n = 48 :=
by
  sorry

end find_divisible_by_3_l228_228605


namespace nancy_total_spending_l228_228198

theorem nancy_total_spending :
  let this_month_games := 9
  let this_month_price := 5
  let last_month_games := 8
  let last_month_price := 4
  let next_month_games := 7
  let next_month_price := 6
  let total_cost := (this_month_games * this_month_price) +
                    (last_month_games * last_month_price) +
                    (next_month_games * next_month_price)
  total_cost = 119 :=
by
  sorry

end nancy_total_spending_l228_228198


namespace infinitely_many_odd_floors_l228_228894

theorem infinitely_many_odd_floors :
  ∃ᶠ n in Nat, (⌊(2^n : ℝ) / 17⌋) % 2 = 1 := 
sorry

end infinitely_many_odd_floors_l228_228894


namespace find_simple_interest_years_l228_228598

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

constant P_c : ℝ := 4000
constant r_c : ℝ := 0.10
constant n_c : ℕ := 1
constant t_c : ℝ := 2

constant P_s : ℝ := 1750
constant r_s : ℝ := 0.08
constant SI : ℝ := 420

theorem find_simple_interest_years (t : ℝ) : 
  840 / 2 = SI → SI = P_s * r_s * t → t = 3 :=
by 
  sorry

end find_simple_interest_years_l228_228598


namespace probability_of_color_change_is_1_over_6_l228_228312

noncomputable def watchColorChangeProbability : ℚ :=
  let cycleDuration := 45 + 5 + 40
  let favorableDuration := 5 + 5 + 5
  favorableDuration / cycleDuration

theorem probability_of_color_change_is_1_over_6 :
  watchColorChangeProbability = 1 / 6 :=
by
  sorry

end probability_of_color_change_is_1_over_6_l228_228312


namespace proof_of_equivalence_l228_228012

noncomputable def expression : ℚ :=
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (4 * (Real.sqrt (3 + Real.sqrt 5)))

noncomputable def correct_answer : ℚ :=
  (297 - 99 * (Real.sqrt 5) + 108 * (Real.sqrt 6) - 36 * (Real.sqrt 30)) / 64

theorem proof_of_equivalence : expression = correct_answer := by
  sorry

end proof_of_equivalence_l228_228012


namespace minimization_problem_l228_228521

open Real

theorem minimization_problem 
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  (ae := a * e)^2 + (bf := b * f)^2 + (cg := c * g)^2 + (dh := d * h)^2 + (ab := a * b)^2 + (cd := c * d)^2 + (ef := e * f)^2 + (gh := g * h)^2 ≥ 64 :=
  sorry

end minimization_problem_l228_228521


namespace daniel_earnings_l228_228339

def fabric_monday := 20
def yarn_monday := 15

def fabric_tuesday := 2 * fabric_monday
def yarn_tuesday := yarn_monday + 10

def fabric_wednesday := fabric_tuesday / 4
def yarn_wednesday := yarn_tuesday / 2

def price_per_yard_fabric := 2
def price_per_yard_yarn := 3

def total_fabric := fabric_monday + fabric_tuesday + fabric_wednesday
def total_yarn := yarn_monday + yarn_tuesday + yarn_wednesday

def earnings_fabric := total_fabric * price_per_yard_fabric
def earnings_yarn := total_yarn * price_per_yard_yarn

def total_earnings := earnings_fabric + earnings_yarn

theorem daniel_earnings :
  total_earnings = 299 := by
  sorry

end daniel_earnings_l228_228339


namespace denominator_is_zero_at_eight_l228_228382

theorem denominator_is_zero_at_eight :
  (∀ x : ℝ, x^2 - 16 * x + 64 = 0 → x = 8) :=
by {
  intro x h,
  -- Using the quadratic formula or factoring, we can prove that
  -- if (x^2 - 16 * x + 64 = 0) then x must be 8.
  sorry
}

end denominator_is_zero_at_eight_l228_228382


namespace mark_paid_amount_l228_228531

/-!
# Mark's Payment Problem

Mark hires a singer for 3 hours at $15 an hour. He then tips the singer 20%.
How much did he pay?

## Conditions
- Mark hires a singer for 3 hours.
- The rate is $15 per hour.
- He tips the singer 20%.

## Conclusion
Prove that the total amount paid is $54.
-/

theorem mark_paid_amount :
  ∀ (h : ℕ) (r : ℕ) (t : ℚ), 
  h = 3 → 
  r = 15 → 
  t = 0.20 → 
  (h * r : ℚ) * (1 + t) = 54 :=
by
  intros h r t h_eq r_eq t_eq
  rw [h_eq, r_eq, t_eq]
  norm_num
  sorry

end mark_paid_amount_l228_228531


namespace range_of_x_l228_228228

noncomputable def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem range_of_x (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : abs (2 * a - b) + abs (a + b) ≥ abs a * (abs (x - 1) + abs (x + 1))) :
  x ∈ Set.Icc (-3 / 2) (3 / 2) := by
  sorry

end range_of_x_l228_228228


namespace university_diploma_percentage_l228_228922

-- Define variables
variables (P U J : ℝ)  -- P: Percentage of total population (i.e., 1 or 100%), U: Having a university diploma, J: having the job of their choice
variables (h1 : 10 / 100 * P = 10 / 100 * P * (1 - U) * J)        -- 10% of the people do not have a university diploma but have the job of their choice
variables (h2 : 30 / 100 * (P * (1 - J)) = 30 / 100 * P * U * (1 - J))  -- 30% of the people who do not have the job of their choice have a university diploma
variables (h3 : 40 / 100 * P = 40 / 100 * P * J)                   -- 40% of the people have the job of their choice

-- Statement to prove
theorem university_diploma_percentage : 
  48 / 100 * P = (30 / 100 * P * J) + (18 / 100 * P * (1 - J)) :=
by sorry

end university_diploma_percentage_l228_228922


namespace find_number_l228_228022

theorem find_number
  (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 328 - (100 * a + 10 * b + c) = a + b + c) :
  100 * a + 10 * b + c = 317 :=
sorry

end find_number_l228_228022


namespace g_2023_l228_228582

noncomputable def f : ℕ+ → ℕ+
| n => sorry

theorem g_2023 : f(f(n)) = 3n ∧ f(3n + 2) = 3n + 4 → f ⟨2023, _⟩ = ⟨2268, sorry⟩ := 
by
  sorry

end g_2023_l228_228582


namespace triangles_common_incircle_circumcircle_l228_228737

variable {Point : Type} [MetricSpace Point] (P : Point)
variables (A B C A' B' C' : Point)
variable side_distance : Point → Point → Real

def f : Point → Real := λ P, side_distance P A + side_distance P B + side_distance P C
def g : Point → Real := λ P, side_distance P A' + side_distance P B' + side_distance P C'

theorem triangles_common_incircle_circumcircle (common_incircle common_circumcircle : Prop) (P_inside : Prop) :
  common_incircle ∧ common_circumcircle ∧ P_inside → f P = g P := by
    intro h
    sorry

end triangles_common_incircle_circumcircle_l228_228737


namespace number_of_monomials_is_3_l228_228931

def isMonomial (term : String) : Bool :=
  match term with
  | "0" => true
  | "-a" => true
  | "-3x^2y" => true
  | _ => false

def monomialCount (terms : List String) : Nat :=
  terms.filter isMonomial |>.length

theorem number_of_monomials_is_3 :
  monomialCount ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"] = 3 :=
by
  sorry

end number_of_monomials_is_3_l228_228931


namespace original_price_of_sarees_l228_228238

theorem original_price_of_sarees 
  (P : ℝ) 
  (h1 : 0.72 * P = 144) : 
  P = 200 := 
sorry

end original_price_of_sarees_l228_228238


namespace star_24_75_l228_228400

noncomputable def star (a b : ℝ) : ℝ := sorry 

-- Conditions
axiom star_one_one : star 1 1 = 2
axiom star_ab_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : star (a * b) b = a * (star b b)
axiom star_a_one (a : ℝ) (h : 0 < a) : star a 1 = 2 * a

-- Theorem to prove
theorem star_24_75 : star 24 75 = 1800 := 
by 
  sorry

end star_24_75_l228_228400


namespace number_of_trailing_zeroes_base8_l228_228860

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228860


namespace simplify_expression_l228_228783

noncomputable def simplified_result (a b : ℝ) (i : ℂ) (hi : i * i = -1) : ℂ :=
  (a + b * i) * (a - b * i)

theorem simplify_expression (a b : ℝ) (i : ℂ) (hi : i * i = -1) :
  simplified_result a b i hi = a^2 + b^2 := by
  sorry

end simplify_expression_l228_228783


namespace overall_gain_is_correct_l228_228297

noncomputable def overall_gain_percentage : ℝ :=
  let CP_A := 100
  let SP_A := 120 / (1 - 0.20)
  let gain_A := SP_A - CP_A

  let CP_B := 200
  let SP_B := 240 / (1 + 0.10)
  let gain_B := SP_B - CP_B

  let CP_C := 150
  let SP_C := (165 / (1 + 0.05)) / (1 - 0.10)
  let gain_C := SP_C - CP_C

  let CP_D := 300
  let SP_D := (345 / (1 - 0.05)) / (1 + 0.15)
  let gain_D := SP_D - CP_D

  let total_gain := gain_A + gain_B + gain_C + gain_D
  let total_CP := CP_A + CP_B + CP_C + CP_D
  (total_gain / total_CP) * 100

theorem overall_gain_is_correct : abs (overall_gain_percentage - 14.48) < 0.01 := by
  sorry

end overall_gain_is_correct_l228_228297


namespace dima_lives_on_seventh_floor_l228_228006

noncomputable def elevator_speed := 1 / 60 -- floors per second
noncomputable def dima_walk_speed := elevator_speed / 2 -- floors per second

theorem dima_lives_on_seventh_floor :
  ∃ F: ℕ, F = 7 
  ∧ (let n := 9 in let t_up := 70 in let f_d := 60 in
  let floor_reachable := F - (dima_walk_speed * ((t_up - f_d) / 2)) /  (t_up * elevator_speed)) :=
sorry

end dima_lives_on_seventh_floor_l228_228006


namespace arithmetic_sequence_property_l228_228049

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence
variable {d : ℝ} -- Define the common difference
variable {a1 : ℝ} -- Define the first term

-- Suppose the sum of the first 17 terms equals 306
axiom h1 : S 17 = 306
-- Suppose the sum of the first n terms of an arithmetic sequence formula
axiom sum_formula : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
-- Suppose the relation between the first term, common difference and sum of the first 17 terms
axiom relation : a1 + 8 * d = 18 

theorem arithmetic_sequence_property : a 7 - (a 3) / 3 = 12 := 
by sorry

end arithmetic_sequence_property_l228_228049


namespace symmetrical_polynomial_l228_228584

noncomputable def Q (x : ℝ) (f g h i j k : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x + k

theorem symmetrical_polynomial (f g h i j k : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ Q 0 f g h i j k = 0 ∧
    Q x f g h i j k = x * (x - a) * (x + a) * (x - b) * (x + b) * (x - c) ∧
    Q x f g h i j k = Q (-x) f g h i j k) →
  f = 0 :=
by sorry

end symmetrical_polynomial_l228_228584


namespace num_2_edge_paths_l228_228588

-- Let T be a tetrahedron with vertices connected such that each vertex has exactly 3 edges.
-- Prove that the number of distinct 2-edge paths from a starting vertex P to an ending vertex Q is 3.

def tetrahedron : Type := ℕ -- This is a simplified representation of vertices

noncomputable def edges (a b : tetrahedron) : Prop := true -- Each pair of distinct vertices is an edge in a tetrahedron

theorem num_2_edge_paths (P Q : tetrahedron) (hP : P ≠ Q) : 
  -- There are 3 distinct 2-edge paths from P to Q  
  ∃ (paths : Finset (tetrahedron × tetrahedron)), 
    paths.card = 3 ∧ 
    ∀ (p : tetrahedron × tetrahedron), p ∈ paths → 
      edges P p.1 ∧ edges p.1 p.2 ∧ p.2 = Q :=
by 
  sorry

end num_2_edge_paths_l228_228588


namespace b_n_is_nat_l228_228936

def sequence_a : ℕ → ℝ
| 0 => 1
| n+1 => (1 / 2) * (sequence_a n) + (1 / (4 * (sequence_a n)))

def sequence_b (n : ℕ) : ℝ :=
  real.sqrt (2 / (2 * (sequence_a n)^2 - 1))

theorem b_n_is_nat (n : ℕ) (h : n > 0) : ∃ k : ℕ, sequence_b (n+1) = k :=
sorry

end b_n_is_nat_l228_228936


namespace regular_octagon_area_l228_228688

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228688


namespace find_height_of_parallelogram_l228_228365

-- Definitions based on conditions
def Area : ℝ := 704
def Base : ℝ := 32
def Height (A B : ℝ) : ℝ := A / B

-- Theorem statement
theorem find_height_of_parallelogram : Height Area Base = 22 := 
by
  sorry

end find_height_of_parallelogram_l228_228365


namespace factorial_ends_with_base_8_zeroes_l228_228873

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228873


namespace total_pages_read_l228_228537

theorem total_pages_read
  (hours_per_day : ℕ := 24)
  (fraction_of_day_read : ℝ := 1 / 6)
  (novel_rate : ℕ := 21)
  (graphic_novel_rate : ℕ := 30)
  (comic_book_rate : ℕ := 45)
  (fraction_of_reading_time_per_type : ℝ := 1 / 3) :
  let total_reading_time := (hours_per_day : ℝ) * fraction_of_day_read;
  let reading_time_per_type := total_reading_time * fraction_of_reading_time_per_type;
  let pages_novel := novel_rate * reading_time_per_type;
  let pages_graphic_novel := graphic_novel_rate * reading_time_per_type;
  let pages_comic_book := comic_book_rate * reading_time_per_type;
  pages_novel + pages_graphic_novel + pages_comic_book = 128 :=
by
  let total_reading_time := 24 * (1 / 6)
  let reading_time_per_type := total_reading_time * (1 / 3)
  let pages_novel := 21 * reading_time_per_type
  let pages_graphic_novel := 30 * reading_time_per_type
  let pages_comic_book := 45 * reading_time_per_type
  have h1 : total_reading_time = 4 := by norm_num
  have h2 : reading_time_per_type = 4 / 3 := by norm_num
  have h3 : pages_novel = 28 := by norm_num
  have h4 : pages_graphic_novel = 40 := by norm_num
  have h5 : pages_comic_book = 60 := by norm_num
  calc
    pages_novel + pages_graphic_novel + pages_comic_book
        = 28 + 40 + 60 : by rw [h3, h4, h5]
    ... = 128 : by norm_num

end total_pages_read_l228_228537


namespace factorial_base8_trailing_zeros_l228_228891

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228891


namespace cosine_value_B_l228_228127

-- Define the conditions of the problem
variables (a b c : ℝ)
variable (triangle_ABC : Type)
variable (B : triangle_ABC)

-- Assume a, b, and c form both a geometric and an arithmetic sequence
axiom geometric_sequence : a * c = b ^ 2
axiom arithmetic_sequence : a + c = 2 * b

-- Define the cosine value we want to prove
def cosine_B := (a^2 + c^2 - b^2) / (2 * a * c)

-- The theorem stating the desired relationship
theorem cosine_value_B : cosine_B a b c = 1 / 2 :=
by
  sorry

end cosine_value_B_l228_228127


namespace rectangle_y_value_l228_228303

theorem rectangle_y_value :
  ∃ (y : ℝ), 
    y > 2 ∧
    let width := 6 - (-2) in
    let height := y - 2 in
    width * height = 80 ∧
    y = 12 :=
begin
  sorry
end

end rectangle_y_value_l228_228303


namespace reciprocal_geometric_sum_l228_228658

/-- The sum of the new geometric progression formed by taking the reciprocal of each term in the original progression,
    where the original progression has 10 terms, the first term is 2, and the common ratio is 3, is \( \frac{29524}{59049} \). -/
theorem reciprocal_geometric_sum :
  let a := 2
  let r := 3
  let n := 10
  let sn := (2 * (1 - r^n)) / (1 - r)
  let sn_reciprocal := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  (sn_reciprocal = 29524 / 59049) :=
by
  sorry

end reciprocal_geometric_sum_l228_228658


namespace num_trailing_zeroes_500_factorial_l228_228743

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end num_trailing_zeroes_500_factorial_l228_228743


namespace solve_for_x_l228_228059

theorem solve_for_x (x : ℝ) : 3^x + 3^x + 3^x + 3^x = 6561 → x = 6 := by
  sorry

end solve_for_x_l228_228059


namespace meat_per_slice_is_22_l228_228980

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l228_228980


namespace trailing_zeroes_in_500_factorial_l228_228745

theorem trailing_zeroes_in_500_factorial : ∀ n = 500, (∑ k in range (nat.log 5 500 + 1), 500 / 5^k) = 124 :=
by
  sorry

end trailing_zeroes_in_500_factorial_l228_228745


namespace train_speed_l228_228617

theorem train_speed (train_length time : ℝ) (same_speed in_opposite_directions : Prop)
  (h1 : train_length = 120) (h2 : time = 36) (h3 : same_speed) (h4 : in_opposite_directions) :
  let v_km_per_hr := 12.01
  ∃ v : ℝ, v = 3.335 ∧ (v * 3.6) = v_km_per_hr :=
by
  exists 3.335
  constructor
  · refl
  · norm_num
  sorry

end train_speed_l228_228617


namespace area_of_ring_between_outermost_and_middle_circle_l228_228607

noncomputable def pi : ℝ := Real.pi

theorem area_of_ring_between_outermost_and_middle_circle :
  let r_outermost := 12
  let r_middle := 8
  let A_outermost := pi * r_outermost^2
  let A_middle := pi * r_middle^2
  A_outermost - A_middle = 80 * pi :=
by 
  sorry

end area_of_ring_between_outermost_and_middle_circle_l228_228607


namespace hannahs_grapes_per_day_l228_228251

-- Definitions based on conditions
def oranges_per_day : ℕ := 20
def days : ℕ := 30
def total_fruits : ℕ := 1800
def total_oranges : ℕ := oranges_per_day * days

-- The math proof problem to be targeted
theorem hannahs_grapes_per_day : 
  (total_fruits - total_oranges) / days = 40 := 
by
  -- Proof to be filled in here
  sorry

end hannahs_grapes_per_day_l228_228251


namespace simplify_sqrt_expression1_simplify_sqrt_expression2_l228_228331

-- Declare the problems as Lean theorem statements

theorem simplify_sqrt_expression1 : sqrt 2 + 3 * sqrt 2 - 5 * sqrt 2 = -sqrt 2 :=
by sorry

theorem simplify_sqrt_expression2 : sqrt 27 + abs (1 - sqrt 3) + (Real.pi - 3)^0 = 4 * sqrt 3 :=
by sorry

end simplify_sqrt_expression1_simplify_sqrt_expression2_l228_228331


namespace find_m_l228_228403

theorem find_m (x m : ℝ) (hx1 : log 10 (sin x) + log 10 (cos x) = -2) 
  (hx2 : log 10 (sin x - cos x) = (1 / 2) * (log 10 m - 1)) : m = 9.8 := 
by
  sorry

end find_m_l228_228403


namespace largest_hexagon_angle_l228_228569

theorem largest_hexagon_angle (x : ℝ) : 
  (2 * x + 2 * x + 2 * x + 3 * x + 4 * x + 5 * x = 720) → (5 * x = 200) := by
  sorry

end largest_hexagon_angle_l228_228569


namespace no_such_polynomial_exists_l228_228762

noncomputable def is_ones_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

theorem no_such_polynomial_exists :
  ¬ ∃ P : ℤ[x], (∀ n : ℕ, is_ones_number n → is_ones_number (P.eval n)) ∧ P.degree = 2 :=
by
  sorry

end no_such_polynomial_exists_l228_228762


namespace octagon_area_l228_228701

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228701


namespace inequality_solution_l228_228769

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 19 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_solution_l228_228769


namespace factorial_base8_trailing_zeros_l228_228889

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228889


namespace tan_2alpha_of_sin_cos_ratio_l228_228107

theorem tan_2alpha_of_sin_cos_ratio (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1/2) : 
  tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_2alpha_of_sin_cos_ratio_l228_228107


namespace incircle_center_on_line_area_of_triangle_l228_228295

variable (P : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ)

def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 36) + (y^2 / 4) = 1

def line_eq (l : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, l x = (1 / 3) * x + m  

def point_left_of_line (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, P.2 > l P.1 

def points_intersect_ellipse (A B : ℝ × ℝ) (C_eq : ℝ → ℝ → Prop) (l_eq : ℝ → (ℝ → ℝ) → Prop) : Prop :=
  ∃ x y : ℝ , C_eq x y ∧ ∃ l : ℝ → ℝ, l_eq (λ x, y) ∧ ∀ x_l y_l : ℝ , (x_l,y_l) ∈ {A, B}

theorem incircle_center_on_line 
  (hP : point_left_of_line P l)
  (hEllipse : ∀ x y : ℝ, ellipse_eq x y)
  (hAB_on_ellipse : points_intersect_ellipse A B ellipse_eq (λ l_eq, line_eq l_eq))
  (x_incenter : ∀ C : ℝ × ℝ, C = (3 * real.sqrt 2, real.sqrt 2)) :
  x_incenter (incircle_center P A B) :=
sorry

theorem area_of_triangle 
  (hP : point_left_of_line P l)
  (hEllipse : ∀ x y : ℝ, ellipse_eq x y)
  (hAB_on_ellipse : points_intersect_ellipse A B ellipse_eq (λ l_eq, line_eq l_eq))
  (angle_APB_eq : ∀ A B: ℝ , ∃ (angle_APB: ℝ), angle_APB = 60)
  (area_PAB : ∀ \(\angle APB\) : ℝ, ∃ area : ℝ, area = (117 * real.sqrt 3) / 49) :
  area_PAB (angle_APB_eq 60) =
sorry

end incircle_center_on_line_area_of_triangle_l228_228295


namespace ellipse_equation_range_of_m_square_l228_228808

section ellipse_problem

variables {k m : ℝ}

-- Definitions based on conditions
def eccentricity : ℝ := sqrt 3 / 2
def perimeter : ℝ := 4 * sqrt 5

-- The equation of the ellipse given the conditions
theorem ellipse_equation : eccentricity = sqrt 3 / 2 ∧ perimeter = 4 * sqrt 5 → 
  (∃ a b : ℝ, a > b > 0 ∧ (c = sqrt 3) ∧ (a = 2) ∧ (b = 1) ∧ ellipse_eq : ∀ (x y : ℝ), y^2/4 + x^2 = 1) := 
sorry

-- The range of m^2 given the vector condition
theorem range_of_m_square (h : ∥(λ (x : ℝ), 3 * x)∥ = 3) : 
  (1 < m^2) ∧ (m^2 ≤ 4) :=
sorry

end ellipse_problem

end ellipse_equation_range_of_m_square_l228_228808


namespace qy_length_l228_228178

theorem qy_length (Q : Type*) (C : Type*) (X Y Z : Q) (QX QZ QY : ℝ) 
  (h1 : 5 = QX)
  (h2 : QZ = 2 * (QY - QX))
  (PQ_theorem : QX * QY = QZ^2) :
  QY = 10 :=
by
  sorry

end qy_length_l228_228178


namespace coefficient_x2_l228_228342

noncomputable def binomial_coeff : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binomial_coeff (n - 1) (k - 1) + binomial_coeff (n - 1) k

theorem coefficient_x2 :
  let C := binomial_coeff in
  let expansion_coeff_x3 := C 5 2 in
  let expansion_coeff_x2 := C 5 3 in
  (expansion_coeff_x3 - 2 * expansion_coeff_x2 = -10) :=
begin
  let C := binomial_coeff,
  let expansion_coeff_x3 := C 5 2,
  let expansion_coeff_x2 := C 5 3,
  show expansion_coeff_x3 - 2 * expansion_coeff_x2 = -10,
  sorry
end

end coefficient_x2_l228_228342


namespace half_percent_of_160_l228_228619

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l228_228619


namespace four_digit_numbers_meeting_conditions_l228_228849

theorem four_digit_numbers_meeting_conditions
  (d1 d2 d3 d4 : ℕ) :
  (1 ≤ d1 ∧ d1 ≤ 9) ∧ d1 % 2 = 1 ∧         -- d1 is an odd non-zero digit.
  (0 ≤ d2 ∧ d2 ≤ 9) ∧ d2 % 2 = 0 ∧         -- d2 is an even digit.
  (d3 = 10 - d2) ∧                         -- d3 + d2 = 10
  (0 ≤ d3 ∧ d3 ≤ 9) ∧                      -- d3 is a valid digit.
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) ->    -- all digits different.
  560 
:= sorry

end four_digit_numbers_meeting_conditions_l228_228849


namespace midpoint_trajectory_l228_228192

noncomputable def hyperbola := {x: ℝ × ℝ // x.snd^2 - x.fst^2 / 3 = 1}
noncomputable def foci_distance : ℝ := 4
def eccentricity : ℝ := 2
def asymptote_1 (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x
def asymptote_2 (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x

theorem midpoint_trajectory (A B : hyperbola ∩ (λ p, p.snd = asymptote_1 p.fst) ∩ (λ p, p.snd = asymptote_2 p.fst)) : 
  (2 * Real.sqrt ((A.val.fst - B.val.fst)^2 + (A.val.snd - B.val.snd)^2)) = 20 →
  (∃ M : ℝ × ℝ, 2 * M.fst = A.val.fst + B.val.fst ∧ 2 * M.snd = A.val.snd + B.val.snd ∧ 
    (M.fst^2 / 75 + 3 * M.snd^2 / 25 = 1)) :=
sorry

end midpoint_trajectory_l228_228192


namespace sum_first_13_terms_l228_228929

-- Define the arithmetic sequence and conditions
axiom arithmetic_sequence (a : ℕ → ℤ) : Prop

axiom condition (a : ℕ → ℤ) : 
  arithmetic_sequence a →
  3 * a 7 = 12 

-- Prove that the sum of the first 13 terms is 52
theorem sum_first_13_terms (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) (h_cond : condition a h_arith) : 
  (finset.range 13).sum a = 52 :=
sorry

end sum_first_13_terms_l228_228929


namespace find_angle_BHC_l228_228492

variable (A B C D E F H : Type)

variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables (triangle_ABC : triangle ℝ A B C)
variables (altitude_AD altitude_BE altitude_CF : line ℝ A → line ℝ B → line ℝ C)

-- Conditions
axiom altitudes_intersect : (intersection altitudes_AD altitude_BE altitude_CF = H)
axiom angle_ABC : ∠ B A C = 49
axiom angle_ACB : ∠ A C B = 12

-- Theorem to be proved
theorem find_angle_BHC : angle H B C = 61 := sorry

end find_angle_BHC_l228_228492


namespace find_OC_l228_228136

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

def OA (A : Point) : ℝ := sqrt (A.x^2 + A.y^2)
def OB (B : Point) : ℝ := sqrt (B.x^2 + B.y^2)
def OD (D : Point) : ℝ := sqrt (D.x^2 + D.y^2)
def ratio_of_lengths (A B : Point) : ℝ := OA A / OB B

def find_D (A B : Point) : Point :=
  let ratio := ratio_of_lengths A B
  { x := (A.x + ratio * B.x) / (1 + ratio),
    y := (A.y + ratio * B.y) / (1 + ratio) }

-- Given conditions
def A : Point := ⟨0, 1⟩
def B : Point := ⟨-3, 4⟩
def C_magnitude : ℝ := 2

-- Goal to prove
theorem find_OC : Point :=
  let D := find_D A B
  let D_length := OD D
  let scale := C_magnitude / D_length
  { x := D.x * scale,
    y := D.y * scale }

example : find_OC = ⟨-sqrt 10 / 5, 3 * sqrt 10 / 5⟩ := by
  sorry

end find_OC_l228_228136


namespace range_of_a_l228_228469

theorem range_of_a : 
  (∃ a : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (x^2 + a ≤ a*x - 3))) ↔ (a ≥ 7) :=
sorry

end range_of_a_l228_228469


namespace unique_pair_exists_l228_228996

theorem unique_pair_exists (n : ℕ) (hn : n > 0) : 
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ 0 ≤ l ∧ l < k :=
sorry

end unique_pair_exists_l228_228996


namespace factorial_ends_with_base_8_zeroes_l228_228876

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228876


namespace expectation_inequality_l228_228525

noncomputable theory
open MeasureTheory

variables {ξ : ℝ} (f : ℝ → ℝ) [IsProbabilityMeasure (Measure.dirac ξ)]

-- Conditions
def smooth_density (f : ℝ → ℝ) : Prop := differentiable ℝ f

def density_limit_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, tendsto (λ x, x * f x) at_top (nhds 0) ∧ tendsto (λ x, x * f x) at_bot (nhds 0)

-- The proof problem
theorem expectation_inequality (h1 : smooth_density f)
  (h2 : density_limit_condition f) :
  let E (g : ℝ → ℝ) := ∫ x, g x * f x ∂(Measure.dirac ξ) in
  (E (λ x, x^2)) * (E (λ x, (f' x / f x)^2)) ≥ 1 :=
sorry

end expectation_inequality_l228_228525


namespace smallest_n_for_factorization_l228_228785

theorem smallest_n_for_factorization :
  ∃ n : ℤ, (∀ A B : ℤ, A * B = 60 ↔ n = 5 * B + A) ∧ n = 56 :=
by
  sorry

end smallest_n_for_factorization_l228_228785


namespace smallest_prime_with_prime_digit_sum_l228_228026

def is_prime (n : ℕ) : Prop := ¬ ∃ m, m ∣ n ∧ 1 < m ∧ m < n

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_prime_digit_sum :
  ∃ p, is_prime p ∧ is_prime (digit_sum p) ∧ 10 < digit_sum p ∧ p = 29 :=
by
  sorry

end smallest_prime_with_prime_digit_sum_l228_228026


namespace find_a_l228_228804

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_a (h1 : a ≠ 0) (h2 : f a b c (-1) = 0)
    (h3 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x^2 + 1)) :
  a = 1/2 :=
by
  sorry

end find_a_l228_228804


namespace incorrect_derivative_l228_228629

theorem incorrect_derivative :
  ∀ (x : ℝ), (∀ x > 0, (sqrt x)' = 1 / (2 * sqrt x)) →
  (∀ x ≠ 0, (1 / x)' = -1 / (x * x)) →
  (∀ x > 0, (log x)' = 1 / x) →
  ¬((∀ x, (exp (-x))' = exp (-x))) :=
by
  intros x H1 H2 H3
  apply not_forall.2
  use 1
  have : (exp (-1))' = -exp (-1) := by sorry
  exact this

end incorrect_derivative_l228_228629


namespace total_worth_of_presents_l228_228165

theorem total_worth_of_presents :
  let r := 4000
  let c := 2000
  let b := 2 * r
  let g := 0.5 * b
  let j := 1.2 * r
  r + c + b + g + j = 22800 := 
by
  let r := 4000
  let c := 2000
  let b := 2 * r
  let g := 0.5 * b
  let j := 1.2 * r
  calc
    r + c + b + g + j = 4000 + 2000 + 2 * 4000 + 0.5 * (2 * 4000) + 1.2 * 4000 := by rfl
    _ = 4000 + 2000 + 8000 + 4000 + 4800 := by rfl
    _ = 22800 := by rfl

end total_worth_of_presents_l228_228165


namespace order_of_numbers_l228_228344

theorem order_of_numbers (a b : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) :
  (a^(1/3) > a^(2/3)) ∧ (a^(2/3) > b^(2/3)) ↔
  (a^(1/3) > a^(2/3) > b^(2/3)) :=
by { sorry }

end order_of_numbers_l228_228344


namespace exponent_rule_example_l228_228749

theorem exponent_rule_example {a : ℝ} : (a^3)^4 = a^12 :=
by {
  sorry
}

end exponent_rule_example_l228_228749


namespace area_ratio_l228_228956

variable {A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 E1 E2 E3 : Type}

def midpoint (P Q : Type) : Type := sorry  -- Placeholder for midpoint definition
def homothety (P Q : Type) : Type := sorry  -- Placeholder for homothety definition
constant coordinates : Type → (ℝ × ℝ × ℝ)  -- Barycentric coordinates mapping

-- Given conditions
axiom A1_coords : coordinates A1 = (1, 0, 0)
axiom A2_coords : coordinates A2 = (0, 1, 0)
axiom A3_coords : coordinates A3 = (0, 0, 1)
axiom B1_midpoint : midpoint A1 A2 = B1
axiom B2_midpoint : midpoint A2 A3 = B2
axiom B3_midpoint : midpoint A3 A1 = B3
axiom C1_midpoint : midpoint A1 B1 = C1
axiom C2_midpoint : midpoint A2 B2 = C2
axiom C3_midpoint : midpoint A3 B3 = C3
axiom D1_intersect : ∃ (P : Type), P = (D1 : Type) ∧ P ∉ (A1 C2) → P ∉ (B1 A3)
axiom D2_intersect : ∃ (P : Type), P = (D2 : Type) ∧ P ∉ (A2 C3) → P ∉ (B2 A1)
axiom D3_intersect : ∃ (P : Type), P = (D3 : Type) ∧ P ∉ (A3 C1) → P ∉ (B3 A2)
axiom E1_intersect : ∃ (P : Type), P = (E1 : Type) ∧ P ∉ (A1 B2) → P ∉ (C1 A3)
axiom E2_intersect : ∃ (P : Type), P = (E2 : Type) ∧ P ∉ (A2 B3) → P ∉ (C2 A1)
axiom E3_intersect : ∃ (P : Type), P = (E3 : Type) ∧ P ∉ (A3 B1) → P ∉ (C3 A2)

-- The proof problem: ratio of areas
theorem area_ratio :
  let area (triangle : Type) : ℝ := sorry in  -- Placeholder for area calculation
  area D1 D2 D3 / area E1 E2 E3 = 25 / 49 :=
sorry

end area_ratio_l228_228956


namespace residual_at_sample_point_l228_228805

theorem residual_at_sample_point (x y : ℝ) (h : (x, y) = (165, 57)) :
  let y_hat := 0.85 * x - 85.7 in
  let residual := y - y_hat in
  residual = 2.45 :=
by
  unfold y_hat residual
  subst h
  dsimp
  norm_num
  sorry

end residual_at_sample_point_l228_228805


namespace gcd_18_30_45_l228_228361

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l228_228361


namespace range_f_l228_228593

def f (x : ℝ) : ℝ := 1 / (x^2 + 3)

theorem range_f : Set.Ioc 0 (1 / 3) = set_of (λ y, ∃ x : ℝ, f x = y) :=
sorry

end range_f_l228_228593


namespace factorial_base8_trailing_zeros_l228_228887

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228887


namespace even_perfect_number_has_special_form_l228_228997

noncomputable def perfect_number (n : ℕ) : Prop := sorry

noncomputable def mersenne_prime (p : ℕ) : Prop := 
  ∃ k : ℕ, k ≥ 2 ∧ p = 2^k - 1 ∧ Prime p

theorem even_perfect_number_has_special_form (n : ℕ) (k : ℕ) (h2 : k ≥ 2)
    (h1 : perfect_number n) :
  ∃ p, (p = 2^k - 1) ∧ mersenne_prime p ∧ n = 2^(k-1) * p :=
begin
  sorry
end

end even_perfect_number_has_special_form_l228_228997


namespace equation_solution_l228_228211

theorem equation_solution (x y z : ℕ) (h1 : x = 1) (h2 : y = 2) (h3 : z = 2) :
  (z + 1) ^ x - z ^ y = -1 :=
by
  sorry

end equation_solution_l228_228211


namespace max_value_k_l228_228775

theorem max_value_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
(h4 : 4 = k^2 * (x^2 / y^2 + 2 + y^2 / x^2) + k^3 * (x / y + y / x)) : 
k ≤ 4 * (Real.sqrt 2) - 4 :=
by sorry

end max_value_k_l228_228775


namespace select_three_divisible_by_three_l228_228553

open Finset

theorem select_three_divisible_by_three (S : Finset ℤ) (hS : S.card = 5) :
  ∃ (a b c : ℤ), {a, b, c} ⊆ S ∧ (a + b + c) % 3 = 0 :=
by
  sorry

end select_three_divisible_by_three_l228_228553


namespace diff_squares_div_l228_228625

theorem diff_squares_div (a b : ℕ) (h1 : a = 121) (h2 : b = 112) :
  (a^2 - b^2) / 9 = 233 :=
by
  rw [h1, h2]
  calc
    (121^2 - 112^2) / 9 = 2097 / 9 : by rw [nat.pow_succ, nat.pow_succ]; norm_num
                        ...          = 233 : by norm_num

end diff_squares_div_l228_228625


namespace ellipse_area_l228_228480

theorem ellipse_area
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (a : { endpoints_major_axis : (ℝ × ℝ) × (ℝ × ℝ) // endpoints_major_axis = ((x1, y1), (x2, y2)) })
  (b : { point_on_ellipse : ℝ × ℝ // point_on_ellipse = (x3, y3) }) :
  (-5 : ℝ) = x1 ∧ (2 : ℝ) = y1 ∧ (15 : ℝ) = x2 ∧ (2 : ℝ) = y2 ∧
  (8 : ℝ) = x3 ∧ (6 : ℝ) = y3 → 
  100 * Real.pi * Real.sqrt (16 / 91) = 100 * Real.pi * Real.sqrt (16 / 91) :=
by
  sorry

end ellipse_area_l228_228480


namespace bacteria_difference_l228_228300

theorem bacteria_difference (initial_bacteria current_bacteria : ℕ) (h_initial : initial_bacteria = 600) (h_current : current_bacteria = 8917) : 
  current_bacteria - initial_bacteria = 8317 :=
by
  rw [h_initial, h_current]
  norm_num
  sorry

end bacteria_difference_l228_228300


namespace find_y_eq_sqrt_469_l228_228014

theorem find_y_eq_sqrt_469
    (a b c d : ℝ)
    (ao co : ℝ)
    (bo do : ℝ)
    (bd : ℝ)
    (phi : ℝ)
    (h_cos_phi : cos phi = -17 / 42)
    (h_ao : ao = 6)
    (h_co : co = 5)
    (h_bo : bo = 7)
    (h_do : do = 3)
    (h_bd : bd = 9)
    : (6^2 + 5^2 - 2 * 6 * 5 * cos phi = 469) :=
by
  sorry

end find_y_eq_sqrt_469_l228_228014


namespace expected_value_of_n_eq_e_l228_228719

open scoped Real BigOperators

-- Definition of the sequence α_i
variable {α : ℕ → ℝ} (hα : ∀ i, 0 ≤ α i ∧ α i ≤ 1)

-- Definition of the expected value of n
noncomputable def expected_value_n : ℝ :=
  Real.toNnreal (∑' n, n * (Real.exp (-n)))

-- Theorem statement
theorem expected_value_of_n_eq_e
  (h1 : ∀ n, ∑ i in Finset.range n, α i ≤ 1 → (∑ i in Finset.range (n + 1), α i > 1))
  : expected_value_n = Real.exp 1 :=
sorry

end expected_value_of_n_eq_e_l228_228719


namespace min_value_frac_gcd_l228_228951

theorem min_value_frac_gcd {N k : ℕ} (hN_substring : N % 10^5 = 11235) (hN_pos : 0 < N) (hk_pos : 0 < k) (hk_bound : 10^k > N) : 
  (10^k - 1) / Nat.gcd N (10^k - 1) = 89 :=
by
  -- proof goes here
  sorry

end min_value_frac_gcd_l228_228951


namespace combined_distance_for_week_l228_228497

variables (julien_dist_per_day : ℕ) (sarah_dist_per_day : ℕ) (jamir_dist_per_day : ℕ)
variables (lily_speed : ℕ) (lily_time_per_day : ℕ)
variables (julien_speed : ℕ) (days_in_week : ℕ)

-- Julien's daily distance
def julien_distance := 50

-- Sarah swims twice the distance Julien swims
def sarah_distance := 2 * julien_distance

-- Jamir swims 20 more meters per day than Sarah
def jamir_distance := sarah_distance + 20

-- Julien's speed
def julien_speed := julien_distance / 20

-- Lily swims at a constant speed which is 4 times faster than Julien's speed
def lily_speed := 4 * julien_speed

-- Lily swims for 30 minutes daily
def lily_distance_per_day := lily_speed * 30

-- They go to the swimming pool the whole week (7 days)
def days_in_week := 7

-- Total distance for the whole week
def total_distance := 7 * (julien_distance + sarah_distance + jamir_distance + lily_distance_per_day)

theorem combined_distance_for_week : total_distance = 3990 := by
  sorry

end combined_distance_for_week_l228_228497


namespace lily_account_balance_l228_228977

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l228_228977


namespace remainder_of_polynomial_divided_by_x_minus_3_eq_6892_l228_228260

def polynomial := (2 * x ^ 8) - (3 * x ^ 7) + (x ^ 5) - (5 * x ^ 4) + (x ^ 2) - 6

theorem remainder_of_polynomial_divided_by_x_minus_3_eq_6892 :
  eval 3 polynomial = 6892 :=
by
  sorry

end remainder_of_polynomial_divided_by_x_minus_3_eq_6892_l228_228260


namespace area_of_triangle_AMC_l228_228614

theorem area_of_triangle_AMC 
  (A M C : Point)
  (h_right_triangle : right_triangle A M C)
  (h_perpendicular_AM_AC : perp AM AC)
  (h_AM : length AM = 10)
  (h_AC : length AC = 10) :
  area (triangle A M C) = 50 := 
sorry

end area_of_triangle_AMC_l228_228614


namespace sequence_has_irrational_l228_228509

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) ^ 2 = a n + 1

theorem sequence_has_irrational (a : ℕ → ℝ)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_start : true)
  (h_seq : sequence_condition a) :
  ∃ n : ℕ, ¬ is_rat (a n) := 
sorry

end sequence_has_irrational_l228_228509


namespace midpoint_of_chord_l228_228935

-- Define the polar equation of the line
def polar_line (rho theta : ℝ) : Prop := rho * (Real.cos theta - Real.sin theta) + 2 = 0

-- Define the polar equation of the circle
def polar_circle (rho : ℝ) : Prop := rho = 2

-- Define the polar coordinates of a point
def polar_coordinates (x y : ℝ) : ℝ × ℝ := (Real.sqrt (x^2 + y^2), Real.atan2 y x)

theorem midpoint_of_chord :
  ∃ (rho θ : ℝ), polar_coordinates (-1) 1 = (rho, θ) ∧ rho = Real.sqrt 2 ∧ θ = 3/4 * Real.pi :=
by
  sorry

end midpoint_of_chord_l228_228935


namespace log_sum_identity_l228_228761

theorem log_sum_identity :
  ∑ k in Finset.range 10, (Real.log (1 - 1 / (k + 6)) / Real.log 3) = -1 :=
by
  sorry

end log_sum_identity_l228_228761


namespace eq_area_shape_l228_228418

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x)

def x1 : ℝ := (5 * Real.pi / 12)
def x2 : ℝ := (13 * Real.pi / 12)

def area_under_curve (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

def area_segment_shape (x1 x2 : ℝ) (f : ℝ → ℝ) : ℝ :=
  abs (x2 - x1) * 1 - 2 * (area_under_curve x1 (Real.pi / 2) f + area_under_curve (Real.pi / 2) x2 f)

theorem eq_area_shape :
  area_segment_shape x1 x2 f = (2 * Real.pi / 3) + Real.sqrt 3 :=
by
  sorry

end eq_area_shape_l228_228418


namespace product_of_triangle_areas_l228_228927

open Real

noncomputable def on_parabola (A B : Real × Real) : Prop :=
  A.1 = (A.2 ^ 2) / 4 ∧ B.1 = (B.2 ^ 2) / 4

noncomputable def dot_product (A B : Real × Real) : Real :=
  A.1 * B.1 + A.2 * B.2

noncomputable def area (A B F : Real × Real) : Real :=
  0.5 * abs (A.1 * B.2 + B.1 * F.2 + F.1 * A.2 - A.2 * B.1 - B.2 * F.1 - F.2 * A.1)

variable {A B F : Real × Real}

theorem product_of_triangle_areas (h1 : on_parabola A B) (h2 : dot_product A B = -4)
(h3 : F = (1, 0)) :
  let S_OFA := area (0, 0) A F in
  let S_OFB := area (0, 0) B F in
  S_OFA * S_OFB = 2 :=
by
  sorry

end product_of_triangle_areas_l228_228927


namespace num_trailing_zeroes_500_factorial_l228_228744

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end num_trailing_zeroes_500_factorial_l228_228744


namespace triangle_similarity_l228_228479

variables {A B C P Q R : Point}
variables {n : ℕ}

-- Conditions: 
def n_division (A B : Point) (n: ℕ) : list Point := sorry -- Function for getting n division points
def intersects_at (A B P: Point): Prop := sorry -- Intersection predicate

-- Problem:
theorem triangle_similarity 
  (ABC : Triangle) 
  (h1 : n_division A B n) 
  (h2 : n_division B C n) 
  (h3 : n_division A C n) 
  (PQR : Triangle)
  (h4 : intersects_at A (list.get! (n_division B C n) (m - 1)) P)
  (h5 : intersects_at B (list.get! (n_division A C n) (m - 1)) Q)
  (h6 : intersects_at C (list.get! (n_division A B n) (m - 1)) R) :
  similar PQR ABC ∧ ratio PQR ABC = (n-2)/(2n-1) := sorry

end triangle_similarity_l228_228479


namespace factorial_base_8_zeroes_l228_228857

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228857


namespace next_number_in_sequence_l228_228017

theorem next_number_in_sequence (seq : list ℕ) (h1 : seq = [112, 224, 448, 8816, 6612]) :
  ∃ n : ℕ, n = 224 := 
sorry

end next_number_in_sequence_l228_228017


namespace expected_value_sum_marbles_l228_228893

theorem expected_value_sum_marbles :
  let marbles := {2, 3, 4, 5, 6, 7}
  let pairs := { (x, y) | x ∈ marbles ∧ y ∈ marbles ∧ x < y }
  let sums := { x + y | (x, y) ∈ pairs }
  let total_sum := ∑ s in sums.to_finset, s
  let pair_count := finset.card pairs.to_finset
  total_sum / pair_count = 29 / 3 := -- 9.67 is the decimal representation of 29/3
by { 
  sorry 
}

end expected_value_sum_marbles_l228_228893


namespace geometric_sequence_alpha_5_l228_228490

theorem geometric_sequence_alpha_5 (α : ℕ → ℝ) (h1 : α 4 * α 5 * α 6 = 27) (h2 : α 4 * α 6 = (α 5) ^ 2) : α 5 = 3 := 
sorry

end geometric_sequence_alpha_5_l228_228490


namespace cos_of_arithmetic_sequence_sum_l228_228398

theorem cos_of_arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m k, m = n + k → a m = a n + k * (a 1 - a 0))
  (h_sum : a 1 + a 4 + a 7 = 5 / 4 * Real.pi) : 
  Real.cos (a 3 + a 5) = -Real.sqrt 3 / 2 :=
sorry

end cos_of_arithmetic_sequence_sum_l228_228398


namespace octagon_area_l228_228697

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228697


namespace base_of_exponent_l228_228454

theorem base_of_exponent (x : ℤ) (m : ℕ) (h₁ : (-2 : ℤ)^(2 * m) = x^(12 - m)) (h₂ : m = 4) : x = -2 :=
by 
  sorry

end base_of_exponent_l228_228454


namespace intersecting_lines_exists_l228_228801

theorem intersecting_lines_exists (n : ℕ)
    (segments : fin (4 * n) → set (ℝ × ℝ))
    (h_segments_length : ∀ i, ∃ A B : ℝ × ℝ, segments i = {x : ℝ × ℝ | x = A ∨ x = B} ∧ dist A B = 1)
    (circle : set (ℝ × ℝ))
    (radius_condition : ∃ O : ℝ × ℝ, ∀ x ∈ circle, dist O x ≤ n)
    (l : set (ℝ × ℝ)) :
  ∃ l' : set (ℝ × ℝ), (l' ∥ l ∨ l' ⊥ l) ∧ ∃ i j : fin (4 * n), i ≠ j ∧ ∃ x ∈ segments i, ∃ y ∈ segments j, x = y ∈ l' ∩ circle  := 
sorry

end intersecting_lines_exists_l228_228801


namespace ladder_length_difference_l228_228164

theorem ladder_length_difference :
  ∀ (flights : ℕ) (flight_height rope ladder_total_height : ℕ),
    flights = 3 →
    flight_height = 10 →
    rope = (flights * flight_height) / 2 →
    ladder_total_height = 70 →
    ladder_total_height - (flights * flight_height + rope) = 25 →
    ladder_total_height - (flights * flight_height) - rope = 10 :=
by
  intros
  sorry

end ladder_length_difference_l228_228164


namespace simple_interest_rate_l228_228259

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (SI_eq : SI = 260)
  (P_eq : P = 910) (T_eq : T = 4)
  (H : SI = P * R * T / 100) : 
  R = 26000 / 3640 := 
by
  sorry

end simple_interest_rate_l228_228259


namespace find_possible_values_of_x_l228_228407

theorem find_possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1 / y = 12)
    (h2 : y + 1 / x = 7 / 15) :
    x = 6 + 3 * real.sqrt (8 / 7) ∨ x = 6 - 3 * real.sqrt (8 / 7) := 
sorry

end find_possible_values_of_x_l228_228407


namespace only_strictly_increasing_function_satisfying_condition_l228_228358

noncomputable def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b⦄, a < b → f(a) < f(b)

theorem only_strictly_increasing_function_satisfying_condition (f : ℕ → ℕ) :
  strictly_increasing f ∧ (∀ n, f(f(n)) < n + 1) → (∀ n, f(n) = n) :=
by
  sorry

end only_strictly_increasing_function_satisfying_condition_l228_228358


namespace factorial_trailing_zeros_base_8_l228_228867

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228867


namespace arithmetic_sequence_relationship_l228_228123

variable {α : Type*} [Add α] (a : ℕ → α) (d : α)

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_relationship {a : ℕ → α} (h : is_arithmetic_sequence a) :
  a 1 + a 8 = a 4 + a 5 :=
by
  sorry

end arithmetic_sequence_relationship_l228_228123


namespace area_of_inscribed_octagon_l228_228716

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228716


namespace scalene_triangle_proof_l228_228548

variable (Triangle : Type)
variable (l1 l2 S : ℝ) -- lengths of longest & shortest bisectors and area respectively
variable [ScaleneTriangle : Triangle]

-- Define properties of a scalene triangle and the respective angle bisectors and area
axiom longest_bisector (T : Triangle) : ℝ
axiom shortest_bisector (T : Triangle) : ℝ
axiom area (T : Triangle) : ℝ

-- Given a scalene triangle
def is_scalene_triangle (T : Triangle) : Prop := True -- placeholder for actual definition

def problem_statement (T : Triangle) : Prop :=
  is_scalene_triangle T → longest_bisector T ^ 2 > sqrt 3 * area T ∧ sqrt 3 * area T > shortest_bisector T ^ 2

-- Main statement to be proved
theorem scalene_triangle_proof (T : Triangle) (h : is_scalene_triangle T) : problem_statement T := sorry

end scalene_triangle_proof_l228_228548


namespace octagon_area_l228_228698

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228698


namespace factorial_base_8_zeroes_l228_228856

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l228_228856


namespace determine_slope_l228_228005

theorem determine_slope (a : ℝ) :
  (∀ (x : ℝ), (y : ℝ) → y = Real.exp (a * x) * Real.cos x) →
  (∀ (x : ℝ), (y' : ℝ) → y' = (deriv (λ x, Real.exp (a * x) * Real.cos x)) x) →
  (∀ (m : ℝ), m = -1/2) →
  (∀ (x : ℝ), y' = a) →
  (a * (-1/2) = -1) → a = 2 :=
by
  sorry

end determine_slope_l228_228005


namespace landlord_packages_l228_228587

def label_packages_required (start1 end1 start2 end2 start3 end3 : ℕ) : ℕ :=
  let digit_count := 1
  let hundreds_first := (end1 - start1 + 1)
  let hundreds_second := (end2 - start2 + 1)
  let hundreds_third := (end3 - start3 + 1)
  let total_hundreds := hundreds_first + hundreds_second + hundreds_third
  
  let tens_first := ((end1 - start1 + 1) / 10) 
  let tens_second := ((end2 - start2 + 1) / 10) 
  let tens_third := ((end3 - start3 + 1) / 10)
  let total_tens := tens_first + tens_second + tens_third

  let units_per_floor := 5
  let total_units := units_per_floor * 3
  
  let total_ones := total_hundreds + total_tens + total_units
  
  let packages_required := total_ones

  packages_required

theorem landlord_packages : label_packages_required 100 150 200 250 300 350 = 198 := 
  by sorry

end landlord_packages_l228_228587


namespace triangle_trig_identity_l228_228909

theorem triangle_trig_identity
  (A B C : ℝ)
  (hAB : A + B = π / 2)
  (hC : C = π / 3)
  (h : 8 = 8) -- dummy condition for AB = 8
  (h1 : 7 = 7) -- dummy condition for AC = 7
  (h2 : 5 = 5) -- dummy condition for BC = 5
  (h_triangle_ineq : 8 + 7 > 5 ∧ 7 + 5 > 8 ∧ 8 + 5 > 7) :
  (cos (A + B) / 2 / sin (C / 2) - sin (A + B) / 2 / cos (C / 2)) = 0 :=
by {
  sorry
}

end triangle_trig_identity_l228_228909


namespace gen_terms_max_k_l228_228061

variable  (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (c : ℕ → ℚ) (T : ℕ → ℚ)

axiom Sn_def : ∀ n : ℕ, S n = (1/2)*n^2 + (11/2)*n
axiom b_rec : ∀ n : ℕ, b (n+2) = 2 * b (n+1) - b n
axiom b3_val : b 3 = 11
axiom b_sum_9 : (Finset.range 9).sum b = 153

theorem gen_terms (n : ℕ) : 
  (a n = n + 5) ∧ (b n = 3n + 2) := 
sorry

theorem max_k (k : ℕ) : 
  (∀ n : ℕ+, T n > (k : ℚ) / 57) →
  k ≤ 37 := 
sorry

end gen_terms_max_k_l228_228061


namespace factorial_trailing_zeros_base_8_l228_228871

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228871


namespace count_solutions_tan_cot_l228_228781

theorem count_solutions_tan_cot (θ : ℝ) (hθ : θ ∈ set.Ioo 0 π) :
  {θ : ℝ | θ ∈ set.Ioo 0 π ∧ tan (3 * π * cos θ) = cot (3 * π * sin θ)}.to_finset.card = 16 := by
sorry

end count_solutions_tan_cot_l228_228781


namespace pants_price_l228_228655

theorem pants_price (coat_price pants_discount : ℝ) (h_coat : coat_price = 800) (h_discount : pants_discount = 0.4) : coat_price * (1 - pants_discount) = 480 :=
by
  rw [h_coat, h_discount]
  have : 800 * (1 - 0.4) = 800 * 0.6 := by norm_num
  rw this
  norm_num
  -- sorry

end pants_price_l228_228655


namespace crate_contents_problem_l228_228033

inductive Item
| Oranges
| Apples
| Cucumbers
| Potatoes

open Item

def correct_contents (crate1 crate2 crate3 crate4 : Item) : Prop :=
  crate1 = Apples ∧
  crate2 = Cucumbers ∧
  crate3 = Oranges ∧
  crate4 = Potatoes

def labels_correct (crate1 crate2 crate3 crate4: Item) : Prop :=
  (crate1 = Potatoes → crate2 = Potatoes) ∧
  (crate2 = Oranges → False) ∧
  (crate3 = Apples → crate3 = Oranges ∧ crate3 = Cucumbers ∧ crate3 = Potatoes → False) ∧
  ((crate4 = Cucumbers → crate1 = Cucumbers ∨ crate2 = Cucumbers) ∧
  ¬crate3 = Apples ∧ crate2 = Oranges ∧ crate4 = Potatoes

def main_problem (crate1 crate2 crate3 crate4 : Item) : Prop :=
  labels_correct crate1 crate2 crate3 crate4 → correct_contents crate1 crate2 crate3 crate4

theorem crate_contents_problem :
  ∃ crate1 crate2 crate3 crate4 : Item, main_problem crate1 crate2 crate3 crate4 :=
begin
    existsi Apples,
    existsi Cucumbers,
    existsi Oranges,
    existsi Potatoes,
    unfold main_problem,
    unfold labels_correct,
    unfold correct_contents,
    sorry
end

end crate_contents_problem_l228_228033


namespace total_area_to_paint_proof_l228_228660

def barn_width : ℝ := 15
def barn_length : ℝ := 20
def barn_height : ℝ := 8
def door_width : ℝ := 3
def door_height : ℝ := 7
def window_width : ℝ := 2
def window_height : ℝ := 4

noncomputable def wall_area (width length height : ℝ) : ℝ := 2 * (width * height + length * height)
noncomputable def door_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num
noncomputable def window_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num

noncomputable def total_area_to_paint : ℝ := 
  let total_wall_area := wall_area barn_width barn_length barn_height
  let total_door_area := door_area door_width door_height 2
  let total_window_area := window_area window_width window_height 3
  let net_wall_area := total_wall_area - total_door_area - total_window_area
  let ceiling_floor_area := barn_width * barn_length * 2
  net_wall_area * 2 + ceiling_floor_area

theorem total_area_to_paint_proof : total_area_to_paint = 1588 := by
  sorry

end total_area_to_paint_proof_l228_228660


namespace average_cost_per_apple_l228_228325

/-- Define conditions for buying apples -/
def rate1 : ℕ := 4 -- number of apples for 15 cents
def cost1 : ℕ := 15 -- cost in cents for 4 apples

def rate2 : ℕ := 7 -- number of apples for 25 cents
def cost2 : ℕ := 25 -- cost in cents for 7 apples

def bonus_apples : ℕ := 5 -- additional free apples if at least 20 apples are bought
def min_apples_for_bonus : ℕ := 20 -- minimum apples to get the bonus

def purchased_apples : ℕ := 28 -- number of apples initially purchased

/-- The goal is to prove the average cost per apple using the given rates and bonuses. -/
theorem average_cost_per_apple :
  ∃ avg_cost : ℚ, 
    ((purchased_apples ≥ min_apples_for_bonus) →
    (avg_cost = (4 * cost2) / (purchased_apples + bonus_apples))) :=
begin
  let total_apples_bought := purchased_apples + bonus_apples,
  let total_cost := 4 * cost2,
  use total_cost / total_apples_bought,
  intros h,
  rw [total_apples_bought, total_cost],
  exact rfl,
end

end average_cost_per_apple_l228_228325


namespace positive_difference_eq_30_l228_228256

theorem positive_difference_eq_30 : 
  let x1 := 12
      x2 := -18
  in |x1 - x2| = 30 := 
by
  sorry

end positive_difference_eq_30_l228_228256


namespace better_fit_model_l228_228789

theorem better_fit_model (RSS1 RSS2 : ℕ) (h1 : RSS1 = 168) (h2 : RSS2 = 197) :
  RSS1 < RSS2 :=
by
  rw [h1, h2]
  exact Nat.lt_succ_self 168
  sorry

end better_fit_model_l228_228789


namespace ratio_of_triangle_areas_bcx_acx_l228_228328

theorem ratio_of_triangle_areas_bcx_acx
  (BC AC : ℕ) (hBC : BC = 36) (hAC : AC = 45)
  (is_angle_bisector_CX : ∀ BX AX : ℕ, BX / AX = BC / AC) :
  (∃ BX AX : ℕ, BX / AX = 4 / 5) :=
by
  have h_ratio := is_angle_bisector_CX 36 45
  rw [hBC, hAC] at h_ratio
  exact ⟨4, 5, h_ratio⟩

end ratio_of_triangle_areas_bcx_acx_l228_228328


namespace no_power_of_two_sum_in_kxk_table_l228_228648

theorem no_power_of_two_sum_in_kxk_table (k : ℕ) (h : 1 < k) :
  ¬ (∃ (M : fin k → fin k → ℕ), 
    (∀ i : fin k, ∃ n : ℕ, (∑ j, M i j) = 2^n) ∧
    (∀ j : fin k, ∃ n : ℕ, (∑ i, M i j) = 2^n) ∧
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ k^2) ∧
    (finset.univ.sum (λ i, finset.univ.sum (λ j, M i j)) = (k^2 * (k^2 + 1)) / 2)) :=
sorry

end no_power_of_two_sum_in_kxk_table_l228_228648


namespace statement_C_is_incorrect_l228_228907

def f (x : ℝ) : ℝ := Math.cos (2 * x)

def g (x : ℝ) : ℝ := Math.sin (2 * x)

def h (x : ℝ) : ℝ := f x + g x

def point_of_symmetry (P : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 * P.1 - x) = 2 * P.2 - f x

theorem statement_C_is_incorrect : ¬point_of_symmetry (π / 8, 0) h :=
sorry

end statement_C_is_incorrect_l228_228907


namespace correct_calculation_l228_228628

theorem correct_calculation :
  (∃ x : ℕ, (x^2 * x^3 = x^5) ∨ ((a - b)^2 = a^2 - 2 * a * b + b^2) ∨ (a^8 / a^2 = a^6) ∨ ((-2*x)^2 = 4*x^2)) :=
by
  have hA : ∀ x : ℕ, x^2 * x^3 = x^5,
    from λ x, (pow_add x 2 3).symm,
  have hB : ∀ a b : ℕ, (a - b)^2 = a^2 - 2 * a * b + b^2,
    from λ a b, (square_sub a b).symm,
  have hC : ∀ a : ℕ, a^8 / a^2 = a^6,
    from λ a, (pow_add a 6 2).symm,
  have hD : ∀ x : ℕ, (-2 * x)^2 = 4 * x^2,
    from λ x, by ring,
  apply hC,
  sorry

end correct_calculation_l228_228628


namespace ticTacToe_CarlWins_l228_228740

def ticTacToeBoard := Fin 3 × Fin 3

noncomputable def countConfigurations : Nat := sorry

theorem ticTacToe_CarlWins :
  countConfigurations = 148 :=
sorry

end ticTacToe_CarlWins_l228_228740


namespace range_of_f_l228_228366

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin ((π / 4) * sin (sqrt (x - 2) + x + 2) - (5 * π / 2))

theorem range_of_f :
  (forall x : ℝ, x ≥ 2) → (set.range f = set.Icc (-2 : ℝ) (-real.sqrt 2)) :=
by
  sorry

end range_of_f_l228_228366


namespace denomination_of_remaining_notes_eq_500_l228_228662

-- Definitions of the given conditions:
def total_money : ℕ := 10350
def total_notes : ℕ := 126
def n_50_notes : ℕ := 117

-- The theorem stating what we need to prove
theorem denomination_of_remaining_notes_eq_500 :
  ∃ (X : ℕ), X = 500 ∧ total_money = (n_50_notes * 50 + (total_notes - n_50_notes) * X) :=
by
sorry

end denomination_of_remaining_notes_eq_500_l228_228662


namespace fundraising_part1_fundraising_part2_l228_228133

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l228_228133


namespace quotient_is_six_l228_228223

-- Definition of the given conditions
def S : Int := 476
def remainder : Int := 15
def difference : Int := 2395

-- Definition of the larger number based on the given conditions
def L : Int := S + difference

-- The statement we need to prove
theorem quotient_is_six : (L = S * 6 + remainder) := by
  sorry

end quotient_is_six_l228_228223


namespace merchant_marked_price_l228_228663

theorem merchant_marked_price (L P S x : ℝ) (hL : L = 100)
  (hP : P = L * 0.75)
  (hx : S = x * 0.75)
  (profit_condition : S - P = 0.30 * S) :
  x = 142.86 :=
by
  have hL := L = 100
  have hP := P = L * 0.75
  have hx := S = x * 0.75
  have profit_condition := S - P = 0.30 * S
  show x = 142.86, from sorry

end merchant_marked_price_l228_228663


namespace graphs_with_inverses_l228_228374

-- Define the five graphs as functions (simplified)
def graphA : ℝ → ℝ := λ x, x -- A straight line y = x
def graphB : ℝ → ℝ := λ x, x^2 - 1 -- Parabola y = x^2 - 1
def graphC : ℝ → ℝ -- Segments, defined piecewise
| x :=
  if x ≥ -5 ∧ x ≤ -1 then x + 4
  else if x ≥ 1 ∧ x ≤ 5 then -x + 4
  else 0 -- Outside of defined segments
def graphD : ℝ → Option ℝ -- Semicircle restricted to domain
| x :=
  if x ≥ -3 ∧ x ≤ 3 then 
    some (real.sqrt (9 - x^2)) -- Semicircle part
  else 
    none -- Outside of defined range
def graphE : ℝ → ℝ := λ x, (x^3 / 9 + 4 * x / 3 - 1) -- A cubic function with specific points

-- Main theorem asserting which graphs have inverses
theorem graphs_with_inverses :
  {g : ℕ → ℝ → ℝ // 
    ((g = graphA) ∨ (g = graphC)) ∧
    ¬((g = graphB) ∨ (g = graphD) ∨ (g = graphE))
  } sorry

end graphs_with_inverses_l228_228374


namespace sum_of_digits_h4_100_l228_228519

noncomputable def f (x : ℝ) : ℝ := 10 ^ (7 * x)
noncomputable def g (x : ℝ) : ℝ := Real.log10 (x / 10)
noncomputable def h1 (x : ℝ) : ℝ := g (f x)

def hn : ℕ → (ℝ → ℝ)
| 1 := h1
| (n + 1) := h1 ∘ hn n

theorem sum_of_digits_h4_100 : (Nat.digits 10 (hn 4 100)).sum = 21 :=
by
  sorry

end sum_of_digits_h4_100_l228_228519


namespace number_of_ways_to_assign_students_l228_228035

open Combinatorics

noncomputable def num_ways_to_assign_students : ℕ :=
  let choose_students := Nat.choose 4 2
  let choose_university := 4
  let assign_remaining_students := 3 * 2
  choose_students * choose_university * assign_remaining_students

theorem number_of_ways_to_assign_students :
  num_ways_to_assign_students = 144 :=
by
  unfold num_ways_to_assign_students
  norm_num
  sorry

end number_of_ways_to_assign_students_l228_228035


namespace find_f_of_2013_l228_228080

theorem find_f_of_2013 (a α b β : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)) (h4 : f 4 = 3) :
  f 2013 = -3 := 
sorry

end find_f_of_2013_l228_228080


namespace number_of_oranges_l228_228292

-- Definitions based on conditions
def A : ℝ := 0.26
def O : ℝ := A + 0.28
def total_amount_spent : ℝ := 4.56
def num_apples : ℕ := 3
def num_oranges : ℕ := (total_amount_spent - (num_apples * A)) / O

-- Theorem to prove
theorem number_of_oranges (A : ℝ) (O : ℝ) (total_amount_spent : ℝ) (num_apples : ℕ) (num_oranges : ℕ) :
  A = 0.26 →
  O = A + 0.28 →
  total_amount_spent = 4.56 →
  num_apples = 3 →
  num_oranges = (total_amount_spent - (num_apples * A)) / O →
  num_oranges = 7 :=
by
  intros hA hO hTotal hApples hNumOranges
  sorry

-- EOF

end number_of_oranges_l228_228292


namespace central_vs_northern_chess_match_l228_228334

noncomputable def schedule_chess_match : Nat :=
  let players_team1 := ["A", "B", "C"];
  let players_team2 := ["X", "Y", "Z"];
  let total_games := 3 * 3 * 3;
  let games_per_round := 4;
  let total_rounds := 7;
  Nat.factorial total_rounds

theorem central_vs_northern_chess_match :
    schedule_chess_match = 5040 :=
by
  sorry

end central_vs_northern_chess_match_l228_228334


namespace find_vector_c_l228_228844

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (2, 1)

def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem find_vector_c : 
  perp (c.1 + b.1, c.2 + b.2) a ∧ parallel (c.1 - a.1, c.2 + a.2) b :=
by 
  sorry

end find_vector_c_l228_228844


namespace range_of_a_if_extremum_l228_228905

theorem range_of_a_if_extremum (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, Real.exp x - a * x) : 
  (∃ x, f' x = 0) → a ∈ set.Ioi 0 := 
by
  sorry

end range_of_a_if_extremum_l228_228905


namespace octagon_area_l228_228675

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228675


namespace sin_cos_power_eq_neg_one_l228_228071

theorem sin_cos_power_eq_neg_one (θ : ℝ) :
  (sin θ + cos θ + 1 = 0) ∧ (sin θ ≠ cos θ) → (sin θ) ^ 2017 + (cos θ) ^ 2017 = -1 :=
by sorry

end sin_cos_power_eq_neg_one_l228_228071


namespace trains_clear_time_l228_228616

theorem trains_clear_time
  (len_train1 : ℝ) (len_train2 : ℝ)
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (len_train1_eq : len_train1 = 121)
  (len_train2_eq : len_train2 = 165)
  (speed_train1_eq : speed_train1 = 80)
  (speed_train2_eq : speed_train2 = 55) :
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600 in
  let total_distance := len_train1 + len_train2 in
  total_distance / relative_speed = 7.63 :=
by
  sorry

end trains_clear_time_l228_228616


namespace find_d_l228_228370

-- Definition of the problem
def tangency_condition (d : ℝ) :=
  let y := 3 * x + d
  let discriminant := (6 * d - 12)^2 - 4 * 9 * d^2
  discriminant = 0

-- The theorem to prove
theorem find_d : ∃ d : ℝ, tangency_condition d ∧ d = 1 :=
by 
  use 1
  sorry

end find_d_l228_228370


namespace sum_of_digits_l228_228144

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_of_digits (a b c d : ℕ) (h_distinct : distinct_digits a b c d) (h_eqn : 100*a + 60 + b - (400 + 10*c + d) = 2) :
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
sorry

end sum_of_digits_l228_228144


namespace find_d_l228_228369

-- Definition of the problem
def tangency_condition (d : ℝ) :=
  let y := 3 * x + d
  let discriminant := (6 * d - 12)^2 - 4 * 9 * d^2
  discriminant = 0

-- The theorem to prove
theorem find_d : ∃ d : ℝ, tangency_condition d ∧ d = 1 :=
by 
  use 1
  sorry

end find_d_l228_228369


namespace negation_of_P_l228_228438

-- Definitions used in the condition
def P : Prop := ∃ x₀ : ℝ+, log 2 x₀ = 1

-- The proof problem to check are equivalent to the correct answer
theorem negation_of_P : ¬P ↔ ∀ x₀ : ℝ+, log 2 x₀ ≠ 1 :=
by
  sorry

end negation_of_P_l228_228438


namespace find_simple_interest_years_l228_228597

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

constant P_c : ℝ := 4000
constant r_c : ℝ := 0.10
constant n_c : ℕ := 1
constant t_c : ℝ := 2

constant P_s : ℝ := 1750
constant r_s : ℝ := 0.08
constant SI : ℝ := 420

theorem find_simple_interest_years (t : ℝ) : 
  840 / 2 = SI → SI = P_s * r_s * t → t = 3 :=
by 
  sorry

end find_simple_interest_years_l228_228597


namespace ABCD_concyclic_l228_228823

-- Definitions of the angles in terms of the points
variables {A B C D E F G H : Type*}
variable [has_angle A]
variable [has_angle B]
variable [has_angle C]
variable [has_angle D]
variable [has_angle E]
variable [has_angle F]
variable [has_angle G]
variable [has_angle H]

-- Conditions given in the problem
def angle_AEH_eq_FEB (x : angle) : Prop := x = angle (A E H) ∧ x = angle (F E B)
def angle_EFB_eq_CFG (y : angle) : Prop := y = angle (E F B) ∧ y = angle (C F G)
def angle_CGF_eq_DGH (z : angle) : Prop := z = angle (C G F) ∧ z = angle (D G H)
def angle_DHG_eq_AHE (w : angle) : Prop := w = angle (D H G) ∧ w = angle (A H E)

-- The question: points A, B, C, and D are concyclic
theorem ABCD_concyclic (x y z w : angle) 
  (h1 : angle_AEH_eq_FEB x) 
  (h2 : angle_EFB_eq_CFG y)
  (h3 : angle_CGF_eq_DGH z)
  (h4 : angle_DHG_eq_AHE w) : 
  cyclic_quad A B C D := 
sorry

end ABCD_concyclic_l228_228823


namespace sufficient_not_necessary_condition_l228_228511

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = b * complex.i

theorem sufficient_not_necessary_condition
  (a : ℝ) : (∀ (a : ℝ), a = 1 → pure_imaginary ((a-1)*(a+2) + (a+3)*complex.i))
  ∧ (∃ (a : ℝ), pure_imaginary ((a-1)*(a+2) + (a+3)*complex.i) ∧ a ≠ 1) :=
by
  sorry

end sufficient_not_necessary_condition_l228_228511


namespace calc_hourly_rate_l228_228965

variable (B : ℝ) (C : ℝ) (S : ℝ) (P : ℝ) (D : ℝ) (H : ℝ) (W : ℝ) (E : ℝ) (T : ℝ) (R : ℝ)
variable (B_val : B = 576) (C_val : C = 0.03) (S_val : S = 4000) (P_val : P = 75) (D_val : D = 30)
variable (H_val : H = 8) (W_val : W = 6)
variable (E_val : E = B + (C * S) + P - D)
variable (T_val : T = H * W * 4)
variable (R_val : R = E / T)

-- Statement of the theorem to be proved
theorem calc_hourly_rate : R ≈ 3.86 :=
by
  sorry

end calc_hourly_rate_l228_228965


namespace find_n_l228_228814

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : log 10 (sin x) + log 10 (cos x) = -1)
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 n - 2)) :
  n = 120 := by 
  sorry

end find_n_l228_228814


namespace sum_series_eq_one_third_l228_228013

theorem sum_series_eq_one_third :
  ∑' n : ℕ, (if h : n > 0 then (2^n / (1 + 2^n + 2^(n + 1) + 2^(2 * n + 1))) else 0) = 1 / 3 :=
by
  sorry

end sum_series_eq_one_third_l228_228013


namespace total_games_played_l228_228742

theorem total_games_played 
  (y x total_games_won : ℝ)
  (h1 : x / y = 0.60)
  (h2 : total_games_won = x + 8)
  (h3 : total_games_won / (y + 12) = 0.55) :
  y + 12 = 40 := by
begin
  sorry
end

end total_games_played_l228_228742


namespace correct_coordinates_B_D1_match_distance_l228_228142

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def cube_diagonal_distance : ℝ :=
  real.sqrt ((4:ℝ)^2 + (4:ℝ)^2 + (4:ℝ)^2)

theorem correct_coordinates_B_D1_match_distance :
  let B1 := Point3D.mk 0 4 0
  let D1_1 := Point3D.mk (-4) 0 4
  let B2 := Point3D.mk 2 2 (-2)
  let D1_2 := Point3D.mk (-2) (-2) 2 in
  distance B1 D1_1 = 4 * real.sqrt 3 ∧ distance B2 D1_2 = 4 * real.sqrt 3 := by
  sorry

end correct_coordinates_B_D1_match_distance_l228_228142


namespace license_plate_palindrome_probability_find_m_plus_n_l228_228529

noncomputable section

open Nat

def is_palindrome {α : Type} (seq : List α) : Prop :=
  seq = seq.reverse

def number_of_three_digit_palindromes : ℕ :=
  10 * 10  -- explanation: 10 choices for the first and last digits, 10 for the middle digit

def total_three_digit_numbers : ℕ :=
  10^3  -- 1000

def prob_three_digit_palindrome : ℚ :=
  number_of_three_digit_palindromes / total_three_digit_numbers

def number_of_three_letter_palindromes : ℕ :=
  26 * 26  -- 26 choices for the first and last letters, 26 for the middle letter

def total_three_letter_combinations : ℕ :=
  26^3  -- 26^3

def prob_three_letter_palindrome : ℚ :=
  number_of_three_letter_palindromes / total_three_letter_combinations

def prob_either_palindrome : ℚ :=
  prob_three_digit_palindrome + prob_three_letter_palindrome - (prob_three_digit_palindrome * prob_three_letter_palindrome)

def m : ℕ := 7
def n : ℕ := 52

theorem license_plate_palindrome_probability :
  prob_either_palindrome = 7 / 52 := sorry

theorem find_m_plus_n :
  m + n = 59 := rfl

end license_plate_palindrome_probability_find_m_plus_n_l228_228529


namespace range_of_k_l228_228819

-- Definitions based on conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def given_f (x : ℝ) : ℝ := if x >= 0 then exp (2 * x) else exp (-2 * x)

-- Proof statement
theorem range_of_k (f : ℝ → ℝ) (k : ℝ) : 
  even_function f → (∀ x, x >= 0 → f x = exp (2 * x)) → 
  (∃! x, f x - abs (x - 1) - k * x = 0) → 
  k ∈ [-1, 3] :=
by
  sorry

end range_of_k_l228_228819


namespace find_q_l228_228842

-- Define the conditions and the statement to prove
theorem find_q (p q : ℝ) (hp1 : p > 1) (hq1 : q > 1) 
  (h1 : 1 / p + 1 / q = 3 / 2)
  (h2 : p * q = 9) : q = 6 := 
sorry

end find_q_l228_228842


namespace regular_octagon_area_l228_228708

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228708


namespace amount_of_money_l228_228113

variable (x : ℝ)

-- Conditions
def condition1 : Prop := x < 2000
def condition2 : Prop := 4 * x > 2000
def condition3 : Prop := 4 * x - 2000 = 2000 - x

theorem amount_of_money (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 800 :=
by
  sorry

end amount_of_money_l228_228113


namespace trapezoid_area_l228_228918

theorem trapezoid_area 
  (area_ABE area_ADE : ℝ)
  (DE BE : ℝ)
  (h1 : area_ABE = 40)
  (h2 : area_ADE = 30)
  (h3 : DE = 2 * BE) : 
  area_ABE + area_ADE + area_ADE + 4 * area_ABE = 260 :=
by
  -- sorry admits the goal without providing the actual proof
  sorry

end trapezoid_area_l228_228918


namespace intersection_M_N_l228_228093

def M : set ℕ := {0, 1, 2}

def N : set ℕ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end intersection_M_N_l228_228093


namespace real_values_of_a_l228_228024

noncomputable def P (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b

theorem real_values_of_a (a b : ℝ) :
  (P 0 a b ≠ 0) →
  (P 1 a b ≠ 0) →
  (P 2 a b ≠ 0) →
  (P 1 a b / P 0 a b = P 2 a b / P 1 a b) →
  (∃ b, P x 1 b = 0) :=
by
  sorry

end real_values_of_a_l228_228024


namespace gain_percentages_correct_l228_228665

-- Conditions
def purchase1 : ℕ := 900
def selling1 : ℕ := 1440
def purchase2 : ℕ := 1200
def selling2 : ℕ := 1680
def purchase3 : ℕ := 1500
def selling3 : ℕ := 1950

-- Gains and Gain Percentages for each cycle
def gain1 := selling1 - purchase1
def gain_percentage1 := (gain1 : ℝ) / purchase1 * 100

def gain2 := selling2 - purchase2
def gain_percentage2 := (gain2 : ℝ) / purchase2 * 100

def gain3 := selling3 - purchase3
def gain_percentage3 := (gain3 : ℝ) / purchase3 * 100

-- Total Purchase Price, Total Selling Price, Total Gain and Overall Gain Percentage
def total_purchase := purchase1 + purchase2 + purchase3
def total_selling := selling1 + selling2 + selling3
def total_gain := total_selling - total_purchase
def overall_gain_percentage := (total_gain : ℝ) / total_purchase * 100

-- The final theorem to prove the specified gain percentages
theorem gain_percentages_correct : 
  gain_percentage1 = 60 ∧ 
  gain_percentage2 = 40 ∧ 
  gain_percentage3 = 30 ∧ 
  overall_gain_percentage ≈ 40.8333 :=
by
  sorry

end gain_percentages_correct_l228_228665


namespace cosine_angle_BD1_AF1_l228_228271

variables {A B C A1 B1 C1 D1 F1 : Type*} [InnerProductSpace ℝ Type*]
variables {BC CA CC1 : ℝ}

open_locale real_inner_product_space

-- Assume the type definition for points and vectors
variables [AffineSpace Type* ℝ Type*] 
-- Points are vectors
instance : AddCommGroup Type* := sorry

-- The following conditions create instances and constraints for the problem
def is_triangle_prism (A1 B1 C1 A B C : Type*) := sorry
def is_right_angle (u v : Type*) := sorry
def is_midpoint (p q r : Type*) := sorry

constants (a : ℝ)

-- Conditions
axiom angle_BCA : is_right_angle B C A
axiom midpoint_D1 : is_midpoint D1 A1 B1
axiom midpoint_F1 : is_midpoint F1 A1 C1
axiom BC_eq_CA_C1 : BC = CA ∧ CA = CC1 ∧ CC1 = a

-- Vectors and angle calculation for proof problem
def vector_BD1 := ↑(B1) - ↑(B) + (D1 - B1)
def vector_AF1 := ↑(A1) - ↑(A) + (F1 - A1)
def cos_between_vectors (u v : Type*) : ℝ := (u ⬝ v) / (‖u‖ * ‖v‖)

theorem cosine_angle_BD1_AF1 :
  cos_between_vectors vector_BD1 vector_AF1 = ↑(real.sqrt 30) / 10 :=
sorry

end cosine_angle_BD1_AF1_l228_228271


namespace octagon_area_correct_l228_228696

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228696


namespace regular_octagon_area_l228_228710

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228710


namespace zeroes_at_end_base_8_of_factorial_15_l228_228883

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228883


namespace area_of_inscribed_octagon_l228_228717

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228717


namespace bushes_needed_l228_228162

theorem bushes_needed 
  (petals_per_ounce : ℕ) (petals_per_rose : ℕ) (roses_per_bush : ℕ)
  (ounces_per_bottle : ℕ) (bottles_needed : ℕ)
  (h1 : petals_per_ounce = 320)
  (h2 : petals_per_rose = 8)
  (h3 : roses_per_bush = 12)
  (h4 : ounces_per_bottle = 12)
  (h5 : bottles_needed = 20) : 
  let roses_per_ounce := petals_per_ounce / petals_per_rose,
      roses_per_bottle := roses_per_ounce * ounces_per_bottle,
      total_roses := roses_per_bottle * bottles_needed,
      bushes_needed := total_roses / roses_per_bush
  in bushes_needed = 800 :=
by sorry

end bushes_needed_l228_228162


namespace multiplication_of_fractions_l228_228896

theorem multiplication_of_fractions :
  (77 / 4) * (5 / 2) = 48 + 1 / 8 := 
sorry

end multiplication_of_fractions_l228_228896


namespace mutually_exclusive_events_not_complementary_l228_228999

def event_a (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 1
def event_b (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 2

theorem mutually_exclusive_events_not_complementary :
  (∀ ball box, event_a ball box → ¬ event_b ball box) ∧ 
  (∃ box, ¬((event_a 1 box) ∨ (event_b 1 box))) :=
by
  sorry

end mutually_exclusive_events_not_complementary_l228_228999


namespace regular_octagon_area_l228_228683

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l228_228683


namespace equilateral_triangle_combination_l228_228412

-- Given the interior angle of a polygon in degrees
constant interior_angle : ℕ → ℚ
-- Values of interior angles for each polygon mentioned
def interior_angle_quad := 90    -- Regular Quadrilateral
def interior_angle_hex  := 120   -- Regular Hexagon
def interior_angle_oct  := 135   -- Regular Octagon
def interior_angle_tri  := 60    -- Equilateral Triangle
def fixed_angle := 150  -- Given regular polygon
 
-- Define the seamless combination condition
def seamless_combination (a b : ℚ) : Prop := ∃ k l : ℕ, k ≥ 1 ∧ l ≥ 1 ∧ k * a + l * b = 360

theorem equilateral_triangle_combination:
  seamless_combination fixed_angle interior_angle_tri ∧
  ¬ seamless_combination fixed_angle interior_angle_quad ∧
  ¬ seamless_combination fixed_angle interior_angle_hex ∧
  ¬ seamless_combination fixed_angle interior_angle_oct :=
by
  sorry

end equilateral_triangle_combination_l228_228412


namespace sum_coefficients_zero_sum_even_odd_coefficients_equal_l228_228626

-- Part 1: Proving the property for root 1
theorem sum_coefficients_zero {n : ℕ} (a : Fin n.succ → ℤ) :
  (∑ i in Finset.range n.succ, a i) = 0 ↔ (∑ i in Finset.range n.succ, a i * 1^i) = 0 :=
by sorry

-- Part 2: Proving the property for root -1
theorem sum_even_odd_coefficients_equal {n : ℕ} (b : Fin n.succ → ℤ) :
  (∑ i in Finset.range (n.succ) | i % 2 = 0, b i) = 
  (∑ i in Finset.range (n.succ) | i % 2 = 1, b i) ↔
  (∑ i in Finset.range n.succ, b i * (-1 : ℤ)^i) = 0 :=
by sorry

end sum_coefficients_zero_sum_even_odd_coefficients_equal_l228_228626


namespace periodic_odd_fn_calc_l228_228462

theorem periodic_odd_fn_calc :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, 0 < x ∧ x < 1 → f x = 4^x) →
  f (-5 / 2) + f 2 = -2 :=
by
  intros f h
  sorry

end periodic_odd_fn_calc_l228_228462


namespace number_of_trailing_zeroes_base8_l228_228864

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228864


namespace subset_implies_a_values_l228_228092

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_implies_a_values (a : ℝ) (h : B a ⊆ A) : a ∈ {1, -1} := 
by 
  sorry

end subset_implies_a_values_l228_228092


namespace minimum_toothpicks_to_remove_l228_228768

-- Define the problem statement using Lean's theorem proving language
theorem minimum_toothpicks_to_remove 
  (toothpicks : ℕ)
  (initial_rows : ℕ)
  (triangles_in_row : ℕ → ℕ)
  (alternating_triangles : ℕ → bool)
  (valid_structure : ∀ i, i ≤ initial_rows → triangles_in_row i ≤ initial_rows - i)
  (valid_toothpicks : ∑ i in range (initial_rows + 1), triangles_in_row i * 3 ≤ toothpicks)
  (base_row_triangles : triangles_in_row 0 = initial_rows) :
  (minimal_toothpicks_removed : ℕ) →
  (ensuring_no_triangles_remain : ∀ i, triangles_in_row i = 0 → i ≤ minimal_toothpicks_removed)
  (minimal_toothpicks_removed = 5) :=
sorry

end minimum_toothpicks_to_remove_l228_228768


namespace half_percent_of_160_l228_228620

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l228_228620


namespace every_term_is_integer_l228_228968

-- Define the problem in Lean 4
noncomputable def sequence (a b : ℤ) : ℕ → ℤ
| 0 := a
| 1 := a
| (n+2) := (sequence (n+1))^2 + b) / (sequence n)

theorem every_term_is_integer (a1 a2 b : ℤ) 
  (h1 : ∀ n, sequence a1 a2 b n ≠ 0)
  (h2 : a1 ≠ 0)
  (h3 : a2 ≠ 0)
  (h4 : (a1^2 + a2^2 + b) % (a1 * a2) = 0) : 
  ∀ n, (sequence a1 a2 b n) ∈ ℤ :=
begin
  sorry
end

end every_term_is_integer_l228_228968


namespace integral_f_val_l228_228329
-- Import the comprehensive math library

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 - x else real.sqrt (4 - x^2)

-- State the theorem
theorem integral_f_val : ∫ x in -2..2, f x = real.pi + 10 :=
by
  -- Proof not provided
  sorry

end integral_f_val_l228_228329


namespace smallest_odd_k_prime_poly_l228_228025

theorem smallest_odd_k_prime_poly (k : ℕ) : 
  (∀ (f : ℤ[X]), (∃ (n : ℤ) (hn : |f.eval n|.prime), true) → irreducible f) ↔ k = 5 :=
by sorry

end smallest_odd_k_prime_poly_l228_228025


namespace part1_part2_l228_228179

-- Conditions
def S (n : ℕ) : ℚ := if n = 0 then 0 else n * (1 + 2 / 3) -- Dummy definition based on conditions
def a : ℕ → ℚ 
| 1 := 1
| 2 := 2 / 3
| _ := 0 -- Placeholder for other terms

axiom arithmetic_sequence :
  ∀ n : ℕ, 4 * n * S n + (2 * n + 3) * a n = 9 * n

-- Part (1)
theorem part1 (n : ℕ) (hn : n ≥ 1) :
  ∃ r : ℚ, r ≠ 0 ∧ 
            (∀ m : ℕ, m ≥ 1 → a m = r * ((1/3) : ℚ)^(m-1)) :=
sorry

-- Part (2)
def b : ℕ → ℚ
| n := if n % 2 = 1 then 3^(n-1) * a n else n / a n

noncomputable def T_2n (n : ℕ) : ℚ := 
  ∑ i in (range (2 * n)).filter (λ m => m % 2 = 1), b i +
  ∑ j in (range (2 * n)).filter (λ m => m % 2 = 0), b j

theorem part2 (n : ℕ) :
  T_2n n = n^2 + (3^(2*n + 1) - 3)/8 :=
sorry

end part1_part2_l228_228179


namespace chocolate_cake_cost_is_12_l228_228728

noncomputable theory

-- Let x be the price of each chocolate cake
def price_of_chocolate_cake (x : ℝ) : Prop :=
  -- Conditions
  let chocolate_cost := 3 * x,
      strawberry_cost := 6 * 22,
      total_cost := chocolate_cost + strawberry_cost in
  total_cost = 168

-- Prove that the price of each chocolate cake is $12
theorem chocolate_cake_cost_is_12 : price_of_chocolate_cake 12 :=
by
  have h1 : 3 * 12 = 36 := rfl,
  have h2 : 6 * 22 = 132 := rfl,
  have h3 : 36 + 132 = 168 := rfl,
  show price_of_chocolate_cake 12, from
  begin
    unfold price_of_chocolate_cake,
    rw [h1, h2, h3]
  end

end chocolate_cake_cost_is_12_l228_228728


namespace value_of_a_in_S_l228_228090

variable (S : Set ℕ) (T : Set ℕ) (a : ℕ)
variable (hS : S = {1, 2}) (hT : T = {a}) (h_union : S ∪ T = S)

theorem value_of_a_in_S : a ∈ {1, 2} :=
by
  rw [←hS, ←hT] at h_union
  have : T ⊆ S, from Set.subset_of_union_eq h_union
  rw [Set.singleton_subset_iff] at this
  exact this

end value_of_a_in_S_l228_228090


namespace ellipse_characteristics_l228_228422

theorem ellipse_characteristics :
  ∀ (x y : ℝ), (x^2 / 16 + y^2 / 4 = 1) →
    (∃ (a b c : ℝ), a = 4 ∧ b = 2 ∧ c = 2 * Real.sqrt 3 ∧
      2 * a = 8 ∧                      -- Length of the major axis is 8
      2 * c = 4 * Real.sqrt 3 ∧        -- The focal length is 4 * sqrt(3)
      (0, ±2 * Real.sqrt 3) = 0 → False ∧  -- Coordinates of foci
      c / a = Real.sqrt 3 / 2)          -- The eccentricity is sqrt(3) / 2 :=
by
  intros x y ellipse_eq
  use 4, 2, 2 * Real.sqrt 3
  split
  -- Proof for a = 4
  sorry
  split
  -- Proof for b = 2
  sorry
  split
  -- Proof for c = 2 * Real.sqrt 3
  sorry
  split
  -- Proof for length of the major axis
  exact (2 * 4 = 8)
  split
  -- Proof for focal length
  exact (2 * 2 * Real.sqrt 3 = 4 * Real.sqrt 3)
  split
  -- Proof for coordinates of the foci
  intro foci_eq
  exact False.elim (sorry)  -- Need to show that the coordinates (0, ±2 * Real.sqrt 3) are incorrect
  -- Proof for eccentricity
  exact (2 * Real.sqrt 3 / 4 = Real.sqrt 3 / 2)

-- Temporary "sorry" placeholders will be replaced by the detailed proofs if necessary.

end ellipse_characteristics_l228_228422


namespace mila_daily_phone_hours_l228_228355

def weekly_social_media_hours : ℕ := 21
def week_days : ℕ := 7
def daily_phone_hours (weekly_phone_hours : ℕ) : ℕ := weekly_phone_hours / week_days

theorem mila_daily_phone_hours : 
    let total_weekly_phone_hours := 2 * weekly_social_media_hours in
    daily_phone_hours total_weekly_phone_hours = 6 := 
by
  -- We will skip the proof for now.
  sorry

end mila_daily_phone_hours_l228_228355


namespace range_of_a_l228_228075

open Real

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0

def Q (a : ℝ) : Prop := ∀ x ∈ set.Ici (1 : ℝ), deriv (λ y, 4 * y ^ 2 - a * y) x > 0

theorem range_of_a (a : ℝ) : (P a) ∨ (Q a) → ¬ (P a) → (a ≤ 0 ∨ 4 ≤ a ∧ a ≤ 8) :=
by
  sorry

end range_of_a_l228_228075


namespace min_value_abs_expression_l228_228811

theorem min_value_abs_expression (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) : 
  ∃ α β : ℝ, |2 * x - y - 2| = 5 - sqrt 5 :=
sorry

end min_value_abs_expression_l228_228811


namespace solve_equation_l228_228563
-- Import the necessary library

-- Define the equation as a function
def equation (x : ℝ) : ℝ :=
  real.cbrt (10 * x - 1) + real.cbrt (20 * x + 1) - 3 * real.cbrt (5 * x)

-- State the main theorem
theorem solve_equation :
  (equation 0 = 0) ∧ (equation (1/10) = 0) ∧ (equation (-45/973) = 0) :=
by
  -- Proof is left as an exercise
  sorry

end solve_equation_l228_228563


namespace part1_convergence_largest_a_l228_228047

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 1       := 1
| 2       := 0
| (n + 2) := (sequence a n ^ 2 + sequence a (n + 1) ^ 2) / 4 + a

theorem part1_convergence (a : ℝ) (h : a = 0): 
  ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (sequence a n - L) < ε :=
sorry

theorem largest_a (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (sequence a n - sequence a (2*N)) < ε) ↔ a ≤ 1/2 :=
sorry

end part1_convergence_largest_a_l228_228047


namespace cannot_cover_5x5_board_with_1x2_dominoes_l228_228332

theorem cannot_cover_5x5_board_with_1x2_dominoes :
  ¬ ∃ (f : Fin 25 → Fin 2), (∀ i, True) := 
sorry

end cannot_cover_5x5_board_with_1x2_dominoes_l228_228332


namespace problem_statement_l228_228054

theorem problem_statement (x y : ℝ) (h1 : x^8 + y^8 ≤ 2) : 
    x^2 * y^2 + |x^2 - y^2| ≤ real.pi / 2 :=
sorry

end problem_statement_l228_228054


namespace derivative_y_over_x_l228_228642

noncomputable def x (t : ℝ) : ℝ := (t^2 * Real.log t) / (1 - t^2) + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) : ℝ := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

theorem derivative_y_over_x (t : ℝ) (ht : t ≠ 0) (h1 : t ≠ 1) (hneg1 : t ≠ -1) : 
  (deriv y t) / (deriv x t) = (Real.arcsin t * Real.sqrt (1 - t^2)) / (2 * t * Real.log t) :=
by
  sorry

end derivative_y_over_x_l228_228642


namespace cost_price_correct_l228_228274

noncomputable def cost_price_per_meter (selling_price_per_meter : ℝ) (total_meters : ℝ) (loss_per_meter : ℝ) :=
  (selling_price_per_meter * total_meters + loss_per_meter * total_meters) / total_meters

theorem cost_price_correct :
  cost_price_per_meter 18000 500 5 = 41 :=
by 
  sorry

end cost_price_correct_l228_228274


namespace correct_order_of_y_values_l228_228402

-- Define the conditions as functions and variables in Lean.
variable (k : ℝ)
variable (x1 y1 x2 y2 : ℝ)

def is_on_parabola (x y : ℝ) (k : ℝ) : Prop :=
  y = (x - 2) ^ 2 + k

def x1_on_parabola : Prop := is_on_parabola x1 y1 k
def x2_on_parabola : Prop := is_on_parabola x2 y2 k
def x1_less_than_2 : Prop := x1 < 2
def x2_greater_than_2 : Prop := x2 > 2
def x1_plus_x2_less_than_4 : Prop := x1 + x2 < 4

-- Lean statement to prove the correct answer using given conditions.
theorem correct_order_of_y_values :
  x1_on_parabola ∧ x2_on_parabola ∧ x1_less_than_2 ∧ x2_greater_than_2 ∧ x1_plus_x2_less_than_4 → y1 > y2 ∧ y2 > k :=
by
  sorry

end correct_order_of_y_values_l228_228402


namespace train_cross_time_l228_228153

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  (speed * 1000) / 3600

noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time (length : ℝ) (speed_km_per_hr : ℝ) :
  length = 100 → speed_km_per_hr = 144 → time_to_cross length (speed_km_per_hr_to_m_per_s speed_km_per_hr) = 2.5 :=
by
  intros length_eq speed_eq
  rw [length_eq, speed_eq]
  simp [speed_km_per_hr_to_m_per_s, time_to_cross]
  norm_num
  sorry

end train_cross_time_l228_228153


namespace find_z_range_m_l228_228043

noncomputable def z (b : ℝ) : ℂ := b * complex.I

theorem find_z (b : ℝ) (h : (z b - 2) / (1 + complex.I) ∈ set.range (λ x : ℝ, (x : ℂ))) : z b = -2 * complex.I :=
by sorry

theorem range_m (m : ℝ) (h : ((m + -2 * complex.I) ^ 2).im = 0 ∧ ((m + -2 * complex.I) ^ 2).re > 0) : m < -2 :=
by sorry

end find_z_range_m_l228_228043


namespace range_x_f_inequality_l228_228829

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem range_x_f_inequality :
  (∀ x : ℝ, f (2 * x + 1) ≥ f x) ↔ x ∈ Set.Icc (-1 : ℝ) (-1 / 3) := sorry

end range_x_f_inequality_l228_228829


namespace range_of_f_l228_228825

noncomputable def f (x : ℝ) : ℝ := sin x / cos (x + π / 6)

theorem range_of_f :
  set.range (λ x, f x) = set.Icc ((sqrt 3 - 1) / 2) 1 :=
  sorry

end range_of_f_l228_228825


namespace range_of_a_l228_228468

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (1 : ℝ) 2, x^2 + a ≤ a * x - 3) ↔ 7 ≤ a :=
sorry

end range_of_a_l228_228468


namespace five_in_range_for_all_b_l228_228792

noncomputable def f (x b : ℝ) := x^2 + b * x - 3

theorem five_in_range_for_all_b : ∀ (b : ℝ), ∃ (x : ℝ), f x b = 5 := by 
  sorry

end five_in_range_for_all_b_l228_228792


namespace gray_region_area_l228_228145

theorem gray_region_area 
  (r : ℝ) 
  (h1 : ∀ r : ℝ, (3 * r) - r = 3) 
  (h2 : r = 1.5) 
  (inner_circle_area : ℝ := π * r * r) 
  (outer_circle_area : ℝ := π * (3 * r) * (3 * r)) : 
  outer_circle_area - inner_circle_area = 18 * π := 
by
  sorry

end gray_region_area_l228_228145


namespace range_of_function_l228_228760

theorem range_of_function :
  ∀ (y : ℝ), (∃ (x : ℝ), y = x - real.sqrt (1 - 4 * x) ∧ 1 - 4 * x ≥ 0) ↔ y ≤ 1 / 4 :=
by
  sorry

end range_of_function_l228_228760


namespace augmented_matrix_correct_l228_228218

theorem augmented_matrix_correct:
  ∃ A B, (∀ x y, A.mul_vec (λ i, if i = 0 then x else if i = 1 then y else 0) = B) ↔
    A = ![![1, -2], ![3, 1]] ∧ B = ![5, 8] := by
  sorry

end augmented_matrix_correct_l228_228218


namespace number_of_students_preferring_dogs_l228_228764

-- Define the conditions
def total_students : ℕ := 30
def dogs_video_games_chocolate_percentage : ℚ := 0.50
def dogs_movies_vanilla_percentage : ℚ := 0.10
def cats_video_games_chocolate_percentage : ℚ := 0.20
def cats_movies_vanilla_percentage : ℚ := 0.15

-- Define the target statement to prove
theorem number_of_students_preferring_dogs : 
  (dogs_video_games_chocolate_percentage + dogs_movies_vanilla_percentage) * total_students = 18 :=
by
  sorry

end number_of_students_preferring_dogs_l228_228764


namespace find_ellipse_coefficients_l228_228321

noncomputable def sqrt_with_sign (c : ℤ) : ℝ :=
if 0 ≤ c then real.sqrt (c:ℝ) else -real.sqrt (-(c:ℝ))

def foci := (⟨(1 : ℝ), (1 : ℝ)⟩, ⟨(1 : ℝ), (5 : ℝ)⟩)
def point := (⟨(10 : ℝ), (3 : ℝ)⟩)

theorem find_ellipse_coefficients (a b h k : ℝ) :
  let c := real.sqrt(81 + 4)
  let dist := c + c
  let major := 18 * sqrt_with_sign 2
  let foci_distance := real.sqrt ((1 - 1)^2 + (1 - 5)^2)
  let minor := sqrt_with_sign ((major^2 - foci_distance^2))
  let center := (1, 3)
  a = 2 * sqrt_with_sign 158 →
  b = 9 * sqrt_with_sign 2 →
  h = 1 →
  k = 3 →
  dist = major →
  minor = 4 * sqrt_with_sign 158 →
  foci_distance = 4 →
  center.1 = 1 →
  center.2 = 3 →
  (a, b, h, k) = (2 * sqrt_with_sign 158, 9 * sqrt_with_sign 2, 1, 3) :=
by {
  sorry
}

end find_ellipse_coefficients_l228_228321


namespace smallest_divisor_l228_228020

-- Define the given number and the subtracting number
def original_num : ℕ := 378461
def subtract_num : ℕ := 5

-- Define the resulting number after subtraction
def resulting_num : ℕ := original_num - subtract_num

-- Theorem stating that 47307 is the smallest divisor greater than 5 of 378456
theorem smallest_divisor : ∃ d: ℕ, d > 5 ∧ d ∣ resulting_num ∧ ∀ x: ℕ, x > 5 → x ∣ resulting_num → d ≤ x := 
sorry

end smallest_divisor_l228_228020


namespace positive_difference_of_solutions_l228_228255

theorem positive_difference_of_solutions : 
    (∀ x : ℝ, |x + 3| = 15 → (x = 12 ∨ x = -18)) → 
    (abs (12 - (-18)) = 30) :=
begin
  intros,
  sorry
end

end positive_difference_of_solutions_l228_228255


namespace min_sum_expression_l228_228776

theorem min_sum_expression (x y : ℤ) :
  ∑ i in Finset.range 10, ∑ j in Finset.range 10, ∑ k in Finset.range 10, 
  (|k * (x + y - 10 * i) * (3 * x - 6 * y - 36 * j) * (19 * x + 95 * y - 95 * k)|) 
  = 2394000000 :=
sorry

end min_sum_expression_l228_228776


namespace ratio_constant_l228_228586

theorem ratio_constant (a b c d : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
    (h : ∀ k : ℕ, ∃ m : ℤ, a + c * k = m * (b + d * k)) :
    ∃ m : ℤ, ∀ k : ℕ, a + c * k = m * (b + d * k) :=
    sorry

end ratio_constant_l228_228586


namespace div_count_l228_228505

theorem div_count (n : ℕ) (p : ℕ → ℕ) (hp : ∀ i, i < n → Nat.Prime (p i) ∧ p i > 3) :
  (∃ k, k ≥ 4^n ∧ ∃ divs, (2^(list.prod (list.of_fn p)) + 1) = list.prod (list.take k divs) ∧ divs.nodup) :=
by
  sorry

end div_count_l228_228505


namespace octagon_area_l228_228703

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228703


namespace smallest_period_of_f_range_of_f_l228_228385

def a (x : ℝ) : ℝ × ℝ := (sin x, -cos x)
def b (x : ℝ) : ℝ × ℝ := (cos x, sqrt 3 * cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + sqrt 3 / 2

theorem smallest_period_of_f : ∀ x, f x = f (x + π) :=
by sorry

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : -sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 :=
by sorry

end smallest_period_of_f_range_of_f_l228_228385


namespace largest_area_rectangle_in_circle_is_square_l228_228998

theorem largest_area_rectangle_in_circle_is_square (R x : ℝ) (h : x^2 + (4 * R^2 - x^2) = (2 * R)^2) : 
  2 * R^2 ≥ x * sqrt (4 * R^2 - x^2) := 
sorry

end largest_area_rectangle_in_circle_is_square_l228_228998


namespace three_asleep_simultaneously_l228_228351

noncomputable def five_mathematicians := Finset.range 5

structure Mathematician :=
(asleep1 : ℝ)
(asleep2 : ℝ)

def all_pairs_have_overlap (M : Finset Mathematician) :=
  ∀ (m1 m2 : Mathematician) (h1 : m1 ∈ M) (h2 : m2 ∈ M), (m1 ≠ m2) → ∃ t, (t ∈ {m1.asleep1, m1.asleep2}) ∧ (t ∈ {m2.asleep1, m2.asleep2})

theorem three_asleep_simultaneously :
  ∀ (M : Finset Mathematician), M = five_mathematicians →
  (∀ m ∈ M, m.asleep1 < m.asleep2) →
  all_pairs_have_overlap M →
  ∃ t, ∃ (m1 m2 m3 : Mathematician) (h1 : m1 ∈ M) (h2 : m2 ∈ M) (h3 : m3 ∈ M), 
    t ∈ {m1.asleep1, m1.asleep2} ∧ t ∈ {m2.asleep1, m2.asleep2} ∧ t ∈ {m3.asleep1, m3.asleep2} :=
begin
  sorry
end

end three_asleep_simultaneously_l228_228351


namespace part_a_part_b_l228_228504

noncomputable def sequence_a (n : ℕ) : ℝ :=
if n = 1 then 6 else (2 * (n - 1)) / (n - 1) + real.sqrt(((n - 1) / (n - 1)) * sequence_a (n - 1) + 4)

theorem part_a (n : ℕ) : a = 0 → ∃ l : ℝ, tendsto (λ n, sequence_a (n)) at_top (nhds l) ∧ l = 5 := 
by
  sorry

noncomputable def sequence_b (a : ℝ) (n : ℕ) : ℝ :=
if n = 1 then 6 else ((2 * (n - 1) + a) / (n - 1)) + real.sqrt(((n - 1 + a) / (n - 1)) * sequence_b a (n - 1) + 4)

theorem part_b (a : ℝ) (n : ℕ) : a ≥ 0 → ∃ l : ℝ, tendsto (λ n, sequence_b a (n)) at_top (nhds l) ∧ l = 5 := 
by
  sorry

end part_a_part_b_l228_228504


namespace transformed_sum_l228_228721

variables {α : Type*} [field α]

theorem transformed_sum (x : Finₓ n → α) :
  let s := ∑ i, x i
  in 21 * s - 30 * n = ∑ i, 21 * x i - 30 :=
by
  sorry

end transformed_sum_l228_228721


namespace set_difference_l228_228094

-- Definition of sets
def M : Set ℝ := {x | x^2 + x - 12 ≤ 0}
def N : Set ℝ := {x | 3^x ∈ Ioc 0 3}

-- The proof statement
theorem set_difference (x : ℝ) : x ∈ M ∧ x ∉ N ↔ x ∈ Ico (-4) 0 :=
sorry

end set_difference_l228_228094


namespace factorial_base8_trailing_zeros_l228_228886

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228886


namespace main_theorem_l228_228377

-- Define the conditions as sets
def M_d (d : ℕ) : set ℕ := {n | ¬ ∃ a k : ℕ, k ≥ 2 ∧ n = k * (a + d * (k - 1) / 2)}
def A : set ℕ := {n | ∃ k : ℕ, n = 2^k}
def B : set ℕ := {n | n = 1 ∨ (prime n ∧ n ≠ 2)}
def C : set ℕ := M_d 3

-- Define the main theorem to be proven
theorem main_theorem (c : ℕ) (h : c ∈ C) : ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ c = a * b :=
sorry

end main_theorem_l228_228377


namespace zeroes_at_end_base_8_of_factorial_15_l228_228881

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228881


namespace part1_part2_l228_228425

-- Part 1
noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + a

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≤ a) → a ≥ 1 / Real.exp 1 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (x₀ : ℝ) : 
  (∀ x : ℝ, f x₀ a < f x a → x = x₀) → a < 1 / 2 → 2 * a - 1 < f x₀ a ∧ f x₀ a < 0 :=
by
  sorry

end part1_part2_l228_228425


namespace sum_pairwise_relatively_prime_integers_eq_160_l228_228609

theorem sum_pairwise_relatively_prime_integers_eq_160
  (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_prod : a * b * c = 27000)
  (h_coprime_ab : Nat.gcd a b = 1)
  (h_coprime_bc : Nat.gcd b c = 1)
  (h_coprime_ac : Nat.gcd a c = 1) :
  a + b + c = 160 :=
by
  sorry

end sum_pairwise_relatively_prime_integers_eq_160_l228_228609


namespace true_propositions_l228_228538

open Real

theorem true_propositions :
  (∃ x, 10^x = x) ∧ (∃ x, 10^x = x^2) :=
by
  -- Proposition 1: 10^x = x has real solutions
  have P1 : ∃ x : ℝ, 10^x = x := sorry,
  -- Proposition 2: 10^x = x^2 has real solutions
  have P2 : ∃ x : ℝ, 10^x = x^2 := sorry,
  exact ⟨P1, P2⟩

end true_propositions_l228_228538


namespace stickers_missing_fraction_l228_228946

theorem stickers_missing_fraction (y : ℝ) (h1 : y > 0) :
  let lost := (1 / 3) * y in
  let found := (3 / 4) * lost in
  let remaining := y - lost + found in
  y - remaining = (1 / 12) * y :=
by
  sorry

end stickers_missing_fraction_l228_228946


namespace exists_coloring_l228_228959

-- Definition of the problem in Lean 4

variable (S : Finset α) (hS : S.card = 2002)
variable (N : ℕ) (hN : 0 ≤ N ∧ N ≤ 2^2002)

theorem exists_coloring (S : Finset α) (hS : S.card = 2002) (N : ℕ) (hN : 0 ≤ N ∧ N ≤ 2^2002) :
  ∃ (color : Finset (Finset α) → Prop), (∀ A B, color A → color B → color (A ∪ B)) ∧
    (∀ A B, ¬color A → ¬color B → ¬color (A ∪ B)) ∧ (S.filter color).card = N :=
sorry

end exists_coloring_l228_228959


namespace compound_proposition_l228_228970

def f (x : ℝ) := 2^x
def g (x : ℝ) := cos x

theorem compound_proposition :
  (∀ x, f x > f (x - 1)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end compound_proposition_l228_228970


namespace solve_for_x_l228_228639

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 :=
by {
  sorry
}

end solve_for_x_l228_228639


namespace probability_black_white_l228_228659

structure Jar :=
  (black_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)

def total_balls (j : Jar) : ℕ :=
  j.black_balls + j.white_balls + j.green_balls

def choose (n k : ℕ) : ℕ := n.choose k

theorem probability_black_white (j : Jar) (h_black : j.black_balls = 3) (h_white : j.white_balls = 3) (h_green : j.green_balls = 1) :
  (choose 3 1 * choose 3 1) / (choose (total_balls j) 2) = 3 / 7 :=
by
  sorry

end probability_black_white_l228_228659


namespace smallest_positive_value_of_diff_l228_228784

theorem smallest_positive_value_of_diff (k m : ℕ) (hk : k > 0) (hm : m > 0) :
  ∃ k m, 36^k - 5^m = 11 :=
begin
  sorry
end

end smallest_positive_value_of_diff_l228_228784


namespace distance_AB_eq_4sqrt10_polar_equation_line_AB_l228_228137

noncomputable def parametric_curve (t : ℝ) :=
  (2 - t - t ^ 2, 2 - 3 * t + t ^ 2)

-- Part 1: Prove the distance |AB| between points A and B is 4√10
theorem distance_AB_eq_4sqrt10 : 
  let A := (-4 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 12 : ℝ)
  ∃ (A B : ℝ × ℝ), A = (-4, 0) ∧ B = (0, 12) ∧ dist A B = 4 * real.sqrt 10 := 
by sorry

-- Part 2: Prove the polar coordinate equation of the line AB is 3ρcosθ - ρsinθ + 12 = 0
theorem polar_equation_line_AB :
  let cartesian_eq : ℝ × ℝ → Prop :=
    λ (x y : ℝ), 3 * x - y + 12 = 0
  ∃ (ρ θ : ℝ), cartesian_eq (ρ * real.cos θ) (ρ * real.sin θ) :=
by sorry

end distance_AB_eq_4sqrt10_polar_equation_line_AB_l228_228137


namespace count_integers_in_range_l228_228021

theorem count_integers_in_range : 
  {n : ℤ | 11 < n^2 ∧ n^2 < 121}.to_finset.card = 16 := 
by
  sorry

end count_integers_in_range_l228_228021


namespace stock_percentage_l228_228231

-- Definitions from conditions
def income : ℝ := 756
def investment : ℝ := 8000
def brokerage_rate : ℝ := 0.25 / 100 -- 1/4% brokerage
def brokerage_fee : ℝ := brokerage_rate * investment
def net_investment : ℝ := investment - brokerage_fee
def market_value : ℝ := 110.86111111111111

-- Theorem that needs to be proven
theorem stock_percentage : (income / net_investment) * 100 = 9.47 :=
by
  -- Proof is omitted as per instructions
  sorry

end stock_percentage_l228_228231


namespace number_of_trailing_zeroes_base8_l228_228859

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228859


namespace monic_polynomial_has_root_l228_228356

noncomputable def polynomial_definition : Polynomial ℚ :=
  Polynomial.X^4 - 10 * Polynomial.X^2 + 1

theorem monic_polynomial_has_root :
  Polynomial.monic polynomial_definition ∧
  polynomial_definition.degree = 4 ∧
  (polynomial_definition.eval (Real.sqrt 2 + Real.sqrt 3) = 0) :=
by
  sorry

end monic_polynomial_has_root_l228_228356


namespace mappings_count_l228_228000

-- Define the sets A and B with their respective sizes
def A := Fin 4
def B := Fin 3

-- Define the condition that every element in B must have a pre-image in A
def hasPreimage (f : A → B) : Prop := ∀ b : B, ∃ a : A, f a = b

-- The theorem to prove that the number of such mappings is 36
theorem mappings_count : ∃ (count : ℕ), count = 36 ∧ ∃ (f : A → B), hasPreimage f :=
by
  existsi 36
  split
  repeat trivial
  sorry

end mappings_count_l228_228000


namespace factorial_ends_with_base_8_zeroes_l228_228877

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l228_228877


namespace ratio_of_part_to_whole_l228_228543

theorem ratio_of_part_to_whole (N : ℝ) (h1 : (1/3) * (2/5) * N = 15) (h2 : (40/100) * N = 180) :
  (15 / N) = (1 / 7.5) :=
by
  sorry

end ratio_of_part_to_whole_l228_228543


namespace circle_through_focus_l228_228807

-- Given the ellipse E with the equation (x^2 / 2) + y^2 = 1 and eccentricity e = sqrt(2) / 2
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 2) + y^2 = 1

-- Eccentricity of the ellipse
def eccentricity : ℝ := Real.sqrt 2 / 2

-- Foci of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Tangent line to the ellipse E at point R
def tangent_line (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Intersection of tangent line with x = 2
def intersection_point (k m : ℝ) : ℝ × ℝ := (2, 2 * k + m)

-- Perimeter of the triangle R F1 F2
def perimeter_triangle (R F1 F2 : ℝ × ℝ) : ℝ :=
  dist R F1 + dist F1 F2 + dist F2 R

theorem circle_through_focus (k m : ℝ) (R : ℝ × ℝ) (N : ℝ × ℝ) :
  tangent_line k m R.1 R.2 ∧ (N = intersection_point k m)
  → ∃ c : ℝ × ℝ → ℝ × ℝ → ℝ → Prop, c R N (dist R N) ∧ c R N = λ (P : ℝ × ℝ), P = right_focus := sorry

end circle_through_focus_l228_228807


namespace proof_y_times_1_minus_g_eq_1_l228_228964
noncomputable def y : ℝ := (3 + Real.sqrt 8) ^ 100
noncomputable def m : ℤ := Int.floor y
noncomputable def g : ℝ := y - m

theorem proof_y_times_1_minus_g_eq_1 :
  y * (1 - g) = 1 := 
sorry

end proof_y_times_1_minus_g_eq_1_l228_228964


namespace find_parallel_line_l228_228771

/-- 
Given a line l with equation 3x - 2y + 1 = 0 and a point A(1,1).
Find the equation of a line that passes through A and is parallel to l.
-/
theorem find_parallel_line (a b c : ℝ) (p_x p_y : ℝ) 
    (h₁ : 3 * p_x - 2 * p_y + c = 0) 
    (h₂ : p_x = 1 ∧ p_y = 1)
    (h₃ : a = 3 ∧ b = -2) :
    3 * x - 2 * y - 1 = 0 := 
by 
  sorry

end find_parallel_line_l228_228771


namespace color_tromino_l228_228803

theorem color_tromino (n : ℕ) (h_n : n ≥ 3) (colors : Fin n.succ.succ) (grid : Array (Array colors) (colors : Fin ((n + 2) * (n + 2)).succ / 3)) :
  (∀ i j, i < n ∧ j < n → ∃ (c : colors), grid[i][j] = c) →
  (∃ (i j k : Fin n) (c1 c2 c3 : colors), 
    (grid[i.val][j.val] = c1 ∧ grid[(i + 1).val % n][j.val] = c2 ∧ grid[(i + 2).val % n][j.val] = c3 ∧ c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3) ∨
    (grid[i.val][j.val] = c1 ∧ grid[i.val][(j + 1).val % n] = c2 ∧ grid[i.val][(j + 2).val % n] = c3 ∧ c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3)) :=
  sorry

end color_tromino_l228_228803


namespace octagon_area_l228_228669

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228669


namespace no_positive_integer_satisfies_conditions_l228_228029

theorem no_positive_integer_satisfies_conditions :
  ∀ (n : ℕ), (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) → false :=
by {
  intros n h,
  cases h with h₁ h₂,
  cases h₁ with h₁a h₁b,
  cases h₂ with h₂a h₂b,

  -- From h₁a
  have h₁a' : n ≥ 5000, {
    exact Nat.le_of_div_le_of_pos 1000 h₁a (by norm_num),
  },

  -- From h₂b
  have h₂b' : n ≤ 1999, {
    calc
      n ≤ 5 * n / 5 : by linarith [Nat.div_le_self n 5]
      ... ≤ 9999 / 5 : Nat.div_le_div_right h₂b
      ... = 1999   : by norm_num,
  },

  by_contradiction,
  linarith,
}

end no_positive_integer_satisfies_conditions_l228_228029


namespace luke_win_percentage_l228_228984

theorem luke_win_percentage :
  ∀ x : ℕ, (19 + x) / (20 + x) = 96/100 ↔ x = 5 :=
by
  intro x
  split
  . intro h
    sorry
  . intro h
    rw [h]
    norm_num

end luke_win_percentage_l228_228984


namespace range_of_x_plus_y_l228_228057

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 2 * x * y + 4 * y^2 = 1) : 0 < x + y ∧ x + y < 1 :=
by
  sorry

end range_of_x_plus_y_l228_228057


namespace inclination_angle_of_m_l228_228466

theorem inclination_angle_of_m :
  ∀ (l1 l2 : ℝ → ℝ → Prop) (m : ℝ → ℝ → Prop),
  (∀ x y, l1 x y ↔ x - y + 1 = 0) →
  (∀ x y, l2 x y ↔ x - y + 3 = 0) →
  (∃ d : ℝ, 2 * real.sqrt 2 = d) →
  ∃ θ : ℝ, 
  (θ = 75 ∨ θ = 15) :=
by
  intros l1 l2 m hl1 hl2 hd
  sorry

end inclination_angle_of_m_l228_228466


namespace cylinder_radius_l228_228656

theorem cylinder_radius
  (r h : ℝ) (S : ℝ) (h_cylinder : h = 8) (S_surface : S = 130 * Real.pi)
  (surface_area_eq : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 :=
by
  sorry

end cylinder_radius_l228_228656


namespace simplify_trig_expression_l228_228561

def sin50 := Real.sin (50 * Real.pi / 180)
def cos10 := Real.cos (10 * Real.pi / 180)
def tan10 := Real.tan (10 * Real.pi / 180)
def sin30 := Real.sin (30 * Real.pi / 180)
def sin10 := Real.sin (10 * Real.pi / 180)
def cos40 := Real.cos (40 * Real.pi / 180)
def sin40 := Real.sin (40 * Real.pi / 180)
def sin80 := Real.sin (80 * Real.pi / 180)

theorem simplify_trig_expression : sin50 * (1 + Real.sqrt 3 * tan10) = 1 :=
by 
    sorry

end simplify_trig_expression_l228_228561


namespace probability_two_extreme_points_l228_228427

def f (x : ℝ) (a : ℝ) (b : ℝ) := (1 / 3) * x^3 + a * x^2 + b^2 * x + 1

def has_two_extreme_points (a b : ℝ) : Prop :=
  a > b

theorem probability_two_extreme_points : 
  let choices_a := {1, 2, 3}
  let choices_b := {0, 1, 2}
  ∑ a in choices_a, ∑ b in choices_b, if has_two_extreme_points a b then 1 else 0 = 6 →
  (6 / 9 : ℝ) = 2 / 3
:= by
  sorry

end probability_two_extreme_points_l228_228427


namespace area_of_regular_octagon_in_circle_l228_228680

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228680


namespace find_a_n_l228_228070

def a_n : ℕ → ℕ
| 0     := 0
| 1     := 3
| (n+2) := 2^(n+2) - 2^n

theorem find_a_n (n : ℕ) (Sn : ℕ) (h : log 2 (Sn + 1) = n + 1) : 
  a_n (n+1) = if n + 1 = 1 then 3 else 2^(n+1) :=
begin
  sorry
end

end find_a_n_l228_228070


namespace electronic_dogs_distance_l228_228481

def point : Type := ℕ -- points on the vertices of the cube represented by natural numbers

-- Define functions for the positions of the black and yellow "electronic dogs" after n segments
def black_dog_position (n : ℕ) : point := n % 6 -- the black dog returns to the same point every 6 segments
def yellow_dog_position (n : ℕ) : point := (n % 6 + 3) % 6 -- the yellow dog has a phase shift after completing 6 segments

-- Define the distance function on the cube of edge length 1
def distance (p1 p2 : point) : ℕ := if p1 = p2 then 0 else 1

theorem electronic_dogs_distance :
  distance (black_dog_position 2008) (yellow_dog_position 2009) = 1 :=
by {
  -- The positions modulo 6 determine their positions: black at 2008 % 6 = 4, yellow at (2009 % 6 + 3) % 6 = 5
  have h_black : black_dog_position 2008 = 2 := by sorry,
  have h_yellow : yellow_dog_position 2009 = 0 := by sorry,
  rw [h_black, h_yellow],
  -- The distance between the two positions is 1
  exact rfl,
}

end electronic_dogs_distance_l228_228481


namespace trajectory_of_M_l228_228995

theorem trajectory_of_M {x y x₀ y₀ : ℝ} (P_on_parabola : x₀^2 = 2 * y₀)
(line_PQ_perpendicular : ∀ Q : ℝ, true)
(vector_PM_PQ_relation : x₀ = x ∧ y₀ = 2 * y) :
  x^2 = 4 * y := by
  sorry

end trajectory_of_M_l228_228995


namespace pine_saplings_in_sample_l228_228657

-- Definitions based on conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Main theorem to prove
theorem pine_saplings_in_sample : (pine_saplings * sample_size) / total_saplings = 20 :=
by sorry

end pine_saplings_in_sample_l228_228657


namespace correct_option_D_l228_228630

theorem correct_option_D 
  (A : (sqrt 36 = 6 ∨ sqrt 36 = -6) → False)
  (B : sqrt ((-3 : ℤ)^2) = 3)
  (C : (-4 : ℤ) < 0 → (-sqrt (-4 : ℤ) = 2) → False)
  (D : (cbrt (-8 : ℤ) = -2)) : 
  D := by 
  sorry

end correct_option_D_l228_228630


namespace incircles_touch_other_diagonal_l228_228458

variables {A B C D P Q : Type} [IncirclesTouch A B C D]
-- Assume the quadrilateral ABCD and its diagonals AC and BD, and points P and Q on AC where the incircles touch.

theorem incircles_touch_other_diagonal (A B C D : Type) :
  IncirclesTouch (triangle A B C) (triangle A C D) → IncirclesTouch (triangle A B D) (triangle B C D) :=
sorry

end incircles_touch_other_diagonal_l228_228458


namespace cathy_final_position_l228_228753

def start_position : ℕ × ℕ := (2, -3)

def moves : List ℕ := List.range' 2 2 24

noncomputable def calculate_position (initial: ℕ × ℕ) (moves: List ℕ) : ℕ × ℕ :=
  sorry  -- The function to calculate the final position based on the moves

noncomputable def total_distance (moves: List ℕ) : ℕ :=
  List.sum moves

theorem cathy_final_position :
  calculate_position start_position moves = (-10, -15) ∧
  total_distance moves = 146 :=
sorry

end cathy_final_position_l228_228753


namespace total_legs_on_farm_l228_228242

-- Define the number of each type of animal
def num_ducks : Nat := 6
def num_dogs : Nat := 5
def num_spiders : Nat := 3
def num_three_legged_dogs : Nat := 1

-- Define the number of legs for each type of animal
def legs_per_duck : Nat := 2
def legs_per_dog : Nat := 4
def legs_per_spider : Nat := 8
def legs_per_three_legged_dog : Nat := 3

-- Calculate the total number of legs
def total_duck_legs : Nat := num_ducks * legs_per_duck
def total_dog_legs : Nat := (num_dogs * legs_per_dog) - (num_three_legged_dogs * (legs_per_dog - legs_per_three_legged_dog))
def total_spider_legs : Nat := num_spiders * legs_per_spider

-- The total number of legs on the farm
def total_animal_legs : Nat := total_duck_legs + total_dog_legs + total_spider_legs

-- State the theorem to be proved
theorem total_legs_on_farm : total_animal_legs = 55 :=
by
  -- Assuming conditions and computing as per them
  sorry

end total_legs_on_farm_l228_228242


namespace perfect_square_solution_l228_228016

theorem perfect_square_solution (m n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (∃ k : ℕ, (5 ^ m + 2 ^ n * p) / (5 ^ m - 2 ^ n * p) = k ^ 2)
  ↔ (m = 1 ∧ n = 1 ∧ p = 2 ∨ m = 3 ∧ n = 2 ∧ p = 3 ∨ m = 2 ∧ n = 2 ∧ p = 5) :=
by
  sorry

end perfect_square_solution_l228_228016


namespace f2_is_isomorphic_to_f4_l228_228465

def is_isomorphic (f g : ℝ → ℝ) : Prop := 
  ∃ (a b : ℝ), ∀ x, g(x) = f(x + a) + b

def f1 : ℝ → ℝ := λ x, 2 * log 2 x
def f2 : ℝ → ℝ := λ x, log 2 (x + 2)
def f3 : ℝ → ℝ := λ x, (log 2 x) ^ 2
def f4 : ℝ → ℝ := λ x, log 2 (2 * x)

theorem f2_is_isomorphic_to_f4 : is_isomorphic f2 f4 := sorry

end f2_is_isomorphic_to_f4_l228_228465


namespace final_direction_is_west_l228_228494

def initial_direction := "south"
def clockwise_moves := 7/2
def counterclockwise_moves := 25/4

def net_movement := clockwise_moves - counterclockwise_moves
def net_clockwise_movement := λ (rev : ℚ), (rev % 1 + 1) % 1

theorem final_direction_is_west :
  net_clockwise_movement net_movement = 1/4 →
  initial_direction = "south" →
  "west" = "west" := 
by
  intros
  unfold net_movement at *
  unfold net_clockwise_movement at *
  sorry

end final_direction_is_west_l228_228494


namespace problem_statement_l228_228060

variable (x y : ℝ)

theorem problem_statement
  (h1 : 4 * x + y = 9)
  (h2 : x + 4 * y = 16) :
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 :=
sorry

end problem_statement_l228_228060


namespace find_d_for_tangency_l228_228372

theorem find_d_for_tangency:
  ∃ d: ℝ, (∀ x y: ℝ, y = 3 * x + d ∧ y^2 = 12 * x → (y - 2)^2 = 0) → d = 1 :=
begin
  sorry
end


end find_d_for_tangency_l228_228372


namespace balls_probability_l228_228650

theorem balls_probability :
  let total_ways := Nat.choose 24 4
  let ways_bw := Nat.choose 10 2 * Nat.choose 8 2
  let ways_br := Nat.choose 10 2 * Nat.choose 6 2
  let ways_wr := Nat.choose 8 2 * Nat.choose 6 2
  let target_ways := ways_bw + ways_br + ways_wr
  (target_ways : ℚ) / total_ways = 157 / 845 := by
  sorry

end balls_probability_l228_228650


namespace product_of_roots_l228_228023

theorem product_of_roots : 
  let p1 := 3 * x^4 + 2 * x^3 - 7 * x + 30
  let p2 := 4 * x^3 - 16 * x^2 + 21
  (∀ x, p1 * p2 = 0).root_product = -52.5 :=
sorry

end product_of_roots_l228_228023


namespace volume_of_pyramid_correct_l228_228554

noncomputable def volume_of_pyramid {A B C D P : Type} 
  (square_base : Π (A B C D : Type), Bool) 
  (equilateral_triangle : Π (P A B : Type), Bool) 
  (side_length : ℝ) 
  (PO_length : ℝ) : ℝ :=
  if (square_base A B C D) 
    && (equilateral_triangle P A B) 
    && (side_length = 10)
    && (PO_length = 5 * Real.sqrt 3) then
    (1 / 3) * (side_length ^ 2) * PO_length
  else 0

theorem volume_of_pyramid_correct :
  volume_of_pyramid (λ A B C D, True) (λ P A B, True) 10 (5 * Real.sqrt 3) = (500 * Real.sqrt 3) / 3 :=
sorry

end volume_of_pyramid_correct_l228_228554


namespace turtles_remaining_proof_l228_228293

noncomputable def turtles_original := 50
noncomputable def turtles_additional := 7 * turtles_original - 6
noncomputable def turtles_total_before_frightened := turtles_original + turtles_additional
noncomputable def turtles_frightened := (3 / 7) * turtles_total_before_frightened
noncomputable def turtles_remaining := turtles_total_before_frightened - turtles_frightened

theorem turtles_remaining_proof : turtles_remaining = 226 := by
  sorry

end turtles_remaining_proof_l228_228293


namespace daria_still_owes_l228_228756

-- Definitions of the given conditions
def saved_amount : ℝ := 500
def couch_cost : ℝ := 750
def table_cost : ℝ := 100
def lamp_cost : ℝ := 50

-- Calculation of total cost of the furniture
def total_cost : ℝ := couch_cost + table_cost + lamp_cost

-- Calculation of the remaining amount owed
def remaining_owed : ℝ := total_cost - saved_amount

-- Proof statement that Daria still owes $400 before interest
theorem daria_still_owes : remaining_owed = 400 := by
  -- Skipping the proof
  sorry

end daria_still_owes_l228_228756


namespace z2_in_fourth_quadrant_solve_m_and_n_l228_228376

variables (m n : ℝ)
def z1 := complex.mk m 1 -- z1 = m + i
def z2 := complex.mk m (m - 2) -- z2 = m + (m - 2)i

-- Proof problem 1: Determine the range of values for m if z2 is in the fourth quadrant
theorem z2_in_fourth_quadrant (h1: z2.re > 0) (h2: z2.im < 0) : 0 < m ∧ m < 2 :=
sorry

-- Proof problem 2: Find real numbers m and n such that z2 = z1 * (n : ℂ) * complex.I
theorem solve_m_and_n (h : z2 = z1 * complex.mk 0 n) : (m = 1 ∧ n = -1) ∨ (m = -2 ∧ n = 2) :=
sorry

end z2_in_fourth_quadrant_solve_m_and_n_l228_228376


namespace simplify_fraction_l228_228559

theorem simplify_fraction (x : ℤ) :
  (⟦(2 * x - 3, 4)⟧ + ⟦(5 * x + 2, 5)⟧ : ℚ) = ⟦(30 * x - 7, 20)⟧ := 
by
  sorry

end simplify_fraction_l228_228559


namespace no_positive_integer_pairs_exist_l228_228498

theorem no_positive_integer_pairs_exist :
  ∀ (d j n a b : ℕ), j = 30 → d = 35 → (∀ n : ℕ, n > 0 → 
  (30 + n = 10 * a + b) ∧ (35 + n = 10 * b + a) → b > a) → False :=
begin
  intros d j n a b h_j h_d h_n,
  sorry,
end

end no_positive_integer_pairs_exist_l228_228498


namespace tan_two_beta_l228_228066

variables {α β : Real}

theorem tan_two_beta (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 7) : Real.tan (2 * β) = -3 / 4 :=
by
  sorry

end tan_two_beta_l228_228066


namespace area_of_inscribed_octagon_l228_228712

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228712


namespace identify_negative_number_l228_228580

theorem identify_negative_number :
  ∃ n, n = -1 ∧ n < 0 ∧ (∀ m, m ∈ {-1, 0, 1, 2} → ((m = n) ∨ (m ≥ 0))) :=
by
  sorry

end identify_negative_number_l228_228580


namespace range_of_y_l228_228431

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 2) * (2 * (Real.log x / (2 * Real.log 2)) - 4)

theorem range_of_y :
  (1 ≤ x ∧ x ≤ 8) →
  (∀ t : ℝ, t = Real.log x / Real.log 2 → y x = t^2 - 2 * t - 8 ∧ 0 ≤ t ∧ t ≤ 3) →
  ∃ ymin ymax, (ymin ≤ y x ∧ y x ≤ ymax) ∧ ymin = -9 ∧ ymax = -5 :=
by
  sorry

end range_of_y_l228_228431


namespace area_of_circle_l228_228758

open Real

theorem area_of_circle :
  ∃ (A : ℝ), (∀ x y : ℝ, (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) → A = 16 * π) :=
sorry

end area_of_circle_l228_228758


namespace find_cos_beta_l228_228063

-- Define the problem environment and hypothesis.
variable (α β : ℝ)
variable (h0 : 0 < α ∧ α < π / 2) (h1 : 0 < β ∧ β < π / 2)
variable (h2 : cos α = 1 / 7)
variable (h3 : sin (α + β) = 5 * real.sqrt 3 / 14)

-- Define the theorem that states the desired result.
theorem find_cos_beta : cos β = 1 / 2 := 
by
  sorry

end find_cos_beta_l228_228063


namespace isosceles_right_triangle_xy_l228_228170

noncomputable theory
open_locale classical

variables (A B C D E X Y : ℝ)
variables (AD AE : ℝ)
variables (DX EY XY : ℝ)

def is_right_triangle (A B C : ℝ) : Prop :=
  -- Definition for an isosceles right triangle
  ∠A = 90 ∧ AB = AC

def foot_of_altitude (line1 line2 point : ℝ) : ℝ :=
  -- Definition for the feet of the altitudes
  sorry

theorem isosceles_right_triangle_xy
  (h_tri : is_right_triangle A B C)
  (h_AD : AD = 48 * real.sqrt 2)
  (h_AE : AE = 52 * real.sqrt 2)
  (h_D : D = foot_of_altitude A B AD)
  (h_E : E = foot_of_altitude A C AE)
  (h_X : X = foot_of_altitude D B 0)
  (h_Y : Y = foot_of_altitude E C 0) :
  XY = 100 :=
sorry

end isosceles_right_triangle_xy_l228_228170


namespace problem_statement_l228_228830

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + x^2 + b * x
noncomputable def f_prime (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * x + b
noncomputable def g (a b x : ℝ) : ℝ := f(a, b, x) + f_prime(a, b, x)

/-- Given the function f(x) = -1/3 * x^3 + x^2 and g(x) = f(x) + f'(x) is an odd function.
  (I) Show that f(x) = -1/3 * x^3 + x^2.
  (II) Discuss the monotonicity of g(x) and find the maximum and minimum values of g(x) on [1, 2]. 
-/
theorem problem_statement (a b : ℝ) (h1 : a = -1/3) (h2 : b = 0) :
  (∀ x : ℝ, f(a, b, x) = -1/3 * x^3 + x^2) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g(a, b, x) ≤ g(a, b, sqrt(2)) ∧ g(a, b, x) ≥ g(a, b, 2)) :=
sorry

end problem_statement_l228_228830


namespace sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l228_228668

theorem sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog
  (a r : ℝ)
  (volume_cond : a^3 * r^3 = 288)
  (surface_area_cond : 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r) = 288)
  (geom_prog : True) :
  4 * (a * r^2 + a * r + a) = 92 := 
sorry

end sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l228_228668


namespace rower_rate_in_still_water_l228_228636

theorem rower_rate_in_still_water (V_m V_s : ℝ) (h1 : V_m + V_s = 16) (h2 : V_m - V_s = 12) : V_m = 14 := 
sorry

end rower_rate_in_still_water_l228_228636


namespace sector_area_correct_l228_228277

noncomputable def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
(θ / 360) * π * r^2

theorem sector_area_correct :
  area_of_sector 12 36 = 45.24 := sorry

end sector_area_correct_l228_228277


namespace rosa_total_pages_called_l228_228100

variable (P_last P_this : ℝ)

theorem rosa_total_pages_called (h1 : P_last = 10.2) (h2 : P_this = 8.6) : P_last + P_this = 18.8 :=
by sorry

end rosa_total_pages_called_l228_228100


namespace find_base_l228_228076

noncomputable def f (a x : ℝ) := 1 + (Real.log x) / (Real.log a)

theorem find_base (a : ℝ) (hinv_pass : (∀ y : ℝ, (∀ x : ℝ, f a x = y → x = 4 → y = 3))) : a = 2 :=
by
  sorry

end find_base_l228_228076


namespace arithmetic_sequence_a7_l228_228125

theorem arithmetic_sequence_a7 (a : ℕ → ℕ) (d a1 : ℕ)
  (h1 : ∑ i in finset.range 5, a1 + i * d = 25)
  (h2 : a1 + d = 3) :
  a 7 = 13 := 
by
  sorry

end arithmetic_sequence_a7_l228_228125


namespace infinite_subsets_exists_divisor_l228_228950

-- Definition of the set M
def M : Set ℕ := { n | ∃ a b : ℕ, n = 2^a * 3^b }

-- Infinite family of subsets of M
variable (A : ℕ → Set ℕ)
variables (inf_family : ∀ i, A i ⊆ M)

-- Theorem statement
theorem infinite_subsets_exists_divisor :
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x := by
  sorry

end infinite_subsets_exists_divisor_l228_228950


namespace zeroes_at_end_base_8_of_factorial_15_l228_228884

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l228_228884


namespace area_of_shaded_rectangle_l228_228305

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l228_228305


namespace find_m_l228_228556

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3 * m - 1}
def B (m : ℝ) : Set ℝ := {2 * m, -3}
def C : Set ℝ := {-3}

theorem find_m (m : ℝ) : A m ∩ B m = C → m = 1 :=
by 
  intros h
  sorry

end find_m_l228_228556


namespace T_or_U_closed_mul_l228_228171

open Set

variable (S : Set ℝ) (T U : Set ℝ)
variables [H1 : ∀ a b, a ∈ S → b ∈ S → a * b ∈ S]
variables [H2 : T ∩ U = ∅] [H3 : T ∪ U = S]
variables [H4 : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T]
variables [H5 : ∀ a b c, a ∈ U → b ∈ U → c ∈ U → a * b * c ∈ U]

theorem T_or_U_closed_mul : (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ a b, a ∈ U → b ∈ U → a * b ∈ U) :=
sorry

end T_or_U_closed_mul_l228_228171


namespace number_of_trailing_zeroes_base8_l228_228862

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228862


namespace trigonometric_identity_l228_228821

theorem trigonometric_identity 
  (θ : ℝ)
  (h1 : θ ∈ Icc π (3 * π / 2))  -- this encodes that θ is in the third quadrant
  (h2 : tan θ ^ 2 = -2 * sqrt 2) : 
  sin θ ^ 2 - sin (3 * π + θ) * cos (π + θ) - sqrt 2 * cos θ ^ 2 = (2 - 2 * sqrt 2) / 3 :=
by
  sorry

end trigonometric_identity_l228_228821


namespace minimum_abs_phi_l228_228464

theorem minimum_abs_phi (φ : ℝ) (k : ℤ) : 
  (∃ k : ℤ, φ = k * π - 13 * π / 6) → min (|φ|) = π / 6 :=
by
  sorry

end minimum_abs_phi_l228_228464


namespace circle_line_proof_l228_228820

-- Definition to represent the polar coordinate equation of the circle
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ - 6 * Real.sin θ

-- Definition to transform polar to rectangular coordinates
def rectangular_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 13

-- Definition of the parametric equation of the line
def parametric_line (x y t θ : ℝ) : Prop := x = 4 + t * Real.cos θ ∧ y = t * Real.sin θ

-- Definition representing the intersection condition |PQ| = 4
def intersects_at_two_points (|PQ| : ℝ) : Prop := |PQ| = 4

-- Main statement (proof problem to be proved)
theorem circle_line_proof (ρ θ x y t |PQ| : ℝ) :
  polar_equation ρ θ ∧ parametric_line x y t θ ∧ intersects_at_two_points |PQ| →
  rectangular_equation x y ∧ (Real.tan θ = 0 ∨ Real.tan θ = -12 / 5) :=
by
  sorry

end circle_line_proof_l228_228820


namespace problem_1_problem_2_l228_228423

def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 2 = 1

def right_focus : ℝ × ℝ := (2 * Real.sqrt 2, 0)

def line_through_focus (k x : ℝ) (h : k ≠ 0) : ℝ := k * (x - 2)

noncomputable def intersect_ellipse_with_line (k : ℝ) (h : k ≠ 0) : set (ℝ × ℝ) :=
  {p | ∃ x, (x, line_through_focus k x h) = p ∧ ellipse_eq x (line_through_focus k x h)}

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def intersect_line_with_x3 (k : ℝ) (h : k ≠ 0) : ℝ × ℝ :=
  let N := midpoint (classical.some (intersect_ellipse_with_line k h)) (classical.some (exists_mem_of_nonempty (intersect_ellipse_with_line k h))) in
  (3, (N.2 / N.1) * 3)

theorem problem_1 (k : ℝ) (h : k ≠ 0) :
  let M := intersect_line_with_x3 k h in
  let F := right_focus in
  let Q := classical.some (intersect_ellipse_with_line k h) in
  ∡ F M Q = Real.pi / 2 := 
sorry

theorem problem_2 (k : ℝ) (h : k ≠ 0) :
  let PQ := Real.dist (classical.some (intersect_ellipse_with_line k h)) (classical.some (exists_mem_of_nonempty (intersect_ellipse_with_line k h))) in
  let MF := Real.dist (intersect_line_with_x3 k h) right_focus in 
  PQ / MF ≤ Real.sqrt 3 :=
sorry

end problem_1_problem_2_l228_228423


namespace alex_weekly_water_bill_l228_228727

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def remaining_income : ℝ := 345

theorem alex_weekly_water_bill : ∃ (water_bill : ℝ), 
  let income_after_tax := weekly_income * (1 - tax_rate) in
  let income_after_tithe := income_after_tax - (weekly_income * tithe_rate) in
  remaining_income = income_after_tithe - water_bill ∧ water_bill = 55 :=
by
  sorry

end alex_weekly_water_bill_l228_228727


namespace unit_digit_of_power_l228_228624

theorem unit_digit_of_power (n k : ℕ) : 
  let u := n % 10 in 
  u = 7 →
  (∃ m, k = 4 * m + 1) →
  (n ^ k) % 10 = 7 :=
by
  intros
  sorry

end unit_digit_of_power_l228_228624


namespace remaining_pencils_l228_228323

-- Define the initial conditions
def initial_pencils : Float := 56.0
def pencils_given : Float := 9.0

-- Formulate the theorem stating that the remaining pencils = 47.0
theorem remaining_pencils : initial_pencils - pencils_given = 47.0 := by
  sorry

end remaining_pencils_l228_228323


namespace find_a_value_l228_228928

-- Define the conditions for the problem
variables {a : ℝ} (ha : a > 0)
variables {P : ℝ × ℝ} (P_def : P = (-2, -4))
variables {l : ℝ → ℝ × ℝ}
variables (param_eq : ∀ t, l t = (-2 + (real.sqrt 2 / 2) * t, -4 + (real.sqrt 2 / 2) * t))
variables {curve_C : ℝ × ℝ → Prop}
variables (polar_eq : ∀ (θ : ℝ), curve_C (θ.cos, θ.sin) ↔ (θ.cos ^ 2) * (θ.sin ^ 2) = 2 * a * θ.cos)
variables {A B : ℝ × ℝ}
variables {PA PB AB : ℝ}
variables (PA_eq : PA = dist P A) (PB_eq : PB = dist P B) (AB_eq : AB = dist A B)
variables (intersect_cond : curve_C A ∧ curve_C B ∧ ∃ t₁ t₂, A = l t₁ ∧ B = l t₂)

-- main theorem statement
theorem find_a_value
  (curve_eq : ∀ x y, curve_C (x, y) ↔ y ^ 2 = 2 * a * x)
  (line_eq : ∀ x, ∃ y, l x = (x, y) ∧ y = x - 2)
  (intersection_condition : (∀ t₁ t₂, (PA * PB = AB ^ 2) ↔ (t₁ * t₂ = (t₁ - t₂) ^ 2)))
  (intersection_distance : PA * PB = AB ^ 2) :
  a = 1 :=
sorry

end find_a_value_l228_228928


namespace max_set_correct_transformation_correct_l228_228433

-- Define the function
def f (x : ℝ) : ℝ := sin (x / 2) + sqrt 3 * cos (x / 2)

-- Define the set of x values where f is at its maximum
def max_set : Set ℝ := {x : ℝ | ∃ k : ℤ, x = 4 * k * Real.pi + Real.pi / 3}

-- Define the transformed function
def g (x : ℝ) : ℝ := 2 * sin (x / 2 + Real.pi / 3)

-- The target function for transformation
def h (x : ℝ) : ℝ := sin x

-- Proof statement
theorem max_set_correct : { x : ℝ | ∃ k : ℤ, x = 4 * k * Real.pi + Real.pi / 3 } = max_set := 
sorry

theorem transformation_correct : 
  (∀ x : ℝ, g (x - 2 * Real.pi / 3) = 2 * sin (x / 2)) ∧
  (∀ x : ℝ, sin (2 * x) = 2 * sin x) ∧
  (∀ x : ℝ, (2 * sin x) / 2 = h x) :=
sorry

end max_set_correct_transformation_correct_l228_228433


namespace triangle_area_l228_228315

theorem triangle_area {r : ℝ} (h_r : r = 6) {x : ℝ} 
  (h1 : 5 * x = 2 * r)
  (h2 : x = 12 / 5) : 
  (1 / 2 * (3 * x) * (4 * x) = 34.56) :=
by
  sorry

end triangle_area_l228_228315


namespace prove_a_in_S_l228_228089

open Set

variable {α : Type*} [DecidableEq α]

def S : Set α := {1, 2}
def T (a : α) : Set α := {a}

theorem prove_a_in_S (a : α) (h : S ∪ T a = S) : a ∈ S := by
  have h_sub : T a ⊆ S := by
    rw [union_eq_self_of_subset_left h]
  obtain ⟨a_in_S⟩ : a ∈ S := mem_singleton_iff.mp $ h_sub trivial
  exact a_in_S

end prove_a_in_S_l228_228089


namespace fixed_point_l228_228226

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 1) + 1

theorem fixed_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 2 = 1 := 
by
  sorry

end fixed_point_l228_228226


namespace max_balls_in_cubic_container_l228_228252

def volume_cube_side_length (s : ℝ) := s^3

def volume_sphere_radius (r : ℝ) := (4 / 3) * Real.pi * r^3

def max_balls_in_cube (V_cube V_ball : ℝ) := ⌊V_cube / V_ball⌋

theorem max_balls_in_cubic_container :
  max_balls_in_cube (volume_cube_side_length 4) (volume_sphere_radius 1) = 16 :=
by
  sorry

end max_balls_in_cubic_container_l228_228252


namespace solve_total_rainfall_l228_228128

def rainfall_2010 : ℝ := 50.0
def increase_2011 : ℝ := 3.0
def increase_2012 : ℝ := 4.0

def monthly_rainfall_2011 : ℝ := rainfall_2010 + increase_2011
def monthly_rainfall_2012 : ℝ := monthly_rainfall_2011 + increase_2012

def total_rainfall_2011 : ℝ := monthly_rainfall_2011 * 12
def total_rainfall_2012 : ℝ := monthly_rainfall_2012 * 12

def total_rainfall_2011_2012 : ℝ := total_rainfall_2011 + total_rainfall_2012

theorem solve_total_rainfall :
  total_rainfall_2011_2012 = 1320.0 :=
sorry

end solve_total_rainfall_l228_228128


namespace seamless_assembly_with_equilateral_triangle_l228_228413

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end seamless_assembly_with_equilateral_triangle_l228_228413


namespace area_of_BCD_l228_228955

variables {a b c x y z : ℝ}
variables (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * (2 * c)) (h3 : z = (1 / 2) * a * (2 * c))

theorem area_of_BCD (a b c : ℝ) (x y z : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * (2 * c)) (h3 : z = (1 / 2) * a * (2 * c)):
  sqrt (x^2 + y^2 + z^2) = (sqrt x^2 + y^2 + z^2)  :=
sorry

end area_of_BCD_l228_228955


namespace num_solutions_non_negative_reals_l228_228770

-- Define the system of equations as a function to express the cyclic nature
def system_of_equations (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1 % n) + (x (if k = 0 then n else k) ^ 2) = 4 * x (if k = 0 then n else k)

-- Define the main theorem stating the number of solutions
theorem num_solutions_non_negative_reals {n : ℕ} (hn : 0 < n) : 
  ∃ (s : Finset (ℕ → ℝ)), (∀ x ∈ s, ∀ k, 0 ≤ (x k) ∧ system_of_equations n x k) ∧ s.card = 2^n :=
sorry

end num_solutions_non_negative_reals_l228_228770


namespace area_of_regular_octagon_in_circle_l228_228681

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l228_228681


namespace range_of_f_l228_228078

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

theorem range_of_f : set.Icc (-3 / 2) 3 = set.range (f : set.Icc 0 (π / 2) → ℝ) :=
by
  sorry

end range_of_f_l228_228078


namespace problem_l228_228434

def F (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a^4 + b^3 + c^2 + d

theorem problem :
  (∑ i in Finset.range 20, (-1)^i * F (2019 - i)) = -1 :=
by
  sorry

end problem_l228_228434


namespace number_of_paths_correct_l228_228350

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def initial_point : Point := ⟨0, 0⟩
def final_point : Point := ⟨6, 6⟩
def step_size : ℤ := 1
def max_coordinate : ℤ := 6

def within_bounds (pt : Point) : Prop :=
  abs pt.x ≤ max_coordinate ∧ abs pt.y ≤ max_coordinate

def start_facing_rightward : Prop := True -- Daisy starts facing rightward.

def valid_turns : List Point :=
  [⟨1, 0⟩, ⟨0, 1⟩, ⟨-1, 0⟩, ⟨0, -1⟩] -- Right, Up, Left, Down

def steps_from (p1 p2 : Point) : Bool :=
  valid_turns.any (λ t => p2 = ⟨p1.x + t.x, p1.y + t.y⟩)

def no_repeat_points (path : List Point) : Prop :=
  ∀ (p ∈ path), path.count p = 1

noncomputable def num_paths : ℕ := 131922

theorem number_of_paths_correct :
  ∃ (paths : List (List Point)), 
  ∀ path ∈ paths,
    -- The path must start at (0,0) and end at (6,6)
    path.head = initial_point ∧ 
    path.last = final_point ∧ 
    -- Each step must be to a valid point
    ∀ (i : ℕ), i < path.length - 1 → steps_from (path.nth_le i sorry) (path.nth_le (i + 1) sorry) ∧ 
    -- The path must stay within the defined bounds
    within_bounds (path.nth_le i sorry) ∧ 
    within_bounds (path.nth_le (i + 1) sorry) ∧ 
    -- The path cannot revisit a point
    no_repeat_points path ∧ 
  -- The number of such paths must be:
  paths.length = num_paths := 
sorry

end number_of_paths_correct_l228_228350


namespace only_negative_is_neg1_l228_228577

theorem only_negative_is_neg1 (a b c d : Int) (h_a : a = -1) (h_b : b = 0) (h_c : c = 1) (h_d : d = 2) :
  (∀ x ∈ {a, b, c, d}, x < 0 ↔ x = a) :=
by
  sorry

end only_negative_is_neg1_l228_228577


namespace building_floors_count_l228_228199

theorem building_floors_count
  (N : ℕ) -- total number of floors in the building
  (k : ℕ) -- the floor Shura lives on
  (h1 : k < 6)
  (h2 : 2 * N - 6 - k = 1.5 * (6 - k)) :
  N = 7 :=
by {
  -- This is where the proof would start, but it's omitted as requested.
  sorry
}

end building_floors_count_l228_228199


namespace find_p_at_0_l228_228183

-- Given definitions and conditions
def p (x : ℝ) : ℝ := sorry
axiom poly_deg_5 : ∃ p : ℝ → ℝ, degree p = 5
axiom p_conditions : ∀ (n : ℕ), n ≤ 5 → p (3 ^ n) = 1 / (3 ^ n)

-- Statement to prove
theorem find_p_at_0 : p 0 = 0 :=
by
  sorry

end find_p_at_0_l228_228183


namespace total_cost_of_ads_l228_228352

theorem total_cost_of_ads : 
  let ad1_minute := 2 in
  let ad2_minute := 2 in
  let ad3_minute := 3 in
  let ad4_minute := 3 in
  let ad5_minute := 5 in
  let cost1_per_minute := 4000 in
  let cost2_per_minute := 4000 in
  let cost3_per_minute := 5000 in
  let cost4_per_minute := 5000 in
  let cost5_per_minute := 6000 in
  (ad1_minute * cost1_per_minute) +
  (ad2_minute * cost2_per_minute) +
  (ad3_minute * cost3_per_minute) +
  (ad4_minute * cost4_per_minute) +
  (ad5_minute * cost5_per_minute) = 76000 :=
by
  -- Proof skipped
  sorry

end total_cost_of_ads_l228_228352


namespace graph_of_y_eq_neg2x_passes_quadrant_II_IV_l228_228227

-- Definitions
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

def is_in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main statement
theorem graph_of_y_eq_neg2x_passes_quadrant_II_IV :
  ∀ (x : ℝ), (is_in_quadrant_II x (linear_function (-2) x) ∨ 
               is_in_quadrant_IV x (linear_function (-2) x)) :=
by
  sorry

end graph_of_y_eq_neg2x_passes_quadrant_II_IV_l228_228227


namespace find_white_towels_l228_228530

variable {W : ℕ} -- Define W as a natural number

-- Define the conditions as Lean definitions
def initial_towel_count (W : ℕ) : ℕ := 35 + W
def remaining_towel_count (W : ℕ) : ℕ := initial_towel_count W - 34

-- Theorem statement: Proving that W = 21 given the conditions
theorem find_white_towels (h : remaining_towel_count W = 22) : W = 21 :=
by
  sorry

end find_white_towels_l228_228530


namespace prove_trig_identity_l228_228802

noncomputable def α : ℝ := sorry  -- We need to define α, but its value is inferred from the point (1, 3) on the terminal side

theorem prove_trig_identity {α : ℝ} (h : tan(α) = 3) : 
  (sin (π - α)) / (sin (3 * π / 2 + α)) = -3 :=
sorry

end prove_trig_identity_l228_228802


namespace probability_at_least_one_one_l228_228249

-- Define a Lean statement for the given problem and its proof.
theorem probability_at_least_one_one {Ω : Type} [fintype Ω] (D1 D2 : Ω) (dice1 dice2 : fin 8 → Ω)
  (h1 : ∀ (x : fin 8), (∃ (y : fin 8), dice1 y = x))
  (h2 : ∀ (x : fin 8), (∃ (y : fin 8), dice2 y = x)) :
  (fintype.card {x : fin 8 × fin 8 | (x.fst = 0 ∨ x.snd = 0) }) / (fintype.card Ω * fintype.card Ω) = 15/64 := 
sorry

end probability_at_least_one_one_l228_228249


namespace ratio_of_luxury_to_suv_l228_228651

variable (E L S : Nat)

-- Conditions
def condition1 := E * 2 = L * 3
def condition2 := E * 1 = S * 4

-- The statement to prove
theorem ratio_of_luxury_to_suv 
  (h1 : condition1 E L)
  (h2 : condition2 E S) :
  L * 3 = S * 8 :=
by sorry

end ratio_of_luxury_to_suv_l228_228651


namespace oil_leak_during_work_l228_228736

/-- 
  Given two amounts of oil leakage, one before engineering work started 
  and the total leakage, this theorem calculates the leakage during the work.
-/
def leakage_while_working (initial_leak total_leak : ℕ) : ℕ :=
  total_leak - initial_leak

theorem oil_leak_during_work (initial_leak total_leak expected_leak : ℕ) 
  (h1 : initial_leak = 6522)
  (h2 : total_leak = 11687)
  (h3 : expected_leak = 5165) : 
  leakage_while_work initial_leak total_leak = expected_leak :=
by 
  rw [h1, h2, h3]
  sorry

end oil_leak_during_work_l228_228736


namespace scalene_triangle_angle_bisectors_inequality_l228_228551

theorem scalene_triangle_angle_bisectors_inequality
  (a b c l1 l2 S: ℝ)
  (h_scalene: a > b ∧ b > c ∧ c > 0)
  (h_l1: l1 = max (angle_bisector_length a b c 2 * atan 1) (angle_bisector_length b c a 2 * sin (angle b c a)))
  (h_l2: l2 = min (angle_bisector_length a b c 2 * atan 1) (angle_bisector_length b c a 2 * sin (angle b c a)))
  (h_S: S = 1 / 2 * b * c * sin (2 * atan 1))
  : l1^2 > sqrt 3 * S ∧ sqrt 3 * S > l2^2 :=
by
  sorry

end scalene_triangle_angle_bisectors_inequality_l228_228551


namespace stability_equilibrium_point_l228_228157

noncomputable section

-- Define variables and system of differential equations
variables (x y : ℝ) (dx dt dy : ℝ)
def system_eq1 : Prop := dx = y
def system_eq2 : Prop := dy = -x

-- Define the Lyapunov function
def V (x y : ℝ) : ℝ := x^2 + y^2

-- Define the time derivative of the Lyapunov function
def dV_dt (x y dx dy : ℝ) : ℝ :=
  (2 * x * dx) + (2 * y * dy)

-- Problem statement
theorem stability_equilibrium_point :
  ∀ (x y dx dy : ℝ),
  system_eq1 x y dx →
  system_eq2 x y dy →
  (dV_dt x y dx dy = 0 ∧ V x y > 0 ∧ x = 0 ∧ y = 0) = 
  (stable_not_asymptotically_stable) :=
by
  sorry

end stability_equilibrium_point_l228_228157


namespace smallest_n_solution_unique_l228_228276

theorem smallest_n_solution_unique (a b c d : ℤ) (h : a^2 + b^2 + c^2 = 4 * d^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end smallest_n_solution_unique_l228_228276


namespace who_is_who_l228_228265

-- Defining the structure and terms
structure Brother :=
  (name : String)
  (has_purple_card : Bool)

-- Conditions
def first_brother := Brother.mk "Tralalya" true
def second_brother := Brother.mk "Trulalya" false

/-- Proof that the names and cards of the brothers are as stated. -/
theorem who_is_who :
  ((first_brother.name = "Tralalya" ∧ first_brother.has_purple_card = false) ∧
   (second_brother.name = "Trulalya" ∧ second_brother.has_purple_card = true)) :=
by sorry

end who_is_who_l228_228265


namespace chord_segments_division_l228_228635

-- Definitions based on the conditions
variables (R OM : ℝ) (AB : ℝ)
-- Setting the values as the problem provides 
def radius : ℝ := 15
def distance_from_center : ℝ := 13
def chord_length : ℝ := 18

-- Formulate the problem statement as a theorem
theorem chord_segments_division :
  ∃ (AM MB : ℝ), AM = 14 ∧ MB = 4 :=
by
  let CB := chord_length / 2
  let OC := Real.sqrt (radius^2 - CB^2)
  let MC := Real.sqrt (distance_from_center^2 - OC^2)
  let AM := CB + MC
  let MB := CB - MC
  use AM, MB
  sorry

end chord_segments_division_l228_228635


namespace abs_diff_eq_l228_228217

noncomputable def AM := 100 * a + 10 * b + c
noncomputable def GM := 100 * c + 10 * b + a

theorem abs_diff_eq (x y a b c: ℕ) (hx : x ≠ y) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9)
  (hAM : (x + y) / 2 = AM) (hGM : Real.sqrt (x * y) = GM) :
  | x - y | = 6 * Real.sqrt 1111 :=
sorry

end abs_diff_eq_l228_228217


namespace octagon_area_correct_l228_228694

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228694


namespace bug_visits_all_vertices_l228_228287

noncomputable def tetrahedron_prob : ℚ :=
  let vertices := {A, B, C, D}
  let start_vertex := A
  let moves := 3

  -- Conditions:
  -- 1. Bug starts at one vertex of a tetrahedron.
  -- 2. Moves along edges with each edge having equal probability.
  -- 3. All choices are independent.

  -- Question: What is the probability that after three moves the bug will have visited every vertex exactly once?

  probability_of_visiting_each_vertex_exactly_once := (2 / 9)

theorem bug_visits_all_vertices :
  tetrahedron_prob = (2 / 9) :=
  sorry

end bug_visits_all_vertices_l228_228287


namespace incorrect_proposition_C_l228_228452

theorem incorrect_proposition_C (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a^4 + b^4 + c^4 + d^4 = 2 * (a^2 * b^2 + c^2 * d^2) → ¬ (a = b ∧ b = c ∧ c = d) := 
sorry

end incorrect_proposition_C_l228_228452


namespace fundraising_part1_fundraising_part2_l228_228134

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l228_228134


namespace ellipse_standard_form_l228_228067

theorem ellipse_standard_form 
  (F1 F2 P : ℝ × ℝ)
  (hF1 : F1 = (-1, 0)) 
  (hF2 : F2 = (1, 0)) 
  (hP : P = (1/2, sqrt 14 / 4)) :
  ∃ a b : ℝ, 
    b = 1 ∧ 
    a = 2 * sqrt 2 ∧ 
    (∀ x y : ℝ, (x, y) ∈ set_of (λ (x y : ℝ), (x - F1.1) ^ 2 + (x - F2.1) ^ 2 = 2 * a) →
        ( (x^2 / a^2) + (y^2 / b^2)) = 1) :=
sorry

end ellipse_standard_form_l228_228067


namespace exists_positive_irrationals_with_rational_sum_and_product_l228_228007

theorem exists_positive_irrationals_with_rational_sum_and_product :
  ∃ (x y : ℝ), (0 < x ∧ 0 < y ∧ irrational x ∧ irrational y ∧ rational (x + y) ∧ rational (x * y)) :=
sorry

end exists_positive_irrationals_with_rational_sum_and_product_l228_228007


namespace no_solution_sin_pi_sin_eq_cos_pi_cos_l228_228850

theorem no_solution_sin_pi_sin_eq_cos_pi_cos (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) : ¬ (sin (π * sin x) = cos (π * cos x)) :=
by
  sorry

end no_solution_sin_pi_sin_eq_cos_pi_cos_l228_228850


namespace david_ate_7_cookies_l228_228319

-- Variables for the problem
def total_cookies : ℝ := 24
def amy_fraction : ℝ := 1/4
def bob_fraction : ℝ := 1/3
def cathy_cookies : ℝ := 3.5

-- Definitions using conditions from the problem
def amys_cookies : ℝ := amy_fraction * total_cookies
def remaining_after_amy : ℝ := total_cookies - amys_cookies
def bobs_cookies : ℝ := bob_fraction * remaining_after_amy
def remaining_after_bob : ℝ := remaining_after_amy - bobs_cookies
def remaining_after_cathy : ℝ := remaining_after_bob - cathy_cookies
def davids_cookies : ℝ := 2 * cathy_cookies

-- Theorem to prove the answer
theorem david_ate_7_cookies :
  davids_cookies = 7 :=
by
  -- This proof is omitted as per the instructions.
  sorry

end david_ate_7_cookies_l228_228319


namespace sexagenary_cycle_year_1949_l228_228920

def sexagenary_year_2010 : String := "甲寅"  -- Jia Yin for the year 2010
def cycle_length : Nat := 60  -- Length of the sexagenary cycle
def known_year : Nat := 2010  -- Known Gregorian year
def target_year : Nat := 1949  -- Target Gregorian year

theorem sexagenary_cycle_year_1949 : 
  (target_year + cycle_length = known_year) → sexagenary_year_2010 = "甲寅" → "己丑" :=
by
  sorry  -- Proof to be provided

end sexagenary_cycle_year_1949_l228_228920


namespace smallest_n_inequality_l228_228346

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ (n : ℝ) * (x^4 + y^4 + z^4 + w^4)) ∧
            (∀ m : ℤ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > (m : ℝ) * (x^4 + y^4 + z^4 + w^4)) :=
begin
  use 4, -- smallest integer n satisfying the inequality for all x, y, z, w is 4
  split,
  { -- show that for all x y z w in ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ 4 * (x^4 + y^4 + z^4 + w^4)
    intros x y z w,
    sorry,
  },
  { -- show that for any m < 4, there exist x, y, z, w such that (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)
    intro m,
    intro h,
    sorry,
  }
end

end smallest_n_inequality_l228_228346


namespace perp_condition_parallel_condition_l228_228843

-- Given definitions
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2 * x + 3, -x)

-- Problem 1: Perpendicular case
theorem perp_condition (x : ℝ) (h : a x ⬝ b x = 0) : x = -1 ∨ x = 3 :=
sorry

-- Problem 2: Parallel case
theorem parallel_condition (x : ℝ) (h : a x.1 * b x.2 = x.2 * b x.1) :
  |a x - b x| = 2 * Real.sqrt 5 ∨ |a x - b x| = 2 :=
sorry

end perp_condition_parallel_condition_l228_228843


namespace range_of_a_l228_228424

noncomputable def f (a x : ℝ) := a^x - 1
noncomputable def g (a x : ℝ) := f a (x + 1) - 4

theorem range_of_a
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : ∀ x, x > 0 → f a x > 0)
  (h4 : ∀ x, ¬(g a x < 0 ∧ g a x < x)) :
  a ∈ set.Ioc 1 5 :=
sorry

end range_of_a_l228_228424


namespace proof_solution_l228_228520

variable (c d : ℝ) (h_c : c ≠ 0) (h_d : d ≠ 0)

def g (x : ℝ) : ℝ := 1 / (c * x + d)

theorem proof_solution : g c d (-2) = 1 / (-2 * c + d) :=
sorry

end proof_solution_l228_228520


namespace sectioned_volume_ratio_is_correct_l228_228934

noncomputable def parallelepiped_section_volume_ratio : ℝ := 
let a := 1 -- side length
in let h := 1 -- height
in let V_total := a * a * h -- volume of the parallelepiped
in let x := 2 * a * (5 * ℝ.sqrt 2 - 7) -- value for x derived from the equations
in let y := 2 * a * (3 * ℝ.sqrt 2 - 4) -- value for y derived from the equations
in let z := y * (ℝ.sqrt 2) -- value for z derived from the equations
in let V_sectioned := (1/6) * (x + z) * h * h -- volume of the sectioned part
in (V_sectioned / V_total)

theorem sectioned_volume_ratio_is_correct :
  parallelepiped_section_volume_ratio = 3 * ℝ.sqrt(2) - 4 :=
by sorry

end sectioned_volume_ratio_is_correct_l228_228934


namespace dog_groups_count_l228_228215

/-- 
Suppose we have 15 dogs and we need to divide them into three groups:
one with 4 dogs, one with 6 dogs, and one with 5 dogs.
How many ways can we form these groups 
such that Duke is in the 4-dog group and Bella is in the 6-dog group?
-/
theorem dog_groups_count : 
  let total_dogs := 15
  let group1_count := 4
  let group2_count := 6
  let group3_count := 5
  let remaining_dogs1 := total_dogs - 1  -- After selecting Duke
  let ways_to_select_group1 := Nat.choose remaining_dogs1 (group1_count - 1)
  let remaining_dogs2 := remaining_dogs1 - (group1_count - 1) - 1  -- After selecting Duke and group1_count - 1 dogs
  let ways_to_select_group2 := Nat.choose remaining_dogs2 (group2_count - 1)
  let total_ways := ways_to_select_group1 * ways_to_select_group2
  total_ways = 72072 := by 
    let total_dogs := 15
    let group1_count := 4
    let group2_count := 6
    let group3_count := 5
    let remaining_dogs1 := total_dogs - 1
    let ways_to_select_group1 := Nat.choose remaining_dogs1 (group1_count - 1)
    let remaining_dogs2 := remaining_dogs1 - (group1_count - 1) - 1
    let ways_to_select_group2 := Nat.choose remaining_dogs2 (group2_count - 1)
    let total_ways := ways_to_select_group1 * ways_to_select_group2
    show total_ways = 72072,
    calc
      ways_to_select_group1 = Nat.choose 14 3 : by rfl
      ... = 286 : by decide
      ways_to_select_group2 = Nat.choose 10 5 : by rfl
      ... = 252 : by decide
      total_ways = 286 * 252 : by rfl
      ... = 72072 : by rfl

end dog_groups_count_l228_228215


namespace seamless_assembly_with_equilateral_triangle_l228_228414

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end seamless_assembly_with_equilateral_triangle_l228_228414


namespace ladder_alley_width_l228_228913

theorem ladder_alley_width (l : ℝ) (m : ℝ) (w : ℝ) (h : m = l / 2) :
  w = (l * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end ladder_alley_width_l228_228913


namespace am_gm_inequality_l228_228038

theorem am_gm_inequality (n : ℕ) (hn : n ≥ 2) (a : Fin n → ℝ) (h_pos : ∀ (i : Fin n), a i > 0) : 
  (∑ i, a i) / n ≥ Real.sqrt (Fin n) (∏ i, a i) :=
sorry

end am_gm_inequality_l228_228038


namespace tan_double_angle_l228_228109

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end tan_double_angle_l228_228109


namespace solution_set_of_inequality_l228_228225

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, f x ∈ ℝ 

axiom f_at_neg2 : f (-2) = 2013

axiom f_prime_condition : ∀ x : ℝ, has_deriv_at f (f' x) x ∧ f' x < 2 * x

theorem solution_set_of_inequality :
  { x : ℝ | f(x) > x^2 + 2009 } = set.Iio (-2) :=
sorry

end solution_set_of_inequality_l228_228225


namespace number_of_students_l228_228320

theorem number_of_students : ∃ n : ℕ, n < 500 ∧ n % 25 = 24 ∧ n % 21 = 14 ∧ n = 449 :=
by
  use 449
  split
  · exact dec_trivial -- 449 < 500
  split
  · exact dec_trivial -- 449 % 25 = 24
  split
  · exact dec_trivial -- 449 % 21 = 14
  · exact dec_trivial -- n = 449
  sorry

end number_of_students_l228_228320


namespace original_selling_price_l228_228210

theorem original_selling_price (P : ℝ) (d1 d2 d3 t : ℝ) (final_price : ℝ) :
  d1 = 0.32 → -- first discount
  d2 = 0.10 → -- loyalty discount
  d3 = 0.05 → -- holiday discount
  t = 0.15 → -- state tax
  final_price = 650 → 
  1.15 * P * (1 - d1) * (1 - d2) * (1 - d3) = final_price →
  P = 722.57 :=
sorry

end original_selling_price_l228_228210


namespace inequality_lt_l228_228030

theorem inequality_lt (x y : ℝ) (h1 : x > y) (h2 : y > 0) (n k : ℕ) (h3 : n > k) :
  (x^k - y^k) ^ n < (x^n - y^n) ^ k := 
  sorry

end inequality_lt_l228_228030


namespace probability_divisible_by_11_is_zero_l228_228116

-- Define the conditions
def conditions_five_digit_sum_44 (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n ≤ 99999) ∧ (n.digits.sum = 44)

-- Define the property of being divisible by 11
def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the main theorem
theorem probability_divisible_by_11_is_zero :
  (∃ n, conditions_five_digit_sum_44 n ∧ divisible_by_11 n) → false := 
by
  sorry

end probability_divisible_by_11_is_zero_l228_228116


namespace police_catches_thief_in_two_hours_l228_228311

theorem police_catches_thief_in_two_hours :
  ∀ (S_thief S_police distance initial_delay : ℕ), 
    S_thief = 20 ∧ S_police = 40 ∧ distance = 60 ∧ initial_delay = 1 →
    let Distance_thief_initial := S_thief * initial_delay in
    let Remaining_distance := distance - Distance_thief_initial in
    let Relative_speed := S_police - S_thief in
    let Time := Remaining_distance / Relative_speed in
    Time = 2 :=
by
  intros S_thief S_police distance initial_delay,
  rintro ⟨hS_thief, hS_police, h_distance, h_initial_delay⟩,
  let Distance_thief_initial := S_thief * initial_delay,
  let Remaining_distance := distance - Distance_thief_initial,
  let Relative_speed := S_police - S_thief,
  let Time := Remaining_distance / Relative_speed,
  sorry

end police_catches_thief_in_two_hours_l228_228311


namespace non_adjacent_arrangement_l228_228799

-- Define the number of people
def numPeople : ℕ := 8

-- Define the number of specific people who must not be adjacent
def numSpecialPeople : ℕ := 3

-- Define the number of general people who are not part of the specific group
def numGeneralPeople : ℕ := numPeople - numSpecialPeople

-- Permutations calculation for general people
def permuteGeneralPeople : ℕ := Nat.factorial numGeneralPeople

-- Number of gaps available after arranging general people
def numGaps : ℕ := numGeneralPeople + 1

-- Permutations calculation for special people placed in the gaps
def permuteSpecialPeople : ℕ := Nat.descFactorial numGaps numSpecialPeople

-- Total permutations
def totalPermutations : ℕ := permuteSpecialPeople * permuteGeneralPeople

theorem non_adjacent_arrangement :
  totalPermutations = Nat.descFactorial 6 3 * Nat.factorial 5 := by
  sorry

end non_adjacent_arrangement_l228_228799


namespace smallest_value_at_x_eq_seven_l228_228375

theorem smallest_value_at_x_eq_seven : 
  let x := 7 in 
  let A := 6 / x in
  let B := 6 / (x + 1) in
  let C := 6 / (x - 1) in
  let D := x / 6 in
  let E := (x + 1) / 6 in
  B = min A (min B (min C (min D E))) := 
by
  sorry

end smallest_value_at_x_eq_seven_l228_228375


namespace roots_sum_cubic_poly_l228_228966

theorem roots_sum_cubic_poly 
  (a b c : ℝ)
  (h1 : 3 * a^3 - 9 * a^2 + 54 * a - 12 = 0)
  (h2 : 3 * b^3 - 9 * b^2 + 54 * b - 12 = 0)
  (h3 : 3 * c^3 - 9 * c^2 + 54 * c - 12 = 0)
  (h_roots : (Polynomial.roots (3 * X^3 - 9 * X^2 + 54 * X - 12)).to_finset = {a, b, c}) : 
  (a + 2 * b - 2)^3 + (b + 2 * c - 2)^3 + (c + 2 * a - 2)^3 = 162 := 
by 
  sorry

end roots_sum_cubic_poly_l228_228966


namespace tangent_line_at_a_eq_1_monotonic_intervals_l228_228430

noncomputable def f (x a : ℝ) := x^2 - 2 * (a + 1) * x + 2 * a * Real.log x

theorem tangent_line_at_a_eq_1 :
  let a := 1
  ∃ t : ℝ → ℝ, (∀ x, t x = -3) := sorry

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 → 
    (∀ x, (0 < x ∧ x < a → f' x a > 0) ∧ (a < x ∧ x < 1 → f' x a < 0) ∧ (1 < x → f' x a > 0))) ∧ 
  (a = 1 → 
    ∀ x, (0 < x → f' x a ≥ 0)) ∧ 
  (a > 1 → 
    (∀ x, (0 < x ∧ x < 1 → f' x a > 0) ∧ (1 < x ∧ x < a → f' x a < 0) ∧ (a < x → f' x a > 0))) := sorry

noncomputable def f' (x a : ℝ) : ℝ := (2 * x^2 - 2 * (a + 1) * x + 2 * a) / x

end tangent_line_at_a_eq_1_monotonic_intervals_l228_228430


namespace find_m_find_cosine_l228_228845

variables {m : ℝ}

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (m, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (h : dot_product a b = 1) : m = -2 :=
begin
  sorry
end

theorem find_cosine (h : dot_product a b = 1) :
  dot_product a b / (magnitude a * magnitude b) = 1 / 4 :=
begin
  sorry
end

end find_m_find_cosine_l228_228845


namespace center_of_circle_sum_eq_seven_l228_228221

theorem center_of_circle_sum_eq_seven 
  (h k : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 6 * x + 8 * y - 15 → (x - h)^2 + (y - k)^2 = 10) :
  h + k = 7 := 
sorry

end center_of_circle_sum_eq_seven_l228_228221


namespace sufficient_condition_l228_228387

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem sufficient_condition (a : ℝ) : (∀ x y : ℝ, N x y a → M x y) ↔ (a ≥ 5 / 4) := 
sorry

end sufficient_condition_l228_228387


namespace roundness_720_eq_7_l228_228567

def roundness (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization
    factors.sum (λ _ k => k)
  else 0

theorem roundness_720_eq_7 : roundness 720 = 7 := by
  sorry

end roundness_720_eq_7_l228_228567


namespace value_of_f_at_2_l228_228111

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  -- proof steps would go here
  sorry

end value_of_f_at_2_l228_228111


namespace min_values_l228_228074

theorem min_values (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
    (h3 : log (3 * x) + log y = log (x + y + 1)) :
    (∃ xy_min x_plus_y_min reciprocal_min : ℝ,
        xy_min = 1 ∧
        x_plus_y_min = 2 ∧
        reciprocal_min = 2) :=
by
  sorry

end min_values_l228_228074


namespace rocky_running_ratio_l228_228925

theorem rocky_running_ratio (x y : ℕ) (h1 : x = 4) (h2 : 2 * x + y = 36) : y / (2 * x) = 3 :=
by
  sorry

end rocky_running_ratio_l228_228925


namespace paige_album_count_l228_228992

theorem paige_album_count :
  ∀ (total_pictures pics_in_first_album pics_per_other_album : ℕ),
  total_pictures = 35 →
  pics_in_first_album = 14 →
  pics_per_other_album = 7 →
  (total_pictures - pics_in_first_album) / pics_per_other_album = 3 :=
by
  intros,
  sorry

end paige_album_count_l228_228992


namespace term_x2_coefficient_l228_228406

-- Definitions and conditions
def a : ℝ := ∫ x in 0..real.pi, (real.sin x + real.cos x)

def n : ℕ := if h : (∑ i in finset.range (6 + 1), (nat.choose 6 i)) = 64 then 6 else 0

-- Main statement
theorem term_x2_coefficient :
  a = 2 →
  n = 6 →
  ∃ (r : ℕ), (n - 2 * r) / 2 = 2 ∧
    (((-1 : ℤ) ^ r) * (nat.choose n r : ℤ) * (2 ^ (n - r))) = (-192 : ℤ) :=
by
  intro ha hn
  use 1
  constructor
  case left => calc (n - 2 * 1) / 2 = (6 - 2 * 1) / 2 := by simp [hn]
                               _ = 4 / 2 := by norm_num
                               _ = 2 := by norm_num
  case right => calc (((-1 : ℤ) ^ 1) * (nat.choose 6 1 : ℤ) * (2 ^ (6 - 1))) = (-1) * 6 * 32 := by norm_num
                                  _ = -192 := by norm_num

end term_x2_coefficient_l228_228406


namespace find_a_plus_d_l228_228800

theorem find_a_plus_d (a b c d : ℝ) (h₁ : ab + bc + ca + db = 42) (h₂ : b + c = 6) : a + d = 7 := 
sorry

end find_a_plus_d_l228_228800


namespace proof_problem_l228_228393

noncomputable def parabola_focus (p : ℝ) (hp : 0 < p) : Prop :=
  (0, (p / 2)) = (0, 1)

noncomputable def underscore_parabola (x y : ℝ) (p : ℝ) := x^2 = 2 * p * y

noncomputable def min_angle_and_line (k : ℝ) : Prop :=
  (∀ x1 x2 y1 y2, 
    (x1 + x2 = 4 * k) ∧ 
    (x1 * x2 = -4) ∧ 
    (x1, y1) ∈ parabola_focus p _
    ∧ (x2, y2) ∈ parabola_focus p _) →
    let Q := (2 * k, 2 * k^2 + 1) in
    let AB := 2 * sqrt (k^2 + 1) in
    sin (arcsin ((2 * k^2 + 1) / AB)) = sin (π / 6)

noncomputable def equation_of_line (k : ℝ) : Prop :=
  k = 0 → (λ x, k * x + 1) = (λ x, 1)

theorem proof_problem (p : ℝ) (hp: 0 < p) :
    parabola_focus p hp → p = 2 ∧ 
    (∃ k, min_angle_and_line k ∧ equation_of_line k) := sorry

end proof_problem_l228_228393


namespace projections_on_angle_bisectors_l228_228552

theorem projections_on_angle_bisectors (A B C: Point) : 
  let B1 := projection A (angle_bisector_internal B) in
  let B2 := projection A (angle_bisector_external B) in
  let C1 := projection A (angle_bisector_internal C) in
  let C2 := projection A (angle_bisector_external C) in
  Collinear [B1, B2, C1, C2] :=
begin
  sorry -- Proof goes here
end

end projections_on_angle_bisectors_l228_228552


namespace range_of_a_l228_228472

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x * (x - a) < 1) ↔ a ∈ set.Ioi (-1) :=
sorry

end range_of_a_l228_228472


namespace possible_denominators_count_l228_228214

variable (a b c : ℕ)
-- Conditions
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def no_two_zeros (a b c : ℕ) : Prop := ¬(a = 0 ∧ b = 0) ∧ ¬(b = 0 ∧ c = 0) ∧ ¬(a = 0 ∧ c = 0)
def none_is_eight (a b c : ℕ) : Prop := a ≠ 8 ∧ b ≠ 8 ∧ c ≠ 8

-- Theorem
theorem possible_denominators_count : 
  is_digit a ∧ is_digit b ∧ is_digit c ∧ no_two_zeros a b c ∧ none_is_eight a b c →
  ∃ denoms : Finset ℕ, denoms.card = 7 ∧ ∀ d ∈ denoms, 999 % d = 0 :=
by
  sorry

end possible_denominators_count_l228_228214


namespace max_h_existence_of_m_l228_228435

-- Definition of the function h
def h (x : ℝ) : ℝ := Real.log x - x + 1

-- Maximum value of the function h
theorem max_h : ∀ x > 0, h 1 = 0 ∧ (∀ y > 0, h y ≤ 0) := by
  sorry

-- Existence and range of m
theorem existence_of_m : ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 → 
  ∃ m : ℝ, m ≤ -1/2 ∧ (m * (x2^2 - x1^2) + (x2 * Real.log x2 - x1 * Real.log x1)) > 0 := by
  sorry

end max_h_existence_of_m_l228_228435


namespace num_chords_through_P_integer_length_l228_228204

open Real

variables (P : Point) (O : Point)
variable (distance_PO : dist P O = 12)
variable (radius_O : radius O = 20)

theorem num_chords_through_P_integer_length : 
  count_integer_length_chords P O = 9 :=
sorry

end num_chords_through_P_integer_length_l228_228204


namespace sum_sequence_equals_74_6_l228_228767

noncomputable def sequence_value (n : ℕ) : ℝ :=
  n * (1 - 1 / (n ^ 2))

noncomputable def sequence_sum (a b : ℕ) : ℝ :=
  ∑ n in finset.range (b - a + 1) + a, sequence_value n

theorem sum_sequence_equals_74_6 :
  sequence_sum 3 12 = 74.6 :=
sorry

end sum_sequence_equals_74_6_l228_228767


namespace valid_stone_placement_l228_228001

theorem valid_stone_placement
  (m n : ℕ)
  (odd_m : m % 2 = 1)
  (n_eq : n = (m * (m - 1)) / 2) :
  (∀ i j (hi : i < m) (hj : j < m), matrix m n ℕ)
  (c : ℕ) (all_columns_same : ∀ i j, in_col i = in_col j)
  (unique_rows : ∀ i j, i ≠ j → in_row i ≠ in_row j) :
  True :=
sorry

end valid_stone_placement_l228_228001


namespace members_playing_both_l228_228917

theorem members_playing_both 
  (B T N Neither : ℕ) 
  (h1 : B = 17)
  (h2 : T = 19)
  (h3 : N = 28)
  (h4 : Neither = 2) 
  : B + T - (N - Neither) = 10 :=
by
  rw [h1, h2, h3, h4]
  sorry

end members_playing_both_l228_228917


namespace find_a_coefficient_of_x_rational_terms_largest_terms_l228_228795

open Real

noncomputable def expansion_of_expr (a x : ℝ) : ℝ := (a / x - sqrt x) ^ 6

theorem find_a (a : ℝ) (h : expansion_of_expr a 1 = 60) : a = 2 :=
sorry

theorem coefficient_of_x (a : ℝ) (h : a = 2) : -12 = -12 :=
sorry

theorem rational_terms (a : ℝ) (h : a = 2) : 
  (expansion_of_expr a x) = 120 ∨ (expansion_of_expr a x) = -2 * x^3 :=
sorry

theorem largest_terms (a : ℝ) (h : a = 2) :
  ((expansion_of_expr a x) = -960 * x^(-3 / 2)) ∧
  ((expansion_of_expr a x) = -12) :=
sorry

end find_a_coefficient_of_x_rational_terms_largest_terms_l228_228795


namespace regular_octagon_area_l228_228704

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l228_228704


namespace no_such_complex_numbers_exist_l228_228763

theorem no_such_complex_numbers_exist (a b c : ℂ) (h : ℕ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  ¬ (∀ k l m : ℤ, abs k + abs l + abs m ≥ 1996 → abs (k * a + l * b + m * c) > 1 / h) :=
begin
  sorry
end

end no_such_complex_numbers_exist_l228_228763


namespace math_problem_equiv_proof_l228_228126

variables {A B C H : Type} [InnerProductSpace ℝ H]
variables {a b c : ℝ} -- sides opposite to angles A, B, C respectively
variables {AB AC BC : H} -- vectors corresponding to sides
variable {AH : H} -- altitude on side BC
variable (α β γ : ℝ) -- angles opposite to sides a, b, c

noncomputable def correct_conclusions (α β γ : ℝ) (a b c : ℝ) (AB AC BC AH : H) 
  [InnerProductSpace ℝ H] : Prop :=
(AC • ((1 / norm AH) • AH) = c * Real.sin β) ∧
(BC • (AC - AB) = b^2 + c^2 - 2 * b * c * Real.cos α)

theorem math_problem_equiv_proof :
  correct_conclusions α β γ a b c AB AC BC AH :=
begin
  sorry
end

end math_problem_equiv_proof_l228_228126


namespace smaller_balloon_radius_is_correct_l228_228306

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct_l228_228306


namespace rectangular_parallelepiped_is_cube_l228_228485

theorem rectangular_parallelepiped_is_cube (P : Parallelepiped) (hP : isRectangular P)
  (h : ∃ H : Plane, isCrossSection H P ∧ isRegularHexagon H) : isCube P :=
sorry

end rectangular_parallelepiped_is_cube_l228_228485


namespace teams_B_and_C_worked_together_days_l228_228216

def workload_project_B := 5/4
def time_team_A_project_A := 20
def time_team_B_project_A := 24
def time_team_C_project_A := 30

def equation1 (x y : ℕ) : Prop := 
  3 * x + 5 * y = 60

def equation2 (x y : ℕ) : Prop := 
  9 * x + 5 * y = 150

theorem teams_B_and_C_worked_together_days (x : ℕ) (y : ℕ) :
  equation1 x y ∧ equation2 x y → x = 15 := 
by 
  sorry

end teams_B_and_C_worked_together_days_l228_228216


namespace cos_E_floor_1000_l228_228921

theorem cos_E_floor_1000 {EF GH FG EH : ℝ} {E G : ℝ} (h1 : EF = 200) (h2 : GH = 200) (h3 : FG + EH = 380) (h4 : E = G) (h5 : EH ≠ FG) :
  ∃ (cE : ℝ), cE = 11/16 ∧ ⌊ 1000 * cE ⌋ = 687 :=
by sorry

end cos_E_floor_1000_l228_228921


namespace num_ways_to_express_as_sum_primes_l228_228132

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_two_primes (n : ℕ) : ℕ → Prop :=
λ count, ∃ p q : ℕ, p + q = n ∧ is_prime p ∧ is_prime q ∧ count = 1

theorem num_ways_to_express_as_sum_primes (n : ℕ) (h : n = 10002) : ∃ count, is_sum_of_two_primes 10002 count :=
by {
  use 1,
  sorry
}

end num_ways_to_express_as_sum_primes_l228_228132


namespace simple_interest_years_l228_228599

theorem simple_interest_years
  (CI : ℝ)
  (SI : ℝ)
  (p1 : ℝ := 4000) (r1 : ℝ := 0.10) (t1 : ℝ := 2)
  (p2 : ℝ := 1750) (r2 : ℝ := 0.08)
  (h1 : CI = p1 * (1 + r1) ^ t1 - p1)
  (h2 : SI = CI / 2)
  (h3 : SI = p2 * r2 * t2) :
  t2 = 3 :=
by
  sorry

end simple_interest_years_l228_228599


namespace circle_intersection_distance_l228_228095

theorem circle_intersection_distance (r1 r2 d : ℝ) (h_r1 : r1 = 2) (h_r2 : r2 = 3)
(h_common : ∃ (P : ℝ × ℝ), (∀ (C1 C2 : ℝ × ℝ), dist C1 P = r1 ∧ dist C2 P = r2) ∧ dist C1 C2 = d) :
1 ≤ d ∧ d ≤ 5 :=
by
  -- Hypotheses based on the given conditions.
  have h_radii : r1 = 2 ∧ r2 = 3, from ⟨h_r1, h_r2⟩,
  have h_max : d ≤ r1 + r2, from sorry,
  have h_min : d ≥ |r2 - r1|, from sorry,
  
  -- Combining the inequalities to conclude the proof.
  exact ⟨h_min, h_max⟩

end circle_intersection_distance_l228_228095


namespace triangle_problem_l228_228476

theorem triangle_problem (BC : ℝ) (AC : ℝ) (A C : ℝ) (sin : ℝ → ℝ) (cos : ℝ → ℝ) :
  BC = sqrt 5 ∧ AC = 3 ∧ sin C = 2 * sin A →
  let AB := 2 * sqrt 5 in
  let sin2A_minus_pi4 := sin (2 * A - π / 4) in
  AB = 2 * sqrt 5 ∧ sin2A_minus_pi4 = sqrt 2 / 10 :=
by
  sorry

end triangle_problem_l228_228476


namespace area_of_inscribed_octagon_l228_228711

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228711


namespace certain_number_exceeds_2134_l228_228474

theorem certain_number_exceeds_2134 (x : ℤ) (h1 : x ≤ 3) : ∃ n : ℤ, 2.134 * 10^x < n ∧ n ≥ 2135 :=
by
  sorry

end certain_number_exceeds_2134_l228_228474


namespace cell_division_after_3_hours_l228_228652

-- Define the conditions
def time_interval_division : ℕ := 30  -- In minutes
def total_time : ℕ := 3 * 60  -- 3 hours in minutes

-- Describe the cell division property
def cell_division_every_interval (cells : ℕ) : ℕ := cells * 2

-- State the theorem
theorem cell_division_after_3_hours : 
  let intervals := total_time / time_interval_division in
  let final_cells := (2 : ℕ) ^ intervals in
  final_cells = 64 := 
by 
  sorry

end cell_division_after_3_hours_l228_228652


namespace sum_formula_sum_zero_if_sin_half_theta_zero_l228_228368

noncomputable def S_n (n : ℕ) (θ : ℝ) : ℝ :=
  Σ k in finset.range n, (sin (k * θ)) / (cos (2 * k * θ) + cos θ)

theorem sum_formula (n : ℕ) (θ : ℝ) (h : sin (θ/2) ≠ 0) :
  S_n n θ = (1 / (4 * sin (θ / 2))) * ((1 / cos ((2 * n + 1) / 2 * θ)) - (1 / cos (θ / 2))) :=
sorry

theorem sum_zero_if_sin_half_theta_zero (n : ℕ) (θ : ℝ) (h : sin (θ/2) = 0) :
  S_n n θ = 0 :=
sorry

end sum_formula_sum_zero_if_sin_half_theta_zero_l228_228368


namespace sum_of_angles_l228_228409

theorem sum_of_angles (α β : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π) (h_tan_α: Real.tan α = 1 / 2) (h_tan_β: Real.tan β = 1 / 3) : α + β = π / 4 := 
by 
  sorry

end sum_of_angles_l228_228409


namespace solving_k_l228_228846

def vector_a : ℝ × ℝ :=
  (2 * Real.sin (4 / 3 * Real.pi), Real.cos (5 / 6 * Real.pi))

def vector_b (k : ℝ) : ℝ × ℝ :=
  (k, 1)

noncomputable def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

theorem solving_k :
  ∀ k : ℝ, are_parallel vector_a (vector_b k) → k = 2 :=
by
  intros k h
  -- Proof goes here
  sorry

end solving_k_l228_228846


namespace integer_solutions_to_equation_l228_228448

theorem integer_solutions_to_equation :
  ∀ x : ℤ, 
  (x - 3) ^ (27 - x ^ 3) = 1 ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  intro x
  split
  sorry -- Proof will go here


end integer_solutions_to_equation_l228_228448


namespace five_squares_configuration_l228_228324

-- Definitions for conditions
def small_squares_non_intersecting_within_unit_square : Prop :=
  ∃ (n : ℕ), n = 5 -- five congruent small squares within a unit square

def cent_square_midpoints_are_vertices (a b : ℤ) : Prop :=
  ∃ (s : ℚ), s = (a - real.sqrt 2) / b ∧ a > 0 ∧ b > 0

-- Theorem statement with the conditions
theorem five_squares_configuration (a b : ℤ) 
  (h1 : small_squares_non_intersecting_within_unit_square) 
  (h2 : cent_square_midpoints_are_vertices a b) : a + b = 11 := 
begin
  sorry
end

end five_squares_configuration_l228_228324


namespace range_of_a_l228_228467

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (1 : ℝ) 2, x^2 + a ≤ a * x - 3) ↔ 7 ≤ a :=
sorry

end range_of_a_l228_228467


namespace oil_leak_while_working_l228_228734

theorem oil_leak_while_working:
  ∀ (before working total : ℕ),
  before = 6522 →
  total = 11687 →
  total - before = 5165 :=
by
  intros before total h_before h_total
  rw [h_before, h_total]
  exact rfl

end oil_leak_while_working_l228_228734


namespace find_m_n_l228_228112

noncomputable def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
noncomputable def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem find_m_n (a b c d : ℤ) (m n : ℤ)
  (h1 : (a / 5)^m * (b / 4)^n * (1 / c)^18 = 1 / (d * 10^35))
  (h2 : is_even a)
  (h3 : is_odd b)
  (h4 : 3 ∣ c ∧ 7 ∣ c)
  (h5 : is_prime d) :
  m = 0 ∧ n = 0 :=
sorry

end find_m_n_l228_228112


namespace octagon_area_correct_l228_228692

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l228_228692


namespace proof_possible_trajectories_l228_228840

noncomputable def possible_trajectories (A B : ℝ × ℝ) (m : ℝ) : list string :=
  if m = 0 then ["straight line"]
  else if m = -1 then ["circle"]
  else if m > 0 then ["hyperbola"]
  else if m < 0 then ["ellipse"]
  else []

theorem proof_possible_trajectories :
  ∀ (A B : ℝ × ℝ) (m : ℝ), 
  |possible_trajectories A B m| == ["straight line", "circle", "hyperbola", "ellipse"].length :=
by
  intro A B m
  sorry

end proof_possible_trajectories_l228_228840


namespace octagon_area_l228_228702

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228702


namespace positive_difference_eq_30_l228_228257

theorem positive_difference_eq_30 : 
  let x1 := 12
      x2 := -18
  in |x1 - x2| = 30 := 
by
  sorry

end positive_difference_eq_30_l228_228257


namespace complex_number_quadrant_l228_228573

def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_quadrant : inFourthQuadrant (1 - 2 * complex.i) :=
by
  sorry

end complex_number_quadrant_l228_228573


namespace par_condition_perp_condition_vectors_parallel_or_perpendicular_l228_228444

variable (m : ℝ)

def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, m)
def vec_c : ℝ × ℝ := (-1, 2)

-- Given conditions
theorem par_condition (hm : m = -1) : (1 = 0) := sorry

theorem perp_condition (hm : m = 3 / 2) : (1 = 0) := sorry

-- Main Theorem
theorem vectors_parallel_or_perpendicular :
  (∀ m : ℝ, vec_a + vec_b = (1, m - 1) →
    ((vec_a + vec_b) = λc : ℝ, vec_c c) ∧ m = -1) ∨
  (∀ m : ℝ, vec_a + vec_b = (1, m - 1) →
    (vec_a + vec_b) ⬝ vec_c = 0 ∧ m = 3 / 2) := sorry

end par_condition_perp_condition_vectors_parallel_or_perpendicular_l228_228444


namespace coefficient_of_x_inv_in_expansion_l228_228813

-- For integrals and binomial coefficients
open Real
open Tactic

noncomputable def integral_sin : ℝ :=
∫ x in 0..π, sin x

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
nat.choose n k

theorem coefficient_of_x_inv_in_expansion :
  let a := integral_sin in
  a = 2 → coefficient_of_x_expansion (λ x => (x - (a / x))^5) (-1) = -80 :=
by 
  intros a h
  sorry

end coefficient_of_x_inv_in_expansion_l228_228813


namespace alyssa_gave_away_puppies_l228_228730

def start_puppies : ℕ := 12
def remaining_puppies : ℕ := 5

theorem alyssa_gave_away_puppies : 
  start_puppies - remaining_puppies = 7 := 
by
  sorry

end alyssa_gave_away_puppies_l228_228730


namespace number_of_trailing_zeroes_base8_l228_228861

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l228_228861


namespace entry_exit_ways_l228_228294

theorem entry_exit_ways (n : ℕ) (h : n = 8) : n * (n - 1) = 56 :=
by {
  sorry
}

end entry_exit_ways_l228_228294


namespace no_integer_solutions_l228_228779

theorem no_integer_solutions (m n : ℤ) (h1 : m ^ 3 + n ^ 4 + 130 * m * n = 42875) (h2 : m * n ≥ 0) :
  false :=
sorry

end no_integer_solutions_l228_228779


namespace gcd_18_30_45_l228_228364

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l228_228364


namespace loss_percentage_l228_228732

variables (C : ℝ) (hC : 0 < C)

/-- The selling price with a 60% profit -/
def S1 := 1.60 * C

/-- The selling price when sold at half the price that gave a 60% profit -/
def S2 := 0.80 * C

/-- The loss incurred when selling at S2 -/
def loss := C - S2

/-- The loss percentage when the article is sold at half price that gave a 60% profit -/
theorem loss_percentage : 
  (loss / C) * 100 = 20 :=
by
  rw [loss, S2]
  calc
    (C - 0.80 * C) / C * 100 = (0.20 * C) / C * 100 : by ring
    ... = 0.20 * 100 : by rw div_mul_eq_mul_div; ring
    ... = 20 : by norm_num

-- Adding a sorry to skip the proof
sorry

end loss_percentage_l228_228732


namespace factorial_base8_trailing_zeros_l228_228890

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228890


namespace octagon_area_l228_228672

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228672


namespace octagon_area_l228_228674

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l228_228674


namespace area_of_inscribed_octagon_l228_228714

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228714


namespace value_of_a_in_S_l228_228091

variable (S : Set ℕ) (T : Set ℕ) (a : ℕ)
variable (hS : S = {1, 2}) (hT : T = {a}) (h_union : S ∪ T = S)

theorem value_of_a_in_S : a ∈ {1, 2} :=
by
  rw [←hS, ←hT] at h_union
  have : T ⊆ S, from Set.subset_of_union_eq h_union
  rw [Set.singleton_subset_iff] at this
  exact this

end value_of_a_in_S_l228_228091


namespace value_2x_y_l228_228041

theorem value_2x_y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y + 5 = 0) : 2*x + y = 0 := 
by
  sorry

end value_2x_y_l228_228041


namespace area_BCD_correct_l228_228176

variables (A B C D : ℝ × ℝ × ℝ) (b c d x y z θ : ℝ)

-- Conditions
def points_conditions : Prop := 
  A = (0, 0, 0) ∧ 
  B = (b, 0, 0) ∧
  C = (0, c, 0) ∧
  D = (p, q, d * Real.sin θ) ∧
  (1 / 2) * b * c = x ∧
  -- You may add here any additional necessary constraints based on the problem statement

-- Question
def area_BCD : ℝ :=
  1 / 2 * Real.sqrt ((c * d * Real.sin θ) ^ 2 + (b * d * Real.sin θ) ^ 2 + (-b * c + b * q) ^ 2)

-- Proof problem
theorem area_BCD_correct (A B C D : ℝ × ℝ × ℝ) 
                         (b c d x y z θ p q : ℝ)
                         (h : points_conditions A B C D b c d x y z θ p q) :
  area_BCD A B C D b c d θ p q = 1 / 2 * Real.sqrt ((c * d * Real.sin θ) ^ 2 +
                                                    (b * d * Real.sin θ) ^ 2 +
                                                    (b * c - b * q) ^ 2) :=
sorry

end area_BCD_correct_l228_228176


namespace max_sum_abc_divisible_by_13_l228_228119

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l228_228119


namespace cubic_difference_l228_228410

theorem cubic_difference (x y : ℝ) (h1 : x + y = 15) (h2 : 2 * x + y = 20) : x^3 - y^3 = -875 := 
by
  sorry

end cubic_difference_l228_228410


namespace exists_subset_sum_100_l228_228602

theorem exists_subset_sum_100 (numbers : Fin 100 → ℕ)
  (h1 : ∀ (i : Fin 100), numbers i ≤ 100)
  (h2 : (∑ i, numbers i) = 200) : 
  ∃ (s : Finset (Fin 100)), (∑ i in s, numbers i) = 100 :=
sorry

end exists_subset_sum_100_l228_228602


namespace problem_l228_228243

theorem problem (r : ℝ) (h : r = 2) :
  let K := (sqrt 3 : ℝ)
  ∃ (a b : ℕ), K = real.sqrt a - b ∧ 100 * a + b = 300 := by
  sorry

end problem_l228_228243


namespace factorial_base8_trailing_zeros_l228_228888

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l228_228888


namespace probability_of_letter_in_word_l228_228114

theorem probability_of_letter_in_word (alphabet : Finset Char) (word : String) 
  (unique_letters : Finset Char) (h1 : word = "probability") 
  (h2 : unique_letters = {'P', 'R', 'O', 'B', 'A', 'I', 'L', 'T', 'Y'})
  (h3 : alphabet.card = 26) :
  (unique_letters.card / alphabet.card.toReal) = 9 / 26 := 
by
  sorry

end probability_of_letter_in_word_l228_228114


namespace circle_with_parallel_tangents_l228_228488

theorem circle_with_parallel_tangents (TP TQ PQ r : ℝ) (hTP : TP = 4) (hTQ : TQ = 9) 
  (h_parallel : TP ∥ TQ) (h_tangents : ∀ (C : Set ℝ) (P Q : ℝ), P ∈ C → Q ∈ C → ∀ x, x ∈ C → TP = TQ ∧ PQ = x) : 
  r = 6 :=
sorry

end circle_with_parallel_tangents_l228_228488


namespace solution_set_of_inequality_l228_228827

def f (x : ℝ) : ℝ :=
  (x^3 - x) * (1/2 - 1 / (Real.exp x + 1))

theorem solution_set_of_inequality :
  {x : ℝ | f x * Real.log (x + 2) ≤ 0} = set.Ioo (-2) 1 ∪ {1} :=
by
  sorry

end solution_set_of_inequality_l228_228827


namespace parallelogram_area_l228_228341

def base : ℝ := 5
def height : ℝ := 4

theorem parallelogram_area : base * height = 20 := by
  sorry

end parallelogram_area_l228_228341


namespace largest_number_after_removal_l228_228755

theorem largest_number_after_removal :
  ∀ (s : Nat), s = 1234567891011121314151617181920 -- representing the start of the sequence
  → true
  := by
    sorry

end largest_number_after_removal_l228_228755


namespace PointEEquidistantFromLines_l228_228615

-- Define that two circles touch each other at point P
structure CirclesTouchAt (P : ℝ × ℝ) : Prop :=
mk :: (circle1_tangent : bool) (circle2_tangent : bool)

-- Define a line that is tangent to one circle at E and intersects the other circle at Q and R
structure TangentLineAndIntersections (circle1 circle2 : CirclesTouchAt P) (E Q R : ℝ × ℝ) : Prop :=
mk :: (line_tangent : circle1.circle1_tangent = true ∨ circle2.circle2_tangent = true)
      (intersects_other : true) -- Simulating the condition that Q and R are intersection points of the line and the other circle

-- Prove that point E is equidistant from lines PQ and PR
theorem PointEEquidistantFromLines (P E Q R : ℝ × ℝ) (circle1 circle2 : CirclesTouchAt P) 
  (tangentLine : TangentLineAndIntersections circle1 circle2 E Q R) :
  (distance_to_line E (P, Q) = distance_to_line E (P, R)) :=
  sorry

end PointEEquidistantFromLines_l228_228615


namespace nested_radical_convergence_l228_228140

theorem nested_radical_convergence : 
  let x := sqrt (2 + sqrt (2 + sqrt (2 + ...))) in
  x = 2 :=
by
  sorry

end nested_radical_convergence_l228_228140


namespace pascal_triangle_42nd_number_l228_228621

theorem pascal_triangle_42nd_number (n : ℕ) (k : ℕ) (h₁ : n = 44) (h₂ : k = 41) : 
  nat.choose n k = 13254 := by
  sorry

end pascal_triangle_42nd_number_l228_228621


namespace find_k_series_eq_10_l228_228373

theorem find_k_series_eq_10 :
  ∃ k : ℝ, (∑ n in (0:ℕ) ..(100 : ℕ), (4 + n * k) / (5 ^ n)) = 10 ↔ k = 16 :=
by
  sorry

end find_k_series_eq_10_l228_228373


namespace min_AP_BP_l228_228175

-- Definitions based on conditions in the problem
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The theorem to prove the minimum value of AP + BP
theorem min_AP_BP
  (P : ℝ × ℝ)
  (hP_parabola : parabola P.1 P.2) :
  dist P A + dist P B ≥ 9 :=
sorry

end min_AP_BP_l228_228175


namespace probability_is_three_fifths_l228_228383

def total_balls : ℕ := 5
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3

noncomputable def probability_of_different_colors : ℚ :=
  let total_ways := nat.choose total_balls 2
  let ways_different_colors := yellow_balls * red_balls
  (ways_different_colors : ℚ) / (total_ways : ℚ)

theorem probability_is_three_fifths : probability_of_different_colors = 3/5 :=
by sorry

end probability_is_three_fifths_l228_228383


namespace x_intercept_eq_neg_three_half_l228_228229

-- Define the points
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (3, 9)

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the equation of the line in slope-intercept form
def line_equation (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  slope p1 p2 * (x - p1.1) + p1.2

-- Define the intercept function
def x_intercept_of_line (p1 p2 : ℝ × ℝ) : ℝ :=
  let k := slope p1 p2 in
  let b := p1.2 - k * p1.1 in
  -b / k

-- The proof statement
theorem x_intercept_eq_neg_three_half :
  x_intercept_of_line point1 point2 = -3 / 2 :=
by
  sorry

end x_intercept_eq_neg_three_half_l228_228229


namespace length_AH_length_AE_area_AFCH_l228_228754

-- Definitions necessary for the math problem
structure Trapezoid :=
(ab bc cd ad : ℝ)
(angle_bad : ℝ)
(angle_abc : ℝ)
(angle_bac : ℝ)
(angle_cad : ℝ)

def trapezoidABCD : Trapezoid :=
{ ab := 5,
  bc := 5,
  cd := 5,
  ad := 10,
  angle_bad := 60,
  angle_abc := 120,
  angle_bac := 30,
  angle_cad := 30 }

-- Define E, F, H based on the description provided
structure Point := (x y : ℝ)

def E := Point.mk ?
def F := Point.mk ?
def H := Point.mk ?

-- Convert the given translated problem to Lean statements
theorem length_AH (AH HD : ℝ) : AH = (20 / 3) := sorry

theorem length_AE (AH : ℝ) (cos_30 : ℝ) : AE = (10 * (sqrt 3) / 3) := sorry

theorem area_AFCH (AC FH : ℝ) 
  (area_AFCH : real) : area_AFCH = (50 * sqrt 3 / 3) := sorry

end length_AH_length_AE_area_AFCH_l228_228754


namespace sum_of_all_digits_S_is_93_l228_228507

def is_valid_N (N : ℕ) : Prop :=
  let x := N / 10000
  N % 10000 = 2020 ∧ x * 10000 + 2020 = N ∧ (500 * x + 101) % x = 0

def digit_sum (n : ℕ) : ℕ :=
  n.to_digits.foldr (· + ·) 0

def sum_digits_S : ℕ :=
  (Finset.filter is_valid_N (Finset.range 200000) |>.sum digit_sum)

theorem sum_of_all_digits_S_is_93 : sum_digits_S = 93 :=
  sorry

end sum_of_all_digits_S_is_93_l228_228507


namespace fg_at_3_l228_228388

def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := x^2 + 5

theorem fg_at_3 : f (g 3) = 10 := by
  sorry

end fg_at_3_l228_228388


namespace no_tangent_sinx_l228_228160

theorem no_tangent_sinx (b : ℝ) : ¬∃ x : ℝ, deriv (λ x, sin x) x = 3 / 2 :=
by sorry

end no_tangent_sinx_l228_228160


namespace second_player_wins_12_petals_second_player_wins_11_petals_l228_228291

def daisy_game (n : Nat) : Prop :=
  ∀ (p1_move p2_move : Nat → Nat → Prop), n % 2 = 0 → (∃ k, p1_move n k = false) ∧ (∃ ℓ, p2_move n ℓ = true)

theorem second_player_wins_12_petals : daisy_game 12 := sorry
theorem second_player_wins_11_petals : daisy_game 11 := sorry

end second_player_wins_12_petals_second_player_wins_11_petals_l228_228291


namespace actual_discount_is_correct_and_difference_is_4_l228_228309

-- Define the original price (can be any real number since we're dealing with percentages)
variable (P : ℝ)

-- Define the 40% discount
def first_discount (P : ℝ) : ℝ := 0.60 * P

-- Define the additional 10% discount on the reduced price
def second_discount (P : ℝ) : ℝ := 0.90 * first_discount P

-- Calculate the actual percentage discount
def actual_discount (P : ℝ) : ℝ := 1 - second_discount P / P

-- Define the store's claim
def claimed_discount : ℝ := 0.50

-- Calculate the difference between the actual discount and the claimed discount
def discount_difference (P : ℝ) : ℝ := claimed_discount - actual_discount P

-- The statement we need to prove
theorem actual_discount_is_correct_and_difference_is_4 (P : ℝ) : actual_discount P = 0.46 ∧ discount_difference P = 0.04 :=
by
  sorry  -- proof is omitted

end actual_discount_is_correct_and_difference_is_4_l228_228309


namespace degree_monomial_l228_228574

variable (p m n : ℕ) -- Assume p, m, n are natural numbers.

def monomial : ℕ := degree (2^2 * p * m^2 * n^2)

theorem degree_monomial :
  monomial p m n = 5 := sorry

end degree_monomial_l228_228574


namespace marie_reads_messages_per_day_l228_228985

theorem marie_reads_messages_per_day (x : ℕ) (h : 7 * (x - 6) = 98) : x = 20 :=
begin
  sorry
end

end marie_reads_messages_per_day_l228_228985


namespace length_PT_correct_l228_228915

-- Define the points P, Q, R, S
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (6, 0)
def R : ℝ × ℝ := (1, 0)
def S : ℝ × ℝ := (5, 3)

-- Equations of lines PQ and RS
def linePQ (x : ℝ) : ℝ := -2 / 3 * x + 4
def lineRS (x : ℝ) : ℝ := 3 / 4 * (x - 1)

-- Intersection point T
def T : ℝ × ℝ := (57 / 17, 125 / 51)

-- Length of PT
def PT : ℝ := Real.sqrt ((57 / 17 - 0)^2 + (125 / 51 - 4)^2)

-- Statement to prove the length of PT
theorem length_PT_correct : PT = Real.sqrt ((57 / 17)^2 + (-79 / 51)^2) := by
  sorry

end length_PT_correct_l228_228915


namespace monotonically_increasing_range_a_l228_228906

theorem monotonically_increasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x^3 + a * x) ≤ (y^3 + a * y)) → a ≥ 0 := 
by
  sorry

end monotonically_increasing_range_a_l228_228906


namespace remaining_sum_eq_seven_eighths_l228_228440

noncomputable def sum_series := 
  (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32) + (1 / 64)

noncomputable def removed_terms := 
  (1 / 16) + (1 / 32) + (1 / 64)

theorem remaining_sum_eq_seven_eighths : 
  sum_series - removed_terms = 7 / 8 := by
  sorry

end remaining_sum_eq_seven_eighths_l228_228440


namespace cricket_average_increase_l228_228446

theorem cricket_average_increase
    (A : ℝ) -- average score after 18 innings
    (score19 : ℝ) -- runs scored in 19th inning
    (new_average : ℝ) -- new average after 19 innings
    (score19_def : score19 = 97)
    (new_average_def :  new_average = 25)
    (total_runs_def : 19 * new_average = 18 * A + 97) : 
    new_average - (18 * A + score19) / 19 = 4 := 
by
  sorry

end cricket_average_increase_l228_228446


namespace find_a_b_compare_f_g_l228_228831

-- Definitions of the given functions
def f (x : ℝ) : ℝ := Real.log x
def g (a b x : ℝ) : ℝ := a * x + b / x

-- Condition on the intersection point
def intersection_condition (a b : ℝ) : Prop :=
  g a b 1 = 0

-- Condition on the common tangent line
def tangent_condition (a b : ℝ) : Prop :=
  (g a b 1 = 0) ∧ (a - b = 1)

-- Theorem for question (1)
theorem find_a_b (a b : ℝ) (h1 : intersection_condition a b) (h2 : tangent_condition a b) :
  a = 1/2 ∧ b = -1/2 :=
sorry

-- Function for the difference between f and g
def F (x : ℝ) : ℝ := f x - g (1/2) (-1/2) x

-- Derivative of F
def F_prime (x : ℝ) : ℝ := (1/x) - (1/2) - (1 / (2 * x^2))

-- Theorem for question (2)
theorem compare_f_g (x : ℝ) (hx : x > 0) :
  (0 < x ∧ x <= 1 → f x >= g (1/2) (-1/2) x) ∧ (x > 1 → f x < g (1/2) (-1/2) x) :=
sorry

end find_a_b_compare_f_g_l228_228831


namespace abc_value_l228_228512

theorem abc_value {a b c : ℂ} 
  (h1 : a * b + 5 * b + 20 = 0) 
  (h2 : b * c + 5 * c + 20 = 0) 
  (h3 : c * a + 5 * a + 20 = 0) : 
  a * b * c = 100 := 
by 
  sorry

end abc_value_l228_228512


namespace lines_perpendicular_to_sides_of_triangle_l228_228596

theorem lines_perpendicular_to_sides_of_triangle 
  (T : Triangle) (H : Point) 
  (perpendicular_bisectors : ∀ s ∈ T.sides, ∃ b ∈ T.angle_bisectors, s ⊥ b)
  (vertices_as_midpoints : ∀ v ∈ T.vertices, ∃ t ∈ another_triangle, v = midpoint t.bisectors):
  ∀ v ∈ T.vertices, ∃ s ∈ T.sides, line_through(H, v) ⊥ s :=
begin
  sorry
end

end lines_perpendicular_to_sides_of_triangle_l228_228596


namespace problem1_eval_problem2_simplify_l228_228281

-- Problem 1: Calculate and prove
theorem problem1_eval : (1 / 3)⁻¹ + Real.sqrt 18 - 4 * Real.cos (Math.pi / 4) = 3 + Real.sqrt 2 :=
by
  sorry

-- Problem 2: Simplify and prove
theorem problem2_simplify (a : Real) : 
  (2 * a) / (4 - a ^ 2) / (a / (a - 2)) + (a / (a + 2)) = (a - 2) / (a + 2) :=
by
  sorry

end problem1_eval_problem2_simplify_l228_228281


namespace angle_BAC_equal_to_120_degrees_l228_228919

namespace TriangleProof

open Real

variables {A B C L K : Point}

-- Conditions as definitions
def is_triangle (A B C : Point) : Prop := 
  collinear A B C = false

def angle_bisector_intersection_points (A B C L K : Point) : Prop :=
  (is_triangle A B C) ∧
  (L = intersection_of_angle_bisector B A C) ∧
  (K = intersection_of_angle_bisector A B C)

def segment_KL_is_bisector_of_AKC (K L A C : Point) : Prop :=
  is_angle_bisector (K L) ((A K) ∪ (K C))

-- Theorem statement
theorem angle_BAC_equal_to_120_degrees
  (h1 : is_triangle A B C)
  (h2 : angle_bisector_intersection_points A B C L K)
  (h3 : segment_KL_is_bisector_of_AKC K L A C) :
  angle A K C = 120 := 
  sorry

end TriangleProof

end angle_BAC_equal_to_120_degrees_l228_228919


namespace area_of_inscribed_octagon_l228_228713

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l228_228713


namespace arrangement_count_l228_228610

theorem arrangement_count (n_sing n_dance : ℕ) 
  (h_sing_pos : n_sing = 6) (h_dance_pos : n_dance = 4) : 
  (∃ (arrangements : ℕ), 
     arrangements = P 7 4 * A 6 6) :=
by 
  use (P 7 4 * A 6 6)
  sorry

end arrangement_count_l228_228610


namespace perimeter_of_plot_l228_228230

theorem perimeter_of_plot
  (width : ℝ) 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : cost_per_meter = 6.5)
  (h2 : total_cost = 1170)
  (h3 : total_cost = (2 * (width + (width + 10))) * cost_per_meter) 
  :
  (2 * ((width + 10) + width)) = 180 :=
by
  sorry

end perimeter_of_plot_l228_228230


namespace audrey_key_limes_needed_l228_228739

theorem audrey_key_limes_needed
  (cup_juice_needed : ℚ := 1/4)
  (double_juice : ℚ := 2)
  (yield_per_lime : ℚ := 1)
  (tablespoons_per_cup : ℚ := 16) :
  let total_juice_needed := cup_juice_needed * double_juice
  let juice_in_tablespoons := total_juice_needed * tablespoons_per_cup
  let key_limes_needed := juice_in_tablespoons / yield_per_lime
  key_limes_needed = 8 :=
by
  dunfold total_juice_needed
  dunfold juice_in_tablespoons
  dunfold key_limes_needed
  norm_num
  sorry

end audrey_key_limes_needed_l228_228739


namespace find_m_range_of_x_l228_228045

def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3 * m + 2

theorem find_m (m : ℝ) (H_dec : m^2 - 1 < 0) (H_f1 : f m 1 = 0) : 
  m = 1 / 2 :=
sorry

theorem range_of_x (x : ℝ) :
  f (1 / 2) (x + 1) ≥ x^2 ↔ -3 / 4 ≤ x ∧ x ≤ 0 :=
sorry

end find_m_range_of_x_l228_228045


namespace num_valid_seating_arrangements_l228_228353

-- Define the dimensions of the examination room
def rows : Nat := 5
def columns : Nat := 6
def total_seats : Nat := rows * columns

-- Define the condition for students not sitting next to each other
def valid_seating_arrangements (rows columns : Nat) : Nat := sorry

-- The theorem to prove the number of seating arrangements
theorem num_valid_seating_arrangements : valid_seating_arrangements rows columns = 772 := 
by 
  sorry

end num_valid_seating_arrangements_l228_228353


namespace total_weekly_coffee_cost_l228_228193

-- Define Maddie's coffee consumption details
def cups_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 2
  else if day = "Tuesday" ∨ day = "Thursday" ∨ day = "Saturday" then 3
  else if day = "Sunday" then 1
  else 0

-- Define other constants
def cost_per_bag : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def milk_cost_per_week : ℝ := 4
def syrup_cost_per_bottle : ℝ := 6
def tablespoons_per_bottle : ℕ := 24
def honey_cost_per_jar : ℝ := 5
def teaspoons_per_jar : ℕ := 48
def ounces_per_cup : ℝ := 1.5)

-- Calculate total coffee beans used per week
def total_ounces_used_per_week : ℝ :=
  (3 * (cups_per_day "Monday" * ounces_per_cup)) +
  (3 * (cups_per_day "Tuesday" * ounces_per_cup)) +
  (cups_per_day "Sunday" * ounces_per_cup)

-- Calculate number of coffee bags used per week and their cost
def bags_per_week : ℝ := total_ounces_used_per_week / ounces_per_bag
def coffee_cost_per_week : ℝ := bags_per_week * cost_per_bag

-- Calculate total syrup used per week and their cost
def total_cups_per_week : ℕ :=
  cups_per_day "Monday" + cups_per_day "Tuesday" +
  cups_per_day "Wednesday" + cups_per_day "Thursday" +
  cups_per_day "Friday" + cups_per_day "Saturday" +
  cups_per_day "Sunday"

def syrup_usage_per_week : ℝ := total_cups_per_week
def syrup_cost_per_week : ℝ := (syrup_usage_per_week / tablespoons_per_bottle.toReal) * syrup_cost_per_bottle

-- Calculate total honey used per week and their cost
def honey_usage_per_week : ℝ := 2
def honey_cost_per_week : ℝ := (honey_usage_per_week / teaspoons_per_jar.toReal) * honey_cost_per_jar

-- Total weekly cost
def total_cost_per_week : ℝ := coffee_cost_per_week + milk_cost_per_week + syrup_cost_per_week + honey_cost_per_week

theorem total_weekly_coffee_cost : total_cost_per_week = 26.55 := by
  sorry

end total_weekly_coffee_cost_l228_228193


namespace time_to_cross_pole_is_2_5_l228_228155

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end time_to_cross_pole_is_2_5_l228_228155


namespace distance_AC_squared_range_l228_228722

noncomputable def squareDistanceAC (AB BC : ℝ) (theta : ℝ) : ℝ :=
  AB^2 + BC^2 - 2 * AB * BC * cos theta

theorem distance_AC_squared_range :
  let AB := 15
  let BC := 25
  let theta_min := 30 * (Mathlib.Pi / 180)
  let theta_max := 45 * (Mathlib.Pi / 180)
  525 ≤ squareDistanceAC AB BC theta_min ∧ squareDistanceAC AB BC theta_min ≤ 585 ∧
  525 ≤ squareDistanceAC AB BC theta_max ∧ squareDistanceAC AB BC theta_max ≤ 585 :=
by
  let AB := 15
  let BC := 25
  let theta_min := 30 * (Mathlib.Pi / 180)
  let theta_max := 45 * (Mathlib.Pi / 180)
  have h1 : squareDistanceAC AB BC theta_min ≈ 525.25 := sorry
  have h2 : squareDistanceAC AB BC theta_max ≈ 584.875 := sorry
  split
  · exact h1
  · split
  · exact h1
  · exact h2
  · exact h2

end distance_AC_squared_range_l228_228722


namespace product_of_largest_and_second_largest_prime_factors_of_180_l228_228623

theorem product_of_largest_and_second_largest_prime_factors_of_180:
  let primeFactors := [2, 3, 5] in
  (List.length primeFactors = 3 ∧ primeFactors ≠ []) ∧
  List.last (List.tail (List.sort primeFactors)) * List.last (List.sort primeFactors) = 15 :=
by
  sorry

end product_of_largest_and_second_largest_prime_factors_of_180_l228_228623


namespace total_holes_dug_l228_228163

theorem total_holes_dug :
  (Pearl_digging_rate * 21 + Miguel_digging_rate * 21) = 26 :=
by
  -- Definitions based on conditions
  let Pearl_digging_rate := 4 / 7
  let Miguel_digging_rate := 2 / 3
  -- Sorry placeholder for the proof
  sorry

end total_holes_dug_l228_228163


namespace find_first_group_num_l228_228310

-- Define the conditions
def num_people := 960
def num_selected := 32
def group := ℕ

-- Define the number drawn by the groups
def num_by_group (n : group) (x : ℕ) := x + (n - 1) * 30

-- Define the given conditions
def fifth_group_num := 129

-- The statement to be proved in Lean 4
theorem find_first_group_num (x : ℕ) 
  (h1 : num_by_group 5 x = fifth_group_num) :
  x = 9 :=
by 
  sorry

end find_first_group_num_l228_228310


namespace integer_div_l228_228718

-- Definitions from the problem conditions
def seq_a (c : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ((Σ i in (finset.range n).map nat.succ, f (n / i)) = n ^ 10)

-- Main theorem statement
theorem integer_div (a : ℕ → ℕ) (c : ℕ) (h_seq : seq_a c a) (hc : c > 0) :
  ∀ n : ℕ, n > 0 → (c ^ a n - c ^ a (n - 1)) % n = 0 :=
by
  sorry

end integer_div_l228_228718


namespace find_a_value_l228_228417

theorem find_a_value :
  ∃ a : ℝ, (let coeff_x3 := (choose 6 3) + (choose 6 2) * (-1) * a + (choose 6 1) * a^2 
   in coeff_x3 = 56) ↔ (a = 6 ∨ a = -1) :=
sorry

end find_a_value_l228_228417


namespace modified_poly_has_root_l228_228595

theorem modified_poly_has_root
  (a b : ℝ) (p s : ℝ)
  (h_a : a = -(p + s))
  (h_b : b = ps) :
  (let a' := a + p in let b' := b - p^2 in
   (a' * a') - 4 * b' ≥ 0) :=
by
  sorry

end modified_poly_has_root_l228_228595


namespace min_value_expression_l228_228056

theorem min_value_expression (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ (x y : ℝ), (\frac 1 x + \frac 4 y) ≥ 9 := 
by
  sorry

end min_value_expression_l228_228056


namespace factorial_trailing_zeros_base_8_l228_228866

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l228_228866


namespace range_of_x_l228_228082

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 → -1 < x ∧ x < 2 := by
  sorry

end range_of_x_l228_228082


namespace greatest_integer_equality_l228_228790

theorem greatest_integer_equality (m : ℝ) (h : m ≥ 3) :
  Int.floor ((m * (m + 1)) / (2 * (2 * m - 1))) = Int.floor ((m + 1) / 4) :=
  sorry

end greatest_integer_equality_l228_228790


namespace regular_polygon_sides_l228_228009

theorem regular_polygon_sides (h : ∀ n : ℕ, (120 * n) = 180 * (n - 2)) : 6 = 6 :=
by
  sorry

end regular_polygon_sides_l228_228009


namespace seating_arrangement_l228_228923

theorem seating_arrangement (m n : ℕ) (h1 : m = 4) (h2 : n = 7) :
  ∃ k, k = n * (n-1) * (n-2) * (n-3) ∧ k = 840 :=
by
  use (n * (n-1) * (n-2) * (n-3))
  split
  · rfl
  · sorry

end seating_arrangement_l228_228923


namespace proof_C_ST_l228_228506

-- Definitions for sets and their operations
def A1 : Set ℕ := {0, 1}
def A2 : Set ℕ := {1, 2}
def S : Set ℕ := A1 ∪ A2
def T : Set ℕ := A1 ∩ A2
def C_ST : Set ℕ := S \ T

theorem proof_C_ST : 
  C_ST = {0, 2} := 
by 
  sorry

end proof_C_ST_l228_228506


namespace fixed_vertex_max_min_area_l228_228442

-- Problem 1: Prove the fixed vertex at (-1, 0)
theorem fixed_vertex (m : ℝ) :
  let l1 := λ p : ℝ × ℝ, m * p.1 - p.2 + m = 0
  let l2 := λ p : ℝ × ℝ, p.1 + m * p.2 - m * (m + 1) = 0
  let l3 := λ p : ℝ × ℝ, (m + 1) * p.1 - p.2 + (m + 1) = 0
  ∃ A : ℝ × ℝ, A = (-1, 0) ∧ l1 A ∧ l3 A :=
sorry

-- Problem 2: Prove max and min area occurs at specific m values
theorem max_min_area (m : ℝ) :
  let area := λ m : ℝ, (1/2) * (m^2 + m + 1) / (real.sqrt (m^2 + 1))
  (∀ m : ℝ, area m ≤ 3/4 ∧ area m ≥ 1/4) ∧ (area 1 = 3/4) ∧ (area (-1) = 1/4) :=
sorry

end fixed_vertex_max_min_area_l228_228442


namespace lambda_range_exists_l228_228397

theorem lambda_range_exists (a S b T : ℕ → ℝ) (λ : ℝ) :
  (a 22 - 3 * a 7 = 2) ∧
  (∃ r : ℝ, r ≠ 0 ∧ 1 / a 2 = r ∧ (S 2 - 3).sqrt = r * S 3) ∧
  (∀ n : ℕ, b n = 4 * (n + 1) / (a n ^ 2 * a (n + 2) ^ 2)) ∧
  (∀ n : ℕ, T n = ∑ i in finset.range n, b i) →
  (∀ n : ℕ, 64 * T n < |3 * λ - 1|) →
  (λ ≤ -4/3 ∨ λ ≥ 2) :=
by
  sorry

end lambda_range_exists_l228_228397


namespace mixtape_runtime_l228_228741

-- Conditions definitions
def sideA_song_lengths := [3, 4, 5, 6, 3, 7] -- in minutes
def sideA_transition := 0.5 -- in minutes
def sideA_silence := 2 -- in minutes
def sideA_bonus := 4 -- in minutes

def sideB_song_lengths := [6, 6, 8, 5] -- in minutes
def sideB_transition := 0.75 -- in minutes

def normal_speed := 1
def fast_speed := 1.5

-- Calculate total runtime for Side A
def sideA_runtime : ℝ :=
  (sideA_song_lengths.sum + sideA_transition * (sideA_song_lengths.length - 1)) 
  + sideA_silence + sideA_bonus

-- Calculate total runtime for Side B
def sideB_runtime : ℝ :=
  sideB_song_lengths.sum + sideB_transition * (sideB_song_lengths.length - 1)

-- Calculate total runtimes
def total_runtime_normal : ℝ := sideA_runtime + sideB_runtime
def total_runtime_fast : ℝ := total_runtime_normal / fast_speed

-- Theorem to prove
theorem mixtape_runtime :
  total_runtime_normal = 63.75 ∧
  total_runtime_fast ≈ 42.5 := by
  sorry

end mixtape_runtime_l228_228741


namespace apples_in_basket_l228_228910

theorem apples_in_basket
  (total_rotten : ℝ := 12 / 100)
  (total_spots : ℝ := 7 / 100)
  (total_insects : ℝ := 5 / 100)
  (total_varying_rot : ℝ := 3 / 100)
  (perfect_apples : ℝ := 66) :
  (perfect_apples / ((1 - (total_rotten + total_spots + total_insects + total_varying_rot))) = 90) :=
by
  sorry

end apples_in_basket_l228_228910
