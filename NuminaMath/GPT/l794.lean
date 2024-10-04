import Mathlib

namespace a10_plus_a11_plus_a12_eq_66_l794_794182

variable {ℕ : Type} -- Natural numbers type

-- Define the specifically given conditions
variable {a : ℕ → ℕ}   -- Sequence definition
variable {S : ℕ → ℕ}   -- Partial sum definition where S_n = sum of first n terms

-- Given conditions
axiom S3_eq_12 : S 3 = 12
axiom S6_eq_42 : S 6 = 42

-- To prove
theorem a10_plus_a11_plus_a12_eq_66 : a 10 + a 11 + a 12 = 66 :=
by
  sorry

end a10_plus_a11_plus_a12_eq_66_l794_794182


namespace centroid_of_vector_sum_zero_l794_794322

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V] 
variables {P : Type*} [AddTorsor V P]

open_locale affine

def is_centroid (A B C G : P) : Prop :=
∃ M : P, (midpoint ℝ B C = M) ∧ (affine_combination (finset.univ) ![A, M] ![2/3, 1/3] = G)

theorem centroid_of_vector_sum_zero (A B C G : P) 
  (h : (vector_to ℝ A G) + (vector_to ℝ B G) + (vector_to ℝ C G) = 0) : 
  is_centroid A B C G :=
sorry

end centroid_of_vector_sum_zero_l794_794322


namespace area_of_region_l794_794489

-- Define the given equation of the region
def region_eq (x y : ℝ) := x^2 + y^2 - 10 * x + 24 * y = -144

-- State the theorem for the area enclosed by the region
theorem area_of_region : (∃ x y : ℝ, region_eq x y) -> (area := (25 * π)) sorry

end area_of_region_l794_794489


namespace expected_winnings_of_peculiar_die_l794_794094

noncomputable def peculiar_die_expected_winnings : ℝ := 
  let P_6 := 1 / 4
  let P_even := 2 / 5
  let P_odd := 3 / 5
  let winnings_6 := 4
  let winnings_even := -2
  let winnings_odd := 1
  P_6 * winnings_6 + P_even * winnings_even + P_odd * winnings_odd

theorem expected_winnings_of_peculiar_die :
  peculiar_die_expected_winnings = 0.80 :=
by
  sorry

end expected_winnings_of_peculiar_die_l794_794094


namespace probability_one_boy_one_girl_l794_794088

theorem probability_one_boy_one_girl (boys girls : ℕ) (h_boys : boys = 2) (h_girls : girls = 1) :
  (∃ (students : ℕ) (choose : ℕ) (total_pairs : ℕ) (favorable_pairs : ℕ),
    students = boys + girls ∧
    choose = 2 ∧
    total_pairs = (students.choose choose) ∧
    favorable_pairs = 2 ∧
    (favorable_pairs : ℚ) / (total_pairs : ℚ) = 2 / 3) :=
begin
  use [3, 2, (3.choose 2), 2],
  split,
  { exact eq_add_of_add_eq h_boys h_girls, },
  split,
  { refl, },
  split,
  { apply nat.choose},
  split,
  { refl, },
  { norm_num,
    rw [nat.cast_two, nat.cast_choose ℚ 3 2],
    norm_num, },
end

end probability_one_boy_one_girl_l794_794088


namespace busy_waiter_served_43_customers_l794_794411

-- Definitions for the initial number of customers at each table
def table1_customers : Nat := 2 + 4
def table2_customers : Nat := 4 + 3
def table3_customers : Nat := 3 + 5
def table4_customers : Nat := 5 + 2
def table5_customers : Nat := 2 + 1
def table6_customers : Nat := 1 + 2

-- Definitions for the changes during the shift
def table3_customers_after_leaving : Nat := table3_customers - 2
def table4_customers_after_joining : Nat := table4_customers + (1 + 2)

-- Definition for walk-in customers
def walkin_customers : Nat := 4 + 4

-- Total customers served
def total_customers_served : Nat :=
  table1_customers +
  table2_customers +
  table3_customers_after_leaving +
  table4_customers_after_joining +
  table5_customers +
  table6_customers +
  walkin_customers

-- Proof that the total customers served equals 43
theorem busy_waiter_served_43_customers : total_customers_served = 43 := by
  rw [
    table1_customers,
    table2_customers,
    table3_customers,
    table4_customers,
    table5_customers,
    table6_customers,
    walkin_customers,
    table3_customers_after_leaving,
    table4_customers_after_joining
  ]
  sorry

end busy_waiter_served_43_customers_l794_794411


namespace sum_series_eq_three_halves_l794_794463

theorem sum_series_eq_three_halves :
  (∑' n in (Set.Ici 2), (6 * n^3 - 2 * n^2 - 2 * n + 2) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n)) = 3 / 2 := by
  sorry

end sum_series_eq_three_halves_l794_794463


namespace g_at_5_l794_794746

def f (x : ℝ) : ℝ := 5 / (3 - x)

def g (x : ℝ) : ℝ := 1 / ((3 * x - 5) / x) + 7

theorem g_at_5 : g 5 = 7.5 := by
  sorry

end g_at_5_l794_794746


namespace bob_questions_ratio_l794_794872

theorem bob_questions_ratio :
  ∃ Q_2 Q_3: ℕ, (Q_2 = 26) ∧ (Q_3 = 2 * Q_2) ∧ (13 + Q_2 + Q_3 = 91) ∧ ((Q_2 : ℚ) / 13 = 2) :=
by
  obtain ⟨Q_2, Q_3, hQ_2, hQ_3, hSum⟩ := ⟨26, 52, rfl, by rw [mul_assoc, mul_comm], rfl⟩
  have hRatio : ((Q_2 : ℚ) / 13 = 2) := by norm_num [hQ_2]
  exact ⟨Q_2, Q_3, hQ_2, hQ_3, hSum, hRatio⟩

end bob_questions_ratio_l794_794872


namespace simplify_sum_l794_794254

theorem simplify_sum (a b c : ℕ) (h1 : a = 3) (h2 : b = 16) (h3 : c = 12) : 
  (sqrt 8 + 1 / sqrt 8 + sqrt 9 + 1 / sqrt 9) = (a * sqrt 8 + b * sqrt 9) / c → a + b + c = 31 := 
by
  sorry

end simplify_sum_l794_794254


namespace largest_n_divisibility_condition_l794_794504

def S1 (n : ℕ) : ℕ := (n * (n + 1)) / 2
def S2 (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem largest_n_divisibility_condition : ∀ (n : ℕ), (n = 1) → (S2 n) % (S1 n) = 0 :=
by
  intros n hn
  rw [hn]
  sorry

end largest_n_divisibility_condition_l794_794504


namespace rectangle_area_is_108_l794_794099

-- Definitions derived from the conditions
def square_area : ℝ := 36
def rectangle_width : ℝ := real.sqrt square_area
def rectangle_length : ℝ := 3 * rectangle_width

-- Proof statement
theorem rectangle_area_is_108 : rectangle_width * rectangle_length = 108 := by
  sorry

end rectangle_area_is_108_l794_794099


namespace june_longer_than_laura_l794_794461

def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1/a + 1/b + 1/c)

noncomputable def christopher_sword : ℝ := 15
noncomputable def jameson_sword : ℝ := 2 * christopher_sword + 3
noncomputable def june_sword : ℝ := jameson_sword + 5
noncomputable def laura_sword : ℝ := harmonic_mean christopher_sword jameson_sword june_sword

theorem june_longer_than_laura : (june_sword - laura_sword ≈ 13.68) :=
by
  sorry

end june_longer_than_laura_l794_794461


namespace Omega2_tangent_CD_l794_794797

-- Given definitions based on conditions
variables {Ω Ω₁ Ω₂ : Circle}
variables {M N A B C D : Point}

-- Conditions
axiom touch_internally_Ω1_Ω_at_M : Ω₁.touches Ω M
axiom touch_internally_Ω2_Ω_at_N : Ω₂.touches Ω N
axiom center_Ω2_on_Ω1 : Ω₂.center_on Ω₁
axiom common_chord_intersects_Ω_at_A_B : Ω₁.common_chord.intersects Ω₂ Ω A B
axiom MA_MB_intersect_Ω₁_at_C_D : (MA_line : Line) ∧ (MB_line : Line) ∧ MA_line.intersects Ω₁ C ∧ MB_line.intersects Ω₁ D

-- Theorem to prove
theorem Omega2_tangent_CD : Ω₂.tangent CD :=
sorry

end Omega2_tangent_CD_l794_794797


namespace largest_base5_to_base10_l794_794014

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794014


namespace rationalize_denominator_sum_l794_794727

noncomputable def rationalize_denominator (x y z : ℤ) :=
  x = 4 ∧ y = 49 ∧ z = 35 ∧ y ∣ 343 ∧ z > 0 

theorem rationalize_denominator_sum : 
  ∃ A B C : ℤ, rationalize_denominator A B C ∧ A + B + C = 88 :=
by
  sorry

end rationalize_denominator_sum_l794_794727


namespace num_distinct_integers_written_as_sums_of_special_fractions_l794_794877

theorem num_distinct_integers_written_as_sums_of_special_fractions :
  let special_fraction (a b : ℕ) := a + b = 15
  ∃ n : ℕ, n = 11 ∧ 
    ∀ i j : ℕ, special_fraction i j → 
      ∃ k : ℕ, 
        is_sum_of_special_fractions i j k → k < 29 := sorry

def is_sum_of_special_fractions (i j k : ℕ) : Prop := -- Custom definition to define sum of special fractions.
  -- details to be filled in as necessary
  sorry

end num_distinct_integers_written_as_sums_of_special_fractions_l794_794877


namespace solve_angle_sum_proof_l794_794271

def angle_sum_proof_problem (x : ℝ) : Prop :=
  let angle_abc := 90
  let angle_abd := 3 * x
  let angle_dbc := 2 * x
  angle_abd + angle_dbc = angle_abc ∧ x = 18

theorem solve_angle_sum_proof : ∃ x : ℝ, angle_sum_proof_problem x :=
by
  use 18
  dsimp [angle_sum_proof_problem]
  split
  · norm_num
  · norm_num

end solve_angle_sum_proof_l794_794271


namespace tangent_line_equation_l794_794218

noncomputable def f (x : ℝ) : ℝ := real.log x / x - x * (1 - real.log 1)

theorem tangent_line_equation : f(1) = -1 / 2 ∧ f'(1) = 1 / 2 → ∀ x y : ℝ, x - 2*y - 2 = 0 :=
begin
  sorry,
end

end tangent_line_equation_l794_794218


namespace range_of_a_l794_794213

noncomputable def f (a : ℝ) : ℝ → ℝ 
| x => if x > 1 then a ^ x else (8 - a) * x + 4

theorem range_of_a (a : ℝ) :
  (∀ x > 1, f a x = a ^ x) →
  (∀ x ≤ 1, f a x = (8 - a) * x + 4) →
  (∀ x ∈ ℝ, (x ≤ 1 → f a x = (8 - a) * x + 4) ∧ (1 < x → f a x = a ^ x)) →
  (∀ x, f a x = continuous f a x) →
  ( ∃ a, a ∈ Ioc 6 8) :=
by
  intro h1 h2 h3 h4
  sorry

end range_of_a_l794_794213


namespace gcd_lcm_product_l794_794166

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l794_794166


namespace work_days_A_l794_794078

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end work_days_A_l794_794078


namespace probability_divisible_by_3_remainder_5_l794_794717

theorem probability_divisible_by_3_remainder_5 :
  let total_digits := 10 ^ 3,
      valid_combinations := 37 in
  (valid_combinations / total_digits : ℝ) = 0.037 :=
by
  sorry

end probability_divisible_by_3_remainder_5_l794_794717


namespace bisector_ratio_is_two_l794_794280

section angle_bisector_theorem

variables {X Y Z Q F G : Type}
variables [triangle X Y Z] -- Assume X, Y, Z form a triangle
variables [angle_bisector XF : X -> F]
variables [angle_bisector YG : Y -> G]
variables [XY : distance X Y = 8]
variables [XZ : distance X Z = 6]
variables [YZ : distance Y Z = 4]
variables (Q : intersection XF YG)

theorem bisector_ratio_is_two :
  YG Q / QG = 2 :=
sorry  -- Proof can be filled in later

end angle_bisector_theorem

end bisector_ratio_is_two_l794_794280


namespace volume_relationship_l794_794435

open Real

theorem volume_relationship (r : ℝ) (A M C : ℝ)
  (hA : A = (1/3) * π * r^3)
  (hM : M = π * r^3)
  (hC : C = (4/3) * π * r^3) :
  A + M + (1/2) * C = 2 * π * r^3 :=
by
  sorry

end volume_relationship_l794_794435


namespace triangle_equilateral_of_cos_ratio_l794_794256

variable {α : Type*} [Field α] [Trigonometric α]

def is_equilateral_triangle (A B C a b c : α) : Prop :=
  A = B ∧ B = C

theorem triangle_equilateral_of_cos_ratio
  (A B C a b c : α)
  (h : a / (cos A) = b / (cos B) ∧ b / (cos B) = c / (cos C)) :
  is_equilateral_triangle A B C a b c :=
by 
  sorry

end triangle_equilateral_of_cos_ratio_l794_794256


namespace cone_lateral_surface_area_proof_l794_794570

-- Define the radius and slant height of the cone
def radius (cone : Type) := 5 -- in cm
def slant_height (cone : Type) := 12 -- in cm

-- Define the lateral surface area function given radius and slant height
def lateral_surface_area (r l : ℝ) := (1 / 2) * (2 * Real.pi * r) * l

-- The theorem we need to prove
theorem cone_lateral_surface_area_proof :
  lateral_surface_area 5 12 = 60 * Real.pi := by
  sorry

end cone_lateral_surface_area_proof_l794_794570


namespace equation_of_hyperbola_C_min_distance_P_M_l794_794206

-- Define the conditions given
def hyperbola1 (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1
def a1 := (2 : ℝ)
def b1 := (3 : ℝ)
def c1 := Real.sqrt (a1 + b1)
def foci_hyperbola1 := (fun (a b : ℝ) => ℝ.sqrt (a + b)) a1 b1

-- Equation of hyperbola C based on conditions
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1
def a2 := (2 : ℝ)
def b2 := Real.sqrt ((4 + 1) - 4)
def equation_of_C (x y : ℝ) := x^2 / 4 - y^2 = 1

theorem equation_of_hyperbola_C :
  (forall x y, (hyperbola_C x y)) := by
  sorry

-- Minimum distance calculation
def point_M := (5, 0)
def point_P_on_C (x y : ℝ) : Prop := hyperbola_C x y

def distance_P_M (x y : ℝ) : ℝ :=
  Real.sqrt (((x - 5)^2) + y^2)

theorem min_distance_P_M :
  (forall x y, point_P_on_C x y -> distance_P_M x y = 2) := by
  sorry

end equation_of_hyperbola_C_min_distance_P_M_l794_794206


namespace speed_of_second_part_journey_l794_794427

theorem speed_of_second_part_journey
    (d : ℝ)
    (total_time : ℝ)
    (speed_first : ℝ)
    (speed_third : ℝ)
    (v : ℝ) :
    d = 0.5 →
    total_time = 11 / 60 →
    speed_first = 5 →
    speed_third = 15 →
    0.1 + d / v + 1 / 30 = total_time →
    v = 10 :=
by
  intros d_eq d_0_5 total_time_eq total_time_11_60 speed_first_eq speed_first_5 speed_third_eq speed_third_15 time_eq
  sorry

end speed_of_second_part_journey_l794_794427


namespace no_integer_solutions_l794_794156

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l794_794156


namespace sixth_term_of_arithmetic_sequence_l794_794787

noncomputable def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem sixth_term_of_arithmetic_sequence
  (a d : ℕ)
  (h₁ : sum_first_n_terms a d 4 = 10)
  (h₂ : a + 4 * d = 5) :
  a + 5 * d = 6 :=
by {
  sorry
}

end sixth_term_of_arithmetic_sequence_l794_794787


namespace curve_translation_correct_l794_794382

def original_curve (x y : ℝ) : Prop := y * cos x + 2 * y - 1 = 0

def translated_curve (x y : ℝ) : Prop := (y + 1) * sin x + 2 * y + 1 = 0

theorem curve_translation_correct (x y : ℝ) : 
  original_curve x y → translated_curve (x - π/2) (y - 1) :=
by sorry

end curve_translation_correct_l794_794382


namespace find_a_zero_of_function_solve_inequality_l794_794982

-- Given the function f(x) passes through the point (2,1)
def function_passes_through (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : Prop :=
  log a 2 = 1

-- Prove 1: log_a(2) = 1 implies a = 2
theorem find_a : ∀ (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1), (log a 2 = 1) → a = 2 :=
by
  intros a h1 h2 h_log
  sorry

-- Prove 2: zero of the function log_2(x) is at x = 1
theorem zero_of_function : ∀ (x : ℝ), log 2 x = 0 ↔ x = 1 :=
by
  intros x
  sorry

-- Prove 3: solve the inequality log_2(x) < 1
theorem solve_inequality : ∀ (x : ℝ), log 2 x < 1 ↔ 0 < x ∧ x < 2 :=
by
  intros x
  sorry

end find_a_zero_of_function_solve_inequality_l794_794982


namespace number_of_polynomials_equals_seventeen_l794_794161

theorem number_of_polynomials_equals_seventeen :
  let P (n : ℕ) (coeffs : Fin (n + 1) → ℤ) :=
    (n + (Fin.sum (fun i => abs (coeffs i)))) = 4 in
  (Finset.card { p : Σ n, Fin (n + 1) → ℤ // P p.1 p.2 } = 17) :=
begin
  sorry
end

end number_of_polynomials_equals_seventeen_l794_794161


namespace total_number_of_pieces_paper_l794_794710

-- Define the number of pieces of paper each person picked up
def olivia_pieces : ℝ := 127.5
def edward_pieces : ℝ := 345.25
def sam_pieces : ℝ := 518.75

-- Define the total number of pieces of paper picked up
def total_pieces : ℝ := olivia_pieces + edward_pieces + sam_pieces

-- The theorem to be proven
theorem total_number_of_pieces_paper :
  total_pieces = 991.5 :=
by
  -- Sorry is used as we are not required to provide a proof here
  sorry

end total_number_of_pieces_paper_l794_794710


namespace largest_base5_to_base10_l794_794018

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794018


namespace first_term_of_sequence_l794_794945

theorem first_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end first_term_of_sequence_l794_794945


namespace cuberoot_sum_equals_4_l794_794906

noncomputable def cuberoot_sum : ℝ :=
  Real.cbrt(8 + 3 * Real.sqrt 21) + Real.cbrt(8 - 3 * Real.sqrt 21)

theorem cuberoot_sum_equals_4 :
  cuberoot_sum = 4 :=
sorry

end cuberoot_sum_equals_4_l794_794906


namespace fish_population_estimate_l794_794403

theorem fish_population_estimate (N : ℕ) 
    (h1 : 40 ≤ N) 
    (h2 : (40 : ℚ) / N = (2 : ℚ) / 40) : 
  N = 800 :=
begin
  sorry
end

end fish_population_estimate_l794_794403


namespace trigonometric_inequality_l794_794938

noncomputable def interior_angle := ℝ -- Assuming the angles are in the reals for generality

theorem trigonometric_inequality 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (A B C : interior_angle)
  (hA : A + B + C = π) :
  x * Real.sin A + y * Real.sin B + z * Real.sin C 
  ≤ (1 / 2) * (x * y + y * z + z * x) * Real.sqrt ((x + y + z) / (x * y * z)) :=
by 
  sorry

end trigonometric_inequality_l794_794938


namespace area_triangle_AEB_l794_794643

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    [rect : Rectangle ABCD]
    (AB : Length (side AB) = 9)
    (BC : Length (side BC) = 4)
    (DF : Points F G on side CD)
    (GC : Length (segment DF) = 2)
    (GC : Length (segment GC) = 1)
    (E : Lines Intersection A F B G at E),
  ∃ (area : ℝ),
    area = 27 := sorry

end area_triangle_AEB_l794_794643


namespace cookies_without_ingredients_l794_794311

theorem cookies_without_ingredients (total_cookies : ℕ) (chocolate_chip_cookies : ℕ) (raisin_cookies : ℕ) (almond_cookies : ℕ) (coconut_flake_cookies : ℕ) :
  total_cookies = 48 → chocolate_chip_cookies = 16 → raisin_cookies = 24 → almond_cookies = 18 → coconut_flake_cookies = 12 →
  ∃ (cookies_without_any : ℕ), cookies_without_any = 24 :=
by
  intro htotal hchocolate hraisin halmond hcoconut
  use 24
  sorry

end cookies_without_ingredients_l794_794311


namespace hyperbola_C_equation_is_correct_minimum_PM_distance_is_correct_l794_794207

noncomputable def hyperbola_C_equation (C : ℝ → ℝ → Prop) : Prop :=
  C x y ↔ x^2 / 4 - y^2 = 1

theorem hyperbola_C_equation_is_correct :
  ∃ (C : ℝ → ℝ → Prop), (∀ x y, C x y ↔ x^2 / 4 - y^2 = 1) :=
begin
  use (λ x y, x^2 / 4 - y^2 = 1),
  intro x,
  intro y,
  simp,
end

noncomputable def minimum_PM_distance (PM_distance : ℝ → ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (x_0 y_0 : ℝ), x_0^2 / 4 - y_0^2 = 1 → PM_distance x_0 y_0 5 0 = 2

theorem minimum_PM_distance_is_correct :
  (∀ (x_0 y_0 : ℝ), x_0^2 / 4 - y_0^2 = 1 → (sqrt ((5/4) * (x_0 - 4)^2 + 4)) = 2) :=
begin
  sorry
end

end hyperbola_C_equation_is_correct_minimum_PM_distance_is_correct_l794_794207


namespace base5_to_base10_max_l794_794025

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794025


namespace sum_of_solutions_l794_794515

theorem sum_of_solutions :
  ( ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → 
  ( -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ) ) → 
  ∑ x in {real.sqrt 3, -real.sqrt 3}, x = 0 := 
by
sorry

end sum_of_solutions_l794_794515


namespace min_area_triangle_l794_794679

theorem min_area_triangle : ∃ p q : ℤ, ∀ a b : ℕ, a = 24 ∧ b = 10 → 
  let area : ℝ := 1/2 * (abs (a * ↑q + b * ↑p)) in area = 1 :=
by
  sorry

end min_area_triangle_l794_794679


namespace number_of_students_l794_794930

theorem number_of_students 
    (N : ℕ) 
    (h_percentage_5 : 28 * N % 100 = 0)
    (h_percentage_4 : 35 * N % 100 = 0)
    (h_percentage_3 : 25 * N % 100 = 0)
    (h_percentage_2 : 12 * N % 100 = 0)
    (h_class_limit : N ≤ 4 * 30) 
    (h_num_classes : 4 * 30 < 120)
    : N = 100 := 
by 
  sorry

end number_of_students_l794_794930


namespace H_double_prime_coordinates_l794_794318

/-- Define the points of the parallelogram EFGH and their reflections. --/
structure Point := (x : ℝ) (y : ℝ)

def E : Point := ⟨3, 4⟩
def F : Point := ⟨5, 7⟩
def G : Point := ⟨7, 4⟩
def H : Point := ⟨5, 1⟩

/-- Reflection of a point across the x-axis changes the y-coordinate sign. --/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Reflection of a point across y=x-1 involves translation and reflection across y=x. --/
def reflect_y_x_minus_1 (p : Point) : Point :=
  let translated := Point.mk p.x (p.y + 1)
  let reflected := Point.mk translated.y translated.x
  Point.mk reflected.x (reflected.y - 1)

def H' : Point := reflect_x H
def H'' : Point := reflect_y_x_minus_1 H'

theorem H_double_prime_coordinates : H'' = ⟨0, 4⟩ :=
by
  sorry

end H_double_prime_coordinates_l794_794318


namespace sum_of_solutions_l794_794511

def equation (x : ℝ) : Prop := -12 * x / ((x + 1) * (x - 1)) = 3 * x / (x + 1) - 9 / (x - 1)

theorem sum_of_solutions : 
    let solutions := {x : ℝ | equation x}
    (∑ x in solutions, x) = 0 :=
by {
    sorry
}

end sum_of_solutions_l794_794511


namespace sum_of_coefficients_l794_794147

noncomputable def expand_and_sum_coefficients (d : ℝ) : ℝ :=
  let poly := -2 * (4 - d) * (d + 3 * (4 - d))
  let expanded := -4 * d^2 + 40 * d - 96
  let sum_coefficients := (-4) + 40 + (-96)
  sum_coefficients

theorem sum_of_coefficients (d : ℝ) : expand_and_sum_coefficients d = -60 := by
  sorry

end sum_of_coefficients_l794_794147


namespace correct_number_of_transformations_l794_794884

structure Pattern :=
  (l : ℝ) -- Length of the line segment
  (p : ℝ → ℝ) -- Position function for triangles and circles
  (end_half_circles : Bool) -- Boolean indicating the presence of end half-circles

def number_of_rigid_motion_transformations (P : Pattern) : ℕ :=
  let rotation := false -- Rotation does not preserve the pattern
  let translation := false -- Translation does not preserve the pattern
  let reflection_l := true -- Reflection across line l preserves the pattern
  let reflection_perpendicular := true -- Reflection across a perpendicular line preserves the pattern
  [rotation, translation, reflection_l, reflection_perpendicular].count id

theorem correct_number_of_transformations (P : Pattern) :
  P.end_half_circles →
  number_of_rigid_motion_transformations P = 2 :=
by
  intro h
  dsimp [number_of_rigid_motion_transformations]
  -- [false, false, true, true].count id = 2
  simp [List.count]
  exact rfl

end correct_number_of_transformations_l794_794884


namespace maximum_distance_l794_794097

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def square_side_length := 2

def distance_condition (u v w : ℝ) : Prop := 
  u^2 + v^2 = 2 * w^2

theorem maximum_distance 
  (x y : ℝ) 
  (h1 : point_distance x y 0 0 = u) 
  (h2 : point_distance x y 2 0 = v) 
  (h3 : point_distance x y 2 2 = w)
  (h4 : distance_condition u v w) :
  ∃ (d : ℝ), d = point_distance x y 0 2 ∧ d = 2 * Real.sqrt 5 := sorry

end maximum_distance_l794_794097


namespace minimum_value_of_sine_function_l794_794769

def f (x : ℝ) := Real.sin (2 * x - Real.pi / 4)

theorem minimum_value_of_sine_function : 
  ∃ x ∈ Icc 0 (Real.pi / 2), f x = -Real.sqrt 2 / 2 := 
sorry

end minimum_value_of_sine_function_l794_794769


namespace modulus_complex_number_l794_794359

theorem modulus_complex_number (i : ℂ) (h : i = Complex.I) : 
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end modulus_complex_number_l794_794359


namespace log_inverse_point_l794_794356

theorem log_inverse_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  ((∃ x y : ℝ, x = 4 ∧ y = 1 ∧ y = log a (x - 1)) → a = 3) :=
sorry

end log_inverse_point_l794_794356


namespace family_spent_36_dollars_l794_794424

def ticket_cost : ℝ := 5

def popcorn_cost : ℝ := 0.8 * ticket_cost

def soda_cost : ℝ := 0.5 * popcorn_cost

def tickets_bought : ℕ := 4

def popcorn_bought : ℕ := 2

def sodas_bought : ℕ := 4

def total_spent : ℝ :=
  (tickets_bought * ticket_cost) +
  (popcorn_bought * popcorn_cost) +
  (sodas_bought * soda_cost)

theorem family_spent_36_dollars : total_spent = 36 := by
  sorry

end family_spent_36_dollars_l794_794424


namespace radius_of_circles_in_triangle_l794_794946

theorem radius_of_circles_in_triangle :
  let a := 13
      b := 14
      c := 15
      r := 260 / 129
  in
  ∃ O O₁ O₂ O₃ : Type,
    let tangent_to_sides (circle : Type) (side1 side2 : Type) := true in
    (tangent_to_sides O₁ AB AC) ∧
    (tangent_to_sides O₂ BA BC) ∧
    (tangent_to_sides O₃ CB CA) ∧
    (tangent_to_sides O O₁ O₂) ∧
    (tangent_to_sides O O₂ O₃) ∧
    (tangent_to_sides O O₃ O₁) ∧
    (radius O = r) :=
sorry

end radius_of_circles_in_triangle_l794_794946


namespace arccos_zero_eq_pi_div_two_l794_794465

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l794_794465


namespace solution_set_of_inequality_l794_794372

theorem solution_set_of_inequality : {x : ℝ | abs (2 - x) < 1} = set.Ioo 1 3 :=
by 
-- Add proof steps here
  sorry

end solution_set_of_inequality_l794_794372


namespace arithmetic_sequence_a5_l794_794260

variable {α : Type*}

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ m n p, n - m = p - n → a n = (a m + a p) / 2

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h1 : a 1 + a 9 = 10) : a 5 = 5 :=
by 
  sorry

end arithmetic_sequence_a5_l794_794260


namespace monotonicity_f_inequality_f_when_a_is_1_l794_794215

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - a * Real.log x + (1 - a) * x + 1

theorem monotonicity_f (a : ℝ) :
  (∀ x > 0, f'(x, a) > 0) ∨ (∀ x > a, f'(x, a) > 0 ∧ ∀ x > 0, x < a → f'(x, a) < 0) :=
sorry

theorem inequality_f_when_a_is_1 :
  ∀ x > 0, f(x, 1) ≤ x * (Real.exp x - 1) + (1 / 2) * x^2 - 2 * Real.log x :=
sorry

def f' (x a : ℝ) : ℝ := 
  x - a / x + 1 - a

end monotonicity_f_inequality_f_when_a_is_1_l794_794215


namespace factorial_difference_l794_794132

theorem factorial_difference : 8! - 7! = 35280 := 
by sorry

end factorial_difference_l794_794132


namespace distinct_integer_solutions_abs_sum_eq_20_l794_794709

theorem distinct_integer_solutions_abs_sum_eq_20 :
  (∑ x in finset.Icc (-20) 20, ∑ y in finset.Icc (-20) 20, (|x| + |y| = 20 : Prop).to_finset.card) = 80 :=
sorry

end distinct_integer_solutions_abs_sum_eq_20_l794_794709


namespace seq_general_formula_l794_794944

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

theorem seq_general_formula {a : ℕ → ℝ} (h1 : a 1 = 1) (h2 : seq a) :
  ∀ n, a n = 1 / 2 ^ (n - 1) :=
by
  sorry

end seq_general_formula_l794_794944


namespace marble_sharing_l794_794705

theorem marble_sharing 
  (total_marbles : ℕ) 
  (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 30) 
  (h2 : marbles_per_friend = 6) : 
  total_marbles / marbles_per_friend = 5 := 
by 
  sorry

end marble_sharing_l794_794705


namespace find_x0_l794_794974

def f : ℝ → ℝ :=
  λ x, if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else if x > 2 then 2 * x else -1  -- as a type guard for unexpected values

theorem find_x0 (x0 : ℝ) (h : f x0 = -2) : x0 = Real.sqrt 2 :=
by
  sorry

end find_x0_l794_794974


namespace fewest_tiles_needed_l794_794431

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end fewest_tiles_needed_l794_794431


namespace solve_system_of_equations_real_l794_794741

theorem solve_system_of_equations_real (x y : ℝ) :
  (x^2 + x * y + y^2 = 4) ∧ (x^4 + x^2 * y^2 + y^4 = 8) ↔
  (x =  sqrt((3 + sqrt 5) / 2) ∧ y =  sqrt((3 - sqrt 5) / 2)) ∨
  (x = -sqrt((3 + sqrt 5) / 2) ∧ y =  sqrt((3 - sqrt 5) / 2)) ∨
  (x =  sqrt((3 - sqrt 5) / 2) ∧ y =  sqrt((3 + sqrt 5) / 2)) ∨
  (x = -sqrt((3 - sqrt 5) / 2) ∧ y =  sqrt((3 + sqrt 5) / 2)) ∨
  (x =  sqrt((3 + sqrt 5) / 2) ∧ y = -sqrt((3 - sqrt 5) / 2)) ∨
  (x = -sqrt((3 + sqrt 5) / 2) ∧ y = -sqrt((3 - sqrt 5) / 2)) ∨
  (x =  sqrt((3 - sqrt 5) / 2) ∧ y = -sqrt((3 + sqrt 5) / 2)) ∨
  (x = -sqrt((3 - sqrt 5) / 2) ∧ y = -sqrt((3 + sqrt 5) / 2)) := 
sorry

end solve_system_of_equations_real_l794_794741


namespace Bruce_remaining_amount_l794_794117

/--
Given:
1. initial_amount: the initial amount of money that Bruce's aunt gave him, which is 71 dollars.
2. shirt_cost: the cost of one shirt, which is 5 dollars.
3. num_shirts: the number of shirts Bruce bought, which is 5.
4. pants_cost: the cost of one pair of pants, which is 26 dollars.
Show:
Bruce's remaining amount of money after buying the shirts and the pants is 20 dollars.
-/
theorem Bruce_remaining_amount
  (initial_amount : ℕ)
  (shirt_cost : ℕ)
  (num_shirts : ℕ)
  (pants_cost : ℕ)
  (total_amount_spent : ℕ)
  (remaining_amount : ℕ) :
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  total_amount_spent = shirt_cost * num_shirts + pants_cost →
  remaining_amount = initial_amount - total_amount_spent →
  remaining_amount = 20 :=
by
  intro h_initial h_shirt_cost h_num_shirts h_pants_cost h_total_spent h_remaining
  rw [h_initial, h_shirt_cost, h_num_shirts, h_pants_cost, h_total_spent, h_remaining]
  rfl

end Bruce_remaining_amount_l794_794117


namespace intersection_value_l794_794989

def line_l (t : ℝ) : ℝ × ℝ :=
  (5 + (real.sqrt 3)/2 * t, real.sqrt 3 + 1/2 * t)

def polar_eqn_to_cartesian (θ : ℝ) : (ℝ × ℝ) :=
  let ρ := 2 * real.cos θ
  in (ρ * real.cos θ, ρ * real.sin θ)

def point_M : ℝ × ℝ := (5, real.sqrt 3)

theorem intersection_value :
  let l (t : ℝ) := line_l t in
  let C (θ : ℝ) := polar_eqn_to_cartesian θ in
  -- Here you will need to properly formulate the intersection points A and B. For now, let's assume you have A, B found correctly via computing.
  -- We will write a placeholder.
  let A : ℝ × ℝ := (0,0) in -- Placeholder
  let B : ℝ × ℝ := (0,0) in -- Placeholder
  ∃ A B : ℝ × ℝ, (|point_M.1 - A.1| + |point_M.2 - A.2|) * (|point_M.1 - B.1| + |point_M.2 - B.2|) = 18 :=
sorry

end intersection_value_l794_794989


namespace sum_of_coordinates_of_reflected_midpoint_l794_794319

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem sum_of_coordinates_of_reflected_midpoint :
  let A : ℝ × ℝ := (3, 2)
  let B : ℝ × ℝ := (15, 18)
  let N := midpoint A B
  let A' := reflect_y A
  let B' := reflect_y B
  let N' := midpoint A' B'
  N'.1 + N'.2 = 1 := by
    sorry

end sum_of_coordinates_of_reflected_midpoint_l794_794319


namespace number_is_more_than_sum_l794_794398

theorem number_is_more_than_sum : 20.2 + 33.8 - 5.1 = 48.9 :=
by
  sorry

end number_is_more_than_sum_l794_794398


namespace find_k_l794_794350

theorem find_k (k : ℝ) (h : ∃ (k : ℝ), 3 = k * (-1) - 2) : k = -5 :=
by
  rcases h with ⟨k, hk⟩
  sorry

end find_k_l794_794350


namespace projection_is_constant_l794_794050

variables {a b c d : ℝ}
variables {v w p : ℝ × ℝ}

-- Defining conditions for the problem
def on_line (v : ℝ × ℝ) : Prop := v.2 = (3 / 2) * v.1 + 3
def projection (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2) • w

-- Variables for p and w
def w : ℝ × ℝ := (c, d)
def p : ℝ × ℝ := (-18 / 13, 12 / 13)

-- Theorem statement
theorem projection_is_constant (a b c d : ℝ) (v : ℝ × ℝ)
  (hv : on_line v) (hw : c + (3 / 2) * d = 0) :
  ∀ v : ℝ × ℝ, on_line v → projection v w = p :=
begin
  -- Proof will be inserted here
  sorry
end

end projection_is_constant_l794_794050


namespace value_v4_l794_794873

noncomputable def p (x : ℤ) : ℤ := 3 * x^6 + 5 * x^5 + 6 * x^4 + 20 * x^3 - 8 * x^2 + 35 * x + 12

def horner (coeffs : List ℤ) (x : ℤ) : List ℤ := 
  coeffs.foldr (λ a acc, match acc with
                          | [] => [a]
                          | h::t => (a + x * h)::acc)
               []

example : horner [12, 35, -8, 20, 6, 5, 3] (-2) = [12, 11, 14, -8, 4, -1, 3] := by
  rfl

example : (horner [12, 35, -8, 20, 6, 5, 3] (-2)).nth 3 = some 4 := by
  rfl

example : (horner [12, 35, -8, 20, 6, 5, 3] (-2)).nth 4 = some (-16) := by
  rfl

example : ((horner [12, 35, -8, 20, 6, 5, 3] (-2)).nth 4).getOrElse 0 = -16 := by
  rfl

theorem value_v4 : ((horner [12, 35, -8, 20, 6, 5, 3] (-2)).nth 4).getOrElse 0 = -16 := by
  rfl

end value_v4_l794_794873


namespace digit_156_of_fraction_47_over_777_is_9_l794_794393

theorem digit_156_of_fraction_47_over_777_is_9 :
  let r := 47 / 777 in
  let decimal_expansion := 0.0 * 10^0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5) + -- and so on, repeating every 5 digits as "60459"
  (r = 0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5)) ∧ -- and so on
  let d := 156 in
  decimal_expansion.nth_digit(d) = 9 :=
sorry

end digit_156_of_fraction_47_over_777_is_9_l794_794393


namespace sum_of_distinct_product_GH_l794_794361

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_single_digit (d : ℕ) : Prop :=
  d < 10

theorem sum_of_distinct_product_GH : 
  ∀ (G H : ℕ), 
    is_single_digit G ∧ is_single_digit H ∧ 
    divisible_by_45 (8620000307 + 10000000 * G + H) → 
    (if H = 5 then GH = 6 else if H = 0 then GH = 0 else GH = 0) := 
  sorry

-- Note: This is a simplified representation; tailored more complex conditions and steps may be encapsulated in separate definitions and theorems as needed.

end sum_of_distinct_product_GH_l794_794361


namespace line_intersects_xaxis_at_l794_794870

theorem line_intersects_xaxis_at (x y : ℝ) 
  (h : 4 * y - 5 * x = 15) 
  (hy : y = 0) : (x, y) = (-3, 0) :=
by
  sorry

end line_intersects_xaxis_at_l794_794870


namespace arccos_zero_l794_794475

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l794_794475


namespace carter_trip_duration_without_pit_stops_l794_794127

theorem carter_trip_duration_without_pit_stops
  (stretch_interval : ℝ := 2)
  (num_food_stops : ℕ := 2)
  (num_gas_stops : ℕ := 3)
  (stop_duration : ℝ := 20 / 60) -- in hours
  (total_trip_with_stops : ℝ := 18) :
  total_trip_with_stops - ((total_trip_with_stops / stretch_interval + num_food_stops + num_gas_stops) * stop_duration) = 13.33 := 
begin
  sorry
end

end carter_trip_duration_without_pit_stops_l794_794127


namespace solve_first_equation_solve_second_equation_l794_794330

theorem solve_first_equation (x : ℤ) : 4 * x + 3 = 5 * x - 1 → x = 4 :=
by
  intros h
  sorry

theorem solve_second_equation (x : ℤ) : 4 * (x - 1) = 1 - x → x = 1 :=
by
  intros h
  sorry

end solve_first_equation_solve_second_equation_l794_794330


namespace no_integer_solutions_l794_794155

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l794_794155


namespace three_correct_propositions_l794_794979

def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

def proposition1 : Prop := 
  let y := f (x + Real.pi / 6) 
  ∀ x, y = 4 * Real.sin (2 * x) ∧ 
       ∀ t, y t = - y (-t)

def proposition2 : Prop := 
  ∀ x, f (x - Real.pi / 3) ≠ 4 * Real.sin (2 * x)

def proposition3 : Prop := 
  ∀ x, f (x) = f (-(x + Real.pi / 12))

def proposition4 : Prop := 
  ∀ x, 0 ≤ x ∧ x ≤ (5 * Real.pi / 12) → 
     (2 * x - Real.pi / 3 ≤ Real.pi / 2) ∧ 
     (0 ≤ 2 * x) ∧
     (2 * x - Real.pi / 3 ≥ - Real.pi / 3)

theorem three_correct_propositions :
  (proposition1 ∨ ¬proposition1) ∧
  (proposition2 ∨ ¬proposition2) ∧
  (proposition3 ∨ ¬proposition3) ∧
  (proposition4 ∨ ¬proposition4) →
  3 = [proposition1, proposition2, proposition3, proposition4].count true :=
by
  sorry

end three_correct_propositions_l794_794979


namespace total_games_l794_794096

theorem total_games (n m d1 d2 : ℕ) 
  (h1 : n = 16) 
  (h2 : d1 = 8)
  (h3 : d2 = 8)
  (h4 : n = d1 + d2)
  (h5 : ∀ t : ℕ, t < d1 → (team_plays_intra_division t = 7 * 3)) 
  (h6 : ∀ t : ℕ, t < d2 → (team_plays_inter_division t = 8 * 2)) 
  : n * ((7 * 3) + (8 * 2)) / 2 = 296 := 
by
  sorry

end total_games_l794_794096


namespace son_age_is_26_l794_794815

-- Definitions based on conditions in the problem
variables (S F : ℕ)
axiom cond1 : F = S + 28
axiom cond2 : F + 2 = 2 * (S + 2)

-- Statement to prove that S = 26
theorem son_age_is_26 : S = 26 :=
by 
  -- Proof steps go here
  sorry

end son_age_is_26_l794_794815


namespace zero_ordering_l794_794219

noncomputable def f (x : ℝ) : ℝ := 2^x + x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2 + x
noncomputable def h (x : ℝ) : ℝ := Real.log x / Real.log 2 - 2

lemma zero_of_f_lt_zero (a : ℝ) : f a = 0 → a < 0 :=
by sorry

lemma zero_of_g_between_zero_and_one (b : ℝ) : g b = 0 → 0 < b ∧ b < 1 :=
by sorry

lemma zero_of_h_eq_four (c : ℝ) : h c = 0 → c = 4 :=
by sorry

theorem zero_ordering (a b c : ℝ) (hfa : f a = 0) (hgb : g b = 0) (hhc : h c = 0) : a < b ∧ b < c :=
by {
  have ha : a < 0 := zero_of_f_lt_zero a hfa,
  have hb : 0 < b ∧ b < 1 := zero_of_g_between_zero_and_one b hgb,
  have hc : c = 4 := zero_of_h_eq_four c hhc,
  sorry
}

end zero_ordering_l794_794219


namespace math_problem_l794_794527

def f1 (x : ℝ) : ℝ := Real.log (abs (x - 2) + 1)
def f2 (x : ℝ) : ℝ := (x - 2)^2
def f3 (x : ℝ) : ℝ := Real.cos (x + 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem math_problem :
  (is_even (λ x, f2 (x + 2)) ∧ 
   is_decreasing_on f2 (-∞) 2 ∧ is_increasing_on f2 2 ∞ ∧ 
   is_increasing (λ x, f2 (x + 2) - f2 x)) ∧ 
  (¬ (is_even (λ x, f1 (x + 2)) ∧ 
     is_decreasing_on f1 (-∞) 2 ∧ is_increasing_on f1 2 ∞ ∧ 
     is_increasing (λ x, f1 (x + 2) - f1 x))) ∧ 
  (¬ (is_even (λ x, f3 (x + 2)) ∧ 
     is_decreasing_on f3 (-∞) 2 ∧ is_increasing_on f3 2 ∞ ∧ 
     is_increasing (λ x, f3 (x + 2) - f3 x))) := 
by 
  sorry

end math_problem_l794_794527


namespace sum_of_solutions_l794_794514

theorem sum_of_solutions :
  ( ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → 
  ( -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ) ) → 
  ∑ x in {real.sqrt 3, -real.sqrt 3}, x = 0 := 
by
sorry

end sum_of_solutions_l794_794514


namespace checkerboards_non_coverable_by_dominoes_l794_794139

theorem checkerboards_non_coverable_by_dominoes :
  ∀ B, B ∈ ({(5,5), (5,7), (7,3)} : set (ℕ × ℕ)) →
    ¬ ∃ (t : ℕ), t * (t + (t % 2) = B.1 * B.2) :=
by
  intro B
  intro B_in_set
  cases B with m n
  sorry

end checkerboards_non_coverable_by_dominoes_l794_794139


namespace sum_of_fractions_eq_sum_of_cubes_l794_794723

theorem sum_of_fractions_eq_sum_of_cubes (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  ( (x-1)*(x+1) / (x*(x-1) + 1) + (2*(0.5-x)) / (x*(1-x) -1) ) = 
  ( ((x-1)*(x+1) / (x*(x-1) + 1))^3 + ((2*(0.5-x)) / (x*(1-x) -1))^3 ) :=
sorry

end sum_of_fractions_eq_sum_of_cubes_l794_794723


namespace player_B_wins_l794_794991

theorem player_B_wins (n : ℕ) (h : 2 ≤ n) :
  ∃ (c : ℕ → ℝ), ∃ (x : ℝ), (∑ k in range(2*n - 1), c k * x^k) + x^(2*n) + 1 = 0 :=
by
  sorry

end player_B_wins_l794_794991


namespace largest_base5_number_to_base10_is_3124_l794_794003

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794003


namespace possible_scenario_l794_794197

variable {a b c d : ℝ}

-- Conditions
def abcd_positive : a * b * c * d > 0 := sorry
def a_less_than_c : a < c := sorry
def bcd_negative : b * c * d < 0 := sorry

-- Statement
theorem possible_scenario :
  (a < 0) ∧ (b > 0) ∧ (c < 0) ∧ (d > 0) :=
sorry

end possible_scenario_l794_794197


namespace largest_base5_number_conversion_l794_794010

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794010


namespace sequence_general_term_and_sum_sum_tn_bound_l794_794978

theorem sequence_general_term_and_sum (c : ℝ) (h₁ : c = 1) 
  (f : ℕ → ℝ) (hf : ∀ x, f x = (1 / 3) ^ x) :
  (∀ n, a_n = -2 / 3 ^ n) ∧ (∀ n, b_n = 2 * n - 1) :=
by {
  sorry
}

theorem sum_tn_bound (h₂ : ∀ n > 0, T_n = (1 / 2) * (1 - 1 / (2 * n + 1))) :
  ∃ n, T_n > 1005 / 2014 ∧ n = 252 :=
by {
  sorry
}

end sequence_general_term_and_sum_sum_tn_bound_l794_794978


namespace min_elements_of_B_l794_794988

def A (k : ℝ) : Set ℝ :=
if k < 0 then {x | (k / 4 + 9 / (4 * k) + 3) < x ∧ x < 11 / 2}
else if k = 0 then {x | x < 11 / 2}
else if 0 < k ∧ k < 1 ∨ k > 9 then {x | x < 11 / 2 ∨ x > k / 4 + 9 / (4 * k) + 3}
else if 1 ≤ k ∧ k ≤ 9 then {x | x < k / 4 + 9 / (4 * k) + 3 ∨ x > 11 / 2}
else ∅

def B (k : ℝ) : Set ℤ := {x : ℤ | ↑x ∈ A k}

theorem min_elements_of_B (k : ℝ) (hk : k < 0) : 
  B k = {2, 3, 4, 5} :=
sorry

end min_elements_of_B_l794_794988


namespace number_of_distinct_integers_from_special_fractions_sums_l794_794879

def is_special (a b : ℕ) : Prop := a + b = 15

def special_fractions : List ℚ :=
  (List.range 14).map (λ k => (k+1 : ℚ) / (15 - (k+1)))

def valid_sums (f g : ℚ) : Proposition :=
  (f + g).denom = 1

theorem number_of_distinct_integers_from_special_fractions_sums :
  (special_fractions.product special_fractions).filter (λ p => valid_sums p.1 p.2) .map (λ p => (p.1 + p.2).nat).erase_dup.length = 9 :=
sorry

end number_of_distinct_integers_from_special_fractions_sums_l794_794879


namespace rectangle_area_l794_794100

theorem rectangle_area (A : ℝ) (w : ℝ) (l : ℝ) (h1 : A = 36) (h2 : w^2 = A) (h3 : l = 3 * w) : w * l = 108 :=
by
sorrry

end rectangle_area_l794_794100


namespace coeff_x3_in_expansion_l794_794152

theorem coeff_x3_in_expansion : 
  ∃ c : ℕ, (c = 80) ∧ (∃ r : ℕ, r = 1 ∧ (2 * x + 1 / x) ^ 5 = (2 * x) ^ (5 - r) * (1 / x) ^ r)
:= sorry

end coeff_x3_in_expansion_l794_794152


namespace positive_divisors_840_multiple_of_4_l794_794242

theorem positive_divisors_840_multiple_of_4 :
  let n := 840
  let prime_factors := (2^3 * 3^1 * 5^1 * 7^1)
  (∀ k : ℕ, k ∣ n → k % 4 = 0 → ∀ a b c d : ℕ, 2 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 →
  k = 2^a * 3^b * 5^c * 7^d) → 
  (∃ count, count = 16) :=
by {
  sorry
}

end positive_divisors_840_multiple_of_4_l794_794242


namespace digits_condition_l794_794893

theorem digits_condition (z : ℕ) (hz : z ∈ {0, 1, 3, 7, 9}) : 
  ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ (n^9) % (10^k) = z * (10^k - 1) / 9 % (10^k) :=
by
  sorry

end digits_condition_l794_794893


namespace rectangle_area_is_108_l794_794098

-- Definitions derived from the conditions
def square_area : ℝ := 36
def rectangle_width : ℝ := real.sqrt square_area
def rectangle_length : ℝ := 3 * rectangle_width

-- Proof statement
theorem rectangle_area_is_108 : rectangle_width * rectangle_length = 108 := by
  sorry

end rectangle_area_is_108_l794_794098


namespace no_infinite_family_of_lines_exists_l794_794126

theorem no_infinite_family_of_lines_exists :
  ¬ ∃ (l : ℕ → ℝ) (a b : ℕ → ℝ),
  (∀ n : ℕ, n ≥ 1 → (l n = (λ n, a n - b n))) ∧
  (∀ n : ℕ, n ≥ 1 → ((1, 1) ∈ set.range (λ x, (x, l n x)))) ∧
  (∀ n : ℕ, (l n * l (n + 1)) ≥ 0) :=
sorry

end no_infinite_family_of_lines_exists_l794_794126


namespace sufficient_but_not_necessary_a_eq_2_l794_794899

theorem sufficient_but_not_necessary_a_eq_2 (x a : ℝ) (h : 15*a^2*x^4 = 60*x^4) : 
  (a = 2) ∨ (a = -2) :=
begin
  -- Assuming 15 * a^2 * x^4 = 60 * x^4, divide both sides by x^4 to get 15 * a^2 = 60.
  have ha : 15 * a^2 = 60,
  { 
    rw mul_eq_mul_right_iff at h,
    cases h, 
    { exact h.left },
    { have hx4 := mul_right_injective₀ (15 * a^2) (60:int) h.right, rw <-h.right at h, exact h.right },
  },
  -- Dividing both sides by 15, we obtain a^2 = 4.
  have ha2 : a^2 = 4 := by linarith,
  -- Hence, a is either 2 or -2.
  have : a = 2 ∨ a = -2 := by { exact eq_or_eq_neg_of_sq_eq_sq dec_trivial ha2 },
  exact this
end

end sufficient_but_not_necessary_a_eq_2_l794_794899


namespace sphere_packing_max_spheres_l794_794734

theorem sphere_packing_max_spheres (r : ℝ) (S : Sphere) (touching_spheres : ℕ) 
  (cond1 : r = 1) 
  (cond2 : ∀ (s : Sphere), s ∈ touching_spheres → s.radius = 1) 
  (cond3 : ∀ (s1 s2 : Sphere), s1 ∈ touching_spheres → s2 ∈ touching_spheres → s1 ≠ s2 → ¬(s1.interior ∩ s2.interior).nonempty):
  12 ≤ touching_spheres ∧ touching_spheres ≤ 14 :=
by
  sorry

end sphere_packing_max_spheres_l794_794734


namespace safe_ice_time_l794_794658

def ambient_temperature : ℝ := -10 -- in Celsius
def heat_loss_rate : ℝ := 200 -- in kJ/hr
def ice_thickness : ℝ := 0.1 -- in meters
def water_temperature : ℝ := 0 -- in Celsius
def latent_heat_fusion : ℝ := 330 -- in kJ/kg
def specific_heat_capacity : ℝ := 2100 -- in J/(kg*C)
def ice_density : ℝ := 900 -- in kg/m^3
def freeze_to_temperature : ℝ := -5 -- average temperature in Celsius

theorem safe_ice_time :
  ∃ (t : ℝ), t = 153.225 ∧
    let m := ice_density * 1 * ice_thickness in
    let Q := (latent_heat_fusion * m) + 
             (specific_heat_capacity / 1000 * m * (water_temperature - freeze_to_temperature)) in
    Q / heat_loss_rate = t :=
by sorry

end safe_ice_time_l794_794658


namespace pf1_pf2_dot_product_l794_794957

def ellipse_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2 / 4 + y^2 / 3 = 1)

def foci_condition (F1 F2 : ℝ × ℝ) : Prop :=
  let (F1x, F1y) := F1 in
  let (F2x, F2y) := F2 in
  (F1x = -1) ∧ (F1y = 0) ∧ (F2x = 1) ∧ (F2y = 0)

def incircle_radius_condition (P F1 F2 : ℝ × ℝ) (r : ℝ) : Prop :=
  r = 1/2

noncomputable def dot_product (P F1 F2 : ℝ × ℝ) : ℝ :=
  let (Px, Py) := P in
  let (F1x, F1y) := F1 in
  let (F2x, F2y) := F2 in
  (F1x - Px) * (F2x - Px) + (F1y - Py) * (F2y - Py)

theorem pf1_pf2_dot_product (P F1 F2 : ℝ × ℝ) (r : ℝ) 
  (h₁ : ellipse_condition P) (h₂ : foci_condition F1 F2)
  (h₃ : incircle_radius_condition P F1 F2 r) : 
  dot_product P F1 F2 = 9 / 4 :=
by 
  sorry

end pf1_pf2_dot_product_l794_794957


namespace hyperbola_asymptotes_n_l794_794766

theorem hyperbola_asymptotes_n {y x : ℝ} (n : ℝ) (H : ∀ x y, (y^2 / 16) - (x^2 / 9) = 1 → y = n * x ∨ y = -n * x) : n = 4/3 :=
  sorry

end hyperbola_asymptotes_n_l794_794766


namespace lowest_possible_sale_price_percentage_l794_794858

noncomputable def list_price : ℝ := 80
noncomputable def max_initial_discount_percent : ℝ := 0.5
noncomputable def summer_sale_discount_percent : ℝ := 0.2
noncomputable def membership_discount_percent : ℝ := 0.1
noncomputable def coupon_discount_percent : ℝ := 0.05

theorem lowest_possible_sale_price_percentage :
  let max_initial_discount := max_initial_discount_percent * list_price
  let summer_sale_discount := summer_sale_discount_percent * list_price
  let membership_discount := membership_discount_percent * list_price
  let coupon_discount := coupon_discount_percent * list_price
  let lowest_sale_price := list_price * (1 - max_initial_discount_percent) - summer_sale_discount - membership_discount - coupon_discount
  (lowest_sale_price / list_price) * 100 = 15 :=
by
  sorry

end lowest_possible_sale_price_percentage_l794_794858


namespace inscribed_circle_radius_eq_l794_794895

noncomputable def inscribedCircleRadius :=
  let AB := 6
  let AC := 7
  let BC := 8
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  r

theorem inscribed_circle_radius_eq :
  inscribedCircleRadius = Real.sqrt 413.4375 / 10.5 := by
  sorry

end inscribed_circle_radius_eq_l794_794895


namespace number_of_ordered_triples_lcm_l794_794243

theorem number_of_ordered_triples_lcm :
  ∃ n : ℕ, n = 4 ∧ (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → 
  lcm x y = 180 → lcm x z = 420 → lcm y z = 1260 → 
  true) :=
begin
  use 4,
  split,
  { refl },
  { intros x y z hx hy hz h1 h2 h3,
    -- Placeholder for the actual proof
    trivial
  }
end

end number_of_ordered_triples_lcm_l794_794243


namespace largest_base_5_five_digit_number_in_decimal_l794_794030

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794030


namespace solution_l794_794692

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l794_794692


namespace find_eigenvalues_and_eigenvectors_l794_794221

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![-1, 4]]

theorem find_eigenvalues_and_eigenvectors :
  ∃ (λ1 λ2 : ℝ) (v1 v2 : Fin 2 → ℝ),
    (λ1, λ2) = (2, 3) ∧ 
    Matrix.hasEigenvector matrix_A λ1 v1 ∧ 
    Matrix.hasEigenvector matrix_A λ2 v2 ∧ 
    v1 = ![2, -1] ∧ 
    v2 = ![1, -1] := 
sorry

end find_eigenvalues_and_eigenvectors_l794_794221


namespace rational_point_general_exceptional_positions_l794_794849

noncomputable theory

structure Point :=
  (x : ℚ)
  (y : ℚ)

def distance (A B : Point) : ℚ :=
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def similar_triangles (A B A' B' P : Point) : Prop :=
  distance A' B' / distance A B = distance B' P / distance B P ∧ distance B' P / distance B P = distance P A' / distance P A

theorem rational_point_general (A B A' B' : Point) :
  (similar_triangles A B A' B' P) → ∃ P : Point, true :=
sorry

theorem exceptional_positions (A B A' B' : Point) :
  similar_triangles A B A' B' P → ¬ (∃ P : Point, true) ↔ irrational (distance A' B' / distance A B) :=
sorry

end rational_point_general_exceptional_positions_l794_794849


namespace trigonometric_identity_l794_794188

theorem trigonometric_identity (α : Real) (h : Real.sin α = -3/5) :
  (Real.sin (-α - 3 * Real.pi / 2) * Real.sin (3 * Real.pi / 2 - α) *
   Real.tan (2 * Real.pi - α) ^ 2) /
  (Real.cos (Real.pi / 2 - α) * Real.cos (Real.pi / 2 + α) * Real.cot (Real.pi - α)) = 3/4 :=
by
  sorry

end trigonometric_identity_l794_794188


namespace area_PQRS_is_eight_l794_794363

-- Definition of the reflections and the quadrilateral
structure Point :=
  (x : ℤ) (y : ℤ)

def reflect_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def reflect_line_y_eq_x (p : Point) : Point :=
  {x := p.y, y := p.x}

def reflect_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def area_quadrilateral (p1 p2 p3 p4 : Point) : ℚ :=
  let T := {x := p3.x, y := p1.y}
  let area_triangle (a b c : Point) : ℚ :=
    (abs ((a.x * b.y + b.x * c.y + c.x * a.y) -
          (a.y * b.x + b.y * c.x + c.y * a.x))) / 2
  (area_triangle p1 S T) + (area_triangle p2 p3 T)

-- Given points
def P : Point := {x := -1, y := 4}
def Q : Point := reflect_y_axis P
def R : Point := reflect_line_y_eq_x Q
def S : Point := reflect_x_axis R

-- Theorem to prove the area of quadrilateral PQRS
theorem area_PQRS_is_eight : area_quadrilateral P Q R S = 8 := 
  sorry

end area_PQRS_is_eight_l794_794363


namespace Abby_has_17_quarters_l794_794863

theorem Abby_has_17_quarters (q n : ℕ) (h1 : q + n = 23) (h2 : 25 * q + 5 * n = 455) : q = 17 :=
sorry

end Abby_has_17_quarters_l794_794863


namespace percentage_of_products_by_m1_l794_794634

theorem percentage_of_products_by_m1
  (x : ℝ)
  (h1 : 30 / 100 > 0)
  (h2 : 3 / 100 > 0)
  (h3 : 1 / 100 > 0)
  (h4 : 7 / 100 > 0)
  (h_total_defective : 
    0.036 = 
      (0.03 * x / 100) + 
      (0.01 * 30 / 100) + 
      (0.07 * (100 - x - 30) / 100)) :
  x = 40 :=
by
  sorry

end percentage_of_products_by_m1_l794_794634


namespace arccos_zero_eq_pi_div_two_l794_794467

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l794_794467


namespace smallest_integer_x_l794_794806

theorem smallest_integer_x (x : ℤ) :
  (∃ n : ℤ, n > 3.62 ∧ n = x) ↔ x = 4 := by sorry

end smallest_integer_x_l794_794806


namespace minimum_value_of_a_b_l794_794931

theorem minimum_value_of_a_b 
  (a b : ℝ) 
  (h : log a b = -2) : 
  a + b = 3 * real.cbrt 2 / 2 := 
sorry

end minimum_value_of_a_b_l794_794931


namespace modulus_z3_z5_l794_794505

noncomputable theory

open Complex

theorem modulus_z3_z5 (α : ℝ) (hα : α ∈ set.Ioo π (3 * π / 2)) : 
  complex.abs ((cos α + sin α * I)^3 + (cos α + sin α * I)^5) = 2 * abs (cos α) :=
sorry

end modulus_z3_z5_l794_794505


namespace problem_solution_l794_794400

noncomputable def f1 (x : ℝ) := real.sqrt (-2 * x^3)
noncomputable def g1 (x : ℝ) := x * real.sqrt (-2 * x)

noncomputable def f2 (x : ℝ) := abs x
noncomputable def g2 (x : ℝ) := real.sqrt (x^2)

noncomputable def f3 (x : ℝ) := x^0
noncomputable def g3 (x : ℝ) := 1 / x^0

noncomputable def f4 (x : ℝ) := x^2 - 2*x - 1
noncomputable def g4 (t : ℝ) := t^2 - 2*t - 1

theorem problem_solution :
  ¬(∀ x : ℝ, f1 x = g1 x) ∧
  (∀ x : ℝ, f2 x = g2 x) ∧
  (∀ x : ℝ, f3 x = g3 x) ∧
  (∀ x : ℝ, f4 x = g4 x) := by
sorry

end problem_solution_l794_794400


namespace metro_station_closure_l794_794639

theorem metro_station_closure (G : SimpleGraph (fin n)) [G.Connected] :
  ∃ s : fin n, G.Subgraph_fair s → G.Subgraph_fair (fin n \ s) :=
sorry

end metro_station_closure_l794_794639


namespace total_corn_yield_l794_794790

/-- 
The total corn yield in centners, harvested from a certain field area, is expressed 
as a four-digit number composed of the digits 0, 2, 3, and 5. When the average 
yield per hectare was calculated, it was found to be the same number of centners 
as the number of hectares of the field area. 
This statement proves that the total corn yield is 3025. 
-/
theorem total_corn_yield : ∃ (Y A : ℕ), (Y = A^2) ∧ (A >= 10 ∧ A < 100) ∧ 
  (Y / 1000 != 0) ∧ (Y / 1000 != 1) ∧ (Y / 10 % 10 != 4) ∧ 
  (Y % 10 != 1) ∧ (Y % 10 = 0 ∨ Y % 10 = 5) ∧ 
  (Y / 100 % 10 == 0 ∨ Y / 100 % 10 == 2 ∨ Y / 100 % 10 == 3 ∨ Y / 100 % 10 == 5) ∧ 
  Y = 3025 := 
by 
  sorry

end total_corn_yield_l794_794790


namespace area_triangle_ABC_l794_794394

noncomputable def triangle_area_ABC
  (A B C D : Type) 
  [plane_geom : EuclideanGeom A B C D] 
  (right_angle_at_D : ∠ D = 90°)
  (AC_eq : distance A C = 10)
  (AB_eq : distance A B = 26)
  (DC_eq : distance D C = 6) : ℝ :=
4 + 8*√153

theorem area_triangle_ABC : 
  triangle_area_ABC A B C D right_angle_at_D AC_eq AB_eq DC_eq = 4 + 8*√153 := by
  sorry

end area_triangle_ABC_l794_794394


namespace gcd_lcm_product_l794_794167

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l794_794167


namespace sum_of_solutions_l794_794509

def equation (x : ℝ) : Prop := -12 * x / ((x + 1) * (x - 1)) = 3 * x / (x + 1) - 9 / (x - 1)

theorem sum_of_solutions : 
    let solutions := {x : ℝ | equation x}
    (∑ x in solutions, x) = 0 :=
by {
    sorry
}

end sum_of_solutions_l794_794509


namespace one_hundred_fifty_sixth_digit_is_five_l794_794390

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end one_hundred_fifty_sixth_digit_is_five_l794_794390


namespace circle_line_minimum_distance_l794_794582

theorem circle_line_minimum_distance
  (l : ℝ → ℝ → Prop)
  (C : ℝ → ℝ → ℝ → Prop)
  (k : ℝ)
  (h_l : ∀ (ρ θ : ℝ), l ρ θ ↔ ρ * sin (θ - π / 4) = 4)
  (h_C : ∀ (ρ θ : ℝ), C ρ θ k ↔ ρ = 2 * k * cos (θ + π / 4))
  (h_k : k ≠ 0)
  (h_dist : ∀ (x y : ℝ), l x y → C x y k → dist (x, y) l = 2) :
  (∃ (x y : ℝ), C x y k ∧ x = - (sqrt 2) / 2 ∧ y = (sqrt 2) / 2) ∧ k = -1 :=
by sorry

end circle_line_minimum_distance_l794_794582


namespace find_m_value_l794_794557

-- Definitions based on conditions
variables {a b m : ℝ} (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1)

-- Lean 4 statement of the problem
theorem find_m_value (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1) : m = 10 := sorry

end find_m_value_l794_794557


namespace solution_set_for_fe_lt_zero_l794_794194

theorem solution_set_for_fe_lt_zero
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (h_solution : {x | f x > 0} = {x | x < 1 ∨ x > exp(1)}) :
  {x | f (exp x) < 0} = {x | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_for_fe_lt_zero_l794_794194


namespace proof_of_intersection_l794_794310

open Set

theorem proof_of_intersection :
  let U := ℝ
  let M := compl { x : ℝ | x^2 > 4 }
  let N := { x : ℝ | 1 < x ∧ x ≤ 3 }
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := by
sorry

end proof_of_intersection_l794_794310


namespace curve_intersects_x_axis_at_4_over_5_l794_794644

-- Define the function for the curve
noncomputable def curve (x : ℝ) : ℝ :=
  (3 * x - 1) * (Real.sqrt (9 * x ^ 2 - 6 * x + 5) + 1) +
  (2 * x - 3) * (Real.sqrt (4 * x ^ 2 - 12 * x + 13) + 1)

-- Prove that curve(x) = 0 when x = 4 / 5
theorem curve_intersects_x_axis_at_4_over_5 :
  curve (4 / 5) = 0 :=
by
  sorry

end curve_intersects_x_axis_at_4_over_5_l794_794644


namespace combined_distance_is_12_l794_794385

-- Define the distances the two ladies walked
def distance_second_lady : ℝ := 4
def distance_first_lady := 2 * distance_second_lady

-- Define the combined total distance
def combined_distance := distance_first_lady + distance_second_lady

-- Statement of the problem as a proof goal in Lean
theorem combined_distance_is_12 : combined_distance = 12 :=
by
  -- Definitions required for the proof
  let second := distance_second_lady
  let first := distance_first_lady
  let total := combined_distance
  
  -- Insert the necessary calculations and proof steps here
  -- Conclude with the desired result
  sorry

end combined_distance_is_12_l794_794385


namespace intervals_of_monotonicity_tangent_line_equation_l794_794576

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def g (x a : ℝ) : ℝ := x * Real.log x - a * (x - 1)

theorem intervals_of_monotonicity (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < Real.exp (a - 1) → g x a < g (Real.exp (a - 1)) a) ∧ 
  (∀ x : ℝ, x > Real.exp (a - 1) → g x a > g (Real.exp (a - 1)) a) := 
by
  sorry

theorem tangent_line_equation : ∃ (x : ℝ), x = 1 ∧ (∀ (p: ℝ), p != 0 → f(x) = 0 ∧ (f(p :: 0) = x - 1)) := 
by
  sorry

end intervals_of_monotonicity_tangent_line_equation_l794_794576


namespace min_distance_circle_point_l794_794565

theorem min_distance_circle_point :
  ∀ (x y : ℝ), (x^2 + y^2 + 4x - 2y + 4 = 0) → (∃ (m : ℝ), m = sqrt ((x - 1)^2 + y^2) ∧ m = sqrt 10 - 1) := 
by
  intros x y h
  sorry

end min_distance_circle_point_l794_794565


namespace solution_to_equation_l794_794784

theorem solution_to_equation:
  ∀ x : ℝ, (sqrt (x^2 + 6*x + 10) + sqrt (x^2 - 6*x + 10) = 8)
  ↔ (x = (4 * sqrt 42) / 7 ∨ x = -(4 * sqrt 42) / 7) := by
  sorry

end solution_to_equation_l794_794784


namespace unique_solution_for_equation_l794_794673

theorem unique_solution_for_equation (n : ℕ) (hn : 0 < n) (x : ℝ) (hx : 0 < x) :
  (n : ℝ) * x^2 + ∑ i in Finset.range n, ((i + 2)^2 / (x + (i + 1))) = 
  (n : ℝ) * x + (n * (n + 3) / 2) → x = 1 := 
by 
  sorry

end unique_solution_for_equation_l794_794673


namespace _l794_794299

variable {Ω : Type*} [ProbabilitySpace Ω]

def Cramers_theorem (ξ η : Ω → ℝ) (hξη : Indep ξ η) : 
  (IsGaussian (ξ + η) ↔ IsGaussian ξ ∧ IsGaussian η) :=
sorry

end _l794_794299


namespace parallel_NK_ML_l794_794227

-- Definitions of the points and circles should follow the conditions given
variables {A B C D M N K L : Type}
variables [geometry A]
variables [geometry B]
variables [geometry C]
variables [geometry D]
variables [geometry M]
variables [geometry N]
variables [geometry K]
variables [geometry L]

-- Define trapezoid and parallel lines
def is_trapezoid (ABCD : Type) [parallel BC AD] : Prop
-- Define circles with diameters
def is_circle_with_diameter (C1 : Type) (A B : Type) : Prop :=
  diameter C1 A B

def is_circle_with_diameter (C2 : Type) (C D : Type) : Prop :=
  diameter C2 C D

-- Define intersection points
def is_intersection (p : Type) (AC : Type) (C1 : Type) : Prop
def is_intersection (p : Type) (BD : Type) (C1 : Type) : Prop
def is_intersection (p : Type) (AC : Type) (C2 : Type) : Prop
def is_intersection (p : Type) (BD : Type) (C2 : Type) : Prop

-- Conditions
variables (trapezoid_ABCD : is_trapezoid ABCD)
variables (circle_C1 : is_circle_with_diameter C1 A B)
variables (circle_C2 : is_circle_with_diameter C2 C D)
variables (intersection_M : is_intersection M AC C1)
variables (intersection_N : is_intersection N BD C1)
variables (intersection_K : is_intersection K AC C2)
variables (intersection_L : is_intersection L BD C2)
variables (neq_MA : M ≠ A)
variables (neq_NB : N ≠ B)
variables (neq_KC : K ≠ C)
variables (neq_LD : L ≠ D)

-- Goal
theorem parallel_NK_ML : is_parallel NK ML :=
sorry

end parallel_NK_ML_l794_794227


namespace initial_fliers_l794_794056

theorem initial_fliers (F : ℕ) 
  (morning_fraction : ℝ := 0.1)
  (afternoon_fraction : ℝ := 0.25)
  (remaining_fliers : ℕ := 1350)
  (assumption : F * (0.9 * 0.75) = 1350) :
  F = 2000 := 
by sorry

end initial_fliers_l794_794056


namespace probability_divisors_l794_794677

theorem probability_divisors (T : set ℕ) (a_1 a_2 a_3 : ℕ) (p q : ℕ) (relatively_prime : Nat.gcd p q = 1) (hT : T = {d | d ∣ 12^7}) (h_a1 : a_1 ∈ T) (h_a2 : a_2 ∈ T) (h_a3 : a_3 ∈ T) (h_prob : ∀ a1 a2 a3, a1 ∈ T ∧ a2 ∈ T ∧ a3 ∈ T → a1 ∣ a2 ∧ a2 ∣ a3 ↔ p / q = 17 / 360) :
  p = 17 := 
sorry -- Proof to be provided

end probability_divisors_l794_794677


namespace constant_term_in_expansion_l794_794248

noncomputable def integral_value : ℝ :=
  2 * ∫ x in 0..(Real.pi / 2), sqrt 2 * sin (x + Real.pi / 4)

theorem constant_term_in_expansion :
  let n := integral_value in
  n = 4 → (C (n : ℕ) 2) * 3 ^ 2 = 54 := by
  sorry

end constant_term_in_expansion_l794_794248


namespace find_share_of_A_l794_794813

noncomputable def investment_share_A (initial_investment_A initial_investment_B withdraw_A add_B after_months end_of_year_profit : ℝ) : ℝ :=
  let investment_months_A := (initial_investment_A * after_months) + ((initial_investment_A - withdraw_A) * (12 - after_months))
  let investment_months_B := (initial_investment_B * after_months) + ((initial_investment_B + add_B) * (12 - after_months))
  let total_investment_months := investment_months_A + investment_months_B
  let ratio_A := investment_months_A / total_investment_months
  ratio_A * end_of_year_profit

theorem find_share_of_A : 
  investment_share_A 3000 4000 1000 1000 8 630 = 240 := 
by 
  sorry

end find_share_of_A_l794_794813


namespace triangle_right_angle_l794_794622

variable {A B C a b c : ℝ}

theorem triangle_right_angle (h1 : Real.sin (A / 2) ^ 2 = (c - b) / (2 * c)) 
                             (h2 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : 
                             a^2 + b^2 = c^2 :=
by
  sorry

end triangle_right_angle_l794_794622


namespace tan_J_right_triangle_l794_794909

open Real

theorem tan_J_right_triangle
  (J I K : Point) 
  (JK : ℝ) (IK : ℝ) (IJ : ℝ)
  (h1 : JK = 24)
  (h2 : IK = 26)
  (h3 : IK^2 = IJ^2 + JK^2)
  (right_angle : ∠IJK = π/2) : 
  tan (atan (IJ / JK)) = 5 / 12 :=
by
  have h4 : IK * IK = IJ * IJ + JK * JK := by rw [h2, h1, h3]
  have h5 : IJ^2 = 676 - 576 := by rw [h4]
  let IJ := sqrt 100 
  sorry -- Proof will be constructed here

end tan_J_right_triangle_l794_794909


namespace lowest_score_l794_794749

theorem lowest_score 
    (mean_15 : ℕ → ℕ → ℕ → ℕ)
    (mean_13 : ℕ → ℕ → ℕ)
    (S15 : ℕ := mean_15 15 85)
    (S13 : ℕ := mean_13 13 87)
    (highest_score : ℕ := 105)
    (S_removed : ℕ := S15 - S13) :
    S_removed - highest_score = 39 := 
sorry

end lowest_score_l794_794749


namespace inequality_solution_l794_794739

noncomputable def solve_inequality : ℝ → (ℝ → Prop)
| a :=
  if ha : a = 0 then
    λ x, x > 1
  else if ha : a > 0 then
    λ x, if 0 < a ∧ a < 2 then
            1 < x ∧ x < 2 / a
          else if a = 2 then
            false
          else
            2 / a < x ∧ x < 1
  else
    λ x, x > 1 ∨ x < 2 / a

theorem inequality_solution (a : ℝ) (x : ℝ) : (ax^2 - (a + 2)x + 2 < 0) ↔ solve_inequality a x :=
by sorry

end inequality_solution_l794_794739


namespace binomial_sum_identity_l794_794595

theorem binomial_sum_identity (n r: ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  (∑ d in Finset.range (n - r + 2), nat.choose (n - r + 1) d * nat.choose (r - 1) (d - 1)) = nat.choose n r :=
sorry

end binomial_sum_identity_l794_794595


namespace multiple_of_15_bounds_and_difference_l794_794481

theorem multiple_of_15_bounds_and_difference :
  ∃ (n : ℕ), 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 :=
by
  sorry

end multiple_of_15_bounds_and_difference_l794_794481


namespace largest_base5_to_base10_l794_794015

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794015


namespace max_product_of_xy_l794_794249

open Real

theorem max_product_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) :
  x * y ≤ 1 / 16 := 
sorry

end max_product_of_xy_l794_794249


namespace smallest_whole_number_larger_than_sum_l794_794926

theorem smallest_whole_number_larger_than_sum : 
  let sum := (3 + 1/3) + (4 + 1/4) + (5 + 1/6) + (6 + 1/12) in
  Nat.find (λ n : ℕ, n > sum) = 19 :=
by
  let sum := (3 + 1/3) + (4 + 1/4) + (5 + 1/6) + (6 + 1/12)
  have h : sum = 18 + 5/6 := sorry  -- We do not provide the detailed calculations here
  have h2: 18 + 5/6 < 19 := sorry  -- This will be calculated to show 18+5/6 is less than 19
  have h3: 19 > 18 + 5/6 := sorry  -- This will confirm that 19 is greater than 18+5/6
  exact Nat.find_eq_iff.2 ⟨by linarith, λ m h', by linarith⟩

end smallest_whole_number_larger_than_sum_l794_794926


namespace sale_price_correct_l794_794370

noncomputable def sale_price_including_tax (CP : ℝ) (profit_percentage : ℝ) (tax_rate : ℝ) : ℝ :=
  let profit := (profit_percentage / 100) * CP
  let SP := CP + profit
  let sales_tax := (tax_rate / 100) * SP
  SP + sales_tax

theorem sale_price_correct {CP : ℝ} (hCP : CP = 531.03)
  {profit_percentage : ℝ} (hprofit : profit_percentage = 16)
  {tax_rate : ℝ} (htax : tax_rate = 10) :
  sale_price_including_tax CP profit_percentage tax_rate ≈ 677.59 :=
by
  rcases hCP with rfl
  rcases hprofit with rfl
  rcases htax with rfl
  sorry

end sale_price_correct_l794_794370


namespace min_value_f_l794_794212

noncomputable section

open Real 

def f (a x : ℝ) : ℝ := (exp x - a)^2 + (exp (-x) - a)^2

theorem min_value_f (a : ℝ) (h : 0 < a ∧ a < 2) : ∃ x : ℝ, f a x = 2*(a - 1)^2 :=
by
  use 0
  have t := 2
  have fa_0 := t^2 - 2*a*t + 2*a^2
  rw [min_value_f] at fa_0
  exact fa_0
-- The proof itself is omitted.
sorry

end min_value_f_l794_794212


namespace percentage_students_taking_music_l794_794794

theorem percentage_students_taking_music
  (total_students : ℕ)
  (students_take_dance : ℕ)
  (students_take_art : ℕ)
  (students_take_music : ℕ)
  (percentage_students_taking_music : ℕ) :
  total_students = 400 →
  students_take_dance = 120 →
  students_take_art = 200 →
  students_take_music = total_students - students_take_dance - students_take_art →
  percentage_students_taking_music = (students_take_music * 100) / total_students →
  percentage_students_taking_music = 20 :=
by
  sorry

end percentage_students_taking_music_l794_794794


namespace spacy_subsets_count_l794_794487

theorem spacy_subsets_count :
  let c : ℕ → ℕ := λ n, if n = 1 then 2
                        else if n = 2 then 3
                        else if n = 3 then 4
                        else c (n - 1) + c (n - 3) in
  c 15 = 406 :=
by
  -- We assume the necessary recurrence and initial conditions and conclude the goal directly
  sorry

end spacy_subsets_count_l794_794487


namespace model_height_l794_794443

-- Define the variables and given conditions
def h_actual : ℝ := 60 -- height of the actual lighthouse in meters
def V_actual : ℝ := 150000 -- volume of the actual tank in liters
def V_model : ℝ := 0.15 -- volume of the model tank in liters

-- Define the ratio of volumes and linear scale factor
def volume_ratio : ℝ := V_actual / V_model
def scale_factor : ℝ := real.cbrt volume_ratio

-- Define the height of the model lighthouse
def h_model : ℝ := h_actual / scale_factor

-- The theorem stating that the height of the model lighthouse should be 0.6 meters
theorem model_height : h_model = 0.6 :=
by
  sorry

end model_height_l794_794443


namespace problem_l794_794554

def p : Prop := ∀ x : ℝ, x < 1 → log (1 / 3) x < 0
def q : Prop := ∃ x0 : ℝ, x0 ^ 2 ≥ 2 ^ x0

theorem problem : p ∨ q :=
by
  sorry

end problem_l794_794554


namespace minimum_value_f_on_interval_l794_794919

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^3 / (Real.sin x) + (Real.sin x)^3 / (Real.cos x)

theorem minimum_value_f_on_interval : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 1 ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≥ 1 :=
by sorry

end minimum_value_f_on_interval_l794_794919


namespace proof_problem_l794_794193

variable {a b : ℝ}
variable (cond : sqrt a > sqrt b)

theorem proof_problem (h1 : a > b) (h2 : 0 ≤ a) (h3 : 0 ≤ b) :
  (a^2 > b^2) ∧
  ((b + 1) / (a + 1) > b / a) ∧
  (b + 1 / (b + 1) ≥ 1) :=
by
  sorry

end proof_problem_l794_794193


namespace mean_of_transformed_data_l794_794583

variable (a1 a2 a3 a4 a5 : ℝ)

def variance (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  1 / 5 * (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 - 80)

def mean (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  (a1 + a2 + a3 + a4 + a5) / 5

noncomputable def new_mean (a : ℝ) : ℝ :=
  2 * a + 1

theorem mean_of_transformed_data :
  variance a1 a2 a3 a4 a5 = 16 → mean a1 a2 a3 a4 a5 = 4 →
  new_mean 4 = 9 :=
by 
  intros h_variance h_mean
  sorry

end mean_of_transformed_data_l794_794583


namespace sufficient_but_not_necessary_condition_l794_794655

theorem sufficient_but_not_necessary_condition (A B C : ℝ) (T : Triangle) :
  T.isRightAngle C ↔ (T.angle A + T.angle B = 90 ∧ 
                      (cos (T.angle A) + sin (T.angle A) = cos (T.angle B) + sin (T.angle B)) :=
by sorry

end sufficient_but_not_necessary_condition_l794_794655


namespace find_added_number_l794_794847

def original_number : ℕ := 5
def doubled : ℕ := 2 * original_number
def resultant (added : ℕ) : ℕ := 3 * (doubled + added)
def final_result : ℕ := 57

theorem find_added_number (added : ℕ) (h : resultant added = final_result) : added = 9 :=
sorry

end find_added_number_l794_794847


namespace cube_inequality_l794_794190

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l794_794190


namespace maria_cupcakes_l794_794173

variable (initial : ℕ) (additional : ℕ) (remaining : ℕ)

theorem maria_cupcakes (h_initial : initial = 19) (h_additional : additional = 10) (h_remaining : remaining = 24) : initial + additional - remaining = 5 := by
  sorry

end maria_cupcakes_l794_794173


namespace largest_integral_x_l794_794918

theorem largest_integral_x (x : ℤ) : 
  (1 / 4 : ℝ) < (x / 7) ∧ (x / 7) < (7 / 11 : ℝ) → x ≤ 4 := 
  sorry

end largest_integral_x_l794_794918


namespace max_interesting_pairs_on_5x7_grid_with_9_cells_marked_l794_794711

def interesting_pairs (grid : list (list bool)) (row col : ℕ) : ℕ :=
  let marked_neighs := λ r c, if r < grid.length ∧ c < (grid.head).length ∧ (grid.nth r).get_or_else [] |>.nth c = some tt then 1 else 0
  (if row > 0 then marked_neighs (row - 1) col else 0) + -- above
  (if row + 1 < grid.length then marked_neighs (row + 1) col else 0) + -- below
  (if col > 0 then marked_neighs row (col - 1) else 0) + -- left
  (if col + 1 < (grid.head).length then marked_neighs row (col + 1) else 0) -- right

def interesting_pairs_in_grid (grid : list (list bool)) : ℕ :=
  grid.sum (λ row, row.sum (λ cell, interesting_pairs grid row cell))

theorem max_interesting_pairs_on_5x7_grid_with_9_cells_marked :
  ∀ (grid : list (list bool)), grid.length = 5 ∧ (grid.head).length = 7 ∧ grid.sum (λ row, row.count tt) = 9 →
  interesting_pairs_in_grid grid = 35 := 
sorry

end max_interesting_pairs_on_5x7_grid_with_9_cells_marked_l794_794711


namespace distance_sum_at_least_n_div_2_l794_794712

theorem distance_sum_at_least_n_div_2 (n : ℕ) (X : fin n → ℝ) (hX : ∀ i, 0 ≤ X i ∧ X i ≤ 1) :
  ∃ (P : ℝ), 0 ≤ P ∧ P ≤ 1 ∧ (∀ Y : fin n → ℝ, (∀ i, 0 ≤ Y i ∧ Y i ≤ 1) → 
  ∑ i in finset.univ, abs (Y i - P) ≥ n / 2) :=
sorry

end distance_sum_at_least_n_div_2_l794_794712


namespace largest_base_5_five_digits_base_10_value_l794_794041

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794041


namespace no_integer_solutions_l794_794157

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l794_794157


namespace num_vec_a_exists_l794_794650

-- Define the vectors and the conditions
def vec_a (x y : ℝ) : (ℝ × ℝ) := (x, y)
def vec_b (x y : ℝ) : (ℝ × ℝ) := (x^2, y^2)
def vec_c : (ℝ × ℝ) := (1, 1)

-- Define the dot product
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the conditions
def cond_1 (x y : ℝ) : Prop := (x + y = 1)
def cond_2 (x y : ℝ) : Prop := (x^2 / 4 + (1 - x)^2 / 9 = 1)

-- The proof problem statement
theorem num_vec_a_exists : ∃! (x y : ℝ), cond_1 x y ∧ cond_2 x y := by
  sorry

end num_vec_a_exists_l794_794650


namespace length_ab_l794_794273

section geometry

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the lengths and perimeters as needed
variables (AB AC BC CD DE CE : ℝ)

-- Isosceles Triangle properties
axiom isosceles_abc : AC = BC
axiom isosceles_cde : CD = DE

-- Conditons given in the problem
axiom perimeter_cde : CE + CD + DE = 22
axiom perimeter_abc : AB + BC + AC = 24
axiom length_ce : CE = 8

-- Goal: To prove the length of AB
theorem length_ab : AB = 10 :=
by 
  sorry

end geometry

end length_ab_l794_794273


namespace quadrilateral_is_trapezoid_or_parallelogram_l794_794365

noncomputable def quadrilateral_property (s1 s2 s3 s4 : ℝ) : Prop :=
  (s1 + s2) * (s3 + s4) = (s1 + s4) * (s2 + s3)

theorem quadrilateral_is_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ) (h : quadrilateral_property s1 s2 s3 s4) :
  (s1 = s3) ∨ (s2 = s4) ∨ -- Trapezoid conditions
  ∃ (p : ℝ), (p * s1 = s3 * (s1 + s4)) := -- Add necessary conditions to represent a parallelogram
sorry

end quadrilateral_is_trapezoid_or_parallelogram_l794_794365


namespace hyperbola_asymptotes_l794_794210

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, (x^2)/4 - y^2 = 1) →
  (∀ x : ℝ, y = x / 2 ∨ y = -x / 2) :=
by
  intro h1
  sorry

end hyperbola_asymptotes_l794_794210


namespace bug_total_distance_l794_794834

theorem bug_total_distance :
  let x1 := -3
  let x2 := -8
  let x3 := 2
  let x4 := 7
  abs (x2 - x1) + abs (x3 - x2) + abs (x4 - x3) = 20 :=
by
  have h1 : abs (x2 - x1) = abs (-8 - -3) := rfl
  have h2 : h1 = abs (-5) := rfl
  have h3 : abs (-5) = 5 := abs_neg 5
  have h4 : abs (x3 - x2) = abs (2 - -8) := rfl
  have h5 : h4 = abs (10) := rfl
  have h6 : abs (10) = 10 := rfl
  have h7 : abs (x4 - x3) = abs (7 - 2) := rfl
  have h8 : h7 = abs (5) := rfl
  have h9 : abs (5) = 5 := rfl
  have h10 : abs (x2 - x1) + abs (x3 - x2) + abs (x4 - x3) = 5 + 10 + 5 := by
        rw [h3, h6, h9]
  have h11 : 5 + 10 + 5 = 20 := rfl
  show 5 + 10 + 5 = 20, from h11

end bug_total_distance_l794_794834


namespace octagon_area_minus_semicircles_l794_794446

theorem octagon_area_minus_semicircles :
  let s := 3
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let r := 1.5
  let area_semicircle := (1 / 2) * Real.pi * r^2
  let total_area_semicircles := 8 * area_semicircle
  let shaded_area := area_octagon - total_area_semicircles
  shaded_area = 18 * (1 + Real.sqrt 2) - 9 * Real.pi :=
begin
  sorry
end

end octagon_area_minus_semicircles_l794_794446


namespace number_of_pots_of_rosemary_l794_794110

-- Definitions based on the conditions
def total_leaves_basil (pots_basil : ℕ) (leaves_per_basil : ℕ) : ℕ := pots_basil * leaves_per_basil
def total_leaves_rosemary (pots_rosemary : ℕ) (leaves_per_rosemary : ℕ) : ℕ := pots_rosemary * leaves_per_rosemary
def total_leaves_thyme (pots_thyme : ℕ) (leaves_per_thyme : ℕ) : ℕ := pots_thyme * leaves_per_thyme

-- The given problem conditions
def pots_basil : ℕ := 3
def leaves_per_basil : ℕ := 4
def leaves_per_rosemary : ℕ := 18
def pots_thyme : ℕ := 6
def leaves_per_thyme : ℕ := 30
def total_leaves : ℕ := 354

-- Proving the number of pots of rosemary
theorem number_of_pots_of_rosemary : 
  ∃ (pots_rosemary : ℕ), 
  total_leaves_basil pots_basil leaves_per_basil + 
  total_leaves_rosemary pots_rosemary leaves_per_rosemary + 
  total_leaves_thyme pots_thyme leaves_per_thyme = 
  total_leaves ∧ pots_rosemary = 9 :=
by
  sorry  -- proof is omitted

end number_of_pots_of_rosemary_l794_794110


namespace cost_of_three_pencils_and_two_pens_l794_794756

theorem cost_of_three_pencils_and_two_pens
  (p q : ℝ)
  (h₁ : 8 * p + 3 * q = 5.20)
  (h₂ : 2 * p + 5 * q = 4.40) :
  3 * p + 2 * q = 2.5881 :=
by
  sorry

end cost_of_three_pencils_and_two_pens_l794_794756


namespace exists_n0_find_N_l794_794986

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Definition of the sequence {a_n}
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = f (a n)

-- Problem (1): Existence of n0
theorem exists_n0 (a : ℕ → ℝ) (h_seq : seq a) (h_a1 : a 1 = 3) : 
  ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n :=
  sorry

-- Problem (2): Smallest N
theorem find_N (a : ℕ → ℝ) (h_seq : seq a) (m : ℕ) (h_m : m > 1) 
  (h_a1 : 1 + 1 / (m : ℝ) < a 1 ∧ a 1 < m / (m - 1)) : 
  ∃ N : ℕ, ∀ n ≥ N, 0 < a n ∧ a n < 1 :=
  sorry

end exists_n0_find_N_l794_794986


namespace proof_l794_794526

variable (p : ℕ) (ε : ℤ)
variable (RR NN NR RN : ℕ)

-- Conditions
axiom h1 : ∀ n ≤ p - 2, 
  (n % 2 = 0 ∧ (n + 1) % 2 = 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 ≠ 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 = 0 ) ∨ 
  (n % 2 = 0 ∧ (n + 1) % 2 ≠ 0) 

axiom h2 :  RR + NN - RN - NR = 1

axiom h3 : ε = (-1) ^ ((p - 1) / 2)

axiom h4 : RR + RN = (p - 2 - ε) / 2

axiom h5 : RR + NR = (p - 1) / 2 - 1

axiom h6 : NR + NN = (p - 2 + ε) / 2

axiom h7 : RN + NN = (p - 1) / 2  

-- To prove
theorem proof : 
  RR = (p / 4) - (ε + 4) / 4 ∧ 
  RN = (p / 4) - (ε) / 4 ∧ 
  NN = (p / 4) + (ε - 2) / 4 ∧ 
  NR = (p / 4) + (ε - 2) / 4 := 
sorry

end proof_l794_794526


namespace difference_q_r_share_l794_794109

theorem difference_q_r_share (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x) (h_ratio_q : q = 7 * x) (h_ratio_r : r = 12 * x) (h_diff_pq : q - p = 4400) : q - r = 5500 :=
by
  sorry

end difference_q_r_share_l794_794109


namespace smallest_positive_angle_degree_l794_794170

theorem smallest_positive_angle_degree (y : ℝ) (hy_pos : y > 0)
  (h : sin (4 * y * real.pi / 180) * sin (5 * y * real.pi / 180) = 
       cos (4 * y * real.pi / 180) * cos (5 * y * real.pi / 180)) :
  y = 10 :=
sorry

end smallest_positive_angle_degree_l794_794170


namespace probability_of_X_eq_2_l794_794307

-- Define the random variable distribution condition
def random_variable_distribution (a : ℝ) (P : ℝ → ℝ) : Prop :=
  P 1 = 1 / (2 * a) ∧ P 2 = 2 / (2 * a) ∧ P 3 = 3 / (2 * a) ∧
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) = 1)

-- State the theorem given the conditions and the result
theorem probability_of_X_eq_2 (a : ℝ) (P : ℝ → ℝ) (h : random_variable_distribution a P) : 
  P 2 = 1 / 3 :=
sorry

end probability_of_X_eq_2_l794_794307


namespace cows_and_chickens_l794_794630

-- Definitions based on the problem conditions
variables (C H : ℕ)
def cows := 5
def legs := 4 * cows + 2 * H
def heads := cows + H

-- Proof statement
theorem cows_and_chickens :
  legs C H = 2 * heads C H + 10 :=
by sorry

end cows_and_chickens_l794_794630


namespace axel_has_winning_strategy_for_all_m_gt_1_l794_794289

def HAUKKU_game_winning_strategy (m : ℕ) (h : m > 1): Prop :=
  ∃ strategy : set ℕ → ℕ, ∀ S : set ℕ, S = {d | d ∣ m ∧ d > 0} → 
  ∀ turn : ℕ → ℕ, turn 0 > 1 → axel_has_winning_strategy m

theorem axel_has_winning_strategy_for_all_m_gt_1
  : ∀ (m : ℕ), m > 1 → HAUKKU_game_winning_strategy m :=
sorry

end axel_has_winning_strategy_for_all_m_gt_1_l794_794289


namespace NP_PL_ratio_KP_PM_ratio_l794_794317

variables (A B C D K L M N P : Point)
variable (α β : ℝ)

-- Conditions
axiom convex_quadrilateral : convex_quadrilateral A B C D
axiom K_on_AB : on_segment K A B
axiom L_on_BC : on_segment L B C
axiom M_on_CD : on_segment M C D
axiom N_on_DA : on_segment N D A
axiom AK_KB_ratio : ratio A K K B = α
axiom DM_MC_ratio : ratio D M M C = α
axiom BL_LC_ratio : ratio B L L C = β
axiom AN_ND_ratio : ratio A N N D = β
axiom intersection_KM_LN : intersection_point K M L N = P

-- Goals
theorem NP_PL_ratio : ratio N P P L = α := sorry
theorem KP_PM_ratio : ratio K P P M = β := sorry

end NP_PL_ratio_KP_PM_ratio_l794_794317


namespace find_ab_l794_794678

noncomputable def ω : ℂ := sorry -- Complex number ω such that ω^11 = 1 and ω ≠ 1
def α : ℂ := ω + ω^3 + ω^5
def β : ℂ := ω^2 + ω^6 + ω^9

theorem find_ab (h₁ : ω^11 = 1) (h₂ : ω ≠ 1) : (a: ℂ) (b: ℂ) (α β : ℂ) : (α = ω + ω^3 + ω^5) ∧ (β = ω^2 + ω^6 + ω^9) ∧ (α + β = -1) ∧ (α * β = 2) → (a = -1) ∧ (b = 2) := 
sorry

end find_ab_l794_794678


namespace largest_base_5_five_digit_number_in_decimal_l794_794029

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794029


namespace trajectory_is_hyperbola_l794_794553

-- Defining the points M and N
def M : Prod ℝ ℝ := (-Real.sqrt 5, 0)
def N : Prod ℝ ℝ := (Real.sqrt 5, 0)

-- Defining the condition |PM| - |PN| = 4
def satisfies_condition (P : Prod ℝ ℝ) : Prop :=
  Real.dist P M - Real.dist P N = 4

-- Main theorem statement
theorem trajectory_is_hyperbola (P : Prod ℝ ℝ) :
  satisfies_condition P →
  (P.fst * P.fst / 4 - P.snd * P.snd = 1) → P.fst ≥ 2 :=
sorry

end trajectory_is_hyperbola_l794_794553


namespace f_expression_m_range_l794_794823

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem f_expression (a b : ℝ) (ha : 0 < a) (hb : b = -a) (c : ℝ) (h0 : f 0 = 1) (hsymm : ∀ (x : ℝ), f x = f (1 - x)) 
                      (hbound : ∀ (x : ℝ), 1 - x ≤ f x) : 
                      f = (λ x, x^2 - x + 1) := sorry

theorem m_range (m : ℝ) : 
  ∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x + 2 * x = f m → -2 ≤ m ∧ m ≤ 3 :=
  sorry

end f_expression_m_range_l794_794823


namespace milkshake_cost_proof_l794_794447

-- Define the problem
def milkshake_cost (total_money : ℕ) (hamburger_cost : ℕ) (n_hamburgers : ℕ)
                   (n_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_hamburgers_cost := n_hamburgers * hamburger_cost
  let money_after_hamburgers := total_money - total_hamburgers_cost
  let milkshake_cost := (money_after_hamburgers - remaining_money) / n_milkshakes
  milkshake_cost

-- Statement to prove
theorem milkshake_cost_proof : milkshake_cost 120 4 8 6 70 = 3 :=
by
  -- we skip the proof steps as the problem statement does not require it
  sorry

end milkshake_cost_proof_l794_794447


namespace gcd_lcm_product_l794_794169

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l794_794169


namespace evaluate_expr_at_neg3_l794_794493

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l794_794493


namespace jude_buys_correct_number_of_vehicles_l794_794285

theorem jude_buys_correct_number_of_vehicles :
  let bottle_caps_per_car := 5 in
  let bottle_caps_per_truck := 6 in
  let initial_bottle_caps := 100 in
  let trucks_bought := 10 in
  let trucks_cost := trucks_bought * bottle_caps_per_truck in
  let remaining_bottle_caps := initial_bottle_caps - trucks_cost in
  let spent_on_cars := remaining_bottle_caps * 0.75 in
  let cars_bought := spent_on_cars / bottle_caps_per_car in
  trucks_bought + cars_bought = 16 :=
by
  -- Definitions
  let bottle_caps_per_car := 5
  let bottle_caps_per_truck := 6
  let initial_bottle_caps := 100
  let trucks_bought := 10
  let trucks_cost := trucks_bought * bottle_caps_per_truck
  let remaining_bottle_caps := initial_bottle_caps - trucks_cost
  let spent_on_cars := remaining_bottle_caps * 0.75
  let cars_bought := spent_on_cars / bottle_caps_per_car
  -- Final assertion
  have : trucks_bought + cars_bought = (10 : ℕ) + 6, from sorry,
  exact this

end jude_buys_correct_number_of_vehicles_l794_794285


namespace b_plus_c_for_quadratic_form_l794_794776

theorem b_plus_c_for_quadratic_form :
  ∃ (b c : ℤ), (∀ x : ℝ, x^2 - 20 * x + 49 = (x + b)^2 + c) ∧ (b + c = -61) :=
by
  exists -10, -51
  intro x
  sorry

end b_plus_c_for_quadratic_form_l794_794776


namespace problem_solution_l794_794408

-- Given conditions
variables (f : ℝ → ℝ)
axiom functional_eqn : ∀ x y : ℝ, f(x + 2 * y) - f(x - 2 * y) = 4 * x * y
axiom f_1 : f 1 = 2

-- Statement to prove f(9) = 42
theorem problem_solution : f 9 = 42 := sorry

end problem_solution_l794_794408


namespace expansion_coefficients_binomial_max_term_l794_794824

-- Problem 1: Equivalent Lean statement
theorem expansion_coefficients (n : ℕ) (h : (nat.choose n 2) = (nat.choose n 5)) : n = 7 :=
sorry

-- Problem 2: Equivalent Lean statement
theorem binomial_max_term (n : ℕ) (h : ∑ i in finset.range(n).filter (λ k, k % 2 = 1), nat.choose n k = 128) :
  n = 8 ∧ (
    let max_term := nat.choose 8 4 * (x * sqrt x)^4 * ((1 / (real.cbrt x))^4)
    in max_term = 70 * x^4 * real.cbrt (x^2)) :=
sorry

end expansion_coefficients_binomial_max_term_l794_794824


namespace sum_area_volume_l794_794082

-- Define the conditions
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def area_circle (r : ℝ) : ℝ := Real.pi * r ^ 2
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

-- State the theorem
theorem sum_area_volume (r : ℝ) (h : circumference r = 18 * Real.pi) : 
  area_circle r + volume_sphere r = 1053 * Real.pi :=
by
  sorry -- Proof not required

end sum_area_volume_l794_794082


namespace daily_wage_of_C_l794_794059

theorem daily_wage_of_C (x : ℕ) 
  (ratio_A_B_C : 3 * x)
  (days_A : 6)
  (days_B : 9)
  (days_C : 4)
  (total_earnings : 1480) : 
  5 * x = 100 := 
by
  have A_earnings := 6 * (3 * x)
  have B_earnings := 9 * (4 * x)
  have C_earnings := 4 * (5 * x)
  have sum_earnings := A_earnings + B_earnings + C_earnings
  have sum_equal := sum_earnings = total_earnings
  have eq_1480 := 18 * x + 36 * x + 20 * x = 1480
  have solve_x := 74 * x = 1480
  have x_value := x = 20
  have C_daily_wage := 5 * x
  rw [x_value] at C_daily_wage
  exact C_daily_wage


end daily_wage_of_C_l794_794059


namespace problem_equiv_l794_794874

theorem problem_equiv :
  ((2001 * 2021 + 100) * (1991 * 2031 + 400)) / (2011^4) = 1 :=
by
  sorry

end problem_equiv_l794_794874


namespace tangent_line_at_zero_max_min_values_in_interval_l794_794980

noncomputable
def f (x : ℝ) := (Real.sin x) / (Real.exp x) - x

theorem tangent_line_at_zero :
  let f'(x : ℝ) := (Real.cos x - Real.sin x) / (Real.exp x) - 1 in
  f'(0) = 0 ∧ f 0 = 0 → ∀ (x : ℝ), (0, f 0) = (0, 0) → (f x = 0) :=
by
  intros
  sorry

theorem max_min_values_in_interval :
  let f'(x : ℝ) := (Real.cos x - Real.sin x) / (Real.exp x) - 1 in
  ∀ x ∈ Icc 0 π, 
  (∀ x ∈ (Ioo 0 π), f'(x) < 0) →
  f 0 = 0 ∧ f π = -π →
  (∀ x, f x ≤ f 0 ∧ f x ≥ f π) :=
by
  intros
  sorry

end tangent_line_at_zero_max_min_values_in_interval_l794_794980


namespace obtuse_angle_range_l794_794531

theorem obtuse_angle_range (λ : ℝ) :
  let a := (λ, 2 * λ)
  let b := (-3 * λ, 2)
  obtuse_angle a b → (λ < 0 ∨ λ > 4 / 3) :=
begin
  sorry
end

end obtuse_angle_range_l794_794531


namespace base5_to_base10_max_l794_794027

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794027


namespace largest_divisor_of_expression_l794_794611

theorem largest_divisor_of_expression (x : ℤ) (h_odd : x % 2 = 1) : 
  1200 ∣ ((10 * x - 4) * (10 * x) * (5 * x + 15)) := 
  sorry

end largest_divisor_of_expression_l794_794611


namespace NaOH_combination_l794_794923

def moles_of_NaOH_combined (moles_CH3COOH moles_CH3COONa : ℕ) : ℕ :=
  moles_CH3COONa

theorem NaOH_combination
  (moles_CH3COOH moles_CH3COONa moles_NaOH : ℕ)
  (h1 : moles_CH3COOH = 3)
  (h2 : moles_CH3COONa = 3)
  (h_ratio : moles_NaOH = moles_CH3COOH) :
  moles_NaOH = 3 :=
by
  rw [h1, h2, h_ratio]
  exact rfl

end NaOH_combination_l794_794923


namespace product_gcd_lcm_l794_794162

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l794_794162


namespace count_five_digit_numbers_with_thousands_digit_3_l794_794240

theorem count_five_digit_numbers_with_thousands_digit_3 :
  (number of five-digit positive integers where ten-thousands digit is 3) = 10000 :=
begin
  sorry
end

end count_five_digit_numbers_with_thousands_digit_3_l794_794240


namespace vertical_distance_to_florence_l794_794885

def daniel_pos := (10, -5)
def emma_pos := (0, 20)
def florence_pos := (3, 15)

def meet_y_axis := (0, ((emma_pos.2 + daniel_pos.2) / 2))

theorem vertical_distance_to_florence :
  florence_pos.2 - meet_y_axis.2 = 7.5 := by
  sorry

end vertical_distance_to_florence_l794_794885


namespace simplify_cos_diff_l794_794735

theorem simplify_cos_diff :
  (let x := (real.cos (30 * real.pi / 180)), y := (real.cos (60 * real.pi / 180)) in
    x - y = (sqrt 3 - 1) / 2) :=
by
  have h1 : real.cos (30 * real.pi / 180) = sqrt 3 / 2 := by sorry
  have h2 : real.cos (60 * real.pi / 180) = 1 / 2 := by sorry
  show (sqrt 3 / 2) - (1 / 2) = (sqrt 3 - 1) / 2
  sorry

end simplify_cos_diff_l794_794735


namespace distance_between_lines_proof_l794_794342

def distance_between_parallel_lines
  (A1 : ℝ) (B1 : ℝ) (C1 : ℝ)
  (A2 : ℝ) (B2 : ℝ) (C2 : ℝ) : ℝ :=
  |C2 - C1| / sqrt (A1 ^ 2 + B1 ^ 2)

theorem distance_between_lines_proof :
  distance_between_parallel_lines 1 2 (-1) 1 2 (3/2) = (sqrt 5) / 2 :=
by
  sorry

end distance_between_lines_proof_l794_794342


namespace probability_at_least_5_heads_l794_794842

def fair_coin_probability_at_least_5_heads : ℚ :=
  (Nat.choose 7 5 + Nat.choose 7 6 + Nat.choose 7 7) / 2^7

theorem probability_at_least_5_heads :
  fair_coin_probability_at_least_5_heads = 29 / 128 := 
  by
    sorry

end probability_at_least_5_heads_l794_794842


namespace arc_CD_PA_plus_PC_eq_sqrt2_PB_l794_794313

-- Definitions for points and circle
variable {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

def is_square (A B C D : α) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  dist A C = dist B D

def on_circumcircle (A B C D P : α) : Prop :=
  ∃ r : ℝ, dist A P = r ∧ dist B P = r ∧ dist C P = r ∧ dist D P = r

def arc_CD (C D P : α) : Prop :=
  exists (A B : α), on_circumcircle A B C D P

-- Main theorem
theorem arc_CD_PA_plus_PC_eq_sqrt2_PB
  {A B C D P : α} (h_square : is_square A B C D) (h_on_circ : arc_CD C D P) :
  dist P A + dist P C = real.sqrt 2 * dist P B := by sorry

end arc_CD_PA_plus_PC_eq_sqrt2_PB_l794_794313


namespace speaking_orders_count_l794_794083

theorem speaking_orders_count 
    (students : Finset ℕ) 
    (h_students_card : students.card = 7) 
    (A B : ℕ) 
    (hA : A ∈ students) 
    (hB : B ∈ students) 
    (selected : Finset ℕ) 
    (h_selected_card : selected.card = 4) 
    (h_selected_subset : selected ⊆ students) 
    (hA_or_hB : A ∈ selected ∨ B ∈ selected)
    : (students \ {A, B}).card = 5 → 
    (P (students.card, selected.card) - P ((students \ {A, B}).card, selected.card)) = 720 := 
by 
    sorry

end speaking_orders_count_l794_794083


namespace find_bounds_l794_794425

-- Define the transformation from B to A
def transform (B : ℕ) : ℕ :=
  let b := B % 10
  let B' := B / 10
  10^8 * b + B'

-- Define the conditions
def coprime_with_18 (n : ℕ) : Prop :=
  Nat.gcd n 18 = 1

def valid_B (B : ℕ) : Prop :=
  222222222 < B ∧ coprime_with_18 B ∧ B < 1000000000

-- Prove that A_min and A_max are as specified
theorem find_bounds :
  ∀ B : ℕ, valid_B B → 
    let A := transform B in 
    122222224 ≤ A ∧ A ≤ 999999998 :=
by 
  sorry

end find_bounds_l794_794425


namespace regular_triangular_prism_cosine_l794_794153

-- Define the regular triangular prism and its properties
structure RegularTriangularPrism :=
  (side : ℝ) -- the side length of the base and the lateral edge

-- Define the vertices of the prism
structure Vertices :=
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ)
  (A1 : ℝ × ℝ × ℝ)
  (B1 : ℝ × ℝ × ℝ)
  (C1 : ℝ × ℝ × ℝ)

-- Define the cosine calculation
def cos_angle (prism : RegularTriangularPrism) (v : Vertices) : ℝ := sorry

-- Prove that the cosine of the angle between diagonals AB1 and BC1 is 1/4
theorem regular_triangular_prism_cosine (prism : RegularTriangularPrism) (v : Vertices)
  : cos_angle prism v = 1 / 4 :=
sorry

end regular_triangular_prism_cosine_l794_794153


namespace blocks_per_friend_l794_794999

theorem blocks_per_friend (total_blocks : ℕ) (shared_blocks : ℕ) (friends : ℕ) 
  (h_total : total_blocks = 258)
  (h_shared : shared_blocks = 129)
  (h_friends : friends = 6) :
  (total_blocks - shared_blocks) / friends = 21 :=
by
  rw [h_total, h_shared, h_friends]
  -- This part simplifies the calculations
  have h_blocks_left : total_blocks - shared_blocks = 129,
  { rw [h_total, h_shared], simp },
  rw h_blocks_left,
  have h_result : 129 / 6 = 21.5,
    -- Here we acknowledge the integer part only
  { sorry },
  exact h_result⟩.files

end blocks_per_friend_l794_794999


namespace line_intersects_circle_two_points_trajectory_midpoint_line_through_fixed_division_l794_794538

-- Define the circle and line conditions
def circle_C (x y : ℝ) := x^2 + (y - 1)^2 = 5

def line_L (m x y : ℝ) := mx - y + 1 - m = 0

-- Prove that for m ∈ ℝ, line L always intersects circle C at two distinct points
theorem line_intersects_circle_two_points (m : ℝ) : ∀ m : ℝ, ∃ (x1 x2 y1 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ circle_C x1 y1 ∧ circle_C x2 y2 ∧ line_L m x1 y1 ∧ line_L m x2 y2 :=
by
  sorry

-- Find the equation of the trajectory of the midpoint M of chord AB
theorem trajectory_midpoint (x y : ℝ) : 
(circle_C x y → ∃ Mx My : ℝ, Mx^2 + My^2 - Mx - 2 * My + 1 = 0) :=
by
  sorry

-- Find the equation of line L when point P(1, 1) divides chord AB as AP = 1/2 PB
theorem line_through_fixed_division (x y : ℝ) : ∃ m1 m2 : ℝ, line_L m1 x y ∨ line_L m2 x y :=
by
  sorry

end line_intersects_circle_two_points_trajectory_midpoint_line_through_fixed_division_l794_794538


namespace bruce_money_left_l794_794123

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l794_794123


namespace no_right_triangle_l794_794055

theorem no_right_triangle (a b c : ℕ) (h1 : (3, 4, 5) = (a, b, c)  ∨ (1, 1, nat.sqrt 2) = (a, b, c)  ∨
                          (8, 15, 18) = (a, b, c) ∨ (5, 12, 13) = (a, b, c)  ∨ (6, 8, 10) = (a, b, c)) :
                          ¬(c^2 = a^2 + b^2) ↔ (a, b, c) = (8, 15, 18) :=
by
  sorry

end no_right_triangle_l794_794055


namespace log_diff_eq_35_l794_794251

theorem log_diff_eq_35 {a b : ℝ} (h₁ : a > b) (h₂ : b > 1)
  (h₃ : (1 / Real.log a / Real.log b) + (1 / (Real.log b / Real.log a)) = Real.sqrt 1229) :
  (1 / (Real.log b / Real.log (a * b))) - (1 / (Real.log a / Real.log (a * b))) = 35 :=
sorry

end log_diff_eq_35_l794_794251


namespace largest_base5_number_to_base10_is_3124_l794_794002

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794002


namespace max_distance_from_center_of_square_l794_794850

theorem max_distance_from_center_of_square :
  let A := (0, 0)
  let B := (1, 0)
  let C := (1, 1)
  let D := (0, 1)
  let O := (0.5, 0.5)
  ∃ P : ℝ × ℝ, 
  (let u := dist P A
   let v := dist P B
   let w := dist P C
   u^2 + v^2 + w^2 = 2)
  → dist O P = (1 + 2 * Real.sqrt 2) / (3 * Real.sqrt 2) :=
by sorry

end max_distance_from_center_of_square_l794_794850


namespace expand_expression_l794_794499

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := 
by
  sorry

end expand_expression_l794_794499


namespace largest_base5_to_base10_l794_794017

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794017


namespace calculate_series_sum_l794_794453

theorem calculate_series_sum :
  ∑ k in (finset.range (nat.succ 0)).filter (λ k, k % 1 = 0), (2^(2^k)) / (4^(2^k) - 4) = 1 :=
sorry

end calculate_series_sum_l794_794453


namespace log_equation_solution_l794_794898

theorem log_equation_solution (x : ℝ) :
  (log (x + 5) + log (2 * x - 3) = log (2 * x ^ 2 + x - 15)) → x > 1.5 :=
by
  sorry

end log_equation_solution_l794_794898


namespace probability_of_ends_with_5_l794_794846

-- Definitions based on the conditions in the problem:
def is_valid_set (n : ℕ) : Prop := (40 ≤ n ∧ n ≤ 990)
def ends_with_5 (n : ℕ) : Prop := n % 10 = 5

-- The main theorem statement
theorem probability_of_ends_with_5 :
  let total_elements := 951 in
  let valid_elements := 95 in
  (∃ total_elements valid_elements, 
    ∑ n in finset.filter is_valid_set (finset.range 1000), 1 = total_elements ∧ 
    ∑ n in finset.filter (λ n, is_valid_set n ∧ ends_with_5 n) (finset.range 1000), 1 = valid_elements ∧ 
    valid_elements / total_elements = 95 / 951
  ) := 
  sorry

end probability_of_ends_with_5_l794_794846


namespace find_pqr_l794_794441

-- First, we're dealing with positive integers p, q, r.
def positive_integer (x : ℕ) : Prop := x > 0

-- The required conditions translated into Lean definitions.
def conditions (p q r : ℕ) : Prop :=
  p < q ∧ q < r ∧ positive_integer p ∧ positive_integer q ∧ positive_integer r

def received_candies (p q r : ℕ) (A B C : ℕ) : Prop :=
  A = 20 ∧ B = 10 ∧ C = 9

def last_round_B_drew_r (r : ℕ) (B : ℕ) : Prop :=
  -- This condition must be described in the proof:
  B receives r - number of candies in the last round (we use a placeholder as detailed proof isn't required here)
  sorry

def sum_C_papers (p q r : ℕ) (C : ℕ) : Prop :=
  C = 18

-- The main theorem statement using all the conditions.
theorem find_pqr (p q r : ℕ) :
  conditions p q r →
  received_candies p q r A B C →
  last_round_B_drew_r r B →
  sum_C_papers p q r C →
  p = 3 ∧ q = 6 ∧ r = 13 :=
by
  intros
  sorry

end find_pqr_l794_794441


namespace remainder_when_divided_by_5_l794_794805

theorem remainder_when_divided_by_5 (n : ℕ) (hn : n = 2367905) : n % 5 = 0 :=
by
  rw [hn]
  -- Add proof here
  sorry

end remainder_when_divided_by_5_l794_794805


namespace solve_system1_l794_794812

theorem solve_system1 (x y : ℝ) :
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25 →
  x = 4 ∧ y = 3 :=
by
  sorry

end solve_system1_l794_794812


namespace work_spring_l794_794600

noncomputable def spring_constant (k : ℝ) : Prop :=
  1 = 0.01 * k

noncomputable def work_done (w : ℝ) (k : ℝ) : Prop :=
  w = ∫ x in 0..0.06, k * x

theorem work_spring (k w : ℝ) (h1 : spring_constant k) (h2 : work_done w k) : w = 0.18 :=
sorry

end work_spring_l794_794600


namespace transformed_graph_l794_794346

def g : ℝ → ℝ := λ x, x^2

theorem transformed_graph :
  ∀ x : ℝ, g(-2 * x) + 3 = 4 * x^2 + 3 :=
by
  intros x
  simp only [g]
  show (-2 * x)^2 + 3 = 4 * x^2 + 3
  calc
    (-2 * x)^2 + 3 = 4 * x^2 + 3 : by rw [pow_two, mul_comm, mul_assoc, ←pow_two]
#exit

end transformed_graph_l794_794346


namespace middle_person_distance_l794_794830

noncomputable def Al_position (t : ℝ) : ℝ := 6 * t
noncomputable def Bob_position (t : ℝ) : ℝ := 10 * t - 12
noncomputable def Cy_position (t : ℝ) : ℝ := 8 * t - 32

theorem middle_person_distance (t : ℝ) (h₁ : t ≥ 0) (h₂ : t ≥ 2) (h₃ : t ≥ 4) :
  (Al_position t = 52) ∨ (Bob_position t = 52) ∨ (Cy_position t = 52) :=
sorry

end middle_person_distance_l794_794830


namespace star_15_star_eq_neg_15_l794_794525

-- Define the operations as given
def y_star (y : ℤ) := 9 - y
def star_y (y : ℤ) := y - 9

-- The theorem stating the required proof
theorem star_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by
  sorry

end star_15_star_eq_neg_15_l794_794525


namespace domain_of_g_l794_794804

-- Define the function g
def g (x : ℝ) : ℝ := (3 * x + 1) / (x + 8)

-- State the theorem
theorem domain_of_g : {x : ℝ | x ≠ -8} = set.Ioo (-(real.nnreal.of_real 1|∞)) (-8) ∪ set.Ioo (-8) (real.nnreal.of_real 1|∞) :=
begin
  -- Proof goes here
  sorry
end

end domain_of_g_l794_794804


namespace smallest_monochrome_triangle_n_l794_794883

-- Define the problem in Lean
def monochrome_triangle_nine_points (n : Nat) : Prop :=
  ∀ (points : Fin 9 → ℝ × ℝ × ℝ), 
  (∀ i j k l : Fin 9, ¬(points i).1 * (points j).2 * (points k).3 = (points l).1 * (points l).2 * (points l).3) →
  ∀ (coloring : Fin 9 → Fin 9 → option (Bool)),
  (Finset.card (Finset.univ.image (λ p, {i : Fin 9 | coloring p i ≠ none})) = n) →
  ∃ (a b c : Fin 9), 
  (coloring a b = coloring b c ∧ coloring b c = coloring a c ∧ coloring a b ≠ none)

theorem smallest_monochrome_triangle_n : 
  ∃ n, monochrome_triangle_nine_points n ∧ (∀ m, monochrome_triangle_nine_points m → m ≥ n) :=
sorry

end smallest_monochrome_triangle_n_l794_794883


namespace find_b_l794_794575

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 2 * x + b

theorem find_b (b : ℝ) (h : ∫ x in -1..0, f x b = 2) : b = 3 :=
by
  sorry

end find_b_l794_794575


namespace min_value_b_over_a_l794_794981

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (Real.exp 1 - a) * x - b

theorem min_value_b_over_a 
  (a b : ℝ)
  (h_cond : ∀ x > 0, f x a b ≤ 0)
  (h_b : b = -1 - Real.log (a - Real.exp 1)) 
  (h_a_gt_e : a > Real.exp 1) :
  ∃ (x : ℝ), x = 2 * Real.exp 1 ∧ (b / a) = - (1 / Real.exp 1) := 
sorry

end min_value_b_over_a_l794_794981


namespace tea_to_cheese_ratio_l794_794386

-- Definitions based on conditions
def total_cost : ℝ := 21
def tea_cost : ℝ := 10
def butter_to_cheese_ratio : ℝ := 0.8
def bread_to_butter_ratio : ℝ := 0.5

-- Main theorem statement
theorem tea_to_cheese_ratio (B C Br : ℝ) (hBr : Br = B * bread_to_butter_ratio) (hB : B = butter_to_cheese_ratio * C) (hTotal : B + Br + C + tea_cost = total_cost) :
  10 / C = 2 :=
  sorry

end tea_to_cheese_ratio_l794_794386


namespace dihedral_angle_of_regular_triangular_pyramid_l794_794758

theorem dihedral_angle_of_regular_triangular_pyramid
    (height : ℝ)
    (R : ℝ)
    (H1 : R = 9 * height)
    (is_regular_triangle : ∀ {ABC : Type*} (A B C : ℝ), is_regular_triangle A B C) :
  ∀ angle : ℝ, angle = 120 :=
by
  -- Proof proceeds here
  sorry

end dihedral_angle_of_regular_triangular_pyramid_l794_794758


namespace simplify_and_evaluate_expression_l794_794736

theorem simplify_and_evaluate_expression (a : ℕ) (h : (a - 1 : ℝ)/2 ≤ 1) (ha1 : a = 1) :
  (a^2 - 6 * a + 9) / (a - 2) / (a + 2 + 5 / (2 - a)) = -1/2 :=
by
  sorry

end simplify_and_evaluate_expression_l794_794736


namespace probability_within_circle_eq_pi_over_nine_l794_794428

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let circle_area := Real.pi * (2 ^ 2)
  let square_area := 6 * 6
  circle_area / square_area

theorem probability_within_circle_eq_pi_over_nine :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_circle_eq_pi_over_nine_l794_794428


namespace total_students_in_college_l794_794061

theorem total_students_in_college (ratio_boys_girls : ℕ → ℕ → Prop) 
  (num_girls : ℕ) : ratio_boys_girls 8 5 → num_girls = 190 → ∃ total_students : ℕ, total_students = 494 :=
by
  -- Definitions for contextual setup
  let num_boys := 8 * 38 -- from ratio_boys_girls 8 5 and num_girls = 190, we find x = 38
  let total_students := num_boys + num_girls
  have h1 : num_girls = 190 := sorry -- This comes from the condition provided.
  have h2 : total_students = 494 := sorry -- This follows from the calculated answer.
  -- Prove that total_students is indeed 494 given the conditions
  exact Exists.intro total_students h2

end total_students_in_college_l794_794061


namespace gcd_lcm_product_l794_794168

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end gcd_lcm_product_l794_794168


namespace average_score_of_class_l794_794628

-- Conditions
def num_students : ℕ := 25
def high_score_students : ℕ := 3
def high_score : ℤ := 95
def zero_score_students : ℕ := 3
def rest_average_score : ℤ := 45
def rest_students : ℕ := num_students - high_score_students - zero_score_students

-- Theorem statement
theorem average_score_of_class :
  (high_score_students * high_score +
   zero_score_students * 0 +
   rest_students * rest_average_score) / num_students = 45.6 :=
by
  sorry

end average_score_of_class_l794_794628


namespace quadrilateral_midpoint_distance_squared_l794_794259

noncomputable section

open Classical

variables {A B C D X Y : ℝ}

def is_convex_quadrilateral (A B C D : ℝ): Prop :=
  -- Definition of the convex quadrilateral
  -- to be expanded according to the specific geometry if needed
  true

def midpoint (a b : ℝ) : ℝ :=
  (a + b) / 2

def squared_distance (x y : ℝ) : ℝ :=
  (x - y) ^ 2

theorem quadrilateral_midpoint_distance_squared
  (h_conv: is_convex_quadrilateral A B C D)
  (h_AB: A = B ∧ B = 15)
  (h_CD: C = D ∧ D = 20)
  (h_angle: A = 60)
  (h_midpoints: X = midpoint B C ∧ Y = midpoint D A) :
  squared_distance X Y = (5375 / 4) + 25 * Real.sqrt 15 :=
sorry

end quadrilateral_midpoint_distance_squared_l794_794259


namespace Area_S_inequality_l794_794293

def S (t : ℝ) (x y : ℝ) : Prop :=
  let T := Real.sin (Real.pi * t)
  |x - T| + |y - T| ≤ T

theorem Area_S_inequality (t : ℝ) :
  let T := Real.sin (Real.pi * t)
  0 ≤ 2 * T^2 := by
  sorry

end Area_S_inequality_l794_794293


namespace similar_triangles_l794_794112

-- Define the circles with centers O1 and O2 and radii r1 and r2 respectively
variables {O1 O2 A P B C E : Type}
variables [metric_space O1] [metric_space O2]
variables {r1 r2 : ℝ}

-- Assume the circles are externally tangent at point A
def circles_tangent_at (O1 O2 A : Type) [metric_space O1] [metric_space O2] : Prop := 
  dist O1 O2 = r1 + r2

-- Assume PB and PC are tangent to the respective circles at points B and C
def tangent_line (P B : Type) (r : ℝ) : Prop :=
  dist P B = r

-- Given the ratio of tangents is equal to the ratio of the radii
def tangent_ratio (P B C : Type) (r1 r2 : ℝ) : Prop :=
  dist P B / dist P C = r1 / r2

-- Given PA intersects O2 at point E
def PA_intersects_at (P A E : Type) : Prop :=
  sorry -- needs a proper definition of points on intersection

-- Prove that triangles PAB and PEC are similar
theorem similar_triangles
  (h_tangent : circles_tangent_at O1 O2 A)
  (h_tangent1 : tangent_line P B r1)
  (h_tangent2 : tangent_line P C r2)
  (h_ratio : tangent_ratio P B C r1 r2)
  (h_intersect : PA_intersects_at P A E) :
  similar_triangles P A B P E C :=
begin
  sorry -- proof omitted as instructed
end

end similar_triangles_l794_794112


namespace find_k_and_general_term_l794_794970

noncomputable def sum_of_first_n_terms (n k : ℝ) : ℝ :=
  -n^2 + (10 + k) * n + (k - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  -2 * n + 12

theorem find_k_and_general_term :
  (∀ n k : ℝ, sum_of_first_n_terms n k = sum_of_first_n_terms n (1 : ℝ)) ∧
  (∀ n : ℕ, ∃ an : ℝ, an = general_term n) :=
by
  sorry

end find_k_and_general_term_l794_794970


namespace complex_number_unique_l794_794915

theorem complex_number_unique (z : ℂ) (h1 : |z - 2| = |z + 4|) (h2 : |z + 4| = |z + 2 * I|) : z = -1 - I :=
by sorry

end complex_number_unique_l794_794915


namespace andrea_needs_to_buy_sod_squares_l794_794817

theorem andrea_needs_to_buy_sod_squares :
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  1500 = total_area / area_of_sod_square :=
by
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  sorry

end andrea_needs_to_buy_sod_squares_l794_794817


namespace product_of_Q_at_roots_of_P_is_five_l794_794569

noncomputable def P (x : ℂ) : ℂ := x^5 - x^2 + 1
noncomputable def Q (x : ℂ) : ℂ := x^2 + 1

theorem product_of_Q_at_roots_of_P_is_five
  (r : ℕ → ℂ) (hroot: ∀ i : ℕ, i < 5 → P (r i) = 0) :
  ∏ i in finset.range 5, Q (r i) = 5 := 
sorry

end product_of_Q_at_roots_of_P_is_five_l794_794569


namespace root_polynomial_satisfies_expression_l794_794301

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end root_polynomial_satisfies_expression_l794_794301


namespace discriminant_of_quadratic_l794_794916

-- Define the quadratic equation coefficients
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

-- Define the discriminant function
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- State the theorem
theorem discriminant_of_quadratic :
  discriminant a b c = 81 / 4 :=
by
  -- We provide the result of the computation directly
  sorry

end discriminant_of_quadratic_l794_794916


namespace goods_train_cross_platform_time_l794_794421

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let total_distance : ℝ := length_train + length_platform
  total_distance / speed_mps

theorem goods_train_cross_platform_time :
  time_to_cross_platform 72 290.04 230 = 26.002 :=
by
  -- The proof is omitted
  sorry

end goods_train_cross_platform_time_l794_794421


namespace contrapositive_equiv_l794_794755

variable (a b : ℝ)

def original_proposition : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

def contrapositive_proposition : Prop := a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0

theorem contrapositive_equiv : original_proposition a b ↔ contrapositive_proposition a b :=
by
  sorry

end contrapositive_equiv_l794_794755


namespace inequality_proof_l794_794137

noncomputable def A : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def C : Set ℝ := {x : ℝ | -4 < x ∧ x < 1}
def D : Set ℝ := λ x, B x ∧ C x -- Intersection of B and the complement of A

theorem inequality_proof (a b : ℝ) (ha : D a) (hb : D b) : 
    (|a + b| / 2) < |1 + (a * b) / 4| := 
sorry

end inequality_proof_l794_794137


namespace exists_set_H_l794_794065

-- Define 3-dimensional space
variable (ℝ: Type) [RealSpace : Real ℝ]
noncomputable def point_3D := List (List ℝ)

-- Definition of H
def H (set : Type) := set

-- Problem definition in Lean 4
theorem exists_set_H : ∃ (H : set (point_3D ℝ)), 
  (#H = 2006) ∧ 
  (¬∃ (p : List ℝ → Prop), ∀ (q : List ℝ), q ∈ H → p q) ∧ 
  (∀ (a b c : List ℝ), a ∈ H ∧ b ∈ H ∧ c ∈ H → ¬collinear (a :: b :: c :: [])) ∧
  (∀ (x y : List ℝ), x ∈ H ∧ y ∈ H → ∃ (z w : List ℝ), z ∈ H ∧ w ∈ H ∧ (x ≠ z ∨ y ≠ w) ∧ parallel (x, y) (z, w)) :=
by
  sorry

-- Auxiliary notations for collinear and parallel
def collinear (points : List (List ℝ)) := sorry
def parallel (line1 line2 : (List ℝ) × (List ℝ)) := sorry

end exists_set_H_l794_794065


namespace range_of_a_l794_794196

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x = 3 ∧ 3 * x - (a * x + 1) / 2 < 4 * x / 3) → a > 3 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  sorry

end range_of_a_l794_794196


namespace intersection_M_N_l794_794586

def M := {x : ℝ | (x + 2) * (x - 2) > 0}
def N := {-3, -2, 2, 3, 4}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = ({-3, 3, 4} : set ℝ) :=
by
  sorry

end intersection_M_N_l794_794586


namespace knocks_to_knicks_l794_794602

def knicks := ℕ
def knacks := ℕ
def knocks := ℕ

axiom knicks_to_knacks_ratio (k : knicks) (n : knacks) : 5 * k = 3 * n
axiom knacks_to_knocks_ratio (n : knacks) (o : knocks) : 4 * n = 6 * o

theorem knocks_to_knicks (k : knicks) (n : knacks) (o : knocks) (h1 : 5 * k = 3 * n) (h2 : 4 * n = 6 * o) :
  36 * o = 40 * k :=
sorry

end knocks_to_knicks_l794_794602


namespace range_of_a_l794_794892

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ¬ ∃ x_0 > 2, (x_0 - a) ⊗ x_0 > a + 2) : a ≤ 7 :=
by
  sorry

end range_of_a_l794_794892


namespace divides_lcm_condition_l794_794747

theorem divides_lcm_condition (x y : ℕ) (h₀ : 1 < x) (h₁ : 1 < y)
  (h₂ : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x := 
sorry

end divides_lcm_condition_l794_794747


namespace units_digit_difference_l794_794046

-- Conditions based on the problem statement
def units_digit_of_power_of_5 (n : ℕ) : ℕ := 5

def units_digit_of_power_of_3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0     => 1
  | 1     => 3
  | 2     => 9
  | 3     => 7
  | _     => 0  -- impossible due to mod 4

-- Problem statement in Lean as a theorem
theorem units_digit_difference : (5^2019 - 3^2019) % 10 = 8 :=
by
  have h1 : (5^2019 % 10) = units_digit_of_power_of_5 2019 := sorry
  have h2 : (3^2019 % 10) = units_digit_of_power_of_3 2019 := sorry
  -- The core proof step will go here
  sorry

end units_digit_difference_l794_794046


namespace arithmetic_seq_a9_l794_794261

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 - a 2 = -2) (h2 : a 7 = -2) : a 9 = -6 := 
by sorry

end arithmetic_seq_a9_l794_794261


namespace quadrilateral_inequality_l794_794786

-- Definitions based on conditions in a)
variables {A B C D : Type}
variables (AB AC AD BC CD : ℝ)
variable (angleA angleC: ℝ)
variable (convex := angleA + angleC < 180)

-- Lean statement that encodes the problem
theorem quadrilateral_inequality 
  (Hconvex : convex = true)
  : AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l794_794786


namespace sqrt_30_estimate_l794_794145

theorem sqrt_30_estimate : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end sqrt_30_estimate_l794_794145


namespace initial_candies_count_l794_794072

theorem initial_candies_count (x : ℕ) (h1 : x % 4 = 0)
                              (h2 : x / 2 - 20 ≥ 4)
                              (h3 : x / 2 - 20 ≤ 8) :
                              x = 48 :=
by
  -- Rewriting the conditions into mathematical inequalities
  have h4: 24 ≤ x / 2 := by linarith [h2]
  have h5: x / 2 ≤ 28 := by linarith [h3]
  -- Multiplying by 2 to shift the inequalities on x
  have h6 : 48 ≤ x := by linarith
  have h7 : x ≤ 56 := by linarith
  -- Considering x divisible by 4 from the condition h1
  have h8 : ∃ n, x = 4 * n := by
    use x / 4
    exact (Nat.div_mul_cancel h1).symm
  -- Using the range and having x divisible by 4 implies uniqueness
  cases h8 with n hn
  rw [hn] at *
  have h9 : 4 * n ≥ 48 := by linarith
  have h10 : 4 * n ≤ 56 := by linarith
  norm_num at h9 h10
  interval_cases n with _ n [12]
  repeat { norm_num }
  -- Conclusion
  all_goals
  {
    exact rfl,
  }

end initial_candies_count_l794_794072


namespace product_g_roots_l794_794687

noncomputable def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 3

theorem product_g_roots (x_1 x_2 x_3 x_4 : ℝ) (hx : ∀ x, (x = x_1 ∨ x = x_2 ∨ x = x_3 ∨ x = x_4) ↔ f x = 0) :
  g x_1 * g x_2 * g x_3 * g x_4 = 142 :=
by sorry

end product_g_roots_l794_794687


namespace sum_divides_sum_powers_l794_794327

theorem sum_divides_sum_powers (n : ℕ) (k : ℕ) (h : k % 2 = 1) : 
  (∑ i in Finset.range (n + 1), i) ∣ (∑ i in Finset.range (n + 1), i^k) := 
sorry

end sum_divides_sum_powers_l794_794327


namespace problem_l794_794685

theorem problem
  (r s t : ℝ)
  (h₀ : r^3 - 15 * r^2 + 13 * r - 8 = 0)
  (h₁ : s^3 - 15 * s^2 + 13 * s - 8 = 0)
  (h₂ : t^3 - 15 * t^2 + 13 * t - 8 = 0) :
  (r / (1 / r + s * t) + s / (1 / s + t * r) + t / (1 / t + r * s) = 199 / 9) :=
sorry

end problem_l794_794685


namespace correct_operation_l794_794054

variable (x y a : ℝ)

lemma correct_option_C :
  -4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2 :=
by sorry

lemma wrong_option_A :
  x * (2 * x + 3) ≠ 2 * x^2 + 3 :=
by sorry

lemma wrong_option_B :
  a^2 + a^3 ≠ a^5 :=
by sorry

lemma wrong_option_D :
  x^3 * x^2 ≠ x^6 :=
by sorry

theorem correct_operation :
  ((-4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2) ∧
   (x * (2 * x + 3) ≠ 2 * x^2 + 3) ∧
   (a^2 + a^3 ≠ a^5) ∧
   (x^3 * x^2 ≠ x^6)) :=
by
  exact ⟨correct_option_C x y, wrong_option_A x, wrong_option_B a, wrong_option_D x⟩

end correct_operation_l794_794054


namespace area_ratio_l794_794434

noncomputable def midpoint (S O : Point) : Point :=
{
  x := (S.x + O.x) / 2,
  y := (S.y + O.y) / 2,
  z := (S.z + O.z) / 2,
}

variables (S A B C O : Point)
variables (r : ℝ) (h₁ : r = 1) (h₂ : sphere r O S) (h₃ : tetrahedron_regular S A B C) 
variables (h₄ : great_circle r O B A) (h₅ : great_circle r O C A) (h₆ : great_circle r O C B)

def M := midpoint S O

theorem area_ratio (h₇ : midpoint S O = M) : 
  let triangle_area := (sqrt 3 / 4 : ℝ) * (sqrt 3)^2,
      cross_section_tetrahedron_area := (1 / 4 : ℝ) * (sqrt 3 / 4 * (sqrt 3)^2),
      circle_cross_section_radius := sqrt (1^2 - (1 / 2)^2),
      circle_cross_section_area := π * (circle_cross_section_radius)^2 in 
  cross_section_tetrahedron_area / circle_cross_section_area = (sqrt 3) / (4 * π) :=
by 
  sorry

end area_ratio_l794_794434


namespace number_of_persons_in_first_group_eq_39_l794_794742

theorem number_of_persons_in_first_group_eq_39 :
  ∀ (P : ℕ),
    (P * 12 * 5 = 15 * 26 * 6) →
    P = 39 :=
by
  intros P h
  have h1 : P = (15 * 26 * 6) / (12 * 5) := sorry
  simp at h1
  exact h1

end number_of_persons_in_first_group_eq_39_l794_794742


namespace distance_from_center_to_line_l794_794343

def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 2

def line_eq : ℝ → ℝ → Prop := λ x y, y = x + Real.sqrt 2

def dist_point_to_line (x0 y0 a b c : ℝ) : ℝ :=
  (Real.abs (a * x0 + b * y0 + c)) / (Real.sqrt (a^2 + b^2))

theorem distance_from_center_to_line :
  dist_point_to_line 0 0 1 (-1) (Real.sqrt 2) = 1 :=
by
  sorry

end distance_from_center_to_line_l794_794343


namespace dihedral_angle_at_edge_l794_794429

theorem dihedral_angle_at_edge (φ α : ℝ) (h : ∀ {S A B C : ℝ} (β : ℝ), 
  isosceles_triangle A B C α → 
  lateral_edge_inclined S A φ → 
  lateral_edge_inclined S B φ → 
  lateral_edge_inclined S C φ → 
  midside_vertex_point_is_mid BC A → 
  midside_vertex_point_is_mid BC B → 
  midside_vertex_point_is_mid BC C → 
  equal_lateral_edges S A S B S C →
  β = φ) : 
  dihedral_angle_at_edge α = φ :=
by
  sorry

end dihedral_angle_at_edge_l794_794429


namespace set_aside_bars_each_day_l794_794998

-- Definitions for the conditions
def total_bars : Int := 20
def bars_traded : Int := 3
def bars_per_sister : Int := 5
def number_of_sisters : Int := 2
def days_in_week : Int := 7

-- Our goal is to prove that Greg set aside 1 bar per day
theorem set_aside_bars_each_day
  (h1 : 20 - 3 = 17)
  (h2 : 5 * 2 = 10)
  (h3 : 17 - 10 = 7)
  (h4 : 7 / 7 = 1) :
  (total_bars - bars_traded - (bars_per_sister * number_of_sisters)) / days_in_week = 1 := by
  sorry

end set_aside_bars_each_day_l794_794998


namespace mean_age_euler_family_l794_794336

theorem mean_age_euler_family :
  let ages := [6, 6, 6, 6, 10, 10, 16] in
  (List.sum ages : ℚ) / (List.length ages) = 60 / 7 := by
  sorry

end mean_age_euler_family_l794_794336


namespace surface_area_relation_l794_794993

variables {R1 R2 R3 S1 S2 S3 : ℝ}

-- Define the surface area in terms of radius
def surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

axiom radius_relation : R1 + 2 * R2 = 3 * R3

axiom surface_area1 : S1 = surface_area R1
axiom surface_area2 : S2 = surface_area R2
axiom surface_area3 : S3 = surface_area R3

theorem surface_area_relation (R1 R2 R3 S1 S2 S3 : ℝ) :
  R1 + 2 * R2 = 3 * R3 →
  S1 = surface_area R1 →
  S2 = surface_area R2 →
  S3 = surface_area R3 →
  Real.sqrt S1 + 2 * Real.sqrt S2 = 3 * Real.sqrt S3 :=
by
  sorry

end surface_area_relation_l794_794993


namespace beef_original_weight_l794_794856

-- Assume the original weight 'W' and the after-processing weight 'afterProcessingWeight'
def originalWeight (W : ℝ) (afterProcessingWeight : ℝ) : Prop :=
  afterProcessingWeight = 0.65 * W

theorem beef_original_weight (W : ℝ) (afterProcessingWeight : ℝ) (h : originalWeight W afterProcessingWeight) :
  W = 861.54 :=
by 
  have h1 : afterProcessingWeight = 560 := sorry
  have h2 : 0.65 * W = 560 := by rw [←h, h1]
  have h3 : W = 560 / 0.65 := by linarith
  exact sorry

end beef_original_weight_l794_794856


namespace calculate_x_l794_794272

-- Let A, B, and C be points of tangency in an equilateral triangle DEF
variable (A B C : Point)
variable (DEF : Triangle)
variable (equilateral_DEF : is_equilateral DEF)
variable (tangency_AB : is_tangent_circle DEF A)
variable (tangency_BC : is_tangent_circle DEF B)
variable (tangency_CA : is_tangent_circle DEF C)

-- Given the radius of the inscribed circle
constant r : ℝ
constant r_eq : r = 3 / 16

-- Calculate x based on the provided problem setup
constant x : ℝ
axiom x_eq : x = r -- Place the derived x here based on the problem setup

-- The proof statement we need to show
theorem calculate_x : x = 1 / 16 :=
by
  rw [x_eq, r_eq]
  norm_num
  done

end calculate_x_l794_794272


namespace complex_number_unique_l794_794914

theorem complex_number_unique (z : ℂ) (h1 : |z - 2| = |z + 4|) (h2 : |z + 4| = |z + 2 * I|) : z = -1 - I :=
by sorry

end complex_number_unique_l794_794914


namespace problem_solution_l794_794983

-- Definitions for conditions
def ω : ℝ := 2
def φ : ℝ := π / 6
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)
def A : ℝ := π / 3
def b : ℝ := 3 / 2 + Real.sqrt(3) / 2
def c : ℝ := 3 / 2 - Real.sqrt(3) / 2
def a : ℝ := Real.sqrt 6

-- Lean statement
theorem problem_solution : 
  (f 0 = 1 / 2) ∧ 
  (ω > 0) ∧
  (0 < φ < π/2) ∧
  (∀ t : ℝ, f t = Real.sin (2 * t + π / 6) -- Function expression correct
  ∧ (0 ≤ t ∧ t ≤ π → (t ∈ [0, π / 6] ∨ t ∈ [2 * π / 3, π])) -- Intervals of monotonic increase correct
  ∧ (f (A / 2) - Real.cos A = 1 / 2) -- Additional condition for triangle
  ∧ (b * c = 1) ∧ (b + c = 3) → a = Real.sqrt 6) := 
by
  sorry

end problem_solution_l794_794983


namespace trig_identity_l794_794972

theorem trig_identity {α : ℝ} (h1 : ∃ (P : ℝ × ℝ), P = (3/5, 4/5) ∧ (P.1 = cos α) ∧ (P.2 = sin α)) :
  sin α + 2 * cos α = 2 :=
sorry

end trig_identity_l794_794972


namespace concentric_circles_l794_794640

open EuclideanGeometry

noncomputable def isosceles_trapezoid {A B C D : Point}
  (hABCD : IsoscelesTrapezoid A B C D) : Prop :=
IsoscelesTrapezoid A B C D

noncomputable def reflection (P Q : Point) (l : Line) : Point :=
Reflection P l

noncomputable def circumcircle (A B C : Point) : Circle :=
Circumcircle A B C

theorem concentric_circles
  {A B C D E F X Y : Point}
  (hAB_parallel_CD : AB.parallel CD)
  (hDE_CF : E ∈ LineSegment D C ∧ F ∈ LineSegment D C ∧ D, E, F, C are_in_order ∧ Distance D E = Distance C F)
  (hX_reflection : X = reflection E (LineThrough A D))
  (hY_reflection : Y = reflection C (LineThrough A F))
  (hCircumcenter : O = Circumcenter (Triangle A F D)) :
  Circumcircle (Triangle A F D) = Circumcircle (Triangle B X Y) :=
sorry

end concentric_circles_l794_794640


namespace g_n_plus_one_minus_g_n_minus_one_eq_g_n_l794_794670

theorem g_n_plus_one_minus_g_n_minus_one_eq_g_n (g : ℕ → ℝ)
  (hg : ∀ n, g n = (2 + real.sqrt 2) / 4 * ((1 + real.sqrt 2) / 2)^n + (2 - real.sqrt2) / 4 * ((1 - real.sqrt 2) / 2)^n) :
  ∀ n, g (n + 1) - g (n - 1) = g n :=
by
  sorry

end g_n_plus_one_minus_g_n_minus_one_eq_g_n_l794_794670


namespace joan_total_spending_l794_794666

def basketball_game_price : ℝ := 5.20
def basketball_game_discount : ℝ := 0.15 * basketball_game_price
def basketball_game_discounted : ℝ := basketball_game_price - basketball_game_discount

def racing_game_price : ℝ := 4.23
def racing_game_discount : ℝ := 0.10 * racing_game_price
def racing_game_discounted : ℝ := racing_game_price - racing_game_discount

def puzzle_game_price : ℝ := 3.50

def total_before_tax : ℝ := basketball_game_discounted + racing_game_discounted + puzzle_game_price
def sales_tax : ℝ := 0.08 * total_before_tax
def total_with_tax : ℝ := total_before_tax + sales_tax

theorem joan_total_spending : (total_with_tax : ℝ) = 12.67 := by
  sorry

end joan_total_spending_l794_794666


namespace hyperbola_C_equation_is_correct_minimum_PM_distance_is_correct_l794_794208

noncomputable def hyperbola_C_equation (C : ℝ → ℝ → Prop) : Prop :=
  C x y ↔ x^2 / 4 - y^2 = 1

theorem hyperbola_C_equation_is_correct :
  ∃ (C : ℝ → ℝ → Prop), (∀ x y, C x y ↔ x^2 / 4 - y^2 = 1) :=
begin
  use (λ x y, x^2 / 4 - y^2 = 1),
  intro x,
  intro y,
  simp,
end

noncomputable def minimum_PM_distance (PM_distance : ℝ → ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (x_0 y_0 : ℝ), x_0^2 / 4 - y_0^2 = 1 → PM_distance x_0 y_0 5 0 = 2

theorem minimum_PM_distance_is_correct :
  (∀ (x_0 y_0 : ℝ), x_0^2 / 4 - y_0^2 = 1 → (sqrt ((5/4) * (x_0 - 4)^2 + 4)) = 2) :=
begin
  sorry
end

end hyperbola_C_equation_is_correct_minimum_PM_distance_is_correct_l794_794208


namespace trailing_zeros_in_100_fac_l794_794244

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def number_of_trailing_zeros (n : ℕ) : ℕ :=
let count_factors := λ p n, n / p + n / (p * p) + n / (p * p * p)
in count_factors 5 n

theorem trailing_zeros_in_100_fac : number_of_trailing_zeros 100 = 24 :=
by
  sorry

end trailing_zeros_in_100_fac_l794_794244


namespace count_cuboso_sequences_l794_794855

/--
A sequence of numbers is *platense* if the first number is greater than 1, and 
\( a_{n+1} = \frac{a_n}{p_n} \) where \( p_n \) is the least prime divisor of 
\( a_n \), and the sequence ends if \( a_n = 1 \).
A sequence is *cuboso* if some term is a perfect cube greater than 1.
-/
def is_platense (seq : List ℕ) : Prop := 
  seq.head > 1 ∧ 
  ∀ (n : ℕ), n < seq.length - 1 → seq.get! (n + 1) = seq.get! n / Nat.minFac (seq.get! n) ∧
  seq.get! (seq.length - 1) = 1

def is_cuboso (seq : List ℕ) : Prop :=
  ∃ (x : ℕ), x > 1 ∧ 3 ∣ Nat.log 3 x ∧ x ∈ seq

/-- 
Determine the number of sequences cuboso which the initial term is less than 2022.
-/
theorem count_cuboso_sequences (N : ℕ := 2022) : 
  ∃ (count : ℕ), count = 30 ∧ 
  ∀ (s : ℕ), s < N → 
  (∃ seq : List ℕ, seq.head = s ∧ is_platense seq ∧ is_cuboso seq) ↔ s < N := 
begin
  sorry
end

end count_cuboso_sequences_l794_794855


namespace find_N_l794_794649

-- Definitions based on the given conditions
def small_semicircle_area (r : ℝ) : ℝ := (π * r^2) / 2
def large_semicircle_area (N : ℝ) (r : ℝ) : ℝ := (π * (N * r)^2) / 2
def region_area (N : ℝ) (r : ℝ) : ℝ := large_semicircle_area N r - N * small_semicircle_area r

-- Given the ratio of the areas
def ratio_area (N : ℝ) (r : ℝ) : Prop := small_semicircle_area r * N / (region_area N r) = 1 / 12

-- The proof problem statement
theorem find_N (N : ℝ) (r : ℝ) (hN : ratio_area N r) : N = 13 := 
sorry -- Replace with the proof later

end find_N_l794_794649


namespace bridge_length_l794_794440

theorem bridge_length 
  (train_length : ℕ)
  (crossing_time : ℕ)
  (train_speed : ℝ)
  (total_distance := crossing_time * train_speed)
  (bridge_length := total_distance - train_length) :
  train_length = 400 →
  crossing_time = 45 →
  train_speed = 55.99999999999999 →
  bridge_length = 2120 :=
by
  intros h1 h2 h3
  unfold bridge_length
  unfold total_distance
  simp [h1, h2, h3]
  norm_num
  sorry

end bridge_length_l794_794440


namespace possible_remainder_degrees_l794_794049

theorem possible_remainder_degrees (p : Polynomial ℤ) :
  ∃ r : Polynomial ℤ, degree (3 * X ^ 3 - 4 * X ^ 2 + X - 5) = 3 →
  (degree r < 3 ∧ (degree r = 0 ∨ degree r = 1 ∨ degree r = 2)) := sorry

end possible_remainder_degrees_l794_794049


namespace ellipse_properties_l794_794574

-- Declare the conditions as within a structure
variables {a b e : ℝ} (α β : ℝ) {x y : ℝ}

-- Define the ellipse properties and conditions
def ellipse_equation (x y : ℝ) : Prop := (x ^ 2) / 4 + (y ^ 2) / 3 = 1

-- Prove the result for the required ratio |AB|^2 / |DE|
theorem ellipse_properties 
  (hαβ: α + β = Real.pi) 
  (h_geometric_mean: b^2 = 3 * e * a)
  (hf1_focus: 1 = Real.eccentricity a b)
  (hb_squared: b^2 = 3) 
  (ha_squared: a^2 = 4) : 
  ∀ A B D E, 
    (A ≠ B) →
    (D ≠ E) →
    (is_incident A (line_α α)) →
    (is_incident B (line_α α)) →
    (is_incident D (line_β β)) →
    (is_incident E (line_β β)) →
    (|AB| ^ 2) / |DE| = 4 := by 
    sorry

end ellipse_properties_l794_794574


namespace sqrt_221_between_15_and_16_l794_794905

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end sqrt_221_between_15_and_16_l794_794905


namespace find_p_q_l794_794697

theorem find_p_q (p q : ℚ)
  (h1 : (4 : ℚ) * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2):
  (p, q) = (-29/12 : ℚ, 43/12 : ℚ) :=
by 
  sorry

end find_p_q_l794_794697


namespace sum_of_D_coordinates_l794_794718

-- Definition of the midpoint condition
def is_midpoint (N C D : ℝ × ℝ) : Prop :=
  N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2

-- Given points
def N : ℝ × ℝ := (5, -1)
def C : ℝ × ℝ := (11, 10)

-- Statement of the problem
theorem sum_of_D_coordinates :
  ∃ D : ℝ × ℝ, is_midpoint N C D ∧ (D.1 + D.2 = -13) :=
  sorry

end sum_of_D_coordinates_l794_794718


namespace geometric_common_ratio_l794_794969

theorem geometric_common_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (a₁ * (1 - q ^ 3)) / (1 - q) / ((a₁ * (1 - q ^ 2)) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  -- Proof omitted
  sorry

end geometric_common_ratio_l794_794969


namespace product_gcd_lcm_l794_794163

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l794_794163


namespace smoothie_combinations_l794_794857

theorem smoothie_combinations :
  let flavors := 5
  let supplements := 8
  (flavors * Nat.choose supplements 3) = 280 :=
by
  sorry

end smoothie_combinations_l794_794857


namespace solution_set_fg_lt_zero_l794_794682

variables {R : Type} [LinearOrderedField R] 
  {f g : R → R}

-- Conditions from the problem
def odd_function (f : R → R) : Prop := ∀ x, f (-x) = - f x
def even_function (g : R → R) : Prop := ∀ x, g (-x) = g x
def positive_derivative_sum (f g : R → R) : Prop := ∀ x < 0, f' x * g x + f x * g' x > 0
def g_at_neg_3_zero (g : R → R) : Prop := g (-3) = 0

-- Statement to prove
theorem solution_set_fg_lt_zero (hf_odd : odd_function f) (hg_even : even_function g)
  (h_positive_derivative_sum : positive_derivative_sum f g) (h_g_neg_3_zero : g_at_neg_3_zero g) :
  {x : R | f x * g x < 0} = {x : R | x < -3} ∪ {x : R | 0 < x ∧ x < 3} :=
sorry

end solution_set_fg_lt_zero_l794_794682


namespace problem_solution_l794_794482

noncomputable def p (x : ℝ) : ℝ := (2/5) * (x + 2)
noncomputable def q (x : ℝ) : ℝ := (1/5) * (x + 1) * (x - 2) * (x + 2)

theorem problem_solution :
  ∀ x, p(x) + q(x) = (1/5) * x^3 + x + (4/5) :=
by
  sorry

end problem_solution_l794_794482


namespace Warriors_won_30_l794_794772

variable {Games : Type}
variable {Sharks Falcons Warriors Knights Royals : Games}
variable {n20 n25 n30 n35 n40 : ℕ}
variable Games_won : list ℕ := [20, 25, 30, 35, 40]

-- The conditions
def condition1 : Prop := nSharks > nFalcons
def condition2 : Prop := nKnights >= 18
def condition3 : Prop := nWarriors > nKnights ∧ nWarriors < nRoyals

-- The relationships between the games won
variable Games_relation : list (Games × ℕ) := [(Sharks, n35), (Falcons, n30), (Warriors, n30), (Knights, n20), (Royals, n40)] 

theorem Warriors_won_30 : ∃ w : ℕ, w ∈ Games_won ∧ w = 30 :=
  by
    have h1 := condition1
    have h2 := condition2
    have h3 := condition3
    use 30
    -- Demonstrate that 30 is in the list of games won
    have h30_in_list : 30 ∈ Games_won := by simp [Games_won]
    -- Use the conditions to derive the conclusion
    sorry

end Warriors_won_30_l794_794772


namespace ellipse_equation_line_ab_equation_l794_794548

theorem ellipse_equation (
  a b : ℝ,
  (h1 : a > b) (h2 : b > 0)
  (h3 : a - real.sqrt (3) = -1)
  (h4 : b = real.sqrt 2)
  (h5 : a^2 = b^2 + (real.sqrt 3)^2)
) : 
  (∃ x y : ℝ, (x^2 / 3 + y^2 / 2 = 1)) := 
begin
  sorry
end

theorem line_ab_equation (
  (h1 : (real.sqrt 2 * x - y + real.sqrt 2 = 0) ∨ (real.sqrt 2 * x + y + real.sqrt 2 = 0))
  (h2 : (2*x = 3*real.sqrt 3 / 2))
) :
  ∃ x y : ℝ, true :=
begin
  sorry
end

end ellipse_equation_line_ab_equation_l794_794548


namespace general_formula_sum_of_na_n_l794_794544

-- Given conditions
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 3

-- Part (I)
theorem general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : sum_first_n_terms a S) :
  ∀ n : ℕ, 0 < n → a n = 3 * 2^(n - 1) :=
sorry

-- Part (II)
def sum_na_n_terms (a : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → T n = 3 * ∑ i in finset.range n, i * 2^i + 3

theorem sum_of_na_n (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) 
  (h : sum_first_n_terms a S) :
  sum_na_n_terms a T :=
sorry

end general_formula_sum_of_na_n_l794_794544


namespace min_value_fraction_l794_794348

open Real

-- Definitions based on the conditions
def passes_through_fixed_point (y : ℝ → ℝ) (A : ℝ × ℝ) : Prop :=
  y A.1 = A.2

def line_condition (A : ℝ × ℝ) (m n : ℝ) : Prop :=
  m * A.1 + n * A.2 + 1 = 0

-- The actual theorem to be proven, the core of the Lean statement
theorem min_value_fraction (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m > 0) (h4 : n > 0)
  (h5 : passes_through_fixed_point (λ x, log a (x + 2) - 1) (-1, -1))
  (h6 : line_condition (-1, -1) m n) : 
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
sorry

end min_value_fraction_l794_794348


namespace solve_equation_l794_794738

theorem solve_equation (x : ℝ) : 
  (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = -3 - Real.sqrt 6) ↔ 
  (x^4 / (2 * x + 1) + x^2 = 6 * (2 * x + 1)) := by
  sorry

end solve_equation_l794_794738


namespace derivative_periodic_l794_794721

variable {X : Type} [Real X]
variable (f : X → X)

-- Definition of periodic function with period T
def is_periodic (f : X → X) (T : X) : Prop :=
  ∀ x, f (x + T) = f x

-- Proving the periodicity of the derivative
theorem derivative_periodic {T : X} (hf : is_periodic f T) :
  is_periodic (deriv f) T :=
by
  sorry

end derivative_periodic_l794_794721


namespace base5_to_base10_max_l794_794026

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794026


namespace bruce_money_left_l794_794122

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l794_794122


namespace part1_part2_l794_794577

-- Definitions for the first part
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

-- Statement for the first part
theorem part1 (a : ℝ) (h : 0 ≤ a ∧ a < 2 * Real.exp 1) :
  ∀ x > 0, f x a > 0 := by
  sorry

-- Definitions for the second part
def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- Statement for the second part
theorem part2 (x₁ x₂ m : ℝ) (h1 : x₁ < x₂)
  (h2 : (f x₁) / Real.sqrt x₁ = Real.sqrt m)
  (h3 : (f x₂) / Real.sqrt x₂ = Real.sqrt m) :
  x₂ - x₁ < (Real.sqrt (m^2 - 16)) / 2 := by
  sorry

end part1_part2_l794_794577


namespace evaluate_expression_at_neg3_l794_794495

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l794_794495


namespace angle_B_is_pi_over_2_l794_794255

theorem angle_B_is_pi_over_2
  (A B C O : EuclideanGeometry.Point)
  (hO: EuclideanGeometry.is_circumcenter O A B C)
  (h: EuclideanGeometry.dot_product (O - A) (B - A) + EuclideanGeometry.dot_product (O - B) (C - B) = EuclideanGeometry.dot_product (O - C) (A - C)) :
  EuclideanGeometry.angle B A C = Real.pi / 2 :=
sorry

end angle_B_is_pi_over_2_l794_794255


namespace points_on_one_side_or_circle_l794_794535

theorem points_on_one_side_or_circle {P₁ P₂ P₃ : ℝ × ℝ}
  (hP₁P₂P₃ : P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₃ ≠ P₁) :
  ∃ L : ℝ × ℝ → Prop, (∀ (Q : ℝ × ℝ), Q = P₁ ∨ Q = P₂ ∨ Q = P₃ ∨  
    (∃ Q₁ Q₂ : ℝ × ℝ, Q ≠ Q₁ ∧ Q ≠ Q₂ ∧ Q = reflect (midpoint Q₁ Q₂) (perp_bisector Q₁ Q₂) P₁) ∨
    (∀ (Q₁ Q₂ : ℝ × ℝ), reflect (midpoint Q₁ Q₂) (perp_bisector Q₁ Q₂) Q₃ = P₂ ∨
                               reflect (midpoint Q₁ Q₂) (perp_bisector Q₁ Q₂) Q₃ = P₃)) →
      ( ∃ line : set (ℝ × ℝ), (∀ P : ℝ × ℝ, P ∈ line) ) ∨ 
      ( ∃ circle : set (ℝ × ℝ), (∀ P : ℝ × ℝ, P ∈ circle) )) :=
sorry

end points_on_one_side_or_circle_l794_794535


namespace ratio_b_a_l794_794529

variable {a b c : ℝ}
variables {θ₁ θ₂ : ℝ}
variables {F₁ F₂ Q : Point} -- Define points for foci and moving point Q

-- Defining ellipse conditions
def ellipse (x y a b : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Given conditions
axiom a_gt_b_gt_zero : a > b ∧ b > 0
axiom point_on_ellipse : ellipse Q.x Q.y a b
axiom angle_conditions : θ₁ = ∠ Q F₁ F₂ ∧ θ₂ = ∠ Q F₂ F₁
axiom equation_condition : 3 * sin θ₁ * (1 - cos θ₂) = 2 * sin θ₂ * (1 + cos θ₁)

-- The theorem to prove the correct value of b / a
theorem ratio_b_a : b / a = 2 * sqrt 6 / 5 :=
sorry

end ratio_b_a_l794_794529


namespace one_hundred_fifty_sixth_digit_is_five_l794_794391

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end one_hundred_fifty_sixth_digit_is_five_l794_794391


namespace largest_base5_number_conversion_l794_794013

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794013


namespace light_ray_distance_l794_794888

theorem light_ray_distance (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) :
  let d := Real.sqrt ((a + x) ^ 2 + (b + y) ^ 2)
  in d = Real.sqrt ((a + x) ^ 2 + (b + y) ^ 2) :=
by
  unfold d
  sorry

end light_ray_distance_l794_794888


namespace t_plus_s_value_l794_794564

noncomputable def z (θ : ℝ) : ℂ := 3 * complex.exp (θ * complex.I)

noncomputable def w (z : ℂ) : ℂ := (z + 1/z) / 2

noncomputable def Γ (w : ℂ) : Prop := 
  (w.re ^ 2) / (25 / 9) + (w.im ^ 2) / (16 / 9) = 1

noncomputable def line_l (P A B M : ℂ) : Prop := 
  A.re * B.im - A.im * B.re = 0 ∧ M.re = 0 ∧ (B.re - A.re) * (P.im - A.im) = (B.im - A.im) * (P.re - A.re)

noncomputable def conditions (P A B M : ℂ) : Prop := 
  P = 1 ∧ Γ A ∧ Γ B ∧ line_l P A B M

theorem t_plus_s_value (Z : Type) [has_zero Z]
  (P A B M : Z)
  (t s : ℝ) [has_add Z]
  (hAP : M - A = t * (A - P))
  (hBP : M - B = s * (B - P)) 
  (hcond : conditions P A B M) : t + s = -25 / 8 := by
  sorry

end t_plus_s_value_l794_794564


namespace knicks_from_knocks_l794_794607

variable (knicks knacks knocks : Type)
variable [HasSmul ℚ knicks] [HasSmul ℚ knacks] [HasSmul ℚ knocks]

variable (k1 : knicks) (k2 : knacks) (k3 : knocks)
variable (h1 : 5 • k1 = 3 • k2)
variable (h2 : 4 • k2 = 6 • k3)

theorem knicks_from_knocks : 36 • k3 = 40 • k1 :=
by {
  sorry
}

end knicks_from_knocks_l794_794607


namespace infinite_series_sum_l794_794130

theorem infinite_series_sum :
  ∑' (k : ℕ) (hk : k > 0), (8 ^ k) / ((2 ^ k - 3 ^ k) * (2 ^ (k + 1) - 3 ^ (k + 1))) = 3 :=
sorry

end infinite_series_sum_l794_794130


namespace perpendicular_vectors_a_eq_one_l794_794234

/-
Given two vectors in ℝ³, m = (-a, 2, 1) and n = (1, 2a, -3), which are perpendicular.
Prove that a = 1.
-/
theorem perpendicular_vectors_a_eq_one (a : ℝ) 
  (m : ℝ×ℝ×ℝ := (-a, 2, 1)) (n : ℝ×ℝ×ℝ := (1, 2a, -3)) 
  (h : m.1 * n.1 + m.2 * n.2 + m.3 * n.3 = 0) : 
  a = 1 := 
by 
  sorry

end perpendicular_vectors_a_eq_one_l794_794234


namespace φ_when_even_monotonically_increasing_interval_l794_794217

-- Given conditions
variables {ω φ : ℝ}
variables (k : ℤ)
def f (x : ℝ) := Real.sin (ω * x + φ)
def T : ℝ := π

-- The function f(x) = sin(ωx + φ) has smallest positive period π.
axiom period_assumption : T = π

-- Given ω > 0 and 0 < φ < 2π/3
axiom ω_positive : ω > 0
axiom φ_interval : 0 < φ ∧ φ < 2 * Real.pi / 3

-- Problem 1: Find φ when f is even
theorem φ_when_even (h_even : ∀ x, f(-x) = f(x)) : φ = Real.pi / 2 := sorry

-- Problem 2: Monotonically increasing interval when f passes through (π/6, sqrt(3)/2)
theorem monotonically_increasing_interval 
    (h_point : f (Real.pi / 6) = Real.sqrt 3 / 2) 
    : ∃ k : ℤ, 
        ∀ x : ℝ, 
            (k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 12) 
                ↔ f' x > 0 := sorry

end φ_when_even_monotonically_increasing_interval_l794_794217


namespace rectangle_area_l794_794368

theorem rectangle_area (w l : ℕ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) : l * w = 150 :=
by
  -- We provide the conditions in the theorem's signature:
  -- l is the length which is 15 cm, given by h1
  -- The ratio of the perimeter to the width is 5:1, given by h2
  sorry

end rectangle_area_l794_794368


namespace smallest_possible_r_l794_794302

theorem smallest_possible_r (p q r : ℤ) (hpq: p < q) (hqr: q < r) 
  (hgeo: q^2 = p * r) (harith: 2 * q = p + r) : r = 4 :=
sorry

end smallest_possible_r_l794_794302


namespace geometric_sequence_common_ratio_l794_794204

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l794_794204


namespace corn_syrup_quantity_in_sport_drink_l794_794279

theorem corn_syrup_quantity_in_sport_drink 
  (oz_water : ℕ)
  (ratio_flavor_corn_standard : ℕ × ℕ)
  (ratio_flavor_water_standard : ℕ × ℕ)
  (ratio_mult_flavor_corn_sport : ℕ)
  (ratio_div_flavor_water_sport : ℕ)
  (water_in_sport : ℕ) 
  : oz_water = water_in_sport →
    ratio_flavor_corn_standard = (1, 12) →
    ratio_flavor_water_standard = (1, 30) →
    ratio_mult_flavor_corn_sport = 3 →
    ratio_div_flavor_water_sport = 2 →
    ∃ (oz_corn_syrup : ℕ), 
    oz_corn_syrup = water_in_sport :=
begin
    intros h_water h_ratio_fc h_ratio_fw h_mult h_div,
    -- Conversion of standard to sport formulation
    have ratio_flavor_corn_sport : ℕ × ℕ := (ratio_flavor_corn_standard.1, ratio_flavor_corn_standard.2 / ratio_mult_flavor_corn_sport),
    have ratio_flavor_water_sport : ℕ × ℕ := (ratio_flavor_water_standard.1, ratio_flavor_water_standard.2 * ratio_div_flavor_water_sport),
    -- Combining the ratios to find the proportions
    have combined_ratio_flavor_corn_water_sport := (15, 60, 60),
    use water_in_sport,
    -- Since the ratios are equal, the quantity of corn syrup equals the quantity of water
    exact h_water,
end

end corn_syrup_quantity_in_sport_drink_l794_794279


namespace linear_function_quadrants_l794_794541

theorem linear_function_quadrants
  (k : ℝ) (h₀ : k ≠ 0) (h₁ : ∀ x : ℝ, x > 0 → k*x < 0) :
  (∃ x > 0, 2*x + k > 0) ∧
  (∃ x > 0, 2*x + k < 0) ∧
  (∃ x < 0, 2*x + k < 0) :=
  by
  sorry

end linear_function_quadrants_l794_794541


namespace max_value_expression_l794_794928

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (M : ℝ), M = (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ∧ M = 1/24 := 
sorry

end max_value_expression_l794_794928


namespace largest_base5_to_base10_l794_794016

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794016


namespace hike_took_one_hour_l794_794592

-- Define the constants and conditions
def initial_cups : ℕ := 6
def remaining_cups : ℕ := 1
def leak_rate : ℕ := 1 -- cups per hour
def drank_last_mile : ℚ := 1
def drank_first_3_miles_per_mile : ℚ := 2/3
def first_3_miles : ℕ := 3

-- Define the hike duration we want to prove
def hike_duration := 1

-- The total water drank
def total_drank := drank_last_mile + drank_first_3_miles_per_mile * first_3_miles

-- Prove the hike took 1 hour
theorem hike_took_one_hour :
  ∃ hours : ℕ, (initial_cups - remaining_cups = hours * leak_rate + total_drank) ∧ (hours = hike_duration) :=
by
  sorry

end hike_took_one_hour_l794_794592


namespace length_of_train_is_200_l794_794816

-- Definitions based on given conditions
def speed_of_train_kmph : ℝ := 100
def speed_of_bike_kmph : ℝ := 64
def time_seconds : ℝ := 20

-- Convert speeds from km/h to m/s
def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)
def speed_of_bike_mps : ℝ := speed_of_bike_kmph * (1000 / 3600)

-- Calculate relative speed
def relative_speed_mps : ℝ := speed_of_train_mps - speed_of_bike_mps

-- Calculate the length of the train
def length_of_train : ℝ := relative_speed_mps * time_seconds

-- Prove the length of the train is 200 meters
theorem length_of_train_is_200 : length_of_train = 200 := by
  sorry

end length_of_train_is_200_l794_794816


namespace sequence_sum_remainder_l794_794479

noncomputable def a : ℕ → ℚ
| 0     := 1
| (n+1) := a n / (4 * 1004)

def S (n : ℕ) : ℚ := ∑ i in Finset.range n, a i

theorem sequence_sum_remainder :
  S 50 = 4016 ∧ (4017 % 1004 = 1005) :=
by 
  sorry

end sequence_sum_remainder_l794_794479


namespace prod_72516_9999_l794_794524

theorem prod_72516_9999 : 72516 * 9999 = 724987484 :=
by
  sorry

end prod_72516_9999_l794_794524


namespace width_of_paper_is_72_inches_l794_794632

theorem width_of_paper_is_72_inches (length : ℤ) (volume : ℤ) (foot_to_inches : ℤ) : 
  length = 48 → volume = 8 → foot_to_inches = 12 → 
  ∃ width : ℤ, (width * length = 6 * (foot_to_inches * (volume ^ (1/3))) * (foot_to_inches * (volume ^ (1/3)))) ∧ width = 72 :=
by
  assume h_length : length = 48
  assume h_volume : volume = 8
  assume h_foot_to_inches : foot_to_inches = 12
  sorry

end width_of_paper_is_72_inches_l794_794632


namespace proof_problem_l794_794623

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 1
noncomputable def λ_min : ℝ := 2 * Real.sqrt 2

variables {A B C : ℝ}
variables (a b c : ℝ) (m n : ℝ × ℝ)

-- Define given conditions
variables (c_cond : c^2 = (5 + 2 * Real.sqrt 3) * b^2 )
variables (area_cond : 1/2 = 1/4 * a * b )

def m_vec : ℝ × ℝ := ⟨c * Real.cos C, Real.sqrt 3⟩
def n_vec : ℝ × ℝ := ⟨2, a * Real.cos B + b * Real.cos A⟩

-- Perpendicular vectors condition
def perp_cond : m_vec · n_vec = 0 

-- Trigonometric equation for λ_min 
def trig_eq (λ A : ℝ) := λ * Real.sin (4 * A) = Real.sin (2 * A) + Real.cos (2 * A)

-- Main theorem
theorem proof_problem : 
  (perp_cond ∧ c_cond ∧ area_cond) → 
  (a = 2 ∧ b = 1 ∧ 
   ∀ A : ℝ, (0 < A ∧ A < π / 6) → 
   (λ_min = Real.sqrt 8)) :=
by
  intro h,
  sorry

end proof_problem_l794_794623


namespace largest_base5_number_to_base10_is_3124_l794_794006

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794006


namespace solution_correct_l794_794674

noncomputable def problem_statement (n : ℕ) (a : Fin n → ℝ) : Prop :=
  n >= 2 ∧ 
  (∃ (σ : Equiv.Perm (Fin n)), ∀ i, a i - 2 * a (σ i) = a (σ (i + 1) % n)) 

theorem solution_correct : ∀ (n : ℕ) (a : Fin n → ℝ), 
  problem_statement n a → ∀ i, a i = 0 :=
by
  intros n a h
  sorry

end solution_correct_l794_794674


namespace find_a_l794_794765

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.sin x

theorem find_a (a : ℝ) : (∀ f', f' = (fun x => a * Real.exp x - Real.cos x) → f' 0 = 0) → a = 1 :=
by
  intros h
  specialize h (fun x => a * Real.exp x - Real.cos x) rfl
  sorry  -- proof is omitted

end find_a_l794_794765


namespace problem1_problem2_problem3_l794_794456

-- (1)
theorem problem1 (x : ℝ) (hx : x < 3) : 
  (sqrt (x^2 - 6*x + 9) - abs (4 - x) = -1) :=
by sorry

-- (2)
theorem problem2 : 
  (Real.logb 2 (4^7 * 2^5) + Real.logb 2 6 - Real.logb 2 3 = 20) :=
by sorry

-- (3)
theorem problem3 : 
  (0.0081^(1 / 4) + (4^(-3 / 4))^2 + (Real.sqrt 8)^(-4 / 3) - 16^(-0.75) = 11 / 20) :=
by sorry

end problem1_problem2_problem3_l794_794456


namespace number_of_distinct_a_values_l794_794585

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem number_of_distinct_a_values :
  (∀ a : ℝ, B a ⊆ A) → (∃ S : Set ℝ, S = {a | B a ⊆ A} ∧ S.card = 3) :=
by
  sorry

end number_of_distinct_a_values_l794_794585


namespace sum_of_solutions_l794_794512

theorem sum_of_solutions :
  ( ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → 
  ( -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ) ) → 
  ∑ x in {real.sqrt 3, -real.sqrt 3}, x = 0 := 
by
sorry

end sum_of_solutions_l794_794512


namespace remainder_of_c_plus_d_l794_794146

-- Definitions based on conditions
def c (k : ℕ) : ℕ := 60 * k + 53
def d (m : ℕ) : ℕ := 40 * m + 29

-- Statement of the problem
theorem remainder_of_c_plus_d (k m : ℕ) :
  ((c k + d m) % 20) = 2 :=
by
  unfold c
  unfold d
  sorry

end remainder_of_c_plus_d_l794_794146


namespace find_n_l794_794389

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 100 * n ≡ 85 [MOD 103]) : n = 6 := 
sorry

end find_n_l794_794389


namespace projection_ratio_l794_794297

variables (V : Type*) [inner_product_space ℝ V]
variables (v w p q : V)
hypothesis (hp : p = (inner_product v w / inner_product w w) • w)
hypothesis (hq : q = (inner_product p v / inner_product v v) • v)
hypothesis (h_v_norm : ∥p∥ / ∥v∥ = 4 / 11)

theorem projection_ratio : ∥q∥ / ∥v∥ = 16 / 121 :=
sorry

end projection_ratio_l794_794297


namespace eval_expression_l794_794410

theorem eval_expression :
  (sqrt(1 * (9 / 4)) - (-0.96) ^ 0 - (27 / 8) ^ (-2 / 3) + (3 / 2) ^ (-2) + ((-32) ^ (-4)) ^ (-3 / 4)) = 5 / 2 :=
by
  sorry

end eval_expression_l794_794410


namespace days_for_A_to_complete_work_l794_794080

theorem days_for_A_to_complete_work (
  B_work_days : ℕ,
  together_work_days : ℕ,
  work_left_fraction : ℝ
) : (B_work_days = 20) → (together_work_days = 5) → (work_left_fraction = 0.41666666666666663) → 
    ∃ x, 5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 ∧ x = 15 := 
by 
  intros hB ht hw 
  use 15
  sorry

end days_for_A_to_complete_work_l794_794080


namespace ratio_red_white_paint_l794_794426

theorem ratio_red_white_paint (red_needed white_needed total_paint : ℕ) 
  (h1 : total_paint = 30) 
  (h2 : red_needed = 15) : 
  red_needed / (total_paint - red_needed) = 1 := 
by 
  have h3 : total_paint - red_needed = 15 := by linarith
  have h4 : red_needed / 15 = 1 := by rw [h2, div_self]; linarith
  rwa h3 at h4

end ratio_red_white_paint_l794_794426


namespace squared_sums_of_sides_of_triangle_l794_794675

noncomputable def centroid (A B C : ℝ^3) : ℝ^3 :=
  (A + B + C) / 3

def squared_distance (P Q : ℝ^3) : ℝ :=
  ∥P - Q∥^2

theorem squared_sums_of_sides_of_triangle (A B C G : ℝ^3)
    (hG : G = centroid A B C)
    (h : squared_distance G A + squared_distance G B + squared_distance G C = 58) :
  squared_distance A B + squared_distance A C + squared_distance B C = 174 :=
begin
  sorry
end

end squared_sums_of_sides_of_triangle_l794_794675


namespace real_y_values_for_given_x_l794_794488

theorem real_y_values_for_given_x (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ (x ≤ -2 / 3 ∨ x ≥ 4) :=
by
  sorry

end real_y_values_for_given_x_l794_794488


namespace spelling_bee_participants_l794_794631

theorem spelling_bee_participants (n : ℕ)
  (h1 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (k - 1 < 74 ∨ k - 1 > 74))
  (h2 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (75 - k > 0 ∨ k - 1 > 74)) :
  n = 149 := by
  sorry

end spelling_bee_participants_l794_794631


namespace angle_c_sufficient_not_necessary_l794_794653

theorem angle_c_sufficient_not_necessary (A B C : ℝ) (h_triangle : A + B + C = π) (h_right_angle : C = π / 2) :
  (cos A + sin A = cos B + sin B) ∧ ¬(forall A B C, A + B + C = π ∧ (cos A + sin A = cos B + sin B) → C = π / 2) :=
by
  sorry

end angle_c_sufficient_not_necessary_l794_794653


namespace find_y_l794_794997

variable (a : ℝ × ℝ × ℝ)
variable (b : ℝ × ℝ × ℝ)
variable (k : ℝ)
variable (y : ℝ)

-- Conditions: 
-- 1. a = (2, 4, 5)
-- 2. b = (3, -6, y)
-- 3. a is parallel to b, meaning b = k * a
def vectors_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

theorem find_y (h_par : vectors_parallel (2,4,5) (3, -6, y)) : y = 7.5 :=
  sorry

end find_y_l794_794997


namespace arccos_zero_l794_794472

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l794_794472


namespace conditional_two_exits_one_effective_l794_794134

def conditional_structure (decide : Bool) : Prop :=
  if decide then True else False

theorem conditional_two_exits_one_effective (decide : Bool) :
  conditional_structure decide ↔ True :=
by
  sorry

end conditional_two_exits_one_effective_l794_794134


namespace quadratic_with_real_roots_l794_794528

theorem quadratic_with_real_roots: 
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4 * x₁ + k = 0 ∧ x₂^2 + 4 * x₂ + k = 0) ↔ (k ≤ 4) := 
by 
  sorry

end quadratic_with_real_roots_l794_794528


namespace min_val_f_is_three_l794_794344

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_val_f_is_three : ∀ x : ℝ, f x ≥ 3 ∧ ∃ y : ℝ, y ∈ set.Icc (-1 : ℝ) 2 ∧ f y = 3 := 
by
  intro x
  sorry

end min_val_f_is_three_l794_794344


namespace largest_odd_not_sum_of_three_distinct_composites_l794_794503

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem largest_odd_not_sum_of_three_distinct_composites :
  ∀ n : ℕ, is_odd n → (¬ ∃ (a b c : ℕ), is_composite a ∧ is_composite b ∧ is_composite c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a + b + c) → n ≤ 17 :=
by
  sorry

end largest_odd_not_sum_of_three_distinct_composites_l794_794503


namespace medicine_division_l794_794085

theorem medicine_division (weight_kg : ℝ) (dosage_ml_per_kg : ℝ) (dose_per_part_mg : ℝ) :
    (weight_kg = 30) ∧ (dosage_ml_per_kg = 5) ∧ (dose_per_part_mg = 50) →
    let full_dose_ml := weight_kg * dosage_ml_per_kg,
        full_dose_mg := full_dose_ml * 1000,
        number_of_parts := full_dose_mg / dose_per_part_mg
    in number_of_parts = 3000 :=
by
  intro h
  sorry

end medicine_division_l794_794085


namespace herder_bulls_l794_794819

theorem herder_bulls (total_bulls : ℕ) (herder_fraction : ℚ) (claims : total_bulls = 70) (fraction_claim : herder_fraction = (2/3) * (1/3)) : herder_fraction * (total_bulls : ℚ) = 315 :=
by sorry

end herder_bulls_l794_794819


namespace quadratic_coefficients_l794_794366

theorem quadratic_coefficients :
  ∃ a b c : ℤ, a = 4 ∧ b = 0 ∧ c = -3 ∧ 4 * x^2 = 3 := sorry

end quadratic_coefficients_l794_794366


namespace find_smallest_number_l794_794378

theorem find_smallest_number (a b c : ℕ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : b = 31)
  (h4 : c = b + 6)
  (h5 : (a + b + c) / 3 = 30) :
  a = 22 := 
sorry

end find_smallest_number_l794_794378


namespace average_percentage_of_10_students_l794_794827

theorem average_percentage_of_10_students 
  (avg_15_students : ℕ := 80)
  (n_15_students : ℕ := 15)
  (total_students : ℕ := 25)
  (overall_avg : ℕ := 84) : 
  ∃ (x : ℕ), ((n_15_students * avg_15_students + 10 * x) / total_students = overall_avg) → x = 90 := 
sorry

end average_percentage_of_10_students_l794_794827


namespace new_person_weight_l794_794340

-- Definitions based on given conditions
def avg_weight_increase : ℝ := 3.2
def number_of_persons : ℕ := 10
def weight_replaced_person : ℝ := 65

-- The weight of the new person that we need to prove
theorem new_person_weight :
  let total_weight_increase := number_of_persons * avg_weight_increase in
  let new_weight := weight_replaced_person + total_weight_increase in
  new_weight = 97 :=
by
  sorry

end new_person_weight_l794_794340


namespace sin_810_cos_neg60_l794_794142

theorem sin_810_cos_neg60 :
  Real.sin (810 * Real.pi / 180) + Real.cos (-60 * Real.pi / 180) = 3 / 2 :=
by
  sorry

end sin_810_cos_neg60_l794_794142


namespace eval_expression_eq_30_l794_794904

theorem eval_expression_eq_30 : ⌈4 * (8 - 3/4 + 1/4)⌉ = 30 := by
  sorry

end eval_expression_eq_30_l794_794904


namespace exists_equilateral_triangle_l794_794901

-- Define the problem and necessary conditions
variables {R : Type*}
[linear_ordered_field R] 

-- Given three parallel and equidistant lines l1, l2, l3
-- l1, l2, and l3 are parallel and located at y = y1, y = y2, y = y3 respectively

def equidistant_parallel_lines (y1 y2 y3 : R) : Prop :=
  (y2 - y1) = (y3 - y2)

-- Define what it means to be an equilateral triangle with vertices on these lines
def equilateral_triangle_on_lines (y1 y2 y3 : R) (A B C : R × R) : Prop :=
  A.2 = y1 ∧ 
  B.2 = y2 ∧ 
  C.2 = y3 ∧ 
  dist A B = dist B C ∧ 
  dist B C = dist C A

-- The main theorem to prove
theorem exists_equilateral_triangle : 
  ∀ {y1 y2 y3 : R}, 
    equidistant_parallel_lines y1 y2 y3 → 
    ∃ (A B C : R × R), 
      equilateral_triangle_on_lines y1 y2 y3 A B C :=
by
  intros y1 y2 y3 h
  -- Construct the vertices and prove the equilateral triangle
  sorry

end exists_equilateral_triangle_l794_794901


namespace compare_abc_l794_794933

noncomputable def a := Real.log 7 / Real.log 3
noncomputable def b := 3 * Real.log 2
noncomputable def c := Real.root 5 6

theorem compare_abc : b > a ∧ a > c :=
by
  have ha_upper : a < 2 := by
    sorry -- proof that a < 2
  have ha_lower : 1.5 < a := by
    sorry -- proof that 1.5 < a
  have hb : b > 2 := by
    sorry -- proof that b > 2
  have hc : c < 1.5 := by
    sorry -- proof that c < 1.5
  exact And.intro hb (And.intro ha_upper ha_lower hc)

end compare_abc_l794_794933


namespace knicks_eq_knocks_l794_794605

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l794_794605


namespace find_p_q_l794_794698

theorem find_p_q (p q : ℚ)
  (h1 : (4 : ℚ) * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2):
  (p, q) = (-29/12 : ℚ, 43/12 : ℚ) :=
by 
  sorry

end find_p_q_l794_794698


namespace points_lie_on_hyperbola_l794_794480

noncomputable def curve (t : ℝ) : ℝ × ℝ :=
  (2 * (Real.exp t + Real.exp (-t)), 4 * (Real.exp t - Real.exp (-t)))

theorem points_lie_on_hyperbola (t : ℝ) :
  let (x, y) := curve t in
  (x^2 / 10 - y^2 / 160 = 1) :=
by
  let (x, y) := curve t
  sorry

end points_lie_on_hyperbola_l794_794480


namespace imaginary_part_conjugate_z_l794_794415

def is_imaginary_part_conjugate_z_one (z : ℂ) : Prop :=
  ((z + 1) * complex.I = 1 - complex.I) → complex.im (conj z) = 1

theorem imaginary_part_conjugate_z (z : ℂ) : is_imaginary_part_conjugate_z_one z :=
by
  sorry

end imaginary_part_conjugate_z_l794_794415


namespace transport_cargo_l794_794836

theorem transport_cargo (total_weight : ℝ) (box_weight : ℝ) (truck_capacity : ℝ) (num_trucks : ℝ) :
  total_weight = 13.5 → box_weight ≤ 0.35 → truck_capacity = 1.5 → num_trucks = 11 →
  (∃ trucks: ℕ, trucks ≤ 11 ∧ total_weight / truck_capacity ≤ trucks) :=
by {
  intros h_total_weight h_box_weight h_truck_capacity h_num_trucks,
  use 11,
  split,
  { exact le_of_eq h_num_trucks },
  { rw h_truck_capacity,
    have h_nonzero : 1.5 ≠ 0 := by norm_num,
    rw ← div_eq_iff h_nonzero,
    exact h_total_weight.symm ▸ by norm_num },
  },
  sorry 
}

end transport_cargo_l794_794836


namespace log_cos_tan_defined_iff_l794_794396

theorem log_cos_tan_defined_iff (θ : Real) :
  (∃ θ ∈ Set.Ioo (0 : Real) (π), ∀ θ' ∈ Set.Ioo 0 π, θ = θ' ∧ sin θ > 0 ∧ sin θ ≠ 1) ↔ 
  (log (cos θ * tan θ)).dom = true :=
by sorry

end log_cos_tan_defined_iff_l794_794396


namespace square_diagonal_y_coordinate_l794_794270

theorem square_diagonal_y_coordinate 
(point_vertex : ℝ × ℝ) 
(x_int : ℝ) 
(area_square : ℝ) 
(y_int : ℝ) :
(point_vertex = (-6, -4)) →
(x_int = 3) →
(area_square = 324) →
(y_int = 5) → 
y_int = 5 := 
by
  intros h1 h2 h3 h4
  exact h4

end square_diagonal_y_coordinate_l794_794270


namespace unique_positive_integer_solution_l794_794924

theorem unique_positive_integer_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (x^{2 * y} + (x + 1)^{2 * y} = (x + 2)^{2 * y}) ↔ (x = 3 ∧ y = 1) := 
by
  sorry

end unique_positive_integer_solution_l794_794924


namespace monotonic_m_range_l794_794247

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 12

-- Prove the range of m where f(x) is monotonic on [m, m+4]
theorem monotonic_m_range {m : ℝ} :
  (∀ x y : ℝ, m ≤ x ∧ x ≤ m + 4 ∧ m ≤ y ∧ y ≤ m + 4 → (x ≤ y → f x ≤ f y ∨ f x ≥ f y))
  ↔ (m ≤ -5 ∨ m ≥ 2) :=
sorry

end monotonic_m_range_l794_794247


namespace fewest_tiles_needed_l794_794433

-- Define the dimensions of the tile
def tile_width : ℕ := 2
def tile_height : ℕ := 5

-- Define the dimensions of the floor in feet
def floor_width_ft : ℕ := 3
def floor_height_ft : ℕ := 6

-- Convert the floor dimensions to inches
def floor_width_inch : ℕ := floor_width_ft * 12
def floor_height_inch : ℕ := floor_height_ft * 12

-- Calculate the areas in square inches
def tile_area : ℕ := tile_width * tile_height
def floor_area : ℕ := floor_width_inch * floor_height_inch

-- Calculate the minimum number of tiles required, rounding up
def min_tiles_required : ℕ := Float.ceil (floor_area / tile_area)

-- The theorem statement: prove that the minimum tiles required is 260
theorem fewest_tiles_needed : min_tiles_required = 260 := 
  by 
    sorry

end fewest_tiles_needed_l794_794433


namespace correct_graph_representation_l794_794460

-- Define the constants for speeds and times
variables (v t : ℝ)

-- Define the speed and time of Car X
def carX_speed : ℝ := v
def carX_time : ℝ := t
def carX_distance : ℝ := carX_speed v * carX_time t

-- Define the speed and time of Car Y
def carY_speed : ℝ := 3 * carX_speed v
def carY_time : ℝ := carX_time t / 3
def carY_distance : ℝ := carY_speed v * carY_time t

-- Prove the equivalence of distances and correct graph
theorem correct_graph_representation 
  (hv : v > 0) (ht : t > 0) : 
  carX_distance v t = carY_distance v t ∧ 
  carY_speed v = 3 * carX_speed v ∧ carY_time v t = t / 3 :=
by 
  sorry

end correct_graph_representation_l794_794460


namespace largest_base_5_five_digits_base_10_value_l794_794036

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794036


namespace find_cost_price_l794_794402

noncomputable def pct_to_decimal (pct : ℝ) := pct / 100

constant selling_price : ℝ := 100
constant profit_pct : ℝ := 45

def cost_price (sp : ℝ) (pct : ℝ) := sp / (1 + pct_to_decimal pct)

theorem find_cost_price :
  cost_price selling_price profit_pct ≈ 68.97 :=
sorry

end find_cost_price_l794_794402


namespace part1_monotonic_intervals_part2_root_existence_l794_794580

-- Continuity and monotonicity analysis for the function
def monotonic_intervals (m x : ℝ) : (ℝ → ℝ) :=
  (λ x => (m * x^2 + 1) * Real.exp (-x))

theorem part1_monotonic_intervals (m : ℝ) :
  if m = 0 then
    ∀ x, (monotonic_intervals m x) < 0
  else if 0 < m ∧ m ≤ 1 then
    ∀ x, (monotonic_intervals m x) ≤ 0
  else if m > 1 then
    -- Monotonic intervals for m > 1
    sorry
  else if m < 0 then
    -- Monotonic intervals for m < 0
    sorry :=
sorry

-- Existence of a root in (0, 1) for function g and the range for m
def g (m n x : ℝ) : ℝ :=
  (m * x^2 + 1) * Real.exp (-x) + n * x * Real.exp (-x) - 1

theorem part2_root_existence (m n : ℝ) (h : m + n = Real.exp 1 - 1) :
  (∃ x, 0 < x ∧ x < 1 ∧ g m n x = 0) ↔ (e-2 < m ∧ m < 1) :=
sorry

end part1_monotonic_intervals_part2_root_existence_l794_794580


namespace find_total_photos_l794_794381

noncomputable def total_photos (T : ℕ) (Paul Tim Tom : ℕ) : Prop :=
  Tim = T - 100 ∧ Paul = Tim + 10 ∧ Tom = 38 ∧ Tom + Tim + Paul = T

theorem find_total_photos : ∃ T, total_photos T (T - 90) (T - 100) 38 :=
sorry

end find_total_photos_l794_794381


namespace number_of_concave_numbers_l794_794438

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ b < c ∧ a ≠ c

theorem number_of_concave_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ ∃ a b c, n = 100 * a + 10 * b + c ∧ is_concave_number a b c}.card = 240 :=
sorry

end number_of_concave_numbers_l794_794438


namespace median_score_interval_l794_794136

theorem median_score_interval (
    scores_60_64 : ℕ,
    scores_65_69 : ℕ,
    scores_70_74 : ℕ,
    scores_75_79 : ℕ,
    scores_80_84 : ℕ,
    scores_85_89 : ℕ,
    h_total_students : scores_60_64 + scores_65_69 + scores_70_74 + scores_75_79 + scores_80_84 + scores_85_89 = 101
  ) : 
  33 < 51 ∧ 51 ≤ 33 + scores_70_74 :=
sorry

end median_score_interval_l794_794136


namespace set_intersection_l794_794953

noncomputable def SetA : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
noncomputable def SetB : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem set_intersection :
  SetA ∩ SetB = {x | 2 < x ∧ x < 3} := by
  sorry

end set_intersection_l794_794953


namespace perpendicular_OA_PQ_eq_square_AP_2AD_OM_l794_794708

variables (A B C O D E F P Q M : Point) (triangle_ABC : Triangle A B C)
variables [acute_triangle : Triangle.AngleSumEqPI A B C]
noncomputable theory

def circumcenter (triangle_ABC : Triangle A B C) := O
def altitude_A (A B C D : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left A D B C
def altitude_B (A B C E : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left B E A C
def altitude_C (A B C F : Point) (triangle_ABC : Triangle A B C) := Line.perpendicular_left C F A B
def midpoint_BC (B C M : Point) := Midpoint B C M

theorem perpendicular_OA_PQ
  (circumcenter_O : circumcenter triangle_ABC = O)
  (altitude_AD : altitude_A A B C D triangle_ABC)
  (altitude_BE : altitude_B A B C E triangle_ABC)
  (altitude_CF : altitude_C A B C F triangle_ABC)
  (points_PQ : ∃ (P Q : Point), Line.cut_circle EF P Q)
  : Line.perpendicular O A P Q :=
sorry

theorem eq_square_AP_2AD_OM
  (midpoint_M : midpoint_BC B C M)
  : (AP.square = 2 * AD.length * OM.length) :=
sorry

end perpendicular_OA_PQ_eq_square_AP_2AD_OM_l794_794708


namespace bottles_remaining_correct_l794_794859

-- Definitions based on conditions
def small_bottles : ℕ := 6000
def big_bottles : ℕ := 14000
def medium_bottles : ℕ := 9000

def percentage_sold_small : ℚ := 20 / 100
def percentage_sold_big : ℚ := 23 / 100
def percentage_sold_medium : ℚ := 15 / 100

def sold_bottles (total : ℕ) (percent : ℚ) : ℕ := (percent * total).to_nat

-- Total bottles left calculation based on Lean definitions
def remaining_bottles : ℕ :=
  small_bottles - sold_bottles small_bottles percentage_sold_small +
  big_bottles - sold_bottles big_bottles percentage_sold_big +
  medium_bottles - sold_bottles medium_bottles percentage_sold_medium

-- Theorem statement only, no proof.
theorem bottles_remaining_correct : remaining_bottles = 23230 :=
by
  sorry

end bottles_remaining_correct_l794_794859


namespace work_days_A_l794_794077

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end work_days_A_l794_794077


namespace largest_base5_number_to_base10_is_3124_l794_794001

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794001


namespace sum_of_digits_square_1111_l794_794476

-- Define the number 1111
def num := 1111

-- Function to calculate the sum of the digits of a number
def sum_of_digits (n: ℕ) : ℕ :=
  (toString n).foldl (λ acc d, acc + (d.toNat - '0'.toNat)) 0

-- Statement: the sum of the digits of the square of the number 1111 equals 16
theorem sum_of_digits_square_1111 : sum_of_digits (num^2) = 16 :=
by
  sorry

end sum_of_digits_square_1111_l794_794476


namespace days_for_A_to_complete_work_l794_794079

theorem days_for_A_to_complete_work (
  B_work_days : ℕ,
  together_work_days : ℕ,
  work_left_fraction : ℝ
) : (B_work_days = 20) → (together_work_days = 5) → (work_left_fraction = 0.41666666666666663) → 
    ∃ x, 5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 ∧ x = 15 := 
by 
  intros hB ht hw 
  use 15
  sorry

end days_for_A_to_complete_work_l794_794079


namespace fraction_of_ABCD_is_shaded_l794_794743

noncomputable def squareIsDividedIntoTriangles : Type := sorry
noncomputable def areTrianglesIdentical (s : squareIsDividedIntoTriangles) : Prop := sorry
noncomputable def isFractionShadedCorrect : Prop := 
  ∃ (s : squareIsDividedIntoTriangles), 
  areTrianglesIdentical s ∧ 
  (7 / 16 : ℚ) = 7 / 16

theorem fraction_of_ABCD_is_shaded (s : squareIsDividedIntoTriangles) :
  areTrianglesIdentical s → (7 / 16 : ℚ) = 7 / 16 :=
sorry

end fraction_of_ABCD_is_shaded_l794_794743


namespace faye_books_l794_794908

-- Define the initial number of coloring books
def initial_books : ℕ := 34

-- Define the fraction of books given away
def fraction_given_away : ℚ := 1 / 2

-- Define the percentage of books bought later
def percentage_bought : ℚ := 60 / 100

-- Define the number of books given away
def books_given_away : ℕ := (34 * 1 / 2).toInt

-- Define the number of books left after giving away
def books_left : ℕ := initial_books - books_given_away

-- Define the number of additional books Faye bought (rounded to the nearest whole number)
def additional_books_bought : ℕ := (books_left * percentage_bought).toInt

-- Define the total number of books Faye now has
def total_books : ℕ := books_left + additional_books_bought

-- The proof statement
theorem faye_books : total_books = 27 := by
  sorry

end faye_books_l794_794908


namespace integer_multiple_of_ten_l794_794445

theorem integer_multiple_of_ten (x : ℤ) :
  10 * x = 30 ↔ x = 3 :=
by
  sorry

end integer_multiple_of_ten_l794_794445


namespace grape_juice_amount_l794_794087

theorem grape_juice_amount 
  (T : ℝ) -- total amount of the drink 
  (orange_juice_percentage watermelon_juice_percentage : ℝ) -- percentages 
  (combined_amount_of_oj_wj : ℝ) -- combined amount of orange and watermelon juice 
  (h1 : orange_juice_percentage = 0.15)
  (h2 : watermelon_juice_percentage = 0.60)
  (h3 : combined_amount_of_oj_wj = 120)
  (h4 : combined_amount_of_oj_wj = (orange_juice_percentage + watermelon_juice_percentage) * T) : 
  (T * (1 - (orange_juice_percentage + watermelon_juice_percentage)) = 40) := 
sorry

end grape_juice_amount_l794_794087


namespace value_of_fraction_l794_794490

theorem value_of_fraction (a b : ℕ) (h₁ : a = 7) (h₂ : b = 4) : 5 / (a - b)^2 = 5 / 9 :=
by
  rw [h₁, h₂]
  rw [Nat.sub_self]
  sorry

end value_of_fraction_l794_794490


namespace expression_equals_eight_l794_794380

theorem expression_equals_eight
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : b + c = 2 * a)
  (h3 : a + c = 2 * b) :
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end expression_equals_eight_l794_794380


namespace distance_D_E_l794_794315

-- Given triangle with points A, B, C, D, E and distance BC
variables (A B C D E : Type)
variable [metric_space A]

variable h1 : dist A D = dist A B
variable h2 : dist A E = dist A C
variable h3 : dist B C = 5

-- Proof problem: Prove that the distance between points D and E is 5
theorem distance_D_E (h1 : dist A D = dist A B) (h2 : dist A E = dist A C) (h3 : dist B C = 5) :
  dist D E = 5 :=
sorry

end distance_D_E_l794_794315


namespace imaginary_part_of_z_l794_794195

-- Step 1: Define the imaginary unit.
def i : ℂ := Complex.I  -- ℂ represents complex numbers in Lean and Complex.I is the imaginary unit.

-- Step 2: Define the complex number z.
noncomputable def z : ℂ := (4 - 3 * i) / i

-- Step 3: State the theorem.
theorem imaginary_part_of_z : Complex.im z = -4 :=
by 
  sorry

end imaginary_part_of_z_l794_794195


namespace evaluate_expr_at_neg3_l794_794494

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l794_794494


namespace calculate_speed_in_still_water_l794_794091

structure SwimConditions where
  downstream_distance : ℝ
  downstream_time : ℝ
  downstream_current_speed : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  upstream_current_speed : ℝ

def average_speed_in_still_water (conditions : SwimConditions) : ℝ :=
  let downstream_speed := conditions.downstream_distance / conditions.downstream_time
  let upstream_speed := conditions.upstream_distance / conditions.upstream_time
  let speed_still_water_downstream := downstream_speed - conditions.downstream_current_speed
  let speed_still_water_upstream := upstream_speed + conditions.upstream_current_speed
  (speed_still_water_downstream + speed_still_water_upstream) / 2

theorem calculate_speed_in_still_water :
  ∀ conditions : SwimConditions, 
    conditions.downstream_distance = 28 → 
    conditions.downstream_time = 2 → 
    conditions.downstream_current_speed = 3 → 
    conditions.upstream_distance = 12 → 
    conditions.upstream_time = 4 → 
    conditions.upstream_current_speed = 1 → 
    average_speed_in_still_water conditions = 7.5 :=
by
  intros conditions h1 h2 h3 h4 h5 h6
  simp [average_speed_in_still_water, h1, h2, h3, h4, h5, h6]
  sorry

end calculate_speed_in_still_water_l794_794091


namespace projection_is_constant_l794_794051

variables {a b c d : ℝ}
variables {v w p : ℝ × ℝ}

-- Defining conditions for the problem
def on_line (v : ℝ × ℝ) : Prop := v.2 = (3 / 2) * v.1 + 3
def projection (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2) • w

-- Variables for p and w
def w : ℝ × ℝ := (c, d)
def p : ℝ × ℝ := (-18 / 13, 12 / 13)

-- Theorem statement
theorem projection_is_constant (a b c d : ℝ) (v : ℝ × ℝ)
  (hv : on_line v) (hw : c + (3 / 2) * d = 0) :
  ∀ v : ℝ × ℝ, on_line v → projection v w = p :=
begin
  -- Proof will be inserted here
  sorry
end

end projection_is_constant_l794_794051


namespace lowest_score_l794_794750

theorem lowest_score 
    (mean_15 : ℕ → ℕ → ℕ → ℕ)
    (mean_13 : ℕ → ℕ → ℕ)
    (S15 : ℕ := mean_15 15 85)
    (S13 : ℕ := mean_13 13 87)
    (highest_score : ℕ := 105)
    (S_removed : ℕ := S15 - S13) :
    S_removed - highest_score = 39 := 
sorry

end lowest_score_l794_794750


namespace complex_pow_difference_l794_794599

theorem complex_pow_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 12 - (1 - i) ^ 12 = 0 :=
  sorry

end complex_pow_difference_l794_794599


namespace quadratic_function_inequality_l794_794887

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem quadratic_function_inequality (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic_function a b c x = quadratic_function a b c (2 - x)) :
  ∀ x : ℝ, quadratic_function a b c (2 ^ x) < quadratic_function a b c (3 ^ x) :=
by
  sorry

end quadratic_function_inequality_l794_794887


namespace base5_to_base10_max_l794_794022

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794022


namespace largest_integer_value_of_x_l794_794395

theorem largest_integer_value_of_x (x : ℤ) (h : 8 - 5 * x > 22) : x ≤ -3 :=
sorry

end largest_integer_value_of_x_l794_794395


namespace range_of_m_l794_794621

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m+1)*x^2 + (m+1)*x + (m+2) ≥ 0) ↔ m ≥ -1 := by
  sorry

end range_of_m_l794_794621


namespace single_discount_eq_l794_794089

/--
A jacket is originally priced at $50. It is on sale for 25% off. After applying the sale discount, 
John uses a coupon that gives an additional 10% off of the discounted price. If there is a 5% sales 
tax on the final price, what single percent discount (before taxes) is equivalent to these series 
of discounts followed by the tax? --/
theorem single_discount_eq :
  let P0 := 50
  let discount1 := 0.25
  let discount2 := 0.10
  let tax := 0.05
  let discounted_price := P0 * (1 - discount1) * (1 - discount2)
  let after_tax_price := discounted_price * (1 + tax)
  let single_discount := (P0 - discounted_price) / P0
  single_discount * 100 = 32.5 :=
by
  sorry

end single_discount_eq_l794_794089


namespace hyperbola_eccentricity_sqrt2_plus_1_l794_794767

-- Define the conditions provided in the problem
def conditions_hyperbola_parabola :
  Prop := 
  let f2 : Point := (1, 0) in 
  let parabola := {p : Point | p.y^2 = 4 * p.x} in
  let intercept (C : Curve) (par : Curve) := ∃ A : Point, A ∈ C ∧ A ∈ par in
  let hyperbola := λ C : Curve, ∃ f1 f2 : Point, f2 = (1, 0) ∧ intercept C parabola ∧ 
                    (∃ A : Point, A ∈ hyperbola ∧ A ∈ parabola ∧ 
                    ∃ F₁ F₂ : Point, F₂ = f2 ∧ (triangle_isosceles_base A F₁ F₂)) in 
  ∃ C : Curve, hyperbola C

-- Prove the eccentricity of the hyperbola C equals sqrt(2) + 1 given the conditions
theorem hyperbola_eccentricity_sqrt2_plus_1 : 
  conditions_hyperbola_parabola → ∃ e : ℝ, e = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_sqrt2_plus_1_l794_794767


namespace arccos_zero_eq_pi_div_two_l794_794469

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l794_794469


namespace range_of_a_l794_794533

noncomputable 
theorem range_of_a {a : ℝ} : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), x^2 - 2 * a * x + 2 ≥ a) ↔ a ∈ Iic (1 : ℝ) :=
by
  sorry

end range_of_a_l794_794533


namespace fixed_point_of_exponential_function_l794_794976

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
  let f := λ x : ℝ, a^(2*x - 1) + 2 in
  f (1 / 2) = 3 :=
by
  let f := λ x : ℝ, a^(2*x - 1) + 2
  sorry

end fixed_point_of_exponential_function_l794_794976


namespace area_of_triangle_l794_794209

-- Definitions based on given conditions
def square_area_1 : ℝ := 121
def square_area_2 : ℝ := 64
def square_area_3 : ℝ := 225

-- Lengths of the sides of the triangle
def leg1 : ℝ := Real.sqrt square_area_1
def leg2 : ℝ := Real.sqrt square_area_2

-- The math proof problem statement
theorem area_of_triangle : 
  0.5 * leg1 * leg2 = 44 := 
by
  unfold leg1 leg2
  sorry

end area_of_triangle_l794_794209


namespace problem1_l794_794822

noncomputable def m (b a : ℝ) : ℝ := Real.pow (a + 4) (1 / (b - 1))
noncomputable def n (a b : ℝ) : ℝ := Real.pow (3 * b - 1) (1 / (a - 2))

theorem problem1 (b a : ℝ) (hb : b = 3) (ha : a = 5) :
  Real.cbrt (m b a - 2 * n a b) = -1 :=
by
  sorry

end problem1_l794_794822


namespace min_value_a4b3c2_l794_794300

theorem min_value_a4b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : (∀ a b c : ℝ, a^4 * b^3 * c^2 ≥ 1/(9^9)) :=
by
  sorry

end min_value_a4b3c2_l794_794300


namespace revolutions_per_minute_gear_p_l794_794462

-- Define conditions
def rpm_q : ℕ := 40

-- Calculate revolutions per second for gear q
def rps_q : ℝ := rpm_q / 60.0

-- Calculate revolutions in 4 seconds for gear q
def revs_q_4s : ℝ := rps_q * 4

-- Gear p makes 2 fewer revolutions than gear q in 4 seconds
def revs_p_4s : ℝ := revs_q_4s - 2

-- Calculate revolutions per minute for gear p from revolutions in 4 seconds
def rpm_p : ℝ := (revs_p_4s * 60) / 4

-- Proof statement
theorem revolutions_per_minute_gear_p : rpm_p = 10 := 
by 
  -- Placeholder for proof steps, if required
  sorry

end revolutions_per_minute_gear_p_l794_794462


namespace arccos_zero_eq_pi_div_two_l794_794464

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l794_794464


namespace find_line_m_l794_794962

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 2 * y + 8 = 0
def parallel_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def symmetric_line (m : ℝ → ℝ → Prop) := ∀ x y, circle_eq x y → m x y = 0

theorem find_line_m : 
  (∃ (m : ℝ → ℝ → Prop), symmetric_line m ∧ (∀ x y, parallel_line x y → parallel_line x y = m x y) ∧ ∀ x y, m x y := 2 * x - y - 7) :=
sorry

end find_line_m_l794_794962


namespace area_of_triangle_MDA_l794_794257

theorem area_of_triangle_MDA (r : ℝ) (O A B M D : EuclideanSpace ℝ 2):
  dist O A = r ∧
  dist O B = r ∧
  dist A B = r * Real.sqrt 2 ∧
  (∃ M, dist (orthProj ℝ (affineSpanℝ {O, B}) M) M = 0) ∧
  (∃ D, dist (orthProj ℝ (affineSpanℝ {O, A}) D) D = 0 ∧ dist O A * dist M D = r^2 / 4) →
  area (triangle M D A) = r^2 / 4 :=
sorry

end area_of_triangle_MDA_l794_794257


namespace additional_time_needed_l794_794423

theorem additional_time_needed (total_parts apprentice_first_phase remaining_parts apprentice_rate master_rate combined_rate : ℕ)
  (h1 : total_parts = 500)
  (h2 : apprentice_first_phase = 45)
  (h3 : remaining_parts = total_parts - apprentice_first_phase)
  (h4 : apprentice_rate = 15)
  (h5 : master_rate = 20)
  (h6 : combined_rate = apprentice_rate + master_rate) :
  remaining_parts / combined_rate = 13 := 
by {
  sorry
}

end additional_time_needed_l794_794423


namespace count_valid_numbers_l794_794236

-- Definition of the problem conditions
def is_valid_number (n: ℕ) : Prop :=
 n >= 200 ∧ n <= 998 ∧
 (n % 2 = 0) ∧
 let digits := List.ofDigits (Nat.digits 10 n) in
 digits.Nodup

-- The statement to be proved
theorem count_valid_numbers : 
  { n // is_valid_number n }.count = 408 :=
sorry

end count_valid_numbers_l794_794236


namespace length_of_AE_l794_794483

theorem length_of_AE
  (AB CD AC : ℝ)
  (hAB : AB = 12)
  (hCD : CD = 16)
  (hAC : AC = 20)
  (hEqualAreas: ∃ E : ℝ, AC = AE + E ∧ triangle_area A E D = triangle_area B E C) :
  AE = 10 :=
by
  sorry

end length_of_AE_l794_794483


namespace chessboard_partition_l794_794886

theorem chessboard_partition :
  ∃ (p : ℕ) (a : Fin p → ℕ),
    (p = 7) ∧
    (∀ i, a i > 0) ∧
    (∀ i j, i < j → a i < a j) ∧
    (Finset.univ.sum a = 32) ∧
    ((a = [1, 2, 3, 4, 5, 7, 10]) ∨ (a = [1, 2, 3, 4, 5, 8, 9]) ∨ 
     (a = [1, 2, 3, 4, 6, 7, 9]) ∨ (a = [1, 2, 3, 5, 6, 7, 8])) :=
begin
  sorry
end

end chessboard_partition_l794_794886


namespace sum_of_roots_l794_794224

theorem sum_of_roots (m n : ℝ) (h1 : m ≠ 1) (h2 : n ≠ -m) :
  (let a := 1
       b := (m - 1)
       c := (m + n)
   in -(b / a) = 1 - m) :=
by
  sorry

end sum_of_roots_l794_794224


namespace policeman_catches_thief_l794_794635

def same_parity (a b : ℕ) : Prop := (a % 2 = b % 2)

theorem policeman_catches_thief (i j : ℕ) : 
  let start_policeman := (1, 1)
  let start_thief := (i, j)
  same_parity 1 1 → same_parity i j :=
begin
  -- Proof would go here
  sorry
end

end policeman_catches_thief_l794_794635


namespace minimum_value_1_div_Sp_plus_1_div_Sq_l794_794179

section ProofProblem

variables {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {p q : ℕ}

axiom a_1 : a 1 = 1
axiom S_def : ∀ n, S n = ∑ i in finset.range (n + 1), a i
axiom Sn_Sm : ∀ (m n : ℕ), m < n → S n - S m = 2^m * S (n - m)
axiom pq_sum : p + q = 6

theorem minimum_value_1_div_Sp_plus_1_div_Sq :
  (1:ℝ) / S p + (1:ℝ) / S q ≥ 1 / 4 :=
begin
  sorry
end

end ProofProblem

end minimum_value_1_div_Sp_plus_1_div_Sq_l794_794179


namespace part1_part2_l794_794992

variable (n p q m : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  -x^2 + 2 * m * x - 3

def point_A := (n-2, p)
def point_B := (4, q)
def point_C := (n, p)

theorem part1 (h1 : m > 0) (h2 : quadratic_function n = p) (h3 : quadratic_function (n-2) = p) : 
  m = n - 1 :=
by sorry

theorem part2 (h1 : -3 < q) (h2 : q < p) (h3 : quadratic_function 4 = q) (h4 : quadratic_function n = p) :
  3 < n ∧ n < 4 ∨ n > 6 :=
by sorry

end part1_part2_l794_794992


namespace convert_base_10_to_base_7_l794_794891

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 3500) : 
  ∃ k : ℕ, k = 13130 ∧ n = 1 * 7^4 + 3 * 7^3 + 1 * 7^2 + 3 * 7^1 + 0 * 7^0 :=
by
  use 13130
  split
  { refl }
  { rw h
    norm_num }
  sorry

end convert_base_10_to_base_7_l794_794891


namespace range_of_a_l794_794223

open Real

noncomputable def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

noncomputable def prop_q (a : ℝ) : Prop :=
  let f := λ x : ℝ, x^2 + (a - 1) * x + 1
  in 0 < a + 2 * (a - 1)^2 + (a - 1)^3 ∧ (0 < -a^2 + 2 * a - 1 < 0)

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a ∧ ¬ (prop_p a ∧ prop_q a)) → (-2 < a ∧ a ≤ -3/2) ∨ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l794_794223


namespace polynomial_exists_iff_l794_794911

noncomputable def P(x : ℝ) := x^2 + c * x + d

theorem polynomial_exists_iff (a b : ℝ) :
  (∃ (P : ℝ → ℝ), ∀ (x : ℝ), P(P(x)) = x^4 - 8x^3 + a * x^2 + b * x + 40) ↔ 
  (a = 28 ∧ b = -48) ∨ (a = 2 ∧ b = 56) :=
by
  sorry

end polynomial_exists_iff_l794_794911


namespace jewelry_store_gross_profit_l794_794814

theorem jewelry_store_gross_profit (purchase_price selling_price new_selling_price gross_profit : ℝ)
    (h1 : purchase_price = 240)
    (h2 : markup = 0.25 * selling_price)
    (h3 : selling_price = purchase_price + markup)
    (h4 : decrease = 0.20 * selling_price)
    (h5 : new_selling_price = selling_price - decrease)
    (h6 : gross_profit = new_selling_price - purchase_price) :
    gross_profit = 16 :=
by
    sorry

end jewelry_store_gross_profit_l794_794814


namespace num_distinct_integers_written_as_sums_of_special_fractions_l794_794876

theorem num_distinct_integers_written_as_sums_of_special_fractions :
  let special_fraction (a b : ℕ) := a + b = 15
  ∃ n : ℕ, n = 11 ∧ 
    ∀ i j : ℕ, special_fraction i j → 
      ∃ k : ℕ, 
        is_sum_of_special_fractions i j k → k < 29 := sorry

def is_sum_of_special_fractions (i j k : ℕ) : Prop := -- Custom definition to define sum of special fractions.
  -- details to be filled in as necessary
  sorry

end num_distinct_integers_written_as_sums_of_special_fractions_l794_794876


namespace find_p_q_l794_794695

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l794_794695


namespace base5_to_base10_max_l794_794024

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794024


namespace part1_part2_l794_794176

noncomputable def f (x : ℝ) := |x - 3| + |x - 4|

theorem part1 (a : ℝ) (h : ∃ x : ℝ, f x < a) : a > 1 :=
sorry

theorem part2 (x : ℝ) : f x ≥ 7 + 7 * x - x ^ 2 ↔ x ≤ 0 ∨ 7 ≤ x :=
sorry

end part1_part2_l794_794176


namespace imaginary_part_of_z_l794_794354

noncomputable def complex_number (z : ℂ) : Prop :=
z * (2 + Complex.i) = 3 - 6 * Complex.i

theorem imaginary_part_of_z (z : ℂ) (h : complex_number z) : z.im = -3 := by
  sorry

end imaginary_part_of_z_l794_794354


namespace ellipse_equation_product_of_slopes_l794_794184

-- Definitions based on conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a c : ℝ) : Prop :=
  c / a = (sqrt 3) / 2

def point_on_ellipse (A : ℝ × ℝ) (a b : ℝ) : Prop :=
  A.1 ^ 2 / a ^ 2 + A.2 ^ 2 / b ^ 2 = 1

def intersect_with_circle (l : ℝ → ℝ) (P1 P2 : ℝ × ℝ) : Prop :=
  P1.1^2 + P1.2^2 = 5 ∧ P2.1^2 + P2.2^2 = 5

def slopes (O P1 P2 : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  k1 = P1.2 / P1.1 ∧ k2 = P2.2 / P2.1 ∧ P1.1 ≠ 0 ∧ P2.1 ≠ 0

-- Lean theorem statements for the problem
theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ) (h_a_b : a > b ∧ b > 0)
  (h_ecc : eccentricity a (sqrt (a^2 - b^2)))
  (h_point : point_on_ellipse A a b) :
  (∃ a b : ℝ, a = 2 ∧ b = 1 ∧ is_ellipse A.1 A.2 a b) :=
sorry

theorem product_of_slopes (A P1 P2 : ℝ × ℝ) (a b : ℝ) (l : ℝ → ℝ)
    (k1 k2 : ℝ) (h1 : is_ellipse A.1 A.2 a b)
    (h2 : a = 2 ∧ b = 1) (h3 : intersect_with_circle l P1 P2)
    (h4 : slopes (0,0) P1 P2 k1 k2) :
  k1 * k2 = -1 / 4 :=
sorry

end ellipse_equation_product_of_slopes_l794_794184


namespace solution_set_of_inequality_l794_794783

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2*x + 3 > 0 ↔ (-1 < x ∧ x < 3) :=
sorry

end solution_set_of_inequality_l794_794783


namespace equilateral_triangle_CP_length_equilateral_triangle_lambda_value_equilateral_triangle_lambda_range_l794_794185

-- Definition of equilateral triangle ABC
structure EquilateralTriangle := 
  (A B C : Point)
  (side_length : ℝ)
  (side_length_pos : side_length > 0)
  (equilateral : dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length)

-- Point P on segment AB, 0 <= λ <= 1 
variables (A B C : Point) (side_length : ℝ) (hAB : dist A B = side_length)
  (P : Point) (λ : ℝ) (hλ : 0 ≤ λ ∧ λ ≤ 1) (hP : dist A P = λ * dist A B)

-- Given λ = 1/3, prove |CP| = 2√7
theorem equilateral_triangle_CP_length
  (hABC : EquilateralTriangle A B C side_length)
  (hλ_13 : λ = 1/3)
  : dist C P = 2 * real.sqrt 7 :=
  sorry

-- Given AP = (3/5) PB, prove λ = 3/8
theorem equilateral_triangle_lambda_value
  (hABC : EquilateralTriangle A B C side_length)
  (hAP_PB : dist A P = (3/5) * dist P B)
  : λ = 3/8 :=
  sorry

-- Given CP•AB >= PA•PB, find range of λ
theorem equilateral_triangle_lambda_range
  (hABC : EquilateralTriangle A B C side_length)
  (hCP_dot_AB : vector_dot (vector C P) (vector A B))
  (hPA_dot_PB : vector_dot (vector P A) (vector P B))
  (hineq : vector_dot (vector C P) (vector A B) ≥ vector_dot (vector P A) (vector P B))
  : (2 - real.sqrt 2) / 2 ≤ λ ∧ λ ≤ 1 :=
  sorry

end equilateral_triangle_CP_length_equilateral_triangle_lambda_value_equilateral_triangle_lambda_range_l794_794185


namespace expected_value_gt_median_l794_794838

variable {a b : ℝ} (f : ℝ → ℝ) [measure_space ℝ] (X : ℝ → ℝ) [is_probability_measure X]

-- Assume the conditions of the problem
-- f is the probability density function of the random variable X
-- f(x) = 0 for x < a and x >= b
-- f is continuous, positive, and monotonically decreasing on [a, b)

def pdf_support (x : ℝ) : Prop := (a ≤ x) ∧ (x < b)
def pdf_zero_outside (x : ℝ) : Prop := (x < a) ∨ (x ≥ b) → f x = 0
def pdf_positive (x : ℝ) : Prop := pdf_support x → 0 < f x
def pdf_monotonic (x₁ x₂ : ℝ) : Prop := (pdf_support x₁ ∧ pdf_support x₂ ∧ x₁ < x₂) → f x₁ ≥ f x₂

theorem expected_value_gt_median
  (median_zero : ∫ x in (set_of pdf_support), f x dX = 0.5) 
  (h_zero_outside : ∀ x, pdf_zero_outside x) 
  (h_continuous : continuous_on f (set_of pdf_support))
  (h_positive : ∀ x, pdf_positive x)
  (h_monotonic : ∀ x₁ x₂, pdf_monotonic x₁ x₂) :
  ∫ x, x * f x dX > 0 := 
by
  sorry

end expected_value_gt_median_l794_794838


namespace number_of_5_dollar_bills_l794_794832

theorem number_of_5_dollar_bills (x y : ℝ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
sorry

end number_of_5_dollar_bills_l794_794832


namespace find_angle_LBC_l794_794277

variables {A B C H L : Type*} [EuclideanGeometry B H]
variables (BL BH BC AC right_angle : ℝ)
variables {AH : ℝ}

-- Conditions
def conditions :=
  (right_angle > 0) ∧
  (BL = 4) ∧
  (AH = 9 / (2 * Real.sqrt 7)) ∧
  (BH * BH = BL * BL - ((AH * 2 * (Real.sqrt 7) - 1)^2))

-- Proof goal
theorem find_angle_LBC
  (A B C H L : Type*)
  [EuclideanGeometry B H]
  (BL BH BC AC right_angle : ℝ)
  (AH : ℝ)
  (hcond : conditions A B C H L BL BH BC AC right_angle AH) :
  ∃ θ, θ = Real.arccos (23 / (4 * Real.sqrt 37)) :=
sorry

end find_angle_LBC_l794_794277


namespace gifts_wrapped_with_third_roll_l794_794730

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l794_794730


namespace cos_double_angle_zero_l794_794956

theorem cos_double_angle_zero (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = Real.cos (Real.pi / 6 + α)) : Real.cos (2 * α) = 0 := 
sorry

end cos_double_angle_zero_l794_794956


namespace sin_squared_sum_l794_794587

-- angles of a triangle
variables {A B C : ℝ}
hypothesis h_triangle_angles : A + B + C = Real.pi

theorem sin_squared_sum (h_triangle_angles : A + B + C = Real.pi) :
  sin A ^ 2 + sin B ^ 2 + sin C ^ 2 - 2 * cos A * cos B * cos C = 2 := by
  sorry

end sin_squared_sum_l794_794587


namespace hh3_eq_2943_l794_794610

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 2

-- Prove that h(h(3)) = 2943
theorem hh3_eq_2943 : h (h 3) = 2943 :=
by
  sorry

end hh3_eq_2943_l794_794610


namespace maximize_profit_l794_794757

noncomputable def profit_function (x : ℝ) : ℝ :=
  (x - 30) * (200 - x)

theorem maximize_profit : ∃ x : ℝ, (∀ y : ℝ, profit_function x ≥ profit_function y) ∧ x = 115 :=
by
  let S := profit_function
  let dSdx := (fun x => -2 * x + 230)
  have h1 : dSdx 115 = 0 := by
    -- the proof calculation steps here
    sorry
  have h2 : ∀ y, (dSdx y = 0 → y = 115) := by
    -- the proof calculation steps here
    sorry
  have h3 : ∀ y, S(115) ≥ S(y) := by
    -- the proof calculation steps here
    sorry
  use 115
  exact ⟨h3, rfl⟩

end maximize_profit_l794_794757


namespace exists_zero_point_in_interval_l794_794216

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem exists_zero_point_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
by
  -- The proof details would go here
  -- Ensure there are necessary imports if any more are required
  sorry

end exists_zero_point_in_interval_l794_794216


namespace solve_inequality_l794_794740

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end solve_inequality_l794_794740


namespace jacob_charge_per_rung_is_2_l794_794663

-- Definitions based on provided conditions
def ladders_with_50_rungs : ℕ := 10
def rungs_per_50_ladder : ℕ := 50
def ladders_with_60_rungs : ℕ := 20
def rungs_per_60_ladder : ℕ := 60
def total_amount_paid : ℝ := 3400

-- Compute total number of rungs based on the conditions
def total_rungs : ℕ := ladders_with_50_rungs * rungs_per_50_ladder + ladders_with_60_rungs * rungs_per_60_ladder

-- Define the charge per rung
def charge_per_rung : ℝ := total_amount_paid / total_rungs

-- The theorem that we need to prove
theorem jacob_charge_per_rung_is_2 : charge_per_rung = 2 := by
  sorry

end jacob_charge_per_rung_is_2_l794_794663


namespace knocks_to_knicks_l794_794601

def knicks := ℕ
def knacks := ℕ
def knocks := ℕ

axiom knicks_to_knacks_ratio (k : knicks) (n : knacks) : 5 * k = 3 * n
axiom knacks_to_knocks_ratio (n : knacks) (o : knocks) : 4 * n = 6 * o

theorem knocks_to_knicks (k : knicks) (n : knacks) (o : knocks) (h1 : 5 * k = 3 * n) (h2 : 4 * n = 6 * o) :
  36 * o = 40 * k :=
sorry

end knocks_to_knicks_l794_794601


namespace polynomial_divisible_by_x_minus_one_cubed_l794_794324

theorem polynomial_divisible_by_x_minus_one_cubed (n : ℕ) :
  let P := λ x : ℝ, x^(2*n) - n^2 * x^(n+1) + 2*(n^2 - 1)*x^n + 1 - n^2 * x^(n-1)
  in (x - 1)^3 ∣ P x :=
by
  sorry

end polynomial_divisible_by_x_minus_one_cubed_l794_794324


namespace possible_n_values_l794_794796

theorem possible_n_values (n : ℕ) (h : n ≥ 3) (total_people : 20)
  (wise_men : 11) (jesters : 9)
  (around_table : ∀ k : Fin n, k.1 < n → (k.1 + 1) % n < n) :
  n ∈ {3, 4, 5, 6, 7, 8, 9, 12, 15} :=
sorry

end possible_n_values_l794_794796


namespace solution_set_of_inequality_l794_794760

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h1 : ∀ x, f x ∈ ℝ)
                                   (h2 : f (-2) = 2018)
                                   (h3 : ∀ x, f' x < 2 * x) :
                                   {x | f x < x^2 + 2014} = set.Ioi (-2) :=
begin
  sorry
end

end solution_set_of_inequality_l794_794760


namespace sum_of_solutions_eq_zero_l794_794518

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | -12*x/(x^2-1) = 3*x/(x+1) - 9/(x-1)}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l794_794518


namespace parallel_symmetric_lines_l794_794636

/-- Given an acute triangle ABC with altitudes AA', BB', and CC', and symmetric points C_a, C_b, A_b, A_c, B_c, B_a
     as described in the problem, prove that the lines A_bB_a, B_cC_b, and C_aA_c are parallel.
-/
theorem parallel_symmetric_lines (A B C A' B' C' A_b A_c B_c B_a C_a C_b : Type)
  [hacute : TriangleAcute A B C]
  [haltitudes : IsAltitude A A' B B' C C']
  [ha_symm : SymmetricPoints A A' C' C_a]
  [hb_symm : SymmetricPoints B B' C' C_b]
  [ac_symm : SymmetricPoints A A' B' B_a]
  [bc_symm : SymmetricPoints B B' A' A_b]
  [ca_symm : SymmetricPoints C C' A' A_c]
  [cb_symm : SymmetricPoints C C' B' B_c] :
  ParallelLines A_b B_a B_c C_b C_a A_c :=
sorry

end parallel_symmetric_lines_l794_794636


namespace greatest_possible_remainder_l794_794589

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end greatest_possible_remainder_l794_794589


namespace sufficient_but_not_necessary_condition_l794_794654

theorem sufficient_but_not_necessary_condition (A B C : ℝ) (T : Triangle) :
  T.isRightAngle C ↔ (T.angle A + T.angle B = 90 ∧ 
                      (cos (T.angle A) + sin (T.angle A) = cos (T.angle B) + sin (T.angle B)) :=
by sorry

end sufficient_but_not_necessary_condition_l794_794654


namespace frequency_distribution_table_understanding_l794_794484

theorem frequency_distribution_table_understanding (size_sample_group : Prop) :
  (∃ (size_proportion : Prop) (corresponding_situation : Prop),
    size_sample_group → size_proportion ∧ corresponding_situation) :=
sorry

end frequency_distribution_table_understanding_l794_794484


namespace compute_m_div_18_l794_794159

noncomputable def ten_pow (n : ℕ) : ℕ := Nat.pow 10 n

def valid_digits (m : ℕ) : Prop :=
  ∀ d ∈ m.digits 10, d = 0 ∨ d = 8

def is_multiple_of_18 (m : ℕ) : Prop :=
  m % 18 = 0

theorem compute_m_div_18 :
  ∃ m, valid_digits m ∧ is_multiple_of_18 m ∧ m / 18 = 493827160 :=
by
  sorry

end compute_m_div_18_l794_794159


namespace find_percentage_x_l794_794860

noncomputable def percentage_x (P : ℝ) : Prop :=
  let solution_y_volume  := 100
  let solution_x_volume  := 300
  let total_volume       := solution_x_volume + solution_y_volume
  let solution_y_alcohol := 0.30 * solution_y_volume
  let final_alcohol_vol  := 0.15 * total_volume
  P * solution_x_volume + solution_y_alcohol = final_alcohol_vol

theorem find_percentage_x : ∃ P : ℝ, percentage_x P ∧ P = 0.10 :=
by
  let solution_y_volume  := 100
  let solution_x_volume  := 300
  let total_volume       := solution_x_volume + solution_y_volume
  let solution_y_alcohol := 0.30 * solution_y_volume
  let final_alcohol_vol  := 0.15 * total_volume
  use (0.10 : ℝ)
  simp [percentage_x]
  sorry

end find_percentage_x_l794_794860


namespace fewest_tiles_needed_l794_794430

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end fewest_tiles_needed_l794_794430


namespace new_person_weight_l794_794752

theorem new_person_weight (W x : ℝ) (h1 : (W - 55 + x) / 8 = (W / 8) + 2.5) : x = 75 := by
  -- Proof omitted
  sorry

end new_person_weight_l794_794752


namespace problem1_problem2_l794_794455

-- Problem (1)
theorem problem1 : 
  0.25 * (- (1 / 2))^(-4) - 4 / 2^0 - (1 / 16)^(- (1 / 2)) = -4 :=
by sorry

-- Problem (2)
theorem problem2 : 
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - (log 4 3 + log 8 3) * (log 3 2 + log 9 2) = 3 / 4 :=
by sorry

end problem1_problem2_l794_794455


namespace boat_speed_downstream_l794_794833

theorem boat_speed_downstream (s_still_water s_upstream: ℝ) (h1: s_still_water = 11) (h2: s_upstream = 7) :
  s_still_water + (s_still_water - s_upstream) = 15 :=
by
  rw [h1, h2]
  simp
  sorry

end boat_speed_downstream_l794_794833


namespace factorization_eq1_factorization_eq2_l794_794148

-- Definitions for the given conditions
variables (a b x y m : ℝ)

-- The problem statement as Lean definitions and the goal theorems
def expr1 : ℝ := -6 * a * b + 3 * a^2 + 3 * b^2
def factored1 : ℝ := 3 * (a - b)^2

def expr2 : ℝ := y^2 * (2 - m) + x^2 * (m - 2)
def factored2 : ℝ := (m - 2) * (x + y) * (x - y)

-- Theorem statements for equivalence
theorem factorization_eq1 : expr1 a b = factored1 a b :=
by
  sorry

theorem factorization_eq2 : expr2 x y m = factored2 x y m :=
by
  sorry

end factorization_eq1_factorization_eq2_l794_794148


namespace cost_of_each_taco_l794_794058

variables (T E : ℝ)

-- Conditions
axiom condition1 : 2 * T + 3 * E = 7.80
axiom condition2 : 3 * T + 5 * E = 12.70

-- Question to prove
theorem cost_of_each_taco : T = 0.90 :=
by
  sorry

end cost_of_each_taco_l794_794058


namespace gcd_7429_12345_l794_794502

theorem gcd_7429_12345 : Int.gcd 7429 12345 = 1 := 
by 
  sorry

end gcd_7429_12345_l794_794502


namespace triangle_inequality_equality_condition_l794_794689

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_equality_condition_l794_794689


namespace farmer_fatima_goats_sold_l794_794500

theorem farmer_fatima_goats_sold : ∃ c g : ℕ, c + g = 80 ∧ 2 * c + 4 * g = 200 ∧ g = 20 :=
by {
  use 60, 20,
  split,
  { exact rfl, },  -- c + g = 80 is true for c = 60, g = 20
  split,
  { norm_num, },   -- 2*c + 4*g = 200 is true for c = 60, g = 20
  { norm_num, }    -- g = 20 is by construction
}

end farmer_fatima_goats_sold_l794_794500


namespace true_propositions_count_l794_794701

theorem true_propositions_count (a b c : ℝ) : 
  let original := (a > b) → (a * c^2 > b * c^2)
  let converse := (a * c^2 > b * c^2) → (a > b)
  let inverse := (a ≤ b) → (a * c^2 ≤ b * c^2)
  let contrapositive := (a * c^2 ≤ b * c^2) → (a ≤ b) in
  (cond ((a > b) = (a * c^2 > b * c^2), cond ((a * c^2 > b * c^2) = (a > b), 
        cond ((a ≤ b) = (a * c^2 ≤ b * c^2), cond ((a * c^2 ≤ b * c^2) = (a ≤ b), 
        4, 3), 2), 1), 0) = 2 := 
by
  sorry

end true_propositions_count_l794_794701


namespace twin_prime_probability_split_17_l794_794093

theorem twin_prime_probability_split_17 :
  let primes := {2, 3, 5, 7}
  let twin_prime (p q : ℕ) := (Nat.Prime p) ∧ (Nat.Prime q) ∧ (|p - q| = 2)
  let twin_prime_pairs := (primes.toFinset.pairCombinations.filter (fun pq => twin_prime pq.1 pq.2)).toFinset
  let total_combinations := primes.toFinset.pairCombinations.card
  let twin_prime_probability := twin_prime_pairs.card / total_combinations
  twin_prime_probability = 1/3 := 
by
  sorry

end twin_prime_probability_split_17_l794_794093


namespace sum_of_2009th_powers_divisible_by_2009_l794_794722

theorem sum_of_2009th_powers_divisible_by_2009 :
  (∑ k in Finset.range 2009, k ^ 2009) % 2009 = 0 :=
sorry

end sum_of_2009th_powers_divisible_by_2009_l794_794722


namespace jellybean_probability_l794_794075

theorem jellybean_probability :
  let total_ways := Nat.choose 15 4
  let red_ways := Nat.choose 5 2
  let blue_ways := Nat.choose 3 2
  let favorable_ways := red_ways * blue_ways
  let probability := favorable_ways / total_ways
  probability = (2 : ℚ) / 91 := by
  sorry

end jellybean_probability_l794_794075


namespace inequality_system_two_integer_solutions_l794_794617

theorem inequality_system_two_integer_solutions (m : ℝ) : (-1 : ℝ) ≤ m ∧ m < 0 ↔ ∃ x : ℤ, (x < 1) ∧ (x > m - 1) ∧ {
  (∃ y : ℤ, (y < 1) ∧ (y > m - 1) ∧ x ≠ y)
  ∧ ∀ z : ℤ, (z < 1) ∧ (z > m - 1) → (z = x ∨ z = y)
}

end inequality_system_two_integer_solutions_l794_794617


namespace find_projection_vector_l794_794053

variable (a c d : ℝ)

def projection_constant (v w : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let norm_w := w.1^2 + w.2^2
  let proj_v_w := (v.1 * w.1 + v.2 * w.2) / norm_w * w in
  proj_v_w = p

theorem find_projection_vector (c d : ℝ) (h : c = -3/2 * d) :
  ∀ a : ℝ, projection_constant (a, 3/2 * a + 3) (c, d) (-18/13, 12/13) :=
by
  intro a
  rw [projection_constant, h]
  sorry

end find_projection_vector_l794_794053


namespace average_value_of_u_on_0_1_is_2_l794_794913

def u (x : ℝ) := 1 / Real.sqrt x

noncomputable def avg_value_on_half_interval : ℝ :=
  lim (standardPartitions (Ioo 0 1)) (λ t, 1 / (bdryMeasure t) 0 * (integral u t) 0)

theorem average_value_of_u_on_0_1_is_2 :
  avg_value_on_half_interval = 2 := sorry

end average_value_of_u_on_0_1_is_2_l794_794913


namespace choose_students_l794_794376

noncomputable def triangle_exists (A B C : Finset ℕ) (n : ℕ)
  (hA : A.card = n) (hB : B.card = n) (hC : C.card = n)
  (hK : ∀ (a ∈ A) (b ∈ B) (c ∈ C), (∃ (k : ℕ), (k ∈ B ∪ C \ (finset.singleton b ∪ finset.singleton c))) ∨ 
                                      ∃ (k : ℕ), (k ∈ A ∪ C \ (finset.singleton a ∪ finset.singleton c)) ∨ 
                                      ∃ (k : ℕ), (k ∈ A ∪ B \ (finset.singleton a ∪ finset.singleton b))) : Prop :=
  ∃ (x ∈ A) (y ∈ B) (z ∈ C), 
    (x ∈ B ∪ C) ∧ (y ∈ A ∪ C) ∧ (z ∈ A ∪ B)

theorem choose_students (A B C : Finset ℕ) (n : ℕ)
  (hA : A.card = n) (hB : B.card = n) (hC : C.card = n)
  (hK : ∀ x ∈ A ∪ B ∪ C, (A.BUnion A.Pred) (B.BUnion B.Pred).sum (C.BUnion C.Pred).succ): ∃ (x ∈ A) (y ∈ B) (z ∈ C), 
    (x ∈ B ∪ C) ∧ (y ∈ A ∪ C) ∧ (z ∈ A ∪ B) :=
 sorry

end choose_students_l794_794376


namespace largest_base5_number_conversion_l794_794007

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794007


namespace morgan_foggy_time_l794_794312

theorem morgan_foggy_time :
  let d := 30 / 2 in       -- Each distance part is half of 30 miles
  let t_clear := d / 45 in  -- Time in clear weather
  let t_foggy := d / 15 in  -- Time in foggy weather
  let total_time := t_clear + t_foggy in
  total_time = 1 → t_foggy = 1 / 3 :=   -- Total time is 1 hour implies time in foggy is 1/3 hour
begin
  intros d t_clear t_foggy total_time h,
  have h1 : d = 15, by linarith,
  have h2 : t_clear = 15 / 45, by rw [h1],
  have h3 : t_foggy = 15 / 15, by rw [h1],
  have h4 : t_clear = 1 / 3, by rw [h2]; norm_cast,
  have h5 : t_foggy = 1, by rw [h3]; norm_cast,
  rw [h4, h5] at h,
  linarith,
end

end morgan_foggy_time_l794_794312


namespace equilateral_triangle_exists_l794_794671

-- Define the problem's conditions and the main theorem statement
noncomputable def triangle_equilateral_exists (A B C : Point) (r : ℝ) : Prop :=
  let D_A := closed_ball A r
  let D_B := closed_ball B r
  let D_C := closed_ball C r
  ∃ (X ∈ D_A) (Y ∈ D_B) (Z ∈ D_C), equilateral_triangle X Y Z

theorem equilateral_triangle_exists {A B C : Point} (AC : ℝ) (hB : angle A B C = 30) :
  ∃ (X ∈ closed_ball A (AC / 3)) (Y ∈ closed_ball B (AC / 3)) (Z ∈ closed_ball C (AC / 3)), 
  equilateral_triangle X Y Z :=
sorry

end equilateral_triangle_exists_l794_794671


namespace final_color_after_2019_l794_794103

def f (n : ℕ) : ℕ :=
if n ≤ 17 then 3 * n - 2 else abs (129 - 2 * n)

theorem final_color_after_2019 (n : ℕ) (initial_color : n = 5) :
  (Function.iterate f 2019 n) = 55 :=
by
  sorry

end final_color_after_2019_l794_794103


namespace polar_bear_daily_salmon_consumption_l794_794903

/-- Polar bear's fish consumption conditions and daily salmon amount calculation -/
theorem polar_bear_daily_salmon_consumption (h1: ℝ) (h2: ℝ) : 
  (h1 = 0.2) → (h2 = 0.6) → (h2 - h1 = 0.4) :=
by
  sorry

end polar_bear_daily_salmon_consumption_l794_794903


namespace total_amount_is_correct_l794_794839

-- Given conditions
def original_price : ℝ := 200
def discount_rate: ℝ := 0.25
def coupon_value: ℝ := 10
def tax_rate: ℝ := 0.05

-- Define the price calculations
def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
def price_after_coupon (p : ℝ) (c : ℝ) : ℝ := p - c
def final_price_with_tax (p : ℝ) (t : ℝ) : ℝ := p * (1 + t)

-- Goal: Prove the final amount the customer pays
theorem total_amount_is_correct : final_price_with_tax (price_after_coupon (discounted_price original_price discount_rate) coupon_value) tax_rate = 147 := by
  sorry

end total_amount_is_correct_l794_794839


namespace function_characterization_l794_794821

theorem function_characterization (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, n^3 - n^2 ≤ f(n) * (f(f(n)))^2 ∧ f(n) * (f(f(n)))^2 ≤ n^3 + n^2) →
  (∀ n : ℕ+, f(n) = n - 1 ∨ f(n) = n ∨ f(n) = n + 1) :=
sorry

end function_characterization_l794_794821


namespace max_f_sin4theta_l794_794235

open Real

noncomputable def f (x : ℝ) : ℝ :=
  let a := ((1 + sin (2 * x)), (sin x - cos x))
  let b := (1, (sin x + cos x))
  a.1 * b.1 + a.2 * b.2

theorem max_f : ∃ x : ℝ, f x = 1 + √2 :=
sorry

theorem sin4theta (θ : ℝ) (h : f θ = 8 / 5) : sin (4 * θ) = 16 / 25 :=
sorry

end max_f_sin4theta_l794_794235


namespace non_neg_int_solutions_m_value_integer_values_of_m_l794_794226

-- 1. Non-negative integer solutions of x + 2y = 3
theorem non_neg_int_solutions (x y : ℕ) :
  x + 2 * y = 3 ↔ (x = 3 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
sorry

-- 2. If (x, y) = (1, 1) satisfies both x + 2y = 3 and x + y = 2, then m = -4
theorem m_value (m : ℝ) :
  (1 + 2 * 1 = 3) ∧ (1 + 1 = 2) ∧ (1 - 2 * 1 + m * 1 = -5) → m = -4 :=
sorry

-- 3. Given n = 3, integer values of m are -2 or 0
theorem integer_values_of_m (m : ℤ) :
  ∃ x y : ℤ, 3 * x + 4 * y = 5 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0 :=
sorry

end non_neg_int_solutions_m_value_integer_values_of_m_l794_794226


namespace trapezoid_lateral_side_length_l794_794753

theorem trapezoid_lateral_side_length (a b : ℝ) (trapezoid : Type)
  [trapezoid : has_basis trapezoid (λ x : trapezoid, x.base1 = a) (λ y : trapezoid, y.base2 = b)]
  (midline_property : ∃ (M : trapezoid) (l : ℝ), is_midline l M ∧ divides_trapezoid_into_inscribable_quadrilaterals trapezoid M) :
  ∃ (lateral : ℝ), lateral = a + b :=
begin
  sorry
end

end trapezoid_lateral_side_length_l794_794753


namespace savings_when_combined_l794_794861

open Function

-- Definitions
def window_price : ℕ := 100

def num_free_windows (purchased_windows : ℕ) : ℕ :=
  2 * (purchased_windows / 8)

def total_cost (needed_windows : ℕ) : ℕ :=
  let effective_windows := needed_windows - num_free_windows (needed_windows)
  window_price * effective_windows

def dave_needed : ℕ := 9
def doug_needed : ℕ := 10

def cost_if_separate (dave needed_windows : ℕ) (doug needed_windows : ℕ) : ℕ :=
  total_cost dave_needed + total_cost doug_needed

def combined_needed : ℕ := dave_needed + doug_needed

def cost_if_combined (combined_needed : ℕ) : ℕ :=
  total_cost combined_needed

-- Theorem statement
theorem savings_when_combined :
  cost_if_separate dave_needed doug_needed cost_if_combined combined_needed 
  = 100 :=
sorry

end savings_when_combined_l794_794861


namespace min_value_l794_794532

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) :
  2 * a + b + c ≥ 4 :=
sorry

end min_value_l794_794532


namespace quadratic_intersection_l794_794345

def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_intersection:
  ∃ a b c : ℝ, 
  quadratic a b c (-3) = 16 ∧ 
  quadratic a b c 0 = -5 ∧ 
  quadratic a b c 3 = -8 ∧ 
  quadratic a b c (-1) = 0 :=
sorry

end quadratic_intersection_l794_794345


namespace even_increasing_implies_absolute_l794_794690

variable {R : Type*} [linear_ordered_field R]
variable {f : R → R}

-- Define the conditions
def even_function (f : R → R) : Prop := ∀ x : R, f(-x) = f(x)
def increasing_on_nonneg (f : R → R) : Prop := ∀ x y : R, 0 ≤ x → 0 ≤ y → x < y → f(x) < f(y)

-- The proof problem
theorem even_increasing_implies_absolute (h_even : even_function f) (h_inc : increasing_on_nonneg f) 
(a b : R) (h : f(a) < f(b)) : abs(a) < abs(b) :=
by
  sorry

end even_increasing_implies_absolute_l794_794690


namespace a_n_formula_b_n_geometric_sequence_l794_794339

noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 1

def S_n (n : ℕ) : ℝ := sorry -- Sum of the first n terms of b_n

def b_n (n : ℕ) : ℝ := 2 - 2 * S_n n

theorem a_n_formula (n : ℕ) : a_n n = 3 * n - 1 :=
by { sorry }

theorem b_n_geometric_sequence : ∀ n ≥ 2, b_n n / b_n (n - 1) = 1 / 3 :=
by { sorry }

end a_n_formula_b_n_geometric_sequence_l794_794339


namespace sum_f_a_n_2018_l794_794967

noncomputable def f (x : ℝ) : ℝ := sorry -- Definition of f(x) is provided by the problem

def a (n : ℕ) : ℝ := 2 * n - 1 -- Arithmetic sequence (a_n = 2n - 1)

lemma f_odd (x : ℝ) : f(x) = -f(-x) := sorry -- f(x) is odd function

lemma f_periodic (x : ℝ) : f(x - 3/2) = f(x) := sorry -- f(x) is periodic with period 3/2

lemma f_value : f(-2) = -3 := sorry -- Given f(-2) = -3

theorem sum_f_a_n_2018 : (∑ n in finset.range 2018, f (a (n + 1))) = -3 := sorry

end sum_f_a_n_2018_l794_794967


namespace maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l794_794987

def f (x : ℝ) (t : ℝ) : ℝ := abs (2 * x - 1) - abs (t * x + 3)

theorem maximum_value_when_t_is_2 :
  ∃ x : ℝ, (f x 2) ≤ 4 ∧ ∀ y : ℝ, (f y 2) ≤ (f x 2) := sorry

theorem solve_for_t_when_maximum_value_is_2 :
  ∃ t : ℝ, t > 0 ∧ (∀ x : ℝ, (f x t) ≤ 2 ∧ (∃ y : ℝ, (f y t) = 2)) → t = 6 := sorry

end maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l794_794987


namespace min_max_product_l794_794686

noncomputable def min_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the minimum value of 3x^2 + 4xy + 3y^2
  sorry

noncomputable def max_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the maximum value of 3x^2 + 4xy + 3y^2
  sorry

theorem min_max_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) :
  min_value x y h * max_value x y h = 7 / 16 :=
sorry

end min_max_product_l794_794686


namespace points_on_circle_l794_794720

theorem points_on_circle (N : ℕ) :
  ∃ (P : fin N → ℝ × ℝ),
    (∀ i j k : fin N, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (P i) (P j) (P k)) ∧ 
    (∀ i j : fin N, i ≠ j → ∃ d : ℤ, dist (P i) (P j) = d) :=
sorry

end points_on_circle_l794_794720


namespace complement_A_in_U_l794_794228

open Set

-- Define the universal set U
def U : Set ℤ := {-2, 0, 1, 2}

-- Define the set A using the given condition x^2 + x - 2 = 0
def A : Set ℤ := {x | x^2 + x - 2 = 0}

-- State the theorem to prove that the complement of A in U is {0, 2}
theorem complement_A_in_U : compl U A = {0, 2} :=
  sorry -- Proof is skipped

end complement_A_in_U_l794_794228


namespace part_i_part_ii_l794_794294

variable {G T : Type}
variable [tree G T]
variable (V : Set T)
variable (S : Set T := V)

noncomputable def separated_by (x y : T) : Prop :=
  ∃ z ∈ x.closure ∩ y.closure, z ∈ V

theorem part_i (T_normal : normal_tree T G)
    (x y : T) : separated_by x y :=
  sorry

theorem part_ii 
    (S_closed : down_closed S) 
    (x : T)
    (x_minimal : minimal x (T \ S)) :
  supports_all_branches (floor x) (G \ S) :=
  sorry

end part_i_part_ii_l794_794294


namespace sum_of_roots_l794_794522

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in ({√3, -√3} : set ℝ), x

theorem sum_of_roots :
  (∀ x : ℝ, (x ≠ 1) ∧ (x ≠ -1) → ( -12 * x) / (x ^ 2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  sum_of_solutions = 0 :=
sorry

end sum_of_roots_l794_794522


namespace no_solution_exists_l794_794785

theorem no_solution_exists :
  ¬ ∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) :=
by sorry

end no_solution_exists_l794_794785


namespace equal_angles_oc_po_and_qob_l794_794994

open EuclideanGeometry

/-- Given the rays OP and OQ, and points M and N such that ∠POM = ∠QON and ∠POM < ∠PON,
    if two circles, one passing through points O, P, N and another passing through O, M, Q
    intersect at points B and C, then prove ∠POC = ∠QOB. -/
theorem equal_angles_oc_po_and_qob
  {P Q M N B C O : Point}
  (hRays : rays O P Q)
  (hAngle1 : angle P O M = angle Q O N)
  (hAngle2 : angle P O M < angle P O N)
  (hCircle1 : circle O P N)
  (hCircle2 : circle O M Q)
  (hIntersect : intersects_circle O P N B C ∧ intersects_circle O M Q B C) :
  angle P O C = angle Q O B :=
sorry

end equal_angles_oc_po_and_qob_l794_794994


namespace knicks_eq_knocks_l794_794604

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l794_794604


namespace groupA_forms_triangle_l794_794768

theorem groupA_forms_triangle (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 20) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  sorry
}

end groupA_forms_triangle_l794_794768


namespace find_polynomial_f_l794_794198
-- Import the Mathlib library to bring in the necessary functionalities.

-- Define the condition that f(x) is an n-degree polynomial where n > 0.
def is_n_degree_polynomial (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ a : ℕ → ℝ, (∀ m, m > n → a m = 0) ∧ f = λ x, ∑ i in finset.range (n + 1), a i * x^i

-- Define the main theorem to be proved.
theorem find_polynomial_f (f : ℝ → ℝ) (n : ℕ) (h1 : n > 0)
  (h2 : ∀ x : ℝ, 8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0) :
  f = (λ x, x^3 - 2) := 
sorry

end find_polynomial_f_l794_794198


namespace total_area_of_field_l794_794420

-- Definitions for the conditions
def fertilizer_per_area := 700 / 5600
def total_fertilizer := 1200

-- Define the theorem statement
theorem total_area_of_field :
  ∃ A : ℕ, A = (5600 * 1200) / 700 ∧ (1200 / A) = fertilizer_per_area :=
by
  use 9600
  split
  { 
    calc 9600 = (5600 * 1200) / 700 : by norm_num
  }
  sorry

end total_area_of_field_l794_794420


namespace min_value_PQ_l794_794651

variable (t : ℝ) (x y : ℝ)

-- Parametric equations of line l
def line_l : Prop := (x = 4 * t - 1) ∧ (y = 3 * t - 3 / 2)

-- Polar equation of circle C
def polar_eq_circle_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4)

-- General equation of line l
def general_eq_line_l (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Rectangular equation of circle C
def rectangular_eq_circle_c (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Definition of the condition where P is on line l
def p_on_line_l (x y : ℝ) : Prop := ∃ t : ℝ, line_l t x y

-- Minimum value of |PQ|
theorem min_value_PQ :
  p_on_line_l x y →
  general_eq_line_l x y →
  rectangular_eq_circle_c x y →
  ∃ d : ℝ, d = Real.sqrt 2 :=
by intros; sorry

end min_value_PQ_l794_794651


namespace sum_of_roots_l794_794520

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in ({√3, -√3} : set ℝ), x

theorem sum_of_roots :
  (∀ x : ℝ, (x ≠ 1) ∧ (x ≠ -1) → ( -12 * x) / (x ^ 2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  sum_of_solutions = 0 :=
sorry

end sum_of_roots_l794_794520


namespace largest_base_5_five_digits_base_10_value_l794_794038

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794038


namespace arccos_zero_eq_pi_div_two_l794_794468

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l794_794468


namespace sum_of_squares_and_cube_unique_l794_794266

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem sum_of_squares_and_cube_unique : 
  ∃! (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_cube c ∧ a + b + c = 100 :=
sorry

end sum_of_squares_and_cube_unique_l794_794266


namespace range_of_m_l794_794614

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end range_of_m_l794_794614


namespace balanced_palindromic_sum_is_palindrome_l794_794092

def is_palindrome (n : ℕ) : Prop := 
  let a := n / 1000 in 
  let b := (n % 1000) / 100 in
  let c := (n % 100) / 10 in
  let d := n % 10 in
  a = d ∧ b = c

def is_balanced (n : ℕ) : Prop := 
  let a := n / 1000 in 
  let b := (n % 1000) / 100 in
  let c := (n % 100) / 10 in
  let d := n % 10 in
  a + b = c + d

theorem balanced_palindromic_sum_is_palindrome (n : ℕ) (h : n < 10000 ∧ n ≥ 1000) 
  (h1 : is_balanced n) 
  (h2 : ∃ p q : ℕ, is_palindrome p ∧ is_palindrome q ∧ n = p + q) :
  is_palindrome n :=
sorry

end balanced_palindromic_sum_is_palindrome_l794_794092


namespace possible_α_values_l794_794296

noncomputable def α_values : Set ℂ :=
  {α : ℂ | α ≠ 1 ∧ (|α^2 - 1| = 3 * |α - 1|) ∧ (|α^3 - 1| = 5 * |α - 1|)}

theorem possible_α_values (α : ℂ) (h1 : α ∈ α_values) :
  α = Complex.I * Complex.sqrt 8 ∨ α = -Complex.I * Complex.sqrt 8 :=
by
  sorry

end possible_α_values_l794_794296


namespace intersection_of_line_and_hyperbola_l794_794966

theorem intersection_of_line_and_hyperbola (k : ℝ) :
  (∃ (x y : ℝ), y = k*x + 2 ∧ x^2 - y^2 = 6 ∧ ∀ x y1 y2, y1 ≠ y2) →
  - (Real.sqrt 15) / 3 < k ∧ k < -1 :=
sorry

end intersection_of_line_and_hyperbola_l794_794966


namespace computer_sale_price_percent_l794_794837

theorem computer_sale_price_percent (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) :
  original_price = 500 ∧ discount1 = 0.25 ∧ discount2 = 0.10 ∧ discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price * 100 = 64.13 :=
by
  intro h
  sorry

end computer_sale_price_percent_l794_794837


namespace oak_trees_remaining_l794_794795

theorem oak_trees_remaining (initial_trees cut_down_trees remaining_trees : ℕ)
  (h1 : initial_trees = 9)
  (h2 : cut_down_trees = 2)
  (h3 : remaining_trees = initial_trees - cut_down_trees) :
  remaining_trees = 7 :=
by 
  sorry

end oak_trees_remaining_l794_794795


namespace wrapping_third_roll_l794_794728

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l794_794728


namespace total_soda_bottles_l794_794422

def regular_soda : ℕ := 57
def diet_soda : ℕ := 26
def lite_soda : ℕ := 27

theorem total_soda_bottles : regular_soda + diet_soda + lite_soda = 110 := by
  sorry

end total_soda_bottles_l794_794422


namespace candy_bar_calories_l794_794128

theorem candy_bar_calories
  (miles_walked : ℕ)
  (calories_per_mile : ℕ)
  (net_calorie_deficit : ℕ)
  (total_calories_burned : ℕ)
  (candy_bar_calories : ℕ)
  (h1 : miles_walked = 3)
  (h2 : calories_per_mile = 150)
  (h3 : net_calorie_deficit = 250)
  (h4 : total_calories_burned = miles_walked * calories_per_mile)
  (h5 : candy_bar_calories = total_calories_burned - net_calorie_deficit) :
  candy_bar_calories = 200 := 
by
  sorry

end candy_bar_calories_l794_794128


namespace units_digit_of_27_pow_45_l794_794047

theorem units_digit_of_27_pow_45 : (27 ^ 45) % 10 = 7 :=
by {
  -- Conditions for the units digit cycle of 7^n:
  have h_cycle : ∀ n, (n % 4 = 0 → 7 ^ n % 10 = 1) ∧
                    (n % 4 = 1 → 7 ^ n % 10 = 7) ∧
                    (n % 4 = 2 → 7 ^ n % 10 = 9) ∧
                    (n % 4 = 3 → 7 ^ n % 10 = 3),
  {
    intro n,
    split; 
    -- The cycle of units digits for 7^n
    intro h,
    { calc
      (7 ^ (4 * (n / 4 + 1))) % 10 = (7 ^ 4) % 10 : by rw [nat.mul_div_cancel_left _ (dec_trivial : 4 > 0)]
      ... = 1 : by norm_num,
    },
    { split; 
      { calc
      (7 ^ (4 * (n / 4) + 1)) % 10 = (7 ^ (4 * (n / 4))) % 10 * 7 % 10 : by norm_num
      ... = 7 : by norm_num,
      },
      { split; 
        { calc
        (7 ^ (4 * (n / 4) + 2)) % 10 = (7 ^ (4 * (n / 4))) % 10 * 49 % 10 : by norm_num
        ... = 9 : by norm_num,
        },
        { calc
        (7 ^ (4 * (n / 4) + 3)) % 10 = (7 ^ (4 * (n / 4))) % 10 * 343 % 10 : by norm_num
        ... = 3 : by norm_num,
        },
      },
    },
  },

  -- Given n = 45
  have h_45 : 45 % 4 = 1 := by norm_num,

  -- Applying the cycle property based on the remainder
  exact h_cycle 45 (or.inr (or.inl h_45)),
}

end units_digit_of_27_pow_45_l794_794047


namespace smallest_integer_value_l794_794397

theorem smallest_integer_value (x : ℤ) (h : 7 - 3 * x < 22) : x ≥ -4 := 
sorry

end smallest_integer_value_l794_794397


namespace determine_omega_value_l794_794578

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + (π / 6))

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x - (ω * π / 6) + (π / 6))

theorem determine_omega_value (ω : ℝ) : 
  (ω > 0) →
  (∀ x, g ω (x + π / 6) = f ω x) →
  (∀ x, g ω (-π / 6 - x) = g ω (-π / 6 + x)) →
  (∀ x, (π / 6) ≤ x ∧ x ≤ (π / 3) → monotone_decreasing (f ω x)) →
  ω = 2 :=
by
  sorry

end determine_omega_value_l794_794578


namespace min_value_of_squares_attains_min_value_l794_794681

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  (a^2 + b^2 + c^2) ≥ (t^2 / 3) :=
sorry

theorem attains_min_value (a b c t : ℝ) (h : a = t / 3 ∧ b = t / 3 ∧ c = t / 3) :
  (a^2 + b^2 + c^2) = (t^2 / 3) :=
sorry

end min_value_of_squares_attains_min_value_l794_794681


namespace arithmetic_sequence_a1_value_l794_794545

-- Define the conditions and state the proof goal
theorem arithmetic_sequence_a1_value 
  (a : ℕ → ℤ) -- sequence definition
  (S : ℕ → ℤ) -- sum definition
  (h_arith : ∀ n, a(n + 1) = a(n) + (a 2 - a 1)) -- arithmetic sequence condition
  (h_S : ∀ n, S(n) = n * (a 1 + a n) / 2) -- sum of the first n terms
  (h_a13 : a 13 = 13) -- given condition a13 = 13
  (h_S13 : S 13 = 13) -- given condition S13 = 13
: a 1 = -11 :=
sorry

end arithmetic_sequence_a1_value_l794_794545


namespace dog_run_distance_approx_l794_794841
noncomputable def dog_run_distance := π * 10

theorem dog_run_distance_approx : dog_run_distance ≈ 31.42 := 
by
  sorry

end dog_run_distance_approx_l794_794841


namespace find_point_A_coordinates_min_length_segment_AB_l794_794964

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus_l : ℝ × ℝ := (1, 0)
def point_A (x1 y1 : ℝ) (F : ℝ × ℝ) (d : ℝ) : Prop := abs (x1 - F.1) = d / 2

-- The first part: Prove coordinates of point A
theorem find_point_A_coordinates (d : ℝ) (h1 : d = 4) : (∃ (x1 y1 : ℝ), 
  parabola x1 y1 ∧ point_A x1 y1 focus_l d ∧ (x1 = 3 ∧ (y1 = 2 * real.sqrt 3 ∨ y1 = -2 * real.sqrt 3))) :=
sorry

-- The second part: Prove the minimum length of segment AB
theorem min_length_segment_AB : 
  ∃ (l : ℝ × ℝ → Prop), (∀ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ parabola x2 y2 ∧ l (x1, y1) ∧ 
  l (x2, y2) ∧ (x1 = 1) ∧ (y1 = 2) ∧ (x2 = 1) ∧ (y2 = -2)) → (4 ≤ abs (4) + abs(4)) :=
sorry

end find_point_A_coordinates_min_length_segment_AB_l794_794964


namespace minimum_value_of_a_b_l794_794932

theorem minimum_value_of_a_b 
  (a b : ℝ) 
  (h : log a b = -2) : 
  a + b = 3 * real.cbrt 2 / 2 := 
sorry

end minimum_value_of_a_b_l794_794932


namespace monotonic_intervals_of_f_range_of_b_l794_794975

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  Real.log x + 0.5 * a * x ^ 2 + b * x

theorem monotonic_intervals_of_f :
  ∃ x : Set ℝ, ∃ y : Set ℝ, 
  (∀ (a b : ℝ), a = -2 → b = 1 → ∀ ⦃x : ℝ⦄, x > 0 → 
    ((∀ x ∈ Set.Ioc 0 1, deriv (λ x, f x a b) x > 0) ∧ 
     (∀ x ∈ Set.Ioi 1, deriv (λ x, f x a b) x < 0))
  ∧ x = Set.Ioc 0 1 ∧ y = Set.Ioi 1) := sorry

theorem range_of_b (a b : ℝ) : 
  (∀ a ∈ Set.Ici (1 : ℝ), ∀ ⦃x : ℝ⦄, x ∈ Set.Ici (1 : ℝ) → (f x a b ≥ 0)) → b ≥ -1/2 := sorry

end monotonic_intervals_of_f_range_of_b_l794_794975


namespace asian_population_in_west_percentage_l794_794871

def total_population (data : List (List ℕ)) (ethnic_group : ℕ) : ℕ :=
  data.map (fun row => row.nthD ethnic_group 0).sum

def percentage_of_population (part : ℕ) (total : ℕ) : ℚ :=
  (part.toRat / total.toRat) * 100

theorem asian_population_in_west_percentage :
  let population_data := [[48, 55, 62, 31], [6, 4, 18, 1], [2, 3, 4, 8], [1, 2, 3, 5]]
  let asian_group := 2
  let west_region := 3
  let total_asian_population := total_population population_data asian_group
  let asian_population_west := population_data.nthD 2 [].nthD west_region 0
  percentage_of_population asian_population_west total_asian_population ≈ 47 := sorry

end asian_population_in_west_percentage_l794_794871


namespace proof_problem_l794_794680

-- Define real numbers a, b, c
variables (a b c : ℝ)

-- Assume the condition a + b + c = 3
def condition_holds : Prop := a + b + c = 3

-- Define the set S of possible values of ab + ac + bc
def possible_values : Set ℝ := {x | ∃ a b c : ℝ, (a + b + c = 3) ∧ (x = ab + ac + bc)}

-- Statement of the theorem to be proved
theorem proof_problem : possible_values = Icc (-9 / 2) 3 :=
sorry

end proof_problem_l794_794680


namespace bisection_method_step_two_l794_794659

noncomputable def f (x : ℝ) : ℝ := x^5 + 8 * x^3 - 1

theorem bisection_method_step_two :
  f 0 < 0 ∧ f 0.5 > 0 →
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5) ∧ f 0.25 = f 0.25 :=
by
  intro h,
  -- the proof would proceed from here
  sorry

end bisection_method_step_two_l794_794659


namespace snack_machine_total_cost_l794_794782

theorem snack_machine_total_cost 
  (candy_bar_price ∈ {1.50, 2, 2.50} : Real)
  (chip_price1 ∈ {0.75, 1, 1.25} : Real)
  (chip_price2 ∈ {0.75, 1, 1.25} : Real)
  (chip_price1 ≠ chip_price2)
  (cookie_price: Real := 1.2) :
  5 * (candy_bar_price + chip_price1 + chip_price2 + 2 * cookie_price) = 30.75 := 
  by
    have h_candy_bar: candy_bar_price = 2 := by sorry
    have h_chip1: chip_price1 = 0.75 := by sorry
    have h_chip2: chip_price2 = 1 := by sorry
    calc 5 * (candy_bar_price + chip_price1 + chip_price2 + 2 * cookie_price) 
        = 5 * (2 + 0.75 + 1 + 2 * 1.2) : by rw [h_candy_bar, h_chip1, h_chip2]
    ... = 5 * (2 + 0.75 + 1 + 2.4) : by rfl
    ... = 5 * 6.15 : by norm_num
    ... = 30.75 : by norm_num


end snack_machine_total_cost_l794_794782


namespace find_projection_vector_l794_794052

variable (a c d : ℝ)

def projection_constant (v w : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let norm_w := w.1^2 + w.2^2
  let proj_v_w := (v.1 * w.1 + v.2 * w.2) / norm_w * w in
  proj_v_w = p

theorem find_projection_vector (c d : ℝ) (h : c = -3/2 * d) :
  ∀ a : ℝ, projection_constant (a, 3/2 * a + 3) (c, d) (-18/13, 12/13) :=
by
  intro a
  rw [projection_constant, h]
  sorry

end find_projection_vector_l794_794052


namespace gifts_wrapped_with_third_roll_l794_794731

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l794_794731


namespace equal_amounts_hot_and_cold_water_l794_794084

theorem equal_amounts_hot_and_cold_water (time_to_fill_cold : ℕ) (time_to_fill_hot : ℕ) (t_c : ℤ) : 
  time_to_fill_cold = 19 → 
  time_to_fill_hot = 23 → 
  t_c = 2 :=
by
  intros h_c h_h
  sorry

end equal_amounts_hot_and_cold_water_l794_794084


namespace find_p_q_l794_794699

theorem find_p_q (p q : ℚ)
  (h1 : (4 : ℚ) * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2):
  (p, q) = (-29/12 : ℚ, 43/12 : ℚ) :=
by 
  sorry

end find_p_q_l794_794699


namespace exists_constant_a_l794_794958

theorem exists_constant_a (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : (m : ℝ) / n < Real.sqrt 7) :
  ∃ (a : ℝ), a > 1 ∧ (7 - (m^2 : ℝ) / (n^2 : ℝ) ≥ a / (n^2 : ℝ)) ∧ a = 3 :=
by
  sorry

end exists_constant_a_l794_794958


namespace interval_of_a_l794_794409

theorem interval_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_monotone : ∀ x y, x < y → f y ≤ f x)
  (h_condition : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) : 
  a ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 5 :=
by
  sorry

end interval_of_a_l794_794409


namespace polynomial_exponents_4_l794_794150

theorem polynomial_exponents_4 (P : ℤ[X]) (h : ∀ x y : ℤ, 0 < x → 0 < y → (P.eval x - P.eval y) % (x^2 + y^2) = 0) :
  ∀ n : ℕ, P.coeff n = 0 → n % 4 = 0 := 
sorry

end polynomial_exponents_4_l794_794150


namespace max_leap_years_in_400_years_period_l794_794114

theorem max_leap_years_in_400_years_period :
  ∀ (years : ℕ) (is_leap_year : ℕ → Prop), 
  (∀ y, is_leap_year y ↔ (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))) →
  ∑ k in finset.range 400, if is_leap_year k then 1 else 0 = 89 :=
by sorry

end max_leap_years_in_400_years_period_l794_794114


namespace knicks_eq_knocks_l794_794606

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end knicks_eq_knocks_l794_794606


namespace largest_base5_number_to_base10_is_3124_l794_794004

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794004


namespace arccos_zero_eq_pi_div_two_l794_794466

-- Let's define a proof problem to show that arccos 0 equals π/2.
theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l794_794466


namespace inequality_1_inequality_2_inequality_3_inequality_4_l794_794700

noncomputable def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = Real.pi

theorem inequality_1 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin a + Real.sin b + Real.sin c ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_2 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos (a / 2) + Real.cos (b / 2) + Real.cos (c / 2) ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_3 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos a * Real.cos b * Real.cos c ≤ (1 / 8) :=
sorry

theorem inequality_4 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin (2 * a) + Real.sin (2 * b) + Real.sin (2 * c) ≤ Real.sin a + Real.sin b + Real.sin c :=
sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l794_794700


namespace brendan_weekly_capacity_l794_794451

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end brendan_weekly_capacity_l794_794451


namespace equal_differences_l794_794937

theorem equal_differences 
  (n : ℕ) (hn : n ≥ 3) 
  (a : Fin 2n → ℕ) 
  (h1 : 1 ≤ a 0) 
  (h2 : ∀ i j : Fin 2n, i < j → a i < a j)
  (h3 : ∀ i : Fin 2n, a i ≤ n * n) : 
  ∃ i1 i2 i3 i4 i5 i6 : Fin 2n, 
    i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧ 
    (a i2 - a i1 = a i4 - a i3) ∧ (a i4 - a i3 = a i6 - a i5) :=
by
  sorry

end equal_differences_l794_794937


namespace range_of_k_for_circle_intersection_l794_794268

noncomputable def circleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 12 = 0

def lineEquation (x y k : ℝ) : Prop :=
  y = k * x - 2

def distance (x1 y1 k : ℝ) : ℝ :=
  |4 * k - 2| / Real.sqrt (k^2 + 1)

theorem range_of_k_for_circle_intersection :
  ∀ k : ℝ,
  (∀ x y : ℝ, circleEquation x y → lineEquation x y k → distance 4 0 k ≤ 2) ↔ (0 ≤ k ∧ k ≤ 4 / 3) :=
by
  sorry

end range_of_k_for_circle_intersection_l794_794268


namespace range_of_k_to_intersect_ellipse_l794_794965

theorem range_of_k_to_intersect_ellipse (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ 2 * x ^ 2 + 3 * y ^ 2 = 6) ↔
  k ∈ set.Ioi (real.sqrt (2 / 3)) ∪ set.Iio (- real.sqrt (2 / 3)) :=
by
  sorry

end range_of_k_to_intersect_ellipse_l794_794965


namespace sum_of_solutions_of_equation_l794_794044

theorem sum_of_solutions_of_equation :
  let solutions := {x | (9 * x) / 27 = 6 / x},
  x ∈ solutions → x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 →
  (3 * Real.sqrt 2 + -3 * Real.sqrt 2) = 0 :=
by
  sorry

end sum_of_solutions_of_equation_l794_794044


namespace xyz_expression_l794_794305

theorem xyz_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
    (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
    (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)) = -3 / (2 * (x^2 + y^2 + xy)) :=
by sorry

end xyz_expression_l794_794305


namespace cyclist_return_speed_l794_794416

theorem cyclist_return_speed : ∀ (d : ℝ) (avg_speed dist : ℝ),
  d > 0 →
  avg_speed > 0 →
  dist = 2 * d →
  avg_speed = 25 →
  (2 * d) / (d / 30 + d / r) = avg_speed →
  r = 750 / 35 :=
begin
  intros d avg_speed dist h_d_pos h_avg_speed_pos h_dist h_avg_speed h_eq,
  have h1 := h_eq,
  rw h_dist at h1,
  rw h_avg_speed at h1,
  rw div_eq_iff_mul_eq at h1,
  { linarith },
  { linarith [(d + d / r)] }
end

end cyclist_return_speed_l794_794416


namespace circle_radius_l794_794646

-- Defining the geometrical setup and conditions
variables {O A B C D : Type} 
variables (r : ℝ) (AB : AB = 6) (BD : BD = 3) 
variables (angle_C : ∠AOC = 2 * π / 3)

-- Statement of the problem
theorem circle_radius (r : ℝ) 
  (A_on_circle : ∃ (A : O), distance O A = r)
  (AB_tangent : ∀ (A B : O), A ≠ B → AB = 6)
  (C_position : ∃ (C : O), distance O C = r ∧ ∠AOC = 2 * π / 3)
  (BD_intersect : ∃ (D : O), distance B D = 3 ∧ ∃ C : O, ∠BDC = 60) :
  r = sqrt (42 - 6 * sqrt 13) :=
sorry

end circle_radius_l794_794646


namespace largest_base5_to_base10_l794_794020

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794020


namespace exists_infinitely_many_good_numbers_l794_794802

-- Define what it means for a sequence of n positive integers to be "good"
def isGood (a : List ℕ) : Prop :=
  let sum := a.sum
  sum = a.pairs.Sum (λ (x : ℕ × ℕ), x.fst.gcd x.snd)

-- Define the main theorem
theorem exists_infinitely_many_good_numbers :
  ∃ᶠ n in at_top, ∃ (a : List ℕ), a.length = n ∧ isGood a := 
sorry

end exists_infinitely_many_good_numbers_l794_794802


namespace parallel_AC_MN_l794_794258

/-- In a convex quadrilateral ABCD with perpendicular diagonals AC and BD, and points M on AD and 
N on CD such that ∠ABN and ∠CBM are right angles, lines AC and MN are parallel. -/
theorem parallel_AC_MN
  {A B C D M N : Type*}
  [add_comm_group A] [module ℝ A] [add_comm_group B] [module ℝ B]
  [add_comm_group D] [module ℝ D] [add_comm_group M] [module ℝ M]
  [add_comm_group N] [module ℝ N]
  (h_convex : convex ℝ (convex_hull ℝ ({A, B, C, D} : set ℝ)))
  (h_perpendicular : ∀ {X Y Z W : A}, X = A → Y = B → Z = C → W = D → ⟪C - A, D - B⟫ = 0)
  (h_M : ∀ {X Y : A}, X = A → Y = D → ∃ M, M ∈ line[X, Y] ∧ ∠ (AB) (MN) = π / 2)
  (h_N : ∀ {X Y : A}, X = C → Y = D → ∃ N, N ∈ line[X, Y] ∧ ∠ (CB) (MN) = π / 2) :
  parallel (line[A, C]) (line[M, N]) :=
sorry

end parallel_AC_MN_l794_794258


namespace convex_hull_not_triangle_for_some_2n_subset_l794_794820

-- Conditions
variables {n : ℕ} (points : set (ℝ × ℝ))
hypothesis h_count : points.card = 3 * n - 1
hypothesis h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → 
  ¬Collinear p1 p2 p3

-- Define convex_hull
def convex_hull (s : set (ℝ × ℝ)) :set (ℝ × ℝ) := sorry

-- Define Collinear
def Collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Statement to be proved
theorem convex_hull_not_triangle_for_some_2n_subset :
  ∃ (subset : set (ℝ × ℝ)), subset.card = 2 * n ∧ convex_hull subset ≠ set.triangle := 
begin
  sorry
end

end convex_hull_not_triangle_for_some_2n_subset_l794_794820


namespace log2_128_minus_log2_8_eq_4_l794_794124

-- Definitions based on given problem
def log2 (x : ℝ) : ℝ := real.log x / real.log 2

-- The theorem we need to prove
theorem log2_128_minus_log2_8_eq_4 :
  log2 128 - log2 8 = 4 :=
sorry

end log2_128_minus_log2_8_eq_4_l794_794124


namespace stormi_additional_money_needed_l794_794332

noncomputable def earnings_from_jobs : ℝ :=
  let washing_cars := 5 * 8.50
  let walking_dogs := 4 * 6.75
  let mowing_lawns := 3 * 12.25
  let gardening := 2 * 7.40
  washing_cars + walking_dogs + mowing_lawns + gardening

noncomputable def discounted_prices : ℝ :=
  let bicycle := 150.25 * (1 - 0.15)
  let helmet := 35.75 - 5.00
  let lock := 24.50
  bicycle + helmet + lock

noncomputable def total_cost_after_tax : ℝ :=
  let cost_before_tax := discounted_prices
  cost_before_tax * 1.05

noncomputable def amount_needed : ℝ :=
  total_cost_after_tax - earnings_from_jobs

theorem stormi_additional_money_needed : amount_needed = 71.06 := by
  sorry

end stormi_additional_money_needed_l794_794332


namespace argument_sum_equals_l794_794452

noncomputable def z1 : ℂ := complex.exp (complex.I * 8 * real.pi / 40)
noncomputable def z2 : ℂ := complex.exp (complex.I * 13 * real.pi / 40)
noncomputable def z3 : ℂ := complex.exp (complex.I * 18 * real.pi / 40)
noncomputable def z4 : ℂ := complex.exp (complex.I * 23 * real.pi / 40)
noncomputable def z5 : ℂ := complex.exp (complex.I * 28 * real.pi / 40)
noncomputable def z6 : ℂ := complex.exp (complex.I * 33 * real.pi / 40)

theorem argument_sum_equals (θ : ℝ) :
  θ = real.pi * 41 / 80 ↔ complex.arg (z1 + z2 + z3 + z4 + z5 + z6) = real.pi * 41 / 80 :=
by sorry

end argument_sum_equals_l794_794452


namespace jude_buys_correct_number_of_vehicles_l794_794284

theorem jude_buys_correct_number_of_vehicles :
  let bottle_caps_per_car := 5 in
  let bottle_caps_per_truck := 6 in
  let initial_bottle_caps := 100 in
  let trucks_bought := 10 in
  let trucks_cost := trucks_bought * bottle_caps_per_truck in
  let remaining_bottle_caps := initial_bottle_caps - trucks_cost in
  let spent_on_cars := remaining_bottle_caps * 0.75 in
  let cars_bought := spent_on_cars / bottle_caps_per_car in
  trucks_bought + cars_bought = 16 :=
by
  -- Definitions
  let bottle_caps_per_car := 5
  let bottle_caps_per_truck := 6
  let initial_bottle_caps := 100
  let trucks_bought := 10
  let trucks_cost := trucks_bought * bottle_caps_per_truck
  let remaining_bottle_caps := initial_bottle_caps - trucks_cost
  let spent_on_cars := remaining_bottle_caps * 0.75
  let cars_bought := spent_on_cars / bottle_caps_per_car
  -- Final assertion
  have : trucks_bought + cars_bought = (10 : ℕ) + 6, from sorry,
  exact this

end jude_buys_correct_number_of_vehicles_l794_794284


namespace even_three_digit_numbers_count_l794_794387

theorem even_three_digit_numbers_count : 
  { n : ℕ // n < 700 ∧ n ≥ 100 ∧ even n ∧ ∀ d ∈ digits n, digits n ∈ {1, 2, 3, 4, 5, 6} ∧ no_repeat (digits n) } = 81 :=
by
  sorry

def digits (n : ℕ) : list ℕ := sorry

def no_repeat (l : list ℕ) : Prop := sorry

end even_three_digit_numbers_count_l794_794387


namespace solve_equation_l794_794973

-- Define the equation as a function of y
def equation (y : ℝ) : ℝ :=
  y^4 - 20 * y + 1

-- State the theorem that y = -1 satisfies the equation.
theorem solve_equation : equation (-1) = 22 := 
  sorry

end solve_equation_l794_794973


namespace sum_of_squares_prime_4n3_l794_794818

theorem sum_of_squares_prime_4n3 (a x y z : ℤ) (ha : prime a) (h_form : ∃ n : ℤ, a = 4 * n + 3) (h_eq : x^2 + y^2 = a * z^2) : x = 0 ∧ y = 0 :=
sorry

end sum_of_squares_prime_4n3_l794_794818


namespace find_k_l794_794138

noncomputable def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := a n + (a n) ^ 2 / 2023

theorem find_k : ∃ k : ℕ, a k < 1 ∧ 1 < a (k + 1) := by
  sorry

end find_k_l794_794138


namespace largest_base_5_five_digit_number_in_decimal_l794_794033

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794033


namespace bug_at_A_after_seven_steps_l794_794542

-- Definitions of the problem conditions
def tetrahedron_edge_length := 1
def initial_vertex := A
def steps := 7

-- Recursive function to calculate the probability a_n
def a_n : ℕ → ℚ
| 0       := 1
| (n + 1) := (1 - a_n n) / 3

-- Statement to prove the correct answer
theorem bug_at_A_after_seven_steps : a_n 7 = 182 / 729 := 
sorry

end bug_at_A_after_seven_steps_l794_794542


namespace sequence_term_n_l794_794645

theorem sequence_term_n (a : ℕ → ℕ) (a1 d : ℕ) (n : ℕ) (h1 : a 1 = a1) (h2 : d = 2)
  (h3 : a n = 19) (h_seq : ∀ n, a n = a1 + (n - 1) * d) : n = 10 :=
by
  sorry

end sequence_term_n_l794_794645


namespace perpendicular_vectors_l794_794174

theorem perpendicular_vectors (x : ℝ) :
  let AB := (3 : ℝ, x)
  let CD := (-2 : ℝ, 6 : ℝ)
  AB.1 * CD.1 + AB.2 * CD.2 = 0 → x = 1 :=
by
  let AB := (3 : ℝ, x)
  let CD := (-2 : ℝ, 6 : ℝ)
  have h : AB.1 * CD.1 + AB.2 * CD.2 = 0 := sorry
  show x = 1 from by
    sorry

end perpendicular_vectors_l794_794174


namespace locus_of_points_l794_794539

variables {k A : Type} [ordered_ring k]
variables (A : k) (t : k → k) (c : k) (M : k → k) (d : k → k → k)

-- Assuming distance function and line properties as follows
def on_line (x : k) (l : k → k) : Prop := l x = 0
def is_distance (d : k → k → k) (x y : k) : Prop := d x y ≥ 0 -- Abstract distance measure

theorem locus_of_points (hA : on_line A t) (hc : ∃ c ≠ 0, ∀ M, (M ≠ A) → (M ≠ t (d M A)) → (M ≠ t (d A M)) → (MA ^ 2) / (d M t A) = c) :
  ∃ center₁ radius₁ center₂ radius₂, (∀ M, ((M - center₁) ^ 2 = radius₁ ^ 2) ∨ ((M - center₂) ^ 2 = radius₂ ^ 2)) := sorry

end locus_of_points_l794_794539


namespace sum_of_n_in_interval_l794_794881

theorem sum_of_n_in_interval : 
  ∑ n in Finset.filter (λ n, (2*n + 3 ∣ 2^(Nat.factorial n) - 1) = false) 
    (Finset.Icc 50 100), n = 222 :=
by
  sorry

end sum_of_n_in_interval_l794_794881


namespace january_10_is_friday_l794_794491

def day_of_week : Type := Nat

def wednesday : day_of_week := 3 -- Assuming 0 = Sunday, 1 = Monday, ..., 6 = Saturday
def friday : day_of_week := 5

theorem january_10_is_friday (days_in_week : ℕ) : day_of_week :=
  let dec_25 := wednesday
  let jan_8 := (dec_25 + 7) % days_in_week -- January 1 is also a Wednesday.
  let jan_10 := (jan_8 + 2) % days_in_week -- 2 days after January 8
  jan_10 = friday := 
sorry

end january_10_is_friday_l794_794491


namespace solution_l794_794246

theorem solution (a b : ℝ) (h1 : a^2 + 2 * a - 2016 = 0) (h2 : b^2 + 2 * b - 2016 = 0) :
  a^2 + 3 * a + b = 2014 := 
sorry

end solution_l794_794246


namespace bridge_length_is_correct_l794_794439

noncomputable def speed_km_per_hour_to_m_per_s (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed_m_per_s time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

def bridge_length (total_distance train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem bridge_length_is_correct : 
  let train_length := 110 
  let speed_kmph := 72
  let time_s := 12.099
  let speed_m_per_s := speed_km_per_hour_to_m_per_s speed_kmph
  let total_distance := total_distance_covered speed_m_per_s time_s
  bridge_length total_distance train_length = 131.98 := 
by
  sorry

end bridge_length_is_correct_l794_794439


namespace tan_terminal_angle_l794_794954

theorem tan_terminal_angle (θ : ℝ) (hθ : P(sin θ, cos θ) = P(sin (-π/3), cos (-π/3))) :
  tan θ = - √3 / 3 :=
sorry

end tan_terminal_angle_l794_794954


namespace largest_base5_number_to_base10_is_3124_l794_794000

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794000


namespace fraction_sum_l794_794826

theorem fraction_sum :
  (1 / 3 : ℚ) + (1 / 2 : ℚ) - (5 / 6 : ℚ) + (1 / 5 : ℚ) + (1 / 4 : ℚ) - (9 / 20 : ℚ) - (2 / 15 : ℚ) = -2 / 15 :=
by {
  sorry
}

end fraction_sum_l794_794826


namespace percentage_correct_l794_794798

-- Define the amounts paid to employees A and B.
def A_amount : ℝ := 550 - 249.99999999999997
def B_amount : ℝ := 249.99999999999997

-- Define the equation for percentage of A's amount compared to B's amount.
def percentage_A_to_B : ℝ := (A_amount / B_amount) * 100

-- The main theorem stating the proof problem.
theorem percentage_correct :
  A_amount + B_amount = 550 ∧ B_amount = 249.99999999999997 →
  percentage_A_to_B = 120 :=
by
  -- Conditions
  intro h
  sorry

end percentage_correct_l794_794798


namespace product_gcd_lcm_l794_794164

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l794_794164


namespace convert_base_10_to_base_7_l794_794890

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 3500) : 
  ∃ k : ℕ, k = 13130 ∧ n = 1 * 7^4 + 3 * 7^3 + 1 * 7^2 + 3 * 7^1 + 0 * 7^0 :=
by
  use 13130
  split
  { refl }
  { rw h
    norm_num }
  sorry

end convert_base_10_to_base_7_l794_794890


namespace bruce_money_left_l794_794121

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l794_794121


namespace difference_between_D_and_C_diff_l794_794104

/-- Let the common multiple of their shares be x -/
variable (x : ℝ)

/-- Let the shares be in the proportion 2 : 3 : 4 : 6 and B's share is $1050 -/
variable (A_share B_share C_share D_share : ℝ)

noncomputable def shares : Prop :=
  let A := 2 * x
  let B := 3 * x
  let C := 4 * x
  let D := 6 * x
  B = 1050 ∧
  A_share = A ∧
  B_share = B ∧
  C_share = C ∧
  D_share = D ∧
  D - C = 700

/-- Prove that the difference between D's and C's share is 700 dollars -/
theorem difference_between_D_and_C_diff :
  shares x A_share B_share C_share D_share :=
by
  sorry

end difference_between_D_and_C_diff_l794_794104


namespace new_shipment_cars_l794_794835

theorem new_shipment_cars (original_cars : ℕ) (silver_percent : ℝ) (new_shipment_percent_not_silver : ℝ)
                          (total_silver_percent : ℝ) (new_shipment : ℕ) :
  original_cars = 40 →
  silver_percent = 0.15 →
  new_shipment_percent_not_silver = 0.30 →
  total_silver_percent = 0.25 →
  let silver_cars := silver_percent * original_cars in
  let new_shipment_percent_silver := 1 - new_shipment_percent_not_silver in
  let total_cars := original_cars + new_shipment in
  let new_shipment_silver := new_shipment_percent_silver * real.of_nat new_shipment in
  let total_silver_cars := silver_cars + new_shipment_silver in
  total_silver_cars / total_cars = total_silver_percent →
  new_shipment = 9 :=
begin
  sorry
end

end new_shipment_cars_l794_794835


namespace problem_statement_l794_794304

def a (k : ℕ) : ℚ := 2 ^ k / (3 ^ (2 ^ k) + 1)

def A : ℚ := ∑ i in Finset.range 10, a i

def B : ℚ := ∏ i in Finset.range 10, a i

theorem problem_statement : (A / B) = (3 ^ (2 ^ 10) - 2 ^ 11 - 1) / (2 ^ 47) :=
by
  sorry

end problem_statement_l794_794304


namespace median_min_value_l794_794950

theorem median_min_value (l : List ℕ) (h_len : l.length = 10)
  (h_subset : {3, 5, 9, 1, 7} ⊆ l.toFinset) :
  ∃ l' : List ℕ, l'.length = 10 ∧
    (∀ n ∈ l', n ≥ 0) ∧
    {3, 5, 9, 1, 7} ⊆ l'.toFinset ∧
    (median (l'.sort) = 2.5) :=
begin
  -- proof to be filled in
  sorry
end

end median_min_value_l794_794950


namespace largest_base_5_five_digits_base_10_value_l794_794039

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794039


namespace rectangle_diagonal_length_l794_794792

theorem rectangle_diagonal_length (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x y : ℝ, (π * x^2 * y = a) ∧ (π * y^2 * x = b) ∧ (real.sqrt ((x^2 + y^2) = real.sqrt (a^2 + b^2) / (ab) * real.sqrt (ab / π^2)^2)) :=
sorry

end rectangle_diagonal_length_l794_794792


namespace smallest_number_of_rectangles_l794_794807

-- Defining the given problem conditions
def rectangle_area : ℕ := 3 * 4
def smallest_square_side_length : ℕ := 12

-- Lean 4 statement to prove the problem
theorem smallest_number_of_rectangles 
    (h : ∃ n : ℕ, n * n = smallest_square_side_length * smallest_square_side_length)
    (h1 : ∃ m : ℕ, m * rectangle_area = smallest_square_side_length * smallest_square_side_length) :
    m = 9 :=
by
  sorry

end smallest_number_of_rectangles_l794_794807


namespace original_card_count_l794_794840

theorem original_card_count
  (r b : ℕ)
  (initial_prob_red : (r : ℚ) / (r + b) = 2 / 5)
  (prob_red_after_adding_black : (r : ℚ) / (r + (b + 6)) = 1 / 3) :
  r + b = 30 := sorry

end original_card_count_l794_794840


namespace largest_base5_number_conversion_l794_794009

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794009


namespace centroid_quadrilateral_area_l794_794306

-- Defining the square and the point Q inside it
variables {X Y Z W Q : Type*} [point : XY] [square : XYZ] [point_XQ : metric_space] [point_YQ : metric_space]

-- Given conditions
axiom square_side_length (XYZW : square) : (side_length XYZW = 24)
axiom point_XQ (X Q : XY) : (dist X Q = 15)
axiom point_YQ (Y Q : XY) : (dist Y Q = 20)

-- Problem
theorem centroid_quadrilateral_area :
  ∀ (XYZW : square) (XQ : metric_space) (YQ : metric_space),
  square_side_length XYZW →
  point_XQ X Q →
  point_YQ Y Q →
  ∃ (area : ℝ), area = 128 :=
by
  sorry

end centroid_quadrilateral_area_l794_794306


namespace distinct_real_roots_not_two_l794_794567

noncomputable 
def f (x a b c: ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

theorem distinct_real_roots_not_two (a b c m: ℝ) 
  (h1: f m a b c = m) 
  (h2: f' m a b c = 0) : 
  ¬(∃ x1 x2: ℝ, x1 ≠ x2 ∧ 3 * (f x1 a b c)^2 - 2 * a * (f x1 a b c) - b = 0 ∧ 
                         3 * (f x2 a b c)^2 - 2 * a * (f x2 a b c) - b = 0) :=
sorry

end distinct_real_roots_not_two_l794_794567


namespace nth_term_pattern_term_2012_l794_794211

-- Define the sequence as given in the problem.
def sequence (n : ℕ) : ℤ :=
  match n with
  | 0       => -2
  | (n + 1) => (-2) * (sequence n)

-- Prove the nth term of the sequence using the established pattern.
theorem nth_term_pattern (n : ℕ) : sequence n = (-1)^n * 2^n :=
  sorry

-- Prove the specific case for the 2012th term.
theorem term_2012 : sequence 2012 = 2^2012 :=
  by 
    have h : sequence 2012 = (-1)^2012 * 2^2012 := nth_term_pattern 2012
    rw pow_even (show 2012 % 2 = 0, by norm_num) at h
    rw one_mul at h
    exact h

end nth_term_pattern_term_2012_l794_794211


namespace elevator_ride_combinations_l794_794624

-- Given conditions
def elevators : List String := ["A", "B", "C", "D"]
def num_people : ℕ := 3
def same_elevator_count : ℕ := 2

-- Problem: Prove that the number of different ways three people can ride the elevators with exactly two people taking the same elevator is 36
theorem elevator_ride_combinations : 
  ∃ (ways : ℕ), ways = 36 ∧ 
  (num_people = 3 ∧ same_elevator_count = 2 ∧ length elevators = 4) := 
sorry

end elevator_ride_combinations_l794_794624


namespace temperature_on_friday_l794_794405

theorem temperature_on_friday 
  (M T W Th F : ℤ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : M = 43) : 
  F = 35 := 
by
  sorry

end temperature_on_friday_l794_794405


namespace base5_to_base10_max_l794_794021

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794021


namespace three_consecutive_odd_numbers_l794_794788

theorem three_consecutive_odd_numbers (x : ℤ) (h : x - 2 + x + x + 2 = 27) : 
  (x + 2, x, x - 2) = (11, 9, 7) :=
by
  sorry

end three_consecutive_odd_numbers_l794_794788


namespace line_intersects_circle_l794_794774

noncomputable def line (m : ℝ) : ℝ × ℝ -> Prop :=
  λ (p : ℝ × ℝ), m * p.1 - p.2 + 1 - m = 0

def circle : ℝ × ℝ -> Prop :=
  λ (p : ℝ × ℝ), p.1 ^ 2 + (p.2 - 1) ^ 2 = 5

theorem line_intersects_circle (m : ℝ) :
  ∃ p : ℝ × ℝ, line m p ∧ circle p :=
sorry

end line_intersects_circle_l794_794774


namespace compute_expression_l794_794131

theorem compute_expression : 6^3 - 4 * 5 + 2^4 = 212 := by
  sorry

end compute_expression_l794_794131


namespace prime_divides_sum_of_fractions_l794_794290

theorem prime_divides_sum_of_fractions (p : ℕ) (k m n : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : p = 4 * k + 1) 
  (h3 : Nat.gcd m n = 1) 
  (h4 : ∑ a in Finset.range (p - 2), 1 / (a ^ ((p - 1) / 2) + a ^ ((p + 1) / 2)) = m / n) : 
  p ∣ (m + n) := 
sorry

end prime_divides_sum_of_fractions_l794_794290


namespace log_identity_l794_794454

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_identity :
    2 * log_base_10 2 + log_base_10 (5 / 8) - log_base_10 25 = -1 :=
by 
  sorry

end log_identity_l794_794454


namespace books_distribution_1_books_distribution_2_l794_794373

-- Problem 1
theorem books_distribution_1 :
  let books := 7
  let distribute_to (a b c : ℕ) (books : ℕ) := a + b + c = books
  let num_ways := choose books 1 * choose (books - 1) 2 * choose (books - 3) 4 * fact 3 in
  distribute_to 1 2 4 books → num_ways = 630 :=
by
  intros
  sorry

-- Problem 2
theorem books_distribution_2 :
  let books := 7
  let distribute_to (a b c : ℕ) (books : ℕ) := a + b + c = books
  let num_ways := choose books 2 * choose (books - 2) 2 * choose (books - 4) 3 * fact 3 / fact 2 in
  distribute_to 3 2 2 books → num_ways = 630 :=
by
  intros
  sorry

end books_distribution_1_books_distribution_2_l794_794373


namespace compare_magnitudes_l794_794558

theorem compare_magnitudes (a b : ℝ) (ha : a > 0) (hb : b > 0) : (Math.sqrt a + Math.sqrt b) > Math.sqrt (a + b) := 
by
  sorry

end compare_magnitudes_l794_794558


namespace total_votes_approximately_9333_l794_794263

/-- In an election, Candidate A gets 50% of the votes, Candidate B gets 35% of the votes,
    and Candidate B is behind Candidate A by 1400 votes.
    Prove that the total number of votes polled is approximately 9333. -/
theorem total_votes_approximately_9333 (V : ℕ)
    (hA : V * 50 / 100 = x)  -- Candidate A gets 50% of the votes
    (hB : V * 35 / 100 = y)  -- Candidate B gets 35% of the votes
    (hAB : x - y = 1400) :   -- Candidate A is ahead of Candidate B by 1400 votes
    V ≈ 9333 := sorry

end total_votes_approximately_9333_l794_794263


namespace domain_of_f_l794_794759

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / (Real.sqrt (1 - x)) + Real.log (3 * x + 1)

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 → (- 1/3 < x ∧ x < 1) :=
by {
  intro x,
  split,
  { intro h,
    sorry
  }
}

end domain_of_f_l794_794759


namespace inverse_parallelogram_prop_l794_794357

theorem inverse_parallelogram_prop (Q : Type) [quadrilateral Q] : 
  (∀ (P : Type) [parallelogram P], has_two_pairs_of_parallel_sides P) → 
  (∀ Q, has_two_pairs_of_parallel_sides Q → is_parallelogram Q) :=
sorry

end inverse_parallelogram_prop_l794_794357


namespace prob_no_rain_correct_l794_794775

-- Define the probability of rain on each of the next five days
def prob_rain_each_day : ℚ := 1 / 2

-- Define the probability of no rain on a single day
def prob_no_rain_one_day : ℚ := 1 - prob_rain_each_day

-- Define the probability of no rain in any of the next five days
def prob_no_rain_five_days : ℚ := prob_no_rain_one_day ^ 5

-- Theorem statement
theorem prob_no_rain_correct : prob_no_rain_five_days = 1 / 32 := by
  sorry

end prob_no_rain_correct_l794_794775


namespace perpendicular_lines_alternative_l794_794233

theorem perpendicular_lines (a : ℝ) : (a ≠ 0 ∧ (-(2*a - 1)/(a) * a = -1)) → (a = 2) :=
  begin
    sorry
  end

theorem alternative (a : ℝ) : (a = 0)

end perpendicular_lines_alternative_l794_794233


namespace solution_set_f_gt_2x_l794_794568

variable (f : ℝ → ℝ)

def f_dom : Set ℝ := Set.univ

theorem solution_set_f_gt_2x (h1 : ∀ x : ℝ, f'' x > 2)
    (h2 : f 1 = 2) : {x : ℝ | f x > 2 * x} = {x : ℝ | 1 < x} :=
  sorry

end solution_set_f_gt_2x_l794_794568


namespace hyperbola_equation_l794_794791

noncomputable def distance_between_vertices : ℝ := 8
noncomputable def eccentricity : ℝ := 5 / 4

theorem hyperbola_equation :
  ∃ a b c : ℝ, 2 * a = distance_between_vertices ∧ 
               c = a * eccentricity ∧ 
               b^2 = c^2 - a^2 ∧ 
               (a = 4 ∧ c = 5 ∧ b^2 = 9) ∧ 
               ∀ x y : ℝ, (x^2 / (a:ℝ)^2) - (y^2 / (b:ℝ)^2) = 1 :=
by 
  sorry

end hyperbola_equation_l794_794791


namespace current_in_circuit_l794_794637

open Complex

theorem current_in_circuit
  (V : ℂ := 2 + 3 * I)
  (Z : ℂ := 4 - 2 * I) :
  (V / Z) = (1 / 10 + 4 / 5 * I) :=
  sorry

end current_in_circuit_l794_794637


namespace exist_acute_angled_triangle_l794_794889

noncomputable def construct_triangle (A' B' C' : Point) : Triangle := sorry

theorem exist_acute_angled_triangle (A' B' C' : Point) 
  (h_A': reflection_orhtocenter A' (side BC)) 
  (h_B': reflection_orhtocenter B' (side CA)) 
  (h_C': reflection_orhtocenter C' (side AB)) : 
  ∃ (ABC : Triangle), is_acute_angled ABC ∧ 
                      reflection_orhtocenter A' (side BC) ∧ 
                      reflection_orhtocenter B' (side CA) ∧ 
                      reflection_orhtocenter C' (side AB) := 
sorry

end exist_acute_angled_triangle_l794_794889


namespace count_even_numbers_with_distinct_digits_l794_794238

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  digits.0 ≠ digits.1 ∧ digits.0 ≠ digits.2 ∧ digits.1 ≠ digits.2

def valid_even_numbers (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 998 ∧ is_even n ∧ has_distinct_digits n

theorem count_even_numbers_with_distinct_digits :
  { n : ℕ | valid_even_numbers n }.to_finset.card = 288 :=
by 
  sorry

end count_even_numbers_with_distinct_digits_l794_794238


namespace inequality_D_no_solution_l794_794107

theorem inequality_D_no_solution :
  ¬ ∃ x : ℝ, 2 - 3 * x + 2 * x^2 ≤ 0 := 
sorry

end inequality_D_no_solution_l794_794107


namespace sample_arithmetic_sequence_l794_794379

def is_arithmetic_sequence (s: List ℕ) : Prop :=
  ∃ d, ∀ i j, i < j ∧ j < s.length → s[j] - s[i] = (j - i) * d

def valid_systematic_sample (total_buses : ℕ) (sampling_ratio : ℕ) (sample : List ℕ) : Prop :=
  (sampling_ratio >= 1) ∧ (sample.length = total_buses / sampling_ratio)

theorem sample_arithmetic_sequence:
  ∀ (total_buses sampling_ratio: ℕ) (sample: List ℕ),
    valid_systematic_sample total_buses sampling_ratio sample → sample = [103, 133, 153, 193] →
    is_arithmetic_sequence sample :=
by
  -- Proof to be completed
  sorry

end sample_arithmetic_sequence_l794_794379


namespace leading_coefficient_of_polynomial_is_five_l794_794160

def leading_coefficient (p : Polynomial ℤ) : ℤ := 
  p.coeff (p.natDegree)

def polynomial := 5 * Polynomial.x^5 - 15 * Polynomial.x^4 + 10 * Polynomial.x^3 -
  6 * Polynomial.x^5 - 6 * Polynomial.x^3 - 6 * Polynomial.C 1 + 
  6 * Polynomial.x^5 - 2 * Polynomial.x^4 + 2 * Polynomial.x^2

theorem leading_coefficient_of_polynomial_is_five :
  leading_coefficient polynomial = 5 :=
by
  sorry

end leading_coefficient_of_polynomial_is_five_l794_794160


namespace infinite_broken_line_length_correct_l794_794947

-- Define the acute angle, α, in radians
variables (α : ℝ) (hα : 0 < α ∧ α < π / 2)

-- Define the length m of the first segment A1A2
variables (m : ℝ) (hm : 0 < m)

-- Define the recursive segment lengths using cosine
def segment_length (n : ℕ) : ℝ :=
m * (Real.cos α) ^ (n - 1)

-- Define the sum of the infinite geometric series for the segment lengths
def infinite_broken_line_length : ℝ :=
m / (1 - Real.cos α)

-- Main theorem
theorem infinite_broken_line_length_correct :
  ∑' n, segment_length α m n = infinite_broken_line_length α m := by
  sorry

end infinite_broken_line_length_correct_l794_794947


namespace sum_of_digits_of_fraction_is_nine_l794_794897

theorem sum_of_digits_of_fraction_is_nine : 
  ∃ (x y : Nat), (4 / 11 : ℚ) = x / 10 + y / 100 + x / 1000 + y / 10000 + (x + y) / 100000 -- and other terms
  ∧ x + y = 9 := 
sorry

end sum_of_digits_of_fraction_is_nine_l794_794897


namespace bicycle_speed_l794_794800

theorem bicycle_speed (x : ℝ) (h : (2.4 / x) - (2.4 / (4 * x)) = 0.5) : 4 * x = 14.4 :=
by
  sorry

end bicycle_speed_l794_794800


namespace sum_of_divisors_of_12_equals_28_l794_794896

-- Define the problem conditions and the required proof
theorem sum_of_divisors_of_12_equals_28 :
  (∑ n in Finset.filter (λ n, (12 % n = 0)) (Finset.range 13), n) = 28 :=
sorry

end sum_of_divisors_of_12_equals_28_l794_794896


namespace largest_base_5_five_digits_base_10_value_l794_794040

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794040


namespace min_cos_A_max_area_triangle_l794_794656

def triangle := Type
variables {A B C : ℝ} {a b c : ℝ}
variables {A B C : triangle} -- Define a generic triangle

/- Helper variables and definitions for side lengths -/
variable (side_a : ℝ := 2) -- Given a = 2
variables (side_b side_c : ℝ) -- Define sides b and c

/-- Condition: b^2, a^2, c^2 form an arithmetic progression -/
def arithmetic_progression (side_a side_b side_c : ℝ) : Prop :=
  2 * side_a^2 = side_b^2 + side_c^2

/-- Find the minimum value of cos A -/
theorem min_cos_A {A B C : Triangle}
  (h_prog: arithmetic_progression side_a side_b side_c) :
  cos A ≥ 1/2 := 
sorry

/-- Find the maximum area of triangle ABC when A is largest and a = 2 -/
theorem max_area_triangle
  (h_prog: arithmetic_progression side_a side_b side_c)
  (side_a_eq_two : side_a = 2) :
  ∃ S, S = sqrt 3 := 
sorry

end min_cos_A_max_area_triangle_l794_794656


namespace correct_circle_eq_l794_794940

variables (a b r : ℝ)

def point_on_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

def symmetric_point (x y : ℝ) : ℝ × ℝ :=
  let u := x + 2 * y in
  ((2 * u - x), (-u))

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 1) / Real.sqrt 2

def circle_eq (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), 
    a + 2 * b = 0 ∧
    point_on_circle x y a b r ∧
    ∃ z w, symmetric_point x y = (z, w) ∧ point_on_circle z w a b r ∧
    r^2 - (distance_to_line a b)^2 = 2

theorem correct_circle_eq 
  (h : circle_eq 2 3) :
  ((a = 6 ∧ b = -3 ∧ r = Real.sqrt 52) ∨ 
   (a = 14 ∧ b = -7 ∧ r = Real.sqrt 244)) → 
  ((x - 6)^2 + (y + 3)^2 = 52) ∨ 
  ((x - 14)^2 + (y + 7)^2 = 244) :=
begin 
  sorry 
end

end correct_circle_eq_l794_794940


namespace sum_b_sq_lt_5_div_12_l794_794276

noncomputable def a_seq : ℕ → ℕ
| 0     := 4
| (n+1) := let Sn := (finset.range (n+1)).sum a_seq in
           (Sn^2 - n * Sn - n - 1) / (Sn + 1)

noncomputable def S_sum (n : ℕ) : ℕ := (finset.range n).sum a_seq

noncomputable def b_seq (n : ℕ) : ℝ := 
  let an := a_seq n in
  (2^(n-1 : ℕ) + 1 : ℝ) / ((3 * n - 2) * an)

noncomputable def T_sum (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, (b_seq (i+1))^2)

theorem sum_b_sq_lt_5_div_12 (n : ℕ) : T_sum (n+1) < 5 / 12 := 
sorry

end sum_b_sq_lt_5_div_12_l794_794276


namespace number_of_zeros_of_f_l794_794977

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - Real.log x - x + 1

def f' (x : ℝ) : ℝ := Real.log x - 1 / x

theorem number_of_zeros_of_f' : ∃! x : ℝ, 0 < x ∧ f' x = 0 :=
  sorry

end number_of_zeros_of_f_l794_794977


namespace arccos_zero_eq_pi_div_two_l794_794470

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l794_794470


namespace shaded_areas_total_l794_794647

-- Define the conditions
def larger_circle_area : ℝ := 100 * Real.pi
def larger_circle_sectors : ℕ := 4
def smaller_circle_sectors : ℕ := 2
def shaded_larger_circle_sectors : ℕ := 2
def shaded_smaller_circle_sectors : ℕ := 1

-- Define the theorem to be proven
theorem shaded_areas_total : 
  let R := Real.sqrt(larger_circle_area / Real.pi) in
  let small_circle_radius := R / 2 in
  let small_circle_area := Real.pi * (small_circle_radius ^ 2) in
  shaded_larger_circle_sectors * (larger_circle_area / larger_circle_sectors) + 
  shaded_smaller_circle_sectors * (small_circle_area / smaller_circle_sectors) = 62.5 * Real.pi := 
sorry

end shaded_areas_total_l794_794647


namespace geometric_sequence_product_l794_794959

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Typically defined by the specifics of the geometric sequence

open Classical

theorem geometric_sequence_product :
  (∀ n : ℕ, a_n n > 0) →
  (Real.log10 (a_n 3 * a_n 8 * a_n 13) = 6) →
  (a_n 1 * a_n 15 = 10000) :=
by
  intros pos_terms log_eq
  sorry -- Proof steps would go here

end geometric_sequence_product_l794_794959


namespace common_ratio_is_two_l794_794201

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l794_794201


namespace min_distance_from_C1_to_C2_l794_794269

-- Define the parametric equations of l1 and l2.
def l1 (t k : ℝ) : ℝ × ℝ := (t - real.sqrt 3, k * t)
def l2 (m k : ℝ) : ℝ × ℝ := (real.sqrt 3 - m, m / (3 * k))

-- Definition of the curve C1 and the condition on k.
def C1_eq (x y : ℝ) : Prop := (x^2 / 3 + y^2 = 1) ∧ (y ≠ 0)

-- Define the polar equation of line C2.
def C2_eq (rho θ : ℝ) : Prop := rho * real.sin (θ + real.pi / 4) = 4 * real.sqrt 2

-- Conversion from polar to Cartesian equation.
def C2_cartesian (x y : ℝ) : Prop := x + y - 8 = 0

-- Define a point Q on curve C1.
def Q_on_C1 (α : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.cos α, real.sin α)

-- Define the distance from a point to a line.
def distance (x y : ℝ) (a b c : ℝ) : ℝ := (real.abs (a * x + b * y + c)) / real.sqrt (a^2 + b^2)

-- Define the minimum distance proof.
theorem min_distance_from_C1_to_C2 :
  ∀ α : ℝ, let (x_Q, y_Q) := Q_on_C1 α in
  C1_eq x_Q y_Q →
  (distance x_Q y_Q 1 1 (-8) = 3 * real.sqrt 2) :=
by sorry

end min_distance_from_C1_to_C2_l794_794269


namespace minimum_chord_length_l794_794252

theorem minimum_chord_length (a : ℝ) :
  let d := sqrt (2 * ((arcsin a)^2 + (arccos a)^2)) 
  in d ≥ (π / 2) := 
sorry

end minimum_chord_length_l794_794252


namespace parabola_max_curvature_at_vertex_parabola_curvature_decreases_hyperbola_max_curvature_near_vertices_ellipse_max_curvature_at_minor_axis_ends_ellipse_min_curvature_at_major_axis_ends_l794_794660

-- Parabola y = ax^2
def parabola_curvature (a x : ℝ) : ℝ :=
  |2 * a| / (1 + (2 * a * x)^2)^(3/2)

theorem parabola_max_curvature_at_vertex (a : ℝ) : 
  parabola_curvature a 0 = |2 * a| :=
sorry

theorem parabola_curvature_decreases (a x : ℝ) (h : 0 < x) :
  parabola_curvature a x < |2 * a| :=
sorry

-- Hyperbola x^2/a^2 - y^2/b^2 = 1
theorem hyperbola_max_curvature_near_vertices :
  ∀ (a b x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → 
  (∃ (x₀ y₀ : ℝ), (x₀^2 / a^2 - y₀^2 / b^2 = 1) ∧ 
  (parabola_curvature a x < parabola_curvature a x₀)) :=
sorry

-- Ellipse x^2/a^2 + y^2/b^2 = 1
theorem ellipse_max_curvature_at_minor_axis_ends (a b : ℝ) :
  parabola_curvature a b = a / b^2 :=
sorry

theorem ellipse_min_curvature_at_major_axis_ends (a b : ℝ) :
  parabola_curvature b a = b / a^2 :=
sorry

end parabola_max_curvature_at_vertex_parabola_curvature_decreases_hyperbola_max_curvature_near_vertices_ellipse_max_curvature_at_minor_axis_ends_ellipse_min_curvature_at_major_axis_ends_l794_794660


namespace largest_base5_to_base10_l794_794019

theorem largest_base5_to_base10 : 
  let n := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 := 
by 
  let n := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  have h : n = 3124 := 
    by 
    -- calculations skipped, insert actual calculation steps or 'sorry'
    sorry
  exact h

end largest_base5_to_base10_l794_794019


namespace tetrahedron_altitudes_do_not_intersect_at_single_point_l794_794638

theorem tetrahedron_altitudes_do_not_intersect_at_single_point :
  ∃ (A B C D : ℝ × ℝ × ℝ),
    let tetrahedron : set (ℝ × ℝ × ℝ) := {A, B, C, D} in
    let altitudes : set (ℝ × ℝ × ℝ) := {p | ∃ (V : ℝ × ℝ × ℝ) (plane : set (ℝ × ℝ × ℝ)),
      V ∈ tetrahedron ∧ plane ⊆ convex_hull 𝕜 ({A, B, C, D} \ {V}) ∧
      ∃ (h : line ℝ), p ∈ h ∧ h ⊆ perpendicular ℝ V plane } in
    ¬disjoint altitudes :=
begin
  sorry
end

end tetrahedron_altitudes_do_not_intersect_at_single_point_l794_794638


namespace ab_perfect_cube_l794_794961

theorem ab_perfect_cube (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) (h_lt : b < a)
  (h_div : ab(a - b) ∣ a^3 + b^3 + ab) : ∃ m : ℕ, ab = m^3 :=
by
  sorry

end ab_perfect_cube_l794_794961


namespace smallest_num_students_in_choir_l794_794627

-- Problem Definitions
def num_students_per_row (x : ℕ) : ℕ := 5 * x + 3
def condition (x : ℕ) : Prop := (5 * x + 3) > 45

-- Statement of the problem
theorem smallest_num_students_in_choir : ∃ (x : ℕ), condition x ∧ num_students_per_row x = 48 :=
by {
  exists 9,
  split,
  { simp [condition], norm_num, },
  { simp [num_students_per_row], norm_num, }
}

end smallest_num_students_in_choir_l794_794627


namespace hyperbola_asymptotes_ellipse_equation_l794_794581

variables (a b c : ℝ)
variables (x y : ℝ)

def hyperbola := (x^2) / (a^2) - (y^2) / (b^2) = 1
def ellipse := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem hyperbola_asymptotes (a b c: ℝ) (x y: ℝ) (h1: a > 0) (h2: b > 0) (h3: b = 1) (h4: c = sqrt 3) :
    hyperbola x y a b → (a^2 = 2) → y = ± (sqrt 2 / 2) * x := sorry

theorem ellipse_equation (a b c: ℝ) (x y: ℝ) (h1: a = sqrt 3) (h2: c = sqrt 2) :
    ellipse x y a b → (b^2 = 1) → (x^2 / 3 + y^2 = 1) := sorry

end hyperbola_asymptotes_ellipse_equation_l794_794581


namespace arithmetic_expression_eval_l794_794875

theorem arithmetic_expression_eval :
  -1 ^ 4 + (4 - ((3 / 8 + 1 / 6 - 3 / 4) * 24)) / 5 = 0.8 := by
  sorry

end arithmetic_expression_eval_l794_794875


namespace inscribed_circle_center_on_tangent_circle_l794_794413

structure Circle (point : Type) :=
(center : point)
(radius : ℝ)

variables {point : Type} [MetricSpace point]

theorem inscribed_circle_center_on_tangent_circle {A B C : point} (ω : Circle point)
  (h_tangent_AB : dist ω.center B = ω.radius)
  (h_tangent_AC : dist ω.center C = ω.radius) :
  let incenter := sorry in
  dist incenter ω.center = ω.radius := sorry

end inscribed_circle_center_on_tangent_circle_l794_794413


namespace angle_SAB_length_CQ_l794_794314

-- Definitions of the points and constants given in the problem
def point := ℝ × ℝ × ℝ

constant A B C S K L M N P Q : point
constant ω1 ω2 : set point

-- Conditions from the problem
axiom point_K_on_AC : ∃ t, K = (1-t) • A + t • C
axiom point_L_on_BC : ∃ t, L = (1-t) • B + t • C
axiom point_M_on_BS : ∃ t, M = (1-t) • B + t • S
axiom point_N_on_AS : ∃ t, N = (1-t) • A + t • S
axiom KLMN_coplanar : ∃ π : set point, K ∈ π ∧ L ∈ π ∧ M ∈ π ∧ N ∈ π
axiom KL_eq_MN : dist K L = 2 ∧ dist M N = 2
axiom KN_eq_LM : dist K N = 18 ∧ dist L M = 18
axiom Ω1_tangent_conditions : ω1 ∈ sphere ∧ ∀ (x ∈ ω1), x ∈ plane ∧ dist x KN = r ∧ dist x KL = r ∧ dist x LM = r
axiom Ω2_tangent_conditions : ω2 ∈ sphere ∧ ∀ (x ∈ ω2), x ∈ plane ∧ dist x KN = r ∧ dist x LM = r ∧ dist x MN = r
axiom cone_F1_conditions : vertex_P_on_AB : ∃ t, P = (1-t) • A + t • B
axiom cone_F2_conditions : vertex_Q_on_CS : ∃ t, Q = (1-t) • C + t • S

-- Questions to prove
theorem angle_SAB : ∃ θ, θ = arccos (1 / 6) :=
sorry

theorem length_CQ : ∃ d, d = 52 / 3 :=
sorry

end angle_SAB_length_CQ_l794_794314


namespace number_of_distinct_integers_from_special_fractions_sums_l794_794878

def is_special (a b : ℕ) : Prop := a + b = 15

def special_fractions : List ℚ :=
  (List.range 14).map (λ k => (k+1 : ℚ) / (15 - (k+1)))

def valid_sums (f g : ℚ) : Proposition :=
  (f + g).denom = 1

theorem number_of_distinct_integers_from_special_fractions_sums :
  (special_fractions.product special_fractions).filter (λ p => valid_sums p.1 p.2) .map (λ p => (p.1 + p.2).nat).erase_dup.length = 9 :=
sorry

end number_of_distinct_integers_from_special_fractions_sums_l794_794878


namespace cost_of_Colombian_coffee_beans_l794_794803

theorem cost_of_Colombian_coffee_beans (
  total_weight : ℝ := 40,
  cost_per_pound_mix : ℝ := 4.60,
  weight_Colombian : ℝ := 28.8,
  cost_per_pound_Peruvian : ℝ := 4.25
) : 
  let total_cost := total_weight * cost_per_pound_mix
      weight_Peruvian := total_weight - weight_Colombian
      total_cost_Peruvian := weight_Peruvian * cost_per_pound_Peruvian
      total_cost_Colombian := total_cost - total_cost_Peruvian
      cost_per_pound_Colombian := total_cost_Colombian / weight_Colombian
  in cost_per_pound_Colombian = 4.74 :=
by
  sorry

end cost_of_Colombian_coffee_beans_l794_794803


namespace find_center_of_symmetry_l794_794985

noncomputable def f (x : ℝ) : ℝ :=
  (sin (2 * x)) ^ 2 - (sin (2 * x)) * (cos (2 * x))

noncomputable def isCenterOfSymmetry (A : ℝ × ℝ) : Prop :=
  ∃ x0 ∈ set.Icc 0 (π / 2), A = (x0, 1 / 2) ∧
    (∃ k : ℤ, 4 * x0 + π / 4 = k * π)

theorem find_center_of_symmetry :
  (isCenterOfSymmetry (3 * π / 16, 1 / 2)) ∨ (isCenterOfSymmetry (7 * π / 16, 1 / 2)) :=
by 
  sorry

end find_center_of_symmetry_l794_794985


namespace number_of_toys_gained_l794_794090

theorem number_of_toys_gained
  (num_toys : ℕ) (selling_price : ℕ) (cost_price_one_toy : ℕ)
  (total_cp := num_toys * cost_price_one_toy)
  (profit := selling_price - total_cp)
  (num_toys_equiv_to_profit := profit / cost_price_one_toy) :
  num_toys = 18 → selling_price = 23100 → cost_price_one_toy = 1100 → num_toys_equiv_to_profit = 3 :=
by
  intros h1 h2 h3
  -- Proof to be completed
  sorry

end number_of_toys_gained_l794_794090


namespace nonneg_real_sum_inequality_l794_794560

theorem nonneg_real_sum_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end nonneg_real_sum_inequality_l794_794560


namespace correct_transformation_l794_794810

theorem correct_transformation (x : ℝ) :
  3 + x = 7 ∧ ¬ (x = 7 + 3) ∧
  5 * x = -4 ∧ ¬ (x = -5 / 4) ∧
  (7 / 4) * x = 3 ∧ ¬ (x = 3 * (7 / 4)) ∧
  -((x - 2) / 4) = 1 ∧ (-(x - 2)) = 4 :=
by
  sorry

end correct_transformation_l794_794810


namespace largest_isosceles_right_triangle_proof_l794_794902

noncomputable def largest_isosceles_right_triangle (ABC : Triangle) : Triangle :=
sorry

theorem largest_isosceles_right_triangle_proof (ABC : Triangle) (AC BC : Segment) 
  (circumcircle_AC circumcircle_BC : Circle) (E F : Point) 
  (H1 : circumcircle_AC.is_circumcircle AC) 
  (H2 : circumcircle_BC.is_circumcircle BC)
  (H3 : circumcircle_AC.angle_subtended AC = 45) 
  (H4 : circumcircle_BC.angle_subtended BC = 45)
  (H5 : parallel (line_through C (center circumcircle_AC)) (line_through (center circumcircle_AC) (center circumcircle_BC)) = (line_through E F))
  : largest_isosceles_right_triangle ABC = triangle E F (line_through E F) :=
sorry

end largest_isosceles_right_triangle_proof_l794_794902


namespace train_speed_is_63_kmph_l794_794105

def train_length := 560 -- meters
def time_to_pass_tree := 32 -- seconds

def speed_of_train : ℝ := (train_length / 1000) / (time_to_pass_tree / 3600)

theorem train_speed_is_63_kmph : speed_of_train ≈ 63 := 
by
  sorry

end train_speed_is_63_kmph_l794_794105


namespace find_angle_A_find_value_n_l794_794657

-- Define the conditions 
def angles_arithmetic_sequence (A B C: ℝ) (h1:  2 * B = A + C) (h2: A + B + C = Real.pi): Prop :=
  True

def side_condition (a b c: ℝ) (h: c = 2 * a): Prop :=
  True

-- Define the first problem: finding angle A
theorem find_angle_A (A B C a b c : ℝ) (habc : angles_arithmetic_sequence A B C (2 * B = A + C) (A + B + C = Real.pi))
  (hside : side_condition a b c (c = 2 * a)) : 
  A = Real.pi / 6 :=
sorry

-- Define the sequence and its properties
def sequence_term (n C : ℕ) : ℝ := 2^n * |Real.cos (n * C)|
def sequence_sum (n C : ℕ) : ℝ := Finset.sum (Finset.range n) (λ k, sequence_term k C)

-- Define the second problem: find value of n
theorem find_value_n (n k : ℕ) (S_n : ℝ) (hk : k = 2) (hS_n : S_n = 20) : 
  sequence_sum n (Real.pi / 2) = 20 → n = 4 ∨ n = 5 :=
sorry

end find_angle_A_find_value_n_l794_794657


namespace largest_base5_number_conversion_l794_794008

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794008


namespace arithmetic_sequence_sum_l794_794780

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end arithmetic_sequence_sum_l794_794780


namespace ball_reaches_height_less_than_2_after_6_bounces_l794_794831

theorem ball_reaches_height_less_than_2_after_6_bounces :
  ∃ (k : ℕ), 16 * (2/3) ^ k < 2 ∧ ∀ (m : ℕ), m < k → 16 * (2/3) ^ m ≥ 2 :=
by
  sorry

end ball_reaches_height_less_than_2_after_6_bounces_l794_794831


namespace distance_between_lines_l794_794951

-- Define lines l1 and l2
def line_l1 (x y : ℝ) := x + y + 1 = 0
def line_l2 (x y : ℝ) := 2 * x + 2 * y + 3 = 0

-- Proof statement for the distance between parallel lines
theorem distance_between_lines :
  let a := 1
  let b := 1
  let c1 := 1
  let c2 := 3 / 2
  let distance := |c2 - c1| / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 2 / 4 :=
by
  sorry

end distance_between_lines_l794_794951


namespace range_of_a_for_local_max_l794_794573

theorem range_of_a_for_local_max (a : ℝ) :
  (∀ x : ℝ, (f'(x) = a * (x + 1) * (x - a)) →  (∀ x < a, a * (x + 1) * (x - a) > 0) ∧ (∀ x > a, a * (x + 1) * (x - a) < 0)) →
  -1 < a ∧ a < 0 := 
sorry


end range_of_a_for_local_max_l794_794573


namespace order_of_six_does_not_exist_l794_794486

def f (x : ℕ) : ℕ := x^2 % 13

def order_of_six (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 -> n ≠ (λ k, f^[k] 6) k

theorem order_of_six_does_not_exist : ¬ (∃ n : ℕ, order_of_six n) := 
by
  sorry

end order_of_six_does_not_exist_l794_794486


namespace hyperbola_eccentricity_2_l794_794990

noncomputable def parabola_focus : ℝ × ℝ :=
(2, 0)

noncomputable def hyperbola_focus (m : ℝ) : ℝ :=
has_foci (m + 3) 3

theorem hyperbola_eccentricity_2 : ∀ (m : ℝ), parabola_focus = (2, 0) → hyperbola_focus m = (2, 0) → m = 1 → eccentricity (hyperbola_focus m) = 2 :=
by
  sorry

end hyperbola_eccentricity_2_l794_794990


namespace sum_of_solutions_l794_794513

theorem sum_of_solutions :
  ( ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → 
  ( -12 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) ) ) → 
  ∑ x in {real.sqrt 3, -real.sqrt 3}, x = 0 := 
by
sorry

end sum_of_solutions_l794_794513


namespace no_distance_preserving_map_l794_794303

theorem no_distance_preserving_map (O : Point) (R r : ℝ) (S : SphericalCap) (f : S → Plane) :
  (∀ A B ∈ S, dist_sphere O R A B = dist_plane (f A) (f B)) →
  2 * π * R * sin (r / R) ≠ 2 * π * r →
  false :=
by
  sorry

end no_distance_preserving_map_l794_794303


namespace rectangle_area_l794_794101

theorem rectangle_area (A : ℝ) (w : ℝ) (l : ℝ) (h1 : A = 36) (h2 : w^2 = A) (h3 : l = 3 * w) : w * l = 108 :=
by
sorrry

end rectangle_area_l794_794101


namespace csc_pi_over_18_minus_4_sin_pi_over_9_equals_2_l794_794880

theorem csc_pi_over_18_minus_4_sin_pi_over_9_equals_2 :
  Real.csc (Real.pi / 18) - 4 * Real.sin (Real.pi / 9) = 2 :=
  sorry

end csc_pi_over_18_minus_4_sin_pi_over_9_equals_2_l794_794880


namespace smallest_P_value_is_954_l794_794773

noncomputable def smallest_P_value : ℕ :=
  let nums := {2, 3, 4, 5, 6, 7, 8, 9, 10}
  in if h₁ : (∃ (p1 p2 p3 q1 q2 q3 r1 r2 r3 : ℕ),
      p1 < p2 ∧ p2 < p3 ∧ q1 < q2 ∧ q2 < q3 ∧ r1 < r2 ∧ r2 < r3 ∧
      {p1, p2, p3, q1, q2, q3, r1, r2, r3} = nums ∧
      let P := p1 * p2 * p3 + q1 * q2 * q3 + r1 * r2 * r3
      in P = 954) then 954 else 0

theorem smallest_P_value_is_954 : smallest_P_value = 954 :=
by
  have h₁ : (∃ (p1 p2 p3 q1 q2 q3 r1 r2 r3 : ℕ),
    p1 < p2 ∧ p2 < p3 ∧ q1 < q2 ∧ q2 < q3 ∧ r1 < r2 ∧ r2 < r3 ∧
    {p1, p2, p3, q1, q2, q3, r1, r2, r3} = {2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
    let P := p1 * p2 * p3 + q1 * q2 * q3 + r1 * r2 * r3
    in P = 954), from sorry,

  simp [smallest_P_value, h₁]

end smallest_P_value_is_954_l794_794773


namespace dogwood_trees_after_planting_l794_794793

-- Define the number of current dogwood trees and the number to be planted.
def current_dogwood_trees : ℕ := 34
def trees_to_be_planted : ℕ := 49

-- Problem statement to prove the total number of dogwood trees after planting.
theorem dogwood_trees_after_planting : current_dogwood_trees + trees_to_be_planted = 83 := by
  -- A placeholder for proof
  sorry

end dogwood_trees_after_planting_l794_794793


namespace expansion_non_integer_terms_l794_794598

noncomputable def integral_value : ℝ :=
  2 * (interval_integral (λ x : ℝ, x + |x|) (-3) 0 +
      interval_integral (λ x : ℝ, x + |x|) 0 3)

theorem expansion_non_integer_terms :
  let a := integral_value in
  2 * integral_value = 18 → 
  ∃ n : ℕ, n * (interval_integral (λ x : ℝ, x + |x|) 0 3) = 18 ∧
  n = 19 - 4 → n = 15 :=
by sorry

end expansion_non_integer_terms_l794_794598


namespace bullying_instances_l794_794287

-- Let's denote the total number of suspension days due to bullying and serious incidents.
def total_suspension_days : ℕ := (3 * (10 + 10)) + 14

-- Each instance of bullying results in a 3-day suspension.
def days_per_instance : ℕ := 3

-- The number of instances of bullying given the total suspension days.
def instances_of_bullying := total_suspension_days / days_per_instance

-- We must prove that Kris is responsible for 24 instances of bullying.
theorem bullying_instances : instances_of_bullying = 24 := by
  sorry

end bullying_instances_l794_794287


namespace digit_156_of_fraction_47_over_777_is_9_l794_794392

theorem digit_156_of_fraction_47_over_777_is_9 :
  let r := 47 / 777 in
  let decimal_expansion := 0.0 * 10^0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5) + -- and so on, repeating every 5 digits as "60459"
  (r = 0 + 6 * 10^(-1) + 0 * 10^(-2) + 4 * 10^(-3) + 5 * 10^(-4) + 9 * 10^(-5)) ∧ -- and so on
  let d := 156 in
  decimal_expansion.nth_digit(d) = 9 :=
sorry

end digit_156_of_fraction_47_over_777_is_9_l794_794392


namespace range_of_polynomial_fn_l794_794067

noncomputable def polynomial_fn (p0 : ℝ[X]) (p : ℕ → ℝ[X]) (a : ℕ → ℝ) (n : ℕ) : ℝ → ℝ :=
  λ x, p0.eval x + ∑ k in finset.range (n + 1), a k * |(p k).eval x|

theorem range_of_polynomial_fn (p0 : ℝ[X]) (p : ℕ → ℝ[X]) (a : ℕ → ℝ) (n : ℕ)
  (H : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → polynomial_fn p0 p a n x₁ ≠ polynomial_fn p0 p a n x₂) :
  set.range (polynomial_fn p0 p a n) = set.univ :=
sorry

end range_of_polynomial_fn_l794_794067


namespace knicks_from_knocks_l794_794608

variable (knicks knacks knocks : Type)
variable [HasSmul ℚ knicks] [HasSmul ℚ knacks] [HasSmul ℚ knocks]

variable (k1 : knicks) (k2 : knacks) (k3 : knocks)
variable (h1 : 5 • k1 = 3 • k2)
variable (h2 : 4 • k2 = 6 • k3)

theorem knicks_from_knocks : 36 • k3 = 40 • k1 :=
by {
  sorry
}

end knicks_from_knocks_l794_794608


namespace find_m_value_l794_794540

theorem find_m_value (a : ℕ → ℝ) (m x : ℝ)
  (h_poly : ∀ x, x^10 = ∑ i in finset.range 11, a i * (m - x)^i)
  (h_a8 : a 8 = 180) 
  (h_m_pos : 0 < m) :
  m = 2 :=
sorry

end find_m_value_l794_794540


namespace count_of_possible_x_l794_794245

theorem count_of_possible_x:
  {x : ℤ | (⌊real.sqrt (x:ℝ)⌋ = 6)}.card = 13 :=
by
  sorry

end count_of_possible_x_l794_794245


namespace suma_work_rate_l794_794404

theorem suma_work_rate (r s : ℝ) (hr : r = 1 / 5) (hrs : r + s = 1 / 4) : 1 / s = 20 := by
  sorry

end suma_work_rate_l794_794404


namespace find_f_property_l794_794683

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_property :
  (f 0 = 3) ∧ (∀ x y : ℝ, f (xy) = f ((x^2 + y^2) / 2) + (x - y)^2) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 - 2 * x) :=
by
  intros hypothesis
  -- Proof would be placed here
  sorry

end find_f_property_l794_794683


namespace distance_from_point_to_line_B1H_l794_794917

open Real

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2))

theorem distance_from_point_to_line_B1H :
  let B1 := (5, 8, -3)
  let D1 := (-3, 10, -5)
  let B := (3, 4, 1)
  ∃ H : ℝ × ℝ × ℝ,
  let DH_vector := (D1.1 - B.1, D1.2 - B.2, D1.3 - B.3)
  let n := (2 : ℝ) / (3 : ℝ)
  let H := (D1.1 + n * DH_vector.1, D1.2 + n * DH_vector.2, D1.3 + n * DH_vector.3)
  distance B1 H = 2 * sqrt 6 :=
begin
  sorry
end

end distance_from_point_to_line_B1H_l794_794917


namespace problem_1_problem_2_problem_3_l794_794086

-- Problem 1
noncomputable def a_n (n : ℕ) : ℝ := 1 / 2014 * Float.sin ((2 * n - 1) * Float.pi / 2)

theorem problem_1 : 
  (∀ n, 1 ≤ n ∧ n ≤ 2014 → (a_n n) = 1 / 2014 * Float.sin ((2 * n - 1) * Float.pi / 2)) →
  (∑ i in Finset.range 2014, a_n (i + 1)) = 0 ∧ ∑ i in Finset.range 2014, |a_n (i + 1)| = 1 := 
by sorry

-- Problem 2
noncomputable def b_n (k n : ℕ) : ℝ := if n % 2 = 1 then 1 / (2 * k) * (-1) ^ (n - 1) else -1 / (2 * k) * (-1) ^ (n - 1)

theorem problem_2 (k : ℕ) (hk : k > 0) : 
  (∀ n, 1 ≤ n ∧ n ≤ 2 * k → 
    (b_n k n) = if n % 2 = 1 then 1 / (2 * k) * (-1) ^ (n - 1) else -1 / (2 * k) * (-1) ^ (n - 1)) →
  ((∑ i in Finset.range (2 * k), b_n k (i + 1)) = 0) ∧ 
  (∑ i in Finset.range (2 * k), |b_n k (i + 1)| = 1) := 
by sorry

-- Problem 3
noncomputable def c_n (k n : ℕ) : ℝ := (-2 * k - 1 + 2 * n) / (2 * k ^ 2)

theorem problem_3 (k : ℕ) (hk : k > 0) : 
  (∀ n, 1 ≤ n ∧ n ≤ 2 * k → 
    (c_n k n) = (-2 * k - 1 + 2 * n) / (2 * k ^ 2) ∧ c_n k n < c_n k (n + 1)) →
  ((∑ i in Finset.range (2 * k), c_n k (i + 1)) = 0) ∧ 
  (∑ i in Finset.range (2 * k), |c_n k (i + 1)| = 1) := 
by sorry

end problem_1_problem_2_problem_3_l794_794086


namespace number_of_lines_through_P_l794_794448

theorem number_of_lines_through_P (A B C D A1 B1 C1 D1 P : Type)
  [Point A] [Point B] [Point C] [Point D]
  [Point A1] [Point B1] [Point C1] [Point D1]
  [Point P]
  (h_cube : is_cube A B C D A1 B1 C1 D1)
  (h_P_on_AB : on_edge P A B) :
  ∃ l1 l2 : Line, 
    passes_through l1 P ∧ 
    passes_through l2 P ∧ 
    angle_with_plane l1 (plane A B C D) = 30 ∧ 
    angle_with_plane l1 (plane A B C1 D1) = 30 ∧ 
    angle_with_plane l2 (plane A B C D) = 30 ∧ 
    angle_with_plane l2 (plane A B C1 D1) = 30 ∧ 
    l1 ≠ l2 := 
sorry

end number_of_lines_through_P_l794_794448


namespace probability_of_at_least_one_red_ball_l794_794074

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

def total_balls : ℕ := 6
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def green_balls : ℕ := 2
def drawn_balls : ℕ := 2

def total_outcomes : ℕ := combination total_balls drawn_balls
def non_red_balls : ℕ := yellow_balls + green_balls
def non_red_outcomes : ℕ := combination non_red_balls drawn_balls

theorem probability_of_at_least_one_red_ball :
  (1 - (non_red_outcomes / total_outcomes : ℚ)) = 3 / 5 :=
by
  sorry

end probability_of_at_least_one_red_ball_l794_794074


namespace pow_two_ge_square_l794_794733

theorem pow_two_ge_square {n : ℕ} (hn : n ≥ 4) : 2^n ≥ n^2 :=
sorry

end pow_two_ge_square_l794_794733


namespace line_through_point_with_opposite_intercepts_l794_794762

theorem line_through_point_with_opposite_intercepts :
  ∀ (x y : ℝ), (x = 3 ∧ y = -4) →
    (∃ a : ℝ, (a ≠ 0 ∧ a = 7) ∨ (a = 0 ∧ y = -4 / 3 * x)) →
    (4 * x + 3 * y = 0 ∨ x - y - 7 = 0) :=
by
  intros x y h1 h2
  rcases h1 with ⟨hx, hy⟩
  rcases h2 with ⟨a, h3 | h4⟩
  {
    use 7,
    split,
    assumption,
  sorry,
  sorry
  }
  {
  sorry,
  sorry,
  }

end line_through_point_with_opposite_intercepts_l794_794762


namespace inequality_system_two_integer_solutions_l794_794616

theorem inequality_system_two_integer_solutions (m : ℝ) : (-1 : ℝ) ≤ m ∧ m < 0 ↔ ∃ x : ℤ, (x < 1) ∧ (x > m - 1) ∧ {
  (∃ y : ℤ, (y < 1) ∧ (y > m - 1) ∧ x ≠ y)
  ∧ ∀ z : ℤ, (z < 1) ∧ (z > m - 1) → (z = x ∨ z = y)
}

end inequality_system_two_integer_solutions_l794_794616


namespace sum_of_every_second_term_l794_794854

theorem sum_of_every_second_term (x : ℕ → ℕ) (h1 : ∀ n, x (n + 1) = x n + 1) (h2 : (Finset.range 2010).sum x = 5307) : 
  (Finset.range 1005).sum (λ n, x (2 * n)) = 2151 := 
sorry

end sum_of_every_second_term_l794_794854


namespace larger_box_glasses_l794_794869

-- Definitions according to the conditions
def glasses_in_small_box := 12
def avg_glasses_per_box := 15

-- Using the given conditions to set up the problem
theorem larger_box_glasses : ∃ G : ℕ, (12 + G) / 2 = 15 ∧ G = 18 :=
by
  use 18
  split
  {
    -- Show that (12 + 18) / 2 = 15
    show (12 + 18) / 2 = 15
    calc
      (12 + 18) / 2 = 30 / 2 : by norm_num
      ... = 15 : by norm_num
  }
  -- Show that G = 18
  rfl

end larger_box_glasses_l794_794869


namespace simple_primes_finite_l794_794388

open Real

def is_simple_prime (p : ℕ) : Prop :=
  prime p ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ ⌊sqrt (p : ℝ)⌋.to_nat → (frac (p / k) ≥ 0.01)

theorem simple_primes_finite : set.finite {p : ℕ | is_simple_prime p} :=
sorry

end simple_primes_finite_l794_794388


namespace part1_part2_l794_794534

theorem part1 (x m : ℝ) (h_m : m = 4) (h_p : x^2 - 7 * x + 10 < 0) (h_q : (x - m) * (x - 3 * m) < 0) :
  4 < x ∧ x < 5 :=
begin
  sorry
end

theorem part2 (x m : ℝ) (h_neg_q : ¬((x - m) * (x - 3 * m) < 0) ↔ ¬(2 < x ∧ x < 5)) :
  (5 / 3) ≤ m ∧ m ≤ 2 :=
begin
  sorry
end

end part1_part2_l794_794534


namespace question1_question2_l794_794225

-- Definition of the set A as per the given condition
def A (a : ℝ) : set ℝ := {x | a * x ^ 2 - x + a + 2 = 0}

-- Proof Problem for Question 1
theorem question1 (a : ℝ) :
  (∃ x : ℝ, A a = {x}) → (a = 0 ∨ a = -2 + real.sqrt 5 ∨ a = -2 - real.sqrt 5) :=
sorry

-- Proof Problem for Question 2
theorem question2 (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ A a ∧ x2 ∈ A a → x1 = x2) ↔ (a < -2 - real.sqrt 5 ∨ a = 0 ∨ -2 + real.sqrt 5 ≤ a) :=
sorry

end question1_question2_l794_794225


namespace factorize_quadratic_l794_794149

theorem factorize_quadratic (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 :=
by
  sorry

end factorize_quadratic_l794_794149


namespace monotone_decreasing_sequence_monotone_increasing_sequence_l794_794180

theorem monotone_decreasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) < a n) ↔ c < 0 :=
by sorry

theorem monotone_increasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) > a n) ↔ c > 1/4 :=
by sorry

end monotone_decreasing_sequence_monotone_increasing_sequence_l794_794180


namespace line_MN_parallel_to_plane_PBC_l794_794866

-- Given definitions and conditions
variables {A B C D P M N : Type}
variables {V : Type} [inner_product_space ℝ V]

noncomputable def is_parallel_to_plane {A B C D P M N : V} : Prop :=
∃ (k l m : ℝ), k • (B - A) + l • (D - A) + m • (P - A) = 0

-- Lean statement to prove the equivalence
theorem line_MN_parallel_to_plane_PBC (P A B C D M N : V)
  (h1 : ∃ (k : ℝ), k ❨P - A)) = M)
  (h2 : ∃ (l : ℝ), l • (D - B) = N)
  (ratio1 : ∃ (k : ℝ), k = 5 / (5 + 8))
  (ratio2 : ∃ (l : ℝ), l = 5 / (5 + 8))
  : is_parallel_to_plane M N P :=
sorry

end line_MN_parallel_to_plane_PBC_l794_794866


namespace solution_set_f_inequality_l794_794934

/-- Define the piecewise function f(x) -/
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x
  else -x^2 + x

/-- Theorem to find the solution set of f(x^2 - x + 1) < 12 -/
theorem solution_set_f_inequality :
  {x : ℝ | f(x^2 - x + 1) < 12} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_set_f_inequality_l794_794934


namespace minimum_value_exists_l794_794141

noncomputable def min_value (a b c : ℝ) : ℝ :=
  a / (3 * b^2) + b / (4 * c^3) + c / (5 * a^4)

theorem minimum_value_exists :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → abc = 1 → min_value a b c ≥ 1 :=
by
  sorry

end minimum_value_exists_l794_794141


namespace part1_part2_l794_794412

theorem part1 (x y : ℝ) (h1 : x + 3 * y = 26) (h2 : 2 * x + y = 22) : x = 8 ∧ y = 6 :=
by
  sorry

theorem part2 (m : ℝ) (h : 8 * m + 6 * (15 - m) ≤ 100) : m ≤ 5 :=
by
  sorry

end part1_part2_l794_794412


namespace min_AC_plus_CB_l794_794552

/-- 
  Given points A(-2, -3) and B(5, 3) in the xy-plane; point C(2, m) is chosen so 
  that AC + CB is a minimum. Determine the value of m.

  The value of m that minimizes AC + CB is -3/7.
 -/
theorem min_AC_plus_CB (A B C : Real × Real) (hA : A = (-2, -3)) (hB : B = (5, 3)) 
  (hC : C = (2, m)) : 
  m = -3 / 7 :=
begin
  sorry
end

end min_AC_plus_CB_l794_794552


namespace common_ratio_is_two_l794_794202

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l794_794202


namespace simple_fraction_pow_l794_794133

theorem simple_fraction_pow : (66666^4 / 22222^4) = 81 := by
  sorry

end simple_fraction_pow_l794_794133


namespace roots_of_quadratic_form_arithmetic_and_geometric_sequence_l794_794597

theorem roots_of_quadratic_form_arithmetic_and_geometric_sequence
(p q a b : ℝ) (hp : p > 0) (hq : q > 0)
(h1 : a + b = p) (h2 : a * b = q) 
(h3 : (∃ c, c • a - 2 = b ∨ c • b - 2 = a) ∧ (a = 1 ∨ a = 4) ∧ (b = 1 ∨ b = 4)) :
  p + q = 9 :=
sorry

end roots_of_quadratic_form_arithmetic_and_geometric_sequence_l794_794597


namespace sequence_general_term_l794_794536

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), a 1 = 2 ^ (5 / 2) ∧ 
  (∀ n, a (n+1) = 4 * (4 * a n) ^ (1/4)) →
  ∀ n, a n = 2 ^ (10 / 3 * (1 - 1 / 4 ^ n)) :=
by
  intros a h1 h_rec
  sorry

end sequence_general_term_l794_794536


namespace foci_distance_of_hyperbola_l794_794154

-- The original problem's conditions are contained in the definitions below.
noncomputable def hyperbola_foci_distance : ℝ :=
  let a_squared : ℝ := (169 / 9)
  let b_squared : ℝ := (169 / 16)
  let c_squared : ℝ := a_squared + b_squared
  let c : ℝ := Real.sqrt c_squared
  2 * c

-- Lean theorem stating the distance between the foci of the hyperbola.
theorem foci_distance_of_hyperbola :
  hyperbola_foci_distance = 8.667 :=
by
  sorry

end foci_distance_of_hyperbola_l794_794154


namespace people_joined_after_leaving_l794_794449

theorem people_joined_after_leaving 
  (p_initial : ℕ) (p_left : ℕ) (p_final : ℕ) (p_joined : ℕ) :
  p_initial = 30 → p_left = 10 → p_final = 25 → p_joined = p_final - (p_initial - p_left) → p_joined = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end people_joined_after_leaving_l794_794449


namespace cos_alpha_minus_pi_over_six_l794_794955

theorem cos_alpha_minus_pi_over_six
  (α : ℝ)
  (h1 : π/2 < α) (h2 : α < π)
  (h3 : sin (α + π/6) = 3/5) :
  cos (α - π/6) = (3 * real.sqrt 3 - 4) / 10 := 
sorry

end cos_alpha_minus_pi_over_six_l794_794955


namespace jude_total_matchbox_vehicles_l794_794282

/-- Definition of variables based on the given conditions -/
def bottle_caps_for_car : ℕ := 5
def bottle_caps_for_truck : ℕ := 6
def total_bottle_caps : ℕ := 100
def trucks_bought : ℕ := 10
def rem_bottle_caps_fraction_for_cars : ℚ := 0.75

/-- Definition to calculate the total matchbox vehicles Jude buys -/
def total_matchbox_vehicles (bottle_caps_for_car : ℕ) (bottle_caps_for_truck : ℕ) (total_bottle_caps : ℕ) (trucks_bought : ℕ) (rem_bottle_caps_fraction_for_cars : ℚ) : ℕ :=
  let bottle_caps_spent_on_trucks := trucks_bought * bottle_caps_for_truck
  let remaining_bottle_caps := total_bottle_caps - bottle_caps_spent_on_trucks
  let bottle_caps_spent_on_cars := (rem_bottle_caps_fraction_for_cars * remaining_bottle_caps).to_nat
  let cars_bought := bottle_caps_spent_on_cars / bottle_caps_for_car
  trucks_bought + cars_bought

/-- Theorem to prove the total number of matchbox vehicles is 16 -/
theorem jude_total_matchbox_vehicles : total_matchbox_vehicles bottle_caps_for_car bottle_caps_for_truck total_bottle_caps trucks_bought rem_bottle_caps_fraction_for_cars = 16 :=
by sorry

end jude_total_matchbox_vehicles_l794_794282


namespace multiplication_result_l794_794125

theorem multiplication_result :
  10 * 9.99 * 0.999 * 100 = (99.9)^2 := 
by
  sorry

end multiplication_result_l794_794125


namespace complement_of_intersection_eq_l794_794230

-- Definitions of sets with given conditions
def U : Set ℝ := {x | 0 ≤ x ∧ x < 10}
def A : Set ℝ := {x | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Complement of a set with respect to U
def complement_U (S : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ S}

-- Intersect two sets
def intersection (S1 S2 : Set ℝ) : Set ℝ := {x | x ∈ S1 ∧ x ∈ S2}

theorem complement_of_intersection_eq :
  complement_U (intersection A B) = {x | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := 
by
  sorry

end complement_of_intersection_eq_l794_794230


namespace prove_number_of_functions_l794_794921

noncomputable def number_of_functions (a b c : ℤ) : ℕ :=
if g_xg_negx_eq_g_x3 a b c then
  if a = 0 ∨ a = -1 then
    if b = 0 ∨ b = 1 then
      if c = 0 ∨ c = 1 then 1 else 0
    else 0
  else 0
else 0

def g_xg_negx_eq_g_x3 (a b c: ℤ) : Prop :=
let g_x := a * (x : ℤ) ^ 3 + b * x + c in
let g_negx := a * (-x) ^ 3 + b * -x + c in
let g_x3 := a * x ^ 9 + b * x ^ 3 + c in
g_x * g_negx = g_x3

theorem prove_number_of_functions : ∑ a in {-1, 0}, ∑ b in {0, 1}, ∑ c in {0, 1}, 
  if g_xg_negx_eq_g_x3 a b c then 1 else 0 = 8 :=
by sorry

end prove_number_of_functions_l794_794921


namespace part1_part2_l794_794763

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x y : ℝ, f (x - y) = f x / f y
axiom h2 : ∀ x : ℝ, f x > 0
axiom h3 : ∀ x y : ℝ, x < y → f x > f y

-- First part: f(0) = 1 and proving f(x + y) = f(x) * f(y)
theorem part1 : f 0 = 1 ∧ (∀ x y : ℝ, f (x + y) = f x * f y) :=
sorry

-- Second part: Given f(-1) = 3, solve the inequality
axiom h4 : f (-1) = 3

theorem part2 : {x : ℝ | (x ≤ 3) ∨ (x ≥ 4)} = {x : ℝ | f (x^2 - 7*x + 10) ≤ f (-2)} :=
sorry

end part1_part2_l794_794763


namespace AM_plus_AL_ge_2AN_l794_794789

open EuclideanGeometry

variables {A B C F M L N : Point}
variables (circumcircle_ABC : Circle) (triangle_ABC : Triangle)

-- Assume the conditions given in the problem.
axiom tangents_intersect_at_F :
  tangent_line circumcircle_ABC B ∩ tangent_line circumcircle_ABC C = {F}

axiom feet_of_perpendiculars :
  (foot_of_perpendicular A (line FB) = M) ∧
  (foot_of_perpendicular A (line FC) = L) ∧
  (foot_of_perpendicular A (line BC) = N)

-- The theorem to be proved.
theorem AM_plus_AL_ge_2AN
  (AM : length (line_segment A M))
  (AL : length (line_segment A L))
  (AN : length (line_segment A N)) :
  AM + AL ≥ 2 * AN :=
by sorry

end AM_plus_AL_ge_2AN_l794_794789


namespace solution_l794_794691

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l794_794691


namespace value_of_a_l794_794253

theorem value_of_a (a : ℝ) 
  (hP : ∃ α : ℝ, sin α * cos α = √3 / 4 ∧ sin α = a / real.sqrt (16 + a^2) ∧ cos α = -4 / real.sqrt (16 + a^2)) : 
  a = -4 * √3 ∨ a = - (4 / 3) * √3 :=
by
  sorry

end value_of_a_l794_794253


namespace graph_shift_l794_794349

noncomputable def f (x : ℝ) : ℝ := sin x + (sqrt 3 * cos x)
noncomputable def g (x : ℝ) : ℝ := sin x - (sqrt 3 * cos x)

theorem graph_shift :
  ∀ (x : ℝ), f(x - (2 * π / 3)) = g(x) := 
by
  sorry

end graph_shift_l794_794349


namespace ellipse_condition_l794_794771

theorem ellipse_condition (m : ℝ) : 
  (4 + m > 0) ∧ (2 - m > 0) ∧ (4 + m ≠ 2 - m) → m ∈ set.Ioo (-4 : ℝ) (-1) ∪ set.Ioo (-1) 2 :=
sorry

end ellipse_condition_l794_794771


namespace problem_cosine_of_angle_l794_794066

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_from_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def cos_angle (v1 v2 : Point3D) : ℝ :=
  (dot_product v1 v2) / (magnitude v1 * magnitude v2)

noncomputable def validate_cosine : Prop :=
  let A := { x := 7, y := 0, z := 2 }
  let B := { x := 7, y := 1, z := 3 }
  let C := { x := 8, y := -1, z := 2 }
  let AB := vector_from_points A B
  let AC := vector_from_points A C
  cos_angle AB AC = -1 / 2

theorem problem_cosine_of_angle : validate_cosine := by
  sorry

end problem_cosine_of_angle_l794_794066


namespace exists_subset_with_property_P_l794_794704

open Nat

-- Define the property P
def has_property_P (A : Set ℕ) : Prop :=
  ∃ (m : ℕ), ∀ (k : ℕ), k > 0 → ∃ (a : Fin k → ℕ), (∀ j < k-1, 1 ≤ a j.succ - a j ∧ a j.succ - a j ≤ m) ∧ (∀ j < k, a j ∈ A)

-- Main theorem statement
theorem exists_subset_with_property_P (N : Set ℕ) (r : ℕ) (A : Fin r → Set ℕ)
  (h_disjoint : ∀ i j, i ≠ j → Disjoint (A i) (A j))
  (h_union : (⋃ i, A i) = N) :
  ∃ i, has_property_P (A i) :=
sorry

end exists_subset_with_property_P_l794_794704


namespace choose_coPresidents_l794_794414

-- Define the number of members in the club
def members : ℕ := 15

-- Define the number of co-presidents to choose
def coPresidents : ℕ := 2

-- State the theorem
theorem choose_coPresidents (members coPresidents : ℕ) (h_mem : members = 15) (h_coP : coPresidents = 2) :
  (members.choose coPresidents) = 105 := 
by {
  rw [h_mem, h_coP],
  sorry
}

end choose_coPresidents_l794_794414


namespace Juan_friends_count_l794_794668

theorem Juan_friends_count :
  ∃ (n : ℕ), (2 * factorial (n - 1) = 48) ∧ (n = 5) :=
by
  sorry

end Juan_friends_count_l794_794668


namespace find_a_l794_794563

theorem find_a (a : ℝ) : (4, -5).2 = (a - 2, a + 1).2 → a = -6 :=
by
  intro h
  sorry

end find_a_l794_794563


namespace cube_surface_area_increase_l794_794048

-- Given definitions
variables (L : ℝ) 

-- Define original surface area
def SA_original : ℝ := 6 * L^2

-- Define new edge length after growth
def L_new : ℝ := 1.20 * L

-- Define new surface area
def SA_new : ℝ := 6 * (1.20 * L)^2

-- Define percentage increase function
def percentage_increase (original new : ℝ) : ℝ := ((new - original) / original) * 100

-- Statement to prove 
theorem cube_surface_area_increase : percentage_increase (SA_original L) (SA_new L) = 44 := by
  -- proof should be filled in here
  sorry

end cube_surface_area_increase_l794_794048


namespace max_y_l794_794571

variable {α β m x y : ℝ}

-- Assuming α and β are acute angles
variable h_proper_angles : ∃ α β, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 

-- Given conditions
variable (h_cond : sin β = m * cos (α + β) * sin α)
variable (h_m_pos : 0 < m)
variable (h_alpha_beta_not_pi_over_2 : α + β ≠ π / 2)

-- Definitions as given
def tan_alpha : ℝ := tan α
def tan_beta : ℝ := tan β

-- Expression of y in terms of x
def f (x : ℝ) := m * x / (1 + (m + 1) * x^2)

-- Maximum value of function y under the given range for α
theorem max_y (h_range : π/4 ≤ α ∧ α < π/2) :
  y = tan β → y = f (tan α) ∧ ∀ x ≥ (tan (π/4)), f x ≤ m / (m + 2) :=
sorry

end max_y_l794_794571


namespace team_arrangements_l794_794633

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem team_arrangements :
  let num_players := 10
  let team_blocks := 4
  let cubs_players := 3
  let red_sox_players := 3
  let yankees_players := 2
  let dodgers_players := 2
  (factorial team_blocks) * (factorial cubs_players) * (factorial red_sox_players) * (factorial yankees_players) * (factorial dodgers_players) = 3456 := 
by
  -- Proof steps will be inserted here
  sorry

end team_arrangements_l794_794633


namespace find_a_l794_794229

theorem find_a (a : ℝ) (U A CU: Set ℝ) (hU : U = {2, 3, a^2 - a - 1}) (hA : A = {2, 3}) (hCU : CU = {1}) (hComplement : CU = U \ A) :
  a = -1 ∨ a = 2 :=
by
  sorry

end find_a_l794_794229


namespace silvia_shorter_jerry_l794_794665

def jerry_distance : ℕ := 3 + 4
def silvia_distance : ℕ := Int.natAbs (Int.sqrt (3^2 + 4^2))

def percentage_reduction (j s : ℤ) : ℤ :=
  ((j - s) * 100) / j

theorem silvia_shorter_jerry :
  percentage_reduction jerry_distance silvia_distance ≈ 40 := sorry

end silvia_shorter_jerry_l794_794665


namespace largest_base5_number_conversion_l794_794011

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794011


namespace square_area_ABC_D_is_36_l794_794070

-- The conditions given in the problem
variables (A B C D : Type) [square A B C D]
variables (S1 S2 : Type) [square S1] [square S2]
variables (area_S1 : sq_area S1 = 4) (area_S2 : sq_area S2 = 16)

-- The goal to be proved
theorem square_area_ABC_D_is_36 : sq_area A = 36 := 
sorry

end square_area_ABC_D_is_36_l794_794070


namespace sequence_no_repetitions_l794_794371

-- A sequence of integers greater than 1 with no repetitions
noncomputable def a : ℕ → ℤ := sorry

theorem sequence_no_repetitions :
  (∀ n, a n > 1) ∧ (∀ m n, m ≠ n → a m ≠ a n) →
  ∃^∞ n, a n > n :=
by
  intros h
  sorry

end sequence_no_repetitions_l794_794371


namespace percentage_invalid_l794_794262

theorem percentage_invalid (total_votes valid_votes_A : ℕ) (percent_A : ℝ) (total_valid_votes : ℝ) (percent_invalid : ℝ) :
  total_votes = 560000 →
  valid_votes_A = 333200 →
  percent_A = 0.70 →
  (1 - percent_invalid / 100) * total_votes = total_valid_votes →
  percent_A * total_valid_votes = valid_votes_A →
  percent_invalid = 15 :=
by
  intros h_total_votes h_valid_votes_A h_percent_A h_total_valid_votes h_valid_poll_A
  sorry

end percentage_invalid_l794_794262


namespace solve_system_of_equations_l794_794187

theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x - 2 * y = 1) (h2 : x + y = 2) : x^2 - 2 * y^2 = -1 :=
by
  sorry

end solve_system_of_equations_l794_794187


namespace speaker_is_tweedledee_l794_794450

-- Definitions
variable (Speaks : Prop) (is_tweedledum : Prop) (has_black_card : Prop)

-- Condition: If the speaker is Tweedledum, then the card in the speaker's pocket is not a black suit.
axiom A1 : is_tweedledum → ¬ has_black_card

-- Goal: Prove that the speaker is Tweedledee.
theorem speaker_is_tweedledee (h1 : Speaks) : ¬ is_tweedledum :=
by
  sorry

end speaker_is_tweedledee_l794_794450


namespace digit_one_occurrence_l794_794477

/--
Consider the sum N of the following series of positive integers:
1. \( N = \sum_{k=1}^{2018} (10^k - 1) \)
    Show that the digit 1 occurs 2014 times in the decimal representation of N.
-/
theorem digit_one_occurrence :
  let N := ∑ k in Finset.range 2018, (10^(k+1) - 1)
  in (count_digits 1 N) = 2014 :=
sorry

end digit_one_occurrence_l794_794477


namespace parallel_vectors_x_value_l794_794530

theorem parallel_vectors_x_value (x : ℝ) :
  (∀ k : ℝ, k ≠ 0 → (4, 2) = (k * x, k * (-3))) → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l794_794530


namespace quadratic_inequality_condition_l794_794770

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 3 := 
sorry

end quadratic_inequality_condition_l794_794770


namespace f_cos_x_l794_794748

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_cos_x (T : ℝ) (n : ℕ) (hT : T = 2 * π * n)
  (h1 : ∀ x : ℝ, cos x = f x - 3 * f (x - π))
  (h2 : ∀ x : ℝ, cos (x - T) = f (x - T) - 3 * f (x - T - π)) :
  f x = 1/4 * cos x :=
by 
  sorry

end f_cos_x_l794_794748


namespace first_pack_weight_l794_794845

variable (initial_supplies_per_mile : Float) (resupply_percentage : Float) (hiking_rate : Float) (hiking_hours_per_day : Float) (hiking_days : Float)

#check initial_supplies_per_mile
#check resupply_percentage
#check hiking_rate
#check hiking_hours_per_day
#check hiking_days

theorem first_pack_weight 
    (hiking_rate = 2.5)
    (hiking_hours_per_day = 8)
    (hiking_days = 5)
    (initial_supplies_per_mile = 0.5)
    (resupply_percentage = 0.25) :
    (let total_distance := hiking_rate * hiking_hours_per_day * hiking_days in
     let total_supplies := total_distance * initial_supplies_per_mile in
     let resupply_weight := total_supplies * resupply_percentage in
     let total_after_resupply := total_supplies + resupply_weight in
     let first_pack_weight := total_after_resupply - resupply_weight in 
     first_pack_weight = 50) :=
begin
  sorry,
end

end first_pack_weight_l794_794845


namespace six_point_four_five_minus_six_point_four_five_star_l794_794172

def z : ℝ := 6.45

def z_star (y : ℝ) : ℝ := 
  if y ≥ 6 then 6
  else if y ≥ 4 then 4
  else if y ≥ 2 then 2
  else 0  -- assuming y ≥ 0 without loss of generality for positive even integers

theorem six_point_four_five_minus_six_point_four_five_star : z - z_star z = 0.45 :=
by
  let z := 6.45
  have z_str := z_star z
  have h : z_star 6.45 = 6 := by simp [z_star]
  rw h
  exact (by norm_num : 6.45 - 6 = 0.45)

end six_point_four_five_minus_six_point_four_five_star_l794_794172


namespace polynomial_form_l794_794912

def is_even_poly (P : ℝ → ℝ) : Prop := 
  ∀ x, P x = P (-x)

theorem polynomial_form (P : ℝ → ℝ) (hP : ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) : 
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x ^ 4 + b * x ^ 2 := 
  sorry

end polynomial_form_l794_794912


namespace distance_xiao_ming_walked_l794_794401

theorem distance_xiao_ming_walked (speed_bin speed_ming distance_to_school time_meet : ℕ) 
  (h0 : speed_bin = 15) 
  (h1 : speed_ming = 5) 
  (h2 : distance_to_school = 30) 
  (h3 : time_meet = 3) 
  : speed_ming * time_meet = 15 :=
by {
  -- Given speed_bin = 15 and speed_ming = 5
  have h_bin_speed : speed_bin = 15, from h0,
  have h_ming_speed : speed_ming = 5, from h1,
  -- Calculate the total distance when they meet based on the given speeds and time
  let total_distance_when_meet := speed_bin * time_meet + speed_ming * time_meet,
  -- Total distance must be twice the distance to school, so we calculate time_meet
  have h_total_distance : total_distance_when_meet = 60, from sorry,
  -- Given time_meet = 3 hours
  have h_time_meet : time_meet = 3, from h3,
  -- Calculate the distance walked by Xiao Ming
  have h_distance_ming_walked : speed_ming * time_meet = 15,
  {
    rw [h_ming_speed, h_time_meet],
    calc 5 * 3 = 15 : rfl,
  },
  exact h_distance_ming_walked,
}

end distance_xiao_ming_walked_l794_794401


namespace height_on_hypotenuse_of_right_triangle_l794_794618

theorem height_on_hypotenuse_of_right_triangle (a b : ℝ) (h_a : a = 2) (h_b : b = 3) :
  ∃ h : ℝ, h = (6 * Real.sqrt 13) / 13 :=
by
  sorry

end height_on_hypotenuse_of_right_triangle_l794_794618


namespace parallelogram_slope_sum_l794_794341

noncomputable def slope_AB := (22 - 0) / (4 - 0)
def slope_CD_exists (x y : ℤ) : Prop :=
  (x = 0 ∧ y = 0) ∨ (4 + x ∈ ℤ ∧ 22 + y ∈ ℤ)

theorem parallelogram_slope_sum : slope_AB = 11 / 2 → ∑ (x y : ℤ) in {p : ℤ × ℤ | slope_CD_exists p.1 p.2}, abs ((22 + y) / (4 + x)) = 11 / 2 → 11 + 2 = 13 :=
by
  sorry

end parallelogram_slope_sum_l794_794341


namespace cube_inequality_l794_794189

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l794_794189


namespace part1_part2_part3_l794_794543

-- Define the sequence according to the given conditions.
def a : ℕ → ℕ
| 1 := 0
| 2 := 2
-- Recursive definition does not directly capture the condition but acts as a placeholder.
| n := sorry

-- Define the condition for the sequence given in the problem.
axiom sequence_condition : ∀ m n : ℕ,
  m > 0 ∧ n > 0 →
  a (2 * m - 1) + a (2 * n - 1) = 2 * a (m + n - 1) + 2 * (m - n)^2

-- Prove the value of a_3, a_4, and a_5
theorem part1 : a 3 = 6 ∧ a 4 = 12 ∧ a 5 = 20 :=
  sorry

-- Prove the general formula for the sequence
theorem part2 : ∀ n, a n = n * (n - 1) :=
  sorry

-- Prove the non-existence of distinct positive integers p, q, r forming an arithmetic sequence and a_p, a_q, a_r also forming an arithmetic sequence
theorem part3 : ¬ ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (q = (p + r) / 2) ∧ 
  (a p + a r = 2 * a q) :=
  sorry

end part1_part2_part3_l794_794543


namespace find_second_discount_l794_794779

-- Define the conditions
def original_price : ℝ := 400
def first_discount_percent : ℝ := 25
def first_discount_amount : ℝ := original_price * first_discount_percent / 100
def price_after_first_discount : ℝ := original_price - first_discount_amount
def final_price : ℝ := 240
def second_discount_amount (D : ℝ) : ℝ := (D / 100) * price_after_first_discount

-- The statement to prove the second discount percentage
theorem find_second_discount : ∃ D : ℝ, price_after_first_discount - second_discount_amount D = final_price ∧ D = 20 :=
by
  sorry

end find_second_discount_l794_794779


namespace smallest_n_for_rotation_matrix_l794_794925

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem smallest_n_for_rotation_matrix :
  let A := rotation_matrix (135 * Real.pi / 180) in
  ∃ (n : ℕ), n > 0 ∧ A^n = 1 ∧ ∀ (m : ℕ), m > 0 ∧ A^m = 1 → n ≤ m :=
  sorry

end smallest_n_for_rotation_matrix_l794_794925


namespace bruce_money_left_l794_794118

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l794_794118


namespace intersecting_lines_parallel_plane_l794_794943

-- Defining the geometric elements
variable (Δ : Plane) (a b : Line) (m : ℝ)

-- Define the planes parallel to Δ at a distance m
def Δ1 : Plane := sorry -- Definition of Δ1
def Δ2 : Plane := sorry -- Definition of Δ2

-- Coordinates of intersection points on the planes Δ1 and Δ2
def A1 : Point := sorry -- Intersection of a with Δ1
def B1 : Point := sorry -- Intersection of b with Δ1
def A2 : Point := sorry -- Intersection of a with Δ2
def B2 : Point := sorry -- Intersection of b with Δ2

-- Define the lines connecting the intersection points
def L : Line := Line_through A1 B1
def M : Line := Line_through A2 B2

theorem intersecting_lines_parallel_plane :
  (L.parallel_to Δ ∧ L.distance_from_plane Δ = m ∧ L.intersects a ∧ L.intersects b) ∧
  (M.parallel_to Δ ∧ M.distance_from_plane Δ = m ∧ M.intersects a ∧ M.intersects b) := sorry

end intersecting_lines_parallel_plane_l794_794943


namespace find_lambda_div_mu_l794_794566

variable {V : Type*} [InnerProductSpace ℝ V]

-- Conditions of the problem

-- The angle between vectors AB and AC is 90 degrees, hence their dot product is zero
variable (A B C M : V) (λ μ : ℝ)
variables (h1 : ⟪B - A, C - A⟫ = 0)

-- Length of AB is 2
variables (h2 : ∥B - A∥ = 2)

-- AM is a linear combination of AB and AC
variables (h3 : M - A = λ • (B - A) + μ • (C - A))

-- AM is perpendicular to BC
variables (h4 : ⟪M - A, C - B⟫ = 0)

-- Result to prove
theorem find_lambda_div_mu : λ / μ = 1 / 4 := 
by 
  sorry

end find_lambda_div_mu_l794_794566


namespace algebraic_expression_value_l794_794177

-- Define the equation and its roots.
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 = 0

def is_root (x : ℝ) : Prop := quadratic_eq x

-- The main theorem.
theorem algebraic_expression_value (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) :
  (x1 + x2) / (1 + x1 * x2) = 1 :=
sorry

end algebraic_expression_value_l794_794177


namespace swim_club_members_l794_794063

theorem swim_club_members (X : ℝ) 
  (h1 : 0.30 * X = 0.30 * X)
  (h2 : 0.70 * X = 42) : X = 60 :=
sorry

end swim_club_members_l794_794063


namespace sum_arithmetic_sequence_l794_794183

theorem sum_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 4 = 4)
    (ha : ∀ n, a n = 1 + (n - 1) * 1) :
    (∑ i in range (n+1), a i) = 2 * n^2 - n :=
by
  sorry

end sum_arithmetic_sequence_l794_794183


namespace tax_rate_l794_794485

theorem tax_rate (total_spent tax_free cost_of_tax: ℝ) (h1 : total_spent = 40)
    (h2 : tax_free = 34.7) (h3 : cost_of_tax = 0.30) :
    let taxable_items := total_spent - tax_free in
    let tax_rate := (cost_of_tax * 100) / taxable_items in
    tax_rate ≈ 5.66 :=
by
  let taxable_items := total_spent - tax_free
  let tax_rate := (cost_of_tax * 100) / taxable_items
  have proof_tax_rate : tax_rate ≈ 5.66 := sorry 
  exact proof_tax_rate

end tax_rate_l794_794485


namespace isosceles_triangle_side_lengths_l794_794865

theorem isosceles_triangle_side_lengths {A B C P : Type*} (s t : ℝ) :
  (triangle.is_isosceles A B C) ∧ (dist A P = 2) ∧ (dist B P = sqrt 5) ∧ (dist C P = 3) →
  s = 2 * sqrt 3 ∧ t = sqrt 5 := by
sorry

end isosceles_triangle_side_lengths_l794_794865


namespace third_square_in_sequence_is_G_l794_794492

theorem third_square_in_sequence_is_G
  (squares : list string)
  (order : squares = ["F", "H", "G", "D", "A", "B", "C", "E"]) :
  squares.nth 2 = some "G" :=
by {
  rw order,
  simp,
  refl
}

end third_square_in_sequence_is_G_l794_794492


namespace range_of_a_l794_794222

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) → a ≤ -1 :=
sorry

end range_of_a_l794_794222


namespace XiaoMing_reading_problem_l794_794057

theorem XiaoMing_reading_problem :
  ∀ (total_pages days first_days first_rate remaining_rate : ℕ),
    total_pages = 72 →
    days = 10 →
    first_days = 2 →
    first_rate = 5 →
    (first_days * first_rate) + ((days - first_days) * remaining_rate) ≥ total_pages →
    remaining_rate ≥ 8 :=
by
  intros total_pages days first_days first_rate remaining_rate
  intro h1 h2 h3 h4 h5
  sorry

end XiaoMing_reading_problem_l794_794057


namespace moles_NaOH_combined_eq_2_l794_794922

theorem moles_NaOH_combined_eq_2 :
  ∀ (n: ℕ) , (n : ℕ) = 2 ↔ by reaction_produces_H2O 2 → moles_of_NaOH n := sorry

end moles_NaOH_combined_eq_2_l794_794922


namespace collinear_O_N_M_iff_equality_l794_794948

variable (O N M A1 A2 A3 A4 B1 B2 B3 B4 : Point)
variable (O_A1 : Line O A1) (O_A2 : Line O A2) (O_A3 : Line O A3) (O_A4 : Line O A4)
variable (O_B1 : Line O B1) (O_B2 : Line O B2) (O_B3 : Line O B3) (O_B4 : Line O B4)
variable (A1_B1 : Line A1 B1) (A2_B2 : Line A2 B2) (A3_B3 : Line A3 B3) (A4_B4 : Line A4 B4)
variable (N_on_A1_B1_A2_B2 : IntersectedAt A1_B1 A2_B2 N)
variable (M_on_A3_B3_A4_B4 : IntersectedAt A3_B3 A4_B4 M)

theorem collinear_O_N_M_iff_equality :
  Collinear O N M ↔
  (dist O B1 / dist O B3 * dist O B2 / dist O B4 * dist B3 B4 / dist B1 B2 =
  dist O A1 / dist O A3 * dist O A2 / dist O A4 * dist A3 A4 / dist A1 A2) :=
sorry

end collinear_O_N_M_iff_equality_l794_794948


namespace smallest_set_size_l794_794337

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

theorem smallest_set_size (n : ℕ) (hn : n ≥ 2) :
  ∃ S : set ℤ, (∀ k, 2 ≤ k → k ≤ n → ∃ x y ∈ S, x - y = fib k) ∧ S.card = (n / 2).ceil + 1 :=
sorry

end smallest_set_size_l794_794337


namespace part_a_part_b_l794_794726

-- Definitions and conditions
variables {A B C D K P Q R : Type} [incircle : ∀ {A B C D}, is_cyclic_quad A B C D]
variables [tangents : ∀ {B D K}, is_tangent B K D] [linear : lies_on K A C]

-- Part (a): Prove that AB · CD = BC · AD
theorem part_a :
  ∀ (AB BC CD AD : ℝ), AB * CD = BC * AD :=
sorry

-- Definitions for Part (b)
variables [parallel : Parallel C K P]
variables [intersects : Intersects B A P B D Q B C R]

-- Part (b): Prove that PQ = QR
theorem part_b {PQ QR : ℝ} :
  PQ = QR :=
sorry

end part_a_part_b_l794_794726


namespace largest_base_5_five_digit_number_in_decimal_l794_794028

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794028


namespace range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l794_794186

-- Question 1
def prop_p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def prop_q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0
def neg_p (x : ℝ) : Prop := ¬ prop_p x
def neg_q (x m : ℝ) : Prop := ¬ prop_q x m

theorem range_m_if_neg_p_implies_neg_q : 
  (∀ x, neg_p x → neg_q x m) → 0 < m ∧ m ≤ 1 :=
by
  sorry

-- Question 2
theorem range_x_if_m_is_5_and_p_or_q_true_p_and_q_false : 
  (∀ x, (prop_p x ∨ prop_q x 5) ∧ ¬ (prop_p x ∧ prop_q x 5)) → 
  ∀ x, (x = 5 ∨ (-5 ≤ x ∧ x < -1)) :=
by
  sorry

end range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l794_794186


namespace max_possible_value_l794_794274

-- Define the expressions and the conditions
def expr1 := 10 * 10
def expr2 := 10 / 10
def expr3 := expr1 + 10
def expr4 := expr3 - expr2

-- Define our main statement that asserts the maximum value is 109
theorem max_possible_value: expr4 = 109 := by
  sorry

end max_possible_value_l794_794274


namespace problem1_solution_problem2_solution_l794_794458

theorem problem1_solution (x : ℝ) : (x^2 - 4 * x = 5) → (x = 5 ∨ x = -1) :=
by sorry

theorem problem2_solution (x : ℝ) : (2 * x^2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1/2) :=
by sorry

end problem1_solution_problem2_solution_l794_794458


namespace arccos_zero_l794_794473

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l794_794473


namespace length_of_BD_l794_794113

noncomputable def points_on_circle (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ) : Prop :=
  BC = 4 ∧ CD = 4 ∧ AE = 6 ∧ (0 < y) ∧ (0 < z) ∧ (AE * 2 = y * z) ∧ (8 > y + z)

theorem length_of_BD (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ)
  (h : points_on_circle A B C D E BD AE BC CD y z) : 
  BD = 7 :=
by
  sorry

end length_of_BD_l794_794113


namespace find_p_q_l794_794696

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l794_794696


namespace interval_of_decrease_log_function_l794_794894

noncomputable def loga (a u : ℝ) : ℝ := Real.log u / Real.log a

theorem interval_of_decrease_log_function {a : ℝ} (h_a : 0 < a ∧ a < 1) :
    Ioo 1 +∞ = { x : ℝ | ∀ y : ℝ, y = loga a (2 * x^2 - 3 * x + 1) → y < 0 ↔ x ∈ Ioo 1 +∞ } :=
sorry

end interval_of_decrease_log_function_l794_794894


namespace number_of_lines_through_point_intersecting_parabola_at_one_point_l794_794844

theorem number_of_lines_through_point_intersecting_parabola_at_one_point :
  let M := (2, 4)
  let parabola := set_of (λ p : ℝ × ℝ, p.2^2 = 8 * p.1)
  ∀ line : set (ℝ × ℝ), M ∈ line ∧ (∃! P ∈ parabola, P ∈ line) → 
  (line = tangent_to_parabola_at M parabola ∨ line = axis_parallel_line_through M parabola) → 
  2 :=
by 
  sorry

end number_of_lines_through_point_intersecting_parabola_at_one_point_l794_794844


namespace max_pawns_on_chessboard_l794_794043

-- Define the chessboard squares
inductive Square : Type
| a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8
| b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8
| d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8
| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8
| g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8
| h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8

open Square

-- Define the symmetry relative to e4
def symmetric (s1 s2 : Square) : Prop := 
  s1 ≠ e4 ∧ s2 ≠ e4 ∧ (s1, s2) ∈ [{ (a1, h8), (a2, h7), (a3, h6), (a4, h5), (a5, h4), (a6, h3), (a7, h2), (a8, h1),
                                      (b1, g8), (b2, g7), (b3, g6), (b4, g5), (b5, g4), (b6, g3), (b7, g2), (b8, g1),
                                      (c1, f8), (c2, f7), (c3, f6), (c4, f5), (c5, f4), (c6, f3), (c7, f2), (c8, f1),
                                      (d1, e8), (d2, e7), (d3, e6), (d4, e5), (d5, e4), (d6, e3), (d7, e2), (d8, e1),
                                      (e1, d8), (e2, d7), (e3, d6), (e5, d4), (e6, d3), (e7, d2), (e8, d1),
                                      (f1, c8), (f2, c7), (f3, c6), (f4, c5), (f5, c4), (f6, c3), (f7, c2), (f8, c1),
                                      (g1, b8), (g2, b7), (g3, b6), (g4, b5), (g5, b4), (g6, b3), (g7, b2), (g8, b1),
                                      (h1, a8), (h2, a7), (h3, a6), (h4, a5), (h5, a4), (h6, a3), (h7, a2), (h8, a1) }]

-- Prove the maximum number of pawns
theorem max_pawns_on_chessboard : ∀ (pawns : finset Square),
  (∀ s ∈ pawns, s ≠ e4) →
  (∀ s1 s2 ∈ pawns, s1 ≠ s2 → ¬symmetric s1 s2) →
  pawns.card ≤ 39 :=
by sorry

end max_pawns_on_chessboard_l794_794043


namespace susan_change_sum_susan_possible_sums_l794_794334

theorem susan_change_sum
  (change : ℕ)
  (h_lt_100 : change < 100)
  (h_nickels : ∃ k : ℕ, change = 5 * k + 2)
  (h_quarters : ∃ m : ℕ, change = 25 * m + 5) :
  change = 30 ∨ change = 55 ∨ change = 80 :=
sorry

theorem susan_possible_sums :
  30 + 55 + 80 = 165 :=
by norm_num

end susan_change_sum_susan_possible_sums_l794_794334


namespace sum_of_infinite_areas_l794_794144

noncomputable def sum_of_circle_areas : ℝ :=
  let r : ℕ → ℝ := λ n, 2 * (1 / 2^(n - 1))
  let A : ℕ → ℝ := λ n, π * (r n)^2
  let series_sum : ℝ := ∑' n, A n
  in series_sum

theorem sum_of_infinite_areas :
  sum_of_circle_areas = (16 * π) / 3 :=
sorry

end sum_of_infinite_areas_l794_794144


namespace expression_never_prime_l794_794900

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (n : ℕ) (h : is_prime n) : ¬is_prime (n^2 + 75) :=
sorry

end expression_never_prime_l794_794900


namespace rhombus_area_l794_794725

def is_rhombus (A B C D : Point) : Prop := 
  -- Placeholder: Define the property of a rhombus, all sides equal and opposite angles equal
  sorry

def perimeter (ABCD : Quadrilateral) : ℝ :=
  -- Placeholder: Define the perimeter of the quadrilateral
  sorry

def diagonal_length (A C : Point) : ℝ :=
  -- Placeholder: Define the length of the diagonal AC
  sorry

theorem rhombus_area (A B C D : Point) (P : is_rhombus A B C D) 
  (h_perimeter : perimeter ⟨A, B, C, D⟩ = 40) 
  (h_diag_AC : diagonal_length A C = 16) : 
  area ⟨A, B, C, D⟩ = 96 := 
by
  sorry

end rhombus_area_l794_794725


namespace probability_point_above_parabola_l794_794744

theorem probability_point_above_parabola : 
  (∑ a b, if b > a + a * a then 1 else 0).toRat / (9 * 9) = 23 / 27 :=
sorry

end probability_point_above_parabola_l794_794744


namespace domain_of_f_f_of_minus_one_f_of_twelve_l794_794214

def f (x : ℝ) : ℝ := (6 / (x - 1)) - (Real.sqrt (x + 4))

theorem domain_of_f :
  {x : ℝ | x ≠ 1 ∧ x ≥ -4} = {x : ℝ | (∃ r, f r = (6 / (r - 1)) - (Real.sqrt (r + 4))) } :=
by sorry

theorem f_of_minus_one : f (-1) = -3 - Real.sqrt 3 :=
by sorry

theorem f_of_twelve : f 12 = -38 / 11 :=
by sorry

end domain_of_f_f_of_minus_one_f_of_twelve_l794_794214


namespace probability_of_selection_l794_794629

-- defining necessary parameters and the systematic sampling method
def total_students : ℕ := 52
def selected_students : ℕ := 10
def exclusion_probability := 2 / total_students
def inclusion_probability_exclude := selected_students / (total_students - 2)
def final_probability := (1 - exclusion_probability) * inclusion_probability_exclude

-- the main theorem stating the probability calculation
theorem probability_of_selection :
  final_probability = 5 / 26 :=
by
  -- we skip the proof part and end with sorry since it is not required
  sorry

end probability_of_selection_l794_794629


namespace find_PQ_l794_794949

noncomputable def obtuse_triangle (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  ∀ (AC CP PQ : ℝ) (angle_ACP angle_CPQ : ℝ),
    AC = 25 → CP = 20 → angle_ACP = 90 → angle_CPQ = 90 → 
    ∃ PQ, PQ = 16

theorem find_PQ :
  ∀ (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C],
  obtuse_triangle A B C P Q 25 20 16 90 90 :=
begin
  sorry,
end

end find_PQ_l794_794949


namespace chess_tournament_problem_l794_794626

theorem chess_tournament_problem :
  ∀ (M : Type) (masters : finset M) (days : finset ℕ) 
    (played : M → M → ℕ → Prop),
    (masters.card = 8) →
    (days.card = 7) →
    (∀ m1 m2 : M, m1 ≠ m2 → ∃! d ∈ days, played m1 m2 d) →
    (∀ d ∈ days, (∃ t : finset (M × M), t.card = 4 ∧ (∀ ⟨m1, m2⟩ ∈ t, m1 ∈ masters ∧ m2 ∈ masters ∧ played m1 m2 d))) →
    ∃ S : finset M, S.card ≥ 4 ∧ (∀ m1 m2 ∈ S, ∃ d ∈ (days.filter (≤ 5)), played m1 m2 d) :=
begin
  sorry

end chess_tournament_problem_l794_794626


namespace math_problem_solution_l794_794546

noncomputable def ellipse_equation_and_eccentricity : Prop :=
  ∃ (a b: ℝ), a = 2 ∧ b = 1 ∧ (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1) ∧ a > b ∧ b > 0 ∧ 
    (let c := real.sqrt (a^2 - b^2) in (c / a) = real.sqrt 3 / 2)

noncomputable def line_PH_equation : Prop :=
  ∃ (a b k: ℝ), a = 2 ∧ b = 1 ∧ k = real.sqrt 6 / 6 ∧ 
    (let PH1 := (λ x y: ℝ, x - k * y + 1 = 0) in 
     let PH2 := (λ x y: ℝ, x + k * y + 1 = 0) in 
     ∀ x y, PH1 x y ∨ PH2 x y)

theorem math_problem_solution :
  ellipse_equation_and_eccentricity ∧ line_PH_equation :=
begin
  split,
  { unfold ellipse_equation_and_eccentricity,
    use 2, use 1,
    split,
    { reflexivity },
    split,
    { reflexivity },
    split,
    { intros,
      exact (x^2 / 4 + y^2 = 1) },
    split,
    { linarith },
    split,
    { linarith },
    let c := real.sqrt (4 - 1),
    have hc: c = real.sqrt 3, by { norm_num },
    rw [hc],
    norm_num, 
  },
  { unfold line_PH_equation,
    use 2, use 1, use real.sqrt 6 / 6,
    split,
    { reflexivity },
    split,
    { reflexivity },
    split,
    { reflexivity },
    intros,
    left,
    exact (x - real.sqrt 6 / 6 * y + 1 = 0)
  }
end

end math_problem_solution_l794_794546


namespace perpendicular_BC_DN_l794_794713

-- Define points and geometric relationships
variables {A D B C M N : Point}

-- Define the conditions
-- Assume a semicircle with diameter AD, B and C on the semicircle
-- M is the midpoint of BC, and M is the midpoint of AN
axiom semicircle (A D B C : Point) : ∃ O : Point, is_center O (circumference (semicircle A D)) ∧ on_circle B (semicircle A D) ∧ on_circle C (semicircle A D)
axiom midpoint (x y z : Point) : M = (x + y) / 2
axiom midpoint_AN (A N : Point) : M = (A + N) / 2

-- Prove that BC and DN are perpendicular
theorem perpendicular_BC_DN : 
  (∃ AD BC DN : Line, is_diameter AD (semicircle A D) ∧ BC = line_through B C ∧ DN = line_through D N) → 
  is_midpoint M B C → 
  is_midpoint M A N → 
  Perp BC DN :=
by sorry

end perpendicular_BC_DN_l794_794713


namespace probability_of_diamond_ace_joker_l794_794081

noncomputable def probability_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  event_cards / total_cards

noncomputable def probability_not_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_event total_cards event_cards

noncomputable def probability_none_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  (probability_not_event total_cards event_cards) * (probability_not_event total_cards event_cards)

noncomputable def probability_at_least_one_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_none_event_two_trials total_cards event_cards

theorem probability_of_diamond_ace_joker 
  (total_cards : ℕ := 54) (event_cards : ℕ := 18) :
  probability_at_least_one_event_two_trials total_cards event_cards = 5 / 9 :=
by
  sorry

end probability_of_diamond_ace_joker_l794_794081


namespace odd_n_divisibility_l794_794910

theorem odd_n_divisibility (n : ℤ) : (∃ a : ℤ, n ∣ 4 * a^2 - 1) ↔ (n % 2 ≠ 0) :=
by
  sorry

end odd_n_divisibility_l794_794910


namespace largest_base_5_five_digit_number_in_decimal_l794_794031

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794031


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l794_794192

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l794_794192


namespace largest_base5_number_conversion_l794_794012

noncomputable def largest_base5_number_in_base10 : ℕ := 3124

theorem largest_base5_number_conversion :
  (4 * 5^4) + (4 * 5^3) + (4 * 5^2) + (4 * 5^1) + (4 * 5^0) = largest_base5_number_in_base10 :=
by
  -- Proof would go here
  sorry

end largest_base5_number_conversion_l794_794012


namespace ratio_a11_b11_correct_l794_794232

-- Define that for a given natural number n, the sum of the first n terms of two arithmetic sequences S_n and T_n 
-- comply with the condition Sn / Tn = (2n) / (3n + 1).
def arithmetic_sequences (S_n T_n : ℕ → ℚ) :=
  ∀ n : ℕ, S_n n / T_n n = (2 * n) / (3 * n + 1)

noncomputable def ratio_a11_b11 (S_n T_n : ℕ → ℚ) [arithmetic_sequences S_n T_n] : ℚ :=
  let a₁₁ := S_n 21 / 21
  let b₁₁ := T_n 21 / 21
  a₁₁ / b₁₁

theorem ratio_a11_b11_correct (S_n T_n : ℕ → ℚ) [arithmetic_sequences S_n T_n] :
  ratio_a11_b11 S_n T_n = 21 / 32 := by
  sorry

end ratio_a11_b11_correct_l794_794232


namespace savings_calculation_l794_794355

theorem savings_calculation (income expenditure : ℝ) (h_ratio : income = 5 / 4 * expenditure) (h_income : income = 19000) :
  income - expenditure = 3800 := 
by
  -- The solution will be filled in here,
  -- showing the calculus automatically.
  sorry

end savings_calculation_l794_794355


namespace range_of_m_l794_794615

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end range_of_m_l794_794615


namespace problem_solution_l794_794936

def f (x : ℝ) : ℝ := log 4 (2 * x + 3 - x ^ 2)

theorem problem_solution :
  (∀ x, x > -1 ∧ x < 3 → 2 * x + 3 - x^2 > 0) ∧ -- Domain condition
  (∀ x, x ∈ Ioo (-1 : ℝ) 1 → monotone f x) ∧     -- Increasing interval
  (∀ x, x ∈ Ioo (1 : ℝ) 3 → antitone f x) ∧    -- Decreasing interval
  (f 1 = 1) :=                                       -- Maximum value
by
  sorry

end problem_solution_l794_794936


namespace area_of_square_l794_794095

-- Define the structure of the problem
variable (A B C : Type) [NormedField A] [NormedSpace A B]

-- Define the right triangle with specific area
variable (rightTriangle : B)
variable (A_real : Real)
variable (hypotenuse_position : Real)

-- Given conditions
def area_of_triangle (rightTriangle : B) : Real := 36

def vertex_position (hypotenuse_position : Real) : Prop := hypotenuse_position = 1/3

-- Statement to prove
theorem area_of_square 
  (h_triangle : area_of_triangle rightTriangle = 36)
  (h_position : vertex_position hypotenuse_position) :
  ∃ (square_in_triangle : B), area_of_triangle square_in_triangle = 16 :=
by
  sorry

end area_of_square_l794_794095


namespace maria_total_distance_in_miles_l794_794320

theorem maria_total_distance_in_miles :
  ∀ (steps_per_mile : ℕ) (full_cycles : ℕ) (remaining_steps : ℕ),
    steps_per_mile = 1500 →
    full_cycles = 50 →
    remaining_steps = 25000 →
    (100000 * full_cycles + remaining_steps) / steps_per_mile = 3350 := by
  intros
  sorry

end maria_total_distance_in_miles_l794_794320


namespace solutions_to_cubic_l794_794507

noncomputable def solve_cubic : Set ℂ := {z : ℂ | z^3 = -27}

theorem solutions_to_cubic :
  solve_cubic = {-3, 1.5 + 1.5 * complex.I * real.sqrt 3, 1.5 - 1.5 * complex.I * real.sqrt 3} :=
sorry

end solutions_to_cubic_l794_794507


namespace base6_div_correct_2123_23_l794_794907

def base6_to_base10 (digits : List ℕ) : ℕ :=
  List.foldr (fun (digit exp : ℕ) => digit + exp * 6) 0 digits.reverse

def div_in_base6 (n₆ m₆ : List ℕ) : List ℕ × ℕ :=
  let n₁₀ := base6_to_base10 n₆
  let m₁₀ := base6_to_base10 m₆
  (nat.div n₁₀ m₁₀, n₁₀ % m₁₀)

theorem base6_div_correct_2123_23 :
  div_in_base6 [2, 1, 2, 3] [2, 3] = ([5, 2], 3) :=
  sorry

end base6_div_correct_2123_23_l794_794907


namespace inequality_interval_l794_794688

def differentiable_on_R (f : ℝ → ℝ) : Prop := Differentiable ℝ f
def strictly_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

theorem inequality_interval (f : ℝ → ℝ)
  (h_diff : differentiable_on_R f)
  (h_cond : ∀ x : ℝ, f x > deriv f x)
  (h_init : f 0 = 1) :
  ∀ x : ℝ, (x > 0) ↔ (f x / Real.exp x < 1) := 
by
  sorry

end inequality_interval_l794_794688


namespace john_saved_25_percent_before_tax_l794_794667

def original_price (amount_spent amount_saved : ℝ) : ℝ :=
  amount_spent + amount_saved

def percentage_saved (amount_saved original_price : ℝ) : ℝ :=
  (amount_saved / original_price) * 100

theorem john_saved_25_percent_before_tax :
  ∀ (amount_spent amount_saved tax_rate : ℝ),
  amount_spent = 45 →
  amount_saved = 15 →
  tax_rate = 0.10 →
  percentage_saved amount_saved (original_price amount_spent amount_saved) = 25 :=
by
  intros amount_spent amount_saved tax_rate h_spent h_saved h_tax
  rw [h_spent, h_saved]
  calc
    percentage_saved 15 (original_price 45 15)
        = percentage_saved 15 60 : by sorry
    ... = 25 : by sorry

end john_saved_25_percent_before_tax_l794_794667


namespace find_a0_l794_794556

theorem find_a0 : 
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (∀ x : ℝ, (x + 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  a_0 = 32 := 
begin
  sorry
end

end find_a0_l794_794556


namespace unique_plane_through_n_parallel_to_m_l794_794562

-- Definition of skew lines 
def skew (m n : set ℝ^3) : Prop :=
  ¬(∃ p : ℝ^3, p ∈ m ∧ p ∈ n) ∧ ∀ (v1 ∈ m) (v2 ∈ n), v1 - v2 ≠ 0

-- This is the statement form of the problem rephrased as a Lean theorem
theorem unique_plane_through_n_parallel_to_m
  (m n : set ℝ^3)
  (hmn : skew m n) :
  ∃! α : set ℝ^3, (∀ (p ∈ n), p ∈ α) ∧ (∀ (v ∈ m), ∃ q ∈ α, parallel q v) :=
sorry

end unique_plane_through_n_parallel_to_m_l794_794562


namespace RM_squared_eq_RP_mul_RQ_l794_794828

variable {A B C D M P Q R : Type}
variables [AddCommGroup A] [Module ℝ A]
variables [AddCommGroup B] [Module ℝ B]
variables [AddCommGroup C] [Module ℝ C]
variables [AddCommGroup D] [Module ℝ D]
variables [AddCommGroup M] [Module ℝ M]
variables [AddCommGroup P] [Module ℝ P]
variables [AddCommGroup Q] [Module ℝ Q]
variables [AddCommGroup R] [Module ℝ R]

/-- Given a quadrilateral ABCD where M is the intersection of its diagonals AC and BD,
   and a secant line through M parallel to AB intersects sides at points P, Q, R (R on CD),
   prove that RM^2 = RP * RQ. -/
theorem RM_squared_eq_RP_mul_RQ
  (hM : M = line_intersection (diagonal A C) (diagonal B D))
  (h_parallel : secant_through M ∥ side A B)
  (h_intersects : secant_through M ∩ other_sides _ = {P, Q, R})
  (h_R_CD : R ∈ line C D) :
  (distance R M) ^ 2 = (distance R P) * (distance R Q) := 
by sorry

end RM_squared_eq_RP_mul_RQ_l794_794828


namespace new_average_l794_794135

theorem new_average (initial_count : ℕ) (initial_average : ℕ) (remove1 remove2 add : ℕ) 
(h_initial_count : initial_count = 60) (h_initial_average : initial_average = 40) 
(h_remove1 : remove1 = 50) (h_remove2 : remove2 = 60) (h_add : add = 35) : 
    let original_sum := initial_count * initial_average,
        new_sum := original_sum - remove1 - remove2 + add,
        new_count := initial_count - 1 in 
    new_sum / new_count = 39.41 :=
begin
    -- steps to prove this theorem would go here
    sorry
end

end new_average_l794_794135


namespace sum_of_roots_in_interval_l794_794549

-- Given conditions
variables {f : ℝ → ℝ} (hf_even : ∀ x, f x = f (-x)) (hf_period : ∀ x, f (x + 8) = f x)

-- Adding the condition for the root at x = 2
def root_in_interval (hf_root : f 2 = 0) := true

-- The mathematical problem stating question == answer
theorem sum_of_roots_in_interval :
  ∀ f : ℝ → ℝ, ∀ (hf_even : ∀ x, f x = f (-x)) (hf_period : ∀ x, f (x + 8) = f x)
  (hf_root : f 2 = 0), 
  ∑ x in ( { x | ∃ k : ℤ, x = 2 + 8 * k ∧ 0 ≤ x ∧ x ≤ 1000 } ∪ { x | ∃ k : ℤ, x = -2 + 8 * k ∧ 0 ≤ x ∧ x ≤ 1000 }),
  x = 125000 :=
by
  sorry

end sum_of_roots_in_interval_l794_794549


namespace identify_incorrect_calculation_l794_794808

theorem identify_incorrect_calculation : 
  (∀ x : ℝ, x^2 * x^3 = x^5) ∧ 
  (∀ x : ℝ, x^3 + x^3 = 2 * x^3) ∧ 
  (∀ x : ℝ, x^6 / x^2 = x^4) ∧ 
  ¬ (∀ x : ℝ, (-3 * x)^2 = 6 * x^2) := 
by
  sorry

end identify_incorrect_calculation_l794_794808


namespace symmetry_preserves_distance_l794_794323

-- Definitions for points and distance symmetry
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_point (p : Point) (axis : ℝ) : Point :=
  { x := 2 * axis - p.x, y := p.y }

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

theorem symmetry_preserves_distance (A B : Point) (axis : ℝ) :
  let A' := symmetric_point A axis
  let B' := symmetric_point B axis
  distance A B = distance A' B' :=
by
  let A' := symmetric_point A axis
  let B' := symmetric_point B axis
  sorry

end symmetry_preserves_distance_l794_794323


namespace quadratic_inequality_l794_794321

noncomputable def ax2_plus_bx_c (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |ax2_plus_bx_c a b c x| ≤ 1 / 2) →
  ∀ x : ℝ, |x| ≥ 1 → |ax2_plus_bx_c a b c x| ≤ x^2 - 1 / 2 :=
by
  sorry

end quadratic_inequality_l794_794321


namespace cloth_woven_on_30th_day_l794_794335

theorem cloth_woven_on_30th_day :
  (∃ d : ℚ, (30 * 5 + ((30 * 29) / 2) * d = 390) ∧ (5 + 29 * d = 21)) :=
by sorry

end cloth_woven_on_30th_day_l794_794335


namespace log_sum_of_x_n_l794_794941

theorem log_sum_of_x_n : 
  (∑ n in finset.range 2014, real.logb 2015 (n / (n + 1))) = -1 :=
sorry

end log_sum_of_x_n_l794_794941


namespace count_even_numbers_with_distinct_digits_l794_794239

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  digits.0 ≠ digits.1 ∧ digits.0 ≠ digits.2 ∧ digits.1 ≠ digits.2

def valid_even_numbers (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 998 ∧ is_even n ∧ has_distinct_digits n

theorem count_even_numbers_with_distinct_digits :
  { n : ℕ | valid_even_numbers n }.to_finset.card = 288 :=
by 
  sorry

end count_even_numbers_with_distinct_digits_l794_794239


namespace problem_statement_l794_794362

noncomputable theory

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def domain_of_f_div_2 : set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

theorem problem_statement
  (f1 : ℝ → ℝ)
  (f2 : ℝ → ℝ)
  (f3 : ℝ → ℝ)
  (f4 : ℝ → ℝ)
  (h1 : f1 = λ x, x^2 - (1 / x^2))
  (h2 : ∀ x, f2 (x / 2) ∈ domain_of_f_div_2 → x ∈ Icc 10 100)
  (h3 : ∀ x, -x^2 + 4*x + 5 > 0 → f3 x = log (-x^2 + 4*x + 5))
  (h4 : (∀ x, f4 x = sqrt (2^(mx^2 + 4*mx + 3) - 1)) → (0 ≤ m ∧ m ≤ 3 / 4)) :
  4 = 4 :=
sorry

end problem_statement_l794_794362


namespace arithmetic_b_sum_S_l794_794278

noncomputable theory

def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (a n) + 2^n

def b (n : ℕ) : ℕ := a n / 2^(n-1)

def is_arithmetic_sequence (seq : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n+1) - seq n = d

def S (n : ℕ) := ∑ k in Finset.range n, a k

theorem arithmetic_b : is_arithmetic_sequence b :=
  sorry

theorem sum_S (n : ℕ) : S n = (n-1) * 2^n + 1 :=
  sorry

end arithmetic_b_sum_S_l794_794278


namespace min_disks_required_l794_794664

def num_files : ℕ := 35
def disk_size : ℕ := 2
def file_size_0_9 : ℕ := 4
def file_size_0_8 : ℕ := 15
def file_size_0_5 : ℕ := num_files - file_size_0_9 - file_size_0_8

-- Prove the minimum number of disks required to store all files.
theorem min_disks_required 
  (n : ℕ) 
  (disk_storage : ℕ)
  (num_files_0_9 : ℕ)
  (num_files_0_8 : ℕ)
  (num_files_0_5 : ℕ) :
  n = num_files → disk_storage = disk_size → num_files_0_9 = file_size_0_9 → num_files_0_8 = file_size_0_8 → num_files_0_5 = file_size_0_5 → 
  ∃ (d : ℕ), d = 15 :=
by 
  intros H1 H2 H3 H4 H5
  sorry

end min_disks_required_l794_794664


namespace boat_travel_distance_along_stream_l794_794641

theorem boat_travel_distance_along_stream :
  ∀ (v_s : ℝ), (5 - v_s = 2) → (5 + v_s) * 1 = 8 :=
by
  intro v_s
  intro h1
  have vs_value : v_s = 3 := by linarith
  rw [vs_value]
  norm_num

end boat_travel_distance_along_stream_l794_794641


namespace equation_of_hyperbola_C_min_distance_P_M_l794_794205

-- Define the conditions given
def hyperbola1 (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1
def a1 := (2 : ℝ)
def b1 := (3 : ℝ)
def c1 := Real.sqrt (a1 + b1)
def foci_hyperbola1 := (fun (a b : ℝ) => ℝ.sqrt (a + b)) a1 b1

-- Equation of hyperbola C based on conditions
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1
def a2 := (2 : ℝ)
def b2 := Real.sqrt ((4 + 1) - 4)
def equation_of_C (x y : ℝ) := x^2 / 4 - y^2 = 1

theorem equation_of_hyperbola_C :
  (forall x y, (hyperbola_C x y)) := by
  sorry

-- Minimum distance calculation
def point_M := (5, 0)
def point_P_on_C (x y : ℝ) : Prop := hyperbola_C x y

def distance_P_M (x y : ℝ) : ℝ :=
  Real.sqrt (((x - 5)^2) + y^2)

theorem min_distance_P_M :
  (forall x y, point_P_on_C x y -> distance_P_M x y = 2) := by
  sorry

end equation_of_hyperbola_C_min_distance_P_M_l794_794205


namespace triangular_pyramid_total_surface_area_l794_794353

noncomputable def total_surface_area_of_triangular_pyramid (H : ℝ) : ℝ :=
  3 * sqrt 3 * H^2

theorem triangular_pyramid_total_surface_area (H : ℝ) (hH : H > 0)
  (angle_condition : ∀ (F A B C : ℝ), ∠ FLA = 60 ∧ ∠ KBL = 30) :
  total_surface_area_of_triangular_pyramid H = 3 * sqrt 3 * H^2 :=
sorry

end triangular_pyramid_total_surface_area_l794_794353


namespace problem_proof_l794_794200

noncomputable def parabola_focus (y_squared_2px : ℝ → Prop) (p : ℝ) (F : ℝ × ℝ) (px_gt_0 : 0 < p) : Prop :=
  ∀ (x y : ℝ), y_squared_2px y → y = 2 * F.1 ∧ x = F.1

def given_conditions (P : ℝ × ℝ) (PF_dist : ℝ) (parabola_line_AB : ℝ → Prop) (line_slope : ℝ) : Prop :=
  P = (-3, 2) ∧ PF_dist = 2 * sqrt 5 ∧ line_slope = 1

def area_ratio (S1 S2 : ℝ) (rat_range : Set ℝ) : Prop :=
  rat_range = Set.Ioo 0 1 ∧ ∃ (y1 y2 : ℝ), S1 = 4 * abs (y1 - y2) ∧ S2 = 4 * abs (y1 + y2 - 4)

theorem problem_proof :
  ∃ E : ℝ → Prop, ∀ (P : ℝ × ℝ) (PF_dist : ℝ) (parabola_line_AB : ℝ → Prop) (line_slope : ℝ) (S1 S2 : ℝ) (rat_range : Set ℝ),
    given_conditions P PF_dist parabola_line_AB line_slope →
    (∃ p : ℝ, parabola_focus E p (p / 2, 0) (0 < p) ∧ E = (λ y : ℝ, y^2 = 4 * p * x)) ∧
    ∀ (A B C : ℝ × ℝ), (BC_line : ℝ → Prop) -> 
      parabola_line_AB A.1 ∧ parabola_line_AB B.1 ∧ BC_line (B.1 * C.1) →
      BC_line(C.1) = true →
      BC_line = (λ y : ℝ, y = 5 + 2) ∧ (BC_line = fixed_point (5,2)) ∧
    area_ratio S1 S2 rat_range.
    sorry

end problem_proof_l794_794200


namespace ephraim_keiko_same_tails_l794_794286

def outcomes : List (List Char) := [
  ['H', 'H'], ['H', 'T'], ['T', 'H'], ['T', 'T']
]

def count_tails (lst : List Char) : Nat :=
  lst.count (· == 'T')

def favorable_cases : Nat :=
  List.filter (λ a, List.any (λ b, count_tails a = count_tails b) outcomes) outcomes |>.length

theorem ephraim_keiko_same_tails : favorable_cases / outcomes.length.toRat ^ 2 = 3 / 8 := by
  sorry

end ephraim_keiko_same_tails_l794_794286


namespace max_value_expression_l794_794952

variable (a b : ℝ)

theorem max_value_expression (h : a^2 + b^2 = 3 + a * b) : 
  ∃ a b : ℝ, (2 * a - 3 * b)^2 + (a + 2 * b) * (a - 2 * b) = 22 :=
by
  -- This is a placeholder for the actual proof
  sorry

end max_value_expression_l794_794952


namespace sum_of_solutions_l794_794510

def equation (x : ℝ) : Prop := -12 * x / ((x + 1) * (x - 1)) = 3 * x / (x + 1) - 9 / (x - 1)

theorem sum_of_solutions : 
    let solutions := {x : ℝ | equation x}
    (∑ x in solutions, x) = 0 :=
by {
    sorry
}

end sum_of_solutions_l794_794510


namespace minTrianglesGreaterThanOneHalf_l794_794250

noncomputable def minNumTrianglesWithLargeArea (points : List (Int × Int)) : Nat :=
  let triangles := (combinatorial.all_triangles points)  -- Hypothetical combinatorial function
  let countLargeAreaTriangles := triangles.count (λ t, trianglularArea t > 1 / 2)  -- Hypothetical area function
  countLargeAreaTriangles

theorem minTrianglesGreaterThanOneHalf (points : List (Int × Int))
  (h_length : points.length = 5)
  (h_lattice : ∀ p ∈ points, ∃ x y : Int, p = (x, y))
  (h_noncollinear : ∀ t ∈ (combinatorial.all_triangles points), ¬ (areCollinear t)) -- Hypothetical collinear function
  : minNumTrianglesWithLargeArea points ≥ 4 :=
sorry

end minTrianglesGreaterThanOneHalf_l794_794250


namespace smallest_side_length_1008_l794_794102

def smallest_side_length_original_square :=
  let n := Nat.lcm 7 8
  let n := Nat.lcm n 9
  let lcm := Nat.lcm n 10
  2 * lcm

theorem smallest_side_length_1008 :
  smallest_side_length_original_square = 1008 := by
  sorry

end smallest_side_length_1008_l794_794102


namespace list_length_arithmetic_sequence_l794_794241

/-- The sequence in the list is arithmetic with a common difference of 8,
  starting from -35 and ending at 61. The length of this list is 18. -/
theorem list_length_arithmetic_sequence : 
  let s : List ℤ := [-35, -27, -19, -11, -3, 5, 13, 21, 29, 37, 45, 53, 61]
  in s.length = 18 :=
by
  sorry

end list_length_arithmetic_sequence_l794_794241


namespace last_bead_is_yellow_l794_794662

def beadColor : Nat → String
| 0 => "red"
| 1 => "orange"
| 2 => "yellow"
| 3 => "yellow"
| 4 => "yellow"
| 5 => "green"
| 6 => "blue"
| 7 => "blue"
| n => beadColor (n % 8)

theorem last_bead_is_yellow (n : Nat) (h : n = 85) : beadColor (n - 1) = "yellow" := 
by
  have hn : 85 = 85 := rfl
  rw [h]
  have h_div := Nat.div_mod_eq_of_lt (n - 1) (8) (by decide)
  rw [Nat.mod_eq_of_lt (by decide : (n - 1) % 8 < 8), show (n - 1) % 8 = 4, by decide]
  exact rfl

end last_bead_is_yellow_l794_794662


namespace number_of_incorrect_statements_l794_794108

theorem number_of_incorrect_statements :
  let residuals_effective_judgement := True in
  let regression_increase_decrease := False in
  let regression_line_through_mean := True in
  let contingency_table_dependence := True in
  let statements := [residuals_effective_judgement, 
                     regression_increase_decrease, 
                     regression_line_through_mean, 
                     contingency_table_dependence] in
  (statements.filter (λ s => ¬s)).length = 1 :=
by
  let residuals_effective_judgement := True 
  let regression_increase_decrease := False 
  let regression_line_through_mean := True 
  let contingency_table_dependence := True 
  let statements := [residuals_effective_judgement, 
                     regression_increase_decrease, 
                     regression_line_through_mean, 
                     contingency_table_dependence]
  have h : (statements.filter (λ s => ¬s)).length = 1
  from sorry
  exact h

end number_of_incorrect_statements_l794_794108


namespace volume_ratio_spheres_l794_794778

theorem volume_ratio_spheres (r1 r2 r3 v1 v2 v3 : ℕ)
  (h_rad_ratio : r1 = 1 ∧ r2 = 2 ∧ r3 = 3)
  (h_vol_ratio : v1 = r1^3 ∧ v2 = r2^3 ∧ v3 = r3^3) :
  v3 = 3 * (v1 + v2) := by
  -- main proof goes here
  sorry

end volume_ratio_spheres_l794_794778


namespace length_PP_l794_794383

-- Definitions of the points and their reflection
def P : ℝ × ℝ := (2, 5)
def P' : ℝ × ℝ := (2, -5)

-- The length of the segment from P to P'
def length_pp' : ℝ := Real.sqrt ((P'.1 - P.1)^2 + (P'.2 - P.2)^2)

-- The theorem stating the length of PP' is 10
theorem length_PP'_is_10 : length_pp' = 10 := by
  sorry

end length_PP_l794_794383


namespace prime_product_blackened_squares_l794_794867

-- Define the initial positions of the elements in Figure 1
structure Position :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (blue : ℕ)

def Figure1 : Position := ⟨red := 1, yellow := 2, green := 3, blue := 4⟩
def Figure2 : Position := ⟨red := 4, yellow := 1, green := 3, blue := 2⟩
def Figure3 : Position := ⟨red := 1, yellow := 2, green := 4, blue := 3⟩
def Figure4 : Position := ⟨red := 5, yellow := 6, green := 7, blue := 8⟩

-- Mapping positions in Figure3 to Figure4 according to the given pattern
def figureTransformation (fig: Position) : Position :=
  { red := fig.blue,
    yellow := fig.red,
    green := fig.yellow,
    blue := fig.green }

-- Define Figure4 generated by the described transformation pattern
def GeneratedFigure4 := figureTransformation Figure3

-- Condition: Blackened squares in Figure 4 hold the numbers (7 and 5),
-- we verify that these positions hold primes and calculate their product.
def blackenedSquares {pos : Position} := (pos.yellow, pos.green) --positions of (7, 5)

theorem prime_product_blackened_squares (fig: Position) :
  is_prime fig.green ∧ is_prime fig.yellow →
  fig.green = 5 ∧ fig.yellow = 7 →
  fig.green * fig.yellow = 35 :=
by {
  sorry
}

-- Using the generated Figure4 to assert the theorem
example : GeneratedFigure4.green * GeneratedFigure4.yellow = 35 :=
  begin
    -- Assuming the primes are identified correctly in the blackened squares
    have G_is_prime := prime_product_blackened_squares GeneratedFigure4
      (by sorry) -- demonstrate the prime property
      (by sorry), -- demonstrate the placement property
    exact G_is_prime,
  end


end prime_product_blackened_squares_l794_794867


namespace range_of_t_l794_794333

noncomputable def f : ℝ → ℝ := sorry
def P (t : ℝ) : set ℝ := {x | |f (x + t) - 1| < 2}
def Q : set ℝ := {x | f x < -1}

lemma decreasing (a b : ℝ) (h : a < b) : f a > f b := sorry
lemma f_at_0 : f 0 = 3 := sorry
lemma f_at_3 : f 3 = -1 := sorry
lemma sufficient_but_not_necessary {t : ℝ} : (∀ x, x ∈ P t → x ∈ Q) ∧ (∃ x, ¬(x ∈ P t) ∧ x ∈ Q) := sorry

theorem range_of_t : {t : ℝ | t ≤ -3} = {t : ℝ | ∀ x, (x ∈ P t → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ ¬(x ∈ P t))} :=
by sorry

end range_of_t_l794_794333


namespace kathryn_rent_l794_794669

theorem kathryn_rent (R F: ℝ) 
  (h1: R = 1/2 * F) 
  (salary: ℝ) (remaining: ℝ)
  (h2: salary = 5000) 
  (h3: remaining = 2000) 
  (total_expenses = salary - remaining): 
  total_expenses = R + F → 
  R = 1000 := by
  sorry

end kathryn_rent_l794_794669


namespace sequence_inequality_l794_794584

noncomputable def sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 7 then 16 / 3
  else if n ≥ 2 then (3 * a (n - 1) + 4) / (7 - a (n - 1))
  else 0  -- arbitrary value for n < 2, not relevant for n ≥ 2

theorem sequence_inequality :
  ∃ m : ℕ, (m > 0) ∧ ∀ n ≥ m, sequence a n > (sequence a (n - 1) + sequence a (n + 1)) / 2 :=
sorry

end sequence_inequality_l794_794584


namespace division_of_2301_base4_by_21_base4_l794_794143

noncomputable def divide_in_base4 : ℕ := sorry

theorem division_of_2301_base4_by_21_base4 :
  let q := 112
  let r := 0
  let q_base10 := 22
  divide_in_base4 = (q, r) ∧ q = 112 ∧ r = 0 ∧ q_base10 = 22 :=
sorry

end division_of_2301_base4_by_21_base4_l794_794143


namespace boys_from_clay_middle_school_l794_794288

theorem boys_from_clay_middle_school (total_students boys girls jonas_students clay_students pine_students jonas_girls pine_boys : ℕ) 
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : jonas_students = 50)
  (h5 : clay_students = 40)
  (h6 : pine_students = 30)
  (h7 : jonas_girls = 30)
  (h8 : pine_boys = 10) :
  let jonas_boys := jonas_students - jonas_girls,
      boys_at_clay := boys - jonas_boys - pine_boys
  in boys_at_clay = 40 := sorry

end boys_from_clay_middle_school_l794_794288


namespace gerbils_left_l794_794848

theorem gerbils_left (initial count sold : ℕ) (h_initial : count = 85) (h_sold : sold = 69) : 
  count - sold = 16 := 
by 
  sorry

end gerbils_left_l794_794848


namespace part1_part2_l794_794295

def U : Set ℝ := {x : ℝ | True}

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part 1: Prove the range of m when 4 ∈ B(m) is [5/2, 3]
theorem part1 (m : ℝ) : (4 ∈ B m) → (5/2 ≤ m ∧ m ≤ 3) := by
  sorry

-- Part 2: Prove the range of m when x ∈ A is a necessary but not sufficient condition for x ∈ B(m) 
theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ∧ ¬(∀ x, x ∈ A → x ∈ B m) → (m ≤ 3) := by
  sorry

end part1_part2_l794_794295


namespace percent_of_100_is_30_l794_794829

theorem percent_of_100_is_30 : (30 / 100) * 100 = 30 := 
by
  sorry

end percent_of_100_is_30_l794_794829


namespace angle_c_sufficient_not_necessary_l794_794652

theorem angle_c_sufficient_not_necessary (A B C : ℝ) (h_triangle : A + B + C = π) (h_right_angle : C = π / 2) :
  (cos A + sin A = cos B + sin B) ∧ ¬(forall A B C, A + B + C = π ∧ (cos A + sin A = cos B + sin B) → C = π / 2) :=
by
  sorry

end angle_c_sufficient_not_necessary_l794_794652


namespace max_value_y_on_interval_l794_794358

noncomputable def y (x: ℝ) : ℝ := x^4 - 8 * x^2 + 2

theorem max_value_y_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y x = 11 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y z ≤ 11 := 
sorry

end max_value_y_on_interval_l794_794358


namespace hyungyeong_hula_hoops_l794_794594

theorem hyungyeong_hula_hoops :
  ∀ (h1 h2 h3 h4 : ℕ),
  h1 = 18 →
  h2 = 2 * h1 →
  h3 = 2 * h2 →
  h4 = 2 * h3 →
  h4 = 144 :=
by
  intros h1 h2 h3 h4 h1_def h2_def h3_def h4_def
  unfold_coes at *
  rw [h1_def, h2_def, h3_def, h4_def]
  sorry

end hyungyeong_hula_hoops_l794_794594


namespace new_tax_rate_is_correct_l794_794619

noncomputable def new_tax_rate (old_rate : ℝ) (income : ℝ) (savings : ℝ) : ℝ := 
  let old_tax := old_rate * income / 100
  let new_tax := (income - savings) / income * old_tax
  let rate := new_tax / income * 100
  rate

theorem new_tax_rate_is_correct :
  ∀ (income : ℝ) (old_rate : ℝ) (savings : ℝ),
    old_rate = 42 →
    income = 34500 →
    savings = 4830 →
    new_tax_rate old_rate income savings = 28 := 
by
  intros income old_rate savings h1 h2 h3
  sorry

end new_tax_rate_is_correct_l794_794619


namespace num_valid_lists_l794_794291

-- Variables section
variables {α : Type} [DecidableEq α] [Fintype α]

-- Definition of valid list
def is_valid_list (l : list ℕ) : Prop :=
  l.length = 12 ∧ (∀ i, 2 ≤ i → i < 12 → ((l.nth_le i sorry - 2 ∈ l.take i) ∨ (l.nth_le i sorry + 2 ∈ l.take i)))

-- Theorem statement
theorem num_valid_lists : (set_of is_valid_list).to_finset.card = 256 :=
sorry

end num_valid_lists_l794_794291


namespace min_value_m_l794_794559

theorem min_value_m (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + a 1)
  (h_geometric : ∀ n, b (n + 1) = b 1 * (b 1 ^ n))
  (h_b1_mean : 2 * b 1 = a 1 + a 2)
  (h_a3 : a 3 = 5)
  (h_b3 : b 3 = a 4 + 1)
  (h_S_formula : ∀ n, S n = n^2)
  (h_S_le_b : ∀ n ≥ 4, S n ≤ b n) :
  ∃ m, ∀ n, (n ≥ m → S n ≤ b n) ∧ m = 4 := sorry

end min_value_m_l794_794559


namespace find_f_1_l794_794367

variable {α : Type*} [Field α]

def f (x : α) : α
def g (x : α) : α

axiom f_2_eq_f_3: f 2 = 2
axiom f_3_eq_2: f 3 = 2
axiom g_2_eq_2: g 2 = 2
axiom g_3_eq_2: g 3 = 2
axiom g_1_eq_2: g 1 = 2
axiom f_5_eq_7: f 5 = 7
axiom g_5_eq_2: g 5 = 2

theorem find_f_1 : f 1 = 5 :=
by
  sorry

end find_f_1_l794_794367


namespace monotonic_increasing_interval_l794_794360

noncomputable def f : ℝ → ℝ := λ x, (3 - x) * Real.exp x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x < 2 → 0 < (2 - x) * Real.exp x :=
by
  intro x hx
  sorry

end monotonic_increasing_interval_l794_794360


namespace cross_product_scalar_multiplication_l794_794612

variable (c d : Vector ℝ 3)

theorem cross_product_scalar_multiplication
  (h : c × d = ![-3, 6, 2]) :
  c × (4 • d) = ![-12, 24, 8] :=
by
  sorry

end cross_product_scalar_multiplication_l794_794612


namespace speed_faster_correct_l794_794799

noncomputable def distance : ℕ := 960
noncomputable def time_faster : ℕ := 16
noncomputable def time_slower : ℕ := 17
noncomputable def speed_faster : ℕ := 60

theorem speed_faster_correct : 
  ∀ (d : ℕ) (t_f : ℕ) (t_s : ℕ) (s_f : ℕ), 
  d = distance → 
  t_f = time_faster → 
  t_s = time_slower → 
  s_f = speed_faster → 
  s_f = distance / t_f :=
by
  intros d t_f t_s s_f hd hf hs hs_f
  rw [hd, hf, hs_f]
  rfl

end speed_faster_correct_l794_794799


namespace combined_work_time_l794_794062

-- Define the time taken by Paul and Rose to complete the work individually
def paul_days : ℕ := 80
def rose_days : ℕ := 120

-- Define the work rates of Paul and Rose
def paul_rate := 1 / (paul_days : ℚ)
def rose_rate := 1 / (rose_days : ℚ)

-- Define the combined work rate
def combined_rate := paul_rate + rose_rate

-- Statement to prove: Together they can complete the work in 48 days.
theorem combined_work_time : combined_rate = 1 / 48 := by 
  sorry

end combined_work_time_l794_794062


namespace find_value_l794_794555

variable (m x : Real)
-- Hypothesis from the given condition
axiom h1 : m * Tan x = 2

-- Main theorem statement to prove
theorem find_value : (6 * m * sin (2 * x) + 2 * m * cos (2 * x)) / (m * cos (2 * x) - 3 * m * sin (2 * x)) = -2 / 5 :=
by 
  sorry

end find_value_l794_794555


namespace proof_U_eq_A_union_complement_B_l794_794309

noncomputable def U : Set Nat := {1, 2, 3, 4, 5, 7}
noncomputable def A : Set Nat := {1, 3, 5, 7}
noncomputable def B : Set Nat := {3, 5}
noncomputable def complement_U_B := U \ B

theorem proof_U_eq_A_union_complement_B : U = A ∪ complement_U_B := by
  sorry

end proof_U_eq_A_union_complement_B_l794_794309


namespace EF_eq_FL_l794_794550

noncomputable def IsoscelesRightTriangle (A B C : Point) : Prop :=
  ∠ A C B = 90 ∧ dist A B = dist A C

noncomputable def EqualSegments (B K C L : Point) : Prop :=
  dist B K = dist C L

noncomputable def Perpendicular (L1 L2 : Line) : Prop :=
  ∠ L1 L2 = 90

theorem EF_eq_FL
  (A B C K L E F : Point)
  (hABC : IsoscelesRightTriangle A B C)
  (hBK_CL : EqualSegments B K C L)
  (hE : ∃ D : Point, D ∈ Line_through B E ∧ Perpendicular (Line_through B E) (Line_through K C))
  (hF : ∃ G : Point, G ∈ Line_through A F ∧ Perpendicular (Line_through A F) (Line_through K C))
  : dist E F = dist F L := sorry

end EF_eq_FL_l794_794550


namespace evaluate_expression_l794_794498

theorem evaluate_expression (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
    (x - (y - z)) - ((x - y) - z) = 14 := by 
  sorry

end evaluate_expression_l794_794498


namespace monthly_interest_payment_l794_794864

theorem monthly_interest_payment (P : ℝ) (R : ℝ) (monthly_payment : ℝ)
  (hP : P = 28800) (hR : R = 0.09) : 
  monthly_payment = (P * R) / 12 :=
by
  sorry

end monthly_interest_payment_l794_794864


namespace Bruce_remaining_amount_l794_794115

/--
Given:
1. initial_amount: the initial amount of money that Bruce's aunt gave him, which is 71 dollars.
2. shirt_cost: the cost of one shirt, which is 5 dollars.
3. num_shirts: the number of shirts Bruce bought, which is 5.
4. pants_cost: the cost of one pair of pants, which is 26 dollars.
Show:
Bruce's remaining amount of money after buying the shirts and the pants is 20 dollars.
-/
theorem Bruce_remaining_amount
  (initial_amount : ℕ)
  (shirt_cost : ℕ)
  (num_shirts : ℕ)
  (pants_cost : ℕ)
  (total_amount_spent : ℕ)
  (remaining_amount : ℕ) :
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  total_amount_spent = shirt_cost * num_shirts + pants_cost →
  remaining_amount = initial_amount - total_amount_spent →
  remaining_amount = 20 :=
by
  intro h_initial h_shirt_cost h_num_shirts h_pants_cost h_total_spent h_remaining
  rw [h_initial, h_shirt_cost, h_num_shirts, h_pants_cost, h_total_spent, h_remaining]
  rfl

end Bruce_remaining_amount_l794_794115


namespace eval_fn_inv_expr_l794_794745

variable {α β : Type}

noncomputable def f : α → β := sorry
noncomputable def f_inv : β → α := sorry

-- Given Conditions
axiom f_6_eq_5 : f 6 = 5
axiom f_5_eq_1 : f 5 = 1
axiom f_1_eq_4 : f 1 = 4

-- Definition of inverse function based on given conditions
axiom inv_f_5_eq_6 : f_inv 5 = 6
axiom inv_f_4_eq_1 : f_inv 4 = 1

-- Equivalent proof problem statement
theorem eval_fn_inv_expr : f_inv (f_inv 5 * f_inv 4) = 6 :=
by
  sorry

end eval_fn_inv_expr_l794_794745


namespace bruce_money_left_l794_794119

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l794_794119


namespace log2_prob_l794_794716

noncomputable theory

open Set

theorem log2_prob (x : ℝ) (h1 : x ≥ 0) (h2 : x ≤ 4) : 
  let P := { x : ℝ | 0 < x ∧ x < 2 }.measure / { x : ℝ | 0 ≤ x ∧ x ≤ 4 }.measure in
  P = 1 / 2 :=
by
  sorry

end log2_prob_l794_794716


namespace quotients_and_remainders_identical_l794_794942

-- Let's define the problem in Lean.
theorem quotients_and_remainders_identical (n : ℕ) (h : n > 1) :
  ∀ d : ℕ, d ∣ (n + 1) →
    let q := (n / d) in
    let r := (n % d) in
    ∃ f : ℕ, n + 1 = d * f ∧ q = f - 1 ∧ r = d - 1 :=
begin
  sorry
end

end quotients_and_remainders_identical_l794_794942


namespace find_total_distance_l794_794417

-- Define the variable for total distance
variable (D : ℝ)

-- Conditions
def segment1_time : ℝ := 0.25 * D / 12
def segment2_time : ℝ := 0.35 * D / 9
def segment3_time : ℝ := 0.25 * D / 6
def segment4_time : ℝ := 0.15 * D / 8

-- Problem statement
def total_time_eq_five : Prop :=
  segment1_time D + segment2_time D + segment3_time D + segment4_time D = 5

-- Main statement: Prove that D satisfies the given equation
theorem find_total_distance (D : ℝ) (H : total_time_eq_five D) : D = 360 / 8.65 :=
sorry

end find_total_distance_l794_794417


namespace calc_sqrt_eq_solve_inequality_l794_794069

-- Problem 1: 
theorem calc_sqrt_eq : (∛(-8 : ℝ) + √((-4 : ℝ)^2) - -√2) = 2 + √2 := 
by
  sorry

-- Problem 2:
theorem solve_inequality (x : ℝ) : 7 * (1 - x) ≤ 5 * (3 - x) - 1 ↔ x ≥ -(7/2) := 
by
  sorry

end calc_sqrt_eq_solve_inequality_l794_794069


namespace select_students_with_A_or_B_l794_794732

-- Defining the parameters of the problem
def totalStudents : Nat := 9
def groupSize : Nat := 4
def excludedStudents : Nat := 7

-- Calculate total ways to choose 4 from 9
def totalWays : Nat := Nat.choose totalStudents groupSize

-- Calculate ways to choose 4 from 7 (excluding A and B)
def excludedWays : Nat := Nat.choose excludedStudents groupSize

-- Prove that the number of ways to select the group with at least one of A or B is 91
theorem select_students_with_A_or_B : 
  totalWays - excludedWays = 91 :=
by {
  -- Total ways to choose 4 out of 9
  have h1 : totalWays = Nat.choose 9 4 := rfl,
  -- Ways to choose 4 out of 7 excluding A and B
  have h2 : excludedWays = Nat.choose 7 4 := rfl,
  -- Perform the calculation
  calc
    totalWays - excludedWays 
      = Nat.choose 9 4 - Nat.choose 7 4  : by rw [h1, h2]
      = 126 - 35 : by rfl
      = 91 : by rfl
}

end select_students_with_A_or_B_l794_794732


namespace base5_to_base10_max_l794_794023

theorem base5_to_base10_max :
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  in max_base5 = 3124 :=
by
  let max_base5 := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  sorry

end base5_to_base10_max_l794_794023


namespace triangle_inequality_l794_794316

noncomputable theory -- Required for dealing with real numbers in certain computations

-- Define points on a plane with coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the function to calculate the area of a triangle given three points
def area (A B C : Point) : ℝ :=
(abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))) / 2

variables (A B C T X Y : Point)

-- Condition: Points X and Y are on sides AB and BC respectively
def X_on_AB : Prop := (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X.x = (1 - t) * A.x + t * B.x ∧ X.y = (1 - t) * A.y + t * B.y)
def Y_on_BC : Prop := (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Y.x = (1 - t) * B.x + t * C.x ∧ Y.y = (1 - t) * B.y + t * C.y)

-- Condition: Segments CX and AY intersect at point T
def T_on_CX_AY : Prop := 
  ∃ s u : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧ 
  T.x = (1 - s) * C.x + s * X.x ∧ T.y = (1 - s) * C.y + s * X.y ∧ 
  T.x = (1 - u) * A.x + u * Y.x ∧ T.y = (1 - u) * A.y + u * Y.y

-- The Lean 4 statement to prove the required problem
theorem triangle_inequality (hX : X_on_AB A B X) (hY : Y_on_BC B C Y) (hT : T_on_CX_AY A C T X Y) :
  area X B Y > area X T Y := sorry

end triangle_inequality_l794_794316


namespace bk_eq_2cd_l794_794960

noncomputable def triangle_proof (A B C D K : Point) : Prop :=
  ∃ (α : ℝ), 
  (∠ B C A = 90) ∧ 
  (∠ B A C = α) ∧ 
  (∠ C A B = 90 - α) ∧ 
  (D ∈ line_segment A C) ∧ 
  (K ∈ line_segment B D) ∧ 
  (∠ B A D = 90 - α) ∧ 
  (∠ B D C = 90 + α) ∧ 
  (∠ K A D = α) ∧ 
  (∠ A K D = α) ∧ 
  (∠ K D A = α) ∧ 
  BK = 2 * DC

-- The actual theorem
theorem bk_eq_2cd {A B C D K : Point} (h : triangle_proof A B C D K) : BK = 2 * DC :=
sorry

end bk_eq_2cd_l794_794960


namespace tangential_cyclic_quad_perpendicular_l794_794292

variables {A B C D E F G H : Point}
variables {π : Circle}
variables [cyclic_quad : cyclicQuadrilateral A B C D]
variables [tangential_quad : tangentialQuadrilateral A B C D π]
variables [inscribed_points : inscribedCircleTangentPoints π A B C D E F G H]

theorem tangential_cyclic_quad_perpendicular
  (hEG : line E G)
  (hHF : line H F)
  : ⊥ ⟂ (line E G) (line H F) :=
sorry

end tangential_cyclic_quad_perpendicular_l794_794292


namespace point_in_fourth_quadrant_l794_794572

open Complex

theorem point_in_fourth_quadrant (z : ℂ) (h : (3 + 4 * I) * z = 25) : 
  Complex.arg z > -π / 2 ∧ Complex.arg z < 0 := 
by
  sorry

end point_in_fourth_quadrant_l794_794572


namespace find_triples_l794_794151

-- Define the conditions in Lean 4
def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the math proof problem
theorem find_triples (m n p : ℕ) (hp : is_prime p) (hm : is_positive_integer m) (hn : is_positive_integer n) : 
  p^n + 3600 = m^2 ↔ (m = 61 ∧ n = 2 ∧ p = 11) ∨ (m = 65 ∧ n = 4 ∧ p = 5) ∨ (m = 68 ∧ n = 10 ∧ p = 2) :=
by
  sorry

end find_triples_l794_794151


namespace largest_base_5_five_digit_number_in_decimal_l794_794032

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794032


namespace smallest_possible_N_l794_794684

theorem smallest_possible_N :
  ∀ (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0),
  p + q + r + s + t = 4020 →
  (∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1342) :=
by
  intros p q r s t hp hq hr hs ht h
  use 1342
  sorry

end smallest_possible_N_l794_794684


namespace smallest_pos_int_m_exists_l794_794171

theorem smallest_pos_int_m_exists :
  ∃ m : ℕ, ∀ n : ℕ, odd n → (55^n + m * 32^n) % 2001 = 0 ∧ 
    ∀ m' : ℕ, (∀ n : ℕ, odd n → (55^n + m' * 32^n) % 2001 = 0) → m ≤ m' :=
begin
  -- proof goes here
  sorry
end

end smallest_pos_int_m_exists_l794_794171


namespace trigonometric_identity_l794_794719

theorem trigonometric_identity :
  cos (73 * Real.pi / 180) ^ 2 + 
  cos (47 * Real.pi / 180) ^ 2 + 
  cos (73 * Real.pi / 180) * cos (47 * Real.pi / 180) = 3 / 4 := 
sorry

end trigonometric_identity_l794_794719


namespace minimum_colors_for_hexagon_tessellation_l794_794181

theorem minimum_colors_for_hexagon_tessellation : 
  ∀ (hexagon : Type) (tessellate : hexagon → set hexagon),
  (∀ h : hexagon, card (tessellate h) = 3) → 
  ∃ (coloring : hexagon → ℕ), 
  (∀ h1 h2 : hexagon, h1 ≠ h2 → h1 ∈ tessellate h2 → coloring h1 ≠ coloring h2) ∧ 
  (∀ h : hexagon, coloring h < 3 + 1)
:=
  sorry

end minimum_colors_for_hexagon_tessellation_l794_794181


namespace probability_5_and_14_same_group_l794_794375

-- Define the main problem structure
def number_of_cards : ℕ := 20
def drawn_cards : list ℕ := [5, 14]
def remaining_cards : ℕ := number_of_cards - drawn_cards.length

-- Define the calculation for the total number of ways to draw 2 cards from the remaining cards
def total_ways : ℕ := remaining_cards * (remaining_cards - 1)

-- Define the favorable outcomes where 5 and 14 are in the same group
def favorable_outcomes : ℕ :=
  let case1 := (remaining_cards - 4) * (remaining_cards - 5) -- When 5 and 14 are the smallest
  let case2 := 4 * 3 -- When 5 and 14 are the largest
  case1 + case2

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_ways

-- Now write the final theorem
theorem probability_5_and_14_same_group :
  probability = 7 / 51 :=
by
  sorry

end probability_5_and_14_same_group_l794_794375


namespace bananas_to_pears_equivalence_l794_794868

-- Definitions based on conditions
def bananas := ℕ
def apples := ℕ
def pears := ℕ

axiom bananas_to_apples : ∀ (b a : bananas), 4 * b = 3 * a
axiom apples_to_pears : ∀ (a p : apples), 5 * a = 4 * p

-- Problem statement: 20 bananas equivalent to number of pears
theorem bananas_to_pears_equivalence : ∃ p : pears, 20 = p * 12 :=
by sorry

end bananas_to_pears_equivalence_l794_794868


namespace find_m_to_produce_double_root_l794_794737

theorem find_m_to_produce_double_root :
  ∀ (m : ℝ), (∀ x : ℝ, (x - 3) / (x - 1) = m / (x - 1) → (x = 1 → m = -2)) :=
begin
  assume m,
  assume x,
  assume h : (x - 3) / (x - 1) = m / (x - 1),
  assume hx : x = 1,
  sorry
end

end find_m_to_produce_double_root_l794_794737


namespace women_in_luxury_suites_l794_794326

theorem women_in_luxury_suites (total_passengers : ℕ) (percentage_women : ℝ) (percentage_in_luxury : ℝ)
  (hp : total_passengers = 250) (hw : percentage_women = 0.7) (hl : percentage_in_luxury = 0.15) :
  let women := total_passengers * percentage_women 
  let women_in_luxury := women * percentage_in_luxury
  women_in_luxury.round.to_nat = 26 :=
by 
  -- let statements for clarity in Lean
  let women := total_passengers * percentage_women 
  let women_in_luxury := women * percentage_in_luxury
  -- rounding step as required
  have h : women_in_luxury.round.to_nat = 26 := sorry
  -- prove given theorem
  exact h

end women_in_luxury_suites_l794_794326


namespace range_of_k_l794_794298

open Real

variables (k : ℝ) (e1 e2 : ℝ × ℝ) 

theorem range_of_k (hne_collinear : ¬(∃ λ : ℝ, e1 = λ • e2)) 
  (h_a : (2 * e1.1 + e2.1, 2 * e1.2 + e2.2) = (2, 1)) 
  (h_b : (k * e1.1 + 3 * e2.1, k * e1.2 + 3 * e2.2) ≠ (2, 1)) : 
  k ≠ 6 ∧ k ∈ ℝ := 
sorry

end range_of_k_l794_794298


namespace prob_less_than_9_l794_794364

def prob_10 : ℝ := 0.24
def prob_9 : ℝ := 0.28
def prob_8 : ℝ := 0.19

theorem prob_less_than_9 : prob_10 + prob_9 + prob_8 < 1 → 1 - prob_10 - prob_9 = 0.48 := 
by {
  sorry
}

end prob_less_than_9_l794_794364


namespace length_EF_eq_two_thirds_l794_794267

theorem length_EF_eq_two_thirds : 
  ∀ (A B C D E F M : ℝ) (AB BC CD : ℝ), 
  AB = 4 → BC = 12 →
  M = 6 →
  A = (0, 4) →
  moves_to (A, M) →
  E = (6, 4) →
  F = (16/3, 4) →
  length (E - F) = 2/3 := 
begin
  sorry -- proof omitted
end

end length_EF_eq_two_thirds_l794_794267


namespace no_common_prime_factors_l794_794672

variable {a b : ℤ}

theorem no_common_prime_factors :
  ∃ p q : ℤ, ∀ n : ℤ, ∀ r : ℕ, Prime r → ¬ (r ∣ (p + n * a) ∧ r ∣ (q + n * b)) :=
begin
  sorry
end

end no_common_prime_factors_l794_794672


namespace solve_for_y_l794_794329

theorem solve_for_y (y : ℝ) : (∀ y, ((1/9)^(3*y+9) = 81^(3*y+7)) → y = -23/9) :=
begin
  sorry
end

end solve_for_y_l794_794329


namespace combined_sleep_hours_l794_794882

theorem combined_sleep_hours :
  let connor_sleep_hours := 6
  let luke_sleep_hours := connor_sleep_hours + 2
  let emma_sleep_hours := connor_sleep_hours - 1
  let ava_sleep_hours :=
    2 * 5 + 
    2 * (5 + 1) + 
    2 * (5 + 2) + 
    (5 + 3)
  let puppy_sleep_hours := 2 * luke_sleep_hours
  let cat_sleep_hours := 4 + 7
  7 * connor_sleep_hours +
  7 * luke_sleep_hours +
  7 * emma_sleep_hours +
  ava_sleep_hours +
  7 * puppy_sleep_hours +
  7 * cat_sleep_hours = 366 :=
by
  sorry

end combined_sleep_hours_l794_794882


namespace smallest_lambda_exists_l794_794506

theorem smallest_lambda_exists :
  ∃ (λ : ℝ), λ = real.root 4 20 ∧ ∀ (n : ℕ), (nat.gcd n (⌊n * real.sqrt 5⌋)) < λ * real.sqrt n := sorry

end smallest_lambda_exists_l794_794506


namespace variance_of_X_l794_794852

noncomputable def variance (X : ℝ → ℝ) (a b : ℝ) := ∫ x in a..b, x^2 * X x

def pdf (c : ℝ) (x : ℝ) : ℝ :=
  if x ∈ (-c, c) then 1 / (π * (sqrt (c^2 - x^2))) else 0

theorem variance_of_X (c : ℝ) (hc : 0 < c) :
  variance (pdf c) (-c) c = c^3 / 2 :=
sorry

end variance_of_X_l794_794852


namespace necessary_but_not_sufficient_l794_794551

-- Definitions based on the conditions given
def line1 (x y : ℝ) := (3 * x) + (4 * y) - 3 = 0
def line2 (n : ℝ) (x y : ℝ) := (6 * x) + (8 * y) + n = 0
def distance_lines (c1 c2 : ℤ) := (|c1 - c2| : ℝ) / (Real.sqrt (3^2 + 4^2))

-- The proof statement
theorem necessary_but_not_sufficient (n : ℝ) : 
  ((distance_lines (-3) (-n / 2) = 2) ↔ (n = 14)) :=
sorry

end necessary_but_not_sufficient_l794_794551


namespace factorization_example_l794_794399

theorem factorization_example :
  (4 : ℤ) * x^2 - 1 = (2 * x + 1) * (2 * x - 1) := 
by
  sorry

end factorization_example_l794_794399


namespace problem1_problem2_l794_794071

-- First Problem
theorem problem1 (x : ℝ) (h : log 2 (16 - 2^x) = x) : x = 3 :=
sorry

-- Second Problem
theorem problem2 : 
  (-1 / (sqrt 5 - sqrt 3))^0 + 81^0.75 - sqrt ((-3)^2) * 8^(2/3) + log 7 25 = 18 :=
sorry

end problem1_problem2_l794_794071


namespace cheryl_initial_skitttles_l794_794129

-- Given conditions
def cheryl_ends_with (ends_with : ℕ) : Prop := ends_with = 97
def kathryn_gives (gives : ℕ) : Prop := gives = 89

-- To prove: cheryl_starts_with + kathryn_gives = cheryl_ends_with
theorem cheryl_initial_skitttles (cheryl_starts_with : ℕ) :
  (∃ ends_with gives, cheryl_ends_with ends_with ∧ kathryn_gives gives ∧ 
  cheryl_starts_with + gives = ends_with) →
  cheryl_starts_with = 8 :=
by
  sorry

end cheryl_initial_skitttles_l794_794129


namespace yellow_balls_count_l794_794265

-- Definitions from conditions
def total_balls : ℕ := 50
def frequency_yellow : ℝ := 0.3

-- The question to prove is that the number of yellow balls is 15 given the conditions.
theorem yellow_balls_count :
  ∃ (x : ℕ), (x / total_balls.to_float = frequency_yellow) ∧ x = 15 :=
by {
  sorry 
}

end yellow_balls_count_l794_794265


namespace infinitely_many_k_numbers_unique_k_4_l794_794927

theorem infinitely_many_k_numbers_unique_k_4 :
  ∀ k : ℕ, (∃ n : ℕ, (∃ r : ℕ, n = r * (r + k)) ∧ (∃ m : ℕ, n = m^2 - k)
          ∧ ∀ N : ℕ, ∃ r : ℕ, ∃ m : ℕ, N < r ∧ (r * (r + k) = m^2 - k)) ↔ k = 4 :=
by
  sorry

end infinitely_many_k_numbers_unique_k_4_l794_794927


namespace Bruce_remaining_amount_l794_794116

/--
Given:
1. initial_amount: the initial amount of money that Bruce's aunt gave him, which is 71 dollars.
2. shirt_cost: the cost of one shirt, which is 5 dollars.
3. num_shirts: the number of shirts Bruce bought, which is 5.
4. pants_cost: the cost of one pair of pants, which is 26 dollars.
Show:
Bruce's remaining amount of money after buying the shirts and the pants is 20 dollars.
-/
theorem Bruce_remaining_amount
  (initial_amount : ℕ)
  (shirt_cost : ℕ)
  (num_shirts : ℕ)
  (pants_cost : ℕ)
  (total_amount_spent : ℕ)
  (remaining_amount : ℕ) :
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  total_amount_spent = shirt_cost * num_shirts + pants_cost →
  remaining_amount = initial_amount - total_amount_spent →
  remaining_amount = 20 :=
by
  intro h_initial h_shirt_cost h_num_shirts h_pants_cost h_total_spent h_remaining
  rw [h_initial, h_shirt_cost, h_num_shirts, h_pants_cost, h_total_spent, h_remaining]
  rfl

end Bruce_remaining_amount_l794_794116


namespace largest_base5_number_to_base10_is_3124_l794_794005

theorem largest_base5_number_to_base10_is_3124 :
  let largest_base_5_number := 44444
  in (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
by
  sorry

end largest_base5_number_to_base10_is_3124_l794_794005


namespace largest_base_5_five_digit_number_in_decimal_l794_794034

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l794_794034


namespace range_of_f_l794_794777

def f (x : ℝ) := (1/2)^(-x^2 + 4*x + 1)

theorem range_of_f : 
  set.range (λ x : ℝ, if 0 ≤ x ∧ x ≤ 3 then f x else 0) = set.Icc (1/32:ℝ) (1/2:ℝ) := 
sorry

end range_of_f_l794_794777


namespace part1_part2_l794_794996

variables {x m : ℝ}
def a := (sqrt 3 * sin x, -1)
def b := (cos x, m)

-- Part 1: prove \(\cos^2 x - \sin 2x = \frac{3}{2}\) given conditions
theorem part1 (hx : m = sqrt 3) (hab : a.1 * b.2 = a.2 * b.1) : 
  cos x ^ 2 - sin (2 * x) = 3 / 2 := 
by 
  sorry

-- Part 2: prove the range of values for \(m\) given conditions
def f (x : ℝ) : ℝ := 2 * ((sqrt 3 * sin x + cos x, m) + (cos x, m)) . dot ((cos x, m)) - 2 * m ^ 2 - 1
def g (x : ℝ) : ℝ := f (x - π / 6)

theorem part2 (H : ∃ x ∈ set.Icc (0 : ℝ) (π / 2), g x = 0) : 
  -1/2 ≤ m ∧ m ≤ 1 := 
by 
  sorry

end part1_part2_l794_794996


namespace Joneal_stop_quarter_l794_794714

theorem Joneal_stop_quarter
  (circumference : ℕ)
  (distance_run : ℕ)
  (quarter_feet : ℕ)
  (quarters : list string)
  (circumference_eq_60 : circumference = 60)
  (distance_run_eq_5300 : distance_run = 5300)
  (quarter_feet_eq : quarter_feet = circumference / 4)
  (quarters_def : quarters = ["A", "B", "C", "D"]) :
  (quotient distance_run circumference) * circumference + (distance_run % circumference) < 5300 → 
  quarters.nth (distance_run % circumference / quarter_feet) = some "B" :=
by sorry

end Joneal_stop_quarter_l794_794714


namespace num_distinct_product_divisors_l794_794676

theorem num_distinct_product_divisors (T : set ℕ) (hT : T = {d | d ∣ 8000 ∧ d > 0}) : 
  ∃ n, n = 87 ∧ (∀ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b → (a * b) ∣ (8000 ^ 2) ∧ (a * b) ∈ T) := sorry

end num_distinct_product_divisors_l794_794676


namespace range_of_a_l794_794199

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → ln x - a * (1 - 1 / x) ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l794_794199


namespace polygon_sides_l794_794851

theorem polygon_sides (n : ℕ) (h : n * (n - 3) / 2 = 20) : n = 8 :=
by
  sorry

end polygon_sides_l794_794851


namespace log_expression_simplification_l794_794497

theorem log_expression_simplification :
  (Real.sqrt (log 12 / log 4 + log 12 / log 6) = Real.sqrt ((log 144) / (log 4 * log 6))) :=
by
  sorry

end log_expression_simplification_l794_794497


namespace necessary_but_not_sufficient_for_increasing_function_l794_794764

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

def necessary_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 1) > f(x)

theorem necessary_but_not_sufficient_for_increasing_function (f : ℝ → ℝ) :
  necessary_condition f → 
  ¬ (∀ x y : ℝ, x < y → f(x) < f(y)) → 
  true :=
begin
  sorry,
end

end necessary_but_not_sufficient_for_increasing_function_l794_794764


namespace monotonic_intervals_of_f_range_of_a_l794_794935

def f (x : ℝ) : ℝ := x * Real.log x
def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 - x + 2

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x > 0 → 
  ((x < 1 / Real.exp 1 → (f derivative x < 0)) ∧ (x > 1 / Real.exp 1 → (f derivative x > 0))) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * f x ≤ (g derivative x a) + 2) → a ≥ -2 :=
sorry

end monotonic_intervals_of_f_range_of_a_l794_794935


namespace minimum_value_of_expression_l794_794537

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  1 / x + 4 / y + 9 / z

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  min_value_expression x y z ≥ 36 :=
sorry

end minimum_value_of_expression_l794_794537


namespace Jane_l794_794281

noncomputable def number_of_divisors (n : ℕ) := 
  ∏ d in divisors n, 1

def sum_of_divisors (n : ℕ) : ℕ := 
  ∑ d in divisors n, d

def sum_of_prime_divisors (n : ℕ) : ℕ :=
  ∑ p in (factors n).to_finset, p

def satisfies_conditions (n : ℕ) : Prop := 
  (500 < n) ∧ 
  (1000 > n) ∧ 
  (number_of_divisors n = 20) ∧ 
  (∃ m, m ≠ n ∧ sum_of_divisors m = sum_of_divisors n) ∧ 
  (∃ m, m ≠ n ∧ sum_of_prime_divisors m = sum_of_prime_divisors n)

theorem Jane's_number : ∃ n : ℕ, satisfies_conditions n ∧ n = 880 :=
by
  existsi 880
  split
  -- Now utilize sorry to skip the actual proof.
  sorry

end Jane_l794_794281


namespace circle_center_and_radius_l794_794939

-- Define the given conditions
variable (a : ℝ) (h : a^2 = a + 2 ∧ a ≠ 0)

-- Define the equation
noncomputable def circle_equation (x y : ℝ) : ℝ := a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

-- Lean definition to represent the problem
theorem circle_center_and_radius :
  (∃a : ℝ, a ≠ 0 ∧ a^2 = a + 2 ∧
    (∃x y : ℝ, circle_equation a x y = 0) ∧
    ((a = -1) → ((∃x y : ℝ, (x + 2)^2 + (y + 4)^2 = 25) ∧
                 (center_x = -2) ∧ (center_y = -4) ∧ (radius = 5)))) :=
by
  sorry

end circle_center_and_radius_l794_794939


namespace households_in_village_l794_794625

theorem households_in_village 
  (water_per_household : ℕ)
  (total_water : ℕ)
  (water_duration : ℕ)
  (H1 : water_per_household = 200)
  (H2 : total_water = 2000)
  (H3 : water_duration = 1) :
  total_water / water_per_household = 10 :=
by
  rw [H1, H2]
  exact rfl

end households_in_village_l794_794625


namespace books_distribution_1_books_distribution_2_l794_794374

-- Problem 1
theorem books_distribution_1 :
  let books := 7
  let distribute_to (a b c : ℕ) (books : ℕ) := a + b + c = books
  let num_ways := choose books 1 * choose (books - 1) 2 * choose (books - 3) 4 * fact 3 in
  distribute_to 1 2 4 books → num_ways = 630 :=
by
  intros
  sorry

-- Problem 2
theorem books_distribution_2 :
  let books := 7
  let distribute_to (a b c : ℕ) (books : ℕ) := a + b + c = books
  let num_ways := choose books 2 * choose (books - 2) 2 * choose (books - 4) 3 * fact 3 / fact 2 in
  distribute_to 3 2 2 books → num_ways = 630 :=
by
  intros
  sorry

end books_distribution_1_books_distribution_2_l794_794374


namespace largest_base_5_five_digits_base_10_value_l794_794037

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794037


namespace constant_polynomial_condition_l794_794929

variable {R : Type*} [CommRing R]

-- Definition of bivariate polynomial
def is_const_poly (P : R[X][Y]) : Prop :=
  ∃ c : R, ∀ x y : R, P.coeff x y = c

-- Statement of the problem in Lean 4
theorem constant_polynomial_condition (P : R[X][Y]) (h : ∀ x y : R, P.coeff x y = P.coeff (x + y) (x - y)) : 
  is_const_poly P :=
sorry

end constant_polynomial_condition_l794_794929


namespace problem_part_I_problem_part_II_l794_794825

-- Part (I)
theorem problem_part_I (α : ℝ) (hα : α = π / 6) :
  (2 * sin(π + α) * cos(π - α) - cos(π + α)) / 
  (1 + sin(α)^2 + sin(π - α) - cos(π + α)^2) = sqrt(3) :=
by
  sorry

-- Part (II)
theorem problem_part_II (α : ℝ) (hα : tan(α) / (tan(α) - 6) = -1) :
  (2 * cos(α) - 3 * sin(α)) / (3 * cos(α) + 4 * sin(α)) = -7 / 15 :=
by
  sorry

end problem_part_I_problem_part_II_l794_794825


namespace min_sum_of_arithmetic_sequence_l794_794971

-- Given conditions
variables (a1 : ℤ) (S3 : ℤ)
-- Define the arithmetic sequence property
def S (n : ℕ) (d : ℤ) := n * a1 + (n * (n - 1) / 2) * d

theorem min_sum_of_arithmetic_sequence (a1_eq : a1 = -7) (S3_eq : S 3 2 = -15) :
  ∃ d : ℤ, ∀ n : ℕ, S n d = (n * n - 8 * n) → (∀ m, S m d ≥ -16) :=
sorry

end min_sum_of_arithmetic_sequence_l794_794971


namespace andy_solves_49_problems_l794_794060

theorem andy_solves_49_problems : ∀ (a b : ℕ), a = 78 → b = 125 → b - a + 1 = 49 :=
by
  introv ha hb
  rw [ha, hb]
  norm_num
  sorry

end andy_solves_49_problems_l794_794060


namespace jude_total_matchbox_vehicles_l794_794283

/-- Definition of variables based on the given conditions -/
def bottle_caps_for_car : ℕ := 5
def bottle_caps_for_truck : ℕ := 6
def total_bottle_caps : ℕ := 100
def trucks_bought : ℕ := 10
def rem_bottle_caps_fraction_for_cars : ℚ := 0.75

/-- Definition to calculate the total matchbox vehicles Jude buys -/
def total_matchbox_vehicles (bottle_caps_for_car : ℕ) (bottle_caps_for_truck : ℕ) (total_bottle_caps : ℕ) (trucks_bought : ℕ) (rem_bottle_caps_fraction_for_cars : ℚ) : ℕ :=
  let bottle_caps_spent_on_trucks := trucks_bought * bottle_caps_for_truck
  let remaining_bottle_caps := total_bottle_caps - bottle_caps_spent_on_trucks
  let bottle_caps_spent_on_cars := (rem_bottle_caps_fraction_for_cars * remaining_bottle_caps).to_nat
  let cars_bought := bottle_caps_spent_on_cars / bottle_caps_for_car
  trucks_bought + cars_bought

/-- Theorem to prove the total number of matchbox vehicles is 16 -/
theorem jude_total_matchbox_vehicles : total_matchbox_vehicles bottle_caps_for_car bottle_caps_for_truck total_bottle_caps trucks_bought rem_bottle_caps_fraction_for_cars = 16 :=
by sorry

end jude_total_matchbox_vehicles_l794_794283


namespace find_m_l794_794596

open_locale big_operators

theorem find_m (m : ℝ) (a : ℕ → ℝ) (h1 : ∀ x, (1 + m * x)^6 = ∑ i in finset.range 7, a i * x^i)
  (h2 : ∑ i in finset.range 6, a (i + 1) = 63) : 
  m = 1 :=
sorry

end find_m_l794_794596


namespace knicks_from_knocks_l794_794609

variable (knicks knacks knocks : Type)
variable [HasSmul ℚ knicks] [HasSmul ℚ knacks] [HasSmul ℚ knocks]

variable (k1 : knicks) (k2 : knacks) (k3 : knocks)
variable (h1 : 5 • k1 = 3 • k2)
variable (h2 : 4 • k2 = 6 • k3)

theorem knicks_from_knocks : 36 • k3 = 40 • k1 :=
by {
  sorry
}

end knicks_from_knocks_l794_794609


namespace rhombus_triangle_area_l794_794406

theorem rhombus_triangle_area (d1 d2 : ℝ) (h_d1 : d1 = 15) (h_d2 : d2 = 20) :
  ∃ (area : ℝ), area = 75 := 
by
  sorry

end rhombus_triangle_area_l794_794406


namespace find_positive_integral_values_l794_794369

theorem find_positive_integral_values 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, x^3 - 3 * a * x^2 + b * x + 18 * c = 0 → 
    ∃ α β γ : ℝ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ β = α + d ∧ γ = α + 2 * d)
  (h2 : ∀ x : ℝ, x^3 + b * x^2 + x - c^3 = 0 → 
    ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ v = u * r ∧ w = v * r)
  (h3 : a > 0 ∧ b > 0 ∧ a ∈ ℕ ∧ b ∈ ℕ) : 
  a = 2 ∧ b = 9 :=
sorry

end find_positive_integral_values_l794_794369


namespace sum_of_solutions_eq_zero_l794_794516

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | -12*x/(x^2-1) = 3*x/(x+1) - 9/(x-1)}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l794_794516


namespace evaluate_expression_at_neg3_l794_794496

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end evaluate_expression_at_neg3_l794_794496


namespace coefficient_of_x_in_expansion_l794_794968

theorem coefficient_of_x_in_expansion (n : ℕ) (h : 2 ^ n = 128) :
  ∃ k : ℕ, (2 * (7.choose k) * 2 ^ (7 - k) * x^((7:ℕ) - 3*k/2)) = 280 * x :=
by sorry

end coefficient_of_x_in_expansion_l794_794968


namespace sum_of_solutions_l794_794508

def equation (x : ℝ) : Prop := -12 * x / ((x + 1) * (x - 1)) = 3 * x / (x + 1) - 9 / (x - 1)

theorem sum_of_solutions : 
    let solutions := {x : ℝ | equation x}
    (∑ x in solutions, x) = 0 :=
by {
    sorry
}

end sum_of_solutions_l794_794508


namespace cube_volumes_total_l794_794045

theorem cube_volumes_total :
  let v1 := 5^3
  let v2 := 6^3
  let v3 := 7^3
  v1 + v2 + v3 = 684 := by
  -- Here will be the proof using Lean's tactics
  sorry

end cube_volumes_total_l794_794045


namespace actual_average_height_of_boys_l794_794751

theorem actual_average_height_of_boys :
  let n := 50
  let incorrect_average_height := 175
  let recorded_heights := [155, 185, 170]
  let actual_heights := [145, 195, 160]
  let total_difference := (recorded_heights.zip actual_heights).sum (λ p, p.1 - p.2)
  let incorrect_total_height := incorrect_average_height * n
  let correct_total_height := incorrect_total_height - total_difference
  actual_average_height = Float.round (correct_total_height / n) 2 :=
  sorry

end actual_average_height_of_boys_l794_794751


namespace greatest_possible_remainder_l794_794588

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end greatest_possible_remainder_l794_794588


namespace geometric_sequence_common_ratio_l794_794203

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l794_794203


namespace students_helped_on_fourth_day_l794_794436

theorem students_helped_on_fourth_day (total_books : ℕ) (books_per_student : ℕ)
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ)
  (H1 : total_books = 120) (H2 : books_per_student = 5)
  (H3 : day1_students = 4) (H4 : day2_students = 5) (H5 : day3_students = 6) :
  (total_books - (day1_students * books_per_student + day2_students * books_per_student + day3_students * books_per_student)) / books_per_student = 9 :=
by
  sorry

end students_helped_on_fourth_day_l794_794436


namespace smallest_positive_period_of_f_max_min_value_of_f_on_interval_l794_794984

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 6) * cos x + 1

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := 
sorry

theorem max_min_value_of_f_on_interval :
  let I := set.Icc (π / 12) (π / 2) in
  ∀ x ∈ I, 
    (∀ y ∈ I, f y ≤ f x) → f x = 5 / 4 ∨ 
    (∀ y ∈ I, f y ≥ f x) → f x = 3 / 4 := 
sorry

end smallest_positive_period_of_f_max_min_value_of_f_on_interval_l794_794984


namespace maria_must_earn_l794_794706

-- Define the given conditions
def retail_price : ℕ := 600
def maria_savings : ℕ := 120
def mother_contribution : ℕ := 250

-- Total amount Maria has from savings and her mother's contribution
def total_savings : ℕ := maria_savings + mother_contribution

-- Prove that Maria must earn $230 to be able to buy the bike
theorem maria_must_earn : 600 - total_savings = 230 :=
by sorry

end maria_must_earn_l794_794706


namespace sampling_is_systematic_l794_794418

-- Define conditions based on the given problem
structure Factory :=
  (produces_products : Prop)
  (uses_conveyor_belt : Prop)

structure Inspection :=
  (sampling_method : ℕ → Prop)  -- A sampling method is defined by a function which given an interval, returns a proposition

-- Define that a certain type of sampling method is systematic
def is_systematic_sampling (f : ℕ → Prop) (interval : ℕ) : Prop :=
  ∃ start_point, f interval ∧ (∀ n, f ((interval * n) + start_point))

-- Define the conditions present in the problem
axiom factory : Factory
axiom inspection : Inspection
axiom sampling_interval : ℕ := 10

-- State the theorem to prove that the given sampling method is systematic
theorem sampling_is_systematic : inspection.sampling_method sampling_interval → is_systematic_sampling inspection.sampling_method sampling_interval :=
sorry

end sampling_is_systematic_l794_794418


namespace knocks_to_knicks_l794_794603

def knicks := ℕ
def knacks := ℕ
def knocks := ℕ

axiom knicks_to_knacks_ratio (k : knicks) (n : knacks) : 5 * k = 3 * n
axiom knacks_to_knocks_ratio (n : knacks) (o : knocks) : 4 * n = 6 * o

theorem knocks_to_knicks (k : knicks) (n : knacks) (o : knocks) (h1 : 5 * k = 3 * n) (h2 : 4 * n = 6 * o) :
  36 * o = 40 * k :=
sorry

end knocks_to_knicks_l794_794603


namespace no_integer_solutions_l794_794158

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l794_794158


namespace max_remainder_is_10_l794_794591

theorem max_remainder_is_10 (x : ℕ) (h : x % 11 ≠ 0) : x % 11 = 10 :=
begin
  sorry
end

end max_remainder_is_10_l794_794591


namespace wrapping_third_roll_l794_794729

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l794_794729


namespace mode_and_median_l794_794338

def data_set : List ℕ := [29, 32, 33, 35, 35, 40]

theorem mode_and_median (mode median : ℕ) (h_mode : mode = 35) (h_median : median = 34) :
  List.mode data_set = mode ∧ List.median data_set = median := by
  sorry

end mode_and_median_l794_794338


namespace find_a_b_and_extrema_l794_794175

noncomputable def f (a b x : ℝ) : ℝ := -2 * a * sin (2 * x + π / 6) + 2 * a + b

theorem find_a_b_and_extrema (a b : ℝ) (h : 0 < a) :
  (∀ x ∈ Icc (0 : ℝ) (π / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) →
  (a = 2 ∧ b = -5) ∧
  (∀ x ∈ Icc (0 : ℝ) (π / 4), 
    (f 2 (-5) 0 = -3) ∧ (f 2 (-5) (π / 6) = -5)) :=
begin
  intros h1,
  split,
  { sorry },
  { intros x hx,
    split,
    { exact sorry },
    { exact sorry } }
end

end find_a_b_and_extrema_l794_794175


namespace area_of_original_triangle_l794_794106

/-- 
Given an equilateral triangle with side length 1, prove that the area 
of the original triangle, which is derived using the provided relationship 
with the intuitive diagram, equals \(\frac{\sqrt{6}}{2}\).
-/
theorem area_of_original_triangle :
  let side_length := 1
  let S_intuitive_diagram := (sqrt 3) / 4
  let S_original_triangle := S_intuitive_diagram * 2 * sqrt 2
  S_original_triangle = (sqrt 6) / 2 := 
by 
  let side_length := 1
  let S_intuitive_diagram := (sqrt 3) / 4
  let S_original_triangle := S_intuitive_diagram * 2 * sqrt 2
  show S_original_triangle = (sqrt 6) / 2
  from sorry

end area_of_original_triangle_l794_794106


namespace x1_plus_x2_eq_3_l794_794561

noncomputable def x1 := 
  Classical.some (exists.intro (3 - log 10 3) (by simp [log]; sorry))

noncomputable def x2 := 
  Classical.some (exists.intro (3 - 3) (by simp [pow]; sorry))

theorem x1_plus_x2_eq_3 
  (hx1 : x1 + log 10 x1 = 3) 
  (hx2 : x2 + 10 ^ x2 = 3) : 
  x1 + x2 = 3 := 
by
  sorry

end x1_plus_x2_eq_3_l794_794561


namespace product_gcd_lcm_l794_794165

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l794_794165


namespace distinct_digit_sum_l794_794648

theorem distinct_digit_sum (a b : ℕ) (h_a : a ∈ {0,1,2,3,4,5,6,7,8,9}) (h_b : b ∈ {0,1,2,3,4,5,6,7,8,9})
  (h_distinct : a ≠ b) (h_eq : 45 = 11 * a + 2 * b): 
  ∑ x in ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ {a, b}), x = 36 :=
by
  sorry

end distinct_digit_sum_l794_794648


namespace arccos_zero_l794_794474

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l794_794474


namespace profit_percent_is_20_l794_794064

-- Given conditions
variables {C P : ℝ}
variable (h : (2 / 3) * P = 0.8 * C)

-- Proof statement
theorem profit_percent_is_20 (C P : ℝ) (h : (2 / 3) * P = 0.8 * C) : 
  let profit := P - C in
  let profit_percent := (profit / C) * 100 in
  profit_percent = 20 :=
by
  sorry

end profit_percent_is_20_l794_794064


namespace paper_unfolded_with_four_holes_l794_794478

-- Definitions and conditions
structure Rectangle :=
  (width height : ℝ)

def fold_diagonal (rect : Rectangle) : Rectangle := 
  { width := rect.width / 2, height := rect.height / 2 }

def fold_bottom_to_top (rect : Rectangle) : Rectangle := 
  { width := rect.width / 2, height := rect.height / 2 }

def fold_right_to_left (rect : Rectangle) : Rectangle := 
  { width := rect.width / 2, height := rect.height / 2 }

def punch_hole (rect : Rectangle) : Rectangle := rect

-- Theorem to be proved
theorem paper_unfolded_with_four_holes 
  (rect : Rectangle) 
  (h1 : fold_diagonal rect = fold_bottom_to_top rect)
  (h2 : fold_bottom_to_top rect = fold_right_to_left rect)
  (h3 : punch_hole (fold_right_to_left rect) = rect) : 
  (∃ holes, holes = 4) :=
sorry

end paper_unfolded_with_four_holes_l794_794478


namespace find_distance_between_intersections_l794_794347

theorem find_distance_between_intersections (N : ℕ) :
  let points := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2) ∧ x^2 = 5 * x + 24} in
  ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → 
  (p1 ≠ p2 → (sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) = sqrt N) → N = 3146 :=
begin
  sorry -- proof goes here
end

end find_distance_between_intersections_l794_794347


namespace boat_speed_in_still_water_l794_794076

theorem boat_speed_in_still_water (V_s : ℕ) (D_d : ℕ) (T_d : ℕ) (H : V_s = 5) (H1 : D_d = 90) (H2 : T_d = 3) : 
  ∃ V_b : ℕ, D_d = (V_b + V_s) * T_d ∧ V_b = 25 :=
by {
  use 25,
  rw [H, H1, H2],
  show 90 = (25 + 5) * 3,
  norm_num,
  exact ⟨rfl, rfl⟩,
  sorry
}

end boat_speed_in_still_water_l794_794076


namespace domino_tiling_exist_l794_794715

noncomputable def infinite_sheet_graph_paper_domino_tiling : Prop :=
  ∃ (tiling : ℤ × ℤ → ℤ × ℤ → ℕ), 
    (∀ (x y : ℤ), ∃ (x' y' : ℤ), tiling (x, y) = (x', y') ∧
        (|x - x'| + |y - y'| = 1) ∧  -- Each domino covers adjacent cells
        ∀ l, (∀ n : ℕ, ∃ i : ℕ, l i = x + n ∨ l i = y + n) → -- For any line following the grid
        finite {d | ∃ n, tiling (l n) = d}).                 -- Intersects only a finite number of dominos

theorem domino_tiling_exist :
  infinite_sheet_graph_paper_domino_tiling := sorry

end domino_tiling_exist_l794_794715


namespace sum_of_squares_of_distances_l794_794843

theorem sum_of_squares_of_distances (n : ℕ) (h : n > 0) : 
  let vertices := (fin n) → ℝ × ℝ  -- coordinates of vertices in 2D plane, forming an n-gon
  let unit_circle (v : ℝ × ℝ) := v.1^2 + v.2^2 = 1  -- each vertex is on unit circle
  let regular_ngon (v : fin n → ℝ × ℝ) := ∀ k : fin n, unit_circle (v k)  -- n-gon inscribed in unit circle
  let sum_distance_squares (l : ℝ × ℝ) (v : fin n → ℝ × ℝ) := ∑ k in (finset.univ : finset (fin n)), 
    let distance := abs ((l.1 * (v k).1 + l.2 * (v k).2) / sqrt (l.1^2 + l.2^2)) in distance ^ 2
  in ∀ vertices : fin n → ℝ × ℝ, regular_ngon vertices → ∃ l : ℝ × ℝ, abs(l.1) + abs(l.2) = 1  ∧  sum_distance_squares l vertices = n / 2 :=
sorry

end sum_of_squares_of_distances_l794_794843


namespace find_p_q_l794_794694

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l794_794694


namespace average_female_students_l794_794352

def n_8A := 10
def n_8B := 14
def n_8C := 7
def n_8D := 9
def n_8E := 13

theorem average_female_students : (n_8A + n_8B + n_8C + n_8D + n_8E) / 5 = 10.6 := by
  sorry

end average_female_students_l794_794352


namespace quadratic_shift_correct_l794_794351

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x + a)
def shift_down (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x - b

def quadratic_shift_test : Prop :=
  let f := λ x, -2 * (x + 1) ^ 2 + 5 in
  let g := λ x, -2 * (x + 3) ^ 2 + 1 in
  (shift_down (shift_left f 2) 4 = g)

theorem quadratic_shift_correct : quadratic_shift_test := by
  sorry

end quadratic_shift_correct_l794_794351


namespace total_men_employed_l794_794862

-- Definition of the given conditions
def initial_num_of_men := M : ℕ
def days_to_finish_originally := 12
def days_to_finish_with_extra_men := 9
def extra_men := 10
def work_equation (M : ℕ) : Prop := M * days_to_finish_originally = (M + extra_men) * days_to_finish_with_extra_men

-- The goal is to prove the total number of men employed in total to finish the work
theorem total_men_employed (M : ℕ) (h : work_equation M) : M + extra_men = 40 :=
by {
  sorry
}

end total_men_employed_l794_794862


namespace winston_initial_gas_l794_794811

theorem winston_initial_gas (max_gas : ℕ) (store_gas : ℕ) (doctor_gas : ℕ) :
  store_gas = 6 → doctor_gas = 2 → max_gas = 12 → max_gas - (store_gas + doctor_gas) = 4 → max_gas = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end winston_initial_gas_l794_794811


namespace a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l794_794191

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l794_794191


namespace exists_power_of_two_with_last_n_digits_ones_and_twos_l794_794724

theorem exists_power_of_two_with_last_n_digits_ones_and_twos (N : ℕ) (hN : 0 < N) :
  ∃ k : ℕ, ∀ i < N, ∃ (d : ℕ), d = 1 ∨ d = 2 ∧ 
    (2^k % 10^N) / 10^i % 10 = d :=
sorry

end exists_power_of_two_with_last_n_digits_ones_and_twos_l794_794724


namespace general_term_of_sequence_sum_of_reciprocals_of_bn_l794_794308

-- Condition: Sum of the first n terms
def Sn (n : ℕ) (k q : ℝ) := k * q^n - k

-- General term definition derived from conditions 
def a (n : ℕ) : ℝ := 3^n

-- Condition: Define b_n
def b (n : ℕ) : ℝ := a n - (1 / (a n))

-- Theorem statement for the first part of the problem: General term of the sequence 
theorem general_term_of_sequence (n : ℕ) (k q : ℝ) (h₁ : a 1 = 3) (h₂ : a 4 = 81) : a n = 3^n := by 
  sorry

-- Theorem statement for the second part of the problem: Sum of reciprocals of b_n
theorem sum_of_reciprocals_of_bn (n : ℕ) : (∑ i in range (n + 1), 1 / (b i)) < 9 / 16 := by 
  sorry

end general_term_of_sequence_sum_of_reciprocals_of_bn_l794_794308


namespace sin_log_x_eq_1_infinitely_many_times_l794_794593

theorem sin_log_x_eq_1_infinitely_many_times : ¬finite { x : ℝ | 0 < x ∧ x < 1 ∧ sin (log x) = 1 } :=
sorry

end sin_log_x_eq_1_infinitely_many_times_l794_794593


namespace trimino_tilings_greater_l794_794178

noncomputable def trimino_tilings (n : ℕ) : ℕ := sorry
noncomputable def domino_tilings (n : ℕ) : ℕ := sorry

theorem trimino_tilings_greater (n : ℕ) (h : n > 1) : trimino_tilings (3 * n) > domino_tilings (2 * n) :=
sorry

end trimino_tilings_greater_l794_794178


namespace max_value_A_l794_794073

/--
A and B play a number-changing game on a 5 × 5 grid: A starts and both take turns filling empty spaces, with A filling each space with the number 1 and B filling each space with the number 0. After the grid is completely filled, the sum of the numbers in each 3 × 3 square is calculated, and the maximum sum among these squares is denoted as A. A tries to maximize A, while B tries to minimize A. Prove that the maximum value of A that A can achieve is 6.
-/
theorem max_value_A (grid : Array (Array ℕ)) (h_dim : grid.size = 5) (h_fill : ∀ i j, grid[i][j] = 0 ∨ grid[i][j] = 1) :
  ∃ A, A = 6 := 
sorry

end max_value_A_l794_794073


namespace PS_length_l794_794642

noncomputable def length_of_PS (PQ QR RS : ℝ) (angleQ_right angleR_right : Prop) : ℝ := 
  let PT := QR 
  let TS := RS - PQ 
  real.sqrt (PT^2 + TS^2)

theorem PS_length (PQ QR RS : ℝ) (hPQ : PQ = 6) (hQR : QR = 10) (hRS : RS = 25) 
  (angleQ_right : true) (angleR_right : true) : 
  length_of_PS PQ QR RS angleQ_right angleR_right = real.sqrt 461 := by
  sorry

end PS_length_l794_794642


namespace length_of_AC_is_sqrt_29_l794_794275

open_locale real

noncomputable def length_of_AC (AB AD AA' : ℝ) (angle_A'AB angle_A'AD : ℝ) : ℝ :=
  real.sqrt (AB^2 + AD^2 + AA'^2 + 2 * AB * AD * real.cos (angle_A'AB) + 2 * AD * AA' * real.cos (angle_A'AD) + 2 * AB * AA' * real.cos (angle_A'AD))

theorem length_of_AC_is_sqrt_29 :
  length_of_AC 2 2 3 (real.pi / 3) (real.pi / 3) = real.sqrt 29 :=
sorry

end length_of_AC_is_sqrt_29_l794_794275


namespace volume_of_pyramid_l794_794325

noncomputable theory -- Necessary for dealing with non-constructive proofs

open Real -- Use Real numbers

/-- We define the conditions given in the problem. --/
def regular_pentagon (A B C D E F : Point) : Prop :=
  -- Definition of regular pentagon
  is_regular_pentagon A B C D E F

def equilateral_triangle_PAD (P A D : Point) (side_len : ℝ) : Prop :=
  -- Definition of equilateral triangle with specific side length
  equilateral_triangle P A D ∧ side_len = 10

/-- The question to prove is the volume of the pyramid. --/
theorem volume_of_pyramid
  (A B C D E F P : Point)
  (h_pentagon : regular_pentagon A B C D E F)
  (h_triangle : equilateral_triangle_PAD P A D 10) :
  volume_of_pyramid P A B C D E F = 125 * sqrt (15 * (5 + 2 * sqrt 5)) / 12 :=
begin
  sorry -- Proof to be provided
end

end volume_of_pyramid_l794_794325


namespace sea_turtle_age_conversion_l794_794853

def octal_to_decimal (n : List ℕ) : ℕ :=
  n.foldr (λ (digit acc pow8 : ℕ), digit * pow8 + acc) 0

theorem sea_turtle_age_conversion : octal_to_decimal [5, 3, 6] = 350 := by
  sorry

end sea_turtle_age_conversion_l794_794853


namespace norm_difference_unit_vectors_l794_794377

open Real

variables {u1 u2 : ℝ} (x y z : ℝ)

-- Definition: unit vectors
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2 = 1

-- First unit vector and angle condition
def vector1 := (3, -1, 1)

def condition1 (u : ℝ × ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), cos (π / 6) = (u.1 * 3 + u.2 * -1 + u.3 * 1) / sqrt (3^2 + (-1)^2 + (1)^2)

-- Second unit vector and angle condition
def vector2 := (1, 2, 2)

def condition2 (u : ℝ × ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), cos (π / 4) = (u.1 * 1 + u.2 * 2 + u.3 * 2) / sqrt (1^2 + (2)^2 + (2)^2)

-- Lean 4 statement for the problem
theorem norm_difference_unit_vectors :
  is_unit_vector u1 →
  is_unit_vector u2 →
  condition1 u1 →
  condition2 u1 →
  condition1 u2 →
  condition2 u2 →
  u1 ≠ u2 →
  ∃ (norm_val : ℝ), norm_val = ∥u1 - u2∥ :=
sorry

end norm_difference_unit_vectors_l794_794377


namespace variance_of_transformed_binomial_l794_794703

open ProbabilityTheory

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

noncomputable def D (Y : ℚ) : ℚ := binomial_variance 3 ((5/9 : ℚ) - (4/9 : ℚ))

theorem variance_of_transformed_binomial :
  ∃ (p : ℚ),
    (Pr (X ≥ 1) = 5 / 9) ∧ (D (3 * Y + 1) = 6) :=
begin
  sorry
end

end variance_of_transformed_binomial_l794_794703


namespace island_inhabitants_even_l794_794068

theorem island_inhabitants_even (P : Type) [Inhabited P]
  (is_inhabitant : P → Prop)
  (is_liar : P → Prop)
  (is_knight : P → Prop)
  (friends : P → set P)
  (odd_friends : ∀ p : P, is_inhabitant p → (∃ k n : ℕ, k ∈ friends p ∧ n ∈ friends p ∧ n % 2 = 0 ∧ k % 2 = 1))
  (statements : ∀ p : P, is_inhabitant p → 
    ( ∃ (knights_friends : ℕ) (liars_friends : ℕ),
      knights_friends % 2 = 1 ∧
      liars_friends % 2 = 0)) :
  (is_even (count {x : P | is_inhabitant x})) :=
by
  sorry

end island_inhabitants_even_l794_794068


namespace greatest_radius_l794_794613

theorem greatest_radius (A : ℝ) (hA : A < 60 * Real.pi) : ∃ r : ℕ, r = 7 ∧ (r : ℝ) * (r : ℝ) < 60 :=
by
  sorry

end greatest_radius_l794_794613


namespace sum_of_solutions_eq_zero_l794_794517

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | -12*x/(x^2-1) = 3*x/(x+1) - 9/(x-1)}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l794_794517


namespace math_problems_l794_794442

structure Person (name : String) :=
(age : ℕ)

noncomputable def age_diff (a b : Person) : ℕ :=
a.age - b.age

def Albert := Person.mk "Albert" 18
def Albert_father_at_birth := 48
def Albert_mother_at_birth_of_brother := 46
def Albert_younger_brother_diff := 2
def Albert_mother_at_birth_of_sister := 50
def Eldest_sibling_diff := 4
def Father_age_at_birth_of_eldest := 40

noncomputable def mother_age_at_birth_of_Albert : ℕ :=
Albert_mother_at_birth_of_brother - Albert_younger_brother_diff

noncomputable def mother_father_age_diff := 
Albert_father_at_birth - mother_age_at_birth_of_Albert

noncomputable def mother_age_at_birth_of_eldest := 
Father_age_at_birth_of_eldest - mother_father_age_diff

noncomputable def father_current_age := 
Albert_father_at_birth + Albert.age

noncomputable def mother_current_age := 
mother_age_at_birth_of_Albert + Albert.age

noncomputable def eldest_sibling_current_age := 
Albert.age + Eldest_sibling_diff

noncomputable def brother_current_age := 
Albert.age - Albert_younger_brother_diff

noncomputable def sister_current_age := 
Albert.age - (Albert_mother_at_birth_of_sister - mother_age_at_birth_of_Albert)

theorem math_problems :
  age_diff (Person.mk "father" father_current_age) 
           (Person.mk "mother" mother_current_age) = 4 ∧
  age_diff (Person.mk "brother" brother_current_age) 
           (Person.mk "sister" sister_current_age) = 4 ∧
  father_current_age = 66 ∧
  mother_current_age = 62 ∧
  eldest_sibling_current_age = 22 ∧
  brother_current_age = 16 ∧
  sister_current_age = 12 :=
by {
  sorry
}

end math_problems_l794_794442


namespace triangle_BF_length_l794_794328

theorem triangle_BF_length (A B C D F G : Type*)
  (h_triangle_ABC : Triangle A B C)
  (h_parallel : Parallel D F G A B)
  (h_D_on_AC : D ∈ Segment A C)
  (h_F_on_BC : F ∈ Segment B C)
  (h_AF_bisects_BGF : Bisects A F (Angle B G F))
  (h_AB_len : Length A B = 10)
  (h_DF_len : Length D F = 4)
  : Length B F = 20 / 3 := 
sorry

end triangle_BF_length_l794_794328


namespace sock_ratio_l794_794707

variable (b : ℕ) -- number of blue sock pairs
variable (x : ℕ) -- price per pair of blue socks
variable (y : ℕ) -- price per pair of black socks = 3 * x
variable (C_original : ℕ) -- original cost
variable (C_interchanged : ℕ) -- interchanged cost

-- original conditions
def original_black_socks := 5
def original_blue_socks := b
def price_per_pair_blue := x
def price_per_pair_black := 3 * x
def original_cost := 5 * (3 * x) + b * x
def interchanged_cost := b * (3 * x) + 5 * x

-- proof condition: interchanged cost is 60% more than original cost
def condition := interchanged_cost = 1.6 * original_cost

-- final proof statement
theorem sock_ratio (b x : ℕ) (h : condition) :
    5 / b = 5 / 14 := 
sorry

end sock_ratio_l794_794707


namespace possible_values_of_m_l794_794620

theorem possible_values_of_m (m : ℤ) (x y : ℝ) :
  (∃ a b : ℝ, (4 : ℝ) = a^2 ∧ (9 : ℝ) = b^2 ∧ (4x^2 + m * x * y + 9 * y^2) = (a * x + b * y) ^ 2) →
  m = 12 ∨ m = -12 :=
by
  sorry

end possible_values_of_m_l794_794620


namespace fewest_tiles_needed_l794_794432

-- Define the dimensions of the tile
def tile_width : ℕ := 2
def tile_height : ℕ := 5

-- Define the dimensions of the floor in feet
def floor_width_ft : ℕ := 3
def floor_height_ft : ℕ := 6

-- Convert the floor dimensions to inches
def floor_width_inch : ℕ := floor_width_ft * 12
def floor_height_inch : ℕ := floor_height_ft * 12

-- Calculate the areas in square inches
def tile_area : ℕ := tile_width * tile_height
def floor_area : ℕ := floor_width_inch * floor_height_inch

-- Calculate the minimum number of tiles required, rounding up
def min_tiles_required : ℕ := Float.ceil (floor_area / tile_area)

-- The theorem statement: prove that the minimum tiles required is 260
theorem fewest_tiles_needed : min_tiles_required = 260 := 
  by 
    sorry

end fewest_tiles_needed_l794_794432


namespace correct_propositions_l794_794809

variables {R : Type*} [RealField R]
variables (a e : EuclideanSpace R)
variables (λ : R)

-- Definition: A vector is parallel to another vector if there exists a λ such that b = λa.
def is_parallel (a b : EuclideanSpace R) : Prop := ∃ (λ : R), b = λ • a

-- Definition: e is a unit vector
def is_unit_vector (e : EuclideanSpace R) : Prop := ∥e∥ = 1

-- Proposition B: |(a⋅a)a| = |a|^3
def prop_B (a : EuclideanSpace R) : Prop := ∥(a ⋅ a) • a∥ = ∥a∥^3

-- Proposition C: If e is a unit vector, and a is parallel to e, then a = ±|a|e.
def prop_C (a e : EuclideanSpace R) : Prop :=
  is_unit_vector e → is_parallel a e → (a = ∥a∥ • e ∨ a = -∥a∥ • e)

-- Main theorem statement
theorem correct_propositions :
  prop_B a ∧ prop_C a e :=
by
  sorry  -- Placeholder for the proof

end correct_propositions_l794_794809


namespace bruce_money_left_l794_794120

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l794_794120


namespace digit_sum_remainder_l794_794501

theorem digit_sum_remainder (n : ℕ) (h : n > 1) : 
  (∀ k, 0 ≤ k ∧ k < n → ∃ m, (m % n = 0) ∧ ((sum_digits m) % n = k)) ↔ (¬ (3 ∣ n)) :=
sorry

end digit_sum_remainder_l794_794501


namespace no_such_alpha_exists_l794_794407

theorem no_such_alpha_exists :
  ¬ ∃ α : ℝ, irrational (cos α)
    ∧ rational (cos (2 * α))
    ∧ rational (cos (3 * α))
    ∧ rational (cos (4 * α))
    ∧ rational (cos (5 * α)) := 
sorry

end no_such_alpha_exists_l794_794407


namespace count_valid_numbers_l794_794237

-- Definition of the problem conditions
def is_valid_number (n: ℕ) : Prop :=
 n >= 200 ∧ n <= 998 ∧
 (n % 2 = 0) ∧
 let digits := List.ofDigits (Nat.digits 10 n) in
 digits.Nodup

-- The statement to be proved
theorem count_valid_numbers : 
  { n // is_valid_number n }.count = 408 :=
sorry

end count_valid_numbers_l794_794237


namespace find_initial_children_l794_794331

-- Definition of conditions
def initial_children_on_bus (X : ℕ) := 
  let final_children := (X + 40) - 60 
  final_children = 2

-- Theorem statement
theorem find_initial_children : 
  ∃ X : ℕ, initial_children_on_bus X ∧ X = 22 :=
by
  sorry

end find_initial_children_l794_794331


namespace cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l794_794661

-- Define the production cost function
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Define the profit function
def profit (n : ℕ) : ℤ := 90 * n - 4000 - 50 * n

-- 1. Prove that the cost for producing 1000 pairs of shoes is 54,000 yuan
theorem cost_of_1000_pairs : production_cost 1000 = 54000 := 
by sorry

-- 2. Prove that if the production cost is 48,000 yuan, then 880 pairs of shoes were produced
theorem pairs_for_48000_yuan (n : ℕ) (h : production_cost n = 48000) : n = 880 := 
by sorry

-- 3. Prove that at least 100 pairs of shoes must be produced each day to avoid a loss
theorem minimum_pairs_to_avoid_loss (n : ℕ) : profit n ≥ 0 ↔ n ≥ 100 := 
by sorry

end cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l794_794661


namespace sum_of_cubes_of_roots_l794_794761

noncomputable def cube_root (x : ℝ) : ℝ := sorry -- Placeholder definition for cube root

theorem sum_of_cubes_of_roots :
  let δ := cube_root 7
  let ε := cube_root 29
  let ζ := cube_root 61
  ∃ u v w : ℝ,
    (u - δ) * (v - ε) * (w - ζ) = 1 / 5 ∧
    u^3 + v^3 + w^3 = 97.6 :=
by
  let δ := cube_root 7
  let ε := cube_root 29
  let ζ := cube_root 61
  use [δ, ε, ζ]
  split
  · sorry -- The proof of polynomial roots condition
  · sorry -- The proof of sum of cubes

end sum_of_cubes_of_roots_l794_794761


namespace proof_problem_l794_794547

-- Conditions for the ellipse
def ellipse_coefs (a b : ℝ) := 0 < b ∧ b < a 
def focal_length (a b : ℝ) := (a^2 - b^2 = 8)
def minor_semi_axis (b : ℝ) := b = 2

-- Equation of the Ellipse
noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, ellipse_coefs a b ∧ focal_length a b ∧ minor_semi_axis b ∧ 
    (∀ x y : ℝ, ((x^2/a^2 + y^2/b^2 = 1) ↔ ((x^2/12 + y^2/4 = 1)))

-- Line passing through point P with slope 1
noncomputable def line_through_p (y x : ℝ) : Prop := 
  (∀ x y : ℝ, (y = x + 3 ↔ ((x+2)*(x+2) + 1 = y)))

-- Length of the Chord
noncomputable def chord_length : ℝ :=
  ∃ (x1 x2 : ℝ), (-6) * (253 / 286) = 0 ∧ 4 * ((x1 * x2) - (x1 / 2 * x2 * y)) = 42

-- Combined, for readability
theorem proof_problem : Prop :=
  ellipse_equation ∧ line_through_p ∧ chord_length = (sqrt 42 / 2)

end proof_problem_l794_794547


namespace min_subset_condition_l794_794920

theorem min_subset_condition (m : ℕ) : 
  (∀ S ⊆ (Finset.range 2017), S.card = m → ∃ a b ∈ S, a ≠ b ∧ |a - b| ≤ 3) ↔ m ≥ 505 := by
sorry

end min_subset_condition_l794_794920


namespace part_one_part_two_l794_794579

noncomputable def sqrt_e : ℝ := real.sqrt (exp 1)

def f (x a : ℝ) : ℝ := exp x - a * x - a / 2

theorem part_one (a : ℝ) (h1 : ∀ x : ℝ, f x a ≥ 0) : 0 ≤ a ∧ a ≤ sqrt_e :=
sorry

theorem part_two (m : ℝ) (h2 : ∀ x > 0, exp x ≥ real.log x + m) : m > 2.3 :=
sorry

end part_one_part_two_l794_794579


namespace solution_l794_794693

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end solution_l794_794693


namespace least_number_to_add_l794_794042

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) : n = 1100 → d = 23 → r = n % d → (r ≠ 0) → (d - r) = 4 :=
by
  intros h₀ h₁ h₂ h₃
  simp [h₀, h₁] at h₂
  sorry

end least_number_to_add_l794_794042


namespace shopper_saved_percentage_l794_794437

theorem shopper_saved_percentage (amount_paid : ℝ) (amount_saved : ℝ) (original_price : ℝ)
  (h1 : amount_paid = 45) (h2 : amount_saved = 5) (h3 : original_price = amount_paid + amount_saved) :
  (amount_saved / original_price) * 100 = 10 :=
by
  -- The proof is omitted
  sorry

end shopper_saved_percentage_l794_794437


namespace minimum_odd_integers_l794_794384

theorem minimum_odd_integers :
  ∀ (a b c d e f : ℤ), 
  a + b = 28 →
  a + b + c + d = 46 →
  a + b + c + d + e + f = 66 →
  (∀ x ∈ {a, b, c, d, e, f}, even x) :=
sorry

end minimum_odd_integers_l794_794384


namespace count_valid_f_values_l794_794702

theorem count_valid_f_values 
  (p q : ℕ)
  (h_coprime : Nat.gcd p q = 1)
  (h_p_ge_two : p ≥ 2) 
  (h_interval : x = q / p) 
  (h_condition : ∀ x ∈ Icc 0 1, f x > 1 / 5 → x ∈ Icc 0 1) :
  (∃ x ∈ Icc 0 1, f x > 1 / 5) → x.card = 5 := 
by
  sorry

end count_valid_f_values_l794_794702


namespace sum_of_roots_l794_794521

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in ({√3, -√3} : set ℝ), x

theorem sum_of_roots :
  (∀ x : ℝ, (x ≠ 1) ∧ (x ≠ -1) → ( -12 * x) / (x ^ 2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  sum_of_solutions = 0 :=
sorry

end sum_of_roots_l794_794521


namespace max_remainder_is_10_l794_794590

theorem max_remainder_is_10 (x : ℕ) (h : x % 11 ≠ 0) : x % 11 = 10 :=
begin
  sorry
end

end max_remainder_is_10_l794_794590


namespace just_passed_students_l794_794264

theorem just_passed_students (total_students : ℕ) 
  (math_first_division_perc : ℕ) 
  (math_second_division_perc : ℕ)
  (eng_first_division_perc : ℕ)
  (eng_second_division_perc : ℕ)
  (sci_first_division_perc : ℕ)
  (sci_second_division_perc : ℕ) 
  (math_just_passed : ℕ)
  (eng_just_passed : ℕ)
  (sci_just_passed : ℕ) :
  total_students = 500 →
  math_first_division_perc = 35 →
  math_second_division_perc = 48 →
  eng_first_division_perc = 25 →
  eng_second_division_perc = 60 →
  sci_first_division_perc = 40 →
  sci_second_division_perc = 45 →
  math_just_passed = (100 - (math_first_division_perc + math_second_division_perc)) * total_students / 100 →
  eng_just_passed = (100 - (eng_first_division_perc + eng_second_division_perc)) * total_students / 100 →
  sci_just_passed = (100 - (sci_first_division_perc + sci_second_division_perc)) * total_students / 100 →
  math_just_passed = 85 ∧ eng_just_passed = 75 ∧ sci_just_passed = 75 :=
by
  intros ht hf1 hf2 he1 he2 hs1 hs2 hjm hje hjs
  sorry

end just_passed_students_l794_794264


namespace polynomial_rearrangement_l794_794111

theorem polynomial_rearrangement :
  (λ x : ℝ, x^4 + 2 * x^3 - 3 * x^2 - 4 * x + 1) =
  (λ x : ℝ, (x+1)^4 - 2 * (x+1)^3 - 3 * (x+1)^2 + 4 * (x+1) + 1) :=
by
  sorry

end polynomial_rearrangement_l794_794111


namespace building_shadow_length_l794_794419

theorem building_shadow_length :
  ∀ (height_flagpole shadow_flagpole height_building : ℕ), 
  height_flagpole = 18 → shadow_flagpole = 45 → height_building = 22 →
  ∃ (shadow_building : ℕ), 
  shadow_building * height_flagpole = shadow_flagpole * height_building ∧ shadow_building = 55 := 
by 
  intros height_flagpole shadow_flagpole height_building h1 h2 h3
  use 55
  split
  · rw [h1, h2, h3]
    norm_num
  · rfl
  sorry

end building_shadow_length_l794_794419


namespace largest_base_5_five_digits_base_10_value_l794_794035

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l794_794035


namespace sequence_periodicity_l794_794781

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : a 1 = 6 / 7)
  (h_rec : ∀ n, 0 ≤ a n ∧ a n < 1 → a (n+1) = if a n ≤ 1/2 then 2 * a n else 2 * a n - 1) :
  a 2017 = 6 / 7 :=
  sorry

end sequence_periodicity_l794_794781


namespace sum_of_solutions_eq_zero_l794_794519

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | -12*x/(x^2-1) = 3*x/(x+1) - 9/(x-1)}, x) = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l794_794519


namespace part1_part2_part3_l794_794220

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (m : ℝ) := x + m
noncomputable def F (x : ℝ) (m : ℝ) := f x - g x m

theorem part1 (x : ℝ) (h : x > 0) : 
  (x > 1 → (∃ c, (∀ y > 1, F y c < F x c))) ∧ (0 < x ∧ x < 1 → (∃ c, (∀ y, 0 < y ∧ y < 1 → F y c > F x c))) :=
sorry

theorem part2 (m : ℝ) (h : ∀ x > 0, f x ≤ g x m) : m ≥ -1 :=
sorry

theorem part3 (n : ℕ) (h : n > 0) :
  Real.log (n + 1) ≤ 2 + (∑ i in Finset.range n, 1 / (i + 2) : ℝ) - (n : ℝ) :=
sorry

end part1_part2_part3_l794_794220


namespace car_b_speed_l794_794459

theorem car_b_speed (v : ℕ) (h1 : ∀ (v : ℕ), CarA_speed = 3 * v)
                   (h2 : ∀ (time : ℕ), CarA_time = 6)
                   (h3 : ∀ (time : ℕ), CarB_time = 2)
                   (h4 : Car_total_distance = 1000) :
    v = 50 :=
by
  sorry

end car_b_speed_l794_794459


namespace probability_stopping_same_color_l794_794801

theorem probability_stopping_same_color :
  let total_socks := 2016
  let red_socks := 2
  let green_socks := 2
  let blue_socks := 2
  let magenta_socks := 2
  let lavender_socks := 2
  let neon_socks := 2
  let mauve_socks := 2
  let wisteria_socks := 2
  let copper_socks := 2000
  let favorable_outcomes_same_color := 
    (copper_socks * (copper_socks - 1) / 2) + 
    (red_socks * (red_socks - 1) / 2) +
    (green_socks * (green_socks - 1) / 2) +
    (blue_socks * (blue_socks - 1) / 2) +
    (magenta_socks * (magenta_socks - 1) / 2) +
    (lavender_socks * (lavender_socks - 1) / 2) +
    (neon_socks * (neon_socks - 1) / 2) +
    (mauve_socks * (mauve_socks - 1) / 2) +
    (wisteria_socks * (wisteria_socks - 1) / 2) +
    2 * 2 -- red-green pair
  let total_ways_to_draw_two_socks := total_socks * (total_socks - 1) / 2
  in (favorable_outcomes_same_color / total_ways_to_draw_two_socks) = 1999012 / 2031120 := 
sorry

end probability_stopping_same_color_l794_794801


namespace all_numbers_same_parity_in_tame_array_all_numbers_equal_in_turbo_tame_array_l794_794444

-- Definitions for tame and turbo tame arrays
def is_tame (a : list ℤ) (h_len : a.length = 13) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 13 → ∃ l1 l2, l1.sum = l2.sum ∧ l1 ++ l2 = (a.removeAt i)

def is_turbo_tame (a : list ℤ) (h_len : a.length = 13) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 13 → ∃ l1 l2, l1.length = 6 ∧ l2.length = 6 ∧ l1.sum = l2.sum ∧ l1 ++ l2 = (a.removeAt i)

-- Proof statements
theorem all_numbers_same_parity_in_tame_array (a : list ℤ) (h_len : a.length = 13) (h_tame : is_tame a h_len) : 
  ∀ i1 i2, 0 ≤ i1 ∧ i1 < 13 ∧ 0 ≤ i2 ∧ i2 < 13 → (a.nth i1) % 2 = (a.nth i2) % 2 :=
sorry

theorem all_numbers_equal_in_turbo_tame_array (a : list ℤ) (h_len : a.length = 13) (h_turbo_tame : is_turbo_tame a h_len) : 
  ∀ i1 i2, 0 ≤ i1 ∧ i1 < 13 ∧ 0 ≤ i2 ∧ i2 < 13 → a.nth i1 = a.nth i2 :=
sorry

end all_numbers_same_parity_in_tame_array_all_numbers_equal_in_turbo_tame_array_l794_794444


namespace sum_of_possible_values_of_n_l794_794140

theorem sum_of_possible_values_of_n (S : Finset ℕ) (hS : S = {n | ∃ k ≥ 3, (3 * k ≤ 80) ∧ (n = ⌊(1 + (80 - k)/ 2)⌋) }) : S.sum id = 469 :=
by
  sorry

end sum_of_possible_values_of_n_l794_794140


namespace find_lambda_l794_794231

variable (a b c : ℝ × ℝ)
variable (λ : ℝ)

def vector_a := (1, 2)
def vector_b := (3, 0)
def vector_c := (1, -2)

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda
  (h_a : a = vector_a)
  (h_b : b = vector_b)
  (h_c : c = vector_c)
  (h_perpendicular : is_perpendicular (a.1 + λ * b.1, a.2 + λ * b.2) c) :
  λ = 1 := sorry

end find_lambda_l794_794231


namespace max_dominoes_in_checkered_plane_l794_794754

def checkered_plane := {p : ℕ × ℕ // p.1 % 2 ≠ p.2 % 2}

def domino := {s : set (ℕ × ℕ) // ∃ a b, s = {(a,b), (a+1,b)} ∨ s = {(a,b), (a,b+1)}}

def max_k_100x100 : ℕ :=
  50 * (2 * 50 - 1)

theorem max_dominoes_in_checkered_plane : ∀ (S : set (ℕ × ℕ)) (h : ∀ p ∈ S, p ∈ checkered_plane)
  (hs : S = finset.image (λ (i : ℕ), (i % 100, i / 100)) (finset.range 10000))
  , ∃ D : set domino, (∀ d ∈ D, d ⊆ S) ∧ max_k_100x100 ≤ D.card :=
by {
  -- Proof to be constructed
  sorry
}

end max_dominoes_in_checkered_plane_l794_794754


namespace sum_of_roots_l794_794523

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in ({√3, -√3} : set ℝ), x

theorem sum_of_roots :
  (∀ x : ℝ, (x ≠ 1) ∧ (x ≠ -1) → ( -12 * x) / (x ^ 2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  sum_of_solutions = 0 :=
sorry

end sum_of_roots_l794_794523


namespace trajectory_is_ellipse_l794_794995

-- Definitions of Points F1 and F2
variable (F1 F2 M : Point)

-- Distances between the points
variable (d1 d2 : ℝ)

-- Given conditions
axiom hF1F2 : dist F1 F2 = 8
axiom hMF1plusMF2 : dist M F1 + dist M F2 = 10

-- Objective: Prove that the trajectory of M forms an ellipse
theorem trajectory_is_ellipse (hF1F2 : dist F1 F2 = 8) (hMF1plusMF2 : dist M F1 + dist M F2 = 10) : is_ellipse (trajectory M) :=
sorry

end trajectory_is_ellipse_l794_794995


namespace find_z_l794_794963

theorem find_z (z : ℂ) (hz : (complex.abs z - 2 * complex.I) * (2 + complex.I) = 6 - 2 * complex.I) :
  z = complex.sqrt 3 + complex.I :=
sorry

end find_z_l794_794963


namespace angle_problem_l794_794457

-- Definitions for degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Adding two angles
def add_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := a1.minutes + a2.minutes
  let extra_degrees := total_minutes / 60
  { degrees := a1.degrees + a2.degrees + extra_degrees,
    minutes := total_minutes % 60 }

-- Subtracting two angles
def sub_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := if a1.minutes < a2.minutes then a1.minutes + 60 else a1.minutes
  let extra_deg := if a1.minutes < a2.minutes then 1 else 0
  { degrees := a1.degrees - a2.degrees - extra_deg,
    minutes := total_minutes - a2.minutes }

-- Multiplying an angle by a constant
def mul_angle (a : Angle) (k : ℕ) : Angle :=
  let total_minutes := a.minutes * k
  let extra_degrees := total_minutes / 60
  { degrees := a.degrees * k + extra_degrees,
    minutes := total_minutes % 60 }

-- Given angles
def angle1 : Angle := { degrees := 24, minutes := 31}
def angle2 : Angle := { degrees := 62, minutes := 10}

-- Prove the problem statement
theorem angle_problem : sub_angles (mul_angle angle1 4) angle2 = { degrees := 35, minutes := 54} :=
  sorry

end angle_problem_l794_794457


namespace arccos_zero_eq_pi_div_two_l794_794471

theorem arccos_zero_eq_pi_div_two : arccos 0 = π / 2 :=
by
  -- We know from trigonometric identities that cos (π / 2) = 0
  have h_cos : cos (π / 2) = 0 := sorry,
  -- Hence arccos 0 should equal π / 2 because that's the angle where cosine is 0
  exact sorry

end arccos_zero_eq_pi_div_two_l794_794471
