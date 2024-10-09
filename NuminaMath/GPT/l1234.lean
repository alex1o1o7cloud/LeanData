import Mathlib

namespace min_value_z_l1234_123471

variable (x y : ℝ)

theorem min_value_z : ∃ (x y : ℝ), 2 * x + 3 * y = 9 :=
sorry

end min_value_z_l1234_123471


namespace forty_percent_of_number_is_240_l1234_123419

-- Define the conditions as assumptions in Lean
variable (N : ℝ)
variable (h1 : (1/4) * (1/3) * (2/5) * N = 20)

-- Prove that 40% of the number N is 240
theorem forty_percent_of_number_is_240 (h1: (1/4) * (1/3) * (2/5) * N = 20) : 0.40 * N = 240 :=
  sorry

end forty_percent_of_number_is_240_l1234_123419


namespace find_k_l1234_123466

theorem find_k (k : ℝ) : 
  (k - 10) / (-8) = (5 - k) / (-8) → k = 7.5 :=
by
  intro h
  let slope1 := (k - 10) / (-8)
  let slope2 := (5 - k) / (-8)
  have h_eq : slope1 = slope2 := h
  sorry

end find_k_l1234_123466


namespace ten_thousand_times_ten_thousand_l1234_123488

theorem ten_thousand_times_ten_thousand :
  10000 * 10000 = 100000000 :=
by
  sorry

end ten_thousand_times_ten_thousand_l1234_123488


namespace Emily_age_is_23_l1234_123432

variable (UncleBob Daniel Emily Zoe : ℕ)

-- Conditions
axiom h1 : UncleBob = 54
axiom h2 : Daniel = UncleBob / 2
axiom h3 : Emily = Daniel - 4
axiom h4 : Emily = 2 * Zoe / 3

-- Question: Prove that Emily's age is 23
theorem Emily_age_is_23 : Emily = 23 :=
by
  sorry

end Emily_age_is_23_l1234_123432


namespace rectangle_diagonal_l1234_123420

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end rectangle_diagonal_l1234_123420


namespace ratio_of_administrators_to_teachers_l1234_123407

-- Define the conditions
def graduates : ℕ := 50
def parents_per_graduate : ℕ := 2
def teachers : ℕ := 20
def total_chairs : ℕ := 180

-- Calculate intermediate values
def parents : ℕ := graduates * parents_per_graduate
def graduates_and_parents_chairs : ℕ := graduates + parents
def total_graduates_parents_teachers_chairs : ℕ := graduates_and_parents_chairs + teachers
def administrators : ℕ := total_chairs - total_graduates_parents_teachers_chairs

-- Specify the theorem to prove the ratio of administrators to teachers
theorem ratio_of_administrators_to_teachers : administrators / teachers = 1 / 2 :=
by
  -- Proof is omitted; placeholder 'sorry'
  sorry

end ratio_of_administrators_to_teachers_l1234_123407


namespace smallest_z_value_l1234_123437

-- Definitions: w, x, y, and z as consecutive even positive integers
def consecutive_even_cubes (w x y z : ℤ) : Prop :=
  w % 2 = 0 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  w < x ∧ x < y ∧ y < z ∧
  x = w + 2 ∧ y = x + 2 ∧ z = y + 2

-- Problem statement: Smallest possible value of z
theorem smallest_z_value :
  ∃ w x y z : ℤ, consecutive_even_cubes w x y z ∧ w^3 + x^3 + y^3 = z^3 ∧ z = 12 :=
by
  sorry

end smallest_z_value_l1234_123437


namespace find_c_of_parabola_l1234_123480

theorem find_c_of_parabola 
  (a b c : ℝ)
  (h_eq : ∀ y, -3 = a * (y - 1)^2 + b * (y - 1) - 3)
  (h1 : -1 = a * (3 - 1)^2 + b * (3 - 1) - 3) :
  c = -5/2 := by
  sorry

end find_c_of_parabola_l1234_123480


namespace problem_statement_l1234_123416

theorem problem_statement (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (r s : ℕ)
  (consecutive_primes : Nat.Prime r ∧ Nat.Prime s ∧ (r + 1 = s ∨ s + 1 = r))
  (roots_condition : r + s = p ∧ r * s = 2 * q) :
  (r * s = 2 * q) ∧ (Nat.Prime (p^2 - 2 * q)) ∧ (Nat.Prime (p + 2 * q)) :=
by
  sorry

end problem_statement_l1234_123416


namespace expansion_correct_l1234_123433

-- Define the polynomials
def poly1 (z : ℤ) : ℤ := 3 * z^2 + 4 * z - 5
def poly2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2

-- Define the expected expanded polynomial
def expanded_poly (z : ℤ) : ℤ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- The theorem that proves the equivalence of the expanded form
theorem expansion_correct (z : ℤ) : (poly1 z) * (poly2 z) = expanded_poly z := by
  sorry

end expansion_correct_l1234_123433


namespace num_factors_of_M_l1234_123468

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end num_factors_of_M_l1234_123468


namespace hyperbola_center_l1234_123490

theorem hyperbola_center :
  (∃ h k : ℝ,
    (∀ x y : ℝ, ((4 * x - 8) / 9)^2 - ((5 * y + 5) / 7)^2 = 1 ↔ (x - h)^2 / (81 / 16) - (y - k)^2 / (49 / 25) = 1) ∧
    (h = 2) ∧ (k = -1)) :=
sorry

end hyperbola_center_l1234_123490


namespace total_families_l1234_123400

theorem total_families (F_2dogs F_1dog F_2cats total_animals total_families : ℕ) 
  (h1: F_2dogs = 15)
  (h2: F_1dog = 20)
  (h3: total_animals = 80)
  (h4: 2 * F_2dogs + F_1dog + 2 * F_2cats = total_animals) :
  total_families = F_2dogs + F_1dog + F_2cats := 
by 
  sorry

end total_families_l1234_123400


namespace monotonicity_of_f_range_of_a_l1234_123423

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.log x - x^2 + a

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f a x ≤ f a (x - 1)) ∧ 
           (a > 0 → ((x < Real.sqrt a → f a x ≤ f a (x + 1)) ∨ 
                     (x > Real.sqrt a → f a x ≥ f a (x - 1))))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 1) := sorry

end monotonicity_of_f_range_of_a_l1234_123423


namespace only_triple_l1234_123426

theorem only_triple (a b c : ℕ) (h1 : (a * b + 1) % c = 0)
                                (h2 : (a * c + 1) % b = 0)
                                (h3 : (b * c + 1) % a = 0) :
    (a = 1 ∧ b = 1 ∧ c = 1) :=
by
  sorry

end only_triple_l1234_123426


namespace solve_inequality_l1234_123470

theorem solve_inequality (x : ℝ) : -4 * x - 8 > 0 → x < -2 := sorry

end solve_inequality_l1234_123470


namespace function_monotonicity_l1234_123434

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x) / (b^x + c^x) + (b^x) / (a^x + c^x) + (c^x) / (a^x + b^x)

theorem function_monotonicity (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, y ≤ x → x < 0 → f a b c x ≤ f a b c y) :=
by
  sorry

end function_monotonicity_l1234_123434


namespace min_value_expr_l1234_123448

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l1234_123448


namespace sphere_diagonal_property_l1234_123491

variable {A B C D : ℝ}

-- conditions provided
variable (radius : ℝ) (x y z : ℝ)
variable (h_radius : radius = 1)
variable (h_non_coplanar : ¬(is_coplanar A B C D))
variable (h_AB_CD : dist A B = x ∧ dist C D = x)
variable (h_BC_DA : dist B C = y ∧ dist D A = y)
variable (h_CA_BD : dist C A = z ∧ dist B D = z)

theorem sphere_diagonal_property :
  x^2 + y^2 + z^2 = 8 := 
sorry

end sphere_diagonal_property_l1234_123491


namespace cos_C_correct_l1234_123443

noncomputable def cos_C (B : ℝ) (AD BD : ℝ) : ℝ :=
  let sinB := Real.sin B
  let angleBAC := (2 : ℝ) * Real.arcsin ((Real.sqrt 3 / 3) * (sinB / 2)) -- derived from bisector property.
  let cosA := (2 : ℝ) * Real.cos angleBAC / 2 - 1
  let sinA := 2 * Real.sin angleBAC / 2 * Real.cos angleBAC / 2
  let cos2thirds := -1 / 2
  let sin2thirds := Real.sqrt 3 / 2
  cos2thirds * cosA + sin2thirds * sinA

theorem cos_C_correct : 
  ∀ (π : ℝ), 
  ∀ (A B C : ℝ),
  B = π / 3 →
  ∀ (AD : ℝ), AD = 3 →
  ∀ (BD : ℝ), BD = 2 →
  cos_C B AD BD = (2 * Real.sqrt 6 - 1) / 6 :=
by
  intros π A B C hB angleBisectorI hAD hBD
  sorry

end cos_C_correct_l1234_123443


namespace solve_arithmetic_sequence_l1234_123494

variable {a : ℕ → ℝ}
variable {d a1 a2 a3 a10 a11 a6 a7 : ℝ}

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a1 + n * d

def arithmetic_condition (h : a 2 + a 3 + a 10 + a 11 = 32) : Prop :=
  a 6 + a 7 = 16

theorem solve_arithmetic_sequence (h : a 2 + a 3 + a 10 + a 11 = 32) : a 6 + a 7 = 16 :=
  by
    -- Proof will go here
    sorry

end solve_arithmetic_sequence_l1234_123494


namespace simplify_expression_l1234_123483

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end simplify_expression_l1234_123483


namespace f_2_equals_12_l1234_123409

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12_l1234_123409


namespace gem_stone_necklaces_sold_l1234_123474

-- Definitions and conditions
def bead_necklaces : ℕ := 7
def total_earnings : ℝ := 90
def price_per_necklace : ℝ := 9

-- Theorem to prove the number of gem stone necklaces sold
theorem gem_stone_necklaces_sold : 
  ∃ (G : ℕ), G * price_per_necklace = total_earnings - (bead_necklaces * price_per_necklace) ∧ G = 3 :=
by
  sorry

end gem_stone_necklaces_sold_l1234_123474


namespace find_f_7_l1234_123493

noncomputable def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

theorem find_f_7 : f 7 = 17 :=
  by
  -- The proof steps will go here
  sorry

end find_f_7_l1234_123493


namespace greatest_diff_l1234_123446

theorem greatest_diff (x y : ℤ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) : y - x = 7 :=
sorry

end greatest_diff_l1234_123446


namespace jessica_deposit_fraction_l1234_123456

theorem jessica_deposit_fraction (init_balance withdraw_amount final_balance : ℝ)
  (withdraw_fraction remaining_fraction deposit_fraction : ℝ) :
  remaining_fraction = withdraw_fraction - (2/5) → 
  init_balance * withdraw_fraction = init_balance - withdraw_amount →
  init_balance * remaining_fraction + deposit_fraction * (init_balance * remaining_fraction) = final_balance →
  init_balance = 500 →
  final_balance = 450 →
  withdraw_amount = 200 →
  remaining_fraction = (3/5) →
  deposit_fraction = 1/2 :=
by
  intros hr hw hrb hb hf hwamount hr_remain
  sorry

end jessica_deposit_fraction_l1234_123456


namespace train_length_is_150_l1234_123405

-- Let length_of_train be the length of the train in meters
def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_s

theorem train_length_is_150 (speed_kmh time_s : ℕ) (h_speed : speed_kmh = 180) (h_time : time_s = 3) :
  length_of_train speed_kmh time_s = 150 := by
  sorry

end train_length_is_150_l1234_123405


namespace prime_sum_square_mod_3_l1234_123478

theorem prime_sum_square_mod_3 (p : Fin 100 → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_distinct : Function.Injective p) :
  let N := (Finset.univ : Finset (Fin 100)).sum (λ i => (p i)^2)
  N % 3 = 1 := by
  sorry

end prime_sum_square_mod_3_l1234_123478


namespace valid_differences_of_squares_l1234_123479

theorem valid_differences_of_squares (n : ℕ) (h : 2 * n + 1 < 150) :
    (2 * n + 1 = 129 ∨ 2 * n +1 = 147) :=
by
  sorry

end valid_differences_of_squares_l1234_123479


namespace golden_section_length_l1234_123467

noncomputable def golden_section_point (a b : ℝ) := a / (a + b) = b / a

theorem golden_section_length (A B P : ℝ) (h : golden_section_point A P) (hAP_gt_PB : A > P) (hAB : A + P = 2) : 
  A = Real.sqrt 5 - 1 :=
by
  -- Proof goes here
  sorry

end golden_section_length_l1234_123467


namespace cost_price_is_700_l1234_123403

noncomputable def cost_price_was_700 : Prop :=
  ∃ (CP : ℝ),
    (∀ (SP1 SP2 : ℝ),
      SP1 = CP * 0.84 ∧
        SP2 = CP * 1.04 ∧
        SP2 = SP1 + 140) ∧
    CP = 700

theorem cost_price_is_700 : cost_price_was_700 :=
  sorry

end cost_price_is_700_l1234_123403


namespace value_of_a_l1234_123477

theorem value_of_a (x a : ℤ) (h : x = 4) (h_eq : 5 * (x - 1) - 3 * a = -3) : a = 6 :=
by {
  sorry
}

end value_of_a_l1234_123477


namespace brandon_cards_l1234_123481

theorem brandon_cards (b m : ℕ) 
  (h1 : m = b + 8) 
  (h2 : 14 = m / 2) : 
  b = 20 := by
  sorry

end brandon_cards_l1234_123481


namespace smallest_n_between_76_and_100_l1234_123428

theorem smallest_n_between_76_and_100 :
  ∃ (n : ℕ), (n > 1) ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 5 = 1) ∧ (76 < n) ∧ (n < 100) :=
sorry

end smallest_n_between_76_and_100_l1234_123428


namespace B_investment_is_72000_l1234_123430

noncomputable def A_investment : ℝ := 27000
noncomputable def C_investment : ℝ := 81000
noncomputable def C_profit : ℝ := 36000
noncomputable def total_profit : ℝ := 80000

noncomputable def B_investment : ℝ :=
  let total_investment := (C_investment * total_profit) / C_profit
  total_investment - A_investment - C_investment

theorem B_investment_is_72000 :
  B_investment = 72000 :=
by
  sorry

end B_investment_is_72000_l1234_123430


namespace base4_sum_correct_l1234_123444

/-- Define the base-4 numbers as natural numbers. -/
def a := 3 * 4^2 + 1 * 4^1 + 2 * 4^0
def b := 3 * 4^1 + 1 * 4^0
def c := 3 * 4^0

/-- Define their sum in base 10. -/
def sum_base_10 := a + b + c

/-- Define the target sum in base 4 as a natural number. -/
def target := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

/-- Prove that the sum of the base-4 numbers equals the target sum in base 4. -/
theorem base4_sum_correct : sum_base_10 = target := by
  sorry

end base4_sum_correct_l1234_123444


namespace team_a_wins_at_least_2_l1234_123473

def team_a_wins_at_least (total_games lost_games : ℕ) (points : ℕ) (won_points draw_points lost_points : ℕ) : Prop :=
  ∃ (won_games : ℕ), 
    total_games = won_games + (total_games - lost_games - won_games) + lost_games ∧
    won_games * won_points + (total_games - lost_games - won_games) * draw_points > points ∧
    won_games ≥ 2

theorem team_a_wins_at_least_2 :
  team_a_wins_at_least 5 1 7 3 1 0 :=
by
  -- Proof goes here
  sorry

end team_a_wins_at_least_2_l1234_123473


namespace units_digit_47_pow_47_l1234_123485

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end units_digit_47_pow_47_l1234_123485


namespace min_value_exprB_four_min_value_exprC_four_l1234_123451

noncomputable def exprB (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def exprC (x : ℝ) : ℝ := 1 / (Real.sin x)^2 + 1 / (Real.cos x)^2

theorem min_value_exprB_four : ∃ x : ℝ, exprB x = 4 := sorry

theorem min_value_exprC_four : ∃ x : ℝ, exprC x = 4 := sorry

end min_value_exprB_four_min_value_exprC_four_l1234_123451


namespace cos_alpha_third_quadrant_l1234_123439

theorem cos_alpha_third_quadrant (α : ℝ) (hα1 : π < α ∧ α < 3 * π / 2) (hα2 : Real.tan α = 4 / 3) :
  Real.cos α = -3 / 5 :=
sorry

end cos_alpha_third_quadrant_l1234_123439


namespace centroid_tetrahedron_l1234_123445

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D M : V)

def is_centroid (M A B C D : V) : Prop :=
  M = (1/4:ℝ) • (A + B + C + D)

theorem centroid_tetrahedron (h : is_centroid M A B C D) :
  (M - A) + (M - B) + (M - C) + (M - D) = (0 : V) :=
by {
  sorry
}

end centroid_tetrahedron_l1234_123445


namespace expected_value_of_difference_is_4_point_5_l1234_123429

noncomputable def expected_value_difference : ℚ :=
  (2 * 6 / 56 + 3 * 10 / 56 + 4 * 12 / 56 + 5 * 12 / 56 + 6 * 10 / 56 + 7 * 6 / 56)

theorem expected_value_of_difference_is_4_point_5 :
  expected_value_difference = 4.5 := sorry

end expected_value_of_difference_is_4_point_5_l1234_123429


namespace application_methods_count_l1234_123498

theorem application_methods_count (n_graduates m_universities : ℕ) (h_graduates : n_graduates = 5) (h_universities : m_universities = 3) :
  (m_universities ^ n_graduates) = 243 :=
by
  rw [h_graduates, h_universities]
  show 3 ^ 5 = 243
  sorry

end application_methods_count_l1234_123498


namespace no_positive_integer_satisfies_inequality_l1234_123469

theorem no_positive_integer_satisfies_inequality :
  ∀ x : ℕ, 0 < x → ¬ (15 < -3 * (x : ℤ) + 18) := by
  sorry

end no_positive_integer_satisfies_inequality_l1234_123469


namespace minimum_A_l1234_123417

noncomputable def minA : ℝ := (1 + Real.sqrt 2) / 2

theorem minimum_A (x y z w : ℝ) (A : ℝ) 
    (h : xy + 2 * yz + zw ≤ A * (x^2 + y^2 + z^2 + w^2)) :
    A ≥ minA := 
sorry

end minimum_A_l1234_123417


namespace range_of_m_l1234_123461

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem range_of_m (m : ℝ) : f m > 1 → m < 0 := by
  sorry

end range_of_m_l1234_123461


namespace right_triangle_side_lengths_l1234_123496

theorem right_triangle_side_lengths (a b c : ℝ) (varrho r : ℝ) (h_varrho : varrho = 8) (h_r : r = 41) : 
  (a = 80 ∧ b = 18 ∧ c = 82) ∨ (a = 18 ∧ b = 80 ∧ c = 82) :=
by
  sorry

end right_triangle_side_lengths_l1234_123496


namespace will_net_calorie_intake_is_600_l1234_123414

-- Given conditions translated into Lean definitions and assumptions
def breakfast_calories : ℕ := 900
def jogging_time_minutes : ℕ := 30
def calories_burned_per_minute : ℕ := 10

-- Proof statement in Lean
theorem will_net_calorie_intake_is_600 :
  breakfast_calories - (jogging_time_minutes * calories_burned_per_minute) = 600 :=
by
  sorry

end will_net_calorie_intake_is_600_l1234_123414


namespace min_height_of_cuboid_l1234_123408

theorem min_height_of_cuboid (h : ℝ) (side_len : ℝ) (small_spheres_r : ℝ) (large_sphere_r : ℝ) :
  side_len = 4 → 
  small_spheres_r = 1 → 
  large_sphere_r = 2 → 
  ∃ h_min : ℝ, h_min = 2 + 2 * Real.sqrt 7 ∧ h ≥ h_min := 
by
  sorry

end min_height_of_cuboid_l1234_123408


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l1234_123497

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l1234_123497


namespace quadratic_distinct_real_roots_l1234_123453

theorem quadratic_distinct_real_roots (k : ℝ) :
  (k > -2 ∧ k ≠ 0) ↔ ( ∃ (a b c : ℝ), a = k ∧ b = -4 ∧ c = -2 ∧ (b^2 - 4 * a * c) > 0) :=
by
  sorry

end quadratic_distinct_real_roots_l1234_123453


namespace difference_of_numbers_l1234_123412

/-- Given two natural numbers a and 10a whose sum is 23,320,
prove that the difference between them is 19,080. -/
theorem difference_of_numbers (a : ℕ) (h : a + 10 * a = 23320) : 10 * a - a = 19080 := by
  sorry

end difference_of_numbers_l1234_123412


namespace coloring_satisfies_conditions_l1234_123425

-- Define lattice points as points with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the color function
def color (p : LatticePoint) : ℕ :=
  if (p.x % 2 = 0) ∧ (p.y % 2 = 1) then 0 -- Black
  else if (p.x % 2 = 1) ∧ (p.y % 2 = 0) then 1 -- White
  else 2 -- Red

-- Define condition (1)
def infinite_lines_with_color (c : ℕ) : Prop :=
  ∀ k : ℤ, ∃ p : LatticePoint, color p = c ∧ p.x = k

-- Define condition (2)
def parallelogram_exists (A B C : LatticePoint) (wc rc bc : ℕ) : Prop :=
  (color A = wc) ∧ (color B = rc) ∧ (color C = bc) →
  ∃ D : LatticePoint, color D = rc ∧ D.x = C.x + (A.x - B.x) ∧ D.y = C.y + (A.y - B.y)

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : ℕ, ∃ p : LatticePoint, infinite_lines_with_color c) ∧
  (∀ A B C : LatticePoint, ∃ wc rc bc : ℕ, parallelogram_exists A B C wc rc bc) :=
sorry

end coloring_satisfies_conditions_l1234_123425


namespace part_a_part_b_l1234_123492

variable {A B C A₁ B₁ C₁ : Prop}
variables {a b c a₁ b₁ c₁ S S₁ : ℝ}

-- Assume basic conditions of triangles
variable (h1 : IsTriangle A B C)
variable (h2 : IsTriangleWithCentersAndSquares A B C A₁ B₁ C₁ a b c a₁ b₁ c₁ S S₁)
variable (h3 : IsExternalSquaresConstructed A B C A₁ B₁ C₁)

-- Part (a)
theorem part_a : a₁^2 + b₁^2 + c₁^2 = a^2 + b^2 + c^2 + 6 * S := 
sorry

-- Part (b)
theorem part_b : S₁ - S = (a^2 + b^2 + c^2) / 8 := 
sorry

end part_a_part_b_l1234_123492


namespace percentage_vanaspati_after_adding_ghee_l1234_123476

theorem percentage_vanaspati_after_adding_ghee :
  ∀ (original_quantity new_pure_ghee percentage_ghee percentage_vanaspati : ℝ),
    original_quantity = 30 →
    percentage_ghee = 0.5 →
    percentage_vanaspati = 0.5 →
    new_pure_ghee = 20 →
    (percentage_vanaspati * original_quantity) /
    (original_quantity + new_pure_ghee) * 100 = 30 :=
by
  intros original_quantity new_pure_ghee percentage_ghee percentage_vanaspati
  sorry

end percentage_vanaspati_after_adding_ghee_l1234_123476


namespace complex_multiplication_l1234_123438

variable (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_multiplication :
  i * (2 * i - 1) = -2 - i :=
  sorry

end complex_multiplication_l1234_123438


namespace three_zeros_implies_a_lt_neg3_l1234_123411

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l1234_123411


namespace sampling_methods_correct_l1234_123401

def company_sales_outlets (A B C D : ℕ) : Prop :=
  A = 150 ∧ B = 120 ∧ C = 180 ∧ D = 150 ∧ A + B + C + D = 600

def investigation_samples (total_samples large_outlets region_C_sample : ℕ) : Prop :=
  total_samples = 100 ∧ large_outlets = 20 ∧ region_C_sample = 7

def appropriate_sampling_methods (investigation1_method investigation2_method : String) : Prop :=
  investigation1_method = "Stratified sampling" ∧ investigation2_method = "Simple random sampling"

theorem sampling_methods_correct :
  company_sales_outlets 150 120 180 150 →
  investigation_samples 100 20 7 →
  appropriate_sampling_methods "Stratified sampling" "Simple random sampling" :=
by
  intros h1 h2
  sorry

end sampling_methods_correct_l1234_123401


namespace smallest_x_y_sum_l1234_123486

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end smallest_x_y_sum_l1234_123486


namespace num_real_roots_of_eq_l1234_123462

theorem num_real_roots_of_eq (x : ℝ) (h : x * |x| - 3 * |x| - 4 = 0) : 
  ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 :=
sorry

end num_real_roots_of_eq_l1234_123462


namespace arithmetic_contains_geometric_progression_l1234_123489

theorem arithmetic_contains_geometric_progression (a d : ℕ) (h_pos : d > 0) :
  ∃ (a' : ℕ) (r : ℕ), a' = a ∧ r = 1 + d ∧ (∀ k : ℕ, ∃ n : ℕ, a' * r^k = a + (n-1)*d) :=
by
  sorry

end arithmetic_contains_geometric_progression_l1234_123489


namespace negation_of_universal_proposition_l1234_123495

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 > 1) ↔ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 ≤ 1 := 
sorry

end negation_of_universal_proposition_l1234_123495


namespace five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l1234_123463

-- Define what "5 PM" and "10 PM" mean in hours
def five_pm: ℕ := 17
def ten_pm: ℕ := 22

-- Define function for converting from PM to 24-hour time
def pm_to_hours (n: ℕ): ℕ := n + 12

-- Define the times in minutes for comparison
def time_16_40: ℕ := 16 * 60 + 40
def time_17_20: ℕ := 17 * 60 + 20

-- Define the differences in minutes
def minutes_passed (start end_: ℕ): ℕ := end_ - start

-- Prove the equivalences
theorem five_pm_is_seventeen_hours: pm_to_hours 5 = five_pm := by 
  unfold pm_to_hours
  unfold five_pm
  rfl

theorem ten_pm_is_twenty_two_hours: pm_to_hours 10 = ten_pm := by 
  unfold pm_to_hours
  unfold ten_pm
  rfl

theorem time_difference_is_forty_minutes: minutes_passed time_16_40 time_17_20 = 40 := by 
  unfold time_16_40
  unfold time_17_20
  unfold minutes_passed
  rfl

#check five_pm_is_seventeen_hours
#check ten_pm_is_twenty_two_hours
#check time_difference_is_forty_minutes

end five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l1234_123463


namespace total_money_l1234_123455

theorem total_money (John Alice Bob : ℝ) (hJohn : John = 5 / 8) (hAlice : Alice = 7 / 20) (hBob : Bob = 1 / 4) :
  John + Alice + Bob = 1.225 := 
by 
  sorry

end total_money_l1234_123455


namespace inverse_100_mod_101_l1234_123458

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l1234_123458


namespace negation_correct_l1234_123410

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct_l1234_123410


namespace expand_polynomial_l1234_123460

variable (x : ℝ)

theorem expand_polynomial :
  2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 :=
by
  sorry

end expand_polynomial_l1234_123460


namespace capacity_of_each_bag_is_approximately_63_l1234_123421

noncomputable def capacity_of_bag (total_sand : ℤ) (num_bags : ℤ) : ℤ :=
  Int.ceil (total_sand / num_bags)

theorem capacity_of_each_bag_is_approximately_63 :
  capacity_of_bag 757 12 = 63 :=
by
  sorry

end capacity_of_each_bag_is_approximately_63_l1234_123421


namespace cost_of_cherries_l1234_123472

theorem cost_of_cherries (total_spent amount_for_grapes amount_for_cherries : ℝ)
  (h1 : total_spent = 21.93)
  (h2 : amount_for_grapes = 12.08)
  (h3 : amount_for_cherries = total_spent - amount_for_grapes) :
  amount_for_cherries = 9.85 :=
sorry

end cost_of_cherries_l1234_123472


namespace problem_statement_l1234_123402

variable (a : ℝ)

theorem problem_statement (h : 5 = a + a⁻¹) : a^4 + (a⁻¹)^4 = 527 := 
by 
  sorry

end problem_statement_l1234_123402


namespace juice_cost_l1234_123436

-- Given conditions
def sandwich_cost : ℝ := 0.30
def total_money : ℝ := 2.50
def num_friends : ℕ := 4

-- Cost calculation
def total_sandwich_cost : ℝ := num_friends * sandwich_cost
def remaining_money : ℝ := total_money - total_sandwich_cost

-- The theorem to prove
theorem juice_cost : (remaining_money / num_friends) = 0.325 := by
  sorry

end juice_cost_l1234_123436


namespace min_value_of_expression_l1234_123447

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l1234_123447


namespace ab_finish_job_in_15_days_l1234_123418

theorem ab_finish_job_in_15_days (A B C : ℝ) (h1 : A + B + C = 1/12) (h2 : C = 1/60) : 1 / (A + B) = 15 := 
by
  sorry

end ab_finish_job_in_15_days_l1234_123418


namespace num_candidates_l1234_123424

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l1234_123424


namespace multiplication_of_fractions_l1234_123450

theorem multiplication_of_fractions :
  (77 / 4) * (5 / 2) = 48 + 1 / 8 := 
sorry

end multiplication_of_fractions_l1234_123450


namespace fish_population_estimation_l1234_123404

-- Definitions based on conditions
def fish_tagged_day1 : ℕ := 80
def fish_caught_day2 : ℕ := 100
def fish_tagged_day2 : ℕ := 20
def fish_caught_day3 : ℕ := 120
def fish_tagged_day3 : ℕ := 36

-- The average percentage of tagged fish caught on the second and third days
def avg_tag_percentage : ℚ := (20 / 100 + 36 / 120) / 2

-- Statement of the proof problem
theorem fish_population_estimation :
  (avg_tag_percentage * P = fish_tagged_day1) → 
  P = 320 :=
by
  -- Proof goes here
  sorry

end fish_population_estimation_l1234_123404


namespace arithmetic_sequence_first_term_l1234_123422

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_terms_int : ∀ n, ∃ k : ℤ, a n = k) 
  (ha20 : a 20 = 205) : a 1 = 91 :=
sorry

end arithmetic_sequence_first_term_l1234_123422


namespace pizza_area_increase_l1234_123427

theorem pizza_area_increase (A1 A2 r1 r2 : ℝ) (r1_eq : r1 = 7) (r2_eq : r2 = 5) (A1_eq : A1 = Real.pi * r1^2) (A2_eq : A2 = Real.pi * r2^2) :
  ((A1 - A2) / A2) * 100 = 96 := by
  sorry

end pizza_area_increase_l1234_123427


namespace cyclist_waits_15_minutes_l1234_123440

-- Definitions
def hiker_rate := 7 -- miles per hour
def cyclist_rate := 28 -- miles per hour
def wait_time := 15 / 60 -- hours, as the cyclist waits 15 minutes, converted to hours

-- The statement to be proven
theorem cyclist_waits_15_minutes :
  ∃ t : ℝ, t = 15 / 60 ∧
  (∀ d : ℝ, d = (hiker_rate * wait_time) →
            d = (cyclist_rate * t - hiker_rate * t)) :=
by
  sorry

end cyclist_waits_15_minutes_l1234_123440


namespace dilution_problem_l1234_123454

/-- Samantha needs to add 7.2 ounces of water to achieve a 25% alcohol concentration
given that she starts with 12 ounces of solution containing 40% alcohol. -/
theorem dilution_problem (x : ℝ) : (12 + x) * 0.25 = 4.8 ↔ x = 7.2 :=
by sorry

end dilution_problem_l1234_123454


namespace math_problem_l1234_123459

theorem math_problem (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 :=
by
  sorry

end math_problem_l1234_123459


namespace proof_l1234_123441

noncomputable def proof_problem (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem proof (
  a b c : ℝ
) (h1 : (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 = 3 * (a^3 - b^3) * (b^3 - c^3) * (c^3 - a^3))
  (h2 : (a - b)^3 + (b - c)^3 + (c - a)^3 = 3 * (a - b) * (b - c) * (c - a)) :
  proof_problem a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end proof_l1234_123441


namespace find_number_mul_l1234_123415

theorem find_number_mul (n : ℕ) (h : n * 9999 = 724777430) : n = 72483 :=
by
  sorry

end find_number_mul_l1234_123415


namespace xy_value_l1234_123442

theorem xy_value (x y : ℝ) (h₁ : x + y = 2) (h₂ : x^2 * y^3 + y^2 * x^3 = 32) :
  x * y = 2^(5/3) :=
by
  sorry

end xy_value_l1234_123442


namespace triangle_area_qin_jiushao_l1234_123449

theorem triangle_area_qin_jiushao (a b c : ℝ) (h1: a = 2) (h2: b = 3) (h3: c = Real.sqrt 13) :
  Real.sqrt ((1 / 4) * (a^2 * b^2 - (1 / 4) * (a^2 + b^2 - c^2)^2)) = 3 :=
by
  -- Hypotheses
  rw [h1, h2, h3]
  sorry

end triangle_area_qin_jiushao_l1234_123449


namespace pages_per_donut_l1234_123457

def pages_written (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ) : ℕ :=
  let donuts := total_calories / calories_per_donut
  total_pages / donuts

theorem pages_per_donut (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ): 
  total_pages = 12 → calories_per_donut = 150 → total_calories = 900 → pages_written total_pages calories_per_donut total_calories = 2 := by
  intros
  sorry

end pages_per_donut_l1234_123457


namespace semicircle_radius_l1234_123484

theorem semicircle_radius (P : ℝ) (r : ℝ) (h₁ : P = π * r + 2 * r) (h₂ : P = 198) :
  r = 198 / (π + 2) :=
sorry

end semicircle_radius_l1234_123484


namespace nth_term_series_l1234_123464

def a_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem nth_term_series (n : ℕ) : a_n n = 1.5 + 5.5 * (-1) ^ n :=
by
  sorry

end nth_term_series_l1234_123464


namespace number_of_red_squares_in_19th_row_l1234_123482

-- Define the number of squares in the n-th row
def number_of_squares (n : ℕ) : ℕ := 3 * n - 1

-- Define the number of red squares in the n-th row
def red_squares (n : ℕ) : ℕ := (number_of_squares n) / 2

-- The theorem stating the problem
theorem number_of_red_squares_in_19th_row : red_squares 19 = 28 := by
  -- Proof goes here
  sorry

end number_of_red_squares_in_19th_row_l1234_123482


namespace max_value_of_expression_l1234_123487

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l1234_123487


namespace jerry_birthday_games_l1234_123406

def jerry_original_games : ℕ := 7
def jerry_total_games_after_birthday : ℕ := 9
def games_jerry_got_for_birthday (original total : ℕ) : ℕ := total - original

theorem jerry_birthday_games :
  games_jerry_got_for_birthday jerry_original_games jerry_total_games_after_birthday = 2 := by
  sorry

end jerry_birthday_games_l1234_123406


namespace lowest_fraction_done_in_an_hour_by_two_people_l1234_123475

def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 5
def c_rate : ℚ := 1 / 6

theorem lowest_fraction_done_in_an_hour_by_two_people : 
  min (min (a_rate + b_rate) (a_rate + c_rate)) (b_rate + c_rate) = 11 / 30 := 
by
  sorry

end lowest_fraction_done_in_an_hour_by_two_people_l1234_123475


namespace area_increase_correct_l1234_123431

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l1234_123431


namespace intersection_lines_l1234_123413

theorem intersection_lines (c d : ℝ) (h1 : 6 = 2 * 4 + c) (h2 : 6 = 5 * 4 + d) : c + d = -16 := 
by
  sorry

end intersection_lines_l1234_123413


namespace same_parity_iff_exists_c_d_l1234_123465

theorem same_parity_iff_exists_c_d (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a % 2 = b % 2) ↔ ∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2 := 
by 
  sorry

end same_parity_iff_exists_c_d_l1234_123465


namespace present_age_of_B_l1234_123452

theorem present_age_of_B :
  ∃ (A B : ℕ), (A + 20 = 2 * (B - 20)) ∧ (A = B + 10) ∧ (B = 70) :=
by
  sorry

end present_age_of_B_l1234_123452


namespace sums_same_remainder_exists_l1234_123499

theorem sums_same_remainder_exists (n : ℕ) (h : n > 0) (a : Fin (2 * n) → Fin (2 * n)) (ha_permutation : Function.Bijective a) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i) % (2 * n) = (a j + j) % (2 * n)) :=
by sorry

end sums_same_remainder_exists_l1234_123499


namespace problem_l1234_123435

def seq (a : ℕ → ℝ) := a 0 = 1 / 2 ∧ ∀ n > 0, a n = a (n - 1) + (1 / n^2) * (a (n - 1))^2

theorem problem (a : ℕ → ℝ) (n : ℕ) (h_seq : seq a) (h_n_pos : n > 0) :
  (1 / a (n - 1) - 1 / a n < 1 / n^2) ∧
  (∀ n > 0, a n < n) ∧
  (∀ n > 0, 1 / a n < 5 / 6 + 1 / (n + 1)) :=
by
  sorry

end problem_l1234_123435
