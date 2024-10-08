import Mathlib

namespace sin_of_7pi_over_6_l157_157694

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l157_157694


namespace max_min_diff_c_l157_157925

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l157_157925


namespace unreachable_y_l157_157782

noncomputable def y_function (x : ℝ) : ℝ := (2 - 3 * x) / (5 * x - 1)

theorem unreachable_y : ¬ ∃ x : ℝ, y_function x = -3 / 5 ∧ x ≠ 1 / 5 :=
by {
  sorry
}

end unreachable_y_l157_157782


namespace g_h_of_2_eq_2340_l157_157269

def g (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_of_2_eq_2340 : g (h 2) = 2340 := 
  sorry

end g_h_of_2_eq_2340_l157_157269


namespace mitya_age_l157_157809

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l157_157809


namespace inequality_proof_l157_157096

theorem inequality_proof
  (a b c d : ℝ)
  (hpos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (hcond: (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2 * a + b + c) * (2 * b + c + d) * (2 * c + d + a) * (2 * d + a + b) * (a * b * c * d) ^ 2 ≤ 1 / 16 := 
by
  sorry

end inequality_proof_l157_157096


namespace simplify_expression_l157_157389

variable (a b c x : ℝ)

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def p (x a b c : ℝ) : ℝ :=
  (x - a)^3/(a - b)*(a - c) + a*x +
  (x - b)^3/(b - a)*(b - c) + b*x +
  (x - c)^3/(c - a)*(c - b) + c*x

theorem simplify_expression (h : distinct a b c) :
  p x a b c = a + b + c + 3*x + 1 := by
  sorry

end simplify_expression_l157_157389


namespace autumn_grain_purchase_exceeds_1_8_billion_tons_l157_157469

variable (x : ℝ)

theorem autumn_grain_purchase_exceeds_1_8_billion_tons 
  (h : x > 0.18) : 
  x > 1.8 := 
by 
  sorry

end autumn_grain_purchase_exceeds_1_8_billion_tons_l157_157469


namespace day_of_50th_in_year_N_minus_1_l157_157878

theorem day_of_50th_in_year_N_minus_1
  (N : ℕ)
  (day250_in_year_N_is_sunday : (250 % 7 = 0))
  (day150_in_year_N_plus_1_is_sunday : (150 % 7 = 0))
  : 
  (50 % 7 = 1) := 
sorry

end day_of_50th_in_year_N_minus_1_l157_157878


namespace simplify_expression_l157_157818

theorem simplify_expression (x : ℝ) (h : x^2 + x - 6 = 0) : 
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 :=
sorry

end simplify_expression_l157_157818


namespace value_of_expression_l157_157829

theorem value_of_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) : 
  (x^4 + 3 * y^3 + 10) / 7 = 283 / 7 := by
  sorry

end value_of_expression_l157_157829


namespace probability_at_least_2_defective_is_one_third_l157_157952

noncomputable def probability_at_least_2_defective (good defective : ℕ) (total_selected : ℕ) : ℚ :=
  let total_ways := Nat.choose (good + defective) total_selected
  let ways_2_defective_1_good := Nat.choose defective 2 * Nat.choose good 1
  let ways_3_defective := Nat.choose defective 3
  (ways_2_defective_1_good + ways_3_defective) / total_ways

theorem probability_at_least_2_defective_is_one_third :
  probability_at_least_2_defective 6 4 3 = 1 / 3 :=
by
  sorry

end probability_at_least_2_defective_is_one_third_l157_157952


namespace rationalize_denominator_l157_157995

theorem rationalize_denominator : 
  let a := 32
  let b := 8
  let c := 2
  let d := 4
  (a / (c * Real.sqrt c) + b / (d * Real.sqrt c)) = (9 * Real.sqrt c) :=
by
  sorry

end rationalize_denominator_l157_157995


namespace Oshea_needs_30_small_planters_l157_157266

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l157_157266


namespace base_conversion_l157_157122

theorem base_conversion (b : ℝ) (h : 2 * b^2 + 3 = 51) : b = 2 * Real.sqrt 6 :=
by
  sorry

end base_conversion_l157_157122


namespace cards_per_set_is_13_l157_157171

-- Definitions based on the conditions
def total_cards : ℕ := 365
def sets_to_brother : ℕ := 8
def sets_to_sister : ℕ := 5
def sets_to_friend : ℕ := 2
def total_sets_given : ℕ := sets_to_brother + sets_to_sister + sets_to_friend
def total_cards_given : ℕ := 195

-- The problem to prove
theorem cards_per_set_is_13 : total_cards_given / total_sets_given = 13 :=
  by
  -- Here we would provide the proof, but for now, we use sorry
  sorry

end cards_per_set_is_13_l157_157171


namespace sin_25_over_6_pi_l157_157275

noncomputable def sin_value : ℝ :=
  Real.sin (25 / 6 * Real.pi)

theorem sin_25_over_6_pi : sin_value = 1 / 2 := by
  sorry

end sin_25_over_6_pi_l157_157275


namespace cube_remainder_l157_157790

theorem cube_remainder (n : ℤ) (h : n % 13 = 5) : (n^3) % 17 = 6 :=
by
  sorry

end cube_remainder_l157_157790


namespace range_of_a_l157_157104

open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7 * x - 18 < 0}

theorem range_of_a (a : ℝ) : A a ⊆ B → (-2 : ℝ) ≤ a ∧ a ≤ 9 :=
by sorry

end range_of_a_l157_157104


namespace evaluate_expression_l157_157258

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end evaluate_expression_l157_157258


namespace min_m_for_four_elements_l157_157371

open Set

theorem min_m_for_four_elements (n : ℕ) (hn : n ≥ 2) :
  ∃ m, m = 2 * n + 2 ∧ 
  (∀ (S : Finset ℕ), S.card = m → 
    (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a = b + c + d)) :=
by
  sorry

end min_m_for_four_elements_l157_157371


namespace smallest_constant_c_l157_157353

def satisfies_conditions (f : ℝ → ℝ) :=
  ∀ ⦃x : ℝ⦄, (0 ≤ x ∧ x ≤ 1) → (f x ≥ 0 ∧ (x = 1 → f 1 = 1) ∧
  (∀ y, 0 ≤ y → y ≤ 1 → x + y ≤ 1 → f x + f y ≤ f (x + y)))

theorem smallest_constant_c :
  ∀ {f : ℝ → ℝ},
  satisfies_conditions f →
  ∃ c : ℝ, (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x) ∧
  (∀ c', c' < 2 → ∃ x, 0 ≤ x → x ≤ 1 ∧ f x > c' * x) :=
by sorry

end smallest_constant_c_l157_157353


namespace fraction_defined_l157_157005

theorem fraction_defined (x : ℝ) : (1 - 2 * x ≠ 0) ↔ (x ≠ 1 / 2) :=
by sorry

end fraction_defined_l157_157005


namespace total_weekly_sleep_correct_l157_157399

-- Definition of the weekly sleep time for cougar, zebra, and lion
def cougar_sleep_even_days : Nat := 4
def cougar_sleep_odd_days : Nat := 6
def zebra_sleep_even_days := (cougar_sleep_even_days + 2)
def zebra_sleep_odd_days := (cougar_sleep_odd_days + 2)
def lion_sleep_even_days := (zebra_sleep_even_days - 3)
def lion_sleep_odd_days := (cougar_sleep_odd_days + 1)

def total_weekly_sleep_time : Nat :=
  (4 * cougar_sleep_odd_days + 3 * cougar_sleep_even_days) + -- Cougar's total sleep in a week
  (4 * zebra_sleep_odd_days + 3 * zebra_sleep_even_days) + -- Zebra's total sleep in a week
  (4 * lion_sleep_odd_days + 3 * lion_sleep_even_days) -- Lion's total sleep in a week

theorem total_weekly_sleep_correct : total_weekly_sleep_time = 123 := 
by
  -- Total for the week according to given conditions
  sorry -- Proof is omitted, only the statement is required

end total_weekly_sleep_correct_l157_157399


namespace no_snuggly_two_digit_l157_157588

theorem no_snuggly_two_digit (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) : ¬ (10 * a + b = a + b^3) :=
by {
  sorry
}

end no_snuggly_two_digit_l157_157588


namespace find_f4_l157_157953

theorem find_f4 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f (-x + 1)) 
  (h2 : ∀ x, f (x - 1) = f (-x - 1)) 
  (h3 : f 0 = 2) : 
  f 4 = -2 :=
sorry

end find_f4_l157_157953


namespace initial_persons_count_l157_157379

open Real

def average_weight_increase (n : ℕ) (increase_per_person : ℝ) : ℝ :=
  increase_per_person * n

def weight_difference (new_weight old_weight : ℝ) : ℝ :=
  new_weight - old_weight

theorem initial_persons_count :
  ∀ (n : ℕ),
  average_weight_increase n 2.5 = weight_difference 95 75 → n = 8 :=
by
  intro n h
  sorry

end initial_persons_count_l157_157379


namespace contrapositive_even_addition_l157_157793

theorem contrapositive_even_addition (a b : ℕ) :
  (¬((a % 2 = 0) ∧ (b % 2 = 0)) → (a + b) % 2 ≠ 0) :=
sorry

end contrapositive_even_addition_l157_157793


namespace find_y_for_line_slope_45_degrees_l157_157458

theorem find_y_for_line_slope_45_degrees :
  ∃ y, (∃ x₁ y₁ x₂ y₂, x₁ = 4 ∧ y₁ = y ∧ x₂ = 2 ∧ y₂ = -3 ∧ (y₂ - y₁) / (x₂ - x₁) = 1) → y = -1 :=
by
  sorry

end find_y_for_line_slope_45_degrees_l157_157458


namespace theorem1_theorem2_theorem3_l157_157099

-- Given conditions as definitions
variables {x y p q : ℝ}

-- Condition definitions
def condition1 : x + y = -p := sorry
def condition2 : x * y = q := sorry

-- Theorems to be proved
theorem theorem1 (h1 : x + y = -p) (h2 : x * y = q) : x^2 + y^2 = p^2 - 2 * q := sorry

theorem theorem2 (h1 : x + y = -p) (h2 : x * y = q) : x^3 + y^3 = -p^3 + 3 * p * q := sorry

theorem theorem3 (h1 : x + y = -p) (h2 : x * y = q) : x^4 + y^4 = p^4 - 4 * p^2 * q + 2 * q^2 := sorry

end theorem1_theorem2_theorem3_l157_157099


namespace carol_used_tissue_paper_l157_157034

theorem carol_used_tissue_paper (initial_pieces : ℕ) (remaining_pieces : ℕ) (usage: ℕ)
  (h1 : initial_pieces = 97)
  (h2 : remaining_pieces = 93)
  (h3: usage = initial_pieces - remaining_pieces) : 
  usage = 4 :=
by
  -- We only need to set up the problem; proof can be provided later.
  sorry

end carol_used_tissue_paper_l157_157034


namespace ratio_of_screams_to_hours_l157_157889

-- Definitions from conditions
def hours_hired : ℕ := 6
def current_babysitter_rate : ℕ := 16
def new_babysitter_rate : ℕ := 12
def extra_charge_per_scream : ℕ := 3
def cost_difference : ℕ := 18

-- Calculate necessary costs
def current_babysitter_cost : ℕ := current_babysitter_rate * hours_hired
def new_babysitter_base_cost : ℕ := new_babysitter_rate * hours_hired
def new_babysitter_total_cost : ℕ := current_babysitter_cost - cost_difference
def screams_cost : ℕ := new_babysitter_total_cost - new_babysitter_base_cost
def number_of_screams : ℕ := screams_cost / extra_charge_per_scream

-- Theorem to prove the ratio
theorem ratio_of_screams_to_hours : number_of_screams / hours_hired = 1 := by
  sorry

end ratio_of_screams_to_hours_l157_157889


namespace age_problem_l157_157850

theorem age_problem :
  (∃ (x y : ℕ), 
    (3 * x - 7 = 5 * (x - 7)) ∧ 
    (42 + y = 2 * (14 + y)) ∧ 
    (2 * x = 28) ∧ 
    (x = 14) ∧ 
    (3 * 14 = 42) ∧ 
    (42 - 14 = 28) ∧ 
    (y = 14)) :=
by
  sorry

end age_problem_l157_157850


namespace cone_volume_from_half_sector_l157_157214

theorem cone_volume_from_half_sector (R : ℝ) (V : ℝ) : 
  R = 6 →
  V = (1/3) * Real.pi * (R / 2)^2 * (R * Real.sqrt 3) →
  V = 9 * Real.pi * Real.sqrt 3 := by sorry

end cone_volume_from_half_sector_l157_157214


namespace find_constants_l157_157924

theorem find_constants (t s : ℤ) :
  (∀ x : ℤ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s) →
  t = -2 ∧ s = s :=
by
  intros h
  sorry

end find_constants_l157_157924


namespace factor_polynomial_l157_157866

theorem factor_polynomial :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 = 
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := 
by sorry

end factor_polynomial_l157_157866


namespace circle_equation_value_l157_157420

theorem circle_equation_value (a : ℝ) :
  (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 → False) → a = -1 :=
by
  intros h
  sorry

end circle_equation_value_l157_157420


namespace distinct_real_numbers_condition_l157_157243

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end distinct_real_numbers_condition_l157_157243


namespace circle_equation_l157_157360

theorem circle_equation : 
  ∃ (b : ℝ), (∀ (x y : ℝ), (x^2 + (y - b)^2 = 1 ↔ (x = 1 ∧ y = 2) → b = 2)) :=
sorry

end circle_equation_l157_157360


namespace retail_price_eq_120_l157_157083

noncomputable def retail_price : ℝ :=
  let W := 90
  let P := 0.20 * W
  let SP := W + P
  SP / 0.90

theorem retail_price_eq_120 : retail_price = 120 := by
  sorry

end retail_price_eq_120_l157_157083


namespace some_number_value_correct_l157_157627

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l157_157627


namespace line_equation_l157_157246

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end line_equation_l157_157246


namespace proof_problem_l157_157533

/- Define relevant concepts -/
def is_factor (a b : Nat) := ∃ k, b = a * k
def is_divisor := is_factor

/- Given conditions with their translations -/
def condition_A : Prop := is_factor 5 35
def condition_B : Prop := is_divisor 21 252 ∧ ¬ is_divisor 21 48
def condition_C : Prop := ¬ (is_divisor 15 90 ∨ is_divisor 15 74)
def condition_D : Prop := is_divisor 18 36 ∧ ¬ is_divisor 18 72
def condition_E : Prop := is_factor 9 180

/- The main proof problem statement -/
theorem proof_problem : condition_A ∧ condition_B ∧ ¬ condition_C ∧ ¬ condition_D ∧ condition_E :=
by
  sorry

end proof_problem_l157_157533


namespace find_rolls_of_toilet_paper_l157_157600

theorem find_rolls_of_toilet_paper (visits : ℕ) (squares_per_visit : ℕ) (squares_per_roll : ℕ) (days : ℕ)
  (h_visits : visits = 3)
  (h_squares_per_visit : squares_per_visit = 5)
  (h_squares_per_roll : squares_per_roll = 300)
  (h_days : days = 20000) : (visits * squares_per_visit * days) / squares_per_roll = 1000 :=
by
  sorry

end find_rolls_of_toilet_paper_l157_157600


namespace least_negative_b_l157_157656

theorem least_negative_b (x b : ℤ) (h1 : x^2 + b * x = 22) (h2 : b < 0) : b = -21 :=
sorry

end least_negative_b_l157_157656


namespace floor_ceil_eq_l157_157455

theorem floor_ceil_eq (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 :=
by
  sorry

end floor_ceil_eq_l157_157455


namespace garden_width_l157_157400

theorem garden_width (w : ℕ) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end garden_width_l157_157400


namespace necessary_but_not_sufficient_l157_157909

theorem necessary_but_not_sufficient (a : ℝ) : (a ≠ 1) → (a^2 ≠ 1) → (a ≠ 1) ∧ ¬((a ≠ 1) → (a^2 ≠ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l157_157909


namespace inequality_transform_l157_157959

theorem inequality_transform (x y : ℝ) (h : y > x) : 2 * y > 2 * x := 
  sorry

end inequality_transform_l157_157959


namespace find_value_l157_157154

theorem find_value (x y z : ℝ) (h₁ : y = 3 * x) (h₂ : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end find_value_l157_157154


namespace radius_of_inscribed_circle_l157_157769

theorem radius_of_inscribed_circle (a b x : ℝ) (hx : 0 < x) 
  (h_side_length : a > 20) 
  (h_TM : a = x + 8) 
  (h_OM : b = x + 9) 
  (h_Pythagorean : (a - 8)^2 + (b - 9)^2 = x^2) :
  x = 29 :=
by
  -- Assume all conditions and continue to the proof part.
  sorry

end radius_of_inscribed_circle_l157_157769


namespace inverse_matrix_eigenvalues_l157_157967

theorem inverse_matrix_eigenvalues 
  (c d : ℝ) 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (eigenvalue1 eigenvalue2 : ℝ) 
  (eigenvector1 eigenvector2 : Fin 2 → ℝ) :
  A = ![![1, 2], ![c, d]] →
  eigenvalue1 = 2 →
  eigenvalue2 = 3 →
  eigenvector1 = ![2, 1] →
  eigenvector2 = ![1, 1] →
  (A.vecMul eigenvector1 = (eigenvalue1 • eigenvector1)) →
  (A.vecMul eigenvector2 = (eigenvalue2 • eigenvector2)) →
  A⁻¹ = ![![2 / 3, -1 / 3], ![1 / 6, 1 / 6]] :=
sorry

end inverse_matrix_eigenvalues_l157_157967


namespace equation_of_circle_min_distance_PA_PB_l157_157581

-- Definition of the given points, lines, and circle
def point (x y : ℝ) : Prop := true

def circle_through_points (x1 y1 x2 y2 x3 y3 : ℝ) (a b r : ℝ) : Prop :=
  (x1 + a) * (x1 + a) + y1 * y1 = r ∧
  (x2 + a) * (x2 + a) + y2 * y2 = r ∧
  (x3 + a) * (x3 + a) + y3 * y3 = r

def line (a b : ℝ) : Prop := true

-- Specific points
def D := point 0 1
def E := point (-2) 1
def F := point (-1) (Real.sqrt 2)

-- Lines l1 and l2
def l₁ (x : ℝ) : ℝ := x - 2
def l₂ (x : ℝ) : ℝ := x + 1

-- Intersection points A and B
def A := point 0 1
def B := point (-2) (-1)

-- Question Ⅰ: Find the equation of the circle
theorem equation_of_circle :
  ∃ a b r, circle_through_points 0 1 (-2) 1 (-1) (Real.sqrt 2) a b r ∧ (a = -1 ∧ b = 0 ∧ r = 2) :=
  sorry

-- Question Ⅱ: Find the minimum value of |PA|^2 + |PB|^2
def dist_sq (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

theorem min_distance_PA_PB :
  real := sorry

end equation_of_circle_min_distance_PA_PB_l157_157581


namespace solve_x_l157_157451

noncomputable def x : ℝ := 4.7

theorem solve_x : (10 - x) ^ 2 = x ^ 2 + 6 :=
by
  sorry

end solve_x_l157_157451


namespace pentagon_product_condition_l157_157576

theorem pentagon_product_condition :
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 ∧
  ∃ (a' b' c' d' e' : ℝ), 
    (a', b', c', d', e') ∈ {perm | perm = (a, b, c, d, e) ∨ perm = (b, c, d, e, a) ∨ perm = (c, d, e, a, b) ∨ perm = (d, e, a, b, c) ∨ perm = (e, a, b, c, d)} ∧
    (a'*b' ≤ 1/9 ∧ b'*c' ≤ 1/9 ∧ c'*d' ≤ 1/9 ∧ d'*e' ≤ 1/9 ∧ e'*a' ≤ 1/9) := sorry

end pentagon_product_condition_l157_157576


namespace possible_values_for_D_l157_157687

noncomputable def distinct_digit_values (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  B < 10 ∧ A < 10 ∧ D < 10 ∧ C < 10 ∧ C = 9 ∧ (B + A = 9 + D)

theorem possible_values_for_D :
  ∃ (Ds : Finset Nat), (∀ D ∈ Ds, ∃ A B C, distinct_digit_values A B C D) ∧
  Ds.card = 5 :=
sorry

end possible_values_for_D_l157_157687


namespace yellow_marbles_l157_157188

-- Define the conditions from a)
variables (total_marbles red blue green yellow : ℕ)
variables (h1 : total_marbles = 110)
variables (h2 : red = 8)
variables (h3 : blue = 4 * red)
variables (h4 : green = 2 * blue)
variables (h5 : yellow = total_marbles - (red + blue + green))

-- Prove the question in c)
theorem yellow_marbles : yellow = 6 :=
by
  -- Proof will be inserted here
  sorry

end yellow_marbles_l157_157188


namespace sum_of_interior_angles_at_vertex_A_l157_157816

-- Definitions of the interior angles for a square and a regular octagon.
def square_interior_angle : ℝ := 90
def octagon_interior_angle : ℝ := 135

-- Theorem that states the sum of the interior angles at vertex A formed by the square and octagon.
theorem sum_of_interior_angles_at_vertex_A : square_interior_angle + octagon_interior_angle = 225 := by
  sorry

end sum_of_interior_angles_at_vertex_A_l157_157816


namespace no_such_primes_l157_157876

theorem no_such_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_three : p > 3) (hq_gt_three : q > 3) (hq_div_p2_minus_1 : q ∣ (p^2 - 1)) 
  (hp_div_q2_minus_1 : p ∣ (q^2 - 1)) : false := 
sorry

end no_such_primes_l157_157876


namespace smallest_range_between_allocations_l157_157060

-- Problem statement in Lean
theorem smallest_range_between_allocations :
  ∀ (A B C D E : ℕ), 
  (A = 30000) →
  (B < 18000 ∨ B > 42000) →
  (C < 18000 ∨ C > 42000) →
  (D < 58802 ∨ D > 82323) →
  (E < 58802 ∨ E > 82323) →
  min B (min C (min D E)) = 17999 →
  max B (max C (max D E)) = 82323 →
  82323 - 17999 = 64324 :=
by
  intros A B C D E hA hB hC hD hE hmin hmax
  sorry

end smallest_range_between_allocations_l157_157060


namespace angle_size_proof_l157_157828

-- Define the problem conditions
def fifteen_points_on_circle (θ : ℕ) : Prop :=
  θ = 360 / 15 

-- Define the central angles
def central_angle_between_adjacent_points (θ : ℕ) : ℕ :=
  360 / 15  

-- Define the two required central angles
def central_angle_A1O_A3 (θ : ℕ) : ℕ :=
  2 * θ

def central_angle_A3O_A7 (θ : ℕ) : ℕ :=
  4 * θ

-- Define the problem using the given conditions and the proven answer
noncomputable def angle_A1_A3_A7 : ℕ :=
  108

-- Lean 4 statement of the math problem to prove
theorem angle_size_proof (θ : ℕ) (h1 : fifteen_points_on_circle θ) :
  central_angle_A1O_A3 θ = 48 ∧ central_angle_A3O_A7 θ = 96 → 
  angle_A1_A3_A7 = 108 :=
by sorry

#check angle_size_proof

end angle_size_proof_l157_157828


namespace suzhou_visitors_accuracy_l157_157720

/--
In Suzhou, during the National Day holiday in 2023, the city received 17.815 million visitors.
Given that number, prove that it is accurate to the thousands place.
-/
theorem suzhou_visitors_accuracy :
  (17.815 : ℝ) * 10^6 = 17815000 ∧ true := 
by
sorry

end suzhou_visitors_accuracy_l157_157720


namespace find_M_l157_157167

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the complement of M with respect to U
def complement_M : Set ℕ := {2}

-- Define M as U without the complement of M
def M : Set ℕ := U \ complement_M

-- Prove that M is {0, 1, 3}
theorem find_M : M = {0, 1, 3} := by
  sorry

end find_M_l157_157167


namespace total_supermarkets_FGH_chain_l157_157242

def supermarkets_us : ℕ := 47
def supermarkets_difference : ℕ := 10
def supermarkets_canada : ℕ := supermarkets_us - supermarkets_difference
def total_supermarkets : ℕ := supermarkets_us + supermarkets_canada

theorem total_supermarkets_FGH_chain : total_supermarkets = 84 :=
by 
  sorry

end total_supermarkets_FGH_chain_l157_157242


namespace range_of_a_l157_157637

def p (a x : ℝ) : Prop := a * x^2 + a * x - 1 < 0
def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

theorem range_of_a (a : ℝ) :
  ¬ (∀ x, p a x ∨ q a) → a ≤ -4 ∨ 1 ≤ a :=
by sorry

end range_of_a_l157_157637


namespace employee_pay_l157_157755

variable (X Y Z : ℝ)

-- Conditions
def X_pay (Y : ℝ) := 1.2 * Y
def Z_pay (X : ℝ) := 0.75 * X

-- Proof statement
theorem employee_pay (h1 : X = X_pay Y) (h2 : Z = Z_pay X) (total_pay : X + Y + Z = 1540) : 
  X + Y + Z = 1540 :=
by
  sorry

end employee_pay_l157_157755


namespace ratio_a_to_b_l157_157989

variable (a x c d b : ℝ)
variable (h1 : d = 3 * x + c)
variable (h2 : b = 4 * x)

theorem ratio_a_to_b : a / b = -1 / 4 := by 
  sorry

end ratio_a_to_b_l157_157989


namespace birds_in_sky_l157_157523

theorem birds_in_sky (wings total_wings : ℕ) (h1 : total_wings = 26) (h2 : wings = 2) : total_wings / wings = 13 := 
by
  sorry

end birds_in_sky_l157_157523


namespace inequality_proof_l157_157874

variable {x1 x2 y1 y2 z1 z2 : ℝ}

theorem inequality_proof (hx1 : x1 > 0) (hx2 : x2 > 0)
   (hxy1 : x1 * y1 - z1^2 > 0) (hxy2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
  sorry

end inequality_proof_l157_157874


namespace increased_percentage_l157_157897

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l157_157897


namespace final_temperature_is_58_32_l157_157599

-- Initial temperature
def T₀ : ℝ := 40

-- Sequence of temperature adjustments
def T₁ : ℝ := 2 * T₀
def T₂ : ℝ := T₁ - 30
def T₃ : ℝ := T₂ * (1 - 0.30)
def T₄ : ℝ := T₃ + 24
def T₅ : ℝ := T₄ * (1 - 0.10)
def T₆ : ℝ := T₅ + 8
def T₇ : ℝ := T₆ * (1 + 0.20)
def T₈ : ℝ := T₇ - 15

-- Proof statement
theorem final_temperature_is_58_32 : T₈ = 58.32 :=
by sorry

end final_temperature_is_58_32_l157_157599


namespace correct_masks_l157_157017

def elephant_mask := 6
def mouse_mask := 4
def pig_mask := 8
def panda_mask := 1

theorem correct_masks :
  (elephant_mask = 6) ∧
  (mouse_mask = 4) ∧
  (pig_mask = 8) ∧
  (panda_mask = 1) := 
by
  sorry

end correct_masks_l157_157017


namespace arithmetic_sequence_properties_l157_157368

theorem arithmetic_sequence_properties
    (n s1 s2 s3 : ℝ)
    (h1 : s1 = 8)
    (h2 : s2 = 50)
    (h3 : s3 = 134)
    (h4 : n = 8) :
    n^2 * s3 - 3 * n * s1 * s2 + 2 * s1^2 = 0 := 
by {
  sorry
}

end arithmetic_sequence_properties_l157_157368


namespace ellipse_range_of_k_l157_157351

theorem ellipse_range_of_k (k : ℝ) :
  (4 - k > 0) → (k - 1 > 0) → (4 - k ≠ k - 1) → (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_of_k_l157_157351


namespace find_divisor_l157_157473

theorem find_divisor :
  ∃ D : ℝ, 527652 = (D * 392.57) + 48.25 ∧ D = 1344.25 :=
by
  sorry

end find_divisor_l157_157473


namespace number_of_diamonds_in_F10_l157_157554

def sequence_of_figures (F : ℕ → ℕ) : Prop :=
  F 1 = 4 ∧
  (∀ n ≥ 2, F n = F (n-1) + 4 * (n + 2)) ∧
  F 3 = 28

theorem number_of_diamonds_in_F10 (F : ℕ → ℕ) (h : sequence_of_figures F) : F 10 = 336 :=
by
  sorry

end number_of_diamonds_in_F10_l157_157554


namespace mile_time_sum_is_11_l157_157150

def mile_time_sum (Tina_time Tony_time Tom_time : ℕ) : ℕ :=
  Tina_time + Tony_time + Tom_time

theorem mile_time_sum_is_11 :
  ∃ (Tina_time Tony_time Tom_time : ℕ),
  (Tina_time = 6 ∧ Tony_time = Tina_time / 2 ∧ Tom_time = Tina_time / 3) →
  mile_time_sum Tina_time Tony_time Tom_time = 11 :=
by
  sorry

end mile_time_sum_is_11_l157_157150


namespace original_average_l157_157229

theorem original_average (A : ℝ) (h : 5 * A = 130) : A = 26 :=
by
  have h1 : 5 * A / 5 = 130 / 5 := by sorry
  sorry

end original_average_l157_157229


namespace original_number_of_friends_l157_157235

theorem original_number_of_friends (F : ℕ) (h₁ : 5000 / F - 125 = 5000 / (F + 8)) : F = 16 :=
sorry

end original_number_of_friends_l157_157235


namespace apple_harvest_l157_157263

theorem apple_harvest (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 →
  num_sections = 8 →
  total_sacks = sacks_per_section * num_sections →
  total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end apple_harvest_l157_157263


namespace sqrt_of_square_is_identity_l157_157290

variable {a : ℝ} (h : a > 0)

theorem sqrt_of_square_is_identity (h : a > 0) : Real.sqrt (a^2) = a := 
  sorry

end sqrt_of_square_is_identity_l157_157290


namespace newer_model_distance_l157_157791

-- Given conditions
def older_model_distance : ℕ := 160
def newer_model_factor : ℝ := 1.25

-- The statement to be proved
theorem newer_model_distance :
  newer_model_factor * (older_model_distance : ℝ) = 200 := by
  sorry

end newer_model_distance_l157_157791


namespace find_stream_speed_l157_157733

-- Define the conditions
def boat_speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def upstream_time : ℝ := 1.5
def speed_of_stream (v : ℝ) : Prop :=
  let downstream_speed := boat_speed_in_still_water + v
  let upstream_speed := boat_speed_in_still_water - v
  (downstream_speed * downstream_time) = (upstream_speed * upstream_time)

-- Define the theorem to prove
theorem find_stream_speed : ∃ v, speed_of_stream v ∧ v = 3 :=
by {
  sorry
}

end find_stream_speed_l157_157733


namespace max_area_equilateral_in_rectangle_l157_157841

-- Define the dimensions of the rectangle
def length_efgh : ℕ := 15
def width_efgh : ℕ := 8

-- The maximum possible area of an equilateral triangle inscribed in the rectangle
theorem max_area_equilateral_in_rectangle : 
  ∃ (s : ℝ), 
  s = ((16 * Real.sqrt 3) / 3) ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ length_efgh → 
    (∃ (area : ℝ), area = (Real.sqrt 3 / 4 * s^2) ∧
      area = 64 * Real.sqrt 3)) :=
by sorry

end max_area_equilateral_in_rectangle_l157_157841


namespace total_marks_l157_157937

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l157_157937


namespace smallest_rectangle_area_contains_L_shape_l157_157088

-- Condition: Side length of each square
def side_length : ℕ := 8

-- Condition: Number of squares
def num_squares : ℕ := 6

-- The correct answer (to be proven equivalent)
def expected_area : ℕ := 768

-- The main theorem stating the expected proof problem
theorem smallest_rectangle_area_contains_L_shape 
  (side_length : ℕ) (num_squares : ℕ) (h_shape : side_length = 8 ∧ num_squares = 6) : 
  ∃area, area = expected_area :=
by
  sorry

end smallest_rectangle_area_contains_L_shape_l157_157088


namespace product_mod_five_remainder_l157_157542

theorem product_mod_five_remainder :
  (114 * 232 * 454 * 454 * 678) % 5 = 4 := by
  sorry

end product_mod_five_remainder_l157_157542


namespace new_person_weight_l157_157067

theorem new_person_weight (N : ℝ) (h : N - 65 = 22.5) : N = 87.5 :=
by
  sorry

end new_person_weight_l157_157067


namespace division_remainder_190_21_l157_157481

theorem division_remainder_190_21 :
  190 = 21 * 9 + 1 :=
sorry

end division_remainder_190_21_l157_157481


namespace probability_of_odd_divisor_l157_157911

noncomputable def factorial_prime_factors : ℕ → List (ℕ × ℕ)
| 21 => [(2, 18), (3, 9), (5, 4), (7, 3), (11, 1), (13, 1), (17, 1), (19, 1)]
| _ => []

def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

def number_of_odd_factors (factors : List (ℕ × ℕ)) : ℕ :=
  number_of_factors (factors.filter (λ ⟨p, _⟩ => p != 2))

theorem probability_of_odd_divisor : (number_of_odd_factors (factorial_prime_factors 21)) /
(number_of_factors (factorial_prime_factors 21)) = 1 / 19 := 
by
  sorry

end probability_of_odd_divisor_l157_157911


namespace probability_divisible_by_3_l157_157309

-- Define the set of numbers
def S : Set ℕ := {2, 3, 5, 6}

-- Define the pairs of numbers whose product is divisible by 3
def valid_pairs : Set (ℕ × ℕ) := {(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)}

-- Define the total number of pairs
def total_pairs := 6

-- Define the number of valid pairs
def valid_pairs_count := 5

-- Prove that the probability of choosing two numbers whose product is divisible by 3 is 5/6
theorem probability_divisible_by_3 : (valid_pairs_count / total_pairs : ℚ) = 5 / 6 := by
  sorry

end probability_divisible_by_3_l157_157309


namespace mary_flour_amount_l157_157281

noncomputable def cups_of_flour_already_put_in
    (total_flour_needed : ℕ)
    (total_sugar_needed : ℕ)
    (extra_flour_needed : ℕ)
    (flour_to_be_added : ℕ) : ℕ :=
total_flour_needed - (total_sugar_needed + extra_flour_needed)

theorem mary_flour_amount
    (total_flour_needed : ℕ := 9)
    (total_sugar_needed : ℕ := 6)
    (extra_flour_needed : ℕ := 1) :
    cups_of_flour_already_put_in total_flour_needed total_sugar_needed extra_flour_needed (total_sugar_needed + extra_flour_needed) = 2 := by
  sorry

end mary_flour_amount_l157_157281


namespace distance_to_x_axis_l157_157055

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : |P.2| = 2 := 
by sorry

end distance_to_x_axis_l157_157055


namespace extra_bananas_each_child_gets_l157_157009

-- Define the total number of students and the number of absent students
def total_students : ℕ := 260
def absent_students : ℕ := 130

-- Define the total number of bananas
variable (B : ℕ)

-- The proof statement
theorem extra_bananas_each_child_gets :
  ∀ B : ℕ, (B / (total_students - absent_students)) = (B / total_students) + (B / total_students) :=
by
  intro B
  sorry

end extra_bananas_each_child_gets_l157_157009


namespace equivalent_discount_l157_157566

theorem equivalent_discount {x : ℝ} (h₀ : x > 0) :
    let first_discount := 0.10
    let second_discount := 0.20
    let single_discount := 0.28
    (1 - (1 - first_discount) * (1 - second_discount)) = single_discount := by
    sorry

end equivalent_discount_l157_157566


namespace algebraic_expression_value_l157_157066

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 :=
by
  sorry

end algebraic_expression_value_l157_157066


namespace percentage_of_x_l157_157210

variable (x : ℝ)

theorem percentage_of_x (x : ℝ) : ((40 / 100) * (50 / 100) * x) = (20 / 100) * x := by
  sorry

end percentage_of_x_l157_157210


namespace sum_of_squares_l157_157212

def gcd (a b c : Nat) : Nat := (Nat.gcd (Nat.gcd a b) c)

theorem sum_of_squares {a b c : ℕ} (h1 : 3 * a + 2 * b = 4 * c)
                                   (h2 : 3 * c ^ 2 = 4 * a ^ 2 + 2 * b ^ 2)
                                   (h3 : gcd a b c = 1) :
  a^2 + b^2 + c^2 = 45 :=
by
  sorry

end sum_of_squares_l157_157212


namespace boat_speed_in_still_water_l157_157939

theorem boat_speed_in_still_water (x : ℕ) 
  (h1 : x + 17 = 77) (h2 : x - 17 = 43) : x = 60 :=
by
  sorry

end boat_speed_in_still_water_l157_157939


namespace discriminant_of_quadratic_eq_l157_157095

/-- The discriminant of a quadratic equation -/
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_eq : discriminant 1 3 (-1) = 13 := by
  sorry

end discriminant_of_quadratic_eq_l157_157095


namespace cosine_of_five_pi_over_three_l157_157901

theorem cosine_of_five_pi_over_three :
  Real.cos (5 * Real.pi / 3) = 1 / 2 :=
sorry

end cosine_of_five_pi_over_three_l157_157901


namespace value_of_x_l157_157090

theorem value_of_x (w : ℝ) (hw : w = 90) (z : ℝ) (hz : z = 2 / 3 * w) (y : ℝ) (hy : y = 1 / 4 * z) (x : ℝ) (hx : x = 1 / 2 * y) : x = 7.5 :=
by
  -- Proof skipped; conclusion derived from conditions
  sorry

end value_of_x_l157_157090


namespace isosceles_triangle_base_length_l157_157982

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : 2 * a + b = 24) : b = 10 := 
by 
  sorry

end isosceles_triangle_base_length_l157_157982


namespace king_william_probability_l157_157797

theorem king_william_probability :
  let m := 2
  let n := 15
  m + n = 17 :=
by
  sorry

end king_william_probability_l157_157797


namespace fraction_order_l157_157806

theorem fraction_order :
  (19 / 15 < 17 / 13) ∧ (17 / 13 < 15 / 11) :=
by
  sorry

end fraction_order_l157_157806


namespace original_workers_l157_157219

theorem original_workers (x y : ℝ) (h : x = (65 / 100) * y) : y = (20 / 13) * x :=
by sorry

end original_workers_l157_157219


namespace complement_intersection_l157_157753

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def N : Set ℝ := {x | (x < -3) ∨ (x > 0)}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | x < -3 ∨ x > 2} :=
by
  sorry

end complement_intersection_l157_157753


namespace determine_x_value_l157_157518

theorem determine_x_value (a b c x : ℕ) (h1 : x = a + 7) (h2 : a = b + 12) (h3 : b = c + 25) (h4 : c = 95) : x = 139 := by
  sorry

end determine_x_value_l157_157518


namespace julia_error_approx_97_percent_l157_157200

theorem julia_error_approx_97_percent (x : ℝ) : 
  abs ((6 * x - x / 6) / (6 * x) * 100 - 97) < 1 :=
by 
  sorry

end julia_error_approx_97_percent_l157_157200


namespace complex_div_symmetry_l157_157146

open Complex

-- Definitions based on conditions
def z1 : ℂ := 1 + I
def z2 : ℂ := -1 + I

-- Theorem to prove
theorem complex_div_symmetry : z2 / z1 = I := by
  sorry

end complex_div_symmetry_l157_157146


namespace plant_supplier_earnings_l157_157776

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end plant_supplier_earnings_l157_157776


namespace number_of_solutions_eq_one_l157_157532

theorem number_of_solutions_eq_one :
  ∃! (n : ℕ), 0 < n ∧ 
              (∃ k : ℕ, (n + 1500) = 90 * k ∧ k = Int.floor (Real.sqrt n)) :=
sorry

end number_of_solutions_eq_one_l157_157532


namespace find_x_l157_157834

noncomputable def e_squared := Real.exp 2

theorem find_x (x : ℝ) (h : Real.log (x^2 - 5*x + 10) = 2) :
  x = 4.4 ∨ x = 0.6 :=
sorry

end find_x_l157_157834


namespace vaishali_total_stripes_l157_157045

theorem vaishali_total_stripes
  (hats1 : ℕ) (stripes1 : ℕ)
  (hats2 : ℕ) (stripes2 : ℕ)
  (hats3 : ℕ) (stripes3 : ℕ)
  (hats4 : ℕ) (stripes4 : ℕ)
  (total_stripes : ℕ) :
  hats1 = 4 → stripes1 = 3 →
  hats2 = 3 → stripes2 = 4 →
  hats3 = 6 → stripes3 = 0 →
  hats4 = 2 → stripes4 = 5 →
  total_stripes = (hats1 * stripes1) + (hats2 * stripes2) + (hats3 * stripes3) + (hats4 * stripes4) →
  total_stripes = 34 := by
  sorry

end vaishali_total_stripes_l157_157045


namespace modulus_sum_complex_l157_157383

theorem modulus_sum_complex :
  let z1 : Complex := Complex.mk 3 (-8)
  let z2 : Complex := Complex.mk 4 6
  Complex.abs (z1 + z2) = Real.sqrt 53 := by
  sorry

end modulus_sum_complex_l157_157383


namespace find_initial_quantities_l157_157643

/-- 
Given:
- x + y = 92
- (2/5) * x + (1/4) * y = 26

Prove:
- x = 20
- y = 72
-/
theorem find_initial_quantities (x y : ℝ) (h1 : x + y = 92) (h2 : (2/5) * x + (1/4) * y = 26) :
  x = 20 ∧ y = 72 :=
sorry

end find_initial_quantities_l157_157643


namespace problem_solution_l157_157008

theorem problem_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end problem_solution_l157_157008


namespace number_of_square_free_odd_integers_between_1_and_200_l157_157115

def count_square_free_odd_integers (a b : ℕ) (squares : List ℕ) : ℕ :=
  (b - (a + 1)) / 2 + 1 - List.foldl (λ acc sq => acc + ((b - 1) / sq).div 2 + 1) 0 squares

theorem number_of_square_free_odd_integers_between_1_and_200 :
  count_square_free_odd_integers 1 200 [9, 25, 49, 81, 121] = 81 :=
by
  apply sorry

end number_of_square_free_odd_integers_between_1_and_200_l157_157115


namespace percentage_of_hundred_l157_157489

theorem percentage_of_hundred : (30 / 100) * 100 = 30 := 
by
  sorry

end percentage_of_hundred_l157_157489


namespace solve_for_S_l157_157346

theorem solve_for_S (S : ℝ) (h : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120) : S = 120 :=
sorry

end solve_for_S_l157_157346


namespace central_cell_value_l157_157103

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l157_157103


namespace circle_center_coordinates_l157_157674

theorem circle_center_coordinates :
  ∀ (x y : ℝ), x^2 + y^2 - 10 * x + 6 * y + 25 = 0 → (5, -3) = ((-(-10) / 2), (-6 / 2)) :=
by
  intros x y h
  have H : (5, -3) = ((-(-10) / 2), (-6 / 2)) := sorry
  exact H

end circle_center_coordinates_l157_157674


namespace min_value_of_function_l157_157672

theorem min_value_of_function (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, y = x + 1 / (x - 4) ∧ (∀ z : ℝ, z = x + 1 / (x - 4) → z ≥ 6) :=
sorry

end min_value_of_function_l157_157672


namespace base5_to_base4_last_digit_l157_157567

theorem base5_to_base4_last_digit (n : ℕ) (h : n = 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4) : (n % 4 = 2) :=
by sorry

end base5_to_base4_last_digit_l157_157567


namespace sum_distinct_prime_factors_of_n_l157_157486

theorem sum_distinct_prime_factors_of_n (n : ℕ) 
    (h1 : n < 1000) 
    (h2 : ∃ k : ℕ, 42 * n = 180 * k) : 
    ∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n % p1 = 0 ∧ n % p2 = 0 ∧ n % p3 = 0 ∧ p1 + p2 + p3 = 10 := 
sorry

end sum_distinct_prime_factors_of_n_l157_157486


namespace common_difference_is_minus_two_l157_157571

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
noncomputable def sum_arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem common_difference_is_minus_two
  (a1 d : ℤ)
  (h1 : sum_arith_seq a1 d 5 = 15)
  (h2 : arith_seq a1 d 2 = 5) :
  d = -2 :=
by
  sorry

end common_difference_is_minus_two_l157_157571


namespace machine_working_days_l157_157305

variable {V a b c x y z : ℝ} 

noncomputable def machine_individual_times_condition (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), (x = a + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (y = b + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (z = (-(c * (a + b)) + c * Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (c > 1)

theorem machine_working_days (h1 : x = (z / c) + a) (h2 : y = (z / c) + b) (h3 : z = c * (z / c)) :
  machine_individual_times_condition a b c :=
by
  sorry

end machine_working_days_l157_157305


namespace jake_has_peaches_l157_157551

variable (Jake Steven Jill : Nat)

def given_conditions : Prop :=
  (Steven = 15) ∧ (Steven = Jill + 14) ∧ (Jake = Steven - 7)

theorem jake_has_peaches (h : given_conditions Jake Steven Jill) : Jake = 8 :=
by
  cases h with
  | intro hs1 hrest =>
      cases hrest with
      | intro hs2 hs3 =>
          sorry

end jake_has_peaches_l157_157551


namespace divisible_by_117_l157_157692

theorem divisible_by_117 (n : ℕ) (hn : 0 < n) :
  117 ∣ (3^(2*(n+1)) * 5^(2*n) - 3^(3*n+2) * 2^(2*n)) :=
sorry

end divisible_by_117_l157_157692


namespace truth_values_of_p_and_q_l157_157113

theorem truth_values_of_p_and_q (p q : Prop) (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬p) : ¬p ∧ q :=
by
  sorry

end truth_values_of_p_and_q_l157_157113


namespace trig_functions_symmetry_l157_157084

theorem trig_functions_symmetry :
  ∀ k₁ k₂ : ℤ,
  (∃ x, x = k₁ * π / 2 + π / 3 ∧ x = k₂ * π + π / 3) ∧
  (¬ ∃ x, (x, 0) = (k₁ * π / 2 + π / 12, 0) ∧ (x, 0) = (k₂ * π + 5 * π / 6, 0)) :=
by
  sorry

end trig_functions_symmetry_l157_157084


namespace min_sum_complementary_events_l157_157303

theorem min_sum_complementary_events (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hP : (1 / y) + (4 / x) = 1) : x + y ≥ 9 :=
sorry

end min_sum_complementary_events_l157_157303


namespace n_fraction_of_sum_l157_157665

theorem n_fraction_of_sum (l : List ℝ) (h1 : l.length = 21) (n : ℝ) (h2 : n ∈ l)
  (h3 : ∃ m, l.erase n = m ∧ m.length = 20 ∧ n = 4 * (m.sum / 20)) :
  n = (l.sum) / 6 :=
by
  sorry

end n_fraction_of_sum_l157_157665


namespace remainder_of_division_l157_157026

noncomputable def P (x : ℝ) := x ^ 888
noncomputable def Q (x : ℝ) := (x ^ 2 - x + 1) * (x + 1)

theorem remainder_of_division :
  ∀ x : ℝ, (P x) % (Q x) = 1 :=
sorry

end remainder_of_division_l157_157026


namespace possible_values_of_p_l157_157549

theorem possible_values_of_p (a b c : ℝ) (h₁ : (-a + b + c) / a = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (a + b - c) / c) :
  ∃ p ∈ ({-1, 8} : Set ℝ), p = (a + b) * (b + c) * (c + a) / (a * b * c) :=
by sorry

end possible_values_of_p_l157_157549


namespace parallelogram_to_rhombus_l157_157500

theorem parallelogram_to_rhombus {a b m1 m2 x : ℝ} (h_area : a * m1 = x * m2) (h_proportion : b / m1 = x / m2) : x = Real.sqrt (a * b) :=
by
  -- Proof goes here
  sorry

end parallelogram_to_rhombus_l157_157500


namespace certain_number_is_negative_425_l157_157766

theorem certain_number_is_negative_425 (x : ℝ) :
  (3 - (1/5) * x = 88) ∧ (4 - (1/7) * 210 = -26) → x = -425 :=
by
  sorry

end certain_number_is_negative_425_l157_157766


namespace sum_of_squares_is_149_l157_157815

-- Define the integers and their sum and product
def integers_sum (b : ℤ) : ℤ := (b - 1) + b + (b + 1)
def integers_product (b : ℤ) : ℤ := (b - 1) * b * (b + 1)

-- Define the condition given in the problem
def condition (b : ℤ) : Prop :=
  integers_product b = 12 * integers_sum b + b^2

-- Define the sum of squares of three consecutive integers
def sum_of_squares (b : ℤ) : ℤ :=
  (b - 1)^2 + b^2 + (b + 1)^2

-- The main statement to be proved
theorem sum_of_squares_is_149 (b : ℤ) (h : condition b) : sum_of_squares b = 149 :=
by
  sorry

end sum_of_squares_is_149_l157_157815


namespace more_than_1000_triplets_l157_157484

theorem more_than_1000_triplets :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 1000 < S.card ∧ 
  ∀ (a b c : ℕ), (a, b, c) ∈ S → a^15 + b^15 = c^16 :=
by sorry

end more_than_1000_triplets_l157_157484


namespace inverse_proportion_point_passes_through_l157_157186

theorem inverse_proportion_point_passes_through
  (m : ℝ) (h1 : (4, 6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst})
  : (-4, -6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst} :=
sorry

end inverse_proportion_point_passes_through_l157_157186


namespace geometric_sequence_a4_l157_157264

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the geometric sequence

axiom a_2 : a 2 = -2
axiom a_6 : a 6 = -32
axiom geom_seq (n : ℕ) : a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geometric_sequence_a4 : a 4 = -8 := 
by
  sorry

end geometric_sequence_a4_l157_157264


namespace ananthu_can_complete_work_in_45_days_l157_157812

def amit_work_rate : ℚ := 1 / 15

def time_amit_worked : ℚ := 3

def total_work : ℚ := 1

def total_days : ℚ := 39

noncomputable def ananthu_days (x : ℚ) : Prop :=
  let amit_work_done := time_amit_worked * amit_work_rate
  let remaining_work := total_work - amit_work_done
  let ananthu_work_rate := remaining_work / (total_days - time_amit_worked)
  1 /x = ananthu_work_rate

theorem ananthu_can_complete_work_in_45_days :
  ananthu_days 45 :=
by
  sorry

end ananthu_can_complete_work_in_45_days_l157_157812


namespace purely_imaginary_complex_l157_157215

theorem purely_imaginary_complex (a : ℝ) 
  (h₁ : a^2 + 2 * a - 3 = 0)
  (h₂ : a + 3 ≠ 0) : a = 1 := by
  sorry

end purely_imaginary_complex_l157_157215


namespace harry_average_sleep_l157_157256

-- Conditions
def sleep_time_monday : ℕ × ℕ := (8, 15)
def sleep_time_tuesday : ℕ × ℕ := (7, 45)
def sleep_time_wednesday : ℕ × ℕ := (8, 10)
def sleep_time_thursday : ℕ × ℕ := (10, 25)
def sleep_time_friday : ℕ × ℕ := (7, 50)

-- Total sleep time calculation
def total_sleep_time : ℕ × ℕ :=
  let (h1, m1) := sleep_time_monday
  let (h2, m2) := sleep_time_tuesday
  let (h3, m3) := sleep_time_wednesday
  let (h4, m4) := sleep_time_thursday
  let (h5, m5) := sleep_time_friday
  (h1 + h2 + h3 + h4 + h5, m1 + m2 + m3 + m4 + m5)

-- Convert minutes to hours and minutes
def convert_minutes (mins : ℕ) : ℕ × ℕ :=
  (mins / 60, mins % 60)

-- Final total sleep time
def final_total_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := total_sleep_time
  let (extra_hours, remaining_minutes) := convert_minutes total_minutes
  (total_hours + extra_hours, remaining_minutes)

-- Average calculation
def average_sleep_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := final_total_time
  (total_hours / 5, (total_hours % 5) * 60 / 5 + total_minutes / 5)

-- The proof statement
theorem harry_average_sleep :
  average_sleep_time = (8, 29) :=
  by
    sorry

end harry_average_sleep_l157_157256


namespace correct_expression_l157_157907

-- Definitions based on given conditions
def expr1 (a b : ℝ) := 3 * a + 2 * b = 5 * a * b
def expr2 (a : ℝ) := 2 * a^3 - a^3 = a^3
def expr3 (a b : ℝ) := a^2 * b - a * b = a
def expr4 (a : ℝ) := a^2 + a^2 = 2 * a^4

-- Statement to prove that expr2 is the only correct expression
theorem correct_expression (a b : ℝ) : 
  expr2 a := by
  sorry

end correct_expression_l157_157907


namespace no_solutions_in_domain_l157_157742

-- Define the function g
def g (x : ℝ) : ℝ := -0.5 * x^2 + x + 3

-- Define the condition on the domain of g
def in_domain (x : ℝ) : Prop := x ≥ -3 ∧ x ≤ 3

-- State the theorem to be proved
theorem no_solutions_in_domain :
  ∀ x : ℝ, in_domain x → ¬ (g (g x) = 3) :=
by
  -- Provide a placeholder for the proof
  sorry

end no_solutions_in_domain_l157_157742


namespace faster_speed_l157_157407

theorem faster_speed (v : ℝ) :
  (∀ t : ℝ, (40 / 10 = t) ∧ (60 / v = t)) → v = 15 :=
by
  sorry

end faster_speed_l157_157407


namespace isosceles_triangle_area_l157_157291

theorem isosceles_triangle_area (p x : ℝ) 
  (h1 : 2 * p = 6 * x) 
  (h2 : 0 < p) 
  (h3 : 0 < x) :
  (1 / 2) * (2 * x) * (Real.sqrt (8 * p^2 / 9)) = (Real.sqrt 8 * p^2) / 3 :=
by
  sorry

end isosceles_triangle_area_l157_157291


namespace driving_speed_l157_157003

variable (total_distance : ℝ) (break_time : ℝ) (total_trip_time : ℝ)

theorem driving_speed (h1 : total_distance = 480)
                      (h2 : break_time = 1)
                      (h3 : total_trip_time = 9) : 
  (total_distance / (total_trip_time - break_time)) = 60 :=
by
  sorry

end driving_speed_l157_157003


namespace exponent_division_example_l157_157561

theorem exponent_division_example : ((3^2)^4) / (3^2) = 729 := by
  sorry

end exponent_division_example_l157_157561


namespace goldfish_equal_number_after_n_months_l157_157431

theorem goldfish_equal_number_after_n_months :
  ∃ (n : ℕ), 2 * 4^n = 162 * 3^n ∧ n = 6 :=
by
  sorry

end goldfish_equal_number_after_n_months_l157_157431


namespace wire_length_from_sphere_volume_l157_157794

theorem wire_length_from_sphere_volume
  (r_sphere : ℝ) (r_cylinder : ℝ) (h : ℝ)
  (h_sphere : r_sphere = 12)
  (h_cylinder : r_cylinder = 4)
  (volume_conservation : (4/3 * Real.pi * r_sphere^3) = (Real.pi * r_cylinder^2 * h)) :
  h = 144 :=
by {
  sorry
}

end wire_length_from_sphere_volume_l157_157794


namespace angle_bisector_median_ineq_l157_157614

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (l_a l_b l_c m_a m_b m_c : ℝ)

theorem angle_bisector_median_ineq
  (hl_a : l_a > 0) (hl_b : l_b > 0) (hl_c : l_c > 0)
  (hm_a : m_a > 0) (hm_b : m_b > 0) (hm_c : m_c > 0) :
  l_a / m_a + l_b / m_b + l_c / m_c > 1 :=
sorry

end angle_bisector_median_ineq_l157_157614


namespace simplify_and_evaluate_l157_157616

theorem simplify_and_evaluate (a : ℝ) (h₁ : a^2 - 4 * a + 3 = 0) (h₂ : a ≠ 3) : 
  ( (a^2 - 9) / (a^2 - 3 * a) / ( (a^2 + 9) / a + 6 ) = 1 / 4 ) :=
by 
  sorry

end simplify_and_evaluate_l157_157616


namespace train_overtake_distance_l157_157577

/--
 Train A leaves the station traveling at 30 miles per hour.
 Two hours later, Train B leaves the same station traveling in the same direction at 42 miles per hour.
 Prove that Train A is overtaken by Train B 210 miles from the station.
-/
theorem train_overtake_distance
    (speed_A : ℕ) (speed_B : ℕ) (delay_B : ℕ)
    (hA : speed_A = 30)
    (hB : speed_B = 42)
    (hDelay : delay_B = 2) :
    ∃ d : ℕ, d = 210 ∧ ∀ t : ℕ, (speed_B * t = (speed_A * t + speed_A * delay_B) → d = speed_B * t) :=
by
  sorry

end train_overtake_distance_l157_157577


namespace value_of_f_at_7_l157_157904

theorem value_of_f_at_7
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = 2 :=
by
  -- Proof will be filled here
  sorry

end value_of_f_at_7_l157_157904


namespace circle_proof_problem_l157_157803

variables {P Q R : Type}
variables {p q r dPQ dPR dQR : ℝ}

-- Given Conditions
variables (hpq : p > q) (hqr : q > r)
variables (hdPQ : ℝ) (hdPR : ℝ) (hdQR : ℝ)

-- Statement of the problem: prove that all conditions can be true
theorem circle_proof_problem :
  (∃ hpq' : dPQ = p + q, true) ∧
  (∃ hqr' : dQR = q + r, true) ∧
  (∃ hpr' : dPR > p + r, true) ∧
  (∃ hpq_diff : dPQ > p - q, true) →
  false := 
sorry

end circle_proof_problem_l157_157803


namespace sufficient_not_necessary_range_l157_157810

theorem sufficient_not_necessary_range (x a : ℝ) : (∀ x, x < 1 → x < a) ∧ (∃ x, x < a ∧ ¬ (x < 1)) ↔ 1 < a := by
  sorry

end sufficient_not_necessary_range_l157_157810


namespace correct_propositions_l157_157434

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem correct_propositions :
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) ∧
  (Real.sqrt 2 = f (Real.pi / 24)) ∧
  (f (-1) ≠ f 1) ∧
  (∀ x, Real.pi / 24 ≤ x ∧ x ≤ 13 * Real.pi / 24 -> (f (x + 1e-6) < f x)) ∧
  (∀ x, (Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) = f x)
  := by
    sorry

end correct_propositions_l157_157434


namespace number_of_swaps_independent_l157_157204

theorem number_of_swaps_independent (n : ℕ) (hn : n = 20) (p : Fin n → Fin n) :
    (∀ i, p i ≠ i → ∃ j, p j ≠ j ∧ p (p j) = j) →
    ∃ s : List (Fin n × Fin n), List.length s ≤ n ∧
    (∀ σ : List (Fin n × Fin n), (∀ (i j : Fin n), (i, j) ∈ σ → p i ≠ i → ∃ p', σ = (i, p') :: (p', j) :: σ) →
     List.length σ = List.length s) :=
  sorry

end number_of_swaps_independent_l157_157204


namespace area_change_l157_157443

variable (p k : ℝ)
variable {N : ℝ}

theorem area_change (hN : N = 1/2 * (p * p)) (q : ℝ) (hq : q = k * p) :
  q = k * p -> (1/2 * (q * q) = k^2 * N) :=
by
  intros
  sorry

end area_change_l157_157443


namespace min_value_of_squares_l157_157460

theorem min_value_of_squares (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 :=
sorry

end min_value_of_squares_l157_157460


namespace find_y_from_condition_l157_157331

variable (y : ℝ) (h : (3 * y) / 7 = 15)

theorem find_y_from_condition : y = 35 :=
by {
  sorry
}

end find_y_from_condition_l157_157331


namespace stephen_speed_l157_157490

theorem stephen_speed (v : ℝ) 
  (time : ℝ := 0.25)
  (speed_second_third : ℝ := 12)
  (speed_last_third : ℝ := 20)
  (total_distance : ℝ := 12) :
  (v * time + speed_second_third * time + speed_last_third * time = total_distance) → 
  v = 16 :=
by
  intro h
  -- introducing the condition h: v * 0.25 + 3 + 5 = 12
  sorry

end stephen_speed_l157_157490


namespace vasim_share_l157_157918

theorem vasim_share (x : ℝ)
  (h_ratio : ∀ (f v r : ℝ), f = 3 * x ∧ v = 5 * x ∧ r = 6 * x)
  (h_diff : 6 * x - 3 * x = 900) :
  5 * x = 1500 :=
by
  try sorry

end vasim_share_l157_157918


namespace inequality_solution_set_l157_157697

theorem inequality_solution_set (x : ℝ) :
  abs (1 + x + x^2 / 2) < 1 ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_set_l157_157697


namespace ratio_is_one_half_l157_157448

noncomputable def ratio_of_females_to_males (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ) : ℚ :=
  (f : ℚ) / (m : ℚ)

theorem ratio_is_one_half (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ)
  (h_female_age : avg_female_age = 45)
  (h_male_age : avg_male_age = 30)
  (h_total_age : avg_total_age = 35)
  (h_total_avg : (45 * f + 30 * m) / (f + m) = 35) :
  ratio_of_females_to_males f m avg_female_age avg_male_age avg_total_age = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l157_157448


namespace youngest_person_age_l157_157615

theorem youngest_person_age (n : ℕ) (average_age : ℕ) (average_age_when_youngest_born : ℕ) 
    (h1 : n = 7) (h2 : average_age = 30) (h3 : average_age_when_youngest_born = 24) :
    ∃ Y : ℚ, Y = 66 / 7 :=
by
  sorry

end youngest_person_age_l157_157615


namespace base_b_expression_not_divisible_l157_157977

theorem base_b_expression_not_divisible 
  (b : ℕ) : 
  (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) →
  (2 * b^3 - 2 * b^2 + b - 1) % 5 ≠ 0 ↔ (b ≠ 6) :=
by
  sorry

end base_b_expression_not_divisible_l157_157977


namespace rectangle_area_l157_157447

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l157_157447


namespace xiao_ming_min_correct_answers_l157_157068

theorem xiao_ming_min_correct_answers (x : ℕ) : (10 * x - 5 * (20 - x) > 100) → (x ≥ 14) := by
  sorry

end xiao_ming_min_correct_answers_l157_157068


namespace square_center_sum_l157_157398

noncomputable def sum_of_center_coordinates (A B C D : ℝ × ℝ) : ℝ :=
  let center : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  center.1 + center.2

theorem square_center_sum
  (A B C D : ℝ × ℝ)
  (h1 : 9 = A.1) (h2 : 0 = A.2)
  (h3 : 4 = B.1) (h4 : 0 = B.2)
  (h5 : 0 = C.1) (h6 : 3 = C.2)
  (h7: A.1 < B.1) (h8: A.2 < C.2) :
  sum_of_center_coordinates A B C D = 8 := 
by
  sorry

end square_center_sum_l157_157398


namespace Jason_saturday_hours_l157_157638

theorem Jason_saturday_hours (x y : ℕ) 
  (h1 : 4 * x + 6 * y = 88)
  (h2 : x + y = 18) : 
  y = 8 :=
sorry

end Jason_saturday_hours_l157_157638


namespace part_I_part_II_l157_157760

-- Let the volume V of the tetrahedron ABCD be given
def V : ℝ := sorry

-- Areas of the faces opposite vertices A, B, C, D
def S_A : ℝ := sorry
def S_B : ℝ := sorry
def S_C : ℝ := sorry
def S_D : ℝ := sorry

-- Definitions of the edge lengths and angles
def a : ℝ := sorry -- BC
def a' : ℝ := sorry -- DA
def b : ℝ := sorry -- CA
def b' : ℝ := sorry -- DB
def c : ℝ := sorry -- AB
def c' : ℝ := sorry -- DC
def alpha : ℝ := sorry -- Angle between BC and DA
def beta : ℝ := sorry -- Angle between CA and DB
def gamma : ℝ := sorry -- Angle between AB and DC

theorem part_I : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 = 
  (1 / 4) * ((a * a' * Real.sin alpha)^2 + (b * b' * Real.sin beta)^2 + (c * c' * Real.sin gamma)^2) := 
  sorry

theorem part_II : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 ≥ 9 * (3 * V^4)^(1/3) :=
  sorry

end part_I_part_II_l157_157760


namespace slope_of_line_through_origin_and_center_l157_157801

def Point := (ℝ × ℝ)

def is_center (p : Point) : Prop :=
  p = (3, 1)

def is_dividing_line (l : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, l x = y → y / x = 1 / 3

theorem slope_of_line_through_origin_and_center :
  ∃ l : ℝ → ℝ, (∀ p1 p2 : Point,
  p1 = (0, 0) →
  p2 = (3, 1) →
  is_center p2 →
  is_dividing_line l) :=
sorry

end slope_of_line_through_origin_and_center_l157_157801


namespace range_of_a_l157_157983

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, y = (a * (Real.cos x)^2 - 3) * (Real.sin x) ∧ y ≥ -3) 
  → a ∈ Set.Icc (-3/2 : ℝ) 12 :=
sorry

end range_of_a_l157_157983


namespace sum_of_natural_numbers_l157_157405

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  2^(n-1)

theorem sum_of_natural_numbers (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k = number_of_ways n :=
by
  use 2^(n-1)
  sorry

end sum_of_natural_numbers_l157_157405


namespace B_cycling_speed_l157_157562

variable (A_speed B_distance B_time B_speed : ℕ)
variable (t1 : ℕ := 7)
variable (d_total : ℕ := 140)
variable (B_catch_time : ℕ := 7)

theorem B_cycling_speed :
  A_speed = 10 → 
  d_total = 140 →
  B_catch_time = 7 → 
  B_speed = 20 :=
by
  sorry

end B_cycling_speed_l157_157562


namespace asymptotes_of_hyperbola_min_focal_distance_l157_157668

theorem asymptotes_of_hyperbola_min_focal_distance :
  ∀ (x y m : ℝ),
  (m = 1 → 
   (∀ x y : ℝ, (x^2 / (m^2 + 8) - y^2 / (6 - 2 * m) = 1) → 
   (y = 2/3 * x ∨ y = -2/3 * x))) := 
  sorry

end asymptotes_of_hyperbola_min_focal_distance_l157_157668


namespace two_marbles_different_colors_probability_l157_157117

-- Definitions
def red_marbles : Nat := 3
def green_marbles : Nat := 4
def white_marbles : Nat := 5
def blue_marbles : Nat := 3
def total_marbles : Nat := red_marbles + green_marbles + white_marbles + blue_marbles

-- Combinations of different colored marbles
def red_green : Nat := red_marbles * green_marbles
def red_white : Nat := red_marbles * white_marbles
def red_blue : Nat := red_marbles * blue_marbles
def green_white : Nat := green_marbles * white_marbles
def green_blue : Nat := green_marbles * blue_marbles
def white_blue : Nat := white_marbles * blue_marbles

-- Total favorable outcomes
def total_favorable : Nat := red_green + red_white + red_blue + green_white + green_blue + white_blue

-- Total outcomes when drawing 2 marbles from the jar
def total_outcomes : Nat := Nat.choose total_marbles 2

-- Probability calculation
noncomputable def probability_different_colors : Rat := total_favorable / total_outcomes

-- Proof that the probability is 83/105
theorem two_marbles_different_colors_probability :
  probability_different_colors = 83 / 105 := by
  sorry

end two_marbles_different_colors_probability_l157_157117


namespace size_of_each_group_l157_157461

theorem size_of_each_group 
  (boys : ℕ) (girls : ℕ) (groups : ℕ)
  (total_students : boys + girls = 63)
  (num_groups : groups = 7) :
  63 / 7 = 9 :=
by
  sorry

end size_of_each_group_l157_157461


namespace largest_valid_integer_l157_157703

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def satisfies_conditions (n : ℕ) : Prop :=
  (100 ≤ n ∧ n < 1000) ∧
  ∀ d ∈ n.digits 10, d ≠ 0 ∧ n % d = 0 ∧
  sum_of_digits n % 6 = 0

theorem largest_valid_integer : ∃ n : ℕ, satisfies_conditions n ∧ (∀ m : ℕ, satisfies_conditions m → m ≤ n) ∧ n = 936 :=
by
  sorry

end largest_valid_integer_l157_157703


namespace car_mpg_in_city_l157_157031

theorem car_mpg_in_city
  (H C T : ℕ)
  (h1 : H * T = 462)
  (h2 : C * T = 336)
  (h3 : C = H - 9) : C = 24 := by
  sorry

end car_mpg_in_city_l157_157031


namespace number_of_fifth_graders_l157_157565

-- Define the conditions given in the problem.
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Derived definitions with the help of the conditions.
def total_seats : ℕ := buses * seats_per_bus
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_sixth_and_seventh_graders : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := total_sixth_and_seventh_graders + total_chaperones
def seats_for_fifth_graders : ℕ := total_seats - seats_taken

-- The final statement to prove the number of fifth graders.
theorem number_of_fifth_graders : seats_for_fifth_graders = 109 :=
by
  sorry

end number_of_fifth_graders_l157_157565


namespace maximum_members_in_dance_troupe_l157_157157

theorem maximum_members_in_dance_troupe (m : ℕ) (h1 : 25 * m % 31 = 7) (h2 : 25 * m < 1300) : 25 * m = 875 :=
by {
  sorry
}

end maximum_members_in_dance_troupe_l157_157157


namespace eval_f_at_800_l157_157913

-- Given conditions in Lean 4:
def f : ℝ → ℝ := sorry -- placeholder for the function definition
axiom func_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_at_1000 : f 1000 = 4

-- The goal/proof statement:
theorem eval_f_at_800 : f 800 = 5 := sorry

end eval_f_at_800_l157_157913


namespace total_loads_washed_l157_157073

theorem total_loads_washed (a b : ℕ) (h1 : a = 8) (h2 : b = 6) : a + b = 14 :=
by
  sorry

end total_loads_washed_l157_157073


namespace total_money_made_from_jerseys_l157_157646

def price_per_jersey : ℕ := 76
def jerseys_sold : ℕ := 2

theorem total_money_made_from_jerseys : price_per_jersey * jerseys_sold = 152 := 
by
  -- The actual proof steps will go here
  sorry

end total_money_made_from_jerseys_l157_157646


namespace route_length_is_140_l157_157094

-- Conditions of the problem
variable (D : ℝ)  -- Length of the route
variable (Vx Vy t : ℝ)  -- Speeds of Train X and Train Y, and time to meet

-- Given conditions
axiom train_X_trip_time : D / Vx = 4
axiom train_Y_trip_time : D / Vy = 3
axiom train_X_distance_when_meet : Vx * t = 60
axiom total_distance_covered_on_meeting : Vx * t + Vy * t = D

-- Goal: Prove that the length of the route is 140 kilometers
theorem route_length_is_140 : D = 140 := by
  -- Proof omitted
  sorry

end route_length_is_140_l157_157094


namespace James_distance_ridden_l157_157355

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l157_157355


namespace four_brothers_money_l157_157696

theorem four_brothers_money 
  (a_1 a_2 a_3 a_4 : ℝ) 
  (x : ℝ)
  (h1 : a_1 + a_2 + a_3 + a_4 = 48)
  (h2 : a_1 + 3 = x)
  (h3 : a_2 - 3 = x)
  (h4 : 3 * a_3 = x)
  (h5 : a_4 / 3 = x) :
  a_1 = 6 ∧ a_2 = 12 ∧ a_3 = 3 ∧ a_4 = 27 :=
by
  sorry

end four_brothers_money_l157_157696


namespace waiter_earned_total_tips_l157_157144

def tips (c1 c2 c3 c4 c5 : ℝ) := c1 + c2 + c3 + c4 + c5

theorem waiter_earned_total_tips :
  tips 1.50 2.75 3.25 4.00 5.00 = 16.50 := 
by 
  sorry

end waiter_earned_total_tips_l157_157144


namespace polynomial_comparison_l157_157928

theorem polynomial_comparison {x : ℝ} :
  let A := (x - 3) * (x - 2)
  let B := (x + 1) * (x - 6)
  A > B :=
by 
  sorry -- Proof is omitted.

end polynomial_comparison_l157_157928


namespace no_natural_numbers_satisfy_conditions_l157_157863

theorem no_natural_numbers_satisfy_conditions : 
  ¬ ∃ (a b : ℕ), 
    (∃ (k : ℕ), k^2 = a^2 + 2 * b^2) ∧ 
    (∃ (m : ℕ), m^2 = b^2 + 2 * a) :=
by {
  -- Proof steps and logical deductions can be written here.
  sorry
}

end no_natural_numbers_satisfy_conditions_l157_157863


namespace find_A_l157_157477

theorem find_A (A : ℤ) (h : 10 + A = 15) : A = 5 := by
  sorry

end find_A_l157_157477


namespace roots_lost_extraneous_roots_l157_157885

noncomputable def f1 (x : ℝ) := Real.arcsin x
noncomputable def g1 (x : ℝ) := 2 * Real.arcsin (x / Real.sqrt 2)
noncomputable def f2 (x : ℝ) := x
noncomputable def g2 (x : ℝ) := 2 * x

theorem roots_lost :
  ∃ x : ℝ, f1 x = g1 x ∧ ¬ ∃ y : ℝ, Real.tan (f1 y) = Real.tan (g1 y) :=
sorry

theorem extraneous_roots :
  ∃ x : ℝ, ¬ f2 x = g2 x ∧ ∃ y : ℝ, Real.tan (f2 y) = Real.tan (g2 y) :=
sorry

end roots_lost_extraneous_roots_l157_157885


namespace remaining_people_l157_157998

def initial_football_players : ℕ := 13
def initial_cheerleaders : ℕ := 16
def quitting_football_players : ℕ := 10
def quitting_cheerleaders : ℕ := 4

theorem remaining_people :
  (initial_football_players - quitting_football_players) 
  + (initial_cheerleaders - quitting_cheerleaders) = 15 := by
    -- Proof steps would go here, if required
    sorry

end remaining_people_l157_157998


namespace max_lateral_surface_area_of_tetrahedron_l157_157324

open Real

theorem max_lateral_surface_area_of_tetrahedron :
  ∀ (PA PB PC : ℝ), (PA^2 + PB^2 + PC^2 = 36) → (PA * PB + PB * PC + PA * PC ≤ 36) →
  (1/2 * (PA * PB + PB * PC + PA * PC) ≤ 18) :=
by
  intro PA PB PC hsum hineq
  sorry

end max_lateral_surface_area_of_tetrahedron_l157_157324


namespace average_first_21_multiples_of_8_l157_157142

noncomputable def average_of_multiples (n : ℕ) (a : ℕ) : ℕ :=
  let sum := (n * (a + a * n)) / 2
  sum / n

theorem average_first_21_multiples_of_8 : average_of_multiples 21 8 = 88 :=
by
  sorry

end average_first_21_multiples_of_8_l157_157142


namespace circle_center_count_l157_157836

noncomputable def num_circle_centers (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) : ℕ :=
  if (c = d) then 4 else 8

-- Here is the theorem statement
theorem circle_center_count (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) :
  num_circle_centers b c d h₁ h₂ = if (c = d) then 4 else 8 :=
sorry

end circle_center_count_l157_157836


namespace total_shells_l157_157409

theorem total_shells :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let ed_scallop_shells := 3
  let jacob_more_shells := 2
  let marissa_limpet_shells := 5
  let marissa_oyster_shells := 6
  let marissa_conch_shells := 3
  let marissa_scallop_shells := 1
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + ed_scallop_shells
  let jacob_shells := ed_shells + jacob_more_shells
  let marissa_shells := marissa_limpet_shells + marissa_oyster_shells + marissa_conch_shells + marissa_scallop_shells
  let shells_at_beach := ed_shells + jacob_shells + marissa_shells
  let total_shells := shells_at_beach + initial_shells
  total_shells = 51 := by
  sorry

end total_shells_l157_157409


namespace mrs_hilt_travel_distance_l157_157597

theorem mrs_hilt_travel_distance :
  let distance_water_fountain := 30
  let distance_main_office := 50
  let distance_teacher_lounge := 35
  let trips_water_fountain := 4
  let trips_main_office := 2
  let trips_teacher_lounge := 3
  (distance_water_fountain * trips_water_fountain +
   distance_main_office * trips_main_office +
   distance_teacher_lounge * trips_teacher_lounge) = 325 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_travel_distance_l157_157597


namespace unique_integer_n_l157_157037

theorem unique_integer_n (n : ℤ) (h : ⌊(n^2 : ℚ) / 5⌋ - ⌊(n / 2 : ℚ)⌋^2 = 3) : n = 5 :=
  sorry

end unique_integer_n_l157_157037


namespace rational_sum_eq_neg2_l157_157593

theorem rational_sum_eq_neg2 (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 :=
sorry

end rational_sum_eq_neg2_l157_157593


namespace quadratic_form_sum_l157_157602

theorem quadratic_form_sum :
  ∃ a b c : ℝ, (∀ x : ℝ, 5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) :=
sorry

end quadratic_form_sum_l157_157602


namespace sum_xyz_l157_157934

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_l157_157934


namespace inversely_proportional_x_y_l157_157496

theorem inversely_proportional_x_y (x y c : ℝ) 
  (h1 : x * y = c) (h2 : 8 * 16 = c) : y = -32 → x = -4 :=
by
  sorry

end inversely_proportional_x_y_l157_157496


namespace sum_of_consecutive_integers_l157_157988

theorem sum_of_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 930) : x + (x + 1) = 61 :=
sorry

end sum_of_consecutive_integers_l157_157988


namespace mark_profit_l157_157708

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l157_157708


namespace interval_length_difference_l157_157076

noncomputable def log2_abs (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem interval_length_difference :
  ∀ (a b : ℝ), (∀ x, a ≤ x ∧ x ≤ b → 0 ≤ log2_abs x ∧ log2_abs x ≤ 2) → 
               (b - a = 15 / 4 - 3 / 4) :=
by
  intros a b h
  sorry

end interval_length_difference_l157_157076


namespace tan_theta_minus_pi_four_l157_157365

theorem tan_theta_minus_pi_four (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) (h3 : Real.sin θ = -3/5) :
  Real.tan (θ - π / 4) = -1 / 7 :=
sorry

end tan_theta_minus_pi_four_l157_157365


namespace sample_size_ratio_l157_157560

theorem sample_size_ratio (n : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
                          (total_ratio : ℕ) (B_in_sample : ℕ)
                          (h_ratio : ratio_A = 1 ∧ ratio_B = 3 ∧ ratio_C = 5)
                          (h_total : total_ratio = ratio_A + ratio_B + ratio_C)
                          (h_B_sample : B_in_sample = 27)
                          (h_sampling_ratio_B : ratio_B / total_ratio = 1 / 3) :
                          n = 81 :=
by sorry

end sample_size_ratio_l157_157560


namespace arrange_COMMUNICATION_l157_157402

theorem arrange_COMMUNICATION : 
  let n := 12
  let o_count := 2
  let i_count := 2
  let n_count := 2
  let m_count := 2
  let total_repeats := o_count * i_count * n_count * m_count
  n.factorial / (o_count.factorial * i_count.factorial * n_count.factorial * m_count.factorial) = 29937600 :=
by sorry

end arrange_COMMUNICATION_l157_157402


namespace hyperbola_eccentricity_is_sqrt2_l157_157282

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (h : a ≠ 0) (hb : b = a) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c / a)

theorem hyperbola_eccentricity_is_sqrt2 {a : ℝ} (h : a ≠ 0) :
  eccentricity_of_hyperbola h (rfl) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt2_l157_157282


namespace brownies_count_l157_157584

theorem brownies_count {B : ℕ} 
  (h1 : B/2 = (B - B / 2))
  (h2 : B/4 = (B - B / 2) / 2)
  (h3 : B/4 - 2 = B/4 - 2)
  (h4 : B/4 - 2 = 3) : 
  B = 20 := 
by 
  sorry

end brownies_count_l157_157584


namespace calories_per_shake_l157_157689

theorem calories_per_shake (total_calories_per_day : ℕ) (breakfast_calories : ℕ)
  (lunch_percentage_increase : ℕ) (dinner_multiplier : ℕ) (number_of_shakes : ℕ)
  (daily_calories : ℕ) :
  total_calories_per_day = breakfast_calories +
                            (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100)) +
                            (2 * (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100))) →
  daily_calories = total_calories_per_day + number_of_shakes * (daily_calories - total_calories_per_day) / number_of_shakes →
  daily_calories = 3275 → breakfast_calories = 500 →
  lunch_percentage_increase = 25 →
  dinner_multiplier = 2 →
  number_of_shakes = 3 →
  (daily_calories - total_calories_per_day) / number_of_shakes = 300 := by 
  sorry

end calories_per_shake_l157_157689


namespace calculate_expression_l157_157022

theorem calculate_expression : 7 * (12 + 2 / 5) - 3 = 83.8 :=
by
  sorry

end calculate_expression_l157_157022


namespace range_of_b_l157_157260

theorem range_of_b (b : ℝ) (hb : b > 0) : (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) :=
by
  sorry

end range_of_b_l157_157260


namespace solve_quadratic_eqn_l157_157802

theorem solve_quadratic_eqn (x : ℝ) : 3 * x ^ 2 = 27 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end solve_quadratic_eqn_l157_157802


namespace value_of_b_l157_157758

theorem value_of_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := 
by 
  sorry

end value_of_b_l157_157758


namespace inequality_not_necessarily_hold_l157_157012

theorem inequality_not_necessarily_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) :=
sorry

end inequality_not_necessarily_hold_l157_157012


namespace aunt_li_more_cost_effective_l157_157558

theorem aunt_li_more_cost_effective (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (100 * a + 100 * b) / 200 ≥ 200 / ((100 / a) + (100 / b)) :=
by
  sorry

end aunt_li_more_cost_effective_l157_157558


namespace find_c_d_l157_157681

def star (c d : ℕ) : ℕ := c^d + c*d

theorem find_c_d (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d) (h_star : star c d = 28) : c + d = 7 :=
by
  sorry

end find_c_d_l157_157681


namespace sum_of_products_l157_157660

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  ab + bc + ac = 72 := 
by 
  sorry

end sum_of_products_l157_157660


namespace paving_cost_l157_157236

-- Definitions based on conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 600
def expected_cost : ℝ := 12375

-- The problem statement
theorem paving_cost :
  (length * width * rate_per_sqm = expected_cost) :=
sorry

end paving_cost_l157_157236


namespace silvia_shorter_route_l157_157363

theorem silvia_shorter_route :
  let jerry_distance := 3 + 4
  let silvia_distance := Real.sqrt (3^2 + 4^2)
  let percentage_reduction := ((jerry_distance - silvia_distance) / jerry_distance) * 100
  (28.5 ≤ percentage_reduction ∧ percentage_reduction < 30.5) →
  percentage_reduction = 30 := by
  intro h
  sorry

end silvia_shorter_route_l157_157363


namespace range_of_a_l157_157860

theorem range_of_a 
  (a : ℕ) 
  (an : ℕ → ℕ)
  (Sn : ℕ → ℕ)
  (h1 : a_1 = a)
  (h2 : ∀ n : ℕ, n ≥ 2 → Sn n + Sn (n - 1) = 4 * n^2)
  (h3 : ∀ n : ℕ, an n < an (n + 1)) : 
  3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l157_157860


namespace find_a_l157_157716

variable {x a : ℝ}

def A (x : ℝ) : Prop := x ≤ -1 ∨ x > 2
def B (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem find_a (hA : ∀ x, (x + 1) / (x - 2) ≥ 0 ↔ A x)
                (hB : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a > 0 ↔ B x a)
                (hSub : ∀ x, A x → B x a) :
  -1 < a ∧ a ≤ 1 :=
sorry

end find_a_l157_157716


namespace hosting_schedules_count_l157_157883

theorem hosting_schedules_count :
  let n_universities := 6
  let n_years := 8
  let total_ways := 6 * 5 * 4^6
  let excluding_one := 6 * 5 * 4 * 3^6
  let excluding_two := 15 * 4 * 3 * 2^6
  let excluding_three := 20 * 3 * 2 * 1^6
  total_ways - excluding_one + excluding_two - excluding_three = 46080 := 
by
  sorry

end hosting_schedules_count_l157_157883


namespace cherry_tomatoes_ratio_l157_157946

theorem cherry_tomatoes_ratio (T P B : ℕ) (M : ℕ := 3) (h1 : P = 4 * T) (h2 : B = 4 * P) (h3 : B / 3 = 32) :
  (T : ℚ) / M = 2 :=
by
  sorry

end cherry_tomatoes_ratio_l157_157946


namespace B_spends_85_percent_salary_l157_157730

theorem B_spends_85_percent_salary (A_s B_s : ℝ) (A_savings : ℝ) :
  A_s + B_s = 2000 →
  A_s = 1500 →
  A_savings = 0.05 * A_s →
  (B_s - (B_s * (1 - 0.05))) = A_savings →
  (1 - 0.85) * B_s = 0.15 * B_s := 
by
  intros h1 h2 h3 h4
  sorry

end B_spends_85_percent_salary_l157_157730


namespace tangent_lines_parabola_through_point_l157_157929

theorem tangent_lines_parabola_through_point :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = x ^ 2 + 1 → (y - 0) = m * (x - 0)) 
     ∧ ((m = 2 ∧ y = 2 * x) ∨ (m = -2 ∧ y = -2 * x)) :=
sorry

end tangent_lines_parabola_through_point_l157_157929


namespace total_alligators_seen_l157_157173

-- Definitions for the conditions
def SamaraSaw : Nat := 35
def NumberOfFriends : Nat := 6
def AverageFriendsSaw : Nat := 15

-- Statement of the proof problem
theorem total_alligators_seen :
  SamaraSaw + NumberOfFriends * AverageFriendsSaw = 125 := by
  -- Skipping the proof
  sorry

end total_alligators_seen_l157_157173


namespace smallest_m_4_and_n_229_l157_157376

def satisfies_condition (m n : ℕ) : Prop :=
  19 * m + 8 * n = 1908

def is_smallest_m (m n : ℕ) : Prop :=
  ∀ m' n', satisfies_condition m' n' → m' > 0 → n' > 0 → m ≤ m'

theorem smallest_m_4_and_n_229 : ∃ (m n : ℕ), satisfies_condition m n ∧ is_smallest_m m n ∧ m = 4 ∧ n = 229 :=
by
  sorry

end smallest_m_4_and_n_229_l157_157376


namespace sum_of_coordinates_l157_157250

theorem sum_of_coordinates (x : ℚ) : (0, 0) = (0, 0) ∧ (x, -3) = (x, -3) ∧ ((-3 - 0) / (x - 0) = 4 / 5) → x - 3 = -27 / 4 := 
sorry

end sum_of_coordinates_l157_157250


namespace inequality_part_1_inequality_part_2_l157_157711

noncomputable def f (x : ℝ) := |x - 2| + 2
noncomputable def g (x : ℝ) (m : ℝ) := m * |x|

theorem inequality_part_1 (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem inequality_part_2 (m : ℝ) : (∀ x, f x ≥ g x m) ↔ m ≤ 1 := by
  sorry

end inequality_part_1_inequality_part_2_l157_157711


namespace usual_time_to_school_l157_157712

variables (R T : ℝ)

theorem usual_time_to_school :
  (3 / 2) * R * (T - 4) = R * T -> T = 12 :=
by sorry

end usual_time_to_school_l157_157712


namespace Carrie_hourly_wage_l157_157799

theorem Carrie_hourly_wage (hours_per_week : ℕ) (weeks_per_month : ℕ) (cost_bike : ℕ) (remaining_money : ℕ)
  (total_hours : ℕ) (total_savings : ℕ) (x : ℕ) :
  hours_per_week = 35 → 
  weeks_per_month = 4 → 
  cost_bike = 400 → 
  remaining_money = 720 → 
  total_hours = hours_per_week * weeks_per_month → 
  total_savings = cost_bike + remaining_money → 
  total_savings = total_hours * x → 
  x = 8 :=
by 
  intros h_hw h_wm h_cb h_rm h_th h_ts h_tx
  sorry

end Carrie_hourly_wage_l157_157799


namespace percent_of_y_l157_157166

theorem percent_of_y (y : ℝ) (h : y > 0) : (2 * y) / 10 + (3 * y) / 10 = (50 / 100) * y :=
by
  sorry

end percent_of_y_l157_157166


namespace distinct_positive_integer_roots_pq_l157_157537

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l157_157537


namespace how_many_buns_each_student_gets_l157_157980

theorem how_many_buns_each_student_gets 
  (packages : ℕ) 
  (buns_per_package : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages = 30)
  (h3 : classes = 4)
  (h4 : students_per_class = 30) :
  (packages * buns_per_package) / (classes * students_per_class) = 2 :=
by sorry

end how_many_buns_each_student_gets_l157_157980


namespace find_pairs_of_positive_integers_l157_157573

theorem find_pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 = 4 * (x^2 * y + x * y^2 - 5) → (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1) :=
by
  sorry

end find_pairs_of_positive_integers_l157_157573


namespace square_prism_surface_area_eq_volume_l157_157870

theorem square_prism_surface_area_eq_volume :
  ∃ (a b : ℕ), (a > 0) ∧ (2 * a^2 + 4 * a * b = a^2 * b)
  ↔ (a = 12 ∧ b = 3) ∨ (a = 8 ∧ b = 4) ∨ (a = 6 ∧ b = 6) ∨ (a = 5 ∧ b = 10) :=
by
  sorry

end square_prism_surface_area_eq_volume_l157_157870


namespace find_m_even_fn_l157_157951

theorem find_m_even_fn (m : ℝ) (f : ℝ → ℝ) 
  (Hf : ∀ x : ℝ, f x = x * (10^x + m * 10^(-x))) 
  (Heven : ∀ x : ℝ, f (-x) = f x) : m = -1 := by
  sorry

end find_m_even_fn_l157_157951


namespace harper_water_intake_l157_157107

theorem harper_water_intake
  (cases_cost : ℕ := 12)
  (cases_count : ℕ := 24)
  (total_spent : ℕ)
  (days : ℕ)
  (total_days_spent : ℕ := 240)
  (total_money_spent: ℕ := 60)
  (total_water: ℕ := 5 * 24)
  (water_per_day : ℝ := 0.5):
  total_spent = total_money_spent ->
  days = total_days_spent ->
  water_per_day = (total_water : ℝ) / total_days_spent :=
by
  sorry

end harper_water_intake_l157_157107


namespace set_equality_l157_157164

noncomputable def alpha_set : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi / 2 - Real.pi / 5 ∧ (-Real.pi < α ∧ α < Real.pi)}

theorem set_equality : alpha_set = {-Real.pi / 5, -7 * Real.pi / 10, 3 * Real.pi / 10, 4 * Real.pi / 5} :=
by
  -- proof omitted
  sorry

end set_equality_l157_157164


namespace sin_minus_cos_value_complex_trig_value_l157_157216

noncomputable def sin_cos_equation (x : Real) :=
  -Real.pi / 2 < x ∧ x < Real.pi / 2 ∧ Real.sin x + Real.cos x = -1 / 5

theorem sin_minus_cos_value (x : Real) (h : sin_cos_equation x) :
  Real.sin x - Real.cos x = 7 / 5 :=
sorry

theorem complex_trig_value (x : Real) (h : sin_cos_equation x) :
  (Real.sin (Real.pi + x) + Real.sin (3 * Real.pi / 2 - x)) / 
  (Real.tan (Real.pi - x) + Real.sin (Real.pi / 2 - x)) = 3 / 11 :=
sorry

end sin_minus_cos_value_complex_trig_value_l157_157216


namespace range_of_a_l157_157336

variable (a : ℝ)
variable (x : ℝ)

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ∀ x, otimes (x - a) (x + a) < 1) : - 1 / 2 < a ∧ a < 3 / 2 :=
sorry

end range_of_a_l157_157336


namespace vampire_count_after_two_nights_l157_157772

noncomputable def vampire_growth : Nat :=
  let first_night_new_vampires := 3 * 7
  let total_vampires_after_first_night := first_night_new_vampires + 3
  let second_night_new_vampires := total_vampires_after_first_night * (7 + 1)
  second_night_new_vampires + total_vampires_after_first_night

theorem vampire_count_after_two_nights : vampire_growth = 216 :=
by
  -- Skipping the detailed proof steps for now
  sorry

end vampire_count_after_two_nights_l157_157772


namespace wrapping_paper_area_correct_l157_157297

-- Conditions as given in the problem
variables (l w h : ℝ)
variable (hlw : l > w)

-- Definition of the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 2 * h) * (w + 2 * h)

-- Proof statement
theorem wrapping_paper_area_correct (hlw : l > w) : 
  wrapping_paper_area l w h = l * w + 2 * l * h + 2 * w * h + 4 * h^2 :=
by
  sorry

end wrapping_paper_area_correct_l157_157297


namespace correct_expression_l157_157838

theorem correct_expression :
  ¬ (|4| = -4) ∧
  ¬ (|4| = -4) ∧
  (-(4^2) ≠ 16)  ∧
  ((-4)^2 = 16) := by
  sorry

end correct_expression_l157_157838


namespace ratio_of_speeds_l157_157739

variable (x y n : ℝ)

-- Conditions
def condition1 : Prop := 3 * (x - y) = n
def condition2 : Prop := 2 * (x + y) = n

-- Problem Statement
theorem ratio_of_speeds (h1 : condition1 x y n) (h2 : condition2 x y n) : x = 5 * y :=
by
  sorry

end ratio_of_speeds_l157_157739


namespace pascal_triangle_21st_number_l157_157350

theorem pascal_triangle_21st_number 
: (Nat.choose 22 2) = 231 :=
by 
  sorry

end pascal_triangle_21st_number_l157_157350


namespace find_a_if_odd_f_monotonically_increasing_on_pos_l157_157517

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - 1) / (x + a)

-- Part 1: Proving that a = 0
theorem find_a_if_odd : (∀ x : ℝ, f x a = -f (-x) a) → a = 0 := by sorry

-- Part 2: Proving that f(x) is monotonically increasing on (0, +∞) given a = 0
theorem f_monotonically_increasing_on_pos : (∀ x : ℝ, x > 0 → 
  ∃ y : ℝ, y > 0 ∧ f x 0 < f y 0) := by sorry

end find_a_if_odd_f_monotonically_increasing_on_pos_l157_157517


namespace find_number_l157_157974

theorem find_number (n : ℕ) (h : 2 * 2 + n = 6) : n = 2 := by
  sorry

end find_number_l157_157974


namespace am_gm_inequality_l157_157721

theorem am_gm_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ a + b + c :=
by
  sorry

end am_gm_inequality_l157_157721


namespace sum_reciprocal_l157_157110

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l157_157110


namespace no_perfect_square_l157_157255

theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 2 * 13^n + 5 * 7^n + 26 :=
sorry

end no_perfect_square_l157_157255


namespace trigonometric_identity_l157_157127

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = 10 / 3 :=
sorry

end trigonometric_identity_l157_157127


namespace domain_of_f_l157_157041

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (4 * x - 3))

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.log (4 * x - 3) ≠ 0} = 
  {x : ℝ | x ∈ Set.Ioo (3 / 4) 1 ∪ Set.Ioi 1} :=
by
  sorry

end domain_of_f_l157_157041


namespace arc_length_of_circle_l157_157652

section circle_arc_length

def diameter (d : ℝ) : Prop := d = 4
def central_angle_deg (θ_d : ℝ) : Prop := θ_d = 36

theorem arc_length_of_circle
  (d : ℝ) (θ_d : ℝ) (r : ℝ := d / 2) (θ : ℝ := θ_d * (π / 180)) (l : ℝ := θ * r) :
  diameter d → central_angle_deg θ_d → l = 2 * π / 5 :=
by
  intros h1 h2
  sorry

end circle_arc_length

end arc_length_of_circle_l157_157652


namespace peters_brother_read_percentage_l157_157663

-- Definitions based on given conditions
def total_books : ℕ := 20
def peter_read_percentage : ℕ := 40
def difference_between_peter_and_brother : ℕ := 6

-- Statement to prove
theorem peters_brother_read_percentage :
  peter_read_percentage / 100 * total_books - difference_between_peter_and_brother = 2 → 
  2 / total_books * 100 = 10 := by
  sorry

end peters_brother_read_percentage_l157_157663


namespace divisibility_of_poly_l157_157063

theorem divisibility_of_poly (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x):
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * (y-z) * (z-x) * (x-y) * k :=
by
  sorry

end divisibility_of_poly_l157_157063


namespace sum_of_squares_of_real_solutions_l157_157418

theorem sum_of_squares_of_real_solutions :
  (∀ x : ℝ, |x^2 - 3 * x + 1 / 400| = 1 / 400)
  → ((0^2 : ℝ) + 3^2 + (9 - 1 / 100) = 999 / 100) := sorry

end sum_of_squares_of_real_solutions_l157_157418


namespace power_function_value_l157_157183

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f x = x ^ α) (h₂ : f (1 / 2) = 4) : f 8 = 1 / 64 := by
  sorry

end power_function_value_l157_157183


namespace triangle_area_arithmetic_sequence_l157_157943

theorem triangle_area_arithmetic_sequence :
  ∃ (S_1 S_2 S_3 S_4 S_5 : ℝ) (d : ℝ),
  S_1 + S_2 + S_3 + S_4 + S_5 = 420 ∧
  S_2 = S_1 + d ∧
  S_3 = S_1 + 2 * d ∧
  S_4 = S_1 + 3 * d ∧
  S_5 = S_1 + 4 * d ∧
  S_5 = 112 :=
by
  sorry

end triangle_area_arithmetic_sequence_l157_157943


namespace car_balanced_by_cubes_l157_157587

variable (M Ball Cube : ℝ)

-- Conditions from the problem
axiom condition1 : M = Ball + 2 * Cube
axiom condition2 : M + Cube = 2 * Ball

-- Theorem to prove
theorem car_balanced_by_cubes : M = 5 * Cube := sorry

end car_balanced_by_cubes_l157_157587


namespace total_surface_area_of_square_pyramid_is_correct_l157_157192

-- Define the base side length and height from conditions
def a : ℝ := 3
def PD : ℝ := 4

-- Conditions
def square_pyramid : Prop :=
  let AD := a
  let PA := Real.sqrt (PD^2 - a^2)
  let Area_PAD := (1 / 2) * AD * PA
  let Area_PCD := Area_PAD
  let Area_base := a * a
  let Total_surface_area := Area_base + 2 * Area_PAD + 2 * Area_PCD
  Total_surface_area = 9 + 6 * Real.sqrt 7

-- Theorem statement
theorem total_surface_area_of_square_pyramid_is_correct : square_pyramid := sorry

end total_surface_area_of_square_pyramid_is_correct_l157_157192


namespace evaluate_f_at_4_l157_157158

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem evaluate_f_at_4 : f 4 = 9 := by
  sorry

end evaluate_f_at_4_l157_157158


namespace fraction_of_roll_used_l157_157508

theorem fraction_of_roll_used 
  (x : ℚ) 
  (h1 : 3 * x + 3 * x + x + 2 * x = 9 * x)
  (h2 : 9 * x = (2 / 5)) : 
  x = 2 / 45 :=
by
  sorry

end fraction_of_roll_used_l157_157508


namespace no_transform_to_1998_power_7_l157_157516

theorem no_transform_to_1998_power_7 :
  ∀ n : ℕ, (exists m : ℕ, n = 7^m) ->
  ∀ k : ℕ, n = 10 * k + (n % 10) ->
  ¬ (∃ t : ℕ, (t = (1998 ^ 7))) := 
by sorry

end no_transform_to_1998_power_7_l157_157516


namespace no_int_b_exists_l157_157896

theorem no_int_b_exists (k n a : ℕ) (hk3 : k ≥ 3) (hn3 : n ≥ 3) (hk_odd : k % 2 = 1) (hn_odd : n % 2 = 1)
  (ha1 : a ≥ 1) (hka : k ∣ (2^a + 1)) (hna : n ∣ (2^a - 1)) :
  ¬ ∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
sorry

end no_int_b_exists_l157_157896


namespace ratio_of_sweater_vests_to_shirts_l157_157807

theorem ratio_of_sweater_vests_to_shirts (S V O : ℕ) (h1 : S = 3) (h2 : O = 18) (h3 : O = V * S) : (V : ℚ) / (S : ℚ) = 2 := 
  by
  sorry

end ratio_of_sweater_vests_to_shirts_l157_157807


namespace div_remainder_l157_157195

theorem div_remainder (B x : ℕ) (h1 : B = 301) (h2 : B % 7 = 0) : x = 3 :=
  sorry

end div_remainder_l157_157195


namespace milk_purchase_maximum_l157_157268

theorem milk_purchase_maximum :
  let num_1_liter_bottles := 6
  let num_half_liter_bottles := 6
  let value_per_1_liter_bottle := 20
  let value_per_half_liter_bottle := 15
  let price_per_liter := 22
  let total_value := num_1_liter_bottles * value_per_1_liter_bottle + num_half_liter_bottles * value_per_half_liter_bottle
  total_value / price_per_liter = 5 :=
by
  sorry

end milk_purchase_maximum_l157_157268


namespace ratio_problem_l157_157610

theorem ratio_problem (x : ℕ) : (20 / 1 : ℝ) = (x / 10 : ℝ) → x = 200 := by
  sorry

end ratio_problem_l157_157610


namespace original_solution_concentration_l157_157531

variable (C : ℝ) -- Concentration of the original solution as a percentage.
variable (v_orig : ℝ := 12) -- 12 ounces of the original vinegar solution.
variable (w_added : ℝ := 50) -- 50 ounces of water added.
variable (v_final_pct : ℝ := 7) -- Final concentration of 7%.

theorem original_solution_concentration :
  (C / 100 * v_orig = v_final_pct / 100 * (v_orig + w_added)) →
  C = (v_final_pct * (v_orig + w_added)) / v_orig :=
sorry

end original_solution_concentration_l157_157531


namespace emails_in_inbox_l157_157124

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l157_157124


namespace total_students_correct_l157_157272

-- Define the given conditions
variables (A B C : ℕ)

-- Number of students in class B
def B_def : ℕ := 25

-- Number of students in class A (B is 8 fewer than A)
def A_def : ℕ := B_def + 8

-- Number of students in class C (C is 5 times B)
def C_def : ℕ := 5 * B_def

-- The total number of students
def total_students : ℕ := A_def + B_def + C_def

-- The proof statement
theorem total_students_correct : total_students = 183 := by
  sorry

end total_students_correct_l157_157272


namespace price_increase_needed_l157_157745

theorem price_increase_needed (P : ℝ) (hP : P > 0) : (100 * ((P / (0.85 * P)) - 1)) = 17.65 :=
by
  sorry

end price_increase_needed_l157_157745


namespace gcd_of_78_and_104_l157_157949

theorem gcd_of_78_and_104 : Int.gcd 78 104 = 26 := by
  sorry

end gcd_of_78_and_104_l157_157949


namespace count100DigitEvenNumbers_is_correct_l157_157190

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l157_157190


namespace mean_inequality_l157_157779

variable (a b : ℝ)

-- Conditions: a and b are distinct and non-zero
axiom h₀ : a ≠ b
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0

theorem mean_inequality (h₀ : a ≠ b) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) :=
sorry -- Proof is not provided, only statement.

end mean_inequality_l157_157779


namespace surface_area_sphere_l157_157119

-- Definitions based on conditions
def SA : ℝ := 3
def SB : ℝ := 4
def SC : ℝ := 5
def vertices_perpendicular : Prop := ∀ (a b c : ℝ), (a = SA ∧ b = SB ∧ c = SC) → (a * b * c = SA * SB * SC)

-- Definition of the theorem based on problem and correct answer
theorem surface_area_sphere (h1 : vertices_perpendicular) : 
  4 * Real.pi * ((Real.sqrt (SA^2 + SB^2 + SC^2)) / 2)^2 = 50 * Real.pi :=
by
  -- skip the proof
  sorry

end surface_area_sphere_l157_157119


namespace oranges_in_buckets_l157_157832

theorem oranges_in_buckets :
  ∀ (x : ℕ),
  (22 + x + (x - 11) = 89) →
  (x - 22 = 17) :=
by
  intro x h
  sorry

end oranges_in_buckets_l157_157832


namespace Q_contribution_l157_157955

def P_contribution : ℕ := 4000
def P_months : ℕ := 12
def Q_months : ℕ := 8
def profit_ratio_PQ : ℚ := 2 / 3

theorem Q_contribution :
  ∃ X : ℕ, (P_contribution * P_months) / (X * Q_months) = profit_ratio_PQ → X = 9000 := 
by sorry

end Q_contribution_l157_157955


namespace largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l157_157966

-- Definitions based on conditions
def floor_div_7 (x : ℕ) : ℕ := x / 7
def floor_div_8 (x : ℕ) : ℕ := x / 8

-- The statement of the problem
theorem largest_x_FloorDiv7_eq_FloorDiv8_plus_1 :
  ∃ x : ℕ, (floor_div_7 x = floor_div_8 x + 1) ∧ (∀ y : ℕ, floor_div_7 y = floor_div_8 y + 1 → y ≤ x) ∧ x = 104 :=
sorry

end largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l157_157966


namespace recommended_sleep_hours_l157_157701

theorem recommended_sleep_hours
  (R : ℝ)   -- The recommended number of hours of sleep per day
  (h1 : 2 * 3 + 5 * (0.60 * R) = 30) : R = 8 :=
sorry

end recommended_sleep_hours_l157_157701


namespace jimin_rank_l157_157651

theorem jimin_rank (seokjin_rank : ℕ) (h1 : seokjin_rank = 4) (h2 : ∃ jimin_rank, jimin_rank = seokjin_rank + 1) : 
  ∃ jimin_rank, jimin_rank = 5 := 
by
  sorry

end jimin_rank_l157_157651


namespace only_composite_positive_integer_with_divisors_form_l157_157079

theorem only_composite_positive_integer_with_divisors_form (n : ℕ) (composite : ¬Nat.Prime n ∧ 1 < n)
  (H : ∀ d ∈ Nat.divisors n, ∃ (a r : ℕ), a ≥ 0 ∧ r ≥ 2 ∧ d = a^r + 1) : n = 10 :=
by
  sorry

end only_composite_positive_integer_with_divisors_form_l157_157079


namespace exponentiation_multiplication_identity_l157_157919

theorem exponentiation_multiplication_identity :
  (-4)^(2010) * (-0.25)^(2011) = -0.25 :=
by
  sorry

end exponentiation_multiplication_identity_l157_157919


namespace percentage_error_divide_instead_of_multiply_l157_157317

theorem percentage_error_divide_instead_of_multiply (x : ℝ) : 
  let correct_result := 5 * x 
  let incorrect_result := x / 10 
  let error := correct_result - incorrect_result 
  let percentage_error := (error / correct_result) * 100 
  percentage_error = 98 :=
by
  sorry

end percentage_error_divide_instead_of_multiply_l157_157317


namespace dice_composite_probability_l157_157932

theorem dice_composite_probability :
  let total_outcomes := (8:ℕ)^6
  let non_composite_outcomes := 1 + 4 * 6 
  let composite_probability := 1 - (non_composite_outcomes / total_outcomes) 
  composite_probability = 262119 / 262144 := by
  sorry

end dice_composite_probability_l157_157932


namespace primes_quadratic_roots_conditions_l157_157550

theorem primes_quadratic_roots_conditions (p q : ℕ)
  (hp : Prime p) (hq : Prime q)
  (h1 : ∃ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p) :
  (¬ (∀ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p → (x - y) % 2 = 0)) ∧
  (∃ (x : ℕ), x * 2 = 2 * q ∨ x * q = 2 * q ∧ Prime x) ∧
  (¬ Prime (p * p + 2 * q)) ∧
  (Prime (p - q)) :=
by sorry

end primes_quadratic_roots_conditions_l157_157550


namespace find_N_l157_157004

/--
If 15% of N is 45% of 2003, then N is 6009.
-/
theorem find_N (N : ℕ) (h : 15 / 100 * N = 45 / 100 * 2003) : 
  N = 6009 :=
sorry

end find_N_l157_157004


namespace alice_cranes_ratio_alice_cranes_l157_157525

theorem alice_cranes {A : ℕ} (h1 : A + (1/5 : ℝ) * (1000 - A) + 400 = 1000) :
  A = 500 := by
  sorry

theorem ratio_alice_cranes :
  (500 : ℝ) / 1000 = 1 / 2 := by
  sorry

end alice_cranes_ratio_alice_cranes_l157_157525


namespace shaded_area_is_20_l157_157538

theorem shaded_area_is_20 (large_square_side : ℕ) (num_small_squares : ℕ) 
  (shaded_squares : ℕ) 
  (h1 : large_square_side = 10) (h2 : num_small_squares = 25) 
  (h3 : shaded_squares = 5) : 
  (large_square_side^2 / num_small_squares) * shaded_squares = 20 :=
by
  sorry

end shaded_area_is_20_l157_157538


namespace son_l157_157145

-- Define the context of the problem with conditions
variables (S M : ℕ)

-- Condition 1: The man is 28 years older than his son
def condition1 : Prop := M = S + 28

-- Condition 2: In two years, the man's age will be twice the son's age
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The final statement to prove the son's present age
theorem son's_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 26 :=
by
  sorry

end son_l157_157145


namespace at_least_one_not_less_than_one_third_l157_157069

theorem at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 :=
sorry

end at_least_one_not_less_than_one_third_l157_157069


namespace brittany_money_times_brooke_l157_157661

theorem brittany_money_times_brooke 
  (kent_money : ℕ) (brooke_money : ℕ) (brittany_money : ℕ) (alison_money : ℕ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : alison_money = 4000)
  (h4 : alison_money = brittany_money / 2) :
  brittany_money = 4 * brooke_money :=
by
  sorry

end brittany_money_times_brooke_l157_157661


namespace smallest_real_number_among_minus3_minus2_0_2_is_minus3_l157_157804

theorem smallest_real_number_among_minus3_minus2_0_2_is_minus3 :
  min (min (-3:ℝ) (-2)) (min 0 2) = -3 :=
by {
    sorry
}

end smallest_real_number_among_minus3_minus2_0_2_is_minus3_l157_157804


namespace cone_volume_ratio_l157_157935

theorem cone_volume_ratio (r_C h_C r_D h_D : ℝ) (h_rC : r_C = 20) (h_hC : h_C = 40) 
  (h_rD : r_D = 40) (h_hD : h_D = 20) : 
  (1 / 3 * pi * r_C^2 * h_C) / (1 / 3 * pi * r_D^2 * h_D) = 1 / 2 :=
by
  rw [h_rC, h_hC, h_rD, h_hD]
  sorry

end cone_volume_ratio_l157_157935


namespace total_carpets_l157_157446

theorem total_carpets 
(house1 : ℕ) 
(house2 : ℕ) 
(house3 : ℕ) 
(house4 : ℕ) 
(h1 : house1 = 12) 
(h2 : house2 = 20) 
(h3 : house3 = 10) 
(h4 : house4 = 2 * house3) : 
  house1 + house2 + house3 + house4 = 62 := 
by 
  -- The proof will be inserted here
  sorry

end total_carpets_l157_157446


namespace product_of_real_roots_eq_one_l157_157465

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, x ^ Real.log x = Real.exp 1 → (x = Real.exp 1 ∨ x = Real.exp (-1)) →
  x * (if x = Real.exp 1 then Real.exp (-1) else Real.exp 1) = 1 :=
by sorry

end product_of_real_roots_eq_one_l157_157465


namespace coin_probability_l157_157397

theorem coin_probability :
  let value_quarters : ℚ := 15.00
  let value_nickels : ℚ := 15.00
  let value_dimes : ℚ := 10.00
  let value_pennies : ℚ := 5.00
  let number_quarters := value_quarters / 0.25
  let number_nickels := value_nickels / 0.05
  let number_dimes := value_dimes / 0.10
  let number_pennies := value_pennies / 0.01
  let total_coins := number_quarters + number_nickels + number_dimes + number_pennies
  let probability := (number_quarters + number_dimes) / total_coins
  probability = (1 / 6) := by 
sorry

end coin_probability_l157_157397


namespace total_price_of_25_shirts_l157_157347

theorem total_price_of_25_shirts (S W : ℝ) (H1 : W = S + 4) (H2 : 75 * W = 1500) : 
  25 * S = 400 :=
by
  -- Proof would go here
  sorry

end total_price_of_25_shirts_l157_157347


namespace problem1_l157_157690

theorem problem1 : 20 + (-14) - (-18) + 13 = 37 :=
by
  sorry

end problem1_l157_157690


namespace no_pairs_of_a_and_d_l157_157387

theorem no_pairs_of_a_and_d :
  ∀ (a d : ℝ), (∀ (x y: ℝ), 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0) -> False :=
by 
  sorry

end no_pairs_of_a_and_d_l157_157387


namespace find_positive_integer_pair_l157_157527

theorem find_positive_integer_pair (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ c_n : ℕ, a^n + b^n = c_n^(n + 1)) : a = 2 ∧ b = 2 := 
sorry

end find_positive_integer_pair_l157_157527


namespace surface_area_reduction_of_spliced_cuboid_l157_157047

theorem surface_area_reduction_of_spliced_cuboid 
  (initial_faces : ℕ := 12)
  (faces_lost : ℕ := 2)
  (percentage_reduction : ℝ := (2 / 12) * 100) :
  percentage_reduction = 16.7 :=
by
  sorry

end surface_area_reduction_of_spliced_cuboid_l157_157047


namespace line_passing_through_points_l157_157013

-- Definition of points
def point1 : ℝ × ℝ := (1, 0)
def point2 : ℝ × ℝ := (0, -2)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Theorem statement
theorem line_passing_through_points : 
  line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2 :=
by
  sorry

end line_passing_through_points_l157_157013


namespace triangles_not_necessarily_congruent_l157_157427

-- Define the triangles and their properties
structure Triangle :=
  (A B C : ℝ)

-- Define angles and measures for heights and medians
def angle (t : Triangle) : ℝ := sorry
def height_from (t : Triangle) (v : ℝ) : ℝ := sorry
def median_from (t : Triangle) (v : ℝ) : ℝ := sorry

theorem triangles_not_necessarily_congruent
  (T₁ T₂ : Triangle)
  (h_angle : angle T₁ = angle T₂)
  (h_height : height_from T₁ T₁.B = height_from T₂ T₂.B)
  (h_median : median_from T₁ T₁.C = median_from T₂ T₂.C) :
  ¬ (T₁ = T₂) := 
sorry

end triangles_not_necessarily_congruent_l157_157427


namespace first_tap_fill_time_l157_157650

theorem first_tap_fill_time (T : ℝ) (h1 : T > 0) (h2 : 12 > 0) 
  (h3 : 1/T - 1/12 = 1/12) : T = 6 :=
sorry

end first_tap_fill_time_l157_157650


namespace value_of_x_is_10_l157_157620

-- Define the conditions
def condition1 (x : ℕ) : ℕ := 3 * x
def condition2 (x : ℕ) : ℕ := (26 - x) + 14

-- Define the proof problem
theorem value_of_x_is_10 (x : ℕ) (h1 : condition1 x = condition2 x) : x = 10 :=
by {
  sorry
}

end value_of_x_is_10_l157_157620


namespace candy_distribution_l157_157882

theorem candy_distribution (candy : ℕ) (people : ℕ) (hcandy : candy = 30) (hpeople : people = 5) :
  ∃ k : ℕ, candy - k = people * (candy / people) ∧ k = 0 := 
by
  sorry

end candy_distribution_l157_157882


namespace gcd_binom_is_integer_l157_157220

theorem gcd_binom_is_integer 
  (m n : ℤ) 
  (hm : m ≥ 1) 
  (hn : n ≥ m)
  (gcd_mn : ℤ := Int.gcd m n)
  (binom_nm : ℤ := Nat.choose n.toNat m.toNat) :
  (gcd_mn * binom_nm) % n.toNat = 0 := by
  sorry

end gcd_binom_is_integer_l157_157220


namespace calculate_expression_l157_157506

theorem calculate_expression :
  2^3 - (Real.tan (Real.pi / 3))^2 = 5 := by
  sorry

end calculate_expression_l157_157506


namespace percent_decrease_l157_157956

variable (OriginalPrice : ℝ) (SalePrice : ℝ)

theorem percent_decrease : 
  OriginalPrice = 100 → 
  SalePrice = 30 → 
  ((OriginalPrice - SalePrice) / OriginalPrice) * 100 = 70 :=
by
  intros h1 h2
  sorry

end percent_decrease_l157_157956


namespace isosceles_triangle_perimeter_l157_157339

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 3) (h2 : b = 1) :
  (a = 3 ∧ b = 1) ∧ (a + b > b ∨ b + b > a) → a + a + b = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l157_157339


namespace least_number_subtracted_divisible_l157_157429

theorem least_number_subtracted_divisible (n : ℕ) (divisor : ℕ) (rem : ℕ) :
  n = 427398 → divisor = 15 → n % divisor = rem → rem = 3 → ∃ k : ℕ, n - k = 427395 :=
by
  intros
  use 3
  sorry

end least_number_subtracted_divisible_l157_157429


namespace simplify_radicals_l157_157222

open Real

theorem simplify_radicals : sqrt 72 + sqrt 32 = 10 * sqrt 2 := by
  sorry

end simplify_radicals_l157_157222


namespace z_investment_correct_l157_157298

noncomputable def z_investment 
    (x_investment : ℕ) 
    (y_investment : ℕ) 
    (z_profit : ℕ) 
    (total_profit : ℕ)
    (profit_z : ℕ) : ℕ := 
  let x_time := 12
  let y_time := 12
  let z_time := 8
  let x_share := x_investment * x_time
  let y_share := y_investment * y_time
  let profit_ratio := total_profit - profit_z
  (x_share + y_share) * z_time / profit_ratio

theorem z_investment_correct : 
  z_investment 36000 42000 4032 13860 4032 = 52000 :=
by sorry

end z_investment_correct_l157_157298


namespace triangle_type_l157_157622

theorem triangle_type (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 30) 
  (h2 : c = 15) 
  (h3 : b = 5 * Real.sqrt 3) 
  (h4 : a ≠ 0) 
  (h5 : b ≠ 0)
  (h6 : c ≠ 0) 
  (h7 : 0 < A ∧ A < 180) 
  (h8 : 0 < B ∧ B < 180) 
  (h9 : 0 < C ∧ C < 180) 
  (h10 : A + B + C = 180) : 
  (A = 90 ∨ A = C) ∧ A + B + C = 180 :=
by 
  sorry

end triangle_type_l157_157622


namespace age_double_after_5_years_l157_157996

-- Defining the current ages of the brothers
def older_brother_age := 15
def younger_brother_age := 5

-- Defining the condition
def after_x_years (x : ℕ) := older_brother_age + x = 2 * (younger_brother_age + x)

-- The main theorem with the condition
theorem age_double_after_5_years : after_x_years 5 :=
by sorry

end age_double_after_5_years_l157_157996


namespace vip_seat_cost_is_65_l157_157016

noncomputable def cost_of_VIP_seat (G V_T V : ℕ) (cost : ℕ) : Prop :=
  G + V_T = 320 ∧
  (15 * G + V * V_T = cost) ∧
  V_T = G - 212 → V = 65

theorem vip_seat_cost_is_65 :
  ∃ (G V_T V : ℕ), cost_of_VIP_seat G V_T V 7500 :=
  sorry

end vip_seat_cost_is_65_l157_157016


namespace map_distance_l157_157032

theorem map_distance (scale : ℝ) (d_actual_km : ℝ) (d_actual_m : ℝ) (d_actual_cm : ℝ) (d_map : ℝ) :
  scale = 1 / 250000 →
  d_actual_km = 5 →
  d_actual_m = d_actual_km * 1000 →
  d_actual_cm = d_actual_m * 100 →
  d_map = (1 * d_actual_cm) / (1 / scale) →
  d_map = 2 :=
by sorry

end map_distance_l157_157032


namespace taxi_fare_ride_distance_l157_157270

theorem taxi_fare_ride_distance (fare_first: ℝ) (first_mile: ℝ) (additional_fare_rate: ℝ) (additional_distance: ℝ) (total_amount: ℝ) (tip: ℝ) (x: ℝ) :
  fare_first = 3.00 ∧ first_mile = 0.75 ∧ additional_fare_rate = 0.25 ∧ additional_distance = 0.1 ∧ total_amount = 15 ∧ tip = 3 ∧
  (total_amount - tip) = fare_first + additional_fare_rate * (x - first_mile) / additional_distance → x = 4.35 :=
by
  intros
  sorry

end taxi_fare_ride_distance_l157_157270


namespace tunnel_length_l157_157194

noncomputable def train_speed_mph : ℝ := 75
noncomputable def train_length_miles : ℝ := 1 / 4
noncomputable def passing_time_minutes : ℝ := 3

theorem tunnel_length :
  let speed_mpm := train_speed_mph / 60
  let total_distance_traveled := speed_mpm * passing_time_minutes
  let tunnel_length := total_distance_traveled - train_length_miles
  tunnel_length = 3.5 :=
by
  sorry

end tunnel_length_l157_157194


namespace dice_probability_same_face_l157_157123

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l157_157123


namespace matrix_solution_l157_157859

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -3], ![4, -1]]
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := ![![ -8,  5], ![ 11, -7]]

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![ -1.2, -1.4], ![1.7, 1.9]]

theorem matrix_solution : M * A = B :=
by sorry

end matrix_solution_l157_157859


namespace sum_of_divisors_5_cubed_l157_157412

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l157_157412


namespace num_cows_correct_l157_157729

-- Definitions from the problem's conditions
def total_animals : ℕ := 500
def percentage_chickens : ℤ := 10
def remaining_animals := total_animals - (percentage_chickens * total_animals / 100)
def goats (cows: ℕ) : ℕ := 2 * cows

-- Statement to prove
theorem num_cows_correct : ∃ cows, remaining_animals = cows + goats cows ∧ 3 * cows = 450 :=
by
  sorry

end num_cows_correct_l157_157729


namespace election_total_votes_l157_157539

theorem election_total_votes
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (h1 : votes_A = 55 * total_votes / 100)
  (h2 : votes_B = 35 * total_votes / 100)
  (h3 : votes_C = total_votes - votes_A - votes_B)
  (h4 : votes_A = votes_B + 400) :
  total_votes = 2000 := by
  sorry

end election_total_votes_l157_157539


namespace num_days_c_worked_l157_157596

theorem num_days_c_worked (d : ℕ) :
  let daily_wage_c := 100
  let daily_wage_b := (4 * 20)
  let daily_wage_a := (3 * 20)
  let total_earning := 1480
  let earning_a := 6 * daily_wage_a
  let earning_b := 9 * daily_wage_b
  let earning_c := d * daily_wage_c
  total_earning = earning_a + earning_b + earning_c →
  d = 4 :=
by {
  sorry
}

end num_days_c_worked_l157_157596


namespace price_increase_is_12_percent_l157_157552

theorem price_increase_is_12_percent
    (P : ℝ) (d : ℝ) (P' : ℝ) (sale_price : ℝ) (increase : ℝ) (percentage_increase : ℝ) :
    P = 470 → d = 0.16 → P' = 442.18 → 
    sale_price = P - P * d →
    increase = P' - sale_price →
    percentage_increase = (increase / sale_price) * 100 →
    percentage_increase = 12 :=
  by
  sorry

end price_increase_is_12_percent_l157_157552


namespace emma_average_speed_last_segment_l157_157120

open Real

theorem emma_average_speed_last_segment :
  ∀ (d1 d2 d3 : ℝ) (t1 t2 t3 : ℝ),
    d1 + d2 + d3 = 120 →
    t1 + t2 + t3 = 2 →
    t1 = 2 / 3 → t2 = 2 / 3 → 
    t1 = d1 / 50 → t2 = d2 / 55 → 
    ∃ x : ℝ, t3 = d3 / x ∧ x = 75 := 
by
  intros d1 d2 d3 t1 t2 t3 h1 h2 ht1 ht2 hs1 hs2
  use 75 / (2 / 3)
  -- skipped proof for simplicity
  sorry

end emma_average_speed_last_segment_l157_157120


namespace problem_dorlir_ahmeti_equality_case_l157_157316

theorem problem_dorlir_ahmeti (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h : x^2 + y^2 + z^2 = x + y + z) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 :=
sorry
  
theorem equality_case (x y z : ℝ)
  (hx : x = 0) (hy : y = 0) (hz : z = 0) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) = 3 :=
sorry

end problem_dorlir_ahmeti_equality_case_l157_157316


namespace repeating_decimal_to_fraction_l157_157960

theorem repeating_decimal_to_fraction (h : (0.0909090909 : ℝ) = 1 / 11) : (0.2727272727 : ℝ) = 3 / 11 :=
sorry

end repeating_decimal_to_fraction_l157_157960


namespace b_2016_result_l157_157147

theorem b_2016_result (b : ℕ → ℤ) (h₁ : b 1 = 1) (h₂ : b 2 = 5)
  (h₃ : ∀ n : ℕ, b (n + 2) = b (n + 1) - b n) : b 2016 = -4 := sorry

end b_2016_result_l157_157147


namespace smallest_of_powers_l157_157630

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l157_157630


namespace value_of_f_neg_11_over_2_l157_157254

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodicity (x : ℝ) : f (x + 2) = - (f x)⁻¹
axiom interval_value (h : 2 ≤ 5 / 2 ∧ 5 / 2 ≤ 3) : f (5 / 2) = 5 / 2

theorem value_of_f_neg_11_over_2 : f (-11 / 2) = 5 / 2 :=
by
  sorry

end value_of_f_neg_11_over_2_l157_157254


namespace count_integers_l157_157528

theorem count_integers (n : ℕ) (h : n = 33000) :
  ∃ k : ℕ, k = 1600 ∧
  (∀ x, 1 ≤ x ∧ x ≤ n → (x % 11 = 0 → (x % 3 ≠ 0 ∧ x % 5 ≠ 0) → x ≤ x)) :=
by 
  sorry

end count_integers_l157_157528


namespace candy_pieces_given_l157_157225

theorem candy_pieces_given (initial total : ℕ) (h1 : initial = 68) (h2 : total = 93) :
  total - initial = 25 :=
by
  sorry

end candy_pieces_given_l157_157225


namespace sale_day_intersection_in_july_l157_157020

def is_multiple_of_five (d : ℕ) : Prop :=
  d % 5 = 0

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ (k : ℕ), d = 3 + k * 6

theorem sale_day_intersection_in_july : 
  (∃ d, is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31) = (1 = Nat.card {d | is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31}) :=
by
  sorry

end sale_day_intersection_in_july_l157_157020


namespace find_middle_number_l157_157631

theorem find_middle_number (a b c : ℕ) (h1 : a + b = 16) (h2 : a + c = 21) (h3 : b + c = 27) : b = 11 := by
  sorry

end find_middle_number_l157_157631


namespace no_twelve_consecutive_primes_in_ap_l157_157479

theorem no_twelve_consecutive_primes_in_ap (d : ℕ) (h : d < 2000) :
  ∀ a : ℕ, ¬(∀ n : ℕ, n < 12 → (Prime (a + n * d))) :=
sorry

end no_twelve_consecutive_primes_in_ap_l157_157479


namespace more_students_suggested_bacon_than_mashed_potatoes_l157_157717

-- Define the number of students suggesting each type of food
def students_suggesting_mashed_potatoes := 479
def students_suggesting_bacon := 489

-- State the theorem that needs to be proven
theorem more_students_suggested_bacon_than_mashed_potatoes :
  students_suggesting_bacon - students_suggesting_mashed_potatoes = 10 := 
  by
  sorry

end more_students_suggested_bacon_than_mashed_potatoes_l157_157717


namespace calculate_discount_l157_157510

def original_price := 22
def sale_price := 16

theorem calculate_discount : original_price - sale_price = 6 := 
by
  sorry

end calculate_discount_l157_157510


namespace pyramid_max_volume_height_l157_157248

-- Define the conditions and the theorem
theorem pyramid_max_volume_height
  (a h V : ℝ)
  (SA : ℝ := 2 * Real.sqrt 3)
  (h_eq : h = Real.sqrt (SA^2 - (Real.sqrt 2 * a / 2)^2))
  (V_eq : V = (1 / 3) * a^2 * h)
  (derivative_at_max : ∀ a, (48 * a^3 - 3 * a^5 = 0) → (a = 0 ∨ a = 4))
  (max_a_value : a = 4):
  h = 2 :=
by
  sorry

end pyramid_max_volume_height_l157_157248


namespace trigonometric_identity_l157_157780

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (θ + Real.pi / 4) = 2) : 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := 
sorry

end trigonometric_identity_l157_157780


namespace milk_production_l157_157586

-- Variables representing the problem parameters
variables {a b c f d e g : ℝ}

-- Preconditions
axiom pos_a : a > 0
axiom pos_c : c > 0
axiom pos_f : f > 0
axiom pos_d : d > 0
axiom pos_e : e > 0
axiom pos_g : g > 0

theorem milk_production (a b c f d e g : ℝ) (h_a : a > 0) (h_c : c > 0) (h_f : f > 0) (h_d : d > 0) (h_e : e > 0) (h_g : g > 0) :
  d * e * g * (b / (a * c * f)) = (b * d * e * g) / (a * c * f) := by
  sorry

end milk_production_l157_157586


namespace interior_angles_sum_l157_157814

theorem interior_angles_sum (n : ℕ) (h : ∀ (k : ℕ), k = n → 60 * n = 360) : 
  180 * (n - 2) = 720 :=
by
  sorry

end interior_angles_sum_l157_157814


namespace maria_towels_l157_157529

-- Define the initial total towels
def initial_total : ℝ := 124.5 + 67.7

-- Define the towels given to her mother
def towels_given : ℝ := 85.35

-- Define the remaining towels (this is what we need to prove)
def towels_remaining : ℝ := 106.85

-- The theorem that states Maria ended up with the correct number of towels
theorem maria_towels :
  initial_total - towels_given = towels_remaining :=
by
  -- Here we would provide the proof, but we use sorry for now
  sorry

end maria_towels_l157_157529


namespace binary_addition_subtraction_l157_157563

def bin_10101 : ℕ := 0b10101
def bin_1011 : ℕ := 0b1011
def bin_1110 : ℕ := 0b1110
def bin_110001 : ℕ := 0b110001
def bin_1101 : ℕ := 0b1101
def bin_101100 : ℕ := 0b101100

theorem binary_addition_subtraction :
  bin_10101 + bin_1011 + bin_1110 + bin_110001 - bin_1101 = bin_101100 := 
sorry

end binary_addition_subtraction_l157_157563


namespace q_sufficient_but_not_necessary_for_p_l157_157968

variable (x : ℝ)

def p : Prop := (x - 2) ^ 2 ≤ 1
def q : Prop := 2 / (x - 1) ≥ 1

theorem q_sufficient_but_not_necessary_for_p : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) := 
by
  sorry

end q_sufficient_but_not_necessary_for_p_l157_157968


namespace total_pencils_is_5_l157_157800

-- Define the initial number of pencils and the number of pencils Tim added
def initial_pencils : Nat := 2
def pencils_added_by_tim : Nat := 3

-- Prove the total number of pencils is equal to 5
theorem total_pencils_is_5 : initial_pencils + pencils_added_by_tim = 5 := by
  sorry

end total_pencils_is_5_l157_157800


namespace solve_exponents_l157_157056

theorem solve_exponents (x y z : ℕ) (hx : x < y) (hy : y < z) 
  (h : 3^x + 3^y + 3^z = 179415) : x = 4 ∧ y = 7 ∧ z = 11 :=
by sorry

end solve_exponents_l157_157056


namespace at_least_one_zero_l157_157513

theorem at_least_one_zero (a b : ℤ) : (¬ (a ≠ 0) ∨ ¬ (b ≠ 0)) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end at_least_one_zero_l157_157513


namespace find_x_l157_157244

theorem find_x (x : ℝ) (h : (40 / 80) = Real.sqrt (x / 80)) : x = 20 := 
by 
  sorry

end find_x_l157_157244


namespace cost_of_five_juices_l157_157930

-- Given conditions as assumptions
variables {J S : ℝ}

axiom h1 : 2 * S = 6
axiom h2 : S + J = 5

-- Prove the statement
theorem cost_of_five_juices : 5 * J = 10 :=
sorry

end cost_of_five_juices_l157_157930


namespace elephant_weight_equivalence_l157_157433

variable (y : ℝ)
variable (porter_weight : ℝ := 120)
variable (blocks_1 : ℝ := 20)
variable (blocks_2 : ℝ := 21)
variable (porters_1 : ℝ := 3)
variable (porters_2 : ℝ := 1)

theorem elephant_weight_equivalence :
  (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 := 
sorry

end elephant_weight_equivalence_l157_157433


namespace distance_between_Jay_and_Paul_l157_157314

theorem distance_between_Jay_and_Paul
  (initial_distance : ℕ)
  (jay_speed : ℕ)
  (paul_speed : ℕ)
  (time : ℕ)
  (jay_distance_walked : ℕ)
  (paul_distance_walked : ℕ) :
  initial_distance = 3 →
  jay_speed = 1 / 20 →
  paul_speed = 3 / 40 →
  time = 120 →
  jay_distance_walked = jay_speed * time →
  paul_distance_walked = paul_speed * time →
  initial_distance + jay_distance_walked + paul_distance_walked = 18 := by
  sorry

end distance_between_Jay_and_Paul_l157_157314


namespace find_expression_value_l157_157170

variable (a b : ℝ)

theorem find_expression_value (h : a - 2 * b = 7) : 6 - 2 * a + 4 * b = -8 := by
  sorry

end find_expression_value_l157_157170


namespace linear_system_reduction_transformation_l157_157499

theorem linear_system_reduction_transformation :
  ∀ (use_substitution_or_elimination : Bool), 
    (use_substitution_or_elimination = true) ∨ (use_substitution_or_elimination = false) → 
    "Reduction and transformation" = "Reduction and transformation" :=
by
  intro use_substitution_or_elimination h
  sorry

end linear_system_reduction_transformation_l157_157499


namespace total_books_l157_157747

def number_of_zoology_books : ℕ := 16
def number_of_botany_books : ℕ := 4 * number_of_zoology_books

theorem total_books : number_of_zoology_books + number_of_botany_books = 80 := by
  sorry

end total_books_l157_157747


namespace range_of_a_l157_157100

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + a * x - 2 < 0) → a < -1 :=
by
  sorry

end range_of_a_l157_157100


namespace domain_of_function_l157_157789

theorem domain_of_function :
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 0)) :=
sorry

end domain_of_function_l157_157789


namespace arithmetic_sequence_S7_geometric_sequence_k_l157_157861

noncomputable def S_n (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_S7 (a_4 : ℕ) (h : a_4 = 8) : S_n a_4 1 7 = 56 := by
  sorry

def Sn_formula (k : ℕ) : ℕ := k^2 + k
def a (i d : ℕ) := i * d

theorem geometric_sequence_k (a_1 k : ℕ) (h1 : a_1 = 2) (h2 : (2 * k + 2)^2 = 6 * (k^2 + k)) :
  k = 2 := by
  sorry

end arithmetic_sequence_S7_geometric_sequence_k_l157_157861


namespace distinct_solutions_for_quadratic_l157_157757

theorem distinct_solutions_for_quadratic (n : ℕ) : ∃ (xs : Finset ℤ), xs.card = n ∧ ∀ x ∈ xs, ∃ y : ℤ, x^2 + 2^(n + 1) = y^2 :=
by sorry

end distinct_solutions_for_quadratic_l157_157757


namespace rain_at_least_once_prob_l157_157169

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l157_157169


namespace passing_time_for_platform_l157_157152

def train_length : ℕ := 1100
def time_to_cross_tree : ℕ := 110
def platform_length : ℕ := 700
def speed := train_length / time_to_cross_tree
def combined_length := train_length + platform_length

theorem passing_time_for_platform : 
  let speed := train_length / time_to_cross_tree
  let combined_length := train_length + platform_length
  combined_length / speed = 180 :=
by
  sorry

end passing_time_for_platform_l157_157152


namespace measure_angle_ACB_l157_157595

-- Definitions of angles and a given triangle
variable (α β γ : ℝ)
variable (angleABD angle75 : ℝ)
variable (triangleABC : Prop)

-- Conditions from the problem
def angle_supplementary : Prop := angleABD + α = 180
def sum_angles_triangle : Prop := α + β + γ = 180
def known_angle : Prop := β = 75
def angleABD_value : Prop := angleABD = 150

-- The theorem to prove
theorem measure_angle_ACB : 
  angle_supplementary angleABD α ∧
  sum_angles_triangle α β γ ∧
  known_angle β ∧
  angleABD_value angleABD
  → γ = 75 := by
  sorry


end measure_angle_ACB_l157_157595


namespace problem1_l157_157686

theorem problem1 (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 := 
sorry

end problem1_l157_157686


namespace factorize_a3_sub_a_l157_157644

theorem factorize_a3_sub_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a3_sub_a_l157_157644


namespace equilateral_triangle_area_l157_157180

theorem equilateral_triangle_area (h : Real) (h_eq : h = Real.sqrt 12):
  ∃ A : Real, A = 12 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l157_157180


namespace num_integers_condition_l157_157249

theorem num_integers_condition : 
  (∃ (n1 n2 n3 : ℤ), 0 < n1 ∧ n1 < 30 ∧ (∃ k1 : ℤ, (30 - n1) / n1 = k1 ^ 2) ∧
                     0 < n2 ∧ n2 < 30 ∧ (∃ k2 : ℤ, (30 - n2) / n2 = k2 ^ 2) ∧
                     0 < n3 ∧ n3 < 30 ∧ (∃ k3 : ℤ, (30 - n3) / n3 = k3 ^ 2) ∧
                     ∀ n : ℤ, 0 < n ∧ n < 30 ∧ (∃ k : ℤ, (30 - n) / n = k ^ 2) → 
                              (n = n1 ∨ n = n2 ∨ n = n3)) :=
sorry

end num_integers_condition_l157_157249


namespace stuffed_dogs_count_l157_157906

theorem stuffed_dogs_count (D : ℕ) (h1 : 14 + D % 7 = 0) : D = 7 :=
by {
  sorry
}

end stuffed_dogs_count_l157_157906


namespace solve_inequalities_l157_157993

theorem solve_inequalities :
  {x : ℤ | (x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - 5 < -3 * x} = {-1, 0} :=
by
  sorry

end solve_inequalities_l157_157993


namespace sum_of_seven_digits_l157_157444

theorem sum_of_seven_digits : 
  ∃ (digits : Finset ℕ), 
    digits.card = 7 ∧ 
    digits ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    ∃ (a b c d e f g : ℕ), 
      a + b + c = 25 ∧ 
      d + e + f + g = 17 ∧ 
      digits = {a, b, c, d, e, f, g} ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
      c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
      d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
      e ≠ f ∧ e ≠ g ∧
      f ≠ g ∧
      (a + b + c + d + e + f + g = 33) := sorry

end sum_of_seven_digits_l157_157444


namespace football_team_selection_l157_157087

theorem football_team_selection :
  let team_members : ℕ := 12
  let offensive_lineman_choices : ℕ := 4
  let tight_end_choices : ℕ := 2
  let players_left_after_offensive : ℕ := team_members - 1
  let players_left_after_tightend : ℕ := players_left_after_offensive - 1
  let quarterback_choices : ℕ := players_left_after_tightend
  let players_left_after_quarterback : ℕ := quarterback_choices - 1
  let running_back_choices : ℕ := players_left_after_quarterback
  let players_left_after_runningback : ℕ := running_back_choices - 1
  let wide_receiver_choices : ℕ := players_left_after_runningback
  offensive_lineman_choices * tight_end_choices * 
  quarterback_choices * running_back_choices * 
  wide_receiver_choices = 5760 := 
by 
  sorry

end football_team_selection_l157_157087


namespace correct_answer_is_B_l157_157865

def lack_of_eco_friendly_habits : Prop := true
def major_global_climate_change_cause (s : String) : Prop :=
  s = "cause"

theorem correct_answer_is_B :
  major_global_climate_change_cause "cause" ∧ lack_of_eco_friendly_habits → "B" = "cause" :=
by
  sorry

end correct_answer_is_B_l157_157865


namespace product_of_powers_eq_nine_l157_157910

variable (a : ℕ)

theorem product_of_powers_eq_nine : a^3 * a^6 = a^9 := 
by sorry

end product_of_powers_eq_nine_l157_157910


namespace A_in_second_quadrant_l157_157052

-- Define the coordinates of point A
def A_x : ℝ := -2
def A_y : ℝ := 3

-- Define the condition that point A lies in the second quadrant
def is_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State the theorem
theorem A_in_second_quadrant : is_second_quadrant A_x A_y :=
by
  -- The proof will be provided here.
  sorry

end A_in_second_quadrant_l157_157052


namespace solve_equation_l157_157858

theorem solve_equation (x y : ℕ) (h_xy : x ≠ y) : x = 2 ∧ y = 4 ∨ x = 4 ∧ y = 2 :=
by {
  sorry -- Proof skipped
}

end solve_equation_l157_157858


namespace N_is_composite_l157_157218

def N : ℕ := 2011 * 2012 * 2013 * 2014 + 1

theorem N_is_composite : ¬ Prime N := by
  sorry

end N_is_composite_l157_157218


namespace ant_crawling_routes_ratio_l157_157713

theorem ant_crawling_routes_ratio 
  (m n : ℕ) 
  (h1 : m = 2) 
  (h2 : n = 6) : 
  n / m = 3 :=
by
  -- Proof is omitted (we only need the statement as per the instruction)
  sorry

end ant_crawling_routes_ratio_l157_157713


namespace factorize_expression_l157_157921

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l157_157921


namespace two_digit_number_satisfies_conditions_l157_157356

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end two_digit_number_satisfies_conditions_l157_157356


namespace minimum_value_l157_157071

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) :
  (1 / x + 1 / y) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end minimum_value_l157_157071


namespace seating_arrangement_six_people_l157_157965

theorem seating_arrangement_six_people : 
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e f : ℕ),
    -- Alice, Bob, and Carla indexing
    1 ≤ a ∧ a ≤ 6 ∧ 
    1 ≤ b ∧ b ≤ 6 ∧ 
    1 ≤ c ∧ c ≤ 6 ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    (a ≠ b + 1 ∧ a ≠ b - 1) ∧
    (a ≠ c + 1 ∧ a ≠ c - 1) ∧
    
    -- Derek, Eric, and Fiona indexing
    1 ≤ d ∧ d ≤ 6 ∧ 
    1 ≤ e ∧ e ≤ 6 ∧ 
    1 ≤ f ∧ f ≤ 6 ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    (d ≠ e + 1 ∧ d ≠ e - 1) ∧
    (d ≠ f + 1 ∧ d ≠ f - 1)) -> 
  n = 216 := 
sorry

end seating_arrangement_six_people_l157_157965


namespace polygon_sides_l157_157647

theorem polygon_sides (n : ℕ) (hn : (n - 2) * 180 = 5 * 360) : n = 12 :=
by
  sorry

end polygon_sides_l157_157647


namespace find_legs_of_right_triangle_l157_157603

theorem find_legs_of_right_triangle (x y a Δ : ℝ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : 2 * Δ = x * y) : 
  x = (Real.sqrt (a^2 + 4 * Δ) + Real.sqrt (a^2 - 4 * Δ)) / 2 ∧ 
  y = (Real.sqrt (a^2 + 4 * Δ) - Real.sqrt (a^2 - 4 * Δ)) / 2 :=
sorry

end find_legs_of_right_triangle_l157_157603


namespace max_min_values_of_f_l157_157306

noncomputable def f (x : ℝ) : ℝ := x^2

theorem max_min_values_of_f : 
  (∀ x, -3 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 0) :=
by
  sorry

end max_min_values_of_f_l157_157306


namespace days_collected_money_l157_157771

-- Defining constants and parameters based on the conditions
def households_per_day : ℕ := 20
def money_per_pair : ℕ := 40
def total_money_collected : ℕ := 2000
def money_from_households : ℕ := (households_per_day / 2) * money_per_pair

-- The theorem that needs to be proven
theorem days_collected_money :
  (total_money_collected / money_from_households) = 5 :=
sorry -- Proof not provided

end days_collected_money_l157_157771


namespace f_neg2_range_l157_157784

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem f_neg2_range (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2) (h2 : 2 ≤ f (1) ∧ f (1) ≤ 4) :
  ∀ k, f (-2) = k → 5 ≤ k ∧ k ≤ 10 :=
  sorry

end f_neg2_range_l157_157784


namespace length_of_EF_l157_157416

theorem length_of_EF (AB BC : ℝ) (DE DF : ℝ) (Area_ABC : ℝ) (Area_DEF : ℝ) (EF : ℝ) 
  (h₁ : AB = 10) (h₂ : BC = 15) (h₃ : DE = DF) (h₄ : Area_DEF = (1/3) * Area_ABC) 
  (h₅ : Area_ABC = AB * BC) (h₆ : Area_DEF = (1/2) * (DE * DF)) : 
  EF = 10 * Real.sqrt 2 := 
by 
  sorry

end length_of_EF_l157_157416


namespace largest_multiple_of_9_less_than_110_l157_157377

theorem largest_multiple_of_9_less_than_110 : ∃ x, (x < 110 ∧ x % 9 = 0 ∧ ∀ y, (y < 110 ∧ y % 9 = 0) → y ≤ x) ∧ x = 108 :=
by
  sorry

end largest_multiple_of_9_less_than_110_l157_157377


namespace find_valid_pairs_l157_157457

def divides (a b : Nat) : Prop := ∃ k, b = a * k

def valid_pair (a b : Nat) : Prop :=
  divides (a^2 * b) (b^2 + 3 * a)

theorem find_valid_pairs :
  {ab | valid_pair ab.1 ab.2} = ({(1, 1), (1, 3)} : Set (Nat × Nat)) :=
by
  sorry

end find_valid_pairs_l157_157457


namespace colorings_10x10_board_l157_157367

def colorings_count (n : ℕ) : ℕ := 2^11 - 2

theorem colorings_10x10_board : colorings_count 10 = 2046 :=
by
  sorry

end colorings_10x10_board_l157_157367


namespace quadratic_inequality_solution_l157_157984

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -12 * x^2 + 5 * x - 2 < 0 := by
  sorry

end quadratic_inequality_solution_l157_157984


namespace avg_of_combined_data_l157_157475

variables (x1 x2 x3 y1 y2 y3 a b : ℝ)

-- condition: average of x1, x2, x3 is a
axiom h1 : (x1 + x2 + x3) / 3 = a

-- condition: average of y1, y2, y3 is b
axiom h2 : (y1 + y2 + y3) / 3 = b

-- Prove that the average of 3x1 + y1, 3x2 + y2, 3x3 + y3 is 3a + b
theorem avg_of_combined_data : 
  ((3 * x1 + y1) + (3 * x2 + y2) + (3 * x3 + y3)) / 3 = 3 * a + b :=
by
  sorry

end avg_of_combined_data_l157_157475


namespace arithmetic_sequence_sum_ratio_l157_157340

theorem arithmetic_sequence_sum_ratio 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (a : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (a_n_def : ∀ n, a_n n = a + (n - 1) * d) 
  (S_n_def : ∀ n, S_n n = n * (2 * a + (n - 1) * d) / 2) 
  (h : 3 * (a + 4 * d) = 5 * (a + 2 * d)) : 
  S_n 5 / S_n 3 = 5 / 2 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l157_157340


namespace apples_to_grapes_proof_l157_157308

theorem apples_to_grapes_proof :
  (3 / 4 * 12 = 9) → (1 / 3 * 9 = 3) :=
by
  sorry

end apples_to_grapes_proof_l157_157308


namespace RachelFurnitureAssemblyTime_l157_157102

/-- Rachel bought seven new chairs and three new tables for her house.
    She spent four minutes on each piece of furniture putting it together.
    Prove that it took her 40 minutes to finish putting together all the furniture. -/
theorem RachelFurnitureAssemblyTime :
  let chairs := 7
  let tables := 3
  let time_per_piece := 4
  let total_time := (chairs + tables) * time_per_piece
  total_time = 40 := by
    sorry

end RachelFurnitureAssemblyTime_l157_157102


namespace f_3_minus_f_4_l157_157987

noncomputable def f : ℝ → ℝ := sorry
axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom initial_condition : f 1 = 1

theorem f_3_minus_f_4 : f 3 - f 4 = -1 :=
by
  sorry

end f_3_minus_f_4_l157_157987


namespace simplify_and_evaluate_l157_157466

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 2) (h₃ : x ≠ -2) :
  ((x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2 * x + 1))) = x + 1 ∧ ((x = 1) → (x + 1 = 2)) :=
by
  sorry

end simplify_and_evaluate_l157_157466


namespace stickers_total_proof_l157_157556

def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem stickers_total_proof : total_stickers = 220 := by
  sorry

end stickers_total_proof_l157_157556


namespace jump_rope_total_l157_157958

theorem jump_rope_total :
  (56 * 3) + (35 * 4) = 308 :=
by
  sorry

end jump_rope_total_l157_157958


namespace smallest_sum_of_xy_l157_157206

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l157_157206


namespace difference_white_black_l157_157714

def total_stones : ℕ := 928
def white_stones : ℕ := 713
def black_stones : ℕ := total_stones - white_stones

theorem difference_white_black :
  (white_stones - black_stones = 498) :=
by
  -- Leaving the proof for later
  sorry

end difference_white_black_l157_157714


namespace length_of_goods_train_l157_157785

theorem length_of_goods_train 
  (speed_km_per_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
  (h1 : speed_km_per_hr = 72) (h2 : platform_length_m = 300) (h3 : time_sec = 26) : 
  ∃ length_of_train : ℕ, length_of_train = 220 :=
by
  sorry

end length_of_goods_train_l157_157785


namespace conditional_probability_l157_157619

variable (P : ℕ → ℚ)
variable (A B : ℕ)

def EventRain : Prop := P A = 4/15
def EventWind : Prop := P B = 2/15
def EventBoth : Prop := P (A * B) = 1/10

theorem conditional_probability 
  (h1 : EventRain P A) 
  (h2 : EventWind P B) 
  (h3 : EventBoth P A B) 
  : (P (A * B) / P A) = 3 / 8 := 
by
  sorry

end conditional_probability_l157_157619


namespace fermat_large_prime_solution_l157_157437

theorem fermat_large_prime_solution (n : ℕ) (hn : n > 0) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (x y z : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^n + y^n ≡ z^n [ZMOD p]) :=
sorry

end fermat_large_prime_solution_l157_157437


namespace eagles_points_l157_157778

theorem eagles_points (s e : ℕ) (h1 : s + e = 52) (h2 : s - e = 6) : e = 23 :=
by
  sorry

end eagles_points_l157_157778


namespace max_point_diff_l157_157315

theorem max_point_diff (n : ℕ) : ∃ max_diff, max_diff = 2 :=
by
  -- Conditions from (a)
  -- - \( n \) teams participate in a football tournament.
  -- - Each team plays against every other team exactly once.
  -- - The winning team is awarded 2 points.
  -- - A draw gives -1 point to each team.
  -- - The losing team gets 0 points.
  -- Correct Answer from (b)
  -- - The maximum point difference between teams that are next to each other in the ranking is 2.
  sorry

end max_point_diff_l157_157315


namespace poly_has_integer_roots_iff_a_eq_one_l157_157628

-- Definition: a positive real number
def pos_real (a : ℝ) : Prop := a > 0

-- The polynomial
def p (a : ℝ) (x : ℝ) : ℝ := a^3 * x^3 + a^2 * x^2 + a * x + a

-- The main theorem
theorem poly_has_integer_roots_iff_a_eq_one (a : ℝ) (x : ℤ) :
  (pos_real a ∧ ∃ x : ℤ, p a x = 0) ↔ a = 1 :=
by sorry

end poly_has_integer_roots_iff_a_eq_one_l157_157628


namespace length_of_PQ_l157_157288

theorem length_of_PQ (p : ℝ) (h : p > 0) (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1x2 : x1 + x2 = 3 * p) (hy1 : y1^2 = 2 * p * x1) (hy2 : y2^2 = 2 * p * x2) 
  (focus : ¬ (y1 = 0)) : (abs (x1 - x2 + 2 * p) = 4 * p) := 
sorry

end length_of_PQ_l157_157288


namespace isosceles_triangle_sides_l157_157618

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides_l157_157618


namespace weight_ratio_l157_157441

variable (J : ℕ) (T : ℕ) (L : ℕ) (S : ℕ)

theorem weight_ratio (h_jake_weight : J = 152) (h_total_weight : J + S = 212) (h_weight_loss : L = 32) :
    (J - L) / (T - J) = 2 :=
by
  sorry

end weight_ratio_l157_157441


namespace total_children_is_11_l157_157227

noncomputable def num_of_children (b g : ℕ) := b + g

theorem total_children_is_11 (b g : ℕ) :
  (∃ c : ℕ, b * c + g * (c + 1) = 47) ∧
  (∃ m : ℕ, b * (m + 1) + g * m = 74) → 
  num_of_children b g = 11 :=
by
  -- The proof steps would go here to show that b + g = 11
  sorry

end total_children_is_11_l157_157227


namespace households_subscribing_B_and_C_l157_157252

/-- Each household subscribes to 2 different newspapers.
Residents only subscribe to newspapers A, B, and C.
There are 30 subscriptions for newspaper A.
There are 34 subscriptions for newspaper B.
There are 40 subscriptions for newspaper C.
Thus, the number of households that subscribe to both
newspaper B and newspaper C is 22. -/
theorem households_subscribing_B_and_C (subs_A subs_B subs_C households : ℕ) 
    (hA : subs_A = 30) (hB : subs_B = 34) (hC : subs_C = 40) (h_total : households = (subs_A + subs_B + subs_C) / 2) :
  (households - subs_A) = 22 :=
by
  -- Substitute the values to demonstrate equality based on the given conditions.
  sorry

end households_subscribing_B_and_C_l157_157252


namespace quadratic_has_two_distinct_real_roots_l157_157683

theorem quadratic_has_two_distinct_real_roots (p : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - 3) * (x1 - 2) - p^2 = 0 ∧ (x2 - 3) * (x2 - 2) - p^2 = 0 :=
by
  -- This part will be replaced with the actual proof
  sorry

end quadratic_has_two_distinct_real_roots_l157_157683


namespace apples_left_proof_l157_157372

def apples_left (mike_apples : Float) (nancy_apples : Float) (keith_apples_eaten : Float): Float :=
  mike_apples + nancy_apples - keith_apples_eaten

theorem apples_left_proof :
  apples_left 7.0 3.0 6.0 = 4.0 :=
by
  unfold apples_left
  norm_num
  sorry

end apples_left_proof_l157_157372


namespace range_of_x_l157_157931

theorem range_of_x (x : ℝ) (hx1 : 1 / x ≤ 3) (hx2 : 1 / x ≥ -2) : x ≥ 1 / 3 := 
sorry

end range_of_x_l157_157931


namespace folder_cost_calc_l157_157540

noncomputable def pencil_cost : ℚ := 0.5
noncomputable def dozen_pencils : ℕ := 24
noncomputable def num_folders : ℕ := 20
noncomputable def total_cost : ℚ := 30
noncomputable def total_pencil_cost : ℚ := dozen_pencils * pencil_cost
noncomputable def remaining_cost := total_cost - total_pencil_cost
noncomputable def folder_cost := remaining_cost / num_folders

theorem folder_cost_calc : folder_cost = 0.9 := by
  -- Definitions
  have pencil_cost_def : pencil_cost = 0.5 := rfl
  have dozen_pencils_def : dozen_pencils = 24 := rfl
  have num_folders_def : num_folders = 20 := rfl
  have total_cost_def : total_cost = 30 := rfl
  have total_pencil_cost_def : total_pencil_cost = dozen_pencils * pencil_cost := rfl
  have remaining_cost_def : remaining_cost = total_cost - total_pencil_cost := rfl
  have folder_cost_def : folder_cost = remaining_cost / num_folders := rfl

  -- Calculation steps given conditions
  sorry

end folder_cost_calc_l157_157540


namespace find_a_values_l157_157700

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l157_157700


namespace ratio_milk_water_larger_vessel_l157_157024

-- Definitions for the conditions given in the problem
def ratio_volume (V1 V2 : ℝ) : Prop := V1 / V2 = 3 / 5
def ratio_milk_water_vessel1 (M1 W1 : ℝ) : Prop := M1 / W1 = 1 / 2
def ratio_milk_water_vessel2 (M2 W2 : ℝ) : Prop := M2 / W2 = 3 / 2

-- The final goal to prove
theorem ratio_milk_water_larger_vessel (V1 V2 M1 W1 M2 W2 : ℝ)
  (h1 : ratio_volume V1 V2) 
  (h2 : V1 = M1 + W1) 
  (h3 : V2 = M2 + W2) 
  (h4 : ratio_milk_water_vessel1 M1 W1) 
  (h5 : ratio_milk_water_vessel2 M2 W2) :
  (M1 + M2) / (W1 + W2) = 1 :=
by
  -- Proof is omitted
  sorry

end ratio_milk_water_larger_vessel_l157_157024


namespace production_cost_per_performance_l157_157752

theorem production_cost_per_performance
  (overhead : ℕ)
  (revenue_per_performance : ℕ)
  (num_performances : ℕ)
  (production_cost : ℕ)
  (break_even : num_performances * revenue_per_performance = overhead + num_performances * production_cost) :
  production_cost = 7000 :=
by
  have : num_performances = 9 := by sorry
  have : revenue_per_performance = 16000 := by sorry
  have : overhead = 81000 := by sorry
  exact sorry

end production_cost_per_performance_l157_157752


namespace proof_statement_l157_157512

variables {K_c A_c K_d B_d A_d B_c : ℕ}

def conditions (K_c A_c K_d B_d A_d B_c : ℕ) :=
  K_c > A_c ∧ K_d > B_d ∧ A_d > K_d ∧ B_c > A_c

noncomputable def statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : Prop :=
  A_d > max K_d B_d

theorem proof_statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : statement K_c A_c K_d B_d A_d B_c h :=
sorry

end proof_statement_l157_157512


namespace estate_area_is_correct_l157_157424

noncomputable def actual_area_of_estate (length_in_inches : ℕ) (width_in_inches : ℕ) (scale : ℕ) : ℕ :=
  let actual_length := length_in_inches * scale
  let actual_width := width_in_inches * scale
  actual_length * actual_width

theorem estate_area_is_correct :
  actual_area_of_estate 9 6 350 = 6615000 := by
  -- Here, we would provide the proof steps, but for this exercise, we use sorry.
  sorry

end estate_area_is_correct_l157_157424


namespace scientific_notation_of_1_5_million_l157_157395

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end scientific_notation_of_1_5_million_l157_157395


namespace ratio_of_boys_l157_157575

theorem ratio_of_boys (p : ℝ) (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 := 
by 
  sorry

end ratio_of_boys_l157_157575


namespace fraction_of_remaining_supplies_used_l157_157824

theorem fraction_of_remaining_supplies_used 
  (initial_food : ℕ)
  (food_used_first_day_fraction : ℚ)
  (food_remaining_after_three_days : ℕ) 
  (food_used_second_period_fraction : ℚ) :
  initial_food = 400 →
  food_used_first_day_fraction = 2 / 5 →
  food_remaining_after_three_days = 96 →
  (initial_food - initial_food * food_used_first_day_fraction) * (1 - food_used_second_period_fraction) = food_remaining_after_three_days →
  food_used_second_period_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_remaining_supplies_used_l157_157824


namespace decimal_to_fraction_l157_157706

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l157_157706


namespace hamburger_cost_l157_157873

def annie's_starting_money : ℕ := 120
def num_hamburgers_bought : ℕ := 8
def price_milkshake : ℕ := 3
def num_milkshakes_bought : ℕ := 6
def leftover_money : ℕ := 70

theorem hamburger_cost :
  ∃ (H : ℕ), 8 * H + 6 * price_milkshake = annie's_starting_money - leftover_money ∧ H = 4 :=
by
  use 4
  sorry

end hamburger_cost_l157_157873


namespace max_servings_hot_chocolate_l157_157591

def recipe_servings : ℕ := 5
def chocolate_required : ℕ := 2 -- squares of chocolate required for 5 servings
def sugar_required : ℚ := 1 / 4 -- cups of sugar required for 5 servings
def water_required : ℕ := 1 -- cups of water required (not limiting)
def milk_required : ℕ := 4 -- cups of milk required for 5 servings

def chocolate_available : ℕ := 5 -- squares of chocolate Jordan has
def sugar_available : ℚ := 2 -- cups of sugar Jordan has
def milk_available : ℕ := 7 -- cups of milk Jordan has
def water_available_lots : Prop := True -- Jordan has lots of water (not limited)

def servings_from_chocolate := (chocolate_available / chocolate_required) * recipe_servings
def servings_from_sugar := (sugar_available / sugar_required) * recipe_servings
def servings_from_milk := (milk_available / milk_required) * recipe_servings

def max_servings (a b c : ℚ) : ℚ := min (min a b) c

theorem max_servings_hot_chocolate :
  max_servings servings_from_chocolate servings_from_sugar servings_from_milk = 35 / 4 :=
by
  sorry

end max_servings_hot_chocolate_l157_157591


namespace binom_12_6_l157_157439

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l157_157439


namespace find_m_value_l157_157827

theorem find_m_value :
  ∃ (m : ℝ), (∃ (midpoint: ℝ × ℝ), midpoint = ((5 + m) / 2, 1) ∧ midpoint.1 - 2 * midpoint.2 = 0) -> m = -1 :=
by
  sorry

end find_m_value_l157_157827


namespace seven_distinct_integers_exist_pair_l157_157613

theorem seven_distinct_integers_exist_pair (a : Fin 7 → ℕ) (h_distinct : Function.Injective a)
  (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 126) :
  ∃ i j : Fin 7, i ≠ j ∧ (1 / 2 : ℚ) ≤ (a i : ℚ) / a j ∧ (a i : ℚ) / a j ≤ 2 := sorry

end seven_distinct_integers_exist_pair_l157_157613


namespace extracurricular_popularity_order_l157_157574

def fraction_likes_drama := 9 / 28
def fraction_likes_music := 13 / 36
def fraction_likes_art := 11 / 24

theorem extracurricular_popularity_order :
  fraction_likes_art > fraction_likes_music ∧ 
  fraction_likes_music > fraction_likes_drama :=
by
  sorry

end extracurricular_popularity_order_l157_157574


namespace value_of_expression_l157_157880

theorem value_of_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) : x + 2 * y = 10 :=
by
  -- Proof goes here
  sorry

end value_of_expression_l157_157880


namespace train_pass_platform_in_correct_time_l157_157817

def length_of_train : ℝ := 2500
def time_to_cross_tree : ℝ := 90
def length_of_platform : ℝ := 1500

noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_tree
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance_to_cover / speed_of_train

theorem train_pass_platform_in_correct_time :
  abs (time_to_pass_platform - 143.88) < 0.01 :=
sorry

end train_pass_platform_in_correct_time_l157_157817


namespace mark_savings_l157_157326

-- Given conditions
def original_price : ℝ := 300
def discount_rate : ℝ := 0.20
def cheaper_lens_price : ℝ := 220

-- Definitions derived from conditions
def discount_amount : ℝ := original_price * discount_rate
def discounted_price : ℝ := original_price - discount_amount
def savings : ℝ := discounted_price - cheaper_lens_price

-- Statement to prove
theorem mark_savings : savings = 20 :=
by
  -- Definitions incorporated
  have h1 : discount_amount = 300 * 0.20 := rfl
  have h2 : discounted_price = 300 - discount_amount := rfl
  have h3 : cheaper_lens_price = 220 := rfl
  have h4 : savings = discounted_price - cheaper_lens_price := rfl
  sorry

end mark_savings_l157_157326


namespace simplify_polynomial_simplify_expression_l157_157362

-- Problem 1:
theorem simplify_polynomial (x : ℝ) : 
  2 * x^3 - 4 * x^2 - 3 * x - 2 * x^2 - x^3 + 5 * x - 7 = x^3 - 6 * x^2 + 2 * x - 7 := 
by
  sorry

-- Problem 2:
theorem simplify_expression (m n : ℝ) (A B : ℝ) (hA : A = 2 * m^2 - m * n) (hB : B = m^2 + 2 * m * n - 5) : 
  4 * A - 2 * B = 6 * m^2 - 8 * m * n + 10 := 
by
  sorry

end simplify_polynomial_simplify_expression_l157_157362


namespace sam_added_later_buckets_l157_157688

variable (initial_buckets : ℝ) (total_buckets : ℝ)

def buckets_added_later (initial_buckets total_buckets : ℝ) : ℝ :=
  total_buckets - initial_buckets

theorem sam_added_later_buckets :
  initial_buckets = 1 ∧ total_buckets = 9.8 → buckets_added_later initial_buckets total_buckets = 8.8 := by
  sorry

end sam_added_later_buckets_l157_157688


namespace num_tents_needed_l157_157888

def count_people : ℕ :=
  let matts_family := 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 2 + 2
  let joes_family := 1 + 1 + 3 + 1
  matts_family + joes_family

def house_capacity : ℕ := 6

def tent_capacity : ℕ := 2

theorem num_tents_needed : (count_people - house_capacity) / tent_capacity = 7 := by
  sorry

end num_tents_needed_l157_157888


namespace circle_equation_l157_157702

theorem circle_equation 
  (h k : ℝ) 
  (H_center : k = 2 * h)
  (H_tangent : ∃ (r : ℝ), (h - 1)^2 + (k - 0)^2 = r^2 ∧ r = k) :
  (x - 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_l157_157702


namespace bread_rolls_count_l157_157994

theorem bread_rolls_count (total_items croissants bagels : Nat) 
  (h1 : total_items = 90) 
  (h2 : croissants = 19) 
  (h3 : bagels = 22) : 
  total_items - croissants - bagels = 49 := 
by
  sorry

end bread_rolls_count_l157_157994


namespace interior_angle_regular_octagon_l157_157241

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l157_157241


namespace hexagon_area_is_20_l157_157578

theorem hexagon_area_is_20 :
  let upper_base1 := 3
  let upper_base2 := 2
  let upper_height := 4
  let lower_base1 := 3
  let lower_base2 := 2
  let lower_height := 4
  let upper_trapezoid_area := (upper_base1 + upper_base2) * upper_height / 2
  let lower_trapezoid_area := (lower_base1 + lower_base2) * lower_height / 2
  let total_area := upper_trapezoid_area + lower_trapezoid_area
  total_area = 20 := 
by {
  sorry
}

end hexagon_area_is_20_l157_157578


namespace earl_envelope_rate_l157_157038

theorem earl_envelope_rate:
  ∀ (E L : ℝ),
  L = (2/3) * E ∧
  (E + L = 60) →
  E = 36 :=
by
  intros E L h
  sorry

end earl_envelope_rate_l157_157038


namespace increase_by_150_percent_l157_157569

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l157_157569


namespace find_k_l157_157302

theorem find_k
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), (x - k * y - 5 = 0 ∧ x^2 + y^2 = 10 ∧ (A = (x, y) ∨ B = (x, y))))
  (h2 : (A.fst^2 + A.snd^2 = 10) ∧ (B.fst^2 + B.snd^2 = 10))
  (h3 : (A.fst - k * A.snd - 5 = 0) ∧ (B.fst - k * B.snd - 5 = 0))
  (h4 : A.fst * B.fst + A.snd * B.snd = 0) :
  k = 2 ∨ k = -2 :=
by
  sorry

end find_k_l157_157302


namespace unique_integer_solution_l157_157854

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2 * n^2 + m^2 + n^2 + 6 * m * n ↔ m = 0 ∧ n = 0 :=
by
  sorry

end unique_integer_solution_l157_157854


namespace find_m_l157_157074

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - 4 * x + m = 0) → m = 4 :=
by
  intro h
  sorry

end find_m_l157_157074


namespace BC_equals_expected_BC_l157_157177

def point := ℝ × ℝ -- Define a point as a pair of real numbers (coordinates).

def vector_sub (v1 v2 : point) : point := (v1.1 - v2.1, v1.2 - v2.2) -- Define vector subtraction.

-- Definitions of points A and B and vector AC
def A : point := (-1, 1)
def B : point := (0, 2)
def AC : point := (-2, 3)

-- Calculate vector AB
def AB : point := vector_sub B A

-- Calculate vector BC
def BC : point := vector_sub AC AB

-- Expected result
def expected_BC : point := (-3, 2)

-- Proof statement
theorem BC_equals_expected_BC : BC = expected_BC := by
  unfold BC AB AC A B vector_sub
  simp
  sorry

end BC_equals_expected_BC_l157_157177


namespace algebraic_expression_value_l157_157746

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := 
by 
  sorry

end algebraic_expression_value_l157_157746


namespace minimum_cost_l157_157251

noncomputable def volume : ℝ := 4800
noncomputable def depth : ℝ := 3
noncomputable def base_cost_per_sqm : ℝ := 150
noncomputable def wall_cost_per_sqm : ℝ := 120
noncomputable def base_area (volume depth : ℝ) : ℝ := volume / depth
noncomputable def wall_surface_area (x : ℝ) : ℝ :=
  6 * x + (2 * (volume * depth / x))

noncomputable def construction_cost (x : ℝ) : ℝ :=
  wall_surface_area x * wall_cost_per_sqm + base_area volume depth * base_cost_per_sqm

theorem minimum_cost :
  ∃(x : ℝ), x = 40 ∧ construction_cost x = 297600 := by
  sorry

end minimum_cost_l157_157251


namespace excluded_number_is_35_l157_157320

theorem excluded_number_is_35 (numbers : List ℝ) 
  (h_len : numbers.length = 5)
  (h_avg1 : (numbers.sum / 5) = 27)
  (h_len_excl : (numbers.length - 1) = 4)
  (avg_remaining : ℝ)
  (remaining_numbers : List ℝ)
  (remaining_condition : remaining_numbers.length = 4)
  (h_avg2 : (remaining_numbers.sum / 4) = 25) :
  numbers.sum - remaining_numbers.sum = 35 :=
by sorry

end excluded_number_is_35_l157_157320


namespace sum_of_intersections_l157_157111

theorem sum_of_intersections :
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (∀ x y : ℝ, y = (x - 2)^2 ↔ x + 1 = (y - 2)^2) ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 20) :=
sorry

end sum_of_intersections_l157_157111


namespace inequality_solution_l157_157505

def solution_set_inequality : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem inequality_solution (x : ℝ) : 
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x ∈ solution_set_inequality :=
by 
  sorry

end inequality_solution_l157_157505


namespace projection_of_a_onto_b_l157_157899

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

theorem projection_of_a_onto_b : projection vec_a vec_b = Real.sqrt 5 :=
by
  sorry

end projection_of_a_onto_b_l157_157899


namespace narrow_black_stripes_are_8_l157_157594

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l157_157594


namespace coordinates_of_point_M_l157_157848

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end coordinates_of_point_M_l157_157848


namespace discount_rate_l157_157914

theorem discount_rate (marked_price selling_price discount_rate: ℝ) 
  (h₁: marked_price = 80)
  (h₂: selling_price = 68)
  (h₃: discount_rate = ((marked_price - selling_price) / marked_price) * 100) : 
  discount_rate = 15 :=
by
  sorry

end discount_rate_l157_157914


namespace a6_b6_gt_a4b2_ab4_l157_157709

theorem a6_b6_gt_a4b2_ab4 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 :=
sorry

end a6_b6_gt_a4b2_ab4_l157_157709


namespace S8_value_l157_157149

theorem S8_value (x : ℝ) (h : x + 1/x = 4) (S : ℕ → ℝ) (S_def : ∀ m, S m = x^m + 1/x^m) :
  S 8 = 37634 :=
sorry

end S8_value_l157_157149


namespace solve_for_m_l157_157322

theorem solve_for_m (x m : ℝ) (hx : 0 < x) (h_eq : m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) : m = 6 :=
sorry

end solve_for_m_l157_157322


namespace trip_first_part_distance_l157_157357

theorem trip_first_part_distance (x : ℝ) :
  let total_distance : ℝ := 60
  let speed_first : ℝ := 48
  let speed_remaining : ℝ := 24
  let avg_speed : ℝ := 32
  (x / speed_first + (total_distance - x) / speed_remaining = total_distance / avg_speed) ↔ (x = 30) :=
by sorry

end trip_first_part_distance_l157_157357


namespace total_shingles_for_all_roofs_l157_157001

def roof_A_length : ℕ := 20
def roof_A_width : ℕ := 40
def roof_A_shingles_per_sqft : ℕ := 8

def roof_B_length : ℕ := 25
def roof_B_width : ℕ := 35
def roof_B_shingles_per_sqft : ℕ := 10

def roof_C_length : ℕ := 30
def roof_C_width : ℕ := 30
def roof_C_shingles_per_sqft : ℕ := 12

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def total_area (length : ℕ) (width : ℕ) : ℕ :=
  2 * area length width

def total_shingles_needed (length : ℕ) (width : ℕ) (shingles_per_sqft : ℕ) : ℕ :=
  total_area length width * shingles_per_sqft

theorem total_shingles_for_all_roofs :
  total_shingles_needed roof_A_length roof_A_width roof_A_shingles_per_sqft +
  total_shingles_needed roof_B_length roof_B_width roof_B_shingles_per_sqft +
  total_shingles_needed roof_C_length roof_C_width roof_C_shingles_per_sqft = 51900 :=
by
  sorry

end total_shingles_for_all_roofs_l157_157001


namespace female_students_count_l157_157585

variable (F M : ℕ)

def numberOfMaleStudents (F : ℕ) : ℕ := 3 * F

def totalStudents (F M : ℕ) : Prop := F + M = 52

theorem female_students_count :
  totalStudents F (numberOfMaleStudents F) → F = 13 :=
by
  intro h
  sorry

end female_students_count_l157_157585


namespace minimize_squared_distances_l157_157078

variable {P : ℝ}

/-- Points A, B, C, D, E are collinear with distances AB = 3, BC = 3, CD = 5, and DE = 7 -/
def collinear_points : Prop :=
  ∀ (A B C D E : ℝ), B = A + 3 ∧ C = B + 3 ∧ D = C + 5 ∧ E = D + 7

/-- Define the squared distance function -/
def squared_distances (P A B C D E : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Statement of the proof problem -/
theorem minimize_squared_distances :
  collinear_points →
  ∀ (A B C D E P : ℝ), 
    squared_distances P A B C D E ≥ 181.2 :=
by
  sorry

end minimize_squared_distances_l157_157078


namespace trapezoid_possible_and_area_sum_l157_157856

theorem trapezoid_possible_and_area_sum (a b c d : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : c = 8) (h4 : d = 12) :
  ∃ (S : ℚ), S = 72 := 
by
  -- conditions ensure one pair of sides is parallel
  -- area calculation based on trapezoid properties
  sorry

end trapezoid_possible_and_area_sum_l157_157856


namespace average_marks_of_all_students_l157_157767

theorem average_marks_of_all_students :
  (22 * 40 + 28 * 60) / (22 + 28) = 51.2 :=
by
  sorry

end average_marks_of_all_students_l157_157767


namespace number_of_students_not_enrolled_in_biology_l157_157762

noncomputable def total_students : ℕ := 880

noncomputable def biology_enrollment_percent : ℕ := 40

noncomputable def students_not_enrolled_in_biology : ℕ :=
  (100 - biology_enrollment_percent) * total_students / 100

theorem number_of_students_not_enrolled_in_biology :
  students_not_enrolled_in_biology = 528 :=
by
  -- Proof goes here.
  -- Use sorry to skip the proof for this placeholder:
  sorry

end number_of_students_not_enrolled_in_biology_l157_157762


namespace angle_bisector_slope_l157_157325

theorem angle_bisector_slope :
  let m₁ := 2
  let m₂ := 5
  let k := (7 - 2 * Real.sqrt 5) / 11
  True :=
by admit

end angle_bisector_slope_l157_157325


namespace solution_is_singleton_l157_157604

def solution_set : Set (ℝ × ℝ) := { (x, y) | 2 * x + y = 3 ∧ x - 2 * y = -1 }

theorem solution_is_singleton : solution_set = { (1, 1) } :=
by
  sorry

end solution_is_singleton_l157_157604


namespace real_estate_commission_l157_157553

theorem real_estate_commission (r : ℝ) (P : ℝ) (C : ℝ) (h : r = 0.06) (hp : P = 148000) : C = P * r :=
by
  -- Definitions and proof steps will go here.
  sorry

end real_estate_commission_l157_157553


namespace player_one_wins_l157_157086

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end player_one_wins_l157_157086


namespace common_chord_of_circles_l157_157942

theorem common_chord_of_circles : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 2*x = 0 ∧ x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := 
by 
  sorry

end common_chord_of_circles_l157_157942


namespace coefficients_divisible_by_5_l157_157193

theorem coefficients_divisible_by_5 
  (a b c d : ℤ) 
  (h : ∀ x : ℤ, 5 ∣ (a * x^3 + b * x^2 + c * x + d)) : 
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d := 
by {
  sorry
}

end coefficients_divisible_by_5_l157_157193


namespace purchase_price_of_article_l157_157915

theorem purchase_price_of_article (P : ℝ) (h : 45 = 0.20 * P + 12) : P = 165 :=
by
  sorry

end purchase_price_of_article_l157_157915


namespace spinning_class_frequency_l157_157300

/--
We define the conditions given in the problem:
- duration of each class in hours,
- calorie burn rate per minute,
- total calories burned per week.
We then state that the number of classes James attends per week is equal to 3.
-/
def class_duration_hours : ℝ := 1.5
def calories_per_minute : ℝ := 7
def total_calories_per_week : ℝ := 1890

theorem spinning_class_frequency :
  total_calories_per_week / (class_duration_hours * 60 * calories_per_minute) = 3 :=
by
  sorry

end spinning_class_frequency_l157_157300


namespace JuliaPlayedTuesday_l157_157777

variable (Monday : ℕ) (Wednesday : ℕ) (Total : ℕ)
variable (KidsOnTuesday : ℕ)

theorem JuliaPlayedTuesday :
  Monday = 17 →
  Wednesday = 2 →
  Total = 34 →
  KidsOnTuesday = Total - (Monday + Wednesday) →
  KidsOnTuesday = 15 :=
by
  intros hMon hWed hTot hTue
  rw [hTot, hMon, hWed] at hTue
  exact hTue

end JuliaPlayedTuesday_l157_157777


namespace clock_90_degree_angle_times_l157_157121

noncomputable def first_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 90

noncomputable def second_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 270

theorem clock_90_degree_angle_times :
  ∃ t₁ t₂ : ℝ,
  first_time_90_degree_angle t₁ ∧ 
  second_time_90_degree_angle t₂ ∧ 
  t₁ = (180 / 11 : ℝ) ∧ 
  t₂ = (540 / 11 : ℝ) :=
by
  sorry

end clock_90_degree_angle_times_l157_157121


namespace triangle_area_l157_157488

theorem triangle_area (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (h₄ : a * a + b * b = c * c) :
  (1/2) * a * b = 30 :=
by
  sorry

end triangle_area_l157_157488


namespace basketball_game_l157_157502

theorem basketball_game 
    (a b x : ℕ)
    (h1 : 3 * b = 2 * a)
    (h2 : x = 2 * b)
    (h3 : 2 * a + 3 * b + x = 72) : 
    x = 18 :=
sorry

end basketball_game_l157_157502


namespace fifth_graders_buy_more_l157_157609

-- Define the total payments made by eighth graders and fifth graders
def eighth_graders_payment : ℕ := 210
def fifth_graders_payment : ℕ := 240
def number_of_fifth_graders : ℕ := 25

-- The price per notebook in whole cents
def price_per_notebook (p : ℕ) : Prop :=
  ∃ k1 k2 : ℕ, k1 * p = eighth_graders_payment ∧ k2 * p = fifth_graders_payment

-- The difference in the number of notebooks bought by the fifth graders and the eighth graders
def notebook_difference (p : ℕ) : ℕ :=
  let eighth_graders_notebooks := eighth_graders_payment / p
  let fifth_graders_notebooks := fifth_graders_payment / p
  fifth_graders_notebooks - eighth_graders_notebooks

-- Theorem stating the difference in the number of notebooks equals 2
theorem fifth_graders_buy_more (p : ℕ) (h : price_per_notebook p) : notebook_difference p = 2 :=
  sorry

end fifth_graders_buy_more_l157_157609


namespace balls_color_equality_l157_157202

theorem balls_color_equality (r g b: ℕ) (h1: r + g + b = 20) (h2: b ≥ 7) (h3: r ≥ 4) (h4: b = 2 * g) : 
  r = b ∨ r = g :=
by
  sorry

end balls_color_equality_l157_157202


namespace find_c_l157_157153

-- Given conditions
variables {a b c d e : ℕ} (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e)
variables (h6 : a + b = e - 1) (h7 : a * b = d + 1)

-- Required to prove
theorem find_c : c = 4 := by
  sorry

end find_c_l157_157153


namespace abs_eq_5_iff_l157_157657

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l157_157657


namespace arithmetic_sequence_formula_and_sum_l157_157467

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h0 : a 1 = 1) 
  (h1 : a 3 = -3)
  (hS : ∃ k, S k = -35):
  (∀ n, a n = 3 - 2 * n) ∧ (∃ k, S k = -35 ∧ k = 7) :=
by
  -- Given an arithmetic sequence where a_1 = 1 and a_3 = -3,
  -- prove that the general formula is a_n = 3 - 2n
  -- and the sum of the first k terms S_k = -35 implies k = 7
  sorry

end arithmetic_sequence_formula_and_sum_l157_157467


namespace problem_correctness_l157_157384

theorem problem_correctness
  (correlation_A : ℝ)
  (correlation_B : ℝ)
  (chi_squared : ℝ)
  (P_chi_squared_5_024 : ℝ)
  (P_chi_squared_6_635 : ℝ)
  (P_X_leq_2 : ℝ)
  (P_X_lt_0 : ℝ) :
  correlation_A = 0.66 →
  correlation_B = -0.85 →
  chi_squared = 6.352 →
  P_chi_squared_5_024 = 0.025 →
  P_chi_squared_6_635 = 0.01 →
  P_X_leq_2 = 0.68 →
  P_X_lt_0 = 0.32 →
  (abs correlation_B > abs correlation_A) ∧
  (1 - P_chi_squared_5_024 < 0.99) ∧
  (P_X_lt_0 = 1 - P_X_leq_2) ∧
  (false) := sorry

end problem_correctness_l157_157384


namespace painting_house_cost_l157_157333

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end painting_house_cost_l157_157333


namespace remainder_of_M_mod_210_l157_157514

def M : ℤ := 1234567891011

theorem remainder_of_M_mod_210 :
  (M % 210) = 31 :=
by
  have modulus1 : M % 6 = 3 := by sorry
  have modulus2 : M % 5 = 1 := by sorry
  have modulus3 : M % 7 = 2 := by sorry
  -- Using Chinese Remainder Theorem
  sorry

end remainder_of_M_mod_210_l157_157514


namespace modified_cube_cubies_l157_157976

structure RubiksCube :=
  (original_cubies : ℕ := 27)
  (removed_corners : ℕ := 8)
  (total_layers : ℕ := 3)
  (edges_per_layer : ℕ := 4)
  (faces_center_cubies : ℕ := 6)
  (center_cubie : ℕ := 1)

noncomputable def cubies_with_n_faces (n : ℕ) : ℕ :=
  if n = 4 then 12
  else if n = 1 then 6
  else if n = 0 then 1
  else 0

theorem modified_cube_cubies :
  (cubies_with_n_faces 4 = 12) ∧ (cubies_with_n_faces 1 = 6) ∧ (cubies_with_n_faces 0 = 1) := by
  sorry

end modified_cube_cubies_l157_157976


namespace distinct_real_roots_range_l157_157675

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 - a = 0) ∧ (x2^2 - 4*x2 - a = 0)) ↔ a > -4 :=
by
  sorry

end distinct_real_roots_range_l157_157675


namespace taxes_paid_l157_157176

theorem taxes_paid (gross_pay net_pay : ℤ) (h1 : gross_pay = 450) (h2 : net_pay = 315) :
  gross_pay - net_pay = 135 := 
by 
  rw [h1, h2] 
  norm_num

end taxes_paid_l157_157176


namespace prime_pair_solution_l157_157344

-- Steps a) and b) are incorporated into this Lean statement
theorem prime_pair_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q ∣ 3^p + 3^q ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) :=
sorry

end prime_pair_solution_l157_157344


namespace age_problem_l157_157821

open Classical

noncomputable def sum_cubes_ages (r j m : ℕ) : ℕ :=
  r^3 + j^3 + m^3

theorem age_problem (r j m : ℕ) (h1 : 5 * r + 2 * j = 3 * m)
    (h2 : 3 * m^2 + 2 * j^2 = 5 * r^2) (h3 : Nat.gcd r (Nat.gcd j m) = 1) :
    sum_cubes_ages r j m = 3 := by
  sorry

end age_problem_l157_157821


namespace geometric_series_sum_l157_157373

theorem geometric_series_sum (a r : ℝ) 
  (h1 : a * (1 - r / (1 - r)) = 18) 
  (h2 : a * (r / (1 - r)) = 8) : r = 4 / 5 :=
by sorry

end geometric_series_sum_l157_157373


namespace age_of_youngest_child_l157_157831

theorem age_of_youngest_child (mother_fee : ℝ) (child_fee_per_year : ℝ) 
  (total_fee : ℝ) (t : ℝ) (y : ℝ) (child_fee : ℝ)
  (h_mother_fee : mother_fee = 2.50)
  (h_child_fee_per_year : child_fee_per_year = 0.25)
  (h_total_fee : total_fee = 4.00)
  (h_child_fee : child_fee = total_fee - mother_fee)
  (h_y : y = 6 - 2 * t)
  (h_fee_eq : child_fee = y * child_fee_per_year) : y = 2 := 
by
  sorry

end age_of_youngest_child_l157_157831


namespace prove_angle_C_prove_max_area_l157_157247

open Real

variables {A B C : ℝ} {a b c : ℝ} (abc_is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (R : ℝ) (circumradius_is_sqrt2 : R = sqrt 2)
variables (H : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
variables (law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C)

-- Part 1: Prove that angle C = π / 3
theorem prove_angle_C : C = π / 3 :=
sorry

-- Part 2: Prove that the maximum value of the area S of triangle ABC is (3 * sqrt 3) / 2
theorem prove_max_area : (1 / 2) * a * b * sin C ≤ (3 * sqrt 3) / 2 :=
sorry

end prove_angle_C_prove_max_area_l157_157247


namespace butterfat_milk_mixture_l157_157822

theorem butterfat_milk_mixture :
  ∃ (x : ℝ), 0.10 * x + 0.45 * 8 = 0.20 * (x + 8) ∧ x = 20 := by
  sorry

end butterfat_milk_mixture_l157_157822


namespace course_gender_relationship_expected_value_X_l157_157265

-- Define the data based on the problem statement
def total_students := 450
def total_boys := 250
def total_girls := 200
def boys_course_b := 150
def girls_course_a := 50
def boys_course_a := total_boys - boys_course_b -- 100
def girls_course_b := total_girls - girls_course_a -- 150

-- Test statistic for independence (calculated)
def chi_squared := 22.5
def critical_value := 10.828

-- Null hypothesis for independence
def H0 := "The choice of course is independent of gender"

-- part 1: proving independence rejection based on chi-squared value
theorem course_gender_relationship : chi_squared > critical_value :=
  by sorry

-- For part 2, stratified sampling and expected value
-- Define probabilities and expected value
def P_X_0 := 1/6
def P_X_1 := 1/2
def P_X_2 := 3/10
def P_X_3 := 1/30

def expected_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- part 2: proving expected value E(X) calculation
theorem expected_value_X : expected_X = 6/5 :=
  by sorry

end course_gender_relationship_expected_value_X_l157_157265


namespace units_digit_m_sq_plus_2_m_l157_157029

def m := 2017^2 + 2^2017

theorem units_digit_m_sq_plus_2_m (m := 2017^2 + 2^2017) : (m^2 + 2^m) % 10 = 3 := 
by
  sorry

end units_digit_m_sq_plus_2_m_l157_157029


namespace compare_fractions_l157_157754

variable {a b : ℝ}

theorem compare_fractions (h1 : 3 * a > b) (h2 : b > 0) :
  (a / b) > ((a + 1) / (b + 3)) :=
by
  sorry

end compare_fractions_l157_157754


namespace original_cost_is_49_l157_157608

-- Define the conditions as assumptions
def original_cost_of_jeans (x : ℝ) : Prop :=
  let discounted_price := x / 2
  let wednesday_price := discounted_price - 10
  wednesday_price = 14.5

-- The theorem to prove
theorem original_cost_is_49 :
  ∃ x : ℝ, original_cost_of_jeans x ∧ x = 49 :=
by
  sorry

end original_cost_is_49_l157_157608


namespace polygon_sum_13th_position_l157_157617

theorem polygon_sum_13th_position :
  let sum_n : ℕ := (100 * 101) / 2;
  2 * sum_n = 10100 :=
by
  sorry

end polygon_sum_13th_position_l157_157617


namespace probability_closer_to_6_than_0_is_0_6_l157_157725

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end probability_closer_to_6_than_0_is_0_6_l157_157725


namespace M_inter_N_eq_M_l157_157847

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | y ≥ 1}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end M_inter_N_eq_M_l157_157847


namespace matrix_power_100_l157_157676

def matrix_100_pow : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![200, 1]]

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

theorem matrix_power_100 (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A^100 = matrix_100_pow :=
by
  sorry

end matrix_power_100_l157_157676


namespace point_segment_length_eq_l157_157605

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x ^ 2 / 25 + y ^ 2 / 16 = 1)

noncomputable def line_eq (x : ℝ) : Prop := (x = 3)

theorem point_segment_length_eq :
  ∀ (A B : ℝ × ℝ), (ellipse_eq A.1 A.2) → (ellipse_eq B.1 B.2) → 
  (line_eq A.1) → (line_eq B.1) → (A = (3, 16/5) ∨ A = (3, -16/5)) → 
  (B = (3, 16/5) ∨ B = (3, -16/5)) → 
  |A.2 - B.2| = 32 / 5 := sorry

end point_segment_length_eq_l157_157605


namespace sushi_cost_l157_157750

variable (x : ℕ)

theorem sushi_cost (h1 : 9 * x = 180) : x + (9 * x) = 200 :=
by 
  sorry

end sushi_cost_l157_157750


namespace triangle_tangent_identity_l157_157267

theorem triangle_tangent_identity (A B C : ℝ) (h : A + B + C = Real.pi) : 
  (Real.tan (A / 2) * Real.tan (B / 2)) + (Real.tan (B / 2) * Real.tan (C / 2)) + (Real.tan (C / 2) * Real.tan (A / 2)) = 1 :=
by
  sorry

end triangle_tangent_identity_l157_157267


namespace compute_f_g_2_l157_157857

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem compute_f_g_2 : f (g 2) = -19 := 
by {
  sorry
}

end compute_f_g_2_l157_157857


namespace profit_percent_l157_157396

theorem profit_percent (CP SP : ℤ) (h : CP/SP = 2/3) : (SP - CP) * 100 / CP = 50 := 
by
  sorry

end profit_percent_l157_157396


namespace sin_theta_value_l157_157021

theorem sin_theta_value 
  (θ : ℝ)
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) :
  Real.sin θ = 3/5 :=
sorry

end sin_theta_value_l157_157021


namespace medians_inequality_l157_157456

  variable {a b c : ℝ} (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)

  noncomputable def median_length (a b c : ℝ) : ℝ :=
    1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

  noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
    (a + b + c) / 2

  theorem medians_inequality (m_a m_b m_c s: ℝ)
    (h_ma : m_a = median_length a b c)
    (h_mb : m_b = median_length b c a)
    (h_mc : m_c = median_length c a b)
    (h_s : s = semiperimeter a b c) :
    m_a^2 + m_b^2 + m_c^2 ≥ s^2 := by
  sorry
  
end medians_inequality_l157_157456


namespace range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l157_157161

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

def p (m : ℝ) : Prop :=
  ∀ x ∈ (Set.Ioo m (m + 1)), (x - 9 / x) < 0

def q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3

theorem range_of_m_when_p_true :
  ∀ m : ℝ, p m → 0 ≤ m ∧ m ≤ 2 :=
sorry

theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (0 ≤ m ∧ m ≤ 1) ∨ (2 < m ∧ m < 3) :=
sorry

end range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l157_157161


namespace prob_B_given_A_l157_157633

theorem prob_B_given_A (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.06) (h2 : P_B = 0.08) (h3 : P_A_and_B = 0.02) :
  (P_A_and_B / P_A) = (1 / 3) :=
by
  -- substitute values
  sorry

end prob_B_given_A_l157_157633


namespace cost_of_bananas_l157_157655

/-- We are given that the rate of bananas is $6 per 3 kilograms. -/
def rate_per_3_kg : ℝ := 6

/-- We need to find the cost for 12 kilograms of bananas. -/
def weight_in_kg : ℝ := 12

/-- We are asked to prove that the cost of 12 kilograms of bananas is $24. -/
theorem cost_of_bananas (rate_per_3_kg weight_in_kg : ℝ) :
  (weight_in_kg / 3) * rate_per_3_kg = 24 :=
by
  sorry

end cost_of_bananas_l157_157655


namespace book_arrangement_count_l157_157287

theorem book_arrangement_count :
  let total_books := 6
  let identical_science_books := 3
  let unique_other_books := total_books - identical_science_books
  (total_books! / (identical_science_books! * unique_other_books!)) = 120 := by
  sorry

end book_arrangement_count_l157_157287


namespace calculate_expression_l157_157691

theorem calculate_expression :
  (-0.25) ^ 2014 * (-4) ^ 2015 = -4 :=
by
  sorry

end calculate_expression_l157_157691


namespace monotonic_increasing_condition_l157_157391

open Real

noncomputable def f (x : ℝ) (l a : ℝ) : ℝ := x^2 - x + l + a * log x

theorem monotonic_increasing_condition (l a : ℝ) (x : ℝ) (hx : x > 0) 
  (h : ∀ x, x > 0 → deriv (f l a) x ≥ 0) : 
  a > 1 / 8 :=
by
  sorry

end monotonic_increasing_condition_l157_157391


namespace find_B_l157_157723

variable (A B : Set ℤ)
variable (U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6})

theorem find_B (hU : U = {x | 0 ≤ x ∧ x ≤ 6})
               (hA_complement_B : A ∩ (U \ B) = {1, 3, 5}) :
  B = {0, 2, 4, 6} :=
sorry

end find_B_l157_157723


namespace exterior_angle_hexagon_l157_157743

theorem exterior_angle_hexagon (θ : ℝ) (hθ : θ = 60) (h_sum : θ * 6 = 360) : n = 6 :=
sorry

end exterior_angle_hexagon_l157_157743


namespace new_total_lines_is_240_l157_157209

-- Define the original number of lines, the increase, and the percentage increase
variables (L : ℝ) (increase : ℝ := 110) (percentage_increase : ℝ := 84.61538461538461 / 100)

-- The statement to prove
theorem new_total_lines_is_240 (h : increase = percentage_increase * L) : L + increase = 240 := sorry

end new_total_lines_is_240_l157_157209


namespace mary_score_is_95_l157_157426

theorem mary_score_is_95
  (s c w : ℕ)
  (h1 : s > 90)
  (h2 : s = 35 + 5 * c - w)
  (h3 : c + w = 30)
  (h4 : ∀ c' w', s = 35 + 5 * c' - w' → c + w = c' + w' → (c', w') = (c, w)) :
  s = 95 :=
by
  sorry

end mary_score_is_95_l157_157426


namespace inequality_proof_l157_157023

theorem inequality_proof (x : ℝ) (h₁ : 3/2 ≤ x) (h₂ : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end inequality_proof_l157_157023


namespace range_a_I_range_a_II_l157_157116

variable (a: ℝ)

-- Define the proposition p and q
def p := (Real.sqrt (a^2 + 13) > Real.sqrt 17)
def q := ∀ x, (0 < x ∧ x < 3) → (x^2 - 2 * a * x - 2 = 0)

-- Prove question (I): If proposition p is true, find the range of the real number $a$
theorem range_a_I (h_p : p a) : a < -2 ∨ a > 2 :=
by sorry

-- Prove question (II): If both the proposition "¬q" and "p ∧ q" are false, find the range of the real number $a$
theorem range_a_II (h_neg_q : ¬ q a) (h_p_and_q : ¬ (p a ∧ q a)) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_a_I_range_a_II_l157_157116


namespace eq_sets_M_N_l157_157521

def setM : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def setN : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem eq_sets_M_N : setM = setN := by
  sorry

end eq_sets_M_N_l157_157521


namespace quadratic_coefficients_l157_157625

theorem quadratic_coefficients (a b c : ℝ) (h₀: 0 < a) 
  (h₁: |a + b + c| = 3) 
  (h₂: |4 * a + 2 * b + c| = 3) 
  (h₃: |9 * a + 3 * b + c| = 3) : 
  (a = 6 ∧ b = -24 ∧ c = 21) ∨ (a = 3 ∧ b = -15 ∧ c = 15) ∨ (a = 3 ∧ b = -9 ∧ c = 3) :=
sorry

end quadratic_coefficients_l157_157625


namespace flour_needed_l157_157390

theorem flour_needed (cookies : ℕ) (flour : ℕ) (k : ℕ) (f_whole_wheat f_all_purpose : ℕ) 
  (h : cookies = 45) (h1 : flour = 3) (h2 : k = 90) (h3 : (k / 2) = 45) 
  (h4 : f_all_purpose = (flour * (k / cookies)) / 2) 
  (h5 : f_whole_wheat = (flour * (k / cookies)) / 2) : 
  f_all_purpose = 3 ∧ f_whole_wheat = 3 := 
by
  sorry

end flour_needed_l157_157390


namespace finish_together_in_4_days_l157_157547

-- Definitions for the individual days taken by A, B, and C
def days_for_A := 12
def days_for_B := 24
def days_for_C := 8 -- C's approximated days

-- The rates are the reciprocals of the days
def rate_A := 1 / days_for_A
def rate_B := 1 / days_for_B
def rate_C := 1 / days_for_C

-- The combined rate of A, B, and C
def combined_rate := rate_A + rate_B + rate_C

-- The total days required to finish the work together
def total_days := 1 / combined_rate

-- Theorem stating that the total days required is 4
theorem finish_together_in_4_days : total_days = 4 := 
by 
-- proof omitted
sorry

end finish_together_in_4_days_l157_157547


namespace carolyn_marbles_l157_157726

theorem carolyn_marbles (initial_marbles : ℕ) (shared_items : ℕ) (end_marbles: ℕ) : 
  initial_marbles = 47 → shared_items = 42 → end_marbles = initial_marbles - shared_items → end_marbles = 5 :=
by
  intros h₀ h₁ h₂
  rw [h₀, h₁] at h₂
  exact h₂

end carolyn_marbles_l157_157726


namespace genuine_coin_remains_l157_157826

theorem genuine_coin_remains (n : ℕ) (g f : ℕ) (h : n = 2022) (h_g : g > n/2) (h_f : f = n - g) : 
  (after_moves : ℕ) -> after_moves = n - 1 -> ∃ remaining_g : ℕ, remaining_g > 0 :=
by
  intros
  sorry

end genuine_coin_remains_l157_157826


namespace fraction_meaningful_condition_l157_157261

theorem fraction_meaningful_condition (m : ℝ) : (m + 3 ≠ 0) → (m ≠ -3) :=
by
  intro h
  sorry

end fraction_meaningful_condition_l157_157261


namespace range_of_ab_c2_l157_157196

theorem range_of_ab_c2 (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
    0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
sorry

end range_of_ab_c2_l157_157196


namespace Gumble_words_total_l157_157979

noncomputable def num_letters := 25
noncomputable def exclude_B := 24

noncomputable def total_5_letters_or_less (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 5 then num_letters^n - exclude_B^n else 0

noncomputable def total_Gumble_words : ℕ :=
  (total_5_letters_or_less 1) + (total_5_letters_or_less 2) + (total_5_letters_or_less 3) +
  (total_5_letters_or_less 4) + (total_5_letters_or_less 5)

theorem Gumble_words_total :
  total_Gumble_words = 1863701 := by
  sorry

end Gumble_words_total_l157_157979


namespace find_number_l157_157470

theorem find_number (x : ℕ) : ((52 + x) * 3 - 60) / 8 = 15 → x = 8 :=
by
  sorry

end find_number_l157_157470


namespace figure4_total_length_l157_157526

-- Define the conditions
def top_segments_sum := 3 + 1 + 1  -- Sum of top segments in Figure 3
def bottom_segment := top_segments_sum -- Bottom segment length in Figure 3
def vertical_segment1 := 10  -- First vertical segment length
def vertical_segment2 := 9  -- Second vertical segment length
def remaining_segment := 1  -- The remaining horizontal segment

-- Total length of remaining segments in Figure 4
theorem figure4_total_length : 
  bottom_segment + vertical_segment1 + vertical_segment2 + remaining_segment = 25 := by
  sorry

end figure4_total_length_l157_157526


namespace total_books_received_l157_157178

theorem total_books_received (initial_books additional_books total_books: ℕ)
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  (initial_books + additional_books = 77) := by
  sorry

end total_books_received_l157_157178


namespace good_students_count_l157_157044

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l157_157044


namespace larger_number_l157_157425

theorem larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end larger_number_l157_157425


namespace range_of_m_l157_157046

open Real

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - m * x + m > 0) ↔ (0 < m ∧ m < 4) :=
by
  sorry

end range_of_m_l157_157046


namespace budget_spent_on_research_and_development_l157_157900

theorem budget_spent_on_research_and_development:
  (∀ budget_total : ℝ, budget_total > 0) →
  (∀ transportation : ℝ, transportation = 15) →
  (∃ research_and_development : ℝ, research_and_development ≥ 0) →
  (∀ utilities : ℝ, utilities = 5) →
  (∀ equipment : ℝ, equipment = 4) →
  (∀ supplies : ℝ, supplies = 2) →
  (∀ salaries_degrees : ℝ, salaries_degrees = 234) →
  (∀ total_degrees : ℝ, total_degrees = 360) →
  (∀ percentage_salaries : ℝ, percentage_salaries = (salaries_degrees / total_degrees) * 100) →
  (∀ known_percentages : ℝ, known_percentages = transportation + utilities + equipment + supplies + percentage_salaries) →
  (∀ rnd_percent : ℝ, rnd_percent = 100 - known_percentages) →
  (rnd_percent = 9) :=
  sorry

end budget_spent_on_research_and_development_l157_157900


namespace monotonicity_f_range_of_b_l157_157359

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def p (a b : ℝ) (x : ℝ) : Prop := f a x ≤ 2 * b
def q (b : ℝ) : Prop := ∀ x, (x = -3 → (x^2 + (2*b + 1)*x - b - 1) > 0) ∧ 
                           (x = -2 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 0 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 1 → (x^2 + (2*b + 1)*x - b - 1) > 0)

theorem monotonicity_f (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : ∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 := by
  sorry

theorem range_of_b (b : ℝ) (hp_or : ∃ x, p a b x ∨ q b) (hp_and : ∀ x, ¬(p a b x ∧ q b)) :
    (1/5 < b ∧ b < 1/2) ∨ (b ≥ 5/7) := by
    sorry

end monotonicity_f_range_of_b_l157_157359


namespace three_w_seven_l157_157048

def operation_w (a b : ℤ) : ℤ := b + 5 * a - 3 * a^2

theorem three_w_seven : operation_w 3 7 = -5 :=
by
  sorry

end three_w_seven_l157_157048


namespace fraction_comparison_l157_157118

theorem fraction_comparison (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) :=
by
  sorry

end fraction_comparison_l157_157118


namespace sum_of_roots_is_zero_l157_157534

variables {R : Type*} [Field R] {a b c p q : R}

theorem sum_of_roots_is_zero (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a^3 + p * a + q = 0) (h₅ : b^3 + p * b + q = 0) (h₆ : c^3 + p * c + q = 0) :
  a + b + c = 0 :=
by
  sorry

end sum_of_roots_is_zero_l157_157534


namespace prev_geng_yin_year_2010_is_1950_l157_157589

def heavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "You", "Xu", "Hai"]

def cycleLength : Nat := Nat.lcm 10 12

def prev_geng_yin_year (current_year : Nat) : Nat :=
  if cycleLength ≠ 0 then
    current_year - cycleLength
  else
    current_year -- This line is just to handle the case where LCM is incorrectly zero, which shouldn't happen practically.

theorem prev_geng_yin_year_2010_is_1950 : prev_geng_yin_year 2010 = 1950 := by
  sorry

end prev_geng_yin_year_2010_is_1950_l157_157589


namespace inequality_solution_set_l157_157819

theorem inequality_solution_set (x : ℝ) : 
  (∃ x, (2 < x ∧ x < 3)) ↔ 
  ((x - 2) * (x - 3) / (x^2 + 1) < 0) :=
by sorry

end inequality_solution_set_l157_157819


namespace radius_of_circumscribed_circle_l157_157382

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l157_157382


namespace surface_area_of_sphere_l157_157852

/-- Given a right prism with all vertices on a sphere, a height of 4, and a volume of 64,
    the surface area of this sphere is 48π -/
theorem surface_area_of_sphere (h : ℝ) (V : ℝ) (S : ℝ) :
  h = 4 → V = 64 → S = 48 * Real.pi := by
  sorry

end surface_area_of_sphere_l157_157852


namespace solve_equation_l157_157179

theorem solve_equation : ∀ x : ℝ, x * (x + 2) = 3 * x + 6 ↔ (x = -2 ∨ x = 3) := by
  sorry

end solve_equation_l157_157179


namespace sequence_property_l157_157240

theorem sequence_property :
  ∃ (a_0 a_1 a_2 a_3 : ℕ),
    a_0 + a_1 + a_2 + a_3 = 4 ∧
    (a_0 = ([a_0, a_1, a_2, a_3].count 0)) ∧
    (a_1 = ([a_0, a_1, a_2, a_3].count 1)) ∧
    (a_2 = ([a_0, a_1, a_2, a_3].count 2)) ∧
    (a_3 = ([a_0, a_1, a_2, a_3].count 3)) :=
sorry

end sequence_property_l157_157240


namespace percent_increase_l157_157010

theorem percent_increase (x : ℝ) (h : (1 / 2) * x = 1) : ((x - (1 / 2)) / (1 / 2)) * 100 = 300 := by
  sorry

end percent_increase_l157_157010


namespace negation_equivalence_l157_157684

variable (x : ℝ)

def original_proposition := ∃ x : ℝ, x^2 - 3*x + 3 < 0

def negation_proposition := ∀ x : ℝ, x^2 - 3*x + 3 ≥ 0

theorem negation_equivalence : ¬ original_proposition ↔ negation_proposition :=
by 
  -- Lean doesn’t require the actual proof here
  sorry

end negation_equivalence_l157_157684


namespace LineChart_characteristics_and_applications_l157_157738

-- Definitions related to question and conditions
def LineChart : Type := sorry
def represents_amount (lc : LineChart) : Prop := sorry
def reflects_increase_or_decrease (lc : LineChart) : Prop := sorry

-- Theorem related to the correct answer
theorem LineChart_characteristics_and_applications (lc : LineChart) :
  represents_amount lc ∧ reflects_increase_or_decrease lc :=
sorry

end LineChart_characteristics_and_applications_l157_157738


namespace correct_operation_l157_157386

theorem correct_operation : 
  (a^2 + a^2 = 2 * a^2) = false ∧ 
  ((-3 * a * b^2)^2 = -6 * a^2 * b^4) = false ∧ 
  (a^6 / (-a)^2 = a^4) = true ∧ 
  ((a - b)^2 = a^2 - b^2) = false :=
sorry

end correct_operation_l157_157386


namespace parabola_focus_at_centroid_l157_157922

theorem parabola_focus_at_centroid (A B C : ℝ × ℝ) (a : ℝ) 
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hC : C = (4, -6))
  (h_focus : (a/4, 0) = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a = 8 :=
by
  sorry

end parabola_focus_at_centroid_l157_157922


namespace sid_initial_money_l157_157197

variable (M : ℝ)
variable (spent_on_accessories : ℝ := 12)
variable (spent_on_snacks : ℝ := 8)
variable (remaining_money_condition : ℝ := (M / 2) + 4)

theorem sid_initial_money : (M = 48) → (remaining_money_condition = M - (spent_on_accessories + spent_on_snacks)) :=
by
  sorry

end sid_initial_money_l157_157197


namespace simpl_eval_l157_157327

variable (a b : ℚ)

theorem simpl_eval (h_a : a = 1/2) (h_b : b = -1/3) :
    5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (- a * b ^ 2 + 3 * a ^ 2 * b) = -11 / 36 := by
  sorry

end simpl_eval_l157_157327


namespace profit_days_l157_157635

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

end profit_days_l157_157635


namespace right_triangle_median_l157_157334

noncomputable def median_to_hypotenuse_length (a b : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (a^2 + b^2)
  hypotenuse / 2

theorem right_triangle_median
  (a b : ℝ) (h_a : a = 3) (h_b : b = 4) :
  median_to_hypotenuse_length a b = 2.5 :=
by
  sorry

end right_triangle_median_l157_157334


namespace completing_square_l157_157520

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l157_157520


namespace max_value_expression_l157_157392

noncomputable def a (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def b (φ : ℝ) : ℝ := 3 * Real.sin φ

theorem max_value_expression (φ θ : ℝ) : 
  ∃ c : ℝ, c = 3 * Real.cos (θ - φ) ∧ c ≤ 3 := by
  sorry

end max_value_expression_l157_157392


namespace x_squared_plus_y_squared_l157_157798

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l157_157798


namespace Patrick_fish_count_l157_157948

variable (Angus Patrick Ollie : ℕ)

-- Conditions
axiom h1 : Ollie + 7 = Angus
axiom h2 : Angus = Patrick + 4
axiom h3 : Ollie = 5

-- Theorem statement
theorem Patrick_fish_count : Patrick = 8 := 
by
  sorry

end Patrick_fish_count_l157_157948


namespace johns_percentage_increase_l157_157330

theorem johns_percentage_increase (original_amount new_amount : ℕ) (h₀ : original_amount = 30) (h₁ : new_amount = 40) :
  (new_amount - original_amount) * 100 / original_amount = 33 :=
by
  sorry

end johns_percentage_increase_l157_157330


namespace minimum_area_for_rectangle_l157_157744

theorem minimum_area_for_rectangle 
(length width : ℝ) 
(h_length_min : length = 4 - 0.5) 
(h_width_min : width = 5 - 1) :
length * width = 14 := 
by 
  simp [h_length_min, h_width_min]
  sorry

end minimum_area_for_rectangle_l157_157744


namespace equal_roots_a_l157_157454

theorem equal_roots_a {a : ℕ} :
  (a * a - 4 * (a + 3) = 0) → a = 6 := 
sorry

end equal_roots_a_l157_157454


namespace ribbon_length_per_gift_l157_157524

theorem ribbon_length_per_gift (gifts : ℕ) (initial_ribbon remaining_ribbon : ℝ) (total_used_ribbon : ℝ) (length_per_gift : ℝ):
  gifts = 8 →
  initial_ribbon = 15 →
  remaining_ribbon = 3 →
  total_used_ribbon = initial_ribbon - remaining_ribbon →
  length_per_gift = total_used_ribbon / gifts →
  length_per_gift = 1.5 :=
by
  intros
  sorry

end ribbon_length_per_gift_l157_157524


namespace five_goats_choir_l157_157820

theorem five_goats_choir 
  (total_members : ℕ)
  (num_rows : ℕ)
  (total_members_eq : total_members = 51)
  (num_rows_eq : num_rows = 4) :
  ∃ row_people : ℕ, row_people ≥ 13 :=
by 
  sorry

end five_goats_choir_l157_157820


namespace total_area_three_plots_l157_157823

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots_l157_157823


namespace minimum_value_exists_l157_157756

-- Definitions of the components
noncomputable def quadratic_expression (k x y : ℝ) : ℝ := 
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12

theorem minimum_value_exists (k : ℝ) :
  (∃ x y : ℝ, quadratic_expression k x y = 0) ↔ k = 2 := 
sorry

end minimum_value_exists_l157_157756


namespace fuel_cost_equation_l157_157072

theorem fuel_cost_equation (x : ℝ) (h : (x / 4) - (x / 6) = 8) : x = 96 :=
sorry

end fuel_cost_equation_l157_157072


namespace inequality_proof_l157_157751

theorem inequality_proof (a b c d : ℝ) (hnonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (hsum : a + b + c + d = 1) :
  abcd + bcda + cdab + dabc ≤ 1/27 + (176/27) * abcd :=
by
  sorry

end inequality_proof_l157_157751


namespace school_total_payment_l157_157410

def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def chaperones_per_class : ℕ := 5
def student_fee : ℝ := 5.50
def adult_fee : ℝ := 6.50

def total_students : ℕ := num_classes * students_per_class
def total_adults : ℕ := num_classes * chaperones_per_class

def total_student_cost : ℝ := total_students * student_fee
def total_adult_cost : ℝ := total_adults * adult_fee

def total_cost : ℝ := total_student_cost + total_adult_cost

theorem school_total_payment : total_cost = 1010.0 := by
  sorry

end school_total_payment_l157_157410


namespace faster_train_length_l157_157006

noncomputable def length_of_faster_train 
    (speed_train_1_kmph : ℤ) 
    (speed_train_2_kmph : ℤ) 
    (time_seconds : ℤ) : ℤ := 
    (speed_train_1_kmph + speed_train_2_kmph) * 1000 / 3600 * time_seconds

theorem faster_train_length 
    (speed_train_1_kmph : ℤ)
    (speed_train_2_kmph : ℤ)
    (time_seconds : ℤ)
    (h1 : speed_train_1_kmph = 36)
    (h2 : speed_train_2_kmph = 45)
    (h3 : time_seconds = 12) :
    length_of_faster_train speed_train_1_kmph speed_train_2_kmph time_seconds = 270 :=
by
    sorry

end faster_train_length_l157_157006


namespace perfect_square_divisors_count_l157_157432

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : Nat := factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 *
                                   factorial 6 * factorial 7 * factorial 8 * factorial 9 * factorial 10

def count_perfect_square_divisors (n : Nat) : Nat := sorry -- This would involve the correct function implementation.

theorem perfect_square_divisors_count :
  count_perfect_square_divisors product_of_factorials = 2160 :=
sorry

end perfect_square_divisors_count_l157_157432


namespace impossible_to_reduce_time_l157_157284

def current_speed := 60 -- speed in km/h
def time_per_km (v : ℕ) : ℕ := 60 / v -- 60 minutes divided by speed in km/h gives time per km in minutes

theorem impossible_to_reduce_time (v : ℕ) (h : v = current_speed) : time_per_km v = 1 → ¬(time_per_km v - 1 = 0) :=
by
  intros h1 h2
  sorry

end impossible_to_reduce_time_l157_157284


namespace at_least_one_shooter_hits_target_l157_157501

-- Definition stating the probability of the first shooter hitting the target
def prob_A1 : ℝ := 0.7

-- Definition stating the probability of the second shooter hitting the target
def prob_A2 : ℝ := 0.8

-- The event that at least one shooter hits the target
def prob_at_least_one_hit : ℝ := prob_A1 + prob_A2 - (prob_A1 * prob_A2)

-- Prove that the probability that at least one shooter hits the target is 0.94
theorem at_least_one_shooter_hits_target : prob_at_least_one_hit = 0.94 :=
by
  sorry

end at_least_one_shooter_hits_target_l157_157501


namespace cos_squared_value_l157_157311

theorem cos_squared_value (α : ℝ) (h : Real.tan (α + π/4) = 3/4) : Real.cos (π/4 - α) ^ 2 = 9 / 25 :=
sorry

end cos_squared_value_l157_157311


namespace square_division_possible_l157_157545

theorem square_division_possible :
  ∃ (S a b c : ℕ), 
    S^2 = a^2 + 3 * b^2 + 5 * c^2 ∧ 
    a = 3 ∧ 
    b = 2 ∧ 
    c = 1 :=
  by {
    sorry
  }

end square_division_possible_l157_157545


namespace necessary_but_not_sufficient_condition_l157_157223

def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l157_157223


namespace units_digit_char_of_p_l157_157707

theorem units_digit_char_of_p (p : ℕ) (h_pos : 0 < p) (h_even : p % 2 = 0)
    (h_units_zero : (p^3 % 10) - (p^2 % 10) = 0) (h_units_eleven : (p + 5) % 10 = 1) :
    p % 10 = 6 :=
sorry

end units_digit_char_of_p_l157_157707


namespace frank_money_left_l157_157337

theorem frank_money_left (initial_money : ℝ) (spent_groceries : ℝ) (spent_magazine : ℝ) :
  initial_money = 600 →
  spent_groceries = (1/5) * initial_money →
  spent_magazine = (1/4) * (initial_money - spent_groceries) →
  initial_money - spent_groceries - spent_magazine = 360 := 
by
  intro h1 h2 h3
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  sorry

end frank_money_left_l157_157337


namespace simplify_expression_l157_157895

theorem simplify_expression (x y : ℝ) : 
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 :=
by
  sorry

end simplify_expression_l157_157895


namespace english_students_23_l157_157349

def survey_students_total : Nat := 35
def students_in_all_three : Nat := 2
def solely_english_three_times_than_french (x y : Nat) : Prop := y = 3 * x
def english_but_not_french_or_spanish (x y : Nat) : Prop := y + students_in_all_three = 35 ∧ y - students_in_all_three = 23

theorem english_students_23 :
  ∃ (x y : Nat), solely_english_three_times_than_french x y ∧ english_but_not_french_or_spanish x y :=
by
  sorry

end english_students_23_l157_157349


namespace cookies_in_box_l157_157112

/-- Graeme is weighing cookies to see how many he can fit in his box. His box can only hold
    40 pounds of cookies. If each cookie weighs 2 ounces, how many cookies can he fit in the box? -/
theorem cookies_in_box (box_capacity_pounds : ℕ) (cookie_weight_ounces : ℕ) (pound_to_ounces : ℕ)
  (h_box_capacity : box_capacity_pounds = 40)
  (h_cookie_weight : cookie_weight_ounces = 2)
  (h_pound_to_ounces : pound_to_ounces = 16) :
  (box_capacity_pounds * pound_to_ounces) / cookie_weight_ounces = 320 := by 
  sorry

end cookies_in_box_l157_157112


namespace distance_between_points_l157_157607

theorem distance_between_points :
  ∀ (D : ℝ), (10 + 2) * (5 / D) + (10 - 2) * (5 / D) = 24 ↔ D = 24 := 
sorry

end distance_between_points_l157_157607


namespace min_value_x2_minus_x1_l157_157205

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_value_x2_minus_x1 :
  (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) → |x2 - x1| = 2 :=
sorry

end min_value_x2_minus_x1_l157_157205


namespace work_problem_l157_157338

/-- 
  Suppose A can complete a work in \( x \) days alone, 
  B can complete the work in 20 days,
  and together they work for 7 days, leaving a fraction of 0.18333333333333335 of the work unfinished.
  Prove that \( x = 15 \).
 -/
theorem work_problem (x : ℝ) : 
  (∀ (B : ℝ), B = 20 → (∀ (f : ℝ), f = 0.18333333333333335 → (7 * (1 / x + 1 / B) = 1 - f)) → x = 15) := 
sorry

end work_problem_l157_157338


namespace number_of_five_digit_numbers_l157_157184

def count_five_identical_digits: Nat := 9
def count_two_different_digits: Nat := 1215
def count_three_different_digits: Nat := 6480
def count_four_different_digits: Nat := 22680
def count_five_different_digits: Nat := 27216

theorem number_of_five_digit_numbers :
  count_five_identical_digits + count_two_different_digits +
  count_three_different_digits + count_four_different_digits +
  count_five_different_digits = 57600 :=
by
  sorry

end number_of_five_digit_numbers_l157_157184


namespace unique_peg_placement_l157_157348

theorem unique_peg_placement :
  ∃! f : Fin 6 → Fin 6 → Option (Fin 6), ∀ i j k, 
    (∃ c, f i k = some c) →
    (∃ c, f j k = some c) →
    i = j ∧ match f i j with
    | some c => f j k ≠ some c
    | none => True :=
  sorry

end unique_peg_placement_l157_157348


namespace find_value_of_z_l157_157732

open Complex

-- Define the given complex number z and imaginary unit i
def z : ℂ := sorry
def i : ℂ := Complex.I

-- Given condition
axiom condition : z / (1 - i) = i ^ 2019

-- Proof that z equals -1 - i
theorem find_value_of_z : z = -1 - i :=
by
  sorry

end find_value_of_z_l157_157732


namespace alicia_bought_more_markers_l157_157419

theorem alicia_bought_more_markers (price_per_marker : ℝ) (n_h : ℝ) (n_a : ℝ) (m : ℝ) 
    (h_hector : n_h * price_per_marker = 2.76) 
    (h_alicia : n_a * price_per_marker = 4.07)
    (h_diff : n_a - n_h = m) : 
  m = 13 :=
sorry

end alicia_bought_more_markers_l157_157419


namespace number_of_factors_l157_157559

theorem number_of_factors (a b c d : ℕ) (h₁ : a = 6) (h₂ : b = 6) (h₃ : c = 5) (h₄ : d = 1) :
  ((a + 1) * (b + 1) * (c + 1) * (d + 1) = 588) :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

end number_of_factors_l157_157559


namespace possible_to_divide_into_two_groups_l157_157509

-- Define a type for People
universe u
variable {Person : Type u}

-- Define friend and enemy relations (assume they are given as functions)
variable (friend enemy : Person → Person)

-- Define the main statement
theorem possible_to_divide_into_two_groups (h_friend : ∀ p : Person, ∃ q : Person, friend p = q)
                                           (h_enemy : ∀ p : Person, ∃ q : Person, enemy p = q) :
  ∃ (company : Person → Bool),
    ∀ p : Person, company p ≠ company (friend p) ∧ company p ≠ company (enemy p) :=
by
  sorry

end possible_to_divide_into_two_groups_l157_157509


namespace mark_and_alice_probability_l157_157835

def probability_sunny_days : ℚ := 51 / 250

theorem mark_and_alice_probability :
  (∀ (day : ℕ), day < 5 → (∃ rain_prob sun_prob : ℚ, rain_prob = 0.8 ∧ sun_prob = 0.2 ∧ rain_prob + sun_prob = 1))
  → probability_sunny_days = 51 / 250 :=
by sorry

end mark_and_alice_probability_l157_157835


namespace least_common_multiple_of_812_and_3214_is_correct_l157_157438

def lcm_812_3214 : ℕ :=
  Nat.lcm 812 3214

theorem least_common_multiple_of_812_and_3214_is_correct :
  lcm_812_3214 = 1304124 := by
  sorry

end least_common_multiple_of_812_and_3214_is_correct_l157_157438


namespace compute_fraction_power_l157_157912

theorem compute_fraction_power : (45000 ^ 3 / 15000 ^ 3) = 27 :=
by
  sorry

end compute_fraction_power_l157_157912


namespace solve_for_x_l157_157923

theorem solve_for_x (x : ℝ) (h : 4 / (1 + 3 / x) = 1) : x = 1 :=
sorry

end solve_for_x_l157_157923


namespace left_vertex_of_ellipse_l157_157664

theorem left_vertex_of_ellipse :
  ∃ (a b c : ℝ), 
    (a > b) ∧ (b > 0) ∧ (b = 4) ∧ (c = 3) ∧ 
    (c^2 = a^2 - b^2) ∧ 
    (3^2 = a^2 - 4^2) ∧ 
    (a = 5) ∧ 
    (∀ x y : ℝ, (x, y) = (-5, 0)) := 
sorry

end left_vertex_of_ellipse_l157_157664


namespace permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l157_157833

-- Part (a)
theorem permutations_with_exactly_one_descent (n : ℕ) : 
  ∃ (count : ℕ), count = 2^n - n - 1 := sorry

-- Part (b)
theorem permutations_with_exactly_two_descents (n : ℕ) : 
  ∃ (count : ℕ), count = 3^n - 2^n * (n + 1) + (n * (n + 1)) / 2 := sorry

end permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l157_157833


namespace inequality_proof_l157_157285

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) :=
by
  sorry

end inequality_proof_l157_157285


namespace evaluate_polynomial_at_4_l157_157926

-- Define the polynomial f
noncomputable def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Given x = 4, prove that f(4) = 1559
theorem evaluate_polynomial_at_4 : f 4 = 1559 :=
  by
    sorry

end evaluate_polynomial_at_4_l157_157926


namespace bicycle_final_price_l157_157768

theorem bicycle_final_price : 
  let original_price := 200 
  let weekend_discount := 0.40 * original_price 
  let price_after_weekend_discount := original_price - weekend_discount 
  let wednesday_discount := 0.20 * price_after_weekend_discount 
  let final_price := price_after_weekend_discount - wednesday_discount 
  final_price = 96 := 
by 
  sorry

end bicycle_final_price_l157_157768


namespace rabbit_jumps_before_dog_catches_l157_157626

/-- Prove that the number of additional jumps the rabbit can make before the dog catches up is 700,
    given the initial conditions:
      1. The rabbit has a 50-jump head start.
      2. The dog makes 5 jumps in the time the rabbit makes 6 jumps.
      3. The distance covered by 7 jumps of the dog equals the distance covered by 9 jumps of the rabbit. -/
theorem rabbit_jumps_before_dog_catches (h_head_start : ℕ) (h_time_ratio : ℚ) (h_distance_ratio : ℚ) : 
    h_head_start = 50 → h_time_ratio = 5/6 → h_distance_ratio = 7/9 → 
    ∃ (rabbit_additional_jumps : ℕ), rabbit_additional_jumps = 700 :=
by
  intro h_head_start_intro h_time_ratio_intro h_distance_ratio_intro
  have rabbit_additional_jumps := 700
  use rabbit_additional_jumps
  sorry

end rabbit_jumps_before_dog_catches_l157_157626


namespace find_remainder_l157_157685

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder_l157_157685


namespace total_spent_after_three_years_l157_157450

def iPhone_cost : ℝ := 1000
def contract_cost_per_month : ℝ := 200
def case_cost_before_discount : ℝ := 0.20 * iPhone_cost
def headphones_cost_before_discount : ℝ := 0.5 * case_cost_before_discount
def charger_cost : ℝ := 60
def warranty_cost_for_two_years : ℝ := 150
def discount_rate : ℝ := 0.10
def time_in_years : ℝ := 3

def contract_cost_for_three_years := contract_cost_per_month * 12 * time_in_years
def case_cost_after_discount := case_cost_before_discount * (1 - discount_rate)
def headphones_cost_after_discount := headphones_cost_before_discount * (1 - discount_rate)

def total_cost : ℝ :=
  iPhone_cost +
  contract_cost_for_three_years +
  case_cost_after_discount +
  headphones_cost_after_discount +
  charger_cost +
  warranty_cost_for_two_years

theorem total_spent_after_three_years : total_cost = 8680 :=
  by
    sorry

end total_spent_after_three_years_l157_157450


namespace avg_rate_change_l157_157172

def f (x : ℝ) : ℝ := x^2 + x

theorem avg_rate_change : (f 2 - f 1) / (2 - 1) = 4 := by
  -- here the proof steps should follow
  sorry

end avg_rate_change_l157_157172


namespace range_of_a_l157_157679

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
by sorry

end range_of_a_l157_157679


namespace distance_to_big_rock_l157_157040

variables (D : ℝ) (stillWaterSpeed : ℝ) (currentSpeed : ℝ) (totalTime : ℝ)

-- Define the conditions as constraints
def conditions := 
  stillWaterSpeed = 6 ∧
  currentSpeed = 1 ∧
  totalTime = 1 ∧
  (D / (stillWaterSpeed - currentSpeed) + D / (stillWaterSpeed + currentSpeed) = totalTime)

-- The theorem to prove the distance to Big Rock
theorem distance_to_big_rock (h : conditions D 6 1 1) : D = 35 / 12 :=
sorry

end distance_to_big_rock_l157_157040


namespace simplify_and_evaluate_l157_157893

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  ( (x + 3) / x - 1 ) / ( (x^2 - 1) / (x^2 + x) ) = Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l157_157893


namespace trapezoid_area_l157_157623

theorem trapezoid_area (x y : ℝ) (hx : y^2 + x^2 = 625) (hy : y^2 + (25 - x)^2 = 900) :
  1 / 2 * (11 + 36) * 24 = 564 :=
by
  sorry

end trapezoid_area_l157_157623


namespace album_cost_l157_157165

-- Definition of the cost variables
variable (B C A : ℝ)

-- Conditions given in the problem
axiom h1 : B = C + 4
axiom h2 : B = 18
axiom h3 : C = 0.70 * A

-- Theorem to prove the cost of the album
theorem album_cost : A = 20 := sorry

end album_cost_l157_157165


namespace average_speed_l157_157053

-- Definitions of conditions
def speed_first_hour : ℝ := 120
def speed_second_hour : ℝ := 60
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := 2

-- Theorem stating the equivalent proof problem
theorem average_speed : total_distance / total_time = 90 := by
  sorry

end average_speed_l157_157053


namespace negation_universal_prop_l157_157108

theorem negation_universal_prop:
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
  sorry

end negation_universal_prop_l157_157108


namespace set_operation_result_l157_157015

def M : Set ℕ := {2, 3}

def bin_op (A : Set ℕ) : Set ℕ :=
  {x | ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem set_operation_result : bin_op M = {4, 5, 6} :=
by
  sorry

end set_operation_result_l157_157015


namespace scientific_notation_1_3_billion_l157_157541

theorem scientific_notation_1_3_billion : 1300000000 = 1.3 * 10^9 := 
sorry

end scientific_notation_1_3_billion_l157_157541


namespace area_of_region_below_and_left_l157_157000

theorem area_of_region_below_and_left (x y : ℝ) :
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 4^2) ∧ y ≤ 0 ∧ y ≤ x - 4 →
  π * 4^2 / 4 = 4 * π :=
by sorry

end area_of_region_below_and_left_l157_157000


namespace cost_per_student_admission_l157_157680

-- Definitions based on the conditions.
def cost_to_rent_bus : ℕ := 100
def total_budget : ℕ := 350
def number_of_students : ℕ := 25

-- The theorem that we need to prove.
theorem cost_per_student_admission : (total_budget - cost_to_rent_bus) / number_of_students = 10 :=
by
  sorry

end cost_per_student_admission_l157_157680


namespace sandy_initial_cost_l157_157057

theorem sandy_initial_cost 
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (gain_percent : ℝ)
  (h1 : repairs_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ P : ℝ, P = 800 :=
by
  -- Proof steps would go here
  sorry

end sandy_initial_cost_l157_157057


namespace find_number_l157_157345

theorem find_number (x : ℝ) (h : 0.40 * x = 130 + 190) : x = 800 :=
by {
  -- The proof will go here
  sorry
}

end find_number_l157_157345


namespace sally_quarters_l157_157323

noncomputable def initial_quarters : ℕ := 760
noncomputable def spent_quarters : ℕ := 418
noncomputable def remaining_quarters : ℕ := 342

theorem sally_quarters : initial_quarters - spent_quarters = remaining_quarters :=
by sorry

end sally_quarters_l157_157323


namespace joan_initial_books_l157_157329

variable (books_sold : ℕ)
variable (books_left : ℕ)

theorem joan_initial_books (h1 : books_sold = 26) (h2 : books_left = 7) : books_sold + books_left = 33 := by
  sorry

end joan_initial_books_l157_157329


namespace compute_expression_l157_157089

theorem compute_expression (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 6) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end compute_expression_l157_157089


namespace find_k_l157_157132

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Vectors expressions
def k_a_add_b (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def a_sub_3b : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)

-- Condition of collinearity
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 = 0 ∨ v2.1 = 0 ∨ v1.1 * v2.2 = v1.2 * v2.1)

-- Statement to prove
theorem find_k :
  collinear (k_a_add_b (-1/3)) a_sub_3b :=
sorry

end find_k_l157_157132


namespace compound_interest_rate_l157_157781

theorem compound_interest_rate
  (A P : ℝ) (t n : ℝ)
  (HA : A = 1348.32)
  (HP : P = 1200)
  (Ht : t = 2)
  (Hn : n = 1) :
  ∃ r : ℝ, 0 ≤ r ∧ ((A / P) ^ (1 / (n * t)) - 1) = r ∧ r = 0.06 := 
sorry

end compound_interest_rate_l157_157781


namespace university_diploma_percentage_l157_157991

variables (population : ℝ)
          (U : ℝ) -- percentage of people with a university diploma
          (J : ℝ := 0.40) -- percentage of people with the job of their choice
          (S : ℝ := 0.10) -- percentage of people with a secondary school diploma pursuing further education

-- Condition 1: 18% of the people do not have a university diploma but have the job of their choice.
-- Condition 2: 25% of the people who do not have the job of their choice have a university diploma.
-- Condition 3: 10% of the people have a secondary school diploma and are pursuing further education.
-- Condition 4: 60% of the people with secondary school diploma have the job of their choice.
-- Condition 5: 30% of the people in further education have a job of their choice as well.
-- Condition 6: 40% of the people have the job of their choice.

axiom condition_1 : 0.18 * population = (0.18 * (1 - U)) * (population)
axiom condition_2 : 0.25 * (100 - J * 100) = 0.25 * (population - J * population)
axiom condition_3 : S * population = 0.10 * population
axiom condition_4 : 0.60 * S * population = (0.60 * S) * population
axiom condition_5 : 0.30 * S * population = (0.30 * S) * population
axiom condition_6 : J * population = 0.40 * population

theorem university_diploma_percentage : U * 100 = 37 :=
by sorry

end university_diploma_percentage_l157_157991


namespace parallel_vectors_y_value_l157_157070

theorem parallel_vectors_y_value 
  (y : ℝ) 
  (a : ℝ × ℝ := (6, 2)) 
  (b : ℝ × ℝ := (y, 3)) 
  (h : ∃ k : ℝ, b = k • a) : y = 9 :=
sorry

end parallel_vectors_y_value_l157_157070


namespace raisins_in_other_boxes_l157_157796

theorem raisins_in_other_boxes (total_raisins : ℕ) (raisins_box1 : ℕ) (raisins_box2 : ℕ) (other_boxes : ℕ) (num_other_boxes : ℕ) :
  total_raisins = 437 →
  raisins_box1 = 72 →
  raisins_box2 = 74 →
  num_other_boxes = 3 →
  other_boxes = (total_raisins - raisins_box1 - raisins_box2) / num_other_boxes →
  other_boxes = 97 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end raisins_in_other_boxes_l157_157796


namespace find_pairs_l157_157950

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ a b, (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3))
  ↔ (∃ a b, a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0) := by
  sorry

end find_pairs_l157_157950


namespace sin_75_mul_sin_15_eq_one_fourth_l157_157648

theorem sin_75_mul_sin_15_eq_one_fourth : 
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_75_mul_sin_15_eq_one_fourth_l157_157648


namespace smallest_result_l157_157511

theorem smallest_result :
  let a := (-2)^3
  let b := (-2) + 3
  let c := (-2) * 3
  let d := (-2) - 3
  a < b ∧ a < c ∧ a < d :=
by
  -- Lean proof steps would go here
  sorry

end smallest_result_l157_157511


namespace ellipse_major_axis_value_l157_157698

theorem ellipse_major_axis_value (m : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∀ {x y : ℝ}, (x, y) = P → (x^2 / m) + (y^2 / 16) = 1)
  (h2 : dist P F1 = 3)
  (h3 : dist P F2 = 7)
  : m = 25 :=
sorry

end ellipse_major_axis_value_l157_157698


namespace evaluate_expression_l157_157139

theorem evaluate_expression : (-2)^3 - (-3)^2 = -17 :=
by sorry

end evaluate_expression_l157_157139


namespace order_of_f_l157_157737

-- Define the function f
variables {f : ℝ → ℝ}

-- Definition of even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of monotonic increasing function on [0, +∞)
def monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x ≤ y) → f x ≤ f y

-- The main problem statement
theorem order_of_f (h_even : even_function f) (h_mono : monotonically_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
  sorry

end order_of_f_l157_157737


namespace reciprocal_real_roots_l157_157853

theorem reciprocal_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 * x2 = 1 ∧ x1 + x2 = 2 * (m + 2)) ∧ 
  (x1^2 - 2 * (m + 2) * x1 + (m^2 - 4) = 0) → m = Real.sqrt 5 := 
sorry

end reciprocal_real_roots_l157_157853


namespace tenth_day_is_monday_l157_157378

theorem tenth_day_is_monday (runs_20_mins : ∀ d ∈ [1, 7], d = 1 ∨ d = 6 ∨ d = 7 → True)
                            (total_minutes : 5 * 60 = 300)
                            (first_day_is_saturday : 1 = 6) :
   (10 % 7 = 3) :=
by
  sorry

end tenth_day_is_monday_l157_157378


namespace findMonthlyIncome_l157_157126

-- Variables and conditions
variable (I : ℝ) -- Raja's monthly income
variable (saving : ℝ) (r1 r2 r3 r4 r5 : ℝ) -- savings and monthly percentages

-- Conditions
def condition1 : r1 = 0.45 := by sorry
def condition2 : r2 = 0.12 := by sorry
def condition3 : r3 = 0.08 := by sorry
def condition4 : r4 = 0.15 := by sorry
def condition5 : r5 = 0.10 := by sorry
def conditionSaving : saving = 5000 := by sorry

-- Define the main equation
def mainEquation (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) : Prop :=
  (r1 * I) + (r2 * I) + (r3 * I) + (r4 * I) + (r5 * I) + saving = I

-- Main theorem to prove
theorem findMonthlyIncome (I : ℝ) (r1 r2 r3 r4 r5 saving : ℝ) 
  (h1 : r1 = 0.45) (h2 : r2 = 0.12) (h3 : r3 = 0.08) (h4 : r4 = 0.15) (h5 : r5 = 0.10) (hSaving : saving = 5000) :
  mainEquation I r1 r2 r3 r4 r5 saving → I = 50000 :=
  by sorry

end findMonthlyIncome_l157_157126


namespace value_of_g_l157_157875

-- Defining the function g and its property
def g (x : ℝ) : ℝ := 5

-- Theorem to prove g(x - 3) = 5 for any real number x
theorem value_of_g (x : ℝ) : g (x - 3) = 5 := by
  sorry

end value_of_g_l157_157875


namespace milburg_population_l157_157515

theorem milburg_population 
    (adults : ℕ := 5256) 
    (children : ℕ := 2987) 
    (teenagers : ℕ := 1709) 
    (seniors : ℕ := 2340) : 
    adults + children + teenagers + seniors = 12292 := 
by 
  sorry

end milburg_population_l157_157515


namespace solve_otimes_n_1_solve_otimes_2005_2_l157_157030

-- Define the operation ⊗
noncomputable def otimes (x y : ℕ) : ℕ :=
sorry -- the definition is abstracted away as per conditions

-- Conditions from the problem
axiom otimes_cond_1 : ∀ x : ℕ, otimes x 0 = x + 1
axiom otimes_cond_2 : ∀ x : ℕ, otimes 0 (x + 1) = otimes 1 x
axiom otimes_cond_3 : ∀ x y : ℕ, otimes (x + 1) (y + 1) = otimes (otimes x (y + 1)) y

-- Prove the required equalities
theorem solve_otimes_n_1 (n : ℕ) : otimes n 1 = n + 2 :=
sorry

theorem solve_otimes_2005_2 : otimes 2005 2 = 4013 :=
sorry

end solve_otimes_n_1_solve_otimes_2005_2_l157_157030


namespace time_taken_to_cross_platform_l157_157462

noncomputable def length_of_train : ℝ := 100 -- in meters
noncomputable def speed_of_train_km_hr : ℝ := 60 -- in km/hr
noncomputable def length_of_platform : ℝ := 150 -- in meters

noncomputable def speed_of_train_m_s := speed_of_train_km_hr * (1000 / 3600) -- converting km/hr to m/s
noncomputable def total_distance := length_of_train + length_of_platform
noncomputable def time_taken := total_distance / speed_of_train_m_s

theorem time_taken_to_cross_platform : abs (time_taken - 15) < 0.1 :=
by
  sorry

end time_taken_to_cross_platform_l157_157462


namespace min_value_2x_plus_y_l157_157313

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/(y + 1) = 2) : 2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l157_157313


namespace find_chord_line_eq_l157_157058

theorem find_chord_line_eq (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
    (hP : P = (1, 1)) (hC : C = (3, 0)) (hr : r = 3)
    (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + y^2 = r^2) :
    ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := by
  sorry

end find_chord_line_eq_l157_157058


namespace side_length_a_cosine_A_l157_157394

variable (A B C : Real)
variable (a b c : Real)
variable (triangle_inequality : a + b + c = 10)
variable (sine_equation : Real.sin B + Real.sin C = 4 * Real.sin A)
variable (bc_product : b * c = 16)

theorem side_length_a :
  a = 2 :=
  sorry

theorem cosine_A :
  b + c = 8 → 
  a = 2 → 
  b * c = 16 →
  Real.cos A = 7 / 8 :=
  sorry

end side_length_a_cosine_A_l157_157394


namespace sum_of_three_numbers_eq_zero_l157_157093

theorem sum_of_three_numbers_eq_zero (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : (a + b + c) / 3 = a + 20) (h3 : (a + b + c) / 3 = c - 10) (h4 : b = 10) : 
  a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l157_157093


namespace value_of_2m_plus_3n_l157_157452

theorem value_of_2m_plus_3n (m n : ℝ) (h : (m^2 + 4 * m + 5) * (n^2 - 2 * n + 6) = 5) : 2 * m + 3 * n = -1 :=
by
  sorry

end value_of_2m_plus_3n_l157_157452


namespace complex_division_l157_157482

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (7 - i) / (3 + i) = 2 - i := by
  sorry

end complex_division_l157_157482


namespace range_of_a_l157_157213

noncomputable def f (a x : ℝ) := a * Real.log x + x - 1

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l157_157213


namespace bowling_tournament_prize_orders_l157_157491
-- Import necessary Lean library

-- Define the conditions
def match_outcome (num_games : ℕ) : ℕ := 2 ^ num_games

-- Theorem statement
theorem bowling_tournament_prize_orders : match_outcome 5 = 32 := by
  -- This is the statement, proof is not required
  sorry

end bowling_tournament_prize_orders_l157_157491


namespace total_items_bought_l157_157049

def total_money : ℝ := 40
def sandwich_cost : ℝ := 5
def chip_cost : ℝ := 2
def soft_drink_cost : ℝ := 1.5

/-- Ike and Mike spend their total money on sandwiches, chips, and soft drinks.
  We want to prove that the total number of items bought (sandwiches, chips, and soft drinks)
  is equal to 8. -/
theorem total_items_bought :
  ∃ (s c d : ℝ), (sandwich_cost * s + chip_cost * c + soft_drink_cost * d ≤ total_money) ∧
  (∀x : ℝ, sandwich_cost * s ≤ total_money) ∧ ((s + c + d) = 8) :=
by {
  sorry
}

end total_items_bought_l157_157049


namespace find_number_l157_157156

theorem find_number 
  (m : ℤ)
  (h13 : m % 13 = 12)
  (h12 : m % 12 = 11)
  (h11 : m % 11 = 10)
  (h10 : m % 10 = 9)
  (h9 : m % 9 = 8)
  (h8 : m % 8 = 7)
  (h7 : m % 7 = 6)
  (h6 : m % 6 = 5)
  (h5 : m % 5 = 4)
  (h4 : m % 4 = 3)
  (h3 : m % 3 = 2) :
  m = 360359 :=
by
  sorry

end find_number_l157_157156


namespace right_triangle_hypotenuse_l157_157279

theorem right_triangle_hypotenuse (a h : ℝ) (r : ℝ) (h1 : r = 8) (h2 : h = a * Real.sqrt 2)
  (h3 : r = (a - h) / 2) : h = 16 * (Real.sqrt 2 + 1) := 
by
  sorry

end right_triangle_hypotenuse_l157_157279


namespace max_profit_l157_157286

-- Definition of the conditions
def production_requirements (tonAprodA tonAprodB tonBprodA tonBprodB: ℕ )
  := tonAprodA = 3 ∧ tonAprodB = 1 ∧ tonBprodA = 2 ∧ tonBprodB = 3

def profit_per_ton ( profitA profitB: ℕ )
  := profitA = 50000 ∧ profitB = 30000

def raw_material_limits ( rawA rawB: ℕ)
  := rawA = 13 ∧ rawB = 18

theorem max_profit 
  (production_requirements: production_requirements 3 1 2 3)
  (profit_per_ton: profit_per_ton 50000 30000)
  (raw_material_limits: raw_material_limits 13 18)
: ∃ (maxProfit: ℕ), maxProfit = 270000 := 
by 
  sorry

end max_profit_l157_157286


namespace calculate_product_l157_157917

theorem calculate_product (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3*x1*y1^2 = 2030)
  (h2 : y1^3 - 3*x1^2*y1 = 2029)
  (h3 : x2^3 - 3*x2*y2^2 = 2030)
  (h4 : y2^3 - 3*x2^2*y2 = 2029)
  (h5 : x3^3 - 3*x3*y3^2 = 2030)
  (h6 : y3^3 - 3*x3^2*y3 = 2029) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 / 1015 :=
sorry

end calculate_product_l157_157917


namespace jason_optimal_reroll_probability_l157_157639

-- Define the probability function based on the three dice roll problem
def probability_of_rerolling_two_dice : ℚ := 
  -- As per the problem, the computed and fixed probability is 7/64.
  7 / 64

-- Prove that Jason's optimal strategy leads to rerolling exactly two dice with a probability of 7/64.
theorem jason_optimal_reroll_probability : probability_of_rerolling_two_dice = 7 / 64 := 
  sorry

end jason_optimal_reroll_probability_l157_157639


namespace two_lines_intersections_with_ellipse_l157_157155

open Set

def ellipse (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem two_lines_intersections_with_ellipse {L1 L2 : ℝ → ℝ → Prop} :
  (∀ x y, L1 x y → ¬(ellipse x y)) →
  (∀ x y, L2 x y → ¬(ellipse x y)) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L1 x1 y1 ∧ L1 x2 y2) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L2 x1 y1 ∧ L2 x2 y2) →
  ∃ n, n = 2 ∨ n = 4 :=
by
  sorry

end two_lines_intersections_with_ellipse_l157_157155


namespace problem_quadratic_roots_l157_157843

theorem problem_quadratic_roots (m : ℝ) :
  (∀ x : ℝ, (m + 3) * x^2 - 4 * m * x + 2 * m - 1 = 0 →
    (∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ |x₁| > x₂)) ↔ -3 < m ∧ m < 0 :=
sorry

end problem_quadratic_roots_l157_157843


namespace simplify_polynomial_l157_157667

theorem simplify_polynomial (r : ℝ) :
  (2 * r ^ 3 + 5 * r ^ 2 - 4 * r + 8) - (r ^ 3 + 9 * r ^ 2 - 2 * r - 3)
  = r ^ 3 - 4 * r ^ 2 - 2 * r + 11 :=
by sorry

end simplify_polynomial_l157_157667


namespace part1_part2_l157_157138

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

theorem part1 (a : ℝ) (x : ℝ) (hx1 : 1 ≤ x) (hx2 : x ≤ Real.exp 1) :
  a = 1 →
  (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 + (Real.exp 1)^2 / 2) ∧ (∀ x, 1 ≤ x → x ≤ Real.exp 1 → f a x = 1 / 2) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 - 2 * a * x + Real.log x

theorem part2 (a : ℝ) :
  (-1/2 ≤ a ∧ a ≤ 1/2) ↔
  ∀ x, 1 < x → g a x < 0 :=
sorry

end part1_part2_l157_157138


namespace poly_not_33_l157_157445

theorem poly_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by sorry

end poly_not_33_l157_157445


namespace sebastian_older_than_jeremy_by_4_l157_157002

def J : ℕ := 40
def So : ℕ := 60 - 3
def sum_ages_in_3_years (S : ℕ) : Prop := (J + 3) + (S + 3) + (So + 3) = 150

theorem sebastian_older_than_jeremy_by_4 (S : ℕ) (h : sum_ages_in_3_years S) : S - J = 4 := by
  -- proof will be filled in
  sorry

end sebastian_older_than_jeremy_by_4_l157_157002


namespace olympiad_scores_l157_157332

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l157_157332


namespace jacob_has_more_money_l157_157472

def exchange_rate : ℝ := 1.11
def Mrs_Hilt_total_in_dollars : ℝ := 
  3 * 0.01 + 2 * 0.10 + 2 * 0.05 + 5 * 0.25 + 1 * 1.00

def Jacob_total_in_euros : ℝ := 
  4 * 0.01 + 1 * 0.05 + 1 * 0.10 + 3 * 0.20 + 2 * 0.50 + 2 * 1.00

def Jacob_total_in_dollars : ℝ := Jacob_total_in_euros * exchange_rate

def difference : ℝ := Jacob_total_in_dollars - Mrs_Hilt_total_in_dollars

theorem jacob_has_more_money : difference = 1.63 :=
by sorry

end jacob_has_more_money_l157_157472


namespace expression_value_l157_157036

theorem expression_value (x y z : ℕ) (hx : x = 2) (hy : y = 5) (hz : z = 3) :
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  rw [hx, hy, hz]
  sorry

end expression_value_l157_157036


namespace exists_k_composite_l157_157519

theorem exists_k_composite (h : Nat) : ∃ k : ℕ, ∀ n : ℕ, 0 < n → ∃ p : ℕ, Prime p ∧ p ∣ (k * 2 ^ n + 1) :=
by
  sorry

end exists_k_composite_l157_157519


namespace wood_pieces_gathered_l157_157598

theorem wood_pieces_gathered (sacks : ℕ) (pieces_per_sack : ℕ) (total_pieces : ℕ)
  (h1 : sacks = 4)
  (h2 : pieces_per_sack = 20)
  (h3 : total_pieces = sacks * pieces_per_sack) :
  total_pieces = 80 :=
by
  sorry

end wood_pieces_gathered_l157_157598


namespace sqrt_3x_eq_5x_largest_value_l157_157092

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l157_157092


namespace x_squared_y_cubed_eq_200_l157_157890

theorem x_squared_y_cubed_eq_200 (x y : ℕ) (h : 2^x * 9^y = 200) : x^2 * y^3 = 200 := by
  sorry

end x_squared_y_cubed_eq_200_l157_157890


namespace salt_amount_evaporation_l157_157677

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end salt_amount_evaporation_l157_157677


namespace triangle_inequality_l157_157624

theorem triangle_inequality {A B C : ℝ} {n : ℕ} (h : B = n * C) (hA : A + B + C = π) :
  B ≤ n * C :=
by
  sorry

end triangle_inequality_l157_157624


namespace quadratic_roots_prime_distinct_l157_157728

theorem quadratic_roots_prime_distinct (a α β m : ℕ) (h1: α ≠ β) (h2: Nat.Prime α) (h3: Nat.Prime β) (h4: α + β = m / a) (h5: α * β = 1996 / a) :
    a = 2 := by
  sorry

end quadratic_roots_prime_distinct_l157_157728


namespace probability_of_event_l157_157964

def is_uniform (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

theorem probability_of_event : 
  ∀ (a : ℝ), is_uniform a → ∀ (p : ℚ), (3 * a - 1 > 0) → p = 2 / 3 → 
  (∃ b, 0 ≤ b ∧ b ≤ 1 ∧ 3 * b - 1 > 0) := 
by
  intro a h_uniform p h_event h_prob
  sorry

end probability_of_event_l157_157964


namespace basketball_first_half_score_l157_157464

/-- 
In a college basketball match between Team Alpha and Team Beta, the game was tied at the end 
of the second quarter. The number of points scored by Team Alpha in each of the four quarters
formed an increasing geometric sequence, and the number of points scored by Team Beta in each
of the four quarters formed an increasing arithmetic sequence. At the end of the fourth quarter, 
Team Alpha had won by two points, with neither team scoring more than 100 points. 
Prove that the total number of points scored by the two teams in the first half is 24.
-/
theorem basketball_first_half_score 
  (a r : ℕ) (b d : ℕ)
  (h1 : a + a * r = b + (b + d))
  (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a + a * r + a * r^2 + a * r^3 ≤ 100)
  (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 100) : 
  a + a * r + b + (b + d) = 24 :=
  sorry

end basketball_first_half_score_l157_157464


namespace kiana_and_her_siblings_age_sum_l157_157404

theorem kiana_and_her_siblings_age_sum :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 256 ∧ a + b + c = 38 :=
by
sorry

end kiana_and_her_siblings_age_sum_l157_157404


namespace inverse_g_of_neg_92_l157_157143

noncomputable def g (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 1

theorem inverse_g_of_neg_92 : g (-3) = -92 :=
by 
-- This would be the proof but we are skipping it as requested
sorry

end inverse_g_of_neg_92_l157_157143


namespace ratio_equality_proof_l157_157137

theorem ratio_equality_proof
  (m n k a b c x y z : ℝ)
  (h : x / (m * (n * b + k * c - m * a)) = y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = k / (z * (a * x + b * y - c * z)) :=
by
  sorry

end ratio_equality_proof_l157_157137


namespace abs_diff_60th_terms_arithmetic_sequences_l157_157845

theorem abs_diff_60th_terms_arithmetic_sequences :
  let C : (ℕ → ℤ) := λ n => 25 + 15 * (n - 1)
  let D : (ℕ → ℤ) := λ n => 40 - 15 * (n - 1)
  |C 60 - D 60| = 1755 :=
by
  sorry

end abs_diff_60th_terms_arithmetic_sequences_l157_157845


namespace range_of_a_l157_157543

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 2)^x₁ > (a - 2)^x₂) → (2 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l157_157543


namespace find_larger_number_l157_157654

theorem find_larger_number :
  ∃ (L S : ℕ), L - S = 1365 ∧ L = 6 * S + 15 ∧ L = 1635 :=
sorry

end find_larger_number_l157_157654


namespace number_of_girls_l157_157233

theorem number_of_girls (classes : ℕ) (students_per_class : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : classes = 4) 
  (h2 : students_per_class = 25) 
  (h3 : boys = 56) 
  (h4 : girls = (classes * students_per_class) - boys) : 
  girls = 44 :=
by
  sorry

end number_of_girls_l157_157233


namespace condition_relationship_l157_157592

noncomputable def M : Set ℝ := {x | x > 2}
noncomputable def P : Set ℝ := {x | x < 3}

theorem condition_relationship :
  ∀ x, (x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧ ¬ (x ∈ (M ∪ P) → x ∈ (M ∩ P)) :=
by
  sorry

end condition_relationship_l157_157592


namespace zeros_of_f_l157_157375

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f : (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 2 = 0) :=
by 
  -- Placeholder for the proof
  sorry

end zeros_of_f_l157_157375


namespace horse_tile_system_l157_157411

theorem horse_tile_system (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + (1 / 3 : ℚ) * y = 100) : 
  ∃ (x y : ℕ), (x + y = 100) ∧ (3 * x + (1 / 3 : ℚ) * y = 100) :=
by sorry

end horse_tile_system_l157_157411


namespace cos_neg_60_equals_half_l157_157019

  theorem cos_neg_60_equals_half : Real.cos (-60 * Real.pi / 180) = 1 / 2 :=
  by
    sorry
  
end cos_neg_60_equals_half_l157_157019


namespace matilda_jellybeans_l157_157301

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l157_157301


namespace evaluate_F_2_f_3_l157_157710

def f (a : ℤ) : ℤ := a^2 - 1

def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 510 := by
  sorry

end evaluate_F_2_f_3_l157_157710


namespace simple_interest_years_l157_157887

noncomputable def simple_interest (P r t : ℕ) : ℕ :=
  P * r * t / 100

noncomputable def compound_interest (P r n : ℕ) : ℕ :=
  P * (1 + r / 100)^n - P

theorem simple_interest_years
  (P_si r_si P_ci r_ci n_ci si_half_ci si_si : ℕ)
  (h_si : simple_interest P_si r_si si_si = si_half_ci)
  (h_ci : compound_interest P_ci r_ci n_ci = si_half_ci * 2) :
  si_si = 2 :=
by
  sorry

end simple_interest_years_l157_157887


namespace lindsey_owns_more_cars_than_cathy_l157_157722

theorem lindsey_owns_more_cars_than_cathy :
  ∀ (cathy carol susan lindsey : ℕ),
    cathy = 5 →
    carol = 2 * cathy →
    susan = carol - 2 →
    cathy + carol + susan + lindsey = 32 →
    lindsey = cathy + 4 :=
by
  intros cathy carol susan lindsey h1 h2 h3 h4
  sorry

end lindsey_owns_more_cars_than_cathy_l157_157722


namespace even_factors_count_l157_157413

theorem even_factors_count (n : ℕ) (h : n = 2^4 * 3^2 * 5 * 7) : 
  ∃ k : ℕ, k = 48 ∧ ∃ a b c d : ℕ, 
  1 ≤ a ∧ a ≤ 4 ∧
  0 ≤ b ∧ b ≤ 2 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  0 ≤ d ∧ d ≤ 1 ∧
  k = (4 - 1 + 1) * (2 + 1) * (1 + 1) * (1 + 1) := by
  sorry

end even_factors_count_l157_157413


namespace sara_initial_savings_l157_157741

-- Given conditions as definitions
def save_rate_sara : ℕ := 10
def save_rate_jim : ℕ := 15
def weeks : ℕ := 820

-- Prove that the initial savings of Sara is 4100 dollars given the conditions
theorem sara_initial_savings : 
  ∃ S : ℕ, S + save_rate_sara * weeks = save_rate_jim * weeks → S = 4100 := 
sorry

end sara_initial_savings_l157_157741


namespace base6_subtraction_proof_l157_157940

-- Define the operations needed
def base6_add (a b : Nat) : Nat := sorry
def base6_subtract (a b : Nat) : Nat := sorry

axiom base6_add_correct : ∀ (a b : Nat), base6_add a b = (a + b)
axiom base6_subtract_correct : ∀ (a b : Nat), base6_subtract a b = (if a ≥ b then a - b else 0)

-- Define the problem conditions in base 6
def a := 5*6^2 + 5*6^1 + 5*6^0
def b := 5*6^1 + 5*6^0
def c := 2*6^2 + 0*6^1 + 2*6^0

-- Define the expected result
def result := 6*6^2 + 1*6^1 + 4*6^0

-- State the proof problem
theorem base6_subtraction_proof : base6_subtract (base6_add a b) c = result :=
by
  rw [base6_add_correct, base6_subtract_correct]
  sorry

end base6_subtraction_proof_l157_157940


namespace systematic_sampling_40th_number_l157_157749

theorem systematic_sampling_40th_number
  (total_students sample_size : ℕ)
  (first_group_start first_group_end selected_first_group_number steps : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_group_start = 1)
  (h4 : first_group_end = 20)
  (h5 : selected_first_group_number = 15)
  (h6 : steps = total_students / sample_size)
  (h7 : first_group_end - first_group_start + 1 = steps)
  : (selected_first_group_number + steps * (40 - 1)) = 795 :=
sorry

end systematic_sampling_40th_number_l157_157749


namespace value_of_a_minus_b_l157_157773

theorem value_of_a_minus_b (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : ab = 6) : a - b = 5 ∨ a - b = -5 := 
by
  sorry

end value_of_a_minus_b_l157_157773


namespace max_rock_value_l157_157065

def rock_value (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  14 * weight_5 + 11 * weight_4 + 2 * weight_1

def total_weight (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  5 * weight_5 + 4 * weight_4 + 1 * weight_1

theorem max_rock_value : ∃ (weight_5 weight_4 weight_1 : Nat), 
  total_weight weight_5 weight_4 weight_1 ≤ 18 ∧ 
  rock_value weight_5 weight_4 weight_1 = 50 :=
by
  -- We need to find suitable weight_5, weight_4, and weight_1.
  use 2, 2, 0 -- Example values
  apply And.intro
  -- Prove the total weight condition
  show total_weight 2 2 0 ≤ 18
  sorry
  -- Prove the value condition
  show rock_value 2 2 0 = 50
  sorry

end max_rock_value_l157_157065


namespace penumbra_ring_area_l157_157480

theorem penumbra_ring_area (r_umbra r_penumbra : ℝ) (h_ratio : r_umbra / r_penumbra = 2 / 6) (h_umbra : r_umbra = 40) :
  π * (r_penumbra ^ 2 - r_umbra ^ 2) = 12800 * π := by
  sorry

end penumbra_ring_area_l157_157480


namespace divisibility_problem_l157_157335

theorem divisibility_problem
  (h1 : 5^3 ∣ 1978^100 - 1)
  (h2 : 10^4 ∣ 3^500 - 1)
  (h3 : 2003 ∣ 2^286 - 1) :
  2^4 * 5^7 * 2003 ∣ (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) :=
by sorry

end divisibility_problem_l157_157335


namespace div_by_7_l157_157971

theorem div_by_7 (n : ℕ) : (3 ^ (12 * n + 1) + 2 ^ (6 * n + 2)) % 7 = 0 := by
  sorry

end div_by_7_l157_157971


namespace defective_chip_ratio_l157_157985

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end defective_chip_ratio_l157_157985


namespace polynomial_square_b_value_l157_157492

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l157_157492


namespace train_and_car_combined_time_l157_157862

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l157_157862


namespace novels_next_to_each_other_l157_157226

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem novels_next_to_each_other (n_essays n_novels : Nat) (condition_novels : n_novels = 2) (condition_essays : n_essays = 3) :
  let total_units := (n_novels - 1) + n_essays
  factorial total_units * factorial n_novels = 48 :=
by
  sorry

end novels_next_to_each_other_l157_157226


namespace triangle_cosine_theorem_l157_157872

def triangle_sums (a b c : ℝ) : ℝ := 
  b^2 + c^2 - a^2 + a^2 + c^2 - b^2 + a^2 + b^2 - c^2

theorem triangle_cosine_theorem (a b c : ℝ) (cos_A cos_B cos_C : ℝ) :
  a = 2 → b = 3 → c = 4 → 2 * b * c * cos_A + 2 * c * a * cos_B + 2 * a * b * cos_C = 29 :=
by
  intros h₁ h₂ h₃
  sorry

end triangle_cosine_theorem_l157_157872


namespace distributive_property_l157_157546

theorem distributive_property (a b : ℝ) : 3 * a * (2 * a - b) = 6 * a^2 - 3 * a * b :=
by
  sorry

end distributive_property_l157_157546


namespace sum_first_99_terms_l157_157151

def geom_sum (n : ℕ) : ℕ := (2^n) - 1

def seq_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum geom_sum

theorem sum_first_99_terms :
  seq_sum 99 = 2^100 - 101 := by
  sorry

end sum_first_99_terms_l157_157151


namespace ratio_length_width_l157_157557

theorem ratio_length_width (A L W : ℕ) (hA : A = 432) (hW : W = 12) (hArea : A = L * W) : L / W = 3 := 
by
  -- Placeholders for the actual mathematical proof
  sorry

end ratio_length_width_l157_157557


namespace option_a_option_b_option_c_option_d_l157_157352

open Real

theorem option_a (x : ℝ) (h1 : 0 < x) (h2 : x < π) : x > sin x :=
sorry

theorem option_b (x : ℝ) (h : 0 < x) : ¬ (1 - (1 / x) > log x) :=
sorry

theorem option_c (x : ℝ) : (x + 1) * exp x >= -1 / (exp 2) :=
sorry

theorem option_d : ¬ (∀ x : ℝ, x^2 > - (1 / x)) :=
sorry

end option_a_option_b_option_c_option_d_l157_157352


namespace weight_of_balls_l157_157849

theorem weight_of_balls (x y : ℕ) (h1 : 5 * x + 3 * y = 42) (h2 : 5 * y + 3 * x = 38) :
  x = 6 ∧ y = 4 :=
by
  sorry

end weight_of_balls_l157_157849


namespace solve_parallelogram_l157_157474

variables (x y : ℚ)

def condition1 : Prop := 6 * y - 2 = 12 * y - 10
def condition2 : Prop := 4 * x + 5 = 8 * x + 1

theorem solve_parallelogram : condition1 y → condition2 x → x + y = 7 / 3 :=
by
  intros h1 h2
  sorry

end solve_parallelogram_l157_157474


namespace abc_sum_l157_157007

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l157_157007


namespace relationship_between_areas_l157_157388

-- Assume necessary context and setup
variables (A B C C₁ C₂ : ℝ)
variables (a b c : ℝ) (h : a^2 + b^2 = c^2)

-- Define the conditions
def right_triangle := a = 8 ∧ b = 15 ∧ c = 17
def circumscribed_circle (d : ℝ) := d = 17
def areas_relation (A B C₁ C₂ : ℝ) := (C₁ < C₂) ∧ (A + B = C₁ + C₂)

-- Problem statement in Lean 4
theorem relationship_between_areas (ht : right_triangle 8 15 17) (hc : circumscribed_circle 17) :
  areas_relation A B C₁ C₂ :=
by sorry

end relationship_between_areas_l157_157388


namespace three_digit_number_l157_157294

theorem three_digit_number (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100 * c + 10 * b + a = 100 * a + 10 * b + c + 99) : (100 * a + 10 * b + c) = 253 := 
by
  sorry

end three_digit_number_l157_157294


namespace find_t_l157_157159

theorem find_t:
  (∃ t, (∀ (x y: ℝ), (x = 2 ∧ y = 8) ∨ (x = 4 ∧ y = 14) ∨ (x = 6 ∧ y = 20) → 
                (∀ (m b: ℝ), y = m * x + b) ∧ 
                (∀ (m b: ℝ), y = 3 * x + b ∧ b = 2 ∧ (t = 3 * 50 + 2) ∧ t = 152))) := by
  sorry

end find_t_l157_157159


namespace odd_and_even_inter_empty_l157_157422

-- Define the set of odd numbers
def odd_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define the set of even numbers
def even_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- The theorem stating that the intersection of odd numbers and even numbers is empty
theorem odd_and_even_inter_empty : odd_numbers ∩ even_numbers = ∅ :=
by
  -- placeholder for the proof
  sorry

end odd_and_even_inter_empty_l157_157422


namespace find_m_l157_157091

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -11.6 :=
sorry

end find_m_l157_157091


namespace radius_ratio_l157_157135

noncomputable def volume_large_sphere : ℝ := 432 * Real.pi

noncomputable def volume_small_sphere : ℝ := 0.08 * volume_large_sphere

noncomputable def radius_large_sphere : ℝ :=
  (3 * volume_large_sphere / (4 * Real.pi)) ^ (1 / 3)

noncomputable def radius_small_sphere : ℝ :=
  (3 * volume_small_sphere / (4 * Real.pi)) ^ (1 / 3)

theorem radius_ratio (V_L V_s : ℝ) (hL : V_L = 432 * Real.pi) (hS : V_s = 0.08 * V_L) :
  (radius_small_sphere / radius_large_sphere) = (2/5)^(1/3) :=
by
  sorry

end radius_ratio_l157_157135


namespace nth_equation_l157_157570

theorem nth_equation (n : ℕ) : 
  n ≥ 1 → (∃ k, k = n + 1 ∧ (k^2 - n^2 - 1) / 2 = n) :=
by
  intros h
  use n + 1
  sorry

end nth_equation_l157_157570


namespace min_value_A_mul_abs_x1_minus_x2_l157_157025

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

theorem min_value_A_mul_abs_x1_minus_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  2 * |x1 - x2| = (2 * Real.pi) / 2017 :=
sorry

end min_value_A_mul_abs_x1_minus_x2_l157_157025


namespace range_of_a_l157_157601

def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_a (a : ℝ) :
  exclusive_or (condition1 a) (condition2 a) → (1 ≤ a ∧ a < 2) ∨ a ≤ -2 :=
by
  -- Proof omitted
  sorry

end range_of_a_l157_157601


namespace total_bees_count_l157_157109

-- Definitions
def initial_bees : ℕ := 16
def additional_bees : ℕ := 7

-- Problem statement to prove
theorem total_bees_count : initial_bees + additional_bees = 23 := by
  -- The proof will be given here
  sorry

end total_bees_count_l157_157109


namespace proof_problem_l157_157718

-- Define the function f(x) = -x - x^3
def f (x : ℝ) : ℝ := -x - x^3

-- Define the main theorem according to the conditions and the required proofs.
theorem proof_problem (x1 x2 : ℝ) (h : x1 + x2 ≤ 0) :
  (f x1) * (f (-x1)) ≤ 0 ∧ (f x1 + f x2) ≥ (f (-x1) + f (-x2)) :=
by
  sorry

end proof_problem_l157_157718


namespace compare_exponents_l157_157645

theorem compare_exponents :
  let a := (3 / 2) ^ 0.1
  let b := (3 / 2) ^ 0.2
  let c := (3 / 2) ^ 0.08
  c < a ∧ a < b := by
  sorry

end compare_exponents_l157_157645


namespace range_of_a_l157_157181

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else x^2 + x -- Note: Using the specific definition matches the problem constraints clearly.

theorem range_of_a (a : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_ineq : f a + f (-a) < 4) : -1 < a ∧ a < 1 := 
by sorry

end range_of_a_l157_157181


namespace potato_sales_l157_157938

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end potato_sales_l157_157938


namespace find_physics_marks_l157_157221

theorem find_physics_marks (P C M : ℕ) (h1 : P + C + M = 210) (h2 : P + M = 180) (h3 : P + C = 140) : P = 110 :=
sorry

end find_physics_marks_l157_157221


namespace find_angle_A_l157_157182

theorem find_angle_A (a b c : ℝ) (h : a^2 - c^2 = b^2 - b * c) : 
  ∃ (A : ℝ), A = π / 3 :=
by
  sorry

end find_angle_A_l157_157182


namespace a_minus_b_value_l157_157415

theorem a_minus_b_value (a b c : ℝ) (x : ℝ) 
    (h1 : (2 * x - 3) ^ 2 = a * x ^ 2 + b * x + c)
    (h2 : x = 0 → c = 9)
    (h3 : x = 1 → a + b + c = 1)
    (h4 : x = -1 → (2 * (-1) - 3) ^ 2 = a * (-1) ^ 2 + b * (-1) + c) : 
    a - b = 16 :=
by  
  sorry

end a_minus_b_value_l157_157415


namespace travel_options_l157_157238

-- Define the conditions
def trains_from_A_to_B := 3
def ferries_from_B_to_C := 2

-- State the proof problem
theorem travel_options (t : ℕ) (f : ℕ) (h1 : t = trains_from_A_to_B) (h2 : f = ferries_from_B_to_C) : t * f = 6 :=
by
  rewrite [h1, h2]
  sorry

end travel_options_l157_157238


namespace solve_equation_l157_157731

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
sorry

end solve_equation_l157_157731


namespace constant_function_of_inequality_l157_157894

theorem constant_function_of_inequality
  (f : ℤ → ℝ)
  (h_bound : ∃ M : ℝ, ∀ n : ℤ, f n ≤ M)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n := by
  sorry

end constant_function_of_inequality_l157_157894


namespace rowed_upstream_distance_l157_157669

def distance_downstream := 120
def time_downstream := 2
def distance_upstream := 2
def speed_stream := 15

def speed_boat (V_b : ℝ) := V_b

theorem rowed_upstream_distance (V_b : ℝ) (D_u : ℝ) :
  (distance_downstream = (V_b + speed_stream) * time_downstream) ∧
  (D_u = (V_b - speed_stream) * time_upstream) →
  D_u = 60 :=
by 
  sorry

end rowed_upstream_distance_l157_157669


namespace third_quadrant_angle_to_fourth_l157_157085

theorem third_quadrant_angle_to_fourth {α : ℝ} (k : ℤ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  -90 - k * 360 < 180 - α ∧ 180 - α < -k * 360 :=
by
  sorry

end third_quadrant_angle_to_fourth_l157_157085


namespace circuit_analysis_l157_157059

/-
There are 3 conducting branches connected between points A and B.
First branch: a 2 Volt EMF and a 2 Ohm resistor connected in series.
Second branch: a 2 Volt EMF and a 1 Ohm resistor.
Third branch: a conductor with a resistance of 1 Ohm.
Prove the currents and voltage drop are as follows:
- Current in first branch: i1 = 0.4 A
- Current in second branch: i2 = 0.8 A
- Current in third branch: i3 = 1.2 A
- Voltage between A and B: E_AB = 1.2 Volts
-/
theorem circuit_analysis :
  ∃ (i1 i2 i3 : ℝ) (E_AB : ℝ),
    (i1 = 0.4) ∧
    (i2 = 0.8) ∧
    (i3 = 1.2) ∧
    (E_AB = 1.2) ∧
    (2 = 2 * i1 + i3) ∧
    (2 = i2 + i3) ∧
    (i3 = i1 + i2) ∧
    (E_AB = i3 * 1) := sorry

end circuit_analysis_l157_157059


namespace carla_highest_final_number_l157_157319

def alice_final_number (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 - 3
  let step3 := step2 / 3
  step3 + 4

def bob_final_number (initial : ℕ) : ℕ :=
  let step1 := initial + 5
  let step2 := step1 * 2
  let step3 := step2 - 4
  step3 / 2

def carla_final_number (initial : ℕ) : ℕ :=
  let step1 := initial - 2
  let step2 := step1 * 2
  let step3 := step2 + 3
  step3 * 2

theorem carla_highest_final_number : carla_final_number 12 > bob_final_number 12 ∧ carla_final_number 12 > alice_final_number 12 :=
  by
  have h_alice : alice_final_number 12 = 11 := by rfl
  have h_bob : bob_final_number 12 = 15 := by rfl
  have h_carla : carla_final_number 12 = 46 := by rfl
  sorry

end carla_highest_final_number_l157_157319


namespace geometric_mean_of_1_and_9_is_pm3_l157_157341

theorem geometric_mean_of_1_and_9_is_pm3 (a b c : ℝ) (h₀ : a = 1) (h₁ : b = 9) (h₂ : c^2 = a * b) : c = 3 ∨ c = -3 := by
  sorry

end geometric_mean_of_1_and_9_is_pm3_l157_157341


namespace rabbit_total_distance_l157_157837

theorem rabbit_total_distance 
  (r₁ r₂ : ℝ) 
  (h1 : r₁ = 7) 
  (h2 : r₂ = 15) 
  (q : ∀ (x : ℕ), x = 4) 
  : (3.5 * π + 8 + 7.5 * π + 8 + 3.5 * π + 8) = 14.5 * π + 24 := 
by
  sorry

end rabbit_total_distance_l157_157837


namespace range_of_expression_l157_157361

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  - π / 6 < 2 * α - β / 2 ∧ 2 * α - β / 2 < π :=
sorry

end range_of_expression_l157_157361


namespace mathematician_daily_questions_l157_157442

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l157_157442


namespace circle_intersection_l157_157765

theorem circle_intersection (a : ℝ) :
  ((-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2)) ↔
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 1) :=
sorry

end circle_intersection_l157_157765


namespace symmetry_origin_points_l157_157271

theorem symmetry_origin_points (x y : ℝ) (h₁ : (x, -2) = (-3, -y)) : x + y = -1 :=
sorry

end symmetry_origin_points_l157_157271


namespace range_of_sum_of_products_l157_157128

theorem range_of_sum_of_products (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)
  (h_sum : a + b + c = (Real.sqrt 3) / 2) :
  0 < (a * b + b * c + c * a) ∧ (a * b + b * c + c * a) ≤ 1 / 4 :=
by
  sorry

end range_of_sum_of_products_l157_157128


namespace sumata_miles_per_day_l157_157033

theorem sumata_miles_per_day (total_miles : ℝ) (total_days : ℝ) (h1 : total_miles = 250.0) (h2 : total_days = 5.0) :
  total_miles / total_days = 50.0 :=
by
  sorry

end sumata_miles_per_day_l157_157033


namespace boat_speed_in_still_water_l157_157366

namespace BoatSpeed

variables (V_b V_s : ℝ)

def condition1 : Prop := V_b + V_s = 15
def condition2 : Prop := V_b - V_s = 5

theorem boat_speed_in_still_water (h1 : condition1 V_b V_s) (h2 : condition2 V_b V_s) : V_b = 10 :=
by
  sorry

end BoatSpeed

end boat_speed_in_still_water_l157_157366


namespace simple_interest_rate_l157_157136

theorem simple_interest_rate (P T SI : ℝ) (hP : P = 10000) (hT : T = 1) (hSI : SI = 400) :
    (SI = P * 0.04 * T) := by
  rw [hP, hT, hSI]
  sorry

end simple_interest_rate_l157_157136


namespace mike_last_5_shots_l157_157039

theorem mike_last_5_shots :
  let initial_shots := 30
  let initial_percentage := 40 / 100
  let additional_shots_1 := 10
  let new_percentage_1 := 45 / 100
  let additional_shots_2 := 5
  let new_percentage_2 := 46 / 100
  
  let initial_makes := initial_shots * initial_percentage
  let total_shots_after_1 := initial_shots + additional_shots_1
  let makes_after_1 := total_shots_after_1 * new_percentage_1 - initial_makes
  let total_makes_after_1 := initial_makes + makes_after_1
  let total_shots_after_2 := total_shots_after_1 + additional_shots_2
  let final_makes := total_shots_after_2 * new_percentage_2
  let makes_in_last_5 := final_makes - total_makes_after_1
  
  makes_in_last_5 = 2
:=
by
  sorry

end mike_last_5_shots_l157_157039


namespace solve_for_x_l157_157231

def condition (x : ℝ) : Prop := (x - 5)^3 = (1 / 27)⁻¹

theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 8 := by
  use 8
  unfold condition
  sorry

end solve_for_x_l157_157231


namespace minimum_single_discount_l157_157611

theorem minimum_single_discount (n : ℕ) :
  (∀ x : ℝ, 0 < x → 
    ((1 - n / 100) * x < (1 - 0.18) * (1 - 0.18) * x) ∧
    ((1 - n / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x) ∧
    ((1 - n / 100) * x < (1 - 0.28) * (1 - 0.07) * x))
  ↔ n = 34 :=
by
  sorry

end minimum_single_discount_l157_157611


namespace ratio_equality_l157_157364

theorem ratio_equality (x y u v p q : ℝ) (h : (x / y) * (u / v) * (p / q) = 1) :
  (x / y) * (u / v) * (p / q) = 1 := 
by sorry

end ratio_equality_l157_157364


namespace cake_pieces_per_sister_l157_157961

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end cake_pieces_per_sister_l157_157961


namespace parabola_vertex_on_x_axis_l157_157805

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + c = 0) → c = 1 := by
  sorry

end parabola_vertex_on_x_axis_l157_157805


namespace largest_divisor_of_n_cube_minus_n_minus_six_l157_157927

theorem largest_divisor_of_n_cube_minus_n_minus_six (n : ℤ) : 6 ∣ (n^3 - n - 6) :=
by sorry

end largest_divisor_of_n_cube_minus_n_minus_six_l157_157927


namespace initial_pocket_money_l157_157936

variable (P : ℝ)

-- Conditions
axiom chocolates_expenditure : P * (1/9) ≥ 0
axiom fruits_expenditure : P * (2/5) ≥ 0
axiom remaining_money : P * (22/45) = 220

-- Theorem statement
theorem initial_pocket_money : P = 450 :=
by
  have h₁ : P * (1/9) + P * (2/5) = P * (23/45) := by sorry
  have h₂ : P * (1 - 23/45) = P * (22/45) := by sorry
  have h₃ : P = 220 / (22/45) := by sorry
  have h₄ : P = 220 * (45/22) := by sorry
  have h₅ : P = 450 := by sorry
  exact h₅

end initial_pocket_money_l157_157936


namespace number_of_cars_l157_157580

theorem number_of_cars (total_wheels cars_bikes trash_can tricycle roller_skates : ℕ) 
  (h1 : cars_bikes = 2) 
  (h2 : trash_can = 2) 
  (h3 : tricycle = 3) 
  (h4 : roller_skates = 4) 
  (h5 : total_wheels = 25) 
  : (total_wheels - (cars_bikes * 2 + trash_can * 2 + tricycle * 3 + roller_skates * 4)) / 4 = 3 :=
by
  sorry

end number_of_cars_l157_157580


namespace cos_min_sin_eq_neg_sqrt_seven_half_l157_157761

variable (θ : ℝ)

theorem cos_min_sin_eq_neg_sqrt_seven_half (h1 : Real.sin θ + Real.cos θ = 0.5)
    (h2 : π / 2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = - Real.sqrt 7 / 2 := by
  sorry

end cos_min_sin_eq_neg_sqrt_seven_half_l157_157761


namespace difference_in_ages_l157_157483

variables (J B : ℕ)

-- The conditions: Jack's age is twice Bill's age, and in eight years, Jack will be three times Bill's age then.
axiom condition1 : J = 2 * B
axiom condition2 : J + 8 = 3 * (B + 8)

-- The theorem statement we are proving: The difference in their current ages is 16.
theorem difference_in_ages : J - B = 16 :=
by
  sorry

end difference_in_ages_l157_157483


namespace proof_area_of_squares_l157_157879

noncomputable def area_of_squares : Prop :=
  let side_C := 48
  let side_D := 60
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  (area_C / area_D = (16 / 25)) ∧ 
  ((area_D - area_C) / area_C = (36 / 100))

theorem proof_area_of_squares : area_of_squares := sorry

end proof_area_of_squares_l157_157879


namespace f_monotonicity_l157_157920

noncomputable def f (x : ℝ) : ℝ := abs (x^2 - 1)

theorem f_monotonicity :
  (∀ x y : ℝ, (-1 < x ∧ x < 0 ∧ x < y ∧ y < 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < -1 ∧ y < -1 ∧ y < x) → f x < f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ y < x) → f x < f y) :=
by
  sorry

end f_monotonicity_l157_157920


namespace correct_operation_l157_157042

theorem correct_operation (a b : ℝ) : 
  (2 * a) * (3 * a) = 6 * a^2 :=
by
  -- The proof would be here; using "sorry" to skip the actual proof steps.
  sorry

end correct_operation_l157_157042


namespace bodies_distance_apart_l157_157908

def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

theorem bodies_distance_apart (t : ℝ) (h₁ : 220.5 = distance_fallen t - distance_fallen (t - 5)) : t = 7 :=
by {
  sorry
}

end bodies_distance_apart_l157_157908


namespace number_of_lucky_tickets_l157_157770

def is_leningrad_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ + a₂ + a₃ = a₄ + a₅ + a₆

def is_moscow_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₂ + a₄ + a₆ = a₁ + a₃ + a₅

def is_symmetric (a₂ a₅ : ℕ) : Prop :=
  a₂ = a₅

def is_valid_ticket (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  is_leningrad_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_moscow_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_symmetric a₂ a₅

theorem number_of_lucky_tickets : 
  ∃ n : ℕ, n = 6700 ∧ 
  (∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℕ, 
    0 ≤ a₁ ∧ a₁ ≤ 9 ∧
    0 ≤ a₂ ∧ a₂ ≤ 9 ∧
    0 ≤ a₃ ∧ a₃ ≤ 9 ∧
    0 ≤ a₄ ∧ a₄ ≤ 9 ∧
    0 ≤ a₅ ∧ a₅ ≤ 9 ∧
    0 ≤ a₆ ∧ a₆ ≤ 9 →
    is_valid_ticket a₁ a₂ a₃ a₄ a₅ a₆ →
    n = 6700) := sorry

end number_of_lucky_tickets_l157_157770


namespace side_length_of_square_l157_157328

theorem side_length_of_square (r : ℝ) (A : ℝ) (s : ℝ) 
  (h1 : π * r^2 = 36 * π) 
  (h2 : s = 2 * r) : 
  s = 12 :=
by 
  sorry

end side_length_of_square_l157_157328


namespace friends_meeting_both_movie_and_games_l157_157583

theorem friends_meeting_both_movie_and_games 
  (T M P G M_and_P P_and_G M_and_P_and_G : ℕ) 
  (hT : T = 31) 
  (hM : M = 10) 
  (hP : P = 20) 
  (hG : G = 5) 
  (hM_and_P : M_and_P = 4) 
  (hP_and_G : P_and_G = 0) 
  (hM_and_P_and_G : M_and_P_and_G = 2) : (M + P + G - M_and_P - T + M_and_P_and_G - 2) = 2 := 
by 
  sorry

end friends_meeting_both_movie_and_games_l157_157583


namespace find_least_number_subtracted_l157_157903

theorem find_least_number_subtracted (n m : ℕ) (h : n = 78721) (h1 : m = 23) : (n % m) = 15 := by
  sorry

end find_least_number_subtracted_l157_157903


namespace kelly_chris_boxes_ratio_l157_157129

theorem kelly_chris_boxes_ratio (X : ℝ) (h : X > 0) :
  (0.4 * X) / (0.6 * X) = 2 / 3 :=
by sorry

end kelly_chris_boxes_ratio_l157_157129


namespace solution_interval_l157_157612

theorem solution_interval (x : ℝ) (h1 : x / 2 ≤ 5 - x) (h2 : 5 - x < -3 * (2 + x)) :
  x < -11 / 2 := 
sorry

end solution_interval_l157_157612


namespace triangle_ratio_l157_157682

-- Define the conditions and the main theorem statement
theorem triangle_ratio (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h_eq : b * Real.cos C + c * Real.cos B = 2 * b) 
  (h_law_sines_a : a = 2 * b * Real.sin B / Real.sin A) 
  (h_angles : A + B + C = Real.pi) :
  b / a = 1 / 2 :=
by 
  sorry

end triangle_ratio_l157_157682


namespace exists_irreducible_fractions_l157_157082

theorem exists_irreducible_fractions:
  ∃ (f : Fin 2018 → ℚ), 
    (∀ i j : Fin 2018, i ≠ j → (f i).den ≠ (f j).den) ∧ 
    (∀ i j : Fin 2018, i ≠ j → ∀ d : ℚ, d = f i - f j → d ≠ 0 → d.den < (f i).den ∧ d.den < (f j).den) :=
by
  -- proof is omitted
  sorry

end exists_irreducible_fractions_l157_157082


namespace average_weasels_caught_per_week_l157_157310

-- Definitions based on the conditions
def initial_weasels : ℕ := 100
def initial_rabbits : ℕ := 50
def foxes : ℕ := 3
def rabbits_caught_per_week_per_fox : ℕ := 2
def weeks : ℕ := 3
def remaining_animals : ℕ := 96

-- Main theorem statement
theorem average_weasels_caught_per_week :
  (foxes * weeks * rabbits_caught_per_week_per_fox +
   foxes * weeks * W = initial_weasels + initial_rabbits - remaining_animals) →
  W = 4 :=
sorry

end average_weasels_caught_per_week_l157_157310


namespace parametric_equation_of_line_passing_through_M_l157_157097

theorem parametric_equation_of_line_passing_through_M (
  t : ℝ
) : 
    ∃ x y : ℝ, 
      x = 1 + (t * (Real.cos (Real.pi / 3))) ∧ 
      y = 5 + (t * (Real.sin (Real.pi / 3))) ∧ 
      x = 1 + (1/2) * t ∧ 
      y = 5 + (Real.sqrt 3 / 2) * t := 
by
  sorry

end parametric_equation_of_line_passing_through_M_l157_157097


namespace book_width_l157_157503

noncomputable def golden_ratio : Real := (1 + Real.sqrt 5) / 2

theorem book_width (length : Real) (width : Real) 
(h1 : length = 20) 
(h2 : width / length = golden_ratio) : 
width = 12.36 := 
by 
  sorry

end book_width_l157_157503


namespace number_of_fish_given_to_dog_l157_157788

-- Define the conditions
def condition1 (D C : ℕ) : Prop := C = D / 2
def condition2 (D C : ℕ) : Prop := D + C = 60

-- Theorem to prove the number of fish given to the dog
theorem number_of_fish_given_to_dog (D : ℕ) (C : ℕ) (h1 : condition1 D C) (h2 : condition2 D C) : D = 40 :=
by
  sorry

end number_of_fish_given_to_dog_l157_157788


namespace math_problem_l157_157408

variable (x : ℕ)
variable (h : x + 7 = 27)

theorem math_problem : (x = 20) ∧ (((x / 5) + 5) * 7 = 63) :=
by
  have h1 : x = 20 := by {
    -- x can be solved here using the condition, but we use sorry to skip computation.
    sorry
  }
  have h2 : (((x / 5) + 5) * 7 = 63) := by {
    -- The second part result can be computed using the derived x value, but we use sorry to skip computation.
    sorry
  }
  exact ⟨h1, h2⟩

end math_problem_l157_157408


namespace initial_men_in_camp_l157_157764

theorem initial_men_in_camp (days_initial men_initial : ℕ) (days_plus_thirty men_plus_thirty : ℕ)
(h1 : days_initial = 20)
(h2 : men_plus_thirty = men_initial + 30)
(h3 : days_plus_thirty = 5)
(h4 : (men_initial * days_initial) = (men_plus_thirty * days_plus_thirty)) :
  men_initial = 10 :=
by sorry

end initial_men_in_camp_l157_157764


namespace value_range_l157_157283

-- Step to ensure proofs about sine and real numbers are within scope
open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sin x * cos x - 1

theorem value_range (x : ℝ) : -2 ≤ y x ∧ y x ≤ 0 :=
by sorry

end value_range_l157_157283


namespace abs_inequality_solution_l157_157606

theorem abs_inequality_solution :
  { x : ℝ | |x - 2| + |x + 3| < 6 } = { x | -7 / 2 < x ∧ x < 5 / 2 } :=
by
  sorry

end abs_inequality_solution_l157_157606


namespace range_of_c_l157_157207

variable (c : ℝ)

def p : Prop := ∀ x : ℝ, x > 0 → c^x = c^(x+1) / c
def q : Prop := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → x + 1/x > 1/c

theorem range_of_c (h1 : c > 0) (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l157_157207


namespace buses_required_is_12_l157_157507

-- Define the conditions given in the problem
def students : ℕ := 535
def bus_capacity : ℕ := 45

-- Define the minimum number of buses required
def buses_needed (students : ℕ) (bus_capacity : ℕ) : ℕ :=
  (students + bus_capacity - 1) / bus_capacity

-- The theorem stating the number of buses required is 12
theorem buses_required_is_12 :
  buses_needed students bus_capacity = 12 :=
sorry

end buses_required_is_12_l157_157507


namespace inequality_l157_157417

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 3) :
  1 / (4 - a^2) + 1 / (4 - b^2) + 1 / (4 - c^2) ≤ 9 / (a + b + c)^2 :=
by
  sorry

end inequality_l157_157417


namespace max_f_geq_fraction_3_sqrt3_over_2_l157_157035

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq_fraction_3_sqrt3_over_2 : ∃ x : ℝ, f x ≥ (3 + Real.sqrt 3) / 2 := 
sorry

end max_f_geq_fraction_3_sqrt3_over_2_l157_157035


namespace possible_landing_l157_157658

-- There are 1985 airfields
def num_airfields : ℕ := 1985

-- 50 airfields where planes could potentially land
def num_land_airfields : ℕ := 50

-- Define the structure of the problem
structure AirfieldSetup :=
  (airfields : Fin num_airfields → Fin num_land_airfields)

-- There exists a configuration such that the conditions are met
theorem possible_landing : ∃ (setup : AirfieldSetup), 
  (∀ i : Fin num_airfields, -- For each airfield
    ∃ j : Fin num_land_airfields, -- There exists a landing airfield
    setup.airfields i = j) -- The plane lands at this airfield.
:=
sorry

end possible_landing_l157_157658


namespace length_of_faster_train_is_370_l157_157471

noncomputable def length_of_faster_train (vf vs : ℕ) (t : ℕ) : ℕ :=
  let rel_speed := vf - vs
  let rel_speed_m_per_s := rel_speed * 1000 / 3600
  rel_speed_m_per_s * t

theorem length_of_faster_train_is_370 :
  length_of_faster_train 72 36 37 = 370 := 
  sorry

end length_of_faster_train_is_370_l157_157471


namespace base_of_triangle_is_24_l157_157141

def triangle_sides_sum := 50
def left_side : ℕ := 12
def right_side := left_side + 2
def base := triangle_sides_sum - left_side - right_side

theorem base_of_triangle_is_24 :
  base = 24 :=
by 
  have h : left_side = 12 := rfl
  have h2 : right_side = 14 := by simp [right_side, h]
  have h3 : base = 24 := by simp [base, triangle_sides_sum, h, h2]
  exact h3

end base_of_triangle_is_24_l157_157141


namespace find_worst_competitor_l157_157945

structure Competitor :=
  (name : String)
  (gender : String)
  (generation : String)

-- Define the competitors
def man : Competitor := ⟨"man", "male", "generation1"⟩
def wife : Competitor := ⟨"wife", "female", "generation1"⟩
def son : Competitor := ⟨"son", "male", "generation2"⟩
def sister : Competitor := ⟨"sister", "female", "generation1"⟩

-- Conditions
def opposite_genders (c1 c2 : Competitor) : Prop :=
  c1.gender ≠ c2.gender

def different_generations (c1 c2 : Competitor) : Prop :=
  c1.generation ≠ c2.generation

noncomputable def worst_competitor : Competitor :=
  sister

def is_sibling (c1 c2 : Competitor) : Prop :=
  (c1 = man ∧ c2 = sister) ∨ (c1 = sister ∧ c2 = man)

-- Theorem statement
theorem find_worst_competitor (best_competitor : Competitor) :
  (opposite_genders worst_competitor best_competitor) ∧
  (different_generations worst_competitor best_competitor) ∧
  ∃ (sibling : Competitor), (is_sibling worst_competitor sibling) :=
  sorry

end find_worst_competitor_l157_157945


namespace find_f_at_2_l157_157459

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h1 : ∀ x, f x = x^3 + 3 * x * f'' 2)
variable (h2 : f' 2 = 12 + 3 * f' 2)

theorem find_f_at_2 : f 2 = -28 :=
by
  sorry

end find_f_at_2_l157_157459


namespace even_function_must_be_two_l157_157162

def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-2)*x + (m^2 - 7*m + 12)

theorem even_function_must_be_two (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) ↔ m = 2 :=
by
  sorry

end even_function_must_be_two_l157_157162


namespace digit_in_2017th_place_l157_157969

def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_in_2017th_place :
  digit_at_position 2017 = 7 :=
by sorry

end digit_in_2017th_place_l157_157969


namespace sum_of_products_leq_one_third_l157_157187

theorem sum_of_products_leq_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 :=
sorry

end sum_of_products_leq_one_third_l157_157187


namespace floor_exponents_eq_l157_157342

theorem floor_exponents_eq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_inf_k : ∃ᶠ k in at_top, ∃ (k : ℕ), ⌊a ^ k⌋ + ⌊b ^ k⌋ = ⌊a⌋ ^ k + ⌊b⌋ ^ k) :
  ⌊a ^ 2014⌋ + ⌊b ^ 2014⌋ = ⌊a⌋ ^ 2014 + ⌊b⌋ ^ 2014 := by
  sorry

end floor_exponents_eq_l157_157342


namespace work_increase_percentage_l157_157825

theorem work_increase_percentage (p w : ℕ) (hp : p > 0) : 
  (((4 / 3 : ℚ) * w) - w) / w * 100 = 33.33 := 
sorry

end work_increase_percentage_l157_157825


namespace smallest_integer_n_satisfying_inequality_l157_157485

theorem smallest_integer_n_satisfying_inequality :
  ∃ n : ℤ, n^2 - 13 * n + 36 ≤ 0 ∧ (∀ m : ℤ, m^2 - 13 * m + 36 ≤ 0 → m ≥ n) ∧ n = 4 := 
by
  sorry

end smallest_integer_n_satisfying_inequality_l157_157485


namespace union_eq_C_l157_157695

def A: Set ℝ := { x | x > 2 }
def B: Set ℝ := { x | x < 0 }
def C: Set ℝ := { x | x * (x - 2) > 0 }

theorem union_eq_C : (A ∪ B) = C :=
by
  sorry

end union_eq_C_l157_157695


namespace product_of_functions_l157_157028

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem product_of_functions (x : ℝ) (hx : x ≠ -3) : f x * g x = x - 3 := by
  -- proof goes here
  sorry

end product_of_functions_l157_157028


namespace order_of_6_with_respect_to_f_is_undefined_l157_157724

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end order_of_6_with_respect_to_f_is_undefined_l157_157724


namespace solve_equation_l157_157185

theorem solve_equation : ∃ x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := 
by
  -- Introducing x as a real number and stating the goal
  use 4
  -- Show that 2 * 4 - 3 = 5
  simp
  -- Adding the sorry to skip the proof step
  sorry

end solve_equation_l157_157185


namespace steven_owes_jeremy_l157_157775

-- Definitions for the conditions
def base_payment_per_room := (13 : ℚ) / 3
def rooms_cleaned := (5 : ℚ) / 2
def additional_payment_per_room := (1 : ℚ) / 2

-- Define the total amount of money Steven owes Jeremy
def total_payment (base_payment_per_room rooms_cleaned additional_payment_per_room : ℚ) : ℚ :=
  let base_payment := base_payment_per_room * rooms_cleaned
  let additional_payment := if rooms_cleaned > 2 then additional_payment_per_room * rooms_cleaned else 0
  base_payment + additional_payment

-- The statement to prove
theorem steven_owes_jeremy :
  total_payment base_payment_per_room rooms_cleaned additional_payment_per_room = 145 / 12 :=
by
  sorry

end steven_owes_jeremy_l157_157775


namespace range_of_expression_l157_157954

noncomputable def expression (a b c d : ℝ) : ℝ :=
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + 
  Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)

theorem range_of_expression (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2)
  (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2)
  (h7 : 0 ≤ d) (h8 : d ≤ 2) :
  4 * Real.sqrt 2 ≤ expression a b c d ∧ expression a b c d ≤ 16 :=
by
  sorry

end range_of_expression_l157_157954


namespace domain_of_func_l157_157855

noncomputable def func (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_func :
  ∀ x : ℝ, x ≠ 1 / 2 ↔ ∃ y : ℝ, y = func x := sorry

end domain_of_func_l157_157855


namespace number_of_belts_l157_157428

def ties := 34
def black_shirts := 63
def white_shirts := 42

def jeans := (2 / 3 : ℚ) * (black_shirts + white_shirts)
def scarves (B : ℚ) := (1 / 2 : ℚ) * (ties + B)

theorem number_of_belts (B : ℚ) : jeans = scarves B + 33 → B = 40 := by
  -- This theorem states the required proof but leaves the proof itself as a placeholder.
  -- The proof would involve solving equations algebraically as shown in the solution steps.
  sorry

end number_of_belts_l157_157428


namespace max_min_sum_eq_two_l157_157436

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 + Real.sqrt 2 * Real.sin (x + Real.pi / 4)) / (2 * x ^ 2 + Real.cos x)

theorem max_min_sum_eq_two (a b : ℝ) (h_max : ∀ x, f x ≤ a) (h_min : ∀ x, b ≤ f x) (h_max_val : ∃ x, f x = a) (h_min_val : ∃ x, f x = b) :
  a + b = 2 := 
sorry

end max_min_sum_eq_two_l157_157436


namespace max_A_l157_157435

noncomputable def A (x y : ℝ) : ℝ :=
  x^4 * y + x * y^4 + x^3 * y + x * y^3 + x^2 * y + x * y^2

theorem max_A (x y : ℝ) (h : x + y = 1) : A x y ≤ 7 / 16 :=
sorry

end max_A_l157_157435


namespace total_waiting_days_l157_157374

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end total_waiting_days_l157_157374


namespace tips_fraction_of_salary_l157_157846

theorem tips_fraction_of_salary (S T x : ℝ) (h1 : T = x * S) 
  (h2 : T / (S + T) = 1 / 3) : x = 1 / 2 := by
  sorry

end tips_fraction_of_salary_l157_157846


namespace part1_part2_part3_l157_157851

-- Part 1
theorem part1 (B_count : ℕ) : 
  (1 * 100) + (B_count * 68) + (4 * 20) = 520 → 
  B_count = 5 := 
by sorry

-- Part 2
theorem part2 (A_count B_count : ℕ) : 
  A_count + B_count = 5 → 
  (100 * A_count) + (68 * B_count) = 404 → 
  A_count = 2 ∧ B_count = 3 := 
by sorry

-- Part 3
theorem part3 : 
  ∃ (A_count B_count C_count : ℕ), 
  (A_count <= 16) ∧ (B_count <= 16) ∧ (C_count <= 16) ∧ 
  (A_count + B_count + C_count <= 16) ∧ 
  (100 * A_count + 68 * B_count = 708 ∨ 
   68 * B_count + 20 * C_count = 708 ∨ 
   100 * A_count + 20 * C_count = 708) → 
  ((A_count = 3 ∧ B_count = 6 ∧ C_count = 0) ∨ 
   (A_count = 0 ∧ B_count = 6 ∧ C_count = 15)) := 
by sorry

end part1_part2_part3_l157_157851


namespace find_first_number_l157_157947

open Int

theorem find_first_number (A : ℕ) : 
  (Nat.lcm A 671 = 2310) ∧ (Nat.gcd A 671 = 61) → 
  A = 210 :=
by
  intro h
  sorry

end find_first_number_l157_157947


namespace part1_l157_157678

def is_Xn_function (n : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2 ∧ x1 + x2 = 2 * n

theorem part1 : is_Xn_function 0 (fun x => abs x) ∧ is_Xn_function (1/2) (fun x => x^2 - x) :=
by
  sorry

end part1_l157_157678


namespace union_eq_l157_157133

-- Define the sets M and N
def M : Finset ℕ := {0, 3}
def N : Finset ℕ := {1, 2, 3}

-- Define the proof statement
theorem union_eq : M ∪ N = {0, 1, 2, 3} := 
by
  sorry

end union_eq_l157_157133


namespace false_statement_l157_157944

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end false_statement_l157_157944


namespace least_n_for_obtuse_triangle_l157_157941

namespace obtuse_triangle

-- Define angles and n
def alpha (n : ℕ) : ℝ := 59 + n * 0.02
def beta : ℝ := 60
def gamma (n : ℕ) : ℝ := 61 - n * 0.02

-- Define condition for the triangle being obtuse
def is_obtuse_triangle (n : ℕ) : Prop :=
  alpha n > 90 ∨ gamma n > 90

-- Statement about the smallest n such that the triangle is obtuse
theorem least_n_for_obtuse_triangle : ∃ n : ℕ, n = 1551 ∧ is_obtuse_triangle n :=
by
  -- existence proof ends here, details for proof to be provided separately
  sorry

end obtuse_triangle

end least_n_for_obtuse_triangle_l157_157941


namespace weights_system_l157_157740

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l157_157740


namespace rubble_money_left_l157_157014

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

end rubble_money_left_l157_157014


namespace hexagon_perimeter_l157_157572

theorem hexagon_perimeter (side_length : ℝ) (sides : ℕ) (h_sides : sides = 6) (h_side_length : side_length = 10) :
  sides * side_length = 60 :=
by
  rw [h_sides, h_side_length]
  norm_num

end hexagon_perimeter_l157_157572


namespace right_triangle_side_length_l157_157986

theorem right_triangle_side_length (hypotenuse : ℝ) (θ : ℝ) (sin_30 : Real.sin 30 = 1 / 2) (h : θ = 30) 
  (hyp_len : hypotenuse = 10) : 
  let opposite_side := hypotenuse * Real.sin θ
  opposite_side = 5 := by
  sorry

end right_triangle_side_length_l157_157986


namespace zero_in_A_l157_157666

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end zero_in_A_l157_157666


namespace jack_change_l157_157981

theorem jack_change :
  let discountedCost1 := 4.50
  let discountedCost2 := 4.50
  let discountedCost3 := 5.10
  let cost4 := 7.00
  let totalDiscountedCost := discountedCost1 + discountedCost2 + discountedCost3 + cost4
  let tax := totalDiscountedCost * 0.05
  let taxRounded := 1.06 -- Tax rounded to nearest cent
  let totalCostWithTax := totalDiscountedCost + taxRounded
  let totalCostWithServiceFee := totalCostWithTax + 2.00
  let totalPayment := 20 + 10 + 4 * 1
  let change := totalPayment - totalCostWithServiceFee
  change = 9.84 :=
by
  sorry

end jack_change_l157_157981


namespace john_multiple_is_correct_l157_157403

noncomputable def compute_multiple (cost_per_computer : ℝ) 
                                   (num_computers : ℕ)
                                   (rent : ℝ)
                                   (non_rent_expenses : ℝ)
                                   (profit : ℝ) : ℝ :=
  let total_revenue := (num_computers : ℝ) * cost_per_computer
  let total_expenses := (num_computers : ℝ) * 800 + rent + non_rent_expenses
  let x := (total_expenses + profit) / total_revenue
  x

theorem john_multiple_is_correct :
  compute_multiple 800 60 5000 3000 11200 = 1.4 := by
  sorry

end john_multiple_is_correct_l157_157403


namespace sum_of_volumes_is_correct_l157_157522

-- Define the dimensions of the base of the tank
def tank_base_length : ℝ := 44
def tank_base_width : ℝ := 35

-- Define the increase in water height when the train and the car are submerged
def train_water_height_increase : ℝ := 7
def car_water_height_increase : ℝ := 3

-- Calculate the area of the base of the tank
def base_area : ℝ := tank_base_length * tank_base_width

-- Calculate the volumes of the toy train and the toy car
def volume_train : ℝ := base_area * train_water_height_increase
def volume_car : ℝ := base_area * car_water_height_increase

-- Theorem to prove the sum of the volumes is 15400 cubic centimeters
theorem sum_of_volumes_is_correct : volume_train + volume_car = 15400 := by
  sorry

end sum_of_volumes_is_correct_l157_157522


namespace least_multiple_of_15_greater_than_520_l157_157080

theorem least_multiple_of_15_greater_than_520 : ∃ n : ℕ, n > 520 ∧ n % 15 = 0 ∧ (∀ m : ℕ, m > 520 ∧ m % 15 = 0 → n ≤ m) ∧ n = 525 := 
by
  sorry

end least_multiple_of_15_greater_than_520_l157_157080


namespace number_of_truthful_dwarfs_l157_157748

def num_dwarfs : Nat := 10

def likes_vanilla : Nat := num_dwarfs

def likes_chocolate : Nat := num_dwarfs / 2

def likes_fruit : Nat := 1

theorem number_of_truthful_dwarfs : 
  ∃ t l : Nat, 
  t + l = num_dwarfs ∧  -- total number of dwarfs
  t + 2 * l = likes_vanilla + likes_chocolate + likes_fruit ∧  -- total number of hand raises
  t = 4 :=  -- number of truthful dwarfs
  sorry

end number_of_truthful_dwarfs_l157_157748


namespace even_function_is_a_4_l157_157973

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_is_a_4 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 := by
  sorry

end even_function_is_a_4_l157_157973


namespace parallel_vectors_have_proportional_direction_ratios_l157_157555

theorem parallel_vectors_have_proportional_direction_ratios (m : ℝ) :
  let a := (1, 2)
  let b := (m, 1)
  (a.1 / b.1) = (a.2 / b.2) → m = 1/2 :=
by
  let a := (1, 2)
  let b := (m, 1)
  intro h
  sorry

end parallel_vectors_have_proportional_direction_ratios_l157_157555


namespace exclusive_movies_count_l157_157318

-- Define the conditions
def shared_movies : Nat := 15
def andrew_movies : Nat := 25
def john_movies_exclusive : Nat := 8

-- Define the result calculation
def exclusive_movies (andrew_movies shared_movies john_movies_exclusive : Nat) : Nat :=
  (andrew_movies - shared_movies) + john_movies_exclusive

-- Statement to prove
theorem exclusive_movies_count : exclusive_movies andrew_movies shared_movies john_movies_exclusive = 18 := by
  sorry

end exclusive_movies_count_l157_157318


namespace computer_price_ratio_l157_157468

theorem computer_price_ratio (d : ℝ) (h1 : d + 0.30 * d = 377) :
  ((d + 377) / d) = 2.3 := by
  sorry

end computer_price_ratio_l157_157468


namespace increase_factor_l157_157101

noncomputable def old_plates : ℕ := 26 * 10^3
noncomputable def new_plates : ℕ := 26^4 * 10^4
theorem increase_factor : (new_plates / old_plates) = 175760 := by
  sorry

end increase_factor_l157_157101


namespace total_earnings_first_three_months_l157_157497

-- Definitions
def earning_first_month : ℕ := 350
def earning_second_month : ℕ := 2 * earning_first_month + 50
def earning_third_month : ℕ := 4 * (earning_first_month + earning_second_month)

-- Question restated as a theorem
theorem total_earnings_first_three_months : 
  (earning_first_month + earning_second_month + earning_third_month = 5500) :=
by 
  -- Placeholder for the proof
  sorry

end total_earnings_first_three_months_l157_157497


namespace solution_greater_iff_l157_157642

variables {c c' d d' : ℝ}
variables (hc : c ≠ 0) (hc' : c' ≠ 0)

theorem solution_greater_iff : (∃ x, x = -d / c) > (∃ x, x = -d' / c') ↔ (d' / c') < (d / c) :=
by sorry

end solution_greater_iff_l157_157642


namespace find_distance_of_post_office_from_village_l157_157105

-- Conditions
def rate_to_post_office : ℝ := 12.5
def rate_back_village : ℝ := 2
def total_time : ℝ := 5.8

-- Statement of the theorem
theorem find_distance_of_post_office_from_village (D : ℝ) 
  (travel_time_to : D / rate_to_post_office = D / 12.5) 
  (travel_time_back : D / rate_back_village = D / 2)
  (journey_time_total : D / 12.5 + D / 2 = total_time) : 
  D = 10 := 
sorry

end find_distance_of_post_office_from_village_l157_157105


namespace height_of_water_in_cylindrical_tank_l157_157075

theorem height_of_water_in_cylindrical_tank :
  let r_cone := 15  -- radius of base of conical tank in cm
  let h_cone := 24  -- height of conical tank in cm
  let r_cylinder := 18  -- radius of base of cylindrical tank in cm
  let V_cone := (1 / 3 : ℝ) * Real.pi * r_cone^2 * h_cone  -- volume of conical tank
  let h_cyl := V_cone / (Real.pi * r_cylinder^2)  -- height of water in cylindrical tank
  h_cyl = 5.56 :=
by
  sorry

end height_of_water_in_cylindrical_tank_l157_157075


namespace abs_value_expression_l157_157868

theorem abs_value_expression (m n : ℝ) (h1 : m < 0) (h2 : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 :=
sorry

end abs_value_expression_l157_157868


namespace quadrilateral_area_ratio_l157_157662

noncomputable def area_of_octagon (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

noncomputable def area_of_square (s : ℝ) : ℝ := s^2

theorem quadrilateral_area_ratio (a : ℝ) (s : ℝ)
    (h1 : s = a * Real.sqrt (2 + Real.sqrt 2))
    : (area_of_square s) / (area_of_octagon a) = Real.sqrt 2 / 2 :=
by
  sorry

end quadrilateral_area_ratio_l157_157662


namespace parabola_vertex_and_point_l157_157106

theorem parabola_vertex_and_point (a b c : ℝ) (h_vertex : (1, -2) = (1, a * 1^2 + b * 1 + c))
  (h_point : (3, 7) = (3, a * 3^2 + b * 3 + c)) : a = 3 := 
by {
  sorry
}

end parabola_vertex_and_point_l157_157106


namespace rice_containers_l157_157535

theorem rice_containers (pound_to_ounce : ℕ) (total_rice_lb : ℚ) (container_oz : ℕ) : 
  pound_to_ounce = 16 → 
  total_rice_lb = 33 / 4 → 
  container_oz = 33 → 
  (total_rice_lb * pound_to_ounce) / container_oz = 4 :=
by sorry

end rice_containers_l157_157535


namespace blue_bordered_area_on_outer_sphere_l157_157641

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l157_157641


namespace intersection_M_N_l157_157430

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l157_157430


namespace jenny_eggs_per_basket_l157_157406

theorem jenny_eggs_per_basket :
  ∃ n, (30 % n = 0 ∧ 42 % n = 0 ∧ 18 % n = 0 ∧ n >= 6) → n = 6 :=
by
  sorry

end jenny_eggs_per_basket_l157_157406


namespace obtuse_angle_of_parallel_vectors_l157_157933

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem obtuse_angle_of_parallel_vectors (θ : ℝ) :
  let a := (2, 1 - Real.cos θ)
  let b := (1 + Real.cos θ, 1 / 4)
  is_parallel a b → 90 < θ ∧ θ < 180 → θ = 135 :=
by
  intro ha hb
  sorry

end obtuse_angle_of_parallel_vectors_l157_157933


namespace random_events_l157_157640

/-- Definition of what constitutes a random event --/
def is_random_event (e : String) : Prop :=
  e = "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)" ∨
  e = "Forgetting the last digit of a phone number, randomly pressing and it is correct" ∨
  e = "Winning the first prize in a sports lottery"

/-- Define the specific events --/
def event_1 := "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)"
def event_2 := "Forgetting the last digit of a phone number, randomly pressing and it is correct"
def event_3 := "Opposite electric charges attract each other"
def event_4 := "Winning the first prize in a sports lottery"

/-- Lean 4 statement for the proof problem --/
theorem random_events :
  (is_random_event event_1) ∧
  (is_random_event event_2) ∧
  ¬(is_random_event event_3) ∧
  (is_random_event event_4) :=
by 
  sorry

end random_events_l157_157640


namespace fraction_of_august_tips_l157_157671

variable {A : ℝ} -- A denotes the average monthly tips for the other months.
variable {total_tips_6_months : ℝ} (h1 : total_tips_6_months = 6 * A)
variable {august_tips : ℝ} (h2 : august_tips = 6 * A)
variable {total_tips : ℝ} (h3 : total_tips = total_tips_6_months + august_tips)

theorem fraction_of_august_tips (h1 : total_tips_6_months = 6 * A)
                                (h2 : august_tips = 6 * A)
                                (h3 : total_tips = total_tips_6_months + august_tips) :
    (august_tips / total_tips) = 1 / 2 :=
by
    sorry

end fraction_of_august_tips_l157_157671


namespace three_card_deal_probability_l157_157018

theorem three_card_deal_probability :
  (4 / 52) * (4 / 51) * (4 / 50) = 16 / 33150 := 
by 
  sorry

end three_card_deal_probability_l157_157018


namespace acute_angles_sine_relation_l157_157886

theorem acute_angles_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : α < β :=
by
  sorry

end acute_angles_sine_relation_l157_157886


namespace sum_digits_largest_N_l157_157217

-- Define the conditions
def is_multiple_of_six (N : ℕ) : Prop := N % 6 = 0

def P (N : ℕ) : ℚ := 
  let favorable_positions := (N + 1) *
    (⌊(1:ℚ) / 3 * N⌋ + 1 + (N - ⌈(2:ℚ) / 3 * N⌉ + 1))
  favorable_positions / (N + 1)

axiom P_6_equals_1 : P 6 = 1
axiom P_large_N : ∀ ε > 0, ∃ N > 0, is_multiple_of_six N ∧ P N ≥ (5/6) - ε

-- Main theorem statement
theorem sum_digits_largest_N : 
  ∃ N : ℕ, is_multiple_of_six N ∧ P N > 3/4 ∧ (N.digits 10).sum = 6 :=
sorry

end sum_digits_largest_N_l157_157217


namespace total_pages_in_book_l157_157871

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end total_pages_in_book_l157_157871


namespace trigonometric_inequality_l157_157673

theorem trigonometric_inequality (a b A B : ℝ) (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos 2 * x - B * Real.sin 2 * x ≥ 0) : 
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
sorry

end trigonometric_inequality_l157_157673


namespace total_ticket_cost_l157_157579

theorem total_ticket_cost (adult_tickets student_tickets : ℕ) 
    (price_adult price_student : ℕ) 
    (total_tickets : ℕ) (n_adult_tickets : adult_tickets = 410) 
    (n_student_tickets : student_tickets = 436) 
    (p_adult : price_adult = 6) 
    (p_student : price_student = 3) 
    (total_tickets_sold : total_tickets = 846) : 
    (adult_tickets * price_adult + student_tickets * price_student) = 3768 :=
by
  sorry

end total_ticket_cost_l157_157579


namespace max_value_expression_l157_157792

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (M : ℝ), M = (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ∧ M = 1/24 := 
sorry

end max_value_expression_l157_157792


namespace arithmetic_geometric_progression_l157_157054

theorem arithmetic_geometric_progression (a b : ℝ) :
  (b = 2 - a) ∧ (b = 1 / a ∨ b = -1 / a) →
  (a = 1 ∧ b = 1) ∨
  (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
  (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
by
  sorry

end arithmetic_geometric_progression_l157_157054


namespace gcd_of_numbers_l157_157839

theorem gcd_of_numbers :
  let a := 125^2 + 235^2 + 349^2
  let b := 124^2 + 234^2 + 350^2
  gcd a b = 1 := by
  sorry

end gcd_of_numbers_l157_157839


namespace ab_equals_one_l157_157278

theorem ab_equals_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (f : ℝ → ℝ) (h3 : f = abs ∘ log) (h4 : f a = f b) : a * b = 1 :=
by
  sorry

end ab_equals_one_l157_157278


namespace runners_meet_time_l157_157027

theorem runners_meet_time (t_P t_Q : ℕ) (hP: t_P = 252) (hQ: t_Q = 198) : Nat.lcm t_P t_Q = 2772 :=
by
  rw [hP, hQ]
  -- The proof can be continued by proving the LCM calculation step, which we omit here
  sorry

end runners_meet_time_l157_157027


namespace solve_m_l157_157659

def f (x : ℝ) := 4 * x ^ 2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) := x ^ 2 - m * x - 8

theorem solve_m : ∃ (m : ℝ), f 8 - g 8 m = 20 ∧ m = -25.5 := by
  sorry

end solve_m_l157_157659


namespace sum_as_fraction_l157_157191

theorem sum_as_fraction :
  (0.1 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) = (13467 / 100000 : ℝ) :=
by
  sorry

end sum_as_fraction_l157_157191


namespace base8_subtraction_and_conversion_l157_157237

-- Define the base 8 numbers
def num1 : ℕ := 7463 -- 7463 in base 8
def num2 : ℕ := 3254 -- 3254 in base 8

-- Define the subtraction in base 8 and conversion to base 10
def result_base8 : ℕ := 4207 -- Expected result in base 8
def result_base10 : ℕ := 2183 -- Expected result in base 10

-- Helper function to convert from base 8 to base 10
def convert_base8_to_base10 (n : ℕ) : ℕ := 
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8 + (n % 10)
 
-- Main theorem statement
theorem base8_subtraction_and_conversion :
  (num1 - num2 = result_base8) ∧ (convert_base8_to_base10 result_base8 = result_base10) :=
by
  sorry

end base8_subtraction_and_conversion_l157_157237


namespace fraction_of_pizza_peter_ate_l157_157293

theorem fraction_of_pizza_peter_ate (total_slices : ℕ) (peter_slices : ℕ) (shared_slices : ℚ) 
  (pizza_fraction : ℚ) : 
  total_slices = 16 → 
  peter_slices = 2 → 
  shared_slices = 1/3 → 
  pizza_fraction = peter_slices / total_slices + (1 / 2) * shared_slices / total_slices → 
  pizza_fraction = 13 / 96 :=
by 
  intros h1 h2 h3 h4
  -- to be proved later
  sorry

end fraction_of_pizza_peter_ate_l157_157293


namespace solve_quadratic_eq_l157_157385

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 → x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l157_157385


namespace photos_per_album_correct_l157_157536

-- Define the conditions
def total_photos : ℕ := 4500
def first_batch_photos : ℕ := 1500
def first_batch_albums : ℕ := 30
def second_batch_albums : ℕ := 60
def remaining_photos : ℕ := total_photos - first_batch_photos

-- Define the number of photos per album for the first batch (should be 50)
def photos_per_album_first_batch : ℕ := first_batch_photos / first_batch_albums

-- Define the number of photos per album for the second batch (should be 50)
def photos_per_album_second_batch : ℕ := remaining_photos / second_batch_albums

-- Statement to prove
theorem photos_per_album_correct :
  photos_per_album_first_batch = 50 ∧ photos_per_album_second_batch = 50 :=
by
  simp [photos_per_album_first_batch, photos_per_album_second_batch, remaining_photos]
  sorry

end photos_per_album_correct_l157_157536


namespace sequence_sum_l157_157140

theorem sequence_sum :
  (3 + 13 + 23 + 33 + 43 + 53) + (5 + 15 + 25 + 35 + 45 + 55) = 348 := by
  sorry

end sequence_sum_l157_157140


namespace function_divisibility_l157_157957

theorem function_divisibility
    (f : ℤ → ℕ)
    (h_pos : ∀ x, 0 < f x)
    (h_div : ∀ m n : ℤ, (f m - f n) % f (m - n) = 0) :
    ∀ m n : ℤ, f m ≤ f n → f m ∣ f n :=
by sorry

end function_divisibility_l157_157957


namespace james_total_spent_l157_157369

noncomputable def total_cost : ℝ :=
  let milk_price := 3.0
  let bananas_price := 2.0
  let bread_price := 1.5
  let cereal_price := 4.0
  let milk_tax := 0.20
  let bananas_tax := 0.15
  let bread_tax := 0.10
  let cereal_tax := 0.25
  let milk_total := milk_price * (1 + milk_tax)
  let bananas_total := bananas_price * (1 + bananas_tax)
  let bread_total := bread_price * (1 + bread_tax)
  let cereal_total := cereal_price * (1 + cereal_tax)
  milk_total + bananas_total + bread_total + cereal_total

theorem james_total_spent : total_cost = 12.55 :=
  sorry

end james_total_spent_l157_157369


namespace hyperbola_asymptote_eqn_l157_157393

theorem hyperbola_asymptote_eqn :
  ∀ (x y : ℝ),
  (y ^ 2 / 4 - x ^ 2 = 1) → (y = 2 * x ∨ y = -2 * x) := by
sorry

end hyperbola_asymptote_eqn_l157_157393


namespace merry_boxes_on_sunday_l157_157304

theorem merry_boxes_on_sunday
  (num_boxes_saturday : ℕ := 50)
  (apples_per_box : ℕ := 10)
  (total_apples_sold : ℕ := 720)
  (remaining_boxes : ℕ := 3) :
  num_boxes_saturday * apples_per_box ≤ total_apples_sold →
  (total_apples_sold - num_boxes_saturday * apples_per_box) / apples_per_box + remaining_boxes = 25 := by
  intros
  sorry

end merry_boxes_on_sunday_l157_157304


namespace solve_quadratic_eqn_l157_157257

theorem solve_quadratic_eqn :
  ∃ x₁ x₂ : ℝ, (x - 6) * (x + 2) = 0 ↔ (x = x₁ ∨ x = x₂) ∧ x₁ = 6 ∧ x₂ = -2 :=
by
  sorry

end solve_quadratic_eqn_l157_157257


namespace inequality_no_solution_iff_a_le_neg3_l157_157653

theorem inequality_no_solution_iff_a_le_neg3 (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x + 2| < a)) ↔ a ≤ -3 := 
sorry

end inequality_no_solution_iff_a_le_neg3_l157_157653


namespace work_problem_l157_157098

theorem work_problem (A B C : ℝ) (hB : B = 3) (h1 : 1 / B + 1 / C = 1 / 2) (h2 : 1 / A + 1 / C = 1 / 2) : A = 3 := by
  sorry

end work_problem_l157_157098


namespace graph_does_not_pass_first_quadrant_l157_157234

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l157_157234


namespace largest_angle_isosceles_triangle_l157_157704

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end largest_angle_isosceles_triangle_l157_157704


namespace contradiction_to_at_least_one_not_greater_than_60_l157_157381

-- Define a condition for the interior angles of a triangle being > 60
def all_angles_greater_than_60 (α β γ : ℝ) : Prop :=
  α > 60 ∧ β > 60 ∧ γ > 60

-- Define the negation of the proposition "At least one of the interior angles is not greater than 60"
def at_least_one_not_greater_than_60 (α β γ : ℝ) : Prop :=
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The mathematically equivalent proof problem
theorem contradiction_to_at_least_one_not_greater_than_60 (α β γ : ℝ) :
  ¬ at_least_one_not_greater_than_60 α β γ ↔ all_angles_greater_than_60 α β γ := by
  sorry

end contradiction_to_at_least_one_not_greater_than_60_l157_157381


namespace fractional_part_painted_correct_l157_157292

noncomputable def fractional_part_painted (time_fence : ℕ) (time_hole : ℕ) : ℚ :=
  (time_hole : ℚ) / time_fence

theorem fractional_part_painted_correct : fractional_part_painted 60 40 = 2 / 3 := by
  sorry

end fractional_part_painted_correct_l157_157292


namespace total_candy_bars_correct_l157_157230

-- Define the number of each type of candy bar.
def snickers : Nat := 3
def marsBars : Nat := 2
def butterfingers : Nat := 7

-- Define the total number of candy bars.
def totalCandyBars : Nat := snickers + marsBars + butterfingers

-- Formulate the theorem about the total number of candy bars.
theorem total_candy_bars_correct : totalCandyBars = 12 :=
sorry

end total_candy_bars_correct_l157_157230


namespace master_parts_per_hour_l157_157504

variable (x : ℝ)

theorem master_parts_per_hour (h1 : 300 / x = 100 / (40 - x)) : 300 / x = 100 / (40 - x) :=
sorry

end master_parts_per_hour_l157_157504


namespace chocolate_candy_cost_l157_157274

-- Define the constants and conditions
def cost_per_box : ℕ := 5
def candies_per_box : ℕ := 30
def discount_rate : ℝ := 0.1

-- Define the total number of candies to buy
def total_candies : ℕ := 450

-- Define the threshold for applying discount
def discount_threshold : ℕ := 300

-- Calculate the number of boxes needed
def boxes_needed (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the total cost without discount
def total_cost (boxes_needed : ℕ) (cost_per_box : ℕ) : ℝ :=
  boxes_needed * cost_per_box

-- Calculate the discounted cost
def discounted_cost (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

-- Statement to be proved
theorem chocolate_candy_cost :
  discounted_cost 
    (total_cost (boxes_needed total_candies candies_per_box) cost_per_box) 
    discount_rate = 67.5 :=
by
  -- Proof is needed here, using the correct steps from the solution.
  sorry

end chocolate_candy_cost_l157_157274


namespace exists_powers_mod_eq_l157_157999

theorem exists_powers_mod_eq (N : ℕ) (A : ℤ) : ∃ r s : ℕ, r ≠ s ∧ (A ^ r - A ^ s) % N = 0 :=
sorry

end exists_powers_mod_eq_l157_157999


namespace solve_inequality_l157_157548

theorem solve_inequality (a x : ℝ) : 
  if a > 0 then -a < x ∧ x < 2*a else if a < 0 then 2*a < x ∧ x < -a else False :=
by sorry

end solve_inequality_l157_157548


namespace profit_amount_l157_157160

theorem profit_amount (SP : ℝ) (P : ℝ) (profit : ℝ) : 
  SP = 850 → P = 36 → profit = SP - SP / (1 + P / 100) → profit = 225 :=
by
  intros hSP hP hProfit
  rw [hSP, hP] at *
  simp at *
  sorry

end profit_amount_l157_157160


namespace circumference_to_diameter_ratio_l157_157670

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) :
  C / D = 3.14 :=
by
  rw [hC, hD]
  norm_num

end circumference_to_diameter_ratio_l157_157670


namespace evaluate_expression_l157_157621

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l157_157621


namespace pen_case_cost_l157_157273

noncomputable def case_cost (p i c : ℝ) : Prop :=
  p + i + c = 2.30 ∧
  p = 1.50 + i ∧
  c = 0.5 * i →
  c = 0.1335

theorem pen_case_cost (p i c : ℝ) : case_cost p i c :=
by
  sorry

end pen_case_cost_l157_157273


namespace f_divisible_by_13_l157_157763

def f : ℕ → ℤ := sorry

theorem f_divisible_by_13 :
  (f 0 = 0) ∧ (f 1 = 0) ∧
  (∀ n, f (n + 2) = 4 ^ (n + 2) * f (n + 1) - 16 ^ (n + 1) * f n + n * 2 ^ (n ^ 2)) →
  (f 1989 % 13 = 0) ∧ (f 1990 % 13 = 0) ∧ (f 1991 % 13 = 0) :=
by
  intros h
  sorry

end f_divisible_by_13_l157_157763


namespace race_completion_times_l157_157844

theorem race_completion_times :
  ∃ (Patrick Manu Amy Olivia Sophie Jack : ℕ),
  Patrick = 60 ∧
  Manu = Patrick + 12 ∧
  Amy = Manu / 2 ∧
  Olivia = (2 * Amy) / 3 ∧
  Sophie = Olivia - 10 ∧
  Jack = Sophie + 8 ∧
  Manu = 72 ∧
  Amy = 36 ∧
  Olivia = 24 ∧
  Sophie = 14 ∧
  Jack = 22 := 
by
  -- proof here
  sorry

end race_completion_times_l157_157844


namespace workers_new_daily_wage_l157_157131

def wage_before : ℝ := 25
def increase_percentage : ℝ := 0.40

theorem workers_new_daily_wage : wage_before * (1 + increase_percentage) = 35 :=
by
  -- sorry will be replaced by the actual proof steps
  sorry

end workers_new_daily_wage_l157_157131


namespace max_small_packages_l157_157370

theorem max_small_packages (L S : ℝ) (W : ℝ) (h1 : W = 12 * L) (h2 : W = 20 * S) :
  (∃ n_smalls, n_smalls = 5 ∧ W - 9 * L = n_smalls * S) :=
by
  sorry

end max_small_packages_l157_157370


namespace fraction_of_income_from_tips_l157_157990

variable (S T I : ℝ)

theorem fraction_of_income_from_tips (h1 : T = (5 / 2) * S) (h2 : I = S + T) : 
  T / I = 5 / 7 := by
  sorry

end fraction_of_income_from_tips_l157_157990


namespace curve_is_circle_l157_157354

theorem curve_is_circle (r θ : ℝ) (h : r = 3 * Real.sin θ) : 
  ∃ c : ℝ × ℝ, c = (0, 3 / 2) ∧ ∀ p : ℝ × ℝ, ∃ R : ℝ, R = 3 / 2 ∧ 
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = R^2 :=
sorry

end curve_is_circle_l157_157354


namespace solve_system_l157_157902

theorem solve_system :
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (14.996, 19.994)) ∨
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (0.421, 1.561)) :=
  sorry

end solve_system_l157_157902


namespace ratio_shorter_longer_l157_157232

theorem ratio_shorter_longer (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 21) 
  (h2 : shorter_length = 6) 
  (h3 : longer_length = total_length - shorter_length) 
  (h4 : shorter_length / longer_length = 2 / 5) : 
  shorter_length / longer_length = 2 / 5 :=
by sorry

end ratio_shorter_longer_l157_157232


namespace wombats_count_l157_157916

theorem wombats_count (W : ℕ) (H : 4 * W + 3 = 39) : W = 9 := 
sorry

end wombats_count_l157_157916


namespace turkey_2003_problem_l157_157813

theorem turkey_2003_problem (x m n : ℕ) (hx : 0 < x) (hm : 0 < m) (hn : 0 < n) (h : x^m = 2^(2 * n + 1) + 2^n + 1) :
  x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1 ∨ x = 23 ∧ m = 2 ∧ n = 4 :=
sorry

end turkey_2003_problem_l157_157813


namespace value_of_expression_l157_157830

theorem value_of_expression (a b : ℤ) (h : a - 2 * b - 3 = 0) : 9 - 2 * a + 4 * b = 3 := 
by 
  sorry

end value_of_expression_l157_157830


namespace trigonometric_identity_l157_157783

theorem trigonometric_identity (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (Real.cos (α / 2) ^ 2) = 4 * Real.sin α :=
by
  sorry

end trigonometric_identity_l157_157783


namespace valid_outfits_l157_157259

-- Let's define the conditions first:
variable (shirts colors pairs : ℕ)

-- Suppose we have the following constraints according to the given problem:
def totalShirts : ℕ := 6
def totalPants : ℕ := 6
def totalHats : ℕ := 6
def totalShoes : ℕ := 6
def numOfColors : ℕ := 6

-- We refuse to wear an outfit in which all 4 items are the same color, or in which the shoes match the color of any other item.
theorem valid_outfits : 
  (totalShirts * totalPants * totalHats * (totalShoes - 1) + (totalShirts * 5 - totalShoes)) = 1104 :=
by sorry

end valid_outfits_l157_157259


namespace solution_k_system_eq_l157_157208

theorem solution_k_system_eq (x y k : ℝ) 
  (h1 : x + y = 5 * k) 
  (h2 : x - y = k) 
  (h3 : 2 * x + 3 * y = 24) : k = 2 :=
by
  sorry

end solution_k_system_eq_l157_157208


namespace num_zeros_of_g_l157_157276

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x
else -(x^2 - 2 * -x)

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem num_zeros_of_g : ∃! x : ℝ, g x = 0 := sorry

end num_zeros_of_g_l157_157276


namespace shopkeeper_standard_weight_l157_157869

theorem shopkeeper_standard_weight
    (cost_price : ℝ)
    (actual_weight_used : ℝ)
    (profit_percentage : ℝ)
    (standard_weight : ℝ)
    (H1 : actual_weight_used = 800)
    (H2 : profit_percentage = 25) :
    standard_weight = 1000 :=
by 
    sorry

end shopkeeper_standard_weight_l157_157869


namespace sum_first_10_terms_arithmetic_sequence_l157_157312

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l157_157312


namespace maximize_profit_l157_157787

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l157_157787


namespace probability_queen_of_diamonds_l157_157239

/-- 
A standard deck of 52 cards consists of 13 ranks and 4 suits.
We want to prove that the probability the top card is the Queen of Diamonds is 1/52.
-/
theorem probability_queen_of_diamonds 
  (total_cards : ℕ) 
  (queen_of_diamonds : ℕ)
  (h1 : total_cards = 52)
  (h2 : queen_of_diamonds = 1) : 
  (queen_of_diamonds : ℚ) / (total_cards : ℚ) = 1 / 52 := 
by 
  sorry

end probability_queen_of_diamonds_l157_157239


namespace squirrel_nuts_collection_l157_157307

theorem squirrel_nuts_collection (n : ℕ) (e u : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = u k + k) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n → u k = e (k + 1) + u k / 100) ∧
  e n = n →
  n = 99 → 
  (∃ S : ℕ, (∀ k, 1 ≤ k ∧ k ≤ n → e k = S)) ∧ 
  S = 9801 :=
sorry

end squirrel_nuts_collection_l157_157307


namespace num_convex_quadrilateral_angles_arith_prog_l157_157125

theorem num_convex_quadrilateral_angles_arith_prog :
  ∃ (S : Finset (Finset ℤ)), S.card = 29 ∧
    ∀ {a b c d : ℤ}, {a, b, c, d} ∈ S →
      a + b + c + d = 360 ∧
      a < b ∧ b < c ∧ c < d ∧
      ∃ (m d_diff : ℤ), 
        m - d_diff = a ∧
        m = b ∧
        m + d_diff = c ∧
        m + 2 * d_diff = d ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end num_convex_quadrilateral_angles_arith_prog_l157_157125


namespace probability_spade_heart_diamond_l157_157811

-- Condition: Definition of probability functions and a standard deck
def probability_of_first_spade (deck : Finset ℕ) : ℚ := 13 / 52
def probability_of_second_heart (deck : Finset ℕ) (first_card_spade : Prop) : ℚ := 13 / 51
def probability_of_third_diamond (deck : Finset ℕ) (first_card_spade : Prop) (second_card_heart : Prop) : ℚ := 13 / 50

-- Combined probability calculation
def probability_sequence_spade_heart_diamond (deck : Finset ℕ) : ℚ := 
  probability_of_first_spade deck * 
  probability_of_second_heart deck (true) * 
  probability_of_third_diamond deck (true) (true)

-- Lean statement proving the problem
theorem probability_spade_heart_diamond :
  probability_sequence_spade_heart_diamond (Finset.range 52) = 2197 / 132600 :=
by
  -- Proof steps will go here
  sorry

end probability_spade_heart_diamond_l157_157811


namespace Faye_age_l157_157453

theorem Faye_age (D E C F : ℕ) (h1 : D = E - 5) (h2 : E = C + 3) (h3 : F = C + 2) (hD : D = 18) : F = 22 :=
by
  sorry

end Faye_age_l157_157453


namespace maximize_winning_probability_l157_157262

def ahmet_wins (n : ℕ) : Prop :=
  n = 13

theorem maximize_winning_probability :
  ∃ n ∈ {x : ℕ | x ≥ 1 ∧ x ≤ 25}, ahmet_wins n :=
by
  sorry

end maximize_winning_probability_l157_157262


namespace eval_expression_l157_157148

theorem eval_expression (a : ℕ) (h : a = 2) : a^3 * a^6 = 512 := by
  sorry

end eval_expression_l157_157148


namespace derivative_value_at_pi_over_12_l157_157970

open Real

theorem derivative_value_at_pi_over_12 :
  let f (x : ℝ) := cos (2 * x + π / 3)
  deriv f (π / 12) = -2 :=
by
  let f (x : ℝ) := cos (2 * x + π / 3)
  sorry

end derivative_value_at_pi_over_12_l157_157970


namespace ratio_of_inradii_l157_157043

-- Given triangle XYZ with sides XZ=5, YZ=12, XY=13
-- Let W be on XY such that ZW bisects ∠ YZX
-- The inscribed circles of triangles ZWX and ZWY have radii r_x and r_y respectively
-- Prove the ratio r_x / r_y = 1/6

theorem ratio_of_inradii
  (XZ YZ XY : ℝ)
  (W : ℝ)
  (r_x r_y : ℝ)
  (h1 : XZ = 5)
  (h2 : YZ = 12)
  (h3 : XY = 13)
  (h4 : r_x / r_y = 1/6) :
  r_x / r_y = 1/6 :=
by sorry

end ratio_of_inradii_l157_157043


namespace meaningful_expression_range_l157_157530

theorem meaningful_expression_range (x : ℝ) : (¬ (x - 1 = 0)) ↔ (x ≠ 1) := 
by
  sorry

end meaningful_expression_range_l157_157530


namespace relationship_among_abc_l157_157487

-- Define a, b, c
def a : ℕ := 22 ^ 55
def b : ℕ := 33 ^ 44
def c : ℕ := 55 ^ 33

-- State the theorem regarding the relationship among a, b, and c
theorem relationship_among_abc : a > b ∧ b > c := 
by
  -- Placeholder for the proof, not required for this task
  sorry

end relationship_among_abc_l157_157487


namespace find_smaller_number_l157_157975

theorem find_smaller_number (x : ℕ) (h1 : ∃ y, y = 3 * x) (h2 : x + 3 * x = 124) : x = 31 :=
by
  -- Proof will be here
  sorry

end find_smaller_number_l157_157975


namespace base_value_l157_157719

theorem base_value (b : ℕ) : (b - 1)^2 * (b - 2) = 256 → b = 17 :=
by
  sorry

end base_value_l157_157719


namespace num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l157_157380

theorem num_three_digit_numbers_divisible_by_5_and_6_with_digit_6 : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (6 ∈ n.digits 10)) ∧ S.card = 6 :=
by
  sorry

end num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l157_157380


namespace total_decorations_l157_157081

-- Define the conditions
def decorations_per_box := 4 + 1 + 5
def total_boxes := 11 + 1

-- Statement of the problem: Prove that the total number of decorations handed out is 120
theorem total_decorations : total_boxes * decorations_per_box = 120 := by
  sorry

end total_decorations_l157_157081


namespace min_blue_eyes_with_lunchbox_l157_157494

theorem min_blue_eyes_with_lunchbox (B L : Finset Nat) (hB : B.card = 15) (hL : L.card = 25) (students : Finset Nat) (hst : students.card = 35)  : 
  ∃ (x : Finset Nat), x ⊆ B ∧ x ⊆ L ∧ x.card ≥ 5 :=
by
  sorry

end min_blue_eyes_with_lunchbox_l157_157494


namespace largest_n_for_factorable_polynomial_l157_157130

theorem largest_n_for_factorable_polynomial :
  ∃ (n : ℤ), (∀ A B : ℤ, 7 * A * B = 56 → n ≤ 7 * B + A) ∧ n = 393 :=
by {
  sorry
}

end largest_n_for_factorable_polynomial_l157_157130


namespace tory_toys_sold_is_7_l157_157064

-- Define the conditions as Lean definitions
def bert_toy_phones_sold : Nat := 8
def price_per_toy_phone : Nat := 18
def bert_earnings : Nat := bert_toy_phones_sold * price_per_toy_phone
def tory_earnings : Nat := bert_earnings - 4
def price_per_toy_gun : Nat := 20
def tory_toys_sold := tory_earnings / price_per_toy_gun

-- Prove that the number of toy guns Tory sold is 7
theorem tory_toys_sold_is_7 : tory_toys_sold = 7 :=
by
  sorry

end tory_toys_sold_is_7_l157_157064


namespace factor_difference_of_squares_l157_157163

theorem factor_difference_of_squares (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) :=
by
  sorry

end factor_difference_of_squares_l157_157163


namespace max_value_of_sin_l157_157884

theorem max_value_of_sin (x : ℝ) : (2 * Real.sin x) ≤ 2 :=
by
  -- this theorem directly implies that 2sin(x) has a maximum value of 2.
  sorry

end max_value_of_sin_l157_157884


namespace oil_tank_depth_l157_157051

theorem oil_tank_depth (L r A : ℝ) (h : ℝ) (L_pos : L = 8) (r_pos : r = 2) (A_pos : A = 16) :
  h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

end oil_tank_depth_l157_157051


namespace sum_of_coordinates_D_l157_157321

theorem sum_of_coordinates_D
    (C N D : ℝ × ℝ) 
    (hC : C = (10, 5))
    (hN : N = (4, 9))
    (h_midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
    C.1 + D.1 + (C.2 + D.2) = 22 :=
  by sorry

end sum_of_coordinates_D_l157_157321


namespace tan_half_product_values_l157_157568

theorem tan_half_product_values (a b : ℝ) (h : 3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0) : 
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = -4 ∨ x = -1) := sorry

end tan_half_product_values_l157_157568


namespace sophie_perceived_height_in_mirror_l157_157734

noncomputable def inch_to_cm : ℝ := 2.5

noncomputable def sophie_height_in_inches : ℝ := 50

noncomputable def sophie_height_in_cm := sophie_height_in_inches * inch_to_cm

noncomputable def perceived_height := sophie_height_in_cm * 2

theorem sophie_perceived_height_in_mirror : perceived_height = 250 :=
by
  unfold perceived_height
  unfold sophie_height_in_cm
  unfold sophie_height_in_inches
  unfold inch_to_cm
  sorry

end sophie_perceived_height_in_mirror_l157_157734


namespace smallest_real_number_l157_157715

theorem smallest_real_number (A B C D : ℝ) 
  (hA : A = |(-2 : ℝ)|) 
  (hB : B = -1) 
  (hC : C = 0) 
  (hD : D = -1 / 2) : 
  min A (min B (min C D)) = B := 
by
  sorry

end smallest_real_number_l157_157715


namespace yunjeong_locker_problem_l157_157867

theorem yunjeong_locker_problem
  (l r f b : ℕ)
  (h_l : l = 7)
  (h_r : r = 13)
  (h_f : f = 8)
  (h_b : b = 14)
  (same_rows : ∀ pos1 pos2 : ℕ, pos1 = pos2) :
  (l - 1) + (r - 1) + (f - 1) + (b - 1) = 399 := sorry

end yunjeong_locker_problem_l157_157867


namespace find_f_of_neg2_l157_157992

theorem find_f_of_neg2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x + 1) = 9 * x ^ 2 - 6 * x + 5) : f (-2) = 20 :=
by
  sorry

end find_f_of_neg2_l157_157992


namespace number_of_C_animals_l157_157759

-- Define the conditions
def A : ℕ := 45
def B : ℕ := 32
def C : ℕ := 5

-- Define the theorem that we need to prove
theorem number_of_C_animals : B + C = A - 8 :=
by
  -- placeholder to complete the proof (not part of the problem's requirement)
  sorry

end number_of_C_animals_l157_157759


namespace units_digit_7_pow_1023_l157_157544

-- Define a function for the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_1023 :
  units_digit (7 ^ 1023) = 3 :=
by
  sorry

end units_digit_7_pow_1023_l157_157544


namespace fixed_point_is_one_three_l157_157421

noncomputable def fixed_point_of_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : ℝ × ℝ :=
  (1, 3)

theorem fixed_point_is_one_three {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  fixed_point_of_function a h_pos h_ne_one = (1, 3) :=
  sorry

end fixed_point_is_one_three_l157_157421


namespace conversion_1_conversion_2_conversion_3_l157_157629

theorem conversion_1 : 2 * 1000 = 2000 := sorry

theorem conversion_2 : 9000 / 1000 = 9 := sorry

theorem conversion_3 : 8 * 1000 = 8000 := sorry

end conversion_1_conversion_2_conversion_3_l157_157629


namespace A_lent_5000_to_B_l157_157343

noncomputable def principalAmountB
    (P_C : ℝ)
    (r : ℝ)
    (total_interest : ℝ)
    (P_B : ℝ) : Prop :=
  let I_B := P_B * r * 2
  let I_C := P_C * r * 4
  I_B + I_C = total_interest

theorem A_lent_5000_to_B :
  principalAmountB 3000 0.10 2200 5000 :=
by
  sorry

end A_lent_5000_to_B_l157_157343


namespace rectangular_solid_width_l157_157062

theorem rectangular_solid_width 
  (l : ℝ) (w : ℝ) (h : ℝ) (S : ℝ)
  (hl : l = 5)
  (hh : h = 1)
  (hs : S = 58) :
  2 * l * w + 2 * l * h + 2 * w * h = S → w = 4 := 
by
  intros h_surface_area 
  sorry

end rectangular_solid_width_l157_157062


namespace standard_equation_of_tangent_circle_l157_157735

theorem standard_equation_of_tangent_circle (r h k : ℝ)
  (h_r : r = 1) 
  (h_k : k = 1) 
  (h_center_quadrant : h > 0 ∧ k > 0)
  (h_tangent_x_axis : k = r) 
  (h_tangent_line : r = abs (4 * h - 3) / 5)
  : (x - 2)^2 + (y - 1)^2 = 1 := 
by {
  sorry
}

end standard_equation_of_tangent_circle_l157_157735


namespace trapezoid_area_l157_157203

theorem trapezoid_area (EF GH h : ℕ) (hEF : EF = 60) (hGH : GH = 30) (hh : h = 15) : 
  (EF + GH) * h / 2 = 675 := by 
  sorry

end trapezoid_area_l157_157203


namespace q_sufficient_not_necessary_p_l157_157011

theorem q_sufficient_not_necessary_p (x : ℝ) (p : Prop) (q : Prop) :
  (p ↔ |x| < 2) →
  (q ↔ x^2 - x - 2 < 0) →
  (q → p) ∧ (p ∧ ¬q) :=
by
  sorry

end q_sufficient_not_necessary_p_l157_157011


namespace intersection_of_sets_l157_157077

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x < 2

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by
  sorry

end intersection_of_sets_l157_157077


namespace ratio_student_adult_tickets_l157_157478

theorem ratio_student_adult_tickets (A : ℕ) (S : ℕ) (total_tickets: ℕ) (multiple: ℕ) :
  (A = 122) →
  (total_tickets = 366) →
  (S = multiple * A) →
  (S + A = total_tickets) →
  (S / A = 2) :=
by
  intros hA hTotal hMultiple hSum
  -- The proof will go here
  sorry

end ratio_student_adult_tickets_l157_157478


namespace combined_total_cost_is_correct_l157_157114

-- Define the number and costs of balloons for each person
def Fred_yellow_count : ℕ := 5
def Fred_red_count : ℕ := 3
def Fred_yellow_cost_per : ℕ := 3
def Fred_red_cost_per : ℕ := 4

def Sam_yellow_count : ℕ := 6
def Sam_red_count : ℕ := 4
def Sam_yellow_cost_per : ℕ := 4
def Sam_red_cost_per : ℕ := 5

def Mary_yellow_count : ℕ := 7
def Mary_red_count : ℕ := 5
def Mary_yellow_cost_per : ℕ := 5
def Mary_red_cost_per : ℕ := 6

def Susan_yellow_count : ℕ := 4
def Susan_red_count : ℕ := 6
def Susan_yellow_cost_per : ℕ := 6
def Susan_red_cost_per : ℕ := 7

def Tom_yellow_count : ℕ := 10
def Tom_red_count : ℕ := 8
def Tom_yellow_cost_per : ℕ := 2
def Tom_red_cost_per : ℕ := 3

-- Formula to calculate total cost for a given person
def total_cost (yellow_count red_count yellow_cost_per red_cost_per : ℕ) : ℕ :=
  (yellow_count * yellow_cost_per) + (red_count * red_cost_per)

-- Total costs for each person
def Fred_total_cost := total_cost Fred_yellow_count Fred_red_count Fred_yellow_cost_per Fred_red_cost_per
def Sam_total_cost := total_cost Sam_yellow_count Sam_red_count Sam_yellow_cost_per Sam_red_cost_per
def Mary_total_cost := total_cost Mary_yellow_count Mary_red_count Mary_yellow_cost_per Mary_red_cost_per
def Susan_total_cost := total_cost Susan_yellow_count Susan_red_count Susan_yellow_cost_per Susan_red_cost_per
def Tom_total_cost := total_cost Tom_yellow_count Tom_red_count Tom_yellow_cost_per Tom_red_cost_per

-- Combined total cost
def combined_total_cost : ℕ :=
  Fred_total_cost + Sam_total_cost + Mary_total_cost + Susan_total_cost + Tom_total_cost

-- Lean statement to prove
theorem combined_total_cost_is_correct : combined_total_cost = 246 :=
by
  dsimp [combined_total_cost, Fred_total_cost, Sam_total_cost, Mary_total_cost, Susan_total_cost, Tom_total_cost, total_cost]
  sorry

end combined_total_cost_is_correct_l157_157114


namespace s_eq_sin_c_eq_cos_l157_157636

open Real

variables (s c : ℝ → ℝ)

-- Conditions
def s_prime := ∀ x, deriv s x = c x
def c_prime := ∀ x, deriv c x = -s x
def initial_conditions := (s 0 = 0) ∧ (c 0 = 1)

-- Theorem to prove
theorem s_eq_sin_c_eq_cos
  (h1 : s_prime s c)
  (h2 : c_prime s c)
  (h3 : initial_conditions s c) :
  (∀ x, s x = sin x) ∧ (∀ x, c x = cos x) :=
sorry

end s_eq_sin_c_eq_cos_l157_157636


namespace lowest_possible_price_l157_157699

theorem lowest_possible_price 
  (MSRP : ℝ)
  (regular_discount_percentage additional_discount_percentage : ℝ)
  (h1 : MSRP = 40)
  (h2 : regular_discount_percentage = 0.30)
  (h3 : additional_discount_percentage = 0.20) : 
  (MSRP * (1 - regular_discount_percentage) * (1 - additional_discount_percentage) = 22.40) := 
by
  sorry

end lowest_possible_price_l157_157699


namespace tens_digit_of_smallest_even_five_digit_number_l157_157189

def smallest_even_five_digit_number (digits : List ℕ) : ℕ :=
if h : 0 ∈ digits ∧ 3 ∈ digits ∧ 5 ∈ digits ∧ 6 ∈ digits ∧ 8 ∈ digits then
  35086
else
  0  -- this is just a placeholder to make the function total

theorem tens_digit_of_smallest_even_five_digit_number : 
  ∀ digits : List ℕ, 
    0 ∈ digits ∧ 
    3 ∈ digits ∧ 
    5 ∈ digits ∧ 
    6 ∈ digits ∧ 
    8 ∈ digits ∧ 
    digits.length = 5 → 
    (smallest_even_five_digit_number digits) / 10 % 10 = 8 :=
by
  intros digits h
  sorry

end tens_digit_of_smallest_even_five_digit_number_l157_157189


namespace power_problem_l157_157253

theorem power_problem (k : ℕ) (h : 6 ^ k = 4) : 6 ^ (2 * k + 3) = 3456 := 
by 
  sorry

end power_problem_l157_157253


namespace simplify_and_evaluate_l157_157277

-- Define the expression
def expr (x : ℝ) : ℝ := x^2 * (x + 1) - x * (x^2 - x + 1)

-- The main theorem stating the equivalence
theorem simplify_and_evaluate (x : ℝ) (h : x = 5) : expr x = 45 :=
by {
  sorry
}

end simplify_and_evaluate_l157_157277


namespace cake_remaining_portion_l157_157564

theorem cake_remaining_portion (initial_cake : ℝ) (alex_share_percentage : ℝ) (jordan_share_fraction : ℝ) :
  initial_cake = 1 ∧ alex_share_percentage = 0.4 ∧ jordan_share_fraction = 0.5 →
  (initial_cake - alex_share_percentage * initial_cake) * (1 - jordan_share_fraction) = 0.3 :=
by
  sorry

end cake_remaining_portion_l157_157564


namespace rectangle_perimeter_l157_157168

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b = 2 * (a + b))) : 2 * (a + b) = 36 :=
by sorry

end rectangle_perimeter_l157_157168


namespace sleep_hours_for_desired_average_l157_157649

theorem sleep_hours_for_desired_average 
  (s_1 s_2 : ℝ) (h_1 h_2 : ℝ) (k : ℝ) 
  (h_inverse_relation : ∀ s h, s * h = k)
  (h_s1 : s_1 = 75)
  (h_h1 : h_1 = 6)
  (h_average : (s_1 + s_2) / 2 = 85) : 
  h_2 = 450 / 95 := 
by 
  sorry

end sleep_hours_for_desired_average_l157_157649


namespace ed_initial_money_l157_157050

-- Define initial conditions
def cost_per_hour_night : ℝ := 1.50
def hours_at_night : ℕ := 6
def cost_per_hour_morning : ℝ := 2
def hours_in_morning : ℕ := 4
def money_left : ℝ := 63

-- Total cost calculation
def total_cost : ℝ :=
  (cost_per_hour_night * hours_at_night) + (cost_per_hour_morning * hours_in_morning)

-- Problem statement to prove
theorem ed_initial_money : money_left + total_cost = 80 :=
by sorry

end ed_initial_money_l157_157050


namespace preferred_dividend_rate_l157_157414

noncomputable def dividend_rate_on_preferred_shares
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℕ)
  (semi_annual_dividend_common : ℚ)
  (total_annual_dividend : ℚ)
  (dividend_rate_preferred : ℚ) : Prop :=
  preferred_shares * par_value * (dividend_rate_preferred / 100) +
  2 * (common_shares * par_value * (semi_annual_dividend_common / 100)) =
  total_annual_dividend

theorem preferred_dividend_rate
  (h1 : 1200 = 1200)
  (h2 : 3000 = 3000)
  (h3 : 50 = 50)
  (h4 : 3.5 = 3.5)
  (h5 : 16500 = 16500) :
  dividend_rate_on_preferred_shares 1200 3000 50 3.5 16500 10 :=
by sorry

end preferred_dividend_rate_l157_157414


namespace scientific_notation_l157_157891

theorem scientific_notation : 350000000 = 3.5 * 10^8 :=
by
  sorry

end scientific_notation_l157_157891


namespace Kylie_uses_3_towels_in_one_month_l157_157774

-- Define the necessary variables and conditions
variable (daughters_towels : Nat) (husband_towels : Nat) (loads : Nat) (towels_per_load : Nat)
variable (K : Nat) -- number of bath towels Kylie uses

-- Given conditions
axiom h1 : daughters_towels = 6
axiom h2 : husband_towels = 3
axiom h3 : loads = 3
axiom h4 : towels_per_load = 4
axiom h5 : (K + daughters_towels + husband_towels) = (loads * towels_per_load)

-- Prove that K = 3
theorem Kylie_uses_3_towels_in_one_month : K = 3 :=
by
  sorry

end Kylie_uses_3_towels_in_one_month_l157_157774


namespace number_of_possible_IDs_l157_157198

theorem number_of_possible_IDs : 
  ∃ (n : ℕ), 
  (∀ (a b : Fin 26) (x y : Fin 10),
    a = b ∨ x = y ∨ (a = b ∧ x = y) → 
    n = 9100) :=
sorry

end number_of_possible_IDs_l157_157198


namespace maximum_area_of_region_l157_157061

/-- Given four circles with radii 2, 4, 6, and 8, tangent to the same point B 
on a line ℓ, with the two largest circles (radii 6 and 8) on the same side of ℓ,
prove that the maximum possible area of the region consisting of points lying
inside exactly one of these circles is 120π. -/
theorem maximum_area_of_region 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) (radius4 : ℝ)
  (line : ℝ → Prop) (B : ℝ)
  (tangent1 : ∀ x, line x → dist x B = radius1) 
  (tangent2 : ∀ x, line x → dist x B = radius2)
  (tangent3 : ∀ x, line x → dist x B = radius3)
  (tangent4 : ∀ x, line x → dist x B = radius4)
  (side1 : ℕ)
  (side2 : ℕ)
  (equal_side : side1 = side2)
  (r1 : ℝ := 2) 
  (r2 : ℝ := 4)
  (r3 : ℝ := 6) 
  (r4 : ℝ := 8) :
  (π * (radius1 * radius1) + π * (radius2 * radius2) + π * (radius3 * radius3) + π * (radius4 * radius4)) = 120 * π := 
sorry

end maximum_area_of_region_l157_157061


namespace scientific_notation_of_12400_l157_157840

theorem scientific_notation_of_12400 :
  12400 = 1.24 * 10^4 :=
sorry

end scientific_notation_of_12400_l157_157840


namespace probability_multiple_of_4_l157_157289

def prob_at_least_one_multiple_of_4 : ℚ :=
  1 - (38/50)^3

theorem probability_multiple_of_4 (n : ℕ) (h : n = 3) : 
  prob_at_least_one_multiple_of_4 = 28051 / 50000 :=
by
  rw [prob_at_least_one_multiple_of_4, ← h]
  sorry

end probability_multiple_of_4_l157_157289


namespace arithmetic_sequence_zero_term_l157_157498

theorem arithmetic_sequence_zero_term (a : ℕ → ℤ) (d : ℤ) (h : d ≠ 0) 
  (h_seq : ∀ n, a n = a 1 + (n-1) * d)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 :=
by { sorry }

end arithmetic_sequence_zero_term_l157_157498


namespace area_of_plot_l157_157449

def central_square_area : ℕ := 64

def common_perimeter : ℕ := 32

-- This statement formalizes the proof problem: "The area of Mrs. Lígia's plot is 256 m² given the provided conditions."
theorem area_of_plot (a b : ℕ) 
  (h1 : a * a = central_square_area)
  (h2 : b = a) 
  (h3 : 4 * a = common_perimeter)  
  (h4 : ∀ (x y : ℕ), x + y = 16)
  (h5 : ∀ (x : ℕ), x + a = 16) 
  : a * 16 = 256 :=
sorry

end area_of_plot_l157_157449


namespace sue_answer_is_106_l157_157174

-- Definitions based on conditions
def ben_step1 (x : ℕ) : ℕ := x * 3
def ben_step2 (x : ℕ) : ℕ := ben_step1 x + 2
def ben_step3 (x : ℕ) : ℕ := ben_step2 x * 2

def sue_step1 (y : ℕ) : ℕ := y + 3
def sue_step2 (y : ℕ) : ℕ := sue_step1 y - 2
def sue_step3 (y : ℕ) : ℕ := sue_step2 y * 2

-- Ben starts with the number 8
def ben_number : ℕ := 8

-- Ben gives the number to Sue
def given_to_sue : ℕ := ben_step3 ben_number

-- Lean statement to prove
theorem sue_answer_is_106 : sue_step3 given_to_sue = 106 :=
by
  sorry

end sue_answer_is_106_l157_157174


namespace total_saltwater_animals_l157_157463

variable (numSaltwaterAquariums : Nat)
variable (animalsPerAquarium : Nat)

theorem total_saltwater_animals (h1 : numSaltwaterAquariums = 22) (h2 : animalsPerAquarium = 46) : 
    numSaltwaterAquariums * animalsPerAquarium = 1012 := 
  by
    sorry

end total_saltwater_animals_l157_157463


namespace determinant_expr_l157_157175

theorem determinant_expr (a b c p q r : ℝ) 
  (h1 : ∀ x, Polynomial.eval x (Polynomial.C a * Polynomial.C b * Polynomial.C c - Polynomial.C p * (Polynomial.C a * Polynomial.C b + Polynomial.C b * Polynomial.C c + Polynomial.C c * Polynomial.C a) + Polynomial.C q * (Polynomial.C a + Polynomial.C b + Polynomial.C c) - Polynomial.C r) = 0) :
  Matrix.det ![
    ![2 + a, 1, 1],
    ![1, 2 + b, 1],
    ![1, 1, 2 + c]
  ] = r + 2*q + 4*p + 4 :=
sorry

end determinant_expr_l157_157175


namespace gasoline_price_percentage_increase_l157_157881

theorem gasoline_price_percentage_increase 
  (price_month1_euros : ℝ) (price_month3_dollars : ℝ) (exchange_rate : ℝ) 
  (price_month1 : ℝ) (percent_increase : ℝ):
  price_month1_euros = 20 →
  price_month3_dollars = 15 →
  exchange_rate = 1.2 →
  price_month1 = price_month1_euros * exchange_rate →
  percent_increase = ((price_month1 - price_month3_dollars) / price_month3_dollars) * 100 →
  percent_increase = 60 :=
by intros; sorry

end gasoline_price_percentage_increase_l157_157881


namespace divisibility_of_n_pow_n_minus_1_l157_157590

theorem divisibility_of_n_pow_n_minus_1 (n : ℕ) (h : n > 1): (n^ (n - 1) - 1) % (n - 1)^2 = 0 := 
  sorry

end divisibility_of_n_pow_n_minus_1_l157_157590


namespace reduced_price_is_3_84_l157_157440

noncomputable def reduced_price_per_dozen (original_price : ℝ) (bananas_for_40 : ℕ) : ℝ := 
  let reduced_price := 0.6 * original_price
  let total_bananas := bananas_for_40 + 50
  let price_per_banana := 40 / total_bananas
  12 * price_per_banana

theorem reduced_price_is_3_84 
  (original_price : ℝ) 
  (bananas_for_40 : ℕ) 
  (h₁ : 40 = bananas_for_40 * original_price) 
  (h₂ : bananas_for_40 = 75) 
    : reduced_price_per_dozen original_price bananas_for_40 = 3.84 :=
sorry

end reduced_price_is_3_84_l157_157440


namespace calculate_total_bricks_l157_157476

-- Given definitions based on the problem.
variables (a d g h : ℕ)

-- Definitions for the questions in terms of variables.
def days_to_build_bricks (a d g : ℕ) : ℕ :=
  (a * g) / d

def total_bricks_with_additional_men (a d g h : ℕ) : ℕ :=
  a + ((d + h) * a) / 2

theorem calculate_total_bricks (a d g h : ℕ)
  (h1 : 0 < d)
  (h2 : 0 < g)
  (h3 : 0 < a) :
  days_to_build_bricks a d g = a * g / d ∧
  total_bricks_with_additional_men a d g h = (3 * a + h * a) / 2 :=
  by sorry

end calculate_total_bricks_l157_157476


namespace inequality_proof_l157_157211

variable (ha la r R : ℝ)
variable (α β γ : ℝ)

-- Conditions
def condition1 : Prop := ha / la = Real.cos ((β - γ) / 2)
def condition2 : Prop := 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2) = 2 * r / R

-- The theorem to be proved
theorem inequality_proof (h1 : condition1 ha la β γ) (h2 : condition2 α β γ r R) :
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (2 * r / R) :=
sorry

end inequality_proof_l157_157211


namespace max_cos_a_l157_157962

theorem max_cos_a (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 :=
by
  -- Proof goes here
  sorry

end max_cos_a_l157_157962


namespace parametric_to_standard_l157_157495

theorem parametric_to_standard (θ : ℝ) (x y : ℝ)
  (h1 : x = 1 + 2 * Real.cos θ)
  (h2 : y = 2 * Real.sin θ) :
  (x - 1)^2 + y^2 = 4 := 
sorry

end parametric_to_standard_l157_157495


namespace sixth_year_fee_l157_157423

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l157_157423


namespace part1_l157_157296

theorem part1 (a b c : ℚ) (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : c^3 = 27) (h4 : a * b < 0) (h5 : b * c > 0) : 
  a * b - b * c + c * a = -33 := by
  sorry

end part1_l157_157296


namespace cubic_sum_l157_157401

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l157_157401


namespace jane_dolls_l157_157245

theorem jane_dolls (jane_dolls jill_dolls : ℕ) (h1 : jane_dolls + jill_dolls = 32) (h2 : jill_dolls = jane_dolls + 6) : jane_dolls = 13 := 
by {
  sorry
}

end jane_dolls_l157_157245


namespace perpendicular_slopes_l157_157224

theorem perpendicular_slopes {m : ℝ} (h : (1 : ℝ) * -m = -1) : m = 1 :=
by sorry

end perpendicular_slopes_l157_157224


namespace solution_to_equation1_solution_to_equation2_l157_157134

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := 3 * x^3 + 4 = -20

-- State the theorems with the correct answers
theorem solution_to_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = -3) :=
by
  sorry

theorem solution_to_equation2 (x : ℝ) : equation2 x ↔ (x = -2) :=
by
  sorry

end solution_to_equation1_solution_to_equation2_l157_157134


namespace binom_10_4_eq_210_l157_157842

theorem binom_10_4_eq_210 : Nat.choose 10 4 = 210 :=
  by sorry

end binom_10_4_eq_210_l157_157842


namespace emily_beads_l157_157358

-- Definitions of the conditions as per step a)
def beads_per_necklace : ℕ := 8
def necklaces : ℕ := 2

-- Theorem statement to prove the equivalent math problem
theorem emily_beads : beads_per_necklace * necklaces = 16 :=
by
  sorry

end emily_beads_l157_157358


namespace find_m_for_parallel_lines_l157_157795

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
by
  sorry

end find_m_for_parallel_lines_l157_157795


namespace bruno_initial_books_l157_157705

theorem bruno_initial_books (X : ℝ)
  (h1 : X - 4.5 + 10.25 = 39.75) :
  X = 34 := by
  sorry

end bruno_initial_books_l157_157705


namespace molecular_weight_of_BaF2_l157_157963

theorem molecular_weight_of_BaF2 (mw_6_moles : ℕ → ℕ) (h : mw_6_moles 6 = 1050) : mw_6_moles 1 = 175 :=
by
  sorry

end molecular_weight_of_BaF2_l157_157963


namespace weight_of_replaced_person_l157_157978

theorem weight_of_replaced_person 
  (avg_increase : ℝ) (new_person_weight : ℝ) (n : ℕ) (original_weight : ℝ) 
  (h1 : avg_increase = 2.5)
  (h2 : new_person_weight = 95)
  (h3 : n = 8)
  (h4 : original_weight = new_person_weight - n * avg_increase) : 
  original_weight = 75 := 
by
  sorry

end weight_of_replaced_person_l157_157978


namespace sum_gcd_lcm_eight_twelve_l157_157972

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l157_157972


namespace shorter_leg_of_right_triangle_l157_157727

theorem shorter_leg_of_right_triangle (a b : ℕ) (h1 : a < b)
    (h2 : a^2 + b^2 = 65^2) : a = 16 :=
sorry

end shorter_leg_of_right_triangle_l157_157727


namespace deepak_age_l157_157201

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l157_157201


namespace initial_percentage_water_l157_157892

theorem initial_percentage_water (W_initial W_final N_initial N_final : ℝ) (h1 : W_initial = 100) 
    (h2 : N_initial = W_initial - W_final) (h3 : W_final = 25) (h4 : W_final / N_final = 0.96) : N_initial / W_initial = 0.99 := 
by
  sorry

end initial_percentage_water_l157_157892


namespace number_of_children_is_five_l157_157299

/-- The sum of the ages of children born at intervals of 2 years each is 50 years, 
    and the age of the youngest child is 6 years.
    Prove that the number of children is 5. -/
theorem number_of_children_is_five (n : ℕ) (h1 : (0 < n ∧ n / 2 * (8 + 2 * n) = 50)): n = 5 :=
sorry

end number_of_children_is_five_l157_157299


namespace solve_system_and_find_6a_plus_b_l157_157808

theorem solve_system_and_find_6a_plus_b (x y a b : ℝ)
  (h1 : 3 * x - 2 * y + 20 = 0)
  (h2 : 2 * x + 15 * y - 3 = 0)
  (h3 : a * x - b * y = 3) :
  6 * a + b = -3 := by
  sorry

end solve_system_and_find_6a_plus_b_l157_157808


namespace vectors_parallel_iff_l157_157997

-- Define the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

-- Define what it means for two vectors to be parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement that we need to prove
theorem vectors_parallel_iff (m : ℝ) : parallel a (b m) ↔ m = 1 := by
  sorry

end vectors_parallel_iff_l157_157997


namespace village_population_l157_157228

variable (Px : ℕ) (t : ℕ) (dX dY : ℕ)
variable (Py : ℕ := 42000) (rateX : ℕ := 1200) (rateY : ℕ := 800) (timeYears : ℕ := 15)

theorem village_population : (Px - rateX * timeYears = Py + rateY * timeYears) → Px = 72000 :=
by
  sorry

end village_population_l157_157228


namespace inequality_proof_l157_157493

variable (a b c : ℝ)

theorem inequality_proof :
  1 < (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end inequality_proof_l157_157493


namespace min_sum_of_gcd_and_lcm_eq_three_times_sum_l157_157877

theorem min_sum_of_gcd_and_lcm_eq_three_times_sum (a b d : ℕ) (h1 : d = Nat.gcd a b)
  (h2 : Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) :
  a + b = 12 :=
by
sorry

end min_sum_of_gcd_and_lcm_eq_three_times_sum_l157_157877


namespace find_f_5_l157_157632

section
variables (f : ℝ → ℝ)

-- Given condition
def functional_equation (x : ℝ) : Prop := x * f x = 2 * f (1 - x) + 1

-- Prove that f(5) = 1/12 given the condition
theorem find_f_5 (h : ∀ x, functional_equation f x) : f 5 = 1 / 12 :=
sorry
end

end find_f_5_l157_157632


namespace problem_statement_l157_157786

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

theorem problem_statement (a x₁ x₂: ℝ) (ha : a > 0) (hx : x₁ < x₂) (hxsum : x₁ + x₂ = 0) :
  f a x₁ < f a x₂ := by
  sorry

end problem_statement_l157_157786


namespace cylinder_cone_surface_area_l157_157693

theorem cylinder_cone_surface_area (r h : ℝ) (π : ℝ) (l : ℝ)
    (h_relation : h = Real.sqrt 3 * r)
    (l_relation : l = 2 * r)
    (cone_lateral_surface_area : π * r * l = 2 * π * r ^ 2) :
    (2 * π * r * h) / (π * r ^ 2) = 2 * Real.sqrt 3 :=
by
    sorry

end cylinder_cone_surface_area_l157_157693


namespace minutes_after_2017_is_0554_l157_157905

theorem minutes_after_2017_is_0554 :
  let initial_time := (20, 17) -- time in hours and minutes
  let total_minutes := 2017
  let hours_passed := total_minutes / 60
  let minutes_passed := total_minutes % 60
  let days_passed := hours_passed / 24
  let remaining_hours := hours_passed % 24
  let resulting_hours := (initial_time.fst + remaining_hours) % 24
  let resulting_minutes := initial_time.snd + minutes_passed
  let final_hours := if resulting_minutes >= 60 then resulting_hours + 1 else resulting_hours
  let final_minutes := if resulting_minutes >= 60 then resulting_minutes - 60 else resulting_minutes
  final_hours % 24 = 5 ∧ final_minutes = 54 := by
  sorry

end minutes_after_2017_is_0554_l157_157905


namespace ordered_pairs_m_n_l157_157864

theorem ordered_pairs_m_n :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ (p.1 ^ 2 - p.2 ^ 2 = 72)) ∧ s.card = 3 :=
by
  sorry

end ordered_pairs_m_n_l157_157864


namespace Nara_is_1_69_meters_l157_157582

-- Define the heights of Sangheon, Chiho, and Nara
def Sangheon_height : ℝ := 1.56
def Chiho_height : ℝ := Sangheon_height - 0.14
def Nara_height : ℝ := Chiho_height + 0.27

-- The statement to be proven
theorem Nara_is_1_69_meters : Nara_height = 1.69 :=
by
  -- the proof goes here
  sorry

end Nara_is_1_69_meters_l157_157582


namespace find_divisor_l157_157736

theorem find_divisor (d q r : ℕ) :
  (919 = d * q + r) → (q = 17) → (r = 11) → d = 53 :=
by
  sorry

end find_divisor_l157_157736


namespace inequality_solution_l157_157898

theorem inequality_solution (x : ℤ) : (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 → x ≥ -5 := 
by
  sorry

end inequality_solution_l157_157898


namespace car_distance_and_velocity_l157_157280

def acceleration : ℝ := 12 -- constant acceleration in m/s^2
def time : ℝ := 36 -- time in seconds
def conversion_factor : ℝ := 3.6 -- conversion factor from m/s to km/h

theorem car_distance_and_velocity :
  (1/2 * acceleration * time^2 = 7776) ∧ (acceleration * time * conversion_factor = 1555.2) :=
by
  sorry

end car_distance_and_velocity_l157_157280


namespace Reema_loan_problem_l157_157295

-- Define problem parameters
def Principal : ℝ := 150000
def Interest : ℝ := 42000
def ProfitRate : ℝ := 0.1
def Profit : ℝ := 25000

-- State the problem as a Lean 4 theorem
theorem Reema_loan_problem (R : ℝ) (Investment : ℝ) : 
  Principal * (R / 100) * R = Interest ∧ 
  Profit = Investment * ProfitRate * R ∧ 
  R = 5 ∧ 
  Investment = 50000 :=
by
  sorry

end Reema_loan_problem_l157_157295


namespace eval_x_power_x_power_x_at_3_l157_157634

theorem eval_x_power_x_power_x_at_3 : (3^3)^(3^3) = 27^27 := by
    sorry

end eval_x_power_x_power_x_at_3_l157_157634


namespace robins_initial_hair_length_l157_157199

variable (L : ℕ)

def initial_length_after_cutting := L - 11
def length_after_growth := initial_length_after_cutting L + 12
def final_length := 17

theorem robins_initial_hair_length : length_after_growth L = final_length → L = 16 := 
by sorry

end robins_initial_hair_length_l157_157199
