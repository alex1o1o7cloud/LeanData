import Mathlib

namespace NUMINAMATH_GPT_cylinder_sphere_surface_area_ratio_l945_94534

theorem cylinder_sphere_surface_area_ratio 
  (d : ℝ) -- d represents the diameter of the sphere and the height of the cylinder
  (S1 S2 : ℝ) -- Surface areas of the cylinder and the sphere
  (r := d / 2) -- radius of the sphere
  (S1 := 6 * π * r ^ 2) -- surface area of the cylinder
  (S2 := 4 * π * r ^ 2) -- surface area of the sphere
  : S1 / S2 = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_cylinder_sphere_surface_area_ratio_l945_94534


namespace NUMINAMATH_GPT_function_form_l945_94500

noncomputable def f : ℕ → ℕ := sorry

theorem function_form (c d a : ℕ) (h1 : c > 1) (h2 : a - c > 1)
  (hf : ∀ n : ℕ, f n + f (n + 1) = f (n + 2) + f (n + 3) - 168) :
  (∀ n : ℕ, f (2 * n) = c + n * d) ∧ (∀ n : ℕ, f (2 * n + 1) = (168 - d) * n + a - c) :=
sorry

end NUMINAMATH_GPT_function_form_l945_94500


namespace NUMINAMATH_GPT_smallest_part_division_l945_94598

theorem smallest_part_division (S : ℚ) (P1 P2 P3 : ℚ) (total : ℚ) :
  (P1, P2, P3) = (1, 2, 3) →
  total = 64 →
  S = total / (P1 + P2 + P3) →
  S = 10 + 2/3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_part_division_l945_94598


namespace NUMINAMATH_GPT_evaluate_trig_expression_l945_94580

theorem evaluate_trig_expression (α : ℝ) (h : Real.tan α = -4/3) : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_trig_expression_l945_94580


namespace NUMINAMATH_GPT_infinitely_many_lovely_no_lovely_square_gt_1_l945_94589

def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ),
    n = (List.ofFn d).prod ∧
    ∀ i, (d i)^2 ∣ n + (d i)

theorem infinitely_many_lovely : ∀ N : ℕ, ∃ n > N, lovely n :=
  sorry

theorem no_lovely_square_gt_1 : ∀ n : ℕ, n > 1 → lovely n → ¬∃ m, n = m^2 :=
  sorry

end NUMINAMATH_GPT_infinitely_many_lovely_no_lovely_square_gt_1_l945_94589


namespace NUMINAMATH_GPT_sum_a5_a6_a7_l945_94571

def S (n : ℕ) : ℕ :=
  n^2 + 2 * n + 5

theorem sum_a5_a6_a7 : S 7 - S 4 = 39 :=
  by sorry

end NUMINAMATH_GPT_sum_a5_a6_a7_l945_94571


namespace NUMINAMATH_GPT_Clinton_belts_l945_94585

variable {Shoes Belts Hats : ℕ}

theorem Clinton_belts :
  (Shoes = 14) → (Shoes = 2 * Belts) → Belts = 7 :=
by
  sorry

end NUMINAMATH_GPT_Clinton_belts_l945_94585


namespace NUMINAMATH_GPT_chess_tournament_games_l945_94568

theorem chess_tournament_games (P : ℕ) (TotalGames : ℕ) (hP : P = 21) (hTotalGames : TotalGames = 210) : 
  ∃ G : ℕ, G = 20 ∧ TotalGames = (P * (P - 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l945_94568


namespace NUMINAMATH_GPT_intersection_M_N_l945_94552

def set_M : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def set_N : Set ℝ := { x : ℝ | x^2 - 2 * x - 8 ≤ 0 }

theorem intersection_M_N : (set_M ∩ set_N) = { x : ℝ | -2 ≤ x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l945_94552


namespace NUMINAMATH_GPT_intersection_A_B_l945_94541

def A := {x : ℝ | x^2 - ⌊x⌋ = 2}
def B := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l945_94541


namespace NUMINAMATH_GPT_necklace_cost_l945_94501

theorem necklace_cost (N : ℕ) (h1 : N + (N + 5) = 73) : N = 34 := by
  sorry

end NUMINAMATH_GPT_necklace_cost_l945_94501


namespace NUMINAMATH_GPT_remainder_of_x_mod_10_l945_94567

def x : ℕ := 2007 ^ 2008

theorem remainder_of_x_mod_10 : x % 10 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_x_mod_10_l945_94567


namespace NUMINAMATH_GPT_length_AC_l945_94532

theorem length_AC {AB BC : ℝ} (h1: AB = 6) (h2: BC = 4) : (AC = 2 ∨ AC = 10) :=
sorry

end NUMINAMATH_GPT_length_AC_l945_94532


namespace NUMINAMATH_GPT_ellipse_major_minor_axis_condition_l945_94597

theorem ellipse_major_minor_axis_condition (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1) 
                                          (h2 : ∀ a b : ℝ, a = 2 * b) :
  m = 1 / 4 :=
sorry

end NUMINAMATH_GPT_ellipse_major_minor_axis_condition_l945_94597


namespace NUMINAMATH_GPT_train_speed_l945_94533

theorem train_speed (L : ℝ) (T : ℝ) (L_pos : 0 < L) (T_pos : 0 < T) (L_eq : L = 150) (T_eq : T = 3) : L / T = 50 := by
  sorry

end NUMINAMATH_GPT_train_speed_l945_94533


namespace NUMINAMATH_GPT_sequence_sum_l945_94508

-- Definitions for the sequences
def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

-- The theorem we need to prove
theorem sequence_sum : a (b 1) + a (b 2) + a (b 3) + a (b 4) = 19 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_l945_94508


namespace NUMINAMATH_GPT_x_intercept_of_line_l945_94570

theorem x_intercept_of_line (x y : ℚ) (h : 4 * x + 6 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l945_94570


namespace NUMINAMATH_GPT_sum_of_numbers_equal_16_l945_94565

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_equal_16_l945_94565


namespace NUMINAMATH_GPT_find_shares_l945_94538

def shareA (B : ℝ) : ℝ := 3 * B
def shareC (B : ℝ) : ℝ := B - 25
def shareD (A B : ℝ) : ℝ := A + B - 10
def total_share (A B C D : ℝ) : ℝ := A + B + C + D

theorem find_shares :
  ∃ (A B C D : ℝ),
  A = 744.99 ∧
  B = 248.33 ∧
  C = 223.33 ∧
  D = 983.32 ∧
  A = shareA B ∧
  C = shareC B ∧
  D = shareD A B ∧
  total_share A B C D = 2200 := 
sorry

end NUMINAMATH_GPT_find_shares_l945_94538


namespace NUMINAMATH_GPT_sum_of_terms_in_sequence_is_215_l945_94558

theorem sum_of_terms_in_sequence_is_215 (a d : ℕ) (h1: Nat.Prime a) (h2: Nat.Prime d)
  (hAP : a + 50 = a + 50)
  (hGP : (a + d) * (a + 50) = (a + 2 * d) ^ 2) :
  (a + (a + d) + (a + 2 * d) + (a + 50)) = 215 := sorry

end NUMINAMATH_GPT_sum_of_terms_in_sequence_is_215_l945_94558


namespace NUMINAMATH_GPT_distance_from_pole_to_line_l945_94545

/-- Definition of the line in polar coordinates -/
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of the pole in Cartesian coordinates -/
def pole_cartesian : ℝ × ℝ := (0, 0)

/-- Convert the line from polar to Cartesian -/
def line_cartesian (x y : ℝ) : Prop := x = 2

/-- The distance function between a point and a line in Cartesian coordinates -/
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (p.1 - 2)

/-- Prove that the distance from the pole to the line is 2 -/
theorem distance_from_pole_to_line : distance_to_line pole_cartesian = 2 := by
  sorry

end NUMINAMATH_GPT_distance_from_pole_to_line_l945_94545


namespace NUMINAMATH_GPT_ali_spending_ratio_l945_94582

theorem ali_spending_ratio
  (initial_amount : ℝ := 480)
  (remaining_amount : ℝ := 160)
  (F : ℝ)
  (H1 : (initial_amount - F - (1/3) * (initial_amount - F) = remaining_amount))
  : (F / initial_amount) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ali_spending_ratio_l945_94582


namespace NUMINAMATH_GPT_geometric_common_ratio_l945_94515

theorem geometric_common_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (a₁ * (1 - q ^ 3)) / (1 - q) / ((a₁ * (1 - q ^ 2)) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_l945_94515


namespace NUMINAMATH_GPT_military_unit_soldiers_l945_94590

theorem military_unit_soldiers:
  ∃ (x N : ℕ), 
      (N = x * (x + 5)) ∧
      (N = 5 * (x + 845)) ∧
      N = 4550 :=
by
  sorry

end NUMINAMATH_GPT_military_unit_soldiers_l945_94590


namespace NUMINAMATH_GPT_minimize_sum_of_squares_l945_94593

theorem minimize_sum_of_squares (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 16) :
  a^2 + b^2 + c^2 ≥ 86 :=
sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_l945_94593


namespace NUMINAMATH_GPT_two_vertical_asymptotes_l945_94527

theorem two_vertical_asymptotes (k : ℝ) : 
  (∀ x : ℝ, (x ≠ 3 ∧ x ≠ -2) → 
           (∃ δ > 0, ∀ ε > 0, ∃ x' : ℝ, x + δ > x' ∧ x' > x - δ ∧ 
                             (x' ≠ 3 ∧ x' ≠ -2) → 
                             |(x'^2 + 2 * x' + k) / (x'^2 - x' - 6)| > 1/ε)) ↔ 
  (k ≠ -15 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_GPT_two_vertical_asymptotes_l945_94527


namespace NUMINAMATH_GPT_solve_rational_eq_l945_94559

theorem solve_rational_eq {x : ℝ} (h : 1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 4 * x - 5) + 1 / (x^2 - 15 * x - 12) = 0) :
  x = 3 ∨ x = -4 ∨ x = 1 ∨ x = -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_rational_eq_l945_94559


namespace NUMINAMATH_GPT_package_weights_l945_94584

theorem package_weights (a b c : ℕ) 
  (h1 : a + b = 108) 
  (h2 : b + c = 132) 
  (h3 : c + a = 138) 
  (h4 : a ≥ 40) 
  (h5 : b ≥ 40) 
  (h6 : c ≥ 40) : 
  a + b + c = 189 :=
sorry

end NUMINAMATH_GPT_package_weights_l945_94584


namespace NUMINAMATH_GPT_range_of_a_l945_94549

noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a x : ℝ) := a * x + Real.cos x

theorem range_of_a :
  (∀ x : ℝ, ∃ y : ℝ, (g a y - g a y) / (y - y) * ((f x - f x) / (x - x)) = -1) →
  (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l945_94549


namespace NUMINAMATH_GPT_prime_only_one_solution_l945_94542

theorem prime_only_one_solution (p : ℕ) (hp : Nat.Prime p) : 
  (∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2) → p = 3 := 
by 
  sorry

end NUMINAMATH_GPT_prime_only_one_solution_l945_94542


namespace NUMINAMATH_GPT_parrots_per_cage_l945_94530

theorem parrots_per_cage (P : ℕ) (total_birds total_cages parakeets_per_cage : ℕ)
  (h1 : total_cages = 4)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 40)
  (h4 : total_birds = total_cages * (P + parakeets_per_cage)) :
  P = 8 :=
by
  sorry

end NUMINAMATH_GPT_parrots_per_cage_l945_94530


namespace NUMINAMATH_GPT_min_eq_floor_sqrt_l945_94516

theorem min_eq_floor_sqrt (n : ℕ) (h : n > 0) : 
  (∀ k : ℕ, k > 0 → (k + n / k) ≥ ⌊(Real.sqrt (4 * n + 1))⌋) := 
sorry

end NUMINAMATH_GPT_min_eq_floor_sqrt_l945_94516


namespace NUMINAMATH_GPT_number_of_multiples_of_six_ending_in_four_and_less_than_800_l945_94574

-- Definitions from conditions
def is_multiple_of_six (n : ℕ) : Prop := n % 6 = 0
def ends_with_four (n : ℕ) : Prop := n % 10 = 4
def less_than_800 (n : ℕ) : Prop := n < 800

-- Theorem to prove
theorem number_of_multiples_of_six_ending_in_four_and_less_than_800 :
  ∃ k : ℕ, k = 26 ∧ ∀ n : ℕ, (is_multiple_of_six n ∧ ends_with_four n ∧ less_than_800 n) → n = 24 + 60 * k ∨ n = 54 + 60 * k :=
sorry

end NUMINAMATH_GPT_number_of_multiples_of_six_ending_in_four_and_less_than_800_l945_94574


namespace NUMINAMATH_GPT_probability_exactly_three_primes_l945_94513

noncomputable def prime_faces : Finset ℕ := {2, 3, 5, 7, 11}

def num_faces : ℕ := 12
def num_dice : ℕ := 7
def target_primes : ℕ := 3

def probability_three_primes : ℚ :=
  35 * ((5 / 12)^3 * (7 / 12)^4)

theorem probability_exactly_three_primes :
  probability_three_primes = (4375 / 51821766) :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_three_primes_l945_94513


namespace NUMINAMATH_GPT_rachel_lunch_problems_l945_94557

theorem rachel_lunch_problems (problems_per_minute minutes_before_bed total_problems : ℕ) 
    (h1 : problems_per_minute = 5)
    (h2 : minutes_before_bed = 12)
    (h3 : total_problems = 76) : 
    (total_problems - problems_per_minute * minutes_before_bed) = 16 :=
by
    sorry

end NUMINAMATH_GPT_rachel_lunch_problems_l945_94557


namespace NUMINAMATH_GPT_find_alpha_l945_94595

theorem find_alpha (α β : ℝ) (h1 : Real.arctan α = 1/2) (h2 : Real.arctan (α - β) = 1/3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α = π/4 := by
  sorry

end NUMINAMATH_GPT_find_alpha_l945_94595


namespace NUMINAMATH_GPT_find_real_parts_l945_94506

theorem find_real_parts (a b : ℝ) (i : ℂ) (hi : i*i = -1) 
(h : a + b*i = (1 - i) * i) : a = 1 ∧ b = -1 :=
sorry

end NUMINAMATH_GPT_find_real_parts_l945_94506


namespace NUMINAMATH_GPT_lemonade_calories_l945_94572

theorem lemonade_calories 
    (lime_juice_weight : ℕ)
    (lime_juice_calories_per_grams : ℕ)
    (sugar_weight : ℕ)
    (sugar_calories_per_grams : ℕ)
    (water_weight : ℕ)
    (water_calories_per_grams : ℕ)
    (mint_weight : ℕ)
    (mint_calories_per_grams : ℕ)
    :
    lime_juice_weight = 150 →
    lime_juice_calories_per_grams = 30 →
    sugar_weight = 200 →
    sugar_calories_per_grams = 390 →
    water_weight = 500 →
    water_calories_per_grams = 0 →
    mint_weight = 50 →
    mint_calories_per_grams = 7 →
    (300 * ((150 * 30 + 200 * 390 + 500 * 0 + 50 * 7) / 900) = 276) :=
by
  sorry

end NUMINAMATH_GPT_lemonade_calories_l945_94572


namespace NUMINAMATH_GPT_gcd_12012_18018_l945_94503

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end NUMINAMATH_GPT_gcd_12012_18018_l945_94503


namespace NUMINAMATH_GPT_baseball_card_decrease_l945_94520

theorem baseball_card_decrease (V₀ : ℝ) (V₁ V₂ : ℝ)
  (h₁: V₁ = V₀ * (1 - 0.20))
  (h₂: V₂ = V₁ * (1 - 0.20)) :
  ((V₀ - V₂) / V₀) * 100 = 36 :=
by
  sorry

end NUMINAMATH_GPT_baseball_card_decrease_l945_94520


namespace NUMINAMATH_GPT_find_x_y_l945_94547

theorem find_x_y (x y : ℝ) (h1 : (10 + 25 + x + y) / 4 = 20) (h2 : x * y = 156) :
  (x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12) :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l945_94547


namespace NUMINAMATH_GPT_variance_is_4_l945_94592

variable {datapoints : List ℝ}

noncomputable def variance (datapoints : List ℝ) : ℝ :=
  let n := datapoints.length
  let mean := (datapoints.sum / n : ℝ)
  (1 / n : ℝ) * ((datapoints.map (λ x => x ^ 2)).sum - n * mean ^ 2)

theorem variance_is_4 :
  (datapoints.length = 20)
  → ((datapoints.map (λ x => x ^ 2)).sum = 800)
  → (datapoints.sum / 20 = 6)
  → variance datapoints = 4 := by
  intros length_cond sum_squares_cond mean_cond
  sorry

end NUMINAMATH_GPT_variance_is_4_l945_94592


namespace NUMINAMATH_GPT_intersection_complement_M_N_l945_94525

def M := { x : ℝ | x ≤ 1 / 2 }
def N := { x : ℝ | x^2 ≤ 1 }
def complement_M := { x : ℝ | x > 1 / 2 }

theorem intersection_complement_M_N :
  (complement_M ∩ N = { x : ℝ | 1 / 2 < x ∧ x ≤ 1 }) :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_M_N_l945_94525


namespace NUMINAMATH_GPT_original_fund_was_830_l945_94519

/- Define the number of employees as a variable -/
variables (n : ℕ)

/- Define the conditions given in the problem -/
def initial_fund := 60 * n - 10
def new_fund_after_distributing_50 := initial_fund - 50 * n
def remaining_fund := 130

/- State the proof goal -/
theorem original_fund_was_830 :
  initial_fund = 830 :=
by sorry

end NUMINAMATH_GPT_original_fund_was_830_l945_94519


namespace NUMINAMATH_GPT_smallest_n_l945_94560

/-- The smallest value of n > 20 that satisfies
    n ≡ 4 [MOD 6]
    n ≡ 3 [MOD 7]
    n ≡ 5 [MOD 8] is 220. -/
theorem smallest_n (n : ℕ) : 
  (n > 20) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (n % 8 = 5) ↔ (n = 220) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_n_l945_94560


namespace NUMINAMATH_GPT_find_fraction_increase_l945_94578

noncomputable def present_value : ℝ := 64000
noncomputable def value_after_two_years : ℝ := 87111.11111111112

theorem find_fraction_increase (f : ℝ) :
  64000 * (1 + f) ^ 2 = 87111.11111111112 → f = 0.1666666666666667 := 
by
  intro h
  -- proof steps here
  sorry

end NUMINAMATH_GPT_find_fraction_increase_l945_94578


namespace NUMINAMATH_GPT_distance_probability_at_least_sqrt2_over_2_l945_94566

noncomputable def prob_dist_at_least : ℝ := 
  let T := ((0,0), (1,0), (0,1))
  -- Assumes conditions incorporated through identifying two random points within the triangle T.
  let area_T : ℝ := 0.5
  let valid_area : ℝ := 0.5 - (Real.pi * (Real.sqrt 2 / 2)^2 / 8 + ((Real.sqrt 2 / 2)^2 / 2) / 2)
  valid_area / area_T

theorem distance_probability_at_least_sqrt2_over_2 :
  prob_dist_at_least = (4 - π) / 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_probability_at_least_sqrt2_over_2_l945_94566


namespace NUMINAMATH_GPT_quadratic_function_fixed_points_range_l945_94510

def has_two_distinct_fixed_points (c : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
               (x1 = x1^2 - x1 + c) ∧ 
               (x2 = x2^2 - x2 + c) ∧ 
               x1 < 2 ∧ 2 < x2

theorem quadratic_function_fixed_points_range (c : ℝ) :
  has_two_distinct_fixed_points c ↔ c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_function_fixed_points_range_l945_94510


namespace NUMINAMATH_GPT_anika_sequence_correct_l945_94529

noncomputable def anika_sequence : ℚ :=
  let s0 := 1458
  let s1 := s0 * 3
  let s2 := s1 / 2
  let s3 := s2 * 3
  let s4 := s3 / 2
  let s5 := s4 * 3
  s5

theorem anika_sequence_correct :
  anika_sequence = (3^9 : ℚ) / 2 := by
  sorry

end NUMINAMATH_GPT_anika_sequence_correct_l945_94529


namespace NUMINAMATH_GPT_alice_lawn_area_l945_94551

theorem alice_lawn_area (posts : ℕ) (distance : ℕ) (ratio : ℕ) : 
    posts = 24 → distance = 5 → ratio = 3 → 
    ∃ (short_side long_side : ℕ), 
        (2 * (short_side + long_side - 2) = posts) ∧
        (long_side = ratio * short_side) ∧
        (distance * (short_side - 1) * distance * (long_side - 1) = 825) :=
by
  intros h_posts h_distance h_ratio
  sorry

end NUMINAMATH_GPT_alice_lawn_area_l945_94551


namespace NUMINAMATH_GPT_triangle_condition_proof_l945_94505

variables {A B C D M K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace K]
variables (AB AC AD : ℝ)

-- Definitions based on the conditions
def is_isosceles (A B C : Type*) (AB AC : ℝ) : Prop :=
  AB = AC

def is_altitude (A D B C : Type*) : Prop :=
  true -- Ideally, this condition is more complex and involves perpendicular projection

def is_midpoint (M A D : Type*) : Prop :=
  true -- Ideally, this condition is more specific and involves equality of segments

def extends_to (C M A B K : Type*) : Prop :=
  true -- Represents the extension relationship

-- The theorem to be proved
theorem triangle_condition_proof (A B C D M K : Type*)
  (h_iso : is_isosceles A B C AB AC)
  (h_alt : is_altitude A D B C)
  (h_mid : is_midpoint M A D)
  (h_ext : extends_to C M A B K)
  : AB = 3 * AK :=
  sorry

end NUMINAMATH_GPT_triangle_condition_proof_l945_94505


namespace NUMINAMATH_GPT_triangle_side_length_l945_94575

-- Definitions based on problem conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variables (AC BC AD AB CD : ℝ)

-- Conditions from the problem
axiom h1 : BC = 2 * AC
axiom h2 : AD = (1 / 3) * AB

-- Theorem statement to be proved
theorem triangle_side_length (h1 : BC = 2 * AC) (h2 : AD = (1 / 3) * AB) : CD = 2 * AD :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l945_94575


namespace NUMINAMATH_GPT_remainder_when_divided_by_2_l945_94588

-- Define the main parameters
def n : ℕ := sorry  -- n is a positive integer
def k : ℤ := sorry  -- Provided for modular arithmetic context

-- Conditions
axiom h1 : n > 0  -- n is a positive integer
axiom h2 : (n + 1) % 6 = 4  -- When n + 1 is divided by 6, the remainder is 4

-- The theorem statement
theorem remainder_when_divided_by_2 : n % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_2_l945_94588


namespace NUMINAMATH_GPT_RSA_next_challenge_digits_l945_94512

theorem RSA_next_challenge_digits (previous_digits : ℕ) (prize_increase : ℕ) :
  previous_digits = 193 ∧ prize_increase > 10000 → ∃ N : ℕ, N = 212 :=
by {
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_RSA_next_challenge_digits_l945_94512


namespace NUMINAMATH_GPT_find_m_value_l945_94561

def magic_box_output (a b : ℝ) : ℝ := a^2 + b - 1

theorem find_m_value :
  ∃ m : ℝ, (magic_box_output m (-2 * m) = 2) ↔ (m = 3 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l945_94561


namespace NUMINAMATH_GPT_simplify_t_l945_94528

theorem simplify_t (t : ℝ) (cbrt3 : ℝ) (h : cbrt3 ^ 3 = 3) 
  (ht : t = 1 / (1 - cbrt3)) : 
  t = - (1 + cbrt3 + cbrt3 ^ 2) / 2 := 
sorry

end NUMINAMATH_GPT_simplify_t_l945_94528


namespace NUMINAMATH_GPT_triangle_side_lengths_l945_94540

theorem triangle_side_lengths 
  (r : ℝ) (CD : ℝ) (DB : ℝ) 
  (h_r : r = 4) 
  (h_CD : CD = 8) 
  (h_DB : DB = 10) :
  ∃ (AB AC : ℝ), AB = 14.5 ∧ AC = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l945_94540


namespace NUMINAMATH_GPT_max_area_rectangular_playground_l945_94577

theorem max_area_rectangular_playground (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 360) 
  (h_length : l ≥ 90) 
  (h_width : w ≥ 50) : 
  (l * w) ≤ 8100 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangular_playground_l945_94577


namespace NUMINAMATH_GPT_sets_relationship_l945_94550

def M : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 3 * k - 2}
def P : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def S : Set ℤ := {x : ℤ | ∃ m : ℤ, x = 6 * m + 1}

theorem sets_relationship : S ⊆ P ∧ M = P := by
  sorry

end NUMINAMATH_GPT_sets_relationship_l945_94550


namespace NUMINAMATH_GPT_problem_1_problem_2_l945_94553

open Set Real

-- Definition of the sets A, B, and the complement of B in the real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Proof problem (1): Prove that A ∩ (complement of B) = [1, 2]
theorem problem_1 : (A ∩ (compl B)) = {x | 1 ≤ x ∧ x ≤ 2} := sorry

-- Proof problem (2): Prove that the set of values for the real number a such that C(a) ∩ A = C(a)
-- is (-∞, 3]
theorem problem_2 : { a : ℝ | C a ⊆ A } = { a : ℝ | a ≤ 3 } := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l945_94553


namespace NUMINAMATH_GPT_color_block_prob_l945_94543

-- Definitions of the problem's conditions
def colors : List (List String) := [
    ["red", "blue", "yellow", "green"],
    ["red", "blue", "yellow", "white"]
]

-- The events in which at least one box receives 3 blocks of the same color
def event_prob : ℚ := 3 / 64

-- Tuple as a statement to prove in Lean
theorem color_block_prob (m n : ℕ) (h : m + n = 67) : 
  ∃ (m n : ℕ), (m / n : ℚ) = event_prob := 
by
  use 3
  use 64
  simp
  sorry

end NUMINAMATH_GPT_color_block_prob_l945_94543


namespace NUMINAMATH_GPT_pentagon_area_l945_94576

theorem pentagon_area (a b c d e : ℝ)
  (ht_base ht_height : ℝ)
  (trap_base1 trap_base2 trap_height : ℝ)
  (side_a : a = 17)
  (side_b : b = 22)
  (side_c : c = 30)
  (side_d : d = 26)
  (side_e : e = 22)
  (rt_height : ht_height = 17)
  (rt_base : ht_base = 22)
  (trap_base1_eq : trap_base1 = 26)
  (trap_base2_eq : trap_base2 = 30)
  (trap_height_eq : trap_height = 22)
  : 1/2 * ht_base * ht_height + 1/2 * (trap_base1 + trap_base2) * trap_height = 803 :=
by sorry

end NUMINAMATH_GPT_pentagon_area_l945_94576


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_l945_94511

theorem geometric_sequence_arithmetic (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h2 : 2 * S 6 = S 3 + S 9) : 
  q^3 = -1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_l945_94511


namespace NUMINAMATH_GPT_Mary_chestnuts_l945_94562

noncomputable def MaryPickedTwicePeter (P M : ℕ) := M = 2 * P
noncomputable def LucyPickedMorePeter (P L : ℕ) := L = P + 2
noncomputable def TotalPicked (P M L : ℕ) := P + M + L = 26

theorem Mary_chestnuts (P M L : ℕ) (h1 : MaryPickedTwicePeter P M) (h2 : LucyPickedMorePeter P L) (h3 : TotalPicked P M L) :
  M = 12 :=
sorry

end NUMINAMATH_GPT_Mary_chestnuts_l945_94562


namespace NUMINAMATH_GPT_statement_B_statement_C_statement_D_l945_94504

-- Statement B
theorem statement_B (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a^3 * c < b^3 * c :=
sorry

-- Statement C
theorem statement_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : (a / (c - a)) > (b / (c - b)) :=
sorry

-- Statement D
theorem statement_D (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 :=
sorry

end NUMINAMATH_GPT_statement_B_statement_C_statement_D_l945_94504


namespace NUMINAMATH_GPT_range_x_y_l945_94583

variable (x y : ℝ)

theorem range_x_y (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) : 
  27 < x - y ∧ x - y < 56 :=
sorry

end NUMINAMATH_GPT_range_x_y_l945_94583


namespace NUMINAMATH_GPT_cylinder_surface_area_l945_94535

namespace SurfaceAreaProof

variables (a b : ℝ)

theorem cylinder_surface_area (a b : ℝ) :
  (2 * Real.pi * a * b) = (2 * Real.pi * a * b) :=
by sorry

end SurfaceAreaProof

end NUMINAMATH_GPT_cylinder_surface_area_l945_94535


namespace NUMINAMATH_GPT_find_a_l945_94507

theorem find_a (x a a1 a2 a3 a4 : ℝ) :
  (x + a) ^ 4 = x ^ 4 + a1 * x ^ 3 + a2 * x ^ 2 + a3 * x + a4 → 
  a1 + a2 + a3 = 64 → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l945_94507


namespace NUMINAMATH_GPT_toys_sold_in_first_week_l945_94524

/-
  Problem statement:
  An online toy store stocked some toys. It sold some toys at the first week and 26 toys at the second week.
  If it had 19 toys left and there were 83 toys in stock at the beginning, how many toys were sold in the first week?
-/

theorem toys_sold_in_first_week (initial_stock toys_left toys_sold_second_week : ℕ) 
  (h_initial_stock : initial_stock = 83) 
  (h_toys_left : toys_left = 19) 
  (h_toys_sold_second_week : toys_sold_second_week = 26) : 
  (initial_stock - toys_left - toys_sold_second_week) = 38 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_toys_sold_in_first_week_l945_94524


namespace NUMINAMATH_GPT_apollo_total_cost_l945_94526

def hephaestus_first_half_months : ℕ := 6
def hephaestus_first_half_rate : ℕ := 3
def hephaestus_second_half_rate : ℕ := hephaestus_first_half_rate * 2

def athena_rate : ℕ := 5
def athena_months : ℕ := 12

def ares_first_period_months : ℕ := 9
def ares_first_period_rate : ℕ := 4
def ares_second_period_months : ℕ := 3
def ares_second_period_rate : ℕ := 6

def total_cost := hephaestus_first_half_months * hephaestus_first_half_rate
               + hephaestus_first_half_months * hephaestus_second_half_rate
               + athena_months * athena_rate
               + ares_first_period_months * ares_first_period_rate
               + ares_second_period_months * ares_second_period_rate

theorem apollo_total_cost : total_cost = 168 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_apollo_total_cost_l945_94526


namespace NUMINAMATH_GPT_susan_change_sum_susan_possible_sums_l945_94569

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

end NUMINAMATH_GPT_susan_change_sum_susan_possible_sums_l945_94569


namespace NUMINAMATH_GPT_problem_statement_l945_94521

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l945_94521


namespace NUMINAMATH_GPT_percent_decrease_call_cost_l945_94514

theorem percent_decrease_call_cost (c1990 c2010 : ℝ) (h1990 : c1990 = 50) (h2010 : c2010 = 10) :
  ((c1990 - c2010) / c1990) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_call_cost_l945_94514


namespace NUMINAMATH_GPT_value_of_a_l945_94548

theorem value_of_a (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {a, a^2}) (hB : B = {1, b}) (hAB : A = B) : a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l945_94548


namespace NUMINAMATH_GPT_appropriate_chart_for_milk_powder_l945_94573

-- Define the chart requirements and the correctness condition
def ChartType := String
def pie : ChartType := "pie"
def line : ChartType := "line"
def bar : ChartType := "bar"

-- The condition we need for our proof
def representsPercentagesWell (chart: ChartType) : Prop :=
  chart = pie

-- The main theorem statement
theorem appropriate_chart_for_milk_powder : representsPercentagesWell pie :=
by
  sorry

end NUMINAMATH_GPT_appropriate_chart_for_milk_powder_l945_94573


namespace NUMINAMATH_GPT_find_number_l945_94594

theorem find_number (x : ℝ) : 
  ( ((x - 1.9) * 1.5 + 32) / 2.5 = 20 ) → x = 13.9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l945_94594


namespace NUMINAMATH_GPT_length_more_than_breadth_by_200_l945_94537

-- Definitions and conditions
def rectangular_floor_length := 23
def painting_cost := 529
def painting_rate := 3
def floor_area := painting_cost / painting_rate
def floor_breadth := floor_area / rectangular_floor_length

-- Prove that the length is more than the breadth by 200%
theorem length_more_than_breadth_by_200 : 
  rectangular_floor_length = floor_breadth * (1 + 200 / 100) :=
sorry

end NUMINAMATH_GPT_length_more_than_breadth_by_200_l945_94537


namespace NUMINAMATH_GPT_mary_sailboat_canvas_l945_94555

def rectangular_sail_area (length width : ℕ) : ℕ :=
  length * width

def triangular_sail_area (base height : ℕ) : ℕ :=
  (base * height) / 2

def total_canvas_area (length₁ width₁ base₁ height₁ base₂ height₂ : ℕ) : ℕ :=
  rectangular_sail_area length₁ width₁ +
  triangular_sail_area base₁ height₁ +
  triangular_sail_area base₂ height₂

theorem mary_sailboat_canvas :
  total_canvas_area 5 8 3 4 4 6 = 58 :=
by
  -- Begin proof (proof steps omitted, we just need the structure here)
  sorry -- end proof

end NUMINAMATH_GPT_mary_sailboat_canvas_l945_94555


namespace NUMINAMATH_GPT_find_red_cards_l945_94586

-- We use noncomputable here as we are dealing with real numbers in a theoretical proof context.
noncomputable def red_cards (r b : ℕ) (_initial_prob : r / (r + b) = 1 / 5) 
                            (_added_prob : r / (r + b + 6) = 1 / 7) : ℕ := 
r

theorem find_red_cards 
  {r b : ℕ}
  (h1 : r / (r + b) = 1 / 5)
  (h2 : r / (r + b + 6) = 1 / 7) : 
  red_cards r b h1 h2 = 3 :=
sorry  -- Proof not required

end NUMINAMATH_GPT_find_red_cards_l945_94586


namespace NUMINAMATH_GPT_PlanY_more_cost_effective_l945_94554

-- Define the gigabytes Tim uses
variable (y : ℕ)

-- Define the cost functions for Plan X and Plan Y in cents
def cost_PlanX (y : ℕ) := 25 * y
def cost_PlanY (y : ℕ) := 1500 + 15 * y

-- Prove that Plan Y is cheaper than Plan X when y >= 150
theorem PlanY_more_cost_effective (y : ℕ) : y ≥ 150 → cost_PlanY y < cost_PlanX y := by
  sorry

end NUMINAMATH_GPT_PlanY_more_cost_effective_l945_94554


namespace NUMINAMATH_GPT_min_value_of_f_l945_94599

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x - 3 / x

theorem min_value_of_f : ∃ x < 0, ∀ y : ℝ, y = f x → y ≥ 1 + 2 * Real.sqrt 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end NUMINAMATH_GPT_min_value_of_f_l945_94599


namespace NUMINAMATH_GPT_fraction_spent_l945_94531

theorem fraction_spent (borrowed_from_brother borrowed_from_father borrowed_from_mother gift_from_granny savings remaining amount_spent : ℕ)
  (h_borrowed_from_brother : borrowed_from_brother = 20)
  (h_borrowed_from_father : borrowed_from_father = 40)
  (h_borrowed_from_mother : borrowed_from_mother = 30)
  (h_gift_from_granny : gift_from_granny = 70)
  (h_savings : savings = 100)
  (h_remaining : remaining = 65)
  (h_amount_spent : amount_spent = borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings - remaining) :
  (amount_spent : ℚ) / (borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_l945_94531


namespace NUMINAMATH_GPT_three_digit_non_multiples_of_3_or_11_l945_94563

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end NUMINAMATH_GPT_three_digit_non_multiples_of_3_or_11_l945_94563


namespace NUMINAMATH_GPT_volume_of_pond_rect_prism_l945_94517

-- Define the problem as a proposition
theorem volume_of_pond_rect_prism :
  let l := 28
  let w := 10
  let h := 5
  V = l * w * h →
  V = 1400 :=
by
  intros l w h h1
  -- Here, the theorem states the equivalence of the volume given the defined length, width, and height being equal to 1400 cubic meters.
  have : V = 28 * 10 * 5 := by sorry
  exact this

end NUMINAMATH_GPT_volume_of_pond_rect_prism_l945_94517


namespace NUMINAMATH_GPT_problem_inequality_l945_94587

variable {a b : ℝ}

theorem problem_inequality 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_a_gt_b : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) := 
by 
  sorry

end NUMINAMATH_GPT_problem_inequality_l945_94587


namespace NUMINAMATH_GPT_Yvettes_final_bill_l945_94522

namespace IceCreamShop

def sundae_price_Alicia : Real := 7.50
def sundae_price_Brant : Real := 10.00
def sundae_price_Josh : Real := 8.50
def sundae_price_Yvette : Real := 9.00
def tip_rate : Real := 0.20

theorem Yvettes_final_bill :
  let total_cost := sundae_price_Alicia + sundae_price_Brant + sundae_price_Josh + sundae_price_Yvette
  let tip := tip_rate * total_cost
  let final_bill := total_cost + tip
  final_bill = 42.00 :=
by
  -- calculations are skipped here
  sorry

end IceCreamShop

end NUMINAMATH_GPT_Yvettes_final_bill_l945_94522


namespace NUMINAMATH_GPT_Kyler_wins_l945_94556

variable (K : ℕ) -- Kyler's wins

/- Constants based on the problem statement -/
def Peter_wins := 5
def Peter_losses := 3
def Emma_wins := 2
def Emma_losses := 4
def Total_games := 15
def Kyler_losses := 4

/- Definition that calculates total games played -/
def total_games_played := 2 * Total_games

/- Game equation based on the total count of played games -/
def game_equation := Peter_wins + Peter_losses + Emma_wins + Emma_losses + K + Kyler_losses = total_games_played

/- Question: Calculate Kyler's wins assuming the given conditions -/
theorem Kyler_wins : K = 1 :=
by
  sorry

end NUMINAMATH_GPT_Kyler_wins_l945_94556


namespace NUMINAMATH_GPT_reduced_price_l945_94523

theorem reduced_price (P R : ℝ) (Q : ℝ) (h₁ : R = 0.80 * P) 
                      (h₂ : 800 = Q * P) 
                      (h₃ : 800 = (Q + 5) * R) 
                      : R = 32 :=
by
  -- Code that proves the theorem goes here.
  sorry

end NUMINAMATH_GPT_reduced_price_l945_94523


namespace NUMINAMATH_GPT_ratio_of_a_b_l945_94502

variable (x y a b : ℝ)

theorem ratio_of_a_b (h₁ : 4 * x - 2 * y = a)
                     (h₂ : 6 * y - 12 * x = b)
                     (hb : b ≠ 0)
                     (ha_solution : ∃ x y, 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b) :
                     a / b = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_a_b_l945_94502


namespace NUMINAMATH_GPT_total_weight_of_sections_l945_94546

theorem total_weight_of_sections :
  let doll_length := 5
  let doll_weight := 29 / 8
  let tree_length := 4
  let tree_weight := 2.8
  let section_length := 2
  let doll_weight_per_meter := doll_weight / doll_length
  let tree_weight_per_meter := tree_weight / tree_length
  let doll_section_weight := doll_weight_per_meter * section_length
  let tree_section_weight := tree_weight_per_meter * section_length
  doll_section_weight + tree_section_weight = 57 / 20 :=
sorry

end NUMINAMATH_GPT_total_weight_of_sections_l945_94546


namespace NUMINAMATH_GPT_average_score_of_all_matches_is_36_l945_94596

noncomputable def average_score_of_all_matches
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) : ℝ :=
  (x + y + a + b + c) / 5

theorem average_score_of_all_matches_is_36
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  average_score_of_all_matches x y a b c h1 h2 h3x h3y h3a h3b h3c h4 = 36 := 
  by 
  sorry

end NUMINAMATH_GPT_average_score_of_all_matches_is_36_l945_94596


namespace NUMINAMATH_GPT_smallest_of_seven_even_numbers_sum_448_l945_94564

theorem smallest_of_seven_even_numbers_sum_448 :
  ∃ n : ℤ, n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10) + (n+12) = 448 ∧ n = 58 := 
by
  sorry

end NUMINAMATH_GPT_smallest_of_seven_even_numbers_sum_448_l945_94564


namespace NUMINAMATH_GPT_find_number_of_cows_l945_94539

-- Definitions from the conditions
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := sorry

-- Define the number of legs and heads
def legs := 2 * number_of_ducks + 4 * number_of_cows
def heads := number_of_ducks + number_of_cows

-- Given condition from the problem
def condition := legs = 2 * heads + 32

-- Assert the number of cows
theorem find_number_of_cows (h : condition) : number_of_cows = 16 :=
sorry

end NUMINAMATH_GPT_find_number_of_cows_l945_94539


namespace NUMINAMATH_GPT_sum_13_gt_0_l945_94544

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

axiom a7_gt_0 : 0 < a_n 7
axiom a8_lt_0 : a_n 8 < 0

theorem sum_13_gt_0 : S_n 13 > 0 :=
sorry

end NUMINAMATH_GPT_sum_13_gt_0_l945_94544


namespace NUMINAMATH_GPT_coins_left_l945_94579

-- Define the initial number of coins from each source
def piggy_bank_coins : ℕ := 15
def brother_coins : ℕ := 13
def father_coins : ℕ := 8

-- Define the number of coins given to Laura
def given_to_laura_coins : ℕ := 21

-- Define the total initial coins collected by Kylie
def total_initial_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

-- Lean statement to prove
theorem coins_left : total_initial_coins - given_to_laura_coins = 15 :=
by
  sorry

end NUMINAMATH_GPT_coins_left_l945_94579


namespace NUMINAMATH_GPT_sum_first_six_terms_geometric_seq_l945_94536

theorem sum_first_six_terms_geometric_seq (a r : ℝ)
  (h1 : a + a * r = 12)
  (h2 : a + a * r + a * r^2 + a * r^3 = 36) :
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 84 :=
sorry

end NUMINAMATH_GPT_sum_first_six_terms_geometric_seq_l945_94536


namespace NUMINAMATH_GPT_fraction_habitable_earth_l945_94581

theorem fraction_habitable_earth (one_fifth_land: ℝ) (one_third_inhabitable: ℝ)
  (h_land_fraction : one_fifth_land = 1 / 5)
  (h_inhabitable_fraction : one_third_inhabitable = 1 / 3) :
  (one_fifth_land * one_third_inhabitable) = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_habitable_earth_l945_94581


namespace NUMINAMATH_GPT_total_staff_correct_l945_94591

noncomputable def total_staff_weekdays_weekends : ℕ := 84

theorem total_staff_correct :
  let chefs_weekdays := 16
  let waiters_weekdays := 16
  let busboys_weekdays := 10
  let hostesses_weekdays := 5
  let additional_chefs_weekends := 5
  let additional_hostesses_weekends := 2
  
  let chefs_leave := chefs_weekdays * 25 / 100
  let waiters_leave := waiters_weekdays * 20 / 100
  let busboys_leave := busboys_weekdays * 30 / 100
  let hostesses_leave := hostesses_weekdays * 15 / 100
  
  let chefs_left_weekdays := chefs_weekdays - chefs_leave
  let waiters_left_weekdays := waiters_weekdays - Nat.floor waiters_leave
  let busboys_left_weekdays := busboys_weekdays - busboys_leave
  let hostesses_left_weekdays := hostesses_weekdays - Nat.ceil hostesses_leave

  let total_staff_weekdays := chefs_left_weekdays + waiters_left_weekdays + busboys_left_weekdays + hostesses_left_weekdays

  let chefs_weekends := chefs_weekdays + additional_chefs_weekends
  let waiters_weekends := waiters_left_weekdays
  let busboys_weekends := busboys_left_weekdays
  let hostesses_weekends := hostesses_weekdays + additional_hostesses_weekends
  
  let total_staff_weekends := chefs_weekends + waiters_weekends + busboys_weekends + hostesses_weekends

  total_staff_weekdays + total_staff_weekends = total_staff_weekdays_weekends
:= by
  sorry

end NUMINAMATH_GPT_total_staff_correct_l945_94591


namespace NUMINAMATH_GPT_intersection_points_rectangular_coords_l945_94509

theorem intersection_points_rectangular_coords :
  ∃ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = 2 * Real.cos θ ∧ ρ^2 * (Real.cos θ)^2 - 4 * ρ^2 * (Real.sin θ)^2 = 4 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
    (x = (1 + Real.sqrt 13) / 3 ∧ y = 0) := 
sorry

end NUMINAMATH_GPT_intersection_points_rectangular_coords_l945_94509


namespace NUMINAMATH_GPT_smallest_value_of_y_l945_94518

open Real

theorem smallest_value_of_y : 
  ∃ (y : ℝ), 6 * y^2 - 29 * y + 24 = 0 ∧ (∀ z : ℝ, 6 * z^2 - 29 * z + 24 = 0 → y ≤ z) ∧ y = 4 / 3 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_y_l945_94518
