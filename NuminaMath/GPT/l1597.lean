import Mathlib

namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1597_159704

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  let e := (1 + (b^2) / (a^2)).sqrt
  e

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a + b = 5)
  (h2 : a * b = 6)
  (h3 : a > b) :
  eccentricity a b = Real.sqrt 13 / 3 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1597_159704


namespace NUMINAMATH_GPT_range_of_a_l1597_159789

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ (a < -3 ∨ a > 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1597_159789


namespace NUMINAMATH_GPT_find_slant_height_l1597_159711

-- Definitions of the given conditions
variable (r1 r2 L A1 A2 : ℝ)
variable (π : ℝ := Real.pi)

-- The conditions as given in the problem
def conditions : Prop := 
  r1 = 3 ∧ r2 = 4 ∧ 
  (π * L * (r1 + r2) = A1 + A2) ∧ 
  (A1 = π * r1^2) ∧ 
  (A2 = π * r2^2)

-- The theorem stating the question and the correct answer
theorem find_slant_height (h : conditions r1 r2 L A1 A2) : 
  L = 5 := 
sorry

end NUMINAMATH_GPT_find_slant_height_l1597_159711


namespace NUMINAMATH_GPT_binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l1597_159792

-- Definitions and lemma statement
theorem binomial_coefficient_divisible_by_prime
  {p k : ℕ} (hp : Prime p) (hk : 0 < k) (hkp : k < p) :
  p ∣ Nat.choose p k := 
sorry

-- Theorem for k = 0 and k = p cases
theorem binomial_coefficient_extreme_cases {p : ℕ} (hp : Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end NUMINAMATH_GPT_binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l1597_159792


namespace NUMINAMATH_GPT_negative_to_zero_power_l1597_159758

theorem negative_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a) ^ 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_negative_to_zero_power_l1597_159758


namespace NUMINAMATH_GPT_max_elements_A_union_B_l1597_159774

noncomputable def sets_with_conditions (A B : Finset ℝ ) (n : ℕ) : Prop :=
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ A → s.sum id ∈ B) ∧
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ B → s.prod id ∈ A)

theorem max_elements_A_union_B {A B : Finset ℝ} (n : ℕ) (hn : 1 < n)
    (hA : A.card ≥ n) (hB : B.card ≥ n)
    (h_condition : sets_with_conditions A B n) :
    A.card + B.card ≤ 2 * n :=
  sorry

end NUMINAMATH_GPT_max_elements_A_union_B_l1597_159774


namespace NUMINAMATH_GPT_total_students_l1597_159735

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_total_students_l1597_159735


namespace NUMINAMATH_GPT_Z_is_divisible_by_10001_l1597_159793

theorem Z_is_divisible_by_10001
    (Z : ℕ) (a b c d : ℕ) (ha : a ≠ 0)
    (hZ : Z = 1000 * 10001 * a + 100 * 10001 * b + 10 * 10001 * c + 10001 * d)
    : 10001 ∣ Z :=
by {
    -- Proof omitted
    sorry
}

end NUMINAMATH_GPT_Z_is_divisible_by_10001_l1597_159793


namespace NUMINAMATH_GPT_krishan_money_l1597_159775

theorem krishan_money (R G K : ℕ) (h₁ : 7 * G = 17 * R) (h₂ : 7 * K = 17 * G) (h₃ : R = 686) : K = 4046 :=
  by sorry

end NUMINAMATH_GPT_krishan_money_l1597_159775


namespace NUMINAMATH_GPT_game_ends_and_last_numbers_depend_on_start_l1597_159743
-- Given that there are three positive integers a, b, c initially.
variables (a b c : ℕ)
-- Assume that a, b, and c are greater than zero.
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the gcd of the three numbers.
def g := gcd (gcd a b) c

-- Define the game step condition.
def step_condition (a b c : ℕ): Prop := a > gcd b c

-- Define the termination condition.
def termination_condition (a b c : ℕ): Prop := ¬ step_condition a b c

-- The main theorem
theorem game_ends_and_last_numbers_depend_on_start (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ n, ∃ b' c', termination_condition n b' c' ∧
  n = g ∧ b' = g ∧ c' = g :=
sorry

end NUMINAMATH_GPT_game_ends_and_last_numbers_depend_on_start_l1597_159743


namespace NUMINAMATH_GPT_height_relationship_l1597_159782

theorem height_relationship (r1 r2 h1 h2 : ℝ) (h_radii : r2 = 1.2 * r1) (h_volumes : π * r1^2 * h1 = π * r2^2 * h2) : h1 = 1.44 * h2 :=
by
  sorry

end NUMINAMATH_GPT_height_relationship_l1597_159782


namespace NUMINAMATH_GPT_jason_initial_cards_l1597_159708

theorem jason_initial_cards (a : ℕ) (b : ℕ) (x : ℕ) : 
  a = 224 → 
  b = 452 → 
  x = a + b → 
  x = 676 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jason_initial_cards_l1597_159708


namespace NUMINAMATH_GPT_amanda_bought_30_candy_bars_l1597_159755

noncomputable def candy_bars_bought (c1 c2 c3 c4 : ℕ) : ℕ :=
  let c5 := c4 * c2
  let c6 := c3 - c2
  let c7 := (c6 + c5) - c1
  c7

theorem amanda_bought_30_candy_bars :
  candy_bars_bought 7 3 22 4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_amanda_bought_30_candy_bars_l1597_159755


namespace NUMINAMATH_GPT_quadratic_roots_and_T_range_l1597_159764

theorem quadratic_roots_and_T_range
  (m : ℝ)
  (h1 : m ≥ -1)
  (x1 x2 : ℝ)
  (h2 : x1^2 + 2*(m-2)*x1 + (m^2 - 3*m + 3) = 0)
  (h3 : x2^2 + 2*(m-2)*x2 + (m^2 - 3*m + 3) = 0)
  (h4 : x1 ≠ x2)
  (h5 : x1^2 + x2^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 ∧ (0 < ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≤ 4 ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_and_T_range_l1597_159764


namespace NUMINAMATH_GPT_solve_for_x_l1597_159767

theorem solve_for_x (x : ℚ) (h : (x + 8) / (x - 4) = (x - 3) / (x + 6)) : 
  x = -12 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1597_159767


namespace NUMINAMATH_GPT_find_a1_geometric_sequence_l1597_159724

theorem find_a1_geometric_sequence (a₁ q : ℝ) (h1 : q ≠ 1) 
    (h2 : a₁ * (1 - q^3) / (1 - q) = 7)
    (h3 : a₁ * (1 - q^6) / (1 - q) = 63) :
    a₁ = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_geometric_sequence_l1597_159724


namespace NUMINAMATH_GPT_sum_squares_of_roots_of_quadratic_l1597_159763

theorem sum_squares_of_roots_of_quadratic:
  ∀ (s_1 s_2 : ℝ),
  (s_1 + s_2 = 20) ∧ (s_1 * s_2 = 32) →
  (s_1^2 + s_2^2 = 336) :=
by
  intros s_1 s_2 h
  sorry

end NUMINAMATH_GPT_sum_squares_of_roots_of_quadratic_l1597_159763


namespace NUMINAMATH_GPT_two_distinct_nonzero_complex_numbers_l1597_159790

noncomputable def count_distinct_nonzero_complex_numbers_satisfying_conditions : ℕ :=
sorry

theorem two_distinct_nonzero_complex_numbers :
  count_distinct_nonzero_complex_numbers_satisfying_conditions = 2 :=
sorry

end NUMINAMATH_GPT_two_distinct_nonzero_complex_numbers_l1597_159790


namespace NUMINAMATH_GPT_units_digit_of_7_pow_6_cubed_l1597_159794

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_6_cubed_l1597_159794


namespace NUMINAMATH_GPT_oranges_harvest_per_day_l1597_159721

theorem oranges_harvest_per_day (total_sacks : ℕ) (days : ℕ) (sacks_per_day : ℕ) 
  (h1 : total_sacks = 498) (h2 : days = 6) : total_sacks / days = sacks_per_day ∧ sacks_per_day = 83 :=
by
  sorry

end NUMINAMATH_GPT_oranges_harvest_per_day_l1597_159721


namespace NUMINAMATH_GPT_cube_volume_and_surface_area_l1597_159720

theorem cube_volume_and_surface_area (s : ℝ) (h : 12 * s = 72) :
  s^3 = 216 ∧ 6 * s^2 = 216 :=
by 
  sorry

end NUMINAMATH_GPT_cube_volume_and_surface_area_l1597_159720


namespace NUMINAMATH_GPT_line_parallel_condition_l1597_159768

theorem line_parallel_condition (a : ℝ) :
    (a = 1) → (∀ (x y : ℝ), (ax + 2 * y - 1 = 0) ∧ (x + (a + 1) * y + 4 = 0)) → (a = 1 ∨ a = -2) :=
by
sorry

end NUMINAMATH_GPT_line_parallel_condition_l1597_159768


namespace NUMINAMATH_GPT_Mark_water_balloon_spending_l1597_159731

theorem Mark_water_balloon_spending :
  let budget := 24
  let small_bag_cost := 4
  let small_bag_balloons := 50
  let medium_bag_balloons := 75
  let extra_large_bag_cost := 12
  let extra_large_bag_balloons := 200
  let total_balloons := 400
  (2 * extra_large_bag_balloons = total_balloons) → (2 * extra_large_bag_cost = budget) :=
by
  intros
  sorry

end NUMINAMATH_GPT_Mark_water_balloon_spending_l1597_159731


namespace NUMINAMATH_GPT_grace_earnings_september_l1597_159780

def charge_small_lawn_per_hour := 6
def charge_large_lawn_per_hour := 10
def charge_pull_small_weeds_per_hour := 11
def charge_pull_large_weeds_per_hour := 15
def charge_small_mulch_per_hour := 9
def charge_large_mulch_per_hour := 13

def hours_small_lawn := 20
def hours_large_lawn := 43
def hours_small_weeds := 4
def hours_large_weeds := 5
def hours_small_mulch := 6
def hours_large_mulch := 4

def earnings_small_lawn := hours_small_lawn * charge_small_lawn_per_hour
def earnings_large_lawn := hours_large_lawn * charge_large_lawn_per_hour
def earnings_small_weeds := hours_small_weeds * charge_pull_small_weeds_per_hour
def earnings_large_weeds := hours_large_weeds * charge_pull_large_weeds_per_hour
def earnings_small_mulch := hours_small_mulch * charge_small_mulch_per_hour
def earnings_large_mulch := hours_large_mulch * charge_large_mulch_per_hour

def total_earnings : ℕ :=
  earnings_small_lawn + earnings_large_lawn + earnings_small_weeds + earnings_large_weeds +
  earnings_small_mulch + earnings_large_mulch

theorem grace_earnings_september : total_earnings = 775 :=
by
  sorry

end NUMINAMATH_GPT_grace_earnings_september_l1597_159780


namespace NUMINAMATH_GPT_TreyHasSevenTimesAsManyTurtles_l1597_159759

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)

-- Conditions
def KristenHas12 : Kristen_turtles = 12 := sorry
def KrisHasQuarterOfKristen : Kris_turtles = Kristen_turtles / 4 := sorry
def TreyHas9MoreThanKristen : Trey_turtles = Kristen_turtles + 9 := sorry

-- Question: Prove that Trey has 7 times as many turtles as Kris
theorem TreyHasSevenTimesAsManyTurtles :
  Kristen_turtles = 12 → 
  Kris_turtles = Kristen_turtles / 4 → 
  Trey_turtles = Kristen_turtles + 9 → 
  Trey_turtles = 7 * Kris_turtles := sorry

end NUMINAMATH_GPT_TreyHasSevenTimesAsManyTurtles_l1597_159759


namespace NUMINAMATH_GPT_greatest_value_of_sum_l1597_159785

variable (x y : ℝ)

-- Conditions
axiom sum_of_squares : x^2 + y^2 = 130
axiom product : x * y = 36

-- Statement to prove
theorem greatest_value_of_sum : x + y ≤ Real.sqrt 202 := sorry

end NUMINAMATH_GPT_greatest_value_of_sum_l1597_159785


namespace NUMINAMATH_GPT_Kiera_envelopes_l1597_159713

theorem Kiera_envelopes (blue yellow green : ℕ) (total_envelopes : ℕ) 
  (cond1 : blue = 14) 
  (cond2 : total_envelopes = 46) 
  (cond3 : green = 3 * yellow) 
  (cond4 : total_envelopes = blue + yellow + green) : yellow = 6 - 8 := 
by sorry

end NUMINAMATH_GPT_Kiera_envelopes_l1597_159713


namespace NUMINAMATH_GPT_jamies_score_l1597_159760

def quiz_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct * 2) + (incorrect * (-0.5)) + (unanswered * 0.25)

theorem jamies_score :
  quiz_score 16 10 4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_jamies_score_l1597_159760


namespace NUMINAMATH_GPT_area_of_BEIH_l1597_159791

def calculate_area_of_quadrilateral (A B C D E F I H : (ℝ × ℝ)) : ℝ := 
  sorry

theorem area_of_BEIH : 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1, 0)
  let I := (3 / 5, 9 / 5)
  let H := (3 / 4, 3 / 4)
  calculate_area_of_quadrilateral A B C D E F I H = 27 / 40 :=
sorry

end NUMINAMATH_GPT_area_of_BEIH_l1597_159791


namespace NUMINAMATH_GPT_simplify_expression_l1597_159714

theorem simplify_expression : (Real.cos (18 * Real.pi / 180) * Real.cos (42 * Real.pi / 180) - 
                              Real.cos (72 * Real.pi / 180) * Real.sin (42 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1597_159714


namespace NUMINAMATH_GPT_no_real_roots_other_than_zero_l1597_159781

theorem no_real_roots_other_than_zero (k : ℝ) (h : k ≠ 0):
  ¬(∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_other_than_zero_l1597_159781


namespace NUMINAMATH_GPT_greg_total_earnings_correct_l1597_159797

def charge_per_dog := 20
def charge_per_minute := 1

def earnings_one_dog := charge_per_dog + charge_per_minute * 10
def earnings_two_dogs := 2 * (charge_per_dog + charge_per_minute * 7)
def earnings_three_dogs := 3 * (charge_per_dog + charge_per_minute * 9)

def total_earnings := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

theorem greg_total_earnings_correct : total_earnings = 171 := by
  sorry

end NUMINAMATH_GPT_greg_total_earnings_correct_l1597_159797


namespace NUMINAMATH_GPT_trigonometric_identity_l1597_159723

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1597_159723


namespace NUMINAMATH_GPT_problem_l1597_159706

noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x * y + y^2) * Real.sqrt (y^2 + y * z + z^2)) +
  (Real.sqrt (y^2 + y * z + z^2) * Real.sqrt (z^2 + z * x + x^2)) +
  (Real.sqrt (z^2 + z * x + x^2) * Real.sqrt (x^2 + x * y + y^2))

theorem problem (x y z : ℝ) (α β : ℝ) 
  (h1 : ∀ x y z, α * (x * y + y * z + z * x) ≤ M x y z)
  (h2 : ∀ x y z, M x y z ≤ β * (x^2 + y^2 + z^2)) :
  (∀ α, α ≤ 3) ∧ (∀ β, β ≥ 3) :=
sorry

end NUMINAMATH_GPT_problem_l1597_159706


namespace NUMINAMATH_GPT_inequality_part1_inequality_part2_l1597_159741

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

end NUMINAMATH_GPT_inequality_part1_inequality_part2_l1597_159741


namespace NUMINAMATH_GPT_pudding_distribution_l1597_159718

theorem pudding_distribution {puddings students : ℕ} (h1 : puddings = 315) (h2 : students = 218) : 
  ∃ (additional_puddings : ℕ), additional_puddings >= 121 ∧ ∃ (cups_per_student : ℕ), 
  (puddings + additional_puddings) ≥ students * cups_per_student :=
by
  sorry

end NUMINAMATH_GPT_pudding_distribution_l1597_159718


namespace NUMINAMATH_GPT_find_a_l1597_159752

noncomputable def a_b_c_complex (a b c : ℂ) : Prop :=
  a.re = a ∧ a + b + c = 4 ∧ a * b + b * c + c * a = 6 ∧ a * b * c = 8

theorem find_a (a b c : ℂ) (h : a_b_c_complex a b c) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1597_159752


namespace NUMINAMATH_GPT_percentage_of_first_relative_to_second_l1597_159757

theorem percentage_of_first_relative_to_second (X : ℝ) 
  (first_number : ℝ := 8/100 * X) 
  (second_number : ℝ := 16/100 * X) :
  (first_number / second_number) * 100 = 50 := 
sorry

end NUMINAMATH_GPT_percentage_of_first_relative_to_second_l1597_159757


namespace NUMINAMATH_GPT_feathers_per_flamingo_l1597_159722

theorem feathers_per_flamingo (num_boa : ℕ) (feathers_per_boa : ℕ) (num_flamingoes : ℕ) (pluck_rate : ℚ)
  (total_feathers : ℕ) (feathers_per_flamingo : ℕ) :
  num_boa = 12 →
  feathers_per_boa = 200 →
  num_flamingoes = 480 →
  pluck_rate = 0.25 →
  total_feathers = num_boa * feathers_per_boa →
  total_feathers = num_flamingoes * feathers_per_flamingo * pluck_rate →
  feathers_per_flamingo = 20 :=
by
  intros h_num_boa h_feathers_per_boa h_num_flamingoes h_pluck_rate h_total_feathers h_feathers_eq
  sorry

end NUMINAMATH_GPT_feathers_per_flamingo_l1597_159722


namespace NUMINAMATH_GPT_sqrt_comparison_l1597_159753

theorem sqrt_comparison :
  let a := Real.sqrt 2
  let b := Real.sqrt 7 - Real.sqrt 3
  let c := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by
{
  sorry
}

end NUMINAMATH_GPT_sqrt_comparison_l1597_159753


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l1597_159742

theorem solve_system1 (x y : ℚ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) :
  x = 3 / 2 ∧ y = -7 / 2 := 
sorry

theorem solve_system2 (x y : ℚ) (h1 : 3 * x - 2 * y = 1) (h2 : 7 * x + 4 * y = 11) :
  x = 1 ∧ y = 1 := 
sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l1597_159742


namespace NUMINAMATH_GPT_tom_has_7_blue_tickets_l1597_159778

def number_of_blue_tickets_needed_for_bible := 10 * 10 * 10
def toms_current_yellow_tickets := 8
def toms_current_red_tickets := 3
def toms_needed_blue_tickets := 163

theorem tom_has_7_blue_tickets : 
  (number_of_blue_tickets_needed_for_bible - 
    (toms_current_yellow_tickets * 10 * 10 + 
     toms_current_red_tickets * 10 + 
     toms_needed_blue_tickets)) = 7 :=
by
  -- Proof can be provided here
  sorry

end NUMINAMATH_GPT_tom_has_7_blue_tickets_l1597_159778


namespace NUMINAMATH_GPT_cristobal_read_more_pages_l1597_159729

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end NUMINAMATH_GPT_cristobal_read_more_pages_l1597_159729


namespace NUMINAMATH_GPT_cube_volume_l1597_159788

theorem cube_volume (s : ℝ) (h : s ^ 2 = 64) : s ^ 3 = 512 :=
sorry

end NUMINAMATH_GPT_cube_volume_l1597_159788


namespace NUMINAMATH_GPT_mixture_ratio_l1597_159783

variables (p q V W : ℝ)

-- Condition summaries:
-- - First jar has volume V, ratio of alcohol to water is p:1.
-- - Second jar has volume W, ratio of alcohol to water is q:2.

theorem mixture_ratio (hp : p > 0) (hq : q > 0) (hV : V > 0) (hW : W > 0) : 
  (p * V * (p + 2) + q * W * (p + 1)) / ((p + 1) * (q + 2) * (V + 2 * W)) =
  (p * V) / (p + 1) + (q * W) / (q + 2) :=
sorry

end NUMINAMATH_GPT_mixture_ratio_l1597_159783


namespace NUMINAMATH_GPT_find_principal_l1597_159799

-- Conditions as definitions
def amount : ℝ := 1120
def rate : ℝ := 0.05
def time : ℝ := 2

-- Required to add noncomputable due to the use of division and real numbers
noncomputable def principal : ℝ := amount / (1 + rate * time)

-- The main theorem statement which needs to be proved
theorem find_principal :
  principal = 1018.18 :=
sorry  -- Proof is not required; it is left as sorry

end NUMINAMATH_GPT_find_principal_l1597_159799


namespace NUMINAMATH_GPT_total_amount_collected_l1597_159728

theorem total_amount_collected 
  (num_members : ℕ)
  (annual_fee : ℕ)
  (cost_hardcover : ℕ)
  (num_hardcovers : ℕ)
  (cost_paperback : ℕ)
  (num_paperbacks : ℕ)
  (total_collected : ℕ) :
  num_members = 6 →
  annual_fee = 150 →
  cost_hardcover = 30 →
  num_hardcovers = 6 →
  cost_paperback = 12 →
  num_paperbacks = 6 →
  total_collected = (annual_fee + cost_hardcover * num_hardcovers + cost_paperback * num_paperbacks) * num_members →
  total_collected = 2412 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_total_amount_collected_l1597_159728


namespace NUMINAMATH_GPT_train_length_1080_l1597_159796

def length_of_train (speed time : ℕ) : ℕ := speed * time

theorem train_length_1080 (speed time : ℕ) (h1 : speed = 108) (h2 : time = 10) : length_of_train speed time = 1080 := by
  sorry

end NUMINAMATH_GPT_train_length_1080_l1597_159796


namespace NUMINAMATH_GPT_minimum_questionnaires_l1597_159750

theorem minimum_questionnaires (responses_needed : ℕ) (response_rate : ℝ)
  (h1 : responses_needed = 300) (h2 : response_rate = 0.70) :
  ∃ (n : ℕ), n = Nat.ceil (responses_needed / response_rate) ∧ n = 429 :=
by
  sorry

end NUMINAMATH_GPT_minimum_questionnaires_l1597_159750


namespace NUMINAMATH_GPT_JoggerDifference_l1597_159709

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end NUMINAMATH_GPT_JoggerDifference_l1597_159709


namespace NUMINAMATH_GPT_triangle_right_angled_l1597_159766

theorem triangle_right_angled (A B C : ℝ) (h : A + B + C = 180) (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x) :
  C = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_right_angled_l1597_159766


namespace NUMINAMATH_GPT_number_of_people_chose_pop_l1597_159740

theorem number_of_people_chose_pop (total_people : ℕ) (angle_pop : ℕ) (h1 : total_people = 540) (h2 : angle_pop = 270) : (total_people * (angle_pop / 360)) = 405 := by
  sorry

end NUMINAMATH_GPT_number_of_people_chose_pop_l1597_159740


namespace NUMINAMATH_GPT_boris_clock_time_l1597_159733

-- Define a function to compute the sum of digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem boris_clock_time (h m : ℕ) :
  sum_digits h + sum_digits m = 6 ∧ h + m = 15 ↔
  (h, m) = (0, 15) ∨ (h, m) = (1, 14) ∨ (h, m) = (2, 13) ∨ (h, m) = (3, 12) ∨
  (h, m) = (4, 11) ∨ (h, m) = (5, 10) ∨ (h, m) = (10, 5) ∨ (h, m) = (11, 4) ∨
  (h, m) = (12, 3) ∨ (h, m) = (13, 2) ∨ (h, m) = (14, 1) ∨ (h, m) = (15, 0) :=
by sorry

end NUMINAMATH_GPT_boris_clock_time_l1597_159733


namespace NUMINAMATH_GPT_factorial_expression_l1597_159730

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression (N : ℕ) (h : N > 0) :
  (factorial (N + 1) + factorial (N - 1)) / factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3 * N^2 + 2 * N) :=
by
  sorry

end NUMINAMATH_GPT_factorial_expression_l1597_159730


namespace NUMINAMATH_GPT_adam_spent_money_on_ferris_wheel_l1597_159795

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9
def tickets_used : ℕ := tickets_bought - tickets_left

theorem adam_spent_money_on_ferris_wheel :
  tickets_used * ticket_cost = 81 :=
by
  sorry

end NUMINAMATH_GPT_adam_spent_money_on_ferris_wheel_l1597_159795


namespace NUMINAMATH_GPT_rick_bought_30_guppies_l1597_159798

theorem rick_bought_30_guppies (G : ℕ) (T C : ℕ) 
  (h1 : T = 4 * C) 
  (h2 : C = 2 * G) 
  (h3 : G + C + T = 330) : 
  G = 30 := 
by 
  sorry

end NUMINAMATH_GPT_rick_bought_30_guppies_l1597_159798


namespace NUMINAMATH_GPT_probability_of_pink_tie_l1597_159761

theorem probability_of_pink_tie 
  (black_ties gold_ties pink_ties : ℕ) 
  (h_black : black_ties = 5) 
  (h_gold : gold_ties = 7) 
  (h_pink : pink_ties = 8) 
  (h_total : (5 + 7 + 8) = (black_ties + gold_ties + pink_ties)) 
  : (pink_ties : ℚ) / (black_ties + gold_ties + pink_ties) = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_pink_tie_l1597_159761


namespace NUMINAMATH_GPT_problem_proof_l1597_159771

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end NUMINAMATH_GPT_problem_proof_l1597_159771


namespace NUMINAMATH_GPT_fraction_before_simplification_is_24_56_l1597_159737

-- Definitions of conditions
def fraction_before_simplification_simplifies_to_3_7 (a b : ℕ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧ Int.gcd a b = 1 ∧ (a = 3 * Int.gcd a b ∧ b = 7 * Int.gcd a b)

def sum_of_numerator_and_denominator_is_80 (a b : ℕ) : Prop :=
  a + b = 80

-- Theorem to prove
theorem fraction_before_simplification_is_24_56 (a b : ℕ) :
  fraction_before_simplification_simplifies_to_3_7 a b →
  sum_of_numerator_and_denominator_is_80 a b →
  (a, b) = (24, 56) :=
sorry

end NUMINAMATH_GPT_fraction_before_simplification_is_24_56_l1597_159737


namespace NUMINAMATH_GPT_arith_seq_a1_a7_sum_l1597_159748

variable (a : ℕ → ℝ) (d : ℝ)

-- Conditions
def arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

def condition_sum : Prop :=
  a 3 + a 4 + a 5 = 12

-- Equivalent proof problem statement
theorem arith_seq_a1_a7_sum :
  arithmetic_sequence a d →
  condition_sum a →
  a 1 + a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arith_seq_a1_a7_sum_l1597_159748


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1597_159787

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1597_159787


namespace NUMINAMATH_GPT_problem_l1597_159772

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end NUMINAMATH_GPT_problem_l1597_159772


namespace NUMINAMATH_GPT_shares_distribution_correct_l1597_159784

def shares_distributed (a b c d e : ℕ) : Prop :=
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600

theorem shares_distribution_correct (a b c d e : ℕ) :
  (a = (1/2 : ℚ) * b)
  ∧ (b = (1/3 : ℚ) * c)
  ∧ (c = 2 * d)
  ∧ (d = (1/4 : ℚ) * e)
  ∧ (a + b + c + d + e = 1200) → shares_distributed a b c d e :=
sorry

end NUMINAMATH_GPT_shares_distribution_correct_l1597_159784


namespace NUMINAMATH_GPT_decorations_cost_correct_l1597_159719

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end NUMINAMATH_GPT_decorations_cost_correct_l1597_159719


namespace NUMINAMATH_GPT_molecular_weight_compound_l1597_159712

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_Cl : ℝ := 35.453

def molecular_weight (nH nC nO nN nCl : ℕ) : ℝ :=
  nH * atomic_weight_H + nC * atomic_weight_C + nO * atomic_weight_O + nN * atomic_weight_N + nCl * atomic_weight_Cl

theorem molecular_weight_compound :
  molecular_weight 4 2 3 1 2 = 160.964 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l1597_159712


namespace NUMINAMATH_GPT_find_P_eq_30_l1597_159769

theorem find_P_eq_30 (P Q R S : ℕ) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  P * Q = 120 ∧ R * S = 120 ∧ P - Q = R + S → P = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_P_eq_30_l1597_159769


namespace NUMINAMATH_GPT_area_of_regular_octagon_l1597_159705

/-- The perimeters of a square and a regular octagon are equal.
    The area of the square is 16.
    Prove that the area of the regular octagon is 8 + 8 * sqrt 2. -/
theorem area_of_regular_octagon (a b : ℝ) (h1 : 4 * a = 8 * b) (h2 : a^2 = 16) :
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_regular_octagon_l1597_159705


namespace NUMINAMATH_GPT_volume_of_cut_cone_l1597_159726

theorem volume_of_cut_cone (V_frustum : ℝ) (A_bottom : ℝ) (A_top : ℝ) (V_cut_cone : ℝ) :
  V_frustum = 52 ∧ A_bottom = 9 * A_top → V_cut_cone = 54 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cut_cone_l1597_159726


namespace NUMINAMATH_GPT_ticket_cost_l1597_159736

theorem ticket_cost
    (rows : ℕ) (seats_per_row : ℕ)
    (fraction_sold : ℚ) (total_earnings : ℚ)
    (N : ℕ := rows * seats_per_row)
    (S : ℚ := fraction_sold * N)
    (C : ℚ := total_earnings / S)
    (h1 : rows = 20) (h2 : seats_per_row = 10)
    (h3 : fraction_sold = 3 / 4) (h4 : total_earnings = 1500) :
    C = 10 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_l1597_159736


namespace NUMINAMATH_GPT_three_digit_number_with_ones_digit_5_divisible_by_5_l1597_159747

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end NUMINAMATH_GPT_three_digit_number_with_ones_digit_5_divisible_by_5_l1597_159747


namespace NUMINAMATH_GPT_ice_cream_sundaes_l1597_159746

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String) (h1 : vanilla ∈ flavors) (h2 : flavors.card = 8) :
  let remaining_flavors := flavors.erase vanilla
  remaining_flavors.card = 7 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_sundaes_l1597_159746


namespace NUMINAMATH_GPT_systematic_sampling_twentieth_group_number_l1597_159770

theorem systematic_sampling_twentieth_group_number 
  (total_students : ℕ) 
  (total_groups : ℕ) 
  (first_group_number : ℕ) 
  (interval : ℕ) 
  (n : ℕ) 
  (drawn_number : ℕ) :
  total_students = 400 →
  total_groups = 20 →
  first_group_number = 11 →
  interval = 20 →
  n = 20 →
  drawn_number = 11 + 20 * (n - 1) →
  drawn_number = 391 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_twentieth_group_number_l1597_159770


namespace NUMINAMATH_GPT_price_of_sundae_l1597_159716

variable (num_ice_cream_bars num_sundaes : ℕ)
variable (total_price : ℚ)
variable (price_per_ice_cream_bar : ℚ)
variable (price_per_sundae : ℚ)

theorem price_of_sundae :
  num_ice_cream_bars = 125 →
  num_sundaes = 125 →
  total_price = 225 →
  price_per_ice_cream_bar = 0.60 →
  price_per_sundae = (total_price - (num_ice_cream_bars * price_per_ice_cream_bar)) / num_sundaes →
  price_per_sundae = 1.20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_price_of_sundae_l1597_159716


namespace NUMINAMATH_GPT_first_tap_time_l1597_159701

-- Define the variables and conditions
variables (T : ℝ)
-- The cistern can be emptied by the second tap in 9 hours
-- Both taps together fill the cistern in 7.2 hours.
def first_tap_fills_cistern_in_time (T : ℝ) :=
  (1 / T) - (1 / 9) = 1 / 7.2

theorem first_tap_time :
  first_tap_fills_cistern_in_time 4 :=
by
  -- now we can use the definition to show the proof
  unfold first_tap_fills_cistern_in_time
  -- directly substitute and show
  sorry

end NUMINAMATH_GPT_first_tap_time_l1597_159701


namespace NUMINAMATH_GPT_cube_volume_l1597_159700

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end NUMINAMATH_GPT_cube_volume_l1597_159700


namespace NUMINAMATH_GPT_reciprocal_of_36_recurring_decimal_l1597_159756

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end NUMINAMATH_GPT_reciprocal_of_36_recurring_decimal_l1597_159756


namespace NUMINAMATH_GPT_cannot_be_zero_l1597_159777

-- Define polynomial Q(x)
def Q (x : ℝ) (f g h i j : ℝ) : ℝ := x^5 + f * x^4 + g * x^3 + h * x^2 + i * x + j

-- Define the hypotheses for the proof
def distinct_roots (a b c d e : ℝ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def one_root_is_one (f g h i j : ℝ) := Q 1 f g h i j = 0

-- Statement to prove
theorem cannot_be_zero (f g h i j a b c d : ℝ)
  (h1 : Q 1 f g h i j = 0)
  (h2 : distinct_roots 1 a b c d)
  (h3 : Q 1 f g h i j = (1-a)*(1-b)*(1-c)*(1-d)) :
  i ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_zero_l1597_159777


namespace NUMINAMATH_GPT_paint_time_l1597_159739

theorem paint_time (n₁ n₂ h: ℕ) (t₁ t₂: ℕ) (constant: ℕ):
  n₁ = 6 → t₁ = 8 → h = 2 → constant = 96 →
  constant = n₁ * t₁ * h → n₂ = 4 → constant = n₂ * t₂ * h →
  t₂ = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paint_time_l1597_159739


namespace NUMINAMATH_GPT_quad_inequality_solution_set_is_reals_l1597_159776

theorem quad_inequality_solution_set_is_reals (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) := 
sorry

end NUMINAMATH_GPT_quad_inequality_solution_set_is_reals_l1597_159776


namespace NUMINAMATH_GPT_odd_multiple_of_9_implies_multiple_of_3_l1597_159745

theorem odd_multiple_of_9_implies_multiple_of_3 :
  ∀ (S : ℤ), (∀ (n : ℤ), 9 * n = S → ∃ (m : ℤ), 3 * m = S) ∧ (S % 2 ≠ 0) → (∃ (m : ℤ), 3 * m = S) :=
by
  sorry

end NUMINAMATH_GPT_odd_multiple_of_9_implies_multiple_of_3_l1597_159745


namespace NUMINAMATH_GPT_visible_yellow_bus_length_correct_l1597_159762

noncomputable def red_bus_length : ℝ := 48
noncomputable def orange_car_length : ℝ := red_bus_length / 4
noncomputable def yellow_bus_length : ℝ := 3.5 * orange_car_length
noncomputable def green_truck_length : ℝ := 2 * orange_car_length
noncomputable def total_vehicle_length : ℝ := yellow_bus_length + green_truck_length
noncomputable def visible_yellow_bus_length : ℝ := 0.75 * yellow_bus_length

theorem visible_yellow_bus_length_correct :
  visible_yellow_bus_length = 31.5 := 
sorry

end NUMINAMATH_GPT_visible_yellow_bus_length_correct_l1597_159762


namespace NUMINAMATH_GPT_find_m_n_l1597_159786

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end NUMINAMATH_GPT_find_m_n_l1597_159786


namespace NUMINAMATH_GPT_two_primes_equal_l1597_159707

theorem two_primes_equal
  (a b c : ℕ)
  (p q r : ℕ)
  (hp : p = b^c + a ∧ Nat.Prime p)
  (hq : q = a^b + c ∧ Nat.Prime q)
  (hr : r = c^a + b ∧ Nat.Prime r) :
  p = q ∨ q = r ∨ r = p := 
sorry

end NUMINAMATH_GPT_two_primes_equal_l1597_159707


namespace NUMINAMATH_GPT_count_satisfying_pairs_l1597_159732

theorem count_satisfying_pairs :
  ∃ (count : ℕ), count = 540 ∧ 
  (∀ (w n : ℕ), (w % 23 = 5) ∧ (w < 450) ∧ (n % 17 = 7) ∧ (n < 450) → w < 450 ∧ n < 450) := 
by
  sorry

end NUMINAMATH_GPT_count_satisfying_pairs_l1597_159732


namespace NUMINAMATH_GPT_program_total_cost_l1597_159727

-- Define the necessary variables and constants
def ms_to_s : Float := 0.001
def os_overhead : Float := 1.07
def cost_per_ms : Float := 0.023
def mount_cost : Float := 5.35
def time_required : Float := 1.5

-- Calculate components of the total cost
def total_cost_for_computer_time := (time_required * 1000) * cost_per_ms
def total_cost := os_overhead + total_cost_for_computer_time + mount_cost

-- State the theorem
theorem program_total_cost : total_cost = 40.92 := by
  sorry

end NUMINAMATH_GPT_program_total_cost_l1597_159727


namespace NUMINAMATH_GPT_chosen_numbers_divisibility_l1597_159749

theorem chosen_numbers_divisibility (n : ℕ) (S : Finset ℕ) (hS : S.card > (n + 1) / 2) :
  ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ a ∣ b :=
by sorry

end NUMINAMATH_GPT_chosen_numbers_divisibility_l1597_159749


namespace NUMINAMATH_GPT_pascal_triangle_ratio_l1597_159702

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (3 * r + 3 = 2 * n - 2 * r))
  (h2 : (4 * r + 8 = 3 * n - 3 * r - 3)) : 
  n = 34 :=
sorry

end NUMINAMATH_GPT_pascal_triangle_ratio_l1597_159702


namespace NUMINAMATH_GPT_smallest_integer_x_l1597_159703

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 12) : x ≥ 7 :=
sorry

end NUMINAMATH_GPT_smallest_integer_x_l1597_159703


namespace NUMINAMATH_GPT_total_length_figure_2_l1597_159715

-- Define the conditions for Figure 1
def left_side_figure_1 := 10
def right_side_figure_1 := 7
def top_side_figure_1 := 3
def bottom_side_figure_1_seg1 := 2
def bottom_side_figure_1_seg2 := 1

-- Define the conditions for Figure 2 after removal
def left_side_figure_2 := left_side_figure_1
def right_side_figure_2 := right_side_figure_1
def top_side_figure_2 := 0
def bottom_side_figure_2 := top_side_figure_1 + bottom_side_figure_1_seg1 + bottom_side_figure_1_seg2

-- The Lean statement proving the total length in Figure 2
theorem total_length_figure_2 : 
  left_side_figure_2 + right_side_figure_2 + top_side_figure_2 + bottom_side_figure_2 = 23 := by
  sorry

end NUMINAMATH_GPT_total_length_figure_2_l1597_159715


namespace NUMINAMATH_GPT_ratio_of_area_to_breadth_l1597_159773

variable (l b : ℕ)

theorem ratio_of_area_to_breadth 
  (h1 : b = 14) 
  (h2 : l - b = 10) : 
  (l * b) / b = 24 := by
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_breadth_l1597_159773


namespace NUMINAMATH_GPT_simplify_expression_l1597_159734

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 5 * x + 9 * x^2 + 8 - (6 - 5 * x - 3 * x^2)

-- Define the expected simplified form
def expected_expression (x : ℝ) : ℝ := 12 * x^2 + 10 * x + 2

-- The theorem we want to prove
theorem simplify_expression (x : ℝ) : given_expression x = expected_expression x := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1597_159734


namespace NUMINAMATH_GPT_ticket_price_for_children_l1597_159725

open Nat

theorem ticket_price_for_children
  (C : ℕ)
  (adult_ticket_price : ℕ := 12)
  (num_adults : ℕ := 3)
  (num_children : ℕ := 3)
  (total_cost : ℕ := 66)
  (H : num_adults * adult_ticket_price + num_children * C = total_cost) :
  C = 10 :=
sorry

end NUMINAMATH_GPT_ticket_price_for_children_l1597_159725


namespace NUMINAMATH_GPT_find_m_l1597_159754

noncomputable def f : ℝ → ℝ := sorry

theorem find_m (h₁ : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h₂ : f 2 = m) : m = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1597_159754


namespace NUMINAMATH_GPT_fruit_salad_cost_3_l1597_159738

def cost_per_fruit_salad (num_people sodas_per_person soda_cost sandwich_cost num_snacks snack_cost total_cost : ℕ) : ℕ :=
  let total_soda_cost := num_people * sodas_per_person * soda_cost
  let total_sandwich_cost := num_people * sandwich_cost
  let total_snack_cost := num_snacks * snack_cost
  let total_known_cost := total_soda_cost + total_sandwich_cost + total_snack_cost
  let total_fruit_salad_cost := total_cost - total_known_cost
  total_fruit_salad_cost / num_people

theorem fruit_salad_cost_3 :
  cost_per_fruit_salad 4 2 2 5 3 4 60 = 3 :=
by
  sorry

end NUMINAMATH_GPT_fruit_salad_cost_3_l1597_159738


namespace NUMINAMATH_GPT_find_valid_primes_and_integers_l1597_159779

def is_prime (p : ℕ) : Prop := Nat.Prime p

def valid_pair (p x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * p ∧ x^(p-1) ∣ (p-1)^x + 1

theorem find_valid_primes_and_integers (p x : ℕ) (hp : is_prime p) 
  (hx : valid_pair p x) : 
  (p = 2 ∧ x = 1) ∨ 
  (p = 2 ∧ x = 2) ∨ 
  (p = 3 ∧ x = 1) ∨ 
  (p = 3 ∧ x = 3) ∨
  (x = 1) :=
sorry

end NUMINAMATH_GPT_find_valid_primes_and_integers_l1597_159779


namespace NUMINAMATH_GPT_midpoint_sum_is_correct_l1597_159744

theorem midpoint_sum_is_correct:
  let A := (10, 8)
  let B := (-4, -6)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + midpoint.2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_sum_is_correct_l1597_159744


namespace NUMINAMATH_GPT_closest_perfect_square_to_273_l1597_159751

theorem closest_perfect_square_to_273 : ∃ n : ℕ, (n^2 = 289) ∧ 
  ∀ m : ℕ, (m^2 < 273 → 273 - m^2 ≥ 1) ∧ (m^2 > 273 → m^2 - 273 ≥ 16) :=
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_273_l1597_159751


namespace NUMINAMATH_GPT_percent_profit_l1597_159765

theorem percent_profit (C S : ℝ) (h : 60 * C = 40 * S) : (S - C) / C * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percent_profit_l1597_159765


namespace NUMINAMATH_GPT_roots_triple_relation_l1597_159717

theorem roots_triple_relation (a b c : ℤ) (α β : ℤ)
    (h_quad : a ≠ 0)
    (h_roots : α + β = -b / a)
    (h_prod : α * β = c / a)
    (h_triple : β = 3 * α) :
    3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_GPT_roots_triple_relation_l1597_159717


namespace NUMINAMATH_GPT_number_of_rowers_l1597_159710

theorem number_of_rowers (total_coaches : ℕ) (votes_per_coach : ℕ) (votes_per_rower : ℕ) 
  (htotal_coaches : total_coaches = 36) (hvotes_per_coach : votes_per_coach = 5) 
  (hvotes_per_rower : votes_per_rower = 3) : 
  (total_coaches * votes_per_coach) / votes_per_rower = 60 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_rowers_l1597_159710
