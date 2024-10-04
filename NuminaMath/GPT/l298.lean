import Mathlib

namespace greatest_x_lcm_l298_298902

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298902


namespace algebraic_inequality_solution_l298_298625

theorem algebraic_inequality_solution (x : ℝ) : (1 + 2 * x ≤ 8 + 3 * x) → (x ≥ -7) :=
by
  sorry

end algebraic_inequality_solution_l298_298625


namespace intersection_of_A_and_B_l298_298192

open Set

def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { y | ∃ x, y = (1 / 2) ^ x ∧ x > -1 }

theorem intersection_of_A_and_B : A ∩ B = { y | 0 < y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l298_298192


namespace inequality_solution_l298_298227

variable {a b c : ℝ}

theorem inequality_solution (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c ≥ 1) :
  (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) ≤ 1) ∧ (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_solution_l298_298227


namespace factorize_difference_of_squares_l298_298144

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l298_298144


namespace time_spent_on_seals_l298_298446

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l298_298446


namespace no_prime_p_for_base_eqn_l298_298763

theorem no_prime_p_for_base_eqn (p : ℕ) (hp: p.Prime) :
  let f (p : ℕ) := 1009 * p^3 + 307 * p^2 + 115 * p + 126 + 7
  let g (p : ℕ) := 143 * p^2 + 274 * p + 361
  f p = g p → false :=
sorry

end no_prime_p_for_base_eqn_l298_298763


namespace Jungkook_blue_balls_unchanged_l298_298734

variable (initialRedBalls : ℕ) (initialBlueBalls : ℕ) (initialYellowBalls : ℕ)
variable (newYellowBallGifted: ℕ)

-- Define the initial conditions
def Jungkook_balls := initialRedBalls = 5 ∧ initialBlueBalls = 4 ∧ initialYellowBalls = 3 ∧ newYellowBallGifted = 1

-- State the theorem to prove
theorem Jungkook_blue_balls_unchanged (h : Jungkook_balls initRed initBlue initYellow newYellowGift): initialBlueBalls = 4 := 
by
sorry

end Jungkook_blue_balls_unchanged_l298_298734


namespace fish_caught_300_l298_298891

def fish_caught_at_dawn (F : ℕ) : Prop :=
  (3 * F / 5) = 180

theorem fish_caught_300 : ∃ F, fish_caught_at_dawn F ∧ F = 300 := 
by 
  use 300 
  have h1 : 3 * 300 / 5 = 180 := by norm_num 
  exact ⟨h1, rfl⟩

end fish_caught_300_l298_298891


namespace range_of_a_for_three_zeros_l298_298518

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298518


namespace fundraising_exceeded_goal_l298_298245

theorem fundraising_exceeded_goal:
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  total - goal = 600 :=
by
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  have h_goal : goal = 4000 := rfl
  have h_ken : ken = 600 := rfl
  have h_mary : mary = 5 * ken := rfl
  have h_scott : scott = mary / 3 := rfl
  have h_total : total = ken + mary + scott := rfl
  calc total - goal = (ken + mary + scott) - goal : by rw h_total
  ... = (600 + 3000 + 1000) - 4000 : by {rw [h_ken, h_mary, h_scott], norm_num}
  ... = 600 : by norm_num

end fundraising_exceeded_goal_l298_298245


namespace total_fish_l298_298320

theorem total_fish :
  let Billy := 10
  let Tony := 3 * Billy
  let Sarah := Tony + 5
  let Bobby := 2 * Sarah
  in Billy + Tony + Sarah + Bobby = 145 :=
by
  sorry

end total_fish_l298_298320


namespace find_a_for_quadratic_l298_298171

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l298_298171


namespace ratio_of_perimeters_l298_298079

theorem ratio_of_perimeters (s : ℝ) (hs : s > 0) :
  let small_triangle_perimeter := s + (s / 2) + (s / 2)
  let large_rectangle_perimeter := 2 * (s + (s / 2))
  small_triangle_perimeter / large_rectangle_perimeter = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l298_298079


namespace expand_product_l298_298824

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := 
by 
  sorry

end expand_product_l298_298824


namespace number_of_multiples_of_six_ending_in_four_and_less_than_800_l298_298512

-- Definitions from conditions
def is_multiple_of_six (n : ℕ) : Prop := n % 6 = 0
def ends_with_four (n : ℕ) : Prop := n % 10 = 4
def less_than_800 (n : ℕ) : Prop := n < 800

-- Theorem to prove
theorem number_of_multiples_of_six_ending_in_four_and_less_than_800 :
  ∃ k : ℕ, k = 26 ∧ ∀ n : ℕ, (is_multiple_of_six n ∧ ends_with_four n ∧ less_than_800 n) → n = 24 + 60 * k ∨ n = 54 + 60 * k :=
sorry

end number_of_multiples_of_six_ending_in_four_and_less_than_800_l298_298512


namespace arcsin_inequality_l298_298486

theorem arcsin_inequality (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  (Real.arcsin x + Real.arcsin y > Real.pi / 2) ↔ (x ≥ 0 ∧ y ≥ 0 ∧ (y^2 + x^2 > 1)) := by
sorry

end arcsin_inequality_l298_298486


namespace line_intersects_midpoint_l298_298893

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end line_intersects_midpoint_l298_298893


namespace estimate_expr_range_l298_298488

theorem estimate_expr_range :
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) < 6 :=
  sorry

end estimate_expr_range_l298_298488


namespace greatest_x_lcm_105_l298_298917

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298917


namespace solution_set_of_abs_inequality_l298_298087

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality_l298_298087


namespace least_number_to_subtract_l298_298100

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 427398) (k : d = 13) (r_val : r = 2) : 
  ∃ x : ℕ, (n - x) % d = 0 ∧ r = x :=
by sorry

end least_number_to_subtract_l298_298100


namespace impossible_transform_l298_298314

-- Definition of a word and its tripling
def word := List Bool
def tripling (A : word) : word := A ++ A ++ A

-- Definition of the value function v
def value_function (A : word) : ℕ :=
  A.foldl (fun (acc: ℕ × ℕ) (x: Bool) => (acc.1 + 1, acc.2 + (if x then acc.1 + 1 else 0))) (0, 0) |>.2

-- Theorem stating the impossibility of transforming '10' to '01' using the given operations.
theorem impossible_transform (A B : word) (h_trip : ∀ C : word, A = B ++ tripling C ∨ ∃ D, A = D ++ tripling C ++ B) :
  A = [true, false] → B = [false, true] → False :=
by
  intro hA hB
  let vA := value_function A
  let vB := value_function B
  -- Value function mod 3 should be preserved
  have h_mod : vA % 3 = vB % 3 := sorry -- Derived from given conditions
  have h_val_10 : vA = 1 := by
    unfold value_function;
    simp [hA]
  have h_val_01 : vB = 2 := by
    unfold value_function;
    simp [hB]
  -- Contradiction
  have h_contra : 1 % 3 ≠ 2 % 3 := by
    simp
  contradiction

end impossible_transform_l298_298314


namespace B2F_base16_to_base10_l298_298135

theorem B2F_base16_to_base10 :
  let d2 := 11
  let d1 := 2
  let d0 := 15
  d2 * 16^2 + d1 * 16^1 + d0 * 16^0 = 2863 :=
by
  let d2 := 11
  let d1 := 2
  let d0 := 15
  sorry

end B2F_base16_to_base10_l298_298135


namespace biased_die_sum_is_odd_l298_298955

def biased_die_probabilities : Prop :=
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let scenarios := [
    (1/3) * (2/3)^2,
    (1/3)^3
  ]
  let sum := scenarios.sum
  sum = 13 / 27

theorem biased_die_sum_is_odd :
  biased_die_probabilities := by
    sorry

end biased_die_sum_is_odd_l298_298955


namespace greatest_x_lcm_105_l298_298916

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298916


namespace problem_1_problem_2_l298_298196

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

theorem problem_1 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 :=
by
  sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + 2 * b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
by
  sorry

end problem_1_problem_2_l298_298196


namespace direct_proportion_function_l298_298695

theorem direct_proportion_function (m : ℝ) : 
  (m^2 + 2 * m ≠ 0) ∧ (m^2 - 3 = 1) → m = 2 :=
by {
  sorry
}

end direct_proportion_function_l298_298695


namespace find_value_of_a_l298_298042

theorem find_value_of_a (a : ℚ) (h : a + a / 4 - 1 / 2 = 2) : a = 2 :=
by
  sorry

end find_value_of_a_l298_298042


namespace statement1_statement2_l298_298075

def is_pow_of_two (a : ℕ) : Prop := ∃ n : ℕ, a = 2^(n + 1)
def in_A (a : ℕ) : Prop := is_pow_of_two a
def not_in_A (a : ℕ) : Prop := ¬ in_A a ∧ a ≠ 1

theorem statement1 : 
  ∀ (a : ℕ), in_A a → ∀ (b : ℕ), b < 2 * a - 1 → ¬ (2 * a ∣ b * (b + 1)) := 
by {
  sorry
}

theorem statement2 :
  ∀ (a : ℕ), not_in_A a → ∃ (b : ℕ), b < 2 * a - 1 ∧ (2 * a ∣ b * (b + 1)) :=
by {
  sorry
}

end statement1_statement2_l298_298075


namespace discount_difference_l298_298813

noncomputable def single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

noncomputable def successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate => acc * (1 - rate)) amount

theorem discount_difference:
  let amount := 12000
  let single_rate := 0.35
  let successive_rates := [0.25, 0.08, 0.02]
  single_discount amount single_rate - successive_discounts amount successive_rates = 314.4 := 
  sorry

end discount_difference_l298_298813


namespace greatest_value_of_x_l298_298930

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298930


namespace b_catches_A_distance_l298_298102

noncomputable def speed_A := 10 -- kmph
noncomputable def speed_B := 20 -- kmph
noncomputable def time_diff := 7 -- hours
noncomputable def distance_A := speed_A * time_diff -- km
noncomputable def relative_speed := speed_B - speed_A -- kmph
noncomputable def catch_up_time := distance_A / relative_speed -- hours
noncomputable def distance_B := speed_B * catch_up_time -- km

theorem b_catches_A_distance :
  distance_B = 140 := by
  sorry

end b_catches_A_distance_l298_298102


namespace time_spent_on_seals_l298_298445

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l298_298445


namespace find_second_angle_l298_298890

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem find_second_angle
  (A B C : ℝ)
  (hA : A = 32)
  (hC : C = 2 * A - 12)
  (hB : B = 3 * A)
  (h_sum : angle_in_triangle A B C) :
  B = 96 :=
by sorry

end find_second_angle_l298_298890


namespace sufficient_not_necessary_l298_298503

theorem sufficient_not_necessary (a : ℝ) (h : a ≠ 0) : 
  (a > 1 → a > 1 / a) ∧ (¬ (a > 1) → a > 1 / a → -1 < a ∧ a < 0) :=
sorry

end sufficient_not_necessary_l298_298503


namespace joe_initial_paint_amount_l298_298216

theorem joe_initial_paint_amount (P : ℝ) 
  (h1 : (2/3) * P + (1/15) * P = 264) : P = 360 :=
sorry

end joe_initial_paint_amount_l298_298216


namespace parity_related_to_phi_not_omega_l298_298600

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem parity_related_to_phi_not_omega (ω : ℝ) (φ : ℝ) (h : 0 < ω) :
  (∃ k : ℤ, φ = k * Real.pi → ∀ x : ℝ, f ω φ (-x) = -f ω φ x) ∧
  (∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 → ∀ x : ℝ, f ω φ (-x) = f ω φ x) :=
sorry

end parity_related_to_phi_not_omega_l298_298600


namespace jogger_ahead_distance_l298_298109

def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def train_length_m : ℝ := 120
def passing_time_s : ℝ := 31

theorem jogger_ahead_distance :
  let V_rel := (train_speed_kmh - jogger_speed_kmh) * (1000 / 3600)
  let Distance_train := V_rel * passing_time_s 
  Distance_train = 310 → 
  Distance_train = 190 + train_length_m :=
by
  intros
  sorry

end jogger_ahead_distance_l298_298109


namespace smallest_integer_with_20_divisors_l298_298648

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, (n > 0 ∧ (∃ (d : ℕ → Prop), (∀ m, d m ↔ m ∣ n) ∧ (card { m : ℕ | d m } = 20)) ∧ (∀ k : ℕ, k > 0 ∧ (∃ (d' : ℕ → Prop), (∀ m, d' m ↔ m ∣ k) ∧ (card { m : ℕ | d' m } = 20)) → k ≥ n)) ∧ n = 240 :=
by { sorry }

end smallest_integer_with_20_divisors_l298_298648


namespace smallest_possible_X_l298_298861

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end smallest_possible_X_l298_298861


namespace jamies_mother_twice_age_l298_298606

theorem jamies_mother_twice_age (y : ℕ) :
  ∀ (jamie_age_2010 mother_age_2010 : ℕ), 
  jamie_age_2010 = 10 → 
  mother_age_2010 = 5 * jamie_age_2010 → 
  mother_age_2010 + y = 2 * (jamie_age_2010 + y) → 
  2010 + y = 2040 :=
by
  intros jamie_age_2010 mother_age_2010 h_jamie h_mother h_eq
  sorry

end jamies_mother_twice_age_l298_298606


namespace alice_bob_numbers_count_101_l298_298124

theorem alice_bob_numbers_count_101 : 
  ∃ n : ℕ, (∀ x, 3 ≤ x ∧ x ≤ 2021 → (∃ k l, x = 3 + 5 * k ∧ x = 2021 - 4 * l)) → n = 101 :=
by
  sorry

end alice_bob_numbers_count_101_l298_298124


namespace proof_bd_leq_q2_l298_298032

variables {a b c d p q : ℝ}

theorem proof_bd_leq_q2 
  (h1 : ab + cd = 2pq)
  (h2 : ac ≥ p^2)
  (h3 : p^2 > 0) :
  bd ≤ q^2 :=
sorry

end proof_bd_leq_q2_l298_298032


namespace calc_x6_plus_inv_x6_l298_298021

theorem calc_x6_plus_inv_x6 (x : ℝ) (hx : x + (1 / x) = 7) : x^6 + (1 / x^6) = 103682 := by
  sorry

end calc_x6_plus_inv_x6_l298_298021


namespace Carter_reads_30_pages_in_1_hour_l298_298322

variables (C L O : ℕ)

def Carter_reads_half_as_many_pages_as_Lucy_in_1_hour (C L : ℕ) : Prop :=
  C = L / 2

def Lucy_reads_20_more_pages_than_Oliver_in_1_hour (L O : ℕ) : Prop :=
  L = O + 20

def Oliver_reads_40_pages_in_1_hour (O : ℕ) : Prop :=
  O = 40

theorem Carter_reads_30_pages_in_1_hour
  (C L O : ℕ)
  (h1 : Carter_reads_half_as_many_pages_as_Lucy_in_1_hour C L)
  (h2 : Lucy_reads_20_more_pages_than_Oliver_in_1_hour L O)
  (h3 : Oliver_reads_40_pages_in_1_hour O) : 
  C = 30 :=
by
  sorry

end Carter_reads_30_pages_in_1_hour_l298_298322


namespace Tom_has_38_photos_l298_298438

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l298_298438


namespace raise_3000_yuan_probability_l298_298945

def prob_correct_1 : ℝ := 0.9
def prob_correct_2 : ℝ := 0.5
def prob_correct_3 : ℝ := 0.4
def prob_incorrect_3 : ℝ := 1 - prob_correct_3

def fund_first : ℝ := 1000
def fund_second : ℝ := 2000
def fund_third : ℝ := 3000

def prob_raise_3000_yuan : ℝ := prob_correct_1 * prob_correct_2 * prob_incorrect_3

theorem raise_3000_yuan_probability :
  prob_raise_3000_yuan = 0.27 :=
by
  sorry

end raise_3000_yuan_probability_l298_298945


namespace dihedral_angle_ge_l298_298802

-- Define the problem conditions and goal in Lean
theorem dihedral_angle_ge (n : ℕ) (h : 3 ≤ n) (ϕ : ℝ) :
  ϕ ≥ π * (1 - 2 / n) := 
sorry

end dihedral_angle_ge_l298_298802


namespace square_area_and_position_l298_298115

/-!
A square is given in the plane with consecutively placed vertices A, B, C, D and a point O.
It is known that OA = OC = 10, OD = 6√2, and the side length of the square does not exceed 3.
Find the area of the square. Is point O located outside or inside the square?
-/

-- Define the coordinates of the points and the required properties.
def point (α : Type*) := (x y : α)

variables {α : Type*} [linear_ordered_field α]
variables (A B C D O : point α)
variables (s : α)

-- Given conditions
def sq_vertices := -- (Angle and vertices construction)
s ≤ 3

def OA := (A.x - O.x)^2 + (A.y - O.y)^2 = 10^2
def OC := (C.x - O.x)^2 + (C.y - O.y)^2 = 10^2
def OD := (D.x - O.x)^2 + (D.y - O.y)^2 = (6 * real.sqrt 2)^2

-- Question to determine the area of the square and position of O
theorem square_area_and_position 
  (A B C D O : point ℝ)
  (s : ℝ)
  (OA : (A.x - O.x)^2 + (A.y - O.y)^2 = 100)
  (OC : (C.x - O.x)^2 + (C.y - O.y)^2 = 100)
  (OD : (D.x - O.x)^2 + (D.y - O.y)^2 = 72)
  (sq_vertices : s ≤ 3) :
  s^2 = 4 ∧ -- Area of the square
  (O.x, O.y) ∉ set.univ := -- O is outside the square
sorry

end square_area_and_position_l298_298115


namespace problem_statement_l298_298864

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem problem_statement : vec_dot (vec_add a (vec_scalar_mul 2 b)) c = -3 := 
by
  sorry

end problem_statement_l298_298864


namespace find_function_l298_298331

theorem find_function (f : ℕ → ℕ) (h : ∀ m n, f (m + f n) = f (f m) + f n) :
  ∃ d, d > 0 ∧ (∀ m, ∃ k, f m = k * d) :=
sorry

end find_function_l298_298331


namespace acid_concentration_third_flask_l298_298285

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l298_298285


namespace find_x_in_terms_of_abc_l298_298717

variable {x y z a b c : ℝ}

theorem find_x_in_terms_of_abc
  (h1 : xy / (x + y + 1) = a)
  (h2 : xz / (x + z + 1) = b)
  (h3 : yz / (y + z + 1) = c) :
  x = 2 * a * b * c / (a * b + a * c - b * c) := 
sorry

end find_x_in_terms_of_abc_l298_298717


namespace cos_pi_over_3_plus_2theta_l298_298340

theorem cos_pi_over_3_plus_2theta 
  (theta : ℝ)
  (h : Real.sin (Real.pi / 3 - theta) = 3 / 4) : 
  Real.cos (Real.pi / 3 + 2 * theta) = 1 / 8 :=
by 
  sorry

end cos_pi_over_3_plus_2theta_l298_298340


namespace time_to_meet_in_minutes_l298_298790

def distance_between_projectiles : ℕ := 1998
def speed_projectile_1 : ℕ := 444
def speed_projectile_2 : ℕ := 555

theorem time_to_meet_in_minutes : 
  (distance_between_projectiles / (speed_projectile_1 + speed_projectile_2)) * 60 = 120 := 
by
  sorry

end time_to_meet_in_minutes_l298_298790


namespace probability_of_selection_is_equal_l298_298337

-- Define the conditions of the problem
def total_students := 2004
def eliminated_students := 4
def remaining_students := total_students - eliminated_students -- 2000
def selected_students := 50
def k := remaining_students / selected_students -- 40

-- Define the probability calculation
def probability_selected := selected_students / remaining_students

-- The theorem stating that every student has a 1/40 probability of being selected
theorem probability_of_selection_is_equal :
  probability_selected = 1 / 40 :=
by
  -- insert proof logic here
  sorry

end probability_of_selection_is_equal_l298_298337


namespace length_of_train_a_l298_298631

theorem length_of_train_a
  (speed_train_a : ℝ) (speed_train_b : ℝ) 
  (clearing_time : ℝ) (length_train_b : ℝ)
  (h1 : speed_train_a = 42)
  (h2 : speed_train_b = 30)
  (h3 : clearing_time = 12.998960083193344)
  (h4 : length_train_b = 160) :
  ∃ length_train_a : ℝ, length_train_a = 99.9792016638669 :=
by 
  sorry

end length_of_train_a_l298_298631


namespace geometric_sequence_an_l298_298188

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 3 else 3 * (2:ℝ)^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 1 then 3 else (3 * (2:ℝ)^n - 3)

theorem geometric_sequence_an (n : ℕ) (h1 : a 1 = 3) (h2 : S 2 = 9) :
  a n = 3 * 2^(n-1) ∧ S n = 3 * (2^n - 1) :=
by
  sorry

end geometric_sequence_an_l298_298188


namespace equation_has_two_distinct_roots_l298_298157

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l298_298157


namespace company_pays_each_man_per_hour_l298_298215

theorem company_pays_each_man_per_hour
  (men : ℕ) (hours_per_job : ℕ) (jobs : ℕ) (total_pay : ℕ)
  (completion_time : men * hours_per_job = 1)
  (total_jobs_time : jobs * hours_per_job = 5)
  (total_earning : total_pay = 150) :
  (total_pay / (jobs * men * hours_per_job)) = 10 :=
sorry

end company_pays_each_man_per_hour_l298_298215


namespace problem_l298_298761

theorem problem (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (1 / 2) = 0) : 
  f (-201) = 403 :=
sorry

end problem_l298_298761


namespace perpendicular_lines_l298_298719

theorem perpendicular_lines (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), ((a + 1) * m₁ + a * m₂ = 0) ∧ 
                  (a * m₁ + 2 * m₂ = 1) ∧ 
                  m₁ * m₂ = -1) ↔ (a = 0 ∨ a = -3) := 
sorry

end perpendicular_lines_l298_298719


namespace original_cost_of_each_bag_l298_298692

theorem original_cost_of_each_bag (C : ℕ) (hC : C % 13 = 0) (h4 : (85 * C) % 400 = 0) : C / 5 = 208 := by
  sorry

end original_cost_of_each_bag_l298_298692


namespace greatest_x_lcm_105_l298_298920

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298920


namespace three_zeros_implies_a_lt_neg3_l298_298561

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298561


namespace smallest_integer_with_20_divisors_l298_298651

theorem smallest_integer_with_20_divisors :
  ∃ n : ℕ, (∀ k : ℕ, k ∣ n → k > 0) ∧ n = 432 ∧ (∃ (p1 p2 : ℕ) (a1 a2 : ℕ),
    p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ (a1 + 1) * (a2 + 1) = 20 ∧ n = p1^a1 * p2^a2) :=
sorry

end smallest_integer_with_20_divisors_l298_298651


namespace solution_set_f_x_minus_1_lt_0_l298_298043

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x - 1 else -x - 1

theorem solution_set_f_x_minus_1_lt_0 :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_f_x_minus_1_lt_0_l298_298043


namespace cone_base_radius_half_l298_298766

theorem cone_base_radius_half :
  let R : ℝ := sorry
  let semicircle_radius : ℝ := 1
  let unfolded_circumference : ℝ := π
  let base_circumference : ℝ := 2 * π * R
  base_circumference = unfolded_circumference -> R = 1 / 2 :=
by
  sorry

end cone_base_radius_half_l298_298766


namespace part1_part2_l298_298341

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

theorem part1 : {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} := sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x + a * x - 1 > 0) → a > -5/2 := sorry

end part1_part2_l298_298341


namespace piecewise_linear_function_y_at_x_10_l298_298226

theorem piecewise_linear_function_y_at_x_10
  (k1 k2 : ℝ)
  (y : ℝ → ℝ)
  (hx1 : ∀ x < 0, y x = k1 * x)
  (hx2 : ∀ x ≥ 0, y x = k2 * x)
  (h_y_pos : y 2 = 4)
  (h_y_neg : y (-5) = -20) :
  y 10 = 20 :=
by
  sorry

end piecewise_linear_function_y_at_x_10_l298_298226


namespace total_sum_money_l298_298304

theorem total_sum_money (a b c : ℝ) (h1 : b = 0.65 * a) (h2 : c = 0.40 * a) (h3 : c = 64) :
  a + b + c = 328 :=
by
  sorry

end total_sum_money_l298_298304


namespace range_of_a_for_three_zeros_l298_298540

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298540


namespace prove_mouse_cost_l298_298007

variable (M K : ℕ)

theorem prove_mouse_cost (h1 : K = 3 * M) (h2 : M + K = 64) : M = 16 :=
by
  sorry

end prove_mouse_cost_l298_298007


namespace log_sum_nine_l298_298429

-- Define that {a_n} is a geometric sequence and satisfies the given conditions.
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * r ^ (n - 1)

-- Given conditions
axiom a_pos (a : ℕ → ℝ) : (∀ n, a n > 0)      -- All terms are positive
axiom a2a8_eq_4 (a : ℕ → ℝ) : a 2 * a 8 = 4    -- a₂a₈ = 4

theorem log_sum_nine (a : ℕ → ℝ) 
  (geo_seq : geometric_sequence a) 
  (pos : ∀ n, a n > 0)
  (eq4 : a 2 * a 8 = 4) :
  (Real.logb 2 (a 1) + Real.logb 2 (a 2) + Real.logb 2 (a 3) + Real.logb 2 (a 4)
  + Real.logb 2 (a 5) + Real.logb 2 (a 6) + Real.logb 2 (a 7) + Real.logb 2 (a 8)
  + Real.logb 2 (a 9)) = 9 :=
by
  sorry

end log_sum_nine_l298_298429


namespace eval_polynomial_at_neg2_l298_298985

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

-- Statement of the problem, proving that the polynomial equals 11 when x = -2
theorem eval_polynomial_at_neg2 : polynomial (-2) = 11 := by
  sorry

end eval_polynomial_at_neg2_l298_298985


namespace solve_for_x_l298_298048

theorem solve_for_x (x: ℚ) (h: (3/5 - 1/4) = 4/x) : x = 80/7 :=
by
  sorry

end solve_for_x_l298_298048


namespace range_of_a_for_three_zeros_l298_298541

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298541


namespace factorize_polynomial_l298_298689

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem factorize_polynomial :
  (zeta^3 = 1) ∧ (zeta^2 + zeta + 1 = 0) → (x : ℂ) → (x^15 + x^10 + x) = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1)
:= sorry

end factorize_polynomial_l298_298689


namespace number_of_houses_around_square_l298_298873

namespace HouseCounting

-- Definitions for the conditions
def M (k : ℕ) : ℕ := k
def J (k : ℕ) : ℕ := k

-- The main theorem stating the solution
theorem number_of_houses_around_square (n : ℕ)
  (h1 : M 5 % n = J 12 % n)
  (h2 : J 5 % n = M 30 % n) : n = 32 :=
sorry

end HouseCounting

end number_of_houses_around_square_l298_298873


namespace part1_part2_l298_298997

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 1 < x ∧ x < 5 }
def setC (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 3 }

-- part (1)
theorem part1 (x : ℝ) : (x ∈ setA ∨ x ∈ setB) ↔ (-2 ≤ x ∧ x < 5) :=
sorry

-- part (2)
theorem part2 (a : ℝ) : ((setA ∩ setC a) = setC a) ↔ (a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2)) :=
sorry

end part1_part2_l298_298997


namespace angle_leq_60_degrees_l298_298417

-- Define the conditions
variables {a b c : ℝ} (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
variables (h_geometric_mean : a^2 = b * c)
variables (α : ℝ) (h_cosine_rule : cos α = (b^2 + c^2 - a^2) / (2 * b * c))

-- State the theorem to be proven
theorem angle_leq_60_degrees (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) (h_geometric_mean : a^2 = b * c) 
  (α : ℝ) (h_cosine_rule : cos α = (b^2 + c^2 - a^2) / (2 * b * c)) : 
  α ≤ (60 : ℝ) :=
  sorry

end angle_leq_60_degrees_l298_298417


namespace at_least_one_solves_l298_298237

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]

-- Define the events
variable (A B : Event Ω)

-- Define the given probabilities
variable (hA : P(A) = 0.5) (hB : P(B) = 0.4)

-- Define the probability of at least one event occurring
def prob_one_solves : Prop :=
  P(A ∪ B) = 0.7

-- Theorem statement
theorem at_least_one_solves (Ω : Type) [ProbabilitySpace Ω] (A B : Event Ω)
  (hA : P(A) = 0.5) (hB : P(B) = 0.4) : prob_one_solves Ω A B :=
by
  unfold prob_one_solves
  sorry

end at_least_one_solves_l298_298237


namespace geometric_sequence_sum_l298_298938

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℚ) (a : ℚ) :
  (∀ n, S n = (1 / 2) * 3^(n + 1) - a) →
  S 1 - (S 2 - S 1)^2 = (S 2 - S 1) * (S 3 - S 2) →
  a = 3 / 2 :=
by
  intros hSn hgeo
  sorry

end geometric_sequence_sum_l298_298938


namespace floor_div_eq_floor_div_l298_298223

theorem floor_div_eq_floor_div
  (a : ℝ) (n : ℤ) (ha_pos : 0 < a) :
  (⌊⌊a⌋ / n⌋ : ℤ) = ⌊a / n⌋ := 
sorry

end floor_div_eq_floor_div_l298_298223


namespace factor_polynomial_l298_298327

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l298_298327


namespace four_digit_cubes_divisible_by_16_l298_298711

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l298_298711


namespace solve_fraction_eq_zero_l298_298367

theorem solve_fraction_eq_zero (x : ℝ) (h : x - 2 ≠ 0) : (x + 1) / (x - 2) = 0 ↔ x = -1 :=
by
  sorry

end solve_fraction_eq_zero_l298_298367


namespace range_of_a_if_f_has_three_zeros_l298_298554

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298554


namespace tom_walking_distance_l298_298857

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end tom_walking_distance_l298_298857


namespace original_number_of_girls_l298_298493

theorem original_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 4 * (b - 60) = g - 20) : 
  g = 460 / 11 :=
by
  sorry

end original_number_of_girls_l298_298493


namespace emily_total_cost_l298_298667

-- Definition of the monthly cell phone plan costs and usage details
def base_cost : ℝ := 30
def cost_per_text : ℝ := 0.10
def cost_per_extra_minute : ℝ := 0.15
def cost_per_extra_gb : ℝ := 5
def free_hours : ℝ := 25
def free_gb : ℝ := 15
def texts : ℝ := 150
def hours : ℝ := 26
def gb : ℝ := 16

-- Calculate the total cost
def total_cost : ℝ :=
  base_cost +
  (texts * cost_per_text) +
  ((hours - free_hours) * 60 * cost_per_extra_minute) +
  ((gb - free_gb) * cost_per_extra_gb)

-- The proof statement that Emily had to pay $59
theorem emily_total_cost :
  total_cost = 59 := by
  sorry

end emily_total_cost_l298_298667


namespace number_of_3_letter_words_with_at_least_one_A_l298_298362

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end number_of_3_letter_words_with_at_least_one_A_l298_298362


namespace greatest_x_lcm_l298_298929

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298929


namespace pet_store_cats_left_l298_298295

theorem pet_store_cats_left (siamese house sold : ℕ) (h_siamese : siamese = 38) (h_house : house = 25) (h_sold : sold = 45) :
  siamese + house - sold = 18 :=
by
  sorry

end pet_store_cats_left_l298_298295


namespace smallest_integer_with_20_divisors_l298_298637

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end smallest_integer_with_20_divisors_l298_298637


namespace circle_equation_l298_298690

open Real

theorem circle_equation (x y : ℝ) :
  let center := (2, -1)
  let line := (x + y = 7)
  (center.1 - 2)^2 + (center.2 + 1)^2 = 18 :=
by
  sorry

end circle_equation_l298_298690


namespace Evan_dog_weight_l298_298822

-- Define the weights of the dogs as variables
variables (E I : ℕ)

-- Conditions given in the problem
def Evan_dog_weight_wrt_Ivan (I : ℕ) : ℕ := 7 * I
def dogs_total_weight (E I : ℕ) : Prop := E + I = 72

-- Correct answer we need to prove
theorem Evan_dog_weight (h1 : Evan_dog_weight_wrt_Ivan I = E)
                          (h2 : dogs_total_weight E I)
                          (h3 : I = 9) : E = 63 :=
by
  sorry

end Evan_dog_weight_l298_298822


namespace range_of_f_l298_298194

noncomputable def f : ℝ → ℝ := sorry

-- The conditions given in the problem
axiom f_deriv : ∀ x : ℝ, has_deriv_at f (deriv f x) x 
axiom f_zero : f 0 = 2
axiom f_deriv_ineq : ∀ x : ℝ, deriv f x - f x > exp x

-- The theorem statement proving the range of x
theorem range_of_f (x : ℝ) (h : x > 0) : f x > x * exp x + 2 * exp x := sorry

end range_of_f_l298_298194


namespace complex_magnitude_l298_298193

variable (a b : ℝ)

theorem complex_magnitude :
  ((1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I) →
  Complex.normSq (a + b * Complex.I) = 5/4 :=
by
  intro h
  -- Add missing logic to transform assumption to the norm result
  sorry

end complex_magnitude_l298_298193


namespace count_three_letter_words_with_A_l298_298363

theorem count_three_letter_words_with_A : 
  let total_words := 5^3 in
  let words_without_A := 4^3 in
  total_words - words_without_A = 61 :=
by
  sorry

end count_three_letter_words_with_A_l298_298363


namespace union_A_B_m_eq_3_range_of_m_l298_298839

def A (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def B (x m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem union_A_B_m_eq_3 :
  A x ∨ B x 3 ↔ (-3 : ℝ) ≤ x ∧ x ≤ 5 := sorry

theorem range_of_m (h : ∀ x, A x ∨ B x m ↔ A x) : m ≤ (5 / 2) := sorry

end union_A_B_m_eq_3_range_of_m_l298_298839


namespace smallest_possible_l_l298_298966

theorem smallest_possible_l (a b c L : ℕ) (h1 : a * b = 7) (h2 : a * c = 27) (h3 : b * c = L) (h4 : ∃ k, a * b * c = k * k) : L = 21 := sorry

end smallest_possible_l_l298_298966


namespace solve_quadratic_1_solve_quadratic_2_l298_298078

theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 8 * x + 4 = 0 ↔ x = 2/3 ∨ x = 2 := by
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1)^2 = (x - 3)^2 ↔ x = 4/3 ∨ x = -2 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l298_298078


namespace greatest_divisor_l298_298293

theorem greatest_divisor (d : ℕ) :
  (1657 % d = 6 ∧ 2037 % d = 5) → d = 127 := by
  sorry

end greatest_divisor_l298_298293


namespace correct_actual_profit_l298_298463

def profit_miscalculation (calculated_profit actual_profit : ℕ) : Prop :=
  let err1 := 5 * 100  -- Error due to mistaking 3 for 8 in the hundreds place
  let err2 := 3 * 10   -- Error due to mistaking 8 for 5 in the tens place
  actual_profit = calculated_profit - err1 + err2

theorem correct_actual_profit : profit_miscalculation 1320 850 :=
by
  sorry

end correct_actual_profit_l298_298463


namespace number_of_red_balls_l298_298061

theorem number_of_red_balls (total_balls : ℕ) (prob_red : ℚ) (h : total_balls = 20 ∧ prob_red = 0.25) : ∃ x : ℕ, x = 5 :=
by
  sorry

end number_of_red_balls_l298_298061


namespace range_of_a_l298_298524

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298524


namespace perpendicular_lines_intersection_l298_298767

theorem perpendicular_lines_intersection (a b c d : ℝ)
    (h_perpendicular : (a / 2) * (-2 / b) = -1)
    (h_intersection1 : a * 2 - 2 * (-3) = d)
    (h_intersection2 : 2 * 2 + b * (-3) = c) :
    d = 12 := 
sorry

end perpendicular_lines_intersection_l298_298767


namespace alex_jellybeans_l298_298123

theorem alex_jellybeans (x : ℕ) : x = 254 → x ≥ 150 ∧ x % 15 = 14 ∧ x % 17 = 16 :=
by
  sorry

end alex_jellybeans_l298_298123


namespace quadratic_roots_l298_298619

theorem quadratic_roots (m x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1*x1 + m*x1 + 2*m = 0) (h3 : x2*x2 + m*x2 + 2*m = 0) : x1 * x2 = -2 := 
by sorry

end quadratic_roots_l298_298619


namespace maria_green_beans_l298_298406

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l298_298406


namespace min_sum_ab_l298_298068

theorem min_sum_ab {a b : ℤ} (h : a * b = 36) : a + b ≥ -37 := sorry

end min_sum_ab_l298_298068


namespace milk_production_l298_298424

theorem milk_production (a b c d e f : ℕ) (h₁ : a > 0) (h₂ : c > 0) (h₃ : f > 0) : 
  ((d * e * b * f) / (100 * a * c)) = (d * e * b * f / (100 * a * c)) :=
by
  sorry

end milk_production_l298_298424


namespace least_possible_value_m_n_l298_298739

theorem least_possible_value_m_n :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd (m + n) 330 = 1 ∧ n ∣ m^m ∧ ¬(m % n = 0) ∧ (m + n = 377) :=
by
  sorry

end least_possible_value_m_n_l298_298739


namespace rectangle_area_l298_298791

theorem rectangle_area (b l : ℕ) (P : ℕ) (h1 : l = 3 * b) (h2 : P = 64) (h3 : P = 2 * (l + b)) :
  l * b = 192 :=
by
  sorry

end rectangle_area_l298_298791


namespace a4_binomial_coefficient_l298_298423

theorem a4_binomial_coefficient :
  ∀ (a_n a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ),
  (x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (x^5 = (1 + (x - 1))^5) →
  a_4 = 5 :=
by
  intros a_n a_1 a_2 a_3 a_4 a_5 x hx1 hx2
  sorry

end a4_binomial_coefficient_l298_298423


namespace problem_1_problem_2_l298_298199

def A := {x : ℝ | 1 < 2 * x - 1 ∧ 2 * x - 1 < 7}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}

theorem problem_1 : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem problem_2 : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 4} :=
sorry

end problem_1_problem_2_l298_298199


namespace smallest_integer_with_20_divisors_l298_298647

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, (n > 0 ∧ (∃ (d : ℕ → Prop), (∀ m, d m ↔ m ∣ n) ∧ (card { m : ℕ | d m } = 20)) ∧ (∀ k : ℕ, k > 0 ∧ (∃ (d' : ℕ → Prop), (∀ m, d' m ↔ m ∣ k) ∧ (card { m : ℕ | d' m } = 20)) → k ≥ n)) ∧ n = 240 :=
by { sorry }

end smallest_integer_with_20_divisors_l298_298647


namespace value_of_x_l298_298262

-- Define the variables x, y, z
variables (x y z : ℕ)

-- Hypothesis based on the conditions of the problem
hypothesis h1 : x = y / 3
hypothesis h2 : y = z / 4
hypothesis h3 : z = 48

-- The statement to be proved
theorem value_of_x : x = 4 :=
by { sorry }

end value_of_x_l298_298262


namespace function_has_three_zeros_l298_298535

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298535


namespace tangent_line_at_2_eq_l298_298004

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2_eq :
  let x := (2 : ℝ)
  let slope := (deriv f) x
  let y := f x
  ∃ (m y₀ : ℝ), m = slope ∧ y₀ = y ∧ 
    (∀ (x y : ℝ), y = m * (x - 2) + y₀ → x - y - 4 = 0)
:= sorry

end tangent_line_at_2_eq_l298_298004


namespace digit_y_in_base_7_divisible_by_19_l298_298003

def base7_to_decimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem digit_y_in_base_7_divisible_by_19 (y : ℕ) (hy : y < 7) :
  (∃ k : ℕ, base7_to_decimal 5 2 y 3 = 19 * k) ↔ y = 8 :=
by {
  sorry
}

end digit_y_in_base_7_divisible_by_19_l298_298003


namespace total_distance_total_distance_alt_l298_298311

variable (D : ℝ) -- declare the variable for the total distance

-- defining the conditions
def speed_walking : ℝ := 4 -- speed in km/hr when walking
def speed_running : ℝ := 8 -- speed in km/hr when running
def total_time : ℝ := 3.75 -- total time in hours

-- proving that D = 10 given the conditions
theorem total_distance 
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time) : 
    D = 10 := 
sorry

-- Alternative theorem version declaring variables directly
theorem total_distance_alt
    (speed_walking speed_running total_time : ℝ) -- declaring variables
    (D : ℝ) -- the total distance
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time)
    (hw : speed_walking = 4)
    (hr : speed_running = 8)
    (ht : total_time = 3.75) : 
    D = 10 := 
sorry

end total_distance_total_distance_alt_l298_298311


namespace point_P_in_fourth_quadrant_l298_298349

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_P_in_fourth_quadrant (m : ℝ) : point_in_fourth_quadrant (1 + m^2) (-1) :=
by
  sorry

end point_P_in_fourth_quadrant_l298_298349


namespace possible_values_of_reciprocal_sum_l298_298397

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end possible_values_of_reciprocal_sum_l298_298397


namespace range_of_2a_minus_b_l298_298999

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2 * a^2 - a * b - b^2 - 4 = 0) :
  (2 * a - b) ∈ (Set.Ici (8 / 3)) :=
sorry

end range_of_2a_minus_b_l298_298999


namespace expand_and_simplify_l298_298986

theorem expand_and_simplify :
  ∀ x : ℝ, (x^3 - 3*x + 3)*(x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by sorry

end expand_and_simplify_l298_298986


namespace other_acute_angle_in_right_triangle_l298_298212

theorem other_acute_angle_in_right_triangle (α : ℝ) (β : ℝ) (γ : ℝ) 
  (h1 : α + β + γ = 180) (h2 : γ = 90) (h3 : α = 30) : β = 60 := 
sorry

end other_acute_angle_in_right_triangle_l298_298212


namespace child_ticket_cost_is_2_l298_298664

-- Define the conditions
def adult_ticket_cost : ℕ := 5
def total_tickets_sold : ℕ := 85
def total_revenue : ℕ := 275
def adult_tickets_sold : ℕ := 35

-- Define the function to calculate child ticket cost
noncomputable def child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets_sold : ℕ) (total_revenue : ℕ) (adult_tickets_sold : ℕ) : ℕ :=
  let total_adult_revenue := adult_tickets_sold * adult_ticket_cost
  let total_child_revenue := total_revenue - total_adult_revenue
  let child_tickets_sold := total_tickets_sold - adult_tickets_sold
  total_child_revenue / child_tickets_sold

theorem child_ticket_cost_is_2 : child_ticket_cost adult_ticket_cost total_tickets_sold total_revenue adult_tickets_sold = 2 := 
by
  -- This is a placeholder for the actual proof which we can fill in separately.
  sorry

end child_ticket_cost_is_2_l298_298664


namespace max_x_lcm_15_21_105_l298_298909

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298909


namespace gumball_problem_l298_298842

theorem gumball_problem:
  ∀ (total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs: ℕ),
    total_gumballs = 45 →
    given_to_Todd = 4 →
    given_to_Alisha = 2 * given_to_Todd →
    remaining_gumballs = 6 →
    given_to_Todd + given_to_Alisha + given_to_Bobby + remaining_gumballs = total_gumballs →
    given_to_Bobby = 45 - 18 →
    4 * given_to_Alisha - given_to_Bobby = 5 :=
by
  intros total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs ht hTodd hAlisha hRemaining hSum hBobby
  rw [ht, hTodd] at *
  rw [hAlisha, hRemaining] at *
  sorry

end gumball_problem_l298_298842


namespace expand_binomial_l298_298490

theorem expand_binomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11 * x + 24 :=
by sorry

end expand_binomial_l298_298490


namespace sufficient_not_necessary_condition_l298_298464

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 2) → ((x + 1) * (x - 2) > 0) ∧ ¬(∀ y, (y + 1) * (y - 2) > 0 → y > 2) := 
sorry

end sufficient_not_necessary_condition_l298_298464


namespace necessary_but_not_sufficient_condition_l298_298403

variables {a b c : ℝ × ℝ}

def nonzero_vector (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

theorem necessary_but_not_sufficient_condition (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c) :
  (a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) ↔ (b = c) :=
sorry

end necessary_but_not_sufficient_condition_l298_298403


namespace successful_purchase_probability_l298_298684

theorem successful_purchase_probability
  (m n : ℕ)
  (h : m ≥ n) :
  let total_ways := Nat.choose (m + n) m,
      favorable_ways := Nat.choose (m + n) m - Nat.choose (m + n) (m + 1) in
  (favorable_ways / total_ways : ℚ) = (m - n + 1) / (m + 1) := by
  sorry

end successful_purchase_probability_l298_298684


namespace greatest_x_lcm_l298_298903

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298903


namespace abe_age_sum_l298_298776

theorem abe_age_sum (x : ℕ) : 25 + (25 - x) = 29 ↔ x = 21 :=
by sorry

end abe_age_sum_l298_298776


namespace monthly_expenses_last_month_was_2888_l298_298885

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l298_298885


namespace polygon_side_count_l298_298775

theorem polygon_side_count (n : ℕ) 
    (h : (n - 2) * 180 + 1350 - (n - 2) * 180 = 1350) : n = 9 :=
by
  sorry

end polygon_side_count_l298_298775


namespace my_op_evaluation_l298_298716

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end my_op_evaluation_l298_298716


namespace matching_red_pair_probability_l298_298093

def total_socks := 8
def red_socks := 4
def blue_socks := 2
def green_socks := 2

noncomputable def total_pairs := Nat.choose total_socks 2
noncomputable def red_pairs := Nat.choose red_socks 2
noncomputable def blue_pairs := Nat.choose blue_socks 2
noncomputable def green_pairs := Nat.choose green_socks 2
noncomputable def total_matching_pairs := red_pairs + blue_pairs + green_pairs
noncomputable def probability_red := (red_pairs : ℚ) / total_matching_pairs

theorem matching_red_pair_probability : probability_red = 3 / 4 :=
  by sorry

end matching_red_pair_probability_l298_298093


namespace exists_2016_integers_with_product_9_and_sum_0_l298_298389

theorem exists_2016_integers_with_product_9_and_sum_0 :
  ∃ (L : List ℤ), L.length = 2016 ∧ L.prod = 9 ∧ L.sum = 0 := by
  sorry

end exists_2016_integers_with_product_9_and_sum_0_l298_298389


namespace four_digit_cubes_divisible_by_16_l298_298712

theorem four_digit_cubes_divisible_by_16 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.finite
  ∧ ∃ n, n = 3 ∧ {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ∃ k : ℕ, x = k^3 ∧ 16 ∣ x}.card = n := 
by
  -- The proof steps would go here.
  sorry

end four_digit_cubes_divisible_by_16_l298_298712


namespace average_of_multiples_of_6_l298_298094

def first_n_multiples_sum (n : ℕ) : ℕ :=
  (n * (6 + 6 * n)) / 2

def first_n_multiples_avg (n : ℕ) : ℕ :=
  (first_n_multiples_sum n) / n

theorem average_of_multiples_of_6 (n : ℕ) : first_n_multiples_avg n = 66 → n = 11 := by
  sorry

end average_of_multiples_of_6_l298_298094


namespace binomial_equality_l298_298749

theorem binomial_equality (k : ℕ) :
  (∑ i in Finset.range (4 * k + 1), binomial (4 * k) i * (-3) ^ i) =
  (∑ j in Finset.range (2 * k + 1), binomial (2 * k) j * (-5) ^ j) :=
by 
  sorry

end binomial_equality_l298_298749


namespace equivalent_problem_l298_298979

theorem equivalent_problem :
  let a : ℤ := (-6)
  let b : ℤ := 6
  let c : ℤ := 2
  let d : ℤ := 4
  (a^4 / b^2 - c^5 + d^2 = 20) :=
by
  sorry

end equivalent_problem_l298_298979


namespace range_of_a_for_three_zeros_l298_298523

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298523


namespace total_time_spent_in_hours_l298_298743

/-- Miriam's time spent on each task in minutes. -/
def time_laundry := 30
def time_bathroom := 15
def time_room := 35
def time_homework := 40

/-- The function to convert minutes to hours. -/
def minutes_to_hours (minutes : ℕ) := minutes / 60

/-- The total time spent in minutes. -/
def total_time_minutes := time_laundry + time_bathroom + time_room + time_homework

/-- The total time spent in hours. -/
def total_time_hours := minutes_to_hours total_time_minutes

/-- The main statement to be proved: total_time_hours equals 2. -/
theorem total_time_spent_in_hours : total_time_hours = 2 := 
by
  sorry

end total_time_spent_in_hours_l298_298743


namespace repeated_two_digit_number_divisible_by_101_l298_298474

theorem repeated_two_digit_number_divisible_by_101 (a b : ℕ) :
  (10 ≤ a ∧ a ≤ 99 ∧ 0 ≤ b ∧ b ≤ 9) →
  ∃ k, (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) = 101 * k :=
by
  intro h
  sorry

end repeated_two_digit_number_divisible_by_101_l298_298474


namespace fraction_of_roll_used_l298_298243

theorem fraction_of_roll_used 
  (x : ℚ) 
  (h1 : 3 * x + 3 * x + x + 2 * x = 9 * x)
  (h2 : 9 * x = (2 / 5)) : 
  x = 2 / 45 :=
by
  sorry

end fraction_of_roll_used_l298_298243


namespace smallest_lcm_of_4_digit_integers_l298_298377

open Nat

theorem smallest_lcm_of_4_digit_integers (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h_gcd : gcd k l = 5) :
  lcm k l = 203010 := sorry

end smallest_lcm_of_4_digit_integers_l298_298377


namespace sample_systematic_draw_first_group_l298_298440

theorem sample_systematic_draw_first_group :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 8 →
  (x + 15 * 8 = 126) →
  x = 6 :=
by
  intros x h1 h2
  sorry

end sample_systematic_draw_first_group_l298_298440


namespace probability_correct_l298_298779

-- Define the set and the probability calculation
def set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if the difference condition holds
def valid_triplet (a b c: ℕ) : Prop := a < b ∧ b < c ∧ c - a = 4

-- Total number of ways to pick 3 numbers and ways that fit the condition
noncomputable def total_ways : ℕ := Nat.choose 9 3
noncomputable def valid_ways : ℕ := 5 * 2

-- Calculate the probability
noncomputable def probability : ℚ := valid_ways / total_ways

-- The theorem statement
theorem probability_correct : probability = 5 / 42 := by sorry

end probability_correct_l298_298779


namespace three_digit_multiples_of_three_count_l298_298973

open Finset

noncomputable def num_three_digit_multiples_of_three : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000 in
  let is_multiple_of_3 (n : ℕ) := n % 3 = 0 in
  let choices := digits.to_list.perms.erase_dup.filter 
    (λ l, l.length = 3 ∧ is_three_digit (10 * (l.nth 0).get_or_else 0 + 10 * (l.nth 1).get_or_else 0 + (l.nth 2).get_or_else 0) ∧ 
          is_multiple_of_3 (10 * (l.nth 0).get_or_else 0 + 10 * (l.nth 1).get_or_else 0 + (l.nth 2).get_or_else 0)) in
  choices.length

theorem three_digit_multiples_of_three_count : num_three_digit_multiples_of_three = 228 :=
by sorry

end three_digit_multiples_of_three_count_l298_298973


namespace greatest_x_lcm_105_l298_298923

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298923


namespace p_add_inv_p_gt_two_l298_298416

theorem p_add_inv_p_gt_two {p : ℝ} (hp_pos : p > 0) (hp_neq_one : p ≠ 1) : p + 1 / p > 2 :=
by
  sorry

end p_add_inv_p_gt_two_l298_298416


namespace concentration_of_acid_in_third_flask_is_correct_l298_298268

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l298_298268


namespace percentage_of_men_speaking_french_l298_298794

theorem percentage_of_men_speaking_french {total_employees men women french_speaking_employees french_speaking_women french_speaking_men : ℕ}
    (h1 : total_employees = 100)
    (h2 : men = 60)
    (h3 : women = 40)
    (h4 : french_speaking_employees = 50)
    (h5 : french_speaking_women = 14)
    (h6 : french_speaking_men = french_speaking_employees - french_speaking_women)
    (h7 : french_speaking_men * 100 / men = 60) : true :=
by
  sorry

end percentage_of_men_speaking_french_l298_298794


namespace concentration_in_third_flask_l298_298277

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l298_298277


namespace divisors_not_multiples_of_14_l298_298222

theorem divisors_not_multiples_of_14 (m : ℕ)
  (h1 : ∃ k : ℕ, m = 2 * k ∧ (k : ℕ) * k = m / 2)  
  (h2 : ∃ k : ℕ, m = 3 * k ∧ (k : ℕ) * k * k = m / 3)  
  (h3 : ∃ k : ℕ, m = 7 * k ∧ (k : ℕ) ^ 7 = m / 7) : 
  let total_divisors := (6 + 1) * (10 + 1) * (7 + 1)
  let divisors_divisible_by_14 := (5 + 1) * (10 + 1) * (6 + 1)
  total_divisors - divisors_divisible_by_14 = 154 :=
by
  sorry

end divisors_not_multiples_of_14_l298_298222


namespace least_positive_integer_l298_298011

theorem least_positive_integer (x : ℕ) (h : x + 5600 ≡ 325 [MOD 15]) : x = 5 :=
sorry

end least_positive_integer_l298_298011


namespace find_arithmetic_progression_terms_l298_298989

noncomputable def arithmetic_progression_terms (a1 a2 a3 : ℕ) (d : ℕ) 
  (condition1 : a1 + (a1 + d) = 3 * 2^2) 
  (condition2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) : Prop := 
  a1 = 3 ∧ a2 = 9 ∧ a3 = 15

theorem find_arithmetic_progression_terms
  (a1 a2 a3 : ℕ) (d : ℕ)
  (cond1 : a1 + (a1 + d) = 3 * 2^2)
  (cond2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) :
  arithmetic_progression_terms a1 a2 a3 d cond1 cond2 :=
sorry

end find_arithmetic_progression_terms_l298_298989


namespace area_of_square_l298_298807

-- Define the diagonal length condition.
def diagonal_length : ℝ := 12 * real.sqrt 2

-- Define the side length of the square computed from the diagonal using the 45-45-90 triangle property.
def side_length : ℝ := diagonal_length / real.sqrt 2

-- Define the area of the square in terms of its side length.
def square_area : ℝ := side_length * side_length

-- Prove that the area is indeed 144 square centimeters.
theorem area_of_square (d : ℝ) (h : d = 12 * real.sqrt 2) : (d / real.sqrt 2) * (d / real.sqrt 2) = 144 :=
by
  rw [h, ←real.mul_div_cancel (12 * real.sqrt 2) (real.sqrt 2)],
  { norm_num },
  { exact real.sqrt_ne_zero'.2 (by norm_num) }

end area_of_square_l298_298807


namespace cubic_has_three_zeros_l298_298544

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298544


namespace valid_a_value_l298_298165

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l298_298165


namespace line_is_x_axis_l298_298849

theorem line_is_x_axis (A B C : ℝ) (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : A = 0 ∧ B ≠ 0 ∧ C = 0 :=
by sorry

end line_is_x_axis_l298_298849


namespace burn_all_bridges_mod_1000_l298_298483

theorem burn_all_bridges_mod_1000 :
  let m := 2013 * 2 ^ 2012
  let n := 3 ^ 2012
  (m + n) % 1000 = 937 :=
by
  sorry

end burn_all_bridges_mod_1000_l298_298483


namespace tom_walks_distance_l298_298858

theorem tom_walks_distance (t : ℝ) (d : ℝ) :
  t = 15 ∧ d = (1 / 18) * t → d ≈ 0.8 :=
by
  sorry

end tom_walks_distance_l298_298858


namespace investment_B_l298_298478

theorem investment_B {x : ℝ} :
  let a_investment := 6300
  let c_investment := 10500
  let total_profit := 12100
  let a_share_profit := 3630
  (6300 / (6300 + x + 10500) = 3630 / 12100) →
  x = 13650 :=
by { sorry }

end investment_B_l298_298478


namespace function_y_neg3x_plus_1_quadrants_l298_298018

theorem function_y_neg3x_plus_1_quadrants :
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x + 1) ∧ (
    (x < 0 ∧ y > 0) ∨ -- Second quadrant
    (x > 0 ∧ y > 0) ∨ -- First quadrant
    (x > 0 ∧ y < 0)   -- Fourth quadrant
  )
:= sorry

end function_y_neg3x_plus_1_quadrants_l298_298018


namespace experts_expected_points_probability_fifth_envelope_l298_298592

theorem experts_expected_points (n : ℕ) (h1 : n = 100) (h2 : n = 13) :
  ∃ e : ℚ, e = 465 :=
sorry

theorem probability_fifth_envelope (m : ℕ) (h1 : m = 13) :
  ∃ p : ℚ, p = 0.715 :=
sorry

end experts_expected_points_probability_fifth_envelope_l298_298592


namespace arithmetic_sequence_sum_l298_298024

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (a1 d : ℕ) 
  (h1 : a1 + d = 6) 
  (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) 
  (h3 : d ≠ 0) : 
  S_n a1 d 8 = 88 := 
by 
  sorry

end arithmetic_sequence_sum_l298_298024


namespace value_of_a_with_two_distinct_roots_l298_298175

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l298_298175


namespace simplify_sqrt_expression_l298_298247

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 :=
by
  sorry

end simplify_sqrt_expression_l298_298247


namespace concentration_of_acid_in_third_flask_is_correct_l298_298269

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l298_298269


namespace ticket_cost_is_25_l298_298803

-- Define the given conditions
def num_tickets_first_show : ℕ := 200
def num_tickets_second_show : ℕ := 3 * num_tickets_first_show
def total_tickets : ℕ := num_tickets_first_show + num_tickets_second_show
def total_revenue_in_dollars : ℕ := 20000

-- Claim to prove
theorem ticket_cost_is_25 : ∃ x : ℕ, total_tickets * x = total_revenue_in_dollars ∧ x = 25 :=
by
  -- sorry is used here to skip the proof
  sorry

end ticket_cost_is_25_l298_298803


namespace adam_and_simon_distance_l298_298122

theorem adam_and_simon_distance :
  ∀ (t : ℝ), (10 * t)^2 + (12 * t)^2 = 16900 → t = 65 / Real.sqrt 61 :=
by
  sorry

end adam_and_simon_distance_l298_298122


namespace value_of_expression_l298_298371

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l298_298371


namespace equation_has_roots_l298_298154

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l298_298154


namespace number_of_parallelograms_l298_298770

-- Given conditions
def num_horizontal_lines : ℕ := 4
def num_vertical_lines : ℕ := 4

-- Mathematical function for combinations
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Proof statement
theorem number_of_parallelograms :
  binom num_horizontal_lines 2 * binom num_vertical_lines 2 = 36 :=
by
  sorry

end number_of_parallelograms_l298_298770


namespace range_of_a_for_three_zeros_l298_298520

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298520


namespace sqrt_4_eq_2_or_neg2_l298_298774

theorem sqrt_4_eq_2_or_neg2 (y : ℝ) (h : y^2 = 4) : y = 2 ∨ y = -2 :=
sorry

end sqrt_4_eq_2_or_neg2_l298_298774


namespace find_y_l298_298110

open Real

theorem find_y : ∃ y : ℝ, (sqrt ((3 - (-5))^2 + (y - 4)^2) = 12) ∧ (y > 0) ∧ (y = 4 + 4 * sqrt 5) :=
by
  use 4 + 4 * sqrt 5
  -- The proof steps would go here.
  sorry

end find_y_l298_298110


namespace greatest_x_lcm_l298_298928

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298928


namespace problem_statement_l298_298937

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^7 - 6 * x^5 + 5 * x^3 - x = 0 :=
sorry

end problem_statement_l298_298937


namespace value_of_a_with_two_distinct_roots_l298_298179

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l298_298179


namespace trivia_team_total_points_l298_298477

/-- Given the points scored by the 5 members who showed up in a trivia team game,
    prove that the total points scored by the team is 29. -/
theorem trivia_team_total_points 
  (points_first : ℕ := 5) 
  (points_second : ℕ := 9) 
  (points_third : ℕ := 7) 
  (points_fourth : ℕ := 5) 
  (points_fifth : ℕ := 3) 
  (total_points : ℕ := points_first + points_second + points_third + points_fourth + points_fifth) :
  total_points = 29 :=
by
  sorry

end trivia_team_total_points_l298_298477


namespace correct_judgment_is_C_l298_298195

-- Definitions based on conditions
def three_points_determine_a_plane (p1 p2 p3 : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by three points
  sorry

def line_and_point_determine_a_plane (l : Line) (p : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by a line and a point not on the line
  sorry

def two_parallel_lines_and_intersecting_line_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Axiom 3 and its corollary stating that two parallel lines intersected by the same line are in the same plane
  sorry

def three_lines_intersect_pairwise_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Definition stating that three lines intersecting pairwise might be co-planar or not
  sorry

-- Statement of the problem in Lean
theorem correct_judgment_is_C :
    ¬ (three_points_determine_a_plane p1 p2 p3)
  ∧ ¬ (line_and_point_determine_a_plane l p)
  ∧ (two_parallel_lines_and_intersecting_line_same_plane l1 l2 l3)
  ∧ ¬ (three_lines_intersect_pairwise_same_plane l1 l2 l3) :=
  sorry

end correct_judgment_is_C_l298_298195


namespace acid_concentration_third_flask_l298_298284

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l298_298284


namespace equation_has_roots_l298_298150

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l298_298150


namespace vertical_asymptote_at_9_over_4_l298_298335

def vertical_asymptote (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', x' ≠ x → abs (x' - x) < δ → abs (y x') > ε)

noncomputable def function_y (x : ℝ) : ℝ :=
  (2 * x + 3) / (4 * x - 9)

theorem vertical_asymptote_at_9_over_4 :
  vertical_asymptote function_y (9 / 4) :=
sorry

end vertical_asymptote_at_9_over_4_l298_298335


namespace distance_AD_35_l298_298879

-- Definitions based on conditions
variables (A B C D : Point)
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (dueEast : Point → Point → Prop)
variable (northOf : Point → Point → Prop)

-- Conditions
def conditions : Prop :=
  dueEast A B ∧
  angle A B C = 90 ∧
  distance A C = 15 * Real.sqrt 3 ∧
  angle B A C = 30 ∧
  northOf D C ∧
  distance C D = 10

-- The question: Proving the distance between points A and D
theorem distance_AD_35 (h : conditions A B C D distance angle dueEast northOf) :
  distance A D = 35 :=
sorry

end distance_AD_35_l298_298879


namespace solve_quadratic_l298_298760

def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem solve_quadratic : (quadratic_eq (-2) 1 3 (-1)) ∧ (quadratic_eq (-2) 1 3 (3/2)) :=
by
  sorry

end solve_quadratic_l298_298760


namespace not_traversable_n_62_l298_298431

theorem not_traversable_n_62 :
  ¬ (∃ (path : ℕ → ℕ), ∀ i < 62, path (i + 1) = (path i + 8) % 62 ∨ path (i + 1) = (path i + 9) % 62 ∨ path (i + 1) = (path i + 10) % 62) :=
by sorry

end not_traversable_n_62_l298_298431


namespace number_line_is_line_l298_298309

-- Define the terms
def number_line : Type := ℝ -- Assume number line can be considered real numbers for simplicity
def is_line (l : Type) : Prop := l = ℝ

-- Proving that number line is a line.
theorem number_line_is_line : is_line number_line :=
by {
  -- by definition of the number_line and is_line
  sorry
}

end number_line_is_line_l298_298309


namespace number_of_people_l298_298726

theorem number_of_people (x : ℕ) (H : x * (x - 1) = 72) : x = 9 :=
sorry

end number_of_people_l298_298726


namespace concentration_third_flask_l298_298273

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l298_298273


namespace top_z_teams_l298_298580

theorem top_z_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 45) : n = 10 := 
sorry

end top_z_teams_l298_298580


namespace problem_units_digit_1_probability_l298_298308

def units_digit (x : Nat) : Nat :=
  x % 10

def has_units_digit_1 (m n : Nat) : Prop :=
  units_digit (m^n) = 1

noncomputable def probability {α : Type*} [Fintype α] (s : Set α) (P : α → Prop) : Real :=
  Fintype.card (SetOf P) / Fintype.card α

theorem problem_units_digit_1_probability :
  probability (SetOf (λ mn : Nat × Nat, mn.1 ∈ {18, 22, 25, 27, 29} ∧ mn.2 ∈ Finset.range 20 + 2001)) (λ mn, has_units_digit_1 mn.1 mn.2) = 3 / 4 :=
by
  sorry

end problem_units_digit_1_probability_l298_298308


namespace greatest_possible_x_max_possible_x_l298_298894

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298894


namespace mapping_image_l298_298506

theorem mapping_image (x y l m : ℤ) (h1 : x = 4) (h2 : y = 6) (h3 : l = x + y) (h4 : m = x - y) :
  (l, m) = (10, -2) := by
  sorry

end mapping_image_l298_298506


namespace remainder_div_14_l298_298452

def S : ℕ := 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077

theorem remainder_div_14 : S % 14 = 7 :=
by
  sorry

end remainder_div_14_l298_298452


namespace bd_le_q2_l298_298031

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end bd_le_q2_l298_298031


namespace non_negative_sums_in_table_l298_298384

theorem non_negative_sums_in_table (n : ℕ) (A : Finₓ 1 × Finₓ n → ℝ) :
  ∃ B : Finₓ 1 × Finₓ n → ℝ, (∀ i : Finₓ 1, 0 ≤ ∑ j : Finₓ n, B (i, j)) ∧ (∀ j : Finₓ n, 0 ≤ ∑ i : Finₓ 1, B (i, j)) :=
sorry

end non_negative_sums_in_table_l298_298384


namespace train_length_is_180_l298_298476

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ := 
  (speed_kmh * 5 / 18) * time_seconds

theorem train_length_is_180 : train_length 9 72 = 180 :=
by
  sorry

end train_length_is_180_l298_298476


namespace factorize_difference_of_squares_l298_298145

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l298_298145


namespace circular_garden_radius_l298_298957

theorem circular_garden_radius (r : ℝ) (h : 2 * Real.pi * r = (1 / 8) * Real.pi * r^2) : r = 16 :=
sorry

end circular_garden_radius_l298_298957


namespace increasing_iff_a_gt_neg1_l298_298566

noncomputable def increasing_function_condition (a : ℝ) (b : ℝ) (x : ℝ) : Prop :=
  let y := (a + 1) * x + b
  a > -1

theorem increasing_iff_a_gt_neg1 (a : ℝ) (b : ℝ) : (∀ x : ℝ, (a + 1) > 0) ↔ a > -1 :=
by
  sorry

end increasing_iff_a_gt_neg1_l298_298566


namespace intersection_of_A_and_B_l298_298832

def setA : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}
def setB : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem intersection_of_A_and_B : setA ∩ setB = {y | y ≥ 0} :=
by
  sorry

end intersection_of_A_and_B_l298_298832


namespace perimeter_of_triangle_l298_298085

theorem perimeter_of_triangle
  (P : ℝ)
  (r : ℝ := 1.5)
  (A : ℝ := 29.25)
  (h : A = r * (P / 2)) :
  P = 39 :=
by
  sorry

end perimeter_of_triangle_l298_298085


namespace solve_for_A_plus_B_l298_298683

theorem solve_for_A_plus_B (A B : ℤ) (h : ∀ ω, ω^2 + ω + 1 = 0 → ω^103 + A * ω + B = 0) : A + B = -1 :=
sorry

end solve_for_A_plus_B_l298_298683


namespace each_child_play_time_l298_298758

-- Define the conditions
def number_of_children : ℕ := 6
def pair_play_time : ℕ := 120
def pairs_playing_at_a_time : ℕ := 2

-- Define main theorem
theorem each_child_play_time : 
  (pairs_playing_at_a_time * pair_play_time) / number_of_children = 40 :=
sorry

end each_child_play_time_l298_298758


namespace powerjet_30_minutes_500_gallons_per_hour_l298_298253

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l298_298253


namespace pq_square_sum_l298_298070

theorem pq_square_sum (p q : ℝ) (h1 : p * q = 9) (h2 : p + q = 6) : p^2 + q^2 = 18 := 
by
  sorry

end pq_square_sum_l298_298070


namespace difference_in_dimes_l298_298074

theorem difference_in_dimes : 
  ∀ (a b c : ℕ), (a + b + c = 100) → (5 * a + 10 * b + 25 * c = 835) → 
  (∀ b_max b_min, (b_max = 67) ∧ (b_min = 3) → (b_max - b_min = 64)) :=
by
  intros a b c h1 h2 b_max b_min h_bounds
  sorry

end difference_in_dimes_l298_298074


namespace greatest_x_lcm_105_l298_298912

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298912


namespace oranges_kilos_bought_l298_298050

-- Definitions based on the given conditions
variable (O A x : ℝ)

-- Definitions from conditions
def A_value : Prop := A = 29
def equation1 : Prop := x * O + 5 * A = 419
def equation2 : Prop := 5 * O + 7 * A = 488

-- The theorem we want to prove
theorem oranges_kilos_bought {O A x : ℝ} (A_value: A = 29) (h1: x * O + 5 * A = 419) (h2: 5 * O + 7 * A = 488) : x = 5 :=
by
  -- start of proof
  sorry  -- proof omitted

end oranges_kilos_bought_l298_298050


namespace mary_days_eq_11_l298_298230

variable (x : ℝ) -- Number of days Mary takes to complete the work
variable (m_eff : ℝ) -- Efficiency of Mary (work per day)
variable (r_eff : ℝ) -- Efficiency of Rosy (work per day)

-- Given conditions
axiom rosy_efficiency : r_eff = 1.1 * m_eff
axiom rosy_days : r_eff * 10 = 1

-- Define the efficiency of Mary in terms of days
axiom mary_efficiency : m_eff = 1 / x

-- The theorem to prove
theorem mary_days_eq_11 : x = 11 :=
by
  sorry

end mary_days_eq_11_l298_298230


namespace men_in_first_group_l298_298207

theorem men_in_first_group (M : ℕ) (h1 : (M * 15) = (M + 0) * 15) (h2 : (15 * 36) = 540) : M = 36 :=
by
  -- Proof would go here
  sorry

end men_in_first_group_l298_298207


namespace pow_mul_eq_add_l298_298980

theorem pow_mul_eq_add (a : ℝ) : a^3 * a^4 = a^7 :=
by
  -- This is where the proof would go.
  sorry

end pow_mul_eq_add_l298_298980


namespace range_of_a_l298_298529

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298529


namespace simplify_and_evaluate_expression_l298_298248

variable (x y : ℝ)

theorem simplify_and_evaluate_expression (h₁ : x = -2) (h₂ : y = 1/2) :
  (x + 2 * y) ^ 2 - (x + y) * (3 * x - y) - 5 * y ^ 2 / (2 * x) = 2 + 1 / 2 := 
sorry

end simplify_and_evaluate_expression_l298_298248


namespace union_sets_intersection_complement_sets_l298_298200

universe u
variable {U A B : Set ℝ}

def universal_set : Set ℝ := {x | x ≤ 4}
def set_A : Set ℝ := {x | -2 < x ∧ x < 3}
def set_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem union_sets : set_A ∪ set_B = {x | -3 ≤ x ∧ x < 3} := by
  sorry

theorem intersection_complement_sets :
  set_A ∩ (universal_set \ set_B) = {x | 2 < x ∧ x < 3} := by
  sorry

end union_sets_intersection_complement_sets_l298_298200


namespace greatest_x_lcm_l298_298925

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298925


namespace greatest_x_lcm_l298_298900

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298900


namespace quadratic_symmetric_l298_298381

-- Conditions: Graph passes through the point P(-2,4)
-- y = ax^2 is symmetric with respect to the y-axis

theorem quadratic_symmetric (a : ℝ) (h : a * (-2)^2 = 4) : a * 2^2 = 4 :=
by
  sorry

end quadratic_symmetric_l298_298381


namespace exam_students_count_l298_298105

theorem exam_students_count (n : ℕ) (T : ℕ) (h1 : T = 90 * n) 
                            (h2 : (T - 90) / (n - 2) = 95) : n = 20 :=
by {
  sorry
}

end exam_students_count_l298_298105


namespace no_solution_for_inequalities_l298_298830

theorem no_solution_for_inequalities (x : ℝ) :
  ¬(5 * x^2 - 7 * x + 1 < 0 ∧ x^2 - 9 * x + 30 < 0) :=
sorry

end no_solution_for_inequalities_l298_298830


namespace star_problem_l298_298386

def star_problem_proof (p q r s u : ℤ) (S : ℤ): Prop :=
  (S = 64) →
  ({n : ℤ | n = 19 ∨ n = 21 ∨ n = 23 ∨ n = 25 ∨ n = 27} = {p, q, r, s, u}) →
  (p + q + r + s + u = 115) →
  (9 + p + q + 7 = S) →
  (3 + p + u + 15 = S) →
  (3 + q + r + 11 = S) →
  (9 + u + s + 11 = S) →
  (15 + s + r + 7 = S) →
  (q = 27)

theorem star_problem : ∃ p q r s u S, star_problem_proof p q r s u S := by
  -- Proof goes here
  sorry

end star_problem_l298_298386


namespace cubic_has_three_zeros_l298_298546

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298546


namespace parallel_perpendicular_implies_perpendicular_l298_298741

-- Definitions of the geometric relationships
variables {Line Plane : Type}
variables (a b : Line) (alpha beta : Plane)

-- Conditions as per the problem statement
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Lean statement of the proof problem
theorem parallel_perpendicular_implies_perpendicular
  (h1 : parallel_line_plane a alpha)
  (h2 : perpendicular_line_plane b alpha) :
  perpendicular_lines a b :=  
sorry

end parallel_perpendicular_implies_perpendicular_l298_298741


namespace variance_3ξ_plus_2_l298_298344

-- Main definition
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Hypothesis: ξ follows a binomial distribution with parameters n = 5 and p = 1/3
def ξ : Type := sorry  -- ξ is a random variable following B(5, 1/3)

-- Prove that D(3ξ + 2) = 10
theorem variance_3ξ_plus_2 : (binomial_variance 5 (1/3)) = (10 : ℝ) := by
  sorry

end variance_3ξ_plus_2_l298_298344


namespace maria_green_beans_l298_298407

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l298_298407


namespace value_of_b_l298_298484

def g (x : ℝ) : ℝ := 5 * x - 6

theorem value_of_b (b : ℝ) : g b = 0 ↔ b = 6 / 5 :=
by sorry

end value_of_b_l298_298484


namespace find_a_for_quadratic_l298_298170

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l298_298170


namespace number_of_songs_l298_298393

-- Definition of the given conditions
def total_storage_GB : ℕ := 16
def used_storage_GB : ℕ := 4
def storage_per_song_MB : ℕ := 30
def GB_to_MB : ℕ := 1000

-- Theorem stating the result
theorem number_of_songs (total_storage remaining_storage song_size conversion_factor : ℕ) :
  total_storage = total_storage_GB →
  remaining_storage = total_storage - used_storage_GB →
  song_size = storage_per_song_MB →
  conversion_factor = GB_to_MB →
  (remaining_storage * conversion_factor) / song_size = 400 :=
by
  intros h_total h_remaining h_song_size h_conversion
  rw [h_total, h_remaining, h_song_size, h_conversion]
  sorry

end number_of_songs_l298_298393


namespace max_sum_distinct_factors_2029_l298_298388

theorem max_sum_distinct_factors_2029 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2029 ∧ A + B + C = 297 :=
by
  sorry

end max_sum_distinct_factors_2029_l298_298388


namespace sequence_solution_l298_298351

theorem sequence_solution (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * a n - 2^n + 1) : a n = n * 2^(n-1) :=
sorry

end sequence_solution_l298_298351


namespace gcd_45045_30030_l298_298325

/-- The greatest common divisor of 45045 and 30030 is 15015. -/
theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 :=
by 
  sorry

end gcd_45045_30030_l298_298325


namespace smallest_three_digit_plus_one_multiple_l298_298652

theorem smallest_three_digit_plus_one_multiple (x : ℕ) : 
  (421 = x) →
  (x ≥ 100 ∧ x < 1000) ∧ 
  ∃ k : ℕ, x = k * Nat.lcm (Nat.lcm 3 4) * Nat.lcm 5 7 + 1 :=
by
  sorry

end smallest_three_digit_plus_one_multiple_l298_298652


namespace weekly_earnings_correct_l298_298604

-- Definitions based on the conditions
def hours_weekdays : Nat := 5 * 5
def hours_weekends : Nat := 3 * 2
def hourly_rate_weekday : Nat := 3
def hourly_rate_weekend : Nat := 3 * 2
def earnings_weekdays : Nat := hours_weekdays * hourly_rate_weekday
def earnings_weekends : Nat := hours_weekends * hourly_rate_weekend

-- The total weekly earnings Mitch gets
def weekly_earnings : Nat := earnings_weekdays + earnings_weekends

-- The theorem we need to prove:
theorem weekly_earnings_correct : weekly_earnings = 111 :=
by
  sorry

end weekly_earnings_correct_l298_298604


namespace calories_350_grams_mint_lemonade_l298_298872

-- Definitions for the weights of ingredients in grams
def lemon_juice_weight := 150
def sugar_weight := 200
def water_weight := 300
def mint_weight := 50
def total_weight := lemon_juice_weight + sugar_weight + water_weight + mint_weight

-- Definitions for the caloric content per specified weight
def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def mint_calories_per_10g := 7
def water_calories := 0

-- Calculate total calories from each ingredient
def lemon_juice_calories := (lemon_juice_calories_per_100g * lemon_juice_weight) / 100
def sugar_calories := (sugar_calories_per_100g * sugar_weight) / 100
def mint_calories := (mint_calories_per_10g * mint_weight) / 10

-- Calculate total calories in the lemonade
def total_calories := lemon_juice_calories + sugar_calories + mint_calories + water_calories

noncomputable def calories_in_350_grams : ℕ := (total_calories * 350) / total_weight

-- Theorem stating the number of calories in 350 grams of Marco’s lemonade
theorem calories_350_grams_mint_lemonade : calories_in_350_grams = 440 := 
by
  sorry

end calories_350_grams_mint_lemonade_l298_298872


namespace smallest_positive_integer_with_20_divisors_is_432_l298_298642

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end smallest_positive_integer_with_20_divisors_is_432_l298_298642


namespace solve_for_x_l298_298046

theorem solve_for_x (x : ℝ) (h : 0.20 * x = 0.15 * 1500 - 15) : x = 1050 := 
by
  sorry

end solve_for_x_l298_298046


namespace weight_of_5_diamonds_l298_298626

-- Define the weight of one diamond and one jade
variables (D J : ℝ)

-- Conditions:
-- 1. Total weight of 4 diamonds and 2 jades
def condition1 : Prop := 4 * D + 2 * J = 140
-- 2. A jade is 10 g heavier than a diamond
def condition2 : Prop := J = D + 10

-- Total weight of 5 diamonds
def total_weight_of_5_diamonds : ℝ := 5 * D

-- Theorem: Prove that the total weight of 5 diamonds is 100 g
theorem weight_of_5_diamonds (h1 : condition1 D J) (h2 : condition2 D J) : total_weight_of_5_diamonds D = 100 :=
by {
  sorry
}

end weight_of_5_diamonds_l298_298626


namespace permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l298_298364

-- Part (a)
theorem permutations_with_exactly_one_descent (n : ℕ) : 
  ∃ (count : ℕ), count = 2^n - n - 1 := sorry

-- Part (b)
theorem permutations_with_exactly_two_descents (n : ℕ) : 
  ∃ (count : ℕ), count = 3^n - 2^n * (n + 1) + (n * (n + 1)) / 2 := sorry

end permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l298_298364


namespace walter_time_spent_at_seals_l298_298448

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l298_298448


namespace inf_coprime_naturals_l298_298750

theorem inf_coprime_naturals (a b : ℤ) (h : a ≠ b) : 
  ∃ᶠ n in Filter.atTop, Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) = 1 := 
sorry

end inf_coprime_naturals_l298_298750


namespace probability_of_distance_less_than_8000_miles_l298_298624

def city := ℕ -- Representing cities as natural numbers

def distance (a b : city) : ℕ :=
  match (a, b) with
  | (0, 1) => 6300
  | (0, 2) => 6609
  | (0, 3) => 5944
  | (0, 4) => 8671
  | (1, 2) => 11535
  | (1, 3) => 5989
  | (1, 4) => 7900
  | (2, 3) => 7240
  | (2, 4) => 4986
  | (3, 4) => 3460
  | (1, 0) => 6300
  | (2, 0) => 6609
  | (3, 0) => 5944
  | (4, 0) => 8671
  | (2, 1) => 11535
  | (3, 1) => 5989
  | (4, 1) => 7900
  | (3, 2) => 7240
  | (4, 2) => 4986
  | (4, 3) => 3460
  | _ => 0 -- Default for the identity and invalid pairs
  end

def probability_distance_lt_8000 : ℚ :=
  let pairs := [6300, 6609, 5944, 8671, 11535, 5989, 7900, 7240, 4986, 3460]
  let count_valid = pairs.count (λ d => d < 8000)
  let total_pairs = pairs.length
  count_valid / total_pairs

theorem probability_of_distance_less_than_8000_miles :
  probability_distance_lt_8000 = 3 / 5 :=
by
  -- Proof to be completed
  sorry

end probability_of_distance_less_than_8000_miles_l298_298624


namespace value_of_expression_l298_298370

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l298_298370


namespace total_money_needed_l298_298733

-- Declare John's initial amount
def john_has : ℝ := 0.75

-- Declare the additional amount John needs
def john_needs_more : ℝ := 1.75

-- The theorem statement that John needs a total of $2.50
theorem total_money_needed : john_has + john_needs_more = 2.5 :=
  by
  sorry

end total_money_needed_l298_298733


namespace total_ages_l298_298778

theorem total_ages (bride_age groom_age : ℕ) (h1 : bride_age = 102) (h2 : groom_age = bride_age - 19) : bride_age + groom_age = 185 :=
by
  sorry

end total_ages_l298_298778


namespace spadesuit_evaluation_l298_298059

-- Define the operation
def spadesuit (x y : ℚ) : ℚ := x - (1 / y)

-- Prove the main statement
theorem spadesuit_evaluation : spadesuit 3 (spadesuit 3 (3 / 2)) = 18 / 7 :=
by
  sorry

end spadesuit_evaluation_l298_298059


namespace average_rate_of_change_l298_298455

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x x₁ : α)
variable (h₁ : x ≠ x₁)

theorem average_rate_of_change : 
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) :=
by
  sorry

end average_rate_of_change_l298_298455


namespace polynomial_expansion_l298_298823

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 :=
by
  sorry

end polynomial_expansion_l298_298823


namespace total_steps_l298_298415

theorem total_steps (steps_per_floor : ℕ) (n : ℕ) (m : ℕ) (h : steps_per_floor = 20) (hm : m = 11) (hn : n = 1) : 
  steps_per_floor * (m - n) = 200 :=
by
  sorry

end total_steps_l298_298415


namespace num_of_valid_three_digit_numbers_l298_298844

def valid_three_digit_numbers : ℕ :=
  let valid_numbers : List (ℕ × ℕ × ℕ) :=
    [(2, 3, 4), (4, 6, 8)]
  valid_numbers.length

theorem num_of_valid_three_digit_numbers :
  valid_three_digit_numbers = 2 :=
by
  sorry

end num_of_valid_three_digit_numbers_l298_298844


namespace log_base_2_of_1024_l298_298489

theorem log_base_2_of_1024 (h : 2^10 = 1024) : Real.logb 2 1024 = 10 :=
by
  sorry

end log_base_2_of_1024_l298_298489


namespace smallest_perfect_cube_divisor_l298_298399

theorem smallest_perfect_cube_divisor 
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) 
  (hpr : p ≠ r) (hqr : q ≠ r) (s := 4) (hs : ¬ Nat.Prime s) 
  (hdiv : Nat.Prime 2) :
  ∃ n : ℕ, n = (p * q * r^2 * s)^3 ∧ ∀ m : ℕ, (∃ a b c d : ℕ, a = 3 ∧ b = 3 ∧ c = 6 ∧ d = 3 ∧ m = p^a * q^b * r^c * s^d) → m ≥ n :=
sorry

end smallest_perfect_cube_divisor_l298_298399


namespace total_fish_l298_298317

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l298_298317


namespace weight_of_one_apple_l298_298795

-- Conditions
def total_weight_of_bag_with_apples : ℝ := 1.82
def weight_of_empty_bag : ℝ := 0.5
def number_of_apples : ℕ := 6

-- The proposition to prove: the weight of one apple
theorem weight_of_one_apple : (total_weight_of_bag_with_apples - weight_of_empty_bag) / number_of_apples = 0.22 := 
by
  sorry

end weight_of_one_apple_l298_298795


namespace log_diff_l298_298792

theorem log_diff : (Real.log (12:ℝ) / Real.log (2:ℝ)) - (Real.log (3:ℝ) / Real.log (2:ℝ)) = 2 := 
by
  sorry

end log_diff_l298_298792


namespace bricks_needed_for_wall_l298_298958

noncomputable def number_of_bricks_needed
    (brick_length : ℕ)
    (brick_width : ℕ)
    (brick_height : ℕ)
    (wall_length_m : ℕ)
    (wall_height_m : ℕ)
    (wall_thickness_cm : ℕ) : ℕ :=
  let wall_length_cm := wall_length_m * 100
  let wall_height_cm := wall_height_m * 100
  let wall_volume := wall_length_cm * wall_height_cm * wall_thickness_cm
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume -- This rounds up to the nearest whole number.

theorem bricks_needed_for_wall : number_of_bricks_needed 5 11 6 8 6 2 = 2910 :=
sorry

end bricks_needed_for_wall_l298_298958


namespace max_ski_trips_l298_298622

/--
The ski lift carries skiers from the bottom of the mountain to the top, taking 15 minutes each way, 
and it takes 5 minutes to ski back down the mountain. 
Given that the total available time is 2 hours, prove that the maximum number of trips 
down the mountain in that time is 6.
-/
theorem max_ski_trips (ride_up_time : ℕ) (ski_down_time : ℕ) (total_time : ℕ) :
  ride_up_time = 15 →
  ski_down_time = 5 →
  total_time = 120 →
  (total_time / (ride_up_time + ski_down_time) = 6) :=
by
  intros h1 h2 h3
  sorry

end max_ski_trips_l298_298622


namespace range_of_a_if_f_has_three_zeros_l298_298556

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298556


namespace area_excluding_hole_l298_298665

theorem area_excluding_hole (x : ℝ) : 
  (2 * x + 8) * (x + 6) - (2 * x - 2) * (x - 1) = 24 * x + 46 :=
by
  sorry

end area_excluding_hole_l298_298665


namespace parallel_lines_a_unique_l298_298209

theorem parallel_lines_a_unique (a : ℝ) :
  (∀ x y : ℝ, x + (a + 1) * y + (a^2 - 1) = 0 → x + 2 * y = 0 → -a / 2 = -1 / (a + 1)) →
  a = -2 :=
by
  sorry

end parallel_lines_a_unique_l298_298209


namespace solve_linear_system_l298_298992

theorem solve_linear_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ 
  x = -219 / 88 ∧ y = 97 / 22 :=
by
  sorry

end solve_linear_system_l298_298992


namespace find_a9_l298_298853

variable (a : ℕ → ℝ)  -- Define a sequence a_n.

-- Define the conditions for the arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

variables (h_arith_seq : is_arithmetic_sequence a)
          (h_a3 : a 3 = 8)   -- Condition a_3 = 8
          (h_a6 : a 6 = 5)   -- Condition a_6 = 5 

-- State the theorem.
theorem find_a9 : a 9 = 2 := by
  sorry

end find_a9_l298_298853


namespace greatest_value_of_x_l298_298935

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298935


namespace range_of_a_for_three_zeros_l298_298548

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298548


namespace integer_solutions_of_cubic_equation_l298_298149

theorem integer_solutions_of_cubic_equation :
  ∀ (n m : ℤ),
    n ^ 6 + 3 * n ^ 5 + 3 * n ^ 4 + 2 * n ^ 3 + 3 * n ^ 2 + 3 * n + 1 = m ^ 3 ↔
    (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
by
  intro n m
  apply Iff.intro
  { intro h
    sorry }
  { intro h
    sorry }

end integer_solutions_of_cubic_equation_l298_298149


namespace cubic_has_three_zeros_l298_298542

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298542


namespace smallest_with_20_divisors_is_144_l298_298645

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end smallest_with_20_divisors_is_144_l298_298645


namespace initial_average_mark_l298_298888

theorem initial_average_mark (A : ℕ) (A_excluded : ℕ := 20) (A_remaining : ℕ := 90) (n_total : ℕ := 14) (n_excluded : ℕ := 5) :
    (n_total * A = n_excluded * A_excluded + (n_total - n_excluded) * A_remaining) → A = 65 :=
by 
  intros h
  sorry

end initial_average_mark_l298_298888


namespace teal_bluish_count_l298_298660

theorem teal_bluish_count (n G Bg N B : ℕ) (h1 : n = 120) (h2 : G = 80) (h3 : Bg = 35) (h4 : N = 20) :
  B = 55 :=
by
  sorry

end teal_bluish_count_l298_298660


namespace total_books_l298_298408

variable (M K G : ℕ)

-- Conditions
def Megan_books := 32
def Kelcie_books := Megan_books / 4
def Greg_books := 2 * Kelcie_books + 9

-- Theorem to prove
theorem total_books : Megan_books + Kelcie_books + Greg_books = 65 := by
  unfold Megan_books Kelcie_books Greg_books
  sorry

end total_books_l298_298408


namespace gcd_78_143_l298_298786

theorem gcd_78_143 : Nat.gcd 78 143 = 13 :=
by
  sorry

end gcd_78_143_l298_298786


namespace line_passes_through_fixed_point_l298_298516

variable {a b : ℝ}

theorem line_passes_through_fixed_point : 
  (∀ (x y : ℝ), a + 2 * b = 1 ∧ ax + 3 * y + b = 0 → (x, y) = (1/2, -1/6)) :=
by
  sorry

end line_passes_through_fixed_point_l298_298516


namespace sum_of_largest_100_l298_298433

theorem sum_of_largest_100 (a : Fin 123 → ℝ) (h1 : (Finset.univ.sum a) = 3813) 
  (h2 : ∀ i j : Fin 123, i ≤ j → a i ≤ a j) : 
  ∃ s : Finset (Fin 123), s.card = 100 ∧ (s.sum a) ≥ 3100 :=
by
  sorry

end sum_of_largest_100_l298_298433


namespace xiaohua_final_score_l298_298965

-- Definitions for conditions
def education_score : ℝ := 9
def experience_score : ℝ := 7
def work_attitude_score : ℝ := 8
def weight_education : ℝ := 1
def weight_experience : ℝ := 2
def weight_attitude : ℝ := 2

-- Computation of the final score
noncomputable def final_score : ℝ :=
  education_score * (weight_education / (weight_education + weight_experience + weight_attitude)) +
  experience_score * (weight_experience / (weight_education + weight_experience + weight_attitude)) +
  work_attitude_score * (weight_attitude / (weight_education + weight_experience + weight_attitude))

-- The statement we want to prove
theorem xiaohua_final_score :
  final_score = 7.8 :=
sorry

end xiaohua_final_score_l298_298965


namespace concentration_of_acid_in_third_flask_l298_298275

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l298_298275


namespace smallest_integer_with_20_divisors_l298_298639

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end smallest_integer_with_20_divisors_l298_298639


namespace range_of_a_for_three_zeros_l298_298536

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298536


namespace circle_equation_with_diameter_endpoints_l298_298083

theorem circle_equation_with_diameter_endpoints (A B : ℝ × ℝ) (x y : ℝ) :
  A = (1, 4) → B = (3, -2) → (x-2)^2 + (y-1)^2 = 10 :=
by
  sorry

end circle_equation_with_diameter_endpoints_l298_298083


namespace calc_difference_l298_298040

theorem calc_difference :
  let a := (7/12 : ℚ) * 450
  let b := (3/5 : ℚ) * 320
  let c := (5/9 : ℚ) * 540
  let d := b + c
  d - a = 229.5 := by
  -- declare the variables and provide their values
  sorry

end calc_difference_l298_298040


namespace necessary_and_sufficient_condition_l298_298504

theorem necessary_and_sufficient_condition 
  (a b c : ℝ) :
  (a^2 = b^2 + c^2) ↔
  (∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0) := 
sorry

end necessary_and_sufficient_condition_l298_298504


namespace total_trip_time_l298_298596

-- Definitions: conditions from the problem
def time_in_first_country : Nat := 2
def time_in_second_country := 2 * time_in_first_country
def time_in_third_country := 2 * time_in_first_country

-- Statement: prove that the total time spent is 10 weeks
theorem total_trip_time : time_in_first_country + time_in_second_country + time_in_third_country = 10 := by
  sorry

end total_trip_time_l298_298596


namespace sin_product_difference_proof_l298_298820

noncomputable def sin_product_difference : Prop :=
  sin 70 * sin 65 - sin 20 * sin 25 = sqrt 2 / 2

theorem sin_product_difference_proof : sin_product_difference :=
  by sorry

end sin_product_difference_proof_l298_298820


namespace lemonade_sales_l298_298949

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l298_298949


namespace math_problem_l298_298633

theorem math_problem : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end math_problem_l298_298633


namespace solve_for_y_l298_298514

theorem solve_for_y (y : ℕ) (h : 9 / y^2 = 3 * y / 81) : y = 9 :=
sorry

end solve_for_y_l298_298514


namespace find_n_value_l298_298332

theorem find_n_value (AB AC n m : ℕ) (h1 : AB = 33) (h2 : AC = 21) (h3 : AD = m) (h4 : DE = m) (h5 : EC = m) (h6 : BC = n) : 
  ∃ m : ℕ, m > 7 ∧ m < 21 ∧ n = 30 := 
by sorry

end find_n_value_l298_298332


namespace total_ranking_sequences_l298_298977

-- Define teams
inductive Team
| A | B | C | D

-- Define the conditions
def qualifies (t : Team) : Prop := 
  -- Each team must win its qualifying match to participate
  true

def plays_saturday (t1 t2 t3 t4 : Team) : Prop :=
  (t1 = Team.A ∧ t2 = Team.B) ∨ (t3 = Team.C ∧ t4 = Team.D)

def plays_sunday (t1 t2 t3 t4 : Team) : Prop := 
  -- Winners of Saturday's matches play for 1st and 2nd, losers play for 3rd and 4th
  true

-- Lean statement for the proof problem
theorem total_ranking_sequences : 
  (∀ t : Team, qualifies t) → 
  (∀ t1 t2 t3 t4 : Team, plays_saturday t1 t2 t3 t4) → 
  (∀ t1 t2 t3 t4 : Team, plays_sunday t1 t2 t3 t4) → 
  ∃ n : ℕ, n = 16 :=
by 
  sorry

end total_ranking_sequences_l298_298977


namespace BoatCrafters_l298_298978

/-
  Let J, F, M, A represent the number of boats built in January, February,
  March, and April respectively.

  Conditions:
  1. J = 4
  2. F = J / 2
  3. M = F * 3
  4. A = M * 3

  Goal:
  Prove that J + F + M + A = 30.
-/

def BoatCrafters.total_boats_built : Nat := 4 + (4 / 2) + ((4 / 2) * 3) + (((4 / 2) * 3) * 3)

theorem BoatCrafters.boats_built_by_end_of_April : 
  BoatCrafters.total_boats_built = 30 :=   
by 
  sorry

end BoatCrafters_l298_298978


namespace expected_points_earned_by_experts_over_100_games_probability_envelope_5_chosen_in_next_game_l298_298587

-- Definitions based on given conditions
def num_envelopes := 13
def points_to_win := 6
def evenly_matched_teams := true

-- Part (a) statement
theorem expected_points_earned_by_experts_over_100_games :
  (100 * 6 - 100 * (6 * finset.sum (finset.range (11 + 1) \ n.choose (n - 1)))) = 465 := sorry

-- Part (b) statement
theorem probability_envelope_5_chosen_in_next_game :
  12 / 13 = 0.715 := sorry

end expected_points_earned_by_experts_over_100_games_probability_envelope_5_chosen_in_next_game_l298_298587


namespace smallest_boxes_l298_298959

-- Definitions based on the conditions:
def divisible_by (n d : Nat) : Prop := ∃ k, n = d * k

-- The statement to be proved:
theorem smallest_boxes (n : Nat) : 
  divisible_by n 5 ∧ divisible_by n 24 -> n = 120 :=
by sorry

end smallest_boxes_l298_298959


namespace inequality_always_true_l298_298515

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d :=
by sorry

end inequality_always_true_l298_298515


namespace total_fare_for_100_miles_l298_298972

theorem total_fare_for_100_miles (b c : ℝ) (h₁ : 200 = b + 80 * c) : 240 = b + 100 * c :=
sorry

end total_fare_for_100_miles_l298_298972


namespace balls_in_boxes_l298_298854

theorem balls_in_boxes:
  ∃ (x y z : ℕ), 
  x + y + z = 320 ∧ 
  6 * x + 11 * y + 15 * z = 1001 ∧
  x > 0 ∧ y > 0 ∧ z > 0 :=
by
  sorry

end balls_in_boxes_l298_298854


namespace cost_of_each_sale_puppy_l298_298814

-- Conditions
def total_cost (total: ℚ) : Prop := total = 800
def non_sale_puppy_cost (cost: ℚ) : Prop := cost = 175
def num_puppies (num: ℕ) : Prop := num = 5

-- Question to Prove
theorem cost_of_each_sale_puppy (total cost : ℚ) (num: ℕ):
  total_cost total →
  non_sale_puppy_cost cost →
  num_puppies num →
  (total - 2 * cost) / (num - 2) = 150 := 
sorry

end cost_of_each_sale_puppy_l298_298814


namespace pages_in_first_issue_l298_298784

-- Define variables for the number of pages in the issues and total pages
variables (P : ℕ) (total_pages : ℕ) (eqn : total_pages = 3 * P + 4)

-- State the theorem using the given conditions and question
theorem pages_in_first_issue (h : total_pages = 220) : P = 72 :=
by
  -- Use the given equation
  have h_eqn : total_pages = 3 * P + 4 := eqn
  sorry

end pages_in_first_issue_l298_298784


namespace avg_visitors_other_days_l298_298967

-- Definitions for average visitors on Sundays and average visitors over the month
def avg_visitors_on_sundays : ℕ := 600
def avg_visitors_over_month : ℕ := 300
def days_in_month : ℕ := 30

-- Given conditions
def num_sundays_in_month : ℕ := 5
def total_days : ℕ := days_in_month
def total_visitors_over_month : ℕ := avg_visitors_over_month * days_in_month

-- Goal: Calculate the average number of visitors on other days (Monday to Saturday)
theorem avg_visitors_other_days :
  (avg_visitors_on_sundays * num_sundays_in_month + (total_days - num_sundays_in_month) * 240) = total_visitors_over_month :=
by
  -- Proof expected here, but skipped according to the instructions
  sorry

end avg_visitors_other_days_l298_298967


namespace three_zeros_condition_l298_298208

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x) + Real.cos (ω * x)

theorem three_zeros_condition (ω : ℝ) (hω : ω > 0) :
  (∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 2 * Real.pi ∧
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0) →
  (∀ ω, (11 / 8 : ℝ) ≤ ω ∧ ω < (15 / 8 : ℝ) ∧
  (∀ x, f ω x = 0 ↔ x = (5 * Real.pi) / (4 * ω))) :=
sorry

end three_zeros_condition_l298_298208


namespace range_of_a_l298_298525

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298525


namespace carmen_parsley_left_l298_298133

theorem carmen_parsley_left (plates_whole_sprig : ℕ) (plates_half_sprig : ℕ) (initial_sprigs : ℕ) :
  plates_whole_sprig = 8 →
  plates_half_sprig = 12 →
  initial_sprigs = 25 →
  initial_sprigs - (plates_whole_sprig + plates_half_sprig / 2) = 11 := by
  intros
  sorry

end carmen_parsley_left_l298_298133


namespace triangle_maximum_area_l298_298228

variable {A B C : ℝ} {a b c : ℝ}

theorem triangle_maximum_area (h1 : a ∣ cos C - (1 / 2) * c = b)
  (h2 : a = 2 * Real.sqrt 3) : 
  (∃ b c, ∆abc_has_sides a b c ∧ ∆abc_area a b c ≤ Real.sqrt 3) :=
by
  sorry

end triangle_maximum_area_l298_298228


namespace total_flowers_l298_298941

def number_of_pots : ℕ := 141
def flowers_per_pot : ℕ := 71

theorem total_flowers : number_of_pots * flowers_per_pot = 10011 :=
by
  -- formal proof goes here
  sorry

end total_flowers_l298_298941


namespace h_at_4_l298_298738

noncomputable def f (x : ℝ) := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) := 3 - (4 / x)

noncomputable def h (x : ℝ) := (1 / f_inv x) + 10

theorem h_at_4 : h 4 = 10.5 :=
by
  sorry

end h_at_4_l298_298738


namespace problem_statement_l298_298187

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) :
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) :=
sorry

end problem_statement_l298_298187


namespace find_product_of_offsets_l298_298699

theorem find_product_of_offsets
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a * b + a + b = 99)
  (h3 : b * c + b + c = 99)
  (h4 : c * a + c + a = 99) :
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
  sorry

end find_product_of_offsets_l298_298699


namespace solve_x_squared_plus_15_eq_y_squared_l298_298788

theorem solve_x_squared_plus_15_eq_y_squared (x y : ℤ) : x^2 + 15 = y^2 → x = 7 ∨ x = -7 ∨ x = 1 ∨ x = -1 := by
  sorry

end solve_x_squared_plus_15_eq_y_squared_l298_298788


namespace expected_points_earned_by_experts_over_100_games_probability_envelope_5_chosen_in_next_game_l298_298588

-- Definitions based on given conditions
def num_envelopes := 13
def points_to_win := 6
def evenly_matched_teams := true

-- Part (a) statement
theorem expected_points_earned_by_experts_over_100_games :
  (100 * 6 - 100 * (6 * finset.sum (finset.range (11 + 1) \ n.choose (n - 1)))) = 465 := sorry

-- Part (b) statement
theorem probability_envelope_5_chosen_in_next_game :
  12 / 13 = 0.715 := sorry

end expected_points_earned_by_experts_over_100_games_probability_envelope_5_chosen_in_next_game_l298_298588


namespace total_bottles_l298_298301

theorem total_bottles (n : ℕ) (h1 : ∃ one_third two_third: ℕ, one_third = n / 3 ∧ two_third = 2 * (n / 3) ∧ 3 * one_third = n)
    (h2 : 25 ≤ n)
    (h3 : ∃ damage1 damage2 damage_diff : ℕ, damage1 = 25 * 160 ∧ damage2 = (n / 3) * 160 + ((2 * (n / 3) - 25) * 130) ∧ damage1 - damage2 = 660) :
    n = 36 :=
by
  sorry

end total_bottles_l298_298301


namespace tangent_circles_BC_length_l298_298952

theorem tangent_circles_BC_length
  (rA rB : ℝ) (A B C : ℝ × ℝ) (distAB distAC : ℝ) 
  (hAB : rA + rB = distAB)
  (hAC : distAB + 2 = distAC) 
  (h_sim : ∀ AD BE BC AC : ℝ, AD / BE = rA / rB → BC / AC = rB / rA) :
  BC = 52 / 7 := sorry

end tangent_circles_BC_length_l298_298952


namespace three_zeros_implies_a_lt_neg3_l298_298562

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298562


namespace function_has_three_zeros_l298_298530

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298530


namespace sqrt_div_equality_l298_298828

noncomputable def sqrt_div (x y : ℝ) : ℝ := Real.sqrt x / Real.sqrt y

theorem sqrt_div_equality (x y : ℝ)
  (h : ( ( (1/3 : ℝ) ^ 2 + (1/4 : ℝ) ^ 2 ) / ( (1/5 : ℝ) ^ 2 + (1/6 : ℝ) ^ 2 ) = 25 * x / (73 * y) )) :
  sqrt_div x y = 5 / 2 :=
sorry

end sqrt_div_equality_l298_298828


namespace smurfs_gold_coins_l298_298425

theorem smurfs_gold_coins (x y : ℕ) (h1 : x + y = 200) (h2 : (2 / 3 : ℚ) * x = (4 / 5 : ℚ) * y + 38) : x = 135 :=
by
  sorry

end smurfs_gold_coins_l298_298425


namespace remainder_when_divided_l298_298017
-- First, import the necessary library.

-- Define the problem conditions and the goal.
theorem remainder_when_divided (P Q Q' R R' S T D D' D'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + D'' * R' + R')
  (h3 : S = D'' * T)
  (h4 : R' = S + T) :
  P % (D * D' * D'') = D * R' + R := by
  sorry

end remainder_when_divided_l298_298017


namespace sufficient_condition_for_gt_l298_298623

theorem sufficient_condition_for_gt (a : ℝ) : (∀ x : ℝ, x > a → x > 1) → (∃ x : ℝ, x > 1 ∧ x ≤ a) → a > 1 :=
by
  sorry

end sufficient_condition_for_gt_l298_298623


namespace limit_na_n_l298_298681

def L (x : ℝ) : ℝ := x - x^2 / 2

def a_n (n : ℕ) : ℝ := (L^[2 * n]) (25 / n)

theorem limit_na_n : tendsto (λ n : ℕ, n * a_n n) atTop (𝓝 (50 / 27)) :=
sorry

end limit_na_n_l298_298681


namespace people_landed_in_virginia_l298_298677

def initial_passengers : ℕ := 124
def texas_out : ℕ := 58
def texas_in : ℕ := 24
def north_carolina_out : ℕ := 47
def north_carolina_in : ℕ := 14
def crew_members : ℕ := 10

def final_passengers := initial_passengers - texas_out + texas_in - north_carolina_out + north_carolina_in
def total_people_landed := final_passengers + crew_members

theorem people_landed_in_virginia : total_people_landed = 67 :=
by
  sorry

end people_landed_in_virginia_l298_298677


namespace find_constants_l298_298180

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ -2 → 
  (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) = 
  A / (x - 1) + B / (x - 4) + C / (x + 2)) →
  A = 4 / 9 ∧ B = 28 / 9 ∧ C = -1 / 3 :=
by
  sorry

end find_constants_l298_298180


namespace least_number_to_add_l298_298096

theorem least_number_to_add (x : ℕ) (h : 1055 % 23 = 20) : x = 3 :=
by
  -- Proof goes here.
  sorry

end least_number_to_add_l298_298096


namespace relationship_among_abc_l298_298020

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end relationship_among_abc_l298_298020


namespace not_true_diamond_self_zero_l298_298819

-- Define the operator ⋄
def diamond (x y : ℝ) := |x - 2*y|

-- The problem statement in Lean4
theorem not_true_diamond_self_zero : ¬ (∀ x : ℝ, diamond x x = 0) := by
  sorry

end not_true_diamond_self_zero_l298_298819


namespace express_train_speed_ratio_l298_298125

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end express_train_speed_ratio_l298_298125


namespace number_of_relatively_prime_to_18_l298_298038

theorem number_of_relatively_prime_to_18 : 
  ∃ N : ℕ, N = 30 ∧ ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → Nat.gcd n 18 = 1 ↔ false :=
by
  sorry

end number_of_relatively_prime_to_18_l298_298038


namespace coeff_a4_b3_c2_in_expansion_l298_298983

def term_coefficient (a b c : ℕ) (n : ℕ): ℕ := 
  Nat.choose n a * Nat.choose (n - a) b

theorem coeff_a4_b3_c2_in_expansion : 
  term_coefficient 4 5 3 9 = 1260 :=
by 
  sorry

end coeff_a4_b3_c2_in_expansion_l298_298983


namespace sum_of_1_to_17_is_odd_l298_298065

-- Define the set of natural numbers from 1 to 17
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

-- Proof that the sum of these numbers is odd
theorem sum_of_1_to_17_is_odd : (List.sum nums) % 2 = 1 := 
by
  sorry  -- Proof goes here

end sum_of_1_to_17_is_odd_l298_298065


namespace number_is_10_l298_298798

theorem number_is_10 (x : ℕ) (h : x * 15 = 150) : x = 10 :=
sorry

end number_is_10_l298_298798


namespace team_A_wins_exactly_4_of_7_l298_298214

noncomputable def probability_team_A_wins_4_of_7 : ℚ :=
  (Nat.choose 7 4) * ((1/2)^4) * ((1/2)^3)

theorem team_A_wins_exactly_4_of_7 :
  probability_team_A_wins_4_of_7 = 35 / 128 := by
sorry

end team_A_wins_exactly_4_of_7_l298_298214


namespace Harry_Terry_difference_l298_298202

theorem Harry_Terry_difference : 
(12 - (4 * 3)) - (12 - 4 * 3) = -24 := 
by
  sorry

end Harry_Terry_difference_l298_298202


namespace rectangle_square_division_l298_298113

theorem rectangle_square_division (a b : ℝ) (n : ℕ) (h1 : (∃ (s1 : ℝ), s1^2 * (n : ℝ) = a * b))
                                            (h2 : (∃ (s2 : ℝ), s2^2 * (n + 76 : ℝ) = a * b)) :
    n = 324 := 
by
  sorry

end rectangle_square_division_l298_298113


namespace min_value_expression_l298_298682

theorem min_value_expression : ∃ x y : ℝ, (x = 2 ∧ y = -3/2) ∧ ∀ a b : ℝ, 2 * a^2 + 2 * b^2 - 8 * a + 6 * b + 28 ≥ 10.5 :=
sorry

end min_value_expression_l298_298682


namespace smallest_integer_with_20_divisors_l298_298638

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end smallest_integer_with_20_divisors_l298_298638


namespace incorrect_inequality_l298_298292

theorem incorrect_inequality : ¬ (-2 < -3) :=
by {
  -- Proof goes here
  sorry
}

end incorrect_inequality_l298_298292


namespace minimum_value_of_f_l298_298097

def f (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 1357

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) := 
by 
  use 1337
  sorry

end minimum_value_of_f_l298_298097


namespace find_b_from_root_l298_298501

theorem find_b_from_root (b : ℝ) :
  (Polynomial.eval (-10) (Polynomial.C 1 * X^2 + Polynomial.C b * X + Polynomial.C (-30)) = 0) →
  b = 7 :=
by
  intro h
  sorry

end find_b_from_root_l298_298501


namespace range_of_a_l298_298526

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298526


namespace greatest_x_lcm_l298_298901

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298901


namespace interest_rate_proof_l298_298049

noncomputable def compound_interest_rate (P A : ℝ) (n : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r)^n

noncomputable def interest_rate (initial  final: ℝ) (years : ℕ) : ℝ := 
  (4: ℝ)^(1/(years: ℝ)) - 1

theorem interest_rate_proof :
  compound_interest_rate 8000 32000 36 (interest_rate 8000 32000 36) ∧
  abs (interest_rate 8000 32000 36 * 100 - 3.63) < 0.01 :=
by
  -- Conditions from the problem for compound interest
  -- Using the formula for interest rate and the condition checks
  sorry

end interest_rate_proof_l298_298049


namespace fraction_defined_iff_l298_298781

theorem fraction_defined_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (|x| - 6)) ↔ (x ≠ 6 ∧ x ≠ -6) :=
by 
  sorry

end fraction_defined_iff_l298_298781


namespace solve_for_x_l298_298454

theorem solve_for_x :
  (∀ x : ℝ, (1 / Real.log x / Real.log 3 + 1 / Real.log x / Real.log 4 + 1 / Real.log x / Real.log 5 = 2))
  → x = 2 * Real.sqrt 15 :=
by
  sorry

end solve_for_x_l298_298454


namespace equation_has_at_least_two_distinct_roots_l298_298160

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l298_298160


namespace valid_a_value_l298_298166

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l298_298166


namespace rate_of_increase_twice_l298_298961

theorem rate_of_increase_twice {x : ℝ} (h : (1 + x)^2 = 2) : x = (Real.sqrt 2) - 1 :=
sorry

end rate_of_increase_twice_l298_298961


namespace perpendicular_case_parallel_case_l298_298201

variable (a b : ℝ)

-- Define the lines
def line1 (a b x y : ℝ) := a * x - b * y + 4 = 0
def line2 (a b x y : ℝ) := (a - 1) * x + y + b = 0

-- Define perpendicular condition
def perpendicular (a b : ℝ) := a * (a - 1) - b = 0

-- Define point condition
def passes_through (a b : ℝ) := -3 * a + b + 4 = 0

-- Define parallel condition
def parallel (a b : ℝ) := a * (a - 1) + b = 0

-- Define intercepts equal condition
def intercepts_equal (a b : ℝ) := b = -a

theorem perpendicular_case
    (h1 : perpendicular a b)
    (h2 : passes_through a b) :
    a = 2 ∧ b = 2 :=
sorry

theorem parallel_case
    (h1 : parallel a b)
    (h2 : intercepts_equal a b) :
    a = 2 ∧ b = -2 :=
sorry

end perpendicular_case_parallel_case_l298_298201


namespace range_of_m_l298_298722

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 4 * x - 1 ∧ x > m → x > 2) → m ≤ 2 :=
by
  intro h
  have h₁ := h 2
  sorry

end range_of_m_l298_298722


namespace smallest_value_of_y_l298_298098

open Real

theorem smallest_value_of_y : 
  ∃ (y : ℝ), 6 * y^2 - 29 * y + 24 = 0 ∧ (∀ z : ℝ, 6 * z^2 - 29 * z + 24 = 0 → y ≤ z) ∧ y = 4 / 3 := 
sorry

end smallest_value_of_y_l298_298098


namespace lewis_weekly_earning_l298_298405

def total_amount_earned : ℕ := 178
def number_of_weeks : ℕ := 89
def weekly_earning (total : ℕ) (weeks : ℕ) : ℕ := total / weeks

theorem lewis_weekly_earning : weekly_earning total_amount_earned number_of_weeks = 2 :=
by
  -- The proof will go here
  sorry

end lewis_weekly_earning_l298_298405


namespace ratio_of_areas_l298_298439

-- Define the proof problem
theorem ratio_of_areas (triangle_area : ℝ) (region_area : ℝ) : 
  (region_area / triangle_area = 1 / 4) :=
sorry

end ratio_of_areas_l298_298439


namespace smallest_n_divisible_by_31997_l298_298306

noncomputable def smallest_n_divisible_by_prime : Nat :=
  let p := 31997
  let k := p
  2 * k

theorem smallest_n_divisible_by_31997 :
  smallest_n_divisible_by_prime = 63994 :=
by
  unfold smallest_n_divisible_by_prime
  rfl

end smallest_n_divisible_by_31997_l298_298306


namespace records_given_l298_298077

theorem records_given (X : ℕ) (started_with : ℕ) (bought : ℕ) (days_per_record : ℕ) (total_days : ℕ)
  (h1 : started_with = 8) (h2 : bought = 30) (h3 : days_per_record = 2) (h4 : total_days = 100) :
  X = 12 := by
  sorry

end records_given_l298_298077


namespace sum_diff_9114_l298_298460

def sum_odd_ints (n : ℕ) := (n + 1) / 2 * (1 + n)
def sum_even_ints (n : ℕ) := n / 2 * (2 + n)

theorem sum_diff_9114 : 
  let m := sum_odd_ints 215
  let t := sum_even_ints 100
  m - t = 9114 :=
by
  sorry

end sum_diff_9114_l298_298460


namespace concentration_third_flask_l298_298272

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l298_298272


namespace square_area_from_diagonal_l298_298806

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ A : ℝ, A = 144 :=
by
  let s := d / Real.sqrt 2
  have s_eq : s = 12 := by
    rw [h]
    field_simp
    norm_num
  use s * s
  rw [s_eq]
  norm_num
  sorry

end square_area_from_diagonal_l298_298806


namespace number_of_students_above_120_l298_298120

noncomputable theory
open_locale classical

-- Given conditions
def total_students : ℕ := 1000
def score_distribution (ξ : ℝ) : Prop := ∀ (μ σ : ℝ), ξ ~ N(μ = 100, σ^2)
def probability_interval : Prop := P(80 ≤ ξ ∧ ξ ≤ 100) = 0.45

-- Problem statement
theorem number_of_students_above_120 :
  (∀ (ξ : ℝ), score_distribution ξ) →
  probability_interval →
  ∑ x in (multiset.filter (≥ 120) (multiset.range total_students)), 1 = 50 :=
by
  intro score_distribution probability_interval
  sorry

end number_of_students_above_120_l298_298120


namespace Parkway_Elementary_girls_not_playing_soccer_l298_298063

/-
  In the fifth grade at Parkway Elementary School, there are 500 students. 
  350 students are boys and 250 students are playing soccer.
  86% of the students that play soccer are boys.
  Prove that the number of girl students that are not playing soccer is 115.
-/
theorem Parkway_Elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (H1 : total_students = 500)
  (H2 : boys = 350)
  (H3 : playing_soccer = 250)
  (H4 : percentage_boys_playing_soccer = 0.86) :
  ∃ (girls_not_playing_soccer : ℕ), girls_not_playing_soccer = 115 :=
by
  sorry

end Parkway_Elementary_girls_not_playing_soccer_l298_298063


namespace mural_total_cost_is_192_l298_298396

def mural_width : ℝ := 6
def mural_height : ℝ := 3
def paint_cost_per_sqm : ℝ := 4
def area_per_hour : ℝ := 1.5
def hourly_rate : ℝ := 10

def mural_area := mural_width * mural_height
def paint_cost := mural_area * paint_cost_per_sqm
def labor_hours := mural_area / area_per_hour
def labor_cost := labor_hours * hourly_rate
def total_mural_cost := paint_cost + labor_cost

theorem mural_total_cost_is_192 : total_mural_cost = 192 := by
  -- Definitions
  sorry

end mural_total_cost_is_192_l298_298396


namespace max_x_lcm_15_21_105_l298_298908

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298908


namespace hoseok_more_than_minyoung_l298_298603

-- Define the initial amounts and additional earnings
def initial_amount : ℕ := 1500000
def additional_min : ℕ := 320000
def additional_hos : ℕ := 490000

-- Define the new amounts
def new_amount_min : ℕ := initial_amount + additional_min
def new_amount_hos : ℕ := initial_amount + additional_hos

-- Define the proof problem: Hoseok's new amount - Minyoung's new amount = 170000
theorem hoseok_more_than_minyoung : (new_amount_hos - new_amount_min) = 170000 :=
by
  -- The proof is skipped.
  sorry

end hoseok_more_than_minyoung_l298_298603


namespace greatest_value_of_x_l298_298931

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298931


namespace first_digit_base5_of_312_is_2_l298_298635

theorem first_digit_base5_of_312_is_2 :
  ∃ d : ℕ, d = 2 ∧ (∀ n : ℕ, d * 5 ^ n ≤ 312 ∧ 312 < (d + 1) * 5 ^ n) :=
by
  sorry

end first_digit_base5_of_312_is_2_l298_298635


namespace find_larger_number_l298_298990

theorem find_larger_number :
  ∃ (L S : ℕ), L - S = 1365 ∧ L = 6 * S + 15 ∧ L = 1635 :=
sorry

end find_larger_number_l298_298990


namespace paint_cost_for_flag_l298_298783

noncomputable def flag_width : ℕ := 12
noncomputable def flag_height : ℕ := 10
noncomputable def paint_cost_per_quart : ℝ := 3.5
noncomputable def coverage_per_quart : ℕ := 4

theorem paint_cost_for_flag : (flag_width * flag_height * 2 / coverage_per_quart : ℝ) * paint_cost_per_quart = 210 := by
  sorry

end paint_cost_for_flag_l298_298783


namespace neither_prime_nor_composite_probability_l298_298851

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def is_neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬is_prime n ∧ ¬is_composite n

theorem neither_prime_nor_composite_probability : 
  let draw_one : ℕ := 1 in
  let total_pieces : ℕ := 100 in
  let count_neither_prime_nor_composite := (finset.range 101).filter is_neither_prime_nor_composite in
  (count_neither_prime_nor_composite.card = 1 ∧ 
  (1 : ℝ) / total_pieces = 1 / 100) := by
  sorry

end neither_prime_nor_composite_probability_l298_298851


namespace convert_base_10_to_base_7_l298_298289

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 784) : 
  ∃ a b c d : ℕ, n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by
  sorry

end convert_base_10_to_base_7_l298_298289


namespace total_fish_l298_298321

theorem total_fish :
  let Billy := 10
  let Tony := 3 * Billy
  let Sarah := Tony + 5
  let Bobby := 2 * Sarah
  in Billy + Tony + Sarah + Bobby = 145 :=
by
  sorry

end total_fish_l298_298321


namespace algebraic_expression_value_l298_298343

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x - 2 = 0) :
  x * (x + 2) + (x + 1)^2 = 5 :=
by
  sorry

end algebraic_expression_value_l298_298343


namespace part1_part2_l298_298402

theorem part1 (a : ℝ) (x : ℝ) (h : a > 0) :
  (|x + 1/a| + |x - a + 1|) ≥ 1 :=
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : |3 + 1/a| + |3 - a + 1| < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part1_part2_l298_298402


namespace max_x_lcm_15_21_105_l298_298906

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298906


namespace sum_of_digits_l298_298809

theorem sum_of_digits (N : ℕ) (h : N * (N + 1) / 2 = 3003) : (7 + 7) = 14 := by
  sorry

end sum_of_digits_l298_298809


namespace cos_of_7pi_over_4_l298_298147

theorem cos_of_7pi_over_4 : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 :=
by
  sorry

end cos_of_7pi_over_4_l298_298147


namespace people_visited_on_Sunday_l298_298974

theorem people_visited_on_Sunday (ticket_price : ℕ) 
                                 (people_per_day_week : ℕ) 
                                 (people_on_Saturday : ℕ) 
                                 (total_revenue : ℕ) 
                                 (days_week : ℕ)
                                 (total_days : ℕ) 
                                 (people_per_day_mf : ℕ) 
                                 (people_on_other_days : ℕ) 
                                 (revenue_other_days : ℕ)
                                 (revenue_Sunday : ℕ)
                                 (people_Sunday : ℕ) :
    ticket_price = 3 →
    people_per_day_week = 100 →
    people_on_Saturday = 200 →
    total_revenue = 3000 →
    days_week = 5 →
    total_days = 7 →
    people_per_day_mf = people_per_day_week * days_week →
    people_on_other_days = people_per_day_mf + people_on_Saturday →
    revenue_other_days = people_on_other_days * ticket_price →
    revenue_Sunday = total_revenue - revenue_other_days →
    people_Sunday = revenue_Sunday / ticket_price →
    people_Sunday = 300 := 
by 
  sorry

end people_visited_on_Sunday_l298_298974


namespace factor_polynomial_l298_298328

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l298_298328


namespace matrix_pow_six_identity_l298_298736

variable {n : Type} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℂ}

theorem matrix_pow_six_identity 
  (h1 : A^2 = B^2) (h2 : B^2 = C^2) (h3 : B^3 = A * B * C + 2 * (1 : Matrix n n ℂ)) : 
  A^6 = 1 :=
by 
  sorry

end matrix_pow_six_identity_l298_298736


namespace scientific_notation_conversion_l298_298250

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l298_298250


namespace concentration_in_third_flask_l298_298278

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l298_298278


namespace equation_has_two_distinct_roots_l298_298158

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l298_298158


namespace three_zeros_implies_a_lt_neg3_l298_298560

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298560


namespace equality_of_ha_l298_298076

theorem equality_of_ha 
  {p a b α β γ : ℝ} 
  (h1 : h_a = (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2))
  (h2 : h_a = (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2)) : 
  (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2) = 
  (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2) :=
by sorry

end equality_of_ha_l298_298076


namespace percentage_below_50000_l298_298670

-- Define all the conditions
def cities_between_50000_and_100000 := 35 -- percentage
def cities_below_20000 := 45 -- percentage
def cities_between_20000_and_50000 := 10 -- percentage
def cities_above_100000 := 10 -- percentage

-- The proof statement
theorem percentage_below_50000 : 
    cities_below_20000 + cities_between_20000_and_50000 = 55 :=
by
    unfold cities_below_20000 cities_between_20000_and_50000
    sorry

end percentage_below_50000_l298_298670


namespace part_a_part_b_l298_298413

-- Define the function with the given conditions
variable {f : ℝ → ℝ}
variable (h_nonneg : ∀ x, 0 ≤ x → 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadditivity : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Part (a): Prove that f(x) ≤ 2x for all x ∈ [0, 1]
theorem part_a : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
by
  sorry -- Proof required.

-- Part (b): Prove that it is not true that f(x) ≤ 1.9x for all x ∈ [0,1]
theorem part_b : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 1.9 * x < f x :=
by
  sorry -- Proof required.

end part_a_part_b_l298_298413


namespace problem_1_problem_2_l298_298296

open Real

theorem problem_1 : sqrt 3 * cos (π / 12) - sin (π / 12) = sqrt 2 := 
sorry

theorem problem_2 : ∀ θ : ℝ, sqrt 3 * cos θ - sin θ ≤ 2 := 
sorry

end problem_1_problem_2_l298_298296


namespace cube_surface_area_increase_l298_298653

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
      new_edge_length := 1.4 * s
      new_surface_area := 6 * (new_edge_length)^2
      increase := new_surface_area - original_surface_area
      percentage_increase := (increase / original_surface_area) * 100 in
  percentage_increase = 96 := by sorry

end cube_surface_area_increase_l298_298653


namespace size_of_former_apartment_l298_298859

open Nat

theorem size_of_former_apartment
  (former_rent_rate : ℕ)
  (new_apartment_cost : ℕ)
  (savings_per_year : ℕ)
  (split_factor : ℕ)
  (savings_per_month : ℕ)
  (share_new_rent : ℕ)
  (former_rent : ℕ)
  (apartment_size : ℕ)
  (h1 : former_rent_rate = 2)
  (h2 : new_apartment_cost = 2800)
  (h3 : savings_per_year = 1200)
  (h4 : split_factor = 2)
  (h5 : savings_per_month = savings_per_year / 12)
  (h6 : share_new_rent = new_apartment_cost / split_factor)
  (h7 : former_rent = share_new_rent + savings_per_month)
  (h8 : apartment_size = former_rent / former_rent_rate) :
  apartment_size = 750 :=
by
  sorry

end size_of_former_apartment_l298_298859


namespace non_congruent_triangles_proof_l298_298039

noncomputable def non_congruent_triangles_count : ℕ :=
  let points := [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  9

theorem non_congruent_triangles_proof :
  non_congruent_triangles_count = 9 :=
sorry

end non_congruent_triangles_proof_l298_298039


namespace arithmetic_sequence_a5_l298_298856

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_a5 (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
by
  sorry

end arithmetic_sequence_a5_l298_298856


namespace total_shaded_area_is_71_l298_298287

-- Define the dimensions of the first rectangle
def rect1_length : ℝ := 4
def rect1_width : ℝ := 12

-- Define the dimensions of the second rectangle
def rect2_length : ℝ := 5
def rect2_width : ℝ := 7

-- Define the dimensions of the overlap area
def overlap_length : ℝ := 3
def overlap_width : ℝ := 4

-- Define the area calculation
def area (length width : ℝ) : ℝ := length * width

-- Calculate the areas of the rectangles and the overlap
def rect1_area : ℝ := area rect1_length rect1_width
def rect2_area : ℝ := area rect2_length rect2_width
def overlap_area : ℝ := area overlap_length overlap_width

-- Total shaded area calculation
def total_shaded_area : ℝ := rect1_area + rect2_area - overlap_area

-- Proof statement to show that the total shaded area is 71 square units
theorem total_shaded_area_is_71 : total_shaded_area = 71 := by
  sorry

end total_shaded_area_is_71_l298_298287


namespace ways_to_stand_on_staircase_l298_298436

theorem ways_to_stand_on_staircase (A B C : Type) (steps : Fin 7) : 
  ∃ ways : Nat, ways = 336 := by sorry

end ways_to_stand_on_staircase_l298_298436


namespace cos_beta_calculation_l298_298401

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
variable (h2 : 0 < β ∧ β < π / 2) -- β is an acute angle
variable (h3 : Real.cos α = Real.sqrt 5 / 5)
variable (h4 : Real.sin (α - β) = Real.sqrt 10 / 10)

theorem cos_beta_calculation :
  Real.cos β = Real.sqrt 2 / 2 :=
  sorry

end cos_beta_calculation_l298_298401


namespace cube_surface_area_sum_of_edges_l298_298259

noncomputable def edge_length (sum_of_edges : ℝ) (num_of_edges : ℝ) : ℝ :=
  sum_of_edges / num_of_edges

noncomputable def surface_area (edge_length : ℝ) : ℝ :=
  6 * edge_length ^ 2

theorem cube_surface_area_sum_of_edges (sum_of_edges : ℝ) (num_of_edges : ℝ) (expected_area : ℝ) :
  num_of_edges = 12 → sum_of_edges = 72 → surface_area (edge_length sum_of_edges num_of_edges) = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cube_surface_area_sum_of_edges_l298_298259


namespace plane_passes_through_line_l298_298687

-- Definition for a plane α and a line l
variable {α : Set Point} -- α represents the set of points in plane α
variable {l : Set Point} -- l represents the set of points in line l

-- The condition given
def passes_through (α : Set Point) (l : Set Point) : Prop :=
  l ⊆ α

-- The theorem statement
theorem plane_passes_through_line (α : Set Point) (l : Set Point) :
  passes_through α l = (l ⊆ α) :=
by
  sorry

end plane_passes_through_line_l298_298687


namespace minimum_value_g_l298_298012

noncomputable def g (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x : ℝ, x > 0 → g x ≥ 7 :=
by
  intros x hx
  sorry

end minimum_value_g_l298_298012


namespace ratio_A_B_l298_298944

variable (A B C : ℕ)

theorem ratio_A_B 
  (h1: A + B + C = 98) 
  (h2: B = 30) 
  (h3: (B : ℚ) / C = 5 / 8) 
  : (A : ℚ) / B = 2 / 3 :=
sorry

end ratio_A_B_l298_298944


namespace youngest_sibling_is_42_l298_298426

-- Definitions for the problem conditions
def consecutive_even_integers (a : ℤ) := [a, a + 2, a + 4, a + 6]
def sum_of_ages_is_180 (ages : List ℤ) := ages.sum = 180

-- Main statement
theorem youngest_sibling_is_42 (a : ℤ) 
  (h1 : sum_of_ages_is_180 (consecutive_even_integers a)) :
  a = 42 := 
sorry

end youngest_sibling_is_42_l298_298426


namespace george_run_speed_l298_298338

theorem george_run_speed (usual_distance : ℝ) (usual_speed : ℝ) (today_first_distance : ℝ) (today_first_speed : ℝ)
  (remaining_distance : ℝ) (expected_time : ℝ) :
  usual_distance = 1.5 →
  usual_speed = 3 →
  today_first_distance = 1 →
  today_first_speed = 2.5 →
  remaining_distance = 0.5 →
  expected_time = usual_distance / usual_speed →
  today_first_distance / today_first_speed + remaining_distance / (remaining_distance / (expected_time - today_first_distance / today_first_speed)) = expected_time →
  remaining_distance / (expected_time - today_first_distance / today_first_speed) = 5 :=
by sorry

end george_run_speed_l298_298338


namespace max_marks_l298_298231

theorem max_marks (M : ℝ) (h_pass : 0.30 * M = 231) : M = 770 := sorry

end max_marks_l298_298231


namespace water_usage_l298_298062

theorem water_usage (payment : ℝ) (usage : ℝ) : 
  payment = 7.2 → (usage ≤ 6 → payment = usage * 0.8) → (usage > 6 → payment = 4.8 + (usage - 6) * 1.2) → usage = 8 :=
by
  sorry

end water_usage_l298_298062


namespace find_units_digit_l298_298993

def units_digit (n : ℕ) : ℕ := n % 10

theorem find_units_digit :
  units_digit (3 * 19 * 1933 - 3^4) = 0 :=
by
  sorry

end find_units_digit_l298_298993


namespace georges_final_score_l298_298456

theorem georges_final_score :
  (6 + 4) * 3 = 30 := 
by
  sorry

end georges_final_score_l298_298456


namespace common_difference_d_l298_298186

open Real

-- Define the arithmetic sequence and relevant conditions
variable (a : ℕ → ℝ) -- Define the sequence as a function from natural numbers to real numbers
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific conditions from our problem
def problem_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  is_arithmetic_sequence a d ∧
  a 1 = 1 ∧
  (a 2) ^ 2 = a 1 * a 6

-- The goal is to prove that the common difference d is either 0 or 3
theorem common_difference_d (a : ℕ → ℝ) (d : ℝ) :
  problem_conditions a d → (d = 0 ∨ d = 3) := by
  sorry

end common_difference_d_l298_298186


namespace scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l298_298472

-- Definitions and Conditions
structure Triangle :=
(a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)

-- Statement of the problem
theorem scalene_triangle_cannot_be_divided_into_two_congruent_triangles (T : Triangle) :
  ¬(∃ (D : ℝ) (ABD ACD : Triangle), ABD.a = ACD.a ∧ ABD.b = ACD.b ∧ ABD.c = ACD.c) :=
sorry

end scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l298_298472


namespace monthly_expenses_last_month_l298_298883

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l298_298883


namespace total_fish_count_l298_298319

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l298_298319


namespace range_of_k_l298_298355

theorem range_of_k 
  (h1 : ∀ x, (x ≠ 1) → (x^2 + k * x + 3) / (x - 1) = 3 * x + k)
  (h2 : ∃! x, x > 0 ∧ ∃ y, (x, y) ∈ ({(x, (3 * x + k) * (x - 1)) | x ≠ 1} : set (ℝ × ℝ))) :
  k = -33 / 8 ∨ k = -4 ∨ k ≥ -3 :=
sorry

end range_of_k_l298_298355


namespace undefined_value_l298_298984

theorem undefined_value (x : ℝ) : (x^2 - 16 * x + 64 = 0) → (x = 8) := by
  sorry

end undefined_value_l298_298984


namespace equation_has_roots_l298_298153

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l298_298153


namespace boys_collected_in_all_l298_298759

-- Definition of the problem’s conditions
variables (solomon juwan levi : ℕ)

-- Given conditions as assumptions
def conditions : Prop :=
  solomon = 66 ∧
  solomon = 3 * juwan ∧
  levi = juwan / 2

-- Total cans collected by all boys
def total_cans (solomon juwan levi : ℕ) : ℕ := solomon + juwan + levi

theorem boys_collected_in_all : ∃ solomon juwan levi : ℕ, 
  conditions solomon juwan levi ∧ total_cans solomon juwan levi = 99 :=
by {
  sorry
}

end boys_collected_in_all_l298_298759


namespace abc_value_l298_298598

theorem abc_value (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 30) 
  (h5 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := 
sorry

end abc_value_l298_298598


namespace valid_a_value_l298_298167

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l298_298167


namespace quadratic_relationship_l298_298033

theorem quadratic_relationship :
  ∀ (x z : ℕ), (x = 1 ∧ z = 5) ∨ (x = 2 ∧ z = 12) ∨ (x = 3 ∧ z = 23) ∨ (x = 4 ∧ z = 38) ∨ (x = 5 ∧ z = 57) →
  z = 2 * x^2 + x + 2 :=
by
  sorry

end quadratic_relationship_l298_298033


namespace equation_has_at_least_two_distinct_roots_l298_298163

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l298_298163


namespace square_side_length_l298_298051

theorem square_side_length (x : ℝ) (h : x^2 = 12) : x = 2 * Real.sqrt 3 :=
sorry

end square_side_length_l298_298051


namespace paidAmount_Y_l298_298951

theorem paidAmount_Y (X Y : ℝ) (h1 : X + Y = 638) (h2 : X = 1.2 * Y) : Y = 290 :=
by
  sorry

end paidAmount_Y_l298_298951


namespace find_k_l298_298718

theorem find_k (k x y : ℕ) (h : k * 2 + 1 = 5) : k = 2 :=
by {
  -- Proof will go here
  sorry
}

end find_k_l298_298718


namespace max_x_2y_l298_298071

noncomputable def max_value (x y : ℝ) : ℝ :=
√(5 / 18) + 1 / 2

theorem max_x_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ max_value x y := by
  sorry

end max_x_2y_l298_298071


namespace student_failed_by_l298_298313

-- Definitions based on the problem conditions
def total_marks : ℕ := 500
def passing_percentage : ℕ := 40
def marks_obtained : ℕ := 150
def passing_marks : ℕ := (passing_percentage * total_marks) / 100

-- The theorem statement
theorem student_failed_by :
  (passing_marks - marks_obtained) = 50 :=
by
  -- The proof is omitted
  sorry

end student_failed_by_l298_298313


namespace smallest_lcm_of_4digit_gcd_5_l298_298378

theorem smallest_lcm_of_4digit_gcd_5 :
  ∃ (m n : ℕ), (1000 ≤ m ∧ m < 10000) ∧ (1000 ≤ n ∧ n < 10000) ∧ 
               m.gcd n = 5 ∧ m.lcm n = 203010 :=
by sorry

end smallest_lcm_of_4digit_gcd_5_l298_298378


namespace greatest_x_lcm_105_l298_298918

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298918


namespace geo_sequence_arithmetic_l298_298697

variable {d : ℝ} (hd : d ≠ 0)
variable {a : ℕ → ℝ} (ha : ∀ n, a (n+1) = a n + d)

-- Hypothesis that a_5, a_9, a_15 form a geometric sequence
variable (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d))

theorem geo_sequence_arithmetic (hd : d ≠ 0) (ha : ∀ n, a (n + 1) = a n + d) (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d)) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end geo_sequence_arithmetic_l298_298697


namespace J_3_15_10_eq_68_over_15_l298_298829

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_15_10_eq_68_over_15 : J 3 15 10 = 68 / 15 := by
  sorry

end J_3_15_10_eq_68_over_15_l298_298829


namespace no_three_digit_numbers_divisible_by_30_l298_298843

def digits_greater_than_6 (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d > 6

theorem no_three_digit_numbers_divisible_by_30 :
  ∀ n, (100 ≤ n ∧ n < 1000 ∧ digits_greater_than_6 n ∧ n % 30 = 0) → false :=
by
  sorry

end no_three_digit_numbers_divisible_by_30_l298_298843


namespace greatest_possible_x_max_possible_x_l298_298897

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298897


namespace factorize_expr_l298_298141

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l298_298141


namespace h_j_h_of_3_l298_298867

def h (x : ℤ) : ℤ := 5 * x + 2
def j (x : ℤ) : ℤ := 3 * x + 4

theorem h_j_h_of_3 : h (j (h 3)) = 277 := by
  sorry

end h_j_h_of_3_l298_298867


namespace find_b2_l298_298620

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 23) (h10 : b 10 = 123) 
  (h : ∀ n ≥ 3, b n = (b 1 + b 2 + (n - 3) * b 3) / (n - 1)) : b 2 = 223 :=
sorry

end find_b2_l298_298620


namespace solutions_diff_l298_298740

theorem solutions_diff (a b : ℝ) (h1: (a-5)*(a+5) = 26*a - 130) (h2: (b-5)*(b+5) = 26*b - 130) (h3 : a ≠ b) (h4: a > b) : a - b = 16 := 
by
  sorry 

end solutions_diff_l298_298740


namespace greatest_x_lcm_l298_298905

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298905


namespace gcd_of_A_and_B_l298_298052

theorem gcd_of_A_and_B (A B : ℕ) (h_lcm : Nat.lcm A B = 120) (h_ratio : A * 4 = B * 3) : Nat.gcd A B = 10 :=
sorry

end gcd_of_A_and_B_l298_298052


namespace polygon_diagonals_with_restricted_vertices_l298_298799

theorem polygon_diagonals_with_restricted_vertices
  (vertices : ℕ) (non_contributing_vertices : ℕ)
  (h_vertices : vertices = 35)
  (h_non_contributing_vertices : non_contributing_vertices = 5) :
  (vertices - non_contributing_vertices) * (vertices - non_contributing_vertices - 3) / 2 = 405 :=
by {
  sorry
}

end polygon_diagonals_with_restricted_vertices_l298_298799


namespace percent_of_volume_filled_by_cubes_l298_298971

theorem percent_of_volume_filled_by_cubes :
  let box_width := 8
  let box_height := 6
  let box_length := 12
  let cube_size := 2
  let box_volume := box_width * box_height * box_length
  let cube_volume := cube_size ^ 3
  let num_cubes := (box_width / cube_size) * (box_height / cube_size) * (box_length / cube_size)
  let cubes_volume := num_cubes * cube_volume
  (cubes_volume / box_volume : ℝ) * 100 = 100 := by
  sorry

end percent_of_volume_filled_by_cubes_l298_298971


namespace functional_eq_solutions_l298_298987

theorem functional_eq_solutions
  (f : ℚ → ℚ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  ∀ x : ℚ, f x = x ∨ f x = -x := 
sorry

end functional_eq_solutions_l298_298987


namespace correct_diagram_l298_298744

-- Definitions based on the conditions
def word : String := "KANGAROO"
def diagrams : List (String × Bool) :=
  [("Diagram A", False), ("Diagram B", False), ("Diagram C", False),
   ("Diagram D", False), ("Diagram E", True)]

-- Statement to prove that Diagram E correctly shows "KANGAROO"
theorem correct_diagram :
  ∃ d, (d.1 = "Diagram E") ∧ d.2 = True ∧ d ∈ diagrams :=
by
-- skipping the proof for now
sorry

end correct_diagram_l298_298744


namespace curve_not_parabola_l298_298044

theorem curve_not_parabola (k : ℝ) : ¬ ∃ (x y : ℝ), (x^2 + k * y^2 = 1) ↔ (k = -y / x) :=
by
  sorry

end curve_not_parabola_l298_298044


namespace pentadecagon_triangle_count_l298_298365

-- Define a regular pentadecagon as a 15-sided polygon
def pentadecagon : Finset (Fin 15) := (Finset.univ : Finset (Fin 15))

-- Define a function to determine if three vertices are consecutive
def consecutive (a b c: Fin 15) : Prop :=
  (a.val + 1) % 15 = b.val ∧ (b.val + 1) % 15 = c.val

-- Theorem to state the number of valid triangles
theorem pentadecagon_triangle_count :
  let vertices := pentadecagon
  let triangles := vertices.powerset.filter (λ s, s.card = 3)
  let valid_triangles := triangles.filter (λ s, ¬ ∃ a b c ∈ s, consecutive a b c)
  valid_triangles.card = 440 :=
  by
  sorry

end pentadecagon_triangle_count_l298_298365


namespace Mitya_age_l298_298233

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l298_298233


namespace max_x_lcm_15_21_105_l298_298910

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298910


namespace percentage_of_black_marbles_l298_298414

variable (T : ℝ) -- Total number of marbles
variable (C : ℝ) -- Number of clear marbles
variable (B : ℝ) -- Number of black marbles
variable (O : ℝ) -- Number of other colored marbles

-- Conditions
def condition1 := C = 0.40 * T
def condition2 := O = (2 / 5) * T
def condition3 := C + B + O = T

-- Proof statement
theorem percentage_of_black_marbles :
  C = 0.40 * T → O = (2 / 5) * T → C + B + O = T → B = 0.20 * T :=
by
  intros hC hO hTotal
  -- Intermediate steps would go here, but we use sorry to skip the proof.
  sorry

end percentage_of_black_marbles_l298_298414


namespace expression_equiv_l298_298482

theorem expression_equiv (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) =
  2*x^2*y^2 + 2/(x^2*y^2) :=
by 
  sorry

end expression_equiv_l298_298482


namespace four_digit_perfect_cubes_divisible_by_16_l298_298710

theorem four_digit_perfect_cubes_divisible_by_16 : (∃ k : ℕ, k = 3) :=
by
  let possible_cubes := [12 ^ 3, 16 ^ 3, 20 ^ 3]
  have h1 : 12 ^ 3 = 1728 := by norm_num
  have h2 : 16 ^ 3 = 4096 := by norm_num
  have h3 : 20 ^ 3 = 8000 := by norm_num

  have h4 : (1728, 4096, 8000).all (λ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 16 = 0)
    := by norm_num

  use 3
  trivial

end four_digit_perfect_cubes_divisible_by_16_l298_298710


namespace julia_fascinating_last_digits_l298_298878

theorem julia_fascinating_last_digits : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (∃ y : ℕ, x = 10 * y) → x % 10 < 10) :=
by
  sorry

end julia_fascinating_last_digits_l298_298878


namespace expected_points_experts_prob_envelope_5_l298_298585

-- Conditions
def num_envelopes := 13
def win_points := 6
def total_games := 100
def envelope_prob := 1 / num_envelopes

-- Part (a): Expected points earned by Experts over 100 games
theorem expected_points_experts 
  (evenly_matched : true) -- Placeholder condition, actual game dynamics assumed
  : (expected (fun (game : ℕ) => game_points_experts game ) (range total_games)) = 465 := 
sorry

-- Part (b): Probability that envelope number 5 will be chosen in the next game
theorem prob_envelope_5 
  : (prob (λ (envelope : ℕ), envelope = 5) (range num_envelopes)) = 12 / 13 :=   -- Simplified calculation
sorry

end expected_points_experts_prob_envelope_5_l298_298585


namespace incorrect_statement_A_l298_298101

-- Definitions based on conditions
def equilibrium_shifts (condition: Type) : Prop := sorry
def value_K_changes (condition: Type) : Prop := sorry

-- The incorrect statement definition
def statement_A (condition: Type) : Prop := equilibrium_shifts condition → value_K_changes condition

-- The final theorem stating that 'statement_A' is incorrect
theorem incorrect_statement_A (condition: Type) : ¬ statement_A condition :=
sorry

end incorrect_statement_A_l298_298101


namespace floor_add_ceil_eq_five_l298_298090

theorem floor_add_ceil_eq_five (x : ℝ) :
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 5 ↔ 2 < x ∧ x < 3 :=
by sorry

end floor_add_ceil_eq_five_l298_298090


namespace find_a_for_quadratic_l298_298174

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l298_298174


namespace speed_of_stream_l298_298099

theorem speed_of_stream (c v : ℝ) (h1 : c - v = 8) (h2 : c + v = 12) : v = 2 :=
by {
  -- proof will go here
  sorry
}

end speed_of_stream_l298_298099


namespace chemistry_marks_l298_298117

-- Definitions based on given conditions
def total_marks (P C M : ℕ) : Prop := P + C + M = 210
def avg_physics_math (P M : ℕ) : Prop := (P + M) / 2 = 90
def physics_marks (P : ℕ) : Prop := P = 110
def avg_physics_other_subject (P C : ℕ) : Prop := (P + C) / 2 = 70

-- The proof problem statement
theorem chemistry_marks {P C M : ℕ} (h1 : total_marks P C M) (h2 : avg_physics_math P M) (h3 : physics_marks P) : C = 30 ∧ avg_physics_other_subject P C :=
by 
  -- Proof goes here
  sorry

end chemistry_marks_l298_298117


namespace values_range_l298_298398

noncomputable def possible_values (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : set ℝ :=
  {x | x = 1 / a + 1 / b}

theorem values_range : 
  ∀ (a b : ℝ), 
  a > 0 → 
  b > 0 → 
  a + b = 2 → 
  possible_values a b (by assumption) (by assumption) (by assumption) = {x | 2 ≤ x} :=
sorry

end values_range_l298_298398


namespace motorists_with_tickets_l298_298607

section SpeedingTickets

variables
  (total_motorists : ℕ)
  (percent_speeding : ℝ) -- percent_speeding is 25% (given)
  (percent_not_ticketed : ℝ) -- percent_not_ticketed is 60% (given)

noncomputable def percent_ticketed : ℝ :=
  let speeding_motorists := percent_speeding * total_motorists / 100
  let ticketed_motorists := speeding_motorists * ((100 - percent_not_ticketed) / 100)
  ticketed_motorists / total_motorists * 100

theorem motorists_with_tickets (total_motorists : ℕ) 
  (h1 : percent_speeding = 25)
  (h2 : percent_not_ticketed = 60) :
  percent_ticketed total_motorists percent_speeding percent_not_ticketed = 10 := 
by
  unfold percent_ticketed
  rw [h1, h2]
  sorry

end SpeedingTickets

end motorists_with_tickets_l298_298607


namespace initial_cell_count_l298_298953

-- Defining the constants and parameters given in the problem
def doubling_time : ℕ := 20 -- minutes
def culture_time : ℕ := 240 -- minutes (4 hours converted to minutes)
def final_bacterial_cells : ℕ := 4096

-- Definition to find the number of doublings
def num_doublings (culture_time doubling_time : ℕ) : ℕ :=
  culture_time / doubling_time

-- Definition for exponential growth formula
def exponential_growth (initial_cells : ℕ) (doublings : ℕ) : ℕ :=
  initial_cells * (2 ^ doublings)

-- The main theorem to be proven
theorem initial_cell_count :
  exponential_growth 1 (num_doublings culture_time doubling_time) = final_bacterial_cells :=
  sorry

end initial_cell_count_l298_298953


namespace expected_points_experts_over_100_games_probability_of_envelope_five_selected_l298_298583

-- Game conditions and probabilities
def game_conditions (experts_points audience_points : ℕ) : Prop :=
  experts_points = 6 ∨ audience_points = 6

noncomputable def equal_teams := (1 : ℝ) / 2

-- Expected score of Experts over 100 games
noncomputable def expected_points_experts (games : ℕ) := 465

-- Probability that envelope number 5 is chosen in the next game
noncomputable def probability_envelope_five := (12 : ℝ) / 13

theorem expected_points_experts_over_100_games : 
  expected_points_experts 100 = 465 := 
sorry

theorem probability_of_envelope_five_selected : 
  probability_envelope_five = 0.715 := 
sorry

end expected_points_experts_over_100_games_probability_of_envelope_five_selected_l298_298583


namespace greatest_x_lcm_105_l298_298922

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298922


namespace contrapositive_of_lt_l298_298428

theorem contrapositive_of_lt (a b c : ℝ) :
  (a < b → a + c < b + c) → (a + c ≥ b + c → a ≥ b) :=
by
  intro h₀ h₁
  sorry

end contrapositive_of_lt_l298_298428


namespace equation_has_two_distinct_roots_l298_298155

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l298_298155


namespace boat_travel_time_l298_298468

noncomputable def total_travel_time (stream_speed boat_speed distance_AB : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance_BC := distance_AB / 2
  (distance_AB / downstream_speed) + (distance_BC / upstream_speed)

theorem boat_travel_time :
  total_travel_time 4 14 180 = 19 :=
by
  sorry

end boat_travel_time_l298_298468


namespace solution_set_for_inequality_l298_298183

-- Define the function involved
def rational_function (x : ℝ) : ℝ :=
  (3 * x - 1) / (2 - x)

-- Define the main theorem to state the solution set for the given inequality
theorem solution_set_for_inequality (x : ℝ) :
  (rational_function x ≥ 1) ↔ (3 / 4 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_for_inequality_l298_298183


namespace expected_points_experts_probability_envelope_5_l298_298590

-- Define the conditions
def evenly_matched_teams : Prop := 
  -- Placeholder for the definition of evenly matched teams
  sorry 

def envelopes_random_choice : Prop := 
  -- Placeholder for the definition of random choice from 13 envelopes
  sorry

def game_conditions (experts_score tv_audience_score : ℕ) : Prop := 
  experts_score = 6 ∨ tv_audience_score = 6

-- Statement for part (a)
theorem expected_points_experts (h1 : evenly_matched_teams) (h2 : envelopes_random_choice) :
  game_conditions experts_score tv_audience_score →
  expected_points experts_score (100 : ℕ) = 465 :=
sorry

-- Statement for part (b)
theorem probability_envelope_5 (h1 : evenly_matched_teams) (h2 : envelopes_random_choice) :
  game_conditions experts_score tv_audience_score →
  probability_envelope_selected (5 : ℕ) = 0.715 :=
sorry

end expected_points_experts_probability_envelope_5_l298_298590


namespace walter_time_at_seals_l298_298442

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l298_298442


namespace time_spent_on_seals_l298_298447

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l298_298447


namespace inclination_angle_of_line_m_l298_298567

theorem inclination_angle_of_line_m
  (m : ℝ → ℝ → Prop)
  (l₁ l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x - y + 1 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x - y - 1 = 0)
  (intersect_segment_length : ℝ)
  (h₃ : intersect_segment_length = 2 * Real.sqrt 2) :
  (∃ α : ℝ, (α = 15 ∨ α = 75) ∧ (∃ k : ℝ, ∀ x y, m x y ↔ y = k * x)) :=
by
  sorry

end inclination_angle_of_line_m_l298_298567


namespace tan_theta_value_l298_298701

theorem tan_theta_value (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 :=
sorry

end tan_theta_value_l298_298701


namespace solve_quadratic_l298_298505

theorem solve_quadratic (h₁ : 48 * (3/4:ℚ)^2 - 74 * (3/4:ℚ) + 47 = 0) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 48 * x^2 - 74 * x + 47 = 0 ∧ x = 11/12 := 
by
  sorry

end solve_quadratic_l298_298505


namespace mike_travel_distance_l298_298409

theorem mike_travel_distance
  (mike_start : ℝ := 2.50)
  (mike_per_mile : ℝ := 0.25)
  (annie_start : ℝ := 2.50)
  (annie_toll : ℝ := 5.00)
  (annie_per_mile : ℝ := 0.25)
  (annie_miles : ℝ := 14)
  (mike_cost : ℝ)
  (annie_cost : ℝ) :
  mike_cost = annie_cost → mike_cost = mike_start + mike_per_mile * 34 := by
  sorry

end mike_travel_distance_l298_298409


namespace greatest_x_lcm_l298_298927

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298927


namespace min_abs_val_of_36_power_minus_5_power_l298_298811

theorem min_abs_val_of_36_power_minus_5_power :
  ∃ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| = 11 := sorry

end min_abs_val_of_36_power_minus_5_power_l298_298811


namespace point_of_tangency_l298_298084

theorem point_of_tangency (x y : ℝ) (h : (y = x^3 + x - 2)) (slope : 4 = 3 * x^2 + 1) : (x, y) = (-1, -4) := 
sorry

end point_of_tangency_l298_298084


namespace mean_of_y_l298_298312

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def regression_line (x : ℝ) : ℝ :=
  2 * x + 45

theorem mean_of_y (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  mean [regression_line 1, regression_line 5, regression_line 7, regression_line 13, regression_line 19] = 63 := by
  sorry

end mean_of_y_l298_298312


namespace seeds_in_small_gardens_l298_298676

theorem seeds_in_small_gardens 
  (total_seeds : ℕ)
  (planted_seeds : ℕ)
  (small_gardens : ℕ)
  (remaining_seeds := total_seeds - planted_seeds) 
  (seeds_per_garden := remaining_seeds / small_gardens) :
  total_seeds = 101 → planted_seeds = 47 → small_gardens = 9 → seeds_per_garden = 6 := by
  sorry

end seeds_in_small_gardens_l298_298676


namespace tennis_tournament_matches_l298_298940

theorem tennis_tournament_matches (n : ℕ) (h₁ : n = 128) (h₂ : ∃ m : ℕ, m = 32) (h₃ : ∃ k : ℕ, k = 96) (h₄ : ∀ i : ℕ, i > 1 → i ≤ n → ∃ j : ℕ, j = 1 + (i - 1)) :
  ∃ total_matches : ℕ, total_matches = 127 := 
by 
  sorry

end tennis_tournament_matches_l298_298940


namespace smallest_sum_of_five_consecutive_primes_divisible_by_three_l298_298015

-- Definition of the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (a b c d e : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  (b = a + 1 ∨ b = a + 2) ∧ (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2) ∧ (e = d + 1 ∨ e = d + 2)

theorem smallest_sum_of_five_consecutive_primes_divisible_by_three :
  ∃ a b c d e, consecutive_primes a b c d e ∧ a + b + c + d + e = 39 ∧ 39 % 3 = 0 :=
sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_three_l298_298015


namespace factor_expression_l298_298688

theorem factor_expression (z : ℤ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := 
by sorry

end factor_expression_l298_298688


namespace simplify_polynomial_l298_298757

variable (p : ℝ)

theorem simplify_polynomial :
  (7 * p ^ 5 - 4 * p ^ 3 + 8 * p ^ 2 - 5 * p + 3) + (- p ^ 5 + 3 * p ^ 3 - 7 * p ^ 2 + 6 * p + 2) =
  6 * p ^ 5 - p ^ 3 + p ^ 2 + p + 5 :=
by
  sorry

end simplify_polynomial_l298_298757


namespace find_a_for_tangent_parallel_l298_298706

theorem find_a_for_tangent_parallel : 
  ∀ a : ℝ,
  (∀ (x y : ℝ), y = Real.log x - a * x → x = 1 → 2 * x + y - 1 = 0) →
  a = 3 :=
by
  sorry

end find_a_for_tangent_parallel_l298_298706


namespace smallest_with_20_divisors_is_144_l298_298644

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end smallest_with_20_divisors_is_144_l298_298644


namespace q0_r0_eq_three_l298_298868

variable (p q r s : Polynomial ℝ)
variable (hp_const : p.coeff 0 = 2)
variable (hs_eq : s = p * q * r)
variable (hs_const : s.coeff 0 = 6)

theorem q0_r0_eq_three : (q.coeff 0) * (r.coeff 0) = 3 := by
  sorry

end q0_r0_eq_three_l298_298868


namespace tangent_line_equation_l298_298599

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + a * x^2 + (a - 3) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_equation (a : ℝ) (h : ∀ x : ℝ, f a (-x) = f a x) :
    9 * (2 : ℝ) - f a 2 - 16 = 0 :=
by
  sorry

end tangent_line_equation_l298_298599


namespace train_speed_is_28_l298_298674

-- Define the given conditions
def train_length : ℕ := 1200
def overbridge_length : ℕ := 200
def crossing_time : ℕ := 50

-- Define the total distance
def total_distance := train_length + overbridge_length

-- Define the speed calculation function
def speed (distance time : ℕ) : ℕ := 
  distance / time

-- State the theorem to be proven
theorem train_speed_is_28 : speed total_distance crossing_time = 28 := 
by
  -- Proof to be provided
  sorry

end train_speed_is_28_l298_298674


namespace abs_sum_a_to_7_l298_298996

-- Sequence definition with domain
def a (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Lean's ℕ includes 0, so use (n + 1) instead of n here.

-- Prove absolute value sum of first seven terms
theorem abs_sum_a_to_7 : (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 25) :=
by
  -- Placeholder for actual proof
  sorry

end abs_sum_a_to_7_l298_298996


namespace min_distance_point_to_line_l298_298509

theorem min_distance_point_to_line (P : ℝ × ℝ)
  (hP : (P.1 - 2)^2 + P.2^2 = 1) :
  ∃ (d : ℝ), d = √3 - 1 ∧ min_dist P (λ x : ℝ, (√3) * x) = d := by
sorry

end min_distance_point_to_line_l298_298509


namespace f_at_zero_f_on_negative_l298_298507

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f(x) for x > 0 condition
def f_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = x^2 + x - 1

-- Lean statement for the first proof: f(0) = 0
theorem f_at_zero (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) : f 0 = 0 :=
sorry

-- Lean statement for the second proof: for x < 0, f(x) = -x^2 + x + 1
theorem f_on_negative (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) :
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
sorry

end f_at_zero_f_on_negative_l298_298507


namespace divides_equiv_l298_298756

theorem divides_equiv (m n : ℤ) : 
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) :=
by
  sorry

end divides_equiv_l298_298756


namespace sarah_jim_ratio_l298_298780

theorem sarah_jim_ratio
  (Tim_toads : ℕ)
  (hTim : Tim_toads = 30)
  (Jim_toads : ℕ)
  (hJim : Jim_toads = Tim_toads + 20)
  (Sarah_toads : ℕ)
  (hSarah : Sarah_toads = 100) :
  Sarah_toads / Jim_toads = 2 :=
by
  sorry

end sarah_jim_ratio_l298_298780


namespace certain_multiple_l298_298009

theorem certain_multiple (n m : ℤ) (h : n = 5) (eq : 7 * n - 15 = m * n + 10) : m = 2 :=
by
  sorry

end certain_multiple_l298_298009


namespace range_of_a_for_three_zeros_l298_298553

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298553


namespace number_of_terms_in_arithmetic_sequence_l298_298204

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → 6 + (k - 1) * 2 = 202)) ∧ n = 99 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l298_298204


namespace division_of_positive_by_negative_l298_298679

theorem division_of_positive_by_negative :
  4 / (-2) = -2 := 
by
  sorry

end division_of_positive_by_negative_l298_298679


namespace fresh_fruit_water_content_l298_298303

theorem fresh_fruit_water_content (W N : ℝ) 
  (fresh_weight_dried: W + N = 50) 
  (dried_weight: (0.80 * 5) = N) : 
  ((W / (W + N)) * 100 = 92) :=
by
  sorry

end fresh_fruit_water_content_l298_298303


namespace john_wages_decrease_percentage_l298_298219

theorem john_wages_decrease_percentage (W : ℝ) (P : ℝ) :
  (0.20 * (W - P/100 * W)) = 0.50 * (0.30 * W) → P = 25 :=
by 
  intro h
  -- Simplification and other steps omitted; focus on structure
  sorry

end john_wages_decrease_percentage_l298_298219


namespace find_y_for_slope_l298_298698

theorem find_y_for_slope (y : ℝ) :
  let R := (-3, 9)
  let S := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 ↔ y = -3 :=
by
  simp [slope]
  sorry

end find_y_for_slope_l298_298698


namespace function_has_three_zeros_l298_298533

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298533


namespace area_of_triangle_OAB_is_5_l298_298863

-- Define the parameters and assumptions
def OA : ℝ × ℝ := (-2, 1)
def OB : ℝ × ℝ := (4, 3)

noncomputable def area_triangle_OAB (OA OB : ℝ × ℝ) : ℝ :=
  1 / 2 * (OA.1 * OB.2 - OA.2 * OB.1)

-- The theorem we want to prove:
theorem area_of_triangle_OAB_is_5 : area_triangle_OAB OA OB = 5 := by
  sorry

end area_of_triangle_OAB_is_5_l298_298863


namespace equation_has_at_least_two_distinct_roots_l298_298164

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l298_298164


namespace sequence_formula_l298_298198

theorem sequence_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n + 1) : 
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) :=
by
  intro n hn
  sorry

end sequence_formula_l298_298198


namespace coin_problem_l298_298480

theorem coin_problem (n d q : ℕ) 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 25 * q = 410)
  (h3 : d = n + 4) : q - n = 2 :=
by
  sorry

end coin_problem_l298_298480


namespace octahedron_tetrahedron_surface_area_ratio_l298_298671

theorem octahedron_tetrahedron_surface_area_ratio 
  (s : ℝ) 
  (h₁ : s = 1)
  (A_octahedron : ℝ := 2 * Real.sqrt 3)
  (A_tetrahedron : ℝ := Real.sqrt 3)
  (h₂ : A_octahedron = 2 * Real.sqrt 3 * s^2 / 2 * Real.sqrt 3 * (1/4) * s^2) 
  (h₃ : A_tetrahedron = Real.sqrt 3 * s^2 / 4)
  :
  A_octahedron / A_tetrahedron = 2 := 
by
  sorry

end octahedron_tetrahedron_surface_area_ratio_l298_298671


namespace binom_arithmetic_sequence_l298_298347

noncomputable def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_arithmetic_sequence {n : ℕ} (h : 2 * binom n 5 = binom n 4 + binom n 6) (n_eq : n = 14) : binom n 12 = 91 := by
  sorry

end binom_arithmetic_sequence_l298_298347


namespace number_of_four_digit_cubes_divisible_by_16_l298_298713

theorem number_of_four_digit_cubes_divisible_by_16 :
  (finset.Icc 5 10).card = 6 :=
by sorry

end number_of_four_digit_cubes_divisible_by_16_l298_298713


namespace value_of_x_l298_298263

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l298_298263


namespace line_through_intersections_of_circles_l298_298708

-- Define the first circle
def circle₁ (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

-- Define the second circle
def circle₂ (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 20

-- The statement of the mathematically equivalent proof problem
theorem line_through_intersections_of_circles : 
    (∃ (x y : ℝ), circle₁ x y ∧ circle₂ x y) → (∃ (x y : ℝ), x + 3 * y - 5 = 0) :=
by
  intro h
  sorry

end line_through_intersections_of_circles_l298_298708


namespace card_count_l298_298238

theorem card_count (x y : ℕ) (h1 : x + y + 2 = 10) (h2 : 3 * x + 4 * y + 10 = 39) : x = 3 :=
by {
  sorry
}

end card_count_l298_298238


namespace complex_solutions_x2_eq_neg4_l298_298427

-- Lean statement for the proof problem
theorem complex_solutions_x2_eq_neg4 (x : ℂ) (hx : x^2 = -4) : x = 2 * Complex.I ∨ x = -2 * Complex.I :=
by 
  sorry

end complex_solutions_x2_eq_neg4_l298_298427


namespace knight_reachability_l298_298412

theorem knight_reachability (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) :
  (p + q) % 2 = 1 ∧ Nat.gcd p q = 1 ↔
  ∀ x y x' y', ∃ k h n m, x' = x + k * p + h * q ∧ y' = y + n * p + m * q :=
by
  sorry

end knight_reachability_l298_298412


namespace variance_of_data_set_l298_298581

open Real

def dataSet := [11, 12, 15, 18, 13, 15]

theorem variance_of_data_set :
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  variance = 16 / 3 :=
by
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  have h : mean = 14 := sorry
  have h_variance : variance = 16 / 3 := sorry
  exact h_variance

end variance_of_data_set_l298_298581


namespace find_a_from_conditions_l298_298846

theorem find_a_from_conditions (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 9) 
  (h3 : c = 4) : 
  a = -1 := 
by 
  sorry

end find_a_from_conditions_l298_298846


namespace height_of_middle_brother_l298_298943

theorem height_of_middle_brother (h₁ h₂ h₃ : ℝ) (h₁_le_h₂ : h₁ ≤ h₂) (h₂_le_h₃ : h₂ ≤ h₃)
  (avg_height : (h₁ + h₂ + h₃) / 3 = 1.74) (avg_height_tallest_shortest : (h₁ + h₃) / 2 = 1.75) :
  h₂ = 1.72 :=
by
  -- Proof to be filled here
  sorry

end height_of_middle_brother_l298_298943


namespace sqrt_one_div_four_is_one_div_two_l298_298088

theorem sqrt_one_div_four_is_one_div_two : Real.sqrt (1 / 4) = 1 / 2 :=
by
  sorry

end sqrt_one_div_four_is_one_div_two_l298_298088


namespace binary_modulo_eight_l298_298655

theorem binary_modulo_eight : (0b1110101101101 : ℕ) % 8 = 5 := 
by {
  -- This is where the proof would go.
  sorry
}

end binary_modulo_eight_l298_298655


namespace james_prom_total_cost_l298_298390

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l298_298390


namespace arrange_books_l298_298673

-- We define the conditions about the number of books
def num_algebra_books : ℕ := 4
def num_calculus_books : ℕ := 5
def total_books : ℕ := num_algebra_books + num_calculus_books

-- The combination function which calculates binomial coefficients
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem stating that there are 126 ways to arrange the books
theorem arrange_books : combination total_books num_algebra_books = 126 :=
  by
    sorry

end arrange_books_l298_298673


namespace exponent_division_simplification_l298_298817

theorem exponent_division_simplification :
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 :=
by
  sorry

end exponent_division_simplification_l298_298817


namespace range_of_m_l298_298995

noncomputable def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (m < 0)

noncomputable def q (m : ℝ) : Prop :=
  (16*(m-2)^2 - 16 < 0)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  intro h
  sorry

end range_of_m_l298_298995


namespace square_area_from_diagonal_l298_298804

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  (let s := d / Real.sqrt 2 in s * s) = 144 := by
sorry

end square_area_from_diagonal_l298_298804


namespace shifted_parabola_transformation_l298_298577

theorem shifted_parabola_transformation (x : ℝ) :
  let f := fun x => (x + 1)^2 + 3 in
  let f' := fun x => (x - 1)^2 + 2 in
  f (x - 2) - 1 = f' x :=
by
  sorry

end shifted_parabola_transformation_l298_298577


namespace cousin_points_correct_l298_298457

-- Conditions translated to definitions
def paul_points : ℕ := 3103
def total_points : ℕ := 5816

-- Dependent condition to get cousin's points
def cousin_points : ℕ := total_points - paul_points

-- The goal of our proof problem
theorem cousin_points_correct : cousin_points = 2713 :=
by
    sorry

end cousin_points_correct_l298_298457


namespace find_f_at_2_l298_298342

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem find_f_at_2 (a b : ℝ) 
  (h1 : 3 + 2 * a + b = 0) 
  (h2 : 1 + a + b + 1 = -2) : 
  f a b 2 = 3 := 
by
  dsimp [f]
  sorry

end find_f_at_2_l298_298342


namespace domain_w_l298_298333

noncomputable def w (y : ℝ) : ℝ := (y - 3)^(1/3) + (15 - y)^(1/3)

theorem domain_w : ∀ y : ℝ, ∃ x : ℝ, w y = x := by
  sorry

end domain_w_l298_298333


namespace powerjet_30_minutes_500_gallons_per_hour_l298_298254

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l298_298254


namespace number_of_people_in_group_l298_298725

variable (T L : ℕ)

theorem number_of_people_in_group
  (h1 : 90 + L = T)
  (h2 : (L : ℚ) / T = 0.4) :
  T = 150 := by
  sorry

end number_of_people_in_group_l298_298725


namespace fraction_of_alcohol_l298_298728

theorem fraction_of_alcohol (A : ℚ) (water_volume : ℚ) (alcohol_to_water_ratio : ℚ) 
  (h1 : water_volume = 4/5) 
  (h2 : alcohol_to_water_ratio = 3/4) 
  (h3 : A / water_volume = alcohol_to_water_ratio) : 
  A = 3/5 :=
by 
  rw [h1, h2] at h3
  field_simp at h3
  sorry

end fraction_of_alcohol_l298_298728


namespace mary_age_l298_298632

theorem mary_age (x : ℤ) (n m : ℤ) : (x - 2 = n^2) ∧ (x + 2 = m^3) → x = 6 := by
  sorry

end mary_age_l298_298632


namespace john_trip_duration_l298_298595

-- Definitions based on the conditions
def staysInFirstCountry : ℕ := 2
def staysInEachOtherCountry : ℕ := 2 * staysInFirstCountry

-- The proof problem statement
theorem john_trip_duration : (staysInFirstCountry + 2 * staysInEachOtherCountry) = 10 := 
begin
  sorry
end

end john_trip_duration_l298_298595


namespace proper_subset_of_A_l298_298481

def A : Set ℝ := {x | x^2 < 5 * x}

theorem proper_subset_of_A :
  (∀ x, x ∈ Set.Ioc 1 5 → x ∈ A ∧ ∀ y, y ∈ A → y ∉ Set.Ioc 1 5 → ¬(Set.Ioc 1 5 = A)) :=
sorry

end proper_subset_of_A_l298_298481


namespace calculate_shaded_area_l298_298126

noncomputable def square_shaded_area : ℝ := 
  let a := 10 -- side length of the square
  let s := a / 2 -- half side length, used for midpoints
  let total_area := a * a / 2 -- total area of a right triangle with legs a and a
  let triangle_DMA := total_area / 2 -- area of triangle DAM
  let triangle_DNG := triangle_DMA / 5 -- area of triangle DNG
  let triangle_CDM := total_area -- area of triangle CDM
  let shaded_area := triangle_CDM + triangle_DNG - triangle_DMA -- area of shaded region
  shaded_area

theorem calculate_shaded_area : square_shaded_area = 35 := 
by 
sorry

end calculate_shaded_area_l298_298126


namespace Powerjet_pumps_250_gallons_in_30_minutes_l298_298251

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l298_298251


namespace decreasing_interval_l298_298618

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem decreasing_interval : ∀ x ∈ Set.Ioo (Real.pi / 6) (5 * Real.pi / 6), 
  (1 / 2 - Real.sin x) < 0 := sorry

end decreasing_interval_l298_298618


namespace total_earnings_in_september_l298_298035

theorem total_earnings_in_september (
  mowing_rate: ℕ := 6
  mowing_hours: ℕ := 63
  weeds_rate: ℕ := 11
  weeds_hours: ℕ := 9
  mulch_rate: ℕ := 9
  mulch_hours: ℕ := 10
): 
  mowing_rate * mowing_hours + weeds_rate * weeds_hours + mulch_rate * mulch_hours = 567 := 
by
  sorry

end total_earnings_in_september_l298_298035


namespace minimum_value_of_x_plus_2y_l298_298837

open Real

theorem minimum_value_of_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 := by
  sorry

end minimum_value_of_x_plus_2y_l298_298837


namespace polynomial_remainder_l298_298182

theorem polynomial_remainder (x : ℝ) : 
  (x - 1)^100 + (x - 2)^200 = (x^2 - 3 * x + 2) * (some_q : ℝ) + 1 :=
sorry

end polynomial_remainder_l298_298182


namespace green_balls_in_bag_l298_298796

theorem green_balls_in_bag (b : ℕ) (P_blue : ℚ) (g : ℕ) (h1 : b = 8) (h2 : P_blue = 1 / 3) (h3 : P_blue = (b : ℚ) / (b + g)) :
  g = 16 :=
by
  sorry

end green_balls_in_bag_l298_298796


namespace mean_noon_temperature_l298_298432

def temperatures : List ℝ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_noon_temperature :
  (List.sum temperatures) / (temperatures.length) = 770 / 9 := by
  sorry

end mean_noon_temperature_l298_298432


namespace smallest_integer_with_20_divisors_l298_298650

theorem smallest_integer_with_20_divisors :
  ∃ n : ℕ, (∀ k : ℕ, k ∣ n → k > 0) ∧ n = 432 ∧ (∃ (p1 p2 : ℕ) (a1 a2 : ℕ),
    p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ (a1 + 1) * (a2 + 1) = 20 ∧ n = p1^a1 * p2^a2) :=
sorry

end smallest_integer_with_20_divisors_l298_298650


namespace value_of_x_l298_298380

theorem value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 6 = 3 * y) : x = 108 :=
by
  sorry

end value_of_x_l298_298380


namespace inequality_reciprocal_of_negative_l298_298368

variable {a b : ℝ}

theorem inequality_reciprocal_of_negative (h : a < b) (h_neg_a : a < 0) (h_neg_b : b < 0) : 
  (1 / a) > (1 / b) := by
  sorry

end inequality_reciprocal_of_negative_l298_298368


namespace investment_time_l298_298258

theorem investment_time
  (p_investment_ratio : ℚ) (q_investment_ratio : ℚ)
  (profit_ratio_p : ℚ) (profit_ratio_q : ℚ)
  (q_investment_time : ℕ)
  (h1 : p_investment_ratio / q_investment_ratio = 7 / 5)
  (h2 : profit_ratio_p / profit_ratio_q = 7 / 10)
  (h3 : q_investment_time = 40) :
  ∃ t : ℚ, t = 28 :=
by
  sorry

end investment_time_l298_298258


namespace bill_harry_combined_l298_298127

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l298_298127


namespace find_y_l298_298359

-- Definitions of vectors and parallel relationship
def vector_a : ℝ × ℝ := (4, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (6, y)
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- The theorem we want to prove
theorem find_y (y : ℝ) (h : parallel vector_a (vector_b y)) : y = 3 :=
sorry

end find_y_l298_298359


namespace equation_has_two_distinct_roots_l298_298159

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l298_298159


namespace Mitya_age_l298_298235

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l298_298235


namespace t_mobile_first_two_lines_cost_l298_298876

theorem t_mobile_first_two_lines_cost :
  ∃ T : ℝ,
  (T + 16 * 3) = (45 + 14 * 3 + 11) → T = 50 :=
by
  sorry

end t_mobile_first_two_lines_cost_l298_298876


namespace domino_trick_l298_298969

theorem domino_trick (x y : ℕ) (h1 : x ≤ 6) (h2 : y ≤ 6)
  (h3 : 10 * x + y + 30 = 62) : x = 3 ∧ y = 2 :=
by
  sorry

end domino_trick_l298_298969


namespace smoothie_cost_l298_298605

-- Definitions of costs and amounts paid.
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def amount_paid : ℕ := 20
def change_received : ℕ := 11

-- Define the total cost of the order and the known costs.
def total_order_cost : ℕ := amount_paid - change_received
def known_costs : ℕ := hamburger_cost + onion_rings_cost

-- State the problem: the cost of the smoothie.
theorem smoothie_cost : total_order_cost - known_costs = 3 :=
by 
  sorry

end smoothie_cost_l298_298605


namespace Mitya_age_l298_298234

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l298_298234


namespace line_through_point_hyperbola_l298_298968

theorem line_through_point_hyperbola {x y k : ℝ} : 
  (∃ k : ℝ, ∃ x y : ℝ, y = k * (x - 3) ∧ x^2 / 4 - y^2 = 1 ∧ (1 - 4 * k^2) = 0) → 
  (∃! k : ℝ, (k = 1 / 2) ∨ (k = -1 / 2)) := 
sorry

end line_through_point_hyperbola_l298_298968


namespace range_of_a_l298_298569
noncomputable section

open Real

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * x + a + 2 > 0) : a > -1 :=
sorry

end range_of_a_l298_298569


namespace problem_M_l298_298845

theorem problem_M (M : ℤ) (h : 1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M) : M = 35 :=
by
  sorry

end problem_M_l298_298845


namespace alice_has_ball_after_two_turns_l298_298810

noncomputable def probability_alice_has_ball_twice_turns : ℚ :=
  let P_AB_A : ℚ := 1/2 * 1/3
  let P_ABC_A : ℚ := 1/2 * 1/3 * 1/2
  let P_AA : ℚ := 1/2 * 1/2
  P_AB_A + P_ABC_A + P_AA

theorem alice_has_ball_after_two_turns :
  probability_alice_has_ball_twice_turns = 1/2 := 
by
  sorry

end alice_has_ball_after_two_turns_l298_298810


namespace smallest_lcm_l298_298374

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l298_298374


namespace cubic_has_three_zeros_l298_298545

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298545


namespace all_numbers_divisible_by_5_l298_298608

variable {a b c d e f g : ℕ}

-- Seven natural numbers and the condition that the sum of any six is divisible by 5
axiom cond_a : (a + b + c + d + e + f) % 5 = 0
axiom cond_b : (b + c + d + e + f + g) % 5 = 0
axiom cond_c : (a + c + d + e + f + g) % 5 = 0
axiom cond_d : (a + b + c + e + f + g) % 5 = 0
axiom cond_e : (a + b + c + d + f + g) % 5 = 0
axiom cond_f : (a + b + c + d + e + g) % 5 = 0
axiom cond_g : (a + b + c + d + e + f) % 5 = 0

theorem all_numbers_divisible_by_5 :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end all_numbers_divisible_by_5_l298_298608


namespace range_of_m_l298_298029

def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
def g (m x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
by
  sorry

end range_of_m_l298_298029


namespace rectangular_prism_volume_is_60_l298_298470

def rectangularPrismVolume (a b c : ℕ) : ℕ := a * b * c 

theorem rectangular_prism_volume_is_60 (a b c : ℕ) 
  (h_ge_2 : a ≥ 2) (h_ge_2_b : b ≥ 2) (h_ge_2_c : c ≥ 2)
  (h_one_face : 2 * ((a-2)*(b-2) + (b-2)*(c-2) + (a-2)*(c-2)) = 24)
  (h_two_faces : 4 * ((a-2) + (b-2) + (c-2)) = 28) :
  rectangularPrismVolume a b c = 60 := 
  by sorry

end rectangular_prism_volume_is_60_l298_298470


namespace calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l298_298487

def probability_10_ring : ℝ := 0.13
def probability_9_ring : ℝ := 0.28
def probability_8_ring : ℝ := 0.31

def probability_10_or_9_ring : ℝ := probability_10_ring + probability_9_ring

def probability_less_than_9_ring : ℝ := 1 - probability_10_or_9_ring

theorem calc_probability_10_or_9_ring :
  probability_10_or_9_ring = 0.41 :=
by
  sorry

theorem calc_probability_less_than_9_ring :
  probability_less_than_9_ring = 0.59 :=
by
  sorry

end calc_probability_10_or_9_ring_calc_probability_less_than_9_ring_l298_298487


namespace part1_part2_l298_298840

open Set

variable {R : Type} [OrderedRing R]

def U : Set R := univ
def A : Set R := {x | x^2 - 2*x - 3 > 0}
def B : Set R := {x | 4 - x^2 <= 0}

theorem part1 : A ∩ B = {x | -2 ≤ x ∧ x < -1} :=
sorry

theorem part2 : (U \ A) ∪ (U \ B) = {x | x < -2 ∨ x > -1} :=
sorry

end part1_part2_l298_298840


namespace jen_hours_per_week_l298_298395

theorem jen_hours_per_week (B : ℕ) (h1 : ∀ t : ℕ, t * (B + 7) = 6 * B) : B + 7 = 21 := by
  sorry

end jen_hours_per_week_l298_298395


namespace find_number_l298_298991

theorem find_number (x : ℝ) (h : x + (2/3) * x + 1 = 10) : x = 27/5 := 
by
  sorry

end find_number_l298_298991


namespace circle_radius_condition_l298_298291

theorem circle_radius_condition (c : ℝ) : 
  (∃ x y : ℝ, (x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ 
  (radius = 6) ↔ 
  c = -23 := by
  sorry

end circle_radius_condition_l298_298291


namespace range_of_a_l298_298838

def valid_real_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a

theorem range_of_a :
  (∀ a : ℝ, (¬ valid_real_a a)) ↔ (a < 1 ∨ a > 3) :=
sorry

end range_of_a_l298_298838


namespace value_of_x_l298_298264

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l298_298264


namespace sum_of_squares_neq_fourth_powers_l298_298000

theorem sum_of_squares_neq_fourth_powers (m n : ℕ) : 
  m^2 + (m + 1)^2 ≠ n^4 + (n + 1)^4 :=
by 
  sorry

end sum_of_squares_neq_fourth_powers_l298_298000


namespace median_length_of_right_triangle_l298_298030

noncomputable def length_of_median (a b c : ℕ) : ℝ := 
  if a * a + b * b = c * c then c / 2 else 0

theorem median_length_of_right_triangle :
  length_of_median 9 12 15 = 7.5 :=
by
  -- Insert the proof here
  sorry

end median_length_of_right_triangle_l298_298030


namespace snowboard_price_after_discounts_l298_298411

noncomputable def final_snowboard_price (P_original : ℝ) (d_Friday : ℝ) (d_Monday : ℝ) : ℝ :=
  P_original * (1 - d_Friday) * (1 - d_Monday)

theorem snowboard_price_after_discounts :
  final_snowboard_price 100 0.50 0.30 = 35 :=
by 
  sorry

end snowboard_price_after_discounts_l298_298411


namespace certain_fraction_is_half_l298_298511

theorem certain_fraction_is_half (n : ℕ) (fraction : ℚ) (h : (37 + 1/2) / fraction = 75) : fraction = 1/2 :=
by
    sorry

end certain_fraction_is_half_l298_298511


namespace shaded_areas_total_l298_298582

theorem shaded_areas_total (r R : ℝ) (h_divides : ∀ (A : ℝ), ∃ (B : ℝ), B = A / 3)
  (h_center : True) (h_area : π * R^2 = 81 * π) :
  (π * R^2 / 3) + (π * (R / 2)^2 / 3) = 33.75 * π :=
by
  -- The proof here will be added.
  sorry

end shaded_areas_total_l298_298582


namespace gcd_x_y_not_8_l298_298571

theorem gcd_x_y_not_8 (x y : ℕ) (hx : x > 0) (hy : y = x^2 + 8) : ¬ ∃ d, d = 8 ∧ d ∣ x ∧ d ∣ y :=
by
  sorry

end gcd_x_y_not_8_l298_298571


namespace part1_part2_l298_298841

variable (x y z : ℕ)

theorem part1 (h1 : 3 * x + 5 * y = 98) (h2 : 8 * x + 3 * y = 158) : x = 16 ∧ y = 10 :=
sorry

theorem part2 (hx : x = 16) (hy : y = 10) (hz : 16 * z + 10 * (40 - z) ≤ 550) : z ≤ 25 :=
sorry

end part1_part2_l298_298841


namespace pollen_scientific_notation_correct_l298_298211

def moss_flower_pollen_diameter := 0.0000084
def pollen_scientific_notation := 8.4 * 10^(-6)

theorem pollen_scientific_notation_correct :
  moss_flower_pollen_diameter = pollen_scientific_notation :=
by
  -- Proof skipped
  sorry

end pollen_scientific_notation_correct_l298_298211


namespace angle_A_in_triangle_l298_298572

theorem angle_A_in_triangle (a b c : ℝ) (h : a^2 = b^2 + b * c + c^2) : A = 120 :=
sorry

end angle_A_in_triangle_l298_298572


namespace concentration_of_acid_in_third_flask_is_correct_l298_298270

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l298_298270


namespace ratio_albert_betty_l298_298675

theorem ratio_albert_betty (A M B : ℕ) (h1 : A = 2 * M) (h2 : M = A - 10) (h3 : B = 5) :
  A / B = 4 :=
by
  -- the proof goes here
  sorry

end ratio_albert_betty_l298_298675


namespace div_by_seven_iff_multiple_of_three_l298_298466

theorem div_by_seven_iff_multiple_of_three (n : ℕ) (hn : 0 < n) : 
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := 
sorry

end div_by_seven_iff_multiple_of_three_l298_298466


namespace car_mileage_city_l298_298663

theorem car_mileage_city {h c t : ℝ} (H1: 448 = h * t) (H2: 336 = c * t) (H3: c = h - 6) : c = 18 :=
sorry

end car_mileage_city_l298_298663


namespace tangent_length_is_five_sqrt_two_l298_298001

noncomputable def point := (ℝ × ℝ)
noncomputable def distance (p1 p2 : point) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2
noncomputable def is_on_circle (p : point) (center : point) (radius : ℝ) : Prop :=
  distance p center = radius ^ 2

noncomputable def circumncircle_radius (A B C : point) : ℝ := 
  let h := ... -- some expression to determine h
  let k := ... -- some expression to determine k
  let r := distance A (h,k) -- radius from one of the points to the center
  r

noncomputable def length_of_tangent (P A B C : point) : ℝ :=
  let r := circumncircle_radius A B C
  let dist_PA := sqrt $ distance P A
  let dist_PB := sqrt $ distance P B
  sqrt (dist_PA * dist_PB)

noncomputable def length_of_segment_tangent : ℝ :=
  length_of_tangent (1,1) (4,5) (7,9) (6,14)

theorem tangent_length_is_five_sqrt_two :
  length_of_segment_tangent = 5 * sqrt 2 := by
  sorry

end tangent_length_is_five_sqrt_two_l298_298001


namespace walter_time_spent_at_seals_l298_298450

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l298_298450


namespace model_represents_feet_l298_298768

def height_statue : ℝ := 120
def height_model : ℝ := 6
def feet_per_inch_model : ℝ := height_statue / height_model

theorem model_represents_feet : feet_per_inch_model = 20 := 
by
  sorry

end model_represents_feet_l298_298768


namespace fixed_point_of_function_l298_298430

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : (2, 3) ∈ { (x, y) | y = 2 + a^(x-2) } :=
sorry

end fixed_point_of_function_l298_298430


namespace more_non_persistent_days_l298_298575

-- Definitions based on the problem's conditions
/-
 * Let n > 4 be the number of athletes.
 * Define each player.
 * Define the notion of persistence.
 * Formulate the games and the total number of game days.
 * Define and state the problem clearly in Lean.
-/

structure Player := 
  (id : ℕ)

def isPersistent (wins : List (Player × Player)) (player : Player) : Prop := 
  ∃ (first_win : wins), 
    (first_win.1 = player ∧
      (∀ (later_game : wins), later_game.1 = player → later_game.1 = first_win))

def playedAgainstAll (games : List (Player × Player)) (player : Player) : Prop := 
  ∀ (other : Player), other ≠ player → ∃ (game : games), 
    (game.1 = player ∧ game.2 = other) ∨
    (game.1 = other ∧ game.2 = player)

def hadNonPersistentDays (games : List (Player × Player)) : Prop := 
  sorry -- To define the exact number of days non-persistent players played against each other

theorem more_non_persistent_days (n : ℕ) (games : List (Player × Player)) 
  (h1 : n > 4)
  (h2 : ∀ player, (playedAgainstAll games player))
  (h3 : ∀ player, (∃ win, win ∈ games ∧ isPersistent games player) ∨ (¬ isPersistent games player)) :
  hadNonPersistentDays games := 
sorry -- The proof would be here

end more_non_persistent_days_l298_298575


namespace number_of_students_above_120_l298_298119

def normal_distribution (μ σ : ℝ) : Prop := sorry -- Assumes the definition of a normal distribution

-- Given conditions
def number_of_students := 1000
def μ := 100
def some_distribution (ξ : ℝ) := normal_distribution μ (σ^2)
def probability_between (a b : ℝ) : ℝ := sorry -- Assumes the definition to calculate probabilities in normal distribution
def given_probability := (probability_between 80 100) = 0.45

-- Question and proof problem
theorem number_of_students_above_120 (σ : ℝ) (h : some_distribution ξ) (hp : given_probability) :
  ∃ n : ℕ, n = 50 := by
    sorry

end number_of_students_above_120_l298_298119


namespace min_value_is_correct_l298_298383

noncomputable def min_value (P : ℝ × ℝ) (A B C : ℝ × ℝ) : ℝ := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 +
  PB.1 * PC.1 + PB.2 * PC.2 +
  PC.1 * PA.1 + PC.2 * PA.2

theorem min_value_is_correct :
  ∃ P : ℝ × ℝ, P = (5/3, 1/3) ∧
  min_value P (1, 4) (4, 1) (0, -4) = -62/3 :=
by
  sorry

end min_value_is_correct_l298_298383


namespace time_to_fill_pool_l298_298609

-- Define constants based on the conditions
def pool_capacity : ℕ := 30000
def hose_count : ℕ := 5
def flow_rate_per_hose : ℕ := 25 / 10  -- 2.5 gallons per minute
def conversion_minutes_to_hours : ℕ := 60

-- Define the total flow rate per minute
def total_flow_rate_per_minute : ℕ := hose_count * flow_rate_per_hose

-- Define the total flow rate per hour
def total_flow_rate_per_hour : ℕ := total_flow_rate_per_minute * conversion_minutes_to_hours

-- Theorem stating the number of hours required to fill the pool
theorem time_to_fill_pool : pool_capacity / total_flow_rate_per_hour = 40 := by
  sorry -- Proof will be provided here

end time_to_fill_pool_l298_298609


namespace probability_two_yellow_apples_l298_298392

theorem probability_two_yellow_apples (total_apples : ℕ) (red_apples : ℕ) (green_apples : ℕ) (yellow_apples : ℕ) (choose : ℕ → ℕ → ℕ) (probability : ℕ → ℕ → ℝ) :
  total_apples = 10 →
  red_apples = 5 →
  green_apples = 3 →
  yellow_apples = 2 →
  choose total_apples 2 = 45 →
  choose yellow_apples 2 = 1 →
  probability (choose yellow_apples 2) (choose total_apples 2) = 1 / 45 := 
  by
  sorry

end probability_two_yellow_apples_l298_298392


namespace second_year_associates_l298_298213

theorem second_year_associates (not_first_year : ℝ) (more_than_two_years : ℝ) 
  (h1 : not_first_year = 0.75) (h2 : more_than_two_years = 0.5) : 
  (not_first_year - more_than_two_years) = 0.25 :=
by 
  sorry

end second_year_associates_l298_298213


namespace function_has_three_zeros_l298_298531

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298531


namespace function_identity_l298_298324

theorem function_identity (f : ℕ → ℕ) 
  (h_pos : f 1 > 0) 
  (h_property : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l298_298324


namespace find_a_l298_298723

theorem find_a :
  ∃ a : ℝ, 
    (∀ x : ℝ, f x = 3 * x + a * x^3) ∧ 
    (f 1 = a + 3) ∧ 
    (∃ k : ℝ, k = 6 ∧ k = deriv f 1 ∧ ((∀ x : ℝ, deriv f x = 3 + 3 * a * x^2))) → 
    a = 1 :=
by sorry

end find_a_l298_298723


namespace max_blocks_l298_298299

theorem max_blocks (box_height box_width box_length : ℝ) 
  (typeA_height typeA_width typeA_length typeB_height typeB_width typeB_length : ℝ) 
  (h_box : box_height = 8) (w_box : box_width = 10) (l_box : box_length = 12) 
  (h_typeA : typeA_height = 3) (w_typeA : typeA_width = 2) (l_typeA : typeA_length = 4) 
  (h_typeB : typeB_height = 4) (w_typeB : typeB_width = 3) (l_typeB : typeB_length = 5) : 
  max (⌊box_height / typeA_height⌋ * ⌊box_width / typeA_width⌋ * ⌊box_length / typeA_length⌋)
      (⌊box_height / typeB_height⌋ * ⌊box_width / typeB_width⌋ * ⌊box_length / typeB_length⌋) = 30 := 
  by
  sorry

end max_blocks_l298_298299


namespace cos_alpha_minus_7pi_over_2_l298_298496

-- Given conditions
variable (α : Real) (h : Real.sin α = 3/5)

-- Statement to prove
theorem cos_alpha_minus_7pi_over_2 : Real.cos (α - 7 * Real.pi / 2) = -3/5 :=
by
  sorry

end cos_alpha_minus_7pi_over_2_l298_298496


namespace combinations_medical_team_l298_298628

noncomputable def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_medical_team : 
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  numWaysMale * numWaysFemale = 75 :=
by
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  show numWaysMale * numWaysFemale = 75 
  sorry

end combinations_medical_team_l298_298628


namespace second_sum_is_1704_l298_298459

theorem second_sum_is_1704
    (total_sum : ℝ)
    (x : ℝ)
    (interest_rate_first_part : ℝ)
    (time_first_part : ℝ)
    (interest_rate_second_part : ℝ)
    (time_second_part : ℝ)
    (h1 : total_sum = 2769)
    (h2 : interest_rate_first_part = 3)
    (h3 : time_first_part = 8)
    (h4 : interest_rate_second_part = 5)
    (h5 : time_second_part = 3)
    (h6 : 24 * x / 100 = (total_sum - x) * 15 / 100) :
    total_sum - x = 1704 :=
  by
    sorry

end second_sum_is_1704_l298_298459


namespace general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l298_298434

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℕ := 2 ^ (a n)

noncomputable def S (n : ℕ) : ℕ := (n * (2 * n + 2)) / 2

noncomputable def T (n : ℕ) : ℕ := (8 * (4 ^ n - 1)) / 3

-- Statements to be proved
theorem general_formula_an : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem geometric_sequence_bn : ∀ n : ℕ, b n = 2 ^ (2 * n + 1) := sorry

theorem sum_of_geometric_sequence_Tn : ∀ n : ℕ, T n = (8 * (4 ^ n - 1)) / 3 := sorry

end general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l298_298434


namespace circle_radius_l298_298773

-- Given the equation of a circle, we want to prove its radius
theorem circle_radius : ∀ (x y : ℝ), x^2 + y^2 - 6*y - 16 = 0 → (∃ r, r = 5) :=
  by
    sorry

end circle_radius_l298_298773


namespace three_zeros_implies_a_lt_neg3_l298_298565

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298565


namespace percentage_material_B_in_final_mixture_l298_298422

-- Conditions
def percentage_material_A_in_Solution_X : ℝ := 20
def percentage_material_B_in_Solution_X : ℝ := 80
def percentage_material_A_in_Solution_Y : ℝ := 30
def percentage_material_B_in_Solution_Y : ℝ := 70
def percentage_material_A_in_final_mixture : ℝ := 22

-- Goal
theorem percentage_material_B_in_final_mixture :
  100 - percentage_material_A_in_final_mixture = 78 := by
  sorry

end percentage_material_B_in_final_mixture_l298_298422


namespace total_cost_of_books_l298_298714

theorem total_cost_of_books (C1 C2 : ℝ) 
  (hC1 : C1 = 268.33)
  (h_selling_prices_equal : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 459.15 :=
by
  -- placeholder for the proof
  sorry

end total_cost_of_books_l298_298714


namespace bill_and_harry_nuts_l298_298129

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l298_298129


namespace lemonade_sales_l298_298948

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l298_298948


namespace profit_equations_l298_298964

-- Define the conditions
def total_workers : ℕ := 150
def fabric_per_worker_per_day : ℕ := 30
def clothing_per_worker_per_day : ℕ := 4
def fabric_needed_per_clothing : ℝ := 1.5
def profit_per_meter : ℝ := 2
def profit_per_clothing : ℝ := 25

-- Define the profit functions
def profit_clothing (x : ℕ) : ℝ := profit_per_clothing * clothing_per_worker_per_day * x
def profit_fabric (x : ℕ) : ℝ := profit_per_meter * (fabric_per_worker_per_day * (total_workers - x) - fabric_needed_per_clothing * clothing_per_worker_per_day * x)

-- Define the total profit function
def total_profit (x : ℕ) : ℝ := profit_clothing x + profit_fabric x

-- Prove the given statements
theorem profit_equations (x : ℕ) :
  profit_clothing x = 100 * x ∧
  profit_fabric x = 9000 - 72 * x ∧
  total_profit 100 = 11800 :=
by
  -- Proof omitted
  sorry

end profit_equations_l298_298964


namespace smallest_positive_integer_with_20_divisors_is_432_l298_298641

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end smallest_positive_integer_with_20_divisors_is_432_l298_298641


namespace range_of_a_for_three_zeros_l298_298552

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298552


namespace sequence_solution_l298_298747

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 5 ∧ a 8 = 8 ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 20) ∧
  (a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 5 ∧ a 5 = 8 ∧ a 6 = 7 ∧ a 7 = 5 ∧ a 8 = 8) :=
by {
  sorry
}

end sequence_solution_l298_298747


namespace unique_solution_l298_298323

def system_of_equations (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) :=
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧
  a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_solution
  (x1 x2 x3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h_pos: 0 < a11 ∧ 0 < a22 ∧ 0 < a33)
  (h_neg: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0)
  (h_sum_pos: 0 < a11 + a12 + a13 ∧ 0 < a21 + a22 + a23 ∧ 0 < a31 + a32 + a33)
  (h_system: system_of_equations a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3):
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := sorry

end unique_solution_l298_298323


namespace wire_pieces_difference_l298_298975

theorem wire_pieces_difference (L1 L2 : ℝ) (H1 : L1 = 14) (H2 : L2 = 16) : L2 - L1 = 2 :=
by
  rw [H1, H2]
  norm_num

end wire_pieces_difference_l298_298975


namespace number_of_terms_in_expansion_l298_298132

theorem number_of_terms_in_expansion (A B : Finset ℕ) (h1 : A.card = 4) (h2 : B.card = 5) :
  (A.product B).card = 20 :=
by
  sorry

end number_of_terms_in_expansion_l298_298132


namespace correct_ordering_of_powers_l298_298290

theorem correct_ordering_of_powers : 
  7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 :=
  by
    sorry

end correct_ordering_of_powers_l298_298290


namespace greatest_x_lcm_105_l298_298915

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298915


namespace paint_needed_to_buy_l298_298008

def total_paint := 333
def existing_paint := 157

theorem paint_needed_to_buy : total_paint - existing_paint = 176 := by
  sorry

end paint_needed_to_buy_l298_298008


namespace trapezoid_area_l298_298112

theorem trapezoid_area (u l h : ℕ) (hu : u = 12) (hl : l = u + 4) (hh : h = 10) : 
  (1 / 2 : ℚ) * (u + l) * h = 140 := by
  sorry

end trapezoid_area_l298_298112


namespace correct_answer_l298_298002

theorem correct_answer (x : ℝ) (h1 : 2 * x = 60) : x / 2 = 15 :=
by
  sorry

end correct_answer_l298_298002


namespace cloth_sold_l298_298473

theorem cloth_sold (C S P: ℝ) (N : ℕ) 
  (h1 : S = 3 * C)
  (h2 : P = 10 * S)
  (h3 : (200 : ℝ) = (P / (N * C)) * 100) : N = 15 := 
sorry

end cloth_sold_l298_298473


namespace value_of_expression_l298_298369

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l298_298369


namespace grace_earnings_in_september_l298_298036

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end grace_earnings_in_september_l298_298036


namespace find_t_l298_298206

-- Defining variables and assumptions
variables (V V0 g S t : Real)
variable (h1 : V = g * t + V0)
variable (h2 : S = (1 / 2) * g * t^2 + V0 * t)

-- The goal: to prove t equals 2S / (V + V0)
theorem find_t (V V0 g S t : Real) (h1 : V = g * t + V0) (h2 : S = (1 / 2) * g * t^2 + V0 * t):
  t = 2 * S / (V + V0) := by
  sorry

end find_t_l298_298206


namespace proof_l298_298346

noncomputable def problem_statement (a b : ℝ) :=
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1)

theorem proof : ∀ a b : ℝ, problem_statement a b := sorry

end proof_l298_298346


namespace highest_power_of_3_dividing_N_is_1_l298_298765

-- Define the integer N as described in the problem
def N : ℕ := 313233515253

-- State the problem
theorem highest_power_of_3_dividing_N_is_1 : ∃ k : ℕ, (3^k ∣ N) ∧ ∀ m > 1, ¬ (3^m ∣ N) ∧ k = 1 :=
by
  -- Specific solution details and steps are not required here
  sorry

end highest_power_of_3_dividing_N_is_1_l298_298765


namespace smallest_integer_with_20_divisors_l298_298649

theorem smallest_integer_with_20_divisors :
  ∃ n : ℕ, (∀ k : ℕ, k ∣ n → k > 0) ∧ n = 432 ∧ (∃ (p1 p2 : ℕ) (a1 a2 : ℕ),
    p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ (a1 + 1) * (a2 + 1) = 20 ∧ n = p1^a1 * p2^a2) :=
sorry

end smallest_integer_with_20_divisors_l298_298649


namespace flag_covering_proof_l298_298661

def grid_covering_flag_ways (m n num_flags cells_per_flag : ℕ) :=
  if m * n / cells_per_flag = num_flags then 2^num_flags else 0

theorem flag_covering_proof :
  grid_covering_flag_ways 9 18 18 9 = 262144 := by
  sorry

end flag_covering_proof_l298_298661


namespace total_weight_marble_purchased_l298_298073

theorem total_weight_marble_purchased (w1 w2 w3 : ℝ) (h1 : w1 = 0.33) (h2 : w2 = 0.33) (h3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 := by
  sorry

end total_weight_marble_purchased_l298_298073


namespace first_line_shift_time_l298_298286

theorem first_line_shift_time (x y : ℝ) (h1 : (1 / x) + (1 / (x - 2)) + (1 / y) = 1.5 * ((1 / x) + (1 / (x - 2)))) 
  (h2 : x - 24 / 5 = (1 / ((1 / (x - 2)) + (1 / y)))) :
  x = 8 :=
sorry

end first_line_shift_time_l298_298286


namespace sum_of_tens_and_units_digit_of_7_pow_2023_l298_298453

theorem sum_of_tens_and_units_digit_of_7_pow_2023 :
  let n := 7 ^ 2023
  (n % 100).div 10 + (n % 10) = 16 :=
by
  sorry

end sum_of_tens_and_units_digit_of_7_pow_2023_l298_298453


namespace altitude_circumradius_relation_l298_298239

variable (a b c R ha : ℝ)
-- Assume S is the area of the triangle
variable (S : ℝ)
-- conditions
axiom area_circumradius : S = (a * b * c) / (4 * R)
axiom area_altitude : S = (a * ha) / 2

-- Prove the equivalence
theorem altitude_circumradius_relation 
  (area_circumradius : S = (a * b * c) / (4 * R)) 
  (area_altitude : S = (a * ha) / 2) : 
  ha = (b * c) / (2 * R) :=
sorry

end altitude_circumradius_relation_l298_298239


namespace sequence_general_term_l298_298189

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1) :
  (∀ n, a n = if n = 1 then 2 else 2 * n - 1) :=
by
  sorry

end sequence_general_term_l298_298189


namespace auction_theorem_l298_298037

def auctionProblem : Prop :=
  let starting_value := 300
  let harry_bid_round1 := starting_value + 200
  let alice_bid_round1 := harry_bid_round1 * 2
  let bob_bid_round1 := harry_bid_round1 * 3
  let highest_bid_round1 := bob_bid_round1
  let carol_bid_round2 := highest_bid_round1 * 1.5
  let sum_previous_increases := (harry_bid_round1 - starting_value) + 
                                 (alice_bid_round1 - harry_bid_round1) + 
                                 (bob_bid_round1 - harry_bid_round1)
  let dave_bid_round2 := carol_bid_round2 + sum_previous_increases
  let highest_other_bid_round3 := dave_bid_round2
  let harry_final_bid_round3 := 6000
  let difference := harry_final_bid_round3 - highest_other_bid_round3
  difference = 2050

theorem auction_theorem : auctionProblem :=
by
  sorry

end auction_theorem_l298_298037


namespace train_average_speed_with_stoppages_l298_298787

theorem train_average_speed_with_stoppages (D : ℝ) :
  let speed_without_stoppages := 200
  let stoppage_time_per_hour_in_hours := 12 / 60.0
  let effective_running_time := 1 - stoppage_time_per_hour_in_hours
  let speed_with_stoppages := effective_running_time * speed_without_stoppages
  speed_with_stoppages = 160 := by
  sorry

end train_average_speed_with_stoppages_l298_298787


namespace equation_has_at_least_two_distinct_roots_l298_298161

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l298_298161


namespace find_overlap_length_l298_298485

-- Define the given conditions
def plank_length : ℝ := 30 -- length of each plank in cm
def number_of_planks : ℕ := 25 -- number of planks
def total_fence_length : ℝ := 690 -- total length of the fence in cm

-- Definition for the overlap length
def overlap_length (y : ℝ) : Prop :=
  total_fence_length = (13 * plank_length) + (12 * (plank_length - 2 * y))

-- Theorem statement to prove the required overlap length
theorem find_overlap_length : ∃ y : ℝ, overlap_length y ∧ y = 2.5 :=
by 
  -- The proof goes here
  sorry

end find_overlap_length_l298_298485


namespace algae_coverage_day_21_l298_298612

-- Let "algae_coverage n" denote the percentage of lake covered by algae on day n.
noncomputable def algaeCoverage : ℕ → ℝ
| 0 => 1 -- initial state on day 0 taken as baseline (can be adjusted accordingly)
| (n+1) => 2 * algaeCoverage n

-- Define the problem statement
theorem algae_coverage_day_21 :
  algaeCoverage 24 = 100 → algaeCoverage 21 = 12.5 :=
by
  sorry

end algae_coverage_day_21_l298_298612


namespace experts_expected_points_probability_fifth_envelope_l298_298591

theorem experts_expected_points (n : ℕ) (h1 : n = 100) (h2 : n = 13) :
  ∃ e : ℚ, e = 465 :=
sorry

theorem probability_fifth_envelope (m : ℕ) (h1 : m = 13) :
  ∃ p : ℚ, p = 0.715 :=
sorry

end experts_expected_points_probability_fifth_envelope_l298_298591


namespace crayons_left_l298_298942

-- Define the initial number of crayons and the number taken
def initial_crayons : ℕ := 7
def crayons_taken : ℕ := 3

-- Prove the number of crayons left in the drawer
theorem crayons_left : initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l298_298942


namespace greatest_possible_x_max_possible_x_l298_298895

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298895


namespace value_of_x_l298_298266

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l298_298266


namespace machine_c_more_bottles_l298_298092

theorem machine_c_more_bottles (A B C : ℕ) 
  (hA : A = 12)
  (hB : B = A - 2)
  (h_total : 10 * A + 10 * B + 10 * C = 370) :
  C - B = 5 :=
by
  sorry

end machine_c_more_bottles_l298_298092


namespace tonya_large_lemonade_sales_l298_298947

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l298_298947


namespace concentration_third_flask_l298_298281

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l298_298281


namespace period_length_divisor_l298_298751

theorem period_length_divisor (p d : ℕ) (hp_prime : Nat.Prime p) (hd_period : ∀ n : ℕ, n ≥ 1 → 10^n % p = 1 ↔ n = d) :
  d ∣ (p - 1) :=
sorry

end period_length_divisor_l298_298751


namespace solve_for_x_l298_298610

theorem solve_for_x : ∀ x : ℚ, 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l298_298610


namespace miranda_saves_half_of_salary_l298_298242

noncomputable def hourly_wage := 10
noncomputable def daily_hours := 10
noncomputable def weekly_days := 5
noncomputable def weekly_salary := hourly_wage * daily_hours * weekly_days

noncomputable def robby_saving_fraction := 2 / 5
noncomputable def jaylen_saving_fraction := 3 / 5
noncomputable def total_savings := 3000
noncomputable def weeks := 4

noncomputable def robby_weekly_savings := robby_saving_fraction * weekly_salary
noncomputable def jaylen_weekly_savings := jaylen_saving_fraction * weekly_salary
noncomputable def robby_total_savings := robby_weekly_savings * weeks
noncomputable def jaylen_total_savings := jaylen_weekly_savings * weeks
noncomputable def combined_savings_rj := robby_total_savings + jaylen_total_savings
noncomputable def miranda_total_savings := total_savings - combined_savings_rj
noncomputable def miranda_weekly_savings := miranda_total_savings / weeks

noncomputable def miranda_saving_fraction := miranda_weekly_savings / weekly_salary

theorem miranda_saves_half_of_salary:
  miranda_saving_fraction = 1 / 2 := 
by sorry

end miranda_saves_half_of_salary_l298_298242


namespace polygon_perimeter_l298_298107

theorem polygon_perimeter :
  let AB := 2
  let BC := 2
  let CD := 2
  let DE := 2
  let EF := 2
  let FG := 3
  let GH := 3
  let HI := 3
  let IJ := 3
  let JA := 4
  AB + BC + CD + DE + EF + FG + GH + HI + IJ + JA = 26 :=
by {
  sorry
}

end polygon_perimeter_l298_298107


namespace blender_sales_inversely_proportional_l298_298680

theorem blender_sales_inversely_proportional (k : ℝ) (p : ℝ) (c : ℝ) 
  (h1 : p * c = k) (h2 : 10 * 300 = k) : (p * 600 = k) → p = 5 := 
by
  intros
  sorry

end blender_sales_inversely_proportional_l298_298680


namespace max_handshakes_l298_298793

-- Definitions based on the given conditions
def num_people := 30
def handshake_formula (n : ℕ) := n * (n - 1) / 2

-- Formal statement of the problem
theorem max_handshakes : handshake_formula num_people = 435 :=
by
  -- Calculation here would be carried out in the proof, but not included in the statement itself.
  sorry

end max_handshakes_l298_298793


namespace value_of_a_with_two_distinct_roots_l298_298178

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l298_298178


namespace range_of_a_for_three_zeros_l298_298550

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298550


namespace apples_to_grapes_equivalent_l298_298080

-- Definitions based on the problem conditions
def apples := ℝ
def grapes := ℝ

-- Given conditions
def given_condition : Prop := (3 / 4) * 12 = 9

-- Question to prove
def question : Prop := (1 / 2) * 6 = 3

-- The theorem statement combining given conditions to prove the question
theorem apples_to_grapes_equivalent : given_condition → question := 
by
    intros
    sorry

end apples_to_grapes_equivalent_l298_298080


namespace team_members_count_l298_298315

theorem team_members_count (x : ℕ) (h1 : 3 * x + 2 * x = 33 ∨ 4 * x + 2 * x = 33) : x = 6 := by
  sorry

end team_members_count_l298_298315


namespace walter_time_spent_at_seals_l298_298449

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l298_298449


namespace correct_propositions_l298_298028

theorem correct_propositions (a b c d m : ℝ) :
  (ab > 0 → a > b → (1 / a < 1 / b)) ∧
  (a > |b| → a ^ 2 > b ^ 2) ∧
  ¬ (a > b ∧ c < d → a - d > b - c) ∧
  ¬ (a < b ∧ m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end correct_propositions_l298_298028


namespace games_given_away_correct_l298_298218

-- Define initial and remaining games
def initial_games : ℕ := 50
def remaining_games : ℕ := 35

-- Define the number of games given away
def games_given_away : ℕ := initial_games - remaining_games

-- Prove that the number of games given away is 15
theorem games_given_away_correct : games_given_away = 15 := by
  -- This is a placeholder for the actual proof
  sorry

end games_given_away_correct_l298_298218


namespace number_of_numbers_is_11_l298_298629

noncomputable def total_number_of_numbers 
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) : ℝ :=
if h : avg_all = 60 ∧ avg_first_6 = 58 ∧ avg_last_6 = 65 ∧ num_6th = 78 
then 11 else 0 

-- The theorem statement assuming the problem conditions
theorem number_of_numbers_is_11
  {n S : ℝ}
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) 
  (h1 : avg_all = 60) 
  (h2 : avg_first_6 = 58)
  (h3 : avg_last_6 = 65)
  (h4 : num_6th = 78) 
  (h5 : S = 6 * avg_first_6 + 6 * avg_last_6 - num_6th)
  (h6 : S = avg_all * n) : 
  n = 11 := sorry

end number_of_numbers_is_11_l298_298629


namespace impossible_transformation_l298_298729

variable (G : Type) [Group G]

/-- Initial word represented by 2003 'a's followed by 'b' --/
def initial_word := "aaa...ab"

/-- Transformed word represented by 'b' followed by 2003 'a's --/
def transformed_word := "baaa...a"

/-- Hypothetical group relations derived from transformations --/
axiom aba_to_b (a b : G) : (a * b * a = b)
axiom bba_to_a (a b : G) : (b * b * a = a)

/-- Impossible transformation proof --/
theorem impossible_transformation (a b : G) : 
  (initial_word = transformed_word) → False := by
  sorry

end impossible_transformation_l298_298729


namespace cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l298_298288

-- Problem 1
theorem cross_fraction_eq1 (x : ℝ) : (x + 12 / x = -7) → 
  ∃ (x₁ x₂ : ℝ), (x₁ = -3 ∧ x₂ = -4 ∧ x = x₁ ∨ x = x₂) :=
sorry

-- Problem 2
theorem cross_fraction_eq2 (a b : ℝ) 
    (h1 : a * b = -6) 
    (h2 : a + b = -5) : (a ≠ 0 ∧ b ≠ 0) →
    (b / a + a / b + 1 = -31 / 6) :=
sorry

-- Problem 3
theorem cross_fraction_eq3 (k x₁ x₂ : ℝ)
    (hk : k > 2)
    (hx1 : x₁ = 2022 * k - 2022)
    (hx2 : x₂ = k + 1) :
    (x₁ > x₂) →
    (x₁ + 4044) / x₂ = 2022 :=
sorry

end cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l298_298288


namespace smallest_lcm_l298_298375

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l298_298375


namespace hot_dogs_per_pack_l298_298956

-- Define the givens / conditions
def total_hot_dogs : ℕ := 36
def buns_pack_size : ℕ := 9
def same_quantity (h : ℕ) (b : ℕ) := h = b

-- State the theorem to be proven
theorem hot_dogs_per_pack : ∃ h : ℕ, (total_hot_dogs / h = buns_pack_size) ∧ same_quantity (total_hot_dogs / h) (total_hot_dogs / buns_pack_size) := 
sorry

end hot_dogs_per_pack_l298_298956


namespace find_number_l298_298379

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end find_number_l298_298379


namespace football_starting_lineup_count_l298_298800

variable (n_team_members n_offensive_linemen : ℕ)
variable (H_team_members : 12 = n_team_members)
variable (H_offensive_linemen : 5 = n_offensive_linemen)

theorem football_starting_lineup_count :
  n_team_members = 12 → n_offensive_linemen = 5 →
  (n_offensive_linemen * (n_team_members - 1) * (n_team_members - 2) * ((n_team_members - 3) * (n_team_members - 4) / 2)) = 19800 := 
by
  intros
  sorry

end football_starting_lineup_count_l298_298800


namespace inequality_transform_l298_298045

variable {x y : ℝ}

theorem inequality_transform (h : x < y) : - (x / 2) > - (y / 2) :=
sorry

end inequality_transform_l298_298045


namespace find_mass_of_water_vapor_l298_298108

noncomputable def heat_balance_problem : Prop :=
  ∃ (m_s : ℝ), m_s * 536 + m_s * 80 = 
  (50 * 80 + 50 * 20 + 300 * 20 + 100 * 0.5 * 20)
  ∧ m_s = 19.48

theorem find_mass_of_water_vapor : heat_balance_problem := by
  sorry

end find_mass_of_water_vapor_l298_298108


namespace bruce_mango_purchase_l298_298816

theorem bruce_mango_purchase (m : ℕ) 
  (cost_grapes : 8 * 70 = 560)
  (cost_total : 560 + 55 * m = 1110) : 
  m = 10 :=
by
  sorry

end bruce_mango_purchase_l298_298816


namespace colorful_tartan_distribution_l298_298771

-- Define the set of characters and their multiplicities
def letters : finset (char × ℕ) :=
  {('C', 1), ('O', 2), ('L', 2), ('R', 2), ('F', 1), 
   ('U', 1), ('T', 2), ('A', 2), ('N', 1)}

-- Define a function to calculate the number of ways to distribute the blocks
def ways_to_distribute (chars : finset (char × ℕ)) : ℕ :=
  2^4 -- Since there are 4 remaining letters that can be placed in either bag

theorem colorful_tartan_distribution :
  ways_to_distribute letters = 16 :=
by
  -- Proof to be filled in later
  sorry

end colorful_tartan_distribution_l298_298771


namespace numWaysToSeat7WithTwoTogether_is_240_l298_298510

-- Define the number of ways to arrange 7 people around a round table with two specific individuals sitting next to each other
def numWaysToSeat7WithTwoTogether : Nat :=
  let arrange_5_around_table := Nat.factorial 5
  let ways_to_arrange_within_unit := 2
  arrange_5_around_table * ways_to_arrange_within_unit

-- Theorem stating the calculated value
theorem numWaysToSeat7WithTwoTogether_is_240 :
  numWaysToSeat7WithTwoTogether = 240 :=
by
  unfold numWaysToSeat7WithTwoTogether
  simp [Nat.factorial]
  sorry

end numWaysToSeat7WithTwoTogether_is_240_l298_298510


namespace Ceva_theorem_l298_298881

variables {A B C K L M P : Point}
variables {BK KC CL LA AM MB : ℝ}

-- Assume P is inside the triangle ABC and KP, LP, and MP intersect BC, CA, and AB at points K, L, and M respectively
-- We need to prove the ratio product property according to Ceva's theorem
theorem Ceva_theorem 
  (h1: BK / KC = b)
  (h2: CL / LA = c)
  (h3: AM / MB = a)
  (h4: (b * c * a = 1)): 
  (BK / KC) * (CL / LA) * (AM / MB) = 1 :=
sorry

end Ceva_theorem_l298_298881


namespace scientific_notation_conversion_l298_298249

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l298_298249


namespace problem1_problem2_problem3_l298_298146

-- Problem 1
theorem problem1 (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B :=
by sorry

-- Problem 2
theorem problem2 (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 :=
by sorry

end problem1_problem2_problem3_l298_298146


namespace greatest_possible_x_max_possible_x_l298_298896

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298896


namespace div2_implies_div2_of_either_l298_298753

theorem div2_implies_div2_of_either (a b : ℕ) (h : 2 ∣ a * b) : (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end div2_implies_div2_of_either_l298_298753


namespace resting_time_is_thirty_l298_298877

-- Defining the conditions as Lean 4 definitions
def speed := 10 -- miles per hour
def time_first_part := 30 -- minutes
def distance_second_part := 15 -- miles
def distance_third_part := 20 -- miles
def total_time := 270 -- minutes

-- Function to convert hours to minutes
def hours_to_minutes (h : ℕ) : ℕ := h * 60

-- Problem statement in Lean 4: Proving the resting time is 30 minutes
theorem resting_time_is_thirty :
  let distance_first := speed * (time_first_part / 60)
  let time_second_part := (distance_second_part / speed) * 60
  let time_third_part := (distance_third_part / speed) * 60
  let times_sum := time_first_part + time_second_part + time_third_part
  total_time = times_sum + 30 := 
  sorry

end resting_time_is_thirty_l298_298877


namespace combinations_seven_choose_three_l298_298421

theorem combinations_seven_choose_three : nat.choose 7 3 = 35 := by
  sorry

end combinations_seven_choose_three_l298_298421


namespace expected_points_experts_prob_envelope_5_l298_298586

-- Conditions
def num_envelopes := 13
def win_points := 6
def total_games := 100
def envelope_prob := 1 / num_envelopes

-- Part (a): Expected points earned by Experts over 100 games
theorem expected_points_experts 
  (evenly_matched : true) -- Placeholder condition, actual game dynamics assumed
  : (expected (fun (game : ℕ) => game_points_experts game ) (range total_games)) = 465 := 
sorry

-- Part (b): Probability that envelope number 5 will be chosen in the next game
theorem prob_envelope_5 
  : (prob (λ (envelope : ℕ), envelope = 5) (range num_envelopes)) = 12 / 13 :=   -- Simplified calculation
sorry

end expected_points_experts_prob_envelope_5_l298_298586


namespace positive_iff_triangle_l298_298594

def is_triangle_inequality (x y z : ℝ) : Prop :=
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

noncomputable def poly (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem positive_iff_triangle (x y z : ℝ) : 
  poly |x| |y| |z| > 0 ↔ is_triangle_inequality |x| |y| |z| :=
sorry

end positive_iff_triangle_l298_298594


namespace factory_earnings_l298_298573

-- Definition of constants and functions based on the conditions:
def material_A_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def material_B_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def convert_B_to_C (material_B : ℕ) : ℕ := material_B / 2
def earnings (amount : ℕ) (price_per_unit : ℕ) : ℕ := amount * price_per_unit

-- Given conditions for the problem:
def hours_machine_1_and_2 : ℕ := 23
def hours_machine_3 : ℕ := 23
def hours_machine_4 : ℕ := 12
def rate_A_machine_1_and_2 : ℕ := 2
def rate_B_machine_1_and_2 : ℕ := 1
def rate_A_machine_3_and_4 : ℕ := 3
def rate_B_machine_3_and_4 : ℕ := 2
def price_A : ℕ := 50
def price_C : ℕ := 100

-- Calculations based on problem conditions:
noncomputable def total_A : ℕ := 
  2 * material_A_production hours_machine_1_and_2 rate_A_machine_1_and_2 + 
  material_A_production hours_machine_3 rate_A_machine_3_and_4 + 
  material_A_production hours_machine_4 rate_A_machine_3_and_4

noncomputable def total_B : ℕ := 
  2 * material_B_production hours_machine_1_and_2 rate_B_machine_1_and_2 + 
  material_B_production hours_machine_3 rate_B_machine_3_and_4 + 
  material_B_production hours_machine_4 rate_B_machine_3_and_4

noncomputable def total_C : ℕ := convert_B_to_C total_B

noncomputable def total_earnings : ℕ :=
  earnings total_A price_A + earnings total_C price_C

-- The theorem to prove the total earnings:
theorem factory_earnings : total_earnings = 15650 :=
by
  sorry

end factory_earnings_l298_298573


namespace greatest_x_lcm_l298_298926

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298926


namespace factor_polynomial_l298_298329

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l298_298329


namespace gain_percent_l298_298104

-- Let C be the cost price of one chocolate
-- Let S be the selling price of one chocolate
-- Given: 35 * C = 21 * S
-- Prove: The gain percent is 66.67%

theorem gain_percent (C S : ℝ) (h : 35 * C = 21 * S) : (S - C) / C * 100 = 200 / 3 :=
by sorry

end gain_percent_l298_298104


namespace find_a_l298_298737

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l298_298737


namespace find_pairs_l298_298988

theorem find_pairs (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : (mn - 1) ∣ (n^3 - 1)) :
  ∃ k : ℕ, 1 < k ∧ ((m = k ∧ n = k^2) ∨ (m = k^2 ∧ n = k)) :=
sorry

end find_pairs_l298_298988


namespace roll_2_four_times_last_not_2_l298_298047

def probability_of_rolling_2_four_times_last_not_2 : ℚ :=
  (1/6)^4 * (5/6)

theorem roll_2_four_times_last_not_2 :
  probability_of_rolling_2_four_times_last_not_2 = 5 / 7776 := 
by
  sorry

end roll_2_four_times_last_not_2_l298_298047


namespace f_2017_plus_f_2019_eq_zero_l298_298703

-- Definitions of even and odd functions and corresponding conditions
variables {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry
noncomputable def g : R → R := λ x, f (x - 1)

axiom even_f : ∀ x : R, f (-x) = f x
axiom odd_g : ∀ x : R, g (-x) = -g x

theorem f_2017_plus_f_2019_eq_zero : f 2017 + f 2019 = 0 := sorry

end f_2017_plus_f_2019_eq_zero_l298_298703


namespace james_prom_total_cost_l298_298391

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l298_298391


namespace max_dance_counts_possible_l298_298298

noncomputable def max_dance_counts : ℕ := 29

theorem max_dance_counts_possible (boys girls : ℕ) (dance_count : ℕ → ℕ) :
   boys = 29 → girls = 15 → 
   (∀ b, b < boys → dance_count b ≤ girls) → 
   (∀ g, g < girls → ∃ d, d ≤ boys ∧ dance_count d = g) →
   (∃ d, d ≤ max_dance_counts ∧
     (∀ k, k ≤ d → (∃ b, b < boys ∧ dance_count b = k) ∨ (∃ g, g < girls ∧ dance_count g = k))) := 
sorry

end max_dance_counts_possible_l298_298298


namespace greatest_x_lcm_105_l298_298913

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298913


namespace arithmetic_sequence_a20_l298_298027

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l298_298027


namespace base_six_conversion_addition_l298_298785

def base_six_to_base_ten (n : ℕ) : ℕ :=
  4 * 6^0 + 1 * 6^1 + 2 * 6^2

theorem base_six_conversion_addition : base_six_to_base_ten 214 + 15 = 97 :=
by
  sorry

end base_six_conversion_addition_l298_298785


namespace x_is_perfect_square_l298_298882

theorem x_is_perfect_square {x y : ℕ} (hx : x > 0) (hy : y > 0) (h : (x^2 + y^2 - x) % (2 * x * y) = 0) : ∃ z : ℕ, x = z^2 :=
by
  -- The proof will proceed here
  sorry

end x_is_perfect_square_l298_298882


namespace plane_crash_probabilities_eq_l298_298111

noncomputable def crashing_probability_3_engines (p : ℝ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

noncomputable def crashing_probability_5_engines (p : ℝ) : ℝ :=
  10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

theorem plane_crash_probabilities_eq (p : ℝ) :
  crashing_probability_3_engines p = crashing_probability_5_engines p ↔ p = 0 ∨ p = 1/2 ∨ p = 1 :=
by
  sorry

end plane_crash_probabilities_eq_l298_298111


namespace greatest_value_of_x_l298_298932

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298932


namespace find_f_half_l298_298834

variable {α : Type} [DivisionRing α]

theorem find_f_half {f : α → α} (h : ∀ x, f (1 - 2 * x) = 1 / (x^2)) : f (1 / 2) = 16 :=
by
  sorry

end find_f_half_l298_298834


namespace speed_of_stream_l298_298666

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 14) (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 :=
by
  rw [h1, h2]
  norm_num

end speed_of_stream_l298_298666


namespace geometric_sequence_problem_l298_298702

theorem geometric_sequence_problem (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : ∀ n, a (n + 1) = r * a n) 
  (h_cond: a 4 + a 6 = 8) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
  sorry

end geometric_sequence_problem_l298_298702


namespace sum_kml_l298_298982

theorem sum_kml (k m l : ℤ) (b : ℤ → ℤ)
  (h_seq : ∀ n, ∃ k, b n = k * (Int.floor (Real.sqrt (n + m : ℝ))) + l)
  (h_b1 : b 1 = 2) :
  k + m + l = 3 := by
  sorry

end sum_kml_l298_298982


namespace power_expression_l298_298451

theorem power_expression : (1 / ((-5)^4)^2) * (-5)^9 = -5 := sorry

end power_expression_l298_298451


namespace phone_call_probability_within_four_rings_l298_298727

variables (P_A P_B P_C P_D : ℝ)

-- Assuming given probabilities
def probabilities_given : Prop :=
  P_A = 0.1 ∧ P_B = 0.3 ∧ P_C = 0.4 ∧ P_D = 0.1

theorem phone_call_probability_within_four_rings (h : probabilities_given P_A P_B P_C P_D) :
  P_A + P_B + P_C + P_D = 0.9 :=
sorry

end phone_call_probability_within_four_rings_l298_298727


namespace cubic_has_three_zeros_l298_298543

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298543


namespace total_packs_l298_298745

theorem total_packs (cards_per_person : ℕ) (cards_per_pack : ℕ) (people_count : ℕ) (cards_per_person_eq : cards_per_person = 540) (cards_per_pack_eq : cards_per_pack = 20) (people_count_eq : people_count = 4) :
  (cards_per_person / cards_per_pack) * people_count = 108 :=
by
  sorry

end total_packs_l298_298745


namespace gcd_of_1887_and_2091_is_51_l298_298764

variable (a b : Nat)
variable (coefficient1 coefficient2 quotient1 quotient2 quotient3 remainder1 remainder2 : Nat)

def gcd_condition1 : Prop := (b = 1 * a + remainder1)
def gcd_condition2 : Prop := (a = quotient1 * remainder1 + remainder2)
def gcd_condition3 : Prop := (remainder1 = quotient2 * remainder2)

def numbers_1887_and_2091 : Prop := (a = 1887) ∧ (b = 2091)

theorem gcd_of_1887_and_2091_is_51 :
  numbers_1887_and_2091 a b ∧
  gcd_condition1 a b remainder1 ∧ 
  gcd_condition2 a remainder1 remainder2 quotient1 ∧ 
  gcd_condition3 remainder1 remainder2 quotient2 → 
  Nat.gcd 1887 2091 = 51 :=
by
  sorry

end gcd_of_1887_and_2091_is_51_l298_298764


namespace num_perfect_squares_in_range_l298_298205

-- Define the range for the perfect squares
def lower_bound := 75
def upper_bound := 400

-- Define the smallest integer whose square is greater than lower_bound
def lower_int := 9

-- Define the largest integer whose square is less than or equal to upper_bound
def upper_int := 20

-- State the proof problem
theorem num_perfect_squares_in_range : 
  (upper_int - lower_int + 1) = 12 :=
by
  -- Skipping the proof
  sorry

end num_perfect_squares_in_range_l298_298205


namespace polynomial_solution_l298_298136

open Polynomial

noncomputable def p (x : ℝ) : ℝ := -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2

theorem polynomial_solution (x : ℝ) :
  4 * x^5 + 3 * x^3 + 2 * x^2 + (-4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) = 6 * x^3 - 5 * x^2 + 4 * x - 2 :=
by
  -- Verification of the equality
  sorry

end polynomial_solution_l298_298136


namespace no_integer_solutions_l298_298755

theorem no_integer_solutions (x y : ℤ) : x^3 + 3 ≠ 4 * y * (y + 1) :=
sorry

end no_integer_solutions_l298_298755


namespace find_c_l298_298892

-- Define points and the line equation.
def point_A := (1, 3)
def point_B := (5, 11)
def midpoint (A B : ℚ × ℚ) : ℚ × ℚ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The line equation 2x - y = c
def line_eq (x y c : ℚ) : Prop :=
  2 * x - y = c

-- Define the proof problem
theorem find_c : 
  let M := midpoint point_A point_B in
  line_eq M.1 M.2 (-1) :=
by
  sorry

end find_c_l298_298892


namespace pow_four_inequality_l298_298240

theorem pow_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x * y * (x + y)^2 :=
by
  sorry

end pow_four_inequality_l298_298240


namespace smallest_lcm_of_4_digit_integers_l298_298376

open Nat

theorem smallest_lcm_of_4_digit_integers (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h_gcd : gcd k l = 5) :
  lcm k l = 203010 := sorry

end smallest_lcm_of_4_digit_integers_l298_298376


namespace election_win_by_votes_l298_298630

/-- Two candidates in an election, the winner received 56% of votes and won the election
by receiving 1344 votes. We aim to prove that the winner won by 288 votes. -/
theorem election_win_by_votes
  (V : ℝ)  -- total number of votes
  (w : ℝ)  -- percentage of votes received by the winner
  (w_votes : ℝ)  -- votes received by the winner
  (l_votes : ℝ)  -- votes received by the loser
  (w_percentage : w = 0.56)
  (w_votes_given : w_votes = 1344)
  (total_votes : V = 1344 / 0.56)
  (l_votes_calc : l_votes = (V * 0.44)) :
  1344 - l_votes = 288 :=
by
  -- Proof goes here
  sorry

end election_win_by_votes_l298_298630


namespace range_of_a_for_three_zeros_l298_298522

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298522


namespace equation_has_roots_l298_298152

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l298_298152


namespace exceeded_goal_by_600_l298_298244

noncomputable def ken_collection : ℕ := 600
noncomputable def mary_collection : ℕ := 5 * ken_collection
noncomputable def scott_collection : ℕ := mary_collection / 3
noncomputable def goal : ℕ := 4000
noncomputable def total_raised : ℕ := mary_collection + scott_collection + ken_collection

theorem exceeded_goal_by_600 : total_raised - goal = 600 := by
  have h1 : ken_collection = 600 := rfl
  have h2 : mary_collection = 5 * ken_collection := rfl
  have h3 : scott_collection = mary_collection / 3 := rfl
  have h4 : goal = 4000 := rfl
  have h5 : total_raised = mary_collection + scott_collection + ken_collection := rfl
  have hken : ken_collection = 600 := rfl
  have hmary : mary_collection = 5 * 600 := by rw [hken]; rfl
  have hscott : scott_collection = 3000 / 3 := by rw [hmary]; rfl
  have htotal : total_raised = 3000 + 1000 + 600 := by rw [hmary, hscott, hken]; rfl
  have hexceeded : total_raised - goal = 4600 - 4000 := by rw [htotal, h4]; rfl
  exact hexceeded

end exceeded_goal_by_600_l298_298244


namespace milo_running_distance_l298_298602

theorem milo_running_distance : 
  ∀ (cory_speed milo_skate_speed milo_run_speed time miles_run : ℕ),
  cory_speed = 12 →
  milo_skate_speed = cory_speed / 2 →
  milo_run_speed = milo_skate_speed / 2 →
  time = 2 →
  miles_run = milo_run_speed * time →
  miles_run = 6 :=
by 
  intros cory_speed milo_skate_speed milo_run_speed time miles_run hcory hmilo_skate hmilo_run htime hrun 
  -- Proof steps would go here
  sorry

end milo_running_distance_l298_298602


namespace even_function_implies_a_zero_l298_298106

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, (x^2 - |x + a|) = (x^2 - |x - a|)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l298_298106


namespace tangent_parallel_coordinates_l298_298777

theorem tangent_parallel_coordinates :
  (∃ (x1 y1 x2 y2 : ℝ), 
    (y1 = x1^3 - 2) ∧ (y2 = x2^3 - 2) ∧ 
    ((3 * x1^2 = 3) ∧ (3 * x2^2 = 3)) ∧ 
    ((x1 = 1 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -3))) :=
sorry

end tangent_parallel_coordinates_l298_298777


namespace sphere_volume_proof_l298_298499

noncomputable def sphereVolume (d : ℝ) (S : ℝ) : ℝ :=
  let r := Real.sqrt (S / Real.pi)
  let R := Real.sqrt (r^2 + d^2)
  (4 / 3) * Real.pi * R^3

theorem sphere_volume_proof : sphereVolume 1 (2 * Real.pi) = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end sphere_volume_proof_l298_298499


namespace find_A_l298_298185

def divisible_by(a b : ℕ) := b % a = 0

def valid_digit_A (A : ℕ) : Prop := (A = 0 ∨ A = 2 ∨ A = 4 ∨ A = 6 ∨ A = 8) ∧ divisible_by A 75

theorem find_A : ∃! A : ℕ, valid_digit_A A :=
by {
  sorry
}

end find_A_l298_298185


namespace shifting_parabola_l298_298578

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shifted_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem shifting_parabola : ∀ x : ℝ, shifted_function x = original_function (x + 2) - 1 := 
by 
  sorry

end shifting_parabola_l298_298578


namespace range_of_a_for_three_zeros_l298_298537

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298537


namespace geom_seq_increasing_sufficient_necessary_l298_298865

theorem geom_seq_increasing_sufficient_necessary (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n : ℕ, a n = a 0 * r ^ n) 
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) : 
  (a 0 < a 1 ∧ a 1 < a 2) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end geom_seq_increasing_sufficient_necessary_l298_298865


namespace bill_and_harry_nuts_l298_298130

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l298_298130


namespace expand_product_l298_298825

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := 
by 
  sorry

end expand_product_l298_298825


namespace smallest_value_a_plus_b_l298_298700

theorem smallest_value_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 3^7 * 5^3 = a^b) : a + b = 3376 :=
sorry

end smallest_value_a_plus_b_l298_298700


namespace intersection_eq_l298_298742

def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x) / Real.log 3}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_eq_l298_298742


namespace probability_hundreds_digit_triple_ones_digit_l298_298724

def is_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem probability_hundreds_digit_triple_ones_digit :
  let favorable_outcomes : ℤ :=
    {n : ℕ | is_3_digit_number n ∧ hundreds_digit n = 3 * ones_digit n} .to_finset.card in
  let total_outcomes : ℤ :=
    {n : ℕ | is_3_digit_number n} .to_finset.card in
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 30 :=
by {
  sorry
}

end probability_hundreds_digit_triple_ones_digit_l298_298724


namespace relationship_abcd_l298_298221

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end relationship_abcd_l298_298221


namespace simplify_expression_l298_298954

theorem simplify_expression (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 :=
by
  sorry

end simplify_expression_l298_298954


namespace new_prism_volume_l298_298669

-- Define the original volume
def original_volume : ℝ := 12

-- Define the dimensions modification factors
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 3

-- Define the volume of the new prism
def new_volume := (length_factor * width_factor * height_factor) * original_volume

-- State the theorem to prove
theorem new_prism_volume : new_volume = 144 := 
by sorry

end new_prism_volume_l298_298669


namespace total_cost_of_tennis_balls_l298_298874

theorem total_cost_of_tennis_balls
  (packs : ℕ) (balls_per_pack : ℕ) (cost_per_ball : ℕ)
  (h1 : packs = 4) (h2 : balls_per_pack = 3) (h3 : cost_per_ball = 2) : 
  packs * balls_per_pack * cost_per_ball = 24 := by
  sorry

end total_cost_of_tennis_balls_l298_298874


namespace equation_has_at_least_two_distinct_roots_l298_298162

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l298_298162


namespace number_of_roses_l298_298267

def total_flowers : ℕ := 10
def carnations : ℕ := 5
def roses : ℕ := total_flowers - carnations

theorem number_of_roses : roses = 5 := by
  sorry

end number_of_roses_l298_298267


namespace cost_of_article_l298_298715

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end cost_of_article_l298_298715


namespace find_n_l298_298465

theorem find_n (n : ℕ) : (16 : ℝ)^(1/4) = 2^n ↔ n = 1 := by
  sorry

end find_n_l298_298465


namespace chiquita_height_l298_298236

theorem chiquita_height (C : ℝ) :
  (C + (C + 2) = 12) → (C = 5) :=
by
  intro h
  sorry

end chiquita_height_l298_298236


namespace gcd_digit_bound_l298_298053

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l298_298053


namespace simplify_complex_l298_298616

open Complex

theorem simplify_complex : (5 : ℂ) / (I - 2) = -2 - I := by
  sorry

end simplify_complex_l298_298616


namespace xiao_ming_total_score_l298_298385

theorem xiao_ming_total_score :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5 ∧ 
  a_1 + a_2 = 10 ∧ 
  a_4 + a_5 = 18 ∧ 
  a_1 + a_2 + a_3 + a_4 + a_5 = 35 :=
by
  sorry

end xiao_ming_total_score_l298_298385


namespace evaluate_4_over_04_eq_400_l298_298005

noncomputable def evaluate_fraction : Float :=
  (0.4)^4 / (0.04)^3

theorem evaluate_4_over_04_eq_400 : evaluate_fraction = 400 :=
by
  sorry

end evaluate_4_over_04_eq_400_l298_298005


namespace female_athletes_drawn_is_7_l298_298808

-- Given conditions as definitions
def male_athletes := 64
def female_athletes := 56
def drawn_male_athletes := 8

-- The function that represents the equation in stratified sampling
def stratified_sampling_eq (x : Nat) : Prop :=
  (drawn_male_athletes : ℚ) / (male_athletes) = (x : ℚ) / (female_athletes)

-- The theorem which states that the solution to the problem is x = 7
theorem female_athletes_drawn_is_7 : ∃ x : Nat, stratified_sampling_eq x ∧ x = 7 :=
by
  sorry

end female_athletes_drawn_is_7_l298_298808


namespace find_X_eq_A_l298_298014

variable {α : Type*}
variable (A X : Set α)

theorem find_X_eq_A (h : X ∩ A = X ∪ A) : X = A := by
  sorry

end find_X_eq_A_l298_298014


namespace value_of_a_with_two_distinct_roots_l298_298177

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l298_298177


namespace last_digit_2019_digit_number_l298_298662

theorem last_digit_2019_digit_number :
  ∃ n : ℕ → ℕ,  
    (∀ k, 0 ≤ k → k < 2018 → (n k * 10 + n (k + 1)) % 13 = 0) ∧ 
    n 0 = 6 ∧ 
    n 2018 = 2 :=
sorry

end last_digit_2019_digit_number_l298_298662


namespace average_score_of_class_l298_298812

theorem average_score_of_class (n : ℕ) (k : ℕ) (jimin_score : ℕ) (jungkook_score : ℕ) (avg_others : ℕ) 
  (total_students : n = 40) (excluding_students : k = 38) 
  (avg_excluding_others : avg_others = 79) 
  (jimin : jimin_score = 98) 
  (jungkook : jungkook_score = 100) : 
  (98 + 100 + (38 * 79)) / 40 = 80 :=
sorry

end average_score_of_class_l298_298812


namespace greatest_x_lcm_105_l298_298921

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298921


namespace number_of_candidates_is_9_l298_298116

-- Defining the problem
def num_ways_to_select_president_and_vp (n : ℕ) : ℕ :=
  n * (n - 1)

-- Main theorem statement
theorem number_of_candidates_is_9 (n : ℕ) (h : num_ways_to_select_president_and_vp n = 72) : n = 9 :=
by
  sorry

end number_of_candidates_is_9_l298_298116


namespace geometric_sequence_general_term_l298_298730

theorem geometric_sequence_general_term (n : ℕ) (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) 
  (h1 : a1 = 4) (h2 : q = 3) (h3 : ∀ n, a n = a1 * (q ^ (n - 1))) :
  a n = 4 * 3^(n - 1) := by
  sorry

end geometric_sequence_general_term_l298_298730


namespace average_candies_l298_298568

theorem average_candies {a b c d e f : ℕ} (h₁ : a = 16) (h₂ : b = 22) (h₃ : c = 30) (h₄ : d = 26) (h₅ : e = 18) (h₆ : f = 20) :
  (a + b + c + d + e + f) / 6 = 22 := by
  sorry

end average_candies_l298_298568


namespace range_of_a_for_three_zeros_l298_298549

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298549


namespace selection_ways_l298_298419

-- Define the problem parameters
def male_students : ℕ := 4
def female_students : ℕ := 3
def total_selected : ℕ := 3

-- Define the binomial coefficient function for combinatorial calculations
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define conditions
def both_genders_must_be_represented : Prop :=
  total_selected = 3 ∧ male_students >= 1 ∧ female_students >= 1

-- Problem statement: proof that the total ways to select 3 students is 30
theorem selection_ways : both_genders_must_be_represented → 
  (binomial male_students 2 * binomial female_students 1 +
   binomial male_students 1 * binomial female_students 2) = 30 :=
by
  sorry

end selection_ways_l298_298419


namespace larger_number_225_l298_298709

theorem larger_number_225 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a - b = 120) 
  (h4 : Nat.lcm a b = 105 * Nat.gcd a b) : 
  max a b = 225 :=
by
  sorry

end larger_number_225_l298_298709


namespace value_of_x_l298_298265

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l298_298265


namespace weight_ratio_l298_298848

variable (J : ℕ) (T : ℕ) (L : ℕ) (S : ℕ)

theorem weight_ratio (h_jake_weight : J = 152) (h_total_weight : J + S = 212) (h_weight_loss : L = 32) :
    (J - L) / (T - J) = 2 :=
by
  sorry

end weight_ratio_l298_298848


namespace equation_of_line_l_l298_298352

-- Define the conditions for the parabola and the line
def parabola_vertex : Prop := 
  ∃ C : ℝ × ℝ, C = (0, 0)

def parabola_symmetry_axis : Prop := 
  ∃ l : ℝ → ℝ, ∀ x, l x = -1

def midpoint_of_AB (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

def parabola_equation (A B : ℝ × ℝ) : Prop :=
  A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1

-- State the theorem to be proven
theorem equation_of_line_l (A B : ℝ × ℝ) :
  parabola_vertex ∧ parabola_symmetry_axis ∧ midpoint_of_AB A B ∧ parabola_equation A B →
  ∃ l : ℝ → ℝ, ∀ x, l x = 2 * x - 3 :=
by sorry

end equation_of_line_l_l298_298352


namespace june_1_friday_l298_298852

open Nat

-- Define the days of the week as data type
inductive DayOfWeek : Type
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open DayOfWeek

-- Define that June has 30 days
def june_days := 30

-- Hypotheses that June has exactly three Mondays and exactly three Thursdays
def three_mondays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Monday → 3 ≤ n / 7) -- there are exactly three Mondays
  
def three_thursdays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Thursday → 3 ≤ n / 7) -- there are exactly three Thursdays

-- Theorem to prove June 1 falls on a Friday given those conditions
theorem june_1_friday : ∀ (d : DayOfWeek), 
  three_mondays d → three_thursdays d → (d = Friday) :=
by
  sorry

end june_1_friday_l298_298852


namespace xyz_sum_eq_7x_plus_5_l298_298400

variable (x y z : ℝ)

theorem xyz_sum_eq_7x_plus_5 (h1: y = 3 * x) (h2: z = y + 5) : x + y + z = 7 * x + 5 :=
by
  sorry

end xyz_sum_eq_7x_plus_5_l298_298400


namespace soccer_team_games_count_l298_298475

variable (total_games won_games : ℕ)
variable (h1 : won_games = 70)
variable (h2 : won_games = total_games / 2)

theorem soccer_team_games_count : total_games = 140 :=
by
  -- Proof goes here
  sorry

end soccer_team_games_count_l298_298475


namespace range_of_a_l298_298527

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298527


namespace percent_increase_l298_298772

theorem percent_increase (P x : ℝ) (h1 : P + x/100 * P - 0.2 * (P + x/100 * P) = P) : x = 25 :=
by
  sorry

end percent_increase_l298_298772


namespace correct_value_calculation_l298_298041

theorem correct_value_calculation (x : ℤ) (h : 2 * (x + 6) = 28) : 6 * x = 48 :=
by
  -- Proof steps would be here
  sorry

end correct_value_calculation_l298_298041


namespace gcd_digits_bounded_by_lcm_l298_298058

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l298_298058


namespace intersection_eq_l298_298998

def set1 : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def set2 : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_eq : (set1 ∩ set2) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_eq_l298_298998


namespace expected_points_experts_over_100_games_probability_of_envelope_five_selected_l298_298584

-- Game conditions and probabilities
def game_conditions (experts_points audience_points : ℕ) : Prop :=
  experts_points = 6 ∨ audience_points = 6

noncomputable def equal_teams := (1 : ℝ) / 2

-- Expected score of Experts over 100 games
noncomputable def expected_points_experts (games : ℕ) := 465

-- Probability that envelope number 5 is chosen in the next game
noncomputable def probability_envelope_five := (12 : ℝ) / 13

theorem expected_points_experts_over_100_games : 
  expected_points_experts 100 = 465 := 
sorry

theorem probability_of_envelope_five_selected : 
  probability_envelope_five = 0.715 := 
sorry

end expected_points_experts_over_100_games_probability_of_envelope_five_selected_l298_298584


namespace range_of_m_l298_298191

-- Define the propositions
def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Problem statement to derive m's range
theorem range_of_m (m : ℝ) (h1: ¬ (p m ∧ q m)) (h2: p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l298_298191


namespace distinct_cube_arrangements_count_l298_298686

def is_valid_face_sum (face : Finset ℕ) : Prop :=
  face.sum id = 34

def is_valid_opposite_sum (v1 v2 : ℕ) : Prop :=
  v1 + v2 = 16

def is_unique_up_to_rotation (cubes : List (Finset ℕ)) : Prop := sorry -- Define rotational uniqueness check

noncomputable def count_valid_arrangements : ℕ := sorry -- Define counting logic

theorem distinct_cube_arrangements_count : count_valid_arrangements = 3 :=
  sorry

end distinct_cube_arrangements_count_l298_298686


namespace product_of_consecutive_even_numbers_l298_298091

theorem product_of_consecutive_even_numbers
  (a b c : ℤ)
  (h : a + b + c = 18 ∧ 2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ a < b ∧ b < c ∧ b - a = 2 ∧ c - b = 2) :
  a * b * c = 192 :=
sorry

end product_of_consecutive_even_numbers_l298_298091


namespace smallest_integer_with_20_divisors_l298_298646

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, (n > 0 ∧ (∃ (d : ℕ → Prop), (∀ m, d m ↔ m ∣ n) ∧ (card { m : ℕ | d m } = 20)) ∧ (∀ k : ℕ, k > 0 ∧ (∃ (d' : ℕ → Prop), (∀ m, d' m ↔ m ∣ k) ∧ (card { m : ℕ | d' m } = 20)) → k ≥ n)) ∧ n = 240 :=
by { sorry }

end smallest_integer_with_20_divisors_l298_298646


namespace Albert_has_more_rocks_than_Jose_l298_298066

noncomputable def Joshua_rocks : ℕ := 80
noncomputable def Jose_rocks : ℕ := Joshua_rocks - 14
noncomputable def Albert_rocks : ℕ := Joshua_rocks + 6

theorem Albert_has_more_rocks_than_Jose :
  Albert_rocks - Jose_rocks = 20 := by
  sorry

end Albert_has_more_rocks_than_Jose_l298_298066


namespace piecewise_function_continuity_l298_298735

theorem piecewise_function_continuity :
  (∃ a c : ℝ, (2 * a * 2 + 4 = 2^2 - 2) ∧ (4 - 2 = 3 * (-2) - c) ∧ a + c = -17 / 2) :=
by
  sorry

end piecewise_function_continuity_l298_298735


namespace range_of_a_for_three_zeros_l298_298521

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298521


namespace tonya_large_lemonade_sales_l298_298946

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l298_298946


namespace monroe_legs_total_l298_298410

def num_spiders : ℕ := 8
def num_ants : ℕ := 12
def legs_per_spider : ℕ := 8
def legs_per_ant : ℕ := 6

theorem monroe_legs_total :
  num_spiders * legs_per_spider + num_ants * legs_per_ant = 136 :=
by
  sorry

end monroe_legs_total_l298_298410


namespace zeros_of_quadratic_l298_298627

def f (x : ℝ) := x^2 - 2 * x - 3

theorem zeros_of_quadratic : ∀ x, f x = 0 ↔ (x = 3 ∨ x = -1) := 
by 
  sorry

end zeros_of_quadratic_l298_298627


namespace new_babysitter_rate_l298_298601

theorem new_babysitter_rate (x : ℝ) :
  (6 * 16) - 18 = 6 * x + 3 * 2 → x = 12 :=
by
  intros h
  sorry

end new_babysitter_rate_l298_298601


namespace find_x_l298_298613

theorem find_x (x : ℝ) (A1 A2 : ℝ) (P1 P2 : ℝ)
    (hA1 : A1 = x^2 + 4*x + 4)
    (hA2 : A2 = 4*x^2 - 12*x + 9)
    (hP : P1 + P2 = 32)
    (hP1 : P1 = 4 * (x + 2))
    (hP2 : P2 = 4 * (2*x - 3)) :
    x = 3 :=
by
  sorry

end find_x_l298_298613


namespace sum_of_roots_l298_298072

open Real

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 4 * x1^2 - k * x1 = c) (h2 : 4 * x2^2 - k * x2 = c) (h3 : x1 ≠ x2) :
  x1 + x2 = k / 4 :=
by
  sorry

end sum_of_roots_l298_298072


namespace Tom_has_38_photos_l298_298437

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l298_298437


namespace gcd_digit_bound_l298_298055

theorem gcd_digit_bound (a b : ℕ) (h₁ : 10^6 ≤ a) (h₂ : a < 10^7) (h₃ : 10^6 ≤ b) (h₄ : b < 10^7) 
  (h₅ : 10^{10} ≤ lcm a b) (h₆ : lcm a b < 10^{11}) : 
  gcd a b < 10^4 :=
sorry

end gcd_digit_bound_l298_298055


namespace max_x_lcm_15_21_105_l298_298911

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298911


namespace toll_for_18_wheel_truck_l298_298659

-- Definitions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def rear_axle_wheels_per_axle : ℕ := 4
def toll_formula (x : ℕ) : ℝ := 0.50 + 0.50 * (x - 2)

-- Theorem statement
theorem toll_for_18_wheel_truck : 
  ∃ t : ℝ, t = 2.00 ∧
  ∃ x : ℕ, x = (1 + ((total_wheels - front_axle_wheels) / rear_axle_wheels_per_axle)) ∧
  t = toll_formula x := 
by
  -- Proof to be provided
  sorry

end toll_for_18_wheel_truck_l298_298659


namespace partitions_equal_l298_298224

namespace MathProof

-- Define the set of natural numbers
def nat := ℕ

-- Define the partition functions (placeholders)
def num_distinct_partitions (n : nat) : nat := sorry
def num_odd_partitions (n : nat) : nat := sorry

-- Statement of the theorem
theorem partitions_equal (n : nat) : 
  num_distinct_partitions n = num_odd_partitions n :=
sorry

end MathProof

end partitions_equal_l298_298224


namespace min_expression_l298_298026

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l298_298026


namespace smallest_positive_integer_with_20_divisors_is_432_l298_298640

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end smallest_positive_integer_with_20_divisors_is_432_l298_298640


namespace three_zeros_implies_a_lt_neg3_l298_298564

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298564


namespace number_of_students_l298_298976

theorem number_of_students (total_students : ℕ) :
  (total_students = 19 * 6 + 4) ∧ 
  (∃ (x y : ℕ), x + y = 22 ∧ x > 7 ∧ total_students = x * 6 + y * 5) →
  total_students = 118 :=
by
  sorry

end number_of_students_l298_298976


namespace a_5_eq_31_l298_298064

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

theorem a_5_eq_31 (a : ℕ → ℕ) (h : seq a) : a 5 = 31 :=
by
  sorry
 
end a_5_eq_31_l298_298064


namespace decoded_word_is_correct_l298_298950

-- Assume that we have a way to represent figures and encoded words
structure Figure1
structure Figure2

-- Assume the existence of a key that maps arrow patterns to letters
def decode (f1 : Figure1) (f2 : Figure2) : String := sorry

theorem decoded_word_is_correct (f1 : Figure1) (f2 : Figure2) :
  decode f1 f2 = "КОМПЬЮТЕР" :=
by
  sorry

end decoded_word_is_correct_l298_298950


namespace primes_p_q_divisibility_l298_298847

theorem primes_p_q_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_eq : q = p + 2) :
  (p + q) ∣ (p ^ q + q ^ p) := 
sorry

end primes_p_q_divisibility_l298_298847


namespace smallest_sum_of_integers_on_square_vertices_l298_298970

theorem smallest_sum_of_integers_on_square_vertices :
  ∃ (a b c d : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  (a % b = 0 ∨ b % a = 0) ∧ (c % a = 0 ∨ a % c = 0) ∧ 
  (d % b = 0 ∨ b % d = 0) ∧ (d % c = 0 ∨ c % d = 0) ∧ 
  a % c ≠ 0 ∧ a % d ≠ 0 ∧ b % c ≠ 0 ∧ b % d ≠ 0 ∧ 
  (a + b + c + d = 35) := sorry

end smallest_sum_of_integers_on_square_vertices_l298_298970


namespace max_ski_trips_l298_298621

/--
The ski lift carries skiers from the bottom of the mountain to the top, taking 15 minutes each way, 
and it takes 5 minutes to ski back down the mountain. 
Given that the total available time is 2 hours, prove that the maximum number of trips 
down the mountain in that time is 6.
-/
theorem max_ski_trips (ride_up_time : ℕ) (ski_down_time : ℕ) (total_time : ℕ) :
  ride_up_time = 15 →
  ski_down_time = 5 →
  total_time = 120 →
  (total_time / (ride_up_time + ski_down_time) = 6) :=
by
  intros h1 h2 h3
  sorry

end max_ski_trips_l298_298621


namespace a_10_value_l298_298086

-- Definitions for the initial conditions and recurrence relation.
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧
  ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * (Real.sqrt (4 ^ n - a n ^ 2))

-- Statement that proves a_10 = 24576 / 25 given the conditions.
theorem a_10_value (a : ℕ → ℝ) (h : seq a) : a 10 = 24576 / 25 :=
by
  sorry

end a_10_value_l298_298086


namespace pizza_cost_l298_298336

theorem pizza_cost
  (initial_money_frank : ℕ)
  (initial_money_bill : ℕ)
  (final_money_bill : ℕ)
  (pizza_cost : ℕ)
  (number_of_pizzas : ℕ)
  (money_given_to_bill : ℕ) :
  initial_money_frank = 42 ∧
  initial_money_bill = 30 ∧
  final_money_bill = 39 ∧
  number_of_pizzas = 3 ∧
  money_given_to_bill = final_money_bill - initial_money_bill →
  3 * pizza_cost + money_given_to_bill = initial_money_frank →
  pizza_cost = 11 :=
by
  sorry

end pizza_cost_l298_298336


namespace binom_2023_2_eq_l298_298134

theorem binom_2023_2_eq : Nat.choose 2023 2 = 2045323 := by
  sorry

end binom_2023_2_eq_l298_298134


namespace number_of_zeros_l298_298006

noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

theorem number_of_zeros (n : ℕ) : (1 < x ∧ x < Real.exp Real.pi) → (∃! x : ℝ, g x = 0 ∧ 1 < x ∧ x < Real.exp Real.pi) → n = 1 :=
sorry

end number_of_zeros_l298_298006


namespace quadratic_inequality_solution_l298_298137

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + m > 0) ↔ m > 1 :=
by
  sorry

end quadratic_inequality_solution_l298_298137


namespace monthly_expenses_last_month_l298_298884

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l298_298884


namespace henry_age_l298_298260

theorem henry_age (H J : ℕ) 
  (sum_ages : H + J = 40) 
  (age_relation : H - 11 = 2 * (J - 11)) : 
  H = 23 := 
sorry

end henry_age_l298_298260


namespace range_of_a_if_f_has_three_zeros_l298_298557

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298557


namespace three_zeros_implies_a_lt_neg3_l298_298563

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l298_298563


namespace pentagon_number_arrangement_l298_298462

def no_common_divisor_other_than_one (a b : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → (d ∣ a ∧ d ∣ b) → false

def has_common_divisor_greater_than_one (a b : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

theorem pentagon_number_arrangement :
  ∃ (A B C D E : ℕ),
    no_common_divisor_other_than_one A B ∧
    no_common_divisor_other_than_one B C ∧
    no_common_divisor_other_than_one C D ∧
    no_common_divisor_other_than_one D E ∧
    no_common_divisor_other_than_one E A ∧
    has_common_divisor_greater_than_one A C ∧
    has_common_divisor_greater_than_one A D ∧
    has_common_divisor_greater_than_one B D ∧
    has_common_divisor_greater_than_one B E ∧
    has_common_divisor_greater_than_one C E :=
sorry

end pentagon_number_arrangement_l298_298462


namespace cuboid_diagonals_and_edges_l298_298418

theorem cuboid_diagonals_and_edges (a b c : ℝ) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 :=
by
  sorry

end cuboid_diagonals_and_edges_l298_298418


namespace model_to_statue_scale_l298_298769

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end model_to_statue_scale_l298_298769


namespace solve_special_sequence_l298_298060

noncomputable def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ a 2 = 1015 ∧ ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem solve_special_sequence :
  ∃ a : ℕ → ℕ, special_sequence a ∧ a 1000 = 1676 :=
by
  sorry

end solve_special_sequence_l298_298060


namespace product_is_zero_l298_298656

variables {a b c d : ℤ}

def system_of_equations (a b c d : ℤ) :=
  2 * a + 3 * b + 5 * c + 7 * d = 34 ∧
  3 * (d + c) = b ∧
  3 * b + c = a ∧
  c - 1 = d

theorem product_is_zero (h : system_of_equations a b c d) : 
  a * b * c * d = 0 :=
sorry

end product_is_zero_l298_298656


namespace chord_midpoint_line_l298_298353

open Real 

theorem chord_midpoint_line (x y : ℝ) (P : ℝ × ℝ) 
  (hP : P = (1, 1)) (hcircle : ∀ (x y : ℝ), x^2 + y^2 = 10) :
  x + y - 2 = 0 :=
by
  sorry

end chord_midpoint_line_l298_298353


namespace power_function_m_l298_298721

theorem power_function_m (m : ℝ) 
  (h_even : ∀ x : ℝ, x^m = (-x)^m) 
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → x^m > y^m) : m = -2 :=
sorry

end power_function_m_l298_298721


namespace value_of_expression_l298_298372

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l298_298372


namespace no_y_satisfies_both_inequalities_l298_298010

variable (y : ℝ)

theorem no_y_satisfies_both_inequalities :
  ¬ (3 * y^2 - 4 * y - 5 < (y + 1)^2 ∧ (y + 1)^2 < 4 * y^2 - y - 1) :=
by
  sorry

end no_y_satisfies_both_inequalities_l298_298010


namespace greatest_possible_x_max_possible_x_l298_298899

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298899


namespace rectangle_symmetry_l298_298114

-- Define basic geometric terms and the notion of symmetry
structure Rectangle where
  length : ℝ
  width : ℝ
  (length_pos : 0 < length)
  (width_pos : 0 < width)

def is_axes_of_symmetry (r : Rectangle) (n : ℕ) : Prop :=
  -- A hypothetical function that determines whether a rectangle r has n axes of symmetry
  sorry

theorem rectangle_symmetry (r : Rectangle) : is_axes_of_symmetry r 2 := 
  -- This theorem states that a rectangle has exactly 2 axes of symmetry
  sorry

end rectangle_symmetry_l298_298114


namespace common_chord_length_of_two_circles_l298_298836

-- Define the equations of the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 4 = 0
def circle2 (x y : ℝ) : Prop := (x + 3 / 2)^2 + (y - 3 / 2)^2 = 11 / 2

-- The theorem stating the length of the common chord
theorem common_chord_length_of_two_circles :
  ∃ l : ℝ, (∀ (x y : ℝ), circle1 x y ↔ circle2 x y) → l = 2 :=
by simp [circle1, circle2]; sorry

end common_chord_length_of_two_circles_l298_298836


namespace ice_cream_cost_proof_l298_298366

-- Assume the cost of the ice cream and toppings
def cost_of_ice_cream : ℝ := 2 -- Ice cream cost in dollars
def cost_per_topping : ℝ := 0.5 -- Cost per topping in dollars
def total_cost_of_sundae_with_10_toppings : ℝ := 7 -- Total cost in dollars

theorem ice_cream_cost_proof :
  (∀ (cost_of_ice_cream : ℝ), 
    total_cost_of_sundae_with_10_toppings = cost_of_ice_cream + 10 * cost_per_topping) →
  cost_of_ice_cream = 2 :=
by
  sorry

end ice_cream_cost_proof_l298_298366


namespace find_a_for_quadratic_l298_298172

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l298_298172


namespace smallest_sector_angle_l298_298217

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_sector_angle 
  (a : ℕ) (d : ℕ) (n : ℕ := 15) (sum_angles : ℕ := 360) 
  (angles_arith_seq : arithmetic_sequence_sum a d n = sum_angles) 
  (h_poses : ∀ m : ℕ, arithmetic_sequence_sum a d m = sum_angles -> m = n) 
  : a = 3 := 
by 
  sorry

end smallest_sector_angle_l298_298217


namespace range_of_a_l298_298850

open Real

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * exp x1 - a = 0) ∧ (x2 * exp x2 - a = 0)) ↔ -1 / exp 1 < a ∧ a < 0 :=
sorry

end range_of_a_l298_298850


namespace quadrilateral_angles_l298_298855

theorem quadrilateral_angles 
  (A B C D : Type) 
  (a d b c : Float)
  (hAD : a = d ∧ d = c) 
  (hBDC_twice_BDA : ∃ x : Float, b = 2 * x) 
  (hBDA_CAD_ratio : ∃ x : Float, d = 2/3 * x) :
  (∃ α β γ δ : Float, 
    α = 75 ∧ 
    β = 135 ∧ 
    γ = 60 ∧ 
    δ = 90) := 
sorry

end quadrilateral_angles_l298_298855


namespace factorize_difference_of_squares_l298_298143

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l298_298143


namespace range_of_a_l298_298528

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l298_298528


namespace bill_harry_combined_l298_298128

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l298_298128


namespace cabinets_ratio_proof_l298_298731

-- Definitions for the conditions
def initial_cabinets : ℕ := 3
def total_cabinets : ℕ := 26
def additional_cabinets : ℕ := 5
def number_of_counters : ℕ := 3

-- Definition for the unknown cabinets installed per counter
def cabinets_per_counter : ℕ := (total_cabinets - additional_cabinets - initial_cabinets) / number_of_counters

-- The ratio to be proven
theorem cabinets_ratio_proof : (cabinets_per_counter : ℚ) / initial_cabinets = 2 / 1 :=
by
  -- Proof goes here
  sorry

end cabinets_ratio_proof_l298_298731


namespace factor_polynomial_l298_298330

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l298_298330


namespace quadratic_trinomial_prime_l298_298190

theorem quadratic_trinomial_prime (p x : ℤ) (hp : p > 1) (hx : 0 ≤ x ∧ x < p)
  (h_prime : Prime (x^2 - x + p)) : x = 0 ∨ x = 1 :=
by
  sorry

end quadratic_trinomial_prime_l298_298190


namespace maximum_cards_without_equal_pair_sums_l298_298210

def max_cards_no_equal_sum_pairs : ℕ :=
  let card_points := {x : ℕ | 1 ≤ x ∧ x ≤ 13}
  6

theorem maximum_cards_without_equal_pair_sums (deck : Finset ℕ) (h_deck : deck = {x : ℕ | 1 ≤ x ∧ x ≤ 13}) :
  ∃ S ⊆ deck, S.card = 6 ∧ ∀ {a b c d : ℕ}, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a + b = c + d → a = c ∧ b = d ∨ a = d ∧ b = c := 
sorry

end maximum_cards_without_equal_pair_sums_l298_298210


namespace smallest_n_for_Tn_gt_2006_over_2016_l298_298089

-- Definitions from the given problem
def Sn (n : ℕ) : ℚ := n^2 / (n + 1)
def an (n : ℕ) : ℚ := if n = 1 then 1 / 2 else Sn n - Sn (n - 1)
def bn (n : ℕ) : ℚ := an n / (n^2 + n - 1)

-- Definition of Tn sum
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k => bn (k + 1))

-- The main statement
theorem smallest_n_for_Tn_gt_2006_over_2016 : ∃ n : ℕ, Tn n > 2006 / 2016 := by
  sorry

end smallest_n_for_Tn_gt_2006_over_2016_l298_298089


namespace Mitya_age_l298_298232

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l298_298232


namespace gcd_digit_bound_l298_298054

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l298_298054


namespace greatest_x_lcm_l298_298904

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l298_298904


namespace gcd_digits_bounded_by_lcm_l298_298057

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l298_298057


namespace problem_l298_298705

noncomputable def f(x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1
noncomputable def f_prime(x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

theorem problem (a b : ℝ) 
  (h₁ : f_prime 1 a b = 4) 
  (h₂ : f 1 a b = 3) : 
  a + b = 2 :=
sorry

end problem_l298_298705


namespace total_fish_count_l298_298318

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l298_298318


namespace cut_difference_l298_298118

-- define the conditions
def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

-- theorem to prove the correctness of the difference
theorem cut_difference : (skirt_cut - pants_cut = 0.25) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end cut_difference_l298_298118


namespace general_term_min_S9_and_S10_sum_b_seq_l298_298023

-- Definitions for the arithmetic sequence {a_n}
def a_seq (n : ℕ) : ℤ := 2 * ↑n - 20

-- Conditions provided in the problem
def cond1 : Prop := a_seq 4 = -12
def cond2 : Prop := a_seq 8 = -4

-- The sum of the first n terms S_n of the arithmetic sequence {a_n}
def S_n (n : ℕ) : ℤ := n * (a_seq 1 + a_seq n) / 2

-- Definitions for the new sequence {b_n}
def b_seq (n : ℕ) : ℤ := 2^n - 20

-- The sum of the first n terms of the new sequence {b_n}
def T_n (n : ℕ) : ℤ := (2^(n + 1) - 2) - 20 * n

-- Lean 4 theorem statements
theorem general_term (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a_seq n = 2 * ↑n - 20 :=
sorry

theorem min_S9_and_S10 (h1 : cond1) (h2 : cond2) : S_n 9 = -90 ∧ S_n 10 = -90 :=
sorry

theorem sum_b_seq (n : ℕ) : ∀ k : ℕ, (k < n) → T_n k = (2^(k+1) - 20 * k - 2) :=
sorry

end general_term_min_S9_and_S10_sum_b_seq_l298_298023


namespace gcd_digit_bound_l298_298056

theorem gcd_digit_bound (a b : ℕ) (h₁ : 10^6 ≤ a) (h₂ : a < 10^7) (h₃ : 10^6 ≤ b) (h₄ : b < 10^7) 
  (h₅ : 10^{10} ≤ lcm a b) (h₆ : lcm a b < 10^{11}) : 
  gcd a b < 10^4 :=
sorry

end gcd_digit_bound_l298_298056


namespace range_of_a_for_three_zeros_l298_298539

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298539


namespace g_zero_eq_zero_l298_298887

noncomputable def g : ℝ → ℝ :=
  sorry

axiom functional_equation (a b : ℝ) :
  g (3 * a + 2 * b) + g (3 * a - 2 * b) = 2 * g (3 * a) + 2 * g (2 * b)

theorem g_zero_eq_zero : g 0 = 0 :=
by
  let a := 0
  let b := 0
  have eqn := functional_equation a b
  sorry

end g_zero_eq_zero_l298_298887


namespace find_b_value_l298_298225

noncomputable def find_b (p q : ℕ) : ℕ := p^2 + q^2

theorem find_b_value
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_distinct : p ≠ q) (h_roots : p + q = 13 ∧ p * q = 22) :
  find_b p q = 125 :=
by
  sorry

end find_b_value_l298_298225


namespace min_expression_l298_298025

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l298_298025


namespace average_people_per_hour_rounded_l298_298593

def people_moving_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let total_hours := days * hours_per_day
  (total_people / total_hours : ℕ)

theorem average_people_per_hour_rounded :
  people_moving_per_hour 4500 5 24 = 38 := 
  sorry

end average_people_per_hour_rounded_l298_298593


namespace smallest_with_20_divisors_is_144_l298_298643

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end smallest_with_20_divisors_is_144_l298_298643


namespace number_of_real_solutions_l298_298862

noncomputable def system_of_equations_solutions_count (x : ℝ) : Prop :=
  3 * x^2 - 45 * (⌊x⌋:ℝ) + 60 = 0 ∧ 2 * x - 3 * (⌊x⌋:ℝ) + 1 = 0

theorem number_of_real_solutions : ∃ (x₁ x₂ x₃ : ℝ), system_of_equations_solutions_count x₁ ∧ system_of_equations_solutions_count x₂ ∧ system_of_equations_solutions_count x₃ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end number_of_real_solutions_l298_298862


namespace greatest_possible_x_max_possible_x_l298_298898

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l298_298898


namespace train_speed_kmph_l298_298801

theorem train_speed_kmph (length time : ℝ) (h_length : length = 90) (h_time : time = 8.999280057595392) :
  (length / time) * 3.6 = 36.003 :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- the norm_num tactic might simplify this enough, otherwise further steps would be added here.

end train_speed_kmph_l298_298801


namespace reduced_price_l298_298657

-- Definitions based on given conditions
def original_price (P : ℝ) : Prop := P > 0

def condition1 (P X : ℝ) : Prop := P * X = 700

def condition2 (P X : ℝ) : Prop := 0.7 * P * (X + 3) = 700

-- Main theorem to prove the reduced price per kg is 70
theorem reduced_price (P X : ℝ) (h1 : original_price P) (h2 : condition1 P X) (h3 : condition2 P X) : 
  0.7 * P = 70 := sorry

end reduced_price_l298_298657


namespace walter_time_at_seals_l298_298443

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l298_298443


namespace tangency_of_parabolas_l298_298013

theorem tangency_of_parabolas :
  ∃ x y : ℝ, y = x^2 + 12*x + 40
  ∧ x = y^2 + 44*y + 400
  ∧ x = -11 / 2
  ∧ y = -43 / 2 := by
sorry

end tangency_of_parabolas_l298_298013


namespace range_of_a_if_f_has_three_zeros_l298_298559

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298559


namespace shortest_distance_from_vertex_to_path_l298_298022

theorem shortest_distance_from_vertex_to_path
  (r l : ℝ)
  (hr : r = 1)
  (hl : l = 3) :
  ∃ d : ℝ, d = 1.5 :=
by
  -- Given a cone with a base radius of 1 cm and a slant height of 3 cm
  -- We need to prove the shortest distance from the vertex to the path P back to P is 1.5 cm
  sorry

end shortest_distance_from_vertex_to_path_l298_298022


namespace first_thrilling_thursday_after_start_l298_298471

theorem first_thrilling_thursday_after_start (start_date : ℕ) (school_start_month : ℕ) (school_start_day_of_week : ℤ) (month_length : ℕ → ℕ) (day_of_week_on_first_of_month : ℕ → ℤ) : 
    school_start_month = 9 ∧ school_start_day_of_week = 2 ∧ start_date = 12 ∧ month_length 9 = 30 ∧ day_of_week_on_first_of_month 10 = 0 → 
    ∃ day_of_thursday : ℕ, day_of_thursday = 26 :=
by
  sorry

end first_thrilling_thursday_after_start_l298_298471


namespace function_has_three_zeros_l298_298532

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298532


namespace expected_points_experts_probability_envelope_5_l298_298589

-- Define the conditions
def evenly_matched_teams : Prop := 
  -- Placeholder for the definition of evenly matched teams
  sorry 

def envelopes_random_choice : Prop := 
  -- Placeholder for the definition of random choice from 13 envelopes
  sorry

def game_conditions (experts_score tv_audience_score : ℕ) : Prop := 
  experts_score = 6 ∨ tv_audience_score = 6

-- Statement for part (a)
theorem expected_points_experts (h1 : evenly_matched_teams) (h2 : envelopes_random_choice) :
  game_conditions experts_score tv_audience_score →
  expected_points experts_score (100 : ℕ) = 465 :=
sorry

-- Statement for part (b)
theorem probability_envelope_5 (h1 : evenly_matched_teams) (h2 : envelopes_random_choice) :
  game_conditions experts_score tv_audience_score →
  probability_envelope_selected (5 : ℕ) = 0.715 :=
sorry

end expected_points_experts_probability_envelope_5_l298_298589


namespace total_fish_l298_298316

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l298_298316


namespace transformed_function_is_correct_l298_298579

noncomputable theory

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def right_shift_function (x : ℝ) : ℝ := (x - 2 + 1)^2 + 3

def down_shift_function (x : ℝ) : ℝ := right_shift_function x - 1

theorem transformed_function_is_correct:
  (∀ x : ℝ, down_shift_function x = (x - 1)^2 + 2) := by
  sorry

end transformed_function_is_correct_l298_298579


namespace expand_product_l298_298827

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l298_298827


namespace linear_function_details_l298_298704

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end linear_function_details_l298_298704


namespace line_through_points_l298_298694

theorem line_through_points (x1 y1 x2 y2 : ℝ) :
  (3 * x1 - 4 * y1 - 2 = 0) →
  (3 * x2 - 4 * y2 - 2 = 0) →
  (∀ x y : ℝ, (x = x1) → (y = y1) ∨ (x = x2) → (y = y2) → 3 * x - 4 * y - 2 = 0) :=
by
  sorry

end line_through_points_l298_298694


namespace annual_interest_rate_l298_298479

theorem annual_interest_rate
  (principal : ℝ) (monthly_payment : ℝ) (months : ℕ)
  (H1 : principal = 150) (H2 : monthly_payment = 13) (H3 : months = 12) :
  (monthly_payment * months - principal) / principal * 100 = 4 :=
by
  sorry

end annual_interest_rate_l298_298479


namespace find_divisor_l298_298658

-- Define the given conditions
def dividend : ℕ := 122
def quotient : ℕ := 6
def remainder : ℕ := 2

-- Define the proof problem to find the divisor
theorem find_divisor : 
  ∃ D : ℕ, dividend = (D * quotient) + remainder ∧ D = 20 :=
by sorry

end find_divisor_l298_298658


namespace songs_can_be_stored_l298_298394

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end songs_can_be_stored_l298_298394


namespace min_value_of_f_product_of_zeros_l298_298508

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x - x - m

theorem min_value_of_f (m : ℝ) (h : m < -2) :
  ∀ x : ℝ, x ∈ set.Icc (1 / real.exp 1) real.exp 1 → f x m ≤ 1 - real.exp 1 - m :=
sorry

theorem product_of_zeros (m : ℝ) (h : m < -2) (x1 x2 : ℝ) (hx1 : f x1 m = 0) (hx2 : f x2 m = 0) (h_order : x1 < x2) :
  x1 * x2 < 1 :=
sorry

end min_value_of_f_product_of_zeros_l298_298508


namespace find_solutions_l298_298148

noncomputable
def is_solution (a b c d : ℝ) : Prop :=
  a + b + c = d ∧ (1 / a + 1 / b + 1 / c = 1 / d)

theorem find_solutions (a b c d : ℝ) :
  is_solution a b c d ↔ (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) :=
by
  sorry

end find_solutions_l298_298148


namespace lemango_eating_mangos_l298_298860

theorem lemango_eating_mangos :
  ∃ (mangos_eaten : ℕ → ℕ), 
    (mangos_eaten 1 * (2^6 - 1) = 364 * (2 - 1)) ∧
    (mangos_eaten 6 = 128) :=
by
  sorry

end lemango_eating_mangos_l298_298860


namespace car_kilometers_per_gallon_l298_298300

-- Define the given conditions as assumptions
variable (total_distance : ℝ) (total_gallons : ℝ)
-- Assume the given conditions
axiom h1 : total_distance = 180
axiom h2 : total_gallons = 4.5

-- The statement to be proven
theorem car_kilometers_per_gallon : (total_distance / total_gallons) = 40 :=
by
  -- Sorry is used to skip the proof
  sorry

end car_kilometers_per_gallon_l298_298300


namespace compute_expression_l298_298818

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 :=
by sorry

end compute_expression_l298_298818


namespace Powerjet_pumps_250_gallons_in_30_minutes_l298_298252

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l298_298252


namespace find_a_for_quadratic_l298_298173

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l298_298173


namespace min_a_b_div_1176_l298_298373

theorem min_a_b_div_1176 (a b : ℕ) (h : b^3 = 1176 * a) : a = 63 :=
by sorry

end min_a_b_div_1176_l298_298373


namespace first_digit_base5_of_312_is_2_l298_298634

theorem first_digit_base5_of_312_is_2 :
  ∃ d : ℕ, d = 2 ∧ (∀ n : ℕ, d * 5 ^ n ≤ 312 ∧ 312 < (d + 1) * 5 ^ n) :=
by
  sorry

end first_digit_base5_of_312_is_2_l298_298634


namespace min_value_of_xsquare_ysquare_l298_298339

variable {x y : ℝ}

theorem min_value_of_xsquare_ysquare (h : 5 * x^2 * y^2 + y^4 = 1) : x^2 + y^2 ≥ 4 / 5 :=
sorry

end min_value_of_xsquare_ysquare_l298_298339


namespace cars_meet_after_40_minutes_l298_298880

noncomputable def time_to_meet 
  (BC CD : ℝ) (speed : ℝ) 
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) : ℝ :=
  (BC + CD) / speed * 40 / 60

-- Define the condition that must hold: cars meet at 40 minutes
theorem cars_meet_after_40_minutes
  (BC CD : ℝ) (speed : ℝ)
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) :
  time_to_meet BC CD speed constant_speed = 40 := sorry

end cars_meet_after_40_minutes_l298_298880


namespace concentration_of_acid_in_third_flask_l298_298276

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l298_298276


namespace quadrilateral_area_l298_298668

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle A C D

def A : Point := ⟨2, 2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨3, -1⟩
def D : Point := ⟨2007, 2008⟩

theorem quadrilateral_area :
  area_of_quadrilateral A B C D = 2008006.5 :=
by
  sorry

end quadrilateral_area_l298_298668


namespace smallest_value_l298_298821

theorem smallest_value : 54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end smallest_value_l298_298821


namespace range_of_a_l298_298500

noncomputable def f (a x : ℝ) := a * x^2 - (2 - a) * x + 1
noncomputable def g (x : ℝ) := x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) :=
by
  sorry

end range_of_a_l298_298500


namespace chloe_cherries_l298_298361

noncomputable def cherries_received (x y : ℝ) : Prop :=
  x = y + 8 ∧ y = x / 3

theorem chloe_cherries : ∃ (x : ℝ), ∀ (y : ℝ), cherries_received x y → x = 12 := 
by
  sorry

end chloe_cherries_l298_298361


namespace minimum_value_of_expr_l298_298181

noncomputable def expr (x y : ℝ) : ℝ := 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4

theorem minimum_value_of_expr : ∃ x y : ℝ, expr x y = -1 ∧ ∀ (a b : ℝ), expr a b ≥ -1 := 
by
  sorry

end minimum_value_of_expr_l298_298181


namespace find_numbers_l298_298229

theorem find_numbers (A B C D : ℚ) 
  (h1 : A + B = 44)
  (h2 : 5 * A = 6 * B)
  (h3 : C = 2 * (A - B))
  (h4 : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := 
  by 
    sorry

end find_numbers_l298_298229


namespace equation_has_roots_l298_298151

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l298_298151


namespace vector_parallel_m_l298_298297

theorem vector_parallel_m {m : ℝ} (h : (2:ℝ) * m - (-1 * -1) = 0) : m = 1 / 2 := 
by
  sorry

end vector_parallel_m_l298_298297


namespace arithmetic_geometric_sequence_product_l298_298255

theorem arithmetic_geometric_sequence_product :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 = 3 →
    (a 1) + (a 1 * q^2) + (a 1 * q^4) = 21 →
    (a 2) * (a 6) = 72 :=
by 
  intros a q h1 h2 
  sorry

end arithmetic_geometric_sequence_product_l298_298255


namespace range_of_a_if_f_has_three_zeros_l298_298558

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298558


namespace GCF_of_48_180_98_l298_298095

theorem GCF_of_48_180_98 : Nat.gcd (Nat.gcd 48 180) 98 = 2 :=
by
  sorry

end GCF_of_48_180_98_l298_298095


namespace value_of_a_with_two_distinct_roots_l298_298176

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l298_298176


namespace value_of_one_stamp_l298_298732

theorem value_of_one_stamp (matches_per_book : ℕ) (initial_stamps : ℕ) (trade_matchbooks : ℕ) (stamps_left : ℕ) :
  matches_per_book = 24 → initial_stamps = 13 → trade_matchbooks = 5 → stamps_left = 3 →
  (trade_matchbooks * matches_per_book) / (initial_stamps - stamps_left) = 12 :=
by
  intros h1 h2 h3 h4
  -- Insert the logical connection assertions here, concluding with the final proof step.
  sorry

end value_of_one_stamp_l298_298732


namespace sqrt_14_plus_2_range_l298_298138

theorem sqrt_14_plus_2_range :
  5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 :=
by
  sorry

end sqrt_14_plus_2_range_l298_298138


namespace max_xy_min_x2y2_l298_298497

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x * y ≤ 1 / 8) :=
sorry

theorem min_x2y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x ^ 2 + y ^ 2 ≥ 1 / 5) :=
sorry


end max_xy_min_x2y2_l298_298497


namespace allocate_25_rubles_in_4_weighings_l298_298441

theorem allocate_25_rubles_in_4_weighings :
  ∃ (coins : ℕ) (coins5 : ℕ → ℕ), 
    (coins = 1600) ∧ 
    (coins5 0 = 800 ∧ coins5 1 = 800) ∧
    (coins5 2 = 400 ∧ coins5 3 = 400) ∧
    (coins5 4 = 200 ∧ coins5 5 = 200) ∧
    (coins5 6 = 100 ∧ coins5 7 = 100) ∧
    (
      25 = 20 + 5 ∧ 
      (∃ i j k l m n, coins5 i = 400 ∧ coins5 j = 400 ∧ coins5 k = 200 ∧
        coins5 l = 200 ∧ coins5 m = 100 ∧ coins5 n = 100)
    )
  := 
sorry

end allocate_25_rubles_in_4_weighings_l298_298441


namespace percentage_salt_in_mixture_l298_298305

-- Conditions
def volume_pure_water : ℝ := 1
def volume_salt_solution : ℝ := 2
def salt_concentration : ℝ := 0.30
def total_volume : ℝ := volume_pure_water + volume_salt_solution
def amount_of_salt_in_solution : ℝ := salt_concentration * volume_salt_solution

-- Theorem
theorem percentage_salt_in_mixture :
  (amount_of_salt_in_solution / total_volume) * 100 = 20 :=
by
  sorry

end percentage_salt_in_mixture_l298_298305


namespace jenna_hike_duration_l298_298815

-- Definitions from conditions
def initial_speed : ℝ := 25
def exhausted_speed : ℝ := 10
def total_distance : ℝ := 140
def total_time : ℝ := 8

-- The statement to prove:
theorem jenna_hike_duration : ∃ x : ℝ, 25 * x + 10 * (8 - x) = 140 ∧ x = 4 := by
  sorry

end jenna_hike_duration_l298_298815


namespace linda_original_savings_l298_298871

theorem linda_original_savings (S : ℝ) (f : ℝ) (a : ℝ) (t : ℝ) 
  (h1 : f = 7 / 13 * S) (h2 : a = 3 / 13 * S) 
  (h3 : t = S - f - a) (h4 : t = 180) (h5 : a = 360) : 
  S = 1560 :=
by 
  sorry

end linda_original_savings_l298_298871


namespace right_triangle_properties_l298_298617

theorem right_triangle_properties (a b c : ℕ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) :
  ∃ (area perimeter : ℕ), area = 30 ∧ perimeter = 30 ∧ (a < c ∧ b < c) :=
by
  let area := 1 / 2 * a * b
  let perimeter := a + b + c
  have acute_angles : a < c ∧ b < c := by sorry
  exact ⟨area, perimeter, ⟨sorry, sorry, acute_angles⟩⟩

end right_triangle_properties_l298_298617


namespace greatest_x_lcm_105_l298_298914

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l298_298914


namespace trigonometric_identity_l298_298350

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ π / 2)
  (h2 : cos α = 3 / 5) :
  (1 + sqrt 2 * cos (2 * α - π / 4)) / sin (α + π / 2) = 14 / 5 :=
by
  sorry

end trigonometric_identity_l298_298350


namespace greatest_x_lcm_l298_298924

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l298_298924


namespace unique_int_pair_exists_l298_298404

theorem unique_int_pair_exists (a b : ℤ) : 
  ∃! (x y : ℤ), (x + 2 * y - a)^2 + (2 * x - y - b)^2 ≤ 1 :=
by
  sorry

end unique_int_pair_exists_l298_298404


namespace prove_p_false_and_q_true_l298_298382

variables (p q : Prop)

theorem prove_p_false_and_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by {
  -- proof placeholder
  sorry
}

end prove_p_false_and_q_true_l298_298382


namespace solve_quadratic_l298_298835

theorem solve_quadratic (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
sorry

end solve_quadratic_l298_298835


namespace square_area_l298_298805

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l298_298805


namespace quadratic_shift_l298_298576

theorem quadratic_shift (x : ℝ) :
  let f := (x + 1)^2 + 3
  let g := (x - 1)^2 + 2
  shift_right (f, 2) -- condition 2: shift right by 2
  shift_down (f, 1) -- condition 3: shift down by 1
  f = g :=
sorry

# where shift_right and shift_down are placeholder for actual implementation 

end quadratic_shift_l298_298576


namespace ratio_of_doctors_to_lawyers_l298_298082

variable (d l : ℕ) -- number of doctors and lawyers
variable (h1 : (40 * d + 55 * l) / (d + l) = 45) -- overall average age condition

theorem ratio_of_doctors_to_lawyers : d = 2 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l298_298082


namespace circle_circumference_ratio_l298_298615

theorem circle_circumference_ratio (q r p : ℝ) (hq : p = q + r) : 
  (2 * Real.pi * q + 2 * Real.pi * r) / (2 * Real.pi * p) = 1 :=
by
  sorry

end circle_circumference_ratio_l298_298615


namespace find_coef_of_quadratic_l298_298748

-- Define the problem conditions
def solutions_of_abs_eq : Set ℤ := {x | abs (x - 3) = 4}

-- Given that the solutions are 7 and -1
def paul_solutions : Set ℤ := {7, -1}

-- The problem translates to proving the equivalence of two sets
def equivalent_equation_solutions (d e : ℤ) : Prop :=
  ∀ x, x ∈ solutions_of_abs_eq ↔ x^2 + d * x + e = 0

theorem find_coef_of_quadratic :
  equivalent_equation_solutions (-6) (-7) :=
by
  sorry

end find_coef_of_quadratic_l298_298748


namespace find_k_l298_298491

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 3 → k * (x^2 + 6 * x - k) * (x^2 + x - 12) > 0) ↔ (k ≤ -9) :=
by sorry

end find_k_l298_298491


namespace order_of_products_l298_298597

theorem order_of_products (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) : b * x > a * x ∧ a * x > a ^ 2 :=
by
  sorry

end order_of_products_l298_298597


namespace concentration_third_flask_l298_298282

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l298_298282


namespace concentration_of_acid_in_third_flask_l298_298274

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l298_298274


namespace range_of_a_for_three_zeros_l298_298551

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∃ f : ℝ → ℝ, f = λ x, x^3 + a * x + 2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) → a < -3 :=
by
  -- Proof omitted
  sorry

end range_of_a_for_three_zeros_l298_298551


namespace sum_of_areas_l298_298981

theorem sum_of_areas (r s t : ℝ)
  (h1 : r + s = 13)
  (h2 : s + t = 5)
  (h3 : r + t = 12)
  (h4 : t = r / 2) : 
  π * (r ^ 2 + s ^ 2 + t ^ 2) = 105 * π := 
by
  sorry

end sum_of_areas_l298_298981


namespace vincent_back_to_A_after_5_min_p_plus_q_computation_l298_298960

def probability (n : ℕ) : ℚ :=
  if n = 0 then 1
  else 1 / 4 * (1 - probability (n - 1))

theorem vincent_back_to_A_after_5_min : 
  probability 5 = 51 / 256 :=
by sorry

theorem p_plus_q_computation :
  51 + 256 = 307 :=
by linarith

end vincent_back_to_A_after_5_min_p_plus_q_computation_l298_298960


namespace common_root_solutions_l298_298570

theorem common_root_solutions (a : ℝ) (b : ℝ) :
  (a^2 * b^2 + a * b - 1 = 0) ∧ (b^2 - a * b - a^2 = 0) →
  a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
  a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2 :=
by
  intro h
  sorry

end common_root_solutions_l298_298570


namespace range_of_a_for_three_zeros_l298_298519

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end range_of_a_for_three_zeros_l298_298519


namespace polynomial_expansion_l298_298139

variable (x : ℝ)

theorem polynomial_expansion : 
  (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 :=
by
  sorry

end polynomial_expansion_l298_298139


namespace factorize_expr_l298_298140

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l298_298140


namespace k_is_2_l298_298517

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1
def g (x : ℝ) : ℝ := 0
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

theorem k_is_2 :
  (∀ x ∈ Set.Icc 1 (2 * Real.exp 1), 0 ≤ f k x ∧ f k x ≤ h x) ↔ (k = 2) :=
  sorry

end k_is_2_l298_298517


namespace third_square_length_l298_298131

theorem third_square_length 
  (A1 : 8 * 5 = 40) 
  (A2 : 10 * 7 = 70) 
  (A3 : 15 * 9 = 135) 
  (L : ℕ) 
  (A4 : 40 + 70 + L * 5 = 135) 
  : L = 5 := 
sorry

end third_square_length_l298_298131


namespace ash_cloud_ratio_l298_298121

theorem ash_cloud_ratio
  (distance_ashes_shot_up : ℕ)
  (radius_ash_cloud : ℕ)
  (h1 : distance_ashes_shot_up = 300)
  (h2 : radius_ash_cloud = 2700) :
  (2 * radius_ash_cloud) / distance_ashes_shot_up = 18 :=
by
  sorry

end ash_cloud_ratio_l298_298121


namespace common_altitude_l298_298614

theorem common_altitude (A1 A2 b1 b2 h : ℝ)
    (hA1 : A1 = 800)
    (hA2 : A2 = 1200)
    (hb1 : b1 = 40)
    (hb2 : b2 = 60)
    (h1 : A1 = 1 / 2 * b1 * h)
    (h2 : A2 = 1 / 2 * b2 * h) :
    h = 40 := 
sorry

end common_altitude_l298_298614


namespace equation_has_two_distinct_roots_l298_298156

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l298_298156


namespace expand_product_l298_298826

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l298_298826


namespace concentration_third_flask_l298_298271

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l298_298271


namespace number_of_terms_in_arithmetic_sequence_l298_298203

-- Definitions and conditions
def a : ℤ := -58  -- First term
def d : ℤ := 7   -- Common difference
def l : ℤ := 78  -- Last term

-- Statement of the problem
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 20 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l298_298203


namespace acid_concentration_third_flask_l298_298283

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l298_298283


namespace shuxue_count_l298_298387

theorem shuxue_count : 
  (∃ (count : ℕ), count = (List.length (List.filter (λ n => (30 * n.1 + 3 * n.2 < 100) 
    ∧ (30 * n.1 + 3 * n.2 > 9)) 
      (List.product 
        (List.range' 1 3) -- Possible values for "a" are 1 to 3
        (List.range' 1 9)) -- Possible values for "b" are 1 to 9
    ))) ∧ count = 9 :=
  sorry

end shuxue_count_l298_298387


namespace digit_at_position_2020_l298_298869

def sequence_digit (n : Nat) : Nat :=
  -- Function to return the nth digit of the sequence formed by concatenating the integers from 1 to 1000
  sorry

theorem digit_at_position_2020 : sequence_digit 2020 = 7 :=
  sorry

end digit_at_position_2020_l298_298869


namespace polynomial_abs_sum_l298_298067

theorem polynomial_abs_sum (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  (1 - (2:ℝ) * x) ^ 8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = (3:ℝ) ^ 8 :=
sorry

end polynomial_abs_sum_l298_298067


namespace minimum_triangle_area_l298_298696

theorem minimum_triangle_area (r a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = b) : 
  ∀ T, (T = (a + b) * r / 2) → T = 2 * r * r :=
by 
  sorry

end minimum_triangle_area_l298_298696


namespace distance_between_parallel_lines_l298_298889

theorem distance_between_parallel_lines (A B c1 c2 : Real) (hA : A = 2) (hB : B = 3) 
(hc1 : c1 = -3) (hc2 : c2 = 2) : 
    (abs (c1 - c2) / Real.sqrt (A^2 + B^2)) = (5 * Real.sqrt 13 / 13) := by
  sorry

end distance_between_parallel_lines_l298_298889


namespace tax_rate_as_percent_l298_298963

def TaxAmount (amount : ℝ) : Prop := amount = 82
def BaseAmount (amount : ℝ) : Prop := amount = 100

theorem tax_rate_as_percent {tax_amt base_amt : ℝ} 
  (h_tax : TaxAmount tax_amt) (h_base : BaseAmount base_amt) : 
  (tax_amt / base_amt) * 100 = 82 := 
by 
  sorry

end tax_rate_as_percent_l298_298963


namespace isolate_y_l298_298693

theorem isolate_y (x y : ℝ) (h : 3 * x - 2 * y = 6) : y = 3 * x / 2 - 3 :=
sorry

end isolate_y_l298_298693


namespace o_l298_298326

theorem o'hara_triple_example (a b x : ℕ) (h₁ : a = 49) (h₂ : b = 16) (h₃ : x = (Int.sqrt a).toNat + (Int.sqrt b).toNat) : x = 11 := 
by
  sorry

end o_l298_298326


namespace range_of_a_range_of_f_diff_l298_298256

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔ (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) :=
by
  sorry

theorem range_of_f_diff (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h12 : x1 ≠ x2) : 
  0 < f a x1 - f a x2 :=
by
  sorry

end range_of_a_range_of_f_diff_l298_298256


namespace max_x_lcm_15_21_105_l298_298907

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l298_298907


namespace jose_investment_proof_l298_298782

noncomputable def jose_investment (total_profit jose_share : ℕ) (tom_investment : ℕ) (months_tom months_jose : ℕ) : ℕ :=
  let tom_share := total_profit - jose_share
  let tom_investment_mr := tom_investment * months_tom
  let ratio := tom_share * months_jose
  tom_investment_mr * jose_share / ratio

theorem jose_investment_proof : 
  ∃ (jose_invested : ℕ), 
    let total_profit := 5400
    let jose_share := 3000
    let tom_invested := 3000
    let months_tom := 12
    let months_jose := 10
    jose_investment total_profit jose_share tom_invested months_tom months_jose = 4500 :=
by
  use 4500
  sorry

end jose_investment_proof_l298_298782


namespace tennis_tournament_non_persistent_days_l298_298574

-- Definitions based on conditions
structure TennisTournament where
  n : ℕ -- Number of players
  h_n_gt4 : n > 4 -- More than 4 players
  matches : Finset (Fin n × Fin n) -- Set of matches
  h_matches_unique : ∀ (i j : Fin n), i ≠ j → ((i, j) ∈ matches ↔ (j, i) ∈ matches)
  persistent : Fin n → Prop
  nonPersistent : Fin n → Prop
  h_players : ∀ i, persistent i ∨ nonPersistent i
  h_oneGamePerDay : ∀ {A B : Fin n}, (A, B) ∈ matches → (A ≠ B)

-- Main theorem based on the proof problem
theorem tennis_tournament_non_persistent_days (tournament : TennisTournament) :
  ∃ days_nonPersistent, 2 * days_nonPersistent > tournament.n - 1 := by
  sorry

end tennis_tournament_non_persistent_days_l298_298574


namespace biker_distance_and_speed_l298_298789

variable (D V : ℝ)

theorem biker_distance_and_speed (h1 : D / 2 = V * 2.5)
                                  (h2 : D / 2 = (V + 2) * (7 / 3)) :
  D = 140 ∧ V = 28 :=
by
  sorry

end biker_distance_and_speed_l298_298789


namespace range_of_a_if_f_has_three_zeros_l298_298555

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l298_298555


namespace prob_seven_heads_in_ten_tosses_l298_298241

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.choose n k)

noncomputable def probability_of_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (0.5^k : ℚ) * (0.5^(n - k) : ℚ)

theorem prob_seven_heads_in_ten_tosses :
  probability_of_heads 10 7 = 15 / 128 :=
by
  sorry

end prob_seven_heads_in_ten_tosses_l298_298241


namespace probability_of_winning_l298_298962

def total_products_in_box : ℕ := 6
def winning_products_in_box : ℕ := 2

theorem probability_of_winning : (winning_products_in_box : ℚ) / (total_products_in_box : ℚ) = 1 / 3 :=
by sorry

end probability_of_winning_l298_298962


namespace total_cost_of_crayons_l298_298875

-- Definition of the initial conditions
def usual_price : ℝ := 2.5
def discount_rate : ℝ := 0.15
def packs_initial : ℕ := 4
def packs_to_buy : ℕ := 2

-- Calculate the discounted price for one pack
noncomputable def discounted_price : ℝ :=
  usual_price - (usual_price * discount_rate)

-- Calculate the total cost of packs after purchase and validate it
theorem total_cost_of_crayons :
  (packs_initial * usual_price) + (packs_to_buy * discounted_price) = 14.25 :=
by
  sorry

end total_cost_of_crayons_l298_298875


namespace no5_battery_mass_l298_298302

theorem no5_battery_mass :
  ∃ (x y : ℝ), 2 * x + 2 * y = 72 ∧ 3 * x + 2 * y = 96 ∧ x = 24 :=
by
  sorry

end no5_battery_mass_l298_298302


namespace Sanji_received_86_coins_l298_298019

noncomputable def total_coins := 280

def Jack_coins (x : ℕ) := x
def Jimmy_coins (x : ℕ) := x + 11
def Tom_coins (x : ℕ) := x - 15
def Sanji_coins (x : ℕ) := x + 20

theorem Sanji_received_86_coins (x : ℕ) (hx : Jack_coins x + Jimmy_coins x + Tom_coins x + Sanji_coins x = total_coins) : Sanji_coins x = 86 :=
sorry

end Sanji_received_86_coins_l298_298019


namespace tangent_line_y_intercept_l298_298469

noncomputable def y_intercept_tangent_line (R1_center R2_center : ℝ × ℝ)
  (R1_radius R2_radius : ℝ) : ℝ :=
if R1_center = (3,0) ∧ R2_center = (8,0) ∧ R1_radius = 3 ∧ R2_radius = 2
then 15 * Real.sqrt 26 / 26
else 0

theorem tangent_line_y_intercept : 
  y_intercept_tangent_line (3,0) (8,0) 3 2 = 15 * Real.sqrt 26 / 26 :=
by
  -- proof goes here
  sorry

end tangent_line_y_intercept_l298_298469


namespace function_has_three_zeros_l298_298534

theorem function_has_three_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
    ∀ x, (x = x1 ∨ x = x2 ∨ x = x3) ↔ (x^3 + a * x + 2 = 0)) → a < -3 := by
  sorry

end function_has_three_zeros_l298_298534


namespace probability_each_box_2_fruits_l298_298636

noncomputable def totalWaysToDistributePears : ℕ := (Nat.choose 8 4)
noncomputable def totalWaysToDistributeApples : ℕ := 5^6

noncomputable def case1 : ℕ := (Nat.choose 5 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))
noncomputable def case2 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1))
noncomputable def case3 : ℕ := (Nat.choose 5 4) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))

noncomputable def totalFavorableDistributions : ℕ := case1 + case2 + case3
noncomputable def totalPossibleDistributions : ℕ := totalWaysToDistributePears * totalWaysToDistributeApples

noncomputable def probability : ℚ := (totalFavorableDistributions : ℚ) / totalPossibleDistributions * 100

theorem probability_each_box_2_fruits :
  probability = 0.74 := 
sorry

end probability_each_box_2_fruits_l298_298636


namespace ten_thousand_times_ten_thousand_l298_298081

theorem ten_thousand_times_ten_thousand :
  10000 * 10000 = 100000000 :=
by
  sorry

end ten_thousand_times_ten_thousand_l298_298081


namespace john_has_22_dimes_l298_298939

theorem john_has_22_dimes (d q : ℕ) (h1 : d = q + 4) (h2 : 10 * d + 25 * q = 680) : d = 22 :=
by
sorry

end john_has_22_dimes_l298_298939


namespace prob_all_boys_l298_298307

-- Define the events
def is_boy (child : ℕ) : Prop := child = 1 -- Assuming 1 represents a boy and 0 represents a girl

noncomputable def probability_all_boys_given_conditions : ℝ :=
  let sample_space := {c | c.length = 4 ∧ is_boy (c.head) ∧ (∃ i, i ≠ 0 ∧ is_boy (c.get! i))} in
  let favorable_outcome := {c | c = [1, 1, 1, 1]} in
  (favorable_outcome ∩ sample_space).card.to_real / sample_space.card.to_real

-- Theorem to prove the correctness of the answer
theorem prob_all_boys :
  probability_all_boys_given_conditions = 1 / 5 :=
sorry

end prob_all_boys_l298_298307


namespace arrangement_count_l298_298458

noncomputable def count_arrangements (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  sorry -- The implementation of this function is out of scope for this task

theorem arrangement_count :
  count_arrangements ({1, 2, 3, 4} : Finset ℕ) ({1, 2, 3} : Finset ℕ) = 18 :=
sorry

end arrangement_count_l298_298458


namespace probability_event_l298_298754

open MeasureTheory

-- Conditions: Method of choosing numbers
def coin_flip_distribution : measure ℝ :=
  (1/2) • uniform_of [0, 1] + (1/4) • dirac 0 + (1/4) • dirac 0.5

-- Probability measure resulting from two independent selections
noncomputable def prob_distribution : measure (ℝ × ℝ) :=
  coin_flip_distribution.prod coin_flip_distribution

-- Probability event definition
def event (x y : ℝ) := |x - y| ≥ 1/2

-- Desired Probability calculation
noncomputable def desired_probability : ℝ :=
  prob_distribution.to_outer_measure.measure_of {xy | event xy.1 xy.2}

theorem probability_event : desired_probability = 1/8 :=
by
  -- Proof would go here, but is left out
  sorry

end probability_event_l298_298754


namespace walter_time_at_seals_l298_298444

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l298_298444


namespace converse_and_inverse_false_l298_298707

-- Define the property of being a rhombus and a parallelogram
def is_rhombus (R : Type) : Prop := sorry
def is_parallelogram (P : Type) : Prop := sorry

-- Given: If a quadrilateral is a rhombus, then it is a parallelogram
def quad_imp (Q : Type) : Prop := is_rhombus Q → is_parallelogram Q

-- Prove that the converse and inverse are false
theorem converse_and_inverse_false (Q : Type) 
  (h1 : quad_imp Q) : 
  ¬(is_parallelogram Q → is_rhombus Q) ∧ ¬(¬(is_rhombus Q) → ¬(is_parallelogram Q)) :=
by
  sorry

end converse_and_inverse_false_l298_298707


namespace product_of_values_l298_298498

-- Given definitions: N as a real number and R as a real constant
variables (N R : ℝ)

-- Condition
def condition : Prop := N - 5 / N = R

-- The proof statement
theorem product_of_values (h : condition N R) : ∀ (N1 N2 : ℝ), ((N1 - 5 / N1 = R) ∧ (N2 - 5 / N2 = R)) → (N1 * N2 = -5) :=
by sorry

end product_of_values_l298_298498


namespace valid_a_value_l298_298169

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l298_298169


namespace min_value_a_l298_298016

theorem min_value_a (a b c : ℤ) (α β : ℝ)
  (h_a_pos : a > 0) 
  (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 0 → (x = α ∨ x = β))
  (h_alpha_beta_order : 0 < α ∧ α < β ∧ β < 1) :
  a ≥ 5 :=
sorry

end min_value_a_l298_298016


namespace total_quartet_songs_l298_298994

/-- 
Five girls — Mary, Alina, Tina, Hanna, and Elsa — sang songs in a concert as quartets,
with one girl sitting out each time. Hanna sang 9 songs, which was more than any other girl,
and Mary sang 3 songs, which was fewer than any other girl. If the total number of songs
sung by Alina and Tina together was 16, then the total number of songs sung by these quartets is 8. -/
theorem total_quartet_songs
  (hanna_songs : ℕ) (mary_songs : ℕ) (alina_tina_songs : ℕ) (total_songs : ℕ)
  (h_hanna : hanna_songs = 9)
  (h_mary : mary_songs = 3)
  (h_alina_tina : alina_tina_songs = 16) :
  total_songs = 8 :=
sorry

end total_quartet_songs_l298_298994


namespace semicircle_circumference_correct_l298_298257

noncomputable def perimeter_of_rectangle (l b : ℝ) : ℝ := 2 * (l + b)
noncomputable def side_of_square_by_rectangle (l b : ℝ) : ℝ := perimeter_of_rectangle l b / 4
noncomputable def circumference_of_semicircle (d : ℝ) : ℝ := (Real.pi * (d / 2)) + d

theorem semicircle_circumference_correct :
  let l := 16
  let b := 12
  let d := side_of_square_by_rectangle l b
  circumference_of_semicircle d = 35.98 :=
by
  sorry

end semicircle_circumference_correct_l298_298257


namespace valid_a_value_l298_298168

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l298_298168


namespace find_x_plus_inv_x_l298_298197

theorem find_x_plus_inv_x (x : ℝ) (hx_pos : 0 < x) (h : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) :
  x + 1/x = 3 :=
by
  sorry

end find_x_plus_inv_x_l298_298197


namespace quadratic_two_distinct_roots_l298_298720

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k*x^2 - 6*x + 9 = 0) ∧ (k*y^2 - 6*y + 9 = 0)) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_roots_l298_298720


namespace concentration_in_third_flask_l298_298279

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l298_298279


namespace B_work_days_proof_l298_298797

-- Define the main variables
variables (W : ℝ) (x : ℝ) (daysA : ℝ) (daysBworked : ℝ) (daysAremaining : ℝ)

-- Given conditions from the problem
def A_work_days : ℝ := 6
def B_work_days : ℝ := x
def B_worked_days : ℝ := 10
def A_remaining_days : ℝ := 2

-- We are asked to prove this statement
theorem B_work_days_proof (h1 : daysA = A_work_days)
                           (h2 : daysBworked = B_worked_days)
                           (h3 : daysAremaining = A_remaining_days) 
                           (hx : (W/6 = (W - 10*W/x) / 2)) : x = 15 :=
by 
  -- Proof omitted
  sorry 

end B_work_days_proof_l298_298797


namespace cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l298_298746

-- Definitions based on conditions
def distanceAB := 18  -- km
def speedCarA := 54   -- km/h
def speedCarB := 36   -- km/h
def targetDistance := 45  -- km

-- Proof problem statements
theorem cars_towards_each_other {y : ℝ} : 54 * y + 36 * y = 18 + 45 ↔ y = 0.7 :=
by sorry

theorem cars_same_direction_A_to_B {x : ℝ} : 54 * x - (36 * x + 18) = 45 ↔ x = 3.5 :=
by sorry

theorem cars_same_direction_B_to_A {x : ℝ} : 54 * x + 18 - 36 * x = 45 ↔ x = 1.5 :=
by sorry

end cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l298_298746


namespace negate_proposition_l298_298345

variable (x : ℝ)

theorem negate_proposition :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0)) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 :=
by
  sorry

end negate_proposition_l298_298345


namespace negate_exponential_inequality_l298_298936

theorem negate_exponential_inequality :
  ¬ (∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x :=
by
  sorry

end negate_exponential_inequality_l298_298936


namespace number_of_children_l298_298611

namespace CurtisFamily

variables {m x : ℕ} {xy : ℕ}

/-- Given conditions for Curtis family average ages. -/
def family_average_age (m x xy : ℕ) : Prop := (m + 50 + xy) / (2 + x) = 25

def mother_children_average_age (m x xy : ℕ) : Prop := (m + xy) / (1 + x) = 20

/-- The number of children in Curtis family is 4, given the average age conditions. -/
theorem number_of_children (m xy : ℕ) (h1 : family_average_age m 4 xy) (h2 : mother_children_average_age m 4 xy) : x = 4 :=
by
  sorry

end CurtisFamily

end number_of_children_l298_298611


namespace problem1_problem2_l298_298069

theorem problem1 (a b : ℤ) (h : Even (5 * b + a)) : Even (a - 3 * b) :=
sorry

theorem problem2 (a b : ℤ) (h : Odd (5 * b + a)) : Odd (a - 3 * b) :=
sorry

end problem1_problem2_l298_298069


namespace problem_statement_l298_298360

noncomputable def f (x : ℝ) : ℝ := (sin x + sqrt 3 * cos x) * sin x + 3 / 2

theorem problem_statement :
  (∀ k : ℤ, by
    let I := Set.Icc (-π / 6 + k * π) (π / 3 + k * π)
    (monotonic_increasing f I)) ∧
  (let A : ℝ := π / 3,
       b : ℝ := 2,
       S : ℝ := 2 * sqrt 3 in
    (a = 2 * sqrt 3) ∧ (c = 4) ∧ (f A = 3) → A = π / 3 ∧ b = 2 ∧ S = 2 * sqrt 3
  )
:= 
sorry

end problem_statement_l298_298360


namespace max_xy_l298_298762

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions given in the problem
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom eq1 : x + 1/y = 3
axiom eq2 : y + 2/x = 3

theorem max_xy : ∃ (xy : ℝ), 
  xy = x * y ∧ xy = 3 + Real.sqrt 7 := sorry

end max_xy_l298_298762


namespace true_weight_third_object_proof_l298_298467

noncomputable def true_weight_third_object (A a B b C : ℝ) : ℝ :=
  let h := Real.sqrt ((a - b) / (A - B))
  let k := (b * A - a * B) / ((A - B) * (h + 1))
  h * C + k

theorem true_weight_third_object_proof (A a B b C : ℝ) (h := Real.sqrt ((a - b) / (A - B))) (k := (b * A - a * B) / ((A - B) * (h + 1))) :
  true_weight_third_object A a B b C = h * C + k := by
  sorry

end true_weight_third_object_proof_l298_298467


namespace largest_fraction_l298_298502

theorem largest_fraction
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d) :
  (c + d) / (a + b) ≥ (a + b) / (c + d)
  ∧ (c + d) / (a + b) ≥ (a + d) / (b + c)
  ∧ (c + d) / (a + b) ≥ (b + c) / (a + d)
  ∧ (c + d) / (a + b) ≥ (b + d) / (a + c) :=
by
  sorry

end largest_fraction_l298_298502


namespace power_function_value_l298_298357

theorem power_function_value
  (α : ℝ)
  (h : 2^α = Real.sqrt 2) :
  (4 : ℝ) ^ α = 2 :=
by {
  sorry
}

end power_function_value_l298_298357


namespace total_coin_value_l298_298513

theorem total_coin_value (total_coins : ℕ) (two_dollar_coins : ℕ) (one_dollar_value : ℕ)
  (two_dollar_value : ℕ) (h_total_coins : total_coins = 275)
  (h_two_dollar_coins : two_dollar_coins = 148)
  (h_one_dollar_value : one_dollar_value = 1)
  (h_two_dollar_value : two_dollar_value = 2) :
  total_coins - two_dollar_coins = 275 - 148
  ∧ ((total_coins - two_dollar_coins) * one_dollar_value + two_dollar_coins * two_dollar_value) = 423 :=
by
  sorry

end total_coin_value_l298_298513


namespace tan_product_cos_conditions_l298_298833

variable {α β : ℝ}

theorem tan_product_cos_conditions
  (h1 : Real.cos (α + β) = 2 / 3)
  (h2 : Real.cos (α - β) = 1 / 3) :
  Real.tan α * Real.tan β = -1 / 3 :=
sorry

end tan_product_cos_conditions_l298_298833


namespace part1_part2_1_part2_2_l298_298356

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → (2 * a * x - Real.log x - 1) ≥ 0) ↔ a ≥ 0.5 := 
sorry

theorem part2_1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) :
  0 < a ∧ a < 1 := 
sorry

theorem part2_2 (a x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f a x1 = x1) (h3 : f a x2 = x2) (h4 : x2 ≥ 3 * x1) :
  x1 * x2 ≥ 9 / Real.exp 2 := 
sorry

end part1_part2_1_part2_2_l298_298356


namespace pizza_toppings_combination_l298_298420

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end pizza_toppings_combination_l298_298420


namespace necessary_condition_for_acute_angle_l298_298034

-- Defining vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (x - 3, 2)
def vec_b : ℝ × ℝ := (1, 1)

-- Condition for the dot product to be positive
def dot_product_positive (x : ℝ) : Prop :=
  let (ax1, ax2) := vec_a x
  let (bx1, bx2) := vec_b
  ax1 * bx1 + ax2 * bx2 > 0

-- Statement for necessary condition
theorem necessary_condition_for_acute_angle (x : ℝ) :
  (dot_product_positive x) → (1 < x) :=
sorry

end necessary_condition_for_acute_angle_l298_298034


namespace surface_area_increase_96_percent_l298_298654

variable (s : ℝ)

def original_surface_area : ℝ := 6 * s^2
def new_edge_length : ℝ := 1.4 * s
def new_surface_area : ℝ := 6 * (new_edge_length s)^2

theorem surface_area_increase_96_percent :
  (new_surface_area s - original_surface_area s) / (original_surface_area s) * 100 = 96 :=
by
  simp [original_surface_area, new_edge_length, new_surface_area]
  sorry

end surface_area_increase_96_percent_l298_298654


namespace probability_of_selection_l298_298672

/-- A school selects 80 students for a discussion from a total of 883 students. First, 3 people are eliminated using simple random sampling, and then 80 are selected from the remaining 880 using systematic sampling. Prove that the probability of each person being selected is 80/883. -/
theorem probability_of_selection (total_students : ℕ) (students_eliminated : ℕ) (students_selected : ℕ) 
  (h_total : total_students = 883) (h_eliminated : students_eliminated = 3) (h_selected : students_selected = 80) :
  ((total_students - students_eliminated) * students_selected) / (total_students * (total_students - students_eliminated)) = 80 / 883 :=
by
  sorry

end probability_of_selection_l298_298672


namespace inequality_solution_set_l298_298184

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end inequality_solution_set_l298_298184


namespace a_seq_correct_l298_298220

-- Define the sequence and the sum condition
def a_seq (n : ℕ) : ℚ := if n = 0 then 0 else (2 ^ n - 1) / 2 ^ (n - 1)

def S_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.sum (Finset.range n) a_seq)

axiom condition (n : ℕ) (hn : n > 0) : S_n n + a_seq n = 2 * n

theorem a_seq_correct (n : ℕ) (hn : n > 0) : 
  a_seq n = (2 ^ n - 1) / 2 ^ (n - 1) := sorry

end a_seq_correct_l298_298220


namespace greatest_x_lcm_105_l298_298919

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l298_298919


namespace factorize_difference_of_squares_l298_298142

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2 * x) * (3 + 2 * x) :=
sorry

end factorize_difference_of_squares_l298_298142


namespace cubic_has_three_zeros_l298_298547

theorem cubic_has_three_zeros (a : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x^3 + a * x + 2 = 0) ∧ (y^3 + a * y + 2 = 0) ∧ (z^3 + a * z + 2 = 0)) ↔ a ∈ set.Ioo (⟩ -∞) (-3) := 
sorry

end cubic_has_three_zeros_l298_298547


namespace roots_equation_1352_l298_298866

theorem roots_equation_1352 {c d : ℝ} (hc : c^2 - 6 * c + 8 = 0) (hd : d^2 - 6 * d + 8 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 1352 :=
by
  sorry

end roots_equation_1352_l298_298866


namespace inequality_proof_l298_298294

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l298_298294


namespace find_k_l298_298494

theorem find_k (k : ℕ) (h_pos : k > 0) (h_coef : 15 * k^4 < 120) : k = 1 :=
sorry

end find_k_l298_298494


namespace convex_quadrilateral_area_lt_a_sq_l298_298685

theorem convex_quadrilateral_area_lt_a_sq {a x y z t : ℝ} (hx : x < a) (hy : y < a) (hz : z < a) (ht : t < a) :
  (∃ S : ℝ, S < a^2) :=
sorry

end convex_quadrilateral_area_lt_a_sq_l298_298685


namespace sin_double_angle_identity_l298_298495

open Real 

theorem sin_double_angle_identity 
  (A : ℝ) 
  (h1 : 0 < A) 
  (h2 : A < π / 2) 
  (h3 : cos A = 3 / 5) : 
  sin (2 * A) = 24 / 25 :=
by 
  sorry

end sin_double_angle_identity_l298_298495


namespace concentration_third_flask_l298_298280

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l298_298280


namespace num_solutions_l298_298492

theorem num_solutions (k : ℤ) :
  (∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    (a^2 + b^2 = k * c * (a + b)) ∧
    (b^2 + c^2 = k * a * (b + c)) ∧
    (c^2 + a^2 = k * b * (c + a))) ↔ k = 1 ∨ k = -2 :=
sorry

end num_solutions_l298_298492


namespace range_of_a_for_three_zeros_l298_298538

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l298_298538


namespace three_right_angled_triangles_l298_298752

theorem three_right_angled_triangles 
  (a b c : ℕ)
  (h_area : 1/2 * (a * b) = 2 * (a + b + c))
  (h_pythagorean : a^2 + b^2 = c^2)
  (h_int_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (a = 9 ∧ b = 40 ∧ c = 41) ∨ 
  (a = 10 ∧ b = 24 ∧ c = 26) ∨ 
  (a = 12 ∧ b = 16 ∧ c = 20) := 
sorry

end three_right_angled_triangles_l298_298752


namespace a_minus_c_value_l298_298461

theorem a_minus_c_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := 
by 
  -- We provide the proof inline with sorry
  sorry

end a_minus_c_value_l298_298461


namespace greatest_value_of_x_l298_298933

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298933


namespace solve_eq1_solve_eq2_l298_298334

theorem solve_eq1 (x : ℝ) : (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : (x + 3)^3 = -27 ↔ x = -6 :=
by sorry

end solve_eq1_solve_eq2_l298_298334


namespace max_A_l298_298691

theorem max_A (A : ℝ) : (∀ (x y : ℕ), 0 < x → 0 < y → 3 * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) ↔ A ≤ 5 / 3 := by
  sorry

end max_A_l298_298691


namespace teamA_fraction_and_sum_l298_298310

def time_to_minutes (t : ℝ) : ℝ := t * 60

def fraction_teamA_worked (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) : Prop :=
  (90 - 60) / 150 = m / n

theorem teamA_fraction_and_sum (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) :
  90 / 150 = 1 / 5 → m + n = 6 :=
by
  sorry

end teamA_fraction_and_sum_l298_298310


namespace magnitude_of_z_l298_298354

open Complex

theorem magnitude_of_z (z : ℂ) (h : z + I = (2 + I) / I) : abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l298_298354


namespace fundraising_exceeded_goal_l298_298246

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l298_298246


namespace andrew_age_l298_298678

-- Definitions based on the conditions
variables (a g : ℝ)

-- The conditions
def condition1 : Prop := g = 9 * a
def condition2 : Prop := g - a = 63

-- The theorem we want to prove
theorem andrew_age (h1 : condition1 a g) (h2 : condition2 a g) : a = 63 / 8 :=
by
  intros
  sorry

end andrew_age_l298_298678


namespace greatest_value_of_x_l298_298934

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l298_298934


namespace result_is_21_l298_298435

theorem result_is_21 (n : ℕ) (h : n = 55) : (n / 5 + 10) = 21 :=
by
  sorry

end result_is_21_l298_298435


namespace incorrect_C_l298_298870

variable (D : ℝ → ℝ)

-- Definitions to encapsulate conditions
def range_D : Set ℝ := {0, 1}
def is_even := ∀ x, D x = D (-x)
def is_periodic := ∀ T > 0, ∃ p, ∀ x, D (x + p) = D x
def is_monotonic := ∀ x y, x < y → D x ≤ D y

-- The proof statement
theorem incorrect_C : ¬ is_periodic D :=
sorry

end incorrect_C_l298_298870


namespace value_of_x_l298_298261

-- Define the variables x, y, z
variables (x y z : ℕ)

-- Hypothesis based on the conditions of the problem
hypothesis h1 : x = y / 3
hypothesis h2 : y = z / 4
hypothesis h3 : z = 48

-- The statement to be proved
theorem value_of_x : x = 4 :=
by { sorry }

end value_of_x_l298_298261


namespace monthly_expenses_last_month_was_2888_l298_298886

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l298_298886


namespace solve_for_y_l298_298103

theorem solve_for_y (y : ℕ) (h : 9^y = 3^14) : y = 7 := 
by
  sorry

end solve_for_y_l298_298103


namespace roots_are_irrational_l298_298358

open Polynomial

theorem roots_are_irrational (k : ℝ) :
  (∃ (α β : ℝ), (x^2 - 5*k*x + (3*k^2 - 2) = 0) ∧ α * β = 9 ∧ is_root (x^2 - 5*k*x + (3*k^2 - 2)) α ∧ is_root (x^2 - 5*k*x + (3*k^2 - 2)) β) →
  (¬ is_integral α ∨ ¬ is_integral β) ∧ (¬ (¬ is_integral α ∧ ¬ is_integral β)) :=
sorry

end roots_are_irrational_l298_298358


namespace min_value_geq_4_plus_2sqrt2_l298_298348

theorem min_value_geq_4_plus_2sqrt2
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 1)
  (h4: a + b = 1) :
  ( ( (a^2 + 1) / (a * b) - 2 ) * c + (Real.sqrt 2) / (c - 1) ) ≥ (4 + 2 * (Real.sqrt 2)) :=
sorry

end min_value_geq_4_plus_2sqrt2_l298_298348


namespace prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l298_298831

noncomputable def total_outcomes := 24
noncomputable def outcomes_two_correct := 6
noncomputable def outcomes_at_least_two_correct := 7
noncomputable def outcomes_all_incorrect := 9

theorem prob_two_correct : (outcomes_two_correct : ℚ) / total_outcomes = 1 / 4 := by
  sorry

theorem prob_at_least_two_correct : (outcomes_at_least_two_correct : ℚ) / total_outcomes = 7 / 24 := by
  sorry

theorem prob_all_incorrect : (outcomes_all_incorrect : ℚ) / total_outcomes = 3 / 8 := by
  sorry

end prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l298_298831
