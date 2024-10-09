import Mathlib

namespace Billy_current_age_l345_34517

variable (B : ℕ)

theorem Billy_current_age 
  (h1 : ∃ B, 4 * B - B = 12) : B = 4 := by
  sorry

end Billy_current_age_l345_34517


namespace solve_system_eq_l345_34579

theorem solve_system_eq (a b c x y z : ℝ) (h1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (h2 : x / a + y / b + z / c = a + b + c) (h3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c :=
by
  sorry

end solve_system_eq_l345_34579


namespace remainder_276_l345_34501

theorem remainder_276 (y : ℤ) (k : ℤ) (hk : y = 23 * k + 19) : y % 276 = 180 :=
sorry

end remainder_276_l345_34501


namespace paint_for_smaller_statues_l345_34500

open Real

theorem paint_for_smaller_statues :
  ∀ (paint_needed : ℝ) (height_big_statue height_small_statue : ℝ) (num_small_statues : ℝ),
  height_big_statue = 10 → height_small_statue = 2 → paint_needed = 5 → num_small_statues = 200 →
  (paint_needed / (height_big_statue / height_small_statue) ^ 2) * num_small_statues = 40 :=
by
  intros paint_needed height_big_statue height_small_statue num_small_statues
  intros h_big_height h_small_height h_paint_needed h_num_small
  rw [h_big_height, h_small_height, h_paint_needed, h_num_small]
  sorry

end paint_for_smaller_statues_l345_34500


namespace percentage_number_l345_34506

theorem percentage_number (b : ℕ) (h : b = 100) : (320 * b / 100) = 320 :=
by
  sorry

end percentage_number_l345_34506


namespace butterfinger_count_l345_34561

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l345_34561


namespace largest_even_number_l345_34520

theorem largest_even_number (x : ℤ) (h1 : 3 * x + 6 = (x + (x + 2) + (x + 4)) / 3 + 44) : 
  x + 4 = 24 := 
by 
  sorry

end largest_even_number_l345_34520


namespace fraction_of_male_birds_l345_34599

theorem fraction_of_male_birds (T : ℕ) (h_cond1 : T ≠ 0) :
  let robins := (2 / 5) * T
  let bluejays := T - robins
  let male_robins := (2 / 3) * robins
  let male_bluejays := (1 / 3) * bluejays
  (male_robins + male_bluejays) / T = 7 / 15 :=
by 
  sorry

end fraction_of_male_birds_l345_34599


namespace canoe_total_weight_calculation_canoe_maximum_weight_limit_l345_34542

def canoe_max_people : ℕ := 8
def people_with_pets_ratio : ℚ := 3 / 4
def adult_weight : ℚ := 150
def child_weight : ℚ := adult_weight / 2
def dog_weight : ℚ := adult_weight / 3
def cat1_weight : ℚ := adult_weight / 10
def cat2_weight : ℚ := adult_weight / 8

def canoe_capacity_with_pets : ℚ := people_with_pets_ratio * canoe_max_people

def total_weight_adults_and_children : ℚ := 4 * adult_weight + 2 * child_weight
def total_weight_pets : ℚ := dog_weight + cat1_weight + cat2_weight
def total_weight : ℚ := total_weight_adults_and_children + total_weight_pets

def max_weight_limit : ℚ := canoe_max_people * adult_weight

theorem canoe_total_weight_calculation :
  total_weight = 833 + 3 / 4 := by
  sorry

theorem canoe_maximum_weight_limit :
  max_weight_limit = 1200 := by
  sorry

end canoe_total_weight_calculation_canoe_maximum_weight_limit_l345_34542


namespace first_reduction_percentage_l345_34576

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.70 = P * 0.525 ↔ x = 25 := by
  sorry

end first_reduction_percentage_l345_34576


namespace vectors_coplanar_l345_34516

def vector3 := ℝ × ℝ × ℝ

def scalar_triple_product (a b c : vector3) : ℝ :=
  match a, b, c with
  | (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) =>
    a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product (-3, 3, 3) (-4, 7, 6) (3, 0, -1) = 0 :=
by
  sorry

end vectors_coplanar_l345_34516


namespace arithmetic_sequence_difference_l345_34588

theorem arithmetic_sequence_difference :
  ∀ (a d : ℤ), a = -2 → d = 7 →
  |(a + (3010 - 1) * d) - (a + (3000 - 1) * d)| = 70 :=
by
  intros a d a_def d_def
  rw [a_def, d_def]
  sorry

end arithmetic_sequence_difference_l345_34588


namespace length_PC_l345_34528

-- Define lengths of the sides of triangle ABC.
def AB := 10
def BC := 8
def CA := 7

-- Define the similarity condition
def similar_triangles (PA PC : ℝ) : Prop :=
  PA / PC = AB / CA

-- Define the extension of side BC to point P
def extension_condition (PA PC : ℝ) : Prop :=
  PA = PC + BC

theorem length_PC (PC : ℝ) (PA : ℝ) :
  similar_triangles PA PC → extension_condition PA PC → PC = 56 / 3 :=
by
  intro h_sim h_ext
  sorry

end length_PC_l345_34528


namespace quad_eq_pos_neg_root_l345_34546

theorem quad_eq_pos_neg_root (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 2 ∧ x₁ * x₂ = a + 1) ↔ a < -1 :=
by sorry

end quad_eq_pos_neg_root_l345_34546


namespace algebraic_expression_value_l345_34529

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) :
  x^2 - 4 * y^2 + 1 = -3 := by
  sorry

end algebraic_expression_value_l345_34529


namespace divisors_of_30_l345_34554

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l345_34554


namespace athlete_runs_entire_track_in_44_seconds_l345_34571

noncomputable def time_to_complete_track (flags : ℕ) (time_to_4th_flag : ℕ) : ℕ :=
  let distances_between_flags := flags - 1
  let distances_to_4th_flag := 4 - 1
  let time_per_distance := time_to_4th_flag / distances_to_4th_flag
  distances_between_flags * time_per_distance

theorem athlete_runs_entire_track_in_44_seconds :
  time_to_complete_track 12 12 = 44 :=
by
  sorry

end athlete_runs_entire_track_in_44_seconds_l345_34571


namespace determine_function_l345_34564

theorem determine_function (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 := 
sorry

end determine_function_l345_34564


namespace min_ab_l345_34521

theorem min_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 1 / b = 1) : ab = 4 :=
  sorry

end min_ab_l345_34521


namespace arithmetic_sequence_common_difference_l345_34548

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l345_34548


namespace flowers_total_l345_34586

def red_roses := 1491
def yellow_carnations := 3025
def white_roses := 1768
def purple_tulips := 2150
def pink_daisies := 3500
def blue_irises := 2973
def orange_marigolds := 4234
def lavender_orchids := 350
def sunflowers := 815
def violet_lilies := 26

theorem flowers_total :
  red_roses +
  yellow_carnations +
  white_roses +
  purple_tulips +
  pink_daisies +
  blue_irises +
  orange_marigolds +
  lavender_orchids +
  sunflowers +
  violet_lilies = 21332 := 
by
  -- Simplify and add up all given numbers
  sorry

end flowers_total_l345_34586


namespace range_of_a_l345_34512

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x = 1 → ¬ ((x + 1) / (x + a) < 2))) ↔ -1 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l345_34512


namespace stadium_length_in_yards_l345_34505

theorem stadium_length_in_yards (length_in_feet : ℕ) (conversion_factor : ℕ) : ℕ :=
    length_in_feet / conversion_factor

example : stadium_length_in_yards 240 3 = 80 :=
by sorry

end stadium_length_in_yards_l345_34505


namespace probability_longer_piece_l345_34502

theorem probability_longer_piece {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) :
  (∃ (p : ℝ), p = 2 / (x * y + 1)) :=
by
  sorry

end probability_longer_piece_l345_34502


namespace remainder_when_N_divided_by_1000_l345_34570

def number_of_factors_of_5 (n : Nat) : Nat :=
  if n = 0 then 0 
  else n / 5 + number_of_factors_of_5 (n / 5)

def total_factors_of_5_upto (n : Nat) : Nat := 
  match n with
  | 0 => 0
  | n + 1 => number_of_factors_of_5 (n + 1) + total_factors_of_5_upto n

def product_factorial_5s : Nat := total_factors_of_5_upto 100

def N : Nat := product_factorial_5s

theorem remainder_when_N_divided_by_1000 : N % 1000 = 124 := by
  sorry

end remainder_when_N_divided_by_1000_l345_34570


namespace haley_picked_carrots_l345_34565

variable (H : ℕ)
variable (mom_carrots : ℕ := 38)
variable (good_carrots : ℕ := 64)
variable (bad_carrots : ℕ := 13)
variable (total_carrots : ℕ := good_carrots + bad_carrots)

theorem haley_picked_carrots : H + mom_carrots = total_carrots → H = 39 := by
  sorry

end haley_picked_carrots_l345_34565


namespace remainder_45_to_15_l345_34536

theorem remainder_45_to_15 : ∀ (N : ℤ) (k : ℤ), N = 45 * k + 31 → N % 15 = 1 :=
by
  intros N k h
  sorry

end remainder_45_to_15_l345_34536


namespace angle_is_50_l345_34509

-- Define the angle, supplement, and complement
def angle (x : ℝ) := x
def supplement (x : ℝ) := 180 - x
def complement (x : ℝ) := 90 - x
def condition (x : ℝ) := supplement x = 3 * (complement x) + 10

theorem angle_is_50 :
  ∃ x : ℝ, condition x ∧ x = 50 :=
by
  -- Here we show the existence of x that satisfies the condition and is equal to 50
  sorry

end angle_is_50_l345_34509


namespace dodecagon_enclosure_l345_34587

theorem dodecagon_enclosure (m n : ℕ) (h1 : m = 12) 
  (h2 : ∀ (x : ℕ), x ∈ { k | ∃ p : ℕ, p = n ∧ 12 = k * p}) :
  n = 12 :=
by
  -- begin proof steps here
sorry

end dodecagon_enclosure_l345_34587


namespace Josh_marbles_count_l345_34567

-- Definitions of the given conditions
def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

-- The statement we aim to prove
theorem Josh_marbles_count : (initial_marbles - lost_marbles) = 9 :=
by
  -- Skipping the proof with sorry
  sorry

end Josh_marbles_count_l345_34567


namespace time_in_still_water_l345_34547

-- Define the conditions
variable (S x y : ℝ)
axiom condition1 : S / (x + y) = 6
axiom condition2 : S / (x - y) = 8

-- Define the proof statement
theorem time_in_still_water : S / x = 48 / 7 :=
by
  -- The proof is omitted
  sorry

end time_in_still_water_l345_34547


namespace negation_red_cards_in_deck_l345_34573

variable (Deck : Type) (is_red : Deck → Prop) (is_in_deck : Deck → Prop)

theorem negation_red_cards_in_deck :
  (¬ ∃ x : Deck, is_red x ∧ is_in_deck x) ↔ (∃ x : Deck, is_red x ∧ is_in_deck x) :=
by {
  sorry
}

end negation_red_cards_in_deck_l345_34573


namespace shingle_area_l345_34590

-- Definitions from conditions
def length := 10 -- uncut side length in inches
def width := 7   -- uncut side width in inches
def trapezoid_base1 := 6 -- base of the trapezoid in inches
def trapezoid_height := 2 -- height of the trapezoid in inches

-- Definition derived from conditions
def trapezoid_base2 := length - trapezoid_base1 -- the second base of the trapezoid

-- Required proof in Lean
theorem shingle_area : (length * width - (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height)) = 60 := 
by
  sorry

end shingle_area_l345_34590


namespace model_A_selected_count_l345_34527

def production_A := 1200
def production_B := 6000
def production_C := 2000
def total_selected := 46

def total_production := production_A + production_B + production_C

theorem model_A_selected_count :
  (production_A / total_production) * total_selected = 6 := by
  sorry

end model_A_selected_count_l345_34527


namespace most_and_least_l345_34575

variables {Jan Kim Lee Ron Zay : ℝ}

-- Conditions as hypotheses
axiom H1 : Lee < Jan
axiom H2 : Kim < Jan
axiom H3 : Zay < Ron
axiom H4 : Zay < Lee
axiom H5 : Zay < Jan
axiom H6 : Jan < Ron

theorem most_and_least :
  (Ron > Jan) ∧ (Ron > Kim) ∧ (Ron > Lee) ∧ (Ron > Zay) ∧ 
  (Zay < Jan) ∧ (Zay < Kim) ∧ (Zay < Lee) ∧ (Zay < Ron) :=
by {
  -- Proof is omitted
  sorry
}

end most_and_least_l345_34575


namespace find_digit_D_l345_34518

theorem find_digit_D (A B C D : ℕ)
  (h_add : 100 + 10 * A + B + 100 * C + 10 * A + A = 100 * D + 10 * A + B)
  (h_sub : 100 + 10 * A + B - (100 * C + 10 * A + A) = 100 + 10 * A) :
  D = 1 :=
by
  -- Since we're skipping the proof and focusing on the statement only
  sorry

end find_digit_D_l345_34518


namespace derivative_of_f_l345_34562

noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_f :
  (deriv f) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end derivative_of_f_l345_34562


namespace children_count_l345_34572

theorem children_count (C : ℕ) 
    (cons : ℕ := 12)
    (total_cost : ℕ := 76)
    (child_ticket_cost : ℕ := 7)
    (adult_ticket_cost : ℕ := 10)
    (num_adults : ℕ := 5)
    (adult_cost := num_adults * adult_ticket_cost)
    (cost_with_concessions := total_cost - adult_cost )
    (children_cost := cost_with_concessions - cons):
    C = children_cost / child_ticket_cost :=
by
    sorry

end children_count_l345_34572


namespace subtraction_property_l345_34558

theorem subtraction_property : (12.56 - (5.56 - 2.63)) = (12.56 - 5.56 + 2.63) := 
by 
  sorry

end subtraction_property_l345_34558


namespace negation_of_p_is_universal_l345_34551

-- Define the proposition p
def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- The proof statement for the negation of p
theorem negation_of_p_is_universal : ¬p ↔ ∀ x : ℝ, Real.exp x - x - 1 > 0 :=
by sorry

end negation_of_p_is_universal_l345_34551


namespace quadratic_inequality_solution_l345_34577

theorem quadratic_inequality_solution 
  (x : ℝ) (b c : ℝ)
  (h : ∀ x, -x^2 + b*x + c < 0 ↔ x < -3 ∨ x > 2) :
  (6 * x^2 + x - 1 > 0) ↔ (x < -1/2 ∨ x > 1/3) := 
sorry

end quadratic_inequality_solution_l345_34577


namespace total_quantities_l345_34511

theorem total_quantities (n S S₃ S₂ : ℕ) (h₁ : S = 6 * n) (h₂ : S₃ = 4 * 3) (h₃ : S₂ = 33 * 2) (h₄ : S = S₃ + S₂) : n = 13 :=
by
  sorry

end total_quantities_l345_34511


namespace equation1_solutions_equation2_solutions_l345_34523

theorem equation1_solutions (x : ℝ) :
  x ^ 2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := by
  sorry

theorem equation2_solutions (x : ℝ) :
  2 * x ^ 2 - 2 * x = 1 ↔ x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2 := by
  sorry

end equation1_solutions_equation2_solutions_l345_34523


namespace odd_sol_exists_l345_34557

theorem odd_sol_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x_n y_n : ℕ), (x_n % 2 = 1) ∧ (y_n % 2 = 1) ∧ (x_n^2 + 7 * y_n^2 = 2^n) := 
sorry

end odd_sol_exists_l345_34557


namespace sum_of_ages_l345_34560

theorem sum_of_ages (a b c : ℕ) 
  (h1 : a = 18 + b + c) 
  (h2 : a^2 = 2016 + (b + c)^2) : 
  a + b + c = 112 := 
sorry

end sum_of_ages_l345_34560


namespace power_expansion_l345_34563

theorem power_expansion (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := 
by 
  sorry

end power_expansion_l345_34563


namespace max_diff_consecutive_slightly_unlucky_l345_34540

def is_slightly_unlucky (n : ℕ) : Prop := (n.digits 10).sum % 13 = 0

theorem max_diff_consecutive_slightly_unlucky :
  ∃ n m : ℕ, is_slightly_unlucky n ∧ is_slightly_unlucky m ∧ (m > n) ∧ ∀ k, (is_slightly_unlucky k ∧ k > n ∧ k < m) → false → (m - n) = 79 :=
sorry

end max_diff_consecutive_slightly_unlucky_l345_34540


namespace counterexample_not_prime_implies_prime_l345_34504

theorem counterexample_not_prime_implies_prime (n : ℕ) (h₁ : ¬Nat.Prime n) (h₂ : n = 27) : ¬Nat.Prime (n - 2) :=
by
  sorry

end counterexample_not_prime_implies_prime_l345_34504


namespace gcd_difference_5610_210_10_l345_34514

theorem gcd_difference_5610_210_10 : Int.gcd 5610 210 - 10 = 20 := by
  sorry

end gcd_difference_5610_210_10_l345_34514


namespace root_interval_sum_l345_34592

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 8

def has_root_in_interval (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  a < b ∧ b - a = 1 ∧ f a < 0 ∧ f b > 0

theorem root_interval_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : has_root_in_interval a b h1 h2) : 
  a + b = 5 :=
sorry

end root_interval_sum_l345_34592


namespace number_of_oxygen_atoms_l345_34534

theorem number_of_oxygen_atoms 
  (M_weight : ℝ)
  (H_weight : ℝ)
  (Cl_weight : ℝ)
  (O_weight : ℝ)
  (MW_formula : M_weight = H_weight + Cl_weight + n * O_weight)
  (M_weight_eq : M_weight = 68)
  (H_weight_eq : H_weight = 1)
  (Cl_weight_eq : Cl_weight = 35.5)
  (O_weight_eq : O_weight = 16)
  : n = 2 := 
  by sorry

end number_of_oxygen_atoms_l345_34534


namespace f2011_eq_two_l345_34522

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity_eqn : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom f1_eq_two : f 1 = 2

theorem f2011_eq_two : f 2011 = 2 := 
by 
  sorry

end f2011_eq_two_l345_34522


namespace oblique_area_l345_34526

theorem oblique_area (side_length : ℝ) (A_ratio : ℝ) (S_original : ℝ) (S_oblique : ℝ) 
  (h1 : side_length = 1) 
  (h2 : A_ratio = (Real.sqrt 2) / 4) 
  (h3 : S_original = side_length ^ 2) 
  (h4 : S_oblique / S_original = A_ratio) : 
  S_oblique = (Real.sqrt 2) / 4 :=
by 
  sorry

end oblique_area_l345_34526


namespace green_balloons_count_l345_34524

-- Define the conditions
def total_balloons : Nat := 50
def red_balloons : Nat := 12
def blue_balloons : Nat := 7

-- Define the proof problem
theorem green_balloons_count : 
  let green_balloons := total_balloons - (red_balloons + blue_balloons)
  green_balloons = 31 :=
by
  sorry

end green_balloons_count_l345_34524


namespace not_even_nor_odd_l345_34555

def f (x : ℝ) : ℝ := x^2

theorem not_even_nor_odd (x : ℝ) (h₁ : -1 < x) (h₂ : x ≤ 1) : ¬(∀ y, f y = f (-y)) ∧ ¬(∀ y, f y = -f (-y)) :=
by
  sorry

end not_even_nor_odd_l345_34555


namespace smallest_N_divisible_l345_34519

theorem smallest_N_divisible (N x : ℕ) (H: N - 24 = 84 * Nat.lcm x 60) : N = 5064 :=
by
  sorry

end smallest_N_divisible_l345_34519


namespace average_temperature_l345_34596

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l345_34596


namespace max_sum_abc_min_sum_reciprocal_l345_34594

open Real

variables {a b c : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 2)

-- Maximum of a + b + c
theorem max_sum_abc : a + b + c ≤ sqrt 6 :=
by sorry

-- Minimum of 1/(a + b) + 1/(b + c) + 1/(c + a)
theorem min_sum_reciprocal : (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 * sqrt 6 / 4 :=
by sorry

end max_sum_abc_min_sum_reciprocal_l345_34594


namespace sufficient_but_not_necessary_condition_l345_34544

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l345_34544


namespace length_of_field_l345_34569

variable (l w : ℝ)

theorem length_of_field : 
  (l = 2 * w) ∧ (8 * 8 = 64) ∧ ((8 * 8) = (1 / 50) * l * w) → l = 80 :=
by
  sorry

end length_of_field_l345_34569


namespace correct_conclusion_l345_34543

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * 2^n

theorem correct_conclusion (n : ℕ) (h₁ : ∀ k : ℕ, k > 0 → a_n (k + 1) - 2 * a_n k = 2^(k + 1)) :
  a_n n = n * 2 ^ n :=
by
  sorry

end correct_conclusion_l345_34543


namespace units_digit_2009_2008_plus_2013_l345_34574

theorem units_digit_2009_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 :=
by
  sorry

end units_digit_2009_2008_plus_2013_l345_34574


namespace solutions_to_equation_l345_34532

theorem solutions_to_equation :
  ∀ x : ℝ, (x + 1) * (x - 2) = x + 1 ↔ x = -1 ∨ x = 3 :=
by
  sorry

end solutions_to_equation_l345_34532


namespace negation_of_universal_statement_l345_34578

def P (x : ℝ) : Prop := x^3 - x^2 + 1 ≤ 0

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by {
  sorry
}

end negation_of_universal_statement_l345_34578


namespace smallest_a1_l345_34537

noncomputable def is_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 7 * a (n - 1) - 2 * n

noncomputable def is_positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 0, a n > 0

theorem smallest_a1 (a : ℕ → ℝ)
  (h_seq : is_sequence a)
  (h_pos : is_positive_sequence a) :
  a 1 ≥ 13 / 18 :=
sorry

end smallest_a1_l345_34537


namespace problem_I_problem_II_l345_34545

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end problem_I_problem_II_l345_34545


namespace sum_of_integers_l345_34530

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l345_34530


namespace equilateral_triangle_side_length_l345_34593

noncomputable def side_length (a : ℝ) := if a = 0 then 0 else (a : ℝ) * (3 : ℝ) / 2

theorem equilateral_triangle_side_length
  (a : ℝ)
  (h1 : a ≠ 0)
  (A := (a, - (1 / 3) * a^2))
  (B := (-a, - (1 / 3) * a^2))
  (Habo : (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2) :
  ∃ s : ℝ, s = 9 / 2 :=
by
  sorry

end equilateral_triangle_side_length_l345_34593


namespace tiffany_bags_on_monday_l345_34508

theorem tiffany_bags_on_monday : 
  ∃ M : ℕ, M = 8 ∧ ∃ T : ℕ, T = 7 ∧ M = T + 1 :=
by
  sorry

end tiffany_bags_on_monday_l345_34508


namespace find_ab_solutions_l345_34556

theorem find_ab_solutions (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a + 1) ∣ (a ^ 3 * b - 1))
  (h2 : (b - 1) ∣ (b ^ 3 * a + 1)) : 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) :=
sorry

end find_ab_solutions_l345_34556


namespace female_democrats_count_l345_34585

theorem female_democrats_count (F M : ℕ) (h1 : F + M = 750) 
  (h2 : F / 2 ≠ 0) (h3 : M / 4 ≠ 0) 
  (h4 : F / 2 + M / 4 = 750 / 3) : F / 2 = 125 :=
by
  sorry

end female_democrats_count_l345_34585


namespace radius_of_circle_with_tangent_parabolas_l345_34597

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end radius_of_circle_with_tangent_parabolas_l345_34597


namespace complement_intersection_l345_34515

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {3} :=
sorry

end complement_intersection_l345_34515


namespace volume_comparison_l345_34559

-- Define the properties for the cube and the cuboid.
def cube_side_length : ℕ := 1 -- in meters
def cuboid_width : ℕ := 50  -- in centimeters
def cuboid_length : ℕ := 50 -- in centimeters
def cuboid_height : ℕ := 20 -- in centimeters

-- Convert cube side length to centimeters.
def cube_side_length_cm := cube_side_length * 100 -- in centimeters

-- Calculate volumes.
def cube_volume : ℕ := cube_side_length_cm ^ 3 -- in cubic centimeters
def cuboid_volume : ℕ := cuboid_width * cuboid_length * cuboid_height -- in cubic centimeters

-- The theorem stating the problem.
theorem volume_comparison : cube_volume / cuboid_volume = 20 :=
by sorry

end volume_comparison_l345_34559


namespace problem_proof_l345_34582

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

def num_multiples_of_lt (m bound : ℕ) : ℕ :=
  (bound - 1) / m

-- Definitions for the conditions
def a := num_multiples_of_lt 8 40
def b := num_multiples_of_lt 8 40

-- Proof statement
theorem problem_proof : (a - b)^3 = 0 := by
  sorry

end problem_proof_l345_34582


namespace differential_savings_is_4830_l345_34550

-- Defining the conditions
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.28
def annual_income : ℝ := 34500

-- Defining the calculation of tax before and after the tax rate change
def tax_before : ℝ := annual_income * initial_tax_rate
def tax_after : ℝ := annual_income * new_tax_rate

-- Defining the differential savings
def differential_savings : ℝ := tax_before - tax_after

-- Statement asserting that the differential savings is $4830
theorem differential_savings_is_4830 : differential_savings = 4830 := by sorry

end differential_savings_is_4830_l345_34550


namespace four_dice_min_rolls_l345_34581

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l345_34581


namespace smallest_value_of_3a_plus_2_l345_34566

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  3 * a + 2 = 1 / 2 :=
sorry

end smallest_value_of_3a_plus_2_l345_34566


namespace nominal_rate_of_interest_annual_l345_34510

theorem nominal_rate_of_interest_annual (EAR nominal_rate : ℝ) (n : ℕ) (h1 : EAR = 0.0816) (h2 : n = 2) : 
  nominal_rate = 0.0796 :=
by 
  sorry

end nominal_rate_of_interest_annual_l345_34510


namespace circle_area_l345_34513

theorem circle_area (C : ℝ) (hC : C = 31.4) : 
  ∃ (A : ℝ), A = 246.49 / π := 
by
  sorry -- proof not required

end circle_area_l345_34513


namespace find_d_minus_c_l345_34591

noncomputable def point_transformed (c d : ℝ) : Prop :=
  let Q := (c, d)
  let R := (2 * 2 - c, 2 * 3 - d)  -- Rotating Q by 180º about (2, 3)
  let S := (d, c)                -- Reflecting Q about the line y = x
  (S.1, S.2) = (2, -1)           -- Result is (2, -1)

theorem find_d_minus_c (c d : ℝ) (h : point_transformed c d) : d - c = -1 :=
by {
  sorry
}

end find_d_minus_c_l345_34591


namespace five_ones_make_100_l345_34535

noncomputable def concatenate (a b c : Nat) : Nat :=
  a * 100 + b * 10 + c

theorem five_ones_make_100 :
  let one := 1
  let x := concatenate one one one -- 111
  let y := concatenate one one 0 / 10 -- 11, concatenation of 1 and 1 treated as 110, divided by 10
  x - y = 100 :=
by
  sorry

end five_ones_make_100_l345_34535


namespace complement_of_M_in_U_l345_34533

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l345_34533


namespace speed_of_sound_l345_34583

theorem speed_of_sound (time_blasts : ℝ) (distance_traveled : ℝ) (time_heard : ℝ) (speed : ℝ) 
  (h_blasts : time_blasts = 30 * 60) -- time between the two blasts in seconds 
  (h_distance : distance_traveled = 8250) -- distance in meters
  (h_heard : time_heard = 30 * 60 + 25) -- time when man heard the second blast
  (h_relationship : speed = distance_traveled / (time_heard - time_blasts)) : 
  speed = 330 :=
sorry

end speed_of_sound_l345_34583


namespace find_side_b_l345_34549

theorem find_side_b
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2 / 3) :
  b = Real.sqrt 6 :=
by
  sorry

end find_side_b_l345_34549


namespace binom_factorial_eq_120_factorial_l345_34538

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_factorial_eq_120_factorial : (factorial (binomial 10 3)) = factorial 120 := by
  sorry

end binom_factorial_eq_120_factorial_l345_34538


namespace common_ratio_geometric_sequence_l345_34541

variables {a : ℕ → ℝ} -- 'a' is a sequence of positive real numbers
variable {q : ℝ} -- 'q' is the common ratio of the geometric sequence

-- Definition of a geometric sequence with common ratio 'q'
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition from the problem statement
def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  2 * a 5 - 3 * a 4 = 2 * a 3

-- Main theorem: If the sequence {a_n} is a geometric sequence with positive terms and satisfies the condition, 
-- then the common ratio q = 2
theorem common_ratio_geometric_sequence :
  (∀ n, 0 < a n) → geometric_sequence a q → condition a q → q = 2 :=
by
  intro h_pos h_geom h_cond
  sorry

end common_ratio_geometric_sequence_l345_34541


namespace train_usual_time_l345_34552

theorem train_usual_time (S T_new T : ℝ) (h_speed : T_new = 7 / 6 * T) (h_delay : T_new = T + 1 / 6) : T = 1 := by
  sorry

end train_usual_time_l345_34552


namespace find_t_l345_34595

variables {t : ℝ}

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (t : ℝ) : ℝ × ℝ := (-2, t)

def are_parallel (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

theorem find_t (h : are_parallel vector_a (vector_b t)) : t = -4 :=
by sorry

end find_t_l345_34595


namespace average_marks_l345_34553

-- Define the conditions
variables (M P C : ℕ)
axiom condition1 : M + P = 30
axiom condition2 : C = P + 20

-- Define the target statement
theorem average_marks : (M + C) / 2 = 25 :=
by
  sorry

end average_marks_l345_34553


namespace quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l345_34589

structure Point where
  x : ℚ
  y : ℚ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 6, y := 1 }

def line_eq_y_eq_kx_plus_b (k b x : ℚ) : ℚ := k * x + b

def intersects (A : Point) (P : Point × Point) (x y : ℚ) : Prop :=
  ∃ k b, P.1.y = line_eq_y_eq_kx_plus_b k b P.1.x ∧ P.2.y = line_eq_y_eq_kx_plus_b k b P.2.x ∧
         y = line_eq_y_eq_kx_plus_b k b x

theorem quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176 :
  ∃ (p q r s : ℚ), 
    gcd p q = 1 ∧ gcd r s = 1 ∧ intersects A (C, D) (p / q) (r / s) ∧
    (p + q + r + s = 176) :=
sorry

end quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l345_34589


namespace rope_fold_length_l345_34525

theorem rope_fold_length (L : ℝ) (hL : L = 1) :
  (L / 2 / 2 / 2) = (1 / 8) :=
by
  -- proof steps here
  sorry

end rope_fold_length_l345_34525


namespace bridge_extension_length_l345_34503

theorem bridge_extension_length (width_of_river length_of_existing_bridge additional_length_needed : ℕ)
  (h1 : width_of_river = 487)
  (h2 : length_of_existing_bridge = 295)
  (h3 : additional_length_needed = width_of_river - length_of_existing_bridge) :
  additional_length_needed = 192 :=
by {
  -- The steps of the proof would go here, but we use sorry for now.
  sorry
}

end bridge_extension_length_l345_34503


namespace plates_arrangement_l345_34584

theorem plates_arrangement :
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  non_adjacent_green_arrangements = 588 :=
by
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  sorry

end plates_arrangement_l345_34584


namespace smallest_sum_infinite_geometric_progression_l345_34580

theorem smallest_sum_infinite_geometric_progression :
  ∃ (a q A : ℝ), (a * q = 3) ∧ (0 < q) ∧ (q < 1) ∧ (A = a / (1 - q)) ∧ (A = 12) :=
by
  sorry

end smallest_sum_infinite_geometric_progression_l345_34580


namespace least_integer_sum_of_primes_l345_34598

-- Define what it means to be prime and greater than a number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greater_than_ten (n : ℕ) : Prop := n > 10

-- Main theorem statement
theorem least_integer_sum_of_primes :
  ∃ n, (∀ p1 p2 p3 p4 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
                        greater_than_ten p1 ∧ greater_than_ten p2 ∧ greater_than_ten p3 ∧ greater_than_ten p4 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
                        n = p1 + p2 + p3 + p4 → n ≥ 60) ∧
        n = 60 :=
  sorry

end least_integer_sum_of_primes_l345_34598


namespace pascal_sixth_element_row_20_l345_34531

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
by 
  sorry

end pascal_sixth_element_row_20_l345_34531


namespace train_pass_station_time_l345_34539

-- Define the lengths of the train and station
def length_train : ℕ := 250
def length_station : ℕ := 200

-- Define the speed of the train in km/hour
def speed_kmh : ℕ := 36

-- Convert the speed to meters per second
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance the train needs to cover
def total_distance : ℕ := length_train + length_station

-- Define the expected time to pass the station
def expected_time : ℕ := 45

-- State the theorem that needs to be proven
theorem train_pass_station_time :
  total_distance / speed_mps = expected_time := by
  sorry

end train_pass_station_time_l345_34539


namespace third_year_students_sampled_correct_l345_34507

-- The given conditions
def first_year_students := 700
def second_year_students := 670
def third_year_students := 630
def total_samples := 200
def total_students := first_year_students + second_year_students + third_year_students

-- The proportion of third-year students
def third_year_proportion := third_year_students / total_students

-- The number of third-year students to be selected
def samples_third_year := total_samples * third_year_proportion

theorem third_year_students_sampled_correct :
  samples_third_year = 63 :=
by
  -- We skip the actual proof for this statement with sorry
  sorry

end third_year_students_sampled_correct_l345_34507


namespace apples_initial_count_l345_34568

theorem apples_initial_count 
  (trees : ℕ)
  (apples_per_tree_picked : ℕ)
  (apples_picked_in_total : ℕ)
  (apples_remaining : ℕ)
  (initial_apples : ℕ) 
  (h1 : trees = 3) 
  (h2 : apples_per_tree_picked = 8) 
  (h3 : apples_picked_in_total = trees * apples_per_tree_picked)
  (h4 : apples_remaining = 9) 
  (h5 : initial_apples = apples_picked_in_total + apples_remaining) : 
  initial_apples = 33 :=
by sorry

end apples_initial_count_l345_34568
