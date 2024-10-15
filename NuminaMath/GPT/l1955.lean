import Mathlib

namespace NUMINAMATH_GPT_outer_circle_radius_l1955_195525

theorem outer_circle_radius (r R : ℝ) (hr : r = 4)
  (radius_increase : ∀ R, R' = 1.5 * R)
  (radius_decrease : ∀ r, r' = 0.75 * r)
  (area_increase : ∀ (A1 A2 : ℝ), A2 = 3.6 * A1)
  (initial_area : ∀ A1, A1 = π * R^2 - π * r^2)
  (new_area : ∀ A2 R' r', A2 = π * R'^2 - π * r'^2) :
  R = 6 := sorry

end NUMINAMATH_GPT_outer_circle_radius_l1955_195525


namespace NUMINAMATH_GPT_darnel_jog_laps_l1955_195561

theorem darnel_jog_laps (x : ℝ) (h1 : 0.88 = x + 0.13) : x = 0.75 := by
  sorry

end NUMINAMATH_GPT_darnel_jog_laps_l1955_195561


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1955_195502

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1955_195502


namespace NUMINAMATH_GPT_mean_height_of_players_l1955_195528

def heights_50s : List ℕ := [57, 59]
def heights_60s : List ℕ := [62, 64, 64, 65, 65, 68, 69]
def heights_70s : List ℕ := [70, 71, 73, 75, 75, 77, 78]

def all_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s

def mean_height (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem mean_height_of_players :
  mean_height all_heights = 68.25 :=
by
  sorry

end NUMINAMATH_GPT_mean_height_of_players_l1955_195528


namespace NUMINAMATH_GPT_find_percentage_l1955_195593

theorem find_percentage (P : ℝ) : 
  (P / 100) * 700 = 210 ↔ P = 30 := by
  sorry

end NUMINAMATH_GPT_find_percentage_l1955_195593


namespace NUMINAMATH_GPT_greatest_int_radius_lt_75pi_l1955_195582

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end NUMINAMATH_GPT_greatest_int_radius_lt_75pi_l1955_195582


namespace NUMINAMATH_GPT_rooks_placement_possible_l1955_195595

/-- 
  It is possible to place 8 rooks on a chessboard such that they do not attack each other
  and each rook stands on cells of different colors, given that the chessboard is divided 
  into 32 colors with exactly two cells of each color.
-/
theorem rooks_placement_possible :
  ∃ (placement : Fin 8 → Fin 8 × Fin 8),
    (∀ i j, i ≠ j → (placement i).fst ≠ (placement j).fst ∧ (placement i).snd ≠ (placement j).snd) ∧
    (∀ i j, i ≠ j → (placement i ≠ placement j)) ∧
    (∀ c : Fin 32, ∃! p1 p2, placement p1 = placement p2 ∧ (placement p1).fst ≠ (placement p2).fst 
                        ∧ (placement p1).snd ≠ (placement p2).snd) :=
by
  sorry

end NUMINAMATH_GPT_rooks_placement_possible_l1955_195595


namespace NUMINAMATH_GPT_left_building_percentage_l1955_195568

theorem left_building_percentage (L R : ℝ)
  (middle_building_height : ℝ := 100)
  (total_height : ℝ := 340)
  (condition1 : L + middle_building_height + R = total_height)
  (condition2 : R = L + middle_building_height - 20) :
  (L / middle_building_height) * 100 = 80 := by
  sorry

end NUMINAMATH_GPT_left_building_percentage_l1955_195568


namespace NUMINAMATH_GPT_range_of_a_l1955_195579

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 3 * x + 2 = 0) → ∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1955_195579


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l1955_195532

theorem condition_sufficient_but_not_necessary (a : ℝ) : (a > 9 → (1 / a < 1 / 9)) ∧ ¬(1 / a < 1 / 9 → a > 9) :=
by 
  sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l1955_195532


namespace NUMINAMATH_GPT_lego_set_cost_l1955_195513

-- Define the cost per doll and number of dolls
def costPerDoll : ℝ := 15
def numberOfDolls : ℝ := 4

-- Define the total amount spent on the younger sister's dolls
def totalAmountOnDolls : ℝ := numberOfDolls * costPerDoll

-- Define the number of lego sets
def numberOfLegoSets : ℝ := 3

-- Define the total amount spent on lego sets (needs to be equal to totalAmountOnDolls)
def totalAmountOnLegoSets : ℝ := 60

-- Define the cost per lego set that we need to prove
def costPerLegoSet : ℝ := 20

-- Theorem to prove that the cost per lego set is $20
theorem lego_set_cost (h : totalAmountOnLegoSets = totalAmountOnDolls) : 
  totalAmountOnLegoSets / numberOfLegoSets = costPerLegoSet := by
  sorry

end NUMINAMATH_GPT_lego_set_cost_l1955_195513


namespace NUMINAMATH_GPT_roots_sum_product_l1955_195503

theorem roots_sum_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : ∀ x : ℝ, x^2 - p*x - 2*q = 0) :
  (p + q = p) ∧ (p * q = -2*q) :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_product_l1955_195503


namespace NUMINAMATH_GPT_number_of_dissimilar_terms_l1955_195577

theorem number_of_dissimilar_terms :
  let n := 7;
  let k := 4;
  let number_of_terms := Nat.choose (n + k - 1) (k - 1);
  number_of_terms = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dissimilar_terms_l1955_195577


namespace NUMINAMATH_GPT_g_at_zero_l1955_195583

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_at_zero : g 0 = -Real.sqrt 2 :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_g_at_zero_l1955_195583


namespace NUMINAMATH_GPT_all_girls_probability_l1955_195556

-- Definition of the problem conditions
def probability_of_girl : ℚ := 1 / 2
def events_independent (P1 P2 P3 : ℚ) : Prop := P1 * P2 = P1 ∧ P2 * P3 = P2

-- The statement to prove
theorem all_girls_probability :
  events_independent probability_of_girl probability_of_girl probability_of_girl →
  (probability_of_girl * probability_of_girl * probability_of_girl) = 1 / 8 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_all_girls_probability_l1955_195556


namespace NUMINAMATH_GPT_painted_cube_l1955_195526

theorem painted_cube (n : ℕ) (h : 3 / 4 * (6 * n ^ 3) = 4 * n ^ 2) : n = 2 := sorry

end NUMINAMATH_GPT_painted_cube_l1955_195526


namespace NUMINAMATH_GPT_second_number_l1955_195514

theorem second_number (x : ℝ) (h : 3 + x + 333 + 33.3 = 399.6) : x = 30.3 :=
sorry

end NUMINAMATH_GPT_second_number_l1955_195514


namespace NUMINAMATH_GPT_find_coefficients_l1955_195589

variable (P Q x : ℝ)

theorem find_coefficients :
  (∀ x, x^2 - 8 * x - 20 = (x - 10) * (x + 2))
  → (∀ x, 6 * x - 4 = P * (x + 2) + Q * (x - 10))
  → P = 14 / 3 ∧ Q = 4 / 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_coefficients_l1955_195589


namespace NUMINAMATH_GPT_smallest_n_l1955_195540

theorem smallest_n (n : ℕ) :
  (1 / 4 : ℚ) + (n / 8 : ℚ) > 1 ↔ n ≥ 7 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l1955_195540


namespace NUMINAMATH_GPT_find_supplementary_angle_l1955_195559

def A := 45
def supplementary_angle (A S : ℕ) := A + S = 180
def complementary_angle (A C : ℕ) := A + C = 90
def thrice_complementary (S C : ℕ) := S = 3 * C

theorem find_supplementary_angle : 
  ∀ (A S C : ℕ), 
    A = 45 → 
    supplementary_angle A S →
    complementary_angle A C →
    thrice_complementary S C → 
    S = 135 :=
by
  intros A S C hA hSupp hComp hThrice
  have h1 : A = 45 := by assumption
  have h2 : A + S = 180 := by assumption
  have h3 : A + C = 90 := by assumption
  have h4 : S = 3 * C := by assumption
  sorry

end NUMINAMATH_GPT_find_supplementary_angle_l1955_195559


namespace NUMINAMATH_GPT_height_of_smaller_cone_l1955_195567

theorem height_of_smaller_cone (h_frustum : ℝ) (area_lower_base area_upper_base : ℝ) 
  (h_frustum_eq : h_frustum = 18) 
  (area_lower_base_eq : area_lower_base = 144 * Real.pi) 
  (area_upper_base_eq : area_upper_base = 16 * Real.pi) : 
  ∃ (x : ℝ), x = 9 :=
by
  -- Definitions and assumptions go here
  sorry

end NUMINAMATH_GPT_height_of_smaller_cone_l1955_195567


namespace NUMINAMATH_GPT_black_friday_sales_l1955_195550

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end NUMINAMATH_GPT_black_friday_sales_l1955_195550


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1955_195551

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l1955_195551


namespace NUMINAMATH_GPT_ball_more_expensive_l1955_195573

theorem ball_more_expensive (B L : ℝ) (h1 : 2 * B + 3 * L = 1300) (h2 : 3 * B + 2 * L = 1200) : 
  L - B = 100 := 
sorry

end NUMINAMATH_GPT_ball_more_expensive_l1955_195573


namespace NUMINAMATH_GPT_lines_parallel_iff_a_eq_neg2_l1955_195557

def line₁_eq (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line₂_eq (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y - 1 = 0

theorem lines_parallel_iff_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, line₁_eq a x y → line₂_eq a x y) ↔ a = -2 :=
by sorry

end NUMINAMATH_GPT_lines_parallel_iff_a_eq_neg2_l1955_195557


namespace NUMINAMATH_GPT_original_perimeter_of_rectangle_l1955_195591

theorem original_perimeter_of_rectangle
  (a b : ℝ)
  (h : (a + 3) * (b + 3) - a * b = 90) :
  2 * (a + b) = 54 :=
sorry

end NUMINAMATH_GPT_original_perimeter_of_rectangle_l1955_195591


namespace NUMINAMATH_GPT_inequality_positive_l1955_195584

theorem inequality_positive (x : ℝ) : (1 / 3) * x - x > 0 ↔ (-2 / 3) * x > 0 := 
  sorry

end NUMINAMATH_GPT_inequality_positive_l1955_195584


namespace NUMINAMATH_GPT_total_turns_to_fill_drum_l1955_195590

variable (Q : ℝ) -- Capacity of bucket Q
variable (turnsP : ℝ) (P_capacity : ℝ) (R_capacity : ℝ) (drum_capacity : ℝ)

-- Condition: It takes 60 turns for bucket P to fill the empty drum
def bucketP_fills_drum_in_60_turns : Prop := turnsP = 60 ∧ P_capacity = 3 * Q ∧ drum_capacity = 60 * P_capacity

-- Condition: Bucket P has thrice the capacity as bucket Q
def bucketP_capacity : Prop := P_capacity = 3 * Q

-- Condition: Bucket R has half the capacity as bucket Q
def bucketR_capacity : Prop := R_capacity = Q / 2

-- Computation: Using all three buckets together, find the combined capacity filled in one turn
def combined_capacity_per_turn : ℝ := P_capacity + Q + R_capacity

-- Main Theorem: It takes 40 turns to fill the drum using all three buckets together
theorem total_turns_to_fill_drum
  (h1 : bucketP_fills_drum_in_60_turns Q turnsP P_capacity drum_capacity)
  (h2 : bucketP_capacity Q P_capacity)
  (h3 : bucketR_capacity Q R_capacity) :
  drum_capacity / combined_capacity_per_turn Q P_capacity (Q / 2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_turns_to_fill_drum_l1955_195590


namespace NUMINAMATH_GPT_karen_drive_l1955_195517

theorem karen_drive (a b c x : ℕ) (h1 : a ≥ 1) (h2 : a + b + c ≤ 9) (h3 : 33 * (c - a) = 25 * x) :
  a^2 + b^2 + c^2 = 75 :=
sorry

end NUMINAMATH_GPT_karen_drive_l1955_195517


namespace NUMINAMATH_GPT_combined_CD_length_l1955_195598

def CD1 := 1.5
def CD2 := 1.5
def CD3 := 2 * CD1

theorem combined_CD_length : CD1 + CD2 + CD3 = 6 := 
by
  sorry

end NUMINAMATH_GPT_combined_CD_length_l1955_195598


namespace NUMINAMATH_GPT_rate_of_interest_l1955_195566

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest 8925 R 5 = 4016.25 ∧ R = 9 := 
by
  use 9
  simp [simple_interest]
  norm_num
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1955_195566


namespace NUMINAMATH_GPT_arith_seq_a1_eq_15_l1955_195527

variable {a : ℕ → ℤ} (a_seq : ∀ n, a n = a 1 + (n-1) * d)
variable {a_4 : ℤ} (h4 : a 4 = 9)
variable {a_8 : ℤ} (h8 : a 8 = -a 9)

theorem arith_seq_a1_eq_15 (a_seq : ∀ n, a n = a 1 + (n-1) * d) (h4 : a 4 = 9) (h8 : a 8 = -a 9) : a 1 = 15 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_arith_seq_a1_eq_15_l1955_195527


namespace NUMINAMATH_GPT_factorize_cubic_l1955_195546

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_cubic_l1955_195546


namespace NUMINAMATH_GPT_mass_percentage_Al_in_AlBr3_l1955_195537

theorem mass_percentage_Al_in_AlBr3 
  (molar_mass_Al : Real := 26.98) 
  (molar_mass_Br : Real := 79.90) 
  (molar_mass_AlBr3 : Real := molar_mass_Al + 3 * molar_mass_Br)
  : (molar_mass_Al / molar_mass_AlBr3) * 100 = 10.11 := 
by 
  -- Here we would provide the proof; skipping with sorry
  sorry

end NUMINAMATH_GPT_mass_percentage_Al_in_AlBr3_l1955_195537


namespace NUMINAMATH_GPT_simplify_expression_l1955_195571

theorem simplify_expression : (1 / (1 + Real.sqrt 3) * 1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1955_195571


namespace NUMINAMATH_GPT_nuts_mixture_weight_l1955_195564

variable (m n : ℕ)
variable (weight_almonds per_part total_weight : ℝ)

theorem nuts_mixture_weight (h1 : m = 5) (h2 : n = 2) (h3 : weight_almonds = 250) 
  (h4 : per_part = weight_almonds / m) (h5 : total_weight = per_part * (m + n)) : 
  total_weight = 350 := by
  sorry

end NUMINAMATH_GPT_nuts_mixture_weight_l1955_195564


namespace NUMINAMATH_GPT_intersection_A_B_l1955_195529

-- Definition of set A
def A (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Definition of set B
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem stating the intersection of sets A and B
theorem intersection_A_B (x : ℝ) : (A x ∧ B x) ↔ (-1 < x ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1955_195529


namespace NUMINAMATH_GPT_suitcase_lock_settings_l1955_195552

-- Define the number of settings for each dial choice considering the conditions
noncomputable def first_digit_choices : ℕ := 9
noncomputable def second_digit_choices : ℕ := 9
noncomputable def third_digit_choices : ℕ := 8
noncomputable def fourth_digit_choices : ℕ := 7

-- Theorem to prove the total number of different settings
theorem suitcase_lock_settings : first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices = 4536 :=
by sorry

end NUMINAMATH_GPT_suitcase_lock_settings_l1955_195552


namespace NUMINAMATH_GPT_square_area_l1955_195501

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_square_area_l1955_195501


namespace NUMINAMATH_GPT_television_price_l1955_195558

theorem television_price (SP : ℝ) (RP : ℕ) (discount : ℝ) (h1 : discount = 0.20) (h2 : SP = RP - discount * RP) (h3 : SP = 480) : RP = 600 :=
by
  sorry

end NUMINAMATH_GPT_television_price_l1955_195558


namespace NUMINAMATH_GPT_present_worth_approx_l1955_195530

noncomputable def amount_after_years (P : ℝ) : ℝ :=
  let A1 := P * (1 + 5 / 100)                      -- Amount after the first year.
  let A2 := A1 * (1 + 5 / 100)^2                   -- Amount after the second year.
  let A3 := A2 * (1 + 3 / 100)^4                   -- Amount after the third year.
  A3

noncomputable def banker's_gain (P : ℝ) : ℝ :=
  amount_after_years P - P

theorem present_worth_approx :
  ∃ P : ℝ, abs (P - 114.94) < 1 ∧ banker's_gain P = 36 :=
sorry

end NUMINAMATH_GPT_present_worth_approx_l1955_195530


namespace NUMINAMATH_GPT_determine_N_l1955_195538

variable (U M N : Set ℕ)

theorem determine_N (h1 : U = {1, 2, 3, 4, 5})
  (h2 : U = M ∪ N)
  (h3 : M ∩ (U \ N) = {2, 4}) :
  N = {1, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_determine_N_l1955_195538


namespace NUMINAMATH_GPT_lilyPadsFullCoverage_l1955_195515

def lilyPadDoubling (t: ℕ) : ℕ :=
  t + 1

theorem lilyPadsFullCoverage (t: ℕ) (h: t = 47) : lilyPadDoubling t = 48 :=
by
  rw [h]
  unfold lilyPadDoubling
  rfl

end NUMINAMATH_GPT_lilyPadsFullCoverage_l1955_195515


namespace NUMINAMATH_GPT_length_of_BD_is_six_l1955_195545

-- Definitions of the conditions
def AB : ℕ := 6
def BC : ℕ := 11
def CD : ℕ := 6
def DA : ℕ := 8
def BD : ℕ := 6 -- adding correct answer into definition

-- The statement we want to prove
theorem length_of_BD_is_six (hAB : AB = 6) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 8) (hBD_int : BD = 6) : 
  BD = 6 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_length_of_BD_is_six_l1955_195545


namespace NUMINAMATH_GPT_division_result_l1955_195555

-- Define the arithmetic expression
def arithmetic_expression : ℕ := (20 + 15 * 3) - 10

-- Define the main problem
def problem : Prop := 250 / arithmetic_expression = 250 / 55

-- The theorem statement that needs to be proved
theorem division_result : problem := by
    sorry

end NUMINAMATH_GPT_division_result_l1955_195555


namespace NUMINAMATH_GPT_complement_union_l1955_195560

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4}

-- Define the set S
def S : Set ℕ := {1, 3}

-- Define the set T
def T : Set ℕ := {4}

-- Define the complement of S in I
def complement_I_S : Set ℕ := I \ S

-- State the theorem to be proved
theorem complement_union : (complement_I_S ∪ T) = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_l1955_195560


namespace NUMINAMATH_GPT_fraction_inequality_l1955_195587

theorem fraction_inequality (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) :
  (a / d) < (b / c) :=
sorry

end NUMINAMATH_GPT_fraction_inequality_l1955_195587


namespace NUMINAMATH_GPT_job_completion_days_l1955_195536

variable (m r h d : ℕ)

theorem job_completion_days :
  (m + 2 * r) * (h + 1) * (m * h * d / ((m + 2 * r) * (h + 1))) = m * h * d :=
by
  sorry

end NUMINAMATH_GPT_job_completion_days_l1955_195536


namespace NUMINAMATH_GPT_bonnets_per_orphanage_l1955_195516

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end NUMINAMATH_GPT_bonnets_per_orphanage_l1955_195516


namespace NUMINAMATH_GPT_sqrt_x2y_l1955_195508

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end NUMINAMATH_GPT_sqrt_x2y_l1955_195508


namespace NUMINAMATH_GPT_perpendicular_lines_b_l1955_195521

theorem perpendicular_lines_b (b : ℝ) : 
  (∃ (k m: ℝ), k = 3 ∧ 2 * m + b * k = 14 ∧ (k * m = -1)) ↔ b = 2 / 3 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_b_l1955_195521


namespace NUMINAMATH_GPT_cyclist_speed_ratio_l1955_195535

-- conditions: 
variables (T₁ T₂ o₁ o₂ : ℝ)
axiom h1 : o₁ + T₁ = o₂ + T₂
axiom h2 : T₁ = 2 * o₂
axiom h3 : T₂ = 4 * o₁

-- Proof statement to show that the second cyclist rides 1.5 times faster:
theorem cyclist_speed_ratio : T₁ / T₂ = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_l1955_195535


namespace NUMINAMATH_GPT_max_z_under_D_le_1_l1955_195511

noncomputable def f (x a b : ℝ) : ℝ := x - a * x^2 + b
noncomputable def f0 (x b0 : ℝ) : ℝ := x^2 + b0
noncomputable def g (x a b b0 : ℝ) : ℝ := f x a b - f0 x b0

theorem max_z_under_D_le_1 
  (a b b0 : ℝ) (D : ℝ)
  (h_a : a = 0) 
  (h_b0 : b0 = 0) 
  (h_D : D ≤ 1)
  (h_maxD : ∀ x : ℝ, - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 → g (Real.sin x) a b b0 ≤ D) :
  ∃ z : ℝ, z = b - a^2 / 4 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_z_under_D_le_1_l1955_195511


namespace NUMINAMATH_GPT_Wolfgang_marble_count_l1955_195580

theorem Wolfgang_marble_count
  (W L M : ℝ)
  (hL : L = 5/4 * W)
  (hM : M = 2/3 * (W + L))
  (hTotal : W + L + M = 60) :
  W = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_Wolfgang_marble_count_l1955_195580


namespace NUMINAMATH_GPT_rectangle_perimeter_l1955_195520

theorem rectangle_perimeter {w l : ℝ} 
  (h_area : l * w = 450)
  (h_length : l = 2 * w) :
  2 * (l + w) = 90 :=
by sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1955_195520


namespace NUMINAMATH_GPT_four_digit_non_convertible_to_1992_multiple_l1955_195569

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_multiple_of_1992 (n : ℕ) : Prop :=
  n % 1992 = 0

def reachable (n m : ℕ) (k : ℕ) : Prop :=
  ∃ x y z : ℕ, 
    x ≠ m ∧ y ≠ m ∧ z ≠ m ∧
    (n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3)) % 1992 = 0 ∧
    n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3) < 10000

theorem four_digit_non_convertible_to_1992_multiple :
  ∃ n : ℕ, is_four_digit n ∧ (∀ m : ℕ, is_four_digit m ∧ is_multiple_of_1992 m → ¬ reachable n m 3) :=
sorry

end NUMINAMATH_GPT_four_digit_non_convertible_to_1992_multiple_l1955_195569


namespace NUMINAMATH_GPT_find_n_l1955_195554

theorem find_n (n : ℕ) : 
  (1/5 : ℝ)^35 * (1/4 : ℝ)^n = (1 : ℝ) / (2 * 10^35) → n = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_n_l1955_195554


namespace NUMINAMATH_GPT_average_visitors_on_other_days_l1955_195534

theorem average_visitors_on_other_days 
  (avg_sunday : ℕ) (avg_month : ℕ) 
  (days_in_month : ℕ) (sundays : ℕ) (other_days : ℕ) 
  (visitors_on_other_days : ℕ) :
  avg_sunday = 510 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  (sundays * avg_sunday + other_days * visitors_on_other_days = avg_month * days_in_month) →
  visitors_on_other_days = 240 :=
by
  intros hs hm hd hsunded hotherdays heq
  sorry

end NUMINAMATH_GPT_average_visitors_on_other_days_l1955_195534


namespace NUMINAMATH_GPT_symmetry_about_x2_symmetry_about_2_0_l1955_195592

-- Define the conditions and their respective conclusions.
theorem symmetry_about_x2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) : 
  ∀ x, f (x) = f (4 - x) := 
sorry

theorem symmetry_about_2_0 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = -f (3 + x)) : 
  ∀ x, f (x) = -f (4 - x) := 
sorry

end NUMINAMATH_GPT_symmetry_about_x2_symmetry_about_2_0_l1955_195592


namespace NUMINAMATH_GPT_largest_interior_angle_l1955_195585

theorem largest_interior_angle (x : ℝ) (h₀ : 50 + 55 + x = 180) : 
  max 50 (max 55 x) = 75 := by
  sorry

end NUMINAMATH_GPT_largest_interior_angle_l1955_195585


namespace NUMINAMATH_GPT_toothpicks_150th_stage_l1955_195542

-- Define the arithmetic sequence parameters
def first_term : ℕ := 4
def common_difference : ℕ := 4

-- Define the term number we are interested in
def stage_number : ℕ := 150

-- The total number of toothpicks in the nth stage of an arithmetic sequence
def num_toothpicks (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

-- Theorem stating the number of toothpicks in the 150th stage
theorem toothpicks_150th_stage : num_toothpicks first_term common_difference stage_number = 600 :=
by
  sorry

end NUMINAMATH_GPT_toothpicks_150th_stage_l1955_195542


namespace NUMINAMATH_GPT_possible_values_of_a_l1955_195522

theorem possible_values_of_a :
  (∀ x, (x^2 - 3 * x + 2 = 0) → (ax - 2 = 0)) → (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l1955_195522


namespace NUMINAMATH_GPT_simplify_fraction_result_l1955_195506

theorem simplify_fraction_result : (130 / 16900) * 65 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_result_l1955_195506


namespace NUMINAMATH_GPT_beth_sold_coins_l1955_195581

theorem beth_sold_coins :
  let initial_coins := 125
  let gift_coins := 35
  let total_coins := initial_coins + gift_coins
  let sold_coins := total_coins / 2
  sold_coins = 80 :=
by
  sorry

end NUMINAMATH_GPT_beth_sold_coins_l1955_195581


namespace NUMINAMATH_GPT_breakfast_plate_contains_2_eggs_l1955_195547

-- Define the conditions
def breakfast_plate := Nat
def num_customers := 14
def num_bacon_strips := 56

-- Define the bacon strips per plate
def bacon_strips_per_plate (num_bacon_strips num_customers : Nat) : Nat :=
  num_bacon_strips / num_customers

-- Define the number of eggs per plate given twice as many bacon strips as eggs
def eggs_per_plate (bacon_strips_per_plate : Nat) : Nat :=
  bacon_strips_per_plate / 2

-- The main theorem we need to prove
theorem breakfast_plate_contains_2_eggs :
  eggs_per_plate (bacon_strips_per_plate 56 14) = 2 :=
by
  sorry

end NUMINAMATH_GPT_breakfast_plate_contains_2_eggs_l1955_195547


namespace NUMINAMATH_GPT_find_point_B_l1955_195594

theorem find_point_B (A B : ℝ) (h1 : A = 2) (h2 : abs (B - A) = 5) : B = -3 ∨ B = 7 :=
by
  -- This is where the proof steps would go, but we can skip it with sorry.
  sorry

end NUMINAMATH_GPT_find_point_B_l1955_195594


namespace NUMINAMATH_GPT_boat_travel_distance_l1955_195523

variable (v c d : ℝ) (c_eq_1 : c = 1)

theorem boat_travel_distance : 
  (∀ (v : ℝ), d = (v + c) * 4 → d = (v - c) * 6) → d = 24 := 
by
  intro H
  sorry

end NUMINAMATH_GPT_boat_travel_distance_l1955_195523


namespace NUMINAMATH_GPT_sum_of_squares_arithmetic_geometric_l1955_195541

theorem sum_of_squares_arithmetic_geometric (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 225) : x^2 + y^2 = 1150 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_arithmetic_geometric_l1955_195541


namespace NUMINAMATH_GPT_tetrahedron_in_cube_l1955_195543

theorem tetrahedron_in_cube (a x : ℝ) (h : a = 6) :
  (∃ x, x = 6 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_tetrahedron_in_cube_l1955_195543


namespace NUMINAMATH_GPT_four_digit_sum_divisible_l1955_195500

theorem four_digit_sum_divisible (A B C D : ℕ) :
  (10 * A + B + 10 * C + D = 94) ∧ (1000 * A + 100 * B + 10 * C + D % 94 = 0) →
  false :=
by
  sorry

end NUMINAMATH_GPT_four_digit_sum_divisible_l1955_195500


namespace NUMINAMATH_GPT_find_principal_sum_l1955_195599

theorem find_principal_sum 
  (CI SI P : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (hCI : CI = 11730) 
  (hSI : SI = 10200) 
  (hT : T = 2) 
  (hCI_formula : CI = P * ((1 + R / 100)^T - 1)) 
  (hSI_formula : SI = (P * R * T) / 100) 
  (h_diff : CI - SI = 1530) :
  P = 34000 := 
by 
  sorry

end NUMINAMATH_GPT_find_principal_sum_l1955_195599


namespace NUMINAMATH_GPT_valid_propositions_l1955_195533

theorem valid_propositions :
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧ (∃ n : ℝ, ∀ m : ℝ, m * n = m) :=
by
  sorry

end NUMINAMATH_GPT_valid_propositions_l1955_195533


namespace NUMINAMATH_GPT_combined_exceeds_limit_l1955_195576

-- Let Zone A, Zone B, and Zone C be zones on a road.
-- Let pA be the percentage of motorists exceeding the speed limit in Zone A.
-- Let pB be the percentage of motorists exceeding the speed limit in Zone B.
-- Let pC be the percentage of motorists exceeding the speed limit in Zone C.
-- Each zone has an equal amount of motorists.

def pA : ℝ := 15
def pB : ℝ := 20
def pC : ℝ := 10

/-
Prove that the combined percentage of motorists who exceed the speed limit
across all three zones is 15%.
-/
theorem combined_exceeds_limit :
  (pA + pB + pC) / 3 = 15 := 
by sorry

end NUMINAMATH_GPT_combined_exceeds_limit_l1955_195576


namespace NUMINAMATH_GPT_chord_to_diameter_ratio_l1955_195524

open Real

theorem chord_to_diameter_ratio
  (r R : ℝ) (h1 : r = R / 2)
  (a : ℝ)
  (h2 : r^2 = a^2 * 3 / 2) :
  3 * a / (2 * R) = 3 * sqrt 6 / 8 :=
by
  sorry

end NUMINAMATH_GPT_chord_to_diameter_ratio_l1955_195524


namespace NUMINAMATH_GPT_sum_of_five_consecutive_even_integers_l1955_195578

theorem sum_of_five_consecutive_even_integers (a : ℤ) (h : a + (a + 4) = 150) :
  a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_even_integers_l1955_195578


namespace NUMINAMATH_GPT_gcd_g_y_l1955_195505

noncomputable def g (y : ℕ) : ℕ := (3 * y + 5) * (6 * y + 7) * (10 * y + 3) * (5 * y + 11) * (y + 7)

theorem gcd_g_y (y : ℕ) (h : ∃ k : ℕ, y = 18090 * k) : Nat.gcd (g y) y = 8085 := 
sorry

end NUMINAMATH_GPT_gcd_g_y_l1955_195505


namespace NUMINAMATH_GPT_area_of_quadrilateral_EFGH_l1955_195562

noncomputable def trapezium_ABCD_midpoints_area : ℝ :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (4, 3)
  let D := (0, 3)
  let E := ((B.1 + C.1)/2, (B.2 + C.2)/2) -- midpoint of BC
  let F := ((C.1 + D.1)/2, (C.2 + D.2)/2) -- midpoint of CD
  let G := ((A.1 + D.1)/2, (A.2 + D.2)/2) -- midpoint of AD
  let H := ((G.1 + E.1)/2, (G.2 + E.2)/2) -- midpoint of GE
  let area := (E.1 * F.2 + F.1 * G.2 + G.1 * H.2 + H.1 * E.2 - F.1 * E.2 - G.1 * F.2 - H.1 * G.2 - E.1 * H.2) / 2
  abs area

theorem area_of_quadrilateral_EFGH : trapezium_ABCD_midpoints_area = 0.75 := by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_EFGH_l1955_195562


namespace NUMINAMATH_GPT_log_exp_sum_l1955_195572

theorem log_exp_sum :
  2^(Real.log 3 / Real.log 2) + Real.log (Real.sqrt 5) / Real.log 10 + Real.log (Real.sqrt 20) / Real.log 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_log_exp_sum_l1955_195572


namespace NUMINAMATH_GPT_six_digit_number_all_equal_l1955_195563

open Nat

theorem six_digit_number_all_equal (n : ℕ) (h : n = 21) : 12 * n^2 + 12 * n + 11 = 5555 :=
by
  rw [h]  -- Substitute n = 21
  sorry  -- Omit the actual proof steps

end NUMINAMATH_GPT_six_digit_number_all_equal_l1955_195563


namespace NUMINAMATH_GPT_length_of_platform_l1955_195518

variable (Vtrain : Real := 55)
variable (str_len : Real := 360)
variable (cross_time : Real := 57.59539236861051)
variable (conversion_factor : Real := 5/18)

theorem length_of_platform :
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  ∃ L : Real, str_len + L = distance_covered → L = 520 :=
by
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  exists (distance_covered - str_len)
  intro h
  have h1 : distance_covered - str_len = 520 := sorry
  exact h1


end NUMINAMATH_GPT_length_of_platform_l1955_195518


namespace NUMINAMATH_GPT_wicket_keeper_older_than_captain_l1955_195539

variables (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ)

def x_older_than_captain (captain_age team_avg_age num_players remaining_avg_age : ℕ) : ℕ :=
  team_avg_age * num_players - remaining_avg_age * (num_players - 2) - 2 * captain_age

theorem wicket_keeper_older_than_captain 
  (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ) 
  (h1 : captain_age = 25) (h2 : team_avg_age = 23) (h3 : num_players = 11) (h4 : remaining_avg_age = 22) :
  x_older_than_captain captain_age team_avg_age num_players remaining_avg_age = 5 :=
by sorry

end NUMINAMATH_GPT_wicket_keeper_older_than_captain_l1955_195539


namespace NUMINAMATH_GPT_symmetry_condition_l1955_195544

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = x ↔ x = (ax + b) / (cx - d)) ∧ 
  (∀ x y : ℝ, y = -x ↔ x = (-ax + b) / (-cx - d)) → 
  d + b = 0 :=
by sorry

end NUMINAMATH_GPT_symmetry_condition_l1955_195544


namespace NUMINAMATH_GPT_cyclists_original_number_l1955_195553

theorem cyclists_original_number (x : ℕ) (h : x > 2) : 
  (80 / (x - 2 : ℕ) = 80 / x + 2) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_original_number_l1955_195553


namespace NUMINAMATH_GPT_second_order_arithmetic_progression_a100_l1955_195510

theorem second_order_arithmetic_progression_a100 :
  ∀ (a : ℕ → ℕ), 
    a 1 = 2 → 
    a 2 = 3 → 
    a 3 = 5 → 
    (∀ n, a (n + 1) - a n = n) → 
    a 100 = 4952 :=
by
  intros a h1 h2 h3 hdiff
  sorry

end NUMINAMATH_GPT_second_order_arithmetic_progression_a100_l1955_195510


namespace NUMINAMATH_GPT_speed_in_still_water_l1955_195565

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 24 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1955_195565


namespace NUMINAMATH_GPT_shirts_per_kid_l1955_195574

-- Define given conditions
def n_buttons : Nat := 63
def buttons_per_shirt : Nat := 7
def n_kids : Nat := 3

-- The proof goal
theorem shirts_per_kid : (n_buttons / buttons_per_shirt) / n_kids = 3 := by
  sorry

end NUMINAMATH_GPT_shirts_per_kid_l1955_195574


namespace NUMINAMATH_GPT_min_segment_length_l1955_195512

theorem min_segment_length 
  (angle : ℝ) (P : ℝ × ℝ)
  (dist_x : ℝ) (dist_y : ℝ) 
  (hx : P.1 ≤ dist_x ∧ P.2 = dist_y)
  (hy : P.2 ≤ dist_y ∧ P.1 = dist_x)
  (right_angle : angle = 90) 
  : ∃ (d : ℝ), d = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_segment_length_l1955_195512


namespace NUMINAMATH_GPT_sin_squared_alpha_plus_pi_over_4_l1955_195507

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + Real.pi / 4) ^ 2 = 5 / 6 := 
sorry

end NUMINAMATH_GPT_sin_squared_alpha_plus_pi_over_4_l1955_195507


namespace NUMINAMATH_GPT_distribute_candies_l1955_195588

-- Definition of the problem conditions
def candies : ℕ := 10

-- The theorem stating the proof problem
theorem distribute_candies : (2 ^ (candies - 1)) = 512 := 
by
  sorry

end NUMINAMATH_GPT_distribute_candies_l1955_195588


namespace NUMINAMATH_GPT_angle_C_in_triangle_l1955_195519

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 110) (ht : A + B + C = 180) : C = 70 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l1955_195519


namespace NUMINAMATH_GPT_remaining_oak_trees_l1955_195597

def initial_oak_trees : ℕ := 9
def cut_down_oak_trees : ℕ := 2

theorem remaining_oak_trees : initial_oak_trees - cut_down_oak_trees = 7 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_oak_trees_l1955_195597


namespace NUMINAMATH_GPT_not_possible_select_seven_distinct_weights_no_equal_subsets_l1955_195531

theorem not_possible_select_seven_distinct_weights_no_equal_subsets :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 27 → s.card = 7 → ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum id = b.sum id :=
by
  intro s hs hcard
  sorry

end NUMINAMATH_GPT_not_possible_select_seven_distinct_weights_no_equal_subsets_l1955_195531


namespace NUMINAMATH_GPT_base_case_inequality_induction_inequality_l1955_195548

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_base_case_inequality_induction_inequality_l1955_195548


namespace NUMINAMATH_GPT_portion_apples_weight_fraction_l1955_195586

-- Given conditions
def total_apples : ℕ := 28
def total_weight_kg : ℕ := 3
def number_of_portions : ℕ := 7

-- Proof statement
theorem portion_apples_weight_fraction :
  (1 / number_of_portions = 1 / 7) ∧ (3 / number_of_portions = 3 / 7) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_portion_apples_weight_fraction_l1955_195586


namespace NUMINAMATH_GPT_number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l1955_195570

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime 
  (n : ℕ) (h : n ≥ 2) : (∃ (a b : ℕ), a ≠ b ∧ is_prime (a^3 + 2) ∧ is_prime (b^3 + 2)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l1955_195570


namespace NUMINAMATH_GPT_emily_51_49_calculations_l1955_195575

theorem emily_51_49_calculations :
  (51^2 = 50^2 + 101) ∧ (49^2 = 50^2 - 99) :=
by
  sorry

end NUMINAMATH_GPT_emily_51_49_calculations_l1955_195575


namespace NUMINAMATH_GPT_smallest_n_condition_smallest_n_value_l1955_195596

theorem smallest_n_condition :
  ∃ (n : ℕ), n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) ∧ 
  ∀ m, (m < 1000 ∧ (99999 % m = 0) ∧ (9999 % (m + 7) = 0)) → n ≤ m := 
sorry

theorem smallest_n_value :
  ∃ (n : ℕ), n = 266 ∧ n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) := 
sorry

end NUMINAMATH_GPT_smallest_n_condition_smallest_n_value_l1955_195596


namespace NUMINAMATH_GPT_students_end_year_10_l1955_195549

def students_at_end_of_year (initial_students : ℕ) (left_students : ℕ) (increase_percent : ℕ) : ℕ :=
  let remaining_students := initial_students - left_students
  let increased_students := (remaining_students * increase_percent) / 100
  remaining_students + increased_students

theorem students_end_year_10 : 
  students_at_end_of_year 10 4 70 = 10 := by 
  sorry

end NUMINAMATH_GPT_students_end_year_10_l1955_195549


namespace NUMINAMATH_GPT_Z_4_1_eq_27_l1955_195504

def Z (a b : ℕ) : ℕ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem Z_4_1_eq_27 : Z 4 1 = 27 := by
  sorry

end NUMINAMATH_GPT_Z_4_1_eq_27_l1955_195504


namespace NUMINAMATH_GPT_bee_flight_time_l1955_195509

theorem bee_flight_time (t : ℝ) : 
  let speed_daisy_to_rose := 2.6
  let speed_rose_to_poppy := speed_daisy_to_rose + 3
  let distance_daisy_to_rose := speed_daisy_to_rose * 10
  let distance_rose_to_poppy := distance_daisy_to_rose - 8
  distance_rose_to_poppy = speed_rose_to_poppy * t
  ∧ abs (t - 3) < 1 := 
sorry

end NUMINAMATH_GPT_bee_flight_time_l1955_195509
