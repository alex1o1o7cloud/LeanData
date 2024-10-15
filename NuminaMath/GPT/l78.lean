import Mathlib

namespace NUMINAMATH_GPT_workers_allocation_l78_7868

-- Definitions based on conditions
def num_workers := 90
def bolt_per_worker := 15
def nut_per_worker := 24
def bolt_matching_requirement := 2

-- Statement of the proof problem
theorem workers_allocation (x y : ℕ) :
  x + y = num_workers ∧
  bolt_matching_requirement * bolt_per_worker * x = nut_per_worker * y →
  x = 40 ∧ y = 50 :=
by
  sorry

end NUMINAMATH_GPT_workers_allocation_l78_7868


namespace NUMINAMATH_GPT_ned_total_mows_l78_7870

def ned_mowed_front (spring summer fall : Nat) : Nat :=
  spring + summer + fall

def ned_mowed_backyard (spring summer fall : Nat) : Nat :=
  spring + summer + fall

theorem ned_total_mows :
  let front_spring := 6
  let front_summer := 5
  let front_fall := 4
  let backyard_spring := 5
  let backyard_summer := 7
  let backyard_fall := 3
  ned_mowed_front front_spring front_summer front_fall +
  ned_mowed_backyard backyard_spring backyard_summer backyard_fall = 30 := by
  sorry

end NUMINAMATH_GPT_ned_total_mows_l78_7870


namespace NUMINAMATH_GPT_factorize1_factorize2_factorize3_l78_7894

-- Proof problem 1: Prove m^2 + 4m + 4 = (m + 2)^2
theorem factorize1 (m : ℝ) : m^2 + 4 * m + 4 = (m + 2)^2 :=
sorry

-- Proof problem 2: Prove a^2 b - 4ab^2 + 3b^3 = b(a-b)(a-3b)
theorem factorize2 (a b : ℝ) : a^2 * b - 4 * a * b^2 + 3 * b^3 = b * (a - b) * (a - 3 * b) :=
sorry

-- Proof problem 3: Prove (x^2 + y^2)^2 - 4x^2 y^2 = (x + y)^2 (x - y)^2
theorem factorize3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

end NUMINAMATH_GPT_factorize1_factorize2_factorize3_l78_7894


namespace NUMINAMATH_GPT_integer_values_of_a_l78_7801

theorem integer_values_of_a (x : ℤ) (a : ℤ)
  (h : x^3 + 3*x^2 + a*x + 11 = 0) :
  a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end NUMINAMATH_GPT_integer_values_of_a_l78_7801


namespace NUMINAMATH_GPT_Petya_has_24_chips_l78_7883

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end NUMINAMATH_GPT_Petya_has_24_chips_l78_7883


namespace NUMINAMATH_GPT_mean_equals_sum_of_squares_l78_7834

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_equals_sum_of_squares (x y z : ℝ) (h1 : arithmetic_mean x y z = 10)
  (h2 : geometric_mean x y z = 6) (h3 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 :=
  sorry

end NUMINAMATH_GPT_mean_equals_sum_of_squares_l78_7834


namespace NUMINAMATH_GPT_distance_between_house_and_school_l78_7840

variable (T D : ℝ)

axiom cond1 : 9 * (T + 20 / 60) = D
axiom cond2 : 12 * (T - 20 / 60) = D
axiom cond3 : 15 * (T - 40 / 60) = D

theorem distance_between_house_and_school : D = 24 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_house_and_school_l78_7840


namespace NUMINAMATH_GPT_sum_sum_sum_sum_eq_one_l78_7880

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end NUMINAMATH_GPT_sum_sum_sum_sum_eq_one_l78_7880


namespace NUMINAMATH_GPT_no_intersection_of_ellipses_l78_7859

theorem no_intersection_of_ellipses :
  (∀ (x y : ℝ), (9*x^2 + y^2 = 9) ∧ (x^2 + 16*y^2 = 16) → false) :=
sorry

end NUMINAMATH_GPT_no_intersection_of_ellipses_l78_7859


namespace NUMINAMATH_GPT_find_number_l78_7809

theorem find_number (f : ℝ → ℝ) (x : ℝ)
  (h : f (x * 0.004) / 0.03 = 9.237333333333334)
  (h_linear : ∀ a, f a = a) :
  x = 69.3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_number_l78_7809


namespace NUMINAMATH_GPT_sufficient_condition_for_sets_l78_7821

theorem sufficient_condition_for_sets (A B : Set ℝ) (m : ℝ) :
    (∀ x, x ∈ A → x ∈ B) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
    have A_def : A = {y | ∃ x, y = x^2 - (3 / 2) * x + 1 ∧ (1 / 4) ≤ x ∧ x ≤ 2} := sorry
    have B_def : B = {x | x ≥ 1 - m^2} := sorry
    sorry

end NUMINAMATH_GPT_sufficient_condition_for_sets_l78_7821


namespace NUMINAMATH_GPT_combined_weight_l78_7839

theorem combined_weight (a b c : ℕ) (h1 : a + b = 122) (h2 : b + c = 125) (h3 : c + a = 127) : 
  a + b + c = 187 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l78_7839


namespace NUMINAMATH_GPT_batter_sugar_is_one_l78_7872

-- Definitions based on the conditions given
def initial_sugar : ℕ := 3
def sugar_per_bag : ℕ := 6
def num_bags : ℕ := 2
def frosting_sugar_per_dozen : ℕ := 2
def total_dozen_cupcakes : ℕ := 5

-- Total sugar Lillian has
def total_sugar : ℕ := initial_sugar + num_bags * sugar_per_bag

-- Sugar needed for frosting
def frosting_sugar_needed : ℕ := frosting_sugar_per_dozen * total_dozen_cupcakes

-- Sugar used for the batter
def batter_sugar_total : ℕ := total_sugar - frosting_sugar_needed

-- Question asked in the problem
def batter_sugar_per_dozen : ℕ := batter_sugar_total / total_dozen_cupcakes

theorem batter_sugar_is_one :
  batter_sugar_per_dozen = 1 :=
by
  sorry -- Proof is not required here

end NUMINAMATH_GPT_batter_sugar_is_one_l78_7872


namespace NUMINAMATH_GPT_country_math_l78_7803

theorem country_math (h : (1 / 3 : ℝ) * 4 = 6) : 
  ∃ x : ℝ, (1 / 6 : ℝ) * x = 15 ∧ x = 405 :=
by
  sorry

end NUMINAMATH_GPT_country_math_l78_7803


namespace NUMINAMATH_GPT_floor_neg_seven_thirds_l78_7853

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end NUMINAMATH_GPT_floor_neg_seven_thirds_l78_7853


namespace NUMINAMATH_GPT_paige_folders_l78_7814

def initial_files : Nat := 135
def deleted_files : Nat := 27
def files_per_folder : Rat := 8.5
def folders_rounded_up (files_left : Nat) (per_folder : Rat) : Nat :=
  (Rat.ceil (Rat.ofInt files_left / per_folder)).toNat

theorem paige_folders :
  folders_rounded_up (initial_files - deleted_files) files_per_folder = 13 :=
by
  sorry

end NUMINAMATH_GPT_paige_folders_l78_7814


namespace NUMINAMATH_GPT_minimum_value_of_a_plus_4b_l78_7824

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hgeo : Real.sqrt (a * b) = 2)

theorem minimum_value_of_a_plus_4b : a + 4 * b = 8 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_a_plus_4b_l78_7824


namespace NUMINAMATH_GPT_range_of_a_plus_b_l78_7893

theorem range_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b)
    (h3 : |2 - a^2| = |2 - b^2|) : 2 < a + b ∧ a + b < 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l78_7893


namespace NUMINAMATH_GPT_probability_change_needed_l78_7856

noncomputable def toy_prices : List ℝ := List.range' 1 11 |>.map (λ n => n * 0.25)

def favorite_toy_price : ℝ := 2.25

def total_quarters : ℕ := 12

def total_toy_count : ℕ := 10

def total_orders : ℕ := Nat.factorial total_toy_count

def ways_to_buy_without_change : ℕ :=
  (Nat.factorial (total_toy_count - 1)) + 2 * (Nat.factorial (total_toy_count - 2))

def probability_without_change : ℚ :=
  ↑ways_to_buy_without_change / ↑total_orders

def probability_with_change : ℚ :=
  1 - probability_without_change

theorem probability_change_needed : probability_with_change = 79 / 90 :=
  sorry

end NUMINAMATH_GPT_probability_change_needed_l78_7856


namespace NUMINAMATH_GPT_probability_of_convex_quadrilateral_l78_7891

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_convex_quadrilateral :
  let num_points := 8
  let total_chords := binomial num_points 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
by
  -- definitions
  let num_points := 8
  let total_chords := binomial 8 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  
  -- assertion of result
  have h : (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
    sorry
  exact h

end NUMINAMATH_GPT_probability_of_convex_quadrilateral_l78_7891


namespace NUMINAMATH_GPT_parrot_age_is_24_l78_7885

variable (cat_age : ℝ) (rabbit_age : ℝ) (dog_age : ℝ) (parrot_age : ℝ)

def ages (cat_age rabbit_age dog_age parrot_age : ℝ) : Prop :=
  cat_age = 8 ∧
  rabbit_age = cat_age / 2 ∧
  dog_age = rabbit_age * 3 ∧
  parrot_age = cat_age + rabbit_age + dog_age

theorem parrot_age_is_24 (cat_age rabbit_age dog_age parrot_age : ℝ) :
  ages cat_age rabbit_age dog_age parrot_age → parrot_age = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parrot_age_is_24_l78_7885


namespace NUMINAMATH_GPT_triangle_enlargement_invariant_l78_7812

theorem triangle_enlargement_invariant (α β γ : ℝ) (h_sum : α + β + γ = 180) (f : ℝ) :
  (α * f ≠ α) ∧ (β * f ≠ β) ∧ (γ * f ≠ γ) → (α * f + β * f + γ * f = 180 * f) → α + β + γ = 180 :=
by
  sorry

end NUMINAMATH_GPT_triangle_enlargement_invariant_l78_7812


namespace NUMINAMATH_GPT_set_subset_condition_l78_7846

theorem set_subset_condition (a : ℝ) :
  (∀ x, (1 < a * x ∧ a * x < 2) → (-1 < x ∧ x < 1)) → (|a| ≥ 2 ∨ a = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_set_subset_condition_l78_7846


namespace NUMINAMATH_GPT_parallelogram_vector_sum_l78_7866

theorem parallelogram_vector_sum (A B C D : ℝ × ℝ) (parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ (C - A = D - B) ∧ (B - D = A - C)) :
  (B - A) + (C - B) = C - A :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_vector_sum_l78_7866


namespace NUMINAMATH_GPT_a_2015_eq_neg6_l78_7895

noncomputable def a : ℕ → ℤ
| 0 => 3
| 1 => 6
| (n+2) => a (n+1) - a n

theorem a_2015_eq_neg6 : a 2015 = -6 := 
by 
  sorry

end NUMINAMATH_GPT_a_2015_eq_neg6_l78_7895


namespace NUMINAMATH_GPT_stripes_distance_l78_7817

theorem stripes_distance (d : ℝ) (L : ℝ) (c : ℝ) (y : ℝ) 
  (hd : d = 40) (hL : L = 50) (hc : c = 15)
  (h_ratio : y / d = c / L) : y = 12 :=
by
  rw [hd, hL, hc] at h_ratio
  sorry

end NUMINAMATH_GPT_stripes_distance_l78_7817


namespace NUMINAMATH_GPT_value_of_x_l78_7886

variable (x y z a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h1 : x * y / (x + y) = a)
variable (h2 : x * z / (x + z) = b)
variable (h3 : y * z / (y + z) = c)

theorem value_of_x : x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end NUMINAMATH_GPT_value_of_x_l78_7886


namespace NUMINAMATH_GPT_shuttle_speed_in_kph_l78_7858

def sec_per_min := 60
def min_per_hour := 60
def sec_per_hour := sec_per_min * min_per_hour
def speed_in_kps := 12
def speed_in_kph := speed_in_kps * sec_per_hour

theorem shuttle_speed_in_kph :
  speed_in_kph = 43200 :=
by
  -- No proof needed
  sorry

end NUMINAMATH_GPT_shuttle_speed_in_kph_l78_7858


namespace NUMINAMATH_GPT_johns_donation_l78_7876

theorem johns_donation (A J : ℝ) 
  (h1 : (75 / 1.5) = A) 
  (h2 : A * 2 = 100)
  (h3 : (100 + J) / 3 = 75) : 
  J = 125 :=
by 
  sorry

end NUMINAMATH_GPT_johns_donation_l78_7876


namespace NUMINAMATH_GPT_domain_tan_2x_plus_pi_over_3_l78_7841

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def domain_tan_transformed : Set ℝ :=
  {x | ∃ k : ℤ, x = k * (Real.pi / 2) + Real.pi / 12}

theorem domain_tan_2x_plus_pi_over_3 :
  (∀ x, ¬ (x ∈ domain_tan)) ↔ (∀ x, ¬ (x ∈ domain_tan_transformed)) :=
by
  sorry

end NUMINAMATH_GPT_domain_tan_2x_plus_pi_over_3_l78_7841


namespace NUMINAMATH_GPT_illegally_parked_percentage_l78_7843

theorem illegally_parked_percentage (total_cars : ℕ) (towed_cars : ℕ)
  (ht : towed_cars = 2 * total_cars / 100) (not_towed_percentage : ℕ)
  (hp : not_towed_percentage = 80) : 
  (100 * (5 * towed_cars) / total_cars) = 10 :=
by
  sorry

end NUMINAMATH_GPT_illegally_parked_percentage_l78_7843


namespace NUMINAMATH_GPT_simplified_expression_value_l78_7865

noncomputable def a : ℝ := Real.sqrt 3 + 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

theorem simplified_expression_value :
  ( (a ^ 2 / (a - b) - (2 * a * b - b ^ 2) / (a - b)) / (a - b) * a * b ) = 2 := by
  sorry

end NUMINAMATH_GPT_simplified_expression_value_l78_7865


namespace NUMINAMATH_GPT_hypotenuse_length_l78_7855

theorem hypotenuse_length (x a b: ℝ) (h1: a = 7) (h2: b = x - 1) (h3: a^2 + b^2 = x^2) : x = 25 :=
by {
  -- Condition h1 states that one leg 'a' is 7 cm.
  -- Condition h2 states that the other leg 'b' is 1 cm shorter than the hypotenuse 'x', i.e., b = x - 1.
  -- Condition h3 is derived from the Pythagorean theorem, i.e., a^2 + b^2 = x^2.
  -- We need to prove that x = 25 cm.
  sorry
}

end NUMINAMATH_GPT_hypotenuse_length_l78_7855


namespace NUMINAMATH_GPT_range_of_f_l78_7830

open Set

noncomputable def f (x : ℝ) : ℝ := 3^x + 5

theorem range_of_f :
  range f = Ioi 5 :=
sorry

end NUMINAMATH_GPT_range_of_f_l78_7830


namespace NUMINAMATH_GPT_luncheon_cost_l78_7832

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 3 * s + 7 * c + p = 3.15)
  (h2 : 4 * s + 10 * c + p = 4.20) :
  s + c + p = 1.05 :=
by sorry

end NUMINAMATH_GPT_luncheon_cost_l78_7832


namespace NUMINAMATH_GPT_sandwiches_left_l78_7857

theorem sandwiches_left 
    (initial_sandwiches : ℕ)
    (first_coworker : ℕ)
    (second_coworker : ℕ)
    (third_coworker : ℕ)
    (kept_sandwiches : ℕ) :
    initial_sandwiches = 50 →
    first_coworker = 4 →
    second_coworker = 3 →
    third_coworker = 2 * first_coworker →
    kept_sandwiches = 3 * second_coworker →
    initial_sandwiches - (first_coworker + second_coworker + third_coworker + kept_sandwiches) = 26 :=
by
  intros h_initial h_first h_second h_third h_kept
  rw [h_initial, h_first, h_second, h_third, h_kept]
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_sandwiches_left_l78_7857


namespace NUMINAMATH_GPT_unique_number_outside_range_f_l78_7861

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_outside_range_f (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : f a b c d 19 = 19) (h6 : f a b c d 97 = 97)
  (h7 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
  ∀ y : ℝ, y ≠ 58 → ∃ x : ℝ, f a b c d x ≠ y :=
sorry

end NUMINAMATH_GPT_unique_number_outside_range_f_l78_7861


namespace NUMINAMATH_GPT_weight_of_one_liter_ghee_brand_b_l78_7815

theorem weight_of_one_liter_ghee_brand_b (wa w_mix : ℕ) (vol_a vol_b : ℕ) (w_mix_total : ℕ) (wb : ℕ) :
  wa = 900 ∧ vol_a = 3 ∧ vol_b = 2 ∧ w_mix = 3360 →
  (vol_a * wa + vol_b * wb = w_mix →
  wb = 330) :=
by
  intros h_eq h_eq2
  obtain ⟨h_wa, h_vol_a, h_vol_b, h_w_mix⟩ := h_eq
  rw [h_wa, h_vol_a, h_vol_b, h_w_mix] at h_eq2
  sorry

end NUMINAMATH_GPT_weight_of_one_liter_ghee_brand_b_l78_7815


namespace NUMINAMATH_GPT_edge_ratio_of_cubes_l78_7802

theorem edge_ratio_of_cubes (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_edge_ratio_of_cubes_l78_7802


namespace NUMINAMATH_GPT_repeating_decimal_addition_l78_7822

def repeating_decimal_45 := (45 / 99 : ℚ)
def repeating_decimal_36 := (36 / 99 : ℚ)

theorem repeating_decimal_addition :
  repeating_decimal_45 + repeating_decimal_36 = 9 / 11 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_addition_l78_7822


namespace NUMINAMATH_GPT_correct_statement_of_abs_l78_7826

theorem correct_statement_of_abs (r : ℚ) :
  ¬ (∀ r : ℚ, abs r > 0) ∧
  ¬ (∀ a b : ℚ, a ≠ b → abs a ≠ abs b) ∧
  (∀ r : ℚ, abs r ≥ 0) ∧
  ¬ (∀ r : ℚ, r < 0 → abs r = -r ∧ abs r < 0 → abs r ≠ -r) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_of_abs_l78_7826


namespace NUMINAMATH_GPT_unique_diff_of_cubes_l78_7869

theorem unique_diff_of_cubes (n k : ℕ) (h : 61 = n^3 - k^3) : n = 5 ∧ k = 4 :=
sorry

end NUMINAMATH_GPT_unique_diff_of_cubes_l78_7869


namespace NUMINAMATH_GPT_sum_of_slopes_correct_l78_7818

noncomputable def sum_of_slopes : ℚ :=
  let Γ1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
  let Γ2 := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 11)^2 = 1}
  let l := {k : ℝ | ∃ p1 ∈ Γ1, ∃ p2 ∈ Γ1, ∃ p3 ∈ Γ2, ∃ p4 ∈ Γ2, p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1.2 = k * p1.1 ∧ p3.2 = k * p3.1}
  let valid_slopes := {k | k ∈ l ∧ (k = 11/10 ∨ k = 1 ∨ k = 5/4)}
  (11 / 10) + 1 + (5 / 4)

theorem sum_of_slopes_correct : sum_of_slopes = 67 / 20 := 
  by sorry

end NUMINAMATH_GPT_sum_of_slopes_correct_l78_7818


namespace NUMINAMATH_GPT_find_k_l78_7899

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) ∧ 
  (∀ x : ℝ, y = k * x + 4) ∧ 
  (1, 5) ∈ { p | ∃ x, p = (x, 2 * x + 3) } ∧ 
  (1, 5) ∈ { q | ∃ x, q = (x, k * x + 4) } → 
  k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l78_7899


namespace NUMINAMATH_GPT_find_a_if_f_is_odd_function_l78_7838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (a * 2^x - 2^(-x))

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_if_f_is_odd_function : 
  ∀ a : ℝ, is_odd_function (f a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_f_is_odd_function_l78_7838


namespace NUMINAMATH_GPT_monotonicity_and_extrema_of_f_l78_7898

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem monotonicity_and_extrema_of_f :
  (∀ (x_1 x_2 : ℝ), x_1 ∈ Set.Icc (-1 : ℝ) 2 → x_2 ∈ Set.Icc (-1 : ℝ) 2 → x_1 < x_2 → f x_1 < f x_2) ∧ 
  (f (-1) = -1) ∧ 
  (f 2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_and_extrema_of_f_l78_7898


namespace NUMINAMATH_GPT_roof_length_width_difference_l78_7879

theorem roof_length_width_difference (w l : ℝ) 
  (h1 : l = 5 * w) 
  (h2 : l * w = 720) : l - w = 48 := 
sorry

end NUMINAMATH_GPT_roof_length_width_difference_l78_7879


namespace NUMINAMATH_GPT_ways_to_score_at_least_7_points_l78_7852

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end NUMINAMATH_GPT_ways_to_score_at_least_7_points_l78_7852


namespace NUMINAMATH_GPT_goldfish_in_first_tank_l78_7889

-- Definitions of conditions
def num_fish_third_tank : Nat := 10
def num_fish_second_tank := 3 * num_fish_third_tank
def num_fish_first_tank := num_fish_second_tank / 2
def goldfish_and_beta_sum (G : Nat) : Prop := G + 8 = num_fish_first_tank

-- Theorem to prove the number of goldfish in the first fish tank
theorem goldfish_in_first_tank (G : Nat) (h : goldfish_and_beta_sum G) : G = 7 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_in_first_tank_l78_7889


namespace NUMINAMATH_GPT_jenni_age_l78_7805

theorem jenni_age (B J : ℕ) (h1 : B + J = 70) (h2 : B - J = 32) : J = 19 :=
by
  sorry

end NUMINAMATH_GPT_jenni_age_l78_7805


namespace NUMINAMATH_GPT_even_function_b_eq_zero_l78_7831

theorem even_function_b_eq_zero (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x) = (x^2 - b * x)) → b = 0 :=
by sorry

end NUMINAMATH_GPT_even_function_b_eq_zero_l78_7831


namespace NUMINAMATH_GPT_equation_satisfying_solution_l78_7842

theorem equation_satisfying_solution (x y : ℤ) :
  (x = 1 ∧ y = 4 → x + 3 * y ≠ 7) ∧
  (x = 2 ∧ y = 1 → x + 3 * y ≠ 7) ∧
  (x = -2 ∧ y = 3 → x + 3 * y = 7) ∧
  (x = 4 ∧ y = 2 → x + 3 * y ≠ 7) :=
by
  sorry

end NUMINAMATH_GPT_equation_satisfying_solution_l78_7842


namespace NUMINAMATH_GPT_lcm_1404_972_l78_7875

def num1 := 1404
def num2 := 972

theorem lcm_1404_972 : Nat.lcm num1 num2 = 88452 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_1404_972_l78_7875


namespace NUMINAMATH_GPT_coordinates_of_point_P_l78_7835

theorem coordinates_of_point_P 
  (P : ℝ × ℝ)
  (h1 : P.1 < 0 ∧ P.2 < 0) 
  (h2 : abs P.2 = 3)
  (h3 : abs P.1 = 5) :
  P = (-5, -3) :=
sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l78_7835


namespace NUMINAMATH_GPT_odd_function_zero_unique_l78_7862

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = - f (- x)

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2

theorem odd_function_zero_unique
  (h_odd : odd_function f)
  (h_func_eq : functional_eq f) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_zero_unique_l78_7862


namespace NUMINAMATH_GPT_olivia_money_left_l78_7864

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end NUMINAMATH_GPT_olivia_money_left_l78_7864


namespace NUMINAMATH_GPT_arc_length_l78_7890

theorem arc_length (circumference : ℝ) (angle : ℝ) (h1 : circumference = 72) (h2 : angle = 45) :
  ∃ length : ℝ, length = 9 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_l78_7890


namespace NUMINAMATH_GPT_cube_of_sum_l78_7851

theorem cube_of_sum :
  (100 + 2) ^ 3 = 1061208 :=
by
  sorry

end NUMINAMATH_GPT_cube_of_sum_l78_7851


namespace NUMINAMATH_GPT_cone_prism_volume_ratio_correct_l78_7810

noncomputable def cone_prism_volume_ratio (π : ℝ) : ℝ :=
  let r := 1.5
  let h := 5
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio_correct (π : ℝ) : 
  cone_prism_volume_ratio π = π / 4.8 :=
sorry

end NUMINAMATH_GPT_cone_prism_volume_ratio_correct_l78_7810


namespace NUMINAMATH_GPT_parabola_directrix_l78_7867

theorem parabola_directrix (x y : ℝ) (h : x^2 = 2 * y) : y = -1 / 2 := 
  sorry

end NUMINAMATH_GPT_parabola_directrix_l78_7867


namespace NUMINAMATH_GPT_middle_integer_is_five_l78_7833

-- Define the conditions of the problem
def consecutive_one_digit_positive_odd_integers (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  a + 2 = b ∧ b + 2 = c ∨ a + 2 = c ∧ c + 2 = b

def sum_is_one_seventh_of_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 7

-- Define the theorem to prove
theorem middle_integer_is_five :
  ∃ (b : ℤ), consecutive_one_digit_positive_odd_integers (b - 2) b (b + 2) ∧
             sum_is_one_seventh_of_product (b - 2) b (b + 2) ∧
             b = 5 :=
sorry

end NUMINAMATH_GPT_middle_integer_is_five_l78_7833


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l78_7845

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l78_7845


namespace NUMINAMATH_GPT_x_eq_one_l78_7837

theorem x_eq_one (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (div_cond : ∀ n : ℕ, 0 < n → (2^n * y + 1) ∣ (x^(2^n) - 1)) : x = 1 := by
  sorry

end NUMINAMATH_GPT_x_eq_one_l78_7837


namespace NUMINAMATH_GPT_students_without_an_A_l78_7800

theorem students_without_an_A :
  ∀ (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (computing_A : ℕ)
    (math_and_history_A : ℕ) (history_and_computing_A : ℕ)
    (math_and_computing_A : ℕ) (all_three_A : ℕ),
  total_students = 40 →
  history_A = 10 →
  math_A = 18 →
  computing_A = 9 →
  math_and_history_A = 5 →
  history_and_computing_A = 3 →
  math_and_computing_A = 4 →
  all_three_A = 2 →
  total_students - (history_A + math_A + computing_A - math_and_history_A - history_and_computing_A - math_and_computing_A + all_three_A) = 13 :=
by
  intros total_students history_A math_A computing_A math_and_history_A history_and_computing_A math_and_computing_A all_three_A 
         ht_total_students ht_history_A ht_math_A ht_computing_A ht_math_and_history_A ht_history_and_computing_A ht_math_and_computing_A ht_all_three_A
  sorry

end NUMINAMATH_GPT_students_without_an_A_l78_7800


namespace NUMINAMATH_GPT_slope_of_line_l78_7836

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 1) (h₂ : y₁ = 3) (h₃ : x₂ = 4) (h₄ : y₂ = -6) : 
  (y₂ - y₁) / (x₂ - x₁) = -3 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_l78_7836


namespace NUMINAMATH_GPT_triangle_iff_inequality_l78_7884

variable {a b c : ℝ}

theorem triangle_iff_inequality :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) := sorry

end NUMINAMATH_GPT_triangle_iff_inequality_l78_7884


namespace NUMINAMATH_GPT_find_h_neg_one_l78_7807

theorem find_h_neg_one (h : ℝ → ℝ) (H : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_h_neg_one_l78_7807


namespace NUMINAMATH_GPT_inverse_proportional_fraction_l78_7887

theorem inverse_proportional_fraction (N : ℝ) (d f : ℝ) (h : N ≠ 0):
  d * f = N :=
sorry

end NUMINAMATH_GPT_inverse_proportional_fraction_l78_7887


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l78_7896

theorem average_of_remaining_numbers (s : ℝ) (a b c d e f : ℝ)
  (h1: (a + b + c + d + e + f) / 6 = 3.95)
  (h2: (a + b) / 2 = 4.4)
  (h3: (c + d) / 2 = 3.85) :
  ((e + f) / 2 = 3.6) :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l78_7896


namespace NUMINAMATH_GPT_green_sequins_per_row_correct_l78_7877

def total_blue_sequins : ℕ := 6 * 8
def total_purple_sequins : ℕ := 5 * 12
def total_green_sequins : ℕ := 162 - (total_blue_sequins + total_purple_sequins)
def green_sequins_per_row : ℕ := total_green_sequins / 9

theorem green_sequins_per_row_correct : green_sequins_per_row = 6 := 
by 
  sorry

end NUMINAMATH_GPT_green_sequins_per_row_correct_l78_7877


namespace NUMINAMATH_GPT_avg_score_calculation_l78_7820

-- Definitions based on the conditions
def directly_proportional (a b : ℝ) : Prop := ∃ k, a = k * b

variables (score_math : ℝ) (score_science : ℝ)
variables (hours_math : ℝ := 4) (hours_science : ℝ := 5)
variables (next_hours_math_science : ℝ := 5)
variables (expected_avg_score : ℝ := 97.5)

axiom h1 : directly_proportional 80 4
axiom h2 : directly_proportional 95 5

-- Define the goal: Expected average score given the study hours next time
theorem avg_score_calculation :
  (score_math / hours_math = score_science / hours_science) →
  (score_math = 100 ∧ score_science = 95) →
  ((next_hours_math_science * score_math / hours_math + next_hours_math_science * score_science / hours_science) / 2 = expected_avg_score) :=
by sorry

end NUMINAMATH_GPT_avg_score_calculation_l78_7820


namespace NUMINAMATH_GPT_equivalent_proof_problem_l78_7874

def math_problem (x y : ℚ) : ℚ :=
((x + y) * (3 * x - y) + y^2) / (-x)

theorem equivalent_proof_problem (hx : x = 4) (hy : y = -(1/4)) :
  math_problem x y = -23 / 2 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l78_7874


namespace NUMINAMATH_GPT_work_problem_solution_l78_7881

theorem work_problem_solution :
  (∃ C: ℝ, 
    B_work_days = 8 ∧ 
    (1 / A_work_rate + 1 / B_work_days + C = 1 / 3) ∧ 
    C = 1 / 8
  ) → 
  A_work_days = 12 :=
by
  sorry

end NUMINAMATH_GPT_work_problem_solution_l78_7881


namespace NUMINAMATH_GPT_village_population_l78_7816

theorem village_population (P : ℕ) (h : 80 * P = 32000 * 100) : P = 40000 :=
sorry

end NUMINAMATH_GPT_village_population_l78_7816


namespace NUMINAMATH_GPT_box_depth_is_10_l78_7825

variable (depth : ℕ)

theorem box_depth_is_10 
  (length width : ℕ)
  (cubes : ℕ)
  (h1 : length = 35)
  (h2 : width = 20)
  (h3 : cubes = 56)
  (h4 : ∃ (cube_size : ℕ), ∀ (c : ℕ), c = cube_size → (length % cube_size = 0 ∧ width % cube_size = 0 ∧ 56 * cube_size^3 = length * width * depth)) :
  depth = 10 :=
by
  sorry

end NUMINAMATH_GPT_box_depth_is_10_l78_7825


namespace NUMINAMATH_GPT_fraction_inequality_solution_set_l78_7871

theorem fraction_inequality_solution_set : 
  {x : ℝ | (2 - x) / (x + 4) > 0} = {x : ℝ | -4 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_fraction_inequality_solution_set_l78_7871


namespace NUMINAMATH_GPT_value_of_2a_plus_b_l78_7808

theorem value_of_2a_plus_b : ∀ (a b : ℝ), (∀ x : ℝ, x^2 - 4*x + 7 = 19 → (x = a ∨ x = b)) → a ≥ b → 2 * a + b = 10 :=
by
  intros a b h_sol h_order
  sorry

end NUMINAMATH_GPT_value_of_2a_plus_b_l78_7808


namespace NUMINAMATH_GPT_total_weight_cashew_nuts_and_peanuts_l78_7813

theorem total_weight_cashew_nuts_and_peanuts (weight_cashew_nuts weight_peanuts : ℕ) (h1 : weight_cashew_nuts = 3) (h2 : weight_peanuts = 2) : 
  weight_cashew_nuts + weight_peanuts = 5 := 
by
  sorry

end NUMINAMATH_GPT_total_weight_cashew_nuts_and_peanuts_l78_7813


namespace NUMINAMATH_GPT_largest_three_digit_perfect_square_and_cube_l78_7863

theorem largest_three_digit_perfect_square_and_cube :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a : ℕ), n = a^6) ∧ ∀ (m : ℕ), ((100 ≤ m ∧ m ≤ 999) ∧ (∃ (b : ℕ), m = b^6)) → m ≤ n := 
by 
  sorry

end NUMINAMATH_GPT_largest_three_digit_perfect_square_and_cube_l78_7863


namespace NUMINAMATH_GPT_least_sum_of_bases_l78_7882

theorem least_sum_of_bases (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : 4 * a + 7 = 7 * b + 4) (h4 : 4 * a + 3 % 7 = 0) :
  a + b = 24 :=
sorry

end NUMINAMATH_GPT_least_sum_of_bases_l78_7882


namespace NUMINAMATH_GPT_part_a_cube_edge_length_part_b_cube_edge_length_l78_7897

-- Part (a)
theorem part_a_cube_edge_length (small_cubes : ℕ) (edge_length_original : ℤ) :
  small_cubes = 512 → edge_length_original^3 = small_cubes → edge_length_original = 8 :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem part_b_cube_edge_length (small_cubes_internal : ℕ) (edge_length_inner : ℤ) (edge_length_original : ℤ) :
  small_cubes_internal = 512 →
  edge_length_inner^3 = small_cubes_internal → 
  edge_length_original = edge_length_inner + 2 →
  edge_length_original = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_part_a_cube_edge_length_part_b_cube_edge_length_l78_7897


namespace NUMINAMATH_GPT_min_value_g_squared_plus_f_l78_7848

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_plus_f (a b c : ℝ) (h : a ≠ 0) 
  (min_f_squared_plus_g : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ 4)
  (exists_x_min : ∃ x : ℝ, (f a b x)^2 + g a c x = 4) :
  ∃ x : ℝ, (g a c x)^2 + f a b x = -9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_g_squared_plus_f_l78_7848


namespace NUMINAMATH_GPT_ratio_of_times_l78_7806

-- Given conditions as definitions
def distance : ℕ := 630 -- distance in km
def previous_time : ℕ := 6 -- time in hours
def new_speed : ℕ := 70 -- speed in km/h

-- Calculation of times
def previous_speed : ℕ := distance / previous_time

def new_time : ℕ := distance / new_speed

-- Main theorem statement
theorem ratio_of_times :
  (new_time : ℚ) / (previous_time : ℚ) = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_ratio_of_times_l78_7806


namespace NUMINAMATH_GPT_compare_powers_l78_7828

theorem compare_powers (a b c d : ℝ) (h1 : a + b = 0) (h2 : c + d = 0) : a^5 + d^6 = c^6 - b^5 :=
by
  sorry

end NUMINAMATH_GPT_compare_powers_l78_7828


namespace NUMINAMATH_GPT_circle_through_and_tangent_l78_7811

noncomputable def circle_eq (a b r : ℝ) (x y : ℝ) : ℝ :=
  (x - a) ^ 2 + (y - b) ^ 2 - r ^ 2

theorem circle_through_and_tangent
(h1 : circle_eq 1 2 2 1 0 = 0)
(h2 : ∀ x y, circle_eq 1 2 2 x y = 0 → (x = 1 → y = 2 ∨ y = -2))
: ∀ x y, circle_eq 1 2 2 x y = 0 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_through_and_tangent_l78_7811


namespace NUMINAMATH_GPT_functional_equation_initial_condition_unique_f3_l78_7844

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : f (f x + y) = f (x ^ 2 - y) + 2 * f x * y := sorry

theorem initial_condition : f 1 = 1 := sorry

theorem unique_f3 : f 3 = 9 := sorry

end NUMINAMATH_GPT_functional_equation_initial_condition_unique_f3_l78_7844


namespace NUMINAMATH_GPT_travel_ways_A_to_C_l78_7823

-- We define the number of ways to travel from A to B
def ways_A_to_B : ℕ := 3

-- We define the number of ways to travel from B to C
def ways_B_to_C : ℕ := 2

-- We state the problem as a theorem
theorem travel_ways_A_to_C : ways_A_to_B * ways_B_to_C = 6 :=
by
  sorry

end NUMINAMATH_GPT_travel_ways_A_to_C_l78_7823


namespace NUMINAMATH_GPT_total_money_amount_l78_7850

-- Define the conditions
def num_bills : ℕ := 3
def value_per_bill : ℕ := 20
def initial_amount : ℕ := 75

-- Define the statement about the total amount of money James has
theorem total_money_amount : num_bills * value_per_bill + initial_amount = 135 := 
by 
  -- Since the proof is not required, we use 'sorry' to skip it
  sorry

end NUMINAMATH_GPT_total_money_amount_l78_7850


namespace NUMINAMATH_GPT_exists_int_squares_l78_7873

theorem exists_int_squares (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  ∃ x y : ℤ, (a^2 + b^2)^n = x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_int_squares_l78_7873


namespace NUMINAMATH_GPT_points_on_inverse_proportion_l78_7847

theorem points_on_inverse_proportion (y_1 y_2 : ℝ) :
  (2:ℝ) = 5 / y_1 → (3:ℝ) = 5 / y_2 → y_1 > y_2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_points_on_inverse_proportion_l78_7847


namespace NUMINAMATH_GPT_article_cost_price_l78_7819

theorem article_cost_price :
  ∃ C : ℝ, 
  (1.05 * C) - 2 = (1.045 * C) ∧ 
  ∃ C_new : ℝ, C_new = (0.95 * C) ∧ ((1.045 * C) = (C_new + 0.1 * C_new)) ∧ C = 400 := 
sorry

end NUMINAMATH_GPT_article_cost_price_l78_7819


namespace NUMINAMATH_GPT_choir_members_count_l78_7854

theorem choir_members_count (n : ℕ) 
  (h1 : 150 < n) 
  (h2 : n < 300) 
  (h3 : n % 6 = 1) 
  (h4 : n % 8 = 3) 
  (h5 : n % 9 = 2) : 
  n = 163 :=
sorry

end NUMINAMATH_GPT_choir_members_count_l78_7854


namespace NUMINAMATH_GPT_product_gcd_lcm_24_60_l78_7878

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_24_60_l78_7878


namespace NUMINAMATH_GPT_find_number_of_breeding_rabbits_l78_7849

def breeding_rabbits_condition (B : ℕ) : Prop :=
  ∃ (kittens_first_spring remaining_kittens_first_spring kittens_second_spring remaining_kittens_second_spring : ℕ),
    kittens_first_spring = 10 * B ∧
    remaining_kittens_first_spring = 5 * B + 5 ∧
    kittens_second_spring = 60 ∧
    remaining_kittens_second_spring = kittens_second_spring - 4 ∧
    B + remaining_kittens_first_spring + remaining_kittens_second_spring = 121

theorem find_number_of_breeding_rabbits (B : ℕ) : breeding_rabbits_condition B → B = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_breeding_rabbits_l78_7849


namespace NUMINAMATH_GPT_part_a_part_b_l78_7892

theorem part_a (p : ℕ) (hp : Nat.Prime p) (a b : ℤ) (h : a ≡ b [ZMOD p]) : a ^ p ≡ b ^ p [ZMOD p^2] :=
  sorry

theorem part_b (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n | n ∈ Finset.range (p^2) ∧ ∃ x, x ^ p ≡ n [ZMOD p^2] } = p :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_l78_7892


namespace NUMINAMATH_GPT_wuzhen_conference_arrangements_l78_7860

theorem wuzhen_conference_arrangements 
  (countries : Finset ℕ)
  (hotels : Finset ℕ)
  (h_countries_count : countries.card = 5)
  (h_hotels_count : hotels.card = 3) :
  ∃ f : ℕ → ℕ,
  (∀ c ∈ countries, f c ∈ hotels) ∧
  (∀ h ∈ hotels, ∃ c ∈ countries, f c = h) ∧
  (Finset.card (Set.toFinset (f '' countries)) = 3) ∧
  ∃ n : ℕ,
  n = 150 := 
sorry

end NUMINAMATH_GPT_wuzhen_conference_arrangements_l78_7860


namespace NUMINAMATH_GPT_vector_calculation_l78_7829

namespace VectorProof

variables (a b : ℝ × ℝ) (m : ℝ)

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k • v2)

theorem vector_calculation
  (h₁ : a = (1, -2))
  (h₂ : b = (m, 4))
  (h₃ : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end VectorProof

end NUMINAMATH_GPT_vector_calculation_l78_7829


namespace NUMINAMATH_GPT_find_C_and_D_l78_7888

theorem find_C_and_D (C D : ℚ) :
  (∀ x : ℚ, ((6 * x - 8) / (2 * x^2 + 5 * x - 3) = (C / (x - 1)) + (D / (2 * x + 3)))) →
  (2*x^2 + 5*x - 3 = (2*x - 1)*(x + 3)) →
  (∀ x : ℚ, ((C*(2*x + 3) + D*(x - 1)) / ((2*x - 1)*(x + 3))) = ((6*x - 8) / ((2*x - 1)*(x + 3)))) →
  (∀ x : ℚ, C*(2*x + 3) + D*(x - 1) = 6*x - 8) →
  C = -2/5 ∧ D = 34/5 := 
by 
  sorry

end NUMINAMATH_GPT_find_C_and_D_l78_7888


namespace NUMINAMATH_GPT_smallest_number_l78_7804

theorem smallest_number (a b c d e : ℕ) (h₁ : a = 12) (h₂ : b = 16) (h₃ : c = 18) (h₄ : d = 21) (h₅ : e = 28) : 
    ∃ n : ℕ, (n - 4) % Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 0 ∧ n = 1012 :=
by
    sorry

end NUMINAMATH_GPT_smallest_number_l78_7804


namespace NUMINAMATH_GPT_initial_caps_correct_l78_7827

variable (bought : ℕ)
variable (total : ℕ)

def initial_bottle_caps (bought : ℕ) (total : ℕ) : ℕ :=
  total - bought

-- Given conditions
def bought_caps : ℕ := 7
def total_caps : ℕ := 47

theorem initial_caps_correct : initial_bottle_caps bought_caps total_caps = 40 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_initial_caps_correct_l78_7827
