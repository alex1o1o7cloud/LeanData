import Mathlib

namespace a_2017_eq_2_l237_23791

variable (n : ℕ)
variable (S : ℕ → ℤ)

/-- Define the sequence sum Sn -/
def S_n (n : ℕ) : ℤ := 2 * n - 1

/-- Define the sequence term an -/
def a_n (n : ℕ) : ℤ := S_n n - S_n (n - 1)

theorem a_2017_eq_2 : a_n 2017 = 2 := 
by
  have hSn : ∀ n, S_n n = (2 * n - 1) := by intro; simp [S_n] 
  have ha : ∀ n, a_n n = (S_n n - S_n (n - 1)) := by intro; simp [a_n]
  simp only [ha, hSn] 
  sorry

end a_2017_eq_2_l237_23791


namespace ann_age_l237_23756

theorem ann_age {a b y : ℕ} (h1 : a + b = 44) (h2 : y = a - b) (h3 : b = a / 2 + 2 * (a - b)) : a = 24 :=
by
  sorry

end ann_age_l237_23756


namespace equal_roots_quadratic_k_eq_one_l237_23798

theorem equal_roots_quadratic_k_eq_one
  (k : ℝ)
  (h : ∃ x : ℝ, x^2 - 2 * x + k == 0 ∧ x^2 - 2 * x + k == 0) :
  k = 1 :=
by {
  sorry
}

end equal_roots_quadratic_k_eq_one_l237_23798


namespace cows_black_more_than_half_l237_23717

theorem cows_black_more_than_half (t b : ℕ) (h1 : t = 18) (h2 : t - 4 = b) : b - t / 2 = 5 :=
by
  sorry

end cows_black_more_than_half_l237_23717


namespace count_multiples_200_to_400_l237_23700

def count_multiples_in_range (a b n : ℕ) : ℕ :=
  (b / n) - ((a + n - 1) / n) + 1

theorem count_multiples_200_to_400 :
  count_multiples_in_range 200 400 78 = 3 :=
by
  sorry

end count_multiples_200_to_400_l237_23700


namespace royal_children_count_l237_23767

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end royal_children_count_l237_23767


namespace initial_books_count_l237_23770

-- Definitions in conditions
def books_sold : ℕ := 42
def books_left : ℕ := 66

-- The theorem to prove the initial books count
theorem initial_books_count (initial_books : ℕ) : initial_books = books_sold + books_left :=
  by sorry

end initial_books_count_l237_23770


namespace total_cost_is_90_l237_23714

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end total_cost_is_90_l237_23714


namespace find_circle_center_l237_23780

theorem find_circle_center
  (x y : ℝ)
  (h1 : 5 * x - 4 * y = 10)
  (h2 : 3 * x - y = 0)
  : x = -10 / 7 ∧ y = -30 / 7 :=
by {
  sorry
}

end find_circle_center_l237_23780


namespace mrs_hilt_apple_pies_l237_23724

-- Given definitions
def total_pies := 30 * 5
def pecan_pies := 16

-- The number of apple pies
def apple_pies := total_pies - pecan_pies

-- The proof statement
theorem mrs_hilt_apple_pies : apple_pies = 134 :=
by
  sorry -- Proof step to be filled

end mrs_hilt_apple_pies_l237_23724


namespace find_angle_ACD_l237_23705

-- Define the vertices of the quadrilateral
variables {A B C D : Type*}

-- Given angles and side equality
variables (angle_DAC : ℝ) (angle_DBC : ℝ) (angle_BCD : ℝ) (eq_BC_AD : Prop)

-- The given conditions in the problem
axiom angle_DAC_is_98 : angle_DAC = 98
axiom angle_DBC_is_82 : angle_DBC = 82
axiom angle_BCD_is_70 : angle_BCD = 70
axiom BC_eq_AD : eq_BC_AD = true

-- Target angle to be proven
def angle_ACD : ℝ := 28

-- The theorem
theorem find_angle_ACD (h1 : angle_DAC = 98)
                       (h2 : angle_DBC = 82)
                       (h3 : angle_BCD = 70)
                       (h4 : eq_BC_AD) : angle_ACD = 28 := 
by
  sorry  -- Proof of the theorem

end find_angle_ACD_l237_23705


namespace john_must_work_10_more_days_l237_23771

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l237_23771


namespace number_of_fours_is_even_l237_23746

theorem number_of_fours_is_even 
  (x y z : ℕ) 
  (h1 : x + y + z = 80) 
  (h2 : 3 * x + 4 * y + 5 * z = 276) : 
  Even y :=
by
  sorry

end number_of_fours_is_even_l237_23746


namespace probability_X_eq_2_l237_23759

namespace Hypergeometric

def combin (n k : ℕ) : ℕ := n.choose k

noncomputable def hypergeometric (N M n k : ℕ) : ℚ :=
  (combin M k * combin (N - M) (n - k)) / combin N n

theorem probability_X_eq_2 :
  hypergeometric 8 5 3 2 = 15 / 28 := by
  sorry

end Hypergeometric

end probability_X_eq_2_l237_23759


namespace total_surface_area_of_resulting_solid_is_12_square_feet_l237_23706

noncomputable def height_of_D :=
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  2 - (h_A + h_B + h_C)

theorem total_surface_area_of_resulting_solid_is_12_square_feet :
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  let h_D := 2 - (h_A + h_B + h_C)
  let top_and_bottom_area := 4 * 2
  let side_area := 2 * (h_A + h_B + h_C + h_D)
  top_and_bottom_area + side_area = 12 := by
  sorry

end total_surface_area_of_resulting_solid_is_12_square_feet_l237_23706


namespace candy_cost_correct_l237_23745

-- Given conditions:
def given_amount : ℝ := 1.00
def change_received : ℝ := 0.46

-- Define candy cost based on given conditions
def candy_cost : ℝ := given_amount - change_received

-- Statement to be proved
theorem candy_cost_correct : candy_cost = 0.54 := 
by
  sorry

end candy_cost_correct_l237_23745


namespace ratio_of_weights_l237_23715

variable (x y : ℝ)

theorem ratio_of_weights (h : x + y = 7 * (x - y)) (h1 : x > y) : x / y = 4 / 3 :=
sorry

end ratio_of_weights_l237_23715


namespace building_height_l237_23787

theorem building_height
    (flagpole_height : ℝ)
    (flagpole_shadow_length : ℝ)
    (building_shadow_length : ℝ)
    (h : ℝ)
    (h_eq : flagpole_height / flagpole_shadow_length = h / building_shadow_length)
    (flagpole_height_eq : flagpole_height = 18)
    (flagpole_shadow_length_eq : flagpole_shadow_length = 45)
    (building_shadow_length_eq : building_shadow_length = 65) :
  h = 26 := by
  sorry

end building_height_l237_23787


namespace tiles_painted_in_15_minutes_l237_23782

open Nat

theorem tiles_painted_in_15_minutes:
  let don_rate := 3
  let ken_rate := don_rate + 2
  let laura_rate := 2 * ken_rate
  let kim_rate := laura_rate - 3
  don_rate + ken_rate + laura_rate + kim_rate == 25 → 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = 375 :=
by
  intros
  sorry

end tiles_painted_in_15_minutes_l237_23782


namespace rectangle_in_triangle_area_l237_23776

theorem rectangle_in_triangle_area (b h : ℕ) (hb : b = 12) (hh : h = 8)
  (x : ℕ) (hx : x = h / 2) : (b * x / 2) = 48 := 
by
  sorry

end rectangle_in_triangle_area_l237_23776


namespace solve_for_x_l237_23774

theorem solve_for_x (x : ℝ) (h : 40 / x - 1 = 19) : x = 2 :=
by {
  sorry
}

end solve_for_x_l237_23774


namespace sequences_properties_l237_23737

-- Definitions for properties P and P'
def is_property_P (seq : List ℕ) : Prop := sorry
def is_property_P' (seq : List ℕ) : Prop := sorry

-- Define sequences
def sequence1 := [1, 2, 3, 1]
def sequence2 := [1, 234, 5]  -- Extend as needed

-- Conditions
def bn_is_permutation_of_an (a b : List ℕ) : Prop := sorry -- Placeholder for permutation check

-- Main Statement 
theorem sequences_properties :
  is_property_P sequence1 ∧
  is_property_P' sequence2 := 
by
  sorry

-- Additional theorem to check permutation if needed
-- theorem permutation_check :
--  bn_is_permutation_of_an sequence1 sequence2 :=
-- by
--  sorry

end sequences_properties_l237_23737


namespace Tonya_buys_3_lego_sets_l237_23735

-- Definitions based on conditions
def num_sisters : Nat := 2
def num_dolls : Nat := 4
def price_per_doll : Nat := 15
def price_per_lego_set : Nat := 20

-- The amount of money spent on each sister should be the same
def amount_spent_on_younger_sister := num_dolls * price_per_doll
def amount_spent_on_older_sister := (amount_spent_on_younger_sister / price_per_lego_set)

-- Proof statement
theorem Tonya_buys_3_lego_sets : amount_spent_on_older_sister = 3 :=
by
  sorry

end Tonya_buys_3_lego_sets_l237_23735


namespace sqrt_37_between_6_and_7_l237_23793

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := 
by 
  have h₁ : Real.sqrt 36 = 6 := by sorry
  have h₂ : Real.sqrt 49 = 7 := by sorry
  sorry

end sqrt_37_between_6_and_7_l237_23793


namespace gardener_hourly_wage_l237_23716

-- Conditions
def rose_bushes_count : Nat := 20
def cost_per_rose_bush : Nat := 150
def hours_per_day : Nat := 5
def days_worked : Nat := 4
def soil_volume : Nat := 100
def cost_per_cubic_foot_soil : Nat := 5
def total_cost : Nat := 4100

-- Theorem statement
theorem gardener_hourly_wage :
  let cost_of_rose_bushes := rose_bushes_count * cost_per_rose_bush
  let cost_of_soil := soil_volume * cost_per_cubic_foot_soil
  let total_material_cost := cost_of_rose_bushes + cost_of_soil
  let labor_cost := total_cost - total_material_cost
  let total_hours_worked := hours_per_day * days_worked
  (labor_cost / total_hours_worked) = 30 := 
by {
  -- Proof placeholder
  sorry
}

end gardener_hourly_wage_l237_23716


namespace second_crane_height_l237_23738

noncomputable def height_of_second_crane : ℝ :=
  let crane1 := 228
  let building1 := 200
  let building2 := 100
  let crane3 := 147
  let building3 := 140
  let avg_building_height := (building1 + building2 + building3) / 3
  let avg_crane_height := avg_building_height * 1.13
  let h := (avg_crane_height * 3) - (crane1 - building1 + crane3 - building3) + building2
  h

theorem second_crane_height : height_of_second_crane = 122 := 
  sorry

end second_crane_height_l237_23738


namespace truck_distance_l237_23769

theorem truck_distance (d: ℕ) (g: ℕ) (eff: ℕ) (new_g: ℕ) (total_distance: ℕ)
  (h1: d = 300) (h2: g = 10) (h3: eff = d / g) (h4: new_g = 15) (h5: total_distance = eff * new_g):
  total_distance = 450 :=
sorry

end truck_distance_l237_23769


namespace find_integers_l237_23702

theorem find_integers (a b m : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 :=
by
  sorry

end find_integers_l237_23702


namespace complement_of_A_in_U_l237_23781

variable {U : Set ℤ}
variable {A : Set ℤ}

theorem complement_of_A_in_U (hU : U = {-1, 0, 1}) (hA : A = {0, 1}) : U \ A = {-1} := by
  sorry

end complement_of_A_in_U_l237_23781


namespace bag_with_cracks_number_l237_23766

def marbles : List ℕ := [18, 19, 21, 23, 25, 34]

def total_marbles : ℕ := marbles.sum

def modulo_3 (n : ℕ) : ℕ := n % 3

theorem bag_with_cracks_number :
  ∃ (c : ℕ), c ∈ marbles ∧ 
    (total_marbles - c) % 3 = 0 ∧
    c = 23 :=
by 
  sorry

end bag_with_cracks_number_l237_23766


namespace extreme_points_inequality_l237_23752

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log (1 - x)

theorem extreme_points_inequality (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1 / 4) 
  (h_sum : x1 + x2 = 1) (h_prod : x1 * x2 = a) (h_order : x1 < x2) :
  f x2 a - x1 > -(3 + Real.log 4) / 8 := 
by
  -- proof needed
  sorry

end extreme_points_inequality_l237_23752


namespace largest_n_exists_unique_k_l237_23719

theorem largest_n_exists_unique_k (n k : ℕ) :
  (∃! k, (8 : ℚ) / 15 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 7 / 13) →
  n ≤ 112 :=
sorry

end largest_n_exists_unique_k_l237_23719


namespace triangle_area_ABC_l237_23788

variable {A : Prod ℝ ℝ}
variable {B : Prod ℝ ℝ}
variable {C : Prod ℝ ℝ}

noncomputable def area_of_triangle (A B C : Prod ℝ ℝ ) : ℝ :=
  (1 / 2) * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))

theorem triangle_area_ABC : 
  ∀ {A B C : Prod ℝ ℝ}, 
  A = (2, 3) → 
  B = (5, 7) → 
  C = (6, 1) → 
  area_of_triangle A B C = 11 
:= by
  intros
  subst_vars
  simp [area_of_triangle]
  sorry

end triangle_area_ABC_l237_23788


namespace minutes_in_hours_l237_23734

theorem minutes_in_hours (h : ℝ) (m : ℝ) (H : h = 3.5) (M : m = 60) : h * m = 210 := by
  sorry

end minutes_in_hours_l237_23734


namespace inscribed_circle_radius_l237_23797

theorem inscribed_circle_radius (A p r s : ℝ) (h₁ : A = 2 * p) (h₂ : p = 2 * s) (h₃ : A = r * s) : r = 4 :=
by sorry

end inscribed_circle_radius_l237_23797


namespace range_of_m_l237_23755

noncomputable def A (x : ℝ) : Prop := |x - 2| ≤ 4
noncomputable def B (x : ℝ) (m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, (¬A x) → (¬B x m)) ∧ (∃ x, (¬B x m) ∧ ¬(¬A x)) → m ≥ 5 :=
sorry

end range_of_m_l237_23755


namespace minimum_value_of_fraction_l237_23768

theorem minimum_value_of_fraction (a b : ℝ) (h1 : a > 2 * b) (h2 : 2 * b > 0) :
  (a^4 + 1) / (b * (a - 2 * b)) >= 16 :=
sorry

end minimum_value_of_fraction_l237_23768


namespace quadratic_has_real_roots_l237_23789

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 := by
  sorry

end quadratic_has_real_roots_l237_23789


namespace maximum_marks_l237_23707

noncomputable def passing_mark (M : ℝ) : ℝ := 0.35 * M

theorem maximum_marks (M : ℝ) (h1 : passing_mark M = 210) : M = 600 :=
  by
  sorry

end maximum_marks_l237_23707


namespace linear_system_solution_l237_23777

theorem linear_system_solution (k x y : ℝ) (h₁ : x + y = 5 * k) (h₂ : x - y = 9 * k) (h₃ : 2 * x + 3 * y = 6) :
  k = 3 / 4 :=
by
  sorry

end linear_system_solution_l237_23777


namespace product_of_roots_eq_neg25_l237_23794

theorem product_of_roots_eq_neg25 : 
  ∀ (x : ℝ), 24 * x^2 + 36 * x - 600 = 0 → x * (x - ((-36 - 24 * x)/24)) = -25 :=
by
  sorry

end product_of_roots_eq_neg25_l237_23794


namespace age_of_new_person_l237_23712

-- Definitions based on conditions
def initial_avg : ℕ := 15
def new_avg : ℕ := 17
def n : ℕ := 9

-- Statement to prove
theorem age_of_new_person : 
    ∃ (A : ℕ), (initial_avg * n + A) / (n + 1) = new_avg ∧ A = 35 := 
by {
    -- Proof steps would go here, but since they are not required, we add 'sorry' to skip the proof
    sorry
}

end age_of_new_person_l237_23712


namespace latest_time_for_60_degrees_l237_23778

def temperature_at_time (t : ℝ) : ℝ :=
  -2 * t^2 + 16 * t + 40

theorem latest_time_for_60_degrees (t : ℝ) :
  temperature_at_time t = 60 → t = 5 :=
sorry

end latest_time_for_60_degrees_l237_23778


namespace prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l237_23701

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_gt_3_div_24 (p : ℕ) (hp : is_prime p) (h : p > 3) : 
  24 ∣ (p^2 - 1) :=
sorry

theorem num_form_6n_plus_minus_1_div_24 (n : ℕ) : 
  24 ∣ (6 * n + 1)^2 - 1 ∧ 24 ∣ (6 * n - 1)^2 - 1 :=
sorry

end prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l237_23701


namespace Ceva_theorem_l237_23708

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

end Ceva_theorem_l237_23708


namespace curve_representation_l237_23757

def curve_set (x y : Real) : Prop := 
  ((x + y - 1) * Real.sqrt (x^2 + y^2 - 4) = 0)

def line_set (x y : Real) : Prop :=
  (x + y - 1 = 0) ∧ (x^2 + y^2 ≥ 4)

def circle_set (x y : Real) : Prop :=
  (x^2 + y^2 = 4)

theorem curve_representation (x y : Real) :
  curve_set x y ↔ (line_set x y ∨ circle_set x y) :=
sorry

end curve_representation_l237_23757


namespace first_motorcyclist_laps_per_hour_l237_23748

noncomputable def motorcyclist_laps (x y z : ℝ) (P1 : 0 < x - y) (P2 : 0 < x - z) (P3 : 0 < y - z) : Prop :=
  (4.5 / (x - y) = 4.5) ∧ (4.5 / (x - z) = 4.5 - 0.5) ∧ (3 / (y - z) = 3) → x = 3

theorem first_motorcyclist_laps_per_hour (x y z : ℝ) (P1: 0 < x - y) (P2: 0 < x - z) (P3: 0 < y - z) :
  motorcyclist_laps x y z P1 P2 P3 →
  x = 3 :=
sorry

end first_motorcyclist_laps_per_hour_l237_23748


namespace crayons_count_l237_23711

-- Define the initial number of crayons
def initial_crayons : ℕ := 1453

-- Define the number of crayons given away
def crayons_given_away : ℕ := 563

-- Define the number of crayons lost
def crayons_lost : ℕ := 558

-- Define the final number of crayons left
def final_crayons_left : ℕ := initial_crayons - crayons_given_away - crayons_lost

-- State that the final number of crayons left is 332
theorem crayons_count : final_crayons_left = 332 :=
by
    -- This is where the proof would go, which we're skipping with sorry
    sorry

end crayons_count_l237_23711


namespace b_can_finish_work_in_15_days_l237_23739

theorem b_can_finish_work_in_15_days (W : ℕ) (r_A : ℕ) (r_B : ℕ) (h1 : r_A = W / 21) (h2 : 10 * r_B + 7 * r_A / 21 = W) : r_B = W / 15 :=
by sorry

end b_can_finish_work_in_15_days_l237_23739


namespace xyz_sum_divisible_l237_23758

-- Define variables and conditions
variable (p x y z : ℕ) [Fact (Prime p)]
variable (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
variable (h_eq1 : x^3 % p = y^3 % p)
variable (h_eq2 : y^3 % p = z^3 % p)

-- Theorem statement
theorem xyz_sum_divisible (p x y z : ℕ) [Fact (Prime p)]
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
  (h_eq1 : x^3 % p = y^3 % p)
  (h_eq2 : y^3 % p = z^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := 
  sorry

end xyz_sum_divisible_l237_23758


namespace min_value_of_vector_sum_l237_23741

noncomputable def min_vector_sum_magnitude (P Q: (ℝ×ℝ)) : ℝ :=
  let x := P.1
  let y := P.2
  let a := Q.1
  let b := Q.2
  Real.sqrt ((x + a)^2 + (y + b)^2)

theorem min_value_of_vector_sum :
  ∃ P Q, 
  (P.1 - 2)^2 + (P.2 - 2)^2 = 1 ∧ 
  Q.1 + Q.2 = 1 ∧ 
  min_vector_sum_magnitude P Q = (5 * Real.sqrt 2 - 2) / 2 :=
by
  sorry

end min_value_of_vector_sum_l237_23741


namespace thomas_score_l237_23761

def average (scores : List ℕ) : ℚ := scores.sum / scores.length

variable (scores : List ℕ)

theorem thomas_score (h_length : scores.length = 19)
                     (h_avg_before : average scores = 78)
                     (h_avg_after : average ((98 :: scores)) = 79) :
  let thomas_score := 98
  thomas_score = 98 := sorry

end thomas_score_l237_23761


namespace fishing_problem_l237_23733

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l237_23733


namespace functional_relationship_l237_23747

-- Define the conditions and question for Scenario ①
def scenario1 (x y k : ℝ) (h1 : k ≠ 0) : Prop :=
  y = k / x

-- Define the conditions and question for Scenario ②
def scenario2 (n S k : ℝ) (h2 : k ≠ 0) : Prop :=
  S = k / n

-- Define the conditions and question for Scenario ③
def scenario3 (t s k : ℝ) (h3 : k ≠ 0) : Prop :=
  s = k * t

-- The main theorem
theorem functional_relationship (x y n S t s k : ℝ) (h1 : k ≠ 0) :
  (scenario1 x y k h1) ∧ (scenario2 n S k h1) ∧ ¬(scenario3 t s k h1) := 
sorry

end functional_relationship_l237_23747


namespace playback_methods_proof_l237_23795

/-- A TV station continuously plays 5 advertisements, consisting of 3 different commercial advertisements
and 2 different Olympic promotional advertisements. The requirements are:
  1. The last advertisement must be an Olympic promotional advertisement.
  2. The 2 Olympic promotional advertisements can be played consecutively.
-/
def number_of_playback_methods (commercials olympics: ℕ) (last_ad_olympic: Bool) (olympics_consecutive: Bool) : ℕ :=
  if commercials = 3 ∧ olympics = 2 ∧ last_ad_olympic ∧ olympics_consecutive then 36 else 0

theorem playback_methods_proof :
  number_of_playback_methods 3 2 true true = 36 := by
  sorry

end playback_methods_proof_l237_23795


namespace denominator_of_fractions_l237_23792

theorem denominator_of_fractions (y a : ℝ) (hy : y > 0) 
  (h : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 :=
by
  sorry

end denominator_of_fractions_l237_23792


namespace pq_conditions_l237_23796

theorem pq_conditions (p q : ℝ) (hp : p > 1) (hq : q > 1) (hq_inverse : 1 / p + 1 / q = 1) (hpq : p * q = 9) :
  (p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨ (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2) :=
  sorry

end pq_conditions_l237_23796


namespace inequality_true_l237_23704

theorem inequality_true (a b : ℝ) (h : a > b) : (2 * a - 1) > (2 * b - 1) :=
by {
  sorry
}

end inequality_true_l237_23704


namespace number_of_gigs_played_l237_23764

-- Definitions based on given conditions
def earnings_per_member : ℕ := 20
def number_of_members : ℕ := 4
def total_earnings : ℕ := 400

-- Proof statement in Lean 4
theorem number_of_gigs_played : (total_earnings / (earnings_per_member * number_of_members)) = 5 :=
by
  sorry

end number_of_gigs_played_l237_23764


namespace students_wearing_other_colors_l237_23790

-- Definitions according to the problem conditions
def total_students : ℕ := 900
def percentage_blue : ℕ := 44
def percentage_red : ℕ := 28
def percentage_green : ℕ := 10

-- Goal: Prove the number of students who wear other colors
theorem students_wearing_other_colors :
  (total_students * (100 - (percentage_blue + percentage_red + percentage_green))) / 100 = 162 :=
by
  -- Skipping the proof steps with sorry
  sorry

end students_wearing_other_colors_l237_23790


namespace at_least_one_nonnegative_l237_23709

theorem at_least_one_nonnegative (x : ℝ) (a b : ℝ) (h1 : a = x^2 - 1) (h2 : b = 4 * x + 5) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end at_least_one_nonnegative_l237_23709


namespace find_cans_lids_l237_23753

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l237_23753


namespace simplify_triangle_expression_l237_23751

theorem simplify_triangle_expression (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c :=
by
  sorry

end simplify_triangle_expression_l237_23751


namespace total_paths_from_X_to_Z_l237_23772

variable (X Y Z : Type)
variables (f : X → Y → Z)
variables (g : X → Z)

-- Conditions
def paths_X_to_Y : ℕ := 3
def paths_Y_to_Z : ℕ := 4
def direct_paths_X_to_Z : ℕ := 1

-- Proof problem statement
theorem total_paths_from_X_to_Z : paths_X_to_Y * paths_Y_to_Z + direct_paths_X_to_Z = 13 := sorry

end total_paths_from_X_to_Z_l237_23772


namespace smallest_gcd_of_lcm_eq_square_diff_l237_23718

theorem smallest_gcd_of_lcm_eq_square_diff (x y : ℕ) (h : Nat.lcm x y = (x - y) ^ 2) : Nat.gcd x y = 2 :=
sorry

end smallest_gcd_of_lcm_eq_square_diff_l237_23718


namespace red_balls_in_bag_l237_23744

theorem red_balls_in_bag (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total_balls = 60 → 
  white_balls = 22 → 
  green_balls = 18 → 
  yellow_balls = 8 → 
  purple_balls = 7 → 
  prob_neither_red_nor_purple = 0.8 → 
  ( ∃ (red_balls : ℕ), red_balls = 5 ) :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end red_balls_in_bag_l237_23744


namespace jogger_distance_ahead_l237_23742

theorem jogger_distance_ahead
  (train_speed_km_hr : ℝ) (jogger_speed_km_hr : ℝ)
  (train_length_m : ℝ) (time_seconds : ℝ)
  (relative_speed_m_s : ℝ) (distance_covered_m : ℝ)
  (D : ℝ)
  (h1 : train_speed_km_hr = 45)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_length_m = 100)
  (h4 : time_seconds = 25)
  (h5 : relative_speed_m_s = 36 * (5/18))
  (h6 : distance_covered_m = 10 * 25)
  (h7 : D + train_length_m = distance_covered_m) :
  D = 150 :=
by sorry

end jogger_distance_ahead_l237_23742


namespace hundred_squared_plus_two_hundred_one_is_composite_l237_23730

theorem hundred_squared_plus_two_hundred_one_is_composite : 
    ¬ Prime (100^2 + 201) :=
by {
  sorry
}

end hundred_squared_plus_two_hundred_one_is_composite_l237_23730


namespace cosine_F_in_triangle_DEF_l237_23762

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l237_23762


namespace triangle_is_isosceles_l237_23729

-- lean statement
theorem triangle_is_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) : 
  ∃ k : ℝ, a = k ∧ b = k := 
sorry

end triangle_is_isosceles_l237_23729


namespace simplest_fraction_l237_23723

theorem simplest_fraction (x y : ℝ) (h1 : 2 * x ≠ 0) (h2 : x + y ≠ 0) :
  let A := (2 * x) / (4 * x^2)
  let B := (x^2 + y^2) / (x + y)
  let C := (x^2 + 2 * x + 1) / (x + 1)
  let D := (x^2 - 4) / (x + 2)
  B = (x^2 + y^2) / (x + y) ∧
  A ≠ (2 * x) / (4 * x^2) ∧
  C ≠ (x^2 + 2 * x + 1) / (x + 1) ∧
  D ≠ (x^2 - 4) / (x + 2) := sorry

end simplest_fraction_l237_23723


namespace contrapositive_proposition_l237_23703

theorem contrapositive_proposition (x : ℝ) : (x > 10 → x > 1) ↔ (x ≤ 1 → x ≤ 10) :=
by
  sorry

end contrapositive_proposition_l237_23703


namespace circle_area_circle_circumference_l237_23765

section CircleProperties

variable (r : ℝ) -- Define the radius of the circle as a real number

-- State the theorem for the area of the circle
theorem circle_area (A : ℝ) : A = π * r^2 :=
sorry

-- State the theorem for the circumference of the circle
theorem circle_circumference (C : ℝ) : C = 2 * π * r :=
sorry

end CircleProperties

end circle_area_circle_circumference_l237_23765


namespace convex_polygon_from_non_overlapping_rectangles_is_rectangle_l237_23754

def isConvexPolygon (P : Set Point) : Prop := sorry
def canBeFormedByNonOverlappingRectangles (P : Set Point) (rects: List (Set Point)) : Prop := sorry
def isRectangle (P : Set Point) : Prop := sorry

theorem convex_polygon_from_non_overlapping_rectangles_is_rectangle
  (P : Set Point)
  (rects : List (Set Point))
  (h_convex : isConvexPolygon P)
  (h_form : canBeFormedByNonOverlappingRectangles P rects) :
  isRectangle P :=
sorry

end convex_polygon_from_non_overlapping_rectangles_is_rectangle_l237_23754


namespace racers_meet_at_start_again_l237_23785

-- We define the conditions as given
def RacingMagic_time := 60
def ChargingBull_time := 60 * 60 / 40 -- 90 seconds
def SwiftShadow_time := 80
def SpeedyStorm_time := 100

-- Prove the LCM of their lap times is 3600 seconds,
-- which is equivalent to 60 minutes.
theorem racers_meet_at_start_again :
  Nat.lcm (Nat.lcm (Nat.lcm RacingMagic_time ChargingBull_time) SwiftShadow_time) SpeedyStorm_time = 3600 ∧
  3600 / 60 = 60 := by
  sorry

end racers_meet_at_start_again_l237_23785


namespace dart_board_probability_l237_23732

variable {s : ℝ} (hexagon_area : ℝ := (3 * Real.sqrt 3) / 2 * s^2) (center_hexagon_area : ℝ := (3 * Real.sqrt 3) / 8 * s^2)

theorem dart_board_probability (s : ℝ) (P : ℝ) (h : P = center_hexagon_area / hexagon_area) :
  P = 1 / 4 :=
by
  sorry

end dart_board_probability_l237_23732


namespace hyperbola_sqrt3_eccentricity_l237_23779

noncomputable def hyperbola_eccentricity (m : ℝ) : ℝ :=
  let a := 2
  let b := m
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_sqrt3_eccentricity (m : ℝ) (h_m_pos : 0 < m) (h_slope : m = 2 * Real.sqrt 2) :
  hyperbola_eccentricity m = Real.sqrt 3 :=
by
  unfold hyperbola_eccentricity
  rw [h_slope]
  simp
  sorry

end hyperbola_sqrt3_eccentricity_l237_23779


namespace log_sum_eq_two_l237_23710

theorem log_sum_eq_two : 
  ∀ (lg : ℝ → ℝ),
  (∀ x y : ℝ, lg (x * y) = lg x + lg y) →
  (∀ x y : ℝ, lg (x ^ y) = y * lg x) →
  lg 4 + 2 * lg 5 = 2 :=
by
  intros lg h1 h2
  sorry

end log_sum_eq_two_l237_23710


namespace cracked_seashells_zero_l237_23725

/--
Tom found 15 seashells, and Fred found 43 seashells. After cleaning, it was discovered that Fred had 28 more seashells than Tom. Prove that the number of cracked seashells is 0.
-/
theorem cracked_seashells_zero
(Tom_seashells : ℕ)
(Fred_seashells : ℕ)
(cracked_seashells : ℕ)
(Tom_after_cleaning : ℕ := Tom_seashells - cracked_seashells)
(Fred_after_cleaning : ℕ := Fred_seashells - cracked_seashells)
(h1 : Tom_seashells = 15)
(h2 : Fred_seashells = 43)
(h3 : Fred_after_cleaning = Tom_after_cleaning + 28) :
  cracked_seashells = 0 :=
by
  -- Placeholder for the proof
  sorry

end cracked_seashells_zero_l237_23725


namespace find_rs_l237_23721

-- Define a structure to hold the conditions
structure Conditions (r s : ℝ) : Prop :=
  (positive_r : 0 < r)
  (positive_s : 0 < s)
  (eq1 : r^3 + s^3 = 1)
  (eq2 : r^6 + s^6 = (15 / 16))

-- State the theorem
theorem find_rs (r s : ℝ) (h : Conditions r s) : rs = 1 / (48 : ℝ)^(1/3) :=
by
  sorry

end find_rs_l237_23721


namespace algebraic_expression_standard_l237_23784

theorem algebraic_expression_standard :
  (∃ (expr : String), expr = "-(1/3)m" ∧
    expr ≠ "1(2/5)a" ∧
    expr ≠ "m / n" ∧
    expr ≠ "t × 3") :=
  sorry

end algebraic_expression_standard_l237_23784


namespace turn_all_black_l237_23727

def invertColor (v : Vertex) (G : Graph) : Graph := sorry

theorem turn_all_black (G : Graph) (n : ℕ) (whiteBlack : Vertex → Bool) :
  (∀ v : Vertex, whiteBlack v = false) :=
by
 -- Providing the base case for induction
  induction n with 
  | zero => sorry -- The base case for graphs with one vertex
  | succ n ih =>
    -- Inductive step: assume true for graph with n vertices and prove for graph with n+1 vertices
    sorry

end turn_all_black_l237_23727


namespace length_of_bridge_l237_23713

theorem length_of_bridge (length_train : ℕ) (speed_train_kmh : ℕ) (crossing_time_sec : ℕ)
    (h_length_train : length_train = 125)
    (h_speed_train_kmh : speed_train_kmh = 45)
    (h_crossing_time_sec : crossing_time_sec = 30) : 
    ∃ (length_bridge : ℕ), length_bridge = 250 := by
  sorry

end length_of_bridge_l237_23713


namespace syllogism_error_l237_23750

-- Definitions based on conditions from a)
def major_premise (a: ℝ) : Prop := a^2 > 0

def minor_premise (a: ℝ) : Prop := true

-- Theorem stating that the conclusion does not necessarily follow
theorem syllogism_error (a : ℝ) (h_minor : minor_premise a) : ¬major_premise 0 :=
by
  sorry

end syllogism_error_l237_23750


namespace sum_of_first_8_terms_l237_23726

theorem sum_of_first_8_terms (seq : ℕ → ℝ) (q : ℝ) (h_q : q = 2) 
  (h_sum_first_4 : seq 0 + seq 1 + seq 2 + seq 3 = 1) 
  (h_geom : ∀ n, seq (n + 1) = q * seq n) : 
  seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6 + seq 7 = 17 := 
sorry

end sum_of_first_8_terms_l237_23726


namespace find_angle_x_l237_23749

theorem find_angle_x (angle_ABC angle_BAC angle_BCA angle_DCE angle_CED x : ℝ)
  (h1 : angle_ABC + angle_BAC + angle_BCA = 180)
  (h2 : angle_ABC = 70) 
  (h3 : angle_BAC = 50)
  (h4 : angle_DCE + angle_CED = 90)
  (h5 : angle_DCE = angle_BCA) :
  x = 30 :=
by
  sorry

end find_angle_x_l237_23749


namespace greatest_common_factor_36_45_l237_23783

theorem greatest_common_factor_36_45 : 
  ∃ g, g = (gcd 36 45) ∧ g = 9 :=
by {
  sorry
}

end greatest_common_factor_36_45_l237_23783


namespace f_increasing_f_at_2_solve_inequality_l237_23728

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a + f b - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

theorem f_at_2 : f 2 = 3 :=
sorry

theorem solve_inequality (m : ℝ) : f (3 * m^2 - m - 2) < 3 ↔ -1 < m ∧ m < 4 / 3 :=
sorry

end f_increasing_f_at_2_solve_inequality_l237_23728


namespace prove_minimality_of_smallest_three_digit_multiple_of_17_l237_23773

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end prove_minimality_of_smallest_three_digit_multiple_of_17_l237_23773


namespace sum_of_fractions_le_half_l237_23763

theorem sum_of_fractions_le_half {a b c : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 1) :
  1 / (a^2 + 2 * b^2 + 3) + 1 / (b^2 + 2 * c^2 + 3) + 1 / (c^2 + 2 * a^2 + 3) ≤ 1 / 2 :=
by
  sorry

end sum_of_fractions_le_half_l237_23763


namespace laura_house_distance_l237_23722

-- Definitions based on conditions
def x : Real := 10  -- Distance from Laura's house to her school in miles

def distance_to_school_per_day := 2 * x
def school_days_per_week := 5
def distance_to_school_per_week := school_days_per_week * distance_to_school_per_day

def distance_to_supermarket := x + 10
def supermarket_trips_per_week := 2
def distance_to_supermarket_per_trip := 2 * distance_to_supermarket
def distance_to_supermarket_per_week := supermarket_trips_per_week * distance_to_supermarket_per_trip

def total_distance_per_week := 220

-- The proof statement
theorem laura_house_distance :
  distance_to_school_per_week + distance_to_supermarket_per_week = total_distance_per_week ∧ x = 10 := by
  sorry

end laura_house_distance_l237_23722


namespace probability_sum_divisible_by_3_l237_23731

theorem probability_sum_divisible_by_3:
  ∀ (n a b c : ℕ), a + b + c = n →
  4 * (a^3 + b^3 + c^3 + 6 * a * b * c) ≥ (a + b + c)^3 :=
by 
  intros n a b c habc_eq_n
  sorry

end probability_sum_divisible_by_3_l237_23731


namespace pages_to_read_tomorrow_l237_23760

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l237_23760


namespace minimize_a_plus_b_l237_23740

theorem minimize_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 4 * a + b = 30) :
  a + b = 9 → (a, b) = (7, 2) := sorry

end minimize_a_plus_b_l237_23740


namespace multiply_square_expression_l237_23720

theorem multiply_square_expression (x : ℝ) : ((-3 * x) ^ 2) * (2 * x) = 18 * x ^ 3 := by
  sorry

end multiply_square_expression_l237_23720


namespace gcd_g50_g51_l237_23786

-- Define the polynomial g(x)
def g (x : ℤ) : ℤ := x^2 + x + 2023

-- State the theorem with necessary conditions
theorem gcd_g50_g51 : Int.gcd (g 50) (g 51) = 17 :=
by
  -- Goals and conditions stated
  sorry  -- Placeholder for the proof

end gcd_g50_g51_l237_23786


namespace problem1_problem2_l237_23775

-- Definitions for first problem
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem for first problem
theorem problem1 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : ∀ x, -3 ≤ x → x ≤ 3) (h : f (m + 1) > f (2 * m - 1)) :
  -1 ≤ m ∧ m < 2 :=
sorry

-- Definitions for second problem
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem for second problem
theorem problem2 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : odd_function f) (h3 : f 2 = 1) (h4 : ∀ x, -3 ≤ x → x ≤ 3) :
  ∀ x, f (x + 1) + 1 > 0 ↔ -3 < x ∧ x ≤ 2 :=
sorry

end problem1_problem2_l237_23775


namespace inverse_proportion_comparison_l237_23736

theorem inverse_proportion_comparison (y1 y2 : ℝ) 
  (h1 : y1 = - 6 / 2)
  (h2 : y2 = - 6 / -1) : 
  y1 < y2 :=
by
  sorry

end inverse_proportion_comparison_l237_23736


namespace perpendicular_lines_l237_23799

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + y + 1 = 0) ∧ (∀ x y : ℝ, x + a * y + 3 = 0) ∧ (∀ A1 B1 A2 B2 : ℝ, A1 * A2 + B1 * B2 = 0) →
  a = -2 :=
by
  intros h
  sorry

end perpendicular_lines_l237_23799


namespace journey_distance_last_day_l237_23743

theorem journey_distance_last_day (S₆ : ℕ) (q : ℝ) (n : ℕ) (a₁ : ℝ) : 
  S₆ = 378 ∧ q = 1 / 2 ∧ n = 6 ∧ S₆ = a₁ * (1 - q^n) / (1 - q)
  → a₁ * q^(n - 1) = 6 :=
by
  intro h
  sorry

end journey_distance_last_day_l237_23743
