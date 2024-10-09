import Mathlib

namespace sqrt_x2y_l2309_230914

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end sqrt_x2y_l2309_230914


namespace breakfast_plate_contains_2_eggs_l2309_230982

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

end breakfast_plate_contains_2_eggs_l2309_230982


namespace perpendicular_vectors_l2309_230948

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n : ℝ × ℝ := (-3, 2)

-- Define the conditions to be checked
def km_plus_n (k : ℝ) : ℝ × ℝ := (k * m.1 + n.1, k * m.2 + n.2)
def m_minus_3n : ℝ × ℝ := (m.1 - 3 * n.1, m.2 - 3 * n.2)

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for k = 19, the two vectors are perpendicular
theorem perpendicular_vectors (k : ℝ) (h : k = 19) : dot_product (km_plus_n k) (m_minus_3n) = 0 := by
  rw [h]
  simp [km_plus_n, m_minus_3n, dot_product]
  sorry

end perpendicular_vectors_l2309_230948


namespace lilyPadsFullCoverage_l2309_230965

def lilyPadDoubling (t: ℕ) : ℕ :=
  t + 1

theorem lilyPadsFullCoverage (t: ℕ) (h: t = 47) : lilyPadDoubling t = 48 :=
by
  rw [h]
  unfold lilyPadDoubling
  rfl

end lilyPadsFullCoverage_l2309_230965


namespace inclination_angle_of_line_l2309_230958

-- Lean definition for the line equation and inclination angle problem
theorem inclination_angle_of_line : 
  ∃ θ : ℝ, (θ ∈ Set.Ico 0 Real.pi) ∧ (∀ x y: ℝ, x + y - 1 = 0 → Real.tan θ = -1) ∧ θ = 3 * Real.pi / 4 :=
sorry

end inclination_angle_of_line_l2309_230958


namespace correct_result_l2309_230904

-- Given condition
def mistaken_calculation (x : ℤ) : Prop :=
  x / 3 = 45

-- Proposition to prove the correct result
theorem correct_result (x : ℤ) (h : mistaken_calculation x) : 3 * x = 405 := by
  -- Here we can solve the proof later
  sorry

end correct_result_l2309_230904


namespace angle_C_in_triangle_l2309_230955

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 110) (ht : A + B + C = 180) : C = 70 :=
by
  -- proof steps go here
  sorry

end angle_C_in_triangle_l2309_230955


namespace arrange_books_correct_l2309_230960

def math_books : Nat := 4
def history_books : Nat := 4

def arrangements (m h : Nat) : Nat := sorry

theorem arrange_books_correct :
  arrangements math_books history_books = 576 := sorry

end arrange_books_correct_l2309_230960


namespace complement_union_l2309_230977

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

end complement_union_l2309_230977


namespace karen_drive_l2309_230935

theorem karen_drive (a b c x : ℕ) (h1 : a ≥ 1) (h2 : a + b + c ≤ 9) (h3 : 33 * (c - a) = 25 * x) :
  a^2 + b^2 + c^2 = 75 :=
sorry

end karen_drive_l2309_230935


namespace bonnets_per_orphanage_l2309_230926

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

end bonnets_per_orphanage_l2309_230926


namespace cyclists_original_number_l2309_230988

theorem cyclists_original_number (x : ℕ) (h : x > 2) : 
  (80 / (x - 2 : ℕ) = 80 / x + 2) → x = 10 :=
by
  sorry

end cyclists_original_number_l2309_230988


namespace condition_sufficient_but_not_necessary_l2309_230993

theorem condition_sufficient_but_not_necessary (a : ℝ) : (a > 9 → (1 / a < 1 / 9)) ∧ ¬(1 / a < 1 / 9 → a > 9) :=
by 
  sorry

end condition_sufficient_but_not_necessary_l2309_230993


namespace common_ratio_of_geometric_seq_l2309_230925

variable {α : Type} [LinearOrderedField α] 
variables (a d : α) (h₁ : d ≠ 0) (h₂ : (a + 2 * d) / (a + d) = (a + 5 * d) / (a + 2 * d))

theorem common_ratio_of_geometric_seq : (a + 2 * d) / (a + d) = 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l2309_230925


namespace darnel_jog_laps_l2309_230968

theorem darnel_jog_laps (x : ℝ) (h1 : 0.88 = x + 0.13) : x = 0.75 := by
  sorry

end darnel_jog_laps_l2309_230968


namespace job_completion_days_l2309_230995

variable (m r h d : ℕ)

theorem job_completion_days :
  (m + 2 * r) * (h + 1) * (m * h * d / ((m + 2 * r) * (h + 1))) = m * h * d :=
by
  sorry

end job_completion_days_l2309_230995


namespace correct_order_of_numbers_l2309_230910

theorem correct_order_of_numbers :
  let a := (4 / 5 : ℝ)
  let b := (81 / 100 : ℝ)
  let c := 0.801
  (a ≤ c ∧ c ≤ b) :=
by
  sorry

end correct_order_of_numbers_l2309_230910


namespace bee_flight_time_l2309_230933

theorem bee_flight_time (t : ℝ) : 
  let speed_daisy_to_rose := 2.6
  let speed_rose_to_poppy := speed_daisy_to_rose + 3
  let distance_daisy_to_rose := speed_daisy_to_rose * 10
  let distance_rose_to_poppy := distance_daisy_to_rose - 8
  distance_rose_to_poppy = speed_rose_to_poppy * t
  ∧ abs (t - 3) < 1 := 
sorry

end bee_flight_time_l2309_230933


namespace cone_prism_volume_ratio_l2309_230944

/--
Given:
- The base of the prism is a rectangle with side lengths 2r and 3r.
- The height of the prism is h.
- The base of the cone is a circle with radius r and height h.

Prove:
- The ratio of the volume of the cone to the volume of the prism is (π / 18).
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * Real.pi * r^2 * h) / (6 * r^2 * h) = Real.pi / 18 := by
  sorry

end cone_prism_volume_ratio_l2309_230944


namespace salary_increase_after_three_years_l2309_230903

-- Define the initial salary S and the raise percentage 12%
def initial_salary (S : ℝ) : ℝ := S
def raise_percentage : ℝ := 0.12

-- Define the salary after n raises
def salary_after_raises (S : ℝ) (n : ℕ) : ℝ :=
  S * (1 + raise_percentage)^n

-- Prove that the percentage increase after 3 years is 40.49%
theorem salary_increase_after_three_years (S : ℝ) :
  ((salary_after_raises S 3 - S) / S) * 100 = 40.49 :=
by sorry

end salary_increase_after_three_years_l2309_230903


namespace smallest_prime_perimeter_l2309_230922

-- Define a function that checks if a number is an odd prime
def is_odd_prime (n : ℕ) : Prop :=
  n > 2 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)) ∧ (n % 2 = 1)

-- Define a function that checks if three numbers are consecutive odd primes
def consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  b = a + 2 ∧ c = b + 2

-- Define a function that checks if three numbers form a scalene triangle and satisfy the triangle inequality
def scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem to prove
theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), consecutive_odd_primes a b c ∧ scalene_triangle a b c ∧ (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l2309_230922


namespace toothpicks_150th_stage_l2309_230992

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

end toothpicks_150th_stage_l2309_230992


namespace sum_of_five_consecutive_even_integers_l2309_230991

theorem sum_of_five_consecutive_even_integers (a : ℤ) (h : a + (a + 4) = 150) :
  a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385 :=
by
  sorry

end sum_of_five_consecutive_even_integers_l2309_230991


namespace range_of_a_l2309_230999

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 3 * x + 2 = 0) → ∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end range_of_a_l2309_230999


namespace distribute_candies_l2309_230967

-- Definition of the problem conditions
def candies : ℕ := 10

-- The theorem stating the proof problem
theorem distribute_candies : (2 ^ (candies - 1)) = 512 := 
by
  sorry

end distribute_candies_l2309_230967


namespace triangle_shortest_side_l2309_230954

theorem triangle_shortest_side (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) 
    (r : ℝ) (h3 : r = 5) 
    (h4 : a = 4) (h5 : b = 10)
    (circumcircle_tangent_property : 2 * (4 + 10) * r = 30) :
  min a (min b c) = 30 :=
by 
  sorry

end triangle_shortest_side_l2309_230954


namespace max_z_under_D_le_1_l2309_230915

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

end max_z_under_D_le_1_l2309_230915


namespace ball_more_expensive_l2309_230985

theorem ball_more_expensive (B L : ℝ) (h1 : 2 * B + 3 * L = 1300) (h2 : 3 * B + 2 * L = 1200) : 
  L - B = 100 := 
sorry

end ball_more_expensive_l2309_230985


namespace length_of_platform_l2309_230916

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


end length_of_platform_l2309_230916


namespace range_of_a_l2309_230917

noncomputable def A : Set ℝ := {x | x^2 ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : a ≥ 1 := 
by
  sorry

end range_of_a_l2309_230917


namespace find_supplementary_angle_l2309_230976

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

end find_supplementary_angle_l2309_230976


namespace root_relation_l2309_230912

theorem root_relation (a b x y : ℝ)
  (h1 : x + y = a)
  (h2 : (1 / x) + (1 / y) = 1 / b)
  (h3 : x = 3 * y)
  (h4 : y = a / 4) :
  b = 3 * a / 16 :=
by
  sorry

end root_relation_l2309_230912


namespace melted_mixture_weight_l2309_230902

/-- 
If the ratio of zinc to copper is 9:11 and 27 kg of zinc has been consumed, then the total weight of the melted mixture is 60 kg.
-/
theorem melted_mixture_weight (zinc_weight : ℕ) (ratio_zinc_to_copper : ℕ → ℕ → Prop)
  (h_ratio : ratio_zinc_to_copper 9 11) (h_zinc : zinc_weight = 27) :
  ∃ (total_weight : ℕ), total_weight = 60 :=
by
  sorry

end melted_mixture_weight_l2309_230902


namespace vertex_y_coord_of_h_l2309_230949

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 1
def h (x : ℝ) : ℝ := f x - g x

theorem vertex_y_coord_of_h : h (-1 / 10) = 79 / 20 := by
  sorry

end vertex_y_coord_of_h_l2309_230949


namespace simplify_expression_l2309_230989

theorem simplify_expression : (1 / (1 + Real.sqrt 3) * 1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 :=
by
  sorry

end simplify_expression_l2309_230989


namespace constant_value_AP_AQ_l2309_230929

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end constant_value_AP_AQ_l2309_230929


namespace product_remainder_mod_7_l2309_230911

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l2309_230911


namespace determine_N_l2309_230975

variable (U M N : Set ℕ)

theorem determine_N (h1 : U = {1, 2, 3, 4, 5})
  (h2 : U = M ∪ N)
  (h3 : M ∩ (U \ N) = {2, 4}) :
  N = {1, 3, 5} :=
by
  sorry

end determine_N_l2309_230975


namespace range_of_a_l2309_230907

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by
  intro h
  sorry

end range_of_a_l2309_230907


namespace factorize_cubic_l2309_230981

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l2309_230981


namespace painted_cube_l2309_230952

theorem painted_cube (n : ℕ) (h : 3 / 4 * (6 * n ^ 3) = 4 * n ^ 2) : n = 2 := sorry

end painted_cube_l2309_230952


namespace total_turns_to_fill_drum_l2309_230973

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

end total_turns_to_fill_drum_l2309_230973


namespace mass_percentage_of_Cl_in_compound_l2309_230905

theorem mass_percentage_of_Cl_in_compound (mass_percentage_Cl : ℝ) (h : mass_percentage_Cl = 92.11) : mass_percentage_Cl = 92.11 :=
sorry

end mass_percentage_of_Cl_in_compound_l2309_230905


namespace intersection_A_B_l2309_230962

-- Definition of set A
def A (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Definition of set B
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem stating the intersection of sets A and B
theorem intersection_A_B (x : ℝ) : (A x ∧ B x) ↔ (-1 < x ∧ x ≤ 0) :=
by sorry

end intersection_A_B_l2309_230962


namespace no_real_solutions_l2309_230938

theorem no_real_solutions :
  ∀ z : ℝ, ¬ ((-6 * z + 27) ^ 2 + 4 = -2 * |z|) :=
by
  sorry

end no_real_solutions_l2309_230938


namespace cylinder_new_volume_l2309_230932

-- Definitions based on conditions
def original_volume_r_h (π R H : ℝ) : ℝ := π * R^2 * H

def new_volume (π R H : ℝ) : ℝ := π * (3 * R)^2 * (2 * H)

theorem cylinder_new_volume (π R H : ℝ) (h_original_volume : original_volume_r_h π R H = 15) :
  new_volume π R H = 270 :=
by sorry

end cylinder_new_volume_l2309_230932


namespace four_digit_sum_divisible_l2309_230927

theorem four_digit_sum_divisible (A B C D : ℕ) :
  (10 * A + B + 10 * C + D = 94) ∧ (1000 * A + 100 * B + 10 * C + D % 94 = 0) →
  false :=
by
  sorry

end four_digit_sum_divisible_l2309_230927


namespace roots_sum_product_l2309_230928

theorem roots_sum_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : ∀ x : ℝ, x^2 - p*x - 2*q = 0) :
  (p + q = p) ∧ (p * q = -2*q) :=
by
  sorry

end roots_sum_product_l2309_230928


namespace boat_travel_distance_l2309_230946

variable (v c d : ℝ) (c_eq_1 : c = 1)

theorem boat_travel_distance : 
  (∀ (v : ℝ), d = (v + c) * 4 → d = (v - c) * 6) → d = 24 := 
by
  intro H
  sorry

end boat_travel_distance_l2309_230946


namespace calculate_expression_l2309_230920

theorem calculate_expression :
  107 * 107 + 93 * 93 = 20098 := by
  sorry

end calculate_expression_l2309_230920


namespace possible_values_of_a_l2309_230961

theorem possible_values_of_a :
  (∀ x, (x^2 - 3 * x + 2 = 0) → (ax - 2 = 0)) → (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  intro h
  sorry

end possible_values_of_a_l2309_230961


namespace gcd_apb_ab_eq1_gcd_aplusb_aminsb_l2309_230941

theorem gcd_apb_ab_eq1 (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a * b) = 1 ∧ Int.gcd (a - b) (a * b) = 1 := by
  sorry

theorem gcd_aplusb_aminsb (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a - b) = 1 ∨ Int.gcd (a + b) (a - b) = 2 := by
  sorry

end gcd_apb_ab_eq1_gcd_aplusb_aminsb_l2309_230941


namespace polynomial_exists_l2309_230930

open Polynomial

noncomputable def exists_polynomial_2013 : Prop :=
  ∃ (f : Polynomial ℤ), (∀ (n : ℕ), n ≤ f.natDegree → (coeff f n = 1 ∨ coeff f n = -1))
                         ∧ ((X - 1) ^ 2013 ∣ f)

theorem polynomial_exists : exists_polynomial_2013 :=
  sorry

end polynomial_exists_l2309_230930


namespace lego_set_cost_l2309_230951

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

end lego_set_cost_l2309_230951


namespace students_suggested_tomatoes_l2309_230900

theorem students_suggested_tomatoes (students_total mashed_potatoes bacon tomatoes : ℕ) 
  (h_total : students_total = 826)
  (h_mashed_potatoes : mashed_potatoes = 324)
  (h_bacon : bacon = 374)
  (h_tomatoes : students_total = mashed_potatoes + bacon + tomatoes) :
  tomatoes = 128 :=
by {
  sorry
}

end students_suggested_tomatoes_l2309_230900


namespace Maddie_bought_two_white_packs_l2309_230908

theorem Maddie_bought_two_white_packs 
  (W : ℕ)
  (total_cost : ℕ)
  (cost_per_shirt : ℕ)
  (white_pack_size : ℕ)
  (blue_pack_size : ℕ)
  (blue_packs : ℕ)
  (cost_per_white_pack : ℕ)
  (cost_per_blue_pack : ℕ) :
  total_cost = 66 ∧ cost_per_shirt = 3 ∧ white_pack_size = 5 ∧ blue_pack_size = 3 ∧ blue_packs = 4 ∧ cost_per_white_pack = white_pack_size * cost_per_shirt ∧ cost_per_blue_pack = blue_pack_size * cost_per_shirt ∧ 3 * (white_pack_size * W + blue_pack_size * blue_packs) = total_cost → W = 2 :=
by
  sorry

end Maddie_bought_two_white_packs_l2309_230908


namespace division_result_l2309_230978

-- Define the arithmetic expression
def arithmetic_expression : ℕ := (20 + 15 * 3) - 10

-- Define the main problem
def problem : Prop := 250 / arithmetic_expression = 250 / 55

-- The theorem statement that needs to be proved
theorem division_result : problem := by
    sorry

end division_result_l2309_230978


namespace mutually_exclusive_event_l2309_230918

def shooting_twice : Type := 
  { hit_first : Bool // hit_first = true ∨ hit_first = false }

def hitting_at_least_once (shoots : shooting_twice) : Prop :=
  shoots.1 ∨ (¬shoots.1 ∧ true)

def missing_both_times (shoots : shooting_twice) : Prop :=
  ¬shoots.1 ∧ (¬true ∨ true)

def mutually_exclusive (A : Prop) (B : Prop) : Prop :=
  A ∨ B → ¬ (A ∧ B)

theorem mutually_exclusive_event :
  ∀ shoots : shooting_twice, 
  mutually_exclusive (hitting_at_least_once shoots) (missing_both_times shoots) :=
by
  intro shoots
  unfold mutually_exclusive
  sorry

end mutually_exclusive_event_l2309_230918


namespace sin_squared_alpha_plus_pi_over_4_l2309_230919

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + Real.pi / 4) ^ 2 = 5 / 6 := 
sorry

end sin_squared_alpha_plus_pi_over_4_l2309_230919


namespace beth_sold_coins_l2309_230997

theorem beth_sold_coins :
  let initial_coins := 125
  let gift_coins := 35
  let total_coins := initial_coins + gift_coins
  let sold_coins := total_coins / 2
  sold_coins = 80 :=
by
  sorry

end beth_sold_coins_l2309_230997


namespace combination_10_5_l2309_230906

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l2309_230906


namespace minimum_value_condition_l2309_230950

theorem minimum_value_condition (a b : ℝ) (h : 16 * a ^ 2 + 2 * a + 8 * a * b + b ^ 2 - 1 = 0) : 
  ∃ m : ℝ, m = 3 * a + b ∧ m ≥ -1 :=
sorry

end minimum_value_condition_l2309_230950


namespace speed_in_still_water_l2309_230972

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 24 := by
  sorry

end speed_in_still_water_l2309_230972


namespace square_area_l2309_230939

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l2309_230939


namespace monotonic_decreasing_interval_l2309_230940

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end monotonic_decreasing_interval_l2309_230940


namespace find_a_l2309_230931

-- Define the curve y = x^2 + x
def curve (x : ℝ) : ℝ := x^2 + x

-- Line equation ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, line a x y → y = x^2 + x) ∧
  (deriv curve 1 = 2 * 1 + 1) →
  (2 * 1 + 1 = -1 / a) →
  a = -1 / 3 :=
by
  sorry

end find_a_l2309_230931


namespace portion_apples_weight_fraction_l2309_230969

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

end portion_apples_weight_fraction_l2309_230969


namespace present_worth_approx_l2309_230980

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

end present_worth_approx_l2309_230980


namespace cyclist_speed_ratio_l2309_230990

-- conditions: 
variables (T₁ T₂ o₁ o₂ : ℝ)
axiom h1 : o₁ + T₁ = o₂ + T₂
axiom h2 : T₁ = 2 * o₂
axiom h3 : T₂ = 4 * o₁

-- Proof statement to show that the second cyclist rides 1.5 times faster:
theorem cyclist_speed_ratio : T₁ / T₂ = 1.5 :=
by
  sorry

end cyclist_speed_ratio_l2309_230990


namespace rectangle_perimeter_l2309_230956

theorem rectangle_perimeter {w l : ℝ} 
  (h_area : l * w = 450)
  (h_length : l = 2 * w) :
  2 * (l + w) = 90 :=
by sorry

end rectangle_perimeter_l2309_230956


namespace gcd_g_y_l2309_230913

noncomputable def g (y : ℕ) : ℕ := (3 * y + 5) * (6 * y + 7) * (10 * y + 3) * (5 * y + 11) * (y + 7)

theorem gcd_g_y (y : ℕ) (h : ∃ k : ℕ, y = 18090 * k) : Nat.gcd (g y) y = 8085 := 
sorry

end gcd_g_y_l2309_230913


namespace mean_height_of_players_l2309_230943

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

end mean_height_of_players_l2309_230943


namespace second_number_l2309_230964

theorem second_number (x : ℝ) (h : 3 + x + 333 + 33.3 = 399.6) : x = 30.3 :=
sorry

end second_number_l2309_230964


namespace area_of_largest_square_l2309_230921

theorem area_of_largest_square (a b c : ℕ) (h_triangle : c^2 = a^2 + b^2) (h_sum_areas : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end area_of_largest_square_l2309_230921


namespace base_case_inequality_induction_inequality_l2309_230983

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end base_case_inequality_induction_inequality_l2309_230983


namespace fraction_inequality_l2309_230970

theorem fraction_inequality (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) :
  (a / d) < (b / c) :=
sorry

end fraction_inequality_l2309_230970


namespace area_of_quadrilateral_EFGH_l2309_230984

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

end area_of_quadrilateral_EFGH_l2309_230984


namespace maximum_obtuse_vectors_l2309_230959

-- Definition: A vector in 3D space
structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition: Dot product of two vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition: Two vectors form an obtuse angle if their dot product is negative
def obtuse_angle (v1 v2 : Vector3D) : Prop :=
  dot_product v1 v2 < 0

-- Main statement incorporating the conditions and the conclusion
theorem maximum_obtuse_vectors :
  ∀ (v1 v2 v3 v4 : Vector3D),
  (obtuse_angle v1 v2) →
  (obtuse_angle v1 v3) →
  (obtuse_angle v1 v4) →
  (obtuse_angle v2 v3) →
  (obtuse_angle v2 v4) →
  (obtuse_angle v3 v4) →
  -- Conclusion: At most 4 vectors can be pairwise obtuse
  ∃ (v5 : Vector3D),
  ¬ (obtuse_angle v1 v5 ∧ obtuse_angle v2 v5 ∧ obtuse_angle v3 v5 ∧ obtuse_angle v4 v5) :=
sorry

end maximum_obtuse_vectors_l2309_230959


namespace chord_to_diameter_ratio_l2309_230947

open Real

theorem chord_to_diameter_ratio
  (r R : ℝ) (h1 : r = R / 2)
  (a : ℝ)
  (h2 : r^2 = a^2 * 3 / 2) :
  3 * a / (2 * R) = 3 * sqrt 6 / 8 :=
by
  sorry

end chord_to_diameter_ratio_l2309_230947


namespace Z_4_1_eq_27_l2309_230909

def Z (a b : ℕ) : ℕ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem Z_4_1_eq_27 : Z 4 1 = 27 := by
  sorry

end Z_4_1_eq_27_l2309_230909


namespace range_of_ab_c2_l2309_230924

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end range_of_ab_c2_l2309_230924


namespace smallest_three_digit_multiple_of_17_l2309_230986

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l2309_230986


namespace original_perimeter_of_rectangle_l2309_230974

theorem original_perimeter_of_rectangle
  (a b : ℝ)
  (h : (a + 3) * (b + 3) - a * b = 90) :
  2 * (a + b) = 54 :=
sorry

end original_perimeter_of_rectangle_l2309_230974


namespace min_segment_length_l2309_230936

theorem min_segment_length 
  (angle : ℝ) (P : ℝ × ℝ)
  (dist_x : ℝ) (dist_y : ℝ) 
  (hx : P.1 ≤ dist_x ∧ P.2 = dist_y)
  (hy : P.2 ≤ dist_y ∧ P.1 = dist_x)
  (right_angle : angle = 90) 
  : ∃ (d : ℝ), d = 10 :=
by
  sorry

end min_segment_length_l2309_230936


namespace second_order_arithmetic_progression_a100_l2309_230934

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

end second_order_arithmetic_progression_a100_l2309_230934


namespace tetrahedron_in_cube_l2309_230979

theorem tetrahedron_in_cube (a x : ℝ) (h : a = 6) :
  (∃ x, x = 6 * Real.sqrt 2) :=
sorry

end tetrahedron_in_cube_l2309_230979


namespace simplify_fraction_result_l2309_230963

theorem simplify_fraction_result : (130 / 16900) * 65 = 1 / 2 :=
by sorry

end simplify_fraction_result_l2309_230963


namespace greatest_int_radius_lt_75pi_l2309_230998

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l2309_230998


namespace valid_propositions_l2309_230994

theorem valid_propositions :
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧ (∃ n : ℝ, ∀ m : ℝ, m * n = m) :=
by
  sorry

end valid_propositions_l2309_230994


namespace min_x_plus_y_l2309_230901

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) :
  x + y ≥ 16 :=
sorry

end min_x_plus_y_l2309_230901


namespace selling_price_l2309_230942

def initial_cost : ℕ := 600
def food_cost_per_day : ℕ := 20
def number_of_days : ℕ := 40
def vaccination_and_deworming_cost : ℕ := 500
def profit : ℕ := 600

theorem selling_price (S : ℕ) :
  S = initial_cost + (food_cost_per_day * number_of_days) + vaccination_and_deworming_cost + profit :=
by
  sorry

end selling_price_l2309_230942


namespace perpendicular_lines_b_l2309_230945

theorem perpendicular_lines_b (b : ℝ) : 
  (∃ (k m: ℝ), k = 3 ∧ 2 * m + b * k = 14 ∧ (k * m = -1)) ↔ b = 2 / 3 :=
sorry

end perpendicular_lines_b_l2309_230945


namespace students_not_taking_french_or_spanish_l2309_230937

theorem students_not_taking_french_or_spanish 
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (both_languages_students : ℕ) 
  (h_total_students : total_students = 28)
  (h_french_students : french_students = 5)
  (h_spanish_students : spanish_students = 10)
  (h_both_languages_students : both_languages_students = 4) :
  total_students - (french_students + spanish_students - both_languages_students) = 17 := 
by {
  -- Correct answer can be verified with the given conditions
  -- The proof itself is omitted (as instructed)
  sorry
}

end students_not_taking_french_or_spanish_l2309_230937


namespace arith_seq_a1_eq_15_l2309_230953

variable {a : ℕ → ℤ} (a_seq : ∀ n, a n = a 1 + (n-1) * d)
variable {a_4 : ℤ} (h4 : a 4 = 9)
variable {a_8 : ℤ} (h8 : a 8 = -a 9)

theorem arith_seq_a1_eq_15 (a_seq : ∀ n, a n = a 1 + (n-1) * d) (h4 : a 4 = 9) (h8 : a 8 = -a 9) : a 1 = 15 :=
by
  -- Proof should go here
  sorry

end arith_seq_a1_eq_15_l2309_230953


namespace shirts_per_kid_l2309_230966

-- Define given conditions
def n_buttons : Nat := 63
def buttons_per_shirt : Nat := 7
def n_kids : Nat := 3

-- The proof goal
theorem shirts_per_kid : (n_buttons / buttons_per_shirt) / n_kids = 3 := by
  sorry

end shirts_per_kid_l2309_230966


namespace avg_of_arithmetic_series_is_25_l2309_230923

noncomputable def arithmetic_series_avg : ℝ :=
  let a₁ := 15
  let d := 1 / 4
  let aₙ := 35
  let n := (aₙ - a₁) / d + 1
  let S := n * (a₁ + aₙ) / 2
  S / n

theorem avg_of_arithmetic_series_is_25 : arithmetic_series_avg = 25 := 
by
  -- Sorry, proof omitted due to instruction.
  sorry

end avg_of_arithmetic_series_is_25_l2309_230923


namespace average_visitors_on_other_days_l2309_230971

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

end average_visitors_on_other_days_l2309_230971


namespace mass_percentage_Al_in_AlBr3_l2309_230996

theorem mass_percentage_Al_in_AlBr3 
  (molar_mass_Al : Real := 26.98) 
  (molar_mass_Br : Real := 79.90) 
  (molar_mass_AlBr3 : Real := molar_mass_Al + 3 * molar_mass_Br)
  : (molar_mass_Al / molar_mass_AlBr3) * 100 = 10.11 := 
by 
  -- Here we would provide the proof; skipping with sorry
  sorry

end mass_percentage_Al_in_AlBr3_l2309_230996


namespace outer_circle_radius_l2309_230957

theorem outer_circle_radius (r R : ℝ) (hr : r = 4)
  (radius_increase : ∀ R, R' = 1.5 * R)
  (radius_decrease : ∀ r, r' = 0.75 * r)
  (area_increase : ∀ (A1 A2 : ℝ), A2 = 3.6 * A1)
  (initial_area : ∀ A1, A1 = π * R^2 - π * r^2)
  (new_area : ∀ A2 R' r', A2 = π * R'^2 - π * r'^2) :
  R = 6 := sorry

end outer_circle_radius_l2309_230957


namespace suitcase_lock_settings_l2309_230987

-- Define the number of settings for each dial choice considering the conditions
noncomputable def first_digit_choices : ℕ := 9
noncomputable def second_digit_choices : ℕ := 9
noncomputable def third_digit_choices : ℕ := 8
noncomputable def fourth_digit_choices : ℕ := 7

-- Theorem to prove the total number of different settings
theorem suitcase_lock_settings : first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices = 4536 :=
by sorry

end suitcase_lock_settings_l2309_230987
