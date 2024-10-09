import Mathlib

namespace mary_max_earnings_l2138_213805

def max_hours : ℕ := 40
def regular_rate : ℝ := 8
def first_hours : ℕ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate

def earnings : ℝ := 
  (first_hours * regular_rate) +
  ((max_hours - first_hours) * overtime_rate)

theorem mary_max_earnings : earnings = 360 := by
  sorry

end mary_max_earnings_l2138_213805


namespace geometric_sequence_q_l2138_213876

theorem geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 4 + a 8 = 8) :
  q = 1 :=
by
  sorry

end geometric_sequence_q_l2138_213876


namespace relationship_between_coefficients_l2138_213871

theorem relationship_between_coefficients
  (b c : ℝ)
  (h_discriminant : b^2 - 4 * c ≥ 0)
  (h_root_condition : ∃ x1 x2 : ℝ, x1^2 = -x2 ∧ x1 + x2 = -b ∧ x1 * x2 = c):
  b^3 - 3 * b * c - c^2 - c = 0 :=
by
  sorry

end relationship_between_coefficients_l2138_213871


namespace sum_of_decimals_l2138_213868

theorem sum_of_decimals : 1.000 + 0.101 + 0.011 + 0.001 = 1.113 :=
by
  sorry

end sum_of_decimals_l2138_213868


namespace calculate_expression_l2138_213888

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l2138_213888


namespace last_student_score_is_61_l2138_213854

noncomputable def average_score_19_students := 82
noncomputable def average_score_20_students := 84
noncomputable def total_students := 20
noncomputable def oliver_multiplier := 2

theorem last_student_score_is_61 
  (total_score_19_students : ℝ := total_students - 1 * average_score_19_students)
  (total_score_20_students : ℝ := total_students * average_score_20_students)
  (oliver_score : ℝ := total_score_20_students - total_score_19_students)
  (last_student_score : ℝ := oliver_score / oliver_multiplier) :
  last_student_score = 61 :=
sorry

end last_student_score_is_61_l2138_213854


namespace exists_n_not_coprime_l2138_213836

theorem exists_n_not_coprime (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : q > p) (h3 : q - p > 1) :
  ∃ (n : ℕ), Nat.gcd (p + n) (q + n) ≠ 1 :=
by
  sorry

end exists_n_not_coprime_l2138_213836


namespace find_number_l2138_213886

-- Define the problem constants
def total : ℝ := 1.794
def part1 : ℝ := 0.123
def part2 : ℝ := 0.321
def target : ℝ := 1.350

-- The equivalent proof problem
theorem find_number (x : ℝ) (h : part1 + part2 + x = total) : x = target := by
  -- Proof is intentionally omitted
  sorry

end find_number_l2138_213886


namespace product_of_two_numbers_l2138_213827

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l2138_213827


namespace smallest_integer_quad_ineq_l2138_213859

-- Definition of the condition
def quad_ineq (n : ℤ) := n^2 - 14 * n + 45 > 0

-- Lean 4 statement of the math proof problem
theorem smallest_integer_quad_ineq : ∃ n : ℤ, quad_ineq n ∧ ∀ m : ℤ, quad_ineq m → n ≤ m :=
  by
    existsi 10
    sorry

end smallest_integer_quad_ineq_l2138_213859


namespace average_speed_round_trip_l2138_213830

variable (D : ℝ) (u v : ℝ)
  
theorem average_speed_round_trip (h1 : u = 96) (h2 : v = 88) : 
  (2 * u * v) / (u + v) = 91.73913043 := 
by 
  sorry

end average_speed_round_trip_l2138_213830


namespace monotonic_increasing_condition_l2138_213832

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → y a x₁ ≤ y a x₂) ↔ 
  (a = 0 ∨ a > 0) :=
sorry

end monotonic_increasing_condition_l2138_213832


namespace number_zero_points_eq_three_l2138_213828

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - x^2

theorem number_zero_points_eq_three : ∃ x1 x2 x3 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (∀ y : ℝ, f y = 0 → (y = x1 ∨ y = x2 ∨ y = x3)) :=
sorry

end number_zero_points_eq_three_l2138_213828


namespace hyperbola_equation_of_midpoint_l2138_213801

-- Define the hyperbola E
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ) (hapos : a > 0) (hbpos : b > 0)
variables (F : ℝ × ℝ) (hF : F = (-2, 0))
variables (M : ℝ × ℝ) (hM : M = (-3, -1))

-- The statement requiring proof
theorem hyperbola_equation_of_midpoint (hE : hyperbola a b (-2) 0) 
(hFocus : a^2 + b^2 = 4) : 
  (∃ a' b', a' = 3 ∧ b' = 1 ∧ hyperbola a' b' (-3) (-1)) :=
sorry

end hyperbola_equation_of_midpoint_l2138_213801


namespace B_work_days_l2138_213845

theorem B_work_days (A B C : ℕ) (hA : A = 15) (hC : C = 30) (H : (5 / 15) + ((10 * (1 / C + 1 / B)) / (1 / C + 1 / B)) = 1) : B = 30 := by
  sorry

end B_work_days_l2138_213845


namespace grassy_plot_width_l2138_213813

theorem grassy_plot_width (L : ℝ) (P : ℝ) (C : ℝ) (cost_per_sqm : ℝ) (W : ℝ) : 
  L = 110 →
  P = 2.5 →
  C = 510 →
  cost_per_sqm = 0.6 →
  (115 * (W + 5) - 110 * W = C / cost_per_sqm) →
  W = 55 :=
by
  intros hL hP hC hcost_per_sqm harea
  sorry

end grassy_plot_width_l2138_213813


namespace find_original_intensity_l2138_213873

variable (I : ℝ)  -- Define intensity of the original red paint (in percentage).

-- Conditions:
variable (fractionReplaced : ℝ) (newIntensity : ℝ) (replacingIntensity : ℝ)
  (fractionReplaced_eq : fractionReplaced = 0.8)
  (newIntensity_eq : newIntensity = 30)
  (replacingIntensity_eq : replacingIntensity = 25)

-- Theorem statement:
theorem find_original_intensity :
  (1 - fractionReplaced) * I + fractionReplaced * replacingIntensity = newIntensity → I = 50 :=
sorry

end find_original_intensity_l2138_213873


namespace total_molecular_weight_of_products_l2138_213818

/-- Problem Statement: Determine the total molecular weight of the products formed when
    8 moles of Copper(II) carbonate (CuCO3) react with 6 moles of Diphosphorus pentoxide (P4O10)
    to form Copper(II) phosphate (Cu3(PO4)2) and Carbon dioxide (CO2). -/
theorem total_molecular_weight_of_products 
  (moles_CuCO3 : ℕ) 
  (moles_P4O10 : ℕ)
  (atomic_weight_Cu : ℝ := 63.55)
  (atomic_weight_P : ℝ := 30.97)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_C : ℝ := 12.01)
  (molecular_weight_CuCO3 : ℝ := atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O)
  (molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O)
  (molecular_weight_Cu3PO4_2 : ℝ := (3 * atomic_weight_Cu) + (2 * atomic_weight_P) + (8 * atomic_weight_O))
  (moles_Cu3PO4_2_formed : ℝ := (8 : ℝ) / 3)
  (moles_CO2_formed : ℝ := 8)
  (total_molecular_weight_Cu3PO4_2 : ℝ := moles_Cu3PO4_2_formed * molecular_weight_Cu3PO4_2)
  (total_molecular_weight_CO2 : ℝ := moles_CO2_formed * molecular_weight_CO2) : 
  (total_molecular_weight_Cu3PO4_2 + total_molecular_weight_CO2) = 1368.45 := by
  sorry

end total_molecular_weight_of_products_l2138_213818


namespace count_odd_perfect_squares_less_than_16000_l2138_213802

theorem count_odd_perfect_squares_less_than_16000 : 
  ∃ n : ℕ, n = 31 ∧ ∀ k < 16000, 
    ∃ b : ℕ, b = 2 * n + 1 ∧ k = (4 * n + 3) ^ 2 ∧ (∃ m : ℕ, m = b + 1 ∧ m % 2 = 0) := 
sorry

end count_odd_perfect_squares_less_than_16000_l2138_213802


namespace library_books_new_releases_l2138_213819

theorem library_books_new_releases (P Q R S : Prop) 
  (h : ¬P) 
  (P_iff_Q : P ↔ Q)
  (Q_implies_R : Q → R)
  (S_iff_notP : S ↔ ¬P) : 
  Q ∧ S := by 
  sorry

end library_books_new_releases_l2138_213819


namespace line_contains_point_iff_k_eq_neg1_l2138_213869

theorem line_contains_point_iff_k_eq_neg1 (k : ℝ) :
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ (2 - k * x = -4 * y)) ↔ k = -1 :=
by
  sorry

end line_contains_point_iff_k_eq_neg1_l2138_213869


namespace distribute_seedlings_l2138_213870

noncomputable def box_contents : List ℕ := [28, 51, 135, 67, 123, 29, 56, 38, 79]

def total_seedlings (contents : List ℕ) : ℕ := contents.sum

def obtainable_by_sigmas (contents : List ℕ) (σs : List ℕ) : Prop :=
  ∃ groups : List (List ℕ),
    (groups.length = σs.length) ∧
    (∀ g ∈ groups, contents.contains g.sum) ∧
    (∀ g, g ∈ groups → g.sum ∈ σs)

theorem distribute_seedlings : 
  total_seedlings box_contents = 606 →
  obtainable_by_sigmas box_contents [202, 202, 202] ∧
  ∃ way1 way2 : List (List ℕ),
    (way1 ≠ way2) ∧
    (obtainable_by_sigmas box_contents [202, 202, 202]) :=
by
  sorry

end distribute_seedlings_l2138_213870


namespace temperature_increase_l2138_213803

variable (T_morning T_afternoon : ℝ)

theorem temperature_increase : 
  (T_morning = -3) → (T_afternoon = 5) → (T_afternoon - T_morning = 8) :=
by
intros h1 h2
rw [h1, h2]
sorry

end temperature_increase_l2138_213803


namespace sphere_surface_area_l2138_213843

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (S : ℝ)
  (hV : V = 36 * π)
  (hvol : V = (4 / 3) * π * r^3) :
  S = 4 * π * r^2 :=
by
  sorry

end sphere_surface_area_l2138_213843


namespace unique_k_value_l2138_213897

noncomputable def findK (k : ℝ) : Prop :=
  ∃ (x : ℝ), (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4) ∧ k ≠ 0 ∧ k = -3

theorem unique_k_value : ∀ (k : ℝ), findK k :=
by
  intro k
  sorry

end unique_k_value_l2138_213897


namespace profit_percentage_is_30_percent_l2138_213839

theorem profit_percentage_is_30_percent (CP SP : ℕ) (h1 : CP = 280) (h2 : SP = 364) :
  ((SP - CP : ℤ) / (CP : ℤ) : ℚ) * 100 = 30 :=
by sorry

end profit_percentage_is_30_percent_l2138_213839


namespace angle_BDC_correct_l2138_213814

theorem angle_BDC_correct (A B C D : Type) 
  (angle_A : ℝ) (angle_B : ℝ) (angle_DBC : ℝ) : 
  angle_A = 60 ∧ angle_B = 70 ∧ angle_DBC = 40 → 
  ∃ angle_BDC : ℝ, angle_BDC = 100 := 
by
  intro h
  sorry

end angle_BDC_correct_l2138_213814


namespace length_of_longest_side_l2138_213815

theorem length_of_longest_side (l w : ℝ) (h_fencing : 2 * l + 2 * w = 240) (h_area : l * w = 8 * 240) : max l w = 96 :=
by sorry

end length_of_longest_side_l2138_213815


namespace B_max_at_125_l2138_213893

noncomputable def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 : ℝ) ^ k

theorem B_max_at_125 :
  ∃ k, 0 ≤ k ∧ k ≤ 500 ∧ (∀ n, 0 ≤ n ∧ n ≤ 500 → B k ≥ B n) ∧ k = 125 :=
by
  sorry

end B_max_at_125_l2138_213893


namespace larger_model_ratio_smaller_model_ratio_l2138_213822

-- Definitions for conditions
def statue_height := 305 -- The height of the actual statue in feet
def larger_model_height := 10 -- The height of the larger model in inches
def smaller_model_height := 5 -- The height of the smaller model in inches

-- The ratio calculation for larger model
theorem larger_model_ratio : 
  (statue_height : ℝ) / (larger_model_height : ℝ) = 30.5 := by
  sorry

-- The ratio calculation for smaller model
theorem smaller_model_ratio : 
  (statue_height : ℝ) / (smaller_model_height : ℝ) = 61 := by
  sorry

end larger_model_ratio_smaller_model_ratio_l2138_213822


namespace ravi_nickels_l2138_213895

variables (n q d : ℕ)

-- Defining the conditions
def quarters (n : ℕ) : ℕ := n + 2
def dimes (q : ℕ) : ℕ := q + 4

-- Using these definitions to form the Lean theorem
theorem ravi_nickels : 
  ∃ n, q = quarters n ∧ d = dimes q ∧ 
  (0.05 * n + 0.25 * q + 0.10 * d : ℝ) = 3.50 ∧ n = 6 :=
sorry

end ravi_nickels_l2138_213895


namespace mark_weekly_reading_l2138_213856

-- Using the identified conditions
def daily_reading_hours : ℕ := 2
def additional_weekly_hours : ℕ := 4

-- Prove the total number of hours Mark wants to read per week is 18 hours
theorem mark_weekly_reading : (daily_reading_hours * 7 + additional_weekly_hours) = 18 := by
  -- Placeholder for proof
  sorry

end mark_weekly_reading_l2138_213856


namespace license_plate_increase_factor_l2138_213837

def old_plate_count : ℕ := 26^2 * 10^3
def new_plate_count : ℕ := 26^4 * 10^4
def increase_factor : ℕ := new_plate_count / old_plate_count

theorem license_plate_increase_factor : increase_factor = 2600 :=
by
  unfold increase_factor
  rw [old_plate_count, new_plate_count]
  norm_num
  sorry

end license_plate_increase_factor_l2138_213837


namespace relationship_among_abc_l2138_213882

noncomputable def a := Real.sqrt 5 + 2
noncomputable def b := 2 - Real.sqrt 5
noncomputable def c := Real.sqrt 5 - 2

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end relationship_among_abc_l2138_213882


namespace annual_interest_payment_l2138_213891

def principal : ℝ := 10000
def quarterly_rate : ℝ := 0.05

theorem annual_interest_payment :
  (principal * quarterly_rate * 4) = 2000 :=
by sorry

end annual_interest_payment_l2138_213891


namespace e_exp_f_neg2_l2138_213872

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
axiom h_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_ln_pos : ∀ x : ℝ, x > 0 → f x = Real.log x

-- Theorem to prove:
theorem e_exp_f_neg2 : Real.exp (f (-2)) = 1 / 2 := by
  sorry

end e_exp_f_neg2_l2138_213872


namespace arcsin_zero_l2138_213820

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end arcsin_zero_l2138_213820


namespace range_of_m_l2138_213867

def quadratic_nonnegative (m : ℝ) : Prop :=
∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0

theorem range_of_m (m : ℝ) :
  quadratic_nonnegative m ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l2138_213867


namespace missing_pieces_l2138_213885

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l2138_213885


namespace volume_of_each_cube_is_correct_l2138_213841

def box_length : ℕ := 12
def box_width : ℕ := 16
def box_height : ℕ := 6
def total_volume : ℕ := 1152
def number_of_cubes : ℕ := 384

theorem volume_of_each_cube_is_correct :
  (total_volume / number_of_cubes = 3) :=
by
  sorry

end volume_of_each_cube_is_correct_l2138_213841


namespace rectangular_solid_dimension_change_l2138_213824

theorem rectangular_solid_dimension_change (a b : ℝ) (h : 2 * a^2 + 4 * a * b = 0.6 * (6 * a^2)) : b = 0.4 * a :=
by sorry

end rectangular_solid_dimension_change_l2138_213824


namespace minimum_value_inequality_l2138_213846

theorem minimum_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (Real.sqrt ((x^2 + 4 * y^2) * (2 * x^2 + 3 * y^2)) / (x * y)) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end minimum_value_inequality_l2138_213846


namespace macaroon_count_l2138_213878

def baked_red_macaroons : ℕ := 50
def baked_green_macaroons : ℕ := 40
def ate_green_macaroons : ℕ := 15
def ate_red_macaroons := 2 * ate_green_macaroons

def remaining_macaroons : ℕ := (baked_red_macaroons - ate_red_macaroons) + (baked_green_macaroons - ate_green_macaroons)

theorem macaroon_count : remaining_macaroons = 45 := by
  sorry

end macaroon_count_l2138_213878


namespace determinant_of_triangle_angles_l2138_213890

theorem determinant_of_triangle_angles (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Matrix.det ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ] = 0 :=
by
  -- Proof statement goes here
  sorry

end determinant_of_triangle_angles_l2138_213890


namespace flagpole_height_l2138_213862

theorem flagpole_height (h : ℕ) (shadow_flagpole : ℕ) (height_building : ℕ) (shadow_building : ℕ) (similar_conditions : Prop) 
  (H1 : shadow_flagpole = 45) 
  (H2 : height_building = 24) 
  (H3 : shadow_building = 60) 
  (H4 : similar_conditions) 
  (H5 : h / 45 = 24 / 60) : h = 18 := 
by 
sorry

end flagpole_height_l2138_213862


namespace matthew_total_time_l2138_213823

def assemble_time : ℝ := 1
def bake_time_normal : ℝ := 1.5
def decorate_time : ℝ := 1
def bake_time_double : ℝ := bake_time_normal * 2

theorem matthew_total_time :
  assemble_time + bake_time_double + decorate_time = 5 := 
by 
  -- The proof will be filled in here
  sorry

end matthew_total_time_l2138_213823


namespace purely_imaginary_complex_number_l2138_213816

theorem purely_imaginary_complex_number (a : ℝ) (h : (a^2 - 3 * a + 2) = 0 ∧ (a - 2) ≠ 0) : a = 1 :=
by {
  sorry
}

end purely_imaginary_complex_number_l2138_213816


namespace smallest_possible_value_expression_l2138_213844

open Real

noncomputable def min_expression_value (a b c : ℝ) : ℝ :=
  (a + b)^2 + (b - c)^2 + (c - a)^2 / a^2

theorem smallest_possible_value_expression :
  ∀ (a b c : ℝ), a > b → b > c → a + c = 2 * b → a ≠ 0 → min_expression_value a b c = 7 / 2 := by
  sorry

end smallest_possible_value_expression_l2138_213844


namespace least_distance_on_cone_l2138_213865

noncomputable def least_distance_fly_could_crawl_cone (R C : ℝ) (slant_height : ℝ) (start_dist vertex_dist : ℝ) : ℝ :=
  if start_dist = 150 ∧ vertex_dist = 450 ∧ R = 500 ∧ C = 800 * Real.pi ∧ slant_height = R ∧ 
     (500 * (8 * Real.pi / 5) = 800 * Real.pi) then 600 else 0

theorem least_distance_on_cone : least_distance_fly_could_crawl_cone 500 (800 * Real.pi) 500 150 450 = 600 :=
by
  sorry

end least_distance_on_cone_l2138_213865


namespace average_weight_of_girls_l2138_213864

theorem average_weight_of_girls (avg_weight_boys : ℕ) (num_boys : ℕ) (avg_weight_class : ℕ) (num_students : ℕ) :
  num_boys = 15 →
  avg_weight_boys = 48 →
  num_students = 25 →
  avg_weight_class = 45 →
  ( (avg_weight_class * num_students - avg_weight_boys * num_boys) / (num_students - num_boys) ) = 27 :=
by
  intros h_num_boys h_avg_weight_boys h_num_students h_avg_weight_class
  sorry

end average_weight_of_girls_l2138_213864


namespace num_2_coins_l2138_213899

open Real

theorem num_2_coins (x y z : ℝ) (h1 : x + y + z = 900)
                     (h2 : x + 2 * y + 5 * z = 1950)
                     (h3 : z = 0.5 * x) : y = 450 :=
by sorry

end num_2_coins_l2138_213899


namespace percent_of_N_in_M_l2138_213847

theorem percent_of_N_in_M (N M : ℝ) (hM : M ≠ 0) : (N / M) * 100 = 100 * N / M :=
by
  sorry

end percent_of_N_in_M_l2138_213847


namespace jacob_younger_than_michael_l2138_213829

variables (J M : ℕ)

theorem jacob_younger_than_michael (h1 : M + 9 = 2 * (J + 9)) (h2 : J = 5) : M - J = 14 :=
by
  -- Insert proof steps here
  sorry

end jacob_younger_than_michael_l2138_213829


namespace boys_under_six_ratio_l2138_213863

theorem boys_under_six_ratio (total_students : ℕ) (two_third_boys : (2/3 : ℚ) * total_students = 25) (boys_under_six : ℕ) (boys_under_six_eq : boys_under_six = 19) :
  boys_under_six / 25 = 19 / 25 :=
by
  sorry

end boys_under_six_ratio_l2138_213863


namespace beads_per_necklace_l2138_213852

theorem beads_per_necklace (n : ℕ) (b : ℕ) (total_beads : ℕ) (total_necklaces : ℕ)
  (h1 : total_necklaces = 6) (h2 : total_beads = 18) (h3 : b * total_necklaces = total_beads) :
  b = 3 :=
by {
  sorry
}

end beads_per_necklace_l2138_213852


namespace total_seats_taken_l2138_213875

def students_per_bus : ℝ := 14.0
def number_of_buses : ℝ := 2.0

theorem total_seats_taken :
  students_per_bus * number_of_buses = 28.0 :=
by
  sorry

end total_seats_taken_l2138_213875


namespace g_at_5_l2138_213810

def g : ℝ → ℝ := sorry

axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

theorem g_at_5 : g 5 = -20 :=
by {
  apply sorry
}

end g_at_5_l2138_213810


namespace fourth_intersection_point_of_curve_and_circle_l2138_213811

theorem fourth_intersection_point_of_curve_and_circle (h k R : ℝ)
  (h1 : (3 - h)^2 + (2 / 3 - k)^2 = R^2)
  (h2 : (-4 - h)^2 + (-1 / 2 - k)^2 = R^2)
  (h3 : (1 / 2 - h)^2 + (4 - k)^2 = R^2) :
  ∃ (x y : ℝ), xy = 2 ∧ (x, y) ≠ (3, 2 / 3) ∧ (x, y) ≠ (-4, -1 / 2) ∧ (x, y) ≠ (1 / 2, 4) ∧ 
    (x - h)^2 + (y - k)^2 = R^2 ∧ (x, y) = (2 / 3, 3) := 
sorry

end fourth_intersection_point_of_curve_and_circle_l2138_213811


namespace max_correct_answers_l2138_213800

variables {a b c : ℕ} -- Define a, b, and c as natural numbers

theorem max_correct_answers : 
  ∀ a b c : ℕ, (a + b + c = 50) → (5 * a - 2 * c = 150) → a ≤ 35 :=
by
  -- Proof steps can be skipped by adding sorry
  sorry

end max_correct_answers_l2138_213800


namespace larger_number_is_20_l2138_213881

theorem larger_number_is_20 (a b : ℕ) (h1 : a + b = 9 * (a - b)) (h2 : a + b = 36) (h3 : a > b) : a = 20 :=
by
  sorry

end larger_number_is_20_l2138_213881


namespace simplify_expression_l2138_213817

theorem simplify_expression : 20 * (9 / 14) * (1 / 18) = 5 / 7 :=
by sorry

end simplify_expression_l2138_213817


namespace other_asymptote_l2138_213896

/-- Problem Statement:
One of the asymptotes of a hyperbola is y = 2x. The foci have the same 
x-coordinate, which is 4. Prove that the equation of the other asymptote
of the hyperbola is y = -2x + 16.
-/
theorem other_asymptote (focus_x : ℝ) (asymptote1: ℝ → ℝ) (asymptote2 : ℝ → ℝ) :
  focus_x = 4 →
  (∀ x, asymptote1 x = 2 * x) →
  (asymptote2 4 = 8) → 
  (∀ x, asymptote2 x = -2 * x + 16) :=
sorry

end other_asymptote_l2138_213896


namespace interest_rate_eq_five_percent_l2138_213833

def total_sum : ℝ := 2665
def P2 : ℝ := 1332.5
def P1 : ℝ := total_sum - P2

theorem interest_rate_eq_five_percent :
  (3 * 0.03 * P1 = r * 0.03 * P2) → r = 5 :=
by
  sorry

end interest_rate_eq_five_percent_l2138_213833


namespace work_completion_in_16_days_l2138_213840

theorem work_completion_in_16_days (A B : ℕ) :
  (1 / A + 1 / B = 1 / 40) → (10 * (1 / A + 1 / B) = 1 / 4) →
  (12 * 1 / A = 3 / 4) → A = 16 :=
by
  intros h1 h2 h3
  -- Proof is omitted by "sorry".
  sorry

end work_completion_in_16_days_l2138_213840


namespace all_points_same_value_l2138_213861

theorem all_points_same_value {f : ℤ × ℤ → ℕ}
  (h : ∀ x y : ℤ, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ k : ℕ, ∀ x y : ℤ, f (x, y) = k :=
sorry

end all_points_same_value_l2138_213861


namespace smaller_inscribed_cube_volume_is_192_sqrt_3_l2138_213807

noncomputable def volume_of_smaller_inscribed_cube : ℝ :=
  let edge_length_of_larger_cube := 12
  let diameter_of_sphere := edge_length_of_larger_cube
  let side_length_of_smaller_cube := diameter_of_sphere / Real.sqrt 3
  let volume := side_length_of_smaller_cube ^ 3
  volume

theorem smaller_inscribed_cube_volume_is_192_sqrt_3 : 
  volume_of_smaller_inscribed_cube = 192 * Real.sqrt 3 := 
by
  sorry

end smaller_inscribed_cube_volume_is_192_sqrt_3_l2138_213807


namespace ababab_divisible_by_13_l2138_213831

theorem ababab_divisible_by_13 (a b : ℕ) (ha: a < 10) (hb: b < 10) : 
  13 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) := 
by
  sorry

end ababab_divisible_by_13_l2138_213831


namespace find_length_BD_l2138_213887

theorem find_length_BD (c : ℝ) (h : c ≥ Real.sqrt 7) :
  ∃BD, BD = Real.sqrt (c^2 - 7) :=
sorry

end find_length_BD_l2138_213887


namespace variance_of_scores_l2138_213821

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def mean (xs : List ℕ) : ℚ := xs.sum / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end variance_of_scores_l2138_213821


namespace birth_date_of_older_friend_l2138_213838

/-- Lean 4 statement for the proof problem --/
theorem birth_date_of_older_friend
  (d m y : ℕ)
  (h1 : y ≥ 1900 ∧ y < 2000)
  (h2 : d + 7 < 32) -- Assuming the month has at most 31 days
  (h3 : ((d+7) * 10^4 + m * 10^2 + y % 100) = 6 * (d * 10^4 + m * 10^2 + y % 100))
  (h4 : m > 0 ∧ m < 13)  -- Months are between 1 and 12
  (h5 : (d * 10^4 + m * 10^2 + y % 100) < (d+7) * 10^4 + m * 10^2 + y % 100) -- d < d+7 so older means smaller number
  : d = 1 ∧ m = 4 ∧ y = 1900 :=
by
  sorry -- Proof omitted

end birth_date_of_older_friend_l2138_213838


namespace total_plate_combinations_l2138_213855

open Nat

def valid_letters := 24
def letter_positions := (choose 4 2)
def valid_digits := 10
def total_combinations := letter_positions * (valid_letters * valid_letters) * (valid_digits ^ 3)

theorem total_plate_combinations : total_combinations = 3456000 :=
  by
    -- Replace this sorry with steps to prove the theorem
    sorry

end total_plate_combinations_l2138_213855


namespace total_brownies_correct_l2138_213851

def brownies_initial : Nat := 24
def father_ate : Nat := brownies_initial / 3
def remaining_after_father : Nat := brownies_initial - father_ate
def mooney_ate : Nat := remaining_after_father / 4
def remaining_after_mooney : Nat := remaining_after_father - mooney_ate
def benny_ate : Nat := (remaining_after_mooney * 2) / 5
def remaining_after_benny : Nat := remaining_after_mooney - benny_ate
def snoopy_ate : Nat := 3
def remaining_after_snoopy : Nat := remaining_after_benny - snoopy_ate
def new_batch : Nat := 24
def total_brownies : Nat := remaining_after_snoopy + new_batch

theorem total_brownies_correct : total_brownies = 29 :=
by
  sorry

end total_brownies_correct_l2138_213851


namespace rotated_triangle_forms_two_cones_l2138_213848

/-- Prove that the spatial geometric body formed when a right-angled triangle 
is rotated 360° around its hypotenuse is two cones. -/
theorem rotated_triangle_forms_two_cones (a b c : ℝ) (h1 : a^2 + b^2 = c^2) : 
  ∃ (cones : ℕ), cones = 2 :=
by
  sorry

end rotated_triangle_forms_two_cones_l2138_213848


namespace cost_per_minute_l2138_213858

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l2138_213858


namespace union_of_sets_l2138_213879

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 5 }
def B := { x : ℝ | 3 < x ∧ x < 9 }

theorem union_of_sets : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 9 } :=
by
  sorry

end union_of_sets_l2138_213879


namespace gcd_18_30_45_l2138_213894

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l2138_213894


namespace paula_twice_as_old_as_karl_6_years_later_l2138_213809

theorem paula_twice_as_old_as_karl_6_years_later
  (P K : ℕ)
  (h1 : P - 5 = 3 * (K - 5))
  (h2 : P + K = 54) :
  P + 6 = 2 * (K + 6) :=
sorry

end paula_twice_as_old_as_karl_6_years_later_l2138_213809


namespace max_x_real_nums_l2138_213806

theorem max_x_real_nums (x y z : ℝ) (h₁ : x + y + z = 6) (h₂ : x * y + x * z + y * z = 10) : x ≤ 2 :=
sorry

end max_x_real_nums_l2138_213806


namespace find_x_l2138_213850

variable (BrandA_millet : ℝ) (Mix_millet : ℝ) (Mix_ratio_A : ℝ) (Mix_ratio_B : ℝ)

axiom BrandA_contains_60_percent_millet : BrandA_millet = 0.60
axiom Mix_contains_50_percent_millet : Mix_millet = 0.50
axiom Mix_composition : Mix_ratio_A = 0.60 ∧ Mix_ratio_B = 0.40

theorem find_x (x : ℝ) :
  Mix_ratio_A * BrandA_millet + Mix_ratio_B * x = Mix_millet →
  x = 0.35 :=
by
  sorry

end find_x_l2138_213850


namespace min_val_of_a2_plus_b2_l2138_213860

variable (a b : ℝ)

def condition := 3 * a - 4 * b - 2 = 0

theorem min_val_of_a2_plus_b2 : condition a b → (∃ a b : ℝ, a^2 + b^2 = 4 / 25) := by 
  sorry

end min_val_of_a2_plus_b2_l2138_213860


namespace anna_reading_time_l2138_213834

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l2138_213834


namespace jennie_speed_difference_l2138_213849

noncomputable def average_speed_difference : ℝ :=
  let distance := 200
  let time_heavy_traffic := 5
  let construction_delay := 0.5
  let rest_stops_heavy := 0.5
  let time_no_traffic := 4
  let rest_stops_no_traffic := 1 / 3
  let actual_driving_time_heavy := time_heavy_traffic - construction_delay - rest_stops_heavy
  let actual_driving_time_no := time_no_traffic - rest_stops_no_traffic
  let average_speed_heavy := distance / actual_driving_time_heavy
  let average_speed_no := distance / actual_driving_time_no
  average_speed_no - average_speed_heavy

theorem jennie_speed_difference :
  average_speed_difference = 4.5 :=
sorry

end jennie_speed_difference_l2138_213849


namespace physics_experiment_l2138_213889

theorem physics_experiment (x : ℕ) (h : 1 + x + (x + 1) * x = 36) :
  1 + x + (x + 1) * x = 36 :=
  by                        
  exact h

end physics_experiment_l2138_213889


namespace shorter_leg_of_right_triangle_with_hypotenuse_65_l2138_213877

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 (a b : ℕ) (h : a^2 + b^2 = 65^2) : a = 16 ∨ b = 16 :=
by sorry

end shorter_leg_of_right_triangle_with_hypotenuse_65_l2138_213877


namespace min_x_plus_y_l2138_213857

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y = 9 :=
sorry

end min_x_plus_y_l2138_213857


namespace innings_played_l2138_213853

noncomputable def cricket_player_innings : Nat :=
  let average_runs := 32
  let increase_in_average := 6
  let next_innings_runs := 158
  let new_average := average_runs + increase_in_average
  let runs_before_next_innings (n : Nat) := average_runs * n
  let total_runs_after_next_innings (n : Nat) := runs_before_next_innings n + next_innings_runs
  let total_runs_with_new_average (n : Nat) := new_average * (n + 1)

  let n := (total_runs_after_next_innings 20) - (total_runs_with_new_average 20)
  
  n
     
theorem innings_played : cricket_player_innings = 20 := by
  sorry

end innings_played_l2138_213853


namespace probability_auntie_em_can_park_l2138_213825

/-- A parking lot has 20 spaces in a row. -/
def total_spaces : ℕ := 20

/-- Fifteen cars arrive, each requiring one parking space, and their drivers choose spaces at random from among the available spaces. -/
def cars : ℕ := 15

/-- Auntie Em's SUV requires 3 adjacent empty spaces. -/
def required_adjacent_spaces : ℕ := 3

/-- Calculate the probability that there are 3 consecutive empty spaces among the 5 remaining spaces after 15 cars are parked in 20 spaces.
Expected answer is (12501 / 15504) -/
theorem probability_auntie_em_can_park : 
    (1 - (↑(Nat.choose 15 5) / ↑(Nat.choose 20 5))) = (12501 / 15504) := 
sorry

end probability_auntie_em_can_park_l2138_213825


namespace minimum_value_l2138_213804

theorem minimum_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) : 
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 192 := 
  sorry

end minimum_value_l2138_213804


namespace crayons_in_judahs_box_l2138_213874

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l2138_213874


namespace find_A_when_B_is_largest_l2138_213880

theorem find_A_when_B_is_largest :
  ∃ A : ℕ, ∃ B : ℕ, A = 17 * 25 + B ∧ B < 17 ∧ B = 16 ∧ A = 441 :=
by
  sorry

end find_A_when_B_is_largest_l2138_213880


namespace inequality_of_sum_of_squares_l2138_213884

theorem inequality_of_sum_of_squares (a b c : ℝ) (h : a * b + b * c + a * c = 1) : (a + b + c) ^ 2 ≥ 3 :=
sorry

end inequality_of_sum_of_squares_l2138_213884


namespace find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l2138_213898

def linear_function (a b x : ℝ) : ℝ := a * x + b

theorem find_a_and_b : ∃ (a b : ℝ), 
  linear_function a b 1 = 1 ∧ 
  linear_function a b 2 = -5 ∧ 
  a = -6 ∧ 
  b = 7 :=
sorry

theorem function_value_at_0 : 
  ∀ a b, 
  a = -6 → b = 7 → 
  linear_function a b 0 = 7 :=
sorry

theorem function_positive_x_less_than_7_over_6 :
  ∀ a b x, 
  a = -6 → b = 7 → 
  x < 7 / 6 → 
  linear_function a b x > 0 :=
sorry

end find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l2138_213898


namespace carrots_total_l2138_213892

variables (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat)

def totalCarrots (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat) :=
  initiallyPicked - thrownOut + pickedNextDay

theorem carrots_total (h1 : initiallyPicked = 19)
                     (h2 : thrownOut = 4)
                     (h3 : pickedNextDay = 46) :
  totalCarrots initiallyPicked thrownOut pickedNextDay = 61 :=
by
  sorry

end carrots_total_l2138_213892


namespace arithmetic_mean_of_scores_l2138_213812

theorem arithmetic_mean_of_scores :
  let s1 := 85
  let s2 := 94
  let s3 := 87
  let s4 := 93
  let s5 := 95
  let s6 := 88
  let s7 := 90
  (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7 = 90.2857142857 :=
by
  sorry

end arithmetic_mean_of_scores_l2138_213812


namespace ordered_pair_solution_l2138_213835

theorem ordered_pair_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end ordered_pair_solution_l2138_213835


namespace zeros_of_shifted_function_l2138_213883

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_shifted_function :
  {x : ℝ | f (x - 1) = 0} = {0, 2} :=
sorry

end zeros_of_shifted_function_l2138_213883


namespace kolacky_bounds_l2138_213826

theorem kolacky_bounds (x y : ℕ) (h : 9 * x + 4 * y = 219) :
  294 ≤ 12 * x + 6 * y ∧ 12 * x + 6 * y ≤ 324 :=
sorry

end kolacky_bounds_l2138_213826


namespace range_of_m_l2138_213842

theorem range_of_m (a m : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  m * (a + 1/a) / Real.sqrt 2 > 1 → m ≥ Real.sqrt 2 / 2 := by
  sorry

end range_of_m_l2138_213842


namespace measure_of_angle_D_l2138_213808

-- Definitions of angles in pentagon ABCDE
variables (A B C D E : ℝ)

-- Conditions
def condition1 := D = A + 30
def condition2 := E = A + 50
def condition3 := B = C
def condition4 := A = B - 45
def condition5 := A + B + C + D + E = 540

-- Theorem to prove
theorem measure_of_angle_D (h1 : condition1 A D)
                           (h2 : condition2 A E)
                           (h3 : condition3 B C)
                           (h4 : condition4 A B)
                           (h5 : condition5 A B C D E) :
  D = 104 :=
sorry

end measure_of_angle_D_l2138_213808


namespace days_spent_on_Orbius5_l2138_213866

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l2138_213866
