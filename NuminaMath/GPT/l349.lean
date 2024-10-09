import Mathlib

namespace incenter_x_coordinate_eq_l349_34920

theorem incenter_x_coordinate_eq (x y : ℝ) :
  (x = y) ∧ 
  (y = -x + 3) → 
  x = 3 / 2 := 
sorry

end incenter_x_coordinate_eq_l349_34920


namespace train_length_calculation_l349_34963

noncomputable def train_length (speed_km_hr : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_km_hr * 1000 / 3600) * time_sec

theorem train_length_calculation :
  train_length 250 6 = 416.67 :=
by
  sorry

end train_length_calculation_l349_34963


namespace books_remaining_in_special_collection_l349_34969

theorem books_remaining_in_special_collection
  (initial_books : ℕ)
  (loaned_books : ℕ)
  (returned_percentage : ℕ)
  (initial_books_eq : initial_books = 75)
  (loaned_books_eq : loaned_books = 45)
  (returned_percentage_eq : returned_percentage = 80) :
  ∃ final_books : ℕ, final_books = initial_books - (loaned_books - (loaned_books * returned_percentage / 100)) ∧ final_books = 66 :=
by
  sorry

end books_remaining_in_special_collection_l349_34969


namespace bailey_total_spending_l349_34907

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end bailey_total_spending_l349_34907


namespace cost_price_per_meter_l349_34975

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (h1 : total_cost = 397.75) (h2 : total_length = 9.25) : total_cost / total_length = 43 :=
by
  -- Proof omitted
  sorry

end cost_price_per_meter_l349_34975


namespace tens_digit_of_9_pow_1801_l349_34997

theorem tens_digit_of_9_pow_1801 : 
  ∀ n : ℕ, (9 ^ (1801) % 100) / 10 % 10 = 0 :=
by
  sorry

end tens_digit_of_9_pow_1801_l349_34997


namespace smallest_even_integer_cube_mod_1000_l349_34989

theorem smallest_even_integer_cube_mod_1000 :
  ∃ n : ℕ, (n % 2 = 0) ∧ (n > 0) ∧ (n^3 % 1000 = 392) ∧ (∀ m : ℕ, (m % 2 = 0) ∧ (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m) ∧ n = 892 := 
sorry

end smallest_even_integer_cube_mod_1000_l349_34989


namespace distance_by_which_A_beats_B_l349_34925

noncomputable def speed_of_A : ℝ := 1000 / 192
noncomputable def time_difference : ℝ := 8
noncomputable def distance_beaten : ℝ := speed_of_A * time_difference

theorem distance_by_which_A_beats_B :
  distance_beaten = 41.67 := by
  sorry

end distance_by_which_A_beats_B_l349_34925


namespace coprime_pairs_solution_l349_34931

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end coprime_pairs_solution_l349_34931


namespace expr_eval_l349_34905

noncomputable def expr_value : ℕ :=
  (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6)

theorem expr_eval : expr_value = 18 := by
  sorry

end expr_eval_l349_34905


namespace circle_diameter_l349_34952

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l349_34952


namespace clock_angle_4_oclock_l349_34948

theorem clock_angle_4_oclock :
  let total_degrees := 360
  let hours := 12
  let degree_per_hour := total_degrees / hours
  let hour_position := 4
  let minute_hand_position := 0
  let hour_hand_angle := hour_position * degree_per_hour
  hour_hand_angle = 120 := sorry

end clock_angle_4_oclock_l349_34948


namespace initial_amount_l349_34928

theorem initial_amount (X : ℝ) (h1 : 0.70 * X = 2800) : X = 4000 :=
by
  sorry

end initial_amount_l349_34928


namespace sequence_general_term_l349_34976

open Nat

/-- Define the sequence recursively -/
def a : ℕ → ℤ
| 0     => -1
| (n+1) => 3 * a n - 1

/-- The general term of the sequence is given by - (3^n - 1) / 2 -/
theorem sequence_general_term (n : ℕ) : a n = - (3^n - 1) / 2 := 
by
  sorry

end sequence_general_term_l349_34976


namespace solve_x_l349_34955

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l349_34955


namespace guess_x_30_guess_y_127_l349_34990

theorem guess_x_30 : 120 = 4 * 30 := 
  sorry

theorem guess_y_127 : 87 = 127 - 40 := 
  sorry

end guess_x_30_guess_y_127_l349_34990


namespace second_train_speed_is_correct_l349_34982

noncomputable def speed_of_second_train (length_first : ℝ) (speed_first : ℝ) (time_cross : ℝ) (length_second : ℝ) : ℝ :=
let total_distance := length_first + length_second
let relative_speed := total_distance / time_cross
let relative_speed_kmph := relative_speed * 3.6
relative_speed_kmph - speed_first

theorem second_train_speed_is_correct :
  speed_of_second_train 270 120 9 230.04 = 80.016 :=
by
  sorry

end second_train_speed_is_correct_l349_34982


namespace fraction_equation_solution_l349_34962

theorem fraction_equation_solution (x : ℝ) (h : x ≠ 3) : (2 - x) / (x - 3) + 3 = 2 / (3 - x) ↔ x = 5 / 2 := by
  sorry

end fraction_equation_solution_l349_34962


namespace distance_between_locations_A_and_B_l349_34994

-- Define the conditions
variables {x y s t : ℝ}

-- Conditions specified in the problem
axiom bus_a_meets_bus_b_after_85_km : 85 / x = (s - 85) / y 
axiom buses_meet_again_after_turnaround : (s - 85 + 65) / x + 1 / 2 = (85 + (s - 65)) / y + 1 / 2

-- The theorem to be proved
theorem distance_between_locations_A_and_B : s = 190 :=
by
  sorry

end distance_between_locations_A_and_B_l349_34994


namespace find_m_l349_34979

variable (a b m : ℝ)

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem find_m 
  (h₁ : right_triangle a b 5)
  (h₂ : a + b = 2*m - 1)
  (h₃ : a * b = 4 * (m - 1)) : 
  m = 4 := 
sorry

end find_m_l349_34979


namespace geometric_series_sum_l349_34941

noncomputable def first_term : ℝ := 6
noncomputable def common_ratio : ℝ := -2 / 3

theorem geometric_series_sum :
  (|common_ratio| < 1) → (first_term / (1 - common_ratio) = 18 / 5) :=
by
  intros h
  simp [first_term, common_ratio]
  sorry

end geometric_series_sum_l349_34941


namespace sum_squares_l349_34935

theorem sum_squares (w x y z : ℝ) (h1 : w + x + y + z = 0) (h2 : w^2 + x^2 + y^2 + z^2 = 1) :
  -1 ≤ w * x + x * y + y * z + z * w ∧ w * x + x * y + y * z + z * w ≤ 0 := 
by 
  sorry

end sum_squares_l349_34935


namespace interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l349_34946

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi →
    ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧
      (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem axis_of_symmetry (k : ℤ) :
  ∃ x : ℝ, x = Real.pi / 3 + k * (Real.pi / 2) := sorry

theorem max_and_min_values :
  ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      ((f x = 2 ∧ x = Real.pi / 3) ∨ (f x = -1 ∧ x = 0))) := sorry

end interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l349_34946


namespace lcm_of_9_12_18_l349_34985

-- Let's declare the numbers involved
def num1 : ℕ := 9
def num2 : ℕ := 12
def num3 : ℕ := 18

-- Define what it means for a number to be the LCM of num1, num2, and num3
def is_lcm (a b c l : ℕ) : Prop :=
  l % a = 0 ∧ l % b = 0 ∧ l % c = 0 ∧
  ∀ m, (m % a = 0 ∧ m % b = 0 ∧ m % c = 0) → l ≤ m

-- Now state the theorem
theorem lcm_of_9_12_18 : is_lcm num1 num2 num3 36 :=
by
  sorry

end lcm_of_9_12_18_l349_34985


namespace group_sizes_correct_l349_34974

-- Define the number of fruits and groups
def num_bananas : Nat := 527
def num_oranges : Nat := 386
def num_apples : Nat := 319

def groups_bananas : Nat := 11
def groups_oranges : Nat := 103
def groups_apples : Nat := 17

-- Define the expected sizes of each group
def bananas_per_group : Nat := 47
def oranges_per_group : Nat := 3
def apples_per_group : Nat := 18

-- Prove the sizes of the groups are as expected
theorem group_sizes_correct :
  (num_bananas / groups_bananas = bananas_per_group) ∧
  (num_oranges / groups_oranges = oranges_per_group) ∧
  (num_apples / groups_apples = apples_per_group) :=
by
  -- Division in Nat rounds down
  have h1 : num_bananas / groups_bananas = 47 := by sorry
  have h2 : num_oranges / groups_oranges = 3 := by sorry
  have h3 : num_apples / groups_apples = 18 := by sorry
  exact ⟨h1, h2, h3⟩

end group_sizes_correct_l349_34974


namespace increase_80_by_50_percent_l349_34978

theorem increase_80_by_50_percent : 
  let original_number := 80
  let percentage_increase := 0.5
  let increase := original_number * percentage_increase
  let final_number := original_number + increase
  final_number = 120 := 
by 
  sorry

end increase_80_by_50_percent_l349_34978


namespace students_with_both_pets_l349_34939

theorem students_with_both_pets :
  ∀ (total_students students_with_dog students_with_cat students_with_both : ℕ),
    total_students = 45 →
    students_with_dog = 25 →
    students_with_cat = 34 →
    total_students = students_with_dog + students_with_cat - students_with_both →
    students_with_both = 14 :=
by
  intros total_students students_with_dog students_with_cat students_with_both
  sorry

end students_with_both_pets_l349_34939


namespace problem_part_1_problem_part_2_l349_34988

noncomputable def f (x : ℝ) (m : ℝ) := |x + 1| + |x - 2| - m

theorem problem_part_1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 3} :=
by sorry

theorem problem_part_2 (h : ∀ x : ℝ, f x m ≥ 2) : m ≤ 1 :=
by sorry

end problem_part_1_problem_part_2_l349_34988


namespace commute_time_variance_l349_34934

theorem commute_time_variance
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) :
  x^2 + y^2 = 208 :=
by
  sorry

end commute_time_variance_l349_34934


namespace distinct_natural_numbers_circles_sum_equal_impossible_l349_34966

theorem distinct_natural_numbers_circles_sum_equal_impossible :
  ¬∃ (f : ℕ → ℕ) (distinct : ∀ i j, i ≠ j → f i ≠ f j) (equal_sum : ∀ i j k, (f i + f j + f k = f (i+1) + f (j+1) + f (k+1))),
  true :=
  sorry

end distinct_natural_numbers_circles_sum_equal_impossible_l349_34966


namespace total_amount_paid_l349_34930

theorem total_amount_paid (B : ℕ) (hB : B = 232) (A : ℕ) (hA : A = 3 / 2 * B) :
  A + B = 580 :=
by
  sorry

end total_amount_paid_l349_34930


namespace gcd_polynomials_l349_34953

def even_multiple_of_2927 (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * 2927 * k

theorem gcd_polynomials (a : ℤ) (h : even_multiple_of_2927 a) :
  Int.gcd (3 * a ^ 2 + 61 * a + 143) (a + 19) = 7 :=
by
  sorry

end gcd_polynomials_l349_34953


namespace daily_wage_of_c_is_71_l349_34923

theorem daily_wage_of_c_is_71 (x : ℚ) :
  let a_days := 16
  let b_days := 9
  let c_days := 4
  let total_earnings := 1480
  let wage_ratio_a := 3
  let wage_ratio_b := 4
  let wage_ratio_c := 5
  let total_contribution := a_days * wage_ratio_a * x + b_days * wage_ratio_b * x + c_days * wage_ratio_c * x
  total_contribution = total_earnings →
  c_days * wage_ratio_c * x = 71 := by
  sorry

end daily_wage_of_c_is_71_l349_34923


namespace circles_internally_tangent_l349_34957

theorem circles_internally_tangent :
  let C1 := (3, -2)
  let r1 := 1
  let C2 := (7, 1)
  let r2 := 6
  let d := Real.sqrt (((7 - 3)^2 + (1 - (-2))^2) : ℝ)
  d = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l349_34957


namespace largest_even_k_for_sum_of_consecutive_integers_l349_34972

theorem largest_even_k_for_sum_of_consecutive_integers (k n : ℕ) (h_k_even : k % 2 = 0) :
  (3^10 = k * (2 * n + k + 1)) → k ≤ 162 :=
sorry

end largest_even_k_for_sum_of_consecutive_integers_l349_34972


namespace ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l349_34927

theorem ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one
  (m n : ℕ) : (10 ^ m + 1) % (10 ^ n - 1) ≠ 0 := 
  sorry

end ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l349_34927


namespace mia_high_school_has_2000_students_l349_34903

variables (M Z : ℕ)

def mia_high_school_students : Prop :=
  M = 4 * Z ∧ M + Z = 2500

theorem mia_high_school_has_2000_students (h : mia_high_school_students M Z) : 
  M = 2000 := by
  sorry

end mia_high_school_has_2000_students_l349_34903


namespace double_series_evaluation_l349_34900

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, if h : n ≥ m then 1 / (m * n * (m + n + 2)) else 0) = (Real.pi ^ 2) / 6 := sorry

end double_series_evaluation_l349_34900


namespace sum_of_exterior_segment_angles_is_540_l349_34937

-- Define the setup of the problem
def quadrilateral_inscribed_in_circle (A B C D : Type) : Prop := sorry
def angle_externally_inscribed (segment : Type) : ℝ := sorry

-- Main theorem statement
theorem sum_of_exterior_segment_angles_is_540
  (A B C D : Type)
  (h_quad : quadrilateral_inscribed_in_circle A B C D)
  (alpha beta gamma delta : ℝ)
  (h_alpha : alpha = angle_externally_inscribed A)
  (h_beta : beta = angle_externally_inscribed B)
  (h_gamma : gamma = angle_externally_inscribed C)
  (h_delta : delta = angle_externally_inscribed D) :
  alpha + beta + gamma + delta = 540 :=
sorry

end sum_of_exterior_segment_angles_is_540_l349_34937


namespace angle_A_l349_34993

variable (a b c : ℝ) (A B C : ℝ)

-- Hypothesis: In triangle ABC, (a + c)(a - c) = b(b + c)
def condition (a b c : ℝ) : Prop := (a + c) * (a - c) = b * (b + c)

-- The goal is to show that under given conditions, ∠A = 2π/3
theorem angle_A (h : condition a b c) : A = 2 * π / 3 :=
sorry

end angle_A_l349_34993


namespace power_inequality_l349_34938

theorem power_inequality (a b n : ℕ) (h_ab : a > b) (h_b1 : b > 1)
  (h_odd_b : b % 2 = 1) (h_n_pos : 0 < n) (h_div : b^n ∣ a^n - 1) :
  a^b > 3^n / n :=
by
  sorry

end power_inequality_l349_34938


namespace value_of_a_l349_34932

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l349_34932


namespace find_p_of_binomial_distribution_l349_34981

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem find_p_of_binomial_distribution (p : ℝ) (h : binomial_mean 5 p = 2) : p = 0.4 :=
by
  sorry

end find_p_of_binomial_distribution_l349_34981


namespace equate_operations_l349_34921

theorem equate_operations :
  (15 * 5) / (10 + 2) = 3 → 8 / 4 = 2 → ((18 * 6) / (14 + 4) = 6) :=
by
sorry

end equate_operations_l349_34921


namespace minimum_value_of_z_l349_34904

theorem minimum_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : ∃ min_z, min_z = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → min_z ≤ z :=
by
  sorry

end minimum_value_of_z_l349_34904


namespace find_E_l349_34942

variable (A H C S M N E : ℕ)
variable (x y z l : ℕ)

theorem find_E (h1 : A * x + H * y + C * z = l)
 (h2 : S * x + M * y + N * z = l)
 (h3 : E * x = l)
 (h4 : A ≠ S ∧ A ≠ H ∧ A ≠ C ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧ H ≠ C ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧ M ≠ N ∧ M ≠ E ∧ N ≠ E)
 : E = (A * M + C * N - S * H - N * H) / (M + N - H) := 
sorry

end find_E_l349_34942


namespace stratified_sampling_num_of_female_employees_l349_34929

theorem stratified_sampling_num_of_female_employees :
  ∃ (total_employees male_employees sample_size female_employees_to_draw : ℕ),
    total_employees = 750 ∧
    male_employees = 300 ∧
    sample_size = 45 ∧
    female_employees_to_draw = (total_employees - male_employees) * sample_size / total_employees ∧
    female_employees_to_draw = 27 :=
by
  sorry

end stratified_sampling_num_of_female_employees_l349_34929


namespace polygon_sides_l349_34991

theorem polygon_sides {S n : ℕ} (h : S = 2160) (hs : S = 180 * (n - 2)) : n = 14 := 
by
  sorry

end polygon_sides_l349_34991


namespace inscribed_triangle_area_l349_34949

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C)

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  ∀ (A B C : ℝ), A = 40 * Real.pi / 180 → B = 80 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  triangle_area r A B C = 359.4384 / Real.pi^2 :=
by
  intros
  unfold triangle_area
  sorry

end inscribed_triangle_area_l349_34949


namespace required_HCl_moles_l349_34987

-- Definitions of chemical substances:
def HCl: Type := Unit
def NaHCO3: Type := Unit
def H2O: Type := Unit
def CO2: Type := Unit
def NaCl: Type := Unit

-- The reaction as a balanced chemical equation:
def balanced_eq (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl) : Prop :=
  ∃ (m: ℕ), m = 1

-- Given conditions:
def condition1: Prop := balanced_eq () () () () ()
def condition2 (moles_H2O moles_CO2 moles_NaCl: ℕ): Prop :=
  moles_H2O = moles_CO2 ∧ moles_CO2 = moles_NaCl ∧ moles_NaCl = moles_H2O

def condition3: ℕ := 3  -- moles of NaHCO3

-- The theorem statement:
theorem required_HCl_moles (moles_HCl moles_NaHCO3: ℕ)
  (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl)
  (balanced: balanced_eq hcl nahco3 h2o co2 nacl)
  (equal_moles: condition2 moles_H2O moles_CO2 moles_NaCl)
  (nahco3_eq_3: moles_NaHCO3 = condition3):
  moles_HCl = 3 :=
sorry

end required_HCl_moles_l349_34987


namespace pencils_initial_count_l349_34995

theorem pencils_initial_count (pencils_initially: ℕ) :
  (∀ n, n > 0 → n < 36 → 36 % n = 1) →
  pencils_initially + 30 = 36 → 
  pencils_initially = 6 :=
by
  intro h hn
  sorry

end pencils_initial_count_l349_34995


namespace exchange_rate_5_CAD_to_JPY_l349_34922

theorem exchange_rate_5_CAD_to_JPY :
  (1 : ℝ) * 85 * 5 = 425 :=
by
  sorry

end exchange_rate_5_CAD_to_JPY_l349_34922


namespace no_real_roots_l349_34910

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^6 - 3 * x^5 + 6 * x^4 - 6 * x^3 - x + 8

-- The problem can be stated as proving that Q(x) has no real roots
theorem no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end no_real_roots_l349_34910


namespace multiple_of_interest_rate_l349_34961

theorem multiple_of_interest_rate (P r m : ℝ) (h1 : P * r^2 = 40) (h2 : P * (m * r)^2 = 360) : m = 3 :=
by
  sorry

end multiple_of_interest_rate_l349_34961


namespace ramsey_theorem_six_people_l349_34970

theorem ramsey_theorem_six_people (S : Finset Person)
  (hS: S.card = 6)
  (R : Person → Person → Prop): 
  (∃ (has_relation : Person → Person → Prop), 
    ∀ A B : Person, A ≠ B → R A B ∨ ¬ R A B) →
  (∃ (T : Finset Person), T.card = 3 ∧ 
    ((∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → R x y) ∨ 
     (∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → ¬ R x y))) :=
by
  sorry

end ramsey_theorem_six_people_l349_34970


namespace find_age_l349_34901

open Nat

-- Definition of ages
def Teacher_Zhang_age (z : Nat) := z
def Wang_Bing_age (w : Nat) := w

-- Conditions
axiom teacher_zhang_condition (z w : Nat) : z = 3 * w + 4
axiom age_comparison_condition (z w : Nat) : z - 10 = w + 10

-- Proposition to prove
theorem find_age (z w : Nat) (hz : z = 3 * w + 4) (hw : z - 10 = w + 10) : z = 28 ∧ w = 8 := by
  sorry

end find_age_l349_34901


namespace range_of_m_l349_34933

variable (f : ℝ → ℝ) (m : ℝ)

-- Given conditions
def condition1 := ∀ x, f (-x) = -f x -- f(x) is an odd function
def condition2 := ∀ x, f (x + 3) = f x -- f(x) has a minimum positive period of 3
def condition3 := f 2015 > 1 -- f(2015) > 1
def condition4 := f 1 = (2 * m + 3) / (m - 1) -- f(1) = (2m + 3) / (m - 1)

-- We aim to prove that -2/3 < m < 1 given these conditions.
theorem range_of_m : condition1 f → condition2 f → condition3 f → condition4 f m → -2 / 3 < m ∧ m < 1 := by
  intros
  sorry

end range_of_m_l349_34933


namespace probability_odd_3_in_6_rolls_l349_34959

-- Definitions based on problem conditions
def probability_of_odd (outcome: ℕ) : ℚ := if outcome % 2 = 1 then 1/2 else 0 

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  ((Nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k)))

-- Given problem
theorem probability_odd_3_in_6_rolls : 
  binomial_probability 6 3 (1/2) = 5 / 16 :=
by
  sorry

end probability_odd_3_in_6_rolls_l349_34959


namespace minimum_value_l349_34967

noncomputable def f (x : ℝ) (a b : ℝ) := a^x - b
noncomputable def g (x : ℝ) := x + 1

theorem minimum_value (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f (0 : ℝ) a b * g 0 ≤ 0)
  (h4 : ∀ x : ℝ, f x a b * g x ≤ 0) : (1 / a + 4 / b) ≥ 4 :=
sorry

end minimum_value_l349_34967


namespace original_loaf_slices_l349_34914

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l349_34914


namespace Veenapaniville_high_schools_l349_34971

theorem Veenapaniville_high_schools :
  ∃ (districtA districtB districtC : ℕ),
    districtA + districtB + districtC = 50 ∧
    (districtA + districtB + districtC = 50) ∧
    (∃ (publicB parochialB privateB : ℕ), 
      publicB + parochialB + privateB = 17 ∧ privateB = 2) ∧
    (∃ (publicC parochialC privateC : ℕ),
      publicC = 9 ∧ parochialC = 9 ∧ privateC = 9 ∧ publicC + parochialC + privateC = 27) ∧
    districtB = 17 ∧
    districtC = 27 →
    districtA = 6 := by
  sorry

end Veenapaniville_high_schools_l349_34971


namespace solution_set_of_inequality_l349_34992

theorem solution_set_of_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l349_34992


namespace sqrt_sum_l349_34947

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l349_34947


namespace correct_operation_l349_34913

theorem correct_operation :
  (∀ (a : ℤ), 2 * a - a ≠ 1) ∧
  (∀ (a : ℤ), (a^2)^4 ≠ a^6) ∧
  (∀ (a b : ℤ), (a * b)^2 ≠ a * b^2) ∧
  (∀ (a : ℤ), a^3 * a^2 = a^5) :=
by
  sorry

end correct_operation_l349_34913


namespace comb_15_6_eq_5005_perm_6_eq_720_l349_34944

open Nat

-- Prove that \frac{15!}{6!(15-6)!} = 5005
theorem comb_15_6_eq_5005 : (factorial 15) / (factorial 6 * factorial (15 - 6)) = 5005 := by
  sorry

-- Prove that the number of ways to arrange 6 items in a row is 720
theorem perm_6_eq_720 : factorial 6 = 720 := by
  sorry

end comb_15_6_eq_5005_perm_6_eq_720_l349_34944


namespace incorrect_height_is_151_l349_34960

def incorrect_height (average_initial correct_height average_corrected : ℝ) : ℝ :=
  (30 * average_initial) - (30 * average_corrected) + correct_height

theorem incorrect_height_is_151 :
  incorrect_height 175 136 174.5 = 151 :=
by
  sorry

end incorrect_height_is_151_l349_34960


namespace cos_double_angle_l349_34940

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l349_34940


namespace raisins_in_other_three_boxes_l349_34919

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end raisins_in_other_three_boxes_l349_34919


namespace exists_b_gt_a_divides_l349_34965

theorem exists_b_gt_a_divides (a : ℕ) (h : 0 < a) :
  ∃ b : ℕ, b > a ∧ (1 + 2^a + 3^a) ∣ (1 + 2^b + 3^b) :=
sorry

end exists_b_gt_a_divides_l349_34965


namespace jane_20_cent_items_l349_34902

theorem jane_20_cent_items {x y z : ℕ} (h1 : x + y + z = 50) (h2 : 20 * x + 150 * y + 250 * z = 5000) : x = 31 :=
by
  -- The formal proof would go here
  sorry

end jane_20_cent_items_l349_34902


namespace p_sufficient_not_necessary_for_q_l349_34996

variable (a : ℝ)

def p : Prop := a > 0
def q : Prop := a^2 + a ≥ 0

theorem p_sufficient_not_necessary_for_q : (p a → q a) ∧ ¬ (q a → p a) := by
  sorry

end p_sufficient_not_necessary_for_q_l349_34996


namespace determine_m_first_degree_inequality_l349_34917

theorem determine_m_first_degree_inequality (m : ℝ) (x : ℝ) :
  (m + 1) * x ^ |m| + 2 > 0 → |m| = 1 → m = 1 :=
by
  intro h1 h2
  sorry

end determine_m_first_degree_inequality_l349_34917


namespace complement_of_A_in_U_l349_34968

def U : Set ℤ := {x | -2 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {x | ∃ n : ℕ, (x = 2 * n ∧ n ≤ 3)}

theorem complement_of_A_in_U : (U \ A) = {-2, -1, 1, 3, 5} :=
by
  sorry

end complement_of_A_in_U_l349_34968


namespace logarithm_argument_positive_l349_34958

open Real

theorem logarithm_argument_positive (a : ℝ) : 
  (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + a * sin x * cos x > 0) ↔ -1 / 2 < a ∧ a < 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end logarithm_argument_positive_l349_34958


namespace intersection_M_N_l349_34912

def set_M : Set ℝ := { x | x < 2 }
def set_N : Set ℝ := { x | x > 0 }
def set_intersection : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_M_N : set_M ∩ set_N = set_intersection := 
by
  sorry

end intersection_M_N_l349_34912


namespace dice_probability_sum_15_l349_34964

def is_valid_combination (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 15

def count_outcomes : ℕ :=
  6 * 6 * 6

def count_valid_combinations : ℕ :=
  10  -- From the list of valid combinations

def probability (valid_count total_count : ℕ) : ℚ :=
  valid_count / total_count

theorem dice_probability_sum_15 : probability count_valid_combinations count_outcomes = 5 / 108 :=
by
  sorry

end dice_probability_sum_15_l349_34964


namespace floor_ceiling_sum_l349_34954

theorem floor_ceiling_sum : 
    Int.floor (0.998 : ℝ) + Int.ceil (2.002 : ℝ) = 3 := by
  sorry

end floor_ceiling_sum_l349_34954


namespace minimum_employees_for_identical_training_l349_34918

def languages : Finset String := {"English", "French", "Spanish", "German"}

noncomputable def choose_pairings_count (n k : ℕ) : ℕ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem minimum_employees_for_identical_training 
  (num_languages : ℕ := 4) 
  (employees_per_pairing : ℕ := 4)
  (pairings : ℕ := choose_pairings_count num_languages 2) 
  (total_employees : ℕ := employees_per_pairing * pairings)
  (minimum_employees : ℕ := total_employees + 1):
  minimum_employees = 25 :=
by
  -- We skip the proof details as per the instructions
  sorry

end minimum_employees_for_identical_training_l349_34918


namespace cuboid_edge_length_l349_34973

-- This is the main statement we want to prove
theorem cuboid_edge_length (L : ℝ) (w : ℝ) (h : ℝ) (V : ℝ) (w_eq : w = 5) (h_eq : h = 3) (V_eq : V = 30) :
  V = L * w * h → L = 2 :=
by
  -- Adding the sorry allows us to compile and acknowledge the current placeholder for the proof.
  sorry

end cuboid_edge_length_l349_34973


namespace binom_eq_sum_l349_34926

theorem binom_eq_sum (x : ℕ) : (∃ x : ℕ, Nat.choose 7 x = 21) ∧ Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4 :=
by
  sorry

end binom_eq_sum_l349_34926


namespace multiply_polynomials_l349_34936

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l349_34936


namespace largest_final_digit_l349_34950

theorem largest_final_digit (seq : Fin 1002 → Fin 10) 
  (h1 : seq 0 = 2) 
  (h2 : ∀ n : Fin 1001, (17 ∣ (10 * seq n + seq (n + 1))) ∨ (29 ∣ (10 * seq n + seq (n + 1)))) : 
  seq 1001 = 5 :=
sorry

end largest_final_digit_l349_34950


namespace cone_sphere_ratio_l349_34906

theorem cone_sphere_ratio (r h : ℝ) (h_r_ne_zero : r ≠ 0)
  (h_vol_cone : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := 
by
  sorry

end cone_sphere_ratio_l349_34906


namespace hexagon_monochromatic_triangles_l349_34998

theorem hexagon_monochromatic_triangles :
  let hexagon_edges := 15 -- $\binom{6}{2}$
  let monochromatic_tri_prob := (1 / 3) -- Prob of one triangle being monochromatic
  let combinations := 20 -- $\binom{6}{3}$, total number of triangles in K_6
  let exactly_two_monochromatic := (combinations.choose 2) * (monochromatic_tri_prob ^ 2) * ((2 / 3) ^ 18)
  (exactly_two_monochromatic = 49807360 / 3486784401) := sorry

end hexagon_monochromatic_triangles_l349_34998


namespace remaining_days_to_complete_job_l349_34986

-- Define the given conditions
def in_10_days (part_of_job_done : ℝ) (days : ℕ) : Prop :=
  part_of_job_done = 1 / 8 ∧ days = 10

-- Define the complete job condition
def complete_job (total_days : ℕ) : Prop :=
  total_days = 80

-- Define the remaining days to finish the job
def remaining_days (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) : Prop :=
  total_days_worked = 80 ∧ days_worked = 10 ∧ remaining = 70

-- The theorem statement
theorem remaining_days_to_complete_job (part_of_job_done : ℝ) (days : ℕ) (total_days : ℕ) (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) :
  in_10_days part_of_job_done days → complete_job total_days → remaining_days total_days_worked days_worked remaining :=
sorry

end remaining_days_to_complete_job_l349_34986


namespace trigonometric_identity_l349_34924

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) = - (1 / 3) := 
by
  sorry

end trigonometric_identity_l349_34924


namespace largest_three_digit_number_l349_34943

theorem largest_three_digit_number (a b c : ℕ) (h1 : a = 8) (h2 : b = 0) (h3 : c = 7) :
  ∃ (n : ℕ), ∀ (x : ℕ), (x = a * 100 + b * 10 + c) → x = 870 :=
by
  sorry

end largest_three_digit_number_l349_34943


namespace total_caps_produced_l349_34956

-- Define the production of each week as given in the conditions.
def week1_caps : ℕ := 320
def week2_caps : ℕ := 400
def week3_caps : ℕ := 300

-- Define the average of the first three weeks.
def average_caps : ℕ := (week1_caps + week2_caps + week3_caps) / 3

-- Define the production increase for the fourth week.
def increase_caps : ℕ := average_caps / 5  -- 20% is equivalent to dividing by 5

-- Calculate the total production for the fourth week (including the increase).
def week4_caps : ℕ := average_caps + increase_caps

-- Calculate the total number of caps produced in four weeks.
def total_caps : ℕ := week1_caps + week2_caps + week3_caps + week4_caps

-- Theorem stating the total production over the four weeks.
theorem total_caps_produced : total_caps = 1428 := by sorry

end total_caps_produced_l349_34956


namespace all_three_items_fans_l349_34951

theorem all_three_items_fans 
  (h1 : ∀ n, n = 4800 % 80 → n = 0)
  (h2 : ∀ n, n = 4800 % 40 → n = 0)
  (h3 : ∀ n, n = 4800 % 60 → n = 0)
  (h4 : ∀ n, n = 4800):
  (∃ k, k = 20):=
by
  sorry

end all_three_items_fans_l349_34951


namespace Caroline_lost_4_pairs_of_socks_l349_34977

theorem Caroline_lost_4_pairs_of_socks 
  (initial_pairs : ℕ) (pairs_donated_fraction : ℚ)
  (new_pairs_purchased : ℕ) (new_pairs_gifted : ℕ)
  (final_pairs : ℕ) (L : ℕ) :
  initial_pairs = 40 →
  pairs_donated_fraction = 2/3 →
  new_pairs_purchased = 10 →
  new_pairs_gifted = 3 →
  final_pairs = 25 →
  (initial_pairs - L) * (1 - pairs_donated_fraction) + new_pairs_purchased + new_pairs_gifted = final_pairs →
  L = 4 :=
by {
  sorry
}

end Caroline_lost_4_pairs_of_socks_l349_34977


namespace impossible_transformation_l349_34983

def f (x : ℝ) := x^2 + 5 * x + 4
def g (x : ℝ) := x^2 + 10 * x + 8

theorem impossible_transformation :
  (∀ x, f (x) = x^2 + 5 * x + 4) →
  (∀ x, g (x) = x^2 + 10 * x + 8) →
  (¬ ∃ t : ℝ → ℝ → ℝ, (∀ x, t (f x) x = g x)) :=
by
  sorry

end impossible_transformation_l349_34983


namespace perpendicular_lines_l349_34911

def line1 (m : ℝ) (x y : ℝ) := m * x - 3 * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) := (3 * m - 2) * x - m * y + 2 = 0

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y) →
  (∀ x y : ℝ, line2 m x y) →
  (∀ x y : ℝ, (m / 3) * ((3 * m - 2) / m) = -1) →
  m = 0 ∨ m = -1/3 :=
by
  intros
  sorry

end perpendicular_lines_l349_34911


namespace odd_function_expression_l349_34980

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2*x) :
  ∀ x : ℝ, f x = x * (|x| - 2) :=
by
  sorry

end odd_function_expression_l349_34980


namespace range_of_m_l349_34908

namespace MathProof

def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - m * x + m - 1 = 0 }

theorem range_of_m (m : ℝ) (h : A ∪ (B m) = A) : m = 3 :=
  sorry

end MathProof

end range_of_m_l349_34908


namespace complete_square_formula_D_l349_34916

-- Definitions of polynomial multiplications
def poly_A (a b : ℝ) : ℝ := (a - b) * (a + b)
def poly_B (a b : ℝ) : ℝ := -((a + b) * (b - a))
def poly_C (a b : ℝ) : ℝ := (a + b) * (b - a)
def poly_D (a b : ℝ) : ℝ := (a - b) * (b - a)

theorem complete_square_formula_D (a b : ℝ) : 
  poly_D a b = -(a - b)*(a - b) :=
by sorry

end complete_square_formula_D_l349_34916


namespace area_of_triangle_ABC_l349_34945

theorem area_of_triangle_ABC (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let y_vertex := (4 * a * c - b^2) / (4 * a)
  0.5 * (|x2 - x1|) * |y_vertex| = (b^2 - 4 * a * c) * Real.sqrt (b^2 - 4 * a * c) / (8 * a^2) :=
sorry

end area_of_triangle_ABC_l349_34945


namespace hayden_earnings_l349_34915

theorem hayden_earnings 
  (wage_per_hour : ℕ) 
  (pay_per_ride : ℕ)
  (bonus_per_review : ℕ)
  (number_of_rides : ℕ)
  (hours_worked : ℕ)
  (gas_cost_per_gallon : ℕ)
  (gallons_of_gas : ℕ)
  (positive_reviews : ℕ)
  : wage_per_hour = 15 → 
    pay_per_ride = 5 → 
    bonus_per_review = 20 → 
    number_of_rides = 3 → 
    hours_worked = 8 → 
    gas_cost_per_gallon = 3 → 
    gallons_of_gas = 17 → 
    positive_reviews = 2 → 
    (hours_worked * wage_per_hour + number_of_rides * pay_per_ride + positive_reviews * bonus_per_review + gallons_of_gas * gas_cost_per_gallon) = 226 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Further proof processing with these assumptions
  sorry

end hayden_earnings_l349_34915


namespace find_distance_between_stations_l349_34999

noncomputable def distance_between_stations (D T : ℝ) : Prop :=
  D = 100 * T ∧
  D = 50 * (T + 15 / 60) ∧
  D = 70 * (T + 7 / 60)

theorem find_distance_between_stations :
  ∃ D T : ℝ, distance_between_stations D T ∧ D = 25 :=
by
  sorry

end find_distance_between_stations_l349_34999


namespace inequality_abc_l349_34984

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c)) :=
by
  sorry

end inequality_abc_l349_34984


namespace max_player_salary_l349_34909

theorem max_player_salary (n : ℕ) (min_salary total_salary : ℕ) (player_count : ℕ)
  (h1 : player_count = 25)
  (h2 : min_salary = 15000)
  (h3 : total_salary = 850000)
  (h4 : n = 24 * min_salary)
  : (total_salary - n) = 490000 := 
by
  -- assumptions ensure that n represents the total minimum salaries paid to 24 players
  sorry

end max_player_salary_l349_34909
