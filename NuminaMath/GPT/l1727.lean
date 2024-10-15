import Mathlib

namespace NUMINAMATH_GPT_rectangle_base_length_l1727_172704

theorem rectangle_base_length
  (h : ℝ) (b : ℝ)
  (common_height_nonzero : h ≠ 0)
  (triangle_base : ℝ := 24)
  (same_area : (1/2) * triangle_base * h = b * h) :
  b = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_base_length_l1727_172704


namespace NUMINAMATH_GPT_younger_brother_age_l1727_172726

variable (x y : ℕ)

-- Conditions
axiom sum_of_ages : x + y = 46
axiom younger_is_third_plus_ten : y = x / 3 + 10

theorem younger_brother_age : y = 19 := 
by
  sorry

end NUMINAMATH_GPT_younger_brother_age_l1727_172726


namespace NUMINAMATH_GPT_simplify_expression_l1727_172712

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1727_172712


namespace NUMINAMATH_GPT_cost_price_of_table_l1727_172744

theorem cost_price_of_table (CP : ℝ) (SP : ℝ) (h1 : SP = CP * 1.10) (h2 : SP = 8800) : CP = 8000 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_table_l1727_172744


namespace NUMINAMATH_GPT_scientific_notation_26_billion_l1727_172720

theorem scientific_notation_26_billion :
  ∃ (a : ℝ) (n : ℤ), (26 * 10^8 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.6 ∧ n = 9 :=
sorry

end NUMINAMATH_GPT_scientific_notation_26_billion_l1727_172720


namespace NUMINAMATH_GPT_find_a_l1727_172736

variable {x y a : ℤ}

theorem find_a (h1 : 3 * x + y = 1 + 3 * a) (h2 : x + 3 * y = 1 - a) (h3 : x + y = 0) : a = -1 := 
sorry

end NUMINAMATH_GPT_find_a_l1727_172736


namespace NUMINAMATH_GPT_xy_eq_zero_l1727_172740

theorem xy_eq_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end NUMINAMATH_GPT_xy_eq_zero_l1727_172740


namespace NUMINAMATH_GPT_no_conclusions_deducible_l1727_172779

open Set

variable {U : Type}  -- Universe of discourse

-- Conditions
variables (Bars Fins Grips : Set U)

def some_bars_are_not_fins := ∃ x, x ∈ Bars ∧ x ∉ Fins
def no_fins_are_grips := ∀ x, x ∈ Fins → x ∉ Grips

-- Lean statement
theorem no_conclusions_deducible 
  (h1 : some_bars_are_not_fins Bars Fins)
  (h2 : no_fins_are_grips Fins Grips) :
  ¬((∃ x, x ∈ Bars ∧ x ∉ Grips) ∨
    (∃ x, x ∈ Grips ∧ x ∉ Bars) ∨
    (∀ x, x ∈ Bars → x ∉ Grips) ∨
    (∃ x, x ∈ Bars ∧ x ∈ Grips)) :=
sorry

end NUMINAMATH_GPT_no_conclusions_deducible_l1727_172779


namespace NUMINAMATH_GPT_range_of_a_for_monotonic_function_l1727_172730

theorem range_of_a_for_monotonic_function (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 0 ≤ (1 / x) + a) → a ≥ -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_monotonic_function_l1727_172730


namespace NUMINAMATH_GPT_brian_traveled_correct_distance_l1727_172716

def miles_per_gallon : Nat := 20
def gallons_used : Nat := 3
def expected_miles : Nat := 60

theorem brian_traveled_correct_distance : (miles_per_gallon * gallons_used) = expected_miles := by
  sorry

end NUMINAMATH_GPT_brian_traveled_correct_distance_l1727_172716


namespace NUMINAMATH_GPT_arithmetic_sequence_21st_term_l1727_172755

theorem arithmetic_sequence_21st_term (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 13) (h3 : a 3 = 23) :
  a 21 = 203 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_21st_term_l1727_172755


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1727_172729

/-- Let {a_n} be a geometric sequence with positive common ratio, a_1 = 2, and a_3 = a_2 + 4.
    Prove the general formula for a_n is 2^n, and the sum of the first n terms, S_n, of the sequence { (2n+1)a_n }
    is (2n-1) * 2^(n+1) + 2. -/
theorem geometric_sequence_sum
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h3 : a 3 = a 2 + 4) :
  (∀ n, a n = 2^n) ∧
  (∀ S : ℕ → ℕ, ∀ n, S n = (2 * n - 1) * 2 ^ (n + 1) + 2) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1727_172729


namespace NUMINAMATH_GPT_find_cd_l1727_172702

noncomputable def period := (3 / 4) * Real.pi
noncomputable def x_value := (1 / 8) * Real.pi
noncomputable def y_value := 3
noncomputable def tangent_value := Real.tan (Real.pi / 6) -- which is 1 / sqrt(3)
noncomputable def c_value := 3 * Real.sqrt 3

theorem find_cd (c d : ℝ) 
  (h_period : d = 4 / 3) 
  (h_point : y_value = c * Real.tan (d * x_value)) :
  c * d = 4 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_find_cd_l1727_172702


namespace NUMINAMATH_GPT_brick_width_is_correct_l1727_172719

-- Defining conditions
def wall_length : ℝ := 200 -- wall length in cm
def wall_width : ℝ := 300 -- wall width in cm
def wall_height : ℝ := 2   -- wall height in cm
def brick_length : ℝ := 25 -- brick length in cm
def brick_height : ℝ := 6  -- brick height in cm
def num_bricks : ℝ := 72.72727272727273

-- Total volume of wall
def vol_wall : ℝ := wall_length * wall_width * wall_height

-- Volume of one brick
def vol_brick (width : ℝ) : ℝ := brick_length * width * brick_height

-- Proof statement
theorem brick_width_is_correct : ∃ width : ℝ, vol_wall = vol_brick width * num_bricks ∧ width = 11 :=
by
  sorry

end NUMINAMATH_GPT_brick_width_is_correct_l1727_172719


namespace NUMINAMATH_GPT_smallest_value_3a_2_l1727_172741

theorem smallest_value_3a_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 2 = - (5 / 2) := sorry

end NUMINAMATH_GPT_smallest_value_3a_2_l1727_172741


namespace NUMINAMATH_GPT_units_digit_sum_2_pow_a_5_pow_b_l1727_172737

theorem units_digit_sum_2_pow_a_5_pow_b (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 100)
  (h2 : 1 ≤ b ∧ b ≤ 100) :
  (2 ^ a + 5 ^ b) % 10 ≠ 8 :=
sorry

end NUMINAMATH_GPT_units_digit_sum_2_pow_a_5_pow_b_l1727_172737


namespace NUMINAMATH_GPT_minimum_value_l1727_172771

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 3)

theorem minimum_value : 
  (1 / (3 * a + 5 * b)) + (1 / (3 * b + 5 * c)) + (1 / (3 * c + 5 * a)) ≥ 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1727_172771


namespace NUMINAMATH_GPT_original_radius_eq_n_div_3_l1727_172746

theorem original_radius_eq_n_div_3 (r n : ℝ) (h : (r + n)^2 = 4 * r^2) : r = n / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_radius_eq_n_div_3_l1727_172746


namespace NUMINAMATH_GPT_salary_increase_l1727_172798

theorem salary_increase (original_salary reduced_salary : ℝ) (hx : reduced_salary = original_salary * 0.5) : 
  (reduced_salary + reduced_salary * 1) = original_salary :=
by
  -- Prove the required increase percent to return to original salary
  sorry

end NUMINAMATH_GPT_salary_increase_l1727_172798


namespace NUMINAMATH_GPT_remainder_of_72nd_integers_div_by_8_is_5_l1727_172751

theorem remainder_of_72nd_integers_div_by_8_is_5 (s : Set ℤ) (h₁ : ∀ x ∈ s, ∃ k : ℤ, x = 8 * k + r) 
  (h₂ : 573 ∈ (s : Set ℤ)) : 
  ∃ (r : ℤ), r = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_72nd_integers_div_by_8_is_5_l1727_172751


namespace NUMINAMATH_GPT_proposition_d_correct_l1727_172734

theorem proposition_d_correct (a b c : ℝ) (h : a > b) : a - c > b - c := 
by
  sorry

end NUMINAMATH_GPT_proposition_d_correct_l1727_172734


namespace NUMINAMATH_GPT_comic_book_arrangement_l1727_172789

theorem comic_book_arrangement :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let groups := 3
  Nat.factorial spiderman_books * Nat.factorial archie_books * Nat.factorial garfield_books * Nat.factorial groups = 248005440 :=
by
  sorry

end NUMINAMATH_GPT_comic_book_arrangement_l1727_172789


namespace NUMINAMATH_GPT_g_of_neg_5_is_4_l1727_172750

def f (x : ℝ) : ℝ := 3 * x - 8
def g (y : ℝ) : ℝ := 2 * y^2 + 5 * y - 3

theorem g_of_neg_5_is_4 : g (-5) = 4 :=
by
  sorry

end NUMINAMATH_GPT_g_of_neg_5_is_4_l1727_172750


namespace NUMINAMATH_GPT_max_S_n_l1727_172714

theorem max_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) (h2 : d < 0) (h3 : S 6 = 5 * (a 1) + 10 * d) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, S m ≤ S n) :=
by
  sorry

end NUMINAMATH_GPT_max_S_n_l1727_172714


namespace NUMINAMATH_GPT_tens_digit_19_2021_l1727_172743

theorem tens_digit_19_2021 : (19^2021 % 100) / 10 % 10 = 1 :=
by sorry

end NUMINAMATH_GPT_tens_digit_19_2021_l1727_172743


namespace NUMINAMATH_GPT_bee_total_correct_l1727_172752

def initial_bees : Nat := 16
def incoming_bees : Nat := 10
def total_bees : Nat := initial_bees + incoming_bees

theorem bee_total_correct : total_bees = 26 := by
  sorry

end NUMINAMATH_GPT_bee_total_correct_l1727_172752


namespace NUMINAMATH_GPT_acme_cheaper_min_shirts_l1727_172727

theorem acme_cheaper_min_shirts :
  ∃ x : ℕ, 60 + 11 * x < 10 + 16 * x ∧ x = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_acme_cheaper_min_shirts_l1727_172727


namespace NUMINAMATH_GPT_average_percentage_of_kernels_popped_l1727_172793

theorem average_percentage_of_kernels_popped :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 82
  let bag3_total := 100
  let percentage (popped total : ℕ) := (popped : ℚ) / total * 100
  let p1 := percentage bag1_popped bag1_total
  let p2 := percentage bag2_popped bag2_total
  let p3 := percentage bag3_popped bag3_total
  let avg := (p1 + p2 + p3) / 3
  avg = 82 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_of_kernels_popped_l1727_172793


namespace NUMINAMATH_GPT_magnitude_z1_condition_z2_range_condition_l1727_172706

-- Define and set up the conditions and problem statements
open Complex

def complex_number_condition (z₁ : ℂ) (m : ℝ) : Prop :=
  z₁ = 1 + m * I ∧ ((z₁ * (1 - I)).re = 0)

def z₂_condition (z₂ z₁ : ℂ) (n : ℝ) : Prop :=
  z₂ = z₁ * (n - I) ∧ z₂.re < 0 ∧ z₂.im < 0

-- Prove that if z₁ = 1 + m * I and z₁ * (1 - I) is pure imaginary, then |z₁| = sqrt 2
theorem magnitude_z1_condition (m : ℝ) (z₁ : ℂ) 
  (h₁ : complex_number_condition z₁ m) : abs z₁ = Real.sqrt 2 :=
by sorry

-- Prove that if z₂ = z₁ * (n + i^3) is in the third quadrant, then n is in the range (-1, 1)
theorem z2_range_condition (n : ℝ) (m : ℝ) (z₁ z₂ : ℂ)
  (h₁ : complex_number_condition z₁ m)
  (h₂ : z₂_condition z₂ z₁ n) : -1 < n ∧ n < 1 :=
by sorry

end NUMINAMATH_GPT_magnitude_z1_condition_z2_range_condition_l1727_172706


namespace NUMINAMATH_GPT_tangency_of_parabolas_l1727_172787

theorem tangency_of_parabolas :
  ∃ x y : ℝ, y = x^2 + 12*x + 40
  ∧ x = y^2 + 44*y + 400
  ∧ x = -11 / 2
  ∧ y = -43 / 2 := by
sorry

end NUMINAMATH_GPT_tangency_of_parabolas_l1727_172787


namespace NUMINAMATH_GPT_shipping_cost_correct_l1727_172742

-- Definitions of given conditions
def total_weight_of_fish : ℕ := 540
def weight_of_each_crate : ℕ := 30
def total_shipping_cost : ℚ := 27

-- Calculating the number of crates
def number_of_crates : ℕ := total_weight_of_fish / weight_of_each_crate

-- Definition of the target shipping cost per crate
def shipping_cost_per_crate : ℚ := total_shipping_cost / number_of_crates

-- Lean statement to prove the given problem
theorem shipping_cost_correct :
  shipping_cost_per_crate = 1.50 := by
  sorry

end NUMINAMATH_GPT_shipping_cost_correct_l1727_172742


namespace NUMINAMATH_GPT_hyperbola_equation_l1727_172757

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                           (h3 : b = 2 * a) (h4 : ((4 : ℝ), 1) ∈ {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1}) :
    {p : ℝ × ℝ | (p.1)^2 / 12 - (p.2)^2 / 3 = 1} = {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1} :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1727_172757


namespace NUMINAMATH_GPT_tickets_per_ride_factor_l1727_172772

theorem tickets_per_ride_factor (initial_tickets spent_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 40) 
  (h2 : spent_tickets = 28) 
  (h3 : remaining_tickets = initial_tickets - spent_tickets) : 
  ∃ k : ℕ, remaining_tickets = 12 ∧ (∀ m : ℕ, m ∣ remaining_tickets → m = k) → (k ∣ 12) :=
by
  sorry

end NUMINAMATH_GPT_tickets_per_ride_factor_l1727_172772


namespace NUMINAMATH_GPT_train_length_is_250_l1727_172748

-- Define the length of the train
def train_length (L : ℝ) (V : ℝ) :=
  -- Condition 1
  (V = L / 10) → 
  -- Condition 2
  (V = (L + 1250) / 60) → 
  -- Question
  L = 250

-- Here's the statement that we expect to prove
theorem train_length_is_250 (L V : ℝ) : train_length L V :=
by {
  -- sorry is a placeholder to indicate the theorem proof is omitted
  sorry
}

end NUMINAMATH_GPT_train_length_is_250_l1727_172748


namespace NUMINAMATH_GPT_find_number_l1727_172723

theorem find_number (x : ℤ) (h : (x + 305) / 16 = 31) : x = 191 :=
sorry

end NUMINAMATH_GPT_find_number_l1727_172723


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1727_172722

theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : a + a * r = 8)
  (h2 : a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 120) :
  a * (1 + r + r^2 + r^3) = 30 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1727_172722


namespace NUMINAMATH_GPT_mike_peaches_eq_120_l1727_172731

def original_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def total_peaches (orig : ℝ) (picked : ℝ) : ℝ := orig + picked

theorem mike_peaches_eq_120 : total_peaches original_peaches picked_peaches = 120.0 := 
by
  sorry

end NUMINAMATH_GPT_mike_peaches_eq_120_l1727_172731


namespace NUMINAMATH_GPT_solve_fraction_eq_l1727_172717

theorem solve_fraction_eq : 
  ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 := by
  intros x h_ne_zero h_eq
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1727_172717


namespace NUMINAMATH_GPT_find_a_l1727_172756

def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

theorem find_a (a : ℝ) (h : F a 3 4 = F a 2 5) : a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1727_172756


namespace NUMINAMATH_GPT_inequality_proof_l1727_172715

variable (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
variable (h_sum : a + b + c + d = 1)

theorem inequality_proof :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ (1 / 27) + (176 / 27) * a * b * c * d :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1727_172715


namespace NUMINAMATH_GPT_intersecting_circles_l1727_172768

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end NUMINAMATH_GPT_intersecting_circles_l1727_172768


namespace NUMINAMATH_GPT_max_sin_A_plus_sin_C_l1727_172732

variables {a b c S : ℝ}
variables {A B C : ℝ}

-- Assume the sides of the triangle
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Assume the angles of the triangle
variables (hA : A > 0) (hB : B > (Real.pi / 2)) (hC : C > 0)
variables (hSumAngles : A + B + C = Real.pi)

-- Assume the relationship between the area and the sides
variables (hArea : S = (1/2) * a * c * Real.sin B)

-- Assume the given equation holds
variables (hEquation : 4 * b * S = a * (b^2 + c^2 - a^2))

-- The statement to prove
theorem max_sin_A_plus_sin_C : (Real.sin A + Real.sin C) ≤ 9 / 8 :=
sorry

end NUMINAMATH_GPT_max_sin_A_plus_sin_C_l1727_172732


namespace NUMINAMATH_GPT_find_t_l1727_172707

variables (s t : ℚ)

theorem find_t (h1 : 12 * s + 7 * t = 154) (h2 : s = 2 * t - 3) : t = 190 / 31 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1727_172707


namespace NUMINAMATH_GPT_trains_total_distance_l1727_172739

theorem trains_total_distance (speedA_kmph speedB_kmph time_min : ℕ)
                             (hA : speedA_kmph = 70)
                             (hB : speedB_kmph = 90)
                             (hT : time_min = 15) :
    let speedA_kmpm := (speedA_kmph : ℝ) / 60
    let speedB_kmpm := (speedB_kmph : ℝ) / 60
    let distanceA := speedA_kmpm * (time_min : ℝ)
    let distanceB := speedB_kmpm * (time_min : ℝ)
    distanceA + distanceB = 40 := 
by 
  sorry

end NUMINAMATH_GPT_trains_total_distance_l1727_172739


namespace NUMINAMATH_GPT_cleaned_area_correct_l1727_172799

def lizzie_cleaned : ℚ := 3534 + 2/3
def hilltown_team_cleaned : ℚ := 4675 + 5/8
def green_valley_cleaned : ℚ := 2847 + 7/9
def riverbank_cleaned : ℚ := 6301 + 1/3
def meadowlane_cleaned : ℚ := 3467 + 4/5

def total_cleaned : ℚ := lizzie_cleaned + hilltown_team_cleaned + green_valley_cleaned + riverbank_cleaned + meadowlane_cleaned
def total_farmland : ℚ := 28500

def remaining_area_to_clean : ℚ := total_farmland - total_cleaned

theorem cleaned_area_correct : remaining_area_to_clean = 7672.7964 :=
by
  sorry

end NUMINAMATH_GPT_cleaned_area_correct_l1727_172799


namespace NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1727_172708

theorem ratio_of_ages_in_two_years
    (S : ℕ) (M : ℕ) 
    (h1 : M = S + 32)
    (h2 : S = 30) : 
    (M + 2) / (S + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1727_172708


namespace NUMINAMATH_GPT_sum_of_first_19_terms_l1727_172760

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a1 + a_n a1 d n)

theorem sum_of_first_19_terms (a1 d : ℝ) (h : a1 + 9 * d = 1) : S_n a1 d 19 = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_19_terms_l1727_172760


namespace NUMINAMATH_GPT_inequality_solution_set_l1727_172725

theorem inequality_solution_set 
  (c : ℝ) (a : ℝ) (b : ℝ) (h : c > 0) (hb : b = (5 / 2) * c) (ha : a = - (3 / 2) * c) :
  ∀ x : ℝ, (a * x^2 + b * x + c ≥ 0) ↔ (- (1 / 3) ≤ x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1727_172725


namespace NUMINAMATH_GPT_solve_x_1_solve_x_2_solve_x_3_l1727_172747

-- Proof 1: Given 356 * x = 2492, prove that x = 7
theorem solve_x_1 (x : ℕ) (h : 356 * x = 2492) : x = 7 :=
sorry

-- Proof 2: Given x / 39 = 235, prove that x = 9165
theorem solve_x_2 (x : ℕ) (h : x / 39 = 235) : x = 9165 :=
sorry

-- Proof 3: Given 1908 - x = 529, prove that x = 1379
theorem solve_x_3 (x : ℕ) (h : 1908 - x = 529) : x = 1379 :=
sorry

end NUMINAMATH_GPT_solve_x_1_solve_x_2_solve_x_3_l1727_172747


namespace NUMINAMATH_GPT_exponent_problem_l1727_172738

theorem exponent_problem (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_exponent_problem_l1727_172738


namespace NUMINAMATH_GPT_martha_total_clothes_l1727_172770

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end NUMINAMATH_GPT_martha_total_clothes_l1727_172770


namespace NUMINAMATH_GPT_length_of_base_l1727_172749

-- Define the conditions of the problem
def base_of_triangle (b : ℕ) : Prop :=
  ∃ c : ℕ, b + 3 + c = 12 ∧ 9 + b*b = c*c

-- Statement to prove
theorem length_of_base : base_of_triangle 4 :=
  sorry

end NUMINAMATH_GPT_length_of_base_l1727_172749


namespace NUMINAMATH_GPT_num_ordered_pairs_l1727_172728

theorem num_ordered_pairs : ∃ (n : ℕ), n = 24 ∧ ∀ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 → n = 24 :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_l1727_172728


namespace NUMINAMATH_GPT_caffeine_over_goal_l1727_172766

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end NUMINAMATH_GPT_caffeine_over_goal_l1727_172766


namespace NUMINAMATH_GPT_speeds_of_cars_l1727_172759

theorem speeds_of_cars (d_A d_B : ℝ) (v_A v_B : ℝ) (h1 : d_A = 300) (h2 : d_B = 250) (h3 : v_A = v_B + 5) (h4 : d_A / v_A = d_B / v_B) :
  v_B = 25 ∧ v_A = 30 :=
by
  sorry

end NUMINAMATH_GPT_speeds_of_cars_l1727_172759


namespace NUMINAMATH_GPT_math_problem_l1727_172777

theorem math_problem
  (x : ℕ) (y : ℕ)
  (h1 : x = (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id)
  (h2 : y = ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card)
  (h3 : x + y = 611) :
  (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id = 605 ∧
  ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card = 6 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1727_172777


namespace NUMINAMATH_GPT_line_always_passes_through_fixed_point_l1727_172721

theorem line_always_passes_through_fixed_point (k : ℝ) : 
  ∀ x y, y + 2 = k * (x + 1) → (x = -1 ∧ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_line_always_passes_through_fixed_point_l1727_172721


namespace NUMINAMATH_GPT_total_number_of_birds_l1727_172754

theorem total_number_of_birds (B C G S W : ℕ) (h1 : C = 2 * B) (h2 : G = 4 * B)
  (h3 : S = (C + G) / 2) (h4 : W = 8) (h5 : B = 2 * W) :
  C + G + S + W + B = 168 :=
  by
  sorry

end NUMINAMATH_GPT_total_number_of_birds_l1727_172754


namespace NUMINAMATH_GPT_total_surface_area_with_holes_l1727_172745

def cube_edge_length : ℝ := 5
def hole_side_length : ℝ := 2

/-- Calculate the total surface area of a modified cube with given edge length and holes -/
theorem total_surface_area_with_holes 
  (l : ℝ) (h : ℝ)
  (hl_pos : l > 0) (hh_pos : h > 0) (hh_lt_hl : h < l) : 
  (6 * l^2 - 6 * h^2 + 6 * 4 * h^2) = 222 :=
by sorry

end NUMINAMATH_GPT_total_surface_area_with_holes_l1727_172745


namespace NUMINAMATH_GPT_polynomial_solution_l1727_172794

noncomputable def p (x : ℝ) := 2 * Real.sqrt 3 * x^4 - 6

theorem polynomial_solution (x : ℝ) : 
  (p (x^4) - p (x^4 - 3) = (p x)^3 - 18) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1727_172794


namespace NUMINAMATH_GPT_circle_radius_3_l1727_172735

theorem circle_radius_3 :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y - 7 = 0) → (∃ r : ℝ, r = 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_3_l1727_172735


namespace NUMINAMATH_GPT_actual_cost_of_article_l1727_172791

theorem actual_cost_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 :=
sorry

end NUMINAMATH_GPT_actual_cost_of_article_l1727_172791


namespace NUMINAMATH_GPT_not_basic_logic_structure_l1727_172753

def SequenceStructure : Prop := true
def ConditionStructure : Prop := true
def LoopStructure : Prop := true
def DecisionStructure : Prop := true

theorem not_basic_logic_structure : ¬ (SequenceStructure ∨ ConditionStructure ∨ LoopStructure) -> DecisionStructure := by
  sorry

end NUMINAMATH_GPT_not_basic_logic_structure_l1727_172753


namespace NUMINAMATH_GPT_rex_has_399_cards_left_l1727_172785

def Nicole_cards := 700

def Cindy_cards := 3 * Nicole_cards + (40 / 100) * (3 * Nicole_cards)
def Tim_cards := (4 / 5) * Cindy_cards
def combined_total := Nicole_cards + Cindy_cards + Tim_cards
def Rex_and_Joe_cards := (60 / 100) * combined_total

def cards_per_person := Nat.floor (Rex_and_Joe_cards / 9)

theorem rex_has_399_cards_left : cards_per_person = 399 := by
  sorry

end NUMINAMATH_GPT_rex_has_399_cards_left_l1727_172785


namespace NUMINAMATH_GPT_city_partition_exists_l1727_172724

-- Define a market and street as given
structure City where
  markets : Type
  street : markets → markets → Prop
  leaves_exactly_two : ∀ (m : markets), ∃ (m1 m2 : markets), street m m1 ∧ street m m2

-- Our formal proof statement
theorem city_partition_exists (C : City) : 
  ∃ (partition : C.markets → Fin 1014), 
    (∀ (m1 m2 : C.markets), C.street m1 m2 → partition m1 ≠ partition m2) ∧
    (∀ (d1 d2 : Fin 1014) (m1 m2 : C.markets), (partition m1 = d1) ∧ (partition m2 = d2) → 
     (C.street m1 m2 ∨ C.street m2 m1) →  (∀ (k l : Fin 1014), (k = d1) → (l = d2) → (∀ (a b : C.markets), (partition a = k) → (partition b = l) → (C.street a b ∨ C.street b a)))) :=
sorry

end NUMINAMATH_GPT_city_partition_exists_l1727_172724


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l1727_172788

noncomputable def S (a₁ q : ℝ) : ℝ := a₁ / (1 - q)

theorem necessary_not_sufficient_condition (a₁ q : ℝ) (h₁ : |q| < 1) :
  (a₁ + q = 1) → (S a₁ q = 1) ∧ ¬((S a₁ q = 1) → (a₁ + q = 1)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l1727_172788


namespace NUMINAMATH_GPT_modulus_of_z_eq_sqrt2_l1727_172765

noncomputable def complex_z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

theorem modulus_of_z_eq_sqrt2 : Complex.abs complex_z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_modulus_of_z_eq_sqrt2_l1727_172765


namespace NUMINAMATH_GPT_pet_store_cages_l1727_172783

theorem pet_store_cages 
  (snakes parrots rabbits snake_cage_capacity parrot_cage_capacity rabbit_cage_capacity : ℕ)
  (h_snakes : snakes = 4) 
  (h_parrots : parrots = 6) 
  (h_rabbits : rabbits = 8) 
  (h_snake_cage_capacity : snake_cage_capacity = 2) 
  (h_parrot_cage_capacity : parrot_cage_capacity = 3) 
  (h_rabbit_cage_capacity : rabbit_cage_capacity = 4) 
  : (snakes / snake_cage_capacity) + (parrots / parrot_cage_capacity) + (rabbits / rabbit_cage_capacity) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_pet_store_cages_l1727_172783


namespace NUMINAMATH_GPT_sum_of_coefficients_equals_28_l1727_172718

def P (x : ℝ) : ℝ :=
  2 * (4 * x^8 - 5 * x^5 + 9 * x^3 - 6) + 8 * (x^6 - 4 * x^3 + 6)

theorem sum_of_coefficients_equals_28 : P 1 = 28 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_equals_28_l1727_172718


namespace NUMINAMATH_GPT_area_triangle_QCA_l1727_172713

/--
  Given:
  - θ (θ is acute) is the angle at Q between QA and QC
  - Q is at the coordinates (0, 12)
  - A is at the coordinates (3, 12)
  - C is at the coordinates (0, p)

  Prove that the area of triangle QCA is (3/2) * (12 - p) * sin(θ).
-/
theorem area_triangle_QCA (p θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let base := 3
  let height := (12 - p) * Real.sin θ
  let area := (1 / 2) * base * height
  area = (3 / 2) * (12 - p) * Real.sin θ := by
  sorry

end NUMINAMATH_GPT_area_triangle_QCA_l1727_172713


namespace NUMINAMATH_GPT_correct_operation_l1727_172711

theorem correct_operation (a b : ℝ) :
  (3 * a^2 - a^2 ≠ 3) ∧
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((-3 * a * b^2)^2 ≠ -6 * a^2 * b^4) →
  a^3 / a^2 = a :=
by
sorry

end NUMINAMATH_GPT_correct_operation_l1727_172711


namespace NUMINAMATH_GPT_min_students_l1727_172767

theorem min_students (S a b c : ℕ) (h1 : 3 * a > S) (h2 : 10 * b > 3 * S) (h3 : 11 * c > 4 * S) (h4 : S = a + b + c) : S ≥ 173 :=
by
  sorry

end NUMINAMATH_GPT_min_students_l1727_172767


namespace NUMINAMATH_GPT_unique_spicy_pair_l1727_172773

def is_spicy (n : ℕ) : Prop :=
  let A := (n / 100) % 10
  let B := (n / 10) % 10
  let C := n % 10
  n = A^3 + B^3 + C^3

theorem unique_spicy_pair : ∃! n : ℕ, is_spicy n ∧ is_spicy (n + 1) ∧ 100 ≤ n ∧ n < 1000 ∧ n = 370 := 
sorry

end NUMINAMATH_GPT_unique_spicy_pair_l1727_172773


namespace NUMINAMATH_GPT_intersection_M_N_l1727_172764

def M := {x : ℝ | x < 1}

def N := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1727_172764


namespace NUMINAMATH_GPT_inverse_B2_l1727_172776

def matrix_B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 7; -2, -4]

def matrix_B2_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-5, -7; 2, 2]

theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) ℝ) (hB_inv : B⁻¹ = matrix_B_inv) :
  (B^2)⁻¹ = matrix_B2_inv :=
sorry

end NUMINAMATH_GPT_inverse_B2_l1727_172776


namespace NUMINAMATH_GPT_triangle_ratio_l1727_172778

theorem triangle_ratio (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let p := 12;
  let q := 8;
  let segment_length := L / p;
  let segment_width := W / q;
  let area_X := (segment_length * segment_width) / 2;
  let area_rectangle := L * W;
  (area_X / area_rectangle) = (1 / 192) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_ratio_l1727_172778


namespace NUMINAMATH_GPT_four_digit_numbers_sum_even_l1727_172774

theorem four_digit_numbers_sum_even : 
  ∃ N : ℕ, 
    (∀ (digits : Finset ℕ) (thousands hundreds tens units : ℕ), 
      digits = {1, 2, 3, 4, 5, 6} ∧ 
      ∀ n ∈ digits, (0 < n ∧ n < 10) ∧ 
      (thousands ∈ digits ∧ hundreds ∈ digits ∧ tens ∈ digits ∧ units ∈ digits) ∧ 
      (thousands ≠ hundreds ∧ thousands ≠ tens ∧ thousands ≠ units ∧ 
       hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units) ∧ 
      (tens + units) % 2 = 0 → N = 324) :=
sorry

end NUMINAMATH_GPT_four_digit_numbers_sum_even_l1727_172774


namespace NUMINAMATH_GPT_mrs_martin_pays_l1727_172775

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def mr_martin_scoops : ℕ := 1
def mrs_martin_scoops : ℕ := 1
def children_scoops : ℕ := 2
def teenage_children_scoops : ℕ := 3

def total_cost : ℕ :=
  (mr_martin_scoops + mrs_martin_scoops) * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost

theorem mrs_martin_pays : total_cost = 32 :=
  by sorry

end NUMINAMATH_GPT_mrs_martin_pays_l1727_172775


namespace NUMINAMATH_GPT_region_diff_correct_l1727_172709

noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
noncomputable def one_triangle_area : ℝ := (Real.sqrt 3) / 4
noncomputable def triangles_area : ℝ := 18 * one_triangle_area
noncomputable def R_area : ℝ := hexagon_area + triangles_area
noncomputable def S_area : ℝ := 4 * (1 + Real.sqrt 2)
noncomputable def region_diff : ℝ := S_area - R_area

theorem region_diff_correct :
  region_diff = 4 + 4 * Real.sqrt 2 - 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_region_diff_correct_l1727_172709


namespace NUMINAMATH_GPT_find_xy_l1727_172796

theorem find_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 :=
sorry

end NUMINAMATH_GPT_find_xy_l1727_172796


namespace NUMINAMATH_GPT_time_to_complete_job_l1727_172701

-- Define the conditions
variables {A B : ℕ} -- Efficiencies of A and B

-- Assume B's efficiency is 100 units, and A is 130 units.
def efficiency_A : ℕ := 130
def efficiency_B : ℕ := 100

-- Given: A can complete the job in 23 days
def days_A : ℕ := 23

-- Compute total work W. Since A can complete the job in 23 days and its efficiency is 130 units/day:
def total_work : ℕ := efficiency_A * days_A

-- Combined efficiency of A and B
def combined_efficiency : ℕ := efficiency_A + efficiency_B

-- Determine the time taken by A and B working together
def time_A_B_together : ℕ := total_work / combined_efficiency

-- Prove that the time A and B working together is 13 days
theorem time_to_complete_job : time_A_B_together = 13 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_time_to_complete_job_l1727_172701


namespace NUMINAMATH_GPT_range_of_f_l1727_172786

noncomputable def f (x : ℝ) := Real.log (2 - x^2) / Real.log (1 / 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l1727_172786


namespace NUMINAMATH_GPT_show_revenue_and_vacancies_l1727_172705

theorem show_revenue_and_vacancies:
  let total_seats := 600
  let vip_seats := 50
  let general_seats := 400
  let balcony_seats := 150
  let vip_price := 40
  let general_price := 25
  let balcony_price := 15
  let vip_filled_rate := 0.80
  let general_filled_rate := 0.70
  let balcony_filled_rate := 0.50
  let vip_filled := vip_filled_rate * vip_seats
  let general_filled := general_filled_rate * general_seats
  let balcony_filled := balcony_filled_rate * balcony_seats
  let vip_revenue := vip_filled * vip_price
  let general_revenue := general_filled * general_price
  let balcony_revenue := balcony_filled * balcony_price
  let overall_revenue := vip_revenue + general_revenue + balcony_revenue
  let vip_vacant := vip_seats - vip_filled
  let general_vacant := general_seats - general_filled
  let balcony_vacant := balcony_seats - balcony_filled
  vip_revenue = 1600 ∧
  general_revenue = 7000 ∧
  balcony_revenue = 1125 ∧
  overall_revenue = 9725 ∧
  vip_vacant = 10 ∧
  general_vacant = 120 ∧
  balcony_vacant = 75 :=
by
  sorry

end NUMINAMATH_GPT_show_revenue_and_vacancies_l1727_172705


namespace NUMINAMATH_GPT_cats_in_shelter_l1727_172762

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end NUMINAMATH_GPT_cats_in_shelter_l1727_172762


namespace NUMINAMATH_GPT_largest_divisor_of_product_l1727_172758

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_product_l1727_172758


namespace NUMINAMATH_GPT_Robert_GRE_exam_l1727_172733

/-- Robert started preparation for GRE entrance examination in the month of January and prepared for 5 months. Prove that he could write the examination any date after the end of May.-/
theorem Robert_GRE_exam (start_month : ℕ) (prep_duration : ℕ) : 
  start_month = 1 → prep_duration = 5 → ∃ exam_date, exam_date > 5 :=
by
  sorry

end NUMINAMATH_GPT_Robert_GRE_exam_l1727_172733


namespace NUMINAMATH_GPT_obtuse_triangle_sum_range_l1727_172795

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_sum_range_l1727_172795


namespace NUMINAMATH_GPT_problem_statement_l1727_172790

namespace CoinFlipping

/-- 
Define the probability that Alice and Bob both get the same number of heads
when flipping three coins where two are fair and one is biased with a probability
of 3/5 for heads. We aim to calculate p + q where p/q is this probability and 
output the final result - p + q should equal 263.
-/
def same_heads_probability_sum : ℕ :=
  let p := 63
  let q := 200
  p + q

theorem problem_statement : same_heads_probability_sum = 263 :=
  by
  -- proof to be filled in
  sorry

end CoinFlipping

end NUMINAMATH_GPT_problem_statement_l1727_172790


namespace NUMINAMATH_GPT_unique_valid_quintuple_l1727_172710

theorem unique_valid_quintuple :
  ∃! (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
    a^2 + b^2 + c^3 + d^3 + e^3 = 5 ∧
    (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25 :=
sorry

end NUMINAMATH_GPT_unique_valid_quintuple_l1727_172710


namespace NUMINAMATH_GPT_find_m_range_l1727_172703

noncomputable def quadratic_inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem find_m_range :
  { m : ℝ | quadratic_inequality_condition m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
sorry

end NUMINAMATH_GPT_find_m_range_l1727_172703


namespace NUMINAMATH_GPT_simplify_expression_l1727_172784

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : (2 * x ^ 3) ^ 3 = 8 * x ^ 9 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1727_172784


namespace NUMINAMATH_GPT_initial_group_size_l1727_172780

theorem initial_group_size
  (n : ℕ) (W : ℝ)
  (h_avg_increase : ∀ W n, ((W + 12) / n) = (W / n + 3))
  (h_new_person_weight : 82 = 70 + 12) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_group_size_l1727_172780


namespace NUMINAMATH_GPT_triangle_construction_conditions_l1727_172761

open Classical

noncomputable def construct_triangle (m_a m_b s_c : ℝ) : Prop :=
  m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c

theorem triangle_construction_conditions (m_a m_b s_c : ℝ) :
  construct_triangle m_a m_b s_c ↔ (m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_construction_conditions_l1727_172761


namespace NUMINAMATH_GPT_fractional_equation_positive_root_l1727_172781

theorem fractional_equation_positive_root (a : ℝ) (ha : ∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_fractional_equation_positive_root_l1727_172781


namespace NUMINAMATH_GPT_find_pairs_of_numbers_l1727_172792

theorem find_pairs_of_numbers (a b : ℝ) :
  (a^2 + b^2 = 15 * (a + b)) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b))
  ↔ (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
sorry

end NUMINAMATH_GPT_find_pairs_of_numbers_l1727_172792


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1727_172782

theorem sufficient_not_necessary (x : ℝ) : (x > 3 → x > 1) ∧ ¬ (x > 1 → x > 3) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1727_172782


namespace NUMINAMATH_GPT_correct_comparison_l1727_172797

theorem correct_comparison :
  ( 
    (-1 > -0.1) = false ∧ 
    (-4 / 3 < -5 / 4) = true ∧ 
    (-1 / 2 > -(-1 / 3)) = false ∧ 
    (Real.pi = 3.14) = false 
  ) :=
by
  sorry

end NUMINAMATH_GPT_correct_comparison_l1727_172797


namespace NUMINAMATH_GPT_work_ratio_l1727_172769

theorem work_ratio (r : ℕ) (w : ℕ) (m₁ m₂ d₁ d₂ : ℕ)
  (h₁ : m₁ = 5) 
  (h₂ : d₁ = 15) 
  (h₃ : m₂ = 3) 
  (h₄ : d₂ = 25)
  (h₅ : w = (m₁ * r * d₁) + (m₂ * r * d₂)) :
  ((m₁ * r * d₁):ℚ) / (w:ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_work_ratio_l1727_172769


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1727_172763

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Definition of the expected intersection of A and B
def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem stating the proof problem
theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x ∈ expected_intersection :=
by
  intro x
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1727_172763


namespace NUMINAMATH_GPT_Q_div_P_l1727_172700

theorem Q_div_P (P Q : ℚ) (h : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
  P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x * (x + 3) * (x - 5))) :
  Q / P = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_Q_div_P_l1727_172700
