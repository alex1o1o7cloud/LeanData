import Mathlib

namespace correct_calculation_l1518_151809

theorem correct_calculation (m n : ℝ) : 4 * m + 2 * n - (n - m) = 5 * m + n :=
by sorry

end correct_calculation_l1518_151809


namespace valid_arrangements_count_is_20_l1518_151876

noncomputable def count_valid_arrangements : ℕ :=
  sorry

theorem valid_arrangements_count_is_20 :
  count_valid_arrangements = 20 :=
  by
    sorry

end valid_arrangements_count_is_20_l1518_151876


namespace radius_of_scrap_cookie_l1518_151804

theorem radius_of_scrap_cookie
  (r_cookies : ℝ) (n_cookies : ℕ) (radius_layout : Prop)
  (circle_diameter_twice_width : Prop) :
  (r_cookies = 0.5 ∧ n_cookies = 9 ∧ radius_layout ∧ circle_diameter_twice_width)
  →
  (∃ r_scrap : ℝ, r_scrap = Real.sqrt 6.75) :=
by
  sorry

end radius_of_scrap_cookie_l1518_151804


namespace f_a1_a3_a5_positive_l1518_151830

theorem f_a1_a3_a5_positive (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf_odd : ∀ x, f (-x) = - f x)
  (hf_mono : ∀ x y, x < y → f x < f y)
  (ha_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha3_pos : 0 < a 3) :
  0 < f (a 1) + f (a 3) + f (a 5) :=
sorry

end f_a1_a3_a5_positive_l1518_151830


namespace correct_expression_l1518_151893

theorem correct_expression (a b : ℝ) : (a^2 * b)^3 = (a^6 * b^3) := 
by
sorry

end correct_expression_l1518_151893


namespace find_initial_strawberries_l1518_151825

-- Define the number of strawberries after picking 35 more to be 63
def strawberries_after_picking := 63

-- Define the number of strawberries picked
def strawberries_picked := 35

-- Define the initial number of strawberries
def initial_strawberries := 28

-- State the theorem
theorem find_initial_strawberries (x : ℕ) (h : x + strawberries_picked = strawberries_after_picking) : x = initial_strawberries :=
by
  -- Proof omitted
  sorry

end find_initial_strawberries_l1518_151825


namespace expression_value_l1518_151832

theorem expression_value (x y z : ℤ) (h1 : x = 25) (h2 : y = 30) (h3 : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 :=
by
  sorry

end expression_value_l1518_151832


namespace area_region_eq_6_25_l1518_151852

noncomputable def area_of_region : ℝ :=
  ∫ x in -0.5..4.5, (5 - |x - 2| - |x - 2|)

theorem area_region_eq_6_25 :
  area_of_region = 6.25 :=
sorry

end area_region_eq_6_25_l1518_151852


namespace greatest_possible_number_of_blue_chips_l1518_151824

-- Definitions based on conditions
def total_chips : Nat := 72

-- Definition of the relationship between red and blue chips where p is a prime number
def is_prime (n : Nat) : Prop := Nat.Prime n

def satisfies_conditions (r b p : Nat) : Prop :=
  r + b = total_chips ∧ r = b + p ∧ is_prime p

-- The statement to prove
theorem greatest_possible_number_of_blue_chips (r b p : Nat) 
  (h : satisfies_conditions r b p) : b = 35 := 
sorry

end greatest_possible_number_of_blue_chips_l1518_151824


namespace subset_implies_a_ge_2_l1518_151834

theorem subset_implies_a_ge_2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → x ≤ a) → a ≥ 2 :=
by sorry

end subset_implies_a_ge_2_l1518_151834


namespace find_length_l1518_151836

-- Let's define the conditions given in the problem
variables (b l : ℝ)

-- Length is more than breadth by 200%
def length_eq_breadth_plus_200_percent (b l : ℝ) : Prop := l = 3 * b

-- Total cost and rate per square meter
def cost_eq_area_times_rate (total_cost rate area : ℝ) : Prop := total_cost = rate * area

-- Given values
def total_cost : ℝ := 529
def rate_per_sq_meter : ℝ := 3

-- We need to prove that the length l is approximately 23 meters
theorem find_length (h1 : length_eq_breadth_plus_200_percent b l) 
    (h2 : cost_eq_area_times_rate total_cost rate_per_sq_meter (3 * b^2)) : 
    abs (l - 23) < 1 :=
by
  sorry -- Proof to be filled

end find_length_l1518_151836


namespace proof_problem_l1518_151815

theorem proof_problem (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 + 2 * a * b = 64 :=
sorry

end proof_problem_l1518_151815


namespace find_triangle_sides_l1518_151886

noncomputable def side_lengths (k c d : ℕ) : Prop :=
  let p1 := 26
  let p2 := 32
  let p3 := 30
  (2 * k = 6) ∧ (2 * k + 6 * c = p3) ∧ (2 * c + 2 * d = p1)

theorem find_triangle_sides (k c d : ℕ) (h1 : side_lengths k c d) : k = 3 ∧ c = 4 ∧ d = 5 := 
  sorry

end find_triangle_sides_l1518_151886


namespace square_of_binomial_l1518_151859

theorem square_of_binomial (a : ℝ) : 16 * x^2 + 32 * x + a = (4 * x + 4)^2 :=
by
  sorry

end square_of_binomial_l1518_151859


namespace intersection_eq_l1518_151890

namespace Proof

universe u

-- Define the natural number set M
def M : Set ℕ := { x | x > 0 ∧ x < 6 }

-- Define the set N based on the condition |x-1| ≤ 2
def N : Set ℝ := { x | abs (x - 1) ≤ 2 }

-- Define the complement of N with respect to the real numbers
def ComplementN : Set ℝ := { x | x < -1 ∨ x > 3 }

-- Define the intersection of M and the complement of N
def IntersectMCompN : Set ℕ := { x | x ∈ M ∧ (x : ℝ) ∈ ComplementN }

-- Provide the theorem to be proved
theorem intersection_eq : IntersectMCompN = { 4, 5 } :=
by
  sorry

end Proof

end intersection_eq_l1518_151890


namespace linear_regression_equation_l1518_151899

-- Given conditions
variables (x y : ℝ)
variable (corr_pos : x ≠ 0 → y / x > 0)
noncomputable def x_mean : ℝ := 2.4
noncomputable def y_mean : ℝ := 3.2

-- Regression line equation
theorem linear_regression_equation :
  (y = 0.5 * x + 2) ∧ (∀ x' y', (x' = x_mean ∧ y' = y_mean) → (y' = 0.5 * x' + 2)) :=
by
  sorry

end linear_regression_equation_l1518_151899


namespace base_of_right_angled_triangle_l1518_151888

theorem base_of_right_angled_triangle 
  (height : ℕ) (area : ℕ) (hypotenuse : ℕ) (b : ℕ) 
  (h_height : height = 8)
  (h_area : area = 24)
  (h_hypotenuse : hypotenuse = 10) 
  (h_area_eq : area = (1 / 2 : ℕ) * b * height)
  (h_pythagorean : hypotenuse^2 = height^2 + b^2) : 
  b = 6 := 
sorry

end base_of_right_angled_triangle_l1518_151888


namespace nonnegative_fraction_iff_interval_l1518_151889

theorem nonnegative_fraction_iff_interval (x : ℝ) : 
  0 ≤ x ∧ x < 3 ↔ 0 ≤ (x^2 - 12 * x^3 + 36 * x^4) / (9 - x^3) := by
  sorry

end nonnegative_fraction_iff_interval_l1518_151889


namespace train_speed_l1518_151833

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (conversion_factor : ℝ) :
  length_of_train = 200 → 
  time_to_cross = 24 → 
  conversion_factor = 3600 → 
  (length_of_train / 1000) / (time_to_cross / conversion_factor) = 30 := 
by
  sorry

end train_speed_l1518_151833


namespace triangular_pyramid_volume_l1518_151847

theorem triangular_pyramid_volume (a b c : ℝ)
  (h1 : 1/2 * a * b = 1.5)
  (h2 : 1/2 * b * c = 2)
  (h3 : 1/2 * a * c = 6) :
  (1/6 * a * b * c = 2) :=
by {
  -- Here, we would provide the proof steps, but for now we leave it as sorry
  sorry
}

end triangular_pyramid_volume_l1518_151847


namespace polynomial_real_root_l1518_151875

variable {A B C D E : ℝ}

theorem polynomial_real_root
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
by
  sorry

end polynomial_real_root_l1518_151875


namespace inverse_of_square_l1518_151801

theorem inverse_of_square (A : Matrix (Fin 2) (Fin 2) ℝ) (hA_inv : A⁻¹ = ![![3, 4], ![-2, -2]]) :
  (A^2)⁻¹ = ![![1, 4], ![-2, -4]] :=
by
  sorry

end inverse_of_square_l1518_151801


namespace mixed_number_calculation_l1518_151898

theorem mixed_number_calculation :
  47 * (2 + 2/3 - (3 + 1/4)) / (3 + 1/2 + (2 + 1/5)) = -4 - 25/38 :=
by
  sorry

end mixed_number_calculation_l1518_151898


namespace find_a_for_min_l1518_151870

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 - 6 * a * x + 2

theorem find_a_for_min {a x0 : ℝ} (hx0 : 1 < x0 ∧ x0 < 3) (h : ∀ x : ℝ, deriv (f a) x0 = 0) : a = -2 :=
by
  sorry

end find_a_for_min_l1518_151870


namespace cos_150_eq_neg_sqrt3_div_2_l1518_151851

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  unfold Real.cos
  sorry

end cos_150_eq_neg_sqrt3_div_2_l1518_151851


namespace price_reduction_equation_l1518_151861

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l1518_151861


namespace equivalence_of_min_perimeter_and_cyclic_quadrilateral_l1518_151816

-- Definitions for points P, Q, R, S on sides of quadrilateral ABCD
-- Function definitions for conditions and equivalence of stated problems

variable {A B C D P Q R S : Type*} 

def is_on_side (P : Type*) (A B : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B : Type*) : Prop := sorry
def is_convex_quadrilateral (A B C D : Type*) : Prop := sorry
def is_cyclic_quadrilateral (A B C D : Type*) : Prop := sorry
def has_circumcenter_interior (A B C D : Type*) : Prop := sorry
def has_minimal_perimeter (P Q R S : Type*) : Prop := sorry

theorem equivalence_of_min_perimeter_and_cyclic_quadrilateral 
  (h1 : is_convex_quadrilateral A B C D) 
  (hP : is_on_side P A B ∧ is_interior_point P A B) 
  (hQ : is_on_side Q B C ∧ is_interior_point Q B C) 
  (hR : is_on_side R C D ∧ is_interior_point R C D) 
  (hS : is_on_side S D A ∧ is_interior_point S D A) :
  (∃ P' Q' R' S', has_minimal_perimeter P' Q' R' S') ↔ (is_cyclic_quadrilateral A B C D ∧ has_circumcenter_interior A B C D) :=
sorry

end equivalence_of_min_perimeter_and_cyclic_quadrilateral_l1518_151816


namespace circle_units_diff_l1518_151843

-- Define the context where we verify the claim about the circle

noncomputable def radius : ℝ := 3
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Lean Theorem statement that needs to be proved
theorem circle_units_diff (r : ℝ) (h₀ : r = radius) :
  circumference r ≠ area r :=
by sorry

end circle_units_diff_l1518_151843


namespace tissue_magnification_l1518_151878

theorem tissue_magnification
  (diameter_magnified : ℝ)
  (diameter_actual : ℝ)
  (h1 : diameter_magnified = 5)
  (h2 : diameter_actual = 0.005) :
  diameter_magnified / diameter_actual = 1000 :=
by
  -- proof goes here
  sorry

end tissue_magnification_l1518_151878


namespace flowchart_correct_option_l1518_151897

-- Definitions based on conditions
def typical_flowchart (start_points end_points : ℕ) : Prop :=
  start_points = 1 ∧ end_points ≥ 1

-- Theorem to prove
theorem flowchart_correct_option :
  ∃ (start_points end_points : ℕ), typical_flowchart start_points end_points ∧ "Option C" = "Option C" :=
by {
  sorry -- This part skips the proof itself,
}

end flowchart_correct_option_l1518_151897


namespace cory_initial_money_l1518_151882

variable (cost_per_pack : ℝ) (packs : ℕ) (additional_needed : ℝ) (total_cost : ℝ) (initial_money : ℝ)

-- Conditions
def cost_per_pack_def : Prop := cost_per_pack = 49
def packs_def : Prop := packs = 2
def additional_needed_def : Prop := additional_needed = 78
def total_cost_def : Prop := total_cost = packs * cost_per_pack
def initial_money_def : Prop := initial_money = total_cost - additional_needed

-- Theorem
theorem cory_initial_money : cost_per_pack = 49 ∧ packs = 2 ∧ additional_needed = 78 → initial_money = 20 := by
  intro h
  have h1 : cost_per_pack = 49 := h.1
  have h2 : packs = 2 := h.2.1
  have h3 : additional_needed = 78 := h.2.2
  -- sorry
  sorry

end cory_initial_money_l1518_151882


namespace largest_constant_inequality_l1518_151874

theorem largest_constant_inequality (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) :
  (y*z + z*x + x*y)^2 * (x + y + z) ≥ 4 * x*y*z * (x^2 + y^2 + z^2) :=
sorry

end largest_constant_inequality_l1518_151874


namespace cos_sq_alpha_cos_sq_beta_range_l1518_151850

theorem cos_sq_alpha_cos_sq_beta_range
  (α β : ℝ)
  (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  (Real.cos α)^2 + (Real.cos β)^2 ∈ Set.Icc (14 / 9) 2 :=
sorry

end cos_sq_alpha_cos_sq_beta_range_l1518_151850


namespace trains_meet_at_noon_l1518_151891

noncomputable def meeting_time_of_trains : Prop :=
  let distance_between_stations := 200
  let speed_of_train_A := 20
  let starting_time_A := 7
  let speed_of_train_B := 25
  let starting_time_B := 8
  let initial_distance_covered_by_A := speed_of_train_A * (starting_time_B - starting_time_A)
  let remaining_distance := distance_between_stations - initial_distance_covered_by_A
  let relative_speed := speed_of_train_A + speed_of_train_B
  let time_to_meet_after_B_starts := remaining_distance / relative_speed
  let meeting_time := starting_time_B + time_to_meet_after_B_starts
  meeting_time = 12

theorem trains_meet_at_noon : meeting_time_of_trains :=
by
  sorry

end trains_meet_at_noon_l1518_151891


namespace prism_faces_even_or_odd_l1518_151818

theorem prism_faces_even_or_odd (n : ℕ) (hn : 3 ≤ n) : ¬ (2 + n) % 2 = 1 :=
by
  sorry

end prism_faces_even_or_odd_l1518_151818


namespace hcf_of_two_numbers_l1518_151868

-- Definitions based on conditions
def LCM (x y : ℕ) : ℕ := sorry  -- Assume some definition of LCM
def HCF (x y : ℕ) : ℕ := sorry  -- Assume some definition of HCF

-- Given conditions
axiom cond1 (x y : ℕ) : LCM x y = 600
axiom cond2 (x y : ℕ) : x * y = 18000

-- Statement to prove
theorem hcf_of_two_numbers (x y : ℕ) (h1 : LCM x y = 600) (h2 : x * y = 18000) : HCF x y = 30 :=
by {
  -- Proof omitted, hence we use sorry
  sorry
}

end hcf_of_two_numbers_l1518_151868


namespace B_joined_after_8_months_l1518_151864

-- Define the initial investments and time
def A_investment : ℕ := 36000
def B_investment : ℕ := 54000
def profit_ratio_A_B := 2 / 1

-- Define a proposition which states that B joined the business after x = 8 months
theorem B_joined_after_8_months (x : ℕ) (h : (A_investment * 12) / (B_investment * (12 - x)) = profit_ratio_A_B) : x = 8 :=
by
  sorry

end B_joined_after_8_months_l1518_151864


namespace isosceles_triangle_largest_angle_l1518_151814

theorem isosceles_triangle_largest_angle (α : ℝ) (β : ℝ)
  (h1 : 0 < α) (h2 : α = 30) (h3 : β = 30):
  ∃ γ : ℝ, γ = 180 - 2 * α ∧ γ = 120 := by
  sorry

end isosceles_triangle_largest_angle_l1518_151814


namespace general_formula_no_arithmetic_sequence_l1518_151854

-- Given condition
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n - 3 * n

-- Theorem 1: General formula for the sequence a_n
theorem general_formula (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) : 
  a n = 3 * 2^n - 3 :=
sorry

-- Theorem 2: No three terms of the sequence form an arithmetic sequence
theorem no_arithmetic_sequence (a : ℕ → ℤ) (x y z : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) (hx : x < y) (hy : y < z) :
  ¬ (a x + a z = 2 * a y) :=
sorry

end general_formula_no_arithmetic_sequence_l1518_151854


namespace sin_sum_triangle_l1518_151867

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l1518_151867


namespace balls_into_boxes_l1518_151800

/-- There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes. -/
theorem balls_into_boxes : (2 : ℕ) ^ 7 = 128 := by
  sorry

end balls_into_boxes_l1518_151800


namespace find_a5_div_a7_l1518_151849

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {aₙ} is a positive geometric sequence.
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom pos_seq (n : ℕ) : 0 < a n

-- Given conditions
axiom a2a8_eq_6 : a 2 * a 8 = 6
axiom a4_plus_a6_eq_5 : a 4 + a 6 = 5
axiom decreasing_seq (n : ℕ) : a (n + 1) < a n

theorem find_a5_div_a7 : a 5 / a 7 = 3 / 2 := 
sorry

end find_a5_div_a7_l1518_151849


namespace magnitude_quotient_l1518_151862

open Complex

theorem magnitude_quotient : 
  abs ((1 + 2 * I) / (2 - I)) = 1 := 
by 
  sorry

end magnitude_quotient_l1518_151862


namespace possible_values_of_b_l1518_151866

theorem possible_values_of_b (b : ℝ) : (¬ ∃ x : ℝ, x^2 + b * x + 1 ≤ 0) → -2 < b ∧ b < 2 :=
by
  intro h
  sorry

end possible_values_of_b_l1518_151866


namespace park_area_l1518_151853

variable (length width : ℝ)
variable (cost_per_meter total_cost : ℝ)
variable (ratio_length ratio_width : ℝ)
variable (x : ℝ)

def rectangular_park_ratio (length width : ℝ) (ratio_length ratio_width : ℝ) : Prop :=
  length / width = ratio_length / ratio_width

def fencing_cost (cost_per_meter total_cost : ℝ) (perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

theorem park_area (length width : ℝ) (cost_per_meter total_cost : ℝ)
  (ratio_length ratio_width : ℝ) (x : ℝ)
  (h1 : rectangular_park_ratio length width ratio_length ratio_width)
  (h2 : cost_per_meter = 0.70)
  (h3 : total_cost = 175)
  (h4 : ratio_length = 3)
  (h5 : ratio_width = 2)
  (h6 : length = 3 * x)
  (h7 : width = 2 * x)
  (h8 : fencing_cost cost_per_meter total_cost (2 * (length + width))) :
  length * width = 3750 := by
  sorry

end park_area_l1518_151853


namespace Lily_points_l1518_151894

variable (x y z : ℕ) -- points for inner ring (x), middle ring (y), and outer ring (z)

-- Tom's score
axiom Tom_score : 3 * x + y + 2 * z = 46

-- John's score
axiom John_score : x + 3 * y + 2 * z = 34

-- Lily's score
def Lily_score : ℕ := 40

theorem Lily_points : ∀ (x y z : ℕ), 3 * x + y + 2 * z = 46 → x + 3 * y + 2 * z = 34 → Lily_score = 40 := by
  intros x y z Tom_score John_score
  sorry

end Lily_points_l1518_151894


namespace basketball_team_count_l1518_151819

theorem basketball_team_count :
  (∃ n : ℕ, n = (Nat.choose 13 4) ∧ n = 715) :=
by
  sorry

end basketball_team_count_l1518_151819


namespace size_of_former_apartment_l1518_151841

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

end size_of_former_apartment_l1518_151841


namespace inequality_abc_l1518_151826

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l1518_151826


namespace max_area_triangle_bqc_l1518_151846

noncomputable def triangle_problem : ℝ :=
  let a := 112.5
  let b := 56.25
  let c := 3
  a + b + c

theorem max_area_triangle_bqc : triangle_problem = 171.75 :=
by
  -- The proof would involve validating the steps to ensure the computations
  -- for the maximum area of triangle BQC match the expression 112.5 - 56.25 √3,
  -- and thus confirm that a = 112.5, b = 56.25, c = 3
  -- and verifying that a + b + c = 171.75.
  sorry

end max_area_triangle_bqc_l1518_151846


namespace calculate_expression_l1518_151871

theorem calculate_expression (f : ℕ → ℝ) (h1 : ∀ a b, f (a + b) = f a * f b) (h2 : f 1 = 2) : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) = 6 := 
sorry

end calculate_expression_l1518_151871


namespace general_term_formula_l1518_151839

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)
variable (a1 d : ℤ)

-- Given conditions
axiom a2_eq : a 2 = 8
axiom S10_eq : S 10 = 185
axiom S_def : ∀ n, S n = n * (a 1 + a n) / 2
axiom a_def : ∀ n, a (n + 1) = a 1 + n * d

-- Prove the general term formula
theorem general_term_formula : a n = 3 * n + 2 := sorry

end general_term_formula_l1518_151839


namespace find_a_l1518_151820

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a^2 = 12) : a = 3 :=
by sorry

end find_a_l1518_151820


namespace find_x_l1518_151883

theorem find_x (x : ℕ) : (x > 20) ∧ (x < 120) ∧ (∃ y : ℕ, x = y^2) ∧ (x % 3 = 0) ↔ (x = 36) ∨ (x = 81) :=
by
  sorry

end find_x_l1518_151883


namespace pentomino_symmetry_count_l1518_151829

noncomputable def num_symmetric_pentominoes : Nat :=
  15 -- This represents the given set of 15 different pentominoes

noncomputable def symmetric_pentomino_count : Nat :=
  -- Here we are asserting that the count of pentominoes with at least one vertical symmetry is 8
  8

theorem pentomino_symmetry_count :
  symmetric_pentomino_count = 8 :=
sorry

end pentomino_symmetry_count_l1518_151829


namespace max_distance_travel_l1518_151802

-- Each car can carry at most 24 barrels of gasoline
def max_gasoline_barrels : ℕ := 24

-- Each barrel allows a car to travel 60 kilometers
def distance_per_barrel : ℕ := 60

-- The maximum distance one car can travel one way on a full tank
def max_one_way_distance := max_gasoline_barrels * distance_per_barrel

-- Total trip distance for the furthest traveling car
def total_trip_distance := 2160

-- Distance the other car turns back
def turn_back_distance := 360

-- Formalize in Lean
theorem max_distance_travel :
  (∃ x : ℕ, x = turn_back_distance ∧ max_gasoline_barrels * distance_per_barrel = 360) ∧
  (∃ y : ℕ, y = max_one_way_distance * 3 - turn_back_distance * 6 ∧ y = total_trip_distance) :=
by
  sorry

end max_distance_travel_l1518_151802


namespace negation_of_P_l1518_151811

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def P : Prop := ∀ n : ℕ, is_prime n → is_odd n

theorem negation_of_P : ¬ P ↔ ∃ n : ℕ, is_prime n ∧ ¬ is_odd n :=
by sorry

end negation_of_P_l1518_151811


namespace ellipse_condition_l1518_151892

theorem ellipse_condition (m n : ℝ) :
  (mn > 0) → (¬ (∃ x y : ℝ, (m = 1) ∧ (n = 1) ∧ (x^2)/m + (y^2)/n = 1 ∧ (x, y) ≠ (0,0))) :=
sorry

end ellipse_condition_l1518_151892


namespace swimmers_meetings_in_15_minutes_l1518_151806

noncomputable def swimmers_pass_each_other_count 
    (pool_length : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) (time_minutes : ℕ) : ℕ :=
sorry -- Definition of the function to count passing times

theorem swimmers_meetings_in_15_minutes :
  swimmers_pass_each_other_count 120 4 3 15 = 23 :=
sorry -- The proof is not required as per instruction.

end swimmers_meetings_in_15_minutes_l1518_151806


namespace line_passing_through_quadrants_l1518_151872

theorem line_passing_through_quadrants (a : ℝ) :
  (∀ x : ℝ, (3 * a - 1) * x - 1 ≠ 0) →
  (3 * a - 1 > 0) →
  a > 1 / 3 :=
by
  intro h1 h2
  -- proof to be filled
  sorry

end line_passing_through_quadrants_l1518_151872


namespace negation_of_diagonals_equal_l1518_151812

def Rectangle : Type := sorry -- Let's assume there exists a type Rectangle
def diagonals_equal (r : Rectangle) : Prop := sorry -- Assume a function that checks if diagonals are equal

theorem negation_of_diagonals_equal :
  ¬(∀ r : Rectangle, diagonals_equal r) ↔ ∃ r : Rectangle, ¬diagonals_equal r :=
by
  sorry

end negation_of_diagonals_equal_l1518_151812


namespace cost_per_pound_mixed_feed_correct_l1518_151822

noncomputable def total_weight_of_feed : ℝ := 17
noncomputable def cost_per_pound_cheaper_feed : ℝ := 0.11
noncomputable def cost_per_pound_expensive_feed : ℝ := 0.50
noncomputable def weight_cheaper_feed : ℝ := 12.2051282051

noncomputable def total_cost_of_feed : ℝ :=
  (cost_per_pound_cheaper_feed * weight_cheaper_feed) + 
  (cost_per_pound_expensive_feed * (total_weight_of_feed - weight_cheaper_feed))

noncomputable def cost_per_pound_mixed_feed : ℝ :=
  total_cost_of_feed / total_weight_of_feed

theorem cost_per_pound_mixed_feed_correct : 
  cost_per_pound_mixed_feed = 0.22 :=
  by
    sorry

end cost_per_pound_mixed_feed_correct_l1518_151822


namespace comparison_1_comparison_2_l1518_151848

noncomputable def expr1 := -(-((6: ℝ) / 7))
noncomputable def expr2 := -((abs (-((4: ℝ) / 5))))
noncomputable def expr3 := -((4: ℝ) / 5)
noncomputable def expr4 := -((2: ℝ) / 3)

theorem comparison_1 : expr1 > expr2 := sorry
theorem comparison_2 : expr3 < expr4 := sorry

end comparison_1_comparison_2_l1518_151848


namespace number_of_20_paise_coins_l1518_151817

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7000) : x = 220 :=
  sorry

end number_of_20_paise_coins_l1518_151817


namespace nat_power_digit_condition_l1518_151857

theorem nat_power_digit_condition (n k : ℕ) : 
  (10^(k-1) < n^n ∧ n^n < 10^k) → (10^(n-1) < k^k ∧ k^k < 10^n) → 
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end nat_power_digit_condition_l1518_151857


namespace angle_CAB_in_regular_hexagon_l1518_151855

-- Define a regular hexagon
structure regular_hexagon (A B C D E F : Type) :=
  (interior_angle : ℝ)
  (all_sides_equal : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F)
  (all_angles_equal : interior_angle = 120)

-- Define the problem of finding the angle CAB
theorem angle_CAB_in_regular_hexagon 
  (A B C D E F : Type)
  (hex : regular_hexagon A B C D E F)
  (diagonal_AC : A = C)
  : ∃ (CAB : ℝ), CAB = 30 :=
sorry

end angle_CAB_in_regular_hexagon_l1518_151855


namespace red_peaches_per_basket_l1518_151823

theorem red_peaches_per_basket (R : ℕ) (green_peaches_per_basket : ℕ) (number_of_baskets : ℕ) (total_peaches : ℕ) (h1 : green_peaches_per_basket = 4) (h2 : number_of_baskets = 15) (h3 : total_peaches = 345) : R = 19 :=
by
  sorry

end red_peaches_per_basket_l1518_151823


namespace second_wrongly_copied_number_l1518_151821

theorem second_wrongly_copied_number 
  (avg_err : ℝ) 
  (total_nums : ℕ) 
  (sum_err : ℝ) 
  (first_err_corr : ℝ) 
  (correct_avg : ℝ) 
  (correct_num : ℝ) 
  (second_num_wrong : ℝ) :
  (avg_err = 40.2) → 
  (total_nums = 10) → 
  (sum_err = total_nums * avg_err) → 
  (first_err_corr = 16) → 
  (correct_avg = 40) → 
  (correct_num = 31) → 
  sum_err - first_err_corr + (correct_num - second_num_wrong) = total_nums * correct_avg → 
  second_num_wrong = 17 := 
by 
  intros h_avg h_total h_sum_err h_first_corr h_correct_avg h_correct_num h_corrected_sum 
  sorry

end second_wrongly_copied_number_l1518_151821


namespace sum_simplest_form_probability_eq_7068_l1518_151896

/-- A jar has 15 red candies and 20 blue candies. Terry picks three candies at random,
    then Mary picks three of the remaining candies at random.
    Given that the probability that they get the same color combination (all reds or all blues, irrespective of order),
    find this probability in the simplest form. The sum of the numerator and denominator in simplest form is: 7068. -/
noncomputable def problem_statement : Nat :=
  let total_candies := 15 + 20;
  let terry_red_prob := (15 * 14 * 13) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_red_prob := (12 * 11 * 10) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_red := terry_red_prob * mary_red_prob;

  let terry_blue_prob := (20 * 19 * 18) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_blue_prob := (17 * 16 * 15) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_blue := terry_blue_prob * mary_blue_prob;

  let total_probability := both_red + both_blue;
  let simplest := 243 / 6825; -- This should be simplified form
  243 + 6825 -- Sum of numerator and denominator

theorem sum_simplest_form_probability_eq_7068 : problem_statement = 7068 :=
by sorry

end sum_simplest_form_probability_eq_7068_l1518_151896


namespace attendees_chose_water_l1518_151877

theorem attendees_chose_water
  (total_attendees : ℕ)
  (juice_percentage water_percentage : ℝ)
  (attendees_juice : ℕ)
  (h1 : juice_percentage = 0.7)
  (h2 : water_percentage = 0.3)
  (h3 : attendees_juice = 140)
  (h4 : total_attendees * juice_percentage = attendees_juice)
  : total_attendees * water_percentage = 60 := by
  sorry

end attendees_chose_water_l1518_151877


namespace distance_between_trees_l1518_151879

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_spaces : ℕ) (distance : ℕ)
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_spaces = num_trees - 1)
  (h4 : distance = yard_length / num_spaces) :
  distance = 18 :=
by
  sorry

end distance_between_trees_l1518_151879


namespace largest_possible_value_of_N_l1518_151865

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end largest_possible_value_of_N_l1518_151865


namespace comb_12_9_eq_220_l1518_151838

theorem comb_12_9_eq_220 : (Nat.choose 12 9) = 220 := by
  sorry

end comb_12_9_eq_220_l1518_151838


namespace ratio_of_divisors_l1518_151863

def M : Nat := 75 * 75 * 140 * 343

noncomputable def sumOfOddDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all odd divisors of n. (placeholder)
  sorry

noncomputable def sumOfEvenDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all even divisors of n. (placeholder)
  sorry

theorem ratio_of_divisors :
  let sumOdd := sumOfOddDivisors M
  let sumEven := sumOfEvenDivisors M
  sumOdd / sumEven = 1 / 6 := 
by
  sorry

end ratio_of_divisors_l1518_151863


namespace Jenny_minutes_of_sleep_l1518_151805

def hours_of_sleep : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem Jenny_minutes_of_sleep : hours_of_sleep * minutes_per_hour = 480 := by
  sorry

end Jenny_minutes_of_sleep_l1518_151805


namespace find_sum_of_xy_l1518_151844

theorem find_sum_of_xy (x y : ℝ) (hx_ne_y : x ≠ y) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0)
  (h_equation : x^4 - 2018 * x^3 - 2018 * y^2 * x = y^4 - 2018 * y^3 - 2018 * y * x^2) :
  x + y = 2018 :=
sorry

end find_sum_of_xy_l1518_151844


namespace zeros_of_geometric_sequence_quadratic_l1518_151827

theorem zeros_of_geometric_sequence_quadratic (a b c : ℝ) (h_geometric : b^2 = a * c) (h_pos : a * c > 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := by
sorry

end zeros_of_geometric_sequence_quadratic_l1518_151827


namespace birds_joined_l1518_151842

-- Definitions based on the identified conditions
def initial_birds : ℕ := 3
def initial_storks : ℕ := 2
def total_after_joining : ℕ := 10

-- Theorem statement that follows from the problem setup
theorem birds_joined :
  total_after_joining - (initial_birds + initial_storks) = 5 := by
  sorry

end birds_joined_l1518_151842


namespace calculation_is_correct_l1518_151895

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l1518_151895


namespace largest_pillar_radius_l1518_151808

-- Define the dimensions of the crate
def crate_length := 12
def crate_width := 8
def crate_height := 3

-- Define the condition that the pillar is a right circular cylinder
def is_right_circular_cylinder (r : ℝ) (h : ℝ) : Prop :=
  r > 0 ∧ h > 0

-- The theorem stating the radius of the largest volume pillar that can fit in the crate
theorem largest_pillar_radius (r h : ℝ) (cylinder_fits : is_right_circular_cylinder r h) :
  r = 1.5 := 
sorry

end largest_pillar_radius_l1518_151808


namespace max_value_of_gems_l1518_151840

/-- Conditions -/
structure Gem :=
  (weight : ℕ)
  (value : ℕ)

def Gem1 : Gem := ⟨3, 9⟩
def Gem2 : Gem := ⟨6, 20⟩
def Gem3 : Gem := ⟨2, 5⟩

-- Laura can carry maximum of 21 pounds.
def max_weight : ℕ := 21

-- She is able to carry at least 15 of each type
def min_count := 15

/-- Prove that the maximum value Laura can carry is $69 -/
theorem max_value_of_gems : ∃ (n1 n2 n3 : ℕ), (n1 >= min_count) ∧ (n2 >= min_count) ∧ (n3 >= min_count) ∧ 
  (Gem1.weight * n1 + Gem2.weight * n2 + Gem3.weight * n3 ≤ max_weight) ∧ 
  (Gem1.value * n1 + Gem2.value * n2 + Gem3.value * n3 = 69) :=
sorry

end max_value_of_gems_l1518_151840


namespace sarah_driving_distance_l1518_151828

def sarah_car_mileage (miles_per_gallon : ℕ) (tank_capacity : ℕ) (initial_drive : ℕ) (refuel : ℕ) (remaining_fraction : ℚ) : Prop :=
  ∃ (total_drive : ℚ),
    (initial_drive / miles_per_gallon + refuel - (tank_capacity * remaining_fraction / 1)) * miles_per_gallon = total_drive ∧
    total_drive = 467

theorem sarah_driving_distance :
  sarah_car_mileage 28 16 280 6 (1 / 3) :=
by
  sorry

end sarah_driving_distance_l1518_151828


namespace area_of_circumcircle_l1518_151884

-- Define the problem:
theorem area_of_circumcircle 
  (a b c : ℝ) 
  (A B C : Real) 
  (h_cosC : Real.cos C = (2 * Real.sqrt 2) / 3) 
  (h_bcosA_acoB : b * Real.cos A + a * Real.cos B = 2)
  (h_sides : c = 2):
  let sinC := Real.sqrt (1 - (2 * Real.sqrt 2 / 3)^2)
  let R := c / (2 * sinC)
  let area := Real.pi * R^2
  area = 9 * Real.pi / 5 :=
by 
  sorry

end area_of_circumcircle_l1518_151884


namespace cost_of_western_european_postcards_before_1980s_l1518_151813

def germany_cost_1950s : ℝ := 5 * 0.07
def france_cost_1950s : ℝ := 8 * 0.05

def germany_cost_1960s : ℝ := 6 * 0.07
def france_cost_1960s : ℝ := 9 * 0.05

def germany_cost_1970s : ℝ := 11 * 0.07
def france_cost_1970s : ℝ := 10 * 0.05

def total_germany_cost : ℝ := germany_cost_1950s + germany_cost_1960s + germany_cost_1970s
def total_france_cost : ℝ := france_cost_1950s + france_cost_1960s + france_cost_1970s

def total_western_europe_cost : ℝ := total_germany_cost + total_france_cost

theorem cost_of_western_european_postcards_before_1980s :
  total_western_europe_cost = 2.89 := by
  sorry

end cost_of_western_european_postcards_before_1980s_l1518_151813


namespace shelby_gold_stars_today_l1518_151880

-- Define the number of gold stars Shelby earned yesterday
def gold_stars_yesterday := 4

-- Define the total number of gold stars Shelby earned
def total_gold_stars := 7

-- Define the number of gold stars Shelby earned today
def gold_stars_today := total_gold_stars - gold_stars_yesterday

-- The theorem to prove
theorem shelby_gold_stars_today : gold_stars_today = 3 :=
by 
  -- The proof will go here.
  sorry

end shelby_gold_stars_today_l1518_151880


namespace g_1986_l1518_151831

def g : ℕ → ℤ := sorry

axiom g_def : ∀ n : ℕ, g n ≥ 0
axiom g_one : g 1 = 3
axiom g_func_eq : ∀ (a b : ℕ), g (a + b) = g a + g b - 3 * g (a * b)

theorem g_1986 : g 1986 = 0 :=
by
  sorry

end g_1986_l1518_151831


namespace transformation_correct_l1518_151881

noncomputable def original_function (x : ℝ) : ℝ := 2^x
noncomputable def transformed_function (x : ℝ) : ℝ := 2^x - 1
noncomputable def log_function (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = log_function (original_function x) :=
by
  intros x
  rw [transformed_function, log_function, original_function]
  sorry

end transformation_correct_l1518_151881


namespace merchant_profit_condition_l1518_151810

theorem merchant_profit_condition (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) :
  (P = 0.70 * L) →
  (S = 0.80 * M) →
  (S - P = 0.30 * S) →
  (M = 1.25 * L) := 
by
  intros h1 h2 h3
  sorry

end merchant_profit_condition_l1518_151810


namespace gcd_5670_9800_l1518_151885

-- Define the two given numbers
def a := 5670
def b := 9800

-- State that the GCD of a and b is 70
theorem gcd_5670_9800 : Int.gcd a b = 70 := by
  sorry

end gcd_5670_9800_l1518_151885


namespace find_integer_l1518_151887

theorem find_integer (n : ℤ) (h : 5 * (n - 2) = 85) : n = 19 :=
sorry

end find_integer_l1518_151887


namespace tommy_first_house_price_l1518_151803

theorem tommy_first_house_price (C : ℝ) (P : ℝ) (loan_rate : ℝ) (interest_rate : ℝ)
  (term : ℝ) (property_tax_rate : ℝ) (insurance_cost : ℝ) 
  (price_ratio : ℝ) (monthly_payment : ℝ) :
  C = 500000 ∧ price_ratio = 1.25 ∧ P * price_ratio = C ∧
  loan_rate = 0.75 ∧ interest_rate = 0.035 ∧ term = 15 ∧
  property_tax_rate = 0.015 ∧ insurance_cost = 7500 → 
  P = 400000 :=
by sorry

end tommy_first_house_price_l1518_151803


namespace jogger_ahead_distance_l1518_151860

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

end jogger_ahead_distance_l1518_151860


namespace find_PR_in_triangle_l1518_151873

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l1518_151873


namespace coordinates_of_P_l1518_151837

-- Define the point P with given coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P(3, 5)
def P : Point := ⟨3, 5⟩

-- Define a theorem stating that the coordinates of P are (3, 5)
theorem coordinates_of_P : P = ⟨3, 5⟩ :=
  sorry

end coordinates_of_P_l1518_151837


namespace maria_total_earnings_l1518_151856

noncomputable def total_earnings : ℕ := 
  let tulips_day1 := 30
  let roses_day1 := 20
  let lilies_day1 := 15
  let sunflowers_day1 := 10
  let tulips_day2 := tulips_day1 * 2
  let roses_day2 := roses_day1 * 2
  let lilies_day2 := lilies_day1
  let sunflowers_day2 := sunflowers_day1 * 3
  let tulips_day3 := tulips_day2 / 10
  let roses_day3 := 16
  let lilies_day3 := lilies_day1 / 2
  let sunflowers_day3 := sunflowers_day2
  let price_tulip := 2
  let price_rose := 3
  let price_lily := 4
  let price_sunflower := 5
  let day1_earnings := tulips_day1 * price_tulip + roses_day1 * price_rose + lilies_day1 * price_lily + sunflowers_day1 * price_sunflower
  let day2_earnings := tulips_day2 * price_tulip + roses_day2 * price_rose + lilies_day2 * price_lily + sunflowers_day2 * price_sunflower
  let day3_earnings := tulips_day3 * price_tulip + roses_day3 * price_rose + lilies_day3 * price_lily + sunflowers_day3 * price_sunflower
  day1_earnings + day2_earnings + day3_earnings

theorem maria_total_earnings : total_earnings = 920 := 
by 
  unfold total_earnings
  sorry

end maria_total_earnings_l1518_151856


namespace number_of_Ca_atoms_in_compound_l1518_151835

theorem number_of_Ca_atoms_in_compound
  (n : ℤ)
  (total_weight : ℝ)
  (ca_weight : ℝ)
  (i_weight : ℝ)
  (n_i_atoms : ℤ)
  (molecular_weight : ℝ) :
  n_i_atoms = 2 →
  molecular_weight = 294 →
  ca_weight = 40.08 →
  i_weight = 126.90 →
  n * ca_weight + n_i_atoms * i_weight = molecular_weight →
  n = 1 :=
by
  sorry

end number_of_Ca_atoms_in_compound_l1518_151835


namespace greatest_possible_value_of_a_l1518_151858

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ x : ℤ, x * (x + a) = -24 → x * (x + a) = -24) ∧ (∀ b : ℕ, (∀ x : ℤ, x * (x + b) = -24 → x * (x + b) = -24) → b ≤ a) ∧ a = 25 :=
sorry

end greatest_possible_value_of_a_l1518_151858


namespace value_of_a_l1518_151845

theorem value_of_a (a : ℝ) (x : ℝ) (h : 2 * x + 3 * a = -1) (hx : x = 1) : a = -1 :=
by
  sorry

end value_of_a_l1518_151845


namespace min_value_of_x_plus_2y_l1518_151869

noncomputable def min_value_condition (x y : ℝ) : Prop :=
x > -1 ∧ y > 0 ∧ (1 / (x + 1) + 2 / y = 1)

theorem min_value_of_x_plus_2y (x y : ℝ) (h : min_value_condition x y) : x + 2 * y ≥ 8 :=
sorry

end min_value_of_x_plus_2y_l1518_151869


namespace integral_result_l1518_151807

open Real

theorem integral_result :
  (∫ x in (0:ℝ)..(π/2), (x^2 - 5 * x + 6) * sin (3 * x)) = (67 - 3 * π) / 27 := by
  sorry

end integral_result_l1518_151807
