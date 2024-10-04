import Mathlib

namespace intersection_area_l297_297115

variable (E F XY YE EX FX : ℝ)
variable (congruent : ∀ (ΔXYE ΔFYX : Triangle), ΔXYE ≅ ΔFYX → True)

def problem_conditions :=
  E ≠ F ∧ XYE ≅ FYX ∧
  XY = 12 ∧ YE = 13 ∧ EX = 20 ∧ 
  FX = 13 ∧ FY = 20

theorem intersection_area (h : problem_conditions E F XY YE EX FX) :
  ∃ (p q : ℕ), p + q = 167 := by
  sorry

end intersection_area_l297_297115


namespace volume_triangular_pyramid_correctness_l297_297267

noncomputable def volume_of_regular_triangular_pyramid 
  (a α l : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α

theorem volume_triangular_pyramid_correctness (a α l : ℝ) : volume_of_regular_triangular_pyramid a α l =
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α := 
sorry

end volume_triangular_pyramid_correctness_l297_297267


namespace part1_part2_l297_297730

-- Definition of the function
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 5)

-- First problem in Lean 4 statement:
theorem part1 (a : ℝ) : (∀ x : ℝ, f x a ≥ 3) ↔ (a ≤ 2 ∨ 8 ≤ a) :=
sorry

-- Second problem in Lean 4 statement:
theorem part2 (x : ℝ) : (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + real.sqrt 3) :=
sorry

end part1_part2_l297_297730


namespace pentagon_fifth_angle_l297_297178

theorem pentagon_fifth_angle (a b c d : ℝ) (h1 : a = 110) (h2 : b = 110) (h3 : c = 110) (h4 : d = 110) : 
  ∃ e : ℝ, e = 100 :=
by
  -- The sum of the interior angles of a pentagon is 540 degrees
  have h_sum : a + b + c + d + 100 = 540,
    calc
      a + b + c + d + 100 = 110 + 110 + 110 + 110 + 100 : by rw [h1, h2, h3, h4]
                       ... = 540 : by norm_num,
  use 100,
  sorry

end pentagon_fifth_angle_l297_297178


namespace variance_of_planted_trees_l297_297047

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l297_297047


namespace bananas_in_basket_E_l297_297082

variable (AverageFruitsPerBasket : ℕ)
variable (TotalBaskets : ℕ)
variable (FruitsInBasketA : ℕ)
variable (FruitsInBasketB : ℕ)
variable (FruitsInBasketC : ℕ)
variable (FruitsInBasketD : ℕ)

theorem bananas_in_basket_E
  (h_avg : AverageFruitsPerBasket = 25)
  (h_baskets : TotalBaskets = 5)
  (h_fruits_A : FruitsInBasketA = 15)
  (h_fruits_B : FruitsInBasketB = 30)
  (h_fruits_C : FruitsInBasketC = 20)
  (h_fruits_D : FruitsInBasketD = 25) :
  let TotalFruits := AverageFruitsPerBasket * TotalBaskets,
      FruitsInBasketsABCD := FruitsInBasketA + FruitsInBasketB + FruitsInBasketC + FruitsInBasketD,
      FruitsInBasketE := TotalFruits - FruitsInBasketsABCD
  in FruitsInBasketE = 35 :=
by
  sorry

end bananas_in_basket_E_l297_297082


namespace sum_diff_le_abs_l297_297809

theorem sum_diff_le_abs (n : ℕ) (h_n : 0 < n) (a : Fin n → ℝ) :
  ∃ (m k : ℕ), m ≤ n ∧ k < n ∧ 
  |(∑ i in Finset.range m, a ⟨i, Nat.lt_of_lt_pred m_le_n⟩) -
   (∑ i in Finset.Ico m n, a ⟨i, Nat.lt_of_le_of_lt (Finset.Ico_subset_left m n)⟩)| 
  ≤ |a ⟨k, h_k⟩| := 
by
  sorry

end sum_diff_le_abs_l297_297809


namespace max_2x2_squares_in_2019x2019_grid_l297_297060

-- Define the maximum number of selected squares
def max_selected_squares : ℕ := 509545

theorem max_2x2_squares_in_2019x2019_grid :
  ∀ (grid : ℕ × ℕ), (grid = (2019, 2019)) → ∃ (max : ℕ), 
  max = max_selected_squares ∧ 
  (∀ (selected_squares : ℕ), (forall (square1 square2 : ℕ × ℕ), 
    -- Condition that no two squares share a common edge
    edge_sharing square1 square2 -> false) -> selected_squares ≤ max) :=
by
  sorry

-- Define edge_sharing condition between two 2x2 squares to be used in the theorem
def edge_sharing (square1 square2 : ℕ × ℕ) : Prop :=
  let (x1, y1) := square1
  let (x2, y2) := square2
  (x1 = x2 ∧ abs (y1 - y2) = 2) ∨ (y1 = y2 ∧ abs (x1 - x2) = 2) 

end max_2x2_squares_in_2019x2019_grid_l297_297060


namespace possible_values_of_k_l297_297741

open Set

theorem possible_values_of_k (A : Set ℝ) (k : ℝ) : 
  A = {-1, 1} → 
  (B : Set ℝ) → (B = {x : ℝ | k * x = 1}) → B ⊆ A → (k = -1 ∨ k = 0 ∨ k = 1) :=
by
  intros hA hB hSub
  rw [hA, hB] at hSub
  sorry

end possible_values_of_k_l297_297741


namespace find_abc_sol_l297_297686

theorem find_abc_sol (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / ↑a + 1 / ↑b + 1 / ↑c = 1) →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end find_abc_sol_l297_297686


namespace hire_applicant_A_l297_297983

-- Define the test scores for applicants A and B
def education_A := 7
def experience_A := 8
def attitude_A := 9

def education_B := 10
def experience_B := 7
def attitude_B := 8

-- Define the weights for the test items
def weight_education := 1 / 6
def weight_experience := 2 / 6
def weight_attitude := 3 / 6

-- Define the final scores
def final_score_A := education_A * weight_education + experience_A * weight_experience + attitude_A * weight_attitude
def final_score_B := education_B * weight_education + experience_B * weight_experience + attitude_B * weight_attitude

-- Prove that Applicant A is hired because their final score is higher
theorem hire_applicant_A : final_score_A > final_score_B :=
by sorry

end hire_applicant_A_l297_297983


namespace empty_set_l297_297206

def setA := {x : ℝ | x^2 - 4 = 0}
def setB := {x : ℝ | x > 9 ∨ x < 3}
def setC := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}
def setD := {x : ℝ | x > 9 ∧ x < 3}

theorem empty_set : setD = ∅ := 
  sorry

end empty_set_l297_297206


namespace fraction_product_simplification_l297_297213

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l297_297213


namespace sphere_radius_proof_l297_297592

-- Define the given conditions
def pole_height : ℝ := 1.5
def pole_shadow_length : ℝ := 3
def sphere_shadow_length : ℝ := 15
def tan_theta : ℝ := pole_height / pole_shadow_length

-- Define the target radius to prove
def target_radius : ℝ := 7.5

-- The main statement to prove
theorem sphere_radius_proof (h_tan_theta : tan_theta = 0.5) (h_sphere_shadow : sphere_shadow_length = 15) : 
  target_radius = sphere_shadow_length * tan_theta :=
by 
  -- to be filled in with a proof, currently omitted
  sorry

end sphere_radius_proof_l297_297592


namespace no_valid_grid_filling_l297_297013

-- Define a type alias for coordinates in the grid
def Coord := (ℤ × ℤ)

-- Define a type alias for the values to fill in the grid
inductive GridCellValue
| one : GridCellValue
| two : GridCellValue
| three : GridCellValue

-- Define function type for grid filling
def GridFilling := Coord → GridCellValue

-- Define the conditions as predicates
def condition1 (grid : GridFilling) : Prop :=
  ∀ (x y : ℤ), (grid (x, y) ≠ grid (x + 1, y)) ∧ (grid (x, y) ≠ grid (x, y + 1))

def condition2 (grid : GridFilling) : Prop :=
  ∀ (x y : ℤ), ¬ ((grid (x, y) = GridCellValue.one ∧ grid (x + 1, y) = GridCellValue.two ∧ grid (x + 2, y) = GridCellValue.three)
                 ∨ (grid (x, y) = GridCellValue.one ∧ grid (x, y + 1) = GridCellValue.two ∧ grid (x, y + 2) = GridCellValue.three))

def condition3 (grid : GridFilling) (n : ℕ) (hn : n % 2 = 1) : Prop :=
  ∀ (i j : ℤ), let subGridSum1 := ∑ x in finRange (fin (nat_abs ↑n)), ∑ y in finRange (fin (nat_abs ↑n)), (grid (i+x.1, j+y.1))
                let subGridSum2 := ∑ x in finRange (fin (nat_abs ↑n)), ∑ y in finRange (fin (nat_abs ↑n)), (grid (i-1+x.1, j-1+y.1))
                subGridSum1 = subGridSum2

-- Define the main theorem statement
theorem no_valid_grid_filling (n : ℕ) (hn : n % 2 = 1) :
  ¬ ∃ (grid : GridFilling), condition1 grid ∧ condition2 grid ∧ condition3 grid n hn :=
sorry

end no_valid_grid_filling_l297_297013


namespace problem_condition_1_problem_condition_2_problem_monotonicity_g_min_a_for_decreasing_f_l297_297732

noncomputable def g (x : ℝ) : ℝ := x / Real.log x
noncomputable def f (x a : ℝ) : ℝ := g x - a * x

theorem problem_condition_1 (x : ℝ) (h : x > 0 ∧ x ≠ 1) : x / Real.log x = g x := 
begin
  rw g,
  refl,
end

theorem problem_condition_2 (x a : ℝ) (h : x > 1) : g x - a * x = f x a := 
begin
  rw f,
  refl,
end

theorem problem_monotonicity_g : 
  (∀ x, e < x → 0 < (Real.log x - 1) / (Real.log x)^2) ∧ 
  (∀ x, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < e) → (Real.log x - 1) / (Real.log x)^2 < 0) :=
sorry

theorem min_a_for_decreasing_f : 
  ∃ a, (∀ x, 1 < x → (Real.log x - 1 - a * (Real.log x)^2) / (Real.log x)^2 ≤ 0) ∧ a = 1 / 4 :=
sorry

end problem_condition_1_problem_condition_2_problem_monotonicity_g_min_a_for_decreasing_f_l297_297732


namespace polar_coordinates_of_point_l297_297229

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x ^ 2 + y ^ 2)
  let θ := if x = 0 then (if y > 0 then Real.pi / 2 else 3 * Real.pi / 2) else (Real.atan2 y x)
  (r, θ)

theorem polar_coordinates_of_point :
  rectangular_to_polar (-Real.sqrt 3) (Real.sqrt 3) = (Real.sqrt 6, 2 * Real.pi / 3) :=
by
  sorry

end polar_coordinates_of_point_l297_297229


namespace official_exchange_rate_l297_297143

theorem official_exchange_rate (E : ℝ)
  (h1 : 70 = 10 * (7 / 5) * E) :
  E = 5 :=
by
  sorry

end official_exchange_rate_l297_297143


namespace count_numbers_of_form_divisible_by_5_l297_297127

theorem count_numbers_of_form_divisible_by_5 :
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  ∃ count : ℕ, count = (card a_vals) * (card b_vals) * (card c_vals) ∧ count = 180 :=
by
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  existsi ((card a_vals) * (card b_vals) * (card c_vals))
  split
  sorry
  exact 180

end count_numbers_of_form_divisible_by_5_l297_297127


namespace find_k_range_l297_297734

open Nat

def a_n (n : ℕ) : ℕ := 2^ (5 - n)

def b_n (n : ℕ) (k : ℤ) : ℤ := n + k

def c_n (n : ℕ) (k : ℤ) : ℤ :=
if (a_n n : ℤ) ≤ (b_n n k) then b_n n k else a_n n

theorem find_k_range : 
  (∀ n ∈ { m : ℕ | m > 0 }, c_n 5 = a_n 5 ∧ c_n 5 ≤ c_n n) → 
  (∃ k : ℤ, -5 ≤ k ∧ k ≤ -3) :=
by
  sorry

end find_k_range_l297_297734


namespace Barbara_Mike_ratio_is_one_half_l297_297041

-- Define the conditions
def Mike_age_current : ℕ := 16
def Mike_age_future : ℕ := 24
def Barbara_age_future : ℕ := 16

-- Define Barbara's current age based on the conditions
def Barbara_age_current : ℕ := Mike_age_current - (Mike_age_future - Barbara_age_future)

-- Define the ratio of Barbara's age to Mike's age
def ratio_Barbara_Mike : ℚ := Barbara_age_current / Mike_age_current

-- Prove that the ratio is 1:2
theorem Barbara_Mike_ratio_is_one_half : ratio_Barbara_Mike = 1 / 2 := by
  sorry

end Barbara_Mike_ratio_is_one_half_l297_297041


namespace ellipse_equation_find_x0_l297_297296

theorem ellipse_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (hc : c = 2 * real.sqrt 2) (h_sum : ∀ M : ℝ × ℝ, M.1^2 / a^2 + M.2^2 / b^2 = 1 → 
  ∑ i in ({(2*real.sqrt 2, 0), (-2*real.sqrt 2, 0)} : finset (ℝ × ℝ)), (real.sqrt ((M.1 - i.1)^2 + (M.2 - i.2)^2)) = 4 * real.sqrt 3) :
  a = 2 * real.sqrt 3 → b^2 = 4 → ∀ x y, x^2 / 12 + y^2 / 4 = 1 := 
sorry

theorem find_x0 (m x_0 : ℝ) (h_intersects : ∀ A B : ℝ × ℝ, A ≠ B → 
  (A.2 = A.1 + m) ∧ (B.2 = B.1 + m) → 4 * A.1^2 + 6 * m * A.1 + 3 * m^2 - 12 = 0 → 
  4 * B.1^2 + 6 * m * B.1 + 3 * m^2 - 12 = 0 → 
  ∥A - B∥ = 3 * real.sqrt 2) :
  ∃ x_0, x_0 = -3 ∨ x_0 = -1 :=
sorry

end ellipse_equation_find_x0_l297_297296


namespace gcd_with_prime_exists_l297_297577

theorem gcd_with_prime_exists (S : Set ℕ) (h_finite : S.Finite) (h_composite : ∀ s ∈ S, ∃ p q : ℕ, s = p * q ∧ 1 < p ∧ 1 < q)
  (h_gcd_condition : ∀ n : ℕ, ∃ s ∈ S, Nat.gcd s n = 1 ∨ Nat.gcd s n = s) :
  ∃ s t ∈ S, Nat.Prime (Nat.gcd s t) :=
sorry

end gcd_with_prime_exists_l297_297577


namespace ratio_OQ_OP_l297_297444

-- Definitions based on the conditions from the problem
variable (A B C D M O X Y P Q : Point)
variable (t : ℝ) (rhombus : Rhombus A B C D) (angle_A_lt_90 : ∠A < 90)
variable (M_def : M = midpoint A C) (intersect_diagonals : intersection_point A C B D = M)
variable (O_on_MC : on_segment O M C) (OB_LT_OC : distance O B < distance O C) (O_ne_M : O ≠ M)
variable (circle_O : Circle O) (passes_through_B_D : passes_through circle_O B ∧ passes_through circle_O D)
variable (intersection_AB : Set.Points_on_circle circle_O = {B, X})
variable (intersection_BC : Set.Points_on_circle circle_O = {B, Y})
variable (P_on_AC_DX : intersection_point A C (line_through D X) = P)
variable (Q_on_AC_DY : intersection_point A C (line_through D Y) = Q)

-- Goal
theorem ratio_OQ_OP (h : t = (distance M A) / (distance M O)) : (distance O Q) / (distance O P) = (t + 1) / (t - 1) :=
by
  sorry

end ratio_OQ_OP_l297_297444


namespace collinear_perpendicular_to_ON1_l297_297293

theorem collinear_perpendicular_to_ON1
  {A B C O N N1 A1 B1 C1 : Point}
  (h_acute : is_acute_triangle A B C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_nine_point_center : is_nine_point_center N A B C)
  (h_ngle1 : ∠N A B = ∠N1 A C)
  (h_ngle2 : ∠N B C = ∠N1 B A)
  (A1_def : is_intersection (perpendicular_bisector O A) (line_through B C) A1)
  (B1_def : is_intersection (perpendicular_bisector O B) (line_through C A) B1)
  (C1_def : is_intersection (perpendicular_bisector O C) (line_through A B) C1)
  : collinear A1 B1 C1 ∧ perpendicular (line_through A1 B1 C1) (line_through O N1) :=
sorry

end collinear_perpendicular_to_ON1_l297_297293


namespace trees_variance_l297_297049

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end trees_variance_l297_297049


namespace min_val_l297_297757

theorem min_val (x y : ℝ) (h : x + 2 * y = 1) : 2^x + 4^y = 2 * Real.sqrt 2 :=
sorry

end min_val_l297_297757


namespace overall_loss_percentage_l297_297896

theorem overall_loss_percentage
  (cost_price : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (depreciation : ℝ)
  (final_selling_price : ℝ) :
  cost_price = 1900 →
  discount = 0.15 →
  sales_tax = 0.12 →
  depreciation = 0.05 →
  final_selling_price = 1330 →
  ((cost_price - (discount * cost_price)) * (1 + sales_tax) * (1 - depreciation) - final_selling_price) / cost_price * 100 = 20.44 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overall_loss_percentage_l297_297896


namespace cos_and_sin_double_angle_value_of_3alpha_plus_beta_l297_297692

variables {α β : ℝ}

theorem cos_and_sin_double_angle (hα : α ∈ set.Ioo 0 real.pi)
                                 (hβ : β ∈ set.Ioo 0 real.pi)
                                 (hcosα : real.cos α = real.sqrt 5 / 5)
                                (hsinαβ : real.sin (α + β) = -real.sqrt 2 / 10) :
  real.cos (2 * α) = -3 / 5 ∧ real.sin (2 * α) = 4 / 5 :=
begin
  sorry
end

theorem value_of_3alpha_plus_beta (hα : α ∈ set.Ioo 0 real.pi)
                                  (hβ : β ∈ set.Ioo 0 real.pi)
                                  (hcosα : real.cos α = real.sqrt 5 / 5)
                                  (hsinαβ : real.sin (α + β) = -real.sqrt 2 / 10) :
  3 * α + β = 7 * real.pi / 4 :=
begin
  sorry
end

end cos_and_sin_double_angle_value_of_3alpha_plus_beta_l297_297692


namespace chocolate_chip_cookie_recipe_l297_297977

theorem chocolate_chip_cookie_recipe (n : ℕ) 
  (h1 : 2 * n = 46) : 
  n = 23 :=
begin
  sorry
end

end chocolate_chip_cookie_recipe_l297_297977


namespace prob_even_product_eq_19_over_20_l297_297538

-- Define the set of integers from 1 to 6
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Lean function to calculate the probability
noncomputable def even_product_probability : ℕ :=
  let total_ways := (S.card.choose 3) in
  let odd_elements := {1, 3, 5} : Finset ℕ in
  let odd_ways := (odd_elements.card.choose 3) in
  let odd_probability := odd_ways / total_ways in
  1 - odd_probability

-- The theorem to prove
theorem prob_even_product_eq_19_over_20 : even_product_probability = 19/20 :=
  sorry

end prob_even_product_eq_19_over_20_l297_297538


namespace probability_prime_rolled_l297_297885

open Finset

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def prime_outcomes : Finset ℕ := outcomes.filter is_prime

theorem probability_prime_rolled : (prime_outcomes.card : ℚ) / outcomes.card = 1 / 2 :=
by
  -- Proof would go here
  sorry

end probability_prime_rolled_l297_297885


namespace arrange_abc_l297_297312

theorem arrange_abc : 
  let a := Real.log 5 / Real.log 0.6
  let b := 2 ^ (4 / 5)
  let c := Real.sin 1
  a < c ∧ c < b := 
by
  sorry

end arrange_abc_l297_297312


namespace intersection_correct_l297_297305

def A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem intersection_correct : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_correct_l297_297305


namespace evaporation_period_l297_297162

theorem evaporation_period
  (initial_amount : ℚ)
  (evaporation_rate : ℚ)
  (percentage_evaporated : ℚ)
  (actual_days : ℚ)
  (h_initial : initial_amount = 10)
  (h_evap_rate : evaporation_rate = 0.007)
  (h_percentage : percentage_evaporated = 3.5000000000000004)
  (h_days : actual_days = (percentage_evaporated / 100) * initial_amount / evaporation_rate):
  actual_days = 50 := by
  sorry

end evaporation_period_l297_297162


namespace angle_CAD_eq_15_l297_297780

-- Define the angles and equality of sides
variables (A B C D : Type) [IsConvexQuadrilateral A B C D]
           (angle_A : ∠A = 65)
           (angle_B : ∠B = 80)
           (angle_C : ∠C = 75)
           (AB_eq_BD : AB = BD)
           
-- State the theorem to prove the measure of ∠CAD
theorem angle_CAD_eq_15 : ∠CAD = 15 :=
sorry

end angle_CAD_eq_15_l297_297780


namespace hyperbola_eccentricity_l297_297697

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (asymptote : ∀ x, asymptote := (y = √2 * x)) :
  e = √3 := by
  sorry

end hyperbola_eccentricity_l297_297697


namespace min_x_plus_y_positive_reals_l297_297821

theorem min_x_plus_y_positive_reals (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 6 * y - x * y = 0) : 
  x + y = 8 + 4 * sqrt 3 := 
sorry

end min_x_plus_y_positive_reals_l297_297821


namespace probability_of_multiple_of_3_l297_297964

theorem probability_of_multiple_of_3 (n : ℕ) (h1 : n = 24) : 
  let total_tickets := 24 in
  let multiples_of_3 := [3, 6, 9, 12, 15, 18, 21, 24] in
  let favorable_outcomes := multiples_of_3.length in
  let probability := (favorable_outcomes : ℚ) / total_tickets in
  probability = 1 / 3 :=
by
  -- Proof omitted
  sorry

end probability_of_multiple_of_3_l297_297964


namespace number_of_odd_factors_of_252_l297_297360

def numOddFactors (n : ℕ) : ℕ :=
  if ∀ d : ℕ, n % d = 0 → ¬(d % 2 = 0) then d
  else 0

theorem number_of_odd_factors_of_252 : numOddFactors 252 = 6 := by
  -- Definition of n
  let n := 252
  -- Factor n into 2^2 * 63
  have h1 : n = 2^2 * 63 := rfl
  -- Find the number of odd factors of 63 since factors of 252 that are odd are the same as factors of 63
  have h2 : 63 = 3^2 * 7 := rfl
  -- Check the number of factors of 63
  sorry

end number_of_odd_factors_of_252_l297_297360


namespace part1_part2_l297_297789

-- Define the problem conditions and theorem statements

-- Points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 0⟩
def B : Point := ⟨0, 2⟩
def C (α : ℝ) : Point := ⟨Real.cos α, Real.sin α⟩

-- Vectors from A to C and B to C
def AC (α : ℝ) : Point := ⟨Real.cos α - 2, Real.sin α⟩
def BC (α : ℝ) : Point := ⟨Real.cos α, Real.sin α - 2⟩

-- Vector length
def vec_length (p : Point) : ℝ := Real.sqrt (p.x^2 + p.y^2)

-- Dot product
def dot_product (u v : Point) : ℝ := u.x * v.x + u.y * v.y

-- Part (1)
theorem part1 {α : ℝ} (hα : α ∈ Set.Ioo 0 Real.pi) (h1 : vec_length (AC α) = vec_length (BC α)) : α = Real.pi / 4 := 
sorry

-- Part (2)
theorem part2 {α : ℝ} (hα : α ∈ Set.Ioo 0 Real.pi) (h2 : dot_product (AC α) (BC α) = 1 / 3) : 
  (2 * Real.sin(α)^2 + Real.sin (2 * α)) / (1 + Real.tan(α)) = -8 / 9 := 
sorry

end part1_part2_l297_297789


namespace fraction_product_simplification_l297_297212

theorem fraction_product_simplification : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_simplification_l297_297212


namespace count_ordered_pairs_l297_297235

theorem count_ordered_pairs : ∃ (n : ℕ), n = 4 ∧ ∀ (M N : ℕ), (M * N = 35) → (
  ((M = 1 ∧ N = 35) ∨ (M = 35 ∧ N = 1)) ∨ 
  ((M = 5 ∧ N = 7) ∨ (M = 7 ∧ N = 5))
) :=
by 
  exists 4
  split
  { refl }
  { intros M N h
    cases (M, N) with
    | (1, 35) => tauto
    | (35, 1) => tauto
    | (5, 7) => tauto
    | (7, 5) => tauto
    sorry
  }

end count_ordered_pairs_l297_297235


namespace remaining_pens_l297_297850

theorem remaining_pens (initial_pens sold_pens : ℕ) (h1 : initial_pens = 42) (h2 : sold_pens = 23) : initial_pens - sold_pens = 19 := 
by
  rw [h1, h2]
  exact rfl

end remaining_pens_l297_297850


namespace min_moves_to_reset_counters_l297_297917

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l297_297917


namespace ariel_birth_year_is_1992_l297_297620

-- Define the conditions as constants and hypotheses
constant current_year : ℕ := 2022 -- Just setting a context, not directly related to the problem
constant fencing_start_year : ℕ := 2006
constant years_fencing : ℕ := 16
constant current_age : ℕ := 30

-- Define hypothesis for the problem
def ariel_birth_year (birth_year year_started fencing_years current_years : ℕ) : Prop :=
  year_started - (current_years - fencing_years) = birth_year

-- Theorem statement for the problem
theorem ariel_birth_year_is_1992 : ∃ birth_year, ariel_birth_year birth_year fencing_start_year years_fencing current_age ∧ birth_year = 1992 :=
by
  existsi (fencing_start_year - (current_age - years_fencing))
  simp [ariel_birth_year]
  sorry

end ariel_birth_year_is_1992_l297_297620


namespace no_extreme_values_l297_297234

/-
  Given the function f(x) = (x^2 - 3x + 3) / (2x - 4),
  we need to determine whether there is a minimum or maximum value for x
  in the interval -2 < x < 3.
-/

def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) / (2*x - 4)

theorem no_extreme_values (x : ℝ) (h : -2 < x ∧ x < 3) :
  ¬ (∃ a : ℝ, ∀ y : ℝ, (h.1 < y ∧ y < h.2) → f(y) ≤ f(a)) ∧
  ¬ (∃ b : ℝ, ∀ z : ℝ, (h.1 < z ∧ z < h.2) → f(z) ≥ f(b)) :=
sorry

end no_extreme_values_l297_297234


namespace water_level_drop_correct_l297_297972

noncomputable def base_area (a : ℝ) := (3 * real.sqrt 3 * a^2) / 2
noncomputable def submerged_volume (r h : ℝ) := (real.pi * (r - h)^2 / 3) * (3 * r - (r - h))
noncomputable def water_level_drop (K A : ℝ) := K / A

theorem water_level_drop_correct :
  let a := 5 in
  let r := 4 in
  let h := 1 in
  let A_hex := base_area a in
  let K_ball := submerged_volume r h in
  let x := water_level_drop K_ball A_hex in
  x = 3.95 := 
by { sorry }

end water_level_drop_correct_l297_297972


namespace laura_total_miles_per_week_l297_297807

def round_trip_school : ℕ := 20
def round_trip_supermarket : ℕ := 40
def round_trip_gym : ℕ := 10
def round_trip_friends_house : ℕ := 24

def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2
def gym_trips_per_week : ℕ := 3
def friends_house_trips_per_week : ℕ := 1

def total_miles_driven_per_week :=
  round_trip_school * school_trips_per_week +
  round_trip_supermarket * supermarket_trips_per_week +
  round_trip_gym * gym_trips_per_week +
  round_trip_friends_house * friends_house_trips_per_week

theorem laura_total_miles_per_week : total_miles_driven_per_week = 234 :=
by
  sorry

end laura_total_miles_per_week_l297_297807


namespace part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l297_297991

-- Definitions for times needed by copiers A and B
def time_A : ℕ := 90
def time_B : ℕ := 60

-- (1) Combined time for both copiers
theorem part1_combined_time : 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 36 = 1 := 
by sorry

-- (2) Time left for copier A alone
theorem part2_copier_A_insufficient (mins_combined : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → time_left = 13 → 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + time_left / (time_A : ℝ) ≠ 1 := 
by sorry

-- (3) Combined time with B after repair is sufficient
theorem part3_combined_after_repair (mins_combined : ℕ) (mins_repair_B : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → mins_repair_B = 9 → time_left = 13 →
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + 9 / (time_A : ℝ) + 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 2.4 = 1 := 
by sorry

end part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l297_297991


namespace g_1000_value_l297_297817

noncomputable def g : ℝ → ℝ := sorry

theorem g_1000_value :
  (∀ (x y : ℝ), x > 0 → y > 0 → g(x * y) = g(x) / y) →
  g(800) = 2 →
  g(1000) = 8 / 5 :=
by
  intros h₁ h₂
  sorry

end g_1000_value_l297_297817


namespace min_moves_to_reset_counters_l297_297918

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l297_297918


namespace find_a_plus_b_l297_297283

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 3 * a + b

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) (h2 : 2 * a = 3 - a) : a + b = 1 :=
by
  unfold f at h1
  sorry

end find_a_plus_b_l297_297283


namespace arithmetic_seq_min_S19_l297_297294

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_min_S19
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_S8 : S a 8 ≤ 6)
  (h_S11 : S a 11 ≥ 27) :
  S a 19 ≥ 133 :=
sorry

end arithmetic_seq_min_S19_l297_297294


namespace sin_2phi_value_l297_297340

theorem sin_2phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π)
  (sym : ∀ x, sin (π * x + φ) - 2 * cos (π * x + φ) = sin (π * (1 - x) + φ) - 2 * cos (π * (1 - x) + φ)) :
  sin (2 * φ) = -4/5 :=
by
  sorry

end sin_2phi_value_l297_297340


namespace max_M_eq_sqrt2_add_1_l297_297029

theorem max_M_eq_sqrt2_add_1 (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  (sqrt (abs (x - y)) + sqrt (abs (y - z)) + sqrt (abs (z - x))) ≤ (sqrt 2 + 1) := 
sorry

end max_M_eq_sqrt2_add_1_l297_297029


namespace frog_final_position_probability_l297_297988

noncomputable def frog_position_probability (positions : ℕ → ℝ^2) : ℚ :=
  sorry -- Define position sequence and probability distribution accurately

theorem frog_final_position_probability :
  frog_position_probability (λ n, (1 : ℝ^2)) = 1/8 :=
sorry -- Prove that the probability of the final position being within a 1-meter radius after 4 jumps is 1/8

end frog_final_position_probability_l297_297988


namespace math_problem_l297_297709

variables (a b : ℝ×ℝ)
variable (θ : ℝ)

-- Conditions
def cond1 := a = (1, -2)
def cond2 := b = (-1, -1)
def cond3 := θ = Real.pi / 4
def a_plus_b := (a.1 + b.1, a.2 + b.2)
def two_a_minus_b := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Question 1
def magnitude_eq (v : ℝ×ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def question1 := magnitude_eq two_a_minus_b = 3 * Real.sqrt 2

-- Question 2
def dot_product (v w : ℝ×ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def question2 := dot_product a b = 1

-- Theorem statement to prove
theorem math_problem :
  cond1 → cond2 → cond3 → question1 ∧ question2 :=
by intros; sorry

end math_problem_l297_297709


namespace savings_correct_l297_297525

noncomputable def school_price_math : Float := 45
noncomputable def school_price_science : Float := 60
noncomputable def school_price_literature : Float := 35

noncomputable def discount_math : Float := 0.20
noncomputable def discount_science : Float := 0.25
noncomputable def discount_literature : Float := 0.15

noncomputable def tax_school : Float := 0.07
noncomputable def tax_alt : Float := 0.06
noncomputable def shipping_alt : Float := 10

noncomputable def alt_price_math : Float := (school_price_math * (1 - discount_math)) * (1 + tax_alt)
noncomputable def alt_price_science : Float := (school_price_science * (1 - discount_science)) * (1 + tax_alt)
noncomputable def alt_price_literature : Float := (school_price_literature * (1 - discount_literature)) * (1 + tax_alt)

noncomputable def total_alt_cost : Float := alt_price_math + alt_price_science + alt_price_literature + shipping_alt

noncomputable def school_price_math_tax : Float := school_price_math * (1 + tax_school)
noncomputable def school_price_science_tax : Float := school_price_science * (1 + tax_school)
noncomputable def school_price_literature_tax : Float := school_price_literature * (1 + tax_school)

noncomputable def total_school_cost : Float := school_price_math_tax + school_price_science_tax + school_price_literature_tax

noncomputable def savings : Float := total_school_cost - total_alt_cost

theorem savings_correct : savings = 22.40 := by
  sorry

end savings_correct_l297_297525


namespace quadrilateral_circle_problem_l297_297981

noncomputable def square_of_radius (EX XF GY YH : ℝ) : ℝ :=
  let r := 77.538 in r^2

theorem quadrilateral_circle_problem :
  ∀ (EX XF GY YH : ℝ),
    EX = 13 →
    XF = 31 →
    GY = 43 →
    YH = 17 →
    square_of_radius EX XF GY YH = 6011.978 :=
by
  intros EX XF GY YH hEX hXF hGY hYH
  rw [hEX, hXF, hGY, hYH]
  exact rfl

end quadrilateral_circle_problem_l297_297981


namespace smallest_circle_equation_l297_297661

theorem smallest_circle_equation :
  ∃ (x y r : ℝ), x > 0 ∧ y = 2 / x ∧ r = real.abs (2 * x + 2 / x + 1) / real.sqrt 5 ∧ 
  (∀ (a : ℝ), a > 0 → (real.abs (2 * a + 2 / a + 1) / real.sqrt 5) ≥ r) ∧ 
  r = real.sqrt 5 ∧ 
  ∀ (r : ℝ), (x-1) ^ 2 + (y-2) ^ 2 = r ^ 2 :=
by
  sorry

end smallest_circle_equation_l297_297661


namespace isosceles_right_triangle_AB_length_l297_297401

-- Definitions
def isosceles_right_triangle (A B C : ℝ) : Prop :=
  ∀ (angle_A angle_B angle_C hypotenuse_BC : ℝ),
    angle_B = 90 ∧ angle_A = 45 ∧ angle_C = 45 ∧
    hypotenuse_BC = 10 * real.sqrt 2 → 
    A = 10

-- Theorem Statement
theorem isosceles_right_triangle_AB_length {A B C : ℝ}
  (h : isosceles_right_triangle A B C) : A = 10 := by 
  sorry

end isosceles_right_triangle_AB_length_l297_297401


namespace ratio_of_first_to_second_l297_297534

theorem ratio_of_first_to_second (x y : ℕ) 
  (h1 : x + y + (1 / 3 : ℚ) * x = 110)
  (h2 : y = 30) :
  x / y = 2 :=
by
  sorry

end ratio_of_first_to_second_l297_297534


namespace ordered_pairs_count_l297_297368

theorem ordered_pairs_count : 
  ∃ pairs : Set (ℝ × ℤ), (∀ p ∈ pairs, 0 < p.1 ∧ 3 ≤ p.2 ∧ p.2 ≤ 300 ∧ (Real.log p.1 / Real.log p.2) ^ 101 = Real.log (p.1 ^ 101) / Real.log p.2) ∧ pairs.card = 894 :=
sorry

end ordered_pairs_count_l297_297368


namespace stratified_sampling_type_B_l297_297846

/-- Given the production volumes of four types of dairy products, A, B, C, and D, in a dairy factory
and a sample of size 60 selected using stratified sampling based on production volumes,
prove that the number of type B dairy products in the sample is 15. -/
theorem stratified_sampling_type_B
  (A B C D : ℕ) (n : ℕ)
  (hA : A = 2000) (hB : B = 1250) (hC : C = 1250) (hD : D = 500) (hn : n = 60) :
  let total := A + B + C + D in
  let proportionB := (B : ℚ) / total in
  let sampleB := proportionB * n in
  sampleB = 15 :=
by
  sorry

end stratified_sampling_type_B_l297_297846


namespace pipe_B_fill_time_l297_297608

theorem pipe_B_fill_time (t : ℕ) : 
  (∀ A B, A = 60 ∧ ∃ t, (30 * (1 / t) + 30 * ((1 / 60) + (1 / t)) = 1)) → t = 40 := by
  sorry

end pipe_B_fill_time_l297_297608


namespace value_of_t_l297_297723

theorem value_of_t (t : ℝ) (x y : ℝ) (h : 3 * x^(t-1) + y - 5 = 0) :
  t = 2 :=
sorry

end value_of_t_l297_297723


namespace number_of_pairs_l297_297826

theorem number_of_pairs (n : ℕ) :
  let num_pairs := (n^2 - 1)^2
  in ∃ num_pairs_a_b : ℕ, num_pairs_a_b = num_pairs ∧ 
     ∀ (a b : ℕ), (4 * a - b) * (4 * b - a) = 2010^n ↔ num_pairs_a_b = (n^2 - 1)^2 :=
by
  sorry

end number_of_pairs_l297_297826


namespace altitudes_reciprocal_inequality_l297_297857

theorem altitudes_reciprocal_inequality
  {a b c m_a m_b m_c s t δ : ℝ}
  (h₁ : a + b + c = 2 * s)
  (h₂ : a * m_a = 2 * t)
  (h₃ : b * m_b = 2 * t)
  (h₄ : (a + b + c) * δ = 2 * t)
  (h₅ : δ = varrho)
  (h₆ : c < a + b)
  (h₇ : a + b < a + b + 2 * c) :
  (1 / (2 * varrho)) < (1 / m_a) + (1 / m_b) ∧ (1 / m_a) + (1 / m_b) < 1 / varrho :=
begin
  sorry
end

end altitudes_reciprocal_inequality_l297_297857


namespace triangle_ABC_is_acute_l297_297767

-- Define a mathematical context for the problem
def condition1 (A : ℝ) : Prop := A = real.arctan 2
def condition2 (B : ℝ) : Prop := B = real.arctan 3

-- The statement to prove
theorem triangle_ABC_is_acute (A B C : ℝ) 
  (h1 : A = real.arctan 2)
  (h2 : B = real.arctan 3)
  (h3 : A + B + C = real.pi) : 
  (0 < A) ∧ (A < real.pi / 2) ∧ (0 < B) ∧ (B < real.pi / 2) ∧ (0 < C) ∧ (C < real.pi / 2) :=
sorry

end triangle_ABC_is_acute_l297_297767


namespace total_time_with_dog_l297_297471

theorem total_time_with_dog :
  let bathing_time := 20
  let blow_drying_time := 10
  let fetch_time := 15
  let training_time := 10
  let walk_first_mile_time := (1 / 6) * 60
  let walk_second_mile_time := (1 / 4) * 60
  let walk_third_mile_time := (1 / 8) * 60
  bathing_time + blow_drying_time + fetch_time + training_time
  + walk_first_mile_time + walk_second_mile_time + walk_third_mile_time = 87.5 :=
by
  let bathing_time := 20
  let blow_drying_time := 10
  let fetch_time := 15
  let training_time := 10
  let walk_first_mile_time := (1 / 6) * 60
  let walk_second_mile_time := (1 / 4) * 60
  let walk_third_mile_time := (1 / 8) * 60
  have : bathing_time + blow_drying_time + fetch_time + training_time
         + walk_first_mile_time + walk_second_mile_time + walk_third_mile_time = 87.5 := by sorry
  exact this

end total_time_with_dog_l297_297471


namespace modulus_of_z_l297_297716

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the given complex number
def z : ℂ := (1 - i) / (1 + i)

-- State the theorem: The modulus of z equals 1
theorem modulus_of_z : complex.abs z = 1 :=
by sorry

end modulus_of_z_l297_297716


namespace sum_of_first_11_terms_l297_297409

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + a 8 = 16) : (11 / 2) * (a 1 + a 11) = 88 :=
by
  sorry

end sum_of_first_11_terms_l297_297409


namespace six_people_paint_time_l297_297003

noncomputable def time_to_paint_house_with_six_people 
    (initial_people : ℕ) (initial_time : ℝ) (less_efficient_worker_factor : ℝ) 
    (new_people : ℕ) : ℝ :=
  let initial_total_efficiency := initial_people - 1 + less_efficient_worker_factor
  let total_work := initial_total_efficiency * initial_time
  let new_total_efficiency := (new_people - 1) + less_efficient_worker_factor
  total_work / new_total_efficiency

theorem six_people_paint_time (initial_people : ℕ) (initial_time : ℝ) 
    (less_efficient_worker_factor : ℝ) (new_people : ℕ) :
    initial_people = 5 → initial_time = 10 → less_efficient_worker_factor = 0.5 → new_people = 6 →
    time_to_paint_house_with_six_people initial_people initial_time less_efficient_worker_factor new_people = 8.18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end six_people_paint_time_l297_297003


namespace triangle_ABC_proof_by_contradiction_l297_297484

theorem triangle_ABC_proof_by_contradiction
  (A B C : Point)
  (hAB_AC : AB = AC)
  (hC_contradiction : ∠C ≥ 90) :
  False :=
sorry

end triangle_ABC_proof_by_contradiction_l297_297484


namespace smallest_x_value_l297_297325

def f (x : ℝ) : ℝ :=
  1 + x - x^3 / 3 + x^5 / 5 - x^7 / 7 + x^9 / 9 - x^11 / 11 + x^13 / 13

theorem smallest_x_value 
  (t : ℝ) (ht : -2 < t ∧ t < -1) 
  (h_f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :
  ∃ x : ℤ, x ≥ ⌈t + 1⌉ :=
begin
  -- Provided statement to guarantee Lean script completeness
  sorry,
end

end smallest_x_value_l297_297325


namespace median_of_data_set_l297_297291

theorem median_of_data_set (x : ℝ) (h_mode : multiset.mode ({-1, 4, x, 6, 15} : multiset ℝ) = 6) : 
  multiset.median ({-1, 4, x, 6, 15} : multiset ℝ) = 6 := 
sorry

end median_of_data_set_l297_297291


namespace final_temperature_l297_297059

theorem final_temperature (initial_temp cost_per_tree spent amount temperature_drop : ℝ) 
  (h1 : initial_temp = 80) 
  (h2 : cost_per_tree = 6)
  (h3 : spent = 108) 
  (h4 : temperature_drop = 0.1) 
  (trees_planted : ℝ) 
  (h5 : trees_planted = spent / cost_per_tree) 
  (temp_reduction : ℝ) 
  (h6 : temp_reduction = trees_planted * temperature_drop) 
  (final_temp : ℝ) 
  (h7 : final_temp = initial_temp - temp_reduction) : 
  final_temp = 78.2 := 
by
  sorry

end final_temperature_l297_297059


namespace carl_candy_bars_l297_297244

def weekly_earnings := 0.75
def weeks := 4
def candy_bar_cost := 0.50

theorem carl_candy_bars :
  (weeks * weekly_earnings) / candy_bar_cost = 6 := by
  sorry

end carl_candy_bars_l297_297244


namespace box_side_length_l297_297948

theorem box_side_length 
  (cost_per_box : ℝ := 0.9) 
  (total_volume : ℝ := 3.06 * 10^6) 
  (min_total_cost : ℝ := 459) 
  (n_boxes : ℝ := min_total_cost / cost_per_box) 
  (volume_per_box : ℝ := total_volume / n_boxes) : 
  real.sqrt3 volume_per_box ≈ 18.18 :=
by 
  sorry

end box_side_length_l297_297948


namespace prove_PMQN_area_prove_OPQ_area_l297_297903

variables (A B C D O M N P Q : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space O] [metric_space M] [metric_space N] [metric_space P] [metric_space Q]

-- Conditions
variable (h1 : is_convex_quadrilateral A B C D)
variable (h2 : extension_of_sides_intersect_at O A D B C)
variable (h_m : midpoint M A B)
variable (h_n : midpoint N C D)
variable (h_p : midpoint P A C)
variable (h_q : midpoint Q B D)

-- Part (a)
theorem prove_PMQN_area 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : extension_of_sides_intersect_at O A D B C)
  (h_m : midpoint M A B)
  (h_n : midpoint N C D)
  (h_p : midpoint P A C)
  (h_q : midpoint Q B D) :
  area P M Q N = abs (area A B D - area A C D) / 2 := 
sorry

-- Part (b)
theorem prove_OPQ_area 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : extension_of_sides_intersect_at O A D B C)
  (h_m : midpoint M A B)
  (h_n : midpoint N C D)
  (h_p : midpoint P A C)
  (h_q : midpoint Q B D) :
  area O P Q = area A B C D / 4 := 
sorry

end prove_PMQN_area_prove_OPQ_area_l297_297903


namespace regular_hexagon_area_l297_297489

theorem regular_hexagon_area :
  ∀ (P R : ℝ × ℝ),
    P = (0, 0) →
    R = (8, 2) →
    let PR := dist P R in
    (PR^2 = 68) →
    area_of_hexagon PR = 34 * Real.sqrt 3 :=
by
  intro P R hp hr hpr
  unfold area_of_hexagon
  rw [hp, hr]
  simp [dist]
  sorry

end regular_hexagon_area_l297_297489


namespace Alice_wins_tournament_l297_297197

noncomputable def probAliceWins : ℚ := 6 / 7

theorem Alice_wins_tournament :
  let n := 8
  let playStyle (player: ℕ) : String := if player == 0 then "rock"
                                         else if player == 1 then "paper"
                                         else "scissors"
  -- Define rules for rock, paper, scissors
  let beats (a b : String) : Bool := if (a = "rock" ∧ b = "scissors") ∨
                                       (a = "scissors" ∧ b = "paper") ∨
                                       (a = "paper" ∧ b = "rock")
                                     then true
                                     else false
  -- Define a function to pair players randomly and simulate the tournament
  -- Pseudocode: pair up players, simulate matches, and find winner iteratively
  
  let randomPairingLogic (p : ℕ) : String := sorry -- Simulation logic here  

  -- Probability that Alice (Player 0) wins under given conditions
  let probAliceWins : ℚ := 6 / 7
in 

probAliceWins = 6 / 7 := sorry

end Alice_wins_tournament_l297_297197


namespace binary_101_eq_5_l297_297650

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l297_297650


namespace coral_population_decreases_l297_297674

theorem coral_population_decreases {P : ℝ} (hP : 0 < P):
  ∃ n : ℕ, (n = 9 ∧ 0.75^n * P < 0.05 * P) := 
begin
  use 9,
  have : 0.75 ^ 9 * P < 0.05 * P,
  { calc 0.75 ^ 9 * P = 0.075084686279296875 * P : by norm_num
                   ... < 0.05 * P               : by simp [hP]; norm_num, },
  exact ⟨rfl, this⟩,
end

end coral_population_decreases_l297_297674


namespace triangle_perimeter_l297_297601

theorem triangle_perimeter (a : ℝ) (h : a = 70 * Real.sqrt 2) : 
  let leg := a in
  let hypotenuse := a * Real.sqrt 2 in
  let perimeter := 2 * leg + hypotenuse in
  perimeter = 140 * Real.sqrt 2 + 140 :=
by {
  sorry
}

end triangle_perimeter_l297_297601


namespace sum_of_integers_between_100_and_500_ending_in_3_l297_297225

theorem sum_of_integers_between_100_and_500_ending_in_3 : 
  (∑ n in finset.filter (λ n : ℤ, 100 ≤ n ∧ n ≤ 500 ∧ n % 10 = 3) (finset.Icc (100 : ℤ) 500)) = 11920 := 
by { sorry }

end sum_of_integers_between_100_and_500_ending_in_3_l297_297225


namespace xy_product_l297_297861

theorem xy_product (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
sorry

end xy_product_l297_297861


namespace vanessa_recycled_correct_l297_297557

-- Define conditions as separate hypotheses
variable (weight_per_point : ℕ := 9)
variable (points_earned : ℕ := 4)
variable (friends_recycled : ℕ := 16)

-- Define the total weight recycled as points earned times the weight per point
def total_weight_recycled (points_earned weight_per_point : ℕ) : ℕ := points_earned * weight_per_point

-- Define the weight recycled by Vanessa
def vanessa_recycled (total_recycled friends_recycled : ℕ) : ℕ := total_recycled - friends_recycled

-- Main theorem statement
theorem vanessa_recycled_correct (weight_per_point points_earned friends_recycled : ℕ) 
    (hw : weight_per_point = 9) (hp : points_earned = 4) (hf : friends_recycled = 16) : 
    vanessa_recycled (total_weight_recycled points_earned weight_per_point) friends_recycled = 20 := 
by 
  sorry

end vanessa_recycled_correct_l297_297557


namespace functional_equation_solution_unique_l297_297676

theorem functional_equation_solution_unique {f : ℝ → ℝ} 
  (H : ∀ x y : ℝ, f (f x + y) = f (x + y) + y * f y) :
  f = λ x, 0 :=
by
  -- Proof goes here
  sorry

end functional_equation_solution_unique_l297_297676


namespace value_of_a_l297_297766

/-
The center of the circle x^2 + y^2 + 2x - 4y = 0 is (-1, 2).
If the line 3x + y + a = 0 passes through the center of the circle,
then prove that a = 1.
-/
theorem value_of_a (a : ℝ) :
  let center := (-1, 2)
  let line_through_center := 3 * center.1 + center.2 + a = 0
  line_through_center → a = 1 :=
by
  intros center line_through_center
  rw [←line_through_center]
  sorry

end value_of_a_l297_297766


namespace cos_sin_36_pow_195_eq_neg1_l297_297639

noncomputable def complex_cos_sin (θ : ℝ) : ℂ := complex.exp (θ * real.cos θ + complex.I * real.sin θ)

theorem cos_sin_36_pow_195_eq_neg1 :
  (complex_cos_sin 195) ^ 36 = -1 :=
by
  sorry

end cos_sin_36_pow_195_eq_neg1_l297_297639


namespace profit_percent_l297_297386

theorem profit_percent (SP : ℝ) (h : SP > 0) : 
  let CP := 0.4 * SP in 
  let Profit := SP - CP in
  (Profit / CP * 100) = 150 := 
by 
  sorry

end profit_percent_l297_297386


namespace number_of_odd_factors_of_252_l297_297363

def numOddFactors (n : ℕ) : ℕ :=
  if ∀ d : ℕ, n % d = 0 → ¬(d % 2 = 0) then d
  else 0

theorem number_of_odd_factors_of_252 : numOddFactors 252 = 6 := by
  -- Definition of n
  let n := 252
  -- Factor n into 2^2 * 63
  have h1 : n = 2^2 * 63 := rfl
  -- Find the number of odd factors of 63 since factors of 252 that are odd are the same as factors of 63
  have h2 : 63 = 3^2 * 7 := rfl
  -- Check the number of factors of 63
  sorry

end number_of_odd_factors_of_252_l297_297363


namespace eccentricity_of_ellipse_l297_297594

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse :
  ∃ (a b c e : ℝ),
    2 * (-1) - 0 + 2 = 0 ∧ -- condition for left focus
    2 * 0 - 2 + 2 = 0 ∧ -- condition for vertex
    a^2 = b^2 + c^2 ∧ -- ellipse relationship
    e = ellipse_eccentricity a b c ∧ -- eccentricity formula
    e = (Real.sqrt 5) / 5 := -- correct answer
by
  use [Real.sqrt 5, 2, 1, (Real.sqrt 5) / 5]
  have h1 : ∀ x y : ℝ, 2 * x - y + 2 = 0 → (x, y) = (-1, 0) ∨ (x, y) = (0, 2), 
    from sorry
  split,
  { exact h1 (-1) 0 (by linarith) },
  split,
  { exact h1 0 2 (by linarith) },
  split,
  { calc (Real.sqrt 5)^2 = 5 : by rw [Real.sq_sqrt (by norm_num : 5 ≥ 0)]
         ... = 4 + 1 : by norm_num },
  split,
  { refl },
  { refl }

end eccentricity_of_ellipse_l297_297594


namespace second_x_intersection_is_10_l297_297979

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

noncomputable def circle_equation (center : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = r ^ 2

theorem second_x_intersection_is_10 :
  let a : ℝ × ℝ := (0, 0)
  let b : ℝ × ℝ := (10, -6)
  let c := midpoint a b
  let r := distance a c
  ∃ p : ℝ × ℝ, circle_equation c r p ∧ p.2 = 0 ∧ p.1 = 10 :=
by
  sorry

end second_x_intersection_is_10_l297_297979


namespace num_odd_factors_of_252_l297_297354

theorem num_odd_factors_of_252 : 
  ∃ n : ℕ, n = 252 ∧ 
  ∃ k : ℕ, (k = ∏ d in (divisors_filter (λ x, x % 2 = 1) n), 1) 
  ∧ k = 6 := 
sorry

end num_odd_factors_of_252_l297_297354


namespace find_pair_l297_297346

-- Assume the necessary conditions are given
variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b p : V)

-- Define the condition as a hypothesis
def condition : Prop := dist p b = 3 * dist p a

-- Define the target pair (s, v)
def target_pair : ℝ × ℝ := (9 / 8, -1 / 8)

-- The main theorem proving the existence of (s, v) such that p is at a fixed distance from s*a + v*b
theorem find_pair (h : condition a b p) : 
  ∃ (s v : ℝ), (dist p (s • a + v • b) = dist p (target_pair.1 • a + target_pair.2 • b)) ∧ (s, v) = target_pair :=
sorry

end find_pair_l297_297346


namespace probability_sum_is_nine_l297_297935

theorem probability_sum_is_nine :
  ∃ p : ℚ, p = 1 / 4 ∧
  let possible_sums := [(1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1)], 
  let total_outcomes := 8 * 8,
  let favorable_outcomes := 2 * possible_sums.length,
  p = favorable_outcomes / total_outcomes :=
by
  let possible_sums := [(1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1)]
  have total_outcomes : ℕ := 8 * 8
  have favorable_outcomes : ℕ := 2 * possible_sums.length
  have p : ℚ := favorable_outcomes / total_outcomes
  use p
  split
  · -- Show p = 1 / 4
    have : favorable_outcomes / total_outcomes = (2 * possible_sums.length) / (8 * 8) := by
      simp [favorable_outcomes, total_outcomes]
    have : (2 * possible_sums.length) / (8 * 8) = 1 / 4 := by
      norm_num [possible_sums.length]
    exact this
  · rfl


end probability_sum_is_nine_l297_297935


namespace probability_at_least_one_three_l297_297545

theorem probability_at_least_one_three :
  let E := { (d1, d2) : Fin 8 × Fin 8 | d1 = 2 ∨ d2 = 2 } in
  (↑E.card / ↑((Fin 8 × Fin 8).card) : ℚ) = 15 / 64 :=
by
  /- Let E be the set of outcomes where at least one die shows a 3. -/
  sorry

end probability_at_least_one_three_l297_297545


namespace throw_percentage_first_day_l297_297616

variable (initial_apples remaining_apples first_day_second_half_apples first_day_throw_percent : ℕ)

-- Conditions
def sells_half : Prop := initial_apples / 2 = remaining_apples
def throws_away_first_half : Prop := first_day_throw_percent * remaining_apples / 100 = first_day_second_half_apples
def sells_half_next_day : Prop := (remaining_apples - first_day_second_half_apples) / 2 = initial_apples / 4
def throws_away_second_half : Prop := (remaining_apples - first_day_second_half_apples) / 2 = first_day_second_half_apples
def total_throws_30 : Prop := first_day_second_half_apples + first_day_second_half_apples = initial_apples * 30 / 100

-- Theorem to prove
theorem throw_percentage_first_day:
  sells_half initial_apples remaining_apples →
  throws_away_first_half initial_apples remaining_apples first_day_second_half_apples first_day_throw_percent →
  sells_half_next_day initial_apples remaining_apples first_day_second_half_apples →
  throws_away_second_half initial_apples remaining_apples first_day_second_half_apples →
  total_throws_30 initial_apples remaining_apples first_day_second_half_apples →
  first_day_throw_percent = 20 := by
  sorry

end throw_percentage_first_day_l297_297616


namespace arithmetic_sequence_properties_exists_n_1008_l297_297835

noncomputable def a : ℕ → ℕ 
| 1       := 1
| (n + 1) := let S_n := (n + 1) * a (n + 1) - 2 * (n + 1) * n
             in S_n / (n + 1) + 2 * n

noncomputable def S : ℕ → ℕ 
| 1       := 1
| (n + 1) := (n + 1) * a (n + 1) - 2 * (n + 1) * n

theorem arithmetic_sequence_properties :
  (a 2 = 5) ∧ (a 3 = 9) ∧ (∀ n, a (n + 1) - a n = 4) ∧ (∀ n, S (n + 1) = 2 * (n + 1)^2 - (n + 1)) :=
by
  sorry

theorem exists_n_1008 :
  ∃ n, (S 1 + S 2 / 2 + S 3 / 3 + ... + S n / n - (n - 1)^2 = 2015 ∧ n = 1008) :=
by
  sorry

end arithmetic_sequence_properties_exists_n_1008_l297_297835


namespace even_perm_as_3_cycles_l297_297023

/-- Definition of a 3-cycle permutation -/
def is_3_cycle {n : ℕ} (c : Fin n → Fin n) : Prop :=
  ∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i2 ≠ i3 ∧ i3 ≠ i1 ∧
  c i1 = i2 ∧ c i2 = i3 ∧ c i3 = i1 ∧ 
  (∀ j : Fin n, j ≠ i1 ∧ j ≠ i2 ∧ j ≠ i3 → c j = j)

/-- An even permutation can be written as a composition of 3-cycles -/
theorem even_perm_as_3_cycles {n : ℕ} (σ : Equiv.Perm (Fin n))
  (h_even : σ.sign = 1) : 
  ∃ c : List (Equiv.Perm (Fin n)), 
  (∀ τ ∈ c, is_3_cycle τ) ∧ σ = c.prod :=
sorry

end even_perm_as_3_cycles_l297_297023


namespace Maxim_is_correct_l297_297154

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l297_297154


namespace survey_C_count_l297_297938

theorem survey_C_count 
    (N : ℕ) (Ks : ℕ) (first_number : ℕ) (interval : ℕ) 
    (survey_C_range_start : ℕ) (survey_C_range_end : ℕ) : 
    N = 1000 ∧ Ks = 50 ∧ first_number = 8 ∧ interval = 20 ∧ 
    survey_C_range_start = 751 ∧ survey_C_range_end = 1000 
    → 
    let a_n := λ n : ℕ, first_number + (n - 1) * interval in
    let valid_ns := λ n : ℕ, survey_C_range_start ≤ a_n n ∧ a_n n ≤ survey_C_range_end in
    Nat.card (set_of valid_ns) = 12 :=
sorry

end survey_C_count_l297_297938


namespace find_angle_AIE_l297_297415

def angle_AIE (BAC ABC : ℝ) : ℝ := (BAC + ABC) / 2

theorem find_angle_AIE (BAC ABC ACB : ℝ) (hACB : ACB = 45) (h_sum : BAC + ABC + ACB = 180) : 
  angle_AIE BAC ABC = 67.5 := 
by 
  have hBAC_ABC : BAC + ABC = 135 :=
    calc
      BAC + ABC := 180 - ACB : by rw [hACB, ←h_sum, sub_self]
      ... = 135 : by norm_num
  have hAIE := calc
    angle_AIE BAC ABC = (BAC + ABC) / 2 : by rfl
    ... = 135 / 2 : by rw hBAC_ABC
    ... = 67.5 : by norm_num
  exact hAIE

end find_angle_AIE_l297_297415


namespace probability_prime_l297_297883

open finset

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

def possible_outcomes : finset ℕ := finset.range 7

def prime_outcomes : finset ℕ := {2, 3, 5}

theorem probability_prime :
  (prime_outcomes.card.to_real / possible_outcomes.card.to_real) = 1 / 2 :=
by
  sorry

end probability_prime_l297_297883


namespace intersection_P_Q_l297_297236

open Set

def P : Set ℝ := {x | abs (x - 2) ≤ 1}
def Q : Set ℕ := {x | True}

theorem intersection_P_Q : (P ∩ Q : Set ℝ) = {1, 2, 3} := by
  sorry

end intersection_P_Q_l297_297236


namespace oak_trees_after_five_days_l297_297799

theorem oak_trees_after_five_days :
  let initial_trees := 5
  let planting_days := [3, 3, 4, 4, 4]
  let removing_days := [2, 2, 2, 1, 1]
  let net_change := list.sum (list.map2 (λ p r => p - r) planting_days removing_days)
  initial_trees + net_change = 15 :=
by
  let initial_trees := 5
  let planting_days := [3, 3, 4, 4, 4]
  let removing_days := [2, 2, 2, 1, 1]
  let net_change := list.sum (list.map2 (λ p r => p - r) planting_days removing_days)
  show initial_trees + net_change = 15
  sorry

end oak_trees_after_five_days_l297_297799


namespace parallel_sides_l297_297705

-- Define types and main structure
variables {ABC : Type} [is_triangle ABC]
variables {C₁ A₁ B₁ C₂ A₂ B₂ : ABC}
variables {n : ℝ}

-- Define the conditions
def ratio_cond_ABC (ABC : Type) [is_triangle ABC] (C₁ A₁ B₁ : ABC) (n : ℝ) :=
  (AC₁ / C₁B = 1 / n) ∧ (BA₁ / A₁C = 1 / n) ∧ (CB₁ / B₁A = 1 / n)

def ratio_cond_A₁B₁C₁ (ABC : Type) [is_triangle ABC] (C₂ A₂ B₂ : ABC) (n : ℝ) :=
  (A₁C₂ / C₂B₁ = n) ∧ (B₁A₂ / A₂C₁ = n) ∧ (C₁B₂ / B₂A₁ = n)

-- State the theorem
theorem parallel_sides (h_ABC : is_triangle ABC)
    (h_ratio_ABC : ratio_cond_ABC ABC C₁ A₁ B₁ n)
    (h_ratio_A₁B₁C₁ : ratio_cond_A₁B₁C₁ ABC C₂ A₂ B₂ n) :
  (parallel A₂C₂ AC) ∧ (parallel C₂B₂ CB) ∧ (parallel B₂A₂ BA) :=
by
  sorry

end parallel_sides_l297_297705


namespace min_val_x_2y_l297_297026

noncomputable def min_x_2y (x y : ℝ) : ℝ :=
  x + 2 * y

theorem min_val_x_2y : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / (x + 2) + 1 / (y + 2) = 1 / 3) → 
  min_x_2y x y ≥ 3 + 6 * Real.sqrt 2 :=
by
  intros x y x_pos y_pos eqn
  sorry

end min_val_x_2y_l297_297026


namespace find_DL_l297_297541

variable (D E F : Type)

structure Triangle (D E F : Type) :=
  (DE EF FD : ℝ)
  (DE_pos : 0 < DE)
  (EF_pos : 0 < EF)
  (FD_pos : 0 < FD)
  (DE_length : DE = 6)
  (EF_length : EF = 10)
  (FD_length : FD = 8)

structure Circle (ω : Type) (P Q : Type) :=
  (passes_through_P : ω → P)
  (tangent_at_D : ω → P)

variable (ω1 ω2 L : Type)

axiom ω1_tangent : ∀ (ω : Type), Circle ω D E
axiom ω2_tangent : ∀ (ω : Type), Circle ω D F

theorem find_DL {D E F : Type} (t : Triangle D E F) (L : Type)
  (ω1 : ∀ (ω : Type), Circle ω D E)
  (ω2 : ∀ (ω : Type), Circle ω D F)
  (ω1ω2_intersect : ∃ L, L ≠ D ∧ L ∈ ω1 ∧ L ∈ ω2)
  : True := sorry

end find_DL_l297_297541


namespace xiaoming_in_top_7_l297_297394

-- Given conditions
variables (unique_scores : Fin 15 → ℝ)
variable (xiaoming_score : ℝ)
#check List.median -- The right median is provided by this helper.

-- Definition of the proof problem
theorem xiaoming_in_top_7 (median_score : ℝ) 
  (h : List.median (List.ofFn unique_scores) = median_score) : 
  (xiaoming_score ∈ (List.take 7 (List.sort (≤) (List.ofFn unique_scores)))) ↔
  (xiaoming_score > median_score) :=
sorry

end xiaoming_in_top_7_l297_297394


namespace n_is_prime_l297_297659

-- Definitions for Euler's totient function and number of distinct prime divisors
def φ (n : ℕ) : ℕ := if n = 0 then 0 else (n.nat_pred).succ.gcd_totient

def ω (n : ℕ) : ℕ := n.factors.erase_dup.length

-- Problem statement: Prove that n is a prime number given the conditions
theorem n_is_prime (n : ℕ) (h1 : φ(n) ∣ (n - 1)) (h2 : ω(n) ≤ 3) : prime n :=
sorry

end n_is_prime_l297_297659


namespace third_year_profit_option1_more_cost_effective_l297_297160

-- Define the initial conditions and parameters
def initial_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def annual_expense_increase : ℕ := 40000
def annual_income : ℕ := 500000

-- Define the function to compute expenses for a given year
def expenses (year : ℕ) : ℕ :=
if year = 1 then first_year_expenses
else first_year_expenses + (year - 1) * annual_expense_increase

-- Define the function to compute total profit by a given year
def total_profit (year : ℕ) : ℕ :=
year * annual_income - initial_cost - (List.sum (List.map expenses (List.range year)))

-- Define the function to compute average profit by a given year
def average_profit (year : ℕ) : ℕ :=
total_profit year / year

-- Theorem to prove the company starts to make a profit from the third year
theorem third_year_profit : total_profit 3 > 0 := by
  sorry

-- Define a function to compute cost-effectiveness of options
def cost_effectiveness (option1_value option2_value : ℕ) : Bool :=
option1_value > option2_value

-- Suppose we calculate and both values for the options are available
def option1_value : ℕ := -- some calculated value for option 1
def option2_value : ℕ := -- some calculated value for option 2

-- Theorem to prove the first option is more cost-effective
theorem option1_more_cost_effective : cost_effectiveness option1_value option2_value = true := by
  sorry

end third_year_profit_option1_more_cost_effective_l297_297160


namespace problem_1_problem_2_problem_3_l297_297832

variable {f : ℝ → ℝ}

axiom f_add : ∀ (x y : ℝ), f(x + y) = f(x) + f(y)
axiom f_neg : ∀ (x : ℝ), x > 0 → f(x) < 0
axiom f_one : f(1) = -2

theorem problem_1 : ∀ (x : ℝ), f(-x) = -f(x) := sorry

theorem problem_2 : ∀ (x1 x2 : ℝ), x1 < x2 → f(x1) > f(x2) := sorry

theorem problem_3 : ∃ (y z : ℝ), -3 ≤ y ∧ y ≤ 3 ∧ -3 ≤ z ∧ z ≤ 3 ∧ 
  (∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → f(y) ≤ f(x) ∧ f(x) ≤ f(z)) :=
sorry

end problem_1_problem_2_problem_3_l297_297832


namespace initial_parking_hours_proof_l297_297897

noncomputable def initial_parking_hours (total_cost : ℝ) (excess_hourly_rate : ℝ) (average_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let h := (total_hours * average_cost - total_cost) / excess_hourly_rate
  h

theorem initial_parking_hours_proof : initial_parking_hours 21.25 1.75 2.361111111111111 9 = 2 :=
by
  sorry

end initial_parking_hours_proof_l297_297897


namespace height_of_sky_island_l297_297411

theorem height_of_sky_island (day_climb : ℕ) (night_slide : ℕ) (days : ℕ) (final_day_climb : ℕ) :
  day_climb = 25 →
  night_slide = 3 →
  days = 64 →
  final_day_climb = 25 →
  (days - 1) * (day_climb - night_slide) + final_day_climb = 1411 :=
by
  -- Add the formal proof here
  sorry

end height_of_sky_island_l297_297411


namespace solution_set_of_inequality_l297_297097

theorem solution_set_of_inequality :
  { x : ℝ | (2 * x - 1) / (x + 1) ≤ 1 } = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_of_inequality_l297_297097


namespace tip_calculation_l297_297074

def women's_haircut_cost : ℝ := 48
def children's_haircut_cost : ℝ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℝ := 0.20

theorem tip_calculation : 
  let total_cost := (children's_haircut_cost * number_of_children) + women's_haircut_cost in
  let tip := total_cost * tip_percentage in
  tip = 24 :=
by 
  sorry

end tip_calculation_l297_297074


namespace correct_number_of_relations_is_two_l297_297816

noncomputable def a : ℕ := 3

noncomputable def M : set ℝ := {x | x ≤ real.sqrt 10}

def is_correct_relation (r : Prop) : Prop := r

def relation1 : Prop := a ⊆ M
def relation2 : Prop := M ⊇ {a}
def relation3 : Prop := {a} ∈ M
def relation4 : Prop := 2 * a ∉ M
def relation5 : Prop := {∅} ∈ {a}

def num_correct_relations : ℕ := ({relation2, relation4}.filter is_correct_relation).card

theorem correct_number_of_relations_is_two :
  num_correct_relations = 2 := by sorry

end correct_number_of_relations_is_two_l297_297816


namespace find_monotonic_interval_find_value_l297_297337

noncomputable def f (x : ℝ) := sin (2 * x) - 2 * (cos x) ^ 2

theorem find_monotonic_interval :
  ∀ k : ℤ, -((Real.pi) / 8) + k * Real.pi ≤ x ∧ x ≤ ((3 * Real.pi) / 8) + k * Real.pi →
  (∀ x1 x2 : ℝ, -((Real.pi) / 8) + k * Real.pi ≤ x1 ∧ x1 ≤ ((3 * Real.pi) / 8) + k * Real.pi →
  -((Real.pi) / 8) + k * Real.pi ≤ x2 ∧ x2 ≤ ((3 * Real.pi) / 8) + k * Real.pi → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem find_value (α : ℝ) (k : ℤ) (h : f (α / 2) = Real.sqrt 2 - 1) :
  α = (3 * Real.pi / 4) + 2 * k * Real.pi →
  ( ∃ α : ℝ, α = (3 * Real.pi / 4) + 2 * k * Real.pi ∧ 
    ∀ α : ℝ, α = (3 * Real.pi / 4) + 2 * k * Real.pi → 
    ( (sin (2 * α)) / ((sin α) ^ 4 + (cos α) ^ 4) = -2 )) := sorry

end find_monotonic_interval_find_value_l297_297337


namespace E_n_range_l297_297271

noncomputable def E_n (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (x k) / (x ((k + 1) % n))

theorem E_n_range (n : ℕ) (x : ℕ → ℝ)
  (h_1 : n ≥ 3)
  (h_2 : ∀ i, 1 ≤ x i)
  (h_3 : ∀ i, |x i - x ((i + 1) % n)| ≤ 1) :
  n ≤ E_n x n ∧ E_n x n ≤ ∑ k in finset.range n, (k + 1) / (k + 2) + n :=
  sorry

end E_n_range_l297_297271


namespace solve_xy_l297_297064

theorem solve_xy (x y : ℝ) (hx: x ≠ 0) (hxy: x + y ≠ 0) : 
  (x + y) / x = 2 * y / (x + y) + 1 → (x = y ∨ x = -3 * y) := 
by 
  intros h 
  sorry

end solve_xy_l297_297064


namespace graph_intersect_points_l297_297069

-- Define f as a function defined on all real numbers and invertible
variable (f : ℝ → ℝ) (hf : Function.Injective f)

-- Define the theorem to find the number of intersection points
theorem graph_intersect_points : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (x : ℝ), (f (x^2) = f (x^6)) :=
  by
    -- Outline sketch: We aim to show there are 3 real solutions satisfying the equation
    -- The proof here is skipped, hence we put sorry
    sorry

end graph_intersect_points_l297_297069


namespace travel_remaining_distance_l297_297201

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l297_297201


namespace solve_for_x_l297_297830

theorem solve_for_x (x : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + Complex.i) (h2 : z2 = x + 2 * Complex.i) (h3 : ∃ r : ℝ, z1 * z2 = r) : 
  x = -2 := 
by 
  sorry

end solve_for_x_l297_297830


namespace sophie_owes_jordan_l297_297068

theorem sophie_owes_jordan (dollars_per_window : ℚ) (windows_cleaned : ℚ) : 
  dollars_per_window = 13 / 3 → windows_cleaned = 8 / 5 → 
  let total_owed := dollars_per_window * windows_cleaned in 
  total_owed = 104 / 15 :=
by
  intro h1 h2
  rw [h1, h2]
  unfold total_owed
  sorry

end sophie_owes_jordan_l297_297068


namespace Sam_beats_John_by_7_seconds_l297_297393

variable (sam john speed_john: ℝ)
variable (time_sam distance total_distance distance_ahead : ℝ)
variable (same_time: Prop)

def run_100_m_race (t : ℝ) := total_distance / t

axiom Sam_time_to_run_100m: time_sam = 13 -- Sam completes 100 meters in 13 seconds
axiom Sam_beats_John : sam = run_100_m_race 100
axiom distance: total_distance = 100
axiom distance_ahead: distance_ahead = 35
axiom same_time: run_100_m_race 65 / total_distance = time_sam

theorem Sam_beats_John_by_7_seconds: john - sam = 7 := sorry

end Sam_beats_John_by_7_seconds_l297_297393


namespace repeating_decimal_to_fraction_denominator_is_33_l297_297900

theorem repeating_decimal_to_fraction_denominator_is_33 : 
  let S := 0.15 + (15 : ℝ)/100 * (∑ n, (10 : ℝ)^{-2*n}) in 
  S = (5:ℝ) / 33 :=
begin
  sorry
end

end repeating_decimal_to_fraction_denominator_is_33_l297_297900


namespace fraction_of_succeeding_number_l297_297478

theorem fraction_of_succeeding_number (N : ℝ) (hN : N = 24.000000000000004) :
  ∃ f : ℝ, (1 / 4) * N > f * (N + 1) + 1 ∧ f = 0.2 :=
by
  sorry

end fraction_of_succeeding_number_l297_297478


namespace triangle_angle_contradiction_l297_297134

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α > 60 ∧ β > 60 ∧ γ > 60 → False :=
by
  sorry

end triangle_angle_contradiction_l297_297134


namespace inequality_proof_l297_297855

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / real.sqrt (a^2 + a + 1) ≥ 2 :=
sorry

end inequality_proof_l297_297855


namespace pipe_B_fill_time_l297_297609

theorem pipe_B_fill_time (t : ℕ) : 
  (∀ A B, A = 60 ∧ ∃ t, (30 * (1 / t) + 30 * ((1 / 60) + (1 / t)) = 1)) → t = 40 := by
  sorry

end pipe_B_fill_time_l297_297609


namespace max_value_f_l297_297403

noncomputable def op_add (a b : ℝ) : ℝ :=
if a >= b then a else b^2

noncomputable def f (x : ℝ) : ℝ :=
(op_add 1 x) + (op_add 2 x)

theorem max_value_f :
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, ∀ y ∈ Set.Icc (-2 : ℝ) 3, f y ≤ f x := 
sorry

end max_value_f_l297_297403


namespace total_valid_schedules_l297_297615

def consecutive (A B : ℕ) : Prop := A + 1 = B ∨ B + 1 = A

variables (A B C D : ℕ)
variables (schedules : Finset Finset (Fin 7))

def is_valid_schedule (schedule : Finset (Fin 7)) : Prop :=
  consecutive A B ∧ -- A and B are scheduled on consecutive days
  C ≠ 0 ∧           -- C is not scheduled on New Year's Eve (first day)
  D ≠ 1             -- D is not scheduled on the first day of the lunar year

noncomputable def count_valid_schedules : ℕ := 
  schedules.filter is_valid_schedule |>.card

theorem total_valid_schedules : count_valid_schedules = 1056 :=
sorry

end total_valid_schedules_l297_297615


namespace compare_fractions_l297_297220

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end compare_fractions_l297_297220


namespace part_a_1_part_a_2_part_b_l297_297572

noncomputable def T : ℕ → (ℝ → ℝ) 
| 1 := id
| n := λ x, x * T (n-1) x - (∑ m in range (n-1), U m x * (1 - x ^ 2))

noncomputable def U : ℕ → (ℝ → ℝ) 
| 0 := λ x, 1
| 1 := λ x, 2 * x
| n := λ x, x * U (n-1) x + T n x

theorem part_a_1 (n : ℕ) (φ : ℝ) : cos (n * φ) = (T n) (cos φ) :=
sorry

theorem part_a_2 (n : ℕ) (φ : ℝ) : sin ((n+1) * φ) = U n (cos φ) * sin φ :=
sorry

noncomputable def P : ℕ → (ℝ → ℝ)
| k := λ x, (∑ m in range k, binomial (2 * k + 1) (2 * m + 1) * x ^ (2 * m + 1))

theorem part_b (k : ℕ) (φ : ℝ) : sin ((2 * k + 1) * φ) = sin φ * P k (sin φ * sin φ) :=
sorry

end part_a_1_part_a_2_part_b_l297_297572


namespace max_difference_is_62_l297_297559

open Real

noncomputable def max_difference_of_integers : ℝ :=
  let a (k : ℝ) := 2 * k + 1 + sqrt (8 * k)
  let b (k : ℝ) := 2 * k + 1 - sqrt (8 * k)
  let diff (k : ℝ) := a k - b k
  let max_k := 120 -- Maximum integer value k such that 2k + 1 + sqrt(8k) < 1000
  diff max_k

theorem max_difference_is_62 :
  max_difference_of_integers = 62 :=
sorry

end max_difference_is_62_l297_297559


namespace jessie_interest_l297_297424

noncomputable def compoundInterest 
  (P : ℝ) -- Principal
  (r : ℝ) -- annual interest rate
  (n : ℕ) -- number of times interest applied per time period
  (t : ℝ) -- time periods elapsed
  : ℝ :=
  P * (1 + r / n)^(n * t)

theorem jessie_interest :
  let P := 1200
  let annual_rate := 0.08
  let periods_per_year := 2
  let years := 5
  let A := compoundInterest P annual_rate periods_per_year years
  let interest := A - P
  interest = 576.29 :=
by
  sorry

end jessie_interest_l297_297424


namespace inverse_function_graph_passes_through_point_l297_297469

def has_inverse (f : ℝ → ℝ) := ∃ g : ℝ → ℝ, ∀ x y, g (f x) = x ∧ f (g y) = y

theorem inverse_function_graph_passes_through_point (f : ℝ → ℝ) (hf : has_inverse f) (hpt : (1 - f(1) = 2)) :
  (∃ g : ℝ → ℝ, g (f 1) = -1 ∧ g (-1) - (-1) = 2) :=
sorry

end inverse_function_graph_passes_through_point_l297_297469


namespace angle_CAD_in_convex_quadrilateral_l297_297783

theorem angle_CAD_in_convex_quadrilateral {A B C D : Type} [EuclideanGeometry A B C D]
  (AB_eq_BD : A = B, B = D)
  (angle_A : ∠ A = 65)
  (angle_B : ∠ B = 80)
  (angle_C : ∠ C = 75)
  : ∠ A = 15 :=
by
  sorry

end angle_CAD_in_convex_quadrilateral_l297_297783


namespace find_expression_for_g_l297_297457

variable (f g : ℝ → ℝ)

/-- 
  We are given that f(x) = x^2 + g(x) and f is an even function.
  We need to prove that g(x) = cos(x) is the only even function.
-/
theorem find_expression_for_g (h : ∀ x : ℝ, f x = x^2 + g x) (is_even_f : ∀ x : ℝ, f x = f (-x)) : 
    ∀ x : ℝ, g x = cos x :=
sorry


end find_expression_for_g_l297_297457


namespace count_pos_ints_not_div_by_6_or_8_l297_297093

theorem count_pos_ints_not_div_by_6_or_8 :
  let n := 1200 in
  let count_div_by_6 := (1199 / 6).nat_floor in
  let count_div_by_8 := (1199 / 8).nat_floor in
  let count_div_by_24 := (1199 / 24).nat_floor in
  let total_div_by_6_or_8 := count_div_by_6 + count_div_by_8 - count_div_by_24 in
  let total_pos_ints := n - 1 in
  total_pos_ints - total_div_by_6_or_8 = 900 :=
by
  sorry

end count_pos_ints_not_div_by_6_or_8_l297_297093


namespace angle_CAD_in_convex_quadrilateral_l297_297782

theorem angle_CAD_in_convex_quadrilateral {A B C D : Type} [EuclideanGeometry A B C D]
  (AB_eq_BD : A = B, B = D)
  (angle_A : ∠ A = 65)
  (angle_B : ∠ B = 80)
  (angle_C : ∠ C = 75)
  : ∠ A = 15 :=
by
  sorry

end angle_CAD_in_convex_quadrilateral_l297_297782


namespace farmer_apples_after_giving_l297_297088

-- Define the initial number of apples and the number of apples given to the neighbor
def initial_apples : ℕ := 127
def given_apples : ℕ := 88

-- Define the expected number of apples after giving some away
def remaining_apples : ℕ := 39

-- Formulate the proof problem
theorem farmer_apples_after_giving : initial_apples - given_apples = remaining_apples := by
  sorry

end farmer_apples_after_giving_l297_297088


namespace zeros_before_nonzero_l297_297347

theorem zeros_before_nonzero (x : ℝ) (h : x = 1 / (2^3 * 5^7)) : 
  ∃ (n : ℕ), n = 5 ∧ (∃ a b : ℤ, x = a / 10^7 ∧ a % 10^b = a ∧ a ≠ 0) :=
sorry

end zeros_before_nonzero_l297_297347


namespace fixed_point_on_curve_E_l297_297420

-- Conditions
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9
variables {P Q : ℝ → ℝ → Prop}

-- The trajectory of the center of the moving circle P forms curve E
-- Equation of the ellipse E
def curve_E (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

-- Point M where curve E intersects the positive half of the y-axis
def point_M : Prop := ∃ (x y : ℝ), curve_E x y ∧ x = 0 ∧ y = √3

-- Two distinct points A(x1, y1) and B(x2, y2) on curve E
variables {A B : ℝ × ℝ}
def on_curve_E (a : ℝ × ℝ) : Prop := curve_E a.1 a.2

-- Condition about slopes
def slope_condition (M A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0) ∧ (B.1 ≠ 0) ∧
  let k_MA := (A.2 - M.2) / (A.1 - M.1)
  let k_MB := (B.2 - M.2) / (B.1 - M.1)
  in k_MA * k_MB = 1 / 4

-- Fixed point assertion for line AB
def fixed_point (N : ℝ × ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), 
  on_curve_E A → on_curve_E B →
  slope_condition (0, √3) A B →
  (∃ k m, B.2 = k * B.1 + m ∧ A.2 = k * A.1 + m ∧ (N.1 = 0) ∧ (N.2 = 2 * √3))

-- Lean statement to prove the fixed point property and the curve E equation
theorem fixed_point_on_curve_E :
  (∀ {x y : ℝ}, P x y → circle_F1 x y ∧ circle_F2 x y) →
  (∀ {a : ℝ × ℝ}, on_curve_E a → curve_E a.1 a.2) →
  fixed_point (0, 2 * √3) :=
by
  intros hP hcurve_E
  sorry

end fixed_point_on_curve_E_l297_297420


namespace find_number_l297_297175

theorem find_number (k r n : ℤ) (hk : k = 38) (hr : r = 7) (h : n = 23 * k + r) : n = 881 := 
  by
  sorry

end find_number_l297_297175


namespace intersection_of_M_and_N_l297_297743

noncomputable def M : Set ℝ := {x | x < 2}
noncomputable def N : Set ℝ := {x | 3^x > (1/3)}

theorem intersection_of_M_and_N :
  (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} :=
begin
  sorry
end

end intersection_of_M_and_N_l297_297743


namespace angel_clears_rubbish_l297_297853

-- Structure for the initial setup of the problem
structure Warehouse where
  piles : Fin 100 → Nat -- initially 100 piles, each containing 100 pieces of rubbish

-- Definitions of Angel's and Demon's moves
inductive AngelMove
| clearPile (pile_idx : Fin 100)
| clearOnePieceEach

inductive DemonMove
| addOnePieceEach
| createNewPile

-- Function representing the Angel's score
def angel_score (w : Warehouse) : Int :=
  -1 + (w.piles.toList.map (fun x => min x 2)).sum

-- Function representing the Demon's score
def demon_score (alpha beta : Nat) : Int :=
  if alpha = 0 then 2 * beta - 1 else 2 * beta

-- Predicate asserting the conditions hold and the rubbish is cleared
def cleared (w : Warehouse) : Prop :=
  ∀ i, w.piles i = 0

-- Problem statement
theorem angel_clears_rubbish : ∃ n, n = 199 ∧ ∀ (w : Warehouse),
  warehouse_initial_state w →
  angel_score w > 0 →
  ∀ (angel_day : ℕ → AngelMove) (demon_night : ℕ → DemonMove),
    (angel_score (apply_moves w angel_day demon_night n) = 0 ∧ cleared (apply_moves w angel_day demon_night n)) :=
begin
  sorry -- Proof is skipped
end

end angel_clears_rubbish_l297_297853


namespace price_solution_cost_effectiveness_l297_297587

-- Define the base prices and conditions
variable (u f : ℕ)

-- Given conditions
def condition1 : Prop := u = f + 60
def condition2 : Prop := 3 * u = 5 * f

-- Solution to the price of each uniform and each football
theorem price_solution : (u = 150 ∧ f = 90) :=
by {
  have h1 : u = f + 60 := sorry,
  have h2 : 3 * u = 5 * f := sorry,
  have h3 : u = 150 := sorry,
  have h4 : f = 90 := sorry,
  exact ⟨h3, h4⟩,
}

-- Define cost expressions for Market A and Market B
def market_a_cost (y : ℕ) : ℕ := 90 * y + 14100
def market_b_cost (y : ℕ) : ℕ := 72 * y + 15000

-- Determine the more cost-effective market
theorem cost_effectiveness (y : ℕ) (hy : y > 10):
  (y = 50 → market_a_cost y = market_b_cost y) ∧
  (y < 50 → market_a_cost y < market_b_cost y) ∧
  (y > 50 → market_a_cost y > market_b_cost y) :=
by {
  have ha : market_a_cost 50 = market_b_cost 50 := sorry,
  have hb : ∀ y < 50, market_a_cost y < market_b_cost y := sorry,
  have hc : ∀ y > 50, market_a_cost y > market_b_cost y := sorry,
  exact ⟨ha, hb, hc⟩,
}

end price_solution_cost_effectiveness_l297_297587


namespace monotonically_decreasing_interval_range_of_f_l297_297092

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (abs (x - 1))

theorem monotonically_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1 } := by sorry

end monotonically_decreasing_interval_range_of_f_l297_297092


namespace find_imaginary_part_l297_297287

noncomputable def imaginary_part_of_z (a b : ℝ) (h1: b > 0) (h2: ∀ z : ℂ, z = a+bi → (z.deriv = z ^ 2)) : ℝ :=
b

theorem find_imaginary_part (a b : ℝ) (h1: b > 0)
(h2: ∀ z : ℂ, z = a + bi → (z.deriv = z ^ 2))
(h3: -b = 2 * a * b)
(h4: a = a ^ 2 - b ^ 2) : imaginary_part_of_z a b h1 h2 = sqrt(3)/2 :=
begin
  sorry
end

end find_imaginary_part_l297_297287


namespace conjugate_of_z_l297_297085

theorem conjugate_of_z (z : ℂ) (hz : z = 1 / (1 - I)) : conj z = 1 / 2 - I / 2 :=
sorry

end conjugate_of_z_l297_297085


namespace b1_b2_P6_property_lambda_no_Pk_H2n_minus1_property_l297_297456

-- Define the arithmetic sequence {a_n} with common difference d and general term
def arithmetic_seq (a₀ d : ℝ) (n : ℕ) : ℝ := a₀ + n * d

-- Define the sum T_n of the first n terms of sequence {b_n}
def sum_seq (b : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Define the conditions
noncomputable def d : ℝ := sorry  -- Given d = a_5 = b_2
noncomputable def a_n (n : ℕ) : ℝ := arithmetic_seq (-3/4) (1/4) n  -- Given a_1 = -3/4 and d = 1/4
noncomputable def b_n (n : ℕ) : ℝ := sorry  -- To substitute the defined relationships
noncomputable def T_n (n : ℕ) : ℝ := sum_seq b_n n + 1 / 2^n

-- Property P_k
def P_k (k : ℕ) (x : ℝ) : Prop := a_n (k - 2) < x ∧ x < a_n (k + 3)

-- 1. Determine whether b_1 and b_2 have property P_6
theorem b1_b2_P6_property : ∃ k, k = 6 ∧ P_k k (b_n 1) = false ∧ P_k k (b_n 2) = true := by
  sorry

-- 2. Prove λ does not have property P_k for any k (k >= 3)
theorem lambda_no_Pk (λ : ℝ) (h : ∀ n, ∑ i in finset.range (n + 1), a_n i - 2 * λ * a_n n ≥ (∑ i in finset.range n, a_n i - 2 * λ * a_n i)) : ∀ k : ℕ, k ≥ 3 → ¬ P_k k λ := by
  sorry

-- 3. Find all possible values of k for H_{2n-1} having property P_k
theorem H2n_minus1_property : ∃ (k_values : set ℕ), k_values = {3, 4} ∧ ∀ n k, H_n (2 * n - 1) ∈ k_values → P_k k (H_n (2 * n - 1)) := by
  sorry

end b1_b2_P6_property_lambda_no_Pk_H2n_minus1_property_l297_297456


namespace tan_cos_identity_l297_297638

theorem tan_cos_identity :
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  (tanθ^2 - cosθ^2) / (tanθ^2 * cosθ^2) = -5 / 3 :=
by
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  have h_tan : tanθ^2 = (Real.sin θ)^2 / (Real.cos θ)^2 := by sorry; -- Given condition 1
  have h_cos : cosθ^2 = 3 / 4 := by sorry; -- Given condition 2
  -- Prove the statement
  sorry

end tan_cos_identity_l297_297638


namespace nitin_rank_last_l297_297962

theorem nitin_rank_last (total_students : ℕ) (rank_start : ℕ) (rank_last : ℕ) 
  (h1 : total_students = 58) 
  (h2 : rank_start = 24) 
  (h3 : rank_last = total_students - rank_start + 1) : 
  rank_last = 35 := 
by 
  -- proof can be filled in here
  sorry

end nitin_rank_last_l297_297962


namespace count_ababc_divisible_by_5_l297_297119

theorem count_ababc_divisible_by_5 : 
  let a_vals := {a | 1 ≤ a ∧ a ≤ 9},
      b_vals := {b | 0 ≤ b ∧ b ≤ 9},
      c_vals := {c | c = 0 ∨ c = 5} in
  (∑ a in a_vals, ∑ b in b_vals, ∑ c in c_vals, 1) = 180 := 
by {
  sorry
}

end count_ababc_divisible_by_5_l297_297119


namespace total_cupcakes_baked_l297_297006

-- Conditions
def morning_cupcakes : ℕ := 20
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

-- Goal
theorem total_cupcakes_baked :
  (morning_cupcakes + afternoon_cupcakes) = 55 :=
by
  sorry

end total_cupcakes_baked_l297_297006


namespace work_distribution_l297_297378

def people_to_do_work (people_per_unit : ℕ) (work_units : ℕ) (days : ℕ) : ℕ :=
  (people_per_unit * work_units) / days

theorem work_distribution :
  ∀ (days : ℕ), (days = 3) → people_to_do_work 3 6 days = 6 :=
by
  intros days h1
  rw [h1]
  have h2 : people_to_do_work 3 6 3 = (3 * 6) / 3 := rfl
  rw [h2]
  norm_num
  exact rfl

end work_distribution_l297_297378


namespace monotonicity_of_f_range_of_a_for_g_nonnegative_l297_297288

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * log x

noncomputable def g (x : ℝ) (a : ℝ) := x * log x - a * x + 1

theorem monotonicity_of_f (a : ℝ) (x : ℝ) :
  (a ≤ 0 → 0 < x → deriv (λ x, f x a) x ≥ 0) ∧
  (0 < a → 0 < x ∧ x < sqrt (a / 2) / 2 → deriv (λ x, f x a) x < 0 ∧ 
   sqrt (a / 2) / 2 < x → deriv (λ x, f x a) x ≥ 0) :=
sorry

theorem range_of_a_for_g_nonnegative (a : ℝ) :
  (∀ x : ℝ, 0 < x → g x a ≥ 0) ↔ a ≤ 1 :=
sorry

end monotonicity_of_f_range_of_a_for_g_nonnegative_l297_297288


namespace divide_trapezoid_area_in_half_l297_297610

theorem divide_trapezoid_area_in_half 
    (angle_A_eq_90 : ∠A = 90)
    (AB_eq_30 : AB = 30)
    (AD_eq_20 : AD = 20)
    (DC_eq_45 : DC = 45) :
    ∃ x : ℝ, 20 * x = 375 ∧ x = 18.75 :=
by 
  sorry

end divide_trapezoid_area_in_half_l297_297610


namespace xy_value_l297_297867

theorem xy_value (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
by
  sorry

end xy_value_l297_297867


namespace perimeter_of_rectangle_l297_297870

theorem perimeter_of_rectangle (u v : ℝ) (a b : ℝ)
  (h1 : u * v = 4020)
  (h2 : u + v = 2 * a)
  (h3 : sqrt (u^2 + v^2) = 2 * sqrt (a^2 - b^2))
  (h4 : 4020 * π = π * a * b)
  (h5 : b = sqrt 2010)
  : 2 * (u + v) = 8 * sqrt 2010 :=
by sorry

end perimeter_of_rectangle_l297_297870


namespace harmonic_series_induction_induction_step_additional_terms_l297_297053

theorem harmonic_series_induction (n : ℕ) (h : n > 1) :
  (∑ k in range (2^n - 1), (1 : ℚ) / k.succ) < n :=
sorry

theorem induction_step_additional_terms (k : ℕ) :
  card (range ((2^(k + 1) - 1) - (2^k - 1))) = 2^k :=
sorry

end harmonic_series_induction_induction_step_additional_terms_l297_297053


namespace totalSleepIsThirtyHours_l297_297431

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l297_297431


namespace smallest_x_for_f_984_l297_297170

noncomputable def f : ℝ → ℝ :=
λ x, if (1 ≤ x) ∧ (x ≤ 5) then 1 - |x - 1| else sorry -- define the full function

axiom functional_property : ∀ x > 0, f(3 * x) = 3 * f(x)

theorem smallest_x_for_f_984 : ∃ x, 1 ≤ x ∧ x ≤ 5 ∧ f(x) = f(984) ∧ ∀ y, (1 ≤ y ∧ y ≤ 5 ∧ f(y) = f(984)) → x ≤ y :=
sorry

-- The exact definitions and other details can be filled with the missing parts

end smallest_x_for_f_984_l297_297170


namespace find_n_l297_297259

theorem find_n :
  (∃ n : ℕ, 0 < n ∧ arctan (1 / 2) + arctan (1 / 3) + arctan (1 / 7) + arctan (1 / (n:ℝ)) = π / 4) :=
begin
  use 7,
  split,
  { exact nat.succ_pos', },  -- Ensuring n > 0
  { simp, sorry, },
end

end find_n_l297_297259


namespace cube_root_of_neg_27_l297_297516

theorem cube_root_of_neg_27 : ∃ x : ℝ, x ^ 3 = -27 ∧ x = -3 :=
begin
  use -3,
  split,
  { norm_num },
  { refl }
end

end cube_root_of_neg_27_l297_297516


namespace int_solutions_of_quadratic_l297_297389

theorem int_solutions_of_quadratic (k : ℝ) :
  (∃ x1 x2 : ℝ, 
    (k^2 - 2*k) * x1^2 - (6*k - 4) * x1 + 8 = 0 ∧ 
    (k^2 - 2*k) * x2^2 - (6*k - 4) * x2 + 8 = 0 ∧
    x1 ∈ ℤ ∧ x2 ∈ ℤ) → 
  (k = 0 ∨ k = 2 ∨ k = 1 ∨ k = -2 ∨ k = 2/3) :=
sorry

end int_solutions_of_quadratic_l297_297389


namespace rotate_isosceles_trapezoid_solid_l297_297873

theorem rotate_isosceles_trapezoid_solid (b1 b2 h : ℝ) (hb1 : b1 > b2) :
  (rotate_trapezoid_around_base b1 b2 h).includes_solid (Solid.cylinder) ∧
  (rotate_trapezoid_around_base b1 b2 h).includes_solids (Solid.cone, 2) :=
sorry -- No proof required

end rotate_isosceles_trapezoid_solid_l297_297873


namespace average_side_lengths_of_squares_is_ten_l297_297889

theorem average_side_lengths_of_squares_is_ten (h1 : (side_length : ℕ) → ((side_length = Int.sqrt 25 ∨ side_length = Int.sqrt 64 ∨ side_length = Int.sqrt 144 ∨ side_length = Int.sqrt 225)) : 
  ((Int.sqrt 25 + Int.sqrt 64 + Int.sqrt 144 + Int.sqrt 225) / 4 = 10) :=
by 
  -- h1 provides that the side lengths are calculated correctly.
  -- We need to prove that their average is 10.
  sorry

end average_side_lengths_of_squares_is_ten_l297_297889


namespace count_divisible_by_5_of_ababc_l297_297124

theorem count_divisible_by_5_of_ababc :
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  (a_vals.card * b_vals.card * c_vals.card) = 180 :=
by
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  have h_a_card : a_vals.card = 9 := by simp
  have h_b_card : b_vals.card = 10 := by simp
  have h_c_card : c_vals.card = 2 := by simp
  have total := h_a_card * h_b_card * h_c_card
  show total = 180 from sorry

end count_divisible_by_5_of_ababc_l297_297124


namespace divide_curvilinear_triangle_l297_297575

/-- The side length of the encompassing square is assumed to be 1. -/
def side_length : ℝ := 1

/-- The area of the encompassing square is 1. -/
def area_square : ℝ := side_length * side_length

/-- The area of the curvilinear triangle can be calculated using the formula.
This assumes the area between a curve and a straight line splitting is:
(π * sqrt(2)^2 / 4) / 2. -/
def area_curvilinear_triangle : ℝ := 0.5 * ((π * (real.sqrt 2)^2 / 4) - 1)

/-- To divide the curvilinear triangle into two parts of equal area,
an arc centered at one marked point passing through the other is drawn. -/
theorem divide_curvilinear_triangle :
    ∃ (center marked_point : ℝ×ℝ), 
      arc_division (center, marked_point) area_curvilinear_triangle := 
by
  -- Details are skipped as per the procedure
  sorry

end divide_curvilinear_triangle_l297_297575


namespace binary_101_is_5_l297_297655

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l297_297655


namespace cardinals_count_l297_297216

theorem cardinals_count (C R B S : ℕ) 
  (hR : R = 4 * C)
  (hB : B = 2 * C)
  (hS : S = 3 * C + 1)
  (h_total : C + R + B + S = 31) :
  C = 3 :=
by
  sorry

end cardinals_count_l297_297216


namespace parabola_equation_k1k2_constant_l297_297737

/-
  Given the parabola \( E: x^2 = 2py \) where \( p > 0 \),
  and the line \( y = kx + 2 \) intersects \( E \) at points \( A \) and \( B \),
  and \( \overrightarrow{OA} \cdot \overrightarrow{OB} = 2 \),
  where \( O \) is the origin.

  (1) Prove that the equation of the parabola \( E \) is \(x^2 = y \).

  (2) The coordinates of point \( C \) are \( (0, -2) \).
  Let the slopes of lines \( CA \) and \( CB \) be \( k_1 \) and \( k_2 \), respectively.
  Prove that \( k_1^2 + k_2^2 - 2k^2 \) is a constant.
-/

variables {O A B C : ℝ × ℝ} {k x1 x2 y1 y2 p : ℝ}

-- Conditions
def parabola (x y p : ℝ) : Prop := x^2 = 2 * p * y
def line (x k : ℝ) : ℝ := k * x + 2
def dot_product (O A B : ℝ × ℝ) : ℝ := O.1 * B.1 + O.2 * B.2
def point_C (C : ℝ × ℝ) : Prop := C = (0, -2)

-- 1) Prove the equation of the parabola based on given conditions
theorem parabola_equation (h1 : parabola x y p) (hp_pos : p > 0) 
(h2 : ∃ x1 x2, parabola x1 y1 p ∧ parabola x2 y2 p ∧ dot_product O A B = 2) :
parabola x y (1 / 2) := sorry

-- 2) Prove that \( k_1^2 + k_2^2 - 2k^2 \) is a constant given point C coordinates.
theorem k1k2_constant (h3 : point_C C) (h4 : ∀ A, parabola x1 y1 (1 / 2)) 
(h5 : ∀ B, parabola x2 y2 (1 / 2)) 
(h6 : x1 + x2 = k) (h7 : x1 * x2 = -2) :
(∃ k1 k2, k1 = (y1 + 2) / x1 ∧ k2 = (y2 + 2) / x2 ∧ k1^2 + k2^2 - 2 * k^2 = 16) := sorry

end parabola_equation_k1k2_constant_l297_297737


namespace largest_value_l297_297138

theorem largest_value :
  max (max (max (max (4^2) (4 * 2)) (4 - 2)) (4 / 2)) (4 + 2) = 4^2 :=
by sorry

end largest_value_l297_297138


namespace price_solution_cost_effectiveness_l297_297586

-- Define the base prices and conditions
variable (u f : ℕ)

-- Given conditions
def condition1 : Prop := u = f + 60
def condition2 : Prop := 3 * u = 5 * f

-- Solution to the price of each uniform and each football
theorem price_solution : (u = 150 ∧ f = 90) :=
by {
  have h1 : u = f + 60 := sorry,
  have h2 : 3 * u = 5 * f := sorry,
  have h3 : u = 150 := sorry,
  have h4 : f = 90 := sorry,
  exact ⟨h3, h4⟩,
}

-- Define cost expressions for Market A and Market B
def market_a_cost (y : ℕ) : ℕ := 90 * y + 14100
def market_b_cost (y : ℕ) : ℕ := 72 * y + 15000

-- Determine the more cost-effective market
theorem cost_effectiveness (y : ℕ) (hy : y > 10):
  (y = 50 → market_a_cost y = market_b_cost y) ∧
  (y < 50 → market_a_cost y < market_b_cost y) ∧
  (y > 50 → market_a_cost y > market_b_cost y) :=
by {
  have ha : market_a_cost 50 = market_b_cost 50 := sorry,
  have hb : ∀ y < 50, market_a_cost y < market_b_cost y := sorry,
  have hc : ∀ y > 50, market_a_cost y > market_b_cost y := sorry,
  exact ⟨ha, hb, hc⟩,
}

end price_solution_cost_effectiveness_l297_297586


namespace set_intersection_complement_l297_297837

-- Define the universal set U and sets M, N
def U := {1, 2, 3, 4, 5}
def M := {1, 2, 3}
def N := {2, 5}

-- Define the complement of N with respect to the universal set U
def complement_U_N := U \ N

-- The hypothesis we need is that U, M, N, and the complement of N in U are defined as given above,
-- and we need to prove that M ∩ complement_U_N equals {1, 3}
theorem set_intersection_complement :
  M ∩ complement_U_N = {1, 3} := 
by
  sorry

end set_intersection_complement_l297_297837


namespace orthocenter_on_line_l_l297_297031

-- Definitions and setup
variables {A B C D M N : Type}
variables [plane_geometry A] [plane_geometry B] [plane_geometry C]
variables [plane_geometry D] [plane_geometry M] [plane_geometry N]

-- Points of the quadrilateral and conditions
def quadrilateral_with_incircle (A B C D : Point) : Prop := 
  sorry  -- precise definition needed

def line_through_point (l : Line) (A : Point) : Prop :=
  sorry  -- precise definition needed

def intersects (l : Line) (BC DC : Line) (M N : Point) : Prop :=
  sorry  -- precise definition needed

def incenters (A B C D M N : Point) : Point × Point × Point :=
  sorry  -- precise definition needed

-- Proving the orthocenter lies on the line l
theorem orthocenter_on_line_l (A B C D M N I1 I2 I3 : Point) (l : Line)
  (h1 : quadrilateral_with_incircle A B C D)
  (h2 : line_through_point l A)
  (h3 : intersects l BC DC M N)
  (h4 : incenters A B C D M N = (I1, I2, I3)) :
  orthocenter_triangle I1 I2 I3 ∈ l :=
sorry

end orthocenter_on_line_l_l297_297031


namespace polygon_sides_l297_297598

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1680 + a) (a < 180 ∧ 0 < a) : n = 12 :=
sorry

end polygon_sides_l297_297598


namespace angle_CAD_eq_15_l297_297778

-- Define the angles and equality of sides
variables (A B C D : Type) [IsConvexQuadrilateral A B C D]
           (angle_A : ∠A = 65)
           (angle_B : ∠B = 80)
           (angle_C : ∠C = 75)
           (AB_eq_BD : AB = BD)
           
-- State the theorem to prove the measure of ∠CAD
theorem angle_CAD_eq_15 : ∠CAD = 15 :=
sorry

end angle_CAD_eq_15_l297_297778


namespace cube_root_of_neg_27_l297_297515

theorem cube_root_of_neg_27 : ∃ x : ℝ, x ^ 3 = -27 ∧ x = -3 :=
begin
  use -3,
  split,
  { norm_num },
  { refl }
end

end cube_root_of_neg_27_l297_297515


namespace binary_addition_to_decimal_l297_297943

theorem binary_addition_to_decimal : (0b111111111 + 0b1000001 = 576) :=
by {
  sorry
}

end binary_addition_to_decimal_l297_297943


namespace min_moves_to_zero_l297_297930

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l297_297930


namespace find_theta_l297_297680

def is_valid_digit (n : ℕ) : Prop :=
  n <= 9

theorem find_theta :
  ∃ (Θ : ℕ), is_valid_digit Θ ∧ 
             (476 / Θ = 50 + Θ + 3 * Θ) ∧ 
             Θ = 6 := 
begin
  sorry
end

end find_theta_l297_297680


namespace negation_of_p_l297_297739
open Classical

variable (n : ℕ)

def p : Prop := ∀ n : ℕ, n^2 < 2^n

theorem negation_of_p : ¬ p ↔ ∃ n₀ : ℕ, n₀^2 ≥ 2^n₀ := 
by
  sorry

end negation_of_p_l297_297739


namespace testing_methods_first_problem_testing_methods_second_problem_l297_297322

open_locale big_operators

-- Definition of the problem and its constraints:
def products := range 8
def defective_products := 3

-- Question 1 condition:
def first_defective_on_second_test := true
def last_defective_on_sixth_test := true

-- Question 2 condition:
def at_most_five_tests := true

-- Lean statement:

-- The first proof problem:
theorem testing_methods_first_problem (products : Finset ℕ) (defective_products : ℕ)
  (first_defective_on_second_test : Bool) (last_defective_on_sixth_test : Bool) :
  first_defective_on_second_test = true →
  last_defective_on_sixth_test = true →
  products.card = 8 →
  defective_products = 3 →
  -- The number of distinct testing methods is 1080
  (∑ p in products.powerset, p.card) = 1080 :=
sorry

-- The second proof problem:
theorem testing_methods_second_problem (products : Finset ℕ) (defective_products : ℕ)
  (at_most_five_tests : Bool) :
  at_most_five_tests = true →
  products.card = 8 →
  defective_products = 3 →
  -- The number of distinct testing methods is 936
  (∑ p in products.powerset, p.card) = 936 :=
sorry

end testing_methods_first_problem_testing_methods_second_problem_l297_297322


namespace range_of_x_l297_297032

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x + 1 else 2^x

theorem range_of_x (x : ℝ) (h : f x + f (x - 1/2) > 1) : x > -1/4 :=
begin
  sorry
end

end range_of_x_l297_297032


namespace exists_polygon_without_center_of_symmetry_that_can_be_divided_l297_297668

theorem exists_polygon_without_center_of_symmetry_that_can_be_divided :
  ∃ P (P1 P2 : Set (ℝ × ℝ)),
    (¬∃ C1, P.SymmetricCenter = some C1) ∧
    (Convex P) ∧
    (Convex P1) ∧
    (Convex P2) ∧
    (∃ C2 : ℝ × ℝ, P1.SymmetricCenter = some C2) ∧
    (∃ C3 : ℝ × ℝ, P2.SymmetricCenter = some C3) ∧
    (P = P1 ∪ P2) :=
by
  sorry

end exists_polygon_without_center_of_symmetry_that_can_be_divided_l297_297668


namespace butterflies_in_the_garden_l297_297671

variable (total_butterflies : Nat) (fly_away : Nat)

def butterflies_left (total_butterflies : Nat) (fly_away : Nat) : Nat :=
  total_butterflies - fly_away

theorem butterflies_in_the_garden :
  (total_butterflies = 9) → (fly_away = 1 / 3 * total_butterflies) → butterflies_left total_butterflies fly_away = 6 :=
by
  intro h1 h2
  sorry

end butterflies_in_the_garden_l297_297671


namespace train_speed_l297_297998

theorem train_speed (length_of_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_of_train = 500) (h_time : time_to_cross = 20) : 
  length_of_train / time_to_cross = 25 :=
by
  rw [h_length, h_time]
  norm_num
  sorry

end train_speed_l297_297998


namespace frank_can_buy_seven_candies_l297_297566

def tickets_won_whackamole := 33
def tickets_won_skeeball := 9
def cost_per_candy := 6

theorem frank_can_buy_seven_candies : (tickets_won_whackamole + tickets_won_skeeball) / cost_per_candy = 7 :=
by
  sorry

end frank_can_buy_seven_candies_l297_297566


namespace sufficient_not_necessary_l297_297561

theorem sufficient_not_necessary (x : ℝ) : abs x < 2 → (x^2 - x - 6 < 0) ∧ (¬(x^2 - x - 6 < 0) → abs x ≥ 2) :=
by
  sorry

end sufficient_not_necessary_l297_297561


namespace question1_question2_l297_297341

open Real

def f (x b : ℝ) := (2 * x + b) * exp x
def F (x b : ℝ) := b * x - log x
def g (x b : ℝ) := b * x^2 - 2 * x - (b * x - log x)

noncomputable def condition1 (b : ℝ) : Prop :=
  b < 0 ∧ ∃ (M : set ℝ), M ⊆ (⋂ x, {y | differentiable_at ℝ (f x b) y ↔ differentiable_at ℝ (F y b) y}) ∧ 
    ((∀ x ∈ M, deriv (f x b) x = 0) ↔ (∀ x ∈ M, deriv (F x b) x = 0))

noncomputable def condition2 (b : ℝ) : Prop :=
  b > 0 ∧
    ∀ x ∈ Icc 1 (exp 1), g (Icc 1 (exp 1)) b ≥ -2

theorem question1 (b : ℝ) (h : condition1 b) : b < -2 :=
sorry

theorem question2 (b : ℝ) (h : condition2 b) : (0 < b ∧ b ≤ 1 / exp 1) ∨ (1 ≤ b) :=
sorry

end question1_question2_l297_297341


namespace range_of_function_l297_297526

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (floor (Real.sin x).toℝ) + Real.arccos (floor (Real.cos x).toℝ)

theorem range_of_function :
  {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi ∧ f x = y} = {0, Real.pi / 2, Real.pi} :=
sorry

end range_of_function_l297_297526


namespace max_product_centroid_l297_297004

/-
Inside the triangle \(ABC\), a point \(O\) is taken. Let \(d_a, d_b, d_c\) be the distances from point \(O\) to the lines \(BC\), \(CA\), and \(AB\). The product \(d_a d_b d_c\) is maximal when \(O\) is the centroid of \(ABC\).
-/

noncomputable def distances_max_product (A B C O : Point) : Prop :=
  let d_a := distance_to_line O (line B C)
  let d_b := distance_to_line O (line C A)
  let d_c := distance_to_line O (line A B)
  let centroid := centroid_of_triangle A B C
  O = centroid →
  ∀ O', let d'_a := distance_to_line O' (line B C) in
        let d'_b := distance_to_line O' (line C A) in
        let d'_c := distance_to_line O' (line A B) in
        d_a * d_b * d_c ≥ d'_a * d'_b * d'_c

-- Now define the main theorem statement

theorem max_product_centroid (A B C : Point) :
  ∃ O : Point, distances_max_product A B C O :=
by
  sorry

end max_product_centroid_l297_297004


namespace total_carrots_l297_297495

theorem total_carrots (Sandy_carrots Sam_carrots Sarah_carrots : ℕ)
  (hSandy : Sandy_carrots = 6)
  (hSam : Sam_carrots = 3)
  (hSarah : Sarah_carrots = 5) :
  Sandy_carrots + Sam_carrots + Sarah_carrots = 14 :=
by {
  have hs : Sandy_carrots = 6 := hSandy,
  have hsam : Sam_carrots = 3 := hSam,
  have hsarah : Sarah_carrots = 5 := hSarah,
  let total := Sandy_carrots + Sam_carrots + Sarah_carrots,
  calc
    total = 6 + 3 + 5 : by rw [hs, hsam, hsarah]
    ... = 14 : by norm_num,
}

end total_carrots_l297_297495


namespace avg_income_pr_l297_297890

theorem avg_income_pr (P Q R : ℝ) 
  (h_avgPQ : (P + Q) / 2 = 5050) 
  (h_avgQR : (Q + R) / 2 = 6250)
  (h_P : P = 4000) 
  : (P + R) / 2 = 5200 := 
by 
  sorry

end avg_income_pr_l297_297890


namespace sequence_product_l297_297488

open Nat

-- Define the sequence a_n
def a : ℕ → ℚ
| 0     := 3 / 4
| (n+1) := 2 + (a n - 1)^2

-- The target statement to prove
theorem sequence_product : (inf.prod (λ n, a n)) = 4 / 5 :=
sorry

end sequence_product_l297_297488


namespace garden_width_l297_297995

theorem garden_width :
  ∃ w : ℝ, 3 * w ^ 2 + 16 * w - 959 = 0 ∧ w > 0 := 
by
  -- The approach uses the quadratic formula
  let Δ := 16 ^ 2 + 4 * 3 * 959
  have hΔ : Δ = 11800, by norm_num
  let sqrt_Δ := Real.sqrt Δ
  have hsqrt_Δ : sqrt_Δ = 108.66025403784438, by norm_num
  let w_pos := (-16 + sqrt_Δ) / (2 * 3)
  have hw_pos : w_pos = 15.443375728224064, by norm_num
  exists w_pos
  split
  norm_num [Real.sqrt_eq_rpow]
  exact rfl
  linarith

end garden_width_l297_297995


namespace sequence_bound_l297_297343

theorem sequence_bound (g : ℕ+ → ℝ)
  (h₁ : g 1 = 3)
  (h₂ : ∀ n : ℕ+, 4 * g (n + 1) - 3 * g n < 2)
  (h₃ : ∀ n : ℕ+, 2 * g (n + 1) - g n > 2) :
  ∀ n : ℕ+, 2 + (1/2 : ℝ)^n < g (n + 1) ∧ g (n + 1) < 2 + (3/4 : ℝ)^n :=
sorry

end sequence_bound_l297_297343


namespace alice_champion_probability_l297_297199

theorem alice_champion_probability :
  ∀ (players : Fin 8 → ℕ), -- 8 players
  (players 0 = 1) → -- Alice plays rock
  (players 1 = 2) → -- Bob plays paper
  (∀ i, (2 ≤ i) → (i < 8) → (players i = 3)) → -- Others play scissors
  (∀ (game_result : Fin 8 → Fin 8 → ℕ), -- Game results
  (∀ i j, (players i = 1) → (players j = 3) → (game_result i j = 1)) → -- Rock beats Scissors
  (∀ i j, (players i = 3) → (players j = 2) → (game_result i j = 1)) → -- Scissors beat Paper
  (∀ i j, (players i = 2) → (players j = 1) → (game_result i j = 1)) → -- Paper beats Rock
  (∀ i j, (players i = players j) → (0 ≤ game_result i j) ∧ (game_result i j ≤ 1)) → -- Tie handled by coin flip
  (optimally_random_pairing : (ℕ → option (Fin 8 × Fin 8))) → -- Random pairing for each round
  ∀ rnd_result, -- For any round results
  (∃ (prob_win_alice : ℝ), prob_win_alice = (6 / 7)) :=
begin
  sorry
end

end alice_champion_probability_l297_297199


namespace alice_can_win_l297_297915

-- Definitions given in conditions
def buns_with_jam : ℕ := 20
def buns_with_treacle : ℕ := 20
def total_buns : ℕ := buns_with_jam + buns_with_treacle

-- Alice and Bob's objective and strategy
def alice_wins (order : list bool) : Prop :=
  ∃ (take_bun_from_end : ℕ → bool), -- A function defining Alice's choice strategy
    (∀ (choices : list bool), 
      (list.length choices = total_buns) ∧
        (alice_collects order take_bun_from_end choices)) -- Definition to describe Alice's collection strategy across multiple rounds.
        
noncomputable def alice_collects (order : list bool) (take_bun_from_end : ℕ → bool) (choices : list bool) : Prop :=
  -- Alice collection criteria to be defined
  sorry

-- The final theorem to prove
theorem alice_can_win (order : list bool) : alice_wins order :=
  sorry

end alice_can_win_l297_297915


namespace area_of_rectangle_abcd_l297_297224

-- Definition of the problem's conditions and question
def small_square_side_length : ℝ := 1
def large_square_side_length : ℝ := 1.5
def area_rectangle_abc : ℝ := 4.5

-- Lean 4 statement: Prove the area of rectangle ABCD is 4.5 square inches
theorem area_of_rectangle_abcd :
  (3 * small_square_side_length) * large_square_side_length = area_rectangle_abc :=
by
  sorry

end area_of_rectangle_abcd_l297_297224


namespace distance_M_to_line_AB_l297_297978

theorem distance_M_to_line_AB (M A B : Point) (line circle : set Point) 
  (a b : ℝ) (ha : dist M A = a) (hb : dist M B = b) :
  tangent_touch M line circle ∧ perpendicular_drop A line a ∧ perpendicular_drop B line b →
  ∃ x : ℝ, x = sqrt (a * b) :=
by
  sorry

end distance_M_to_line_AB_l297_297978


namespace sum_of_extreme_values_of_c_l297_297708

variables (a b c : ℝ × ℝ)

-- Conditions
def condition1 : |a| = 1 :=
sorry

def condition2 : |b| = 1 :=
sorry

def condition3 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0 :=
sorry

def condition4 : (c.1 - 2 * a.1) * (c.1 - b.1) + (c.2 - 2 * a.2) * (c.2 - b.2) = 0 :=
sorry

def sum_of_extreme_values_norm_c : ℝ :=
2 * real.sqrt ((5/4)^2 + (real.sqrt 3 / 4)^2)

-- Main statement
theorem sum_of_extreme_values_of_c :
  ∀ (a b c : ℝ × ℝ), (|a| = 1) →
  (|b| = 1) →
  (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) →
  ((c.1 - 2 * a.1) * (c.1 - b.1) + (c.2 - 2 * a.2) * (c.2 - b.2) = 0) →
  sum_of_extreme_values_norm_c = real.sqrt 7 :=
begin
  intros,
  sorry
end

end sum_of_extreme_values_of_c_l297_297708


namespace angle_sum_l297_297460

-- Definitions based on given conditions
variables (n : ℕ) (hn : 3 ≤ n) -- \(n\)-sided polygon, n>=3
variables (A : fin n → ℝ × ℝ) -- coordinates of vertices A₁, A₂, ..., Aₙ

-- A regular n-gon
def is_regular (A : fin n → ℝ × ℝ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → dist (A ⟨i, sorry⟩) (A ⟨j, sorry⟩) = dist (A ⟨0, sorry⟩) (A ⟨1, sorry⟩)

-- B₁ is the midpoint of A₁A₂ and Bₙ₋₁ is the midpoint of Aₙ₋₁Aₙ
def B (i : ℕ) (A : fin n → ℝ × ℝ) : ℝ × ℝ :=
  if i = 0 then midpoint (A ⟨0, sorry⟩) (A ⟨1, sorry⟩) else
  if i = n-1 then midpoint (A ⟨n-2, sorry⟩) (A ⟨n-1, sorry⟩) else
  sorry

-- To prove the sum of specific angles is equal to 180°
theorem angle_sum (A : fin n → ℝ × ℝ) :
  is_regular A → 
  ∑ i in finset.range (n-1), angle (A 0) (B i A) (A (n-1)) = 180 :=
  sorry

end angle_sum_l297_297460


namespace odd_and_even_derivative_behavior_l297_297314

theorem odd_and_even_derivative_behavior 
  (f g : ℝ → ℝ)
  (Hf_odd : ∀ x, f (-x) = -f x)
  (Hg_even : ∀ x, g (-x) = g x)
  (H_f_deriv_neg : ∀ x, x < 0 → deriv f x > 0)
  (H_g_deriv_neg : ∀ x, x < 0 → deriv g x < 0) :
  (∀ x, 0 < x → deriv f x > 0) ∧ (∀ x, 0 < x → deriv g x > 0) := 
sorry

end odd_and_even_derivative_behavior_l297_297314


namespace largest_inscribed_triangle_area_l297_297637

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 12) : ∃ A : ℝ, A = 144 :=
by
  sorry

end largest_inscribed_triangle_area_l297_297637


namespace probability_at_least_one_die_shows_three_l297_297550

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l297_297550


namespace min_cos_A_l297_297391

theorem min_cos_A (a b c : ℝ) (h : b^2 + c^2 = 2 * a^2) : 
  ∃ (eps : unit), (∀ b c (hₛq : b = c), cos_arccos (b^2 + c^2 - a^2) / (2 * b * c) ≥ 1/2) := sorry

end min_cos_A_l297_297391


namespace relative_speed_is_4_point_44_l297_297552

noncomputable def speed_A := 56 -- Speed in kmph
noncomputable def speed_B := 72 -- Speed in kmph

def kmph_to_mps (speed : ℕ) : ℝ := speed * 1000 / 3600

theorem relative_speed_is_4_point_44 :
  kmph_to_mps (speed_B - speed_A) = 4.44 :=
by
  sorry

end relative_speed_is_4_point_44_l297_297552


namespace range_of_b_l297_297731

def f (x : ℝ) : ℝ :=
  abs (x * Real.exp (x + 1))

theorem range_of_b (b : ℝ) :
  (∃! x : ℝ, (f x)^2 + b * f x + 2 = 0) → b < -3 :=
sorry

end range_of_b_l297_297731


namespace sum_of_intercepts_l297_297531

theorem sum_of_intercepts (x y : ℝ) (h : x / 3 - y / 4 = 1) : (x / 3 = 1 ∧ y / (-4) = 1) → 3 + (-4) = -1 :=
by
  sorry

end sum_of_intercepts_l297_297531


namespace find_angle_CAD_l297_297770

-- Definitions representing the problem conditions
variables {A B C D : Type} [has_angle A] [has_angle B] [has_angle C] [has_angle D]

-- Given conditions
def is_convex (A B C D : Type) : Prop := sorry  -- convex quadrilateral ABCD
def angle_A (A : Type) : ℝ := 65
def angle_B (B : Type) : ℝ := 80
def angle_C (C : Type) : ℝ := 75
def equal_sides (A B D : Type) : Prop := sorry  -- AB = BD

-- Theorem statement: Given the conditions, prove the desired angle
theorem find_angle_CAD {A B C D : Type}
  [is_convex A B C D] 
  [equal_sides A B D]
  (h1 : angle_A A = 65)
  (h2 : angle_B B = 80)
  (h3 : angle_C C = 75)
  : ∃ (CAD : ℝ), CAD = 15 := 
sorry -- proof omitted

end find_angle_CAD_l297_297770


namespace apples_left_over_l297_297838

-- Defining the number of apples collected by Liam, Mia, and Noah
def liam_apples := 53
def mia_apples := 68
def noah_apples := 22

-- The total number of apples collected
def total_apples := liam_apples + mia_apples + noah_apples

-- Proving that the remainder when the total number of apples is divided by 10 is 3
theorem apples_left_over : total_apples % 10 = 3 := by
  -- Placeholder for proof
  sorry

end apples_left_over_l297_297838


namespace find_a_for_parallel_lines_l297_297344

-- Define the lines l1 and l2
def line1 (a : ℝ) := λ x y : ℝ, a * x - y + a = 0
def line2 (a : ℝ) := λ x y : ℝ, (2 * a - 3) * x + a * y - a = 0

-- Define what it means for two lines to be parallel
def parallel_lines (l1 l2 : ℝ → ℝ → Prop) := ∀ x1 y1 x2 y2, l1 x1 y1 → l2 x2 y2 → (x1 * y2 - x2 * y1 = 0)

-- The main theorem statement
theorem find_a_for_parallel_lines (a : ℝ) : 
  parallel_lines (line1 a) (line2 a) → a = -3 :=
by
  sorry

end find_a_for_parallel_lines_l297_297344


namespace range_of_a_l297_297387

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 8 ∧ a ≠ 4) ↔
  (a > 1 ∧ a < 8) ∧ (a > -4 ∧ a ≠ 4) :=
by sorry

end range_of_a_l297_297387


namespace solve_equation_l297_297677

theorem solve_equation (x : ℝ) : 
  (sqrt (sqrt x) = 12 / (7 - sqrt (sqrt x))) ↔ (x = 81 ∨ x = 256) :=
by
  sorry

end solve_equation_l297_297677


namespace angle_D_measure_l297_297470

variable (p q r s : Line)
variable (A B C D : Angle)

-- Conditions
variable (parallel_pq : p ∥ q)
variable (intersects_r_pq : r ∩ p = some A ∧ r ∩ q = some B)
variable (angle_A_B : A = B / 10)
variable (intersects_s_pq : s ∩ p = some C ∧ s ∩ q = some D)
variable (angle_C_A : C = A)

theorem angle_D_measure : D = 90 := by
  sorry

end angle_D_measure_l297_297470


namespace find_least_m_l297_297233

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 7 else (sequence (n-1))^2 + 7 * (sequence (n-1)) + 12 / (sequence (n-1)) + 8

theorem find_least_m :
  ∃ m : ℕ, m > 0 ∧ sequence m ≤ 5 + 1 / 2^15 ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end find_least_m_l297_297233


namespace angle_CAD_eq_15_l297_297781

-- Define the angles and equality of sides
variables (A B C D : Type) [IsConvexQuadrilateral A B C D]
           (angle_A : ∠A = 65)
           (angle_B : ∠B = 80)
           (angle_C : ∠C = 75)
           (AB_eq_BD : AB = BD)
           
-- State the theorem to prove the measure of ∠CAD
theorem angle_CAD_eq_15 : ∠CAD = 15 :=
sorry

end angle_CAD_eq_15_l297_297781


namespace perimeter_of_figure_l297_297965

theorem perimeter_of_figure :
  let sides := [1, 1/2, 1/4, 1/8, 1/16, 1/32]
  let segments := [2 * sides[0], 2 * sides[1], 2 * sides[2], 2 * sides[3], 2 * sides[4], 3 * sides[5]]
  let perimeter := ∑ i in segments, i
  perimeter = 127 / 32 :=
by
  sorry

end perimeter_of_figure_l297_297965


namespace kyle_sales_money_proof_l297_297442

variable (initial_cookies initial_brownies : Nat)
variable (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
variable (price_per_cookie price_per_brownie : Float)

def kyle_total_money (initial_cookies initial_brownies : Nat) 
    (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
    (price_per_cookie price_per_brownie : Float) : Float := 
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies)
  let remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies)
  let money_from_cookies := remaining_cookies * price_per_cookie
  let money_from_brownies := remaining_brownies * price_per_brownie
  money_from_cookies + money_from_brownies

theorem kyle_sales_money_proof : 
  kyle_total_money 60 32 2 1 2 2 1 1.50 = 99 :=
by
  sorry

end kyle_sales_money_proof_l297_297442


namespace tan_C_in_triangle_l297_297414

theorem tan_C_in_triangle (A B C : ℝ) (hA : Real.tan A = 1 / 2) (hB : Real.cos B = 3 * Real.sqrt 10 / 10) :
  Real.tan C = -1 :=
sorry

end tan_C_in_triangle_l297_297414


namespace car_speed_l297_297584

theorem car_speed (dist_sound : ℝ) (time_duration : ℝ) (speed_sound : ℝ)
  (h1 : dist_sound = 1200)
  (h2 : time_duration = 3.9669421487603307)
  (h3 : speed_sound = 330) :
  let v_c := (speed_sound * time_duration - dist_sound) / time_duration
  in v_c = 27.5 :=
by
  intros
  sorry

end car_speed_l297_297584


namespace volume_of_region_truncated_pyramid_l297_297268

noncomputable def frustum_volume (B1 B2 h : ℝ) : ℝ :=
  (h / 3) * (B1 + real.sqrt (B1 * B2) + B2)

theorem volume_of_region_truncated_pyramid : 
  ∃ (B1 B2 h : ℝ), 
  (∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ |x + 2 * y + z| + |x - y - z| ≤ 10 → 
  max (x + 2 * y + z) (x - y - z) ≤ 5) → 
  ∃ V : ℝ, V = frustum_volume B1 B2 h :=
begin
  sorry
end

end volume_of_region_truncated_pyramid_l297_297268


namespace james_transport_capacity_l297_297007

theorem james_transport_capacity :
  ∃ (vans : Nat → ℕ), 
  vans 1 = 8000 ∧ vans 2 = 8000 ∧ 
  vans 3 = 8000 - 2400 ∧
  (∀ i, 4 ≤ i ∧ i ≤ 6 → vans i = 8000 + 4000) ∧
  (∑ i in finset.range 1 7, vans i) = 57600
:= sorry

end james_transport_capacity_l297_297007


namespace inequality_l297_297465

noncomputable def sum_natural (n : ℕ) (a : ℕ → ℝ) : ℝ :=
( finset.range n ).sum (λ i, a i)

noncomputable def sum_pairs (n : ℕ) (a : ℕ → ℝ) : ℝ :=
( finset.range n ).sum (λ i, ( finset.range i ).sum (λ j, a i * a j))

theorem inequality (n : ℕ) (a : ℕ → ℝ) (h_n : 3 ≤ n) (h_a : ∀ i, 0 < a i) :
  (sum_natural n a) * (sum_pairs n (λ i, ( sum_pairs n a ) / (a i + a i) )) ≤ (( n : ℝ) / 2) * sum_pairs n a :=
    sorry

end inequality_l297_297465


namespace product_of_midpoint_l297_297684

-- Define the coordinates of the endpoints
def x1 := 5
def y1 := -4
def x2 := 1
def y2 := 14

-- Define the formulas for the midpoint coordinates
def xm := (x1 + x2) / 2
def ym := (y1 + y2) / 2

-- Define the product of the midpoint coordinates
def product := xm * ym

-- Now state the theorem
theorem product_of_midpoint :
  product = 15 := 
by
  -- Optional: detailed steps can go here if necessary
  sorry

end product_of_midpoint_l297_297684


namespace odd_subset_sum_divisible_by_5_l297_297351

theorem odd_subset_sum_divisible_by_5 : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (subsets: Finset (Finset ℕ)), 
    (∀ A ∈ subsets, A ⊆ {1, 3, 5, 7, 9} ∧ 
                    A ≠ ∅ ∧ 
                    (∑ x in A, x) % 5 = 0) ∧ 
    subsets.card = 4 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  -- Given the odd numbers from the set S
  let odd_numbers := {1, 3, 5, 7, 9}
  -- Construct the set of valid subsets
  let valid_subsets := { {5}, {1, 9}, {3, 7}, {1, 3, 5, 7, 9} }
  existsi valid_subsets
  split
  {
    intros A hA
    split
    {
      -- A is a subset of the odd numbers
      dsimp at hA
      exact Finset.subset.trans hA <| Finset.subset_univ _
    }
    split
    {
      -- A is non-empty
      dsimp [valid_subsets] at hA
      cases hA <;> simp
    }
    {
      -- The sum of elements of A is divisible by 5
      dsimp [valid_subsets] at hA
      cases hA
      repeat_case { {5} }
        { simp }
      repeat_case { {1, 9} }
        { simp }
      repeat_case { {3, 7} }
        { simp }
      repeat_case { {1, 3, 5, 7, 9} }
        { simp }
    }
  }
  -- The cardinality of the set of valid subsets is 4
  dsimp [valid_subsets]
  simp
  done
  sorry

end odd_subset_sum_divisible_by_5_l297_297351


namespace binary_101_to_decimal_l297_297648

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l297_297648


namespace find_omega_value_l297_297336

noncomputable def possible_value_of_omega (ω : ℕ) : Prop :=
  (ω > 0 ∧ ω ≤ 12) ∧
  (∃ φ : ℝ, 0 < φ ∧ φ < π ∧
  (∀ x : ℝ, sin (ω * x + φ) = -sin (ω * -x + φ)) ∧
  ¬monotone (λ x : ℝ, sin (ω * x + φ)) (Icc (π / 4) (π / 2)))

theorem find_omega_value : possible_value_of_omega 9 :=
by
  sorry

end find_omega_value_l297_297336


namespace totalSleepIsThirtyHours_l297_297430

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l297_297430


namespace green_tiles_in_50th_row_l297_297395

-- Conditions
def tiles_in_row (n : ℕ) : ℕ := 2 * n - 1

def green_tiles_in_row (n : ℕ) : ℕ := (tiles_in_row n - 1) / 2

-- Prove the number of green tiles in the 50th row
theorem green_tiles_in_50th_row : green_tiles_in_row 50 = 49 :=
by
  -- Placeholder proof
  sorry

end green_tiles_in_50th_row_l297_297395


namespace correct_solutions_l297_297951

noncomputable def sample_size : ℕ := 5
noncomputable def population_size : ℕ := 50
noncomputable def prob_individual_selected : ℝ := 0.1

def data_set : List ℝ := [10, 11, 11, 12, 13, 14, 16, 18, 20, 22]

noncomputable def transformed_variance : ℝ := 8
noncomputable def original_variance : ℝ := 2

noncomputable def strata_mean_1 : ℝ := 0  -- placeholder, as it equals strata_mean_2
noncomputable def strata_mean_2 : ℝ := strata_mean_1
noncomputable def strata_var_1 : ℝ := 0  -- needs an appropriate value
noncomputable def strata_var_2 : ℝ := 0  -- needs an appropriate value
noncomputable def population_variance : ℝ := 1 / 2 * (strata_var_1 + strata_var_2)

theorem correct_solutions : 
  (prob_individual_selected = (sample_size:ℝ) / (population_size:ℝ)) ∧
  (list.nth_le data_set 5 sorry = 14 → list.nth_le data_set 6 sorry = 16 → 
   (list.nth_le data_set 5 sorry + list.nth_le data_set 6 sorry) / 2 ≠ 15) ∧
  (transformed_variance = 4 * original_variance) ∧
  population_variance ≠ 1 / 2 * (strata_var_1 + strata_var_2) →
  "{A, C}" = "{correct_solutions}"
:= sorry

end correct_solutions_l297_297951


namespace relationship_abc_l297_297693

noncomputable def a : ℝ := 4 / 5
noncomputable def b : ℝ := Real.sin (2 / 3)
noncomputable def c : ℝ := Real.cos (1 / 3)

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l297_297693


namespace alpha_plus_2beta_l297_297021

open Real Trigonometry

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1) 
    (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) : α + 2 * β = π / 2 := 
sorry

end alpha_plus_2beta_l297_297021


namespace complex_magnitude_problem_l297_297459

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : complex.abs z = 2)
  (hw : complex.abs w = 4)
  (hz_w : complex.abs (z + w) = 5) :
  complex.abs ((1 / z) + (1 / w)) = 5 / 8 := by
  sorry

end complex_magnitude_problem_l297_297459


namespace coefficient_x2_in_expansion_l297_297084

theorem coefficient_x2_in_expansion : 
  let T (r : ℕ) : ℚ := (nat.choose 6 r : ℚ) * (2^(6-r)) * ((-1/2)^r) 
  in T 4 = 15 / 4 :=
by {
  have T_def : ∀ r : ℕ, 
    T r = (nat.choose 6 r : ℚ) * (2^(6-r)) * ((-1/2)^r) := 
    λ r, by rfl,
  sorry
}

end coefficient_x2_in_expansion_l297_297084


namespace jessica_seashells_l297_297874

theorem jessica_seashells (sally tom jessica total : ℕ)
  (h_sally : sally = 9)
  (h_tom : tom = 7)
  (h_total : total = 21) :
  jessica = total - (sally + tom) :=
by
  have h := h_total ▸ h_sally ▸ h_tom ▸ rfl
  have r : total - (sally + tom) = 5 := by
    simp [h_sally, h_tom, h_total]
  exact r

end jessica_seashells_l297_297874


namespace log_product_evaluation_l297_297673

noncomputable def evaluate_log_product : ℝ :=
  Real.log 9 / Real.log 2 * Real.log 16 / Real.log 3 * Real.log 27 / Real.log 7

theorem log_product_evaluation : evaluate_log_product = 24 := 
  sorry

end log_product_evaluation_l297_297673


namespace angle_CAD_in_convex_quadrilateral_l297_297775

theorem angle_CAD_in_convex_quadrilateral
  (ABCD : Type)
  [convex_quadrilateral ABCD]
  (A B C D : ABCD)
  (h1 : AB = BD)
  (h2 : ∠A = 65)
  (h3 : ∠B = 80)
  (h4 : ∠C = 75)
  : ∠CAD = 15 := sorry

end angle_CAD_in_convex_quadrilateral_l297_297775


namespace rhombus_properties_l297_297727

-- Definitions of the conditions given in the problem
def equation_AB (x y : ℝ) : Prop := x - 3 * y + 10 = 0
def equation_diagonal (x y : ℝ) : Prop := x + 4 * y - 4 = 0
def intersection_point (P : ℝ × ℝ) : Prop := P = (0, 1)

noncomputable def equations_other_sides (x y : ℝ) : Prop :=
  (39 * x + 37 * y + 82 = 0) ∧
  (x - 3 * y - 4 = 0) ∧
  (39 * x + 37 * y - 156 = 0)

noncomputable def distance_to_AB : ℝ := 7 * Real.sqrt 10 / 10

def internal_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 = 115 ∧ θ2 = 65

-- Lean statement of the proof problem
theorem rhombus_properties :
  ∀ (x y : ℝ) (P : ℝ × ℝ),
  equation_AB x y →
  equation_diagonal x y →
  intersection_point P →
  equations_other_sides x y ∧
  Real.dist (0 : ℝ, 1 : ℝ) (x, y) = distance_to_AB ∧
  internal_angles 115 65 :=
by
  sorry

end rhombus_properties_l297_297727


namespace carpet_exchange_l297_297163

theorem carpet_exchange (a b : ℝ) (h_a : 1 < a) (h_b : 1 < b) :
  ∃ c > 1, (c * b > 1 ∧ a / c * b < 1) :=
begin
  -- Choose c such that c > a
  let c := a + 1,
  have h_c_gt_1 : c > 1, from by linarith,
  use [c, h_c_gt_1],
  split,
  -- Prove that c * b > 1
  { -- Since c > a > 1 and b > 1, c * b > 1
    exact mul_pos h_c_gt_1 h_b },
  -- Prove that a / c * b < 1
  { -- Since a / c < 1 (c > a)
    have h_a_div_c_lt_1 : a / c < 1, from div_lt_one_of_lt h_a (by linarith),
    -- then a / c * b < b < b + 1 = c
    exact (mul_lt_iff_lt_one_left h_b).mpr h_a_div_c_lt_1 }
end

end carpet_exchange_l297_297163


namespace problem1_problem2_l297_297333

open Real

-- The function definition
def f (a b x : ℝ) : ℝ := log (a * x + b) + exp (x - 1)

-- The first proof statement
theorem problem1 (x : ℝ) : 
  (∀ x < 1, f (-1) 1 x = log (-x + 1) + exp (x - 1)) →
  (∃ c, 0 < c ∧ c < 1 ∧ f (-1) 1 c = 0) :=
sorry

-- The second proof statement
theorem problem2 (a b : ℝ) : 
  (∀ x, f a b x ≤ exp (x - 1) + x + 1) →
  a > 0 ∧ b ≤ 2 * a - a * log a →
  ab ≤ (1 / 2) * exp 3 :=
sorry

end problem1_problem2_l297_297333


namespace circumscribed_radius_of_intersection_triangle_l297_297111

theorem circumscribed_radius_of_intersection_triangle (r1 r2 r3 : ℝ) (s : ℝ)
    (h1 : r1 = sqrt 3)
    (h2 : r2 = sqrt 3)
    (h3 : r3 = 2 * sqrt ((39 - 18 * sqrt 3) / 13))
    (h4 : s = 3) :
    ∃ R : ℝ, 
    (TriangleFormedByIntersectionPointsCircumscribedRadius r1 r2 r3 s = R ∧ R = 12 - 6 * sqrt 3) :=
by
  sorry

end circumscribed_radius_of_intersection_triangle_l297_297111


namespace euler_totient_comparison_l297_297711

-- Define Euler's Totient function
noncomputable def totient (n : ℕ) : ℕ :=
  if n = 0 then 0 else Multiset.card (Multiset.filter (λ m, Nat.gcd m n = 1) (Multiset.range n))

-- Statement of the problem
theorem euler_totient_comparison (m n : ℕ) (h : Nat.gcd m n > 1) : 
  totient (m * n) > totient m * totient n :=
sorry

end euler_totient_comparison_l297_297711


namespace pool_filling_time_correct_l297_297844

noncomputable def fill_time(A B C D P : ℝ) : ℝ :=
  if h1 : (A + B = P / 3) ∧
          (A + C = P / 6) ∧
          (A + D = P / 4) ∧
          (B + C = P / 9) ∧
          (B + D = P / 6) ∧
          (C + D = P / 8) then
    (48 / 43)
  else
    0

theorem pool_filling_time_correct (A B C D P : ℝ) 
    (h1 : A + B = P / 3)
    (h2 : A + C = P / 6)
    (h3 : A + D = P / 4)
    (h4 : B + C = P / 9)
    (h5 : B + D = P / 6)
    (h6 : C + D = P / 8) :
  fill_time A B C D P = 48 / 43 :=
by
  rw [fill_time]
  simp [h1, h2, h3, h4, h5, h6, if_pos]
  sorry

end pool_filling_time_correct_l297_297844


namespace length_of_first_train_l297_297937

-- Given conditions as definitions
def speed_train1_kmph : ℝ := 80
def speed_train2_kmph : ℝ := 70
def length_train2 : ℝ := 100
def time_to_pass : ℝ := 5.999520038396928

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : ℝ := (speed_train1_kmph + speed_train2_kmph) * kmph_to_mps

-- Total combined length of the trains when passing each other in meters
def combined_length : ℝ := relative_speed_mps * time_to_pass

-- Theorem asserting the length of the first train
theorem length_of_first_train : (combined_length - length_train2) = 150 := by
  sorry

end length_of_first_train_l297_297937


namespace xy_product_l297_297862

theorem xy_product (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
sorry

end xy_product_l297_297862


namespace selling_price_l297_297804

variable (P : ℝ)

def johnSells80Percent (n : ℕ) : ℕ := (0.80 * n.toFloat).toInt

def totalRevenue (numSold : ℕ) (P : ℝ) : ℝ := (numSold : ℝ) * P

def buyingCostPerNewspaper (P : ℝ) : ℝ := 0.25 * P

def totalBuyingCost (numBought : ℕ) (buyingPricePerItem : ℝ) : ℝ := 
  (numBought : ℝ) * buyingPricePerItem

def profit (totalRevenue : ℝ) (totalCost : ℝ) : ℝ := totalRevenue - totalCost

axiom problem_conditions :
  let numNewspapers := 500
  ∧ let numSold := johnSells80Percent numNewspapers
  ∧ let totalRevenue := totalRevenue numSold P
  ∧ let buyingPrice := buyingCostPerNewspaper P
  ∧ let totalCost := totalBuyingCost numNewspapers buyingPrice
  → profit totalRevenue totalCost = 550

theorem selling_price : ∃ P : ℝ, 
  let numNewspapers := 500
  let numSold := johnSells80Percent numNewspapers
  let totalRevenue := totalRevenue numSold P
  let buyingPrice := buyingCostPerNewspaper P
  let totalCost := totalBuyingCost numNewspapers buyingPrice
  ∧ profit totalRevenue totalCost = 550
  → P = 2 :=
by
  sorry

end selling_price_l297_297804


namespace clerical_percentage_l297_297044

theorem clerical_percentage (total_employees clerical_fraction reduce_fraction: ℕ) 
  (h1 : total_employees = 3600) 
  (h2 : clerical_fraction = 1 / 3)
  (h3 : reduce_fraction = 1 / 2) : 
  ( (reduce_fraction * (clerical_fraction * total_employees)) / 
    (total_employees - reduce_fraction * (clerical_fraction * total_employees))) * 100 = 20 :=
by
  sorry

end clerical_percentage_l297_297044


namespace number_of_players_in_tournament_l297_297879

theorem number_of_players_in_tournament (G : ℕ) (h1 : G = 42) (h2 : ∀ n : ℕ, G = n * (n - 1)) : ∃ n : ℕ, G = 42 ∧ n = 7 :=
by
  -- Let's suppose n is the number of players, then we need to prove
  -- ∃ n : ℕ, 42 = n * (n - 1) ∧ n = 7
  sorry

end number_of_players_in_tournament_l297_297879


namespace area_AQPBO_constant_l297_297289

open EuclideanGeometry

variables {r β γ : ℝ}
variables (O A B C P Q : Point ℝ)
variables (h_angle : angle A O B < π / 2)
variables (h_O_center : is_center O [O, A, B])
variables (C_on_arc : lies_on_arc C A B O)
variables (P_on_OC : lies_on_segment P O C)
variables (Q_on_OC : lies_on_segment Q O C)
variables (Q_parallel : lies_on_parallel_through Q B A P)

--Now we state the goal
theorem area_AQPBO_constant :
  area_polygon [A, Q, P, B, O] = (1/2) * r^2 * (real.sin β) :=
by sorry

end area_AQPBO_constant_l297_297289


namespace find_a2_a3_a4_Sn_l297_297800

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a n * a (n + 1) = 4^n

theorem find_a2_a3_a4_Sn (a : ℕ → ℝ) (h : sequence a) :
  a 2 = 4 ∧ a 3 = 4 ∧ a 4 = 16 ∧ (∀ n, (∑ k in Finset.range n, a (2 * k + 2)) = (4 / 3) * (4^n - 1)) := 
by
  sorry

end find_a2_a3_a4_Sn_l297_297800


namespace angle_CAD_in_convex_quadrilateral_l297_297784

theorem angle_CAD_in_convex_quadrilateral {A B C D : Type} [EuclideanGeometry A B C D]
  (AB_eq_BD : A = B, B = D)
  (angle_A : ∠ A = 65)
  (angle_B : ∠ B = 80)
  (angle_C : ∠ C = 75)
  : ∠ A = 15 :=
by
  sorry

end angle_CAD_in_convex_quadrilateral_l297_297784


namespace pollution_remains_percentage_l297_297987

theorem pollution_remains_percentage
  (P_0 : ℝ) (k t : ℝ)
  (h1 : ∀ t, t ≥ 0 → (P_0 * real.exp (-k * t)) = P_0 * real.exp (-k * t)) -- Defining the equation P = P_0e^{-kt}
  (h2 : P_0 * real.exp (-5 * k) = 0.9 * P_0) -- Condition that 10% pollutants are eliminated in the first 5 hours
  : P_0 * real.exp (-10 * k) = 0.81 * P_0 := 
sorry

end pollution_remains_percentage_l297_297987


namespace count_numbers_of_form_divisible_by_5_l297_297125

theorem count_numbers_of_form_divisible_by_5 :
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  ∃ count : ℕ, count = (card a_vals) * (card b_vals) * (card c_vals) ∧ count = 180 :=
by
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  existsi ((card a_vals) * (card b_vals) * (card c_vals))
  split
  sorry
  exact 180

end count_numbers_of_form_divisible_by_5_l297_297125


namespace carl_candy_bars_l297_297243

def weekly_earnings := 0.75
def weeks := 4
def candy_bar_cost := 0.50

theorem carl_candy_bars :
  (weeks * weekly_earnings) / candy_bar_cost = 6 := by
  sorry

end carl_candy_bars_l297_297243


namespace grain_to_rice_system_l297_297796

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l297_297796


namespace find_physics_marks_l297_297571

theorem find_physics_marks (P C M : ℕ) (h1 : P + C + M = 210) (h2 : P + M = 180) (h3 : P + C = 140) : P = 110 :=
sorry

end find_physics_marks_l297_297571


namespace Carl_can_buy_six_candy_bars_l297_297240

-- Define the earnings per week
def earnings_per_week : ℝ := 0.75

-- Define the number of weeks Carl works
def weeks : ℕ := 4

-- Define the cost of one candy bar
def cost_per_candy_bar : ℝ := 0.50

-- Calculate total earnings
def total_earnings := earnings_per_week * weeks

-- Calculate the number of candy bars Carl can buy
def number_of_candy_bars := total_earnings / cost_per_candy_bar

-- State the theorem that Carl can buy exactly 6 candy bars
theorem Carl_can_buy_six_candy_bars : number_of_candy_bars = 6 := by
  sorry

end Carl_can_buy_six_candy_bars_l297_297240


namespace sum_of_squares_inequality_l297_297876

theorem sum_of_squares_inequality (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end sum_of_squares_inequality_l297_297876


namespace sequence_b_10_value_l297_297646

theorem sequence_b_10_value :
  let b : ℕ → ℕ := λ n, if n = 1 then 3 else (λ k, b (k-1) + 2 * (k-1) + (k-1)^2) (n-1)
  in (b 10 = 378) :=
  by {
    sorry
  }

end sequence_b_10_value_l297_297646


namespace exists_m_n_nat_l297_297419

-- Given conditions
variables {x y : ℝ} {a b : ℚ}

-- Hypotheses
def sin_cos_condition_1 (x y : ℝ) (a : ℚ) : Prop := real.sin x + real.cos y = a
def sin_cos_condition_2 (x y : ℝ) (b : ℚ) : Prop := real.sin y + real.cos x = b

theorem exists_m_n_nat (hx : sin_cos_condition_1 x y a) (hy : sin_cos_condition_2 x y b)
  (ha_pos : 0 < a) (hb_pos : 0 < b) :
  ∃ (m n : ℕ), ∃ (k : ℕ), real.sin x * (m : ℝ) + real.cos x * (n : ℝ) = k :=
begin
  -- proof goes here
  sorry
end

end exists_m_n_nat_l297_297419


namespace imaginary_part_of_product_l297_297385

open Complex  -- Open the complex number namespace

theorem imaginary_part_of_product :
  let z := (3 - 2 * I) * (1 + I) in
  z.im = 1 :=
by
  let z := (3 - 2 * I) * (1 + I)
  have h : z = 5 + I := by sorry
  show z.im = 1 from by
    rw h
    simp

end imaginary_part_of_product_l297_297385


namespace cassie_brian_meet_time_l297_297217

open scoped Classical

noncomputable def meet_time := by
  -- Define the variables
  let x := 2.359375 -- Number of hours after 8:15 AM
  let cassie_distance := 14 * x
  let brian_distance := 18 * (x - 0.75)
  let route_distance := 62
  
  -- Equation Setup: distance covered by Cassie and Brian
  have h : cassie_distance + brian_distance = route_distance := by
    calc
      14 * x + 18 * (x - 0.75) = 14 * x + 18 * x - 18 * 0.75  : by ring
                         ... = 32 * x - 13.5                : by ring
                         ... = 75.5                         : by norm_num
                         ... = route_distance               : by norm_num
  
  -- Extract the meeting time from x
  have meeting_time := 8 * 60 + 15 + (x * 60).toNat
  
  -- Prove the meeting time is 10:30
  have meet_at_1030 := meeting_time = 630
  exact meet_at_1030

theorem cassie_brian_meet_time : meet_time = 10 * 60 + 30 := by
  exact meet_time

end cassie_brian_meet_time_l297_297217


namespace m_squared_divisible_by_64_l297_297763

theorem m_squared_divisible_by_64 (m : ℕ) (h : 8 ∣ m) : 64 ∣ m * m :=
sorry

end m_squared_divisible_by_64_l297_297763


namespace min_operations_to_determine_rectangle_l297_297109

-- Lean definitions for measurements and comparisons
variable (A B C D : Type)
variable dist : A → A → ℝ
variable cmp : ℝ → ℝ → Prop

-- Distances between points
variable (AB BC CD DA AC BD : ℝ)

-- Conditions: measure the distances and compare them
variable measure_AB measure_BC measure_CD measure_DA measure_AC measure_BD : ℝ
variable cmp_AB_CD cmp_BC_DA cmp_AC_BD : Prop

-- Lean statement for the minimum operations to determine if quadrilateral ABCD is a rectangle
theorem min_operations_to_determine_rectangle (A B C D : Type) 
  (dist : A → A → ℝ) 
  (cmp : ℝ → ℝ → Prop)
  (measure_AB measure_BC measure_CD measure_DA measure_AC measure_BD : ℝ) 
  (cmp_AB_CD cmp_BC_DA cmp_AC_BD : Prop)
  : measure_AB + measure_BC + measure_CD + measure_DA + measure_AC + measure_BD +
    cmp_AB_CD + cmp_BC_DA + cmp_AC_BD ≥ 9 :=
sorry

end min_operations_to_determine_rectangle_l297_297109


namespace period_length_equals_repunit_ones_l297_297858

theorem period_length_equals_repunit_ones (p : ℕ) (hp : Nat.Prime p) (hp3 : p ≠ 3) :
  ∃ d : ℕ, (10^d - 1) % p = 0 ∧ (Nat.digits 10 (10^d - 1) = List.repeat 1 d) :=
by
  sorry

end period_length_equals_repunit_ones_l297_297858


namespace total_students_in_line_l297_297718

theorem total_students_in_line (students_ahead_of_eunjung students_between_eunjung_and_yoojung : ℕ) 
(eunjung_position : students_ahead_of_eunjung = 4)
(yoojung_last : students_between_eunjung_and_yoojung = 8) : 
students_ahead_of_eunjung + 1 + students_between_eunjung_and_yoojung + 1 = 14 :=
by
  rw [eunjung_position, yoojung_last]
  sorry

end total_students_in_line_l297_297718


namespace sum_of_consecutive_naturals_l297_297252

theorem sum_of_consecutive_naturals (n : ℕ) : 
  ∃ S : ℕ, S = n * (n + 1) / 2 :=
by
  sorry

end sum_of_consecutive_naturals_l297_297252


namespace number_of_zeros_in_1_over_12_pow_12_l297_297129

theorem number_of_zeros_in_1_over_12_pow_12 : 
  let n := 12 in
  let d := 24 in
  let r := 11 in
  ∀ k : ℕ, k = 1 / (n ^ d) → ∃ m : ℕ, k = (10 ^ (-m)) * _ ∧ m = r := by sorry

end number_of_zeros_in_1_over_12_pow_12_l297_297129


namespace polygon_division_l297_297597

theorem polygon_division (P : Type) (h1 : can_be_divided_into P 100 rectangles) (h2 : ¬ can_be_divided_into P 99 rectangles) : ¬ can_be_divided_into P 100 triangles :=
sorry

end polygon_division_l297_297597


namespace Ride_code_is_1652_l297_297532

def code_mapping : Char → Option ℕ
| 'G' := some 0
| 'R' := some 1
| 'E' := some 2
| 'A' := some 3
| 'T' := some 4
| 'D' := some 5
| 'I' := some 6
| 'S' := some 7
| 'C' := some 8
| 'O' := some 9
| 'V' := some 10
| _   := none

theorem Ride_code_is_1652 : 
  (code_mapping 'R' = some 1) ∧ 
  (code_mapping 'I' = some 6) ∧ 
  (code_mapping 'D' = some 5) ∧ 
  (code_mapping 'E' = some 2) →
  "1652".toList.map (λ c, c.toNat - '0'.toNat) = [1, 6, 5, 2] :=
by
  sorry

end Ride_code_is_1652_l297_297532


namespace wire_division_l297_297180

/-- A wire is 5 feet 4 inches long, with 1 foot equals 12 inches, and it’s divided into 4 equal parts.
    We would like to prove that each part of the wire is 16 inches long. -/
theorem wire_division (total_feet : ℕ) (total_inches : ℕ) (foot_to_inch : ℕ) (num_parts : ℕ) :
  total_feet = 5 → total_inches = 4 → foot_to_inch = 12 → num_parts = 4 →
  (total_feet * foot_to_inch + total_inches) / num_parts = 16 :=
by
  intros h_feet h_inches h_conversion h_parts
  rw [h_feet, h_inches, h_conversion, h_parts]
  calc
    (5 * 12 + 4) / 4 = (60 + 4) / 4 : by rfl
                  ... = 64 / 4       : by rfl
                  ... = 16           : by rfl

end wire_division_l297_297180


namespace marie_clears_messages_in_29_days_l297_297037

theorem marie_clears_messages_in_29_days :
  ∀ (initial_messages read_per_day new_messages_per_day : ℕ),
    initial_messages = 198 → 
    read_per_day = 15 → 
    new_messages_per_day = 8 → 
    (let net_reduction := read_per_day - new_messages_per_day in
     initial_messages / net_reduction = 28 ∧ initial_messages % net_reduction ≠ 0 → 29) :=
begin
  intros initial_messages read_per_day new_messages_per_day h1 h2 h3,
  let net_reduction := read_per_day - new_messages_per_day,
  obtain (division_eq : initial_messages / net_reduction = 28) (remainder_ne_zero : initial_messages % net_reduction ≠ 0),
  -- sorry placeholder to finish the proof
  sorry,
end

end marie_clears_messages_in_29_days_l297_297037


namespace number_of_pizzas_l297_297182

theorem number_of_pizzas (n : ℕ) (h : n = 8) : 
  ∑ k in {1, 2, 3}, Nat.choose n k = 92 := by
  sorry

end number_of_pizzas_l297_297182


namespace order_of_magnitude_l297_297207

theorem order_of_magnitude (a b c : ℝ) (ha : a = 0.3 ^ 3) (hb : b = 3 ^ 0.3) (hc : c = log 3 0.3) : c < a ∧ a < b :=
by
  rw [ha, hb, hc]
  sorry

end order_of_magnitude_l297_297207


namespace sheet_width_l297_297614

theorem sheet_width (L : ℕ) (w : ℕ) (A_typist : ℚ) 
  (L_length : L = 30)
  (A_typist_percentage : A_typist = 0.64) 
  (width_used : ∀ w, w > 0 → (w - 4) * (24 : ℕ) = A_typist * w * 30) : 
  w = 20 :=
by
  intros
  sorry

end sheet_width_l297_297614


namespace sum_abc_eq_binom_l297_297012

def S := { p : (ℕ × ℕ × ℕ) // p.1 > 0 ∧ p.2 > 0 ∧ p.2.2 > 0 ∧ p.1 + p.2 + p.2.2 = 2013 }

theorem sum_abc_eq_binom :
  ∑ (p : S), p.val.1 * p.val.2 * p.val.2.2 = nat.choose 2015 5 :=
by
  sorry

end sum_abc_eq_binom_l297_297012


namespace find_n_l297_297258

theorem find_n :
  (∃ n : ℕ, 0 < n ∧ arctan (1 / 2) + arctan (1 / 3) + arctan (1 / 7) + arctan (1 / (n:ℝ)) = π / 4) :=
begin
  use 7,
  split,
  { exact nat.succ_pos', },  -- Ensuring n > 0
  { simp, sorry, },
end

end find_n_l297_297258


namespace diana_apollo_probability_l297_297667

theorem diana_apollo_probability :
  let outcomes := (6 * 6)
  let successful := (5 + 4 + 3 + 2 + 1)
  (successful / outcomes) = 5 / 12 := sorry

end diana_apollo_probability_l297_297667


namespace totalSleepIsThirtyHours_l297_297429

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end totalSleepIsThirtyHours_l297_297429


namespace ladybugs_with_spots_l297_297500

theorem ladybugs_with_spots (total_ladybugs without_spots with_spots : ℕ) 
  (h1 : total_ladybugs = 67082) 
  (h2 : without_spots = 54912) 
  (h3 : with_spots = total_ladybugs - without_spots) : 
  with_spots = 12170 := 
by 
  -- hole for the proof 
  sorry

end ladybugs_with_spots_l297_297500


namespace find_square_length_l297_297798

noncomputable def cyclic_quadrilateral (AB CD AO OC BO OD : ℝ) (is_parallel : Prop) (side_AB : ℝ) (side_CD : ℝ) : Prop :=
  AB = 5 ∧ CD = 8 ∧ AO = 5 ∧ OC = 5 ∧ BO = 4 ∧ OD = 4 ∧ is_parallel = (AB = CD)

noncomputable def square_length (AB CD PQRS_side : ℝ) : Prop :=
  quadrilateral_properties: cyclic_quadrilateral AB CD 5 5 4 4 (AB = CD ∧ CD ≠ 0) →
  PQ_side = 2.5

theorem find_square_length (AB CD AO OC BO OD : ℝ) (is_parallel : Prop):
  cyclic_quadrilateral AB CD AO OC BO OD is_parallel 5 8 →
  (∃ PQ_side, square_length PQ_side 2.5) :=
begin
  intros,
  sorry
end

end find_square_length_l297_297798


namespace at_least_one_member_has_few_amicable_foes_l297_297880

theorem at_least_one_member_has_few_amicable_foes (n q : ℕ) 
  (persons : Finₓ n → Type) 
  (is_amicable : ∀ (i j : Finₓ n), Prop)
  (is_hostile : ∀ (i j : Finₓ n), Prop)
  (amicable_pairs: Finₓ n → Finₓ n → Prop)
  (hostile_condition: ∀ (i j k : Finₓ n), ∃ p, p ≠ i ∧ p ≠ j ∧ p ≠ k ∧ is_hostile p p) : 
  ∃ x : Finₓ n, (∑ y in persons, amicable_pairs x y) ≤ q * (1 - 4 * q / n^2) := 
sorry

end at_least_one_member_has_few_amicable_foes_l297_297880


namespace cylinder_lateral_surface_area_l297_297898
noncomputable def lateralSurfaceArea (S : ℝ) : ℝ :=
  let l := Real.sqrt S
  let d := l
  let r := d / 2
  let h := l
  2 * Real.pi * r * h

theorem cylinder_lateral_surface_area (S : ℝ) (hS : S ≥ 0) : 
  lateralSurfaceArea S = Real.pi * S := by
  sorry

end cylinder_lateral_surface_area_l297_297898


namespace which_is_lying_l297_297112

-- Ben's statement
def ben_says (dan_truth cam_truth : Bool) : Bool :=
  (dan_truth ∧ ¬ cam_truth) ∨ (¬ dan_truth ∧ cam_truth)

-- Dan's statement
def dan_says (ben_truth cam_truth : Bool) : Bool :=
  (ben_truth ∧ ¬ cam_truth) ∨ (¬ ben_truth ∧ cam_truth)

-- Cam's statement
def cam_says (ben_truth dan_truth : Bool) : Bool :=
  ¬ ben_truth ∧ ¬ dan_truth

-- Lean statement to be proven
theorem which_is_lying :
  (∃ (ben_truth dan_truth cam_truth : Bool), 
    ben_says dan_truth cam_truth ∧ 
    dan_says ben_truth cam_truth ∧ 
    cam_says ben_truth dan_truth ∧
    ¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) ↔ (¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) :=
sorry

end which_is_lying_l297_297112


namespace opposite_of_four_l297_297944

theorem opposite_of_four : ∃ y : ℤ, 4 + y = 0 ∧ y = -4 :=
by {
  use -4,
  split,
  { simp },
  { refl },
}

end opposite_of_four_l297_297944


namespace zeros_before_nonzero_l297_297348

theorem zeros_before_nonzero (x : ℝ) (h : x = 1 / (2^3 * 5^7)) : 
  ∃ (n : ℕ), n = 5 ∧ (∃ a b : ℤ, x = a / 10^7 ∧ a % 10^b = a ∧ a ≠ 0) :=
sorry

end zeros_before_nonzero_l297_297348


namespace tip_is_24_l297_297071

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end tip_is_24_l297_297071


namespace problem_statement_l297_297724

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def sequence_b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  2 * Real.log (a n) / Real.log 2 - 1

noncomputable def T_n (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, a (k + 1) + b (k + 1))

theorem problem_statement
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 2 = 4)
  (h3 : a 3 + 2 = (a 2 + a 4) / 2) :
  (∀ n : ℕ, a n = 2 ^ n) ∧
  (∀ n : ℕ, T_n a (sequence_b a) n = 2 ^ (n + 1) - 2 + n^2) :=
by
  sorry

end problem_statement_l297_297724


namespace f_at_five_l297_297380

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 3 * n + 17

theorem f_at_five : f 5 = 207 := 
by 
sorry

end f_at_five_l297_297380


namespace count_ababc_divisible_by_5_l297_297120

theorem count_ababc_divisible_by_5 : 
  let a_vals := {a | 1 ≤ a ∧ a ≤ 9},
      b_vals := {b | 0 ≤ b ∧ b ≤ 9},
      c_vals := {c | c = 0 ∨ c = 5} in
  (∑ a in a_vals, ∑ b in b_vals, ∑ c in c_vals, 1) = 180 := 
by {
  sorry
}

end count_ababc_divisible_by_5_l297_297120


namespace min_moves_to_reset_counters_l297_297920

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l297_297920


namespace range_of_t_value_of_a_l297_297335

def f (a x : ℝ) : ℝ := log a ((1 + x) / (1 - x))

theorem range_of_t (a : ℝ) (t : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ f a (t^2 - t - 1) + f a (t - 2) < 0) →
    ((a > 1 ∧ 1 < t ∧ t < real.sqrt 3) ∨ (0 < a ∧ a < 1 ∧ real.sqrt 3 < t ∧ t < 2)) :=
begin
  intros h,
  sorry
end

theorem value_of_a :
  (∃ (a : ℝ), 0 < a ∧ a ≠ 1 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 / 2 → f a x ∈ set.Icc 0 1) →
    a = 3 :=
begin
  intros h,
  sorry
end

end range_of_t_value_of_a_l297_297335


namespace arthur_dinner_discount_l297_297622

theorem arthur_dinner_discount :
  let appetizer := 8
  let entree := 20
  let wine := 2 * 3
  let dessert := 6
  let full_cost := appetizer + entree + wine + dessert
  let tip := 0.20 * full_cost
  full_cost + tip - D = 38 →
  D = 10 :=
by
  let appetizer := 8
  let entree := 20
  let wine := 2 * 3
  let dessert := 6
  let full_cost := appetizer + entree + wine + dessert
  let tip := 0.20 * full_cost
  assume h : full_cost + tip - D = 38
  sorry

end arthur_dinner_discount_l297_297622


namespace point_on_angle_bisector_l297_297141

-- Define the properties and theorems relevant to the problem.

theorem point_on_angle_bisector {α : Type*} [euclidean_geometry α] 
  {A B : α} (P : α) (h : on_angle_bisector P A B) : 
  is_equidistant P A B :=
begin
  -- The proof here would show that any point on the bisector is equidistant from the two sides of the angle
  sorry
end

end point_on_angle_bisector_l297_297141


namespace find_a10_l297_297290

theorem find_a10 (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a n - a (n+1) = a n * a (n+1)) : 
  a 10 = 1 / 10 :=
sorry

end find_a10_l297_297290


namespace time_to_cross_bridge_l297_297959

-- Defining the given conditions
def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 140

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

-- Calculating the speed in m/s
def speed_of_train_ms : ℚ := kmh_to_ms speed_of_train_kmh

-- Calculating total distance to be covered
def total_distance : ℕ := length_of_train + length_of_bridge

-- Expected time to cross the bridge
def expected_time : ℚ := total_distance / speed_of_train_ms

-- The proof statement
theorem time_to_cross_bridge :
  expected_time = 12.5 := by
  sorry

end time_to_cross_bridge_l297_297959


namespace alpha_beta_sum_l297_297719

open Real

theorem alpha_beta_sum (α β : ℝ) 
  (h₀ : 0 < α ∧ α < π / 2)
  (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : (sin α, cos β) ∥ (cos α, sin β)) : 
  α + β = π / 2 :=
sorry

end alpha_beta_sum_l297_297719


namespace proof_problem_l297_297452

open Real

noncomputable def problem_condition1 (A B : ℝ) : Prop :=
  (sin A - sin B) * (sin A + sin B) = sin (π/3 - B) * sin (π/3 + B)

noncomputable def problem_condition2 (b c : ℝ) (a : ℝ) (dot_product : ℝ) : Prop :=
  b * c * cos (π / 3) = dot_product ∧ a = 2 * sqrt 7

noncomputable def problem_condition3 (a b c : ℝ) : Prop := 
  a^2 = (b + c)^2 - 3 * b * c

noncomputable def problem_condition4 (b c : ℝ) : Prop := 
  b < c

theorem proof_problem (A B : ℝ) (a b c dot_product : ℝ)
  (h1 : problem_condition1 A B)
  (h2 : problem_condition2 b c a dot_product)
  (h3 : problem_condition3 a b c)
  (h4 : problem_condition4 b c) :
  (A = π / 3) ∧ (b = 4 ∧ c = 6) :=
by {
  sorry
}

end proof_problem_l297_297452


namespace relationship_between_skew_lines_l297_297761

-- Define the concept of skew lines
def skew_lines (a b : Type*) : Prop :=
  ∃ (p q : a), ∃ (r s : b), ¬ (p = q) ∧ ¬ (r = s) ∧ line_through p q = ∅ ∧ line_through r s = ∅

-- Lean 4 theorem statement
theorem relationship_between_skew_lines (a b c : Type*) :
  (skew_lines a b) ∧ (skew_lines b c) → 
  (skew_lines a c) ∨ (a = c ∨ ¬ (line_through a c = ∅)) :=
sorry

end relationship_between_skew_lines_l297_297761


namespace problem_parabola_value_of_p_l297_297094

noncomputable def parabola_p (p a b : ℝ) : Prop :=
  b^2 = 2 * p * a ∧
  (let fx := p / 2, fy := 0 in
   dist (a, b) (fx, fy) = 2 * |a|) ∧
  1 / 2 * |p / 2| * |b| = 1

theorem problem_parabola_value_of_p {p : ℝ} (hp : 0 < p) (a b : ℝ) :
  (∃ (a b : ℝ), parabola_p p a b) → p = 2 := sorry

end problem_parabola_value_of_p_l297_297094


namespace permutation_sum_l297_297503

theorem permutation_sum (n : ℕ) (P : Fin n → Fin (n+1))
  (h1 : 2 ≤ n)
  (h2 : ∀ i j, i ≠ j → P i ≠ P j)
  (h3 : ∀ i, P i < n+1) :
  (Finset.range (n-1)).sum (λ i, 1 / (↑(P ⟨i, Nat.lt_of_lt_of_le i.des max_lt_succ h1⟩) + P ⟨i + 1, Nat.lt_of_le_of_lt (Nat.succ_le_succ (Fin.prop (⟨i, Nat.lt_succ_iff_lt.mp i.des⟩))) (Nat.lt_succ_iff_lt.mp i.des)⟩))) >
  (n - 1) / (n + 2: ℝ) := sorry

end permutation_sum_l297_297503


namespace ratio_of_rise_32_l297_297110

noncomputable def ratio_of_rise_of_liquid_level 
  (V : Volume)
  (r1 r2 : ℝ) 
  (h1 h2 : ℝ) 
  (sm_radius m1_radius : ℝ) 
  (lg_radius m2_radius : ℝ) 
  (rise_sm rise_lg : ℝ) : ℝ :=
(if r1 = 4 ∧ r2 = 8 ∧ sm_radius = 2 ∧ lg_radius = 1 ∧ V = \frac{π}{3} * r1^2 * h1 ∧ V = \frac{π}{3} * r2^2 * h2 
then rise_sm / rise_lg 
else sorry)

theorem ratio_of_rise_32 (r1 r2 : ℝ) (h1 h2 : ℝ) (sm_radius m1_radius : ℝ) (lg_radius m2_radius : ℝ) :
  r1 = 4 ∧ r2 = 8 ∧ sm_radius = 2 ∧ lg_radius = 1 ∧ \frac{16}{3} * π * h1 = \frac{64}{3} * π * h2 
  ∧ rise_of_liquid_level (sm_radius m1_radius) (lg_radius m2_radius) 
  → (ratio_of_rise_of_liquid_level (V: π * (r1^2) * h1) (r1 = 4) (r2 = 8) 
  (h1: Float) (h2: Float) (sm_radius:2) (lg_radius:1)) 
  = \frac{rise_sm}{\frac{1}{16}} 
  := by {
sorry
}

end ratio_of_rise_32_l297_297110


namespace mr_smith_children_l297_297042

noncomputable def gender_probability (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let equal_gender_ways := Nat.choose n (n / 2)
  let favourable_outcomes := total_outcomes - equal_gender_ways
  favourable_outcomes / total_outcomes

theorem mr_smith_children (n : ℕ) (h : n = 8) : 
  gender_probability n = 93 / 128 :=
by
  rw [h]
  sorry

end mr_smith_children_l297_297042


namespace alice_champion_probability_l297_297200

theorem alice_champion_probability :
  ∀ (players : Fin 8 → ℕ), -- 8 players
  (players 0 = 1) → -- Alice plays rock
  (players 1 = 2) → -- Bob plays paper
  (∀ i, (2 ≤ i) → (i < 8) → (players i = 3)) → -- Others play scissors
  (∀ (game_result : Fin 8 → Fin 8 → ℕ), -- Game results
  (∀ i j, (players i = 1) → (players j = 3) → (game_result i j = 1)) → -- Rock beats Scissors
  (∀ i j, (players i = 3) → (players j = 2) → (game_result i j = 1)) → -- Scissors beat Paper
  (∀ i j, (players i = 2) → (players j = 1) → (game_result i j = 1)) → -- Paper beats Rock
  (∀ i j, (players i = players j) → (0 ≤ game_result i j) ∧ (game_result i j ≤ 1)) → -- Tie handled by coin flip
  (optimally_random_pairing : (ℕ → option (Fin 8 × Fin 8))) → -- Random pairing for each round
  ∀ rnd_result, -- For any round results
  (∃ (prob_win_alice : ℝ), prob_win_alice = (6 / 7)) :=
begin
  sorry
end

end alice_champion_probability_l297_297200


namespace percentage_per_cup_l297_297554

-- Define the total capacity of the pitcher and the amount filled
def total_pitcher_capacity (P : ℝ) : ℝ := P
def filled_amount (P : ℝ) : ℝ := (2 / 3) * P

-- Given the amount is distributed into 8 cups
def amount_per_cup (P : ℝ) : ℝ := filled_amount P / 8

-- Prove that each cup received 8.33% of the total capacity of the pitcher
theorem percentage_per_cup (P : ℝ) : 
  ((amount_per_cup P) / (total_pitcher_capacity P)) * 100 = 8.33 := by
  sorry

end percentage_per_cup_l297_297554


namespace circumcenter_of_AIC_on_circumcircle_of_ABC_l297_297461

theorem circumcenter_of_AIC_on_circumcircle_of_ABC
  (ABC : Triangle)
  (I : Point)
  (h1 : is_incenter I ABC)
  (h2 : equidistant_from_sides I ABC)
  (B : Point)
  (h3 : bisects_angle B I ABC)
  (K : Point)
  (h4 : intersection_circle B I ABC K) :
  is_on_circumcircle K (triangle_circumcircle ABC) :=
sorry

end circumcenter_of_AIC_on_circumcircle_of_ABC_l297_297461


namespace ali_babas_cave_min_moves_l297_297924

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l297_297924


namespace problem1_problem2_l297_297034

def A : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }

def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem problem1 (a: ℝ) (h : a = 1/5) : ¬ (B a ⊆ A) :=
by {
    /- The proof will go here. -/
    sorry
}

theorem problem2 (a: ℝ) : (B a ⊆ A) → a ∈ {0, 1/3, 1/5} :=
by {
    /- The proof will go here. -/
    sorry
}

end problem1_problem2_l297_297034


namespace isosceles_vertex_angle_l297_297402

-- Let T be a type representing triangles, with a function base_angle returning the degree of a base angle,
-- and vertex_angle representing the degree of the vertex angle.
axiom Triangle : Type
axiom is_isosceles (t : Triangle) : Prop
axiom base_angle_deg (t : Triangle) : ℝ
axiom vertex_angle_deg (t : Triangle) : ℝ

theorem isosceles_vertex_angle (t : Triangle) (h_isosceles : is_isosceles t)
  (h_base_angle : base_angle_deg t = 50) : vertex_angle_deg t = 80 := by
  sorry

end isosceles_vertex_angle_l297_297402


namespace monthly_sales_trend_l297_297191

def decreasing_then_increasing (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ ∀ x, x < a → f' x < 0 ∧ a < x ∧ x < b → f' x > 0

theorem monthly_sales_trend :
  ∃ f : ℝ → ℝ, decreasing_then_increasing f ∧ f 1 = 8 ∧ f 3 = 2 ∧
  (f = (λ x, 20 * (0.5) ^ x) ∨
   f = (λ x, -6 * log 3 x + 8) ∨
   f = (λ x, x^2 - 12 * x + 19) ∨
   f = (λ x, x^2 - 7 * x + 14)) :=
sorry

end monthly_sales_trend_l297_297191


namespace xy_product_l297_297865

theorem xy_product (x y : ℝ) (h1 : 2 ^ x = 16 ^ (y + 1)) (h2 : 27 ^ y = 3 ^ (x - 2)) : x * y = 8 :=
by
  sorry

end xy_product_l297_297865


namespace back_wheel_circumference_proof_l297_297083

noncomputable def cart_problem (front_wheel_circumference : ℝ) 
  (back_wheel_circumference : ℝ) 
  (distance_traveled : ℝ) 
  (revolution_diff : ℕ) :=
  front_wheel_circumference = 30 ∧
  distance_traveled = 2400 ∧
  revolution_diff = 5 ∧
  distance_traveled / front_wheel_circumference - revolution_diff = distance_traveled / back_wheel_circumference

theorem back_wheel_circumference_proof :
  ∃ (back_wheel_circumference : ℝ), 
  cart_problem 30 back_wheel_circumference 2400 5 ∧ back_wheel_circumference = 32 :=
begin
  sorry
end

end back_wheel_circumference_proof_l297_297083


namespace midpoints_on_same_sphere_l297_297847

-- Define two spheres ω1 and ω2 with centers O1 and O2
variables {α : Type} [euclidean_space α]
variables (O1 O2 : α) (r1 r2 : ℝ)

-- Define fixed points A on ω1 and B on ω2
variables (A : α) (B : α)

-- Define variable points X on ω1 and Y on ω2
variables (X : α) (Y : α)

-- Define the predicate that X is on sphere ω1 and Y is on sphere ω2
def on_sphere (P O : α) (r : ℝ) : Prop := (dist P O = r)

-- Definition of midpoints
def midpoint (P Q : α) : α := (P + Q) / 2

-- Given condition AX parallel BY
def parallel (u v : α) : Prop := ∃ k : ℝ, k ≠ 0 ∧ u = k • v

-- The main theorem
theorem midpoints_on_same_sphere
  (h_X_on_sphere1 : on_sphere X O1 r1)
  (h_Y_on_sphere2 : on_sphere Y O2 r2)
  (h_parallel : parallel (X - A) (Y - B)) :
  ∃ R : ℝ, ∀ X Y, on_sphere X O1 r1 → on_sphere Y O2 r2 → parallel (X - A) (Y - B) → 
  on_sphere (midpoint X Y) (midpoint O1 O2) R :=
by {
  sorry
}

end midpoints_on_same_sphere_l297_297847


namespace num_valid_two_digit_numbers_l297_297613

theorem num_valid_two_digit_numbers : 
  {n : ℕ // ∃ a b, n = 10 * a + b ∧ b % 2 = 0 ∧ (9 * a) % 10 = 4} = 5 :=
sorry

end num_valid_two_digit_numbers_l297_297613


namespace number_of_true_statements_l297_297871

def are_equal (u v : ℝ × ℝ) : Prop := u.1 = v.1 ∧ u.2 = v.2
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (u.1, u.2) = (k * v.1, k * v.2)
def are_collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1, u.2) = (k * v.1, k * v.2)

theorem number_of_true_statements : 
  let s1 := ∀ u v : ℝ × ℝ, are_parallel u v → are_equal u v,
      s2 := ∀ u v : ℝ × ℝ, ¬ are_equal u v → ¬ are_parallel u v,
      s3 := ∀ u v : ℝ × ℝ, are_collinear u v → are_equal u v,
      s4 := ∀ u v : ℝ × ℝ, are_equal u v → are_collinear u v in
  (nat.pred (nat.succ (nat.succ (nat.succ (nat.pred (nat.succ (nat.pred (nat.pred (ite (s1) 1 0) 
  + ite (s2) 1 0) + ite (s3) 1 0) + ite (s4) 1 0))))) = 1 :=
by sorry

end number_of_true_statements_l297_297871


namespace binary_101_to_decimal_l297_297649

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l297_297649


namespace num_shirts_sold_l297_297165

theorem num_shirts_sold (p_jeans : ℕ) (c_shirt : ℕ) (total_earnings : ℕ) (h1 : p_jeans = 10) (h2 : c_shirt = 10) (h3 : total_earnings = 400) : ℕ :=
  let c_jeans := 2 * c_shirt
  let n_shirts := 20
  have h4 : p_jeans * c_jeans + n_shirts * c_shirt = total_earnings := by sorry
  n_shirts

end num_shirts_sold_l297_297165


namespace pizza_toppings_count_l297_297184

theorem pizza_toppings_count : 
  let toppings := 8 in
  (toppings + (Nat.choose toppings 2) + (Nat.choose toppings 3)) = 92 := 
by 
  let toppings := 8
  have h1 : (Nat.choose toppings 2) = 28 := by sorry
  have h2 : (Nat.choose toppings 3) = 56 := by sorry
  sorry

end pizza_toppings_count_l297_297184


namespace shaded_region_area_l297_297529

theorem shaded_region_area (RS : ℝ) (num_squares : ℕ) (area_per_square : ℝ)
  (h1 : RS = 8) (h2 : num_squares = 20) (h3 : area_per_square = (RS^2 / 50)) :
  num_squares * area_per_square = 25.6 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end shaded_region_area_l297_297529


namespace find_prob_of_X_within_interval_l297_297320

noncomputable def normal_dist_prob_1 := 0.6826
noncomputable def normal_dist_prob_2 := 0.9544

variables (μ : ℝ) (σ : ℝ) (X : ℝ → Type)

theorem find_prob_of_X_within_interval 
  (Z : X Z → Prop) [NormalDist Z μ σ]
  (h1 : P(μ - σ < Z ≤ μ + σ) = normal_dist_prob_1)
  (h2 : P(μ - 2σ < Z ≤ μ + 2σ) = normal_dist_prob_2)
  (X_dist : X X → Prop) [NormalDist X 6 4] :
  P(2 < X ≤ 8) = 0.8185 := 
sorry

end find_prob_of_X_within_interval_l297_297320


namespace part_I_part_II_l297_297744

noncomputable def set_A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
noncomputable def set_B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_I (m : ℝ) : (set_A ∩ set_B m = Set.Icc 0 3) → m = 2 := sorry

theorem part_II (m : ℝ) : (set_A ⊆ (Set.Univ \ set_B m)) → m < -3 ∨ m > 5 := sorry

end part_I_part_II_l297_297744


namespace range_of_a_l297_297227

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- The theorem we need to prove
theorem range_of_a {a : ℝ} (h : A a ⊆ B) : 1 ≤ a ∧ a < 5 := 
sorry

end range_of_a_l297_297227


namespace varphi_n_limit_l297_297453

noncomputable def varphi_n (n : ℕ) (t : ℝ) : ℝ :=
  (1 : ℝ) / (2 * n) * ∫ x in -n..n, Real.exp (Complex.I * t * x)

theorem varphi_n_limit (t : ℝ) :
  tendsto (λ n : ℕ, varphi_n n t) at_top (𝓝 (if t = 0 then 1 else 0)) :=
sorry

end varphi_n_limit_l297_297453


namespace parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l297_297703

-- Definitions for the problem conditions
def parabola_symmetry_axis := "coordinate axis"
def parabola_vertex := (0, 0)
def directrix_equation := "x = -1"
def intersects_at_two_points (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) := (l P.1 = P.2) ∧ (l Q.1 = Q.2) ∧ (P ≠ Q)

-- Main theorem statements
theorem parabola_standard_equation : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") → 
  ∃ p, 0 < p ∧ ∀ y x, y^2 = 4 * p * x := 
  sorry

theorem oa_dot_ob_value (l : ℝ → ℝ) (focus : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  l focus.1 = focus.2 → 
  (P.1 * Q.1 + P.2 * Q.2 = -3) := 
  sorry

theorem line_passes_fixed_point (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = -4) → 
  ∃ fp, fp = (2,0) := 
  sorry

end parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l297_297703


namespace cost_of_larger_jar_l297_297172

noncomputable def pi : Real :=
  3.141592653589793

def volume (r : ℝ) (h : ℝ) : ℝ :=
  pi * r^2 * h

def cost_per_volume : ℝ := 1.0 / volume (4/2) 5

def discounted_cost (diameter : ℝ) (height : ℝ) (discount : ℝ) : ℝ :=
  let vol := volume (diameter / 2) height
  let cost := vol * cost_per_volume
  cost * (1 - discount)

theorem cost_of_larger_jar : discounted_cost 8 10 0.10 = 14.40 :=
by
  sorry

end cost_of_larger_jar_l297_297172


namespace alternating_sum_eq_zero_l297_297818

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem alternating_sum_eq_zero :
  (Finset.range 2020).sum (λ k, (-1)^(k + 1) * g (k + 1 : ℝ / 2021)) = 0 :=
by
  sorry

end alternating_sum_eq_zero_l297_297818


namespace three_lines_one_point_l297_297237

-- Given the conditions provided
variables {Square : Type} [MetricSpace Square]

-- Each of the 9 lines divide the square into two quadrilaterals with an area ratio of 2:3
def divides_square (L : Square → Square → Prop) := ∃ a b : Square, L a b ∧ area_quadrilateral a b a.fst.to_set / area_quadrilateral a b a.snd.to_set = 2 / 3

-- There are exactly 9 such lines
axiom nine_lines : ∃ (L : list (Square → Square → Prop)), L.length = 9 ∧ ∀ l ∈ L, divides_square l

-- Prove at least three lines intersect at one point
theorem three_lines_one_point :
  ∃ p : Square, ∃ L_sub : list (Square → Square → Prop), L_sub.length ≥ 3 ∧ ∀ l ∈ L_sub, ∃ a b : Square, l a p ∧ l p b :=
sorry

end three_lines_one_point_l297_297237


namespace kyle_total_revenue_l297_297439

-- Define the conditions
def initial_cookies := 60
def initial_brownies := 32
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2
def price_per_cookie := 1
def price_per_brownie := 1.5

-- Statement of the proof
theorem kyle_total_revenue :
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies),
      remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies),
      revenue_from_cookies := remaining_cookies * price_per_cookie,
      revenue_from_brownies := remaining_brownies * price_per_brownie,
      total_revenue := revenue_from_cookies + revenue_from_brownies
  in total_revenue = 99 :=
by
  sorry

end kyle_total_revenue_l297_297439


namespace cubic_yards_to_cubic_feet_l297_297350

def conversion_factor := 3 -- 1 yard = 3 feet
def cubic_conversion := conversion_factor ^ 3 -- 1 cubic yard = (3 feet) ^ 3

theorem cubic_yards_to_cubic_feet :
  5 * cubic_conversion = 135 :=
by
  unfold conversion_factor cubic_conversion
  norm_num
  sorry

end cubic_yards_to_cubic_feet_l297_297350


namespace exists_minimal_A_l297_297463

variable {ι : Type} [LinearOrder ι] [Finite ι]

theorem exists_minimal_A
    {x : ι → ℝ} {a : ι → ℝ} {n : ℕ}
    (hx_sum : (Finset.univ.sum x) = 0)
    (hx_abs_sum : (Finset.univ.sum (λ i, abs (x i))) = 1)
    (ha_monotone : ∀i j : ι, i ≤ j → a i ≥ a j) :
    ∃ A, (∀ {A}, |Finset.univ.sum (λ i, a i * x i)| ≤ A * (a (Finset.min' _) - a (Finset.max' _))) ↔ A ≥ 1/2 :=
begin
  sorry
end

end exists_minimal_A_l297_297463


namespace horizontal_distance_parabola_l297_297177

theorem horizontal_distance_parabola :
  ∀ x_p x_q : ℝ, 
  (x_p^2 + 3*x_p - 4 = 8) → 
  (x_q^2 + 3*x_q - 4 = 0) → 
  x_p ≠ x_q → 
  abs (x_p - x_q) = 2 :=
sorry

end horizontal_distance_parabola_l297_297177


namespace margaret_fraction_of_dollar_l297_297011

theorem margaret_fraction_of_dollar 
  (lance_cents : ℕ) 
  (guy_quarters : ℕ) 
  (guy_dime : ℕ) 
  (bill_dimes : ℕ) 
  (total_cents : ℕ) 
  (combined_cents : ℕ) 
  (guy_cents: ℕ := (guy_quarters * 25 + guy_dime * 10))
  (bill_cents: ℕ := bill_dimes * 10)
  (others_cents: ℕ := lance_cents + guy_cents + bill_cents)
  (margaret_cents : ℕ := combined_cents - others_cents) :

  lance_cents = 70 -> 
  guy_quarters = 2 -> 
  guy_dime = 1 -> 
  bill_dimes = 6 -> 
  combined_cents = 265 -> 
  margaret_cents = 75 ->
  (75 / 100 : ℚ) = (3 / 4 : ℚ) :=
by
  intros
  simp
  sorry

end margaret_fraction_of_dollar_l297_297011


namespace no_equal_partition_product_l297_297055

theorem no_equal_partition_product (n : ℕ) (h : n > 1) : 
  ¬ ∃ A B : Finset ℕ, 
    (A ∪ B = (Finset.range n).erase 0 ∧ A ∩ B = ∅ ∧ (A ≠ ∅) ∧ (B ≠ ∅) 
    ∧ A.prod id = B.prod id) := 
sorry

end no_equal_partition_product_l297_297055


namespace consecutive_labels_probability_l297_297583

theorem consecutive_labels_probability :
  let total_labels := 10
  let total_pairs := (total_labels * (total_labels - 1)) / 2
  let consecutive_pairs := total_labels - 1
  (consecutive_pairs : ℚ) / total_pairs = 1 / 5 :=
by
  let total_labels : ℕ := 10
  let total_pairs : ℚ := (total_labels * (total_labels - 1)) / 2
  let consecutive_pairs: ℚ := total_labels - 1
  have h : consecutive_pairs / total_pairs = 1 / 5 :=
    by norm_num; exact div_eq_of_eq_mul_right (by norm_num) rfl
  exact h

end consecutive_labels_probability_l297_297583


namespace product_increase_l297_297116

theorem product_increase (a b : ℝ) (h : (a + 1) * (b + 1) = 2 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a^2 - 1) * (b^2 - 1) = 4 * a * b) :=
sorry

end product_increase_l297_297116


namespace probability_at_least_one_three_l297_297544

theorem probability_at_least_one_three :
  let E := { (d1, d2) : Fin 8 × Fin 8 | d1 = 2 ∨ d2 = 2 } in
  (↑E.card / ↑((Fin 8 × Fin 8).card) : ℚ) = 15 / 64 :=
by
  /- Let E be the set of outcomes where at least one die shows a 3. -/
  sorry

end probability_at_least_one_three_l297_297544


namespace min_moves_to_zero_l297_297931

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l297_297931


namespace evaluate_expression_m_4_evaluate_expression_m_negative_4_l297_297372

variables (a b c d m : ℝ)

theorem evaluate_expression_m_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_4 : m = 4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = 35 :=
by sorry

theorem evaluate_expression_m_negative_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_negative_4 : m = -4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = -13 :=
by sorry

end evaluate_expression_m_4_evaluate_expression_m_negative_4_l297_297372


namespace triangle_garden_area_ratio_l297_297593

theorem triangle_garden_area_ratio (side_length : ℝ) (trisect_dist : ℝ) (triangle_side_length : ℝ) 
  (square_area : ℝ) : side_length = 12 → trisect_dist = 4 → triangle_side_length = 8 → 
  square_area = side_length * side_length → 
  (area_equilateral_triangle triangle_side_length) / square_area = (Real.sqrt 3) / 9 :=
by 
  sorry

def area_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

end triangle_garden_area_ratio_l297_297593


namespace match_piles_l297_297108

theorem match_piles (a b c : ℕ) (h : a + b + c = 96)
    (h1 : 2 * b = a + c) (h2 : 2 * c = b + a) (h3 : 2 * a = c + b) : 
    a = 44 ∧ b = 28 ∧ c = 24 :=
  sorry

end match_piles_l297_297108


namespace at_least_one_person_with_both_neighbors_as_boys_l297_297533

theorem at_least_one_person_with_both_neighbors_as_boys :
  ∀ (table : list bool),
  length table = 50 →
  (count (λ x, x = tt) table = 25) →
  (count (λ x, x = ff) table = 25) →
  ∃ i, table[(i + 49) % 50] = tt ∧ table[i] = tt ∧ table[(i + 1) % 50] = tt :=
by
  intros table h_len h_boys h_girls
  sorry

end at_least_one_person_with_both_neighbors_as_boys_l297_297533


namespace male_red_ants_percentage_l297_297148

noncomputable def percentage_of_total_ant_population_that_are_red_females (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    (percent_red_are_females / 100) * red_population_pct

noncomputable def percentage_of_total_ant_population_that_are_red_males (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    red_population_pct - percentage_of_total_ant_population_that_are_red_females total_population_pct red_population_pct percent_red_are_females

theorem male_red_ants_percentage (total_population_pct red_population_pct percent_red_are_females male_red_ants_pct : ℝ) :
    red_population_pct = 85 → percent_red_are_females = 45 → male_red_ants_pct = 46.75 →
    percentage_of_total_ant_population_that_are_red_males total_population_pct red_population_pct percent_red_are_females = male_red_ants_pct :=
by
sorry

end male_red_ants_percentage_l297_297148


namespace total_area_union_rectangles_l297_297057

-- Define the lengths of the sides of the initial rectangle R_0
def side_length1_R0 := 3
def side_length2_R0 := 4

-- Define the conditions of the problem
def common_vertex_shared (P : Type) := P

def diagonal_Rn_of_Rn_1 (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

def opposite_side_passes_through_vertex (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

def center_location_counterclockwise (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem stating the total area covered by the union is 30
theorem total_area_union_rectangles : 
  ∃ (A0 A1 A2 A3 : ℕ), 
    A0 = side_length1_R0 * side_length2_R0 ∧
    A1 = sorry ∧ -- Area of R1 based on its sides
    A2 = sorry ∧ -- Area of R2 based on its sides
    A3 = sorry ∧ -- Area of R3 based on its sides
    A0 + A1 + A2 + A3 = 30 := 
begin
  sorry
end

end total_area_union_rectangles_l297_297057


namespace ali_babas_cave_min_moves_l297_297922

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l297_297922


namespace fraction_equality_l297_297214

theorem fraction_equality :
  (2 - (1 / 2) * (1 - (1 / 4))) / (2 - (1 - (1 / 3))) = 39 / 32 := 
  sorry

end fraction_equality_l297_297214


namespace john_order_cost_l297_297604

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end john_order_cost_l297_297604


namespace ellipse_equation_through_point_with_foci_l297_297902

theorem ellipse_equation_through_point_with_foci (a b c : ℝ) (h1 : a^2 = 9) (h2 : b^2 = 4) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (p : ℝ × ℝ) (h4 : p = (3, -2)) : 
  (∃ m n : ℝ, m > n > 0 ∧ m - n = 5 ∧ (3^2 / m + (-2)^2 / n = 1) ∧ ((∀ (x y : ℝ), x^2 / m + y^2 / n = 1 → (x, y) = (3, -2)))) 
  ↔ (3^2 / 15 + (-2)^2 / 10 = 1) := 
by 
  sorry

end ellipse_equation_through_point_with_foci_l297_297902


namespace AF_Calculation_l297_297568

theorem AF_Calculation (A B C P D E F : Point) (s t : ℝ) 
  (hP_inside_ABC : ∃ (α β γ : ℝ), α + β + γ = 1 ∧ α > 0 ∧ β > 0 ∧ γ > 0 ∧ A = α • B + β • C + γ • B)
  (h_intersection_D : D ∈ line_through(A, P) ∧ D ∈ line_through(B, C))
  (h_intersection_E : E ∈ line_through(B, P) ∧ E ∈ line_through(C, A))
  (h_intersection_F : F ∈ line_through(C, P) ∧ F ∈ line_through(A, B))
  (h_right_angle : angle A P B = 90°)
  (h_equal_sides : dist A C = s ∧ dist B C = s)
  (h_adjacent_segments : dist A B = t ∧ dist B D = t)
  (h_BF_length : dist B F = 1)
  (h_BC_length : dist B C = 999) :
  dist A F = 499/500 :=
sorry

end AF_Calculation_l297_297568


namespace Kyle_makes_99_dollars_l297_297436

-- Define the initial numbers of cookies and brownies
def initial_cookies := 60
def initial_brownies := 32

-- Define the numbers of cookies and brownies eaten by Kyle and his mom
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2

-- Define the prices for each cookie and brownie
def price_per_cookie := 1
def price_per_brownie := 1.50

-- Define the remaining cookies and brownies after consumption
def remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
def remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies

-- Define the total money Kyle will make
def money_from_cookies := remaining_cookies * price_per_cookie
def money_from_brownies := remaining_brownies * price_per_brownie

-- Define the total money Kyle will make from selling all baked goods
def total_money := money_from_cookies + money_from_brownies

-- Proof statement
theorem Kyle_makes_99_dollars :
  total_money = 99 :=
by
  sorry

end Kyle_makes_99_dollars_l297_297436


namespace complex_magnitude_problem_l297_297458

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : complex.abs z = 2)
  (hw : complex.abs w = 4)
  (hz_w : complex.abs (z + w) = 5) :
  complex.abs ((1 / z) + (1 / w)) = 5 / 8 := by
  sorry

end complex_magnitude_problem_l297_297458


namespace has_root_in_interval_l297_297907

def f (x : ℝ) := x^3 - 3*x - 3

theorem has_root_in_interval : ∃ c ∈ (Set.Ioo (2:ℝ) 3), f c = 0 :=
by 
    sorry

end has_root_in_interval_l297_297907


namespace cost_50_jasmines_discounted_l297_297624

variable (cost_per_8_jasmines : ℝ) (num_jasmines : ℕ) (discount : ℝ)
variable (proportional : Prop) (c_50_jasmines : ℝ)

-- Given the cost of a bouquet with 8 jasmines
def cost_of_8_jasmines : ℝ := 24

-- Given the price is directly proportional to the number of jasmines
def price_proportional := ∀ (n : ℕ), num_jasmines = 8 → proportional

-- Given the bouquet with 50 jasmines
def num_jasmines_50 : ℕ := 50

-- Applying a 10% discount
def ten_percent_discount : ℝ := 0.9

-- Prove the cost of the bouquet with 50 jasmines after a 10% discount
theorem cost_50_jasmines_discounted :
  proportional ∧ (c_50_jasmines = (cost_of_8_jasmines / 8) * num_jasmines_50) →
  (c_50_jasmines * ten_percent_discount) = 135 :=
by
  sorry

end cost_50_jasmines_discounted_l297_297624


namespace sin_sum_simplification_l297_297062

open Real

theorem sin_sum_simplification (α : ℝ) (n : ℕ) :
  (∑ k in Finset.range n, sin (2 * (k + 1) * α)) = (sin ((n + 1) * α) * sin (n * α)) / (sin α) :=
sorry

end sin_sum_simplification_l297_297062


namespace original_sheet_perimeter_l297_297188

theorem original_sheet_perimeter (L W : ℕ) 
  (hL : L = 18)
  (hW : W = 28)
  (h1 : 2 * (L + W) - 2 * (L + W - 10) = 20)
  (h2 : 2 * (L + W - 10) - 2 * (L + W - 10 - 8) = 16) :
  2 * (L + W) = 92 := by
  rw [hL, hW]
  sorry

end original_sheet_perimeter_l297_297188


namespace probability_at_least_one_three_l297_297543

theorem probability_at_least_one_three :
  let E := { (d1, d2) : Fin 8 × Fin 8 | d1 = 2 ∨ d2 = 2 } in
  (↑E.card / ↑((Fin 8 × Fin 8).card) : ℚ) = 15 / 64 :=
by
  /- Let E be the set of outcomes where at least one die shows a 3. -/
  sorry

end probability_at_least_one_three_l297_297543


namespace find_m_l297_297765

theorem find_m (m : ℤ) (h1 : m + 1 ≠ 0) (h2 : m^2 + 3 * m + 1 = -1) : m = -2 := 
by 
  sorry

end find_m_l297_297765


namespace cannot_be_factored_into_consecutive_l297_297854

noncomputable def f (n k : ℕ) : ℕ := 2 * n ^ (3 * k) + 4 * n ^ k + 10

theorem cannot_be_factored_into_consecutive (n k : ℕ) : 
  ¬ ∃ seq : ℕ → ℕ, (∀ i, seq i = seq 0 + i ∧ f(n, k) = ∏ i in finset.range (f(n, k)), seq i)
:= sorry

end cannot_be_factored_into_consecutive_l297_297854


namespace a_2012_eq_6_l297_297282

-- Define the sequence using the provided conditions.
noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 3
  else if n = 2 then 6
  else a (n - 1) - a (n - 2)

-- The theorem we want to prove.
theorem a_2012_eq_6 : a 2012 = 6 := 
sorry

end a_2012_eq_6_l297_297282


namespace find_a_find_range_l297_297330

-- Function definition and assumption of extremum at x = -1 implies a = 1
theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^3 - 3 * a * x - 1)
  (extremum_at_minus_one : ∀ f', (∀ x, f' x = deriv f x) →
    f' (-1) = 0) : a = 1 := sorry

-- Prove the range of the function when x ∈ [-2, 1) given a = 1
theorem find_range (f : ℝ → ℝ) (a : ℝ) (h : a = 1) (h_f : f = λ x, x^3 - 3 * x - 1)
  (interval : set.Icc (-2 : ℝ) 1)
  (range : set.Icc (-3 : ℝ) 1) : 
  ∀ y ∈ range, ∃ x ∈ interval, f x = y := sorry

end find_a_find_range_l297_297330


namespace monotonic_intervals_of_f_unique_extreme_point_value_l297_297334

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln x - a * x - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (ln x - a * x - 1) + (1 / 2) * x ^ 2 + 2 * x

theorem monotonic_intervals_of_f (a : ℝ) : 
  (∀ x > 0, (f a)' x > 0) ↔ a ≤ 0 ∧ (∀ x ∈ Set.Ioi 0, x < 1 / a → (f a)' x > 0) :=
by
  sorry

theorem unique_extreme_point_value (m : ℤ) (x1 x2 : ℝ) :
  (∀ a = 1, ∃ x1 ∈ Set.Ioo 0 1, g 1 x1 = 0 ∧ (∀ x ∈ Set.Ioo 0 x1, g' 1 x < x) ∧ (∀ x ∈ Set.Ioo x1 1, g' 1 x > x)) ∧
  (∃ x2 ∈ Set.Ioo 3 4, g 1 x2 = 0 ∧ (∀ x ∈ Set.Ioo 3 x2, g' 1 x > x) ∧ (∀ x ∈ Set.Ioo x2 4, g' 1 x < x)) →
  m = 0 ∨ m = 3 :=
by
  sorry

end monotonic_intervals_of_f_unique_extreme_point_value_l297_297334


namespace imaginary_part_of_z_l297_297906

def z : ℂ := (1 - complex.i) * complex.i

theorem imaginary_part_of_z : z.im = 1 := by
  sorry

end imaginary_part_of_z_l297_297906


namespace men_population_percentage_l297_297960

theorem men_population_percentage (M : ℕ) (h : ℕ) (h₁ : h = 50) : (2 * h₁ = 100) :=
by
  sorry

end men_population_percentage_l297_297960


namespace cost_of_perfume_l297_297218

-- Definitions and Constants
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def neighbors_yards_mowed : ℕ := 4
def charge_per_yard : ℕ := 5
def dogs_walked : ℕ := 6
def charge_per_dog : ℕ := 2
def additional_amount_needed : ℕ := 6

-- Theorem Statement
theorem cost_of_perfume :
  let christian_earnings := neighbors_yards_mowed * charge_per_yard
  let sue_earnings := dogs_walked * charge_per_dog
  let christian_savings := christian_initial_savings + christian_earnings
  let sue_savings := sue_initial_savings + sue_earnings
  let total_savings := christian_savings + sue_savings
  total_savings + additional_amount_needed = 50 := 
by
  sorry

end cost_of_perfume_l297_297218


namespace ordered_pairs_count_l297_297683

theorem ordered_pairs_count :
  ∃ count : ℕ, count = 29 ∧
  (∀ (m n : ℤ), mn_ge_zero : (m * n ≥ 0), eqn : (m^3 + n^3 + 81 * m * n = 27^3) → count = 29) :=
sorry

end ordered_pairs_count_l297_297683


namespace discount_is_20_percent_l297_297433

noncomputable def discount_percentage 
  (puppy_cost : ℝ := 20.0)
  (dog_food_cost : ℝ := 20.0)
  (treat_cost : ℝ := 2.5)
  (num_treats : ℕ := 2)
  (toy_cost : ℝ := 15.0)
  (crate_cost : ℝ := 20.0)
  (bed_cost : ℝ := 20.0)
  (collar_leash_cost : ℝ := 15.0)
  (total_spent : ℝ := 96.0) : ℝ := 
  let total_cost_before_discount := dog_food_cost + (num_treats * treat_cost) + toy_cost + crate_cost + bed_cost + collar_leash_cost
  let spend_at_store := total_spent - puppy_cost
  let discount_amount := total_cost_before_discount - spend_at_store
  (discount_amount / total_cost_before_discount) * 100

theorem discount_is_20_percent : discount_percentage = 20 := sorry

end discount_is_20_percent_l297_297433


namespace converse_chord_intersect_converse_secant_intersect_l297_297574
-- First Theorem (Converse)

theorem converse_chord_intersect (A B C D M : Point) (h : AM * MB = CM * MD) : 
  is_intersection_point_chords_inside_circle A B C D M := sorry

-- Second Theorem (Converse)

theorem converse_secant_intersect (A B C D M : Point) (h : MA * MB = MC * MD) : 
  is_intersection_point_secants_outside_circle A C M := sorry

end converse_chord_intersect_converse_secant_intersect_l297_297574


namespace toys_in_row_l297_297105

theorem toys_in_row (n_left n_right : ℕ) (hy : 10 = n_left + 1) (hy' : 7 = n_right + 1) :
  n_left + n_right + 1 = 16 :=
by
  -- Fill in the proof here
  sorry

end toys_in_row_l297_297105


namespace inequality_always_true_l297_297562

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_always_true_l297_297562


namespace coconut_to_almond_ratio_l297_297010

-- Conditions
def number_of_coconut_candles (C : ℕ) : Prop :=
  ∃ L A : ℕ, L = 2 * C ∧ A = 10

-- Question
theorem coconut_to_almond_ratio (C : ℕ) (h : number_of_coconut_candles C) :
  ∃ r : ℚ, r = C / 10 := by
  sorry

end coconut_to_almond_ratio_l297_297010


namespace ratio_HD_HA_l297_297398

theorem ratio_HD_HA (a b c : ℝ) (triangle : Triangle a b c) (orthocenter : Point)
  (altitude_AD : Line) (H_on_AD : OnLine orthocenter altitude_AD) 
  (triangle_sides : a = 13 ∧ b = 14 ∧ c = 15 ∧ rightTriangle a b alt):
  ratio_HD_HA (orthocenter AD A) = (5 : ℝ) / (11 : ℝ) :=
sorry

end ratio_HD_HA_l297_297398


namespace find_integer_part_of_m_l297_297524

theorem find_integer_part_of_m {m : ℝ} (h_lecture_duration : m > 0) 
    (h_swap_positions : ∃ k : ℤ, 120 + m = 60 + k * 12 * 60 / 13 ∧ (120 + m) % 60 = 60 * (120 + m) / 720) : 
    ⌊m⌋ = 46 :=
by
  sorry

end find_integer_part_of_m_l297_297524


namespace binary_101_is_5_l297_297654

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l297_297654


namespace max_triangle_area_l297_297707

noncomputable def ellipse := { p : ℝ × ℝ // (p.1^2 / 4) + (p.2^2 / 3) = 1 }

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def triangle_perimeter (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  dist M F1 + dist M F2 + dist F1 F2

def R (M : ℝ × ℝ) : ℝ :=
  dist M F1

def intersects_line (M : ℝ × ℝ) : Prop :=
  (4 - M.1)^2 ≤ (M.1 + 1)^2 + M.2^2

theorem max_triangle_area :
  ∃ M : ellipse, intersects_line (M : ℝ × ℝ) ∧
  (∃ area : ℝ, area = abs (M.2) ∧ area = sqrt 15 / 3) := sorry

end max_triangle_area_l297_297707


namespace computer_program_output_l297_297966

theorem computer_program_output :
  ∃ n : ℕ, let x := 3 + 2 * n in
           let S := n^2 + 4 * n in
           S ≥ 10000 ∧ x = 201 :=
begin
  sorry
end

end computer_program_output_l297_297966


namespace number_of_valid_arrangements_l297_297663

-- Define the set of available numbers
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Predicate to ensure no three consecutive numbers are increasing or decreasing
def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i, i + 2 < arr.length → ¬ (arr[i] < arr[i + 1] ∧ arr[i + 1] < arr[i + 2])
  ∧ ¬ (arr[i] > arr[i + 1] ∧ arr[i + 1] > arr[i + 2])

-- Predicate to ensure even numbers occupy the 2nd, 4th, and 6th positions
def even_positions_correct (arr : List ℕ) : Prop :=
  arr.length = 6
  ∧ arr[1] ∈ [2, 4, 6]
  ∧ arr[3] ∈ [2, 4, 6]
  ∧ arr[5] ∈ [2, 4, 6]

-- Overall predicate combining the conditions
def valid_sequence (arr : List ℕ) : Prop :=
  arr.perm numbers
  ∧ valid_arrangement arr
  ∧ even_positions_correct arr

-- Statement to prove
theorem number_of_valid_arrangements :
  ∃ (num_ways : ℕ), num_ways = 72 ∧ (∃ arrangements, arrangements.length = num_ways ∧ ∀ arr ∈ arrangements, valid_sequence arr) :=
sorry

end number_of_valid_arrangements_l297_297663


namespace max_min_f_range_m_l297_297332

def f (x : ℝ) : ℝ := 4 * (Real.sin (x + Real.pi / 4))^2 - 2 * Real.sqrt 3 * Real.sin (2 * x + Real.pi / 2) - 1

theorem max_min_f (x : ℝ) (h1 : Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) :
  3 ≤ f x ∧ f x ≤ 5 := sorry

theorem range_m (m : ℝ) :
  (∀ x, Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 → |f x - m| < 2) → 3 ≤ m ∧ m ≤ 5 := sorry

end max_min_f_range_m_l297_297332


namespace sum_cubes_mod_6_l297_297643

theorem sum_cubes_mod_6 {b : ℕ → ℕ} (h_seq : ∀ i j, i < j → b i < b j) (h_sum : (∑ i in finset.range 100, b i) = 1000) :
  (∑ i in finset.range 100, (b i)^3) % 6 = 4 :=
by
  sorry

end sum_cubes_mod_6_l297_297643


namespace minimum_moves_to_reset_counters_l297_297927

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l297_297927


namespace mass_calculation_l297_297537

def linear_density (x : ℝ) : ℝ := 2 * x

def mass_of_rod (l : ℝ) : ℝ := ∫ (x : ℝ) in 0..l, linear_density x 

theorem mass_calculation (l : ℝ) : mass_of_rod l = l^2 :=
by 
  sorry

end mass_calculation_l297_297537


namespace width_of_rectangle_l297_297187

-- Define the given values
def length : ℝ := 2
def area : ℝ := 8

-- State the theorem
theorem width_of_rectangle : ∃ width : ℝ, area = length * width ∧ width = 4 :=
by
  -- The proof is omitted
  sorry

end width_of_rectangle_l297_297187


namespace nth_numbers_consecutive_sum_378_sum_2024th_numbers_l297_297475

-- Define sequences
def first_row (n : ℕ) : ℤ := (-3 : ℤ) ^ n
def second_row (n : ℕ) : ℤ := -2 * first_row n
def third_row (n : ℕ) : ℤ := first_row n + 2

-- Questions
theorem nth_numbers (n : ℕ) : 
  first_row n = (-3)^n ∧ 
  second_row n = -2 * (-3)^n ∧ 
  third_row n = (-3)^n + 2 := 
by 
  unfold first_row second_row third_row;
  repeat { split <|> rfl };
  sorry

theorem consecutive_sum_378 :
  ∃ x : ℤ, 
    x + (-3 * x) + (9 * x) = 378 := 
by 
  let x := 54
  use x
  calc 
    x + (-3 * x) + (9 * x) 
    = 54 + (-3 * 54) + (9 * 54) : by rfl
    ... = 54 - 162 + 486 : by norm_num
    ... = 378 : by norm_num

theorem sum_2024th_numbers :
  let x := first_row 2024 in
  let y := second_row 2024 in
  let z := third_row 2024 in
  x + y + z = 2 := 
by 
  let x := first_row 2024
  let y := second_row 2024
  let z := third_row 2024
  calc x + y + z 
    = (-3)^2024 + (-2)*(-3)^2024 + ((-3)^2024 + 2) : by simp [x, y, z, first_row, second_row, third_row]
    ... = (-3)^2024 + 2*(-3)^2024 + 2 : by ring
    ... = 3^2024 + 2 : by ring
    ... = 2 : by sorry -- proof skipped

end nth_numbers_consecutive_sum_378_sum_2024th_numbers_l297_297475


namespace maximum_quizzes_below_A_condition_met_l297_297840

def total_quizzes : ℕ := 60
def goal_percentage : ℝ := 0.85
def goal_quizzes : ℕ := (goal_percentage * total_quizzes).ceil.to_nat -- At least 85% of 60 quizzes
def quizzes_done : ℕ := 40
def quizzes_with_A_done : ℕ := 32
def remaining_quizzes : ℕ := total_quizzes - quizzes_done
def additional_A_quizzes_needed : ℕ := goal_quizzes - quizzes_with_A_done
def max_quizzes_below_A : ℕ := remaining_quizzes - additional_A_quizzes_needed

theorem maximum_quizzes_below_A_condition_met :
  max_quizzes_below_A = 1 :=
sorry

end maximum_quizzes_below_A_condition_met_l297_297840


namespace general_formula_l297_297413

/-
The sequence is defined by:
  a_1 = 5
  a_{n+1} = a_n + 3

We are to prove that the general formula for this sequence is a_n = 3n + 2.
-/

def a : ℕ → ℤ
| 0       := 5 -- By convention of sequence indexing starting at 1, we define a_0 to adjust for Lean's 0 indexing.
| (n + 1) := a n + 3

theorem general_formula (n : ℕ) : a n = 3 * n + 2 :=
sorry

end general_formula_l297_297413


namespace sum_of_remainders_is_9_l297_297687

theorem sum_of_remainders_is_9 (a b c d e : ℕ) :
  a % 13 = 3 → b % 13 = 5 → c % 13 = 7 → d % 13 = 9 → e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by {
  intros ha hb hc hd he,
  sorry
}

end sum_of_remainders_is_9_l297_297687


namespace lauren_mail_total_l297_297443

theorem lauren_mail_total :
  let mail_monday := 65
  let mail_tuesday := mail_monday + 10
  let mail_wednesday := mail_tuesday - 5
  let mail_thursday := mail_wednesday + 15
  mail_monday + mail_tuesday + mail_wednesday + mail_thursday = 295 :=
by
  let mail_monday := 65
  let mail_tuesday := mail_monday + 10
  let mail_wednesday := mail_tuesday - 5
  let mail_thursday := mail_wednesday + 15
  have h : mail_monday + mail_tuesday + mail_wednesday + mail_thursday = 65 + 75 + 70 + 85
  exact h
  have : 65 + 75 + 70 + 85 = 295 := by norm_num
  rw this
  exact rfl

end lauren_mail_total_l297_297443


namespace megan_folders_l297_297040

theorem megan_folders (files_initial : ℝ) (files_added : ℝ) (files_per_folder : ℝ) (files_total : ℝ) (folders : ℝ)
  (h1 : files_initial = 93.5)
  (h2 : files_added = 21.25)
  (h3 : files_per_folder = 8.75)
  (h4 : files_total = files_initial + files_added) :
  folders = Real.ceil (files_total / files_per_folder) :=
by
  have files_initial_def : files_initial = 93.5, from h1
  have files_added_def : files_added = 21.25, from h2
  have files_per_folder_def : files_per_folder = 8.75, from h3
  have files_total_def : files_total = files_initial + files_added, from h4
  sorry

end megan_folders_l297_297040


namespace minimum_cells_to_prevent_L_shapes_l297_297558

-- Definitions of the grid and conditions
structure Grid (n : ℕ) :=
  (cells : ℕ → ℕ → Prop)  -- Grid cells identifier

def is_L_shape {n : ℕ} (grid : Grid n) (x y : ℕ) : Prop :=
  (grid.cells x y ∧ grid.cells (x + 1) y ∧ grid.cells x (y + 1)) ∨
  (grid.cells x y ∧ grid.cells (x + 1) y ∧ grid.cells (x + 1) (y + 1)) ∨
  (grid.cells x y ∧ grid.cells x (y + 1) ∧ grid.cells (x + 1) (y + 1)) ∨
  (grid.cells (x + 1) y ∧ grid.cells (x + 1) (y + 1) ∧ grid.cells x (y + 1))

def prevents_L_shapes {n : ℕ} (grid : Grid n) (red_cells : ℕ → ℕ → Prop) : Prop :=
  ∀ x y, ¬is_L_shape grid x y ∨ red_cells x y ∨ red_cells (x + 1) y ∨ red_cells x (y + 1) ∨ red_cells (x + 1) (y + 1)

-- The main theorem statement
theorem minimum_cells_to_prevent_L_shapes :
  ∃ (red_cells : ℕ → ℕ → Prop), prevents_L_shapes (Grid.mk (λ x y, x < 4 ∧ y < 4)) red_cells ∧
  (∃ (count : ℕ), count = 3 ∧ (∀ (i j : ℕ), red_cells i j → count = count - 1)) :=
sorry

end minimum_cells_to_prevent_L_shapes_l297_297558


namespace possible_values_of_inverse_sum_l297_297814

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end possible_values_of_inverse_sum_l297_297814


namespace multiplying_by_fraction_equivalent_l297_297950

theorem multiplying_by_fraction_equivalent (x : ℚ) :
  (x * (4/5)) / (2/7) = x * (14/5) :=
by
  calc
    (x * (4/5)) / (2/7)
        = x * (4/5) * (7/2) : by rw [div_eq_mul_inv, rat.inv_def (2/7)]
    ... = x * ((4/5) * (7/2)) : by rw mul_assoc
    ... = x * (28/10) : by rw mul_div_cancel'
    ... = x * (14/5) : by norm_cast; linarith

end multiplying_by_fraction_equivalent_l297_297950


namespace base7_to_base10_321_is_162_l297_297996

-- Define the conversion process from a base-7 number to base-10
def convert_base7_to_base10 (n: ℕ) : ℕ :=
  3 * 7^2 + 2 * 7^1 + 1 * 7^0

theorem base7_to_base10_321_is_162 :
  convert_base7_to_base10 321 = 162 :=
by
  sorry

end base7_to_base10_321_is_162_l297_297996


namespace canoe_kayak_rental_l297_297560

theorem canoe_kayak_rental:
  ∀ (C K : ℕ), 
    12 * C + 18 * K = 504 → 
    C = (3 * K) / 2 → 
    C - K = 7 :=
  by
    intro C K
    intros h1 h2
    sorry

end canoe_kayak_rental_l297_297560


namespace faucets_filling_time_l297_297277

theorem faucets_filling_time (rate_per_faucet : ℚ) (tub1_vol : ℚ) (tub2_vol : ℚ) (time1 : ℚ) : 
  (4 : ℚ) * rate_per_faucet * time1 = tub1_vol →
  rate_per_faucet = tub1_vol / (4 * time1) →
  (8 : ℚ) * rate_per_faucet = 50 / ((4 : ℚ) / 3) →
  tub2_vol = 50 →
  sorry :=
begin
  sorry
end

end faucets_filling_time_l297_297277


namespace polynomial_identity_l297_297504

theorem polynomial_identity (x : ℂ) (h₁ : x ^ 2019 - 3 * x + 1 = 0) (h₂ : x ≠ 1) :
  x ^ 2018 + x ^ 2017 + ... + x + 1 = 3 :=
sorry

end polynomial_identity_l297_297504


namespace ternary_to_decimal_decimal_to_base_seven_l297_297968

-- Question 1: Convert the ternary number $10212_{(3)}$ to decimal.
theorem ternary_to_decimal :
  let n := 10212
  in n = 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 1 * 3^1 + 2 * 3^0 → n = 104 := by
  sorry

-- Question 2: Convert the decimal number $1234$ to base seven.
theorem decimal_to_base_seven :
  let n := 1234
  in nat.to_digits 7 n = [3, 4, 1, 2] := by
  sorry

end ternary_to_decimal_decimal_to_base_seven_l297_297968


namespace maxim_is_correct_l297_297153

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l297_297153


namespace john_order_cost_l297_297603

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end john_order_cost_l297_297603


namespace minimum_moves_to_reset_counters_l297_297926

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l297_297926


namespace trisecting_points_abcircle_l297_297664

open Set EuclideanGeometry Real

theorem trisecting_points_abcircle (O A B C D E F : Point) (r : ℝ) :
  -- Condition: O is the center of the circle
  let circle := Circle O r in
  A ∈ circle ∧ B ∈ circle ∧
  -- Condition: Points dividing the larger arc AB into three equal parts are C and D
  let arcAB := largerArc A B circle in
  C ∈ arcAB ∧ D ∈ arcAB ∧ isTrisectionPoints arcAB C D ∧
  -- Condition: The intersection points of lines extended from OC and OD with AB are E and F respectively
  onLine (rayThrough O C) E ∧ onLine (rayThrough O D) F ∧
  intersectAB (lineThrough E F) A B ∧
  -- Given condition EA + BF + AB = 3AB
  length (segment E A) + length (segment B F) + length (segment A B) = 3 * length (segment A B)
  :=
sorry -- Proof omitted

end trisecting_points_abcircle_l297_297664


namespace isosceles_trapezoid_area_l297_297210

theorem isosceles_trapezoid_area
  (legs : ℝ)
  (diagonals : ℝ)
  (longer_base : ℝ)
  (shorter_base : ℝ)
  (height : ℝ) :
  legs = 40 →
  diagonals = 50 →
  longer_base = 60 →
  shorter_base = longer_base - 2 * (√(legs^2 - (height^2))) →
  height = ((longer_base * diagonals * legs * 0.5) / height) →
  0.5 * (longer_base + shorter_base) * height = 54000 / 3 :=
begin
  intros,
  sorry
end

end isosceles_trapezoid_area_l297_297210


namespace find_number_of_hens_l297_297956

theorem find_number_of_hens
  (H C : ℕ)
  (h1 : H + C = 48)
  (h2 : 2 * H + 4 * C = 140) :
  H = 26 :=
by
  sorry

end find_number_of_hens_l297_297956


namespace original_average_of_five_numbers_l297_297891

theorem original_average_of_five_numbers 
  (A : ℝ) 
  (H1 : ∀ (nums : list ℝ), nums.length = 5 → list.average nums = A) 
  (H2 : ∃ (nums : list ℝ), nums.length = 5 ∧ 6 ∈ nums) 
  (H3 : ∀ (nums : list ℝ), nums.length = 5 → 6 ∈ nums → nums.map (λ x, if x = 6 then 18 else x)).average = 9.2 
  : A = 6.8 := 
sorry

end original_average_of_five_numbers_l297_297891


namespace julie_pulled_weeds_3_hours_in_september_l297_297805

variable (W : ℕ)

theorem julie_pulled_weeds_3_hours_in_september
  (h1 : ∀ t : ℕ, t > 0 → W = t ∧ 200 + 16 * t = 248) :
  W = 3 :=
by
  have h2 : 200 + 16 * W = 248 := by sorry
  have h3 : 16 * W = 48 := by sorry
  have h4 : W = 48 / 16 := by sorry
  have h5 : 48 / 16 = 3 := by sorry
  exact h5

end julie_pulled_weeds_3_hours_in_september_l297_297805


namespace running_time_of_BeastOfWar_is_100_l297_297910

noncomputable def Millennium := 120  -- minutes
noncomputable def AlphaEpsilon := Millennium - 30  -- minutes
noncomputable def BeastOfWar := AlphaEpsilon + 10  -- minutes
noncomputable def DeltaSquadron := 2 * BeastOfWar  -- minutes

theorem running_time_of_BeastOfWar_is_100 :
  BeastOfWar = 100 :=
by
  -- Proof goes here
  sorry

end running_time_of_BeastOfWar_is_100_l297_297910


namespace cos_angle_between_line_and_plane_l297_297451

noncomputable def cos_theta : ℝ :=
  let d : ℝ × ℝ × ℝ := (4, 2, 5) in
  let n : ℝ × ℝ × ℝ := (8, 4, -1) in
  let dot_product := d.1 * n.1 + d.2 * n.2 + d.3 * n.3 in
  let magnitude_d := Real.sqrt (d.1^2 + d.2^2 + d.3^2) in
  let magnitude_n := Real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  dot_product / (magnitude_d * magnitude_n)

theorem cos_angle_between_line_and_plane :
  cos_theta = (7 * Real.sqrt 5) / 45 :=
by
  sorry

end cos_angle_between_line_and_plane_l297_297451


namespace odd_factors_252_l297_297359

theorem odd_factors_252 : 
  {n : ℕ | n ∈ finset.filter (λ d, d % 2 = 1) (finset.divisors 252)}.card = 6 := 
sorry

end odd_factors_252_l297_297359


namespace not_a_quadratic_radical_l297_297136

-- Definitions for the given options
def optionA (x : ℝ) : ℝ := real.sqrt (x^2 + 1)
def optionB : ℝ := real.sqrt (-4)
def optionC : ℝ := real.sqrt 0
def optionD (a b : ℝ) : ℝ := real.sqrt ((a - b)^2)

-- Definition of a quadratic radical (for the purpose of this problem)
def is_quadratic_radical (r : ℝ) : Prop :=
  ∃ x : ℝ, r = real.sqrt x ∧ x ≥ 0

-- The main statement asserting which option is not a quadratic radical
theorem not_a_quadratic_radical (x a b : ℝ) : 
  ¬ is_quadratic_radical optionB :=
by
  sorry

end not_a_quadratic_radical_l297_297136


namespace find_b_minus_a_l297_297095

def rotation (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := c
  (2 * h - x, 2 * k - y)

def reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  reflection (rotation p c)

def final_position (a b : ℝ) := (a, b)

theorem find_b_minus_a (a b : ℝ) 
  (h : transformations (a, b) (2, 3) = (1, -4)) :
  b - a = -3 :=
by
  sorry

end find_b_minus_a_l297_297095


namespace fill_time_of_cistern_l297_297982

-- Define the condition for the rates based on the given problem
def fill_rate_no_leak (t : ℝ) : ℝ := 1 / t
def fill_rate_with_leak (t : ℝ) : ℝ := 1 / (t + 2)
def leak_rate : ℝ := 1 / 60

-- State the theorem to be proved
theorem fill_time_of_cistern : ∃ t : ℝ, fill_rate_no_leak t - leak_rate = fill_rate_with_leak t ∧ t = 10 :=
  by sorry

end fill_time_of_cistern_l297_297982


namespace sergio_more_correct_than_sylvia_l297_297506

theorem sergio_more_correct_than_sylvia :
  let total_questions := 50
  let incorrect_sylvia := total_questions / 5
  let incorrect_sergio := 4
  let correct_sylvia := total_questions - incorrect_sylvia
  let correct_sergio := total_questions - incorrect_sergio
  in correct_sergio - correct_sylvia = 6 := 
by {
  -- Definitions
  let total_questions := 50,
  let incorrect_sylvia := total_questions / 5,
  let incorrect_sergio := 4,
  let correct_sylvia := total_questions - incorrect_sylvia,
  let correct_sergio := total_questions - incorrect_sergio,

  -- Proof
  have h1 : incorrect_sylvia = 10 := by norm_num,
  have h2 : correct_sylvia = 40 := by norm_num,
  have h3 : correct_sergio = 46 := by norm_num,
  show correct_sergio - correct_sylvia = 6,
  calc
    correct_sergio - correct_sylvia = 46 - 40     : by congr
                         ...              = 6     : by norm_num,
}

end sergio_more_correct_than_sylvia_l297_297506


namespace right_triangle_med_ad_l297_297002

theorem right_triangle_med_ad (BC : ℝ) (BC_val : BC = 10) (AD : ℝ) (AD_val : AD = 6) 
  (BD CD : ℝ) (BD_val : BD = 5) (CD_val : CD = 5) (right_angle : ∠ BAC = 90) :
  let M := (AB^2 + AC^2) in 
  let m := (AB^2 + AC^2) in 
  M - m = 0 :=
by
  sorry

end right_triangle_med_ad_l297_297002


namespace odd_factors_252_l297_297357

theorem odd_factors_252 : 
  {n : ℕ | n ∈ finset.filter (λ d, d % 2 = 1) (finset.divisors 252)}.card = 6 := 
sorry

end odd_factors_252_l297_297357


namespace shortest_side_of_triangle_l297_297050

noncomputable def triangle_shortest_side_length (a b r : ℝ) (shortest : ℝ) : Prop :=
a = 8 ∧ b = 6 ∧ r = 4 ∧ shortest = 12

theorem shortest_side_of_triangle 
  (a b r shortest : ℝ) 
  (h : triangle_shortest_side_length a b r shortest) : shortest = 12 :=
sorry

end shortest_side_of_triangle_l297_297050


namespace cosine_angle_ABC_l297_297251

def point3d := (ℝ × ℝ × ℝ)

def A : point3d := (0, 1, -2)
def B : point3d := (3, 1, 2)
def C : point3d := (4, 1, 1)

def vector_sub (p1 p2 : point3d) := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def dot_product (v1 v2 : point3d) := (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3)

def magnitude (v : point3d) := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def cosine_between_vectors (v1 v2 : point3d) :=
  (dot_product v1 v2) / ((magnitude v1) * (magnitude v2))

theorem cosine_angle_ABC :
  let AB := vector_sub A B in
  let AC := vector_sub A C in
  cosine_between_vectors AB AC = 0.96 :=
by
  sorry

end cosine_angle_ABC_l297_297251


namespace equation_of_line_through_origin_equation_of_line_perpendicular_l297_297736

noncomputable def intersection_point (l1 l2 : ℕ → ℝ → ℕ) : (ℝ × ℝ) :=
(l1, l2) → ℝ × ℝ:=
∃ x y, x + y - 2 = 0 ∧ 2 * x + y + 2 = 0

noncomputable def line_through_origin (P : ℝ × ℝ) :=
let (x, y) := P in 3 * x + 2 * y = 0

noncomputable def line_perpendicular_through_P (P : ℝ × ℝ) :=
let (x, y) := P in 3 * x + y + 6 = 0

theorem equation_of_line_through_origin : intersection_point (λ x y, x + y - 2 = 0) (λ x y, 2 * x + y + 2 = 0) = (-4, 6) → 
  line_through_origin (-4, 6) :=
begin
  sorry
end

theorem equation_of_line_perpendicular : intersection_point (λ x y, x + y - 2 = 0) (λ x y, 2 * x + y + 2 = 0) = (-4, 6) → 
  line_perpendicular_through_P (-4, 6) := 
begin
  sorry
end

end equation_of_line_through_origin_equation_of_line_perpendicular_l297_297736


namespace steve_total_time_on_roads_l297_297517

variables (d : ℝ) (v_back : ℝ) (v_to_work : ℝ)

-- Constants from the problem statement
def distance := 10 -- The distance from Steve's house to work is 10 km
def speed_back := 5 -- Steve's speed on the way back from work is 5 km/h

-- Given conditions
def speed_to_work := speed_back / 2 -- On the way back, Steve drives twice as fast as he did on the way to work

-- Define the time to get to work and back
def time_to_work := distance / speed_to_work
def time_back_home := distance / speed_back

-- Total time on roads
def total_time := time_to_work + time_back_home

-- The theorem to prove
theorem steve_total_time_on_roads : total_time = 6 := by
  -- Proof here
  sorry

end steve_total_time_on_roads_l297_297517


namespace find_angle_CAD_l297_297773

-- Definitions representing the problem conditions
variables {A B C D : Type} [has_angle A] [has_angle B] [has_angle C] [has_angle D]

-- Given conditions
def is_convex (A B C D : Type) : Prop := sorry  -- convex quadrilateral ABCD
def angle_A (A : Type) : ℝ := 65
def angle_B (B : Type) : ℝ := 80
def angle_C (C : Type) : ℝ := 75
def equal_sides (A B D : Type) : Prop := sorry  -- AB = BD

-- Theorem statement: Given the conditions, prove the desired angle
theorem find_angle_CAD {A B C D : Type}
  [is_convex A B C D] 
  [equal_sides A B D]
  (h1 : angle_A A = 65)
  (h2 : angle_B B = 80)
  (h3 : angle_C C = 75)
  : ∃ (CAD : ℝ), CAD = 15 := 
sorry -- proof omitted

end find_angle_CAD_l297_297773


namespace tiling_replacement_impossible_l297_297600

theorem tiling_replacement_impossible (m n : ℕ) (h1 : m * n % 2 = 0) (h2 : m * n ≥ 4) :
  ¬ exists (t : ℕ), (t = 1 ∨ t = 2) ∧ ∃ f : ℕ → ℕ → bool, 
      (∀ i j, 
        (0 < i ∧ i ≤ m ∧ 0 < j ∧ j ≤ n → 
          (f i j = (if t = 1 then (i + j) % 2 = 0 else i % 2 = 0 ∧ j % 2 = 0))) ∧
      (if t = 1 
        then (∃ i₀ j₀, 0 < i₀ ∧ i₀ ≤ m ∧ 0 < j₀ ∧ j₀ ≤ n ∧ f i₀ j₀ ≠ f (i₀ + 1) j₀ ∧ f i₀ j₀ ≠ f i₀ (j₀ + 1))
        else (∃ i₀ j₀, 0 < i₀ ∧ i₀ ≤ m ∧ 0 < j₀ ∧ j₀ ≤ n ∧ f i₀ j₀ ≠ f (i₀ + 1) j₀ ∧ f i₀ j₀ = f i₀ (j₀ + 1))
      )
  )

end tiling_replacement_impossible_l297_297600


namespace part_I_part_II_l297_297338

noncomputable def f (x m : ℝ) : ℝ := |3 * x + m|
noncomputable def g (x m : ℝ) : ℝ := f x m - 2 * |x - 1|

theorem part_I (m : ℝ) : (∀ x : ℝ, (f x m - m ≤ 9) ↔ (-1 ≤ x ∧ x ≤ 3)) → m = -3 :=
by
  sorry

theorem part_II (m : ℝ) (h : m > 0) : (∃ A B C : ℝ × ℝ, 
  let A := (-m-2, 0)
  let B := ((2-m)/5, 0)
  let C := (-m/3, -2*m/3-2)
  let Area : ℝ := 1/2 * |(B.1 - A.1) * (C.2 - 0) - (B.2 - A.2) * (C.1 - A.1)|
  Area > 60 ) → m > 12 :=
by
  sorry

end part_I_part_II_l297_297338


namespace ellie_mowing_time_correct_l297_297238

noncomputable def ellie_mowing_time (lawn_length lawn_width swath_width overlap: ℝ) (speed_long speed_short: ℝ) : ℝ :=
  let effective_swath_width := (swath_width - overlap) / 12
  let number_of_strips := lawn_width / effective_swath_width
  let strip_length := lawn_length
  let total_distance := number_of_strips * strip_length
  let speed := if strip_length > 50 then speed_long else speed_short
  total_distance / speed

theorem ellie_mowing_time_correct :
  ellie_mowing_time 60 120 30 6 4000 5000 = 0.9 :=
by
  rw [ellie_mowing_time]
  norm_num
  sorry

end ellie_mowing_time_correct_l297_297238


namespace rem_calc_final_value_l297_297657

def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

theorem rem_calc :
  rem (5 / 7) (3 / 4) = 5 / 7 :=
by 
  sorry

theorem final_value :
  -2 * rem (5 / 7) (3 / 4) = -10 / 7 :=
by
  rw rem_calc
  norm_num
  sorry

end rem_calc_final_value_l297_297657


namespace expression_for_f_pos_f_monotone_on_pos_l297_297831

section

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_neg : ∀ x, -1 ≤ x ∧ x < 0 → f x = 2 * x + 1 / x^2)

-- Part 1: Prove the expression for f(x) when x ∈ (0,1]
theorem expression_for_f_pos (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  f x = 2 * x - 1 / x^2 :=
sorry

-- Part 2: Prove the monotonicity of f(x) on (0,1]
theorem f_monotone_on_pos : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f x < f y :=
sorry

end

end expression_for_f_pos_f_monotone_on_pos_l297_297831


namespace sequence_sum_32_l297_297740

theorem sequence_sum_32 :
  ∀ (a : ℕ → ℝ) (n : ℕ), 
  a 1 = 2 → 
  (∀ n : ℕ, n > 0 → (a (n + 1))^2 + (a n)^2 = 2 * (a (n + 1)) * (a n)) → 
  ∑ k in (finset.range 32).map (λ x, x + 1), a k = 64 :=
by
  intros a n ha1 hrec
  sorry

end sequence_sum_32_l297_297740


namespace coeff_of_x3_in_sum_of_expansions_l297_297579

theorem coeff_of_x3_in_sum_of_expansions :
  (∑ k in {5, 6, 7, 8}, (-1 : ℤ) ^ 3 * Nat.choose k 3) = -121 := 
by 
  simp [Finset.sum], sorry

end coeff_of_x3_in_sum_of_expansions_l297_297579


namespace mod_inverse_sum_l297_297128

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end mod_inverse_sum_l297_297128


namespace speed_of_stream_is_correct_l297_297145

-- Define the downstream speed
def downstream_speed : ℝ := 14

-- Define the upstream speed
def upstream_speed : ℝ := 8

-- Define the speed of the stream based on the given conditions
def speed_of_stream : ℝ := (downstream_speed - upstream_speed) / 2

-- The theorem statement to prove that the speed of the stream is 3 kmph
theorem speed_of_stream_is_correct : speed_of_stream = 3 := 
by
  sorry

end speed_of_stream_is_correct_l297_297145


namespace system_of_equations_solution_exists_l297_297878

theorem system_of_equations_solution_exists :
  ∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    (x = 3 ∧ y = 2021 ∧ z = 4 ∨ 
    x = -1 ∧ y = 2019 ∧ z = -2) := 
sorry

end system_of_equations_solution_exists_l297_297878


namespace ellipse_standard_equation_l297_297520

theorem ellipse_standard_equation (c a : ℝ) (h1 : 2 * c = 8) (h2 : 2 * a = 10) : 
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ ( ( ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) ∨ ( ∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ) )) :=
by
  sorry

end ellipse_standard_equation_l297_297520


namespace touchdowns_points_l297_297076

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l297_297076


namespace find_parallelogram_angles_l297_297892

variables (A B C D M : ℝ)
variables (parallelogram : Prop) (angle_bisector_A : Prop) (angle_bisector_AMC : Prop)
variables (angle_MDC : ℝ) [Angle_MDC_is_45 : angle_MDC = 45]

theorem find_parallelogram_angles
  (h_parallelogram : parallelogram)
  (h_angle_bisector_A : angle_bisector_A)
  (h_angle_bisector_AMC : angle_bisector_AMC)
  (h_angle_MDC : angle_MDC = 45):
  ∠A = 60 ∧ ∠B = 120 :=
sorry

end find_parallelogram_angles_l297_297892


namespace feet_of_altitudes_l297_297824

variables {A B C M_A M_B M_C H_A H_B H_C : Type} 
[AddGroup M_A] [AddGroup M_B] [AddGroup M_C]
[AddGroup H_A] [AddGroup H_B] [AddGroup H_C]
 
-- Definitions for midpoints M_A, M_B, M_C of the scalene triangle ABC
def is_midpoint (p m q : M_A) : Prop := p + q = 2 * m

-- Conditions
def conditions :=
(is_midpoint A M_A B) ∧ 
(is_midpoint B M_B C) ∧ 
(is_midpoint C M_C A) ∧ 
(distance M_A H_B = distance M_A H_C) ∧ 
(distance M_B H_A = distance M_B H_C) ∧ 
(distance M_C H_A = distance M_C H_B)

-- The theorem to prove
theorem feet_of_altitudes (h : conditions) : 
  is_foot_of_altitude A H_A ∧ 
  is_foot_of_altitude B H_B ∧ 
  is_foot_of_altitude C H_C :=
sorry

end feet_of_altitudes_l297_297824


namespace incorrect_statement_C_l297_297564

theorem incorrect_statement_C : 
  (∀ x : ℝ, |x| = x → x = 0 ∨ x = 1) ↔ False :=
by
  -- Proof goes here
  sorry

end incorrect_statement_C_l297_297564


namespace A_plus_B_plus_C_l297_297521

def g (x : ℝ) (A B C : ℤ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem A_plus_B_plus_C (A B C : ℤ) (h1 : ∀ x > 3, g x A B C > 0.5) : A + B + C = -8 :=
begin
  sorry
end

end A_plus_B_plus_C_l297_297521


namespace not_universally_better_l297_297839

-- Definitions based on the implicitly given conditions
def can_show_quantity (chart : Type) : Prop := sorry
def can_reflect_changes (chart : Type) : Prop := sorry

-- Definitions of bar charts and line charts
inductive BarChart
| mk : BarChart

inductive LineChart
| mk : LineChart

-- Assumptions based on characteristics of the charts
axiom bar_chart_shows_quantity : can_show_quantity BarChart 
axiom line_chart_shows_quantity : can_show_quantity LineChart 
axiom line_chart_reflects_changes : can_reflect_changes LineChart 

-- Proof problem statement
theorem not_universally_better : ¬(∀ (c1 c2 : Type), can_show_quantity c1 → can_reflect_changes c1 → ¬can_show_quantity c2 → ¬can_reflect_changes c2) :=
  sorry

end not_universally_better_l297_297839


namespace total_cost_of_glasses_l297_297008

theorem total_cost_of_glasses
    (frames_cost : ℕ)
    (lenses_cost : ℕ)
    (insurance_coverage_percent : ℝ)
    (coupon : ℕ)
    (insurance_coverage : ℕ := (insurance_coverage_percent * lenses_cost).to_nat)
    (frames_cost_after_coupon : ℕ := frames_cost - coupon)
    (lenses_cost_after_insurance : ℕ := lenses_cost - insurance_coverage)
    (total_cost : ℕ := frames_cost_after_coupon + lenses_cost_after_insurance)
    (h1 : frames_cost = 200)
    (h2 : lenses_cost = 500)
    (h3 : insurance_coverage_percent = 0.80)
    (h4 : coupon = 50)
    (h5 : insurance_coverage = 400)
    (h6 : frames_cost_after_coupon = 150)
    (h7 : lenses_cost_after_insurance = 100)
    (h8 : total_cost = 250) : 
    total_cost = 250 := 
by 
  sorry

end total_cost_of_glasses_l297_297008


namespace depth_of_first_hole_l297_297969

theorem depth_of_first_hole (n1 t1 n2 t2 : ℕ) (D : ℝ) (r : ℝ) 
  (h1 : n1 = 45) (h2 : t1 = 8) (h3 : n2 = 90) (h4 : t2 = 6) 
  (h5 : r = 1 / 12) (h6 : D = n1 * t1 * r) (h7 : n2 * t2 * r = 45) : 
  D = 30 := 
by 
  sorry

end depth_of_first_hole_l297_297969


namespace grain_to_rice_system_l297_297797

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l297_297797


namespace sum_transformed_roots_l297_297899

theorem sum_transformed_roots :
  ∀ (a b c : ℝ),
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by
  intros a b c h_bounds h_poly_a h_poly_b h_poly_c h_distinct
  sorry

end sum_transformed_roots_l297_297899


namespace verify_points_location_l297_297301

def point := (ℝ, ℝ)

def A : point := (0, 0)
def B : point := (-2, 1)
def C : point := (3, 3)
def D : point := (2, -1)

def circle_center : point := (1, -3)
def circle_radius_squared : ℝ := 25

def distance_squared (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def point_relation_to_circle (p : point) : string :=
  if distance_squared p circle_center < circle_radius_squared then "inside"
  else if distance_squared p circle_center = circle_radius_squared then "on"
  else "outside"

theorem verify_points_location :
  point_relation_to_circle A = "inside" ∧
  point_relation_to_circle B = "on" ∧
  point_relation_to_circle C = "outside" ∧
  point_relation_to_circle D = "inside" :=
by
  sorry

end verify_points_location_l297_297301


namespace generate_sequence_next_three_members_l297_297089

-- Define the function that generates the sequence
def f (n : ℕ) : ℕ := 2 * (n + 1) ^ 2 * (n + 2) ^ 2

-- Define the predicate that checks if a number can be expressed as the sum of squares of two positive integers
def is_sum_of_squares_of_two_positives (k : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = k

-- The problem statement to prove the equivalence
theorem generate_sequence_next_three_members :
  is_sum_of_squares_of_two_positives (f 1) ∧
  is_sum_of_squares_of_two_positives (f 2) ∧
  is_sum_of_squares_of_two_positives (f 3) ∧
  is_sum_of_squares_of_two_positives (f 4) ∧
  is_sum_of_squares_of_two_positives (f 5) ∧
  is_sum_of_squares_of_two_positives (f 6) ∧
  f 1 = 72 ∧
  f 2 = 288 ∧
  f 3 = 800 ∧
  f 4 = 1800 ∧
  f 5 = 3528 ∧
  f 6 = 6272 :=
sorry

end generate_sequence_next_three_members_l297_297089


namespace AP_sum_equal_zero_l297_297399

-- Definitions of sums in an arithmetic progression
def AP_sum (a₁ d : ℝ) (k : ℕ) : ℝ :=
  (k / 2) * (2 * a₁ + d * (k - 1))

-- Statement of the problem using the conditions and the required proof
theorem AP_sum_equal_zero
  (a₁ d : ℝ) (m n : ℕ)
  (hneq : m ≠ n)
  (h_eq : AP_sum a₁ d m = AP_sum a₁ d n) :
  AP_sum a₁ d (m+n) = 0 :=
sorry

end AP_sum_equal_zero_l297_297399


namespace similar_triangles_leg_length_l297_297189

theorem similar_triangles_leg_length (y : ℝ) 
  (triangle1_leg1 : ℝ := 15) (triangle1_leg2 : ℝ := 12)
  (triangle2_leg2 : ℝ := 9) 
  (similarity_condition : triangle1_leg1 / y = triangle1_leg2 / triangle2_leg2) :
  y ≈ 11.25 :=
by
  sorry

end similar_triangles_leg_length_l297_297189


namespace necessary_sufficient_condition_l297_297688

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + (4 - 2 * a)

theorem necessary_sufficient_condition (a : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) : 
  (∀ (x : ℝ), f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end necessary_sufficient_condition_l297_297688


namespace cost_of_one_dozen_pens_l297_297895

theorem cost_of_one_dozen_pens (pen pencil : ℝ) (h_ratios : pen = 5 * pencil) (h_total : 3 * pen + 5 * pencil = 240) :
  12 * pen = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l297_297895


namespace largest_possible_degree_q_l297_297905

-- Representation of the problem conditions
def denominator := 2 * X^5 + X^4 - 7 * X^2 + 1

-- The theorem to prove the largest possible degree of q(x) for the function to have a horizontal asymptote
theorem largest_possible_degree_q (q : ℤ[X]) (hq : ∀ x, X ∈ q.eval x) : degree q ≤ degree denominator → degree q = 5 :=
by
  sorry

end largest_possible_degree_q_l297_297905


namespace initial_quantity_of_A_l297_297569

theorem initial_quantity_of_A
  (A B : ℝ) -- A is the initial quantity of liquid A, and B is the initial quantity of liquid B
  (h_ratio : A / B = 7 / 5) -- Initial ratio of A to B is 7:5
  (h_remove : 9 = 7/12 * A + 5/12 * B) -- 9 liters of mixture (in ratio 7:5) is removed
  (h_new_ratio : (A - 7/12 * A) / (B - 5/12 * B + 9) = 7 / 9) -- New ratio of A to B is 7:9 after removing 9 liters and adding 9 liters of B
  : A = 21 :=
begin
  sorry
end

end initial_quantity_of_A_l297_297569


namespace compare_fractions_l297_297221

theorem compare_fractions (a b c d : ℤ) (h1 : a = -(2)) (h2 : b = 3) (h3 : c = -(3)) (h4 : d = 5) :
  (a : ℚ) / b < c / d := 
by {
  simp [h1, h2, h3, h4],
  norm_num,
  simp [rat.lt_iff],
  exact lt_trans (neg_lt_zero.mpr (nat.cast_pos.mpr (show 3 * 5 > 0, by norm_num))) (show -(2 * 5 : ℤ) < -(3 * 3), by norm_num [lt_one_mul]),
}

end compare_fractions_l297_297221


namespace min_circles_in_square_l297_297602

theorem min_circles_in_square (N : ℕ) (side_length radius segment_length : ℝ) 
  (square : set (ℝ × ℝ)) (circles : set (set (ℝ × ℝ))) 
  (covers_segment : ∀ (s : set (ℝ × ℝ)), 
    (∀ (a b : ℝ × ℝ), (a ∈ s ∧ b ∈ s ∧ dist a b = segment_length) → 
      (∃ c ∈ circles, s ∩ c ≠ ∅)) → 
    (∀ (a b : ℝ × ℝ), (a ∈ square ∧ b ∈ square ∧ dist a b = segment_length) 
      → (∃ c ∈ circles, line_segment a b ∩ c ≠ ∅))) :
  (∀ c ∈ circles, ∃ center : ℝ × ℝ, ∃ r : ℝ, r = radius) → 
  (side_length = 100) → (radius = 1) → (segment_length = 10) →
  (measure square = 100 * 100) →
  N ≥ 400 :=
by
  sorry

end min_circles_in_square_l297_297602


namespace original_rice_amount_l297_297794

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l297_297794


namespace polar_equation_of_curve_and_segment_length_l297_297791

theorem polar_equation_of_curve_and_segment_length :
  (∀ (x y ρ θ : ℝ), 
    ((x - 1)^2 + y^2 = 1 ∧ x = ρ * cos θ ∧ y = ρ * sin θ → ρ = 2 * cos θ)) ∧
  (∀(ρ θ L : ℝ),
    (2 * ρ * sin (θ + π / 3) + 3 * real.sqrt 3 = 0 ∧ θ = π / 3 ∧ L = |ρ - (-3)| 
    → L = 4)) := 
by
  sorry

end polar_equation_of_curve_and_segment_length_l297_297791


namespace xiaoming_problems_per_day_l297_297970

/-- Define statements for the conditions -/
def initial_problems_xiaoqiang : ℕ := 60
def days_before_school_start : ℕ := 6
def x (num_problems_per_day_xiaoqiang : ℕ) : Prop := num_problems_per_day_xiaoqiang = 5
def problems_per_day_xiaoming (num_problems_per_day_xiaoqiang : ℕ) : ℕ := 3 * num_problems_per_day_xiaoqiang

/-- Main theorem to prove the number of problems Xiaoming completes per day on average -/
theorem xiaoming_problems_per_day :
  ∀ (num_problems_per_day_xiaoqiang : ℕ),
  x num_problems_per_day_xiaoqiang →
  (problems_per_day_xiaoming num_problems_per_day_xiaoqiang) = 15 :=
begin
  intros num_problems_per_day_xiaoqiang h,
  rw h,
  unfold problems_per_day_xiaoming,
  exact rfl,
end

end xiaoming_problems_per_day_l297_297970


namespace number_of_solutions_l297_297662

open Real

theorem number_of_solutions (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) :
  ∃ count : ℕ, count = 4 ∧ 
  ∀ x : ℝ, a ≤ x ∧ x ≤ b → 
  (sin ((π / 2) * cos x) + x = cos ((π / 2) * sin x) + x → 
  ∃ xs : List ℝ, xs.nodup ∧ xs.length = count 
  ∧ ∀ y ∈ xs, a ≤ y ∧ y ≤ b ∧ sin ((π / 2) * cos y) + y = cos ((π / 2) * sin y) + y) :=
sorry

end number_of_solutions_l297_297662


namespace third_trial_point_l297_297941

noncomputable def interval_a : ℝ := 1000
noncomputable def interval_b : ℝ := 2000

noncomputable def x1 : ℝ := interval_a + 0.618 * (interval_b - interval_a)
noncomputable def x2 : ℝ := interval_a + (interval_b - x1)

axiom result_better (x1_better: Prop) : x1_better → (interval_a ≤ x1 ∧ x2 ≤ interval_b) 

theorem third_trial_point (x1_better : x1 > x2) : 
  let x3 := x2 + (interval_b - x1) in
  x3 = 1764 :=
by
  sorry

end third_trial_point_l297_297941


namespace intersection_A_B_l297_297742

open Set

noncomputable def A : Set ℝ := { x | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l297_297742


namespace maxim_is_correct_l297_297152

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l297_297152


namespace smallest_positive_z_l297_297501

open Real

theorem smallest_positive_z (x y z : ℝ) (m k n : ℤ) 
  (h1 : cos x = 0) 
  (h2 : sin y = 1) 
  (h3 : cos (x + z) = -1 / 2) :
  z = 5 * π / 6 :=
by
  sorry

end smallest_positive_z_l297_297501


namespace discrete_subgroup_classification_finite_symmetry_group_classification_l297_297146

-- Part (a): Discrete subgroups of (ℝ², +)
theorem discrete_subgroup_classification (H : set (ℝ × ℝ)) :
  (H = {(0, 0)}) ∨
  (∃ v : ℝ × ℝ, H = {m • v | m : ℤ}) ∨
  (∃ v w : ℝ × ℝ, 
     linear_independent ℝ ![v, w] ∧
     H = {m • v + n • w | m n : ℤ}) := sorry

-- Part (b): Finite group of symmetries fixing the origin and a lattice L
theorem finite_symmetry_group_classification (L : set (ℝ × ℝ)) (G : finset (ℝ × ℝ →ₗ[ℝ] ℝ × ℝ)) :
  (∀ T ∈ G, ∀ x ∈ L, T x ∈ L) →
  (∃ i : ℕ, i ∈ {1, 2, 3, 4, 6} ∧
    (G = finset.range i ∨ ∃ j : ℕ, j = 2 * i ∧ G = finset.range j)) := sorry


end discrete_subgroup_classification_finite_symmetry_group_classification_l297_297146


namespace sum_of_first_10_good_numbers_eq_182_l297_297993

-- Definition of proper divisors of a natural number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range (n - 1)).filter (λ m => m > 1 ∧ n % m = 0)

-- Definition of a "good" natural number
def is_good (n : ℕ) : Prop :=
  n > 1 ∧ (proper_divisors n).prod = n

-- Collecting the first 10 "good" natural numbers
def good_numbers : List ℕ :=
  List.filter is_good [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

-- Taking the first 10 "good" numbers
def first_10_good_numbers : List ℕ :=
  good_numbers.take 10

-- Sum of the first 10 "good" natural numbers
def sum_first_10_good : ℕ :=
  (first_10_good_numbers).sum

-- The theorem statement
theorem sum_of_first_10_good_numbers_eq_182 :
  sum_first_10_good = 182 :=
  by
    sorry

end sum_of_first_10_good_numbers_eq_182_l297_297993


namespace cloth_sales_worth_l297_297958

theorem cloth_sales_worth 
  (commission : ℝ) 
  (commission_rate : ℝ) 
  (commission_received : ℝ) 
  (commission_rate_of_sales : commission_rate = 2.5)
  (commission_received_rs : commission_received = 21) 
  : (commission_received / (commission_rate / 100)) = 840 :=
by
  sorry

end cloth_sales_worth_l297_297958


namespace inequality_proof_l297_297485

theorem inequality_proof (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) : 
  2 * Real.sin α + Real.tan α > 3 * α := 
by
  sorry

end inequality_proof_l297_297485


namespace problem_statement_l297_297710

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def vector_subtract (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z  }

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2  }

theorem problem_statement :
  let A := Point3D.mk 3 (-6) 8 in
  let B := Point3D.mk 1 (-4) 2 in
  vector_subtract B A = Point3D.mk (-2) 2 (-6) ∧
  midpoint A B = Point3D.mk 2 (-5) 5 :=
by
  sorry

end problem_statement_l297_297710


namespace sequence_sum_ineq_l297_297912

noncomputable def sequence (n : ℕ) : ℕ → ℝ 
| 0 := 2
| (k + 1) := (sequence k)^2 - (sequence k) + 1

theorem sequence_sum_ineq :
  1 - 1 / (2003 ^ 2003) < (finset.range 2003).sum (λ k, 1 / sequence (k + 1)) ∧
  (finset.range 2003).sum (λ k, 1 / sequence (k + 1)) < 1 := sorry

end sequence_sum_ineq_l297_297912


namespace binary_to_decimal_1101_l297_297619

theorem binary_to_decimal_1101 :
  let binary_1101 := [1, 1, 0, 1].reverse
  let decimal_value := (binary_1101.enum.map (λ (p : ℕ × ℕ), let (i, d) := p in d * 2^i)).sum
  decimal_value = 13 :=
by
  let binary_1101 := [1, 1, 0, 1].reverse
  let decimal_value := (binary_1101.enum.map (λ (p : ℕ × ℕ), let (i, d) := p in d * 2^i)).sum
  show decimal_value = 13,
  sorry

end binary_to_decimal_1101_l297_297619


namespace slope_of_PQ_l297_297331

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

theorem slope_of_PQ :
  ∃ P Q : ℝ × ℝ,
    P = (0, 0) ∧ Q = (1, 8 / 3) ∧
    (∃ m : ℝ,
      m = 2 * Real.cos 0 ∧
      m = Real.sqrt 1 + 1 / Real.sqrt 1) ∧
    (Q.snd - P.snd) / (Q.fst - P.fst) = 8 / 3 :=
by
  sorry

end slope_of_PQ_l297_297331


namespace language_books_arrangement_l297_297483

theorem language_books_arrangement :
  let arabic_books := 2
  let german_books := 3
  let spanish_books := 4
  let total_units := arabic_books + 1 /* one unit of German books */ + 1 /* one unit of Spanish books */
  24 /* arrangements of 4 units */ * 6 /* arrangements within German unit */ * 24 /* arrangements within Spanish unit */ = 3456 :=
by
  sorry

end language_books_arrangement_l297_297483


namespace number_of_terms_in_seq_l297_297760

def sum_of_arithmetic_seq (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem number_of_terms_in_seq
  (a d n : ℕ)
  (h1 : a + (a + d) + (a + 2 * d) = 34)
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146)
  (h3 : sum_of_arithmetic_seq a d n = 390) :
  n = 13 :=
begin
  sorry
end

end number_of_terms_in_seq_l297_297760


namespace xy_product_l297_297863

theorem xy_product (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
sorry

end xy_product_l297_297863


namespace sunset_time_l297_297845

def length_of_daylight_in_minutes := 11 * 60 + 12
def sunrise_time_in_minutes := 6 * 60 + 45
def sunset_time_in_minutes := sunrise_time_in_minutes + length_of_daylight_in_minutes
def sunset_time_hour := sunset_time_in_minutes / 60
def sunset_time_minute := sunset_time_in_minutes % 60
def sunset_time_12hr_format := if sunset_time_hour >= 12 
    then (sunset_time_hour - 12, sunset_time_minute)
    else (sunset_time_hour, sunset_time_minute)

theorem sunset_time : sunset_time_12hr_format = (5, 57) :=
by
  sorry

end sunset_time_l297_297845


namespace sum_powers_of_i_l297_297640

-- Let i be a complex number such that i^2 = -1
def i : ℂ := Complex.I

theorem sum_powers_of_i : 
  i^(2023) + i^(2022) + i^(2021) + ... + i^1 + 1 = 0 :=
by 
  have h_cycle : ∀ k, i^(4*k+1) = i ∧ i^(4*k+2) = -1 ∧ i^(4*k+3) = -i ∧ i^(4*k+4) = 1,
  { intro k, sorry },
  have h2023_mod4 : 2023 % 4 = 3, sorry,
  have h_sum : i^(2023) + i^(2022) + i^(2021) + i^(2020) + ... + i^4 + i^3 + i^2 + i + 1 = -1 + 1,
  sorry,
  calc
    i^(2023) + i^(2022) + i^(2021) + ... + i + 1
      = -1 + 1 : sorry
  ... = 0 : by ring

end sum_powers_of_i_l297_297640


namespace john_total_cost_after_discount_l297_297605

/-- A store gives a 10% discount for the amount of the sell that was over $1000.
John buys 7 items that each cost $200. What does his order cost after the discount? -/
theorem john_total_cost_after_discount : 
  let discount_rate := 0.1
  let threshold := 1000
  let item_cost := 200
  let item_count := 7
  let total_cost := item_cost * item_count
  let discount := discount_rate * max 0 (total_cost - threshold)
  let final_cost := total_cost - discount
  in final_cost = 1360 :=
by 
  sorry

end john_total_cost_after_discount_l297_297605


namespace Nils_game_l297_297043

noncomputable def largest_possible_final_fortune (n : ℕ) : ℝ :=
  2^n / (n + 1)

theorem Nils_game (n : ℕ) (start : ℝ := 1) (x : ℝ) (y : ℝ) (0 ≤ x) (x ≤ y) :
  ∃ Y : ℝ, Y = largest_possible_final_fortune n := sorry

end Nils_game_l297_297043


namespace factor_expression_l297_297689

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l297_297689


namespace vertex_angle_90_or_30_l297_297118

-- Define an isosceles triangle and its properties
structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  isosceles : angle1 = angle2 ∨ angle2 = angle3 ∨ angle1 = angle3
  angle_sum : angle1 + angle2 + angle3 = 180

-- Define the interior angle positive value
def interiorAnglePositiveValue (T : IsoscelesTriangle) : ℝ :=
  (max (max T.angle1 T.angle2) T.angle3) - (min (min T.angle1 T.angle2) T.angle3)

-- The theorem to prove
theorem vertex_angle_90_or_30 (T : IsoscelesTriangle) (h : interiorAnglePositiveValue T = 45) :
  (T.angle1 = 90 ∨ T.angle2 = 90 ∨ T.angle3 = 90) ∨ (T.angle1 = 30 ∨ T.angle2 = 30 ∨ T.angle3 = 30) := 
by 
  sorry

end vertex_angle_90_or_30_l297_297118


namespace count_inequalities_l297_297410

theorem count_inequalities : 
  let expr1 := (λ x y : ℝ, x - y = 2)
  let expr2 := (λ x y : ℝ, x ≤ y)
  let expr3 := (λ x y : ℝ, x + y)
  let expr4 := (λ x y : ℝ, x^2 - 3y)
  let expr5 := (λ x : ℝ, x ≥ 0)
  let expr6 := (λ x : ℝ, (1 / 2) * x ≠ 3)
in
  (if expr2 0 0 then 1 else 0) + (if expr5 0 then 1 else 0) + (if expr6 0 then 1 else 0) = 3 :=
by
  sorry

end count_inequalities_l297_297410


namespace trailing_zeroes_factorial_1000_l297_297691

theorem trailing_zeroes_factorial_1000 :
  let v_5_factorial (n : ℕ) := ∑ i in (Finset.range (n.log 5).succ), n / 5^i
  v_5_factorial 1000 = 249 :=
by
  sorry

end trailing_zeroes_factorial_1000_l297_297691


namespace rahul_work_days_l297_297860

variable (R : ℕ)

theorem rahul_work_days
  (rajesh_days : ℕ := 2)
  (total_money : ℕ := 355)
  (rahul_share : ℕ := 142)
  (rajesh_share : ℕ := total_money - rahul_share)
  (payment_ratio : ℕ := rahul_share / rajesh_share)
  (work_rate_ratio : ℕ := rajesh_days / R) :
  payment_ratio = work_rate_ratio → R = 3 :=
by
  sorry

end rahul_work_days_l297_297860


namespace min_value_x_plus_y_l297_297822

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_plus_y_l297_297822


namespace find_positive_integer_l297_297257

theorem find_positive_integer (n : ℕ) :
  (arctan (1 / 2) + arctan (1 / 3) + arctan (1 / 7) + arctan (1 / n) = π / 4) → n = 7 :=
by
  sorry

end find_positive_integer_l297_297257


namespace hyperbola_eccentricity_sqrt3_l297_297700

theorem hyperbola_eccentricity_sqrt3
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b / a = Real.sqrt 2) :
  (let e := Real.sqrt (1 + (b^2 / a^2)) in e = Real.sqrt 3) :=
by
  sorry

end hyperbola_eccentricity_sqrt3_l297_297700


namespace parabola_intersection_probability_correct_l297_297936

noncomputable def parabola_intersection_probability : ℚ := sorry

theorem parabola_intersection_probability_correct :
  parabola_intersection_probability = 209 / 216 := sorry

end parabola_intersection_probability_correct_l297_297936


namespace second_grade_survey_count_l297_297976

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end second_grade_survey_count_l297_297976


namespace inequality_has_real_solution_l297_297158

variable {f : ℝ → ℝ}

theorem inequality_has_real_solution (h : ∃ x : ℝ, f x > 0) : 
    (∃ x : ℝ, f x > 0) :=
by
  sorry

end inequality_has_real_solution_l297_297158


namespace complex_div_eq_i_l297_297894

open Complex

theorem complex_div_eq_i : (1 + I) / (1 - I) = I := by
  sorry

end complex_div_eq_i_l297_297894


namespace num_unpainted_cubes_l297_297985

theorem num_unpainted_cubes (n : ℕ) (h1 : n ^ 3 = 125) : (n - 2) ^ 3 = 27 :=
by
  sorry

end num_unpainted_cubes_l297_297985


namespace sum_of_cubes_eq_10_l297_297098

noncomputable def floor_sum_of_solutions_cubes (x : ℝ) : ℝ :=
  if h : x^3 - 4 * ⌊x⌋ = 5 then x else 0

theorem sum_of_cubes_eq_10 :
  let solutions := [floor_sum_of_solutions_cubes (real.cbrt (-3)), floor_sum_of_solutions_cubes (real.cbrt 13)]
  ∑ x in solutions, x ^ 3 = 10 :=
by
  sorry

end sum_of_cubes_eq_10_l297_297098


namespace angle_CAD_in_convex_quadrilateral_l297_297785

theorem angle_CAD_in_convex_quadrilateral {A B C D : Type} [EuclideanGeometry A B C D]
  (AB_eq_BD : A = B, B = D)
  (angle_A : ∠ A = 65)
  (angle_B : ∠ B = 80)
  (angle_C : ∠ C = 75)
  : ∠ A = 15 :=
by
  sorry

end angle_CAD_in_convex_quadrilateral_l297_297785


namespace factor_of_99_l297_297174

theorem factor_of_99 (k : ℕ) 
  (h : ∀ n : ℕ, k ∣ n → k ∣ (nat.reverse_digits n)) : k ∣ 99 :=
sorry

end factor_of_99_l297_297174


namespace smallest_value_floor_sum_l297_297281

theorem smallest_value_floor_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = a * b * c) :
  (⟦(a^2 + b^2) / c⟧ + ⟦(b^2 + c^2) / a⟧ + ⟦(c^2 + a^2) / b⟧) ≥ 4 :=
by
  sorry

end smallest_value_floor_sum_l297_297281


namespace sum_squares_of_sines_eq_half89_l297_297666

theorem sum_squares_of_sines_eq_half89 :
  (Finset.range 89).sum (λ k, real.sin ((k + 1 : ℕ) * (real.pi / 180)) ^ 2) = 89 / 2 := 
sorry

end sum_squares_of_sines_eq_half89_l297_297666


namespace smallest_u_n_correct_l297_297820

noncomputable def smallest_u_n (n : ℕ) : ℕ :=
  Inf {u : ℕ | ∀ d : ℕ, (d > 0) → ∀ a : ℕ, ∃ b ∈ set.range (λ k, a + 2*k), b % d = 0}

theorem smallest_u_n_correct (n : ℕ) (hn : n > 0) : smallest_u_n n = 2*n - 1 :=
  sorry

end smallest_u_n_correct_l297_297820


namespace construct_triangle_l297_297228

theorem construct_triangle (a b l : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_l_pos : 0 < l) :
  ∃ (ABC : Type) [triangle ABC], 
    has_side ABC a ∧ 
    has_side ABC b ∧ 
    has_angle_bisector ABC l :=
sorry

end construct_triangle_l297_297228


namespace police_can_catch_thief_l297_297596

theorem police_can_catch_thief (v : ℝ) (hv : 0 < v) :
  let police_speed := v
  let thief_speed := 0.9 * v
  police_speed > thief_speed :=
begin
  let police_speed := v,
  let thief_speed := 0.9 * v,
  show police_speed > thief_speed, from sorry,
end

end police_can_catch_thief_l297_297596


namespace f_min_value_f_geq_nine_l297_297303

variable (a b c x : ℝ)

-- Conditions
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_one : Prop := a + b + c = 1

-- Function definition
def f (x : ℝ) : ℝ := abs (x - (1/a) - (1/b)) + abs (x + (1/c))

-- Goals
theorem f_min_value (h1: pos_numbers a b c) (h2 : sum_one a b c) : 
  ∃ x : ℝ, f a b c x = 9 := sorry

theorem f_geq_nine (h1: pos_numbers a b c) (h2 : sum_one a b c) (x : ℝ) :
  f a b c x ≥ 9 := 
sorry

end f_min_value_f_geq_nine_l297_297303


namespace problem1_problem2_l297_297215

theorem problem1 :
  ((-7 / 9 + 5 / 6 - 3 / 4) * (-36)) = 25 := sorry

theorem problem2 :
  (-1^4 - (1 - 0.5) * (1 / 2) * abs (1 - (-3)^2)) = -3 := sorry

end problem1_problem2_l297_297215


namespace ali_babas_cave_min_moves_l297_297923

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l297_297923


namespace difference_largest_smallest_l297_297113

noncomputable def ratio_2_3_5 := 2 / 3
noncomputable def ratio_3_5 := 3 / 5
noncomputable def int_sum := 90

theorem difference_largest_smallest :
  ∃ (a b c : ℝ), 
    a + b + c = int_sum ∧
    b / a = ratio_2_3_5 ∧
    c / a = 5 / 2 ∧
    b / a = 3 / 2 ∧
    c - a = 12.846 := 
by
  sorry

end difference_largest_smallest_l297_297113


namespace line_through_points_m_plus_b_l297_297518

theorem line_through_points_m_plus_b :
  ∃ m b, (m, b) = (-3, 9) ∧ (6 = m + b) := by
  let m := (-6 - 3) / (5 - 2)
  let b := 3 - (m * 2)
  use m, b
  calc
    (m, b) = (-3, 9) : by
      have h1 : m = -3 := by
        calc
          m = (-6 - 3) / (5 - 2) := by rfl
          ... = -9 / 3 := by rfl
          ... = -3 := by norm_num
      have h2 : b = 9 := by
        calc
          b = 3 - (m * 2) := by rfl
          ... = 3 - ((-3) * 2) := by rw [h1]
          ... = 3 - (-6) := by rfl
          ... = 9 := by norm_num
      exact ⟨h1, h2⟩
    6 = m + b := by norm_num
  sorry

end line_through_points_m_plus_b_l297_297518


namespace domain_of_inverse_l297_297729

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem domain_of_inverse (x : ℝ) : f x > 0 :=
by
  sorry

end domain_of_inverse_l297_297729


namespace score_entered_twice_l297_297474

theorem score_entered_twice (scores : List ℕ) (h : scores = [68, 74, 77, 82, 85, 90]) :
  ∃ (s : ℕ), s = 82 ∧ ∀ (entered : List ℕ), entered.length = 7 ∧ (∀ i, (List.take (i + 1) entered).sum % (i + 1) = 0) →
  (List.count (List.insertNth i 82 scores)) = 2 ∧ (∀ x, x ∈ scores.remove 82 → x ≠ s) :=
by
  sorry

end score_entered_twice_l297_297474


namespace find_standard_equation_of_ellipse_maximum_area_of_triangle_PMNP_l297_297295

-- Definitions for the given conditions
def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

def circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2 }

def foci_distance (c : ℝ) := 2 * c

def perimeter_triangle (A B F₂ : (ℝ × ℝ)) : ℝ :=
  dist A B + dist B F₂ + dist F₂ A

-- Proving the standard equation of the ellipse
theorem find_standard_equation_of_ellipse (a b : ℝ) (h1 : a > b > 0) (h2 : foci_distance √2 = 2 * √2) (h3 : perimeter_triangle (1,0) (0,1) (0,-1) = 4 * √3) :
  ellipse √3 1 = {p | (p.1^2 / 3) + (p.2^2) = 1 } :=
sorry

-- Proving the maximum area of △PMN
theorem maximum_area_of_triangle_PMNP (P : (ℝ × ℝ)) (hP : P ∈ circle 2) :
  ∃ M N : (ℝ × ℝ), (M ∈ circle 2) ∧ (N ∈ circle 2) ∧ (area_triangle P M N = 4) :=
sorry

end find_standard_equation_of_ellipse_maximum_area_of_triangle_PMNP_l297_297295


namespace sin_x_value_l297_297309

noncomputable def solve_sin_x_from_sec_x_plus_tan_x (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : Real :=
  if (Real.sin x = 8 / 17) then 8 / 17 else 0

theorem sin_x_value (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : 
  solve_sin_x_from_sec_x_plus_tan_x x h = 8 / 17 :=
sorry

end sin_x_value_l297_297309


namespace value_of_diamond_expression_l297_297274

noncomputable def diamond (x y : ℝ) : ℝ :=
  (x^2 + y^2) / (x^2 - y^2)

theorem value_of_diamond_expression :
  ((diamond 1 2) ≠ ((diamond (-(5 / 3)) 4)) = - (169 / 119)) :=
by
  sorry

end value_of_diamond_expression_l297_297274


namespace joan_gave_sam_seashells_l297_297425

-- Definitions of initial conditions
def initial_seashells : ℕ := 70
def remaining_seashells : ℕ := 27

-- Theorem statement
theorem joan_gave_sam_seashells : initial_seashells - remaining_seashells = 43 :=
by
  sorry

end joan_gave_sam_seashells_l297_297425


namespace sin_pi_over_4_minus_alpha_l297_297713

theorem sin_pi_over_4_minus_alpha (α : ℝ) (hα1 : sin (2 * α) = 4 / 5) (hα2 : 0 < α ∧ α < π / 4) :
  sin (π / 4 - α) = sqrt 10 / 10 :=
sorry

end sin_pi_over_4_minus_alpha_l297_297713


namespace original_rice_amount_l297_297795

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l297_297795


namespace problem_p_3_l297_297014

theorem problem_p_3 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) (hn : n = (2^(2*p) - 1) / 3) : n ∣ 2^n - 2 := by
  sorry

end problem_p_3_l297_297014


namespace finite_solutions_of_eq_l297_297054

theorem finite_solutions_of_eq (x y z : ℕ) :
  (1 / x + 1 / y + 1 / z = 1 / 1983) → finite { ⟨x, y, z⟩ : ℕ × ℕ × ℕ | 1 / x + 1 / y + 1 / z = 1 / 1983 } :=
by
  sorry

end finite_solutions_of_eq_l297_297054


namespace coefficient_of_x_squared_in_ffx_l297_297468

def f : ℝ → ℝ :=
λ x, if x ≥ 1 then x^6 else if x ≤ -1 then -2 * x - 1 else 0

theorem coefficient_of_x_squared_in_ffx {x : ℝ} (h : x ≤ -1) :
  let ffx := f (f x) in
  (∃ a b c, ffx = a * x^2 + b * x + c ∧ a = 60) :=
sorry

end coefficient_of_x_squared_in_ffx_l297_297468


namespace vector_relation_l297_297299

open_locale classical

variables {A B C : Type} [inner_product_space ℝ Type]
variables (A B C : Type) [add_comm_group Type] [module ℝ Type]

variables (A B C : Type) [affine_space Type]

theorem vector_relation (A B C : point ℝ) (pos : C ∈ line_segment ℝ A B) (h : dist A C = (2/7) * dist C B) :
  (vector A B) = - (9/7) * (vector B C) :=
sorry

end vector_relation_l297_297299


namespace median_and_mean_free_throws_l297_297581

def free_throws : List ℕ := [8, 20, 16, 14, 20, 13, 20, 17, 15, 18]

theorem median_and_mean_free_throws:
  let sorted_throws := List.sort free_throws
  let median := (sorted_throws.get! 4 + sorted_throws.get! 5) / 2
  let mean := (free_throws.sum / 10 : ℝ)
  median = 16.5 ∧ mean = 16.1 :=
by
  sorry

end median_and_mean_free_throws_l297_297581


namespace perimeter_of_grid_l297_297081

theorem perimeter_of_grid (total_area : ℝ) (n_squares : ℤ) (grid_dim : ℤ) (area_per_square : ℝ) (side_length : ℝ) (perimeter : ℝ) :
  total_area = 576 ∧
  n_squares = 9 ∧
  grid_dim = 3 ∧
  area_per_square = total_area / n_squares ∧  
  side_length = real.sqrt area_per_square ∧
  perimeter = 4 * grid_dim * side_length →
  perimeter = 192 :=
  by
    sorry

end perimeter_of_grid_l297_297081


namespace clerical_percentage_after_reduction_l297_297045

noncomputable def totalEmployees : ℕ := 5000
noncomputable def clericalProportion : ℚ := 1/5
noncomputable def technicalProportion : ℚ := 2/5
noncomputable def managerialProportion : ℚ := 2/5

noncomputable def clericalReduction : ℚ := 1/3
noncomputable def technicalReduction : ℚ := 1/4
noncomputable def managerialReduction : ℚ := 1/5

theorem clerical_percentage_after_reduction :
  let clericalBeforeReduction := clericalProportion * totalEmployees
  let technicalBeforeReduction := technicalProportion * totalEmployees
  let managerialBeforeReduction := managerialProportion * totalEmployees
  let clericalAfterReduction := clericalBeforeReduction * (1 - clericalReduction)
  let technicalAfterReduction := technicalBeforeReduction * (1 - technicalReduction)
  let managerialAfterReduction := managerialBeforeReduction * (1 - managerialReduction)
  let totalAfterReduction := clericalAfterReduction + technicalAfterReduction + managerialAfterReduction
  let clericalPercentage := (clericalAfterReduction / totalAfterReduction) * 100
  clericalPercentage ≈ 17.7 := sorry

end clerical_percentage_after_reduction_l297_297045


namespace proposition_3_proposition_4_l297_297819

variable {Line Plane : Type} -- Introduce the types for lines and planes
variable (m n : Line) (α β : Plane) -- Introduce specific lines and planes

-- Define parallel and perpendicular relations
variables {parallel : Line → Plane → Prop} {perpendicular : Line → Plane → Prop}
variables {parallel_line : Line → Line → Prop} {perpendicular_line : Line → Line → Prop}
variables {parallel_plane : Plane → Plane → Prop} {perpendicular_plane : Plane → Plane → Prop}

-- Define subset: a line n is in a plane α
variable {subset : Line → Plane → Prop}

-- Hypotheses for propositions 3 and 4
axiom prop3_hyp1 : perpendicular m α
axiom prop3_hyp2 : parallel_line m n
axiom prop3_hyp3 : parallel_plane α β

axiom prop4_hyp1 : perpendicular_line m n
axiom prop4_hyp2 : perpendicular m α
axiom prop4_hyp3 : perpendicular n β

theorem proposition_3 (h1 : perpendicular m α) (h2 : parallel_line m n) (h3 : parallel_plane α β) : perpendicular n β := sorry

theorem proposition_4 (h1 : perpendicular_line m n) (h2 : perpendicular m α) (h3 : perpendicular n β) : perpendicular_plane α β := sorry

end proposition_3_proposition_4_l297_297819


namespace Kyle_makes_99_dollars_l297_297435

-- Define the initial numbers of cookies and brownies
def initial_cookies := 60
def initial_brownies := 32

-- Define the numbers of cookies and brownies eaten by Kyle and his mom
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2

-- Define the prices for each cookie and brownie
def price_per_cookie := 1
def price_per_brownie := 1.50

-- Define the remaining cookies and brownies after consumption
def remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
def remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies

-- Define the total money Kyle will make
def money_from_cookies := remaining_cookies * price_per_cookie
def money_from_brownies := remaining_brownies * price_per_brownie

-- Define the total money Kyle will make from selling all baked goods
def total_money := money_from_cookies + money_from_brownies

-- Proof statement
theorem Kyle_makes_99_dollars :
  total_money = 99 :=
by
  sorry

end Kyle_makes_99_dollars_l297_297435


namespace xy_value_l297_297869

theorem xy_value (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
by
  sorry

end xy_value_l297_297869


namespace find_missing_number_l297_297090

theorem find_missing_number (mean : ℕ) (count : ℕ) (known_numbers_sum : ℕ) x :
  mean = 20 ∧ count = 8 ∧ known_numbers_sum = (1 + 22 + 23 + 25 + 26 + 27 + 2) →
  x = 34 := by
  intro h
  rcases h with ⟨h_mean, h_count, h_sum⟩
  have total_sum := h_mean * h_count
  have missing_number := total_sum - h_sum
  have missing_correct : missing_number = 34 := by
    calc
      missing_number = 160 - 126 : by rw [h_sum, h_mean, h_count]; simp
      ... = 34 : by norm_num
    done
  exact missing_correct
  sorry

end find_missing_number_l297_297090


namespace tan_sum_eq_tan_prod_l297_297764

noncomputable def tan (x : Real) : Real :=
  Real.sin x / Real.cos x

theorem tan_sum_eq_tan_prod (α β γ : Real) (h : tan α + tan β + tan γ = tan α * tan β * tan γ) :
  ∃ k : Int, α + β + γ = k * Real.pi :=
by
  sorry

end tan_sum_eq_tan_prod_l297_297764


namespace total_and_per_suitcase_profit_l297_297193

theorem total_and_per_suitcase_profit
  (num_suitcases : ℕ)
  (purchase_price_per_suitcase : ℕ)
  (total_sales_revenue : ℕ)
  (total_profit : ℕ)
  (profit_per_suitcase : ℕ)
  (h_num_suitcases : num_suitcases = 60)
  (h_purchase_price : purchase_price_per_suitcase = 100)
  (h_total_sales : total_sales_revenue = 8100)
  (h_total_profit : total_profit = total_sales_revenue - num_suitcases * purchase_price_per_suitcase)
  (h_profit_per_suitcase : profit_per_suitcase = total_profit / num_suitcases) :
  total_profit = 2100 ∧ profit_per_suitcase = 35 := by
  sorry

end total_and_per_suitcase_profit_l297_297193


namespace binary_mul_correct_l297_297265

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l297_297265


namespace find_p_and_t_l297_297644

-- Definitions of the variables and conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def focus := (2 : ℝ, 0 : ℝ)
def point_p (p : ℝ) : Prop := p > 0
def point_t (t : ℝ) : Prop := ellipse_eq 0 t ∧ (∃ p, angle_PTF_is_90 p t)

-- Helper definition for the angle condition
-- Assuming suitable definitions or constructs for angle measurement in the context
def angle_PTF_is_90 (p t : ℝ) : Prop := 
  ∀ (F P T : (ℝ × ℝ)), 
  F = (2, 0) ∧ P = (p, 0) ∧ T = (0, t) → angle P T F = 90

-- The statement to be proven
theorem find_p_and_t (p t : ℝ) : 
  (point_p p) ∧ (point_t t) → p = 2 * real.sqrt 2 ∧ (t = 1 ∨ t = -1) := 
sorry

end find_p_and_t_l297_297644


namespace probability_at_least_one_die_shows_three_l297_297549

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l297_297549


namespace number_of_odd_factors_of_252_l297_297362

def numOddFactors (n : ℕ) : ℕ :=
  if ∀ d : ℕ, n % d = 0 → ¬(d % 2 = 0) then d
  else 0

theorem number_of_odd_factors_of_252 : numOddFactors 252 = 6 := by
  -- Definition of n
  let n := 252
  -- Factor n into 2^2 * 63
  have h1 : n = 2^2 * 63 := rfl
  -- Find the number of odd factors of 63 since factors of 252 that are odd are the same as factors of 63
  have h2 : 63 = 3^2 * 7 := rfl
  -- Check the number of factors of 63
  sorry

end number_of_odd_factors_of_252_l297_297362


namespace parabolas_kite_area_l297_297523

theorem parabolas_kite_area (a b : ℝ) 
  (h1 : ∀ x, (a * x^2 - 3 = 0) → x = sqrt (3/a) ∨ x = -sqrt (3/a)) 
  (h2 : ∀ x, (5 - b * x^2 = 0) → x = sqrt (5/b) ∨ x = -sqrt (5/b))
  (h3 : (forall x, x = 0 → a*x^2 - 3 = -3))
  (h4 : (forall x, x = 0 → 5 - b*x^2 = 5))
  (h5 : 3 * b = 5 * a )
  (h6 : ∃ (area : ℝ), area = 18 ∧ (1/2)*(8)*(2*sqrt(3/a)) = area) :
  a + b = 128/81 := 
begin
  sorry
end

end parabolas_kite_area_l297_297523


namespace square_area_formula_l297_297980

-- Let s be the side length of the square, and r be the radius of the circle inscribed in the square
noncomputable def square_area (s r : ℝ) : ℝ :=
  s^2

-- circle passes through the midpoint of one of the diagonals of the square
theorem square_area_formula (r : ℝ) (h : ∀ s, (s * sqrt 2) / 2 = r): square_area (sqrt 2 * r) r = 2 * r^2 :=
by {
  sorry
}

end square_area_formula_l297_297980


namespace count_divisible_by_5_of_ababc_l297_297122

theorem count_divisible_by_5_of_ababc :
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  (a_vals.card * b_vals.card * c_vals.card) = 180 :=
by
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  have h_a_card : a_vals.card = 9 := by simp
  have h_b_card : b_vals.card = 10 := by simp
  have h_c_card : c_vals.card = 2 := by simp
  have total := h_a_card * h_b_card * h_c_card
  show total = 180 from sorry

end count_divisible_by_5_of_ababc_l297_297122


namespace asymptote_of_hyperbola_l297_297510

theorem asymptote_of_hyperbola (h : (∀ x y : ℝ, y^2 / 3 - x^2 / 2 = 1)) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt6 / 2) * x ∨ y = - (sqrt6 / 2) * x) :=
sorry

end asymptote_of_hyperbola_l297_297510


namespace volcanoes_exploded_percentage_l297_297992

theorem volcanoes_exploded_percentage (x : ℝ) :
  let total_volcanoes := 200 in
  let exploded_first := 2 * x in
  let remaining_after_first := total_volcanoes - exploded_first in
  let exploded_mid := 0.40 * remaining_after_first in
  let remaining_after_mid := remaining_after_first - exploded_mid in
  let exploded_end := 0.50 * remaining_after_mid in
  let remaining_after_end := remaining_after_mid - exploded_end in
  remaining_after_end = 48 →
  x = 20 :=
begin
  sorry
end

end volcanoes_exploded_percentage_l297_297992


namespace enrollment_difference_l297_297556

theorem enrollment_difference (a b c d : ℕ) (ha : a = 1650) (hb : b = 1130) (hc : c = 1400) (hd : d = 1320) :
  let numbers := [a, b, c, d].qsort (· ≤ ·),
  second_largest := numbers[-2],
  second_smallest := numbers[1],
  positive_difference := second_largest - second_smallest
  in
  positive_difference = 80 :=
by
  sorry

end enrollment_difference_l297_297556


namespace daily_lunch_break_is_1_hour_l297_297479

noncomputable def daily_lunch_break_duration (p a : ℝ) : ℝ :=
  let L := (8 - 1) * (1 / 30 + 0.06) in
  L

theorem daily_lunch_break_is_1_hour (p a : ℝ) :
  (8 - 1) * (1 / 30 + 0.06) = 0.6 :=
by
  sorry

end daily_lunch_break_is_1_hour_l297_297479


namespace average_sleep_time_l297_297211

def sleep_times : List ℕ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end average_sleep_time_l297_297211


namespace ali_babas_cave_min_moves_l297_297921

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l297_297921


namespace gold_silver_weight_problem_l297_297967

theorem gold_silver_weight_problem (x y : ℕ) (h1 : 9 * x = 11 * y) (h2 : (10 * y + x) - (8 * x + y) = 13) :
  9 * x = 11 * y ∧ (10 * y + x) - (8 * x + y) = 13 :=
by
  refine ⟨h1, h2⟩

end gold_silver_weight_problem_l297_297967


namespace probability_of_at_least_one_three_l297_297548

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l297_297548


namespace correct_propositions_count_l297_297658

def equal_square_difference_sequence (a : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = p

theorem correct_propositions_count :
  let prop1 : Prop := ¬ ∀ (a : ℕ → ℝ) (p : ℝ), equal_square_difference_sequence a p → ∀ n, (n ≥ 2) → (1 / a n) - (1 / a (n - 1)) = (1 / a 1) - (1 / a 0)
  let prop2 : Prop := ¬ equal_square_difference_sequence (λ n, (-2 : ℝ) ^ n) 4
  let prop3 : Prop := ∀ (a : ℕ → ℝ) (p : ℝ), equal_square_difference_sequence a p → ∀ k : ℕ, equal_square_difference_sequence (λ n, a (k * n)) (k * p)
  let prop4 : Prop := ∀ (a : ℕ → ℝ) (d : ℝ), (∀ n, a n = a 0 + d * (n - 1)) ∧ equal_square_difference_sequence a (d ^ 2) → d = 0
  prop1 ∧ prop2 ∧ prop3 ∧ prop4 → 2 := 
by
  sorry

end correct_propositions_count_l297_297658


namespace infinite_solutions_cos_sin_eq_l297_297369

open Real

theorem infinite_solutions_cos_sin_eq (h : ∀ x ∈ Icc (0 : ℝ) (2 * π), 
  cos (π / 2 * cos x + π / 2 * sin x) = sin (π / 2 * cos x - π / 2 * sin x)) : 
  infinite {x | x ∈ Icc (0 : ℝ) (2 * π) ∧ (cos (π / 2 * cos x + π / 2 * sin x) = sin (π / 2 * cos x - π / 2 * sin x))} :=
by
  sorry

end infinite_solutions_cos_sin_eq_l297_297369


namespace roses_distribution_l297_297491

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) : 
  initial_roses = 40 → 
  stolen_roses = 4 → 
  people = 9 → 
  (initial_roses - stolen_roses) / people = 4 :=
by
  intros h_initial_roses h_stolen_roses h_people
  rw [h_initial_roses, h_stolen_roses, h_people]
  norm_num
  sorry

end roses_distribution_l297_297491


namespace binary_101_eq_5_l297_297652

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l297_297652


namespace angle_relation_l297_297417

noncomputable def angle_sum_triangle (A B C : Point) : Prop :=
  let α := angle B A C
  let β := angle C B A
  let γ := angle A C B
  α + β + γ = 180 :=
sorry

noncomputable def cyclic_quadrilateral (A B C P : Point) : Prop :=
  let α := angle B P C
  let β := angle B E C
  α + β = 180 :=
sorry

theorem angle_relation (A B C D E P : Point) 
  (h1 : angle_sum_triangle A B C)
  (h2 : cyclic_quadrilateral B C E P)
  : angle A B C + angle B P C = 180 :=
sorry

end angle_relation_l297_297417


namespace tom_ratio_is_three_fourths_l297_297114

-- Define the years for the different programs
def bs_years : ℕ := 3
def phd_years : ℕ := 5
def tom_years : ℕ := 6
def normal_years : ℕ := bs_years + phd_years

-- Define the ratio of Tom's time to the normal time
def ratio : ℚ := tom_years / normal_years

theorem tom_ratio_is_three_fourths :
  ratio = 3 / 4 :=
by
  unfold ratio normal_years bs_years phd_years tom_years
  -- continued proof steps would go here
  sorry

end tom_ratio_is_three_fourths_l297_297114


namespace binary_101_to_decimal_l297_297647

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l297_297647


namespace fraction_girls_event_l297_297913

theorem fraction_girls_event:
  let students_maplewood := 300
  let boys_to_girls_maplewood := (3, 2)
  let students_brookside := 240
  let boys_to_girls_brookside := (2, 3)
  let girls_maplewood := students_maplewood * snd boys_to_girls_maplewood / (fst boys_to_girls_maplewood + snd boys_to_girls_maplewood)
  let girls_brookside := students_brookside * snd boys_to_girls_brookside / (fst boys_to_girls_brookside + snd boys_to_girls_brookside)
  let total_girls := girls_maplewood + girls_brookside
  let total_students := students_maplewood + students_brookside
  in (total_girls / total_students) = 22 / 45 :=
by sorry

end fraction_girls_event_l297_297913


namespace max_distinct_integer_solutions_le_2_l297_297286

def f (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem max_distinct_integer_solutions_le_2 
  (a b c : ℝ) (h₀ : a > 100) :
  ∀ (x : ℤ), |f a b c (x : ℝ)| ≤ 50 → 
  ∃ (x₁ x₂ : ℤ), x = x₁ ∨ x = x₂ :=
by
  sorry

end max_distinct_integer_solutions_le_2_l297_297286


namespace odd_factors_count_l297_297366

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l297_297366


namespace profit_percentage_40_l297_297190

theorem profit_percentage_40 (SP CP : ℝ) (hSP : SP = 100) (hCP : CP = 71.43) : (SP - CP) / CP * 100 = 40 := 
by {
  have hProfit : SP - CP = 28.57, from by {
    rw [hSP, hCP],
    norm_num
  },
  have hPercentage : (28.57 / 71.43) * 100 = 40, from by {
    norm_num
  },
  rw [←hProfit, hPercentage],
  sorry -- This placeholder means that we skip the final computation/verification proof
}

end profit_percentage_40_l297_297190


namespace find_x_for_g_eq_g_984_l297_297591

noncomputable def g (x : ℝ) : ℝ :=
if h : 1 ≤ x ∧ x ≤ 4 then x^2 - 4*x + 7 else g (x / 4) * 4

theorem find_x_for_g_eq_g_984 :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ g(x) = g(984) :=
sorry

end find_x_for_g_eq_g_984_l297_297591


namespace sin_double_angle_l297_297754

theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : (π / 2) < α ∧ α < π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := sorry

end sin_double_angle_l297_297754


namespace marble_draw_probability_l297_297971

theorem marble_draw_probability :
  let total_marbles := 12
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 3

  let p_red_first := (red_marbles / total_marbles : ℚ)
  let p_white_second := (white_marbles / (total_marbles - 1) : ℚ)
  let p_blue_third := (blue_marbles / (total_marbles - 2) : ℚ)
  
  p_red_first * p_white_second * p_blue_third = (1/22 : ℚ) :=
by
  sorry

end marble_draw_probability_l297_297971


namespace expected_value_is_minus_0_point_38_l297_297473

-- Define the values for each roll
def roll_values (n : ℕ) : ℝ :=
  if n = 2 then 2 else
  if n = 3 then 3 else
  if n = 4 then -4 else
  if n = 8 then -8 else
  if n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 then 0 else
  0

-- Define the probability mass function for a fair 8-sided die
noncomputable def die_pmf : PMF (Fin 8) :=
  PMF.uniform_of_fin (Fin 8)

-- Define the expected value in terms of die rolls
noncomputable def expected_value : ℝ :=
  ∑ n in Finset.univ, (die_pmf n) * roll_values n.val

theorem expected_value_is_minus_0_point_38 :
  expected_value = -0.38 :=
by
  -- Begin your proof here (skipping for the purpose)
  sorry

end expected_value_is_minus_0_point_38_l297_297473


namespace prove_perimeter_form_sum_abc_eq_5_l297_297230

/-
  Define the points as per the given condition and compute the distance between each consecutive pair.
-/
structure Point := (x : ℝ) (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def P1 := Point.mk 0 2
def P2 := Point.mk 1 3
def P3 := Point.mk 2 3
def P4 := Point.mk 2 2
def P5 := Point.mk 3 0
def P6 := Point.mk 2 (-1)
def P7 := P1

/-
  Summarize distances to form the perimeter and ensure it fits the form a + b√2 + c√5.
-/
def perimeter : ℝ :=
  distance P1 P2 + distance P2 P3 + distance P3 P4 + distance P4 P5 + distance P5 P6 + distance P6 P1

theorem prove_perimeter_form :
  ∃ a b c : ℤ, perimeter = a + b * real.sqrt 2 + c * real.sqrt 5 := by
  sorry

theorem sum_abc_eq_5 (a b c : ℤ) (h : perimeter = a + b * real.sqrt 2 + c * real.sqrt 5) : a + b + c = 5 := by
  sorry

end prove_perimeter_form_sum_abc_eq_5_l297_297230


namespace smallest_whole_number_l297_297133

theorem smallest_whole_number :
  ∃ a : ℕ, a % 3 = 2 ∧ a % 5 = 3 ∧ a % 7 = 3 ∧ ∀ b : ℕ, (b % 3 = 2 ∧ b % 5 = 3 ∧ b % 7 = 3 → a ≤ b) :=
sorry

end smallest_whole_number_l297_297133


namespace find_ratio_l297_297379

def given_conditions (a b c x y z : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 25 ∧ x^2 + y^2 + z^2 = 36 ∧ a * x + b * y + c * z = 30

theorem find_ratio (a b c x y z : ℝ)
  (h : given_conditions a b c x y z) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
sorry

end find_ratio_l297_297379


namespace tangent_line_eq_l297_297715

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x

theorem tangent_line_eq
  (A : ℝ × ℝ) 
  (hA : A = (0, 16)) 
  : (∃ (m : ℝ), m = 9 ∧ ∃ b : ℝ, b = 22 ∧ ∀ x y : ℝ, y = f x → y = m * x + b) := 
  sorry

end tangent_line_eq_l297_297715


namespace max_variance_probability_score_4_in_5_l297_297582

variables (P : ℝ) (xi : ℕ → ℝ)

-- Condition: Player's shooting percentage P
def shooting_percentage := P

-- Condition: Score xi for a shot
def score_shot (i : ℕ) : ℝ := xi i

-- Question 1: Maximum value of the variance Dx
theorem max_variance (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) :
  ∃ p : ℝ, p = 1/2 ∧ (P * (1 - P) ≤ 1/4) := 
sorry

-- Question 2: Probability of scoring 4 points out of 5 shots
theorem probability_score_4_in_5
  (hP : P = 1/2) :
  ∃ (n k : ℕ), (n = 5) ∧ (k = 4) ∧ 
  (nat.choose n k * (P^k) * ((1 - P)^(n - k)) = 5/32) :=
sorry

end max_variance_probability_score_4_in_5_l297_297582


namespace xy_product_l297_297864

theorem xy_product (x y : ℝ) (h1 : 2 ^ x = 16 ^ (y + 1)) (h2 : 27 ^ y = 3 ^ (x - 2)) : x * y = 8 :=
by
  sorry

end xy_product_l297_297864


namespace is_not_power_function_f4_l297_297137

-- Definitions for the conditions
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), ∀ b: ℝ, f b = b^a

def f1 (x : ℝ) : ℝ := x^0
def f2 (x : ℝ) : ℝ := Real.sqrt x
def f3 (x : ℝ) : ℝ := x^2
def f4 (x : ℝ) : ℝ := 2^x

-- Proof that f4 is not a power function
theorem is_not_power_function_f4 : ¬is_power_function f4 := by
  sorry

end is_not_power_function_f4_l297_297137


namespace second_grade_survey_count_l297_297975

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end second_grade_survey_count_l297_297975


namespace second_man_start_time_l297_297989

theorem second_man_start_time (P Q : Type) (departure_time_P departure_time_Q meeting_time arrival_time_P arrival_time_Q : ℕ) 
(distance speed : ℝ) (first_man_speed second_man_speed : ℕ → ℝ)
(h1 : departure_time_P = 6) 
(h2 : arrival_time_Q = 10) 
(h3 : arrival_time_P = 12) 
(h4 : meeting_time = 9) 
(h5 : ∀ t, 0 ≤ t ∧ t ≤ 4 → first_man_speed t = distance / 4)
(h6 : ∀ t, second_man_speed t = distance / 4)
(h7 : ∀ t, second_man_speed t * (meeting_time - t) = (3 * distance / 4))
: departure_time_Q = departure_time_P :=
by 
  sorry

end second_man_start_time_l297_297989


namespace angle_CAD_in_convex_quadrilateral_l297_297776

theorem angle_CAD_in_convex_quadrilateral
  (ABCD : Type)
  [convex_quadrilateral ABCD]
  (A B C D : ABCD)
  (h1 : AB = BD)
  (h2 : ∠A = 65)
  (h3 : ∠B = 80)
  (h4 : ∠C = 75)
  : ∠CAD = 15 := sorry

end angle_CAD_in_convex_quadrilateral_l297_297776


namespace find_t_l297_297842

-- Define the utility function
def utility (r j : ℕ) : ℕ := r * j

-- Define the Wednesday and Thursday utilities
def utility_wednesday (t : ℕ) : ℕ := utility (t + 1) (7 - t)
def utility_thursday (t : ℕ) : ℕ := utility (3 - t) (t + 4)

theorem find_t : (utility_wednesday t = utility_thursday t) → t = 5 / 8 :=
by
  sorry

end find_t_l297_297842


namespace percentage_men_attended_picnic_l297_297768

variable (E : ℝ)  -- The total number of employees
variable (M : ℝ)  -- The number of men
variable (W : ℝ)  -- The number of women
variable (P : ℝ)  -- The total percentage of employees who attended the picnic
variable (x : ℝ)  -- The percentage of men who attended the picnic
variable (y : ℝ)  -- The percentage of women who attended the picnic

-- Condition: 45% of all employees are men
axiom h1 : M = 0.45 * E

-- Condition: 40% of the women attended the picnic
axiom h2 : y = 0.40

-- Condition: 31.000000000000007% of all employees attended the picnic
axiom h3 : P = 0.31000000000000007

-- Number of women in terms of E
def W_def := E - M

-- Number of women who attended the picnic
def women_picnic := 0.40 * W_def

-- Total number of picnickers is the sum of men and women who attended
def total_picnickers := x * M + women_picnic

-- Proving the percentage of men who attended the picnic
theorem percentage_men_attended_picnic : 
  E > 0 → 
  M = 0.45 * E → 
  P = 0.31000000000000007 → 
  total_picnickers = P * E → 
  x = 0.2 :=
begin 
  sorry
end

end percentage_men_attended_picnic_l297_297768


namespace product_of_largest_and_second_largest_prime_factors_of_1170_l297_297945

theorem product_of_largest_and_second_largest_prime_factors_of_1170 : 
  ∃ (p1 p2 : ℕ), p1 < p2 ∧ prime p1 ∧ prime p2 ∧ (∃ k l m n : ℕ, 1170 = 2^k * 3^l * p1 * p2^m * 13^n) ∧ (p1 * p2 = 65) := by
sorry

end product_of_largest_and_second_largest_prime_factors_of_1170_l297_297945


namespace correct_proposition_l297_297298

variables {Line Plane : Type*}
variables (m n : Line) (α β : Plane)

-- Define perpendicular and parallel relations
def parallel (l1 l2 : Line) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular (p1 p2 : Plane) : Prop := sorry
def lying_in (l : Line) (p : Plane) : Prop := sorry

theorem correct_proposition 
  (h1 : perpendicular α β)
  (h2 : perpendicular m β)
  (h3 : ¬ lying_in m α) :
  parallel m α := 
sorry

end correct_proposition_l297_297298


namespace compare_fractions_l297_297222

theorem compare_fractions (a b c d : ℤ) (h1 : a = -(2)) (h2 : b = 3) (h3 : c = -(3)) (h4 : d = 5) :
  (a : ℚ) / b < c / d := 
by {
  simp [h1, h2, h3, h4],
  norm_num,
  simp [rat.lt_iff],
  exact lt_trans (neg_lt_zero.mpr (nat.cast_pos.mpr (show 3 * 5 > 0, by norm_num))) (show -(2 * 5 : ℤ) < -(3 * 3), by norm_num [lt_one_mul]),
}

end compare_fractions_l297_297222


namespace camel_cannot_traverse_all_fields_exactly_once_l297_297486

-- Define the hexagonal board with side length 3
def hexagonal_board_side_length := 3
def total_fields : Nat := 1 + 6 * hexagonal_board_side_length + 6 * (hexagonal_board_side_length - 1)

-- Each move of the camel changes the field color
def camel_move_changes_color : Prop := sorry

-- Prove that it is impossible for the camel to traverse all fields exactly once, starting from a corner cell.
theorem camel_cannot_traverse_all_fields_exactly_once :
  total_fields = 19 ∧ camel_move_changes_color → ∀ start_corner : Nat, ¬ ∃ path : List Nat, path.length = 18 ∧ valid_path path :=
sorry

end camel_cannot_traverse_all_fields_exactly_once_l297_297486


namespace line_intersects_axes_l297_297629

theorem line_intersects_axes :
  let line (x y : ℝ) := 2 * y + 5 * x = 10 in
  (∃ x' : ℝ, ∃ y' : ℝ, (line x' 0) ∧ x' = 2 ∧ y' = 0) ∧
  (∃ y' : ℝ, ∃ x' : ℝ, (line 0 y') ∧ x' = 0 ∧ y' = 5) :=
by
  sorry

end line_intersects_axes_l297_297629


namespace find_angle_CAD_l297_297772

-- Definitions representing the problem conditions
variables {A B C D : Type} [has_angle A] [has_angle B] [has_angle C] [has_angle D]

-- Given conditions
def is_convex (A B C D : Type) : Prop := sorry  -- convex quadrilateral ABCD
def angle_A (A : Type) : ℝ := 65
def angle_B (B : Type) : ℝ := 80
def angle_C (C : Type) : ℝ := 75
def equal_sides (A B D : Type) : Prop := sorry  -- AB = BD

-- Theorem statement: Given the conditions, prove the desired angle
theorem find_angle_CAD {A B C D : Type}
  [is_convex A B C D] 
  [equal_sides A B D]
  (h1 : angle_A A = 65)
  (h2 : angle_B B = 80)
  (h3 : angle_C C = 75)
  : ∃ (CAD : ℝ), CAD = 15 := 
sorry -- proof omitted

end find_angle_CAD_l297_297772


namespace P_P_eq_Q_Q_no_real_solution_l297_297156

variables {R : Type*} [CommRing R] [IsDomain R]

noncomputable def P : R[X] := sorry
noncomputable def Q : R[X] := sorry

-- P(Q(x)) = Q(P(x)) for all x ∈ R
axiom P_Q_comm : ∀ x : R, P.eval (Q.eval x) = Q.eval (P.eval x)

-- P(x) = Q(x) has no real solutions
axiom no_real_solutions : ∀ x : R, P.eval x ≠ Q.eval x

theorem P_P_eq_Q_Q_no_real_solution : ∀ x : R, P.eval (P.eval x) ≠ Q.eval (Q.eval x) :=
sorry

end P_P_eq_Q_Q_no_real_solution_l297_297156


namespace probability_of_at_least_one_three_l297_297546

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l297_297546


namespace Polly_tweeting_time_l297_297482

theorem Polly_tweeting_time (h : Polly.happy_frequency = 18) 
                             (hun : Polly.hungry_frequency = 4) 
                             (m : Polly.mirror_frequency = 45) 
                             (total : 3 * t = 1340) :
  t = 20 := by
  sorry

end Polly_tweeting_time_l297_297482


namespace max_y_for_x_less_than_minus_one_l297_297682

theorem max_y_for_x_less_than_minus_one : ∀ x : ℝ, x < -1 → 
  (let y := (x^2 + 7*x + 10) / (x + 1) in y ≤ 1) :=
sorry

end max_y_for_x_less_than_minus_one_l297_297682


namespace correct_props_l297_297450

-- Definitions
variables {α : Type*} [affine_space α ℝ] (a b : ℝ) (plane : set α)

-- Propose the conditions using Lean types and variables
def parallel (a b : ℝ) : Prop := ∀ (x y : α), x ∈ line a → y ∈ line b → x -ᵥ y ∈ plane
def perpendicular (a : ℝ) (plane : set α) : Prop := ∀ (x : α), x ∈ line a → x ∈ plane
def in_plane (a : α) (plane : set α) : Prop := a ∈ plane

-- Proof problem
theorem correct_props (a b : ℝ) (plane : set α) :
  (parallel a b → perpendicular a plane → perpendicular b plane) ∧
  (perpendicular a plane → perpendicular b plane → parallel a b) :=
by sorry

end correct_props_l297_297450


namespace tip_calculation_l297_297073

def women's_haircut_cost : ℝ := 48
def children's_haircut_cost : ℝ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℝ := 0.20

theorem tip_calculation : 
  let total_cost := (children's_haircut_cost * number_of_children) + women's_haircut_cost in
  let tip := total_cost * tip_percentage in
  tip = 24 :=
by 
  sorry

end tip_calculation_l297_297073


namespace problem_statement_l297_297375

variables {Line Plane : Type}
variables {m n : Line} {alpha beta : Plane}

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry
def perp (l : Line) (p : Plane) : Prop := sorry

-- Define that m and n are different lines
axiom diff_lines (m n : Line) : m ≠ n 

-- Define that alpha and beta are different planes
axiom diff_planes (alpha beta : Plane) : alpha ≠ beta

-- Statement to prove: If m ∥ n and m ⟂ α, then n ⟂ α
theorem problem_statement (h1 : parallel m n) (h2 : perp m alpha) : perp n alpha := 
sorry

end problem_statement_l297_297375


namespace sum_exterior_angles_regular_decagon_l297_297102

theorem sum_exterior_angles_regular_decagon : 
  let n := 10 
  ∑ i in (Finset.range n), exterior_angle i = 360 :=
by
  sorry

end sum_exterior_angles_regular_decagon_l297_297102


namespace age_ratio_l297_297888

noncomputable def rahul_present_age (future_age : ℕ) (years_passed : ℕ) : ℕ := future_age - years_passed

theorem age_ratio (future_rahul_age : ℕ) (years_passed : ℕ) (deepak_age : ℕ) :
  future_rahul_age = 26 →
  years_passed = 6 →
  deepak_age = 15 →
  rahul_present_age future_rahul_age years_passed / deepak_age = 4 / 3 :=
by
  intros
  have h1 : rahul_present_age 26 6 = 20 := rfl
  sorry

end age_ratio_l297_297888


namespace original_selling_price_l297_297963

theorem original_selling_price (discount : ℝ) (discounted_price : ℝ) (original_price : ℝ) 
    (discount_eq : discount = 0.25) (discounted_price_eq : discounted_price = 560)
    (price_eq : original_price = discounted_price / (1 - discount)):
  original_price = 746.67 :=
by
  rw [discount_eq, discounted_price_eq, price_eq]
  norm_num
  done

end original_selling_price_l297_297963


namespace no_solution_exists_l297_297374

theorem no_solution_exists : ¬ ∃ n : ℕ, 0 < n ∧ (2^n % 60 = 29 ∨ 2^n % 60 = 31) := 
by
  sorry

end no_solution_exists_l297_297374


namespace equilateral_right_triangle_nonexistent_l297_297142

theorem equilateral_right_triangle_nonexistent :
  ∀ (A B C : ℝ), 
    (A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∨ 
    (A ≠ 60 ∧ B ≠ 60 ∧ C ≠ 60 ∨ A = 90 ∨ B = 90 ∨ C = 90) → 
    false := 
sorry

end equilateral_right_triangle_nonexistent_l297_297142


namespace sum_of_g_49_l297_297028

theorem sum_of_g_49 :
  let f := λ x : ℝ, 5 * x^2 - 4 in
  let g := λ y : ℝ, (classical.some (λ x : ℝ, f x = y))^2 - (classical.some (λ x : ℝ, f x = y)) + 2 in
  g 49 + g 49 = 2 * (g 49)
  :=
by
  let f := λ x : ℝ, 5 * x^2 - 4
  let g := λ y : ℝ, (classical.some (λ x : ℝ, f x = y))^2 - (classical.some (λ x : ℝ, f x = y)) + 2
  show (g 49 + g 49 = 2 * (g 49))
  sorry

end sum_of_g_49_l297_297028


namespace Cd_sum_l297_297018

theorem Cd_sum : ∀ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 3 → (C / (x-3) + D * (x+2) = (-2 * x^2 + 8 * x + 28) / (x-3))) → 
  (C + D = 20) :=
by
  intros C D h
  sorry

end Cd_sum_l297_297018


namespace duplicate_arithmetic_means_l297_297914

theorem duplicate_arithmetic_means :
  ∀ (students: Fin 100) (cards: Fin 101),
  ∃ (a b : Fin 10000), a ≠ b ∧
  (calculate_mean students a = calculate_mean students b) :=
by
  -- define conditions, sets, etc.
  sorry

end duplicate_arithmetic_means_l297_297914


namespace ratio_of_a_to_b_l297_297759

theorem ratio_of_a_to_b 
  (b c d a : ℚ)
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := 
by sorry

end ratio_of_a_to_b_l297_297759


namespace fraction_equality_l297_297223

theorem fraction_equality : (18 / (5 * 107 + 3) = 18 / 538) := 
by
  -- Proof skipped
  sorry

end fraction_equality_l297_297223


namespace second_pipe_filling_time_l297_297477

theorem second_pipe_filling_time :
  ∃ T : ℝ, (1/20 + 1/T) * 2/3 * 16 = 1 ∧ T = 160/7 :=
by
  use 160 / 7
  sorry

end second_pipe_filling_time_l297_297477


namespace fill_pool_time_l297_297476

theorem fill_pool_time (R : ℝ) (T : ℝ) (hSlowerPipe : R = 1 / 9) (hFasterPipe : 1.25 * R = 1.25 / 9)
                     (hCombinedRate : 2.25 * R = 2.25 / 9) : T = 4 := by
  sorry

end fill_pool_time_l297_297476


namespace color_dag_edges_l297_297986

universe u

-- Define the type for vertices
variables (V : Type u) [fintype V]

-- Define the type for directed edges as a pair of vertices
def edge (V : Type u) := V × V

structure directed_graph (V : Type u) :=
(edges : set (edge V))
(acyclic : ∀ (u v w : V), ((u, v) ∈ edges → (v, w) ∈ edges → (w, u) ∉ edges))
(max_path_length : ∀ (u v : V), exists (p : list (edge V)), (u, v) ∈ edges → p.length ≤ 99)

noncomputable def color_edges (G : directed_graph V) : Prop :=
∃ (color : edge V → bool), 
  (∀ (v : V), ∀ (p : list (edge V)), 
    (∀ (e : edge V), e ∈ p → (color e = tt)) → p.length ≤ 9) ∧
  (∀ (v : V), ∀ (p : list (edge V)), 
    (∀ (e : edge V), e ∈ p → (color e = ff)) → p.length ≤ 9)

theorem color_dag_edges : 
  ∀ (V : Type u) [fintype V] (G : directed_graph V), 
    color_edges G :=
sorry

end color_dag_edges_l297_297986


namespace prob_even_product_eq_19_over_20_l297_297539

-- Define the set of integers from 1 to 6
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Lean function to calculate the probability
noncomputable def even_product_probability : ℕ :=
  let total_ways := (S.card.choose 3) in
  let odd_elements := {1, 3, 5} : Finset ℕ in
  let odd_ways := (odd_elements.card.choose 3) in
  let odd_probability := odd_ways / total_ways in
  1 - odd_probability

-- The theorem to prove
theorem prob_even_product_eq_19_over_20 : even_product_probability = 19/20 :=
  sorry

end prob_even_product_eq_19_over_20_l297_297539


namespace quadratic_equation_coefficients_l297_297024
open Complex

-- Definitions based on the conditions
noncomputable def omega : ℂ := sorry -- assume some specific complex root of unity here
axiom omega_nine_one : omega^9 = 1
axiom omega_ne_one : omega ≠ 1

def alpha : ℂ := omega + omega^3 + omega^5
def beta : ℂ := omega^2 + omega^4 + omega^7

-- The proof problem statement
theorem quadratic_equation_coefficients : ∃ (a b : ℝ), a = 0 ∧ b = 3 ∧ (∀ x : ℂ, x^2 + a * x + b = 0 ↔ x = alpha ∨ x = beta) :=
by
  use 0, 3
  split
  . exact rfl
  split
  . exact rfl
  sorry

end quadratic_equation_coefficients_l297_297024


namespace minimum_value_of_exponential_expression_l297_297323

theorem minimum_value_of_exponential_expression :
  ∀ (x y z : ℝ), x + 2 * y + 3 * z = 6 → 2^x + 4^y + 8^z ≥ 12 :=
by {
  intros,
  sorry
}

end minimum_value_of_exponential_expression_l297_297323


namespace rearrange_to_rectangle_l297_297231

-- Definition of a geometric figure and operations
structure Figure where
  parts : List (List (ℤ × ℤ)) -- List of parts represented by lists of coordinates

def is_cut_into_three_parts (fig : Figure) : Prop :=
  fig.parts.length = 3

def can_be_rearranged_to_form_rectangle (fig : Figure) : Prop := sorry

-- Initial given figure
variable (initial_figure : Figure)

-- Conditions
axiom figure_can_be_cut : is_cut_into_three_parts initial_figure
axiom cuts_not_along_grid_lines : True -- Replace with appropriate geometric operation when image is known
axiom parts_can_be_flipped : True -- Replace with operation allowing part flipping

-- Theorem to prove
theorem rearrange_to_rectangle : 
  is_cut_into_three_parts initial_figure →
  can_be_rearranged_to_form_rectangle initial_figure := 
sorry

end rearrange_to_rectangle_l297_297231


namespace petya_wins_when_n_is_odd_l297_297628

theorem petya_wins_when_n_is_odd (n : ℕ) :
  (∃ (is_regular_2n_gon : ∀ (k : ℕ), k < 2*n → Prop) (is_empty_cup : ∀ (k : ℕ), k < 2*n → Prop)
    (can_fill_two_adj_or_symmetric : ∀ (k m : ℕ), k < 2*n → m < 2*n → Prop)
    (peter_starts_first : ∀ (k : ℕ), k = 0 → Prop)
    (player_loses_if_no_move : ∀ (player : ℕ), (player = 0 ∨ player = 1) → Prop),
  odd n → Petya_wins) :=
by
  sorry

end petya_wins_when_n_is_odd_l297_297628


namespace compare_fractions_l297_297219

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end compare_fractions_l297_297219


namespace binomial_coefficients_equal_constant_term_expansion_l297_297721

variable (x : ℝ) (n : ℕ)

theorem binomial_coefficients_equal (h : ∀ (n : ℕ), binomial_coeff n 2 = binomial_coeff n 4) : n = 6 :=
by
  sorry

theorem constant_term_expansion (h : n = 6) : 
  constant_term ((2 * real.sqrt x - 1 / real.sqrt x)^n) = -160 :=
by
  sorry

end binomial_coefficients_equal_constant_term_expansion_l297_297721


namespace calc_expression_is_24_l297_297632

def calc_expression : ℕ := (30 / (8 + 2 - 5)) * 4

theorem calc_expression_is_24 : calc_expression = 24 :=
by
  sorry

end calc_expression_is_24_l297_297632


namespace kyle_total_revenue_l297_297437

-- Define the conditions
def initial_cookies := 60
def initial_brownies := 32
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2
def price_per_cookie := 1
def price_per_brownie := 1.5

-- Statement of the proof
theorem kyle_total_revenue :
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies),
      remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies),
      revenue_from_cookies := remaining_cookies * price_per_cookie,
      revenue_from_brownies := remaining_brownies * price_per_brownie,
      total_revenue := revenue_from_cookies + revenue_from_brownies
  in total_revenue = 99 :=
by
  sorry

end kyle_total_revenue_l297_297437


namespace ricky_roses_l297_297493

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end ricky_roses_l297_297493


namespace friends_riding_area_correct_l297_297540

-- Defining the given conditions
def tommy_north_south : ℕ := 2
def tommy_east_west : ℕ := 3
def tommy_area : ℕ := tommy_north_south * tommy_east_west

-- Friend's area is 4 times Tommy's area
def friends_riding_area (tommy_area : ℕ) : ℕ := tommy_area * 4

-- Theorem to prove the friend's riding area
theorem friends_riding_area_correct : friends_riding_area tommy_area = 24 := by
  let tommy_area := tommy_north_south * tommy_east_west
  have h1 : tommy_area = 6 := by rfl
  show tommy_area * 4 = 24 from
    calc
      tommy_area * 4 = 6 * 4 := by rw [h1]
      ... = 24 := by norm_num

end friends_riding_area_correct_l297_297540


namespace initial_money_jennifer_l297_297009

theorem initial_money_jennifer (M : ℝ) (h1 : (1/5) * M + (1/6) * M + (1/2) * M + 12 = M) : M = 90 :=
sorry

end initial_money_jennifer_l297_297009


namespace number_of_unique_four_digit_numbers_l297_297745

theorem number_of_unique_four_digit_numbers : 
  let digits := [3, 3, 0, 9] in
  (∃ a b c d : ℕ, a ≠ 0 ∧ List.perm digits [a, b, c, d]) →
  6 := 
sorry

end number_of_unique_four_digit_numbers_l297_297745


namespace razorback_shop_total_sales_l297_297507

variable price_per_tshirt : ℝ := 16
variable num_tshirts_sold : ℝ := 45
variable discount_rate : ℝ := 0.10
variable sales_tax_rate : ℝ := 0.06

theorem razorback_shop_total_sales :
  let total_sales := num_tshirts_sold * price_per_tshirt
  let discounted_price := total_sales * (1 - discount_rate)
  let tax := discounted_price * sales_tax_rate
  discounted_price + tax = 686.88 := by
  sorry

end razorback_shop_total_sales_l297_297507


namespace solve_system1_solve_system2_l297_297067

section System1

variables (x y : ℤ)

def system1_sol := x = 4 ∧ y = 8

theorem solve_system1 (h1 : y = 2 * x) (h2 : x + y = 12) : system1_sol x y :=
by 
  sorry

end System1

section System2

variables (x y : ℤ)

def system2_sol := x = 2 ∧ y = 3

theorem solve_system2 (h1 : 3 * x + 5 * y = 21) (h2 : 2 * x - 5 * y = -11) : system2_sol x y :=
by 
  sorry

end System2

end solve_system1_solve_system2_l297_297067


namespace simplest_radical_form_of_given_l297_297205

/-- Define the condition for a radicand to not contain a denominator --/
def no_denominator (r : ℚ) : Prop := (r.denom = 1)

/-- Define the condition for a radicand to not contain a factor that can be completely squared --/
def no_factor_completely_squared (r : ℚ) : Prop := ∀ (n : ℕ), n^2 ∣ r → n = 1

/-- Define the simplest radical form property --/
def simplest_radical_form (r : ℚ) : Prop := 
  no_denominator r ∧ no_factor_completely_squared r

/-- Problem statement with the given conditions and correct answer --/
theorem simplest_radical_form_of_given (
  h1 : ¬simplest_radical_form (√(9:ℚ)),
  h2 : simplest_radical_form (√(10:ℚ)),
  h3 : ¬simplest_radical_form (√(20:ℚ)),
  h4 : ¬simplest_radical_form (√(1/3:ℚ))
) : (√(10:ℚ)) = (√(10:ℚ)) :=
by
  sorry

end simplest_radical_form_of_given_l297_297205


namespace count_subsets_5_or_6_only_but_not_both_l297_297578

open Finset 

theorem count_subsets_5_or_6_only_but_not_both :
  let s := {1, 2, 3, 4, 5, 6}
  ∃ count, (count = 32 ∧ ∀ A ⊆ s, (5 ∈ A ∧ 6 ∉ A ∨ 6 ∈ A ∧ 5 ∉ A) → count = card (powerset s).filter (λ A, (5 ∈ A ∧ 6 ∉ A) ∨ (6 ∈ A ∧ 5 ∉ A))) :=
by
  sorry

end count_subsets_5_or_6_only_but_not_both_l297_297578


namespace solve_for_t_l297_297498

-- Definition of the equation
def equation (t : ℝ) : Prop :=
  2 * 2^t + sqrt (4 * 4^t) = 20

-- The main theorem to prove
theorem solve_for_t : ∃ t : ℝ, equation t ∧ t = real.log 5 / real.log 2 :=
by
  sorry

end solve_for_t_l297_297498


namespace tangent_line_at_origin_eq_range_of_a_l297_297329

-- Mathematical definitions
def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x^2 + 1

-- Part Ⅰ: Equation of the tangent line at (0, f(0)) which is (0, 2)
theorem tangent_line_at_origin_eq (a : ℝ) : 
  tangent_line (f a) 0 = fun x => x + 2 := 
sorry 

-- Part Ⅱ: Range of a given f(x) ≥ 2 for all x ∈ [0,1]
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc (0:ℝ) 1, f a x ≥ 2) -> a ≤ exp 1 - 1 := 
sorry

end tangent_line_at_origin_eq_range_of_a_l297_297329


namespace lucas_change_l297_297841

-- Define the costs of items and the initial amount.
def initial_amount : ℝ := 20.00
def cost_avocados : ℝ := 1.50 + 2.25 + 3.00
def cost_water : ℝ := 2 * 1.75
def cost_apples : ℝ := 4 * 0.75

-- Define the total cost.
def total_cost : ℝ := cost_avocados + cost_water + cost_apples

-- Define the expected change.
def expected_change : ℝ := initial_amount - total_cost

-- The proposition (statement) we want to prove.
theorem lucas_change : expected_change = 6.75 :=
by
  sorry -- Proof to be completed.

end lucas_change_l297_297841


namespace congruent_triangles_of_same_inradius_l297_297285

open Classical

section
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

noncomputable def inradius (a b c : A) : ℝ := sorry

theorem congruent_triangles_of_same_inradius
  (h1 : inradius A B C = inradius B C D)
  (h2 : inradius A B C = inradius A C D)
  (h3 : inradius A B C = inradius A B D)
  (h4 : inradius D B C = inradius A C D) :
  (∀ (A B C D : A), ∃ k : ℝ, inradius A B C = k  ∧ inradius B C D = k ∧ inradius A C D = k ∧ inradius A B D = k) → 
  (is_congruent (triangle A B C) (triangle A B D) ∧ 
   is_congruent (triangle A B C) (triangle A C D) ∧ 
   is_congruent (triangle A B C) (triangle B C D)) :=
by sorry

end

end congruent_triangles_of_same_inradius_l297_297285


namespace variance_of_planted_trees_l297_297046

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l297_297046


namespace probability_AHL_1_project_range_of_P2_after_six_projects_l297_297535

-- Condition Definitions
variable (P1 : ℚ) (P2 : ℚ)
def P1_value := P1 = 2 / 3

-- Question (I)
theorem probability_AHL_1_project (P1_eq: P1_value P1) (P2_eq: P2 = 1 / 2) :
  let prob_A_1_task := (2 / 3) * (1 / 3)
  let prob_B_1_task := (1 / 2) * (1 / 2)
  let prob_1_1_tasks := prob_A_1_task * prob_B_1_task

  let prob_A_2_tasks := (2 / 3) * (2 / 3)
  let prob_B_2_tasks := (1 / 2) * (1 / 2)
  let prob_2_2_tasks := prob_A_2_tasks * prob_B_2_tasks

  let total_prob := prob_1_1_tasks + prob_2_2_tasks
  total_prob = 1 / 6 := by sorry

-- Define expression for probability of being awarded AHL title per project
def AHL_probability (P1 P2 : ℚ) :=
  ((2 / 3) * (1 / 3) * (C (2 - 1) 1) * P2 * (1 - P2)) + ((2 / 3) * (2 / 3) * P2 * P2)

-- Question (II)
theorem range_of_P2_after_six_projects (P1_eq: P1_value P1) (E_xi : ℚ):
  (∀ P2, (6 * AHL_probability P1 P2 >= 2.5) → (3 / 4 <= P2 ∧ P2 <= 1)) := by sorry

end probability_AHL_1_project_range_of_P2_after_six_projects_l297_297535


namespace find_discarded_number_l297_297512

noncomputable def discarded_number (S : ℕ) (X : ℕ) (sum50 : S = 50 * 38) (cond : (S - X - 55) / 48 = 37.5) : ℕ :=
  X

theorem find_discarded_number (X : ℕ) (h_S : (50 * 38) = 1900) (h_cond : (1900 - X - 55) / 48 = 37.5) : X = 45 :=
  by
    unfold discarded_number
    sorry

end find_discarded_number_l297_297512


namespace gcd_5280_12155_l297_297159

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 5 :=
by
  sorry

end gcd_5280_12155_l297_297159


namespace find_positive_integer_l297_297256

theorem find_positive_integer (n : ℕ) :
  (arctan (1 / 2) + arctan (1 / 3) + arctan (1 / 7) + arctan (1 / n) = π / 4) → n = 7 :=
by
  sorry

end find_positive_integer_l297_297256


namespace pizza_toppings_count_l297_297183

theorem pizza_toppings_count : 
  let toppings := 8 in
  (toppings + (Nat.choose toppings 2) + (Nat.choose toppings 3)) = 92 := 
by 
  let toppings := 8
  have h1 : (Nat.choose toppings 2) = 28 := by sorry
  have h2 : (Nat.choose toppings 3) = 56 := by sorry
  sorry

end pizza_toppings_count_l297_297183


namespace arithmetic_sequence_formula_sum_Tn_formula_l297_297321

variable {a : ℕ → ℤ} -- The sequence a_n
variable {S : ℕ → ℤ} -- The sum S_n
variable {a₃ : ℤ} (h₁ : a₃ = 20)
variable {S₃ S₄ : ℤ} (h₂ : 2 * S₃ = S₄ + 8)

/- The general formula for the arithmetic sequence a_n -/
theorem arithmetic_sequence_formula (d : ℤ) (a₁ : ℤ)
  (h₃ : (a₃ = a₁ + 2 * d))
  (h₄ : (S₃ = 3 * a₁ + 3 * d))
  (h₅ : (S₄ = 4 * a₁ + 6 * d)) :
  ∀ n : ℕ, a n = 8 * n - 4 :=
by
  sorry

variable {b : ℕ → ℚ} -- Define b_n
variable {T : ℕ → ℚ} -- Define T_n
variable {S_general : ℕ → ℚ} (h₆ : ∀ n, S n = 4 * n ^ 2)
variable {b_general : ℚ → ℚ} (h₇ : ∀ n, b n = 1 / (S n - 1))
variable {T_general : ℕ → ℚ} -- Define T_n

/- The formula for T_n given b_n -/
theorem sum_Tn_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
by
  sorry

end arithmetic_sequence_formula_sum_Tn_formula_l297_297321


namespace max_value_of_f_l297_297311

noncomputable def f (a x : ℝ) : ℝ := a * sqrt (1 - x^2) + sqrt (1 + x) + sqrt (1 - x)

def g (a : ℝ) : ℝ :=
  if a ≤ - sqrt 2 / 2 then sqrt 2
  else if - sqrt 2 / 2 < a ∧ a ≤ - 1 / 2 then -a - 1 / (2 * a)
  else if - 1 / 2 < a ∧ a < 0 then a + 2
  else -1  -- technically not needed but Lean requires a branch for all ℝ

theorem max_value_of_f (a : ℝ) (h : a < 0) : 
  ∃ x, f a x = 
  (if a ≤ - sqrt 2 / 2 then sqrt 2
   else if - sqrt 2 / 2 < a ∧ a ≤ - 1 / 2 then -a - 1 / (2 * a)
   else a + 2) :=
sorry

end max_value_of_f_l297_297311


namespace evaluate_expression_l297_297672

theorem evaluate_expression : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := 
by 
  sorry

end evaluate_expression_l297_297672


namespace unique_solution_condition_l297_297304

theorem unique_solution_condition (a b c : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 :=
sorry

end unique_solution_condition_l297_297304


namespace find_x_l297_297192

noncomputable def square_side_length : ℝ := 2
noncomputable def total_area : ℝ := square_side_length ^ 2
noncomputable def area_trapezoid (x : ℝ) : ℝ := 1 / 2 * (x + 1)
noncomputable def pentagon_area (A: ℝ) : ℝ := A / 2

theorem find_x :
  ∃ (x : ℝ), 
  (let A := 2 * pentagon_area ((total_area / 5)) in 
   (area_trapezoid x = A) ∧
   (5 * (pentagon_area ((total_area / 5)) = total_area))) :=
begin
  sorry
end

end find_x_l297_297192


namespace value_of_infinite_expression_l297_297793

theorem value_of_infinite_expression :
  ∃ t > 0, t = 1 + (1 / (1 + 1 / (1 + 1 / (1 + 1 / (1 + 1 / ...))))) ∧ t = (Real.sqrt 5 + 1) / 2 := sorry

end value_of_infinite_expression_l297_297793


namespace find_function_l297_297694

theorem find_function (f : ℝ → ℝ) : (∀ x : ℝ, f((1-x)/(1+x)) = x) → (f x = (1-x)/(1+x)) := by
  intros h x
  sorry

end find_function_l297_297694


namespace exists_infinite_relatively_prime_subset_of_form2k_minus3_l297_297859

-- Define the given condition of the problem
def form2k_minus3 (k : ℕ) : ℤ := 2^k - 3

-- Define the necessary properties to be proven
def infinite_relatively_prime_subset : Prop :=
  ∃ S : Set ℤ, Set.Infinite S ∧ (∀ a b ∈ S, a ≠ b → Nat.gcd a.natAbs b.natAbs = 1)

-- Translate the mathematical problem to Lean using the above definitions
theorem exists_infinite_relatively_prime_subset_of_form2k_minus3 :
  infinite_relatively_prime_subset (form2k_minus3 '' {k : ℕ | k ≥ 2}) :=
sorry

end exists_infinite_relatively_prime_subset_of_form2k_minus3_l297_297859


namespace calc_expression_l297_297266

theorem calc_expression : 5 + 2 * (8 - 3) = 15 :=
by
  -- Proof steps would go here
  sorry

end calc_expression_l297_297266


namespace fixed_point_exists_l297_297542

open_locale classical
noncomputable theory 

def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
{ p | dist p center = radius }

variables {O1 O2 A : ℝ × ℝ} {r1 r2 : ℝ}
variables (ω1 := circle O1 r1)
variables (ω2 := circle O2 r2)
variables (X Y : ℝ × ℝ)

-- Assuming that A is an intersection point of ω1 and ω2
def intersection (A : ℝ × ℝ) (ω1 ω2 : set (ℝ × ℝ)) : Prop :=
A ∈ ω1 ∧ A ∈ ω2

-- Assuming that X moves on ω1 and Y moves on ω2, both starting from A
def moves_on_circle (t : ℝ) : Prop :=
X t ∈ ω1 ∧ Y t ∈ ω2

-- Assuming X and Y return to A simultaneously after one revolution
def returns_to_A_simultaneously (T : ℝ) : Prop :=
X T = A ∧ Y T = A

-- The statement to be proved
theorem fixed_point_exists (h_intersection : intersection A ω1 ω2)
  (h_moves_on_circle : ∀ t, moves_on_circle t)
  (h_returns_to_A : returns_to_A_simultaneously 1) :
  ∃ P : ℝ × ℝ, ∀ t, dist X t P = dist Y t P :=
sorry

end fixed_point_exists_l297_297542


namespace initial_amount_A_l297_297502

theorem initial_amount_A (a b c: ℕ) (final_amount: ℕ) (hB : b = 28) (hC : c = 20) 
    (transaction_rule : ∀ g1 g2 g3, (b + g1 = final_amount ∧ c + g2 = final_amount ∧ g3 = a - b - c) → 
    (a, b, c) = (2*(a - b - c), 232 - 2*(a - b - c), final_amount)) :
    a = 54 :=
by
  -- conditions to skip the proof
  intros,
  sorry

end initial_amount_A_l297_297502


namespace proof_maximum_area_l297_297792

open Real

noncomputable def ellipse_parameters (a b : ℝ) (cond_a_gt_b : a > b) (cond_b_gt_0 : b > 0) 
(cond_eccentricity : 1 / 2 = sqrt (1 - (b^2 / a^2))) 
(cond_pass_through : (1, 3 / 2) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1 }) :
  a = 2 ∧ b = sqrt 3 := sorry

noncomputable def ellipse_equation (a : ℝ) (b : ℝ) 
(cond_a : a = 2) (cond_b : b = sqrt 3) :
  (∀ x y : ℝ, (x - 0)^2 / 4 + (y - 0)^2 / 3 = 1 ↔ (x, y) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1 }) := sorry

noncomputable def maximum_area (k : ℝ) (cond_k_zero : k = 0) : 
abs (2 * sqrt 6 / 3) := sorry

structure ellipse_conditions (a b k : ℝ) :=
(a_gt_b : a > b)
(b_gt_0 : b > 0)
(eccentricity : 1 / 2 = sqrt (1 - (b^2 / a^2)))
(pass_through : (1, 3 / 2) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1 })
(l : ℝ → ℝ := λ x, k * x + 1) 
(T : ℝ × ℝ := (0, 1))

theorem proof_maximum_area (a b k : ℝ) (h : ellipse_conditions a b k) :
  ∃ A B : ℝ × ℝ, ∀ x y : ℝ, h.T.fst = 0 ∧ h.T.snd = 1 → k = 0 → 
  (∃ (l : ℝ → ℝ), |abs (2 * sqrt 6 / 3)|) := sorry

end proof_maximum_area_l297_297792


namespace weight_exceeds_bookcase_limit_l297_297423

def shelf_max_weight := 20
def total_shelves := 4
def bookcase_max_weight := total_shelves * shelf_max_weight

def max_hardcover_books := 70
def hardcover_book_weight_range := (0.5, 1.5)
def max_hardcover_book_weight := max_hardcover_books * 1.5

def max_textbooks := 30
def textbook_weight_range := (1.5, 3)
def max_textbook_weight := max_textbooks * 3

def max_knick_knacks := 10
def knick_knack_weight_range := (5, 8)
def max_knick_knack_weight := max_knick_knacks * 8

def total_max_item_weight := max_hardcover_book_weight + max_textbook_weight + max_knick_knack_weight

theorem weight_exceeds_bookcase_limit :
  total_max_item_weight - bookcase_max_weight = 195 :=
by
  -- We have defined the total maximum weight of items and the bookcase's maximum weight limit.
  -- Here we check that the total excess weight is 195 pounds.
  sorry

end weight_exceeds_bookcase_limit_l297_297423


namespace monotonic_solution_l297_297249

-- Definition of a monotonic function
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- The main theorem
theorem monotonic_solution (f : ℝ → ℝ) 
  (mon : monotonic f) 
  (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = -x) :=
sorry

end monotonic_solution_l297_297249


namespace range_of_m_l297_297388

theorem range_of_m (m x : ℝ) (h1 : (3 * x) / (x - 1) = m / (x - 1) + 2) (h2 : x ≥ 0) (h3 : x ≠ 1) : 
  m ≥ 2 ∧ m ≠ 3 := 
sorry

end range_of_m_l297_297388


namespace probability_of_at_least_one_three_l297_297547

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l297_297547


namespace discount_ratio_correct_l297_297421

-- Define the conditions
def cheaper_pair_price : ℝ := 40
def expensive_pair_price : ℝ := 60
def total_without_discounts : ℝ := cheaper_pair_price + expensive_pair_price
def one_fourth_discount : ℝ := total_without_discounts / 4
def total_after_extra_discount : ℝ := total_without_discounts - one_fourth_discount 
def amount_paid : ℝ := 60
def discount_on_cheaper_pair : ℝ := total_after_extra_discount - amount_paid
def original_price_cheaper_pair : ℝ := cheaper_pair_price
def discount_ratio := discount_on_cheaper_pair / original_price_cheaper_pair

-- State and prove the main theorem
theorem discount_ratio_correct :
  discount_ratio = 3 / 8 :=
by 
  sorry

end discount_ratio_correct_l297_297421


namespace train_length_l297_297195

theorem train_length
  (speed_kmh : ℝ) (speed_kmh = 180)
  (time_s : ℝ) (time_s = 10)
  (speed_ms : ℝ) (speed_ms = speed_kmh * 1000 / 3600) :
  speed_ms * time_s = 500 := 
sorry

end train_length_l297_297195


namespace possible_values_of_inverse_sum_l297_297815

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end possible_values_of_inverse_sum_l297_297815


namespace total_flowers_l297_297872

theorem total_flowers (initial_rosas_flowers andre_gifted_flowers : ℝ) 
  (h1 : initial_rosas_flowers = 67.0) 
  (h2 : andre_gifted_flowers = 90.0) : 
  initial_rosas_flowers + andre_gifted_flowers = 157.0 :=
  by
  sorry

end total_flowers_l297_297872


namespace money_given_to_each_friend_l297_297035

-- Define the conditions
def initial_amount : ℝ := 20.10
def money_spent_on_sweets : ℝ := 1.05
def amount_left : ℝ := 17.05
def number_of_friends : ℝ := 2.0

-- Theorem statement
theorem money_given_to_each_friend :
  (initial_amount - amount_left - money_spent_on_sweets) / number_of_friends = 1.00 :=
by
  sorry

end money_given_to_each_friend_l297_297035


namespace description_of_S_l297_297020

noncomputable def S := {p : ℝ × ℝ | (3 = (p.1 + 2) ∧ p.2 - 5 ≤ 3) ∨ 
                                      (3 = (p.2 - 5) ∧ p.1 + 2 ≤ 3) ∨ 
                                      (p.1 + 2 = p.2 - 5 ∧ 3 ≤ p.1 + 2 ∧ 3 ≤ p.2 - 5)}

theorem description_of_S :
  S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 8) ∨ 
                    (p.2 = 8 ∧ p.1 ≤ 1) ∨ 
                    (p.2 = p.1 + 7 ∧ p.1 ≥ 1 ∧ p.2 ≥ 8)} :=
sorry

end description_of_S_l297_297020


namespace hyperbola_equation_l297_297318

theorem hyperbola_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∀ {x y : ℝ}, x^2 / 12 + y^2 / 4 = 1 → True) →
  (∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1 → True) →
  (∀ {x y : ℝ}, y = Real.sqrt 3 * x → True) →
  (∃ k : ℝ, 4 < k ∧ k < 12 ∧ 2 = 12 - k ∧ 6 = k - 4) →
  a = 2 ∧ b = 6 := by
  intros h_ellipse h_hyperbola h_asymptote h_k
  sorry

end hyperbola_equation_l297_297318


namespace min_moves_to_zero_l297_297932

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l297_297932


namespace sum_of_integer_values_of_a_l297_297339

open Real

noncomputable def f (x : ℝ) : ℝ :=
  abs (x + 1) + abs (x + 2) + abs (x - 1) + abs (x - 2)

theorem sum_of_integer_values_of_a :
  (∃ a : ℤ, f (a^2 - 3 * (a : ℝ) + 2) = f (a - 1) ∧ a ∈ {1, 2, 3}) →
  (1 + 2 + 3 = 6) :=
by
  sorry

end sum_of_integer_values_of_a_l297_297339


namespace keystone_arch_trapezoid_angle_l297_297788

theorem keystone_arch_trapezoid_angle 
  (n : ℕ)
  (is_isosceles : ∀ i, 1 ≤ i ∧ i ≤ n → is_isosceles_trapezoid (T i))
  (congruent : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → is_congruent (T i) (T j))
  (assembled_properly : assembled_properly T)
  (bottom_sides_horizontal : bottom_sides_horizontal T) : 
  ∃ x, x ≈ 102.8571 ∧ is_larger_interior_angle_of_trapezoid x :=
begin
  sorry
end

end keystone_arch_trapezoid_angle_l297_297788


namespace total_sleep_correct_l297_297428

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l297_297428


namespace binary_mul_correct_l297_297263

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l297_297263


namespace square_perimeter_from_triangle_l297_297933

variables {a b c: ℝ} (s: ℝ)

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + c^2 = b^2

def triangle_area (a b : ℝ) : ℝ := 0.5 * a * b

theorem square_perimeter_from_triangle 
  (h1: a = 5) (h2: b = 12) (h3: c = 13) : 
  is_right_triangle a b c → 
  triangle_area a b = s^2 → 
  4 * s = 4 * real.sqrt 30 :=
sorry

end square_perimeter_from_triangle_l297_297933


namespace smallest_angle_of_triangle_l297_297508

theorem smallest_angle_of_triangle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) : 3 * x = 45 :=
by
  sorry

end smallest_angle_of_triangle_l297_297508


namespace petya_wins_when_n_is_odd_l297_297627

theorem petya_wins_when_n_is_odd (n : ℕ) :
  (∃ (is_regular_2n_gon : ∀ (k : ℕ), k < 2*n → Prop) (is_empty_cup : ∀ (k : ℕ), k < 2*n → Prop)
    (can_fill_two_adj_or_symmetric : ∀ (k m : ℕ), k < 2*n → m < 2*n → Prop)
    (peter_starts_first : ∀ (k : ℕ), k = 0 → Prop)
    (player_loses_if_no_move : ∀ (player : ℕ), (player = 0 ∨ player = 1) → Prop),
  odd n → Petya_wins) :=
by
  sorry

end petya_wins_when_n_is_odd_l297_297627


namespace problem_solution_l297_297381

def complex_i : ℂ := Complex.I
def x : ℂ := (2 - complex_i * Real.sqrt 3) / 3

theorem problem_solution : (1 / (x^2 - x)) = (9 + 18 * complex_i * Real.sqrt 3) / 13 := 
sorry

end problem_solution_l297_297381


namespace y_in_terms_of_x_l297_297726

theorem y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = -2 * x + 5 :=
sorry

end y_in_terms_of_x_l297_297726


namespace odd_factors_count_l297_297367

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l297_297367


namespace travel_distance_proof_l297_297203

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l297_297203


namespace exists_integer_lt_sqrt_10_l297_297481

theorem exists_integer_lt_sqrt_10 : ∃ k : ℤ, k < Real.sqrt 10 := by
  have h_sqrt_bounds : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
    -- Proof involving basic properties and calculations
    sorry
  exact ⟨3, h_sqrt_bounds.left⟩

end exists_integer_lt_sqrt_10_l297_297481


namespace rolling_cube_dot_path_l297_297590

theorem rolling_cube_dot_path (a b c : ℝ) (h_edge : a = 1) (h_dot_top : True):
  c = (1 + Real.sqrt 5) / 2 := by
  sorry

end rolling_cube_dot_path_l297_297590


namespace container_capacity_l297_297418

theorem container_capacity (C : ℝ) (h₁ : C > 15) (h₂ : 0 < (81 : ℝ)) (h₃ : (337 : ℝ) > 0) :
  ((C - 15) / C) ^ 4 = 81 / 337 :=
sorry

end container_capacity_l297_297418


namespace valid_subsets_count_l297_297808

open Nat Subset

-- Definitions and conditions
def S (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2 * n}
def valid_subset (n : ℕ) (T : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → |a - b| ≠ 1 ∧ |a - b| ≠ n

-- Final proof statement (no proof, just the statement)
theorem valid_subsets_count (n : ℕ) (n_pos : n > 0) :
  ∃ (a_n b_n : ℕ), (∀ T : Set ℕ, T ⊆ S n → valid_subset n T) →
  (a_n - b_n = number_of_valid_subsets n) :=
sorry

end valid_subsets_count_l297_297808


namespace impossible_6x6_table_sums_l297_297005

open Function

theorem impossible_6x6_table_sums :
  ¬ ∃ (f : Fin 6 → Fin 6 → ℕ), 
    (Injective (λ (i : Fin 6 × Fin 6), f i.1 i.2)) ∧
    (∀ (i : Fin 6) (j : Fin 6) (dir : Bool),
       let rect := if dir then (λ k : Fin 5, f i (j + k)) else (λ k : Fin 5, f (i + k) j)
       in (sum (λ k, rect k) = 2022 ∨ sum (λ k, rect k) = 2023)) :=
sorry

end impossible_6x6_table_sums_l297_297005


namespace added_classes_l297_297886

def original_classes := 15
def students_per_class := 20
def new_total_students := 400

theorem added_classes : 
  new_total_students = original_classes * students_per_class + 5 * students_per_class :=
by
  sorry

end added_classes_l297_297886


namespace seventy_second_number_in_S_is_573_l297_297961

open Nat

def S : Set Nat := { k | k % 8 = 5 }

theorem seventy_second_number_in_S_is_573 : ∃ k ∈ (Finset.range 650), k = 8 * 71 + 5 :=
by
  sorry -- Proof goes here

end seventy_second_number_in_S_is_573_l297_297961


namespace y_intercept_correct_l297_297096

variable (m : ℝ) (x1 y1 : ℝ)
-- Given conditions
def slope := 3
def x_intercept := (4, 0 : ℝ)
-- Expected outcome
def y_intercept := (0, -12 : ℝ)

theorem y_intercept_correct :
  m = slope →
  x_intercept = (x1, y1) →
  y_intercept = (0, 3 * 0 - 12 : ℝ) :=
by
  -- Proof steps go here (omitted)
  sorry

end y_intercept_correct_l297_297096


namespace product_of_sums_of_conjugates_l297_297641

theorem product_of_sums_of_conjugates :
  let a := 8 - Real.sqrt 500
  let b := 8 + Real.sqrt 500
  let c := 12 - Real.sqrt 72
  let d := 12 + Real.sqrt 72
  (a + b) * (c + d) = 384 :=
by
  sorry

end product_of_sums_of_conjugates_l297_297641


namespace measure_angle_BAC_l297_297001

theorem measure_angle_BAC
  {A B C X Y : Type}
  (AX XY YB BC : ℝ)
  (h1 : AX = XY)
  (h2 : XY = YB)
  (h3 : YB = BC)
  (h_ABC : ∠ABC = 150) : ∠BAC = 10 :=
sorry

end measure_angle_BAC_l297_297001


namespace can_form_isosceles_triangle_with_given_sides_l297_297140

-- Define a structure for the sides of a triangle
structure Triangle (α : Type _) :=
  (a b c : α)

-- Define the predicate for the triangle inequality
def triangle_inequality {α : Type _} [LinearOrder α] [Add α] (t : Triangle α) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

-- Define the predicate for an isosceles triangle
def is_isosceles {α : Type _} [DecidableEq α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the main theorem which checks if the given sides can form an isosceles triangle
theorem can_form_isosceles_triangle_with_given_sides
  (t : Triangle ℕ)
  (h_tri : triangle_inequality t)
  (h_iso : is_isosceles t) :
  t = ⟨2, 2, 1⟩ :=
  sorry

end can_form_isosceles_triangle_with_given_sides_l297_297140


namespace problem_I_problem_II_problem_III_l297_297275

-- (I) Statement
theorem problem_I : (seq : Fin 10 → Fin 2) 
  (h_seq : seq = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1])
  : (∃ k, 2 ≤ k ∧ k ≤ 9 ∧ ((seq 0) == (seq k) ∧ (seq 1) == (seq k+1) ∧ (seq 2) == (seq k+2) ∧ (seq 3) == (seq k+3) ∧ (seq 4) == (seq k+4))) := 
sorry

-- (II) Statement
theorem problem_II : (n : ℕ)
  (∀ (seq: Fin n → Fin 2), 
     (∃ k : ℕ, 3 ≤ k ∧ k ≤ n - 1 ∧ ∀ i j, seq i = seq j → seq (i+1 mod n) = seq (j+1 mod n) ∧ seq (i+2 mod n) = seq (j+2 mod n) ∧ seq (i+3 mod n) = seq (j+3 mod n))) :=
sorry

-- (III) Statement
theorem problem_III : (a : Fin 4 → Fin 2) (a_4 : Fin 2) (h_neq : a 3 ≠ 1)
  (seq : Fin 5 → Fin 2) 
  (h_seq : seq = [a 0, a 1, a 2, a 3, a_4])
  : ∃ (new_term : Fin 2), 
    (seq_new : Fin 6 → Fin 2)
  (h_new_seq : seq_new = [a 0, a 1, a 2, a 3, a_4, new_term] ∧ 
  ∃ i j, i ≠ j ∧ (seq_new i) == (seq_new j) ∧ (seq_new (i+1)) == (seq_new (j+1)) ∧ (seq_new (i+2)) == (seq_new (j+2)) ∧ (seq_new (i+3)) == (seq_new (j+3)) ∧ (seq_new (i+4)) == (seq_new (j+4))) :=
sorry

end problem_I_problem_II_problem_III_l297_297275


namespace geometric_sequence_sum_l297_297070

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q a2005 a2006 : ℝ), 
    (∀ n, a (n + 1) = a n * q) ∧
    q > 1 ∧
    a2005 + a2006 = 2 ∧ 
    a2005 * a2006 = 3 / 4 ∧ 
    a (2005) = a2005 ∧ 
    a (2006) = a2006 → 
    a (2007) + a (2008) = 18 := 
by
  sorry

end geometric_sequence_sum_l297_297070


namespace jane_chickens_l297_297422

-- Conditions
def eggs_per_chicken_per_week : ℕ := 6
def egg_price_per_dozen : ℕ := 2
def total_income_in_2_weeks : ℕ := 20

-- Mathematical problem
theorem jane_chickens : (total_income_in_2_weeks / egg_price_per_dozen) * 12 / (eggs_per_chicken_per_week * 2) = 10 :=
by
  sorry

end jane_chickens_l297_297422


namespace sin_x_value_l297_297308

theorem sin_x_value (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : Real.sin x = 8 / 17 :=
sorry

end sin_x_value_l297_297308


namespace light_travel_distance_in_km_l297_297099

-- Define the conditions
def speed_of_light_miles_per_sec : ℝ := 186282
def conversion_factor_mile_to_km : ℝ := 1.609
def time_seconds : ℕ := 500
def expected_distance_km : ℝ := 1.498 * 10^8

-- The theorem we need to prove
theorem light_travel_distance_in_km :
  (speed_of_light_miles_per_sec * time_seconds * conversion_factor_mile_to_km) = expected_distance_km :=
  sorry

end light_travel_distance_in_km_l297_297099


namespace painting_time_l297_297269

noncomputable def rate_of_regular_painter := ℝ
noncomputable def job := ℝ
def work_days : ℝ := 1.8
def regular_painter_rate (r : rate_of_regular_painter) := r
def efficient_painter_rate (r : rate_of_regular_painter) := 2 * r
def total_work (r : rate_of_regular_painter) := 5 * r * work_days

theorem painting_time (r : rate_of_regular_painter) (h : r ≠ 0) :
  (total_work r) / (regular_painter_rate r + efficient_painter_rate r) = 3 :=
by
  linear_combination
  calc sorry

end painting_time_l297_297269


namespace find_point_P_l297_297017

-- Definitions of the given points A, B, C, and D
def A := (8, 0, 0 : ℝ × ℝ × ℝ)
def B := (0, -4, 0 : ℝ × ℝ × ℝ)
def C := (0, 0, 6 : ℝ × ℝ × ℝ)
def D := (0, 0, 0 : ℝ × ℝ × ℝ)

-- Proof statement: P = (4, -2, 3) is the point such that AP = BP = CP = DP
theorem find_point_P : ∃ (P : ℝ × ℝ × ℝ), 
  (dist P A = dist P B) ∧ (dist P B = dist P C) ∧ (dist P C = dist P D) ∧ P = (4, -2, 3) :=
by 
  -- Proof is omitted; this line is needed to ensure the Lean statement builds
  sorry

end find_point_P_l297_297017


namespace math_problem_l297_297911

noncomputable def a_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  a 1 = 1 ∧ 
  ∀ n ≥ 2, a n = (2 * (S n) ^ 2) / (2 * S n - 1)

noncomputable def problem_1 (S : ℕ → ℝ) :=
  ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = 2

noncomputable def problem_2 (a S : ℕ → ℝ) :=
  ∀ n ≥ 2, a n = -2 / ((2 * n - 1) * (2 * n - 3))

noncomputable def problem_3 (S : ℕ → ℝ) (k : ℝ) :=
  (1 + S 1) * (1 + S 2) * ... * (1 + S n) ≥ k * (√(2 * n + 1))

theorem math_problem (a S : ℕ → ℝ) (k : ℝ) :
  a_sequence a S →
  problem_1 S →
  problem_2 a S →
  problem_3 S k → 
  k ≤ 2 * (√3) / 3 := sorry

end math_problem_l297_297911


namespace no_polynomial_P_exists_l297_297803

theorem no_polynomial_P_exists :
    ¬ (∃ (P : ℕ → ℕ), ∀ n ≥ 0, P n = (∑ i in range (n + 1), (⌊ (i : ℝ).sqrt ⌋ : ℕ))) :=
begin
  sorry
end

end no_polynomial_P_exists_l297_297803


namespace min_jars_needed_l297_297990

theorem min_jars_needed (capacity_medium_jar : ℕ) (capacity_large_container : ℕ) (potential_loss : ℕ)
                        (capacity_medium_jar_eq : capacity_medium_jar = 50)
                        (capacity_large_container_eq : capacity_large_container = 825)
                        (potential_loss_eq : potential_loss ≤ 1) :
  18 ≤ (ceil (capacity_large_container / capacity_medium_jar : ℚ)).natAbs + potential_loss := by
  sorry

end min_jars_needed_l297_297990


namespace largest_value_when_changing_first_digit_of_05123_to_8_l297_297645

def change_digit_to_eight (d : ℕ → ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => 8.05123
  | 1 => 0.08123
  | 2 => 0.05823
  | 3 => 0.05183
  | 4 => 0.05128
  | _ => d n

theorem largest_value_when_changing_first_digit_of_05123_to_8 :
  ∀ n, change_digit_to_eight (λ _, 0) n ≤ change_digit_to_eight (λ _, 0) 0 :=
by
  sorry

end largest_value_when_changing_first_digit_of_05123_to_8_l297_297645


namespace find_N_such_that_P_is_prime_l297_297825

noncomputable def d (N : ℕ) : ℕ :=
  (Finset.range (N + 1)).filter (N % · = 0).card

def isPrime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def P (N : ℕ) : ℕ := N / d N

theorem find_N_such_that_P_is_prime
  (N : ℕ)
  (h : N = 8 ∨ N = 9 ∨ N = 12 ∨ N = 18 ∨ N = 24
    ∨ ∃ p : ℕ, p > 3 ∧ isPrime p ∧ (N = 8 * p ∨ N = 12 * p ∨ N = 18 * p)) :
  isPrime (P N) :=
sorry

end find_N_such_that_P_is_prime_l297_297825


namespace max_min_values_of_g_l297_297022

noncomputable def f (x : ℝ) : ℝ := (1 - (1 / x)^2 + 2 * (1 / x)) / (1 + (1 / x)^2)

noncomputable def g (x : ℝ) : ℝ := f(x) * f(1 - x)

theorem max_min_values_of_g :
  ∃ (max min : ℝ), (∀ (x : ℝ), (-1 ≤ x ∧ x ≤ 1) → g(x) ≤ max ∧ g(x) ≥ min) ∧
  max = 1 / 25 ∧ min = 4 - real.sqrt 34 :=
sorry

end max_min_values_of_g_l297_297022


namespace incorrect_reasoning_C_l297_297139

theorem incorrect_reasoning_C
  {Point : Type} {Line Plane : Type}
  (A B : Point) (l : Line) (α β : Plane)
  (in_line : Point → Line → Prop)
  (in_plane : Point → Plane → Prop)
  (line_in_plane : Line → Plane → Prop)
  (disjoint : Line → Plane → Prop) :

  ¬(line_in_plane l α) ∧ in_line A l ∧ in_plane A α :=
sorry

end incorrect_reasoning_C_l297_297139


namespace total_value_of_pile_l297_297690

def value_of_pile (total_coins dimes : ℕ) (value_dime value_nickel : ℝ) : ℝ :=
  let nickels := total_coins - dimes
  let value_dimes := dimes * value_dime
  let value_nickels := nickels * value_nickel
  value_dimes + value_nickels

theorem total_value_of_pile :
  value_of_pile 50 14 0.10 0.05 = 3.20 := by
  sorry

end total_value_of_pile_l297_297690


namespace xiaozhang_participates_in_martial_arts_l297_297887

theorem xiaozhang_participates_in_martial_arts
  (row : Prop) (shoot : Prop) (martial : Prop)
  (Zhang Wang Li: Prop → Prop)
  (H1 : ¬  Zhang row ∧ ¬ Wang row)
  (H2 : ∃ (n m : ℕ), Zhang (shoot ∨ martial) = (n > 0) ∧ Wang (shoot ∨ martial) = (m > 0) ∧ m = n + 1)
  (H3 : ¬ Li shoot ∧ (Li martial ∨ Li row)) :
  Zhang martial :=
by
  sorry

end xiaozhang_participates_in_martial_arts_l297_297887


namespace previous_year_rankings_l297_297103

theorem previous_year_rankings :
  ∃ (H I G : String),
    (∃ (E : String), E ≠ H ∧ E ≠ I ∧ E ≠ G ∧ (H = "H" ∨ H = "I" ∨ H = "G")
                      ∧ (I = "H" ∨ I = "I" ∨ I = "G")
                      ∧ (G = "H" ∨ G = "I" ∨ G = "G")
                      ∧ "C" = "C" ∧ (H I G = "HIG" ∨ H I G = "EIG"))
  ∨ (H I G = "HIG" ∨ H I G = "EIG") :=
sorry

end previous_year_rankings_l297_297103


namespace nurse_count_l297_297107

theorem nurse_count (total people : ℕ) (ratio_d_n : ℕ × ℕ) (td : total people = 500) (r : ratio_d_n = (7, 8)) :
  ∃ n : ℕ, n = 264 :=
by
  sorry


end nurse_count_l297_297107


namespace fibonacci_product_l297_297447

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_product : 
  (∏ k in (finset.range 149).map (λ x, x + 2), 
    (fibonacci k / fibonacci (k - 1)) - (fibonacci k / fibonacci (k + 1))) = 
  (fibonacci 150 / fibonacci 151) :=
by
  sorry

end fibonacci_product_l297_297447


namespace chord_lengths_product_l297_297030

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 18)

def A : ℂ := 3
def B : ℂ := -3
def C (k : ℕ) : ℂ := 3 * omega^k

def AC_length (k : ℕ) : ℝ := Complex.abs (A - C k)
def BC_length (k : ℕ) : ℝ := Complex.abs (B - C k)

def chord_product : ℝ :=
  ∏ k in Finset.range 8, AC_length (k + 1) * BC_length (k + 1)

theorem chord_lengths_product :
  chord_product = 4782969 := 
sorry

end chord_lengths_product_l297_297030


namespace annual_interest_rate_l297_297618

theorem annual_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (n : ℕ)
  (hP : P = 20000)
  (hA : A = 60000)
  (ht : t = 10)
  (hn : n = 1) :
  ∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r ≈ (11.6123 / 100) :=
by
  sorry

end annual_interest_rate_l297_297618


namespace cubic_yards_to_cubic_feet_l297_297349

def conversion_factor := 3 -- 1 yard = 3 feet
def cubic_conversion := conversion_factor ^ 3 -- 1 cubic yard = (3 feet) ^ 3

theorem cubic_yards_to_cubic_feet :
  5 * cubic_conversion = 135 :=
by
  unfold conversion_factor cubic_conversion
  norm_num
  sorry

end cubic_yards_to_cubic_feet_l297_297349


namespace angle_CAD_in_convex_quadrilateral_l297_297777

theorem angle_CAD_in_convex_quadrilateral
  (ABCD : Type)
  [convex_quadrilateral ABCD]
  (A B C D : ABCD)
  (h1 : AB = BD)
  (h2 : ∠A = 65)
  (h3 : ∠B = 80)
  (h4 : ∠C = 75)
  : ∠CAD = 15 := sorry

end angle_CAD_in_convex_quadrilateral_l297_297777


namespace find_angle_CAD_l297_297771

-- Definitions representing the problem conditions
variables {A B C D : Type} [has_angle A] [has_angle B] [has_angle C] [has_angle D]

-- Given conditions
def is_convex (A B C D : Type) : Prop := sorry  -- convex quadrilateral ABCD
def angle_A (A : Type) : ℝ := 65
def angle_B (B : Type) : ℝ := 80
def angle_C (C : Type) : ℝ := 75
def equal_sides (A B D : Type) : Prop := sorry  -- AB = BD

-- Theorem statement: Given the conditions, prove the desired angle
theorem find_angle_CAD {A B C D : Type}
  [is_convex A B C D] 
  [equal_sides A B D]
  (h1 : angle_A A = 65)
  (h2 : angle_B B = 80)
  (h3 : angle_C C = 75)
  : ∃ (CAD : ℝ), CAD = 15 := 
sorry -- proof omitted

end find_angle_CAD_l297_297771


namespace ellipse_standard_equation_and_cosine_l297_297316

theorem ellipse_standard_equation_and_cosine :
  (∀ P : ℝ × ℝ, (P.2^2) / 4 + (P.1^2) / 3 = 1) ∧
  (∀ P : ℝ × ℝ, (P.2^2) / 4 + (P.1^2) / 3 = 1 ∧
    abs (dist P (0, -1)) - abs (dist P (0, 1)) = 1 → 
    (∃ PF1 PF2 : ℝ, PF1 = 5/2 ∧ PF2 = 3/2 ∧
      cos (∠ (0, -1) P (0, 1)) = 3 / 5)) :=
sorry

end ellipse_standard_equation_and_cosine_l297_297316


namespace isosceles_triangle_l297_297856

variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β))
variable (triangle_angles : γ = π - (α + β))

theorem isosceles_triangle : α = β :=
by
  sorry

end isosceles_triangle_l297_297856


namespace sheila_will_attend_the_picnic_l297_297496

noncomputable def sheila_attendance_probability : ℝ :=
  let P_rain := 0.4 in
  let P_attend_if_rain := 0.2 in
  let P_no_rain := 0.6 in
  let P_attend_if_no_rain := 0.8 in
  (P_rain * P_attend_if_rain) + (P_no_rain * P_attend_if_no_rain)

theorem sheila_will_attend_the_picnic :
  sheila_attendance_probability = 0.56 :=
by
  -- Definitions and calculations omitted for statement-only requirement
  sorry

end sheila_will_attend_the_picnic_l297_297496


namespace determine_multiplier_l297_297607

theorem determine_multiplier (x : ℝ) : 125 * x - 138 = 112 → x = 2 :=
by
  sorry

end determine_multiplier_l297_297607


namespace repeating_decimal_sum_l297_297751

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = 0 then 0 else
    let n := 57 in
    let d := 99 in
    n / d

theorem repeating_decimal_sum (a b : ℤ) (h : 0 < a ∧ 0 < b ∧ Int.gcd a b = 1 ∧ repeating_decimal_to_fraction (a / b) = 0.5757575757) :
  a + b = 52 :=
sorry

end repeating_decimal_sum_l297_297751


namespace constant_distance_O_AB_l297_297790

noncomputable def curveC := {p : ℝ × ℝ // p.1^2 + p.2^2 = 4}

def transformation (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 1/2 * p.2)

noncomputable def curveC' := {p : ℝ × ℝ // p.1^2 + 4 * p.2^2 = 4}

def polar_eq (ρ θ : ℝ) : ℝ := ρ^2 * (1 + 3 * (sin θ)^2)

theorem constant_distance_O_AB (A B : curveC') (h_perp : (polar_eq (A.val.1) (A.val.2)) ⊥ (polar_eq (B.val.1) (B.val.2))) :
  ∃ d : ℝ, d = (2 * sqrt 5) / 5 :=
sorry

end constant_distance_O_AB_l297_297790


namespace probability_third_smallest_five_l297_297063

theorem probability_third_smallest_five :
  (∃ s : Finset ℕ, s.card = 6 ∧ s ⊆ Finset.range 11 ∧
  ∀ a b c d e f : ℕ, 
  [a, b, c, d, e, f].sorted (<) ∧ 
  s = {a, b, c, d, e, f} ∧ 
  c = 5) →
  (card_subsets 6 (Finset.range 11) ≠ 0) →
  30 / 210 = 1 / 7 :=
by
  sorry

end probability_third_smallest_five_l297_297063


namespace triangle_ABD_area_l297_297553

/-- 
   Given a right triangle ABD with AB = 8 and BD = 4,
   the area of the triangle ABD is 16.
-/
theorem triangle_ABD_area : 
  ∀ (AB BD : ℝ), AB = 8 → BD = 4 → 
  (1 / 2) * AB * BD = 16 := 
by
  intros AB BD hAB hBD
  rw [hAB, hBD]
  norm_num
  sorry

end triangle_ABD_area_l297_297553


namespace sequence_periodic_a_n_plus_2_eq_a_n_l297_297157

-- Definition of the sequence and conditions
noncomputable def seq (a : ℕ → ℤ) :=
  ∀ n : ℕ, ∃ α k : ℕ, a n = Int.ofNat (2^α) * k ∧ Int.gcd (Int.ofNat k) 2 = 1 ∧ a (n+1) = Int.ofNat (2^α) - k

-- Definition of periodic sequence
def periodic (a : ℕ → ℤ) (d : ℕ) :=
  ∀ n : ℕ, a (n + d) = a n

-- Proving the desired property
theorem sequence_periodic_a_n_plus_2_eq_a_n (a : ℕ → ℤ) (d : ℕ) (h_seq : seq a) (h_periodic : periodic a d) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end sequence_periodic_a_n_plus_2_eq_a_n_l297_297157


namespace inverse_49_mod_101_l297_297313

theorem inverse_49_mod_101 (h : (7 : ℤ)⁻¹ ≡ (55 : ℤ) [ZMOD 101]) : 
  (49 : ℤ)⁻¹ ≡ (96 : ℤ) [ZMOD 101] := 
  sorry

end inverse_49_mod_101_l297_297313


namespace horner_method_value_at_neg4_l297_297940

def f (x : ℤ) : ℤ := x^6 + 6 * x^4 + 9 * x^2 + 208

theorem horner_method_value_at_neg4 : f (-4) = 22 :=
by
  def v0 := 1
  def v1 := v0 * -4 + 0
  def v2 := v1 * -4 + 6
  have h : v2 = 22, by
    rw [v0, v1, v2]
    exact rfl
  show f (-4) = v2
  sorry

end horner_method_value_at_neg4_l297_297940


namespace point_B_coordinates_l297_297848

theorem point_B_coordinates (A B : ℝ) (hA : A = -2) (hDist : |A - B| = 3) : B = -5 ∨ B = 1 :=
by
  sorry

end point_B_coordinates_l297_297848


namespace total_surface_area_of_three_face_painted_cubes_l297_297166

def cube_side_length : ℕ := 9
def small_cube_side_length : ℕ := 1
def num_small_cubes_with_three_faces_painted : ℕ := 8
def surface_area_of_each_painted_face : ℕ := 6

theorem total_surface_area_of_three_face_painted_cubes :
  num_small_cubes_with_three_faces_painted * surface_area_of_each_painted_face = 48 := by
  sorry

end total_surface_area_of_three_face_painted_cubes_l297_297166


namespace election_total_votes_l297_297400

noncomputable def totalVotes : ℝ := 
  let total_valid_votes := V * 0.85
  let candidate_A_votes_condition := total_valid_votes * 0.70 = 333200
  (333200 / 0.595)

theorem election_total_votes : (333200 / 0.595) ≈ 560000  := 
by
 have h : (333200 / 0.595) ≈ 560000 := by sorry
 exact h

end election_total_votes_l297_297400


namespace travel_distance_proof_l297_297204

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l297_297204


namespace odd_factors_252_l297_297356

theorem odd_factors_252 : 
  {n : ℕ | n ∈ finset.filter (λ d, d % 2 = 1) (finset.divisors 252)}.card = 6 := 
sorry

end odd_factors_252_l297_297356


namespace min_moves_to_reset_counters_l297_297919

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l297_297919


namespace compute_100a_b_l297_297454

theorem compute_100a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) * (x + 10) = 0 ↔ x = -a ∨ x = -b ∨ x = -10)
  (h2 : a ≠ -4 ∧ b ≠ -4 ∧ 10 ≠ -4)
  (h3 : ∀ x : ℝ, (x + 2 * a) * (x + 5) * (x + 8) = 0 ↔ x = -5)
  (hb : b = 8)
  (ha : 2 * a = 5) :
  100 * a + b = 258 := 
sorry

end compute_100a_b_l297_297454


namespace hyperbola_eccentricity_sqrt3_l297_297701

theorem hyperbola_eccentricity_sqrt3
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b / a = Real.sqrt 2) :
  (let e := Real.sqrt (1 + (b^2 / a^2)) in e = Real.sqrt 3) :=
by
  sorry

end hyperbola_eccentricity_sqrt3_l297_297701


namespace ratio_AF_FD_l297_297801

theorem ratio_AF_FD (A B C D E F : Point) (h_trapezoid : Trapezoid A B C D)
    (h_AB_parallel_CD : AB_parallel_CD A B C D)
    (h_ratio_AB_CD : AB_distance A B = 3 * CD_distance C D)
    (h_E_midpoint : E_midpoint_AC E A C)
    (h_BE_intersections_AD_at_F : BE_intersects_AD_at_F B E A D F) :
    ratio_AF_FD A F D = 3 / 2 := 
sorry

end ratio_AF_FD_l297_297801


namespace tim_income_percentage_less_l297_297038

def income_percentage (T M J : ℝ) : ℝ := 100 - (T / J * 100)

theorem tim_income_percentage_less (T M J : ℝ) 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.64 * J) :
  income_percentage T M J = 60 := 
by
  sorry

end tim_income_percentage_less_l297_297038


namespace problem_solution_l297_297448

-- Definitions based on problem conditions
def P : ℕ := -- the maximum sum of products of adjacent elements across all permutations of (1, 2, 3, 4, 5, 6)
def Q : ℕ := -- the number of permutations that achieve this maximum sum

-- The theorem statement
theorem problem_solution : P + Q = 93 :=
sorry

end problem_solution_l297_297448


namespace peter_has_142_nickels_l297_297480

-- Define the conditions
def nickels (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2

-- The theorem to prove the number of nickels
theorem peter_has_142_nickels : ∃ (n : ℕ), nickels n ∧ n = 142 :=
by {
  sorry
}

end peter_has_142_nickels_l297_297480


namespace sum_of_roots_eq_p_l297_297376

variable (p q : ℝ)
variable (hq : q = p^2 - 1)

theorem sum_of_roots_eq_p (h : q = p^2 - 1) : 
  let r1 := p
  let r2 := q
  r1 + r2 = p := 
sorry

end sum_of_roots_eq_p_l297_297376


namespace range_of_m_l297_297466

def A (x : ℝ) := x^2 - 3 * x - 10 ≤ 0
def B (x m : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m (m : ℝ) (h : ∀ x, B x m → A x) : m ≤ 3 := by
  sorry

end range_of_m_l297_297466


namespace minimum_moves_to_reset_counters_l297_297925

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l297_297925


namespace max_value_of_expression_l297_297828

variable (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = 3)

theorem max_value_of_expression :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end max_value_of_expression_l297_297828


namespace term_number_l297_297342

theorem term_number (n : ℕ) : 
  (n ≥ 1) ∧ (5 * Real.sqrt 3 = Real.sqrt (3 + 4 * (n - 1))) → n = 19 :=
by
  intro h
  let h1 := h.1
  let h2 := h.2
  have h3 : (5 * Real.sqrt 3)^2 = (Real.sqrt (3 + 4 * (n - 1)))^2 := by sorry
  sorry

end term_number_l297_297342


namespace geometric_sequence_common_ratio_l297_297317

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
  (h : ∀ n, a n * a (n + 1) = 16^n) :
  ∃ r : ℝ, r = 4 ∧ ∀ n, a (n + 1) = a n * r :=
sorry

end geometric_sequence_common_ratio_l297_297317


namespace ratio_garbage_zane_dewei_l297_297232

-- Define the weights of garbage picked up by Daliah, Dewei, and Zane.
def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah_garbage - 2
def zane_garbage : ℝ := 62

-- The theorem that we need to prove
theorem ratio_garbage_zane_dewei : zane_garbage / dewei_garbage = 4 :=
by
  sorry

end ratio_garbage_zane_dewei_l297_297232


namespace minimum_jumps_to_cover_circle_l297_297106

/--
Given 2016 points arranged in a circle and the ability to jump either 2 or 3 points clockwise,
prove that the minimum number of jumps required to visit every point at least once and return to the starting 
point is 2017.
-/
theorem minimum_jumps_to_cover_circle (n : Nat) (h : n = 2016) : 
  ∃ (a b : Nat), 2 * a + 3 * b = n ∧ (a + b) = 2017 := 
sorry

end minimum_jumps_to_cover_circle_l297_297106


namespace polynomial_coefficient_sum_l297_297382

theorem polynomial_coefficient_sum :
  ∀ (a : Fin 2010 → ℝ),
  (∀ x : ℝ, (1 - 2*x)^2009 = ∑ i in Finset.range 2010, a i * x^i) →
  (∑ i in Finset.range 2009, a (i + 1) * (1 / 2)^(i + 1)) = -1 := 
by
  intro a h
  sorry

end polynomial_coefficient_sum_l297_297382


namespace jason_earnings_l297_297806

theorem jason_earnings :
  let fred_initial := 49
  let jason_initial := 3
  let emily_initial := 25
  let fred_increase := 1.5 
  let jason_increase := 0.625 
  let emily_increase := 0.40 
  let fred_new := fred_initial * fred_increase
  let jason_new := jason_initial * (1 + jason_increase)
  let emily_new := emily_initial * (1 + emily_increase)
  fred_new = fred_initial * fred_increase ->
  jason_new = jason_initial * (1 + jason_increase) ->
  emily_new = emily_initial * (1 + emily_increase) ->
  jason_new - jason_initial == 1.875 :=
by
  intros
  sorry

end jason_earnings_l297_297806


namespace number_of_sides_of_polygon_24_deg_exterior_angle_l297_297519

theorem number_of_sides_of_polygon_24_deg_exterior_angle :
  (∀ (n : ℕ), (∀ (k : ℕ), k = 360 / 24 → n = k)) :=
by
  sorry

end number_of_sides_of_polygon_24_deg_exterior_angle_l297_297519


namespace number_of_odd_factors_of_252_l297_297361

def numOddFactors (n : ℕ) : ℕ :=
  if ∀ d : ℕ, n % d = 0 → ¬(d % 2 = 0) then d
  else 0

theorem number_of_odd_factors_of_252 : numOddFactors 252 = 6 := by
  -- Definition of n
  let n := 252
  -- Factor n into 2^2 * 63
  have h1 : n = 2^2 * 63 := rfl
  -- Find the number of odd factors of 63 since factors of 252 that are odd are the same as factors of 63
  have h2 : 63 = 3^2 * 7 := rfl
  -- Check the number of factors of 63
  sorry

end number_of_odd_factors_of_252_l297_297361


namespace spider_has_eight_legs_l297_297747

-- Define the number of legs a human has
def human_legs : ℕ := 2

-- Define the number of legs for a spider, based on the given condition
def spider_legs : ℕ := 2 * (2 * human_legs)

-- The theorem to be proven, that the spider has 8 legs
theorem spider_has_eight_legs : spider_legs = 8 :=
by
  sorry

end spider_has_eight_legs_l297_297747


namespace monotonic_increasing_interval_f_l297_297091

-- Define the function f(x)
def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 4 * x + 12)

-- Prove that the monotonic increasing interval of f(x) is [-2, 2]
theorem monotonic_increasing_interval_f : 
  ∃ a b, a = -2 ∧ b = 2 ∧ ( ∀ x y, -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ x < y → f x ≤ f y) :=
sorry

end monotonic_increasing_interval_f_l297_297091


namespace function_zero_solution_l297_297273

def floor (x : ℝ) : ℤ := sorry -- Define floor function properly.

theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = (-1) ^ (floor y) * f x + (-1) ^ (floor x) * f y) →
  (∀ x : ℝ, f x = 0) := 
by
  -- Proof goes here
  sorry

end function_zero_solution_l297_297273


namespace hyperbola_eccentricity_l297_297699

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (asymptote : ∀ x, asymptote := (y = √2 * x)) :
  e = √3 := by
  sorry

end hyperbola_eccentricity_l297_297699


namespace problem_l297_297728

noncomputable def problem_statement : Prop :=
  ∀ (a b c : ℝ),
  (∃ (correct : List (ℕ → Prop)),
  (correct = [(λ i, (a = b ∧ b = c → a = c)),
              (λ i, (a ∥ b ∧ b ∥ c → a ∥ c)),
              (λ i, (abs (a * b) = abs a * abs b)),
              (λ i, (b = c → a * b = a * c))] ∧
  (correct = [correct_1, correct_4]))

theorem problem : problem_statement :=
by
  intros
  sorry

end problem_l297_297728


namespace S_sum_eq_one_iff_odd_l297_297027

def S (n : ℕ) : ℤ :=
  if even n then -(n / 2) else (n + 1) / 2

theorem S_sum_eq_one_iff_odd (a b : ℕ) : 
  S a + S b + S (a + b) = 1 ↔ (odd a ∧ odd b) :=
by
  sorry

end S_sum_eq_one_iff_odd_l297_297027


namespace xy_value_l297_297868

theorem xy_value (x y : ℝ) (h1 : 2^x = 16^(y + 1)) (h2 : 27^y = 3^(x - 2)) : x * y = 8 :=
by
  sorry

end xy_value_l297_297868


namespace time_to_cross_platform_is_correct_l297_297611

noncomputable def speed_of_train := 36 -- speed in km/h
noncomputable def time_to_cross_pole := 12 -- time in seconds
noncomputable def time_to_cross_platform := 49.996960243180546 -- time in seconds

theorem time_to_cross_platform_is_correct : time_to_cross_platform = 49.996960243180546 := by
  sorry

end time_to_cross_platform_is_correct_l297_297611


namespace imaginary_part_magnitude_l297_297253

noncomputable def z : ℂ := 2 / (1 + I)

theorem imaginary_part_magnitude : Complex.abs (Complex.im z) = 1 :=
by
  sorry

end imaginary_part_magnitude_l297_297253


namespace overall_average_score_l297_297511

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / (scores.length)

theorem overall_average_score :
  let male_scores_avg := 82
  let female_scores_avg := 92
  let num_male_students := 8
  let num_female_students := 32
  let total_students := num_male_students + num_female_students
  let combined_scores_total := num_male_students * male_scores_avg + num_female_students * female_scores_avg
  average_score ([combined_scores_total]) / total_students = 90 :=
by 
  sorry

end overall_average_score_l297_297511


namespace find_ellipse_and_fixed_point_l297_297722

noncomputable def ellipse_centered_at_origin : Prop :=
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧ m * (0: ℝ) ^ 2 + n * (-2: ℝ) ^ 2 = 1 ∧ 
    m * (3/2)^2 + n * (-1)^2 = 1)

theorem find_ellipse_and_fixed_point :
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧ m * (0: ℝ) ^ 2 + n * (-2: ℝ) ^ 2 = 1 ∧ 
    m * (3/2)^2 + n * (-1)^2 = 1 ∧
    ∃ E : ℝ → ℝ → Prop, 
      (∀ x y : ℝ, E x y ↔ m * x^2 + n * y^2 = 1) ∧ 
      E x y ↔ (x / sqrt 3) ^ 2 + (y / 2) ^ 2 = 1) ∧
  (∀ P M N T H : ℝ × ℝ,
    P = (1, -2) →
    (∃ k : ℝ, k ≠ 0 ∧ 
      straight_line k P M E ∧ straight_line k P N E ∧ parallel_to_x_axis M T P P E ∧
      on_segment A B T ∧ vector_eq M T T H →
    HN_line_passing_through_fixed_point H N (0, -2))) :=
begin
  sorry,
end

-- Utility conditions and helpers to define intersection, line equations, etc.
def straight_line (k : ℝ) (P M : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, P.1 * k + P.2 = y ∧ E x y

def parallel_to_x_axis (M T : ℝ × ℝ) (P : ℝ × ℝ) (E T : ℝ → ℝ → Prop) : Prop :=
  M.2 = T.2 ∧ ∃ x : ℝ, M.1 = x ∧ E x T.2

def on_segment (A B T: ℝ × ℝ) : Prop :=
  (T.1 - A.1) * (B.2 - A.2) = (T.2 - A.2) * (B.1 - A.1)

def vector_eq (M T H : ℝ × ℝ) : Prop :=
  T.1 - M.1 = H.1 - T.1 ∧ T.2 - M.2 = H.2 - T.2

def HN_line_passing_through_fixed_point (H N fixed: ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, H.2 = m * H.1 + b ∧ N.2 = m * N.1 + b ∧ fixed.2 = m * fixed.1 + b

end find_ellipse_and_fixed_point_l297_297722


namespace pyramid_surface_area_l297_297292

/-- Given:
  - A triangular pyramid A-BCD where plane ABD is perpendicular to plane BCD
  - BC is perpendicular to BD
  - AB = AD = BD = 4 * sqrt 3
  - BC = 6
Prove: the surface area of the circumscribed sphere of the triangular pyramid A-BCD is 100 * pi. -/
def surface_area_circumscribed_sphere (A B C D : Point) (AB AD BD : Real) (BC : Real) : Prop :=
  ∃ R, R = 5 ∧ 4 * pi * R^2 = 100 * pi

theorem pyramid_surface_area 
  {A B C D : Point}
  (h1 : plane A B D ⊥ plane B C D)
  (h2 : BC ⊥ BD)
  (h3 : dist A B = 4 * real.sqrt 3)
  (h4 : dist A D = 4 * real.sqrt 3)
  (h5 : dist B D = 4 * real.sqrt 3)
  (h6 : dist B C = 6) :
  surface_area_circumscribed_sphere A B C D (4 * sqrt 3) 6 :=
  sorry

end pyramid_surface_area_l297_297292


namespace product_inequality_l297_297487

theorem product_inequality (n : ℕ) : (∏ i in finset.range n, (2 * i + 1) / (2 * i + 2)) < 1 / real.sqrt (2 * n + 1) :=
by
sory

end product_inequality_l297_297487


namespace find_d_l297_297908

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def parameterized_point (v d : Vector2D) (t : ℝ) : Vector2D :=
  { x := v.x + t * d.x, y := v.y + t * d.y }

def distance (a b : Vector2D) : ℝ :=
  real.sqrt ((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2)

noncomputable def d : Vector2D :=
  { x := 5 / real.sqrt 41, y := 4 / real.sqrt 41 }

theorem find_d (v d : Vector2D) (t : ℝ)
  (h1 : ∀ x y, y = (4 * x - 6) / 5 → ∃ t, (parameterized_point v d t).x = x ∧ (parameterized_point v d t).y = y)
  (h2 : (parameterized_point v d 0) = v)
  (h3 : ∀ x (hx : x ≥ 4), distance (parameterized_point v d t) {x := 4, y := 2} = t) :
  d = { x := 5 / real.sqrt 41, y := 4 / real.sqrt 41 } :=
sorry

end find_d_l297_297908


namespace integer_roots_l297_297678

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := x^3 - 4 * x^2 - 7 * x + 10

-- Define the proof problem statement
theorem integer_roots :
  {x : ℤ | polynomial x = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_l297_297678


namespace sine_fifth_power_constants_l297_297536
  
theorem sine_fifth_power_constants (b₁ b₂ b₃ b₄ b₅ : ℝ) (h : ∀ θ : ℝ, sin θ ^ 5 = b₁ * sin θ + b₂ * sin (2 * θ) + b₃ * sin (3 * θ) + b₄ * sin (4 * θ) + b₅ * sin (5 * θ)) :
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 128 :=
by
  sorry

end sine_fifth_power_constants_l297_297536


namespace solve_quadratic_eq_l297_297499

theorem solve_quadratic_eq :
  (∃ x₁ x₂ : ℝ, (x₁ = (1/3 + real.sqrt (10)/3)) ∧ (x₂ = (1/3 - real.sqrt (10)/3)) ∧ (∀ x : ℝ, x^2 - (2/3) * x - 1 = 0 ↔ x = x₁ ∨ x = x₂)) :=
sorry

end solve_quadratic_eq_l297_297499


namespace minimum_value_of_10ab_l297_297272

-- Definitions required for rounding functions
def round_to_nearest_integer (x : ℝ) : ℤ := 
  if x - Real.floor x < 0.5 then Real.floor x else Real.ceil x

def round_to_nearest_tenth (x : ℝ) : ℝ := 
  Real.floor (10 * x + 0.5) / 10

-- Problem conditions as Lean definitions
def condition1 (a b : ℝ) : Prop := 
  round_to_nearest_tenth a + round_to_nearest_integer b = 98.6

def condition2 (a b : ℝ) : Prop := 
  round_to_nearest_integer a + round_to_nearest_tenth b = 99.3

-- Lean theorem statement
theorem minimum_value_of_10ab (a b : ℝ) 
  (h1 : condition1 a b) (h2 : condition2 a b) : 
  round_to_nearest_integer (10 * (a + b)) = 988 :=
sorry

end minimum_value_of_10ab_l297_297272


namespace valid_six_digit_numbers_l297_297909

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def digit_sum (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def is_divisible_by_9 (n : Nat) : Prop :=
  digit_sum n % 9 = 0

def is_valid_six_digit_number (n : Nat) : Prop :=
  ∃ (a b : Nat), n = b * 100000 + 20140 + a ∧ is_divisible_by_4 (10 * 2014 + a) ∧ is_divisible_by_9 (b * 100000 + 20140 + a)

theorem valid_six_digit_numbers :
  { n | is_valid_six_digit_number n } = {220140, 720144, 320148} :=
by
  sorry

end valid_six_digit_numbers_l297_297909


namespace find_S8_l297_297408

variable {a1 d : ℕ}

-- Given conditions
def S (n : ℕ) := n * a1 + ((n * (n - 1)) / 2) * d
def condition1 := S 2 = 9
def condition2 := S 4 = 22

-- Statement to prove
theorem find_S8 (h1 : condition1) (h2 : condition2) : S 8 = 60 :=
sorry

end find_S8_l297_297408


namespace find_b_l297_297176

theorem find_b {a b : ℝ} (h₁ : 2 * 2 + b = 1 - 2 * a) (h₂ : -2 * 2 + b = -15 + 2 * a) : 
  b = -7 := sorry

end find_b_l297_297176


namespace not_perpendicular_plane_PDF_ABc_l297_297412

-- Define the regular tetrahedron with vertices P, A, B, C.
variables {P A B C D E F : Type}

-- Assume conditions
-- 1. Regular tetrahedron P-ABC 
-- 2. D, E, and F are midpoints of AB, BC, and CA respectively.
axioms
  (h1 : regular_tetrahedron P A B C)
  (h2 : midpoint D A B)
  (h3 : midpoint E B C)
  (h4 : midpoint F C A)
  
-- Define planes PDF, PAE, and ABC
def plane_PDF : Type := plane P D F
def plane_PAE : Type := plane P A E
def plane_ABC : Type := plane A B C
  
-- Theorem to prove
theorem not_perpendicular_plane_PDF_ABc : ¬(perpendicular plane_PDF plane_ABC) :=
  sorry

end not_perpendicular_plane_PDF_ABc_l297_297412


namespace complex_number_pure_imaginary_l297_297384

theorem complex_number_pure_imaginary (a : ℝ) 
  (h1 : ∃ a : ℝ, (a^2 - 2*a - 3 = 0) ∧ (a + 1 ≠ 0)) 
  : a = 3 := sorry

end complex_number_pure_imaginary_l297_297384


namespace carl_candy_bars_l297_297245

def weekly_earnings := 0.75
def weeks := 4
def candy_bar_cost := 0.50

theorem carl_candy_bars :
  (weeks * weekly_earnings) / candy_bar_cost = 6 := by
  sorry

end carl_candy_bars_l297_297245


namespace oddly_powerful_count_l297_297635

/-- An integer n is oddly powerful if there exist positive integers a and b,
where b > 1, b is odd, and a^b = n. The number of oddly powerful integers
less than 2050 is 18. -/
theorem oddly_powerful_count : Finset.card 
  (Finset.filter (λ n : ℕ,  ∃ (a b : ℕ), b > 1 ∧ Odd b ∧ a^b = n) (Finset.range 2050)) = 18 :=
sorry

end oddly_powerful_count_l297_297635


namespace evaluate_expression_l297_297756

theorem evaluate_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) :
  5 * x + 2 * y * 3 = 38 :=
by
  sorry

end evaluate_expression_l297_297756


namespace cannot_fit_rectangle_l297_297151

theorem cannot_fit_rectangle 
  (w1 h1 : ℕ) (w2 h2 : ℕ) 
  (h1_pos : 0 < h1) (w1_pos : 0 < w1)
  (h2_pos : 0 < h2) (w2_pos : 0 < w2) :
  w1 = 5 → h1 = 6 → w2 = 3 → h2 = 8 →
  ¬(w2 ≤ w1 ∧ h2 ≤ h1) :=
by
  intros H1 W1 H2 W2
  sorry

end cannot_fit_rectangle_l297_297151


namespace largest_cut_valid_l297_297999

-- Define the lengths of the sticks
def length1 : ℕ := 13
def length2 : ℕ := 20
def length3 : ℕ := 21

-- Definition of the condition where the remaining lengths form a degenerate triangle
def forms_degenerate_triangle (x : ℕ) : Prop :=
  (length1 - x + length2 - x = length3 - x ∨
   length1 - x + length3 - x = length2 - x ∨
   length2 - x + length3 - x = length1 - x)

-- Define what it means for the cuts to be valid
def valid_cut (x : ℕ) : Prop :=
  length1 > x ∧ length2 > x ∧ length3 > x

-- The largest piece that can be cut
theorem largest_cut_valid (x : ℕ) :
  forms_degenerate_triangle x ∧ valid_cut x ↔ x = 12 :=
begin
  sorry,
end

end largest_cut_valid_l297_297999


namespace taxi_ride_cost_l297_297194

-- Define the base fare
def base_fare : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the distance traveled
def distance : ℝ := 8.00

-- Define the total cost function
def total_cost (base : ℝ) (per_mile : ℝ) (miles : ℝ) : ℝ :=
  base + (per_mile * miles)

-- The statement to prove: the total cost of an 8-mile taxi ride
theorem taxi_ride_cost : total_cost base_fare cost_per_mile distance = 4.40 :=
by
sorry

end taxi_ride_cost_l297_297194


namespace sum_of_squares_remainder_1_15_l297_297132

theorem sum_of_squares_remainder_1_15 :
  let sum_squares := ∑ k in finset.range 16, k^2 in 
  (sum_squares + 20) % 13 = 1 :=
by {
  let sum_squares := ∑ k in finset.range 16, k^2,
  have h : (sum_squares + 20) % 13 = 1, 
  { sorry },
  exact h,
}

end sum_of_squares_remainder_1_15_l297_297132


namespace max_area_perimeter_ratio_l297_297208

theorem max_area_perimeter_ratio (A B C : Type) [triangle ABC] :
  ∀ (DEF : Type) [nested_triangle DEF ABC], 
  (ratio_area_perimeter DEF ≤ ratio_area_perimeter ABC) := 
sorry

end max_area_perimeter_ratio_l297_297208


namespace alpha_plus_beta_eq_pi_over_two_l297_297712

theorem alpha_plus_beta_eq_pi_over_two (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h3 : (sin α)^4 / (cos β)^2 + (cos α)^4 / (sin β)^2 = 1) : 
    α + β = π / 2 :=
by
  -- Proof will go here
  sorry

end alpha_plus_beta_eq_pi_over_two_l297_297712


namespace max_value_M_l297_297823

def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 17}

def f : ℕ → ℕ := sorry -- f to be defined as in the problem conditions

def iter_f (k : ℕ) (x : ℕ) : ℕ :=
  Nat.recOn k (f x) (λ n fn, f fn)

def condition1 (m M : ℕ) (h : m < M) (i : ℕ) : Prop :=
  (i < 17) → (iter_f m (i + 1) - iter_f m i) % 17 ≠ 1 ∧
  (iter_f m (i + 1) - iter_f m i) % 17 ≠ 16 ∧
  (iter_f m 1 - iter_f m 17) % 17 ≠ 1 ∧
  (iter_f m 1 - iter_f m 17) % 17 ≠ 16

def condition2 (M : ℕ) (i : ℕ) : Prop :=
  (i < 17) → ((iter_f M (i + 1) - iter_f M i) % 17 = 1 ∨ 
              (iter_f M (i + 1) - iter_f M i) % 17 = 16) ∧
  ((iter_f M 1 - iter_f M 17) % 17 = 1 ∨ 
   (iter_f M 1 - iter_f M 17) % 17 = 16)

theorem max_value_M : ∃ M, (∀ m, m < M → ∀ i, condition1 m M m) ∧ (∀ i, condition2 M i) ∧ ∀ m', 
  ((∀ m, m < m' → ∀ i, condition1 m m' m) ∧ (∀ i, condition2 m' i)) → m' ≤ M :=
by
  use 8
  sorry -- the remainder of the proof

end max_value_M_l297_297823


namespace same_solution_implies_value_of_m_l297_297390

theorem same_solution_implies_value_of_m (x m : ℤ) (h₁ : -5 * x - 6 = 3 * x + 10) (h₂ : -2 * m - 3 * x = 10) : m = -2 :=
by
  sorry

end same_solution_implies_value_of_m_l297_297390


namespace volume_of_box_lt_1200_l297_297954

theorem volume_of_box_lt_1200 :
  {x : ℕ // 0 < x ∧ (x + 3) * (x - 3) * (x^2 + x + 1) < 1200}.card = 6 :=
by {
  sorry
}

end volume_of_box_lt_1200_l297_297954


namespace sin_x_value_l297_297307

theorem sin_x_value (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : Real.sin x = 8 / 17 :=
sorry

end sin_x_value_l297_297307


namespace coeff_a9_l297_297280

theorem coeff_a9 :
  (∃ (a : ℕ → ℚ), (1 + x)^10 = ∑ i in finset.range 11, a i * (1 - x)^i) →
  (∃ a_9 : ℚ, a_9 = -20) :=
by
  intro h
  use -20
  sorry

end coeff_a9_l297_297280


namespace toys_produced_each_day_l297_297144

def toys_produced_per_week : ℕ := 6000
def work_days_per_week : ℕ := 4

theorem toys_produced_each_day :
  (toys_produced_per_week / work_days_per_week) = 1500 := 
by
  -- The details of the proof are omitted
  -- The correct answer given the conditions is 1500 toys
  sorry

end toys_produced_each_day_l297_297144


namespace number_of_games_in_season_l297_297997

-- Define the number of teams and divisions
def num_teams := 20
def num_divisions := 4
def teams_per_division := 5

-- Define the games played within and between divisions
def intra_division_games_per_team := 12  -- 4 teams * 3 games each
def inter_division_games_per_team := 15  -- (20 - 5) teams * 1 game each

-- Define the total number of games played by each team
def total_games_per_team := intra_division_games_per_team + inter_division_games_per_team

-- Define the total number of games played (double-counting needs to be halved)
def total_games (num_teams : ℕ) (total_games_per_team : ℕ) : ℕ :=
  (num_teams * total_games_per_team) / 2

-- The theorem to be proven
theorem number_of_games_in_season :
  total_games num_teams total_games_per_team = 270 :=
by
  sorry

end number_of_games_in_season_l297_297997


namespace sum_first_9_terms_arith_seq_l297_297836

theorem sum_first_9_terms_arith_seq:
  ∃ a : ℕ → ℚ,
    a 1 = 9 ∧ (∀ n, a (n + 1) - a n = d ∧ d ∈ ℤ) ∧ 
    (∀ n, ∑ k in range n, a k ≤ ∑ k in range 5, a k) → 
    ∑ n in range 9, (1 / (a n * a (n + 1))) = -1 / 9 :=
begin
  sorry
end

end sum_first_9_terms_arith_seq_l297_297836


namespace local_minimum_at_2_l297_297467

-- Given a function f with the specified definition
def f (x : ℝ) : ℝ := (2 / x) + Real.log x

-- Statement of the proof problem
theorem local_minimum_at_2 : (∃ ε > 0, ∀ x, abs (x - 2) < ε → f x > f 2) :=
sorry

end local_minimum_at_2_l297_297467


namespace find_a_l297_297086

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (ax^2 - ax + 1)'.eval 0 = -a)
  (h3 : ∀ k : ℝ, k = -2) :
  a = -1 / 2 :=
by
  sorry

end find_a_l297_297086


namespace value_of_q_l297_297752

theorem value_of_q (p q : ℝ) (h : (2 - Complex.i) * (2 + Complex.i) = q) : q = 5 :=
by
  sorry

end value_of_q_l297_297752


namespace minimum_possible_value_of_d_l297_297186

theorem minimum_possible_value_of_d :
  (∃ d : ℝ, (3 * sqrt 5, d + 3).dist (0, 0) = 3 * d) → ∃ d : ℝ, (d ≥ 0) ∧ (3 * sqrt 5, d + 3).dist (0, 0) = 3 * d ∧ d = 3 :=
by
  sorry

end minimum_possible_value_of_d_l297_297186


namespace bride_groom_couples_sum_l297_297984

def wedding_reception (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) : Prop :=
  total_guests - friends = couples_guests

theorem bride_groom_couples_sum (B G : ℕ) (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) 
  (h1 : total_guests = 180) (h2 : friends = 100) (h3 : wedding_reception total_guests friends couples_guests) 
  (h4 : couples_guests = 80) : B + G = 40 := 
  by
  sorry

end bride_groom_couples_sum_l297_297984


namespace find_x_l297_297324

theorem find_x (x : ℝ) (h : real.sqrt (real.sqrt x) = 3) : x = 81 := 
  sorry

end find_x_l297_297324


namespace part1_part2_l297_297051

-- Define the conditions for part (1)
def nonEmptyBoxes := ∀ i j k: Nat, (i ≠ j ∧ i ≠ k ∧ j ≠ k)
def ball3inBoxB := ∀ (b3: Nat) (B: Nat), b3 = 3 ∧ B > 0

-- Define the conditions for part (2)
def ball1notInBoxA := ∀ (b1: Nat) (A: Nat), b1 ≠ 1 ∧ A > 0
def ball2notInBoxB := ∀ (b2: Nat) (B: Nat), b2 ≠ 2 ∧ B > 0

-- Theorems to be proved
theorem part1 (h1: nonEmptyBoxes) (h2: ball3inBoxB) : ∃ n, n = 12 := by sorry

theorem part2 (h3: ball1notInBoxA) (h4: ball2notInBoxB) : ∃ n, n = 36 := by sorry

end part1_part2_l297_297051


namespace hyperbola_eccentricity_l297_297698

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hyperbola_eq : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (asymptote : ∀ x, asymptote := (y = √2 * x)) :
  e = √3 := by
  sorry

end hyperbola_eccentricity_l297_297698


namespace find_c_plus_d_l297_297405

noncomputable def square_WXYZ_side_length : ℝ := 10

noncomputable def angle_O1QO2 : ℝ := 150

axiom WQ_greater_than_QZ (W Q Z : ℝ) : W > Q ∧ Q > Z

theorem find_c_plus_d (c d : ℕ) (WQ : ℝ) : 
  WQ = (sqrt (50) + sqrt (100)) → c + d = 150 :=
sorry

end find_c_plus_d_l297_297405


namespace bad_carrots_l297_297576

theorem bad_carrots (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) (total_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : good_carrots = 38)
  (h4 : total_carrots = carol_carrots + mom_carrots)
  (h5 : bad_carrots = total_carrots - good_carrots) :
  bad_carrots = 7 := by
  sorry

end bad_carrots_l297_297576


namespace factorization_correct_l297_297675

theorem factorization_correct (x : ℝ) : 
  (x^2 + 5 * x + 2) * (x^2 + 5 * x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5 * x - 1) :=
by
  sorry

end factorization_correct_l297_297675


namespace num_odd_factors_of_252_l297_297355

theorem num_odd_factors_of_252 : 
  ∃ n : ℕ, n = 252 ∧ 
  ∃ k : ℕ, (k = ∏ d in (divisors_filter (λ x, x % 2 = 1) n), 1) 
  ∧ k = 6 := 
sorry

end num_odd_factors_of_252_l297_297355


namespace race_distance_l297_297396

theorem race_distance (a b c : ℝ) (d : ℝ) 
  (h1 : d / a = (d - 15) / b)
  (h2 : d / b = (d - 30) / c)
  (h3 : d / a = (d - 40) / c) : 
  d = 90 :=
by sorry

end race_distance_l297_297396


namespace negative_integer_solution_l297_297101

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : N^2 + N = 6) : N = -3 := 
by 
  sorry

end negative_integer_solution_l297_297101


namespace integer_S_for_all_n_l297_297464

noncomputable def S (a : ℝ) (n : ℤ) : ℝ := a^n + a^(-n)

theorem integer_S_for_all_n
  (a : ℝ) (h₁ : a ≠ 0)
  (p : ℕ)
  (h₂ : (S a p) ∈ ℤ)
  (h₃ : (S a (p + 1)) ∈ ℤ) :
  ∀ (n : ℤ), (S a n) ∈ ℤ :=
by
  sorry

end integer_S_for_all_n_l297_297464


namespace solve_inequality_l297_297714

theorem solve_inequality (a : ℝ) : 
  ((a-1) * x^2 + (2 * a + 3) * x + (a + 2) < 0) ↔ 
  (if a < -17/8 then true
   else if a = -17/8 then x ≠ -1/5
   else if -17/8 < a ∧ a < 1 then 
     x > (- (2 * a + 3) - sqrt (8 * a + 17)) / (2 * (a - 1)) ∨ 
     x < (- (2 * a + 3) + sqrt (8 * a + 17)) / (2 * (a - 1))
   else if a = 1 then x < -3/5
   else (a > 1 ∧ ( (- (2 * a + 3) - sqrt (8 * a + 17)) / (2 * (a - 1)) < x) ∧
                        (x < (- (2 * a + 3) + sqrt (8 * a + 17)) / (2 * (a - 1)) ))) := 
sorry

end solve_inequality_l297_297714


namespace expected_defective_chips_in_60000_l297_297149

def shipmentS1 := (2, 5000)
def shipmentS2 := (4, 12000)
def shipmentS3 := (2, 15000)
def shipmentS4 := (4, 16000)

def total_defective_chips := shipmentS1.1 + shipmentS2.1 + shipmentS3.1 + shipmentS4.1
def total_chips := shipmentS1.2 + shipmentS2.2 + shipmentS3.2 + shipmentS4.2

def defective_ratio := total_defective_chips / total_chips
def shipment60000 := 60000

def expected_defectives (ratio : ℝ) (total_chips : ℝ) := ratio * total_chips

theorem expected_defective_chips_in_60000 :
  expected_defectives defective_ratio shipment60000 = 15 :=
by
  sorry

end expected_defective_chips_in_60000_l297_297149


namespace smallest_positive_period_find_x_l297_297733

def f (x : ℝ) : ℝ := (Real.sin (2 * x))^2 + Real.sqrt 3 * Real.sin (2 * x) * Real.cos (2 * x)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 2 :=
begin
  sorry
end

theorem find_x (x : ℝ) (hx : x ∈ set.Icc (π / 8) (π / 4)) (h : f x = 1) :
  x = π / 4 :=
begin
  sorry
end

end smallest_positive_period_find_x_l297_297733


namespace identify_shifted_graph_l297_297735

def g (x : ℝ) : ℝ :=
  if (-2 ≤ x ∧ x ≤ 1) then -x
  else if (1 ≤ x ∧ x ≤ 4) then real.sqrt (9 - (x - 3)^2)
  else if (4 ≤ x ∧ x ≤ 6) then x - 4
  else 0

def g_shifted (x : ℝ) : ℝ :=
  g (x + 2)

def is_correct_graph (f : ℝ → ℝ) (graph_label : String) : Prop :=
  graph_label = "A"

theorem identify_shifted_graph : is_correct_graph g_shifted "A" :=
sorry

end identify_shifted_graph_l297_297735


namespace car_distance_in_45_minutes_l297_297748

theorem car_distance_in_45_minutes
  (train_speed : ℝ)
  (car_speed_ratio : ℝ)
  (time_minutes : ℝ)
  (h_train_speed : train_speed = 90)
  (h_car_speed_ratio : car_speed_ratio = 5 / 6)
  (h_time_minutes : time_minutes = 45) :
  ∃ d : ℝ, d = 56.25 ∧ d = (car_speed_ratio * train_speed) * (time_minutes / 60) :=
by
  sorry

end car_distance_in_45_minutes_l297_297748


namespace lucas_series_sum_l297_297079

open BigOperators

def lucas : ℕ → ℚ
| 0       := 2
| 1       := 1
| (n + 2) := lucas (n + 1) + lucas n

def r := ∑' n : ℕ, lucas n / (10 : ℚ)^(n + 1)

theorem lucas_series_sum :
  r = 19 / 89 := sorry

end lucas_series_sum_l297_297079


namespace color_preserving_permutation_exists_l297_297513

namespace TokenBoard

def color3 (x y : ℕ) : ℕ := (x + y) % 3

def tokenReplacement (color : ℕ) : ℕ :=
  if color = 0 then 1 else if color = 1 then 2 else 0

theorem color_preserving_permutation_exists 
  (n d : ℕ)
  (initial_position : ℕ × ℕ → ℕ) -- initial color of each token
  (token_move_limit : ℕ × ℕ → ℕ × ℕ → Prop)
  (Hmove : ∀ (x y : ℕ × ℕ), token_move_limit x y → dist x.snd y.snd ≤ d)
  (Hreplacement : ∀ (x y : ℕ × ℕ), token_move_limit x y → initial_position x = tokenReplacement (initial_position y))
  : ∃ (final_position : ℕ × ℕ → ℕ),
    (∀ (x : ℕ × ℕ), dist x (final_position x) ≤ d + 2) ∧ 
    (∀ (x : ℕ × ℕ), initial_position x = color3 x.fst x.snd := final_position x) :=
  by
  sorry

end TokenBoard

end color_preserving_permutation_exists_l297_297513


namespace prove_hyperbola_l297_297056

noncomputable def verification (t : ℝ) : Prop :=
  let x := Real.cosh t
  let y := Real.sinh t
  x^2 - y^2 = 1

theorem prove_hyperbola (t : ℝ) : verification t :=
by {
  let x := Real.cosh t,
  let y := Real.sinh t,
  have h1 : x = Real.cosh t := rfl,
  have h2 : y = Real.sinh t := rfl,
  rw [← h1, ← h2],
  exact sorry
}

end prove_hyperbola_l297_297056


namespace midpoint_coordinates_l297_297250

-- Define the coordinates of the points
def P1 : ℝ × ℝ := (-3, 7)
def P2 : ℝ × ℝ := (5, -1)

-- State that the midpoint of P1 and P2 is (1, 3)
theorem midpoint_coordinates : 
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2) in
  midpoint = (1, 3) :=
by
  sorry

end midpoint_coordinates_l297_297250


namespace probability_prime_l297_297882

open finset

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

def possible_outcomes : finset ℕ := finset.range 7

def prime_outcomes : finset ℕ := {2, 3, 5}

theorem probability_prime :
  (prime_outcomes.card.to_real / possible_outcomes.card.to_real) = 1 / 2 :=
by
  sorry

end probability_prime_l297_297882


namespace integral_cos_3x_l297_297634

theorem integral_cos_3x :
  ∫ x in (real.pi / 6)..(real.pi / 2), real.cos (3 * x) = -1 / 3 := by
  sorry

end integral_cos_3x_l297_297634


namespace cos_alpha_minus_beta_eq_zero_l297_297279

theorem cos_alpha_minus_beta_eq_zero 
  (α β : ℝ) 
  (h1 : sin α + sqrt 3 * sin β = 1) 
  (h2 : cos α + sqrt 3 * cos β = sqrt 3) : 
  cos (α - β) = 0 := 
sorry

end cos_alpha_minus_beta_eq_zero_l297_297279


namespace perfect_numbers_sum_of_palindromes_l297_297169

def is_palindromic (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = List.reverse digits

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_number (n : ℕ) : Prop :=
  let a := (n / 1000) % 10
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + b = c + d

def count_perfect_sum_of_palindromics : ℕ :=
  (List.range 10000).count (λ n =>
    is_four_digit n ∧ 
    is_perfect_number n ∧ 
    ∃ a b, is_four_digit a ∧ is_four_digit b ∧ is_palindromic a ∧ is_palindromic b ∧ n = a + b)

theorem perfect_numbers_sum_of_palindromes :
  count_perfect_sum_of_palindromics = 80 :=
sorry

end perfect_numbers_sum_of_palindromes_l297_297169


namespace sum_of_two_numbers_l297_297150

-- Define the two numbers and conditions
variables {x y : ℝ}
axiom prod_eq : x * y = 120
axiom sum_squares_eq : x^2 + y^2 = 289

-- The statement we want to prove
theorem sum_of_two_numbers (x y : ℝ) (prod_eq : x * y = 120) (sum_squares_eq : x^2 + y^2 = 289) : x + y = 23 :=
sorry

end sum_of_two_numbers_l297_297150


namespace final_lives_equals_20_l297_297565

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end final_lives_equals_20_l297_297565


namespace least_number_subtracted_l297_297946

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_pos : 0 < x) (h_init : n = 427398) (h_div : ∃ k : ℕ, (n - x) = 14 * k) : x = 6 :=
sorry

end least_number_subtracted_l297_297946


namespace Carl_can_buy_six_candy_bars_l297_297241

-- Define the earnings per week
def earnings_per_week : ℝ := 0.75

-- Define the number of weeks Carl works
def weeks : ℕ := 4

-- Define the cost of one candy bar
def cost_per_candy_bar : ℝ := 0.50

-- Calculate total earnings
def total_earnings := earnings_per_week * weeks

-- Calculate the number of candy bars Carl can buy
def number_of_candy_bars := total_earnings / cost_per_candy_bar

-- State the theorem that Carl can buy exactly 6 candy bars
theorem Carl_can_buy_six_candy_bars : number_of_candy_bars = 6 := by
  sorry

end Carl_can_buy_six_candy_bars_l297_297241


namespace human_height_weight_correlated_l297_297952

-- Define the relationships as types
def taxiFareDistanceRelated : Prop := ∀ x y : ℕ, x = y → True
def houseSizePriceRelated : Prop := ∀ x y : ℕ, x = y → True
def humanHeightWeightCorrelated : Prop := ∃ k : ℕ, ∀ x y : ℕ, x / k = y
def ironBlockMassRelated : Prop := ∀ x y : ℕ, x = y → True

-- Main theorem statement
theorem human_height_weight_correlated : humanHeightWeightCorrelated :=
  sorry

end human_height_weight_correlated_l297_297952


namespace smallest_product_in_set_is_negative32_l297_297685

theorem smallest_product_in_set_is_negative32 :
  let S := {-8, -6, -2, 0, 4}
  in ∃ x y ∈ S, x ≠ y ∧ (x * y = -32) ∧ ∀ u v ∈ S, u ≠ v → u * v ≥ -32 :=
by
  let S := {-8, -6, -2, 0, 4}
  use -8, 4
  use -32
  sorry

end smallest_product_in_set_is_negative32_l297_297685


namespace probability_of_toss_has_2_l297_297168

noncomputable def fair_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}

def is_fair_die (X : ℕ →  Prop) : Prop :=
  ∀ x, x ∈ fair_six_sided_die → (X x) = 1 / 6

def toss_3 (X1 X2 X3 : ℕ) : Prop :=
  X3 = X1 + X2 ∧ X1 ∈ fair_six_sided_die ∧ X2 ∈ fair_six_sided_die ∧ X3 ∈ fair_six_sided_die

def has_2 (X1 X2 X3 : ℕ) : Prop :=
  X1 = 2 ∨ X2 = 2 ∨ X3 = 2

theorem probability_of_toss_has_2 :
  ∀ (X1 X2 X3 : ℕ),
    toss_3 X1 X2 X3 → is_fair_die X1 ∧ is_fair_die X2 ∧ is_fair_die X3 →
    (has_2 X1 X2 X3) = 7 / 15 := 
by
  sorry

end probability_of_toss_has_2_l297_297168


namespace train_pass_bridge_time_l297_297612

noncomputable def trainLength : ℝ := 360
noncomputable def trainSpeedKMH : ℝ := 45
noncomputable def bridgeLength : ℝ := 160
noncomputable def totalDistance : ℝ := trainLength + bridgeLength
noncomputable def trainSpeedMS : ℝ := trainSpeedKMH * (1000 / 3600)
noncomputable def timeToPassBridge : ℝ := totalDistance / trainSpeedMS

theorem train_pass_bridge_time : timeToPassBridge = 41.6 := sorry

end train_pass_bridge_time_l297_297612


namespace simplified_expression_evaluate_at_zero_l297_297061

noncomputable def simplify_expr (x : ℝ) : ℝ :=
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2 * x + 1))

theorem simplified_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  simplify_expr x = 1 / (x - 1) :=
by sorry

theorem evaluate_at_zero (h₁ : (0 : ℝ) ≠ -1) (h₂ : (0 : ℝ) ≠ 1) : 
  simplify_expr 0 = -1 :=
by sorry

end simplified_expression_evaluate_at_zero_l297_297061


namespace kyle_total_revenue_l297_297438

-- Define the conditions
def initial_cookies := 60
def initial_brownies := 32
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2
def price_per_cookie := 1
def price_per_brownie := 1.5

-- Statement of the proof
theorem kyle_total_revenue :
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies),
      remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies),
      revenue_from_cookies := remaining_cookies * price_per_cookie,
      revenue_from_brownies := remaining_brownies * price_per_brownie,
      total_revenue := revenue_from_cookies + revenue_from_brownies
  in total_revenue = 99 :=
by
  sorry

end kyle_total_revenue_l297_297438


namespace Vasya_is_right_l297_297949

-- Defining the initial conditions
def initial_bullets : ℕ := 5
def bullets_per_hit : ℕ := 5
def total_shots_claimed : ℕ := 50
def hits_claimed : ℕ := 8

-- Stating the problem in Lean to prove Vasya is right
theorem Vasya_is_right : ¬(total_shots_claimed = initial_bullets + bullets_per_hit * hits_claimed + hits_claimed) :=
by {
  have H1 : hits_claimed * bullets_per_hit = 8 * 5 := rfl,
  have H2 : initial_bullets + (hits_claimed * bullets_per_hit) = 5 + 40 := rfl,
  have H3 : 45 ≠ 50 - hits_claimed := rfl,
  have H4 : total_shots_claimed ≠ initial_bullets + bullets_per_hit * hits_claimed + hits_claimed := 
    (ne_of_lt (nat.lt_of_sub_lt_sub_left _ _)).trans H3,
  exact H4
}

end Vasya_is_right_l297_297949


namespace num_odd_factors_of_252_l297_297353

theorem num_odd_factors_of_252 : 
  ∃ n : ℕ, n = 252 ∧ 
  ∃ k : ℕ, (k = ∏ d in (divisors_filter (λ x, x % 2 = 1) n), 1) 
  ∧ k = 6 := 
sorry

end num_odd_factors_of_252_l297_297353


namespace sequence_converges_iff_l297_297445

noncomputable def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![a, 1 - a],
  ![1 - a, a]
]

def initial_vector : Fin 2 → ℝ
| ⟨0, _⟩ := 1
| ⟨1, _⟩ := 0

def recurrence (A : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i, Finset.univ.sum (λ j, A i j * v j)

def sequence (a : ℝ) : ℕ → (Fin 2 → ℝ)
| 0 := initial_vector
| (n + 1) := recurrence (A a) (sequence n)

def converges_to (seq : ℕ → (Fin 2 → ℝ)) : Prop :=
  ∃ x : Fin 2 → ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, ∀ i, abs (seq n i - x i) < ε

theorem sequence_converges_iff (a : ℝ) :
  converges_to (λ n, (sequence a n)) ↔ 0 < a ∧ a < 1 :=
sorry

end sequence_converges_iff_l297_297445


namespace total_time_marco_6_laps_total_time_in_minutes_and_seconds_l297_297036

noncomputable def marco_running_time : ℕ :=
  let distance_1 := 150
  let speed_1 := 5
  let time_1 := distance_1 / speed_1

  let distance_2 := 300
  let speed_2 := 4
  let time_2 := distance_2 / speed_2

  let time_per_lap := time_1 + time_2
  let total_laps := 6
  let total_time_seconds := time_per_lap * total_laps

  total_time_seconds

theorem total_time_marco_6_laps : marco_running_time = 630 := sorry

theorem total_time_in_minutes_and_seconds : 10 * 60 + 30 = 630 := sorry

end total_time_marco_6_laps_total_time_in_minutes_and_seconds_l297_297036


namespace dining_bill_proof_l297_297104

noncomputable def original_bill_amount (share_per_person : ℝ) (number_of_people : ℕ) (tip_rate : ℝ) : ℝ :=
  let total_amount_with_tip_per_person := share_per_person * number_of_people
  let total_amount_with_tip := total_amount_with_tip_per_person
  total_amount_with_tip / (1 + tip_rate)

theorem dining_bill_proof :
  original_bill_amount 34.66 7 0.15 ≈ 210.97 :=
begin
  sorry, -- proof goes here
end

end dining_bill_proof_l297_297104


namespace no_valid_six_digit_palindrome_years_l297_297599

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ is_palindrome n

noncomputable def is_four_digit_prime_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_palindrome n ∧ is_prime n

noncomputable def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_palindrome n ∧ is_prime n

theorem no_valid_six_digit_palindrome_years :
  ∀ N : ℕ, is_six_digit_palindrome N →
  ¬ ∃ (p q : ℕ), is_four_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ N = p * q := 
sorry

end no_valid_six_digit_palindrome_years_l297_297599


namespace cyclic_ngon_exists_l297_297052

theorem cyclic_ngon_exists 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) 
  (h_cond : ∀ i, 1 ≤ i ∧ i ≤ n → 2 * a i < (Finset.range n).sum a) :
  ∃ (P : Finset ℝ), P.card = n ∧ ∀ i, 1 ≤ i ∧ i ≤ n → ∃ j, j ∈ P ∧ a i = Dist (P (i - 1) % n) (P i % n) :=
sorry

end cyclic_ngon_exists_l297_297052


namespace non_adjacent_divisibility_l297_297875

theorem non_adjacent_divisibility (a : Fin 7 → ℕ) (h : ∀ i, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) :
  ∃ i j : Fin 7, i ≠ j ∧ (¬(i + 1)%7 = j) ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end non_adjacent_divisibility_l297_297875


namespace find_k_l297_297319

theorem find_k (k : ℝ) (h1 : 0 < k)
  (h2 : (∃ A B C D : ℝ×ℝ, 
          A = (4, 0) ∧ B = (0, 4) ∧ 
          (C.1, C.2) ∈ { p : ℝ×ℝ | p.2 = k / p.1 } ∧
          (D.1, D.2) ∈ { p : ℝ×ℝ | p.2 = k / p.1 } ∧
          A.dist B = sqrt 2 * C.dist D)) : 
  k = 2 := 
by
  sorry

end find_k_l297_297319


namespace determine_polynomial_l297_297665

open Polynomial

/--
A polynomial equality problem.
-/
theorem determine_polynomial (h : Polynomial ℝ) :
  (∀ x : ℝ, 7 * x ^ 4 - 4 * x ^ 3 + x + h.eval x = 5 * x ^ 3 - 7 * x + 6) →
  (∀ x : ℝ, h.eval x = -7 * x ^ 4 + 9 * x ^ 3 - 8 * x + 6) :=
begin
  intro H,
  funext x,
  specialize H x,
  rw Polynomial.eval_add at H,
  rw [Polynomial.eval_C, Polynomial.eval_mul, Polynomial.eval_X, Polynomial.eval_pow] at H,
  have lhs : 7 * x ^ 4 - 4 * x ^ 3 + x + h.eval x = (7 * x ^ 4 - 4 * x ^ 3 + x) + h.eval x,
  { ring, },
  rw lhs at H,
  sorry -- complete the proof here
end

end determine_polynomial_l297_297665


namespace total_sleep_correct_l297_297427

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l297_297427


namespace angle_CAD_eq_15_l297_297779

-- Define the angles and equality of sides
variables (A B C D : Type) [IsConvexQuadrilateral A B C D]
           (angle_A : ∠A = 65)
           (angle_B : ∠B = 80)
           (angle_C : ∠C = 75)
           (AB_eq_BD : AB = BD)
           
-- State the theorem to prove the measure of ∠CAD
theorem angle_CAD_eq_15 : ∠CAD = 15 :=
sorry

end angle_CAD_eq_15_l297_297779


namespace max_a_satisfies_no_lattice_points_l297_297254

-- Define the conditions
def no_lattice_points (m : ℚ) (x_upper : ℕ) :=
  ∀ x : ℕ, 0 < x ∧ x ≤ x_upper → ¬∃ y : ℤ, y = m * x + 3

-- Final statement we need to prove
theorem max_a_satisfies_no_lattice_points :
  ∃ a : ℚ, a = 51 / 151 ∧ ∀ m : ℚ, 1 / 3 < m → m < a → no_lattice_points m 150 :=
sorry

end max_a_satisfies_no_lattice_points_l297_297254


namespace probability_of_A_being_chosen_l297_297530

-- Definitions based on the conditions
def four_people := {'A', 'B', 'C', 'D'}
def choose_two_people (s : Set Char) : Set (Set Char) := { t | t ⊆ s ∧ t.size = 2 }

-- Lean statement
theorem probability_of_A_being_chosen :
  let s := four_people
  let n := (choose_two_people s).size
  let m := (choose_two_people {'B', 'C', 'D'}).size
  (m / n : ℚ) = 1 / 2 := by
  sorry

end probability_of_A_being_chosen_l297_297530


namespace probability_prime_rolled_l297_297884

open Finset

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def prime_outcomes : Finset ℕ := outcomes.filter is_prime

theorem probability_prime_rolled : (prime_outcomes.card : ℚ) / outcomes.card = 1 / 2 :=
by
  -- Proof would go here
  sorry

end probability_prime_rolled_l297_297884


namespace lowest_score_jack_l297_297893

noncomputable def lowest_possible_score (mean : ℝ) (std_dev : ℝ) := 
  max ((1.28 * std_dev) + mean) (mean + 2 * std_dev)

theorem lowest_score_jack (mean : ℝ := 60) (std_dev : ℝ := 10) :
  lowest_possible_score mean std_dev = 73 := 
by
  -- We need to show that the minimum score Jack could get is 73 based on problem conditions
  sorry

end lowest_score_jack_l297_297893


namespace smallest_m_plus_n_l297_297087

theorem smallest_m_plus_n
  (m n : ℕ)
  (h1 : m > 1)
  (h2 : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1 ↔ (1/(n*m) ≤ x ∧ x ≤ m/n))
  (h3 : Real.log m ≠ 0) :
  ∃ m n, (1009 * (m^2 - 1)) / (m * n) = 1 / 1009 ∧ m > 1 ∧ n > 0 ∧ m + n = 7007 := 
sorry

end smallest_m_plus_n_l297_297087


namespace felicity_gasoline_amount_l297_297246

variable D : ℝ

-- Conditions
def felicity_gasoline_usage (D : ℝ) : ℝ := 2.2 * D
def adhira_diesel_usage : ℝ := D
def total_fuel_felicity_adhira (D : ℝ) : ℝ := felicity_gasoline_usage D + adhira_diesel_usage

theorem felicity_gasoline_amount  (h1 : total_fuel_felicity_adhira D = 30) : 
  felicity_gasoline_usage D = 20.625 := 
by
  -- Proof is provided using the given conditions and equations.
  -- Mathlib usage ensures necessary libraries are imported.

  sorry

end felicity_gasoline_amount_l297_297246


namespace tangent_line_equation_l297_297383

theorem tangent_line_equation (n : ℝ) (l : LinearMap ℝ (ℝ → ℝ)) (A : ℝ × ℝ) 
  (tangent_to : ∀ x : ℝ, (x, x^n) = A → x = 2) :
  l.val = (fun x => 12 * x - x - 16) :=
by
  sorry

end tangent_line_equation_l297_297383


namespace sin_x_value_l297_297310

noncomputable def solve_sin_x_from_sec_x_plus_tan_x (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : Real :=
  if (Real.sin x = 8 / 17) then 8 / 17 else 0

theorem sin_x_value (x : ℝ) (h : Real.sec x + Real.tan x = 5 / 3) : 
  solve_sin_x_from_sec_x_plus_tan_x x h = 8 / 17 :=
sorry

end sin_x_value_l297_297310


namespace min_point_transformed_graph_l297_297904

noncomputable def original_eq (x : ℝ) : ℝ := 2 * |x| - 4

noncomputable def translated_eq (x : ℝ) : ℝ := 2 * |x - 3| - 8

theorem min_point_transformed_graph : translated_eq 3 = -8 :=
by
  -- Solution steps would go here
  sorry

end min_point_transformed_graph_l297_297904


namespace count_numbers_of_form_divisible_by_5_l297_297126

theorem count_numbers_of_form_divisible_by_5 :
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  ∃ count : ℕ, count = (card a_vals) * (card b_vals) * (card c_vals) ∧ count = 180 :=
by
  let a_vals := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals := {0, 5}
  existsi ((card a_vals) * (card b_vals) * (card c_vals))
  split
  sorry
  exact 180

end count_numbers_of_form_divisible_by_5_l297_297126


namespace pots_on_each_shelf_l297_297631

variable (x : ℕ)
variable (h1 : 4 * 3 * x = 60)

theorem pots_on_each_shelf : x = 5 := by
  -- proof will go here
  sorry

end pots_on_each_shelf_l297_297631


namespace evaluate_polynomial_at_3_l297_297939

def f (x : ℕ) : ℕ := 3 * x^7 + 2 * x^5 + 4 * x^3 + x

theorem evaluate_polynomial_at_3 : f 3 = 7158 := by
  sorry

end evaluate_polynomial_at_3_l297_297939


namespace angle_CAD_in_convex_quadrilateral_l297_297774

theorem angle_CAD_in_convex_quadrilateral
  (ABCD : Type)
  [convex_quadrilateral ABCD]
  (A B C D : ABCD)
  (h1 : AB = BD)
  (h2 : ∠A = 65)
  (h3 : ∠B = 80)
  (h4 : ∠C = 75)
  : ∠CAD = 15 := sorry

end angle_CAD_in_convex_quadrilateral_l297_297774


namespace find_function_l297_297247

def satisfies_condition (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, f a + b ∣ a^2 + f a * f b

theorem find_function :
  ∀ f : ℕ+ → ℕ+, satisfies_condition f → (∀ a : ℕ+, f a = a) :=
by
  intros f h
  sorry

end find_function_l297_297247


namespace minimize_distance_midpoint_Q5_Q6_l297_297300

theorem minimize_distance_midpoint_Q5_Q6 
  (Q : ℝ → ℝ)
  (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 : ℝ)
  (h1 : Q2 = Q1 + 1)
  (h2 : Q3 = Q2 + 1)
  (h3 : Q4 = Q3 + 1)
  (h4 : Q5 = Q4 + 1)
  (h5 : Q6 = Q5 + 2)
  (h6 : Q7 = Q6 + 2)
  (h7 : Q8 = Q7 + 2)
  (h8 : Q9 = Q8 + 2)
  (h9 : Q10 = Q9 + 2) :
  Q ((Q5 + Q6) / 2) = (Q ((Q1 + Q2) / 2) + Q ((Q3 + Q4) / 2) + Q ((Q7 + Q8) / 2) + Q ((Q9 + Q10) / 2)) :=
sorry

end minimize_distance_midpoint_Q5_Q6_l297_297300


namespace seating_arrangement_count_l297_297497

theorem seating_arrangement_count :
  let n := 6
  let rows := 2
  let seats_per_row := 3
  let seating_arrangements := (λ n rows seats_per_row hAB_near : 
    n = 6 ∧ rows = 2 ∧ seats_per_row = 3 ∧ 
    (∀ i: ℕ, i < 4 → ∃ p: ℕ, p < 2 ∧ (let A B I A_and_B_together := p * 3 + 1 → true))) → 
    ∑ p, if p = 8 * 24 then 1 else 0)
  seating_arrangements n rows seats_per_row 192 = 1 := sorry

end seating_arrangement_count_l297_297497


namespace Maxim_is_correct_l297_297155

-- Defining the parameters
def mortgage_rate := 0.125
def dividend_yield := 0.17

-- Theorem statement
theorem Maxim_is_correct : (dividend_yield - mortgage_rate > 0) := by 
    -- Dividing the proof's logical steps
    sorry

end Maxim_is_correct_l297_297155


namespace find_ratio_l297_297019

variable (Q C D : Type*) [AddCommGroup Q] [Module ℝ Q]
variables [AffineSpace Q]

def point_ratio (Q C D : Q) (v w : ℚ) : Prop :=
  Q = v • (C : Q) + w • (D : Q)

def ratio_proof (Q C D : Q) := 
  Q = (5 / 8 : ℚ) • (C : Q) + (3 / 8 : ℚ) • (D : Q)

theorem find_ratio (Q C D : Q) (h : point_ratio Q C D (3 / 8) (5 / 8)) : ratio_proof Q C D :=
  sorry

end find_ratio_l297_297019


namespace exists_good_placement_l297_297642

-- Define a function that checks if a placement is "good" with respect to a symmetry axis
def is_good (f : Fin 1983 → ℕ) : Prop :=
  ∀ (i : Fin 1983), f i < f (i + 991) ∨ f (i + 991) < f i

-- Prove the existence of a "good" placement for the regular 1983-gon
theorem exists_good_placement : ∃ f : Fin 1983 → ℕ, is_good f :=
sorry

end exists_good_placement_l297_297642


namespace odd_factors_count_l297_297365

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l297_297365


namespace find_x_given_k_l297_297276

-- Define the equation under consideration
def equation (x : ℝ) : Prop := (x - 3) / (x - 4) = (x - 5) / (x - 8)

theorem find_x_given_k {k : ℝ} (h : k = 7) : ∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → equation x → x = 2 :=
by
  intro x hx h_eq
  sorry

end find_x_given_k_l297_297276


namespace travel_remaining_distance_l297_297202

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l297_297202


namespace polarBearConsumption_l297_297669

def dailyConsumptionTrout := 0.2
def dailyConsumptionSalmon := 0.4
def dailyConsumptionHerring := 0.1
def dailyConsumptionMackerel := 0.3

def totalDailyConsumption := 
  dailyConsumptionTrout + dailyConsumptionSalmon + dailyConsumptionHerring + dailyConsumptionMackerel

def daysInMonth := 30

def totalMonthlyConsumption := totalDailyConsumption * daysInMonth

theorem polarBearConsumption : totalMonthlyConsumption = 30 := by
  sorry

end polarBearConsumption_l297_297669


namespace find_m_l297_297449

-- Definitions for the conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_of_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = a 1 * (1 - (a n / a 1)) / (1 - (a 2 / a 1))

def arithmetic_sequence (S3 S9 S6 : ℝ) :=
  2 * S9 = S3 + S6

def condition_3 (a : ℕ → ℝ) (m : ℕ) :=
  a 2 + a 5 = 2 * a m

-- Lean 4 statement that requires proof
theorem find_m 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ) 
  (geom_seq : geometric_sequence a a1 q)
  (sum_geom_seq : sum_of_geometric_sequence S a)
  (arith_seq : arithmetic_sequence (S 3) (S 9) (S 6))
  (cond3 : condition_3 a 8) : 
  8 = 8 := 
sorry

end find_m_l297_297449


namespace remainder_mul_mod_l297_297131

theorem remainder_mul_mod (a b n : ℕ) (h₁ : a ≡ 3 [MOD n]) (h₂ : b ≡ 150 [MOD n]) (n_eq : n = 400) : 
  (a * b) % n = 50 :=
by 
  sorry

end remainder_mul_mod_l297_297131


namespace find_c_value_l297_297392

variable {Triangle : Type}
variable a b c : ℝ
variable (cos_A : ℝ)

-- Given the Law of Cosines in a triangle
def law_of_cosines (a b c cos_A : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2 * b * c * cos_A

-- Given values
def given_values : Prop :=
  a = 4 ∧ b = 2 ∧ cos_A = 1/4

-- Proof goal
theorem find_c_value (h : given_values) : law_of_cosines a b c cos_A := by
  -- Since we need to prove only the theorem's statement, just add sorry as placeholder
  sorry

end find_c_value_l297_297392


namespace binary_101_is_5_l297_297653

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l297_297653


namespace arithmetic_sequence_a10_l297_297704

theorem arithmetic_sequence_a10 (a b c : ℕ) (h1 : a + b + c = 15)
  (h2 : (b + 5) * (b + 5) = (a + 2) * (c + 13)) (d : ℕ) 
  (ha : b = a + d) (hc : c = a + 2 * d) :
  a + 9 * d = 21 :=
by
  exists sorry -- this is a placeholder for the actual proof

#check arithmetic_sequence_a10

end arithmetic_sequence_a10_l297_297704


namespace shape_described_by_constant_phi_is_cone_l297_297270

-- Definition of spherical coordinates
-- (ρ, θ, φ) where ρ is the radial distance,
-- θ is the azimuthal angle, and φ is the polar angle.
structure SphericalCoordinates :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- The condition that φ is equal to a constant d
def satisfies_condition (p : SphericalCoordinates) (d : ℝ) : Prop :=
  p.φ = d

-- The main theorem to prove
theorem shape_described_by_constant_phi_is_cone (d : ℝ) :
  ∃ (S : Set SphericalCoordinates), (∀ p ∈ S, satisfies_condition p d) ∧
  (∀ p, satisfies_condition p d → ∃ ρ θ, p = ⟨ρ, θ, d⟩) ∧
  (∀ ρ θ, ρ > 0 → θ ∈ [0, 2 * Real.pi] → SphericalCoordinates.mk ρ θ d ∈ S) :=
sorry

end shape_described_by_constant_phi_is_cone_l297_297270


namespace three_digit_odd_strictly_decreasing_count_l297_297749

/-- The total number of three-digit integers where all digits are odd and in strictly decreasing order. -/
theorem three_digit_odd_strictly_decreasing_count : 
  ∃ n : ℕ, n = 10 ∧ ∀ a b c : ℕ, 
  a ∈ {1, 3, 5, 7, 9} ∧ b ∈ {1, 3, 5, 7, 9} ∧ c ∈ {1, 3, 5, 7, 9} ∧ 
  a > b ∧ b > c ↔ n = 10 :=
by {
  sorry
}

end three_digit_odd_strictly_decreasing_count_l297_297749


namespace permutation_count_l297_297015

open Nat

theorem permutation_count :
  ∀ (p : Fin 30 → Fin 30),
  (∀ i, ∃ j, p j = i) ∧ (∀ i, ∃ j, p i = j) ∧
  (∑ i, abs (p i + 1 - (i + 1))) = 450 →
  (∃! count, count = (factorial 15)^2) :=
by
  intro p h
  sorry

end permutation_count_l297_297015


namespace area_ABC_ge_two_area_AKL_l297_297623

theorem area_ABC_ge_two_area_AKL
  (A B C K L D M N : Point)
  (h1 : ∠BAC = 90)
  (h2 : altitude A D BC)
  (h3 : excenter M (Triangle A B D) AB)
  (h4 : excenter N (Triangle A C D) AC)
  (h5 : line MN intersects (extension BA) at K)
  (h6 : line MN intersects (extension CA) at L)
  : area (Triangle A B C) ≥ 2 * area (Triangle A K L) :=
sorry

end area_ABC_ge_two_area_AKL_l297_297623


namespace problem_divisibility_l297_297302

theorem problem_divisibility 
  (a b c : ℕ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : b ∣ a^3)
  (h2 : c ∣ b^3)
  (h3 : a ∣ c^3) : 
  (a + b + c) ^ 13 ∣ a * b * c := 
sorry

end problem_divisibility_l297_297302


namespace rohan_food_percentage_l297_297494

noncomputable def rohan_salary : ℝ := 7500
noncomputable def rohan_savings : ℝ := 1500
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def entertainment_percentage : ℝ := 0.10
noncomputable def conveyance_percentage : ℝ := 0.10
noncomputable def total_spent : ℝ := rohan_salary - rohan_savings
noncomputable def known_percentage : ℝ := house_rent_percentage + entertainment_percentage + conveyance_percentage

theorem rohan_food_percentage (F : ℝ) :
  total_spent = rohan_salary * (1 - known_percentage - F) →
  F = 0.20 :=
sorry

end rohan_food_percentage_l297_297494


namespace chess_team_boys_l297_297164

variable {B G : ℕ}

theorem chess_team_boys
    (h1 : B + G = 30)
    (h2 : 1/3 * G + B = 18) :
    B = 12 :=
by
  sorry

end chess_team_boys_l297_297164


namespace max_x_on_circle_l297_297016

theorem max_x_on_circle : 
  ∀ x y : ℝ,
  (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by
  intros x y h
  sorry

end max_x_on_circle_l297_297016


namespace find_jamals_grade_l297_297769

noncomputable def jamals_grade (n_students : ℕ) (absent_students : ℕ) (test_avg_28_students : ℕ) (new_total_avg_30_students : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let total_28_students := 28 * test_avg_28_students
  let total_30_students := 30 * new_total_avg_30_students
  let combined_score := total_30_students - total_28_students
  combined_score - taqeesha_score

theorem find_jamals_grade :
  jamals_grade 30 2 85 86 92 = 108 :=
by
  sorry

end find_jamals_grade_l297_297769


namespace min_area_triangle_MON_l297_297462

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1
noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 9

noncomputable def pointP_in_ellipse (x y : ℝ) : Prop :=
  ellipse x y ∧ 0 < x ∧ 0 < y

noncomputable def tangent_line (x y : ℝ) (θ : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 9

theorem min_area_triangle_MON :
  ∀ (P : ℝ × ℝ) (θ : ℝ),
    pointP_in_ellipse P.1 P.2 →
    (0 < θ ∧ θ < Real.pi / 2) →
    (∃ M N : ℝ × ℝ,
      tangent_line P.1 P.2 θ ∧
      tangent_line P.1 P.2 θ ∧
      (M.2 = 0) ∧ (N.1 = 0) ∧
      ⋆ :=
by
  sorry

end min_area_triangle_MON_l297_297462


namespace locus_of_Q_l297_297851

theorem locus_of_Q (P Q : ℝ × ℝ) :
  (∃ θ : ℝ, P = (2 + cos θ, 1 + sin θ)) ∧
  (Q = (3 + cos θ + sin θ, -1 - cos θ + sin θ)) →
  ∃ (x y : ℝ), (Q = (x, y) ∧ (x - 3)^2 + (y + 1)^2 = 2) :=
by
  intro h
  rcases h with ⟨⟨θ, P_def⟩, Q_def⟩
  use [3 + cos θ + sin θ, -1 - cos θ + sin θ]
  rw [Q_def]
  split
  · refl
  · rw [←add_assoc, ←add_assoc]
    calc
      (3 + (cos θ + sin θ) - 3)^2 + (-1 + (-cos θ + sin θ) + 1)^2
          = 2 * (cos θ + sin θ)^2 + 2 * (-cos θ + sin θ)^2 : by sorry
      ... = 2 * (cos θ + sin θ)^2 + 2 * (cos θ - sin θ)^2 : by sorry
      ... = 2 * (cos θ + sin θ)^2 + 2 * (cos (-θ) - sin (-θ))^2 : by sorry
      ... = 2 : by sorry

end locus_of_Q_l297_297851


namespace area_trapezoid_ABCD_l297_297000

theorem area_trapezoid_ABCD (A B C D O : Type*) 
  (AD : ℝ) (AB BD : ℝ) 
  (angle_CBD : ℝ) 
  (AB_BD_sum : AB + BD = 40) 
  (AD_value : AD = 16) 
  (angle_CBD_value : angle_CBD = 60) 
  (ratio_areas_O : S_ABO / S_BOC = 2) :
  S_ABCD = 126 * sqrt(3) := sorry

end area_trapezoid_ABCD_l297_297000


namespace solveTrigEquation_solution_l297_297066

noncomputable def solveTrigEquation (x : Real) : Prop :=
  (((2 * sin (3 * x)) / (sin x)) - ((cos (3 * x)) / (cos x))) = 5 * abs (cos x)

theorem solveTrigEquation_solution (x : Real) (k : ℤ) (h1 : cos x * sin x ≠ 0):
  solveTrigEquation x ↔ 
  (x = Real.arccos (1 / 4) + k * Real.pi ∨ x = -Real.arccos (1 / 4) + k * Real.pi) :=
sorry

end solveTrigEquation_solution_l297_297066


namespace magnitude_b_l297_297345

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions given in the problem
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := inner a b = 1
def condition3 : Prop := ‖a + b‖ = √5

-- The theorem to prove
theorem magnitude_b (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 a b) : ‖b‖ = √2 :=
  sorry

end magnitude_b_l297_297345


namespace possible_values_of_reciprocal_sum_l297_297813

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  ∃ x, x ∈ set.Ici (2:ℝ) ∧ x = (1 / a + 1 / b) :=
by sorry

end possible_values_of_reciprocal_sum_l297_297813


namespace temperature_representation_l297_297370

def represents_zero_degrees_celsius (t₁ : ℝ) : Prop := t₁ = 10

theorem temperature_representation (t₁ t₂ : ℝ) (h₀ : represents_zero_degrees_celsius t₁) 
    (h₁ : t₂ > t₁):
    t₂ = 17 :=
by
  -- Proof is omitted here
  sorry

end temperature_representation_l297_297370


namespace fraction_power_l297_297633

theorem fraction_power (a b : ℕ) (ha : a = 5) (hb : b = 6) : (a / b : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end fraction_power_l297_297633


namespace student_selected_probability_l297_297171

theorem student_selected_probability :
  ∀ (total_students sampled_students : ℕ), 
  total_students = 303 → 
  sampled_students = 50 → 
  (∀ student : ℕ, student ∈ finset.range total_students → 
  (student_selected_probability student total_students sampled_students = 50 / 303)) :=
begin
  sorry
end

end student_selected_probability_l297_297171


namespace hyperbola_eccentricity_sqrt3_l297_297702

theorem hyperbola_eccentricity_sqrt3
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b / a = Real.sqrt 2) :
  (let e := Real.sqrt (1 + (b^2 / a^2)) in e = Real.sqrt 3) :=
by
  sorry

end hyperbola_eccentricity_sqrt3_l297_297702


namespace points_per_touchdown_l297_297077

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l297_297077


namespace part1_part2_l297_297326

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * (cos x)^2 - sqrt 3

noncomputable def smallest_positive_period (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x ∧ (∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x)

noncomputable def value_of_cos_2x0 (x0 : ℝ) : Prop := 
  ∀ x0 ∈ Icc (π / 4) (π / 2), f (x0 - π / 12) = 6 / 5 → cos (2 * x0) = (3 - 4 * sqrt 3) / 10

-- Proof for the smallest positive period
theorem part1 : smallest_positive_period π := 
by 
  sorry

-- Proof for the value of cos(2 * x0)
theorem part2 : value_of_cos_2x0 x0 := 
by 
  sorry

end part1_part2_l297_297326


namespace log_base_3x_eq_729_l297_297371

theorem log_base_3x_eq_729 (x : ℝ) (hx : (3 * x) > 0) (h : Real.log 729 / Real.log (3 * x) = x) :
  x = 3 ∧ Nat.isNonSquareNonCube 3 :=
by
  -- Proof goes here
  sorry

def Nat.isNonSquareNonCube (n : ℕ) : Prop :=
  ¬ ∃ k : ℕ, k ^ 2 = n ∧ ¬ ∃ m : ℕ, m ^ 3 = n

end log_base_3x_eq_729_l297_297371


namespace employees_bonus_l297_297916

theorem employees_bonus (x y z : ℝ) 
  (h1 : x + y + z = 2970) 
  (h2 : y = (1 / 3) * x + 180) 
  (h3 : z = (1 / 3) * y + 130) :
  x = 1800 ∧ y = 780 ∧ z = 390 :=
by
  sorry

end employees_bonus_l297_297916


namespace Alice_wins_tournament_l297_297198

noncomputable def probAliceWins : ℚ := 6 / 7

theorem Alice_wins_tournament :
  let n := 8
  let playStyle (player: ℕ) : String := if player == 0 then "rock"
                                         else if player == 1 then "paper"
                                         else "scissors"
  -- Define rules for rock, paper, scissors
  let beats (a b : String) : Bool := if (a = "rock" ∧ b = "scissors") ∨
                                       (a = "scissors" ∧ b = "paper") ∨
                                       (a = "paper" ∧ b = "rock")
                                     then true
                                     else false
  -- Define a function to pair players randomly and simulate the tournament
  -- Pseudocode: pair up players, simulate matches, and find winner iteratively
  
  let randomPairingLogic (p : ℕ) : String := sorry -- Simulation logic here  

  -- Probability that Alice (Player 0) wins under given conditions
  let probAliceWins : ℚ := 6 / 7
in 

probAliceWins = 6 / 7 := sorry

end Alice_wins_tournament_l297_297198


namespace binary_multiplication_correct_l297_297261

theorem binary_multiplication_correct :
  nat.of_digits 2 [1, 0, 1, 0, 0, 1, 1] = 
  (nat.of_digits 2 [1, 1, 0, 1] * nat.of_digits 2 [1, 1, 1]) :=
by
  sorry

end binary_multiplication_correct_l297_297261


namespace eventually_constant_l297_297209

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def good_sequence (a : ℕ → ℕ) : Prop :=
  (is_perfect_square (a 1)) ∧
  ∀ n : ℕ, n ≥ 2 → ∃ k : ℕ, k < a n ∧ is_perfect_square (n * a 1 + (n - 1) * a 2 + ... + 2 * a (n - 1) + a n)

theorem eventually_constant {a : ℕ → ℕ} (h : good_sequence a) : 
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a n = a k :=
  sorry

end eventually_constant_l297_297209


namespace area_union_of_reflected_triangles_l297_297196

def point : Type := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_y_eq_1 (P : point) : point := (P.1, 2 * 1 - P.2)

def area_of_union (A B C : point) (f : point → point) : ℝ :=
  let A' := f A
  let B' := f B
  let C' := f C
  triangle_area A B C + triangle_area A' B' C'

theorem area_union_of_reflected_triangles :
  area_of_union (3, 4) (5, -2) (6, 2) reflect_y_eq_1 = 11 :=
  sorry

end area_union_of_reflected_triangles_l297_297196


namespace remainder_102_104_plus_6_div_9_l297_297130

theorem remainder_102_104_plus_6_div_9 :
  ((102 * 104 + 6) % 9) = 3 :=
by
  sorry

end remainder_102_104_plus_6_div_9_l297_297130


namespace abs_expression_simplification_l297_297377

theorem abs_expression_simplification (x : ℝ) (h : x < -2) : |1 - |1 + x|| = 2 + x := 
by 
  sorry

end abs_expression_simplification_l297_297377


namespace total_sleep_correct_l297_297426

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end total_sleep_correct_l297_297426


namespace min_face_sum_l297_297953

/-- The configuration of the numbers 1, 2, 3, 4, 5, 6, 7, 8 on the vertices of a cube
such that the sum of any three numbers on each face is at least 10, has the 
minimum possible sum of the four numbers on any face equal to 16. -/
theorem min_face_sum (a b c d e f g h : ℕ)
  (h_unique: {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_condition : ∀ (x y z : ℕ), {x, y, z} ⊆ {a, b, c, d, e, f, g, h} → x + y + z ≥ 10):
  ∃ (w x y z : ℕ), 
   {w, x, y, z} ⊆ {a, b, c, d, e, f, g, h} ∧ 
   w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
   w + x + y + z = 16 :=
  sorry

end min_face_sum_l297_297953


namespace cindy_jump_time_proof_l297_297636

-- Definitions based on the given conditions
def cindy_jump_time := ℕ -- Cindy's jump time in minutes
def betsy_jump_time (C : ℕ) := C / 2 -- Betsy's jump time in minutes, half of Cindy's time
def tina_jump_time (C : ℕ) := 3 * (C / 2) -- Tina's jump time in minutes, three times Betsy's time

-- Theorem stating that Cindy can jump rope for 12 minutes
theorem cindy_jump_time_proof (C : ℕ) :
  tina_jump_time(C) = C + 6 → 
  C = 12 :=
by
  -- Skipping the formal proof
  sorry

end cindy_jump_time_proof_l297_297636


namespace kyle_sales_money_proof_l297_297440

variable (initial_cookies initial_brownies : Nat)
variable (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
variable (price_per_cookie price_per_brownie : Float)

def kyle_total_money (initial_cookies initial_brownies : Nat) 
    (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
    (price_per_cookie price_per_brownie : Float) : Float := 
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies)
  let remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies)
  let money_from_cookies := remaining_cookies * price_per_cookie
  let money_from_brownies := remaining_brownies * price_per_brownie
  money_from_cookies + money_from_brownies

theorem kyle_sales_money_proof : 
  kyle_total_money 60 32 2 1 2 2 1 1.50 = 99 :=
by
  sorry

end kyle_sales_money_proof_l297_297440


namespace initial_lives_emily_l297_297670

theorem initial_lives_emily (L : ℕ) (h1 : L - 25 + 24 = 41) : L = 42 :=
by
  sorry

end initial_lives_emily_l297_297670


namespace Kyle_makes_99_dollars_l297_297434

-- Define the initial numbers of cookies and brownies
def initial_cookies := 60
def initial_brownies := 32

-- Define the numbers of cookies and brownies eaten by Kyle and his mom
def kyle_eats_cookies := 2
def kyle_eats_brownies := 2
def mom_eats_cookies := 1
def mom_eats_brownies := 2

-- Define the prices for each cookie and brownie
def price_per_cookie := 1
def price_per_brownie := 1.50

-- Define the remaining cookies and brownies after consumption
def remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
def remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies

-- Define the total money Kyle will make
def money_from_cookies := remaining_cookies * price_per_cookie
def money_from_brownies := remaining_brownies * price_per_brownie

-- Define the total money Kyle will make from selling all baked goods
def total_money := money_from_cookies + money_from_brownies

-- Proof statement
theorem Kyle_makes_99_dollars :
  total_money = 99 :=
by
  sorry

end Kyle_makes_99_dollars_l297_297434


namespace roots_not_both_less_than_one_third_l297_297942

theorem roots_not_both_less_than_one_third
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_discriminant : b^2 - 4 * a * c ≥ 0):
  ¬ (∀ (x₁ x₂ : ℝ), (a * x₁^2 - b * x₁ + c = 0) ∧ (a * x₂^2 - b * x₂ + c = 0) ∧ x₁ < 1/3 ∧ x₂ < 1/3) :=
begin
  -- introduce strategy to explore contradiction as in problem solution
  
  sorry
end

end roots_not_both_less_than_one_third_l297_297942


namespace B_completion_time_l297_297570

theorem B_completion_time (A_completion_time : ℝ) (A_work_days : ℝ) (B_remaining_work_days : ℝ) : ℝ :=
  let A_total_work := 1
  let B_total_work := 1
  let A_work_rate := A_total_work / A_completion_time
  let B_work_rate := B_total_work / B_remaining_work_days
  let work_done_by_A := A_work_rate * A_work_days
  let remaining_work := A_total_work - work_done_by_A
  B_total_work / (remaining_work / B_remaining_work_days)
  
example : B_completion_time 15 5 18 = 27 := by
  simp [B_completion_time]
  sorry

end B_completion_time_l297_297570


namespace cubic_sum_l297_297720

theorem cubic_sum (n : ℕ) (x : Fin n → ℤ)
  (h1 : ∀ i, x i ∈ {-2, 0, 1})
  (h2 : ∑ i, x i = -17)
  (h3 : ∑ i, (x i)^2 = 37) :
  ∑ i, (x i)^3 = -71 :=
sorry

end cubic_sum_l297_297720


namespace area_of_circle_l297_297660

-- Define the condition that a point (x, y) lies on the circle
def point_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y + 9 = 0

-- Prove that the area enclosed by the described circle is 16π
theorem area_of_circle :
  (∃ (x y : ℝ), point_on_circle x y) →
  ∀ (r : ℝ), (x^2 - 6*x + y^2 + 8*y + 9 = 0) → 
  (x - 3)^2 + (y + 4)^2 = r^2 → 
  r = 4 → real.pi * r^2 = 16 * real.pi :=
sorry

end area_of_circle_l297_297660


namespace infinite_arithmetic_progression_intersects_segments_l297_297621

open Classical

-- Define segments and their properties
structure Segment :=
(start end : ℝ)
(length : ℝ := end - start)
(non_overlapping : ∀ (s1 s2 : Segment), s1 ≠ s2 → (s1.end ≤ s2.start ∨ s2.end ≤ s1.start))

-- Define the arithmetic progression
def arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem infinite_arithmetic_progression_intersects_segments:
  ∀ (a d : ℝ) (S : set Segment), 
    (∀ s ∈ S, s.length = 1) →
    (∀ s1 s2 ∈ S, s1 ≠ s2 → (s1.end ≤ s2.start ∨ s2.end ≤ s1.start)) →
    ∃ (s : Segment) (n : ℕ), s ∈ S ∧ (arithmetic_progression a d n ∈ set.Icc s.start s.end) :=
by
  sorry

end infinite_arithmetic_progression_intersects_segments_l297_297621


namespace binary_multiplication_correct_l297_297260

theorem binary_multiplication_correct :
  nat.of_digits 2 [1, 0, 1, 0, 0, 1, 1] = 
  (nat.of_digits 2 [1, 1, 0, 1] * nat.of_digits 2 [1, 1, 1]) :=
by
  sorry

end binary_multiplication_correct_l297_297260


namespace area_of_triangle_l297_297509

-- Define the curve y = (1/3)x^3 + x
def curve (x : ℝ) : ℝ := (1/3) * x^3 + x

-- Given point on the curve
def point_on_curve : ℝ × ℝ := (1, (4/3))

-- The equation for the tangent line at the given point (1, 4/3) with slope 2
def tangent_line (x: ℝ) : ℝ := 2 * (x - 1) + (4 / 3)

-- Prove that the area of the triangle formed by the tangent line and the axes is 1/9
theorem area_of_triangle : 
  let x_intercept := (1 / 3) in
  let y_intercept := -(2 / 3) in
  let area := (1 / 2) * x_intercept * -y_intercept in
  area = (1 / 9) :=
by
  let x_intercept := (1 / 3)
  let y_intercept := -(2 / 3)
  let area := (1 / 2) * x_intercept * -y_intercept
  show area = (1 / 9)
  exact sorry

end area_of_triangle_l297_297509


namespace smallest_consecutive_even_number_l297_297881

/-- Let a, a+2, a+4, a+6, and a+8 represent five consecutive even numbers whose sum is 420. 
    We need to prove that the smallest number among them is 80. -/
theorem smallest_consecutive_even_number (a : ℤ) (h : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 420) : 
  a = 80 :=
begin
  sorry
end

end smallest_consecutive_even_number_l297_297881


namespace negation_of_P_l297_297738

variable (x0 : ℝ)
variable (P : Prop)
variable (h : ∃ x0 : ℝ, x0 > 0 ∧ log x0 / log 2 = 1)

theorem negation_of_P : 
  (¬ (∃ x0 : ℝ, x0 > 0 ∧ log x0 / log 2 = 1)) ↔ (∀ x0 : ℝ, x0 > 0 → log x0 / log 2 ≠ 1) := 
by
  assume h
  sorry

end negation_of_P_l297_297738


namespace tyler_total_puppies_l297_297555

/-- 
  Tyler has 15 dogs, and each dog has 5 puppies.
  We want to prove that the total number of puppies is 75.
-/
def tyler_dogs : Nat := 15
def puppies_per_dog : Nat := 5
def total_puppies_tyler_has : Nat := tyler_dogs * puppies_per_dog

theorem tyler_total_puppies : total_puppies_tyler_has = 75 := by
  sorry

end tyler_total_puppies_l297_297555


namespace number_of_books_bought_l297_297472

def initial_books : ℕ := 35
def books_given_away : ℕ := 12
def final_books : ℕ := 56

theorem number_of_books_bought : initial_books - books_given_away + (final_books - (initial_books - books_given_away)) = final_books :=
by
  sorry

end number_of_books_bought_l297_297472


namespace sunil_total_amount_l297_297573

noncomputable def total_amount_sunil : ℝ :=
  let CI := 2828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := CI / ((1 + r / n) ^ (n * t) - 1)
  P + CI

theorem sunil_total_amount : total_amount_sunil = 19828.80 :=
by
  sorry

end sunil_total_amount_l297_297573


namespace time_to_cross_bridge_l297_297595

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (speed_conversion_factor : ℝ) (time_conversion_factor : ℝ) (expected_time : ℝ) :
  speed_km_hr = 5 →
  length_m = 1250 →
  speed_conversion_factor = 1000 →
  time_conversion_factor = 60 →
  expected_time = length_m / (speed_km_hr * (speed_conversion_factor / time_conversion_factor)) →
  expected_time = 15 :=
by
  intros
  sorry

end time_to_cross_bridge_l297_297595


namespace euler_line_equation_l297_297039

theorem euler_line_equation :
  ∀ (A B : ℝ × ℝ),
    A = (2, 0) →
    B = (0, 1) →
    (∃ (C : ℝ × ℝ), dist A C = dist B C) →
    ∀ (x y : ℝ), 4 * x - 2 * y - 3 = 0 :=
by
  let A := (2 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 1 : ℝ)
  intro h₁ h₂ h₃ x y
  sorry

end euler_line_equation_l297_297039


namespace three_letter_initials_conditions_l297_297746

-- Definitions for the given problem conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
def vowels : List Char := ['A', 'E', 'I']
def consonants : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J']

-- The target statement to prove
theorem three_letter_initials_conditions :
  (∃ (vowel_pos : Fin 3), 
    ∃ (vowel : Char), vowel ∈ vowels ∧
    ∃ (c1 c2 : Char), c1 ∈ consonants ∧ c2 ∈ consonants ∧
    ([c1, vowel, c2].toFinList.get vowel_pos = vowel ∨
     [c1, c2, vowel].toFinList.get vowel_pos = vowel ∨
     [vowel, c1, c2].toFinList.get vowel_pos = vowel)
  ) = 441 := 
sorry

end three_letter_initials_conditions_l297_297746


namespace cost_per_metre_of_carpet_l297_297514

theorem cost_per_metre_of_carpet :
  (length_of_room = 18) →
  (breadth_of_room = 7.5) →
  (carpet_width = 0.75) →
  (total_cost = 810) →
  (cost_per_metre = 4.5) :=
by
  intros length_of_room breadth_of_room carpet_width total_cost
  sorry

end cost_per_metre_of_carpet_l297_297514


namespace hole_movable_to_any_corner_l297_297786

-- Define the dimensions of the box with m and n being odd
variables (m n : ℕ) (hm : odd m) (hn : odd n)

-- A 2x1 domino placement and a function to describe a move
structure Domino :=
(x : ℕ)
(y : ℕ)
(valid : x < m ∧ y < n ∧ x % 2 = 0 ∧ (y + 1) % 2 = 0)

-- Initial hole at the (1, 1) position (assuming 1-indexed)
def initial_hole : ℕ × ℕ := (1, 1)

-- Definition of a move: from (x, y) to (x + 1, y) or (x - 1, y)
def move_hole (hole : ℕ × ℕ) (d : Domino) : ℕ × ℕ :=
if hole = (d.x, d.y) then (d.x + 1, d.y)
else if hole = (d.x + 1, d.y) then (d.x - 1, d.y)
else hole

-- Prove that the hole can be moved to any other corner
theorem hole_movable_to_any_corner :
  ∀ (target : ℕ × ℕ), target ∈ {(1, 1), (1, n), (m, 1), (m, n)} →
  ∃ (moves : list Domino), list.foldl move_hole initial_hole moves = target :=
by { intros, sorry }

end hole_movable_to_any_corner_l297_297786


namespace ben_gave_18_fish_l297_297843

variable (initial_fish : ℕ) (total_fish : ℕ) (given_fish : ℕ)

theorem ben_gave_18_fish
    (h1 : initial_fish = 31)
    (h2 : total_fish = 49)
    (h3 : total_fish = initial_fish + given_fish) :
    given_fish = 18 :=
by
  sorry

end ben_gave_18_fish_l297_297843


namespace xy_product_l297_297866

theorem xy_product (x y : ℝ) (h1 : 2 ^ x = 16 ^ (y + 1)) (h2 : 27 ^ y = 3 ^ (x - 2)) : x * y = 8 :=
by
  sorry

end xy_product_l297_297866


namespace maximum_value_of_n_l297_297811

noncomputable def max_n (a b c : ℝ) (n : ℕ) :=
  a > b ∧ b > c ∧ (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2

theorem maximum_value_of_n (a b c : ℝ) (n : ℕ) : 
  a > b → b > c → (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2 :=
  by sorry

end maximum_value_of_n_l297_297811


namespace ricky_roses_l297_297492

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end ricky_roses_l297_297492


namespace sum_of_first_2017_terms_l297_297706

-- Define the arithmetic sequence a_n
def a_seq (n : ℕ) : ℕ := n + 1

-- Conditions given in the problem
def a_2_eq_2 : (a_seq 2) = 2 := sorry
def sum_S_5_eq_15 : (5 * a_seq 1 + 10 * (a_seq 2) / 2) = 15 := sorry

-- Sum of sequence {1 / (a_n * a_{n+1})}
def sum_seq (n : ℕ) : ℚ :=
  (List.range n).map (λ k, 1 / ((a_seq k) * (a_seq (k+1)))).sum

-- Statement to be proved
theorem sum_of_first_2017_terms :
  sum_seq 2017 = 2017/2018 := sorry

end sum_of_first_2017_terms_l297_297706


namespace part1_solution_set_part2_solution_l297_297834

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem part1_solution_set :
  {x : ℝ | f x > 2} = {x | x > 1} ∪ {x | x < -5} :=
by
  sorry

theorem part2_solution (t : ℝ) :
  (∀ x, f x ≥ t^2 - (11 / 2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) :=
by
  sorry

end part1_solution_set_part2_solution_l297_297834


namespace odd_factors_count_l297_297364

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l297_297364


namespace find_missing_number_l297_297787

def initial_set : Set ℤ := {6, 3, 8, 4}

def inserted_value : ℤ := 7

def desired_median : ℤ := 10

/-- The missing number in the set {_, 6, 3, 8, 4} such that the median value of the set 
(with the value 7 inserted) is 10. -/
theorem find_missing_number (x : ℤ) : 
  (∀ (s : Set ℤ), s = initial_set ∪ {x} ∪ {inserted_value} → 
  let sorted_s := List.sort (s.toList) in 
  (sorted_s.nth (sorted_s.length / 2 - 1)).getD 0 + (sorted_s.nth (sorted_s.length / 2)).getD 0 = 2 * desired_median) 
  → x = 14 :=
by
  sorry

end find_missing_number_l297_297787


namespace rowing_time_75_minutes_l297_297239

-- Definition of time duration Ethan rowed.
def EthanRowingTime : ℕ := 25  -- minutes

-- Definition of the time duration Frank rowed.
def FrankRowingTime : ℕ := 2 * EthanRowingTime  -- twice as long as Ethan.

-- Definition of the total rowing time.
def TotalRowingTime : ℕ := EthanRowingTime + FrankRowingTime

-- Theorem statement proving the total rowing time is 75 minutes.
theorem rowing_time_75_minutes : TotalRowingTime = 75 := by
  -- The proof is omitted.
  sorry

end rowing_time_75_minutes_l297_297239


namespace find_constants_l297_297679

theorem find_constants (x : ℝ) :
    (\frac{x^2 - 10 * x + 16}{(x - 2) * (x - 3) * (x - 4)})
    = (\frac{2}{x - 2}) + (\frac{5}{x - 3}) + (\frac{0}{x - 4}) :=
by
  sorry

end find_constants_l297_297679


namespace arithmetic_sequence_a1_l297_297810

theorem arithmetic_sequence_a1
  (a : ℕ → ℝ)  -- define the arithmetic sequence {a_n}
  (s : ℕ → ℝ)  -- define the sum sequence {s_n}
  (d : ℝ := -2)  -- common difference
  (h_diff : ∀ n, a n = a 1 + (n - 1) * d)  -- the n-th term formula
  (h_sum : ∀ n, s n = ∑ i in finset.range (n + 1), a i)  -- the sum of the first n terms
  (h_eq : s 10 = s 11)  -- given condition s_10 = s_11
  : a 1 = 20 := 
sorry

end arithmetic_sequence_a1_l297_297810


namespace trees_variance_l297_297048

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end trees_variance_l297_297048


namespace area_of_circle_from_diameter_l297_297852

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def area_of_circle (A B : ℝ × ℝ) : ℝ :=
  let d := distance A.1 A.2 B.1 B.2 in
  real.pi * (d / 2)^2

-- Define the points A and B
def A : ℝ × ℝ := (3, 5)
def B : ℝ × ℝ := (7, 10)

-- State the theorem
theorem area_of_circle_from_diameter : 
  area_of_circle A B = (41 * real.pi) / 4 :=
by sorry

end area_of_circle_from_diameter_l297_297852


namespace value_of_a_plus_b_l297_297755

theorem value_of_a_plus_b (a b : ℝ) (h1 : a + 2 * b = 8) (h2 : 3 * a + 4 * b = 18) : a + b = 5 := 
by 
  sorry

end value_of_a_plus_b_l297_297755


namespace multiple_of_other_number_l297_297849

theorem multiple_of_other_number (S L k : ℤ) (h₁ : S = 18) (h₂ : L = k * S - 3) (h₃ : S + L = 51) : k = 2 :=
by
  sorry

end multiple_of_other_number_l297_297849


namespace value_of_a_plus_d_l297_297758

variable (a b c d : ℝ)

theorem value_of_a_plus_d
  (h1 : a + b = 4)
  (h2 : b + c = 5)
  (h3 : c + d = 3) :
  a + d = 1 :=
by
sorry

end value_of_a_plus_d_l297_297758


namespace find_value_f2016_f2017_f2018_l297_297297

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic : ∀ x : ℝ, f(x + 4) = f(x)
axiom f1 : f(1) = 1

theorem find_value_f2016_f2017_f2018 : f(2016) + f(2017) + f(2018) = 1 :=
  sorry

end find_value_f2016_f2017_f2018_l297_297297


namespace compute_100c_d_l297_297455

theorem compute_100c_d (c d : ℝ)
  (h1 : ∀ x, (x + c) * (x + d) * (x - 7) = 0 → x ∈ {-c, -d, 7})
  (h2 : ∀ x, (x + 2 * c) * (x + 5) * (x + 8) = 0 → x = -5 → (x + d) ≠ 0 ∧ (x - 7) ≠ 0) :
  100 * c + d = 408 :=
sorry

end compute_100c_d_l297_297455


namespace min_common_perimeter_is_508_l297_297117

/- Definitions related to the problem -/

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

def area (base : ℕ) (side : ℕ) : Real :=
  ∑ sqrt(side^2 - (base^2 / 4))

noncomputable def min_common_perimeter : ℕ :=
  Nat.find ?_

theorem min_common_perimeter_is_508 :
  ∃ (a b c d : ℕ), is_isosceles_triangle a a (3 * b) ∧ is_isosceles_triangle c c (2 * d) ∧ (perimeter a a (3 * b) = perimeter c c (2 * d)) ∧ (3 * b * sqrt(a^2 - (3 * b / 2)^2) = 2 * d * sqrt(c^2 - (d / 2)^2)) ∧  ∃ k : ℕ, 3 / 2 = k / 1 ∧ (perimeter a a (3 * b) = 508) := 
by sorry

end min_common_perimeter_is_508_l297_297117


namespace circle_center_polar_coordinates_max_distance_to_line_l297_297406

noncomputable def parametric_circle (r θ : ℝ) : ℝ × ℝ :=
  ( - (Real.sqrt 2) / 2 + r * Real.cos θ, - (Real.sqrt 2) / 2 + r * Real.sin θ)

noncomputable def polar_line (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + (Real.pi / 4)) = (Real.sqrt 2) / 2

theorem circle_center_polar_coordinates : (1, 5 * Real.pi / 4) =
  let c := parametric_circle 0 0 in
  (Real.sqrt (c.1^2 + c.2^2), Real.arctan (c.2 / c.1) + if c.1 < 0 then Real.pi else 0) := sorry

theorem max_distance_to_line (r : ℝ) : 
  let d := |r * Real.sqrt 2 + 1| / Real.sqrt 2 
  in d + r = 3 ↔ r = (2 - Real.sqrt 2 / 2) := sorry

end circle_center_polar_coordinates_max_distance_to_line_l297_297406


namespace player_win_even_odd_first_player_wins_adjacent_initial_l297_297994

-- Define the conditions of the game.
structure Chessboard (n : ℕ) :=
  (valid_position : ℕ × ℕ → Prop)
  (initial_position : ℕ × ℕ)
  (adjacent : ℕ × ℕ → ℕ × ℕ → Prop)
  (not_revisited : list (ℕ × ℕ) → Prop)

-- Define the winning strategy for different scenarios.
def first_player_wins (n : ℕ) : Prop :=
  ∃ strategy1 : (list (ℕ × ℕ) → ℕ × ℕ), ∀ history, ¬revisited history → valid_move strategy1 history

def second_player_wins (n : ℕ) : Prop :=
  ∃ strategy2 : (list (ℕ × ℕ) → ℕ × ℕ), ∀ history, ¬revisited history → valid_move strategy2 history

-- Part (a) statement in Lean
theorem player_win_even_odd (n : ℕ) (c : Chessboard n) :
  (even n → first_player_wins n) ∧ (odd n → second_player_wins n) :=
sorry

-- Part (b) statement in Lean
theorem first_player_wins_adjacent_initial (n : ℕ) (c : Chessboard n) :
  adjacent_to_corner (c.initial_position) → first_player_wins n :=
sorry

-- Helper definition to check adjacency to the corner.
def adjacent_to_corner (pos : ℕ × ℕ) : Prop :=
  (pos = (0, 1) ∨ pos = (1, 0)) ∨ (pos = (n-1, 0) ∨ pos = (n, 1)) ∨ 
  (pos = (0, n-1) ∨ pos = (1, n)) ∨ (pos = (n-1, n) ∨ pos = (n, n-1))

end player_win_even_odd_first_player_wins_adjacent_initial_l297_297994


namespace part1_part2_second_tier_part2_third_tier_part3_l297_297585

variables {x : ℝ}

-- Definitions for the tiered pricing system
def first_tier (kWh : ℝ) : Prop := kWh ≤ 170
def second_tier (kWh : ℝ) : Prop := 171 ≤ kWh ∧ kWh ≤ 260
def third_tier (kWh : ℝ) : Prop := kWh > 260

-- Price per kWh for each tier
def price_first_tier := 0.52
def price_second_tier := 0.57
def price_third_tier := 0.82

-- Proof for part (1) 
theorem part1 : first_tier 160 → 160 * price_first_tier = 83.2 := by
  intro h
  sorry

-- Proof for part (2), second tier formula
theorem part2_second_tier : second_tier x → (170 * price_first_tier + (x - 170) * price_second_tier) = 0.57 * x - 8.5 := by
  intro h
  sorry

-- Proof for part (2), third tier formula
theorem part2_third_tier : third_tier x → 
  (170 * price_first_tier + 90 * price_second_tier + (x - 260) * price_third_tier) = 0.82 * x - 73.5 := by
  intro h
  sorry

-- Proof for part (3)
theorem part3 : second_tier 240 → 240 * price_second_tier - 8.5 = 128.3 := by
  intro h
  sorry

end part1_part2_second_tier_part2_third_tier_part3_l297_297585


namespace Carl_can_buy_six_candy_bars_l297_297242

-- Define the earnings per week
def earnings_per_week : ℝ := 0.75

-- Define the number of weeks Carl works
def weeks : ℕ := 4

-- Define the cost of one candy bar
def cost_per_candy_bar : ℝ := 0.50

-- Calculate total earnings
def total_earnings := earnings_per_week * weeks

-- Calculate the number of candy bars Carl can buy
def number_of_candy_bars := total_earnings / cost_per_candy_bar

-- State the theorem that Carl can buy exactly 6 candy bars
theorem Carl_can_buy_six_candy_bars : number_of_candy_bars = 6 := by
  sorry

end Carl_can_buy_six_candy_bars_l297_297242


namespace find_w_squared_l297_297750

theorem find_w_squared (w : ℝ) (h : (2 * w + 19) ^ 2 = (4 * w + 9) * (3 * w + 13)) :
  w ^ 2 = ((6 + Real.sqrt 524) / 4) ^ 2 :=
sorry

end find_w_squared_l297_297750


namespace correct_statement_l297_297563

def angle_terminal_side (a b : ℝ) : Prop :=
∃ k : ℤ, a = b + k * 360

def obtuse_angle (θ : ℝ) : Prop :=
90 < θ ∧ θ < 180

def third_quadrant_angle (θ : ℝ) : Prop :=
180 < θ ∧ θ < 270

def first_quadrant_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

def acute_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

theorem correct_statement :
  ¬∀ a b, angle_terminal_side a b → a = b ∧
  ¬∀ θ, obtuse_angle θ → θ < θ - 360 ∧
  ¬∀ θ, first_quadrant_angle θ → acute_angle θ ∧
  ∀ θ, acute_angle θ → first_quadrant_angle θ :=
by
  sorry

end correct_statement_l297_297563


namespace vertex_of_quadratic_l297_297725

theorem vertex_of_quadratic :
  ∃ h k : ℝ, ∀ x : ℝ, -((x - 1) ^ 2) + 2 = (h, k) :=
begin
  use [1, 2], -- the vertex (h, k) is (1, 2)
  sorry
end

end vertex_of_quadratic_l297_297725


namespace john_total_cost_after_discount_l297_297606

/-- A store gives a 10% discount for the amount of the sell that was over $1000.
John buys 7 items that each cost $200. What does his order cost after the discount? -/
theorem john_total_cost_after_discount : 
  let discount_rate := 0.1
  let threshold := 1000
  let item_cost := 200
  let item_count := 7
  let total_cost := item_cost * item_count
  let discount := discount_rate * max 0 (total_cost - threshold)
  let final_cost := total_cost - discount
  in final_cost = 1360 :=
by 
  sorry

end john_total_cost_after_discount_l297_297606


namespace min_value_f_a_eq_one_if_f_nonneg_harmonic_series_log_l297_297328

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1

-- 1. Show that the minimum value of f(x) is a - a * ln a - 1
theorem min_value_f (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, f x a = a - a * Real.log a - 1 :=
sorry

-- 2. Show that a = 1 if f(x) ≥ 0 for all x ∈ ℝ
theorem a_eq_one_if_f_nonneg (h : ∀ x : ℝ, f x a ≥ 0) : a = 1 :=
sorry

-- 3. Prove 1 + 1/2 + 1/3 + ... + 1/n > ln(n+1) for n ∈ ℕ* under the condition a = 1
theorem harmonic_series_log (n : ℕ) (hn : 1 ≤ n):
  (∑ i in Finset.range n, 1 / (↑i + 1 : ℝ)) > Real.log (n + 1) :=
sorry

end min_value_f_a_eq_one_if_f_nonneg_harmonic_series_log_l297_297328


namespace count_divisible_by_5_of_ababc_l297_297123

theorem count_divisible_by_5_of_ababc :
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  (a_vals.card * b_vals.card * c_vals.card) = 180 :=
by
  let a_vals : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let b_vals : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let c_vals : Finset ℕ := {0, 5}
  have h_a_card : a_vals.card = 9 := by simp
  have h_b_card : b_vals.card = 10 := by simp
  have h_c_card : c_vals.card = 2 := by simp
  have total := h_a_card * h_b_card * h_c_card
  show total = 180 from sorry

end count_divisible_by_5_of_ababc_l297_297123


namespace binary_multiplication_correct_l297_297262

theorem binary_multiplication_correct :
  nat.of_digits 2 [1, 0, 1, 0, 0, 1, 1] = 
  (nat.of_digits 2 [1, 1, 0, 1] * nat.of_digits 2 [1, 1, 1]) :=
by
  sorry

end binary_multiplication_correct_l297_297262


namespace perfect_square_polynomial_l297_297947

-- Define the polynomial and the conditions
def polynomial (a b : ℚ) := fun x : ℚ => x^4 + x^3 + 2 * x^2 + a * x + b

-- The expanded form of a quadratic trinomial squared
def quadratic_square (p q : ℚ) := fun x : ℚ =>
  x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- Main theorem statement
theorem perfect_square_polynomial :
  ∃ (a b : ℚ), 
  (∀ x : ℚ, polynomial a b x = (quadratic_square (1/2 : ℚ) (7/8 : ℚ) x)) ↔ 
  a = 7/8 ∧ b = 49/64 :=
by
  sorry

end perfect_square_polynomial_l297_297947


namespace tan_3x_domain_l297_297901

def domain_of_tan_3x {x : ℝ} (k : ℤ) : Prop :=
  x ≠ (π / 6) + (k * π / 3)

theorem tan_3x_domain :
  ∀ x : ℝ, ∀ k : ℤ, domain_of_tan_3x k x ↔ x ∉ { x | ∃ k : ℤ, x = (π / 6) + (k * π / 3) } := by
    sorry

end tan_3x_domain_l297_297901


namespace binary_mul_correct_l297_297264

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l297_297264


namespace compute_expression_l297_297306

theorem compute_expression (x y : ℝ) (hx : 1/x + 1/y = 4) (hy : x*y + x + y = 5) : 
  x^2 * y + x * y^2 + x^2 + y^2 = 18 := 
by 
  -- Proof goes here 
  sorry

end compute_expression_l297_297306


namespace max_C_inequalities_l297_297255

theorem max_C_inequalities :
  ∀ C, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ C * (x + y)) ∧
       (∀ x y : ℝ, x^2 + y^2 + x * y + 1 ≥ C * (x + y)) ↔ C ≤ Real.sqrt 2 :=
begin
  sorry
end

end max_C_inequalities_l297_297255


namespace domain_of_f_intervals_of_increase_and_decrease_max_and_min_values_on_interval_l297_297025

noncomputable def f (x : ℝ) := real.sqrt (-x^2 + 5*x + 6)

theorem domain_of_f :
  {x | f x ≥ 0} = set.Icc (-1) 6 :=
begin
  sorry
end

theorem intervals_of_increase_and_decrease :
  {x | ∃ (l u : ℝ), l ≤ x ∧ x ≤ u ∧ (∀ y, l < y ∧ y < x → f y < f x) ∧ (∀ z, x < z ∧ z < u → f z > f x)} =
  set.Icc (-1) (5 / 2) ∪ set.Icc (5 / 2) 6 :=
begin
  sorry
end

theorem max_and_min_values_on_interval :
  ∃ (max_x min_x : ℝ),
    (1 ≤ max_x ∧ max_x ≤ 5 ∧ ∀ y, 1 ≤ y ∧ y ≤ 5 → f y ≤ f max_x)
    ∧
    (1 ≤ min_x ∧ min_x ≤ 5 ∧ ∀ z, 1 ≤ z ∧ z ≤ 5 → f z ≥ f min_x)
    ∧
    f max_x = 7 / 2
    ∧
    f min_x = real.sqrt 6 :=
begin
  sorry
end

end domain_of_f_intervals_of_increase_and_decrease_max_and_min_values_on_interval_l297_297025


namespace line_l_and_line_a_non_intersecting_l297_297762

-- Definitions for the conditions
variables {α : Type} [plane α] {l a : Type} [line l] [line a]

-- Hypotheses
variables (h1 : parallel_line_to_plane l α)
variables (h2 : contained_in_plane a α)

-- Theorem statement
theorem line_l_and_line_a_non_intersecting :
  ¬ ∃ p, point_on_line p l ∧ point_on_line p a :=
sorry

end line_l_and_line_a_non_intersecting_l297_297762


namespace roses_distribution_l297_297490

theorem roses_distribution (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) : 
  initial_roses = 40 → 
  stolen_roses = 4 → 
  people = 9 → 
  (initial_roses - stolen_roses) / people = 4 :=
by
  intros h_initial_roses h_stolen_roses h_people
  rw [h_initial_roses, h_stolen_roses, h_people]
  norm_num
  sorry

end roses_distribution_l297_297490


namespace volume_proof_l297_297167

-- Definitions based on conditions
def base_area : ℝ := 72  -- base area in square centimeters
def height : ℝ := 6      -- height in centimeters

-- Definitions based on questions
def volume_cylinder (B : ℝ) (h : ℝ) : ℝ := B * h
def volume_cone (V_cylinder : ℝ) : ℝ := V_cylinder / 3

-- The theorem to prove the volumes based on given conditions
theorem volume_proof :
  volume_cylinder base_area height = 432 ∧ volume_cone (volume_cylinder base_area height) = 144 :=
by
  sorry

end volume_proof_l297_297167


namespace area_of_transformed_triangle_l297_297505

def g (x : ℝ) : ℝ := sorry  -- assuming g is some function over real numbers

variables (u1 u2 u3 : ℝ)
variable (A : ℝ)  -- initial area

-- Condition: The initial area of the triangle formed by points (u1, g(u1)), (u2, g(u2)), (u3, g(u3)) is 50.
axiom initial_area : A = 50

-- The problem statement: Prove that the area of the triangle formed by (4*u1, 3*g(u1)), (4*u2, 3*g(u2)), (4*u3, 3*g(u3)) is 600.
theorem area_of_transformed_triangle : 
  let area := 4 * 3 * A in
  area = 600 :=
by
  -- Here you would provide the proof but we replace it with sorry as per instructions.
  sorry


end area_of_transformed_triangle_l297_297505


namespace probability_at_least_one_die_shows_three_l297_297551

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l297_297551


namespace points_per_touchdown_l297_297078

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l297_297078


namespace minimum_moves_to_reset_counters_l297_297928

-- Definitions
def counter_in_initial_range (c : ℕ) := 1 ≤ c ∧ c ≤ 2017
def valid_move (decrements : ℕ) (counters : list ℕ) : list ℕ :=
  counters.map (λ c, if c ≥ decrements then c - decrements else c)
def all_counters_zero (counters : list ℕ) : Prop :=
  counters.all (λ c, c = 0)

-- Problem statement
theorem minimum_moves_to_reset_counters :
  ∀ (counters : list ℕ)
  (h : counters.length = 28)
  (h' : ∀ c ∈ counters, counter_in_initial_range c),
  ∃ (moves : ℕ), moves = 11 ∧
    ∀ (f : ℕ → list ℕ → list ℕ)
    (hm : ∀ ds cs, ds > 0 → cs.length = 28 → 
           (∀ c ∈ cs, counter_in_initial_range c) →
           ds ≤ 2017 → f ds cs = valid_move ds cs),
    all_counters_zero (nat.iterate (f (λ m cs, valid_move m cs)) 11 counters) :=
sorry

end minimum_moves_to_reset_counters_l297_297928


namespace members_play_both_l297_297397

theorem members_play_both (N B T Neither B_inter_T : ℕ) 
  (hN : N = 35) 
  (hB : B = 15) 
  (hT : T = 18) 
  (hNeither : Neither = 5) 
  (hN_eq : N = B + T - B_inter_T + Neither) : 
  B_inter_T = 3 := 
by {
  -- We assume all the conditions,
  have h1 : 35 = 15 + 18 - B_inter_T + 5 := by rw [hN, hB, hT, hNeither],
  -- Simplify the expression to find B_inter_T,
  linarith,
}

end members_play_both_l297_297397


namespace arithmetic_sequence_sum_l297_297528

theorem arithmetic_sequence_sum (x y z d : ℤ)
  (h₀ : d = 10 - 3)
  (h₁ : 10 = 3 + d)
  (h₂ : 17 = 10 + d)
  (h₃ : x = 17 + d)
  (h₄ : y = x + d)
  (h₅ : 31 = y + d)
  (h₆ : z = 31 + d) :
  x + y + z = 93 := by
sorry

end arithmetic_sequence_sum_l297_297528


namespace number_of_pizzas_l297_297181

theorem number_of_pizzas (n : ℕ) (h : n = 8) : 
  ∑ k in {1, 2, 3}, Nat.choose n k = 92 := by
  sorry

end number_of_pizzas_l297_297181


namespace parabola_equation_1_parabola_equation_2_l297_297161

noncomputable def parabola_vertex_focus (vertex focus : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, (focus.1 = p / 2 ∧ focus.2 = 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 24 * x)

noncomputable def standard_parabola_through_point (point : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, ( ( point.1^2 = 2 * p * point.2 ∧ point.2 ≠ 0 ∧ point.1 ≠ 0) ∧ (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y / 2) ) ∨
           ( ( point.2^2 = 2 * p * point.1 ∧ point.1 ≠ 0 ∧ point.2 ≠ 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) )

theorem parabola_equation_1 : parabola_vertex_focus (0, 0) (6, 0) := 
  sorry

theorem parabola_equation_2 : standard_parabola_through_point (1, 2) := 
  sorry

end parabola_equation_1_parabola_equation_2_l297_297161


namespace stratified_sampling_second_grade_survey_l297_297973

theorem stratified_sampling_second_grade_survey :
  let total_students := 1500
  let ratio_first := 4
  let ratio_second := 5
  let ratio_third := 6
  let survey_total := 150
  let total_ratio_parts := ratio_first + ratio_second + ratio_third
  let fraction_second := ratio_second.toReal / total_ratio_parts.toReal
  survey_total * fraction_second = 50 := 
by
  sorry

end stratified_sampling_second_grade_survey_l297_297973


namespace stratified_sampling_second_grade_survey_l297_297974

theorem stratified_sampling_second_grade_survey :
  let total_students := 1500
  let ratio_first := 4
  let ratio_second := 5
  let ratio_third := 6
  let survey_total := 150
  let total_ratio_parts := ratio_first + ratio_second + ratio_third
  let fraction_second := ratio_second.toReal / total_ratio_parts.toReal
  survey_total * fraction_second = 50 := 
by
  sorry

end stratified_sampling_second_grade_survey_l297_297974


namespace game_probability_l297_297432

theorem game_probability :
  let John_initial := 3
  let Emma_initial := 2
  let Lucas_initial := 1
  let total_rounds := 1002
  let probability_distribution := λ (r : ℕ) => (3 - (r % 3), 2 - ((r / 3) % 2), 1 - ((r/6) % 1))
  ∀ round, round = total_rounds → probability_distribution round = (2, 2, 2) ↔ (1 / 27) := 
by
  sorry

end game_probability_l297_297432


namespace simplify_and_evaluate_l297_297877

-- Define the condition
def x : ℝ := Real.sqrt 2 + 1

-- State the problem as a theorem
theorem simplify_and_evaluate : (1 - (1 / (x + 1))) * ((x^2 - 1) / x) = Real.sqrt 2 := 
by
  sorry

end simplify_and_evaluate_l297_297877


namespace xiaohua_ran_distance_l297_297567

theorem xiaohua_ran_distance :
  ∀ (length width : ℕ), 
    length = 55 →
    width = 35 →
    let perimeter := 2 * (length + width) in 
    let distance := 2 * perimeter in
    distance = 360 :=
by
  intros length width hlength hwidth
  let perimeter := 2 * (length + width)
  let distance := 2 * perimeter
  have h1 : perimeter = 2 * (55 + 35) := by rw [hlength, hwidth]
  have h2 : distance = 2 * perimeter := rfl
  simp [h1, h2]
  exact rfl

end xiaohua_ran_distance_l297_297567


namespace domain_f_correct_domain_g_correct_l297_297681

noncomputable def domain_f : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ x ≠ 1}

noncomputable def expected_domain_f : Set ℝ :=
  {x | (-1 ≤ x ∧ x < 1) ∨ x > 1}

theorem domain_f_correct :
  domain_f = expected_domain_f :=
by
  sorry

noncomputable def domain_g : Set ℝ :=
  {x | 3 - 4 * x > 0}

noncomputable def expected_domain_g : Set ℝ :=
  {x | x < 3 / 4}

theorem domain_g_correct :
  domain_g = expected_domain_g :=
by
  sorry

end domain_f_correct_domain_g_correct_l297_297681


namespace infinite_triangle_area_sum_l297_297446

noncomputable def rectangle_area_sum : ℝ :=
  let AB := 2
  let BC := 1
  let Q₁ := 0.5
  let base_area := (1/2) * Q₁ * (1/4)
  base_area * (1/(1 - 1/4))

theorem infinite_triangle_area_sum :
  rectangle_area_sum = 1/12 :=
by
  sorry

end infinite_triangle_area_sum_l297_297446


namespace find_prices_compare_costs_l297_297588

-- Step 1: Define variables and price relationships
variables (x : ℝ) -- price of each football
def uniform_price : ℝ := x + 60 -- price of each set of uniforms

-- Step 2: Given the condition, solve for x and uniform_price
theorem find_prices (h : 3 * uniform_price = 5 * x) : x = 90 ∧ uniform_price = 150 :=
by
  have h₁ : 3 * (x + 60) = 5 * x := h
  calc 
    3 * (x + 60) = 5 * x : by rw h
    3 * x + 180 = 5 * x : by ring
    180 = 2 * x : by linarith
    x = 90 : by linarith
  have uniform_price := x + 60
  show uniform_price = 150, from sorry

-- Step 2: Express costs for Market A and Market B
variables (y : ℝ) -- number of footballs
variable (hy : y > 10)

def cost_market_A : ℝ := 100 * 150 + 90 * (y - 10)
def cost_market_B : ℝ := 100 * 150 + 72 * y

-- Step 3: Compare costs
theorem compare_costs (hy : y > 10) : 
  cost_market_A = cost_market_B ↔ y = 50 ∧ 
  (∀ y < 50, cost_market_A < cost_market_B) ∧ 
  (∀ y > 50, cost_market_A > cost_market_B) :=
by
  have h₁ : cost_market_A = 15000 + 90 * y - 900 := sorry
  have h₂ : cost_market_A = 14100 + 90 * y := sorry
  have h₃ : cost_market_B = 15000 + 72 * y := sorry
  sorry

end find_prices_compare_costs_l297_297588


namespace binary_101_eq_5_l297_297651

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l297_297651


namespace sum_of_series_l297_297827

noncomputable def x : ℂ := sorry -- Assuming x is given or found
def N : ℕ := 2021
def M : ℕ := 2013

-- Conditions
axiom x_power_N_eq_one : x^N = 1
axiom x_ne_one : x ≠ 1

-- The sum expression
def sum_expr : ℂ := ∑ k in Finset.range (M + 1) \ {0}, x^(3*k) / (x^k - 1)

-- Goal to prove
theorem sum_of_series : sum_expr = 1005.5 := sorry

end sum_of_series_l297_297827


namespace Ram_has_amount_l297_297527

theorem Ram_has_amount (R G K : ℕ)
    (h1 : R = 7 * G / 17)
    (h2 : G = 7 * K / 17)
    (h3 : K = 3757) : R = 637 := by
  sorry

end Ram_has_amount_l297_297527


namespace triangle_area_parabola_l297_297226

-- Define the statement
theorem triangle_area_parabola (O A B : ℝ × ℝ) :
  let parabola := λ y : ℝ, 4 * (y^2 / 4)
  let line := λ x : ℝ, x - 1
  let intersection_pts := (set_of (λ pt : ℝ × ℝ, parabola pt.2 = pt.1 ∧ line pt.1 = pt.2))
  let sum_roots_eq := (y₁ y₂ : ℝ) → y₁ + y₂ = -4
  let prod_roots_eq := (y₁ y₂ : ℝ) → y₁ * y₂ = -4
  let dist_y := |y₁ - y₂| = 4 * real.sqrt 2
  O = (0,0) ∧
  (1,0) ∈ intersection_pts ∧
  dist_y → 
  (1 / 2) * real.abs (O.1) * dist_y = 2 * real.sqrt 2 :=
begin
  sorry
end

end triangle_area_parabola_l297_297226


namespace range_of_x0_l297_297833

def f (x : ℝ) : ℝ := if x ≥ 1 then 2*x + 1 else x^2 - 2*x - 2

theorem range_of_x0 (x0 : ℝ) (h : f x0 > 1) : x0 ∈ (-∞, -1) ∪ [1, +∞) := 
by {
    sorry
}

end range_of_x0_l297_297833


namespace monotonic_decreasing_interval_l297_297522

noncomputable def f (x : ℝ) : ℝ := x - 2 * log (2 * x)

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (f' x < 0) := by
  sorry

end monotonic_decreasing_interval_l297_297522


namespace plane_equation_correct_l297_297185

noncomputable def parametric_plane : ℝ × ℝ → ℝ × ℝ × ℝ :=
λ (s t : ℝ), (2 + 2 * s - t, 4 - 2 * s, 5 - 3 * s + 3 * t)

theorem plane_equation_correct (x y z : ℝ) (s t : ℝ):
  parametric_plane s t = (x, y, z) →
  2 * x + y - z - 3 = 0 :=
by
  sorry

end plane_equation_correct_l297_297185


namespace root_monotonicity_l297_297717

noncomputable def f (x : ℝ) := 3^x + 2 / (1 - x)

theorem root_monotonicity
  (x0 : ℝ) (H_root : f x0 = 0)
  (x1 x2 : ℝ) (H1 : x1 > 1) (H2 : x1 < x0) (H3 : x2 > x0) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end root_monotonicity_l297_297717


namespace reciprocal_square_inequality_l297_297696

variable (x y : ℝ)
variable (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≤ y)

theorem reciprocal_square_inequality :
  (1 / y^2) ≤ (1 / x^2) :=
sorry

end reciprocal_square_inequality_l297_297696


namespace probability_challenge_l297_297617

-- Define the conditions as well-defined events and properties in Lean
noncomputable def probability_farm (chosen_letters : Finset Char) : ℚ := 
  if chosen_letters = {'A', 'L', _} then 1 / 2 else 0

noncomputable def probability_benches (chosen_letters : Finset Char) : ℚ := 
  if chosen_letters = {'C', 'H', 'E', _} then 4 / 35 else 0

noncomputable def probability_glove (chosen_letters : Finset Char) : ℚ := 
  if chosen_letters = {'G', 'E'} then 1 / 10 else 0

-- State the theorem
theorem probability_challenge :
  (∀ fl fb fg, (fl ⊆ {'F', 'A', 'R', 'M'}) → (fb ⊆ {'B', 'E', 'N', 'C', 'H', 'E', 'S'}) → (fg ⊆ {'G', 'L', 'O', 'V', 'E'}) → 
    |fl| = 3 → |fb| = 4 → |fg| = 2 →
    probability_farm fl * probability_benches fb * probability_glove fg = 2 / 350) :=
sorry

end probability_challenge_l297_297617


namespace present_age_of_son_l297_297173

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 29) (h2 : M + 2 = 2 * (S + 2)) : S = 27 :=
sorry

end present_age_of_son_l297_297173


namespace piecewise_function_value_l297_297284

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem piecewise_function_value :
  f (f (f (-1))) = Real.pi + 1 :=
by
  sorry

end piecewise_function_value_l297_297284


namespace ab_range_value_l297_297327

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := log (x / (1 - x))

-- Declare the noncomputable context to handle the logarithm function
noncomputable def ab_range (a b : ℝ) : Prop :=
  f(a) + f(b) = 0 ∧ 0 < a ∧ a < b ∧ b < 1 → 0 < a * b ∧ a * b < 1/4

-- Main theorem statement
theorem ab_range_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : f(a) + f(b) = 0) :
  0 < a * b ∧ a * b < 1/4 :=
sorry

end ab_range_value_l297_297327


namespace find_prices_compare_costs_l297_297589

-- Step 1: Define variables and price relationships
variables (x : ℝ) -- price of each football
def uniform_price : ℝ := x + 60 -- price of each set of uniforms

-- Step 2: Given the condition, solve for x and uniform_price
theorem find_prices (h : 3 * uniform_price = 5 * x) : x = 90 ∧ uniform_price = 150 :=
by
  have h₁ : 3 * (x + 60) = 5 * x := h
  calc 
    3 * (x + 60) = 5 * x : by rw h
    3 * x + 180 = 5 * x : by ring
    180 = 2 * x : by linarith
    x = 90 : by linarith
  have uniform_price := x + 60
  show uniform_price = 150, from sorry

-- Step 2: Express costs for Market A and Market B
variables (y : ℝ) -- number of footballs
variable (hy : y > 10)

def cost_market_A : ℝ := 100 * 150 + 90 * (y - 10)
def cost_market_B : ℝ := 100 * 150 + 72 * y

-- Step 3: Compare costs
theorem compare_costs (hy : y > 10) : 
  cost_market_A = cost_market_B ↔ y = 50 ∧ 
  (∀ y < 50, cost_market_A < cost_market_B) ∧ 
  (∀ y > 50, cost_market_A > cost_market_B) :=
by
  have h₁ : cost_market_A = 15000 + 90 * y - 900 := sorry
  have h₂ : cost_market_A = 14100 + 90 * y := sorry
  have h₃ : cost_market_B = 15000 + 72 * y := sorry
  sorry

end find_prices_compare_costs_l297_297589


namespace log_x_125_solution_l297_297753

theorem log_x_125_solution (x : ℝ) (h : log 6 (5 * x) = 3) : log x 125 = 3 * log 5 / (2 * log 6 - log 5) :=
sorry

end log_x_125_solution_l297_297753


namespace compute_4_star_3_l297_297373

def custom_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem compute_4_star_3 : custom_op 4 3 = 13 :=
by
  sorry

end compute_4_star_3_l297_297373


namespace problem_statement_l297_297656

-- Define what it means to be 12-pretty
def is_12_pretty (n : ℕ) : Prop :=
  n > 0 ∧ (∃ k : ℕ, k = 12 ∧ n % k = 0 ∧ Nat.countDivisors n = k ∧ Nat.isSquare n)

-- Define the target sum T and the problem statement
def sum_12_pretty_below_1000 : ℕ :=
  ∑ n in Finset.filter (is_12_pretty) (Finset.range 1000), n

theorem problem_statement : sum_12_pretty_below_1000 / 12 = 405 :=
  sorry

end problem_statement_l297_297656


namespace cardinality_of_A_int_l297_297829

noncomputable def A : Set ℝ := {x | x^2 - 2 * x ≤ 0}

noncomputable def A_int : Set ℤ := { n : ℤ | (n : ℝ) ∈ A }

theorem cardinality_of_A_int : A_int.to_finset.card = 3 := 
by
  sorry

end cardinality_of_A_int_l297_297829


namespace solve_log_equation_l297_297065

theorem solve_log_equation (x : ℝ) (h₁ : log 2 (3 * x - 4) = 1) : x = 2 :=
by
  sorry

end solve_log_equation_l297_297065


namespace number_of_oxen_c_put_for_grazing_l297_297957

theorem number_of_oxen_c_put_for_grazing
  (oxen_a : ℕ) (months_a : ℕ)
  (oxen_b : ℕ) (months_b : ℕ)
  (months_c : ℕ)
  (total_rent : ℚ)
  (share_c : ℚ) :
  oxen_a = 10 → months_a = 7 →
  oxen_b = 12 → months_b = 5 →
  months_c = 3 →
  total_rent = 210 →
  share_c = 53.99999999999999 →
  (15 = (130 / 8.666666666666667)) :=
begin
  sorry,
end

end number_of_oxen_c_put_for_grazing_l297_297957


namespace sum_of_ages_l297_297080

theorem sum_of_ages (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 9) 
  (h2 : 1 ≤ b ∧ b ≤ 9) 
  (h3 : 1 ≤ c ∧ c ≤ 9) 
  (h4 : 1 ≤ d ∧ d ≤ 9) 
  (h5 : 1 ≤ e ∧ e ≤ 9) 
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a * b = 28 ∨ a * c = 28 ∨ a * d = 28 ∨ a * e = 28 ∨ b * c = 28 ∨ b * d = 28 ∨ b * e = 28 ∨ c * d = 28 ∨ c * e = 28 ∨ d * e = 28)
  (h8 : a * b = 20 ∨ a * c = 20 ∨ a * d = 20 ∨ a * e = 20 ∨ b * c = 20 ∨ b * d = 20 ∨ b * e = 20 ∨ c * d = 20 ∨ c * e = 20 ∨ d * e = 20)
  (h9 : a + b = 14 ∨ a + c = 14 ∨ a + d = 14 ∨ a + e = 14 ∨ b + c = 14 ∨ b + d = 14 ∨ b + e = 14 ∨ c + d = 14 ∨ c + e = 14 ∨ d + e = 14) 
  : a + b + c + d + e = 25 :=
by
  sorry

end sum_of_ages_l297_297080


namespace Roselyn_books_left_l297_297058

variable (booksRebecca booksInitially : ℕ)
variable (booksMara : ℕ := 3 * booksRebecca)

theorem Roselyn_books_left (hRebecca : booksRebecca = 40) (hInitially : booksInitially = 220) :
  booksInitially - (booksRebecca + booksMara) = 60 :=
by
  -- Provide initial conditions
  have hMara : booksMara = 3 * booksRebecca := rfl

  -- Substitute the known values
  subst hRebecca
  subst hInitially

  -- Calculate and simplify the expression
  have hTotalGiven : booksRebecca + booksMara = 160 := by
    rw [hMara]
    norm_num

  rw [← hTotalGiven]
  norm_num

end Roselyn_books_left_l297_297058


namespace odd_factors_252_l297_297358

theorem odd_factors_252 : 
  {n : ℕ | n ∈ finset.filter (λ d, d % 2 = 1) (finset.divisors 252)}.card = 6 := 
sorry

end odd_factors_252_l297_297358


namespace knights_probability_l297_297934

theorem knights_probability :
  let total_ways := Nat.choose 20 4 in
  let non_adjacent_ways := 20 * 17 * 15 * 13 in
  let probability := non_adjacent_ways / total_ways in
  let simplest_fraction := Nat.gcd_ratio non_adjacent_ways total_ways in
  probability = simplest_fraction ∧ simplest_fraction = 60 / 7 ∧ (60 + 7 = 67) :=
by
  let total_ways := Nat.choose 20 4
  let non_adjacent_ways := 20 * 17 * 15 * 13
  let probability := non_adjacent_ways / total_ways
  let simplest_fraction := Nat.gcd_ratio non_adjacent_ways total_ways
  have h1 : probability = simplest_fraction := sorry
  have h2 : simplest_fraction = 60 / 7 := sorry
  have h3 : 60 + 7 = 67 := rfl
  exact ⟨h1, h2, h3⟩

end knights_probability_l297_297934


namespace suitable_algorithm_for_conditional_structure_l297_297135

theorem suitable_algorithm_for_conditional_structure (a b : ℝ) (h : a ≠ 0) : 
  ∃ alg : (ℝ → ℝ → Prop), alg = (λ a b, if a > 0 then ax + b > 0 else if a < 0 then ax + b > 0 else false) :=
sorry

end suitable_algorithm_for_conditional_structure_l297_297135


namespace tip_is_24_l297_297072

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end tip_is_24_l297_297072


namespace kyle_sales_money_proof_l297_297441

variable (initial_cookies initial_brownies : Nat)
variable (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
variable (price_per_cookie price_per_brownie : Float)

def kyle_total_money (initial_cookies initial_brownies : Nat) 
    (kyle_eats_cookies mom_eats_cookies kyle_eats_brownies mom_eats_brownies : Nat)
    (price_per_cookie price_per_brownie : Float) : Float := 
  let remaining_cookies := initial_cookies - (kyle_eats_cookies + mom_eats_cookies)
  let remaining_brownies := initial_brownies - (kyle_eats_brownies + mom_eats_brownies)
  let money_from_cookies := remaining_cookies * price_per_cookie
  let money_from_brownies := remaining_brownies * price_per_brownie
  money_from_cookies + money_from_brownies

theorem kyle_sales_money_proof : 
  kyle_total_money 60 32 2 1 2 2 1 1.50 = 99 :=
by
  sorry

end kyle_sales_money_proof_l297_297441


namespace touchdowns_points_l297_297075

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l297_297075


namespace equal_angles_of_incenters_l297_297404

open Nat
open Real

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle (A B C : Point) : Real := sorry
noncomputable def isRightTriangle (A B C : Point) : Prop := ∠A = 90
noncomputable def isAltitude (C D AB : Point) : Prop := sorry

theorem equal_angles_of_incenters
  (A B C D O1 O2 : Point)
  (h_right : isRightTriangle B C A)
  (h_altitude : isAltitude C D AB)
  (h_o1 : O1 = incenter A C D)
  (h_o2 : O2 = incenter B C D) :
  angle A O2 C = angle B O1 C :=
sorry

end equal_angles_of_incenters_l297_297404


namespace time_taken_by_abc_l297_297955

-- Define the work rates for a, b, and c
def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 1 / 41.25

-- Define the combined work rate for a, b, and c
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

-- Define the reciprocal of the combined work rate, which is the time taken
def time_taken : ℚ := 1 / combined_work_rate

-- Prove that the time taken by a, b, and c together is 11 days
theorem time_taken_by_abc : time_taken = 11 := by
  -- Substitute the values to compute the result
  sorry

end time_taken_by_abc_l297_297955


namespace count_ababc_divisible_by_5_l297_297121

theorem count_ababc_divisible_by_5 : 
  let a_vals := {a | 1 ≤ a ∧ a ≤ 9},
      b_vals := {b | 0 ≤ b ∧ b ≤ 9},
      c_vals := {c | c = 0 ∨ c = 5} in
  (∑ a in a_vals, ∑ b in b_vals, ∑ c in c_vals, 1) = 180 := 
by {
  sorry
}

end count_ababc_divisible_by_5_l297_297121


namespace petya_wins_iff_odd_l297_297625

-- Define the conditions of the game first

-- Assume n is a natural number representing half the number of vertices
variable (n : ℕ)

-- The definition of the problem, asserting that Petya wins if and only if n is odd
theorem petya_wins_iff_odd (n : ℕ) : 
  (∀ actions_of_vasya, petya_always_has_a_winning_strategy n actions_of_vasya) ↔ (odd n) := 
sorry

end petya_wins_iff_odd_l297_297625


namespace sin_alpha_plus_beta_minimum_OC_l297_297802

open Real

noncomputable theory

def cos_alpha : ℚ := 3 / 5
def sin_alpha : ℚ := 4 / 5
def sin_beta : ℚ := 12 / 13
def cos_beta : ℚ := 5 / 13
def AB : ℚ := 3 / 2
def cos_angle_AOB : ℚ := -1 / 8

theorem sin_alpha_plus_beta : 
  sin (Real.arcsin (sin_alpha.to_real) + Real.arccos (cos_beta.to_real)) = 56 / 65 := 
sorry

theorem minimum_OC (a : ℚ) :
  ∃ a, (a - 1 / 8)^2 + 63 / 64 = (min_oc^2) :=
sorry

end sin_alpha_plus_beta_minimum_OC_l297_297802


namespace monotonicity_and_extreme_values_l297_297033

noncomputable def f (x : ℝ) : ℝ := log (2 * x + 3) + x^2

theorem monotonicity_and_extreme_values :
  (∀ x ∈ Ioo (-3/2) (-1), deriv f x > 0) ∧
  (∀ x ∈ Ioo (-1) (-1/2), deriv f x < 0) ∧
  (∀ x ∈ Ioi (-1/2), deriv f x > 0) ∧
  (∀ x ∈ Icc (-3/4) (1/4), f x ≥ log 2 + 1/4) ∧
  (∀ x ∈ Icc (-3/4) (1/4), f x ≤ 1/16 + log (7/2)) :=
by
  sorry

end monotonicity_and_extreme_values_l297_297033


namespace possible_values_of_reciprocal_sum_l297_297812

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  ∃ x, x ∈ set.Ici (2:ℝ) ∧ x = (1 / a + 1 / b) :=
by sorry

end possible_values_of_reciprocal_sum_l297_297812


namespace petya_wins_iff_odd_l297_297626

-- Define the conditions of the game first

-- Assume n is a natural number representing half the number of vertices
variable (n : ℕ)

-- The definition of the problem, asserting that Petya wins if and only if n is odd
theorem petya_wins_iff_odd (n : ℕ) : 
  (∀ actions_of_vasya, petya_always_has_a_winning_strategy n actions_of_vasya) ↔ (odd n) := 
sorry

end petya_wins_iff_odd_l297_297626


namespace not_associative_star_l297_297278

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

theorem not_associative_star : ¬ (∀ x y z : ℝ, star (star x y) z = star x (star y z)) :=
by
  sorry

end not_associative_star_l297_297278


namespace pentagon_perimeter_sum_l297_297179

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def pentagon_points : list (ℝ × ℝ) :=
  [(0,0), (1,2), (3,3), (4,1), (2,0)]

def perimeter : ℝ :=
  list.sum (list.map (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), distance p.1.1 p.1.2 p.2.1 p.2.2)
    (list.zip pentagon_points (pentagon_points.tail ++ [pentagon_points.head])))

theorem pentagon_perimeter_sum : ∃ (a b c d e : ℕ), 
  (perimeter = a + b * real.sqrt c + d * real.sqrt e) ∧ (a + b + c + d + e = 11) :=
sorry

end pentagon_perimeter_sum_l297_297179


namespace distance_between_A_and_B_l297_297315

theorem distance_between_A_and_B (a b : ℝ) (theta : ℝ) (dist_AC : a = 10) (dist_BC : b = 15) (angle_A : theta = 25 * (Real.pi / 180)) (angle_B : theta + 35 * (Real.pi / 180) = 60 * (Real.pi / 180)) :
  dist (a, b, theta) = 5 * Real.sqrt 7 := 
sorry

end distance_between_A_and_B_l297_297315


namespace num_odd_factors_of_252_l297_297352

theorem num_odd_factors_of_252 : 
  ∃ n : ℕ, n = 252 ∧ 
  ∃ k : ℕ, (k = ∏ d in (divisors_filter (λ x, x % 2 = 1) n), 1) 
  ∧ k = 6 := 
sorry

end num_odd_factors_of_252_l297_297352


namespace line_point_relation_l297_297407

-- Define the points and equations
def polar_point_A := (real.sqrt 2, real.pi / 4)

def polar_eq_of_line (ρ θ : ℝ) (a : ℝ) := ρ * real.cos (θ - real.pi / 4) = a

-- Points A lies on line l
axiom A_on_l : polar_eq_of_line (real.sqrt 2) (real.pi / 4) a

-- Cartesian equation conversions and the circle definition
def cartesian_eq_of_line (x y a : ℝ) := x + y - a = 0

def parametric_circle (α : ℝ) : (ℝ × ℝ) :=
  (1 + real.cos α, real.sin α)

def cartesian_eq_of_circle (x y : ℝ) := (x - 1)^2 + y^2 = 1

-- The conditions and statements to prove
theorem line_point_relation (a : ℝ) (x y : ℝ) : 
  (polar_point_A = (real.sqrt 2, real.pi / 4)) → 
  (polar_eq_of_line (real.sqrt 2) (real.pi / 4) a) →
  (∀ (α : ℝ), parametric_circle α = (1 + real.cos α, real.sin α)) →
  a = real.sqrt 2 ∧ cartesian_eq_of_line x y 2 = 0 ∧ 
  ∃ α : ℝ, (x, y) = parametric_circle α → 
  cartesian_eq_of_circle x y 
by
  sorry

end line_point_relation_l297_297407


namespace number_of_white_balls_l297_297580

theorem number_of_white_balls (r w : ℕ) (h_r : r = 8) (h_prob : (r : ℚ) / (r + w) = 2 / 5) : w = 12 :=
by sorry

end number_of_white_balls_l297_297580


namespace cos_C_value_l297_297416

theorem cos_C_value (A B C : ℝ) (a b c : ℝ) 
  (hABC : ∀ A B C a b c, ∠A + ∠B + ∠C = π)
  (h : sin A / sin B = 3 / 5 ∧ sin B / sin C = 5 / 7)
  (h₁ : a / sin A = b / sin B ∧ b / sin B = c / sin C):
  cos C = -1/2 :=
begin
  sorry
end

end cos_C_value_l297_297416


namespace find_number_l297_297147

-- Define the conditions from the problem
def is_percentage (percent : ℝ) (x : ℝ) : ℝ := (percent / 100) * x

def condition (x : ℝ) : Prop :=
  let fifty_five_point_five_percent_of_x := is_percentage 65 x in
  let five_percent_of_sixty := is_percentage 5 60 in
  fifty_five_point_five_percent_of_x = five_percent_of_sixty + 23

-- Theorem statement asserting the solution
theorem find_number (x : ℝ) (h : condition x) : x = 40 :=
sorry

end find_number_l297_297147


namespace line_intersects_curve_l297_297100

theorem line_intersects_curve (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ + 16 = x₁^3 ∧ ax₂ + 16 = x₂^3) →
  a = 12 :=
by
  sorry

end line_intersects_curve_l297_297100


namespace intervals_strictly_decreasing_extrema_values_l297_297695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (real.log ((1-x) / (1+x))) - a * x

theorem intervals_strictly_decreasing (a : ℝ) :
  (a ≥ -2 → ∀ x, x ∈ Ioo (-1 : ℝ) 1 → (x < 1) → has_deriv_at (f a) (-(2 / (1 - x^2)) - a) x → (-(2 / (1 - x^2)) - a < 0)) ∧ 
  (a < -2 → ∀ x, x ∈ Ioo ((sqrt ((a+2)/a) : ℝ)) 1 ∨ x ∈ Ioo (-1) (-(sqrt ((a+2)/a) : ℝ)) → has_deriv_at (f a) (-(2 / (1 - x^2)) - a) x → (-(2 / (1 - x^2)) - a < 0)) := 
sorry

theorem extrema_values (a : ℝ) (h : a = -8/3) :
  f a (-1/2) = -4/3 + real.log 3 ∧ 
  f a (1/2) = 4/3 - real.log 3 :=
sorry

end intervals_strictly_decreasing_extrema_values_l297_297695


namespace probability_three_odd_dice_l297_297630

theorem probability_three_odd_dice :
  let p := 1/2 -- Probability of a single die showing an odd number
  let combinations := Nat.choose 5 3 -- Number of ways to choose 3 out of 5 dice
  let prob_exactly_three_odd := combinations * (p^3 * (1-p)^2) -- Combined probability
  in prob_exactly_three_odd = 5/16 :=
by
  sorry

end probability_three_odd_dice_l297_297630


namespace min_moves_to_zero_l297_297929

-- Define the problem setting and conditions

def initial_counters : ℕ := 28
def max_value : ℕ := 2017

-- Definition for the minimum number of moves required to reduce all counters to zero

theorem min_moves_to_zero : 
  ∀ (counters : list ℕ), (∀ c ∈ counters, 1 ≤ c ∧ c ≤ max_value) → counters.length = initial_counters →
  ∃ (m : ℕ), m = 11 ∧ 
    (∀ (f : ℕ → ℕ → ℕ), f 0 0 = 0 → (∃ i, 0 < i ∧ i ≤ m ∧ ∀ n ∈ counters, f i n = 0)) :=
by
  sorry

end min_moves_to_zero_l297_297929


namespace no_integer_solutions_l297_297248

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), (x ≠ 1 ∧ (x^7 - 1) / (x - 1) = (y^5 - 1)) :=
sorry

end no_integer_solutions_l297_297248
