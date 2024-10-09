import Mathlib

namespace vector_sum_magnitude_l1078_107899

variable (a b : EuclideanSpace ℝ (Fin 3)) -- assuming 3-dimensional Euclidean space for vectors

-- Define the conditions
def mag_a : ℝ := 5
def mag_b : ℝ := 6
def dot_prod_ab : ℝ := -6

-- Prove the required magnitude condition
theorem vector_sum_magnitude (ha : ‖a‖ = mag_a) (hb : ‖b‖ = mag_b) (hab : inner a b = dot_prod_ab) :
  ‖a + b‖ = 7 :=
by
  sorry

end vector_sum_magnitude_l1078_107899


namespace fibby_numbers_l1078_107823

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧ ∃ (n : ℕ) (d : ℕ → ℕ),
  (∀ j, 1 ≤ j ∧ j ≤ k - 2 → d (j + 2) = d (j + 1) + d j) ∧
  (∀ (j : ℕ), 1 ≤ j ∧ j ≤ k → d j ∣ n) ∧
  (∀ (m : ℕ), m ∣ n → m < d 1 ∨ m > d k)

theorem fibby_numbers : ∀ (k : ℕ), is_fibby k → k = 3 ∨ k = 4 :=
sorry

end fibby_numbers_l1078_107823


namespace calculation_correct_l1078_107846

noncomputable def problem_calculation : ℝ :=
  4 * Real.sin (Real.pi / 3) - abs (-1) + (Real.sqrt 3 - 1)^0 + Real.sqrt 48

theorem calculation_correct : problem_calculation = 6 * Real.sqrt 3 :=
by
  sorry

end calculation_correct_l1078_107846


namespace least_x_divisible_by_3_l1078_107878

theorem least_x_divisible_by_3 : ∃ x : ℕ, (∀ y : ℕ, (2 + 3 + 5 + 7 + y) % 3 = 0 → y = 1) :=
by
  sorry

end least_x_divisible_by_3_l1078_107878


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l1078_107889

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l1078_107889


namespace range_of_a_l1078_107859

variable {α : Type}

def A (x : ℝ) : Prop := 1 ≤ x ∧ x < 5
def B (x a : ℝ) : Prop := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) → a ≤ -1 := by
  sorry

end range_of_a_l1078_107859


namespace clusters_per_spoonful_l1078_107861

theorem clusters_per_spoonful (spoonfuls_per_bowl : ℕ) (clusters_per_box : ℕ) (bowls_per_box : ℕ) 
  (h_spoonfuls : spoonfuls_per_bowl = 25) 
  (h_clusters : clusters_per_box = 500)
  (h_bowls : bowls_per_box = 5) : 
  clusters_per_box / bowls_per_box / spoonfuls_per_bowl = 4 := 
by 
  have clusters_per_bowl := clusters_per_box / bowls_per_box
  have clusters_per_spoonful := clusters_per_bowl / spoonfuls_per_bowl
  sorry

end clusters_per_spoonful_l1078_107861


namespace inequality_solution_fractional_equation_solution_l1078_107849

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (1 - x) / 3 - x < 3 - (x + 2) / 4 → x > -2 :=
by
  sorry

-- Proof Problem 2
theorem fractional_equation_solution (x : ℝ) : (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) → false :=
by
  sorry

end inequality_solution_fractional_equation_solution_l1078_107849


namespace side_length_square_eq_4_l1078_107851

theorem side_length_square_eq_4 (s : ℝ) (h : s^2 - 3 * s = 4) : s = 4 :=
sorry

end side_length_square_eq_4_l1078_107851


namespace neg_and_eq_or_not_l1078_107800

theorem neg_and_eq_or_not (p q : Prop) : ¬(p ∧ q) ↔ ¬p ∨ ¬q :=
by sorry

end neg_and_eq_or_not_l1078_107800


namespace problem_statement_l1078_107811

theorem problem_statement
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := 
by
  sorry

end problem_statement_l1078_107811


namespace polynomial_factorization_l1078_107883

-- Definitions from conditions
def p (x : ℝ) : ℝ := x^6 - 2 * x^4 + 6 * x^3 + x^2 - 6 * x + 9
def q (x : ℝ) : ℝ := (x^3 - x + 3)^2

-- The theorem statement proving question == answer given conditions
theorem polynomial_factorization : ∀ x : ℝ, p x = q x :=
by
  sorry

end polynomial_factorization_l1078_107883


namespace original_square_side_length_l1078_107836

-- Defining the variables and conditions
variables (x : ℝ) (h₁ : 1.2 * x * (x - 2) = x * x)

-- Theorem statement to prove the side length of the original square is 12 cm
theorem original_square_side_length : x = 12 :=
by
  sorry

end original_square_side_length_l1078_107836


namespace cell_division_50_closest_to_10_15_l1078_107872

theorem cell_division_50_closest_to_10_15 :
  10^14 < 2^50 ∧ 2^50 < 10^16 :=
sorry

end cell_division_50_closest_to_10_15_l1078_107872


namespace total_golf_balls_l1078_107892

theorem total_golf_balls :
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  dan + gus + chris = 132 :=
by
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  sorry

end total_golf_balls_l1078_107892


namespace depth_of_water_l1078_107868

variable (RonHeight DepthOfWater : ℝ)

-- Definitions based on conditions
def RonStandingHeight := 12 -- Ron's height is 12 feet
def DepthOfWaterCalculation := 5 * RonStandingHeight -- Depth is 5 times Ron's height

-- Theorem statement to prove
theorem depth_of_water (hRon : RonHeight = RonStandingHeight) (hDepth : DepthOfWater = DepthOfWaterCalculation) :
  DepthOfWater = 60 := by
  sorry

end depth_of_water_l1078_107868


namespace sum_of_logs_in_acute_triangle_l1078_107812

theorem sum_of_logs_in_acute_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) 
  (h_triangle : A + B + C = π) :
  (Real.log (Real.sin B) / Real.log (Real.sin A)) +
  (Real.log (Real.sin C) / Real.log (Real.sin B)) +
  (Real.log (Real.sin A) / Real.log (Real.sin C)) ≥ 3 := by
  sorry

end sum_of_logs_in_acute_triangle_l1078_107812


namespace probability_digits_different_l1078_107885

theorem probability_digits_different : 
  let total_numbers := 490
  let same_digits_numbers := 13
  let different_digits_numbers := total_numbers - same_digits_numbers 
  let probability := different_digits_numbers / total_numbers 
  probability = 477 / 490 :=
by
  sorry

end probability_digits_different_l1078_107885


namespace toy_truck_cost_is_correct_l1078_107837

-- Define the initial amount, amount spent on the pencil case, and the final amount
def initial_amount : ℝ := 10
def pencil_case_cost : ℝ := 2
def final_amount : ℝ := 5

-- Define the amount spent on the toy truck
def toy_truck_cost : ℝ := initial_amount - pencil_case_cost - final_amount

-- Prove that the amount spent on the toy truck is 3 dollars
theorem toy_truck_cost_is_correct : toy_truck_cost = 3 := by
  sorry

end toy_truck_cost_is_correct_l1078_107837


namespace sum_of_exponents_l1078_107826

-- Define the expression inside the radical
def radicand (a b c : ℝ) : ℝ := 40 * a^6 * b^3 * c^14

-- Define the simplified expression outside the radical
def simplified_expr (a b c : ℝ) : ℝ := (2 * a^2 * b * c^4)

-- State the theorem to prove the sum of the exponents of the variables outside the radical
theorem sum_of_exponents (a b c : ℝ) : 
  let exponents_sum := 2 + 1 + 4
  exponents_sum = 7 :=
by
  sorry

end sum_of_exponents_l1078_107826


namespace fourth_group_students_l1078_107839

theorem fourth_group_students (total_students group1 group2 group3 group4 : ℕ)
  (h_total : total_students = 24)
  (h_group1 : group1 = 5)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 7)
  (h_groups_sum : group1 + group2 + group3 + group4 = total_students) :
  group4 = 4 :=
by
  -- Proof will go here
  sorry

end fourth_group_students_l1078_107839


namespace buoy_radius_l1078_107801

-- Define the conditions based on the given problem
def is_buoy_hole (width : ℝ) (depth : ℝ) : Prop :=
  width = 30 ∧ depth = 10

-- Define the statement to prove the radius of the buoy
theorem buoy_radius : ∀ r x : ℝ, is_buoy_hole 30 10 → (x^2 + 225 = (x + 10)^2) → r = x + 10 → r = 16.25 := by
  intros r x h_cond h_eq h_add
  sorry

end buoy_radius_l1078_107801


namespace john_memory_card_cost_l1078_107896

-- Define conditions
def pictures_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 3
def pictures_per_card : ℕ := 50
def cost_per_card : ℕ := 60

-- Define total days
def total_days (years : ℕ) (days_per_year : ℕ) : ℕ := years * days_per_year

-- Define total pictures
def total_pictures (pictures_per_day : ℕ) (total_days : ℕ) : ℕ := pictures_per_day * total_days

-- Define required cards
def required_cards (total_pictures : ℕ) (pictures_per_card : ℕ) : ℕ :=
  (total_pictures + pictures_per_card - 1) / pictures_per_card  -- ceiling division

-- Define total cost
def total_cost (required_cards : ℕ) (cost_per_card : ℕ) : ℕ := required_cards * cost_per_card

-- Prove the total cost equals $13,140
theorem john_memory_card_cost : total_cost (required_cards (total_pictures pictures_per_day (total_days years days_per_year)) pictures_per_card) cost_per_card = 13140 :=
by
  sorry

end john_memory_card_cost_l1078_107896


namespace final_score_l1078_107828

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l1078_107828


namespace lucille_house_difference_l1078_107860

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end lucille_house_difference_l1078_107860


namespace parallel_planes_transitivity_l1078_107870

-- Define different planes α, β, γ
variables (α β γ : Plane)

-- Define the parallel relation between planes
axiom parallel : Plane → Plane → Prop

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom β_parallel_α : parallel β α
axiom γ_parallel_α : parallel γ α

-- Statement to prove
theorem parallel_planes_transitivity (α β γ : Plane) 
  (h1 : parallel β α) 
  (h2 : parallel γ α) 
  (h3 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : parallel β γ :=
sorry

end parallel_planes_transitivity_l1078_107870


namespace intersection_A_B_l1078_107804

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_A_B_l1078_107804


namespace product_of_solutions_l1078_107897

theorem product_of_solutions (x : ℝ) :
  let a := -2
  let b := -8
  let c := -49
  ∀ x₁ x₂, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) → 
  x₁ * x₂ = 49/2 :=
sorry

end product_of_solutions_l1078_107897


namespace g_neither_even_nor_odd_l1078_107824

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l1078_107824


namespace two_std_dev_less_than_mean_l1078_107822

def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

theorem two_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.0 := 
by sorry

end two_std_dev_less_than_mean_l1078_107822


namespace cyclic_permutations_sum_41234_l1078_107895

theorem cyclic_permutations_sum_41234 :
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  3 * (n1 + n2 + n3 + n4) = 396618 :=
by
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  show 3 * (n1 + n2 + n3 + n4) = 396618
  sorry

end cyclic_permutations_sum_41234_l1078_107895


namespace Tricia_is_five_years_old_l1078_107807

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l1078_107807


namespace least_four_digit_integer_has_3_7_11_as_factors_l1078_107819

theorem least_four_digit_integer_has_3_7_11_as_factors :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧ (3 ∣ x) ∧ (7 ∣ x) ∧ (11 ∣ x) ∧ x = 1155 := by
  sorry

end least_four_digit_integer_has_3_7_11_as_factors_l1078_107819


namespace area_of_rectangle_is_32_proof_l1078_107815

noncomputable def triangle_sides : ℝ := 7.3 + 5.4 + 11.3
def equality_of_perimeters (rectangle_length rectangle_width : ℝ) : Prop := 
  2 * (rectangle_length + rectangle_width) = triangle_sides

def rectangle_length (rectangle_width : ℝ) : ℝ := 2 * rectangle_width

def area_of_rectangle_is_32 (rectangle_width : ℝ) : Prop :=
  rectangle_length rectangle_width * rectangle_width = 32

theorem area_of_rectangle_is_32_proof : 
  ∃ (rectangle_width : ℝ), 
  equality_of_perimeters (rectangle_length rectangle_width) rectangle_width ∧ area_of_rectangle_is_32 rectangle_width :=
by
  sorry

end area_of_rectangle_is_32_proof_l1078_107815


namespace Samuel_fraction_spent_l1078_107834

variable (totalAmount receivedRatio remainingAmount : ℕ)
variable (h1 : totalAmount = 240)
variable (h2 : receivedRatio = 3 / 4)
variable (h3 : remainingAmount = 132)

theorem Samuel_fraction_spent (spend : ℚ) : 
  (spend = (1 / 5)) :=
by
  sorry

end Samuel_fraction_spent_l1078_107834


namespace sunflower_mix_is_50_percent_l1078_107869

-- Define the proportions and percentages given in the problem
def prop_A : ℝ := 0.60 -- 60% of the mix is Brand A
def prop_B : ℝ := 0.40 -- 40% of the mix is Brand B
def sf_A : ℝ := 0.60 -- Brand A is 60% sunflower
def sf_B : ℝ := 0.35 -- Brand B is 35% sunflower

-- Define the final percentage of sunflower in the mix
noncomputable def sunflower_mix_percentage : ℝ :=
  (sf_A * prop_A) + (sf_B * prop_B)

-- Statement to prove that the percentage of sunflower in the mix is 50%
theorem sunflower_mix_is_50_percent : sunflower_mix_percentage = 0.50 :=
by
  sorry

end sunflower_mix_is_50_percent_l1078_107869


namespace find_number_l1078_107825

theorem find_number {x : ℝ} 
  (h : 973 * x - 739 * x = 110305) : 
  x = 471.4 := 
by 
  sorry

end find_number_l1078_107825


namespace percent_area_shaded_l1078_107809

-- Conditions: Square $ABCD$ has a side length of 10, and square $PQRS$ has a side length of 15.
-- The overlap of these squares forms a rectangle $AQRD$ with dimensions $20 \times 25$.

theorem percent_area_shaded 
  (side_ABCD : ℕ := 10) 
  (side_PQRS : ℕ := 15) 
  (dim_AQRD_length : ℕ := 25) 
  (dim_AQRD_width : ℕ := 20) 
  (area_AQRD : ℕ := dim_AQRD_length * dim_AQRD_width)
  (overlap_side : ℕ := 10) 
  (area_shaded : ℕ := overlap_side * overlap_side)
  : (area_shaded * 100) / area_AQRD = 20 := 
by 
  sorry

end percent_area_shaded_l1078_107809


namespace converse_inverse_contrapositive_count_l1078_107845

theorem converse_inverse_contrapositive_count
  (a b : ℝ) : (a = 0 → ab = 0) →
  (if (ab = 0 → a = 0) then 1 else 0) +
  (if (a ≠ 0 → ab ≠ 0) then 1 else 0) +
  (if (ab ≠ 0 → a ≠ 0) then 1 else 0) = 1 :=
sorry

end converse_inverse_contrapositive_count_l1078_107845


namespace quintuplets_babies_l1078_107882

theorem quintuplets_babies (a b c d : ℕ) 
  (h1 : d = 2 * c) 
  (h2 : c = 3 * b) 
  (h3 : b = 2 * a) 
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1200) : 
  5 * d = 18000 / 23 :=
by 
  sorry

end quintuplets_babies_l1078_107882


namespace find_b_for_real_root_l1078_107830

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 + b * x^3 - 2 * x^2 + b * x + 2 = 0

theorem find_b_for_real_root :
  ∀ b : ℝ, polynomial_has_real_root b → b ≤ 0 := by
  sorry

end find_b_for_real_root_l1078_107830


namespace a_1000_value_l1078_107820

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), 
    (a 1 = 2010) ∧
    (a 2 = 2011) ∧
    (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 3) ∧
    (a 1000 = 2676) :=
by {
  -- sorry is used to skip the proof
  sorry 
}

end a_1000_value_l1078_107820


namespace arc_length_of_sector_l1078_107881

theorem arc_length_of_sector (r α : ℝ) (hα : α = Real.pi / 5) (hr : r = 20) : r * α = 4 * Real.pi :=
by
  sorry

end arc_length_of_sector_l1078_107881


namespace right_rectangular_prism_volume_l1078_107866

theorem right_rectangular_prism_volume
    (a b c : ℝ)
    (H1 : a * b = 56)
    (H2 : b * c = 63)
    (H3 : a * c = 72)
    (H4 : c = 3 * a) :
    a * b * c = 2016 * Real.sqrt 6 :=
by
  sorry

end right_rectangular_prism_volume_l1078_107866


namespace inequality_proof_l1078_107813

theorem inequality_proof 
  {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ} (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂ / x₁)^5 + (x₄ / x₂)^5 + (x₆ / x₃)^5 + (x₁ / x₄)^5 + (x₃ / x₅)^5 + (x₅ / x₆)^5 ≥ 
  (x₁ / x₂) + (x₂ / x₄) + (x₃ / x₆) + (x₄ / x₁) + (x₅ / x₃) + (x₆ / x₅) := 
  sorry

end inequality_proof_l1078_107813


namespace xiao_wang_original_plan_l1078_107831

theorem xiao_wang_original_plan (p d1 extra_pages : ℕ) (original_days : ℝ) (x : ℝ) 
  (h1 : p = 200)
  (h2 : d1 = 5)
  (h3 : extra_pages = 5)
  (h4 : original_days = p / x)
  (h5 : original_days - 1 = d1 + (p - (d1 * x)) / (x + extra_pages)) :
  x = 20 := 
  sorry

end xiao_wang_original_plan_l1078_107831


namespace trendy_haircut_cost_l1078_107863

theorem trendy_haircut_cost (T : ℝ) (H1 : 5 * 5 * 7 + 3 * 6 * 7 + 2 * T * 7 = 413) : T = 8 :=
by linarith

end trendy_haircut_cost_l1078_107863


namespace min_fraction_sum_l1078_107874

theorem min_fraction_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (∃ (z : ℝ), z = (1 / (x + 1)) + (4 / (y + 2)) ∧ z = 9 / 4) :=
by 
  sorry

end min_fraction_sum_l1078_107874


namespace fraction_spent_on_sandwich_l1078_107852
    
theorem fraction_spent_on_sandwich 
  (x : ℚ)
  (h1 : 90 * x + 90 * (1/6) + 90 * (1/2) + 12 = 90) : 
  x = 1/5 :=
by
  sorry

end fraction_spent_on_sandwich_l1078_107852


namespace flowchart_output_is_minus_nine_l1078_107879

-- Given initial state and conditions
def initialState : ℤ := 0

-- Hypothetical function representing the sequence of operations in the flowchart
-- (hiding the exact operations since they are speculative)
noncomputable def flowchartOperations (S : ℤ) : ℤ := S - 9  -- Assuming this operation represents the described flowchart

-- The proof problem
theorem flowchart_output_is_minus_nine : flowchartOperations initialState = -9 :=
by
  sorry

end flowchart_output_is_minus_nine_l1078_107879


namespace largest_of_numbers_l1078_107865

theorem largest_of_numbers (a b c d : ℝ) 
  (ha : a = 0) (hb : b = -1) (hc : c = 3.5) (hd : d = Real.sqrt 13) : 
  ∃ x, x = Real.sqrt 13 ∧ (x > a) ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  sorry

end largest_of_numbers_l1078_107865


namespace probability_gather_info_both_workshops_l1078_107898

theorem probability_gather_info_both_workshops :
  ∃ (p : ℚ), p = 56 / 62 :=
by
  sorry

end probability_gather_info_both_workshops_l1078_107898


namespace correct_equation_l1078_107888

variable (x : ℕ)

def three_people_per_cart_and_two_empty_carts (x : ℕ) :=
  x / 3 + 2

def two_people_per_cart_and_nine_walking (x : ℕ) :=
  (x - 9) / 2

theorem correct_equation (x : ℕ) :
  three_people_per_cart_and_two_empty_carts x = two_people_per_cart_and_nine_walking x :=
by
  sorry

end correct_equation_l1078_107888


namespace range_of_y_div_x_l1078_107867

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + (y-3)^2 = 1) : 
  (∃ k : ℝ, k = y / x ∧ (k ≤ -2 * Real.sqrt 2 ∨ k ≥ 2 * Real.sqrt 2)) :=
sorry

end range_of_y_div_x_l1078_107867


namespace ab_value_l1078_107864

theorem ab_value (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b = 33) : a * b = 24 := 
by 
  sorry

end ab_value_l1078_107864


namespace factor_expression_l1078_107884

theorem factor_expression (x : ℝ) :
  4 * x * (x - 5) + 7 * (x - 5) + 12 * (x - 5) = (4 * x + 19) * (x - 5) :=
by
  sorry

end factor_expression_l1078_107884


namespace initial_roses_l1078_107806

theorem initial_roses (x : ℕ) (h1 : x - 3 + 34 = 36) : x = 5 :=
by 
  sorry

end initial_roses_l1078_107806


namespace defective_and_shipped_percent_l1078_107818

def defective_percent : ℝ := 0.05
def shipped_percent : ℝ := 0.04

theorem defective_and_shipped_percent : (defective_percent * shipped_percent) * 100 = 0.2 :=
by
  sorry

end defective_and_shipped_percent_l1078_107818


namespace contractor_initial_people_l1078_107873

theorem contractor_initial_people (P : ℕ) (days_total days_done : ℕ) 
  (percent_done : ℚ) (additional_people : ℕ) (T : ℕ) :
  days_total = 50 →
  days_done = 25 →
  percent_done = 0.4 →
  additional_people = 90 →
  T = P + additional_people →
  (P : ℚ) * 62.5 = (T : ℚ) * 50 →
  P = 360 :=
by
  intros h_days_total h_days_done h_percent_done h_additional_people h_T h_eq
  sorry

end contractor_initial_people_l1078_107873


namespace base_5_representation_l1078_107832

theorem base_5_representation (n : ℕ) (h : n = 84) : 
  ∃ (a b c : ℕ), 
  a < 5 ∧ b < 5 ∧ c < 5 ∧ 
  n = a * 5^2 + b * 5^1 + c * 5^0 ∧ 
  a = 3 ∧ b = 1 ∧ c = 4 :=
by 
  -- Placeholder for the proof
  sorry

end base_5_representation_l1078_107832


namespace series_sum_half_l1078_107840

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l1078_107840


namespace sequence_ratio_l1078_107886

theorem sequence_ratio (S T a b : ℕ → ℚ) (h_sum_ratio : ∀ (n : ℕ), S n / T n = (7*n + 2) / (n + 3)) :
  a 7 / b 7 = 93 / 16 :=
by
  sorry

end sequence_ratio_l1078_107886


namespace percentage_increase_in_savings_l1078_107833

theorem percentage_increase_in_savings (I : ℝ) (hI : 0 < I) :
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  ((S_new - S) / S) * 100 = 50 :=
by
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  sorry

end percentage_increase_in_savings_l1078_107833


namespace simplify_expression_l1078_107844

theorem simplify_expression (x : ℝ) : (5 * x + 2 * x + 7 * x) = 14 * x :=
by
  sorry

end simplify_expression_l1078_107844


namespace angle_measure_l1078_107842

theorem angle_measure : 
  ∃ (x : ℝ), (x + (3 * x + 3) = 90) ∧ x = 21.75 := by
  sorry

end angle_measure_l1078_107842


namespace least_three_digit_with_factors_l1078_107875

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end least_three_digit_with_factors_l1078_107875


namespace total_pages_read_l1078_107890

theorem total_pages_read (days : ℕ)
  (deshaun_books deshaun_pages_per_book lilly_percent ben_extra eva_factor sam_pages_per_day : ℕ)
  (lilly_percent_correct : lilly_percent = 75)
  (ben_extra_correct : ben_extra = 25)
  (eva_factor_correct : eva_factor = 2)
  (total_break_days : days = 80)
  (deshaun_books_correct : deshaun_books = 60)
  (deshaun_pages_per_book_correct : deshaun_pages_per_book = 320)
  (sam_pages_per_day_correct : sam_pages_per_day = 150) :
  deshaun_books * deshaun_pages_per_book +
  (lilly_percent * deshaun_books * deshaun_pages_per_book / 100) +
  (deshaun_books * (100 + ben_extra) / 100) * 280 +
  (eva_factor * (deshaun_books * (100 + ben_extra) / 100 * 280)) +
  (sam_pages_per_day * days) = 108450 := 
sorry

end total_pages_read_l1078_107890


namespace fixed_point_of_function_l1078_107841

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a^(1 - x) - 2

theorem fixed_point_of_function (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = -1 := by
  sorry

end fixed_point_of_function_l1078_107841


namespace circle_area_isosceles_triangle_l1078_107817

theorem circle_area_isosceles_triangle (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 2) :
  ∃ R : ℝ, R = (81 / 32) * Real.pi :=
by sorry

end circle_area_isosceles_triangle_l1078_107817


namespace distance_to_origin_l1078_107808

theorem distance_to_origin (x y : ℤ) (hx : x = -5) (hy : y = 12) :
  Real.sqrt (x^2 + y^2) = 13 := by
  rw [hx, hy]
  norm_num
  sorry

end distance_to_origin_l1078_107808


namespace find_number_of_cows_l1078_107876

-- Definitions for the problem
def number_of_legs (cows chickens : ℕ) := 4 * cows + 2 * chickens
def twice_the_heads_plus_12 (cows chickens : ℕ) := 2 * (cows + chickens) + 12

-- Main statement to prove
theorem find_number_of_cows (h : ℕ) : ∃ c : ℕ, number_of_legs c h = twice_the_heads_plus_12 c h ∧ c = 6 := 
by
  -- Sorry is used as a placeholder for the proof
  sorry

end find_number_of_cows_l1078_107876


namespace value_of_m_solve_system_relationship_x_y_l1078_107858

-- Part 1: Prove the value of m is 1
theorem value_of_m (x : ℝ) (m : ℝ) (h1 : 2 - x = x + 4) (h2 : m * (1 - x) = x + 3) : m = 1 := sorry

-- Part 2: Solve the system of equations given m = 1
theorem solve_system (x y : ℝ) (h1 : 3 * x + 2 * 1 = - y) (h2 : 2 * x + 2 * y = 1 - 1) : x = -1 ∧ y = 1 := sorry

-- Part 3: Relationship between x and y regardless of m
theorem relationship_x_y (x y m : ℝ) (h1 : 3 * x + y = -2 * m) (h2 : 2 * x + 2 * y = m - 1) : 7 * x + 5 * y = -2 := sorry

end value_of_m_solve_system_relationship_x_y_l1078_107858


namespace simplify_expression_l1078_107810

theorem simplify_expression : ((1 + 2 + 3 + 4 + 5 + 6) / 3 + (3 * 5 + 12) / 4) = 13.75 :=
by
-- Proof steps would go here, but we replace them with 'sorry' for now.
sorry

end simplify_expression_l1078_107810


namespace joe_used_fraction_paint_in_first_week_l1078_107893

variable (x : ℝ) -- Define the fraction x as a real number

-- Given conditions
def given_conditions : Prop := 
  let total_paint := 360
  let paint_first_week := x * total_paint
  let remaining_paint := (1 - x) * total_paint
  let paint_second_week := (1 / 2) * remaining_paint
  paint_first_week + paint_second_week = 225

-- The theorem to prove
theorem joe_used_fraction_paint_in_first_week (h : given_conditions x) : x = 1 / 4 :=
sorry

end joe_used_fraction_paint_in_first_week_l1078_107893


namespace fraction_addition_correct_l1078_107835

theorem fraction_addition_correct : (3 / 5 : ℚ) + (2 / 5) = 1 := 
by
  sorry

end fraction_addition_correct_l1078_107835


namespace fraction_equality_l1078_107880

theorem fraction_equality : 
  (3 ^ 8 + 3 ^ 6) / (3 ^ 8 - 3 ^ 6) = 5 / 4 :=
by
  -- Expression rewrite and manipulation inside parenthesis can be ommited
  sorry

end fraction_equality_l1078_107880


namespace athletes_in_camp_hours_l1078_107877

theorem athletes_in_camp_hours (initial_athletes : ℕ) (left_rate : ℕ) (left_hours : ℕ) (arrived_rate : ℕ) 
  (difference : ℕ) (hours : ℕ) 
  (h_initial: initial_athletes = 300) 
  (h_left_rate: left_rate = 28) 
  (h_left_hours: left_hours = 4) 
  (h_arrived_rate: arrived_rate = 15) 
  (h_difference: difference = 7) 
  (h_left: left_rate * left_hours = 112) 
  (h_equation: initial_athletes - (left_rate * left_hours) + (arrived_rate * hours) = initial_athletes - difference) : 
  hours = 7 :=
by
  sorry

end athletes_in_camp_hours_l1078_107877


namespace angle_A_range_l1078_107855

-- Definitions from the conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
axiom triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
axiom longest_side_a : a > b ∧ a > c
axiom inequality_a : a^2 < b^2 + c^2

-- Target proof statement
theorem angle_A_range (triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (longest_side_a : a > b ∧ a > c)
  (inequality_a : a^2 < b^2 + c^2) : 60 < A ∧ A < 90 := 
sorry

end angle_A_range_l1078_107855


namespace kim_boxes_on_thursday_l1078_107856

theorem kim_boxes_on_thursday (Tues Wed Thurs : ℕ) 
(h1 : Tues = 4800)
(h2 : Tues = 2 * Wed)
(h3 : Wed = 2 * Thurs) : Thurs = 1200 :=
by
  sorry

end kim_boxes_on_thursday_l1078_107856


namespace hyperbola_equation_l1078_107802

-- Define the conditions of the problem
def asymptotic_eq (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y → (y = 2 * x ∨ y = -2 * x)

def passes_through_point (C : ℝ → ℝ → Prop) : Prop :=
  C 2 2

-- State the equation of the hyperbola
def is_equation_of_hyperbola (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y ↔ x^2 / 3 - y^2 / 12 = 1

-- The theorem statement combining all conditions to prove the final equation
theorem hyperbola_equation {C : ℝ → ℝ → Prop} :
  asymptotic_eq C →
  passes_through_point C →
  is_equation_of_hyperbola C :=
by
  sorry

end hyperbola_equation_l1078_107802


namespace intersecting_points_are_same_l1078_107838

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 5

def center2 : ℝ × ℝ := (3, 6)
def radius2 : ℝ := 3

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + (y + center1.2)^2 = radius1^2
def circle2 (x y : ℝ) : Prop := (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Prove that points C and D coincide
theorem intersecting_points_are_same : ∃ x y, circle1 x y ∧ circle2 x y → (0 = 0) :=
by
  sorry

end intersecting_points_are_same_l1078_107838


namespace Xiaogang_raised_arm_exceeds_head_l1078_107871

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end Xiaogang_raised_arm_exceeds_head_l1078_107871


namespace solution_set_of_x_squared_geq_four_l1078_107853

theorem solution_set_of_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
sorry

end solution_set_of_x_squared_geq_four_l1078_107853


namespace sum_of_roots_zero_l1078_107887

theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : ∀ x, x^2 + p * x + q = 0) : p + q = 0 := 
by {
  sorry 
}

end sum_of_roots_zero_l1078_107887


namespace average_weight_of_all_boys_l1078_107848

theorem average_weight_of_all_boys (total_boys_16 : ℕ) (avg_weight_boys_16 : ℝ)
  (total_boys_8 : ℕ) (avg_weight_boys_8 : ℝ) 
  (h1 : total_boys_16 = 16) (h2 : avg_weight_boys_16 = 50.25)
  (h3 : total_boys_8 = 8) (h4 : avg_weight_boys_8 = 45.15) : 
  (total_boys_16 * avg_weight_boys_16 + total_boys_8 * avg_weight_boys_8) / (total_boys_16 + total_boys_8) = 48.55 :=
by
  sorry

end average_weight_of_all_boys_l1078_107848


namespace product_of_reciprocals_is_9_over_4_l1078_107827

noncomputable def product_of_reciprocals (a b : ℝ) : ℝ :=
  (1 / a) * (1 / b)

theorem product_of_reciprocals_is_9_over_4 (a b : ℝ) (h : a + b = 3 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  product_of_reciprocals a b = 9 / 4 :=
sorry

end product_of_reciprocals_is_9_over_4_l1078_107827


namespace total_revenue_correct_l1078_107843

def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sneakers_sold : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sandals_sold : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.40
def pairs_boots_sold : ℕ := 11

def calculate_total_revenue : ℝ := 
  let revenue_sneakers := pairs_sneakers_sold * (original_price_sneakers * (1 - discount_sneakers))
  let revenue_sandals := pairs_sandals_sold * (original_price_sandals * (1 - discount_sandals))
  let revenue_boots := pairs_boots_sold * (original_price_boots * (1 - discount_boots))
  revenue_sneakers + revenue_sandals + revenue_boots

theorem total_revenue_correct : calculate_total_revenue = 1068 := by
  sorry

end total_revenue_correct_l1078_107843


namespace cost_of_carton_l1078_107862

-- Definition of given conditions
def totalCost : ℝ := 4.88
def numberOfCartons : ℕ := 4
def costPerCarton : ℝ := 1.22

-- The proof statement
theorem cost_of_carton
  (h : totalCost = 4.88) 
  (n : numberOfCartons = 4) :
  totalCost / numberOfCartons = costPerCarton := 
sorry

end cost_of_carton_l1078_107862


namespace baseball_card_count_l1078_107847

-- Define initial conditions
def initial_cards := 15

-- Maria takes half of one more than the number of initial cards
def maria_takes := (initial_cards + 1) / 2

-- Remaining cards after Maria takes her share
def remaining_after_maria := initial_cards - maria_takes

-- You give Peter 1 card
def remaining_after_peter := remaining_after_maria - 1

-- Paul triples the remaining cards
def final_cards := remaining_after_peter * 3

-- Theorem statement to prove
theorem baseball_card_count :
  final_cards = 18 := by
sorry

end baseball_card_count_l1078_107847


namespace and_false_iff_not_both_true_l1078_107857

variable (p q : Prop)

theorem and_false_iff_not_both_true (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
    sorry

end and_false_iff_not_both_true_l1078_107857


namespace polygon_is_hexagon_l1078_107821

-- Definitions
def side_length : ℝ := 8
def perimeter : ℝ := 48

-- The main theorem to prove
theorem polygon_is_hexagon : (perimeter / side_length = 6) ∧ (48 / 8 = 6) := 
by
  sorry

end polygon_is_hexagon_l1078_107821


namespace linear_term_coefficient_l1078_107894

-- Define the given equation
def equation (x : ℝ) : ℝ := x^2 - 2022*x - 2023

-- The goal is to prove that the coefficient of the linear term in equation is -2022
theorem linear_term_coefficient : ∀ x : ℝ, equation x = x^2 - 2022*x - 2023 → -2022 = -2022 :=
by
  intros x h
  sorry

end linear_term_coefficient_l1078_107894


namespace largest_tan_B_l1078_107854

-- The context of the problem involves a triangle with given side lengths
variables (ABC : Triangle) -- A triangle ABC

-- Define the lengths of sides AB and BC
variables (AB BC : ℝ) 
-- Define the value of tan B
variable (tanB : ℝ)

-- The given conditions
def condition_1 := AB = 25
def condition_2 := BC = 20

-- Define the actual statement we need to prove
theorem largest_tan_B (ABC : Triangle) (AB BC tanB : ℝ) : 
  AB = 25 → BC = 20 → tanB = 3 / 4 := sorry

end largest_tan_B_l1078_107854


namespace neg_mul_reverses_inequality_l1078_107850

theorem neg_mul_reverses_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
  sorry

end neg_mul_reverses_inequality_l1078_107850


namespace cube_surface_area_including_inside_l1078_107891

theorem cube_surface_area_including_inside 
  (original_edge_length : ℝ) 
  (hole_side_length : ℝ) 
  (original_cube_surface_area : ℝ)
  (removed_hole_area : ℝ)
  (newly_exposed_internal_area : ℝ) 
  (total_surface_area : ℝ) 
  (h1 : original_edge_length = 3)
  (h2 : hole_side_length = 1)
  (h3 : original_cube_surface_area = 6 * (original_edge_length * original_edge_length))
  (h4 : removed_hole_area = 6 * (hole_side_length * hole_side_length))
  (h5 : newly_exposed_internal_area = 6 * 4 * (hole_side_length * hole_side_length))
  (h6 : total_surface_area = original_cube_surface_area - removed_hole_area + newly_exposed_internal_area) : 
  total_surface_area = 72 :=
by
  sorry

end cube_surface_area_including_inside_l1078_107891


namespace P_started_following_J_l1078_107803

theorem P_started_following_J :
  ∀ (t : ℝ),
    (6 * 7.3 + 3 = 8 * (7.3 - t)) → t = 1.45 → t + 12 = 13.45 :=
by
  sorry

end P_started_following_J_l1078_107803


namespace find_N_l1078_107829

theorem find_N : 
  (1993 + 1994 + 1995 + 1996 + 1997) / N = (3 + 4 + 5 + 6 + 7) / 5 → 
  N = 1995 :=
by
  sorry

end find_N_l1078_107829


namespace expr_eval_l1078_107814

def expr : ℕ := 3 * 3^4 - 9^27 / 9^25

theorem expr_eval : expr = 162 := by
  -- Proof will be written here if needed
  sorry

end expr_eval_l1078_107814


namespace square_points_sum_of_squares_l1078_107805

theorem square_points_sum_of_squares 
  (a b c d : ℝ) 
  (h₀_a : 0 ≤ a ∧ a ≤ 1)
  (h₀_b : 0 ≤ b ∧ b ≤ 1)
  (h₀_c : 0 ≤ c ∧ c ≤ 1)
  (h₀_d : 0 ≤ d ∧ d ≤ 1) 
  :
  2 ≤ a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ∧
  a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ≤ 4 := 
by
  sorry

end square_points_sum_of_squares_l1078_107805


namespace number_of_avocados_l1078_107816

-- Constants for the given problem
def banana_cost : ℕ := 1
def apple_cost : ℕ := 2
def strawberry_cost_per_12 : ℕ := 4
def avocado_cost : ℕ := 3
def grape_cost_half_bunch : ℕ := 2
def total_cost : ℤ := 28

-- Quantities of the given fruits
def banana_qty : ℕ := 4
def apple_qty : ℕ := 3
def strawberry_qty : ℕ := 24
def grape_qty_full_bunch_cost : ℕ := 4 -- since half bunch cost $2, full bunch cost $4

-- Definition to calculate the cost of the known fruits
def known_fruit_cost : ℤ :=
  (banana_qty * banana_cost) +
  (apple_qty * apple_cost) +
  (strawberry_qty / 12 * strawberry_cost_per_12) +
  grape_qty_full_bunch_cost

-- The cost of avocados needed to fill the total cost
def avocado_cost_needed : ℤ := total_cost - known_fruit_cost

-- Finally, we need to prove that the number of avocados is 2
theorem number_of_avocados (n : ℕ) : n * avocado_cost = avocado_cost_needed → n = 2 :=
by
  -- Problem data
  have h_banana : ℕ := banana_qty * banana_cost
  have h_apple : ℕ := apple_qty * apple_cost
  have h_strawberry : ℕ := (strawberry_qty / 12) * strawberry_cost_per_12
  have h_grape : ℕ := grape_qty_full_bunch_cost
  have h_known : ℕ := h_banana + h_apple + h_strawberry + h_grape
  
  -- Calculation for number of avocados
  have h_avocado : ℤ := total_cost - h_known
  
  -- Proving number of avocados
  sorry

end number_of_avocados_l1078_107816
