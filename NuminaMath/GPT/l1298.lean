import Mathlib

namespace apple_cost_l1298_129886

theorem apple_cost (rate_cost : ℕ) (rate_weight total_weight : ℕ) (h_rate : rate_cost = 5) (h_weight : rate_weight = 7) (h_total : total_weight = 21) :
  ∃ total_cost : ℕ, total_cost = 15 :=
by
  -- The proof will go here
  sorry

end apple_cost_l1298_129886


namespace fill_parentheses_correct_l1298_129811

theorem fill_parentheses_correct (a b : ℝ) :
  (3 * b + a) * (3 * b - a) = 9 * b^2 - a^2 :=
by 
  sorry

end fill_parentheses_correct_l1298_129811


namespace perfect_cubes_in_range_l1298_129807

theorem perfect_cubes_in_range :
  ∃ (n : ℕ), (∀ (k : ℕ), (50 < k^3 ∧ k^3 ≤ 1000) → (k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)) ∧
    (∃ m, (m = 7)) :=
by
  sorry

end perfect_cubes_in_range_l1298_129807


namespace inverse_value_l1298_129896

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value :
  g (-3) = -103 :=
by
  sorry

end inverse_value_l1298_129896


namespace odd_cube_difference_divisible_by_power_of_two_l1298_129831

theorem odd_cube_difference_divisible_by_power_of_two {a b n : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) :
  (2^n ∣ (a^3 - b^3)) ↔ (2^n ∣ (a - b)) :=
by
  sorry

end odd_cube_difference_divisible_by_power_of_two_l1298_129831


namespace integer_pair_solution_l1298_129812

theorem integer_pair_solution (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pair_solution_l1298_129812


namespace quadrilateral_sum_of_squares_l1298_129804

theorem quadrilateral_sum_of_squares
  (a b c d m n t : ℝ) : 
  a^2 + b^2 + c^2 + d^2 = m^2 + n^2 + 4 * t^2 :=
sorry

end quadrilateral_sum_of_squares_l1298_129804


namespace lesser_fraction_l1298_129861

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 17 / 24) (h_prod : x * y = 1 / 8) : min x y = 1 / 3 := by
  sorry

end lesser_fraction_l1298_129861


namespace cost_to_make_each_pop_l1298_129838

-- Define the conditions as given in step a)
def selling_price : ℝ := 1.50
def pops_sold : ℝ := 300
def pencil_cost : ℝ := 1.80
def pencils_to_buy : ℝ := 100

-- Define the total revenue from selling the ice-pops
def total_revenue : ℝ := pops_sold * selling_price

-- Define the total cost to buy the pencils
def total_pencil_cost : ℝ := pencils_to_buy * pencil_cost

-- Define the total profit
def total_profit : ℝ := total_revenue - total_pencil_cost

-- Define the cost to make each ice-pop
theorem cost_to_make_each_pop : total_profit / pops_sold = 0.90 :=
by
  sorry

end cost_to_make_each_pop_l1298_129838


namespace problem_solution_l1298_129800

variable (α β : ℝ)

-- Conditions
variable (h1 : 3 * Real.sin α - Real.cos α = 0)
variable (h2 : 7 * Real.sin β + Real.cos β = 0)
variable (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

theorem problem_solution : 2 * α - β = - (3 * π / 4) := by
  sorry

end problem_solution_l1298_129800


namespace corrected_mean_is_36_74_l1298_129801

noncomputable def corrected_mean (incorrect_mean : ℝ) 
(number_of_observations : ℕ) 
(correct_value wrong_value : ℝ) : ℝ :=
(incorrect_mean * number_of_observations - wrong_value + correct_value) / number_of_observations

theorem corrected_mean_is_36_74 :
  corrected_mean 36 50 60 23 = 36.74 :=
by
  sorry

end corrected_mean_is_36_74_l1298_129801


namespace cupcakes_left_l1298_129813

def pack_count := 3
def cupcakes_per_pack := 4
def cupcakes_eaten := 5

theorem cupcakes_left : (pack_count * cupcakes_per_pack - cupcakes_eaten) = 7 := 
by 
  sorry

end cupcakes_left_l1298_129813


namespace initial_quantity_of_A_l1298_129828

theorem initial_quantity_of_A (x : ℚ) 
    (h1 : 7 * x = a)
    (h2 : 5 * x = b)
    (h3 : a + b = 12 * x)
    (h4 : a' = a - (7 / 12) * 9)
    (h5 : b' = b - (5 / 12) * 9 + 9)
    (h6 : a' / b' = 7 / 9) : 
    a = 23.625 := 
sorry

end initial_quantity_of_A_l1298_129828


namespace range_of_a_l1298_129808

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, -2 < x → -2 < y → x < y → f a x < f a y) → (a > 1/2) :=
by
  sorry

end range_of_a_l1298_129808


namespace average_age_of_dance_group_l1298_129806

theorem average_age_of_dance_group (S_f S_m : ℕ) (avg_females avg_males : ℕ) 
(hf : avg_females = S_f / 12) (hm : avg_males = S_m / 18) 
(h1 : avg_females = 25) (h2 : avg_males = 40) : 
  (S_f + S_m) / 30 = 34 :=
by
  sorry

end average_age_of_dance_group_l1298_129806


namespace richmond_more_than_victoria_l1298_129852

-- Defining the population of Beacon
def beacon_people : ℕ := 500

-- Defining the population of Victoria based on Beacon's population
def victoria_people : ℕ := 4 * beacon_people

-- Defining the population of Richmond
def richmond_people : ℕ := 3000

-- The proof problem: calculating the difference
theorem richmond_more_than_victoria : richmond_people - victoria_people = 1000 := by
  -- The statement of the theorem
  sorry

end richmond_more_than_victoria_l1298_129852


namespace inverse_proportional_l1298_129875

theorem inverse_proportional (p q : ℝ) (k : ℝ) 
  (h1 : ∀ (p q : ℝ), p * q = k)
  (h2 : p = 25)
  (h3 : q = 6) 
  (h4 : q = 15) : 
  p = 10 := 
by
  sorry

end inverse_proportional_l1298_129875


namespace ben_bonus_amount_l1298_129819

variables (B : ℝ)

-- Conditions
def condition1 := B - (1/22) * B - (1/4) * B - (1/8) * B = 867

-- Theorem statement
theorem ben_bonus_amount (h : condition1 B) : B = 1496.50 := 
sorry

end ben_bonus_amount_l1298_129819


namespace factorize_expression_l1298_129825

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end factorize_expression_l1298_129825


namespace min_value_ineq_l1298_129883

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem min_value_ineq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end min_value_ineq_l1298_129883


namespace point_relationship_l1298_129837

variable {m : ℝ}

theorem point_relationship
    (hA : ∃ y1 : ℝ, y1 = (-4 : ℝ)^2 - 2 * (-4 : ℝ) + m)
    (hB : ∃ y2 : ℝ, y2 = (0 : ℝ)^2 - 2 * (0 : ℝ) + m)
    (hC : ∃ y3 : ℝ, y3 = (3 : ℝ)^2 - 2 * (3 : ℝ) + m) :
    (∃ y2 y3 y1 : ℝ, y2 < y3 ∧ y3 < y1) := by
  sorry

end point_relationship_l1298_129837


namespace mark_performance_length_l1298_129830

theorem mark_performance_length :
  ∃ (x : ℕ), (x > 0) ∧ (6 * 5 * x = 90) ∧ (x = 3) :=
by
  sorry

end mark_performance_length_l1298_129830


namespace tangent_line_min_slope_l1298_129817

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3*x - 1

noncomputable def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 3

theorem tangent_line_min_slope :
  ∃ k b : ℝ, (∀ x : ℝ, curve_derivative x ≥ 3) ∧ 
             k = 3 ∧ b = 1 ∧
             (∀ x y : ℝ, y = k * x + b ↔ 3 * x - y + 1 = 0) := 
by {
  sorry
}

end tangent_line_min_slope_l1298_129817


namespace profit_percent_is_26_l1298_129845

variables (P C : ℝ)
variables (h1 : (2/3) * P = 0.84 * C)

theorem profit_percent_is_26 :
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_is_26_l1298_129845


namespace average_last_12_results_l1298_129805

theorem average_last_12_results (S25 S12 S_last12 : ℕ) (A : ℕ) 
  (h1 : S25 = 25 * 24) 
  (h2: S12 = 12 * 14) 
  (h3: 12 * A = S_last12)
  (h4: S25 = S12 + 228 + S_last12) : A = 17 := 
by
  sorry

end average_last_12_results_l1298_129805


namespace Malou_average_is_correct_l1298_129892

def quiz1_score : ℕ := 91
def quiz2_score : ℕ := 90
def quiz3_score : ℕ := 92
def total_score : ℕ := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes : ℕ := 3

def Malous_average_score : ℕ := total_score / number_of_quizzes

theorem Malou_average_is_correct : Malous_average_score = 91 := by
  sorry

end Malou_average_is_correct_l1298_129892


namespace charcoal_drawings_count_l1298_129871

/-- Thomas' drawings problem
  Thomas has 25 drawings in total.
  14 drawings with colored pencils.
  7 drawings with blending markers.
  The rest drawings are made with charcoal.
  We assert that the number of charcoal drawings is 4.
-/
theorem charcoal_drawings_count 
  (total_drawings : ℕ) 
  (colored_pencil_drawings : ℕ) 
  (marker_drawings : ℕ) :
  total_drawings = 25 →
  colored_pencil_drawings = 14 →
  marker_drawings = 7 →
  total_drawings - (colored_pencil_drawings + marker_drawings) = 4 := 
  by
    sorry

end charcoal_drawings_count_l1298_129871


namespace carter_lucy_ratio_l1298_129860

-- Define the number of pages Oliver can read in 1 hour
def oliver_pages : ℕ := 40

-- Define the number of additional pages Lucy can read compared to Oliver
def additional_pages : ℕ := 20

-- Define the number of pages Carter can read in 1 hour
def carter_pages : ℕ := 30

-- Calculate the number of pages Lucy can read in 1 hour
def lucy_pages : ℕ := oliver_pages + additional_pages

-- Prove the ratio of the number of pages Carter can read to the number of pages Lucy can read is 1/2
theorem carter_lucy_ratio : (carter_pages : ℚ) / (lucy_pages : ℚ) = 1 / 2 := by
  sorry

end carter_lucy_ratio_l1298_129860


namespace percentage_of_female_employees_l1298_129881

theorem percentage_of_female_employees (E : ℕ) (hE : E = 1400) 
  (pct_computer_literate : ℚ) (hpct : pct_computer_literate = 0.62)
  (female_computer_literate : ℕ) (hfcl : female_computer_literate = 588)
  (pct_male_computer_literate : ℚ) (hmcl : pct_male_computer_literate = 0.5) :
  100 * (840 / 1400) = 60 := 
by
  sorry

end percentage_of_female_employees_l1298_129881


namespace quadratic_roots_square_cube_sum_l1298_129810

theorem quadratic_roots_square_cube_sum
  (a b c : ℝ) (h : a ≠ 0) (x1 x2 : ℝ)
  (hx : ∀ (x : ℝ), a * x^2 + b * x + c = 0 ↔ x = x1 ∨ x = x2) :
  (x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2) ∧
  (x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3) :=
by
  sorry

end quadratic_roots_square_cube_sum_l1298_129810


namespace division_simplification_l1298_129815

theorem division_simplification :
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18 / 7 :=
by
  sorry

end division_simplification_l1298_129815


namespace common_ratio_of_series_l1298_129826

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end common_ratio_of_series_l1298_129826


namespace sum_first_five_terms_l1298_129829

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

theorem sum_first_five_terms (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 6) : S_5 a = 15 :=
by
  -- skipping actual proof
  sorry

end sum_first_five_terms_l1298_129829


namespace platform_length_proof_l1298_129898

-- Given conditions
def train_length : ℝ := 300
def time_to_cross_platform : ℝ := 27
def time_to_cross_pole : ℝ := 18

-- The length of the platform L to be proved
def length_of_platform (L : ℝ) : Prop := 
  (train_length / time_to_cross_pole) = (train_length + L) / time_to_cross_platform

theorem platform_length_proof : length_of_platform 150 :=
by
  sorry

end platform_length_proof_l1298_129898


namespace find_x_l1298_129848

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 - q.2)

theorem find_x : ∃ x : ℤ, ∃ y : ℤ, star (4, 5) (1, 3) = star (x, y) (2, 1) ∧ x = 3 :=
by 
  sorry

end find_x_l1298_129848


namespace skittles_students_division_l1298_129870

theorem skittles_students_division (n : ℕ) (h1 : 27 % 3 = 0) (h2 : 27 / 3 = n) : n = 9 := by
  sorry

end skittles_students_division_l1298_129870


namespace metallic_sphere_radius_l1298_129876

theorem metallic_sphere_radius 
  (r_wire : ℝ)
  (h_wire : ℝ)
  (r_sphere : ℝ) 
  (V_sphere : ℝ)
  (V_wire : ℝ)
  (h_wire_eq : h_wire = 16)
  (r_wire_eq : r_wire = 12)
  (V_wire_eq : V_wire = π * r_wire^2 * h_wire)
  (V_sphere_eq : V_sphere = (4/3) * π * r_sphere^3)
  (volume_eq : V_sphere = V_wire) :
  r_sphere = 12 :=
by
  sorry

end metallic_sphere_radius_l1298_129876


namespace discriminant_nonnegative_l1298_129869

theorem discriminant_nonnegative (x : ℤ) (h : x^2 * (25 - 24 * x^2) ≥ 0) : x = 0 ∨ x = 1 ∨ x = -1 :=
by sorry

end discriminant_nonnegative_l1298_129869


namespace arnold_danny_age_l1298_129836

theorem arnold_danny_age:
  ∃ x : ℝ, (x + 1) * (x + 1) = x * x + 11 ∧ x = 5 :=
by
  sorry

end arnold_danny_age_l1298_129836


namespace smallest_positive_angle_same_terminal_side_l1298_129859

theorem smallest_positive_angle_same_terminal_side : 
  ∃ k : ℤ, (∃ α : ℝ, α > 0 ∧ α = -660 + k * 360) ∧ (∀ β : ℝ, β > 0 ∧ β = -660 + k * 360 → β ≥ α) :=
sorry

end smallest_positive_angle_same_terminal_side_l1298_129859


namespace rahul_task_days_l1298_129832

theorem rahul_task_days (R : ℕ) (h1 : ∀ x : ℤ, x > 0 → 1 / R + 1 / 84 = 1 / 35) : R = 70 := 
by
  -- placeholder for the proof
  sorry

end rahul_task_days_l1298_129832


namespace sam_original_puppies_count_l1298_129885

theorem sam_original_puppies_count 
  (spotted_puppies_start : ℕ)
  (non_spotted_puppies_start : ℕ)
  (spotted_puppies_given : ℕ)
  (non_spotted_puppies_given : ℕ)
  (spotted_puppies_left : ℕ)
  (non_spotted_puppies_left : ℕ)
  (h1 : spotted_puppies_start = 8)
  (h2 : non_spotted_puppies_start = 5)
  (h3 : spotted_puppies_given = 2)
  (h4 : non_spotted_puppies_given = 3)
  (h5 : spotted_puppies_left = spotted_puppies_start - spotted_puppies_given)
  (h6 : non_spotted_puppies_left = non_spotted_puppies_start - non_spotted_puppies_given)
  (h7 : spotted_puppies_left = 6)
  (h8 : non_spotted_puppies_left = 2) :
  spotted_puppies_start + non_spotted_puppies_start = 13 :=
by
  sorry

end sam_original_puppies_count_l1298_129885


namespace smallest_natural_number_l1298_129853

theorem smallest_natural_number (a : ℕ) : 
  (∃ a, a % 3 = 0 ∧ (a - 1) % 4 = 0 ∧ (a - 2) % 5 = 0) → a = 57 :=
by
  sorry

end smallest_natural_number_l1298_129853


namespace sum_of_transformed_roots_l1298_129874

theorem sum_of_transformed_roots (α β γ : ℂ) (h₁ : α^3 - α + 1 = 0) (h₂ : β^3 - β + 1 = 0) (h₃ : γ^3 - γ + 1 = 0) :
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
by
  sorry

end sum_of_transformed_roots_l1298_129874


namespace largest_quantity_l1298_129878

theorem largest_quantity (x y z w : ℤ) (h : x + 5 = y - 3 ∧ y - 3 = z + 2 ∧ z + 2 = w - 4) : w > y ∧ w > z ∧ w > x :=
by
  sorry

end largest_quantity_l1298_129878


namespace surface_area_with_holes_l1298_129858

-- Define the cube and holes properties
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def number_faces_cube : ℕ := 6

-- Define areas
def area_face_cube := edge_length_cube ^ 2
def area_face_hole := side_length_hole ^ 2
def original_surface_area := number_faces_cube * area_face_cube
def total_hole_area := number_faces_cube * area_face_hole
def new_exposed_area := number_faces_cube * 4 * area_face_hole

-- Calculate the total surface area including holes
def total_surface_area := original_surface_area - total_hole_area + new_exposed_area

-- Lean statement for the proof
theorem surface_area_with_holes :
  total_surface_area = 168 := by
  sorry

end surface_area_with_holes_l1298_129858


namespace isosceles_right_triangle_solution_l1298_129897

theorem isosceles_right_triangle_solution (a b : ℝ) (area : ℝ) 
  (h1 : a = b) (h2 : XY = a * Real.sqrt 2) (h3 : area = (1/2) * a * b) (h4 : area = 36) : 
  XY = 12 :=
by
  sorry

end isosceles_right_triangle_solution_l1298_129897


namespace compute_value_l1298_129835

theorem compute_value {p q : ℝ} (h1 : 3 * p^2 - 5 * p - 8 = 0) (h2 : 3 * q^2 - 5 * q - 8 = 0) :
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 :=
by
  sorry

end compute_value_l1298_129835


namespace hexagon_rectangle_ratio_l1298_129894

theorem hexagon_rectangle_ratio:
  ∀ (h w : ℕ), 
  (6 * h = 24) → (2 * (2 * w + w) = 24) → 
  (h / w = 1) := by
  intros h w
  intro hex_condition
  intro rect_condition
  sorry

end hexagon_rectangle_ratio_l1298_129894


namespace min_value_of_fraction_sum_l1298_129868

theorem min_value_of_fraction_sum (a b : ℤ) (h1 : a = b + 1) : 
  (a > b) -> (∃ x, x > 0 ∧ ((a + b) / (a - b) + (a - b) / (a + b)) = 2) :=
by
  sorry

end min_value_of_fraction_sum_l1298_129868


namespace cubic_polynomial_roots_product_l1298_129842

theorem cubic_polynomial_roots_product :
  (∃ a b c : ℝ, (3*a^3 - 9*a^2 + 5*a - 15 = 0) ∧
               (3*b^3 - 9*b^2 + 5*b - 15 = 0) ∧
               (3*c^3 - 9*c^2 + 5*c - 15 = 0) ∧
               a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  ∃ a b c : ℝ, (3*a*b*c = 5) := 
sorry

end cubic_polynomial_roots_product_l1298_129842


namespace right_triangle_min_area_l1298_129818

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l1298_129818


namespace percentage_of_items_sold_l1298_129887

theorem percentage_of_items_sold (total_items price_per_item discount_rate debt creditors_balance remaining_balance : ℕ)
  (H1 : total_items = 2000)
  (H2 : price_per_item = 50)
  (H3 : discount_rate = 80)
  (H4 : debt = 15000)
  (H5 : remaining_balance = 3000) :
  (total_items * (price_per_item - (price_per_item * discount_rate / 100)) + remaining_balance = debt + remaining_balance) →
  (remaining_balance / (price_per_item - (price_per_item * discount_rate / 100)) / total_items * 100 = 90) :=
by
  sorry

end percentage_of_items_sold_l1298_129887


namespace factor_expression_eq_l1298_129820

theorem factor_expression_eq (x : ℤ) : 75 * x + 50 = 25 * (3 * x + 2) :=
by
  -- The actual proof is omitted
  sorry

end factor_expression_eq_l1298_129820


namespace find_x_l1298_129862

theorem find_x (h₁ : 2994 / 14.5 = 175) (h₂ : 29.94 / x = 17.5) : x = 29.94 / 17.5 :=
by
  -- skipping proofs
  sorry

end find_x_l1298_129862


namespace quadratic_coefficients_sum_l1298_129854

-- Definition of the quadratic function and the conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Conditions
def vertexCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 2 = 3
  
def pointCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 3 = 2

-- The theorem to prove
theorem quadratic_coefficients_sum (a b c : ℝ)
  (hv : vertexCondition a b c)
  (hp : pointCondition a b c):
  a + b + 2 * c = 2 :=
sorry

end quadratic_coefficients_sum_l1298_129854


namespace quadratic_intersects_once_l1298_129864

theorem quadratic_intersects_once (c : ℝ) : (∀ x : ℝ, x^2 - 6 * x + c = 0 → x = 3 ) ↔ c = 9 :=
by
  sorry

end quadratic_intersects_once_l1298_129864


namespace inequality_solution_l1298_129839

theorem inequality_solution 
  (x : ℝ) 
  (h : 2*x^4 + x^2 - 4*x - 3*x^2 * |x - 2| + 4 ≥ 0) : 
  x ∈ Set.Iic (-2) ∪ Set.Icc ((-1 - Real.sqrt 17) / 4) ((-1 + Real.sqrt 17) / 4) ∪ Set.Ici 1 :=
sorry

end inequality_solution_l1298_129839


namespace distance_between_foci_l1298_129851

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end distance_between_foci_l1298_129851


namespace possible_perimeters_l1298_129827

-- Define the condition that the side lengths satisfy the equation
def sides_satisfy_eqn (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Theorem to prove the possible perimeters
theorem possible_perimeters (x y z : ℝ) (h1 : sides_satisfy_eqn x) (h2 : sides_satisfy_eqn y) (h3 : sides_satisfy_eqn z) :
  (x + y + z = 10) ∨ (x + y + z = 6) ∨ (x + y + z = 12) := by
  sorry

end possible_perimeters_l1298_129827


namespace shooter_random_event_l1298_129889

def eventA := "The sun rises from the east"
def eventB := "A coin thrown up from the ground will fall down"
def eventC := "A shooter hits the target with 10 points in one shot"
def eventD := "Xiao Ming runs at a speed of 30 meters per second"

def is_random_event (event : String) := event = eventC

theorem shooter_random_event : is_random_event eventC := 
by
  sorry

end shooter_random_event_l1298_129889


namespace percentage_difference_max_min_l1298_129809

-- Definitions for the sector angles of each department
def angle_manufacturing := 162.0
def angle_sales := 108.0
def angle_research_and_development := 54.0
def angle_administration := 36.0

-- Full circle in degrees
def full_circle := 360.0

-- Compute the percentage representations of each department
def percentage_manufacturing := (angle_manufacturing / full_circle) * 100
def percentage_sales := (angle_sales / full_circle) * 100
def percentage_research_and_development := (angle_research_and_development / full_circle) * 100
def percentage_administration := (angle_administration / full_circle) * 100

-- Prove that the percentage difference between the department with the maximum and minimum number of employees is 35%
theorem percentage_difference_max_min : 
  percentage_manufacturing - percentage_administration = 35.0 :=
by
  -- placeholder for the actual proof
  sorry

end percentage_difference_max_min_l1298_129809


namespace arithmetic_geometric_means_l1298_129822

theorem arithmetic_geometric_means (a b : ℝ) (h1 : 2 * a = 1 + 2) (h2 : b^2 = (-1) * (-16)) : a * b = 6 ∨ a * b = -6 :=
by
  sorry

end arithmetic_geometric_means_l1298_129822


namespace total_number_of_toys_l1298_129823

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end total_number_of_toys_l1298_129823


namespace range_omega_l1298_129857

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def f' (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem range_omega (t ω φ : ℝ) (hω_pos : ω > 0) (hf_t_zero : f t ω φ = 0) (hf'_t_pos : f' t ω φ > 0) (no_min_value : ∀ x, t ≤ x ∧ x < t + 1 → ∃ y, y ≠ x ∧ f y ω φ < f x ω φ) : π < ω ∧ ω ≤ (3 * π / 2) :=
sorry

end range_omega_l1298_129857


namespace domain_of_f_l1298_129846

theorem domain_of_f :
  (∀ x : ℝ, (0 < 1 - x) ∧ (0 < 3 * x + 1) ↔ ( - (1 / 3 : ℝ) < x ∧ x < 1)) :=
by
  sorry

end domain_of_f_l1298_129846


namespace khali_total_snow_volume_l1298_129843

def length1 : ℝ := 25
def width1 : ℝ := 3
def depth1 : ℝ := 0.75

def length2 : ℝ := 15
def width2 : ℝ := 3
def depth2 : ℝ := 1

def volume1 : ℝ := length1 * width1 * depth1
def volume2 : ℝ := length2 * width2 * depth2
def total_volume : ℝ := volume1 + volume2

theorem khali_total_snow_volume : total_volume = 101.25 := by
  sorry

end khali_total_snow_volume_l1298_129843


namespace find_a_l1298_129834

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l1298_129834


namespace gift_items_l1298_129802

theorem gift_items (x y z : ℕ) : 
  x + y + z = 20 ∧ 60 * x + 50 * y + 10 * z = 720 ↔ 
  ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) :=
by sorry

end gift_items_l1298_129802


namespace midpoint_polar_coords_l1298_129841

/-- 
Given two points in polar coordinates: (6, π/6) and (2, -π/6),  
the midpoint of the line segment connecting these points in polar coordinates is (√13, π/6).
-/
theorem midpoint_polar_coords :
  let A := (6, Real.pi / 6)
  let B := (2, -Real.pi / 6)
  let A_cart := (6 * Real.cos (Real.pi / 6), 6 * Real.sin (Real.pi / 6))
  let B_cart := (2 * Real.cos (-Real.pi / 6), 2 * Real.sin (-Real.pi / 6))
  let Mx := ((A_cart.fst + B_cart.fst) / 2)
  let My := ((A_cart.snd + B_cart.snd) / 2)
  let r := Real.sqrt (Mx^2 + My^2)
  let theta := Real.arctan (My / Mx)
  0 <= theta ∧ theta < 2 * Real.pi ∧ r > 0 ∧ (r = Real.sqrt 13 ∧ theta = Real.pi / 6) :=
by 
  sorry

end midpoint_polar_coords_l1298_129841


namespace asymptotes_of_hyperbola_l1298_129899

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ x y : ℝ, y^2 = 12 * x ∧ (x = 3) ∧ (y = 0)) →
  (a^2 = 9) →
  (∀ b c : ℝ, (b, c) ∈ ({(a, b) | (b = a/3 ∨ b = -a/3)})) :=
by
  intro h_focus_coincides vertex_condition
  sorry

end asymptotes_of_hyperbola_l1298_129899


namespace f_divisible_by_k2_k1_l1298_129866

noncomputable def f (n : ℕ) (x : ℤ) : ℤ :=
  x^(n + 2) + (x + 1)^(2 * n + 1)

theorem f_divisible_by_k2_k1 (n : ℕ) (k : ℤ) (hn : n > 0) : 
  ((k^2 + k + 1) ∣ f n k) :=
sorry

end f_divisible_by_k2_k1_l1298_129866


namespace g_f_eval_l1298_129888

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l1298_129888


namespace sin_add_double_alpha_l1298_129884

open Real

theorem sin_add_double_alpha (alpha : ℝ) (h : sin (π / 6 - alpha) = 3 / 5) :
  sin (π / 6 + 2 * alpha) = 7 / 25 :=
by
  sorry

end sin_add_double_alpha_l1298_129884


namespace div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l1298_129856

theorem div_by_3_9_then_mul_by_5_6_eq_div_by_5_2 :
  (∀ (x : ℚ), (x / (3/9)) * (5/6) = x / (5/2)) :=
by
  sorry

end div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l1298_129856


namespace percentage_error_formula_l1298_129880

noncomputable def percentage_error_in_area (a b : ℝ) (x y : ℝ) :=
  let actual_area := a * b
  let measured_area := a * (1 + x / 100) * b * (1 + y / 100)
  let error_percentage := ((measured_area - actual_area) / actual_area) * 100
  error_percentage

theorem percentage_error_formula (a b x y : ℝ) :
  percentage_error_in_area a b x y = x + y + (x * y / 100) :=
by
  sorry

end percentage_error_formula_l1298_129880


namespace total_students_l1298_129847

theorem total_students (f1 f2 f3 total : ℕ)
  (h_ratio : f1 * 2 = f2)
  (h_ratio2 : f1 * 3 = f3)
  (h_f1 : f1 = 6)
  (h_total : total = f1 + f2 + f3) :
  total = 48 :=
by
  sorry

end total_students_l1298_129847


namespace largest_4_digit_number_divisible_by_24_l1298_129849

theorem largest_4_digit_number_divisible_by_24 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 24 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 24 = 0 → m ≤ n :=
sorry

end largest_4_digit_number_divisible_by_24_l1298_129849


namespace calc_expression_l1298_129873

theorem calc_expression : 2 * 0 * 1 + 1 = 1 :=
by
  sorry

end calc_expression_l1298_129873


namespace smallest_number_property_l1298_129824

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end smallest_number_property_l1298_129824


namespace tangent_line_to_parabola_k_value_l1298_129803

theorem tangent_line_to_parabola_k_value (k : ℝ) :
  (∀ x y : ℝ, 4 * x - 3 * y + k = 0 → y^2 = 16 * x → (4 * x - 3 * y + k = 0 ∧ y^2 = 16 * x) ∧ (144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_k_value_l1298_129803


namespace number_of_passed_candidates_l1298_129872

theorem number_of_passed_candidates :
  ∀ (P F : ℕ),
  (P + F = 500) →
  (P * 80 + F * 15 = 500 * 60) →
  P = 346 :=
by
  intros P F h1 h2
  sorry

end number_of_passed_candidates_l1298_129872


namespace intersection_with_x_axis_l1298_129844

theorem intersection_with_x_axis :
  (∃ x, ∃ y, y = 0 ∧ y = -3 * x + 3 ∧ (x = 1 ∧ y = 0)) :=
by
  -- proof will go here
  sorry

end intersection_with_x_axis_l1298_129844


namespace corrected_mean_is_124_931_l1298_129890

/-
Given:
- original_mean : Real = 125.6
- num_observations : Nat = 100
- incorrect_obs1 : Real = 95.3
- incorrect_obs2 : Real = -15.9
- correct_obs1 : Real = 48.2
- correct_obs2 : Real = -35.7

Prove:
- new_mean == 124.931
-/

noncomputable def original_mean : ℝ := 125.6
def num_observations : ℕ := 100
noncomputable def incorrect_obs1 : ℝ := 95.3
noncomputable def incorrect_obs2 : ℝ := -15.9
noncomputable def correct_obs1 : ℝ := 48.2
noncomputable def correct_obs2 : ℝ := -35.7

noncomputable def incorrect_total_sum : ℝ := original_mean * num_observations
noncomputable def sum_incorrect_obs : ℝ := incorrect_obs1 + incorrect_obs2
noncomputable def sum_correct_obs : ℝ := correct_obs1 + correct_obs2
noncomputable def corrected_total_sum : ℝ := incorrect_total_sum - sum_incorrect_obs + sum_correct_obs
noncomputable def new_mean : ℝ := corrected_total_sum / num_observations

theorem corrected_mean_is_124_931 : new_mean = 124.931 := sorry

end corrected_mean_is_124_931_l1298_129890


namespace secant_line_slope_positive_l1298_129814

theorem secant_line_slope_positive (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, 0 < (deriv f x)) :
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → 0 < (f x1 - f x2) / (x1 - x2) :=
by
  intros x1 x2 h_ne
  sorry

end secant_line_slope_positive_l1298_129814


namespace perpendicular_lines_sufficient_l1298_129882

noncomputable def line1_slope (a : ℝ) : ℝ :=
-((a + 2) / (3 * a))

noncomputable def line2_slope (a : ℝ) : ℝ :=
-((a - 2) / (a + 2))

theorem perpendicular_lines_sufficient (a : ℝ) (h : a = -2) :
  line1_slope a * line2_slope a = -1 :=
by
  sorry

end perpendicular_lines_sufficient_l1298_129882


namespace sum_f_a_seq_positive_l1298_129891

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_monotone_decreasing_nonneg : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f y ≤ f x
axiom a_seq : ∀ n : ℕ, ℝ
axiom a_arithmetic : ∀ m n k : ℕ, m + k = 2 * n → a_seq m + a_seq k = 2 * a_seq n
axiom a3_neg : a_seq 3 < 0

theorem sum_f_a_seq_positive :
    f (a_seq 1) + 
    f (a_seq 2) + 
    f (a_seq 3) + 
    f (a_seq 4) + 
    f (a_seq 5) > 0 :=
sorry

end sum_f_a_seq_positive_l1298_129891


namespace count_four_digit_numbers_with_repeated_digits_l1298_129816

def countDistinctFourDigitNumbersWithRepeatedDigits : Nat :=
  let totalNumbers := 4 ^ 4
  let uniqueNumbers := 4 * 3 * 2 * 1
  totalNumbers - uniqueNumbers

theorem count_four_digit_numbers_with_repeated_digits :
  countDistinctFourDigitNumbersWithRepeatedDigits = 232 := by
  sorry

end count_four_digit_numbers_with_repeated_digits_l1298_129816


namespace lucy_packs_of_cake_l1298_129893

theorem lucy_packs_of_cake (total_groceries cookies : ℕ) (h1 : total_groceries = 27) (h2 : cookies = 23) :
  total_groceries - cookies = 4 :=
by
  -- In Lean, we would provide the actual proof here, but we'll use sorry to skip the proof as instructed
  sorry

end lucy_packs_of_cake_l1298_129893


namespace problems_on_each_worksheet_l1298_129840

-- Define the conditions
def worksheets_total : Nat := 9
def worksheets_graded : Nat := 5
def problems_left : Nat := 16

-- Define the number of remaining worksheets and the problems per worksheet
def remaining_worksheets : Nat := worksheets_total - worksheets_graded
def problems_per_worksheet : Nat := problems_left / remaining_worksheets

-- Prove the number of problems on each worksheet
theorem problems_on_each_worksheet : problems_per_worksheet = 4 :=
by
  sorry

end problems_on_each_worksheet_l1298_129840


namespace length_ST_l1298_129821

theorem length_ST (PQ QR RS SP SQ PT RT : ℝ) 
  (h1 : PQ = 6) (h2 : QR = 6)
  (h3 : RS = 6) (h4 : SP = 6)
  (h5 : SQ = 6) (h6 : PT = 14)
  (h7 : RT = 14) : 
  ∃ ST : ℝ, ST = 10 := 
by
  -- sorry is used to complete the theorem without a proof
  sorry

end length_ST_l1298_129821


namespace elevator_max_weight_l1298_129833

theorem elevator_max_weight :
  let avg_weight_adult := 150
  let num_adults := 7
  let avg_weight_child := 70
  let num_children := 5
  let orig_max_weight := 1500
  let weight_adults := num_adults * avg_weight_adult
  let weight_children := num_children * avg_weight_child
  let current_weight := weight_adults + weight_children
  let upgrade_percentage := 0.10
  let new_max_weight := orig_max_weight * (1 + upgrade_percentage)
  new_max_weight - current_weight = 250 := 
  by
    sorry

end elevator_max_weight_l1298_129833


namespace added_water_proof_l1298_129855

variable (total_volume : ℕ) (milk_ratio water_ratio : ℕ) (added_water : ℕ)

theorem added_water_proof 
  (h1 : total_volume = 45) 
  (h2 : milk_ratio = 4) 
  (h3 : water_ratio = 1) 
  (h4 : added_water = 3) 
  (milk_volume : ℕ)
  (water_volume : ℕ)
  (h5 : milk_volume = (milk_ratio * total_volume) / (milk_ratio + water_ratio))
  (h6 : water_volume = (water_ratio * total_volume) / (milk_ratio + water_ratio))
  (new_ratio : ℕ)
  (h7 : new_ratio = milk_volume / (water_volume + added_water)) : added_water = 3 :=
by
  sorry

end added_water_proof_l1298_129855


namespace steve_average_speed_l1298_129879

-- Define the conditions as constants
def hours1 := 5
def speed1 := 40
def hours2 := 3
def speed2 := 80
def hours3 := 2
def speed3 := 60

-- Define a theorem that calculates average speed and proves the result is 56
theorem steve_average_speed :
  (hours1 * speed1 + hours2 * speed2 + hours3 * speed3) / (hours1 + hours2 + hours3) = 56 := by
  sorry

end steve_average_speed_l1298_129879


namespace certain_number_division_l1298_129865

theorem certain_number_division (N G : ℤ) : 
  G = 88 ∧ (∃ k : ℤ, N = G * k + 31) ∧ (∃ m : ℤ, 4521 = G * m + 33) → 
  N = 4519 := 
by
  sorry

end certain_number_division_l1298_129865


namespace largest_B_at_45_l1298_129863

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def B (k : ℕ) : ℝ :=
  if k ≤ 500 then (binomial_coeff 500 k) * (0.1)^k else 0

theorem largest_B_at_45 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → B k ≤ B 45 :=
by
  intros k hk
  sorry

end largest_B_at_45_l1298_129863


namespace xiaoxian_mistake_xiaoxuan_difference_l1298_129895

-- Define the initial expressions and conditions
def original_expr := (-9) * 3 - 5
def xiaoxian_expr (x : Int) := (-9) * 3 - x
def xiaoxuan_expr := (-9) / 3 - 5

-- Given conditions
variable (result_xiaoxian : Int)
variable (result_original : Int)

-- Proof statement
theorem xiaoxian_mistake (hx : xiaoxian_expr 2 = -29) : 
  xiaoxian_expr 5 = result_xiaoxian := sorry

theorem xiaoxuan_difference : 
  abs (xiaoxuan_expr - original_expr) = 24 := sorry

end xiaoxian_mistake_xiaoxuan_difference_l1298_129895


namespace simplified_expression_num_terms_l1298_129850

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end simplified_expression_num_terms_l1298_129850


namespace division_quotient_remainder_l1298_129867

theorem division_quotient_remainder (A : ℕ) (h1 : A / 9 = 2) (h2 : A % 9 = 6) : A = 24 := 
by
  sorry

end division_quotient_remainder_l1298_129867


namespace definite_integral_example_l1298_129877

theorem definite_integral_example : ∫ x in (0 : ℝ)..(π/2), 2 * x = π^2 / 4 := 
by 
  sorry

end definite_integral_example_l1298_129877
