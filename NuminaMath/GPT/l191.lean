import Mathlib

namespace NUMINAMATH_GPT_nina_money_l191_19127

variable (C : ℝ)

def original_widget_count : ℕ := 6
def new_widget_count : ℕ := 8
def price_reduction : ℝ := 1.5

theorem nina_money (h : original_widget_count * C = new_widget_count * (C - price_reduction)) :
  original_widget_count * C = 36 := by
  sorry

end NUMINAMATH_GPT_nina_money_l191_19127


namespace NUMINAMATH_GPT_set_union_inter_proof_l191_19190

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem set_union_inter_proof : A ∪ B = {0, 1, 2, 3} ∧ A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_GPT_set_union_inter_proof_l191_19190


namespace NUMINAMATH_GPT_sum_of_first_ten_nicely_odd_numbers_is_775_l191_19197

def is_nicely_odd (n : ℕ) : Prop :=
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ (Odd p ∧ Odd q) ∧ n = p * q)
  ∨ (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p ^ 3)

theorem sum_of_first_ten_nicely_odd_numbers_is_775 :
  let nicely_odd_nums := [15, 27, 21, 35, 125, 33, 77, 343, 55, 39]
  ∃ (nums : List ℕ), List.length nums = 10 ∧
  (∀ n ∈ nums, is_nicely_odd n) ∧ List.sum nums = 775 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_nicely_odd_numbers_is_775_l191_19197


namespace NUMINAMATH_GPT_number_of_triangles_l191_19180

/-- 
  This statement defines and verifies the number of triangles 
  in the given geometric figure.
-/
theorem number_of_triangles (rectangle : Set ℝ) : 
  (exists lines : Set (List (ℝ × ℝ)), -- assuming a set of lines dividing the rectangle
    let small_right_triangles := 40
    let intermediate_isosceles_triangles := 8
    let intermediate_triangles := 10
    let larger_right_triangles := 20
    let largest_isosceles_triangles := 5
    small_right_triangles + intermediate_isosceles_triangles + intermediate_triangles + larger_right_triangles + largest_isosceles_triangles = 83) :=
sorry

end NUMINAMATH_GPT_number_of_triangles_l191_19180


namespace NUMINAMATH_GPT_area_of_circle_l191_19179

theorem area_of_circle (r : ℝ) : 
  (S = π * r^2) :=
sorry

end NUMINAMATH_GPT_area_of_circle_l191_19179


namespace NUMINAMATH_GPT_kanul_spent_on_raw_materials_l191_19129

theorem kanul_spent_on_raw_materials 
    (total_amount : ℝ)
    (spent_machinery : ℝ)
    (spent_cash_percent : ℝ)
    (spent_cash : ℝ)
    (amount_raw_materials : ℝ)
    (h_total : total_amount = 93750)
    (h_machinery : spent_machinery = 40000)
    (h_percent : spent_cash_percent = 20 / 100)
    (h_cash : spent_cash = spent_cash_percent * total_amount)
    (h_sum : total_amount = amount_raw_materials + spent_machinery + spent_cash) : 
    amount_raw_materials = 35000 :=
sorry

end NUMINAMATH_GPT_kanul_spent_on_raw_materials_l191_19129


namespace NUMINAMATH_GPT_find_a_l191_19177

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a = 0 }

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l191_19177


namespace NUMINAMATH_GPT_distinct_values_of_c_l191_19172

theorem distinct_values_of_c {c p q : ℂ} 
  (h_distinct : p ≠ q) 
  (h_eq : ∀ z : ℂ, (z - p) * (z - q) = (z - c * p) * (z - c * q)) :
  (∃ c_values : ℕ, c_values = 2) :=
sorry

end NUMINAMATH_GPT_distinct_values_of_c_l191_19172


namespace NUMINAMATH_GPT_donna_pizza_slices_left_l191_19120

def total_slices_initial : ℕ := 12
def slices_eaten_lunch (slices : ℕ) : ℕ := slices / 2
def slices_remaining_after_lunch (slices : ℕ) : ℕ := slices - slices_eaten_lunch slices
def slices_eaten_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices / 3
def slices_remaining_after_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices - slices_eaten_dinner slices
def slices_shared_friend (slices : ℕ) : ℕ := slices_remaining_after_dinner slices / 4
def slices_remaining_final (slices : ℕ) : ℕ := slices_remaining_after_dinner slices - slices_shared_friend slices

theorem donna_pizza_slices_left : slices_remaining_final total_slices_initial = 3 :=
sorry

end NUMINAMATH_GPT_donna_pizza_slices_left_l191_19120


namespace NUMINAMATH_GPT_rectangle_area_l191_19165

theorem rectangle_area (P l w : ℕ) (h_perimeter: 2 * l + 2 * w = 60) (h_aspect: l = 3 * w / 2) : l * w = 216 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l191_19165


namespace NUMINAMATH_GPT_sequence_general_formula_l191_19122

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2 - 2 * n + 2):
  (a 1 = 1) ∧ (∀ n, 1 < n → a n = S n - S (n - 1)) → 
  (∀ n, a n = if n = 1 then 1 else 2 * n - 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l191_19122


namespace NUMINAMATH_GPT_area_ratio_of_squares_l191_19132

theorem area_ratio_of_squares (hA : ∃ sA : ℕ, 4 * sA = 16)
                             (hB : ∃ sB : ℕ, 4 * sB = 20)
                             (hC : ∃ sC : ℕ, 4 * sC = 40) :
  (∃ aB aC : ℕ, aB = sB * sB ∧ aC = sC * sC ∧ aB * 4 = aC) := by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l191_19132


namespace NUMINAMATH_GPT_ceil_of_neg_frac_squared_l191_19157

-- Define the negated fraction
def neg_frac : ℚ := -7 / 4

-- Define the squared value of the negated fraction
def squared_value : ℚ := neg_frac ^ 2

-- Define the ceiling function applied to the squared value
def ceil_squared_value : ℤ := Int.ceil squared_value

-- Prove that the ceiling of the squared value is 4
theorem ceil_of_neg_frac_squared : ceil_squared_value = 4 := 
by sorry

end NUMINAMATH_GPT_ceil_of_neg_frac_squared_l191_19157


namespace NUMINAMATH_GPT_intersection_empty_l191_19139

open Set

-- Definition of set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 2 * x + 3) }

-- Definition of set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 4 * x + 1) }

-- The proof problem statement in Lean
theorem intersection_empty : A ∩ B = ∅ := sorry

end NUMINAMATH_GPT_intersection_empty_l191_19139


namespace NUMINAMATH_GPT_sequence_fifth_number_l191_19196

theorem sequence_fifth_number : (5^2 - 1) = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_fifth_number_l191_19196


namespace NUMINAMATH_GPT_values_of_a_l191_19130

noncomputable def quadratic_eq (a x : ℝ) : ℝ :=
(a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_eq a x = 0 → x ≥ 0) ↔ (a = 3 ∨ (-1 ≤ a ∧ a ≤ 1)) :=
sorry

end NUMINAMATH_GPT_values_of_a_l191_19130


namespace NUMINAMATH_GPT_ceil_floor_difference_is_3_l191_19181

noncomputable def ceil_floor_difference : ℤ :=
  Int.ceil ((14:ℚ) / 5 * (-31 / 3)) - Int.floor ((14 / 5) * Int.floor ((-31:ℚ) / 3))

theorem ceil_floor_difference_is_3 : ceil_floor_difference = 3 :=
  sorry

end NUMINAMATH_GPT_ceil_floor_difference_is_3_l191_19181


namespace NUMINAMATH_GPT_geometric_series_sum_l191_19176

theorem geometric_series_sum : 
  let a := 1
  let r := 2
  let n := 21
  a * ((r^n - 1) / (r - 1)) = 2097151 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l191_19176


namespace NUMINAMATH_GPT_flour_ratio_correct_l191_19164

-- Definitions based on conditions
def initial_sugar : ℕ := 13
def initial_flour : ℕ := 25
def initial_baking_soda : ℕ := 35
def initial_cocoa_powder : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa_powder : ℕ := 15

-- Calculate remaining ingredients
def remaining_flour : ℕ := initial_flour - added_flour
def remaining_sugar : ℕ := initial_sugar - added_sugar
def remaining_cocoa_powder : ℕ := initial_cocoa_powder - added_cocoa_powder

-- Calculate ratio
def total_remaining_sugar_and_cocoa : ℕ := remaining_sugar + remaining_cocoa_powder
def flour_to_sugar_cocoa_ratio : ℕ × ℕ := (remaining_flour, total_remaining_sugar_and_cocoa)

-- Proposition stating the desired ratio
theorem flour_ratio_correct : flour_to_sugar_cocoa_ratio = (17, 46) := by
  sorry

end NUMINAMATH_GPT_flour_ratio_correct_l191_19164


namespace NUMINAMATH_GPT_values_of_a_l191_19116

axiom exists_rat : (x y a : ℚ) → Prop

theorem values_of_a (a : ℚ) (h1 : ∀ x y : ℚ, (x/2 - (2*x - 3*y)/5 = a - 1)) (h2 : ∀ x y : ℚ, (x + 3 = y/3)) :
  0.7 < a ∧ a < 6.4 ↔ (∃ x y : ℚ, x < 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_GPT_values_of_a_l191_19116


namespace NUMINAMATH_GPT_part_1_part_2_l191_19115

variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a = b ∧ b = c ∧ A + B + C = 180 ∧ A = 90 ∨ B = 90 ∨ C = 90)
variable (sin_condition : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C)

theorem part_1 (h : a = b) : Real.cos C = 7 / 8 :=
by { sorry }

theorem part_2 (h₁ : B = 90) (h₂ : a = Real.sqrt 2) : b = 2 :=
by { sorry }

end NUMINAMATH_GPT_part_1_part_2_l191_19115


namespace NUMINAMATH_GPT_inequality_always_holds_l191_19101

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 - m * x - 1 < 0) → -4 < m ∧ m ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l191_19101


namespace NUMINAMATH_GPT_range_of_a_l191_19147

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → (x - 1) ^ 2 < Real.log x / Real.log a) → a ∈ Set.Ioc 1 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l191_19147


namespace NUMINAMATH_GPT_complement_of_A_relative_to_I_l191_19185

def I : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

def complement_I_A : Set ℤ := {x ∈ I | x ∉ A}

theorem complement_of_A_relative_to_I :
  complement_I_A = {-2, 2} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_relative_to_I_l191_19185


namespace NUMINAMATH_GPT_belindas_age_l191_19162

theorem belindas_age (T B : ℕ) (h1 : T + B = 56) (h2 : B = 2 * T + 8) (h3 : T = 16) : B = 40 :=
by
  sorry

end NUMINAMATH_GPT_belindas_age_l191_19162


namespace NUMINAMATH_GPT_saved_percentage_this_year_l191_19166

variable (S : ℝ) -- Annual salary last year

-- Conditions
def saved_last_year := 0.06 * S
def salary_this_year := 1.20 * S
def saved_this_year := saved_last_year

-- The goal is to prove that the percentage saved this year is 5%
theorem saved_percentage_this_year :
  (saved_this_year / salary_this_year) * 100 = 5 :=
by sorry

end NUMINAMATH_GPT_saved_percentage_this_year_l191_19166


namespace NUMINAMATH_GPT_simplify_expression_l191_19158

theorem simplify_expression (α : ℝ) (h_sin_ne_zero : Real.sin α ≠ 0) :
    (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := 
sorry

end NUMINAMATH_GPT_simplify_expression_l191_19158


namespace NUMINAMATH_GPT_range_of_a_l191_19178

theorem range_of_a (a : ℝ) :
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l191_19178


namespace NUMINAMATH_GPT_arithmetic_equation_false_l191_19105

theorem arithmetic_equation_false :
  4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 := by
  sorry

end NUMINAMATH_GPT_arithmetic_equation_false_l191_19105


namespace NUMINAMATH_GPT_min_squared_distance_l191_19119

open Real

theorem min_squared_distance : ∀ (x y : ℝ), (3 * x + y = 10) → (x^2 + y^2) ≥ 10 :=
by
  intros x y hxy
  -- Insert the necessary steps or key elements here
  sorry

end NUMINAMATH_GPT_min_squared_distance_l191_19119


namespace NUMINAMATH_GPT_promotional_price_difference_l191_19171

theorem promotional_price_difference
  (normal_price : ℝ)
  (months : ℕ)
  (issues_per_month : ℕ)
  (discount_per_issue : ℝ)
  (h1 : normal_price = 34)
  (h2 : months = 18)
  (h3 : issues_per_month = 2)
  (h4 : discount_per_issue = 0.25) : 
  normal_price - (months * issues_per_month * discount_per_issue) = 9 := 
by 
  sorry

end NUMINAMATH_GPT_promotional_price_difference_l191_19171


namespace NUMINAMATH_GPT_table_length_is_77_l191_19160

theorem table_length_is_77 :
  ∀ (x : ℕ), (∀ (sheets: ℕ), sheets = 72 → x = (5 + sheets)) → x = 77 :=
by {
  sorry
}

end NUMINAMATH_GPT_table_length_is_77_l191_19160


namespace NUMINAMATH_GPT_inequality_solution_l191_19113

theorem inequality_solution (a c : ℝ) (h : ∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ ax^2 + 5*x + c > 0) : a + c = -7 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l191_19113


namespace NUMINAMATH_GPT_negation_of_exists_gt_one_l191_19112

theorem negation_of_exists_gt_one : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_exists_gt_one_l191_19112


namespace NUMINAMATH_GPT_greatest_k_for_factorial_div_l191_19188

-- Definitions for conditions in the problem
def a : Nat := Nat.factorial 100
noncomputable def b (k : Nat) : Nat := 100^k

-- Statement to prove the greatest value of k for which b is a factor of a
theorem greatest_k_for_factorial_div (k : Nat) : 
  (∀ m : Nat, (m ≤ k → b m ∣ a) ↔ m ≤ 12) := 
by
  sorry

end NUMINAMATH_GPT_greatest_k_for_factorial_div_l191_19188


namespace NUMINAMATH_GPT_james_out_of_pocket_cost_l191_19103

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end NUMINAMATH_GPT_james_out_of_pocket_cost_l191_19103


namespace NUMINAMATH_GPT_arctan_combination_l191_19142

noncomputable def find_m : ℕ :=
  133

theorem arctan_combination :
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/find_m)) = (Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_arctan_combination_l191_19142


namespace NUMINAMATH_GPT_Shelby_drive_time_in_rain_l191_19124

theorem Shelby_drive_time_in_rain (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 3) 
  (h3 : 40 * (3 - x) + 25 * x = 85) : x = 140 / 60 :=
  sorry

end NUMINAMATH_GPT_Shelby_drive_time_in_rain_l191_19124


namespace NUMINAMATH_GPT_func_g_neither_even_nor_odd_l191_19123

noncomputable def func_g (x : ℝ) : ℝ := (⌈x⌉ : ℝ) - (1 / 3)

theorem func_g_neither_even_nor_odd :
  (¬ ∀ x, func_g (-x) = func_g x) ∧ (¬ ∀ x, func_g (-x) = -func_g x) :=
by
  sorry

end NUMINAMATH_GPT_func_g_neither_even_nor_odd_l191_19123


namespace NUMINAMATH_GPT_raritet_meets_ferries_l191_19156

theorem raritet_meets_ferries :
  (∀ (n : ℕ), ∃ (ferry_departure : Nat), ferry_departure = n ∧ ferry_departure + 8 = 8) →
  (∀ (m : ℕ), ∃ (raritet_departure : Nat), raritet_departure = m ∧ raritet_departure + 8 = 8) →
  ∃ (total_meetings : Nat), total_meetings = 17 := 
by
  sorry

end NUMINAMATH_GPT_raritet_meets_ferries_l191_19156


namespace NUMINAMATH_GPT_meaningful_range_l191_19170

theorem meaningful_range (x : ℝ) : (x < 4) ↔ (4 - x > 0) := 
by sorry

end NUMINAMATH_GPT_meaningful_range_l191_19170


namespace NUMINAMATH_GPT_percentage_by_which_x_is_more_than_y_l191_19175

variable {z : ℝ} 

-- Define x and y based on the given conditions
def x (z : ℝ) : ℝ := 0.78 * z
def y (z : ℝ) : ℝ := 0.60 * z

-- The main theorem we aim to prove
theorem percentage_by_which_x_is_more_than_y (z : ℝ) : x z = y z + 0.30 * y z := by
  sorry

end NUMINAMATH_GPT_percentage_by_which_x_is_more_than_y_l191_19175


namespace NUMINAMATH_GPT_grill_runtime_l191_19167

theorem grill_runtime
    (burn_rate : ℕ)
    (burn_time : ℕ)
    (bags : ℕ)
    (coals_per_bag : ℕ)
    (total_burnt_coals : ℕ)
    (total_time : ℕ)
    (h1 : burn_rate = 15)
    (h2 : burn_time = 20)
    (h3 : bags = 3)
    (h4 : coals_per_bag = 60)
    (h5 : total_burnt_coals = bags * coals_per_bag)
    (h6 : total_time = (total_burnt_coals / burn_rate) * burn_time) :
    total_time = 240 :=
by sorry

end NUMINAMATH_GPT_grill_runtime_l191_19167


namespace NUMINAMATH_GPT_billy_reads_books_l191_19134

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_billy_reads_books_l191_19134


namespace NUMINAMATH_GPT_initial_ants_count_l191_19173

theorem initial_ants_count (n : ℕ) (h1 : ∀ x : ℕ, x ≠ n - 42 → x ≠ 42) : n = 42 :=
sorry

end NUMINAMATH_GPT_initial_ants_count_l191_19173


namespace NUMINAMATH_GPT_pencils_added_by_Nancy_l191_19100

def original_pencils : ℕ := 27
def total_pencils : ℕ := 72

theorem pencils_added_by_Nancy : ∃ x : ℕ, x = total_pencils - original_pencils := by
  sorry

end NUMINAMATH_GPT_pencils_added_by_Nancy_l191_19100


namespace NUMINAMATH_GPT_pencils_per_row_l191_19152

def total_pencils : ℕ := 32
def rows : ℕ := 4

theorem pencils_per_row : total_pencils / rows = 8 := by
  sorry

end NUMINAMATH_GPT_pencils_per_row_l191_19152


namespace NUMINAMATH_GPT_students_in_A_and_D_combined_l191_19141

theorem students_in_A_and_D_combined (AB BC CD : ℕ) (hAB : AB = 83) (hBC : BC = 86) (hCD : CD = 88) : (AB + CD - BC = 85) :=
by
  sorry

end NUMINAMATH_GPT_students_in_A_and_D_combined_l191_19141


namespace NUMINAMATH_GPT_potion_combinations_l191_19128

-- Definitions of conditions
def roots : Nat := 3
def minerals : Nat := 5
def incompatible_combinations : Nat := 2

-- Statement of the problem
theorem potion_combinations : (roots * minerals) - incompatible_combinations = 13 := by
  sorry

end NUMINAMATH_GPT_potion_combinations_l191_19128


namespace NUMINAMATH_GPT_chef_michel_total_pies_l191_19174

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_chef_michel_total_pies_l191_19174


namespace NUMINAMATH_GPT_claudia_groupings_l191_19118

-- Definition of combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def candles_combinations : ℕ := combination 6 3
def flowers_combinations : ℕ := combination 15 12

-- Lean statement
theorem claudia_groupings : candles_combinations * flowers_combinations = 9100 :=
by
  sorry

end NUMINAMATH_GPT_claudia_groupings_l191_19118


namespace NUMINAMATH_GPT_min_value_expression_l191_19111

theorem min_value_expression (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : (a - b) * (b - c) * (c - a) = -16) : 
  ∃ x : ℝ, x = (1 / (a - b)) + (1 / (b - c)) - (1 / (c - a)) ∧ x = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l191_19111


namespace NUMINAMATH_GPT_ratio_water_duck_to_pig_l191_19159

theorem ratio_water_duck_to_pig :
  let gallons_per_minute := 3
  let pumping_minutes := 25
  let total_gallons := gallons_per_minute * pumping_minutes
  let corn_rows := 4
  let plants_per_row := 15
  let gallons_per_corn_plant := 0.5
  let total_corn_plants := corn_rows * plants_per_row
  let total_corn_water := total_corn_plants * gallons_per_corn_plant
  let pig_count := 10
  let gallons_per_pig := 4
  let total_pig_water := pig_count * gallons_per_pig
  let duck_count := 20
  let total_duck_water := total_gallons - total_corn_water - total_pig_water
  let gallons_per_duck := total_duck_water / duck_count
  let ratio := gallons_per_duck / gallons_per_pig
  ratio = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_ratio_water_duck_to_pig_l191_19159


namespace NUMINAMATH_GPT_SugarWeightLoss_l191_19126

noncomputable def sugar_fraction_lost : Prop :=
  let green_beans_weight := 60
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_lost := (1 / 3) * rice_weight
  let remaining_weight := 120
  let total_initial_weight := green_beans_weight + rice_weight + sugar_weight
  let total_lost := total_initial_weight - remaining_weight
  let sugar_lost := total_lost - rice_lost
  let expected_fraction := (sugar_lost / sugar_weight)
  expected_fraction = (1 / 5)

theorem SugarWeightLoss : sugar_fraction_lost := by
  sorry

end NUMINAMATH_GPT_SugarWeightLoss_l191_19126


namespace NUMINAMATH_GPT_exists_fixed_point_sequence_l191_19182

theorem exists_fixed_point_sequence (N : ℕ) (hN : 0 < N) (a : ℕ → ℕ)
  (ha_conditions : ∀ i < N, a i % 2^(N+1) ≠ 0) :
  ∃ M, ∀ n ≥ M, a n = a M :=
sorry

end NUMINAMATH_GPT_exists_fixed_point_sequence_l191_19182


namespace NUMINAMATH_GPT_tan_difference_l191_19144

variable (α β : ℝ)
variable (tan_α : ℝ := 3)
variable (tan_β : ℝ := 4 / 3)

theorem tan_difference (h₁ : Real.tan α = tan_α) (h₂ : Real.tan β = tan_β) : 
  Real.tan (α - β) = (tan_α - tan_β) / (1 + tan_α * tan_β) := by
  sorry

end NUMINAMATH_GPT_tan_difference_l191_19144


namespace NUMINAMATH_GPT_toms_weekly_earnings_l191_19199

variable (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def total_money_per_week (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_per_week

theorem toms_weekly_earnings :
  total_money_per_week 8 12 5 7 = 3360 :=
by
  sorry

end NUMINAMATH_GPT_toms_weekly_earnings_l191_19199


namespace NUMINAMATH_GPT_muffins_total_is_83_l191_19104

-- Define the given conditions.
def initial_muffins : Nat := 35
def additional_muffins : Nat := 48

-- Define the total number of muffins.
def total_muffins : Nat := initial_muffins + additional_muffins

-- Statement to prove.
theorem muffins_total_is_83 : total_muffins = 83 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_muffins_total_is_83_l191_19104


namespace NUMINAMATH_GPT_geom_seq_sum_six_div_a4_minus_one_l191_19191

theorem geom_seq_sum_six_div_a4_minus_one (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = a 1 * r^n) 
  (h2 : a 1 = 1) 
  (h3 : a 2 * a 6 - 6 * a 4 - 16 = 0) :
  S 6 / (a 4 - 1) = 9 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_six_div_a4_minus_one_l191_19191


namespace NUMINAMATH_GPT_find_puppy_weight_l191_19140

noncomputable def weight_problem (a b c : ℕ) : Prop :=
  a + b + c = 36 ∧ a + c = 3 * b ∧ a + b = c + 6

theorem find_puppy_weight (a b c : ℕ) (h : weight_problem a b c) : a = 12 :=
sorry

end NUMINAMATH_GPT_find_puppy_weight_l191_19140


namespace NUMINAMATH_GPT_find_original_number_l191_19195

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l191_19195


namespace NUMINAMATH_GPT_wrongly_written_height_is_176_l191_19117

-- Definitions and given conditions
def average_height_incorrect := 182
def average_height_correct := 180
def num_boys := 35
def actual_height := 106

-- The difference in total height due to the error
def total_height_incorrect := num_boys * average_height_incorrect
def total_height_correct := num_boys * average_height_correct
def height_difference := total_height_incorrect - total_height_correct

-- The wrongly written height
def wrongly_written_height := actual_height + height_difference

-- Proof statement
theorem wrongly_written_height_is_176 : wrongly_written_height = 176 := by
  sorry

end NUMINAMATH_GPT_wrongly_written_height_is_176_l191_19117


namespace NUMINAMATH_GPT_largest_possible_value_b_l191_19189

theorem largest_possible_value_b : 
  ∃ b : ℚ, (3 * b + 7) * (b - 2) = 4 * b ∧ b = 40 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_largest_possible_value_b_l191_19189


namespace NUMINAMATH_GPT_amount_brought_by_sisters_l191_19109

-- Definitions based on conditions
def cost_per_ticket : ℕ := 8
def number_of_tickets : ℕ := 2
def change_received : ℕ := 9

-- Statement to prove
theorem amount_brought_by_sisters :
  (cost_per_ticket * number_of_tickets + change_received) = 25 :=
by
  -- Using assumptions directly
  let total_cost := cost_per_ticket * number_of_tickets
  have total_cost_eq : total_cost = 16 := by sorry
  let amount_brought := total_cost + change_received
  have amount_brought_eq : amount_brought = 25 := by sorry
  exact amount_brought_eq

end NUMINAMATH_GPT_amount_brought_by_sisters_l191_19109


namespace NUMINAMATH_GPT_grace_age_l191_19187

theorem grace_age 
  (H : ℕ) 
  (I : ℕ) 
  (J : ℕ) 
  (G : ℕ)
  (h1 : H = I - 5)
  (h2 : I = J + 7)
  (h3 : G = 2 * J)
  (h4 : H = 18) : 
  G = 32 := 
sorry

end NUMINAMATH_GPT_grace_age_l191_19187


namespace NUMINAMATH_GPT_no_rational_roots_l191_19136

theorem no_rational_roots (p q : ℤ) (h1 : p % 3 = 2) (h2 : q % 3 = 2) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ a * a = b * b * (p^2 - 4 * q) :=
by
  sorry

end NUMINAMATH_GPT_no_rational_roots_l191_19136


namespace NUMINAMATH_GPT_william_won_more_rounds_than_harry_l191_19102

def rounds_played : ℕ := 15
def william_won_rounds : ℕ := 10
def harry_won_rounds : ℕ := rounds_played - william_won_rounds
def william_won_more_rounds := william_won_rounds > harry_won_rounds

theorem william_won_more_rounds_than_harry : william_won_rounds - harry_won_rounds = 5 := 
by sorry

end NUMINAMATH_GPT_william_won_more_rounds_than_harry_l191_19102


namespace NUMINAMATH_GPT_at_least_one_gt_one_l191_19184

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : (x > 1) ∨ (y > 1) :=
sorry

end NUMINAMATH_GPT_at_least_one_gt_one_l191_19184


namespace NUMINAMATH_GPT_train_departure_time_l191_19150

theorem train_departure_time 
(distance speed : ℕ) (arrival_time_chicago difference : ℕ) (arrival_time_new_york departure_time : ℕ) 
(h_dist : distance = 480) 
(h_speed : speed = 60)
(h_arrival_chicago : arrival_time_chicago = 17) 
(h_difference : difference = 1)
(h_arrival_new_york : arrival_time_new_york = arrival_time_chicago + difference) : 
  departure_time = arrival_time_new_york - distance / speed :=
by
  sorry

end NUMINAMATH_GPT_train_departure_time_l191_19150


namespace NUMINAMATH_GPT_probability_ratio_l191_19145

-- Conditions definitions
def total_choices := Nat.choose 50 5
def p := 10 / total_choices
def q := (Nat.choose 10 2 * Nat.choose 5 2 * Nat.choose 5 3) / total_choices

-- Statement to prove
theorem probability_ratio : q / p = 450 := by
  sorry  -- proof is omitted

end NUMINAMATH_GPT_probability_ratio_l191_19145


namespace NUMINAMATH_GPT_lisa_more_dresses_than_ana_l191_19114

theorem lisa_more_dresses_than_ana :
  ∀ (total_dresses ana_dresses : ℕ),
    total_dresses = 48 →
    ana_dresses = 15 →
    (total_dresses - ana_dresses) - ana_dresses = 18 :=
by
  intros total_dresses ana_dresses h1 h2
  sorry

end NUMINAMATH_GPT_lisa_more_dresses_than_ana_l191_19114


namespace NUMINAMATH_GPT_c_finishes_work_in_18_days_l191_19138

theorem c_finishes_work_in_18_days (A B C : ℝ) 
  (h1 : A = 1 / 12) 
  (h2 : B = 1 / 9) 
  (h3 : A + B + C = 1 / 4) : 
  1 / C = 18 := 
    sorry

end NUMINAMATH_GPT_c_finishes_work_in_18_days_l191_19138


namespace NUMINAMATH_GPT_f_at_2_l191_19148

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by 
  sorry

end NUMINAMATH_GPT_f_at_2_l191_19148


namespace NUMINAMATH_GPT_Ryan_hours_learning_Spanish_is_4_l191_19149

-- Definitions based on conditions
def hoursLearningChinese : ℕ := 5
def hoursLearningSpanish := ∃ x : ℕ, hoursLearningChinese = x + 1

-- Proof Statement
theorem Ryan_hours_learning_Spanish_is_4 : ∃ x : ℕ, hoursLearningSpanish ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_Ryan_hours_learning_Spanish_is_4_l191_19149


namespace NUMINAMATH_GPT_basic_full_fare_l191_19108

theorem basic_full_fare 
  (F R : ℝ)
  (h1 : F + R = 216)
  (h2 : (F + R) + (0.5 * F + R) = 327) :
  F = 210 :=
by
  sorry

end NUMINAMATH_GPT_basic_full_fare_l191_19108


namespace NUMINAMATH_GPT_sum_of_digits_82_l191_19154

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_82 : sum_of_digits 82 = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_82_l191_19154


namespace NUMINAMATH_GPT_chocolates_sold_l191_19193

theorem chocolates_sold (C S : ℝ) (n : ℝ)
  (h1 : 65 * C = n * S)
  (h2 : S = 1.3 * C) :
  n = 50 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_sold_l191_19193


namespace NUMINAMATH_GPT_simplify_expression_l191_19169

variables (a b : ℝ)

theorem simplify_expression : 
  (2 * a^2 - 3 * a * b + 8) - (-a * b - a^2 + 8) = 3 * a^2 - 2 * a * b :=
by sorry

-- Note:
-- ℝ denotes real numbers. Adjust types accordingly if using different numerical domains (e.g., ℚ, ℂ).

end NUMINAMATH_GPT_simplify_expression_l191_19169


namespace NUMINAMATH_GPT_part1_part2_l191_19161

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) := {x : ℝ | -3 * m + 4 ≤ x ∧ x ≤ 2 * m - 1}

def p (m : ℝ) := ∀ x : ℝ, x ∈ A → x ∈ B m
def q (m : ℝ) := ∃ x : ℝ, x ∈ B m ∧ x ∈ A

theorem part1 (m : ℝ) : p m → m ≥ 4 := by
  sorry

theorem part2 (m : ℝ) : q m → m ≥ 3/2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l191_19161


namespace NUMINAMATH_GPT_surface_area_after_removing_corners_l191_19133

-- Define the dimensions of the cubes
def original_cube_side : ℝ := 4
def corner_cube_side : ℝ := 2

-- The surface area function for a cube with given side length
def surface_area (side : ℝ) : ℝ := 6 * side * side

theorem surface_area_after_removing_corners :
  surface_area original_cube_side = 96 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_after_removing_corners_l191_19133


namespace NUMINAMATH_GPT_initial_number_of_girls_l191_19137

theorem initial_number_of_girls (n : ℕ) (A : ℝ) 
  (h1 : (n + 1) * (A + 3) - 70 = n * A + 94) :
  n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_number_of_girls_l191_19137


namespace NUMINAMATH_GPT_exists_nat_n_gt_one_sqrt_expr_nat_l191_19151

theorem exists_nat_n_gt_one_sqrt_expr_nat (n : ℕ) : ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7 / 8) = m :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_n_gt_one_sqrt_expr_nat_l191_19151


namespace NUMINAMATH_GPT_solid_is_cone_l191_19198

-- Define what it means for a solid to have a given view as an isosceles triangle or a circle.
structure Solid :=
(front_view : ℝ → ℝ → Prop)
(left_view : ℝ → ℝ → Prop)
(top_view : ℝ → ℝ → Prop)

-- Definition of isosceles triangle view
def isosceles_triangle (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Definition of circle view with a center
def circle_with_center (x y : ℝ) : Prop := 
  -- not specifying details of this relationship as a placeholder
  sorry

-- Define the solid that satisfies the conditions in the problem
def specified_solid (s : Solid) : Prop :=
  (∀ x y, s.front_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.left_view x y → isosceles_triangle x y) ∧
  (∀ x y, s.top_view x y → circle_with_center x y)

-- Given proof problem statement
theorem solid_is_cone (s : Solid) (h : specified_solid s) : 
  ∃ cone, cone = s :=
sorry

end NUMINAMATH_GPT_solid_is_cone_l191_19198


namespace NUMINAMATH_GPT_quadratic_positive_difference_l191_19110
open Real

theorem quadratic_positive_difference :
  ∀ (x : ℝ), (2*x^2 - 7*x + 1 = x + 31) →
    (abs ((2 + sqrt 19) - (2 - sqrt 19)) = 2 * sqrt 19) :=
by intros x h
   sorry

end NUMINAMATH_GPT_quadratic_positive_difference_l191_19110


namespace NUMINAMATH_GPT_prob_2_lt_X_lt_4_l191_19163

noncomputable def normal_dist_p (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this computes the CDF at x for a normal distribution

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Condition: X follows a normal distribution with mean 3 and variance σ^2
axiom normal_distribution_X : ∀ x, X x = normal_dist_p 3 σ x

-- Condition: P(X ≤ 4) = 0.84
axiom prob_X_leq_4 : normal_dist_p 3 σ 4 = 0.84

-- Goal: Prove P(2 < X < 4) = 0.68
theorem prob_2_lt_X_lt_4 : normal_dist_p 3 σ 4 - normal_dist_p 3 σ 2 = 0.68 := by
  sorry

end NUMINAMATH_GPT_prob_2_lt_X_lt_4_l191_19163


namespace NUMINAMATH_GPT_shelves_for_coloring_books_l191_19121

theorem shelves_for_coloring_books (initial_stock sold donated per_shelf remaining total_used needed_shelves : ℕ) 
    (h_initial : initial_stock = 150)
    (h_sold : sold = 55)
    (h_donated : donated = 30)
    (h_per_shelf : per_shelf = 12)
    (h_total_used : total_used = sold + donated)
    (h_remaining : remaining = initial_stock - total_used)
    (h_needed_shelves : (remaining + per_shelf - 1) / per_shelf = needed_shelves) :
    needed_shelves = 6 :=
by
  sorry

end NUMINAMATH_GPT_shelves_for_coloring_books_l191_19121


namespace NUMINAMATH_GPT_three_correct_deliveries_probability_l191_19155

theorem three_correct_deliveries_probability (n : ℕ) (h1 : n = 5) :
  (∃ p : ℚ, p = 1/6 ∧ 
   (∃ choose3 : ℕ, choose3 = Nat.choose n 3) ∧ 
   (choose3 * 1/5 * 1/4 * 1/3 = p)) :=
by 
  sorry

end NUMINAMATH_GPT_three_correct_deliveries_probability_l191_19155


namespace NUMINAMATH_GPT_chromium_percentage_new_alloy_l191_19107

theorem chromium_percentage_new_alloy :
  let wA := 15
  let pA := 0.12
  let wB := 30
  let pB := 0.08
  let wC := 20
  let pC := 0.20
  let wD := 35
  let pD := 0.05
  let total_weight := wA + wB + wC + wD
  let total_chromium := (wA * pA) + (wB * pB) + (wC * pC) + (wD * pD)
  total_weight = 100 ∧ total_chromium = 9.95 → total_chromium / total_weight * 100 = 9.95 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_new_alloy_l191_19107


namespace NUMINAMATH_GPT_cos_alpha_beta_value_l191_19183

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_beta_value_l191_19183


namespace NUMINAMATH_GPT_problem_statement_l191_19168

theorem problem_statement (x : ℝ) (h : x^3 - 3 * x = 7) : x^7 + 27 * x^2 = 76 * x^2 + 270 * x + 483 :=
sorry

end NUMINAMATH_GPT_problem_statement_l191_19168


namespace NUMINAMATH_GPT_count_cubes_between_bounds_l191_19135

theorem count_cubes_between_bounds : ∃ (n : ℕ), n = 42 ∧
  ∀ x, 2^9 + 1 ≤ x^3 ∧ x^3 ≤ 2^17 + 1 ↔ 9 ≤ x ∧ x ≤ 50 := 
sorry

end NUMINAMATH_GPT_count_cubes_between_bounds_l191_19135


namespace NUMINAMATH_GPT_eggs_ordered_l191_19131

theorem eggs_ordered (E : ℕ) (h1 : E > 0) (h_crepes : E * 1 / 4 = E / 4)
                     (h_cupcakes : 2 / 3 * (3 / 4 * E) = 1 / 2 * E)
                     (h_left : (3 / 4 * E - 2 / 3 * (3 / 4 * E)) = 9) :
  E = 18 := by
  sorry

end NUMINAMATH_GPT_eggs_ordered_l191_19131


namespace NUMINAMATH_GPT_linear_function_not_third_quadrant_l191_19106

theorem linear_function_not_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ¬ (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ y = k * x + 1) :=
sorry

end NUMINAMATH_GPT_linear_function_not_third_quadrant_l191_19106


namespace NUMINAMATH_GPT_quadratic_inequality_solution_minimum_value_expression_l191_19186

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_minimum_value_expression_l191_19186


namespace NUMINAMATH_GPT_range_alpha_sub_beta_l191_19153

theorem range_alpha_sub_beta (α β : ℝ) (h₁ : -π/2 < α) (h₂ : α < β) (h₃ : β < π/2) : -π < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_GPT_range_alpha_sub_beta_l191_19153


namespace NUMINAMATH_GPT_Angle_Not_Equivalent_l191_19125

theorem Angle_Not_Equivalent (θ : ℤ) : (θ = -750) → (680 % 360 ≠ θ % 360) :=
by
  intro h
  have h1 : 680 % 360 = 320 := by norm_num
  have h2 : -750 % 360 = -30 % 360 := by norm_num
  have h3 : -30 % 360 = 330 := by norm_num
  rw [h, h2, h3]
  sorry

end NUMINAMATH_GPT_Angle_Not_Equivalent_l191_19125


namespace NUMINAMATH_GPT_minimum_passing_rate_l191_19194

-- Define the conditions as hypotheses
variable (total_students : ℕ)
variable (correct_q1 : ℕ)
variable (correct_q2 : ℕ)
variable (correct_q3 : ℕ)
variable (correct_q4 : ℕ)
variable (correct_q5 : ℕ)
variable (pass_threshold : ℕ)

-- Assume all percentages are converted to actual student counts based on total_students
axiom students_answered_q1_correctly : correct_q1 = total_students * 81 / 100
axiom students_answered_q2_correctly : correct_q2 = total_students * 91 / 100
axiom students_answered_q3_correctly : correct_q3 = total_students * 85 / 100
axiom students_answered_q4_correctly : correct_q4 = total_students * 79 / 100
axiom students_answered_q5_correctly : correct_q5 = total_students * 74 / 100
axiom passing_criteria : pass_threshold = 3

-- Define the main theorem statement to be proven
theorem minimum_passing_rate (total_students : ℕ) :
  (total_students - (total_students * 19 / 100 + total_students * 9 / 100 + 
  total_students * 15 / 100 + total_students * 21 / 100 + 
  total_students * 26 / 100) / pass_threshold) / total_students * 100 ≥ 70 :=
  by sorry

end NUMINAMATH_GPT_minimum_passing_rate_l191_19194


namespace NUMINAMATH_GPT_max_students_l191_19192

-- Define the constants for pens and pencils
def pens : ℕ := 1802
def pencils : ℕ := 1203

-- State that the GCD of pens and pencils is 1
theorem max_students : Nat.gcd pens pencils = 1 :=
by sorry

end NUMINAMATH_GPT_max_students_l191_19192


namespace NUMINAMATH_GPT_fx_greater_than_2_l191_19143

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem fx_greater_than_2 :
  ∀ x : ℝ, x > 0 → f x > 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_fx_greater_than_2_l191_19143


namespace NUMINAMATH_GPT_percentage_broken_in_second_set_l191_19146

-- Define the given conditions
def first_set_total : ℕ := 50
def first_set_broken_percent : ℚ := 0.10
def second_set_total : ℕ := 60
def total_broken : ℕ := 17

-- The proof problem statement
theorem percentage_broken_in_second_set :
  let first_set_broken := first_set_broken_percent * first_set_total
  let second_set_broken := total_broken - first_set_broken
  (second_set_broken / second_set_total) * 100 = 20 := 
sorry

end NUMINAMATH_GPT_percentage_broken_in_second_set_l191_19146
