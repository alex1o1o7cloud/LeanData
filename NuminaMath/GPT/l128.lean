import Mathlib

namespace stella_profit_loss_l128_128213

theorem stella_profit_loss :
  let dolls := 6
  let clocks := 4
  let glasses := 8
  let vases := 3
  let postcards := 10
  let dolls_price := 8
  let clocks_price := 25
  let glasses_price := 6
  let vases_price := 12
  let postcards_price := 3
  let cost := 250
  let clocks_discount_threshold := 2
  let clocks_discount := 10 / 100
  let glasses_bundle := 3
  let glasses_bundle_price := 2 * glasses_price
  let sales_tax_rate := 5 / 100
  let dolls_revenue := dolls * dolls_price
  let clocks_revenue_full := clocks * clocks_price
  let clocks_discounts_count := clocks / clocks_discount_threshold
  let clocks_discount_amount := clocks_discounts_count * clocks_discount * clocks_discount_threshold * clocks_price
  let clocks_revenue := clocks_revenue_full - clocks_discount_amount
  let glasses_discount_quantity := glasses / glasses_bundle
  let glasses_revenue := (glasses - glasses_discount_quantity) * glasses_price
  let vases_revenue := vases * vases_price
  let postcards_revenue := postcards * postcards_price
  let total_revenue_without_discounts := dolls_revenue + clocks_revenue_full + glasses_revenue + vases_revenue + postcards_revenue
  let total_revenue_with_discounts := dolls_revenue + clocks_revenue + glasses_revenue + vases_revenue + postcards_revenue
  let sales_tax := sales_tax_rate * total_revenue_with_discounts
  let profit := total_revenue_with_discounts - cost - sales_tax
  profit = -17.25 := by sorry

end stella_profit_loss_l128_128213


namespace parallel_lines_m_value_l128_128190

/-- Given two lines l_1: (3 + m) * x + 4 * y = 5 - 3 * m, and l_2: 2 * x + (5 + m) * y = 8,
the value of m for which l_1 is parallel to l_2 is -7. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
sorry

end parallel_lines_m_value_l128_128190


namespace smallest_five_digit_divisible_by_2_3_8_9_l128_128979

-- Definitions for the conditions given in the problem
def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000
def divisible_by (n d : ℕ) : Prop := d ∣ n

-- The main theorem stating the problem
theorem smallest_five_digit_divisible_by_2_3_8_9 :
  ∃ n : ℕ, is_five_digit n ∧ divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 8 ∧ divisible_by n 9 ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_2_3_8_9_l128_128979


namespace divisible_by_xyz_l128_128937

/-- 
Prove that the expression K = (x+y+z)^5 - (-x+y+z)^5 - (x-y+z)^5 - (x+y-z)^5 
is divisible by each of x, y, z.
-/
theorem divisible_by_xyz (x y z : ℝ) :
  ∃ t : ℝ, (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = t * x * y * z :=
by
  -- Proof to be provided
  sorry

end divisible_by_xyz_l128_128937


namespace possible_values_of_AC_l128_128827

theorem possible_values_of_AC (AB CD AC : ℝ) (m n : ℝ) (h1 : AB = 16) (h2 : CD = 4)
  (h3 : Set.Ioo m n = {x : ℝ | 4 < x ∧ x < 16}) : m + n = 20 :=
by
  sorry

end possible_values_of_AC_l128_128827


namespace combined_rainfall_is_23_l128_128090

-- Define the conditions
def monday_hours : ℕ := 7
def monday_rate : ℕ := 1
def tuesday_hours : ℕ := 4
def tuesday_rate : ℕ := 2
def wednesday_hours : ℕ := 2
def wednesday_rate (tuesday_rate : ℕ) : ℕ := 2 * tuesday_rate

-- Calculate the rainfalls
def monday_rainfall : ℕ := monday_hours * monday_rate
def tuesday_rainfall : ℕ := tuesday_hours * tuesday_rate
def wednesday_rainfall (wednesday_rate : ℕ) : ℕ := wednesday_hours * wednesday_rate

-- Define the total rainfall
def total_rainfall : ℕ :=
  monday_rainfall + tuesday_rainfall + wednesday_rainfall (wednesday_rate tuesday_rate)

theorem combined_rainfall_is_23 : total_rainfall = 23 := by
  -- Proof to be filled in
  sorry

end combined_rainfall_is_23_l128_128090


namespace matrix_power_identity_l128_128818

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l128_128818


namespace remaining_students_correct_l128_128421

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end remaining_students_correct_l128_128421


namespace crayon_production_correct_l128_128444

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l128_128444


namespace filling_material_heavier_than_sand_l128_128919

noncomputable def percentage_increase (full_sandbag_weight : ℝ) (partial_fill_percent : ℝ) (full_material_weight : ℝ) : ℝ :=
  let sand_weight := (partial_fill_percent / 100) * full_sandbag_weight
  let material_weight := full_material_weight
  let weight_increase := material_weight - sand_weight
  (weight_increase / sand_weight) * 100

theorem filling_material_heavier_than_sand :
  let full_sandbag_weight := 250
  let partial_fill_percent := 80
  let full_material_weight := 280
  percentage_increase full_sandbag_weight partial_fill_percent full_material_weight = 40 :=
by
  sorry

end filling_material_heavier_than_sand_l128_128919


namespace polynomial_sum_l128_128875

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l128_128875


namespace triangle_side_b_value_l128_128358

theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) (h1 : a = Real.sqrt 3) (h2 : A = 60) (h3 : C = 75) : b = Real.sqrt 2 :=
sorry

end triangle_side_b_value_l128_128358


namespace differentiable_additive_zero_derivative_l128_128496

theorem differentiable_additive_zero_derivative {f : ℝ → ℝ}
  (h1 : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_diff : Differentiable ℝ f) : 
  deriv f 0 = 0 :=
sorry

end differentiable_additive_zero_derivative_l128_128496


namespace reciprocal_inequality_l128_128562

open Real

theorem reciprocal_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a) + (1 / b) > 1 / (a + b) :=
sorry

end reciprocal_inequality_l128_128562


namespace construct_points_PQ_l128_128778

-- Given Conditions
variable (a b c : ℝ)
def triangle_ABC_conditions : Prop := 
  let s := (a + b + c) / 2
  s^2 ≥ 2 * a * b

-- Main Statement
theorem construct_points_PQ (a b c : ℝ) (P Q : ℝ) 
(h1 : triangle_ABC_conditions a b c) :
  let s := (a + b + c) / 2
  let x := (s + Real.sqrt (s^2 - 2 * a * b)) / 2
  let y := (s - Real.sqrt (s^2 - 2 * a * b)) / 2
  x + y = s ∧ x * y = (a * b) / 2 :=
by
  sorry

end construct_points_PQ_l128_128778


namespace koby_boxes_l128_128327

theorem koby_boxes (x : ℕ) (sparklers_per_box : ℕ := 3) (whistlers_per_box : ℕ := 5) 
    (cherie_sparklers : ℕ := 8) (cherie_whistlers : ℕ := 9) (total_fireworks : ℕ := 33) : 
    (sparklers_per_box * x + cherie_sparklers) + (whistlers_per_box * x + cherie_whistlers) = total_fireworks → x = 2 :=
by
  sorry

end koby_boxes_l128_128327


namespace largest_non_factor_product_of_factors_of_100_l128_128682

theorem largest_non_factor_product_of_factors_of_100 :
  ∃ x y : ℕ, 
  (x ≠ y) ∧ 
  (0 < x ∧ 0 < y) ∧ 
  (x ∣ 100 ∧ y ∣ 100) ∧ 
  ¬(x * y ∣ 100) ∧ 
  (∀ a b : ℕ, 
    (a ≠ b) ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (a ∣ 100 ∧ b ∣ 100) ∧ 
    ¬(a * b ∣ 100) → 
    (x * y) ≥ (a * b)) ∧ 
  (x * y) = 40 :=
by
  sorry

end largest_non_factor_product_of_factors_of_100_l128_128682


namespace which_set_forms_triangle_l128_128900

def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem which_set_forms_triangle : 
  satisfies_triangle_inequality 4 3 6 ∧ 
  ¬ satisfies_triangle_inequality 1 2 3 ∧ 
  ¬ satisfies_triangle_inequality 7 8 16 ∧ 
  ¬ satisfies_triangle_inequality 9 10 20 :=
by
  sorry

end which_set_forms_triangle_l128_128900


namespace initial_percentage_water_is_80_l128_128398

noncomputable def initial_kola_solution := 340
noncomputable def added_sugar := 3.2
noncomputable def added_water := 10
noncomputable def added_kola := 6.8
noncomputable def final_percentage_sugar := 14.111111111111112
noncomputable def percentage_kola := 6

theorem initial_percentage_water_is_80 :
  ∃ (W : ℝ), W = 80 :=
by
  sorry

end initial_percentage_water_is_80_l128_128398


namespace ajax_store_price_l128_128434

theorem ajax_store_price (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ)
    (h_original: original_price = 180)
    (h_first_discount : first_discount_rate = 0.5)
    (h_second_discount : second_discount_rate = 0.2) :
    let first_discount_price := original_price * (1 - first_discount_rate)
    let saturday_price := first_discount_price * (1 - second_discount_rate)
    saturday_price = 72 :=
by
    sorry

end ajax_store_price_l128_128434


namespace find_b_l128_128460

theorem find_b (b : ℝ) (h : ∃ (f_inv : ℝ → ℝ), (∀ x y, f_inv (2^x + b) = y) ∧ f_inv 5 = 2) :
    b = 1 := by
  sorry

end find_b_l128_128460


namespace triangle_DEF_all_acute_l128_128840

theorem triangle_DEF_all_acute
  (α : ℝ)
  (hα : 0 < α ∧ α < 90)
  (DEF : Type)
  (D : DEF) (E : DEF) (F : DEF)
  (angle_DFE : DEF → DEF → DEF → ℝ) 
  (angle_FED : DEF → DEF → DEF → ℝ) 
  (angle_EFD : DEF → DEF → DEF → ℝ)
  (h1 : angle_DFE D F E = 45)
  (h2 : angle_FED F E D = 90 - α / 2)
  (h3 : angle_EFD E D F = 45 + α / 2) :
  (0 < angle_DFE D F E ∧ angle_DFE D F E < 90) ∧ 
  (0 < angle_FED F E D ∧ angle_FED F E D < 90) ∧ 
  (0 < angle_EFD E D F ∧ angle_EFD E D F < 90) := by
  sorry

end triangle_DEF_all_acute_l128_128840


namespace total_time_correct_l128_128878

-- Definitions based on problem conditions
def first_time : ℕ := 15
def time_increment : ℕ := 7
def number_of_flights : ℕ := 7

-- Time taken for a specific flight
def time_for_nth_flight (n : ℕ) : ℕ := first_time + (n - 1) * time_increment

-- Sum of the times for the first seven flights
def total_time : ℕ := (number_of_flights * (first_time + time_for_nth_flight number_of_flights)) / 2

-- Statement to be proven
theorem total_time_correct : total_time = 252 := 
by
  sorry

end total_time_correct_l128_128878


namespace ratio_c_d_l128_128142

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
  (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end ratio_c_d_l128_128142


namespace triangle_area_is_correct_l128_128417

noncomputable def isosceles_triangle_area : Prop :=
  let side_large_square := 6 -- sides of the large square WXYZ
  let area_large_square := side_large_square * side_large_square
  let side_small_square := 2 -- sides of the smaller squares
  let BC := side_large_square - 2 * side_small_square -- length of BC
  let height_AM := side_large_square / 2 + side_small_square -- height of the triangle from A to M
  let area_ABC := (BC * height_AM) / 2 -- area of the triangle ABC
  area_large_square = 36 ∧ BC = 2 ∧ height_AM = 5 ∧ area_ABC = 5

theorem triangle_area_is_correct : isosceles_triangle_area := sorry

end triangle_area_is_correct_l128_128417


namespace square_area_l128_128802

theorem square_area (x : ℝ) (G H : ℝ) (hyp_1 : 0 ≤ G) (hyp_2 : G ≤ x) (hyp_3 : 0 ≤ H) (hyp_4 : H ≤ x) (AG : ℝ) (GH : ℝ) (HD : ℝ)
  (hyp_5 : AG = 20) (hyp_6 : GH = 20) (hyp_7 : HD = 20) (hyp_8 : x = 20 * Real.sqrt 2) :
  x^2 = 800 :=
by
  sorry

end square_area_l128_128802


namespace part_a_part_b_l128_128588

def P (m n : ℕ) : ℕ := m^2003 * n^2017 - m^2017 * n^2003

theorem part_a (m n : ℕ) : P m n % 24 = 0 := 
by sorry

theorem part_b : ∃ (m n : ℕ), P m n % 7 ≠ 0 :=
by sorry

end part_a_part_b_l128_128588


namespace geometric_diff_l128_128977

-- Definitions based on conditions
def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ (d2 * d2 = d1 * d3)

-- Problem statement
theorem geometric_diff :
  let largest_geometric := 964
  let smallest_geometric := 124
  is_geometric largest_geometric ∧ is_geometric smallest_geometric ∧
  (largest_geometric - smallest_geometric = 840) :=
by
  sorry

end geometric_diff_l128_128977


namespace monotonic_intervals_range_of_a_l128_128288

noncomputable def f (x a : ℝ) := Real.log x + (a / 2) * x^2 - (a + 1) * x
noncomputable def f' (x a : ℝ) := 1 / x + a * x - (a + 1)

theorem monotonic_intervals (a : ℝ) (ha : f 1 a = -2 ∧ f' 1 a = 0):
  (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f' x a > 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a > 0) ∧ 
  (∀ x : ℝ, (1 / 2) < x ∧ x < 1 → f' x a < 0) := sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℕ, x > 0 → (f x a) / x < (f' x a) / 2):
  a > 2 * Real.exp (- (3 / 2)) - 1 := sorry

end monotonic_intervals_range_of_a_l128_128288


namespace graph_of_equation_l128_128815

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_equation_l128_128815


namespace percentage_error_in_side_measurement_l128_128281

theorem percentage_error_in_side_measurement :
  (forall (S S' : ℝ) (A A' : ℝ), 
    A = S^2 ∧ A' = S'^2 ∧ (A' - A) / A * 100 = 25.44 -> 
    (S' - S) / S * 100 = 12.72) :=
by
  intros S S' A A' h
  sorry

end percentage_error_in_side_measurement_l128_128281


namespace sqrt_225_eq_15_l128_128649

theorem sqrt_225_eq_15 : Real.sqrt 225 = 15 :=
sorry

end sqrt_225_eq_15_l128_128649


namespace trapezium_other_side_l128_128268

theorem trapezium_other_side (x : ℝ) :
  1/2 * (20 + x) * 10 = 150 → x = 10 :=
by
  sorry

end trapezium_other_side_l128_128268


namespace find_a_l128_128093

-- Given conditions
def div_by_3 (a : ℤ) : Prop :=
  (5 * a + 1) % 3 = 0 ∨ (3 * a + 2) % 3 = 0

def div_by_5 (a : ℤ) : Prop :=
  (5 * a + 1) % 5 = 0 ∨ (3 * a + 2) % 5 = 0

-- Proving the question 
theorem find_a (a : ℤ) : div_by_3 a ∧ div_by_5 a → a % 15 = 4 :=
by
  sorry

end find_a_l128_128093


namespace Marty_combinations_l128_128747

def unique_combinations (colors techniques : ℕ) : ℕ :=
  colors * techniques

theorem Marty_combinations :
  unique_combinations 6 5 = 30 := by
  sorry

end Marty_combinations_l128_128747


namespace distinct_ball_placement_l128_128936

def num_distributions (balls boxes : ℕ) : ℕ :=
  if boxes = 3 then 243 - 32 + 16 else 0

theorem distinct_ball_placement : num_distributions 5 3 = 227 :=
by
  sorry

end distinct_ball_placement_l128_128936


namespace second_statue_weight_l128_128338

theorem second_statue_weight (S : ℕ) :
  ∃ S : ℕ,
    (80 = 10 + S + 15 + 15 + 22) → S = 18 :=
by
  sorry

end second_statue_weight_l128_128338


namespace inequality_l128_128263

theorem inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a / (b.sqrt) + b / (a.sqrt)) ≥ (a.sqrt + b.sqrt) :=
by
  sorry

end inequality_l128_128263


namespace sum_of_products_l128_128278

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  ab + bc + ca = 5 :=
by 
  sorry

end sum_of_products_l128_128278


namespace resistor_problem_l128_128172

theorem resistor_problem 
  {x y r : ℝ}
  (h1 : 1 / r = 1 / x + 1 / y)
  (h2 : r = 2.9166666666666665)
  (h3 : y = 7) : 
  x = 5 :=
by
  sorry

end resistor_problem_l128_128172


namespace least_possible_value_of_D_l128_128846

-- Defining the conditions as theorems
theorem least_possible_value_of_D :
  ∃ (A B C D : ℕ), 
  (A + B + C + D) / 4 = 18 ∧
  A = 3 * B ∧
  B = C - 2 ∧
  C = 3 / 2 * D ∧
  (∀ x : ℕ, x ≥ 10 → D = x) := 
sorry

end least_possible_value_of_D_l128_128846


namespace opposite_of_neg_five_l128_128882

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l128_128882


namespace find_common_ratio_l128_128393

-- Declare the sequence and conditions
variables {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions of the problem 
def positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ m n : ℕ, a m = a 0 * q ^ m) ∧ q > 0

def third_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + a 5 = 5

def fifth_term_seventh_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 5 + a 7 = 20

-- The final lean statement proving the common ratio is 2
theorem find_common_ratio 
  (h1 : positive_geometric_sequence a q) 
  (h2 : third_term_condition a q) 
  (h3 : fifth_term_seventh_term_condition a q) : 
  q = 2 :=
sorry

end find_common_ratio_l128_128393


namespace smith_family_mean_age_l128_128126

theorem smith_family_mean_age :
  let children_ages := [8, 8, 8, 12, 11]
  let dogs_ages := [3, 4]
  let all_ages := children_ages ++ dogs_ages
  let total_ages := List.sum all_ages
  let total_individuals := List.length all_ages
  (total_ages : ℚ) / (total_individuals : ℚ) = 7.71 :=
by
  sorry

end smith_family_mean_age_l128_128126


namespace equivalent_angle_terminal_side_l128_128317

theorem equivalent_angle_terminal_side (k : ℤ) (a : ℝ) (c : ℝ) (d : ℝ) : a = -3/10 * Real.pi → c = a * 180 / Real.pi → d = c + 360 * k →
   ∃ k : ℤ, d = 306 :=
sorry

end equivalent_angle_terminal_side_l128_128317


namespace cost_per_book_l128_128949

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l128_128949


namespace find_second_game_points_l128_128512

-- Define Clayton's points for respective games
def first_game_points := 10
def third_game_points := 6

-- Define the points in the second game as P
variable (P : ℕ)

-- Define the points in the fourth game based on the average of first three games
def fourth_game_points := (first_game_points + P + third_game_points) / 3

-- Define the total points over four games
def total_points := first_game_points + P + third_game_points + fourth_game_points

-- Based on the total points, prove P = 14
theorem find_second_game_points (P : ℕ) (h : total_points P = 40) : P = 14 :=
  by
    sorry

end find_second_game_points_l128_128512


namespace number_of_bead_necklaces_sold_is_3_l128_128156

-- Definitions of the given conditions
def total_earnings : ℕ := 36
def gemstone_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 6

-- Define the earnings from gemstone necklaces as a separate definition
def earnings_gemstone_necklaces : ℕ := gemstone_necklaces * cost_per_necklace

-- Define the earnings from bead necklaces based on total earnings and earnings from gemstone necklaces
def earnings_bead_necklaces : ℕ := total_earnings - earnings_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_bead_necklaces / cost_per_necklace

-- The theorem we want to prove
theorem number_of_bead_necklaces_sold_is_3 : bead_necklaces_sold = 3 :=
by
  sorry

end number_of_bead_necklaces_sold_is_3_l128_128156


namespace sum_of_given_geom_series_l128_128466

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l128_128466


namespace avg_of_multiples_l128_128066

theorem avg_of_multiples (n : ℝ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n + 6 * n + 7 * n + 8 * n + 9 * n + 10 * n) / 10 = 60.5) : n = 11 :=
by
  sorry

end avg_of_multiples_l128_128066


namespace min_value_eq_18sqrt3_l128_128170

noncomputable def min_value (x y : ℝ) (h : x + y = 5) : ℝ := 3^x + 3^y

theorem min_value_eq_18sqrt3 {x y : ℝ} (h : x + y = 5) : min_value x y h ≥ 18 * Real.sqrt 3 := 
sorry

end min_value_eq_18sqrt3_l128_128170


namespace portion_of_profit_divided_equally_l128_128472

-- Definitions for the given conditions
def total_investment_mary : ℝ := 600
def total_investment_mike : ℝ := 400
def total_profit : ℝ := 7500
def profit_diff : ℝ := 1000

-- Main statement
theorem portion_of_profit_divided_equally (E P : ℝ) 
  (h1 : total_profit = E + P)
  (h2 : E + (3/5) * P = E + (2/5) * P + profit_diff) :
  E = 2500 :=
by
  sorry

end portion_of_profit_divided_equally_l128_128472


namespace minimize_travel_time_l128_128932

theorem minimize_travel_time
  (a b c d : ℝ)
  (v₁ v₂ v₃ v₄ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : v₁ > v₂)
  (h5 : v₂ > v₃)
  (h6 : v₃ > v₄) : 
  (a / v₁ + b / v₂ + c / v₃ + d / v₄) ≤ (a / v₁ + b / v₄ + c / v₃ + d / v₂) :=
sorry

end minimize_travel_time_l128_128932


namespace triangle_is_right_angled_l128_128269

-- Define the internal angles of a triangle
variables (A B C : ℝ)
-- Condition: A, B, C are internal angles of a triangle
-- This directly implies 0 < A, B, C < pi and A + B + C = pi

-- Internal angles of a triangle sum to π
axiom angles_sum_pi : A + B + C = Real.pi

-- Condition given in the problem
axiom sin_condition : Real.sin A = Real.sin C * Real.cos B

-- We need to prove that triangle ABC is right-angled
theorem triangle_is_right_angled : C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_angled_l128_128269


namespace min_value_of_quadratic_l128_128314

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end min_value_of_quadratic_l128_128314


namespace find_pairs_l128_128706

def regions_divided (h s : ℕ) : ℕ :=
  1 + s * (s + 1) / 2 + h * (s + 1)

theorem find_pairs (h s : ℕ) :
  regions_divided h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  sorry

end find_pairs_l128_128706


namespace jenny_eggs_in_each_basket_l128_128695

theorem jenny_eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 45 % n = 0) (h3 : n ≥ 5) : n = 15 :=
sorry

end jenny_eggs_in_each_basket_l128_128695


namespace initial_number_of_friends_is_six_l128_128441

theorem initial_number_of_friends_is_six
  (car_cost : ℕ)
  (car_wash_earnings : ℕ)
  (F : ℕ)
  (additional_cost_when_one_friend_leaves : ℕ)
  (h1 : car_cost = 1700)
  (h2 : car_wash_earnings = 500)
  (remaining_cost := car_cost - car_wash_earnings)
  (cost_per_friend_before := remaining_cost / F)
  (cost_per_friend_after := remaining_cost / (F - 1))
  (h3 : additional_cost_when_one_friend_leaves = 40)
  (h4 : cost_per_friend_after = cost_per_friend_before + additional_cost_when_one_friend_leaves) :
  F = 6 :=
by
  sorry

end initial_number_of_friends_is_six_l128_128441


namespace candy_bar_cost_l128_128451

theorem candy_bar_cost (initial_amount change : ℕ) (h : initial_amount = 50) (hc : change = 5) : 
  initial_amount - change = 45 :=
by
  -- sorry is used to skip the proof
  sorry

end candy_bar_cost_l128_128451


namespace find_a_l128_128186

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end find_a_l128_128186


namespace largest_rectangle_in_circle_l128_128304

theorem largest_rectangle_in_circle {r : ℝ} (h : r = 6) : 
  ∃ A : ℝ, A = 72 := 
by 
  sorry

end largest_rectangle_in_circle_l128_128304


namespace sum_of_gcd_and_lcm_of_180_and_4620_l128_128683

def gcd_180_4620 : ℕ := Nat.gcd 180 4620
def lcm_180_4620 : ℕ := Nat.lcm 180 4620
def sum_gcd_lcm_180_4620 : ℕ := gcd_180_4620 + lcm_180_4620

theorem sum_of_gcd_and_lcm_of_180_and_4620 :
  sum_gcd_lcm_180_4620 = 13920 :=
by
  sorry

end sum_of_gcd_and_lcm_of_180_and_4620_l128_128683


namespace find_phi_monotone_interval_1_monotone_interval_2_l128_128390

-- Definitions related to the function f
noncomputable def f (x φ a : ℝ) : ℝ :=
  Real.sin (x + φ) + a * Real.cos x

-- Problem Part 1: Given f(π/2) = √2 / 2, find φ
theorem find_phi (a : ℝ) (φ : ℝ) (h : |φ| < Real.pi / 2) (hf : f (π / 2) φ a = Real.sqrt 2 / 2) :
  φ = π / 4 ∨ φ = -π / 4 :=
  sorry

-- Problem Part 2 Condition 1: Given a = √3, φ = -π/3, find the monotonically increasing interval
theorem monotone_interval_1 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-5 * π / 6) + 2 * k * π) ≤ x ∧ x ≤ (π / 6 + 2 * k * π) → 
  f x (-π / 3) (Real.sqrt 3) = Real.sin (x + π / 3) :=
  sorry

-- Problem Part 2 Condition 2: Given a = -1, φ = π/6, find the monotonically increasing interval
theorem monotone_interval_2 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-π / 3) + 2 * k * π) ≤ x ∧ x ≤ ((2 * π / 3) + 2 * k * π) → 
  f x (π / 6) (-1) = Real.sin (x - π / 6) :=
  sorry

end find_phi_monotone_interval_1_monotone_interval_2_l128_128390


namespace solve_y_from_expression_l128_128643

-- Define the conditions
def given_conditions := (784 = 28^2) ∧ (49 = 7^2)

-- Define the equivalency to prove based on the given conditions
theorem solve_y_from_expression (h : given_conditions) : 784 + 2 * 28 * 7 + 49 = 1225 := by
  sorry

end solve_y_from_expression_l128_128643


namespace proof_star_ast_l128_128303

noncomputable def star (a b : ℕ) : ℕ := sorry  -- representing binary operation for star
noncomputable def ast (a b : ℕ) : ℕ := sorry  -- representing binary operation for ast

theorem proof_star_ast :
  star 12 2 * ast 9 3 = 2 →
  (star 7 3 * ast 12 6) = 7 / 6 :=
by
  sorry

end proof_star_ast_l128_128303


namespace hexagon_AF_length_l128_128104

theorem hexagon_AF_length (BC CD DE EF : ℝ) (angleB angleC angleD angleE : ℝ) (angleF : ℝ) 
  (hBC : BC = 2) (hCD : CD = 2) (hDE : DE = 2) (hEF : EF = 2)
  (hangleB : angleB = 135) (hangleC : angleC = 135) (hangleD : angleD = 135) (hangleE : angleE = 135)
  (hangleF : angleF = 90) :
  ∃ (a b : ℝ), (AF = a + 2 * Real.sqrt b) ∧ (a + b = 6) :=
by
  sorry

end hexagon_AF_length_l128_128104


namespace prob_at_least_one_wrong_l128_128168

-- Defining the conditions in mathlib
def prob_wrong : ℝ := 0.1
def num_questions : ℕ := 3

-- Proving the main statement
theorem prob_at_least_one_wrong : 1 - (1 - prob_wrong) ^ num_questions = 0.271 := by
  sorry

end prob_at_least_one_wrong_l128_128168


namespace find_A_of_trig_max_bsquared_plus_csquared_l128_128850

-- Given the geometric conditions and trigonometric identities.

-- Prove: Given 2a * sin B = b * tan A, we have A = π / 3
theorem find_A_of_trig (a b c A B C : Real) (h1 : 2 * a * Real.sin B = b * Real.tan A) :
  A = Real.pi / 3 := sorry

-- Prove: Given a = 2, the maximum value of b^2 + c^2 is 8
theorem max_bsquared_plus_csquared (a b c A : Real) (hA : A = Real.pi / 3) (ha : a = 2) :
  b^2 + c^2 ≤ 8 :=
by
  have hcos : Real.cos A = 1 / 2 := by sorry
  have h : 4 = b^2 + c^2 - b * c * (1/2) := by sorry
  have hmax : b^2 + c^2 + b * c ≤ 8 := by sorry
  sorry -- Proof steps to reach the final result

end find_A_of_trig_max_bsquared_plus_csquared_l128_128850


namespace arithmetic_progression_cubic_eq_l128_128770

theorem arithmetic_progression_cubic_eq (x y z u : ℤ) (d : ℤ) :
  (x, y, z, u) = (3 * d, 4 * d, 5 * d, 6 * d) →
  x^3 + y^3 + z^3 = u^3 →
  ∃ d : ℤ, x = 3 * d ∧ y = 4 * d ∧ z = 5 * d ∧ u = 6 * d :=
by sorry

end arithmetic_progression_cubic_eq_l128_128770


namespace ribbon_fraction_per_box_l128_128923

theorem ribbon_fraction_per_box 
  (total_ribbon_used : ℚ)
  (number_of_boxes : ℕ)
  (h1 : total_ribbon_used = 5/8)
  (h2 : number_of_boxes = 5) :
  (total_ribbon_used / number_of_boxes = 1/8) :=
by
  sorry

end ribbon_fraction_per_box_l128_128923


namespace probability_of_exactly_one_second_class_product_l128_128484

-- Definitions based on the conditions provided
def total_products := 100
def first_class_products := 90
def second_class_products := 10
def selected_products := 4

-- Calculation of the probability
noncomputable def probability : ℚ :=
  (Nat.choose 10 1 * Nat.choose 90 3) / Nat.choose 100 4

-- Statement to prove that the probability is 0.30
theorem probability_of_exactly_one_second_class_product : 
  probability = 0.30 := by
  sorry

end probability_of_exactly_one_second_class_product_l128_128484


namespace same_terminal_side_l128_128427

theorem same_terminal_side (α : ℝ) (k : ℤ) (h : α = -51) : 
  ∃ (m : ℤ), α + m * 360 = k * 360 - 51 :=
by {
    sorry
}

end same_terminal_side_l128_128427


namespace ellipse_focal_length_l128_128907

theorem ellipse_focal_length {m : ℝ} : 
  (m > 2 ∧ 4 ≤ 10 - m ∧ 4 ≤ m - 2) → 
  (10 - m - (m - 2) = 4) ∨ (m - 2 - (10 - m) = 4) :=
by
  sorry

end ellipse_focal_length_l128_128907


namespace isosceles_triangle_base_angle_l128_128721

theorem isosceles_triangle_base_angle (α : ℕ) (base_angle : ℕ) 
  (hα : α = 40) (hsum : α + 2 * base_angle = 180) : 
  base_angle = 70 :=
sorry

end isosceles_triangle_base_angle_l128_128721


namespace polynomial_divisibility_a_l128_128136

theorem polynomial_divisibility_a (n : ℕ) : 
  (n % 3 = 1 ∨ n % 3 = 2) ↔ (x^2 + x + 1 ∣ x^(2*n) + x^n + 1) :=
sorry

end polynomial_divisibility_a_l128_128136


namespace sum_of_numbers_odd_probability_l128_128082

namespace ProbabilityProblem

/-- 
  Given a biased die where the probability of rolling an even number is 
  twice the probability of rolling an odd number, and rolling the die three times,
  the probability that the sum of the numbers rolled is odd is 13/27.
-/
theorem sum_of_numbers_odd_probability :
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let prob_all_odd := (p_odd) ^ 3
  let prob_one_odd_two_even := 3 * (p_odd) * (p_even) ^ 2
  prob_all_odd + prob_one_odd_two_even = 13 / 27 :=
by
  sorry

end sum_of_numbers_odd_probability_l128_128082


namespace sample_size_120_l128_128014

theorem sample_size_120
  (x y : ℕ)
  (h_ratio : x / 2 = y / 3 ∧ y / 3 = 60 / 5)
  (h_max : max x (max y 60) = 60) :
  x + y + 60 = 120 := by
  sorry

end sample_size_120_l128_128014


namespace total_bathing_suits_l128_128877

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969

theorem total_bathing_suits : men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l128_128877


namespace yellow_balls_are_24_l128_128943

theorem yellow_balls_are_24 (x y z : ℕ) (h1 : x + y + z = 68) 
                             (h2 : y = 2 * x) (h3 : 3 * z = 4 * y) : y = 24 :=
by
  sorry

end yellow_balls_are_24_l128_128943


namespace apples_picked_correct_l128_128917

-- Define the conditions as given in the problem
def apples_given_to_Melanie : ℕ := 27
def apples_left : ℕ := 16

-- Define the problem statement
def total_apples_picked := apples_given_to_Melanie + apples_left

-- Prove that the total apples picked is equal to 43 given the conditions
theorem apples_picked_correct : total_apples_picked = 43 := by
  sorry

end apples_picked_correct_l128_128917


namespace mod_remainder_of_expression_l128_128624

theorem mod_remainder_of_expression : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end mod_remainder_of_expression_l128_128624


namespace hannah_total_payment_l128_128530

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l128_128530


namespace carol_weight_l128_128050

variable (a c : ℝ)

theorem carol_weight (h1 : a + c = 240) (h2 : c - a = (2 / 3) * c) : c = 180 :=
by
  sorry

end carol_weight_l128_128050


namespace percent_difference_l128_128334

variables (w q y z x : ℝ)

-- Given conditions
def cond1 : Prop := w = 0.60 * q
def cond2 : Prop := q = 0.60 * y
def cond3 : Prop := z = 0.54 * y
def cond4 : Prop := x = 1.30 * w

-- The proof problem
theorem percent_difference (h1 : cond1 w q)
                           (h2 : cond2 q y)
                           (h3 : cond3 z y)
                           (h4 : cond4 x w) :
  ((z - x) / w) * 100 = 20 :=
by
  sorry

end percent_difference_l128_128334


namespace quadratic_function_properties_l128_128532

def quadratic_function (x : ℝ) : ℝ :=
  -6 * x^2 + 36 * x - 48

theorem quadratic_function_properties :
  quadratic_function 2 = 0 ∧ quadratic_function 4 = 0 ∧ quadratic_function 3 = 6 :=
by
  -- The proof is omitted
  -- Placeholder for the proof
  sorry

end quadratic_function_properties_l128_128532


namespace problem1_problem2_l128_128694

noncomputable def f (x a : ℝ) := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x (-1) ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 :=
sorry

theorem problem2 (a : ℝ) : (∀ x, f x a ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end problem1_problem2_l128_128694


namespace no_negative_product_l128_128330

theorem no_negative_product (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) 
(h1 : x ^ (2 * n) - y ^ (2 * n) > x) (h2 : y ^ (2 * n) - x ^ (2 * n) > y) : x * y ≥ 0 :=
sorry

end no_negative_product_l128_128330


namespace smallest_four_digit_int_mod_9_l128_128970

theorem smallest_four_digit_int_mod_9 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 9 = 5 → n ≤ m :=
sorry

end smallest_four_digit_int_mod_9_l128_128970


namespace jacob_initial_fish_count_l128_128189

theorem jacob_initial_fish_count : 
  ∃ J : ℕ, 
    (∀ A : ℕ, A = 7 * J) → 
    (A' = A - 23) → 
    (J + 26 = A' + 1) → 
    J = 8 := 
by 
  sorry

end jacob_initial_fish_count_l128_128189


namespace james_ride_time_l128_128255

theorem james_ride_time (distance speed : ℝ) (h_distance : distance = 200) (h_speed : speed = 25) : distance / speed = 8 :=
by
  rw [h_distance, h_speed]
  norm_num

end james_ride_time_l128_128255


namespace number_of_integer_pairs_satisfying_conditions_l128_128883

noncomputable def count_integer_pairs (n m : ℕ) : ℕ := Nat.choose (n-1) (m-1)

theorem number_of_integer_pairs_satisfying_conditions :
  ∃ (a b c x y : ℕ), a + b + c = 55 ∧ a + b + c + x + y = 71 ∧ x + y > a + b + c → count_integer_pairs 55 3 * count_integer_pairs 16 2 = 21465 := sorry

end number_of_integer_pairs_satisfying_conditions_l128_128883


namespace inequality_pos_real_l128_128031

theorem inequality_pos_real (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ (2 / 3) := 
sorry

end inequality_pos_real_l128_128031


namespace max_value_AMC_l128_128143

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 15) : 
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 := 
sorry

end max_value_AMC_l128_128143


namespace circle_radius_l128_128786

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 180 * π) : r = 10 := 
by
  sorry

end circle_radius_l128_128786


namespace correct_exponent_calculation_l128_128180

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation_l128_128180


namespace total_games_played_l128_128227

-- Defining the conditions
def games_won : ℕ := 18
def games_lost : ℕ := games_won + 21

-- Problem statement
theorem total_games_played : games_won + games_lost = 57 := by
  sorry

end total_games_played_l128_128227


namespace circle_center_radius_sum_18_l128_128065

-- Conditions from the problem statement
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * y - 9 = -y^2 + 18 * x + 9

-- Goal is to prove a + b + r = 18
theorem circle_center_radius_sum_18 :
  (∃ a b r : ℝ, 
     (∀ x y : ℝ, circle_eq x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
     a + b + r = 18) :=
sorry

end circle_center_radius_sum_18_l128_128065


namespace area_of_square_with_diagonal_40_l128_128395

theorem area_of_square_with_diagonal_40 {d : ℝ} (h : d = 40) : ∃ A : ℝ, A = 800 :=
by
  sorry

end area_of_square_with_diagonal_40_l128_128395


namespace simplify_expression_l128_128894

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) :=
by sorry

end simplify_expression_l128_128894


namespace max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l128_128505

structure BusConfig where
  rows_section1 : ℕ
  seats_per_row_section1 : ℕ
  rows_section2 : ℕ
  seats_per_row_section2 : ℕ
  total_seats : ℕ
  max_children : ℕ

def typeA : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 4,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 40 }

def typeB : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 4,
    rows_section2 := 6,
    seats_per_row_section2 := 5,
    total_seats := 54,
    max_children := 50 }

def typeC : BusConfig :=
  { rows_section1 := 8,
    seats_per_row_section1 := 4,
    rows_section2 := 2,
    seats_per_row_section2 := 2,
    total_seats := 36,
    max_children := 35 }

def typeD : BusConfig :=
  { rows_section1 := 6,
    seats_per_row_section1 := 3,
    rows_section2 := 6,
    seats_per_row_section2 := 3,
    total_seats := 36,
    max_children := 30 }

theorem max_children_typeA : min typeA.total_seats typeA.max_children = 36 := by
  sorry

theorem max_children_typeB : min typeB.total_seats typeB.max_children = 50 := by
  sorry

theorem max_children_typeC : min typeC.total_seats typeC.max_children = 35 := by
  sorry

theorem max_children_typeD : min typeD.total_seats typeD.max_children = 30 := by
  sorry

end max_children_typeA_max_children_typeB_max_children_typeC_max_children_typeD_l128_128505


namespace least_four_digit_perfect_square_and_cube_l128_128845

theorem least_four_digit_perfect_square_and_cube :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ m1 : ℕ, n = m1^2) ∧ (∃ m2 : ℕ, n = m2^3) ∧ n = 4096 := sorry

end least_four_digit_perfect_square_and_cube_l128_128845


namespace smallest_square_value_l128_128641

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (r s : ℕ) (hr : 15 * a + 16 * b = r^2) (hs : 16 * a - 15 * b = s^2) :
  min (r^2) (s^2) = 481^2 :=
  sorry

end smallest_square_value_l128_128641


namespace lesser_fraction_solution_l128_128767

noncomputable def lesser_fraction (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) : ℚ :=
  if x ≤ y then x else y

theorem lesser_fraction_solution (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) :
  lesser_fraction x y h₁ h₂ = (7 - Real.sqrt 17) / 16 := by
  sorry

end lesser_fraction_solution_l128_128767


namespace incorrect_major_premise_l128_128768

noncomputable def Line := Type
noncomputable def Plane := Type

-- Conditions: Definitions
variable (b a : Line) (α : Plane)

-- Assumption: Line b is parallel to Plane α
axiom parallel_to_plane (p : Line) (π : Plane) : Prop

-- Assumption: Line a is in Plane α
axiom line_in_plane (l : Line) (π : Plane) : Prop

-- Define theorem stating the incorrect major premise
theorem incorrect_major_premise 
  (hb_par_α : parallel_to_plane b α)
  (ha_in_α : line_in_plane a α) : ¬ (parallel_to_plane b α → ∀ l, line_in_plane l α → b = l) := 
sorry

end incorrect_major_premise_l128_128768


namespace truck_capacities_transportation_plan_l128_128542

-- Definitions of given conditions
def A_truck_capacity (x y : ℕ) : Prop := x + 2 * y = 50
def B_truck_capacity (x y : ℕ) : Prop := 5 * x + 4 * y = 160
def total_transport_cost (m n : ℕ) : ℕ := 500 * m + 400 * n
def most_cost_effective_plan (m n cost : ℕ) : Prop := 
  m + 2 * n = 10 ∧ (20 * m + 15 * n = 190) ∧ cost = total_transport_cost m n ∧ cost = 4800

-- Proving the capacities of trucks A and B
theorem truck_capacities : 
  ∃ x y : ℕ, A_truck_capacity x y ∧ B_truck_capacity x y ∧ x = 20 ∧ y = 15 := 
sorry

-- Proving the most cost-effective transportation plan
theorem transportation_plan : 
  ∃ m n cost, (total_transport_cost m n = cost) ∧ most_cost_effective_plan m n cost := 
sorry

end truck_capacities_transportation_plan_l128_128542


namespace BANANA_arrangement_l128_128650

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l128_128650


namespace exists_k_for_any_n_l128_128110

theorem exists_k_for_any_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, 2 * k^2 + 2001 * k + 3 ≡ 0 [MOD 2^n] :=
sorry

end exists_k_for_any_n_l128_128110


namespace problem_statement_l128_128146

theorem problem_statement (h : 36 = 6^2) : 6^15 / 36^5 = 7776 := by
  sorry

end problem_statement_l128_128146


namespace number_of_jars_pasta_sauce_l128_128205

-- Conditions
def pasta_cost_per_kg := 1.5
def pasta_weight_kg := 2.0
def ground_beef_cost_per_kg := 8.0
def ground_beef_weight_kg := 1.0 / 4.0
def quesadilla_cost := 6.0
def jar_sauce_cost := 2.0
def total_money := 15.0

-- Helper definitions for total costs
def pasta_total_cost := pasta_weight_kg * pasta_cost_per_kg
def ground_beef_total_cost := ground_beef_weight_kg * ground_beef_cost_per_kg
def other_total_cost := quesadilla_cost + pasta_total_cost + ground_beef_total_cost
def remaining_money := total_money - other_total_cost

-- Proof statement
theorem number_of_jars_pasta_sauce :
  (remaining_money / jar_sauce_cost) = 2 := by
  sorry

end number_of_jars_pasta_sauce_l128_128205


namespace maximize_revenue_l128_128585

-- Defining the revenue function
def revenue (p : ℝ) : ℝ := 200 * p - 4 * p^2

-- Defining the maximum price constraint
def price_constraint (p : ℝ) : Prop := p ≤ 40

-- Statement to be proven
theorem maximize_revenue : ∃ (p : ℝ), price_constraint p ∧ revenue p = 2500 ∧ (∀ q : ℝ, price_constraint q → revenue q ≤ revenue p) :=
sorry

end maximize_revenue_l128_128585


namespace ball_hits_ground_time_l128_128983

theorem ball_hits_ground_time (h : ℝ → ℝ) (t : ℝ) :
  (∀ (t : ℝ), h t = -16 * t ^ 2 - 30 * t + 200) → h t = 0 → t = 2.5 :=
by
  -- Placeholder for the formal proof
  sorry

end ball_hits_ground_time_l128_128983


namespace sum_div_minuend_eq_two_l128_128068

variable (Subtrahend Minuend Difference : ℝ)

theorem sum_div_minuend_eq_two
  (h : Subtrahend + Difference = Minuend) :
  (Subtrahend + Minuend + Difference) / Minuend = 2 :=
by
  sorry

end sum_div_minuend_eq_two_l128_128068


namespace avoid_loss_maximize_profit_max_profit_per_unit_l128_128603

-- Definitions of the functions as per problem conditions
noncomputable def C (x : ℝ) : ℝ := 2 + x
noncomputable def R (x : ℝ) : ℝ := if x ≤ 4 then 4 * x - (1 / 2) * x^2 - (1 / 2) else 7.5
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Proof statements

-- 1. Range to avoid loss
theorem avoid_loss (x : ℝ) : 1 ≤ x ∧ x ≤ 5.5 ↔ L x ≥ 0 :=
by
  sorry

-- 2. Production to maximize profit
theorem maximize_profit (x : ℝ) : x = 3 ↔ ∀ y, L y ≤ L 3 :=
by
  sorry

-- 3. Maximum profit per unit selling price
theorem max_profit_per_unit (x : ℝ) : x = 3 ↔ (R 3 / 3 = 2.33) :=
by
  sorry

end avoid_loss_maximize_profit_max_profit_per_unit_l128_128603


namespace value_of_expression_l128_128272

theorem value_of_expression (x : ℝ) (h : (3 / (x - 3)) + (5 / (2 * x - 6)) = 11 / 2) : 2 * x - 6 = 2 :=
sorry

end value_of_expression_l128_128272


namespace platform_length_is_500_l128_128893

-- Define the length of the train, the time to cross a tree, and the time to cross a platform as given conditions
def train_length := 1500 -- in meters
def time_to_cross_tree := 120 -- in seconds
def time_to_cross_platform := 160 -- in seconds

-- Define the speed based on the train crossing the tree
def train_speed := train_length / time_to_cross_tree -- in meters/second

-- Define the total distance covered when crossing the platform
def total_distance_crossing_platform (platform_length : ℝ) := train_length + platform_length

-- State the main theorem to prove the platform length is 500 meters
theorem platform_length_is_500 (platform_length : ℝ) :
  (train_speed * time_to_cross_platform = total_distance_crossing_platform platform_length) → platform_length = 500 :=
by
  sorry

end platform_length_is_500_l128_128893


namespace root_in_interval_l128_128499

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval : ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
  sorry

end root_in_interval_l128_128499


namespace product_defect_rate_correct_l128_128339

-- Definitions for the defect rates of the stages
def defect_rate_stage1 : ℝ := 0.10
def defect_rate_stage2 : ℝ := 0.03

-- Definitions for the probability of passing each stage without defects
def pass_rate_stage1 : ℝ := 1 - defect_rate_stage1
def pass_rate_stage2 : ℝ := 1 - defect_rate_stage2

-- Definition for the overall probability of a product not being defective
def pass_rate_overall : ℝ := pass_rate_stage1 * pass_rate_stage2

-- Definition for the overall defect rate based on the above probabilities
def defect_rate_product : ℝ := 1 - pass_rate_overall

-- The theorem statement to be proved
theorem product_defect_rate_correct : defect_rate_product = 0.127 :=
by
  -- Proof here
  sorry

end product_defect_rate_correct_l128_128339


namespace problem_l128_128471

def f (x : ℝ) : ℝ := sorry -- We assume f is defined as per the given condition but do not provide an implementation.

theorem problem (h : ∀ x : ℝ, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry -- The proof is omitted

end problem_l128_128471


namespace sqrt_polynomial_eq_l128_128640

variable (a b c : ℝ)

def polynomial := 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2

theorem sqrt_polynomial_eq (a b c : ℝ) : 
  (polynomial a b c) ^ (1 / 2) = (2 * a - 3 * b + 4 * c) :=
by
  sorry

end sqrt_polynomial_eq_l128_128640


namespace olivia_race_time_l128_128812

theorem olivia_race_time (total_time : ℕ) (time_difference : ℕ) (olivia_time : ℕ)
  (h1 : total_time = 112) (h2 : time_difference = 4) (h3 : olivia_time + (olivia_time - time_difference) = total_time) :
  olivia_time = 58 :=
by
  sorry

end olivia_race_time_l128_128812


namespace series_sum_eq_one_fourth_l128_128555

noncomputable def sum_series : ℝ :=
  ∑' n, (3 ^ n / (1 + 3 ^ n + 3 ^ (n + 2) + 3 ^ (2 * n + 2)))

theorem series_sum_eq_one_fourth :
  sum_series = 1 / 4 :=
by
  sorry

end series_sum_eq_one_fourth_l128_128555


namespace platform_length_l128_128733

theorem platform_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (platform_length : ℝ) :
  train_length = 150 ∧ speed_kmph = 75 ∧ time_sec = 20 →
  platform_length = 1350 :=
by
  sorry

end platform_length_l128_128733


namespace sum_of_exponents_correct_l128_128959

-- Define the initial expression
def original_expr (a b c : ℤ) : ℤ := 40 * a^6 * b^9 * c^14

-- Define the simplified expression outside the radical
def simplified_outside_expr (a b c : ℤ) : ℤ := a * b^3 * c^3

-- Define the sum of the exponents
def sum_of_exponents : ℕ := 1 + 3 + 3

-- Prove that the given conditions lead to the sum of the exponents being 7
theorem sum_of_exponents_correct (a b c : ℤ) :
  original_expr a b c = 40 * a^6 * b^9 * c^14 →
  simplified_outside_expr a b c = a * b^3 * c^3 →
  sum_of_exponents = 7 :=
by
  intros
  -- Proof goes here
  sorry

end sum_of_exponents_correct_l128_128959


namespace domain_of_function_is_all_real_l128_128133

def domain_function : Prop :=
  ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 6 ≠ 0

theorem domain_of_function_is_all_real :
  domain_function :=
by
  intros t
  sorry

end domain_of_function_is_all_real_l128_128133


namespace baseball_cards_per_pack_l128_128260

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end baseball_cards_per_pack_l128_128260


namespace olaf_travels_miles_l128_128870

-- Define the given conditions
def men : ℕ := 25
def per_day_water_per_man : ℚ := 1 / 2
def boat_mileage_per_day : ℕ := 200
def total_water : ℚ := 250

-- Define the daily water consumption for the crew
def daily_water_consumption : ℚ := men * per_day_water_per_man

-- Define the number of days the water will last
def days_water_lasts : ℚ := total_water / daily_water_consumption

-- Define the total miles traveled
def total_miles_traveled : ℚ := days_water_lasts * boat_mileage_per_day

-- Theorem statement to prove the total miles traveled is 4000 miles
theorem olaf_travels_miles : total_miles_traveled = 4000 := by
  sorry

end olaf_travels_miles_l128_128870


namespace problem_part1_problem_part2_l128_128697

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem problem_part1 :
  f (Real.pi / 12) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

theorem problem_part2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  Real.sin θ = 4 / 5 →
  f (5 * Real.pi / 12 - θ) = 72 / 25 :=
by
  sorry

end problem_part1_problem_part2_l128_128697


namespace max_value_expression_l128_128673

theorem max_value_expression (θ : ℝ) : 
  2 ≤ 5 + 3 * Real.sin θ ∧ 5 + 3 * Real.sin θ ≤ 8 → 
  (∃ θ, (14 / (5 + 3 * Real.sin θ)) = 7) := 
sorry

end max_value_expression_l128_128673


namespace scientific_notation_correct_l128_128587

-- Define the given condition
def average_daily_users : ℝ := 2590000

-- The proof problem
theorem scientific_notation_correct :
  average_daily_users = 2.59 * 10^6 :=
sorry

end scientific_notation_correct_l128_128587


namespace ab_sum_l128_128392

theorem ab_sum (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 :=
by
  sorry -- this is where the proof would go

end ab_sum_l128_128392


namespace square_of_number_ending_in_5_l128_128011

theorem square_of_number_ending_in_5 (a : ℤ) :
  (10 * a + 5) * (10 * a + 5) = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_number_ending_in_5_l128_128011


namespace length_of_second_race_l128_128813

theorem length_of_second_race :
  ∀ (V_A V_B V_C T T' L : ℝ),
  (V_A * T = 200) →
  (V_B * T = 180) →
  (V_C * T = 162) →
  (V_B * T' = L) →
  (V_C * T' = L - 60) →
  (L = 600) :=
by
  intros V_A V_B V_C T T' L h1 h2 h3 h4 h5
  sorry

end length_of_second_race_l128_128813


namespace range_of_f_l128_128639

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt (5 + 4 * Real.cos x))

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := 
sorry

end range_of_f_l128_128639


namespace min_value_p_plus_q_l128_128200

theorem min_value_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) 
  (h : 17 * (p + 1) = 20 * (q + 1)) : p + q = 37 :=
sorry

end min_value_p_plus_q_l128_128200


namespace angle_terminal_side_l128_128538

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * (180 / Real.pi)

theorem angle_terminal_side :
  ∃ k : ℤ, rad_to_deg (π / 12) + 360 * k = 375 :=
sorry

end angle_terminal_side_l128_128538


namespace slope_of_PQ_l128_128824

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

end slope_of_PQ_l128_128824


namespace evaluate_expression_l128_128248

theorem evaluate_expression : (831 * 831) - (830 * 832) = 1 :=
by
  sorry

end evaluate_expression_l128_128248


namespace power_of_power_rule_l128_128214

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end power_of_power_rule_l128_128214


namespace min_value_of_fraction_l128_128950

noncomputable def min_val (a b : ℝ) : ℝ :=
  1 / a + 2 * b

theorem min_value_of_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 * a * b + 3 = b) :
  min_val a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_l128_128950


namespace angle_ratio_l128_128578

-- Definitions as per the conditions
def bisects (x y z : ℝ) : Prop := x = y / 2
def trisects (x y z : ℝ) : Prop := y = x / 3

theorem angle_ratio (ABC PBQ BM x : ℝ) (h1 : bisects PBQ ABC PQ)
                                    (h2 : trisects PBQ BM M) :
  PBQ = 2 * x →
  PBQ = ABC / 2 →
  MBQ = x →
  ABQ = 4 * x →
  MBQ / ABQ = 1 / 4 :=
by
  intros
  sorry

end angle_ratio_l128_128578


namespace find_circle_equation_l128_128502

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the equation of the asymptote
def asymptote (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0

-- Define the given center of the circle
def center : ℝ × ℝ :=
  (5, 0)

-- Define the radius of the circle
def radius : ℝ :=
  4

-- Define the circle in center-radius form and expand it to standard form
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 9 = 0

theorem find_circle_equation 
  (x y : ℝ) 
  (h : asymptote x y)
  (h_center : (x, y) = center) 
  (h_radius : radius = 4) : circle_eq x y :=
sorry

end find_circle_equation_l128_128502


namespace factorize_expression_find_xy_l128_128566

-- Problem 1: Factorizing the quadratic expression
theorem factorize_expression (x : ℝ) : 
  x^2 - 120 * x + 3456 = (x - 48) * (x - 72) :=
sorry

-- Problem 2: Finding the product xy from the given equation
theorem find_xy (x y : ℝ) (h : x^2 + y^2 + 8 * x - 12 * y + 52 = 0) : 
  x * y = -24 :=
sorry

end factorize_expression_find_xy_l128_128566


namespace greatest_integer_x_l128_128096

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l128_128096


namespace find_triangle_value_l128_128292

theorem find_triangle_value 
  (triangle : ℕ)
  (h_units : (triangle + 3) % 7 = 2)
  (h_tens : (1 + 4 + triangle) % 7 = 4)
  (h_hundreds : (2 + triangle + 1) % 7 = 2)
  (h_thousands : 3 + 0 + 1 = 4) :
  triangle = 6 :=
sorry

end find_triangle_value_l128_128292


namespace balloons_initial_count_l128_128155

theorem balloons_initial_count (B : ℕ) (G : ℕ) : ∃ G : ℕ, B = 7 * G + 4 := sorry

end balloons_initial_count_l128_128155


namespace compute_expression_l128_128521

theorem compute_expression : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 :=
by
  sorry

end compute_expression_l128_128521


namespace problem_inequality_l128_128428

theorem problem_inequality (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + (1 / b)) * (b - 1 + (1 / c)) * (c - 1 + (1 / a)) ≤ 1 :=
sorry

end problem_inequality_l128_128428


namespace find_m_plus_n_l128_128261

def operation (m n : ℕ) : ℕ := m^n + m * n

theorem find_m_plus_n :
  ∃ (m n : ℕ), (2 ≤ m) ∧ (2 ≤ n) ∧ (operation m n = 64) ∧ (m + n = 6) :=
by {
  -- Begin the proof context
  sorry
}

end find_m_plus_n_l128_128261


namespace color_opposite_lightgreen_is_red_l128_128158

-- Define the colors
inductive Color
| Red | White | Green | Brown | LightGreen | Purple

open Color

-- Define the condition
def is_opposite (a b : Color) : Prop := sorry

-- Main theorem
theorem color_opposite_lightgreen_is_red :
  is_opposite LightGreen Red :=
sorry

end color_opposite_lightgreen_is_red_l128_128158


namespace minimum_value_N_div_a4_possible_values_a4_l128_128121

noncomputable def lcm_10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) : ℕ := 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a1 a2) a3) a4) a5) a6) a7) a8) a9) a10

theorem minimum_value_N_div_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10) : 
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 := sorry

theorem possible_values_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10)
  (z: 1 ≤ a4 ∧ a4 ≤ 1300) :
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 → a4 = 360 ∨ a4 = 720 ∨ a4 = 1080 := sorry

end minimum_value_N_div_a4_possible_values_a4_l128_128121


namespace rolls_sold_to_grandmother_l128_128424

theorem rolls_sold_to_grandmother (t u n s g : ℕ) 
  (h1 : t = 45)
  (h2 : u = 10)
  (h3 : n = 6)
  (h4 : s = 28)
  (total_sold : t - s = g + u + n) : 
  g = 1 := 
  sorry

end rolls_sold_to_grandmother_l128_128424


namespace malcolm_initial_white_lights_l128_128614

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l128_128614


namespace tanks_difference_l128_128318

theorem tanks_difference (total_tanks german_tanks allied_tanks sanchalian_tanks : ℕ)
  (h_total : total_tanks = 115)
  (h_german_allied : german_tanks = 2 * allied_tanks + 2)
  (h_allied_sanchalian : allied_tanks = 3 * sanchalian_tanks + 1)
  (h_total_eq : german_tanks + allied_tanks + sanchalian_tanks = total_tanks) :
  german_tanks - sanchalian_tanks = 59 :=
sorry

end tanks_difference_l128_128318


namespace relationship_among_values_l128_128514

-- Define the properties of the function f
variables (f : ℝ → ℝ)

-- Assume necessary conditions
axiom domain_of_f : ∀ x : ℝ, f x ≠ 0 -- Domain of f is ℝ
axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_function : ∀ x y : ℝ, (0 ≤ x) → (x ≤ y) → (f x ≤ f y) -- f is increasing for x in [0, + ∞)

-- Define the main theorem based on the problem statement
theorem relationship_among_values : f π > f (-3) ∧ f (-3) > f (-2) :=
by
  sorry

end relationship_among_values_l128_128514


namespace third_number_lcm_l128_128836

theorem third_number_lcm (n : ℕ) :
  n ∣ 360 ∧ lcm (lcm 24 36) n = 360 →
  n = 5 :=
by sorry

end third_number_lcm_l128_128836


namespace max_value_of_a_l128_128744

theorem max_value_of_a :
  ∃ b : ℤ, ∃ (a : ℝ), 
    (a = 30285) ∧
    (a * b^2 / (a + 2 * b) = 2019) :=
by
  sorry

end max_value_of_a_l128_128744


namespace loss_per_metre_l128_128779

def total_metres : ℕ := 500
def selling_price : ℕ := 18000
def cost_price_per_metre : ℕ := 41

theorem loss_per_metre :
  (cost_price_per_metre * total_metres - selling_price) / total_metres = 5 :=
by sorry

end loss_per_metre_l128_128779


namespace bus_problem_l128_128064

theorem bus_problem (x : ℕ)
  (h1 : 28 + 82 - x = 30) :
  82 - x = 2 :=
by {
  sorry
}

end bus_problem_l128_128064


namespace prob_defective_l128_128887

/-- Assume there are two boxes of components. 
    The first box contains 10 pieces, including 2 defective ones; 
    the second box contains 20 pieces, including 3 defective ones. --/
def box1_total : ℕ := 10
def box1_defective : ℕ := 2
def box2_total : ℕ := 20
def box2_defective : ℕ := 3

/-- Randomly select one box from the two boxes, 
    and then randomly pick 1 component from that box. --/
def prob_select_box : ℚ := 1 / 2

/-- Probability of selecting a defective component given that box 1 was selected. --/
def prob_defective_given_box1 : ℚ := box1_defective / box1_total

/-- Probability of selecting a defective component given that box 2 was selected. --/
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

/-- The probability of selecting a defective component is 7/40. --/
theorem prob_defective :
  prob_select_box * prob_defective_given_box1 + prob_select_box * prob_defective_given_box2 = 7 / 40 :=
sorry

end prob_defective_l128_128887


namespace sally_lost_two_balloons_l128_128052

-- Condition: Sally originally had 9 orange balloons.
def original_orange_balloons := 9

-- Condition: Sally now has 7 orange balloons.
def current_orange_balloons := 7

-- Problem: Prove that Sally lost 2 orange balloons.
theorem sally_lost_two_balloons : original_orange_balloons - current_orange_balloons = 2 := by
  sorry

end sally_lost_two_balloons_l128_128052


namespace right_triangle_num_array_l128_128663

theorem right_triangle_num_array (n : ℕ) (hn : 0 < n) 
    (a : ℕ → ℕ → ℝ) 
    (h1 : a 1 1 = 1/4)
    (hd : ∀ i j, 0 < j → j <= i → a (i+1) 1 = a i 1 + 1/4)
    (hq : ∀ i j, 2 < i → 0 < j → j ≤ i → a i (j+1) = a i j * (1/2)) :
  a n 3 = n / 16 := 
by 
  sorry

end right_triangle_num_array_l128_128663


namespace workshop_average_salary_l128_128543

theorem workshop_average_salary :
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  A = 8000 :=
by
  -- Definitions according to given conditions
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  -- We need to show that A = 8000
  show A = 8000
  sorry

end workshop_average_salary_l128_128543


namespace misha_grade_students_l128_128229

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l128_128229


namespace fraction_equality_l128_128539

def op_at (a b : ℕ) : ℕ := a * b + b^2
def op_hash (a b : ℕ) : ℕ := a + b + a * (b^2)

theorem fraction_equality : (op_at 5 3 : ℚ) / (op_hash 5 3 : ℚ) = 24 / 53 := 
by 
  sorry

end fraction_equality_l128_128539


namespace sum_of_extreme_values_l128_128139

theorem sum_of_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (5 - Real.sqrt 34) / 3
  let M := (5 + Real.sqrt 34) / 3
  m + M = 10 / 3 :=
by
  sorry

end sum_of_extreme_values_l128_128139


namespace balls_per_pack_l128_128996

theorem balls_per_pack (total_packs total_cost cost_per_ball total_balls balls_per_pack : ℕ)
  (h1 : total_packs = 4)
  (h2 : total_cost = 24)
  (h3 : cost_per_ball = 2)
  (h4 : total_balls = total_cost / cost_per_ball)
  (h5 : total_balls = 12)
  (h6 : balls_per_pack = total_balls / total_packs) :
  balls_per_pack = 3 := by 
  sorry

end balls_per_pack_l128_128996


namespace product_of_largest_and_second_largest_l128_128841

theorem product_of_largest_and_second_largest (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  (max (max a b) c * (max (min a (max b c)) (min b (max a c)))) = 132 :=
by
  sorry

end product_of_largest_and_second_largest_l128_128841


namespace total_copies_to_save_40_each_l128_128095

-- Definitions for the conditions.
def cost_per_copy : ℝ := 0.02
def discount_rate : ℝ := 0.25
def min_copies_for_discount : ℕ := 100
def savings_required : ℝ := 0.40
def steve_copies : ℕ := 80
def dinley_copies : ℕ := 80

-- Lean 4 statement to prove the total number of copies 
-- to save $0.40 each.
theorem total_copies_to_save_40_each : 
  (steve_copies + dinley_copies) + 
  (savings_required / (cost_per_copy * discount_rate)) * 2 = 320 :=
by 
  sorry

end total_copies_to_save_40_each_l128_128095


namespace hyperbola_eccentricity_ratio_hyperbola_condition_l128_128746

-- Part (a)
theorem hyperbola_eccentricity_ratio
  (a b c : ℝ) (h1 : c^2 = a^2 + b^2)
  (x0 y0 : ℝ) 
  (P : ℝ × ℝ) (h2 : P = (x0, y0))
  (F : ℝ × ℝ) (h3 : F = (c, 0))
  (D : ℝ) (h4 : D = a^2 / c)
  (d_PF : ℝ) (h5 : d_PF = ( (x0 - c)^2 + y0^2 )^(1/2))
  (d_PD : ℝ) (h6 : d_PD = |x0 - a^2 / c|)
  (e : ℝ) (h7 : e = c / a) :
  d_PF / d_PD = e :=
sorry

-- Part (b)
theorem hyperbola_condition
  (F_l : ℝ × ℝ) (h1 : F_l = (0, k))
  (X_l : ℝ × ℝ) (h2 : X_l = (x, l))
  (d_XF : ℝ) (h3 : d_XF = (x^2 + y^2)^(1/2))
  (d_Xl : ℝ) (h4 : d_Xl = |x - k|)
  (e : ℝ) (h5 : e > 1)
  (h6 : d_XF / d_Xl = e) :
  ∃ a b : ℝ, (x / a)^2 - (y / b)^2 = 1 :=
sorry

end hyperbola_eccentricity_ratio_hyperbola_condition_l128_128746


namespace numbers_not_all_less_than_six_l128_128536

theorem numbers_not_all_less_than_six (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) :=
sorry

end numbers_not_all_less_than_six_l128_128536


namespace arithmetic_square_root_of_4_l128_128781

theorem arithmetic_square_root_of_4 : ∃ y : ℝ, y^2 = 4 ∧ y = 2 := 
  sorry

end arithmetic_square_root_of_4_l128_128781


namespace final_position_D_l128_128920

open Function

-- Define the original points of the parallelogram
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (9, 4)
def D : ℝ × ℝ := (7, 0)

-- Define the reflection across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the translation by (0, 1)
def translate_up (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)
def translate_down (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

-- Define the reflection across y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combine the transformations to get the final reflection across y = x - 1
def reflect_across_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_down (reflect_y_eq_x (translate_up p))

-- Prove that the final position of D after the two transformations is (1, -8)
theorem final_position_D'' : reflect_across_y_eq_x_minus_1 (reflect_y_axis D) = (1, -8) :=
  sorry

end final_position_D_l128_128920


namespace fraction_simplification_l128_128848

theorem fraction_simplification : 
  (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := 
by 
  sorry

end fraction_simplification_l128_128848


namespace distance_between_trees_l128_128414

theorem distance_between_trees (yard_length : ℕ) (number_of_trees : ℕ) (number_of_gaps : ℕ)
  (h1 : yard_length = 400) (h2 : number_of_trees = 26) (h3 : number_of_gaps = number_of_trees - 1) :
  yard_length / number_of_gaps = 16 := by
  sorry

end distance_between_trees_l128_128414


namespace sixth_term_geometric_sequence_l128_128243

theorem sixth_term_geometric_sequence (a r : ℚ) (h_a : a = 16) (h_r : r = 1/2) : 
  a * r^(5) = 1/2 :=
by 
  rw [h_a, h_r]
  sorry

end sixth_term_geometric_sequence_l128_128243


namespace brianna_fraction_left_l128_128795

theorem brianna_fraction_left (m n c : ℕ) (h : (1 : ℚ) / 4 * m = 1 / 2 * n * c) : 
  (m - (n * c) - (1 / 10 * m)) / m = 2 / 5 :=
by
  sorry

end brianna_fraction_left_l128_128795


namespace new_container_volume_l128_128289

theorem new_container_volume (original_volume : ℕ) (factor : ℕ) (new_volume : ℕ) 
    (h1 : original_volume = 5) (h2 : factor = 4 * 4 * 4) : new_volume = 320 :=
by
  sorry

end new_container_volume_l128_128289


namespace cost_of_acai_berry_juice_l128_128148

theorem cost_of_acai_berry_juice 
  (cost_per_litre_cocktail : ℝ) 
  (cost_per_litre_mixed_fruit : ℝ)
  (volume_mixed_fruit : ℝ)
  (volume_acai_berry : ℝ)
  (total_volume : ℝ) 
  (total_cost_of_mixed_fruit : ℝ)
  (total_cost_cocktail : ℝ)
  : cost_per_litre_cocktail = 1399.45 ∧ 
    cost_per_litre_mixed_fruit = 262.85 ∧ 
    volume_mixed_fruit = 37 ∧ 
    volume_acai_berry = 24.666666666666668 ∧ 
    total_volume = 61.666666666666668 ∧ 
    total_cost_of_mixed_fruit = volume_mixed_fruit * cost_per_litre_mixed_fruit ∧
    total_cost_of_mixed_fruit = 9725.45 ∧
    total_cost_cocktail = total_volume * cost_per_litre_cocktail ∧ 
    total_cost_cocktail = 86327.77 
    → 24.666666666666668 * 3105.99 + 9725.45 = 86327.77 :=
sorry

end cost_of_acai_berry_juice_l128_128148


namespace optimal_tablet_combination_exists_l128_128715

/-- Define the daily vitamin requirement structure --/
structure Vitamins (A B C D : ℕ)

theorem optimal_tablet_combination_exists {x y : ℕ} :
  (∃ (x y : ℕ), 
    (3 * x ≥ 3) ∧ (x + y ≥ 9) ∧ (x + 3 * y ≥ 15) ∧ (2 * y ≥ 2) ∧
    (x + y = 9) ∧ 
    (20 * x + 60 * y = 3) ∧ 
    (x + 2 * y = 12) ∧ 
    (x = 6 ∧ y = 3)) := 
  by
  sorry

end optimal_tablet_combination_exists_l128_128715


namespace find_second_half_profit_l128_128291

variable (P : ℝ)
variable (profit_difference total_annual_profit : ℝ)
variable (h_difference : profit_difference = 2750000)
variable (h_total : total_annual_profit = 3635000)

theorem find_second_half_profit (h_eq : P + (P + profit_difference) = total_annual_profit) : 
  P = 442500 :=
by
  rw [h_difference, h_total] at h_eq
  sorry

end find_second_half_profit_l128_128291


namespace greatest_groups_of_stuffed_animals_l128_128074

def stuffed_animals_grouping : Prop :=
  let cats := 26
  let dogs := 14
  let bears := 18
  let giraffes := 22
  gcd (gcd (gcd cats dogs) bears) giraffes = 2

theorem greatest_groups_of_stuffed_animals : stuffed_animals_grouping :=
by sorry

end greatest_groups_of_stuffed_animals_l128_128074


namespace favorite_movies_total_hours_l128_128881

theorem favorite_movies_total_hours (michael_hrs joyce_hrs nikki_hrs ryn_hrs sam_hrs alex_hrs : ℕ)
  (H1 : nikki_hrs = 30)
  (H2 : michael_hrs = nikki_hrs / 3)
  (H3 : joyce_hrs = michael_hrs + 2)
  (H4 : ryn_hrs = (4 * nikki_hrs) / 5)
  (H5 : sam_hrs = (3 * joyce_hrs) / 2)
  (H6 : alex_hrs = 2 * michael_hrs) :
  michael_hrs + joyce_hrs + nikki_hrs + ryn_hrs + sam_hrs + alex_hrs = 114 := 
sorry

end favorite_movies_total_hours_l128_128881


namespace problem_m_n_sum_l128_128271

theorem problem_m_n_sum (m n : ℕ) 
  (h1 : m^2 + n^2 = 3789) 
  (h2 : Nat.gcd m n + Nat.lcm m n = 633) : 
  m + n = 87 :=
sorry

end problem_m_n_sum_l128_128271


namespace ratio_of_ages_l128_128161

open Real

theorem ratio_of_ages (father_age son_age : ℝ) (h1 : father_age = 45) (h2 : son_age = 15) :
  father_age / son_age = 3 :=
by
  sorry

end ratio_of_ages_l128_128161


namespace part_a_part_b_case1_part_b_case2_l128_128233

theorem part_a (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x1 / x2 + x2 / x1 = -9 / 4) : 
  p = -1 / 23 :=
sorry

theorem part_b_case1 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -3 / 8 :=
sorry

theorem part_b_case2 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -15 / 8 :=
sorry

end part_a_part_b_case1_part_b_case2_l128_128233


namespace calculate_outlet_requirements_l128_128527

def outlets_needed := 10
def suites_outlets_needed := 15
def num_standard_rooms := 50
def num_suites := 10
def type_a_percentage := 0.40
def type_b_percentage := 0.60
def type_c_percentage := 1.0

noncomputable def total_outlets_needed := 500 + 150
noncomputable def type_a_outlets_needed := 0.40 * 500
noncomputable def type_b_outlets_needed := 0.60 * 500
noncomputable def type_c_outlets_needed := 150

theorem calculate_outlet_requirements :
  total_outlets_needed = 650 ∧
  type_a_outlets_needed = 200 ∧
  type_b_outlets_needed = 300 ∧
  type_c_outlets_needed = 150 :=
by
  sorry

end calculate_outlet_requirements_l128_128527


namespace gcd_7_nplus2_8_2nplus1_l128_128872

theorem gcd_7_nplus2_8_2nplus1 : 
  ∃ d : ℕ, (∀ n : ℕ, d ∣ (7^(n+2) + 8^(2*n+1))) ∧ (∀ n : ℕ, d = 57) :=
sorry

end gcd_7_nplus2_8_2nplus1_l128_128872


namespace find_s2_length_l128_128984

variables (s r : ℝ)

def condition1 : Prop := 2 * r + s = 2420
def condition2 : Prop := 2 * r + 3 * s = 4040

theorem find_s2_length (h1 : condition1 s r) (h2 : condition2 s r) : s = 810 :=
sorry

end find_s2_length_l128_128984


namespace factorizations_of_4050_l128_128915

theorem factorizations_of_4050 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4050 :=
by
  sorry

end factorizations_of_4050_l128_128915


namespace contradiction_proof_l128_128862

theorem contradiction_proof (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : 0 < c ∧ c < 2) :
  ¬ (a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) :=
sorry

end contradiction_proof_l128_128862


namespace find_abs_xyz_l128_128061

noncomputable def conditions_and_question (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1)

theorem find_abs_xyz (x y z : ℝ) (h : conditions_and_question x y z) : |x * y * z| = 1 :=
  sorry

end find_abs_xyz_l128_128061


namespace problem_l128_128520

-- Definition of triangular number
def is_triangular (n k : ℕ) := n = k * (k + 1) / 2

-- Definition of choosing 2 marbles
def choose_2 (n m : ℕ) := n = m * (m - 1) / 2

-- Definition of Cathy's condition
def cathy_condition (n s : ℕ) := s * s < 2 * n ∧ 2 * n - s * s = 20

theorem problem (n k m s : ℕ) :
  is_triangular n k →
  choose_2 n m →
  cathy_condition n s →
  n = 210 :=
by
  sorry

end problem_l128_128520


namespace closest_fraction_to_medals_won_l128_128769

theorem closest_fraction_to_medals_won :
  let won_ratio : ℚ := 35 / 225
  let choices : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]
  (closest : ℚ) = 1 / 6 → 
  (closest_in_choices : closest ∈ choices) →
  ∀ choice ∈ choices, abs ((7 / 45) - (1 / 6)) ≤ abs ((7 / 45) - choice) :=
by
  let won_ratio := 7 / 45
  let choices := [1/5, 1/6, 1/7, 1/8, 1/9]
  let closest := 1 / 6
  have closest_in_choices : closest ∈ choices := sorry
  intro choice h_choice_in_choices
  sorry

end closest_fraction_to_medals_won_l128_128769


namespace wendy_albums_used_l128_128059

def total_pictures : ℕ := 45
def pictures_in_one_album : ℕ := 27
def pictures_per_album : ℕ := 2

theorem wendy_albums_used :
  let remaining_pictures := total_pictures - pictures_in_one_album
  let albums_used := remaining_pictures / pictures_per_album
  albums_used = 9 :=
by
  sorry

end wendy_albums_used_l128_128059


namespace enlarged_decal_height_l128_128374

theorem enlarged_decal_height (original_width original_height new_width : ℕ)
  (original_width_eq : original_width = 3)
  (original_height_eq : original_height = 2)
  (new_width_eq : new_width = 15)
  (proportions_consistent : ∀ h : ℕ, new_width * original_height = original_width * h) :
  ∃ new_height, new_height = 10 :=
by sorry

end enlarged_decal_height_l128_128374


namespace quadrilateral_perpendicular_diagonals_l128_128821

theorem quadrilateral_perpendicular_diagonals
  (AB BC CD DA : ℝ)
  (m n : ℝ)
  (hAB : AB = 6)
  (hBC : BC = m)
  (hCD : CD = 8)
  (hDA : DA = n)
  (h_diagonals_perpendicular : true)
  : m^2 + n^2 = 100 := 
by
  sorry

end quadrilateral_perpendicular_diagonals_l128_128821


namespace jellybean_count_l128_128129

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l128_128129


namespace output_correct_l128_128758

-- Definitions derived from the conditions
def initial_a : Nat := 3
def initial_b : Nat := 4

-- Proof that the final output of PRINT a, b is (4, 4)
theorem output_correct : 
  let a := initial_a;
  let b := initial_b;
  let a := b;
  let b := a;
  (a, b) = (4, 4) :=
by
  sorry

end output_correct_l128_128758


namespace line_passes_second_and_third_quadrants_l128_128191

theorem line_passes_second_and_third_quadrants 
  (a b c p : ℝ)
  (h1 : a * b * c ≠ 0)
  (h2 : (a + b) / c = p)
  (h3 : (b + c) / a = p)
  (h4 : (c + a) / b = p) :
  ∀ (x y : ℝ), y = p * x + p → 
  ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
sorry

end line_passes_second_and_third_quadrants_l128_128191


namespace circle_radius_tangent_l128_128921

theorem circle_radius_tangent (a : ℝ) (R : ℝ) (h1 : a = 25)
  (h2 : ∀ BP DE CP CE, BP = 2 ∧ DE = 2 ∧ CP = 23 ∧ CE = 23 ∧ BP + CP = a ∧ DE + CE = a)
  : R = 17 :=
sorry

end circle_radius_tangent_l128_128921


namespace not_proportional_x2_y2_l128_128722

def directly_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x = k * y

def inversely_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x * y = k

theorem not_proportional_x2_y2 (x y : ℝ) :
  x^2 + y^2 = 16 → ¬directly_proportional x y ∧ ¬inversely_proportional x y :=
by
  sorry

end not_proportional_x2_y2_l128_128722


namespace not_in_range_l128_128328

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range_l128_128328


namespace new_average_of_adjusted_consecutive_integers_l128_128486

theorem new_average_of_adjusted_consecutive_integers
  (x : ℝ)
  (h1 : (1 / 10) * (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) = 25)
  : (1 / 10) * ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9 - 0)) = 20.5 := 
by sorry

end new_average_of_adjusted_consecutive_integers_l128_128486


namespace sum_of_squares_of_sum_and_difference_l128_128867

theorem sum_of_squares_of_sum_and_difference (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 8) : 
  (x + y)^2 + (x - y)^2 = 640 :=
by
  sorry

end sum_of_squares_of_sum_and_difference_l128_128867


namespace train_speed_l128_128902

noncomputable def jogger_speed : ℝ := 9 -- speed in km/hr
noncomputable def jogger_distance : ℝ := 150 / 1000 -- distance in km
noncomputable def train_length : ℝ := 100 / 1000 -- length in km
noncomputable def time_to_pass : ℝ := 25 -- time in seconds

theorem train_speed 
  (v_j : ℝ := jogger_speed)
  (d_j : ℝ := jogger_distance)
  (L : ℝ := train_length)
  (t : ℝ := time_to_pass) :
  (train_speed_in_kmh : ℝ) = 36 :=
by 
  sorry

end train_speed_l128_128902


namespace find_number_l128_128247

theorem find_number (N : ℤ) (h1 : ∃ k : ℤ, N - 3 = 5 * k) (h2 : ∃ l : ℤ, N - 2 = 7 * l) (h3 : 50 < N ∧ N < 70) : N = 58 :=
by
  sorry

end find_number_l128_128247


namespace field_perimeter_l128_128674

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end field_perimeter_l128_128674


namespace total_pins_cardboard_l128_128160

theorem total_pins_cardboard {length width pins : ℕ} (h_length : length = 34) (h_width : width = 14) (h_pins : pins = 35) :
  2 * pins * (length + width) / (length + width) = 140 :=
by
  sorry

end total_pins_cardboard_l128_128160


namespace find_annual_compound_interest_rate_l128_128743

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_compound_interest_rate :
  compound_interest_rate 10000 24882.50 1 7 0.125 :=
by sorry

end find_annual_compound_interest_rate_l128_128743


namespace Hayley_l128_128905

-- Definitions based on the given conditions
def num_friends : ℕ := 9
def stickers_per_friend : ℕ := 8

-- Theorem statement
theorem Hayley's_total_stickers : num_friends * stickers_per_friend = 72 := by
  sorry

end Hayley_l128_128905


namespace find_b3_b17_l128_128069

variable {a : ℕ → ℤ} -- Arithmetic sequence
variable {b : ℕ → ℤ} -- Geometric sequence

axiom arith_seq {a : ℕ → ℤ} (d : ℤ) : ∀ (n : ℕ), a (n + 1) = a n + d
axiom geom_seq {b : ℕ → ℤ} (r : ℤ) : ∀ (n : ℕ), b (n + 1) = b n * r

theorem find_b3_b17 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) 
  (h_geom : ∃ r, ∀ n, b (n + 1) = b n * r)
  (h_cond1 : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10) :
  b 3 * b 17 = 36 := 
sorry

end find_b3_b17_l128_128069


namespace sum_of_first_four_terms_l128_128266

def arithmetic_sequence_sum (a1 a2 : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * (a2 - a1))) / 2

theorem sum_of_first_four_terms : arithmetic_sequence_sum 4 6 4 = 28 :=
by
  sorry

end sum_of_first_four_terms_l128_128266


namespace solution_set_inequality_l128_128519

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
sorry

end solution_set_inequality_l128_128519


namespace tank_fill_time_l128_128592

/-- Given the rates at which pipes fill a tank, prove the total time to fill the tank using all three pipes. --/
theorem tank_fill_time (R_a R_b R_c : ℝ) (T : ℝ)
  (h1 : R_a = 1 / 35)
  (h2 : R_b = 2 * R_a)
  (h3 : R_c = 2 * R_b)
  (h4 : T = 5) :
  1 / (R_a + R_b + R_c) = T := by
  sorry

end tank_fill_time_l128_128592


namespace intersection_of_A_and_B_l128_128864

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := { x | ∃ m : ℕ, x = 2 * m }

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by sorry

end intersection_of_A_and_B_l128_128864


namespace sufficient_condition_for_perpendicular_l128_128951

variables (m n : Line) (α β : Plane)

def are_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem sufficient_condition_for_perpendicular :
  (are_parallel m n) ∧ (line_perpendicular_to_plane n α) → (line_perpendicular_to_plane m α) :=
sorry

end sufficient_condition_for_perpendicular_l128_128951


namespace factor_expression_l128_128591

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l128_128591


namespace lcm_18_24_eq_72_l128_128553

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l128_128553


namespace jellybeans_original_count_l128_128946

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l128_128946


namespace cubic_inequality_l128_128693

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 :=
by
  sorry

end cubic_inequality_l128_128693


namespace annual_earning_difference_l128_128646

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l128_128646


namespace number_of_subsets_l128_128696

-- Define the set
def my_set : Set ℕ := {1, 2, 3}

-- Theorem statement
theorem number_of_subsets : Finset.card (Finset.powerset {1, 2, 3}) = 8 :=
by
  sorry

end number_of_subsets_l128_128696


namespace turnover_june_l128_128443

variable (TurnoverApril TurnoverMay : ℝ)

theorem turnover_june (h1 : TurnoverApril = 10) (h2 : TurnoverMay = 12) :
  TurnoverMay * (1 + (TurnoverMay - TurnoverApril) / TurnoverApril) = 14.4 := by
  sorry

end turnover_june_l128_128443


namespace sum_of_remainders_l128_128490

theorem sum_of_remainders (a b c d : ℕ) 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 := 
by {
  sorry -- Proof not required as per instructions
}

end sum_of_remainders_l128_128490


namespace David_Marks_in_Mathematics_are_85_l128_128608

theorem David_Marks_in_Mathematics_are_85
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : physics_marks = 92)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 89)
  (h6 : num_subjects = 5) : 
  (86 + 92 + 87 + 95 + 85) / 5 = 89 :=
by sorry

end David_Marks_in_Mathematics_are_85_l128_128608


namespace stan_needs_more_minutes_l128_128442

/-- Stan has 10 songs each of 3 minutes and 15 songs each of 2 minutes. His run takes 100 minutes.
    Prove that he needs 40 more minutes of songs in his playlist. -/
theorem stan_needs_more_minutes 
    (num_3min_songs : ℕ) 
    (num_2min_songs : ℕ) 
    (time_per_3min_song : ℕ) 
    (time_per_2min_song : ℕ) 
    (total_run_time : ℕ) 
    (given_minutes_3min_songs : num_3min_songs = 10)
    (given_minutes_2min_songs : num_2min_songs = 15)
    (given_time_per_3min_song : time_per_3min_song = 3)
    (given_time_per_2min_song : time_per_2min_song = 2)
    (given_total_run_time : total_run_time = 100)
    : num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song = 60 →
      total_run_time - (num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song) = 40 := 
by
    sorry

end stan_needs_more_minutes_l128_128442


namespace quadratic_solution_l128_128890

theorem quadratic_solution (x : ℝ) : (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end quadratic_solution_l128_128890


namespace mass_of_man_l128_128015

theorem mass_of_man (L B h ρ V m: ℝ) (boat_length: L = 3) (boat_breadth: B = 2) 
  (boat_sink_depth: h = 0.01) (water_density: ρ = 1000) 
  (displaced_volume: V = L * B * h) (displaced_mass: m = ρ * V): m = 60 := 
by 
  sorry

end mass_of_man_l128_128015


namespace brad_more_pages_than_greg_l128_128311

def greg_pages_first_week : ℕ := 7 * 18
def greg_pages_next_two_weeks : ℕ := 14 * 22
def greg_total_pages : ℕ := greg_pages_first_week + greg_pages_next_two_weeks

def brad_pages_first_5_days : ℕ := 5 * 26
def brad_pages_remaining_12_days : ℕ := 12 * 20
def brad_total_pages : ℕ := brad_pages_first_5_days + brad_pages_remaining_12_days

def total_required_pages : ℕ := 800

theorem brad_more_pages_than_greg : brad_total_pages - greg_total_pages = 64 :=
by
  sorry

end brad_more_pages_than_greg_l128_128311


namespace melissa_coupe_sale_l128_128969

theorem melissa_coupe_sale :
  ∃ x : ℝ, (0.02 * x + 0.02 * 2 * x = 1800) ∧ x = 30000 :=
by
  sorry

end melissa_coupe_sale_l128_128969


namespace initial_amount_l128_128316

theorem initial_amount (H P L : ℝ) (C : ℝ) (n : ℕ) (T M : ℝ) 
  (hH : H = 10) 
  (hP : P = 2) 
  (hC : C = 1.25) 
  (hn : n = 4) 
  (hL : L = 3) 
  (hT : T = H + P + n * C) 
  (hM : M = T + L) : 
  M = 20 := 
sorry

end initial_amount_l128_128316


namespace f_increasing_on_neg_inf_to_one_l128_128464

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 8

theorem f_increasing_on_neg_inf_to_one :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end f_increasing_on_neg_inf_to_one_l128_128464


namespace math_problem_l128_128178

theorem math_problem (f : ℕ → Prop) (m : ℕ) 
  (h1 : f 1) (h2 : f 2) (h3 : f 3)
  (h_implies : ∀ k : ℕ, f k → f (k + m)) 
  (h_max : m = 3):
  ∀ n : ℕ, 0 < n → f n :=
by
  sorry

end math_problem_l128_128178


namespace evaluate_expression_l128_128537

theorem evaluate_expression :
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  (x^3 * y^2 * z^2 * w) = (1 / 48 : ℚ) :=
by
  let x := (1 / 4 : ℚ)
  let y := (1 / 3 : ℚ)
  let z := (-2 : ℚ)
  let w := (3 : ℚ)
  sorry

end evaluate_expression_l128_128537


namespace fraction_zero_x_value_l128_128416

theorem fraction_zero_x_value (x : ℝ) (h : (x^2 - 4) / (x - 2) = 0) (h2 : x ≠ 2) : x = -2 :=
sorry

end fraction_zero_x_value_l128_128416


namespace original_sticker_price_l128_128147

theorem original_sticker_price (S : ℝ) (h1 : 0.80 * S - 120 = 0.65 * S - 10) : S = 733 := 
by
  sorry

end original_sticker_price_l128_128147


namespace square_side_length_leq_half_l128_128610

theorem square_side_length_leq_half
    (l : ℝ)
    (h_square_inside_unit : l ≤ 1)
    (h_no_center_contain : ∀ (x y : ℝ), x^2 + y^2 > (l/2)^2 → (0.5 ≤ x ∨ 0.5 ≤ y)) :
    l ≤ 0.5 := 
sorry

end square_side_length_leq_half_l128_128610


namespace quadratic_equation_root_and_coef_l128_128563

theorem quadratic_equation_root_and_coef (k x : ℤ) (h1 : x^2 - 3 * x + k = 0)
  (root4 : x = 4) : (x = 4 ∧ k = -4 ∧ ∀ y, y ≠ 4 → y^2 - 3 * y + k = 0 → y = -1) :=
by {
  sorry
}

end quadratic_equation_root_and_coef_l128_128563


namespace fourth_root_difference_l128_128357

theorem fourth_root_difference : (81 : ℝ) ^ (1 / 4 : ℝ) - (1296 : ℝ) ^ (1 / 4 : ℝ) = -3 :=
by
  sorry

end fourth_root_difference_l128_128357


namespace like_term_exists_l128_128313

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l128_128313


namespace hyperbola_vertices_distance_l128_128274

noncomputable def distance_between_vertices : ℝ :=
  2 * Real.sqrt 7.5

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), 4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0 →
  distance_between_vertices = 2 * Real.sqrt 7.5 :=
by sorry

end hyperbola_vertices_distance_l128_128274


namespace admission_price_for_adults_l128_128099

def total_people := 610
def num_adults := 350
def child_price := 1
def total_receipts := 960

theorem admission_price_for_adults (A : ℝ) (h1 : 350 * A + 260 = 960) : A = 2 :=
by {
  -- proof omitted
  sorry
}

end admission_price_for_adults_l128_128099


namespace quadratic_distinct_roots_iff_m_lt_four_l128_128329

theorem quadratic_distinct_roots_iff_m_lt_four (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 4 * x₁ + m = 0) ∧ (x₂^2 - 4 * x₂ + m = 0)) ↔ m < 4 :=
by sorry

end quadratic_distinct_roots_iff_m_lt_four_l128_128329


namespace find_f_minus1_plus_f_2_l128_128948

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin := ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

def f_value_at_zero := f 0 = 1

theorem find_f_minus1_plus_f_2 :
  even_function f →
  symmetric_about_origin f →
  f_value_at_zero f →
  f (-1) + f 2 = -1 :=
by
  intros
  sorry

end find_f_minus1_plus_f_2_l128_128948


namespace modified_counting_game_53rd_term_l128_128312

theorem modified_counting_game_53rd_term :
  let a : ℕ := 1
  let d : ℕ := 2
  a + (53 - 1) * d = 105 :=
by 
  sorry

end modified_counting_game_53rd_term_l128_128312


namespace speed_of_second_car_l128_128994

/-!
Two cars started from the same point, at 5 am, traveling in opposite directions. 
One car was traveling at 50 mph, and they were 450 miles apart at 10 am. 
Prove that the speed of the other car is 40 mph.
-/

variable (S : ℝ) -- Speed of the second car

theorem speed_of_second_car
    (h1 : ∀ t : ℝ, t = 5) -- The time of travel from 5 am to 10 am is 5 hours 
    (h2 : ∀ d₁ : ℝ, d₁ = 50 * 5) -- Distance traveled by the first car
    (h3 : ∀ d₂ : ℝ, d₂ = S * 5) -- Distance traveled by the second car
    (h4 : 450 = 50 * 5 + S * 5) -- Total distance between the two cars
    : S = 40 := sorry

end speed_of_second_car_l128_128994


namespace remainder_of_n_squared_plus_4n_plus_5_l128_128575

theorem remainder_of_n_squared_plus_4n_plus_5 {n : ℤ} (h : n % 50 = 1) : (n^2 + 4*n + 5) % 50 = 10 :=
by
  sorry

end remainder_of_n_squared_plus_4n_plus_5_l128_128575


namespace girls_came_in_classroom_l128_128001

theorem girls_came_in_classroom (initial_boys initial_girls boys_left final_children girls_in_classroom : ℕ)
  (h1 : initial_boys = 5)
  (h2 : initial_girls = 4)
  (h3 : boys_left = 3)
  (h4 : final_children = 8)
  (h5 : girls_in_classroom = final_children - (initial_boys - boys_left)) :
  girls_in_classroom - initial_girls = 2 :=
by
  sorry

end girls_came_in_classroom_l128_128001


namespace virginia_eggs_l128_128352

-- Definitions and conditions
variable (eggs_start : Nat)
variable (eggs_taken : Nat := 3)
variable (eggs_end : Nat := 93)

-- Problem statement to prove
theorem virginia_eggs : eggs_start - eggs_taken = eggs_end → eggs_start = 96 :=
by
  intro h
  sorry

end virginia_eggs_l128_128352


namespace xy_sum_l128_128216

namespace ProofExample

variable (x y : ℚ)

def condition1 : Prop := (1 / x) + (1 / y) = 4
def condition2 : Prop := (1 / x) - (1 / y) = -6

theorem xy_sum : condition1 x y → condition2 x y → (x + y = -4 / 5) := by
  intros
  sorry

end ProofExample

end xy_sum_l128_128216


namespace number_of_valid_x_l128_128208

theorem number_of_valid_x (x : ℕ) : 
  ((x + 3) * (x - 3) * (x ^ 2 + 9) < 500) ∧ (x - 3 > 0) ↔ x = 4 :=
sorry

end number_of_valid_x_l128_128208


namespace problem_l128_128657

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l128_128657


namespace prime_square_sum_of_cubes_equals_three_l128_128544

open Nat

theorem prime_square_sum_of_cubes_equals_three (p : ℕ) (h_prime : p.Prime) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3) → (p = 3) :=
by
  sorry

end prime_square_sum_of_cubes_equals_three_l128_128544


namespace continuous_arrow_loop_encircling_rectangle_l128_128067

def total_orientations : ℕ := 2^4

def favorable_orientations : ℕ := 2 * 2

def probability_loop : ℚ := favorable_orientations / total_orientations

theorem continuous_arrow_loop_encircling_rectangle : probability_loop = 1 / 4 := by
  sorry

end continuous_arrow_loop_encircling_rectangle_l128_128067


namespace julia_tulip_count_l128_128109

def tulip_count (tulips daisies : ℕ) : Prop :=
  3 * daisies = 7 * tulips

theorem julia_tulip_count : 
  ∃ t, tulip_count t 65 ∧ t = 28 := 
by
  sorry

end julia_tulip_count_l128_128109


namespace smallest_k_for_min_period_15_l128_128798

/-- Rational number with minimal period -/
def is_minimal_period (r : ℚ) (n : ℕ) : Prop :=
  ∃ m : ℤ, r = m / (10^n - 1)

variables (a b : ℚ)

-- Conditions for a and b
axiom ha : is_minimal_period a 30
axiom hb : is_minimal_period b 30

-- Condition for a - b
axiom hab_min_period : is_minimal_period (a - b) 15

-- Conclusion
theorem smallest_k_for_min_period_15 : ∃ k : ℕ, k = 6 ∧ is_minimal_period (a + k * b) 15 :=
by sorry

end smallest_k_for_min_period_15_l128_128798


namespace combination_15_3_l128_128843

theorem combination_15_3 :
  (Nat.choose 15 3 = 455) :=
by
  sorry

end combination_15_3_l128_128843


namespace SWE4_l128_128008

theorem SWE4 (a : ℕ → ℕ) (n : ℕ) :
  a 0 = 0 →
  (∀ n, a (n + 1) = 2 * a n + 2^n) →
  (∃ k : ℕ, n = 2^k) →
  ∃ m : ℕ, a n = 2^m :=
by
  intros h₀ h_recurrence h_power
  sorry

end SWE4_l128_128008


namespace original_gift_card_value_l128_128884

def gift_card_cost_per_pound : ℝ := 8.58
def coffee_pounds_bought : ℕ := 4
def remaining_balance_after_purchase : ℝ := 35.68

theorem original_gift_card_value :
  (remaining_balance_after_purchase + coffee_pounds_bought * gift_card_cost_per_pound) = 70.00 :=
by
  -- Proof goes here
  sorry

end original_gift_card_value_l128_128884


namespace wolves_total_games_l128_128998

theorem wolves_total_games
  (x y : ℕ) -- Before district play, the Wolves had won x games out of y games.
  (hx : x = 40 * y / 100) -- The Wolves had won 40% of their basketball games before district play.
  (hx' : 5 * x = 2 * y)
  (hy : 60 * (y + 10) / 100 = x + 9) -- They finished the season having won 60% of their total games.
  : y + 10 = 25 := by
  sorry

end wolves_total_games_l128_128998


namespace graph_of_equation_l128_128801

theorem graph_of_equation :
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (x + y + 2 = 0 ∨ x+y = 0 ∨ x-y = 0) ∧ 
  ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    (x₁ + y₁ + 2 = 0 ∧ x₁ + y₁ = 0) ∧ 
    (x₂ + y₂ + 2 = 0 ∧ x₂ = -x₂) ∧ 
    (x₃ + y₃ + 2 = 0 ∧ x₃ - y₃ = 0)) := 
sorry

end graph_of_equation_l128_128801


namespace largest_k_statement_l128_128956

noncomputable def largest_k (n : ℕ) : ℕ :=
  n - 2

theorem largest_k_statement (S : Finset ℕ) (A : Finset (Finset ℕ)) (h1 : ∀ (A_i : Finset ℕ), A_i ∈ A → 2 ≤ A_i.card ∧ A_i.card < S.card) : 
  largest_k S.card = S.card - 2 :=
by
  sorry

end largest_k_statement_l128_128956


namespace books_sold_in_store_on_saturday_l128_128676

namespace BookshopInventory

def initial_inventory : ℕ := 743
def saturday_online_sales : ℕ := 128
def sunday_online_sales : ℕ := 162
def shipment_received : ℕ := 160
def final_inventory : ℕ := 502

-- Define the total number of books sold
def total_books_sold (S : ℕ) : ℕ := S + saturday_online_sales + 2 * S + sunday_online_sales

-- Net change in inventory equals total books sold minus shipment received
def net_change_in_inventory (S : ℕ) : ℕ := total_books_sold S - shipment_received

-- Prove that the difference between initial and final inventories equals the net change in inventory
theorem books_sold_in_store_on_saturday : ∃ S : ℕ, net_change_in_inventory S = initial_inventory - final_inventory ∧ S = 37 :=
by
  sorry

end BookshopInventory

end books_sold_in_store_on_saturday_l128_128676


namespace max_value_min_4x_y_4y_x2_5y2_l128_128042

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2_l128_128042


namespace proof_problem_l128_128123

variable {a b : ℤ}

theorem proof_problem (h1 : ∃ k : ℤ, a = 4 * k) (h2 : ∃ l : ℤ, b = 8 * l) : 
  (∃ m : ℤ, b = 4 * m) ∧
  (∃ n : ℤ, a - b = 4 * n) ∧
  (∃ p : ℤ, a + b = 2 * p) := 
by
  sorry

end proof_problem_l128_128123


namespace total_length_of_rope_l128_128604

theorem total_length_of_rope (x : ℝ) : (∃ r1 r2 : ℝ, r1 / r2 = 2 / 3 ∧ r1 = 16 ∧ x = r1 + r2) → x = 40 :=
by
  intro h
  cases' h with r1 hr
  cases' hr with r2 hs
  sorry

end total_length_of_rope_l128_128604


namespace value_of_y_l128_128270

noncomputable def k : ℝ := 168.75

theorem value_of_y (x y : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x = 3 * y) : y = -16.875 :=
by 
  sorry

end value_of_y_l128_128270


namespace average_of_modified_set_l128_128320

theorem average_of_modified_set (a1 a2 a3 a4 a5 : ℝ) (h : (a1 + a2 + a3 + a4 + a5) / 5 = 8) :
  ((a1 + 10) + (a2 - 10) + (a3 + 10) + (a4 - 10) + (a5 + 10)) / 5 = 10 :=
by 
  sorry

end average_of_modified_set_l128_128320


namespace total_paint_remaining_l128_128118

-- Definitions based on the conditions
def paint_per_statue : ℚ := 1 / 16
def statues_to_paint : ℕ := 14

-- Theorem statement to prove the answer
theorem total_paint_remaining : (statues_to_paint : ℚ) * paint_per_statue = 7 / 8 := 
by sorry

end total_paint_remaining_l128_128118


namespace trigonometric_identity_l128_128295

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l128_128295


namespace price_of_fruit_l128_128251

theorem price_of_fruit
  (price_milk_per_liter : ℝ)
  (milk_per_batch : ℝ)
  (fruit_per_batch : ℝ)
  (cost_for_three_batches : ℝ)
  (F : ℝ)
  (h1 : price_milk_per_liter = 1.5)
  (h2 : milk_per_batch = 10)
  (h3 : fruit_per_batch = 3)
  (h4 : cost_for_three_batches = 63)
  (h5 : 3 * (milk_per_batch * price_milk_per_liter + fruit_per_batch * F) = cost_for_three_batches) :
  F = 2 :=
by sorry

end price_of_fruit_l128_128251


namespace integers_in_range_of_f_l128_128040

noncomputable def f (x : ℝ) := x^2 + x + 1/2

def count_integers_in_range (n : ℕ) : ℕ :=
  2 * (n + 1)

theorem integers_in_range_of_f (n : ℕ) :
  (count_integers_in_range n) = (2 * (n + 1)) :=
by
  sorry

end integers_in_range_of_f_l128_128040


namespace jackson_weekly_mileage_increase_l128_128277

theorem jackson_weekly_mileage_increase :
  ∃ (weeks : ℕ), weeks = (7 - 3) / 1 := by
  sorry

end jackson_weekly_mileage_increase_l128_128277


namespace cubic_km_to_cubic_m_l128_128692

theorem cubic_km_to_cubic_m (km_to_m : 1 = 1000) : (1 : ℝ) ^ 3 = (1000 : ℝ) ^ 3 :=
by sorry

end cubic_km_to_cubic_m_l128_128692


namespace positive_diff_after_add_five_l128_128911

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end positive_diff_after_add_five_l128_128911


namespace classification_of_square_and_cube_roots_l128_128389

-- Define the three cases: positive, zero, and negative
inductive NumberCase
| positive 
| zero 
| negative 

-- Define the concept of "classification and discussion thinking"
def is_classification_and_discussion_thinking (cases : List NumberCase) : Prop :=
  cases = [NumberCase.positive, NumberCase.zero, NumberCase.negative]

-- The main statement to be proven
theorem classification_of_square_and_cube_roots :
  is_classification_and_discussion_thinking [NumberCase.positive, NumberCase.zero, NumberCase.negative] :=
by
  sorry

end classification_of_square_and_cube_roots_l128_128389


namespace find_a_and_b_l128_128370

theorem find_a_and_b (a b : ℤ) (h : ∀ x : ℝ, x ≤ 0 → (a*x + 2)*(x^2 + 2*b) ≤ 0) : a = 1 ∧ b = -2 := 
by 
  -- Proof steps would go here, but they are omitted as per instructions.
  sorry

end find_a_and_b_l128_128370


namespace x_squared_minus_y_squared_l128_128223

theorem x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 2 / 5) (h2 : x - y = 1 / 10) : x ^ 2 - y ^ 2 = 1 / 25 :=
by
  sorry

end x_squared_minus_y_squared_l128_128223


namespace find_f_comp_f_l128_128197

def f (x : ℚ) : ℚ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_comp_f (h : f (f (5/2)) = 3/2) :
  f (f (5/2)) = 3/2 := by
  sorry

end find_f_comp_f_l128_128197


namespace each_sibling_gets_13_pencils_l128_128955

theorem each_sibling_gets_13_pencils (colored_pencils : ℕ) (black_pencils : ℕ) (siblings : ℕ) (kept_pencils : ℕ) 
  (hyp1 : colored_pencils = 14) (hyp2 : black_pencils = 35) (hyp3 : siblings = 3) (hyp4 : kept_pencils = 10) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end each_sibling_gets_13_pencils_l128_128955


namespace unique_real_solution_l128_128931

theorem unique_real_solution (x y z : ℝ) :
  (x^3 - 3 * x = 4 - y) ∧ 
  (2 * y^3 - 6 * y = 6 - z) ∧ 
  (3 * z^3 - 9 * z = 8 - x) ↔ 
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end unique_real_solution_l128_128931


namespace max_next_person_weight_l128_128173

def avg_weight_adult := 150
def avg_weight_child := 70
def max_weight_elevator := 1500
def num_adults := 7
def num_children := 5

def total_weight_adults := num_adults * avg_weight_adult
def total_weight_children := num_children * avg_weight_child
def current_weight := total_weight_adults + total_weight_children

theorem max_next_person_weight : 
  max_weight_elevator - current_weight = 100 := 
by 
  sorry

end max_next_person_weight_l128_128173


namespace fraction_equation_solution_l128_128163

theorem fraction_equation_solution (x y : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 5) (hy1 : y ≠ 0) (hy2 : y ≠ 7)
  (h : (3 / x) + (2 / y) = 1 / 3) : 
  x = (9 * y) / (y - 6) :=
sorry

end fraction_equation_solution_l128_128163


namespace airline_flights_increase_l128_128116

theorem airline_flights_increase (n k : ℕ) 
  (h : (n + k) * (n + k - 1) / 2 - n * (n - 1) / 2 = 76) :
  (n = 6 ∧ n + k = 14) ∨ (n = 76 ∧ n + k = 77) :=
by
  sorry

end airline_flights_increase_l128_128116


namespace maximize_sum_of_arithmetic_seq_l128_128783

theorem maximize_sum_of_arithmetic_seq (a d : ℤ) (n : ℤ) : d < 0 → a^2 = (a + 10 * d)^2 → n = 5 ∨ n = 6 :=
by
  intro h_d_neg h_a1_eq_a11
  have h_a1_5d_neg : a + 5 * d = 0 := sorry
  have h_sum_max : n = 5 ∨ n = 6 := sorry
  exact h_sum_max

end maximize_sum_of_arithmetic_seq_l128_128783


namespace susie_vacuums_each_room_in_20_minutes_l128_128038

theorem susie_vacuums_each_room_in_20_minutes
  (total_time_hours : ℕ)
  (number_of_rooms : ℕ)
  (total_time_minutes : ℕ)
  (time_per_room : ℕ)
  (h1 : total_time_hours = 2)
  (h2 : number_of_rooms = 6)
  (h3 : total_time_minutes = total_time_hours * 60)
  (h4 : time_per_room = total_time_minutes / number_of_rooms) :
  time_per_room = 20 :=
by
  sorry

end susie_vacuums_each_room_in_20_minutes_l128_128038


namespace number_of_points_max_45_lines_l128_128556

theorem number_of_points_max_45_lines (n : ℕ) (h : n * (n - 1) / 2 ≤ 45) : n = 10 := 
  sorry

end number_of_points_max_45_lines_l128_128556


namespace initial_oranges_per_rupee_l128_128284

theorem initial_oranges_per_rupee (loss_rate_gain_rate cost_rate : ℝ) (initial_oranges : ℤ) : 
  loss_rate_gain_rate = 0.92 ∧ cost_rate = 18.4 ∧ 1.25 * cost_rate = 1.25 * 0.92 * (initial_oranges : ℝ) →
  initial_oranges = 14 := by
  sorry

end initial_oranges_per_rupee_l128_128284


namespace speed_upstream_l128_128734

-- Conditions definitions
def speed_of_boat_still_water : ℕ := 50
def speed_of_current : ℕ := 20

-- Theorem stating the problem
theorem speed_upstream : (speed_of_boat_still_water - speed_of_current = 30) :=
by
  -- Proof is omitted
  sorry

end speed_upstream_l128_128734


namespace arithmetic_progression_rth_term_l128_128177

theorem arithmetic_progression_rth_term (S : ℕ → ℕ) (hS : ∀ n, S n = 5 * n + 4 * n ^ 2) 
  (r : ℕ) : S r - S (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l128_128177


namespace predict_monthly_savings_l128_128446

noncomputable def sum_x_i := 80
noncomputable def sum_y_i := 20
noncomputable def sum_x_i_y_i := 184
noncomputable def sum_x_i_sq := 720
noncomputable def n := 10
noncomputable def x_bar := sum_x_i / n
noncomputable def y_bar := sum_y_i / n
noncomputable def b := (sum_x_i_y_i - n * x_bar * y_bar) / (sum_x_i_sq - n * x_bar^2)
noncomputable def a := y_bar - b * x_bar
noncomputable def regression_eqn(x: ℝ) := b * x + a

theorem predict_monthly_savings :
  regression_eqn 7 = 1.7 :=
by
  sorry

end predict_monthly_savings_l128_128446


namespace cost_of_plastering_is_334_point_8_l128_128399

def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.45

def bottom_area : ℝ := tank_length * tank_width
def long_wall_area : ℝ := 2 * (tank_length * tank_depth)
def short_wall_area : ℝ := 2 * (tank_width * tank_depth)
def total_surface_area : ℝ := bottom_area + long_wall_area + short_wall_area
def total_cost : ℝ := total_surface_area * cost_per_sq_meter

theorem cost_of_plastering_is_334_point_8 :
  total_cost = 334.8 :=
by
  sorry

end cost_of_plastering_is_334_point_8_l128_128399


namespace perimeter_of_ABCD_is_35_2_l128_128787

-- Definitions of geometrical properties and distances
variable (AB BC DC : ℝ)
variable (AB_perp_BC : ∃P, is_perpendicular AB BC)
variable (DC_parallel_AB : ∃Q, is_parallel DC AB)
variable (AB_length : AB = 7)
variable (BC_length : BC = 10)
variable (DC_length : DC = 6)

-- Target statement to be proved
theorem perimeter_of_ABCD_is_35_2
  (h1 : AB_perp_BC)
  (h2 : DC_parallel_AB)
  (h3 : AB_length)
  (h4 : BC_length)
  (h5 : DC_length) :
  ∃ P : ℝ, P = 35.2 :=
sorry

end perimeter_of_ABCD_is_35_2_l128_128787


namespace sample_size_l128_128481

theorem sample_size (n : ℕ) (h1 : n ∣ 36) (h2 : 36 / n ∣ 6) (h3 : (n + 1) ∣ 35) : n = 6 := 
sorry

end sample_size_l128_128481


namespace additional_carpet_needed_is_94_l128_128412

noncomputable def area_room_a : ℝ := 4 * 20

noncomputable def area_room_b : ℝ := area_room_a / 2.5

noncomputable def total_area : ℝ := area_room_a + area_room_b

noncomputable def carpet_jessie_has : ℝ := 18

noncomputable def additional_carpet_needed : ℝ := total_area - carpet_jessie_has

theorem additional_carpet_needed_is_94 :
  additional_carpet_needed = 94 := by
  sorry

end additional_carpet_needed_is_94_l128_128412


namespace percent_democrats_l128_128854

/-- The percentage of registered voters in the city who are democrats and republicans -/
def D : ℝ := sorry -- Percent of democrats
def R : ℝ := sorry -- Percent of republicans

-- Given conditions
axiom H1 : D + R = 100
axiom H2 : 0.65 * D + 0.20 * R = 47

-- Statement to prove
theorem percent_democrats : D = 60 :=
by
  sorry

end percent_democrats_l128_128854


namespace number_of_players_l128_128678
-- Importing the necessary library

-- Define the number of games formula for the tournament
def number_of_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- The theorem to prove the number of players given the conditions
theorem number_of_players (n : ℕ) (h : number_of_games n = 306) : n = 18 :=
by
  sorry

end number_of_players_l128_128678


namespace years_of_interest_l128_128048

noncomputable def principal : ℝ := 2600
noncomputable def interest_difference : ℝ := 78

theorem years_of_interest (R : ℝ) (N : ℝ) (h : (principal * (R + 1) * N / 100) - (principal * R * N / 100) = interest_difference) : N = 3 :=
sorry

end years_of_interest_l128_128048


namespace solution_set_of_inequality_l128_128404

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
sorry

end solution_set_of_inequality_l128_128404


namespace input_language_is_input_l128_128645

def is_print_statement (statement : String) : Prop := 
  statement = "PRINT"

def is_input_statement (statement : String) : Prop := 
  statement = "INPUT"

def is_conditional_statement (statement : String) : Prop := 
  statement = "IF"

theorem input_language_is_input :
  is_input_statement "INPUT" := 
by
  -- Here we need to show "INPUT" is an input statement
  sorry

end input_language_is_input_l128_128645


namespace quadratic_root_value_l128_128799

theorem quadratic_root_value (a b : ℤ) (h : 2 * a - b = -3) : 6 * a - 3 * b + 6 = -3 :=
by 
  sorry

end quadratic_root_value_l128_128799


namespace problem_solution_l128_128165

noncomputable def complex_expression : ℝ :=
  (-(1/2) * (1/100))^5 * ((2/3) * (2/100))^4 * (-(3/4) * (3/100))^3 * ((4/5) * (4/100))^2 * (-(5/6) * (5/100)) * 10^30

theorem problem_solution : complex_expression = -48 :=
by
  sorry

end problem_solution_l128_128165


namespace heidi_paints_fraction_in_10_minutes_l128_128209

variable (Heidi_paint_rate : ℕ → ℝ)
variable (t : ℕ)
variable (fraction : ℝ)

theorem heidi_paints_fraction_in_10_minutes 
  (h1 : Heidi_paint_rate 30 = 1) 
  (h2 : t = 10) 
  (h3 : fraction = 1 / 3) : 
  Heidi_paint_rate t = fraction := 
sorry

end heidi_paints_fraction_in_10_minutes_l128_128209


namespace change_given_back_l128_128866

theorem change_given_back
  (p s t a : ℕ)
  (hp : p = 140)
  (hs : s = 43)
  (ht : t = 15)
  (ha : a = 200) :
  (a - (p + s + t)) = 2 :=
by
  sorry

end change_given_back_l128_128866


namespace license_plate_palindrome_probability_l128_128738

-- Define the two-letter palindrome probability
def prob_two_letter_palindrome : ℚ := 1 / 26

-- Define the four-digit palindrome probability
def prob_four_digit_palindrome : ℚ := 1 / 100

-- Define the joint probability of both two-letter and four-digit palindrome
def prob_joint_palindrome : ℚ := prob_two_letter_palindrome * prob_four_digit_palindrome

-- Define the probability of at least one palindrome using Inclusion-Exclusion
def prob_at_least_one_palindrome : ℚ := prob_two_letter_palindrome + prob_four_digit_palindrome - prob_joint_palindrome

-- Convert the probability to the form of sum of two integers
def sum_of_integers : ℕ := 5 + 104

-- The final proof problem
theorem license_plate_palindrome_probability :
  (prob_at_least_one_palindrome = 5 / 104) ∧ (sum_of_integers = 109) := by
  sorry

end license_plate_palindrome_probability_l128_128738


namespace intersection_is_ge_negative_one_l128_128159

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem intersection_is_ge_negative_one : M ∩ N = {y | y ≥ -1} := by
  sorry

end intersection_is_ge_negative_one_l128_128159


namespace train_passes_jogger_in_39_seconds_l128_128792

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_head_start : ℝ := 270
noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45

noncomputable def to_meters_per_second (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def jogger_speed_mps : ℝ :=
  to_meters_per_second jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  to_meters_per_second train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance : ℝ :=
  jogger_head_start + train_length

noncomputable def time_to_pass_jogger : ℝ :=
  total_distance / relative_speed_mps

theorem train_passes_jogger_in_39_seconds :
  time_to_pass_jogger = 39 := by
  sorry

end train_passes_jogger_in_39_seconds_l128_128792


namespace max_halls_l128_128861

theorem max_halls (n : ℕ) (hall : ℕ → ℕ) (H : ∀ n, hall n = hall (3 * n + 1) ∧ hall n = hall (n + 10)) :
  ∃ (m : ℕ), m = 3 :=
by
  sorry

end max_halls_l128_128861


namespace boris_number_of_bowls_l128_128127

-- Definitions from the conditions
def total_candies : ℕ := 100
def daughter_eats : ℕ := 8
def candies_per_bowl_after_removal : ℕ := 20
def candies_removed_per_bowl : ℕ := 3

-- Derived definitions
def remaining_candies : ℕ := total_candies - daughter_eats
def candies_per_bowl_orig : ℕ := candies_per_bowl_after_removal + candies_removed_per_bowl

-- Statement to prove
theorem boris_number_of_bowls : remaining_candies / candies_per_bowl_orig = 4 :=
by sorry

end boris_number_of_bowls_l128_128127


namespace game_winning_strategy_l128_128851

theorem game_winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 1) ∧ (n % 2 = 1 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 2) :=
by
  sorry

end game_winning_strategy_l128_128851


namespace average_of_first_n_multiples_of_8_is_88_l128_128990

theorem average_of_first_n_multiples_of_8_is_88 (n : ℕ) (h : (n / 2) * (8 + 8 * n) / n = 88) : n = 21 :=
sorry

end average_of_first_n_multiples_of_8_is_88_l128_128990


namespace algebraic_identity_l128_128517

theorem algebraic_identity (a b : ℕ) (h1 : a = 753) (h2 : b = 247)
  (identity : ∀ a b, (a^2 + b^2 - a * b) / (a^3 + b^3) = 1 / (a + b)) : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 0.001 := 
by
  sorry

end algebraic_identity_l128_128517


namespace evaluate_expression_at_3_l128_128058

theorem evaluate_expression_at_3 : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_3_l128_128058


namespace smallest_odd_factors_gt_100_l128_128115

theorem smallest_odd_factors_gt_100 : ∃ n : ℕ, n > 100 ∧ (∀ d : ℕ, d ∣ n → (∃ m : ℕ, n = m * m)) ∧ (∀ m : ℕ, m > 100 ∧ (∀ d : ℕ, d ∣ m → (∃ k : ℕ, m = k * k)) → n ≤ m) :=
by
  sorry

end smallest_odd_factors_gt_100_l128_128115


namespace minimum_value_of_f_l128_128731

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3 * x + 3) + Real.sqrt (x^2 - 3 * x + 3)

theorem minimum_value_of_f : (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ f 0 = 2 * Real.sqrt 3 :=
by
  sorry

end minimum_value_of_f_l128_128731


namespace internet_plan_comparison_l128_128073

theorem internet_plan_comparison (d : ℕ) :
    3000 + 200 * d > 5000 → d > 10 :=
by
  intro h
  -- Proof will be written here
  sorry

end internet_plan_comparison_l128_128073


namespace smallest_y_value_l128_128796

noncomputable def f (y : ℝ) : ℝ := 3 * y ^ 2 + 27 * y - 90
noncomputable def g (y : ℝ) : ℝ := y * (y + 15)

theorem smallest_y_value (y : ℝ) : (∀ y, f y = g y → y ≠ -9) → false := by
  sorry

end smallest_y_value_l128_128796


namespace inclination_angle_of_line_l128_128157

theorem inclination_angle_of_line (m : ℝ) (b : ℝ) (h : b = -3) (h_line : ∀ x : ℝ, x - 3 = m * x + b) : 
  (Real.arctan m * 180 / Real.pi) = 45 := 
by sorry

end inclination_angle_of_line_l128_128157


namespace unique_solution_abc_l128_128432

theorem unique_solution_abc (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
(h1 : b ∣ 2^a - 1) 
(h2 : c ∣ 2^b - 1) 
(h3 : a ∣ 2^c - 1) : 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end unique_solution_abc_l128_128432


namespace original_population_l128_128742

-- Define the conditions
def population_increase (n : ℕ) : ℕ := n + 1200
def population_decrease (p : ℕ) : ℕ := (89 * p) / 100
def final_population (n : ℕ) : ℕ := population_decrease (population_increase n)

-- Claim that needs to be proven
theorem original_population (n : ℕ) (H : final_population n = n - 32) : n = 10000 :=
by
  sorry

end original_population_l128_128742


namespace A_eq_B_l128_128523

open Set

def A := {x | ∃ a : ℝ, x = 5 - 4 * a + a ^ 2}
def B := {y | ∃ b : ℝ, y = 4 * b ^ 2 + 4 * b + 2}

theorem A_eq_B : A = B := sorry

end A_eq_B_l128_128523


namespace jenna_owes_amount_l128_128777

theorem jenna_owes_amount (initial_bill : ℝ) (rate : ℝ) (times : ℕ) : 
  initial_bill = 400 → rate = 0.02 → times = 3 → 
  owed_amount = (400 * (1 + 0.02)^3) := 
by
  intros
  sorry

end jenna_owes_amount_l128_128777


namespace has_root_in_interval_l128_128939

def f (x : ℝ) := x^3 - 3*x - 3

theorem has_root_in_interval : ∃ c ∈ (Set.Ioo (2:ℝ) 3), f c = 0 :=
by 
    sorry

end has_root_in_interval_l128_128939


namespace b_present_age_l128_128487

/-- 
In 10 years, A will be twice as old as B was 10 years ago. 
A is currently 8 years older than B. 
Prove that B's current age is 38.
--/
theorem b_present_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 8) : 
  b = 38 := 
  sorry

end b_present_age_l128_128487


namespace bus_speed_l128_128684

theorem bus_speed (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10)
    (h1 : 9 * (11 * y - x) = 5 * z)
    (h2 : z = 9) :
    ∀ speed, speed = 45 :=
by
  sorry

end bus_speed_l128_128684


namespace westgate_high_school_chemistry_l128_128764

theorem westgate_high_school_chemistry :
  ∀ (total_players physics_both physics : ℕ),
    total_players = 15 →
    physics_both = 3 →
    physics = 8 →
    (total_players - (physics - physics_both)) - physics_both = 10 := by
  intros total_players physics_both physics h1 h2 h3
  sorry

end westgate_high_school_chemistry_l128_128764


namespace probability_of_perpendicular_edges_l128_128169

def is_perpendicular_edge (e1 e2 : ℕ) : Prop :=
-- Define the logic for identifying perpendicular edges here
sorry

def total_outcomes : ℕ := 81

def favorable_outcomes : ℕ :=
-- Calculate the number of favorable outcomes here
20 + 6 + 18

theorem probability_of_perpendicular_edges : 
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 44 / 81 := by
-- Proof for calculating the probability
sorry

end probability_of_perpendicular_edges_l128_128169


namespace average_xyz_l128_128775

theorem average_xyz (x y z : ℝ) (h1 : x = 3) (h2 : y = 2 * x) (h3 : z = 3 * y) : 
  (x + y + z) / 3 = 9 :=
by
  sorry

end average_xyz_l128_128775


namespace perfect_square_condition_l128_128007

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = k^2) ↔ m = 196 :=
by sorry

end perfect_square_condition_l128_128007


namespace video_call_cost_l128_128559

-- Definitions based on the conditions
def charge_rate : ℕ := 30    -- Charge rate in won per ten seconds
def call_duration : ℕ := 2 * 60 + 40  -- Call duration in seconds

-- The proof statement, anticipating the solution to be a total cost calculation
theorem video_call_cost : (call_duration / 10) * charge_rate = 480 :=
by
  -- Placeholder for the proof
  sorry

end video_call_cost_l128_128559


namespace find_d_l128_128265

theorem find_d (a₁: ℤ) (d : ℤ) (Sn : ℤ → ℤ) : 
  a₁ = 190 → 
  (Sn 20 > 0) → 
  (Sn 24 < 0) → 
  (Sn n = n * a₁ + (n * (n - 1)) / 2 * d) →
  d = -17 :=
by
  intros
  sorry

end find_d_l128_128265


namespace avg_speed_trip_l128_128108

noncomputable def distance_travelled (speed time : ℕ) : ℕ := speed * time

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ := total_distance / total_time

theorem avg_speed_trip :
  let first_leg_speed := 75
  let first_leg_time := 4
  let second_leg_speed := 60
  let second_leg_time := 2
  let total_time := first_leg_time + second_leg_time
  let first_leg_distance := distance_travelled first_leg_speed first_leg_time
  let second_leg_distance := distance_travelled second_leg_speed second_leg_time
  let total_distance := first_leg_distance + second_leg_distance
  average_speed total_distance total_time = 70 :=
by
  sorry

end avg_speed_trip_l128_128108


namespace problem_statement_l128_128298

def S : ℤ := (-2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19)

theorem problem_statement (hS : S = -2^20 + 4) : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19 + 2^20 = 6 :=
by
  sorry

end problem_statement_l128_128298


namespace smallest_possible_a_l128_128293

theorem smallest_possible_a (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c) (h4 : a^2 = c * b) : a = 1 :=
by
  sorry

end smallest_possible_a_l128_128293


namespace ratio_diagonals_to_sides_l128_128755

-- Definition of the number of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the condition
def n : ℕ := 5

-- Proof statement that the ratio of the number of diagonals to the number of sides is 1
theorem ratio_diagonals_to_sides (n_eq_5 : n = 5) : 
  (number_of_diagonals n) / n = 1 :=
by {
  -- Proof would go here, but is omitted
  sorry
}

end ratio_diagonals_to_sides_l128_128755


namespace mean_temperature_l128_128230

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end mean_temperature_l128_128230


namespace find_m_from_inequality_l128_128418

theorem find_m_from_inequality :
  (∀ x, x^2 - (m+2)*x > 0 ↔ (x < 0 ∨ x > 2)) → m = 0 :=
by
  sorry

end find_m_from_inequality_l128_128418


namespace gina_keeps_170_l128_128533

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end gina_keeps_170_l128_128533


namespace price_of_third_variety_l128_128569

theorem price_of_third_variety 
    (price1 price2 price3 : ℝ)
    (mix_ratio1 mix_ratio2 mix_ratio3 : ℝ)
    (mixture_price : ℝ)
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mix_ratio1 = 1)
    (h4 : mix_ratio2 = 1)
    (h5 : mix_ratio3 = 2)
    (h6 : mixture_price = 153) :
    price3 = 175.5 :=
by
  sorry

end price_of_third_variety_l128_128569


namespace surface_area_of_cube_l128_128373

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 4 * a
  let face_area := edge_length ^ 2
  let total_surface_area := 6 * face_area
  total_surface_area = 96 * a^2 := by
  sorry

end surface_area_of_cube_l128_128373


namespace sequence_formula_l128_128397

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4 ^ n - 1 :=
by
  sorry

end sequence_formula_l128_128397


namespace tan_45_degrees_l128_128953

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l128_128953


namespace mahesh_worked_days_l128_128741

-- Definitions
def mahesh_work_days := 45
def rajesh_work_days := 30
def total_work_days := 54

-- Theorem statement
theorem mahesh_worked_days (maheshrate : ℕ := mahesh_work_days) (rajeshrate : ℕ := rajesh_work_days) (totaldays : ℕ := total_work_days) :
  ∃ x : ℕ, x = totaldays - rajesh_work_days := by
  apply Exists.intro (54 - 30)
  simp
  sorry

end mahesh_worked_days_l128_128741


namespace relationship_between_a_b_c_d_l128_128036

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.sin x)

open Real

theorem relationship_between_a_b_c_d :
  ∀ (x : ℝ) (a b c d : ℝ),
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, f x ≤ a ∧ b ≤ f x) →
  (∀ x, g x ≤ c ∧ d ≤ g x) →
  a = sin 1 →
  b = -sin 1 →
  c = 1 →
  d = cos 1 →
  b < d ∧ d < a ∧ a < c := by
  sorry

end relationship_between_a_b_c_d_l128_128036


namespace walkway_area_296_l128_128897

theorem walkway_area_296 :
  let bed_length := 4
  let bed_width := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_bed_area := num_rows * num_columns * bed_length * bed_width
  let total_garden_width := num_columns * bed_length + (num_columns + 1) * walkway_width
  let total_garden_height := num_rows * bed_width + (num_rows + 1) * walkway_width
  let total_garden_area := total_garden_width * total_garden_height
  let total_walkway_area := total_garden_area - total_bed_area
  total_walkway_area = 296 :=
by 
  sorry

end walkway_area_296_l128_128897


namespace sum_of_integers_l128_128054

theorem sum_of_integers:
  ∀ (m n p q : ℕ),
    m ≠ n → m ≠ p → m ≠ q → n ≠ p → n ≠ q → p ≠ q →
    (8 - m) * (8 - n) * (8 - p) * (8 - q) = 9 →
    m + n + p + q = 32 :=
by
  intros m n p q hmn hmp hmq hnp hnq hpq heq
  sorry

end sum_of_integers_l128_128054


namespace power_mod_result_l128_128913

-- Define the modulus and base
def mod : ℕ := 8
def base : ℕ := 7
def exponent : ℕ := 202

-- State the theorem
theorem power_mod_result :
  (base ^ exponent) % mod = 1 :=
by
  sorry

end power_mod_result_l128_128913


namespace negation_of_universal_prop_l128_128895

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.sin x > 1

-- The theorem stating the equivalence
theorem negation_of_universal_prop : ¬p ↔ neg_p := 
by sorry

end negation_of_universal_prop_l128_128895


namespace cow_value_increase_l128_128080

theorem cow_value_increase :
  let starting_weight : ℝ := 732
  let increase_factor : ℝ := 1.35
  let price_per_pound : ℝ := 2.75
  let new_weight := starting_weight * increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  let increase_in_value := value_at_new_weight - value_at_starting_weight
  increase_in_value = 704.55 :=
by
  sorry

end cow_value_increase_l128_128080


namespace value_of_expression_l128_128362

noncomputable def f : ℝ → ℝ
| x => if x > 0 then -1 else if x < 0 then 1 else 0

theorem value_of_expression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b := 
sorry

end value_of_expression_l128_128362


namespace cost_price_of_item_l128_128239

theorem cost_price_of_item 
  (retail_price : ℝ) (reduction_percentage : ℝ) 
  (additional_discount : ℝ) (profit_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : retail_price = 900)
  (h2 : reduction_percentage = 0.1)
  (h3 : additional_discount = 48)
  (h4 : profit_percentage = 0.2)
  (h5 : selling_price = 762) :
  ∃ x : ℝ, selling_price = 1.2 * x ∧ x = 635 := 
by {
  sorry
}

end cost_price_of_item_l128_128239


namespace constant_seq_decreasing_implication_range_of_values_l128_128607

noncomputable def sequences (a b : ℕ → ℝ) := 
  (∀ n, a (n+1) = (1/2) * a n + (1/2) * b n) ∧
  (∀ n, (1/b (n+1)) = (1/2) * (1/a n) + (1/2) * (1/b n))

theorem constant_seq (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) :
  ∃ c, ∀ n, a n * b n = c :=
sorry

theorem decreasing_implication (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) (h_dec : ∀ n, a (n+1) < a n) :
  a 1 > b 1 :=
sorry

theorem range_of_values (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 = 4) (h_b1 : b 1 = 1) :
  ∀ n ≥ 2, 2 < a n ∧ a n ≤ 5/2 :=
sorry

end constant_seq_decreasing_implication_range_of_values_l128_128607


namespace cos_C_of_triangle_l128_128296

theorem cos_C_of_triangle
  (sin_A : ℝ) (cos_B : ℝ) 
  (h1 : sin_A = 3/5)
  (h2 : cos_B = 5/13) :
  ∃ (cos_C : ℝ), cos_C = 16/65 :=
by
  -- Place for the proof
  sorry

end cos_C_of_triangle_l128_128296


namespace problem_proof_l128_128807

theorem problem_proof (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) : 3 * a^2 * b + 3 * a * b^2 = 18 := 
by
  sorry

end problem_proof_l128_128807


namespace fraction_product_l128_128012

theorem fraction_product : 
  (7 / 5) * (8 / 16) * (21 / 15) * (14 / 28) * (35 / 25) * (20 / 40) * (49 / 35) * (32 / 64) = 2401 / 10000 :=
by
  -- This line is to skip the proof
  sorry

end fraction_product_l128_128012


namespace range_of_b_l128_128078

theorem range_of_b (b : ℝ) :
  (∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 
    y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔ 
    1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3 :=
by
  sorry

end range_of_b_l128_128078


namespace contradictory_goldbach_l128_128400

theorem contradictory_goldbach : ¬ (∀ n : ℕ, 2 < n ∧ Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
sorry

end contradictory_goldbach_l128_128400


namespace bridge_length_l128_128933

theorem bridge_length (rate : ℝ) (time_minutes : ℝ) (length : ℝ) 
    (rate_condition : rate = 10) 
    (time_condition : time_minutes = 15) : 
    length = 2.5 := 
by
  sorry

end bridge_length_l128_128933


namespace triplet_solution_l128_128952

theorem triplet_solution (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a + b + c = (1 / a) + (1 / b) + (1 / c) ∧ a ^ 2 + b ^ 2 + c ^ 2 = (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2))
  ↔ (∃ x, (a = 1 ∨ a = -1 ∨ a = x ∨ a = 1/x) ∧
           (b = 1 ∨ b = -1 ∨ b = x ∨ b = 1/x) ∧
           (c = 1 ∨ c = -1 ∨ c = x ∨ c = 1/x)) := 
sorry

end triplet_solution_l128_128952


namespace find_k_l128_128992

-- Defining the conditions used in the problem context
def line_condition (k a b : ℝ) : Prop :=
  (b = 4 * k + 1) ∧ (5 = k * a + 1) ∧ (b + 1 = k * a + 1)

-- The statement of the theorem
theorem find_k (a b k : ℝ) (h : line_condition k a b) : k = 3 / 4 :=
by sorry

end find_k_l128_128992


namespace value_of_expr_l128_128636

-- Definitions
def operation (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The proof statement
theorem value_of_expr (a b : ℕ) (h₀ : operation a b = 100) : (a + b) + 6 = 11 := by
  sorry

end value_of_expr_l128_128636


namespace combined_gravitational_force_l128_128724

theorem combined_gravitational_force 
    (d_E_surface : ℝ) (f_E_surface : ℝ) (d_M_surface : ℝ) (f_M_surface : ℝ) 
    (d_E_new : ℝ) (d_M_new : ℝ) 
    (k_E : ℝ) (k_M : ℝ) 
    (h1 : k_E = f_E_surface * d_E_surface^2)
    (h2 : k_M = f_M_surface * d_M_surface^2)
    (h3 : f_E_new = k_E / d_E_new^2)
    (h4 : f_M_new = k_M / d_M_new^2) : 
  f_E_new + f_M_new = 755.7696 :=
by
  sorry

end combined_gravitational_force_l128_128724


namespace inequality_solution_l128_128656

theorem inequality_solution {x : ℝ} :
  {x | (2 * x - 8) * (x - 4) / x ≥ 0} = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end inequality_solution_l128_128656


namespace train_speed_l128_128037

noncomputable def distance : ℝ := 45  -- 45 km
noncomputable def time_minutes : ℝ := 30  -- 30 minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert minutes to hours

theorem train_speed (d : ℝ) (t_m : ℝ) : d = 45 → t_m = 30 → d / (t_m / 60) = 90 :=
by
  intros h₁ h₂
  sorry

end train_speed_l128_128037


namespace knocks_to_knicks_l128_128335

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end knocks_to_knicks_l128_128335


namespace find_point_B_l128_128035

-- Definition of Point
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of conditions
def A : Point := ⟨1, 2⟩
def d : ℝ := 3
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

theorem find_point_B (B : Point) (h_parallel : AB_parallel_x A B) (h_dist : abs (B.x - A.x) = d) :
  (B = ⟨4, 2⟩) ∨ (B = ⟨-2, 2⟩) :=
by
  sorry

end find_point_B_l128_128035


namespace possible_values_of_c_l128_128565

-- Definition of c(S) based on the problem conditions
def c (S : String) (m : ℕ) : ℕ := sorry

-- Condition: m > 1
variable {m : ℕ} (hm : m > 1)

-- Goal: To prove the possible values that c(S) can take
theorem possible_values_of_c (S : String) : ∃ n : ℕ, c S m = 0 ∨ c S m = 2^n :=
sorry

end possible_values_of_c_l128_128565


namespace count_integer_points_l128_128164

-- Define the conditions: the parabola P with focus at (0,0) and passing through (6,4) and (-6,-4)
def parabola (P : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y : ℝ, P (x, y) ↔ y = a*x^2 + b) ∧ 
  P (6, 4) ∧ P (-6, -4)

-- Define the main theorem to be proved: the count of integer points satisfying the inequality
theorem count_integer_points (P : ℝ × ℝ → Prop) (hP : parabola P) :
  ∃ n : ℕ, n = 45 ∧ ∀ (x y : ℤ), P (x, y) → |6 * x + 4 * y| ≤ 1200 :=
sorry

end count_integer_points_l128_128164


namespace total_tickets_needed_l128_128097

-- Define the conditions
def rollercoaster_rides (n : Nat) := 3
def catapult_rides (n : Nat) := 2
def ferris_wheel_rides (n : Nat) := 1
def rollercoaster_cost (n : Nat) := 4
def catapult_cost (n : Nat) := 4
def ferris_wheel_cost (n : Nat) := 1

-- Prove the total number of tickets needed
theorem total_tickets_needed : 
  rollercoaster_rides 0 * rollercoaster_cost 0 +
  catapult_rides 0 * catapult_cost 0 +
  ferris_wheel_rides 0 * ferris_wheel_cost 0 = 21 :=
by 
  sorry

end total_tickets_needed_l128_128097


namespace roots_calc_l128_128929

theorem roots_calc {a b c d : ℝ} (h1: a ≠ 0) (h2 : 125 * a + 25 * b + 5 * c + d = 0) (h3 : -27 * a + 9 * b - 3 * c + d = 0) :
  (b + c) / a = -19 :=
by
  sorry

end roots_calc_l128_128929


namespace circle_equation_tangent_line_l128_128353

theorem circle_equation_tangent_line :
  ∃ r : ℝ, ∀ x y : ℝ, (x - 3)^2 + (y + 5)^2 = r^2 ↔ x - 7 * y + 2 = 0 :=
sorry

end circle_equation_tangent_line_l128_128353


namespace range_of_a_l128_128018

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (a > (1 / 2))

theorem range_of_a (hpq_true: p a ∨ q a) (hpq_false: ¬ (p a ∧ q a)) :
  (0 < a ∧ a ≤ (1 / 2)) ∨ (a ≥ 1) :=
sorry

end range_of_a_l128_128018


namespace negation_of_exists_lt_l128_128198

theorem negation_of_exists_lt :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
by sorry

end negation_of_exists_lt_l128_128198


namespace product_ne_sum_11_times_l128_128438

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0
def prime_sum_product_condition (a b c d : ℕ) : Prop := 
  a * b * c * d = 11 * (a + b + c + d)

theorem product_ne_sum_11_times (a b c d : ℕ)
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (hd : is_prime d)
  (h : prime_sum_product_condition a b c d) :
  (a + b + c + d ≠ 46) ∧ (a + b + c + d ≠ 47) ∧ (a + b + c + d ≠ 48) :=
by  
  sorry

end product_ne_sum_11_times_l128_128438


namespace least_possible_product_of_distinct_primes_gt_50_l128_128716

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_possible_product_of_distinct_primes_gt_50_l128_128716


namespace vanya_correct_answers_l128_128935

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l128_128935


namespace remainder_of_division_l128_128730

def dividend := 1234567
def divisor := 257

theorem remainder_of_division : dividend % divisor = 774 :=
by
  sorry

end remainder_of_division_l128_128730


namespace find_m_l128_128154

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0

theorem find_m (m : ℝ) (h1 : circle_equation (-1) 3) (h2 : symmetric_line (-1) 3 m) : m = -1 := by
  sorry

end find_m_l128_128154


namespace find_k_l128_128375

theorem find_k (k : ℤ) :
  (-x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) → k = -16 :=
by
  sorry

end find_k_l128_128375


namespace least_possible_integer_l128_128306

theorem least_possible_integer :
  ∃ N : ℕ,
    (∀ k, 1 ≤ k ∧ k ≤ 30 → k ≠ 24 → k ≠ 25 → N % k = 0) ∧
    (N % 24 ≠ 0) ∧
    (N % 25 ≠ 0) ∧
    N = 659375723440 :=
by
  sorry

end least_possible_integer_l128_128306


namespace largest_prime_divisor_l128_128482

theorem largest_prime_divisor : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (17^2 + 60^2) → q ≤ p :=
  sorry

end largest_prime_divisor_l128_128482


namespace solve_inequality_l128_128654

theorem solve_inequality :
  {x : ℝ | (x - 3)*(x - 4)*(x - 5) / ((x - 2)*(x - 6)*(x - 7)) > 0} =
  {x : ℝ | x < 2} ∪ {x : ℝ | 4 < x ∧ x < 5} ∪ {x : ℝ | 6 < x ∧ x < 7} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end solve_inequality_l128_128654


namespace range_of_x_l128_128290

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x else 2 * -x

theorem range_of_x {x : ℝ} :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l128_128290


namespace sequence_explicit_formula_l128_128914

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end sequence_explicit_formula_l128_128914


namespace pond_water_amount_l128_128957

-- Definitions based on the problem conditions
def initial_gallons := 500
def evaporation_rate := 1
def additional_gallons := 10
def days_period := 35
def additional_days_interval := 7

-- Calculations based on the conditions
def total_evaporation := days_period * evaporation_rate
def total_additional_gallons := (days_period / additional_days_interval) * additional_gallons

-- Theorem stating the final amount of water
theorem pond_water_amount : initial_gallons - total_evaporation + total_additional_gallons = 515 := by
  -- Proof is omitted
  sorry

end pond_water_amount_l128_128957


namespace solve_for_m_l128_128619

theorem solve_for_m (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 :=
by {
  sorry
}

end solve_for_m_l128_128619


namespace acute_angle_comparison_l128_128763

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem acute_angle_comparison (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (f_even : even_function f)
  (f_periodic : ∀ x, f (x + 1) + f x = 0)
  (f_increasing : increasing_on_interval f 3 4) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end acute_angle_comparison_l128_128763


namespace proof_inequality_l128_128102

theorem proof_inequality (x : ℝ) : (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5 ∨ -9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end proof_inequality_l128_128102


namespace no_solution_intervals_l128_128363

theorem no_solution_intervals :
    ¬ ∃ x : ℝ, (2 / 3 < x ∧ x < 4 / 3) ∧ (1 / 5 < x ∧ x < 3 / 5) :=
by
  sorry

end no_solution_intervals_l128_128363


namespace math_test_total_questions_l128_128305

theorem math_test_total_questions (Q : ℕ) (h : Q - 38 = 7) : Q = 45 :=
by
  sorry

end math_test_total_questions_l128_128305


namespace inequality_solution_l128_128926

theorem inequality_solution (x : ℝ) (h : 0 < x) : x^3 - 9*x^2 + 52*x > 0 := 
sorry

end inequality_solution_l128_128926


namespace compute_factorial_expression_l128_128240

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem compute_factorial_expression :
  factorial 9 - factorial 8 - factorial 7 + factorial 6 = 318240 := by
  sorry

end compute_factorial_expression_l128_128240


namespace original_square_area_is_144_square_centimeters_l128_128081

noncomputable def area_of_original_square (x : ℝ) : ℝ :=
  x^2 - (x - 3) * (x - 5)

theorem original_square_area_is_144_square_centimeters (x : ℝ) (h : area_of_original_square x = 81) :
  (x = 12) → (x^2 = 144) :=
by
  sorry

end original_square_area_is_144_square_centimeters_l128_128081


namespace find_multiplier_l128_128630

theorem find_multiplier (x : ℝ) (h : (9 / 6) * x = 18) : x = 12 := sorry

end find_multiplier_l128_128630


namespace initial_price_after_markup_l128_128958

theorem initial_price_after_markup 
  (wholesale_price : ℝ) 
  (h_markup_80 : ∀ P, P = wholesale_price → 1.80 * P = 1.80 * wholesale_price)
  (h_markup_diff : ∀ P, P = wholesale_price → 2.00 * P - 1.80 * P = 3) 
  : 1.80 * wholesale_price = 27 := 
by
  sorry

end initial_price_after_markup_l128_128958


namespace find_initial_red_marbles_l128_128615

theorem find_initial_red_marbles (x y : ℚ) 
  (h1 : 2 * x = 3 * y) 
  (h2 : 5 * (x - 15) = 2 * (y + 25)) 
  : x = 375 / 11 := 
by
  sorry

end find_initial_red_marbles_l128_128615


namespace trajectory_of_P_eqn_l128_128273

theorem trajectory_of_P_eqn :
  ∀ {x y : ℝ}, -- For all real numbers x and y
  (-(x + 2)^2 + (x - 1)^2 + y^2 = 3*((x - 1)^2 + y^2)) → -- Condition |PA| = 2|PB|
  (x^2 + y^2 - 4*x = 0) := -- Prove the trajectory equation
by
  intros x y h
  sorry -- Proof to be completed

end trajectory_of_P_eqn_l128_128273


namespace total_money_spent_l128_128256

-- Definitions based on conditions
def num_bars_of_soap : Nat := 20
def weight_per_bar_of_soap : Float := 1.5
def cost_per_pound_of_soap : Float := 0.5

def num_bottles_of_shampoo : Nat := 15
def weight_per_bottle_of_shampoo : Float := 2.2
def cost_per_pound_of_shampoo : Float := 0.8

-- The theorem to prove
theorem total_money_spent :
  let cost_per_bar_of_soap := weight_per_bar_of_soap * cost_per_pound_of_soap
  let total_cost_of_soap := Float.ofNat num_bars_of_soap * cost_per_bar_of_soap
  let cost_per_bottle_of_shampoo := weight_per_bottle_of_shampoo * cost_per_pound_of_shampoo
  let total_cost_of_shampoo := Float.ofNat num_bottles_of_shampoo * cost_per_bottle_of_shampoo
  total_cost_of_soap + total_cost_of_shampoo = 41.40 := 
by
  -- proof goes here
  sorry

end total_money_spent_l128_128256


namespace difference_of_smallest_integers_l128_128245

theorem difference_of_smallest_integers (n_1 n_2: ℕ) (h1 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_1 > 1 ∧ n_1 % k = 1)) (h2 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_2 > 1 ∧ n_2 % k = 1)) (h_smallest : n_1 = 61) (h_second_smallest : n_2 = 121) : n_2 - n_1 = 60 :=
by
  sorry

end difference_of_smallest_integers_l128_128245


namespace probability_x_gt_3y_correct_l128_128718

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end probability_x_gt_3y_correct_l128_128718


namespace trihedral_angle_properties_l128_128561

-- Definitions for the problem's conditions
variables {α β γ : ℝ}
variables {A B C S : Type}
variables (angle_ASB angle_BSC angle_CSA : ℝ)

-- Given the conditions of the trihedral angle and the dihedral angles
theorem trihedral_angle_properties 
  (h1 : angle_ASB + angle_BSC + angle_CSA < 2 * Real.pi)
  (h2 : α + β + γ > Real.pi) : 
  true := 
by
  sorry

end trihedral_angle_properties_l128_128561


namespace smallest_integer_condition_l128_128408

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l128_128408


namespace salary_based_on_tax_l128_128346

theorem salary_based_on_tax (salary tax paid_tax excess_800 excess_500 excess_500_2000 : ℤ) 
    (h1 : excess_800 = salary - 800)
    (h2 : excess_500 = min excess_800 500)
    (h3 : excess_500_2000 = excess_800 - excess_500)
    (h4 : paid_tax = (excess_500 * 5 / 100) + (excess_500_2000 * 10 / 100))
    (h5 : paid_tax = 80) :
  salary = 1850 := by
  sorry

end salary_based_on_tax_l128_128346


namespace fill_table_with_numbers_l128_128711

-- Define the main theorem based on the conditions and question.
theorem fill_table_with_numbers (numbers : Finset ℤ) (table : ℕ → ℕ → ℤ)
  (h_numbers_card : numbers.card = 100)
  (h_sum_1x3_horizontal : ∀ i j, (table i j + table i (j + 1) + table i (j + 2) ∈ numbers))
  (h_sum_1x3_vertical : ∀ i j, (table i j + table (i + 1) j + table (i + 2) j ∈ numbers)):
  ∃ (t : ℕ → ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ 6 → ∃ i j, t i j = k) :=
sorry

end fill_table_with_numbers_l128_128711


namespace correct_option_l128_128264

theorem correct_option :
  (3 * a^2 + 5 * a^2 ≠ 8 * a^4) ∧
  (5 * a^2 * b - 6 * a * b^2 ≠ -a * b^2) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (9 * x * y - 6 * x * y = 3 * x * y) :=
by
  sorry

end correct_option_l128_128264


namespace modified_cube_surface_area_l128_128027

noncomputable def total_surface_area_modified_cube : ℝ :=
  let side_length := 10
  let triangle_side := 7 * Real.sqrt 2
  let tunnel_wall_area := 3 * (Real.sqrt 3 / 4 * triangle_side^2)
  let original_surface_area := 6 * side_length^2
  original_surface_area + tunnel_wall_area

theorem modified_cube_surface_area : 
  total_surface_area_modified_cube = 600 + 73.5 * Real.sqrt 3 := 
  sorry

end modified_cube_surface_area_l128_128027


namespace gcd_of_power_of_two_plus_one_l128_128225

theorem gcd_of_power_of_two_plus_one (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := 
sorry

end gcd_of_power_of_two_plus_one_l128_128225


namespace analytical_expression_f_min_value_f_range_of_k_l128_128117

noncomputable def max_real (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
  max_real (|x + 1|) (|x - 2|)

noncomputable def g (x k : ℝ) : ℝ :=
  x^2 - k * f x

-- Problem 1: Proving the analytical expression of f(x)
theorem analytical_expression_f (x : ℝ) :
  f x = if x < 0.5 then 2 - x else x + 1 :=
sorry

-- Problem 2: Proving the minimum value of f(x)
theorem min_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 3 / 2 :=
sorry

-- Problem 3: Proving the range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (g x k) ≤ (g (x - 1) k)) → k ≤ 2 :=
sorry

end analytical_expression_f_min_value_f_range_of_k_l128_128117


namespace product_of_numbers_l128_128315

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := 
by 
  sorry

end product_of_numbers_l128_128315


namespace cos_sum_identity_l128_128765

theorem cos_sum_identity (α : ℝ) (h_cos : Real.cos α = 3 / 5) (h_alpha : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (α + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 :=
by
  sorry

end cos_sum_identity_l128_128765


namespace Tom_Brady_passing_yards_l128_128423

-- Definitions
def record := 5999
def current_yards := 4200
def games_left := 6

-- Proof problem statement
theorem Tom_Brady_passing_yards :
  (record + 1 - current_yards) / games_left = 300 := by
  sorry

end Tom_Brady_passing_yards_l128_128423


namespace fraction_C_D_l128_128021

noncomputable def C : ℝ := ∑' n, if n % 6 = 0 then 0 else if n % 2 = 0 then ((-1)^(n/2 + 1) / (↑n^2)) else 0
noncomputable def D : ℝ := ∑' n, if n % 6 = 0 then ((-1)^(n/6 + 1) / (↑n^2)) else 0

theorem fraction_C_D : C / D = 37 := sorry

end fraction_C_D_l128_128021


namespace fraction_ratio_l128_128063

variable (M Q P N R : ℝ)

theorem fraction_ratio (h1 : M = 0.40 * Q)
                       (h2 : Q = 0.25 * P)
                       (h3 : N = 0.40 * R)
                       (h4 : R = 0.75 * P) :
  M / N = 1 / 3 := 
by
  -- proof steps can be provided here
  sorry

end fraction_ratio_l128_128063


namespace makenna_garden_larger_by_160_l128_128901

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def karl_length : ℕ := 22
def karl_width : ℕ := 50
def makenna_length : ℕ := 28
def makenna_width : ℕ := 45

def karl_area : ℕ := area karl_length karl_width
def makenna_area : ℕ := area makenna_length makenna_width

theorem makenna_garden_larger_by_160 :
  makenna_area = karl_area + 160 := by
  sorry

end makenna_garden_larger_by_160_l128_128901


namespace cheryl_more_eggs_than_others_l128_128331

def kevin_eggs : ℕ := 5
def bonnie_eggs : ℕ := 13
def george_eggs : ℕ := 9
def cheryl_eggs : ℕ := 56

theorem cheryl_more_eggs_than_others : cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 :=
by
  sorry

end cheryl_more_eggs_than_others_l128_128331


namespace monotone_decreasing_sequence_monotone_increasing_sequence_l128_128780

theorem monotone_decreasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) < a n) ↔ c < 0 :=
by sorry

theorem monotone_increasing_sequence (f : ℝ → ℝ) (a : ℕ → ℝ) (c : ℝ) :
  (∀ n : ℕ, a (n + 1) = f (a n)) →
  (a 1 = 0) →
  (∀ x : ℝ, f x = f (1 - x)) →
  (∀ x : ℝ, f x = -x^2 + x + c) →
  (∀ n : ℕ, a (n + 1) > a n) ↔ c > 1/4 :=
by sorry

end monotone_decreasing_sequence_monotone_increasing_sequence_l128_128780


namespace number_of_people_in_first_group_l128_128965

variable (W : ℝ)  -- Amount of work
variable (P : ℝ)  -- Number of people in the first group

-- Condition 1: P people can do 3W work in 3 days
def condition1 : Prop := P * (W / 1) * 3 = 3 * W

-- Condition 2: 5 people can do 5W work in 3 days
def condition2 : Prop := 5 * (W / 1) * 3 = 5 * W

-- Theorem to prove: The number of people in the first group is 3
theorem number_of_people_in_first_group (h1 : condition1 W P) (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l128_128965


namespace find_x_plus_y_l128_128396

theorem find_x_plus_y (x y : ℕ) 
  (h1 : 4^x = 16^(y + 1)) 
  (h2 : 5^(2 * y) = 25^(x - 2)) : 
  x + y = 2 := 
sorry

end find_x_plus_y_l128_128396


namespace natural_numbers_fitting_description_l128_128175

theorem natural_numbers_fitting_description (n : ℕ) (h : 1 / (n : ℚ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) : n = 2 ∨ n = 3 :=
by
  sorry

end natural_numbers_fitting_description_l128_128175


namespace valid_numbers_l128_128195

noncomputable def is_valid_number (a : ℕ) : Prop :=
  ∃ b c d x y : ℕ, 
    a = b * c + d ∧
    a = 10 * x + y ∧
    x > 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧
    10 * x + y = 4 * x + 4 * y

theorem valid_numbers : 
  ∃ a : ℕ, (a = 12 ∨ a = 24 ∨ a = 36 ∨ a = 48) ∧ is_valid_number a :=
by
  sorry

end valid_numbers_l128_128195


namespace suff_but_not_necessary_condition_l128_128976

theorem suff_but_not_necessary_condition (x y : ℝ) :
  (xy ≠ 6 → x ≠ 2 ∨ y ≠ 3) ∧ ¬ (x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) :=
by
  sorry

end suff_but_not_necessary_condition_l128_128976


namespace f_properties_l128_128220

variable (f : ℝ → ℝ)
variable (f_pos : ∀ x : ℝ, f x > 0)
variable (f_eq : ∀ a b : ℝ, f a * f b = f (a + b))

theorem f_properties :
  (f 0 = 1) ∧
  (∀ a : ℝ, f (-a) = 1 / f a) ∧
  (∀ a : ℝ, f a = (f (3 * a))^(1/3)) :=
by {
  sorry
}

end f_properties_l128_128220


namespace proposition_false_at_4_l128_128997

theorem proposition_false_at_4 (P : ℕ → Prop) (hp : ∀ k : ℕ, k > 0 → (P k → P (k + 1))) (h4 : ¬ P 5) : ¬ P 4 :=
by {
    sorry
}

end proposition_false_at_4_l128_128997


namespace difference_of_cats_l128_128026

-- Definitions based on given conditions
def number_of_cats_sheridan : ℕ := 11
def number_of_cats_garrett : ℕ := 24

-- Theorem statement (proof problem) based on the question and correct answer
theorem difference_of_cats : (number_of_cats_garrett - number_of_cats_sheridan) = 13 := by
  sorry

end difference_of_cats_l128_128026


namespace percentage_problem_l128_128728

variable (N P : ℝ)

theorem percentage_problem (h1 : 0.3 * N = 120) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_problem_l128_128728


namespace gcd_45345_34534_l128_128665

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end gcd_45345_34534_l128_128665


namespace rhombus_compression_problem_l128_128337

def rhombus_diagonal_lengths (side longer_diagonal : ℝ) (compression : ℝ) : ℝ × ℝ :=
  let new_longer_diagonal := longer_diagonal - compression
  let new_shorter_diagonal := 1.2 * compression + 24
  (new_longer_diagonal, new_shorter_diagonal)

theorem rhombus_compression_problem :
  let side := 20
  let longer_diagonal := 32
  let compression := 2.62
  rhombus_diagonal_lengths side longer_diagonal compression = (29.38, 27.14) :=
by sorry

end rhombus_compression_problem_l128_128337


namespace retirement_amount_l128_128609

-- Define the principal amount P
def P : ℝ := 750000

-- Define the annual interest rate r
def r : ℝ := 0.08

-- Define the time period in years t
def t : ℝ := 12

-- Define the accumulated amount A
def A : ℝ := P * (1 + r * t)

-- Prove that the accumulated amount A equals 1470000
theorem retirement_amount : A = 1470000 := by
  -- The proof will involve calculating the compound interest
  sorry

end retirement_amount_l128_128609


namespace triangle_perimeter_problem_l128_128573

theorem triangle_perimeter_problem : 
  ∀ (c : ℝ), 20 + 15 > c ∧ 20 + c > 15 ∧ 15 + c > 20 → ¬ (35 + c = 72) :=
by
  intros c h
  sorry

end triangle_perimeter_problem_l128_128573


namespace trigonometric_identity_l128_128491

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l128_128491


namespace trig_identity_example_l128_128710

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) -
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l128_128710


namespace train_speed_calculation_l128_128435

open Real

noncomputable def train_speed_in_kmph (V : ℝ) : ℝ := V * 3.6

theorem train_speed_calculation (L V : ℝ) (h1 : L = 16 * V) (h2 : L + 280 = 30 * V) :
  train_speed_in_kmph V = 72 :=
by
  sorry

end train_speed_calculation_l128_128435


namespace possible_values_of_a_l128_128045

def setA := {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | 2 * a - x > 1}
def complementB (a : ℝ) := {x : ℝ | x ≥ (2 * a - 1)}

theorem possible_values_of_a (a : ℝ) :
  (∀ x, x ∈ setA → x ∈ complementB a) ↔ (a = -2 ∨ a = 0 ∨ a = 2) :=
by
  sorry

end possible_values_of_a_l128_128045


namespace candy_bar_calories_l128_128922

theorem candy_bar_calories :
  let calA := 150
  let calB := 200
  let calC := 250
  let countA := 2
  let countB := 3
  let countC := 4
  (countA * calA + countB * calB + countC * calC) = 1900 :=
by
  sorry

end candy_bar_calories_l128_128922


namespace opposite_neg_9_l128_128995

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l128_128995


namespace wrongly_noted_mark_is_90_l128_128100

-- Define the given conditions
def avg_marks (n : ℕ) (avg : ℚ) : ℚ := n * avg

def wrong_avg_marks : ℚ := avg_marks 10 100
def correct_avg_marks : ℚ := avg_marks 10 92

-- Equate the difference caused by the wrong mark
theorem wrongly_noted_mark_is_90 (x : ℚ) (h₁ : wrong_avg_marks = 1000) (h₂ : correct_avg_marks = 920) (h : x - 10 = 1000 - 920) : x = 90 := 
by {
  -- Proof goes here
  sorry
}

end wrongly_noted_mark_is_90_l128_128100


namespace rectangle_circle_area_ratio_l128_128740

theorem rectangle_circle_area_ratio {d : ℝ} (h : d > 0) :
  let A_rectangle := 2 * d * d
  let A_circle := (π * d^2) / 4
  (A_rectangle / A_circle) = (8 / π) :=
by
  sorry

end rectangle_circle_area_ratio_l128_128740


namespace volume_of_tetrahedron_l128_128889

theorem volume_of_tetrahedron 
  (A B C D E : ℝ)
  (AB AD AE: ℝ)
  (h_AB : AB = 3)
  (h_AD : AD = 4)
  (h_AE : AE = 1)
  (V : ℝ) :
  (V = (4 * Real.sqrt 3) / 3) :=
sorry

end volume_of_tetrahedron_l128_128889


namespace kim_hours_of_classes_per_day_l128_128507

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l128_128507


namespace simplify_expression_l128_128385

variable (x y : ℝ)

theorem simplify_expression :
  (2 * x + 3 * y) ^ 2 - 2 * x * (2 * x - 3 * y) = 18 * x * y + 9 * y ^ 2 :=
by
  sorry

end simplify_expression_l128_128385


namespace cheaper_joint_work_l128_128941

theorem cheaper_joint_work (r L P : ℝ) (hr_pos : 0 < r) (hL_pos : 0 < L) (hP_pos : 0 < P) : 
  (2 * P * L) / (3 * r) < (3 * P * L) / (4 * r) :=
by
  sorry

end cheaper_joint_work_l128_128941


namespace tiffany_lives_l128_128852

theorem tiffany_lives (initial_lives lives_lost lives_after_next_level lives_gained : ℕ)
  (h1 : initial_lives = 43)
  (h2 : lives_lost = 14)
  (h3 : lives_after_next_level = 56)
  (h4 : lives_gained = lives_after_next_level - (initial_lives - lives_lost)) :
  lives_gained = 27 :=
by {
  sorry
}

end tiffany_lives_l128_128852


namespace graduation_messages_total_l128_128839

/-- Define the number of students in the class -/
def num_students : ℕ := 40

/-- Define the combination formula C(n, 2) for choosing 2 out of n -/
def combination (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Prove that the total number of graduation messages written is 1560 -/
theorem graduation_messages_total : combination num_students = 1560 :=
by
  sorry

end graduation_messages_total_l128_128839


namespace original_triangle_area_l128_128467

theorem original_triangle_area (A_new : ℝ) (scale_factor : ℝ) (A_original : ℝ) 
  (h1: scale_factor = 5) (h2: A_new = 200) (h3: A_new = scale_factor^2 * A_original) : 
  A_original = 8 :=
by
  sorry

end original_triangle_area_l128_128467


namespace change_occurs_in_3_years_l128_128944

theorem change_occurs_in_3_years (P A1 A2 : ℝ) (R T : ℝ) (h1 : P = 825) (h2 : A1 = 956) (h3 : A2 = 1055)
    (h4 : A1 = P + (P * R * T) / 100)
    (h5 : A2 = P + (P * (R + 4) * T) / 100) : T = 3 :=
by
  sorry

end change_occurs_in_3_years_l128_128944


namespace find_common_ratio_l128_128689

variable {a : ℕ → ℝ} {q : ℝ}

-- Define that a is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : 0 < q)
  (h3 : a 1 * a 3 = 1)
  (h4 : sum_first_n_terms a 3 = 7) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l128_128689


namespace find_minimal_sum_l128_128084

theorem find_minimal_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * (x + 1)) ∣ (y * (y + 1)) →
  ¬(x ∣ y ∨ x ∣ (y + 1)) →
  ¬((x + 1) ∣ y ∨ (x + 1) ∣ (y + 1)) →
  x = 14 ∧ y = 35 ∧ x^2 + y^2 = 1421 :=
sorry

end find_minimal_sum_l128_128084


namespace holds_for_even_positive_l128_128681

variable {n : ℕ}
variable (p : ℕ → Prop)

-- Conditions
axiom base_case : p 2
axiom inductive_step : ∀ k, p k → p (k + 2)

-- Theorem to prove
theorem holds_for_even_positive (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : p n :=
sorry

end holds_for_even_positive_l128_128681


namespace transform_unit_square_l128_128680

-- Define the unit square vertices in the xy-plane
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Transformation functions from the xy-plane to the uv-plane
def transform_u (x y : ℝ) : ℝ := x^2 - y^2
def transform_v (x y : ℝ) : ℝ := x * y

-- Vertex transformation results
def O_image : ℝ × ℝ := (transform_u 0 0, transform_v 0 0)  -- (0,0)
def A_image : ℝ × ℝ := (transform_u 1 0, transform_v 1 0)  -- (1,0)
def B_image : ℝ × ℝ := (transform_u 1 1, transform_v 1 1)  -- (0,1)
def C_image : ℝ × ℝ := (transform_u 0 1, transform_v 0 1)  -- (-1,0)

-- The Lean 4 theorem statement
theorem transform_unit_square :
  O_image = (0, 0) ∧
  A_image = (1, 0) ∧
  B_image = (0, 1) ∧
  C_image = (-1, 0) :=
  by sorry

end transform_unit_square_l128_128680


namespace evaluate_fractions_l128_128826

-- Define the fractions
def frac1 := 7 / 12
def frac2 := 8 / 15
def frac3 := 2 / 5

-- Prove that the sum and difference is as specified
theorem evaluate_fractions :
  frac1 + frac2 - frac3 = 43 / 60 :=
by
  sorry

end evaluate_fractions_l128_128826


namespace correct_statement_l128_128713

def correct_input_format_1 (s : String) : Prop :=
  s = "INPUT a, b, c"

def correct_input_format_2 (s : String) : Prop :=
  s = "INPUT x="

def correct_output_format_1 (s : String) : Prop :=
  s = "PRINT A="

def correct_output_format_2 (s : String) : Prop :=
  s = "PRINT 3*2"

theorem correct_statement : (correct_input_format_1 "INPUT a; b; c" = false) ∧
                            (correct_input_format_2 "INPUT x=3" = false) ∧
                            (correct_output_format_1 "PRINT“A=4”" = false) ∧
                            (correct_output_format_2 "PRINT 3*2" = true) :=
by sorry

end correct_statement_l128_128713


namespace number_of_lists_l128_128751

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end number_of_lists_l128_128751


namespace inequality_subtraction_l128_128166

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l128_128166


namespace binom_9_5_l128_128891

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l128_128891


namespace min_value_of_a_plus_2b_l128_128192

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : a + 2*b = 3 + 2*Real.sqrt 2 := 
sorry

end min_value_of_a_plus_2b_l128_128192


namespace previous_year_profit_percentage_l128_128534

variables {R P: ℝ}

theorem previous_year_profit_percentage (h1: R > 0)
    (h2: P = 0.1 * R)
    (h3: 0.7 * P = 0.07 * R) :
    (P / R) * 100 = 10 :=
by
  -- Since we have P = 0.1 * R from the conditions and definitions,
  -- it follows straightforwardly that (P / R) * 100 = 10.
  -- We'll continue the proof from here.
  sorry

end previous_year_profit_percentage_l128_128534


namespace diamond_of_2_and_3_l128_128672

def diamond (a b : ℕ) : ℕ := a^3 * b^2 - b + 2

theorem diamond_of_2_and_3 : diamond 2 3 = 71 := by
  sorry

end diamond_of_2_and_3_l128_128672


namespace zoey_finishes_on_monday_l128_128149

def total_reading_days (books : ℕ) : ℕ :=
  (books * (books + 1)) / 2 + books

def day_of_week (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem zoey_finishes_on_monday : 
  day_of_week 2 (total_reading_days 20) = 1 :=
by
  -- Definitions
  let books := 20
  let start_day := 2 -- Corresponding to Tuesday
  let days := total_reading_days books
  
  -- Prove day_of_week 2 (total_reading_days 20) = 1
  sorry

end zoey_finishes_on_monday_l128_128149


namespace determine_a_l128_128009

theorem determine_a (x y a : ℝ) 
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) : 
  a = 0 := 
sorry

end determine_a_l128_128009


namespace servings_per_bottle_l128_128383

-- Definitions based on conditions
def total_guests : ℕ := 120
def servings_per_guest : ℕ := 2
def total_bottles : ℕ := 40

-- Theorem stating that given the conditions, the servings per bottle is 6
theorem servings_per_bottle : (total_guests * servings_per_guest) / total_bottles = 6 := by
  sorry

end servings_per_bottle_l128_128383


namespace evaluate_power_l128_128558

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l128_128558


namespace combined_age_l128_128631

theorem combined_age (H : ℕ) (Ryanne : ℕ) (Jamison : ℕ) 
  (h1 : Ryanne = H + 7) 
  (h2 : H + Ryanne = 15) 
  (h3 : Jamison = 2 * H) : 
  H + Ryanne + Jamison = 23 := 
by 
  sorry

end combined_age_l128_128631


namespace common_z_values_l128_128688

theorem common_z_values (z : ℝ) :
  (∃ x : ℝ, x^2 + z^2 = 9 ∧ x^2 = 4*z - 5) ↔ (z = -2 + 3*Real.sqrt 2 ∨ z = -2 - 3*Real.sqrt 2) := 
sorry

end common_z_values_l128_128688


namespace initial_number_of_earning_members_l128_128873

theorem initial_number_of_earning_members (n : ℕ) 
  (h1 : 840 * n - 650 * (n - 1) = 1410) : n = 4 :=
by {
  -- Proof omitted
  sorry
}

end initial_number_of_earning_members_l128_128873


namespace markup_percentage_is_ten_l128_128463

theorem markup_percentage_is_ten (S C : ℝ)
  (h1 : S - C = 0.0909090909090909 * S) :
  (S - C) / C * 100 = 10 :=
by
  sorry

end markup_percentage_is_ten_l128_128463


namespace fraction_of_price_l128_128360

theorem fraction_of_price (d : ℝ) : d * 0.65 * 0.70 = d * 0.455 :=
by
  sorry

end fraction_of_price_l128_128360


namespace math_problem_l128_128623

theorem math_problem 
  (a b c : ℕ) 
  (h_primea : Nat.Prime a)
  (h_posa : 0 < a)
  (h_posb : 0 < b)
  (h_posc : 0 < c)
  (h_eq : a^2 + b^2 = c^2) :
  (b % 2 ≠ c % 2) ∧ (∃ k, 2 * (a + b + 1) = k^2) := 
sorry

end math_problem_l128_128623


namespace largest_alternating_geometric_four_digit_number_l128_128679

theorem largest_alternating_geometric_four_digit_number :
  ∃ (a b c d : ℕ), 
  (9 = 2 * b) ∧ (b = 2 * c) ∧ (a = 3) ∧ (9 * d = b * c) ∧ 
  (a > b) ∧ (b < c) ∧ (c > d) ∧ (1000 * a + 100 * b + 10 * c + d = 9632) := sorry

end largest_alternating_geometric_four_digit_number_l128_128679


namespace find_f_l128_128832

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (h : ∀ x, x ≠ -1 → f ((1-x) / (1+x)) = (1 - x^2) / (1 + x^2)) 
               (hx : x ≠ -1) :
  f x = 2 * x / (1 + x^2) :=
sorry

end find_f_l128_128832


namespace rectangle_to_rhombus_l128_128668

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ D.2 = C.2 ∧ C.1 = B.1 ∧ B.2 = A.2

def is_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) ≠ 0

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

theorem rectangle_to_rhombus (A B C D : ℝ × ℝ) (h1 : is_rectangle A B C D) :
  ∃ X Y Z W : ℝ × ℝ, is_triangle A B C ∧ is_triangle A D C ∧ is_rhombus X Y Z W :=
by
  sorry

end rectangle_to_rhombus_l128_128668


namespace fraction_incorrect_like_music_l128_128028

-- Define the conditions as given in the problem
def total_students : ℕ := 100
def like_music_percentage : ℝ := 0.7
def dislike_music_percentage : ℝ := 1 - like_music_percentage

def correct_like_percentage : ℝ := 0.75
def incorrect_like_percentage : ℝ := 1 - correct_like_percentage

def correct_dislike_percentage : ℝ := 0.85
def incorrect_dislike_percentage : ℝ := 1 - correct_dislike_percentage

-- The number of students liking music
def like_music_students : ℝ := total_students * like_music_percentage
-- The number of students disliking music
def dislike_music_students : ℝ := total_students * dislike_music_percentage

-- The number of students who correctly say they like music
def correct_like_music_say : ℝ := like_music_students * correct_like_percentage
-- The number of students who incorrectly say they dislike music
def incorrect_dislike_music_say : ℝ := like_music_students * incorrect_like_percentage

-- The number of students who correctly say they dislike music
def correct_dislike_music_say : ℝ := dislike_music_students * correct_dislike_percentage
-- The number of students who incorrectly say they like music
def incorrect_like_music_say : ℝ := dislike_music_students * incorrect_dislike_percentage

-- The total number of students who say they like music
def total_say_like_music : ℝ := correct_like_music_say + incorrect_like_music_say

-- The final theorem we want to prove
theorem fraction_incorrect_like_music : ((incorrect_like_music_say : ℝ) / total_say_like_music) = (5 / 58) :=
by
  -- here we would provide the proof, but for now, we use sorry
  sorry

end fraction_incorrect_like_music_l128_128028


namespace sufficient_but_not_necessary_condition_l128_128450

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l128_128450


namespace probability_truth_or_lies_l128_128252

def probability_truth := 0.30
def probability_lies := 0.20
def probability_both := 0.10

theorem probability_truth_or_lies :
  (probability_truth + probability_lies - probability_both) = 0.40 :=
by
  sorry

end probability_truth_or_lies_l128_128252


namespace complement_union_l128_128868

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ) (hA : A = { x | x < 0 }) (hB : B = { x | x ≥ 2 }) :
  C_U U (A ∪ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end complement_union_l128_128868


namespace largest_vertex_sum_l128_128794

def parabola_vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := T
  let y_vertex := a * T^2 - 2 * a * T^2
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (hA : 0 = a * 0^2 + 0 * 0 + 0)
  (hB : 0 = a * (2 * T)^2 + (2 * T) * (2 * -T))
  (hC : 36 = a * (2 * T + 1)^2 + (2 * T - 2 * T * (2 * T + 1)))
  : parabola_vertex_sum a T hT ≤ -14 :=
sorry

end largest_vertex_sum_l128_128794


namespace prob_a_prob_b_l128_128024

-- Given conditions and question for Part a
def election_prob (p q : ℕ) (h : p > q) : ℚ :=
  (p - q) / (p + q)

theorem prob_a : election_prob 3 2 (by decide) = 1 / 5 :=
  sorry

-- Given conditions and question for Part b
theorem prob_b : election_prob 1010 1009 (by decide) = 1 / 2019 :=
  sorry

end prob_a_prob_b_l128_128024


namespace increase_in_area_correct_l128_128046

-- Define the dimensions of the original rectangular garden
def length_rect := 60
def width_rect := 20

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Calculate the side length of the square garden using the same perimeter.
def side_square := perimeter_rect / 4

-- Define the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Define the area of the square garden
def area_square := side_square * side_square

-- Define the increase in area after reshaping
def increase_in_area := area_square - area_rect

-- Prove that the increase in the area is 400 square feet
theorem increase_in_area_correct : increase_in_area = 400 := by
  -- The proof is omitted
  sorry

end increase_in_area_correct_l128_128046


namespace mary_needs_more_sugar_l128_128369

theorem mary_needs_more_sugar 
  (sugar_needed flour_needed salt_needed already_added_flour : ℕ)
  (h1 : sugar_needed = 11)
  (h2 : flour_needed = 6)
  (h3 : salt_needed = 9)
  (h4 : already_added_flour = 12) :
  (sugar_needed - salt_needed) = 2 :=
by
  sorry

end mary_needs_more_sugar_l128_128369


namespace greatest_product_l128_128789

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l128_128789


namespace function_characterization_l128_128981

def isRelativelyPrime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem function_characterization (f : ℕ → ℤ) (hyp : ∀ x y, isRelativelyPrime x y → f (x + y) = f (x + 1) + f (y + 1)) :
  ∃ a b : ℤ, ∀ n : ℕ, f (2 * n) = (n - 1) * b ∧ f (2 * n + 1) = (n - 1) * b + a :=
by
  sorry

end function_characterization_l128_128981


namespace cos_plus_2sin_eq_one_l128_128310

theorem cos_plus_2sin_eq_one (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) : 
  Real.cos α + 2 * Real.sin α = 1 := 
by
  sorry

end cos_plus_2sin_eq_one_l128_128310


namespace five_digit_divisible_by_four_digit_l128_128359

theorem five_digit_divisible_by_four_digit (x y z u v : ℕ) (h1 : 1 ≤ x) (h2 : x < 10) (h3 : y < 10) (h4 : z < 10) (h5 : u < 10) (h6 : v < 10)
  (h7 : (x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v) % (x * 10^3 + y * 10^2 + u * 10 + v) = 0) : 
  ∃ N, 10 ≤ N ∧ N ≤ 99 ∧ 
  x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v = N * 10^3 ∧
  10 * (x * 10^3 + y * 10^2 + u * 10 + v) = x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v :=
sorry

end five_digit_divisible_by_four_digit_l128_128359


namespace order_theorems_l128_128888

theorem order_theorems : 
  ∃ a b c d e f g : String,
    (a = "H") ∧ (b = "M") ∧ (c = "P") ∧ (d = "C") ∧ 
    (e = "V") ∧ (f = "S") ∧ (g = "E") ∧
    (a = "Heron's Theorem") ∧
    (b = "Menelaus' Theorem") ∧
    (c = "Pascal's Theorem") ∧
    (d = "Ceva's Theorem") ∧
    (e = "Varignon's Theorem") ∧
    (f = "Stewart's Theorem") ∧
    (g = "Euler's Theorem") := 
  sorry

end order_theorems_l128_128888


namespace find_n_l128_128835

variable (a b c n : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)

theorem find_n (h1 : (a + b) / a = 3)
  (h2 : (b + c) / b = 4)
  (h3 : (c + a) / c = n) :
  n = 7 / 6 := 
sorry

end find_n_l128_128835


namespace find_angle_A_l128_128060

open Real

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = sqrt 2) 
  (hb : b = 2) 
  (hB : sin B + cos B = sqrt 2) :
  A = π / 6 := 
  sorry

end find_angle_A_l128_128060


namespace cylinder_volume_ratio_l128_128628

theorem cylinder_volume_ratio
  (h : ℝ)
  (r1 : ℝ)
  (r3 : ℝ := 3 * r1)
  (V1 : ℝ := 40) :
  let V2 := π * r3^2 * h
  (π * r1^2 * h = V1) → 
  V2 = 360 := by
{
  sorry
}

end cylinder_volume_ratio_l128_128628


namespace hyperbola_equation_l128_128406

theorem hyperbola_equation 
  (vertex : ℝ × ℝ) 
  (asymptote_slope : ℝ) 
  (h_vertex : vertex = (2, 0))
  (h_asymptote : asymptote_slope = Real.sqrt 2) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 8 = 1) := 
by
    sorry

end hyperbola_equation_l128_128406


namespace bread_last_days_l128_128825

def total_consumption_per_member_breakfast : ℕ := 4
def total_consumption_per_member_snacks : ℕ := 3
def total_consumption_per_member : ℕ := total_consumption_per_member_breakfast + total_consumption_per_member_snacks
def family_members : ℕ := 6
def daily_family_consumption : ℕ := family_members * total_consumption_per_member
def slices_per_loaf : ℕ := 10
def total_loaves : ℕ := 5
def total_bread_slices : ℕ := total_loaves * slices_per_loaf

theorem bread_last_days : total_bread_slices / daily_family_consumption = 1 :=
by
  sorry

end bread_last_days_l128_128825


namespace range_of_m_l128_128571

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioi 2, 0 ≤ deriv (f m) x) ↔ m ≤ 5 / 2 :=
sorry

end range_of_m_l128_128571


namespace triangle_ab_value_l128_128071

theorem triangle_ab_value (a b c : ℝ) (A B C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by
  sorry

end triangle_ab_value_l128_128071


namespace range_of_a_l128_128022

open Real

noncomputable def f (x : ℝ) := x - sqrt (x^2 + x)

noncomputable def g (x a : ℝ) := log x / log 27 - log x / log 9 + a * log x / log 3

theorem range_of_a (a : ℝ) : (∀ x1 ∈ Set.Ioi 1, ∃ x2 ∈ Set.Icc 3 9, f x1 > g x2 a) → a ≤ -1/12 :=
by
  intro h
  sorry

end range_of_a_l128_128022


namespace rain_on_Tuesday_correct_l128_128047

-- Let the amount of rain on Monday be represented by m
def rain_on_Monday : ℝ := 0.9

-- Let the difference in rain between Monday and Tuesday be represented by d
def rain_difference : ℝ := 0.7

-- Define the calculated amount of rain on Tuesday
def rain_on_Tuesday : ℝ := rain_on_Monday - rain_difference

-- The statement we need to prove
theorem rain_on_Tuesday_correct : rain_on_Tuesday = 0.2 := 
by
  -- Proof omitted (to be provided)
  sorry

end rain_on_Tuesday_correct_l128_128047


namespace fraction_identity_l128_128954

theorem fraction_identity (a b : ℚ) (h : (a - 2 * b) / b = 3 / 5) : a / b = 13 / 5 :=
sorry

end fraction_identity_l128_128954


namespace buratino_cafe_workdays_l128_128916

-- Define the conditions as given in the problem statement
def days_in_april (d : Nat) : Prop := d >= 1 ∧ d <= 30
def is_monday (d : Nat) : Prop := d = 1 ∨ d = 8 ∨ d = 15 ∨ d = 22 ∨ d = 29

-- Define the period April 1 to April 13
def period_1_13 (d : Nat) : Prop := d >= 1 ∧ d <= 13

-- Define the statements made by Kolya
def kolya_statement_1 : Prop := ∀ d : Nat, days_in_april d → (d >= 1 ∧ d <= 20) → ¬is_monday d → ∃ n : Nat, n = 18
def kolya_statement_2 : Prop := ∀ d : Nat, days_in_april d → (d >= 10 ∧ d <= 30) → ¬is_monday d → ∃ n : Nat, n = 18

-- Define the condition stating Kolya made a mistake once
def kolya_made_mistake_once : Prop := kolya_statement_1 ∨ kolya_statement_2

-- The proof problem: Prove the number of working days from April 1 to April 13 is 11
theorem buratino_cafe_workdays : period_1_13 (d) → (¬is_monday d → (∃ n : Nat, n = 11)) := sorry

end buratino_cafe_workdays_l128_128916


namespace technicians_in_workshop_l128_128356

theorem technicians_in_workshop (T R : ℕ) 
    (h1 : 700 * 15 = 800 * T + 650 * R)
    (h2 : T + R = 15) : T = 5 := 
by
  sorry

end technicians_in_workshop_l128_128356


namespace new_solution_percentage_l128_128013

theorem new_solution_percentage 
  (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution_weight : ℝ) 
  (percentage_X : ℝ) (percentage_water : ℝ)
  (total_initial_X : ℝ := initial_weight * percentage_X)
  (initial_water : ℝ := initial_weight * percentage_water)
  (post_evaporation_weight : ℝ := initial_weight - evaporated_water)
  (post_evaporation_X : ℝ := total_initial_X)
  (post_evaporation_water : ℝ := post_evaporation_weight - total_initial_X)
  (added_X : ℝ := added_solution_weight * percentage_X)
  (added_water : ℝ := added_solution_weight * percentage_water)
  (total_X : ℝ := post_evaporation_X + added_X)
  (total_water : ℝ := post_evaporation_water + added_water)
  (new_total_weight : ℝ := post_evaporation_weight + added_solution_weight) :
  (total_X / new_total_weight) * 100 = 41.25 := 
by {
  sorry
}

end new_solution_percentage_l128_128013


namespace tomorrowIsUncertain_l128_128091

-- Definitions as conditions
def isCertainEvent (e : Prop) : Prop := e = true
def isImpossibleEvent (e : Prop) : Prop := e = false
def isInevitableEvent (e : Prop) : Prop := e = true
def isUncertainEvent (e : Prop) : Prop := e ≠ true ∧ e ≠ false

-- Event: Tomorrow will be sunny
def tomorrowWillBeSunny : Prop := sorry -- Placeholder for the actual weather prediction model

-- Problem statement: Prove that "Tomorrow will be sunny" is an uncertain event
theorem tomorrowIsUncertain : isUncertainEvent tomorrowWillBeSunny := sorry

end tomorrowIsUncertain_l128_128091


namespace polynomial_div_6_l128_128993

theorem polynomial_div_6 (n : ℕ) : 6 ∣ (2 * n ^ 3 + 9 * n ^ 2 + 13 * n) := 
sorry

end polynomial_div_6_l128_128993


namespace dot_product_eq_half_l128_128348

noncomputable def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2
  
theorem dot_product_eq_half :
  vector_dot_product (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
                     (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end dot_product_eq_half_l128_128348


namespace koala_fiber_intake_l128_128549

theorem koala_fiber_intake (absorption_percentage : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) :
  absorption_percentage = 0.30 → absorbed_fiber = 12 → absorbed_fiber = absorption_percentage * total_fiber → total_fiber = 40 :=
by
  intros h1 h2 h3
  sorry

end koala_fiber_intake_l128_128549


namespace find_ordered_pair_l128_128621

theorem find_ordered_pair (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (hroots : ∀ x, x^2 + c * x + d = (x - c) * (x - d)) : 
  (c, d) = (1, -2) :=
sorry

end find_ordered_pair_l128_128621


namespace find_complex_number_l128_128720

-- Define the complex number z and the condition
variable (z : ℂ)
variable (h : (conj z) / (1 + I) = 1 - 2 * I)

-- State the theorem
theorem find_complex_number (hz : h) : z = 3 + I := 
sorry

end find_complex_number_l128_128720


namespace least_integer_value_l128_128500

theorem least_integer_value (x : ℝ) (h : |3 * x - 4| ≤ 25) : x = -7 :=
sorry

end least_integer_value_l128_128500


namespace unique_zero_function_l128_128960

theorem unique_zero_function
    (f : ℝ → ℝ)
    (H : ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) :
    ∀ x : ℝ, f x = 0 := 
by 
     sorry

end unique_zero_function_l128_128960


namespace trig_identity_proof_l128_128436

theorem trig_identity_proof (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) :
  Real.sin (2 * α - π / 6) + Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end trig_identity_proof_l128_128436


namespace smallest_sum_Q_lt_7_9_l128_128086

def Q (N k : ℕ) : ℚ := (N + 1) / (N + k + 1)

theorem smallest_sum_Q_lt_7_9 : 
    ∃ N k : ℕ, (N + k) % 4 = 0 ∧ Q N k < 7 / 9 ∧ (∀ N' k' : ℕ, (N' + k') % 4 = 0 ∧ Q N' k' < 7 / 9 → N' + k' ≥ N + k) ∧ N + k = 4 :=
by
  sorry

end smallest_sum_Q_lt_7_9_l128_128086


namespace grid_rows_l128_128669

theorem grid_rows (R : ℕ) :
  let squares_per_row := 15
  let red_squares := 4 * 6
  let blue_squares := 4 * squares_per_row
  let green_squares := 66
  let total_squares := red_squares + blue_squares + green_squares 
  total_squares = squares_per_row * R →
  R = 10 :=
by
  intros
  sorry

end grid_rows_l128_128669


namespace intersection_sets_l128_128005

-- defining sets A and B
def A : Set ℤ := {-1, 2, 4}
def B : Set ℤ := {0, 2, 6}

-- the theorem to be proved
theorem intersection_sets:
  A ∩ B = {2} :=
sorry

end intersection_sets_l128_128005


namespace equation_of_perpendicular_line_l128_128405

theorem equation_of_perpendicular_line (c : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0 ∧ 2 * x + y - 5 = 0) → (x - 2 * y - 3 = 0) := 
by
  sorry

end equation_of_perpendicular_line_l128_128405


namespace count_perfect_cubes_l128_128202

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end count_perfect_cubes_l128_128202


namespace problem1_problem2_l128_128241

variables {a x y : ℝ}

theorem problem1 (h1 : a^x = 2) (h2 : a^y = 3) : a^(x + y) = 6 :=
sorry

theorem problem2 (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x - 3 * y) = 4 / 27 :=
sorry

end problem1_problem2_l128_128241


namespace divisor_of_2n_when_remainder_is_two_l128_128023

theorem divisor_of_2n_when_remainder_is_two (n : ℤ) (k : ℤ) : 
  (n = 22 * k + 12) → ∃ d : ℤ, d = 22 ∧ (2 * n) % d = 2 :=
by
  sorry

end divisor_of_2n_when_remainder_is_two_l128_128023


namespace range_of_m_l128_128847

theorem range_of_m {x m : ℝ} 
  (h1 : 1 / 3 < x) 
  (h2 : x < 1 / 2) 
  (h3 : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  sorry

end range_of_m_l128_128847


namespace value_calculation_l128_128617

-- Define the given number
def given_number : ℝ := 93.75

-- Define the percentages as ratios
def forty_percent : ℝ := 0.4
def sixteen_percent : ℝ := 0.16

-- Calculate the intermediate value for 40% of the given number
def intermediate_value := forty_percent * given_number

-- Final value calculation for 16% of the intermediate value
def final_value := sixteen_percent * intermediate_value

-- The theorem to prove
theorem value_calculation : final_value = 6 := by
  -- Expanding definitions to substitute and simplify
  unfold final_value intermediate_value forty_percent sixteen_percent given_number
  -- Proving the correctness by calculating
  sorry

end value_calculation_l128_128617


namespace area_of_ABCM_l128_128448

-- Definitions of the problem conditions
def length_of_sides (P : ℕ) := 4
def forms_right_angle (P : ℕ) := True
def M_intersection (AG CH : ℝ) := True

-- Proposition that quadrilateral ABCM has the correct area
theorem area_of_ABCM (a b c m : ℝ) :
  (length_of_sides 12 = 4) ∧
  (forms_right_angle 12) ∧
  (M_intersection a b) →
  ∃ area_ABCM : ℝ, area_ABCM = 88/5 :=
by
  sorry

end area_of_ABCM_l128_128448


namespace bishop_safe_squares_l128_128613

def chessboard_size : ℕ := 64
def total_squares_removed_king : ℕ := chessboard_size - 1
def threat_squares : ℕ := 7

theorem bishop_safe_squares : total_squares_removed_king - threat_squares = 30 :=
by
  sorry

end bishop_safe_squares_l128_128613


namespace Camp_Cedar_number_of_counselors_l128_128107

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end Camp_Cedar_number_of_counselors_l128_128107


namespace cos_of_angle_B_l128_128793

theorem cos_of_angle_B (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : 6 * Real.sin A = 4 * Real.sin B) (h3 : 4 * Real.sin B = 3 * Real.sin C) : 
  Real.cos B = Real.sqrt 7 / 4 :=
by
  sorry

end cos_of_angle_B_l128_128793


namespace total_daisies_l128_128228

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l128_128228


namespace ribbons_problem_l128_128986

/-
    In a large box of ribbons, 1/3 are yellow, 1/4 are purple, 1/6 are orange, and the remaining 40 ribbons are black.
    Prove that the total number of orange ribbons is 27.
-/

theorem ribbons_problem :
  ∀ (total : ℕ), 
    (1 / 3 : ℚ) * total + (1 / 4 : ℚ) * total + (1 / 6 : ℚ) * total + 40 = total →
    (1 / 6 : ℚ) * total = 27 := sorry

end ribbons_problem_l128_128986


namespace sequence_S_n_a_n_l128_128885

noncomputable def sequence_S (n : ℕ) : ℝ := -1 / (n : ℝ)

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then -1 else 1 / ((n : ℝ) * (n - 1))

theorem sequence_S_n_a_n (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = -1 →
  (∀ n, (a (n + 1)) / (S (n + 1)) = S n) →
  S n = sequence_S n ∧ a n = sequence_a n :=
by
  intros h1 h2
  sorry

end sequence_S_n_a_n_l128_128885


namespace quadratic_root_is_zero_then_m_neg_one_l128_128817

theorem quadratic_root_is_zero_then_m_neg_one (m : ℝ) (h_eq : (m-1) * 0^2 + 2 * 0 + m^2 - 1 = 0) : m = -1 := by
  sorry

end quadratic_root_is_zero_then_m_neg_one_l128_128817


namespace number_of_herrings_l128_128494

theorem number_of_herrings (total_fishes pikes sturgeons herrings : ℕ)
  (h1 : total_fishes = 145)
  (h2 : pikes = 30)
  (h3 : sturgeons = 40)
  (h4 : total_fishes = pikes + sturgeons + herrings) :
  herrings = 75 :=
by
  sorry

end number_of_herrings_l128_128494


namespace parallelogram_area_correct_l128_128581

def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area_correct :
  parallelogram_area 15 5 = 75 :=
by
  sorry

end parallelogram_area_correct_l128_128581


namespace repair_time_calculation_l128_128420

-- Assume amount of work is represented as units
def work_10_people_45_minutes := 10 * 45
def work_20_people_20_minutes := 20 * 20

-- Assuming the flood destroys 2 units per minute as calculated in the solution
def flood_rate := 2

-- Calculate total initial units of the dike
def dike_initial_units :=
  work_10_people_45_minutes - flood_rate * 45

-- Given 14 people are repairing the dam
def repair_rate_14_people := 14 - flood_rate

-- Statement to prove that 14 people need 30 minutes to repair the dam
theorem repair_time_calculation :
  dike_initial_units / repair_rate_14_people = 30 :=
by
  sorry

end repair_time_calculation_l128_128420


namespace probability_all_letters_SUPERBLOOM_l128_128584

noncomputable def choose (n k : ℕ) : ℕ := sorry

theorem probability_all_letters_SUPERBLOOM :
  let P1 := 1 / (choose 6 3)
  let P2 := 9 / (choose 8 5)
  let P3 := 1 / (choose 5 4)
  P1 * P2 * P3 = 9 / 1120 :=
by
  sorry

end probability_all_letters_SUPERBLOOM_l128_128584


namespace find_first_term_l128_128244

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_first_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 2 = 1)
  (h_a4_a10 : a 3 + a 9 = 18) :
  a 0 = -3 :=
by
  sorry

end find_first_term_l128_128244


namespace teacher_allocation_l128_128732

theorem teacher_allocation :
  ∃ n : ℕ, n = 150 ∧ 
  (∀ t1 t2 t3 t4 t5 : Prop, -- represent the five teachers
    ∃ s1 s2 s3 : Prop, -- represent the three schools
      s1 ∧ s2 ∧ s3 ∧ -- each school receives at least one teacher
        ((t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧ -- allocation condition
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5))) := sorry

end teacher_allocation_l128_128732


namespace range_of_3a_minus_b_l128_128430

theorem range_of_3a_minus_b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3)
                             (h3 : 2 < a - b) (h4 : a - b < 4) :
    ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 11 ∧ x = 3 * a - b :=
sorry

end range_of_3a_minus_b_l128_128430


namespace ratio_malt_to_coke_l128_128137

-- Definitions from conditions
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_choose_malt : ℕ := 6
def females_choose_malt : ℕ := 8

-- Derived values
def total_cheerleaders : ℕ := total_males + total_females
def total_malt : ℕ := males_choose_malt + females_choose_malt
def total_coke : ℕ := total_cheerleaders - total_malt

-- The theorem to be proved
theorem ratio_malt_to_coke : (total_malt / total_coke) = (7 / 6) :=
  by
    -- skipped proof
    sorry

end ratio_malt_to_coke_l128_128137


namespace tank_overflows_after_24_minutes_l128_128187

theorem tank_overflows_after_24_minutes 
  (rateA : ℝ) (rateB : ℝ) (t : ℝ) 
  (hA : rateA = 1) 
  (hB : rateB = 4) :
  t - 1/4 * rateB + t * rateA = 1 → t = 2/5 :=
by 
  intros h
  -- the proof steps go here
  sorry

end tank_overflows_after_24_minutes_l128_128187


namespace slope_of_line_through_origin_and_A_l128_128908

theorem slope_of_line_through_origin_and_A :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 0) → (y1 = 0) → (x2 = -2) → (y2 = -2) →
  (y2 - y1) / (x2 - x1) = 1 :=
by intros; sorry

end slope_of_line_through_origin_and_A_l128_128908


namespace debby_ate_candy_l128_128699

theorem debby_ate_candy (initial_candy : ℕ) (remaining_candy : ℕ) (debby_initial : initial_candy = 12) (debby_remaining : remaining_candy = 3) : initial_candy - remaining_candy = 9 :=
by
  sorry

end debby_ate_candy_l128_128699


namespace composite_number_property_l128_128966

theorem composite_number_property (n : ℕ) 
  (h1 : n > 1) 
  (h2 : ¬ Prime n) 
  (h3 : ∀ (d : ℕ), d ∣ n → 1 ≤ d → d < n → n - 20 ≤ d ∧ d ≤ n - 12) :
  n = 21 ∨ n = 25 :=
by
  sorry

end composite_number_property_l128_128966


namespace sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l128_128145

open Real

-- Problem (a)
theorem sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1 (n k : Nat) :
  (sqrt 2 - 1)^n = sqrt k - sqrt (k - 1) :=
sorry

-- Problem (b)
theorem sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1 (m n k : Nat) :
  (sqrt m - sqrt (m - 1))^n = sqrt k - sqrt (k - 1) :=
sorry

end sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l128_128145


namespace intersection_of_M_and_N_l128_128698

def set_M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}
def set_N : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_M_and_N :
  set_M ∩ set_N = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
by
  sorry

end intersection_of_M_and_N_l128_128698


namespace frank_reads_pages_per_day_l128_128828

theorem frank_reads_pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) : pages_per_book / days_per_book = 83 :=
by {
  sorry
}

end frank_reads_pages_per_day_l128_128828


namespace inequality_holds_l128_128236

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonic_on_nonneg_interval (f : ℝ → ℝ) : Prop := ∀ x y, (0 ≤ x ∧ x < y ∧ y < 8) → f y ≤ f x

axiom condition1 : is_even f
axiom condition2 : is_monotonic_on_nonneg_interval f
axiom condition3 : f (-3) < f 2

-- The statement to be proven
theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) :=
by
  sorry

end inequality_holds_l128_128236


namespace double_exceeds_one_fifth_by_nine_l128_128257

theorem double_exceeds_one_fifth_by_nine (x : ℝ) (h : 2 * x = (1 / 5) * x + 9) : x^2 = 25 :=
sorry

end double_exceeds_one_fifth_by_nine_l128_128257


namespace carolyn_shared_with_diana_l128_128938

theorem carolyn_shared_with_diana (initial final shared : ℕ) 
    (h_initial : initial = 47) 
    (h_final : final = 5)
    (h_shared : shared = initial - final) : shared = 42 := by
  rw [h_initial, h_final] at h_shared
  exact h_shared

end carolyn_shared_with_diana_l128_128938


namespace markese_earnings_l128_128301

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l128_128301


namespace laura_owes_amount_l128_128475

-- Define the given conditions as variables
def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the interest calculation
def interest : ℝ := principal * rate * time

-- Define the final amount owed calculation
def amount_owed : ℝ := principal + interest

-- State the theorem we want to prove
theorem laura_owes_amount
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (interest : ℝ := principal * rate * time)
  (amount_owed : ℝ := principal + interest) :
  amount_owed = 36.75 := 
by 
  -- proof would go here
  sorry

end laura_owes_amount_l128_128475


namespace unique_functional_equation_solution_l128_128648

theorem unique_functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end unique_functional_equation_solution_l128_128648


namespace marked_price_percentage_l128_128401

theorem marked_price_percentage (L C M S : ℝ) 
  (h1 : C = 0.7 * L) 
  (h2 : C = 0.7 * S) 
  (h3 : S = 0.9 * M) 
  (h4 : S = L) 
  : M = (10 / 9) * L := 
by
  sorry

end marked_price_percentage_l128_128401


namespace solve_equation_1_solve_equation_2_l128_128000

open Real

theorem solve_equation_1 (x : ℝ) (h_ne1 : x + 1 ≠ 0) (h_ne2 : x - 3 ≠ 0) : 
  (5 / (x + 1) = 1 / (x - 3)) → x = 4 :=
by
    intro h
    sorry

theorem solve_equation_2 (x : ℝ) (h_ne1 : x - 4 ≠ 0) (h_ne2 : 4 - x ≠ 0) :
    (3 - x) / (x - 4) = 1 / (4 - x) - 2 → False :=
by
    intro h
    sorry

end solve_equation_1_solve_equation_2_l128_128000


namespace integer_pairs_solution_l128_128647

theorem integer_pairs_solution (a b : ℤ) : 
  (a - b - 1 ∣ a^2 + b^2 ∧ (a^2 + b^2) * 19 = (2 * a * b - 1) * 20) ↔
  (a, b) = (22, 16) ∨ (a, b) = (-16, -22) ∨ (a, b) = (8, 6) ∨ (a, b) = (-6, -8) :=
by 
  sorry

end integer_pairs_solution_l128_128647


namespace SetC_not_right_angled_triangle_l128_128871

theorem SetC_not_right_angled_triangle :
  ¬ (7^2 + 24^2 = 26^2) :=
by 
  have h : 7^2 + 24^2 ≠ 26^2 := by decide
  exact h

end SetC_not_right_angled_triangle_l128_128871


namespace variance_of_ξ_l128_128203

noncomputable def probability_distribution (ξ : ℕ) : ℚ :=
  if ξ = 2 ∨ ξ = 4 ∨ ξ = 6 ∨ ξ = 8 ∨ ξ = 10 then 1/5 else 0

def expected_value (ξ_values : List ℕ) (prob : ℕ → ℚ) : ℚ :=
  ξ_values.map (λ ξ => ξ * prob ξ) |>.sum

def variance (ξ_values : List ℕ) (prob : ℕ → ℚ) (Eξ : ℚ) : ℚ :=
  ξ_values.map (λ ξ => prob ξ * (ξ - Eξ) ^ 2) |>.sum

theorem variance_of_ξ :
  let ξ_values := [2, 4, 6, 8, 10]
  let prob := probability_distribution
  let Eξ := expected_value ξ_values prob
  variance ξ_values prob Eξ = 8 :=
by
  -- Proof goes here
  sorry

end variance_of_ξ_l128_128203


namespace triangle_cot_tan_identity_l128_128526

theorem triangle_cot_tan_identity 
  (a b c : ℝ) 
  (h : a^2 + b^2 = 2018 * c^2)
  (A B C : ℝ) 
  (triangle_ABC : ∀ (a b c : ℝ), a + b + c = π) 
  (cot_A : ℝ := Real.cos A / Real.sin A) 
  (cot_B : ℝ := Real.cos B / Real.sin B) 
  (tan_C : ℝ := Real.sin C / Real.cos C) :
  (cot_A + cot_B) * tan_C = -2 / 2017 :=
by sorry

end triangle_cot_tan_identity_l128_128526


namespace find_ab_minus_a_neg_b_l128_128138

variable (a b : ℝ)
variables (h₀ : a > 1) (h₁ : b > 0) (h₂ : a^b + a^(-b) = 2 * Real.sqrt 2)

theorem find_ab_minus_a_neg_b : a^b - a^(-b) = 2 := by
  sorry

end find_ab_minus_a_neg_b_l128_128138


namespace part1_part2_l128_128193

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l128_128193


namespace exists_n_for_sin_l128_128120

theorem exists_n_for_sin (x : ℝ) (h : Real.sin x ≠ 0) :
  ∃ n : ℕ, |Real.sin (n * x)| ≥ Real.sqrt 3 / 2 :=
sorry

end exists_n_for_sin_l128_128120


namespace maximize_revenue_l128_128750

-- Define the revenue function
def revenue (p : ℝ) : ℝ :=
  p * (150 - 4 * p)

-- Define the price constraints
def price_constraint (p : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 30

-- The theorem statement to prove that p = 19 maximizes the revenue
theorem maximize_revenue : ∀ p: ℕ, price_constraint p → revenue p ≤ revenue 19 :=
by
  sorry

end maximize_revenue_l128_128750


namespace cucumbers_after_purchase_l128_128043

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase_l128_128043


namespace min_value_of_y_l128_128326

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end min_value_of_y_l128_128326


namespace figure_perimeter_l128_128232

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter_l128_128232


namespace initial_amount_of_liquid_A_l128_128810

-- Definitions for liquids A and B and their ratios in the initial and modified mixtures
def initial_ratio_A_over_B : ℚ := 4 / 1
def final_ratio_A_over_B_after_replacement : ℚ := 2 / 3
def mixture_replacement_volume : ℚ := 30

-- Proof of the initial amount of liquid A
theorem initial_amount_of_liquid_A (x : ℚ) (A B : ℚ) (initial_mixture : ℚ) :
  (initial_ratio_A_over_B = 4 / 1) →
  (final_ratio_A_over_B_after_replacement = 2 / 3) →
  (mixture_replacement_volume = 30) →
  (A + B = 5 * x) →
  (A / B = 4 / 1) →
  ((A - 24) / (B - 6 + 30) = 2 / 3) →
  A = 48 :=
by {
  sorry
}

end initial_amount_of_liquid_A_l128_128810


namespace convert_cost_to_usd_l128_128455

def sandwich_cost_gbp : Float := 15.0
def conversion_rate : Float := 1.3

theorem convert_cost_to_usd :
  (Float.round ((sandwich_cost_gbp * conversion_rate) * 100) / 100) = 19.50 :=
by
  sorry

end convert_cost_to_usd_l128_128455


namespace min_value_inv_sum_l128_128019

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l128_128019


namespace cars_needed_to_double_earnings_l128_128194

-- Define the conditions
def baseSalary : Int := 1000
def commissionPerCar : Int := 200
def januaryEarnings : Int := 1800

-- The proof goal
theorem cars_needed_to_double_earnings : 
  ∃ (carsSoldInFeb : Int), 
    1000 + commissionPerCar * carsSoldInFeb = 2 * januaryEarnings :=
by
  sorry

end cars_needed_to_double_earnings_l128_128194


namespace inequality_system_solution_l128_128838

theorem inequality_system_solution (a b x : ℝ) 
  (h1 : x - a > 2)
  (h2 : x + 1 < b)
  (h3 : -1 < x)
  (h4 : x < 1) :
  (a + b) ^ 2023 = -1 :=
by 
  sorry

end inequality_system_solution_l128_128838


namespace temperature_range_for_5_percent_deviation_l128_128206

noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5 : ℝ) * C + 32
noncomputable def deviation (C : ℝ) : ℝ := approx_formula C - exact_formula C
noncomputable def percentage_deviation (C : ℝ) : ℝ := abs (deviation C / exact_formula C)

theorem temperature_range_for_5_percent_deviation :
  ∀ (C : ℝ), 1 + 11 / 29 ≤ C ∧ C ≤ 32 + 8 / 11 ↔ percentage_deviation C ≤ 0.05 := sorry

end temperature_range_for_5_percent_deviation_l128_128206


namespace unit_circle_chords_l128_128590

theorem unit_circle_chords (
    s t u v : ℝ
) (hs : s = 1) (ht : t = 1) (hu : u = 2) (hv : v = 3) :
    (v - u = 1) ∧ (v * u = 6) ∧ (v^2 - u^2 = 5) :=
by
  have h1 : v - u = 1 := by rw [hv, hu]; norm_num
  have h2 : v * u = 6 := by rw [hv, hu]; norm_num
  have h3 : v^2 - u^2 = 5 := by rw [hv, hu]; norm_num
  exact ⟨h1, h2, h3⟩

end unit_circle_chords_l128_128590


namespace mary_chopped_tables_l128_128655

-- Define the constants based on the conditions
def chairs_sticks := 6
def tables_sticks := 9
def stools_sticks := 2
def burn_rate := 5

-- Define the quantities of items Mary chopped up
def chopped_chairs := 18
def chopped_stools := 4
def warm_hours := 34
def sticks_from_chairs := chopped_chairs * chairs_sticks
def sticks_from_stools := chopped_stools * stools_sticks
def total_needed_sticks := warm_hours * burn_rate
def sticks_from_tables (chopped_tables : ℕ) := chopped_tables * tables_sticks

-- Define the proof goal
theorem mary_chopped_tables : ∃ chopped_tables, sticks_from_chairs + sticks_from_stools + sticks_from_tables chopped_tables = total_needed_sticks ∧ chopped_tables = 6 :=
by
  sorry

end mary_chopped_tables_l128_128655


namespace keiko_speed_l128_128831

theorem keiko_speed (a b s : ℝ) 
  (width : ℝ := 8) 
  (radius_inner := b) 
  (radius_outer := b + width)
  (time_difference := 48) 
  (L_inner := 2 * a + 2 * Real.pi * radius_inner)
  (L_outer := 2 * a + 2 * Real.pi * radius_outer) :
  (L_outer / s = L_inner / s + time_difference) → 
  s = Real.pi / 3 :=
by 
  sorry

end keiko_speed_l128_128831


namespace katherine_has_5_bananas_l128_128834

theorem katherine_has_5_bananas
  (apples : ℕ) (pears : ℕ) (bananas : ℕ) (total_fruits : ℕ)
  (h1 : apples = 4)
  (h2 : pears = 3 * apples)
  (h3 : total_fruits = apples + pears + bananas)
  (h4 : total_fruits = 21) :
  bananas = 5 :=
by
  sorry

end katherine_has_5_bananas_l128_128834


namespace smallest_integer_switch_add_l128_128152

theorem smallest_integer_switch_add (a b: ℕ) (h1: n = 10 * a + b) 
  (h2: 3 * n = 10 * b + a + 5)
  (h3: 0 ≤ b) (h4: b < 10) (h5: 1 ≤ a) (h6: a < 10): n = 47 :=
by
  sorry

end smallest_integer_switch_add_l128_128152


namespace quadratic_identity_l128_128465

variables {R : Type*} [CommRing R] [IsDomain R]

-- Define the quadratic polynomial P
def P (a b c x : R) : R := a * x^2 + b * x + c

-- Conditions as definitions in Lean
variables (a b c : R) (h₁ : P a b c a = 2021 * b * c)
                (h₂ : P a b c b = 2021 * c * a)
                (h₃ : P a b c c = 2021 * a * b)
                (dist : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c))

-- The main theorem statement
theorem quadratic_identity : a + 2021 * b + c = 0 :=
sorry

end quadratic_identity_l128_128465


namespace bernardo_wins_at_5_l128_128822

theorem bernardo_wins_at_5 :
  ∃ N : ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 27 * N + 360 < 500 ∧ ∀ M : ℕ, (0 ≤ M ∧ M ≤ 499 ∧ 27 * M + 360 < 500 → N ≤ M) :=
by
  sorry

end bernardo_wins_at_5_l128_128822


namespace sin_alpha_value_l128_128814

theorem sin_alpha_value (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_value_l128_128814


namespace find_a_l128_128857
-- Import the entire Mathlib to ensure all necessary primitives and theorems are available.

-- Define a constant equation representing the conditions.
def equation (x a : ℝ) := 3 * x + 2 * a

-- Define a theorem to prove the condition => result structure.
theorem find_a (h : equation 2 a = 0) : a = -3 :=
by sorry

end find_a_l128_128857


namespace bus_stops_per_hour_l128_128762

theorem bus_stops_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h₁ : speed_without_stoppages = 50)
  (h₂ : speed_with_stoppages = 40) :
  ∃ (minutes_stopped : ℝ), minutes_stopped = 12 :=
by
  sorry

end bus_stops_per_hour_l128_128762


namespace math_problem_l128_128701

noncomputable def answer := 21

theorem math_problem 
  (a b c d x : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) : 
  2 * x^2 - (a * b - c - d) + |a * b + 3| = answer := 
sorry

end math_problem_l128_128701


namespace green_space_equation_l128_128184

theorem green_space_equation (x : ℝ) (h_area : x * (x - 30) = 1000) :
  x * (x - 30) = 1000 := 
by
  exact h_area

end green_space_equation_l128_128184


namespace fraction_of_area_above_line_l128_128365

theorem fraction_of_area_above_line :
  let A := (3, 2)
  let B := (6, 0)
  let side_length := B.fst - A.fst
  let square_area := side_length ^ 2
  let triangle_base := B.fst - A.fst
  let triangle_height := A.snd
  let triangle_area := (1 / 2 : ℚ) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  let fraction_above_line := area_above_line / square_area
  fraction_above_line = (2 / 3 : ℚ) :=
by
  sorry

end fraction_of_area_above_line_l128_128365


namespace cookies_per_sheet_is_16_l128_128515

-- Define the number of members
def members : ℕ := 100

-- Define the number of sheets each member bakes
def sheets_per_member : ℕ := 10

-- Define the total number of cookies baked
def total_cookies : ℕ := 16000

-- Calculate the total number of sheets baked
def total_sheets : ℕ := members * sheets_per_member

-- Define the number of cookies per sheet as a result of given conditions
def cookies_per_sheet : ℕ := total_cookies / total_sheets

-- Prove that the number of cookies on each sheet is 16 given the conditions
theorem cookies_per_sheet_is_16 : cookies_per_sheet = 16 :=
by
  -- Assuming all the given definitions and conditions
  sorry

end cookies_per_sheet_is_16_l128_128515


namespace games_given_to_neil_is_five_l128_128006

variable (x : ℕ)

def initial_games_henry : ℕ := 33
def initial_games_neil : ℕ := 2
def games_given_to_neil : ℕ := x

theorem games_given_to_neil_is_five
  (H : initial_games_henry - games_given_to_neil = 4 * (initial_games_neil + games_given_to_neil)) :
  games_given_to_neil = 5 := by
  sorry

end games_given_to_neil_is_five_l128_128006


namespace find_AC_length_l128_128546

theorem find_AC_length (AB BC CD DA : ℕ) 
  (hAB : AB = 10) (hBC : BC = 9) (hCD : CD = 19) (hDA : DA = 5) : 
  14 < AC ∧ AC < 19 → AC = 15 := 
by
  sorry

end find_AC_length_l128_128546


namespace min_degree_g_l128_128282

theorem min_degree_g (f g h : Polynomial ℝ) (hf : f.degree = 8) (hh : h.degree = 9) (h_eq : 3 * f + 4 * g = h) : g.degree ≥ 9 :=
sorry

end min_degree_g_l128_128282


namespace find_b_of_perpendicular_lines_l128_128394

theorem find_b_of_perpendicular_lines (b : ℝ) (h : 4 * b - 8 = 0) : b = 2 := 
by 
  sorry

end find_b_of_perpendicular_lines_l128_128394


namespace darks_washing_time_l128_128364

theorem darks_washing_time (x : ℕ) :
  (72 + x + 45) + (50 + 65 + 54) = 344 → x = 58 :=
by
  sorry

end darks_washing_time_l128_128364


namespace shoes_ratio_l128_128501

theorem shoes_ratio (Scott_shoes : ℕ) (m : ℕ) (h1 : Scott_shoes = 7)
  (h2 : ∀ Anthony_shoes, Anthony_shoes = m * Scott_shoes)
  (h3 : ∀ Jim_shoes, Jim_shoes = Anthony_shoes - 2)
  (h4 : ∀ Anthony_shoes Jim_shoes, Anthony_shoes = Jim_shoes + 2) : 
  ∃ m : ℕ, (Anthony_shoes / Scott_shoes) = m := 
by 
  sorry

end shoes_ratio_l128_128501


namespace imaginary_part_of_z_l128_128325

-- Define complex numbers and necessary conditions
variable (z : ℂ)

-- The main statement
theorem imaginary_part_of_z (h : z * (1 + 2 * I) = 3 - 4 * I) : 
  (z.im = -2) :=
sorry

end imaginary_part_of_z_l128_128325


namespace sequence_properties_l128_128940

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end sequence_properties_l128_128940


namespace trigonometric_ratio_l128_128426

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end trigonometric_ratio_l128_128426


namespace converse_and_inverse_false_l128_128336

-- Define the property of being a rhombus and a parallelogram
def is_rhombus (R : Type) : Prop := sorry
def is_parallelogram (P : Type) : Prop := sorry

-- Given: If a quadrilateral is a rhombus, then it is a parallelogram
def quad_imp (Q : Type) : Prop := is_rhombus Q → is_parallelogram Q

-- Prove that the converse and inverse are false
theorem converse_and_inverse_false (Q : Type) 
  (h1 : quad_imp Q) : 
  ¬(is_parallelogram Q → is_rhombus Q) ∧ ¬(¬(is_rhombus Q) → ¬(is_parallelogram Q)) :=
by
  sorry

end converse_and_inverse_false_l128_128336


namespace total_bouncy_balls_l128_128462

-- Definitions based on the conditions of the problem
def packs_of_red := 4
def packs_of_yellow := 8
def packs_of_green := 4
def balls_per_pack := 10

-- Theorem stating the conclusion to be proven
theorem total_bouncy_balls :
  (packs_of_red + packs_of_yellow + packs_of_green) * balls_per_pack = 160 := 
by
  sorry

end total_bouncy_balls_l128_128462


namespace common_ratio_geometric_series_l128_128572

-- Define the terms of the geometric series
def term (n : ℕ) : ℚ :=
  match n with
  | 0     => 7 / 8
  | 1     => -21 / 32
  | 2     => 63 / 128
  | _     => sorry  -- Placeholder for further terms if necessary

-- Define the common ratio
def common_ratio : ℚ := -3 / 4

-- Prove that the common ratio is consistent for the given series
theorem common_ratio_geometric_series :
  ∀ (n : ℕ), term (n + 1) / term n = common_ratio :=
by
  sorry

end common_ratio_geometric_series_l128_128572


namespace staffing_battle_station_l128_128625

-- Define the qualifications
def num_assistant_engineer := 3
def num_maintenance_1 := 4
def num_maintenance_2 := 4
def num_field_technician := 5
def num_radio_specialist := 5

-- Prove the total number of ways to fill the positions
theorem staffing_battle_station : 
  num_assistant_engineer * num_maintenance_1 * num_maintenance_2 * num_field_technician * num_radio_specialist = 960 := by
  sorry

end staffing_battle_station_l128_128625


namespace solution_mod_5_l128_128025

theorem solution_mod_5 (a : ℤ) : 
  (a^3 + 3 * a + 1) % 5 = 0 ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  sorry

end solution_mod_5_l128_128025


namespace final_price_difference_l128_128886

noncomputable def OP : ℝ := 78.2 / 0.85
noncomputable def IP : ℝ := 78.2 + 0.25 * 78.2
noncomputable def DP : ℝ := 97.75 - 0.10 * 97.75
noncomputable def FP : ℝ := 87.975 + 0.0725 * 87.975

theorem final_price_difference : OP - FP = -2.3531875 := 
by sorry

end final_price_difference_l128_128886


namespace rotated_line_equation_l128_128105

-- Define the original equation of the line
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the rotated line equation we want to prove
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

-- Proof problem statement in Lean 4
theorem rotated_line_equation :
  ∀ (x y : ℝ), original_line x y → rotated_line x y :=
by
  sorry

end rotated_line_equation_l128_128105


namespace parabola_directrix_l128_128611

theorem parabola_directrix (p : ℝ) (hp : p > 0) (H : - (p / 2) = -3) : p = 6 :=
by
  sorry

end parabola_directrix_l128_128611


namespace determine_c_for_quadratic_eq_l128_128737

theorem determine_c_for_quadratic_eq (x1 x2 c : ℝ) 
  (h1 : x1 + x2 = 2)
  (h2 : x1 * x2 = c)
  (h3 : 7 * x2 - 4 * x1 = 47) : 
  c = -15 :=
sorry

end determine_c_for_quadratic_eq_l128_128737


namespace find_a_parallel_lines_l128_128567

theorem find_a_parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, x * a + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = k * (x * a + 2 * y + 2)) ↔ a = -6 := by
  sorry

end find_a_parallel_lines_l128_128567


namespace license_plate_increase_l128_128101

theorem license_plate_increase :
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  new_plates / old_plates = (900 / 17576) * 100 :=
by
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  have h : new_plates / old_plates = (900 / 17576) * 100 := sorry
  exact h

end license_plate_increase_l128_128101


namespace unique_solution_l128_128589

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  x > 0 ∧ (x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18)

theorem unique_solution :
  ∀ x : ℝ, satisfies_condition x ↔ x = 6 :=
by
  intro x
  unfold satisfies_condition
  sorry

end unique_solution_l128_128589


namespace max_points_per_player_l128_128522

theorem max_points_per_player
  (num_players : ℕ)
  (total_points : ℕ)
  (min_points_per_player : ℕ)
  (extra_points : ℕ)
  (scores_by_two_or_three : Prop)
  (fouls : Prop) :
  num_players = 12 →
  total_points = 100 →
  min_points_per_player = 8 →
  scores_by_two_or_three →
  fouls →
  extra_points = (total_points - num_players * min_points_per_player) →
  q = min_points_per_player + extra_points →
  q = 12 :=
by
  intros
  sorry

end max_points_per_player_l128_128522


namespace like_terms_expression_value_l128_128403

theorem like_terms_expression_value (m n : ℤ) (h1 : m = 3) (h2 : n = 1) :
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 33 := by
  sorry

end like_terms_expression_value_l128_128403


namespace area_of_CEF_l128_128903

-- Definitions of points and triangles based on given ratios
def is_right_triangle (A B C : Type) : Prop := sorry -- Placeholder for right triangle condition

def divides_ratio (A B : Type) (ratio : ℚ) : Prop := sorry -- Placeholder for ratio division condition

def area_of_triangle (A B C : Type) : ℚ := sorry -- Function to calculate area of triangle - placeholder

theorem area_of_CEF {A B C E F : Type} 
  (h1 : is_right_triangle A B C)
  (h2 : divides_ratio A C (1/4))
  (h3 : divides_ratio A B (2/3))
  (h4 : area_of_triangle A B C = 50) : 
  area_of_triangle C E F = 25 :=
sorry

end area_of_CEF_l128_128903


namespace problem_mod_l128_128808

theorem problem_mod (a b c d : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) (h4 : d = 2014) :
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end problem_mod_l128_128808


namespace find_k_l128_128182

-- Definitions for the vectors and collinearity condition.

def vector := ℝ × ℝ

def collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Given vectors a and b.
def a (k : ℝ) : vector := (1, k)
def b : vector := (2, 2)

-- Vector addition.
def add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

-- Problem statement
theorem find_k (k : ℝ) (h : collinear (add (a k) b) (a k)) : k = 1 :=
by
  sorry

end find_k_l128_128182


namespace inscribed_circle_area_l128_128774

/-- Defining the inscribed circle problem and its area. -/
theorem inscribed_circle_area (l : ℝ) (h₁ : 90 = 90) (h₂ : true) : 
  ∃ r : ℝ, (r = (2 * (Real.sqrt 2 - 1) * l / Real.pi)) ∧ ((Real.pi * r ^ 2) = (12 - 8 * Real.sqrt 2) * l ^ 2 / Real.pi) :=
  sorry

end inscribed_circle_area_l128_128774


namespace find_x_for_sin_minus_cos_eq_sqrt2_l128_128368

theorem find_x_for_sin_minus_cos_eq_sqrt2 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
by
  sorry

end find_x_for_sin_minus_cos_eq_sqrt2_l128_128368


namespace find_m_and_n_l128_128597

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l128_128597


namespace lemonade_water_cups_l128_128179

theorem lemonade_water_cups
  (W S L : ℕ)
  (h1 : W = 5 * S)
  (h2 : S = 3 * L)
  (h3 : L = 5) :
  W = 75 :=
by {
  sorry
}

end lemonade_water_cups_l128_128179


namespace how_many_cheburashkas_erased_l128_128659

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l128_128659


namespace boys_in_class_is_120_l128_128409

-- Definitions from conditions
def num_boys_in_class (number_of_girls number_of_boys : Nat) : Prop :=
  ∃ x : Nat, number_of_girls = 5 * x ∧ number_of_boys = 6 * x ∧
             (5 * x - 20) * 3 = 2 * (6 * x)

-- The theorem proving that given the conditions, the number of boys in the class is 120.
theorem boys_in_class_is_120 (number_of_girls number_of_boys : Nat) (h : num_boys_in_class number_of_girls number_of_boys) :
  number_of_boys = 120 :=
by
  sorry

end boys_in_class_is_120_l128_128409


namespace probability_digits_different_l128_128918

noncomputable def probability_all_digits_different : ℚ :=
  have tens_digits_probability := (9 / 9) * (8 / 9) * (7 / 9)
  have ones_digits_probability := (10 / 10) * (9 / 10) * (8 / 10)
  (tens_digits_probability * ones_digits_probability)

theorem probability_digits_different :
  probability_all_digits_different = 112 / 225 :=
by 
  -- The proof would go here, but it is not required for this task.
  sorry

end probability_digits_different_l128_128918


namespace sum_of_coordinates_B_l128_128658

theorem sum_of_coordinates_B :
  ∃ (x y : ℝ), (3, 5) = ((x + 6) / 2, (y + 8) / 2) ∧ x + y = 2 := by
  sorry

end sum_of_coordinates_B_l128_128658


namespace value_of_fourth_set_l128_128072

def value_in_set (a b c d : ℕ) : ℕ :=
  (a * b * c * d) - (a + b + c + d)

theorem value_of_fourth_set :
  value_in_set 1 5 6 7 = 191 :=
by
  sorry

end value_of_fourth_set_l128_128072


namespace find_x_in_equation_l128_128912

theorem find_x_in_equation :
  ∃ x : ℝ, x / 18 * (x / 162) = 1 ∧ x = 54 :=
by
  sorry

end find_x_in_equation_l128_128912


namespace wallpaper_three_layers_l128_128820

theorem wallpaper_three_layers
  (A B C : ℝ)
  (hA : A = 300)
  (hB : B = 30)
  (wall_area : ℝ)
  (h_wall_area : wall_area = 180)
  (hC : C = A - (wall_area - B) - B)
  : C = 120 := by
  sorry

end wallpaper_three_layers_l128_128820


namespace probability_is_12_over_2907_l128_128612

noncomputable def probability_drawing_red_red_green : ℚ :=
  (3 / 19) * (2 / 18) * (4 / 17)

theorem probability_is_12_over_2907 :
  probability_drawing_red_red_green = 12 / 2907 :=
sorry

end probability_is_12_over_2907_l128_128612


namespace remainder_correct_l128_128975

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 8 - 2 * x ^ 5 + 5 * x ^ 3 - 9
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) : ℝ := 29 * x - 32

theorem remainder_correct (x : ℝ) :
  ∃ q : ℝ → ℝ, p x = d x * q x + r x :=
sorry

end remainder_correct_l128_128975


namespace find_B_l128_128226

theorem find_B (A C B : ℕ) (hA : A = 520) (hC : C = A + 204) (hCB : C = B + 179) : B = 545 :=
by
  sorry

end find_B_l128_128226


namespace integer_solution_for_system_l128_128051

theorem integer_solution_for_system 
    (x y z : ℕ) 
    (h1 : 3 * x - 4 * y + 5 * z = 10) 
    (h2 : 7 * y + 8 * x - 3 * z = 13) : 
    x = 1 ∧ y = 2 ∧ z = 3 :=
by 
  sorry

end integer_solution_for_system_l128_128051


namespace possible_values_of_a_l128_128422

theorem possible_values_of_a (a : ℚ) : 
  (a^2 = 9 * 16) ∨ (16 * a = 81) ∨ (9 * a = 256) → 
  a = 12 ∨ a = -12 ∨ a = 81 / 16 ∨ a = 256 / 9 :=
by
  intros h
  sorry

end possible_values_of_a_l128_128422


namespace clowns_to_guppies_ratio_l128_128003

theorem clowns_to_guppies_ratio
  (C : ℕ)
  (tetra : ℕ)
  (guppies : ℕ)
  (total_animals : ℕ)
  (h1 : tetra = 4 * C)
  (h2 : guppies = 30)
  (h3 : total_animals = 330)
  (h4 : total_animals = tetra + C + guppies) :
  C / guppies = 2 :=
by
  sorry

end clowns_to_guppies_ratio_l128_128003


namespace domain_of_f_l128_128999

open Set

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_f_l128_128999


namespace k_value_range_l128_128242

noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

theorem k_value_range {k : ℝ} (h : ∀ x : ℝ, 0 < x → f x ≥ k * x - 2) : 
  k ≤ 1 - 1 / Real.exp 2 := 
sorry

end k_value_range_l128_128242


namespace Jackie_apples_count_l128_128224

variable (Adam_apples Jackie_apples : ℕ)
variable (h1 : Adam_apples = 10)
variable (h2 : Adam_apples = Jackie_apples + 8)

theorem Jackie_apples_count : Jackie_apples = 2 := by
  sorry

end Jackie_apples_count_l128_128224


namespace iodine_solution_problem_l128_128516

theorem iodine_solution_problem (init_concentration : Option ℝ) (init_volume : ℝ)
  (final_concentration : ℝ) (added_volume : ℝ) : 
  init_concentration = none 
  → ∃ x : ℝ, init_volume + added_volume = x :=
by
  sorry

end iodine_solution_problem_l128_128516


namespace latest_leave_time_correct_l128_128276

-- Define the conditions
def flight_time := 20 -- 8:00 pm in 24-hour format
def check_in_early := 2 -- 2 hours early
def drive_time := 45 -- 45 minutes
def park_time := 15 -- 15 minutes

-- Define the target time to be at the airport
def at_airport_time := flight_time - check_in_early -- 18:00 or 6:00 pm

-- Total travel time required (minutes)
def total_travel_time := drive_time + park_time -- 60 minutes

-- Convert total travel time to hours
def travel_time_in_hours : ℕ := total_travel_time / 60

-- Define the latest time to leave the house
def latest_leave_time := at_airport_time - travel_time_in_hours

-- Theorem to state the equivalence of the latest time they can leave their house
theorem latest_leave_time_correct : latest_leave_time = 17 :=
    by
    sorry

end latest_leave_time_correct_l128_128276


namespace farmer_harvest_correct_l128_128345

-- Define the conditions
def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

-- The proof statement
theorem farmer_harvest_correct :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l128_128345


namespace circle_properties_l128_128221

theorem circle_properties (D r C A : ℝ) (h1 : D = 15)
  (h2 : r = 7.5)
  (h3 : C = 15 * Real.pi)
  (h4 : A = 56.25 * Real.pi) :
  (9 ^ 2 + 12 ^ 2 = D ^ 2) ∧ (D = 2 * r) ∧ (C = Real.pi * D) ∧ (A = Real.pi * r ^ 2) :=
by
  sorry

end circle_properties_l128_128221


namespace determine_other_number_l128_128583

theorem determine_other_number (a b : ℤ) (h₁ : 3 * a + 4 * b = 161) (h₂ : a = 17 ∨ b = 17) : 
(a = 31 ∨ b = 31) :=
by
  sorry

end determine_other_number_l128_128583


namespace AM_GM_inequality_AM_GM_equality_l128_128899

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) :=
by
  sorry

theorem AM_GM_equality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c :=
by
  sorry

end AM_GM_inequality_AM_GM_equality_l128_128899


namespace option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l128_128308

def teapot_price : ℕ := 20
def teacup_price : ℕ := 6
def discount_rate : ℝ := 0.9

def option1_cost (x : ℕ) : ℕ :=
  5 * teapot_price + (x - 5) * teacup_price

def option2_cost (x : ℕ) : ℝ :=
  discount_rate * (5 * teapot_price + x * teacup_price)

theorem option1_cost_expression (x : ℕ) (h : x > 5) : option1_cost x = 6 * x + 70 := by
  sorry

theorem option2_cost_expression (x : ℕ) (h : x > 5) : option2_cost x = 5.4 * x + 90 := by
  sorry

theorem cost_comparison_x_20 : option1_cost 20 < option2_cost 20 := by
  sorry

theorem more_cost_effective_strategy_cost_x_20 : (5 * teapot_price + 15 * teacup_price * discount_rate) = 181 := by
  sorry

end option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l128_128308


namespace find_divided_number_l128_128056

theorem find_divided_number :
  ∃ (Number : ℕ), ∃ (q r d : ℕ), q = 8 ∧ r = 3 ∧ d = 21 ∧ Number = d * q + r ∧ Number = 171 :=
by
  sorry

end find_divided_number_l128_128056


namespace twenty_five_point_zero_six_million_in_scientific_notation_l128_128415

theorem twenty_five_point_zero_six_million_in_scientific_notation :
  (25.06e6 : ℝ) = 2.506 * 10^7 :=
by
  -- The proof would go here, but we use sorry to skip the proof.
  sorry

end twenty_five_point_zero_six_million_in_scientific_notation_l128_128415


namespace smallest_positive_n_l128_128712

theorem smallest_positive_n (n : ℕ) (h : 19 * n ≡ 789 [MOD 11]) : n = 1 := 
by
  sorry

end smallest_positive_n_l128_128712


namespace problem_1_problem_2_l128_128119

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem problem_1 (m : ℝ) (h_mono : ∀ x y, m ≤ x → x ≤ y → y ≤ m + 1 → f y ≤ f x) : m ≤ 1 :=
  sorry

theorem problem_2 (a b : ℝ) (h_min : a < b) 
  (h_min_val : ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x)
  (h_max_val : ∀ x, a ≤ x ∧ x ≤ b → f x ≤ f b) 
  (h_fa_eq_a : f a = a) (h_fb_eq_b : f b = b) : a = 2 ∧ b = 3 :=
  sorry

end problem_1_problem_2_l128_128119


namespace mix_solutions_l128_128971

variables (Vx : ℚ)

def alcohol_content_x (Vx : ℚ) : ℚ := 0.10 * Vx
def alcohol_content_y : ℚ := 0.30 * 450
def final_alcohol_content (Vx : ℚ) : ℚ := 0.22 * (Vx + 450)

theorem mix_solutions (Vx : ℚ) (h : 0.10 * Vx + 0.30 * 450 = 0.22 * (Vx + 450)) :
  Vx = 300 :=
sorry

end mix_solutions_l128_128971


namespace x_y_n_sum_l128_128323

theorem x_y_n_sum (x y n : ℕ) (h1 : 10 ≤ x ∧ x ≤ 99) (h2 : 10 ≤ y ∧ y ≤ 99) (h3 : y = (x % 10) * 10 + (x / 10)) (h4 : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end x_y_n_sum_l128_128323


namespace shifted_linear_func_is_2x_l128_128660

-- Define the initial linear function
def linear_func (x : ℝ) : ℝ := 2 * x - 3

-- Define the shifted linear function
def shifted_linear_func (x : ℝ) : ℝ := linear_func x + 3

theorem shifted_linear_func_is_2x (x : ℝ) : shifted_linear_func x = 2 * x := by
  -- Proof would go here, but we use sorry to skip it
  sorry

end shifted_linear_func_is_2x_l128_128660


namespace range_of_a_l128_128379

noncomputable def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3 ∨ (-1 / 2 ≤ a ∧ a ≤ 2)) := 
  sorry

end range_of_a_l128_128379


namespace integral_evaluation_l128_128076

noncomputable def integral_result : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) - x)

theorem integral_evaluation :
  integral_result = (Real.pi - 2) / 4 :=
by
  sorry

end integral_evaluation_l128_128076


namespace trapezoid_bisector_segment_length_l128_128044

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end trapezoid_bisector_segment_length_l128_128044


namespace saving_is_zero_cents_l128_128705

-- Define the in-store and online prices
def in_store_price : ℝ := 129.99
def online_payment_per_installment : ℝ := 29.99
def shipping_and_handling : ℝ := 11.99

-- Define the online total price
def online_total_price : ℝ := 4 * online_payment_per_installment + shipping_and_handling

-- Define the saving in cents
def saving_in_cents : ℝ := (in_store_price - online_total_price) * 100

-- State the theorem to prove the number of cents saved
theorem saving_is_zero_cents : saving_in_cents = 0 := by
  sorry

end saving_is_zero_cents_l128_128705


namespace avg_class_weight_is_46_67_l128_128380

-- Define the total number of students in section A
def num_students_a : ℕ := 40

-- Define the average weight of students in section A
def avg_weight_a : ℚ := 50

-- Define the total number of students in section B
def num_students_b : ℕ := 20

-- Define the average weight of students in section B
def avg_weight_b : ℚ := 40

-- Calculate the total weight of section A
def total_weight_a : ℚ := num_students_a * avg_weight_a

-- Calculate the total weight of section B
def total_weight_b : ℚ := num_students_b * avg_weight_b

-- Calculate the total weight of the entire class
def total_weight_class : ℚ := total_weight_a + total_weight_b

-- Calculate the total number of students in the entire class
def total_students_class : ℕ := num_students_a + num_students_b

-- Calculate the average weight of the entire class
def avg_weight_class : ℚ := total_weight_class / total_students_class

-- Theorem to prove
theorem avg_class_weight_is_46_67 :
  avg_weight_class = 46.67 := sorry

end avg_class_weight_is_46_67_l128_128380


namespace arrange_students_l128_128844

theorem arrange_students (students : Fin 7 → Prop) : 
  ∃ arrangements : ℕ, arrangements = 140 :=
by
  -- Define selection of 6 out of 7
  let selection_ways := Nat.choose 7 6
  -- Define arrangement of 6 into two groups of 3 each
  let arrangement_ways := (Nat.choose 6 3) * (Nat.choose 3 3)
  -- Calculate total arrangements by multiplying the two values
  let total_arrangements := selection_ways * arrangement_ways
  use total_arrangements
  simp [selection_ways, arrangement_ways, total_arrangements]
  exact rfl

end arrange_students_l128_128844


namespace sales_neither_notebooks_nor_markers_l128_128652

theorem sales_neither_notebooks_nor_markers (percent_notebooks percent_markers percent_staplers : ℝ) 
  (h1 : percent_notebooks = 25)
  (h2 : percent_markers = 40)
  (h3 : percent_staplers = 15) : 
  percent_staplers + (100 - (percent_notebooks + percent_markers + percent_staplers)) = 35 :=
by
  sorry

end sales_neither_notebooks_nor_markers_l128_128652


namespace area_of_circle_l128_128797

-- Given condition as a Lean definition
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 + 9 * x - 12 * y - 27 = 0

-- Theorem stating the goal
theorem area_of_circle : ∀ (x y : ℝ), circle_eq x y → ∃ r : ℝ, r = 15.25 ∧ ∃ a : ℝ, a = π * r := 
sorry

end area_of_circle_l128_128797


namespace sum_of_A_and_B_l128_128849

theorem sum_of_A_and_B:
  ∃ A B : ℕ, (A = 2 + 4) ∧ (B - 3 = 1) ∧ (A < 10) ∧ (B < 10) ∧ (A + B = 10) :=
by 
  sorry

end sum_of_A_and_B_l128_128849


namespace cost_of_apples_l128_128620

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l128_128620


namespace interest_rate_increase_l128_128259

theorem interest_rate_increase (P : ℝ) (A1 A2 : ℝ) (T : ℝ) (R1 R2 : ℝ) (percentage_increase : ℝ) :
  P = 500 → A1 = 600 → A2 = 700 → T = 2 → 
  (A1 - P) = P * R1 * T →
  (A2 - P) = P * R2 * T →
  percentage_increase = (R2 - R1) / R1 * 100 →
  percentage_increase = 100 :=
by sorry

end interest_rate_increase_l128_128259


namespace committee_meeting_l128_128842

theorem committee_meeting : 
  ∃ (A B : ℕ), 2 * A + B = 7 ∧ A + 2 * B = 11 ∧ A + B = 6 :=
by 
  sorry

end committee_meeting_l128_128842


namespace first_person_job_completion_time_l128_128217

noncomputable def job_completion_time :=
  let A := 1 - (1/5)
  let C := 1/8
  let combined_rate := A + C
  have h1 : combined_rate = 0.325 := by
    sorry
  have h2 : A ≠ 0 := by
    sorry
  (1 / A : ℝ)
  
theorem first_person_job_completion_time :
  job_completion_time = 1.25 :=
by
  sorry

end first_person_job_completion_time_l128_128217


namespace calculate_a3_l128_128246

theorem calculate_a3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n = 2^n - 1) (h2 : ∀ n, a n = S n - S (n-1)) : 
  a 3 = 4 :=
by
  sorry

end calculate_a3_l128_128246


namespace number_of_girls_in_class_l128_128497

section
variables (g b : ℕ)

/-- Given the total number of students and the ratio of girls to boys, this theorem states the number of girls in Ben's class. -/
theorem number_of_girls_in_class (h1 : 3 * b = 4 * g) (h2 : g + b = 35) : g = 15 :=
sorry
end

end number_of_girls_in_class_l128_128497


namespace solve_a1_solve_a2_l128_128586

noncomputable def initial_volume := 1  -- in m^3
noncomputable def initial_pressure := 10^5  -- in Pa
noncomputable def initial_temperature := 300  -- in K

theorem solve_a1 (a1 : ℝ) : a1 = -10^5 :=
  sorry

theorem solve_a2 (a2 : ℝ) : a2 = -1.4 * 10^5 :=
  sorry

end solve_a1_solve_a2_l128_128586


namespace num_terms_arithmetic_sequence_is_15_l128_128686

theorem num_terms_arithmetic_sequence_is_15 :
  ∃ n : ℕ, (∀ (a : ℤ), a = -58 + (n - 1) * 7 → a = 44) ∧ n = 15 :=
by {
  sorry
}

end num_terms_arithmetic_sequence_is_15_l128_128686


namespace probability_four_ones_in_five_rolls_l128_128702

-- Define the probability of rolling a 1 on a fair six-sided die
def prob_one_roll_one : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a fair six-sided die
def prob_one_roll_not_one : ℚ := 5 / 6

-- Define the number of successes needed, here 4 ones in 5 rolls
def num_successes : ℕ := 4

-- Define the total number of trials, here 5 rolls
def num_trials : ℕ := 5

-- Binomial probability calculation for 4 successes in 5 trials with probability of success prob_one_roll_one
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_four_ones_in_five_rolls : binomial_prob num_trials num_successes prob_one_roll_one = 25 / 7776 := 
by
  sorry

end probability_four_ones_in_five_rolls_l128_128702


namespace no_integer_solution_l128_128498

theorem no_integer_solution (m n : ℤ) : m^2 - 11 * m * n - 8 * n^2 ≠ 88 :=
sorry

end no_integer_solution_l128_128498


namespace hexagon_inscribed_circumscribed_symmetric_l128_128988

-- Define the conditions of the problem
variables (R r c : ℝ)

-- Define the main assertion of the problem
theorem hexagon_inscribed_circumscribed_symmetric :
  3 * (R^2 - c^2)^4 - 4 * r^2 * (R^2 - c^2)^2 * (R^2 + c^2) - 16 * R^2 * c^2 * r^4 = 0 :=
by
  -- skipping proof
  sorry

end hexagon_inscribed_circumscribed_symmetric_l128_128988


namespace find_second_number_l128_128582

theorem find_second_number (a b c : ℕ) (h1 : a = 5 * x) (h2 : b = 3 * x) (h3 : c = 4 * x) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l128_128582


namespace number_added_after_division_is_5_l128_128773

noncomputable def number_thought_of : ℕ := 72
noncomputable def result_after_division (n : ℕ) : ℕ := n / 6
noncomputable def final_result (n x : ℕ) : ℕ := result_after_division n + x

theorem number_added_after_division_is_5 :
  ∃ x : ℕ, final_result number_thought_of x = 17 ∧ x = 5 :=
by
  sorry

end number_added_after_division_is_5_l128_128773


namespace geometric_shape_is_sphere_l128_128577

-- Define the spherical coordinate system conditions
def spherical_coordinates (ρ θ φ r : ℝ) : Prop :=
  ρ = r

-- The theorem we want to prove
theorem geometric_shape_is_sphere (ρ θ φ r : ℝ) (h : spherical_coordinates ρ θ φ r) : ∀ (x y z : ℝ), (x^2 + y^2 + z^2 = r^2) :=
by
  sorry

end geometric_shape_is_sphere_l128_128577


namespace train_length_300_l128_128580

/-- 
Proving the length of the train given the conditions on crossing times and length of the platform.
-/
theorem train_length_300 (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 200 = V * 30) : 
  L = 300 := 
by
  sorry

end train_length_300_l128_128580


namespace matrix_zero_product_or_rank_one_l128_128033

variables {n : ℕ}
variables (A B C : matrix (fin n) (fin n) ℝ)

theorem matrix_zero_product_or_rank_one
  (h1 : A * B * C = 0)
  (h2 : B.rank = 1) :
  A * B = 0 ∨ B * C = 0 :=
sorry

end matrix_zero_product_or_rank_one_l128_128033


namespace total_sheets_of_paper_l128_128286

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l128_128286


namespace brick_wall_problem_l128_128492

theorem brick_wall_problem
  (b : ℕ)
  (rate_ben rate_arya : ℕ → ℕ)
  (combined_rate : ℕ → ℕ → ℕ)
  (work_duration : ℕ)
  (effective_combined_rate : ℕ → ℕ × ℕ → ℕ)
  (rate_ben_def : ∀ (b : ℕ), rate_ben b = b / 12)
  (rate_arya_def : ∀ (b : ℕ), rate_arya b = b / 15)
  (combined_rate_def : ∀ (b : ℕ), combined_rate (rate_ben b) (rate_arya b) = rate_ben b + rate_arya b)
  (effective_combined_rate_def : ∀ (b : ℕ), effective_combined_rate b (rate_ben b, rate_arya b) = combined_rate (rate_ben b) (rate_arya b) - 15)
  (work_duration_def : work_duration = 6)
  (completion_condition : ∀ (b : ℕ), work_duration * effective_combined_rate b (rate_ben b, rate_arya b) = b) :
  b = 900 :=
by
  -- Proof would go here
  sorry

end brick_wall_problem_l128_128492


namespace find_m_given_root_exists_l128_128032

theorem find_m_given_root_exists (x m : ℝ) (h : ∃ x, x ≠ 2 ∧ (x / (x - 2) - 2 = m / (x - 2))) : m = 2 :=
by
  sorry

end find_m_given_root_exists_l128_128032


namespace compare_values_of_even_and_monotone_function_l128_128053

variable (f : ℝ → ℝ)

def is_even_function := ∀ x : ℝ, f x = f (-x)
def is_monotone_increasing_on_nonneg := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem compare_values_of_even_and_monotone_function
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  sorry

end compare_values_of_even_and_monotone_function_l128_128053


namespace find_x_l128_128729

theorem find_x (x : ℝ) (h : x - 2 * x + 3 * x = 100) : x = 50 := by
  sorry

end find_x_l128_128729


namespace find_square_tiles_l128_128974

variables (t s p : ℕ)

theorem find_square_tiles
  (h1 : t + s + p = 30)
  (h2 : 3 * t + 4 * s + 5 * p = 120) :
  s = 10 :=
by
  sorry

end find_square_tiles_l128_128974


namespace snowboard_price_after_discounts_l128_128739

noncomputable def final_snowboard_price (P_original : ℝ) (d_Friday : ℝ) (d_Monday : ℝ) : ℝ :=
  P_original * (1 - d_Friday) * (1 - d_Monday)

theorem snowboard_price_after_discounts :
  final_snowboard_price 100 0.50 0.30 = 35 :=
by 
  sorry

end snowboard_price_after_discounts_l128_128739


namespace minimum_road_length_l128_128524

/-- Define the grid points A, B, and C with their coordinates. -/
def A : ℤ × ℤ := (0, 0)
def B : ℤ × ℤ := (3, 2)
def C : ℤ × ℤ := (4, 3)

/-- Define the side length of each grid square in meters. -/
def side_length : ℕ := 100

/-- Calculate the Manhattan distance between two points on the grid. -/
def manhattan_distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1) + Int.natAbs (p.2 - q.2)) * side_length

/-- Statement: The minimum total length of the roads (in meters) to connect A, B, and C is 1000 meters. -/
theorem minimum_road_length : manhattan_distance A B + manhattan_distance B C + manhattan_distance C A = 1000 := by
  sorry

end minimum_road_length_l128_128524


namespace deshaun_read_books_over_summer_l128_128440

theorem deshaun_read_books_over_summer 
  (summer_days : ℕ)
  (average_pages_per_book : ℕ)
  (ratio_closest_person : ℝ)
  (pages_read_per_day_second_person : ℕ)
  (books_read : ℕ)
  (total_pages_second_person_read : ℕ)
  (h1 : summer_days = 80)
  (h2 : average_pages_per_book = 320)
  (h3 : ratio_closest_person = 0.75)
  (h4 : pages_read_per_day_second_person = 180)
  (h5 : total_pages_second_person_read = pages_read_per_day_second_person * summer_days)
  (h6 : books_read * average_pages_per_book = total_pages_second_person_read / ratio_closest_person) :
  books_read = 60 :=
by {
  sorry
}

end deshaun_read_books_over_summer_l128_128440


namespace greatest_t_value_l128_128552

theorem greatest_t_value :
  ∃ t_max : ℝ, (∀ t : ℝ, ((t ≠  8) ∧ (t ≠ -7) → (t^2 - t - 90) / (t - 8) = 6 / (t + 7) → t ≤ t_max)) ∧ t_max = -1 :=
sorry

end greatest_t_value_l128_128552


namespace gcd_poly_l128_128114

-- Defining the conditions
def is_odd_multiple_of_17 (b : ℤ) : Prop := ∃ k : ℤ, b = 17 * (2 * k + 1)

theorem gcd_poly (b : ℤ) (h : is_odd_multiple_of_17 b) : 
  Int.gcd (12 * b^3 + 7 * b^2 + 49 * b + 106) 
          (3 * b + 7) = 1 :=
by sorry

end gcd_poly_l128_128114


namespace eiffel_tower_scale_l128_128085

theorem eiffel_tower_scale (height_model : ℝ) (height_actual : ℝ) (h_model : height_model = 30) (h_actual : height_actual = 984) : 
  height_actual / height_model = 32.8 := by
  sorry

end eiffel_tower_scale_l128_128085


namespace intersection_A_CRB_l128_128267

-- Definition of sets A and C_{R}B
def is_in_A (x: ℝ) := 0 < x ∧ x < 2

def is_in_CRB (x: ℝ) := x ≤ 1 ∨ x ≥ Real.exp 2

-- Proof that the intersection of A and C_{R}B is (0, 1]
theorem intersection_A_CRB : {x : ℝ | is_in_A x} ∩ {x : ℝ | is_in_CRB x} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_CRB_l128_128267


namespace fill_entire_bucket_l128_128238

theorem fill_entire_bucket (h : (2/3 : ℝ) * t = 2) : t = 3 :=
sorry

end fill_entire_bucket_l128_128238


namespace arithmetic_sequence_n_value_l128_128098

noncomputable def common_ratio (a₁ S₃ : ℕ) : ℕ := by sorry

theorem arithmetic_sequence_n_value:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
  (∀ n, a n > 0) →
  a 1 = 3 →
  S 3 = 21 →
  (∃ q, q > 0 ∧ common_ratio 1 q = q ∧ a 5 = 48) →
  n = 5 :=
by
  intros
  sorry

end arithmetic_sequence_n_value_l128_128098


namespace find_8b_l128_128341

variable (a b : ℚ)

theorem find_8b (h1 : 4 * a + 3 * b = 5) (h2 : a = b - 3) : 8 * b = 136 / 7 := by
  sorry

end find_8b_l128_128341


namespace line_through_point_and_parallel_l128_128437

def point_A : ℝ × ℝ × ℝ := (-2, 3, 1)

def plane1 (x y z : ℝ) := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) := 2*x + 3*y - z + 1 = 0

theorem line_through_point_and_parallel (x y z t : ℝ) :
  ∃ t, 
    x = 5 * t - 2 ∧
    y = -t + 3 ∧
    z = 7 * t + 1 :=
sorry

end line_through_point_and_parallel_l128_128437


namespace gcd_pow_sub_one_l128_128457

theorem gcd_pow_sub_one (n m : ℕ) (h1 : n = 1005) (h2 : m = 1016) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2047 := by
  rw [h1, h2]
  sorry

end gcd_pow_sub_one_l128_128457


namespace sum_ratio_l128_128804

variable {α : Type _} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0       => a₁
| (n + 1) => (geometric_sequence a₁ q n) * q

noncomputable def sum_geometric (a₁ q : α) (n : ℕ) : α :=
  if q = 1 then a₁ * (n + 1)
  else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_ratio {a₁ q : α} (h : 8 * (geometric_sequence a₁ q 1) + (geometric_sequence a₁ q 4) = 0) :
  (sum_geometric a₁ q 4) / (sum_geometric a₁ q 1) = -11 :=
sorry

end sum_ratio_l128_128804


namespace plane_intersect_probability_l128_128130

-- Define the vertices of the rectangular prism
def vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (2,0,0), (2,2,0), (0,2,0), 
   (0,0,1), (2,0,1), (2,2,1), (0,2,1)]

-- Calculate total number of ways to choose 3 vertices out of 8
def total_ways : ℕ := Nat.choose 8 3

-- Calculate the number of planes that do not intersect the interior of the prism
def non_intersecting_planes : ℕ := 6 * Nat.choose 4 3

-- Calculate the probability as a fraction
def probability_of_intersecting (total non_intersecting : ℕ) : ℚ :=
  1 - (non_intersecting : ℚ) / (total : ℚ)

-- The main theorem to state the probability is 4/7
theorem plane_intersect_probability : 
  probability_of_intersecting total_ways non_intersecting_planes = 4 / 7 := 
  by
    -- Skipping the proof
    sorry

end plane_intersect_probability_l128_128130


namespace log_sum_l128_128183

open Real

theorem log_sum : log 2 + log 5 = 1 :=
sorry

end log_sum_l128_128183


namespace cost_price_computer_table_l128_128550

-- Define the variables
def cost_price : ℝ := 3840
def selling_price (CP : ℝ) := CP * 1.25

-- State the conditions and the proof problem
theorem cost_price_computer_table 
  (SP : ℝ) 
  (h1 : SP = 4800)
  (h2 : ∀ CP : ℝ, SP = selling_price CP) :
  cost_price = 3840 :=
by 
  sorry

end cost_price_computer_table_l128_128550


namespace f_2020_minus_f_2018_l128_128723

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 5) = f x
axiom f_seven : f 7 = 9

theorem f_2020_minus_f_2018 : f 2020 - f 2018 = 9 := by
  sorry

end f_2020_minus_f_2018_l128_128723


namespace geometric_sequence_sum_l128_128387

-- Define the relations for geometric sequences
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (m n p q : ℕ), m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n, a n > 0)
  (h_cond : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 :=
sorry

end geometric_sequence_sum_l128_128387


namespace merchant_problem_l128_128234

theorem merchant_problem (P C : ℝ) (h1 : P + C = 60) (h2 : 2.40 * P + 6.00 * C = 180) : C = 10 := 
by
  -- Proof goes here
  sorry

end merchant_problem_l128_128234


namespace cone_volume_l128_128132

theorem cone_volume (central_angle : ℝ) (sector_area : ℝ) (h1 : central_angle = 120) (h2 : sector_area = 3 * Real.pi) :
  ∃ V : ℝ, V = (2 * Real.sqrt 2 * Real.pi) / 3 :=
by
  -- We acknowledge the input condition where the angle is 120° and sector area is 3π
  -- The problem requires proving the volume of the cone
  sorry

end cone_volume_l128_128132


namespace find_couples_l128_128162

theorem find_couples (n p q : ℕ) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
    (h_gcd : Nat.gcd p q = 1)
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
by 
  sorry

end find_couples_l128_128162


namespace find_solutions_to_system_l128_128727

theorem find_solutions_to_system (x y z : ℝ) 
    (h1 : 3 * (x^2 + y^2 + z^2) = 1) 
    (h2 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^3) : 
    x = y ∧ y = z ∧ (x = 1 / 3 ∨ x = -1 / 3) :=
by
  sorry

end find_solutions_to_system_l128_128727


namespace domain_of_f_2x_minus_1_l128_128806

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → (f x ≠ 0)) →
  (∀ y, 0 ≤ y ∧ y ≤ 1 ↔ exists x, (2 * x - 1 = y) ∧ (0 ≤ x ∧ x ≤ 1)) :=
by
  sorry

end domain_of_f_2x_minus_1_l128_128806


namespace partition_sum_condition_l128_128541

theorem partition_sum_condition (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c := 
by
  -- sorry is here to acknowledge that no proof is required per instructions.
  sorry

end partition_sum_condition_l128_128541


namespace pentagon_area_is_correct_l128_128667

noncomputable def area_of_pentagon : ℕ :=
  let area_trapezoid := (1 / 2) * (25 + 28) * 30
  let area_triangle := (1 / 2) * 18 * 24
  area_trapezoid + area_triangle

theorem pentagon_area_is_correct (s1 s2 s3 s4 s5 : ℕ) (b1 b2 h1 b3 h2 : ℕ)
  (h₀ : s1 = 18) (h₁ : s2 = 25) (h₂ : s3 = 30) (h₃ : s4 = 28) (h₄ : s5 = 25)
  (h₅ : b1 = 25) (h₆ : b2 = 28) (h₇ : h1 = 30) (h₈ : b3 = 18) (h₉ : h2 = 24) :
  area_of_pentagon = 1011 := by
  -- placeholder for actual proof
  sorry

end pentagon_area_is_correct_l128_128667


namespace rational_expression_nonnegative_l128_128717

theorem rational_expression_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 :=
  sorry

end rational_expression_nonnegative_l128_128717


namespace find_real_pairs_l128_128218

theorem find_real_pairs (x y : ℝ) (h : 2 * x / (1 + x^2) = (1 + y^2) / (2 * y)) : 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end find_real_pairs_l128_128218


namespace sum_of_ages_l128_128756

theorem sum_of_ages (a b c d : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b = 24 ∨ a * c = 24 ∨ a * d = 24 ∨ b * c = 24 ∨ b * d = 24 ∨ c * d = 24)
  (h8 : a * b = 35 ∨ a * c = 35 ∨ a * d = 35 ∨ b * c = 35 ∨ b * d = 35 ∨ c * d = 35)
  (h9 : a < 10) (h10 : b < 10) (h11 : c < 10) (h12 : d < 10)
  (h13 : 0 < a) (h14 : 0 < b) (h15 : 0 < c) (h16 : 0 < d) :
  a + b + c + d = 23 := sorry

end sum_of_ages_l128_128756


namespace cody_books_reading_l128_128934

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l128_128934


namespace terminal_side_in_third_quadrant_l128_128910

def is_equivalent_angle (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

def in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

theorem terminal_side_in_third_quadrant : 
  ∀ θ, θ = 600 → in_third_quadrant (θ % 360) :=
by
  intro θ
  intro hθ
  sorry

end terminal_side_in_third_quadrant_l128_128910


namespace tree_planting_campaign_l128_128016

theorem tree_planting_campaign
  (P : ℝ)
  (h1 : 456 = P * (1 - 1/20))
  (h2 : P ≥ 0)
  : (P * (1 + 0.1)) = (456 / (1 - 1/20) * 1.1) :=
by
  sorry

end tree_planting_campaign_l128_128016


namespace negative_values_count_l128_128141

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l128_128141


namespace smallest_number_of_coins_l128_128319

theorem smallest_number_of_coins (p n d q : ℕ) (total : ℕ) :
  (total < 100) →
  (total = p * 1 + n * 5 + d * 10 + q * 25) →
  (∀ k < 100, ∃ (p n d q : ℕ), k = p * 1 + n * 5 + d * 10 + q * 25) →
  p + n + d + q = 10 :=
sorry

end smallest_number_of_coins_l128_128319


namespace valentine_floral_requirement_l128_128509

theorem valentine_floral_requirement:
  let nursing_home_roses := 90
  let nursing_home_tulips := 80
  let nursing_home_lilies := 100
  let shelter_roses := 120
  let shelter_tulips := 75
  let shelter_lilies := 95
  let maternity_ward_roses := 100
  let maternity_ward_tulips := 110
  let maternity_ward_lilies := 85
  let total_roses := nursing_home_roses + shelter_roses + maternity_ward_roses
  let total_tulips := nursing_home_tulips + shelter_tulips + maternity_ward_tulips
  let total_lilies := nursing_home_lilies + shelter_lilies + maternity_ward_lilies
  let total_flowers := total_roses + total_tulips + total_lilies
  total_roses = 310 ∧
  total_tulips = 265 ∧
  total_lilies = 280 ∧
  total_flowers = 855 :=
by
  sorry

end valentine_floral_requirement_l128_128509


namespace solve_for_cubic_l128_128564

theorem solve_for_cubic (x y : ℝ) (h₁ : x * (x + y) = 49) (h₂: y * (x + y) = 63) : (x + y)^3 = 448 * Real.sqrt 7 := 
sorry

end solve_for_cubic_l128_128564


namespace ball_hits_ground_l128_128898

noncomputable def ball_height (t : ℝ) : ℝ := -9 * t^2 + 15 * t + 72

theorem ball_hits_ground :
  (∃ t : ℝ, t = (5 + Real.sqrt 313) / 6 ∧ ball_height t = 0) :=
sorry

end ball_hits_ground_l128_128898


namespace common_ratio_arithmetic_progression_l128_128124

theorem common_ratio_arithmetic_progression (a3 q : ℝ) (h1 : a3 = 9) (h2 : a3 + a3 * q + 9 = 27) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end common_ratio_arithmetic_progression_l128_128124


namespace leo_current_weight_l128_128593

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end leo_current_weight_l128_128593


namespace solve_star_op_eq_l128_128367

def star_op (a b : ℕ) : ℕ :=
  if a < b then b * b else b * b * b

theorem solve_star_op_eq :
  ∃ x : ℕ, 5 * star_op 5 x = 64 ∧ (x = 4 ∨ x = 8) :=
sorry

end solve_star_op_eq_l128_128367


namespace Jennifer_more_boxes_l128_128816

-- Definitions based on conditions
def Kim_boxes : ℕ := 54
def Jennifer_boxes : ℕ := 71

-- Proof statement (no actual proof needed, just the statement)
theorem Jennifer_more_boxes : Jennifer_boxes - Kim_boxes = 17 := by
  sorry

end Jennifer_more_boxes_l128_128816


namespace cube_face_sum_l128_128253

theorem cube_face_sum (a b c d e f : ℕ) (h1 : e = b) (h2 : 2 * (a * b * c + a * b * f + d * b * c + d * b * f) = 1332) :
  a + b + c + d + e + f = 47 :=
sorry

end cube_face_sum_l128_128253


namespace A_finishes_race_in_36_seconds_l128_128302

-- Definitions of conditions
def distance_A := 130 -- A covers a distance of 130 meters
def distance_B := 130 -- B covers a distance of 130 meters
def time_B := 45 -- B covers the distance in 45 seconds
def distance_B_lag := 26 -- A beats B by 26 meters

-- Statement to prove
theorem A_finishes_race_in_36_seconds : 
  ∃ t : ℝ, distance_A / t + distance_B_lag = distance_B / time_B := sorry

end A_finishes_race_in_36_seconds_l128_128302


namespace problem_I_l128_128782

theorem problem_I (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : 
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 := 
by
  sorry

end problem_I_l128_128782


namespace cassy_initial_jars_l128_128355

theorem cassy_initial_jars (boxes1 jars1 boxes2 jars2 leftover: ℕ) (h1: boxes1 = 10) (h2: jars1 = 12) (h3: boxes2 = 30) (h4: jars2 = 10) (h5: leftover = 80) : 
  boxes1 * jars1 + boxes2 * jars2 + leftover = 500 := 
by 
  sorry

end cassy_initial_jars_l128_128355


namespace find_first_number_of_sequence_l128_128473

theorem find_first_number_of_sequence
    (a : ℕ → ℕ)
    (h1 : ∀ n, 3 ≤ n → a n = a (n-1) * a (n-2))
    (h2 : a 8 = 36)
    (h3 : a 9 = 1296)
    (h4 : a 10 = 46656) :
    a 1 = 60466176 := 
sorry

end find_first_number_of_sequence_l128_128473


namespace percent_non_union_women_l128_128361

-- Definitions used in the conditions:
def total_employees := 100
def percent_men := 50 / 100
def percent_union := 60 / 100
def percent_union_men := 70 / 100

-- Calculate intermediate values
def num_men := total_employees * percent_men
def num_union := total_employees * percent_union
def num_union_men := num_union * percent_union_men
def num_non_union := total_employees - num_union
def num_non_union_men := num_men - num_union_men
def num_non_union_women := num_non_union - num_non_union_men

-- Statement of the problem in Lean
theorem percent_non_union_women : (num_non_union_women / num_non_union) * 100 = 80 := 
by {
  sorry
}

end percent_non_union_women_l128_128361


namespace x_y_value_l128_128980

theorem x_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 30) : x + y = 2 :=
sorry

end x_y_value_l128_128980


namespace minimum_value_is_81_l128_128419

noncomputable def minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) : ℝ :=
a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_is_81 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_value a b c h1 h2 h3 h4 = 81 :=
sorry

end minimum_value_is_81_l128_128419


namespace find_natural_numbers_satisfying_prime_square_l128_128410

-- Define conditions as a Lean statement
theorem find_natural_numbers_satisfying_prime_square (n : ℕ) (h : ∃ p : ℕ, Prime p ∧ (2 * n^2 + 3 * n - 35 = p^2)) :
  n = 4 ∨ n = 12 :=
sorry

end find_natural_numbers_satisfying_prime_square_l128_128410


namespace point_in_which_quadrant_l128_128275

noncomputable def quadrant_of_point (x y : ℝ) : String :=
if (x > 0) ∧ (y > 0) then
    "First"
else if (x < 0) ∧ (y > 0) then
    "Second"
else if (x < 0) ∧ (y < 0) then
    "Third"
else if (x > 0) ∧ (y < 0) then
    "Fourth"
else
    "On Axis"

theorem point_in_which_quadrant (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) : quadrant_of_point (Real.sin α) (Real.cos α) = "Fourth" :=
by {
    sorry
}

end point_in_which_quadrant_l128_128275


namespace at_least_six_destinations_l128_128510

theorem at_least_six_destinations (destinations : ℕ) (tickets_sold : ℕ) (h_dest : destinations = 200) (h_tickets : tickets_sold = 3800) :
  ∃ k ≥ 6, ∃ t : ℕ, (∃ f : Fin destinations → ℕ, (∀ i : Fin destinations, f i ≤ t) ∧ (tickets_sold ≤ t * destinations) ∧ ((∃ i : Fin destinations, f i = k) → k ≥ 6)) :=
by
  sorry

end at_least_six_destinations_l128_128510


namespace sum_of_palindromic_primes_less_than_70_l128_128687

def is_prime (n : ℕ) : Prop := Nat.Prime n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes_less_than_70 :
  let palindromic_primes := [11, 13, 31, 37]
  (∀ p ∈ palindromic_primes, is_palindromic_prime p ∧ p < 70) →
  palindromic_primes.sum = 92 :=
by
  sorry

end sum_of_palindromic_primes_less_than_70_l128_128687


namespace distinct_ordered_pairs_count_l128_128759

theorem distinct_ordered_pairs_count :
  ∃ S : Finset (ℕ × ℕ), 
    (∀ p ∈ S, 1 ≤ p.1 ∧ 1 ≤ p.2 ∧ (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)) ∧
    S.card = 9 := 
by
  sorry

end distinct_ordered_pairs_count_l128_128759


namespace coordinates_of_point_A_l128_128461

def f (x : ℝ) : ℝ := x^2 + 3 * x

theorem coordinates_of_point_A (a : ℝ) (b : ℝ) 
    (slope_condition : deriv f a = 7) 
    (point_condition : f a = b) : 
    a = 2 ∧ b = 10 := 
by {
    sorry
}

end coordinates_of_point_A_l128_128461


namespace minimum_a_l128_128219

open Real

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + a / y) ≥ (16 / (x + y))) → a ≥ 9 := by
sorry

end minimum_a_l128_128219


namespace geometric_solid_is_tetrahedron_l128_128531

-- Definitions based on the conditions provided
def top_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def front_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def side_view_is_triangle : Prop := sorry -- Placeholder for the actual definition

-- Theorem statement to prove the geometric solid is a triangular pyramid
theorem geometric_solid_is_tetrahedron 
  (h_top : top_view_is_triangle)
  (h_front : front_view_is_triangle)
  (h_side : side_view_is_triangle) :
  -- Conclusion that the solid is a triangular pyramid (tetrahedron)
  is_tetrahedron :=
sorry

end geometric_solid_is_tetrahedron_l128_128531


namespace sqrt_log_equality_l128_128333

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem sqrt_log_equality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
    Real.sqrt (log4 x + 2 * log2 y) = Real.sqrt (log2 (x * y^2)) / Real.sqrt 2 :=
sorry

end sqrt_log_equality_l128_128333


namespace rate_of_current_l128_128596

theorem rate_of_current (c : ℝ) (h1 : ∀ d : ℝ, d / (3.9 - c) = 2 * (d / (3.9 + c))) : c = 1.3 :=
sorry

end rate_of_current_l128_128596


namespace simplify_expression_l128_128131

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l128_128131


namespace cost_of_five_dozens_l128_128057

-- Define cost per dozen given the total cost for two dozen
noncomputable def cost_per_dozen : ℝ := 15.60 / 2

-- Define the number of dozen apples we want to calculate the cost for
def number_of_dozens := 5

-- Define the total cost for the given number of dozens
noncomputable def total_cost (n : ℕ) : ℝ := n * cost_per_dozen

-- State the theorem
theorem cost_of_five_dozens : total_cost number_of_dozens = 39 :=
by
  unfold total_cost cost_per_dozen
  sorry

end cost_of_five_dozens_l128_128057


namespace books_sold_on_wednesday_l128_128456

theorem books_sold_on_wednesday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percent_unsold : ℚ) :
  initial_stock = 900 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percent_unsold = 55.333333333333336 →
  ∃ (sold_wednesday : ℕ), sold_wednesday = 64 :=
by
  sorry

end books_sold_on_wednesday_l128_128456


namespace inequality_range_l128_128972

theorem inequality_range (a : ℝ) : (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by
  sorry

end inequality_range_l128_128972


namespace remainder_of_3456_div_97_l128_128545

theorem remainder_of_3456_div_97 :
  3456 % 97 = 61 :=
by
  sorry

end remainder_of_3456_div_97_l128_128545


namespace car_drive_highway_distance_l128_128020

theorem car_drive_highway_distance
  (d_local : ℝ)
  (s_local : ℝ)
  (s_highway : ℝ)
  (s_avg : ℝ)
  (d_total := d_local + s_avg * (d_local / s_local + d_local / s_highway))
  (t_local := d_local / s_local)
  (t_highway : ℝ := (d_total - d_local) / s_highway)
  (t_total := t_local + t_highway)
  (avg_speed := (d_total) / t_total)
  : d_local = 60 → s_local = 20 → s_highway = 60 → s_avg = 36 → avg_speed = 36 → d_total - d_local = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4]
  sorry

end car_drive_highway_distance_l128_128020


namespace find_sum_of_variables_l128_128748

theorem find_sum_of_variables (x y : ℚ) (h1 : 5 * x - 3 * y = 17) (h2 : 3 * x + 5 * y = 1) : x + y = 21 / 17 := 
  sorry

end find_sum_of_variables_l128_128748


namespace kay_age_l128_128528

/-- Let K be Kay's age. If the youngest sibling is 5 less 
than half of Kay's age, the oldest sibling is four times 
as old as the youngest sibling, and the oldest sibling 
is 44 years old, then Kay is 32 years old. -/
theorem kay_age (K : ℕ) (youngest oldest : ℕ) 
  (h1 : youngest = (K / 2) - 5)
  (h2 : oldest = 4 * youngest)
  (h3 : oldest = 44) : K = 32 := 
by
  sorry

end kay_age_l128_128528


namespace speed_of_each_train_l128_128249

theorem speed_of_each_train (v : ℝ) (train_length time_cross : ℝ) (km_pr_s : ℝ) 
  (h_train_length : train_length = 120)
  (h_time_cross : time_cross = 8)
  (h_km_pr_s : km_pr_s = 3.6)
  (h_relative_speed : 2 * v = (2 * train_length) / time_cross) :
  v * km_pr_s = 54 := 
by sorry

end speed_of_each_train_l128_128249


namespace geo_seq_fifth_term_l128_128185

theorem geo_seq_fifth_term (a r : ℝ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h3 : a * r^2 = 8) (h7 : a * r^6 = 18) : a * r^4 = 12 :=
sorry

end geo_seq_fifth_term_l128_128185


namespace combined_teaching_experience_l128_128760

def james_teaching_years : ℕ := 40
def partner_teaching_years : ℕ := james_teaching_years - 10

theorem combined_teaching_experience : james_teaching_years + partner_teaching_years = 70 :=
by
  sorry

end combined_teaching_experience_l128_128760


namespace banana_distinct_arrangements_l128_128833

theorem banana_distinct_arrangements : 
  let n := 6
  let n_b := 1
  let n_a := 3
  let n_n := 2
  ∃ arr : ℕ, arr = n.factorial / (n_b.factorial * n_a.factorial * n_n.factorial) ∧ arr = 60 := by
  sorry

end banana_distinct_arrangements_l128_128833


namespace addition_correct_l128_128199

theorem addition_correct :
  1357 + 2468 + 3579 + 4680 + 5791 = 17875 := 
by
  sorry

end addition_correct_l128_128199


namespace max_sector_area_l128_128280

theorem max_sector_area (r θ : ℝ) (h₁ : 2 * r + r * θ = 16) : 
  (∃ A : ℝ, A = 1/2 * r^2 * θ ∧ A ≤ 16) ∧ (∃ r θ, r = 4 ∧ θ = 2 ∧ 1/2 * r^2 * θ = 16) := 
by
  sorry

end max_sector_area_l128_128280


namespace ratio_of_riding_to_total_l128_128800

-- Define the primary conditions from the problem
variables (H R W : ℕ)
variables (legs_on_ground : ℕ := 50)
variables (total_owners : ℕ := 10)
variables (legs_per_horse : ℕ := 4)
variables (legs_per_owner : ℕ := 2)

-- Express the conditions
def conditions : Prop :=
  (legs_on_ground = 6 * W) ∧
  (total_owners = H) ∧
  (H = R + W) ∧
  (H = 10)

-- Define the theorem with the given conditions and prove the required ratio
theorem ratio_of_riding_to_total (H R W : ℕ) (h : conditions H R W) : R / 10 = 1 / 5 := by
  sorry

end ratio_of_riding_to_total_l128_128800


namespace problem_solution_l128_128106

noncomputable def expression_value : ℝ :=
  ((12.983 * 26) / 200) ^ 3 * Real.log 5 / Real.log 10

theorem problem_solution : expression_value = 3.361 := by
  sorry

end problem_solution_l128_128106


namespace gcd_840_1764_l128_128876

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l128_128876


namespace expression_simplification_l128_128210

open Real

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 3*x + y / 3 ≠ 0) :
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = 1 / (3 * (x * y)) :=
by
  -- proof steps would go here
  sorry

end expression_simplification_l128_128210


namespace ira_addition_olya_subtraction_addition_l128_128909

theorem ira_addition (x : ℤ) (h : (11 + x) / (41 + x : ℚ) = 3 / 8) : x = 7 :=
  sorry

theorem olya_subtraction_addition (y : ℤ) (h : (37 - y) / (63 + y : ℚ) = 3 / 17) : y = 22 :=
  sorry

end ira_addition_olya_subtraction_addition_l128_128909


namespace exists_strictly_positive_c_l128_128349

theorem exists_strictly_positive_c {a : ℕ → ℕ → ℝ} (h_diag_pos : ∀ i, a i i > 0)
  (h_off_diag_neg : ∀ i j, i ≠ j → a i j < 0) :
  ∃ (c : ℕ → ℝ), (∀ i, 
    0 < c i) ∧ 
    ((∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 > 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 < 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 = 0)) :=
by
  sorry

end exists_strictly_positive_c_l128_128349


namespace no_fractional_linear_function_l128_128924

noncomputable def fractional_linear_function (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem no_fractional_linear_function (a b c d : ℝ) :
  ∀ x : ℝ, c ≠ 0 → 
  (fractional_linear_function a b c d x + fractional_linear_function b (-d) c (-a) x ≠ -2) :=
by
  sorry

end no_fractional_linear_function_l128_128924


namespace graph_movement_l128_128749

noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1) ^ 2 + 3

noncomputable def g (x : ℝ) : ℝ := -2 * x ^ 2

theorem graph_movement :
  ∀ (x y : ℝ),
  y = f x →
  g x = y → 
  (∃ Δx Δy, Δx = -1 ∧ Δy = -3 ∧ g (x + Δx) = y + Δy) :=
by
  sorry

end graph_movement_l128_128749


namespace evaluate_floor_of_negative_seven_halves_l128_128459

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end evaluate_floor_of_negative_seven_halves_l128_128459


namespace jane_doe_gift_l128_128788

theorem jane_doe_gift (G : ℝ) (h1 : 0.25 * G + 0.1125 * (0.75 * G) = 15000) : G = 41379 := 
sorry

end jane_doe_gift_l128_128788


namespace compartments_count_l128_128262

-- Definition of initial pennies per compartment
def initial_pennies_per_compartment : ℕ := 2

-- Definition of additional pennies added to each compartment
def additional_pennies_per_compartment : ℕ := 6

-- Definition of total pennies is 96
def total_pennies : ℕ := 96

-- Prove the number of compartments is 12
theorem compartments_count (c : ℕ) 
  (h1 : initial_pennies_per_compartment + additional_pennies_per_compartment = 8)
  (h2 : 8 * c = total_pennies) : 
  c = 12 :=
by
  sorry

end compartments_count_l128_128262


namespace polynomial_form_l128_128809

theorem polynomial_form (P : ℝ → ℝ) (h₁ : P 0 = 0) (h₂ : ∀ x, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_form_l128_128809


namespace tan_pi_over_4_plus_alpha_l128_128034

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l128_128034


namespace domain_log_base_4_l128_128904

theorem domain_log_base_4 (x : ℝ) : {x // x + 2 > 0} = {x | x > -2} :=
by
  sorry

end domain_log_base_4_l128_128904


namespace andrew_current_age_l128_128211

-- Definitions based on conditions.
def initial_age := 11  -- Andrew started donating at age 11
def donation_per_year := 7  -- Andrew donates 7k each year on his birthday
def total_donation := 133  -- Andrew has donated a total of 133k till now

-- The theorem stating the problem and the conclusion.
theorem andrew_current_age : 
  ∃ (A : ℕ), donation_per_year * (A - initial_age) = total_donation :=
by {
  sorry
}

end andrew_current_age_l128_128211


namespace no_rational_roots_l128_128347

theorem no_rational_roots : ¬ ∃ x : ℚ, 5 * x^3 - 4 * x^2 - 8 * x + 3 = 0 :=
by
  sorry

end no_rational_roots_l128_128347


namespace count_valid_three_digit_numbers_l128_128476

def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a * 100 + b * 10 + c < 1000) ∧
  (a * 100 + b * 10 + c >= 100) ∧
  (c = 2 * (b - a) + a)

theorem count_valid_three_digit_numbers : ∃ n : ℕ, n = 90 ∧
  ∃ (a b c : ℕ), three_digit_number a b c :=
by
  sorry

end count_valid_three_digit_numbers_l128_128476


namespace game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l128_128811

-- Definitions and conditions for the problem
def num_girls : ℕ := 1994
def tokens (n : ℕ) := n

-- Main theorem statements
theorem game_terminates_if_n_lt_1994 (n : ℕ) (h : n < num_girls) :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (∀ j : ℕ, 1 ≤ j ∧ j ≤ num_girls → (tokens n % num_girls) ≤ 1) :=
by
  sorry

theorem game_does_not_terminate_if_n_eq_1994 :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (tokens 1994 % num_girls = 0) :=
by
  sorry

end game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l128_128811


namespace num_ways_to_turn_off_lights_l128_128570

-- Let's define our problem in terms of the conditions given
-- Define the total number of lights
def total_lights : ℕ := 12

-- Define that we need to turn off 3 lights
def lights_to_turn_off : ℕ := 3

-- Define that we have 10 possible candidates for being turned off 
def candidates := total_lights - 2

-- Define the gap consumption statement that effectively reduce choices to 7 lights
def effective_choices := candidates - lights_to_turn_off

-- Define the combination formula for the number of ways to turn off the lights
def num_ways := Nat.choose effective_choices lights_to_turn_off

-- Final statement to prove
theorem num_ways_to_turn_off_lights : num_ways = Nat.choose 7 3 :=
by
  sorry

end num_ways_to_turn_off_lights_l128_128570


namespace khali_shovels_snow_l128_128991

theorem khali_shovels_snow :
  let section1_length := 30
  let section1_width := 3
  let section1_depth := 1
  let section2_length := 15
  let section2_width := 2
  let section2_depth := 0.5
  let volume1 := section1_length * section1_width * section1_depth
  let volume2 := section2_length * section2_width * section2_depth
  volume1 + volume2 = 105 :=
by 
  sorry

end khali_shovels_snow_l128_128991


namespace num_factors_of_1320_l128_128978

theorem num_factors_of_1320 : ∃ n : ℕ, (n = 24) ∧ (∃ a b c d : ℕ, 1320 = 2^a * 3^b * 5^c * 11^d ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) = 24) :=
by
  sorry

end num_factors_of_1320_l128_128978


namespace clara_total_cookies_l128_128321

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l128_128321


namespace find_values_l128_128204

theorem find_values (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : a = 2 * b + 5) (h3 : Nat.Prime (a + 7 * b)) : (a = 9 ∧ b = 2) ∨ (a = 17 ∧ b = 6) :=
sorry

end find_values_l128_128204


namespace find_distance_l128_128413

variable (D V : ℕ)

axiom normal_speed : V = 25
axiom time_difference : (D / V) - (D / (V + 5)) = 2

theorem find_distance : D = 300 :=
by
  sorry

end find_distance_l128_128413


namespace largest_n_under_100000_l128_128300

theorem largest_n_under_100000 (n : ℕ) : 
  n < 100000 ∧ (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 → n = 99996 :=
by
  sorry

end largest_n_under_100000_l128_128300


namespace rectangle_to_square_l128_128626

-- Definitions based on conditions
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 3
def area : ℕ := rectangle_width * rectangle_height
def parts : ℕ := 3
def part_area : ℕ := area / parts
def square_side : ℕ := Nat.sqrt area

-- Theorem to restate the problem
theorem rectangle_to_square : (area = 36) ∧ (part_area = 12) ∧ (square_side = 6) ∧
  (rectangle_width / parts = 4) ∧ (rectangle_height = 3) ∧ 
  ((rectangle_width / parts * parts) = rectangle_width) ∧ (parts * rectangle_height = square_side ^ 2) := by
  -- Placeholder for proof
  sorry

end rectangle_to_square_l128_128626


namespace gcd_sum_lcm_eq_gcd_l128_128128

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l128_128128


namespace tangent_line_at_point_l128_128322

def f (x : ℝ) : ℝ := x^3 + x - 16

def f' (x : ℝ) : ℝ := 3*x^2 + 1

def tangent_line (x : ℝ) (f'val : ℝ) (p_x p_y : ℝ) : ℝ := f'val * (x - p_x) + p_y

theorem tangent_line_at_point (x y : ℝ) (h : x = 2 ∧ y = -6 ∧ f 2 = -6) : 
  ∃ a b c : ℝ, a*x + b*y + c = 0 ∧ a = 13 ∧ b = -1 ∧ c = -32 :=
by
  use 13, -1, -32
  sorry

end tangent_line_at_point_l128_128322


namespace shortest_ribbon_length_l128_128010

theorem shortest_ribbon_length :
  ∃ (L : ℕ), (∀ (n : ℕ), n = 2 ∨ n = 5 ∨ n = 7 → L % n = 0) ∧ L = 70 :=
by
  sorry

end shortest_ribbon_length_l128_128010


namespace marked_price_of_article_l128_128633

noncomputable def marked_price (discounted_total : ℝ) (num_articles : ℕ) (discount_rate : ℝ) : ℝ :=
  let selling_price_each := discounted_total / num_articles
  let discount_factor := 1 - discount_rate
  selling_price_each / discount_factor

theorem marked_price_of_article :
  marked_price 50 2 0.10 = 250 / 9 :=
by
  unfold marked_price
  -- Instantiate values:
  -- discounted_total = 50
  -- num_articles = 2
  -- discount_rate = 0.10
  sorry

end marked_price_of_article_l128_128633


namespace rogers_parents_paid_percentage_l128_128378

variables 
  (house_cost : ℝ)
  (down_payment_percentage : ℝ)
  (remaining_balance_owed : ℝ)
  (down_payment : ℝ := down_payment_percentage * house_cost)
  (remaining_balance_after_down : ℝ := house_cost - down_payment)
  (parents_payment : ℝ := remaining_balance_after_down - remaining_balance_owed)
  (percentage_paid_by_parents : ℝ := (parents_payment / remaining_balance_after_down) * 100)

theorem rogers_parents_paid_percentage
  (h1 : house_cost = 100000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : remaining_balance_owed = 56000) :
  percentage_paid_by_parents = 30 :=
sorry

end rogers_parents_paid_percentage_l128_128378


namespace quadratic_expression_value_l128_128480

theorem quadratic_expression_value (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  a^2 - 3 * a + 1 = 0) → 
  a^2 - 2 * a + 2021 + 1 / a = 2023 := 
sorry

end quadratic_expression_value_l128_128480


namespace probability_top_king_of_hearts_l128_128112

def deck_size : ℕ := 52

def king_of_hearts_count : ℕ := 1

def probability_king_of_hearts_top_card (n : ℕ) (k : ℕ) : ℚ :=
  if n ≠ 0 then k / n else 0

theorem probability_top_king_of_hearts : 
  probability_king_of_hearts_top_card deck_size king_of_hearts_count = 1 / 52 :=
by
  -- Proof omitted
  sorry

end probability_top_king_of_hearts_l128_128112


namespace infinite_solutions_l128_128372

theorem infinite_solutions (b : ℤ) : 
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := 
by sorry

end infinite_solutions_l128_128372


namespace inverse_proportion_passing_through_l128_128384

theorem inverse_proportion_passing_through (k : ℝ) :
  (∀ x y : ℝ, (y = k / x) → (x = 3 → y = 2)) → k = 6 := 
by
  sorry

end inverse_proportion_passing_through_l128_128384


namespace proof_w3_u2_y2_l128_128735

variable (x y z w u d : ℤ)

def arithmetic_sequence := x = 1370 ∧ z = 1070 ∧ w = -180 ∧ u = -6430 ∧ (z = x + 2 * d) ∧ (y = x + d)

theorem proof_w3_u2_y2 (h : arithmetic_sequence x y z w u d) : w^3 - u^2 + y^2 = -44200100 :=
  by
    sorry

end proof_w3_u2_y2_l128_128735


namespace fraction_to_decimal_l128_128761

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  -- Prove that the fraction 5/8 equals the decimal 0.625
  sorry

end fraction_to_decimal_l128_128761


namespace integer_part_of_result_is_40_l128_128125

noncomputable def numerator : ℝ := 0.1 + 1.2 + 2.3 + 3.4 + 4.5 + 5.6 + 6.7 + 7.8 + 8.9
noncomputable def denominator : ℝ := 0.01 + 0.03 + 0.05 + 0.07 + 0.09 + 0.11 + 0.13 + 0.15 + 0.17 + 0.19
noncomputable def result : ℝ := numerator / denominator

theorem integer_part_of_result_is_40 : ⌊result⌋ = 40 := 
by
  -- proof goes here
  sorry

end integer_part_of_result_is_40_l128_128125


namespace birds_remaining_l128_128752

variable (initial_birds : ℝ) (birds_flew_away : ℝ)

theorem birds_remaining (h1 : initial_birds = 12.0) (h2 : birds_flew_away = 8.0) : initial_birds - birds_flew_away = 4.0 :=
by
  rw [h1, h2]
  norm_num

end birds_remaining_l128_128752


namespace servings_in_bottle_l128_128714

theorem servings_in_bottle (total_revenue : ℕ) (price_per_serving : ℕ) (h1 : total_revenue = 98) (h2 : price_per_serving = 8) : Nat.floor (total_revenue / price_per_serving) = 12 :=
by
  sorry

end servings_in_bottle_l128_128714


namespace paint_problem_l128_128855

-- Definitions based on conditions
def roomsInitiallyPaintable := 50
def roomsAfterLoss := 40
def cansLost := 5

-- The number of rooms each can could paint
def roomsPerCan := (roomsInitiallyPaintable - roomsAfterLoss) / cansLost

-- The total number of cans originally owned
def originalCans := roomsInitiallyPaintable / roomsPerCan

-- Theorem to prove the number of original cans equals 25
theorem paint_problem : originalCans = 25 := by
  sorry

end paint_problem_l128_128855


namespace students_at_start_of_year_l128_128973

-- Define the initial number of students as a variable S
variables (S : ℕ)

-- Define the conditions
def condition_1 := S - 18 + 14 = 29

-- State the theorem to be proved
theorem students_at_start_of_year (h : condition_1 S) : S = 33 :=
sorry

end students_at_start_of_year_l128_128973


namespace eggs_needed_per_month_l128_128479

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l128_128479


namespace no_solution_frac_eq_l128_128235

theorem no_solution_frac_eq (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  3 / x + 6 / (x - 1) - (x + 5) / (x * (x - 1)) ≠ 0 :=
by {
  sorry
}

end no_solution_frac_eq_l128_128235


namespace sum_of_ages_l128_128299

theorem sum_of_ages (juliet_age maggie_age ralph_age nicky_age : ℕ)
  (h1 : juliet_age = 10)
  (h2 : juliet_age = maggie_age + 3)
  (h3 : ralph_age = juliet_age + 2)
  (h4 : nicky_age = ralph_age / 2) :
  maggie_age + ralph_age + nicky_age = 25 :=
by
  sorry

end sum_of_ages_l128_128299


namespace smallest_mn_sum_l128_128150

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l128_128150


namespace number_square_of_digits_l128_128635

theorem number_square_of_digits (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) :
  ∃ n : ℕ, (∃ (k : ℕ), (1001 * x + 110 * y) = k^2) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_square_of_digits_l128_128635


namespace largest_possible_number_of_pencils_in_a_box_l128_128174

/-- Olivia bought 48 pencils -/
def olivia_pencils : ℕ := 48
/-- Noah bought 60 pencils -/
def noah_pencils : ℕ := 60
/-- Liam bought 72 pencils -/
def liam_pencils : ℕ := 72

/-- The GCD of the number of pencils bought by Olivia, Noah, and Liam is 12 -/
theorem largest_possible_number_of_pencils_in_a_box :
  gcd olivia_pencils (gcd noah_pencils liam_pencils) = 12 :=
by {
  sorry
}

end largest_possible_number_of_pencils_in_a_box_l128_128174


namespace weight_in_one_hand_l128_128632

theorem weight_in_one_hand (total_weight : ℕ) (h : total_weight = 16) : total_weight / 2 = 8 :=
by
  sorry

end weight_in_one_hand_l128_128632


namespace power_function_decreasing_m_l128_128837

theorem power_function_decreasing_m :
  ∀ (m : ℝ), (m^2 - 5*m - 5) * (2*m + 1) < 0 → m = -1 :=
by
  sorry

end power_function_decreasing_m_l128_128837


namespace smaller_cuboid_length_l128_128726

theorem smaller_cuboid_length
  (L : ℝ)
  (h1 : 32 * (L * 4 * 3) = 16 * 10 * 12) :
  L = 5 :=
by
  sorry

end smaller_cuboid_length_l128_128726


namespace find_x_l128_128784

theorem find_x (x : ℝ) 
  (h: 3 * x + 6 * x + 2 * x + x = 360) : 
  x = 30 := 
sorry

end find_x_l128_128784


namespace find_intersection_l128_128495

open Set Real

def domain_A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def domain_B : Set ℝ := {x : ℝ | x < 1}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem find_intersection :
  intersection domain_A domain_B = {x : ℝ | -2 ≤ x ∧ x < 1} := 
by sorry

end find_intersection_l128_128495


namespace Adam_total_candy_l128_128863

theorem Adam_total_candy :
  (2 + 5) * 4 = 28 := 
by 
  sorry

end Adam_total_candy_l128_128863


namespace find_first_term_l128_128606

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

variable (a1 a3 a9 d : ℤ)

-- Given conditions
axiom h1 : arithmetic_seq a1 d 2 = 30
axiom h2 : arithmetic_seq a1 d 8 = 60

theorem find_first_term : a1 = 20 :=
by
  -- mathematical proof steps here
  sorry

end find_first_term_l128_128606


namespace beef_weight_loss_l128_128207

theorem beef_weight_loss (weight_before weight_after: ℕ) 
                         (h1: weight_before = 400) 
                         (h2: weight_after = 240) : 
                         ((weight_before - weight_after) * 100 / weight_before = 40) :=
by 
  sorry

end beef_weight_loss_l128_128207


namespace odd_numbers_not_dividing_each_other_l128_128140

theorem odd_numbers_not_dividing_each_other (n : ℕ) (hn : n ≥ 4) :
  ∃ (a b : ℕ), a ≠ b ∧ (2 ^ (2 * n) < a ∧ a < 2 ^ (3 * n)) ∧ 
  (2 ^ (2 * n) < b ∧ b < 2 ^ (3 * n)) ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  ¬ (a ∣ b * b) ∧ ¬ (b ∣ a * a) := by
sorry

end odd_numbers_not_dividing_each_other_l128_128140


namespace percent_increase_l128_128598

/-- Problem statement: Given (1/2)x = 1, prove that the percentage increase from 1/2 to x is 300%. -/
theorem percent_increase (x : ℝ) (h : (1/2) * x = 1) : 
  ((x - (1/2)) / (1/2)) * 100 = 300 := 
by
  sorry

end percent_increase_l128_128598


namespace problem_statement_l128_128483

-- Mathematical Conditions
variables (a : ℝ)

-- Sufficient but not necessary condition proof statement
def sufficient_but_not_necessary : Prop :=
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧ ¬(∀ a : ℝ, a^2 + a ≥ 0 → a > 0)

-- Main problem to be proved
theorem problem_statement : sufficient_but_not_necessary :=
by
  sorry

end problem_statement_l128_128483


namespace fireworks_display_l128_128062

-- Define numbers and conditions
def display_fireworks_for_number (n : ℕ) : ℕ := 6
def display_fireworks_for_letter (c : Char) : ℕ := 5
def fireworks_per_box : ℕ := 8
def number_boxes : ℕ := 50

-- Calculate fireworks for the year 2023
def fireworks_for_year : ℕ :=
  display_fireworks_for_number 2 * 2 +
  display_fireworks_for_number 0 * 1 +
  display_fireworks_for_number 3 * 1

-- Calculate fireworks for "HAPPY NEW YEAR"
def fireworks_for_phrase : ℕ :=
  12 * display_fireworks_for_letter 'H'

-- Calculate fireworks for 50 boxes
def fireworks_for_boxes : ℕ := number_boxes * fireworks_per_box

-- Total fireworks calculation
def total_fireworks : ℕ := fireworks_for_year + fireworks_for_phrase + fireworks_for_boxes

-- Proof statement
theorem fireworks_display : total_fireworks = 476 := 
  by
  -- This is where the proof would go.
  sorry

end fireworks_display_l128_128062


namespace initial_shirts_count_l128_128279

theorem initial_shirts_count 
  (S T x : ℝ)
  (h1 : 2 * S + x * T = 1600)
  (h2 : S + 6 * T = 1600)
  (h3 : 12 * T = 2400) :
  x = 4 :=
by
  sorry

end initial_shirts_count_l128_128279


namespace find_t_value_l128_128638

theorem find_t_value (t : ℝ) (a b : ℝ × ℝ) (h₁ : a = (t, 1)) (h₂ : b = (1, 2)) 
  (h₃ : (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2) : 
  t = -2 :=
by 
  sorry

end find_t_value_l128_128638


namespace polynomial_integer_roots_l128_128666

theorem polynomial_integer_roots
  (b c : ℤ)
  (x1 x2 x1' x2' : ℤ)
  (h_eq1 : x1 * x2 > 0)
  (h_eq2 : x1' * x2' > 0)
  (h_eq3 : x1^2 + b * x1 + c = 0)
  (h_eq4 : x2^2 + b * x2 + c = 0)
  (h_eq5 : x1'^2 + c * x1' + b = 0)
  (h_eq6 : x2'^2 + c * x2' + b = 0)
  : x1 < 0 ∧ x2 < 0 ∧ b - 1 ≤ c ∧ c ≤ b + 1 ∧ 
    ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) := 
sorry

end polynomial_integer_roots_l128_128666


namespace parrot_consumption_l128_128634

theorem parrot_consumption :
  ∀ (parakeet_daily : ℕ) (finch_daily : ℕ) (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ) (weekly_birdseed : ℕ),
    parakeet_daily = 2 →
    finch_daily = parakeet_daily / 2 →
    num_parakeets = 3 →
    num_parrots = 2 →
    num_finches = 4 →
    weekly_birdseed = 266 →
    14 = (weekly_birdseed - ((num_parakeets * parakeet_daily + num_finches * finch_daily) * 7)) / num_parrots / 7 :=
by
  intros parakeet_daily finch_daily num_parakeets num_parrots num_finches weekly_birdseed
  intros hp1 hp2 hp3 hp4 hp5 hp6
  sorry

end parrot_consumption_l128_128634


namespace larrys_correct_substitution_l128_128171

noncomputable def lucky_larry_expression (a b c d e f : ℤ) : ℤ :=
  a + (b - (c + (d - (e + f))))

noncomputable def larrys_substitution (a b c d e f : ℤ) : ℤ :=
  a + b - c + d - e + f

theorem larrys_correct_substitution : 
  (lucky_larry_expression 2 4 6 8 e 5 = larrys_substitution 2 4 6 8 e 5) ↔ (e = 8) :=
by
  sorry

end larrys_correct_substitution_l128_128171


namespace total_cantaloupes_l128_128551

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end total_cantaloupes_l128_128551


namespace rod_length_l128_128196

theorem rod_length (L : ℝ) (weight : ℝ → ℝ) (weight_6m : weight 6 = 14.04) (weight_L : weight L = 23.4) :
  L = 10 :=
by 
  sorry

end rod_length_l128_128196


namespace probability_two_cards_l128_128989

noncomputable def probability_first_spade_second_ace : ℚ :=
  let total_cards := 52
  let total_spades := 13
  let total_aces := 4
  let remaining_cards := total_cards - 1
  
  let first_spade_non_ace := (total_spades - 1) / total_cards
  let second_ace_after_non_ace := total_aces / remaining_cards
  
  let probability_case1 := first_spade_non_ace * second_ace_after_non_ace
  
  let first_ace_spade := 1 / total_cards
  let second_ace_after_ace := (total_aces - 1) / remaining_cards
  
  let probability_case2 := first_ace_spade * second_ace_after_ace
  
  probability_case1 + probability_case2

theorem probability_two_cards {p : ℚ} (h : p = 1 / 52) : 
  probability_first_spade_second_ace = p := 
by 
  simp only [probability_first_spade_second_ace]
  sorry

end probability_two_cards_l128_128989


namespace solve_quadratic_eq_l128_128342

theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 15 = 0 ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end solve_quadratic_eq_l128_128342


namespace equal_diagonals_implies_quad_or_pent_l128_128431

-- Define a convex polygon with n edges and equal diagonals
structure ConvexPolygon (n : ℕ) :=
(edges : ℕ)
(convex : Prop)
(diagonalsEqualLength : Prop)

-- State the theorem to prove
theorem equal_diagonals_implies_quad_or_pent (n : ℕ) (poly : ConvexPolygon n) 
    (h1 : poly.convex) 
    (h2 : poly.diagonalsEqualLength) :
    (n = 4) ∨ (n = 5) :=
sorry

end equal_diagonals_implies_quad_or_pent_l128_128431


namespace inequality_geq_l128_128485

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end inequality_geq_l128_128485


namespace complex_modulus_eq_one_l128_128535

open Complex

theorem complex_modulus_eq_one (a b : ℝ) (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 2 - Complex.I) :
  abs (a - b * Complex.I) = 1 := by
  sorry

end complex_modulus_eq_one_l128_128535


namespace inequality_f_l128_128212

noncomputable def f (x y z : ℝ) : ℝ :=
  x * y + y * z + z * x - 2 * x * y * z

theorem inequality_f (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ f x y z ∧ f x y z ≤ 7 / 27 :=
  sorry

end inequality_f_l128_128212


namespace real_root_exists_l128_128366

theorem real_root_exists (a : ℝ) : 
    (∃ x : ℝ, x^4 - a * x^3 - x^2 - a * x + 1 = 0) ↔ (-1 / 2 ≤ a) := by
  sorry

end real_root_exists_l128_128366


namespace connected_paper_area_l128_128627

def side_length := 30 -- side of each square paper in cm
def overlap_length := 7 -- overlap length in cm
def num_pieces := 6 -- number of paper pieces

def effective_length (side_length overlap_length : ℕ) := side_length - overlap_length
def total_connected_length (num_pieces : ℕ) (side_length overlap_length : ℕ) :=
  side_length + (num_pieces - 1) * (effective_length side_length overlap_length)

def width := side_length -- width of the connected paper is the side of each square piece of paper

def area (length width : ℕ) := length * width

theorem connected_paper_area : area (total_connected_length num_pieces side_length overlap_length) width = 4350 :=
by
  sorry

end connected_paper_area_l128_128627


namespace largest_number_sum13_product36_l128_128354

-- helper definitions for sum and product of digits
def sum_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.sum
def mul_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.foldr (· * ·) 1

theorem largest_number_sum13_product36 : 
  ∃ n : ℕ, sum_digits n = 13 ∧ mul_digits n = 36 ∧ ∀ m : ℕ, sum_digits m = 13 ∧ mul_digits m = 36 → m ≤ n :=
sorry

end largest_number_sum13_product36_l128_128354


namespace problem_statement_l128_128470

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 4
def g (x : ℝ) : ℝ := 2*x - 1

-- State the theorem and provide the necessary conditions
theorem problem_statement : f (g 5) - g (f 5) = 381 :=
by
  sorry

end problem_statement_l128_128470


namespace count_two_digit_perfect_squares_divisible_by_4_l128_128880

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l128_128880


namespace overall_ratio_men_women_l128_128070

variables (m_w_diff players_total beginners_m beginners_w intermediate_m intermediate_w advanced_m advanced_w : ℕ)

def total_men : ℕ := beginners_m + intermediate_m + advanced_m
def total_women : ℕ := beginners_w + intermediate_w + advanced_w

theorem overall_ratio_men_women 
  (h1 : beginners_m = 2) 
  (h2 : beginners_w = 4)
  (h3 : intermediate_m = 3) 
  (h4 : intermediate_w = 5) 
  (h5 : advanced_m = 1) 
  (h6 : advanced_w = 3) 
  (h7 : m_w_diff = 4)
  (h8 : total_men = 6)
  (h9 : total_women = 12)
  (h10 : players_total = 18) :
  total_men / total_women = 1 / 2 :=
by {
  sorry
}

end overall_ratio_men_women_l128_128070


namespace brownies_pieces_count_l128_128896

theorem brownies_pieces_count
  (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by
  sorry

end brownies_pieces_count_l128_128896


namespace conversion_problem_l128_128489

noncomputable def conversion1 : ℚ :=
  35 * (1/1000)  -- to convert cubic decimeters to cubic meters

noncomputable def conversion2 : ℚ :=
  53 * (1/60)  -- to convert seconds to minutes

noncomputable def conversion3 : ℚ :=
  5 * (1/60)  -- to convert minutes to hours

noncomputable def conversion4 : ℚ :=
  1 * (1/100)  -- to convert square centimeters to square decimeters

noncomputable def conversion5 : ℚ :=
  450 * (1/1000)  -- to convert milliliters to liters

theorem conversion_problem : 
  (conversion1 = 7 / 200) ∧ 
  (conversion2 = 53 / 60) ∧ 
  (conversion3 = 1 / 12) ∧ 
  (conversion4 = 1 / 100) ∧ 
  (conversion5 = 9 / 20) :=
by
  sorry

end conversion_problem_l128_128489


namespace inequality_system_range_l128_128865

theorem inequality_system_range (a : ℝ) :
  (∃ (x : ℤ), (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0)) ∧
  (∀ x : ℤ, (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0) → (x = 2 ∨ x = 3)) →
  6 ≤ a ∧ a < 8 :=
by
  sorry

end inequality_system_range_l128_128865


namespace solution_set_f_over_x_lt_0_l128_128340

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_over_x_lt_0 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x1 x2, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) →
  (f 4 = 0) →
  { x | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
by
  intros _ _ _
  sorry

end solution_set_f_over_x_lt_0_l128_128340


namespace sum_faces_edges_vertices_of_octagonal_pyramid_l128_128344

-- We define an octagonal pyramid with the given geometric properties.
structure OctagonalPyramid :=
  (base_vertices : ℕ) -- the number of vertices of the base
  (base_edges : ℕ)    -- the number of edges of the base
  (apex : ℕ)          -- the single apex of the pyramid
  (faces : ℕ)         -- the total number of faces: base face + triangular faces
  (edges : ℕ)         -- the total number of edges
  (vertices : ℕ)      -- the total number of vertices

-- Now we instantiate the structure based on the conditions.
def octagonalPyramid : OctagonalPyramid :=
  { base_vertices := 8,
    base_edges := 8,
    apex := 1,
    faces := 9,
    edges := 16,
    vertices := 9 }

-- We prove that the total number of faces, edges, and vertices sum to 34.
theorem sum_faces_edges_vertices_of_octagonal_pyramid : 
  (octagonalPyramid.faces + octagonalPyramid.edges + octagonalPyramid.vertices = 34) :=
by
  -- The proof steps are omitted as per instruction.
  sorry

end sum_faces_edges_vertices_of_octagonal_pyramid_l128_128344


namespace calc_expr_l128_128215

theorem calc_expr : 
  (-1: ℝ)^4 - 2 * Real.tan (Real.pi / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := 
by
  sorry

end calc_expr_l128_128215


namespace total_students_is_45_l128_128429

-- Define the initial conditions with the definitions provided
def drunk_drivers : Nat := 6
def speeders : Nat := 7 * drunk_drivers - 3
def total_students : Nat := drunk_drivers + speeders

-- The theorem to prove that the total number of students is 45
theorem total_students_is_45 : total_students = 45 :=
by
  sorry

end total_students_is_45_l128_128429


namespace find_x_l128_128856

theorem find_x (x : ℝ) (h : 6 * x + 3 * x + 4 * x + 2 * x = 360) : x = 24 :=
sorry

end find_x_l128_128856


namespace valve_difference_l128_128776

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end valve_difference_l128_128776


namespace value_of_D_l128_128595

theorem value_of_D (D : ℤ) (h : 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89) : D = -5 :=
by sorry

end value_of_D_l128_128595


namespace alpha_value_l128_128377

theorem alpha_value
  (β γ δ α : ℝ) 
  (h1 : β = 100)
  (h2 : γ = 30)
  (h3 : δ = 150)
  (h4 : α + β + γ + 0.5 * γ = 360) : 
  α = 215 :=
by
  sorry

end alpha_value_l128_128377


namespace children_got_on_bus_l128_128511

-- Definitions based on conditions
def initial_children : ℕ := 22
def children_got_off : ℕ := 60
def children_after_stop : ℕ := 2

-- Define the problem
theorem children_got_on_bus : ∃ x : ℕ, initial_children - children_got_off + x = children_after_stop ∧ x = 40 :=
by
  sorry

end children_got_on_bus_l128_128511


namespace Jen_visits_either_but_not_both_l128_128407

-- Define the events and their associated probabilities
def P_Chile : ℝ := 0.30
def P_Madagascar : ℝ := 0.50

-- Define the probability of visiting both assuming independence
def P_both : ℝ := P_Chile * P_Madagascar

-- Define the probability of visiting either but not both
def P_either_but_not_both : ℝ := P_Chile + P_Madagascar - 2 * P_both

-- The problem statement
theorem Jen_visits_either_but_not_both : P_either_but_not_both = 0.65 := by
  /- The proof goes here -/
  sorry

end Jen_visits_either_but_not_both_l128_128407


namespace value_of_b_l128_128771

theorem value_of_b (x b : ℝ) (h₁ : x = 0.3) 
  (h₂ : (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : 
  b = 2 :=
by
  sorry

end value_of_b_l128_128771


namespace find_second_number_l128_128488

theorem find_second_number :
  ∃ (x y : ℕ), (y = x + 4) ∧ (x + y = 56) ∧ (y = 30) :=
by
  sorry

end find_second_number_l128_128488


namespace probability_A_does_not_lose_l128_128508

theorem probability_A_does_not_lose (pA_wins p_draw : ℝ) (hA_wins : pA_wins = 0.4) (h_draw : p_draw = 0.2) :
  pA_wins + p_draw = 0.6 :=
by
  sorry

end probability_A_does_not_lose_l128_128508


namespace volunteer_arrangements_l128_128874

theorem volunteer_arrangements (students : Fin 5 → String) (events : Fin 3 → String)
  (A : String) (high_jump : String)
  (h : ∀ (arrange : Fin 3 → Fin 5), ¬(students (arrange 0) = A ∧ events 0 = high_jump)) :
  ∃! valid_arrangements, valid_arrangements = 48 :=
by
  sorry

end volunteer_arrangements_l128_128874


namespace nominal_rate_of_interest_correct_l128_128055

noncomputable def nominal_rate_of_interest (EAR : ℝ) (n : ℕ) : ℝ :=
  let i := by 
    sorry
  i

theorem nominal_rate_of_interest_correct :
  nominal_rate_of_interest 0.0609 2 = 0.0598 :=
by 
  sorry

end nominal_rate_of_interest_correct_l128_128055


namespace find_value_of_a_l128_128089

theorem find_value_of_a (a : ℝ) (h : 0.005 * a = 65) : a = 130 := 
by
  sorry

end find_value_of_a_l128_128089


namespace tuples_and_triples_counts_are_equal_l128_128309

theorem tuples_and_triples_counts_are_equal (n : ℕ) (h : n > 0) :
  let countTuples := 8^n - 2 * 7^n + 6^n
  let countTriples := 8^n - 2 * 7^n + 6^n
  countTuples = countTriples :=
by
  sorry

end tuples_and_triples_counts_are_equal_l128_128309


namespace matrix_scalars_exist_l128_128945

namespace MatrixProof

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, -1]]

theorem matrix_scalars_exist :
  ∃ r s : ℝ, B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ r = 0 ∧ s = 64 := by
  sorry

end MatrixProof

end matrix_scalars_exist_l128_128945


namespace range_of_a_l128_128745

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < (π / 2) → a ≤ 1 / Real.sin θ + 1 / Real.cos θ) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_a_l128_128745


namespace actors_per_group_l128_128622

theorem actors_per_group (actors_per_hour : ℕ) (show_time_per_actor : ℕ) (total_show_time : ℕ)
  (h1 : show_time_per_actor = 15) (h2 : actors_per_hour = 20) (h3 : total_show_time = 60) :
  actors_per_hour * show_time_per_actor / total_show_time = 5 :=
by sorry

end actors_per_group_l128_128622


namespace range_of_a_l128_128691

-- Define the function f(x) and its condition
def f (x a : ℝ) : ℝ := x^2 + (a + 2) * x + (a - 1)

-- Given condition: f(-1, a) = -2
def condition (a : ℝ) : Prop := f (-1) a = -2

-- Requirement for the domain of g(x) = ln(f(x) + 3) being ℝ
def domain_requirement (a : ℝ) : Prop := ∀ x : ℝ, f x a + 3 > 0

-- Main theorem to prove the range of a
theorem range_of_a : {a : ℝ // condition a ∧ domain_requirement a} = {a : ℝ // -2 < a ∧ a < 2} :=
by sorry

end range_of_a_l128_128691


namespace consecutive_even_sum_l128_128579

theorem consecutive_even_sum (n : ℤ) (h : (n - 2) + (n + 2) = 156) : n = 78 :=
by
  sorry

end consecutive_even_sum_l128_128579


namespace range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l128_128433

open Real

theorem range_a_of_abs_2x_minus_a_eq_1_two_real_solutions :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (abs (2^x1 - a) = 1) ∧ (abs (2^x2 - a) = 1)} = {a : ℝ | 1 < a} :=
by
  sorry

end range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l128_128433


namespace math_problem_l128_128017

noncomputable def problem_statement : Prop :=
  let A : ℝ × ℝ := (5, 6)
  let B : ℝ × ℝ := (8, 3)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  let y_intercept : ℝ := A.snd - slope * A.fst
  slope + y_intercept = 10

theorem math_problem : problem_statement := sorry

end math_problem_l128_128017


namespace decreasing_interval_l128_128653

theorem decreasing_interval (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 - 2 * x) :
  {x | deriv f x < 0} = {x | x < 1} :=
by
  sorry

end decreasing_interval_l128_128653


namespace complete_the_square_problem_l128_128637

theorem complete_the_square_problem :
  ∃ r s : ℝ, (r = -2) ∧ (s = 9) ∧ (r + s = 7) ∧ ∀ x : ℝ, 15 * x ^ 2 - 60 * x - 135 = 0 ↔ (x + r) ^ 2 = s := 
by
  sorry

end complete_the_square_problem_l128_128637


namespace most_stable_performance_l128_128518

-- Given variances for the students' scores
def variance_A : ℝ := 2.1
def variance_B : ℝ := 3.5
def variance_C : ℝ := 9
def variance_D : ℝ := 0.7

-- Prove that student D has the most stable performance
theorem most_stable_performance : 
  variance_D < variance_A ∧ variance_D < variance_B ∧ variance_D < variance_C := 
  by 
    sorry

end most_stable_performance_l128_128518


namespace coin_flip_heads_probability_l128_128987

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l128_128987


namespace xiao_ming_climb_stairs_8_l128_128651

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => fibonacci n + fibonacci (n + 1)

theorem xiao_ming_climb_stairs_8 :
  fibonacci 8 = 34 :=
sorry

end xiao_ming_climb_stairs_8_l128_128651


namespace roots_quartic_sum_l128_128753

theorem roots_quartic_sum (c d : ℝ) (h1 : c + d = 3) (h2 : c * d = 1) (hc : Polynomial.eval c (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) (hd : Polynomial.eval d (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) :
  c * d + c + d = 4 :=
by
  sorry

end roots_quartic_sum_l128_128753


namespace sum_mod_13_l128_128829

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l128_128829


namespace quadratic_no_real_roots_range_l128_128504

theorem quadratic_no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l128_128504


namespace area_of_gray_region_l128_128381

open Real

-- Define the circles and the radii.
def circleC_center : Prod Real Real := (5, 5)
def radiusC : Real := 5

def circleD_center : Prod Real Real := (15, 5)
def radiusD : Real := 5

-- The main theorem stating the area of the gray region bound by the circles and the x-axis.
theorem area_of_gray_region : 
  let area_rectangle := (10:Real) * (5:Real)
  let area_sectors := (2:Real) * ((1/4) * (5:Real)^2 * π)
  area_rectangle - area_sectors = 50 - 12.5 * π :=
by
  sorry

end area_of_gray_region_l128_128381


namespace algebra_identity_l128_128766

theorem algebra_identity (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) : x^2 - y^2 = 8 := by
  sorry

end algebra_identity_l128_128766


namespace z_pow12_plus_inv_z_pow12_l128_128704

open Complex

theorem z_pow12_plus_inv_z_pow12 (z: ℂ) (h: z + z⁻¹ = 2 * cos (10 * Real.pi / 180)) :
  z^12 + z⁻¹^12 = -1 := by
  sorry

end z_pow12_plus_inv_z_pow12_l128_128704


namespace Mr_A_financial_outcome_l128_128425

def home_worth : ℝ := 200000
def profit_percent : ℝ := 0.15
def loss_percent : ℝ := 0.05

def selling_price := (1 + profit_percent) * home_worth
def buying_price := (1 - loss_percent) * selling_price

theorem Mr_A_financial_outcome : 
  selling_price - buying_price = 11500 :=
by
  sorry

end Mr_A_financial_outcome_l128_128425


namespace angle_bounds_find_configurations_l128_128079

/-- Given four points A, B, C, D on a plane, where α1 and α2 are the two smallest angles,
    and β1 and β2 are the two largest angles formed by these points, we aim to prove:
    1. 0 ≤ α2 ≤ 45 degrees,
    2. 72 degrees ≤ β2 ≤ 180 degrees,
    and to find configurations that achieve α2 = 45 degrees and β2 = 72 degrees. -/
theorem angle_bounds {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ) 
  (h_angles : α1 ≤ α2 ∧ α2 ≤ β2 ∧ β2 ≤ β1 ∧ 
              0 ≤ α2 ∧ α2 ≤ 45 ∧ 
              72 ≤ β2 ∧ β2 ≤ 180) : 
  (0 ≤ α2 ∧ α2 ≤ 45 ∧ 72 ≤ β2 ∧ β2 ≤ 180) := 
by sorry

/-- Find configurations where α2 = 45 degrees and β2 = 72 degrees. -/
theorem find_configurations {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ)
  (h_angles : α1 ≤ α2 ∧ α2 = 45 ∧ β2 = 72 ∧ β2 ≤ β1) :
  (α2 = 45 ∧ β2 = 72) := 
by sorry

end angle_bounds_find_configurations_l128_128079


namespace line_intersects_extension_of_segment_l128_128258

theorem line_intersects_extension_of_segment
  (A B C x1 y1 x2 y2 : ℝ)
  (hnz : A ≠ 0 ∨ B ≠ 0)
  (h1 : (A * x1 + B * y1 + C) * (A * x2 + B * y2 + C) > 0)
  (h2 : |A * x1 + B * y1 + C| > |A * x2 + B * y2 + C|) :
  ∃ t : ℝ, t ≥ 0 ∧ l * (t * (x2 - x1) + x1) + m * (t * (y2 - y1) + y1) = 0 :=
sorry

end line_intersects_extension_of_segment_l128_128258


namespace intersection_is_line_l128_128294

-- Define the two planes as given in the conditions
def plane1 (x y z : ℝ) : Prop := x + 5 * y + 2 * z - 5 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 5 * y - z + 5 = 0

-- The intersection of the planes should satisfy both plane equations
def is_on_line (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the canonical equation of the line
def line_eq (x y z : ℝ) : Prop := (∃ k : ℝ, x = 5 * k ∧ y = 5 * k + 1 ∧ z = -15 * k)

-- The proof statement
theorem intersection_is_line :
  (∀ x y z : ℝ, is_on_line x y z → line_eq x y z) ∧ 
  (∀ x y z : ℝ, line_eq x y z → is_on_line x y z) :=
by
  sorry

end intersection_is_line_l128_128294


namespace find_workers_l128_128819

def total_workers := 20
def male_work_days := 2
def female_work_days := 3

theorem find_workers (X Y : ℕ) 
  (h1 : X + Y = total_workers)
  (h2 : X / male_work_days + Y / female_work_days = 1) : 
  X = 12 ∧ Y = 8 :=
sorry

end find_workers_l128_128819


namespace smallest_digit_divisible_by_9_l128_128964

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, (∃ m : ℕ, m = 2 + 4 + d + 6 + 0 ∧ m % 9 = 0 ∧ d < 10) ∧ d = 6 :=
by
  sorry

end smallest_digit_divisible_by_9_l128_128964


namespace train_passes_jogger_time_l128_128254

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 75
noncomputable def jogger_head_start_m : ℝ := 500
noncomputable def train_length_m : ℝ := 300

noncomputable def km_per_hr_to_m_per_s (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def jogger_speed_m_per_s := km_per_hr_to_m_per_s jogger_speed_km_per_hr
noncomputable def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

noncomputable def relative_speed_m_per_s := train_speed_m_per_s - jogger_speed_m_per_s

noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m

theorem train_passes_jogger_time :
  let time_to_pass := total_distance_to_cover_m / relative_speed_m_per_s
  abs (time_to_pass - 43.64) < 0.01 :=
by
  sorry

end train_passes_jogger_time_l128_128254


namespace polygon_sides_l128_128906

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : 
  n = 8 := 
sorry

end polygon_sides_l128_128906


namespace smallest_non_consecutive_product_not_factor_of_48_l128_128892

def is_factor (a b : ℕ) : Prop := b % a = 0

def non_consecutive_pairs (x y : ℕ) : Prop := (x ≠ y) ∧ (x + 1 ≠ y) ∧ (y + 1 ≠ x)

theorem smallest_non_consecutive_product_not_factor_of_48 :
  ∃ x y, x ∣ 48 ∧ y ∣ 48 ∧ non_consecutive_pairs x y ∧ ¬ (x * y ∣ 48) ∧ (∀ x' y', x' ∣ 48 ∧ y' ∣ 48 ∧ non_consecutive_pairs x' y' ∧ ¬ (x' * y' ∣ 48) → x' * y' ≥ 18) :=
by
  sorry

end smallest_non_consecutive_product_not_factor_of_48_l128_128892


namespace f_inv_f_inv_17_l128_128376

noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def f_inv (y : ℝ) : ℝ := (y + 3) / 4

theorem f_inv_f_inv_17 : f_inv (f_inv 17) = 2 := by
  sorry

end f_inv_f_inv_17_l128_128376


namespace nada_house_size_l128_128493

variable (N : ℕ) -- N represents the size of Nada's house

theorem nada_house_size :
  (1000 = 2 * N + 100) → (N = 450) :=
by
  intro h
  sorry

end nada_house_size_l128_128493


namespace car_speeds_midpoint_condition_l128_128039

theorem car_speeds_midpoint_condition 
  (v k : ℝ) (h_k : k > 1) 
  (A B C D : ℝ) (AB AD CD : ℝ)
  (h_midpoint : AD = AB / 2) 
  (h_CD_AD : CD / AD = 1 / 2)
  (h_D_midpoint : D = (A + B) / 2) 
  (h_C_on_return : C = D - CD) 
  (h_speeds : (v > 0) ∧ (k * v > v)) 
  (h_AB_AD : AB = 2 * AD) :
  k = 2 :=
by
  sorry

end car_speeds_midpoint_condition_l128_128039


namespace minimum_value_of_ex_4e_negx_l128_128287

theorem minimum_value_of_ex_4e_negx : 
  ∃ (x : ℝ), (∀ (y : ℝ), y = Real.exp x + 4 * Real.exp (-x) → y ≥ 4) ∧ (Real.exp x + 4 * Real.exp (-x) = 4) :=
sorry

end minimum_value_of_ex_4e_negx_l128_128287


namespace stone_width_l128_128391

theorem stone_width (length_hall breadth_hall : ℝ) (num_stones length_stone : ℝ) (total_area_hall total_area_stones area_stone : ℝ)
  (h1 : length_hall = 36) (h2 : breadth_hall = 15) (h3 : num_stones = 5400) (h4 : length_stone = 2) 
  (h5 : total_area_hall = length_hall * breadth_hall * (10 * 10))
  (h6 : total_area_stones = num_stones * area_stone) 
  (h7 : area_stone = length_stone * (5 : ℝ)) 
  (h8 : total_area_stones = total_area_hall) : 
  (5 : ℝ) = 5 :=  
by sorry

end stone_width_l128_128391


namespace algebraic_identity_example_l128_128167

-- Define the variables a and b
def a : ℕ := 287
def b : ℕ := 269

-- State the problem and the expected result
theorem algebraic_identity_example :
  a * a + b * b - 2 * a * b = 324 :=
by
  -- Since the proof is not required, we insert sorry here
  sorry

end algebraic_identity_example_l128_128167


namespace find_a4_l128_128642

open Nat

def seq (a : ℕ → ℝ) := (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = (2 * a n) / (a n + 2))

theorem find_a4 (a : ℕ → ℝ) (h : seq a) : a 4 = 2 / 5 :=
  sorry

end find_a4_l128_128642


namespace molecular_weight_CaH2_correct_l128_128477

-- Define the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008

-- Define the formula to compute the molecular weight
def molecular_weight_CaH2 (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) : ℝ :=
  (1 * atomic_weight_Ca) + (2 * atomic_weight_H)

-- Theorem stating that the molecular weight of CaH2 is 42.096 g/mol
theorem molecular_weight_CaH2_correct : molecular_weight_CaH2 atomic_weight_Ca atomic_weight_H = 42.096 := 
by 
  sorry

end molecular_weight_CaH2_correct_l128_128477


namespace perfect_square_term_l128_128599

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * seq (n - 1) - seq (n - 2)

theorem perfect_square_term : ∀ n, (∃ k, seq n = k * k) ↔ n = 0 := by
  sorry

end perfect_square_term_l128_128599


namespace gasoline_price_increase_percentage_l128_128947

theorem gasoline_price_increase_percentage : 
  ∀ (highest_price lowest_price : ℝ), highest_price = 24 → lowest_price = 18 → 
  ((highest_price - lowest_price) / lowest_price) * 100 = 33.33 :=
by
  intros highest_price lowest_price h_highest h_lowest
  rw [h_highest, h_lowest]
  -- To be completed in the proof
  sorry

end gasoline_price_increase_percentage_l128_128947


namespace workman_problem_l128_128030

theorem workman_problem (x : ℝ) (h : (1 / x) + (1 / (2 * x)) = 1 / 32): x = 48 :=
sorry

end workman_problem_l128_128030


namespace factorization_theorem_l128_128134

-- Define the polynomial p(x, y)
def p (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

-- Define the condition for factorization into two linear factors
def can_be_factored (x y m n : ℝ) : Prop :=
  (p x y (m * n)) = ((x + m * y + 1) * (x + n * y + 2))

-- The main theorem proving that k = -3 is the value for factorizability
theorem factorization_theorem (k : ℝ) : (∃ m n : ℝ, can_be_factored x y m n) ↔ k = -3 := by sorry

end factorization_theorem_l128_128134


namespace admission_price_for_adults_l128_128135

theorem admission_price_for_adults (A : ℕ) (ticket_price_children : ℕ) (total_children_tickets : ℕ) 
    (total_amount : ℕ) (total_tickets : ℕ) (children_ticket_costs : ℕ) 
    (adult_tickets : ℕ) (adult_ticket_costs : ℕ) :
    ticket_price_children = 5 → 
    total_children_tickets = 21 → 
    total_amount = 201 → 
    total_tickets = 33 → 
    children_ticket_costs = 21 * 5 → 
    adult_tickets = 33 - 21 → 
    adult_ticket_costs = 201 - 21 * 5 → 
    A = (201 - 21 * 5) / (33 - 21) → 
    A = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end admission_price_for_adults_l128_128135


namespace population_increase_l128_128343

theorem population_increase (i j : ℝ) : 
  ∀ (m : ℝ), m * (1 + i / 100) * (1 + j / 100) = m * (1 + (i + j + i * j / 100) / 100) := 
by
  intro m
  sorry

end population_increase_l128_128343


namespace extra_birds_l128_128004

def num_sparrows : ℕ := 10
def num_robins : ℕ := 5
def num_bluebirds : ℕ := 3
def nests_for_sparrows : ℕ := 4
def nests_for_robins : ℕ := 2
def nests_for_bluebirds : ℕ := 2

theorem extra_birds (num_sparrows : ℕ)
                    (num_robins : ℕ)
                    (num_bluebirds : ℕ)
                    (nests_for_sparrows : ℕ)
                    (nests_for_robins : ℕ)
                    (nests_for_bluebirds : ℕ) :
    num_sparrows = 10 ∧ 
    num_robins = 5 ∧ 
    num_bluebirds = 3 ∧ 
    nests_for_sparrows = 4 ∧ 
    nests_for_robins = 2 ∧ 
    nests_for_bluebirds = 2 ->
    num_sparrows - nests_for_sparrows = 6 ∧ 
    num_robins - nests_for_robins = 3 ∧ 
    num_bluebirds - nests_for_bluebirds = 1 :=
by sorry

end extra_birds_l128_128004


namespace sum_all_products_eq_l128_128297

def group1 : List ℚ := [3/4, 3/20] -- Using 0.15 as 3/20 to work with rationals
def group2 : List ℚ := [4, 2/3]
def group3 : List ℚ := [3/5, 6/5] -- Using 1.2 as 6/5 to work with rationals

def allProducts (a b c : List ℚ) : List ℚ :=
  List.bind a (fun x =>
  List.bind b (fun y =>
  List.map (fun z => x * y * z) c))

theorem sum_all_products_eq :
  (allProducts group1 group2 group3).sum = 7.56 := by
  sorry

end sum_all_products_eq_l128_128297


namespace bernardo_receives_l128_128605

theorem bernardo_receives :
  let amount_distributed (n : ℕ) : ℕ := (n * (n + 1)) / 2
  let is_valid (n : ℕ) : Prop := amount_distributed n ≤ 1000
  let bernardo_amount (k : ℕ) : ℕ := (k * (2 + (k - 1) * 3)) / 2
  ∃ k : ℕ, is_valid (15 * 3) ∧ bernardo_amount 15 = 345 :=
sorry

end bernardo_receives_l128_128605


namespace metallic_weight_problem_l128_128602

variables {m1 m2 m3 m4 : ℝ}

theorem metallic_weight_problem
  (h_total : m1 + m2 + m3 + m4 = 35)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = (3/4) * m3)
  (h3 : m3 = (5/6) * m4) :
  m4 = 105 / 13 :=
sorry

end metallic_weight_problem_l128_128602


namespace find_length_of_room_l128_128113

noncomputable def cost_of_paving : ℝ := 21375
noncomputable def rate_per_sq_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem find_length_of_room :
  ∃ l : ℝ, l = (cost_of_paving / rate_per_sq_meter) / width_of_room ∧ l = 5 := by
  sorry

end find_length_of_room_l128_128113


namespace unit_digit_product_zero_l128_128805

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_product_zero :
  let a := 785846
  let b := 1086432
  let c := 4582735
  let d := 9783284
  let e := 5167953
  let f := 3821759
  let g := 7594683
  unit_digit (a * b * c * d * e * f * g) = 0 := 
by {
  sorry
}

end unit_digit_product_zero_l128_128805


namespace product_of_numbers_eq_120_l128_128942

theorem product_of_numbers_eq_120 (x y P : ℝ) (h1 : x + y = 23) (h2 : x^2 + y^2 = 289) (h3 : x * y = P) : P = 120 := 
sorry

end product_of_numbers_eq_120_l128_128942


namespace incorrect_conclusion_l128_128201

noncomputable def data_set : List ℕ := [4, 1, 6, 2, 9, 5, 8]
def mean_x : ℝ := 2
def mean_y : ℝ := 20
def regression_eq (x : ℝ) : ℝ := 9.1 * x + 1.8
def chi_squared_value : ℝ := 9.632
def alpha : ℝ := 0.001
def critical_value : ℝ := 10.828

theorem incorrect_conclusion : ¬(chi_squared_value ≥ critical_value) := by
  -- Insert proof here
  sorry

end incorrect_conclusion_l128_128201


namespace cricket_average_increase_l128_128576

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase_l128_128576


namespace calculate_expression_l128_128388

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1 / x^2) * (y^2 + 1 / y^2) = x^4 - y^4 := by
  sorry

end calculate_expression_l128_128388


namespace decreasing_interval_l128_128967

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 15 * x^4 - 15 * x^2

-- State the theorem
theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f' x < 0 :=
by sorry

end decreasing_interval_l128_128967


namespace find_n_tangent_eq_1234_l128_128985

theorem find_n_tangent_eq_1234 (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : Real.tan (n * Real.pi / 180) = Real.tan (1234 * Real.pi / 180)) : n = -26 := 
by 
  sorry

end find_n_tangent_eq_1234_l128_128985


namespace num_integer_solutions_quadratic_square_l128_128574

theorem num_integer_solutions_quadratic_square : 
  (∃ xs : Finset ℤ, 
    (∀ x ∈ xs, ∃ k : ℤ, (x^4 + 8*x^3 + 18*x^2 + 8*x + 64) = k^2) ∧ 
    xs.card = 2) := sorry

end num_integer_solutions_quadratic_square_l128_128574


namespace manufacturing_employees_percentage_l128_128525

theorem manufacturing_employees_percentage 
  (total_circle_deg : ℝ := 360) 
  (manufacturing_deg : ℝ := 18) 
  (sector_proportion : ∀ x y, x / y = (x/y : ℝ)) 
  (percentage : ∀ x, x * 100 = (x * 100 : ℝ)) :
  (manufacturing_deg / total_circle_deg) * 100 = 5 := 
by sorry

end manufacturing_employees_percentage_l128_128525


namespace solve_problem_l128_128700

-- Define the polynomial p(x)
noncomputable def p (x : ℂ) : ℂ := x^2 - x + 1

-- Define the root condition
def is_root (α : ℂ) : Prop := p (p (p (p α))) = 0

-- Define the expression to evaluate
noncomputable def expression (α : ℂ) : ℂ := (p α - 1) * p α * p (p α) * p (p (p α))

-- State the theorem asserting the required equality
theorem solve_problem (α : ℂ) (hα : is_root α) : expression α = -1 :=
sorry

end solve_problem_l128_128700


namespace rectangle_area_l128_128153

theorem rectangle_area (x : ℝ) (w : ℝ) (h : ℝ) (H1 : x^2 = w^2 + h^2) (H2 : h = 3 * w) : 
  (w * h = (3 * x^2) / 10) :=
by sorry

end rectangle_area_l128_128153


namespace power_mod_l128_128772

theorem power_mod (n : ℕ) : 2^99 % 7 = 1 := 
by {
  sorry
}

end power_mod_l128_128772


namespace value_of_a_when_x_is_3_root_l128_128503

theorem value_of_a_when_x_is_3_root (a : ℝ) :
  (3 ^ 2 + 3 * a + 9 = 0) -> a = -6 := by
  intros h
  sorry

end value_of_a_when_x_is_3_root_l128_128503


namespace worker_hourly_rate_l128_128513

theorem worker_hourly_rate (x : ℝ) (h1 : 8 * 0.90 = 7.20) (h2 : 42 * x + 7.20 = 32.40) : x = 0.60 :=
by
  sorry

end worker_hourly_rate_l128_128513


namespace part1_solution_part2_solution_l128_128830

-- Conditions
variables (x y : ℕ) -- Let x be the number of parcels each person sorts manually per hour,
                     -- y be the number of machines needed

def machine_efficiency : ℕ := 20 * x
def time_machines (parcels : ℕ) (machines : ℕ) : ℕ := parcels / (machines * machine_efficiency x)
def time_people (parcels : ℕ) (people : ℕ) : ℕ := parcels / (people * x)
def parcels_per_day : ℕ := 100000

-- Problem 1: Find x
axiom problem1 : (time_people 6000 20) - (time_machines 6000 5) = 4

-- Problem 2: Find y to sort 100000 parcels in a day with machines working 16 hours/day
axiom problem2 : 16 * machine_efficiency x * y ≥ parcels_per_day

-- Correct answers:
theorem part1_solution : x = 60 := by sorry
theorem part2_solution : y = 6 := by sorry

end part1_solution_part2_solution_l128_128830


namespace average_velocity_of_particle_l128_128094

theorem average_velocity_of_particle (t : ℝ) (s : ℝ → ℝ) (h_s : ∀ t, s t = t^2 + 1) :
  (s 2 - s 1) / (2 - 1) = 3 :=
by {
  sorry
}

end average_velocity_of_particle_l128_128094


namespace proof_statements_l128_128307

namespace ProofProblem

-- Definitions for each condition
def is_factor (x y : ℕ) : Prop := ∃ n : ℕ, y = n * x
def is_divisor (x y : ℕ) : Prop := is_factor x y

-- Lean 4 statement for the problem
theorem proof_statements :
  is_factor 4 20 ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (¬ is_divisor 12 75 ∧ ¬ is_divisor 12 29) ∧
  (is_divisor 11 33 ∧ ¬ is_divisor 11 64) ∧
  is_factor 9 180 :=
by
  sorry

end ProofProblem

end proof_statements_l128_128307


namespace length_of_longer_leg_of_smallest_triangle_l128_128111

theorem length_of_longer_leg_of_smallest_triangle 
  (hypotenuse_largest : ℝ) 
  (h1 : hypotenuse_largest = 10)
  (h45 : ∀ hyp, (hyp / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2) = hypotenuse_largest / 4) :
  (hypotenuse_largest / 4) = 5 / 2 := by
  sorry

end length_of_longer_leg_of_smallest_triangle_l128_128111


namespace shadow_boundary_l128_128927

theorem shadow_boundary (r : ℝ) (O P : ℝ × ℝ × ℝ) :
  r = 2 → O = (0, 0, 2) → P = (0, -2, 4) → ∀ x : ℝ, ∃ y : ℝ, y = -10 :=
by sorry

end shadow_boundary_l128_128927


namespace first_term_of_geometric_series_l128_128092

-- Define the conditions and the question
theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 180) : a = 10 :=
by sorry

end first_term_of_geometric_series_l128_128092


namespace dietitian_lunch_fraction_l128_128087

theorem dietitian_lunch_fraction
  (total_calories : ℕ)
  (recommended_calories : ℕ)
  (extra_calories : ℕ)
  (h1 : total_calories = 40)
  (h2 : recommended_calories = 25)
  (h3 : extra_calories = 5)
  : (recommended_calories + extra_calories) / total_calories = 3 / 4 :=
by
  sorry

end dietitian_lunch_fraction_l128_128087


namespace investment_of_q_is_correct_l128_128250

-- Define investments and the profit ratio
def p_investment : ℝ := 30000
def profit_ratio_p : ℝ := 2
def profit_ratio_q : ℝ := 3

-- Define q's investment as x
def q_investment : ℝ := 45000

-- The goal is to prove that q_investment is indeed 45000 given the above conditions
theorem investment_of_q_is_correct :
  (p_investment / q_investment) = (profit_ratio_p / profit_ratio_q) :=
sorry

end investment_of_q_is_correct_l128_128250


namespace sqrt_meaningful_range_l128_128285

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x = Real.sqrt (a + 2)) ↔ a ≥ -2 := 
sorry

end sqrt_meaningful_range_l128_128285


namespace weight_of_currants_l128_128859

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end weight_of_currants_l128_128859


namespace no_roots_one_and_neg_one_l128_128560

theorem no_roots_one_and_neg_one (a b : ℝ) : ¬ ((1 + a + b = 0) ∧ (-1 + a + b = 0)) :=
by
  sorry

end no_roots_one_and_neg_one_l128_128560


namespace problem_statement_l128_128860

theorem problem_statement (d : ℕ) (h1 : d > 0) (h2 : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2 * x^2 + 2 * x * y + 3 * y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by
  sorry

end problem_statement_l128_128860


namespace prize_calculations_l128_128469

-- Definitions for the conditions
def total_prizes := 50
def first_prize_unit_price := 20
def second_prize_unit_price := 14
def third_prize_unit_price := 8
def num_second_prize (x : ℕ) := 3 * x - 2
def num_third_prize (x : ℕ) := total_prizes - x - num_second_prize x
def total_cost (x : ℕ) := first_prize_unit_price * x + second_prize_unit_price * num_second_prize x + third_prize_unit_price * num_third_prize x

-- Proof problem statement
theorem prize_calculations (x : ℕ) (h : num_second_prize x = 22) : 
  num_second_prize x = 3 * x - 2 ∧ 
  num_third_prize x = 52 - 4 * x ∧ 
  total_cost x = 30 * x + 388 ∧ 
  total_cost 8 = 628 :=
by
  sorry

end prize_calculations_l128_128469


namespace proof_goats_minus_pigs_l128_128754

noncomputable def number_of_goats : ℕ := 66
noncomputable def number_of_chickens : ℕ := 2 * number_of_goats - 10
noncomputable def number_of_ducks : ℕ := (number_of_goats + number_of_chickens) / 2
noncomputable def number_of_pigs : ℕ := number_of_ducks / 3
noncomputable def number_of_rabbits : ℕ := Nat.floor (Real.sqrt (2 * number_of_ducks - number_of_pigs))
noncomputable def number_of_cows : ℕ := number_of_rabbits ^ number_of_pigs / Nat.factorial (number_of_goats / 2)

theorem proof_goats_minus_pigs : number_of_goats - number_of_pigs = 35 := by
  sorry

end proof_goats_minus_pigs_l128_128754


namespace range_of_x_of_sqrt_x_plus_3_l128_128547

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l128_128547


namespace distance_city_A_C_l128_128548

-- Define the conditions
def starts_simultaneously (A : Prop) (Eddy Freddy : Prop) := Eddy ∧ Freddy
def travels (A B C : Prop) (Eddy Freddy : Prop) := Eddy → 3 = 3 ∧ Freddy → 4 = 4
def distance_AB (A B : Prop) := 600
def speed_ratio (Eddy_speed Freddy_speed : ℝ) := Eddy_speed / Freddy_speed = 1.7391304347826086

noncomputable def distance_AC (Eddy_time Freddy_time : ℝ) (Eddy_speed Freddy_speed : ℝ) 
  := (Eddy_speed / 1.7391304347826086) * Freddy_time

theorem distance_city_A_C 
  (A B C Eddy Freddy : Prop)
  (Eddy_time Freddy_time : ℝ) 
  (Eddy_speed effective_Freddy_speed : ℝ)
  (h1 : starts_simultaneously A Eddy Freddy)
  (h2 : travels A B C Eddy Freddy)
  (h3 : distance_AB A B = 600)
  (h4 : speed_ratio Eddy_speed effective_Freddy_speed)
  (h5 : Eddy_speed = 200)
  (h6 : effective_Freddy_speed = 115)
  : distance_AC Eddy_time Freddy_time Eddy_speed effective_Freddy_speed = 460 := 
  by sorry

end distance_city_A_C_l128_128548


namespace ratio_of_abc_l128_128568

theorem ratio_of_abc {a b c : ℕ} (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
                     (h_ratio : ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x)
                     (h_mean : (a + b + c) / 3 = 42) : 
  a = 28 := 
sorry

end ratio_of_abc_l128_128568


namespace x_greater_than_y_l128_128757

theorem x_greater_than_y (x y z : ℝ) (h1 : x + y + z = 28) (h2 : 2 * x - y = 32) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 
  x > y :=
by 
  sorry

end x_greater_than_y_l128_128757


namespace ella_age_l128_128790

theorem ella_age (s t e : ℕ) (h1 : s + t + e = 36) (h2 : e - 5 = s) (h3 : t + 4 = (3 * (s + 4)) / 4) : e = 15 := by
  sorry

end ella_age_l128_128790


namespace find_a15_l128_128458

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

def arithmetic_sequence (an : ℕ → ℝ) := ∃ (a₁ d : ℝ), ∀ n, an n = a₁ + (n - 1) * d

theorem find_a15
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a9 : a 9 = 10) :
  a 15 = 16 := 
sorry

end ArithmeticSequence

end find_a15_l128_128458


namespace sixty_first_batch_is_1211_l128_128600

-- Definitions based on conditions
def total_bags : ℕ := 3000
def total_batches : ℕ := 150
def first_batch_number : ℕ := 11

-- Define the calculation of the 61st batch number
def batch_interval : ℕ := total_bags / total_batches
def sixty_first_batch_number : ℕ := first_batch_number + 60 * batch_interval

-- The statement of the proof
theorem sixty_first_batch_is_1211 : sixty_first_batch_number = 1211 := by
  sorry

end sixty_first_batch_is_1211_l128_128600


namespace required_bricks_l128_128703

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

theorem required_bricks : 
  let brick_length := 25
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 850
  let wall_width := 600
  let wall_height := 22.5
  (wall_volume wall_length wall_width wall_height) / 
  (brick_volume brick_length brick_width brick_height) = 6800 :=
by
  sorry

end required_bricks_l128_128703


namespace find_x_l128_128188

theorem find_x (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = 1 / 5^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end find_x_l128_128188


namespace value_of_frac_mul_l128_128468

theorem value_of_frac_mul (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 2 * d) :
  (a * c) / (b * d) = 8 :=
by
  sorry

end value_of_frac_mul_l128_128468


namespace oblique_projection_correct_statements_l128_128324

-- Definitions of conditions
def oblique_projection_parallel_invariant : Prop :=
  ∀ (x_parallel y_parallel : Prop), x_parallel ∧ y_parallel

def oblique_projection_length_changes : Prop :=
  ∀ (x y : ℝ), x = y / 2 ∨ x = y

def triangle_is_triangle : Prop :=
  ∀ (t : Type), t = t

def square_is_rhombus : Prop :=
  ∀ (s : Type), s = s → false

def isosceles_trapezoid_is_parallelogram : Prop :=
  ∀ (it : Type), it = it → false

def rhombus_is_rhombus : Prop :=
  ∀ (r : Type), r = r → false

-- Math proof problem
theorem oblique_projection_correct_statements :
  (triangle_is_triangle ∧ oblique_projection_parallel_invariant ∧ oblique_projection_length_changes)
  → ¬square_is_rhombus ∧ ¬isosceles_trapezoid_is_parallelogram ∧ ¬rhombus_is_rhombus :=
by 
  sorry

end oblique_projection_correct_statements_l128_128324


namespace sum_arithmetic_sequence_l128_128506

open Nat

noncomputable def arithmetic_sum (a1 d n : ℕ) : ℝ :=
  (2 * a1 + (n - 1) * d) * n / 2

theorem sum_arithmetic_sequence (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0)
    (S_m S_n : ℝ) (h4 : S_m = m / n) (h5 : S_n = n / m) 
    (a1 d : ℕ) (h6 : S_m = arithmetic_sum a1 d m) (h7 : S_n = arithmetic_sum a1 d n) 
    : arithmetic_sum a1 d (m + n) > 4 :=
by
  sorry

end sum_arithmetic_sequence_l128_128506


namespace max_value_l128_128685

theorem max_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 2) : 
  2 * x * y + 2 * y * z * Real.sqrt 3 ≤ 4 :=
sorry

end max_value_l128_128685


namespace find_total_original_cost_l128_128083

noncomputable def original_total_cost (x y z : ℝ) : ℝ :=
x + y + z

theorem find_total_original_cost (x y z : ℝ)
  (h1 : x * 1.30 = 351)
  (h2 : y * 1.25 = 275)
  (h3 : z * 1.20 = 96) :
  original_total_cost x y z = 570 :=
sorry

end find_total_original_cost_l128_128083


namespace symmetric_line_equation_l128_128662

theorem symmetric_line_equation (x y : ℝ) (h : 4 * x - 3 * y + 5 = 0):
  4 * x + 3 * y + 5 = 0 :=
sorry

end symmetric_line_equation_l128_128662


namespace students_table_tennis_not_basketball_l128_128962

variable (total_students : ℕ)
variable (students_like_basketball : ℕ)
variable (students_like_table_tennis : ℕ)
variable (students_dislike_both : ℕ)

theorem students_table_tennis_not_basketball 
  (h_total : total_students = 40)
  (h_basketball : students_like_basketball = 17)
  (h_table_tennis : students_like_table_tennis = 20)
  (h_dislike : students_dislike_both = 8) : 
  ∃ (students_table_tennis_not_basketball : ℕ), students_table_tennis_not_basketball = 15 :=
by
  sorry

end students_table_tennis_not_basketball_l128_128962


namespace find_a_if_even_function_l128_128540

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- Theorem statement
theorem find_a_if_even_function (a : ℝ) (h : is_even_function (f a)) : a = 1 :=
sorry

end find_a_if_even_function_l128_128540


namespace quadratic_rewrite_h_l128_128332

theorem quadratic_rewrite_h (a k h x : ℝ) :
  (3 * x^2 + 9 * x + 17) = a * (x - h)^2 + k ↔ h = -3/2 :=
by sorry

end quadratic_rewrite_h_l128_128332


namespace angle_same_terminal_side_l128_128386

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ (θ : ℤ), θ = -324 ∧ 
    ∀ α : ℤ, α = 36 + k * 360 → 
            ( (α % 360 = θ % 360) ∨ (α % 360 + 360 = θ % 360) ∨ (θ % 360 + 360 = α % 360)) :=
by
  sorry

end angle_same_terminal_side_l128_128386


namespace nancy_pensils_total_l128_128029

theorem nancy_pensils_total
  (initial: ℕ) 
  (mult_factor: ℕ) 
  (add_pencils: ℕ) 
  (final_total: ℕ) 
  (h1: initial = 27)
  (h2: mult_factor = 4)
  (h3: add_pencils = 45):
  final_total = initial * mult_factor + add_pencils := 
by
  sorry

end nancy_pensils_total_l128_128029


namespace relationship_log2_2_pow_03_l128_128402

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem relationship_log2_2_pow_03 : 
  log_base_2 0.3 < (0.3)^2 ∧ (0.3)^2 < 2^(0.3) :=
by
  sorry

end relationship_log2_2_pow_03_l128_128402


namespace smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l128_128382

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 3 * Real.pi / 5)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := by
  sorry

theorem axis_of_symmetry :
  ∃ k : ℤ, (∀ x, f x = f (11 * Real.pi / 20 + k * Real.pi / 2)) := by
  sorry

theorem minimum_value_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = -1 := by
  sorry

end smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l128_128382


namespace area_ratio_triangle_PQR_ABC_l128_128351

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)

theorem area_ratio_triangle_PQR_ABC {A B C P Q R : ℝ×ℝ} 
  (h1 : dist A B + dist B C + dist C A = 1)
  (h2 : dist A P + dist P Q + dist Q B + dist B C + dist C A = 1)
  (h3 : dist P Q + dist Q R + dist R P = 1)
  (h4 : P.1 <= A.1 ∧ A.1 <= Q.1 ∧ Q.1 <= B.1) :
  area P Q R / area A B C > 2 / 9 :=
by
  sorry

end area_ratio_triangle_PQR_ABC_l128_128351


namespace solve_eq1_solve_eq2_l128_128601

theorem solve_eq1 : (∃ x : ℚ, (5 * x - 1) / 4 = (3 * x + 1) / 2 - (2 - x) / 3) ↔ x = -1 / 7 :=
sorry

theorem solve_eq2 : (∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 - (2 * x + 1) / 5) ↔ x = -9 / 28 :=
sorry

end solve_eq1_solve_eq2_l128_128601


namespace travel_time_l128_128664

-- Definitions of the conditions
variables (x : ℝ) (speed_elder speed_younger : ℝ)
variables (time_elder_total time_younger_total : ℝ)

def elder_speed_condition : Prop := speed_elder = x
def younger_speed_condition : Prop := speed_younger = x - 4
def elder_distance : Prop := 42 / speed_elder + 1 = time_elder_total
def younger_distance : Prop := 42 / speed_younger + 1 / 3 = time_younger_total

-- The main theorem we want to prove
theorem travel_time : ∀ (x : ℝ), 
  elder_speed_condition x speed_elder → 
  younger_speed_condition x speed_younger → 
  elder_distance speed_elder time_elder_total → 
  younger_distance speed_younger time_younger_total → 
  time_elder_total = time_younger_total ∧ time_elder_total = (10 / 3) :=
sorry

end travel_time_l128_128664


namespace oranges_and_apples_costs_l128_128719

theorem oranges_and_apples_costs :
  ∃ (x y : ℚ), 7 * x + 5 * y = 13 ∧ 3 * x + 4 * y = 8 ∧ 37 * x + 45 * y = 93 :=
by 
  sorry

end oranges_and_apples_costs_l128_128719


namespace part1_part2_l128_128670

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Question 1: Prove that f(x) ≥ 3/4
theorem part1 (x a : ℝ) : f x a ≥ 3 / 4 := 
sorry

-- Question 2: Given f(4) < 13, find the range of a
theorem part2 (a : ℝ) (h : f 4 a < 13) : -2 < a ∧ a < 3 := 
sorry

end part1_part2_l128_128670


namespace div_by_6_for_all_k_l128_128661

def b_n_sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem div_by_6_for_all_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 → (b_n_sum_of_squares k) % 6 = 0 :=
by
  intros k hk
  sorry

end div_by_6_for_all_k_l128_128661


namespace remainder_when_squared_mod_seven_l128_128803

theorem remainder_when_squared_mod_seven
  (x y : ℤ) (k m : ℤ)
  (hx : x = 52 * k + 19)
  (hy : 3 * y = 7 * m + 5) :
  ((x + 2 * y)^2 % 7) = 1 := by
  sorry

end remainder_when_squared_mod_seven_l128_128803


namespace geom_series_min_q_l128_128928

theorem geom_series_min_q (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h_geom : ∃ k : ℝ, q = p * k ∧ r = q * k)
  (hpqr : p * q * r = 216) : q = 6 :=
sorry

end geom_series_min_q_l128_128928


namespace find_inverse_sum_l128_128041

def f (x : ℝ) : ℝ := x * |x|^2

theorem find_inverse_sum :
  (∃ x : ℝ, f x = 8) ∧ (∃ y : ℝ, f y = -64) → 
  (∃ a b : ℝ, f a = 8 ∧ f b = -64 ∧ a + b = 6) :=
sorry

end find_inverse_sum_l128_128041


namespace ab_gt_ac_l128_128725

variables {a b c : ℝ}

theorem ab_gt_ac (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end ab_gt_ac_l128_128725


namespace pizza_combinations_l128_128675

/-- The number of unique pizzas that can be made with exactly 5 toppings from a selection of 8 is 56. -/
theorem pizza_combinations : (Nat.choose 8 5) = 56 := by
  sorry

end pizza_combinations_l128_128675


namespace average_stoppage_time_l128_128736

def bus_a_speed_excluding_stoppages := 54 -- kmph
def bus_a_speed_including_stoppages := 45 -- kmph

def bus_b_speed_excluding_stoppages := 60 -- kmph
def bus_b_speed_including_stoppages := 50 -- kmph

def bus_c_speed_excluding_stoppages := 72 -- kmph
def bus_c_speed_including_stoppages := 60 -- kmph

theorem average_stoppage_time :
  (bus_a_speed_excluding_stoppages - bus_a_speed_including_stoppages) / bus_a_speed_excluding_stoppages * 60
  + (bus_b_speed_excluding_stoppages - bus_b_speed_including_stoppages) / bus_b_speed_excluding_stoppages * 60
  + (bus_c_speed_excluding_stoppages - bus_c_speed_including_stoppages) / bus_c_speed_excluding_stoppages * 60
  = 30 / 3 :=
  by sorry

end average_stoppage_time_l128_128736


namespace total_cups_needed_l128_128594

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end total_cups_needed_l128_128594


namespace max_x_values_l128_128049

noncomputable def y (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

theorem max_x_values :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} = {x : ℝ | y x = y (x)} :=
sorry

end max_x_values_l128_128049


namespace avg_rate_first_half_l128_128445

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l128_128445


namespace additional_vegetables_can_be_planted_l128_128103

-- Defines the garden's initial conditions.
def tomatoes_kinds := 3
def tomatoes_each := 5
def cucumbers_kinds := 5
def cucumbers_each := 4
def potatoes := 30
def rows := 10
def spaces_per_row := 15

-- The proof statement.
theorem additional_vegetables_can_be_planted (total_tomatoes : ℕ := tomatoes_kinds * tomatoes_each)
                                              (total_cucumbers : ℕ := cucumbers_kinds * cucumbers_each)
                                              (total_potatoes : ℕ := potatoes)
                                              (total_spaces : ℕ := rows * spaces_per_row) :
  total_spaces - (total_tomatoes + total_cucumbers + total_potatoes) = 85 := 
by 
  sorry

end additional_vegetables_can_be_planted_l128_128103


namespace check_perfect_squares_l128_128449

-- Define the prime factorizations of each option
def optionA := 3^3 * 4^5 * 7^7
def optionB := 3^4 * 4^4 * 7^6
def optionC := 3^6 * 4^3 * 7^8
def optionD := 3^5 * 4^6 * 7^5
def optionE := 3^4 * 4^6 * 7^7

-- Definition of a perfect square (all exponents in prime factorization are even)
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p : ℕ, (p ^ 2 ∣ n) -> (p ∣ n)

-- The Lean statement asserting which options are perfect squares
theorem check_perfect_squares :
  (is_perfect_square optionB) ∧ (is_perfect_square optionC) ∧
  ¬(is_perfect_square optionA) ∧ ¬(is_perfect_square optionD) ∧ ¬(is_perfect_square optionE) :=
by sorry

end check_perfect_squares_l128_128449


namespace papers_left_after_giving_away_l128_128181

variable (x : ℕ)

-- Given conditions:
def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41
def total_initial_sheets := sheets_in_desk + sheets_in_backpack

-- Prove that Maria has 91 - x sheets left after giving away x sheets
theorem papers_left_after_giving_away (h : total_initial_sheets = 91) : 
  ∀ d b : ℕ, d = sheets_in_desk → b = sheets_in_backpack → 91 - x = total_initial_sheets - x :=
by
  sorry

end papers_left_after_giving_away_l128_128181


namespace f_at_1_l128_128618

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + (5 : ℝ) * x
  else if x = 2 then 6
  else  - (x^2 + (5 : ℝ) * x)

theorem f_at_1 : f 1 = 4 :=
by {
  sorry
}

end f_at_1_l128_128618


namespace Elle_practice_time_l128_128557

variable (x : ℕ)

theorem Elle_practice_time : 
  (5 * x) + (3 * x) = 240 → x = 30 :=
by
  intro h
  sorry

end Elle_practice_time_l128_128557


namespace arccos_half_eq_pi_div_three_l128_128968

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l128_128968


namespace fraction_r_over_b_l128_128823

-- Definition of the conditions
def initial_expression (k : ℝ) : ℝ := 8 * k^2 - 12 * k + 20

-- Proposition statement
theorem fraction_r_over_b : ∃ a b r : ℝ, 
  (∀ k : ℝ, initial_expression k = a * (k + b)^2 + r) ∧ 
  r / b = -47.33 :=
sorry

end fraction_r_over_b_l128_128823


namespace fence_perimeter_l128_128644

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end fence_perimeter_l128_128644


namespace volume_of_circumscribed_sphere_l128_128925

theorem volume_of_circumscribed_sphere (vol_cube : ℝ) (h : vol_cube = 8) :
  ∃ (vol_sphere : ℝ), vol_sphere = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_circumscribed_sphere_l128_128925


namespace hide_and_seek_problem_l128_128283

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l128_128283


namespace evaluate_expression_l128_128237

theorem evaluate_expression (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l128_128237


namespace xiaolong_average_speed_l128_128982

noncomputable def averageSpeed (dist_home_store : ℕ) (time_home_store : ℕ) 
                               (speed_store_playground : ℕ) (time_store_playground : ℕ) 
                               (dist_playground_school : ℕ) (speed_playground_school : ℕ) 
                               (total_time : ℕ) : ℕ :=
  let dist_store_playground := speed_store_playground * time_store_playground
  let time_playground_school := dist_playground_school / speed_playground_school
  let total_distance := dist_home_store + dist_store_playground + dist_playground_school
  total_distance / total_time

theorem xiaolong_average_speed :
  averageSpeed 500 7 80 8 300 60 20 = 72 := by
  sorry

end xiaolong_average_speed_l128_128982


namespace mario_garden_total_blossoms_l128_128371

def hibiscus_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

def rose_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

theorem mario_garden_total_blossoms :
  let weeks := 2
  let hibiscus1 := hibiscus_growth 2 3 weeks
  let hibiscus2 := hibiscus_growth (2 * 2) 4 weeks
  let hibiscus3 := hibiscus_growth (4 * (2 * 2)) 5 weeks
  let rose1 := rose_growth 3 2 weeks
  let rose2 := rose_growth 5 3 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 = 64 := 
by
  sorry

end mario_garden_total_blossoms_l128_128371


namespace roots_of_polynomial_l128_128075

theorem roots_of_polynomial :
  {r : ℝ | (10 * r^4 - 55 * r^3 + 96 * r^2 - 55 * r + 10 = 0)} = {2, 1, 1 / 2} :=
sorry

end roots_of_polynomial_l128_128075


namespace macey_saving_weeks_l128_128785

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l128_128785


namespace people_in_room_l128_128474

/-- 
   Problem: Five-sixths of the people in a room are seated in five-sixths of the chairs.
   The rest of the people are standing. If there are 10 empty chairs, 
   prove that there are 60 people in the room.
-/
theorem people_in_room (people chairs : ℕ) 
  (h_condition1 : 5 / 6 * people = 5 / 6 * chairs) 
  (h_condition2 : chairs = 60) :
  people = 60 :=
by
  sorry

end people_in_room_l128_128474


namespace trig_cos_sum_l128_128439

open Real

theorem trig_cos_sum :
  cos (37 * (π / 180)) * cos (23 * (π / 180)) - sin (37 * (π / 180)) * sin (23 * (π / 180)) = 1 / 2 :=
by
  sorry

end trig_cos_sum_l128_128439


namespace statement_B_false_l128_128869

def f (x : ℝ) : ℝ := 3 * x

def diamondsuit (x y : ℝ) : ℝ := abs (f x - f y)

theorem statement_B_false (x y : ℝ) : 3 * diamondsuit x y ≠ diamondsuit (3 * x) (3 * y) :=
by
  sorry

end statement_B_false_l128_128869


namespace determine_C_for_identity_l128_128690

theorem determine_C_for_identity :
  (∀ (x : ℝ), (1/2 * (Real.sin x)^2 + C = -1/4 * Real.cos (2 * x))) → C = -1/4 :=
by
  sorry

end determine_C_for_identity_l128_128690


namespace teachers_no_conditions_percentage_l128_128222

theorem teachers_no_conditions_percentage :
  let total_teachers := 150
  let high_blood_pressure := 90
  let heart_trouble := 60
  let both_hbp_ht := 30
  let diabetes := 10
  let both_diabetes_ht := 5
  let both_diabetes_hbp := 8
  let all_three := 3

  let only_hbp := high_blood_pressure - both_hbp_ht - both_diabetes_hbp - all_three
  let only_ht := heart_trouble - both_hbp_ht - both_diabetes_ht - all_three
  let only_diabetes := diabetes - both_diabetes_hbp - both_diabetes_ht - all_three
  let both_hbp_ht_only := both_hbp_ht - all_three
  let both_hbp_diabetes_only := both_diabetes_hbp - all_three
  let both_ht_diabetes_only := both_diabetes_ht - all_three
  let any_condition := only_hbp + only_ht + only_diabetes + both_hbp_ht_only + both_hbp_diabetes_only + both_ht_diabetes_only + all_three
  let no_conditions := total_teachers - any_condition

  (no_conditions / total_teachers * 100) = 28 :=
by
  sorry

end teachers_no_conditions_percentage_l128_128222


namespace total_cost_correct_l128_128077

-- Define the costs for each day
def day1_rate : ℝ := 150
def day1_miles_cost : ℝ := 0.50 * 620
def gps_service_cost : ℝ := 10
def day1_total_cost : ℝ := day1_rate + day1_miles_cost + gps_service_cost

def day2_rate : ℝ := 100
def day2_miles_cost : ℝ := 0.40 * 744
def day2_total_cost : ℝ := day2_rate + day2_miles_cost + gps_service_cost

def day3_rate : ℝ := 75
def day3_miles_cost : ℝ := 0.30 * 510
def day3_total_cost : ℝ := day3_rate + day3_miles_cost + gps_service_cost

-- Define the total cost
def total_cost : ℝ := day1_total_cost + day2_total_cost + day3_total_cost

-- Prove that the total cost is equal to the calculated value
theorem total_cost_correct : total_cost = 1115.60 :=
by
  -- This is where the proof would go, but we leave it out for now
  sorry

end total_cost_correct_l128_128077


namespace mike_picked_peaches_l128_128677

def initial_peaches : ℕ := 34
def total_peaches : ℕ := 86

theorem mike_picked_peaches : total_peaches - initial_peaches = 52 :=
by
  sorry

end mike_picked_peaches_l128_128677


namespace division_of_polynomials_l128_128151

theorem division_of_polynomials (a b : ℝ) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by
  sorry

end division_of_polynomials_l128_128151


namespace minimum_throws_to_ensure_same_sum_twice_l128_128708

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l128_128708


namespace number_of_arrangements_SEES_l128_128447

theorem number_of_arrangements_SEES : 
  ∃ n : ℕ, 
    (∀ (total_letters E S : ℕ), 
      total_letters = 4 ∧ E = 2 ∧ S = 2 → 
      n = Nat.factorial total_letters / (Nat.factorial E * Nat.factorial S)) → 
    n = 6 := 
by 
  sorry

end number_of_arrangements_SEES_l128_128447


namespace find_prime_p_l128_128554

theorem find_prime_p :
  ∃ p : ℕ, Prime p ∧ (∃ a b : ℤ, p = 5 ∧ 1 < p ∧ p ≤ 11 ∧ (a^2 + p * a - 720 * p = 0) ∧ (b^2 - p * b + 720 * p = 0)) :=
sorry

end find_prime_p_l128_128554


namespace age_difference_l128_128453

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : C + 11 = A :=
by {
  sorry
}

end age_difference_l128_128453


namespace number_ordering_l128_128879

theorem number_ordering : (10^5 < 2^20) ∧ (2^20 < 5^10) :=
by {
  -- We place the proof steps here
  sorry
}

end number_ordering_l128_128879


namespace difference_between_extremes_l128_128853

/-- Define the structure of a 3-digit integer and its digits. -/
structure ThreeDigitInteger where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  val : ℕ := 100 * hundreds + 10 * tens + units

/-- Define the problem conditions. -/
def satisfiesConditions (x : ThreeDigitInteger) : Prop :=
  x.hundreds > 0 ∧
  4 * x.hundreds = 2 * x.tens ∧
  2 * x.tens = x.units

/-- Given conditions prove the difference between the two greatest possible values of x is 124. -/
theorem difference_between_extremes :
  ∃ (x₁ x₂ : ThreeDigitInteger), 
    satisfiesConditions x₁ ∧ satisfiesConditions x₂ ∧
    (x₁.val = 248 ∧ x₂.val = 124 ∧ (x₁.val - x₂.val = 124)) :=
sorry

end difference_between_extremes_l128_128853


namespace min_value_of_expression_l128_128478

theorem min_value_of_expression
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq : x * (x + y) = 5 * x + y) : 2 * x + y ≥ 9 :=
sorry

end min_value_of_expression_l128_128478


namespace find_takeoff_run_distance_l128_128454

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end find_takeoff_run_distance_l128_128454


namespace figure_total_area_l128_128707

theorem figure_total_area (a : ℝ) (h : a^2 - (3/2 * a^2) = 0.6) : 
  5 * a^2 = 6 :=
by
  sorry

end figure_total_area_l128_128707


namespace positive_difference_perimeters_l128_128671

theorem positive_difference_perimeters :
  let w1 := 3
  let h1 := 2
  let w2 := 6
  let h2 := 1
  let P1 := 2 * (w1 + h1)
  let P2 := 2 * (w2 + h2)
  P2 - P1 = 4 := by
  sorry

end positive_difference_perimeters_l128_128671


namespace tan_of_neg_23_over_3_pi_l128_128961

theorem tan_of_neg_23_over_3_pi : (Real.tan (- 23 / 3 * Real.pi) = Real.sqrt 3) :=
by
  sorry

end tan_of_neg_23_over_3_pi_l128_128961


namespace chess_tournament_participants_l128_128231

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 := 
by sorry

end chess_tournament_participants_l128_128231


namespace range_of_a_l128_128963

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 8 → (a * (n^2) + n + 5) > (a * ((n + 1)^2) + (n + 1) + 5)) → 
  (a * (1^2) + 1 + 5 < a * (2^2) + 2 + 5) →
  (a * (2^2) + 2 + 5 < a * (3^2) + 3 + 5) →
  (a * (3^2) + 3 + 5 < a * (4^2) + 4 + 5) →
  (- (1 / 7) < a ∧ a < - (1 / 17)) :=
by
  sorry

end range_of_a_l128_128963


namespace range_of_function_l128_128709

theorem range_of_function :
  ∃ (S : Set ℝ), (∀ x : ℝ, (1 / 2)^(x^2 - 2) ∈ S) ∧ S = Set.Ioc 0 4 := by
  sorry

end range_of_function_l128_128709


namespace vectors_are_perpendicular_l128_128629

def vector_a : ℝ × ℝ := (-5, 6)
def vector_b : ℝ × ℝ := (6, 5)

theorem vectors_are_perpendicular :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 0 :=
by
  sorry

end vectors_are_perpendicular_l128_128629


namespace line_point_t_l128_128452

theorem line_point_t (t : ℝ) : 
  (∃ t, (0, 3) = (0, 3) ∧ (-8, 0) = (-8, 0) ∧ (5 - 3) / t = 3 / 8) → (t = 16 / 3) :=
by
  sorry

end line_point_t_l128_128452


namespace coins_remainder_l128_128144

theorem coins_remainder 
  (n : ℕ)
  (h₁ : n % 8 = 6)
  (h₂ : n % 7 = 2)
  (h₃ : n = 30) :
  n % 9 = 3 :=
sorry

end coins_remainder_l128_128144


namespace min_value_of_f_l128_128529

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l128_128529


namespace pos_divisors_180_l128_128930

theorem pos_divisors_180 : 
  (∃ a b c : ℕ, 180 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 1) →
  (∃ n : ℕ, n = 18 ∧ n = (a + 1) * (b + 1) * (c + 1)) := by
  sorry

end pos_divisors_180_l128_128930


namespace contrapositive_example_l128_128350

theorem contrapositive_example (x : ℝ) : 
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end contrapositive_example_l128_128350


namespace number_of_small_jars_l128_128616

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 :=
by
  sorry

end number_of_small_jars_l128_128616


namespace eval_expression_l128_128411

theorem eval_expression (a b : ℤ) (h₁ : a = 4) (h₂ : b = -2) : -a - b^2 + a*b + a^2 = 0 := by
  sorry

end eval_expression_l128_128411


namespace set_M_properties_l128_128858

def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem set_M_properties :
  M = { x | 0 < x ∧ x < 2 } ∧
  (∀ a, a ∈ M → 
    ((0 < a ∧ a < 1) → (a^2 - a + 1 < 1 / a)) ∧
    (a = 1 → (a^2 - a + 1 = 1 / a)) ∧
    ((1 < a ∧ a < 2) → (a^2 - a + 1 > 1 / a))) := 
by
  sorry

end set_M_properties_l128_128858


namespace solve_for_a_l128_128122

theorem solve_for_a (x a : ℝ) (h1 : x + 2 * a - 6 = 0) (h2 : x = -2) : a = 4 :=
by
  sorry

end solve_for_a_l128_128122


namespace M_inter_N_M_union_not_N_l128_128088

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 0}

theorem M_inter_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 3} := 
sorry

theorem M_union_not_N :
  M ∪ {x | x ≤ 0} = {x | x ≤ 3} := 
sorry

end M_inter_N_M_union_not_N_l128_128088


namespace sum_of_first_9_terms_l128_128791

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- a_n is the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of first n terms of the arithmetic sequence
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Hypotheses
axiom h1 : 2 * a 8 = 6 + a 11
axiom h2 : arithmetic_seq a
axiom h3 : sum_seq S a

-- The theorem we want to prove
theorem sum_of_first_9_terms : S 9 = 54 :=
sorry

end sum_of_first_9_terms_l128_128791


namespace workload_increase_l128_128176

theorem workload_increase (a b c d p : ℕ) (h : p ≠ 0) :
  let total_workload := a + b + c + d
  let workload_per_worker := total_workload / p
  let absent_workers := p / 4
  let remaining_workers := p - absent_workers
  let workload_per_remaining_worker := total_workload / (3 * p / 4)
  workload_per_remaining_worker = (a + b + c + d) * 4 / (3 * p) :=
by
  sorry

end workload_increase_l128_128176


namespace natural_number_x_l128_128002

theorem natural_number_x (x : ℕ) (A : ℕ → ℕ) (h : 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2) : x = 4 :=
sorry

end natural_number_x_l128_128002
