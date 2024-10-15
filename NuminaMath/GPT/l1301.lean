import Mathlib

namespace NUMINAMATH_GPT_bags_of_cookies_l1301_130151

theorem bags_of_cookies (bags : ℕ) (cookies_total candies_total : ℕ) 
    (h1 : bags = 14) (h2 : cookies_total = 28) (h3 : candies_total = 86) :
    bags = 14 :=
by
  exact h1

end NUMINAMATH_GPT_bags_of_cookies_l1301_130151


namespace NUMINAMATH_GPT_profit_calculation_l1301_130159

open Nat

-- Define the conditions 
def cost_of_actors : Nat := 1200 
def number_of_people : Nat := 50
def cost_per_person_food : Nat := 3
def sale_price : Nat := 10000

-- Define the derived costs
def total_food_cost : Nat := number_of_people * cost_per_person_food
def total_combined_cost : Nat := cost_of_actors + total_food_cost
def equipment_rental_cost : Nat := 2 * total_combined_cost
def total_cost : Nat := cost_of_actors + total_food_cost + equipment_rental_cost
def expected_profit : Nat := 5950 

-- Define the profit calculation
def profit : Nat := sale_price - total_cost 

-- The theorem to be proved
theorem profit_calculation : profit = expected_profit := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_profit_calculation_l1301_130159


namespace NUMINAMATH_GPT_minor_premise_of_syllogism_l1301_130154

theorem minor_premise_of_syllogism (P Q : Prop)
  (h1 : ¬ (P ∧ ¬ Q))
  (h2 : Q) :
  Q :=
by
  sorry

end NUMINAMATH_GPT_minor_premise_of_syllogism_l1301_130154


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1301_130112

variables (x y : ℝ)

theorem percent_of_x_is_y (h : 0.30 * (x - y) = 0.20 * (x + y)) : y = 0.20 * x :=
by sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1301_130112


namespace NUMINAMATH_GPT_complex_frac_eq_l1301_130188

theorem complex_frac_eq (a b : ℝ) (i : ℂ) (h : i^2 = -1)
  (h1 : (1 - i) / (1 + i) = a + b * i) : a - b = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_frac_eq_l1301_130188


namespace NUMINAMATH_GPT_largest_digit_7182N_divisible_by_6_l1301_130141

noncomputable def largest_digit_divisible_by_6 : ℕ := 6

theorem largest_digit_7182N_divisible_by_6 (N : ℕ) : 
  (N % 2 = 0) ∧ ((18 + N) % 3 = 0) ↔ (N ≤ 9) ∧ (N = 6) :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_7182N_divisible_by_6_l1301_130141


namespace NUMINAMATH_GPT_three_digit_numbers_with_square_ending_in_them_l1301_130155

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_with_square_ending_in_them_l1301_130155


namespace NUMINAMATH_GPT_identify_conic_section_is_hyperbola_l1301_130105

theorem identify_conic_section_is_hyperbola :
  ∀ x y : ℝ, x^2 - 16 * y^2 - 10 * x + 4 * y + 36 = 0 →
  (∃ a b h c d k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = 0 ∧ (x - c)^2 / a^2 - (y - d)^2 / b^2 = k) :=
by
  sorry

end NUMINAMATH_GPT_identify_conic_section_is_hyperbola_l1301_130105


namespace NUMINAMATH_GPT_total_ounces_of_coffee_l1301_130127

/-
Defining the given conditions
-/
def num_packages_10_oz : Nat := 5
def num_packages_5_oz : Nat := num_packages_10_oz + 2
def ounces_per_10_oz_pkg : Nat := 10
def ounces_per_5_oz_pkg : Nat := 5

/-
Statement to prove the total ounces of coffee
-/
theorem total_ounces_of_coffee :
  (num_packages_10_oz * ounces_per_10_oz_pkg + num_packages_5_oz * ounces_per_5_oz_pkg) = 85 := by
  sorry

end NUMINAMATH_GPT_total_ounces_of_coffee_l1301_130127


namespace NUMINAMATH_GPT_number_of_subsets_of_M_l1301_130147

def M : Set ℝ := { x | x^2 - 2 * x + 1 = 0 }

theorem number_of_subsets_of_M : M = {1} → ∃ n, n = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_M_l1301_130147


namespace NUMINAMATH_GPT_max_perimeter_triangle_l1301_130124

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end NUMINAMATH_GPT_max_perimeter_triangle_l1301_130124


namespace NUMINAMATH_GPT_sum_of_ages_l1301_130138

variable (S F : ℕ)

theorem sum_of_ages (h1 : F - 18 = 3 * (S - 18)) (h2 : F = 2 * S) : S + F = 108 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1301_130138


namespace NUMINAMATH_GPT_midpoint_one_seventh_one_ninth_l1301_130128

theorem midpoint_one_seventh_one_ninth : 
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  (a + b) / 2 = 8 / 63 := 
by
  sorry

end NUMINAMATH_GPT_midpoint_one_seventh_one_ninth_l1301_130128


namespace NUMINAMATH_GPT_find_annual_interest_rate_l1301_130110

theorem find_annual_interest_rate (A P : ℝ) (n t : ℕ) (r : ℝ) :
  A = P * (1 + r / n)^(n * t) →
  A = 5292 →
  P = 4800 →
  n = 1 →
  t = 2 →
  r = 0.05 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l1301_130110


namespace NUMINAMATH_GPT_original_players_count_l1301_130158

theorem original_players_count (n : ℕ) (W : ℕ) :
  (W = n * 103) →
  ((W + 110 + 60) = (n + 2) * 99) →
  n = 7 :=
by sorry

end NUMINAMATH_GPT_original_players_count_l1301_130158


namespace NUMINAMATH_GPT_arithmetic_series_sum_l1301_130197

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 50
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  S = 442 := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l1301_130197


namespace NUMINAMATH_GPT_equation_of_the_line_l1301_130136

noncomputable def line_equation (t : ℝ) : (ℝ × ℝ) := (3 * t + 6, 5 * t - 7)

theorem equation_of_the_line : ∃ m b : ℝ, (∀ t : ℝ, ∃ (x y : ℝ), line_equation t = (x, y) ∧ y = m * x + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_the_line_l1301_130136


namespace NUMINAMATH_GPT_find_OC_l1301_130135

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

def OA (A : Point) : ℝ := sqrt (A.x^2 + A.y^2)
def OB (B : Point) : ℝ := sqrt (B.x^2 + B.y^2)
def OD (D : Point) : ℝ := sqrt (D.x^2 + D.y^2)
def ratio_of_lengths (A B : Point) : ℝ := OA A / OB B

def find_D (A B : Point) : Point :=
  let ratio := ratio_of_lengths A B
  { x := (A.x + ratio * B.x) / (1 + ratio),
    y := (A.y + ratio * B.y) / (1 + ratio) }

-- Given conditions
def A : Point := ⟨0, 1⟩
def B : Point := ⟨-3, 4⟩
def C_magnitude : ℝ := 2

-- Goal to prove
theorem find_OC : Point :=
  let D := find_D A B
  let D_length := OD D
  let scale := C_magnitude / D_length
  { x := D.x * scale,
    y := D.y * scale }

example : find_OC = ⟨-sqrt 10 / 5, 3 * sqrt 10 / 5⟩ := by
  sorry

end NUMINAMATH_GPT_find_OC_l1301_130135


namespace NUMINAMATH_GPT_values_of_a_l1301_130125

open Set

noncomputable def A : Set ℝ := { x | x^2 - 2*x - 3 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else { x | a * x = 1 }

theorem values_of_a (a : ℝ) : (B a ⊆ A) ↔ (a = -1 ∨ a = 0 ∨ a = 1/3) :=
by 
  sorry

end NUMINAMATH_GPT_values_of_a_l1301_130125


namespace NUMINAMATH_GPT_calculate_expression_l1301_130162

theorem calculate_expression : 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1301_130162


namespace NUMINAMATH_GPT_number_of_ways_to_purchase_magazines_l1301_130178

/-
Conditions:
1. The bookstore sells 11 different magazines.
2. 8 of these magazines are priced at 2 yuan each.
3. 3 of these magazines are priced at 1 yuan each.
4. Xiao Zhang has 10 yuan to buy magazines.
5. Xiao Zhang can buy at most one copy of each magazine.
6. Xiao Zhang wants to spend all 10 yuan.

Question:
The number of different ways Xiao Zhang can purchase magazines with 10 yuan.

Answer:
266
-/

theorem number_of_ways_to_purchase_magazines : ∀ (magazines_1_yuan magazines_2_yuan : ℕ),
  magazines_1_yuan = 3 →
  magazines_2_yuan = 8 →
  (∃ (ways : ℕ), ways = 266) :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_ways_to_purchase_magazines_l1301_130178


namespace NUMINAMATH_GPT_car_selling_price_l1301_130132

def car_material_cost : ℕ := 100
def car_production_per_month : ℕ := 4
def motorcycle_material_cost : ℕ := 250
def motorcycles_sold_per_month : ℕ := 8
def motorcycle_selling_price : ℤ := 50
def additional_motorcycle_profit : ℤ := 50

theorem car_selling_price (x : ℤ) :
  (motorcycles_sold_per_month * motorcycle_selling_price - motorcycle_material_cost)
  = (car_production_per_month * x - car_material_cost + additional_motorcycle_profit) →
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_car_selling_price_l1301_130132


namespace NUMINAMATH_GPT_saving_percentage_l1301_130164

variable (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ)

-- Conditions from problem
def condition1 := saved_percent_last_year = 0.06
def condition2 := made_more = 1.20
def condition3 := saved_percent_this_year = 0.05 * made_more

-- The problem statement to prove
theorem saving_percentage (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ) :
  condition1 saved_percent_last_year →
  condition2 made_more →
  condition3 saved_percent_this_year made_more →
  (saved_percent_this_year * made_more = saved_percent_last_year * S * 1) :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_saving_percentage_l1301_130164


namespace NUMINAMATH_GPT_find_number_l1301_130192

theorem find_number (x : ℝ) (h : 4 * (x - 220) = 320) : (5 * x) / 3 = 500 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1301_130192


namespace NUMINAMATH_GPT_books_read_l1301_130174

theorem books_read (total_books remaining_books read_books : ℕ)
  (h_total : total_books = 14)
  (h_remaining : remaining_books = 6)
  (h_eq : read_books = total_books - remaining_books) : read_books = 8 := 
by 
  sorry

end NUMINAMATH_GPT_books_read_l1301_130174


namespace NUMINAMATH_GPT_larger_number_of_hcf_lcm_is_322_l1301_130149

theorem larger_number_of_hcf_lcm_is_322
  (A B : ℕ)
  (hcf: ℕ := 23)
  (factor1 : ℕ := 13)
  (factor2 : ℕ := 14)
  (hcf_condition : ∀ d, d ∣ A → d ∣ B → d ≤ hcf)
  (lcm_condition : ∀ m n, m * n = A * B → m = factor1 * hcf ∨ m = factor2 * hcf) :
  max A B = 322 :=
by sorry

end NUMINAMATH_GPT_larger_number_of_hcf_lcm_is_322_l1301_130149


namespace NUMINAMATH_GPT_trigonometric_identity_l1301_130199

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.sin (3 * Real.pi / 2 - α) = -3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1301_130199


namespace NUMINAMATH_GPT_effect_of_dimension_changes_on_area_l1301_130176

variable {L B : ℝ}  -- Original length and breadth

def original_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.15 * L

def new_breadth (B : ℝ) : ℝ := 0.90 * B

def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem effect_of_dimension_changes_on_area (L B : ℝ) :
  new_area L B = 1.035 * original_area L B :=
by
  sorry

end NUMINAMATH_GPT_effect_of_dimension_changes_on_area_l1301_130176


namespace NUMINAMATH_GPT_import_tax_percentage_l1301_130191

theorem import_tax_percentage
  (total_value : ℝ)
  (non_taxable_portion : ℝ)
  (import_tax_paid : ℝ)
  (h_total_value : total_value = 2610)
  (h_non_taxable_portion : non_taxable_portion = 1000)
  (h_import_tax_paid : import_tax_paid = 112.70) :
  ((import_tax_paid / (total_value - non_taxable_portion)) * 100) = 7 :=
by
  sorry

end NUMINAMATH_GPT_import_tax_percentage_l1301_130191


namespace NUMINAMATH_GPT_cotton_needed_l1301_130185

noncomputable def feet_of_cotton_per_teeshirt := 4
noncomputable def number_of_teeshirts := 15

theorem cotton_needed : feet_of_cotton_per_teeshirt * number_of_teeshirts = 60 := 
by 
  sorry

end NUMINAMATH_GPT_cotton_needed_l1301_130185


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_l1301_130182

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_l1301_130182


namespace NUMINAMATH_GPT_solve_equation_integers_l1301_130117

theorem solve_equation_integers :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2 ∧
  (x = 2 ∧ y = 4 ∧ z = 15 ∨
   x = 2 ∧ y = 5 ∧ z = 9 ∨
   x = 2 ∧ y = 6 ∧ z = 7 ∨
   x = 3 ∧ y = 4 ∧ z = 5 ∨
   x = 3 ∧ y = 3 ∧ z = 8 ∨
   x = 2 ∧ y = 15 ∧ z = 4 ∨
   x = 2 ∧ y = 9 ∧ z = 5 ∨
   x = 2 ∧ y = 7 ∧ z = 6 ∨
   x = 3 ∧ y = 5 ∧ z = 4 ∨
   x = 3 ∧ y = 8 ∧ z = 3) ∧
  (y = 2 ∧ x = 4 ∧ z = 15 ∨
   y = 2 ∧ x = 5 ∧ z = 9 ∨
   y = 2 ∧ x = 6 ∧ z = 7 ∨
   y = 3 ∧ x = 4 ∧ z = 5 ∨
   y = 3 ∧ x = 3 ∧ z = 8 ∨
   y = 15 ∧ x = 4 ∧ z = 2 ∨
   y = 9 ∧ x = 5 ∧ z = 2 ∨
   y = 7 ∧ x = 6 ∧ z = 2 ∨
   y = 5 ∧ x = 4 ∧ z = 3 ∨
   y = 8 ∧ x = 3 ∧ z = 3) ∧
  (z = 2 ∧ x = 4 ∧ y = 15 ∨
   z = 2 ∧ x = 5 ∧ y = 9 ∨
   z = 2 ∧ x = 6 ∧ y = 7 ∨
   z = 3 ∧ x = 4 ∧ y = 5 ∨
   z = 3 ∧ x = 3 ∧ y = 8 ∨
   z = 15 ∧ x = 4 ∧ y = 2 ∨
   z = 9 ∧ x = 5 ∧ y = 2 ∨
   z = 7 ∧ x = 6 ∧ y = 2 ∨
   z = 5 ∧ x = 4 ∧ y = 3 ∨
   z = 8 ∧ x = 3 ∧ y = 3)
:= sorry

end NUMINAMATH_GPT_solve_equation_integers_l1301_130117


namespace NUMINAMATH_GPT_number_of_books_about_trains_l1301_130103

theorem number_of_books_about_trains
  (books_animals : ℕ)
  (books_outer_space : ℕ)
  (book_cost : ℕ)
  (total_spent : ℕ)
  (T : ℕ)
  (hyp1 : books_animals = 8)
  (hyp2 : books_outer_space = 6)
  (hyp3 : book_cost = 6)
  (hyp4 : total_spent = 102)
  (hyp5 : total_spent = (books_animals + books_outer_space + T) * book_cost)
  : T = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_books_about_trains_l1301_130103


namespace NUMINAMATH_GPT_price_per_unit_max_profit_l1301_130194

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end NUMINAMATH_GPT_price_per_unit_max_profit_l1301_130194


namespace NUMINAMATH_GPT_ratio_of_b_to_sum_a_c_l1301_130109

theorem ratio_of_b_to_sum_a_c (a b c : ℕ) (h1 : a + b + c = 60) (h2 : a = 1/3 * (b + c)) (h3 : c = 35) : b = 1/5 * (a + c) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_b_to_sum_a_c_l1301_130109


namespace NUMINAMATH_GPT_fox_cub_distribution_l1301_130115

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end NUMINAMATH_GPT_fox_cub_distribution_l1301_130115


namespace NUMINAMATH_GPT_license_plate_count_l1301_130190

theorem license_plate_count : 
  let consonants := 20
  let vowels := 6
  let digits := 10
  4 * consonants * vowels * consonants * digits = 24000 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_count_l1301_130190


namespace NUMINAMATH_GPT_larger_solution_quadratic_l1301_130195

theorem larger_solution_quadratic : 
  ∀ x1 x2 : ℝ, (x^2 - 13 * x - 48 = 0) → x1 ≠ x2 → (x1 = 16 ∨ x2 = 16) → max x1 x2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_larger_solution_quadratic_l1301_130195


namespace NUMINAMATH_GPT_probability_of_two_pairs_of_same_value_is_correct_l1301_130140

def total_possible_outcomes := 6^6
def number_of_ways_to_form_pairs := 15
def choose_first_pair := 6
def choose_second_pair := 15
def choose_third_pair := 6
def choose_fourth_die := 4
def choose_fifth_die := 3

def successful_outcomes := number_of_ways_to_form_pairs *
                           choose_first_pair *
                           choose_second_pair *
                           choose_third_pair *
                           choose_fourth_die *
                           choose_fifth_die

def probability_of_two_pairs_of_same_value := (successful_outcomes : ℚ) / total_possible_outcomes

theorem probability_of_two_pairs_of_same_value_is_correct :
  probability_of_two_pairs_of_same_value = 25 / 72 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_probability_of_two_pairs_of_same_value_is_correct_l1301_130140


namespace NUMINAMATH_GPT_greatest_possible_avg_speed_l1301_130175

theorem greatest_possible_avg_speed (initial_odometer : ℕ) (max_speed : ℕ) (time_hours : ℕ) (max_distance : ℕ) (target_palindrome : ℕ) :
  initial_odometer = 12321 →
  max_speed = 80 →
  time_hours = 4 →
  (target_palindrome = 12421 ∨ target_palindrome = 12521 ∨ target_palindrome = 12621 ∨ target_palindrome = 12721 ∨ target_palindrome = 12821 ∨ target_palindrome = 12921 ∨ target_palindrome = 13031) →
  target_palindrome - initial_odometer ≤ max_distance →
  max_distance = 300 →
  target_palindrome = 12621 →
  time_hours = 4 →
  target_palindrome - initial_odometer = 300 →
  (target_palindrome - initial_odometer) / time_hours = 75 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_greatest_possible_avg_speed_l1301_130175


namespace NUMINAMATH_GPT_total_money_made_l1301_130189

-- Define the conditions
def dollars_per_day : Int := 144
def number_of_days : Int := 22

-- State the proof problem
theorem total_money_made : (dollars_per_day * number_of_days = 3168) :=
by
  sorry

end NUMINAMATH_GPT_total_money_made_l1301_130189


namespace NUMINAMATH_GPT_interest_is_less_by_1940_l1301_130146

noncomputable def principal : ℕ := 2000
noncomputable def rate : ℕ := 3
noncomputable def time : ℕ := 3

noncomputable def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

noncomputable def difference (sum_lent interest : ℕ) : ℕ :=
  sum_lent - interest

theorem interest_is_less_by_1940 :
  difference principal (simple_interest principal rate time) = 1940 :=
by
  sorry

end NUMINAMATH_GPT_interest_is_less_by_1940_l1301_130146


namespace NUMINAMATH_GPT_qualified_flour_l1301_130169

def is_qualified_flour (weight : ℝ) : Prop :=
  weight ≥ 24.75 ∧ weight ≤ 25.25

theorem qualified_flour :
  is_qualified_flour 24.80 ∧
  ¬is_qualified_flour 24.70 ∧
  ¬is_qualified_flour 25.30 ∧
  ¬is_qualified_flour 25.51 :=
by
  sorry

end NUMINAMATH_GPT_qualified_flour_l1301_130169


namespace NUMINAMATH_GPT_area_under_the_curve_l1301_130104

theorem area_under_the_curve : 
  ∫ x in (0 : ℝ)..1, (x^2 + 1) = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_area_under_the_curve_l1301_130104


namespace NUMINAMATH_GPT_difference_of_squares_example_l1301_130153

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 305) (h2 : b = 295) :
  (a^2 - b^2) / 10 = 600 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_l1301_130153


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1301_130121

-- Definitions of transformations and final sequence S
def transformation (A : List ℕ) : List ℕ := 
  match A with
  | x :: y :: xs => (x + y) :: transformation (y :: xs)
  | _ => []

def nth_transform (A : List ℕ) (n : ℕ) : List ℕ :=
  Nat.iterate (λ L => transformation L) n A

def final_sequence (A : List ℕ) : ℕ :=
  match nth_transform A (A.length - 1) with
  | [x] => x
  | _ => 0

-- Proof Statements

theorem problem1 : final_sequence [1, 2, 3] = 8 := sorry

theorem problem2 (n : ℕ) : final_sequence (List.range (n+1)) = (n + 2) * 2 ^ (n - 1) := sorry

theorem problem3 (A B : List ℕ) (h : A = List.range (B.length)) (h_perm : B.permutations.contains A) : 
  final_sequence B = final_sequence A := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1301_130121


namespace NUMINAMATH_GPT_shifted_sine_odd_function_l1301_130131

theorem shifted_sine_odd_function (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  ∃ k : ℤ, ϕ = (2 * π / 3) + k * π ∧ 0 < (2 * π / 3) + k * π ∧ (2 * π / 3) + k * π < π :=
sorry

end NUMINAMATH_GPT_shifted_sine_odd_function_l1301_130131


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1301_130142

theorem solve_quadratic_inequality (x : ℝ) :
  (-3 * x^2 + 8 * x + 5 > 0) ↔ (x < -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1301_130142


namespace NUMINAMATH_GPT_basketball_free_throws_l1301_130130

theorem basketball_free_throws
  (a b x : ℕ)
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a)
  (h3 : 2 * a + 3 * b + x = 72)
  : x = 24 := by
  sorry

end NUMINAMATH_GPT_basketball_free_throws_l1301_130130


namespace NUMINAMATH_GPT_fraction_calculation_l1301_130183

theorem fraction_calculation : 
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 :=
by sorry

end NUMINAMATH_GPT_fraction_calculation_l1301_130183


namespace NUMINAMATH_GPT_quadratic_inequality_l1301_130114

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1301_130114


namespace NUMINAMATH_GPT_distance_between_QY_l1301_130119

theorem distance_between_QY 
  (m_rate : ℕ) (j_rate : ℕ) (j_distance : ℕ) (headstart : ℕ) 
  (t : ℕ) 
  (h1 : m_rate = 3) 
  (h2 : j_rate = 4) 
  (h3 : j_distance = 24) 
  (h4 : headstart = 1) 
  (h5 : j_distance = j_rate * (t - headstart)) 
  (h6 : t = 7) 
  (distance_m : ℕ := m_rate * t) 
  (distance_j : ℕ := j_distance) :
  distance_j + distance_m = 45 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_QY_l1301_130119


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l1301_130107

theorem geometric_sequence_fourth_term :
  let a₁ := 3^(3/4)
  let a₂ := 3^(2/4)
  let a₃ := 3^(1/4)
  ∃ a₄, a₄ = 1 ∧ a₂ = a₁ * (a₃ / a₂) ∧ a₃ = a₂ * (a₄ / a₃) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l1301_130107


namespace NUMINAMATH_GPT_evaluate_expression_l1301_130181

def a : ℕ := 3^1
def b : ℕ := 3^2
def c : ℕ := 3^3
def d : ℕ := 3^4
def e : ℕ := 3^10
def S : ℕ := a + b + c + d

theorem evaluate_expression : e - S = 58929 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1301_130181


namespace NUMINAMATH_GPT_power_mod_l1301_130156

theorem power_mod (n : ℕ) : 3^100 % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_power_mod_l1301_130156


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1301_130134

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - 4 * y^2 = 1) → (x = 2 * y ∨ x = -2 * y) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1301_130134


namespace NUMINAMATH_GPT_exists_perfect_square_intersection_l1301_130113

theorem exists_perfect_square_intersection : ∃ n : ℕ, n > 1 ∧ ∃ k : ℕ, (2^n - n) = k^2 :=
by sorry

end NUMINAMATH_GPT_exists_perfect_square_intersection_l1301_130113


namespace NUMINAMATH_GPT_mindmaster_code_count_l1301_130163

theorem mindmaster_code_count :
  let colors := 7
  let slots := 5
  (colors ^ slots) = 16807 :=
by
  -- Define the given conditions
  let colors := 7
  let slots := 5
  -- Proof statement to be inserted here
  sorry

end NUMINAMATH_GPT_mindmaster_code_count_l1301_130163


namespace NUMINAMATH_GPT_smallest_positive_integer_l1301_130196

theorem smallest_positive_integer :
  ∃ (n a b m : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ n = 153846 ∧
  (n = 10^m * a + b) ∧
  (7 * n = 2 * (10 * b + a)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1301_130196


namespace NUMINAMATH_GPT_square_difference_l1301_130111

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 36) 
  (h₂ : x * y = 8) : 
  (x - y)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_difference_l1301_130111


namespace NUMINAMATH_GPT_yoongi_class_combination_l1301_130186

theorem yoongi_class_combination : (Nat.choose 10 3 = 120) := by
  sorry

end NUMINAMATH_GPT_yoongi_class_combination_l1301_130186


namespace NUMINAMATH_GPT_sum_ineq_l1301_130126

theorem sum_ineq (x y z t : ℝ) (h₁ : x + y + z + t = 0) (h₂ : x^2 + y^2 + z^2 + t^2 = 1) :
  -1 ≤ x * y + y * z + z * t + t * x ∧ x * y + y * z + z * t + t * x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_ineq_l1301_130126


namespace NUMINAMATH_GPT_right_triangle_legs_l1301_130100

theorem right_triangle_legs (m r x y : ℝ) 
  (h1 : m^2 = x^2 + y^2) 
  (h2 : r = (x + y - m) / 2) 
  (h3 : r ≤ m * (Real.sqrt 2 - 1) / 2) : 
  (x = (2 * r + m + Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) ∧ 
  (y = (2 * r + m - Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_legs_l1301_130100


namespace NUMINAMATH_GPT_integer_solutions_l1301_130177

theorem integer_solutions (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_integer_solutions_l1301_130177


namespace NUMINAMATH_GPT_consecutive_numbers_product_l1301_130102

theorem consecutive_numbers_product (a b c d : ℤ) 
  (h1 : b = a + 1) 
  (h2 : c = a + 2) 
  (h3 : d = a + 3) 
  (h4 : a + d = 109) : 
  b * c = 2970 := by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_product_l1301_130102


namespace NUMINAMATH_GPT_evaluate_expression_l1301_130139

theorem evaluate_expression : (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1301_130139


namespace NUMINAMATH_GPT_perimeter_original_rectangle_l1301_130129

variable {L W : ℕ}

axiom area_original : L * W = 360
axiom area_changed : (L + 10) * (W - 6) = 360

theorem perimeter_original_rectangle : 2 * (L + W) = 76 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_original_rectangle_l1301_130129


namespace NUMINAMATH_GPT_clothing_prices_and_purchase_plans_l1301_130120

theorem clothing_prices_and_purchase_plans :
  ∃ (x y : ℕ) (a : ℤ), 
  x + y = 220 ∧
  6 * x = 5 * y ∧
  120 * a + 100 * (150 - a) ≤ 17000 ∧
  (90 ≤ a ∧ a ≤ 100) ∧
  x = 100 ∧
  y = 120 ∧
  (∀ b : ℤ, (90 ≤ b ∧ b ≤ 100) → 120 * b + 100 * (150 - b) ≥ 16800)
  :=
sorry

end NUMINAMATH_GPT_clothing_prices_and_purchase_plans_l1301_130120


namespace NUMINAMATH_GPT_power_of_product_l1301_130166

theorem power_of_product (x : ℝ) : (-x^4)^3 = -x^12 := 
by sorry

end NUMINAMATH_GPT_power_of_product_l1301_130166


namespace NUMINAMATH_GPT_solve_system_of_equations_l1301_130187

/-- Definition representing our system of linear equations. --/
def system_of_equations (x1 x2 : ℚ) : Prop :=
  (3 * x1 - 5 * x2 = 2) ∧ (2 * x1 + 4 * x2 = 5)

/-- The main theorem stating the solution to our system of equations. --/
theorem solve_system_of_equations : 
  ∃ x1 x2 : ℚ, system_of_equations x1 x2 ∧ x1 = 3/2 ∧ x2 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1301_130187


namespace NUMINAMATH_GPT_parabola_equivalence_l1301_130108

theorem parabola_equivalence :
  ∃ (a : ℝ) (h k : ℝ),
    (a = -3 ∧ h = -1 ∧ k = 2) ∧
    ∀ (x : ℝ), (y = -3 * x^2 + 1) → (y = -3 * (x + 1)^2 + 2) :=
sorry

end NUMINAMATH_GPT_parabola_equivalence_l1301_130108


namespace NUMINAMATH_GPT_LindseyMinimumSavings_l1301_130157
-- Import the library to bring in the necessary definitions and notations

-- Definitions from the problem conditions
def SeptemberSavings : ℕ := 50
def OctoberSavings : ℕ := 37
def NovemberSavings : ℕ := 11
def MomContribution : ℕ := 25
def VideoGameCost : ℕ := 87
def RemainingMoney : ℕ := 36

-- Problem statement as a Lean theorem
theorem LindseyMinimumSavings : 
  (SeptemberSavings + OctoberSavings + NovemberSavings) > 98 :=
  sorry

end NUMINAMATH_GPT_LindseyMinimumSavings_l1301_130157


namespace NUMINAMATH_GPT_reciprocal_of_fraction_diff_l1301_130160

theorem reciprocal_of_fraction_diff : 
  (∃ (a b : ℚ), a = 1/4 ∧ b = 1/5 ∧ (1 / (a - b)) = 20) :=
sorry

end NUMINAMATH_GPT_reciprocal_of_fraction_diff_l1301_130160


namespace NUMINAMATH_GPT_three_digit_sum_reverse_eq_l1301_130118

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_sum_reverse_eq_l1301_130118


namespace NUMINAMATH_GPT_overall_average_commission_rate_l1301_130184

-- Define conditions for the commissions and transaction amounts
def C₁ := 0.25 / 100 * 100 + 0.25 / 100 * 105.25
def C₂ := 0.35 / 100 * 150 + 0.45 / 100 * 155.50
def C₃ := 0.30 / 100 * 80 + 0.40 / 100 * 83
def total_commission := C₁ + C₂ + C₃
def TA := 100 + 105.25 + 150 + 155.50 + 80 + 83

-- The proposition to prove
theorem overall_average_commission_rate : (total_commission / TA) * 100 = 0.3429 :=
  by
  sorry

end NUMINAMATH_GPT_overall_average_commission_rate_l1301_130184


namespace NUMINAMATH_GPT_product_of_invertible_function_labels_l1301_130193

noncomputable def Function6 (x : ℝ) : ℝ := x^3 - 3 * x
def points7 : List (ℝ × ℝ) := [(-6, 3), (-5, 1), (-4, 2), (-3, -1), (-2, 0), (-1, -2), (0, 4), (1, 5)]
noncomputable def Function8 (x : ℝ) : ℝ := Real.sin x
noncomputable def Function9 (x : ℝ) : ℝ := 3 / x

def is_invertible6 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function6 x1 = y ∧ Function6 x2 = y ∧ (-2 ≤ x1 ∧ x1 ≤ 2) ∧ (-2 ≤ x2 ∧ x2 ≤ 2)
def is_invertible7 : Prop := ∀ (y : ℝ), ∃! x : ℝ, (x, y) ∈ points7
def is_invertible8 : Prop := ∀ (x1 x2 : ℝ), Function8 x1 = Function8 x2 → x1 = x2 ∧ (-Real.pi/2 ≤ x1 ∧ x1 ≤ Real.pi/2) ∧ (-Real.pi/2 ≤ x2 ∧ x2 ≤ Real.pi/2)
def is_invertible9 : Prop := ¬ ∃ (y : ℝ), ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ Function9 x1 = y ∧ Function9 x2 = y ∧ (-4 ≤ x1 ∧ x1 ≤ 4 ∧ x1 ≠ 0) ∧ (-4 ≤ x2 ∧ x2 ≤ 4 ∧ x2 ≠ 0)

theorem product_of_invertible_function_labels :
  (is_invertible6 = false) →
  (is_invertible7 = true) →
  (is_invertible8 = true) →
  (is_invertible9 = true) →
  7 * 8 * 9 = 504
:= by
  intros h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_product_of_invertible_function_labels_l1301_130193


namespace NUMINAMATH_GPT_sum_modulo_9_l1301_130143

theorem sum_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  -- Skipping the detailed proof steps
  sorry

end NUMINAMATH_GPT_sum_modulo_9_l1301_130143


namespace NUMINAMATH_GPT_prism_volume_l1301_130123

open Real

theorem prism_volume :
  ∃ (a b c : ℝ), a * b = 15 ∧ b * c = 10 ∧ c * a = 30 ∧ a * b * c = 30 * sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l1301_130123


namespace NUMINAMATH_GPT_exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l1301_130198

def small_numbers (n : ℕ) : Prop := n ≤ 150

theorem exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest :
  ∃ (N : ℕ), (∃ (a b : ℕ), small_numbers a ∧ small_numbers b ∧ (a + 1 = b) ∧ ¬(N % a = 0) ∧ ¬(N % b = 0))
  ∧ (∀ (m : ℕ), small_numbers m → ¬(m = a ∨ m = b) → N % m = 0) :=
sorry

end NUMINAMATH_GPT_exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l1301_130198


namespace NUMINAMATH_GPT_distinct_sequences_ten_flips_l1301_130171

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end NUMINAMATH_GPT_distinct_sequences_ten_flips_l1301_130171


namespace NUMINAMATH_GPT_remainder_of_sum_l1301_130150

theorem remainder_of_sum (p q : ℤ) (c d : ℤ) 
  (hc : c = 100 * p + 78)
  (hd : d = 150 * q + 123) :
  (c + d) % 50 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_sum_l1301_130150


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l1301_130161

theorem gasoline_tank_capacity :
  ∀ (x : ℕ), (5 / 6 * (x : ℚ) - 18 = 1 / 3 * (x : ℚ)) → x = 36 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l1301_130161


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1301_130168

theorem arithmetic_sequence_general_term:
  ∃ (a : ℕ → ℕ), 
    (∀ n, a n + 1 > a n) ∧
    (a 1 = 2) ∧ 
    ((a 2) ^ 2 = a 5 + 6) ∧ 
    (∀ n, a n = 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1301_130168


namespace NUMINAMATH_GPT_michael_max_correct_answers_l1301_130122

theorem michael_max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 30) 
  (h2 : 4 * c - 3 * w = 72) : 
  c ≤ 21 := 
sorry

end NUMINAMATH_GPT_michael_max_correct_answers_l1301_130122


namespace NUMINAMATH_GPT_total_handshakes_is_316_l1301_130180

def number_of_couples : ℕ := 15
def number_of_people : ℕ := number_of_couples * 2

def handshakes_among_men (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)
def handshakes_between_women : ℕ := 1
def total_handshakes (n : ℕ) : ℕ := handshakes_among_men n + handshakes_men_women n + handshakes_between_women

theorem total_handshakes_is_316 : total_handshakes number_of_couples = 316 :=
by
  sorry

end NUMINAMATH_GPT_total_handshakes_is_316_l1301_130180


namespace NUMINAMATH_GPT_basket_weight_l1301_130165

def weight_of_basket_alone (n_pears : ℕ) (weight_per_pear total_weight : ℚ) : ℚ :=
  total_weight - (n_pears * weight_per_pear)

theorem basket_weight :
  weight_of_basket_alone 30 0.36 11.26 = 0.46 := by
  sorry

end NUMINAMATH_GPT_basket_weight_l1301_130165


namespace NUMINAMATH_GPT_zeros_in_expansion_l1301_130172

def num_zeros_expansion (n : ℕ) : ℕ :=
-- This function counts the number of trailing zeros in the decimal representation of n.
sorry

theorem zeros_in_expansion : num_zeros_expansion ((10^12 - 3)^2) = 11 :=
sorry

end NUMINAMATH_GPT_zeros_in_expansion_l1301_130172


namespace NUMINAMATH_GPT_laura_rental_cost_l1301_130116

def rental_cost_per_day : ℝ := 30
def driving_cost_per_mile : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 300

theorem laura_rental_cost : rental_cost_per_day * days_rented + driving_cost_per_mile * miles_driven = 165 := by
  sorry

end NUMINAMATH_GPT_laura_rental_cost_l1301_130116


namespace NUMINAMATH_GPT_smallest_product_of_set_l1301_130148

noncomputable def smallest_product_set : Set ℤ := { -10, -3, 0, 4, 6 }

theorem smallest_product_of_set :
  ∃ (a b : ℤ), a ∈ smallest_product_set ∧ b ∈ smallest_product_set ∧ a ≠ b ∧ a * b = -60 ∧
  ∀ (x y : ℤ), x ∈ smallest_product_set ∧ y ∈ smallest_product_set ∧ x ≠ y → x * y ≥ -60 := 
sorry

end NUMINAMATH_GPT_smallest_product_of_set_l1301_130148


namespace NUMINAMATH_GPT_larger_number_is_23_l1301_130167

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end NUMINAMATH_GPT_larger_number_is_23_l1301_130167


namespace NUMINAMATH_GPT_naomi_total_wheels_l1301_130179

theorem naomi_total_wheels 
  (regular_bikes : ℕ) (children_bikes : ℕ) (tandem_bikes_4_wheels : ℕ) (tandem_bikes_6_wheels : ℕ)
  (wheels_per_regular_bike : ℕ) (wheels_per_children_bike : ℕ) (wheels_per_tandem_4wheel : ℕ) (wheels_per_tandem_6wheel : ℕ) :
  regular_bikes = 7 →
  children_bikes = 11 →
  tandem_bikes_4_wheels = 5 →
  tandem_bikes_6_wheels = 3 →
  wheels_per_regular_bike = 2 →
  wheels_per_children_bike = 4 →
  wheels_per_tandem_4wheel = 4 →
  wheels_per_tandem_6wheel = 6 →
  (regular_bikes * wheels_per_regular_bike) + 
  (children_bikes * wheels_per_children_bike) + 
  (tandem_bikes_4_wheels * wheels_per_tandem_4wheel) + 
  (tandem_bikes_6_wheels * wheels_per_tandem_6wheel) = 96 := 
by
  intros; sorry

end NUMINAMATH_GPT_naomi_total_wheels_l1301_130179


namespace NUMINAMATH_GPT_draw_points_value_l1301_130170

theorem draw_points_value
  (D : ℕ) -- Let D be the number of points for a draw
  (victory_points : ℕ := 3) -- points for a victory
  (defeat_points : ℕ := 0) -- points for a defeat
  (total_matches : ℕ := 20) -- total matches
  (points_after_5_games : ℕ := 8) -- points scored in the first 5 games
  (minimum_wins_remaining : ℕ := 9) -- at least 9 matches should be won in the remaining matches
  (target_points : ℕ := 40) : -- target points by the end of the tournament
  D = 1 := 
by 
  sorry


end NUMINAMATH_GPT_draw_points_value_l1301_130170


namespace NUMINAMATH_GPT_last_digit_of_3_pow_2012_l1301_130144

-- Theorem: The last digit of 3^2012 is 1 given the cyclic pattern of last digits for powers of 3.
theorem last_digit_of_3_pow_2012 : (3 ^ 2012) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_3_pow_2012_l1301_130144


namespace NUMINAMATH_GPT_find_c_value_l1301_130173

theorem find_c_value (b c : ℝ) 
  (h1 : 1 + b + c = 4) 
  (h2 : 25 + 5 * b + c = 4) : 
  c = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_c_value_l1301_130173


namespace NUMINAMATH_GPT_subtraction_of_fractions_l1301_130137

theorem subtraction_of_fractions :
  1 + 1 / 2 - 3 / 5 = 9 / 10 := by
  sorry

end NUMINAMATH_GPT_subtraction_of_fractions_l1301_130137


namespace NUMINAMATH_GPT_total_points_team_l1301_130106

def T : ℕ := 4
def J : ℕ := 2 * T + 6
def S : ℕ := J / 2
def R : ℕ := T + J - 3
def A : ℕ := S + R + 4

theorem total_points_team : T + J + S + R + A = 66 := by
  sorry

end NUMINAMATH_GPT_total_points_team_l1301_130106


namespace NUMINAMATH_GPT_xy_ratio_l1301_130152

variables (x y z t : ℝ)
variables (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t))

theorem xy_ratio (x y : ℝ) (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t)) :
  x / y = 25 :=
sorry

end NUMINAMATH_GPT_xy_ratio_l1301_130152


namespace NUMINAMATH_GPT_fewest_handshakes_is_zero_l1301_130101

noncomputable def fewest_handshakes (n k : ℕ) : ℕ :=
  if h : (n * (n - 1)) / 2 + k = 325 then k else 325

theorem fewest_handshakes_is_zero :
  ∃ n k : ℕ, (n * (n - 1)) / 2 + k = 325 ∧ 0 = fewest_handshakes n k :=
by
  sorry

end NUMINAMATH_GPT_fewest_handshakes_is_zero_l1301_130101


namespace NUMINAMATH_GPT_solve_inequality_l1301_130133

theorem solve_inequality (x : ℝ) : (x^2 - 50 * x + 625 ≤ 25) = (20 ≤ x ∧ x ≤ 30) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1301_130133


namespace NUMINAMATH_GPT_rectangle_dimensions_l1301_130145

theorem rectangle_dimensions (a1 a2 : ℝ) (h1 : a1 * a2 = 216) (h2 : a1 + a2 = 30 - 6)
  (h3 : 6 * 6 = 36) : (a1 = 12 ∧ a2 = 18) ∨ (a1 = 18 ∧ a2 = 12) :=
by
  -- The conditions are set; now we need the proof, which we'll replace with sorry for now.
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1301_130145
