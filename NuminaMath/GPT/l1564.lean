import Mathlib

namespace NUMINAMATH_GPT_number_of_unique_four_digit_numbers_from_2004_l1564_156488

-- Definitions representing the conditions
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_from_2004 (n : ℕ) : Prop := 
  ∀ d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], d ∈ [0, 2, 4]

-- The proposition we need to prove
theorem number_of_unique_four_digit_numbers_from_2004 :
  ∃ n : ℕ, is_four_digit_number n ∧ uses_digits_from_2004 n ∧ n = 6 := 
sorry

end NUMINAMATH_GPT_number_of_unique_four_digit_numbers_from_2004_l1564_156488


namespace NUMINAMATH_GPT_monthly_rent_l1564_156421

theorem monthly_rent (cost : ℕ) (maintenance_percentage : ℚ) (annual_taxes : ℕ) (desired_return_rate : ℚ) (monthly_rent : ℚ) :
  cost = 20000 ∧
  maintenance_percentage = 0.10 ∧
  annual_taxes = 460 ∧
  desired_return_rate = 0.06 →
  monthly_rent = 153.70 := 
sorry

end NUMINAMATH_GPT_monthly_rent_l1564_156421


namespace NUMINAMATH_GPT_f_n_2_l1564_156439

def f (m n : ℕ) : ℝ :=
if h : m = 1 ∧ n = 1 then 1 else
if h : n > m then 0 else 
sorry -- This would be calculated based on the recursive definition

lemma f_2_2 : f 2 2 = 2 :=
sorry

theorem f_n_2 (n : ℕ) (hn : n ≥ 1) : f n 2 = 2^(n - 1) :=
sorry

end NUMINAMATH_GPT_f_n_2_l1564_156439


namespace NUMINAMATH_GPT_min_value_expression_l1564_156424

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b - a - 2 * b = 0) :
  ∃ p : ℝ, p = (a^2/4 - 2/a + b^2 - 1/b) ∧ p = 7 :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l1564_156424


namespace NUMINAMATH_GPT_emily_necklaces_l1564_156431

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : total_beads = 52)
  (h2 : beads_per_necklace = 2)
  (h3 : necklaces_made = total_beads / beads_per_necklace) :
  necklaces_made = 26 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_emily_necklaces_l1564_156431


namespace NUMINAMATH_GPT_calculate_result_l1564_156443

theorem calculate_result : (-3 : ℝ)^(2022) * (1 / 3 : ℝ)^(2023) = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_calculate_result_l1564_156443


namespace NUMINAMATH_GPT_perp_lines_value_of_m_parallel_lines_value_of_m_l1564_156434

theorem perp_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) * ((m - 2) / 3) = -1)) → 
  m = 1 / 2 := 
sorry

theorem parallel_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) = ((m - 2) / 3))) → 
  m = -1 := 
sorry

end NUMINAMATH_GPT_perp_lines_value_of_m_parallel_lines_value_of_m_l1564_156434


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1564_156449

-- The conditions of the problem
variables (a b : ℝ)

-- The proposition to be proved
theorem sufficient_but_not_necessary_condition (h : a + b = 1) : 4 * a * b ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1564_156449


namespace NUMINAMATH_GPT_paper_cut_count_incorrect_l1564_156486

theorem paper_cut_count_incorrect (n : ℕ) (h : n = 1961) : 
  ∀ i, (∃ k, i = 7 ∨ i = 7 + 6 * k) → i % 6 = 1 → n ≠ i :=
by
  sorry

end NUMINAMATH_GPT_paper_cut_count_incorrect_l1564_156486


namespace NUMINAMATH_GPT_Katie_cupcakes_l1564_156400

theorem Katie_cupcakes (initial_cupcakes sold_cupcakes final_cupcakes : ℕ) (h1 : initial_cupcakes = 26) (h2 : sold_cupcakes = 20) (h3 : final_cupcakes = 26) :
  (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_Katie_cupcakes_l1564_156400


namespace NUMINAMATH_GPT_trigonometric_identity_tangent_line_l1564_156474

theorem trigonometric_identity_tangent_line 
  (α : ℝ) 
  (h_tan : Real.tan α = 4) 
  : Real.cos α ^ 2 - Real.sin (2 * α) = - 7 / 17 := 
by sorry

end NUMINAMATH_GPT_trigonometric_identity_tangent_line_l1564_156474


namespace NUMINAMATH_GPT_part1_part2_l1564_156401

noncomputable def f (x a : ℝ) : ℝ := |x + a|
noncomputable def g (x : ℝ) : ℝ := |x + 3| - x

theorem part1 (x : ℝ) : f x 1 < g x → x < 2 :=
sorry

theorem part2 (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x a < g x) → -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1564_156401


namespace NUMINAMATH_GPT_complement_union_complement_intersection_complementA_intersect_B_l1564_156458

def setA (x : ℝ) : Prop := 3 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 2 < x ∧ x < 10

theorem complement_union (x : ℝ) : ¬(setA x ∨ setB x) ↔ x ≤ 2 ∨ x ≥ 10 := sorry

theorem complement_intersection (x : ℝ) : ¬(setA x ∧ setB x) ↔ x < 3 ∨ x ≥ 7 := sorry

theorem complementA_intersect_B (x : ℝ) : (¬setA x ∧ setB x) ↔ (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) := sorry

end NUMINAMATH_GPT_complement_union_complement_intersection_complementA_intersect_B_l1564_156458


namespace NUMINAMATH_GPT_female_managers_count_l1564_156487

-- Definitions for the problem statement

def total_female_employees : ℕ := 500
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Problem parameters
variable (E M FM : ℕ) -- E: total employees, M: male employees, FM: female managers

-- Conditions
def total_employees_eq : Prop := E = M + total_female_employees
def total_managers_eq : Prop := fraction_of_managers * E = fraction_of_male_managers * M + FM

-- The statement we want to prove
theorem female_managers_count (h1 : total_employees_eq E M) (h2 : total_managers_eq E M FM) : FM = 200 :=
by
  -- to be proven
  sorry

end NUMINAMATH_GPT_female_managers_count_l1564_156487


namespace NUMINAMATH_GPT_two_digit_primes_with_digit_sum_10_count_l1564_156408

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end NUMINAMATH_GPT_two_digit_primes_with_digit_sum_10_count_l1564_156408


namespace NUMINAMATH_GPT_paper_string_area_l1564_156427

theorem paper_string_area (side len overlap : ℝ) (n : ℕ) (h_side : side = 30) 
                          (h_len : len = 30) (h_overlap : overlap = 7) (h_n : n = 6) :
  let area_one_sheet := side * len
  let effective_len := side - overlap
  let total_length := len + effective_len * (n - 1)
  let width := side
  let area := total_length * width
  area = 4350 := 
by
  sorry

end NUMINAMATH_GPT_paper_string_area_l1564_156427


namespace NUMINAMATH_GPT_symmetric_circle_l1564_156415

theorem symmetric_circle :
  ∀ (C D : Type) (hD : ∀ x y : ℝ, (x + 2)^2 + (y - 6)^2 = 1) (hline : ∀ x y : ℝ, x - y + 5 = 0), 
  (∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 1) := 
by sorry

end NUMINAMATH_GPT_symmetric_circle_l1564_156415


namespace NUMINAMATH_GPT_find_first_term_of_sequence_l1564_156485

theorem find_first_term_of_sequence (a : ℕ → ℝ)
  (h_rec : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h_a8 : a 8 = 2) :
  a 1 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_first_term_of_sequence_l1564_156485


namespace NUMINAMATH_GPT_obtuse_triangle_contradiction_l1564_156445

theorem obtuse_triangle_contradiction (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) : 
  (A > 90 ∧ B > 90) → false :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_contradiction_l1564_156445


namespace NUMINAMATH_GPT_land_area_l1564_156495

theorem land_area (x : ℝ) (h : (70 * x - 800) / 1.2 * 1.6 + 800 = 80 * x) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_land_area_l1564_156495


namespace NUMINAMATH_GPT_sum_of_integers_from_neg15_to_5_l1564_156481

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_from_neg15_to_5_l1564_156481


namespace NUMINAMATH_GPT_collinear_vectors_l1564_156450

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (OA OB OP : V) (m n : ℝ)

-- Given conditions
def non_collinear (OA OB : V) : Prop :=
  ∀ (t : ℝ), OA ≠ t • OB

def collinear_points (P A B : V) : Prop :=
  ∃ (t : ℝ), P - A = t • (B - A)

def linear_combination (OP OA OB : V) (m n : ℝ) : Prop :=
  OP = m • OA + n • OB

-- The theorem statement
theorem collinear_vectors (noncol : non_collinear OA OB)
  (collinearPAB : collinear_points OP OA OB)
  (lin_comb : linear_combination OP OA OB m n) :
  m = 2 ∧ n = -1 := by
sorry

end NUMINAMATH_GPT_collinear_vectors_l1564_156450


namespace NUMINAMATH_GPT_blue_faces_ratio_l1564_156484

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end NUMINAMATH_GPT_blue_faces_ratio_l1564_156484


namespace NUMINAMATH_GPT_fewer_popsicle_sticks_l1564_156422

theorem fewer_popsicle_sticks :
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  boys_total - girls_total = 6 := 
by
  let boys := 10
  let girls := 12
  let sticks_per_boy := 15
  let sticks_per_girl := 12
  let boys_total := boys * sticks_per_boy
  let girls_total := girls * sticks_per_girl
  show boys_total - girls_total = 6
  sorry

end NUMINAMATH_GPT_fewer_popsicle_sticks_l1564_156422


namespace NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1564_156470

-- Define the people
inductive Person
| A 
| B 
| C

open Person

-- Define the colors
inductive Color
| Red
| Yellow
| Blue

open Color

-- Event A: Person A gets the Red card
def event_a (assignment: Person → Color) : Prop := assignment A = Red

-- Event B: Person B gets the Red card
def event_b (assignment: Person → Color) : Prop := assignment B = Red

-- Definition of mutually exclusive events
def mutually_exclusive (P Q: Prop): Prop := P → ¬Q

-- Definition of complementary events
def complementary (P Q: Prop): Prop := P ↔ ¬Q

theorem mutually_exclusive_not_complementary :
  ∀ (assignment: Person → Color),
  mutually_exclusive (event_a assignment) (event_b assignment) ∧ ¬complementary (event_a assignment) (event_b assignment) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1564_156470


namespace NUMINAMATH_GPT_donuts_left_for_coworkers_l1564_156453

theorem donuts_left_for_coworkers :
  ∀ (total_donuts gluten_free regular gluten_free_chocolate gluten_free_plain regular_chocolate regular_plain consumed_gluten_free consumed_regular afternoon_gluten_free_chocolate afternoon_gluten_free_plain afternoon_regular_chocolate afternoon_regular_plain left_gluten_free_chocolate left_gluten_free_plain left_regular_chocolate left_regular_plain),
  total_donuts = 30 →
  gluten_free = 12 →
  regular = 18 →
  gluten_free_chocolate = 6 →
  gluten_free_plain = 6 →
  regular_chocolate = 11 →
  regular_plain = 7 →
  consumed_gluten_free = 1 →
  consumed_regular = 1 →
  afternoon_gluten_free_chocolate = 2 →
  afternoon_gluten_free_plain = 1 →
  afternoon_regular_chocolate = 2 →
  afternoon_regular_plain = 1 →
  left_gluten_free_chocolate = gluten_free_chocolate - consumed_gluten_free * 0.5 - afternoon_gluten_free_chocolate →
  left_gluten_free_plain = gluten_free_plain - consumed_gluten_free * 0.5 - afternoon_gluten_free_plain →
  left_regular_chocolate = regular_chocolate - consumed_regular * 1 - afternoon_regular_chocolate →
  left_regular_plain = regular_plain - consumed_regular * 0 - afternoon_regular_plain →
  left_gluten_free_chocolate + left_gluten_free_plain + left_regular_chocolate + left_regular_plain = 23 :=
by
  intros
  sorry

end NUMINAMATH_GPT_donuts_left_for_coworkers_l1564_156453


namespace NUMINAMATH_GPT_uncle_jerry_total_tomatoes_l1564_156417

def day1_tomatoes : ℕ := 120
def day2_tomatoes : ℕ := day1_tomatoes + 50
def day3_tomatoes : ℕ := 2 * day2_tomatoes
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes

theorem uncle_jerry_total_tomatoes : total_tomatoes = 630 := by
  sorry

end NUMINAMATH_GPT_uncle_jerry_total_tomatoes_l1564_156417


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1564_156473

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ( (2*x - 1)*x = 0 → x = 0 ) ∧ ( x = 0 → (2*x - 1)*x = 0 ) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1564_156473


namespace NUMINAMATH_GPT_fixed_point_of_function_l1564_156423

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, y = a^(x-1) + 1 ∧ (x, y) = (1, 2) :=
by 
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l1564_156423


namespace NUMINAMATH_GPT_find_multiple_l1564_156447

-- Definitions based on the problem's conditions
def n_drunk_drivers : ℕ := 6
def total_students : ℕ := 45
def num_speeders (M : ℕ) : ℕ := M * n_drunk_drivers - 3

-- The theorem that we need to prove
theorem find_multiple (M : ℕ) (h1: total_students = n_drunk_drivers + num_speeders M) : M = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1564_156447


namespace NUMINAMATH_GPT_eliminate_x3_term_l1564_156462

noncomputable def polynomial (n : ℝ) : Polynomial ℝ :=
  (Polynomial.X ^ 2 + Polynomial.C n * Polynomial.X + Polynomial.C 3) *
  (Polynomial.X ^ 2 - Polynomial.C 3 * Polynomial.X)

theorem eliminate_x3_term (n : ℝ) : (polynomial n).coeff 3 = 0 ↔ n = 3 :=
by
  -- sorry to skip the proof for now as it's not required
  sorry

end NUMINAMATH_GPT_eliminate_x3_term_l1564_156462


namespace NUMINAMATH_GPT_jade_driving_hours_per_day_l1564_156435

variable (Jade Krista : ℕ)
variable (days driving_hours total_hours : ℕ)

theorem jade_driving_hours_per_day :
  (days = 3) →
  (Krista = 6) →
  (total_hours = 42) →
  (total_hours = days * Jade + days * Krista) →
  Jade = 8 :=
by
  intros h_days h_krista h_total_hours h_equation
  sorry

end NUMINAMATH_GPT_jade_driving_hours_per_day_l1564_156435


namespace NUMINAMATH_GPT_find_printer_price_l1564_156471

variable (C P M : ℝ)

theorem find_printer_price
  (h1 : C + P + M = 3000)
  (h2 : P = (1/4) * (C + P + M + 800)) :
  P = 950 :=
sorry

end NUMINAMATH_GPT_find_printer_price_l1564_156471


namespace NUMINAMATH_GPT_mrs_smith_class_boys_girls_ratio_l1564_156432

theorem mrs_smith_class_boys_girls_ratio (total_students boys girls : ℕ) (h1 : boys / girls = 3 / 4) (h2 : boys + girls = 42) : girls = boys + 6 :=
by
  sorry

end NUMINAMATH_GPT_mrs_smith_class_boys_girls_ratio_l1564_156432


namespace NUMINAMATH_GPT_planes_divide_space_l1564_156420

-- Definition of a triangular prism
def triangular_prism (V : Type) (P : Set (Set V)) : Prop :=
  ∃ (A B C D E F : V),
    P = {{A, B, C}, {D, E, F}, {A, B, D, E}, {B, C, E, F}, {C, A, F, D}}

-- The condition: planes containing the faces of a triangular prism
def planes_containing_faces (V : Type) (P : Set (Set V)) : Prop :=
  triangular_prism V P

-- Proof statement: The planes containing the faces of a triangular prism divide the space into 21 parts
theorem planes_divide_space (V : Type) (P : Set (Set V))
  (h : planes_containing_faces V P) :
  ∃ parts : ℕ, parts = 21 := by
  sorry

end NUMINAMATH_GPT_planes_divide_space_l1564_156420


namespace NUMINAMATH_GPT_alexis_suit_coat_expense_l1564_156469

theorem alexis_suit_coat_expense :
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  budget - leftover - other_expenses = 38 := 
by
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  sorry

end NUMINAMATH_GPT_alexis_suit_coat_expense_l1564_156469


namespace NUMINAMATH_GPT_max_value_x_minus_y_l1564_156466

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_x_minus_y_l1564_156466


namespace NUMINAMATH_GPT_fraction_comparison_l1564_156490

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l1564_156490


namespace NUMINAMATH_GPT_total_time_l1564_156416

def time_to_eat_cereal (rate1 rate2 rate3 : ℚ) (amount : ℚ) : ℚ :=
  let combined_rate := rate1 + rate2 + rate3
  amount / combined_rate

theorem total_time (rate1 rate2 rate3 : ℚ) (amount : ℚ) 
  (h1 : rate1 = 1 / 15)
  (h2 : rate2 = 1 / 20)
  (h3 : rate3 = 1 / 30)
  (h4 : amount = 4) : 
  time_to_eat_cereal rate1 rate2 rate3 amount = 80 / 3 := 
by 
  rw [time_to_eat_cereal, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_total_time_l1564_156416


namespace NUMINAMATH_GPT_dice_probability_l1564_156410

theorem dice_probability (D1 D2 D3 : ℕ) (hD1 : 0 ≤ D1) (hD1' : D1 < 10) (hD2 : 0 ≤ D2) (hD2' : D2 < 10) (hD3 : 0 ≤ D3) (hD3' : D3 < 10) :
  ∃ p : ℚ, p = 1 / 10 :=
by
  let outcomes := 10 * 10 * 10
  let favorable := 100
  let expected_probability : ℚ := favorable / outcomes
  use expected_probability
  sorry

end NUMINAMATH_GPT_dice_probability_l1564_156410


namespace NUMINAMATH_GPT_only_solutions_mod_n_l1564_156438

theorem only_solutions_mod_n (n : ℕ) : (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % (n : ℤ) = 0) ↔ (∃ k : ℕ, n = 3 ^ k) := 
sorry

end NUMINAMATH_GPT_only_solutions_mod_n_l1564_156438


namespace NUMINAMATH_GPT_find_rate_percent_l1564_156479

def P : ℝ := 800
def SI : ℝ := 200
def T : ℝ := 4

theorem find_rate_percent (R : ℝ) :
  SI = P * R * T / 100 → R = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l1564_156479


namespace NUMINAMATH_GPT_fraction_of_painted_surface_area_l1564_156404

def total_surface_area_of_smaller_prisms : ℕ := 
  let num_smaller_prisms := 27
  let num_square_faces := num_smaller_prisms * 3
  let num_triangular_faces := num_smaller_prisms * 2
  num_square_faces + num_triangular_faces

def painted_surface_area_of_larger_prism : ℕ :=
  let painted_square_faces := 3 * 9
  let painted_triangular_faces := 2 * 9
  painted_square_faces + painted_triangular_faces

theorem fraction_of_painted_surface_area : 
  (painted_surface_area_of_larger_prism : ℚ) / (total_surface_area_of_smaller_prisms : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_of_painted_surface_area_l1564_156404


namespace NUMINAMATH_GPT_resulting_solution_percentage_l1564_156429

theorem resulting_solution_percentage :
  ∀ (C_init R C_replace : ℚ), 
  C_init = 0.85 → 
  R = 0.6923076923076923 → 
  C_replace = 0.2 → 
  (C_init * (1 - R) + C_replace * R) = 0.4 :=
by
  intros C_init R C_replace hC_init hR hC_replace
  -- Omitted proof here
  sorry

end NUMINAMATH_GPT_resulting_solution_percentage_l1564_156429


namespace NUMINAMATH_GPT_squares_circles_intersections_l1564_156489

noncomputable def number_of_intersections (p1 p2 : (ℤ × ℤ)) (square_side : ℚ) (circle_radius : ℚ) : ℕ :=
sorry -- function definition placeholder

theorem squares_circles_intersections :
  let p1 := (0, 0)
  let p2 := (1009, 437)
  let square_side := (1 : ℚ) / 4
  let circle_radius := (1 : ℚ) / 8
  (number_of_intersections p1 p2 square_side circle_radius) = 526 := by
  sorry

end NUMINAMATH_GPT_squares_circles_intersections_l1564_156489


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1564_156494

theorem arithmetic_sequence_product
  (a d : ℤ)
  (h1 : a + 5 * d = 17)
  (h2 : d = 2) :
  (a + 2 * d) * (a + 3 * d) = 143 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1564_156494


namespace NUMINAMATH_GPT_edward_skee_ball_tickets_l1564_156403

theorem edward_skee_ball_tickets (w_tickets : Nat) (candy_cost : Nat) (num_candies : Nat) (total_tickets : Nat) (skee_ball_tickets : Nat) :
  w_tickets = 3 ∧ candy_cost = 4 ∧ num_candies = 2 ∧ total_tickets = num_candies * candy_cost ∧ total_tickets - w_tickets = skee_ball_tickets → 
  skee_ball_tickets = 5 :=
by
  sorry

end NUMINAMATH_GPT_edward_skee_ball_tickets_l1564_156403


namespace NUMINAMATH_GPT_find_m_l1564_156460

theorem find_m (m : ℕ) (h : m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l1564_156460


namespace NUMINAMATH_GPT_bridesmaids_count_l1564_156476

theorem bridesmaids_count
  (hours_per_dress : ℕ)
  (hours_per_week : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (dresses : ℕ) :
  hours_per_dress = 12 →
  hours_per_week = 4 →
  weeks = 15 →
  total_hours = hours_per_week * weeks →
  dresses = total_hours / hours_per_dress →
  dresses = 5 := by
  sorry

end NUMINAMATH_GPT_bridesmaids_count_l1564_156476


namespace NUMINAMATH_GPT_number_of_students_from_second_department_is_17_l1564_156446

noncomputable def students_selected_from_second_department 
  (total_students : ℕ)
  (num_departments : ℕ)
  (students_per_department : List (ℕ × ℕ))
  (sample_size : ℕ)
  (starting_number : ℕ) : ℕ :=
-- This function will compute the number of students selected from the second department.
sorry

theorem number_of_students_from_second_department_is_17 : 
  students_selected_from_second_department 600 3 
    [(1, 300), (301, 495), (496, 600)] 50 3 = 17 :=
-- Proof is left as an exercise.
sorry

end NUMINAMATH_GPT_number_of_students_from_second_department_is_17_l1564_156446


namespace NUMINAMATH_GPT_largest_num_of_hcf_and_lcm_factors_l1564_156464

theorem largest_num_of_hcf_and_lcm_factors (hcf : ℕ) (f1 f2 : ℕ) (hcf_eq : hcf = 23) (f1_eq : f1 = 13) (f2_eq : f2 = 14) : 
    hcf * max f1 f2 = 322 :=
by
  -- use the conditions to find the largest number
  rw [hcf_eq, f1_eq, f2_eq]
  sorry

end NUMINAMATH_GPT_largest_num_of_hcf_and_lcm_factors_l1564_156464


namespace NUMINAMATH_GPT_petya_friends_l1564_156454

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end NUMINAMATH_GPT_petya_friends_l1564_156454


namespace NUMINAMATH_GPT_bread_calories_l1564_156452

theorem bread_calories (total_calories : Nat) (pb_calories : Nat) (pb_servings : Nat) (bread_pieces : Nat) (bread_calories : Nat)
  (h1 : total_calories = 500)
  (h2 : pb_calories = 200)
  (h3 : pb_servings = 2)
  (h4 : bread_pieces = 1)
  (h5 : total_calories = pb_servings * pb_calories + bread_pieces * bread_calories) : 
  bread_calories = 100 :=
by
  sorry

end NUMINAMATH_GPT_bread_calories_l1564_156452


namespace NUMINAMATH_GPT_wire_length_before_cut_l1564_156441

theorem wire_length_before_cut (S : ℝ) (L : ℝ) (h1 : S = 4) (h2 : S = (2/5) * L) : S + L = 14 :=
by 
  sorry

end NUMINAMATH_GPT_wire_length_before_cut_l1564_156441


namespace NUMINAMATH_GPT_ludek_unique_stamps_l1564_156437

theorem ludek_unique_stamps (K M L : ℕ) (k_m_shared k_l_shared m_l_shared : ℕ)
  (hk : K + M = 101)
  (hl : K + L = 115)
  (hm : M + L = 110)
  (k_m_shared := 5)
  (k_l_shared := 12)
  (m_l_shared := 7) :
  L - k_l_shared - m_l_shared = 43 :=
by
  sorry

end NUMINAMATH_GPT_ludek_unique_stamps_l1564_156437


namespace NUMINAMATH_GPT_number_of_workers_l1564_156499

theorem number_of_workers (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 350000) : W = 1000 :=
sorry

end NUMINAMATH_GPT_number_of_workers_l1564_156499


namespace NUMINAMATH_GPT_distance_difference_l1564_156405

theorem distance_difference (t : ℕ) (speed_alice speed_bob : ℕ) :
  speed_alice = 15 → speed_bob = 10 → t = 6 → (speed_alice * t) - (speed_bob * t) = 30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_distance_difference_l1564_156405


namespace NUMINAMATH_GPT_total_doctors_and_nurses_l1564_156426

theorem total_doctors_and_nurses
    (ratio_doctors_nurses : ℕ -> ℕ -> Prop)
    (num_nurses : ℕ)
    (h₁ : ratio_doctors_nurses 2 3)
    (h₂ : num_nurses = 150) :
    ∃ num_doctors total_doctors_nurses, 
    (total_doctors_nurses = num_doctors + num_nurses) 
    ∧ (num_doctors / num_nurses = 2 / 3) 
    ∧ total_doctors_nurses = 250 := 
by
  sorry

end NUMINAMATH_GPT_total_doctors_and_nurses_l1564_156426


namespace NUMINAMATH_GPT_evaluate_three_star_twostar_one_l1564_156467

def operator_star (a b : ℕ) : ℕ :=
  a^b - b^a

theorem evaluate_three_star_twostar_one : operator_star 3 (operator_star 2 1) = 2 := 
  by
    sorry

end NUMINAMATH_GPT_evaluate_three_star_twostar_one_l1564_156467


namespace NUMINAMATH_GPT_Carly_fourth_week_running_distance_l1564_156442

theorem Carly_fourth_week_running_distance :
  let week1_distance_per_day := 2
  let week2_distance_per_day := (week1_distance_per_day * 2) + 3
  let week3_distance_per_day := week2_distance_per_day * (9 / 7)
  let week4_intended_distance_per_day := week3_distance_per_day * 0.9
  let week4_actual_distance_per_day := week4_intended_distance_per_day * 0.5
  let week4_days_run := 5 -- due to 2 rest days
  (week4_actual_distance_per_day * week4_days_run) = 20.25 := 
by 
    -- We use sorry here to skip the proof
    sorry

end NUMINAMATH_GPT_Carly_fourth_week_running_distance_l1564_156442


namespace NUMINAMATH_GPT_order_of_abc_l1564_156461

noncomputable def a : ℝ := 2017^0
noncomputable def b : ℝ := 2015 * 2017 - 2016^2
noncomputable def c : ℝ := ((-2/3)^2016) * ((3/2)^2017)

theorem order_of_abc : b < a ∧ a < c := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_order_of_abc_l1564_156461


namespace NUMINAMATH_GPT_max_integer_value_l1564_156448

theorem max_integer_value (x : ℝ) : 
  ∃ M : ℤ, ∀ y : ℝ, (M = ⌊ 1 + 10 / (4 * y^2 + 12 * y + 9) ⌋ ∧ M ≤ 11) := 
sorry

end NUMINAMATH_GPT_max_integer_value_l1564_156448


namespace NUMINAMATH_GPT_find_subtracted_number_l1564_156425

variable (initial_number : Real)
variable (sum : Real := initial_number + 5)
variable (product : Real := sum * 7)
variable (quotient : Real := product / 5)
variable (remainder : Real := 33)

theorem find_subtracted_number 
  (initial_number_eq : initial_number = 22.142857142857142)
  : quotient - remainder = 5 := by
  sorry

end NUMINAMATH_GPT_find_subtracted_number_l1564_156425


namespace NUMINAMATH_GPT_convert_to_polar_coordinates_l1564_156419

open Real

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * π - arctan (abs y / abs x) else arctan (abs y / abs x)
  (r, θ)

theorem convert_to_polar_coordinates : 
  polar_coordinates 3 (-3) = (3 * sqrt 2, 7 * π / 4) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_polar_coordinates_l1564_156419


namespace NUMINAMATH_GPT_multiples_of_15_between_17_and_158_l1564_156409

theorem multiples_of_15_between_17_and_158 : 
  let first := 30
  let last := 150
  let step := 15
  Nat.succ ((last - first) / step) = 9 := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_15_between_17_and_158_l1564_156409


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1564_156411

theorem recurring_decimal_to_fraction :
  ∃ (frac : ℚ), frac = 1045 / 1998 ∧ 0.5 + (23 / 999) = frac :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1564_156411


namespace NUMINAMATH_GPT_sum_of_first_six_primes_mod_seventh_prime_l1564_156428

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end NUMINAMATH_GPT_sum_of_first_six_primes_mod_seventh_prime_l1564_156428


namespace NUMINAMATH_GPT_min_value_one_over_x_plus_one_over_y_l1564_156430

theorem min_value_one_over_x_plus_one_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) : 
  (1 / x + 1 / y) ≥ 1 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_min_value_one_over_x_plus_one_over_y_l1564_156430


namespace NUMINAMATH_GPT_quadratic_function_expression_rational_function_expression_l1564_156436

-- Problem 1:
theorem quadratic_function_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) ∧ (f 0 = 1) → (∀ x, f x = (3 / 2) * x^2 - (3 / 2) * x + 1) :=
by
  sorry

-- Problem 2:
theorem rational_function_expression (f : ℝ → ℝ) : 
  (∀ x, x ≠ 0 → 3 * f (1 / x) + f x = x) → 
  (∀ x, x ≠ 0 → f x = 3 / (8 * x) - x / 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_expression_rational_function_expression_l1564_156436


namespace NUMINAMATH_GPT_determine_remaining_sides_l1564_156475

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end NUMINAMATH_GPT_determine_remaining_sides_l1564_156475


namespace NUMINAMATH_GPT_geometric_sequence_condition_l1564_156493

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (∀ n ≥ 2, a n = 2 * a (n-1)) → 
  (∃ r, r = 2 ∧ ∀ n ≥ 2, a n = r * a (n-1)) ∧ 
  (∃ b, b ≠ 0 ∧ ∀ n, a n = 0) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l1564_156493


namespace NUMINAMATH_GPT_c_left_before_completion_l1564_156483

def a_one_day_work : ℚ := 1 / 24
def b_one_day_work : ℚ := 1 / 30
def c_one_day_work : ℚ := 1 / 40
def total_work_completed (days : ℚ) : Prop := days = 11

theorem c_left_before_completion (days_left : ℚ) (h : total_work_completed 11) :
  (11 - days_left) * (a_one_day_work + b_one_day_work + c_one_day_work) +
  (days_left * (a_one_day_work + b_one_day_work)) = 1 :=
sorry

end NUMINAMATH_GPT_c_left_before_completion_l1564_156483


namespace NUMINAMATH_GPT_find_number_l1564_156413

theorem find_number : ∃ x : ℝ, (6 * ((x / 8 + 8) - 30) = 12) ∧ x = 192 :=
by sorry

end NUMINAMATH_GPT_find_number_l1564_156413


namespace NUMINAMATH_GPT_gcd_459_357_l1564_156414

theorem gcd_459_357 :
  Nat.gcd 459 357 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_459_357_l1564_156414


namespace NUMINAMATH_GPT_discount_rate_on_pony_jeans_is_15_l1564_156478

noncomputable def discountProblem : Prop :=
  ∃ (F P : ℝ),
    (15 * 3 * F / 100 + 18 * 2 * P / 100 = 8.55) ∧ 
    (F + P = 22) ∧ 
    (P = 15)

theorem discount_rate_on_pony_jeans_is_15 : discountProblem :=
sorry

end NUMINAMATH_GPT_discount_rate_on_pony_jeans_is_15_l1564_156478


namespace NUMINAMATH_GPT_sum_of_squares_correct_l1564_156451

-- Define the three incorrect entries
def incorrect_entry_1 : Nat := 52
def incorrect_entry_2 : Nat := 81
def incorrect_entry_3 : Nat := 111

-- Define the sum of the squares of these entries
def sum_of_squares : Nat := incorrect_entry_1 ^ 2 + incorrect_entry_2 ^ 2 + incorrect_entry_3 ^ 2

-- State that this sum of squares equals 21586
theorem sum_of_squares_correct : sum_of_squares = 21586 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_correct_l1564_156451


namespace NUMINAMATH_GPT_math_problem_l1564_156433

theorem math_problem :
  let numerator := (15^4 + 400) * (30^4 + 400) * (45^4 + 400) * (60^4 + 400) * (75^4 + 400)
  let denominator := (5^4 + 400) * (20^4 + 400) * (35^4 + 400) * (50^4 + 400) * (65^4 + 400)
  numerator / denominator = 301 :=
by 
  sorry

end NUMINAMATH_GPT_math_problem_l1564_156433


namespace NUMINAMATH_GPT_sweet_tray_GCD_l1564_156465

/-!
Tim has a bag of 36 orange-flavoured sweets and Peter has a bag of 44 grape-flavoured sweets.
They have to divide up the sweets into small trays with equal number of sweets;
each tray containing either orange-flavoured or grape-flavoured sweets only.
The largest possible number of sweets in each tray without any remainder is 4.
-/

theorem sweet_tray_GCD :
  Nat.gcd 36 44 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sweet_tray_GCD_l1564_156465


namespace NUMINAMATH_GPT_petya_run_time_l1564_156406

-- Definitions
def time_petya_4_to_1 : ℕ := 12

-- Conditions
axiom time_mom_condition : ∃ (time_mom : ℕ), time_petya_4_to_1 = time_mom - 2
axiom time_mom_5_to_1_condition : ∃ (time_petya_5_to_1 : ℕ), ∀ time_mom : ℕ, time_mom = time_petya_5_to_1 - 2

-- Proof statement
theorem petya_run_time :
  ∃ (time_petya_4_to_1 : ℕ), time_petya_4_to_1 = 12 :=
sorry

end NUMINAMATH_GPT_petya_run_time_l1564_156406


namespace NUMINAMATH_GPT_hyperbola_equation_l1564_156477

theorem hyperbola_equation (a b c : ℝ)
  (ha : a > 0) (hb : b > 0)
  (eccentricity : c = 2 * a)
  (distance_foci_asymptote : b = 1)
  (hyperbola_eq : c^2 = a^2 + b^2) :
  (3 * x^2 - y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1564_156477


namespace NUMINAMATH_GPT_cost_of_small_bonsai_l1564_156482

variable (cost_small_bonsai cost_big_bonsai : ℝ)

theorem cost_of_small_bonsai : 
  cost_big_bonsai = 20 → 
  3 * cost_small_bonsai + 5 * cost_big_bonsai = 190 → 
  cost_small_bonsai = 30 := 
by
  intros h1 h2 
  sorry

end NUMINAMATH_GPT_cost_of_small_bonsai_l1564_156482


namespace NUMINAMATH_GPT_files_remaining_on_flash_drive_l1564_156440

def initial_music_files : ℕ := 32
def initial_video_files : ℕ := 96
def deleted_files : ℕ := 60

def total_initial_files : ℕ := initial_music_files + initial_video_files

theorem files_remaining_on_flash_drive 
  (h : total_initial_files = 128) : (total_initial_files - deleted_files) = 68 := by
  sorry

end NUMINAMATH_GPT_files_remaining_on_flash_drive_l1564_156440


namespace NUMINAMATH_GPT_cone_volume_l1564_156491

theorem cone_volume (l : ℝ) (S_side : ℝ) (h r V : ℝ)
  (hl : l = 10)
  (hS : S_side = 60 * Real.pi)
  (hr : S_side = π * r * l)
  (hh : h = Real.sqrt (l^2 - r^2))
  (hV : V = (1/3) * π * r^2 * h) :
  V = 96 * Real.pi := 
sorry

end NUMINAMATH_GPT_cone_volume_l1564_156491


namespace NUMINAMATH_GPT_quotient_is_20_l1564_156459

theorem quotient_is_20 (D d r Q : ℕ) (hD : D = 725) (hd : d = 36) (hr : r = 5) (h : D = d * Q + r) :
  Q = 20 :=
by sorry

end NUMINAMATH_GPT_quotient_is_20_l1564_156459


namespace NUMINAMATH_GPT_find_ruv_l1564_156497

theorem find_ruv (u v : ℝ) : 
  (∃ u v : ℝ, 
    (3 + 8 * u + 5, 1 - 4 * u + 2) = (4 + -3 * v + 5, 2 + 4 * v + 2)) →
  (u = -1/2 ∧ v = -1) :=
by
  intros H
  sorry

end NUMINAMATH_GPT_find_ruv_l1564_156497


namespace NUMINAMATH_GPT_total_spent_l1564_156444

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end NUMINAMATH_GPT_total_spent_l1564_156444


namespace NUMINAMATH_GPT_inscribed_sphere_volume_l1564_156418

theorem inscribed_sphere_volume (edge_length : ℝ) (h_edge : edge_length = 12) : 
  ∃ (V : ℝ), V = 288 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_volume_l1564_156418


namespace NUMINAMATH_GPT_sum_of_min_max_l1564_156492

-- Define the necessary parameters and conditions
variables (n k : ℕ)
  (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ)
  (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ)
  (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max)

-- The goal is to prove that the sum of m and M equals n
theorem sum_of_min_max (n k : ℕ) (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ) (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ) (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max) :
  m + M = n := 
sorry

end NUMINAMATH_GPT_sum_of_min_max_l1564_156492


namespace NUMINAMATH_GPT_min_value_f_l1564_156496

theorem min_value_f
  (a b c : ℝ)
  (α β γ : ℤ)
  (hα : α = 1 ∨ α = -1)
  (hβ : β = 1 ∨ β = -1)
  (hγ : γ = 1 ∨ γ = -1)
  (h : a * α + b * β + c * γ = 0) :
  (∃ f_min : ℝ, f_min = ( ((a ^ 3 + b ^ 3 + c ^ 3) / (a * b * c)) ^ 2) ∧ f_min = 9) :=
sorry

end NUMINAMATH_GPT_min_value_f_l1564_156496


namespace NUMINAMATH_GPT_jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l1564_156472

theorem jia_can_formulate_quadratic :
  ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem yi_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem bing_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem ding_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

end NUMINAMATH_GPT_jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l1564_156472


namespace NUMINAMATH_GPT_domain_of_f_l1564_156455

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (Real.sqrt (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = x} = Set.Ioi 7 := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1564_156455


namespace NUMINAMATH_GPT_exists_two_integers_with_difference_divisible_by_2022_l1564_156456

theorem exists_two_integers_with_difference_divisible_by_2022 (a : Fin 2023 → ℤ) : 
  ∃ i j : Fin 2023, i ≠ j ∧ (a i - a j) % 2022 = 0 := by
  sorry

end NUMINAMATH_GPT_exists_two_integers_with_difference_divisible_by_2022_l1564_156456


namespace NUMINAMATH_GPT_kate_needs_more_money_l1564_156412

theorem kate_needs_more_money
  (pen_price : ℝ)
  (notebook_price : ℝ)
  (artset_price : ℝ)
  (kate_pen_money_fraction : ℝ)
  (notebook_discount : ℝ)
  (artset_discount : ℝ)
  (kate_artset_money : ℝ) :
  pen_price = 30 →
  notebook_price = 20 →
  artset_price = 50 →
  kate_pen_money_fraction = 1/3 →
  notebook_discount = 0.15 →
  artset_discount = 0.4 →
  kate_artset_money = 10 →
  (pen_price - kate_pen_money_fraction * pen_price) +
  (notebook_price * (1 - notebook_discount)) +
  (artset_price * (1 - artset_discount) - kate_artset_money) = 57 :=
by
  sorry

end NUMINAMATH_GPT_kate_needs_more_money_l1564_156412


namespace NUMINAMATH_GPT_fraction_value_l1564_156468

theorem fraction_value (m n : ℤ) (h : (m - 8) * (m - 8) + abs (n + 6) = 0) : n / m = -(3 / 4) :=
by sorry

end NUMINAMATH_GPT_fraction_value_l1564_156468


namespace NUMINAMATH_GPT_four_digit_numbers_count_l1564_156463

open Nat

def is_valid_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def four_diff_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

def leading_digit_not_zero (a : ℕ) : Prop :=
  a ≠ 0

def largest_digit_seven (a b c d : ℕ) : Prop :=
  a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7

theorem four_digit_numbers_count :
  ∃ n, n = 45 ∧
  ∀ (a b c d : ℕ),
    four_diff_digits a b c d ∧
    leading_digit_not_zero a ∧
    is_multiple_of_5 (a * 1000 + b * 100 + c * 10 + d) ∧
    is_multiple_of_3 (a * 1000 + b * 100 + c * 10 + d) ∧
    largest_digit_seven a b c d →
    n = 45 :=
sorry

end NUMINAMATH_GPT_four_digit_numbers_count_l1564_156463


namespace NUMINAMATH_GPT_union_A_B_inter_A_compl_B_range_of_a_l1564_156402

-- Define the sets A, B, and C
def A := {x : ℝ | -1 ≤ x ∧ x < 3}
def B := {x : ℝ | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) := {x : ℝ | x ≥ a - 1}

-- Prove A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} :=
by sorry

-- Prove A ∩ (complement B) = {x | -1 ≤ x < 2}
theorem inter_A_compl_B : A ∩ (compl B) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry

-- Prove the range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 :=
by sorry

end NUMINAMATH_GPT_union_A_B_inter_A_compl_B_range_of_a_l1564_156402


namespace NUMINAMATH_GPT_value_of_fraction_l1564_156498

theorem value_of_fraction (a b c d e f : ℚ) (h1 : a / b = 1 / 3) (h2 : c / d = 1 / 3) (h3 : e / f = 1 / 3) :
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l1564_156498


namespace NUMINAMATH_GPT_probability_none_hit_l1564_156480

theorem probability_none_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (1 - p)^5 = (1 - p) * (1 - p) * (1 - p) * (1 - p) * (1 - p) :=
by sorry

end NUMINAMATH_GPT_probability_none_hit_l1564_156480


namespace NUMINAMATH_GPT_model_to_reality_length_l1564_156407

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end NUMINAMATH_GPT_model_to_reality_length_l1564_156407


namespace NUMINAMATH_GPT_min_quadratic_expr_l1564_156457

noncomputable def quadratic_expr (x : ℝ) := x^2 + 10 * x + 3

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = -22 :=
by
  use -5
  simp [quadratic_expr]
  sorry

end NUMINAMATH_GPT_min_quadratic_expr_l1564_156457
