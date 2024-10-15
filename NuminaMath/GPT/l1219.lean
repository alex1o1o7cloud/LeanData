import Mathlib

namespace NUMINAMATH_GPT_closest_point_on_line_l1219_121959

theorem closest_point_on_line (x y : ℝ) (h : y = (x - 3) / 3) : 
  (∃ p : ℝ × ℝ, p = (4, -2) ∧ ∀ q : ℝ × ℝ, (q.1, q.2) = (x, y) ∧ q ≠ p → dist p q ≥ dist p (33/10, 1/10)) :=
sorry

end NUMINAMATH_GPT_closest_point_on_line_l1219_121959


namespace NUMINAMATH_GPT_equation_represents_single_point_l1219_121944

theorem equation_represents_single_point (d : ℝ) :
  (∀ x y : ℝ, 3*x^2 + 4*y^2 + 6*x - 8*y + d = 0 ↔ (x = -1 ∧ y = 1)) → d = 7 :=
sorry

end NUMINAMATH_GPT_equation_represents_single_point_l1219_121944


namespace NUMINAMATH_GPT_B_subset_A_implies_m_le_5_l1219_121949

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def setB (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

theorem B_subset_A_implies_m_le_5 :
  B ⊆ A → (∀ k : ℝ, k ∈ setB m → k ∈ setA) → m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_B_subset_A_implies_m_le_5_l1219_121949


namespace NUMINAMATH_GPT_domain_of_log_function_l1219_121923

noncomputable def domain_f (k : ℤ) : Set ℝ :=
  {x : ℝ | (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
                      (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3))
  ↔ (3 - 4 * Real.sin x ^ 2 > 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_log_function_l1219_121923


namespace NUMINAMATH_GPT_has_real_root_neg_one_l1219_121911

theorem has_real_root_neg_one : 
  (-1)^2 - (-1) - 2 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_has_real_root_neg_one_l1219_121911


namespace NUMINAMATH_GPT_unoccupied_seats_in_business_class_l1219_121941

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end NUMINAMATH_GPT_unoccupied_seats_in_business_class_l1219_121941


namespace NUMINAMATH_GPT_thermostat_range_l1219_121929

theorem thermostat_range (T : ℝ) : 
  |T - 22| ≤ 6 ↔ 16 ≤ T ∧ T ≤ 28 := 
by sorry

end NUMINAMATH_GPT_thermostat_range_l1219_121929


namespace NUMINAMATH_GPT_original_square_area_l1219_121952

theorem original_square_area :
  ∀ (a b : ℕ), 
  (a * a = 24 * 1 * 1 + b * b ∧ 
  ((∃ m n : ℕ, (a + b = m ∧ a - b = n ∧ m * n = 24) ∨ 
  (a + b = n ∧ a - b = m ∧ m * n = 24)))) →
  a * a = 25 :=
by
  sorry

end NUMINAMATH_GPT_original_square_area_l1219_121952


namespace NUMINAMATH_GPT_sum_of_cosines_bounds_l1219_121943

theorem sum_of_cosines_bounds (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π / 2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π / 2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ π / 2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ π / 2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ π / 2)
  (sum_sines_eq : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) : 
  2 ≤ Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ∧ 
      Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cosines_bounds_l1219_121943


namespace NUMINAMATH_GPT_restaurant_meals_l1219_121954

theorem restaurant_meals (k a : ℕ) (ratio_kids_to_adults : k / a = 10 / 7) (kids_meals_sold : k = 70) : a = 49 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_meals_l1219_121954


namespace NUMINAMATH_GPT_min_sum_of_factors_l1219_121925

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1176) :
  a + b + c ≥ 59 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l1219_121925


namespace NUMINAMATH_GPT_min_a_l1219_121972

theorem min_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (1/x + a/y) ≥ 25) : a ≥ 16 :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_min_a_l1219_121972


namespace NUMINAMATH_GPT_apples_and_oranges_l1219_121997

theorem apples_and_oranges :
  ∃ x y : ℝ, 2 * x + 3 * y = 6 ∧ 4 * x + 7 * y = 13 ∧ (16 * x + 23 * y = 47) :=
by
  sorry

end NUMINAMATH_GPT_apples_and_oranges_l1219_121997


namespace NUMINAMATH_GPT_largest_number_among_four_l1219_121993

theorem largest_number_among_four :
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  max a (max b (max c d)) = b := 
sorry

end NUMINAMATH_GPT_largest_number_among_four_l1219_121993


namespace NUMINAMATH_GPT_geom_sequence_a4_times_a7_l1219_121995

theorem geom_sequence_a4_times_a7 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_q : q = 2) 
  (h_a2_a5 : a 2 * a 5 = 32) : 
  a 4 * a 7 = 512 :=
by 
  sorry

end NUMINAMATH_GPT_geom_sequence_a4_times_a7_l1219_121995


namespace NUMINAMATH_GPT_blue_chip_value_l1219_121976

noncomputable def yellow_chip_value := 2
noncomputable def green_chip_value := 5
noncomputable def total_product_value := 16000
noncomputable def num_yellow_chips := 4

def blue_chip_points (b n : ℕ) :=
  yellow_chip_value ^ num_yellow_chips * b ^ n * green_chip_value ^ n = total_product_value

theorem blue_chip_value (b : ℕ) (n : ℕ) (h : blue_chip_points b n) (hn : b^n = 8) : b = 8 :=
by
  have h1 : ∀ k : ℕ, k ^ n = 8 → k = 8 ∧ n = 3 := sorry
  exact (h1 b hn).1

end NUMINAMATH_GPT_blue_chip_value_l1219_121976


namespace NUMINAMATH_GPT_equation_holds_true_l1219_121918

theorem equation_holds_true (a b : ℝ) (h₁ : a ≠ 0) (h₂ : 2 * b - a ≠ 0) :
  ((a + 2 * b) / a = b / (2 * b - a)) ↔ 
  (a = -b * (1 + Real.sqrt 17) / 2 ∨ a = -b * (1 - Real.sqrt 17) / 2) := 
sorry

end NUMINAMATH_GPT_equation_holds_true_l1219_121918


namespace NUMINAMATH_GPT_James_age_is_47_5_l1219_121975

variables (James_Age Mara_Age : ℝ)

def condition1 : Prop := James_Age = 3 * Mara_Age - 20
def condition2 : Prop := James_Age + Mara_Age = 70

theorem James_age_is_47_5 (h1 : condition1 James_Age Mara_Age) (h2 : condition2 James_Age Mara_Age) : James_Age = 47.5 :=
by
  sorry

end NUMINAMATH_GPT_James_age_is_47_5_l1219_121975


namespace NUMINAMATH_GPT_average_marks_l1219_121999

noncomputable def TatuyaScore (IvannaScore : ℝ) : ℝ :=
2 * IvannaScore

noncomputable def IvannaScore (DorothyScore : ℝ) : ℝ :=
(3/5) * DorothyScore

noncomputable def DorothyScore : ℝ := 90

noncomputable def XanderScore (TatuyaScore IvannaScore DorothyScore : ℝ) : ℝ :=
((TatuyaScore + IvannaScore + DorothyScore) / 3) + 10

noncomputable def SamScore (IvannaScore : ℝ) : ℝ :=
(3.8 * IvannaScore) + 5.5

noncomputable def OliviaScore (SamScore : ℝ) : ℝ :=
(3/2) * SamScore

theorem average_marks :
  let I := IvannaScore DorothyScore
  let T := TatuyaScore I
  let S := SamScore I
  let O := OliviaScore S
  let X := XanderScore T I DorothyScore
  let total_marks := T + I + DorothyScore + X + O + S
  (total_marks / 6) = 145.458333 := by sorry

end NUMINAMATH_GPT_average_marks_l1219_121999


namespace NUMINAMATH_GPT_total_problems_is_correct_l1219_121933

/-- Definition of the number of pages of math homework. -/
def math_pages : ℕ := 2

/-- Definition of the number of pages of reading homework. -/
def reading_pages : ℕ := 4

/-- Definition that each page of homework contains 5 problems. -/
def problems_per_page : ℕ := 5

/-- The proof statement: given the number of pages of math and reading homework,
    and the number of problems per page, prove that the total number of problems is 30. -/
theorem total_problems_is_correct : (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_GPT_total_problems_is_correct_l1219_121933


namespace NUMINAMATH_GPT_distance_to_intersection_of_quarter_circles_eq_zero_l1219_121903

open Real

theorem distance_to_intersection_of_quarter_circles_eq_zero (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let center := (s / 2, s / 2)
  let arc_from_A := {p : ℝ × ℝ | p.1^2 + p.2^2 = s^2}
  let arc_from_C := {p : ℝ × ℝ | (p.1 - s)^2 + (p.2 - s)^2 = s^2}
  (center ∈ arc_from_A ∧ center ∈ arc_from_C) →
  let (ix, iy) := (s / 2, s / 2)
  dist (ix, iy) center = 0 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_intersection_of_quarter_circles_eq_zero_l1219_121903


namespace NUMINAMATH_GPT_problem1_problem2_l1219_121913

variable (m n x y : ℝ)

theorem problem1 : 4 * m * n^3 * (2 * m^2 - (3 / 4) * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := sorry

theorem problem2 : (x - 6 * y^2) * (3 * x^3 + y) = 3 * x^4 + x * y - 18 * x^3 * y^2 - 6 * y^3 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1219_121913


namespace NUMINAMATH_GPT_range_of_a_l1219_121968

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - a| < 4) → -1 < a ∧ a < 7 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1219_121968


namespace NUMINAMATH_GPT_rectangle_width_length_ratio_l1219_121908

theorem rectangle_width_length_ratio (w l : ℕ) 
  (h1 : l = 12) 
  (h2 : 2 * w + 2 * l = 36) : 
  w / l = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_width_length_ratio_l1219_121908


namespace NUMINAMATH_GPT_black_to_white_ratio_l1219_121916

/-- 
Given:
- The original square pattern consists of 13 black tiles and 23 white tiles
- Attaching a border of black tiles around the original 6x6 square pattern results in an 8x8 square pattern

To prove:
- The ratio of black tiles to white tiles in the extended 8x8 pattern is 41/23.
-/
theorem black_to_white_ratio (b_orig w_orig b_added b_total w_total : ℕ) 
  (h_black_orig: b_orig = 13)
  (h_white_orig: w_orig = 23)
  (h_size_orig: 6 * 6 = b_orig + w_orig)
  (h_size_ext: 8 * 8 = (b_orig + b_added) + w_orig)
  (h_b_added: b_added = 28)
  (h_b_total: b_total = b_orig + b_added)
  (h_w_total: w_total = w_orig)
  :
  b_total / w_total = 41 / 23 :=
by
  sorry

end NUMINAMATH_GPT_black_to_white_ratio_l1219_121916


namespace NUMINAMATH_GPT_decreasing_power_function_l1219_121937

open Nat

/-- For the power function y = x^(m^2 + 2*m - 3) (where m : ℕ) 
    to be a decreasing function in the interval (0, +∞), prove that m = 0. -/
theorem decreasing_power_function (m : ℕ) (h : m^2 + 2 * m - 3 < 0) : m = 0 := 
by
  sorry

end NUMINAMATH_GPT_decreasing_power_function_l1219_121937


namespace NUMINAMATH_GPT_solve_equation_l1219_121960

variable {x y : ℝ}

theorem solve_equation (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2: y ≠ 4) (h : (3 / x) + (2 / y) = 5 / 6) :
  x = 18 * y / (5 * y - 12) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1219_121960


namespace NUMINAMATH_GPT_average_speed_of_train_l1219_121987

theorem average_speed_of_train
  (d1 d2 : ℝ) (t1 t2 : ℝ)
  (h1 : d1 = 290) (h2 : d2 = 400) (h3 : t1 = 4.5) (h4 : t2 = 5.5) :
  ((d1 + d2) / (t1 + t2)) = 69 :=
by
  -- proof steps can be filled in later
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l1219_121987


namespace NUMINAMATH_GPT_cos_eight_arccos_one_fourth_l1219_121969

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1 / 4)) = 172546 / 1048576 :=
sorry

end NUMINAMATH_GPT_cos_eight_arccos_one_fourth_l1219_121969


namespace NUMINAMATH_GPT_cakes_bought_l1219_121951

theorem cakes_bought (initial_cakes remaining_cakes : ℕ) (h_initial : initial_cakes = 155) (h_remaining : remaining_cakes = 15) : initial_cakes - remaining_cakes = 140 :=
by {
  sorry
}

end NUMINAMATH_GPT_cakes_bought_l1219_121951


namespace NUMINAMATH_GPT_trajectory_of_A_l1219_121992

theorem trajectory_of_A (A B C : (ℝ × ℝ)) (x y : ℝ) : 
  B = (-2, 0) ∧ C = (2, 0) ∧ (dist A (0, 0) = 3) → 
  (x, y) = A → 
  x^2 + y^2 = 9 ∧ y ≠ 0 := 
sorry

end NUMINAMATH_GPT_trajectory_of_A_l1219_121992


namespace NUMINAMATH_GPT_ship_distances_l1219_121970

-- Define the conditions based on the initial problem statement
variables (f : ℕ → ℝ)
def distances_at_known_times : Prop :=
  f 0 = 49 ∧ f 2 = 25 ∧ f 3 = 121

-- Define the questions to prove the distances at unknown times
def distance_at_time_1 : Prop :=
  f 1 = 1

def distance_at_time_4 : Prop :=
  f 4 = 289

-- The proof problem
theorem ship_distances
  (f : ℕ → ℝ)
  (hf : ∀ t, ∃ a b c, f t = a*t^2 + b*t + c)
  (h_known : distances_at_known_times f) :
  distance_at_time_1 f ∧ distance_at_time_4 f :=
by
  sorry

end NUMINAMATH_GPT_ship_distances_l1219_121970


namespace NUMINAMATH_GPT_select_numbers_with_sum_713_l1219_121930

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end NUMINAMATH_GPT_select_numbers_with_sum_713_l1219_121930


namespace NUMINAMATH_GPT_bird_costs_l1219_121974

-- Define the cost of a small bird and a large bird
def cost_small_bird (x : ℕ) := x
def cost_large_bird (x : ℕ) := 2 * x

-- Define total cost calculations for the first and second ladies
def cost_first_lady (x : ℕ) := 5 * cost_large_bird x + 3 * cost_small_bird x
def cost_second_lady (x : ℕ) := 5 * cost_small_bird x + 3 * cost_large_bird x

-- State the main theorem
theorem bird_costs (x : ℕ) (hx : cost_first_lady x = cost_second_lady x + 20) : 
(cost_small_bird x = 10) ∧ (cost_large_bird x = 20) := 
by {
  sorry
}

end NUMINAMATH_GPT_bird_costs_l1219_121974


namespace NUMINAMATH_GPT_angle_between_lines_in_folded_rectangle_l1219_121935

theorem angle_between_lines_in_folded_rectangle
  (a b : ℝ) 
  (h : b > a)
  (dihedral_angle : ℝ)
  (h_dihedral_angle : dihedral_angle = 18) :
  ∃ (angle_AC_MN : ℝ), angle_AC_MN = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_lines_in_folded_rectangle_l1219_121935


namespace NUMINAMATH_GPT_value_of_abc_l1219_121901

noncomputable def f (x a b c : ℝ) := |(1 - x^2) * (x^2 + a * x + b)| - c

theorem value_of_abc :
  (∀ x : ℝ, f (x + 4) 8 15 9 = f (-x) 8 15 9) ∧
  (∃ x : ℝ, f x 8 15 9 = 0) ∧
  (∃ x : ℝ, f (-(x-4)) 8 15 9 = 0) ∧
  (∀ c : ℝ, c ≠ 0) →
  8 + 15 + 9 = 32 :=
by sorry

end NUMINAMATH_GPT_value_of_abc_l1219_121901


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1219_121994

theorem bus_speed_excluding_stoppages 
  (V : ℝ) -- Denote the average speed excluding stoppages as V
  (h1 : 30 / 1 = 30) -- condition 1: average speed including stoppages is 30 km/hr
  (h2 : 1 / 2 = 0.5) -- condition 2: The bus is moving for 0.5 hours per hour due to 30 min stoppage
  (h3 : V = 2 * 30) -- from the condition that the bus must cover the distance in half the time
  : V = 60 :=
by {
  sorry -- proof is not required
}

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1219_121994


namespace NUMINAMATH_GPT_problem_l1219_121953

theorem problem : 3^128 + 8^5 / 8^3 = 65 := sorry

end NUMINAMATH_GPT_problem_l1219_121953


namespace NUMINAMATH_GPT_correct_operation_l1219_121919

theorem correct_operation (a : ℝ) : a^8 / a^2 = a^6 :=
by
  -- proof will go here, let's use sorry to indicate it's unfinished
  sorry

end NUMINAMATH_GPT_correct_operation_l1219_121919


namespace NUMINAMATH_GPT_evaluate_polynomial_l1219_121950

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : 0 < x) : 
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = -8 := 
sorry

end NUMINAMATH_GPT_evaluate_polynomial_l1219_121950


namespace NUMINAMATH_GPT_discriminant_eq_complete_square_form_l1219_121980

theorem discriminant_eq_complete_square_form (a b c t : ℝ) (h : a ≠ 0) (ht : a * t^2 + b * t + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 := 
sorry

end NUMINAMATH_GPT_discriminant_eq_complete_square_form_l1219_121980


namespace NUMINAMATH_GPT_cone_volume_from_half_sector_l1219_121948

theorem cone_volume_from_half_sector (r l : ℝ) (h : ℝ) 
    (h_r : r = 3) (h_l : l = 6) (h_h : h = 3 * Real.sqrt 3) : 
    (1 / 3) * Real.pi * r^2 * h = 9 * Real.pi * Real.sqrt 3 := 
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_cone_volume_from_half_sector_l1219_121948


namespace NUMINAMATH_GPT_total_books_borrowed_lunchtime_correct_l1219_121978

def shelf_A_borrowed (X : ℕ) : Prop :=
  110 - X = 60 ∧ X = 50

def shelf_B_borrowed (Y : ℕ) : Prop :=
  150 - 50 + 20 - Y = 80 ∧ Y = 80

def shelf_C_borrowed (Z : ℕ) : Prop :=
  210 - 45 = 165 ∧ 165 - 130 = Z ∧ Z = 35

theorem total_books_borrowed_lunchtime_correct :
  ∃ (X Y Z : ℕ),
    shelf_A_borrowed X ∧
    shelf_B_borrowed Y ∧
    shelf_C_borrowed Z ∧
    X + Y + Z = 165 :=
by
  sorry

end NUMINAMATH_GPT_total_books_borrowed_lunchtime_correct_l1219_121978


namespace NUMINAMATH_GPT_solution_set_inequalities_l1219_121946

theorem solution_set_inequalities (a b x : ℝ) (h1 : ∃ x, x > a ∧ x < b) :
  (x < 1 - a ∧ x < 1 - b) ↔ x < 1 - b :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequalities_l1219_121946


namespace NUMINAMATH_GPT_cost_per_tshirt_l1219_121947
-- Import necessary libraries

-- Define the given conditions
def t_shirts : ℕ := 20
def total_cost : ℝ := 199

-- Define the target proof statement
theorem cost_per_tshirt : (total_cost / t_shirts) = 9.95 := 
sorry

end NUMINAMATH_GPT_cost_per_tshirt_l1219_121947


namespace NUMINAMATH_GPT_max_area_of_garden_l1219_121963

theorem max_area_of_garden (total_fence : ℝ) (gate : ℝ) (remaining_fence := total_fence - gate) :
  total_fence = 60 → gate = 4 → (remaining_fence / 2) * (remaining_fence / 2) = 196 :=
by 
  sorry

end NUMINAMATH_GPT_max_area_of_garden_l1219_121963


namespace NUMINAMATH_GPT_initial_paint_amount_l1219_121955

theorem initial_paint_amount (P : ℝ) (h1 : P > 0) (h2 : (1 / 4) * P + (1 / 3) * (3 / 4) * P = 180) : P = 360 := by
  sorry

end NUMINAMATH_GPT_initial_paint_amount_l1219_121955


namespace NUMINAMATH_GPT_evaluate_expression_l1219_121902

variable {R : Type} [CommRing R]

theorem evaluate_expression (x y z w : R) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1219_121902


namespace NUMINAMATH_GPT_suitcase_combinations_l1219_121907

def count_odd_numbers (n : Nat) : Nat := n / 2

def count_multiples_of_4 (n : Nat) : Nat := n / 4

def count_multiples_of_5 (n : Nat) : Nat := n / 5

theorem suitcase_combinations : count_odd_numbers 40 * count_multiples_of_4 40 * count_multiples_of_5 40 = 1600 :=
by
  sorry

end NUMINAMATH_GPT_suitcase_combinations_l1219_121907


namespace NUMINAMATH_GPT_yoongi_hoseok_age_sum_l1219_121965

-- Definitions of given conditions
def age_aunt : ℕ := 38
def diff_aunt_yoongi : ℕ := 23
def diff_yoongi_hoseok : ℕ := 4

-- Definitions related to ages of Yoongi and Hoseok derived from given conditions
def age_yoongi : ℕ := age_aunt - diff_aunt_yoongi
def age_hoseok : ℕ := age_yoongi - diff_yoongi_hoseok

-- The theorem we need to prove
theorem yoongi_hoseok_age_sum : age_yoongi + age_hoseok = 26 := by
  sorry

end NUMINAMATH_GPT_yoongi_hoseok_age_sum_l1219_121965


namespace NUMINAMATH_GPT_nurses_count_l1219_121900

theorem nurses_count (total personnel_ratio d_ratio n_ratio : ℕ)
  (ratio_eq: personnel_ratio = 280)
  (ratio_condition: d_ratio = 5)
  (person_count: n_ratio = 9) :
  n_ratio * (personnel_ratio / (d_ratio + n_ratio)) = 180 := by
  -- Total personnel = 280
  -- Ratio of doctors to nurses = 5/9
  -- Prove that the number of nurses is 180
  -- sorry is used to skip proof
  sorry

end NUMINAMATH_GPT_nurses_count_l1219_121900


namespace NUMINAMATH_GPT_ratio_y_to_x_l1219_121926

-- Definitions based on conditions
variable (c : ℝ) -- Cost price
def x : ℝ := 0.8 * c -- Selling price for a loss of 20%
def y : ℝ := 1.25 * c -- Selling price for a gain of 25%

-- Statement to prove the ratio of y to x
theorem ratio_y_to_x : y / x = 25 / 16 := by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_ratio_y_to_x_l1219_121926


namespace NUMINAMATH_GPT_books_left_to_read_l1219_121979

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_books_left_to_read_l1219_121979


namespace NUMINAMATH_GPT_contractor_engaged_days_l1219_121906

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_contractor_engaged_days_l1219_121906


namespace NUMINAMATH_GPT_abs_lt_inequality_solution_l1219_121928

theorem abs_lt_inequality_solution (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_abs_lt_inequality_solution_l1219_121928


namespace NUMINAMATH_GPT_machine_rate_ratio_l1219_121904

theorem machine_rate_ratio (A B : ℕ) (h1 : ∃ A : ℕ, 8 * A = 8 * A)
  (h2 : ∃ W : ℕ, W = 8 * A)
  (h3 : ∃ W1 : ℕ, W1 = 6 * A)
  (h4 : ∃ W2 : ℕ, W2 = 2 * A)
  (h5 : ∃ B : ℕ, 8 * B = 2 * A) :
  (B:ℚ) / (A:ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_machine_rate_ratio_l1219_121904


namespace NUMINAMATH_GPT_jason_car_cost_l1219_121942

theorem jason_car_cost
    (down_payment : ℕ := 8000)
    (monthly_payment : ℕ := 525)
    (months : ℕ := 48)
    (interest_rate : ℝ := 0.05) :
    (down_payment + monthly_payment * months + interest_rate * (monthly_payment * months)) = 34460 := 
by
  sorry

end NUMINAMATH_GPT_jason_car_cost_l1219_121942


namespace NUMINAMATH_GPT_original_price_of_suit_l1219_121920

theorem original_price_of_suit (P : ℝ) (hP : 0.70 * 1.30 * P = 182) : P = 200 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_suit_l1219_121920


namespace NUMINAMATH_GPT_kelsey_remaining_half_speed_l1219_121932

variable (total_hours : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (remaining_half_time : ℝ) (remaining_half_distance : ℝ)

axiom h1 : total_hours = 10
axiom h2 : first_half_speed = 25
axiom h3 : total_distance = 400
axiom h4 : remaining_half_time = total_hours - total_distance / (2 * first_half_speed)
axiom h5 : remaining_half_distance = total_distance / 2

theorem kelsey_remaining_half_speed :
  remaining_half_distance / remaining_half_time = 100
:=
by
  sorry

end NUMINAMATH_GPT_kelsey_remaining_half_speed_l1219_121932


namespace NUMINAMATH_GPT_time_saved_is_six_minutes_l1219_121922

-- Conditions
def distance_monday : ℝ := 3
def distance_wednesday : ℝ := 4
def distance_friday : ℝ := 5

def speed_monday : ℝ := 6
def speed_wednesday : ℝ := 4
def speed_friday : ℝ := 5

def speed_constant : ℝ := 5

-- Question (proof statement)
theorem time_saved_is_six_minutes : 
  (distance_monday / speed_monday + distance_wednesday / speed_wednesday + distance_friday / speed_friday) - (distance_monday + distance_wednesday + distance_friday) / speed_constant = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_time_saved_is_six_minutes_l1219_121922


namespace NUMINAMATH_GPT_original_price_l1219_121989

theorem original_price (x: ℝ) (h1: x * 1.1 * 0.8 = 2) : x = 25 / 11 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1219_121989


namespace NUMINAMATH_GPT_allocation_schemes_for_5_teachers_to_3_buses_l1219_121996

noncomputable def number_of_allocation_schemes (teachers : ℕ) (buses : ℕ) : ℕ :=
  if buses = 3 ∧ teachers = 5 then 150 else 0

theorem allocation_schemes_for_5_teachers_to_3_buses : 
  number_of_allocation_schemes 5 3 = 150 := 
by
  sorry

end NUMINAMATH_GPT_allocation_schemes_for_5_teachers_to_3_buses_l1219_121996


namespace NUMINAMATH_GPT_no_valid_conference_division_l1219_121945

theorem no_valid_conference_division (num_teams : ℕ) (matches_per_team : ℕ) :
  num_teams = 30 → matches_per_team = 82 → 
  ¬ ∃ (k : ℕ) (x y z : ℕ), k + (num_teams - k) = num_teams ∧
                          x + y + z = (num_teams * matches_per_team) / 2 ∧
                          z = ((x + y + z) / 2) := 
by
  sorry

end NUMINAMATH_GPT_no_valid_conference_division_l1219_121945


namespace NUMINAMATH_GPT_intersection_of_sets_l1219_121964

variable (A B : Set ℝ) (x : ℝ)

def setA : Set ℝ := { x | x > 0 }
def setB : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1219_121964


namespace NUMINAMATH_GPT_ratio_problem_l1219_121940

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 2 / 1)
  (h1 : B / C = 1 / 4) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := 
sorry

end NUMINAMATH_GPT_ratio_problem_l1219_121940


namespace NUMINAMATH_GPT_find_principal_l1219_121938

variable (P R : ℝ)
variable (condition1 : P + (P * R * 2) / 100 = 660)
variable (condition2 : P + (P * R * 7) / 100 = 1020)

theorem find_principal : P = 516 := by
  sorry

end NUMINAMATH_GPT_find_principal_l1219_121938


namespace NUMINAMATH_GPT_greatest_integer_value_l1219_121990

theorem greatest_integer_value (x : ℤ) : ∃ x, (∀ y, (x^2 + 2 * x + 10) % (x - 3) = 0 → x ≥ y) → x = 28 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_value_l1219_121990


namespace NUMINAMATH_GPT_paul_bags_on_saturday_l1219_121912

-- Definitions and Conditions
def total_cans : ℕ := 72
def cans_per_bag : ℕ := 8
def extra_bags : ℕ := 3

-- Statement of the problem
theorem paul_bags_on_saturday (S : ℕ) :
  S * cans_per_bag = total_cans - (extra_bags * cans_per_bag) →
  S = 6 :=
sorry

end NUMINAMATH_GPT_paul_bags_on_saturday_l1219_121912


namespace NUMINAMATH_GPT_solve_for_y_l1219_121927

theorem solve_for_y 
  (a b c d y : ℚ) 
  (h₀ : a ≠ b) 
  (h₁ : a ≠ 0) 
  (h₂ : c ≠ d) 
  (h₃ : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1219_121927


namespace NUMINAMATH_GPT_charlie_original_price_l1219_121909

theorem charlie_original_price (acorns_Alice acorns_Bob acorns_Charlie ν_Alice ν_Bob discount price_Charlie_before_discount price_Charlie_after_discount total_paid_by_AliceBob total_acorns_AliceBob average_price_per_acorn price_per_acorn_Alice price_per_acorn_Bob total_paid_Alice total_paid_Bob: ℝ) :
  acorns_Alice = 3600 →
  acorns_Bob = 2400 →
  acorns_Charlie = 4500 →
  ν_Bob = 6000 →
  ν_Alice = 9 * ν_Bob →
  price_per_acorn_Bob = ν_Bob / acorns_Bob →
  price_per_acorn_Alice = ν_Alice / acorns_Alice →
  total_paid_Alice = acorns_Alice * price_per_acorn_Alice →
  total_paid_Bob = ν_Bob →
  total_paid_by_AliceBob = total_paid_Alice + total_paid_Bob →
  total_acorns_AliceBob = acorns_Alice + acorns_Bob →
  average_price_per_acorn = total_paid_by_AliceBob / total_acorns_AliceBob →
  discount = 10 / 100 →
  price_Charlie_after_discount = average_price_per_acorn * (1 - discount) →
  price_Charlie_before_discount = average_price_per_acorn →
  price_Charlie_before_discount = 14.50 →
  price_per_acorn_Alice = 22.50 →
  price_Charlie_before_discount * acorns_Charlie = 4500 * 14.50 :=
by sorry

end NUMINAMATH_GPT_charlie_original_price_l1219_121909


namespace NUMINAMATH_GPT_skier_total_time_l1219_121924

variable (t1 t2 t3 : ℝ)

-- Conditions
def condition1 : Prop := t1 + t2 = 40.5
def condition2 : Prop := t2 + t3 = 37.5
def condition3 : Prop := 1 / t2 = 2 / (t1 + t3)

-- Theorem to prove total time is 58.5 minutes
theorem skier_total_time (h1 : condition1 t1 t2) (h2 : condition2 t2 t3) (h3 : condition3 t1 t2 t3) : t1 + t2 + t3 = 58.5 := 
by
  sorry

end NUMINAMATH_GPT_skier_total_time_l1219_121924


namespace NUMINAMATH_GPT_largest_prime_factor_of_1729_l1219_121966

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_1729_l1219_121966


namespace NUMINAMATH_GPT_num_audio_cassettes_in_second_set_l1219_121991

-- Define the variables and constants
def costOfAudio (A : ℕ) : ℕ := A
def costOfVideo (V : ℕ) : ℕ := V
def totalCost (numOfAudio : ℕ) (numOfVideo : ℕ) (A : ℕ) (V : ℕ) : ℕ :=
  numOfAudio * (costOfAudio A) + numOfVideo * (costOfVideo V)

-- Given conditions
def condition1 (A V : ℕ) : Prop := ∃ X : ℕ, totalCost X 4 A V = 1350
def condition2 (A V : ℕ) : Prop := totalCost 7 3 A V = 1110
def condition3 : Prop := costOfVideo 300 = 300

-- Main theorem to prove: The number of audio cassettes in the second set is 7
theorem num_audio_cassettes_in_second_set :
  ∃ (A : ℕ), condition1 A 300 ∧ condition2 A 300 ∧ condition3 →
  7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_audio_cassettes_in_second_set_l1219_121991


namespace NUMINAMATH_GPT_optimal_selling_price_l1219_121986

-- Define the constants given in the problem
def purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 50

-- Define the function that represents the profit based on the change in price x
def profit (x : ℝ) : ℝ := (initial_selling_price + x) * (initial_sales_volume - x) - (initial_sales_volume - x) * purchase_price

-- State the theorem
theorem optimal_selling_price : ∃ x : ℝ, profit x = -x^2 + 40*x + 500 ∧ (initial_selling_price + x = 70) :=
by
  sorry

end NUMINAMATH_GPT_optimal_selling_price_l1219_121986


namespace NUMINAMATH_GPT_decrease_in_combined_area_l1219_121967

theorem decrease_in_combined_area (r1 r2 r3 : ℝ) :
    let π := Real.pi
    let A_original := π * (r1 ^ 2) + π * (r2 ^ 2) + π * (r3 ^ 2)
    let r1' := r1 * 0.5
    let r2' := r2 * 0.5
    let r3' := r3 * 0.5
    let A_new := π * (r1' ^ 2) + π * (r2' ^ 2) + π * (r3' ^ 2)
    let Decrease := A_original - A_new
    Decrease = 0.75 * π * (r1 ^ 2) + 0.75 * π * (r2 ^ 2) + 0.75 * π * (r3 ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_decrease_in_combined_area_l1219_121967


namespace NUMINAMATH_GPT_symmetric_circle_eq_of_given_circle_eq_l1219_121962

theorem symmetric_circle_eq_of_given_circle_eq
  (x y : ℝ)
  (eq1 : (x - 1)^2 + (y - 2)^2 = 1)
  (line_eq : y = x) :
  (x - 2)^2 + (y - 1)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_of_given_circle_eq_l1219_121962


namespace NUMINAMATH_GPT_eq_positive_root_a_value_l1219_121958

theorem eq_positive_root_a_value (x a : ℝ) (hx : x > 0) :
  ((x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 :=
by
  sorry

end NUMINAMATH_GPT_eq_positive_root_a_value_l1219_121958


namespace NUMINAMATH_GPT_lilibeth_and_friends_strawberries_l1219_121957

-- Define the conditions
def baskets_filled_by_lilibeth : ℕ := 6
def strawberries_per_basket : ℕ := 50
def friends_count : ℕ := 3

-- Define the total number of strawberries picked by Lilibeth and her friends 
def total_strawberries_picked : ℕ :=
  (baskets_filled_by_lilibeth * strawberries_per_basket) * (1 + friends_count)

-- The theorem to prove
theorem lilibeth_and_friends_strawberries : total_strawberries_picked = 1200 := 
by
  sorry

end NUMINAMATH_GPT_lilibeth_and_friends_strawberries_l1219_121957


namespace NUMINAMATH_GPT_interior_lattice_points_of_triangle_l1219_121936

-- Define the vertices of the triangle
def A : (ℤ × ℤ) := (0, 99)
def B : (ℤ × ℤ) := (5, 100)
def C : (ℤ × ℤ) := (2003, 500)

-- The problem is to find the number of interior lattice points
-- according to Pick's Theorem (excluding boundary points).

theorem interior_lattice_points_of_triangle :
  let I : ℤ := 0 -- number of interior lattice points
  I = 0 :=
by
  sorry

end NUMINAMATH_GPT_interior_lattice_points_of_triangle_l1219_121936


namespace NUMINAMATH_GPT_proof_m_plus_n_l1219_121910

variable (m n : ℚ) -- Defining m and n as rational numbers (ℚ)
-- Conditions from the problem:
axiom condition1 : 2 * m + 5 * n + 8 = 1
axiom condition2 : m - n - 3 = 1

-- Proof statement (theorem) that needs to be established:
theorem proof_m_plus_n : m + n = -2/7 :=
by
-- Since the proof is not required, we use "sorry" to placeholder the proof.
sorry

end NUMINAMATH_GPT_proof_m_plus_n_l1219_121910


namespace NUMINAMATH_GPT_fraction_simplification_l1219_121985

theorem fraction_simplification :
    1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1219_121985


namespace NUMINAMATH_GPT_unique_records_l1219_121934

variable (Samantha_records : Nat)
variable (shared_records : Nat)
variable (Lily_unique_records : Nat)

theorem unique_records (h1 : Samantha_records = 24) (h2 : shared_records = 15) (h3 : Lily_unique_records = 9) :
  let Samantha_unique_records := Samantha_records - shared_records
  Samantha_unique_records + Lily_unique_records = 18 :=
by
  sorry

end NUMINAMATH_GPT_unique_records_l1219_121934


namespace NUMINAMATH_GPT_intersection_is_empty_l1219_121905

def A : Set ℝ := { α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3 }
def B : Set ℝ := { β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2 }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_empty_l1219_121905


namespace NUMINAMATH_GPT_mass_increase_l1219_121983

theorem mass_increase (ρ₁ ρ₂ m₁ m₂ a₁ a₂ : ℝ) (cond1 : ρ₂ = 2 * ρ₁) 
                      (cond2 : a₂ = 2 * a₁) (cond3 : m₁ = ρ₁ * (a₁^3)) 
                      (cond4 : m₂ = ρ₂ * (a₂^3)) : 
                      ((m₂ - m₁) / m₁) * 100 = 1500 := by
  sorry

end NUMINAMATH_GPT_mass_increase_l1219_121983


namespace NUMINAMATH_GPT_find_x_l1219_121988

variables (a b c k : ℝ) (h : k ≠ 0)

theorem find_x (x y z : ℝ)
  (h1 : (xy + k) / (x + y) = a)
  (h2 : (xz + k) / (x + z) = b)
  (h3 : (yz + k) / (y + z) = c) :
  x = 2 * a * b * c * d / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) := sorry

end NUMINAMATH_GPT_find_x_l1219_121988


namespace NUMINAMATH_GPT_sequence_unique_integers_l1219_121921

theorem sequence_unique_integers (a : ℕ → ℤ) 
  (H_inf_pos : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n > N) 
  (H_inf_neg : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n < N)
  (H_diff_remainders : ∀ n : ℕ, n > 0 → ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) → (1 ≤ j ∧ j ≤ n) → i ≠ j → (a i % ↑n) ≠ (a j % ↑n)) :
  ∀ m : ℤ, ∃! n : ℕ, a n = m := sorry

end NUMINAMATH_GPT_sequence_unique_integers_l1219_121921


namespace NUMINAMATH_GPT_rotate_D_90_clockwise_l1219_121914

-- Define the point D with its coordinates.
structure Point where
  x : Int
  y : Int

-- Define the original point D.
def D : Point := { x := -3, y := -8 }

-- Define the rotation transformation.
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Statement to be proven.
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = { x := -8, y := 3 } :=
sorry

end NUMINAMATH_GPT_rotate_D_90_clockwise_l1219_121914


namespace NUMINAMATH_GPT_height_difference_l1219_121956

theorem height_difference (B_height A_height : ℝ) (h : A_height = 0.6 * B_height) :
  (B_height - A_height) / A_height * 100 = 66.67 := 
sorry

end NUMINAMATH_GPT_height_difference_l1219_121956


namespace NUMINAMATH_GPT_simplify_complex_expr_l1219_121931

theorem simplify_complex_expr : 
  ∀ (i : ℂ), i^2 = -1 → ( (2 + 4 * i) / (2 - 4 * i) - (2 - 4 * i) / (2 + 4 * i) )
  = -8/5 + (16/5 : ℂ) * i :=
by
  intro i h_i_squared
  sorry

end NUMINAMATH_GPT_simplify_complex_expr_l1219_121931


namespace NUMINAMATH_GPT_y_value_when_x_neg_one_l1219_121981

theorem y_value_when_x_neg_one (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = t^2 + 3 * t + 6) 
  (h3 : x = -1) : 
  y = 16 := 
by sorry

end NUMINAMATH_GPT_y_value_when_x_neg_one_l1219_121981


namespace NUMINAMATH_GPT_point_on_x_axis_l1219_121939

theorem point_on_x_axis (a : ℝ) (h : (1, a + 1).snd = 0) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1219_121939


namespace NUMINAMATH_GPT_linear_function_does_not_pass_first_quadrant_l1219_121971

theorem linear_function_does_not_pass_first_quadrant (k b : ℝ) (h : ∀ x : ℝ, y = k * x + b) :
  k = -1 → b = -2 → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
by
  sorry

end NUMINAMATH_GPT_linear_function_does_not_pass_first_quadrant_l1219_121971


namespace NUMINAMATH_GPT_daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l1219_121973

-- Definitions based on given conditions
noncomputable def purchase_price : ℝ := 30
noncomputable def max_selling_price : ℝ := 55
noncomputable def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 140

-- Definition of daily profit based on selling price x
noncomputable def daily_profit (x : ℝ) : ℝ := (x - purchase_price) * daily_sales_volume x

-- Lean 4 statements for the proofs
theorem daily_profit_at_35_yuan : daily_profit 35 = 350 := sorry

theorem selling_price_for_600_profit : ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ daily_profit x = 600 ∧ x = 40 := sorry

theorem selling_price_impossible_for_900_profit :
  ∀ x, 30 ≤ x ∧ x ≤ 55 → daily_profit x ≠ 900 := sorry

end NUMINAMATH_GPT_daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l1219_121973


namespace NUMINAMATH_GPT_transistors_in_2010_l1219_121977

theorem transistors_in_2010 
  (initial_transistors : ℕ) 
  (initial_year : ℕ) 
  (final_year : ℕ) 
  (doubling_period : ℕ)
  (initial_transistors_eq: initial_transistors = 500000)
  (initial_year_eq: initial_year = 1985)
  (final_year_eq: final_year = 2010)
  (doubling_period_eq : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 2048000000 := 
by 
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_transistors_in_2010_l1219_121977


namespace NUMINAMATH_GPT_product_quantities_l1219_121917

theorem product_quantities (a b x y : ℝ) 
  (h1 : a * x + b * y = 1500)
  (h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529)
  (h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5)
  (h4 : 205 < 2 * x + y ∧ 2 * x + y < 210) :
  (x + 2 * y = 186) ∧ (73 ≤ x ∧ x ≤ 75) :=
by
  sorry

end NUMINAMATH_GPT_product_quantities_l1219_121917


namespace NUMINAMATH_GPT_nim_maximum_product_l1219_121982

def nim_max_product (x y : ℕ) : ℕ :=
43 * 99 * x * y

theorem nim_maximum_product :
  ∃ x y : ℕ, (43 ≠ 0) ∧ (99 ≠ 0) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧
  (43 + 99 + x + y = 0) ∧ (nim_max_product x y = 7704) :=
sorry

end NUMINAMATH_GPT_nim_maximum_product_l1219_121982


namespace NUMINAMATH_GPT_volume_of_prism_l1219_121915

variable (x y z : ℝ)
variable (h1 : x * y = 15)
variable (h2 : y * z = 10)
variable (h3 : x * z = 6)

theorem volume_of_prism : x * y * z = 30 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1219_121915


namespace NUMINAMATH_GPT_percent_equivalence_l1219_121998

theorem percent_equivalence (y : ℝ) : 0.30 * (0.60 * y) = 0.18 * y :=
by sorry

end NUMINAMATH_GPT_percent_equivalence_l1219_121998


namespace NUMINAMATH_GPT_kaleb_toys_can_buy_l1219_121961

theorem kaleb_toys_can_buy (saved_money : ℕ) (allowance_received : ℕ) (allowance_increase_percent : ℕ) (toy_cost : ℕ) (half_total_spend : ℕ) :
  saved_money = 21 →
  allowance_received = 15 →
  allowance_increase_percent = 20 →
  toy_cost = 6 →
  half_total_spend = (saved_money + allowance_received) / 2 →
  (half_total_spend / toy_cost) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_kaleb_toys_can_buy_l1219_121961


namespace NUMINAMATH_GPT_find_xyz_sum_cube_l1219_121984

variable (x y z c d : ℝ) 

theorem find_xyz_sum_cube (h1 : x * y * z = c) (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d := 
by
  sorry

end NUMINAMATH_GPT_find_xyz_sum_cube_l1219_121984
