import Mathlib

namespace simplify_abs_expression_l1727_172741

theorem simplify_abs_expression (x : ℝ) : 
  |2*x + 1| - |x - 3| + |x - 6| = 
  if x < -1/2 then -2*x + 2 
  else if x < 3 then 2*x + 4 
  else if x < 6 then 10 
  else 2*x - 2 :=
by 
  sorry

end simplify_abs_expression_l1727_172741


namespace maximize_profit_correct_l1727_172716

noncomputable def maximize_profit : ℝ × ℝ :=
  let initial_selling_price : ℝ := 50
  let purchase_price : ℝ := 40
  let initial_sales_volume : ℝ := 500
  let sales_volume_decrease_rate : ℝ := 10
  let x := 20
  let optimal_selling_price := initial_selling_price + x
  let maximum_profit := -10 * x^2 + 400 * x + 5000
  (optimal_selling_price, maximum_profit)

theorem maximize_profit_correct :
  maximize_profit = (70, 9000) :=
  sorry

end maximize_profit_correct_l1727_172716


namespace root_interval_exists_l1727_172753

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x + 1

theorem root_interval_exists :
  (f 2 > 0) →
  (f 3 < 0) →
  ∃ ξ, 2 < ξ ∧ ξ < 3 ∧ f ξ = 0 :=
by
  intros h1 h2
  sorry

end root_interval_exists_l1727_172753


namespace small_seats_capacity_l1727_172737

-- Definitions
def num_small_seats : ℕ := 2
def people_per_small_seat : ℕ := 14

-- Statement to prove
theorem small_seats_capacity :
  num_small_seats * people_per_small_seat = 28 :=
by
  -- Proof goes here
  sorry

end small_seats_capacity_l1727_172737


namespace grasshopper_frog_jump_difference_l1727_172725

theorem grasshopper_frog_jump_difference :
  let grasshopper_jump := 19
  let frog_jump := 15
  grasshopper_jump - frog_jump = 4 :=
by
  let grasshopper_jump := 19
  let frog_jump := 15
  sorry

end grasshopper_frog_jump_difference_l1727_172725


namespace infinite_pairs_exists_l1727_172779

noncomputable def exists_infinite_pairs : Prop :=
  ∃ (a b : ℕ), (a + b ∣ a * b + 1) ∧ (a - b ∣ a * b - 1) ∧ b > 1 ∧ a > b * Real.sqrt 3 - 1

theorem infinite_pairs_exists : ∃ (count : ℕ) (a b : ℕ), ∀ n < count, exists_infinite_pairs :=
sorry

end infinite_pairs_exists_l1727_172779


namespace integer_satisfaction_l1727_172736

theorem integer_satisfaction (x : ℤ) : 
  (x + 15 ≥ 16 ∧ -3 * x ≥ -15) ↔ (1 ≤ x ∧ x ≤ 5) :=
by 
  sorry

end integer_satisfaction_l1727_172736


namespace rental_difference_l1727_172724

variable (C K : ℕ)

theorem rental_difference
  (hc : 15 * C + 18 * K = 405)
  (hr : 3 * K = 2 * C) :
  C - K = 5 :=
sorry

end rental_difference_l1727_172724


namespace equalSumSeqDefinition_l1727_172769

def isEqualSumSeq (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n - 1) + s n = s (n + 1)

theorem equalSumSeqDefinition (s : ℕ → ℝ) :
  isEqualSumSeq s ↔ 
  ∀ n : ℕ, n > 0 → s n = s (n - 1) + s (n + 1) :=
by
  sorry

end equalSumSeqDefinition_l1727_172769


namespace mia_has_largest_final_value_l1727_172791

def daniel_final : ℕ := (12 * 2 - 3 + 5)
def mia_final : ℕ := ((15 - 2) * 2 + 3)
def carlos_final : ℕ := (13 * 2 - 4 + 6)

theorem mia_has_largest_final_value : mia_final > daniel_final ∧ mia_final > carlos_final := by
  sorry

end mia_has_largest_final_value_l1727_172791


namespace simplify_expression_l1727_172797
theorem simplify_expression (c : ℝ) : 
    (3 * c + 6 - 6 * c) / 3 = -c + 2 := 
by 
    sorry

end simplify_expression_l1727_172797


namespace total_marks_is_275_l1727_172751

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l1727_172751


namespace orange_bin_count_l1727_172729

theorem orange_bin_count (initial_count throw_away add_new : ℕ) 
  (h1 : initial_count = 40) 
  (h2 : throw_away = 37) 
  (h3 : add_new = 7) : 
  initial_count - throw_away + add_new = 10 := 
by 
  sorry

end orange_bin_count_l1727_172729


namespace area_of_parallelogram_l1727_172794

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 :=
by
  rw [h_base, h_height]
  norm_num

end area_of_parallelogram_l1727_172794


namespace complex_ratio_real_l1727_172798

theorem complex_ratio_real (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ z : ℂ, z = a + b * Complex.I ∧ (z * (1 - 2 * Complex.I)).im = 0) :
  a / b = 1 / 2 :=
sorry

end complex_ratio_real_l1727_172798


namespace additional_boys_went_down_slide_l1727_172780

theorem additional_boys_went_down_slide (initial_boys total_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) : additional_boys = 13 :=
by {
    -- Proof body will be here
    sorry
}

end additional_boys_went_down_slide_l1727_172780


namespace domain_of_ratio_function_l1727_172707

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 ^ x)

theorem domain_of_ratio_function (D : Set ℝ) (hD : D = Set.Icc 1 2):
  ∀ f : ℝ → ℝ, (∀ x, g x = f (2 ^ x)) →
  ∃ D' : Set ℝ, D' = {x | 2 ≤ x ∧ x ≤ 4} →
  ∀ y : ℝ, (2 ≤ y ∧ y ≤ 4) → ∃ x : ℝ, y = x + 1 ∧ x ≠ 1 → (1 < x ∧ x ≤ 3) :=
sorry

end domain_of_ratio_function_l1727_172707


namespace range_of_m_l1727_172727

def p (m : ℝ) : Prop := m^2 - 4 > 0 ∧ m > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

theorem range_of_m (m : ℝ) : condition1 m ∧ condition2 m → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end range_of_m_l1727_172727


namespace parabola_vertex_l1727_172784

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end parabola_vertex_l1727_172784


namespace arrangement_count_l1727_172790

-- Definitions from the conditions
def people : Nat := 5
def valid_positions_for_A : Finset Nat := Finset.range 5 \ {0, 4}

-- The theorem that states the question equals the correct answer given the conditions
theorem arrangement_count (A_positions : Finset Nat := valid_positions_for_A) : 
  ∃ (total_arrangements : Nat), total_arrangements = 72 :=
by
  -- Placeholder for the proof
  sorry

end arrangement_count_l1727_172790


namespace inequality_solution_l1727_172762

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l1727_172762


namespace xyz_expr_min_max_l1727_172795

open Real

theorem xyz_expr_min_max (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 1) :
  ∃ m M : ℝ, m = 0 ∧ M = 1/4 ∧
    (∀ x y z : ℝ, x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
      xy + yz + zx - 3 * xyz ≥ m ∧ xy + yz + zx - 3 * xyz ≤ M) :=
sorry

end xyz_expr_min_max_l1727_172795


namespace arithmetic_sequence_general_term_l1727_172773

theorem arithmetic_sequence_general_term
  (d : ℕ) (a : ℕ → ℕ)
  (ha4 : a 4 = 14)
  (hd : d = 3) :
  ∃ a₁, ∀ n, a n = a₁ + (n - 1) * d := by
  sorry

end arithmetic_sequence_general_term_l1727_172773


namespace calc_expression_result_l1727_172731

theorem calc_expression_result :
  (16^12 * 8^8 / 2^60 = 4096) :=
by
  sorry

end calc_expression_result_l1727_172731


namespace problem_sol_max_distance_from_circle_to_line_l1727_172703

noncomputable def max_distance_circle_line : ℝ :=
  let ρ (θ : ℝ) : ℝ := 8 * Real.sin θ
  let line (θ : ℝ) : Prop := θ = Real.pi / 3
  let circle_center := (0, 4)
  let line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x
  let shortest_distance := 2  -- Already calculated in solution
  let radius := 4
  shortest_distance + radius

theorem problem_sol_max_distance_from_circle_to_line :
  max_distance_circle_line = 6 :=
by
  unfold max_distance_circle_line
  sorry

end problem_sol_max_distance_from_circle_to_line_l1727_172703


namespace tan_frac_a_pi_six_eq_sqrt_three_l1727_172759

theorem tan_frac_a_pi_six_eq_sqrt_three (a : ℝ) (h : (a, 9) ∈ { p : ℝ × ℝ | p.2 = 3 ^ p.1 }) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := 
by
  sorry

end tan_frac_a_pi_six_eq_sqrt_three_l1727_172759


namespace proof_problem_l1727_172748

def star (a b : ℕ) : ℕ := a - a / b

theorem proof_problem : star 18 6 + 2 * 6 = 27 := 
by
  admit  -- proof goes here

end proof_problem_l1727_172748


namespace consecutive_negatives_product_sum_l1727_172712

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l1727_172712


namespace smallest_positive_perfect_cube_l1727_172766

theorem smallest_positive_perfect_cube (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ m : ℕ, m = (a * b * c^2)^3 ∧ (a^2 * b^3 * c^5 ∣ m)
:=
sorry

end smallest_positive_perfect_cube_l1727_172766


namespace complex_magnitude_add_reciprocals_l1727_172799

open Complex

theorem complex_magnitude_add_reciprocals
  (z w : ℂ)
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hz_plus_w : Complex.abs (z + w) = 6) :
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end complex_magnitude_add_reciprocals_l1727_172799


namespace find_n_from_digits_sum_l1727_172756

theorem find_n_from_digits_sum (n : ℕ) (h1 : 777 = (9 * 1) + ((99 - 10 + 1) * 2) + (n - 99) * 3) : n = 295 :=
sorry

end find_n_from_digits_sum_l1727_172756


namespace five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l1727_172745

-- Prove that the number of five-digit numbers is 27216
theorem five_digit_numbers_count : ∃ n, n = 9 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the number of five-digit numbers greater than or equal to 30000 is 21168
theorem five_digit_numbers_ge_30000 : 
  ∃ n, n = 7 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the rank of 50124 among five-digit numbers with distinct digits in descending order is 15119
theorem rank_of_50124 : 
  ∃ n, n = (Nat.factorial 5) - 1 := by
  sorry

end five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l1727_172745


namespace snake_price_correct_l1727_172730

-- Define the conditions
def num_snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def total_eggs : ℕ := num_snakes * eggs_per_snake
def super_rare_multiple : ℕ := 4
def total_revenue : ℕ := 2250

-- The question: How much does each regular baby snake sell for?
def price_of_regular_baby_snake := 250

-- The proof statement
theorem snake_price_correct
  (x : ℕ)
  (h1 : total_eggs = 6)
  (h2 : 5 * x + super_rare_multiple * x = total_revenue)
  :
  x = price_of_regular_baby_snake := 
sorry

end snake_price_correct_l1727_172730


namespace symmetric_point_origin_l1727_172772

theorem symmetric_point_origin (m : ℤ) : 
  (symmetry_condition : (3, m - 2) = (-(-3), -5)) → m = -3 :=
by
  sorry

end symmetric_point_origin_l1727_172772


namespace point_in_third_quadrant_l1727_172702

theorem point_in_third_quadrant (x y : ℝ) (h1 : x = -3) (h2 : y = -2) : 
  x < 0 ∧ y < 0 :=
by
  sorry

end point_in_third_quadrant_l1727_172702


namespace find_integer_solutions_l1727_172740

theorem find_integer_solutions :
  (a b : ℤ) →
  3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 →
  (a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7) :=
sorry

end find_integer_solutions_l1727_172740


namespace sequence_formula_l1727_172747

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^n - 2 :=
by 
sorry

end sequence_formula_l1727_172747


namespace candy_division_l1727_172721

theorem candy_division 
  (total_candy : ℕ)
  (total_bags : ℕ)
  (candies_per_bag : ℕ)
  (chocolate_heart_bags : ℕ)
  (fruit_jelly_bags : ℕ)
  (caramel_chew_bags : ℕ) 
  (H1 : total_candy = 260)
  (H2 : total_bags = 13)
  (H3 : candies_per_bag = total_candy / total_bags)
  (H4 : chocolate_heart_bags = 4)
  (H5 : fruit_jelly_bags = 3)
  (H6 : caramel_chew_bags = total_bags - chocolate_heart_bags - fruit_jelly_bags)
  (H7 : candies_per_bag = 20) :
  (chocolate_heart_bags * candies_per_bag) + 
  (fruit_jelly_bags * candies_per_bag) + 
  (caramel_chew_bags * candies_per_bag) = 260 :=
sorry

end candy_division_l1727_172721


namespace unique_solution_for_digits_l1727_172796

theorem unique_solution_for_digits :
  ∃ (A B C D E : ℕ),
  (A < B ∧ B < C ∧ C < D ∧ D < E) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
   C ≠ D ∧ C ≠ E ∧
   D ≠ E) ∧
  (10 * A + B) * C = 10 * D + E ∧
  (A = 1 ∧ B = 3 ∧ C = 6 ∧ D = 7 ∧ E = 8) :=
sorry

end unique_solution_for_digits_l1727_172796


namespace matrix_cubic_l1727_172782

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l1727_172782


namespace pet_store_cats_left_l1727_172700

theorem pet_store_cats_left (siamese house sold : ℕ) (h_siamese : siamese = 38) (h_house : house = 25) (h_sold : sold = 45) :
  siamese + house - sold = 18 :=
by
  sorry

end pet_store_cats_left_l1727_172700


namespace common_ratio_geometric_sequence_l1727_172728

theorem common_ratio_geometric_sequence (q : ℝ) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = q)
  (h₂ : a 3 = q^2)
  (h₃ : (4 * a 1 + a 3 = 2 * 2 * a 2)) :
  q = 2 :=
by sorry

end common_ratio_geometric_sequence_l1727_172728


namespace car_time_passed_l1727_172744

variable (speed : ℝ) (distance : ℝ) (time_passed : ℝ)

theorem car_time_passed (h_speed : speed = 2) (h_distance : distance = 2) :
  time_passed = distance / speed := by
  rw [h_speed, h_distance]
  norm_num
  sorry

end car_time_passed_l1727_172744


namespace residential_ratio_l1727_172742

theorem residential_ratio (B R O E : ℕ) (h1 : B = 300) (h2 : E = 75) (h3 : E = O ∧ R + 2 * E = B) : R / B = 1 / 2 :=
by
  sorry

end residential_ratio_l1727_172742


namespace minimal_coach_handshakes_l1727_172763

theorem minimal_coach_handshakes (n k1 k2 : ℕ) (h1 : k1 < n) (h2 : k2 < n)
  (hn : (n * (n - 1)) / 2 + k1 + k2 = 300) : k1 + k2 = 0 := by
  sorry

end minimal_coach_handshakes_l1727_172763


namespace median_of_roller_coaster_times_l1727_172775

theorem median_of_roller_coaster_times:
  let data := [80, 85, 90, 125, 130, 135, 140, 145, 195, 195, 210, 215, 240, 245, 300, 305, 315, 320, 325, 330, 300]
  ∃ median_time, median_time = 210 ∧
    (∀ t ∈ data, t ≤ median_time ↔ index_of_median = 11) :=
by
  sorry

end median_of_roller_coaster_times_l1727_172775


namespace obtuse_triangle_has_exactly_one_obtuse_angle_l1727_172771

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- Definition of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

-- The theorem statement
theorem obtuse_triangle_has_exactly_one_obtuse_angle {A B C : ℝ} 
  (h1 : is_obtuse_triangle A B C) : 
  (is_obtuse_angle A ∨ is_obtuse_angle B ∨ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle B) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle B ∧ is_obtuse_angle C) :=
sorry

end obtuse_triangle_has_exactly_one_obtuse_angle_l1727_172771


namespace isosceles_triangle_k_l1727_172710

theorem isosceles_triangle_k (m n k : ℝ) (h_iso : (m = 4 ∨ n = 4 ∨ m = n) ∧ (m ≠ n ∨ (m = n ∧ m + m > 4))) 
  (h_roots : ∀ x, x^2 - 6*x + (k + 2) = 0 → (x = m ∨ x = n)) : k = 6 ∨ k = 7 :=
sorry

end isosceles_triangle_k_l1727_172710


namespace maximum_sum_of_composites_l1727_172776

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def pairwise_coprime (A B C : ℕ) : Prop :=
  Nat.gcd A B = 1 ∧ Nat.gcd A C = 1 ∧ Nat.gcd B C = 1

theorem maximum_sum_of_composites (A B C : ℕ)
  (hA : is_composite A) (hB : is_composite B) (hC : is_composite C)
  (h_pairwise : pairwise_coprime A B C)
  (h_prod_eq : A * B * C = 11011 * 28) :
  A + B + C = 1626 := 
sorry

end maximum_sum_of_composites_l1727_172776


namespace runs_by_running_percentage_l1727_172770

def total_runs := 125
def boundaries := 5
def boundary_runs := boundaries * 4
def sixes := 5
def sixes_runs := sixes * 6
def runs_by_running := total_runs - (boundary_runs + sixes_runs)
def percentage_runs_by_running := (runs_by_running : ℚ) / total_runs * 100

theorem runs_by_running_percentage :
  percentage_runs_by_running = 60 := by sorry

end runs_by_running_percentage_l1727_172770


namespace divisibility_of_expression_l1727_172714

open Int

theorem divisibility_of_expression (a b : ℤ) (ha : Prime a) (hb : Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) :=
sorry

end divisibility_of_expression_l1727_172714


namespace students_on_right_side_l1727_172757

-- Define the total number of students and the number of students on the left side
def total_students : ℕ := 63
def left_students : ℕ := 36

-- Define the number of students on the right side using subtraction
def right_students (total_students left_students : ℕ) : ℕ := total_students - left_students

-- Theorem: Prove that the number of students on the right side is 27
theorem students_on_right_side : right_students total_students left_students = 27 := by
  sorry

end students_on_right_side_l1727_172757


namespace julia_birth_year_is_1979_l1727_172705

-- Definitions based on conditions
def wayne_age_in_2021 : ℕ := 37
def wayne_birth_year : ℕ := 2021 - wayne_age_in_2021
def peter_birth_year : ℕ := wayne_birth_year - 3
def julia_birth_year : ℕ := peter_birth_year - 2

-- Theorem to prove
theorem julia_birth_year_is_1979 : julia_birth_year = 1979 := by
  sorry

end julia_birth_year_is_1979_l1727_172705


namespace find_line_equation_l1727_172722

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end find_line_equation_l1727_172722


namespace solve_tangent_problem_l1727_172701

noncomputable def problem_statement : Prop :=
  ∃ (n : ℤ), (-90 < n ∧ n < 90) ∧ (Real.tan (n * Real.pi / 180) = Real.tan (255 * Real.pi / 180)) ∧ (n = 75)

-- This is the statement of the problem we are proving.
theorem solve_tangent_problem : problem_statement :=
by
  sorry

end solve_tangent_problem_l1727_172701


namespace ratio_identity_l1727_172734

-- Given system of equations
def system_of_equations (k : ℚ) (x y z : ℚ) :=
  x + k * y + 2 * z = 0 ∧
  2 * x + k * y + 3 * z = 0 ∧
  3 * x + 5 * y + 4 * z = 0

-- Prove that for k = -7/5, the system has a nontrivial solution and 
-- that the ratio xz / y^2 equals -25
theorem ratio_identity (x y z : ℚ) (k : ℚ) (h : system_of_equations k x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  k = -7 / 5 → x * z / y^2 = -25 :=
by
  sorry

end ratio_identity_l1727_172734


namespace min_value_eval_l1727_172765

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end min_value_eval_l1727_172765


namespace arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l1727_172793

open Nat

variable (a : ℕ → ℝ)
variable (c : ℕ → ℝ)
variable (k b : ℝ)

-- Condition 1: sequence definition
def sequence_condition := ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + n + 1

-- Condition 2: initial value
def initial_value := a 1 = -1

-- Condition 3: c_n definition
def geometric_sequence_condition := ∀ n : ℕ, 0 < n → c (n + 1) / c n = 2

-- Problem 1: Arithmetic sequence parameters
theorem arith_sequence_parameters (h1 : sequence_condition a) (h2 : initial_value a) : a 1 = -3 ∧ 2 * (a 1 + 2) - a 1 - 7 = -1 :=
by sorry

-- Problem 2: Cannot be a geometric sequence
theorem not_geo_sequence (h1 : sequence_condition a) (h2 : initial_value a) : ¬ (∃ q, ∀ n : ℕ, 0 < n → a n * q = a (n + 1)) :=
by sorry

-- Problem 3: c_n is a geometric sequence and general term for a_n
theorem geo_sequence_and_gen_term (h1 : sequence_condition a) (h2 : initial_value a) 
    (h3 : ∀ n : ℕ, 0 < n → c n = a n + k * n + b)
    (hk : k = 1) (hb : b = 2) : sequence_condition a ∧ initial_value a :=
by sorry

end arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l1727_172793


namespace female_with_advanced_degrees_l1727_172767

theorem female_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_employees_with_advanced_degrees : ℕ)
  (total_employees_with_college_degree_only : ℕ)
  (total_males_with_college_degree_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : total_employees_with_advanced_degrees = 90)
  (h4 : total_employees_with_college_degree_only = 90)
  (h5 : total_males_with_college_degree_only = 35) :
  ∃ (female_with_advanced_degrees : ℕ), female_with_advanced_degrees = 55 :=
by
  -- the proof goes here
  sorry

end female_with_advanced_degrees_l1727_172767


namespace total_screens_sold_is_45000_l1727_172746

-- Define the number of screens sold in each month based on X
variables (X : ℕ)

-- Conditions given in the problem
def screens_in_January := X
def screens_in_February := 2 * X
def screens_in_March := (screens_in_January X + screens_in_February X) / 2
def screens_in_April := min (2 * screens_in_March X) 20000

-- Given that April sales were 18000
axiom apr_sales_18000 : screens_in_April X = 18000

-- Total sales is the sum of sales from January to April
def total_sales := screens_in_January X + screens_in_February X + screens_in_March X + 18000

-- Prove that total sales is 45000
theorem total_screens_sold_is_45000 : total_sales X = 45000 :=
by sorry

end total_screens_sold_is_45000_l1727_172746


namespace tank_insulation_cost_l1727_172743

theorem tank_insulation_cost (l w h : ℝ) (cost_per_sqft : ℝ) (SA : ℝ) (C : ℝ) 
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost_per_sqft : cost_per_sqft = 20) 
  (h_SA : SA = 2 * l * w + 2 * l * h + 2 * w * h)
  (h_C : C = SA * cost_per_sqft) :
  C = 1440 := 
by
  -- proof will be filled in here
  sorry

end tank_insulation_cost_l1727_172743


namespace find_sum_of_bounds_l1727_172738

variable (x y z : ℝ)

theorem find_sum_of_bounds (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) : 
  let m := min x (min y z)
  let M := max x (max y z)
  m + M = 8 / 3 :=
sorry

end find_sum_of_bounds_l1727_172738


namespace rational_roots_of_polynomial_l1727_172758

theorem rational_roots_of_polynomial :
  { x : ℚ | (x + 1) * (x - (2 / 3)) * (x^2 - 2) = 0 } = {-1, 2 / 3} :=
by
  sorry

end rational_roots_of_polynomial_l1727_172758


namespace whale_crossing_time_l1727_172774

theorem whale_crossing_time
  (speed_fast : ℝ)
  (speed_slow : ℝ)
  (length_slow : ℝ)
  (h_fast : speed_fast = 18)
  (h_slow : speed_slow = 15)
  (h_length : length_slow = 45) :
  (length_slow / (speed_fast - speed_slow) = 15) :=
by
  sorry

end whale_crossing_time_l1727_172774


namespace tom_buys_papayas_l1727_172709

-- Defining constants for the costs of each fruit
def lemon_cost : ℕ := 2
def papaya_cost : ℕ := 1
def mango_cost : ℕ := 4

-- Defining the number of each fruit Tom buys
def lemons_bought : ℕ := 6
def mangos_bought : ℕ := 2
def total_paid : ℕ := 21

-- Defining the function to calculate the total cost 
def total_cost (P : ℕ) : ℕ := (lemons_bought * lemon_cost) + (mangos_bought * mango_cost) + (P * papaya_cost)

-- Defining the function to calculate the discount based on the total number of fruits
def discount (P : ℕ) : ℕ := (lemons_bought + mangos_bought + P) / 4

-- Main theorem to prove
theorem tom_buys_papayas (P : ℕ) : total_cost P - discount P = total_paid → P = 4 := 
by
  intro h
  sorry

end tom_buys_papayas_l1727_172709


namespace length_of_second_train_is_correct_l1727_172708

-- Define the known values and conditions
def speed_train1_kmph := 120
def speed_train2_kmph := 80
def length_train1_m := 280
def crossing_time_s := 9

-- Convert speeds from km/h to m/s
def kmph_to_mps (kmph : ℕ) : ℚ := kmph * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

-- Calculate relative speed
def relative_speed_mps := speed_train1_mps + speed_train2_mps

-- Calculate total distance covered when crossing
def total_distance_m := relative_speed_mps * crossing_time_s

-- The length of the second train
def length_train2_m := total_distance_m - length_train1_m

-- Prove the length of the second train
theorem length_of_second_train_is_correct : length_train2_m = 219.95 := by {
  sorry
}

end length_of_second_train_is_correct_l1727_172708


namespace shopkeeper_profit_percent_l1727_172733

theorem shopkeeper_profit_percent
  (initial_value : ℝ)
  (percent_lost_theft : ℝ)
  (percent_total_loss : ℝ)
  (remaining_value : ℝ)
  (total_loss_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (profit_percent : ℝ)
  (h_initial_value : initial_value = 100)
  (h_percent_lost_theft : percent_lost_theft = 20)
  (h_percent_total_loss : percent_total_loss = 12)
  (h_remaining_value : remaining_value = initial_value - (percent_lost_theft / 100) * initial_value)
  (h_total_loss_value : total_loss_value = (percent_total_loss / 100) * initial_value)
  (h_selling_price : selling_price = initial_value - total_loss_value)
  (h_profit : profit = selling_price - remaining_value)
  (h_profit_percent : profit_percent = (profit / remaining_value) * 100) :
  profit_percent = 10 := by
  sorry

end shopkeeper_profit_percent_l1727_172733


namespace tangent_line_of_ellipse_l1727_172760

noncomputable def ellipse_tangent_line (a b x0 y0 x y : ℝ) : Prop :=
  x0 * x / a^2 + y0 * y / b^2 = 1

theorem tangent_line_of_ellipse
  (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_b : a > b) :
  ellipse_tangent_line a b x0 y0 x y :=
sorry

end tangent_line_of_ellipse_l1727_172760


namespace no_obtuse_equilateral_triangle_exists_l1727_172755

theorem no_obtuse_equilateral_triangle_exists :
  ¬(∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = π ∧ a > π/2 ∧ b > π/2 ∧ c > π/2) :=
sorry

end no_obtuse_equilateral_triangle_exists_l1727_172755


namespace triangle_is_isosceles_l1727_172732

theorem triangle_is_isosceles
    (A B C : ℝ)
    (h_angle_sum : A + B + C = 180)
    (h_sinB : Real.sin B = 2 * Real.cos C * Real.sin A)
    : (A = C) := 
by
    sorry

end triangle_is_isosceles_l1727_172732


namespace solutionToEquations_solutionToInequalities_l1727_172778

-- Part 1: Solve the system of equations
def solveEquations (x y : ℝ) : Prop :=
2 * x - y = 3 ∧ x + y = 6

theorem solutionToEquations (x y : ℝ) (h : solveEquations x y) : 
x = 3 ∧ y = 3 :=
sorry

-- Part 2: Solve the system of inequalities
def solveInequalities (x : ℝ) : Prop :=
3 * x > x - 4 ∧ (4 + x) / 3 > x + 2

theorem solutionToInequalities (x : ℝ) (h : solveInequalities x) : 
-2 < x ∧ x < -1 :=
sorry

end solutionToEquations_solutionToInequalities_l1727_172778


namespace polynomial_inequality_solution_l1727_172735

theorem polynomial_inequality_solution (x : ℝ) :
  (x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29) →
  x^3 - 12 * x^2 + 36 * x + 8 > 0 :=
by
  sorry

end polynomial_inequality_solution_l1727_172735


namespace equilateral_triangle_square_ratio_l1727_172718

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l1727_172718


namespace gcd_multiples_l1727_172789

theorem gcd_multiples (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : Nat.gcd p q = 15) : Nat.gcd (8 * p) (18 * q) = 30 :=
by sorry

end gcd_multiples_l1727_172789


namespace correct_assignment_statement_l1727_172713

theorem correct_assignment_statement (n m : ℕ) : 
  ¬ (4 = n) ∧ ¬ (n + 1 = m) ∧ ¬ (m + n = 0) :=
by
  sorry

end correct_assignment_statement_l1727_172713


namespace sum_of_integers_is_106_l1727_172761

theorem sum_of_integers_is_106 (n m : ℕ) 
  (h1: n * (n + 1) = 1320) 
  (h2: m * (m + 1) * (m + 2) = 1320) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 106 :=
  sorry

end sum_of_integers_is_106_l1727_172761


namespace gold_cube_profit_multiple_l1727_172715

theorem gold_cube_profit_multiple :
  let side_length : ℝ := 6
  let density : ℝ := 19
  let cost_per_gram : ℝ := 60
  let profit : ℝ := 123120
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * cost_per_gram
  let selling_price := cost + profit
  let multiple := selling_price / cost
  multiple = 1.5 := by
  sorry

end gold_cube_profit_multiple_l1727_172715


namespace ryan_learning_hours_l1727_172723

theorem ryan_learning_hours (total_hours : ℕ) (chinese_hours : ℕ) (english_hours : ℕ) 
  (h1 : total_hours = 3) (h2 : chinese_hours = 1) : 
  english_hours = 2 :=
by 
  sorry

end ryan_learning_hours_l1727_172723


namespace min_value_problem_l1727_172711

theorem min_value_problem 
  (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ (min_val : ℝ), min_val = 2 * x + 3 * y^2 ∧ min_val = 8 / 9 :=
by
  sorry

end min_value_problem_l1727_172711


namespace length_increase_percentage_l1727_172717

theorem length_increase_percentage (L B : ℝ) (x : ℝ) (h1 : (L + (x / 100) * L) * (B - (5 / 100) * B) = 1.14 * L * B) : x = 20 := by 
  sorry

end length_increase_percentage_l1727_172717


namespace negation_of_universal_l1727_172719

theorem negation_of_universal:
  ¬(∀ x : ℝ, (0 < x ∧ x < (π / 2)) → x > Real.sin x) ↔
  ∃ x : ℝ, (0 < x ∧ x < (π / 2)) ∧ x ≤ Real.sin x := by
  sorry

end negation_of_universal_l1727_172719


namespace draw_3_odd_balls_from_15_is_336_l1727_172786

-- Define the problem setting as given in the conditions
def odd_balls : Finset ℕ := {1, 3, 5, 7, 9, 11, 13, 15}

-- Define the function that calculates the number of ways to draw 3 balls
noncomputable def draw_3_odd_balls (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

-- Prove that the drawing of 3 balls results in 336 ways
theorem draw_3_odd_balls_from_15_is_336 : draw_3_odd_balls odd_balls = 336 := by
  sorry

end draw_3_odd_balls_from_15_is_336_l1727_172786


namespace percentage_of_games_not_won_is_40_l1727_172768

def ratio_games_won_to_lost (games_won games_lost : ℕ) : Prop := 
  games_won / gcd games_won games_lost = 3 ∧ games_lost / gcd games_won games_lost = 2

def total_games (games_won games_lost ties : ℕ) : ℕ :=
  games_won + games_lost + ties

def percentage_games_not_won (games_won games_lost ties : ℕ) : ℕ :=
  ((games_lost + ties) * 100) / (games_won + games_lost + ties)

theorem percentage_of_games_not_won_is_40
  (games_won games_lost ties : ℕ)
  (h_ratio : ratio_games_won_to_lost games_won games_lost)
  (h_ties : ties = 5)
  (h_no_other_games : games_won + games_lost + ties = total_games games_won games_lost ties) :
  percentage_games_not_won games_won games_lost ties = 40 := 
sorry

end percentage_of_games_not_won_is_40_l1727_172768


namespace pair_solution_l1727_172739

theorem pair_solution (a b : ℕ) (h_b_ne_1 : b ≠ 1) :
  (a + 1 ∣ a^3 * b - 1) → (b - 1 ∣ b^3 * a + 1) →
  (a, b) = (0, 0) ∨ (a, b) = (0, 2) ∨ (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3) :=
by
  sorry

end pair_solution_l1727_172739


namespace average_salary_difference_l1727_172749

theorem average_salary_difference :
  let total_payroll_factory := 30000
  let num_factory_workers := 15
  let total_payroll_office := 75000
  let num_office_workers := 30
  (total_payroll_office / num_office_workers) - (total_payroll_factory / num_factory_workers) = 500 :=
by
  sorry

end average_salary_difference_l1727_172749


namespace probability_of_rain_on_at_least_one_day_is_correct_l1727_172750

def rain_on_friday_probability : ℝ := 0.30
def rain_on_saturday_probability : ℝ := 0.45
def rain_on_sunday_probability : ℝ := 0.50

def rain_on_at_least_one_day_probability : ℝ := 1 - (1 - rain_on_friday_probability) * (1 - rain_on_saturday_probability) * (1 - rain_on_sunday_probability)

theorem probability_of_rain_on_at_least_one_day_is_correct :
  rain_on_at_least_one_day_probability = 0.8075 := by
sorry

end probability_of_rain_on_at_least_one_day_is_correct_l1727_172750


namespace smallest_y_l1727_172704

theorem smallest_y (y : ℤ) :
  (∃ k : ℤ, y^2 + 3*y + 7 = k*(y-2)) ↔ y = -15 :=
sorry

end smallest_y_l1727_172704


namespace tangent_line_through_origin_eq_ex_l1727_172781

theorem tangent_line_through_origin_eq_ex :
  ∃ (k : ℝ), (∀ x : ℝ, y = e^x) ∧ (∃ x₀ : ℝ, y - e^x₀ = e^x₀ * (x - x₀)) ∧ 
  (y = k * x) :=
sorry

end tangent_line_through_origin_eq_ex_l1727_172781


namespace circle_radius_and_circumference_l1727_172788

theorem circle_radius_and_circumference (A : ℝ) (hA : A = 64 * Real.pi) :
  ∃ r C : ℝ, r = 8 ∧ C = 2 * Real.pi * r :=
by
  -- statement ensures that with given area A, you can find r and C satisfying the conditions.
  sorry

end circle_radius_and_circumference_l1727_172788


namespace find_E_l1727_172752

variable (x E x1 x2 : ℝ)

/-- Given conditions as assumptions: -/
axiom h1 : (x + 3)^2 / E = 2
axiom h2 : x1 - x2 = 14

/-- Prove the required expression for E in terms of x: -/
theorem find_E : E = (x + 3)^2 / 2 := sorry

end find_E_l1727_172752


namespace number_of_outfits_l1727_172792

theorem number_of_outfits : (5 * 4 * 6 * 3) = 360 := by
  sorry

end number_of_outfits_l1727_172792


namespace f_is_n_l1727_172706

noncomputable def f : ℕ+ → ℤ :=
  sorry

def f_defined_for_all_positive_integers (n : ℕ+) : Prop :=
  ∃ k, f n = k

def f_is_integer (n : ℕ+) : Prop :=
  ∃ k : ℤ, f n = k

def f_two_is_two : Prop :=
  f 2 = 2

def f_multiply_rule (m n : ℕ+) : Prop :=
  f (m * n) = f m * f n

def f_ordered (m n : ℕ+) (h : m > n) : Prop :=
  f m > f n

theorem f_is_n (n : ℕ+) :
  (f_defined_for_all_positive_integers n) →
  (f_is_integer n) →
  (f_two_is_two) →
  (∀ m n, f_multiply_rule m n) →
  (∀ m n (h : m > n), f_ordered m n h) →
  f n = n :=
sorry

end f_is_n_l1727_172706


namespace sum_of_repeating_decimals_l1727_172783

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l1727_172783


namespace min_max_value_z_l1727_172754

theorem min_max_value_z (x y z : ℝ) (h1 : x^2 ≤ y + z) (h2 : y^2 ≤ z + x) (h3 : z^2 ≤ x + y) :
  -1/4 ≤ z ∧ z ≤ 2 :=
by {
  sorry
}

end min_max_value_z_l1727_172754


namespace amount_spent_on_tracksuit_l1727_172720

-- Definitions based on the conditions
def original_price (x : ℝ) := x
def discount_rate : ℝ := 0.20
def savings : ℝ := 30
def actual_spent (x : ℝ) := 0.8 * x

-- Theorem statement derived from the proof translation
theorem amount_spent_on_tracksuit (x : ℝ) (h : (original_price x) * discount_rate = savings) :
  actual_spent x = 120 :=
by
  sorry

end amount_spent_on_tracksuit_l1727_172720


namespace california_vs_texas_license_plates_l1727_172726

theorem california_vs_texas_license_plates :
  (26^4 * 10^4) - (26^3 * 10^3) = 4553200000 :=
by
  sorry

end california_vs_texas_license_plates_l1727_172726


namespace find_other_number_l1727_172764

theorem find_other_number (x : ℕ) (h : x + 42 = 96) : x = 54 :=
by {
  sorry
}

end find_other_number_l1727_172764


namespace book_arrangement_l1727_172787

theorem book_arrangement :
  let total_books := 6
  let identical_books := 3
  let unique_arrangements := Nat.factorial total_books / Nat.factorial identical_books
  unique_arrangements = 120 := by
  sorry

end book_arrangement_l1727_172787


namespace max_fraction_l1727_172777

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  1 + y / x ≤ -2 :=
sorry

end max_fraction_l1727_172777


namespace complex_solution_l1727_172785

open Complex

theorem complex_solution (z : ℂ) (h : z + Complex.abs z = 1 + Complex.I) : z = Complex.I := 
by
  sorry

end complex_solution_l1727_172785
