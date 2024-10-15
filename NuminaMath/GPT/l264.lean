import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_a11_l264_26419

theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h3 : a 3 = 4)
  (h7 : a 7 = 12) : 
  a 11 = 36 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a11_l264_26419


namespace NUMINAMATH_GPT_team_A_more_uniform_l264_26430

noncomputable def average_height : ℝ := 2.07

variables (S_A S_B : ℝ) (h_variance : S_A^2 < S_B^2)

theorem team_A_more_uniform : true ∧ false :=
by
  sorry

end NUMINAMATH_GPT_team_A_more_uniform_l264_26430


namespace NUMINAMATH_GPT_sum_of_decimals_l264_26409

theorem sum_of_decimals :
  let a := 0.3
  let b := 0.08
  let c := 0.007
  a + b + c = 0.387 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l264_26409


namespace NUMINAMATH_GPT_total_wire_length_l264_26488

theorem total_wire_length
  (A B C D E : ℕ)
  (hA : A = 16)
  (h_ratio : 4 * A = 5 * B ∧ 4 * A = 7 * C ∧ 4 * A = 3 * D ∧ 4 * A = 2 * E)
  (hC : C = B + 8) :
  (A + B + C + D + E) = 84 := 
sorry

end NUMINAMATH_GPT_total_wire_length_l264_26488


namespace NUMINAMATH_GPT_prime_quadruple_solution_l264_26429

-- Define the problem statement in Lean
theorem prime_quadruple_solution :
  ∀ (p q r : ℕ) (n : ℕ),
    Prime p → Prime q → Prime r → n > 0 →
    p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_prime_quadruple_solution_l264_26429


namespace NUMINAMATH_GPT_arctan_sum_l264_26485

theorem arctan_sum : 
  Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = Real.pi / 4 := 
by 
  sorry

end NUMINAMATH_GPT_arctan_sum_l264_26485


namespace NUMINAMATH_GPT_total_students_in_class_l264_26497

def current_students : ℕ := 6 * 3
def students_bathroom : ℕ := 5
def students_canteen : ℕ := 5 * 5
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def group4_students : ℕ := 3
def new_group_students : ℕ := group1_students + group2_students + group3_students + group4_students
def germany_students : ℕ := 3
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 2
def spain_students : ℕ := 2
def australia_students : ℕ := 1
def foreign_exchange_students : ℕ :=
  germany_students + france_students + norway_students + italy_students + spain_students + australia_students

def total_students : ℕ :=
  current_students + students_bathroom + students_canteen + new_group_students + foreign_exchange_students

theorem total_students_in_class : total_students = 81 := by
  rfl  -- Reflective equality since total_students already sums to 81 based on the definitions

end NUMINAMATH_GPT_total_students_in_class_l264_26497


namespace NUMINAMATH_GPT_sandy_saved_last_year_percentage_l264_26415

theorem sandy_saved_last_year_percentage (S : ℝ) (P : ℝ) :
  (this_year_salary: ℝ) → (this_year_savings: ℝ) → 
  (this_year_saved_percentage: ℝ) → (saved_last_year_percentage: ℝ) → 
  this_year_salary = 1.1 * S → 
  this_year_saved_percentage = 6 →
  this_year_savings = (this_year_saved_percentage / 100) * this_year_salary →
  (this_year_savings / ((P / 100) * S)) = 0.66 →
  P = 10 :=
by
  -- The proof is to be filled in here.
  sorry

end NUMINAMATH_GPT_sandy_saved_last_year_percentage_l264_26415


namespace NUMINAMATH_GPT_tea_leaves_costs_l264_26464

theorem tea_leaves_costs (a_1 b_1 a_2 b_2 : ℕ) (c_A c_B : ℝ) :
  a_1 * c_A = 4000 ∧ 
  b_1 * c_B = 8400 ∧ 
  b_1 = a_1 + 10 ∧ 
  c_B = 1.4 * c_A ∧ 
  a_2 + b_2 = 100 ∧ 
  (300 - c_A) * (a_2 / 2) + (300 * 0.7 - c_A) * (a_2 / 2) + 
  (400 - c_B) * (b_2 / 2) + (400 * 0.7 - c_B) * (b_2 / 2) = 5800 
  → c_A = 200 ∧ c_B = 280 ∧ a_2 = 40 ∧ b_2 = 60 := 
sorry

end NUMINAMATH_GPT_tea_leaves_costs_l264_26464


namespace NUMINAMATH_GPT_polynomial_diff_l264_26494

theorem polynomial_diff (m n : ℤ) (h1 : 2 * m + 2 = 0) (h2 : n - 4 = 0) :
  (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := 
by {
  -- This is where the proof would go, so we put sorry for now
  sorry
}

end NUMINAMATH_GPT_polynomial_diff_l264_26494


namespace NUMINAMATH_GPT_typing_and_editing_time_l264_26469

-- Definitions for typing and editing times for consultants together and for Mary and Jim individually
def combined_typing_time := 12.5
def combined_editing_time := 7.5
def mary_typing_time := 30.0
def jim_editing_time := 12.0

-- The total time when Jim types and Mary edits
def total_time := 42.0

-- Proof statement
theorem typing_and_editing_time :
  (combined_typing_time = 12.5) ∧ 
  (combined_editing_time = 7.5) ∧ 
  (mary_typing_time = 30.0) ∧ 
  (jim_editing_time = 12.0) →
  total_time = 42.0 := 
by
  intro h
  -- Proof to be filled later
  sorry

end NUMINAMATH_GPT_typing_and_editing_time_l264_26469


namespace NUMINAMATH_GPT_percentage_problem_l264_26487

-- Define the main proposition
theorem percentage_problem (n : ℕ) (a : ℕ) (b : ℕ) (P : ℕ) :
  n = 6000 →
  a = (50 * n) / 100 →
  b = (30 * a) / 100 →
  (P * b) / 100 = 90 →
  P = 10 :=
by
  intros h_n h_a h_b h_Pb
  sorry

end NUMINAMATH_GPT_percentage_problem_l264_26487


namespace NUMINAMATH_GPT_cars_produced_total_l264_26442

theorem cars_produced_total :
  3884 + 2871 = 6755 :=
by
  sorry

end NUMINAMATH_GPT_cars_produced_total_l264_26442


namespace NUMINAMATH_GPT_work_completion_time_l264_26427

theorem work_completion_time (A_works_in : ℕ) (A_works_days : ℕ) (B_works_remainder_in : ℕ) (total_days : ℕ) :
  (A_works_in = 60) → (A_works_days = 15) → (B_works_remainder_in = 30) → (total_days = 24) := 
by
  intros hA_work hA_days hB_work
  sorry

end NUMINAMATH_GPT_work_completion_time_l264_26427


namespace NUMINAMATH_GPT_surface_area_of_rectangular_prism_l264_26406

theorem surface_area_of_rectangular_prism :
  ∀ (length width height : ℝ), length = 8 → width = 4 → height = 2 → 
    2 * (length * width + length * height + width * height) = 112 :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  sorry

end NUMINAMATH_GPT_surface_area_of_rectangular_prism_l264_26406


namespace NUMINAMATH_GPT_math_problem_l264_26441

variable (x b : ℝ)
variable (h1 : x < b)
variable (h2 : b < 0)
variable (h3 : b = -2)

theorem math_problem : x^2 > b * x ∧ b * x > b^2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l264_26441


namespace NUMINAMATH_GPT_olivia_race_time_l264_26424

variable (O E : ℕ)

theorem olivia_race_time (h1 : O + E = 112) (h2 : E = O - 4) : O = 58 :=
sorry

end NUMINAMATH_GPT_olivia_race_time_l264_26424


namespace NUMINAMATH_GPT_gcd_impossible_l264_26400

-- Define the natural numbers a, b, and c
variable (a b c : ℕ)

-- Define the factorial values
def fact_30 := Nat.factorial 30
def fact_40 := Nat.factorial 40
def fact_50 := Nat.factorial 50

-- Define the gcd values to be checked
def gcd_ab := fact_30 + 111
def gcd_bc := fact_40 + 234
def gcd_ca := fact_50 + 666

-- The main theorem to prove the impossibility
theorem gcd_impossible (h1 : Nat.gcd a b = gcd_ab) (h2 : Nat.gcd b c = gcd_bc) (h3 : Nat.gcd c a = gcd_ca) : False :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_gcd_impossible_l264_26400


namespace NUMINAMATH_GPT_prob_both_students_female_l264_26455

-- Define the conditions
def total_students : ℕ := 5
def male_students : ℕ := 2
def female_students : ℕ := 3
def selected_students : ℕ := 2

-- Define the function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 2 female students
def probability_both_female : ℚ := 
  (binomial female_students selected_students : ℚ) / (binomial total_students selected_students : ℚ)

-- The actual theorem to be proved
theorem prob_both_students_female : probability_both_female = 0.3 := by
  sorry

end NUMINAMATH_GPT_prob_both_students_female_l264_26455


namespace NUMINAMATH_GPT_eliminate_alpha_l264_26486

theorem eliminate_alpha (α x y : ℝ) (h1 : x = Real.tan α ^ 2) (h2 : y = Real.sin α ^ 2) : 
  x - y = x * y := 
by
  sorry

end NUMINAMATH_GPT_eliminate_alpha_l264_26486


namespace NUMINAMATH_GPT_point_P_coordinates_l264_26445

-- Definitions based on conditions
def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.2 = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.1 = d

-- The theorem statement based on the proof problem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    in_fourth_quadrant P ∧ 
    distance_to_x_axis P 2 ∧ 
    distance_to_y_axis P 3 ∧ 
    P = (3, -2) :=
by
  sorry

end NUMINAMATH_GPT_point_P_coordinates_l264_26445


namespace NUMINAMATH_GPT_inequality_proof_l264_26473

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ((b + c - a)^2) / (a^2 + (b + c)^2) + ((c + a - b)^2) / (b^2 + (c + a)^2) + ((a + b - c)^2) / (c^2 + (a + b)^2) ≥ 3 / 5 :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l264_26473


namespace NUMINAMATH_GPT_Queen_High_School_teachers_needed_l264_26484

def students : ℕ := 1500
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

theorem Queen_High_School_teachers_needed : 
  (students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by 
  sorry

end NUMINAMATH_GPT_Queen_High_School_teachers_needed_l264_26484


namespace NUMINAMATH_GPT_complex_fraction_identity_l264_26452

theorem complex_fraction_identity (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_complex_fraction_identity_l264_26452


namespace NUMINAMATH_GPT_peaches_total_l264_26408

def peaches_in_basket (a b : Nat) : Nat :=
  a + b 

theorem peaches_total (a b : Nat) (h1 : a = 20) (h2 : b = 25) : peaches_in_basket a b = 45 := 
by
  sorry

end NUMINAMATH_GPT_peaches_total_l264_26408


namespace NUMINAMATH_GPT_unique_solution_to_exponential_poly_equation_l264_26466

noncomputable def polynomial_has_unique_real_solution : Prop :=
  ∃! x : ℝ, (2 : ℝ)^(3 * x + 3) - 3 * (2 : ℝ)^(2 * x + 1) - (2 : ℝ)^x + 1 = 0

theorem unique_solution_to_exponential_poly_equation :
  polynomial_has_unique_real_solution :=
sorry

end NUMINAMATH_GPT_unique_solution_to_exponential_poly_equation_l264_26466


namespace NUMINAMATH_GPT_weight_of_each_bag_l264_26420

theorem weight_of_each_bag (empty_weight loaded_weight : ℕ) (number_of_bags : ℕ) (weight_per_bag : ℕ)
    (h1 : empty_weight = 500)
    (h2 : loaded_weight = 1700)
    (h3 : number_of_bags = 20)
    (h4 : loaded_weight - empty_weight = number_of_bags * weight_per_bag) :
    weight_per_bag = 60 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_bag_l264_26420


namespace NUMINAMATH_GPT_lcm_pair_eq_sum_l264_26463

theorem lcm_pair_eq_sum (x y : ℕ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : Nat.lcm x y = 1 + 2 * x + 3 * y) :
  (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_lcm_pair_eq_sum_l264_26463


namespace NUMINAMATH_GPT_original_number_unique_l264_26444

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_original_number_unique_l264_26444


namespace NUMINAMATH_GPT_cube_volume_l264_26489

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 294) : s^3 = 343 := 
by 
  sorry

end NUMINAMATH_GPT_cube_volume_l264_26489


namespace NUMINAMATH_GPT_B1F_base16_to_base10_is_2847_l264_26418

theorem B1F_base16_to_base10_is_2847 : 
  let B := 11
  let one := 1
  let F := 15
  let base := 16
  B * base^2 + one * base^1 + F * base^0 = 2847 := 
by
  sorry

end NUMINAMATH_GPT_B1F_base16_to_base10_is_2847_l264_26418


namespace NUMINAMATH_GPT_tutors_meet_after_84_days_l264_26492

theorem tutors_meet_after_84_days :
  let jaclyn := 3
  let marcelle := 4
  let susanna := 6
  let wanda := 7
  Nat.lcm (Nat.lcm (Nat.lcm jaclyn marcelle) susanna) wanda = 84 := by
  sorry

end NUMINAMATH_GPT_tutors_meet_after_84_days_l264_26492


namespace NUMINAMATH_GPT_cubic_binomial_expansion_l264_26495

theorem cubic_binomial_expansion :
  49^3 + 3 * 49^2 + 3 * 49 + 1 = 125000 :=
by
  sorry

end NUMINAMATH_GPT_cubic_binomial_expansion_l264_26495


namespace NUMINAMATH_GPT_max_value_of_t_l264_26433

variable (n r t : ℕ)
variable (A : Finset (Finset (Fin n)))
variable (h₁ : n ≤ 2 * r)
variable (h₂ : ∀ s ∈ A, Finset.card s = r)
variable (h₃ : Finset.card A = t)

theorem max_value_of_t : 
  (n < 2 * r → t ≤ Nat.choose n r) ∧ 
  (n = 2 * r → t ≤ Nat.choose n r / 2) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_t_l264_26433


namespace NUMINAMATH_GPT_range_of_a_l264_26482

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x - a ≤ -3) → a ∈ Set.Iic (-6) ∪ Set.Ici 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l264_26482


namespace NUMINAMATH_GPT_leopards_count_l264_26471

theorem leopards_count (L : ℕ) (h1 : 100 + 80 + L + 10 * L + 50 + 2 * (80 + L) = 670) : L = 20 :=
by
  sorry

end NUMINAMATH_GPT_leopards_count_l264_26471


namespace NUMINAMATH_GPT_quadratic_eq_zero_l264_26413

theorem quadratic_eq_zero (x a b : ℝ) (h : x = a ∨ x = b) : x^2 - (a + b) * x + a * b = 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_eq_zero_l264_26413


namespace NUMINAMATH_GPT_cost_function_segments_l264_26446

def C (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 10 then 10 * n
  else if h : 10 < n then 8 * n - 40
  else 0

theorem cost_function_segments :
  (∀ n, 1 ≤ n ∧ n ≤ 10 → C n = 10 * n) ∧
  (∀ n, 10 < n → C n = 8 * n - 40) ∧
  (∀ n, C n = if (1 ≤ n ∧ n ≤ 10) then 10 * n else if (10 < n) then 8 * n - 40 else 0) ∧
  ∃ n₁ n₂, (1 ≤ n₁ ∧ n₁ ≤ 10) ∧ (10 < n₂ ∧ n₂ ≤ 20) ∧ C n₁ = 10 * n₁ ∧ C n₂ = 8 * n₂ - 40 :=
by
  sorry

end NUMINAMATH_GPT_cost_function_segments_l264_26446


namespace NUMINAMATH_GPT_surface_area_ratio_l264_26496

-- Definitions for side lengths in terms of common multiplier x
def side_length_a (x : ℝ) := 2 * x
def side_length_b (x : ℝ) := 1 * x
def side_length_c (x : ℝ) := 3 * x
def side_length_d (x : ℝ) := 4 * x
def side_length_e (x : ℝ) := 6 * x

-- Definitions for surface areas using the given formula
def surface_area (side_length : ℝ) := 6 * side_length^2

def surface_area_a (x : ℝ) := surface_area (side_length_a x)
def surface_area_b (x : ℝ) := surface_area (side_length_b x)
def surface_area_c (x : ℝ) := surface_area (side_length_c x)
def surface_area_d (x : ℝ) := surface_area (side_length_d x)
def surface_area_e (x : ℝ) := surface_area (side_length_e x)

-- Proof statement for the ratio of total surface areas
theorem surface_area_ratio (x : ℝ) (hx : x ≠ 0) :
  (surface_area_a x) / (surface_area_b x) = 4 ∧
  (surface_area_c x) / (surface_area_b x) = 9 ∧
  (surface_area_d x) / (surface_area_b x) = 16 ∧
  (surface_area_e x) / (surface_area_b x) = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_surface_area_ratio_l264_26496


namespace NUMINAMATH_GPT_rectangle_square_overlap_l264_26470

theorem rectangle_square_overlap (ABCD EFGH : Type) (s x y : ℝ)
  (h1 : 0.3 * s^2 = 0.6 * x * y)
  (h2 : AB = 2 * s)
  (h3 : AD = y)
  (h4 : x * y = 0.5 * s^2) :
  x / y = 8 :=
sorry

end NUMINAMATH_GPT_rectangle_square_overlap_l264_26470


namespace NUMINAMATH_GPT_coin_difference_l264_26483

variables (x y : ℕ)

theorem coin_difference (h1 : x + y = 15) (h2 : 2 * x + 5 * y = 51) : x - y = 1 := by
  sorry

end NUMINAMATH_GPT_coin_difference_l264_26483


namespace NUMINAMATH_GPT_mario_time_on_moving_sidewalk_l264_26458

theorem mario_time_on_moving_sidewalk (d w v : ℝ) (h_walk : d = 90 * w) (h_sidewalk : d = 45 * v) : 
  d / (w + v) = 30 :=
by
  sorry

end NUMINAMATH_GPT_mario_time_on_moving_sidewalk_l264_26458


namespace NUMINAMATH_GPT_part1_part2_l264_26490

variable (a b : ℝ)
def A : ℝ := 2 * a * b - a
def B : ℝ := -a * b + 2 * a + b

theorem part1 : 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b := by
  sorry

theorem part2 : (∀ b : ℝ, 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b) -> a = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l264_26490


namespace NUMINAMATH_GPT_lucas_initial_money_l264_26425

theorem lucas_initial_money : (3 * 2 + 14 = 20) := by sorry

end NUMINAMATH_GPT_lucas_initial_money_l264_26425


namespace NUMINAMATH_GPT_find_speed_of_goods_train_l264_26460

noncomputable def speed_of_goods_train (v_man : ℝ) (t_pass : ℝ) (d_goods : ℝ) : ℝ := 
  let v_man_mps := v_man * (1000 / 3600)
  let v_relative := d_goods / t_pass
  let v_goods_mps := v_relative - v_man_mps
  v_goods_mps * (3600 / 1000)

theorem find_speed_of_goods_train :
  speed_of_goods_train 45 8 340 = 108 :=
by sorry

end NUMINAMATH_GPT_find_speed_of_goods_train_l264_26460


namespace NUMINAMATH_GPT_volume_of_figure_eq_half_l264_26491

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_volume_of_figure_eq_half_l264_26491


namespace NUMINAMATH_GPT_number_of_players_in_tournament_l264_26434

theorem number_of_players_in_tournament (n : ℕ) (h : 2 * 30 = n * (n - 1)) : n = 10 :=
sorry

end NUMINAMATH_GPT_number_of_players_in_tournament_l264_26434


namespace NUMINAMATH_GPT_total_area_rectangle_l264_26454

theorem total_area_rectangle (BF CF : ℕ) (A1 A2 x : ℕ) (h1 : BF = 3 * CF) (h2 : A1 = 3 * A2) (h3 : 2 * x = 96) (h4 : 48 = x) (h5 : A1 = 3 * 48) (h6 : A2 = 48) : A1 + A2 = 192 :=
  by sorry

end NUMINAMATH_GPT_total_area_rectangle_l264_26454


namespace NUMINAMATH_GPT_total_profit_is_64000_l264_26404

-- Definitions for investments and periods
variables (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ)

-- Conditions from the problem
def condition1 := IA = 5 * IB
def condition2 := TA = 3 * TB
def condition3 := Profit_B = 4000
def condition4 := Profit_A / Profit_B = (IA * TA) / (IB * TB)

-- Target statement to be proved
theorem total_profit_is_64000 (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ) :
  condition1 IA IB → condition2 TA TB → condition3 Profit_B → condition4 IA TA IB TB Profit_A Profit_B → 
  Total_Profit = Profit_A + Profit_B → Total_Profit = 64000 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_profit_is_64000_l264_26404


namespace NUMINAMATH_GPT_correct_operation_l264_26475

-- Define that m and n are elements of an arbitrary commutative ring
variables {R : Type*} [CommRing R] (m n : R)

theorem correct_operation : (m * n) ^ 2 = m ^ 2 * n ^ 2 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l264_26475


namespace NUMINAMATH_GPT_conversion_bah_rah_yah_l264_26474

theorem conversion_bah_rah_yah (bahs rahs yahs : ℝ) 
  (h1 : 10 * bahs = 16 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) :
  (10 / 16) * (6 / 10) * 500 * yahs = 187.5 * bahs :=
by sorry

end NUMINAMATH_GPT_conversion_bah_rah_yah_l264_26474


namespace NUMINAMATH_GPT_largest_common_divisor_of_product_l264_26423

theorem largest_common_divisor_of_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) :
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) → d ∣ k :=
by
  sorry

end NUMINAMATH_GPT_largest_common_divisor_of_product_l264_26423


namespace NUMINAMATH_GPT_total_items_8_l264_26437

def sandwiches_cost : ℝ := 5.0
def soft_drinks_cost : ℝ := 1.5
def total_money : ℝ := 40.0

noncomputable def total_items (s : ℕ) (d : ℕ) : ℕ := s + d

theorem total_items_8 :
  ∃ (s d : ℕ), 5 * (s : ℝ) + 1.5 * (d : ℝ) = 40 ∧ s + d = 8 := 
by
  sorry

end NUMINAMATH_GPT_total_items_8_l264_26437


namespace NUMINAMATH_GPT_smallest_possible_b_l264_26431

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end NUMINAMATH_GPT_smallest_possible_b_l264_26431


namespace NUMINAMATH_GPT_average_annual_cost_reduction_l264_26407

theorem average_annual_cost_reduction (x : ℝ) (h : (1 - x) ^ 2 = 0.64) : x = 0.2 :=
sorry

end NUMINAMATH_GPT_average_annual_cost_reduction_l264_26407


namespace NUMINAMATH_GPT_find_a_l264_26456

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = Real.log (-a * x)) (h2 : ∀ x : ℝ, f (-x) = -f x) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l264_26456


namespace NUMINAMATH_GPT_partition_pos_integers_100_subsets_l264_26422

theorem partition_pos_integers_100_subsets :
  ∃ (P : (ℕ+ → Fin 100)), ∀ a b c : ℕ+, (a + 99 * b = c) → P a = P c ∨ P a = P b ∨ P b = P c :=
sorry

end NUMINAMATH_GPT_partition_pos_integers_100_subsets_l264_26422


namespace NUMINAMATH_GPT_find_m_l264_26457

theorem find_m (m : ℤ) :
  (2 * m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end NUMINAMATH_GPT_find_m_l264_26457


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_l264_26465

variables {p : ℝ} (hp : p > 0) (x0 y0 x1 y1 x2 y2 : ℝ)
variables (h_parabola : ∀ x y, y^2 = 2*p*x) 
variables (h_point_P : ∀ k m, y0 ≠ 0 ∧ x0 = k*y0 + m)

-- Statement A
theorem statement_A (hy0 : y0 = 0) : y1 * y2 = -2 * p * x0 :=
sorry

-- Statement B
theorem statement_B (hx0 : x0 = 0) : 1 / y1 + 1 / y2 = 1 / y0 :=
sorry

-- Statement C
theorem statement_C : (y0 - y1) * (y0 - y2) = y0^2 - 2 * p * x0 :=
sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_l264_26465


namespace NUMINAMATH_GPT_middle_rectangle_frequency_l264_26432

theorem middle_rectangle_frequency (S A : ℝ) (h1 : S + A = 100) (h2 : A = S / 3) : A = 25 :=
by
  sorry

end NUMINAMATH_GPT_middle_rectangle_frequency_l264_26432


namespace NUMINAMATH_GPT_fries_sold_l264_26447

theorem fries_sold (small_fries large_fries : ℕ) (h1 : small_fries = 4) (h2 : large_fries = 5 * small_fries) :
  small_fries + large_fries = 24 :=
  by
    sorry

end NUMINAMATH_GPT_fries_sold_l264_26447


namespace NUMINAMATH_GPT_quadratic_trinomial_positive_c_l264_26449

theorem quadratic_trinomial_positive_c
  (a b c : ℝ)
  (h1 : b^2 < 4 * a * c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_trinomial_positive_c_l264_26449


namespace NUMINAMATH_GPT_vertex_of_quadratic_l264_26478

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ (h = 1) ∧ (k = -2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_quadratic_l264_26478


namespace NUMINAMATH_GPT_total_price_all_art_l264_26462

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_total_price_all_art_l264_26462


namespace NUMINAMATH_GPT_wire_length_l264_26468

variable (L M l a : ℝ) -- Assume these variables are real numbers.

theorem wire_length (h1 : a ≠ 0) : L = (M / a) * l :=
sorry

end NUMINAMATH_GPT_wire_length_l264_26468


namespace NUMINAMATH_GPT_simon_legos_l264_26411

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end NUMINAMATH_GPT_simon_legos_l264_26411


namespace NUMINAMATH_GPT_arithmetic_problem_l264_26435

theorem arithmetic_problem : 
  let x := 512.52 
  let y := 256.26 
  let diff := x - y 
  let result := diff * 3 
  result = 768.78 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l264_26435


namespace NUMINAMATH_GPT_Cheryl_more_eggs_than_others_l264_26481

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_Cheryl_more_eggs_than_others_l264_26481


namespace NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_inequality_l264_26499

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x : ℝ, f (x + 1) - f x = 2 * x) 
  (h₂ : f 0 = 1) : 
  (f x = x^2 - x + 1) := 
by {
  sorry
}

theorem quadratic_function_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m) ↔ m < -1 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_inequality_l264_26499


namespace NUMINAMATH_GPT_discount_percentage_l264_26450

/-
  A retailer buys 80 pens at the market price of 36 pens from a wholesaler.
  He sells these pens giving a certain discount and his profit is 120%.
  What is the discount percentage he gave on the pens?
-/
theorem discount_percentage
  (P : ℝ)
  (CP SP D DP : ℝ) 
  (h1 : CP = 36 * P)
  (h2 : SP = 2.2 * CP)
  (h3 : D = P - (SP / 80))
  (h4 : DP = (D / P) * 100) :
  DP = 1 := 
sorry

end NUMINAMATH_GPT_discount_percentage_l264_26450


namespace NUMINAMATH_GPT_Zhukov_birth_year_l264_26459

-- Define the conditions
def years_lived_total : ℕ := 78
def years_lived_20th_more_than_19th : ℕ := 70

-- Define the proof problem
theorem Zhukov_birth_year :
  ∃ y19 y20 : ℕ, y19 + y20 = years_lived_total ∧ y20 = y19 + years_lived_20th_more_than_19th ∧ (1900 - y19) = 1896 :=
by
  sorry

end NUMINAMATH_GPT_Zhukov_birth_year_l264_26459


namespace NUMINAMATH_GPT_suitable_communication_l264_26479

def is_suitable_to_communicate (beijing_time : Nat) (sydney_difference : Int) (los_angeles_difference : Int) : Bool :=
  let sydney_time := beijing_time + sydney_difference
  let los_angeles_time := beijing_time - los_angeles_difference
  sydney_time >= 8 ∧ sydney_time <= 22 -- let's assume suitable time is between 8:00 to 22:00

theorem suitable_communication:
  let beijing_time := 18
  let sydney_difference := 2
  let los_angeles_difference := 15
  is_suitable_to_communicate beijing_time sydney_difference los_angeles_difference = true :=
by
  sorry

end NUMINAMATH_GPT_suitable_communication_l264_26479


namespace NUMINAMATH_GPT_calculate_nested_expression_l264_26417

theorem calculate_nested_expression :
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2 = 1457 :=
by
  sorry

end NUMINAMATH_GPT_calculate_nested_expression_l264_26417


namespace NUMINAMATH_GPT_range_of_m_l264_26467

variable (m t : ℝ)

namespace proof_problem

def proposition_p : Prop :=
  ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1) → (t + 2) * (t - 10) < 0

def proposition_q (m : ℝ) : Prop :=
  -m < t ∧ t < m + 1 ∧ m > 0

theorem range_of_m :
  (∃ t, proposition_q m t) → proposition_p t → 0 < m ∧ m ≤ 2 := by
  sorry

end proof_problem

end NUMINAMATH_GPT_range_of_m_l264_26467


namespace NUMINAMATH_GPT_main_theorem_l264_26439

noncomputable def proof_problem (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2)

noncomputable def equality_case (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  α = π / 3 → 2 * Real.sin (2 * α) = Real.cos (α / 2)

theorem main_theorem (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  proof_problem α h1 h2 ∧ equality_case α h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l264_26439


namespace NUMINAMATH_GPT_symmetric_line_equation_l264_26476

-- Define the given lines
def original_line (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := y + 2 = 0

-- Define the problem statement as a theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ), line_of_symmetry x y → (original_line x (2 * (-2 - y) + 1)) ↔ (2 * x + y + 5 = 0) := 
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l264_26476


namespace NUMINAMATH_GPT_circle_center_l264_26440

theorem circle_center (x y: ℝ) : 
  (x + 2)^2 + (y + 3)^2 = 29 ↔ (∃ c1 c2 : ℝ, c1 = -2 ∧ c2 = -3) :=
by sorry

end NUMINAMATH_GPT_circle_center_l264_26440


namespace NUMINAMATH_GPT_part_one_part_two_l264_26443

-- Definitions for the propositions
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1)

-- Theorems for the answers
theorem part_one (m : ℝ) : ¬ proposition_p m → m < 1 :=
by sorry

theorem part_two (m : ℝ) : ¬ (proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) → m < 1 ∨ (4 ≤ m ∧ m ≤ 6) :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l264_26443


namespace NUMINAMATH_GPT_trapezoid_CD_length_l264_26416

theorem trapezoid_CD_length (AB CD AD BC : ℝ) (P : ℝ) 
  (h₁ : AB = 12) 
  (h₂ : AD = 5) 
  (h₃ : BC = 7) 
  (h₄ : P = 40) : CD = 16 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_CD_length_l264_26416


namespace NUMINAMATH_GPT_train_length_calculation_l264_26472

theorem train_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (platform_length_m : ℝ) (train_length_m: ℝ) : speed_kmph = 45 → time_seconds = 51.99999999999999 → platform_length_m = 290 → train_length_m = 360 :=
by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l264_26472


namespace NUMINAMATH_GPT_solution_set_of_inequality_l264_26403

theorem solution_set_of_inequality (a x : ℝ) (h1 : a < 2) (h2 : a * x > 2 * x + a - 2) : x < 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l264_26403


namespace NUMINAMATH_GPT_abs_cube_root_neg_64_l264_26421

-- Definitions required for the problem
def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_value (x : ℝ) : ℝ := abs x

-- The statement of the problem
theorem abs_cube_root_neg_64 : abs_value (cube_root (-64)) = 4 :=
by sorry

end NUMINAMATH_GPT_abs_cube_root_neg_64_l264_26421


namespace NUMINAMATH_GPT_hanging_spheres_ratio_l264_26428

theorem hanging_spheres_ratio (m1 m2 g T_B T_H : ℝ)
  (h1 : T_B = 3 * T_H)
  (h2 : T_H = m2 * g)
  (h3 : T_B = m1 * g + T_H)
  : m1 / m2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_hanging_spheres_ratio_l264_26428


namespace NUMINAMATH_GPT_cube_volume_ratio_l264_26405

theorem cube_volume_ratio (edge1 edge2 : ℕ) (h1 : edge1 = 10) (h2 : edge2 = 36) :
  (edge1^3 : ℚ) / (edge2^3) = 125 / 5832 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_ratio_l264_26405


namespace NUMINAMATH_GPT_confetti_left_correct_l264_26414

-- Define the number of pieces of red and green confetti collected by Eunji
def red_confetti : ℕ := 1
def green_confetti : ℕ := 9

-- Define the total number of pieces of confetti collected by Eunji
def total_confetti : ℕ := red_confetti + green_confetti

-- Define the number of pieces of confetti given to Yuna
def given_to_Yuna : ℕ := 4

-- Define the number of pieces of confetti left with Eunji
def confetti_left : ℕ :=  red_confetti + green_confetti - given_to_Yuna

-- Goal to prove
theorem confetti_left_correct : confetti_left = 6 := by
  -- Here the steps proving the equality would go, but we add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_confetti_left_correct_l264_26414


namespace NUMINAMATH_GPT_value_a6_l264_26410

noncomputable def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n - a (n - 1) = n - 1

theorem value_a6 : ∃ a : ℕ → ℕ, seq a ∧ a 6 = 16 := by
  sorry

end NUMINAMATH_GPT_value_a6_l264_26410


namespace NUMINAMATH_GPT_greatest_distance_is_correct_l264_26451

-- Define the coordinates of the post.
def post_coordinate : ℝ × ℝ := (6, -2)

-- Define the length of the rope.
def rope_length : ℝ := 12

-- Define the origin.
def origin : ℝ × ℝ := (0, 0)

-- Define the formula to calculate the Euclidean distance between two points in ℝ².
noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := by
  sorry

-- Define the distance from the origin to the post.
noncomputable def distance_origin_to_post : ℝ := euclidean_distance origin post_coordinate

-- Define the greatest distance the dog can be from the origin.
noncomputable def greatest_distance_from_origin : ℝ := distance_origin_to_post + rope_length

-- Prove that the greatest distance the dog can be from the origin is 12 + 2 * sqrt 10.
theorem greatest_distance_is_correct : greatest_distance_from_origin = 12 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_greatest_distance_is_correct_l264_26451


namespace NUMINAMATH_GPT_simplified_expression_l264_26436

theorem simplified_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2 * x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := 
by
  sorry

end NUMINAMATH_GPT_simplified_expression_l264_26436


namespace NUMINAMATH_GPT_cleaner_for_dog_stain_l264_26426

theorem cleaner_for_dog_stain (D : ℝ) (H : 6 * D + 3 * 4 + 1 * 1 = 49) : D = 6 :=
by 
  -- Proof steps would go here, but we are skipping the proof.
  sorry

end NUMINAMATH_GPT_cleaner_for_dog_stain_l264_26426


namespace NUMINAMATH_GPT_bob_needs_additional_weeks_l264_26402

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_bob_needs_additional_weeks_l264_26402


namespace NUMINAMATH_GPT_eval_f_neg2_l264_26453

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end NUMINAMATH_GPT_eval_f_neg2_l264_26453


namespace NUMINAMATH_GPT_evaluate_expression_l264_26448

theorem evaluate_expression (x : ℤ) (h : x + 1 = 4) : 
  (-3)^3 + (-3)^2 + (-3 * x) + 3 * x + 3^2 + 3^3 = 18 :=
by
  -- Since we know the condition x + 1 = 4
  have hx : x = 3 := by linarith
  -- Substitution x = 3 into the expression
  rw [hx]
  -- The expression after substitution and simplification
  sorry

end NUMINAMATH_GPT_evaluate_expression_l264_26448


namespace NUMINAMATH_GPT_binomial_sixteen_twelve_eq_l264_26493

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end NUMINAMATH_GPT_binomial_sixteen_twelve_eq_l264_26493


namespace NUMINAMATH_GPT_sector_radius_l264_26461

theorem sector_radius (θ : ℝ) (s : ℝ) (R : ℝ) 
  (hθ : θ = 150)
  (hs : s = (5 / 2) * Real.pi)
  : (θ / 360) * (2 * Real.pi * R) = (5 / 2) * Real.pi → 
  R = 3 := 
sorry

end NUMINAMATH_GPT_sector_radius_l264_26461


namespace NUMINAMATH_GPT_negation_of_forall_pos_l264_26480

open Real

theorem negation_of_forall_pos (h : ∀ x : ℝ, x^2 - x + 1 > 0) : 
  ¬(∀ x : ℝ, x^2 - x + 1 > 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_pos_l264_26480


namespace NUMINAMATH_GPT_sat_marking_problem_l264_26412

-- Define the recurrence relation for the number of ways to mark questions without consecutive markings of the same letter.
def f : ℕ → ℕ
| 0     => 1
| 1     => 2
| 2     => 3
| (n+2) => f (n+1) + f n

-- Define that each letter marking can be done in 32 different ways.
def markWays : ℕ := 32

-- Define the number of questions to be 10.
def numQuestions : ℕ := 10

-- Calculate the number of sequences of length numQuestions with no consecutive same markings.
def numWays := f numQuestions

-- Prove that the number of ways results in 2^20 * 3^10 and compute 100m + n + p where m = 20, n = 10, p = 3.
theorem sat_marking_problem :
  (numWays ^ 5 = 2 ^ 20 * 3 ^ 10) ∧ (100 * 20 + 10 + 3 = 2013) :=
by
  sorry

end NUMINAMATH_GPT_sat_marking_problem_l264_26412


namespace NUMINAMATH_GPT_school_robes_l264_26477

theorem school_robes (total_singers robes_needed : ℕ) (robe_cost total_spent existing_robes : ℕ) 
  (h1 : total_singers = 30)
  (h2 : robe_cost = 2)
  (h3 : total_spent = 36)
  (h4 : total_singers - total_spent / robe_cost = existing_robes) :
  existing_robes = 12 :=
by sorry

end NUMINAMATH_GPT_school_robes_l264_26477


namespace NUMINAMATH_GPT_ranges_of_a_and_m_l264_26438

open Set Real

def A : Set Real := {x | x^2 - 3*x + 2 = 0}
def B (a : Real) : Set Real := {x | x^2 - a*x + a - 1 = 0}
def C (m : Real) : Set Real := {x | x^2 - m*x + 2 = 0}

theorem ranges_of_a_and_m (a m : Real) :
  A ∪ B a = A → A ∩ C m = C m → (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2*sqrt 2 < m ∧ m < 2*sqrt 2)) :=
by
  have hA : A = {1, 2} := sorry
  sorry

end NUMINAMATH_GPT_ranges_of_a_and_m_l264_26438


namespace NUMINAMATH_GPT_slices_with_both_pepperoni_and_mushrooms_l264_26498

theorem slices_with_both_pepperoni_and_mushrooms (n : ℕ)
  (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (all_have_topping : ∀ (s : ℕ), s < total_slices → s < pepperoni_slices ∨ s < mushroom_slices ∨ s < (total_slices - pepperoni_slices - mushroom_slices) )
  (total_condition : total_slices = 16)
  (pepperoni_condition : pepperoni_slices = 8)
  (mushroom_condition : mushroom_slices = 12) :
  (8 - n) + (12 - n) + n = 16 → n = 4 :=
sorry

end NUMINAMATH_GPT_slices_with_both_pepperoni_and_mushrooms_l264_26498


namespace NUMINAMATH_GPT_three_pow_two_digits_count_l264_26401

theorem three_pow_two_digits_count : 
  ∃ n_set : Finset ℕ, (∀ n ∈ n_set, 10 ≤ 3^n ∧ 3^n < 100) ∧ n_set.card = 2 := 
sorry

end NUMINAMATH_GPT_three_pow_two_digits_count_l264_26401
