import Mathlib

namespace NUMINAMATH_GPT_current_job_wage_l2013_201342

variable (W : ℝ) -- Maisy's wage per hour at her current job

-- Define the conditions
def current_job_hours : ℝ := 8
def new_job_hours : ℝ := 4
def new_job_wage_per_hour : ℝ := 15
def new_job_bonus : ℝ := 35
def additional_new_job_earnings : ℝ := 15

-- Assert the given condition
axiom job_earnings_condition : 
  new_job_hours * new_job_wage_per_hour + new_job_bonus 
  = current_job_hours * W + additional_new_job_earnings

-- The theorem we want to prove
theorem current_job_wage : W = 10 := by
  sorry

end NUMINAMATH_GPT_current_job_wage_l2013_201342


namespace NUMINAMATH_GPT_ginger_size_l2013_201367

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end NUMINAMATH_GPT_ginger_size_l2013_201367


namespace NUMINAMATH_GPT_luigi_pizza_cost_l2013_201398

theorem luigi_pizza_cost (num_pizzas pieces_per_pizza cost_per_piece : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : pieces_per_pizza = 5) 
  (h3 : cost_per_piece = 4) :
  num_pizzas * pieces_per_pizza * cost_per_piece / pieces_per_pizza = 80 := by
  sorry

end NUMINAMATH_GPT_luigi_pizza_cost_l2013_201398


namespace NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_l2013_201356

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_l2013_201356


namespace NUMINAMATH_GPT_no_more_than_one_100_l2013_201319

-- Define the score variables and the conditions
variables (R P M : ℕ)

-- Given conditions: R = P - 3 and P = M - 7
def score_conditions : Prop := R = P - 3 ∧ P = M - 7

-- The maximum score condition
def max_score_condition : Prop := R ≤ 100 ∧ P ≤ 100 ∧ M ≤ 100

-- The goal: it is impossible for Vanya to have scored 100 in more than one exam
theorem no_more_than_one_100 (R P M : ℕ) (h1 : score_conditions R P M) (h2 : max_score_condition R P M) :
  (R = 100 ∧ P = 100) ∨ (P = 100 ∧ M = 100) ∨ (M = 100 ∧ R = 100) → false :=
sorry

end NUMINAMATH_GPT_no_more_than_one_100_l2013_201319


namespace NUMINAMATH_GPT_thabo_number_of_hardcover_nonfiction_books_l2013_201318

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_thabo_number_of_hardcover_nonfiction_books_l2013_201318


namespace NUMINAMATH_GPT_number_of_solution_pairs_l2013_201303

def integer_solutions_on_circle : Set (Int × Int) := {
  (1, 7), (1, -7), (-1, 7), (-1, -7),
  (5, 5), (5, -5), (-5, 5), (-5, -5),
  (7, 1), (7, -1), (-7, 1), (-7, -1) 
}

def system_of_equations_has_integer_solution (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), a * ↑x + b * ↑y = 1 ∧ (↑x ^ 2 + ↑y ^ 2 = 50)

theorem number_of_solution_pairs : ∃ (n : ℕ), n = 72 ∧
  (∀ (a b : ℝ), system_of_equations_has_integer_solution a b → n = 72) := 
sorry

end NUMINAMATH_GPT_number_of_solution_pairs_l2013_201303


namespace NUMINAMATH_GPT_volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l2013_201361

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end NUMINAMATH_GPT_volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l2013_201361


namespace NUMINAMATH_GPT_old_barbell_cost_l2013_201358

theorem old_barbell_cost (x : ℝ) (new_barbell_cost : ℝ) (h1 : new_barbell_cost = 1.30 * x) (h2 : new_barbell_cost = 325) : x = 250 :=
by
  sorry

end NUMINAMATH_GPT_old_barbell_cost_l2013_201358


namespace NUMINAMATH_GPT_molecular_weight_AlPO4_correct_l2013_201345

-- Noncomputable because we are working with specific numerical values.
noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_P : ℝ := 30.97
noncomputable def atomic_weight_O : ℝ := 16.00

noncomputable def molecular_weight_AlPO4 : ℝ := 
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

theorem molecular_weight_AlPO4_correct : molecular_weight_AlPO4 = 121.95 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_AlPO4_correct_l2013_201345


namespace NUMINAMATH_GPT_tangent_line_eqn_l2013_201386

theorem tangent_line_eqn (r x0 y0 : ℝ) (h : x0^2 + y0^2 = r^2) : 
  ∃ a b c : ℝ, a = x0 ∧ b = y0 ∧ c = r^2 ∧ (a*x + b*y = c) :=
sorry

end NUMINAMATH_GPT_tangent_line_eqn_l2013_201386


namespace NUMINAMATH_GPT_find_a_plus_b_l2013_201347

def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem find_a_plus_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) :
  a + b = 6 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2013_201347


namespace NUMINAMATH_GPT_calculate_group_A_B_C_and_total_is_correct_l2013_201305

def groupA_1week : Int := 175000
def groupA_2week : Int := 107000
def groupA_3week : Int := 35000
def groupB_1week : Int := 100000
def groupB_2week : Int := 70350
def groupB_3week : Int := 19500
def groupC_1week : Int := 45000
def groupC_2week : Int := 87419
def groupC_3week : Int := 14425
def kids_staying_home : Int := 590796
def kids_outside_county : Int := 22

def total_kids_in_A := groupA_1week + groupA_2week + groupA_3week
def total_kids_in_B := groupB_1week + groupB_2week + groupB_3week
def total_kids_in_C := groupC_1week + groupC_2week + groupC_3week
def total_kids_in_camp := total_kids_in_A + total_kids_in_B + total_kids_in_C
def total_kids := total_kids_in_camp + kids_staying_home + kids_outside_county

theorem calculate_group_A_B_C_and_total_is_correct :
  total_kids_in_A = 317000 ∧
  total_kids_in_B = 189850 ∧
  total_kids_in_C = 146844 ∧
  total_kids = 1244512 := by
  sorry

end NUMINAMATH_GPT_calculate_group_A_B_C_and_total_is_correct_l2013_201305


namespace NUMINAMATH_GPT_tamara_is_17_over_6_times_taller_than_kim_l2013_201324

theorem tamara_is_17_over_6_times_taller_than_kim :
  ∀ (T K : ℕ), T = 68 → T + K = 92 → (T : ℚ) / K = 17 / 6 :=
by
  intros T K hT hSum
  -- proof steps go here, but we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_tamara_is_17_over_6_times_taller_than_kim_l2013_201324


namespace NUMINAMATH_GPT_prove_seq_properties_l2013_201387

theorem prove_seq_properties (a b : ℕ → ℕ) (S T : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_sum : ∀ n, 2 * S n = a n ^ 2 + n)
  (h_b : ∀ n, b n = a (n + 1) * 2 ^ n)
  : (∀ n, a n = n) ∧ (∀ n, T n = n * 2 ^ (n + 1)) :=
sorry

end NUMINAMATH_GPT_prove_seq_properties_l2013_201387


namespace NUMINAMATH_GPT_ms_warren_walking_speed_correct_l2013_201343

noncomputable def walking_speed_proof : Prop :=
  let running_speed := 6 -- mph
  let running_time := 20 / 60 -- hours
  let total_distance := 3 -- miles
  let distance_ran := running_speed * running_time
  let distance_walked := total_distance - distance_ran
  let walking_time := 30 / 60 -- hours
  let walking_speed := distance_walked / walking_time
  walking_speed = 2

theorem ms_warren_walking_speed_correct (walking_speed_proof : Prop) : walking_speed_proof :=
by sorry

end NUMINAMATH_GPT_ms_warren_walking_speed_correct_l2013_201343


namespace NUMINAMATH_GPT_tangent_circle_line_l2013_201344

theorem tangent_circle_line (a : ℝ) :
  (∀ x y : ℝ, (x - y + 3 = 0) → (x^2 + y^2 - 2 * x + 2 - a = 0)) →
  a = 9 :=
by
  sorry

end NUMINAMATH_GPT_tangent_circle_line_l2013_201344


namespace NUMINAMATH_GPT_find_value_l2013_201338

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom symmetric_about_one : ∀ x, f (x - 1) = f (1 - x)
axiom equation_on_interval : ∀ x, 0 < x ∧ x < 1 → f x = 9^x

theorem find_value : f (5 / 2) + f 2 = -3 := 
by sorry

end NUMINAMATH_GPT_find_value_l2013_201338


namespace NUMINAMATH_GPT_not_right_triangle_l2013_201332

theorem not_right_triangle (A B C : ℝ) (hA : A + B = 180 - C) 
  (hB : A = B / 2 ∧ A = C / 3) 
  (hC : A = B / 2 ∧ B = C / 1.5) 
  (hD : A = 2 * B ∧ A = 3 * C):
  (C ≠ 90) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_right_triangle_l2013_201332


namespace NUMINAMATH_GPT_remainder_3012_div_96_l2013_201389

theorem remainder_3012_div_96 : 3012 % 96 = 36 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_3012_div_96_l2013_201389


namespace NUMINAMATH_GPT_simplify_expression_l2013_201373

variables {a b : ℝ}

theorem simplify_expression (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2013_201373


namespace NUMINAMATH_GPT_inequality_hold_l2013_201335

theorem inequality_hold (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧ 
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ 1/8 :=
sorry

end NUMINAMATH_GPT_inequality_hold_l2013_201335


namespace NUMINAMATH_GPT_largest_possible_sum_l2013_201309

theorem largest_possible_sum :
  let a := 12
  let b := 6
  let c := 6
  let d := 12
  a + b = c + d ∧ a + b + 15 = 33 :=
by
  have h1 : 12 + 6 = 6 + 12 := by norm_num
  have h2 : 12 + 6 + 15 = 33 := by norm_num
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_largest_possible_sum_l2013_201309


namespace NUMINAMATH_GPT_monkey_hop_distance_l2013_201331

theorem monkey_hop_distance
    (total_height : ℕ)
    (slip_back : ℕ)
    (hours : ℕ)
    (reach_time : ℕ)
    (hop : ℕ)
    (H1 : total_height = 19)
    (H2 : slip_back = 2)
    (H3 : hours = 17)
    (H4 : reach_time = 16 * (hop - slip_back) + hop)
    (H5 : total_height = reach_time) :
    hop = 3 := by
  sorry

end NUMINAMATH_GPT_monkey_hop_distance_l2013_201331


namespace NUMINAMATH_GPT_range_of_a_l2013_201385

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x < a → x ^ 2 > 1 ∧ ¬(x ^ 2 > 1 → x < a)) : a ≤ -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2013_201385


namespace NUMINAMATH_GPT_remainder_when_doubling_l2013_201392

theorem remainder_when_doubling:
  ∀ (n k : ℤ), n = 30 * k + 16 → (2 * n) % 15 = 2 :=
by
  intros n k h
  sorry

end NUMINAMATH_GPT_remainder_when_doubling_l2013_201392


namespace NUMINAMATH_GPT_pam_bags_count_l2013_201372

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end NUMINAMATH_GPT_pam_bags_count_l2013_201372


namespace NUMINAMATH_GPT_henrietta_paint_gallons_l2013_201307

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end NUMINAMATH_GPT_henrietta_paint_gallons_l2013_201307


namespace NUMINAMATH_GPT_increase_in_circumference_l2013_201316

theorem increase_in_circumference (d e : ℝ) : (fun d e => let C := π * d; let C_new := π * (d + e); C_new - C) d e = π * e :=
by sorry

end NUMINAMATH_GPT_increase_in_circumference_l2013_201316


namespace NUMINAMATH_GPT_range_of_t_l2013_201380

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x + a * Real.log x

noncomputable def g (a x : ℝ) : ℝ :=
  f a x + 1/2 * x^2 - (a - 1) * x - a / x

theorem range_of_t (a x₁ x₂ t : ℝ) (h1 : f a x₁ = f a x₂) (h2 : x₁ + x₂ = a)
  (h3 : x₁ * x₂ = a) (h4 : a > 4) (h5 : g a x₁ + g a x₂ > t * (x₁ + x₂)) :
  t < Real.log 4 - 3 :=
  sorry

end NUMINAMATH_GPT_range_of_t_l2013_201380


namespace NUMINAMATH_GPT_students_who_chose_water_l2013_201378

-- Defining the conditions
def percent_juice : ℚ := 75 / 100
def percent_water : ℚ := 25 / 100
def students_who_chose_juice : ℚ := 90
def ratio_water_to_juice : ℚ := percent_water / percent_juice  -- This should equal 1/3

-- The theorem we need to prove
theorem students_who_chose_water : students_who_chose_juice * ratio_water_to_juice = 30 := 
by
  sorry

end NUMINAMATH_GPT_students_who_chose_water_l2013_201378


namespace NUMINAMATH_GPT_sum_of_squares_first_28_l2013_201327

theorem sum_of_squares_first_28 : 
  (28 * (28 + 1) * (2 * 28 + 1)) / 6 = 7722 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_first_28_l2013_201327


namespace NUMINAMATH_GPT_average_score_bounds_l2013_201379

/-- Problem data definitions -/
def n_100 : ℕ := 2
def n_90_99 : ℕ := 9
def n_80_89 : ℕ := 17
def n_70_79 : ℕ := 28
def n_60_69 : ℕ := 36
def n_50_59 : ℕ := 7
def n_48 : ℕ := 1

def sum_scores_min : ℕ := (100 * n_100 + 90 * n_90_99 + 80 * n_80_89 + 70 * n_70_79 + 60 * n_60_69 + 50 * n_50_59 + 48)
def sum_scores_max : ℕ := (100 * n_100 + 99 * n_90_99 + 89 * n_80_89 + 79 * n_70_79 + 69 * n_60_69 + 59 * n_50_59 + 48)
def total_people : ℕ := n_100 + n_90_99 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_48

/-- Prove the minimum and maximum average scores. -/
theorem average_score_bounds :
  (sum_scores_min / total_people : ℚ) = 68.88 ∧
  (sum_scores_max / total_people : ℚ) = 77.61 :=
by
  sorry

end NUMINAMATH_GPT_average_score_bounds_l2013_201379


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l2013_201383

theorem ratio_of_boys_to_girls {T G B : ℕ} (h1 : (2/3 : ℚ) * G = (1/4 : ℚ) * T) (h2 : T = G + B) : (B : ℚ) / G = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l2013_201383


namespace NUMINAMATH_GPT_probability_at_least_two_boys_one_girl_l2013_201384

-- Define what constitutes a family of four children
def family := {s : Fin 4 → Bool // ∃ (b g : Fin 4), b ≠ g}

-- Define the probability equation
noncomputable def probability_of_boy_or_girl : ℚ := 1 / 2

-- Define what it means to have at least two boys and one girl
def at_least_two_boys_one_girl (f : family) : Prop :=
  ∃ (count_boys count_girls : ℕ), count_boys + count_girls = 4 
  ∧ count_boys ≥ 2 
  ∧ count_girls ≥ 1

-- Calculate the probability
theorem probability_at_least_two_boys_one_girl : 
  (∃ (f : family), at_least_two_boys_one_girl f) →
  probability_of_boy_or_girl ^ 4 * ( (6 / 16 : ℚ) + (4 / 16 : ℚ) + (1 / 16 : ℚ) ) = 11 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_two_boys_one_girl_l2013_201384


namespace NUMINAMATH_GPT_remaining_bananas_l2013_201399

def original_bananas : ℕ := 46
def removed_bananas : ℕ := 5

theorem remaining_bananas : original_bananas - removed_bananas = 41 := by
  sorry

end NUMINAMATH_GPT_remaining_bananas_l2013_201399


namespace NUMINAMATH_GPT_range_of_f_when_a_eq_2_max_value_implies_a_l2013_201382

-- first part
theorem range_of_f_when_a_eq_2 (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 3) :
  (∀ y, (y = x^2 + 3*x - 3) → (y ≥ -21/4 ∧ y ≤ 15)) :=
by sorry

-- second part
theorem max_value_implies_a (a : ℝ) (hx : ∀ x, -1 ≤ x ∧ x ≤ 3 → x^2 + (2*a - 1)*x - 3 ≤ 1) :
  a = -1 ∨ a = -1 / 3 :=
by sorry

end NUMINAMATH_GPT_range_of_f_when_a_eq_2_max_value_implies_a_l2013_201382


namespace NUMINAMATH_GPT_equation_equivalence_and_rst_l2013_201394

theorem equation_equivalence_and_rst 
  (a x y c : ℝ) 
  (r s t : ℤ) 
  (h1 : r = 3) 
  (h2 : s = 1) 
  (h3 : t = 5)
  (h_eq1 : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^3) = a^5 * c^5 ∧ r * s * t = 15 :=
by
  sorry

end NUMINAMATH_GPT_equation_equivalence_and_rst_l2013_201394


namespace NUMINAMATH_GPT_length_of_other_leg_l2013_201366

theorem length_of_other_leg (c a b : ℕ) (h1 : c = 10) (h2 : a = 6) (h3 : c^2 = a^2 + b^2) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_other_leg_l2013_201366


namespace NUMINAMATH_GPT_count_two_digit_integers_sum_seven_l2013_201364

theorem count_two_digit_integers_sum_seven : 
  ∃ n : ℕ, (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 7 → n = 7) := 
by
  sorry

end NUMINAMATH_GPT_count_two_digit_integers_sum_seven_l2013_201364


namespace NUMINAMATH_GPT_decreasing_on_negative_interval_and_max_value_l2013_201360

open Classical

noncomputable def f : ℝ → ℝ := sorry  -- Define f later

variables {f : ℝ → ℝ}

-- Hypotheses
axiom h_even : ∀ x, f x = f (-x)
axiom h_increasing_0_7 : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → y ≤ 7 → f x ≤ f y
axiom h_decreasing_7_inf : ∀ ⦃x y : ℝ⦄, 7 ≤ x → x ≤ y → f x ≥ f y
axiom h_f_7_6 : f 7 = 6

-- Theorem Statement
theorem decreasing_on_negative_interval_and_max_value :
  (∀ ⦃x y : ℝ⦄, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_on_negative_interval_and_max_value_l2013_201360


namespace NUMINAMATH_GPT_exists_complex_on_line_y_eq_neg_x_l2013_201301

open Complex

theorem exists_complex_on_line_y_eq_neg_x :
  ∃ (z : ℂ), ∃ (a b : ℝ), z = a + b * I ∧ b = -a :=
by
  use 1 - I
  use 1, -1
  sorry

end NUMINAMATH_GPT_exists_complex_on_line_y_eq_neg_x_l2013_201301


namespace NUMINAMATH_GPT_cone_volume_divided_by_pi_l2013_201374

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def sector_to_cone_radius (arc_len : ℝ) : ℝ := arc_len / (2 * Real.pi)

noncomputable def cone_height (r_base : ℝ) (slant_height : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def cone_volume (r_base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r_base ^ 2 * height

theorem cone_volume_divided_by_pi (r slant_height θ : ℝ) (h : slant_height = 15 ∧ θ = 270):
  cone_volume (sector_to_cone_radius (arc_length r θ)) (cone_height (sector_to_cone_radius (arc_length r θ)) slant_height) / Real.pi = (453.515625 * Real.sqrt 10.9375) :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_divided_by_pi_l2013_201374


namespace NUMINAMATH_GPT_rectangle_width_is_nine_l2013_201334

theorem rectangle_width_is_nine (w l : ℝ) (h1 : l = 2 * w)
  (h2 : l * w = 3 * 2 * (l + w)) : 
  w = 9 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_is_nine_l2013_201334


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2013_201312

-- Define what we need to prove
theorem problem_part1 (x : ℝ) (a b : ℤ) 
  (h : (2*x - 21)*(3*x - 7) - (3*x - 7)*(x - 13) = (3*x + a)*(x + b)): 
  a + 3*b = -31 := 
by {
  -- We know from the problem that h holds,
  -- thus the values of a and b must satisfy the condition.
  sorry
}

theorem problem_part2 (x : ℝ) : 
  (x^2 - 3*x + 2) = (x - 1)*(x - 2) := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_part1_problem_part2_l2013_201312


namespace NUMINAMATH_GPT_reduced_price_after_exchange_rate_fluctuation_l2013_201328

-- Definitions based on conditions
variables (P : ℝ) -- Original price per kg

def reduced_price_per_kg : ℝ := 0.9 * P

axiom six_kg_costs_900 : 6 * reduced_price_per_kg P = 900

-- Additional conditions
def exchange_rate_factor : ℝ := 1.02

-- Question restated as the theorem to prove
theorem reduced_price_after_exchange_rate_fluctuation : 
  ∃ P : ℝ, reduced_price_per_kg P * exchange_rate_factor = 153 :=
sorry

end NUMINAMATH_GPT_reduced_price_after_exchange_rate_fluctuation_l2013_201328


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l2013_201326

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (k : ℕ) (hk : k ≥ 2)
  (hS : S k = 5)
  (ha_k2_p1 : a (k^2 + 1) = -45)
  (ha_sum : (Finset.range (2 * k + 1) \ Finset.range (k + 1)).sum a = -45) :
  a 1 = 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l2013_201326


namespace NUMINAMATH_GPT_linear_regression_equation_demand_prediction_l2013_201339

def data_x : List ℝ := [12, 11, 10, 9, 8]
def data_y : List ℝ := [5, 6, 8, 10, 11]

noncomputable def mean_x : ℝ := (12 + 11 + 10 + 9 + 8) / 5
noncomputable def mean_y : ℝ := (5 + 6 + 8 + 10 + 11) / 5

noncomputable def numerator : ℝ := 
  (12 - mean_x) * (5 - mean_y) + 
  (11 - mean_x) * (6 - mean_y) +
  (10 - mean_x) * (8 - mean_y) +
  (9 - mean_x) * (10 - mean_y) +
  (8 - mean_x) * (11 - mean_y)

noncomputable def denominator : ℝ := 
  (12 - mean_x)^2 + 
  (11 - mean_x)^2 +
  (10 - mean_x)^2 +
  (9 - mean_x)^2 +
  (8 - mean_x)^2

noncomputable def slope_b : ℝ := numerator / denominator
noncomputable def intercept_a : ℝ := mean_y - slope_b * mean_x

theorem linear_regression_equation :
  (slope_b = -1.6) ∧ (intercept_a = 24) :=
by
  sorry

noncomputable def predicted_y (x : ℝ) : ℝ :=
  slope_b * x + intercept_a

theorem demand_prediction :
  predicted_y 6 = 14.4 ∧ (predicted_y 6 < 15) :=
by
  sorry

end NUMINAMATH_GPT_linear_regression_equation_demand_prediction_l2013_201339


namespace NUMINAMATH_GPT_xiao_ming_excellent_score_probability_l2013_201369

theorem xiao_ming_excellent_score_probability :
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  P_E = 0.2 :=
by
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  sorry

end NUMINAMATH_GPT_xiao_ming_excellent_score_probability_l2013_201369


namespace NUMINAMATH_GPT_tony_lift_ratio_l2013_201306

noncomputable def curl_weight := 90
noncomputable def military_press_weight := 2 * curl_weight
noncomputable def squat_weight := 900

theorem tony_lift_ratio : 
  squat_weight / military_press_weight = 5 :=
by
  sorry

end NUMINAMATH_GPT_tony_lift_ratio_l2013_201306


namespace NUMINAMATH_GPT_sub_from_square_l2013_201322

theorem sub_from_square (n : ℕ) (h : n = 17) : (n * n - n) = 272 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sub_from_square_l2013_201322


namespace NUMINAMATH_GPT_card_paiting_modulus_l2013_201337

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end NUMINAMATH_GPT_card_paiting_modulus_l2013_201337


namespace NUMINAMATH_GPT_abs_c_eq_181_l2013_201363

theorem abs_c_eq_181
  (a b c : ℤ)
  (h_gcd : Int.gcd a (Int.gcd b c) = 1)
  (h_eq : a * (Complex.mk 3 2)^4 + b * (Complex.mk 3 2)^3 + c * (Complex.mk 3 2)^2 + b * (Complex.mk 3 2) + a = 0) :
  |c| = 181 :=
sorry

end NUMINAMATH_GPT_abs_c_eq_181_l2013_201363


namespace NUMINAMATH_GPT_probability_red_or_green_l2013_201313

variable (P_brown P_purple P_green P_red P_yellow : ℝ)

def conditions : Prop :=
  P_brown = 0.3 ∧
  P_brown = 3 * P_purple ∧
  P_green = P_purple ∧
  P_red = P_yellow ∧
  P_brown + P_purple + P_green + P_red + P_yellow = 1

theorem probability_red_or_green (h : conditions P_brown P_purple P_green P_red P_yellow) :
  P_red + P_green = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_or_green_l2013_201313


namespace NUMINAMATH_GPT_sqrt_condition_sqrt_not_meaningful_2_l2013_201370

theorem sqrt_condition (x : ℝ) : 1 - x ≥ 0 ↔ x ≤ 1 := 
by
  sorry

theorem sqrt_not_meaningful_2 : ¬(1 - 2 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_condition_sqrt_not_meaningful_2_l2013_201370


namespace NUMINAMATH_GPT_triangle_area_from_curve_l2013_201362

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_from_curve_l2013_201362


namespace NUMINAMATH_GPT_y_values_l2013_201357

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sin x / |Real.sin x|) + (|Real.cos x| / Real.cos x) + (Real.tan x / |Real.tan x|)

theorem y_values (x : ℝ) (h1 : 0 < x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x ≠ 0) (h4 : Real.cos x ≠ 0) (h5 : Real.tan x ≠ 0) :
  y x = 3 ∨ y x = -1 :=
sorry

end NUMINAMATH_GPT_y_values_l2013_201357


namespace NUMINAMATH_GPT_Bert_total_profit_is_14_90_l2013_201333

-- Define the sales price for each item
def sales_price_barrel : ℝ := 90
def sales_price_tools : ℝ := 50
def sales_price_fertilizer : ℝ := 30

-- Define the tax rates for each item
def tax_rate_barrel : ℝ := 0.10
def tax_rate_tools : ℝ := 0.05
def tax_rate_fertilizer : ℝ := 0.12

-- Define the profit added per item
def profit_per_item : ℝ := 10

-- Define the tax amount for each item
def tax_barrel : ℝ := tax_rate_barrel * sales_price_barrel
def tax_tools : ℝ := tax_rate_tools * sales_price_tools
def tax_fertilizer : ℝ := tax_rate_fertilizer * sales_price_fertilizer

-- Define the cost price for each item
def cost_price_barrel : ℝ := sales_price_barrel - profit_per_item
def cost_price_tools : ℝ := sales_price_tools - profit_per_item
def cost_price_fertilizer : ℝ := sales_price_fertilizer - profit_per_item

-- Define the profit for each item
def profit_barrel : ℝ := sales_price_barrel - tax_barrel - cost_price_barrel
def profit_tools : ℝ := sales_price_tools - tax_tools - cost_price_tools
def profit_fertilizer : ℝ := sales_price_fertilizer - tax_fertilizer - cost_price_fertilizer

-- Define the total profit
def total_profit : ℝ := profit_barrel + profit_tools + profit_fertilizer

-- Assert the total profit is $14.90
theorem Bert_total_profit_is_14_90 : total_profit = 14.90 :=
by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_Bert_total_profit_is_14_90_l2013_201333


namespace NUMINAMATH_GPT_A_alone_work_days_l2013_201368

noncomputable def A_and_B_together : ℕ := 40
noncomputable def A_and_B_worked_together_days : ℕ := 10
noncomputable def B_left_and_C_joined_after_days : ℕ := 6
noncomputable def A_and_C_finish_remaining_work_days : ℕ := 15
noncomputable def C_alone_work_days : ℕ := 60

theorem A_alone_work_days (h1 : A_and_B_together = 40)
                          (h2 : A_and_B_worked_together_days = 10)
                          (h3 : B_left_and_C_joined_after_days = 6)
                          (h4 : A_and_C_finish_remaining_work_days = 15)
                          (h5 : C_alone_work_days = 60) : ∃ (n : ℕ), n = 30 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_A_alone_work_days_l2013_201368


namespace NUMINAMATH_GPT_jane_mean_score_l2013_201320

-- Define the six quiz scores Jane took
def score1 : ℕ := 86
def score2 : ℕ := 91
def score3 : ℕ := 89
def score4 : ℕ := 95
def score5 : ℕ := 88
def score6 : ℕ := 94

-- The number of quizzes
def num_quizzes : ℕ := 6

-- The sum of all quiz scores
def total_score : ℕ := score1 + score2 + score3 + score4 + score5 + score6 

-- The expected mean score
def mean_score : ℚ := 90.5

-- The proof statement
theorem jane_mean_score (h : total_score = 543) : total_score / num_quizzes = mean_score := 
by sorry

end NUMINAMATH_GPT_jane_mean_score_l2013_201320


namespace NUMINAMATH_GPT_number_of_results_l2013_201381

theorem number_of_results (n : ℕ)
  (avg_all : (summation : ℤ) → summation / n = 42)
  (avg_first_5 : (sum_first_5 : ℤ) → sum_first_5 / 5 = 49)
  (avg_last_7 : (sum_last_7 : ℤ) → sum_last_7 / 7 = 52)
  (fifth_result : (r5 : ℤ) → r5 = 147) :
  n = 11 :=
by
  -- Conditions
  let sum_first_5 := 5 * 49
  let sum_last_7 := 7 * 52
  let summed_results := sum_first_5 + sum_last_7 - 147
  let sum_all := 42 * n 
  -- Since sum of all results = 42n
  exact sorry

end NUMINAMATH_GPT_number_of_results_l2013_201381


namespace NUMINAMATH_GPT_option_C_correct_l2013_201311

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end NUMINAMATH_GPT_option_C_correct_l2013_201311


namespace NUMINAMATH_GPT_max_cards_from_poster_board_l2013_201308

theorem max_cards_from_poster_board (card_length card_width poster_length : ℕ) (h1 : card_length = 2) (h2 : card_width = 3) (h3 : poster_length = 12) : 
  (poster_length / card_length) * (poster_length / card_width) = 24 :=
by
  sorry

end NUMINAMATH_GPT_max_cards_from_poster_board_l2013_201308


namespace NUMINAMATH_GPT_f_96_l2013_201354

noncomputable def f : ℕ → ℝ := sorry -- assume f is defined somewhere

axiom f_property (a b k : ℕ) (h : a + b = 3 * 2^k) : f a + f b = 2 * k^2

theorem f_96 : f 96 = 20 :=
by
  -- Here we should provide the proof, but for now we use sorry
  sorry

end NUMINAMATH_GPT_f_96_l2013_201354


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2013_201395

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2013_201395


namespace NUMINAMATH_GPT_permits_cost_l2013_201302

-- Definitions based on conditions
def total_cost : ℕ := 2950
def contractor_hourly_rate : ℕ := 150
def contractor_hours_per_day : ℕ := 5
def contractor_days : ℕ := 3
def inspector_discount_rate : ℕ := 80

-- Proving the cost of permits
theorem permits_cost : ∃ (permits_cost : ℕ), permits_cost = 250 :=
by
  let contractor_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (contractor_hourly_rate * inspector_discount_rate / 100)
  let inspector_cost := contractor_hours * inspector_hourly_rate
  let total_cost_without_permits := contractor_cost + inspector_cost
  let permits_cost := total_cost - total_cost_without_permits
  use permits_cost
  sorry

end NUMINAMATH_GPT_permits_cost_l2013_201302


namespace NUMINAMATH_GPT_point_in_third_quadrant_l2013_201336

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 1 + 2 * m < 0) : m < -1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l2013_201336


namespace NUMINAMATH_GPT_value_of_A_l2013_201321

theorem value_of_A (G F L: ℤ) (H1 : G = 15) (H2 : F + L + 15 = 50) (H3 : F + L + 37 + 15 = 65) (H4 : F + ((58 - F - L) / 2) + ((58 - F - L) / 2) + L = 58) : 
  37 = 37 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_A_l2013_201321


namespace NUMINAMATH_GPT_find_S5_l2013_201393

-- Assuming the sequence is geometric and defining the conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Definitions of the conditions based on the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 5 = 3 * a 3

def condition_2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 9 * a 7) / 2 = 2

-- Sum of the first n terms of a geometric sequence
noncomputable def S_n (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

-- The theorem stating the final goal
theorem find_S5 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) 
    (h1 : condition_1 a q) (h2 : condition_2 a q) : S_n a q 5 = 121 :=
by
  -- This adds "sorry" to bypass the actual proof
  sorry

end NUMINAMATH_GPT_find_S5_l2013_201393


namespace NUMINAMATH_GPT_max_value_a4_b4_c4_d4_l2013_201315

theorem max_value_a4_b4_c4_d4 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  a^4 + b^4 + c^4 + d^4 ≤ 64 :=
sorry

end NUMINAMATH_GPT_max_value_a4_b4_c4_d4_l2013_201315


namespace NUMINAMATH_GPT_length_of_crate_l2013_201317

theorem length_of_crate (h crate_dim : ℕ) (radius : ℕ) (h_radius : radius = 8) 
  (h_dims : crate_dim = 18) (h_fit : 2 * radius = 16)
  : h = 18 := 
sorry

end NUMINAMATH_GPT_length_of_crate_l2013_201317


namespace NUMINAMATH_GPT_div_by_eight_l2013_201365

theorem div_by_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_div_by_eight_l2013_201365


namespace NUMINAMATH_GPT_regular_polygon_property_l2013_201349

variables {n : ℕ}
variables {r : ℝ} -- r is the radius of the circumscribed circle
variables {t_2n : ℝ} -- t_2n is the area of the 2n-gon
variables {k_n : ℝ} -- k_n is the perimeter of the n-gon

theorem regular_polygon_property
  (h1 : t_2n = (n * k_n * r) / 2)
  (h2 : k_n = n * a_n) :
  (t_2n / r^2) = (k_n / (2 * r)) :=
by sorry

end NUMINAMATH_GPT_regular_polygon_property_l2013_201349


namespace NUMINAMATH_GPT_part1_part2_l2013_201346

theorem part1 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (cos_AB: ℝ), cos_AB = 56 / 65 :=
by {
  sorry
}

theorem part2 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (area: ℝ), area = 126 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l2013_201346


namespace NUMINAMATH_GPT_find_abcd_abs_eq_one_l2013_201396

noncomputable def non_zero_real (r : ℝ) := r ≠ 0

theorem find_abcd_abs_eq_one
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : a^2 + (1/b) = b^2 + (1/c) ∧ b^2 + (1/c) = c^2 + (1/d) ∧ c^2 + (1/d) = d^2 + (1/a)) :
  |a * b * c * d| = 1 :=
sorry

end NUMINAMATH_GPT_find_abcd_abs_eq_one_l2013_201396


namespace NUMINAMATH_GPT_interest_earned_l2013_201376

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (I : ℝ):
  P = 1500 → r = 0.12 → n = 4 →
  A = compound_interest P r n →
  I = A - P →
  I = 862.2 :=
by
  intros hP hr hn hA hI
  sorry

end NUMINAMATH_GPT_interest_earned_l2013_201376


namespace NUMINAMATH_GPT_greatest_x_value_l2013_201348

theorem greatest_x_value : 
  (∃ x : ℝ, 2 * x^2 + 7 * x + 3 = 5 ∧ ∀ y : ℝ, (2 * y^2 + 7 * y + 3 = 5) → y ≤ x) → x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_value_l2013_201348


namespace NUMINAMATH_GPT_Parkway_Elementary_girls_not_playing_soccer_l2013_201304

/-
  In the fifth grade at Parkway Elementary School, there are 500 students. 
  350 students are boys and 250 students are playing soccer.
  86% of the students that play soccer are boys.
  Prove that the number of girl students that are not playing soccer is 115.
-/
theorem Parkway_Elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (H1 : total_students = 500)
  (H2 : boys = 350)
  (H3 : playing_soccer = 250)
  (H4 : percentage_boys_playing_soccer = 0.86) :
  ∃ (girls_not_playing_soccer : ℕ), girls_not_playing_soccer = 115 :=
by
  sorry

end NUMINAMATH_GPT_Parkway_Elementary_girls_not_playing_soccer_l2013_201304


namespace NUMINAMATH_GPT_edric_hours_per_day_l2013_201397

/--
Edric's monthly salary is $576. He works 6 days a week for 4 weeks in a month and 
his hourly rate is $3. Prove that Edric works 8 hours in a day.
-/
theorem edric_hours_per_day (m : ℕ) (r : ℕ) (d : ℕ) (w : ℕ)
  (h_m : m = 576) (h_r : r = 3) (h_d : d = 6) (h_w : w = 4) :
  (m / r) / (d * w) = 8 := by
    sorry

end NUMINAMATH_GPT_edric_hours_per_day_l2013_201397


namespace NUMINAMATH_GPT_estimate_correctness_l2013_201371

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end NUMINAMATH_GPT_estimate_correctness_l2013_201371


namespace NUMINAMATH_GPT_paper_sufficient_to_cover_cube_l2013_201355

noncomputable def edge_length_cube : ℝ := 1
noncomputable def side_length_sheet : ℝ := 2.5

noncomputable def surface_area_cube : ℝ := 6
noncomputable def area_sheet : ℝ := 6.25

theorem paper_sufficient_to_cover_cube : area_sheet ≥ surface_area_cube :=
  by
    sorry

end NUMINAMATH_GPT_paper_sufficient_to_cover_cube_l2013_201355


namespace NUMINAMATH_GPT_find_b_l2013_201350

theorem find_b (a b : ℝ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l2013_201350


namespace NUMINAMATH_GPT_simplify_polynomial_l2013_201353

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2013_201353


namespace NUMINAMATH_GPT_students_enrolled_for_german_l2013_201314

theorem students_enrolled_for_german 
  (total_students : ℕ)
  (both_english_german : ℕ)
  (only_english : ℕ)
  (at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ only_english = 10) :
  ∃ G : ℕ, G = 22 :=
by
  -- Lean proof steps will go here.
  sorry

end NUMINAMATH_GPT_students_enrolled_for_german_l2013_201314


namespace NUMINAMATH_GPT_differential_savings_l2013_201330

def original_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

theorem differential_savings : (original_tax_rate * annual_income) - (new_tax_rate * annual_income) = 7200 := by
  sorry

end NUMINAMATH_GPT_differential_savings_l2013_201330


namespace NUMINAMATH_GPT_find_y_l2013_201391

noncomputable def inverse_proportion_y_value (x y k : ℝ) : Prop :=
  (x * y = k) ∧ (x + y = 52) ∧ (x = 3 * y) ∧ (x = -10) → (y = -50.7)

theorem find_y (x y k : ℝ) (h : inverse_proportion_y_value x y k) : y = -50.7 :=
  sorry

end NUMINAMATH_GPT_find_y_l2013_201391


namespace NUMINAMATH_GPT_simplify_expression_l2013_201377

open Real

-- Define the given expression as a function of x
noncomputable def given_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (2 * (1 + sqrt (1 + ( (x^4 - 1) / (2 * x^2) )^2)))

-- Define the expected simplified expression
noncomputable def expected_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^2 + 1) / x

-- Proof statement to verify the simplification
theorem simplify_expression (x : ℝ) (hx : 0 < x) :
  given_expression x hx = expected_expression x hx :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2013_201377


namespace NUMINAMATH_GPT_tom_has_hours_to_spare_l2013_201359

theorem tom_has_hours_to_spare 
  (num_walls : ℕ) 
  (wall_length wall_height : ℕ) 
  (painting_rate : ℕ) 
  (total_hours : ℕ) 
  (num_walls_eq : num_walls = 5) 
  (wall_length_eq : wall_length = 2) 
  (wall_height_eq : wall_height = 3) 
  (painting_rate_eq : painting_rate = 10) 
  (total_hours_eq : total_hours = 10)
  : total_hours - (num_walls * wall_length * wall_height * painting_rate) / 60 = 5 := 
sorry

end NUMINAMATH_GPT_tom_has_hours_to_spare_l2013_201359


namespace NUMINAMATH_GPT_positively_correlated_variables_l2013_201352

-- Define all conditions given in the problem
def weightOfCarVar1 : Type := ℝ
def avgDistPerLiter : Type := ℝ
def avgStudyTime : Type := ℝ
def avgAcademicPerformance : Type := ℝ
def dailySmokingAmount : Type := ℝ
def healthCondition : Type := ℝ
def sideLength : Type := ℝ
def areaOfSquare : Type := ℝ
def fuelConsumptionPerHundredKm : Type := ℝ

-- Define the relationship status between variables
def isPositivelyCorrelated (x y : Type) : Prop := sorry
def isFunctionallyRelated (x y : Type) : Prop := sorry

axiom weight_car_distance_neg : ¬ isPositivelyCorrelated weightOfCarVar1 avgDistPerLiter
axiom study_time_performance_pos : isPositivelyCorrelated avgStudyTime avgAcademicPerformance
axiom smoking_health_neg : ¬ isPositivelyCorrelated dailySmokingAmount healthCondition
axiom side_area_func : isFunctionallyRelated sideLength areaOfSquare
axiom car_weight_fuel_pos : isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm

-- The proof statement to prove C is the correct answer
theorem positively_correlated_variables:
  isPositivelyCorrelated avgStudyTime avgAcademicPerformance ∧
  isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm :=
by
  sorry

end NUMINAMATH_GPT_positively_correlated_variables_l2013_201352


namespace NUMINAMATH_GPT_bill_apples_left_l2013_201300

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end NUMINAMATH_GPT_bill_apples_left_l2013_201300


namespace NUMINAMATH_GPT_part1_part2_l2013_201341

open Set

variable {α : Type*} [PartialOrder α]

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem part1 : A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

theorem part2 : (compl B) = {x | x ≤ 1 ∨ x ≥ 3} :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2013_201341


namespace NUMINAMATH_GPT_center_of_circle_param_eq_l2013_201375

theorem center_of_circle_param_eq (θ : ℝ) : 
  (∃ c : ℝ × ℝ, ∀ θ, 
    ∃ (x y : ℝ), 
      (x = 2 + 2 * Real.cos θ) ∧ 
      (y = 2 * Real.sin θ) ∧ 
      (x - c.1)^2 + y^2 = 4) 
  ↔ 
  c = (2, 0) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_param_eq_l2013_201375


namespace NUMINAMATH_GPT_event_eq_conds_l2013_201325

-- Definitions based on conditions
def Die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }
def sum_points (d1 d2 : Die) : ℕ := d1.val + d2.val

def event_xi_eq_4 (d1 d2 : Die) : Prop := 
  sum_points d1 d2 = 4

def condition_a (d1 d2 : Die) : Prop := 
  d1.val = 2 ∧ d2.val = 2

def condition_b (d1 d2 : Die) : Prop := 
  (d1.val = 3 ∧ d2.val = 1) ∨ (d1.val = 1 ∧ d2.val = 3)

def event_condition (d1 d2 : Die) : Prop :=
  condition_a d1 d2 ∨ condition_b d1 d2

-- The main Lean statement
theorem event_eq_conds (d1 d2 : Die) : 
  event_xi_eq_4 d1 d2 ↔ event_condition d1 d2 := 
by
  sorry

end NUMINAMATH_GPT_event_eq_conds_l2013_201325


namespace NUMINAMATH_GPT_total_number_of_employees_l2013_201340

theorem total_number_of_employees (n : ℕ) (hm : ℕ) (hd : ℕ) 
  (h_ratio : 4 * hd = hm)
  (h_diff : hm = hd + 72) : n = 120 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_total_number_of_employees_l2013_201340


namespace NUMINAMATH_GPT_forty_percent_of_n_l2013_201329

theorem forty_percent_of_n (N : ℝ) (h : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10) : 0.40 * N = 120 := by
  sorry

end NUMINAMATH_GPT_forty_percent_of_n_l2013_201329


namespace NUMINAMATH_GPT_fraction_in_range_l2013_201323

theorem fraction_in_range : 
  (2:ℝ) / 5 < (4:ℝ) / 7 ∧ (4:ℝ) / 7 < 3 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_in_range_l2013_201323


namespace NUMINAMATH_GPT_boats_solution_l2013_201310

theorem boats_solution (x y : ℕ) (h1 : x + y = 42) (h2 : 6 * x = 8 * y) : x = 24 ∧ y = 18 :=
by
  sorry

end NUMINAMATH_GPT_boats_solution_l2013_201310


namespace NUMINAMATH_GPT_school_starts_at_8_l2013_201388

def minutes_to_time (minutes : ℕ) : ℕ × ℕ :=
  let hour := minutes / 60
  let minute := minutes % 60
  (hour, minute)

def add_minutes_to_time (h : ℕ) (m : ℕ) (added_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) + added_minutes)

def subtract_minutes_from_time (h : ℕ) (m : ℕ) (subtracted_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) - subtracted_minutes)

theorem school_starts_at_8 : True := by
  let normal_commute := 30
  let red_light_stops := 3 * 4
  let construction_delay := 10
  let total_additional_time := red_light_stops + construction_delay
  let total_commute_time := normal_commute + total_additional_time
  let depart_time := (7, 15)
  let arrival_time := add_minutes_to_time depart_time.1 depart_time.2 total_commute_time
  let start_time := subtract_minutes_from_time arrival_time.1 arrival_time.2 7

  have : start_time = (8, 0) := by
    sorry

  exact trivial

end NUMINAMATH_GPT_school_starts_at_8_l2013_201388


namespace NUMINAMATH_GPT_linda_five_dollar_bills_l2013_201390

theorem linda_five_dollar_bills :
  ∃ (x y : ℕ), x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_linda_five_dollar_bills_l2013_201390


namespace NUMINAMATH_GPT_probability_of_drawing_2_black_and_2_white_l2013_201351

def total_balls : ℕ := 17
def black_balls : ℕ := 9
def white_balls : ℕ := 8
def balls_drawn : ℕ := 4
def favorable_outcomes := (Nat.choose 9 2) * (Nat.choose 8 2)
def total_outcomes := Nat.choose 17 4
def probability_draw : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_drawing_2_black_and_2_white :
  probability_draw = 168 / 397 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_2_black_and_2_white_l2013_201351
