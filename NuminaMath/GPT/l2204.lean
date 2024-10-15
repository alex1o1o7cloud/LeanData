import Mathlib

namespace NUMINAMATH_GPT_number_of_combinations_of_planets_is_1141_l2204_220496

def number_of_combinations_of_planets : ℕ :=
  (if 7 ≥ 7 ∧ 8 ≥2 then Nat.choose 7 7 * Nat.choose 8 2 else 0) + 
  (if 7 ≥ 6 ∧ 8 ≥ 4 then Nat.choose 7 6 * Nat.choose 8 4 else 0) + 
  (if 7 ≥ 5 ∧ 8 ≥ 6 then Nat.choose 7 5 * Nat.choose 8 6 else 0) +
  (if 7 ≥ 4 ∧ 8 ≥ 8 then Nat.choose 7 4 * Nat.choose 8 8 else 0)

theorem number_of_combinations_of_planets_is_1141 :
  number_of_combinations_of_planets = 1141 :=
by
  sorry

end NUMINAMATH_GPT_number_of_combinations_of_planets_is_1141_l2204_220496


namespace NUMINAMATH_GPT_problem_III_l2204_220414

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem problem_III
  (a x1 x2 : ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_roots : f x1 = a ∧ f x2 = a)
  (h_order : x1 < x2)
  (h_bounds : Real.exp (-1) < x1 ∧ x1 < 1 ∧ 1 < x2) :
  x2 - x1 > 1 / a - 1 :=
sorry

end NUMINAMATH_GPT_problem_III_l2204_220414


namespace NUMINAMATH_GPT_percentage_neither_language_l2204_220498

noncomputable def total_diplomats : ℝ := 120
noncomputable def latin_speakers : ℝ := 20
noncomputable def russian_non_speakers : ℝ := 32
noncomputable def both_languages : ℝ := 0.10 * total_diplomats

theorem percentage_neither_language :
  let D := total_diplomats
  let L := latin_speakers
  let R := D - russian_non_speakers
  let LR := both_languages
  ∃ P, P = 100 * (D - (L + R - LR)) / D :=
by
  existsi ((total_diplomats - (latin_speakers + (total_diplomats - russian_non_speakers) - both_languages)) / total_diplomats * 100)
  sorry

end NUMINAMATH_GPT_percentage_neither_language_l2204_220498


namespace NUMINAMATH_GPT_lindas_savings_l2204_220471

theorem lindas_savings :
  ∃ S : ℝ, (3 / 4 * S) + 150 = S ∧ (S - 150) = 3 / 4 * S := 
sorry

end NUMINAMATH_GPT_lindas_savings_l2204_220471


namespace NUMINAMATH_GPT_gcd_65_130_l2204_220431

theorem gcd_65_130 : Int.gcd 65 130 = 65 := by
  sorry

end NUMINAMATH_GPT_gcd_65_130_l2204_220431


namespace NUMINAMATH_GPT_intersection_lg_1_x_squared_zero_t_le_one_l2204_220422

theorem intersection_lg_1_x_squared_zero_t_le_one  :
  let M := {x | 0 ≤ x ∧ x ≤ 2}
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_lg_1_x_squared_zero_t_le_one_l2204_220422


namespace NUMINAMATH_GPT_list_of_21_numbers_l2204_220499

theorem list_of_21_numbers (numbers : List ℝ) (n : ℝ) (h_length : numbers.length = 21) 
  (h_mem : n ∈ numbers) 
  (h_n_avg : n = 4 * (numbers.sum - n) / 20) 
  (h_n_sum : n = (numbers.sum) / 6) : numbers.length - 1 = 20 :=
by
  -- We provide the statement with the correct hypotheses
  -- the proof is yet to be filled in
  sorry

end NUMINAMATH_GPT_list_of_21_numbers_l2204_220499


namespace NUMINAMATH_GPT_probability_at_least_one_correct_l2204_220405

-- Define the probability of missing a single question
def prob_miss_one : ℚ := 3 / 4

-- Define the probability of missing all six questions
def prob_miss_six : ℚ := prob_miss_one ^ 6

-- Define the probability of getting at least one correct answer
def prob_at_least_one : ℚ := 1 - prob_miss_six

-- The problem statement
theorem probability_at_least_one_correct :
  prob_at_least_one = 3367 / 4096 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_probability_at_least_one_correct_l2204_220405


namespace NUMINAMATH_GPT_volume_to_surface_area_ratio_l2204_220457

-- Definitions based on the conditions
def unit_cube_volume : ℕ := 1
def num_unit_cubes : ℕ := 7
def unit_cube_total_volume : ℕ := num_unit_cubes * unit_cube_volume

def surface_area_of_central_cube : ℕ := 0
def exposed_faces_per_surrounding_cube : ℕ := 5
def num_surrounding_cubes : ℕ := 6
def total_surface_area : ℕ := num_surrounding_cubes * exposed_faces_per_surrounding_cube

-- Mathematical proof statement
theorem volume_to_surface_area_ratio : 
  (unit_cube_total_volume : ℚ) / (total_surface_area : ℚ) = 7 / 30 :=
by sorry

end NUMINAMATH_GPT_volume_to_surface_area_ratio_l2204_220457


namespace NUMINAMATH_GPT_Alfred_spent_on_repairs_l2204_220407

noncomputable def AlfredRepairCost (purchase_price selling_price gain_percent : ℚ) : ℚ :=
  let R := (selling_price - purchase_price * (1 + gain_percent)) / (1 + gain_percent)
  R

theorem Alfred_spent_on_repairs :
  AlfredRepairCost 4700 5800 0.017543859649122806 = 1000 := by
  sorry

end NUMINAMATH_GPT_Alfred_spent_on_repairs_l2204_220407


namespace NUMINAMATH_GPT_center_of_circle_l2204_220438

theorem center_of_circle (x y : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 - 6 * y = 12) → ((x + 2)^2 + (y - 3)^2 = 25) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l2204_220438


namespace NUMINAMATH_GPT_cosine_values_count_l2204_220451

theorem cosine_values_count (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : Real.cos x = -0.65) : 
  ∃ (n : ℕ), n = 2 := by
  sorry

end NUMINAMATH_GPT_cosine_values_count_l2204_220451


namespace NUMINAMATH_GPT_packs_bought_l2204_220470

theorem packs_bought (total_uncommon : ℕ) (cards_per_pack : ℕ) (fraction_uncommon : ℚ) 
  (total_packs : ℕ) (uncommon_per_pack : ℕ)
  (h1 : cards_per_pack = 20)
  (h2 : fraction_uncommon = 1/4)
  (h3 : uncommon_per_pack = fraction_uncommon * cards_per_pack)
  (h4 : total_uncommon = 50)
  (h5 : total_packs = total_uncommon / uncommon_per_pack)
  : total_packs = 10 :=
by 
  sorry

end NUMINAMATH_GPT_packs_bought_l2204_220470


namespace NUMINAMATH_GPT_width_of_property_l2204_220493

theorem width_of_property (W : ℝ) 
  (h1 : ∃ w l, (w = W / 8) ∧ (l = 2250 / 10) ∧ (w * l = 28125)) : W = 1000 :=
by
  -- Formal proof here
  sorry

end NUMINAMATH_GPT_width_of_property_l2204_220493


namespace NUMINAMATH_GPT_not_possible_arrangement_l2204_220412

theorem not_possible_arrangement : 
  ¬ ∃ (f : Fin 4026 → Fin 2014), 
    (∀ k : Fin 2014, ∃ i j : Fin 4026, i < j ∧ f i = k ∧ f j = k ∧ (j.val - i.val - 1) = k.val) :=
sorry

end NUMINAMATH_GPT_not_possible_arrangement_l2204_220412


namespace NUMINAMATH_GPT_smallest_n_identity_matrix_l2204_220488

noncomputable def rotation_45_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_45_matrix ^ n = 1) ∧ ∀ m : ℕ, m > 0 → (rotation_45_matrix ^ m = 1 → n ≤ m) := sorry

end NUMINAMATH_GPT_smallest_n_identity_matrix_l2204_220488


namespace NUMINAMATH_GPT_complement_intersection_l2204_220449

-- Definitions
def A : Set ℝ := { x | x^2 + x - 6 < 0 }
def B : Set ℝ := { x | x > 1 }

-- Stating the problem
theorem complement_intersection (x : ℝ) : x ∈ (Aᶜ ∩ B) ↔ x ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l2204_220449


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l2204_220408

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, (0 < n) ∧ (∃ k m : ℕ, (n + 70 = 2 ^ k * 5 ^ m) ∧ k = 0 ∨ k = 1) ∧ n = 55 :=
by sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l2204_220408


namespace NUMINAMATH_GPT_profit_equations_l2204_220402

-- Define the conditions
def total_workers : ℕ := 150
def fabric_per_worker_per_day : ℕ := 30
def clothing_per_worker_per_day : ℕ := 4
def fabric_needed_per_clothing : ℝ := 1.5
def profit_per_meter : ℝ := 2
def profit_per_clothing : ℝ := 25

-- Define the profit functions
def profit_clothing (x : ℕ) : ℝ := profit_per_clothing * clothing_per_worker_per_day * x
def profit_fabric (x : ℕ) : ℝ := profit_per_meter * (fabric_per_worker_per_day * (total_workers - x) - fabric_needed_per_clothing * clothing_per_worker_per_day * x)

-- Define the total profit function
def total_profit (x : ℕ) : ℝ := profit_clothing x + profit_fabric x

-- Prove the given statements
theorem profit_equations (x : ℕ) :
  profit_clothing x = 100 * x ∧
  profit_fabric x = 9000 - 72 * x ∧
  total_profit 100 = 11800 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_profit_equations_l2204_220402


namespace NUMINAMATH_GPT_neg_of_all_men_are_honest_l2204_220409

variable {α : Type} (man honest : α → Prop)

theorem neg_of_all_men_are_honest :
  ¬ (∀ x, man x → honest x) ↔ ∃ x, man x ∧ ¬ honest x :=
by
  sorry

end NUMINAMATH_GPT_neg_of_all_men_are_honest_l2204_220409


namespace NUMINAMATH_GPT_boat_distance_downstream_l2204_220475

theorem boat_distance_downstream 
    (boat_speed_still : ℝ) 
    (stream_speed : ℝ) 
    (time_downstream : ℝ) 
    (distance_downstream : ℝ) 
    (h_boat_speed_still : boat_speed_still = 13) 
    (h_stream_speed : stream_speed = 6) 
    (h_time_downstream : time_downstream = 3.6315789473684212) 
    (h_distance_downstream : distance_downstream = 19 * 3.6315789473684212): 
    distance_downstream = 69 := 
by 
  have h_effective_speed : boat_speed_still + stream_speed = 19 := by 
    rw [h_boat_speed_still, h_stream_speed]; norm_num 
  rw [h_distance_downstream]; norm_num 
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_l2204_220475


namespace NUMINAMATH_GPT_power_of_prime_implies_n_prime_l2204_220432

theorem power_of_prime_implies_n_prime (n : ℕ) (p : ℕ) (k : ℕ) (hp : Nat.Prime p) :
  3^n - 2^n = p^k → Nat.Prime n :=
by
  sorry

end NUMINAMATH_GPT_power_of_prime_implies_n_prime_l2204_220432


namespace NUMINAMATH_GPT_smallest_rational_number_l2204_220460

theorem smallest_rational_number : ∀ (a b c d : ℚ), (a = -3) → (b = -1) → (c = 0) → (d = 1) → (a < b ∧ a < c ∧ a < d) :=
by
  intros a b c d h₁ h₂ h₃ h₄
  have h₅ : a = -3 := h₁
  have h₆ : b = -1 := h₂
  have h₇ : c = 0 := h₃
  have h₈ : d = 1 := h₄
  sorry

end NUMINAMATH_GPT_smallest_rational_number_l2204_220460


namespace NUMINAMATH_GPT_operation_X_value_l2204_220492

def operation_X (a b : ℤ) : ℤ := b + 7 * a - a^3 + 2 * b

theorem operation_X_value : operation_X 4 3 = -27 := by
  sorry

end NUMINAMATH_GPT_operation_X_value_l2204_220492


namespace NUMINAMATH_GPT_sum_of_numbers_l2204_220441

variable (x y : ℝ)

def condition1 := 0.45 * x = 2700
def condition2 := y = 2 * x

theorem sum_of_numbers (h1 : condition1 x) (h2 : condition2 x y) : x + y = 18000 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_numbers_l2204_220441


namespace NUMINAMATH_GPT_theta_plus_2phi_l2204_220424

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hφ : 0 < φ ∧ φ < π / 2)
  (h_tan_θ : Real.tan θ = 1 / 7) (h_sin_φ : Real.sin φ = 1 / Real.sqrt 10) :
  θ + 2 * φ = π / 4 := 
sorry

end NUMINAMATH_GPT_theta_plus_2phi_l2204_220424


namespace NUMINAMATH_GPT_eval_expression_l2204_220464

theorem eval_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l2204_220464


namespace NUMINAMATH_GPT_expand_expression_l2204_220443

theorem expand_expression (x y : ℝ) : 
  5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := 
by 
  sorry

end NUMINAMATH_GPT_expand_expression_l2204_220443


namespace NUMINAMATH_GPT_geometric_series_sum_l2204_220491

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2204_220491


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l2204_220427

theorem arithmetic_seq_sum (a_n : ℕ → ℝ) (h_arith_seq : ∃ d, ∀ n, a_n (n + 1) = a_n n + d)
    (h_sum : a_n 5 + a_n 8 = 24) : a_n 6 + a_n 7 = 24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l2204_220427


namespace NUMINAMATH_GPT_lowest_score_jack_l2204_220406

noncomputable def lowest_possible_score (mean : ℝ) (std_dev : ℝ) := 
  max ((1.28 * std_dev) + mean) (mean + 2 * std_dev)

theorem lowest_score_jack (mean : ℝ := 60) (std_dev : ℝ := 10) :
  lowest_possible_score mean std_dev = 73 := 
by
  -- We need to show that the minimum score Jack could get is 73 based on problem conditions
  sorry

end NUMINAMATH_GPT_lowest_score_jack_l2204_220406


namespace NUMINAMATH_GPT_ABC_three_digit_number_l2204_220486

theorem ABC_three_digit_number : 
    ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    3 * C % 10 = 8 ∧ 
    3 * B + 1 % 10 = 8 ∧ 
    3 * A + 2 = 8 ∧ 
    100 * A + 10 * B + C = 296 := 
by
  sorry

end NUMINAMATH_GPT_ABC_three_digit_number_l2204_220486


namespace NUMINAMATH_GPT_kaylee_more_boxes_to_sell_l2204_220480

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end NUMINAMATH_GPT_kaylee_more_boxes_to_sell_l2204_220480


namespace NUMINAMATH_GPT_car_distance_in_45_minutes_l2204_220497

theorem car_distance_in_45_minutes
  (train_speed : ℝ)
  (car_speed_ratio : ℝ)
  (time_minutes : ℝ)
  (h_train_speed : train_speed = 90)
  (h_car_speed_ratio : car_speed_ratio = 5 / 6)
  (h_time_minutes : time_minutes = 45) :
  ∃ d : ℝ, d = 56.25 ∧ d = (car_speed_ratio * train_speed) * (time_minutes / 60) :=
by
  sorry

end NUMINAMATH_GPT_car_distance_in_45_minutes_l2204_220497


namespace NUMINAMATH_GPT_number_of_teachers_in_school_l2204_220416

-- Definitions based on provided conditions
def number_of_girls : ℕ := 315
def number_of_boys : ℕ := 309
def total_number_of_people : ℕ := 1396

-- Proof goal: Number of teachers in the school
theorem number_of_teachers_in_school : 
  total_number_of_people - (number_of_girls + number_of_boys) = 772 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teachers_in_school_l2204_220416


namespace NUMINAMATH_GPT_beaver_hid_90_carrots_l2204_220420

-- Defining the number of burrows and carrot condition homomorphic to the problem
def beaver_carrots (x : ℕ) := 5 * x
def rabbit_carrots (y : ℕ) := 7 * y

-- Stating the main theorem based on conditions derived from the problem
theorem beaver_hid_90_carrots (x y : ℕ) (h1 : beaver_carrots x = rabbit_carrots y) (h2 : y = x - 5) : 
  beaver_carrots x = 90 := 
by 
  sorry

end NUMINAMATH_GPT_beaver_hid_90_carrots_l2204_220420


namespace NUMINAMATH_GPT_no_even_and_increasing_function_l2204_220439

-- Definition of a function being even
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Definition of a function being increasing
def is_increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem stating the non-existence of a function that is both even and increasing
theorem no_even_and_increasing_function : ¬ ∃ f : ℝ → ℝ, is_even_function f ∧ is_increasing_function f :=
by
  sorry

end NUMINAMATH_GPT_no_even_and_increasing_function_l2204_220439


namespace NUMINAMATH_GPT_value_of_a_sum_l2204_220489

theorem value_of_a_sum (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^7 = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 128 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_sum_l2204_220489


namespace NUMINAMATH_GPT_time_to_cross_bridge_l2204_220440

theorem time_to_cross_bridge 
  (speed_kmhr : ℕ) 
  (bridge_length_m : ℕ) 
  (h1 : speed_kmhr = 10)
  (h2 : bridge_length_m = 2500) :
  (bridge_length_m / (speed_kmhr * 1000 / 60) = 15) :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l2204_220440


namespace NUMINAMATH_GPT_ben_daily_spending_l2204_220478

variable (S : ℕ)

def daily_savings (S : ℕ) : ℕ := 50 - S

def total_savings (S : ℕ) : ℕ := 7 * daily_savings S

def final_amount (S : ℕ) : ℕ := 2 * total_savings S + 10

theorem ben_daily_spending :
  final_amount 15 = 500 :=
by
  unfold final_amount
  unfold total_savings
  unfold daily_savings
  sorry

end NUMINAMATH_GPT_ben_daily_spending_l2204_220478


namespace NUMINAMATH_GPT_no_integer_solutions_l2204_220425

theorem no_integer_solutions (x y : ℤ) : x^3 + 3 ≠ 4 * y * (y + 1) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l2204_220425


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_eq_l2204_220448

theorem perfect_square_trinomial_m_eq (
    m y : ℝ) (h : ∃ k : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) :
  m = 20 ∨ m = -20 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_eq_l2204_220448


namespace NUMINAMATH_GPT_right_side_longer_l2204_220404

/-- The sum of the three sides of a triangle is 50. 
    The right side of the triangle is a certain length longer than the left side, which has a value of 12 cm. 
    The triangle base has a value of 24 cm. 
    Prove that the right side is 2 cm longer than the left side. -/
theorem right_side_longer (L R B : ℝ) (hL : L = 12) (hB : B = 24) (hSum : L + B + R = 50) : R = L + 2 :=
by
  sorry

end NUMINAMATH_GPT_right_side_longer_l2204_220404


namespace NUMINAMATH_GPT_price_of_each_shirt_l2204_220410

-- Defining the conditions
def total_pants_cost (pants_price : ℕ) (num_pants : ℕ) := num_pants * pants_price
def total_amount_spent (amount_given : ℕ) (change_received : ℕ) := amount_given - change_received
def total_shirts_cost (amount_spent : ℕ) (pants_cost : ℕ) := amount_spent - pants_cost
def price_per_shirt (shirts_total_cost : ℕ) (num_shirts : ℕ) := shirts_total_cost / num_shirts

-- The main statement
theorem price_of_each_shirt (pants_price num_pants amount_given change_received num_shirts : ℕ) :
  num_pants = 2 →
  pants_price = 54 →
  amount_given = 250 →
  change_received = 10 →
  num_shirts = 4 →
  price_per_shirt (total_shirts_cost (total_amount_spent amount_given change_received) 
                   (total_pants_cost pants_price num_pants)) num_shirts = 33
:= by
  sorry

end NUMINAMATH_GPT_price_of_each_shirt_l2204_220410


namespace NUMINAMATH_GPT_rebus_solution_l2204_220433

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rebus_solution_l2204_220433


namespace NUMINAMATH_GPT_option_D_correct_l2204_220458

theorem option_D_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 :=
by sorry

end NUMINAMATH_GPT_option_D_correct_l2204_220458


namespace NUMINAMATH_GPT_vector_addition_proof_l2204_220467

def u : ℝ × ℝ × ℝ := (-3, 2, 5)
def v : ℝ × ℝ × ℝ := (4, -7, 1)
def result : ℝ × ℝ × ℝ := (-2, -3, 11)

theorem vector_addition_proof : (2 • u + v) = result := by
  sorry

end NUMINAMATH_GPT_vector_addition_proof_l2204_220467


namespace NUMINAMATH_GPT_find_k_l2204_220455

-- Assume three lines in the form of equations
def line1 (x y k : ℝ) := x + k * y = 0
def line2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
def line3 (x y : ℝ) := x - y - 1 = 0

-- Assume the intersection point exists
def intersection_point (x y : ℝ) := 
  line2 x y ∧ line3 x y

-- The main theorem statement
theorem find_k (k : ℝ) (x y : ℝ) (h : intersection_point x y) : 
  line1 x y k ↔ k = -1/2 := 
sorry

end NUMINAMATH_GPT_find_k_l2204_220455


namespace NUMINAMATH_GPT_megans_candy_l2204_220444

variable (M : ℕ)

theorem megans_candy (h1 : M * 3 + 10 = 25) : M = 5 :=
by sorry

end NUMINAMATH_GPT_megans_candy_l2204_220444


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2204_220490

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2204_220490


namespace NUMINAMATH_GPT_driver_speed_l2204_220434

theorem driver_speed (v t : ℝ) (h1 : t > 0) (h2 : v > 0) (h3 : v * t = (v + 37.5) * (3 / 8) * t) : v = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_driver_speed_l2204_220434


namespace NUMINAMATH_GPT_quadrilateral_correct_choice_l2204_220473

/-- Define the triangle inequality theorem for four line segments.
    A quadrilateral can be formed if for any:
    - The sum of the lengths of any three segments is greater than the length of the fourth segment.
-/
def is_quadrilateral (a b c d : ℕ) : Prop :=
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a)

/-- Determine which set of three line segments can form a quadrilateral with a fourth line segment of length 5.
    We prove that the correct choice is the set (3, 3, 3). --/
theorem quadrilateral_correct_choice :
  is_quadrilateral 3 3 3 5 ∧  ¬ is_quadrilateral 1 1 1 5 ∧  ¬ is_quadrilateral 1 1 8 5 ∧  ¬ is_quadrilateral 1 2 2 5 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_correct_choice_l2204_220473


namespace NUMINAMATH_GPT_garden_dimensions_l2204_220476

theorem garden_dimensions (l w : ℕ) (h1 : 2 * l + 2 * w = 60) (h2 : l * w = 221) : 
    (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) :=
sorry

end NUMINAMATH_GPT_garden_dimensions_l2204_220476


namespace NUMINAMATH_GPT_solve_inequality_l2204_220411

open Real

theorem solve_inequality (x : ℝ) : (x ≠ 3) ∧ (x * (x + 1) / (x - 3) ^ 2 ≥ 9) ↔ (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2204_220411


namespace NUMINAMATH_GPT_sin_13pi_over_6_equals_half_l2204_220468

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_13pi_over_6_equals_half_l2204_220468


namespace NUMINAMATH_GPT_best_sampling_method_l2204_220430

/-- 
  Given a high school that wants to understand the psychological 
  pressure of students from three different grades, prove that 
  stratified sampling is the best method to use, assuming students
  from different grades may experience different levels of psychological
  pressure.
-/
theorem best_sampling_method
  (students_from_three_grades : Type)
  (survey_psychological_pressure : students_from_three_grades → ℝ)
  (potential_differences_by_grade : students_from_three_grades → ℝ → Prop):
  ∃ sampling_method, sampling_method = "stratified_sampling" :=
sorry

end NUMINAMATH_GPT_best_sampling_method_l2204_220430


namespace NUMINAMATH_GPT_no_partition_of_positive_integers_l2204_220435

theorem no_partition_of_positive_integers :
  ∀ (A B C : Set ℕ), (∀ (x : ℕ), x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (∀ (x y : ℕ), x ∈ A ∧ y ∈ B → x^2 - x * y + y^2 ∈ C) →
  (∀ (x y : ℕ), x ∈ B ∧ y ∈ C → x^2 - x * y + y^2 ∈ A) →
  (∀ (x y : ℕ), x ∈ C ∧ y ∈ A → x^2 - x * y + y^2 ∈ B) →
  False := 
sorry

end NUMINAMATH_GPT_no_partition_of_positive_integers_l2204_220435


namespace NUMINAMATH_GPT_no_point_in_punctured_disk_l2204_220461

theorem no_point_in_punctured_disk (A B C D E F G : ℝ) (hB2_4AC : B^2 - 4 * A * C < 0) :
  ∃ δ > 0, ∀ x y : ℝ, 0 < x^2 + y^2 → x^2 + y^2 < δ^2 → 
    ¬(A * x^2 + B * x * y + C * y^2 + D * x^3 + E * x^2 * y + F * x * y^2 + G * y^3 = 0) :=
sorry

end NUMINAMATH_GPT_no_point_in_punctured_disk_l2204_220461


namespace NUMINAMATH_GPT_solidConstruction_l2204_220428

-- Definitions
structure Solid where
  octagonal_faces : Nat
  hexagonal_faces : Nat
  square_faces : Nat

-- Conditions
def solidFromCube (S : Solid) : Prop :=
  S.octagonal_faces = 6 ∧ S.hexagonal_faces = 8 ∧ S.square_faces = 12

def circumscribedSphere (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

def solidFromOctahedron (S : Solid) : Prop :=
  true  -- Placeholder, actual geometric definition needed

-- Theorem statement
theorem solidConstruction {S : Solid} :
  solidFromCube S ∧ circumscribedSphere S → solidFromOctahedron S :=
by
  sorry

end NUMINAMATH_GPT_solidConstruction_l2204_220428


namespace NUMINAMATH_GPT_general_term_formula_l2204_220400

-- Define the problem parameters
variables (a : ℤ)

-- Definitions based on the conditions
def first_term : ℤ := a - 1
def second_term : ℤ := a + 1
def third_term : ℤ := 2 * a + 3

-- Define the theorem to prove the general term formula
theorem general_term_formula :
  2 * (first_term a + 1) = first_term a + third_term a → a = 0 →
  ∀ n : ℕ, a_n = 2 * n - 3 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_general_term_formula_l2204_220400


namespace NUMINAMATH_GPT_distance_between_first_and_last_stop_in_km_l2204_220423

-- Define the total number of stops
def num_stops := 12

-- Define the distance between the third and sixth stops in meters
def dist_3_to_6 := 3300

-- The distance between consecutive stops is the same
def distance_between_first_and_last_stop : ℕ := (num_stops - 1) * (dist_3_to_6 / 3)

-- The distance in kilometers (1 kilometer = 1000 meters)
noncomputable def distance_km : ℝ := distance_between_first_and_last_stop / 1000

-- Statement to prove
theorem distance_between_first_and_last_stop_in_km : distance_km = 12.1 :=
by
  -- Theorem proof should go here
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_stop_in_km_l2204_220423


namespace NUMINAMATH_GPT_production_difference_l2204_220494

variables (p h : ℕ)

def first_day_production := p * h

def second_day_production := (p + 5) * (h - 3)

-- Given condition
axiom p_eq_3h : p = 3 * h

theorem production_difference : first_day_production p h - second_day_production p h = 4 * h + 15 :=
by
  sorry

end NUMINAMATH_GPT_production_difference_l2204_220494


namespace NUMINAMATH_GPT_units_digit_of_product_of_first_three_positive_composite_numbers_l2204_220469

theorem units_digit_of_product_of_first_three_positive_composite_numbers :
  (4 * 6 * 8) % 10 = 2 :=
by sorry

end NUMINAMATH_GPT_units_digit_of_product_of_first_three_positive_composite_numbers_l2204_220469


namespace NUMINAMATH_GPT_possible_values_of_g_zero_l2204_220465

variable {g : ℝ → ℝ}

theorem possible_values_of_g_zero (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) : g 0 = 0 ∨ g 0 = 1 := 
sorry

end NUMINAMATH_GPT_possible_values_of_g_zero_l2204_220465


namespace NUMINAMATH_GPT_eval_g_at_2_l2204_220454

def g (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem eval_g_at_2 : g 2 = 3 :=
by {
  -- This is the place for proof steps, currently it is filled with sorry.
  sorry
}

end NUMINAMATH_GPT_eval_g_at_2_l2204_220454


namespace NUMINAMATH_GPT_parabola_tangency_point_l2204_220403

-- Definitions of the parabola equations
def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 20
def parabola2 (y : ℝ) : ℝ := y^2 + 36 * y + 380

-- The proof statement
theorem parabola_tangency_point : 
  ∃ (x y : ℝ), 
    parabola1 x = y ∧ parabola2 y = x ∧ x = -9 / 2 ∧ y = -35 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangency_point_l2204_220403


namespace NUMINAMATH_GPT_tan_theta_eq_neg_4_over_3_expression_eval_l2204_220450

theorem tan_theta_eq_neg_4_over_3 (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  Real.tan θ = -4 / 3 :=
sorry

theorem expression_eval (θ : ℝ) (h₁ : Real.sin θ = 4 / 5) (h₂ : Real.pi / 2 < θ ∧ θ < Real.pi) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (3 * Real.sin θ ^ 2 + Real.cos θ ^ 2) = 8 / 25 :=
sorry

end NUMINAMATH_GPT_tan_theta_eq_neg_4_over_3_expression_eval_l2204_220450


namespace NUMINAMATH_GPT_number_of_monkeys_l2204_220485

theorem number_of_monkeys (N : ℕ)
  (h1 : N * 1 * 8 = 8)
  (h2 : 3 * 1 * 8 = 3 * 8) :
  N = 8 :=
sorry

end NUMINAMATH_GPT_number_of_monkeys_l2204_220485


namespace NUMINAMATH_GPT_distance_to_x_axis_l2204_220453

theorem distance_to_x_axis (x y : ℝ) (h : (x, y) = (3, -4)) : abs y = 4 := sorry

end NUMINAMATH_GPT_distance_to_x_axis_l2204_220453


namespace NUMINAMATH_GPT_number_of_distinct_possible_values_for_c_l2204_220459

variables {a b r s t : ℂ}
variables (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variables (h_transform : ∀ z, (a * z + b - r) * (a * z + b - s) * (a * z + b - t) = (z - c * r) * (z - c * s) * (z - c * t))

theorem number_of_distinct_possible_values_for_c (h_nonzero : a ≠ 0) : 
  ∃ (n : ℕ), n = 4 := sorry

end NUMINAMATH_GPT_number_of_distinct_possible_values_for_c_l2204_220459


namespace NUMINAMATH_GPT_min_pie_pieces_l2204_220466

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end NUMINAMATH_GPT_min_pie_pieces_l2204_220466


namespace NUMINAMATH_GPT_quadratic_transform_l2204_220481

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_transform_l2204_220481


namespace NUMINAMATH_GPT_correct_average_l2204_220474

theorem correct_average (S' : ℝ) (a a' b b' c c' : ℝ) (n : ℕ) 
  (incorrect_avg : S' / n = 22) 
  (a_eq : a = 52) (a'_eq : a' = 32)
  (b_eq : b = 47) (b'_eq : b' = 27) 
  (c_eq : c = 68) (c'_eq : c' = 45)
  (n_eq : n = 12) 
  : ((S' - (a' + b' + c') + (a + b + c)) / 12 = 27.25) := 
by
  sorry

end NUMINAMATH_GPT_correct_average_l2204_220474


namespace NUMINAMATH_GPT_inequality_holds_for_all_m_l2204_220483

theorem inequality_holds_for_all_m (m : ℝ) (h1 : ∀ (x : ℝ), x^2 - 8 * x + 20 > 0)
  (h2 : m < -1/2) : ∀ (x : ℝ), (x ^ 2 - 8 * x + 20) / (m * x ^ 2 + 2 * (m + 1) * x + 9 * m + 4) < 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_m_l2204_220483


namespace NUMINAMATH_GPT_days_at_sister_l2204_220442

def total_days_vacation : ℕ := 21
def days_plane : ℕ := 2
def days_grandparents : ℕ := 5
def days_train : ℕ := 1
def days_brother : ℕ := 5
def days_car_to_sister : ℕ := 1
def days_bus_to_sister : ℕ := 1
def extra_days_due_to_time_zones : ℕ := 1
def days_bus_back : ℕ := 1
def days_car_back : ℕ := 1

theorem days_at_sister : 
  total_days_vacation - (days_plane + days_grandparents + days_train + days_brother + days_car_to_sister + days_bus_to_sister + extra_days_due_to_time_zones + days_bus_back + days_car_back) = 3 :=
by
  sorry

end NUMINAMATH_GPT_days_at_sister_l2204_220442


namespace NUMINAMATH_GPT_miles_to_drive_l2204_220419

def total_miles : ℕ := 1200
def miles_driven : ℕ := 768
def miles_remaining : ℕ := total_miles - miles_driven

theorem miles_to_drive : miles_remaining = 432 := by
  -- Proof goes here, omitted as per instructions
  sorry

end NUMINAMATH_GPT_miles_to_drive_l2204_220419


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2204_220456

theorem problem1_solution (x : ℝ): 2 * x^2 + x - 3 = 0 → (x = 1 ∨ x = -3 / 2) :=
by
  intro h
  -- Proof skipped
  sorry

theorem problem2_solution (x : ℝ): (x - 3)^2 = 2 * x * (3 - x) → (x = 3 ∨ x = 1) :=
by
  intro h
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2204_220456


namespace NUMINAMATH_GPT_budget_for_equipment_l2204_220445

theorem budget_for_equipment 
    (transportation_p : ℝ := 20)
    (r_d_p : ℝ := 9)
    (utilities_p : ℝ := 5)
    (supplies_p : ℝ := 2)
    (salaries_degrees : ℝ := 216)
    (total_degrees : ℝ := 360)
    (total_budget : ℝ := 100)
    :
    (total_budget - (transportation_p + r_d_p + utilities_p + supplies_p +
    (salaries_degrees / total_degrees * total_budget))) = 4 := 
sorry

end NUMINAMATH_GPT_budget_for_equipment_l2204_220445


namespace NUMINAMATH_GPT_diameter_inscribed_circle_l2204_220484

noncomputable def diameter_of_circle (r : ℝ) : ℝ :=
2 * r

theorem diameter_inscribed_circle (r : ℝ) (h : 8 * r = π * r ^ 2) : diameter_of_circle r = 16 / π := by
  sorry

end NUMINAMATH_GPT_diameter_inscribed_circle_l2204_220484


namespace NUMINAMATH_GPT_jack_has_42_pounds_l2204_220463

noncomputable def jack_pounds (P : ℕ) : Prop :=
  let euros := 11
  let yen := 3000
  let pounds_per_euro := 2
  let yen_per_pound := 100
  let total_yen := 9400
  let pounds_from_euros := euros * pounds_per_euro
  let pounds_from_yen := yen / yen_per_pound
  let total_pounds := P + pounds_from_euros + pounds_from_yen
  total_pounds * yen_per_pound = total_yen

theorem jack_has_42_pounds : jack_pounds 42 :=
  sorry

end NUMINAMATH_GPT_jack_has_42_pounds_l2204_220463


namespace NUMINAMATH_GPT_eggs_left_after_capital_recovered_l2204_220462

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_after_capital_recovered_l2204_220462


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l2204_220452

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l2204_220452


namespace NUMINAMATH_GPT_cost_of_1500_pencils_l2204_220479

theorem cost_of_1500_pencils (cost_per_box : ℕ) (pencils_per_box : ℕ) (num_pencils : ℕ) :
  cost_per_box = 30 → pencils_per_box = 100 → num_pencils = 1500 → 
  (num_pencils * (cost_per_box / pencils_per_box) = 450) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_cost_of_1500_pencils_l2204_220479


namespace NUMINAMATH_GPT_tradesman_gain_l2204_220495

-- Let's define a structure representing the tradesman's buying and selling operation.
structure Trade where
  true_value : ℝ
  defraud_rate : ℝ
  buy_price : ℕ
  sell_price : ℕ

theorem tradesman_gain (T : Trade) (H1 : T.defraud_rate = 0.2) (H2 : T.true_value = 100)
  (H3 : T.buy_price = T.true_value * (1 - T.defraud_rate))
  (H4 : T.sell_price = T.true_value * (1 + T.defraud_rate)) :
  ((T.sell_price - T.buy_price) / T.buy_price) * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_tradesman_gain_l2204_220495


namespace NUMINAMATH_GPT_ten_pow_n_plus_one_divisible_by_eleven_l2204_220437

theorem ten_pow_n_plus_one_divisible_by_eleven (n : ℕ) (h : n % 2 = 1) : 11 ∣ (10 ^ n + 1) :=
sorry

end NUMINAMATH_GPT_ten_pow_n_plus_one_divisible_by_eleven_l2204_220437


namespace NUMINAMATH_GPT_equilateral_triangle_percentage_l2204_220418

theorem equilateral_triangle_percentage (s : Real) :
  let area_square := s^2
  let area_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area := area_square + area_triangle
  area_triangle / total_area * 100 = (4 * Real.sqrt 3 - 3) / 13 * 100 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_percentage_l2204_220418


namespace NUMINAMATH_GPT_minimum_value_2sqrt5_l2204_220417

theorem minimum_value_2sqrt5 : ∀ x : ℝ, 
  ∃ m : ℝ, (∀ x : ℝ, m ≤ (x^2 + 10) / (Real.sqrt (x^2 + 5))) ∧ (m = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_minimum_value_2sqrt5_l2204_220417


namespace NUMINAMATH_GPT_fraction_spent_on_candy_l2204_220415

theorem fraction_spent_on_candy (initial_quarters : ℕ) (initial_cents remaining_cents cents_per_dollar : ℕ) (fraction_spent : ℝ) :
  initial_quarters = 14 ∧ remaining_cents = 300 ∧ initial_cents = initial_quarters * 25 ∧ cents_per_dollar = 100 →
  fraction_spent = (initial_cents - remaining_cents) / cents_per_dollar →
  fraction_spent = 1 / 2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_fraction_spent_on_candy_l2204_220415


namespace NUMINAMATH_GPT_find_z_l2204_220421

-- Definitions from conditions
def x : ℕ := 22
def y : ℕ := 13
def total_boys_who_went_down_slide : ℕ := x + y
def ratio_slide_to_watch := 5 / 3

-- Statement we need to prove
theorem find_z : ∃ z : ℕ, (5 / 3 = total_boys_who_went_down_slide / z) ∧ z = 21 :=
by
  use 21
  sorry

end NUMINAMATH_GPT_find_z_l2204_220421


namespace NUMINAMATH_GPT_train_crossing_platform_time_l2204_220426

theorem train_crossing_platform_time (train_length : ℝ) (platform_length : ℝ) (time_cross_post : ℝ) :
  train_length = 300 → platform_length = 350 → time_cross_post = 18 → 
  (train_length + platform_length) / (train_length / time_cross_post) = 39 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_crossing_platform_time_l2204_220426


namespace NUMINAMATH_GPT_trajectory_of_M_l2204_220429

open Real

-- Define the endpoints A and B
variable {A B M : Real × Real}

-- Given conditions
def segment_length (A B : Real × Real) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25

def on_axes (A B : Real × Real) : Prop :=
  A.2 = 0 ∧ B.1 = 0

def point_m_relationship (A B M : Real × Real) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let MB := (M.1 - B.1, M.2 - B.2)
  AM.1 = (2 / 3) * MB.1 ∧ AM.2 = (2 / 3) * MB.2 ∧
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4

theorem trajectory_of_M (A B M : Real × Real)
  (h1 : segment_length A B)
  (h2 : on_axes A B)
  (h3 : point_m_relationship A B M) :
  (M.1^2 / 9) + (M.2^2 / 4) = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_M_l2204_220429


namespace NUMINAMATH_GPT_difference_between_sums_l2204_220413

-- Define the arithmetic sequence sums
def sum_seq (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Define sets A and B
def sumA : ℕ := sum_seq 10 75
def sumB : ℕ := sum_seq 76 125

-- State the problem
theorem difference_between_sums : sumB - sumA = 2220 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_difference_between_sums_l2204_220413


namespace NUMINAMATH_GPT_at_most_one_zero_l2204_220401

-- Definition of the polynomial f(x)
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^4 - 1994 * x^3 + (1993 + n) * x^2 - 11 * x + n

-- The target theorem statement
theorem at_most_one_zero (n : ℤ) : ∃! x : ℝ, f n x = 0 :=
by
  sorry

end NUMINAMATH_GPT_at_most_one_zero_l2204_220401


namespace NUMINAMATH_GPT_binary_multiplication_l2204_220477

theorem binary_multiplication : (10101 : ℕ) * (101 : ℕ) = 1101001 :=
by sorry

end NUMINAMATH_GPT_binary_multiplication_l2204_220477


namespace NUMINAMATH_GPT_midpoint_trajectory_l2204_220447

theorem midpoint_trajectory (x y : ℝ) (h : ∃ (xₚ yₚ : ℝ), yₚ = 2 * xₚ^2 + 1 ∧ y = 4 * (xₚ / 2) ^ 2) : y = 4 * x ^ 2 :=
sorry

end NUMINAMATH_GPT_midpoint_trajectory_l2204_220447


namespace NUMINAMATH_GPT_nuts_to_raisins_ratio_l2204_220446

/-- 
Given that Chris mixed 3 pounds of raisins with 4 pounds of nuts 
and the total cost of the raisins was 0.15789473684210525 of the total cost of the mixture, 
prove that the ratio of the cost of a pound of nuts to the cost of a pound of raisins is 4:1. 
-/
theorem nuts_to_raisins_ratio (R N : ℝ)
    (h1 : 3 * R = 0.15789473684210525 * (3 * R + 4 * N)) :
    N / R = 4 :=
sorry  -- proof skipped

end NUMINAMATH_GPT_nuts_to_raisins_ratio_l2204_220446


namespace NUMINAMATH_GPT_non_neg_integers_l2204_220472

open Nat

theorem non_neg_integers (n : ℕ) :
  (∃ x y k : ℕ, x.gcd y = 1 ∧ k ≥ 2 ∧ 3^n = x^k + y^k) ↔ (n = 0 ∨ n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_GPT_non_neg_integers_l2204_220472


namespace NUMINAMATH_GPT_largest_factor_and_smallest_multiple_of_18_l2204_220487

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ x, (x ∈ {d : ℕ | d ∣ 18}) ∧ (∀ y, y ∈ {d : ℕ | d ∣ 18} → y ≤ x) ∧ x = 18)
  ∧ (∃ y, (y ∈ {m : ℕ | 18 ∣ m}) ∧ (∀ z, z ∈ {m : ℕ | 18 ∣ m} → y ≤ z) ∧ y = 18) :=
by
  sorry

end NUMINAMATH_GPT_largest_factor_and_smallest_multiple_of_18_l2204_220487


namespace NUMINAMATH_GPT_additional_money_needed_l2204_220482

/-- Mrs. Smith needs to calculate the additional money required after a discount -/
theorem additional_money_needed
  (initial_amount : ℝ) (ratio_more : ℝ) (discount_rate : ℝ) (final_amount_needed : ℝ) (additional_needed : ℝ)
  (h_initial : initial_amount = 500)
  (h_ratio : ratio_more = 2/5)
  (h_discount : discount_rate = 15/100)
  (h_total_needed : final_amount_needed = initial_amount * (1 + ratio_more) * (1 - discount_rate))
  (h_additional : additional_needed = final_amount_needed - initial_amount) :
  additional_needed = 95 :=
by 
  sorry

end NUMINAMATH_GPT_additional_money_needed_l2204_220482


namespace NUMINAMATH_GPT_enclosed_area_of_curve_l2204_220436

theorem enclosed_area_of_curve :
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let arc_length := Real.pi
  let arc_angle := Real.pi / 2
  let arc_radius := arc_length / arc_angle
  let sector_area := (arc_angle / (2 * Real.pi)) * Real.pi * arc_radius^2
  let total_sector_area := 12 * sector_area
  let enclosed_area := octagon_area + total_sector_area + 3 * Real.pi
  enclosed_area = 54 + 38.4 * Real.sqrt 2 + 3 * Real.pi :=
by
  -- We will use sorry to indicate the proof is omitted.
  sorry

end NUMINAMATH_GPT_enclosed_area_of_curve_l2204_220436
