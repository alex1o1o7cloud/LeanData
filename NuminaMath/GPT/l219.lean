import Mathlib

namespace NUMINAMATH_GPT_weight_of_replaced_person_l219_21959

-- Define the conditions
variables (W : ℝ) (new_person_weight : ℝ) (avg_weight_increase : ℝ)
#check ℝ

def initial_group_size := 10

-- Define the conditions as hypothesis statements
axiom weight_increase_eq : avg_weight_increase = 3.5
axiom new_person_weight_eq : new_person_weight = 100

-- Define the result to be proved
theorem weight_of_replaced_person (W : ℝ) : 
  ∀ (avg_weight_increase : ℝ) (new_person_weight : ℝ),
    avg_weight_increase = 3.5 ∧ new_person_weight = 100 → 
    (new_person_weight - (avg_weight_increase * initial_group_size)) = 65 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l219_21959


namespace NUMINAMATH_GPT_circumference_of_circle_l219_21943

def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def meeting_time : ℝ := 42
def circumference : ℝ := 630

theorem circumference_of_circle :
  (speed_cyclist1 * meeting_time + speed_cyclist2 * meeting_time = circumference) :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_circle_l219_21943


namespace NUMINAMATH_GPT_watch_sticker_price_l219_21914

theorem watch_sticker_price (x : ℝ)
  (hx_X : 0.80 * x - 50 = y)
  (hx_Y : 0.90 * x = z)
  (savings : z - y = 25) : 
  x = 250 := by
  sorry

end NUMINAMATH_GPT_watch_sticker_price_l219_21914


namespace NUMINAMATH_GPT_art_club_students_l219_21948

theorem art_club_students 
    (students artworks_per_student_per_quarter quarters_per_year artworks_in_two_years : ℕ) 
    (h1 : artworks_per_student_per_quarter = 2)
    (h2 : quarters_per_year = 4) 
    (h3 : artworks_in_two_years = 240) 
    (h4 : students * (artworks_per_student_per_quarter * quarters_per_year) * 2 = artworks_in_two_years) :
    students = 15 := 
by
    -- Given conditions for the problem
    sorry

end NUMINAMATH_GPT_art_club_students_l219_21948


namespace NUMINAMATH_GPT_percentage_problem_l219_21960

theorem percentage_problem 
  (number : ℕ)
  (h1 : number = 6400)
  (h2 : 5 * number / 100 = 20 * 650 / 100 + 190) : 
  20 = 20 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_problem_l219_21960


namespace NUMINAMATH_GPT_Ria_original_savings_l219_21924

variables {R F : ℕ}

def initial_ratio (R F : ℕ) : Prop :=
  R * 3 = F * 5

def withdrawn_amount (R : ℕ) : ℕ :=
  R - 160

def new_ratio (R' F : ℕ) : Prop :=
  R' * 5 = F * 3

theorem Ria_original_savings (initial_ratio: initial_ratio R F)
  (new_ratio: new_ratio (withdrawn_amount R) F) : 
  R = 250 :=
by
  sorry

end NUMINAMATH_GPT_Ria_original_savings_l219_21924


namespace NUMINAMATH_GPT_find_f_l219_21996

noncomputable def func_satisfies_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = x * f x - y * f y

theorem find_f (f : ℝ → ℝ) (h : func_satisfies_eq f) : ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_find_f_l219_21996


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l219_21999

variable (v : ℝ) -- Speed of the bus excluding stoppages

-- Conditions
def bus_stops_per_hour := 45 / 60 -- 45 minutes converted to hours
def effective_driving_time := 1 - bus_stops_per_hour -- Effective time driving in an hour

-- Given Condition
def speed_including_stoppages := 12 -- Speed including stoppages in km/hr

theorem bus_speed_excluding_stoppages 
  (h : effective_driving_time * v = speed_including_stoppages) : 
  v = 48 :=
sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l219_21999


namespace NUMINAMATH_GPT_no_bounded_sequences_at_least_one_gt_20_l219_21983

variable (x y z : ℕ → ℝ)
variable (x1 y1 z1 : ℝ)
variable (h0 : x1 > 0) (h1 : y1 > 0) (h2 : z1 > 0)
variable (h3 : ∀ n, x (n + 1) = y n + (1 / z n))
variable (h4 : ∀ n, y (n + 1) = z n + (1 / x n))
variable (h5 : ∀ n, z (n + 1) = x n + (1 / y n))

-- Part (a)
theorem no_bounded_sequences : (∀ n, x n > 0) ∧ (∀ n, y n > 0) ∧ (∀ n, z n > 0) → ¬ (∃ M, ∀ n, x n < M ∧ y n < M ∧ z n < M) :=
sorry

-- Part (b)
theorem at_least_one_gt_20 : x 1 = x1 ∧ y 1 = y1 ∧ z 1 = z1 → x 200 > 20 ∨ y 200 > 20 ∨ z 200 > 20 :=
sorry

end NUMINAMATH_GPT_no_bounded_sequences_at_least_one_gt_20_l219_21983


namespace NUMINAMATH_GPT_snow_probability_january_first_week_l219_21953

noncomputable def P_snow_at_least_once_first_week : ℚ :=
  1 - ((2 / 3) ^ 4 * (3 / 4) ^ 3)

theorem snow_probability_january_first_week :
  P_snow_at_least_once_first_week = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_snow_probability_january_first_week_l219_21953


namespace NUMINAMATH_GPT_solve_inequality_l219_21995

def f (a x : ℝ) : ℝ := a * x * (x + 1) + 1

theorem solve_inequality (a x : ℝ) (h : f a x < 0) : x < (1 / a) ∨ (x > 1 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l219_21995


namespace NUMINAMATH_GPT_bernoulli_inequality_l219_21965

theorem bernoulli_inequality (n : ℕ) (h : 1 ≤ n) (x : ℝ) (h1 : x > -1) : (1 + x) ^ n ≥ 1 + n * x := 
sorry

end NUMINAMATH_GPT_bernoulli_inequality_l219_21965


namespace NUMINAMATH_GPT_original_ratio_of_boarders_to_day_students_l219_21901

theorem original_ratio_of_boarders_to_day_students
    (original_boarders : ℕ)
    (new_boarders : ℕ)
    (new_ratio_b_d : ℕ → ℕ)
    (no_switch : Prop)
    (no_leave : Prop)
  : (original_boarders = 220) ∧ (new_boarders = 44) ∧ (new_ratio_b_d 1 = 2) ∧ no_switch ∧ no_leave →
  ∃ (original_day_students : ℕ), original_day_students = 528 ∧ (220 / 44 = 5) ∧ (528 / 44 = 12)
  := by
    sorry

end NUMINAMATH_GPT_original_ratio_of_boarders_to_day_students_l219_21901


namespace NUMINAMATH_GPT_minimum_value_of_f_l219_21906

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y, y = f x ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l219_21906


namespace NUMINAMATH_GPT_law_of_sines_l219_21947

theorem law_of_sines (a b c : ℝ) (A B C : ℝ) (R : ℝ) 
  (hA : a = 2 * R * Real.sin A)
  (hEquilateral1 : b = 2 * R * Real.sin B)
  (hEquilateral2 : c = 2 * R * Real.sin C):
  (a / Real.sin A) = (b / Real.sin B) ∧ 
  (b / Real.sin B) = (c / Real.sin C) ∧ 
  (c / Real.sin C) = 2 * R :=
by
  sorry

end NUMINAMATH_GPT_law_of_sines_l219_21947


namespace NUMINAMATH_GPT_sequence_formula_l219_21940

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 3^n) :
  ∀ n : ℕ, a n = (3^n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_formula_l219_21940


namespace NUMINAMATH_GPT_lateral_surface_area_base_area_ratio_correct_l219_21917

noncomputable def lateral_surface_area_to_base_area_ratio
  (S P Q R : Type)
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12)
  : ℝ :=
  π * (4 * Real.sqrt 3 - 3) / 13

theorem lateral_surface_area_base_area_ratio_correct
  {S P Q R : Type}
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12) :
  lateral_surface_area_to_base_area_ratio S P Q R angle_PSR angle_SQR angle_PSQ
    h_PSR h_SQR h_PSQ = π * (4 * Real.sqrt 3 - 3) / 13 :=
  by sorry

end NUMINAMATH_GPT_lateral_surface_area_base_area_ratio_correct_l219_21917


namespace NUMINAMATH_GPT_right_triangle_incircle_excircle_condition_l219_21923

theorem right_triangle_incircle_excircle_condition
  (r R : ℝ) 
  (hr_pos : 0 < r) 
  (hR_pos : 0 < R) :
  R ≥ r * (3 + 2 * Real.sqrt 2) := sorry

end NUMINAMATH_GPT_right_triangle_incircle_excircle_condition_l219_21923


namespace NUMINAMATH_GPT_num_cats_l219_21991

-- Definitions based on conditions
variables (C S K Cap : ℕ)
variable (heads : ℕ) (legs : ℕ)

-- Conditions as equations
axiom heads_eq : C + S + K + Cap = 16
axiom legs_eq : 4 * C + 2 * S + 2 * K + 1 * Cap = 41

-- Given values from the problem
axiom K_val : K = 1
axiom Cap_val : Cap = 1

-- The proof goal in terms of satisfying the number of cats
theorem num_cats : C = 5 :=
by
  sorry

end NUMINAMATH_GPT_num_cats_l219_21991


namespace NUMINAMATH_GPT_apples_given_to_father_l219_21985

theorem apples_given_to_father
  (total_apples : ℤ) 
  (people_sharing : ℤ) 
  (apples_per_person : ℤ)
  (jack_and_friends : ℤ) :
  total_apples = 55 →
  people_sharing = 5 →
  apples_per_person = 9 →
  jack_and_friends = 4 →
  (total_apples - people_sharing * apples_per_person) = 10 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_apples_given_to_father_l219_21985


namespace NUMINAMATH_GPT_integer_solutions_count_l219_21922

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l219_21922


namespace NUMINAMATH_GPT_staff_discount_price_l219_21969

theorem staff_discount_price (d : ℝ) : (d - 0.15*d) * 0.90 = 0.765 * d :=
by
  have discount1 : d - 0.15 * d = d * 0.85 :=
    by ring
  have discount2 : (d * 0.85) * 0.90 = d * (0.85 * 0.90) :=
    by ring
  have final_price : d * (0.85 * 0.90) = d * 0.765 :=
    by norm_num
  rw [discount1, discount2, final_price]
  sorry

end NUMINAMATH_GPT_staff_discount_price_l219_21969


namespace NUMINAMATH_GPT_range_of_a_l219_21997

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 := sorry

end NUMINAMATH_GPT_range_of_a_l219_21997


namespace NUMINAMATH_GPT_minimum_value_of_function_l219_21972

-- Define the function y = 2x + 1/(x - 1) with the constraint x > 1
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- Prove that the minimum value of the function for x > 1 is 2√2 + 2
theorem minimum_value_of_function : 
  ∃ x : ℝ, x > 1 ∧ ∀ y : ℝ, (y = f x) → y ≥ 2 * Real.sqrt 2 + 2 := 
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l219_21972


namespace NUMINAMATH_GPT_sum_prime_factors_77_l219_21925

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end NUMINAMATH_GPT_sum_prime_factors_77_l219_21925


namespace NUMINAMATH_GPT_larger_box_cost_l219_21950

-- Definitions based on the conditions

def ounces_large : ℕ := 30
def ounces_small : ℕ := 20
def cost_small : ℝ := 3.40
def price_per_ounce_better_value : ℝ := 0.16

-- The statement to prove
theorem larger_box_cost :
  30 * price_per_ounce_better_value = 4.80 :=
by sorry

end NUMINAMATH_GPT_larger_box_cost_l219_21950


namespace NUMINAMATH_GPT_pencils_per_student_l219_21955

theorem pencils_per_student (total_pencils : ℤ) (num_students : ℤ) (pencils_per_student : ℤ)
  (h1 : total_pencils = 195)
  (h2 : num_students = 65) :
  total_pencils / num_students = 3 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_student_l219_21955


namespace NUMINAMATH_GPT_sin_double_angle_value_l219_21941

theorem sin_double_angle_value (α : ℝ) (h₁ : Real.sin (π / 4 - α) = 3 / 5) (h₂ : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) = 7 / 25 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_value_l219_21941


namespace NUMINAMATH_GPT_complement_A_B_eq_singleton_three_l219_21931

open Set

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

theorem complement_A_B_eq_singleton_three (hA : A = {2, 3, 4})
    (hB : B = {a + 2, a}) (h_inter : A ∩ B = B) : A \ B = {3} :=
  sorry

end NUMINAMATH_GPT_complement_A_B_eq_singleton_three_l219_21931


namespace NUMINAMATH_GPT_smallest_k_l219_21935

-- Define the set S
def S (m : ℕ) : Finset ℕ :=
  (Finset.range (30 * m)).filter (λ n => n % 2 = 1 ∧ n % 5 ≠ 0)

-- Theorem statement
theorem smallest_k (m : ℕ) (k : ℕ) : 
  (∀ (A : Finset ℕ), A ⊆ S m → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x)) ↔ k ≥ 8 * m + 1 :=
sorry

end NUMINAMATH_GPT_smallest_k_l219_21935


namespace NUMINAMATH_GPT_evaluate_nested_fraction_l219_21986

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (2 - (1 / (3 - (1 / (2 - (1 / 2))))))))) = 11 / 26 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_nested_fraction_l219_21986


namespace NUMINAMATH_GPT_terry_age_proof_l219_21954

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end NUMINAMATH_GPT_terry_age_proof_l219_21954


namespace NUMINAMATH_GPT_shoe_price_l219_21904

theorem shoe_price :
  ∀ (P : ℝ),
    (6 * P + 18 * 2 = 27 * 2) → P = 3 :=
by
  intro P H
  sorry

end NUMINAMATH_GPT_shoe_price_l219_21904


namespace NUMINAMATH_GPT_Berry_read_pages_thursday_l219_21987

theorem Berry_read_pages_thursday :
  ∀ (pages_per_day : ℕ) (pages_sunday : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) 
    (pages_wednesday : ℕ) (pages_friday : ℕ) (pages_saturday : ℕ),
    (pages_per_day = 50) →
    (pages_sunday = 43) →
    (pages_monday = 65) →
    (pages_tuesday = 28) →
    (pages_wednesday = 0) →
    (pages_friday = 56) →
    (pages_saturday = 88) →
    pages_sunday + pages_monday + pages_tuesday +
    pages_wednesday + pages_friday + pages_saturday + x = 350 →
    x = 70 := by
  sorry

end NUMINAMATH_GPT_Berry_read_pages_thursday_l219_21987


namespace NUMINAMATH_GPT_log_expression_equality_l219_21903

noncomputable def evaluate_log_expression : Real :=
  let log4_8 := (Real.log 8) / (Real.log 4)
  let log5_10 := (Real.log 10) / (Real.log 5)
  Real.sqrt (log4_8 + log5_10)

theorem log_expression_equality : 
  evaluate_log_expression = Real.sqrt ((5 / 2) + (Real.log 2 / Real.log 5)) :=
by
  sorry

end NUMINAMATH_GPT_log_expression_equality_l219_21903


namespace NUMINAMATH_GPT_part1_inequality_solution_l219_21920

def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 3|

theorem part1_inequality_solution :
  ∀ x : ℝ, f x ≤ 6 ↔ -4 / 3 ≤ x ∧ x ≤ 8 / 3 :=
by sorry

end NUMINAMATH_GPT_part1_inequality_solution_l219_21920


namespace NUMINAMATH_GPT_remainder_div_x_minus_2_l219_21989

noncomputable def q (x : ℝ) (A B C : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 10

theorem remainder_div_x_minus_2 (A B C : ℝ) (h : q 2 A B C = 20) : q (-2) A B C = 20 :=
by sorry

end NUMINAMATH_GPT_remainder_div_x_minus_2_l219_21989


namespace NUMINAMATH_GPT_entire_hike_length_l219_21994

-- Definitions directly from the conditions in part a)
def tripp_backpack_weight : ℕ := 25
def charlotte_backpack_weight : ℕ := tripp_backpack_weight - 7
def miles_hiked_first_day : ℕ := 9
def miles_left_to_hike : ℕ := 27

-- Theorem proving the entire hike length
theorem entire_hike_length :
  miles_hiked_first_day + miles_left_to_hike = 36 :=
by
  sorry

end NUMINAMATH_GPT_entire_hike_length_l219_21994


namespace NUMINAMATH_GPT_hiking_rate_up_the_hill_l219_21951

theorem hiking_rate_up_the_hill (r_down : ℝ) (t_total : ℝ) (t_up : ℝ) (r_up : ℝ) :
  r_down = 6 ∧ t_total = 3 ∧ t_up = 1.2 → r_up * t_up = 9 * t_up :=
by
  intro h
  let ⟨hrd, htt, htu⟩ := h
  sorry

end NUMINAMATH_GPT_hiking_rate_up_the_hill_l219_21951


namespace NUMINAMATH_GPT_base_7_to_base_10_equiv_l219_21971

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end NUMINAMATH_GPT_base_7_to_base_10_equiv_l219_21971


namespace NUMINAMATH_GPT_trig_identity_l219_21976

theorem trig_identity (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  - (Real.sin (2 * α) / Real.cos α) = -6/5 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l219_21976


namespace NUMINAMATH_GPT_min_cut_length_l219_21936

theorem min_cut_length (x : ℝ) (h_longer : 23 - x ≥ 0) (h_shorter : 15 - x ≥ 0) :
  23 - x ≥ 2 * (15 - x) → x ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_min_cut_length_l219_21936


namespace NUMINAMATH_GPT_probability_three_primes_out_of_five_l219_21902

def probability_of_prime (p : ℚ) : Prop := ∃ k, k = 4 ∧ p = 4/10

def probability_of_not_prime (p : ℚ) : Prop := ∃ k, k = 6 ∧ p = 6/10

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_three_primes_out_of_five :
  ∀ p_prime p_not_prime : ℚ, 
  probability_of_prime p_prime →
  probability_of_not_prime p_not_prime →
  (combinations 5 3 * (p_prime^3 * p_not_prime^2) = 720/3125) :=
by
  intros p_prime p_not_prime h_prime h_not_prime
  sorry

end NUMINAMATH_GPT_probability_three_primes_out_of_five_l219_21902


namespace NUMINAMATH_GPT_intersection_M_N_l219_21913

open Set Real

def M : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def N : Set ℝ := {x | log x / log 2 ≤ 1}

theorem intersection_M_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l219_21913


namespace NUMINAMATH_GPT_investment_amounts_proof_l219_21977

noncomputable def investment_proof_statement : Prop :=
  let p_investment_first_year := 52000
  let q_investment := (5/4) * p_investment_first_year
  let r_investment := (6/4) * p_investment_first_year;
  let p_investment_second_year := p_investment_first_year + (20/100) * p_investment_first_year;
  (q_investment = 65000) ∧ (r_investment = 78000) ∧ (q_investment = 65000) ∧ (r_investment = 78000)

theorem investment_amounts_proof : investment_proof_statement :=
  by
    sorry

end NUMINAMATH_GPT_investment_amounts_proof_l219_21977


namespace NUMINAMATH_GPT_sum_of_integers_greater_than_2_and_less_than_15_l219_21938

-- Define the set of integers greater than 2 and less than 15
def integersInRange : List ℕ := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define the sum of these integers
def sumIntegersInRange : ℕ := integersInRange.sum

-- The main theorem to prove the sum
theorem sum_of_integers_greater_than_2_and_less_than_15 : sumIntegersInRange = 102 := by
  -- The proof part is omitted as per instructions
  sorry

end NUMINAMATH_GPT_sum_of_integers_greater_than_2_and_less_than_15_l219_21938


namespace NUMINAMATH_GPT_percentage_less_than_a_plus_d_l219_21934

-- Define the mean, standard deviation, and given conditions
variables (a d : ℝ)
axiom symmetric_distribution : ∀ x, x = 2 * a - x 

-- Main theorem
theorem percentage_less_than_a_plus_d :
  (∃ (P_less_than : ℝ → ℝ), P_less_than (a + d) = 0.84) :=
sorry

end NUMINAMATH_GPT_percentage_less_than_a_plus_d_l219_21934


namespace NUMINAMATH_GPT_student_C_has_sweetest_water_l219_21993

-- Define concentrations for each student
def concentration_A : ℚ := 35 / 175 * 100
def concentration_B : ℚ := 45 / 175 * 100
def concentration_C : ℚ := 65 / 225 * 100

-- Prove that Student C has the highest concentration
theorem student_C_has_sweetest_water :
  concentration_C > concentration_B ∧ concentration_C > concentration_A :=
by
  -- By direct calculation from the provided conditions
  sorry

end NUMINAMATH_GPT_student_C_has_sweetest_water_l219_21993


namespace NUMINAMATH_GPT_min_distance_from_origin_l219_21975

-- Define the condition of the problem
def condition (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 4 = 0

-- Statement of the problem in Lean 4
theorem min_distance_from_origin (x y : ℝ) (h : condition x y) : 
  ∃ m : ℝ, m = Real.sqrt (x^2 + y^2) ∧ m = Real.sqrt 13 - 3 := 
sorry

end NUMINAMATH_GPT_min_distance_from_origin_l219_21975


namespace NUMINAMATH_GPT_overall_average_marks_l219_21967

theorem overall_average_marks 
  (n1 : ℕ) (m1 : ℕ) 
  (n2 : ℕ) (m2 : ℕ) 
  (n3 : ℕ) (m3 : ℕ) 
  (n4 : ℕ) (m4 : ℕ) 
  (h1 : n1 = 70) (h2 : m1 = 50) 
  (h3 : n2 = 35) (h4 : m2 = 60)
  (h5 : n3 = 45) (h6 : m3 = 55)
  (h7 : n4 = 42) (h8 : m4 = 45) :
  (n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4) / (n1 + n2 + n3 + n4) = 9965 / 192 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_marks_l219_21967


namespace NUMINAMATH_GPT_difference_of_interchanged_digits_l219_21944

theorem difference_of_interchanged_digits {x y : ℕ} (h : x - y = 4) :
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end NUMINAMATH_GPT_difference_of_interchanged_digits_l219_21944


namespace NUMINAMATH_GPT_inequality_proof_l219_21949

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                        (hb : 0 ≤ b) (hb1 : b ≤ 1)
                        (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l219_21949


namespace NUMINAMATH_GPT_bruce_will_be_3_times_as_old_in_6_years_l219_21912

variables (x : ℕ)

-- Definitions from conditions
def bruce_age_now := 36
def son_age_now := 8

-- Equivalent Lean 4 statement
theorem bruce_will_be_3_times_as_old_in_6_years :
  (bruce_age_now + x = 3 * (son_age_now + x)) → x = 6 :=
sorry

end NUMINAMATH_GPT_bruce_will_be_3_times_as_old_in_6_years_l219_21912


namespace NUMINAMATH_GPT_delivery_in_april_l219_21930

theorem delivery_in_april (n_jan n_mar : ℕ) (growth_rate : ℝ) :
  n_jan = 100000 → n_mar = 121000 → (1 + growth_rate) ^ 2 = n_mar / n_jan →
  (n_mar * (1 + growth_rate) = 133100) :=
by
  intros n_jan_eq n_mar_eq growth_eq
  sorry

end NUMINAMATH_GPT_delivery_in_april_l219_21930


namespace NUMINAMATH_GPT_sin_double_angle_l219_21910

theorem sin_double_angle {x : ℝ} (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l219_21910


namespace NUMINAMATH_GPT_floor_neg_seven_thirds_l219_21911

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_thirds_l219_21911


namespace NUMINAMATH_GPT_binom_20_17_l219_21926

theorem binom_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end NUMINAMATH_GPT_binom_20_17_l219_21926


namespace NUMINAMATH_GPT_ratio_of_cats_to_dogs_sold_l219_21958

theorem ratio_of_cats_to_dogs_sold (cats dogs : ℕ) (h1 : cats = 16) (h2 : dogs = 8) :
  (cats : ℚ) / dogs = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cats_to_dogs_sold_l219_21958


namespace NUMINAMATH_GPT_mushroom_pickers_at_least_50_l219_21957

-- Given conditions
variables (a : Fin 7 → ℕ) -- Each picker collects a different number of mushrooms.
variables (distinct : ∀ i j, i ≠ j → a i ≠ a j)
variable (total_mushrooms : (Finset.univ.sum a) = 100)

-- The proof that at least three of the pickers collected at least 50 mushrooms together
theorem mushroom_pickers_at_least_50 (a : Fin 7 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (total_mushrooms : (Finset.univ.sum a) = 100) :
    ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
sorry

end NUMINAMATH_GPT_mushroom_pickers_at_least_50_l219_21957


namespace NUMINAMATH_GPT_hyperbola_foci_l219_21998

-- Define the conditions and the question
def hyperbola_equation (x y : ℝ) : Prop := 
  x^2 - 4 * y^2 - 6 * x + 24 * y - 11 = 0

-- The foci of the hyperbola 
def foci (x1 x2 y1 y2 : ℝ) : Prop := 
  (x1, y1) = (3, 3 + 2 * Real.sqrt 5) ∨ (x2, y2) = (3, 3 - 2 * Real.sqrt 5)

-- The proof statement
theorem hyperbola_foci :
  ∃ x1 x2 y1 y2 : ℝ, hyperbola_equation x1 y1 ∧ foci x1 x2 y1 y2 :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_l219_21998


namespace NUMINAMATH_GPT_jacket_final_price_l219_21992

theorem jacket_final_price :
    let initial_price := 150
    let first_discount := 0.30
    let second_discount := 0.10
    let coupon := 10
    let tax := 0.05
    let price_after_first_discount := initial_price * (1 - first_discount)
    let price_after_second_discount := price_after_first_discount * (1 - second_discount)
    let price_after_coupon := price_after_second_discount - coupon
    let final_price := price_after_coupon * (1 + tax)
    final_price = 88.725 :=
by
  sorry

end NUMINAMATH_GPT_jacket_final_price_l219_21992


namespace NUMINAMATH_GPT_unique_seating_arrangements_l219_21956

/--
There are five couples including Charlie and his wife. The five men sit on the 
inner circle and each man's wife sits directly opposite him on the outer circle.
Prove that the number of unique seating arrangements where each man has another 
man seated directly to his right on the inner circle, counting all seat 
rotations as the same but not considering inner to outer flips as different, is 30.
-/
theorem unique_seating_arrangements : 
  ∃ (n : ℕ), n = 30 := 
sorry

end NUMINAMATH_GPT_unique_seating_arrangements_l219_21956


namespace NUMINAMATH_GPT_ramesh_paid_price_l219_21907

variables 
  (P : Real) -- Labelled price of the refrigerator
  (paid_price : Real := 0.80 * P + 125 + 250) -- Price paid after discount and additional costs
  (sell_price : Real := 1.16 * P) -- Price to sell for 16% profit
  (sell_at : Real := 18560) -- Target selling price for given profit

theorem ramesh_paid_price : 
  1.16 * P = 18560 → paid_price = 13175 :=
by
  sorry

end NUMINAMATH_GPT_ramesh_paid_price_l219_21907


namespace NUMINAMATH_GPT_cost_of_gasoline_l219_21990

def odometer_initial : ℝ := 85120
def odometer_final : ℝ := 85150
def fuel_efficiency : ℝ := 30
def price_per_gallon : ℝ := 4.25

theorem cost_of_gasoline : 
  ((odometer_final - odometer_initial) / fuel_efficiency) * price_per_gallon = 4.25 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_gasoline_l219_21990


namespace NUMINAMATH_GPT_part1_part2_l219_21988

def p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 = 0 → x < 0

theorem part1 (a : ℝ) (hp : p a) : a ∈ Set.Iio (-1) ∪ Set.Ioi 6 :=
sorry

theorem part2 (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : a ∈ Set.Iio (-1) ∪ Set.Ioc 2 6 :=
sorry

end NUMINAMATH_GPT_part1_part2_l219_21988


namespace NUMINAMATH_GPT_determine_b_l219_21946

noncomputable def Q (x : ℝ) (b : ℝ) : ℝ := x^3 + 3 * x^2 + b * x + 20

theorem determine_b (b : ℝ) :
  (∃ x : ℝ, x = 4 ∧ Q x b = 0) → b = -33 :=
by
  intro h
  rcases h with ⟨_, rfl, hQ⟩
  sorry

end NUMINAMATH_GPT_determine_b_l219_21946


namespace NUMINAMATH_GPT_work_problem_l219_21973

theorem work_problem (W : ℕ) (T_AB T_A T_B together_worked alone_worked remaining_work : ℕ)
  (h1 : T_AB = 30)
  (h2 : T_A = 60)
  (h3 : together_worked = 20)
  (h4 : T_B = 30)
  (h5 : remaining_work = W / 3)
  (h6 : alone_worked = 20)
  : alone_worked = 20 :=
by
  /- Proof is not required -/
  sorry

end NUMINAMATH_GPT_work_problem_l219_21973


namespace NUMINAMATH_GPT_smallest_a_condition_l219_21918

theorem smallest_a_condition
  (a b : ℝ)
  (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_eq : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a = 15 :=
sorry

end NUMINAMATH_GPT_smallest_a_condition_l219_21918


namespace NUMINAMATH_GPT_dollar_expansion_l219_21915

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2 + a * b

theorem dollar_expansion : dollar ((x - y) ^ 3) ((y - x) ^ 3) = -((x - y) ^ 6) := by
  sorry

end NUMINAMATH_GPT_dollar_expansion_l219_21915


namespace NUMINAMATH_GPT_product_of_divisor_and_dividend_l219_21908

theorem product_of_divisor_and_dividend (d D : ℕ) (q : ℕ := 6) (r : ℕ := 3) 
  (h₁ : D = d + 78) 
  (h₂ : D = d * q + r) : 
  D * d = 1395 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_divisor_and_dividend_l219_21908


namespace NUMINAMATH_GPT_triangle_inequality_for_roots_l219_21900

theorem triangle_inequality_for_roots (p q r : ℝ) (hroots_pos : ∀ (u v w : ℝ), (u > 0) ∧ (v > 0) ∧ (w > 0) ∧ (u * v * w = -r) ∧ (u + v + w = -p) ∧ (u * v + u * w + v * w = q)) :
  p^3 - 4 * p * q + 8 * r > 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_for_roots_l219_21900


namespace NUMINAMATH_GPT_volume_of_prism_l219_21929

theorem volume_of_prism (a b c : ℝ)
  (h_ab : a * b = 36)
  (h_ac : a * c = 54)
  (h_bc : b * c = 72) :
  a * b * c = 648 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l219_21929


namespace NUMINAMATH_GPT_find_x_l219_21968

def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem find_x 
  (x y : ℤ) 
  (h_star1 : star 5 4 2 2 = (7, 2)) 
  (h_eq : star x y 3 3 = (7, 2)) : 
  x = 4 := 
sorry

end NUMINAMATH_GPT_find_x_l219_21968


namespace NUMINAMATH_GPT_Alyssa_next_year_games_l219_21980

theorem Alyssa_next_year_games 
  (games_this_year : ℕ) 
  (games_last_year : ℕ) 
  (total_games : ℕ) 
  (games_up_to_this_year : ℕ)
  (total_up_to_next_year : ℕ) 
  (H1 : games_this_year = 11)
  (H2 : games_last_year = 13)
  (H3 : total_up_to_next_year = 39)
  (H4 : games_up_to_this_year = games_this_year + games_last_year) :
  total_up_to_next_year - games_up_to_this_year = 15 :=
by
  sorry

end NUMINAMATH_GPT_Alyssa_next_year_games_l219_21980


namespace NUMINAMATH_GPT_min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l219_21942

theorem min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared 
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  (ae^2 : ℝ) + (bf^2 : ℝ) + (cg^2 : ℝ) + (dh^2 : ℝ) ≥ 32 := 
sorry

end NUMINAMATH_GPT_min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l219_21942


namespace NUMINAMATH_GPT_divisibility_of_n_squared_plus_n_plus_two_l219_21978

-- Definition: n is a natural number.
def n (n : ℕ) : Prop := True

-- Theorem: For any natural number n, n^2 + n + 2 is always divisible by 2, but not necessarily divisible by 5.
theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) : 
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (¬ ∃ m : ℕ, n^2 + n + 2 = 5 * m) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_of_n_squared_plus_n_plus_two_l219_21978


namespace NUMINAMATH_GPT_ab_value_l219_21927

theorem ab_value (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ∧ (∀ y : ℝ, (x = 0 ∧ (y = 5 ∨ y = -5)))))
  (h2 : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (∀ x : ℝ, (y = 0 ∧ (x = 8 ∨ x = -8))))) :
  |a * b| = Real.sqrt 867.75 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l219_21927


namespace NUMINAMATH_GPT_sphere_radius_l219_21939

theorem sphere_radius (A : ℝ) (k1 k2 k3 : ℝ) (h : A = 64 * Real.pi) : ∃ r : ℝ, r = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sphere_radius_l219_21939


namespace NUMINAMATH_GPT_cost_apples_l219_21964

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_apples_l219_21964


namespace NUMINAMATH_GPT_shawn_divided_into_groups_l219_21916

theorem shawn_divided_into_groups :
  ∀ (total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups : ℕ),
  total_pebbles = 40 →
  red_pebbles = 9 →
  blue_pebbles = 13 →
  remaining_pebbles = total_pebbles - red_pebbles - blue_pebbles →
  remaining_pebbles % 3 = 0 →
  yellow_pebbles = blue_pebbles - 7 →
  remaining_pebbles = groups * yellow_pebbles →
  groups = 3 :=
by
  intros total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups
  intros h_total h_red h_blue h_remaining h_divisible h_yellow h_group
  sorry

end NUMINAMATH_GPT_shawn_divided_into_groups_l219_21916


namespace NUMINAMATH_GPT_sum_of_tangents_l219_21961

theorem sum_of_tangents (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h_tan_α : Real.tan α = 2) (h_tan_β : Real.tan β = 3) : α + β = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tangents_l219_21961


namespace NUMINAMATH_GPT_cara_total_amount_owed_l219_21919

-- Define the conditions
def principal : ℝ := 54
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the simple interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the total amount owed calculation
def total_amount_owed (P R T : ℝ) : ℝ := P + (interest P R T)

-- The proof statement
theorem cara_total_amount_owed : total_amount_owed principal rate time = 56.70 := by
  sorry

end NUMINAMATH_GPT_cara_total_amount_owed_l219_21919


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l219_21932

theorem no_real_roots_of_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l219_21932


namespace NUMINAMATH_GPT_processing_time_600_parts_l219_21984

theorem processing_time_600_parts :
  ∀ (x: ℕ), x = 600 → (∃ y : ℝ, y = 0.01 * x + 0.5 ∧ y = 6.5) :=
by
  sorry

end NUMINAMATH_GPT_processing_time_600_parts_l219_21984


namespace NUMINAMATH_GPT_negation_universal_proposition_l219_21945

theorem negation_universal_proposition {x : ℝ} : 
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := 
sorry

end NUMINAMATH_GPT_negation_universal_proposition_l219_21945


namespace NUMINAMATH_GPT_domain_of_expression_l219_21970

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_domain_of_expression_l219_21970


namespace NUMINAMATH_GPT_smallest_fraction_l219_21962

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) (eqn : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
sorry

end NUMINAMATH_GPT_smallest_fraction_l219_21962


namespace NUMINAMATH_GPT_small_load_clothing_count_l219_21981

def initial_clothes : ℕ := 36
def first_load_clothes : ℕ := 18
def remaining_clothes := initial_clothes - first_load_clothes
def small_load_clothes := remaining_clothes / 2

theorem small_load_clothing_count : 
  small_load_clothes = 9 :=
by
  sorry

end NUMINAMATH_GPT_small_load_clothing_count_l219_21981


namespace NUMINAMATH_GPT_ratio_expression_l219_21937

theorem ratio_expression (p q s u : ℚ) (h1 : p / q = 3 / 5) (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 :=
by
  sorry

end NUMINAMATH_GPT_ratio_expression_l219_21937


namespace NUMINAMATH_GPT_range_of_k_l219_21982

theorem range_of_k {x k : ℝ} :
  (∀ x, ((x - 2) * (x + 1) > 0) → ((2 * x + 7) * (x + k) < 0)) →
  (x = -3 ∨ x = -2) → 
  -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_GPT_range_of_k_l219_21982


namespace NUMINAMATH_GPT_total_amount_740_l219_21979

theorem total_amount_740 (x y z : ℝ) (hz : z = 200) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 740 := by
  sorry

end NUMINAMATH_GPT_total_amount_740_l219_21979


namespace NUMINAMATH_GPT_smallest_uv_non_factor_of_48_l219_21905

theorem smallest_uv_non_factor_of_48 :
  ∃ (u v : ℕ) (hu : u ∣ 48) (hv : v ∣ 48), u ≠ v ∧ ¬ (u * v ∣ 48) ∧ u * v = 18 :=
sorry

end NUMINAMATH_GPT_smallest_uv_non_factor_of_48_l219_21905


namespace NUMINAMATH_GPT_sum_of_multiples_is_even_l219_21963

theorem sum_of_multiples_is_even (a b : ℤ) (h1 : ∃ m : ℤ, a = 4 * m) (h2 : ∃ n : ℤ, b = 6 * n) : Even (a + b) :=
sorry

end NUMINAMATH_GPT_sum_of_multiples_is_even_l219_21963


namespace NUMINAMATH_GPT_theater_total_bills_l219_21928

theorem theater_total_bills (tickets : ℕ) (price : ℕ) (x : ℕ) (number_of_5_bills : ℕ) (number_of_10_bills : ℕ) (number_of_20_bills : ℕ) :
  tickets = 300 →
  price = 40 →
  number_of_20_bills = x →
  number_of_10_bills = 2 * x →
  number_of_5_bills = 2 * x + 20 →
  20 * x + 10 * (2 * x) + 5 * (2 * x + 20) = tickets * price →
  number_of_5_bills + number_of_10_bills + number_of_20_bills = 1210 := by
    intro h_tickets h_price h_20_bills h_10_bills h_5_bills h_total
    sorry

end NUMINAMATH_GPT_theater_total_bills_l219_21928


namespace NUMINAMATH_GPT_average_weight_of_students_l219_21909

theorem average_weight_of_students (b_avg_weight g_avg_weight : ℝ) (num_boys num_girls : ℕ)
  (hb : b_avg_weight = 155) (hg : g_avg_weight = 125) (hb_num : num_boys = 8) (hg_num : num_girls = 5) :
  (num_boys * b_avg_weight + num_girls * g_avg_weight) / (num_boys + num_girls) = 143 :=
by sorry

end NUMINAMATH_GPT_average_weight_of_students_l219_21909


namespace NUMINAMATH_GPT_max_length_shortest_arc_l219_21974

theorem max_length_shortest_arc (C : ℝ) (hC : C = 84) : 
  ∃ shortest_arc_length : ℝ, shortest_arc_length = 2 :=
by
  -- now prove it
  sorry

end NUMINAMATH_GPT_max_length_shortest_arc_l219_21974


namespace NUMINAMATH_GPT_asian_population_percentage_in_west_l219_21933

theorem asian_population_percentage_in_west
    (NE MW South West : ℕ)
    (H_NE : NE = 2)
    (H_MW : MW = 3)
    (H_South : South = 2)
    (H_West : West = 6)
    : (West * 100) / (NE + MW + South + West) = 46 :=
sorry

end NUMINAMATH_GPT_asian_population_percentage_in_west_l219_21933


namespace NUMINAMATH_GPT_train_crossing_time_l219_21966

noncomputable def length_of_train : ℕ := 250
noncomputable def length_of_bridge : ℕ := 350
noncomputable def speed_of_train_kmph : ℕ := 72

noncomputable def speed_of_train_mps : ℕ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem train_crossing_time : total_distance / speed_of_train_mps = 30 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l219_21966


namespace NUMINAMATH_GPT_friends_playing_video_game_l219_21952

def total_lives : ℕ := 64
def lives_per_player : ℕ := 8

theorem friends_playing_video_game (num_friends : ℕ) :
  num_friends = total_lives / lives_per_player :=
sorry

end NUMINAMATH_GPT_friends_playing_video_game_l219_21952


namespace NUMINAMATH_GPT_nonagon_perimeter_l219_21921

theorem nonagon_perimeter (n : ℕ) (side_length : ℝ) (P : ℝ) :
  n = 9 → side_length = 3 → P = n * side_length → P = 27 :=
by sorry

end NUMINAMATH_GPT_nonagon_perimeter_l219_21921
