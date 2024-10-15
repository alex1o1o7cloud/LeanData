import Mathlib

namespace NUMINAMATH_GPT_carla_book_count_l1039_103917

theorem carla_book_count (tiles_count books_count : ℕ) 
  (tiles_monday : tiles_count = 38)
  (total_tuesday_count : 2 * tiles_count + 3 * books_count = 301) : 
  books_count = 75 :=
by
  sorry

end NUMINAMATH_GPT_carla_book_count_l1039_103917


namespace NUMINAMATH_GPT_angle_A_is_60_degrees_value_of_b_plus_c_l1039_103948

noncomputable def triangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  let area := (3 * Real.sqrt 3) / 2
  c + 2 * a * Real.cos C = 2 * b ∧
  1/2 * b * c * Real.sin A = area 

theorem angle_A_is_60_degrees (A B C : ℝ) (a b c : ℝ) :
  triangleABC A B C a b c →
  Real.cos A = 1 / 2 → 
  A = 60 :=
by
  intros h1 h2 
  sorry

theorem value_of_b_plus_c (A B C : ℝ) (b c : ℝ) :
  triangleABC A B C (Real.sqrt 7) b c →
  b * c = 6 →
  (b + c) = 5 :=
by 
  intros h1 h2 
  sorry

end NUMINAMATH_GPT_angle_A_is_60_degrees_value_of_b_plus_c_l1039_103948


namespace NUMINAMATH_GPT_trip_total_hours_l1039_103985

theorem trip_total_hours
    (x : ℕ) -- additional hours of travel
    (dist_1 : ℕ := 30 * 6) -- distance for first 6 hours
    (dist_2 : ℕ := 46 * x) -- distance for additional hours
    (total_dist : ℕ := dist_1 + dist_2) -- total distance
    (total_time : ℕ := 6 + x) -- total time
    (avg_speed : ℕ := total_dist / total_time) -- average speed
    (h : avg_speed = 34) : total_time = 8 :=
by
  sorry

end NUMINAMATH_GPT_trip_total_hours_l1039_103985


namespace NUMINAMATH_GPT_find_number_l1039_103903

theorem find_number (a b x : ℝ) (H1 : 2 * a = x * b) (H2 : a * b ≠ 0) (H3 : (a / 3) / (b / 2) = 1) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1039_103903


namespace NUMINAMATH_GPT_worker_days_total_l1039_103911

theorem worker_days_total
  (W I : ℕ)
  (hw : 20 * W - 3 * I = 280)
  (hi : I = 40) :
  W + I = 60 :=
by
  sorry

end NUMINAMATH_GPT_worker_days_total_l1039_103911


namespace NUMINAMATH_GPT_original_number_count_l1039_103989

theorem original_number_count (k S : ℕ) (M : ℚ)
  (hk : k > 0)
  (hM : M = S / k)
  (h_add15 : (S + 15) / (k + 1) = M + 2)
  (h_add1 : (S + 16) / (k + 2) = M + 1) :
  k = 6 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_original_number_count_l1039_103989


namespace NUMINAMATH_GPT_find_lcm_of_two_numbers_l1039_103983

theorem find_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (prod : ℕ) 
  (h1 : hcf = 22) (h2 : prod = 62216) (h3 : A * B = prod) (h4 : Nat.gcd A B = hcf) :
  Nat.lcm A B = 2828 := 
by
  sorry

end NUMINAMATH_GPT_find_lcm_of_two_numbers_l1039_103983


namespace NUMINAMATH_GPT_original_number_of_men_l1039_103960

-- Define the conditions
def work_days_by_men (M : ℕ) (days : ℕ) : ℕ := M * days
def additional_men (M : ℕ) : ℕ := M + 10
def completed_days : ℕ := 9

-- The main theorem
theorem original_number_of_men : ∀ (M : ℕ), 
  work_days_by_men M 12 = work_days_by_men (additional_men M) completed_days → 
  M = 30 :=
by
  intros M h
  sorry

end NUMINAMATH_GPT_original_number_of_men_l1039_103960


namespace NUMINAMATH_GPT_who_stole_the_broth_l1039_103941

-- Define the suspects
inductive Suspect
| MarchHare : Suspect
| MadHatter : Suspect
| Dormouse : Suspect

open Suspect

-- Define the statements
def stole_broth (s : Suspect) : Prop :=
  s = Dormouse

def told_truth (s : Suspect) : Prop :=
  s = Dormouse

-- The March Hare's testimony
def march_hare_testimony : Prop :=
  stole_broth MadHatter

-- Conditions
def condition1 : Prop := ∃! s, stole_broth s
def condition2 : Prop := ∀ s, told_truth s ↔ stole_broth s
def condition3 : Prop := told_truth MarchHare → stole_broth MadHatter

-- Combining conditions into a single proposition to prove
theorem who_stole_the_broth : 
  (condition1 ∧ condition2 ∧ condition3) → stole_broth Dormouse := sorry

end NUMINAMATH_GPT_who_stole_the_broth_l1039_103941


namespace NUMINAMATH_GPT_alpha_in_second_quadrant_l1039_103914

theorem alpha_in_second_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.cos α - Real.sin α < 0) : 
  π / 2 < α ∧ α < π :=
sorry

end NUMINAMATH_GPT_alpha_in_second_quadrant_l1039_103914


namespace NUMINAMATH_GPT_distance_between_lines_is_two_l1039_103965

noncomputable def distance_between_parallel_lines : ℝ := 
  let A1 := 3
  let B1 := 4
  let C1 := -3
  let A2 := 6
  let B2 := 8
  let C2 := 14
  (|C2 - C1| : ℝ) / Real.sqrt (A2^2 + B2^2)

theorem distance_between_lines_is_two :
  distance_between_parallel_lines = 2 := by
  sorry

end NUMINAMATH_GPT_distance_between_lines_is_two_l1039_103965


namespace NUMINAMATH_GPT_find_x_l1039_103939

noncomputable def vector_a : ℝ × ℝ × ℝ := (-3, 2, 5)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-3) * 1 + 2 * x + 5 * (-1) = 2) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1039_103939


namespace NUMINAMATH_GPT_family_visit_cost_is_55_l1039_103906

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end NUMINAMATH_GPT_family_visit_cost_is_55_l1039_103906


namespace NUMINAMATH_GPT_group_size_l1039_103950

def total_people (I N B Ne : ℕ) : ℕ := I + N - B + B + Ne

theorem group_size :
  let I := 55
  let N := 43
  let B := 61
  let Ne := 63
  total_people I N B Ne = 161 :=
by
  sorry

end NUMINAMATH_GPT_group_size_l1039_103950


namespace NUMINAMATH_GPT_total_leaves_l1039_103919

def fernTypeA_fronds := 15
def fernTypeA_leaves_per_frond := 45
def fernTypeB_fronds := 20
def fernTypeB_leaves_per_frond := 30
def fernTypeC_fronds := 25
def fernTypeC_leaves_per_frond := 40

def fernTypeA_count := 4
def fernTypeB_count := 5
def fernTypeC_count := 3

theorem total_leaves : 
  fernTypeA_count * (fernTypeA_fronds * fernTypeA_leaves_per_frond) + 
  fernTypeB_count * (fernTypeB_fronds * fernTypeB_leaves_per_frond) + 
  fernTypeC_count * (fernTypeC_fronds * fernTypeC_leaves_per_frond) = 
  8700 := 
sorry

end NUMINAMATH_GPT_total_leaves_l1039_103919


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1997_l1039_103924

theorem rightmost_three_digits_of_7_pow_1997 :
  7^1997 % 1000 = 207 :=
by
  sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1997_l1039_103924


namespace NUMINAMATH_GPT_min_value_of_function_l1039_103958

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (x + 1/x + x^2 + 1/x^2 + 1 / (x + 1/x + x^2 + 1/x^2)) = 4.25 := by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l1039_103958


namespace NUMINAMATH_GPT_good_quadruple_inequality_l1039_103935

theorem good_quadruple_inequality {p a b c : ℕ} (hp : Nat.Prime p) (hodd : p % 2 = 1) 
(habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
(hab : (a * b + 1) % p = 0) (hbc : (b * c + 1) % p = 0) (hca : (c * a + 1) % p = 0) :
  p + 2 ≤ (a + b + c) / 3 := 
by
  sorry

end NUMINAMATH_GPT_good_quadruple_inequality_l1039_103935


namespace NUMINAMATH_GPT_paired_products_not_equal_1000_paired_products_equal_10000_l1039_103973

open Nat

theorem paired_products_not_equal_1000 :
  ∀ (a : Fin 1000 → ℤ), (∃ p n : Nat, p + n = 1000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) → False :=
by 
  sorry

theorem paired_products_equal_10000 :
  ∀ (a : Fin 10000 → ℤ), (∃ p n : Nat, p + n = 10000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) ↔ p = 5050 ∨ p = 4950 :=
by 
  sorry

end NUMINAMATH_GPT_paired_products_not_equal_1000_paired_products_equal_10000_l1039_103973


namespace NUMINAMATH_GPT_jay_used_zero_fraction_of_gallon_of_paint_l1039_103938

theorem jay_used_zero_fraction_of_gallon_of_paint
    (dexter_used : ℝ := 3/8)
    (gallon_in_liters : ℝ := 4)
    (paint_left_liters : ℝ := 4) :
    dexter_used = 3/8 ∧ gallon_in_liters = 4 ∧ paint_left_liters = 4 →
    ∃ jay_used : ℝ, jay_used = 0 :=
by
  sorry

end NUMINAMATH_GPT_jay_used_zero_fraction_of_gallon_of_paint_l1039_103938


namespace NUMINAMATH_GPT_sum_first_eight_geom_terms_eq_l1039_103995

noncomputable def S8_geom_sum : ℚ :=
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  a * (1 - r^8) / (1 - r)

theorem sum_first_eight_geom_terms_eq :
  S8_geom_sum = 3280 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_eight_geom_terms_eq_l1039_103995


namespace NUMINAMATH_GPT_seonho_original_money_l1039_103902

variable (X : ℝ)
variable (spent_snacks : ℝ := (1/4) * X)
variable (remaining_after_snacks : ℝ := X - spent_snacks)
variable (spent_food : ℝ := (2/3) * remaining_after_snacks)
variable (final_remaining : ℝ := remaining_after_snacks - spent_food)

theorem seonho_original_money :
  final_remaining = 2500 -> X = 10000 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_seonho_original_money_l1039_103902


namespace NUMINAMATH_GPT_function_is_decreasing_l1039_103975

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2

theorem function_is_decreasing (a b : ℝ) (f_even : ∀ x : ℝ, f a b x = f a b (-x))
  (domain_condition : 1 + a + 2 = 0) :
  ∀ x y : ℝ, 1 ≤ x → x < y → y ≤ 2 → f a 0 x > f a 0 y :=
by
  sorry

end NUMINAMATH_GPT_function_is_decreasing_l1039_103975


namespace NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l1039_103976

-- Problem 1
theorem factorize_problem1 (a b : ℝ) : 
    -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := 
by sorry

-- Problem 2
theorem factorize_problem2 (a b x y : ℝ) : 
    9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) := 
by sorry

end NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l1039_103976


namespace NUMINAMATH_GPT_number_of_students_l1039_103957

-- Define the conditions as hypotheses
def ordered_apples : ℕ := 6 + 15   -- 21 apples ordered
def extra_apples : ℕ := 16         -- 16 extra apples after distribution

-- Define the main theorem statement to prove S = 21
theorem number_of_students (S : ℕ) (H1 : ordered_apples = 21) (H2 : extra_apples = 16) : S = 21 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_l1039_103957


namespace NUMINAMATH_GPT_cards_left_l1039_103922

def number_of_initial_cards : ℕ := 67
def number_of_cards_taken : ℕ := 9

theorem cards_left (l : ℕ) (d : ℕ) (hl : l = number_of_initial_cards) (hd : d = number_of_cards_taken) : l - d = 58 :=
by
  sorry

end NUMINAMATH_GPT_cards_left_l1039_103922


namespace NUMINAMATH_GPT_total_go_stones_correct_l1039_103998

-- Definitions based on the problem's conditions
def stones_per_bundle : Nat := 10
def num_bundles : Nat := 3
def white_stones : Nat := 16

-- A function that calculates the total number of go stones
def total_go_stones : Nat :=
  num_bundles * stones_per_bundle + white_stones

-- The theorem we want to prove
theorem total_go_stones_correct : total_go_stones = 46 :=
by
  sorry

end NUMINAMATH_GPT_total_go_stones_correct_l1039_103998


namespace NUMINAMATH_GPT_probability_fly_reaches_8_10_l1039_103907

theorem probability_fly_reaches_8_10 :
  let total_steps := 2^18
  let right_up_combinations := Nat.choose 18 8
  (right_up_combinations / total_steps : ℚ) = Nat.choose 18 8 / 2^18 := 
sorry

end NUMINAMATH_GPT_probability_fly_reaches_8_10_l1039_103907


namespace NUMINAMATH_GPT_correct_judgements_l1039_103999

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_period_1 : ∀ x : ℝ, f (x + 1) = -f x
axiom f_increasing_0_1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

theorem correct_judgements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧ 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧ 
  ¬(∀ x y : ℝ, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y) :=
by 
  sorry

end NUMINAMATH_GPT_correct_judgements_l1039_103999


namespace NUMINAMATH_GPT_infinite_solutions_l1039_103982

theorem infinite_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (k : ℕ), x = k^3 + 1 ∧ y = (k^3 + 1) * k := 
sorry

end NUMINAMATH_GPT_infinite_solutions_l1039_103982


namespace NUMINAMATH_GPT_sum_of_x_coords_l1039_103952

theorem sum_of_x_coords (x : ℝ) (y : ℝ) :
  y = abs (x^2 - 6*x + 8) ∧ y = 6 - x → (x = (5 + Real.sqrt 17) / 2 ∨ x = (5 - Real.sqrt 17) / 2 ∨ x = 2)
  →  ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) :=
by
  intros h1 h2
  have H : ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) := sorry
  exact H

end NUMINAMATH_GPT_sum_of_x_coords_l1039_103952


namespace NUMINAMATH_GPT_sum_of_possible_B_is_zero_l1039_103964

theorem sum_of_possible_B_is_zero :
  ∀ B : ℕ, B < 10 → (∃ k : ℤ, 7 * k = 500 + 10 * B + 3) -> B = 0 := sorry

end NUMINAMATH_GPT_sum_of_possible_B_is_zero_l1039_103964


namespace NUMINAMATH_GPT_sum_of_primes_eq_100_l1039_103984

theorem sum_of_primes_eq_100 : 
  ∃ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S → Nat.Prime x) ∧ S.sum id = 100 ∧ S.card = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_eq_100_l1039_103984


namespace NUMINAMATH_GPT_max_integer_valued_fractions_l1039_103912

-- Problem Statement:
-- Given a set of natural numbers from 1 to 22,
-- the maximum number of fractions that can be formed such that each fraction is an integer
-- (where an integer fraction is defined as a/b being an integer if and only if b divides a) is 10.

open Nat

theorem max_integer_valued_fractions : 
  ∀ (S : Finset ℕ), (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 22) →
  ∃ P : Finset (ℕ × ℕ), (∀ (a b : ℕ), (a, b) ∈ P → b ∣ a) ∧ P.card = 11 → 
  10 ≤ (P.filter (λ p => p.1 % p.2 = 0)).card :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_max_integer_valued_fractions_l1039_103912


namespace NUMINAMATH_GPT_no_prime_divisible_by_45_l1039_103936

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_45_l1039_103936


namespace NUMINAMATH_GPT_max_value_amc_am_mc_ca_l1039_103959

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end NUMINAMATH_GPT_max_value_amc_am_mc_ca_l1039_103959


namespace NUMINAMATH_GPT_largest_square_side_length_l1039_103980

noncomputable def largestInscribedSquareSide (s : ℝ) (sharedSide : ℝ) : ℝ :=
  let y := (s * Real.sqrt 2 - sharedSide * Real.sqrt 3) / (2 * Real.sqrt 2)
  y

theorem largest_square_side_length :
  let s := 12
  let t := (s * Real.sqrt 6) / 3
  largestInscribedSquareSide s t = 6 - Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_side_length_l1039_103980


namespace NUMINAMATH_GPT_sculpture_height_l1039_103970

def base_height: ℝ := 10  -- height of the base in inches
def combined_height_feet: ℝ := 3.6666666666666665  -- combined height in feet
def inches_per_foot: ℝ := 12  -- conversion factor from feet to inches

-- Convert combined height to inches
def combined_height_inches: ℝ := combined_height_feet * inches_per_foot

-- Math proof problem statement
theorem sculpture_height : combined_height_inches - base_height = 34 := by
  sorry

end NUMINAMATH_GPT_sculpture_height_l1039_103970


namespace NUMINAMATH_GPT_min_distance_from_point_to_line_l1039_103944

theorem min_distance_from_point_to_line : 
  ∀ (x₀ y₀ : Real), 3 * x₀ - 4 * y₀ - 10 = 0 → Real.sqrt (x₀^2 + y₀^2) = 2 :=
by sorry

end NUMINAMATH_GPT_min_distance_from_point_to_line_l1039_103944


namespace NUMINAMATH_GPT_fraction_simplification_l1039_103929

theorem fraction_simplification (x y : ℚ) (h1 : x = 4) (h2 : y = 5) : 
  (1 / y) / (1 / x) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1039_103929


namespace NUMINAMATH_GPT_partitioning_staircase_l1039_103956

def number_of_ways_to_partition_staircase (n : ℕ) : ℕ :=
  2^(n-1)

theorem partitioning_staircase (n : ℕ) : 
  number_of_ways_to_partition_staircase n = 2^(n-1) :=
by 
  sorry

end NUMINAMATH_GPT_partitioning_staircase_l1039_103956


namespace NUMINAMATH_GPT_least_cost_flower_bed_divisdes_l1039_103921

theorem least_cost_flower_bed_divisdes:
  let Region1 := 5 * 2
  let Region2 := 3 * 5
  let Region3 := 2 * 4
  let Region4 := 5 * 4
  let Region5 := 5 * 3
  let Cost_Dahlias := 2.70
  let Cost_Cannas := 2.20
  let Cost_Begonias := 1.70
  let Cost_Freesias := 3.20
  let total_cost := 
    Region1 * Cost_Dahlias + 
    Region2 * Cost_Cannas + 
    Region3 * Cost_Freesias + 
    Region4 * Cost_Begonias + 
    Region5 * Cost_Cannas
  total_cost = 152.60 :=
by
  sorry

end NUMINAMATH_GPT_least_cost_flower_bed_divisdes_l1039_103921


namespace NUMINAMATH_GPT_length_of_floor_y_l1039_103962

theorem length_of_floor_y
  (A B : ℝ)
  (hx : A = 10)
  (hy : B = 18)
  (width_y : ℝ)
  (length_y : ℝ)
  (width_y_eq : width_y = 9)
  (area_eq : A * B = width_y * length_y) :
  length_y = 20 := 
sorry

end NUMINAMATH_GPT_length_of_floor_y_l1039_103962


namespace NUMINAMATH_GPT_evaluate_expression_l1039_103915

def cube_root (x : ℝ) := x^(1/3)

theorem evaluate_expression : (cube_root (9 / 32))^2 = (3/8) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1039_103915


namespace NUMINAMATH_GPT_number_of_dogs_l1039_103972

theorem number_of_dogs (D C B x : ℕ) (h1 : D = 3 * x) (h2 : B = 9 * x) (h3 : D + B = 204) (h4 : 12 * x = 204) : D = 51 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_number_of_dogs_l1039_103972


namespace NUMINAMATH_GPT_ratio_cars_to_dogs_is_two_l1039_103900

-- Definitions of the conditions
def initial_dogs : ℕ := 90
def initial_cars : ℕ := initial_dogs / 3
def additional_cars : ℕ := 210
def current_dogs : ℕ := 120
def current_cars : ℕ := initial_cars + additional_cars

-- The statement to be proven
theorem ratio_cars_to_dogs_is_two :
  (current_cars : ℚ) / (current_dogs : ℚ) = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_cars_to_dogs_is_two_l1039_103900


namespace NUMINAMATH_GPT_ofelia_ratio_is_two_l1039_103987

noncomputable def OfeliaSavingsRatio : ℝ :=
  let january_savings := 10
  let may_savings := 160
  let x := (may_savings / january_savings)^(1/4)
  x

theorem ofelia_ratio_is_two : OfeliaSavingsRatio = 2 := by
  sorry

end NUMINAMATH_GPT_ofelia_ratio_is_two_l1039_103987


namespace NUMINAMATH_GPT_least_amount_to_add_l1039_103930

theorem least_amount_to_add (current_amount : ℕ) (n : ℕ) (divisor : ℕ) [NeZero divisor]
  (current_amount_eq : current_amount = 449774) (n_eq : n = 1) (divisor_eq : divisor = 6) :
  ∃ k : ℕ, (current_amount + k) % divisor = 0 ∧ k = n := by
  sorry

end NUMINAMATH_GPT_least_amount_to_add_l1039_103930


namespace NUMINAMATH_GPT_arithmetic_mean_l1039_103963

theorem arithmetic_mean (x b : ℝ) (h : x ≠ 0) : 
  (1 / 2) * ((2 + (b / x)) + (2 - (b / x))) = 2 :=
by sorry

end NUMINAMATH_GPT_arithmetic_mean_l1039_103963


namespace NUMINAMATH_GPT_infinite_series_value_l1039_103901

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_infinite_series_value_l1039_103901


namespace NUMINAMATH_GPT_binomial_coefficient_plus_ten_l1039_103945

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_plus_ten_l1039_103945


namespace NUMINAMATH_GPT_number_of_ordered_triples_l1039_103910

noncomputable def count_triples : Nat := 50

theorem number_of_ordered_triples 
    (x y z : Nat)
    (hx : x > 0)
    (hy : y > 0)
    (hz : z > 0)
    (H1 : Nat.lcm x y = 500)
    (H2 : Nat.lcm y z = 1000)
    (H3 : Nat.lcm z x = 1000) :
    ∃ (n : Nat), n = count_triples := 
by
    use 50
    sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l1039_103910


namespace NUMINAMATH_GPT_find_fixed_point_c_l1039_103969

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := 2 * x ^ 2 - c

theorem find_fixed_point_c (c : ℝ) : 
  (∃ a : ℝ, f a = a ∧ g a c = a) ↔ (c = 3 ∨ c = 6) := sorry

end NUMINAMATH_GPT_find_fixed_point_c_l1039_103969


namespace NUMINAMATH_GPT_sqrt_four_eq_pm_two_l1039_103974

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_four_eq_pm_two_l1039_103974


namespace NUMINAMATH_GPT_angle_degree_measure_l1039_103991

theorem angle_degree_measure (x : ℝ) (h1 : (x + (90 - x) = 90)) (h2 : (x = 3 * (90 - x))) : x = 67.5 := by
  sorry

end NUMINAMATH_GPT_angle_degree_measure_l1039_103991


namespace NUMINAMATH_GPT_find_first_purchase_find_max_profit_purchase_plan_l1039_103961

-- Defining the parameters for the problem
structure KeychainParams where
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  total_purchase_cost_first : ℕ
  total_keychains_first : ℕ
  total_purchase_cost_second : ℕ
  total_keychains_second : ℕ
  purchase_cap_second : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ

-- Define the initial setup
def params : KeychainParams := {
  purchase_price_A := 30,
  purchase_price_B := 25,
  total_purchase_cost_first := 850,
  total_keychains_first := 30,
  total_purchase_cost_second := 2200,
  total_keychains_second := 80,
  purchase_cap_second := 2200,
  selling_price_A := 45,
  selling_price_B := 37
}

-- Part 1: Prove the number of keychains purchased for each type
theorem find_first_purchase (x y : ℕ)
  (h₁ : x + y = params.total_keychains_first)
  (h₂ : params.purchase_price_A * x + params.purchase_price_B * y = params.total_purchase_cost_first) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2: Prove the purchase plan that maximizes the sales profit
theorem find_max_profit_purchase_plan (m : ℕ)
  (h₃ : m + (params.total_keychains_second - m) = params.total_keychains_second)
  (h₄ : params.purchase_price_A * m + params.purchase_price_B * (params.total_keychains_second - m) ≤ params.purchase_cap_second) :
  m = 40 ∧ (params.selling_price_A - params.purchase_price_A) * m + (params.selling_price_B - params.purchase_price_B) * (params.total_keychains_second - m) = 1080 :=
sorry

end NUMINAMATH_GPT_find_first_purchase_find_max_profit_purchase_plan_l1039_103961


namespace NUMINAMATH_GPT_problem_sum_of_k_l1039_103926

theorem problem_sum_of_k {a b c k : ℂ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_ratio : a / (1 - b) = k ∧ b / (1 - c) = k ∧ c / (1 - a) = k) :
  (if (k^2 - k + 1 = 0) then -(-1)/1 else 0) = 1 :=
sorry

end NUMINAMATH_GPT_problem_sum_of_k_l1039_103926


namespace NUMINAMATH_GPT_ratio_of_perimeter_to_b_l1039_103979

theorem ratio_of_perimeter_to_b (b : ℝ) (hb : b ≠ 0) :
  let p1 := (-2*b, -2*b)
  let p2 := (2*b, -2*b)
  let p3 := (2*b, 2*b)
  let p4 := (-2*b, 2*b)
  let l := (y = b * x)
  let d1 := 4*b
  let d2 := 4*b
  let d3 := 4*b
  let d4 := 4*b*Real.sqrt 2
  let perimeter := d1 + d2 + d3 + d4
  let ratio := perimeter / b
  ratio = 12 + 4 * Real.sqrt 2 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_ratio_of_perimeter_to_b_l1039_103979


namespace NUMINAMATH_GPT_eric_days_waited_l1039_103966

def num_chickens := 4
def eggs_per_chicken_per_day := 3
def total_eggs := 36

def eggs_per_day := num_chickens * eggs_per_chicken_per_day
def num_days := total_eggs / eggs_per_day

theorem eric_days_waited : num_days = 3 :=
by
  sorry

end NUMINAMATH_GPT_eric_days_waited_l1039_103966


namespace NUMINAMATH_GPT_total_weight_lifted_l1039_103967

-- Definitions based on conditions
def original_lift : ℝ := 80
def after_training : ℝ := original_lift * 2
def specialization_increment : ℝ := after_training * 0.10
def specialized_lift : ℝ := after_training + specialization_increment

-- Statement of the theorem to prove total weight lifted
theorem total_weight_lifted : 
  (specialized_lift * 2) = 352 :=
sorry

end NUMINAMATH_GPT_total_weight_lifted_l1039_103967


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1039_103925

-- Define the arithmetic sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := sorry

-- Given condition
axiom h1 : a_n 3 + a_n 7 = 37

-- Proof statement
theorem arithmetic_sequence_sum : a_n 2 + a_n 4 + a_n 6 + a_n 8 = 74 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1039_103925


namespace NUMINAMATH_GPT_negation_correct_l1039_103947

-- Define the initial statement
def initial_statement (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |x| ≥ 3

-- Define the negated statement
def negated_statement (s : Set ℝ) : Prop :=
  ∃ x ∈ s, |x| < 3

-- The theorem to be proven
theorem negation_correct (s : Set ℝ) :
  ¬(initial_statement s) ↔ negated_statement s := by
  sorry

end NUMINAMATH_GPT_negation_correct_l1039_103947


namespace NUMINAMATH_GPT_find_f_x_squared_l1039_103990

-- Define the function f with the given condition
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem find_f_x_squared : f (x^2) = (x^2 + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_x_squared_l1039_103990


namespace NUMINAMATH_GPT_solve_equation_l1039_103968

theorem solve_equation (x y z : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9) (h_eq : 1 / (x + y + z) = (x * 100 + y * 10 + z) / 1000) :
  x = 1 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1039_103968


namespace NUMINAMATH_GPT_point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l1039_103997

-- Define the travel records and the fuel consumption rate
def travel_records : List Int := [18, -9, 7, -14, -6, 13, -6, -8]
def fuel_consumption_rate : Float := 0.4

-- Question 1: Proof that point B is 5 km south of point A
theorem point_B_is_south_of_A : (travel_records.sum = -5) :=
  by sorry

-- Question 2: Proof that total distance traveled is 81 km
theorem total_distance_traveled : (travel_records.map Int.natAbs).sum = 81 :=
  by sorry

-- Question 3: Proof that the fuel consumed is 32 liters (Rounded)
theorem fuel_consumed : Float.floor (81 * fuel_consumption_rate) = 32 :=
  by sorry

end NUMINAMATH_GPT_point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l1039_103997


namespace NUMINAMATH_GPT_calc_pow_l1039_103928

-- Definitions used in the conditions
def base := 2
def exp := 10
def power := 2 / 5

-- Given condition
def given_identity : Pow.pow base exp = 1024 := by sorry

-- Statement to be proved
theorem calc_pow : Pow.pow 1024 power = 16 := by
  -- Use the given identity and known exponentiation rules to derive the result
  sorry

end NUMINAMATH_GPT_calc_pow_l1039_103928


namespace NUMINAMATH_GPT_fish_estimation_l1039_103977

noncomputable def number_caught := 50
noncomputable def number_marked_caught := 2
noncomputable def number_released := 30

theorem fish_estimation (N : ℕ) (h1 : number_caught = 50) 
  (h2 : number_marked_caught = 2) 
  (h3 : number_released = 30) :
  (number_marked_caught : ℚ) / number_caught = number_released / N → 
  N = 750 :=
by
  sorry

end NUMINAMATH_GPT_fish_estimation_l1039_103977


namespace NUMINAMATH_GPT_subproblem1_l1039_103927

theorem subproblem1 (a b c q : ℝ) (h1 : c = b * q) (h2 : c = a * q^2) : 
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 := 
sorry

end NUMINAMATH_GPT_subproblem1_l1039_103927


namespace NUMINAMATH_GPT_trajectory_is_parabola_l1039_103996

theorem trajectory_is_parabola
  (P : ℝ × ℝ) : 
  (dist P (0, P.2 + 1) < dist P (0, 2)) -> 
  (P.1^2 = 8 * (P.2 + 2)) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_is_parabola_l1039_103996


namespace NUMINAMATH_GPT_sum_of_numbers_l1039_103986

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149)
  (h2 : ab + bc + ca = 70) : 
  a + b + c = 17 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1039_103986


namespace NUMINAMATH_GPT_inspection_time_l1039_103953

theorem inspection_time 
  (num_digits : ℕ) (num_letters : ℕ) 
  (letter_opts : ℕ) (start_digits : ℕ) 
  (inspection_time_three_hours : ℕ) 
  (probability : ℝ) 
  (num_vehicles : ℕ) 
  (vehicles_inspected : ℕ)
  (cond1 : num_digits = 4)
  (cond2 : num_letters = 2)
  (cond3 : letter_opts = 3)
  (cond4 : start_digits = 2)
  (cond5 : inspection_time_three_hours = 180) 
  (cond6 : probability = 0.02)
  (cond7 : num_vehicles = 900)
  (cond8 : vehicles_inspected = num_vehicles * probability) :
  vehicles_inspected = (inspection_time_three_hours / 10) :=
  sorry

end NUMINAMATH_GPT_inspection_time_l1039_103953


namespace NUMINAMATH_GPT_total_items_and_cost_per_pet_l1039_103913

theorem total_items_and_cost_per_pet
  (treats_Jane : ℕ)
  (treats_Wanda : ℕ := treats_Jane / 2)
  (bread_Jane : ℕ := (3 * treats_Jane) / 4)
  (bread_Wanda : ℕ := 90)
  (bread_Carla : ℕ := 40)
  (treats_Carla : ℕ := 5 * bread_Carla / 2)
  (items_Peter : ℕ := 140)
  (treats_Peter : ℕ := items_Peter / 3)
  (bread_Peter : ℕ := 2 * treats_Peter)
  (x y z : ℕ) :
  (∀ B : ℕ, B = bread_Jane + bread_Wanda + bread_Carla + bread_Peter) ∧
  (∀ T : ℕ, T = treats_Jane + treats_Wanda + treats_Carla + treats_Peter) ∧
  (∀ Total : ℕ, Total = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter)) ∧
  (∀ ExpectedTotal : ℕ, ExpectedTotal = 427) ∧
  (∀ Cost : ℕ, Cost = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) * x + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter) * y) ∧
  (∀ CostPerPet : ℕ, CostPerPet = Cost / z) ∧
  (B + T = 427) ∧
  ((Cost / z) = (235 * x + 192 * y) / z)
:=
  by
  sorry

end NUMINAMATH_GPT_total_items_and_cost_per_pet_l1039_103913


namespace NUMINAMATH_GPT_diet_soda_ratio_l1039_103923

def total_bottles : ℕ := 60
def diet_soda_bottles : ℕ := 14

theorem diet_soda_ratio : (diet_soda_bottles * 30) = (total_bottles * 7) :=
by {
  -- We're given that total_bottles = 60 and diet_soda_bottles = 14
  -- So to prove the ratio 14/60 is equivalent to 7/30:
  -- Multiplying both sides by 30 and 60 simplifies the arithmetic.
  sorry
}

end NUMINAMATH_GPT_diet_soda_ratio_l1039_103923


namespace NUMINAMATH_GPT_molecular_weight_calculation_l1039_103918

theorem molecular_weight_calculation
    (moles_total_mw : ℕ → ℝ)
    (hw : moles_total_mw 9 = 900) :
    moles_total_mw 1 = 100 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l1039_103918


namespace NUMINAMATH_GPT_ratio_joe_sara_l1039_103909

variables (S J : ℕ) (k : ℕ)

-- Conditions
#check J + S = 120
#check J = k * S + 6
#check J = 82

-- The goal is to prove the ratio J / S = 41 / 19
theorem ratio_joe_sara (h1 : J + S = 120) (h2 : J = k * S + 6) (h3 : J = 82) : J / S = 41 / 19 :=
sorry

end NUMINAMATH_GPT_ratio_joe_sara_l1039_103909


namespace NUMINAMATH_GPT_percentage_of_cash_is_20_l1039_103937

theorem percentage_of_cash_is_20
  (raw_materials : ℕ)
  (machinery : ℕ)
  (total_amount : ℕ)
  (h_raw_materials : raw_materials = 35000)
  (h_machinery : machinery = 40000)
  (h_total_amount : total_amount = 93750) :
  (total_amount - (raw_materials + machinery)) * 100 / total_amount = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_cash_is_20_l1039_103937


namespace NUMINAMATH_GPT_a_and_b_together_work_days_l1039_103971

-- Definitions for the conditions:
def a_work_rate : ℚ := 1 / 9
def b_work_rate : ℚ := 1 / 18

-- The theorem statement:
theorem a_and_b_together_work_days : (a_work_rate + b_work_rate)⁻¹ = 6 := by
  sorry

end NUMINAMATH_GPT_a_and_b_together_work_days_l1039_103971


namespace NUMINAMATH_GPT_find_equation_of_line_l1039_103949

variable (x y : ℝ)

def line_parallel (x y : ℝ) (m : ℝ) :=
  x - 2*y + m = 0

def line_through_point (x y : ℝ) (px py : ℝ) (m : ℝ) :=
  (px - 2 * py + m = 0)
  
theorem find_equation_of_line :
  let px := -1
  let py := 3
  ∃ m, line_parallel x y m ∧ line_through_point x y px py m ∧ m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_equation_of_line_l1039_103949


namespace NUMINAMATH_GPT_log_identity_l1039_103992

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_identity
    (a b c : ℝ)
    (h1 : a ^ 2 + b ^ 2 = c ^ 2)
    (h2 : a > 0)
    (h3 : c > 0)
    (h4 : b > 0)
    (h5 : c > b) :
    log_base (c + b) a + log_base (c - b) a = 2 * log_base (c + b) a * log_base (c - b) a :=
sorry

end NUMINAMATH_GPT_log_identity_l1039_103992


namespace NUMINAMATH_GPT_geometric_sequence_condition_l1039_103933

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < a 1) 
  (h2 : ∀ n, a (n + 1) = a n * q) :
  (a 1 < a 3) ↔ (a 1 < a 3) ∧ (a 3 < a 6) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l1039_103933


namespace NUMINAMATH_GPT_find_number_l1039_103951

theorem find_number (x : ℝ) (h : 0.6667 * x + 0.75 = 1.6667) : x = 1.375 :=
sorry

end NUMINAMATH_GPT_find_number_l1039_103951


namespace NUMINAMATH_GPT_y_squared_range_l1039_103993

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) : 15 ≤ y^2 ∧ y^2 ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_y_squared_range_l1039_103993


namespace NUMINAMATH_GPT_ratio_greater_than_one_ratio_greater_than_one_neg_l1039_103904

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end NUMINAMATH_GPT_ratio_greater_than_one_ratio_greater_than_one_neg_l1039_103904


namespace NUMINAMATH_GPT_circle_radius_condition_l1039_103994

theorem circle_radius_condition (c : ℝ) : 
  (∃ x y : ℝ, (x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ 
  (radius = 6) ↔ 
  c = -23 := by
  sorry

end NUMINAMATH_GPT_circle_radius_condition_l1039_103994


namespace NUMINAMATH_GPT_simplify_to_x5_l1039_103954

theorem simplify_to_x5 (x : ℝ) :
  x^2 * x^3 = x^5 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_simplify_to_x5_l1039_103954


namespace NUMINAMATH_GPT_MrWillamTaxPercentage_l1039_103920

-- Definitions
def TotalTaxCollected : ℝ := 3840
def MrWillamTax : ℝ := 480

-- Theorem Statement
theorem MrWillamTaxPercentage :
  (MrWillamTax / TotalTaxCollected) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_MrWillamTaxPercentage_l1039_103920


namespace NUMINAMATH_GPT_expected_intersections_100gon_l1039_103908

noncomputable def expected_intersections : ℝ :=
  let n := 100
  let total_pairs := (n * (n - 3) / 2)
  total_pairs * (1/3)

theorem expected_intersections_100gon :
  expected_intersections = 4850 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_intersections_100gon_l1039_103908


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l1039_103940

theorem quadratic_distinct_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a = 0 → x^2 - 2*x - a = 0 ∧ (∀ y : ℝ, y ≠ x → y^2 - 2*y - a = 0)) → 
  a > -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l1039_103940


namespace NUMINAMATH_GPT_sequence_solution_l1039_103955

theorem sequence_solution :
  ∀ (a : ℕ → ℝ), (∀ m n : ℕ, a (m^2 + n^2) = a m ^ 2 + a n ^ 2) →
  (0 ≤ a 0 ∧ a 0 ≤ a 1 ∧ a 1 ≤ a 2 ∧ ∀ n, a n ≤ a (n + 1)) →
  (∀ n, a n = 0) ∨ (∀ n, a n = n) ∨ (∀ n, a n = 1 / 2) :=
sorry

end NUMINAMATH_GPT_sequence_solution_l1039_103955


namespace NUMINAMATH_GPT_initial_cats_in_shelter_l1039_103943

theorem initial_cats_in_shelter
  (cats_found_monday : ℕ)
  (cats_found_tuesday : ℕ)
  (cats_adopted_wednesday : ℕ)
  (current_cats : ℕ)
  (total_adopted_cats : ℕ)
  (initial_cats : ℕ) :
  cats_found_monday = 2 →
  cats_found_tuesday = 1 →
  cats_adopted_wednesday = 3 →
  total_adopted_cats = cats_adopted_wednesday * 2 →
  current_cats = 17 →
  initial_cats = current_cats + total_adopted_cats - (cats_found_monday + cats_found_tuesday) →
  initial_cats = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_cats_in_shelter_l1039_103943


namespace NUMINAMATH_GPT_janet_total_distance_l1039_103932

-- Define the distances covered in each week for each activity
def week1_running := 8 * 5
def week1_cycling := 7 * 3

def week2_running := 10 * 4
def week2_swimming := 2 * 2

def week3_running := 6 * 5
def week3_hiking := 3 * 2

-- Total distances for each activity
def total_running := week1_running + week2_running + week3_running
def total_cycling := week1_cycling
def total_swimming := week2_swimming
def total_hiking := week3_hiking

-- Total distance covered
def total_distance := total_running + total_cycling + total_swimming + total_hiking

-- Prove that the total distance is 141 miles
theorem janet_total_distance : total_distance = 141 := by
  sorry

end NUMINAMATH_GPT_janet_total_distance_l1039_103932


namespace NUMINAMATH_GPT_relationship_between_fractions_l1039_103978

variable (a a' b b' : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a' > 0)
variable (h₃ : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2)

theorem relationship_between_fractions
  (a : ℝ) (a' : ℝ) (b : ℝ) (b' : ℝ)
  (h1 : a > 0) (h2 : a' > 0)
  (h3 : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2) :
  (b^2) / (a^2) > (b'^2) / (a'^2) :=
by sorry

end NUMINAMATH_GPT_relationship_between_fractions_l1039_103978


namespace NUMINAMATH_GPT_sector_angle_measure_l1039_103916

theorem sector_angle_measure
  (r l : ℝ)
  (h1 : 2 * r + l = 4)
  (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 :=
sorry

end NUMINAMATH_GPT_sector_angle_measure_l1039_103916


namespace NUMINAMATH_GPT_operation_evaluation_l1039_103981

def my_operation (x y : Int) : Int :=
  x * (y + 1) + x * y

theorem operation_evaluation :
  my_operation (-3) (-4) = 21 := by
  sorry

end NUMINAMATH_GPT_operation_evaluation_l1039_103981


namespace NUMINAMATH_GPT_first_loan_amount_l1039_103931

theorem first_loan_amount :
  ∃ (L₁ L₂ : ℝ) (r : ℝ),
  (L₂ = 4700) ∧
  (L₁ = L₂ + 1500) ∧
  (0.09 * L₂ + r * L₁ = 617) ∧
  (L₁ = 6200) :=
by 
  sorry

end NUMINAMATH_GPT_first_loan_amount_l1039_103931


namespace NUMINAMATH_GPT_n_mod_9_eq_6_l1039_103905

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_n_mod_9_eq_6_l1039_103905


namespace NUMINAMATH_GPT_order_of_values_l1039_103946

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_of_values (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_order_of_values_l1039_103946


namespace NUMINAMATH_GPT_inequality_proof_l1039_103988

open Real

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^2 / (b + c + d) + b^2 / (c + d + a) +
   c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1039_103988


namespace NUMINAMATH_GPT_hockey_team_ties_l1039_103934

theorem hockey_team_ties (W T : ℕ) (h1 : 2 * W + T = 60) (h2 : W = T + 12) : T = 12 :=
by
  sorry

end NUMINAMATH_GPT_hockey_team_ties_l1039_103934


namespace NUMINAMATH_GPT_coprime_squares_l1039_103942

theorem coprime_squares (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ k : ℕ, ab = k^2) : 
  ∃ p q : ℕ, a = p^2 ∧ b = q^2 :=
by
  sorry

end NUMINAMATH_GPT_coprime_squares_l1039_103942
