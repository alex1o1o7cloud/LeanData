import Mathlib

namespace three_digit_numbers_last_three_digits_of_square_l137_137642

theorem three_digit_numbers_last_three_digits_of_square (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n^2 % 1000) = n ↔ n = 376 ∨ n = 625 := 
sorry

end three_digit_numbers_last_three_digits_of_square_l137_137642


namespace nonagon_diagonals_count_l137_137968

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l137_137968


namespace remaining_students_l137_137030

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l137_137030


namespace fraction_simplification_l137_137674

theorem fraction_simplification
  (a b c x : ℝ)
  (hb : b ≠ 0)
  (hxc : c ≠ 0)
  (h : x = a / b)
  (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) :=
by
  sorry

end fraction_simplification_l137_137674


namespace periodic_sequence_a2019_l137_137727

theorem periodic_sequence_a2019 :
  (∃ (a : ℕ → ℤ),
    a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ 
    (∀ n : ℕ, n ≥ 4 → a n = a (n-1) * a (n-3)) ∧
    a 2019 = -1) :=
sorry

end periodic_sequence_a2019_l137_137727


namespace domain_of_f1_x2_l137_137617

theorem domain_of_f1_x2 (f : ℝ → ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → ∃ y, y = f x) → 
  (∀ x, -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → ∃ y, y = f (1 - x^2)) :=
by
  sorry

end domain_of_f1_x2_l137_137617


namespace grape_juice_percentage_l137_137908

theorem grape_juice_percentage
  (original_mixture : ℝ)
  (percent_grape_juice : ℝ)
  (added_grape_juice : ℝ)
  (h1 : original_mixture = 50)
  (h2 : percent_grape_juice = 0.10)
  (h3 : added_grape_juice = 10)
  : (percent_grape_juice * original_mixture + added_grape_juice) / (original_mixture + added_grape_juice) * 100 = 25 :=
by
  sorry

end grape_juice_percentage_l137_137908


namespace problem1_problem2_l137_137625

-- Definitions of the three conditions given
def condition1 (x y : Nat) : Prop := x > y
def condition2 (y z : Nat) : Prop := y > z
def condition3 (x z : Nat) : Prop := 2 * z > x

-- Problem 1: If the number of teachers is 4, prove the maximum number of female students is 6.
theorem problem1 (z : Nat) (hz : z = 4) : ∃ y : Nat, (∀ x : Nat, condition1 x y → condition2 y z → condition3 x z) ∧ y = 6 :=
by
  sorry

-- Problem 2: Prove the minimum number of people in the group is 12.
theorem problem2 : ∃ z x y : Nat, (condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ z < y ∧ y < x ∧ x < 2 * z) ∧ z = 3 ∧ x = 5 ∧ y = 4 ∧ x + y + z = 12 :=
by
  sorry

end problem1_problem2_l137_137625


namespace perpendicular_bisector_eq_l137_137183

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq_l137_137183


namespace two_non_coincident_planes_divide_space_l137_137606

-- Define conditions for non-coincident planes
def non_coincident_planes (P₁ P₂ : Plane) : Prop :=
  ¬(P₁ = P₂)

-- Define the main theorem based on the conditions and the question
theorem two_non_coincident_planes_divide_space (P₁ P₂ : Plane) 
  (h : non_coincident_planes P₁ P₂) :
  ∃ n : ℕ, n = 3 ∨ n = 4 :=
by
  sorry

end two_non_coincident_planes_divide_space_l137_137606


namespace probability_at_least_one_two_l137_137921

open ProbabilityTheory

noncomputable def prob_at_least_one_two :=
  let outcomes := Finset.univ : Finset (Fin 8 × Fin 8 × Fin 8)
  let valid_outcomes := outcomes.filter (λ xyz, xyz.1.val + xyz.2.val = 2 * xyz.3.val)
  let favorable_outcomes := valid_outcomes.filter (λ xyz, xyz.1.val = 1 ∨ xyz.2.val = 1 ∨ xyz.3.val = 1)
  (favorable_outcomes.card : ℚ) / (valid_outcomes.card : ℚ)

theorem probability_at_least_one_two : prob_at_least_one_two = 1 / 8 :=
by
  sorry -- Skip the proof

end probability_at_least_one_two_l137_137921


namespace boxes_left_l137_137856

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l137_137856


namespace ticket_sales_amount_theater_collected_50_dollars_l137_137075

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l137_137075


namespace sally_turnip_count_l137_137572

theorem sally_turnip_count (total_turnips : ℕ) (mary_turnips : ℕ) (sally_turnips : ℕ) 
  (h1: total_turnips = 242) 
  (h2: mary_turnips = 129) 
  (h3: total_turnips = mary_turnips + sally_turnips) : 
  sally_turnips = 113 := 
by 
  sorry

end sally_turnip_count_l137_137572


namespace min_value_of_a_l137_137655

theorem min_value_of_a : 
  ∃ (a : ℤ), ∃ x y : ℤ, x ≠ y ∧ |x| ≤ 10 ∧ (x - y^2 = a) ∧ (y - x^2 = a) ∧ a = -111 :=
by
  sorry

end min_value_of_a_l137_137655


namespace quadratic_eq_coeff_l137_137180

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_eq_coeff_l137_137180


namespace nora_third_tree_oranges_l137_137154

theorem nora_third_tree_oranges (a b c total : ℕ)
  (h_a : a = 80)
  (h_b : b = 60)
  (h_total : total = 260)
  (h_sum : total = a + b + c) :
  c = 120 :=
by
  -- The proof should go here
  sorry

end nora_third_tree_oranges_l137_137154


namespace total_earnings_correct_l137_137574

-- Define the earnings of each individual
def SalvadorEarnings := 1956
def SantoEarnings := SalvadorEarnings / 2
def MariaEarnings := 3 * SantoEarnings
def PedroEarnings := SantoEarnings + MariaEarnings

-- Define the total earnings calculation
def TotalEarnings := SalvadorEarnings + SantoEarnings + MariaEarnings + PedroEarnings

-- State the theorem to prove
theorem total_earnings_correct :
  TotalEarnings = 9780 :=
sorry

end total_earnings_correct_l137_137574


namespace find_two_numbers_l137_137198

theorem find_two_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 5) (harmonic_mean : 2 * a * b / (a + b) = 5 / 3) :
  (a = (15 + Real.sqrt 145) / 4 ∧ b = (15 - Real.sqrt 145) / 4) ∨
  (a = (15 - Real.sqrt 145) / 4 ∧ b = (15 + Real.sqrt 145) / 4) :=
by
  sorry

end find_two_numbers_l137_137198


namespace two_x_plus_y_eq_12_l137_137327

-- Variables representing the prime numbers x and y
variables {x y : ℕ}

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Prime n
def lcm_eq (a b c : ℕ) : Prop := Nat.lcm a b = c

-- The theorem statement
theorem two_x_plus_y_eq_12 (h1 : lcm_eq x y 10) (h2 : is_prime x) (h3 : is_prime y) (h4 : x > y) :
    2 * x + y = 12 :=
sorry

end two_x_plus_y_eq_12_l137_137327


namespace certain_percentage_l137_137420

variable {x p : ℝ}

theorem certain_percentage (h1 : 0.40 * x = 160) : p * x = 200 ↔ p = 0.5 := 
by
  sorry

end certain_percentage_l137_137420


namespace compare_f_minus1_f_1_l137_137657

variable (f : ℝ → ℝ)

-- Given conditions
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x : ℝ, f x = x^2 + 2 * x * (f 2 - 2 * x))

-- Goal statement
theorem compare_f_minus1_f_1 : f (-1) > f 1 :=
by sorry

end compare_f_minus1_f_1_l137_137657


namespace cos_2alpha_value_l137_137533

noncomputable def cos_double_angle (α : ℝ) : ℝ := Real.cos (2 * α)

theorem cos_2alpha_value (α : ℝ): 
  (∃ a : ℝ, α = Real.arctan (-3) + 2 * a * Real.pi) → cos_double_angle α = -4 / 5 :=
by
  intro h
  sorry

end cos_2alpha_value_l137_137533


namespace number_of_triangles_l137_137593

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l137_137593


namespace cannot_be_six_l137_137686

theorem cannot_be_six (n r : ℕ) (h_n : n = 6) : 3 * n ≠ 4 * r :=
by
  sorry

end cannot_be_six_l137_137686


namespace max_a_plus_b_cubed_plus_c_fourth_l137_137292

theorem max_a_plus_b_cubed_plus_c_fourth (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 := sorry

end max_a_plus_b_cubed_plus_c_fourth_l137_137292


namespace arithmetic_sequence_product_l137_137728

theorem arithmetic_sequence_product
  (a d : ℤ)
  (h1 : a + 5 * d = 17)
  (h2 : d = 2) :
  (a + 2 * d) * (a + 3 * d) = 143 :=
by
  sorry

end arithmetic_sequence_product_l137_137728


namespace digit_2_count_divisible_by_3_l137_137133
  
/--
The number of positive integers less than or equal to 3000
that contain at least one digit '2' and are divisible by 3 is 384.
-/
theorem digit_2_count_divisible_by_3 : 
  ∃ count : ℕ, count = nat.count (λ n : ℕ, n ≤ 3000 ∧
                                   (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 2) ∧ 
                                    n % 3 = 0) (range 1 3001) = 384 :=
begin
  sorry
end

end digit_2_count_divisible_by_3_l137_137133


namespace smallest_number_with_unique_digits_summing_to_32_l137_137382

-- Definition: Digits of a number n are all distinct
def all_digits_different (n : ℕ) : Prop :=
  let digits := (n.digits : List ℕ)
  digits.nodup

-- Definition: Sum of the digits of number n equals 32
def sum_of_digits_equals_32 (n : ℕ) : Prop :=
  let digits := (n.digits : List ℕ)
  digits.sum = 32

-- Theorem: The smallest number satisfying the given conditions
theorem smallest_number_with_unique_digits_summing_to_32 
  (n : ℕ) (h1 : all_digits_different n) (h2 : sum_of_digits_equals_32 n) :
  n = 26789 :=
sorry

end smallest_number_with_unique_digits_summing_to_32_l137_137382


namespace smallest_N_winning_strategy_l137_137445

theorem smallest_N_winning_strategy :
  ∃ (N : ℕ), (N > 0) ∧ (∀ (list : List ℕ), 
    (∀ x, x ∈ list → x > 0 ∧ x ≤ 25) ∧ 
    list.sum ≥ 200 → 
    ∃ (sublist : List ℕ), sublist ⊆ list ∧ 
    200 - N ≤ sublist.sum ∧ sublist.sum ≤ 200 + N) ∧ N = 11 :=
sorry

end smallest_N_winning_strategy_l137_137445


namespace value_of_x_l137_137833

theorem value_of_x (x : ℝ) (h : 3 * x + 15 = (1/3) * (7 * x + 45)) : x = 0 :=
by
  sorry

end value_of_x_l137_137833


namespace vanya_speed_l137_137738

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l137_137738


namespace total_games_played_l137_137430

-- Definition of the number of teams
def num_teams : ℕ := 20

-- Definition of the number of games each pair plays
def games_per_pair : ℕ := 10

-- Theorem stating the total number of games played
theorem total_games_played : (num_teams * (num_teams - 1) / 2) * games_per_pair = 1900 :=
by sorry

end total_games_played_l137_137430


namespace sum_of_series_eq_half_l137_137357

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end sum_of_series_eq_half_l137_137357


namespace quadratic_to_vertex_form_l137_137243

theorem quadratic_to_vertex_form:
  ∀ (x : ℝ), (x^2 - 4 * x + 3 = (x - 2)^2 - 1) :=
by
  sorry

end quadratic_to_vertex_form_l137_137243


namespace minimum_teachers_to_cover_all_subjects_l137_137228

/- Define the problem conditions -/
def maths_teachers := 7
def physics_teachers := 6
def chemistry_teachers := 5
def max_subjects_per_teacher := 3

/- The proof statement -/
theorem minimum_teachers_to_cover_all_subjects : 
  (maths_teachers + physics_teachers + chemistry_teachers) / max_subjects_per_teacher = 7 :=
sorry

end minimum_teachers_to_cover_all_subjects_l137_137228


namespace ball_never_returns_l137_137909

-- Define the problem conditions in a mathematically rigorous way

structure BilliardTable :=
  (vertices : Set Point)
  (sides : Set (Point × Point))
  (angle_90 : ∃ (A : vertices), angle_between_sides A = 90)
  (perpendicular_sides : ∀ (A B C : vertices), A ≠ B ∧ B ≠ C ∧ A ≠ C → angle_between_sides B = 90 ∨ angle_between_sides C = 90)

open Set

-- Define the trajectory and reflection properties
def trajectory_path (table : BilliardTable) (start : table.vertices) : sequence (Point × Point) :=
  sorry -- assuming the reflection law is implicitly defined

-- Prove the main theorem
theorem ball_never_returns (table : BilliardTable) (A : table.vertices)
  (Hangle : angle_between_sides A = 90)
  (start_direction : Vector) :
  (∀ trajectory : ℕ → Point, trajectory_path table A = trajectory → (∀ n, trajectory n ≠ A)) :=
sorry

end ball_never_returns_l137_137909


namespace smallest_number_with_unique_digits_sum_32_l137_137379

-- Define the conditions as predicates for clarity.
def all_digits_different (n : ℕ) : Prop :=
  let digits := List.ofString (n.toString)
  digits.nodup

def sum_of_digits (n : ℕ) : ℕ :=
  (List.ofString (n.toString)).foldr (fun d acc => acc + (d.toNat - '0'.toNat)) 0

-- The main statement to prove
theorem smallest_number_with_unique_digits_sum_32 : ∃ n : ℕ, all_digits_different n ∧ sum_of_digits n = 32 ∧ (∀ m : ℕ, all_digits_different m ∧ sum_of_digits m = 32 → n ≤ m) :=
  ∃ n = 26789,
  all_digits_different 26789 ∧ sum_of_digits 26789 = 32 ∧
  (h : all_digits_different m ∧ sum_of_digits m = 32 → 26789 ≤ m)
  sorry -- proof omitted

end smallest_number_with_unique_digits_sum_32_l137_137379


namespace correct_option_is_B_l137_137903

theorem correct_option_is_B :
  (∃ (A B C D : String), A = "√49 = -7" ∧ B = "√((-3)^2) = 3" ∧ C = "-√((-5)^2) = 5" ∧ D = "√81 = ±9" ∧
    (B = "√((-3)^2) = 3")) :=
by
  sorry

end correct_option_is_B_l137_137903


namespace train_length_200_04_l137_137935

-- Define the constants
def speed_kmh : ℝ := 60     -- speed in km/h
def time_seconds : ℕ := 12  -- time in seconds

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def speed_ms : ℝ := (speed_kmh * km_to_m) / hr_to_s

-- Define the length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_200_04 : length_of_train = 200.04 := by
  sorry

end train_length_200_04_l137_137935


namespace function_even_periodic_l137_137924

theorem function_even_periodic (f : ℝ → ℝ) :
  (∀ x : ℝ, f (10 + x) = f (10 - x)) ∧ (∀ x : ℝ, f (5 - x) = f (5 + x)) →
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 10) = f x) :=
by
  sorry

end function_even_periodic_l137_137924


namespace vanya_speed_l137_137737

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l137_137737


namespace Richard_walked_10_miles_third_day_l137_137295

def distance_to_NYC := 70
def day1 := 20
def day2 := (day1 / 2) - 6
def remaining_distance := 36
def day3 := 70 - (day1 + day2 + remaining_distance)

theorem Richard_walked_10_miles_third_day (h : day3 = 10) : day3 = 10 :=
by {
    sorry
}

end Richard_walked_10_miles_third_day_l137_137295


namespace sum_of_roots_l137_137517

theorem sum_of_roots (a b c : ℝ) (h : 3 * x^2 - 7 * x + 2 = 0) : -b / a = 7 / 3 :=
by sorry

end sum_of_roots_l137_137517


namespace find_values_l137_137419

theorem find_values (a b c : ℝ)
  (h1 : 0.005 * a = 0.8)
  (h2 : 0.0025 * b = 0.6)
  (h3 : c = 0.5 * a - 0.1 * b) :
  a = 160 ∧ b = 240 ∧ c = 56 :=
by sorry

end find_values_l137_137419


namespace distinct_diagonals_in_convex_nonagon_l137_137990

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l137_137990


namespace vanya_speed_increased_by_4_l137_137750

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l137_137750


namespace inequality_sum_l137_137393

theorem inequality_sum {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) : a + c > b + d :=
by
  sorry

end inequality_sum_l137_137393


namespace sine_difference_l137_137821

noncomputable def perpendicular_vectors (θ : ℝ) : Prop :=
  let a := (Real.cos θ, -Real.sqrt 3)
  let b := (1, 1 + Real.sin θ)
  a.1 * b.1 + a.2 * b.2 = 0

theorem sine_difference (θ : ℝ) (h : perpendicular_vectors θ) : Real.sin (Real.pi / 6 - θ) = Real.sqrt 3 / 2 :=
by
  sorry

end sine_difference_l137_137821


namespace smallest_n_divisible_l137_137483

theorem smallest_n_divisible {n : ℕ} : 
  (∃ n : ℕ, n > 0 ∧ 18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
    (∀ m : ℕ, m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m)) :=
  sorry

end smallest_n_divisible_l137_137483


namespace angle_sum_in_hexagon_l137_137556

theorem angle_sum_in_hexagon (P Q R s t : ℝ) 
    (hP: P = 40) (hQ: Q = 88) (hR: R = 30)
    (hex_sum: 6 * 180 - 720 = 0): 
    s + t = 312 :=
by
  have hex_interior_sum: 6 * 180 - 720 = 0 := hex_sum
  sorry

end angle_sum_in_hexagon_l137_137556


namespace find_other_number_l137_137878

theorem find_other_number (A B : ℕ) (HCF LCM : ℕ)
  (hA : A = 24)
  (hHCF: (HCF : ℚ) = 16)
  (hLCM: (LCM : ℚ) = 312)
  (hHCF_LCM: HCF * LCM = A * B) : 
  B = 208 :=
by
  sorry

end find_other_number_l137_137878


namespace area_of_square_l137_137066

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l137_137066


namespace solution_set_of_inequalities_l137_137514

theorem solution_set_of_inequalities :
  {x : ℝ | 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9} = {x : ℝ | x > 45 / 26} :=
by sorry

end solution_set_of_inequalities_l137_137514


namespace solve_for_y_l137_137428

variable {y : ℚ}
def algebraic_expression_1 (y : ℚ) : ℚ := 4 * y + 8
def algebraic_expression_2 (y : ℚ) : ℚ := 8 * y - 7

theorem solve_for_y (h : algebraic_expression_1 y = - algebraic_expression_2 y) : y = -1 / 12 :=
by
  sorry

end solve_for_y_l137_137428


namespace vanya_faster_by_4_l137_137734

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l137_137734


namespace vanya_speed_l137_137740

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l137_137740


namespace eccentricity_range_of_ellipse_l137_137259

theorem eccentricity_range_of_ellipse 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (P : ℝ × ℝ) (hP_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci_relation : ∀(θ₁ θ₂ : ℝ), a / (Real.sin θ₁) = c / (Real.sin θ₂)) :
  ∃ (e : ℝ), e = c / a ∧ (Real.sqrt 2 - 1 < e ∧ e < 1) := 
sorry

end eccentricity_range_of_ellipse_l137_137259


namespace solve_for_y_l137_137256

theorem solve_for_y (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 :=
by
  sorry

end solve_for_y_l137_137256


namespace vanya_speed_problem_l137_137746

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l137_137746


namespace sticks_needed_for_4x4_square_largest_square_with_100_sticks_l137_137140

-- Problem a)
def sticks_needed_for_square (n: ℕ) : ℕ := 2 * n * (n + 1)

theorem sticks_needed_for_4x4_square : sticks_needed_for_square 4 = 40 :=
by
  sorry

-- Problem b)
def max_square_side_length (total_sticks : ℕ) : ℕ × ℕ :=
  let n := Nat.sqrt (total_sticks / 2)
  if 2*n*(n+1) <= total_sticks then (n, total_sticks - 2*n*(n+1)) else (n-1, total_sticks - 2*(n-1)*n)

theorem largest_square_with_100_sticks : max_square_side_length 100 = (6, 16) :=
by
  sorry

end sticks_needed_for_4x4_square_largest_square_with_100_sticks_l137_137140


namespace score_of_B_is_correct_l137_137834

theorem score_of_B_is_correct (A B C D E : ℝ)
  (h1 : (A + B + C + D + E) / 5 = 90)
  (h2 : (A + B + C) / 3 = 86)
  (h3 : (B + D + E) / 3 = 95) : 
  B = 93 := 
by 
  sorry

end score_of_B_is_correct_l137_137834


namespace pythagorean_triple_correct_l137_137080

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct :
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 7 9 11 ∧
  ¬ is_pythagorean_triple 6 9 12 ∧
  ¬ is_pythagorean_triple (3/10) (4/10) (5/10) :=
by
  sorry

end pythagorean_triple_correct_l137_137080


namespace trigonometric_identity_l137_137954

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := 
sorry

end trigonometric_identity_l137_137954


namespace product_in_A_l137_137110

def A : Set ℤ := { z | ∃ a b : ℤ, z = a^2 + 4 * a * b + b^2 }

theorem product_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := 
by
  sorry

end product_in_A_l137_137110


namespace value_of_ab_l137_137205

theorem value_of_ab (a b : ℝ) (x : ℝ) 
  (h : ∀ x, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : a * b = 0 :=
sorry

end value_of_ab_l137_137205


namespace population_reduction_l137_137434

theorem population_reduction (initial_population : ℕ) (final_population : ℕ) (left_percentage : ℝ)
    (bombardment_percentage : ℝ) :
    initial_population = 7145 →
    final_population = 4555 →
    left_percentage = 0.75 →
    bombardment_percentage = 100 - 84.96 →
    ∃ (x : ℝ), bombardment_percentage = (100 - x) := 
by
    sorry

end population_reduction_l137_137434


namespace part1_part2_l137_137407

-- Defining the function f
def f (x : ℝ) (a : ℝ) : ℝ := a * abs (x + 1) - abs (x - 1)

-- Part 1: a = 1, finding the solution set of the inequality f(x) < 3/2
theorem part1 (x : ℝ) : f x 1 < 3 / 2 ↔ x < 3 / 4 := 
sorry

-- Part 2: a > 1, and existence of x such that f(x) <= -|2m+1|, finding the range of m
theorem part2 (a : ℝ) (h : 1 < a) (m : ℝ) (x : ℝ) : 
  f x a ≤ -abs (2 * m + 1) → -3 / 2 ≤ m ∧ m ≤ 1 :=
sorry

end part1_part2_l137_137407


namespace pills_first_day_l137_137441

theorem pills_first_day (P : ℕ) 
  (h1 : P + (P + 2) + (P + 4) + (P + 6) + (P + 8) + (P + 10) + (P + 12) = 49) : 
  P = 1 :=
by sorry

end pills_first_day_l137_137441


namespace fraction_of_students_who_say_dislike_but_actually_like_l137_137784

-- Define the conditions
def total_students : ℕ := 100
def like_dancing : ℕ := total_students / 2
def dislike_dancing : ℕ := total_students / 2

def like_dancing_honest : ℕ := (7 * like_dancing) / 10
def like_dancing_dishonest : ℕ := (3 * like_dancing) / 10

def dislike_dancing_honest : ℕ := (4 * dislike_dancing) / 5
def dislike_dancing_dishonest : ℕ := dislike_dancing / 5

-- Define the proof objective
theorem fraction_of_students_who_say_dislike_but_actually_like :
  (like_dancing_dishonest : ℚ) / (total_students - like_dancing_honest - dislike_dancing_dishonest) = 3 / 11 :=
by
  sorry

end fraction_of_students_who_say_dislike_but_actually_like_l137_137784


namespace find_z_coordinate_of_point_on_line_l137_137623

theorem find_z_coordinate_of_point_on_line (x1 y1 z1 x2 y2 z2 x_target : ℝ) 
(h1 : x1 = 1) (h2 : y1 = 3) (h3 : z1 = 2) 
(h4 : x2 = 4) (h5 : y2 = 4) (h6 : z2 = -1)
(h_target : x_target = 7) : 
∃ z_target : ℝ, z_target = -4 := 
by {
  sorry
}

end find_z_coordinate_of_point_on_line_l137_137623


namespace prime_iff_factorial_mod_l137_137719

theorem prime_iff_factorial_mod (p : ℕ) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end prime_iff_factorial_mod_l137_137719


namespace ratio_of_areas_is_five_l137_137158

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end ratio_of_areas_is_five_l137_137158


namespace marks_lost_per_incorrect_sum_l137_137575

variables (marks_per_correct : ℕ) (total_attempts total_marks correct_sums : ℕ)
variable (marks_per_incorrect : ℕ)
variable (incorrect_sums : ℕ)

def calc_marks_per_incorrect_sum : Prop :=
  marks_per_correct = 3 ∧ 
  total_attempts = 30 ∧ 
  total_marks = 50 ∧ 
  correct_sums = 22 ∧ 
  incorrect_sums = total_attempts - correct_sums ∧ 
  (marks_per_correct * correct_sums) - (marks_per_incorrect * incorrect_sums) = total_marks ∧ 
  marks_per_incorrect = 2

theorem marks_lost_per_incorrect_sum : calc_marks_per_incorrect_sum 3 30 50 22 2 (30 - 22) :=
sorry

end marks_lost_per_incorrect_sum_l137_137575


namespace ferry_distance_l137_137222

theorem ferry_distance 
  (x : ℝ)
  (v_w : ℝ := 3)  -- speed of water flow in km/h
  (t_downstream : ℝ := 5)  -- time taken to travel downstream in hours
  (t_upstream : ℝ := 7)  -- time taken to travel upstream in hours
  (eqn : x / t_downstream - v_w = x / t_upstream + v_w) :
  x = 105 :=
sorry

end ferry_distance_l137_137222


namespace Debby_spent_on_yoyo_l137_137236

theorem Debby_spent_on_yoyo 
  (hat_tickets stuffed_animal_tickets total_tickets : ℕ) 
  (h1 : hat_tickets = 2) 
  (h2 : stuffed_animal_tickets = 10) 
  (h3 : total_tickets = 14) 
  : ∃ yoyo_tickets : ℕ, hat_tickets + stuffed_animal_tickets + yoyo_tickets = total_tickets ∧ yoyo_tickets = 2 := 
by 
  sorry

end Debby_spent_on_yoyo_l137_137236


namespace number_of_diagonals_in_nonagon_l137_137971

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l137_137971


namespace bottles_left_after_purchase_l137_137602

def initial_bottles : ℕ := 35
def jason_bottles : ℕ := 5
def harry_bottles : ℕ := 6
def jason_effective_bottles (n : ℕ) : ℕ := n  -- Jason buys 5 bottles
def harry_effective_bottles (n : ℕ) : ℕ := n + 1 -- Harry gets one additional free bottle

theorem bottles_left_after_purchase (j_b h_b i_b : ℕ) (j_effective h_effective : ℕ → ℕ) :
  j_b = 5 → h_b = 6 → i_b = 35 → j_effective j_b = 5 → h_effective h_b = 7 →
  i_b - (j_effective j_b + h_effective h_b) = 23 :=
by
  intros
  sorry

end bottles_left_after_purchase_l137_137602


namespace largest_x_fraction_l137_137949

theorem largest_x_fraction (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 := by
  sorry

end largest_x_fraction_l137_137949


namespace value_of_x_l137_137665

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l137_137665


namespace rectangle_perimeter_l137_137926

variable (x : ℝ) (y : ℝ)

-- Definitions based on conditions
def area_of_rectangle : Prop := x * (x + 5) = 500
def side_length_relation : Prop := y = x + 5

-- The theorem we want to prove
theorem rectangle_perimeter (h_area : area_of_rectangle x) (h_side_length : side_length_relation x y) : 2 * (x + y) = 90 := by
  sorry

end rectangle_perimeter_l137_137926


namespace sum_youngest_oldest_l137_137872

-- Define the ages of the cousins
variables (a1 a2 a3 a4 : ℕ)

-- Conditions given in the problem
def mean_age (a1 a2 a3 a4 : ℕ) : Prop := (a1 + a2 + a3 + a4) / 4 = 8
def median_age (a2 a3 : ℕ) : Prop := (a2 + a3) / 2 = 5

-- Main theorem statement to be proved
theorem sum_youngest_oldest (h_mean : mean_age a1 a2 a3 a4) (h_median : median_age a2 a3) :
  a1 + a4 = 22 :=
sorry

end sum_youngest_oldest_l137_137872


namespace bag_contains_fifteen_balls_l137_137214

theorem bag_contains_fifteen_balls 
  (r b : ℕ) 
  (h1 : r + b = 15) 
  (h2 : (r * (r - 1)) / 210 = 1 / 21) 
  : r = 4 := 
sorry

end bag_contains_fifteen_balls_l137_137214


namespace initial_mean_of_observations_l137_137308

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end initial_mean_of_observations_l137_137308


namespace parallel_lines_value_of_m_l137_137306

theorem parallel_lines_value_of_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + m * y - 2 = 0 = (2 * x + (1 - m) * y + 2 = 0)) : 
  m = 1 / 3 :=
by {
  sorry
}

end parallel_lines_value_of_m_l137_137306


namespace derivative_of_x_log_x_l137_137247

noncomputable def y (x : ℝ) := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (hx : 0 < x) :
  (deriv y x) = Real.log x + 1 :=
sorry

end derivative_of_x_log_x_l137_137247


namespace operation_result_l137_137875

theorem operation_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 12) (h_prod : a * b = 32) 
: (1 / a : ℚ) + (1 / b) = 3 / 8 := by
  sorry

end operation_result_l137_137875


namespace find_same_color_integers_l137_137366

variable (Color : Type) (red blue green yellow : Color)

theorem find_same_color_integers
  (color : ℤ → Color)
  (m n : ℤ)
  (hm : Odd m)
  (hn : Odd n)
  (h_not_zero : m + n ≠ 0) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) :=
sorry

end find_same_color_integers_l137_137366


namespace triangles_from_ten_points_l137_137586

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l137_137586


namespace remaining_students_l137_137031

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l137_137031


namespace total_cost_l137_137925

def copper_pipe_length := 10
def plastic_pipe_length := 15
def copper_pipe_cost_per_meter := 5
def plastic_pipe_cost_per_meter := 3

theorem total_cost (h₁ : copper_pipe_length = 10)
                   (h₂ : plastic_pipe_length = 15)
                   (h₃ : copper_pipe_cost_per_meter = 5)
                   (h₄ : plastic_pipe_cost_per_meter = 3) :
  10 * 5 + 15 * 3 = 95 :=
by sorry

end total_cost_l137_137925


namespace fourth_power_of_cube_of_third_smallest_prime_l137_137895

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p3 := 5 in (p3^3)^4 = 244140625 :=
by
  let p3 := 5
  calc (p3^3)^4 = 244140625 : sorry

end fourth_power_of_cube_of_third_smallest_prime_l137_137895


namespace quadratic_single_solution_positive_n_l137_137808

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l137_137808


namespace incenter_projection_of_tetrahedron_l137_137136

variables {α : Type*} [EuclideanSpace α]

/-- Given a scalene triangle ABC and a point S such that all dihedral angles between each lateral face and the base are equal, and the projection of S onto ABC falls inside the triangle, the point O, the projection of S, is the incenter of triangle ABC. -/
theorem incenter_projection_of_tetrahedron
  {A B C S : α} (hABC_scalene : ¬ ∃ (σ : Perm (Fin 3)), σ •! [A, B, C] = [A, B, C]) 
  (h_dihedral_equal : ∀ (D : Type*), DihedralAngleOfTypeA S A B C = DihedralAngleOfTypeB S A B C) 
  (h_projection_inside : ∃ t1 t2 t3 : ℝ, t1 + t2 + t3 = 1 ∧ 0 < t1 ∧ 0 < t2 ∧ 0 < t3 ∧ S = t1 • A + t2 • B + t3 • C) 
  : is_incenter_of_projection S A B C :=
sorry

end incenter_projection_of_tetrahedron_l137_137136


namespace circumcircle_diameter_of_triangle_l137_137429

theorem circumcircle_diameter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 1) 
  (h_B : B = π/4) 
  (h_area : (1/2) * a * c * Real.sin B = 2) : 
  (2 * b = 5 * Real.sqrt 2) := 
sorry

end circumcircle_diameter_of_triangle_l137_137429


namespace calculate_remaining_area_l137_137557

/-- In a rectangular plot of land ABCD, where AB = 20 meters and BC = 12 meters, 
    a triangular garden ABE is installed where AE = 15 meters and BE intersects AE at a perpendicular angle, 
    the area of the remaining part of the land which is not occupied by the garden is 150 square meters. -/
theorem calculate_remaining_area 
  (AB BC AE : ℝ) 
  (hAB : AB = 20) 
  (hBC : BC = 12) 
  (hAE : AE = 15)
  (h_perpendicular : true) : -- BE ⊥ AE implying right triangle ABE
  ∃ area_remaining : ℝ, area_remaining = 150 :=
by
  sorry

end calculate_remaining_area_l137_137557


namespace cole_trip_time_l137_137490

theorem cole_trip_time 
  (D : ℕ) -- The distance D from home to work
  (T_total : ℕ) -- The total round trip time in hours
  (S1 S2 : ℕ) -- The average speeds (S1, S2) in km/h
  (h1 : S1 = 80) -- The average speed from home to work
  (h2 : S2 = 120) -- The average speed from work to home
  (h3 : T_total = 2) -- The total round trip time is 2 hours
  : (D : ℝ) / 80 + (D : ℝ) / 120 = 2 →
    (T_work : ℝ) = (D : ℝ) / 80 →
    (T_work * 60) = 72 := 
by {
  sorry
}

end cole_trip_time_l137_137490


namespace arithmetic_sequence_fourth_term_l137_137426

theorem arithmetic_sequence_fourth_term (b d : ℝ) (h : 2 * b + 2 * d = 10) : b + d = 5 :=
by
  sorry

end arithmetic_sequence_fourth_term_l137_137426


namespace range_of_a_l137_137109

-- Given function
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- Derivative of the function
def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

-- Discriminant of the derivative
def discriminant (a : ℝ) : ℝ := 4*a^2 - 12*(a + 6)

-- Proof that the range of 'a' is 'a < -3 or a > 6' for f(x) to have both maximum and minimum values
theorem range_of_a (a : ℝ) : discriminant a > 0 ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l137_137109


namespace redistribution_l137_137004

/-
Given:
- b = (12 / 13) * a
- c = (2 / 3) * b
- Person C will contribute 9 dollars based on the amount each person spent

Prove:
- Person C gives 6 dollars to Person A.
- Person C gives 3 dollars to Person B.
-/

theorem redistribution (a b c : ℝ) (h1 : b = (12 / 13) * a) (h2 : c = (2 / 3) * b) : 
  ∃ (x y : ℝ), x + y = 9 ∧ x = 6 ∧ y = 3 :=
by
  sorry

end redistribution_l137_137004


namespace find_x_average_l137_137645

theorem find_x_average :
  ∃ x : ℝ, (x + 8 + (7 * x - 3) + (3 * x + 10) + (-x + 6)) / 4 = 5 * x - 4 ∧ x = 3.7 :=
  by
  use 3.7
  sorry

end find_x_average_l137_137645


namespace towels_per_pack_l137_137887

open Nat

-- Define the given conditions
def packs : Nat := 9
def total_towels : Nat := 27

-- Define the property to prove
theorem towels_per_pack : total_towels / packs = 3 := by
  sorry

end towels_per_pack_l137_137887


namespace range_of_m_l137_137410

noncomputable def setA := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
noncomputable def setB (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem range_of_m (m : ℝ) : (setA ∪ setB m = setA) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l137_137410


namespace soldiers_movement_l137_137762

theorem soldiers_movement (n : ℕ) 
  (initial_positions : Fin (n+3) × Fin (n+1) → Prop) 
  (moves_to_adjacent : ∀ p : Fin (n+3) × Fin (n+1), initial_positions p → initial_positions (p.1 + 1, p.2) ∨ initial_positions (p.1 - 1, p.2) ∨ initial_positions (p.1, p.2 + 1) ∨ initial_positions (p.1, p.2 - 1))
  (final_positions : Fin (n+1) × Fin (n+3) → Prop) : Even n := 
sorry

end soldiers_movement_l137_137762


namespace area_of_square_l137_137067

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l137_137067


namespace original_phone_number_eq_l137_137905

theorem original_phone_number_eq :
  ∃ (a b c d e f : ℕ), 
    (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 282500) ∧
    (1000000 * 2 + 100000 * a + 10000 * 8 + 1000 * b + 100 * c + 10 * d + e = 81 * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) ∧
    (0 ≤ a ∧ a ≤ 9) ∧
    (0 ≤ b ∧ b ≤ 9) ∧
    (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧
    (0 ≤ e ∧ e ≤ 9) ∧
    (0 ≤ f ∧ f ≤ 9) :=
sorry

end original_phone_number_eq_l137_137905


namespace solve_for_a_l137_137423

-- Definitions: Real number a, Imaginary unit i, complex number.
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem solve_for_a :
  ∀ (a : ℝ) (i : ℂ),
    i = Complex.I →
    is_purely_imaginary ( (3 * i / (1 + 2 * i)) * (1 - (a / 3) * i) ) →
    a = -6 :=
by
  sorry

end solve_for_a_l137_137423


namespace deer_meat_distribution_l137_137285

theorem deer_meat_distribution (a d : ℕ) (H1 : a = 100) :
  ∀ (Dafu Bugeng Zanbao Shangzao Gongshe : ℕ),
    Dafu = a - 2 * d →
    Bugeng = a - d →
    Zanbao = a →
    Shangzao = a + d →
    Gongshe = a + 2 * d →
    Dafu + Bugeng + Zanbao + Shangzao + Gongshe = 500 →
    Bugeng + Zanbao + Shangzao = 300 :=
by
  intros Dafu Bugeng Zanbao Shangzao Gongshe hDafu hBugeng hZanbao hShangzao hGongshe hSum
  sorry

end deer_meat_distribution_l137_137285


namespace tickets_sold_l137_137230

theorem tickets_sold (student_tickets non_student_tickets student_ticket_price non_student_ticket_price total_revenue : ℕ)
  (h1 : student_ticket_price = 5)
  (h2 : non_student_ticket_price = 8)
  (h3 : total_revenue = 930)
  (h4 : student_tickets = 90)
  (h5 : non_student_tickets = 60) :
  student_tickets + non_student_tickets = 150 := 
by 
  sorry

end tickets_sold_l137_137230


namespace vector_parallel_example_l137_137409

theorem vector_parallel_example 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (ha : a = (2, 1)) 
  (hb : b = (4, 2))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  3 • a + 2 • b = (14, 7) := 
by
  sorry

end vector_parallel_example_l137_137409


namespace num_diagonals_convex_nonagon_l137_137983

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l137_137983


namespace solve_equation_1_solve_equation_2_l137_137454

theorem solve_equation_1 :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = 9 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x ^ 2 - 4 * x - 12 = 0 ↔ (x = 6 ∨ x = -2) :=
by
  sorry

end solve_equation_1_solve_equation_2_l137_137454


namespace quadratic_inequality_solution_l137_137520

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 6*x - 16 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 8} := by
  sorry

end quadratic_inequality_solution_l137_137520


namespace alice_minimum_speed_exceed_l137_137209

-- Define the conditions

def distance_ab : ℕ := 30  -- Distance from city A to city B is 30 miles
def speed_bob : ℕ := 40    -- Bob's constant speed is 40 miles per hour
def bob_travel_time := distance_ab / speed_bob  -- Bob's travel time in hours
def alice_travel_time := bob_travel_time - (1 / 2)  -- Alice leaves 0.5 hours after Bob

-- Theorem stating the minimum speed Alice must exceed
theorem alice_minimum_speed_exceed : ∃ v : Real, v > 60 ∧ distance_ab / alice_travel_time ≤ v := sorry

end alice_minimum_speed_exceed_l137_137209


namespace smallest_n_satisfying_conditions_l137_137351

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l137_137351


namespace discount_percentage_is_25_l137_137684

-- Define the conditions
def cost_of_coffee : ℕ := 6
def cost_of_cheesecake : ℕ := 10
def final_price_with_discount : ℕ := 12

-- Define the total cost without discount
def total_cost_without_discount : ℕ := cost_of_coffee + cost_of_cheesecake

-- Define the discount amount
def discount_amount : ℕ := total_cost_without_discount - final_price_with_discount

-- Define the percentage discount
def percentage_discount : ℕ := (discount_amount * 100) / total_cost_without_discount

-- Proof Statement
theorem discount_percentage_is_25 : percentage_discount = 25 := by
  sorry

end discount_percentage_is_25_l137_137684


namespace number_of_triangles_l137_137584

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l137_137584


namespace evaluate_triangle_l137_137944

def triangle_op (a b : Int) : Int :=
  a * b - a - b + 1

theorem evaluate_triangle :
  triangle_op (-3) 4 = -12 :=
by
  sorry

end evaluate_triangle_l137_137944


namespace least_stamps_l137_137079

theorem least_stamps (s t : ℕ) (h : 5 * s + 7 * t = 48) : s + t = 8 :=
by sorry

end least_stamps_l137_137079


namespace wendy_total_glasses_l137_137201

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l137_137201


namespace count_congruent_to_3_mod_8_in_300_l137_137271

theorem count_congruent_to_3_mod_8_in_300 : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ n % 8 = 3}.card = 38 := 
by
  sorry

end count_congruent_to_3_mod_8_in_300_l137_137271


namespace goal_amount_is_correct_l137_137726

def earnings_three_families : ℕ := 3 * 10
def earnings_fifteen_families : ℕ := 15 * 5
def total_earned : ℕ := earnings_three_families + earnings_fifteen_families
def goal_amount : ℕ := total_earned + 45

theorem goal_amount_is_correct : goal_amount = 150 :=
by
  -- We are aware of the proof steps but they are not required here
  sorry

end goal_amount_is_correct_l137_137726


namespace regular_pyramid_cannot_be_hexagonal_l137_137211

theorem regular_pyramid_cannot_be_hexagonal (n : ℕ) (h₁ : n = 6) (base_edge_length slant_height : ℝ) 
  (reg_pyramid : base_edge_length = slant_height) : false :=
by
  sorry

end regular_pyramid_cannot_be_hexagonal_l137_137211


namespace Wendy_total_glasses_l137_137199

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l137_137199


namespace probability_individual_selected_l137_137481

theorem probability_individual_selected :
  ∀ (N M : ℕ) (m : ℕ), N = 100 → M = 5 → (m < N) →
  (probability_of_selecting_m : ℝ) =
  (1 / N * M) :=
by
  intros N M m hN hM hm
  sorry

end probability_individual_selected_l137_137481


namespace find_b_15_l137_137653

variable {a : ℕ → ℤ} (b : ℕ → ℤ) (S : ℕ → ℤ)

/-- An arithmetic sequence where S_n is the sum of the first n terms, with S_9 = -18 and S_13 = -52
   and a geometric sequence where b_5 = a_5 and b_7 = a_7. -/
theorem find_b_15 
  (h1 : S 9 = -18) 
  (h2 : S 13 = -52) 
  (h3 : b 5 = a 5) 
  (h4 : b 7 = a 7) 
  : b 15 = -64 := 
sorry

end find_b_15_l137_137653


namespace prob_X_geq_six_minus_m_l137_137701

open MeasureTheory ProbabilityTheory

-- Define the random variable X and its properties
noncomputable def X : MeasureTheory.ProbMeasure ℝ := 
MeasureTheory.ℙ (MeasureTheory.Normal 3 (σ^2))

-- Conditions of the problem
variable (σ : ℝ) (m : ℝ)
hypothesis (h : ∀ m : ℝ, MeasureTheory.ℙ (X > m) = 0.3)

-- Lean statement asserting the final proof goal
theorem prob_X_geq_six_minus_m : MeasureTheory.ℙ (X ≥ 6 - m) = 0.7 := by
  sorry

end prob_X_geq_six_minus_m_l137_137701


namespace initial_violet_balloons_l137_137559

-- Let's define the given conditions
def red_balloons : ℕ := 4
def violet_balloons_lost : ℕ := 3
def violet_balloons_now : ℕ := 4

-- Define the statement to prove
theorem initial_violet_balloons :
  (violet_balloons_now + violet_balloons_lost) = 7 :=
by
  sorry

end initial_violet_balloons_l137_137559


namespace trains_speed_ratio_l137_137035

-- Define the conditions
variables (V1 V2 L1 L2 : ℝ)
axiom time1 : L1 = 27 * V1
axiom time2 : L2 = 17 * V2
axiom timeTogether : L1 + L2 = 22 * (V1 + V2)

-- The theorem to prove the ratio of the speeds
theorem trains_speed_ratio : V1 / V2 = 7.8 :=
sorry

end trains_speed_ratio_l137_137035


namespace number_of_integers_congruent_7_mod_9_lessthan_1000_l137_137549

theorem number_of_integers_congruent_7_mod_9_lessthan_1000 : 
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → 7 + 9 * n < 1000 → k + 1 = 111 :=
by
  sorry

end number_of_integers_congruent_7_mod_9_lessthan_1000_l137_137549


namespace polygon_interior_exterior_angles_l137_137467

theorem polygon_interior_exterior_angles (n : ℕ) :
  (n - 2) * 180 = 360 + 720 → n = 8 := 
by {
  sorry
}

end polygon_interior_exterior_angles_l137_137467


namespace least_positive_x_l137_137610

theorem least_positive_x (x : ℕ) (h : (2 * x + 45)^2 % 43 = 0) : x = 42 :=
  sorry

end least_positive_x_l137_137610


namespace vanya_faster_speed_l137_137756

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l137_137756


namespace interval_contains_n_l137_137104

theorem interval_contains_n (n : ℕ) (h1 : n < 1000) (h2 : n ∣ 999) (h3 : n + 6 ∣ 99) : 1 ≤ n ∧ n ≤ 250 := 
sorry

end interval_contains_n_l137_137104


namespace leila_spending_l137_137698

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l137_137698


namespace find_x_l137_137421

/-- Given real numbers x and y,
    under the condition that (y^3 + 2y - 1)/(y^3 + 2y - 3) = x/(x - 1),
    we want to prove that x = (y^3 + 2y - 1)/2 -/
theorem find_x (x y : ℝ) (h1 : y^3 + 2*y - 3 ≠ 0) (h2 : y^3 + 2*y - 1 ≠ 0)
  (h : x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3)) :
  x = (y^3 + 2*y - 1) / 2 :=
by sorry

end find_x_l137_137421


namespace delta_value_l137_137275

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l137_137275


namespace simplify_expression_l137_137576

theorem simplify_expression (x : ℝ) :
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) = x^4 - 1 :=
  by 
    sorry

end simplify_expression_l137_137576


namespace smallest_n_satisfying_conditions_l137_137350

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l137_137350


namespace eggs_per_chicken_l137_137786

theorem eggs_per_chicken (num_chickens : ℕ) (eggs_per_carton : ℕ) (num_cartons : ℕ) (total_eggs : ℕ) 
  (h1 : num_chickens = 20) (h2 : eggs_per_carton = 12) (h3 : num_cartons = 10) (h4 : total_eggs = num_cartons * eggs_per_carton) : 
  total_eggs / num_chickens = 6 :=
by
  sorry

end eggs_per_chicken_l137_137786


namespace area_of_intersection_is_zero_l137_137893

-- Define the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 16
def circle2 (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the theorem to prove
theorem area_of_intersection_is_zero : 
  ∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    x1 = x2 ∧ y1 = -y2 → 
    0 = 0 :=
by
  sorry -- proof goes here

end area_of_intersection_is_zero_l137_137893


namespace choose_four_socks_at_least_one_blue_l137_137864

-- There are six socks, each of different colors: {blue, brown, black, red, purple, green}
def socks : Finset String := {"blue", "brown", "black", "red", "purple", "green"}

-- We need to choose 4 socks such that at least one is blue.
def num_ways_to_choose_four_with_one_blue : ℕ :=
  (socks.erase "blue").card.choose 3

theorem choose_four_socks_at_least_one_blue :
  num_ways_to_choose_four_with_one_blue = 10 :=
by
  rw [num_ways_to_choose_four_with_one_blue, Finset.card_erase_of_mem]
  { rw Finset.card, decide, sorry }
  { exact Finset.mem_univ "blue" }
  sorry

end choose_four_socks_at_least_one_blue_l137_137864


namespace max_sqrt_expression_l137_137150

open Real

theorem max_sqrt_expression (x y z : ℝ) (h_sum : x + y + z = 3)
  (hx : x ≥ -1) (hy : y ≥ -(2/3)) (hz : z ≥ -2) :
  sqrt (3 * x + 3) + sqrt (3 * y + 2) + sqrt (3 * z + 6) ≤ 2 * sqrt 15 := by
  sorry

end max_sqrt_expression_l137_137150


namespace veranda_area_correct_l137_137871

noncomputable def area_veranda (length_room : ℝ) (width_room : ℝ) (width_veranda : ℝ) (radius_obstacle : ℝ) : ℝ :=
  let total_length := length_room + 2 * width_veranda
  let total_width := width_room + 2 * width_veranda
  let area_total := total_length * total_width
  let area_room := length_room * width_room
  let area_circle := Real.pi * radius_obstacle^2
  area_total - area_room - area_circle

theorem veranda_area_correct :
  area_veranda 18 12 2 3 = 107.726 :=
by sorry

end veranda_area_correct_l137_137871


namespace perimeter_greater_than_diagonals_l137_137500

namespace InscribedQuadrilateral

def is_convex_quadrilateral (AB BC CD DA AC BD: ℝ) : Prop :=
  -- Conditions for a convex quadrilateral (simple check)
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ AC > 0 ∧ BD > 0

def is_inscribed_in_circle (AB BC CD DA AC BD: ℝ) (r: ℝ) : Prop :=
  -- Check if quadrilateral is inscribed in a circle of radius 1
  r = 1

theorem perimeter_greater_than_diagonals 
  (AB BC CD DA AC BD: ℝ) 
  (r: ℝ)
  (h1 : is_convex_quadrilateral AB BC CD DA AC BD) 
  (h2 : is_inscribed_in_circle AB BC CD DA AC BD r) :
  0 < (AB + BC + CD + DA) - (AC + BD) ∧ (AB + BC + CD + DA) - (AC + BD) < 2 :=
by
  sorry 

end InscribedQuadrilateral

end perimeter_greater_than_diagonals_l137_137500


namespace square_difference_l137_137414

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) : (x - y)^2 = 57 :=
by
  sorry

end square_difference_l137_137414


namespace second_time_apart_l137_137966

theorem second_time_apart 
  (glen_speed : ℕ) 
  (hannah_speed : ℕ)
  (initial_distance : ℕ) 
  (initial_time : ℕ)
  (relative_speed : ℕ)
  (hours_later : ℕ) :
  glen_speed = 37 →
  hannah_speed = 15 →
  initial_distance = 130 →
  initial_time = 6 →
  relative_speed = glen_speed + hannah_speed →
  hours_later = initial_distance / relative_speed →
  initial_time + hours_later = 8 + 30 / 60 :=
by
  intros
  sorry

end second_time_apart_l137_137966


namespace inequality_solution_l137_137787

theorem inequality_solution (x : ℝ) : 1 - (2 * x - 2) / 5 < (3 - 4 * x) / 2 → x < 1 / 16 := by
  sorry

end inequality_solution_l137_137787


namespace xavier_yvonne_not_zelda_prob_l137_137614

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end xavier_yvonne_not_zelda_prob_l137_137614


namespace total_sum_value_l137_137660

open Finset

def M : Finset ℕ := filter (λ x, 1 ≤ x ∧ x ≤ 10) (range 11)

def sum_transformed_elements (A : Finset ℕ) : ℤ :=
  A.sum (λ k, (-1) ^ k * k)

def total_sum : ℤ :=
  M.powerset.erase ∅
    .sum sum_transformed_elements

theorem total_sum_value :
  total_sum = 2560 := by {
  -- proof here
  sorry
}

end total_sum_value_l137_137660


namespace triangle_inequality_inequality_l137_137108

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 > 0 := 
by
  sorry

end triangle_inequality_inequality_l137_137108


namespace M_inter_N_l137_137819

def M : Set ℝ := { y | y > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem M_inter_N : M ∩ N = { z | 1 < z ∧ z < 2 } :=
by 
  sorry

end M_inter_N_l137_137819


namespace total_weight_l137_137145

def w1 : ℝ := 9.91
def w2 : ℝ := 4.11

theorem total_weight : w1 + w2 = 14.02 := by 
  sorry

end total_weight_l137_137145


namespace ratio_simplification_l137_137463

theorem ratio_simplification (a b c : ℕ) (h₁ : ∃ (a b c : ℕ), (rat.mk (a * (real.sqrt b)) c) = (real.sqrt (50 / 98)) ∧ (a = 5) ∧ (b = 1) ∧ (c = 7)) : a + b + c = 13 := by
  sorry

end ratio_simplification_l137_137463


namespace smallest_possible_fourth_number_l137_137344

theorem smallest_possible_fourth_number 
  (a b : ℕ) 
  (h1 : 21 + 34 + 65 = 120)
  (h2 : 1 * (21 + 34 + 65 + 10 * a + b) = 4 * (2 + 1 + 3 + 4 + 6 + 5 + a + b)) :
  10 * a + b = 12 := 
sorry

end smallest_possible_fourth_number_l137_137344


namespace expression_equals_thirteen_l137_137321

-- Define the expression
def expression : ℤ :=
    8 + 15 / 3 - 4 * 2 + Nat.pow 2 3

-- State the theorem that proves the value of the expression
theorem expression_equals_thirteen : expression = 13 :=
by
  sorry

end expression_equals_thirteen_l137_137321


namespace B_alone_finishes_in_21_days_l137_137906

theorem B_alone_finishes_in_21_days (W_A W_B : ℝ) (h1 : W_A = 0.5 * W_B) (h2 : W_A + W_B = 1 / 14) : W_B = 1 / 21 :=
by sorry

end B_alone_finishes_in_21_days_l137_137906


namespace vanya_faster_by_4_l137_137735

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l137_137735


namespace jack_total_books_is_541_l137_137287

-- Define the number of books in each section
def american_books : ℕ := 6 * 34
def british_books : ℕ := 8 * 29
def world_books : ℕ := 5 * 21

-- Define the total number of books based on the given sections
def total_books : ℕ := american_books + british_books + world_books

-- Prove that the total number of books is 541
theorem jack_total_books_is_541 : total_books = 541 :=
by
  sorry

end jack_total_books_is_541_l137_137287


namespace horner_method_evaluation_l137_137319

def f (x : ℝ) := 0.5 * x^5 + 4 * x^4 + 0 * x^3 - 3 * x^2 + x - 1

theorem horner_method_evaluation : f 3 = 1 :=
by
  -- Placeholder for the proof
  sorry

end horner_method_evaluation_l137_137319


namespace train_length_is_correct_l137_137930

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l137_137930


namespace barrel_tank_ratio_l137_137772

theorem barrel_tank_ratio
  (B T : ℝ)
  (h1 : (3 / 4) * B = (5 / 8) * T) :
  B / T = 5 / 6 :=
sorry

end barrel_tank_ratio_l137_137772


namespace problem1_problem2_l137_137261

open Real

noncomputable def alpha (hα : 0 < α ∧ α < π / 3) :=
  α

noncomputable def vec_a (hα : 0 < α ∧ α < π / 3) :=
  (sqrt 6 * sin (alpha hα), sqrt 2)

noncomputable def vec_b (hα : 0 < α ∧ α < π / 3) :=
  (1, cos (alpha hα) - sqrt 6 / 2)

theorem problem1 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  tan (alpha hα + π / 6) = sqrt 15 / 5 :=
sorry

theorem problem2 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  cos (2 * alpha hα + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
sorry

end problem1_problem2_l137_137261


namespace interest_difference_correct_l137_137020

-- Define the basic parameters and constants
def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10
def interest1 : ℝ := principal * rate * time1
def interest2 : ℝ := principal * rate * time2
def difference : ℝ := 143.998

-- Theorem statement: The difference between the interests is approximately Rs. 143.998
theorem interest_difference_correct :
  interest2 - interest1 = difference := sorry

end interest_difference_correct_l137_137020


namespace probability_different_grades_l137_137219

theorem probability_different_grades (A B : Type) [Fintype A] [Fintype B] (ha : Fintype.card A = 2) (hb : Fintype.card B = 2) :
  (∃ (s : Finset (A ⊕ B)), s.card = 2) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) = 4) →
  (Fintype.card (Finset (A ⊕ B)).card-choose 2 = 6) →
  (Fintype.card (Finset (A ⊕ B)).filter (λ s, (∃ (a : A) (b : B), s = {sum.inl a, sum.inr b})) /
     Fintype.card (Finset (A ⊕ B)).card-choose 2 = 2 / 3) :=
sorry

end probability_different_grades_l137_137219


namespace ant_weight_statement_l137_137502

variable (R : ℝ) -- Rupert's weight
variable (A : ℝ) -- Antoinette's weight
variable (C : ℝ) -- Charles's weight

-- Conditions
def condition1 : Prop := A = 2 * R - 7
def condition2 : Prop := C = (A + R) / 2 + 5
def condition3 : Prop := A + R + C = 145

-- Question: Prove Antoinette's weight
def ant_weight_proof : Prop :=
  ∃ R A C, condition1 R A ∧ condition2 R A C ∧ condition3 R A C ∧ A = 79

theorem ant_weight_statement : ant_weight_proof :=
sorry

end ant_weight_statement_l137_137502


namespace triangles_from_ten_points_l137_137587

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l137_137587


namespace cards_in_center_pile_l137_137917

/-- Represents the number of cards in each pile initially. -/
def initial_cards (x : ℕ) : Prop := x ≥ 2

/-- Represents the state of the piles after step 2. -/
def step2 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 2 ∧ right = x

/-- Represents the state of the piles after step 3. -/
def step3 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 3 ∧ right = x - 1

/-- Represents the state of the piles after step 4. -/
def step4 (x : ℕ) (left center : ℕ) : Prop :=
  left = 2 * x - 4 ∧ center = 5

/-- Prove that after performing all steps, the number of cards in the center pile is 5. -/
theorem cards_in_center_pile (x : ℕ) :
  initial_cards x →
  (∃ l₁ c₁ r₁, step2 x l₁ c₁ r₁) →
  (∃ l₂ c₂ r₂, step3 x l₂ c₂ r₂) →
  (∃ l₃ c₃, step4 x l₃ c₃) →
  ∃ (center_final : ℕ), center_final = 5 :=
by
  sorry

end cards_in_center_pile_l137_137917


namespace coefficients_divisible_by_5_l137_137016

theorem coefficients_divisible_by_5 
  (a b c d : ℤ) 
  (h : ∀ x : ℤ, 5 ∣ (a * x^3 + b * x^2 + c * x + d)) : 
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d := 
by {
  sorry
}

end coefficients_divisible_by_5_l137_137016


namespace solve_for_x_l137_137416

theorem solve_for_x (x : ℝ) (h : Real.exp (Real.log 7) = 9 * x + 2) : x = 5 / 9 :=
by {
    -- Proof needs to be filled here
    sorry
}

end solve_for_x_l137_137416


namespace part1_part2_l137_137817

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + a * x + 1
noncomputable def g (x : ℝ) := x * Real.exp x

-- Problem Part 1: Prove the range of a for which f(x) has two extreme points
theorem part1 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = f x₂ a) ↔ (0 < a ∧ a < (1 / Real.exp 1)) :=
sorry

-- Problem Part 2: Prove the range of a for which f(x) ≥ 2sin(x) for x ≥ 0
theorem part2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ 2 * Real.sin x) ↔ (2 ≤ a) :=
sorry

end part1_part2_l137_137817


namespace years_passed_l137_137922

-- Let PV be the present value of the machine, FV be the final value of the machine, r be the depletion rate, and t be the time in years.
def PV : ℝ := 900
def FV : ℝ := 729
def r : ℝ := 0.10

-- The formula for exponential decay is FV = PV * (1 - r)^t.
-- Given FV = 729, PV = 900, and r = 0.10, we want to prove that t = 2.

theorem years_passed (t : ℕ) : FV = PV * (1 - r)^t → t = 2 := 
by 
  intro h
  sorry

end years_passed_l137_137922


namespace range_m_single_solution_l137_137424

-- Statement expressing the conditions and conclusion.
theorem range_m_single_solution :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0 → ∃! x, 0 ≤ x ∧ x ≤ 2) ↔ m ∈ (Set.Ico (-2 : ℝ) 0) ∪ {2} := 
sorry

end range_m_single_solution_l137_137424


namespace total_balloons_170_l137_137854

variable (minutes_Max : ℕ) (rate_Max : ℕ) (minutes_Zach : ℕ) (rate_Zach : ℕ) (popped : ℕ)

def balloons_filled_Max := minutes_Max * rate_Max
def balloons_filled_Zach := minutes_Zach * rate_Zach
def total_filled_balloons := balloons_filled_Max + balloons_filled_Zach - popped

theorem total_balloons_170 
  (h1 : minutes_Max = 30) 
  (h2 : rate_Max = 2) 
  (h3 : minutes_Zach = 40) 
  (h4 : rate_Zach = 3) 
  (h5 : popped = 10) : 
  total_filled_balloons minutes_Max rate_Max minutes_Zach rate_Zach popped = 170 := by
  unfold total_filled_balloons
  unfold balloons_filled_Max
  unfold balloons_filled_Zach
  sorry

end total_balloons_170_l137_137854


namespace carl_spends_108_dollars_l137_137360

theorem carl_spends_108_dollars
    (index_cards_per_student : ℕ := 10)
    (periods_per_day : ℕ := 6)
    (students_per_class : ℕ := 30)
    (cost_per_pack : ℕ := 3)
    (cards_per_pack : ℕ := 50) :
  let total_index_cards := index_cards_per_student * students_per_class * periods_per_day in
  let total_packs := total_index_cards / cards_per_pack in
  let total_cost := total_packs * cost_per_pack in
  total_cost = 108 := 
by
  sorry

end carl_spends_108_dollars_l137_137360


namespace fourth_power_of_third_smallest_prime_cube_l137_137896

def third_smallest_prime : ℕ := 5

def cube_of_third_smallest_prime : ℕ := third_smallest_prime ^ 3

def fourth_power_of_cube (n : ℕ) : ℕ := n ^ 4

theorem fourth_power_of_third_smallest_prime_cube :
  fourth_power_of_cube (third_smallest_prime ^ 3) = 244140625 := by
  calc
    (third_smallest_prime ^ 3) ^ 4
      = (5 ^ 3) ^ 4 : by rfl
    ... = 5 ^ (3 * 4) : by rw pow_mul
    ... = 5 ^ 12 : by norm_num
    ... = 244140625 : by norm_num

end fourth_power_of_third_smallest_prime_cube_l137_137896


namespace total_students_l137_137083

theorem total_students (T : ℝ) (h : 0.50 * T = 440) : T = 880 := 
by {
  sorry
}

end total_students_l137_137083


namespace hogwarts_school_students_l137_137544

def total_students_at_school (participants boys : ℕ) (boy_participants girl_non_participants : ℕ) : Prop :=
  participants = 246 ∧ boys = 255 ∧ boy_participants = girl_non_participants + 11 → (boys + (participants - boy_participants + girl_non_participants)) = 490

theorem hogwarts_school_students : total_students_at_school 246 255 (boy_participants) girl_non_participants := 
 sorry

end hogwarts_school_students_l137_137544


namespace factorization_result_l137_137596

theorem factorization_result (a b : ℤ) (h : (16:ℚ) * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) : a + 2 * b = -23 := by
  sorry

end factorization_result_l137_137596


namespace area_of_triangle_BQW_l137_137687

theorem area_of_triangle_BQW (AZ WC AB : ℝ) (h_trap_area : ℝ) (h_eq : AZ = WC) (AZ_val : AZ = 8) (AB_val : AB = 16) (trap_area_val : h_trap_area = 160) : 
  ∃ (BQW_area: ℝ), BQW_area = 48 :=
by
  let h_2 := 2 * h_trap_area / (AZ + AB)
  let h := AZ + h_2
  let BZW_area := h_trap_area - (1 / 2) * AZ * AB
  let BQW_area := 1 / 2 * BZW_area
  have AZ_eq : AZ = 8 := AZ_val
  have AB_eq : AB = 16 := AB_val
  have trap_area_eq : h_trap_area = 160 := trap_area_val
  let h_2_val : ℝ := 10 -- Calculated from h_2 = 2 * 160 / 32
  let h_val : ℝ := AZ + h_2_val -- full height
  let BZW_area_val : ℝ := 96 -- BZW area from 160 - 64
  let BQW_area_val : ℝ := 48 -- Half of BZW
  exact ⟨48, by sorry⟩ -- To complete the theorem

end area_of_triangle_BQW_l137_137687


namespace gcd_polynomial_even_multiple_of_97_l137_137398

theorem gcd_polynomial_even_multiple_of_97 (b : ℤ) (k : ℤ) (h_b : b = 2 * 97 * k) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 :=
by
  sorry

end gcd_polynomial_even_multiple_of_97_l137_137398


namespace number_of_triangles_l137_137583

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l137_137583


namespace value_of_x_l137_137666

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l137_137666


namespace solve_inequality_l137_137266

theorem solve_inequality 
  (k_0 k b m n : ℝ)
  (hM1 : -1 = k_0 * m + b) (hM2 : -1 = k^2 / m)
  (hN1 : 2 = k_0 * n + b) (hN2 : 2 = k^2 / n) :
  {x : ℝ | x^2 > k_0 * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
  sorry

end solve_inequality_l137_137266


namespace count_congruent_3_mod_8_l137_137270

theorem count_congruent_3_mod_8 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 300) :
  ∃ k : ℕ, (1 ≤ 8 * k + 3 ∧ 8 * k + 3 ≤ 300) ∧ n = 38 :=
by
  sorry

end count_congruent_3_mod_8_l137_137270


namespace river_length_l137_137723

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l137_137723


namespace proportion_of_capacity_filled_l137_137770

noncomputable def milk_proportion_8cup_bottle : ℚ := 16 / 3
noncomputable def total_milk := 8

theorem proportion_of_capacity_filled :
  ∃ p : ℚ, (8 * p = milk_proportion_8cup_bottle) ∧ (4 * p = total_milk - milk_proportion_8cup_bottle) ∧ (p = 2 / 3) :=
by
  sorry

end proportion_of_capacity_filled_l137_137770


namespace kalebs_restaurant_bill_l137_137234

theorem kalebs_restaurant_bill :
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  (adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost) = 60 := 
by
  let adults := 6
  let children := 2
  let adult_meal_cost := 6
  let children_meal_cost := 4
  let soda_cost := 2
  calc 
    adults * adult_meal_cost + children * children_meal_cost + (adults + children) * soda_cost 
      = 6 * 6 + 2 * 4 + (6 + 2) * 2 : by rfl
    ... = 36 + 8 + 16 : by rfl
    ... = 60 : by rfl

end kalebs_restaurant_bill_l137_137234


namespace fraction_traditionalists_l137_137621

theorem fraction_traditionalists {P T : ℕ} (h1 : ∀ (i : ℕ), i < 5 → T = P / 15) (h2 : T = P / 15) :
  (5 * T : ℚ) / (P + 5 * T : ℚ) = 1 / 4 :=
by
  sorry

end fraction_traditionalists_l137_137621


namespace detectives_sons_ages_l137_137605

theorem detectives_sons_ages (x y : ℕ) (h1 : x < 5) (h2 : y < 5) (h3 : x * y = 4) (h4 : (∃ x₁ y₁ : ℕ, (x₁ * y₁ = 4 ∧ x₁ < 5 ∧ y₁ < 5) ∧ x₁ ≠ x ∨ y₁ ≠ y)) :
  (x = 1 ∨ x = 4) ∧ (y = 1 ∨ y = 4) :=
by
  sorry

end detectives_sons_ages_l137_137605


namespace parabola_translation_l137_137891

theorem parabola_translation :
  ∀ x : ℝ, (x^2 + 3) = ((x + 1)^2 + 3) :=
by
  skip -- proof is not needed; this is just the statement according to the instruction
  sorry

end parabola_translation_l137_137891


namespace segment_length_tangent_circles_l137_137320

theorem segment_length_tangent_circles
  (r1 r2 : ℝ)
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (h3 : 7 - 4 * Real.sqrt 3 ≤ r1 / r2)
  (h4 : r1 / r2 ≤ 7 + 4 * Real.sqrt 3)
  :
  ∃ d : ℝ, d^2 = (1 / 12) * (14 * r1 * r2 - r1^2 - r2^2) :=
sorry

end segment_length_tangent_circles_l137_137320


namespace number_of_positive_integers_count_positive_integers_dividing_10n_l137_137387

theorem number_of_positive_integers (n : ℕ) :
  (1 + 2 + ... + n) ∣ (10 * n) → n > 0 → n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 9 ∨ n = 19 :=
by
  intro h1 h2
  sorry

theorem count_positive_integers_dividing_10n :
  {n : ℕ | (1 + 2 + ... + n) ∣ (10 * n) ∧ n > 0}.to_finset.card = 5 :=
by
  sorry

end number_of_positive_integers_count_positive_integers_dividing_10n_l137_137387


namespace no_even_is_prime_equiv_l137_137874

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l137_137874


namespace smallest_number_with_unique_digits_sum_32_l137_137377

theorem smallest_number_with_unique_digits_sum_32 :
  ∃ n : ℕ, (∀ i j, i ≠ j → (n.digits 10).nth i ≠ (n.digits 10).nth j) ∧ (n.digits 10).sum = 32 ∧
  (∀ m : ℕ, (∀ i j, i ≠ j → (m.digits 10).nth i ≠ (m.digits 10).nth j) ∧ (m.digits 10).sum = 32 → n ≤ m) → n = 26789 :=
begin
  sorry
end

end smallest_number_with_unique_digits_sum_32_l137_137377


namespace b4_minus_a4_l137_137143

-- Given quadratic equation and specified root, prove the difference of fourth powers.
theorem b4_minus_a4 (a b : ℝ) (h_root : (a^2 - b^2)^2 = x) (h_equation : x^2 + 4 * a^2 * b^2 * x = 4) : b^4 - a^4 = 2 ∨ b^4 - a^4 = -2 :=
sorry

end b4_minus_a4_l137_137143


namespace avg_score_is_94_l137_137792

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l137_137792


namespace clock_rings_in_january_l137_137352

theorem clock_rings_in_january :
  ∀ (days_in_january hours_per_day ring_interval : ℕ)
  (first_ring_time : ℕ) (january_first_hour : ℕ), 
  days_in_january = 31 →
  hours_per_day = 24 →
  ring_interval = 7 →
  january_first_hour = 2 →
  first_ring_time = 30 →
  (days_in_january * hours_per_day) / ring_interval + 1 = 107 := by
  intros days_in_january hours_per_day ring_interval first_ring_time january_first_hour
  sorry

end clock_rings_in_january_l137_137352


namespace pears_weight_l137_137777

theorem pears_weight (x : ℕ) (h : 2 * x + 50 = 250) : x = 100 :=
sorry

end pears_weight_l137_137777


namespace toothpick_removal_l137_137521

/-- Given 40 toothpicks used to create 10 squares and 15 triangles, with each square formed by 
4 toothpicks and each triangle formed by 3 toothpicks, prove that removing 10 toothpicks is 
sufficient to ensure no squares or triangles remain. -/
theorem toothpick_removal (n : ℕ) (squares triangles : ℕ) (sq_toothpicks tri_toothpicks : ℕ) 
    (total_toothpicks : ℕ) (remove_toothpicks : ℕ) 
    (h1 : n = 40) 
    (h2 : squares = 10) 
    (h3 : triangles = 15) 
    (h4 : sq_toothpicks = 4) 
    (h5 : tri_toothpicks = 3) 
    (h6 : total_toothpicks = n) 
    (h7 : remove_toothpicks = 10) 
    (h8 : (squares * sq_toothpicks + triangles * tri_toothpicks) = total_toothpicks) :
  remove_toothpicks = 10 :=
by
  sorry

end toothpick_removal_l137_137521


namespace day_crew_fraction_l137_137938

theorem day_crew_fraction (D W : ℕ) (h1 : ∀ n, n = D / 4) (h2 : ∀ w, w = 4 * W / 5) :
  (D * W) / ((D * W) + ((D / 4) * (4 * W / 5))) = 5 / 6 :=
by 
  sorry

end day_crew_fraction_l137_137938


namespace train_length_is_correct_l137_137931

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l137_137931


namespace rectangle_perimeter_l137_137304

variable (L W : ℝ)

-- Conditions
def width := 70
def length := (7 / 5) * width

-- Perimeter calculation and proof goal
def perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_perimeter : perimeter (length) (width) = 336 := by
  sorry

end rectangle_perimeter_l137_137304


namespace min_value_expression_l137_137763

theorem min_value_expression (a b c : ℝ) (h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (4 / c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end min_value_expression_l137_137763


namespace square_area_correct_l137_137069

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l137_137069


namespace remainder_three_l137_137417

theorem remainder_three (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 3 = 1 :=
sorry

end remainder_three_l137_137417


namespace proposition_truthfulness_l137_137166

-- Definitions
def is_positive (n : ℕ) : Prop := n > 0
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Original proposition
def original_prop (n : ℕ) : Prop := is_positive n ∧ is_even n → ¬ is_prime n

-- Converse proposition
def converse_prop (n : ℕ) : Prop := ¬ is_prime n → is_positive n ∧ is_even n

-- Inverse proposition
def inverse_prop (n : ℕ) : Prop := ¬ (is_positive n ∧ is_even n) → is_prime n

-- Contrapositive proposition
def contrapositive_prop (n : ℕ) : Prop := is_prime n → ¬ (is_positive n ∧ is_even n)

-- Proof problem statement
theorem proposition_truthfulness (n : ℕ) :
  (original_prop n = False) ∧
  (converse_prop n = False) ∧
  (inverse_prop n = False) ∧
  (contrapositive_prop n = True) :=
sorry

end proposition_truthfulness_l137_137166


namespace max_value_of_y_is_2_l137_137291

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem max_value_of_y_is_2 (a : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 2 * a * x + (a - 3)) = (3 * x^2 - 2 * a * x + (a - 3))) : 
  ∃ x : ℝ, f a x = 2 :=
sorry

end max_value_of_y_is_2_l137_137291


namespace triangle_height_l137_137866

theorem triangle_height (base : ℝ) (height : ℝ) (area : ℝ)
  (h_base : base = 8) (h_area : area = 16) (h_area_formula : area = (base * height) / 2) :
  height = 4 :=
by
  sorry

end triangle_height_l137_137866


namespace arithmetic_sequence_l137_137115

open Nat

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_S : S (2 * n + 1) - S (2 * n - 1) + S 2 = 24) 
  (h_S_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2 ∧ d = a 2 - a 1) : 
  a (n + 1) = 6 :=
by
  sorry

end arithmetic_sequence_l137_137115


namespace least_positive_integer_y_l137_137554

theorem least_positive_integer_y (x k y: ℤ) (h1: 24 * x + k * y = 4) (h2: ∃ x: ℤ, ∃ y: ℤ, 24 * x + k * y = 4) : y = 4 :=
sorry

end least_positive_integer_y_l137_137554


namespace solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l137_137206

open Real

theorem solve_diff_eq_for_k_ne_zero (k : ℝ) (h : k ≠ 0) (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x * (f x + g x) ^ k)
  (hg : ∀ x, deriv g x = f x * (f x + g x) ^ k)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) + (1 - k * x) ^ (1 / k)) ∧ g x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) - (1 - k * x) ^ (1 / k))) :=
sorry

theorem solve_diff_eq_for_k_eq_zero (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x)
  (hg : ∀ x, deriv g x = f x)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = cosh x ∧ g x = sinh x) :=
sorry

end solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l137_137206


namespace find_expression_l137_137867

theorem find_expression (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 84) (h2 : a = 32) : E = 80 :=
by
  -- Proof to be filled in here
  sorry

end find_expression_l137_137867


namespace solve_for_a_l137_137532

theorem solve_for_a (a x : ℤ) (h : x + 2 * a = -3) (hx : x = 1) : a = -2 := by
  sorry

end solve_for_a_l137_137532


namespace total_sonnets_written_l137_137547

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l137_137547


namespace strawberries_weight_l137_137702

theorem strawberries_weight (marco_weight dad_increase : ℕ) (h_marco: marco_weight = 30) (h_diff: marco_weight = dad_increase + 13) : marco_weight + (marco_weight - 13) = 47 :=
by
  sorry

end strawberries_weight_l137_137702


namespace education_expenses_l137_137346

noncomputable def totalSalary (savings : ℝ) (savingsPercentage : ℝ) : ℝ :=
  savings / savingsPercentage

def totalExpenses (rent milk groceries petrol misc : ℝ) : ℝ :=
  rent + milk + groceries + petrol + misc

def amountSpentOnEducation (totalSalary totalExpenses savings : ℝ) : ℝ :=
  totalSalary - (totalExpenses + savings)

theorem education_expenses :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let petrol := 2000
  let misc := 700
  let savings := 1800
  let savingsPercentage := 0.10
  amountSpentOnEducation (totalSalary savings savingsPercentage) 
                          (totalExpenses rent milk groceries petrol misc) 
                          savings = 2500 :=
by
  sorry

end education_expenses_l137_137346


namespace carrie_money_left_l137_137633

/-- Carrie was given $91. She bought a sweater for $24, 
    a T-shirt for $6, a pair of shoes for $11,
    and a pair of jeans originally costing $30 with a 25% discount. 
    Prove that she has $27.50 left. -/
theorem carrie_money_left :
  let init_money := 91
  let sweater := 24
  let t_shirt := 6
  let shoes := 11
  let jeans := 30
  let discount := 25 / 100
  let jeans_discounted_price := jeans * (1 - discount)
  let total_cost := sweater + t_shirt + shoes + jeans_discounted_price
  let money_left := init_money - total_cost
  money_left = 27.50 :=
by
  intros
  sorry

end carrie_money_left_l137_137633


namespace vanya_faster_speed_l137_137761

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l137_137761


namespace part_a_part_b_part_c_l137_137212

-- Part (a)
theorem part_a (m : ℤ) : (m^2 + 10) % (m - 2) = 0 ∧ (m^2 + 10) % (m + 4) = 0 ↔ m = -5 ∨ m = 9 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) : ∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 :=
sorry

-- Part (c)
theorem part_c (n : ℤ) : ∃ N : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m < N :=
sorry

end part_a_part_b_part_c_l137_137212


namespace find_pairs_l137_137102

theorem find_pairs :
  ∀ (x y : ℕ), 0 < x → 0 < y → 7 ^ x - 3 * 2 ^ y = 1 → (x, y) = (1, 1) ∨ (x, y) = (2, 4) :=
by
  intros x y hx hy h
  -- Proof would go here
  sorry

end find_pairs_l137_137102


namespace base_conversion_l137_137868

theorem base_conversion (C D : ℕ) (h₁ : 0 ≤ C ∧ C < 8) (h₂ : 0 ≤ D ∧ D < 5) (h₃ : 7 * C = 4 * D) :
  8 * C + D = 0 := by
  sorry

end base_conversion_l137_137868


namespace makeup_set_cost_l137_137106

theorem makeup_set_cost (initial : ℕ) (gift : ℕ) (needed : ℕ) (total_cost : ℕ) :
  initial = 35 → gift = 20 → needed = 10 → total_cost = initial + gift + needed → total_cost = 65 :=
by
  intros h_init h_gift h_needed h_cost
  sorry

end makeup_set_cost_l137_137106


namespace smallest_of_three_consecutive_l137_137879

theorem smallest_of_three_consecutive (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by
  sorry

end smallest_of_three_consecutive_l137_137879


namespace find_moles_of_NaOH_l137_137103

-- Define the conditions
def reaction (NaOH HClO4 NaClO4 H2O : ℕ) : Prop :=
  NaOH = HClO4 ∧ NaClO4 = HClO4 ∧ H2O = 1

def moles_of_HClO4 := 3
def moles_of_NaClO4 := 3

-- Problem statement
theorem find_moles_of_NaOH : ∃ (NaOH : ℕ), NaOH = moles_of_HClO4 ∧ moles_of_NaClO4 = 3 ∧ NaOH = 3 :=
by sorry

end find_moles_of_NaOH_l137_137103


namespace distance_between_cityA_and_cityB_l137_137487

noncomputable def distanceBetweenCities (time_to_cityB time_from_cityB saved_time round_trip_speed: ℝ) : ℝ :=
  let total_distance := 90 * (time_to_cityB + saved_time + time_from_cityB + saved_time) / 2
  total_distance / 2

theorem distance_between_cityA_and_cityB 
  (time_to_cityB : ℝ)
  (time_from_cityB : ℝ)
  (saved_time : ℝ)
  (round_trip_speed : ℝ)
  (distance : ℝ)
  (h1 : time_to_cityB = 6)
  (h2 : time_from_cityB = 4.5)
  (h3 : saved_time = 0.5)
  (h4 : round_trip_speed = 90)
  (h5 : distanceBetweenCities time_to_cityB time_from_cityB saved_time round_trip_speed = distance)
: distance = 427.5 := by
  sorry

end distance_between_cityA_and_cityB_l137_137487


namespace problem_inequality_l137_137861

theorem problem_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n :=
sorry

end problem_inequality_l137_137861


namespace largest_k_l137_137369

theorem largest_k (k n : ℕ) (h1 : 2^11 = (k * (2 * n + k + 1)) / 2) : k = 1 := sorry

end largest_k_l137_137369


namespace production_bottles_l137_137167

-- Definitions from the problem conditions
def machines_production_rate (machines : ℕ) (rate : ℕ) : ℕ := rate / machines
def total_production (machines rate minutes : ℕ) : ℕ := machines * rate * minutes

-- Theorem to prove the solution
theorem production_bottles :
  machines_production_rate 6 300 = 50 →
  total_production 10 50 4 = 2000 :=
by
  intro h
  have : 10 * 50 * 4 = 2000 := by norm_num
  exact this

end production_bottles_l137_137167


namespace mart_income_percentage_j_l137_137703

variables (J T M : ℝ)

-- condition: Tim's income is 40 percent less than Juan's income
def tims_income := T = 0.60 * J

-- condition: Mart's income is 40 percent more than Tim's income
def marts_income := M = 1.40 * T

-- goal: Prove that Mart's income is 84 percent of Juan's income
theorem mart_income_percentage_j (J : ℝ) (T : ℝ) (M : ℝ)
  (h1 : T = 0.60 * J) 
  (h2 : M = 1.40 * T) : 
  M = 0.84 * J := 
sorry

end mart_income_percentage_j_l137_137703


namespace distinct_diagonals_in_convex_nonagon_l137_137988

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l137_137988


namespace greatest_integer_gcd_four_l137_137036

theorem greatest_integer_gcd_four {n : ℕ} (h1 : n < 150) (h2 : Nat.gcd n 12 = 4) : n <= 148 :=
by {
  sorry
}

end greatest_integer_gcd_four_l137_137036


namespace prob_two_more_heads_than_tails_eq_210_1024_l137_137278

-- Let P be the probability of getting exactly two more heads than tails when flipping 10 coins.
def P (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2^n : ℚ)

theorem prob_two_more_heads_than_tails_eq_210_1024 :
  P 10 6 = 210 / 1024 :=
by
  -- The steps leading to the proof are omitted and hence skipped
  sorry

end prob_two_more_heads_than_tails_eq_210_1024_l137_137278


namespace vanya_speed_l137_137739

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l137_137739


namespace nonagon_diagonals_count_l137_137992

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l137_137992


namespace problem_statement_l137_137453

variable (a b c : ℝ)
variable (x : ℝ)

theorem problem_statement (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1 :=
by
  intros x hx
  let f := fun x => a * x^2 - b * x + c
  let g := fun x => (a + b) * x^2 + c
  have h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |f x| < 1 := h
  sorry

end problem_statement_l137_137453


namespace parabola_p_q_r_sum_l137_137367

noncomputable def parabola_vertex (p q r : ℝ) (x_vertex y_vertex : ℝ) :=
  ∀ (x : ℝ), p * (x - x_vertex) ^ 2 + y_vertex = p * x ^ 2 + q * x + r

theorem parabola_p_q_r_sum
  (p q r : ℝ)
  (vertex_x vertex_y : ℝ)
  (hx_vertex : vertex_x = 3)
  (hy_vertex : vertex_y = 10)
  (h_vertex : parabola_vertex p q r vertex_x vertex_y)
  (h_contains : p * (0 - 3) ^ 2 + 10 = 7) :
  p + q + r = 23 / 3 :=
sorry

end parabola_p_q_r_sum_l137_137367


namespace intervals_of_monotonic_increase_max_area_acute_triangle_l137_137568

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (sin x, (sqrt 3 / 2) * (sin x - cos x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, sin x + cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

-- Problem 1: Proving the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  ∀ x₁ x₂ : ℝ, (k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12) → f x₁ ≤ f x₂ :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC
theorem max_area_acute_triangle (A : ℝ) (a b c : ℝ) :
  (f A = 1 / 2) → (a = sqrt 2) →
  ∀ S : ℝ, S ≤ (1 + sqrt 2) / 2 :=
sorry

end intervals_of_monotonic_increase_max_area_acute_triangle_l137_137568


namespace root_expression_value_l137_137850

noncomputable def value_of_expression (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) : ℝ :=
  sorry

theorem root_expression_value (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) :
  value_of_expression p q r h1 h2 h3 = 367 / 183 :=
sorry

end root_expression_value_l137_137850


namespace find_least_N_exists_l137_137096

theorem find_least_N_exists (N : ℕ) :
  (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
    N = (a₁ + 2) * (b₁ + 2) * (c₁ + 2) - 8 ∧ 
    N + 1 = (a₂ + 2) * (b₂ + 2) * (c₂ + 2) - 8) ∧
  N = 55 := 
sorry

end find_least_N_exists_l137_137096


namespace linear_regression_equation_l137_137440

-- Given conditions
variables (x y : ℝ)
variable (corr_pos : x ≠ 0 → y / x > 0)
noncomputable def x_mean : ℝ := 2.4
noncomputable def y_mean : ℝ := 3.2

-- Regression line equation
theorem linear_regression_equation :
  (y = 0.5 * x + 2) ∧ (∀ x' y', (x' = x_mean ∧ y' = y_mean) → (y' = 0.5 * x' + 2)) :=
by
  sorry

end linear_regression_equation_l137_137440


namespace sin_alpha_plus_half_pi_l137_137811

theorem sin_alpha_plus_half_pi (α : ℝ) 
  (h1 : Real.tan (α - Real.pi) = 3 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.sin (α + Real.pi / 2) = -4 / 5 :=
by
  -- Placeholder for the proof
  sorry

end sin_alpha_plus_half_pi_l137_137811


namespace sum_series_l137_137356

theorem sum_series : (∑ k in (Finset.range ∞), (3 ^ (2 ^ k)) / ((3 ^ 2) ^ (2 ^ k) - 1)) = 1 / 2 :=
by sorry

end sum_series_l137_137356


namespace number_of_triangles_l137_137591

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l137_137591


namespace river_length_l137_137721

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l137_137721


namespace farmer_eggs_per_week_l137_137335

theorem farmer_eggs_per_week (E : ℝ) (chickens : ℝ) (price_per_dozen : ℝ) (total_revenue : ℝ) (num_weeks : ℝ) (total_chickens : ℝ) (dozen : ℝ) 
    (H1 : total_chickens = 46)
    (H2 : price_per_dozen = 3)
    (H3 : total_revenue = 552)
    (H4 : num_weeks = 8)
    (H5 : dozen = 12)
    (H6 : chickens = 46)
    : E = 6 :=
by
  sorry

end farmer_eggs_per_week_l137_137335


namespace odd_integers_count_between_fractions_l137_137663

theorem odd_integers_count_between_fractions :
  ∃ (count : ℕ), count = 14 ∧
  ∀ (n : ℤ), (25:ℚ)/3 < (n : ℚ) ∧ (n : ℚ) < (73 : ℚ)/2 ∧ (n % 2 = 1) :=
sorry

end odd_integers_count_between_fractions_l137_137663


namespace lucy_fish_moved_l137_137569

theorem lucy_fish_moved (original_count moved_count remaining_count : ℝ)
  (h1: original_count = 212.0)
  (h2: remaining_count = 144.0) :
  moved_count = original_count - remaining_count :=
by sorry

end lucy_fish_moved_l137_137569


namespace cube_volume_ratio_l137_137482

theorem cube_volume_ratio
  (a : ℕ) (b : ℕ)
  (h₁ : a = 5)
  (h₂ : b = 24)
  : (a^3 : ℚ) / (b^3 : ℚ) = 125 / 13824 := by
  sorry

end cube_volume_ratio_l137_137482


namespace max_hot_dogs_with_300_dollars_l137_137237

def num_hot_dogs (dollars : ℕ) 
  (cost_8 : ℚ) (count_8 : ℕ) 
  (cost_20 : ℚ) (count_20 : ℕ)
  (cost_250 : ℚ) (count_250 : ℕ) : ℕ :=
  sorry

theorem max_hot_dogs_with_300_dollars : 
  num_hot_dogs 300 1.55 8 3.05 20 22.95 250 = 3258 :=
sorry

end max_hot_dogs_with_300_dollars_l137_137237


namespace smallest_number_with_unique_digits_sum_32_l137_137376

theorem smallest_number_with_unique_digits_sum_32 :
  ∃ n : ℕ, (∀ i j, i ≠ j → (n.digits 10).nth i ≠ (n.digits 10).nth j) ∧ (n.digits 10).sum = 32 ∧
  (∀ m : ℕ, (∀ i j, i ≠ j → (m.digits 10).nth i ≠ (m.digits 10).nth j) ∧ (m.digits 10).sum = 32 → n ≤ m) → n = 26789 :=
begin
  sorry
end

end smallest_number_with_unique_digits_sum_32_l137_137376


namespace fraction_of_yard_occupied_l137_137927

theorem fraction_of_yard_occupied (yard_length yard_width triangle_leg length_of_short_parallel length_of_long_parallel : ℕ) 
  (h1 : yard_length = 30) 
  (h2 : yard_width = 18) 
  (h3 : triangle_leg = 6) 
  (h4 : length_of_short_parallel = yard_width) 
  (h5 : length_of_long_parallel = yard_length) 
  : (2 * (1/2 * (triangle_leg * triangle_leg))) / (yard_length * yard_width) = 1 / 15 := by 
sorrry

end fraction_of_yard_occupied_l137_137927


namespace Wendy_total_glasses_l137_137200

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l137_137200


namespace intersection_A_B_is_1_and_2_l137_137397

def A : Set ℝ := {x | x ^ 2 - 3 * x - 4 < 0}
def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B_is_1_and_2 : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_is_1_and_2_l137_137397


namespace number_of_diagonals_in_nonagon_l137_137970

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l137_137970


namespace value_of_k_l137_137459

theorem value_of_k (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p/q = 3/2 ∧ p + q = -10 ∧ p * q = k) → k = 24 :=
by 
  sorry

end value_of_k_l137_137459


namespace total_donation_l137_137629

-- Define the conditions in the problem
def Barbara_stuffed_animals : ℕ := 9
def Trish_stuffed_animals : ℕ := 2 * Barbara_stuffed_animals
def Barbara_sale_price : ℝ := 2
def Trish_sale_price : ℝ := 1.5

-- Define the goal as a theorem to be proven
theorem total_donation : Barbara_sale_price * Barbara_stuffed_animals + Trish_sale_price * Trish_stuffed_animals = 45 := by
  sorry

end total_donation_l137_137629


namespace bungee_cord_extension_l137_137341

variables (m g H k h L₀ T_max : ℝ)
  (mass_nonzero : m ≠ 0)
  (gravity_positive : g > 0)
  (H_positive : H > 0)
  (k_positive : k > 0)
  (L₀_nonnegative : L₀ ≥ 0)
  (T_max_eq : T_max = 4 * m * g)
  (L_eq : L₀ + h = H)
  (hooke_eq : T_max = k * h)

theorem bungee_cord_extension :
  h = H / 2 := sorry

end bungee_cord_extension_l137_137341


namespace hike_duration_l137_137130

def initial_water := 11
def final_water := 2
def leak_rate := 1
def water_drunk := 6

theorem hike_duration (time_hours : ℕ) :
  initial_water - final_water = water_drunk + time_hours * leak_rate →
  time_hours = 3 :=
by intro h; sorry

end hike_duration_l137_137130


namespace kylie_total_apples_l137_137847

-- Define the conditions as given in the problem.
def first_hour_apples : ℕ := 66
def second_hour_apples : ℕ := 2 * first_hour_apples
def third_hour_apples : ℕ := first_hour_apples / 3

-- Define the mathematical proof problem.
theorem kylie_total_apples : 
  first_hour_apples + second_hour_apples + third_hour_apples = 220 :=
by
  -- Proof goes here
  sorry

end kylie_total_apples_l137_137847


namespace geometric_progression_x_value_l137_137249

noncomputable def geometric_progression_solution (x : ℝ) : Prop :=
  let a := -30 + x
  let b := -10 + x
  let c := 40 + x
  b^2 = a * c

theorem geometric_progression_x_value :
  ∃ x : ℝ, geometric_progression_solution x ∧ x = 130 / 3 :=
by
  sorry

end geometric_progression_x_value_l137_137249


namespace smallest_number_with_sum_32_and_distinct_digits_l137_137374

def is_distinct (l : List Nat) : Prop := l.Nodup

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem smallest_number_with_sum_32_and_distinct_digits :
  ∀ n : Nat, is_distinct (Nat.digits 10 n) ∧ sum_of_digits n = 32 → 26789 ≤ n :=
by
  intro n
  intro h
  sorry

end smallest_number_with_sum_32_and_distinct_digits_l137_137374


namespace min_length_QR_l137_137049

theorem min_length_QR (PQ PR SR QS QR : ℕ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  QR > PR - PQ ∧ QR > QS - SR ↔ QR = 16 :=
by
  sorry

end min_length_QR_l137_137049


namespace all_numbers_are_2007_l137_137941

noncomputable def sequence_five_numbers (a b c d e : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ 
  (a = 2007 ∨ b = 2007 ∨ c = 2007 ∨ d = 2007 ∨ e = 2007) ∧ 
  (∃ r1, b = r1 * a ∧ c = r1 * b ∧ d = r1 * c ∧ e = r1 * d) ∧
  (∃ r2, a = r2 * b ∧ c = r2 * a ∧ d = r2 * c ∧ e = r2 * d) ∧
  (∃ r3, a = r3 * c ∧ b = r3 * a ∧ d = r3 * b ∧ e = r3 * d) ∧
  (∃ r4, a = r4 * d ∧ b = r4 * a ∧ c = r4 * b ∧ e = r4 * d) ∧
  (∃ r5, a = r5 * e ∧ b = r5 * a ∧ c = r5 * b ∧ d = r5 * c)

theorem all_numbers_are_2007 (a b c d e : ℤ) 
  (h : sequence_five_numbers a b c d e) : 
  a = 2007 ∧ b = 2007 ∧ c = 2007 ∧ d = 2007 ∧ e = 2007 :=
sorry

end all_numbers_are_2007_l137_137941


namespace average_monthly_growth_rate_l137_137774

-- Define the initial and final production quantities
def initial_production : ℝ := 100
def final_production : ℝ := 144

-- Define the average monthly growth rate
def avg_monthly_growth_rate (x : ℝ) : Prop :=
  initial_production * (1 + x)^2 = final_production

-- Statement of the problem to be verified
theorem average_monthly_growth_rate :
  ∃ x : ℝ, avg_monthly_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_growth_rate_l137_137774


namespace vanya_faster_speed_l137_137759

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l137_137759


namespace Seth_bought_20_cartons_of_ice_cream_l137_137296

-- Definitions from conditions
def ice_cream_cost_per_carton : ℕ := 6
def yogurt_cost_per_carton : ℕ := 1
def num_yogurt_cartons : ℕ := 2
def extra_amount_spent_on_ice_cream : ℕ := 118

-- Let x be the number of cartons of ice cream Seth bought
def num_ice_cream_cartons (x : ℕ) : Prop :=
  ice_cream_cost_per_carton * x = num_yogurt_cartons * yogurt_cost_per_carton + extra_amount_spent_on_ice_cream

-- The proof goal
theorem Seth_bought_20_cartons_of_ice_cream : num_ice_cream_cartons 20 :=
by
  unfold num_ice_cream_cartons
  unfold ice_cream_cost_per_carton yogurt_cost_per_carton num_yogurt_cartons extra_amount_spent_on_ice_cream
  sorry

end Seth_bought_20_cartons_of_ice_cream_l137_137296


namespace bouquets_sold_on_Monday_l137_137223

theorem bouquets_sold_on_Monday
  (tuesday_three_times_monday : ∀ (x : ℕ), bouquets_sold_Tuesday = 3 * x)
  (wednesday_third_of_tuesday : ∀ (bouquets_sold_Tuesday : ℕ), bouquets_sold_Wednesday = bouquets_sold_Tuesday / 3)
  (total_bouquets : bouquets_sold_Monday + bouquets_sold_Tuesday + bouquets_sold_Wednesday = 60)
  : bouquets_sold_Monday = 12 := 
sorry

end bouquets_sold_on_Monday_l137_137223


namespace sequence_formula_l137_137524

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 4 else 3 * n + 1

theorem sequence_formula (n : ℕ) :
  (sequence n = 3 * n + 1) :=
by
  sorry

end sequence_formula_l137_137524


namespace nonagon_diagonals_count_l137_137994

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l137_137994


namespace calc_g_inv_sum_l137_137443

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x * x

noncomputable def g_inv (y : ℝ) : ℝ := 
  if y = -4 then 4
  else if y = 0 then 3
  else if y = 4 then -1
  else 0

theorem calc_g_inv_sum : g_inv (-4) + g_inv 0 + g_inv 4 = 6 :=
by
  sorry

end calc_g_inv_sum_l137_137443


namespace mangoes_harvested_l137_137153

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) (total_mangoes_distributed : ℕ) (total_mangoes : ℕ) :
  neighbors = 8 ∧ mangoes_per_neighbor = 35 ∧ total_mangoes_distributed = neighbors * mangoes_per_neighbor ∧ total_mangoes = 2 * total_mangoes_distributed →
  total_mangoes = 560 :=
by {
  sorry
}

end mangoes_harvested_l137_137153


namespace range_of_m_n_l137_137535

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * Real.exp x + x^2 + n * x

theorem range_of_m_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0) ∧ (∀ x : ℝ, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end range_of_m_n_l137_137535


namespace missed_questions_proof_l137_137043

def num_missed_questions : ℕ := 180

theorem missed_questions_proof (F : ℕ) (h1 : 5 * F + F = 216) : F = 36 ∧ 5 * F = num_missed_questions :=
by {
  sorry
}

end missed_questions_proof_l137_137043


namespace harkamal_total_amount_l137_137822

-- Conditions
def cost_grapes : ℝ := 8 * 80
def cost_mangoes : ℝ := 9 * 55
def cost_apples_before_discount : ℝ := 6 * 120
def cost_oranges : ℝ := 4 * 75
def discount_apples : ℝ := 0.10 * cost_apples_before_discount
def cost_apples_after_discount : ℝ := cost_apples_before_discount - discount_apples

def total_cost_before_tax : ℝ :=
  cost_grapes + cost_mangoes + cost_apples_after_discount + cost_oranges

def sales_tax : ℝ := 0.05 * total_cost_before_tax

def total_amount_paid : ℝ := total_cost_before_tax + sales_tax

-- Question translated into a Lean statement
theorem harkamal_total_amount:
  total_amount_paid = 2187.15 := 
sorry

end harkamal_total_amount_l137_137822


namespace find_angle_l137_137131

theorem find_angle (a b c d e : ℝ) (sum_of_hexagon_angles : ℝ) (h_sum : a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 ∧ sum_of_hexagon_angles = 720) : 
  ∃ P : ℝ, a + b + c + d + e + P = sum_of_hexagon_angles ∧ P = 100 :=
by
  sorry

end find_angle_l137_137131


namespace clock_overlap_24_hours_l137_137272

theorem clock_overlap_24_hours (hour_rotations : ℕ) (minute_rotations : ℕ) 
  (h_hour_rotations: hour_rotations = 2) 
  (h_minute_rotations: minute_rotations = 24) : 
  ∃ (overlaps : ℕ), overlaps = 22 := 
by 
  sorry

end clock_overlap_24_hours_l137_137272


namespace percentage_cities_in_range_l137_137624

-- Definitions of percentages as given conditions
def percentage_cities_between_50k_200k : ℕ := 40
def percentage_cities_below_50k : ℕ := 35
def percentage_cities_above_200k : ℕ := 25

-- Statement of the problem
theorem percentage_cities_in_range :
  percentage_cities_between_50k_200k = 40 := 
by
  sorry

end percentage_cities_in_range_l137_137624


namespace snowflake_stamps_count_l137_137087

theorem snowflake_stamps_count (S : ℕ) (truck_stamps : ℕ) (rose_stamps : ℕ) :
  truck_stamps = S + 9 →
  rose_stamps = S + 9 - 13 →
  S + truck_stamps + rose_stamps = 38 →
  S = 11 :=
by
  intros h1 h2 h3
  sorry

end snowflake_stamps_count_l137_137087


namespace expr_value_l137_137611

-- Define the constants
def w : ℤ := 3
def x : ℤ := -2
def y : ℤ := 1
def z : ℤ := 4

-- Define the expression
def expr : ℤ := (w^2 * x^2 * y * z) - (w * x^2 * y * z^2) + (w * y^3 * z^2) - (w * y^2 * x * z^4)

-- Statement to be proved
theorem expr_value : expr = 1536 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end expr_value_l137_137611


namespace mark_boxes_sold_l137_137853

theorem mark_boxes_sold (n : ℕ) (M A : ℕ) (h1 : A = n - 2) (h2 : M + A < n) (h3 :  1 ≤ M) (h4 : 1 ≤ A) (hn : n = 12) : M = 1 :=
by
  sorry

end mark_boxes_sold_l137_137853


namespace rearrange_terms_not_adjacent_l137_137803

/-- 
Given the expansion of (x^(1/2) + x^(1/3))^12,
prove that the number of ways to rearrange the terms
with positive integer powers of x so they are not adjacent is A 10 10 * A 3 11
-/
theorem rearrange_terms_not_adjacent : 
  let expansion := (x: ℝ)^(1/2) + (x: ℝ)^(1/3)
  let positive_integer_terms := {T | ∃ r, r ∈ {0, 6, 12} ∧ T = binomial 12 r * x ^ (6 - r/6)}
  A 10 10 * A 3 11 = 
    number_of_ways_to_rearrange_terms_not_adjacent expansion positive_integer_terms :=
sorry

end rearrange_terms_not_adjacent_l137_137803


namespace fitted_ball_volume_l137_137053

noncomputable def volume_of_fitted_ball (d_ball d_h1 r_h1 d_h2 r_h2 : ℝ) : ℝ :=
  let r_ball := d_ball / 2
  let v_ball := (4 / 3) * Real.pi * r_ball^3
  let r_hole1 := r_h1
  let r_hole2 := r_h2
  let v_hole1 := Real.pi * r_hole1^2 * d_h1
  let v_hole2 := Real.pi * r_hole2^2 * d_h2
  v_ball - 2 * v_hole1 - v_hole2

theorem fitted_ball_volume :
  volume_of_fitted_ball 24 10 (3 / 2) 10 2 = 2219 * Real.pi :=
by
  sorry

end fitted_ball_volume_l137_137053


namespace vector_parallel_l137_137541

theorem vector_parallel (x y : ℝ) (a b : ℝ × ℝ × ℝ) (h_parallel : a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ k : ℝ, a = k • b) : x + y = 6 :=
by sorry

end vector_parallel_l137_137541


namespace Yura_catches_up_in_five_minutes_l137_137613

-- Define the speeds and distances
variables (v_Lena v_Yura d_Lena d_Yura : ℝ)
-- Assume v_Yura = 2 * v_Lena (Yura is twice as fast)
axiom h1 : v_Yura = 2 * v_Lena 
-- Assume Lena walks for 5 minutes before Yura starts
axiom h2 : d_Lena = v_Lena * 5
-- Assume they walk at constant speeds
noncomputable def t_to_catch_up := 10 / 2 -- time Yura takes to catch up Lena

-- Define the proof problem
theorem Yura_catches_up_in_five_minutes :
    t_to_catch_up = 5 :=
by
    sorry

end Yura_catches_up_in_five_minutes_l137_137613


namespace john_runs_with_dog_for_half_hour_l137_137695

noncomputable def time_with_dog_in_hours (t : ℝ) : Prop := 
  let d1 := 6 * t          -- Distance run with the dog
  let d2 := 4 * (1 / 2)    -- Distance run alone
  (d1 + d2 = 5) ∧ (t = 1 / 2)

theorem john_runs_with_dog_for_half_hour : ∃ t : ℝ, time_with_dog_in_hours t := 
by
  use (1 / 2)
  sorry

end john_runs_with_dog_for_half_hour_l137_137695


namespace find_f2_l137_137956

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l137_137956


namespace find_x_value_l137_137820

-- Define vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the condition that a + b is parallel to 2a - b
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1 - b.1) = k * (a.1 + b.1) ∧ (2 * a.2 - b.2) = k * (a.2 + b.2)

-- Problem statement: Prove that x = -4
theorem find_x_value : ∀ (x : ℝ),
  parallel_vectors vector_a (vector_b x) → x = -4 :=
by
  sorry

end find_x_value_l137_137820


namespace exists_integers_greater_than_N_l137_137452

theorem exists_integers_greater_than_N (N : ℝ) : 
  ∃ (x1 x2 x3 x4 : ℤ), (x1 > N) ∧ (x2 > N) ∧ (x3 > N) ∧ (x4 > N) ∧ 
  (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 = x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4) := 
sorry

end exists_integers_greater_than_N_l137_137452


namespace sqrt_expr_eq_two_l137_137048

noncomputable def expr := Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2)

theorem sqrt_expr_eq_two : expr = 2 := 
by
  sorry

end sqrt_expr_eq_two_l137_137048


namespace range_of_b_l137_137329

theorem range_of_b (a b : ℝ) (h₁ : a ≤ -1) (h₂ : a * 2 * b - b - 3 * a ≥ 0) : b ≤ 1 := by
  sorry

end range_of_b_l137_137329


namespace people_not_in_pool_l137_137696

-- Define families and their members
def karen_donald_family : ℕ := 2 + 6
def tom_eva_family : ℕ := 2 + 4
def luna_aidan_family : ℕ := 2 + 5
def isabel_jake_family : ℕ := 2 + 3

-- Total number of people
def total_people : ℕ := karen_donald_family + tom_eva_family + luna_aidan_family + isabel_jake_family

-- Number of legs in the pool and people in the pool
def legs_in_pool : ℕ := 34
def people_in_pool : ℕ := legs_in_pool / 2

-- People not in the pool: people who went to store and went to bed
def store_people : ℕ := 2
def bed_people : ℕ := 3
def not_available_people : ℕ := store_people + bed_people

-- Prove (given conditions) number of people not in the pool
theorem people_not_in_pool : total_people - people_in_pool - not_available_people = 4 :=
by
  -- ...proof steps or "sorry"
  sorry

end people_not_in_pool_l137_137696


namespace polynomial_product_equals_expected_result_l137_137089

-- Define the polynomials
def polynomial_product (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- Define the expected result of the product
def expected_result (x : ℝ) : ℝ := x^3 + 1

-- The main theorem to prove
theorem polynomial_product_equals_expected_result (x : ℝ) : polynomial_product x = expected_result x :=
by
  -- Placeholder for the proof
  sorry

end polynomial_product_equals_expected_result_l137_137089


namespace total_washer_dryer_cost_l137_137626

def washer_cost : ℕ := 710
def dryer_cost : ℕ := washer_cost - 220

theorem total_washer_dryer_cost :
  washer_cost + dryer_cost = 1200 :=
  by sorry

end total_washer_dryer_cost_l137_137626


namespace vanya_faster_speed_l137_137753

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l137_137753


namespace density_ratio_of_large_cube_l137_137919

theorem density_ratio_of_large_cube 
  (V0 m0 : ℝ) (initial_density replacement_density: ℝ)
  (initial_mass final_mass : ℝ) (V_total : ℝ) 
  (h1 : initial_density = m0 / V0)
  (h2 : replacement_density = 2 * initial_density)
  (h3 : initial_mass = 8 * m0)
  (h4 : final_mass = 6 * m0 + 2 * (2 * m0))
  (h5 : V_total = 8 * V0) :
  initial_density / (final_mass / V_total) = 0.8 :=
sorry

end density_ratio_of_large_cube_l137_137919


namespace find_a_l137_137210

theorem find_a (z a : ℂ) (h1 : ‖z‖ = 2) (h2 : (z - a)^2 = a) : a = 2 :=
sorry

end find_a_l137_137210


namespace horatio_sonnets_count_l137_137545

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l137_137545


namespace sequence_general_formula_l137_137523

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end sequence_general_formula_l137_137523


namespace log_eval_l137_137797

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l137_137797


namespace relationship_p_q_l137_137282

theorem relationship_p_q (p q : ℝ) : 
  (expand_poly : (x : ℝ) → ((x^2 - p * x + q) * (x - 3)) = (x * x * x + (-p - 3) * x * x + (3 * p + q) * x - 3 * q)) → 
  (linear_term_condition : ∀ x, expand_poly x → (3 * p + q = 0)) → 
  q + 3 * p = 0 :=
begin
  sorry
end

end relationship_p_q_l137_137282


namespace min_value_expression_l137_137712

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_condition : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (sqrt (a^6 + b^4 * c^6) / b + 
   sqrt (b^6 + c^4 * a^6) / c + 
   sqrt (c^6 + a^4 * b^6) / a) ≥ 3 * sqrt 2 :=
by
  sorry

end min_value_expression_l137_137712


namespace part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l137_137265

-- Part 1: Prove f(x) ≥ 0 when a = 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem part1_f_ge_0 : ∀ x : ℝ, f x ≥ 0 := sorry

-- Part 2: Discuss the number of zeros of the function f(x)
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part2_number_of_zeros (a : ℝ) : 
  (a ≤ 0 ∨ a = 1) → ∃! x : ℝ, g a x = 0 := sorry

theorem part2_number_of_zeros_case2 (a : ℝ) : 
  (0 < a ∧ a < 1) ∨ (a > 1) → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0 := sorry

end part1_f_ge_0_part2_number_of_zeros_part2_number_of_zeros_case2_l137_137265


namespace pyramid_sphere_area_l137_137654

theorem pyramid_sphere_area (a : ℝ) (PA PB PC : ℝ) 
  (h1 : PA = PB) (h2 : PA = 2 * PC) 
  (h3 : PA = 2 * a) (h4 : PB = 2 * a) 
  (h5 : 4 * π * (PA^2 + PB^2 + PC^2) / 9 = 9 * π) :
  a = 1 :=
by
  sorry

end pyramid_sphere_area_l137_137654


namespace express_y_in_terms_of_x_l137_137305

-- Defining the parameters and assumptions
variables (x y : ℝ)
variables (h : x * y = 30)

-- Stating the theorem
theorem express_y_in_terms_of_x (h : x * y = 30) : y = 30 / x :=
sorry

end express_y_in_terms_of_x_l137_137305


namespace count_positive_integers_dividing_10n_l137_137391

theorem count_positive_integers_dividing_10n : 
  {n : ℕ // n > 0 ∧ (n*(n+1)/2) ∣ (10 * n)}.card = 5 :=
by
  sorry

end count_positive_integers_dividing_10n_l137_137391


namespace vanya_speed_increased_by_4_l137_137747

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l137_137747


namespace smallest_number_with_sum_32_l137_137373

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, digits.count d = 1

def digits_sum_to_32 (n : ℕ) : Prop :=
  n.digits 10 |>.sum = 32

def is_smallest_number (n : ℕ) : Prop :=
  ∀ m : ℕ, digits_sum_to_32 m → all_digits_different m → n ≤ m

theorem smallest_number_with_sum_32 : ∃ n, all_digits_different n ∧ digits_sum_to_32 n ∧ is_smallest_number n ∧ n = 26789 :=
by
  sorry

end smallest_number_with_sum_32_l137_137373


namespace collinear_points_l137_137233

variables (E F N M B D : Type*)
variables [loc : has_points_on a_line E F N M B D]

theorem collinear_points
(BC CD : E F)
(E_on_BC : E ∈ BC)
(F_on_CD : F ∈ CD)
(EN_perp_AF : ∠EN = 90)
(FM_perp_AE : ∠FM = 90)
(EAF_eq_45 : ∠EAF = 45)
: collinear B M N D := 
sorry

end collinear_points_l137_137233


namespace tiles_required_for_floor_l137_137057

def tileDimensionsInFeet (width_in_inches : ℚ) (length_in_inches : ℚ) : ℚ × ℚ :=
  (width_in_inches / 12, length_in_inches / 12)

def area (length : ℚ) (width : ℚ) : ℚ :=
  length * width

noncomputable def numberOfTiles (floor_length : ℚ) (floor_width : ℚ) (tile_length : ℚ) (tile_width : ℚ) : ℚ :=
  (area floor_length floor_width) / (area tile_length tile_width)

theorem tiles_required_for_floor : numberOfTiles 10 15 (5/12) (2/3) = 540 := by
  sorry

end tiles_required_for_floor_l137_137057


namespace vanya_faster_speed_l137_137760

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l137_137760


namespace k_plus_alpha_is_one_l137_137267

variable (f : ℝ → ℝ) (k α : ℝ)

-- Conditions from part a)
def power_function := ∀ x : ℝ, f x = k * x ^ α
def passes_through_point := f (1 / 2) = 2

-- Statement to be proven
theorem k_plus_alpha_is_one (h1 : power_function f k α) (h2 : passes_through_point f) : k + α = 1 :=
sorry

end k_plus_alpha_is_one_l137_137267


namespace intersection_empty_condition_l137_137957

-- Define the sets M and N under the given conditions
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The theorem that we need to prove based on the problem statement
theorem intersection_empty_condition (b : ℝ) :
  (∀ m : ℝ, M ∩ N m b = ∅) ↔ (b^2 > 6 * m^2 + 2) := sorry

end intersection_empty_condition_l137_137957


namespace problem_statement_l137_137667

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l137_137667


namespace repaved_before_today_correct_l137_137918

variable (total_repaved_so_far repaved_today repaved_before_today : ℕ)

axiom given_conditions : total_repaved_so_far = 4938 ∧ repaved_today = 805 

theorem repaved_before_today_correct :
  total_repaved_so_far = 4938 →
  repaved_today = 805 →
  repaved_before_today = total_repaved_so_far - repaved_today →
  repaved_before_today = 4133 :=
by
  intros
  sorry

end repaved_before_today_correct_l137_137918


namespace missing_number_in_proportion_l137_137141

/-- Given the proportion 2 : 5 = x : 3.333333333333333, prove that the missing number x is 1.3333333333333332 -/
theorem missing_number_in_proportion : ∃ x, (2 / 5 = x / 3.333333333333333) ∧ x = 1.3333333333333332 :=
  sorry

end missing_number_in_proportion_l137_137141


namespace boxes_left_l137_137858

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l137_137858


namespace problem_l137_137812

theorem problem (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y + x * y = 3) :
  (0 < x * y ∧ x * y ≤ 1) ∧ (∀ z : ℝ, z = x + 2 * y → z = 4 * Real.sqrt 2 - 3) :=
by
  sorry

end problem_l137_137812


namespace ryan_spit_distance_l137_137238

def billy_distance : ℝ := 30

def madison_distance (b : ℝ) : ℝ := b + (20 / 100 * b)

def ryan_distance (m : ℝ) : ℝ := m - (50 / 100 * m)

theorem ryan_spit_distance : ryan_distance (madison_distance billy_distance) = 18 :=
by
  sorry

end ryan_spit_distance_l137_137238


namespace mark_charged_more_hours_l137_137162

theorem mark_charged_more_hours (P K M : ℕ) 
  (h1 : P + K + M = 135)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 75 := by {

sorry
}

end mark_charged_more_hours_l137_137162


namespace find_numbers_l137_137116

theorem find_numbers (a b c d : ℕ)
  (h1 : a + b + c = 21)
  (h2 : a + b + d = 28)
  (h3 : a + c + d = 29)
  (h4 : b + c + d = 30) : 
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 :=
sorry

end find_numbers_l137_137116


namespace shadow_building_length_l137_137622

-- Define the basic parameters
def height_flagpole : ℕ := 18
def shadow_flagpole : ℕ := 45
def height_building : ℕ := 20

-- Define the condition on similar conditions
def similar_conditions (h₁ s₁ h₂ s₂ : ℕ) : Prop :=
  h₁ * s₂ = h₂ * s₁

-- Theorem statement
theorem shadow_building_length :
  similar_conditions height_flagpole shadow_flagpole height_building 50 := 
sorry

end shadow_building_length_l137_137622


namespace problem1_correct_problem2_correct_l137_137804

noncomputable def problem1_solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

noncomputable def problem2_solution_set : Set ℝ := {x | (-3 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ≤ 7)}

theorem problem1_correct (x : ℝ) :
  (4 - x) / (x^2 + x + 1) ≤ 1 ↔ x ∈ problem1_solution_set :=
sorry

theorem problem2_correct (x : ℝ) :
  (1 < |x - 2| ∧ |x - 2| ≤ 5) ↔ x ∈ problem2_solution_set :=
sorry

end problem1_correct_problem2_correct_l137_137804


namespace calc_f_g_3_minus_g_f_3_l137_137147

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2

theorem calc_f_g_3_minus_g_f_3 :
  (f (g 3) - g (f 3)) = -96 :=
by
  sorry

end calc_f_g_3_minus_g_f_3_l137_137147


namespace football_cost_l137_137705

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l137_137705


namespace problem_l137_137400

def f (x : ℝ) : ℝ := (x^4 + 2*x^3 + 4*x - 5) ^ 2004 + 2004

theorem problem (x : ℝ) (h : x = Real.sqrt 3 - 1) : f x = 2005 :=
by
  sorry

end problem_l137_137400


namespace distance_AK_l137_137729

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def C : ℝ × ℝ := (1, 0)
noncomputable def D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

-- Define the line equations
noncomputable def line_AB (x : ℝ) : Prop := x = 0
noncomputable def line_CD (x y : ℝ) : Prop := y = (Real.sqrt 2) / (2 - Real.sqrt 2) * (x - 1)

-- Define the intersection point K
noncomputable def K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the desired distance
theorem distance_AK : distance A K = Real.sqrt 2 + 1 :=
by
  -- Proof details are omitted
  sorry

end distance_AK_l137_137729


namespace vanya_speed_problem_l137_137744

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l137_137744


namespace min_sticks_to_avoid_rectangles_l137_137157

noncomputable def min_stick_deletions (n : ℕ) : ℕ :=
  if n = 8 then 43 else 0 -- we define 43 as the minimum for an 8x8 chessboard

theorem min_sticks_to_avoid_rectangles : min_stick_deletions 8 = 43 :=
  by
    sorry

end min_sticks_to_avoid_rectangles_l137_137157


namespace min_value_of_expression_l137_137960

theorem min_value_of_expression (x y : ℝ) (h : x^2 + x * y + y^2 = 3) : x^2 - x * y + y^2 ≥ 1 :=
by 
sorry

end min_value_of_expression_l137_137960


namespace binom_sum_mod_2027_l137_137311

theorem binom_sum_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  T % 2027 = 1089 :=
by
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  have h2027_prime : Nat.prime 2027 := by exact dec_trivial
  sorry -- This is the placeholder for the actual proof

end binom_sum_mod_2027_l137_137311


namespace delta_value_l137_137273

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l137_137273


namespace part1_part2_l137_137126

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (a x : ℝ) := a * |x - 1|

theorem part1 (a : ℝ) :
  (∀ x : ℝ, |f x| = g a x → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

end part1_part2_l137_137126


namespace hexagon_area_twice_triangle_l137_137412

theorem hexagon_area_twice_triangle 
  (O A B C D E F : Point) 
  (h_inscribed : InscribedInCircle O A B C D E F)
  (h_diameters : Diameters O A D ∧ Diameters O B E ∧ Diameters O C F) :
  AreaHexagon A B C D E F = 2 * AreaTriangle A C E :=
sorry

end hexagon_area_twice_triangle_l137_137412


namespace number_of_three_digit_multiples_of_6_l137_137134

theorem number_of_three_digit_multiples_of_6 : 
  let lower_bound := 100
  let upper_bound := 999
  let multiple := 6
  let smallest_n := Nat.ceil (100 / multiple)
  let largest_n := Nat.floor (999 / multiple)
  let count_multiples := largest_n - smallest_n + 1
  count_multiples = 150 := by
  sorry

end number_of_three_digit_multiples_of_6_l137_137134


namespace square_area_l137_137073

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l137_137073


namespace fraction_sent_afternoon_l137_137904

theorem fraction_sent_afternoon :
  ∀ (total_fliers morning_fraction fliers_left_next_day : ℕ),
  total_fliers = 3000 →
  morning_fraction = 1/5 →
  fliers_left_next_day = 1800 →
  ((total_fliers - total_fliers * morning_fraction) - fliers_left_next_day) / (total_fliers - total_fliers * morning_fraction) = 1/4 :=
by
  intros total_fliers morning_fraction fliers_left_next_day h1 h2 h3
  sorry

end fraction_sent_afternoon_l137_137904


namespace remainder_7547_div_11_l137_137902

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11_l137_137902


namespace new_number_is_100t_plus_10u_plus_3_l137_137551

theorem new_number_is_100t_plus_10u_plus_3 (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  new_number = 100 * t + 10 * u + 3 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  show new_number = 100 * t + 10 * u + 3
  sorry

end new_number_is_100t_plus_10u_plus_3_l137_137551


namespace constant_term_of_binomial_expansion_l137_137263

noncomputable def constant_in_binomial_expansion (a : ℝ) : ℝ := 
  if h : a = ∫ (x : ℝ) in (0)..(1), 2 * x 
  then ((1 : ℝ) - (a : ℝ)^(-1 : ℝ))^6
  else 0

theorem constant_term_of_binomial_expansion : 
  ∃ a : ℝ, (a = ∫ (x : ℝ) in (0)..(1), 2 * x) → constant_in_binomial_expansion a = (15 : ℝ) := sorry

end constant_term_of_binomial_expansion_l137_137263


namespace winning_candidate_percentage_l137_137474

noncomputable def votes : List ℝ := [15236.71, 20689.35, 12359.23, 30682.49, 25213.17, 18492.93]

theorem winning_candidate_percentage :
  (List.foldr max 0 votes / (List.foldr (· + ·) 0 votes) * 100) = 25.01 :=
by
  sorry

end winning_candidate_percentage_l137_137474


namespace distance_between_points_l137_137771

theorem distance_between_points (x y : ℝ) (h : x + y = 10 / 3) : 
  4 * (x + y) = 40 / 3 :=
sorry

end distance_between_points_l137_137771


namespace find_y_eq_7_5_l137_137371

theorem find_y_eq_7_5 (y : ℝ) (hy1 : 0 < y) (hy2 : ∃ z : ℤ, ((z : ℝ) ≤ y) ∧ (y < z + 1))
  (hy3 : (Int.floor y : ℝ) * y = 45) : y = 7.5 :=
sorry

end find_y_eq_7_5_l137_137371


namespace library_fiction_percentage_l137_137058

theorem library_fiction_percentage:
  let original_volumes := 18360
  let fiction_percentage := 0.30
  let fraction_transferred := 1/3
  let fraction_fiction_transferred := 1/5
  let initial_fiction := fiction_percentage * original_volumes
  let transferred_volumes := fraction_transferred * original_volumes
  let transferred_fiction := fraction_fiction_transferred * transferred_volumes
  let remaining_fiction := initial_fiction - transferred_fiction
  let remaining_volumes := original_volumes - transferred_volumes
  let remaining_fiction_percentage := (remaining_fiction / remaining_volumes) * 100
  remaining_fiction_percentage = 35 := 
by
  sorry

end library_fiction_percentage_l137_137058


namespace coprime_permutations_count_l137_137478

noncomputable def count_coprime_permutations (l : List ℕ) : ℕ :=
if h : l = [1, 2, 3, 4, 5, 6, 7] ∨ l = [1, 2, 3, 7, 5, 6, 4] -- other permutations can be added as needed
then 864
else 0

theorem coprime_permutations_count :
  count_coprime_permutations [1, 2, 3, 4, 5, 6, 7] = 864 :=
sorry

end coprime_permutations_count_l137_137478


namespace soccer_ball_cost_l137_137889

theorem soccer_ball_cost (x : ℝ) (soccer_balls basketballs : ℕ) 
  (soccer_ball_cost basketball_cost : ℝ) 
  (h1 : soccer_balls = 2 * basketballs)
  (h2 : 5000 = soccer_balls * soccer_ball_cost)
  (h3 : 4000 = basketballs * basketball_cost)
  (h4 : basketball_cost = soccer_ball_cost + 30)
  (eqn : 5000 / soccer_ball_cost = 2 * (4000 / basketball_cost)) :
  soccer_ball_cost = x :=
by
  sorry

end soccer_ball_cost_l137_137889


namespace vertex_hyperbola_l137_137024

theorem vertex_hyperbola (a b : ℝ) (h_cond : 8 * a^2 + 4 * a * b = b^3) :
    let xv := -b / (2 * a)
    let yv := (4 * a - b^2) / (4 * a)
    (xv * yv = 1) :=
  by
  sorry

end vertex_hyperbola_l137_137024


namespace not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l137_137122

theorem not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C (A B C : ℝ) (h1 : A = 2 * C) (h2 : B = 2 * C) (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := 
by 
  sorry

end not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l137_137122


namespace fraction_given_to_classmates_l137_137699

theorem fraction_given_to_classmates
  (total_boxes : ℕ) (pens_per_box : ℕ)
  (percentage_to_friends : ℝ) (pens_left_after_classmates : ℕ) :
  total_boxes = 20 →
  pens_per_box = 5 →
  percentage_to_friends = 0.40 →
  pens_left_after_classmates = 45 →
  (15 / (total_boxes * pens_per_box - percentage_to_friends * total_boxes * pens_per_box)) = 1 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_given_to_classmates_l137_137699


namespace triangles_from_ten_points_l137_137585

theorem triangles_from_ten_points : 
  let n := 10 in
  let k := 3 in
  nat.choose n k = 120 :=
by
  sorry

end triangles_from_ten_points_l137_137585


namespace compute_x_squared_first_compute_x_squared_second_l137_137128

variable (x : ℝ)
variable (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1)

theorem compute_x_squared_first : 
  1 / (1 / x - 1 / (x + 1)) - x = x^2 :=
by
  sorry

theorem compute_x_squared_second : 
  1 / (1 / (x - 1) - 1 / x) + x = x^2 :=
by
  sorry

end compute_x_squared_first_compute_x_squared_second_l137_137128


namespace abs_inequality_solution_l137_137485

theorem abs_inequality_solution (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3 / 2 :=
by
  sorry

end abs_inequality_solution_l137_137485


namespace integral_evaluation_l137_137640

noncomputable def definite_integral (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem integral_evaluation : 
  definite_integral 1 2 (fun x => 1 / x + x) = Real.log 2 + 3 / 2 :=
  sorry

end integral_evaluation_l137_137640


namespace square_area_l137_137062

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l137_137062


namespace vector_solution_l137_137553

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_solution (a x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by sorry

end vector_solution_l137_137553


namespace positive_integers_divisors_l137_137389

theorem positive_integers_divisors :
  {n : ℕ | n > 0 ∧ (n * (n + 1) / 2) ∣ (10 * n)}.card = 5 :=
by
  sorry

end positive_integers_divisors_l137_137389


namespace range_of_b_l137_137567

noncomputable def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem range_of_b (b : ℝ) : (∀ n : ℕ, 0 < n → a_n (n+1) b > a_n n b) ↔ (-3 < b) :=
by
    sorry

end range_of_b_l137_137567


namespace positive_n_of_single_solution_l137_137805

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l137_137805


namespace fourth_power_of_cube_third_smallest_prime_l137_137899

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l137_137899


namespace platform_length_l137_137929

theorem platform_length
  (train_length : ℝ := 360) -- The train is 360 meters long
  (train_speed_kmh : ℝ := 45) -- The train runs at a speed of 45 km/hr
  (time_to_pass_platform : ℝ := 60) -- It takes 60 seconds to pass the platform
  (platform_length : ℝ) : platform_length = 390 :=
by
  sorry

end platform_length_l137_137929


namespace karlson_expenditure_exceeds_2000_l137_137846

theorem karlson_expenditure_exceeds_2000 :
  ∃ n m : ℕ, 25 * n + 340 * m > 2000 :=
by {
  -- proof must go here
  sorry
}

end karlson_expenditure_exceeds_2000_l137_137846


namespace consumption_increased_by_27_91_percent_l137_137313
noncomputable def percentage_increase_in_consumption (T C : ℝ) : ℝ :=
  let new_tax_rate := 0.86 * T
  let new_revenue_effect := 1.1000000000000085
  let cons_percentage_increase (P : ℝ) := (new_tax_rate * (C * (1 + P))) = new_revenue_effect * (T * C)
  let P_solution := 0.2790697674418605
  if cons_percentage_increase P_solution then P_solution * 100 else 0

-- The statement we are proving
theorem consumption_increased_by_27_91_percent (T C : ℝ) (hT : 0 < T) (hC : 0 < C) :
  percentage_increase_in_consumption T C = 27.91 :=
by
  sorry

end consumption_increased_by_27_91_percent_l137_137313


namespace solve_for_y_l137_137012

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4)) : y = 1296 := by
  sorry

end solve_for_y_l137_137012


namespace polygon_sides_eq_seven_l137_137137

theorem polygon_sides_eq_seven (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) → n = 7 :=
by
  sorry

end polygon_sides_eq_seven_l137_137137


namespace find_e_m_l137_137186

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![4, 5], ![7, e]]
noncomputable def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (4 * e - 35)) • ![![e, -5], ![-7, 4]]

theorem find_e_m (e m : ℝ) (B_inv_eq_mB : B_inv e = m • B e) : e = -4 ∧ m = 1 / 51 :=
sorry

end find_e_m_l137_137186


namespace unique_triple_solution_zero_l137_137515

theorem unique_triple_solution_zero (m n k : ℝ) :
  (∃ x : ℝ, m * x ^ 2 + n = 0) ∧
  (∃ x : ℝ, n * x ^ 2 + k = 0) ∧
  (∃ x : ℝ, k * x ^ 2 + m = 0) ↔
  (m = 0 ∧ n = 0 ∧ k = 0) := 
sorry

end unique_triple_solution_zero_l137_137515


namespace nonagon_diagonals_count_l137_137995

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l137_137995


namespace sin_cos_identity_l137_137047

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) + Real.cos (20 * Real.pi / 180) * Real.sin (140 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end sin_cos_identity_l137_137047


namespace find_fraction_value_l137_137277

theorem find_fraction_value {m n r t : ℚ}
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 5) :
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 :=
by
  sorry

end find_fraction_value_l137_137277


namespace cone_radius_of_base_l137_137312

noncomputable def radius_of_base (CSA l : ℝ) : ℝ := 
  CSA / (Real.pi * l)

theorem cone_radius_of_base (CSA l r : ℝ) (h₁ : l = 20) (h₂ : CSA = 628.3185307179587) : 
  radius_of_base CSA l = 10 := by
  rw [h₁, h₂]
  -- sorry

end cone_radius_of_base_l137_137312


namespace profit_function_marginal_profit_function_maximize_profit_l137_137779

noncomputable def R (x : ℝ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3
noncomputable def C (x : ℝ) : ℝ := 460 * x + 5000
noncomputable def P (x : ℝ) : ℝ := R(x) - C(x)
noncomputable def MP (x : ℝ) : ℝ := P(x + 1) - P(x)

theorem profit_function (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 20) :
  P(x) = -10 * x^3 + 45 * x^2 + 3240 * x - 5000 := by
sorry

theorem marginal_profit_function (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 19) :
  MP(x) = -30 * x^2 + 60 * x + 3275 := by
sorry

theorem maximize_profit : ∃ x : ℝ, 1 ≤ x ∧ x ≤ 20 ∧ ∀ y : ℝ, 1 ≤ y → y ≤ 20 → P(x) ≥ P(y) := by
  use 12
  split
  · exact dec_trivial
  · split
  · exact dec_trivial
  · intros y hy1 hy2
    -- proof that P(12) is the maximum
  sorry

end profit_function_marginal_profit_function_maximize_profit_l137_137779


namespace blueberry_picking_l137_137323

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end blueberry_picking_l137_137323


namespace prove_sufficient_and_necessary_l137_137615

-- The definition of the focus of the parabola y^2 = 4x.
def focus_parabola : (ℝ × ℝ) := (1, 0)

-- The condition that the line passes through a given point.
def line_passes_through (m b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = m * p.1 + b

-- Let y = x + b and the equation of the parabola be y^2 = 4x.
def sufficient_and_necessary (b : ℝ) : Prop :=
  line_passes_through 1 b focus_parabola ↔ b = -1

theorem prove_sufficient_and_necessary : sufficient_and_necessary (-1) :=
by
  sorry

end prove_sufficient_and_necessary_l137_137615


namespace eval_expression_l137_137639

theorem eval_expression : -20 + 12 * ((5 + 15) / 4) = 40 :=
by
  sorry

end eval_expression_l137_137639


namespace remainder_when_divided_by_x_minus_2_l137_137203

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 33 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l137_137203


namespace time_to_coffee_shop_is_18_l137_137664

variable (cycle_constant_pace : Prop)
variable (time_cycle_library : ℕ)
variable (distance_cycle_library : ℕ)
variable (distance_to_coffee_shop : ℕ)

theorem time_to_coffee_shop_is_18
  (h_const_pace : cycle_constant_pace)
  (h_time_library : time_cycle_library = 30)
  (h_distance_library : distance_cycle_library = 5)
  (h_distance_coffee : distance_to_coffee_shop = 3)
  : (30 / 5) * 3 = 18 :=
by
  sorry

end time_to_coffee_shop_is_18_l137_137664


namespace find_p_l137_137465

variable (A B C D p q u v w : ℝ)
variable (hu : u + v + w = -B / A)
variable (huv : u * v + v * w + w * u = C / A)
variable (huvw : u * v * w = -D / A)
variable (hpq : u^2 + v^2 = -p)
variable (hq : u^2 * v^2 = q)

theorem find_p (A B C D : ℝ) (u v w : ℝ) 
  (H1 : u + v + w = -B / A)
  (H2 : u * v + v * w + w * u = C / A)
  (H3 : u * v * w = -D / A)
  (H4 : v = -u - w)
  : p = (B^2 - 2 * C) / A^2 :=
by sorry

end find_p_l137_137465


namespace prove_2x_plus_y_le_sqrt_11_l137_137401

variable (x y : ℝ)
variable (h : 3 * x^2 + 2 * y^2 ≤ 6)

theorem prove_2x_plus_y_le_sqrt_11 : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end prove_2x_plus_y_le_sqrt_11_l137_137401


namespace largest_even_integer_of_product_2880_l137_137317

theorem largest_even_integer_of_product_2880 :
  ∃ n : ℤ, (n-2) * n * (n+2) = 2880 ∧ n + 2 = 22 := 
by {
  sorry
}

end largest_even_integer_of_product_2880_l137_137317


namespace total_envelopes_l137_137450

def total_stamps : ℕ := 52
def lighter_envelopes : ℕ := 6
def stamps_per_lighter_envelope : ℕ := 2
def stamps_per_heavier_envelope : ℕ := 5

theorem total_envelopes (total_stamps lighter_envelopes stamps_per_lighter_envelope stamps_per_heavier_envelope : ℕ) 
  (h : total_stamps = 52 ∧ lighter_envelopes = 6 ∧ stamps_per_lighter_envelope = 2 ∧ stamps_per_heavier_envelope = 5) : 
  lighter_envelopes + (total_stamps - (stamps_per_lighter_envelope * lighter_envelopes)) / stamps_per_heavier_envelope = 14 :=
by
  sorry

end total_envelopes_l137_137450


namespace remainder_of_difference_l137_137044

open Int

theorem remainder_of_difference (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 3) (h : a > b) : (a - b) % 6 = 5 :=
  sorry

end remainder_of_difference_l137_137044


namespace asymptote_slope_l137_137184

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Lean statement to prove slope of asymptotes
theorem asymptote_slope :
  (∀ x y : ℝ, hyperbola x y → (y/x) = 3/4 ∨ (y/x) = -(3/4)) :=
by
  sorry

end asymptote_slope_l137_137184


namespace problem_solution_l137_137865

theorem problem_solution (x : ℝ) :
          ((3 * x - 4) * (x + 5) ≠ 0) → 
          (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
          (x ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioi (4 / 3)) :=
sorry

end problem_solution_l137_137865


namespace speed_of_man_in_still_water_l137_137489

theorem speed_of_man_in_still_water 
  (V_m V_s : ℝ)
  (h1 : 6 = V_m + V_s)
  (h2 : 4 = V_m - V_s) : 
  V_m = 5 := 
by 
  sorry

end speed_of_man_in_still_water_l137_137489


namespace minimum_value_l137_137852

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃z, z = (x^2 + y^2) / (x + y)^2 ∧ z ≥ 1/2 := 
sorry

end minimum_value_l137_137852


namespace area_of_square_l137_137065

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end area_of_square_l137_137065


namespace problem_solution_correct_l137_137501

open Real

noncomputable def probability_token_covers_black_region
  (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (triangle_leg : ℝ) (token_diameter : ℝ) : ℝ :=
  let total_area := (rectangle_width - token_diameter) * (rectangle_height - token_diameter)
  let triangle_area := 2 * (1 / 2 * triangle_leg^2)
  let additional_area_one_triangle := (π * (token_diameter / 2)^2 / (2 * 2)) + (triangle_leg * sqrt(2) / 2)
  let total_black_area := triangle_area + 2 * additional_area_one_triangle
  (total_black_area) / (total_area)

theorem problem_solution_correct :
  probability_token_covers_black_region 10 6 3 2 = (9 + (π / 2) + 3 * sqrt 2) / 32 :=
by
  -- the proof goes here
  sorry

end problem_solution_correct_l137_137501


namespace positive_integers_dividing_sum_10n_l137_137390

def S (n : ℕ) : ℕ := n * (n + 1) / 2

theorem positive_integers_dividing_sum_10n :
  {n : ℕ | n > 0 ∧ S n ∣ 10 * n}.to_finset.card = 5 :=
by
  sorry

end positive_integers_dividing_sum_10n_l137_137390


namespace total_amount_l137_137098

theorem total_amount {B C : ℝ} 
  (h1 : C = 1600) 
  (h2 : 4 * B = 16 * C) : 
  B + C = 2000 :=
sorry

end total_amount_l137_137098


namespace initial_mean_l137_137307

theorem initial_mean (M : ℝ) (n : ℕ) (observed_wrongly correct_wrongly : ℝ) (new_mean : ℝ) :
  n = 50 ∧ observed_wrongly = 23 ∧ correct_wrongly = 45 ∧ new_mean = 36.5 → M = 36.06 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  have sum_initial := n * M
  have diff := correct_wrongly - observed_wrongly
  have sum_corrected := sum_initial + diff
  have new_sum_corrected := n * new_mean
  have equation := sum_corrected = new_sum_corrected
  rw [←h3, ←h2, ←h1, ←h6] at equation
  have sum_initial_calculated := new_sum_corrected - diff
  have M_calculated_eq := sum_initial_calculated / n
  rw [←h5, ←h1] at M_calculated_eq
  -- Calculate manually to show final M == 36.06 which is the correct proof
  sorry

end initial_mean_l137_137307


namespace remaining_students_correct_l137_137033

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end remaining_students_correct_l137_137033


namespace f2011_eq_two_l137_137399

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity_eqn : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom f1_eq_two : f 1 = 2

theorem f2011_eq_two : f 2011 = 2 := 
by 
  sorry

end f2011_eq_two_l137_137399


namespace total_rainfall_l137_137479

theorem total_rainfall
  (r₁ r₂ : ℕ)
  (T t₁ : ℕ)
  (H1 : r₁ = 30)
  (H2 : r₂ = 15)
  (H3 : T = 45)
  (H4 : t₁ = 20) :
  r₁ * t₁ + r₂ * (T - t₁) = 975 := by
  sorry

end total_rainfall_l137_137479


namespace smallest_number_with_unique_digits_sum_32_l137_137378

-- Define the conditions as predicates for clarity.
def all_digits_different (n : ℕ) : Prop :=
  let digits := List.ofString (n.toString)
  digits.nodup

def sum_of_digits (n : ℕ) : ℕ :=
  (List.ofString (n.toString)).foldr (fun d acc => acc + (d.toNat - '0'.toNat)) 0

-- The main statement to prove
theorem smallest_number_with_unique_digits_sum_32 : ∃ n : ℕ, all_digits_different n ∧ sum_of_digits n = 32 ∧ (∀ m : ℕ, all_digits_different m ∧ sum_of_digits m = 32 → n ≤ m) :=
  ∃ n = 26789,
  all_digits_different 26789 ∧ sum_of_digits 26789 = 32 ∧
  (h : all_digits_different m ∧ sum_of_digits m = 32 → 26789 ≤ m)
  sorry -- proof omitted

end smallest_number_with_unique_digits_sum_32_l137_137378


namespace percentage_increase_in_consumption_l137_137194

-- Define the conditions
variables {T C : ℝ}  -- T: original tax, C: original consumption
variables (P : ℝ)    -- P: percentage increase in consumption

-- Non-zero conditions
variables (hT : T ≠ 0) (hC : C ≠ 0)

-- Define the Lean theorem
theorem percentage_increase_in_consumption 
  (h : 0.8 * (1 + P / 100) = 0.96) : 
  P = 20 :=
by
  sorry

end percentage_increase_in_consumption_l137_137194


namespace abe_family_total_yen_l137_137627

theorem abe_family_total_yen (yen_checking : ℕ) (yen_savings : ℕ) (h₁ : yen_checking = 6359) (h₂ : yen_savings = 3485) : yen_checking + yen_savings = 9844 :=
by
  sorry

end abe_family_total_yen_l137_137627


namespace tan_double_angle_l137_137824

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4 / 3 :=
by
  sorry

end tan_double_angle_l137_137824


namespace vanya_speed_l137_137741

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l137_137741


namespace calculate_expression_l137_137091

theorem calculate_expression : 
  - 3 ^ 2 + (-12) * abs (-1/2) - 6 / (-1) = -9 := 
by 
  sorry

end calculate_expression_l137_137091


namespace abc_sum_eq_11_sqrt_6_l137_137173

variable {a b c : ℝ}

theorem abc_sum_eq_11_sqrt_6 : 
  0 < a → 0 < b → 0 < c → 
  a * b = 36 → 
  a * c = 72 → 
  b * c = 108 → 
  a + b + c = 11 * Real.sqrt 6 :=
by sorry

end abc_sum_eq_11_sqrt_6_l137_137173


namespace value_of_expression_l137_137612

theorem value_of_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) : x + 2 * y = 10 :=
by
  -- Proof goes here
  sorry

end value_of_expression_l137_137612


namespace sum_of_solutions_l137_137436

theorem sum_of_solutions (x y : ℝ) (h₁ : y = 8) (h₂ : x^2 + y^2 = 144) : 
  ∃ x1 x2 : ℝ, (x1 = 4 * Real.sqrt 5 ∧ x2 = -4 * Real.sqrt 5) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_solutions_l137_137436


namespace tallest_is_first_l137_137709

variable (P : Type) -- representing people
variable (line : Fin 9 → P) -- original line order (0 = shortest, 8 = tallest)
variable (Hoseok : P) -- Hoseok

-- Conditions
axiom tallest_person : line 8 = Hoseok

-- Theorem
theorem tallest_is_first :
  ∃ line' : Fin 9 → P, (∀ i : Fin 9, line' i = line (8 - i)) → line' 0 = Hoseok :=
  by
  sorry

end tallest_is_first_l137_137709


namespace longest_chord_of_circle_l137_137531

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l137_137531


namespace total_bill_is_60_l137_137235

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end total_bill_is_60_l137_137235


namespace arithmetic_sequence_fourth_term_l137_137427

theorem arithmetic_sequence_fourth_term (b d : ℝ) (h : 2 * b + 2 * d = 10) : b + d = 5 :=
by
  sorry

end arithmetic_sequence_fourth_term_l137_137427


namespace min_y_value_l137_137244

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value_l137_137244


namespace smallest_unique_digit_sum_32_l137_137384

theorem smallest_unique_digit_sum_32 : ∃ n, 
  (∀ d₁ d₂ ∈ digits n, d₁ ≠ d₂) ∧ 
  (digits n).sum = 32 ∧ 
  ∀ m, 
    (∀ d₁ d₂ ∈ digits m, d₁ ≠ d₂) ∧ 
    (digits m).sum = 32 → 
      m ≥ n := 
begin
  sorry
end

end smallest_unique_digit_sum_32_l137_137384


namespace number_of_triangles_l137_137590

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l137_137590


namespace center_of_circle_l137_137816

theorem center_of_circle (x y : ℝ) :
  x^2 + y^2 - 2 * x - 6 * y + 1 = 0 →
  (1, 3) = (1, 3) :=
by
  intros h
  sorry

end center_of_circle_l137_137816


namespace square_area_l137_137063

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l137_137063


namespace walking_speed_l137_137339

theorem walking_speed 
  (v : ℕ) -- v represents the man's walking speed in kmph
  (distance_formula : distance = speed * time)
  (distance_walking : distance = v * 9)
  (distance_running : distance = 24 * 3) : 
  v = 8 :=
by
  sorry

end walking_speed_l137_137339


namespace part1_part2_l137_137964

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x^2 - 3 * x + 2 = 0 }

theorem part1 (a : ℝ) : (A a = ∅) ↔ (a > 9/8) := sorry

theorem part2 (a : ℝ) : 
  (∃ x, A a = {x}) ↔ 
  (a = 0 ∧ A a = {2 / 3})
  ∨ (a = 9 / 8 ∧ A a = {4 / 3}) := sorry

end part1_part2_l137_137964


namespace mass_percentage_O_is_26_2_l137_137643

noncomputable def mass_percentage_O_in_Benzoic_acid : ℝ :=
  let molar_mass_C := 12.01
  let molar_mass_H := 1.01
  let molar_mass_O := 16.00
  let molar_mass_Benzoic_acid := (7 * molar_mass_C) + (6 * molar_mass_H) + (2 * molar_mass_O)
  let mass_O_in_Benzoic_acid := 2 * molar_mass_O
  (mass_O_in_Benzoic_acid / molar_mass_Benzoic_acid) * 100

theorem mass_percentage_O_is_26_2 :
  mass_percentage_O_in_Benzoic_acid = 26.2 := by
  sorry

end mass_percentage_O_is_26_2_l137_137643


namespace time_to_cross_l137_137050

noncomputable def length_first_train : ℝ := 210
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
noncomputable def length_second_train : ℝ := 290.04
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

noncomputable def relative_speed := speed_first_train + speed_second_train
noncomputable def total_length := length_first_train + length_second_train
noncomputable def crossing_time := total_length / relative_speed

theorem time_to_cross : crossing_time = 9 := by
  let length_first_train : ℝ := 210
  let speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
  let length_second_train : ℝ := 290.04
  let speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

  let relative_speed := speed_first_train + speed_second_train
  let total_length := length_first_train + length_second_train
  let crossing_time := total_length / relative_speed

  show crossing_time = 9
  sorry

end time_to_cross_l137_137050


namespace sequence_general_term_l137_137303

theorem sequence_general_term (n : ℕ) : 
  (2 * n - 1) / (2 ^ n) = a_n := 
sorry

end sequence_general_term_l137_137303


namespace beautiful_ratio_l137_137599

theorem beautiful_ratio (A B C : Type) (l1 l2 b : ℕ) 
  (h : l1 + l2 + b = 20) (h1 : l1 = 8 ∨ l2 = 8 ∨ b = 8) :
  (b / l1 = 1/2) ∨ (b / l2 = 1/2) ∨ (l1 / l2 = 4/3) ∨ (l2 / l1 = 4/3) :=
by
  sorry

end beautiful_ratio_l137_137599


namespace find_p_l137_137190

theorem find_p (m n p : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < p) 
  (h : 3 * m + 3 / (n + 1 / p) = 17) : p = 2 := 
sorry

end find_p_l137_137190


namespace handshakes_count_l137_137084

def women := 6
def teams := 3
def shakes_per_woman := 4
def total_handshakes := (6 * 4) / 2

theorem handshakes_count : total_handshakes = 12 := by
  -- We provide this theorem directly.
  rfl

end handshakes_count_l137_137084


namespace choir_members_minimum_l137_137775

theorem choir_members_minimum (n : Nat) (h9 : n % 9 = 0) (h10 : n % 10 = 0) (h11 : n % 11 = 0) (h14 : n % 14 = 0) : n = 6930 :=
sorry

end choir_members_minimum_l137_137775


namespace distinct_diagonals_in_nonagon_l137_137974

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l137_137974


namespace line_general_eq_curve_rect_eq_exists_max_distance_l137_137408

noncomputable def parametric_line_eq (t : ℝ) : ℝ × ℝ :=
  (-1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

def polar_curve_eq (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ^2 + 3 * ρ^2 * Real.sin θ^2 - 3 = 0

theorem line_general_eq : ∃ (l : ℝ → ℝ), ∀ x y t, 
  parametric_line_eq t = (x, y) → y = x + 1 := sorry

theorem curve_rect_eq : ∃ (C₁ : ℝ × ℝ → Prop), ∀ x y ρ θ,
  polar_curve_eq ρ θ → C₁ (x, y) ↔ (x^2 / 3 + y^2 = 1) := sorry

def distance_to_line (P : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
  abs (P.1 - P.2 + 1) / Real.sqrt 2

theorem exists_max_distance: ∃ P : ℝ × ℝ,
  curve_rect_eq P (sqrt 3 * Real.cos θ, Real.sin θ) ∧
  ∀ (Q : ℝ × ℝ), curve_rect_eq Q (sqrt 3 * Real.cos θ, Real.sin θ) → 
  distance_to_line Q (λ x, x + 1) ≤ distance_to_line P (λ x, x + 1) ∧
  P = (3 / 2, -1 / 2) ∧
  distance_to_line P (λ x, x + 1) = 3 * Real.sqrt 2 / 2 := sorry

end line_general_eq_curve_rect_eq_exists_max_distance_l137_137408


namespace number_of_integers_l137_137997

theorem number_of_integers (n : ℤ) : 
    (100 < n ∧ n < 300) ∧ (n % 7 = n % 9) → 
    (∃ count: ℕ, count = 21) := by
  sorry

end number_of_integers_l137_137997


namespace no_non_negative_solutions_l137_137616

theorem no_non_negative_solutions (a b : ℕ) (h_diff : a ≠ b) (d := Nat.gcd a b) 
                                 (a' := a / d) (b' := b / d) (n := d * (a' * b' - a' - b')) :
  ¬ ∃ x y : ℕ, a * x + b * y = n := 
by
  sorry

end no_non_negative_solutions_l137_137616


namespace find_brick_length_l137_137498

-- Definitions of dimensions
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 750
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5
def num_bricks : ℝ := 6000

-- Volume calculations
def volume_wall : ℝ := wall_length * wall_height * wall_thickness
def volume_brick (x : ℝ) : ℝ := x * brick_width * brick_height

-- Statement of the problem
theorem find_brick_length (length_of_brick : ℝ) :
  volume_wall = num_bricks * volume_brick length_of_brick → length_of_brick = 25 :=
by
  simp [volume_wall, volume_brick, num_bricks, brick_width, brick_height, wall_length, wall_height, wall_thickness]
  intro h 
  sorry

end find_brick_length_l137_137498


namespace tom_dollars_more_than_jerry_l137_137694

theorem tom_dollars_more_than_jerry (total_slices : ℕ)
  (jerry_slices : ℕ)
  (tom_slices : ℕ)
  (plain_cost : ℕ)
  (pineapple_additional_cost : ℕ)
  (total_cost : ℕ)
  (cost_per_slice : ℚ)
  (cost_jerry : ℚ)
  (cost_tom : ℚ)
  (jerry_ate_plain : jerry_slices = 5)
  (tom_ate_pineapple : tom_slices = 5)
  (total_slices_10 : total_slices = 10)
  (plain_cost_10 : plain_cost = 10)
  (pineapple_additional_cost_3 : pineapple_additional_cost = 3)
  (total_cost_13 : total_cost = plain_cost + pineapple_additional_cost)
  (cost_per_slice_calc : cost_per_slice = total_cost / total_slices)
  (cost_jerry_calc : cost_jerry = cost_per_slice * jerry_slices)
  (cost_tom_calc : cost_tom = cost_per_slice * tom_slices) :
  cost_tom - cost_jerry = 0 := by
  sorry

end tom_dollars_more_than_jerry_l137_137694


namespace probability_students_from_different_grades_l137_137220

theorem probability_students_from_different_grades :
  let total_students := 4
  let first_grade_students := 2
  let second_grade_students := 2
  (2 from total_students are selected) ->
  (2 from total_students are from different grades) ->
  ℝ :=
by 
  sorry

end probability_students_from_different_grades_l137_137220


namespace lowest_fraction_of_job_done_l137_137510

theorem lowest_fraction_of_job_done :
  ∀ (rateA rateB rateC rateB_plus_C : ℝ),
  (rateA = 1/4) → (rateB = 1/6) → (rateC = 1/8) →
  (rateB_plus_C = rateB + rateC) →
  rateB_plus_C = 7/24 := by
  intros rateA rateB rateC rateB_plus_C hA hB hC hBC
  sorry

end lowest_fraction_of_job_done_l137_137510


namespace max_value_of_f_l137_137679

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f :
  ∀ a b : ℝ, (f (-1) a b = 0) → (f (-3) a b = 0) → (f 1 a b = 0) → (f (-5) a b = 0) →
  is_symmetric_about_line (f x a b) (-2) → ∀ x : ℝ, f x a b ≤ 16 := 
by
  intros a b h1 h2 h3 h4 h_symm x
  -- Proof steps go here
  sorry

end max_value_of_f_l137_137679


namespace square_area_correct_l137_137068

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l137_137068


namespace differentiable_function_zero_l137_137914

noncomputable def f : ℝ → ℝ := sorry

theorem differentiable_function_zero (f : ℝ → ℝ) (h_diff : ∀ x ≥ 0, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 0) (h_fun : ∀ x ≥ 0, ∀ y ≥ 0, (x = y^2) → deriv f x = f y) : 
  ∀ x ≥ 0, f x = 0 :=
by
  sorry

end differentiable_function_zero_l137_137914


namespace thirty_one_star_thirty_two_l137_137837

def complex_op (x y : ℝ) : ℝ :=
sorry

axiom op_zero (x : ℝ) : complex_op x 0 = 1

axiom op_associative (x y z : ℝ) : complex_op (complex_op x y) z = z * (x * y) + z

theorem thirty_one_star_thirty_two : complex_op 31 32 = 993 :=
by
  sorry

end thirty_one_star_thirty_two_l137_137837


namespace find_angle_C_l137_137691

theorem find_angle_C (A B C : ℝ) (h1 : |Real.cos A - (Real.sqrt 3 / 2)| + (1 - Real.tan B)^2 = 0) :
  C = 105 :=
by
  sorry

end find_angle_C_l137_137691


namespace smallest_number_with_unique_digits_summing_to_32_l137_137383

-- Definition: Digits of a number n are all distinct
def all_digits_different (n : ℕ) : Prop :=
  let digits := (n.digits : List ℕ)
  digits.nodup

-- Definition: Sum of the digits of number n equals 32
def sum_of_digits_equals_32 (n : ℕ) : Prop :=
  let digits := (n.digits : List ℕ)
  digits.sum = 32

-- Theorem: The smallest number satisfying the given conditions
theorem smallest_number_with_unique_digits_summing_to_32 
  (n : ℕ) (h1 : all_digits_different n) (h2 : sum_of_digits_equals_32 n) :
  n = 26789 :=
sorry

end smallest_number_with_unique_digits_summing_to_32_l137_137383


namespace inequality_solution_l137_137172

theorem inequality_solution (x : ℝ) (h1 : 2 * x + 1 > x + 3) (h2 : 2 * x - 4 < x) : 2 < x ∧ x < 4 := sorry

end inequality_solution_l137_137172


namespace smallest_positive_m_l137_137766

theorem smallest_positive_m (m : ℕ) : 
  (∃ n : ℤ, (10 * n * (n + 1) = 600) ∧ (m = 10 * (n + (n + 1)))) → (m = 170) :=
by 
  sorry

end smallest_positive_m_l137_137766


namespace vanya_faster_by_4_l137_137736

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l137_137736


namespace jason_less_than_jenny_l137_137693

-- Definition of conditions

def grade_Jenny : ℕ := 95
def grade_Bob : ℕ := 35
def grade_Jason : ℕ := 2 * grade_Bob -- Bob's grade is half of Jason's grade

-- The theorem we need to prove
theorem jason_less_than_jenny : grade_Jenny - grade_Jason = 25 :=
by
  sorry

end jason_less_than_jenny_l137_137693


namespace sector_area_l137_137958

theorem sector_area (theta l : ℝ) (h_theta : theta = 2) (h_l : l = 2) :
    let r := l / theta
    let S := 1 / 2 * l * r
    S = 1 := by
  sorry

end sector_area_l137_137958


namespace num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l137_137716

theorem num_shoes_sold (price_shoes : ℕ) (num_shirts : ℕ) (price_shirts : ℕ) (total_earn_per_person : ℕ) : ℕ :=
  let total_earnings_shirts := num_shirts * price_shirts
  let total_earnings := total_earn_per_person * 2
  let earnings_from_shoes := total_earnings - total_earnings_shirts
  let num_shoes_sold := earnings_from_shoes / price_shoes
  num_shoes_sold

theorem sab_dane_sold_6_pairs_of_shoes :
  num_shoes_sold 3 18 2 27 = 6 :=
by
  sorry

end num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l137_137716


namespace find_k_l137_137832

theorem find_k (x y k : ℝ) (h1 : 2 * x + y = 4 * k) (h2 : x - y = k) (h3 : x + 2 * y = 12) : k = 4 :=
sorry

end find_k_l137_137832


namespace sum_of_squares_s_comp_r_l137_137506

def r (x : ℝ) : ℝ := x^2 - 4
def s (x : ℝ) : ℝ := -|x + 1|
def s_comp_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_squares_s_comp_r :
  (s_comp_r (-4))^2 + (s_comp_r (-3))^2 + (s_comp_r (-2))^2 + (s_comp_r (-1))^2 +
  (s_comp_r 0)^2 + (s_comp_r 1)^2 + (s_comp_r 2)^2 + (s_comp_r 3)^2 + (s_comp_r 4)^2 = 429 :=
by
  sorry

end sum_of_squares_s_comp_r_l137_137506


namespace mean_first_set_l137_137597

noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length

theorem mean_first_set (x : ℚ) (h : mean [128, 255, 511, 1023, x] = 423) :
  mean [28, x, 42, 78, 104] = 90 :=
sorry

end mean_first_set_l137_137597


namespace g_g_has_two_distinct_real_roots_l137_137700

-- Defining the function g
def g (x c : ℝ) : ℝ := x^2 + 2*x + c^2

-- Stating the main theorem
theorem g_g_has_two_distinct_real_roots (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g (g x₁ c) c = 0 ∧ g (g x₂ c) c = 0) ↔ c = 1 ∨ c = -1 := 
sorry

end g_g_has_two_distinct_real_roots_l137_137700


namespace min_value_expression_l137_137713

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end min_value_expression_l137_137713


namespace jessica_deposit_fraction_l137_137844

theorem jessica_deposit_fraction (init_balance withdraw_amount final_balance : ℝ)
  (withdraw_fraction remaining_fraction deposit_fraction : ℝ) :
  remaining_fraction = withdraw_fraction - (2/5) → 
  init_balance * withdraw_fraction = init_balance - withdraw_amount →
  init_balance * remaining_fraction + deposit_fraction * (init_balance * remaining_fraction) = final_balance →
  init_balance = 500 →
  final_balance = 450 →
  withdraw_amount = 200 →
  remaining_fraction = (3/5) →
  deposit_fraction = 1/2 :=
by
  intros hr hw hrb hb hf hwamount hr_remain
  sorry

end jessica_deposit_fraction_l137_137844


namespace x_plus_p_l137_137675

theorem x_plus_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 :=
by
  sorry

end x_plus_p_l137_137675


namespace problem_stmt_l137_137394

variable (a b : ℝ)

theorem problem_stmt (ha : a > 0) (hb : b > 0) (a_plus_b : a + b = 2):
  3 * a^2 + b^2 ≥ 3 ∧ 4 / (a + 1) + 1 / b ≥ 3 := by
  sorry

end problem_stmt_l137_137394


namespace football_cost_correct_l137_137706

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l137_137706


namespace sequence_term_37_l137_137330

theorem sequence_term_37 (n : ℕ) (h_pos : 0 < n) (h_eq : 3 * n + 1 = 37) : n = 12 :=
by
  sorry

end sequence_term_37_l137_137330


namespace ratio_proof_l137_137539

variables {F : Type*} [Field F] 
variables (w x y z : F)

theorem ratio_proof 
  (h1 : w / x = 4 / 3) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
by sorry

end ratio_proof_l137_137539


namespace age_of_person_A_l137_137884

-- Definitions corresponding to the conditions
variables (x y z : ℕ)
axiom sum_of_ages : x + y = 70
axiom age_difference_A_B : x - z = y
axiom age_difference_B_A_half : y - z = x / 2

-- The proof statement that needs to be proved
theorem age_of_person_A : x = 42 := by 
  -- This is where the proof would go
  sorry

end age_of_person_A_l137_137884


namespace length_BC_fraction_of_AD_l137_137860

-- Define variables and conditions
variables (x y : ℝ)
variable (h1 : 4 * x = 8 * y) -- given: length of AD from both sides
variable (h2 : 3 * x) -- AB = 3 * BD
variable (h3 : 7 * y) -- AC = 7 * CD

-- State the goal to prove
theorem length_BC_fraction_of_AD (x y : ℝ) (h1 : 4 * x = 8 * y) :
  (y / (4 * x)) = 1 / 8 := by
  sorry

end length_BC_fraction_of_AD_l137_137860


namespace smallest_time_for_horses_l137_137315
-- Import the necessary libraries

-- Definition for the problem statement in Lean
theorem smallest_time_for_horses :
  ∃ T > 0, (T = 72) ∧ ∃ horses : Finset ℕ, horses.card ≥ 8 ∧ ∀ k ∈ horses, T % k = 0 :=
begin
  sorry
end

end smallest_time_for_horses_l137_137315


namespace vanya_speed_problem_l137_137743

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l137_137743


namespace square_area_correct_l137_137070

noncomputable def square_area : ℝ :=
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length

theorem square_area_correct :
  let f : ℝ → ℝ := λ x, x^2 + 4 * x + 3
  let y_val : ℝ := 7
  let x1 : ℝ := -2 - 2 * Real.sqrt 2
  let x2 : ℝ := -2 + 2 * Real.sqrt 2
  let side_length := x2 - x1
  side_length * side_length = 32 := by
  sorry

end square_area_correct_l137_137070


namespace stock_percent_change_l137_137781

-- define initial value of stock
def initial_stock_value (x : ℝ) := x

-- define value after first day's decrease
def value_after_day_one (x : ℝ) := 0.85 * x

-- define value after second day's increase
def value_after_day_two (x : ℝ) := 1.25 * value_after_day_one x

-- Theorem stating the overall percent change is 6.25%
theorem stock_percent_change (x : ℝ) (h : x > 0) :
  ((value_after_day_two x - initial_stock_value x) / initial_stock_value x) * 100 = 6.25 := by sorry

end stock_percent_change_l137_137781


namespace length_of_ribbon_l137_137460

theorem length_of_ribbon (perimeter : ℝ) (sides : ℕ) (h1 : perimeter = 42) (h2 : sides = 6) : (perimeter / sides) = 7 :=
by {
  sorry
}

end length_of_ribbon_l137_137460


namespace quadratic_equation_completes_to_square_l137_137578

theorem quadratic_equation_completes_to_square :
  ∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x + 2)^2 = 2 :=
by
  intro x
  intro h
  sorry

end quadratic_equation_completes_to_square_l137_137578


namespace smallest_x_l137_137838

theorem smallest_x (a b x : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (pos_x : x > 0) : x = 200000 := sorry

end smallest_x_l137_137838


namespace problem_a_problem_b_l137_137171

section ProblemA

variable (x : ℝ)

theorem problem_a :
  x ≠ 0 ∧ x ≠ -3/8 ∧ x ≠ 3/7 →
  2 + 5 / (4 * x) - 15 / (4 * x * (8 * x + 3)) = 2 * (7 * x + 1) / (7 * x - 3) →
  x = 9 := by
  sorry

end ProblemA

section ProblemB

variable (x : ℝ)

theorem problem_b :
  x ≠ 0 →
  2 / x + 1 / x^2 - (7 + 10 * x) / (x^2 * (x^2 + 7)) = 2 / (x + 3 / (x + 4 / x)) →
  x = 4 := by
  sorry

end ProblemB

end problem_a_problem_b_l137_137171


namespace successive_numbers_product_2652_l137_137029

theorem successive_numbers_product_2652 (n : ℕ) (h : n * (n + 1) = 2652) : n = 51 :=
sorry

end successive_numbers_product_2652_l137_137029


namespace average_score_l137_137791

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l137_137791


namespace Thabo_harcdover_nonfiction_books_l137_137594

theorem Thabo_harcdover_nonfiction_books 
  (H P F : ℕ)
  (h1 : P = H + 20)
  (h2 : F = 2 * P)
  (h3 : H + P + F = 180) : 
  H = 30 :=
by
  sorry

end Thabo_harcdover_nonfiction_books_l137_137594


namespace Part1_Answer_Part2_Answer_l137_137499

open Nat

-- Definitions for Part 1
def contingencyTable : Type := {
  boysCoord : Nat,
  boysIneq : Nat,
  girlsCoord : Nat,
  girlsIneq : Nat
}
def totalStudents (table : contingencyTable) : Nat :=
  table.boysCoord + table.boysIneq + table.girlsCoord + table.girlsIneq

def chiSquareStatistic (table : contingencyTable) : Real :=
  let n := totalStudents table
  let a := table.boysCoord
  let b := table.boysIneq
  let c := table.girlsCoord
  let d := table.girlsIneq
  n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def isPreferenceRelatedToGender (table : contingencyTable) : Prop :=
  chiSquareStatistic table > 3.841

theorem Part1_Answer :
  isPreferenceRelatedToGender { boysCoord := 15, boysIneq := 25, girlsCoord := 20, girlsIneq := 10 } :=
  -- Sorry will be replaced with the proof
  sorry

-- Definitions for Part 2
def probabilityDist (table : contingencyTable) (stratified_boys : Nat) (total_selected : Nat) (xi : Nat) : Real :=
  if xi = 0 then 4 / 35
  else if xi = 1 then 18 / 35
  else if xi = 2 then 12 / 35
  else if xi = 3 then 1 / 35
  else 0

def expectedValueXi : Real :=
  0 * (4 / 35) + 1 * (18 / 35) + 2 * (12 / 35) + 3 * (1 / 35)

def part2ExpectedValueProof : Real :=
  expectedValueXi

theorem Part2_Answer :
  part2ExpectedValueProof = 9 / 7 :=
  sorry

end Part1_Answer_Part2_Answer_l137_137499


namespace set_inclusion_l137_137818

def setM : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}

def setN : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 2) + (Real.pi / 4)}

def setP : Set ℝ := {a | ∃ k : ℤ, a = (k * Real.pi / 2) + (Real.pi / 4)}

theorem set_inclusion : setP ⊆ setN ∧ setN ⊆ setM := by
  sorry

end set_inclusion_l137_137818


namespace log_sqrt10_1000sqrt10_l137_137796

theorem log_sqrt10_1000sqrt10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 := sorry

end log_sqrt10_1000sqrt10_l137_137796


namespace min_value_of_f_l137_137802

-- Given function:
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x)

-- The minimum value of the function:
theorem min_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -6480.25 :=
by
  have h : ∀ y : ℝ, f y = y^4 - 289 * y^2 + 14400 := sorry
  have h_min : ∀ y : ℝ, (y^4 - 289 * y^2 + 14400) = (y^2 - 144.5)^2 - 6480.25 := sorry
  use 0 -- Since we just need existence, setting x to 0 for the purpose of example
  intro y
  split
  . -- Prove f x ≤ f y (which should simplify to show min at y^2 = 144.5)
    sorry
  . -- Prove f x = -6480.25
    admit
  -- These are place-holders showing where each proof part would be

end min_value_of_f_l137_137802


namespace ticket_sales_amount_theater_collected_50_dollars_l137_137076

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l137_137076


namespace probability_all_white_is_correct_l137_137314

-- Define the total number of balls
def total_balls : ℕ := 25

-- Define the number of white balls
def white_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 15

-- Define the number of balls drawn
def balls_drawn : ℕ := 4

-- Define combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to choose 4 balls from 25
def total_ways : ℕ := C total_balls balls_drawn

-- Ways to choose 4 white balls from 10 white balls
def white_ways : ℕ := C white_balls balls_drawn

-- Probability that all 4 drawn balls are white
def prob_all_white : ℚ := white_ways / total_ways

theorem probability_all_white_is_correct :
  prob_all_white = (3 : ℚ) / 181 := by
  -- Proof statements go here
  sorry

end probability_all_white_is_correct_l137_137314


namespace problem_1_problem_2_l137_137536

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1
theorem problem_1 (x : ℝ) : (∀ x, f x (-2) > 5) ↔ (x < -4 / 3 ∨ x > 2) :=
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) : (∀ x, f x a ≤ a * |x + 3|) → (a ≥ 1 / 2) :=
  sorry

end problem_1_problem_2_l137_137536


namespace final_movie_length_l137_137915

-- Definitions based on conditions
def original_movie_length : ℕ := 60
def cut_scene_length : ℕ := 3

-- Theorem statement proving the final length of the movie
theorem final_movie_length : original_movie_length - cut_scene_length = 57 :=
by
  -- The proof will go here
  sorry

end final_movie_length_l137_137915


namespace ones_digit_of_six_power_l137_137322

theorem ones_digit_of_six_power (n : ℕ) (hn : n ≥ 1) : (6 ^ n) % 10 = 6 :=
by
  sorry

example : (6 ^ 34) % 10 = 6 :=
by
  have h : 34 ≥ 1 := by norm_num
  exact ones_digit_of_six_power 34 h

end ones_digit_of_six_power_l137_137322


namespace intersection_points_l137_137595

theorem intersection_points : 
  (∃ x : ℝ, y = -2 * x + 4 ∧ y = 0 ∧ (x, y) = (2, 0)) ∧
  (∃ y : ℝ, y = -2 * 0 + 4 ∧ (0, y) = (0, 4)) :=
by
  sorry

end intersection_points_l137_137595


namespace sum_of_digits_6608_condition_l137_137405

theorem sum_of_digits_6608_condition :
  ∀ n1 n2 : ℕ, (6 * 1000 + n1 * 100 + n2 * 10 + 8) % 236 = 0 → n1 + n2 = 6 :=
by 
  intros n1 n2 h
  -- This is where the proof would go. Since we're not proving it, we skip it with "sorry".
  sorry

end sum_of_digits_6608_condition_l137_137405


namespace sequence_general_term_l137_137192

-- Define the sequence based on the given conditions
def seq (n : ℕ) : ℚ := if n = 0 then 1 else (n : ℚ) / (2 * n - 1)

theorem sequence_general_term (n : ℕ) :
  seq (n + 1) = (n + 1) / (2 * (n + 1) - 1) :=
by
  sorry

end sequence_general_term_l137_137192


namespace ratio_sheila_purity_l137_137717

theorem ratio_sheila_purity (rose_share : ℕ) (total_rent : ℕ) (purity_share : ℕ) (sheila_share : ℕ) 
  (h1 : rose_share = 1800) 
  (h2 : total_rent = 5400) 
  (h3 : rose_share = 3 * purity_share)
  (h4 : total_rent = purity_share + rose_share + sheila_share) : 
  sheila_share / purity_share = 5 :=
by
  -- Proof will be here
  sorry

end ratio_sheila_purity_l137_137717


namespace boxes_left_l137_137857

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l137_137857


namespace second_reduction_percentage_is_4_l137_137051

def original_price := 500
def first_reduction_percent := 5 / 100
def total_reduction := 44

def first_reduction := first_reduction_percent * original_price
def price_after_first_reduction := original_price - first_reduction
def second_reduction := total_reduction - first_reduction
def second_reduction_percent := (second_reduction / price_after_first_reduction) * 100

theorem second_reduction_percentage_is_4 :
  second_reduction_percent = 4 := by
  sorry

end second_reduction_percentage_is_4_l137_137051


namespace same_answer_l137_137246

structure Person :=
(name : String)
(tellsTruth : Bool)

def Fedya : Person :=
{ name := "Fedya",
  tellsTruth := true }

def Vadim : Person :=
{ name := "Vadim",
  tellsTruth := false }

def question (p : Person) (q : String) : Bool :=
if p.tellsTruth then q = p.name else q ≠ p.name

theorem same_answer (q : String) :
  (question Fedya q = question Vadim q) :=
sorry

end same_answer_l137_137246


namespace isosceles_triangle_base_length_l137_137959

noncomputable def length_of_base (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : ℝ :=
  (12 - 2 * a) / 2

theorem isosceles_triangle_base_length (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : length_of_base a b h_isosceles h_side h_perimeter = 4.5 :=
sorry

end isosceles_triangle_base_length_l137_137959


namespace honey_barrel_problem_l137_137496

theorem honey_barrel_problem
  (x y : ℝ)
  (h1 : x + y = 56)
  (h2 : x / 2 + y = 34) :
  x = 44 ∧ y = 12 :=
by
  sorry

end honey_barrel_problem_l137_137496


namespace nonneg_int_values_of_fraction_condition_l137_137328

theorem nonneg_int_values_of_fraction_condition (n : ℕ) : (∃ k : ℤ, 30 * n + 2 = k * (12 * n + 1)) → n = 0 := by
  sorry

end nonneg_int_values_of_fraction_condition_l137_137328


namespace vanya_speed_increased_by_4_l137_137751

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l137_137751


namespace no_even_is_prime_equiv_l137_137873

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l137_137873


namespace distance_to_station_is_6_l137_137829

noncomputable def distance_man_walks (walking_speed1 walking_speed2 time_diff: ℝ) : ℝ :=
  let D := (time_diff * walking_speed1 * walking_speed2) / (walking_speed1 - walking_speed2)
  D

theorem distance_to_station_is_6 :
  distance_man_walks 5 6 (12 / 60) = 6 :=
by
  sorry

end distance_to_station_is_6_l137_137829


namespace divisibility_condition_l137_137566

theorem divisibility_condition (a p q : ℕ) (hp : p > 0) (ha : a > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ↔ p ∣ a^q) :=
sorry

end divisibility_condition_l137_137566


namespace point_in_third_quadrant_l137_137677

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : 
  (-b < 0) ∧ (a < 0) ∧ (-b > a) :=
by
  sorry

end point_in_third_quadrant_l137_137677


namespace train_length_l137_137933

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l137_137933


namespace at_least_two_equal_l137_137111

theorem at_least_two_equal (x y z : ℝ) (h : x / y + y / z + z / x = z / y + y / x + x / z) : 
  x = y ∨ y = z ∨ z = x := 
  sorry

end at_least_two_equal_l137_137111


namespace art_department_probability_l137_137217

theorem art_department_probability : 
  let students := {s1, s2, s3, s4} 
  let first_grade := {s1, s2}
  let second_grade := {s3, s4}
  let total_pairs := { (x, y) | x ∈ students ∧ y ∈ students ∧ x < y }.to_finset.card
  let diff_grade_pairs := { (x, y) | x ∈ first_grade ∧ y ∈ second_grade ∨ x ∈ second_grade ∧ y ∈ first_grade}.to_finset.card
  (diff_grade_pairs / total_pairs) = 2 / 3 := 
by 
  sorry

end art_department_probability_l137_137217


namespace digit_difference_l137_137208

theorem digit_difference (X Y : ℕ) (h1 : 10 * X + Y - (10 * Y + X) = 36) : X - Y = 4 := by
  sorry

end digit_difference_l137_137208


namespace probability_of_correct_match_l137_137229

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_possible_arrangements : ℕ :=
  factorial 4

def correct_arrangements : ℕ :=
  1

def probability_correct_match : ℚ :=
  correct_arrangements / total_possible_arrangements

theorem probability_of_correct_match : probability_correct_match = 1 / 24 :=
by
  -- Proof is omitted
  sorry

end probability_of_correct_match_l137_137229


namespace find_integer_solutions_l137_137513

theorem find_integer_solutions :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    (a * b - 2 * c * d = 3) ∧ (a * c + b * d = 1) } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end find_integer_solutions_l137_137513


namespace fourth_power_cube_third_smallest_prime_l137_137897

theorem fourth_power_cube_third_smallest_prime :
  (let p := 5 in (p^3)^4 = 244140625) :=
by
  sorry

end fourth_power_cube_third_smallest_prime_l137_137897


namespace total_spent_correct_l137_137543

def cost_gifts : ℝ := 561.00
def cost_giftwrapping : ℝ := 139.00
def total_spent : ℝ := cost_gifts + cost_giftwrapping

theorem total_spent_correct : total_spent = 700.00 := by
  sorry

end total_spent_correct_l137_137543


namespace evaluate_g_at_3_l137_137279

def g (x : ℝ) := 3 * x ^ 4 - 5 * x ^ 3 + 4 * x ^ 2 - 7 * x + 2

theorem evaluate_g_at_3 : g 3 = 125 :=
by
  -- Proof omitted for this exercise.
  sorry

end evaluate_g_at_3_l137_137279


namespace B_finishes_remaining_work_in_3_days_l137_137041

theorem B_finishes_remaining_work_in_3_days
  (A_works_in : ℕ)
  (B_works_in : ℕ)
  (work_days_together : ℕ)
  (A_leaves : A_works_in = 4)
  (B_leaves : B_works_in = 10)
  (work_days : work_days_together = 2) :
  ∃ days_remaining : ℕ, days_remaining = 3 :=
by
  sorry

end B_finishes_remaining_work_in_3_days_l137_137041


namespace convex_quadrilateral_max_two_obtuse_l137_137132

theorem convex_quadrilateral_max_two_obtuse (a b c d : ℝ)
  (h1 : a + b + c + d = 360)
  (h2 : a < 180) (h3 : b < 180) (h4 : c < 180) (h5 : d < 180)
  : (∃ A1 A2, a = A1 ∧ b = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ c < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ c < 90) ∨
    (∃ A1 A2, b = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ d < 90) ∨
    (∃ A1 A2, b = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ c < 90) ∨
    (∃ A1 A2, c = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ b < 90) ∨
    (¬∃ x y z, (x > 90) ∧ (y > 90) ∧ (z > 90) ∧ x + y + z ≤ 360) := sorry

end convex_quadrilateral_max_two_obtuse_l137_137132


namespace unique_function_satisfies_sum_zero_l137_137406

theorem unique_function_satisfies_sum_zero 
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x^3) = (f x)^3)
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2) : 
  f 0 + f 1 + f (-1) = 0 :=
sorry

end unique_function_satisfies_sum_zero_l137_137406


namespace blueberry_picking_relationship_l137_137324

theorem blueberry_picking_relationship (x : ℝ) (hx : x > 10) : 
  let y1 := 60 + 18 * x
  let y2 := 150 + 15 * x
  in y1 = 60 + 18 * x ∧ y2 = 150 + 15 * x := 
by {
  sorry
}

end blueberry_picking_relationship_l137_137324


namespace cat_clothing_probability_l137_137017

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability_l137_137017


namespace selling_price_of_cycle_l137_137920

theorem selling_price_of_cycle (cost_price : ℕ) (gain_percent : ℕ) (cost_price_eq : cost_price = 1500) (gain_percent_eq : gain_percent = 8) :
  ∃ selling_price : ℕ, selling_price = 1620 := 
by
  sorry

end selling_price_of_cycle_l137_137920


namespace find_correct_r_l137_137534

noncomputable def ellipse_tangent_circle_intersection : Prop :=
  ∃ (E F : ℝ × ℝ) (r : ℝ), E ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧
                             F ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧ 
                             (E ≠ F) ∧
                             ((E.1 - 2)^2 + (E.2 - 3/2)^2 = r^2) ∧
                             ((F.1 - 2)^2 + (F.2 - 3/2)^2 = r^2) ∧
                             r = (Real.sqrt 37) / 37

theorem find_correct_r : ellipse_tangent_circle_intersection :=
sorry

end find_correct_r_l137_137534


namespace students_passed_both_tests_l137_137618

theorem students_passed_both_tests
  (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ)
  (h1 : n = 100) 
  (h2 : A = 60) 
  (h3 : B = 40) 
  (h4 : C = 20) :
  A + B - ((n - C) - (A + B - n)) = 20 :=
by
  sorry

end students_passed_both_tests_l137_137618


namespace silk_dyeing_total_correct_l137_137021

open Real

theorem silk_dyeing_total_correct :
  let green := 61921
  let pink := 49500
  let blue := 75678
  let yellow := 34874.5
  let total_without_red := green + pink + blue + yellow
  let red := 0.10 * total_without_red
  let total_with_red := total_without_red + red
  total_with_red = 245270.85 :=
by
  sorry

end silk_dyeing_total_correct_l137_137021


namespace pythagorean_consecutive_numbers_unique_l137_137662

theorem pythagorean_consecutive_numbers_unique :
  ∀ (x : ℕ), (x + 2) * (x + 2) = (x + 1) * (x + 1) + x * x → x = 3 :=
by
  sorry 

end pythagorean_consecutive_numbers_unique_l137_137662


namespace prob_iff_eq_l137_137268

noncomputable def A (m : ℝ) : Set ℝ := { x | x^2 + m * x + 2 ≥ 0 ∧ x ≥ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { y | ∃ x, x ∈ A m ∧ y = Real.sqrt (x^2 + m * x + 2) }

theorem prob_iff_eq (m : ℝ) : (A m = { y | ∃ x, x ^ 2 + m * x + 2 = y ^ 2 ∧ x ≥ 0 } ↔ m = -2 * Real.sqrt 2) :=
by
  sorry

end prob_iff_eq_l137_137268


namespace prove_inequality_l137_137955

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (55 * Real.pi / 180)

theorem prove_inequality : c > b ∧ b > a :=
by
  -- Proof goes here
  sorry

end prove_inequality_l137_137955


namespace jacket_total_selling_price_l137_137226

theorem jacket_total_selling_price :
  let original_price := 120
  let discount_rate := 0.30
  let tax_rate := 0.08
  let processing_fee := 5
  let discounted_price := original_price * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  let total_price := discounted_price + tax + processing_fee
  total_price = 95.72 := by
  sorry

end jacket_total_selling_price_l137_137226


namespace complement_intersection_l137_137965

open Set

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (M_def : M = {2, 3})
variable (N_def : N = {1, 4})

theorem complement_intersection (U M N : Set ℕ) (U_def : U = {1, 2, 3, 4, 5, 6}) (M_def : M = {2, 3}) (N_def : N = {1, 4}) :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  sorry

end complement_intersection_l137_137965


namespace total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l137_137916

-- Define the conditions
def number_of_bags : ℕ := 9
def vitamins_per_bag : ℚ := 0.2

-- Define the total vitamins in the box
def total_vitamins_in_box : ℚ := number_of_bags * vitamins_per_bag

-- Define the vitamins intake by drinking half a bag
def vitamins_per_half_bag : ℚ := vitamins_per_bag / 2

-- Prove that the total grams of vitamins in the box is 1.8 grams
theorem total_vitamins_in_box_correct : total_vitamins_in_box = 1.8 := by
  sorry

-- Prove that the vitamins intake by drinking half a bag is 0.1 grams
theorem vitamins_per_half_bag_correct : vitamins_per_half_bag = 0.1 := by
  sorry

end total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l137_137916


namespace smallest_number_is_28_l137_137477

theorem smallest_number_is_28 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 30) (h4 : b = 29) (h5 : c = b + 4) : a = 28 :=
by
  sorry

end smallest_number_is_28_l137_137477


namespace ticket_sales_l137_137077

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l137_137077


namespace bob_calories_consumed_l137_137630

theorem bob_calories_consumed 
  (total_slices : ℕ)
  (half_slices : ℕ)
  (calories_per_slice : ℕ) 
  (H1 : total_slices = 8) 
  (H2 : half_slices = total_slices / 2) 
  (H3 : calories_per_slice = 300) : 
  half_slices * calories_per_slice = 1200 := 
by 
  sorry

end bob_calories_consumed_l137_137630


namespace problem_statement_l137_137668

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l137_137668


namespace population_at_seven_years_l137_137600

theorem population_at_seven_years (a x : ℕ) (y: ℝ) (h₀: a = 100) (h₁: x = 7) (h₂: y = a * Real.logb 2 (x + 1)):
  y = 300 :=
by
  -- We include the conditions in the theorem statement
  sorry

end population_at_seven_years_l137_137600


namespace positive_n_of_single_solution_l137_137806

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end positive_n_of_single_solution_l137_137806


namespace xiamen_fabric_production_l137_137486

theorem xiamen_fabric_production:
  (∃ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3))) ∧
  (∀ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3)) →
    x = 360 ∧ y = 240 ∧ y / 3 = 240) := 
by
  sorry

end xiamen_fabric_production_l137_137486


namespace prob_at_least_3_correct_l137_137191

-- Define the probability of one patient being cured
def prob_cured : ℝ := 0.9

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of exactly 3 out of 4 patients being cured
def prob_exactly_3 : ℝ :=
  binomial 4 3 * prob_cured^3 * (1 - prob_cured)

-- Define the probability of all 4 patients being cured
def prob_all_4 : ℝ :=
  prob_cured^4

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_3 : ℝ :=
  prob_exactly_3 + prob_all_4

-- The theorem to prove
theorem prob_at_least_3_correct : prob_at_least_3 = 0.9477 :=
  by
  sorry

end prob_at_least_3_correct_l137_137191


namespace inscribed_polygon_sides_l137_137431

-- We start by defining the conditions of the problem in Lean.
def radius := 1
def side_length_condition (n : ℕ) : Prop :=
  1 < 2 * Real.sin (Real.pi / n) ∧ 2 * Real.sin (Real.pi / n) < Real.sqrt 2

-- Now we state the main theorem.
theorem inscribed_polygon_sides (n : ℕ) (h1 : side_length_condition n) : n = 5 :=
  sorry

end inscribed_polygon_sides_l137_137431


namespace anne_carries_16point5_kg_l137_137353

theorem anne_carries_16point5_kg :
  let w1 := 2
  let w2 := 1.5 * w1
  let w3 := 2 * w1
  let w4 := w1 + w2
  let w5 := (w1 + w2) / 2
  w1 + w2 + w3 + w4 + w5 = 16.5 :=
by {
  sorry
}

end anne_carries_16point5_kg_l137_137353


namespace felix_trees_per_sharpening_l137_137101

theorem felix_trees_per_sharpening (dollars_spent : ℕ) (cost_per_sharpen : ℕ) (trees_chopped : ℕ) 
  (h1 : dollars_spent = 35) (h2 : cost_per_sharpen = 5) (h3 : trees_chopped ≥ 91) :
  (91 / (35 / 5)) = 13 := 
by 
  sorry

end felix_trees_per_sharpening_l137_137101


namespace ben_initial_marbles_l137_137940

theorem ben_initial_marbles (B : ℕ) (John_initial_marbles : ℕ) (H1 : John_initial_marbles = 17) (H2 : John_initial_marbles + B / 2 = B / 2 + B / 2 + 17) : B = 34 := by
  sorry

end ben_initial_marbles_l137_137940


namespace post_tax_income_correct_l137_137776

noncomputable def worker_a_pre_tax_income : ℝ :=
  80 * 30 + 50 * 30 * 1.20 + 35 * 30 * 1.50 + (35 * 30 * 1.50) * 0.05

noncomputable def worker_b_pre_tax_income : ℝ :=
  90 * 25 + 45 * 25 * 1.25 + 40 * 25 * 1.45 + (40 * 25 * 1.45) * 0.05

noncomputable def worker_c_pre_tax_income : ℝ :=
  70 * 35 + 40 * 35 * 1.15 + 60 * 35 * 1.60 + (60 * 35 * 1.60) * 0.05

noncomputable def worker_a_post_tax_income : ℝ := 
  worker_a_pre_tax_income * 0.85 - 200

noncomputable def worker_b_post_tax_income : ℝ := 
  worker_b_pre_tax_income * 0.82 - 250

noncomputable def worker_c_post_tax_income : ℝ := 
  worker_c_pre_tax_income * 0.80 - 300

theorem post_tax_income_correct :
  worker_a_post_tax_income = 4775.69 ∧ 
  worker_b_post_tax_income = 3996.57 ∧ 
  worker_c_post_tax_income = 5770.40 :=
by {
  sorry
}

end post_tax_income_correct_l137_137776


namespace hexagon_circle_ratio_correct_l137_137725

noncomputable def hexagon_circle_area_ratio (s r : ℝ) (h : 6 * s = 2 * π * r) : ℝ :=
  let A_hex := (3 * Real.sqrt 3 / 2) * s^2
  let A_circ := π * r^2
  (A_hex / A_circ)

theorem hexagon_circle_ratio_correct (s r : ℝ) (h : 6 * s = 2 * π * r) :
    hexagon_circle_area_ratio s r h = (π * Real.sqrt 3 / 6) :=
sorry

end hexagon_circle_ratio_correct_l137_137725


namespace catriona_total_fish_eq_44_l137_137361

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end catriona_total_fish_eq_44_l137_137361


namespace coefficients_proof_l137_137179

-- Define the given quadratic equation
def quadratic_eq := ∀ x : ℝ, x^2 + 2 = 3x

-- Define the standard form coefficients
def coefficients_quadratic (a b c : ℝ) :=
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2 = 3x

-- The proof problem
theorem coefficients_proof : coefficients_quadratic 1 (-3) 2 :=
by
  sorry

end coefficients_proof_l137_137179


namespace total_amount_spent_on_cookies_l137_137647

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies_l137_137647


namespace discount_difference_is_24_l137_137027

-- Definitions based on conditions
def smartphone_price : ℝ := 800
def single_discount_rate : ℝ := 0.25
def first_successive_discount_rate : ℝ := 0.20
def second_successive_discount_rate : ℝ := 0.10

-- Definitions of discounted prices
def single_discount_price (p : ℝ) (d1 : ℝ) : ℝ := p * (1 - d1)
def successive_discount_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ := 
  let intermediate_price := p * (1 - d1) 
  intermediate_price * (1 - d2)

-- Calculate the difference between the two final prices
def price_difference (p : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (single_discount_price p d1) - (successive_discount_price p d2 d3)

theorem discount_difference_is_24 :
  price_difference smartphone_price single_discount_rate first_successive_discount_rate second_successive_discount_rate = 24 := 
sorry

end discount_difference_is_24_l137_137027


namespace smallest_number_with_unique_digits_summing_to_32_exists_l137_137380

theorem smallest_number_with_unique_digits_summing_to_32_exists : 
  ∃ n : ℕ, n / 10000 < 10 ∧ (n % 10 ≠ (n / 10) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 1000) % 10 ≠ (n / 10000) % 10) ∧ 
  (n % 10 + (n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10 + (n / 10000) % 10 = 32) := 
sorry

end smallest_number_with_unique_digits_summing_to_32_exists_l137_137380


namespace max_value_of_f_l137_137364

noncomputable def f (t : ℝ) : ℝ := ((2^(t+1) - 4*t) * t) / (16^t)

theorem max_value_of_f : ∃ t : ℝ, ∀ u : ℝ, f u ≤ f t ∧ f t = 1 / 16 := by
  sorry

end max_value_of_f_l137_137364


namespace evening_water_usage_is_6_l137_137333

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end evening_water_usage_is_6_l137_137333


namespace tuesday_rainfall_l137_137082

-- Condition: average rainfall for the whole week is 3 cm
def avg_rainfall_week : ℝ := 3

-- Condition: number of days in a week
def days_in_week : ℕ := 7

-- Condition: total rainfall for the week
def total_rainfall_week : ℝ := avg_rainfall_week * days_in_week

-- Condition: total rainfall is twice the rainfall on Tuesday
def total_rainfall_equals_twice_T (T : ℝ) : ℝ := 2 * T

-- Theorem: Prove that the rainfall on Tuesday is 10.5 cm
theorem tuesday_rainfall : ∃ T : ℝ, total_rainfall_equals_twice_T T = total_rainfall_week ∧ T = 10.5 := by
  sorry

end tuesday_rainfall_l137_137082


namespace herd_total_cows_l137_137334

noncomputable def total_cows (n : ℕ) : Prop :=
  let fraction_first_son := 1 / 3
  let fraction_second_son := 1 / 5
  let fraction_third_son := 1 / 9
  let fraction_combined := fraction_first_son + fraction_second_son + fraction_third_son
  let fraction_fourth_son := 1 - fraction_combined
  let cows_fourth_son := 11
  fraction_fourth_son * n = cows_fourth_son

theorem herd_total_cows : ∃ n : ℕ, total_cows n ∧ n = 31 :=
by
  existsi 31
  sorry

end herd_total_cows_l137_137334


namespace area_quadrilateral_extension_l137_137159

variable (A B C D B1 C1 D1 A1 : Type) 
          [AddCommGroup A] [TopologicalSpace A]
          [AddCommGroup B] [TopologicalSpace B]
          [AddCommGroup C] [TopologicalSpace C]
          [AddCommGroup D] [TopologicalSpace D]
          [AddCommGroup B1] [TopologicalSpace B1]
          [AddCommGroup C1] [TopologicalSpace C1]
          [AddCommGroup D1] [TopologicalSpace D1]
          [AddCommGroup A1] [TopologicalSpace A1]

-- Given conditions
def convex_quadrilateral (ABCD : Set A) : Prop := sorry
def point_extension (P Q PQ1 : Set A) (cond : P.length = Q.length) : Prop := sorry

-- Hypotheses
axiom h1 : convex_quadrilateral {A, B, C, D}
axiom h2 : point_extension A B B1
axiom h3 : point_extension B C C1
axiom h4 : point_extension C D D1
axiom h5 : point_extension D A A1

-- Lemma
theorem area_quadrilateral_extension :
  ∀ (S ABCD : ℝ), S ABCD > 0 → S (A1 B1 C1 D1) = 5 * S ABCD := 
sorry

end area_quadrilateral_extension_l137_137159


namespace frequency_interval_20_to_inf_l137_137113

theorem frequency_interval_20_to_inf (sample_size : ℕ)
  (freq_5_10 : ℕ) (freq_10_15 : ℕ) (freq_15_20 : ℕ)
  (freq_20_25 : ℕ) (freq_25_30 : ℕ) (freq_30_35 : ℕ) :
  sample_size = 35 ∧
  freq_5_10 = 5 ∧
  freq_10_15 = 12 ∧
  freq_15_20 = 7 ∧
  freq_20_25 = 5 ∧
  freq_25_30 = 4 ∧
  freq_30_35 = 2 →
  (1 - (freq_5_10 + freq_10_15 + freq_15_20 : ℕ) / (sample_size : ℕ) : ℝ) = 11 / 35 :=
by sorry

end frequency_interval_20_to_inf_l137_137113


namespace find_positive_integers_divisors_l137_137388

theorem find_positive_integers_divisors :
  ∃ n_list : List ℕ, 
    (∀ n ∈ n_list, n > 0 ∧ (n * (n + 1)) / 2 ∣ 10 * n) ∧ n_list.length = 5 :=
sorry

end find_positive_integers_divisors_l137_137388


namespace commutating_matrices_l137_137146

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end commutating_matrices_l137_137146


namespace solution_set_l137_137013

-- Define the system of equations
def system_of_equations (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2 * y + 4 ∧
  (2 * x)^2 - (y + 1)^2 = 3 ∧
  (2 * x - (y + 1)) * (2 * x + (y + 1)) = 3

-- Prove that the solutions to the system are the set we expect
theorem solution_set : 
  { (x, y) : ℤ × ℤ | system_of_equations x y } = { (1, 0), (1, -2), (-1, 0), (-1, -2) } := 
by 
  -- Proof omitted
  sorry

end solution_set_l137_137013


namespace sum_of_consecutive_integers_l137_137473

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 := by
  sorry

end sum_of_consecutive_integers_l137_137473


namespace time_to_watch_all_episodes_l137_137289

theorem time_to_watch_all_episodes 
    (n_seasons : ℕ) (episodes_per_season : ℕ) (last_season_extra_episodes : ℕ) (hours_per_episode : ℚ)
    (h1 : n_seasons = 9)
    (h2 : episodes_per_season = 22)
    (h3 : last_season_extra_episodes = 4)
    (h4 : hours_per_episode = 0.5) :
    n_seasons * episodes_per_season + (episodes_per_season + last_season_extra_episodes) * hours_per_episode = 112 :=
by
  sorry

end time_to_watch_all_episodes_l137_137289


namespace max_value_squared_of_ratio_l137_137851

-- Definition of positive real numbers with given conditions
variables (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 

-- Main statement
theorem max_value_squared_of_ratio 
  (h_ge : a ≥ b)
  (h_eq_1 : a ^ 2 + y ^ 2 = b ^ 2 + x ^ 2)
  (h_eq_2 : b ^ 2 + x ^ 2 = (a - x) ^ 2 + (b + y) ^ 2)
  (h_range_x : 0 ≤ x ∧ x < a)
  (h_range_y : 0 ≤ y ∧ y < b)
  (h_additional_x : x = a - 2 * b)
  (h_additional_y : y = b / 2) : 
  (a / b) ^ 2 = 4 / 9 := 
sorry

end max_value_squared_of_ratio_l137_137851


namespace student_age_is_24_l137_137767

-- Defining the conditions
variables (S M : ℕ)
axiom h1 : M = S + 26
axiom h2 : M + 2 = 2 * (S + 2)

-- The proof statement
theorem student_age_is_24 : S = 24 :=
by
  sorry

end student_age_is_24_l137_137767


namespace quotient_correct_l137_137901

def dividend : ℤ := 474232
def divisor : ℤ := 800
def remainder : ℤ := -968

theorem quotient_correct : (dividend + abs remainder) / divisor = 594 := by
  sorry

end quotient_correct_l137_137901


namespace exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l137_137336

-- Definitions based on the given conditions
def male_students := 3
def female_students := 2
def total_students := male_students + female_students

def at_least_1_male_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ male_students
def all_female_event := ∀ (n : ℕ), n ≤ female_students
def at_least_1_female_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ female_students
def all_male_event := ∀ (n : ℕ), n ≤ male_students
def exactly_1_male_event := ∃ (n : ℕ), n = 1 ∧ n ≤ male_students
def exactly_2_female_event := ∃ (n : ℕ), n = 2 ∧ n ≤ female_students

def mutually_exclusive (e1 e2 : Prop) : Prop := ¬ (e1 ∧ e2)
def complementary (e1 e2 : Prop) : Prop := e1 ∧ ¬ e2 ∨ ¬ e1 ∧ e2

-- Statement of the problem
theorem exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary :
  mutually_exclusive exactly_1_male_event exactly_2_female_event ∧ 
  ¬ complementary exactly_1_male_event exactly_2_female_event :=
by
  sorry

end exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l137_137336


namespace line_equations_l137_137948

theorem line_equations : 
  ∀ (x y : ℝ), (∃ a b c : ℝ, 2 * x + y - 12 = 0 ∨ 2 * x - 5 * y = 0 ∧ (x, y) = (5, 2) ∧ b = 2 * a) :=
by
  sorry

end line_equations_l137_137948


namespace onions_total_l137_137862

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ)
  (h1 : Sara_onions = 4) (h2 : Sally_onions = 5) (h3 : Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 := by
  sorry

end onions_total_l137_137862


namespace boric_acid_solution_l137_137055

theorem boric_acid_solution
  (amount_first_solution: ℝ) (percentage_first_solution: ℝ)
  (amount_second_solution: ℝ) (percentage_second_solution: ℝ)
  (final_amount: ℝ) (final_percentage: ℝ)
  (h1: amount_first_solution = 15)
  (h2: percentage_first_solution = 0.01)
  (h3: amount_second_solution = 15)
  (h4: final_amount = 30)
  (h5: final_percentage = 0.03)
  : percentage_second_solution = 0.05 := 
by
  sorry

end boric_acid_solution_l137_137055


namespace number_of_triangles_l137_137589

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l137_137589


namespace part1_tangent_line_part2_monotonicity_l137_137961

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x ^ 2 - 2 * a * x) * Real.log x - x ^ 2 + 4 * a * x + 1

theorem part1_tangent_line (a : ℝ) (h : a = 0) :
  let e := Real.exp 1
  let f_x := f e 0
  let tangent_line := 4 * e - 3 * e ^ 2 + 1
  tangent_line = 4 * e * (x - e) + f_x :=
sorry

theorem part2_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → f (x) a > 0 ↔ a ≤ 0) ∧
  (∀ x : ℝ, 0 < x → x < a → f (x) a > 0 ↔ 0 < a ∧ a < 1) ∧
  (∀ x : ℝ, 1 < x → x < a → f (x) a < 0 ↔ a > 1) ∧
  (∀ x : ℝ, 0 < x → 1 < x → x < a → f (x) a < 0 ↔ (a > 1)) ∧
  (∀ x : ℝ, x > 1 → f (x) a > 0 ↔ (a < 1)) :=
sorry

end part1_tangent_line_part2_monotonicity_l137_137961


namespace incorrect_expression_l137_137999

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : x / (y - x) ≠ 5 / 2 := 
by
  sorry

end incorrect_expression_l137_137999


namespace train_length_200_04_l137_137934

-- Define the constants
def speed_kmh : ℝ := 60     -- speed in km/h
def time_seconds : ℕ := 12  -- time in seconds

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def speed_ms : ℝ := (speed_kmh * km_to_m) / hr_to_s

-- Define the length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_200_04 : length_of_train = 200.04 := by
  sorry

end train_length_200_04_l137_137934


namespace inequality_solution_set_result_l137_137425

theorem inequality_solution_set_result (a b x : ℝ) :
  (∀ x, a ≤ (3/4) * x^2 - 3 * x + 4 ∧ (3/4) * x^2 - 3 * x + 4 ≤ b) ∧ 
  (∀ x, x ∈ Set.Icc a b ↔ a ≤ x ∧ x ≤ b) →
  a + b = 4 := 
by
  sorry

end inequality_solution_set_result_l137_137425


namespace no_perfect_square_m_in_range_l137_137413

theorem no_perfect_square_m_in_range : 
  ∀ m : ℕ, 4 ≤ m ∧ m ≤ 12 → ¬(∃ k : ℕ, 2 * m^2 + 3 * m + 2 = k^2) := by
sorry

end no_perfect_square_m_in_range_l137_137413


namespace average_absolute_sum_written_as_fraction_l137_137648

def average_absolute_sum_of_permutations : ℚ :=
  let s : ℚ := 
    (2 * (1 + 3 + 6 + 10 + 15 + 21 + 28 + 36 + 45 + 55 + 66)) 
    in (s * 6 / ((12:ℚ).choose 2))

theorem average_absolute_sum_written_as_fraction(p q : ℕ) (hpq : nat.coprime p q) :
  average_absolute_sum_of_permutations = p / q → p + q = 297 :=
  sorry

end average_absolute_sum_written_as_fraction_l137_137648


namespace log_base_sqrt_10_l137_137798

theorem log_base_sqrt_10 :
  log (sqrt 10) (1000 * sqrt 10) = 7 :=
by
  -- Definitions conforming to the problem conditions
  have h1 : sqrt 10 = 10 ^ (1/2) := by sorry
  have h2 : 1000 = 10 ^ 3 := by sorry
  have eq1 : (sqrt 10) ^ 7 = 1000 * sqrt 10 :=
    by rw [h1, h2]; ring
  have eq2 : 1000 * sqrt 10 = 10 ^ (7 / 2) :=
    by rw [h1, h2]; ring

  -- Proof follows from these intermediate steps
  exact log_eq_of_pow_eq (10 ^ (1/2)) (1000 * sqrt 10) 7 eq2 sorry

end log_base_sqrt_10_l137_137798


namespace intersection_eq_l137_137494

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem intersection_eq : M ∩ N = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_eq_l137_137494


namespace remaining_students_correct_l137_137032

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end remaining_students_correct_l137_137032


namespace clown_blew_more_balloons_l137_137869

theorem clown_blew_more_balloons :
  ∀ (initial_balloons final_balloons additional_balloons : ℕ),
    initial_balloons = 47 →
    final_balloons = 60 →
    additional_balloons = final_balloons - initial_balloons →
    additional_balloons = 13 :=
by
  intros initial_balloons final_balloons additional_balloons h1 h2 h3
  sorry

end clown_blew_more_balloons_l137_137869


namespace incoming_class_student_count_l137_137337

theorem incoming_class_student_count (n : ℕ) :
  n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 → n = 418 :=
by
  sorry

end incoming_class_student_count_l137_137337


namespace average_other_students_l137_137221

theorem average_other_students (total_students other_students : ℕ) (mean_score_first : ℕ) 
 (mean_score_class : ℕ) (mean_score_other : ℕ) (h1 : total_students = 20) (h2 : other_students = 10)
 (h3 : mean_score_first = 80) (h4 : mean_score_class = 70) :
 mean_score_other = 60 :=
by
  sorry

end average_other_students_l137_137221


namespace trigonometric_expression_value_l137_137117

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  2 * (Real.sin α)^2 + 4 * Real.sin α * Real.cos α - 9 * (Real.cos α)^2 = 21 / 10 :=
by
  sorry

end trigonometric_expression_value_l137_137117


namespace triangle_angle_B_triangle_area_l137_137540

open Real

theorem triangle_angle_B (A B C a b c : ℝ) (h1 : a + 2 * c = 2 * b * cos A) (h2 : b = 2 * sqrt 3) :
  B = 2 * π / 3 :=
by
  sorry

theorem triangle_area (A B C a b c : ℝ) (h1 : a + 2 * c = 2 * b * cos A) (h2 : b = 2 * sqrt 3)
  (h3 : a + c = 4) (hB : B = 2 * π / 3) :
  (1 / 2) * a * c * sin B = sqrt 3 :=
by
  sorry

end triangle_angle_B_triangle_area_l137_137540


namespace stratified_sampling_male_students_l137_137139

theorem stratified_sampling_male_students (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 900) (h2 : female_students = 0) (h3 : sample_size = 45) : 
  ((total_students - female_students) * sample_size / total_students) = 25 := 
by {
  sorry
}

end stratified_sampling_male_students_l137_137139


namespace height_of_scale_model_eq_29_l137_137783

def empireStateBuildingHeight : ℕ := 1454

def scaleRatio : ℕ := 50

def scaleModelHeight (actualHeight : ℕ) (ratio : ℕ) : ℤ :=
  Int.ofNat actualHeight / ratio

theorem height_of_scale_model_eq_29 : scaleModelHeight empireStateBuildingHeight scaleRatio = 29 :=
by
  -- Proof would go here
  sorry

end height_of_scale_model_eq_29_l137_137783


namespace number_of_valid_n_l137_137386

noncomputable def arithmetic_sum (n : ℕ) := n * (n + 1) / 2 

def divides (a b : ℕ) : Prop := ∃ c : ℕ, b = a * c

def valid_n (n : ℕ) : Prop := divides (arithmetic_sum n) (10 * n)

theorem number_of_valid_n : { n : ℕ | valid_n n ∧ n > 0 }.to_finset = {1, 3, 4, 9, 19}.to_finset := 
by {
  sorry
}

end number_of_valid_n_l137_137386


namespace min_value_y_l137_137526

theorem min_value_y (x : ℝ) (hx : x > 3) : 
  ∃ y, (∀ x > 3, y = min_value) ∧ min_value = 5 :=
by 
  sorry

end min_value_y_l137_137526


namespace find_m_n_l137_137670

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l137_137670


namespace mark_owe_triple_amount_l137_137570

theorem mark_owe_triple_amount (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 2000) (hr : r = 0.04) :
  (1 + r)^t > 3 → t = 30 :=
by
  intro h
  norm_cast at h
  sorry

end mark_owe_triple_amount_l137_137570


namespace distinct_diagonals_convex_nonagon_l137_137976

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l137_137976


namespace find_xy_l137_137187

theorem find_xy (x y : ℤ) 
  (h1 : (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3) : 
  x = -35 ∧ y = -35 :=
by 
  sorry

end find_xy_l137_137187


namespace prod_lcm_gcd_eq_216_l137_137951

theorem prod_lcm_gcd_eq_216 (a b : ℕ) (h1 : a = 12) (h2 : b = 18) :
  (Nat.gcd a b) * (Nat.lcm a b) = 216 := by
  sorry

end prod_lcm_gcd_eq_216_l137_137951


namespace area_of_triangle_formed_by_tangent_line_l137_137455

noncomputable def curve (x : ℝ) : ℝ := Real.log x - 2 * x

noncomputable def slope_of_tangent_at (x : ℝ) : ℝ := (1 / x) - 2

def point_of_tangency : ℝ × ℝ := (1, -2)

-- Define the tangent line equation at the point (1, -2)
noncomputable def tangent_line (x : ℝ) : ℝ := -x - 1

-- Define x and y intercepts of the tangent line
def x_intercept_of_tangent : ℝ := -1
def y_intercept_of_tangent : ℝ := -1

-- Define the area of the triangle formed by the tangent line and the coordinate axes
def triangle_area : ℝ := 0.5 * (-1) * (-1)

-- State the theorem to prove the area of the triangle
theorem area_of_triangle_formed_by_tangent_line : 
  triangle_area = 0.5 := by 
sorry

end area_of_triangle_formed_by_tangent_line_l137_137455


namespace distinct_diagonals_convex_nonagon_l137_137977

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l137_137977


namespace frank_money_l137_137105

-- Define the initial amount, expenses, and incomes as per the conditions
def initialAmount : ℕ := 11
def spentOnGame : ℕ := 3
def spentOnKeychain : ℕ := 2
def receivedFromAlice : ℕ := 4
def allowance : ℕ := 14
def spentOnBusTicket : ℕ := 5

-- Define the total money left for Frank
def finalAmount (initial : ℕ) (game : ℕ) (keychain : ℕ) (gift : ℕ) (allowance : ℕ) (bus : ℕ) : ℕ :=
  initial - game - keychain + gift + allowance - bus

-- Define the theorem stating that the final amount is 19
theorem frank_money : finalAmount initialAmount spentOnGame spentOnKeychain receivedFromAlice allowance spentOnBusTicket = 19 :=
by
  sorry

end frank_money_l137_137105


namespace new_seq_is_arithmetic_l137_137923

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def new_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a n + a (n + 3)

theorem new_seq_is_arithmetic (a : ℕ → ℝ) (d : ℝ) (b : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a d)
  (h_new_seq : new_sequence a b) :
  arithmetic_sequence b (2 * d) :=
sorry

end new_seq_is_arithmetic_l137_137923


namespace ball_hits_ground_at_time_l137_137302

theorem ball_hits_ground_at_time :
  ∀ (t : ℝ), (-18 * t^2 + 30 * t + 60 = 0) ↔ (t = (5 + Real.sqrt 145) / 6) :=
sorry

end ball_hits_ground_at_time_l137_137302


namespace find_shaun_age_l137_137435

def current_ages (K G S : ℕ) :=
  K + 4 = 2 * (G + 4) ∧
  S + 8 = 2 * (K + 8) ∧
  S + 12 = 3 * (G + 12)

theorem find_shaun_age (K G S : ℕ) (h : current_ages K G S) : S = 48 :=
  by
    sorry

end find_shaun_age_l137_137435


namespace divisibility_equiv_l137_137148

-- Definition of the functions a(n) and b(n)
def a (n : ℕ) := n^5 + 5^n
def b (n : ℕ) := n^5 * 5^n + 1

-- Define a positive integer
variables (n : ℕ) (hn : n > 0)

-- The theorem stating the equivalence
theorem divisibility_equiv : (a n) % 11 = 0 ↔ (b n) % 11 = 0 :=
sorry
 
end divisibility_equiv_l137_137148


namespace weight_of_fresh_grapes_is_40_l137_137392

-- Define the weight of fresh grapes and dried grapes
variables (F D : ℝ)

-- Fresh grapes contain 90% water by weight, so 10% is non-water
def fresh_grapes_non_water_content (F : ℝ) : ℝ := 0.10 * F

-- Dried grapes contain 20% water by weight, so 80% is non-water
def dried_grapes_non_water_content (D : ℝ) : ℝ := 0.80 * D

-- Given condition: weight of dried grapes is 5 kg
def weight_of_dried_grapes : ℝ := 5

-- The main theorem to prove
theorem weight_of_fresh_grapes_is_40 :
  fresh_grapes_non_water_content F = dried_grapes_non_water_content weight_of_dried_grapes →
  F = 40 := 
by
  sorry

end weight_of_fresh_grapes_is_40_l137_137392


namespace divisibility_by_11_l137_137415

theorem divisibility_by_11 (m n : ℤ) (h : (5 * m + 3 * n) % 11 = 0) : (9 * m + n) % 11 = 0 := by
  sorry

end divisibility_by_11_l137_137415


namespace boat_speed_in_still_water_l137_137215

theorem boat_speed_in_still_water (D V_s t_down t_up : ℝ) (h_val : V_s = 3) (h_down : D = (15 + V_s) * t_down) (h_up : D = (15 - V_s) * t_up) : 15 = 15 :=
by
  have h1 : 15 = (D / 1 - V_s) := sorry
  have h2 : 15 = (D / 1.5 + V_s) := sorry
  sorry

end boat_speed_in_still_water_l137_137215


namespace guppies_eaten_by_moray_eel_l137_137842

-- Definitions based on conditions
def moray_eel_guppies_per_day : ℕ := sorry -- Number of guppies the moray eel eats per day

def number_of_betta_fish : ℕ := 5

def guppies_per_betta : ℕ := 7

def total_guppies_needed_per_day : ℕ := 55

-- Theorem based on the question
theorem guppies_eaten_by_moray_eel :
  moray_eel_guppies_per_day = total_guppies_needed_per_day - (number_of_betta_fish * guppies_per_betta) :=
sorry

end guppies_eaten_by_moray_eel_l137_137842


namespace find_a_of_inequality_solution_l137_137120

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -3 < ax - 2 ∧ ax - 2 < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 := by
  sorry

end find_a_of_inequality_solution_l137_137120


namespace distinct_diagonals_nonagon_l137_137980

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l137_137980


namespace evaluate_expression_l137_137245

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = 2) : 
  (x^3 * y^4 * z)^2 = 1 / 104976 :=
by 
  sorry

end evaluate_expression_l137_137245


namespace Angela_is_295_cm_l137_137081

noncomputable def Angela_height (Carl_height : ℕ) : ℕ :=
  let Becky_height := 2 * Carl_height
  let Amy_height := Becky_height + Becky_height / 5  -- 20% taller than Becky
  let Helen_height := Amy_height + 3
  let Angela_height := Helen_height + 4
  Angela_height

theorem Angela_is_295_cm : Angela_height 120 = 295 := 
by 
  sorry

end Angela_is_295_cm_l137_137081


namespace domain_g_eq_l137_137175

noncomputable def domain_f : Set ℝ := {x | -8 ≤ x ∧ x ≤ 4}

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

theorem domain_g_eq (f : ℝ → ℝ) (h : ∀ x, x ∈ domain_f → f x ∈ domain_f) :
  {x | x ∈ [-2, 4]} = {x | -2 ≤ x ∧ x ≤ 4} :=
by {
  sorry
}

end domain_g_eq_l137_137175


namespace problem_1_problem_2_l137_137571

theorem problem_1 :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
by
  sorry

theorem problem_2 :
  (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) = (3^32 - 1) / 2 :=
by
  sorry

end problem_1_problem_2_l137_137571


namespace investment_worth_l137_137299

noncomputable def initial_investment (total_earning : ℤ) : ℤ := total_earning / 2

noncomputable def current_worth (initial_investment total_earning : ℤ) : ℤ :=
  initial_investment + total_earning

theorem investment_worth (monthly_earning : ℤ) (months : ℤ) (earnings : ℤ)
  (h1 : monthly_earning * months = earnings)
  (h2 : earnings = 2 * initial_investment earnings) :
  current_worth (initial_investment earnings) earnings = 90 := 
by
  -- We proceed to show the current worth is $90
  -- Proof will be constructed here
  sorry
  
end investment_worth_l137_137299


namespace no_such_function_l137_137512

theorem no_such_function (f : ℕ → ℕ) : ¬ (∀ n : ℕ, f (f n) = n + 2019) :=
sorry

end no_such_function_l137_137512


namespace smallest_number_with_unique_digits_summing_to_32_exists_l137_137381

theorem smallest_number_with_unique_digits_summing_to_32_exists : 
  ∃ n : ℕ, n / 10000 < 10 ∧ (n % 10 ≠ (n / 10) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 1000) % 10 ≠ (n / 10000) % 10) ∧ 
  (n % 10 + (n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10 + (n / 10000) % 10 = 32) := 
sorry

end smallest_number_with_unique_digits_summing_to_32_exists_l137_137381


namespace initial_puppies_count_l137_137503

theorem initial_puppies_count (P : ℕ) (h1 : P - 2 + 3 = 8) : P = 7 :=
sorry

end initial_puppies_count_l137_137503


namespace trajectory_of_midpoint_l137_137280

-- Definitions based on the conditions identified in the problem
variables {x y x1 y1 : ℝ}

-- Condition that point P is on the curve y = 2x^2 + 1
def point_on_curve (x1 y1 : ℝ) : Prop :=
  y1 = 2 * x1^2 + 1

-- Definition of the midpoint M conditions
def midpoint_def (x y x1 y1 : ℝ) : Prop :=
  x = (x1 + 0) / 2 ∧ y = (y1 - 1) / 2

-- Final theorem statement to be proved
theorem trajectory_of_midpoint (x y x1 y1 : ℝ) :
  point_on_curve x1 y1 → midpoint_def x y x1 y1 → y = 4 * x^2 :=
sorry

end trajectory_of_midpoint_l137_137280


namespace add_pure_alcohol_to_achieve_percentage_l137_137495

-- Define the initial conditions
def initial_solution_volume : ℝ := 6
def initial_alcohol_percentage : ℝ := 0.30
def initial_pure_alcohol : ℝ := initial_solution_volume * initial_alcohol_percentage

-- Define the final conditions
def final_alcohol_percentage : ℝ := 0.50

-- Define the unknown to prove
def amount_of_alcohol_to_add : ℝ := 2.4

-- The target statement to prove
theorem add_pure_alcohol_to_achieve_percentage :
  (initial_pure_alcohol + amount_of_alcohol_to_add) / (initial_solution_volume + amount_of_alcohol_to_add) = final_alcohol_percentage :=
by
  sorry

end add_pure_alcohol_to_achieve_percentage_l137_137495


namespace red_balls_in_bag_l137_137497

theorem red_balls_in_bag (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total_balls = 60 → 
  white_balls = 22 → 
  green_balls = 18 → 
  yellow_balls = 8 → 
  purple_balls = 7 → 
  prob_neither_red_nor_purple = 0.8 → 
  ( ∃ (red_balls : ℕ), red_balls = 5 ) :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end red_balls_in_bag_l137_137497


namespace problem_statement_l137_137565

theorem problem_statement : ∃ n : ℤ, 0 < n ∧ (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).den = 1 ∧ ¬ n > 96 := 
by 
  sorry

end problem_statement_l137_137565


namespace distinct_diagonals_in_convex_nonagon_l137_137989

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l137_137989


namespace sale_price_60_l137_137288

theorem sale_price_60 (original_price : ℕ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) 
  (h2 : discount_percentage = 0.40) :
  sale_price = (original_price : ℝ) * (1 - discount_percentage) :=
by
  sorry

end sale_price_60_l137_137288


namespace not_divisible_l137_137332

theorem not_divisible (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 12) : ¬∃ k : ℕ, 120 * a + 2 * b = k * (100 * a + b) := 
sorry

end not_divisible_l137_137332


namespace daleyza_contracted_units_l137_137943

variable (units_building1 : ℕ)
variable (units_building2 : ℕ)
variable (units_building3 : ℕ)

def total_units (units_building1 units_building2 units_building3 : ℕ) : ℕ :=
  units_building1 + units_building2 + units_building3

theorem daleyza_contracted_units :
  units_building1 = 4000 →
  units_building2 = 2 * units_building1 / 5 →
  units_building3 = 120 * units_building2 / 100 →
  total_units units_building1 units_building2 units_building3 = 7520 :=
by
  intros h1 h2 h3
  unfold total_units
  rw [h1, h2, h3]
  sorry

end daleyza_contracted_units_l137_137943


namespace eddie_age_l137_137365

theorem eddie_age (Becky_age Irene_age Eddie_age : ℕ)
  (h1 : Becky_age * 2 = Irene_age)
  (h2 : Irene_age = 46)
  (h3 : Eddie_age = 4 * Becky_age) :
  Eddie_age = 92 := by
  sorry

end eddie_age_l137_137365


namespace solve_fraction_equation_l137_137250

def fraction_equation (x : ℝ) : Prop :=
  1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1) = 5

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  fraction_equation x → 
  x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 :=
by
  sorry

end solve_fraction_equation_l137_137250


namespace elevenRowTriangleTotalPieces_l137_137788

-- Definitions and problem statement
def numRodsInRow (n : ℕ) : ℕ := 3 * n

def sumFirstN (n : ℕ) : ℕ := n * (n + 1) / 2

def totalRods (rows : ℕ) : ℕ := 3 * (sumFirstN rows)

def totalConnectors (rows : ℕ) : ℕ := sumFirstN (rows + 1)

def totalPieces (rows : ℕ) : ℕ := totalRods rows + totalConnectors rows

-- Lean proof problem
theorem elevenRowTriangleTotalPieces : totalPieces 11 = 276 := 
by
  sorry

end elevenRowTriangleTotalPieces_l137_137788


namespace vanya_speed_problem_l137_137742

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l137_137742


namespace distinct_diagonals_nonagon_l137_137979

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l137_137979


namespace contradiction_method_l137_137484

variable (a b : ℝ)

theorem contradiction_method (h1 : a > b) (h2 : 3 * a ≤ 3 * b) : false :=
by sorry

end contradiction_method_l137_137484


namespace total_hours_proof_l137_137711

-- Definitions and conditions
def kate_hours : ℕ := 22
def pat_hours : ℕ := 2 * kate_hours
def mark_hours : ℕ := kate_hours + 110

-- Statement of the proof problem
theorem total_hours_proof : pat_hours + kate_hours + mark_hours = 198 := by
  sorry

end total_hours_proof_l137_137711


namespace relationship_p_q_no_linear_term_l137_137281

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end relationship_p_q_no_linear_term_l137_137281


namespace sequence_2010_eq_4040099_l137_137155

def sequence_term (n : Nat) : Int :=
  if n % 2 = 0 then 
    (n^2 - 1 : Int) 
  else 
    -(n^2 - 1 : Int)

theorem sequence_2010_eq_4040099 : sequence_term 2010 = 4040099 := 
  by 
    sorry

end sequence_2010_eq_4040099_l137_137155


namespace nonagon_diagonals_count_l137_137991

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l137_137991


namespace father_cannot_see_boy_more_than_half_time_l137_137161

def speed_boy := 10 -- speed in km/h
def speed_father := 5 -- speed in km/h

def cannot_see_boy_more_than_half_time (school_perimeter : ℝ) : Prop :=
  ¬(∃ T : ℝ, T > school_perimeter / (2 * speed_boy) ∧ T < school_perimeter / speed_boy)

theorem father_cannot_see_boy_more_than_half_time (school_perimeter : ℝ) (h_school_perimeter : school_perimeter > 0) :
  cannot_see_boy_more_than_half_time school_perimeter :=
by
  sorry

end father_cannot_see_boy_more_than_half_time_l137_137161


namespace calc_value_exponents_l137_137090

theorem calc_value_exponents :
  (3^3) * (5^3) * (3^5) * (5^5) = 15^8 :=
by sorry

end calc_value_exponents_l137_137090


namespace inequality_holds_for_minimal_a_l137_137826

theorem inequality_holds_for_minimal_a :
  ∀ (x : ℝ), (1 ≤ x) → (x ≤ 4) → (1 + x) * Real.log x + x ≤ x * 1.725 :=
by
  intros x h1 h2
  sorry

end inequality_holds_for_minimal_a_l137_137826


namespace number_of_days_to_catch_fish_l137_137099

variable (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ)

theorem number_of_days_to_catch_fish (h1 : fish_per_day = 2) 
                                    (h2 : fillets_per_fish = 2) 
                                    (h3 : total_fillets = 120) : 
                                    (total_fillets / fillets_per_fish) / fish_per_day = 30 :=
by sorry

end number_of_days_to_catch_fish_l137_137099


namespace geometric_sequence_common_ratio_l137_137262

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q)
  (h0 : a 1 = 2) (h1 : a 4 = 1 / 4) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l137_137262


namespace largest_n_divisible_l137_137609

theorem largest_n_divisible (n : ℕ) (h : (n ^ 3 + 144) % (n + 12) = 0) : n ≤ 84 :=
sorry

end largest_n_divisible_l137_137609


namespace number_of_strawberry_cakes_l137_137347

def number_of_chocolate_cakes := 3
def price_of_chocolate_cake := 12
def price_of_strawberry_cake := 22
def total_payment := 168

theorem number_of_strawberry_cakes (S : ℕ) : 
    number_of_chocolate_cakes * price_of_chocolate_cake + S * price_of_strawberry_cake = total_payment → 
    S = 6 :=
by
  sorry

end number_of_strawberry_cakes_l137_137347


namespace num_of_adults_l137_137010

def students : ℕ := 22
def vans : ℕ := 3
def capacity_per_van : ℕ := 8

theorem num_of_adults : (vans * capacity_per_van) - students = 2 := by
  sorry

end num_of_adults_l137_137010


namespace calculate_leakage_rate_l137_137290

variable (B : ℕ) (T : ℕ) (R : ℝ)

-- B represents the bucket's capacity in ounces, T represents time in hours, R represents the rate of leakage per hour in ounces per hour.

def leakage_rate (B : ℕ) (T : ℕ) (R : ℝ) : Prop :=
  (B = 36) ∧ (T = 12) ∧ (B / 2 = T * R)

theorem calculate_leakage_rate : leakage_rate 36 12 1.5 :=
by 
  simp [leakage_rate]
  sorry

end calculate_leakage_rate_l137_137290


namespace vanya_faster_by_4_l137_137733

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l137_137733


namespace milburg_population_l137_137471

theorem milburg_population 
    (adults : ℕ := 5256) 
    (children : ℕ := 2987) 
    (teenagers : ℕ := 1709) 
    (seniors : ℕ := 2340) : 
    adults + children + teenagers + seniors = 12292 := 
by 
  sorry

end milburg_population_l137_137471


namespace smallest_y_value_l137_137204

theorem smallest_y_value (y : ℚ) (h : y / 7 + 2 / (7 * y) = 1 / 3) : y = 2 / 3 :=
sorry

end smallest_y_value_l137_137204


namespace binomial_sum_mod_prime_l137_137310

theorem binomial_sum_mod_prime (T : ℕ) (hT : T = ∑ k in Finset.range 65, Nat.choose 2024 k) : 
  T % 2027 = 1089 :=
by
  have h_prime : Nat.prime 2027 := by sorry -- Given that 2027 is prime
  have h := (2024 : ℤ) % 2027
  sorry -- The proof of the actual sum equivalences

end binomial_sum_mod_prime_l137_137310


namespace technician_round_trip_completion_l137_137928

theorem technician_round_trip_completion (D : ℝ) (h0 : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center := 0.30 * D
  let traveled := to_center + from_center
  traveled / round_trip * 100 = 65 := 
by
  sorry

end technician_round_trip_completion_l137_137928


namespace polyhedron_volume_is_correct_l137_137196

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 12
  let num_squares := 3
  let square_area := side_length * side_length
  let cube_volume := side_length ^ 3
  let polyhedron_volume := cube_volume / 2
  polyhedron_volume

theorem polyhedron_volume_is_correct :
  volume_of_polyhedron = 864 :=
by
  sorry

end polyhedron_volume_is_correct_l137_137196


namespace smallest_k_remainder_2_l137_137764

theorem smallest_k_remainder_2 (k : ℕ) :
  k > 1 ∧
  k % 13 = 2 ∧
  k % 7 = 2 ∧
  k % 3 = 2 →
  k = 275 :=
by sorry

end smallest_k_remainder_2_l137_137764


namespace relay_race_solution_l137_137168

variable (Sadie_time : ℝ) (Sadie_speed : ℝ)
variable (Ariana_time : ℝ) (Ariana_speed : ℝ)
variable (Sarah_speed : ℝ)
variable (total_distance : ℝ)

def relay_race_time : Prop :=
  let Sadie_distance := Sadie_time * Sadie_speed
  let Ariana_distance := Ariana_time * Ariana_speed
  let Sarah_distance := total_distance - Sadie_distance - Ariana_distance
  let Sarah_time := Sarah_distance / Sarah_speed
  Sadie_time + Ariana_time + Sarah_time = 4.5

theorem relay_race_solution (h1: Sadie_time = 2) (h2: Sadie_speed = 3)
  (h3: Ariana_time = 0.5) (h4: Ariana_speed = 6)
  (h5: Sarah_speed = 4) (h6: total_distance = 17) :
  relay_race_time Sadie_time Sadie_speed Ariana_time Ariana_speed Sarah_speed total_distance :=
by
  sorry

end relay_race_solution_l137_137168


namespace value_of_a_squared_plus_b_squared_l137_137676

variable (a b : ℝ)

theorem value_of_a_squared_plus_b_squared (h1 : a - b = 10) (h2 : a * b = 55) : a^2 + b^2 = 210 := 
by 
sorry

end value_of_a_squared_plus_b_squared_l137_137676


namespace vector_projection_line_l137_137466

theorem vector_projection_line (v : ℝ × ℝ) 
  (h : ∃ (x y : ℝ), v = (x, y) ∧ 
       (3 * x + 4 * y) / (3 ^ 2 + 4 ^ 2) = 1) :
  ∃ (x y : ℝ), v = (x, y) ∧ y = -3 / 4 * x + 25 / 4 :=
by
  sorry

end vector_projection_line_l137_137466


namespace arithmetic_geometric_sequence_l137_137264

theorem arithmetic_geometric_sequence (a b : ℝ)
  (h1 : 2 * a = 1 + b)
  (h2 : b^2 = a)
  (h3 : a ≠ b) : a = 1 / 4 :=
by
  sorry

end arithmetic_geometric_sequence_l137_137264


namespace complex_power_difference_l137_137825

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 16 - (1 - i) ^ 16 = 0 := by
  sorry

end complex_power_difference_l137_137825


namespace sum_of_coefficients_equals_28_l137_137240

def P (x : ℝ) : ℝ :=
  2 * (4 * x^8 - 5 * x^5 + 9 * x^3 - 6) + 8 * (x^6 - 4 * x^3 + 6)

theorem sum_of_coefficients_equals_28 : P 1 = 28 := by
  sorry

end sum_of_coefficients_equals_28_l137_137240


namespace first_plane_passengers_l137_137886

-- Definitions and conditions
def speed_plane_empty : ℕ := 600
def slowdown_per_passenger : ℕ := 2
def second_plane_passengers : ℕ := 60
def third_plane_passengers : ℕ := 40
def average_speed : ℕ := 500

-- Definition of the speed of a plane given number of passengers
def speed (passengers : ℕ) : ℕ := speed_plane_empty - slowdown_per_passenger * passengers

-- The problem statement rewritten in Lean 4
theorem first_plane_passengers (P : ℕ) (h_avg : (speed P + speed second_plane_passengers + speed third_plane_passengers) / 3 = average_speed) : P = 50 :=
sorry

end first_plane_passengers_l137_137886


namespace lattice_points_distance_5_l137_137689

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem lattice_points_distance_5 : 
  ∃ S : Finset (ℤ × ℤ × ℤ), 
    (∀ p ∈ S, is_lattice_point p.1 p.2.1 p.2.2) ∧
    S.card = 78 :=
by
  sorry

end lattice_points_distance_5_l137_137689


namespace fourth_power_of_cube_of_third_smallest_prime_l137_137900

theorem fourth_power_of_cube_of_third_smallest_prime :
  (let p3 := 5 in
  let cube := p3^3 in
  let fourth_power := cube^4 in
  fourth_power = 244140625) :=
by
  let p3 := 5
  let cube := p3^3
  let fourth_power := cube^4
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l137_137900


namespace find_k_l137_137963

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 + k * x - 7

-- Define the given condition f(5) - g(5) = 20
def condition (k : ℝ) : Prop := f 5 - g 5 k = 20

-- The theorem to prove that k = 16.4
theorem find_k : ∃ k : ℝ, condition k ∧ k = 16.4 :=
by
  sorry

end find_k_l137_137963


namespace divide_6_books_into_three_parts_each_2_distribute_6_books_to_ABC_each_2_distribute_6_books_to_ABC_distribute_6_books_to_ABC_each_at_least_1_l137_137034

open Finset

-- Proof problem for Ⅰ
theorem divide_6_books_into_three_parts_each_2 : 
  ∃ (S : Finset (Finset ℕ)), S.card = 15 ∧ ∀ s ∈ S, s.card = 2 ∧ s.sum = 6 :=
sorry

-- Proof problem for Ⅱ
theorem distribute_6_books_to_ABC_each_2 : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 90 ∧ 
  ∀ s ∈ S, (∀ t ∈ s, t.card = 2) ∧ S.sum = 6 :=
sorry

-- Proof problem for Ⅲ
theorem distribute_6_books_to_ABC : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 729 ∧ S.sum = 6 :=
sorry

-- Proof problem for Ⅳ
theorem distribute_6_books_to_ABC_each_at_least_1 : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 481 ∧ S.sum = 6 ∧ 
  ∀ s ∈ S, (∀ t ∈ s, t.sum ≥ 1) :=
sorry

end divide_6_books_into_three_parts_each_2_distribute_6_books_to_ABC_each_2_distribute_6_books_to_ABC_distribute_6_books_to_ABC_each_at_least_1_l137_137034


namespace problem_one_problem_two_l137_137658

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x + 4

-- Problem (I)
theorem problem_one (m : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → f x m < 0) ↔ m ≤ -5 :=
sorry

-- Problem (II)
theorem problem_two (m : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → abs ((f x m - x^2) / m) < 1) ↔ (-4 < m ∧ m ≤ -2) :=
sorry

end problem_one_problem_two_l137_137658


namespace subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l137_137562

def A : Set ℝ := {x | x ^ 2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_if_a_neg_third (a : ℝ) (h : a = -1/3) : B a ⊆ A := by
  sorry

theorem set_of_real_numbers_for_A_union_B_eq_A : {a : ℝ | A ∪ B a = A} = {0, -1/3, -1/5} := by
  sorry

end subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l137_137562


namespace num_diagonals_convex_nonagon_l137_137984

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l137_137984


namespace Oscar_height_correct_l137_137953

-- Definitions of the given conditions
def Tobias_height : ℕ := 184
def avg_height : ℕ := 178

def heights_valid (Victor Peter Oscar Tobias : ℕ) : Prop :=
  Tobias = 184 ∧ (Tobias + Victor + Peter + Oscar) / 4 = 178 ∧ 
  Victor = Tobias + (Tobias - Peter) ∧ 
  Oscar = Peter - (Tobias - Peter)

theorem Oscar_height_correct :
  ∃ (k : ℕ), ∀ (Victor Peter Oscar : ℕ), heights_valid Victor Peter Oscar Tobias_height →
  Oscar = 160 :=
by
  sorry

end Oscar_height_correct_l137_137953


namespace min_value_reciprocal_sum_l137_137720

theorem min_value_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ c d : ℝ, 0 < c ∧ 0 < d ∧ c + d = 2 → (1/c + 1/d) ≥ m := 
sorry

end min_value_reciprocal_sum_l137_137720


namespace horatio_sonnets_count_l137_137546

-- Each sonnet consists of 14 lines
def lines_per_sonnet : ℕ := 14

-- The number of sonnets his lady fair heard
def heard_sonnets : ℕ := 7

-- The total number of unheard lines
def unheard_lines : ℕ := 70

-- Calculate sonnets Horatio wrote by the heard and unheard components
def total_sonnets : ℕ := heard_sonnets + (unheard_lines / lines_per_sonnet)

-- Prove the total number of sonnets horatio wrote
theorem horatio_sonnets_count : total_sonnets = 12 := 
by sorry

end horatio_sonnets_count_l137_137546


namespace fourth_power_of_cube_of_third_smallest_prime_l137_137898

theorem fourth_power_of_cube_of_third_smallest_prime :
  let p := 5 in
  let x := p^3 in
  let y := x^4 in
  y = 244140625 :=
by
  sorry

end fourth_power_of_cube_of_third_smallest_prime_l137_137898


namespace sequence_general_term_l137_137142

namespace SequenceSum

def Sn (n : ℕ) : ℕ :=
  2 * n^2 + n

def a₁ (n : ℕ) : ℕ :=
  if n = 1 then Sn n else (Sn n - Sn (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  a₁ n = 4 * n - 1 :=
sorry

end SequenceSum

end sequence_general_term_l137_137142


namespace shirt_cost_correct_l137_137163

-- Definitions based on the conditions
def initial_amount : ℕ := 109
def pants_cost : ℕ := 13
def remaining_amount : ℕ := 74
def total_spent : ℕ := initial_amount - remaining_amount
def shirts_cost : ℕ := total_spent - pants_cost
def number_of_shirts : ℕ := 2

-- Statement to be proved
theorem shirt_cost_correct : shirts_cost / number_of_shirts = 11 := by
  sorry

end shirt_cost_correct_l137_137163


namespace base6_sum_l137_137644

theorem base6_sum (D C : ℕ) (h₁ : D + 2 = C) (h₂ : C + 3 = 7) : C + D = 6 :=
by
  sorry

end base6_sum_l137_137644


namespace Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l137_137008

-- Problem 1: Factorization
theorem Problem1_factorize (a : ℝ) : a^2 - 8 * a + 15 = (a - 3) * (a - 5) :=
  sorry

-- Problem 2: Minimum Perimeter of triangle ABC
theorem Problem2_min_perimeter_triangle (a b c : ℝ) 
  (h : a^2 + b^2 - 14 * a - 8 * b + 65 = 0) (hc : ∃ k : ℤ, 2 * k + 1 = c) : 
  a + b + c ≥ 16 :=
  sorry

-- Problem 3: Maximum Value of the Polynomial
theorem Problem3_max_value_polynomial : 
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, y ≠ -1 → -2 * x^2 - 4 * x + 3 ≥ -2 * y^2 - 4 * y + 3 :=
  sorry

end Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l137_137008


namespace prob_square_l137_137002

def total_figures := 10
def num_squares := 3
def num_circles := 4
def num_triangles := 3

theorem prob_square : (num_squares : ℚ) / total_figures = 3 / 10 :=
by
  rw [total_figures, num_squares]
  exact sorry

end prob_square_l137_137002


namespace find_a_l137_137537

theorem find_a :
  let p1 := (⟨-3, 7⟩ : ℝ × ℝ)
  let p2 := (⟨2, -1⟩ : ℝ × ℝ)
  let direction := (5, -8)
  let target_direction := (a, -2)
  a = (direction.1 * -2) / (direction.2) := by
  sorry

end find_a_l137_137537


namespace smallest_possible_n_l137_137022

-- Definitions needed for the problem
variable (x n : ℕ) (hpos : 0 < x)
variable (m : ℕ) (hm : m = 72)

-- The conditions as already stated
def gcd_cond := Nat.gcd 72 n = x + 8
def lcm_cond := Nat.lcm 72 n = x * (x + 8)

-- The proof statement
theorem smallest_possible_n (h_gcd : gcd_cond x n) (h_lcm : lcm_cond x n) : n = 8 :=
by 
  -- Intuitively outline the proof
  sorry

end smallest_possible_n_l137_137022


namespace problem_l137_137813

theorem problem (a b : ℝ) (h1 : abs a = 4) (h2 : b^2 = 9) (h3 : a / b > 0) : a - b = 1 ∨ a - b = -1 := 
sorry

end problem_l137_137813


namespace vanya_faster_speed_l137_137755

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l137_137755


namespace unique_solution_f_l137_137795

def f : ℕ → ℕ
  := sorry

namespace ProofProblem

theorem unique_solution_f (f : ℕ → ℕ)
  (h1 : ∀ (m n : ℕ), f m + f n - m * n ≠ 0)
  (h2 : ∀ (m n : ℕ), f m + f n - m * n ∣ m * f m + n * f n)
  : (∀ n : ℕ, f n = n^2) :=
sorry

end ProofProblem

end unique_solution_f_l137_137795


namespace solve_system_l137_137213

theorem solve_system (x y : ℝ) (h1 : 5 * x + y = 19) (h2 : x + 3 * y = 1) : 3 * x + 2 * y = 10 :=
by
  sorry

end solve_system_l137_137213


namespace jane_age_problem_l137_137841

variables (J M a b c : ℕ)
variables (h1 : J = 2 * (a + b))
variables (h2 : J / 2 = a + b)
variables (h3 : c = 2 * J)
variables (h4 : M > 0)

theorem jane_age_problem (h5 : J - M = 3 * ((J / 2) - 2 * M))
                         (h6 : J - M = c - M)
                         (h7 : c = 2 * J) :
  J / M = 10 :=
sorry

end jane_age_problem_l137_137841


namespace jamie_avg_is_correct_l137_137892

-- Declare the set of test scores and corresponding sums
def test_scores : List ℤ := [75, 78, 82, 85, 88, 91]

-- Alex's average score
def alex_avg : ℤ := 82

-- Total test score sum
def total_sum : ℤ := test_scores.sum

theorem jamie_avg_is_correct (alex_sum : ℤ) :
    alex_sum = 3 * alex_avg →
    (total_sum - alex_sum) / 3 = 253 / 3 :=
by
  sorry

end jamie_avg_is_correct_l137_137892


namespace park_area_l137_137492

theorem park_area (L B : ℝ) (h1 : L = B / 2) (h2 : 6 * 1000 / 60 * 6 = 2 * (L + B)) : L * B = 20000 :=
by
  -- proof will go here
  sorry

end park_area_l137_137492


namespace basketballs_count_l137_137355

theorem basketballs_count (x : ℕ) : 
  let num_volleyballs := x
  let num_basketballs := 2 * x
  let num_soccer_balls := x - 8
  num_volleyballs + num_basketballs + num_soccer_balls = 100 →
  num_basketballs = 54 :=
by
  intros h
  sorry

end basketballs_count_l137_137355


namespace sum_of_c_and_d_l137_137550

def digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem sum_of_c_and_d 
  (c d : ℕ)
  (hcd : digit c)
  (hdd : digit d)
  (h1: (4*c) * 5 % 10 = 5)
  (h2: 215 = (10 * (4*(d*5) + c*5)) + d*10 + 5) :
  c + d = 5 := 
  sorry

end sum_of_c_and_d_l137_137550


namespace range_of_f_is_0_2_3_l137_137461

def f (x : ℤ) : ℤ := x + 1
def S : Set ℤ := {-1, 1, 2}

theorem range_of_f_is_0_2_3 : Set.image f S = {0, 2, 3} := by
  sorry

end range_of_f_is_0_2_3_l137_137461


namespace steve_marbles_after_trans_l137_137573

def initial_marbles (S T L H : ℕ) : Prop :=
  S = 2 * T ∧
  L = S - 5 ∧
  H = T + 3

def transactions (S T L H : ℕ) (new_S new_T new_L new_H : ℕ) : Prop :=
  new_S = S - 10 ∧
  new_L = L - 4 ∧
  new_T = T + 4 ∧
  new_H = H - 6

theorem steve_marbles_after_trans (S T L H new_S new_T new_L new_H : ℕ) :
  initial_marbles S T L H →
  transactions S T L H new_S new_T new_L new_H →
  new_S = 6 →
  new_T = 12 :=
by
  sorry

end steve_marbles_after_trans_l137_137573


namespace unique_point_value_l137_137581

noncomputable def unique_point_condition : Prop :=
  ∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + 12 = 0

theorem unique_point_value (d : ℝ) : unique_point_condition ↔ d = 12 := 
sorry

end unique_point_value_l137_137581


namespace clothing_percentage_l137_137001

variable (T : ℝ) -- Total amount excluding taxes.
variable (C : ℝ) -- Percentage of total amount spent on clothing.

-- Conditions
def spent_on_food := 0.2 * T
def spent_on_other_items := 0.3 * T

-- Taxes
def tax_on_clothing := 0.04 * (C * T)
def tax_on_food := 0.0
def tax_on_other_items := 0.08 * (0.3 * T)
def total_tax_paid := 0.044 * T

-- Statement to prove
theorem clothing_percentage : 
  0.04 * (C * T) + 0.08 * (0.3 * T) = 0.044 * T ↔ C = 0.5 :=
by
  sorry

end clothing_percentage_l137_137001


namespace sin_double_angle_l137_137810

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end sin_double_angle_l137_137810


namespace smallest_number_with_sum_32_and_distinct_digits_l137_137375

def is_distinct (l : List Nat) : Prop := l.Nodup

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem smallest_number_with_sum_32_and_distinct_digits :
  ∀ n : Nat, is_distinct (Nat.digits 10 n) ∧ sum_of_digits n = 32 → 26789 ≤ n :=
by
  intro n
  intro h
  sorry

end smallest_number_with_sum_32_and_distinct_digits_l137_137375


namespace calculate_square_add_subtract_l137_137888

theorem calculate_square_add_subtract (a b : ℤ) :
  (41 : ℤ)^2 = (40 : ℤ)^2 + 81 ∧ (39 : ℤ)^2 = (40 : ℤ)^2 - 79 :=
by
  sorry

end calculate_square_add_subtract_l137_137888


namespace increase_in_area_l137_137451

-- Define the initial side length and the increment.
def initial_side_length : ℕ := 6
def increment : ℕ := 1

-- Define the original area of the land.
def original_area : ℕ := initial_side_length * initial_side_length

-- Define the new side length after the increase.
def new_side_length : ℕ := initial_side_length + increment

-- Define the new area of the land.
def new_area : ℕ := new_side_length * new_side_length

-- Define the theorem that states the increase in area.
theorem increase_in_area : new_area - original_area = 13 := by
  sorry

end increase_in_area_l137_137451


namespace simplify_vector_eq_l137_137298

-- Define points A, B, C
variables {A B C O : Type} [AddGroup A]

-- Define vector operations corresponding to overrightarrow.
variables (AB OC OB AC AO BO : A)

-- Conditions in Lean definitions
-- Assuming properties like vector addition and subtraction, and associative properties
def vector_eq : Prop := AB + OC - OB = AC

theorem simplify_vector_eq :
  AB + OC - OB = AC :=
by
  -- Proof steps go here
  sorry

end simplify_vector_eq_l137_137298


namespace solution_count_l137_137942

noncomputable def equation_has_one_solution : Prop :=
∀ x : ℝ, (x - (8 / (x - 2))) = (4 - (8 / (x - 2))) → x = 4

theorem solution_count : equation_has_one_solution :=
by
  sorry

end solution_count_l137_137942


namespace count_neither_multiples_of_2_nor_3_l137_137248

theorem count_neither_multiples_of_2_nor_3 : 
  let count_multiples (k n : ℕ) : ℕ := n / k
  let total_numbers := 100
  let multiples_of_2 := count_multiples 2 total_numbers
  let multiples_of_3 := count_multiples 3 total_numbers
  let multiples_of_6 := count_multiples 6 total_numbers
  let multiples_of_2_or_3 := multiples_of_2 + multiples_of_3 - multiples_of_6
  total_numbers - multiples_of_2_or_3 = 33 :=
by 
  sorry

end count_neither_multiples_of_2_nor_3_l137_137248


namespace total_spent_on_index_cards_l137_137359

-- Definitions for conditions
def index_cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

-- Theorem to be proven
theorem total_spent_on_index_cards :
  let total_students := students_per_class * periods_per_day
  let total_cards := total_students * index_cards_per_student
  let packs_needed := total_cards / cards_per_pack
  let total_cost := packs_needed * cost_per_pack
  total_cost = 108 :=
by
  sorry

end total_spent_on_index_cards_l137_137359


namespace vanya_faster_by_4_l137_137732

variable (v : ℝ) -- initial speed in m/s
variable (school_distance : ℝ) -- distance to school in meters
variable (time_to_school : ℝ) -- time to school in seconds
variable (increased_speed : ℝ → ℝ) -- increased speed function

-- Conditions
def initial_speed : (v > 0) := by sorry
def increased_speed_by_2 {v : ℝ} : (v + 2 > 0) := by sorry
def faster_by_2_5 {v : ℝ} : (time_to_school = (school_distance / v)) → (time_to_school / 2.5 = (school_distance / (v + 2))) := by sorry

-- Main statement to prove
theorem vanya_faster_by_4 (h1 : initial_speed v) (h2 : increased_speed_by_2 v) (h3 : faster_by_2_5 v time_to_school school_distance) :
    (time_to_school / 4 = (school_distance / (v + 4))) := by
  sorry

end vanya_faster_by_4_l137_137732


namespace sum_binom_2024_mod_2027_l137_137309

theorem sum_binom_2024_mod_2027 :
  let T := ∑ k in Finset.range 65, Nat.choose 2024 k
  2027.prime →
  T % 2027 = 1089 :=
by
  intros T hp
  sorry

end sum_binom_2024_mod_2027_l137_137309


namespace ratio_sum_of_square_lengths_equals_68_l137_137464

theorem ratio_sum_of_square_lengths_equals_68 (a b c : ℕ) 
  (h1 : (∃ (r : ℝ), r = 50 / 98) → a = 5 ∧ b = 14 ∧ c = 49) :
  a + b + c = 68 :=
by
  sorry -- Proof is not required

end ratio_sum_of_square_lengths_equals_68_l137_137464


namespace least_positive_integer_x_20y_l137_137555

theorem least_positive_integer_x_20y (x y : ℤ) (h : Int.gcd x (20 * y) = 4) : 
  ∃ k : ℕ, k > 0 ∧ k * (x + 20 * y) = 4 := 
sorry

end least_positive_integer_x_20y_l137_137555


namespace sarah_score_l137_137129

theorem sarah_score
  (hunter_score : ℕ)
  (john_score : ℕ)
  (grant_score : ℕ)
  (sarah_score : ℕ)
  (h1 : hunter_score = 45)
  (h2 : john_score = 2 * hunter_score)
  (h3 : grant_score = john_score + 10)
  (h4 : sarah_score = grant_score - 5) :
  sarah_score = 95 :=
by
  sorry

end sarah_score_l137_137129


namespace num_unique_five_topping_pizzas_l137_137060

open Nat

/-- The number of combinations of choosing k items from n items is defined using binomial coefficients. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem num_unique_five_topping_pizzas:
  let toppings := 8
      toppings_per_pizza := 5
  in binomial_coefficient toppings toppings_per_pizza = 56 := by
  sorry

end num_unique_five_topping_pizzas_l137_137060


namespace vectors_perpendicular_vector_combination_l137_137269

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_c : ℝ × ℝ := (1, 1)

-- Auxiliary definition of vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Auxiliary definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

-- Auxiliary definition of scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Proof that (vector_a + vector_b) is perpendicular to vector_c
theorem vectors_perpendicular : dot_product (vector_add vector_a vector_b) vector_c = 0 :=
by sorry

-- Proof that vector_c = 5 * vector_a + 3 * vector_b
theorem vector_combination : vector_c = vector_add (scalar_mul 5 vector_a) (scalar_mul 3 vector_b) :=
by sorry

end vectors_perpendicular_vector_combination_l137_137269


namespace MarionBikeCost_l137_137449

theorem MarionBikeCost (M : ℤ) (h1 : 2 * M + M = 1068) : M = 356 :=
by
  sorry

end MarionBikeCost_l137_137449


namespace number_of_zeros_at_end_l137_137998

def N (n : Nat) := 10^(n+1) + 1

theorem number_of_zeros_at_end (n : Nat) (h : n = 2017) : 
  (N n)^(n + 1) - 1 ≡ 0 [MOD 10^(n + 1)] :=
sorry

end number_of_zeros_at_end_l137_137998


namespace quotient_is_six_l137_137182

def larger_number (L : ℕ) : Prop := L = 1620
def difference (L S : ℕ) : Prop := L - S = 1365
def division_remainder (L S Q : ℕ) : Prop := L = S * Q + 15

theorem quotient_is_six (L S Q : ℕ) 
  (hL : larger_number L) 
  (hdiff : difference L S) 
  (hdiv : division_remainder L S Q) : Q = 6 :=
sorry

end quotient_is_six_l137_137182


namespace number_of_triangles_l137_137588

-- Define the problem conditions
def points_on_circle := 10

-- State the problem in Lean
theorem number_of_triangles (n : ℕ) (h : n = points_on_circle) : (n.choose 3) = 120 :=
by
  rw h
  -- Placeholder for computation steps
  sorry

end number_of_triangles_l137_137588


namespace winning_candidate_percentage_l137_137883

def percentage_votes (votes1 votes2 votes3 : ℕ) : ℚ := 
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := max (max votes1 votes2) votes3
  (winning_votes * 100) / total_votes

theorem winning_candidate_percentage :
  percentage_votes 3000 5000 15000 = (15000 * 100) / (3000 + 5000 + 15000) :=
by 
  -- This computation should give us the exact percentage fraction.
  -- Simplifying it would yield the result approximately 65.22%
  -- Proof steps can be provided here.
  sorry

end winning_candidate_percentage_l137_137883


namespace exists_seq_nat_lcm_decreasing_l137_137638

-- Natural number sequence and conditions
def seq_nat_lcm_decreasing : Prop :=
  ∃ (a : Fin 100 → ℕ), 
  ((∀ i j : Fin 100, i < j → a i < a j) ∧
  (∀ (i : Fin 99), Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))))

theorem exists_seq_nat_lcm_decreasing : seq_nat_lcm_decreasing :=
  sorry

end exists_seq_nat_lcm_decreasing_l137_137638


namespace c_divisible_by_a_l137_137446

theorem c_divisible_by_a {a b c : ℤ} (h1 : a ∣ b * c) (h2 : Int.gcd a b = 1) : a ∣ c :=
by
  sorry

end c_divisible_by_a_l137_137446


namespace base6_addition_problem_l137_137641

theorem base6_addition_problem (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 := 
by
  sorry

end base6_addition_problem_l137_137641


namespace gerry_bananas_eaten_l137_137107

theorem gerry_bananas_eaten (b : ℝ) : 
  (b + (b + 8) + (b + 16) + 0 + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 220) →
  b + 48 = 56.67 :=
by
  sorry

end gerry_bananas_eaten_l137_137107


namespace evaluate_f_at_3_over_4_l137_137444

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ := (1 - y) / y

theorem evaluate_f_at_3_over_4 (h : g (x : ℝ) = 1 - x^2) (x_ne_zero : x ≠ 0) :
  f (3 / 4) = 3 :=
by
  sorry

end evaluate_f_at_3_over_4_l137_137444


namespace football_cost_correct_l137_137707

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l137_137707


namespace price_decrease_l137_137028

theorem price_decrease (current_price original_price : ℝ) (h1 : current_price = 684) (h2 : original_price = 900) :
  ((original_price - current_price) / original_price) * 100 = 24 :=
by
  sorry

end price_decrease_l137_137028


namespace train_speed_l137_137074

theorem train_speed (length : ℤ) (time : ℤ) 
  (h_length : length = 280) (h_time : time = 14) : 
  (length * 3600) / (time * 1000) = 72 := 
by {
  -- The proof would go here, this part is omitted as per instructions
  sorry
}

end train_speed_l137_137074


namespace vanya_speed_problem_l137_137745

theorem vanya_speed_problem (v : ℝ) (hv : (v + 2) / v = 2.5) : ((v + 4) / v) = 4 :=
by
  have v_pos : v ≠ 0 := sorry  -- Prove that v is not zero
  have v_eq : v = 4 / 3 := sorry  -- Use hv to solve for v
  rw [v_eq] at v_eq,
  rw [v_eq]
  sorry

end vanya_speed_problem_l137_137745


namespace num_workers_in_factory_l137_137018

theorem num_workers_in_factory 
  (average_salary_total : ℕ → ℕ → ℕ)
  (old_supervisor_salary : ℕ)
  (average_salary_9_new : ℕ)
  (new_supervisor_salary : ℕ) :
  ∃ (W : ℕ), 
  average_salary_total (W + 1) 430 = W * 430 + 870 ∧ 
  average_salary_9_new = 9 * 390 ∧ 
  W + 1 = (9 * 390 - 510 + 870) / 430 := 
by {
  sorry
}

end num_workers_in_factory_l137_137018


namespace sherman_total_weekly_driving_time_l137_137718

def daily_commute_time : Nat := 1  -- 1 hour for daily round trip commute time
def work_days : Nat := 5  -- Sherman works 5 days a week
def weekend_day_driving_time : Nat := 2  -- 2 hours of driving each weekend day
def weekend_days : Nat := 2  -- There are 2 weekend days

theorem sherman_total_weekly_driving_time :
  daily_commute_time * work_days + weekend_day_driving_time * weekend_days = 9 := 
by
  sorry

end sherman_total_weekly_driving_time_l137_137718


namespace ratio_of_kids_in_morning_to_total_soccer_l137_137316

-- Define the known conditions
def total_kids_in_camp : ℕ := 2000
def kids_going_to_soccer_camp : ℕ := total_kids_in_camp / 2
def kids_going_to_soccer_camp_in_afternoon : ℕ := 750
def kids_going_to_soccer_camp_in_morning : ℕ := kids_going_to_soccer_camp - kids_going_to_soccer_camp_in_afternoon

-- Define the conclusion to be proven
theorem ratio_of_kids_in_morning_to_total_soccer :
  (kids_going_to_soccer_camp_in_morning : ℚ) / (kids_going_to_soccer_camp : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_kids_in_morning_to_total_soccer_l137_137316


namespace percentage_profit_first_bicycle_l137_137294

theorem percentage_profit_first_bicycle :
  ∃ (C1 C2 : ℝ), 
    (C1 + C2 = 1980) ∧ 
    (0.9 * C2 = 990) ∧ 
    (12.5 / 100 * C1 = (990 - C1) / C1 * 100) :=
by
  sorry

end percentage_profit_first_bicycle_l137_137294


namespace find_smaller_number_l137_137880

theorem find_smaller_number (a b : ℤ) (h₁ : a + b = 8) (h₂ : a - b = 4) : b = 2 :=
by
  sorry

end find_smaller_number_l137_137880


namespace relationship_between_A_and_p_l137_137037

variable {x y p : ℝ}

theorem relationship_between_A_and_p (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : x ≠ y * 2) (h4 : x ≠ p * y)
  (A : ℝ) (hA : A = (x^2 - 3 * y^2) / (3 * x^2 + y^2))
  (hEq : (p * x * y) / (x^2 - (2 + p) * x * y + 2 * p * y^2) - y / (x - 2 * y) = 1 / 2) :
  A = (9 * p^2 - 3) / (27 * p^2 + 1) := 
sorry

end relationship_between_A_and_p_l137_137037


namespace smallest_pos_integer_n_l137_137015

theorem smallest_pos_integer_n 
  (x y : ℤ)
  (hx: ∃ k : ℤ, x = 8 * k - 2)
  (hy : ∃ l : ℤ, y = 8 * l + 2) :
  ∃ n : ℤ, n > 0 ∧ ∃ (m : ℤ), x^2 - x*y + y^2 + n = 8 * m ∧ n = 4 := by
  sorry

end smallest_pos_integer_n_l137_137015


namespace jan_skips_in_5_minutes_l137_137840

theorem jan_skips_in_5_minutes 
  (original_speed : ℕ)
  (time_in_minutes : ℕ)
  (doubled : ℕ)
  (new_speed : ℕ)
  (skips_in_5_minutes : ℕ) : 
  original_speed = 70 →
  doubled = 2 →
  new_speed = original_speed * doubled →
  time_in_minutes = 5 →
  skips_in_5_minutes = new_speed * time_in_minutes →
  skips_in_5_minutes = 700 :=
by
  intros 
  sorry

end jan_skips_in_5_minutes_l137_137840


namespace xy_yz_zx_value_l137_137152

namespace MathProof

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 147) 
  (h2 : y^2 + y * z + z^2 = 16) 
  (h3 : z^2 + x * z + x^2 = 163) :
  x * y + y * z + z * x = 56 := 
sorry      

end MathProof

end xy_yz_zx_value_l137_137152


namespace max_five_topping_pizzas_l137_137059

theorem max_five_topping_pizzas : 
  (∃ (n k : ℕ), n = 8 ∧ k = 5 ∧ (nat.choose n k = 56)) :=
begin
  use [8, 5],
  split,
  { refl, },
  split,
  { refl, },
  { sorry }
end

end max_five_topping_pizzas_l137_137059


namespace delta_value_l137_137274

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l137_137274


namespace smallest_positive_multiple_of_17_with_condition_l137_137765

theorem smallest_positive_multiple_of_17_with_condition :
  ∃ k : ℕ, k > 0 ∧ (k % 17 = 0) ∧ (k - 3) % 101 = 0 ∧ k = 306 :=
by
  sorry

end smallest_positive_multiple_of_17_with_condition_l137_137765


namespace width_of_field_l137_137185

variable (w : ℕ)

def length (w : ℕ) : ℕ := (7 * w) / 5

def perimeter (w : ℕ) : ℕ := 2 * (length w) + 2 * w

theorem width_of_field (h : perimeter w = 240) : w = 50 :=
sorry

end width_of_field_l137_137185


namespace part1_part2_part3_l137_137125

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem part1 : ∃ x : ℝ, (h x ≤ 2) := sorry

theorem part2 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) := sorry

theorem part3 (k : ℤ) : (∀ x : ℝ, x > 1 → k * (x - 1) < x * f x + 3 * g' x + 4) ↔ k ≤ 5 := sorry

end part1_part2_part3_l137_137125


namespace participant_guesses_needed_l137_137177

theorem participant_guesses_needed (n k : ℕ) (hk : n > k) : (if k = n / 2 then 2 else 1) = 
  (if k = n / 2 then 2 else 1) := 
begin
  sorry
end

end participant_guesses_needed_l137_137177


namespace find_percentage_l137_137056

theorem find_percentage (P : ℕ) (h1 : P * 64 = 320 * 10) : P = 5 := 
  by
  sorry

end find_percentage_l137_137056


namespace equation_has_real_solution_l137_137325

theorem equation_has_real_solution (m : ℝ) : ∃ x : ℝ, x^2 - m * x + m - 1 = 0 :=
by
  -- provide the hint that the discriminant (Δ) is (m - 2)^2
  have h : (m - 2)^2 ≥ 0 := by apply pow_two_nonneg
  sorry

end equation_has_real_solution_l137_137325


namespace problem_a_plus_b_equals_10_l137_137469

theorem problem_a_plus_b_equals_10 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
  (h_equation : 3 * a + 4 * b = 10 * a + b) : a + b = 10 :=
by {
  sorry
}

end problem_a_plus_b_equals_10_l137_137469


namespace boxes_left_l137_137859

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l137_137859


namespace cos_2alpha_zero_l137_137257

theorem cos_2alpha_zero (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
(h : Real.sin (2 * α) = Real.cos (Real.pi / 4 - α)) : 
  Real.cos (2 * α) = 0 :=
by
  sorry

end cos_2alpha_zero_l137_137257


namespace binomial_expansion_max_coefficient_l137_137835

theorem binomial_expansion_max_coefficient (n : ℕ) (h : n > 0) 
  (h_max_coefficient: ∀ m : ℕ, m ≠ 5 → (Nat.choose n m ≤ Nat.choose n 5)) : 
  n = 10 :=
sorry

end binomial_expansion_max_coefficient_l137_137835


namespace probability_of_divisibility_by_7_l137_137422

noncomputable def count_valid_numbers : Nat :=
  -- Implementation of the count of all five-digit numbers 
  -- such that the sum of the digits is 30 
  sorry

noncomputable def count_divisible_by_7 : Nat :=
  -- Implementation of the count of numbers among these 
  -- which are divisible by 7
  sorry

theorem probability_of_divisibility_by_7 :
  count_divisible_by_7 * 5 = count_valid_numbers :=
sorry

end probability_of_divisibility_by_7_l137_137422


namespace shape_formed_is_line_segment_l137_137093

def point := (ℝ × ℝ)

noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 4)
noncomputable def C : point := (6, 4)
noncomputable def D : point := (6, 0)

noncomputable def line_eq (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1, y2 - y1)

theorem shape_formed_is_line_segment :
  let l1 := line_eq A (1, 1)  -- Line from A at 45°
  let l2 := line_eq B (-1, -1) -- Line from B at -45°
  let l3 := line_eq D (1, -1) -- Line from D at 45°
  let l4 := line_eq C (-1, 5) -- Line from C at -45°
  let intersection1 := (5, 5)  -- Intersection of l1 and l4: solve x = 10 - x
  let intersection2 := (5, -1)  -- Intersection of l2 and l3: solve 4 - x = x - 6
  intersection1.1 = intersection2.1 := 
by
  sorry

end shape_formed_is_line_segment_l137_137093


namespace prob_of_odd_sum_of_dice_rolls_l137_137885

-- Define the probability calculation context
def prob_sum_odd_three_coins : ℚ :=
  let prob_0_heads := (1/2)^3 * 0 in
  let prob_1_head := (3 * (1/2)^3) * (1/2) in
  let prob_2_heads := (3 * (1/2)^3) * (1/2) in
  let prob_3_heads := (1/2)^3 * (1/2) in
  prob_1_head + prob_2_heads + prob_3_heads

theorem prob_of_odd_sum_of_dice_rolls :
  prob_sum_odd_three_coins = 7/16 :=
sorry

end prob_of_odd_sum_of_dice_rolls_l137_137885


namespace find_a1_l137_137114

variable {a_n : ℕ → ℤ}
variable (common_difference : ℤ) (a1 : ℤ)

-- Define that a_n is an arithmetic sequence with common difference of 2
def is_arithmetic_seq (a_n : ℕ → ℤ) (common_difference : ℤ) : Prop :=
  ∀ n, a_n (n + 1) - a_n n = common_difference

-- State the condition that a1, a2, a4 form a geometric sequence
def forms_geometric_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ a1 a2 a4, a2 * a2 = a1 * a4 ∧ a_n 1 = a1 ∧ a_n 2 = a2 ∧ a_n 4 = a4

-- Define the problem statement
theorem find_a1 (h_arith : is_arithmetic_seq a_n 2) (h_geom : forms_geometric_seq a_n) :
  a_n 1 = 2 :=
by
  sorry

end find_a1_l137_137114


namespace bricks_in_wall_l137_137631

-- Definitions for individual working times and breaks
def Bea_build_time := 8  -- hours
def Bea_break_time := 10 / 60  -- hours per hour
def Ben_build_time := 12  -- hours
def Ben_break_time := 15 / 60  -- hours per hour

-- Total effective rates
def Bea_effective_rate (h : ℕ) := h / (Bea_build_time * (1 - Bea_break_time))
def Ben_effective_rate (h : ℕ) := h / (Ben_build_time * (1 - Ben_break_time))

-- Decreased rate due to talking
def total_effective_rate (h : ℕ) := Bea_effective_rate h + Ben_effective_rate h - 12

-- Define the Lean proof statement
theorem bricks_in_wall (h : ℕ) :
  (6 * total_effective_rate h = h) → h = 127 :=
by sorry

end bricks_in_wall_l137_137631


namespace longest_chord_in_circle_l137_137529

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l137_137529


namespace largest_k_sum_consecutive_integers_l137_137370

theorem largest_k_sum_consecutive_integers (k : ℕ) (h1 : k > 0) :
  (∃ n : ℕ, (2^11) = sum (range k).map (λ i, n + i)) ∧ 
  (∀ m : ℕ, m > k → ¬(∃ n : ℕ, (2^11) = sum (range m).map (λ i, n + i))) ↔ k = 1 :=
  sorry

end largest_k_sum_consecutive_integers_l137_137370


namespace average_marks_is_75_l137_137839

-- Define the scores for the four tests based on the given conditions.
def first_test : ℕ := 80
def second_test : ℕ := first_test + 10
def third_test : ℕ := 65
def fourth_test : ℕ := third_test

-- Define the total marks scored in the four tests.
def total_marks : ℕ := first_test + second_test + third_test + fourth_test

-- Number of tests.
def num_tests : ℕ := 4

-- Define the average marks scored in the four tests.
def average_marks : ℕ := total_marks / num_tests

-- Prove that the average marks scored in the four tests is 75.
theorem average_marks_is_75 : average_marks = 75 :=
by
  sorry

end average_marks_is_75_l137_137839


namespace circumcircle_radius_l137_137439

-- Here we define the necessary conditions and prove the radius.
theorem circumcircle_radius
  (A B C : Type)
  (AB : ℝ)
  (angle_B : ℝ)
  (angle_A : ℝ)
  (h_AB : AB = 2)
  (h_angle_B : angle_B = 120)
  (h_angle_A : angle_A = 30) :
  ∃ R, R = 2 :=
by
  -- We will skip the proof using sorry
  sorry

end circumcircle_radius_l137_137439


namespace max_marks_paper_I_l137_137054

theorem max_marks_paper_I (M : ℝ) (h1 : 0.40 * M = 60) : M = 150 :=
by
  sorry

end max_marks_paper_I_l137_137054


namespace overlapping_triangle_area_l137_137258

/-- Given a rectangle with length 8 and width 4, folded along its diagonal, 
    the area of the overlapping part (grey triangle) is 10. --/
theorem overlapping_triangle_area : 
  let length := 8 
  let width := 4 
  let diagonal := (length^2 + width^2)^(1/2) 
  let base := (length^2 / (width^2 + length^2))^(1/2) * width 
  let height := width
  1 / 2 * base * height = 10 := by 
  sorry

end overlapping_triangle_area_l137_137258


namespace max_value_of_f_l137_137680

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end max_value_of_f_l137_137680


namespace age_difference_l137_137340

-- Define the present age of the son.
def S : ℕ := 22

-- Define the present age of the man.
variable (M : ℕ)

-- Given condition: In two years, the man's age will be twice the age of his son.
axiom condition : M + 2 = 2 * (S + 2)

-- Prove that the difference in present ages of the man and his son is 24 years.
theorem age_difference : M - S = 24 :=
by 
  -- We will fill in the proof here
  sorry

end age_difference_l137_137340


namespace largest_value_of_a_l137_137149

theorem largest_value_of_a
  (a b c d e : ℕ)
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : e = d - 10)
  (h5 : e < 105) :
  a ≤ 6824 :=
by {
  -- Proof omitted
  sorry
}

end largest_value_of_a_l137_137149


namespace rectangle_perimeter_l137_137836

theorem rectangle_perimeter (s : ℕ) (ABCD_area : 4 * s * s = 400) :
  2 * (2 * s + 2 * s) = 80 :=
by
  -- Skipping the proof
  sorry

end rectangle_perimeter_l137_137836


namespace product_of_numbers_l137_137042

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := sorry

end product_of_numbers_l137_137042


namespace new_apps_added_l137_137508

theorem new_apps_added (x : ℕ) (h1 : 15 + x - (x + 1) = 14) : x = 0 :=
by
  sorry

end new_apps_added_l137_137508


namespace value_of_k_l137_137518

theorem value_of_k (k : ℝ) :
  (5 + ∑' n : ℕ, (5 + k * (2^n / 4^n))) / 4^n = 10 → k = 15 :=
by
  sorry

end value_of_k_l137_137518


namespace ratio_of_costs_l137_137504

-- Definitions based on conditions
def old_car_cost : ℕ := 1800
def new_car_cost : ℕ := 1800 + 2000

-- Theorem stating the desired proof
theorem ratio_of_costs :
  (new_car_cost / old_car_cost : ℚ) = 19 / 9 :=
by
  sorry

end ratio_of_costs_l137_137504


namespace trains_meet_in_approx_17_45_seconds_l137_137046

noncomputable def train_meet_time
  (length1 length2 distance_between : ℕ)
  (speed1_kmph speed2_kmph : ℕ)
  : ℕ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_approx_17_45_seconds :
  train_meet_time 100 200 660 90 108 = 17 := by
  sorry

end trains_meet_in_approx_17_45_seconds_l137_137046


namespace number_of_diagonals_in_nonagon_l137_137972

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l137_137972


namespace find_single_digit_l137_137225

def isSingleDigit (n : ℕ) : Prop := n < 10

def repeatedDigitNumber (A : ℕ) : ℕ := 10 * A + A 

theorem find_single_digit (A : ℕ) (h1 : isSingleDigit A) (h2 : repeatedDigitNumber A + repeatedDigitNumber A = 132) : A = 6 :=
by
  sorry

end find_single_digit_l137_137225


namespace more_cats_than_spinsters_l137_137877

theorem more_cats_than_spinsters :
  ∀ (S C : ℕ), (S = 18) → (2 * C = 9 * S) → (C - S = 63) :=
by
  intros S C hS hRatio
  sorry

end more_cats_than_spinsters_l137_137877


namespace division_exponent_rule_l137_137038

theorem division_exponent_rule (a : ℝ) (h : a ≠ 0) : (a^8) / (a^2) = a^6 :=
sorry

end division_exponent_rule_l137_137038


namespace matrix_expression_solution_l137_137830

theorem matrix_expression_solution (x : ℝ) :
  let a := 3 * x + 1
  let b := x + 1
  let c := 2
  let d := 2 * x
  ab - cd = 5 :=
by
  sorry

end matrix_expression_solution_l137_137830


namespace value_of_expression_l137_137094

theorem value_of_expression :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end value_of_expression_l137_137094


namespace centroid_inverse_square_sum_l137_137563

theorem centroid_inverse_square_sum
  (α β γ p q r : ℝ)
  (h1 : 1/α^2 + 1/β^2 + 1/γ^2 = 1)
  (hp : p = α / 3)
  (hq : q = β / 3)
  (hr : r = γ / 3) :
  (1/p^2 + 1/q^2 + 1/r^2 = 9) :=
sorry

end centroid_inverse_square_sum_l137_137563


namespace f_f_f_f_f_of_1_l137_137448

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_f_f_f_f_of_1 : f (f (f (f (f 1)))) = 4687 :=
by
  sorry

end f_f_f_f_f_of_1_l137_137448


namespace min_distance_parabola_midpoint_l137_137118

theorem min_distance_parabola_midpoint 
  (a : ℝ) (m : ℝ) (h_pos_a : a > 0) :
  (m ≥ 1 / a → ∃ M_y : ℝ, M_y = (2 * m * a - 1) / (4 * a)) ∧ 
  (m < 1 / a → ∃ M_y : ℝ, M_y = a * m^2 / 4) := 
by 
  sorry

end min_distance_parabola_midpoint_l137_137118


namespace rebecca_haircuts_l137_137715

-- Definitions based on the conditions
def charge_per_haircut : ℕ := 30
def charge_per_perm : ℕ := 40
def charge_per_dye_job : ℕ := 60
def dye_cost_per_job : ℕ := 10
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def tips : ℕ := 50
def total_amount : ℕ := 310

-- Define the unknown number of haircuts scheduled
variable (H : ℕ)

-- Statement of the proof problem
theorem rebecca_haircuts :
  charge_per_haircut * H + charge_per_perm * num_perms + charge_per_dye_job * num_dye_jobs
  - dye_cost_per_job * num_dye_jobs + tips = total_amount → H = 4 :=
by
  sorry

end rebecca_haircuts_l137_137715


namespace magic_square_y_value_l137_137433

theorem magic_square_y_value 
  (a b c d e y : ℝ)
  (h1 : y + 4 + c = 81 + a + c)
  (h2 : y + (y - 77) + e = 81 + b + e)
  (h3 : y + 25 + 81 = 4 + (y - 77) + (2 * y - 158)) : 
  y = 168.5 :=
by
  -- required steps to complete the proof
  sorry

end magic_square_y_value_l137_137433


namespace no_solution_for_12k_plus_7_l137_137007

theorem no_solution_for_12k_plus_7 (k : ℤ) :
  ∀ (a b c : ℕ), 12 * k + 7 ≠ 2^a + 3^b - 5^c := 
by sorry

end no_solution_for_12k_plus_7_l137_137007


namespace find_a_plus_b_l137_137123

noncomputable def f (a b x : ℝ) := a ^ x + b

theorem find_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (dom1 : f a b (-2) = -2) (dom2 : f a b 0 = 0) :
  a + b = (Real.sqrt 3) / 3 - 3 :=
by
  unfold f at dom1 dom2
  sorry

end find_a_plus_b_l137_137123


namespace find_second_number_l137_137601

theorem find_second_number (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4)
  (h3 : y / z = 4 / 7) :
  y = 240 / 7 :=
by sorry

end find_second_number_l137_137601


namespace smallest_number_with_sum_32_l137_137372

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, digits.count d = 1

def digits_sum_to_32 (n : ℕ) : Prop :=
  n.digits 10 |>.sum = 32

def is_smallest_number (n : ℕ) : Prop :=
  ∀ m : ℕ, digits_sum_to_32 m → all_digits_different m → n ≤ m

theorem smallest_number_with_sum_32 : ∃ n, all_digits_different n ∧ digits_sum_to_32 n ∧ is_smallest_number n ∧ n = 26789 :=
by
  sorry

end smallest_number_with_sum_32_l137_137372


namespace find_r_l137_137828

variable (k r : ℝ)

theorem find_r (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (1/2) * Real.log 9 / Real.log 2 :=
sorry

end find_r_l137_137828


namespace total_budget_l137_137061

-- Define the conditions for the problem
def fiscal_months : ℕ := 12
def total_spent_at_six_months : ℕ := 6580
def over_budget_at_six_months : ℕ := 280

-- Calculate the total budget for the project
theorem total_budget (budget : ℕ) 
  (h : 6 * (total_spent_at_six_months - over_budget_at_six_months) * 2 = budget) 
  : budget = 12600 := 
  by
    -- Proof will be here
    sorry

end total_budget_l137_137061


namespace matias_fewer_cards_l137_137144

theorem matias_fewer_cards (J M C : ℕ) (h1 : J = M) (h2 : C = 20) (h3 : C + M + J = 48) : C - M = 6 :=
by
-- To be proven
  sorry

end matias_fewer_cards_l137_137144


namespace prove_partial_fractions_identity_l137_137950

def partial_fraction_identity (x : ℚ) (A B C a b c : ℚ) : Prop :=
  a = 0 ∧ b = 1 ∧ c = -1 ∧
  (A / (x - a) + B / (x - b) + C / (x - c) = 4*x - 2 ∧ x^3 - x ≠ 0)

theorem prove_partial_fractions_identity :
  (partial_fraction_identity x 2 1 (-3) 0 1 (-1)) :=
by {
  sorry
}

end prove_partial_fractions_identity_l137_137950


namespace bounded_expression_l137_137403

theorem bounded_expression (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 :=
sorry

end bounded_expression_l137_137403


namespace denominator_divisor_zero_l137_137456

theorem denominator_divisor_zero (n : ℕ) : n ≠ 0 → (∀ d, d ≠ 0 → d / n ≠ d / 0) :=
by
  sorry

end denominator_divisor_zero_l137_137456


namespace find_group_2013_in_sequence_l137_137782

def co_prime_to_n (n : ℕ) : ℕ → Prop := λ m, Nat.gcd m n = 1

def find_group (a n : ℕ) (h_co_prime : co_prime_to_n n a) : ℕ :=
  let coprimes := (List.range (n + 1)).filter (co_prime_to_n n)
  let pos := coprimes.index_of a + 1 -- Position of a in the list of coprimes

  -- Find the group "g" such that (g-1)^2 < pos ≤ g^2
  let find_g (pos : ℕ) := Nat.find (λ g, (g - 1) * (g - 1) < pos ∧ pos ≤ g * g)
  find_g pos

theorem find_group_2013_in_sequence :
  find_group 2013 2012 (by simp [co_prime_to_n, Nat.gcd]) = 32 :=
sorry

end find_group_2013_in_sequence_l137_137782


namespace probability_of_Y_l137_137603

theorem probability_of_Y (P_X P_both : ℝ) (h1 : P_X = 1/5) (h2 : P_both = 0.13333333333333333) : 
    (0.13333333333333333 / (1 / 5)) = 0.6666666666666667 :=
by sorry

end probability_of_Y_l137_137603


namespace rectangle_length_l137_137598

theorem rectangle_length (P L W : ℕ) (h1 : P = 48) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : L = 16 := by
  sorry

end rectangle_length_l137_137598


namespace map_distance_representation_l137_137156

theorem map_distance_representation
  (d_map : ℕ) (d_actual : ℕ) (conversion_factor : ℕ) (final_length_map : ℕ):
  d_map = 10 →
  d_actual = 80 →
  conversion_factor = d_actual / d_map →
  final_length_map = 18 →
  (final_length_map * conversion_factor) = 144 :=
by
  intros h1 h2 h3 h4
  sorry

end map_distance_representation_l137_137156


namespace infinite_geometric_series_sum_l137_137505

theorem infinite_geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1 / 3
  ∑' (n : ℕ), a * r ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l137_137505


namespace find_m_n_l137_137669

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l137_137669


namespace area_new_rectangle_l137_137522

theorem area_new_rectangle (a b : ℝ) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
sorry

end area_new_rectangle_l137_137522


namespace electricity_consumption_l137_137193

variable (x y : ℝ)

-- y = 0.55 * x
def electricity_fee := 0.55 * x

-- if y = 40.7 then x should be 74
theorem electricity_consumption :
  (∃ x, electricity_fee x = 40.7) → (x = 74) :=
by
  sorry

end electricity_consumption_l137_137193


namespace abs_neg_five_l137_137301

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l137_137301


namespace order_of_magnitudes_l137_137174

variable (x : ℝ)
variable (a : ℝ)

theorem order_of_magnitudes (h1 : x < 0) (h2 : a = 2 * x) : x^2 < a * x ∧ a * x < a^2 := 
by
  sorry

end order_of_magnitudes_l137_137174


namespace problem_statement_l137_137827

variable (a b x : ℝ)

theorem problem_statement (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  a / (a - b) = x / (x - 1) :=
sorry

end problem_statement_l137_137827


namespace equation_of_tangent_circle_l137_137656

/-- Lean Statement for the circle problem -/
theorem equation_of_tangent_circle (center_C : ℝ × ℝ)
    (h1 : ∃ x, center_C = (x, 0) ∧ x - 0 + 1 = 0)
    (circle_tangent : ∃ r, ((2 - (center_C.1))^2 + (3 - (center_C.2))^2 = (2 * Real.sqrt 2) + r)) :
    ∃ r, (x + 1)^2 + y^2 = r^2 := 
sorry

end equation_of_tangent_circle_l137_137656


namespace river_length_l137_137724

theorem river_length (S C : ℝ) (h1 : S = C / 3) (h2 : S + C = 80) : S = 20 :=
by 
  sorry

end river_length_l137_137724


namespace cost_equal_at_60_l137_137488

variable (x : ℝ)

def PlanA_cost (x : ℝ) : ℝ := 0.25 * x + 9
def PlanB_cost (x : ℝ) : ℝ := 0.40 * x

theorem cost_equal_at_60 : PlanA_cost x = PlanB_cost x → x = 60 :=
by
  intro h
  sorry

end cost_equal_at_60_l137_137488


namespace sum_of_simplified_side_length_ratio_l137_137462

theorem sum_of_simplified_side_length_ratio :
  let area_ratio := (50 : ℝ) / 98,
      side_length_ratio := Real.sqrt area_ratio,
      a := 5,
      b := 1,
      c := 7 in
  a + b + c = 13 :=
by
  sorry

end sum_of_simplified_side_length_ratio_l137_137462


namespace distinct_diagonals_in_nonagon_l137_137973

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l137_137973


namespace vanya_faster_speed_l137_137757

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l137_137757


namespace probability_2x2_is_half_l137_137227

noncomputable def probability_2x2_between_0_and_half : ℝ :=
  let μ := MeasureTheory.Measure.dirac (Set.Icc (-1 : ℝ) 1) in
  μ.measure (λ x, 0 ≤ 2 * x ^ 2 ∧ 2 * x ^ 2 ≤ 1 / 2)

theorem probability_2x2_is_half :
  probability_2x2_between_0_and_half = 1 / 2 :=
by
  sorry

end probability_2x2_is_half_l137_137227


namespace sum_of_solutions_comparison_l137_137849

variable (a a' b b' c c' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0)

theorem sum_of_solutions_comparison :
  ( (c - b) / a > (c' - b') / a' ) ↔ ( (c'-b') / a' < (c-b) / a ) :=
by sorry

end sum_of_solutions_comparison_l137_137849


namespace river_length_l137_137722

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end river_length_l137_137722


namespace frank_total_pages_read_l137_137650

-- Definitions of given conditions
def first_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def second_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def third_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days

-- Given values
def pages_first_book := first_book_pages 22 569
def pages_second_book := second_book_pages 35 315
def pages_third_book := third_book_pages 18 450

-- Total number of pages read by Frank
def total_pages := pages_first_book + pages_second_book + pages_third_book

-- Statement to prove
theorem frank_total_pages_read : total_pages = 31643 := by
  sorry

end frank_total_pages_read_l137_137650


namespace range_of_a_l137_137815

theorem range_of_a (x y : ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 4 = 2 * x * y) :
  x^2 + 2 * x * y + y^2 - a * x - a * y + 1 ≥ 0 ↔ a ≤ 17 / 4 := 
sorry

end range_of_a_l137_137815


namespace nonagon_diagonals_l137_137985

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l137_137985


namespace derek_age_l137_137785

theorem derek_age (aunt_beatrice_age : ℕ) (emily_age : ℕ) (derek_age : ℕ)
  (h1 : aunt_beatrice_age = 54)
  (h2 : emily_age = aunt_beatrice_age / 2)
  (h3 : derek_age = emily_age - 7) : derek_age = 20 :=
by
  sorry

end derek_age_l137_137785


namespace trigonometric_identity_l137_137255

theorem trigonometric_identity
  (θ : ℝ) 
  (h_tan : Real.tan θ = 3) :
  (1 - Real.cos θ) / (Real.sin θ) - (Real.sin θ) / (1 + (Real.cos θ)^2) = (11 * Real.sqrt 10 - 101) / 33 := 
by
  sorry

end trigonometric_identity_l137_137255


namespace jeff_pencils_initial_l137_137843

def jeff_initial_pencils (J : ℝ) := J
def jeff_remaining_pencils (J : ℝ) := 0.70 * J
def vicki_initial_pencils (J : ℝ) := 2 * J
def vicki_remaining_pencils (J : ℝ) := 0.25 * vicki_initial_pencils J
def remaining_pencils (J : ℝ) := jeff_remaining_pencils J + vicki_remaining_pencils J

theorem jeff_pencils_initial (J : ℝ) (h : remaining_pencils J = 360) : J = 300 :=
by
  sorry

end jeff_pencils_initial_l137_137843


namespace garrison_reinforcement_l137_137224

theorem garrison_reinforcement (x : ℕ) (h1 : ∀ (n m p : ℕ), n * m = p → x = n - m) :
  (150 * (31 - x) = 450 * 5) → x = 16 :=
by sorry

end garrison_reinforcement_l137_137224


namespace break_even_items_l137_137870

theorem break_even_items (C N : ℝ) (h_cost_inversely_proportional : ∃ k, C * real.sqrt N = k) 
  (h_cost_10_items : C * real.sqrt 10 = 2100 * real.sqrt 10) 
  (h_revenue : 30 * N = C) :
  N = 10 * real.cbrt 49 :=
by
  sorry

end break_even_items_l137_137870


namespace train_length_l137_137932

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l137_137932


namespace negation_necessary_but_not_sufficient_l137_137911

def P (x : ℝ) : Prop := |x - 2| ≥ 1
def Q (x : ℝ) : Prop := x^2 - 3 * x + 2 ≥ 0

theorem negation_necessary_but_not_sufficient (x : ℝ) :
  (¬ P x → ¬ Q x) ∧ ¬ (¬ Q x → ¬ P x) :=
by
  sorry

end negation_necessary_but_not_sufficient_l137_137911


namespace profit_ratio_l137_137493

theorem profit_ratio (SP CP : ℝ) (h : SP / CP = 3) : (SP - CP) / CP = 2 :=
by
  sorry

end profit_ratio_l137_137493


namespace find_x_l137_137286

-- Define the angles as real numbers representing degrees.
variable (angle_SWR angle_WRU angle_x : ℝ)

-- Conditions given in the problem
def conditions (angle_SWR angle_WRU angle_x : ℝ) : Prop :=
  angle_SWR = 50 ∧ angle_WRU = 30 ∧ angle_SWR = angle_WRU + angle_x

-- Main theorem to prove that x = 20 given the conditions
theorem find_x (angle_SWR angle_WRU angle_x : ℝ) :
  conditions angle_SWR angle_WRU angle_x → angle_x = 20 := by
  sorry

end find_x_l137_137286


namespace probability_participation_on_both_days_l137_137646

theorem probability_participation_on_both_days :
  let students := fin 5 -> bool in
  let total_outcomes := 2^5 in
  let same_day_outcomes := 2 in
  let both_days_outcomes := total_outcomes - same_day_outcomes in
  both_days_outcomes / total_outcomes = (15 / 16 : ℚ) :=
by
  let students := fin 5 -> bool
  let total_outcomes := 2^5
  let same_day_outcomes := 2
  let both_days_outcomes := total_outcomes - same_day_outcomes
  have h1 : both_days_outcomes = 30 := by norm_num
  have h2 : total_outcomes = 32 := by norm_num
  show both_days_outcomes / total_outcomes = 15 / 16
  calc 
    30 / 32 = 15 / 16 : by norm_num
        ... = (15 / 16 : ℚ) : by norm_cast

end probability_participation_on_both_days_l137_137646


namespace probability_is_two_thirds_l137_137218

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l137_137218


namespace average_score_l137_137790

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end average_score_l137_137790


namespace eq_sqrt_pattern_l137_137809

theorem eq_sqrt_pattern (a t : ℝ) (ha : a = 6) (ht : t = a^2 - 1) (h_pos : 0 < a ∧ 0 < t) :
  a + t = 41 := by
  sorry

end eq_sqrt_pattern_l137_137809


namespace area_of_30_60_90_triangle_hypotenuse_6sqrt2_l137_137023

theorem area_of_30_60_90_triangle_hypotenuse_6sqrt2 :
  ∀ (a b c : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 3 * Real.sqrt 6 →
  c = 6 * Real.sqrt 2 →
  c = 2 * a →
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  intro a b c ha hb hc h2a
  sorry

end area_of_30_60_90_triangle_hypotenuse_6sqrt2_l137_137023


namespace positive_integral_solution_l137_137368

theorem positive_integral_solution (n : ℕ) (hn : 0 < n) 
  (h : (n : ℚ) / (n + 1) = 125 / 126) : n = 125 := sorry

end positive_integral_solution_l137_137368


namespace Karlson_cannot_prevent_Baby_getting_one_fourth_l137_137086

theorem Karlson_cannot_prevent_Baby_getting_one_fourth 
  (a : ℝ) (h : a > 0) (K : ℝ × ℝ) (hK : 0 < K.1 ∧ K.1 < a ∧ 0 < K.2 ∧ K.2 < a) :
  ∀ (O : ℝ × ℝ) (cut1 cut2 : ℝ), 
    ((O.1 = a/2) ∧ (O.2 = a/2) ∧ (cut1 = K.1 ∧ cut1 = a ∨ cut1 = K.2 ∧ cut1 = a) ∧ 
                             (cut2 = K.1 ∧ cut2 = a ∨ cut2 = K.2 ∧ cut2 = a)) →
  ∃ (piece : ℝ), piece ≥ a^2 / 4 :=
by
  sorry

end Karlson_cannot_prevent_Baby_getting_one_fourth_l137_137086


namespace horners_rule_correct_l137_137731

open Classical

variables (x : ℤ) (poly_val : ℤ)

def original_polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℤ) : ℤ := ((7 * x + 3) * x - 5) * x + 11

theorem horners_rule_correct : (poly_val = horner_evaluation 23) ↔ (poly_val = original_polynomial 23) :=
by {
  sorry
}

end horners_rule_correct_l137_137731


namespace permutation_30_3_l137_137345

-- Define the number of students and the number of tasks
def n : ℕ := 30
def r : ℕ := 3

-- Define the permutation function
def P (n r : ℕ) := n.factorial / (n - r).factorial

-- Statement to prove
theorem permutation_30_3 : P 30 3 = 24360 := by
  unfold P
  norm_num
  sorry

end permutation_30_3_l137_137345


namespace total_sheets_folded_l137_137231

theorem total_sheets_folded (initially_folded : ℕ) (additionally_folded : ℕ) (total_folded : ℕ) :
  initially_folded = 45 → additionally_folded = 18 → total_folded = initially_folded + additionally_folded → total_folded = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end total_sheets_folded_l137_137231


namespace vanya_faster_speed_l137_137754

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l137_137754


namespace probability_of_Xiaojia_selection_l137_137343

theorem probability_of_Xiaojia_selection : 
  let students := 2500
  let teachers := 350
  let support_staff := 150
  let total_individuals := students + teachers + support_staff
  let sampled_individuals := 300
  let student_sample := (students : ℝ)/total_individuals * sampled_individuals
  (student_sample / students) = (1 / 10) := 
by
  sorry

end probability_of_Xiaojia_selection_l137_137343


namespace total_distance_after_fourth_bounce_l137_137620

noncomputable def total_distance_traveled (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let fall_distances := (List.range (num_bounces + 1)).map (λ n => initial_height * bounce_ratio^n)
  let rise_distances := (List.range num_bounces).map (λ n => initial_height * bounce_ratio^(n+1))
  fall_distances.sum + rise_distances.sum

theorem total_distance_after_fourth_bounce :
  total_distance_traveled 25 (5/6 : ℝ) 4 = 154.42 :=
by
  sorry

end total_distance_after_fourth_bounce_l137_137620


namespace hex_arrangements_correct_l137_137823

def hex_arrangements : Nat :=
  let vertices := fin 6
  let nums := {1, 2, 3, 4, 5, 6}
  let larger_than_neighbors (v : fin 6) (f : fin 6 → Nat) : Prop :=
    (f v > f (v + 1)) ∧ (f v > f (v - 1))
  let count_valid_arrangements : Nat :=
    -- Here we would count valid permutations satisfying the problem conditions
    8 -- The number of valid arrangements found in the problem
  count_valid_arrangements

theorem hex_arrangements_correct : hex_arrangements = 8 :=
by
  -- Proof omitted for now
  sorry

end hex_arrangements_correct_l137_137823


namespace people_at_first_table_l137_137189

theorem people_at_first_table (N x : ℕ) 
  (h1 : 20 < N) 
  (h2 : N < 50)
  (h3 : (N - x) % 42 = 0)
  (h4 : N % 8 = 7) : 
  x = 5 :=
sorry

end people_at_first_table_l137_137189


namespace sqrt_difference_l137_137088

theorem sqrt_difference :
  (Real.sqrt 63 - 7 * Real.sqrt (1 / 7)) = 2 * Real.sqrt 7 :=
by
  sorry

end sqrt_difference_l137_137088


namespace multiplicative_magic_square_h_sum_l137_137095

theorem multiplicative_magic_square_h_sum :
  ∃ (h_vals : List ℕ), 
  (∀ h ∈ h_vals, ∃ (e : ℕ), e > 0 ∧ 25 * e = h ∧ 
    ∃ (b c d f g : ℕ), 
    75 * b * c = d * e * f ∧ 
    d * e * f = g * h * 3 ∧ 
    g * h * 3 = c * f * 3 ∧ 
    c * f * 3 = 75 * e * g
  ) ∧ h_vals.sum = 150 :=
by { sorry }

end multiplicative_magic_square_h_sum_l137_137095


namespace arithmetic_geometric_seq_l137_137438

variable {a_n : ℕ → ℝ}
variable {a_1 a_3 a_5 a_6 a_11 : ℝ}

theorem arithmetic_geometric_seq (h₁ : a_1 * a_5 + 2 * a_3 * a_6 + a_1 * a_11 = 16) 
                                  (h₂ : a_1 * a_5 = a_3^2) 
                                  (h₃ : a_1 * a_11 = a_6^2) 
                                  (h₄ : a_3 > 0)
                                  (h₅ : a_6 > 0) : 
    a_3 + a_6 = 4 := 
by {
    sorry
}

end arithmetic_geometric_seq_l137_137438


namespace PetyaWinsAgainstSasha_l137_137005

def MatchesPlayed (name : String) : Nat :=
if name = "Petya" then 12 else if name = "Sasha" then 7 else if name = "Misha" then 11 else 0

def TotalGames : Nat := 15

def GamesMissed (name : String) : Nat :=
if name = "Petya" then TotalGames - MatchesPlayed name else 
if name = "Sasha" then TotalGames - MatchesPlayed name else
if name = "Misha" then TotalGames - MatchesPlayed name else 0

def CanNotMissConsecutiveGames : Prop := True

theorem PetyaWinsAgainstSasha : (GamesMissed "Misha" = 4) ∧ CanNotMissConsecutiveGames → 
  ∃ (winsByPetya : Nat), winsByPetya = 4 :=
by
  sorry

end PetyaWinsAgainstSasha_l137_137005


namespace tan_eleven_pi_over_three_l137_137358

theorem tan_eleven_pi_over_three : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := 
    sorry

end tan_eleven_pi_over_three_l137_137358


namespace domain_of_f_l137_137894

noncomputable def f (x : ℝ) : ℝ := (4 * x - 3) / (2 * x - 5)

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 / 2 → ∃ y : ℝ, f x = y :=
begin
  intros x h,
  use f x,
  exact ⟨⟩,
end

end domain_of_f_l137_137894


namespace trapezoid_length_relation_l137_137442

variables {A B C D M N : Type}
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
variables (a b c d m n : A)
variables (h_parallel_ab_cd : A) (h_parallel_mn_ab : A) 

-- The required proof statement
theorem trapezoid_length_relation (H1 : a = h_parallel_ab_cd) 
(H2 : b = m * n + h_parallel_mn_ab - m * d)
(H3 : c = d * (h_parallel_mn_ab - a))
(H4 : n = d / (n - a))
(H5 : n = c - h_parallel_ab_cd) :
c * m * a + b * c * d = n * d * a :=
sorry

end trapezoid_length_relation_l137_137442


namespace book_cost_l137_137052

variable {b m : ℝ}

theorem book_cost (h1 : b + m = 2.10) (h2 : b = m + 2) : b = 2.05 :=
by
  sorry

end book_cost_l137_137052


namespace longest_chord_in_circle_l137_137528

theorem longest_chord_in_circle {O : Type} (radius : ℝ) (h_radius : radius = 3) :
  ∃ d, d = 6 ∧ ∀ chord, chord <= d :=
by sorry

end longest_chord_in_circle_l137_137528


namespace intersection_A_B_l137_137661

def A : Set ℤ := {-2, -1, 1, 2}

def B : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-2, -1, 2} := by
  sorry

end intersection_A_B_l137_137661


namespace quadratic_single_solution_positive_n_l137_137807

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l137_137807


namespace nonagon_diagonals_l137_137986

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l137_137986


namespace find_y_l137_137789

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ y = 6 := by
  sorry

end find_y_l137_137789


namespace maximum_perfect_matchings_in_triangulation_l137_137112

-- Let's define the context of the problem:
-- A convex 20-gon P, Triangulations T, Perfect matching conditions
open SimpleGraph

def convex_20_gon : Type := List (Fin 20)
def triangulation (P : convex_20_gon) : Type := { T : List (Fin 20 × Fin 20) // ∀ (d₁ d₂ ∈ T), d₁ ≠ d₂ → ¬intersect P d₁ d₂ }

def is_perfect_matching (T : List (Fin 20 × Fin 20)) (M : List (Fin 20 × Fin 20)) : Prop :=
M ⊆ T ∧ ∀ (e₁ e₂ ∈ M), (e₁ ≠ e₂ → shares_endpoint e₁ e₂ = false)

def max_perfect_matchings (P : convex_20_gon) : Nat :=
Nat.fib 10

theorem maximum_perfect_matchings_in_triangulation (P : convex_20_gon) (T : triangulation P) :
  ∃ M : List (Fin 20 × Fin 20), is_perfect_matching T.val M ∧
  ∀ (T' : triangulation P), count_perfect_matchings T' ≤ max_perfect_matchings P := by sorry

end maximum_perfect_matchings_in_triangulation_l137_137112


namespace radius_of_third_circle_l137_137604

noncomputable def circle_radius {r1 r2 : ℝ} (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let A_shaded := (25^2 * Real.pi) - (15^2 * Real.pi)
  let r := Real.sqrt (A_shaded / Real.pi)
  r

theorem radius_of_third_circle (r1 r2 r3 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) :
  circle_radius h1 h2 = 20 :=
by 
  sorry

end radius_of_third_circle_l137_137604


namespace candy_bars_total_l137_137195

theorem candy_bars_total :
  let people : ℝ := 3.0;
  let candy_per_person : ℝ := 1.66666666699999;
  people * candy_per_person = 5.0 :=
by
  let people : ℝ := 3.0
  let candy_per_person : ℝ := 1.66666666699999
  show people * candy_per_person = 5.0
  sorry

end candy_bars_total_l137_137195


namespace complex_point_second_quadrant_l137_137026

theorem complex_point_second_quadrant (i : ℂ) (h1 : i^4 = 1) :
  ∃ (z : ℂ), z = ((i^(2014))/(1 + i) * i) ∧ z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_second_quadrant_l137_137026


namespace negation_of_proposition_l137_137188

theorem negation_of_proposition (x : ℝ) :
  ¬ (x > 1 → x ^ 2 > x) ↔ (x ≤ 1 → x ^ 2 ≤ x) :=
by 
  sorry

end negation_of_proposition_l137_137188


namespace mean_of_second_set_l137_137552

def mean (l: List ℕ) : ℚ :=
  (l.sum: ℚ) / l.length

theorem mean_of_second_set (x: ℕ) 
  (h: mean [28, x, 42, 78, 104] = 90): 
  mean [128, 255, 511, 1023, x] = 423 :=
by
  sorry

end mean_of_second_set_l137_137552


namespace final_price_wednesday_l137_137283

theorem final_price_wednesday :
  let coffee_price := 6
  let cheesecake_price := 10
  let sandwich_price := 8
  let coffee_discount := 0.25
  let cheesecake_discount_wednesday := 0.10
  let additional_discount := 3
  let sales_tax := 0.05
  let discounted_coffee_price := coffee_price - coffee_price * coffee_discount
  let discounted_cheesecake_price := cheesecake_price - cheesecake_price * cheesecake_discount_wednesday
  let total_price_before_additional_discount := discounted_coffee_price + discounted_cheesecake_price + sandwich_price
  let total_price_after_additional_discount := total_price_before_additional_discount - additional_discount
  let total_price_with_tax := total_price_after_additional_discount + total_price_after_additional_discount * sales_tax
  let final_price := total_price_with_tax.round
  final_price = 19.43 :=
by
  sorry

end final_price_wednesday_l137_137283


namespace sin_A_in_right_triangle_l137_137685

theorem sin_A_in_right_triangle (B C A : Real) (hBC: B + C = π / 2) 
(h_sinB: Real.sin B = 3 / 5) (h_sinC: Real.sin C = 4 / 5) : 
Real.sin A = 1 := 
by 
  sorry

end sin_A_in_right_triangle_l137_137685


namespace each_client_selected_cars_l137_137778

theorem each_client_selected_cars (cars clients selections : ℕ) (h1 : cars = 16) (h2 : selections = 3 * cars) (h3 : clients = 24) :
  selections / clients = 2 :=
by
  sorry

end each_client_selected_cars_l137_137778


namespace quadratic_has_two_real_roots_l137_137649

-- Define the condition that the discriminant must be non-negative
def discriminant_nonneg (a b c : ℝ) : Prop := b * b - 4 * a * c ≥ 0

-- Define our specific quadratic equation conditions: x^2 - 2x + m = 0
theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant_nonneg 1 (-2) m → m ≤ 1 :=
by
  sorry

end quadratic_has_two_real_roots_l137_137649


namespace time_to_cover_escalator_l137_137232

def escalator_speed : ℝ := 12
def escalator_length : ℝ := 160
def person_speed : ℝ := 8

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 8 := by
  sorry

end time_to_cover_escalator_l137_137232


namespace hockey_league_games_l137_137045

def num_teams : ℕ := 18
def encounters_per_pair : ℕ := 10
def num_games (n : ℕ) (k : ℕ) : ℕ := (n * (n - 1)) / 2 * k

theorem hockey_league_games :
  num_games num_teams encounters_per_pair = 1530 :=
by
  sorry

end hockey_league_games_l137_137045


namespace Derek_is_42_l137_137939

def Aunt_Anne_age : ℕ := 36

def Brianna_age : ℕ := (2 * Aunt_Anne_age) / 3

def Caitlin_age : ℕ := Brianna_age - 3

def Derek_age : ℕ := 2 * Caitlin_age

theorem Derek_is_42 : Derek_age = 42 := by
  sorry

end Derek_is_42_l137_137939


namespace goldfish_initial_count_l137_137710

theorem goldfish_initial_count (catsfish : ℕ) (fish_left : ℕ) (fish_disappeared : ℕ) (goldfish_initial : ℕ) :
  catsfish = 12 →
  fish_left = 15 →
  fish_disappeared = 4 →
  goldfish_initial = (fish_left + fish_disappeared) - catsfish →
  goldfish_initial = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h1] at h4
  exact h4

end goldfish_initial_count_l137_137710


namespace total_different_books_l137_137197

def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def tony_dean_shared_books : ℕ := 3
def all_three_shared_book : ℕ := 1

theorem total_different_books :
  tony_books + dean_books + breanna_books - tony_dean_shared_books - 2 * all_three_shared_book = 47 := 
by
  sorry 

end total_different_books_l137_137197


namespace vanya_speed_increased_by_4_l137_137748

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l137_137748


namespace max_y_for_f_eq_0_l137_137151

-- Define f(x, y, z) as the remainder when (x - y)! is divided by (x + z).
def f (x y z : ℕ) : ℕ :=
  Nat.factorial (x - y) % (x + z)

-- Conditions given in the problem
variable (x y z : ℕ)
variable (hx : x = 100)
variable (hz : z = 50)

theorem max_y_for_f_eq_0 : 
  f x y z = 0 → y ≤ 75 :=
by
  rw [hx, hz]
  sorry

end max_y_for_f_eq_0_l137_137151


namespace find_1993_star_1935_l137_137318

axiom star (x y : ℕ) : ℕ

axiom star_self {x : ℕ} : star x x = 0
axiom star_assoc {x y z : ℕ} : star x (star y z) = star x y + z

theorem find_1993_star_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_1993_star_1935_l137_137318


namespace daliah_garbage_l137_137794

theorem daliah_garbage (D : ℝ) (h1 : 4 * (D - 2) = 62) : D = 17.5 :=
by
  sorry

end daliah_garbage_l137_137794


namespace solve_inequality_l137_137962

theorem solve_inequality (a x : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 1 > 0) : 
  (-2 < a ∧ a < 1 → a < x ∧ x < 2 - a) ∧ 
  (a = 1 → False) ∧ 
  (1 < a ∧ a < 2 → 2 - a < x ∧ x < a) :=
by
  sorry

end solve_inequality_l137_137962


namespace sector_angle_l137_137019

theorem sector_angle (r α : ℝ) (h₁ : 2 * r + α * r = 4) (h₂ : (1 / 2) * α * r^2 = 1) : α = 2 :=
sorry

end sector_angle_l137_137019


namespace car_rental_budget_l137_137773

def daily_rental_cost : ℝ := 30.0
def cost_per_mile : ℝ := 0.18
def total_miles : ℝ := 250.0

theorem car_rental_budget : daily_rental_cost + (cost_per_mile * total_miles) = 75.0 :=
by 
  sorry

end car_rental_budget_l137_137773


namespace charlies_age_22_l137_137634

variable (A : ℕ) (C : ℕ)

theorem charlies_age_22 (h1 : C = 2 * A + 8) (h2 : C = 22) : A = 7 := by
  sorry

end charlies_age_22_l137_137634


namespace largest_power_of_2_divides_n_l137_137516

def n : ℤ := 17^4 - 13^4

theorem largest_power_of_2_divides_n : ∃ (k : ℕ), 2^4 = k ∧ 2^k ∣ n ∧ ¬ (2^(k + 1) ∣ n) := by
  sorry

end largest_power_of_2_divides_n_l137_137516


namespace square_area_l137_137072

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l137_137072


namespace num_diagonals_convex_nonagon_l137_137982

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l137_137982


namespace kylie_total_apples_l137_137848

theorem kylie_total_apples : (let first_hour := 66 in 
                              let second_hour := 2 * 66 in 
                              let third_hour := 66 / 3 in 
                              first_hour + second_hour + third_hour = 220) :=
by
  let first_hour := 66
  let second_hour := 2 * first_hour
  let third_hour := first_hour / 3
  show first_hour + second_hour + third_hour = 220
  sorry

end kylie_total_apples_l137_137848


namespace remove_terms_sum_equals_one_l137_137009

theorem remove_terms_sum_equals_one :
  let seq := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let remove := [1/12, 1/15]
  (seq.sum - remove.sum) = 1 :=
by
  sorry

end remove_terms_sum_equals_one_l137_137009


namespace Catriona_total_fish_l137_137362

theorem Catriona_total_fish:
  ∃ (goldfish angelfish guppies : ℕ),
  goldfish = 8 ∧
  angelfish = goldfish + 4 ∧
  guppies = 2 * angelfish ∧
  goldfish + angelfish + guppies = 44 :=
by
  -- Define the number of goldfish
  let goldfish := 8

  -- Define the number of angelfish, which is 4 more than goldfish
  let angelfish := goldfish + 4

  -- Define the number of guppies, which is twice the number of angelfish
  let guppies := 2 * angelfish

  -- Prove the total number of fish is 44
  have total_fish : goldfish + angelfish + guppies = 44 := by
    rw [←nat.add_assoc, nat.add_comm 12 8, nat.add_assoc, nat.add_comm 24 12, ←nat.add_assoc]

  use [goldfish, angelfish, guppies]
  exact ⟨rfl, rfl, rfl, total_fish⟩

end Catriona_total_fish_l137_137362


namespace quadratic_coefficients_l137_137881

theorem quadratic_coefficients (b c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 3)) → 
  b = -4 ∧ c = -6 :=
by
  intro h
  -- The proof would go here, but we'll skip it.
  sorry

end quadratic_coefficients_l137_137881


namespace simplify_trig_expr_l137_137297

noncomputable def sin15 := Real.sin (Real.pi / 12)
noncomputable def sin30 := Real.sin (Real.pi / 6)
noncomputable def sin45 := Real.sin (Real.pi / 4)
noncomputable def sin60 := Real.sin (Real.pi / 3)
noncomputable def sin75 := Real.sin (5 * Real.pi / 12)
noncomputable def cos10 := Real.cos (Real.pi / 18)
noncomputable def cos20 := Real.cos (Real.pi / 9)
noncomputable def cos30 := Real.cos (Real.pi / 6)

theorem simplify_trig_expr :
  (sin15 + sin30 + sin45 + sin60 + sin75) / (cos10 * cos20 * cos30) = 5.128 :=
sorry

end simplify_trig_expr_l137_137297


namespace total_filled_water_balloons_l137_137855

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end total_filled_water_balloons_l137_137855


namespace kickers_goals_in_first_period_l137_137945

theorem kickers_goals_in_first_period (K : ℕ) 
  (h1 : ∀ n : ℕ, n = K) 
  (h2 : ∀ n : ℕ, n = 2 * K) 
  (h3 : ∀ n : ℕ, n = K / 2) 
  (h4 : ∀ n : ℕ, n = 4 * K) 
  (h5 : K + 2 * K + (K / 2) + 4 * K = 15) : 
  K = 2 := 
by
  sorry

end kickers_goals_in_first_period_l137_137945


namespace square_area_l137_137064

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l137_137064


namespace distinct_diagonals_in_nonagon_l137_137975

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l137_137975


namespace average_speed_of_trip_l137_137216

theorem average_speed_of_trip :
  let speed1 := 30
  let time1 := 5
  let speed2 := 42
  let time2 := 10
  let total_time := 15
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let average_speed := total_distance / total_time
  average_speed = 38 := 
by 
  sorry

end average_speed_of_trip_l137_137216


namespace integer_solution_of_inequalities_l137_137579

theorem integer_solution_of_inequalities :
  (∀ x : ℝ, 3 * x - 4 ≤ 6 * x - 2 → (2 * x + 1) / 3 - 1 < (x - 1) / 2 → (x = 0)) :=
sorry

end integer_solution_of_inequalities_l137_137579


namespace pyramid_volume_and_base_edge_l137_137178

theorem pyramid_volume_and_base_edge:
  ∀ (r: ℝ) (h: ℝ) (_: r = 5) (_: h = 10), 
  ∃ s V: ℝ,
    s = (10 * Real.sqrt 6) / 3 ∧ 
    V = (2000 / 9) :=
by
    sorry

end pyramid_volume_and_base_edge_l137_137178


namespace football_cost_l137_137704

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l137_137704


namespace nonagon_diagonals_count_l137_137967

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l137_137967


namespace divides_number_of_ones_l137_137681

theorem divides_number_of_ones (n : ℕ) (h1 : ¬(2 ∣ n)) (h2 : ¬(5 ∣ n)) : ∃ k : ℕ, n ∣ ((10^k - 1) / 9) :=
by
  sorry

end divides_number_of_ones_l137_137681


namespace find_four_consecutive_odd_numbers_l137_137936

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers_l137_137936


namespace vanya_faster_speed_l137_137752

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l137_137752


namespace total_sonnets_written_l137_137548

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l137_137548


namespace sides_of_regular_polygon_l137_137468

theorem sides_of_regular_polygon 
    (sum_interior_angles : ∀ n : ℕ, (n - 2) * 180 = 1440) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end sides_of_regular_polygon_l137_137468


namespace problem_statement_l137_137652

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : xy = -2) : (1 - x) * (1 - y) = -3 := by
  sorry

end problem_statement_l137_137652


namespace total_books_left_l137_137011

def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

theorem total_books_left : sandy_books + tim_books - benny_lost_books = 19 :=
by
  sorry

end total_books_left_l137_137011


namespace smallest_value_of_n_l137_137348

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l137_137348


namespace perpendicular_lines_a_eq_neg6_l137_137682

theorem perpendicular_lines_a_eq_neg6 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, ax + 2*y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + 3*y - 2 = 0) 
  (h_perpendicular : ∀ m1 m2 : ℝ, m1 * m2 = -1) : 
  a = -6 := 
by 
  sorry

end perpendicular_lines_a_eq_neg6_l137_137682


namespace min_value_expression_l137_137801

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_l137_137801


namespace relationship_between_a_b_c_l137_137814

theorem relationship_between_a_b_c :
  let m := 2
  let n := 3
  let f (x : ℝ) := x^3
  let a := f (Real.sqrt 3 / 3)
  let b := f (Real.log Real.pi)
  let c := f (Real.sqrt 2 / 2)
  a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l137_137814


namespace vanya_speed_increased_by_4_l137_137749

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end vanya_speed_increased_by_4_l137_137749


namespace divides_both_numerator_and_denominator_l137_137207

theorem divides_both_numerator_and_denominator (x m : ℤ) :
  (x ∣ (5 * m + 6)) ∧ (x ∣ (8 * m + 7)) → (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13) :=
by
  sorry

end divides_both_numerator_and_denominator_l137_137207


namespace find_f_three_l137_137580

noncomputable def f : ℝ → ℝ := sorry -- f(x) is a linear function

axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

axiom equation : ∀ x, f x = 3 * (f⁻¹ x) + 9

axiom f_zero : f 0 = 3

axiom f_inv_three : f⁻¹ 3 = 0

theorem find_f_three : f 3 = 6 * Real.sqrt 3 := 
by sorry

end find_f_three_l137_137580


namespace initial_number_of_girls_l137_137253

theorem initial_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 7 * (b - 54) = g - 20) : g = 39 :=
sorry

end initial_number_of_girls_l137_137253


namespace gcd_of_12347_and_9876_l137_137608

theorem gcd_of_12347_and_9876 : Nat.gcd 12347 9876 = 7 :=
by
  sorry

end gcd_of_12347_and_9876_l137_137608


namespace symmetric_point_correct_line_passes_second_quadrant_l137_137039

theorem symmetric_point_correct (x y: ℝ) (h_line : y = x + 1) :
  (x, y) = (-1, 2) :=
sorry

theorem line_passes_second_quadrant (m x y: ℝ) (h_line: m * x + y + m - 1 = 0) :
  (x, y) = (-1, 1) :=
sorry

end symmetric_point_correct_line_passes_second_quadrant_l137_137039


namespace value_range_of_f_l137_137472

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2
  else if -4 ≤ x ∧ x < 0 then x^2 + 6 * x
  else 0

theorem value_range_of_f : Set.range f = {y : ℝ | -9 ≤ y ∧ y ≤ 1} :=
by
  sorry

end value_range_of_f_l137_137472


namespace orchid_bushes_total_l137_137882

def current_bushes : ℕ := 47
def bushes_today : ℕ := 37
def bushes_tomorrow : ℕ := 25

theorem orchid_bushes_total : current_bushes + bushes_today + bushes_tomorrow = 109 := 
by sorry

end orchid_bushes_total_l137_137882


namespace nonagon_diagonals_count_l137_137969

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l137_137969


namespace bigger_part_l137_137619

theorem bigger_part (x y : ℕ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) : y = 34 :=
sorry

end bigger_part_l137_137619


namespace phillip_initial_marbles_l137_137637

theorem phillip_initial_marbles
  (dilan_marbles : ℕ) (martha_marbles : ℕ) (veronica_marbles : ℕ) 
  (total_after_redistribution : ℕ) 
  (individual_marbles_after : ℕ) :
  dilan_marbles = 14 →
  martha_marbles = 20 →
  veronica_marbles = 7 →
  total_after_redistribution = 4 * individual_marbles_after →
  individual_marbles_after = 15 →
  ∃phillip_marbles : ℕ, phillip_marbles = 19 :=
by
  intro h_dilan h_martha h_veronica h_total_after h_individual
  have total_initial := 60 - (14 + 20 + 7)
  existsi total_initial
  sorry

end phillip_initial_marbles_l137_137637


namespace temperature_difference_l137_137160

theorem temperature_difference (T_south T_north : ℤ) (h1 : T_south = -7) (h2 : T_north = -15) :
  T_south - T_north = 8 :=
by
  sorry

end temperature_difference_l137_137160


namespace earl_stuff_rate_l137_137511

variable (E L : ℕ)

-- Conditions
def ellen_rate : Prop := L = (2 * E) / 3
def combined_rate : Prop := E + L = 60

-- Main statement
theorem earl_stuff_rate (h1 : ellen_rate E L) (h2 : combined_rate E L) : E = 36 := by
  sorry

end earl_stuff_rate_l137_137511


namespace min_value_reciprocal_sum_l137_137673

theorem min_value_reciprocal_sum 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 2) : 
  ∃ x, x = 2 ∧ (∀ y, y = (1 / a) + (1 / b) → x ≤ y) := 
sorry

end min_value_reciprocal_sum_l137_137673


namespace Ryan_spit_distance_correct_l137_137239

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end Ryan_spit_distance_correct_l137_137239


namespace average_of_first_20_even_numbers_not_divisible_by_3_or_5_l137_137607

def first_20_valid_even_numbers : List ℕ :=
  [2, 4, 8, 14, 16, 22, 26, 28, 32, 34, 38, 44, 46, 52, 56, 58, 62, 64, 68, 74]

-- Check the sum of these numbers
def sum_first_20_valid_even_numbers : ℕ :=
  first_20_valid_even_numbers.sum

-- Define average calculation
def average_first_20_valid_even_numbers : ℕ :=
  sum_first_20_valid_even_numbers / 20

theorem average_of_first_20_even_numbers_not_divisible_by_3_or_5 :
  average_first_20_valid_even_numbers = 35 :=
by
  sorry

end average_of_first_20_even_numbers_not_divisible_by_3_or_5_l137_137607


namespace harry_worked_32_hours_l137_137100

variable (x y : ℝ)
variable (harry_pay james_pay : ℝ)

-- Definitions based on conditions
def harry_weekly_pay (h : ℝ) := 30*x + (h - 30)*y
def james_weekly_pay := 40*x + 1*y

-- Condition: Harry and James were paid the same last week
axiom harry_james_same_pay : ∀ (h : ℝ), harry_weekly_pay x y h = james_weekly_pay x y

-- Prove: Harry worked 32 hours
theorem harry_worked_32_hours : ∃ h : ℝ, h = 32 ∧ harry_weekly_pay x y h = james_weekly_pay x y := by
  sorry

end harry_worked_32_hours_l137_137100


namespace frog_reaches_safely_l137_137432

/-- Definition of the probability Q(M) given the frog's jump conditions -/
noncomputable def Q : ℕ → ℚ
| 0     := 0  -- if the frog reaches stone 0, it gets caught
| 14    := 1  -- if the frog reaches stone 14, it reaches safety
| (M+1) := if 1 ≤ M ∧ M < 13 then 
             (M+1 : ℚ)/15 * Q (M-1) + (1 - (M+1 : ℚ)/15) * Q (M+1 + 1)
          else 0

/-- Prove the probability that the frog initially on stone 2 reaches stone 14 safely -/
theorem frog_reaches_safely : Q 2 = 85 / 256 :=
sorry

end frog_reaches_safely_l137_137432


namespace intersection_points_on_circle_l137_137635

theorem intersection_points_on_circle
  (x y : ℝ)
  (h1 : y = (x + 2)^2)
  (h2 : x + 2 = (y - 1)^2) :
  (x + 2)^2 + (y - 1)^2 = 2 :=
sorry

end intersection_points_on_circle_l137_137635


namespace original_population_has_factor_three_l137_137780

theorem original_population_has_factor_three (x y z : ℕ) 
  (hx : ∃ n : ℕ, x = n ^ 2) -- original population is a perfect square
  (h1 : x + 150 = y^2 - 1)  -- after increase of 150, population is one less than a perfect square
  (h2 : y^2 - 1 + 150 = z^2) -- after another increase of 150, population is a perfect square again
  : 3 ∣ x :=
sorry

end original_population_has_factor_three_l137_137780


namespace minimize_abs_a_n_l137_137395

noncomputable def a_n (n : ℕ) : ℝ :=
  14 - (3 / 4) * (n - 1)

theorem minimize_abs_a_n : ∃ n : ℕ, n = 20 ∧ ∀ m : ℕ, |a_n n| ≤ |a_n m| := by
  sorry

end minimize_abs_a_n_l137_137395


namespace back_parking_lot_filled_fraction_l137_137025

theorem back_parking_lot_filled_fraction
    (front_spaces : ℕ) (back_spaces : ℕ) (cars_parked : ℕ) (spaces_available : ℕ)
    (h1 : front_spaces = 52)
    (h2 : back_spaces = 38)
    (h3 : cars_parked = 39)
    (h4 : spaces_available = 32) :
    (back_spaces - (front_spaces + back_spaces - cars_parked - spaces_available)) / back_spaces = 1 / 2 :=
by
  sorry

end back_parking_lot_filled_fraction_l137_137025


namespace coinCombinationCount_l137_137284

-- Definitions for the coin values and the target amount
def quarter := 25
def dime := 10
def nickel := 5
def penny := 1
def total := 400

-- Define a function counting the number of ways to reach the total using given coin values
def countWays : Nat := sorry -- placeholder for the actual computation

-- Theorem stating the problem statement
theorem coinCombinationCount (n : Nat) :
  countWays = n :=
sorry

end coinCombinationCount_l137_137284


namespace gcd_of_polynomials_l137_137251

theorem gcd_of_polynomials (n : ℕ) (h : n > 2^5) : gcd (n^3 + 5^2) (n + 6) = 1 :=
by sorry

end gcd_of_polynomials_l137_137251


namespace species_below_threshold_in_year_2019_l137_137946

-- Definitions based on conditions in the problem.
def initial_species (N : ℝ) : ℝ := N
def yearly_decay_rate : ℝ := 0.70
def threshold : ℝ := 0.05

-- The problem statement to prove.
theorem species_below_threshold_in_year_2019 (N : ℝ) (hN : N > 0):
  ∃ k : ℕ, k ≥ 9 ∧ yearly_decay_rate ^ k * initial_species N < threshold * initial_species N :=
sorry

end species_below_threshold_in_year_2019_l137_137946


namespace weekend_weekday_ratio_l137_137509

-- Defining the basic constants and conditions
def weekday_episodes : ℕ := 8
def total_episodes_in_week : ℕ := 88

-- Defining the main theorem
theorem weekend_weekday_ratio : (2 * (total_episodes_in_week - 5 * weekday_episodes)) / weekday_episodes = 3 :=
by
  sorry

end weekend_weekday_ratio_l137_137509


namespace log_sqrt10_eq_7_l137_137799

theorem log_sqrt10_eq_7 : log (√10) (1000 * √10) = 7 := 
sorry

end log_sqrt10_eq_7_l137_137799


namespace water_evaporation_weight_l137_137326

noncomputable def initial_weight : ℝ := 200
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def final_salt_concentration : ℝ := 0.08

theorem water_evaporation_weight (W_final : ℝ) (evaporation_weight : ℝ) 
  (h1 : W_final = 10 / final_salt_concentration) 
  (h2 : evaporation_weight = initial_weight - W_final) : 
  evaporation_weight = 75 :=
by
  sorry

end water_evaporation_weight_l137_137326


namespace red_balls_count_l137_137331

-- Lean 4 statement for proving the number of red balls in the bag is 336
theorem red_balls_count (x : ℕ) (total_balls red_balls : ℕ) 
  (h1 : total_balls = 60 + 18 * x) 
  (h2 : red_balls = 56 + 14 * x) 
  (h3 : (56 + 14 * x : ℚ) / (60 + 18 * x) = 4 / 5) : red_balls = 336 := 
by
  sorry

end red_balls_count_l137_137331


namespace number_of_triangles_l137_137582

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l137_137582


namespace find_m_n_l137_137672

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l137_137672


namespace t_shaped_grid_sum_l137_137300

open Finset

theorem t_shaped_grid_sum :
  ∃ (a b c d e : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧
    (d ≠ e) ∧
    a + b + c = 20 ∧
    d + e = 7 ∧
    (a + b + c + d + e + b) = 33 :=
sorry

end t_shaped_grid_sum_l137_137300


namespace avg_score_is_94_l137_137793

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l137_137793


namespace measure_angle_C_l137_137690

noncomputable def triangle_angles_sum (a b c : ℝ) : Prop :=
  a + b + c = 180

noncomputable def angle_B_eq_twice_angle_C (b c : ℝ) : Prop :=
  b = 2 * c

noncomputable def angle_A_eq_40 : ℝ := 40

theorem measure_angle_C :
  ∀ (B C : ℝ), triangle_angles_sum angle_A_eq_40 B C → angle_B_eq_twice_angle_C B C → C = 140 / 3 :=
by
  intros B C h1 h2
  sorry

end measure_angle_C_l137_137690


namespace division_problem_l137_137491

theorem division_problem : 75 / 0.05 = 1500 := 
  sorry

end division_problem_l137_137491


namespace ratio_equivalence_l137_137135

theorem ratio_equivalence (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h : y / (x - z) = (x + 2 * y) / z ∧ (x + 2 * y) / z = x / (y + z)) :
  x / (y + z) = (2 * y - z) / (y + z) :=
by
  sorry

end ratio_equivalence_l137_137135


namespace max_a_value_l137_137688

theorem max_a_value (a : ℝ)
  (H : ∀ x : ℝ, (x - 1) * x - (a - 2) * (a + 1) ≥ 1) :
  a ≤ 3 / 2 := by
  sorry

end max_a_value_l137_137688


namespace problem_I_problem_II_l137_137260

variable (x a m : ℝ)

theorem problem_I (h: ¬ (∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0)) : 
  a < -2 ∨ a > 3 := by
  sorry

theorem problem_II (p : ∃ r : ℝ, r^2 - 2*a*r + 2*a^2 - a - 6 = 0) (q : m-1 ≤ a ∧ a ≤ m+3) :
  ∀ a : ℝ, -2 ≤ a ∧ a ≤ 3 → m ∈ [-1, 0] := by
  sorry

end problem_I_problem_II_l137_137260


namespace cos_of_pi_over_3_minus_alpha_l137_137651

theorem cos_of_pi_over_3_minus_alpha (α : Real) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) :
  Real.cos (Real.pi / 3 - α) = 2 / 3 :=
by
  sorry

end cos_of_pi_over_3_minus_alpha_l137_137651


namespace find_k_l137_137542

-- Definitions based on the problem conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Property of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u.1 = c * v.1 ∧ u.2 = c * v.2

-- Theorem statement equivalent to the problem
theorem find_k (k : ℝ) (h : parallel vector_a (vector_b k)) : k = -2 :=
sorry

end find_k_l137_137542


namespace no_x4_term_implies_a_zero_l137_137831

theorem no_x4_term_implies_a_zero (a : ℝ) :
  ¬ (∃ (x : ℝ), -5 * x^3 * (x^2 + a * x + 5) = -5 * x^5 - 5 * a * x^4 - 25 * x^3 + 5 * a * x^4) →
  a = 0 :=
by
  -- Step through the proof process to derive this conclusion
  sorry

end no_x4_term_implies_a_zero_l137_137831


namespace vanya_faster_speed_l137_137758

def vanya_speed (v : ℝ) : Prop :=
  (v + 2) / v = 2.5

theorem vanya_faster_speed (v : ℝ) (h : vanya_speed v) : (v + 4) / v = 4 :=
by
  sorry

end vanya_faster_speed_l137_137758


namespace fruit_order_count_l137_137507

-- Define the initial conditions
def apples := 3
def oranges := 2
def bananas := 2
def totalFruits := apples + oranges + bananas -- which is 7

-- Calculate the factorial of a number
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Noncomputable definition to skip proof
noncomputable def distinctOrders : ℕ :=
  fact totalFruits / (fact apples * fact oranges * fact bananas)

-- Lean statement expressing that the number of distinct orders is 210
theorem fruit_order_count : distinctOrders = 210 :=
by
  sorry

end fruit_order_count_l137_137507


namespace work_time_B_l137_137907

theorem work_time_B (A_efficiency : ℕ) (B_efficiency : ℕ) (days_together : ℕ) (total_work : ℕ) :
  (A_efficiency = 2 * B_efficiency) →
  (days_together = 5) →
  (total_work = (A_efficiency + B_efficiency) * days_together) →
  (total_work / B_efficiency = 15) :=
by
  intros
  sorry

end work_time_B_l137_137907


namespace pizza_topping_combinations_l137_137342

theorem pizza_topping_combinations (T : Finset ℕ) (hT : T.card = 8) : 
  (T.card.choose 1 + T.card.choose 2 + T.card.choose 3 = 92) :=
by
  sorry

end pizza_topping_combinations_l137_137342


namespace gifts_receiving_ribbon_l137_137560

def total_ribbon := 18
def ribbon_per_gift := 2
def remaining_ribbon := 6

theorem gifts_receiving_ribbon : (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 := by
  sorry

end gifts_receiving_ribbon_l137_137560


namespace problem1_problem2_l137_137242

theorem problem1 : -3 + (-2) * 5 - (-3) = -10 :=
by
  sorry

theorem problem2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 :=
by
  sorry

end problem1_problem2_l137_137242


namespace problem_statement_l137_137411

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x + 2) * (x - 1) > 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 0}
def C_U (B : Set ℝ) : Set ℝ := {x | x ∉ B}

theorem problem_statement : A ∪ C_U B = {x | x < -2 ∨ x ≥ 0} :=
by
  sorry

end problem_statement_l137_137411


namespace triangle_area_l137_137121

noncomputable def area_of_triangle := 
  let a := 4
  let b := 5
  let c := 6
  let cosA := 3 / 4
  let sinA := Real.sqrt (1 - cosA ^ 2)
  (1 / 2) * b * c * sinA

theorem triangle_area :
  ∃ (a b c : ℝ), a = 4 ∧ b = 5 ∧ c = 6 ∧ 
  a < b ∧ b < c ∧ 
  -- Additional conditions
  (∃ A B C : ℝ, C = 2 * A ∧ 
   Real.cos A = 3 / 4 ∧ 
   Real.sin A * Real.cos A = sinA * cosA ∧ 
   0 < A ∧ A < Real.pi ∧ 
   (1 / 2) * b * c * sinA = (15 * Real.sqrt 7) / 4) :=
by
  sorry

end triangle_area_l137_137121


namespace people_left_line_l137_137085

theorem people_left_line (L : ℕ) (h_initial : 31 - L + 25 = 31) : L = 25 :=
by
  -- proof will go here
  sorry

end people_left_line_l137_137085


namespace strongest_erosive_power_l137_137636

-- Definition of the options
inductive Period where
  | MayToJune : Period
  | JuneToJuly : Period
  | JulyToAugust : Period
  | AugustToSeptember : Period

-- Definition of the eroding power function (stub)
def erosivePower : Period → ℕ
| Period.MayToJune => 1
| Period.JuneToJuly => 2
| Period.JulyToAugust => 3
| Period.AugustToSeptember => 1

-- Statement that July to August has the maximum erosive power
theorem strongest_erosive_power : erosivePower Period.JulyToAugust = 3 := 
by 
  sorry

end strongest_erosive_power_l137_137636


namespace bounded_expression_l137_137404

theorem bounded_expression (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 :=
sorry

end bounded_expression_l137_137404


namespace first_train_cross_time_is_10_seconds_l137_137480

-- Definitions based on conditions
def length_of_train := 120 -- meters
def time_second_train_cross_telegraph_post := 15 -- seconds
def distance_cross_each_other := 240 -- meters
def time_cross_each_other := 12 -- seconds

-- The speed of the second train
def speed_second_train := length_of_train / time_second_train_cross_telegraph_post -- m/s

-- The relative speed of both trains when crossing each other
def relative_speed := distance_cross_each_other / time_cross_each_other -- m/s

-- The speed of the first train
def speed_first_train := relative_speed - speed_second_train -- m/s

-- The time taken by the first train to cross the telegraph post
def time_first_train_cross_telegraph_post := length_of_train / speed_first_train -- seconds

-- Proof statement
theorem first_train_cross_time_is_10_seconds :
  time_first_train_cross_telegraph_post = 10 := by
  sorry

end first_train_cross_time_is_10_seconds_l137_137480


namespace product_of_five_consecutive_integers_not_square_l137_137170

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (ha : 0 < a) : ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) := sorry

end product_of_five_consecutive_integers_not_square_l137_137170


namespace find_M_N_l137_137910

-- Define positive integers less than 10
def is_pos_int_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Main theorem to prove M = 5 and N = 6 given the conditions
theorem find_M_N (M N : ℕ) (hM : is_pos_int_lt_10 M) (hN : is_pos_int_lt_10 N) 
  (h : 8 * (10 ^ 7) * M + 420852 * 9 = N * (10 ^ 7) * 9889788 * 11) : 
  M = 5 ∧ N = 6 :=
by {
  sorry
}

end find_M_N_l137_137910


namespace distinct_diagonals_nonagon_l137_137981

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l137_137981


namespace wendy_total_glasses_l137_137202

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l137_137202


namespace nonagon_diagonals_l137_137987

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l137_137987


namespace intersection_eq_1_2_l137_137293

-- Define the set M
def M : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | 1 < x}

-- The intersection of M and N
def intersection : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 2 }

-- Our goal is to prove that M ∩ N = (1, 2]
theorem intersection_eq_1_2 : (M ∩ N) = (Set.Ioo 1 2) :=
by
  sorry

end intersection_eq_1_2_l137_137293


namespace perimeter_difference_l137_137558

theorem perimeter_difference (x : ℝ) :
  let small_square_perimeter := 4 * x
  let large_square_perimeter := 4 * (x + 8)
  large_square_perimeter - small_square_perimeter = 32 :=
by
  sorry

end perimeter_difference_l137_137558


namespace prove_2x_plus_y_le_sqrt_11_l137_137402

variable (x y : ℝ)
variable (h : 3 * x^2 + 2 * y^2 ≤ 6)

theorem prove_2x_plus_y_le_sqrt_11 : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end prove_2x_plus_y_le_sqrt_11_l137_137402


namespace sqrt_40_simplified_l137_137863

theorem sqrt_40_simplified : Real.sqrt 40 = 2 * Real.sqrt 10 := 
by
  sorry

end sqrt_40_simplified_l137_137863


namespace min_omega_l137_137124

theorem min_omega (f : Real → Real) (ω φ : Real) (φ_bound : |φ| < π / 2) 
  (h1 : ω > 0) (h2 : f = fun x => Real.sin (ω * x + φ)) 
  (h3 : f 0 = 1/2) 
  (h4 : ∀ x, f x ≤ f (π / 12)) : ω = 4 := 
by
  sorry

end min_omega_l137_137124


namespace quadratic_has_two_real_distinct_roots_and_find_m_l137_137538

theorem quadratic_has_two_real_distinct_roots_and_find_m 
  (m : ℝ) :
  (x : ℝ) → 
  (h1 : x^2 - (2 * m - 2) * x + (m^2 - 2 * m) = 0) →
  (x1 x2 : ℝ) →
  (h2 : x1^2 + x2^2 = 10) →
  (x1 + x2 = 2 * m - 2) →
  (x1 * x2 = m^2 - 2 * m) →
  (x1 ≠ x2) ∧ (m = -1 ∨ m = 3) :=
by sorry

end quadratic_has_two_real_distinct_roots_and_find_m_l137_137538


namespace hyperbola_asymptote_l137_137127

def hyperbola_eqn (m x y : ℝ) := m * x^2 - y^2 = 1

def vertex_distance_condition (m : ℝ) := 2 * Real.sqrt (1 / m) = 4

theorem hyperbola_asymptote (m : ℝ) (h_eq : hyperbola_eqn m x y) (h_dist : vertex_distance_condition m) :
  ∃ k, y = k * x ∧ k = 1 / 2 ∨ k = -1 / 2 := by
  sorry

end hyperbola_asymptote_l137_137127


namespace actual_distance_l137_137457

theorem actual_distance (d_map : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) (H1 : d_map = 20)
    (H2 : scale_inches = 0.5) (H3 : scale_miles = 10) : 
    d_map * (scale_miles / scale_inches) = 400 := 
by
  sorry

end actual_distance_l137_137457


namespace smallest_unique_digit_sum_32_l137_137385

theorem smallest_unique_digit_sum_32 : ∃ n, 
  (∀ d₁ d₂ ∈ digits n, d₁ ≠ d₂) ∧ 
  (digits n).sum = 32 ∧ 
  ∀ m, 
    (∀ d₁ d₂ ∈ digits m, d₁ ≠ d₂) ∧ 
    (digits m).sum = 32 → 
      m ≥ n := 
begin
  sorry
end

end smallest_unique_digit_sum_32_l137_137385


namespace unique_solution_p_zero_l137_137519

theorem unique_solution_p_zero :
  ∃! (x y p : ℝ), 
    (x^2 - y^2 = 0) ∧ 
    (x * y + p * x - p * y = p^2) ↔ 
    p = 0 :=
by sorry

end unique_solution_p_zero_l137_137519


namespace product_of_axes_l137_137006

-- Definitions based on conditions
def ellipse (a b : ℝ) : Prop :=
  a^2 - b^2 = 64

def triangle_incircle_diameter (a b : ℝ) : Prop :=
  b + 8 - a = 4

-- Proving that (AB)(CD) = 240
theorem product_of_axes (a b : ℝ) (h₁ : ellipse a b) (h₂ : triangle_incircle_diameter a b) : 
  (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_of_axes_l137_137006


namespace sqrt_expression_identity_l137_137241

theorem sqrt_expression_identity :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := 
by
  sorry

end sqrt_expression_identity_l137_137241


namespace no_non_trivial_power_ending_222_l137_137092

theorem no_non_trivial_power_ending_222 (x y : ℕ) (hx : x > 1) (hy : y > 1) : ¬ (∃ n : ℕ, n % 1000 = 222 ∧ n = x^y) :=
by
  sorry

end no_non_trivial_power_ending_222_l137_137092


namespace jungkook_has_smallest_collection_l137_137040

-- Define the collections
def yoongi_collection : ℕ := 7
def jungkook_collection : ℕ := 6
def yuna_collection : ℕ := 9

-- State the theorem
theorem jungkook_has_smallest_collection : 
  jungkook_collection = min yoongi_collection (min jungkook_collection yuna_collection) := 
by
  sorry

end jungkook_has_smallest_collection_l137_137040


namespace sum_of_sixth_powers_l137_137564

theorem sum_of_sixth_powers (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 0) 
  (h2 : α₁^2 + α₂^2 + α₃^2 = 2) 
  (h3 : α₁^3 + α₂^3 + α₃^3 = 4) : 
  α₁^6 + α₂^6 + α₃^6 = 7 :=
sorry

end sum_of_sixth_powers_l137_137564


namespace leila_spending_l137_137697

theorem leila_spending (sweater jewelry total money_left : ℕ) (h1 : sweater = 40) (h2 : sweater * 4 = total) (h3 : money_left = 20) (h4 : total - sweater - jewelry = money_left) : jewelry - sweater = 60 :=
by
  sorry

end leila_spending_l137_137697


namespace arithmetic_sequence_S22_zero_l137_137396

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_S22_zero (a d : ℝ) (S : ℕ → ℝ) (h_arith_seq : ∀ n, S n = sum_of_first_n_terms a d n)
  (h1 : a > 0) (h2 : S 5 = S 17) :
  S 22 = 0 :=
by
  sorry

end arithmetic_sequence_S22_zero_l137_137396


namespace focus_of_parabola_proof_l137_137181

noncomputable def focus_of_parabola (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

theorem focus_of_parabola_proof (a : ℝ) (h : a ≠ 0) :
  focus_of_parabola a h = (1 / (4 * a), 0) :=
sorry

end focus_of_parabola_proof_l137_137181


namespace convert_to_base_k_l137_137254

noncomputable def base_k_eq (k : ℕ) : Prop :=
  4 * k + 4 = 36

theorem convert_to_base_k :
  ∃ k : ℕ, base_k_eq k ∧ (67 / k^2 % k^2 % k = 1 ∧ 67 / k % k = 0 ∧ 67 % k = 3) :=
sorry

end convert_to_base_k_l137_137254


namespace min_possible_range_l137_137913

theorem min_possible_range (A B C : ℤ) : 
  (A + 15 ≤ C ∧ B + 25 ≤ C ∧ C ≤ A + 45) → C - A ≤ 45 :=
by
  intros h
  have h1 : A + 15 ≤ C := h.1
  have h2 : B + 25 ≤ C := h.2.1
  have h3 : C ≤ A + 45 := h.2.2
  sorry

end min_possible_range_l137_137913


namespace necessary_but_not_sufficient_l137_137714

def p (x : ℝ) : Prop := x ^ 2 = 3 * x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)

theorem necessary_but_not_sufficient (x : ℝ) : (p x → q x) ∧ ¬ (q x → p x) := by
  sorry

end necessary_but_not_sufficient_l137_137714


namespace power_function_solution_l137_137527

theorem power_function_solution (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = x ^ α) (h2 : f 4 = 2) : f 3 = Real.sqrt 3 :=
sorry

end power_function_solution_l137_137527


namespace senior_discount_percentage_l137_137845

theorem senior_discount_percentage 
    (cost_shorts : ℕ)
    (count_shorts : ℕ)
    (cost_shirts : ℕ)
    (count_shirts : ℕ)
    (amount_paid : ℕ)
    (total_cost : ℕ := (cost_shorts * count_shorts) + (cost_shirts * count_shirts))
    (discount_received : ℕ := total_cost - amount_paid)
    (discount_percentage : ℚ := (discount_received : ℚ) / total_cost * 100) :
    count_shorts = 3 ∧ cost_shorts = 15 ∧ count_shirts = 5 ∧ cost_shirts = 17 ∧ amount_paid = 117 →
    discount_percentage = 10 := 
by
    sorry

end senior_discount_percentage_l137_137845


namespace comic_books_collection_l137_137561

theorem comic_books_collection (initial_ky: ℕ) (rate_ky: ℕ) (initial_la: ℕ) (rate_la: ℕ) (months: ℕ) :
  initial_ky = 50 → rate_ky = 1 → initial_la = 20 → rate_la = 7 → months = 33 →
  initial_la + rate_la * months = 3 * (initial_ky + rate_ky * months) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end comic_books_collection_l137_137561


namespace train_speed_excluding_stoppages_l137_137947

theorem train_speed_excluding_stoppages 
    (speed_including_stoppages : ℕ)
    (stoppage_time_per_hour : ℕ)
    (running_time_per_hour : ℚ)
    (h1 : speed_including_stoppages = 36)
    (h2 : stoppage_time_per_hour = 20)
    (h3 : running_time_per_hour = 2 / 3) :
    ∃ S : ℕ, S = 54 :=
by 
  sorry

end train_speed_excluding_stoppages_l137_137947


namespace new_interest_rate_l137_137458

theorem new_interest_rate 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (initial_rate : ℝ) 
  (time : ℝ) 
  (new_total_interest : ℝ)
  (principal : ℝ)
  (new_rate : ℝ) 
  (h1 : initial_interest = principal * initial_rate * time)
  (h2 : new_total_interest = initial_interest + additional_interest)
  (h3 : new_total_interest = principal * new_rate * time)
  (principal_val : principal = initial_interest / initial_rate) :
  new_rate = 0.05 :=
by
  sorry

end new_interest_rate_l137_137458


namespace sum_of_final_two_numbers_l137_137470

noncomputable def final_sum (X m n : ℚ) : ℚ :=
  3 * m + 3 * n - 14

theorem sum_of_final_two_numbers (X m n : ℚ) 
  (h1 : m + n = X) :
  final_sum X m n = 3 * X - 14 :=
  sorry

end sum_of_final_two_numbers_l137_137470


namespace find_weekday_rate_l137_137176

-- Definitions of given conditions
def num_people : ℕ := 6
def days_weekdays : ℕ := 2
def days_weekend : ℕ := 2
def weekend_rate : ℕ := 540
def payment_per_person : ℕ := 320

-- Theorem to prove the weekday rental rate
theorem find_weekday_rate (W : ℕ) :
  (num_people * payment_per_person) = (days_weekdays * W) + (days_weekend * weekend_rate) →
  W = 420 :=
by 
  intros h
  sorry

end find_weekday_rate_l137_137176


namespace solve_for_x_l137_137437

theorem solve_for_x : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 :=
by
  sorry

end solve_for_x_l137_137437


namespace nonagon_diagonals_count_l137_137993

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l137_137993


namespace find_constant_l137_137354

-- Definitions based on the conditions provided
variable (f : ℕ → ℕ)
variable (c : ℕ)

-- Given conditions
def f_1_eq_0 : f 1 = 0 := sorry
def functional_equation (m n : ℕ) : f (m + n) = f m + f n + c * (m * n - 1) := sorry
def f_17_eq_4832 : f 17 = 4832 := sorry

-- The mathematically equivalent proof problem
theorem find_constant : c = 4 := 
sorry

end find_constant_l137_137354


namespace distinct_diagonals_convex_nonagon_l137_137978

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l137_137978


namespace sum_of_three_numbers_l137_137768

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_of_three_numbers_l137_137768


namespace fraction_of_males_l137_137628

theorem fraction_of_males (M F : ℝ) 
  (h1 : M + F = 1)
  (h2 : (7 / 8) * M + (4 / 5) * F = 0.845) :
  M = 0.6 :=
by
  sorry

end fraction_of_males_l137_137628


namespace meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l137_137769

theorem meters_to_kilometers (h : 1 = 1000) : 6000 / 1000 = 6 := by
  sorry

theorem kilograms_to_grams (h : 1 = 1000) : (5 + 2) * 1000 = 7000 := by
  sorry

theorem centimeters_to_decimeters (h : 10 = 1) : (58 + 32) / 10 = 9 := by
  sorry

theorem hours_to_minutes (h : 60 = 1) : 3 * 60 + 30 = 210 := by
  sorry

end meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l137_137769


namespace not_prime_3999991_l137_137912

   theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
   by
     -- Provide the factorization proof
     sorry
   
end not_prime_3999991_l137_137912


namespace taylor_correct_answers_percentage_l137_137937

theorem taylor_correct_answers_percentage 
  (N : ℕ := 30)
  (alex_correct_alone_percentage : ℝ := 0.85)
  (alex_overall_percentage : ℝ := 0.83)
  (taylor_correct_alone_percentage : ℝ := 0.95)
  (alex_correct_alone : ℕ := 13)
  (alex_correct_total : ℕ := 25)
  (together_correct : ℕ := 12)
  (taylor_correct_alone : ℕ := 14)
  (taylor_correct_total : ℕ := 26) :
  ((taylor_correct_total : ℝ) / (N : ℝ)) * 100 = 87 :=
by
  sorry

end taylor_correct_answers_percentage_l137_137937


namespace guilty_D_l137_137730

def isGuilty (A B C D : Prop) : Prop :=
  ¬A ∧ (B → ∃! x, x ≠ A ∧ (x = C ∨ x = D)) ∧ (C → ∃! x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ A ∧ x₂ ≠ A ∧ ((x₁ = B ∨ x₁ = D) ∧ (x₂ = B ∨ x₂ = D))) ∧ (¬A ∨ B ∨ C ∨ D)

theorem guilty_D (A B C D : Prop) (h : isGuilty A B C D) : D :=
by
  sorry

end guilty_D_l137_137730


namespace number_of_triangles_l137_137592

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l137_137592


namespace combined_avg_of_remaining_two_subjects_l137_137890

noncomputable def avg (scores : List ℝ) : ℝ :=
  scores.foldl (· + ·) 0 / scores.length

theorem combined_avg_of_remaining_two_subjects 
  (S1_avg S2_part_avg all_avg : ℝ)
  (S1_count S2_part_count S2_total_count : ℕ)
  (h1 : S1_avg = 85) 
  (h2 : S2_part_avg = 78) 
  (h3 : all_avg = 80) 
  (h4 : S1_count = 3)
  (h5 : S2_part_count = 5)
  (h6 : S2_total_count = 7) :
  avg [all_avg * (S1_count + S2_total_count) 
       - S1_count * S1_avg 
       - S2_part_count * S2_part_avg] / (S2_total_count - S2_part_count)
  = 77.5 := by
  sorry

end combined_avg_of_remaining_two_subjects_l137_137890


namespace mass_percentages_correct_l137_137800

noncomputable def mass_percentage_of_Ba (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * 137.327 + (y / 153.326) * 137.327) / (x + y) ) * 100

noncomputable def mass_percentage_of_F (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * (2 * 18.998)) / (x + y) ) * 100

noncomputable def mass_percentage_of_O (x y : ℝ) : ℝ :=
  ( ((y / 153.326) * 15.999) / (x + y) ) * 100

theorem mass_percentages_correct (x y : ℝ) :
  ∃ (Ba F O : ℝ), 
    Ba = mass_percentage_of_Ba x y ∧
    F = mass_percentage_of_F x y ∧
    O = mass_percentage_of_O x y :=
sorry

end mass_percentages_correct_l137_137800


namespace color_theorem_l137_137097

/-- The only integers \( k \geq 1 \) such that if each integer is colored in one of these \( k \)
colors, there must exist integers \( a_1 < a_2 < \cdots < a_{2023} \) of the same color where the
differences \( a_2 - a_1, a_3 - a_2, \cdots, a_{2023} - a_{2022} \) are all powers of 2 are
\( k = 1 \) and \( k = 2 \). -/
theorem color_theorem : ∀ (k : ℕ), (k ≥ 1) →
  (∀ f : ℕ → Fin k,
    ∃ a : Fin 2023 → ℕ,
    (∀ i : Fin (2023 - 1), ∃ n : ℕ, 2^n = (a i.succ - a i)) ∧
    (∀ i j : Fin 2023, i < j → f (a i) = f (a j)))
  ↔ k = 1 ∨ k = 2 := by
  sorry

end color_theorem_l137_137097


namespace pool_filled_in_48_minutes_with_both_valves_open_l137_137876

def rate_first_valve_fills_pool_in_2_hours (V1 : ℚ) : Prop :=
  V1 * 120 = 12000

def rate_second_valve_50_more_than_first (V1 V2 : ℚ) : Prop :=
  V2 = V1 + 50

def pool_capacity : ℚ := 12000

def combined_rate (V1 V2 combinedRate : ℚ) : Prop :=
  combinedRate = V1 + V2

def time_to_fill_pool_with_both_valves_open (combinedRate time : ℚ) : Prop :=
  time = pool_capacity / combinedRate

theorem pool_filled_in_48_minutes_with_both_valves_open
  (V1 V2 combinedRate time : ℚ) :
  rate_first_valve_fills_pool_in_2_hours V1 →
  rate_second_valve_50_more_than_first V1 V2 →
  combined_rate V1 V2 combinedRate →
  time_to_fill_pool_with_both_valves_open combinedRate time →
  time = 48 :=
by
  intros
  sorry

end pool_filled_in_48_minutes_with_both_valves_open_l137_137876


namespace plane_through_points_eq_l137_137252

-- Define the points M, N, P
def M := (1, 2, 0)
def N := (1, -1, 2)
def P := (0, 1, -1)

-- Define the target plane equation
def target_plane_eq (x y z : ℝ) := 5 * x - 2 * y + 3 * z - 1 = 0

-- Main theorem statement
theorem plane_through_points_eq :
  ∀ (x y z : ℝ),
    (∃ A B C : ℝ,
      A * (x - 1) + B * (y - 2) + C * z = 0 ∧
      A * (1 - 1) + B * (-1 - 2) + C * (2 - 0) = 0 ∧
      A * (0 - 1) + B * (1 - 2) + C * (-1 - 0) = 0) →
    target_plane_eq x y z :=
by
  sorry

end plane_through_points_eq_l137_137252


namespace sum_of_solutions_eq_seven_l137_137952

theorem sum_of_solutions_eq_seven : 
  ∃ x : ℝ, x + 49/x = 14 ∧ (∀ y : ℝ, y + 49 / y = 14 → y = x) → x = 7 :=
by {
  sorry
}

end sum_of_solutions_eq_seven_l137_137952


namespace repeating_decimal_exceeds_decimal_l137_137632

noncomputable def repeating_decimal_to_fraction : ℚ := 9 / 11
noncomputable def decimal_to_fraction : ℚ := 3 / 4

theorem repeating_decimal_exceeds_decimal :
  repeating_decimal_to_fraction - decimal_to_fraction = 3 / 44 :=
by
  sorry

end repeating_decimal_exceeds_decimal_l137_137632


namespace mono_intervals_range_of_a_l137_137363

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.exp (x - 1)

theorem mono_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, f x a > 0) ∧ 
  (a > 0 → (∀ x, x < 1 - Real.log a → f x a > 0) ∧ (∀ x, x > 1 - Real.log a → f x a < 0)) :=
sorry

theorem range_of_a (h : ∀ x, f x a ≤ 0) : a ≥ 1 :=
sorry

end mono_intervals_range_of_a_l137_137363


namespace rate_in_still_water_l137_137338

theorem rate_in_still_water (with_stream_speed against_stream_speed : ℕ) 
  (h₁ : with_stream_speed = 16) 
  (h₂ : against_stream_speed = 12) : 
  (with_stream_speed + against_stream_speed) / 2 = 14 := 
by
  sorry

end rate_in_still_water_l137_137338


namespace car_travel_time_l137_137000

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l137_137000


namespace P_2_lt_X_lt_4_l137_137119

-- Given definitions
variable (X : ℝ)
variable (σ : ℝ) (hσ : σ > 0)
def normal_dist := MeasureTheory.Normal 2 σ^2
def X_is_normal : MeasureTheory.AEStronglyMeasurable (MeasureTheory.RealMeasLe normal_dist) := sorry
def P_X_lt_0 : ℝ := MeasureTheory.Probability (MeasureTheory.RealMeasLe normal_dist {x | x < 0})
#eval assert (P_X_lt_0 = 0.1)

-- Prove statement
theorem P_2_lt_X_lt_4 : MeasureTheory.Probability (MeasureTheory.RealMeasLe normal_dist {x | 2 < x ∧ x < 4}) = 0.4 :=
by sorry

end P_2_lt_X_lt_4_l137_137119


namespace second_player_wins_l137_137476

def num_of_piles_initial := 3
def total_stones := 10 + 15 + 20
def num_of_piles_final := total_stones
def total_moves := num_of_piles_final - num_of_piles_initial

theorem second_player_wins : total_moves % 2 = 0 :=
sorry

end second_player_wins_l137_137476


namespace sqrt_of_square_neg_l137_137525

variable {a : ℝ}

theorem sqrt_of_square_neg (h : a < 0) : Real.sqrt (a^2) = -a := 
sorry

end sqrt_of_square_neg_l137_137525


namespace nonagon_diagonals_count_l137_137996

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l137_137996


namespace smallest_value_of_n_l137_137349

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l137_137349


namespace inequality_of_function_l137_137169

theorem inequality_of_function (x : ℝ) : 
  (1 / 2 : ℝ) ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 / 2 : ℝ) :=
sorry

end inequality_of_function_l137_137169


namespace ticket_sales_l137_137078

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l137_137078


namespace find_f7_l137_137659

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 7

theorem find_f7 (a b : ℝ) (h : f (-7) a b = -17) : f (7) a b = 31 := 
by
  sorry

end find_f7_l137_137659


namespace longest_chord_of_circle_l137_137530

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end longest_chord_of_circle_l137_137530


namespace count_president_vp_secretary_l137_137003

theorem count_president_vp_secretary (total_members boys girls : ℕ) (total_members_eq : total_members = 30) 
(boys_eq : boys = 18) (girls_eq : girls = 12) :
  ∃ (ways : ℕ), 
  ways = (boys * girls * (boys - 1) + girls * boys * (girls - 1)) ∧
  ways = 6048 :=
by
  sorry

end count_president_vp_secretary_l137_137003


namespace find_percentage_of_other_investment_l137_137014

theorem find_percentage_of_other_investment
  (total_investment : ℝ) (specific_investment : ℝ) (specific_rate : ℝ) (total_interest : ℝ) 
  (other_investment : ℝ) (other_interest : ℝ) (P : ℝ) :
  total_investment = 17000 ∧
  specific_investment = 12000 ∧
  specific_rate = 0.04 ∧
  total_interest = 1380 ∧
  other_investment = total_investment - specific_investment ∧
  other_interest = total_interest - specific_rate * specific_investment ∧ 
  other_interest = (P / 100) * other_investment
  → P = 18 :=
by
  intros
  sorry

end find_percentage_of_other_investment_l137_137014


namespace angle_bisector_inequality_l137_137165

theorem angle_bisector_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_perimeter : (x + y + z) = 6) :
  (1 / x^2) + (1 / y^2) + (1 / z^2) ≥ 1 := by
  sorry

end angle_bisector_inequality_l137_137165


namespace delta_value_l137_137276

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l137_137276


namespace find_m_n_l137_137671

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l137_137671


namespace roots_cubic_reciprocal_l137_137418

theorem roots_cubic_reciprocal (a b c r s : ℝ) (h_eq : a ≠ 0) (h_r : a * r^2 + b * r + c = 0) (h_s : a * s^2 + b * s + c = 0) :
  1 / r^3 + 1 / s^3 = (-b^3 + 3 * a * b * c) / c^3 := 
by
  sorry

end roots_cubic_reciprocal_l137_137418


namespace fraction_pizza_covered_by_pepperoni_l137_137577

/--
Given that six pepperoni circles fit exactly across the diameter of a 12-inch pizza
and a total of 24 circles of pepperoni are placed on the pizza without overlap,
prove that the fraction of the pizza covered by pepperoni is 2/3.
-/
theorem fraction_pizza_covered_by_pepperoni : 
  (∃ d r : ℝ, 6 * r = d ∧ d = 12 ∧ (r * r * π * 24) / (6 * 6 * π) = 2 / 3) := 
sorry

end fraction_pizza_covered_by_pepperoni_l137_137577


namespace problem_statement_l137_137475

variables {x y x1 y1 a b c d : ℝ}

-- The main theorem statement
theorem problem_statement (h0 : ∀ (x y : ℝ), 6 * y ^ 2 = 2 * x ^ 3 + 3 * x ^ 2 + x) 
                           (h1 : x1 = a * x + b) 
                           (h2 : y1 = c * y + d) 
                           (h3 : y1 ^ 2 = x1 ^ 3 - 36 * x1) : 
                           a + b + c + d = 90 := sorry

end problem_statement_l137_137475


namespace find_m_and_f_max_l137_137678

noncomputable def f (x m : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 + m

theorem find_m_and_f_max (m a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∃ y, f y m = 3) →
  (∀ x ∈ Set.Icc a (a + Real.pi), ∃ y, f y m = 6) →
  m = 3 ∧ ∀ x ∈ Set.Icc a (a + Real.pi), f x 3 ≤ 6 :=
sorry

end find_m_and_f_max_l137_137678


namespace cookies_last_days_l137_137692

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days_l137_137692


namespace find_abc_sum_l137_137447

noncomputable def x : ℝ := Real.sqrt ( Real.sqrt 77 / 3 + 5 / 3)

lemma x_squared : x^2 = Real.sqrt 77 / 3 + 5 / 3 :=
begin
  have h1 : x = Real.sqrt ( Real.sqrt 77 / 3 + 5 / 3),
  from rfl,
  rw h1,
  exact Real.sqr_sqrt (Real.sqrt_nonneg 77 / 3 + 5 / 3),
end

lemma x_relations : ∃ (a b c : ℕ), (x^60 = 3*x^57 + 12*x^55 + 9*x^53 - x^30 + a*x^26 + b*x^24 + c*x^20) :=
begin
  have h_sq : 3 * x^2 = Real.sqrt 77 + 5,
  {rw [← x_squared, mul_assoc, mul_div_cancel_left _ (by norm_num : (3:ℝ) ≠ 0)],},
  have h_x4 : x^4 = (10 / 3) * x^2 + (52 / 9),
  {calc x^4 = (x^2)^2 : by ring
            ... = ((Real.sqrt 77 / 3 + 5 / 3))^2 : by rw x_squared
            ... = (77 / 9 + 2 * (5 / 3) * (Real.sqrt 77) / 3 + 25 / 9) : by ring
            ... = (77 / 9 + 10 / 3 * (Real.sqrt 77) / 3 + 25 / 9) : by ring
            ... = (77 / 9 + 10 / 9 + 25 / 9) : by ring
            ... = (102 / 9) : by ring,},
  -- Further manipulation using the provided expressions
  obtain ⟨a, b, c, h⟩ : ∃ a b c: ℕ, x^60 = 3 * x^57 + 12 * x^55 + 9 * x^53 - x^30 + a * x^26 + b * x^24 + c * x^20,
  sorry,

  use [a, b, c],
end

-- We finally state the theorem to find the sum a + b + c
theorem find_abc_sum : ∃ (a b c : ℕ), 
  (x^60 = 3*x^57 + 12*x^55 + 9*x^53 - x^30 + a*x^26 + b*x^24 + c*x^20) ∧ 
  a + b + c = some_value :=  -- Replace some_value with actual sum in final proof.
begin 
  obtain ⟨a, b, c, h⟩ := x_relations,
  use [a, b, c],
  split, assumption,
  sorry,
end

end find_abc_sum_l137_137447


namespace total_percent_sample_candy_l137_137138

theorem total_percent_sample_candy (total_customers : ℕ) (percent_caught : ℝ) (percent_not_caught : ℝ)
  (h1 : percent_caught = 0.22)
  (h2 : percent_not_caught = 0.20)
  (h3 : total_customers = 100) :
  percent_caught + percent_not_caught = 0.28 :=
by
  sorry

end total_percent_sample_candy_l137_137138


namespace square_area_l137_137071

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l137_137071


namespace not_divisible_by_121_l137_137164

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 3 * n + 5)) :=
by
  sorry

end not_divisible_by_121_l137_137164


namespace class_gpa_l137_137683

theorem class_gpa (n : ℕ) (h_n : n = 60)
  (n1 : ℕ) (h_n1 : n1 = 20) (gpa1 : ℕ) (h_gpa1 : gpa1 = 15)
  (n2 : ℕ) (h_n2 : n2 = 15) (gpa2 : ℕ) (h_gpa2 : gpa2 = 17)
  (n3 : ℕ) (h_n3 : n3 = 25) (gpa3 : ℕ) (h_gpa3 : gpa3 = 19) :
  (20 * 15 + 15 * 17 + 25 * 19 : ℕ) / 60 = 1717 / 100 := 
sorry

end class_gpa_l137_137683


namespace soccer_team_lineups_l137_137708

noncomputable def num_starting_lineups (n k t g : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) k)

theorem soccer_team_lineups :
  num_starting_lineups 18 9 1 1 = 3501120 := by
    sorry

end soccer_team_lineups_l137_137708
