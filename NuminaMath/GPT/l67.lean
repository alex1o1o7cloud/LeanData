import Mathlib

namespace power_subtraction_divisibility_l67_6705

theorem power_subtraction_divisibility (N : ℕ) (h : N > 1) : 
  ∃ k : ℕ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) :=
by
  sorry

end power_subtraction_divisibility_l67_6705


namespace keith_spent_on_tires_l67_6738

noncomputable def money_spent_on_speakers : ℝ := 136.01
noncomputable def money_spent_on_cd_player : ℝ := 139.38
noncomputable def total_expenditure : ℝ := 387.85
noncomputable def total_spent_on_speakers_and_cd_player : ℝ := money_spent_on_speakers + money_spent_on_cd_player
noncomputable def money_spent_on_new_tires : ℝ := total_expenditure - total_spent_on_speakers_and_cd_player

theorem keith_spent_on_tires :
  money_spent_on_new_tires = 112.46 :=
by
  sorry

end keith_spent_on_tires_l67_6738


namespace John_spent_fraction_toy_store_l67_6708

variable (weekly_allowance arcade_money toy_store_money candy_store_money : ℝ)
variable (spend_fraction : ℝ)

-- John's conditions
def John_conditions : Prop :=
  weekly_allowance = 3.45 ∧
  arcade_money = 3 / 5 * weekly_allowance ∧
  candy_store_money = 0.92 ∧
  toy_store_money = weekly_allowance - arcade_money - candy_store_money

-- Theorem to prove the fraction spent at the toy store
theorem John_spent_fraction_toy_store :
  John_conditions weekly_allowance arcade_money toy_store_money candy_store_money →
  spend_fraction = toy_store_money / (weekly_allowance - arcade_money) →
  spend_fraction = 1 / 3 :=
by
  sorry

end John_spent_fraction_toy_store_l67_6708


namespace total_questions_attempted_l67_6760

theorem total_questions_attempted 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (total_questions : ℕ) (incorrect_answers : ℕ)
  (h_marks_per_correct : marks_per_correct = 4)
  (h_marks_lost_per_wrong : marks_lost_per_wrong = 1) 
  (h_total_marks : total_marks = 130) 
  (h_correct_answers : correct_answers = 36) 
  (h_score_eq : marks_per_correct * correct_answers - marks_lost_per_wrong * incorrect_answers = total_marks)
  (h_total_questions : total_questions = correct_answers + incorrect_answers) : 
  total_questions = 50 :=
by
  sorry

end total_questions_attempted_l67_6760


namespace find_a7_in_arithmetic_sequence_l67_6794

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem find_a7_in_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3_a5 : a 3 + a 5 = 10) :
  a 7 = 8 :=
sorry

end find_a7_in_arithmetic_sequence_l67_6794


namespace set_intersection_eq_l67_6779

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

def B : Set ℝ := { x | x < -2 ∨ x > 5 }

def C_U (B : Set ℝ) : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }

theorem set_intersection_eq : A ∩ (C_U B) = { x | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

end set_intersection_eq_l67_6779


namespace three_digit_number_is_382_l67_6769

theorem three_digit_number_is_382 
  (x : ℕ) 
  (h1 : x >= 100 ∧ x < 1000) 
  (h2 : 7000 + x - (10 * x + 7) = 3555) : 
  x = 382 :=
by 
  sorry

end three_digit_number_is_382_l67_6769


namespace johns_new_weekly_earnings_l67_6747

-- Define the original weekly earnings and the percentage increase as given conditions:
def original_weekly_earnings : ℕ := 60
def percentage_increase : ℕ := 50

-- Prove that John's new weekly earnings after the raise is 90 dollars:
theorem johns_new_weekly_earnings : original_weekly_earnings + (percentage_increase * original_weekly_earnings / 100) = 90 := by
sorry

end johns_new_weekly_earnings_l67_6747


namespace common_difference_of_arithmetic_sequence_l67_6712

variable (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (d : ℝ)
variable (h₁ : S_n 5 = -15) (h₂ : a_n 2 + a_n 5 = -2)

theorem common_difference_of_arithmetic_sequence :
  d = 4 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l67_6712


namespace eval_f_g_at_4_l67_6741

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem eval_f_g_at_4 : f (g 4) = (25 / 7) * Real.sqrt 21 := by
  sorry

end eval_f_g_at_4_l67_6741


namespace two_pow_n_plus_one_divisible_by_three_l67_6781

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l67_6781


namespace car_owners_without_motorcycles_l67_6788

theorem car_owners_without_motorcycles
  (total_adults : ℕ)
  (car_owners : ℕ)
  (motorcycle_owners : ℕ)
  (all_owners : total_adults = 400)
  (john_owns_cars : car_owners = 370)
  (john_owns_motorcycles : motorcycle_owners = 50)
  (all_adult_owners : total_adults = car_owners + motorcycle_owners - (car_owners - motorcycle_owners)) : 
  (car_owners - (car_owners + motorcycle_owners - total_adults) = 350) :=
by {
  sorry
}

end car_owners_without_motorcycles_l67_6788


namespace multiples_of_4_l67_6775

theorem multiples_of_4 (n : ℕ) (h : n + 23 * 4 = 112) : n = 20 :=
by
  sorry

end multiples_of_4_l67_6775


namespace sum_of_two_numbers_l67_6773

theorem sum_of_two_numbers (x y : ℤ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l67_6773


namespace function_increasing_on_interval_l67_6771

theorem function_increasing_on_interval :
  ∀ x : ℝ, (1 / 2 < x) → (x > 0) → (8 * x - 1 / (x^2)) > 0 :=
sorry

end function_increasing_on_interval_l67_6771


namespace domain_of_f_l67_6797

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3 * x + 2)

theorem domain_of_f :
  {x : ℝ | (x < 1) ∨ (1 < x ∧ x < 2) ∨ (x > 2)} = 
  {x : ℝ | f x ≠ 0} :=
sorry

end domain_of_f_l67_6797


namespace bottles_per_crate_l67_6715

theorem bottles_per_crate (num_bottles total_bottles bottles_not_placed num_crates : ℕ) 
    (h1 : total_bottles = 130)
    (h2 : bottles_not_placed = 10)
    (h3 : num_crates = 10) 
    (h4 : num_bottles = total_bottles - bottles_not_placed) :
    (num_bottles / num_crates) = 12 := 
by 
    sorry

end bottles_per_crate_l67_6715


namespace combination_10_3_eq_120_l67_6786

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l67_6786


namespace nolan_total_savings_l67_6742

-- Define the conditions given in the problem
def monthly_savings : ℕ := 3000
def number_of_months : ℕ := 12

-- State the equivalent proof problem in Lean 4
theorem nolan_total_savings : (monthly_savings * number_of_months) = 36000 := by
  -- Proof is omitted
  sorry

end nolan_total_savings_l67_6742


namespace pipe_A_fill_time_l67_6719

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end pipe_A_fill_time_l67_6719


namespace distinct_real_roots_iff_l67_6763

noncomputable def operation (a b : ℝ) : ℝ := a * b^2 - b 

theorem distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 x1 = k ∧ operation 1 x2 = k) ↔ k > -1/4 :=
by
  sorry

end distinct_real_roots_iff_l67_6763


namespace soap_box_length_l67_6749

def VolumeOfEachSoapBox (L : ℝ) := 30 * L
def VolumeOfCarton := 25 * 42 * 60
def MaximumSoapBoxes := 300

theorem soap_box_length :
  ∀ L : ℝ,
  MaximumSoapBoxes * VolumeOfEachSoapBox L = VolumeOfCarton → 
  L = 7 :=
by
  intros L h
  sorry

end soap_box_length_l67_6749


namespace triangle_equi_if_sides_eq_sum_of_products_l67_6799

theorem triangle_equi_if_sides_eq_sum_of_products (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ac) : a = b ∧ b = c :=
by sorry

end triangle_equi_if_sides_eq_sum_of_products_l67_6799


namespace factorize_expression_l67_6795

variable {R : Type} [CommRing R]

theorem factorize_expression (x y : R) : 
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := 
by 
  sorry

end factorize_expression_l67_6795


namespace solve_for_wood_length_l67_6772

theorem solve_for_wood_length (y x : ℝ) (h1 : y - x = 4.5) (h2 : x - (1/2) * y = 1) :
  ∃! (x y : ℝ), (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  -- The content of the proof is omitted
  sorry

end solve_for_wood_length_l67_6772


namespace total_sold_l67_6776

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end total_sold_l67_6776


namespace pentagon_area_l67_6759

theorem pentagon_area 
  (edge_length : ℝ) 
  (triangle_height : ℝ) 
  (n_pentagons : ℕ) 
  (equal_convex_pentagons : ℕ) 
  (pentagon_area : ℝ) : 
  edge_length = 5 ∧ triangle_height = 2 ∧ n_pentagons = 5 ∧ equal_convex_pentagons = 5 → pentagon_area = 30 := 
by
  sorry

end pentagon_area_l67_6759


namespace solve_equation_l67_6765

theorem solve_equation : ∃ x : ℤ, 3 * x - 2 * x = 7 ∧ x = 7 :=
by
  sorry

end solve_equation_l67_6765


namespace fraction_of_students_with_buddy_l67_6768

variables (f e : ℕ)
-- Given:
axiom H1 : e / 4 = f / 3

-- Prove:
theorem fraction_of_students_with_buddy : 
  (e / 4 + f / 3) / (e + f) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l67_6768


namespace total_paintable_area_correct_l67_6700

-- Define the conditions
def warehouse_width := 12
def warehouse_length := 15
def warehouse_height := 7

def window_count_per_longer_wall := 3
def window_width := 2
def window_height := 3

-- Define areas for walls, ceiling, and floor
def area_wall_1 := warehouse_width * warehouse_height
def area_wall_2 := warehouse_length * warehouse_height
def window_area := window_width * window_height
def window_total_area := window_count_per_longer_wall * window_area
def area_wall_2_paintable := 2 * (area_wall_2 - window_total_area) -- both inside and outside
def area_ceiling := warehouse_width * warehouse_length
def area_floor := warehouse_width * warehouse_length

-- Total paintable area calculation
def total_paintable_area := 2 * area_wall_1 + area_wall_2_paintable + area_ceiling + area_floor

-- Final proof statement
theorem total_paintable_area_correct : total_paintable_area = 876 := by
  sorry

end total_paintable_area_correct_l67_6700


namespace administrative_staff_drawn_in_stratified_sampling_l67_6761

theorem administrative_staff_drawn_in_stratified_sampling
  (total_staff : ℕ)
  (full_time_teachers : ℕ)
  (administrative_staff : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 320)
  (h_teachers : full_time_teachers = 248)
  (h_admin : administrative_staff = 48)
  (h_logistics : logistics_personnel = 24)
  (h_sample : sample_size = 40)
  : (administrative_staff * (sample_size / total_staff) = 6) :=
by
  -- mathematical proof goes here
  sorry

end administrative_staff_drawn_in_stratified_sampling_l67_6761


namespace cos_270_eq_zero_l67_6748

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l67_6748


namespace k_satisfies_triangle_condition_l67_6752

theorem k_satisfies_triangle_condition (k : ℤ) 
  (hk_pos : 0 < k) (a b c : ℝ) (ha_pos : 0 < a) 
  (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (h_ineq : (k : ℝ) * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : k = 6 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end k_satisfies_triangle_condition_l67_6752


namespace correctness_of_solution_set_l67_6777

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := { x | 3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9 }

-- Define the expected solution set derived from the problem
def expected_solution_set : Set ℝ := { x | -1 < x ∧ x ≤ 1 } ∪ { x | 2.5 < x ∧ x < 4.5 }

-- The proof statement
theorem correctness_of_solution_set : solution_set = expected_solution_set :=
  sorry

end correctness_of_solution_set_l67_6777


namespace steve_reading_pages_l67_6740

theorem steve_reading_pages (total_pages: ℕ) (weeks: ℕ) (reading_days_per_week: ℕ) 
  (reads_on_monday: ℕ) (reads_on_wednesday: ℕ) (reads_on_friday: ℕ) :
  total_pages = 2100 → weeks = 7 → reading_days_per_week = 3 → 
  (reads_on_monday = reads_on_wednesday ∧ reads_on_wednesday = reads_on_friday) → 
  ((weeks * reading_days_per_week) > 0) → 
  (total_pages / (weeks * reading_days_per_week)) = reads_on_monday :=
by
  intro h_total_pages h_weeks h_reading_days_per_week h_reads_on_days h_nonzero
  sorry

end steve_reading_pages_l67_6740


namespace black_cards_remaining_proof_l67_6736

def initial_black_cards := 26
def black_cards_taken_out := 4
def black_cards_remaining := initial_black_cards - black_cards_taken_out

theorem black_cards_remaining_proof : black_cards_remaining = 22 := 
by sorry

end black_cards_remaining_proof_l67_6736


namespace vertex_angle_isosceles_triangle_l67_6796

theorem vertex_angle_isosceles_triangle (α : ℝ) (β : ℝ) (sum_of_angles : α + α + β = 180) (base_angle : α = 50) :
  β = 80 :=
by
  sorry

end vertex_angle_isosceles_triangle_l67_6796


namespace triangle_inequality_for_powers_l67_6720

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, (a ^ n + b ^ n > c ^ n)) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b = c) :=
sorry

end triangle_inequality_for_powers_l67_6720


namespace reciprocal_of_2023_l67_6714

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end reciprocal_of_2023_l67_6714


namespace students_voted_both_issues_l67_6753

-- Define the total number of students.
def total_students : ℕ := 150

-- Define the number of students who voted in favor of the first issue.
def voted_first_issue : ℕ := 110

-- Define the number of students who voted in favor of the second issue.
def voted_second_issue : ℕ := 95

-- Define the number of students who voted against both issues.
def voted_against_both : ℕ := 15

-- Theorem: Number of students who voted in favor of both issues is 70.
theorem students_voted_both_issues : 
  ((voted_first_issue + voted_second_issue) - (total_students - voted_against_both)) = 70 :=
by
  sorry

end students_voted_both_issues_l67_6753


namespace no_four_distinct_real_roots_l67_6756

theorem no_four_distinct_real_roots (a b : ℝ) :
  ¬ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0) := 
by {
  sorry
}

end no_four_distinct_real_roots_l67_6756


namespace impossible_sequence_l67_6717

def letters_order : List ℕ := [1, 2, 3, 4, 5]

def is_typing_sequence (order : List ℕ) (seq : List ℕ) : Prop :=
  sorry -- This function will evaluate if a sequence is possible given the order

theorem impossible_sequence : ¬ is_typing_sequence letters_order [4, 5, 2, 3, 1] :=
  sorry

end impossible_sequence_l67_6717


namespace ab_eq_one_l67_6767

theorem ab_eq_one (a b : ℝ) (h1 : a ≠ b) (h2 : abs (Real.log a) = abs (Real.log b)) : a * b = 1 := sorry

end ab_eq_one_l67_6767


namespace leftover_value_correct_l67_6731

noncomputable def leftover_value (nickels_per_roll pennies_per_roll : ℕ) (sarah_nickels sarah_pennies tom_nickels tom_pennies : ℕ) : ℚ :=
  let total_nickels := sarah_nickels + tom_nickels
  let total_pennies := sarah_pennies + tom_pennies
  let leftover_nickels := total_nickels % nickels_per_roll
  let leftover_pennies := total_pennies % pennies_per_roll
  (leftover_nickels * 5 + leftover_pennies) / 100

theorem leftover_value_correct :
  leftover_value 40 50 132 245 98 203 = 1.98 := 
by
  sorry

end leftover_value_correct_l67_6731


namespace arithmetic_sequence_solution_l67_6709

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 + a 4 = 4)
  (h2 : a 2 * a 3 = 3)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2):
  (a 1 = -1 ∧ (∀ n, a n = 2 * n - 3) ∧ (∀ n, S n = n^2 - 2 * n)) ∨ 
  (a 1 = 5 ∧ (∀ n, a n = 7 - 2 * n) ∧ (∀ n, S n = 6 * n - n^2)) :=
sorry

end arithmetic_sequence_solution_l67_6709


namespace ratio_closest_to_10_l67_6711

theorem ratio_closest_to_10 :
  (⌊(10^3000 + 10^3004 : ℝ) / (10^3001 + 10^3003) + 0.5⌋ : ℝ) = 10 :=
sorry

end ratio_closest_to_10_l67_6711


namespace Vasek_solved_18_problems_l67_6766

variables (m v z : ℕ)

theorem Vasek_solved_18_problems (h1 : m + v = 25) (h2 : z + v = 32) (h3 : z = 2 * m) : v = 18 := by 
  sorry

end Vasek_solved_18_problems_l67_6766


namespace find_first_prime_l67_6751

theorem find_first_prime (p1 p2 z : ℕ) 
  (prime_p1 : Nat.Prime p1)
  (prime_p2 : Nat.Prime p2)
  (z_eq : z = p1 * p2)
  (z_val : z = 33)
  (p2_range : 8 < p2 ∧ p2 < 24)
  : p1 = 3 := 
sorry

end find_first_prime_l67_6751


namespace pradeep_pass_percentage_l67_6764

variable (marks_obtained : ℕ) (marks_short : ℕ) (max_marks : ℝ)

theorem pradeep_pass_percentage (h1 : marks_obtained = 150) (h2 : marks_short = 25) (h3 : max_marks = 500.00000000000006) :
  ((marks_obtained + marks_short) / max_marks) * 100 = 35 := 
by
  sorry

end pradeep_pass_percentage_l67_6764


namespace books_sold_wednesday_l67_6718

-- Define the conditions of the problem
def total_books : Nat := 1200
def sold_monday : Nat := 75
def sold_tuesday : Nat := 50
def sold_thursday : Nat := 78
def sold_friday : Nat := 135
def percentage_not_sold : Real := 66.5

-- Define the statement to be proved
theorem books_sold_wednesday : 
  let books_sold := total_books * (1 - percentage_not_sold / 100)
  let known_sales := sold_monday + sold_tuesday + sold_thursday + sold_friday
  books_sold - known_sales = 64 :=
by
  sorry

end books_sold_wednesday_l67_6718


namespace tank_fraction_before_gas_added_l67_6707

theorem tank_fraction_before_gas_added (capacity : ℝ) (added_gasoline : ℝ) (fraction_after : ℝ) (initial_fraction : ℝ) :
  capacity = 42 → added_gasoline = 7 → fraction_after = 9 / 10 → (initial_fraction * capacity + added_gasoline = fraction_after * capacity) → initial_fraction = 733 / 1000 :=
by
  intros h_capacity h_added_gasoline h_fraction_after h_equation
  sorry

end tank_fraction_before_gas_added_l67_6707


namespace final_speed_of_ball_l67_6746

/--
 A small rubber ball moves horizontally between two vertical walls. One wall is fixed, and the other wall moves away from it at a constant speed u.
 The ball's collisions are perfectly elastic. The initial speed of the ball is v₀. Prove that after 10 collisions with the moving wall, the ball's speed is 17 cm/s.
-/
theorem final_speed_of_ball
    (u : ℝ) (v₀ : ℝ) (n : ℕ)
    (u_val : u = 100) (v₀_val : v₀ = 2017) (n_val : n = 10) :
    v₀ - 2 * u * n = 17 := 
    by
    rw [u_val, v₀_val, n_val]
    sorry

end final_speed_of_ball_l67_6746


namespace scientific_notation_21500000_l67_6783

/-- Express the number 21500000 in scientific notation. -/
theorem scientific_notation_21500000 : 21500000 = 2.15 * 10^7 := 
sorry

end scientific_notation_21500000_l67_6783


namespace find_angle_A_triangle_is_right_l67_6780

theorem find_angle_A (A : ℝ) (h : 2 * Real.cos (Real.pi + A) + Real.sin (Real.pi / 2 + 2 * A) + 3 / 2 = 0) :
  A = Real.pi / 3 := 
sorry

theorem triangle_is_right (a b c : ℝ) (A : ℝ) (ha : c - b = (Real.sqrt 3) / 3 * a) (hA : A = Real.pi / 3) :
  c^2 = a^2 + b^2 :=
sorry

end find_angle_A_triangle_is_right_l67_6780


namespace initial_ratio_zinc_copper_l67_6730

theorem initial_ratio_zinc_copper (Z C : ℝ) 
  (h1 : Z + C = 6) 
  (h2 : Z + 8 = 3 * C) : 
  Z / C = 5 / 7 := 
sorry

end initial_ratio_zinc_copper_l67_6730


namespace value_of_a_l67_6754

theorem value_of_a {a : ℝ} 
  (h : ∀ x y : ℝ, ax - 2*y + 2 = 0 ↔ x + (a-3)*y + 1 = 0) : 
  a = 1 := 
by 
  sorry

end value_of_a_l67_6754


namespace find_a_l67_6793

theorem find_a (a : ℝ) (h1 : ∀ θ : ℝ, x = a + 4 * Real.cos θ ∧ y = 1 + 4 * Real.sin θ)
  (h2 : ∃ p : ℝ × ℝ, (3 * p.1 + 4 * p.2 - 5 = 0 ∧ (∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ))))
  (h3 : ∀ (p1 p2 : ℝ × ℝ), 
        (3 * p1.1 + 4 * p1.2 - 5 = 0 ∧ 3 * p2.1 + 4 * p2.2 - 5 = 0) ∧
        (∃ θ1 : ℝ, p1 = (a + 4 * Real.cos θ1, 1 + 4 * Real.sin θ1)) ∧
        (∃ θ2 : ℝ, p2 = (a + 4 * Real.cos θ2, 1 + 4 * Real.sin θ2)) → p1 = p2) :
  a = 7 := by
  sorry

end find_a_l67_6793


namespace deck_cost_l67_6739

variable (rareCount : ℕ := 19)
variable (uncommonCount : ℕ := 11)
variable (commonCount : ℕ := 30)
variable (rareCost : ℝ := 1.0)
variable (uncommonCost : ℝ := 0.5)
variable (commonCost : ℝ := 0.25)

theorem deck_cost : rareCount * rareCost + uncommonCount * uncommonCost + commonCount * commonCost = 32 := by
  sorry

end deck_cost_l67_6739


namespace mira_jogging_distance_l67_6743

def jogging_speed : ℝ := 5 -- speed in miles per hour
def jogging_hours_per_day : ℝ := 2 -- hours per day
def days_count : ℕ := 5 -- number of days

theorem mira_jogging_distance :
  (jogging_speed * jogging_hours_per_day * days_count : ℝ) = 50 :=
by
  sorry

end mira_jogging_distance_l67_6743


namespace probability_of_negative_l67_6710

def set_of_numbers : Set ℤ := {-2, 1, 4, -3, 0}
def negative_numbers : Set ℤ := {-2, -3}
def total_numbers : ℕ := 5
def total_negative_numbers : ℕ := 2

theorem probability_of_negative :
  (total_negative_numbers : ℚ) / (total_numbers : ℚ) = 2 / 5 := 
by 
  sorry

end probability_of_negative_l67_6710


namespace find_original_radius_l67_6716

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius_l67_6716


namespace liam_comic_books_l67_6774

theorem liam_comic_books (cost_per_book : ℚ) (total_money : ℚ) (n : ℚ) : cost_per_book = 1.25 ∧ total_money = 10 → n = 8 :=
by
  intros h
  cases h
  have h1 : 1.25 * n ≤ 10 := by sorry
  have h2 : n ≤ 10 / 1.25 := by sorry
  have h3 : n ≤ 8 := by sorry
  have h4 : n = 8 := by sorry
  exact h4

end liam_comic_books_l67_6774


namespace find_x_l67_6755

theorem find_x (x : ℝ) (h : 0.90 * 600 = 0.50 * x) : x = 1080 :=
sorry

end find_x_l67_6755


namespace cookies_per_bag_l67_6787

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (H1 : total_cookies = 703) (H2 : num_bags = 37) : total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l67_6787


namespace sample_size_l67_6713

theorem sample_size (f_c f_o N: ℕ) (h1: f_c = 8) (h2: f_c = 1 / 4 * f_o) (h3: f_c + f_o = N) : N = 40 :=
  sorry

end sample_size_l67_6713


namespace bridget_bought_17_apples_l67_6728

noncomputable def total_apples (x : ℕ) : Prop :=
  (2 * x / 3) - 5 = 6

theorem bridget_bought_17_apples : ∃ x : ℕ, total_apples x ∧ x = 17 :=
  sorry

end bridget_bought_17_apples_l67_6728


namespace polynomial_root_solution_l67_6704

theorem polynomial_root_solution (a b c : ℝ) (h1 : (2:ℝ)^5 + 4*(2:ℝ)^4 + a*(2:ℝ)^2 = b*(2:ℝ) + 4*c) 
  (h2 : (-2:ℝ)^5 + 4*(-2:ℝ)^4 + a*(-2:ℝ)^2 = b*(-2:ℝ) + 4*c) :
  a = -48 ∧ b = 16 ∧ c = -32 :=
sorry

end polynomial_root_solution_l67_6704


namespace range_of_m_l67_6723

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l67_6723


namespace number_of_students_l67_6702

theorem number_of_students (n : ℕ) (bow_cost : ℕ) (vinegar_cost : ℕ) (baking_soda_cost : ℕ) (total_cost : ℕ) :
  bow_cost = 5 → vinegar_cost = 2 → baking_soda_cost = 1 → total_cost = 184 → 8 * n = total_cost → n = 23 :=
by
  intros h_bow h_vinegar h_baking_soda h_total_cost h_equation
  sorry

end number_of_students_l67_6702


namespace problem_proof_l67_6789

theorem problem_proof (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
  (h2 : a + b + c + d = m^2) 
  (h3 : max (max a b) (max c d) = n^2) : 
  m = 9 ∧ n = 6 :=
by
  sorry

end problem_proof_l67_6789


namespace great_dane_weight_l67_6722

theorem great_dane_weight : 
  ∀ (C P G : ℕ), 
    C + P + G = 439 ∧ P = 3 * C ∧ G = 3 * P + 10 → G = 307 := by
    sorry

end great_dane_weight_l67_6722


namespace bus_speed_excluding_stoppages_l67_6757

noncomputable def average_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppage_time_ratio : ℝ) : ℝ :=
  (speed_including_stoppages * 1) / (1 - stoppage_time_ratio)

theorem bus_speed_excluding_stoppages :
  average_speed_excluding_stoppages 15 (3/4) = 60 := 
by
  sorry

end bus_speed_excluding_stoppages_l67_6757


namespace find_m_and_n_l67_6778

namespace BinomialProof

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n m : ℕ) : Prop :=
  binom (n+1) (m+1) = binom (n+1) m

def condition2 (n m : ℕ) : Prop :=
  binom (n+1) m / binom (n+1) (m-1) = 5 / 3

-- Problem statement
theorem find_m_and_n : ∃ (m n : ℕ), 
  (condition1 n m) ∧ 
  (condition2 n m) ∧ 
  m = 3 ∧ n = 6 := sorry

end BinomialProof

end find_m_and_n_l67_6778


namespace Tim_is_65_l67_6750

def James_age : Nat := 23
def John_age : Nat := 35
def Tim_age : Nat := 2 * John_age - 5

theorem Tim_is_65 : Tim_age = 65 := by
  sorry

end Tim_is_65_l67_6750


namespace alices_favorite_number_l67_6724

theorem alices_favorite_number :
  ∃ n : ℕ, 80 < n ∧ n ≤ 130 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ ((n / 100) + (n % 100 / 10) + (n % 10)) % 4 = 0 ∧ n = 130 :=
by
  sorry

end alices_favorite_number_l67_6724


namespace Larry_wins_game_probability_l67_6785

noncomputable def winning_probability_Larry : ℚ :=
  ∑' n : ℕ, if n % 3 = 0 then (2 / 3) ^ (n / 3 * 3) * (1 / 3) else 0

theorem Larry_wins_game_probability : winning_probability_Larry = 9 / 19 :=
by
  sorry

end Larry_wins_game_probability_l67_6785


namespace eliana_additional_steps_first_day_l67_6706

variables (x : ℝ)

def eliana_first_day_steps := 200 + x
def eliana_second_day_steps := 2 * eliana_first_day_steps
def eliana_third_day_steps := eliana_second_day_steps + 100
def eliana_total_steps := eliana_first_day_steps + eliana_second_day_steps + eliana_third_day_steps

theorem eliana_additional_steps_first_day : eliana_total_steps = 1600 → x = 100 :=
by {
  sorry
}

end eliana_additional_steps_first_day_l67_6706


namespace cost_of_each_taco_l67_6762

variables (T E : ℝ)

-- Conditions
axiom condition1 : 2 * T + 3 * E = 7.80
axiom condition2 : 3 * T + 5 * E = 12.70

-- Question to prove
theorem cost_of_each_taco : T = 0.90 :=
by
  sorry

end cost_of_each_taco_l67_6762


namespace intersection_M_N_l67_6737

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end intersection_M_N_l67_6737


namespace find_m_l67_6733

theorem find_m (m : ℝ) (h : (1 : ℝ) ^ 2 - m * (1 : ℝ) + 2 = 0) : m = 3 :=
by
  sorry

end find_m_l67_6733


namespace nh3_oxidation_mass_l67_6745

theorem nh3_oxidation_mass
  (initial_volume : ℚ)
  (initial_cl2_percentage : ℚ)
  (initial_n2_percentage : ℚ)
  (escaped_volume : ℚ)
  (escaped_cl2_percentage : ℚ)
  (escaped_n2_percentage : ℚ)
  (molar_volume : ℚ)
  (cl2_molar_mass : ℚ)
  (nh3_molar_mass : ℚ) :
  initial_volume = 1.12 →
  initial_cl2_percentage = 0.9 →
  initial_n2_percentage = 0.1 →
  escaped_volume = 0.672 →
  escaped_cl2_percentage = 0.5 →
  escaped_n2_percentage = 0.5 →
  molar_volume = 22.4 →
  cl2_molar_mass = 71 →
  nh3_molar_mass = 17 →
  ∃ (mass_nh3_oxidized : ℚ),
    mass_nh3_oxidized = 0.34 := 
by {
  sorry
}

end nh3_oxidation_mass_l67_6745


namespace fish_count_l67_6790

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l67_6790


namespace remaining_yards_correct_l67_6729

-- Define the conversion constant
def yards_per_mile: ℕ := 1760

-- Define the conditions
def marathon_in_miles: ℕ := 26
def marathon_in_yards: ℕ := 395
def total_marathons: ℕ := 15

-- Define the function to calculate the remaining yards after conversion
def calculate_remaining_yards (marathon_in_miles marathon_in_yards total_marathons yards_per_mile: ℕ): ℕ :=
  let total_yards := total_marathons * marathon_in_yards
  total_yards % yards_per_mile

-- Statement to prove
theorem remaining_yards_correct :
  calculate_remaining_yards marathon_in_miles marathon_in_yards total_marathons yards_per_mile = 645 :=
  sorry

end remaining_yards_correct_l67_6729


namespace find_PQ_length_l67_6744

-- Define the lengths of the sides of the triangles and the angle
def PQ_length : ℝ := 9
def QR_length : ℝ := 20
def PR_length : ℝ := 15
def ST_length : ℝ := 4.5
def TU_length : ℝ := 7.5
def SU_length : ℝ := 15
def angle_PQR : ℝ := 135
def angle_STU : ℝ := 135

-- Define the similarity condition
def triangles_similar (PQ QR PR ST TU SU angle_PQR angle_STU : ℝ) : Prop :=
  angle_PQR = angle_STU ∧ PQ / QR = ST / TU

-- Theorem statement
theorem find_PQ_length (PQ QR PR ST TU SU angle_PQR angle_STU: ℝ) 
  (H : triangles_similar PQ QR PR ST TU SU angle_PQR angle_STU) : PQ = 20 :=
by
  sorry

end find_PQ_length_l67_6744


namespace cost_of_baseball_is_correct_l67_6791

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l67_6791


namespace sqrt_expression_eq_36_l67_6727

theorem sqrt_expression_eq_36 : (Real.sqrt ((3^2 + 3^3)^2)) = 36 := 
by
  sorry

end sqrt_expression_eq_36_l67_6727


namespace domain_of_g_l67_6782

theorem domain_of_g :
  {x : ℝ | -6*x^2 - 7*x + 8 >= 0} = 
  {x : ℝ | (7 - Real.sqrt 241) / 12 ≤ x ∧ x ≤ (7 + Real.sqrt 241) / 12} :=
by
  sorry

end domain_of_g_l67_6782


namespace average_distance_per_day_l67_6784

def miles_monday : ℕ := 12
def miles_tuesday : ℕ := 18
def miles_wednesday : ℕ := 21
def total_days : ℕ := 3

def total_distance : ℕ := miles_monday + miles_tuesday + miles_wednesday

theorem average_distance_per_day : total_distance / total_days = 17 := by
  sorry

end average_distance_per_day_l67_6784


namespace abs_eq_condition_l67_6770

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition_l67_6770


namespace algebraic_expression_no_linear_term_l67_6732

theorem algebraic_expression_no_linear_term (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 1/2) = x^2 - a/2 ↔ a = 1/2) :=
by
  sorry

end algebraic_expression_no_linear_term_l67_6732


namespace average_marks_of_all_students_l67_6792

theorem average_marks_of_all_students (n1 n2 a1 a2 : ℕ) (n1_eq : n1 = 12) (a1_eq : a1 = 40) 
  (n2_eq : n2 = 28) (a2_eq : a2 = 60) : 
  ((n1 * a1 + n2 * a2) / (n1 + n2) : ℕ) = 54 := 
by
  sorry

end average_marks_of_all_students_l67_6792


namespace train_length_l67_6726

/-- Given that the jogger runs at 2.5 m/s,
    the train runs at 12.5 m/s, 
    the jogger is initially 260 meters ahead, 
    and the train takes 38 seconds to pass the jogger,
    prove that the length of the train is 120 meters. -/
theorem train_length (speed_jogger speed_train : ℝ) (initial_distance time_passing : ℝ)
  (hjogger : speed_jogger = 2.5) (htrain : speed_train = 12.5)
  (hinitial : initial_distance = 260) (htime : time_passing = 38) :
  ∃ L : ℝ, L = 120 :=
by
  sorry

end train_length_l67_6726


namespace chocolate_bar_min_breaks_l67_6721

theorem chocolate_bar_min_breaks (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∃ k, k = m * n - 1 := by
  sorry

end chocolate_bar_min_breaks_l67_6721


namespace chapatis_order_count_l67_6758

theorem chapatis_order_count (chapati_cost rice_cost veg_cost total_paid chapati_count : ℕ) 
  (rice_plates veg_plates : ℕ)
  (H1 : chapati_cost = 6)
  (H2 : rice_cost = 45)
  (H3 : veg_cost = 70)
  (H4 : total_paid = 1111)
  (H5 : rice_plates = 5)
  (H6 : veg_plates = 7)
  (H7 : chapati_count = (total_paid - (rice_plates * rice_cost + veg_plates * veg_cost)) / chapati_cost) :
  chapati_count = 66 :=
by
  sorry

end chapatis_order_count_l67_6758


namespace product_of_sisters_and_brothers_l67_6701

-- Lucy's family structure
def lucy_sisters : ℕ := 4
def lucy_brothers : ℕ := 6

-- Liam's siblings count
def liam_sisters : ℕ := lucy_sisters + 1  -- Including Lucy herself
def liam_brothers : ℕ := lucy_brothers    -- Excluding himself

-- Prove the product of Liam's sisters and brothers is 25
theorem product_of_sisters_and_brothers : liam_sisters * (liam_brothers - 1) = 25 :=
by
  sorry

end product_of_sisters_and_brothers_l67_6701


namespace fermats_little_theorem_analogue_l67_6735

theorem fermats_little_theorem_analogue 
  (a : ℤ) (h1 : Int.gcd a 561 = 1) : a ^ 560 ≡ 1 [ZMOD 561] := 
sorry

end fermats_little_theorem_analogue_l67_6735


namespace total_road_signs_l67_6798

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l67_6798


namespace abc_sum_eq_11sqrt6_l67_6725

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l67_6725


namespace find_value_l67_6734

theorem find_value (number : ℕ) (h : number / 5 + 16 = 58) : number / 15 + 74 = 88 :=
sorry

end find_value_l67_6734


namespace locus_of_centers_l67_6703

-- The Lean 4 statement
theorem locus_of_centers (a b : ℝ) 
  (C1 : (x y : ℝ) → x^2 + y^2 = 1)
  (C2 : (x y : ℝ) → (x - 3)^2 + y^2 = 25) :
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end locus_of_centers_l67_6703
