import Mathlib

namespace NUMINAMATH_GPT_chocolate_bars_partial_boxes_l1505_150575

-- Define the total number of bars for each type
def totalA : ℕ := 853845
def totalB : ℕ := 537896
def totalC : ℕ := 729763

-- Define the box capacities for each type
def capacityA : ℕ := 9
def capacityB : ℕ := 11
def capacityC : ℕ := 15

-- State the theorem we want to prove
theorem chocolate_bars_partial_boxes :
  totalA % capacityA = 4 ∧
  totalB % capacityB = 3 ∧
  totalC % capacityC = 8 :=
by
  -- Proof omitted for this task
  sorry

end NUMINAMATH_GPT_chocolate_bars_partial_boxes_l1505_150575


namespace NUMINAMATH_GPT_group_for_2019_is_63_l1505_150524

def last_term_of_group (n : ℕ) : ℕ := (n * (n + 1)) / 2 + n

theorem group_for_2019_is_63 :
  ∃ n : ℕ, (2015 < 2019 ∧ 2019 ≤ 2079) :=
by
  sorry

end NUMINAMATH_GPT_group_for_2019_is_63_l1505_150524


namespace NUMINAMATH_GPT_volume_tetrahedron_constant_l1505_150564

theorem volume_tetrahedron_constant (m n h : ℝ) (ϕ : ℝ) :
  ∃ V : ℝ, V = (1 / 6) * m * n * h * Real.sin ϕ :=
by
  sorry

end NUMINAMATH_GPT_volume_tetrahedron_constant_l1505_150564


namespace NUMINAMATH_GPT_accuracy_l1505_150568

-- Given number and accuracy statement
def given_number : ℝ := 3.145 * 10^8
def expanded_form : ℕ := 314500000

-- Proof statement: the number is accurate to the hundred thousand's place
theorem accuracy (h : given_number = expanded_form) : 
  ∃ n : ℕ, expanded_form = n * 10^5 ∧ (n % 10) ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_accuracy_l1505_150568


namespace NUMINAMATH_GPT_last_date_in_2011_divisible_by_101_is_1221_l1505_150582

def is_valid_date (a b c d : ℕ) : Prop :=
  (10 * a + b) ≤ 12 ∧ (10 * c + d) ≤ 31

def date_as_number (a b c d : ℕ) : ℕ :=
  20110000 + 1000 * a + 100 * b + 10 * c + d

theorem last_date_in_2011_divisible_by_101_is_1221 :
  ∃ (a b c d : ℕ), is_valid_date a b c d ∧ date_as_number a b c d % 101 = 0 ∧ date_as_number a b c d = 20111221 :=
by
  sorry

end NUMINAMATH_GPT_last_date_in_2011_divisible_by_101_is_1221_l1505_150582


namespace NUMINAMATH_GPT_at_least_six_heads_in_10_flips_is_129_over_1024_l1505_150598

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end NUMINAMATH_GPT_at_least_six_heads_in_10_flips_is_129_over_1024_l1505_150598


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1505_150530

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (a1 : a 1 = 3)
  (a4 : a 4 = 24)
  (h_geo : ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 1)) :
  a 3 + a 4 + a 5 = 84 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1505_150530


namespace NUMINAMATH_GPT_boys_count_in_dance_class_l1505_150528

theorem boys_count_in_dance_class
  (total_students : ℕ) 
  (ratio_girls_to_boys : ℕ) 
  (ratio_boys_to_girls: ℕ)
  (total_students_eq : total_students = 35)
  (ratio_eq : ratio_girls_to_boys = 3 ∧ ratio_boys_to_girls = 4) : 
  ∃ boys : ℕ, boys = 20 :=
by
  let k := total_students / (ratio_girls_to_boys + ratio_boys_to_girls)
  have girls := ratio_girls_to_boys * k
  have boys := ratio_boys_to_girls * k
  use boys
  sorry

end NUMINAMATH_GPT_boys_count_in_dance_class_l1505_150528


namespace NUMINAMATH_GPT_no_integer_solutions_l1505_150562

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1505_150562


namespace NUMINAMATH_GPT_Gwendolyn_will_take_50_hours_to_read_l1505_150513

def GwendolynReadingTime (sentences_per_hour : ℕ) (sentences_per_paragraph : ℕ) (paragraphs_per_page : ℕ) (pages : ℕ) : ℕ :=
  (sentences_per_paragraph * paragraphs_per_page * pages) / sentences_per_hour

theorem Gwendolyn_will_take_50_hours_to_read 
  (h1 : 200 = 200)
  (h2 : 10 = 10)
  (h3 : 20 = 20)
  (h4 : 50 = 50) :
  GwendolynReadingTime 200 10 20 50 = 50 := by
  sorry

end NUMINAMATH_GPT_Gwendolyn_will_take_50_hours_to_read_l1505_150513


namespace NUMINAMATH_GPT_max_full_box_cards_l1505_150536

-- Given conditions
def total_cards : ℕ := 94
def unfilled_box_cards : ℕ := 6

-- Define the number of cards that are evenly distributed into full boxes
def evenly_distributed_cards : ℕ := total_cards - unfilled_box_cards

-- Prove that the maximum number of cards a full box can hold is 22
theorem max_full_box_cards (h : evenly_distributed_cards = 88) : ∃ x : ℕ, evenly_distributed_cards % x = 0 ∧ x = 22 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_full_box_cards_l1505_150536


namespace NUMINAMATH_GPT_find_n_cubes_l1505_150555

theorem find_n_cubes (n : ℕ) (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h1 : 837 + n = y^3) (h2 : 837 - n = x^3) : n = 494 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_cubes_l1505_150555


namespace NUMINAMATH_GPT_polar_to_rectangular_l1505_150533

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3 * Real.sqrt 2) (h2 : θ = Real.pi / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l1505_150533


namespace NUMINAMATH_GPT_angle_of_parallel_l1505_150522

-- Define a line and a plane
variable {L : Type} (l : L)
variable {P : Type} (β : P)

-- Define the parallel condition
def is_parallel (l : L) (β : P) : Prop := sorry

-- Define the angle function between a line and a plane
def angle (l : L) (β : P) : ℝ := sorry

-- The theorem stating that if l is parallel to β, then the angle is 0
theorem angle_of_parallel (h : is_parallel l β) : angle l β = 0 := sorry

end NUMINAMATH_GPT_angle_of_parallel_l1505_150522


namespace NUMINAMATH_GPT_projected_percent_increase_l1505_150567

theorem projected_percent_increase (R : ℝ) (p : ℝ) 
  (h1 : 0.7 * R = R * 0.7) 
  (h2 : 0.7 * R = 0.5 * (R + p * R)) : 
  p = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_projected_percent_increase_l1505_150567


namespace NUMINAMATH_GPT_cafeteria_problem_l1505_150525

theorem cafeteria_problem (C : ℕ) 
    (h1 : ∃ h : ℕ, h = 4 * C)
    (h2 : 5 = 5)
    (h3 : C + 4 * C + 5 = 40) : 
    C = 7 := sorry

end NUMINAMATH_GPT_cafeteria_problem_l1505_150525


namespace NUMINAMATH_GPT_product_of_two_numbers_l1505_150510

theorem product_of_two_numbers (a b : ℤ) (h1 : lcm a b = 72) (h2 : gcd a b = 8) :
  a * b = 576 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1505_150510


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1505_150586

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) := sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1505_150586


namespace NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l1505_150504

-- Define what it means for a number to be prime
def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ (n : ℕ), n > 1 ∧ n < p → ¬ (p % n = 0)

-- Define what a Mersenne prime is
def is_mersenne_prime (m : ℕ) : Prop :=
∃ n : ℕ, is_prime n ∧ m = 2^n - 1

-- We state the main theorem we want to prove
theorem largest_mersenne_prime_less_than_500 : ∀ (m : ℕ), is_mersenne_prime m ∧ m < 500 → m ≤ 127 :=
by 
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l1505_150504


namespace NUMINAMATH_GPT_system_of_equations_solution_l1505_150592

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x + z = 0) 
  (h3 : y + z = 1) : 
  x = -1 ∧ y = 0 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1505_150592


namespace NUMINAMATH_GPT_george_final_score_l1505_150573

-- Definitions for points in the first half
def first_half_odd_points (questions : Nat) := 5 * 2
def first_half_even_points (questions : Nat) := 5 * 4
def first_half_bonus_points (questions : Nat) := 3 * 5
def first_half_points := first_half_odd_points 5 + first_half_even_points 5 + first_half_bonus_points 3

-- Definitions for points in the second half
def second_half_odd_points (questions : Nat) := 6 * 3
def second_half_even_points (questions : Nat) := 6 * 5
def second_half_bonus_points (questions : Nat) := 4 * 5
def second_half_points := second_half_odd_points 6 + second_half_even_points 6 + second_half_bonus_points 4

-- Definition of the total points
def total_points := first_half_points + second_half_points

-- The theorem statement to prove the total points
theorem george_final_score : total_points = 113 := by
  unfold total_points
  unfold first_half_points
  unfold second_half_points
  unfold first_half_odd_points first_half_even_points first_half_bonus_points
  unfold second_half_odd_points second_half_even_points second_half_bonus_points
  sorry

end NUMINAMATH_GPT_george_final_score_l1505_150573


namespace NUMINAMATH_GPT_find_mn_l1505_150502

theorem find_mn
  (AB BC : ℝ) -- Lengths of AB and BC
  (m n : ℝ)   -- Coefficients of the quadratic equation
  (h_perimeter : 2 * (AB + BC) = 12)
  (h_area : AB * BC = 5)
  (h_roots_sum : AB + BC = -m)
  (h_roots_product : AB * BC = n) :
  m * n = -30 :=
by
  sorry

end NUMINAMATH_GPT_find_mn_l1505_150502


namespace NUMINAMATH_GPT_polygon_area_is_12_l1505_150501

def polygon_vertices := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

def area_of_polygon (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to compute the area (stub here for now)
  sorry

theorem polygon_area_is_12 :
  area_of_polygon polygon_vertices = 12 :=
by
  sorry

end NUMINAMATH_GPT_polygon_area_is_12_l1505_150501


namespace NUMINAMATH_GPT_speeds_of_bus_and_car_l1505_150558

theorem speeds_of_bus_and_car
  (d t : ℝ) (v1 v2 : ℝ)
  (h1 : 1.5 * v1 + 1.5 * v2 = d)
  (h2 : 2.5 * v1 + 1 * v2 = d) :
  v1 = 40 ∧ v2 = 80 :=
by sorry

end NUMINAMATH_GPT_speeds_of_bus_and_car_l1505_150558


namespace NUMINAMATH_GPT_probability_at_least_one_black_ball_l1505_150545

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black_ball :
  (∃ (p : ℚ), p = 13 / 14 ∧ 
  (number_of_ways_to_choose4_balls_has_at_least_1_black / number_of_ways_to_choose4_balls) = p) :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_black_ball_l1505_150545


namespace NUMINAMATH_GPT_manage_committee_combination_l1505_150538

theorem manage_committee_combination : (Nat.choose 20 3) = 1140 := by
  sorry

end NUMINAMATH_GPT_manage_committee_combination_l1505_150538


namespace NUMINAMATH_GPT_f_even_l1505_150587

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_f_even_l1505_150587


namespace NUMINAMATH_GPT_play_number_of_children_l1505_150571

theorem play_number_of_children (A C : ℕ) (ticket_price_adult : ℕ) (ticket_price_child : ℕ)
    (total_people : ℕ) (total_money : ℕ)
    (h1 : ticket_price_adult = 8)
    (h2 : ticket_price_child = 1)
    (h3 : total_people = 22)
    (h4 : total_money = 50)
    (h5 : A + C = total_people)
    (h6 : ticket_price_adult * A + ticket_price_child * C = total_money) :
    C = 18 := sorry

end NUMINAMATH_GPT_play_number_of_children_l1505_150571


namespace NUMINAMATH_GPT_lcm_gcd_48_180_l1505_150552

theorem lcm_gcd_48_180 :
  Nat.lcm 48 180 = 720 ∧ Nat.gcd 48 180 = 12 :=
by
  sorry

end NUMINAMATH_GPT_lcm_gcd_48_180_l1505_150552


namespace NUMINAMATH_GPT_vinegar_used_is_15_l1505_150560

noncomputable def vinegar_used (T : ℝ) : ℝ :=
  let water := (3 / 5) * 20
  let total_volume := 27
  let vinegar := total_volume - water
  vinegar

theorem vinegar_used_is_15 (T : ℝ) (h1 : (3 / 5) * 20 = 12) (h2 : 27 - 12 = 15) (h3 : (5 / 6) * T = 15) : vinegar_used T = 15 :=
by
  sorry

end NUMINAMATH_GPT_vinegar_used_is_15_l1505_150560


namespace NUMINAMATH_GPT_sum_of_sequences_l1505_150549

noncomputable def arithmetic_sequence (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = 6 + k ∧ b = 6 + 2 * k

noncomputable def geometric_sequence (c d : ℤ) : Prop :=
  ∃ q : ℤ, c = 6 * q ∧ d = 6 * q^2

theorem sum_of_sequences (a b c d : ℤ) 
  (h_arith : arithmetic_sequence a b) 
  (h_geom : geometric_sequence c d) 
  (hb : b = 48) (hd : 6 * c^2 = 48): 
  a + b + c + d = 111 := 
sorry

end NUMINAMATH_GPT_sum_of_sequences_l1505_150549


namespace NUMINAMATH_GPT_cookie_combinations_l1505_150583

theorem cookie_combinations (total_cookies kinds : Nat) (at_least_one : kinds > 0 ∧ ∀ k : Nat, k < kinds → k > 0) : 
  (total_cookies = 8 ∧ kinds = 4) → 
  (∃ comb : Nat, comb = 34) := 
by 
  -- insert proof here 
  sorry

end NUMINAMATH_GPT_cookie_combinations_l1505_150583


namespace NUMINAMATH_GPT_combined_percent_increase_proof_l1505_150588

variable (initial_stock_A_price : ℝ := 25)
variable (initial_stock_B_price : ℝ := 45)
variable (initial_stock_C_price : ℝ := 60)
variable (final_stock_A_price : ℝ := 28)
variable (final_stock_B_price : ℝ := 50)
variable (final_stock_C_price : ℝ := 75)

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  (percent_increase initial_a final_a + percent_increase initial_b final_b + percent_increase initial_c final_c) / 3

theorem combined_percent_increase_proof :
  combined_percent_increase initial_stock_A_price initial_stock_B_price initial_stock_C_price
                            final_stock_A_price final_stock_B_price final_stock_C_price = 16.04 := by
  sorry

end NUMINAMATH_GPT_combined_percent_increase_proof_l1505_150588


namespace NUMINAMATH_GPT_max_marks_is_667_l1505_150563

-- Definitions based on the problem's conditions
def pass_threshold (M : ℝ) : ℝ := 0.45 * M
def student_score : ℝ := 225
def failed_by : ℝ := 75
def passing_marks := student_score + failed_by

-- The actual theorem stating that if the conditions are met, then the maximum marks M is 667
theorem max_marks_is_667 : ∃ M : ℝ, pass_threshold M = passing_marks ∧ M = 667 :=
by
  sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_max_marks_is_667_l1505_150563


namespace NUMINAMATH_GPT_geometric_progression_first_term_l1505_150561

theorem geometric_progression_first_term (a r : ℝ) 
    (h_sum_inf : a / (1 - r) = 8)
    (h_sum_two : a * (1 + r) = 5) :
    a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) :=
sorry

end NUMINAMATH_GPT_geometric_progression_first_term_l1505_150561


namespace NUMINAMATH_GPT_k_bounds_inequality_l1505_150500

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end NUMINAMATH_GPT_k_bounds_inequality_l1505_150500


namespace NUMINAMATH_GPT_solve_system_of_equations_l1505_150597

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1505_150597


namespace NUMINAMATH_GPT_haircuts_away_from_next_free_l1505_150506

def free_haircut (total_paid : ℕ) : ℕ := total_paid / 14

theorem haircuts_away_from_next_free (total_haircuts : ℕ) (free_haircuts : ℕ) (haircuts_per_free : ℕ) :
  total_haircuts = 79 → free_haircuts = 5 → haircuts_per_free = 14 → 
  (haircuts_per_free - (total_haircuts - free_haircuts)) % haircuts_per_free = 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_haircuts_away_from_next_free_l1505_150506


namespace NUMINAMATH_GPT_martha_jar_spices_cost_l1505_150508

def price_per_jar_spices (p_beef p_fv p_oj : ℕ) (price_spices : ℕ) :=
  let total_spent := (3 * p_beef) + (8 * p_fv) + p_oj + (3 * price_spices)
  let total_points := (total_spent / 10) * 50 + if total_spent > 100 then 250 else 0
  total_points

theorem martha_jar_spices_cost (price_spices : ℕ) :
  price_per_jar_spices 11 4 37 price_spices = 850 → price_spices = 6 := by
  sorry

end NUMINAMATH_GPT_martha_jar_spices_cost_l1505_150508


namespace NUMINAMATH_GPT_ralph_did_not_hit_110_balls_l1505_150521

def tennis_problem : Prop :=
  ∀ (total_balls first_batch second_batch hit_first hit_second not_hit_first not_hit_second not_hit_total : ℕ),
  total_balls = 175 →
  first_batch = 100 →
  second_batch = 75 →
  hit_first = 2/5 * first_batch →
  hit_second = 1/3 * second_batch →
  not_hit_first = first_batch - hit_first →
  not_hit_second = second_batch - hit_second →
  not_hit_total = not_hit_first + not_hit_second →
  not_hit_total = 110

theorem ralph_did_not_hit_110_balls : tennis_problem := by
  unfold tennis_problem
  intros
  sorry

end NUMINAMATH_GPT_ralph_did_not_hit_110_balls_l1505_150521


namespace NUMINAMATH_GPT_range_of_k_l1505_150580

noncomputable def point_satisfies_curve (a k : ℝ) : Prop :=
(-a)^2 - a * (-a) + 2 * a + k = 0

theorem range_of_k (a k : ℝ) (h : point_satisfies_curve a k) : k ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1505_150580


namespace NUMINAMATH_GPT_find_A_l1505_150548

def hash_relation (A B : ℕ) : ℕ := A^2 + B^2

theorem find_A (A : ℕ) (h1 : hash_relation A 7 = 218) : A = 13 := 
by sorry

end NUMINAMATH_GPT_find_A_l1505_150548


namespace NUMINAMATH_GPT_dolphins_trained_next_month_l1505_150503

theorem dolphins_trained_next_month
  (total_dolphins : ℕ) 
  (one_fourth_fully_trained : ℚ) 
  (two_thirds_in_training : ℚ)
  (h1 : total_dolphins = 20)
  (h2 : one_fourth_fully_trained = 1 / 4) 
  (h3 : two_thirds_in_training = 2 / 3) :
  (total_dolphins - total_dolphins * one_fourth_fully_trained) * two_thirds_in_training = 10 := 
by 
  sorry

end NUMINAMATH_GPT_dolphins_trained_next_month_l1505_150503


namespace NUMINAMATH_GPT_compare_exp_square_l1505_150511

theorem compare_exp_square (n : ℕ) : 
  (n ≥ 3 → 2^(2 * n) > (2 * n + 1)^2) ∧ ((n = 1 ∨ n = 2) → 2^(2 * n) < (2 * n + 1)^2) :=
by
  sorry

end NUMINAMATH_GPT_compare_exp_square_l1505_150511


namespace NUMINAMATH_GPT_cubic_roots_result_l1505_150590

theorem cubic_roots_result (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 64 + b * 16 + c * 4 + d = 0) (h₃ : a * (-27) + b * 9 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_result_l1505_150590


namespace NUMINAMATH_GPT_convex_polyhedron_same_number_of_sides_l1505_150534

theorem convex_polyhedron_same_number_of_sides {N : ℕ} (hN : N ≥ 4): 
  ∃ (f1 f2 : ℕ), (f1 >= 3 ∧ f1 < N ∧ f2 >= 3 ∧ f2 < N) ∧ f1 = f2 :=
by
  sorry

end NUMINAMATH_GPT_convex_polyhedron_same_number_of_sides_l1505_150534


namespace NUMINAMATH_GPT_sam_initial_money_l1505_150574

theorem sam_initial_money :
  (9 * 7 + 16 = 79) :=
by
  sorry

end NUMINAMATH_GPT_sam_initial_money_l1505_150574


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1505_150576

theorem problem1 : (70.8 - 1.25 - 1.75 = 67.8) := sorry

theorem problem2 : ((8 + 0.8) * 1.25 = 11) := sorry

theorem problem3 : (125 * 0.48 = 600) := sorry

theorem problem4 : (6.7 * (9.3 * (6.2 + 1.7)) = 554.559) := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1505_150576


namespace NUMINAMATH_GPT_perimeter_of_figure_l1505_150547

theorem perimeter_of_figure (x : ℕ) (h : x = 3) : 
  let sides := [x, x + 1, 6, 10]
  (sides.sum = 23) := by 
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_l1505_150547


namespace NUMINAMATH_GPT_cos_half_angle_l1505_150553

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi / 2) : 
    Real.cos (α / 2) = 2 * Real.sqrt 5 / 5 := 
by 
    sorry

end NUMINAMATH_GPT_cos_half_angle_l1505_150553


namespace NUMINAMATH_GPT_power_function_solution_l1505_150596

theorem power_function_solution (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = x ^ α) (h2 : f 4 = 2) : f 3 = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_power_function_solution_l1505_150596


namespace NUMINAMATH_GPT_solve_for_y_l1505_150516

theorem solve_for_y (y : ℕ) (h : 2^y + 8 = 4 * 2^y - 40) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1505_150516


namespace NUMINAMATH_GPT_ratio_of_b_to_a_l1505_150529

variable (V A B : ℝ)

def ten_pours_of_a_cup : Prop := 10 * A = V
def five_pours_of_b_cup : Prop := 5 * B = V

theorem ratio_of_b_to_a (h1 : ten_pours_of_a_cup V A) (h2 : five_pours_of_b_cup V B) : B / A = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_b_to_a_l1505_150529


namespace NUMINAMATH_GPT_max_product_of_three_numbers_l1505_150584

theorem max_product_of_three_numbers (n : ℕ) (h_n_pos : 0 < n) :
  ∃ a b c : ℕ, (a + b + c = 3 * n + 1) ∧ (∀ a' b' c' : ℕ,
        (a' + b' + c' = 3 * n + 1) →
        a' * b' * c' ≤ a * b * c) ∧
    (a * b * c = n^3 + n^2) :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_three_numbers_l1505_150584


namespace NUMINAMATH_GPT_total_classic_books_l1505_150542

-- Definitions for the conditions
def authors := 6
def books_per_author := 33

-- Statement of the math proof problem
theorem total_classic_books : authors * books_per_author = 198 := by
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_total_classic_books_l1505_150542


namespace NUMINAMATH_GPT_weighted_avg_surfers_per_day_l1505_150599

theorem weighted_avg_surfers_per_day 
  (total_surfers : ℕ) 
  (ratio1_day1 ratio1_day2 ratio2_day3 ratio2_day4 : ℕ) 
  (h_total_surfers : total_surfers = 12000)
  (h_ratio_first_two_days : ratio1_day1 = 5 ∧ ratio1_day2 = 7)
  (h_ratio_last_two_days : ratio2_day3 = 3 ∧ ratio2_day4 = 2) 
  : (total_surfers / (ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4)) * 
    ((ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4) / 4) = 3000 :=
by
  sorry

end NUMINAMATH_GPT_weighted_avg_surfers_per_day_l1505_150599


namespace NUMINAMATH_GPT_train_speed_excluding_stoppages_l1505_150520

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

end NUMINAMATH_GPT_train_speed_excluding_stoppages_l1505_150520


namespace NUMINAMATH_GPT_transaction_loss_l1505_150551

theorem transaction_loss 
  (sell_price_house sell_price_store : ℝ)
  (cost_price_house cost_price_store : ℝ)
  (house_loss_percent store_gain_percent : ℝ)
  (house_loss_eq : sell_price_house = (4/5) * cost_price_house)
  (store_gain_eq : sell_price_store = (6/5) * cost_price_store)
  (sell_prices_eq : sell_price_house = 12000 ∧ sell_price_store = 12000)
  (house_loss_percent_eq : house_loss_percent = 0.20)
  (store_gain_percent_eq : store_gain_percent = 0.20) :
  cost_price_house + cost_price_store - (sell_price_house + sell_price_store) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_transaction_loss_l1505_150551


namespace NUMINAMATH_GPT_inequality_solution_l1505_150595

theorem inequality_solution (x : ℝ) : (x^3 - 12*x^2 + 36*x > 0) ↔ (0 < x ∧ x < 6) ∨ (x > 6) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1505_150595


namespace NUMINAMATH_GPT_factorization_correct_l1505_150593

theorem factorization_correct (x : ℝ) : 
  (x^2 + 5 * x + 2) * (x^2 + 5 * x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5 * x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1505_150593


namespace NUMINAMATH_GPT_a10_is_b55_l1505_150505

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the new sequence b_n according to the given insertion rules
def b (k : ℕ) : ℕ := sorry

-- Prove that if a_10 = 19, then 19 is the 55th term in the new sequence b_n
theorem a10_is_b55 : b 55 = a 10 := sorry

end NUMINAMATH_GPT_a10_is_b55_l1505_150505


namespace NUMINAMATH_GPT_exists_xy_for_cube_difference_l1505_150577

theorem exists_xy_for_cube_difference (a : ℕ) (h : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
sorry

end NUMINAMATH_GPT_exists_xy_for_cube_difference_l1505_150577


namespace NUMINAMATH_GPT_cos_alpha_value_l1505_150544

theorem cos_alpha_value (α β γ: ℝ) (h1: β = 2 * α) (h2: γ = 4 * α)
 (h3: 2 * (Real.sin β) = (Real.sin α + Real.sin γ)) : Real.cos α = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l1505_150544


namespace NUMINAMATH_GPT_proof_problem_l1505_150532

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1505_150532


namespace NUMINAMATH_GPT_vote_majority_is_160_l1505_150579

-- Define the total number of votes polled
def total_votes : ℕ := 400

-- Define the percentage of votes polled by the winning candidate
def winning_percentage : ℝ := 0.70

-- Define the percentage of votes polled by the losing candidate
def losing_percentage : ℝ := 0.30

-- Define the number of votes gained by the winning candidate
def winning_votes := winning_percentage * total_votes

-- Define the number of votes gained by the losing candidate
def losing_votes := losing_percentage * total_votes

-- Define the vote majority
def vote_majority := winning_votes - losing_votes

-- Prove that the vote majority is 160 votes
theorem vote_majority_is_160 : vote_majority = 160 :=
sorry

end NUMINAMATH_GPT_vote_majority_is_160_l1505_150579


namespace NUMINAMATH_GPT_polynomial_factorization_l1505_150557

noncomputable def polynomial_equivalence : Prop :=
  ∀ x : ℂ, (x^12 - 3*x^9 + 3*x^3 + 1) = (x + 1)^4 * (x^2 - x + 1)^4

theorem polynomial_factorization : polynomial_equivalence := by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1505_150557


namespace NUMINAMATH_GPT_equation_has_one_real_root_l1505_150527

noncomputable def f (x : ℝ) : ℝ :=
  (3 / 11)^x + (5 / 11)^x + (7 / 11)^x - 1

theorem equation_has_one_real_root :
  ∃! x : ℝ, f x = 0 := sorry

end NUMINAMATH_GPT_equation_has_one_real_root_l1505_150527


namespace NUMINAMATH_GPT_distance_midpoint_AB_to_y_axis_l1505_150566

def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

variable (A B : parabola)
variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)

open scoped Classical

noncomputable def midpoint_x (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_midpoint_AB_to_y_axis 
  (h1 : x1 + x2 = 3) 
  (hA : A.val = (x1, y1))
  (hB : B.val = (x2, y2)) : 
  midpoint_x x1 x2 = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_distance_midpoint_AB_to_y_axis_l1505_150566


namespace NUMINAMATH_GPT_david_still_has_less_than_750_l1505_150512

theorem david_still_has_less_than_750 (S R : ℝ) 
  (h1 : S + R = 1500)
  (h2 : R < S) : 
  R < 750 :=
by 
  sorry

end NUMINAMATH_GPT_david_still_has_less_than_750_l1505_150512


namespace NUMINAMATH_GPT_find_c_gen_formula_l1505_150569

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
a 1 = 2 ∧
(∀ n, a (n + 1) = a n + c * n) ∧
(2 + c) * (2 + c) = 2 * (2 + 3 * c)

theorem find_c (a : ℕ → ℕ) : ∃ c, seq a c :=
by
  sorry

theorem gen_formula (a : ℕ → ℕ) (c : ℕ) (h : seq a c) : (∀ n, a n = n^2 - n + 2) :=
by
  sorry

end NUMINAMATH_GPT_find_c_gen_formula_l1505_150569


namespace NUMINAMATH_GPT_total_cups_l1505_150559

theorem total_cups (m c s : ℕ) (h1 : 3 * c = 2 * m) (h2 : 2 * c = 6) : m + c + s = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_cups_l1505_150559


namespace NUMINAMATH_GPT_part1_inequality_l1505_150578

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

theorem part1_inequality (m : ℝ) : (∀ x : ℝ, g x m > f x) ↔ (m ∈ Set.Ioo (-Real.sqrt 6 - (1/2)) (Real.sqrt 6 - (1/2))) :=
sorry

end NUMINAMATH_GPT_part1_inequality_l1505_150578


namespace NUMINAMATH_GPT_repetitive_decimals_subtraction_correct_l1505_150556

noncomputable def repetitive_decimals_subtraction : Prop :=
  let a : ℚ := 4567 / 9999
  let b : ℚ := 1234 / 9999
  let c : ℚ := 2345 / 9999
  a - b - c = 988 / 9999

theorem repetitive_decimals_subtraction_correct : repetitive_decimals_subtraction :=
  by sorry

end NUMINAMATH_GPT_repetitive_decimals_subtraction_correct_l1505_150556


namespace NUMINAMATH_GPT_initial_investment_l1505_150540

theorem initial_investment (P r : ℝ) 
  (h1 : 600 = P * (1 + 0.02 * r)) 
  (h2 : 850 = P * (1 + 0.07 * r)) : 
  P = 500 :=
sorry

end NUMINAMATH_GPT_initial_investment_l1505_150540


namespace NUMINAMATH_GPT_product_of_consecutive_natural_numbers_l1505_150507

theorem product_of_consecutive_natural_numbers (n : ℕ) : 
  (∃ t : ℕ, n = t * (t + 1) - 1) ↔ ∃ x : ℕ, n^2 - 1 = x * (x + 1) * (x + 2) * (x + 3) := 
sorry

end NUMINAMATH_GPT_product_of_consecutive_natural_numbers_l1505_150507


namespace NUMINAMATH_GPT_simplify_expression_l1505_150585

variable (a b : ℝ)

theorem simplify_expression :
  (a^3 - b^3) / (a * b) - (ab - b^2) / (ab - a^3) = (a^2 + ab + b^2) / b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1505_150585


namespace NUMINAMATH_GPT_mike_initial_marbles_l1505_150517

theorem mike_initial_marbles (n : ℕ) 
  (gave_to_sam : ℕ) (left_with_mike : ℕ)
  (h1 : gave_to_sam = 4)
  (h2 : left_with_mike = 4)
  (h3 : n = gave_to_sam + left_with_mike) : n = 8 := 
by
  sorry

end NUMINAMATH_GPT_mike_initial_marbles_l1505_150517


namespace NUMINAMATH_GPT_minimum_value_ineq_l1505_150581

theorem minimum_value_ineq (x : ℝ) (hx : x >= 4) : x + 4 / (x - 1) >= 5 := by
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l1505_150581


namespace NUMINAMATH_GPT_vertical_lines_count_l1505_150591

theorem vertical_lines_count (n : ℕ) 
  (h_intersections : (18 * n * (n - 1)) = 756) : 
  n = 7 :=
by 
  sorry

end NUMINAMATH_GPT_vertical_lines_count_l1505_150591


namespace NUMINAMATH_GPT_dealership_truck_sales_l1505_150570

theorem dealership_truck_sales (SUVs Trucks : ℕ) (h1 : SUVs = 45) (h2 : 3 * Trucks = 5 * SUVs) : Trucks = 75 :=
by
  sorry

end NUMINAMATH_GPT_dealership_truck_sales_l1505_150570


namespace NUMINAMATH_GPT_factor_expression_l1505_150594

variable (x y : ℝ)

theorem factor_expression :
(3*x^3 + 28*(x^2)*y + 4*x) - (-4*x^3 + 5*(x^2)*y - 4*x) = x*(x + 8)*(7*x + 1) := sorry

end NUMINAMATH_GPT_factor_expression_l1505_150594


namespace NUMINAMATH_GPT_least_value_y_l1505_150539

theorem least_value_y : ∃ y : ℝ, (3 * y ^ 3 + 3 * y ^ 2 + 5 * y + 1 = 5) ∧ ∀ z : ℝ, (3 * z ^ 3 + 3 * z ^ 2 + 5 * z + 1 = 5) → y ≤ z :=
sorry

end NUMINAMATH_GPT_least_value_y_l1505_150539


namespace NUMINAMATH_GPT_initial_deadlift_weight_l1505_150554

theorem initial_deadlift_weight
    (initial_squat : ℕ := 700)
    (initial_bench : ℕ := 400)
    (D : ℕ)
    (squat_loss : ℕ := 30)
    (deadlift_loss : ℕ := 200)
    (new_total : ℕ := 1490) :
    (initial_squat * (100 - squat_loss) / 100) + initial_bench + (D - deadlift_loss) = new_total → D = 800 :=
by
  sorry

end NUMINAMATH_GPT_initial_deadlift_weight_l1505_150554


namespace NUMINAMATH_GPT_no_member_of_T_is_divisible_by_4_or_5_l1505_150537

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4_or_5 :
  ∀ (n : ℤ), ¬ (∃ (T : ℤ), T = sum_of_squares_of_four_consecutive_integers n ∧ (T % 4 = 0 ∨ T % 5 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_member_of_T_is_divisible_by_4_or_5_l1505_150537


namespace NUMINAMATH_GPT_recurring_decimal_sum_l1505_150589

-- Definitions based on the conditions identified
def recurringDecimal (n : ℕ) : ℚ := n / 9
def r8 := recurringDecimal 8
def r2 := recurringDecimal 2
def r6 := recurringDecimal 6
def r6_simplified : ℚ := 2 / 3

-- The theorem to prove
theorem recurring_decimal_sum : r8 + r2 - r6_simplified = 4 / 9 :=
by
  -- Proof steps will go here (but are omitted because of the problem requirements)
  sorry

end NUMINAMATH_GPT_recurring_decimal_sum_l1505_150589


namespace NUMINAMATH_GPT_adam_age_is_8_l1505_150541

variables (A : ℕ) -- Adam's current age
variable (tom_age : ℕ) -- Tom's current age
variable (combined_age : ℕ) -- Their combined age in 12 years

theorem adam_age_is_8 (h1 : tom_age = 12) -- Tom is currently 12 years old
                    (h2 : combined_age = 44) -- In 12 years, their combined age will be 44 years old
                    (h3 : A + 12 + (tom_age + 12) = combined_age) -- Equation representing the combined age in 12 years
                    : A = 8 :=
by
  sorry

end NUMINAMATH_GPT_adam_age_is_8_l1505_150541


namespace NUMINAMATH_GPT_find_m_l1505_150572

-- Definitions from conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1
def major_axis_twice_minor_axis (a b : ℝ) : Prop := a = 2 * b

-- Main statement
theorem find_m (m : ℝ) (h1 : ellipse_eq 0 0 m) (h2 : 0 < m) (h3 : 0 < m ∧ m < 1) :
  m = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1505_150572


namespace NUMINAMATH_GPT_wire_division_l1505_150515

theorem wire_division (L leftover total_length : ℝ) (seg1 seg2 : ℝ)
  (hL : L = 120 * 2)
  (hleftover : leftover = 2.4)
  (htotal : total_length = L + leftover)
  (hseg1 : seg1 = total_length / 3)
  (hseg2 : seg2 = total_length / 3) :
  seg1 = 80.8 ∧ seg2 = 80.8 := by
  sorry

end NUMINAMATH_GPT_wire_division_l1505_150515


namespace NUMINAMATH_GPT_trajectory_equation_l1505_150535

variable (x y a b : ℝ)
variable (P : ℝ × ℝ := (0, -3))
variable (A : ℝ × ℝ := (a, 0))
variable (Q : ℝ × ℝ := (0, b))
variable (M : ℝ × ℝ := (x, y))

theorem trajectory_equation
  (h1 : A.1 = a)
  (h2 : A.2 = 0)
  (h3 : Q.1 = 0)
  (h4 : Q.2 > 0)
  (h5 : (P.1 - A.1) * (x - A.1) + (P.2 - A.2) * y = 0)
  (h6 : (x - A.1, y) = (-3/2 * (-x, b - y))) :
  y = (1 / 4) * x ^ 2 ∧ x ≠ 0 := by
    -- Sorry, proof omitted
    sorry

end NUMINAMATH_GPT_trajectory_equation_l1505_150535


namespace NUMINAMATH_GPT_calculation_l1505_150523

theorem calculation : (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ (2022 : ℤ) = 5 := by
  sorry

end NUMINAMATH_GPT_calculation_l1505_150523


namespace NUMINAMATH_GPT_trajectory_of_point_P_l1505_150514

open Real

theorem trajectory_of_point_P (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (a = 1 → x = 0) ∧ 
    (a ≠ 1 → (x - (a^2 + 1) / (a^2 - 1))^2 + y^2 = 4 * a^2 / (a^2 - 1)^2)) := 
by 
  sorry

end NUMINAMATH_GPT_trajectory_of_point_P_l1505_150514


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1505_150543

-- Define the conditions for the isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c 

-- Define the side lengths
def side_length_1 : ℝ := 15
def side_length_2 : ℝ := 15
def side_length_3 : ℝ := 24

-- State the theorem
theorem isosceles_triangle_area :
  is_isosceles_triangle side_length_1 side_length_2 side_length_3 →
  side_length_1 = 15 →
  side_length_2 = 15 →
  side_length_3 = 24 →
  ∃ A : ℝ, (A = (1 / 2) * 24 * 9) ∧ A = 108 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1505_150543


namespace NUMINAMATH_GPT_longest_segment_in_cylinder_l1505_150565

noncomputable def cylinder_diagonal (radius height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

theorem longest_segment_in_cylinder :
  cylinder_diagonal 4 10 = 2 * Real.sqrt 41 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_longest_segment_in_cylinder_l1505_150565


namespace NUMINAMATH_GPT_percentage_decrease_in_speed_l1505_150531

variable (S : ℝ) (S' : ℝ) (T T' : ℝ)

noncomputable def percentageDecrease (originalSpeed decreasedSpeed : ℝ) : ℝ :=
  ((originalSpeed - decreasedSpeed) / originalSpeed) * 100

theorem percentage_decrease_in_speed :
  T = 40 ∧ T' = 50 ∧ S' = (4 / 5) * S →
  percentageDecrease S S' = 20 :=
by sorry

end NUMINAMATH_GPT_percentage_decrease_in_speed_l1505_150531


namespace NUMINAMATH_GPT_find_people_got_off_at_first_stop_l1505_150509

def total_seats (rows : ℕ) (seats_per_row : ℕ) : ℕ :=
  rows * seats_per_row

def occupied_seats (total_seats : ℕ) (initial_people : ℕ) : ℕ :=
  total_seats - initial_people

def occupied_seats_after_first_stop (initial_people : ℕ) (boarded_first_stop : ℕ) (got_off_first_stop : ℕ) : ℕ :=
  (initial_people + boarded_first_stop) - got_off_first_stop

def occupied_seats_after_second_stop (occupied_after_first_stop : ℕ) (boarded_second_stop : ℕ) (got_off_second_stop : ℕ) : ℕ :=
  (occupied_after_first_stop + boarded_second_stop) - got_off_second_stop

theorem find_people_got_off_at_first_stop
  (initial_people : ℕ := 16)
  (boarded_first_stop : ℕ := 15)
  (total_rows : ℕ := 23)
  (seats_per_row : ℕ := 4)
  (boarded_second_stop : ℕ := 17)
  (got_off_second_stop : ℕ := 10)
  (empty_seats_after_second_stop : ℕ := 57)
  : ∃ x, (occupied_seats_after_second_stop (occupied_seats_after_first_stop initial_people boarded_first_stop x) boarded_second_stop got_off_second_stop) = total_seats total_rows seats_per_row - empty_seats_after_second_stop :=
by
  sorry

end NUMINAMATH_GPT_find_people_got_off_at_first_stop_l1505_150509


namespace NUMINAMATH_GPT_exists_tangent_inequality_l1505_150519

theorem exists_tangent_inequality {x : Fin 8 → ℝ} (h : Function.Injective x) :
  ∃ (i j : Fin 8), i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (Real.pi / 7) :=
by
  sorry

end NUMINAMATH_GPT_exists_tangent_inequality_l1505_150519


namespace NUMINAMATH_GPT_square_field_area_l1505_150526

/-- 
  Statement: Prove that the area of the square field is 69696 square meters 
  given that the wire goes around the square field 15 times and the total 
  length of the wire is 15840 meters.
-/
theorem square_field_area (rounds : ℕ) (total_length : ℕ) (area : ℕ) 
  (h1 : rounds = 15) (h2 : total_length = 15840) : 
  area = 69696 := 
by 
  sorry

end NUMINAMATH_GPT_square_field_area_l1505_150526


namespace NUMINAMATH_GPT_can_cover_101x101_with_102_cells_100_times_l1505_150518

theorem can_cover_101x101_with_102_cells_100_times :
  ∃ f : Fin 100 → Fin 101 → Fin 101 → Bool,
  (∀ i j : Fin 101, (i ≠ 100 ∨ j ≠ 100) → ∃ t : Fin 100, 
    f t i j = true) :=
sorry

end NUMINAMATH_GPT_can_cover_101x101_with_102_cells_100_times_l1505_150518


namespace NUMINAMATH_GPT_product_of_x_y_l1505_150550

theorem product_of_x_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) : x * y = 72 :=
by
  sorry

end NUMINAMATH_GPT_product_of_x_y_l1505_150550


namespace NUMINAMATH_GPT_boxes_containing_neither_l1505_150546

theorem boxes_containing_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_cards : ℕ)
  (boxes_with_both : ℕ)
  (h1 : total_boxes = 15)
  (h2 : boxes_with_stickers = 8)
  (h3 : boxes_with_cards = 5)
  (h4 : boxes_with_both = 3) :
  (total_boxes - (boxes_with_stickers + boxes_with_cards - boxes_with_both)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_boxes_containing_neither_l1505_150546
