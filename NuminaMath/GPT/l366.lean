import Mathlib

namespace exists_finite_set_with_subset_relation_l366_36601

-- Definition of an ordered set (E, ≤)
variable {E : Type} [LE E]

theorem exists_finite_set_with_subset_relation (E : Type) [LE E] :
  ∃ (F : Set (Set E)) (X : E → Set E), 
  (∀ (e1 e2 : E), e1 ≤ e2 ↔ X e2 ⊆ X e1) :=
by
  -- The proof is initially skipped, as per instructions
  sorry

end exists_finite_set_with_subset_relation_l366_36601


namespace students_play_long_tennis_l366_36678

-- Define the parameters for the problem
def total_students : ℕ := 38
def football_players : ℕ := 26
def both_sports_players : ℕ := 17
def neither_sports_players : ℕ := 9

-- Total students playing at least one sport
def at_least_one := total_students - neither_sports_players

-- Define the Lean theorem statement
theorem students_play_long_tennis : at_least_one = football_players + (20 : ℕ) - both_sports_players := 
by 
  -- Translate the given facts into the Lean proof structure
  have h1 : at_least_one = 29 := by rfl -- total_students - neither_sports_players
  have h2 : football_players = 26 := by rfl
  have h3 : both_sports_players = 17 := by rfl
  show 29 = 26 + 20 - 17
  sorry

end students_play_long_tennis_l366_36678


namespace books_per_shelf_l366_36623

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_total_books : total_books = 2250) (h_total_shelves : total_shelves = 150) :
  total_books / total_shelves = 15 :=
by
  sorry

end books_per_shelf_l366_36623


namespace surface_area_small_prism_l366_36658

-- Definitions and conditions
variables (a b c : ℝ)

def small_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * a * b + 2 * a * c + 2 * b * c

def large_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * (3 * b) * (3 * b) + 2 * (3 * b) * (4 * c) + 2 * (4 * c) * (3 * b)

-- Conditions
def conditions : Prop :=
  (3 * b = 2 * a) ∧ (a = 3 * c) ∧ (large_cuboid_surface_area a b c = 360)

-- Desired result
def result : Prop :=
  small_cuboid_surface_area a b c = 88

-- The theorem
theorem surface_area_small_prism (a b c : ℝ) (h : conditions a b c) : result a b c :=
by
  sorry

end surface_area_small_prism_l366_36658


namespace ab_value_l366_36682

theorem ab_value (a b : ℝ) (log_two_3 : ℝ := Real.log 3 / Real.log 2) :
  a * log_two_3 = 1 ∧ (4 : ℝ)^b = 3 → a * b = 1 / 2 := by
  sorry

end ab_value_l366_36682


namespace cost_per_kg_paint_l366_36653

-- Define the basic parameters
variables {sqft_per_kg : ℝ} -- the area covered by 1 kg of paint
variables {total_cost : ℝ} -- the total cost to paint the cube
variables {side_length : ℝ} -- the side length of the cube
variables {num_faces : ℕ} -- the number of faces of the cube

-- Define the conditions given in the problem
def conditions (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) : Prop :=
  sqft_per_kg = 16 ∧
  total_cost = 876 ∧
  side_length = 8 ∧
  num_faces = 6

-- Define the statement to prove, which is the cost per kg of paint
theorem cost_per_kg_paint (sqft_per_kg total_cost side_length : ℝ) (num_faces : ℕ) :
  conditions sqft_per_kg total_cost side_length num_faces →
  ∃ cost_per_kg : ℝ, cost_per_kg = 36.5 :=
by
  sorry

end cost_per_kg_paint_l366_36653


namespace Ryan_funding_goal_l366_36622

theorem Ryan_funding_goal 
  (avg_fund_per_person : ℕ := 10) 
  (people_recruited : ℕ := 80)
  (pre_existing_fund : ℕ := 200) :
  (avg_fund_per_person * people_recruited + pre_existing_fund = 1000) :=
by
  sorry

end Ryan_funding_goal_l366_36622


namespace number_of_children_l366_36673

theorem number_of_children (total_passengers men women : ℕ) (h1 : total_passengers = 54) (h2 : men = 18) (h3 : women = 26) : 
  total_passengers - men - women = 10 :=
by sorry

end number_of_children_l366_36673


namespace num_of_sets_eq_four_l366_36647

open Finset

theorem num_of_sets_eq_four : ∀ B : Finset ℕ, (insert 1 (insert 2 B) = {1, 2, 3, 4, 5}) → (B = {3, 4, 5} ∨ B = {1, 3, 4, 5} ∨ B = {2, 3, 4, 5} ∨ B = {1, 2, 3, 4, 5}) := 
by
  sorry

end num_of_sets_eq_four_l366_36647


namespace arithmetic_sequence_sum_l366_36606

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (h_arith : arithmetic_sequence a)
    (h_a2 : a 2 = 3)
    (h_a1_a6 : a 1 + a 6 = 12) : a 7 + a 8 + a 9 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l366_36606


namespace remainder_of_4n_squared_l366_36650

theorem remainder_of_4n_squared {n : ℤ} (h : n % 13 = 7) : (4 * n^2) % 13 = 1 :=
by
  sorry

end remainder_of_4n_squared_l366_36650


namespace replace_stars_with_identity_l366_36644

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end replace_stars_with_identity_l366_36644


namespace dave_diner_total_cost_l366_36624

theorem dave_diner_total_cost (burger_count : ℕ) (fries_count : ℕ)
  (burger_cost : ℕ) (fries_cost : ℕ)
  (discount_threshold : ℕ) (discount_amount : ℕ)
  (h1 : burger_count >= discount_threshold) :
  burger_count = 6 → fries_count = 5 → burger_cost = 4 → fries_cost = 3 →
  discount_threshold = 4 → discount_amount = 2 →
  (burger_count * (burger_cost - discount_amount) + fries_count * fries_cost) = 27 :=
by
  intros hbc hfc hbcost hfcs dth da
  sorry

end dave_diner_total_cost_l366_36624


namespace value_of_t_l366_36611

theorem value_of_t (t : ℝ) (x y : ℝ) (h : 3 * x^(t-1) + y - 5 = 0) :
  t = 2 :=
sorry

end value_of_t_l366_36611


namespace product_signs_l366_36636

theorem product_signs (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  ( 
    (((-a * b > 0) ∧ (a * c < 0) ∧ (b * d < 0) ∧ (c * d < 0)) ∨ 
    ((-a * b < 0) ∧ (a * c > 0) ∧ (b * d > 0) ∧ (c * d > 0))) ∨
    (((-a * b < 0) ∧ (a * c > 0) ∧ (b * d < 0) ∧ (c * d > 0)) ∨ 
    ((-a * b > 0) ∧ (a * c < 0) ∧ (b * d > 0) ∧ (c * d < 0))) 
  ) := 
sorry

end product_signs_l366_36636


namespace fraction_of_occupied_student_chairs_is_4_over_5_l366_36612

-- Definitions based on the conditions provided
def total_chairs : ℕ := 10 * 15
def awardees_chairs : ℕ := 15
def admin_teachers_chairs : ℕ := 2 * 15
def parents_chairs : ℕ := 2 * 15
def student_chairs : ℕ := total_chairs - (awardees_chairs + admin_teachers_chairs + parents_chairs)
def vacant_student_chairs_given_to_parents : ℕ := 15
def occupied_student_chairs : ℕ := student_chairs - vacant_student_chairs_given_to_parents

-- Theorem statement based on the problem
theorem fraction_of_occupied_student_chairs_is_4_over_5 :
    (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 :=
by
    sorry

end fraction_of_occupied_student_chairs_is_4_over_5_l366_36612


namespace repeating_decimal_subtraction_l366_36672

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end repeating_decimal_subtraction_l366_36672


namespace collinear_points_l366_36603

theorem collinear_points (x y : ℝ) (h_collinear : ∃ k : ℝ, (x + 1, y, 3) = (2 * k, 4 * k, 6 * k)) : x - y = -2 := 
by 
  sorry

end collinear_points_l366_36603


namespace john_work_days_l366_36634

theorem john_work_days (J : ℕ) (H1 : 1 / J + 1 / 480 = 1 / 192) : J = 320 :=
sorry

end john_work_days_l366_36634


namespace num_sets_N_l366_36629

open Set

noncomputable def M : Set ℤ := {-1, 0}

theorem num_sets_N (N : Set ℤ) : M ∪ N = {-1, 0, 1} → 
  (N = {1} ∨ N = {0, 1} ∨ N = {-1, 1} ∨ N = {0, -1, 1}) := 
sorry

end num_sets_N_l366_36629


namespace domain_of_function_l366_36643

theorem domain_of_function (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 1 > 0) ↔ (1 < x ∧ x ≤ 2) :=
by
  sorry

end domain_of_function_l366_36643


namespace growth_rate_of_yield_l366_36656

-- Let x be the growth rate of the average yield per acre
variable (x : ℝ)

-- Initial conditions
def initial_acres := 10
def initial_yield := 20000
def final_yield := 60000

-- Relationship between the growth rates
def growth_relation := x * initial_acres * (1 + 2 * x) * (1 + x) = final_yield / initial_yield

theorem growth_rate_of_yield (h : growth_relation x) : x = 0.5 :=
  sorry

end growth_rate_of_yield_l366_36656


namespace division_by_3_l366_36676

theorem division_by_3 (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := 
sorry

end division_by_3_l366_36676


namespace cut_difference_l366_36698

-- define the conditions
def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

-- theorem to prove the correctness of the difference
theorem cut_difference : (skirt_cut - pants_cut = 0.25) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end cut_difference_l366_36698


namespace unique_solution_l366_36602

theorem unique_solution (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 2) (h₂ : x = z + 2) :
  x = 1 ∧ y = 0 ∧ z = -1 :=
by
  sorry

end unique_solution_l366_36602


namespace medium_sized_fir_trees_count_l366_36694

theorem medium_sized_fir_trees_count 
  (total_trees : ℕ) (ancient_oaks : ℕ) (saplings : ℕ)
  (h1 : total_trees = 96)
  (h2 : ancient_oaks = 15)
  (h3 : saplings = 58) :
  total_trees - ancient_oaks - saplings = 23 :=
by 
  sorry

end medium_sized_fir_trees_count_l366_36694


namespace number_of_triangles_l366_36692

-- Definition of given conditions
def original_wire_length : ℝ := 84
def remaining_wire_length : ℝ := 12
def wire_per_triangle : ℝ := 3

-- The goal is to prove that the number of triangles that can be made is 24
theorem number_of_triangles : (original_wire_length - remaining_wire_length) / wire_per_triangle = 24 := by
  sorry

end number_of_triangles_l366_36692


namespace find_length_of_c_find_measure_of_B_l366_36683

-- Definition of the conditions
def triangle (A B C a b c : ℝ) : Prop :=
  c - b = 2 * b * Real.cos A

noncomputable def value_c (a b : ℝ) : ℝ := sorry

noncomputable def value_B (A B : ℝ) : ℝ := sorry

-- Statement for problem (I)
theorem find_length_of_c (a b : ℝ) (h1 : a = 2 * Real.sqrt 6) (h2 : b = 3) (h3 : ∀ A B C, triangle A B C a b (value_c a b)) : 
  value_c a b = 5 :=
by 
  sorry

-- Statement for problem (II)
theorem find_measure_of_B (B : ℝ) (h1 : ∀ A, A + B = Real.pi / 2) (h2 : B = value_B A B) : 
  value_B A B = Real.pi / 6 :=
by 
  sorry

end find_length_of_c_find_measure_of_B_l366_36683


namespace cubic_sum_identity_l366_36608

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l366_36608


namespace penelope_food_intake_l366_36604

theorem penelope_food_intake
(G P M E : ℕ) -- Representing amount of food each animal eats per day
(h1 : P = 10 * G) -- Penelope eats 10 times Greta's food
(h2 : M = G / 100) -- Milton eats 1/100 of Greta's food
(h3 : E = 4000 * M) -- Elmer eats 4000 times what Milton eats
(h4 : E = P + 60) -- Elmer eats 60 pounds more than Penelope
(G_val : G = 2) -- Greta eats 2 pounds per day
: P = 20 := -- Prove Penelope eats 20 pounds per day
by
  rw [G_val] at h1 -- Replace G with 2 in h1
  norm_num at h1 -- Evaluate the expression in h1
  exact h1 -- Conclude P = 20

end penelope_food_intake_l366_36604


namespace problem_l366_36641

theorem problem (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := by
  sorry

end problem_l366_36641


namespace positive_difference_of_two_numbers_l366_36615

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℝ), x + y = 10 ∧ x^2 - y^2 = 24 ∧ |x - y| = 12 / 5 :=
by
  sorry

end positive_difference_of_two_numbers_l366_36615


namespace rancher_no_cows_l366_36610

theorem rancher_no_cows (s c : ℕ) (h1 : 30 * s + 31 * c = 1200) 
  (h2 : 15 ≤ s) (h3 : s ≤ 35) : c = 0 :=
by
  sorry

end rancher_no_cows_l366_36610


namespace y1_increasing_on_0_1_l366_36691

noncomputable def y1 (x : ℝ) : ℝ := |x|
noncomputable def y2 (x : ℝ) : ℝ := 3 - x
noncomputable def y3 (x : ℝ) : ℝ := 1 / x
noncomputable def y4 (x : ℝ) : ℝ := -x^2 + 4

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem y1_increasing_on_0_1 :
  is_increasing_on y1 0 1 ∧
  ¬ is_increasing_on y2 0 1 ∧
  ¬ is_increasing_on y3 0 1 ∧
  ¬ is_increasing_on y4 0 1 :=
by
  sorry

end y1_increasing_on_0_1_l366_36691


namespace factorization_cd_c_l366_36688

theorem factorization_cd_c (C D : ℤ) (h : ∀ y : ℤ, 20*y^2 - 117*y + 72 = (C*y - 8) * (D*y - 9)) : C * D + C = 25 :=
sorry

end factorization_cd_c_l366_36688


namespace find_a_for_even_function_l366_36625

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l366_36625


namespace angle_B_measure_l366_36609

theorem angle_B_measure (a b : ℝ) (A B : ℝ) (h₁ : a = 4) (h₂ : b = 4 * Real.sqrt 3) (h₃ : A = Real.pi / 6) : 
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end angle_B_measure_l366_36609


namespace height_of_Linda_room_l366_36657

theorem height_of_Linda_room (w l: ℝ) (h a1 a2 a3 paint_area: ℝ) 
  (hw: w = 20) (hl: l = 20) 
  (d1_h: a1 = 3) (d1_w: a2 = 7) 
  (d2_h: a3 = 4) (d2_w: a4 = 6) 
  (d3_h: a5 = 5) (d3_w: a6 = 7) 
  (total_paint_area: paint_area = 560):
  h = 6 := 
by
  sorry

end height_of_Linda_room_l366_36657


namespace max_area_of_right_triangle_with_hypotenuse_4_l366_36687

theorem max_area_of_right_triangle_with_hypotenuse_4 : 
  (∀ (a b : ℝ), a^2 + b^2 = 16 → (∃ S, S = 1/2 * a * b ∧ S ≤ 4)) ∧ 
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ a = b ∧ 1/2 * a * b = 4) :=
by
  sorry

end max_area_of_right_triangle_with_hypotenuse_4_l366_36687


namespace neg_three_is_square_mod_p_l366_36655

theorem neg_three_is_square_mod_p (q : ℤ) (p : ℕ) (prime_p : Nat.Prime p) (condition : p = 3 * q + 1) :
  ∃ x : ℤ, (x^2 ≡ -3 [ZMOD p]) :=
sorry

end neg_three_is_square_mod_p_l366_36655


namespace customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l366_36679

theorem customer_B_cost_effectiveness (box_orig_cost box_spec_cost : ℕ) (orig_price spec_price eggs_per_box remaining_eggs : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : remaining_eggs = 20) : 
    ¬ (spec_price * 2 / (eggs_per_box * 2 - remaining_eggs) < orig_price / eggs_per_box) :=
by
  sorry

theorem customer_A_boxes_and_consumption (orig_price spec_price eggs_per_box total_cost_savings : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : total_cost_savings = 90): 
  ∃ (boxes_bought : ℕ) (avg_daily_consumption : ℕ), 
    (spec_price * boxes_bought = orig_price * boxes_bought * 2 - total_cost_savings) ∧ 
    (avg_daily_consumption = eggs_per_box * boxes_bought / 15) :=
by
  sorry

end customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l366_36679


namespace max_sum_of_multiplication_table_l366_36675

-- Define primes and their sums
def primes : List ℕ := [2, 3, 5, 7, 17, 19]

noncomputable def sum_primes := primes.sum -- 2 + 3 + 5 + 7 + 17 + 19 = 53

-- Define two groups of primes to maximize the product of their sums
def group1 : List ℕ := [2, 3, 17]
def group2 : List ℕ := [5, 7, 19]

noncomputable def sum_group1 := group1.sum -- 2 + 3 + 17 = 22
noncomputable def sum_group2 := group2.sum -- 5 + 7 + 19 = 31

-- Formulate the proof problem
theorem max_sum_of_multiplication_table : 
  ∃ a b c d e f : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
    (a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes) ∧ 
    (a + b + c = sum_group1 ∨ a + b + c = sum_group2) ∧ 
    (d + e + f = sum_group1 ∨ d + e + f = sum_group2) ∧ 
    (a + b + c) ≠ (d + e + f) ∧ 
    ((a + b + c) * (d + e + f) = 682) := 
by
  use 2, 3, 17, 5, 7, 19
  sorry

end max_sum_of_multiplication_table_l366_36675


namespace calculate_f_f_neg3_l366_36626

def f (x : ℚ) : ℚ := (1 / x) + (1 / (x + 1))

theorem calculate_f_f_neg3 : f (f (-3)) = 24 / 5 := by
  sorry

end calculate_f_f_neg3_l366_36626


namespace power_of_i_2016_l366_36648
-- Importing necessary libraries to handle complex numbers

theorem power_of_i_2016 (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : 
  (i^2016 = 1) :=
sorry

end power_of_i_2016_l366_36648


namespace marbles_leftover_l366_36654

theorem marbles_leftover (r p g : ℕ) (hr : r % 7 = 5) (hp : p % 7 = 4) (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 :=
by
  sorry

end marbles_leftover_l366_36654


namespace nell_initial_ace_cards_l366_36633

def initial_ace_cards (initial_baseball_cards final_ace_cards final_baseball_cards given_difference : ℕ) : ℕ :=
  final_ace_cards + (initial_baseball_cards - final_baseball_cards)

theorem nell_initial_ace_cards : 
  initial_ace_cards 239 376 111 265 = 504 :=
by
  /- This is to show that the initial count of Ace cards Nell had is 504 given the conditions -/
  sorry

end nell_initial_ace_cards_l366_36633


namespace proof_problem_l366_36686

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem proof_problem 
  (h0 : f a b c 0 = f a b c 4)
  (h1 : f a b c 0 > f a b c 1) : 
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end proof_problem_l366_36686


namespace solution_set_of_inequality_l366_36664

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 1) * (2 - x) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l366_36664


namespace sum_of_intercepts_l366_36681

theorem sum_of_intercepts (x y : ℝ) 
  (h_eq : y - 3 = -3 * (x - 5)) 
  (hx_intercept : y = 0 ∧ x = 6) 
  (hy_intercept : x = 0 ∧ y = 18) : 
  6 + 18 = 24 :=
by
  sorry

end sum_of_intercepts_l366_36681


namespace gnome_voting_l366_36663

theorem gnome_voting (n : ℕ) :
  (∀ g : ℕ, g < n →  
   (g % 3 = 0 → (∃ k : ℕ, k * 4 = n))
   ∧ (n ≠ 0 ∧ (∀ i : ℕ, i < n → (i + 1) % n ≠ (i + 2) % n) → (∃ k : ℕ, k * 4 = n))) := 
sorry

end gnome_voting_l366_36663


namespace graph_of_equation_is_two_intersecting_lines_l366_36689

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x - 2 * y)^2 = x^2 + y^2 ↔ (y = 0 ∨ y = 4 / 3 * x) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l366_36689


namespace number_division_l366_36652

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l366_36652


namespace cost_of_each_soccer_ball_l366_36628

theorem cost_of_each_soccer_ball (total_amount_paid : ℕ) (change_received : ℕ) (number_of_balls : ℕ)
  (amount_spent := total_amount_paid - change_received)
  (unit_price := amount_spent / number_of_balls) :
  total_amount_paid = 100 →
  change_received = 20 →
  number_of_balls = 2 →
  unit_price = 40 := by
  sorry

end cost_of_each_soccer_ball_l366_36628


namespace fraction_remains_unchanged_l366_36674

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (3 * x)) / (2 * (3 * x) - 3 * y) = (3 * x) / (2 * x - y) :=
by
  sorry

end fraction_remains_unchanged_l366_36674


namespace expression_takes_many_different_values_l366_36666

theorem expression_takes_many_different_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) : 
  ∃ v : ℝ, ∀ x, x ≠ 3 → x ≠ -2 → v = (3*x^2 - 2*x + 3)/((x - 3)*(x + 2)) - (5*x - 6)/((x - 3)*(x + 2)) := 
sorry

end expression_takes_many_different_values_l366_36666


namespace candies_per_house_l366_36693

theorem candies_per_house (candies_per_block : ℕ) (houses_per_block : ℕ) 
  (h1 : candies_per_block = 35) (h2 : houses_per_block = 5) :
  candies_per_block / houses_per_block = 7 := by
  sorry

end candies_per_house_l366_36693


namespace sequence_sum_is_100_then_n_is_10_l366_36607

theorem sequence_sum_is_100_then_n_is_10 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * a 1 + n * (n - 1)) →
  (∃ n, S n = 100) → 
  n = 10 :=
by sorry

end sequence_sum_is_100_then_n_is_10_l366_36607


namespace pool_width_l366_36630

variable (length : ℝ) (depth : ℝ) (chlorine_per_cubic_foot : ℝ) (chlorine_cost_per_quart : ℝ) (total_spent : ℝ)
variable (w : ℝ)

-- defining the conditions
def pool_conditions := length = 10 ∧ depth = 6 ∧ chlorine_per_cubic_foot = 120 ∧ chlorine_cost_per_quart = 3 ∧ total_spent = 12

-- goal statement
theorem pool_width : pool_conditions length depth chlorine_per_cubic_foot chlorine_cost_per_quart total_spent →
  w = 8 :=
by
  sorry

end pool_width_l366_36630


namespace no_solutions_in_natural_numbers_l366_36680

theorem no_solutions_in_natural_numbers (x y : ℕ) : x^2 + x * y + y^2 ≠ x^2 * y^2 :=
  sorry

end no_solutions_in_natural_numbers_l366_36680


namespace find_coordinates_l366_36651

def pointA : ℝ × ℝ := (2, -4)
def pointB : ℝ × ℝ := (0, 6)
def pointC : ℝ × ℝ := (-8, 10)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_coordinates :
  scalar_mult (1/2) (vector pointA pointC) - 
  scalar_mult (1/4) (vector pointB pointC) = (-3, 6) :=
by
  sorry

end find_coordinates_l366_36651


namespace points_calculation_l366_36646

def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_destroyed : ℕ := total_enemies - 3
def total_points_earned : ℕ := enemies_destroyed * points_per_enemy

theorem points_calculation :
  total_points_earned = 72 := by
  sorry

end points_calculation_l366_36646


namespace product_of_935421_and_625_l366_36632

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 :=
by
  sorry

end product_of_935421_and_625_l366_36632


namespace smallest_divisor_sum_of_squares_of_1_to_7_l366_36684

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_divisor_sum_of_squares_of_1_to_7 (S : ℕ) (h : S = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) :
  ∃ m, is_divisor m S ∧ (∀ d, is_divisor d S → 2 ≤ d) :=
sorry

end smallest_divisor_sum_of_squares_of_1_to_7_l366_36684


namespace highest_place_joker_can_achieve_is_6_l366_36690

-- Define the total number of teams
def total_teams : ℕ := 16

-- Define conditions for points in football
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- Condition definitions for Joker's performance in the tournament
def won_against_strong_teams (j k : ℕ) : Prop := j < k
def lost_against_weak_teams (j k : ℕ) : Prop := j > k

-- Define the performance of all teams
def teams (t : ℕ) := {n // n < total_teams}

-- Function to calculate Joker's points based on position k
def joker_points (k : ℕ) : ℕ := (total_teams - k) * points_win

theorem highest_place_joker_can_achieve_is_6 : ∃ k, k = 6 ∧ 
  (∀ j, 
    (j < k → won_against_strong_teams j k) ∧ 
    (j > k → lost_against_weak_teams j k) ∧
    (∃! p, p = joker_points k)) :=
by
  sorry

end highest_place_joker_can_achieve_is_6_l366_36690


namespace fifth_number_selected_l366_36638

-- Define the necessary conditions
def num_students : ℕ := 60
def sample_size : ℕ := 5
def first_selected_number : ℕ := 4
def interval : ℕ := num_students / sample_size

-- Define the proposition to be proved
theorem fifth_number_selected (h1 : 1 ≤ first_selected_number) (h2 : first_selected_number ≤ num_students)
    (h3 : sample_size > 0) (h4 : num_students % sample_size = 0) :
  first_selected_number + 4 * interval = 52 :=
by
  -- Proof omitted
  sorry

end fifth_number_selected_l366_36638


namespace lowest_dropped_score_l366_36696

theorem lowest_dropped_score (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 90)
  (h2 : (A + B + C) / 3 = 85) :
  D = 105 :=
by
  sorry

end lowest_dropped_score_l366_36696


namespace log_comparison_l366_36637

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := 
sorry

end log_comparison_l366_36637


namespace smallest_x_l366_36670

theorem smallest_x (x : ℕ) : (x + 3457) % 15 = 1537 % 15 → x = 15 :=
by
  sorry

end smallest_x_l366_36670


namespace star_evaluation_l366_36605

noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : (star (star 2 3) 4) = 1 / 9 := 
by sorry

end star_evaluation_l366_36605


namespace tallest_is_jie_l366_36685

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end tallest_is_jie_l366_36685


namespace total_muffins_l366_36640

-- Define initial conditions
def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

-- Define the main theorem we want to prove
theorem total_muffins : initial_muffins + additional_muffins = 83 :=
by
  sorry

end total_muffins_l366_36640


namespace decrease_of_negative_five_l366_36620

-- Definition: Positive and negative numbers as explained
def increase (n: ℤ) : Prop := n > 0
def decrease (n: ℤ) : Prop := n < 0

-- Conditions
def condition : Prop := increase 17

-- Theorem stating the solution
theorem decrease_of_negative_five (h : condition) : decrease (-5) ∧ -5 = -5 :=
by
  sorry

end decrease_of_negative_five_l366_36620


namespace no_rational_solution_l366_36613

theorem no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
by sorry

end no_rational_solution_l366_36613


namespace hyperbola_eccentricity_l366_36600

theorem hyperbola_eccentricity (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 1) (h_eccentricity : ∀ e : ℝ, e = 2) :
    k = -1 / 3 := 
sorry

end hyperbola_eccentricity_l366_36600


namespace Sophie_l366_36667

-- Define the prices of each item
def price_cupcake : ℕ := 2
def price_doughnut : ℕ := 1
def price_apple_pie : ℕ := 2
def price_cookie : ℚ := 0.60

-- Define the quantities of each item
def qty_cupcake : ℕ := 5
def qty_doughnut : ℕ := 6
def qty_apple_pie : ℕ := 4
def qty_cookie : ℕ := 15

-- Define the total cost function for each item
def cost_cupcake := qty_cupcake * price_cupcake
def cost_doughnut := qty_doughnut * price_doughnut
def cost_apple_pie := qty_apple_pie * price_apple_pie
def cost_cookie := qty_cookie * price_cookie

-- Define total expenditure
def total_expenditure := cost_cupcake + cost_doughnut + cost_apple_pie + cost_cookie

-- Assertion of total expenditure
theorem Sophie's_total_expenditure : total_expenditure = 33 := by
  -- skipping proof
  sorry

end Sophie_l366_36667


namespace not_set_of_difficult_problems_l366_36677

-- Define the context and entities
inductive Exercise
| ex (n : Nat) : Exercise  -- Example definition for exercises, assumed to be numbered

def is_difficult (ex : Exercise) : Prop := sorry  -- Placeholder for the subjective predicate

-- Define the main problem statement
theorem not_set_of_difficult_problems
  (Difficult : Exercise → Prop) -- Subjective predicate defining difficult problems
  (H_subj : ∀ (e : Exercise), (Difficult e ↔ is_difficult e)) :
  ¬(∃ (S : Set Exercise), ∀ e, e ∈ S ↔ Difficult e) :=
sorry

end not_set_of_difficult_problems_l366_36677


namespace range_of_squared_function_l366_36616

theorem range_of_squared_function (x : ℝ) (hx : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end range_of_squared_function_l366_36616


namespace cherries_on_June_5_l366_36695

theorem cherries_on_June_5 : 
  ∃ c : ℕ, (c + (c + 8) + (c + 16) + (c + 24) + (c + 32) = 130) ∧ (c + 32 = 42) :=
by
  sorry

end cherries_on_June_5_l366_36695


namespace moon_speed_conversion_l366_36627

def moon_speed_km_sec : ℝ := 1.04
def seconds_per_hour : ℝ := 3600

theorem moon_speed_conversion :
  (moon_speed_km_sec * seconds_per_hour) = 3744 := by
  sorry

end moon_speed_conversion_l366_36627


namespace correct_choice_l366_36635

theorem correct_choice (a : ℝ) : -(-a)^2 * a^4 = -a^6 := 
sorry

end correct_choice_l366_36635


namespace probability_sector_F_l366_36659

theorem probability_sector_F (prob_D prob_E prob_F : ℚ)
    (hD : prob_D = 1/4) 
    (hE : prob_E = 1/3) 
    (hSum : prob_D + prob_E + prob_F = 1) :
    prob_F = 5/12 := by
  sorry

end probability_sector_F_l366_36659


namespace zero_in_interval_l366_36662

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 2 * x^2 - 4 * x

theorem zero_in_interval : ∃ (c : ℝ), 1 < c ∧ c < Real.exp 1 ∧ f c = 0 := sorry

end zero_in_interval_l366_36662


namespace probability_of_all_co_captains_l366_36669

def team_sizes : List ℕ := [6, 8, 9, 10]

def captains_per_team : ℕ := 3

noncomputable def probability_all_co_captains (s : ℕ) : ℚ :=
  1 / (Nat.choose s 3 : ℚ)

noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * 
  (probability_all_co_captains 6 + 
   probability_all_co_captains 8 +
   probability_all_co_captains 9 +
   probability_all_co_captains 10)

theorem probability_of_all_co_captains : total_probability = 1 / 84 :=
  sorry

end probability_of_all_co_captains_l366_36669


namespace average_shift_l366_36639

variable (a b c : ℝ)

-- Given condition: The average of the data \(a\), \(b\), \(c\) is 5.
def average_is_five := (a + b + c) / 3 = 5

-- Define the statement to prove: The average of the data \(a-2\), \(b-2\), \(c-2\) is 3.
theorem average_shift (h : average_is_five a b c) : ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 :=
by
  sorry

end average_shift_l366_36639


namespace base6_addition_problem_l366_36631

-- Definitions to capture the base-6 addition problem components.
def base6₀ := 0
def base6₁ := 1
def base6₂ := 2
def base6₃ := 3
def base6₄ := 4
def base6₅ := 5

-- The main hypothesis about the base-6 addition
theorem base6_addition_problem (diamond : ℕ) (h : diamond ∈ [base6₀, base6₁, base6₂, base6₃, base6₄, base6₅]) :
  ((diamond + base6₅) % 6 = base6₃ ∨ (diamond + base6₅) % 6 = (base6₃ + 6 * 1 % 6)) ∧
  (diamond + base6₂ + base6₂ = diamond % 6) →
  diamond = base6₄ :=
sorry

end base6_addition_problem_l366_36631


namespace circus_tent_sections_l366_36642

noncomputable def sections_in_circus_tent (total_capacity : ℕ) (section_capacity : ℕ) : ℕ :=
  total_capacity / section_capacity

theorem circus_tent_sections : sections_in_circus_tent 984 246 = 4 := 
  by 
  sorry

end circus_tent_sections_l366_36642


namespace coordinates_of_P_with_respect_to_y_axis_l366_36665

-- Define the coordinates of point P
def P_x : ℝ := 5
def P_y : ℝ := -1

-- Define the point P
def P : Prod ℝ ℝ := (P_x, P_y)

-- State the theorem
theorem coordinates_of_P_with_respect_to_y_axis :
  (P.1, P.2) = (-P_x, P_y) :=
sorry

end coordinates_of_P_with_respect_to_y_axis_l366_36665


namespace find_rs_l366_36618

theorem find_rs (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 1) (h4 : r^4 + s^4 = 7/8) : 
  r * s = 1/4 :=
sorry

end find_rs_l366_36618


namespace ones_digit_of_7_pow_53_l366_36621

theorem ones_digit_of_7_pow_53 : (7^53 % 10) = 7 := by
  sorry

end ones_digit_of_7_pow_53_l366_36621


namespace geo_prog_sum_463_l366_36617

/-- Given a set of natural numbers forming an increasing geometric progression with an integer
common ratio where the sum equals 463, prove that these numbers must be {463}, {1, 462}, or {1, 21, 441}. -/
theorem geo_prog_sum_463 (n : ℕ) (b₁ q : ℕ) (s : Finset ℕ) (hgeo : ∀ i j, i < j → s.toList.get? i = some (b₁ * q^i) ∧ s.toList.get? j = some (b₁ * q^j))
  (hsum : s.sum id = 463) : 
  s = {463} ∨ s = {1, 462} ∨ s = {1, 21, 441} :=
sorry

end geo_prog_sum_463_l366_36617


namespace probability_at_least_two_red_balls_l366_36619

noncomputable def prob_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) : ℚ :=
if total_balls = 6 ∧ red_balls = 3 ∧ white_balls = 2 ∧ black_balls = 1 ∧ drawn_balls = 3 then
  1 / 2
else
  0

theorem probability_at_least_two_red_balls :
  prob_red_balls 6 3 2 1 3 = 1 / 2 :=
by 
  sorry

end probability_at_least_two_red_balls_l366_36619


namespace depletion_rate_l366_36671

theorem depletion_rate (initial_value final_value : ℝ) (years: ℕ) (r : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2256.25)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) :
  r = 0.05 :=
by
  sorry

end depletion_rate_l366_36671


namespace find_number_l366_36649

theorem find_number
    (x: ℝ)
    (h: 0.60 * x = 0.40 * 30 + 18) : x = 50 :=
    sorry

end find_number_l366_36649


namespace yo_yos_collected_l366_36645

-- Define the given conditions
def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def total_prizes : ℕ := 50

-- Define the problem to prove that the number of yo-yos is 18
theorem yo_yos_collected : (total_prizes - (stuffed_animals + frisbees) = 18) :=
by
  sorry

end yo_yos_collected_l366_36645


namespace license_plate_combinations_l366_36668

open Nat

theorem license_plate_combinations : 
  (∃ (choose_two_letters: ℕ) (place_first_letter: ℕ) (place_second_letter: ℕ) (choose_non_repeated: ℕ)
     (first_digit: ℕ) (second_digit: ℕ) (third_digit: ℕ),
    choose_two_letters = choose 26 2 ∧
    place_first_letter = choose 5 2 ∧
    place_second_letter = choose 3 2 ∧
    choose_non_repeated = 24 ∧
    first_digit = 10 ∧
    second_digit = 9 ∧
    third_digit = 8 ∧
    choose_two_letters * place_first_letter * place_second_letter * choose_non_repeated * first_digit * second_digit * third_digit = 56016000) :=
sorry

end license_plate_combinations_l366_36668


namespace unique_x0_implies_a_in_range_l366_36614

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x * (3 * x - 1) - a * x + a

theorem unique_x0_implies_a_in_range :
  ∃ x0 : ℤ, f x0 a ≤ 0 ∧ a < 1 -> a ∈ Set.Ico (2 / Real.exp 1) 1 := 
sorry

end unique_x0_implies_a_in_range_l366_36614


namespace problem_solution_l366_36660

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l366_36660


namespace hyperbola_s_eq_l366_36699

theorem hyperbola_s_eq (s : ℝ) 
  (hyp1 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (5, -3) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp2 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (3, 0) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp3 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (s, -1) → x^2 / 9 - y^2 / b^2 = 1) :
  s^2 = 873 / 81 :=
sorry

end hyperbola_s_eq_l366_36699


namespace arithmetic_sequence_sum_l366_36661

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 3 = 4)
  (h3 : ∀ n, a (n + 1) - a n = a 2 - a 1) :
  a 4 + a 5 = 17 :=
  sorry

end arithmetic_sequence_sum_l366_36661


namespace intersection_complement_l366_36697

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set M
def M : Set ℕ := {0, 3, 5}

-- Define set N
def N : Set ℕ := {1, 4, 5}

-- Define the complement of N in U
def complement_U_N : Set ℕ := U \ N

-- The main theorem statement
theorem intersection_complement : M ∩ complement_U_N = {0, 3} :=
by
  -- The proof would go here
  sorry

end intersection_complement_l366_36697
