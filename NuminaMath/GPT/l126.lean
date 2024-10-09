import Mathlib

namespace arithmetic_sequence_inequality_l126_12658

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) :
  a 2 * a 4 ≤ a 3 ^ 2 :=
sorry

end arithmetic_sequence_inequality_l126_12658


namespace smallest_x_for_perfect_cube_l126_12690

theorem smallest_x_for_perfect_cube (M : ℤ) :
  ∃ x : ℕ, 1680 * x = M^3 ∧ ∀ y : ℕ, 1680 * y = M^3 → 44100 ≤ y := 
sorry

end smallest_x_for_perfect_cube_l126_12690


namespace quadratic_is_perfect_square_l126_12647

theorem quadratic_is_perfect_square (c : ℝ) :
  (∃ b : ℝ, (3 * (x : ℝ) + b)^2 = 9 * x^2 - 24 * x + c) ↔ c = 16 :=
by sorry

end quadratic_is_perfect_square_l126_12647


namespace arithmetic_identity_l126_12697

theorem arithmetic_identity : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end arithmetic_identity_l126_12697


namespace physics_class_size_l126_12694

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 53)
  (h2 : both = 7)
  (h3 : physics_only = 2 * (math_only + both))
  (h4 : total_students = physics_only + math_only + both) :
  physics_only + both = 40 :=
by
  sorry

end physics_class_size_l126_12694


namespace exists_integers_A_B_C_l126_12654

theorem exists_integers_A_B_C (a b : ℚ) (N_star : Set ℕ) (Q : Set ℚ)
  (h : ∀ x ∈ N_star, (a * (x : ℚ) + b) / (x : ℚ) ∈ Q) : 
  ∃ A B C : ℤ, ∀ x ∈ N_star, 
    (a * (x : ℚ) + b) / (x : ℚ) = (A * (x : ℚ) + B) / (C * (x : ℚ)) := 
sorry

end exists_integers_A_B_C_l126_12654


namespace victoria_donuts_cost_l126_12667

theorem victoria_donuts_cost (n : ℕ) (cost_per_dozen : ℝ) (total_donuts_needed : ℕ) 
  (dozens_needed : ℕ) (actual_total_donuts : ℕ) (total_cost : ℝ) :
  total_donuts_needed ≥ 550 ∧ cost_per_dozen = 7.49 ∧ (total_donuts_needed = 12 * dozens_needed) ∧
  (dozens_needed = Nat.ceil (total_donuts_needed / 12)) ∧ 
  (actual_total_donuts = 12 * dozens_needed) ∧ actual_total_donuts ≥ 550 ∧ 
  (total_cost = dozens_needed * cost_per_dozen) →
  total_cost = 344.54 :=
by
  sorry

end victoria_donuts_cost_l126_12667


namespace cos_seven_pi_over_six_l126_12688

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end cos_seven_pi_over_six_l126_12688


namespace marbles_difference_l126_12645

-- Conditions
def L : ℕ := 23
def F : ℕ := 9

-- Proof statement
theorem marbles_difference : L - F = 14 := by
  sorry

end marbles_difference_l126_12645


namespace samia_walking_distance_l126_12600

noncomputable def total_distance (x : ℝ) : ℝ := 4 * x
noncomputable def biking_distance (x : ℝ) : ℝ := 3 * x
noncomputable def walking_distance (x : ℝ) : ℝ := x
noncomputable def biking_time (x : ℝ) : ℝ := biking_distance x / 12
noncomputable def walking_time (x : ℝ) : ℝ := walking_distance x / 4
noncomputable def total_time (x : ℝ) : ℝ := biking_time x + walking_time x

theorem samia_walking_distance : ∀ (x : ℝ), total_time x = 1 → walking_distance x = 2 :=
by
  sorry

end samia_walking_distance_l126_12600


namespace space_shuttle_new_orbital_speed_l126_12612

noncomputable def new_orbital_speed (v_1 : ℝ) (delta_v : ℝ) : ℝ :=
  let v_new := v_1 + delta_v
  v_new * 3600

theorem space_shuttle_new_orbital_speed : 
  new_orbital_speed 2 (500 / 1000) = 9000 :=
by 
  sorry

end space_shuttle_new_orbital_speed_l126_12612


namespace lizzie_garbage_l126_12662

/-- Let G be the amount of garbage Lizzie's group collected. 
We are given that the second group collected G - 39 pounds of garbage,
and the total amount collected by both groups is 735 pounds.
We need to prove that G is 387 pounds. -/
theorem lizzie_garbage (G : ℕ) (h1 : G + (G - 39) = 735) : G = 387 :=
sorry

end lizzie_garbage_l126_12662


namespace Julie_and_Matt_ate_cookies_l126_12623

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem Julie_and_Matt_ate_cookies : initial_cookies - remaining_cookies = 9 :=
by
  sorry

end Julie_and_Matt_ate_cookies_l126_12623


namespace markup_is_correct_l126_12624

-- The mathematical interpretation of the given conditions
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.05
def net_profit : ℝ := 12

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the total cost calculation
def total_cost : ℝ := purchase_price + overhead_cost

-- Define the selling price calculation
def selling_price : ℝ := total_cost + net_profit

-- Define the markup calculation
def markup : ℝ := selling_price - purchase_price

-- The statement we want to prove
theorem markup_is_correct : markup = 14.40 :=
by
  -- We will eventually prove this, but for now we use sorry as a placeholder
  sorry

end markup_is_correct_l126_12624


namespace inequality_three_integer_solutions_l126_12644

theorem inequality_three_integer_solutions (c : ℤ) :
  (∃ s1 s2 s3 : ℤ, s1 < s2 ∧ s2 < s3 ∧ 
    (∀ x : ℤ, x^2 + c * x + 1 ≤ 0 ↔ x = s1 ∨ x = s2 ∨ x = s3)) ↔ (c = -4 ∨ c = 4) := 
by 
  sorry

end inequality_three_integer_solutions_l126_12644


namespace decision_making_system_reliability_l126_12687

theorem decision_making_system_reliability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (10 * p^3 - 15 * p^4 + 6 * p^5 > 3 * p^2 - 2 * p^3) -> (1 / 2 < p) ∧ (p < 1) :=
by
  sorry

end decision_making_system_reliability_l126_12687


namespace greatest_difference_is_124_l126_12672

-- Define the variables a, b, c, and x
variables (a b c x : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) := 
  (4 * a = 2 * b) ∧ 
  (4 * a = c) ∧ 
  (a > 0) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧ 
  (c < 10)

-- Define the value of a number given its digits
def number (a b c : ℕ) := 100 * a + 10 * b + c

-- Define the maximum and minimum values of x
def max_val (a : ℕ) := number a (2 * a) (4 * a)
def min_val (a : ℕ) := number a (2 * a) (4 * a)

-- Define the greatest difference
def greatest_difference := max_val 2 - min_val 1

-- Prove that the greatest difference is 124
theorem greatest_difference_is_124 : greatest_difference = 124 :=
by 
  unfold greatest_difference 
  unfold max_val 
  unfold min_val 
  unfold number 
  sorry

end greatest_difference_is_124_l126_12672


namespace interview_room_count_l126_12691

-- Define the number of people in the waiting room
def people_in_waiting_room : ℕ := 22

-- Define the increase in number of people
def extra_people_arrive : ℕ := 3

-- Define the total number of people after more people arrive
def total_people_after_arrival : ℕ := people_in_waiting_room + extra_people_arrive

-- Define the relationship between people in waiting room and interview room
def relation (x : ℕ) : Prop := total_people_after_arrival = 5 * x

theorem interview_room_count : ∃ x : ℕ, relation x ∧ x = 5 :=
by
  -- The proof will be provided here
  sorry

end interview_room_count_l126_12691


namespace harold_savings_l126_12616

theorem harold_savings :
  let income_primary := 2500
  let income_freelance := 500
  let rent := 700
  let car_payment := 300
  let car_insurance := 125
  let electricity := 0.25 * car_payment
  let water := 0.15 * rent
  let internet := 75
  let groceries := 200
  let miscellaneous := 150
  let total_income := income_primary + income_freelance
  let total_expenses := rent + car_payment + car_insurance + electricity + water + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let retirement := (1/3) * amount_before_savings
  let emergency := (1/3) * amount_before_savings
  let amount_after_savings := amount_before_savings - retirement - emergency
  amount_after_savings = 423.34 := 
sorry

end harold_savings_l126_12616


namespace zongzi_cost_prices_l126_12618

theorem zongzi_cost_prices (a : ℕ) (n : ℕ)
  (h1 : n * a = 8000)
  (h2 : n * (a - 10) = 6000)
  : a = 40 ∧ a - 10 = 30 :=
by
  sorry

end zongzi_cost_prices_l126_12618


namespace minimum_weighings_for_counterfeit_coin_l126_12641

/-- Given 9 coins, where 8 have equal weight and 1 is heavier (the counterfeit coin), prove that the 
minimum number of weighings required on a balance scale without weights to find the counterfeit coin is 2. -/
theorem minimum_weighings_for_counterfeit_coin (n : ℕ) (coins : Fin n → ℝ) 
  (h_n : n = 9) 
  (h_real : ∃ w : ℝ, ∀ i : Fin n, i.val < 8 → coins i = w) 
  (h_counterfeit : ∃ i : Fin n, ∀ j : Fin n, j ≠ i → coins i > coins j) : 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end minimum_weighings_for_counterfeit_coin_l126_12641


namespace total_cups_for_8_batches_l126_12665

def cups_of_flour (batches : ℕ) : ℝ := 4 * batches
def cups_of_sugar (batches : ℕ) : ℝ := 1.5 * batches
def total_cups (batches : ℕ) : ℝ := cups_of_flour batches + cups_of_sugar batches

theorem total_cups_for_8_batches : total_cups 8 = 44 := 
by
  -- This is where the proof would go
  sorry

end total_cups_for_8_batches_l126_12665


namespace find_other_root_l126_12621

-- Definitions based on conditions
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := x^2 + 2 * k * x + k - 1 = 0

def is_root (k : ℝ) (x : ℝ) : Prop := quadratic_equation k x = true

-- The theorem to prove
theorem find_other_root (k x t: ℝ) (h₁ : is_root k 0) : t = -2 :=
sorry

end find_other_root_l126_12621


namespace log_sum_l126_12614

theorem log_sum : 2 * Real.log 2 + Real.log 25 = 2 := 
by 
  sorry

end log_sum_l126_12614


namespace probability_sum_greater_than_six_l126_12603

variable (A : Finset ℕ) (B : Finset ℕ)
variable (balls_in_A : A = {1, 2}) (balls_in_B : B = {3, 4, 5, 6})

theorem probability_sum_greater_than_six : 
  (∃ selected_pair ∈ (A.product B), selected_pair.1 + selected_pair.2 > 6) →
  (Finset.filter (λ pair => pair.1 + pair.2 > 6) (A.product B)).card / 
  (A.product B).card = 3 / 8 := sorry

end probability_sum_greater_than_six_l126_12603


namespace distance_ratio_l126_12648

variables (KD DM : ℝ)

theorem distance_ratio : 
  KD = 4 ∧ (KD + DM + DM + KD = 12) → (KD / DM = 2) := 
by
  sorry

end distance_ratio_l126_12648


namespace r_minus_s_l126_12685

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l126_12685


namespace students_minus_rabbits_l126_12609

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the number of classrooms
def number_of_classrooms : ℕ := 5

-- Define the total number of students and rabbits
def total_students : ℕ := students_per_classroom * number_of_classrooms
def total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms

-- The main statement to prove
theorem students_minus_rabbits :
  total_students - total_rabbits = 105 :=
by
  sorry

end students_minus_rabbits_l126_12609


namespace abs_add_conditions_l126_12686

theorem abs_add_conditions (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  a + b = 1 ∨ a + b = 7 :=
by
  sorry

end abs_add_conditions_l126_12686


namespace number_of_sports_books_l126_12627

def total_books : ℕ := 58
def school_books : ℕ := 19
def sports_books (total_books school_books : ℕ) : ℕ := total_books - school_books

theorem number_of_sports_books : sports_books total_books school_books = 39 := by
  -- proof goes here
  sorry

end number_of_sports_books_l126_12627


namespace volume_truncated_cone_l126_12699

/-- 
Given a truncated right circular cone with a large base radius of 10 cm,
a smaller base radius of 3 cm, and a height of 9 cm, 
prove that the volume of the truncated cone is 417 π cubic centimeters.
-/
theorem volume_truncated_cone :
  let R := 10
  let r := 3
  let h := 9
  let V := (1/3) * Real.pi * h * (R^2 + R*r + r^2)
  V = 417 * Real.pi :=
by 
  sorry

end volume_truncated_cone_l126_12699


namespace Vann_total_teeth_cleaned_l126_12676

theorem Vann_total_teeth_cleaned :
  let dogs := 7
  let cats := 12
  let pigs := 9
  let horses := 4
  let rabbits := 15
  let dogs_teeth := 42
  let cats_teeth := 30
  let pigs_teeth := 44
  let horses_teeth := 40
  let rabbits_teeth := 28
  (dogs * dogs_teeth) + (cats * cats_teeth) + (pigs * pigs_teeth) + (horses * horses_teeth) + (rabbits * rabbits_teeth) = 1630 :=
by
  sorry

end Vann_total_teeth_cleaned_l126_12676


namespace yuna_has_biggest_number_l126_12642

theorem yuna_has_biggest_number (yoongi : ℕ) (jungkook : ℕ) (yuna : ℕ) (hy : yoongi = 7) (hj : jungkook = 6) (hn : yuna = 9) :
  yuna = 9 ∧ yuna > yoongi ∧ yuna > jungkook :=
by 
  sorry

end yuna_has_biggest_number_l126_12642


namespace handrail_length_is_17_point_3_l126_12639

noncomputable def length_of_handrail (turn : ℝ) (rise : ℝ) (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (turn / 360) * circumference
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_is_17_point_3 : length_of_handrail 270 10 3 = 17.3 :=
by 
  sorry

end handrail_length_is_17_point_3_l126_12639


namespace problem1_problem2_problem3_l126_12617

theorem problem1 : 999 * 999 + 1999 = 1000000 := by
  sorry

theorem problem2 : 9 * 72 * 125 = 81000 := by
  sorry

theorem problem3 : 416 - 327 + 184 - 273 = 0 := by
  sorry

end problem1_problem2_problem3_l126_12617


namespace smallest_prime_with_prime_digit_sum_l126_12659

def is_prime (n : ℕ) : Prop := ¬ ∃ m, m ∣ n ∧ 1 < m ∧ m < n

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_prime_digit_sum :
  ∃ p, is_prime p ∧ is_prime (digit_sum p) ∧ 10 < digit_sum p ∧ p = 29 :=
by
  sorry

end smallest_prime_with_prime_digit_sum_l126_12659


namespace area_of_triangle_ABF_l126_12649

theorem area_of_triangle_ABF :
  let C : Set (ℝ × ℝ) := {p | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let F : ℝ × ℝ := (-1, 0)
  let AB := C ∩ line
  ∃ A B : ℝ × ℝ, A ∈ AB ∧ B ∈ AB ∧ A ≠ B ∧ 
  (1/2) * (2 : ℝ) * (12 * Real.sqrt (2 : ℝ) / 7) = (12 * Real.sqrt (2 : ℝ) / 7) :=
sorry

end area_of_triangle_ABF_l126_12649


namespace difference_between_perfect_and_cracked_l126_12633

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l126_12633


namespace charity_amount_l126_12640

theorem charity_amount (total : ℝ) (charities : ℕ) (amount_per_charity : ℝ) 
  (h1 : total = 3109) (h2 : charities = 25) : 
  amount_per_charity = 124.36 :=
by
  sorry

end charity_amount_l126_12640


namespace bread_slices_leftover_l126_12625

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end bread_slices_leftover_l126_12625


namespace maximum_value_l126_12615

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

theorem maximum_value (x1 x2 t : ℝ) (h1 : 0 < t) (h2 : f x1 = t) (h3 : g x2 = t) : 
  ∃ x1 x2, (t > 0) ∧ (f x1 = t) ∧ (g x2 = t) ∧ ((x1 / (x2 * Real.exp t)) = 1 / Real.exp 1) := 
sorry

end maximum_value_l126_12615


namespace ratio_of_numbers_l126_12679

theorem ratio_of_numbers (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l126_12679


namespace total_fruits_in_baskets_l126_12646

structure Baskets where
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  kiwis : ℕ
  lemons : ℕ

def taniaBaskets : Baskets := {
  mangoes := 18,
  pears := 10,
  pawpaws := 12,
  kiwis := 9,
  lemons := 9
}

theorem total_fruits_in_baskets : taniaBaskets.mangoes + taniaBaskets.pears + taniaBaskets.pawpaws + taniaBaskets.kiwis + taniaBaskets.lemons = 58 :=
by
  sorry

end total_fruits_in_baskets_l126_12646


namespace vessel_base_length_l126_12643

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_displaced (length breadth height: ℝ) : ℝ :=
  length * breadth * height

theorem vessel_base_length
  (breadth : ℝ) 
  (cube_edge : ℝ)
  (water_rise : ℝ)
  (displaced_volume : ℝ) 
  (h1 : breadth = 30) 
  (h2 : cube_edge = 30) 
  (h3 : water_rise = 15) 
  (h4 : volume_of_cube cube_edge = displaced_volume) :
  volume_displaced (displaced_volume / (breadth * water_rise)) breadth water_rise = displaced_volume :=
  by
  sorry

end vessel_base_length_l126_12643


namespace time_with_cat_total_l126_12631

def time_spent_with_cat (petting combing brushing playing feeding cleaning : ℕ) : ℕ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem time_with_cat_total :
  let petting := 12
  let combing := 1/3 * petting
  let brushing := 1/4 * combing
  let playing := 1/2 * petting
  let feeding := 5
  let cleaning := 2/5 * feeding
  time_spent_with_cat petting combing brushing playing feeding cleaning = 30 := by
  sorry

end time_with_cat_total_l126_12631


namespace solution_n_value_l126_12630

open BigOperators

noncomputable def problem_statement (a b n : ℝ) : Prop :=
  ∃ (A B : ℝ), A = Real.log a ∧ B = Real.log b ∧
    (7 * A + 15 * B) - (4 * A + 9 * B) = (11 * A + 20 * B) - (7 * A + 15 * B) ∧
    (4 + 135) * B = Real.log (b^n)

theorem solution_n_value (a b : ℝ) (h_pos : a > 0) (h_pos_b : b > 0) :
  problem_statement a b 139 :=
by
  sorry

end solution_n_value_l126_12630


namespace arithmetic_geometric_relation_l126_12619

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation_l126_12619


namespace tangent_line_to_circle_l126_12632

theorem tangent_line_to_circle {c : ℝ} (h : c > 0) :
  (∀ x y : ℝ, x^2 + y^2 = 8 → x + y = c) ↔ c = 4 := sorry

end tangent_line_to_circle_l126_12632


namespace carnations_in_first_bouquet_l126_12693

theorem carnations_in_first_bouquet 
  (c2 : ℕ) (c3 : ℕ) (avg : ℕ) (n : ℕ) (total_carnations : ℕ) : 
  c2 = 14 → c3 = 13 → avg = 12 → n = 3 → total_carnations = avg * n →
  (total_carnations - (c2 + c3) = 9) :=
by
  sorry

end carnations_in_first_bouquet_l126_12693


namespace abs_eq_1_solution_set_l126_12698

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end abs_eq_1_solution_set_l126_12698


namespace simplify_expression_l126_12683

theorem simplify_expression :
  (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by
  sorry

end simplify_expression_l126_12683


namespace pure_imaginary_number_implies_x_eq_1_l126_12634

theorem pure_imaginary_number_implies_x_eq_1 (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x + 1 ≠ 0) : x = 1 :=
sorry

end pure_imaginary_number_implies_x_eq_1_l126_12634


namespace product_of_four_consecutive_integers_is_perfect_square_l126_12661

-- Define the main statement we want to prove
theorem product_of_four_consecutive_integers_is_perfect_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  -- Proof is omitted
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l126_12661


namespace distance_between_foci_of_ellipse_l126_12606

-- Define the parameters a^2 and b^2 according to the problem
def a_sq : ℝ := 25
def b_sq : ℝ := 16

-- State the problem
theorem distance_between_foci_of_ellipse : 
  (2 * Real.sqrt (a_sq - b_sq)) = 6 := by
  -- Proof content is skipped 
  sorry

end distance_between_foci_of_ellipse_l126_12606


namespace system_solution_l126_12626

theorem system_solution (x y : ℝ) (h1 : 4 * x - y = 3) (h2 : x + 6 * y = 17) : x + y = 4 :=
by
  sorry

end system_solution_l126_12626


namespace inequality_proof_l126_12637

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a + b + c + d + 8 / (a*b + b*c + c*d + d*a) ≥ 6 := 
by
  sorry

end inequality_proof_l126_12637


namespace roots_square_sum_l126_12678

theorem roots_square_sum (a b : ℝ) 
  (h1 : a^2 - 4 * a + 4 = 0) 
  (h2 : b^2 - 4 * b + 4 = 0) 
  (h3 : a = b) :
  a^2 + b^2 = 8 := 
sorry

end roots_square_sum_l126_12678


namespace transform_equation_l126_12613

open Real

theorem transform_equation (m : ℝ) (x : ℝ) (h1 : x^2 + 4 * x = m) (h2 : (x + 2)^2 = 5) : m = 1 := by
  sorry

end transform_equation_l126_12613


namespace y1_mul_y2_eq_one_l126_12604

theorem y1_mul_y2_eq_one (x1 x2 y1 y2 : ℝ) (h1 : y1^2 = x1) (h2 : y2^2 = x2) 
  (h3 : y1 / (y1^2 - 1) = - (y2 / (y2^2 - 1))) (h4 : y1 + y2 ≠ 0) : y1 * y2 = 1 :=
sorry

end y1_mul_y2_eq_one_l126_12604


namespace percentage_of_volume_occupied_l126_12605

-- Define the dimensions of the block
def block_length : ℕ := 9
def block_width : ℕ := 7
def block_height : ℕ := 12

-- Define the dimension of the cube
def cube_side : ℕ := 4

-- Define the volumes
def block_volume : ℕ := block_length * block_width * block_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the count of cubes along each dimension
def cubes_along_length : ℕ := block_length / cube_side
def cubes_along_width : ℕ := block_width / cube_side
def cubes_along_height : ℕ := block_height / cube_side

-- Define the total number of cubes that fit into the block
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height

-- Define the total volume occupied by the cubes
def occupied_volume : ℕ := total_cubes * cube_volume

-- Define the percentage of the block's volume occupied by the cubes (as a float for precision)
def volume_percentage : Float := (Float.ofNat occupied_volume / Float.ofNat block_volume) * 100

-- Statement to prove
theorem percentage_of_volume_occupied :
  volume_percentage = 50.79 := by
  sorry

end percentage_of_volume_occupied_l126_12605


namespace emily_small_gardens_count_l126_12636

-- Definitions based on conditions
def initial_seeds : ℕ := 41
def seeds_planted_in_big_garden : ℕ := 29
def seeds_per_small_garden : ℕ := 4

-- Theorem statement
theorem emily_small_gardens_count (initial_seeds seeds_planted_in_big_garden seeds_per_small_garden : ℕ) :
  initial_seeds = 41 →
  seeds_planted_in_big_garden = 29 →
  seeds_per_small_garden = 4 →
  (initial_seeds - seeds_planted_in_big_garden) / seeds_per_small_garden = 3 :=
by
  intros
  sorry

end emily_small_gardens_count_l126_12636


namespace polygon_perpendiculars_length_l126_12680

noncomputable def RegularPolygon := { n : ℕ // n ≥ 3 }

structure Perpendiculars (P : RegularPolygon) (i : ℕ) :=
  (d_i     : ℝ)
  (d_i_minus_1 : ℝ)
  (d_i_plus_1 : ℝ)
  (line_crosses_interior : Bool)

theorem polygon_perpendiculars_length {P : RegularPolygon} {i : ℕ}
  (hyp : Perpendiculars P i) :
  hyp.d_i = if hyp.line_crosses_interior 
            then hyp.d_i_minus_1 + hyp.d_i_plus_1 
            else abs (hyp.d_i_minus_1 - hyp.d_i_plus_1) :=
sorry

end polygon_perpendiculars_length_l126_12680


namespace algebraic_expression_value_l126_12660

theorem algebraic_expression_value (a : ℝ) (h : (a^2 - 3) * (a^2 + 1) = 0) : a^2 = 3 :=
by
  sorry

end algebraic_expression_value_l126_12660


namespace new_average_rent_l126_12695

theorem new_average_rent 
  (n : ℕ) (h_n : n = 4) 
  (avg_old : ℝ) (h_avg_old : avg_old = 800) 
  (inc_rate : ℝ) (h_inc_rate : inc_rate = 0.16) 
  (old_rent : ℝ) (h_old_rent : old_rent = 1250) 
  (new_rent : ℝ) (h_new_rent : new_rent = old_rent * (1 + inc_rate)) 
  (total_rent_old : ℝ) (h_total_rent_old : total_rent_old = n * avg_old)
  (total_rent_new : ℝ) (h_total_rent_new : total_rent_new = total_rent_old - old_rent + new_rent)
  (avg_new : ℝ) (h_avg_new : avg_new = total_rent_new / n) : 
  avg_new = 850 := 
sorry

end new_average_rent_l126_12695


namespace fruit_basket_combinations_l126_12638

theorem fruit_basket_combinations :
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples+1) * (oranges+1) * (bananas+1)
  let empty_basket := 1
  total_combinations - empty_basket = 159 :=
by
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples + 1) * (oranges + 1) * (bananas + 1)
  let empty_basket := 1
  have h_total_combinations : total_combinations = 4 * 8 * 5 := by sorry
  have h_empty_basket : empty_basket = 1 := by sorry
  have h_subtract : 4 * 8 * 5 - 1 = 159 := by sorry
  exact h_subtract

end fruit_basket_combinations_l126_12638


namespace least_possible_sum_l126_12611

theorem least_possible_sum {c d : ℕ} (hc : c ≥ 2) (hd : d ≥ 2) (h : 3 * c + 6 = 6 * d + 3) : c + d = 5 :=
by
  sorry

end least_possible_sum_l126_12611


namespace perpendicular_lines_have_a_zero_l126_12656

theorem perpendicular_lines_have_a_zero {a : ℝ} :
  ∀ x y : ℝ, (ax + y - 1 = 0) ∧ (x + a*y - 1 = 0) → a = 0 :=
by
  sorry

end perpendicular_lines_have_a_zero_l126_12656


namespace relationship_f_l126_12668

-- Define the function f which is defined on the reals and even
variable (f : ℝ → ℝ)
-- Condition: f is an even function
axiom even_f : ∀ x, f (-x) = f x
-- Condition: (x₁ - x₂)[f(x₁) - f(x₂)] > 0 for all x₁, x₂ ∈ [0, +∞)
axiom increasing_cond : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem relationship_f : f (1/2) < f 1 ∧ f 1 < f (-2) := by
  sorry

end relationship_f_l126_12668


namespace area_to_paint_l126_12671

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def bookshelf_height : ℕ := 3
def bookshelf_length : ℕ := 5

theorem area_to_paint : (wall_height * wall_length) - (bookshelf_height * bookshelf_length) = 135 :=
by 
  sorry

end area_to_paint_l126_12671


namespace physical_education_class_min_size_l126_12650

theorem physical_education_class_min_size :
  ∃ (x : Nat), 3 * x + 2 * (x + 1) > 50 ∧ 5 * x + 2 = 52 := by
  sorry

end physical_education_class_min_size_l126_12650


namespace number_of_bouncy_balls_per_package_l126_12682

theorem number_of_bouncy_balls_per_package (x : ℕ) (h : 4 * x + 8 * x + 4 * x = 160) : x = 10 :=
by
  sorry

end number_of_bouncy_balls_per_package_l126_12682


namespace surface_area_of_cube_l126_12692

noncomputable def cube_edge_length : ℝ := 20

theorem surface_area_of_cube (edge_length : ℝ) (h : edge_length = cube_edge_length) : 
    6 * edge_length ^ 2 = 2400 :=
by
  rw [h]
  sorry  -- proof placeholder

end surface_area_of_cube_l126_12692


namespace weight_of_new_person_l126_12666

theorem weight_of_new_person (W : ℝ) : 
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  weight_new_person = 70 :=
by
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  have : weight_new_person = 70 := sorry
  exact this

end weight_of_new_person_l126_12666


namespace relationship_among_a_b_c_l126_12674

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_among_a_b_c (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x :=
by {
  -- proof steps are skipped
  sorry
}

end relationship_among_a_b_c_l126_12674


namespace composite_sum_l126_12653

theorem composite_sum (a b : ℤ) (h : 56 * a = 65 * b) : ∃ m n : ℤ,  m > 1 ∧ n > 1 ∧ a + b = m * n :=
sorry

end composite_sum_l126_12653


namespace total_call_charges_l126_12608

-- Definitions based on conditions
def base_fee : ℝ := 39
def included_minutes : ℕ := 300
def excess_charge_per_minute : ℝ := 0.19

-- Given variables
variable (x : ℕ) -- excess minutes
variable (y : ℝ) -- total call charges

-- Theorem stating the relationship between y and x
theorem total_call_charges (h : x > 0) : y = 0.19 * x + 39 := 
by sorry

end total_call_charges_l126_12608


namespace sum_square_ends_same_digit_l126_12681

theorem sum_square_ends_same_digit {a b : ℤ} (h : (a + b) % 10 = 0) :
  (a^2 % 10) = (b^2 % 10) :=
by
  sorry

end sum_square_ends_same_digit_l126_12681


namespace num_men_in_second_group_l126_12601

-- Define the conditions
def numMen1 := 4
def hoursPerDay1 := 10
def daysPerWeek := 7
def earningsPerWeek1 := 1200

def hoursPerDay2 := 6
def earningsPerWeek2 := 1620

-- Define the earning per man-hour
def earningPerManHour := earningsPerWeek1 / (numMen1 * hoursPerDay1 * daysPerWeek)

-- Define the total man-hours required for the second amount of earnings
def totalManHours2 := earningsPerWeek2 / earningPerManHour

-- Define the number of men in the second group
def numMen2 := totalManHours2 / (hoursPerDay2 * daysPerWeek)

-- Theorem stating the number of men in the second group 
theorem num_men_in_second_group : numMen2 = 9 := by
  sorry

end num_men_in_second_group_l126_12601


namespace gift_card_value_l126_12689

def latte_cost : ℝ := 3.75
def croissant_cost : ℝ := 3.50
def daily_treat_cost : ℝ := latte_cost + croissant_cost
def weekly_treat_cost : ℝ := daily_treat_cost * 7

def cookie_cost : ℝ := 1.25
def total_cookie_cost : ℝ := cookie_cost * 5

def total_spent : ℝ := weekly_treat_cost + total_cookie_cost
def remaining_balance : ℝ := 43.00

theorem gift_card_value : (total_spent + remaining_balance) = 100 := 
by sorry

end gift_card_value_l126_12689


namespace mr_brown_final_price_is_correct_l126_12628

noncomputable def mr_brown_final_purchase_price :
  Float :=
  let initial_price : Float := 100000
  let mr_brown_price  := initial_price * 1.12
  let improvement := mr_brown_price * 0.05
  let mr_brown_total_investment := mr_brown_price + improvement
  let mr_green_purchase_price := mr_brown_total_investment * 1.04
  let market_decline := mr_green_purchase_price * 0.03
  let value_after_decline := mr_green_purchase_price - market_decline
  let loss := value_after_decline * 0.10
  let ms_white_purchase_price := value_after_decline - loss
  let market_increase := ms_white_purchase_price * 0.08
  let value_after_increase := ms_white_purchase_price + market_increase
  let profit := value_after_increase * 0.05
  let final_price := value_after_increase + profit
  final_price

theorem mr_brown_final_price_is_correct :
  mr_brown_final_purchase_price = 121078.76 := by
  sorry

end mr_brown_final_price_is_correct_l126_12628


namespace pumpkin_pie_degrees_l126_12670

theorem pumpkin_pie_degrees (total_students : ℕ) (peach_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ)
                               (pumpkin_pie : ℕ) (banana_pie : ℕ)
                               (h_total : total_students = 40)
                               (h_peach : peach_pie = 14)
                               (h_apple : apple_pie = 9)
                               (h_blueberry : blueberry_pie = 7)
                               (h_remaining : pumpkin_pie = banana_pie)
                               (h_half_remaining : 2 * pumpkin_pie = 40 - (peach_pie + apple_pie + blueberry_pie)) :
  (pumpkin_pie * 360) / total_students = 45 := by
sorry

end pumpkin_pie_degrees_l126_12670


namespace value_of_expression_l126_12635

theorem value_of_expression (x : ℝ) (h : x^2 - 3 * x = 4) : 3 * x^2 - 9 * x + 8 = 20 := 
by
  sorry

end value_of_expression_l126_12635


namespace journey_total_distance_l126_12663

theorem journey_total_distance :
  let speed1 := 40 -- in kmph
  let time1 := 3 -- in hours
  let speed2 := 60 -- in kmph
  let totalTime := 5 -- in hours
  let distance1 := speed1 * time1
  let time2 := totalTime - time1
  let distance2 := speed2 * time2
  let totalDistance := distance1 + distance2
  totalDistance = 240 := 
by
  sorry

end journey_total_distance_l126_12663


namespace ellipse_equation_correct_coordinates_c_correct_l126_12629

-- Definition of the ellipse Γ with given properties
def ellipse_properties (a b : ℝ) (ecc : ℝ) (c_len : ℝ) :=
  a > b ∧ b > 0 ∧ ecc = (Real.sqrt 2) / 2 ∧ c_len = Real.sqrt 2

-- Correct answer for the equation of the ellipse
def correct_ellipse_equation := ∀ x y : ℝ, (x^2) / 2 + y^2 = 1

-- Proving that given the properties of the ellipse, the equation is as stated
theorem ellipse_equation_correct (a b : ℝ) (h : ellipse_properties a b (Real.sqrt 2 / 2) (Real.sqrt 2)) :
  (x^2) / 2 + y^2 = 1 := 
  sorry

-- Definition of the conditions for points A, B, and C
def triangle_conditions (a b : ℝ) (area : ℝ) :=
  ∀ A B : ℝ × ℝ,
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    area = 3 * Real.sqrt 6 / 4

-- Correct coordinates of point C given the conditions
def correct_coordinates_c (C : ℝ × ℝ) :=
  (C = (1, Real.sqrt 2 / 2) ∨ C = (2, 1))

-- Proving that given the conditions, the coordinates of point C are correct
theorem coordinates_c_correct (a b : ℝ) (h : triangle_conditions a b (3 * Real.sqrt 6 / 4)) (C : ℝ × ℝ) :
  correct_coordinates_c C :=
  sorry

end ellipse_equation_correct_coordinates_c_correct_l126_12629


namespace total_area_of_house_is_2300_l126_12610

-- Definitions based on the conditions in the problem
def area_living_room_dining_room_kitchen : ℕ := 1000
def area_master_bedroom_suite : ℕ := 1040
def area_guest_bedroom : ℕ := area_master_bedroom_suite / 4

-- Theorem to state the total area of the house
theorem total_area_of_house_is_2300 :
  area_living_room_dining_room_kitchen + area_master_bedroom_suite + area_guest_bedroom = 2300 :=
by
  sorry

end total_area_of_house_is_2300_l126_12610


namespace probability_plane_contains_points_inside_octahedron_l126_12622

noncomputable def enhanced_octahedron_probability : ℚ :=
  let total_vertices := 18
  let total_ways := Nat.choose total_vertices 3
  let faces := 8
  let triangles_per_face := 4
  let unfavorable_ways := faces * triangles_per_face
  total_ways - unfavorable_ways

theorem probability_plane_contains_points_inside_octahedron :
  enhanced_octahedron_probability / (816 : ℚ) = 49 / 51 :=
sorry

end probability_plane_contains_points_inside_octahedron_l126_12622


namespace at_least_100_valid_pairs_l126_12673

-- Define the conditions
def boots_distribution (L41 L42 L43 R41 R42 R43 : ℕ) : Prop :=
  L41 + L42 + L43 = 300 ∧ R41 + R42 + R43 = 300 ∧
  (L41 = 200 ∨ L42 = 200 ∨ L43 = 200) ∧
  (R41 = 200 ∨ R42 = 200 ∨ R43 = 200)

-- Define the theorem to be proven
theorem at_least_100_valid_pairs (L41 L42 L43 R41 R42 R43 : ℕ) :
  boots_distribution L41 L42 L43 R41 R42 R43 → 
  (L41 ≥ 100 ∧ R41 ≥ 100 ∨ L42 ≥ 100 ∧ R42 ≥ 100 ∨ L43 ≥ 100 ∧ R43 ≥ 100) → 100 ≤ min L41 R41 ∨ 100 ≤ min L42 R42 ∨ 100 ≤ min L43 R43 :=
  sorry

end at_least_100_valid_pairs_l126_12673


namespace find_special_number_l126_12684

theorem find_special_number : 
  ∃ n, 
  (n % 12 = 11) ∧ 
  (n % 11 = 10) ∧ 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 27719) :=
  sorry

end find_special_number_l126_12684


namespace C_share_per_rs_equals_l126_12677

-- Definitions based on given conditions
def A_share_per_rs (x : ℝ) : ℝ := x
def B_share_per_rs : ℝ := 0.65
def C_share : ℝ := 48
def total_sum : ℝ := 246

-- The target statement to prove
theorem C_share_per_rs_equals : C_share / total_sum = 0.195122 :=
by
  sorry

end C_share_per_rs_equals_l126_12677


namespace number_of_toddlers_l126_12669

-- Definitions based on the conditions provided in the problem
def total_children := 40
def newborns := 4
def toddlers (T : ℕ) := T
def teenagers (T : ℕ) := 5 * T

-- The theorem to prove
theorem number_of_toddlers : ∃ T : ℕ, newborns + toddlers T + teenagers T = total_children ∧ T = 6 :=
by
  sorry

end number_of_toddlers_l126_12669


namespace ratio_of_sides_l126_12652

variable {A B C a b c : ℝ}

theorem ratio_of_sides
  (h1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B)
  (h2 : c = 2 * b) :
  a / b = Real.sqrt 2 := by
  sorry

end ratio_of_sides_l126_12652


namespace technicians_count_l126_12602

/-- Given a workshop with 49 workers, where the average salary of all workers 
    is Rs. 8000, the average salary of the technicians is Rs. 20000, and the
    average salary of the rest is Rs. 6000, prove that the number of 
    technicians is 7. -/
theorem technicians_count (T R : ℕ) (h1 : T + R = 49) (h2 : 10 * T + 3 * R = 196) : T = 7 := 
by
  sorry

end technicians_count_l126_12602


namespace pour_tea_into_containers_l126_12696

-- Define the total number of containers
def total_containers : ℕ := 80

-- Define the amount of tea that Geraldo drank in terms of containers
def geraldo_drank_containers : ℚ := 3.5

-- Define the amount of tea that Geraldo consumed in terms of pints
def geraldo_drank_pints : ℕ := 7

-- Define the conversion factor from pints to gallons
def pints_per_gallon : ℕ := 8

-- Question: How many gallons of tea were poured into the containers?
theorem pour_tea_into_containers 
  (total_containers : ℕ)
  (geraldo_drank_containers : ℚ)
  (geraldo_drank_pints : ℕ)
  (pints_per_gallon : ℕ) :
  (total_containers * (geraldo_drank_pints / geraldo_drank_containers) / pints_per_gallon) = 20 :=
by
  sorry

end pour_tea_into_containers_l126_12696


namespace koala_fiber_consumption_l126_12620

theorem koala_fiber_consumption (x : ℝ) (H : 12 = 0.30 * x) : x = 40 :=
by
  sorry

end koala_fiber_consumption_l126_12620


namespace multiplication_correct_l126_12655

theorem multiplication_correct (x : ℤ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

end multiplication_correct_l126_12655


namespace inequality_problem_l126_12675

theorem inequality_problem
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( ( (2 * a + b + c) ^ 2 ) / ( 2 * a ^ 2 + (b + c) ^ 2 ) ) +
  ( ( (a + 2 * b + c) ^ 2 ) / ( 2 * b ^ 2 + (c + a) ^ 2 ) ) +
  ( ( (a + b + 2 * c) ^ 2 ) / ( 2 * c ^ 2 + (a + b) ^ 2 ) ) ≤ 8 :=
by
  sorry

end inequality_problem_l126_12675


namespace units_digit_n_is_7_l126_12657

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_n_is_7 (m n : ℕ) (h1 : m * n = 31 ^ 4) (h2 : units_digit m = 6) :
  units_digit n = 7 :=
sorry

end units_digit_n_is_7_l126_12657


namespace no_natural_numbers_satisfying_conditions_l126_12607

theorem no_natural_numbers_satisfying_conditions :
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ k : ℕ, b^2 + 4*a = k^2 := by
  sorry

end no_natural_numbers_satisfying_conditions_l126_12607


namespace cube_surface_area_ratio_l126_12651

variable (x : ℝ) (hx : x > 0)

theorem cube_surface_area_ratio (hx : x > 0):
  let side1 := 7 * x
  let side2 := x
  let SA1 := 6 * side1^2
  let SA2 := 6 * side2^2
  (SA1 / SA2) = 49 := 
by 
  sorry

end cube_surface_area_ratio_l126_12651


namespace selling_price_equivalence_l126_12664

noncomputable def cost_price_25_profit : ℝ := 1750 / 1.25
def selling_price_profit := 1520
def selling_price_loss := 1280

theorem selling_price_equivalence
  (cp : ℝ)
  (h1 : cp = cost_price_25_profit)
  (h2 : cp = 1400) :
  (selling_price_profit - cp = cp - selling_price_loss) → (selling_price_loss = 1280) := 
  by
  unfold cost_price_25_profit at h1
  simp [h1] at h2
  sorry

end selling_price_equivalence_l126_12664
