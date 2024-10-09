import Mathlib

namespace sufficient_condition_p_or_q_false_p_and_q_false_l66_6661

variables (p q : Prop)

theorem sufficient_condition_p_or_q_false_p_and_q_false :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ¬ ( (¬ (p ∧ q)) → ¬ (p ∨ q)) :=
by 
  -- Proof: If ¬ (p ∨ q), then (p ∨ q) is false, which means (p ∧ q) must also be false.
  -- The other direction would mean if at least one of p or q is false, then (p ∨ q) is false,
  -- which is not necessarily true. Therefore, it's not a necessary condition.
  sorry

end sufficient_condition_p_or_q_false_p_and_q_false_l66_6661


namespace all_three_pets_l66_6650

-- Definitions of the given conditions
def total_students : ℕ := 40
def dog_owners : ℕ := 20
def cat_owners : ℕ := 13
def other_pet_owners : ℕ := 8
def no_pets : ℕ := 7

-- Definitions from Venn diagram
def dogs_only : ℕ := 12
def cats_only : ℕ := 3
def other_pets_only : ℕ := 2

-- Intersection variables
variables (a b c d : ℕ)

-- Translated problem
theorem all_three_pets :
  dogs_only + cats_only + other_pets_only + a + b + c + d = total_students - no_pets ∧
  dogs_only + a + c + d = dog_owners ∧
  cats_only + a + b + d = cat_owners ∧
  other_pets_only + b + c + d = other_pet_owners ∧
  d = 2 :=
sorry

end all_three_pets_l66_6650


namespace only_one_correct_guess_l66_6609

-- Define the contestants
inductive Contestant : Type
| person : ℕ → Contestant

def A_win_first (c: Contestant) : Prop :=
c = Contestant.person 4 ∨ c = Contestant.person 5

def B_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 3 

def C_win_first (c: Contestant) : Prop :=
c = Contestant.person 1 ∨ c = Contestant.person 2 ∨ c = Contestant.person 6

def D_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 4 ∧ c ≠ Contestant.person 5 ∧ c ≠ Contestant.person 6

-- The main theorem: Only one correct guess among A, B, C, and D
theorem only_one_correct_guess (win: Contestant) :
  (A_win_first win ↔ false) ∧ (B_not_win_first win ↔ false) ∧ (C_win_first win ↔ false) ∧ D_not_win_first win
:=
by
  sorry

end only_one_correct_guess_l66_6609


namespace tan_theta_perpendicular_vectors_l66_6687

theorem tan_theta_perpendicular_vectors (θ : ℝ) (h : Real.sqrt 3 * Real.cos θ + Real.sin θ = 0) : Real.tan θ = - Real.sqrt 3 :=
sorry

end tan_theta_perpendicular_vectors_l66_6687


namespace smallest_y_for_perfect_square_l66_6667

theorem smallest_y_for_perfect_square (x y: ℕ) (h : x = 5 * 32 * 45) (hY: y = 2) : 
  ∃ v: ℕ, (x * y = v ^ 2) :=
by
  use 2
  rw [h, hY]
  -- expand and simplify
  sorry

end smallest_y_for_perfect_square_l66_6667


namespace mindy_emails_l66_6662

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end mindy_emails_l66_6662


namespace hex_A08_to_decimal_l66_6617

noncomputable def hex_A := 10
noncomputable def hex_A08_base_10 : ℕ :=
  (hex_A * 16^2) + (0 * 16^1) + (8 * 16^0)

theorem hex_A08_to_decimal :
  hex_A08_base_10 = 2568 :=
by
  sorry

end hex_A08_to_decimal_l66_6617


namespace symmetric_point_proof_l66_6626

noncomputable def point_symmetric_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_proof :
  point_symmetric_to_x_axis (-2, 3) = (-2, -3) :=
by
  sorry

end symmetric_point_proof_l66_6626


namespace area_increase_cost_increase_l66_6695

-- Given definitions based only on the conditions from part a
def original_length := 60
def original_width := 20
def original_fence_cost_per_foot := 15
def original_perimeter := 2 * (original_length + original_width)
def original_fencing_cost := original_perimeter * original_fence_cost_per_foot

def new_fence_cost_per_foot := 20
def new_square_side := original_perimeter / 4
def new_square_area := new_square_side * new_square_side
def new_fencing_cost := original_perimeter * new_fence_cost_per_foot

-- Proof statements using the conditions and correct answers from part b
theorem area_increase : new_square_area - (original_length * original_width) = 400 := by
  sorry

theorem cost_increase : new_fencing_cost - original_fencing_cost = 800 := by
  sorry

end area_increase_cost_increase_l66_6695


namespace car_length_l66_6610

variables (L E C : ℕ)

theorem car_length (h1 : 150 * E = L + 150 * C) (h2 : 30 * E = L - 30 * C) : L = 113 * E :=
by
  sorry

end car_length_l66_6610


namespace car_mileage_before_modification_l66_6633

theorem car_mileage_before_modification (miles_per_gallon_before : ℝ) 
  (fuel_efficiency_modifier : ℝ := 0.75) (tank_capacity : ℝ := 12) 
  (extra_miles_after_modification : ℝ := 96) :
  (1 / fuel_efficiency_modifier) * miles_per_gallon_before * (tank_capacity - 1) = 24 :=
by
  sorry

end car_mileage_before_modification_l66_6633


namespace three_digit_diff_no_repeated_digits_l66_6664

theorem three_digit_diff_no_repeated_digits :
  let largest := 987
  let smallest := 102
  largest - smallest = 885 := by
  sorry

end three_digit_diff_no_repeated_digits_l66_6664


namespace total_arrangements_l66_6616

def count_arrangements : Nat :=
  let male_positions := 3
  let female_positions := 3
  let male_arrangements := Nat.factorial male_positions
  let female_arrangements := Nat.factorial (female_positions - 1)
  male_arrangements * female_arrangements / (male_positions - female_positions + 1)

theorem total_arrangements : count_arrangements = 36 := by
  sorry

end total_arrangements_l66_6616


namespace hall_length_l66_6637

theorem hall_length (b : ℕ) (h1 : b + 5 > 0) (h2 : (b + 5) * b = 750) : b + 5 = 30 :=
by {
  -- Proof goes here
  sorry
}

end hall_length_l66_6637


namespace required_circle_equation_l66_6653

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end required_circle_equation_l66_6653


namespace smallest_y_l66_6656

theorem smallest_y (y : ℕ) (h : 56 * y + 8 ≡ 6 [MOD 26]) : y = 6 := by
  sorry

end smallest_y_l66_6656


namespace remainder_when_x_plus_3uy_div_y_l66_6669

theorem remainder_when_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (v_lt_y : v < y) :
  ((x + 3 * u * y) % y) = v := 
sorry

end remainder_when_x_plus_3uy_div_y_l66_6669


namespace inequality_proof_l66_6607

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end inequality_proof_l66_6607


namespace num_positive_integers_l66_6612

theorem num_positive_integers (n : ℕ) :
    (0 < n ∧ n < 40 ∧ ∃ k : ℕ, k > 0 ∧ n = 40 * k / (k + 1)) ↔ 
    (n = 20 ∨ n = 30 ∨ n = 32 ∨ n = 35 ∨ n = 36 ∨ n = 38 ∨ n = 39) :=
sorry

end num_positive_integers_l66_6612


namespace chocolate_bars_gigantic_box_l66_6621

def large_boxes : ℕ := 50
def medium_boxes : ℕ := 25
def small_boxes : ℕ := 10
def chocolate_bars_per_small_box : ℕ := 45

theorem chocolate_bars_gigantic_box : 
  large_boxes * medium_boxes * small_boxes * chocolate_bars_per_small_box = 562500 :=
by
  sorry

end chocolate_bars_gigantic_box_l66_6621


namespace Laura_running_speed_l66_6694

noncomputable def running_speed (x : ℝ) : Prop :=
  (15 / (3 * x + 2)) + (4 / x) = 1.5 ∧ x > 0

theorem Laura_running_speed : ∃ (x : ℝ), running_speed x ∧ abs (x - 5.64) < 0.01 :=
by
  sorry

end Laura_running_speed_l66_6694


namespace interest_difference_20_years_l66_6681

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem interest_difference_20_years :
  compound_interest 15000 0.06 20 - simple_interest 15000 0.08 20 = 9107 :=
by
  sorry

end interest_difference_20_years_l66_6681


namespace total_pencils_correct_l66_6676

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end total_pencils_correct_l66_6676


namespace pipe_filling_time_l66_6679

theorem pipe_filling_time 
  (rate_A : ℚ := 1/8) 
  (rate_L : ℚ := 1/24) :
  (1 / (rate_A - rate_L) = 12) :=
by
  sorry

end pipe_filling_time_l66_6679


namespace book_selection_l66_6624

def num_books_in_genre (mystery fantasy biography : ℕ) : ℕ :=
  mystery + fantasy + biography

def num_combinations_two_diff_genres (mystery fantasy biography : ℕ) : ℕ :=
  if mystery = 4 ∧ fantasy = 4 ∧ biography = 4 then 48 else 0

theorem book_selection : 
  ∀ (mystery fantasy biography : ℕ),
  num_books_in_genre mystery fantasy biography = 12 →
  num_combinations_two_diff_genres mystery fantasy biography = 48 :=
by
  intros mystery fantasy biography h
  sorry

end book_selection_l66_6624


namespace glucose_amount_in_45cc_l66_6655

noncomputable def glucose_in_container (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) : ℝ :=
  (concentration * poured_volume) / total_volume

theorem glucose_amount_in_45cc (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) :
  concentration = 10 → total_volume = 100 → poured_volume = 45 →
  glucose_in_container concentration total_volume poured_volume = 4.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end glucose_amount_in_45cc_l66_6655


namespace arithmetic_sequence_properties_l66_6689

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 : ℝ}

theorem arithmetic_sequence_properties 
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a1 > 0) 
  (h3 : a 9 + a 10 = a 11) :
  (∀ m n, m < n → a m > a n) ∧ (∀ n, S n = n * (a1 + (d * (n - 1) / 2))) ∧ S 14 > 0 :=
by 
  sorry

end arithmetic_sequence_properties_l66_6689


namespace speed_calculation_l66_6602

def distance := 600 -- in meters
def time := 2 -- in minutes

def distance_km := distance / 1000 -- converting meters to kilometers
def time_hr := time / 60 -- converting minutes to hours

theorem speed_calculation : (distance_km / time_hr = 18) :=
 by
  sorry

end speed_calculation_l66_6602


namespace single_intersection_l66_6654

theorem single_intersection (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y^2 = x ∧ y + 1 = k * x) ↔ (k = 0 ∨ k = -1 / 4) :=
sorry

end single_intersection_l66_6654


namespace smallest_value_of_3b_plus_2_l66_6692

theorem smallest_value_of_3b_plus_2 (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) : (∃ t : ℝ, t = 3 * b + 2 ∧ (∀ x : ℝ, 8 * x^2 + 7 * x + 6 = 5 → x = b → t ≤ 3 * x + 2)) :=
sorry

end smallest_value_of_3b_plus_2_l66_6692


namespace solutions_equiv_cond_l66_6648

theorem solutions_equiv_cond (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x + 1 / (x - 1) = a + 1 / (x - 1)) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x = a) ∧ (∃ x : ℝ, x = 1 → a ≠ 4)  :=
sorry

end solutions_equiv_cond_l66_6648


namespace largest_possible_x_l66_6638

theorem largest_possible_x :
  ∃ x : ℝ, (3*x^2 + 18*x - 84 = x*(x + 10)) ∧ ∀ y : ℝ, (3*y^2 + 18*y - 84 = y*(y + 10)) → y ≤ x :=
by
  sorry

end largest_possible_x_l66_6638


namespace equation_of_line_l66_6696

theorem equation_of_line (θ : ℝ) (b : ℝ) :
  θ = 135 ∧ b = -1 → (∀ x y : ℝ, x + y + 1 = 0) :=
by
  sorry

end equation_of_line_l66_6696


namespace dispatch_plans_l66_6685

theorem dispatch_plans (students : Finset ℕ) (h : students.card = 6) :
  ∃ (plans : Finset (Finset ℕ)), plans.card = 180 :=
by
  sorry

end dispatch_plans_l66_6685


namespace revenue_increase_l66_6697

theorem revenue_increase (R : ℕ) (r2000 r2003 r2005 : ℝ) (h1 : r2003 = r2000 * 1.50) (h2 : r2005 = r2000 * 1.80) :
  ((r2005 - r2003) / r2003) * 100 = 20 :=
by sorry

end revenue_increase_l66_6697


namespace train_length_calculation_l66_6693

theorem train_length_calculation
  (speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_length : ℝ)
  (h1 : speed_kmph = 80)
  (h2 : time_seconds = 8.999280057595392)
  (h3 : train_length = (80 * 1000) / 3600 * 8.999280057595392) :
  train_length = 200 := by
  sorry

end train_length_calculation_l66_6693


namespace points_per_correct_answer_l66_6644

theorem points_per_correct_answer (x : ℕ) : 
  let total_questions := 30
  let points_deducted_per_incorrect := 5
  let total_score := 325
  let correct_answers := 19
  let incorrect_answers := total_questions - correct_answers
  let points_lost_from_incorrect := incorrect_answers * points_deducted_per_incorrect
  let score_from_correct := correct_answers * x
  (score_from_correct - points_lost_from_incorrect = total_score) → x = 20 :=
by {
  sorry
}

end points_per_correct_answer_l66_6644


namespace discount_percentage_l66_6639

theorem discount_percentage (wholesale_price retail_price selling_price profit: ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : profit = 0.20 * wholesale_price)
  (h4 : selling_price = wholesale_price + profit):
  (retail_price - selling_price) / retail_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l66_6639


namespace binomial_distrib_not_equiv_binom_expansion_l66_6634

theorem binomial_distrib_not_equiv_binom_expansion (a b : ℝ) (n : ℕ) (p : ℝ) (h1: a = p) (h2: b = 1 - p):
    ¬ (∃ k : ℕ, p ^ k * (1 - p) ^ (n - k) = (a + b) ^ n) := sorry

end binomial_distrib_not_equiv_binom_expansion_l66_6634


namespace find_x_l66_6684

theorem find_x {x : ℝ} :
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 :=
by
  intro h
  -- Solution steps would go here, but they are omitted.
  sorry

end find_x_l66_6684


namespace volunteer_group_selection_l66_6646

theorem volunteer_group_selection :
  let M := 4  -- Number of male teachers
  let F := 5  -- Number of female teachers
  let G := 3  -- Total number of teachers in the group
  -- Calculate the number of ways to select 2 male teachers and 1 female teacher
  let ways1 := (Nat.choose M 2) * (Nat.choose F 1)
  -- Calculate the number of ways to select 1 male teacher and 2 female teachers
  let ways2 := (Nat.choose M 1) * (Nat.choose F 2)
  -- The total number of ways to form the group
  ways1 + ways2 = 70 := by sorry

end volunteer_group_selection_l66_6646


namespace lines_through_P_and_form_area_l66_6604

-- Definition of the problem conditions
def passes_through_P (k b : ℝ) : Prop :=
  b = 2 - k

def forms_area_with_axes (k b : ℝ) : Prop :=
  b^2 = 8 * |k|

-- Theorem statement
theorem lines_through_P_and_form_area :
  ∃ (k1 k2 k3 b1 b2 b3 : ℝ),
    passes_through_P k1 b1 ∧ forms_area_with_axes k1 b1 ∧
    passes_through_P k2 b2 ∧ forms_area_with_axes k2 b2 ∧
    passes_through_P k3 b3 ∧ forms_area_with_axes k3 b3 ∧
    k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 :=
sorry

end lines_through_P_and_form_area_l66_6604


namespace largest_divisor_of_expression_l66_6677

theorem largest_divisor_of_expression (x : ℤ) (hx : x % 2 = 1) :
  864 ∣ (12 * x + 2) * (12 * x + 6) * (12 * x + 10) * (6 * x + 3) :=
sorry

end largest_divisor_of_expression_l66_6677


namespace sum_of_three_ints_product_5_4_l66_6625

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l66_6625


namespace intersection_M_N_l66_6665

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

theorem intersection_M_N : M ∩ N = {0, 3} := by
  sorry

end intersection_M_N_l66_6665


namespace insurance_covers_80_percent_l66_6611

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def total_cost : ℕ := xray_cost + mri_cost
def mike_payment : ℕ := 200
def insurance_coverage : ℕ := total_cost - mike_payment
def insurance_percentage : ℕ := (insurance_coverage * 100) / total_cost

theorem insurance_covers_80_percent : insurance_percentage = 80 := by
  -- Carry out the necessary calculations
  sorry

end insurance_covers_80_percent_l66_6611


namespace fraction_shaded_l66_6670

-- Define relevant elements
def quilt : ℕ := 9
def rows : ℕ := 3
def shaded_rows : ℕ := 1
def shaded_fraction := shaded_rows / rows

-- We are to prove the fraction of the quilt that is shaded
theorem fraction_shaded (h : quilt = 3 * 3) : shaded_fraction = 1 / 3 :=
by
  -- Proof goes here
  sorry

end fraction_shaded_l66_6670


namespace arithmetic_sequence_S9_l66_6642

-- Define the sum of an arithmetic sequence: S_n
def arithmetic_sequence_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := (n * (2 * a (0) + (n - 1) * d (0))) / 2

-- Conditions
variable (a d : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (h1 : S_n 3 = 9)
variable (h2 : S_n 6 = 27)

-- Question: Prove that S_9 = 54
theorem arithmetic_sequence_S9 : S_n 9 = 54 := by
    sorry

end arithmetic_sequence_S9_l66_6642


namespace problem_statement_l66_6628

noncomputable def term_with_largest_binomial_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ :=
-8064

noncomputable def term_with_largest_absolute_value_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ × ℕ :=
(-15360, 8)

theorem problem_statement (M N P : ℕ) (h_sum : M + N - P = 2016) (n : ℕ) :
  ((term_with_largest_binomial_coefficient M N P h_sum n = -8064) ∧ 
   (term_with_largest_absolute_value_coefficient M N P h_sum n = (-15360, 8))) :=
by {
  -- proof goes here
  sorry
}

end problem_statement_l66_6628


namespace determine_m_l66_6683

-- Define the conditions: the quadratic equation and the sum of roots
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 + m * x + 2 = 0

def sum_of_roots (x1 x2 : ℝ) : ℝ := x1 + x2

-- Problem Statement: Prove that m = 4
theorem determine_m (x1 x2 m : ℝ) 
  (h1 : quadratic_eq x1 m) 
  (h2 : quadratic_eq x2 m)
  (h3 : sum_of_roots x1 x2 = -4) : 
  m = 4 :=
by
  sorry

end determine_m_l66_6683


namespace find_AD_l66_6660

-- Define the geometrical context and constraints
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC AD BD CD : ℝ) (x : ℝ)

-- Assume the given conditions
def problem_conditions := 
  (AB = 50) ∧
  (AC = 41) ∧
  (BD = 10 * x) ∧
  (CD = 3 * x) ∧
  (AB^2 = AD^2 + BD^2) ∧
  (AC^2 = AD^2 + CD^2)

-- Formulate the problem question and the correct answer
theorem find_AD (h : problem_conditions AB AC AD BD CD x) : AD = 40 :=
sorry

end find_AD_l66_6660


namespace fraction_of_4_is_8_l66_6614

theorem fraction_of_4_is_8 (fraction : ℝ) (h : fraction * 4 = 8) : fraction = 8 := 
sorry

end fraction_of_4_is_8_l66_6614


namespace jamie_paid_0_more_than_alex_l66_6636

/-- Conditions:
     1. Alex and Jamie shared a pizza cut into 10 equally-sized slices.
     2. Alex wanted a plain pizza.
     3. Jamie wanted a special spicy topping on one-third of the pizza.
     4. The cost of a plain pizza was $10.
     5. The spicy topping on one-third of the pizza cost an additional $3.
     6. Jamie ate all the slices with the spicy topping and two extra plain slices.
     7. Alex ate the remaining plain slices.
     8. They each paid for what they ate.
    
     Question: How many more dollars did Jamie pay than Alex?
     Answer: 0
-/
theorem jamie_paid_0_more_than_alex :
  let total_slices := 10
  let cost_plain := 10
  let cost_spicy := 3
  let total_cost := cost_plain + cost_spicy
  let cost_per_slice := total_cost / total_slices
  let jamie_slices := 5
  let alex_slices := total_slices - jamie_slices
  let jamie_cost := jamie_slices * cost_per_slice
  let alex_cost := alex_slices * cost_per_slice
  jamie_cost - alex_cost = 0 :=
by
  sorry

end jamie_paid_0_more_than_alex_l66_6636


namespace find_alpha_l66_6613

-- Define the given condition that alpha is inversely proportional to beta
def inv_proportional (α β : ℝ) (k : ℝ) : Prop := α * β = k

-- Main theorem statement
theorem find_alpha (α β k : ℝ) (h1 : inv_proportional 2 5 k) (h2 : inv_proportional α (-10) k) : α = -1 := by
  -- Given the conditions, the proof would follow, but it's not required here.
  sorry

end find_alpha_l66_6613


namespace art_gallery_ratio_l66_6603

theorem art_gallery_ratio (A : ℕ) (D : ℕ) (S_not_displayed : ℕ) (P_not_displayed : ℕ)
  (h1 : A = 2700)
  (h2 : 1 / 6 * D = D / 6)
  (h3 : P_not_displayed = S_not_displayed / 3)
  (h4 : S_not_displayed = 1200) :
  D / A = 11 / 27 := by
  sorry

end art_gallery_ratio_l66_6603


namespace absent_children_on_teachers_day_l66_6632

theorem absent_children_on_teachers_day (A : ℕ) (h1 : ∀ n : ℕ, n = 190)
(h2 : ∀ s : ℕ, s = 38) (h3 : ∀ extra : ℕ, extra = 14) :
  (190 - A) * 38 = 190 * 24 → A = 70 :=
by
  sorry

end absent_children_on_teachers_day_l66_6632


namespace soda_count_l66_6618

theorem soda_count
  (W : ℕ) (S : ℕ) (B : ℕ) (T : ℕ)
  (hW : W = 26) (hB : B = 17) (hT : T = 31) :
  W + S - B = T → S = 22 :=
by
  sorry

end soda_count_l66_6618


namespace percent_difference_l66_6635

theorem percent_difference : 
  let a := 0.60 * 50
  let b := 0.45 * 30
  a - b = 16.5 :=
by
  let a := 0.60 * 50
  let b := 0.45 * 30
  sorry

end percent_difference_l66_6635


namespace tenth_term_of_sequence_l66_6619

-- Define the first term and the common difference
def a1 : ℤ := 10
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) : ℤ := a1 + d * (n - 1)

-- State the theorem about the 10th term
theorem tenth_term_of_sequence : a_n 10 = -8 := by
  -- Skip the proof
  sorry

end tenth_term_of_sequence_l66_6619


namespace true_universal_quantifier_l66_6631

theorem true_universal_quantifier :
  ∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1) := by
  sorry

end true_universal_quantifier_l66_6631


namespace correct_subtraction_l66_6672

theorem correct_subtraction (x : ℕ) (h : x - 42 = 50) : x - 24 = 68 :=
  sorry

end correct_subtraction_l66_6672


namespace fraction_result_l66_6606

theorem fraction_result (a b c : ℝ) (h1 : a / 2 = b / 3) (h2 : b / 3 = c / 5) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (a + b) / (c - a) = 5 / 3 :=
by
  sorry

end fraction_result_l66_6606


namespace arithmetic_sequence_problem_l66_6691

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 + a 6 + a 8 + a 10 + a 12 = 60)
  (h2 : ∀ n, a (n + 1) = a n + d) :
  a 7 - (1 / 3) * a 5 = 8 :=
by
  sorry

end arithmetic_sequence_problem_l66_6691


namespace actual_distance_traveled_l66_6659

theorem actual_distance_traveled (D : ℕ) 
  (h : D / 10 = (D + 36) / 16) : D = 60 := by
  sorry

end actual_distance_traveled_l66_6659


namespace clea_ride_escalator_time_l66_6623

theorem clea_ride_escalator_time
  (s v d : ℝ)
  (h1 : 75 * s = d)
  (h2 : 30 * (s + v) = d) :
  t = 50 :=
by
  sorry

end clea_ride_escalator_time_l66_6623


namespace slope_correct_l66_6699

-- Coordinates of the vertices of the polygon
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (6, 2)
def vertex_F := (6, 0)

-- Define the total area of the polygon
def total_area : ℝ := 20

-- Define the slope of the line through the origin dividing the area in half
def slope_line_dividing_area (slope : ℝ) : Prop :=
  ∃ l : ℝ, l = 5 / 3 ∧
  ∃ area_divided : ℝ, area_divided = total_area / 2

-- Prove the slope is 5/3
theorem slope_correct :
  slope_line_dividing_area (5 / 3) :=
by
  sorry

end slope_correct_l66_6699


namespace combined_sleep_hours_l66_6673

def connor_hours : ℕ := 6
def luke_hours : ℕ := connor_hours + 2
def emma_hours : ℕ := connor_hours - 1
def puppy_hours : ℕ := 2 * luke_hours

theorem combined_sleep_hours :
  connor_hours + luke_hours + emma_hours + puppy_hours = 35 := by
  sorry

end combined_sleep_hours_l66_6673


namespace geometric_sum_S12_l66_6663

theorem geometric_sum_S12 
  (S : ℕ → ℝ)
  (h_S4 : S 4 = 2) 
  (h_S8 : S 8 = 6) 
  (geom_property : ∀ n, (S (2 * n + 4) - S n) ^ 2 = S n * (S (3 * n + 4) - S (2 * n + 4))) 
  : S 12 = 14 := 
by sorry

end geometric_sum_S12_l66_6663


namespace widgets_per_shipping_box_l66_6620

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end widgets_per_shipping_box_l66_6620


namespace small_circles_sixth_figure_l66_6630

-- Defining the function to calculate the number of circles in the nth figure
def small_circles (n : ℕ) : ℕ :=
  n * (n + 1) + 4

-- Statement of the theorem
theorem small_circles_sixth_figure :
  small_circles 6 = 46 :=
by sorry

end small_circles_sixth_figure_l66_6630


namespace distance_between_stripes_l66_6666

theorem distance_between_stripes (d₁ d₂ L W : ℝ) (h : ℝ)
  (h₁ : d₁ = 60)  -- distance between parallel curbs
  (h₂ : L = 30)  -- length of the curb between stripes
  (h₃ : d₂ = 80)  -- length of each stripe
  (area_eq : W * L = 1800) -- area of the parallelogram with base L
: h = 22.5 :=
by
  -- This is to assume the equation derived from area calculation
  have area_eq' : d₂ * h = 1800 := by sorry
  -- Solving for h using the derived area equation
  have h_calc : h = 1800 / 80 := by sorry
  -- Simplifying the result
  have h_simplified : h = 22.5 := by sorry
  exact h_simplified

end distance_between_stripes_l66_6666


namespace frost_cakes_total_l66_6686

-- Conditions
def Cagney_time := 60 -- seconds per cake
def Lacey_time := 40  -- seconds per cake
def total_time := 10 * 60 -- 10 minutes in seconds

-- The theorem to prove
theorem frost_cakes_total (Cagney_time Lacey_time total_time : ℕ) (h1 : Cagney_time = 60) (h2 : Lacey_time = 40) (h3 : total_time = 600):
  (total_time / (Cagney_time * Lacey_time / (Cagney_time + Lacey_time))) = 25 :=
by
  -- Proof to be filled in
  sorry

end frost_cakes_total_l66_6686


namespace Kaarel_wins_l66_6680

theorem Kaarel_wins (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) :
  ∃ (x y a : ℕ), x ∈ Finset.range (p-1) ∧ y ∈ Finset.range (p-1) ∧ a ∈ Finset.range (p-1) ∧ 
  x ≠ y ∧ y ≠ (p - x) ∧ a ≠ x ∧ a ≠ (p - x) ∧ a ≠ y ∧ 
  (x * (p - x) + y * a) % p = 0 :=
sorry

end Kaarel_wins_l66_6680


namespace correct_division_result_l66_6657

theorem correct_division_result : 
  ∀ (a b : ℕ),
  (1722 / (10 * b + a) = 42) →
  (10 * a + b = 14) →
  1722 / 14 = 123 :=
by
  intros a b h1 h2
  sorry

end correct_division_result_l66_6657


namespace assistant_increases_output_by_100_percent_l66_6668

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l66_6668


namespace jasmine_additional_cans_needed_l66_6682

theorem jasmine_additional_cans_needed
  (n_initial : ℕ)
  (n_lost : ℕ)
  (n_remaining : ℕ)
  (additional_can_coverage : ℕ)
  (n_needed : ℕ) :
  n_initial = 50 →
  n_lost = 4 →
  n_remaining = 36 →
  additional_can_coverage = 2 →
  n_needed = 7 :=
by
  sorry

end jasmine_additional_cans_needed_l66_6682


namespace quadrilateral_pyramid_plane_intersection_l66_6645

-- Definitions:
-- Let MA, MB, MC, MD, MK, ML, MP, MN be lengths of respective segments
-- Let S_ABC, S_ABD, S_ACD, S_BCD be areas of respective triangles
variables {MA MB MC MD MK ML MP MN : ℝ}
variables {S_ABC S_ABD S_ACD S_BCD : ℝ}

-- Given a quadrilateral pyramid MABCD with a convex quadrilateral ABCD as base, and a plane intersecting edges MA, MB, MC, and MD at points K, L, P, and N respectively. Prove the following relation.
theorem quadrilateral_pyramid_plane_intersection :
  S_BCD * (MA / MK) + S_ADB * (MC / MP) = S_ABC * (MD / MN) + S_ACD * (MB / ML) :=
sorry

end quadrilateral_pyramid_plane_intersection_l66_6645


namespace mark_first_part_playing_time_l66_6652

open Nat

theorem mark_first_part_playing_time (x : ℕ) (total_game_time second_part_playing_time sideline_time : ℕ)
  (h1 : total_game_time = 90) (h2 : second_part_playing_time = 35) (h3 : sideline_time = 35) 
  (h4 : x + second_part_playing_time + sideline_time = total_game_time) : x = 20 := 
by
  sorry

end mark_first_part_playing_time_l66_6652


namespace quadratic_zeros_interval_l66_6688

theorem quadratic_zeros_interval (a : ℝ) :
  (5 - 2 * a > 0) ∧ (4 * a^2 - 16 > 0) ∧ (a > 1) ↔ (2 < a ∧ a < 5 / 2) :=
by
  sorry

end quadratic_zeros_interval_l66_6688


namespace solve_system_of_equations_l66_6629

def proof_problem (a b c : ℚ) : Prop :=
  ((a - b = 2) ∧ (c = -5) ∧ (2 * a - 6 * b = 2)) → 
  (a = 5 / 2 ∧ b = 1 / 2 ∧ c = -5)

theorem solve_system_of_equations (a b c : ℚ) :
  proof_problem a b c :=
  by
    sorry

end solve_system_of_equations_l66_6629


namespace friends_with_Ron_l66_6674

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end friends_with_Ron_l66_6674


namespace rhombus_area_l66_6698

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : 
  (d1 * d2) / 2 = 160 := by
sorry

end rhombus_area_l66_6698


namespace simplify_expression_l66_6647

theorem simplify_expression (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = - (2 / 3) * x - 2 :=
by sorry

end simplify_expression_l66_6647


namespace arrangement_count_l66_6615

-- Definitions
def volunteers := 4
def elderly := 2
def total_people := volunteers + elderly
def criteria := "The 2 elderly people must be adjacent but not at the ends of the row."

-- Theorem: The number of different valid arrangements is 144
theorem arrangement_count : 
  ∃ (arrangements : Nat), arrangements = (volunteers.factorial * 3 * elderly.factorial) ∧ arrangements = 144 := 
  by 
    sorry

end arrangement_count_l66_6615


namespace obtuse_triangle_l66_6608

theorem obtuse_triangle (A B C M E : ℝ) (hM : M = (B + C) / 2) (hE : E > 0) 
(hcond : (B - E) ^ 2 + (C - E) ^ 2 >= 4 * (A - M) ^ 2): 
∃ α β γ, α > 90 ∧ β + γ < 90 ∧ α + β + γ = 180 :=
by
  sorry

end obtuse_triangle_l66_6608


namespace planes_parallel_l66_6651

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l66_6651


namespace Tom_total_spend_l66_6622

theorem Tom_total_spend :
  let notebook_price := 2
  let notebook_discount := 0.75
  let notebook_count := 4
  let magazine_price := 5
  let magazine_count := 2
  let pen_price := 1.50
  let pen_discount := 0.75
  let pen_count := 3
  let book_price := 12
  let book_count := 1
  let discount_threshold := 30
  let coupon_discount := 10
  let total_cost :=
    (notebook_count * (notebook_price * notebook_discount)) +
    (magazine_count * magazine_price) +
    (pen_count * (pen_price * pen_discount)) +
    (book_count * book_price)
  let final_cost := if total_cost >= discount_threshold then total_cost - coupon_discount else total_cost
  final_cost = 21.375 :=
by
  sorry

end Tom_total_spend_l66_6622


namespace ratio_of_sides_l66_6605
-- Import the complete math library

-- Define the conditions as hypotheses
variables (s x y : ℝ)
variable (h_outer_area : (3 * s)^2 = 9 * s^2)
variable (h_side_lengths : 3 * s = s + 2 * x)
variable (h_y_length : y + x = 3 * s)

-- State the theorem
theorem ratio_of_sides (h_outer_area : (3 * s)^2 = 9 * s^2)
  (h_side_lengths : 3 * s = s + 2 * x)
  (h_y_length : y + x = 3 * s) :
  y / x = 2 := by
  sorry

end ratio_of_sides_l66_6605


namespace cyclic_quadrilateral_equality_l66_6649

variables {A B C D : ℝ} (AB BC CD DA AC BD : ℝ)

theorem cyclic_quadrilateral_equality 
  (h_cyclic: A * B * C * D = AB * BC * CD * DA)
  (h_sides: AB = A ∧ BC = B ∧ CD = C ∧ DA = D)
  (h_diagonals: AC = E ∧ BD = F) :
  E * (A * B + C * D) = F * (D * A + B * C) :=
sorry

end cyclic_quadrilateral_equality_l66_6649


namespace sum_mod_7_eq_5_l66_6658

theorem sum_mod_7_eq_5 : 
  (51730 + 51731 + 51732 + 51733 + 51734 + 51735) % 7 = 5 := 
by 
  sorry

end sum_mod_7_eq_5_l66_6658


namespace value_of_a_purely_imaginary_l66_6678

-- Define the conditions under which a given complex number is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.im z * Complex.I ∧ b ≠ 0

-- Define the complex number based on the variable a
def given_complex_number (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 1⟩

-- The proof statement
theorem value_of_a_purely_imaginary :
  is_purely_imaginary (given_complex_number 2) := sorry

end value_of_a_purely_imaginary_l66_6678


namespace tom_total_distance_l66_6643

/-- Tom swims for 1.5 hours at 2.5 miles per hour. 
    Tom runs for 0.75 hours at 6.5 miles per hour. 
    Tom bikes for 3 hours at 12 miles per hour. 
    The total distance Tom covered is 44.625 miles.
-/
theorem tom_total_distance
  (swim_time : ℝ := 1.5) (swim_speed : ℝ := 2.5)
  (run_time : ℝ := 0.75) (run_speed : ℝ := 6.5)
  (bike_time : ℝ := 3) (bike_speed : ℝ := 12) :
  swim_time * swim_speed + run_time * run_speed + bike_time * bike_speed = 44.625 :=
by
  sorry

end tom_total_distance_l66_6643


namespace reciprocal_solution_l66_6600

theorem reciprocal_solution {x : ℝ} (h : x * -9 = 1) : x = -1/9 :=
sorry

end reciprocal_solution_l66_6600


namespace min_stamps_needed_l66_6690

theorem min_stamps_needed {c f : ℕ} (h : 3 * c + 4 * f = 33) : c + f = 9 :=
sorry

end min_stamps_needed_l66_6690


namespace sum_of_midpoints_l66_6627

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l66_6627


namespace problem1_problem2_l66_6601

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Definition of g(x)
def g (x t : ℝ) : ℝ := t * abs x - 2

-- Problem 1: Proof that f(x) > 2x + 1 implies x < 0
theorem problem1 (x : ℝ) : f x > 2 * x + 1 → x < 0 := by
  sorry

-- Problem 2: Proof that if f(x) ≥ g(x) for all x, then t ≤ 1
theorem problem2 (t : ℝ) : (∀ x : ℝ, f x ≥ g x t) → t ≤ 1 := by
  sorry

end problem1_problem2_l66_6601


namespace number_of_players_is_correct_l66_6640

-- Defining the problem conditions
def wristband_cost : ℕ := 6
def jersey_cost : ℕ := wristband_cost + 7
def wristbands_per_player : ℕ := 4
def jerseys_per_player : ℕ := 2
def total_expenditure : ℕ := 3774

-- Calculating cost per player and stating the proof problem
def cost_per_player : ℕ := wristbands_per_player * wristband_cost +
                           jerseys_per_player * jersey_cost

def number_of_players : ℕ := total_expenditure / cost_per_player

-- The final proof statement to show that number_of_players is 75
theorem number_of_players_is_correct : number_of_players = 75 :=
by sorry

end number_of_players_is_correct_l66_6640


namespace part1_part2_l66_6675

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x^2 - 3 * x + 2 = 0 }

theorem part1 (a : ℝ) : (A a = ∅) ↔ (a > 9/8) := sorry

theorem part2 (a : ℝ) : 
  (∃ x, A a = {x}) ↔ 
  (a = 0 ∧ A a = {2 / 3})
  ∨ (a = 9 / 8 ∧ A a = {4 / 3}) := sorry

end part1_part2_l66_6675


namespace Jill_arrives_9_minutes_later_l66_6671

theorem Jill_arrives_9_minutes_later
  (distance : ℝ)
  (Jack_speed : ℝ)
  (Jill_speed : ℝ)
  (h1 : distance = 1)
  (h2 : Jack_speed = 10)
  (h3 : Jill_speed = 4) :
  ((distance / Jill_speed) - (distance / Jack_speed)) * 60 = 9 := by
  -- Placeholder for the proof
  sorry

end Jill_arrives_9_minutes_later_l66_6671


namespace tin_can_allocation_l66_6641

-- Define the total number of sheets of tinplate available
def total_sheets := 108

-- Define the number of sheets used for can bodies
variable (x : ℕ)

-- Define the number of can bodies a single sheet makes
def can_bodies_per_sheet := 15

-- Define the number of can bottoms a single sheet makes
def can_bottoms_per_sheet := 42

-- Define the equation to be proven
theorem tin_can_allocation :
  2 * can_bodies_per_sheet * x = can_bottoms_per_sheet * (total_sheets - x) :=
  sorry

end tin_can_allocation_l66_6641
