import Mathlib

namespace find_original_number_l804_80441

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l804_80441


namespace sandwiches_consumption_difference_l804_80409

theorem sandwiches_consumption_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let combined_monday_tuesday := monday_total + tuesday_total

  combined_monday_tuesday - wednesday_total = -5 :=
by
  sorry

end sandwiches_consumption_difference_l804_80409


namespace number_of_triangles_with_perimeter_27_l804_80455

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end number_of_triangles_with_perimeter_27_l804_80455


namespace B_more_than_C_l804_80440

variables (A B C : ℕ)
noncomputable def total_subscription : ℕ := 50000
noncomputable def total_profit : ℕ := 35000
noncomputable def A_profit : ℕ := 14700
noncomputable def A_subscr : ℕ := B + 4000

theorem B_more_than_C (B_subscr C_subscr : ℕ) (h1 : A_subscr + B_subscr + C_subscr = total_subscription)
    (h2 : 14700 * 50000 = 35000 * A_subscr) :
    B_subscr - C_subscr = 5000 :=
sorry

end B_more_than_C_l804_80440


namespace number_of_square_tiles_l804_80454

theorem number_of_square_tiles (a b : ℕ) (h1 : a + b = 32) (h2 : 3 * a + 4 * b = 110) : b = 14 :=
by
  -- the proof steps are skipped
  sorry

end number_of_square_tiles_l804_80454


namespace binary_111_is_7_l804_80449

def binary_to_decimal (b0 b1 b2 : ℕ) : ℕ :=
  b0 * (2^0) + b1 * (2^1) + b2 * (2^2)

theorem binary_111_is_7 : binary_to_decimal 1 1 1 = 7 :=
by
  -- We will provide the proof here.
  sorry

end binary_111_is_7_l804_80449


namespace change_factor_w_l804_80420

theorem change_factor_w (w d z F_w : Real)
  (h_q : ∀ w d z, q = 5 * w / (4 * d * z^2))
  (h1 : d' = 2 * d)
  (h2 : z' = 3 * z)
  (h3 : F_q = 0.2222222222222222)
  : F_w = 4 :=
by
  sorry

end change_factor_w_l804_80420


namespace find_square_sum_l804_80496

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end find_square_sum_l804_80496


namespace train_crossing_time_l804_80450

theorem train_crossing_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) :
  length_of_train = 1500 → speed_of_train = 95 → speed_of_man = 5 → 
  (length_of_train / ((speed_of_train - speed_of_man) * (1000 / 3600))) = 60 :=
by
  intros h1 h2 h3
  have h_rel_speed : ((speed_of_train - speed_of_man) * (1000 / 3600)) = 25 := by
    rw [h2, h3]
    norm_num
  rw [h1, h_rel_speed]
  norm_num

end train_crossing_time_l804_80450


namespace bucket_weight_full_l804_80433

variable (c d : ℝ)

theorem bucket_weight_full (h1 : ∃ x y, x + (1 / 4) * y = c)
                           (h2 : ∃ x y, x + (3 / 4) * y = d) :
  ∃ x y, x + y = (3 * d - c) / 2 :=
by
  sorry

end bucket_weight_full_l804_80433


namespace Lee_payment_total_l804_80423

theorem Lee_payment_total 
  (ticket_price : ℝ := 10.00)
  (booking_fee : ℝ := 1.50)
  (youngest_discount : ℝ := 0.40)
  (oldest_discount : ℝ := 0.30)
  (middle_discount : ℝ := 0.20)
  (youngest_tickets : ℕ := 3)
  (oldest_tickets : ℕ := 3)
  (middle_tickets : ℕ := 4) :
  (youngest_tickets * (ticket_price * (1 - youngest_discount)) + 
   oldest_tickets * (ticket_price * (1 - oldest_discount)) + 
   middle_tickets * (ticket_price * (1 - middle_discount)) + 
   (youngest_tickets + oldest_tickets + middle_tickets) * booking_fee) = 86.00 :=
by 
  sorry

end Lee_payment_total_l804_80423


namespace number_replacement_l804_80403

theorem number_replacement :
  ∃ x : ℝ, ( (x / (1 / 2) * x) / (x * (1 / 2) / x) = 25 ) ↔ x = 2.5 :=
by 
  sorry

end number_replacement_l804_80403


namespace determine_y_l804_80482

theorem determine_y (y : ℝ) (h1 : 0 < y) (h2 : y * (⌊y⌋ : ℝ) = 90) : y = 10 :=
sorry

end determine_y_l804_80482


namespace probability_of_selection_l804_80491

-- defining necessary parameters and the systematic sampling method
def total_students : ℕ := 52
def selected_students : ℕ := 10
def exclusion_probability := 2 / total_students
def inclusion_probability_exclude := selected_students / (total_students - 2)
def final_probability := (1 - exclusion_probability) * inclusion_probability_exclude

-- the main theorem stating the probability calculation
theorem probability_of_selection :
  final_probability = 5 / 26 :=
by
  -- we skip the proof part and end with sorry since it is not required
  sorry

end probability_of_selection_l804_80491


namespace largest_K_inequality_l804_80498

theorem largest_K_inequality :
  ∃ K : ℕ, (K < 12) ∧ (10 * K = 110) := by
  use 11
  sorry

end largest_K_inequality_l804_80498


namespace parallel_lines_iff_m_eq_neg2_l804_80416

theorem parallel_lines_iff_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0 → m * x + 2 * y - m + 2 = 0 ↔ m = -2) :=
sorry

end parallel_lines_iff_m_eq_neg2_l804_80416


namespace solution_set_l804_80429

def op (a b : ℝ) : ℝ := -2 * a + b

theorem solution_set (x : ℝ) : (op x 4 > 0) ↔ (x < 2) :=
by {
  -- proof required here
  sorry
}

end solution_set_l804_80429


namespace total_potatoes_l804_80417

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes_l804_80417


namespace johns_original_earnings_l804_80467

-- Define the conditions
def raises (original : ℝ) (percentage : ℝ) := original + original * percentage

-- The theorem stating the equivalent problem proof
theorem johns_original_earnings :
  ∃ (x : ℝ), raises x 0.375 = 55 ↔ x = 40 :=
sorry

end johns_original_earnings_l804_80467


namespace math_problem_proof_l804_80488

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end math_problem_proof_l804_80488


namespace contrapositive_even_sum_l804_80462

theorem contrapositive_even_sum (a b : ℕ) :
  (¬(a % 2 = 0 ∧ b % 2 = 0) → ¬(a + b) % 2 = 0) ↔ (¬((a + b) % 2 = 0) → ¬(a % 2 = 0 ∧ b % 2 = 0)) :=
by
  sorry

end contrapositive_even_sum_l804_80462


namespace derivative_of_f_tangent_line_at_pi_l804_80443

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = (x * Real.cos x - Real.sin x) / (x ^ 2) :=
  sorry

theorem tangent_line_at_pi : 
  let M := (Real.pi, 0)
  let slope := -1 / Real.pi
  let tangent_line (x : ℝ) : ℝ := -x / Real.pi + 1
  ∀ (x y : ℝ), (x, y) = M → y = tangent_line x :=
  sorry

end derivative_of_f_tangent_line_at_pi_l804_80443


namespace intercepts_l804_80418

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end intercepts_l804_80418


namespace cupcake_packages_l804_80425

theorem cupcake_packages (total_cupcakes eaten_cupcakes cupcakes_per_package number_of_packages : ℕ) 
  (h1 : total_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : cupcakes_per_package = 2)
  (h4 : number_of_packages = (total_cupcakes - eaten_cupcakes) / cupcakes_per_package) :
  number_of_packages = 5 :=
by
  -- The proof goes here, we'll use sorry to indicate it's not needed for now.
  sorry

end cupcake_packages_l804_80425


namespace seating_arrangement_ways_l804_80493

-- Define the problem conditions in Lean 4
def number_of_ways_to_seat (total_chairs : ℕ) (total_people : ℕ) := 
  Nat.factorial total_chairs / Nat.factorial (total_chairs - total_people)

-- Define the specific theorem to be proved
theorem seating_arrangement_ways : number_of_ways_to_seat 8 5 = 6720 :=
by
  sorry

end seating_arrangement_ways_l804_80493


namespace no_conditions_satisfy_l804_80474

-- Define the conditions
def condition1 (a b c : ℤ) : Prop := a = 1 ∧ b = 1 ∧ c = 1
def condition2 (a b c : ℤ) : Prop := a = b - 1 ∧ b = c - 1
def condition3 (a b c : ℤ) : Prop := a = b ∧ b = c
def condition4 (a b c : ℤ) : Prop := a > c ∧ c = b - 1 

-- Define the equations
def equation1 (a b c : ℤ) : ℤ := a * (a - b)^3 + b * (b - c)^3 + c * (c - a)^3
def equation2 (a b c : ℤ) : Prop := a + b + c = 3

-- Proof statement for the original problem
theorem no_conditions_satisfy (a b c : ℤ) :
  ¬ (condition1 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition2 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition3 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition4 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) :=
sorry

end no_conditions_satisfy_l804_80474


namespace bricks_required_l804_80463

-- Courtyard dimensions in meters
def length_courtyard_m := 23
def width_courtyard_m := 15

-- Brick dimensions in centimeters
def length_brick_cm := 17
def width_brick_cm := 9

-- Conversion from meters to centimeters
def meter_to_cm (m : Int) : Int :=
  m * 100

-- Area of courtyard in square centimeters
def area_courtyard_cm2 : Int :=
  meter_to_cm length_courtyard_m * meter_to_cm width_courtyard_m

-- Area of a single brick in square centimeters
def area_brick_cm2 : Int :=
  length_brick_cm * width_brick_cm

-- Calculate the number of bricks needed, ensuring we round up to the nearest whole number
def total_bricks_needed : Int :=
  (area_courtyard_cm2 + area_brick_cm2 - 1) / area_brick_cm2

-- The theorem stating the total number of bricks needed
theorem bricks_required :
  total_bricks_needed = 22550 := by
  sorry

end bricks_required_l804_80463


namespace complex_expression_identity_l804_80438

open Complex

theorem complex_expression_identity
  (x y : ℂ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x^2 + x * y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 :=
by
  sorry

end complex_expression_identity_l804_80438


namespace midpoint_range_l804_80427

variable {x0 y0 : ℝ}

-- Conditions
def point_on_line1 (P : ℝ × ℝ) := P.1 + 2 * P.2 - 1 = 0
def point_on_line2 (Q : ℝ × ℝ) := Q.1 + 2 * Q.2 + 3 = 0
def is_midpoint (P Q M : ℝ × ℝ) := P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2
def midpoint_condition (M : ℝ × ℝ) := M.2 > M.1 + 2

-- Theorem
theorem midpoint_range
  (P Q M : ℝ × ℝ)
  (hP : point_on_line1 P)
  (hQ : point_on_line2 Q)
  (hM : is_midpoint P Q M)
  (h_cond : midpoint_condition M)
  (hx0 : x0 = M.1)
  (hy0 : y0 = M.2)
  : - (1 / 2) < y0 / x0 ∧ y0 / x0 < - (1 / 5) :=
sorry

end midpoint_range_l804_80427


namespace find_a_m_18_l804_80472

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (a1 : ℝ)
variable (m : ℕ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) :=
  ∀ n : ℕ, a n = a1 * r^n

def problem_conditions (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :=
  (geometric_sequence a a1 r) ∧
  a m = 3 ∧
  a (m + 6) = 24

theorem find_a_m_18 (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :
  problem_conditions a r a1 m → a (m + 18) = 1536 :=
by
  sorry

end find_a_m_18_l804_80472


namespace remainder_sum_div_7_l804_80487

theorem remainder_sum_div_7 :
  (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 :=
by
  sorry

end remainder_sum_div_7_l804_80487


namespace zeros_indeterminate_in_interval_l804_80476

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (ha : a < b) (hf : f a * f b < 0)

-- The theorem statement
theorem zeros_indeterminate_in_interval :
  (∀ (f : ℝ → ℝ), f a * f b < 0 → (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∨ (∀ (x : ℝ), a < x ∧ x < b → f x ≠ 0) ∨ (∃ (x1 x2 : ℝ), a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f x1 = 0 ∧ f x2 = 0)) :=
by sorry

end zeros_indeterminate_in_interval_l804_80476


namespace leftmost_square_side_length_l804_80421

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l804_80421


namespace sqrt_14_range_l804_80477

theorem sqrt_14_range : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 :=
by
  -- We know that 9 < 14 < 16, so we can take the square root of all parts to get 3 < sqrt(14) < 4.
  sorry

end sqrt_14_range_l804_80477


namespace marks_in_biology_l804_80405

theorem marks_in_biology (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) (marks_chemistry : ℕ) (average_marks : ℕ) :
  marks_english = 73 → marks_math = 69 → marks_physics = 92 → marks_chemistry = 64 → average_marks = 76 →
  (380 - (marks_english + marks_math + marks_physics + marks_chemistry)) = 82 :=
by
  intros
  sorry

end marks_in_biology_l804_80405


namespace distance_between_Petrovo_and_Nikolaevo_l804_80458

theorem distance_between_Petrovo_and_Nikolaevo :
  ∃ S : ℝ, (10 + (S - 10) / 4) + (20 + (S - 20) / 3) = S ∧ S = 50 := by
    sorry

end distance_between_Petrovo_and_Nikolaevo_l804_80458


namespace probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l804_80407

-- Define the total number of ways to choose 3 leaders from 6 students
def total_ways : ℕ := Nat.choose 6 3

-- Calculate the number of ways in which boy A or girl B is chosen
def boy_A_chosen_ways : ℕ := Nat.choose 4 2 + 4 * 2
def girl_B_chosen_ways : ℕ := Nat.choose 4 1 + Nat.choose 4 2
def either_boy_A_or_girl_B_chosen_ways : ℕ := boy_A_chosen_ways + girl_B_chosen_ways

-- Calculate the probability that either boy A or girl B is chosen
def probability_either_boy_A_or_girl_B : ℚ := either_boy_A_or_girl_B_chosen_ways / total_ways

-- Calculate the probability that girl B is chosen
def girl_B_total_ways : ℕ := Nat.choose 5 2
def probability_B : ℚ := girl_B_total_ways / total_ways

-- Calculate the probability that both boy A and girl B are chosen
def both_A_and_B_chosen_ways : ℕ := Nat.choose 4 1
def probability_AB : ℚ := both_A_and_B_chosen_ways / total_ways

-- Calculate the conditional probability P(A|B) given P(B)
def conditional_probability_A_given_B : ℚ := probability_AB / probability_B

-- Theorem statements
theorem probability_either_boy_A_or_girl_B_correct : probability_either_boy_A_or_girl_B = (4 / 5) := sorry
theorem probability_B_correct : probability_B = (1 / 2) := sorry
theorem conditional_probability_A_given_B_correct : conditional_probability_A_given_B = (2 / 5) := sorry

end probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l804_80407


namespace smaller_of_two_numbers_l804_80480

theorem smaller_of_two_numbers (a b : ℕ) (h1 : a * b = 4761) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 53 :=
by {
  sorry -- proof skips as directed
}

end smaller_of_two_numbers_l804_80480


namespace largest_percentage_increase_is_2013_to_2014_l804_80495

-- Defining the number of students in each year as constants
def students_2010 : ℕ := 50
def students_2011 : ℕ := 56
def students_2012 : ℕ := 62
def students_2013 : ℕ := 68
def students_2014 : ℕ := 77
def students_2015 : ℕ := 81

-- Defining the percentage increase between consecutive years
def percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / (a : ℚ)

-- Calculating all the percentage increases
def pi_2010_2011 := percentage_increase students_2010 students_2011
def pi_2011_2012 := percentage_increase students_2011 students_2012
def pi_2012_2013 := percentage_increase students_2012 students_2013
def pi_2013_2014 := percentage_increase students_2013 students_2014
def pi_2014_2015 := percentage_increase students_2014 students_2015

-- The theorem stating the largest percentage increase is between 2013 and 2014
theorem largest_percentage_increase_is_2013_to_2014 :
  max (pi_2010_2011) (max (pi_2011_2012) (max (pi_2012_2013) (max (pi_2013_2014) (pi_2014_2015)))) = pi_2013_2014 :=
sorry

end largest_percentage_increase_is_2013_to_2014_l804_80495


namespace robin_albums_l804_80412

theorem robin_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums_created : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : pics_per_album = 8)
  (h4 : total_pics = phone_pics + camera_pics)
  (h5 : albums_created = total_pics / pics_per_album) : albums_created = 5 := 
sorry

end robin_albums_l804_80412


namespace find_a_l804_80486

noncomputable def polynomial1 (x : ℝ) : ℝ := x^3 + 3 * x^2 - x - 3
noncomputable def polynomial2 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_a (a : ℝ) (x : ℝ) (hx1 : polynomial1 x > 0)
  (hx2 : polynomial2 x a ≤ 0) (ha : a > 0) : 
  3 / 4 ≤ a ∧ a < 4 / 3 :=
sorry

end find_a_l804_80486


namespace miles_per_gallon_city_l804_80461

theorem miles_per_gallon_city
  (T : ℝ) -- tank size
  (h c : ℝ) -- miles per gallon on highway 'h' and in the city 'c'
  (h_eq : h = (462 / T))
  (c_eq : c = (336 / T))
  (relation : c = h - 9)
  (solution : c = 24) : c = 24 := 
sorry

end miles_per_gallon_city_l804_80461


namespace milk_remaining_l804_80406

def initial_whole_milk := 15
def initial_low_fat_milk := 12
def initial_almond_milk := 8

def jason_buys := 5
def jason_promotion := 2 -- every 2 bottles he gets 1 free

def harry_buys_low_fat := 4
def harry_gets_free_low_fat := 1
def harry_buys_almond := 2

theorem milk_remaining : 
  (initial_whole_milk - jason_buys = 10) ∧ 
  (initial_low_fat_milk - (harry_buys_low_fat + harry_gets_free_low_fat) = 7) ∧ 
  (initial_almond_milk - harry_buys_almond = 6) :=
by
  sorry

end milk_remaining_l804_80406


namespace k_lt_zero_l804_80475

noncomputable def k_negative (k : ℝ) : Prop :=
  (∃ x : ℝ, x < 0 ∧ k * x > 0) ∧ (∃ x : ℝ, x > 0 ∧ k * x < 0)

theorem k_lt_zero (k : ℝ) : k_negative k → k < 0 :=
by
  intros h
  sorry

end k_lt_zero_l804_80475


namespace total_number_of_workers_l804_80402

-- Definitions based on the given conditions
def avg_salary_total : ℝ := 8000
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def num_technicians : ℕ := 7

-- Problem statement in Lean
theorem total_number_of_workers
    (W : ℕ) (N : ℕ)
    (h1 : W * avg_salary_total = num_technicians * avg_salary_technicians + N * avg_salary_non_technicians)
    (h2 : W = num_technicians + N) :
    W = 21 :=
sorry

end total_number_of_workers_l804_80402


namespace employee_y_payment_l804_80411

variable (x y : ℝ)

def total_payment (x y : ℝ) : ℝ := x + y
def x_payment (y : ℝ) : ℝ := 1.20 * y

theorem employee_y_payment : (total_payment x y = 638) ∧ (x = x_payment y) → y = 290 :=
by
  sorry

end employee_y_payment_l804_80411


namespace complete_the_square_l804_80419

theorem complete_the_square (x : ℝ) : (x^2 + 2 * x - 1 = 0) -> ((x + 1)^2 = 2) :=
by
  intro h
  sorry

end complete_the_square_l804_80419


namespace area_of_shaded_region_l804_80464

noncomputable def area_shaded (side : ℝ) : ℝ :=
  let area_square := side * side
  let radius := side / 2
  let area_circle := Real.pi * radius * radius
  area_square - area_circle

theorem area_of_shaded_region :
  let perimeter := 28
  let side := perimeter / 4
  area_shaded side = 49 - π * 12.25 :=
by
  sorry

end area_of_shaded_region_l804_80464


namespace sum_of_digits_1197_l804_80436

theorem sum_of_digits_1197 : (1 + 1 + 9 + 7 = 18) := by sorry

end sum_of_digits_1197_l804_80436


namespace parts_per_hour_l804_80484

variables {x y : ℕ}

-- Condition 1: The time it takes for A to make 90 parts is the same as the time it takes for B to make 120 parts.
def time_ratio (x y : ℕ) := (x:ℚ) / y = 90 / 120

-- Condition 2: A and B together make 35 parts per hour.
def total_parts_per_hour (x y : ℕ) := x + y = 35

-- Given the conditions, prove the number of parts A and B each make per hour.
theorem parts_per_hour (x y : ℕ) (h1 : time_ratio x y) (h2 : total_parts_per_hour x y) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l804_80484


namespace photo_album_requirement_l804_80432

-- Definition of the conditions
def pages_per_album : ℕ := 32
def photos_per_page : ℕ := 5
def total_photos : ℕ := 900

-- Calculation of photos per album
def photos_per_album := pages_per_album * photos_per_page

-- Calculation of required albums
noncomputable def albums_needed := (total_photos + photos_per_album - 1) / photos_per_album

-- Theorem to prove the required number of albums is 6
theorem photo_album_requirement : albums_needed = 6 :=
  by sorry

end photo_album_requirement_l804_80432


namespace find_2a_plus_b_l804_80415

theorem find_2a_plus_b (a b : ℝ) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (h3 : 5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2)
  (h4 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 3) :
  2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l804_80415


namespace find_m_n_l804_80400

noncomputable def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m * x + n = 0}

theorem find_m_n (m n : ℝ) (h_union : A ∪ B m n = A) (h_inter : A ∩ B m n = {5}) :
  m = -10 ∧ n = 25 :=
by
  sorry

end find_m_n_l804_80400


namespace max_b_value_l804_80457

theorem max_b_value (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 10 :=
sorry

end max_b_value_l804_80457


namespace find_m_l804_80483

noncomputable def a_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1 : ℝ) * d)

theorem find_m (a d : ℝ) (m : ℕ) 
  (h1 : a_seq a d (m-1) + a_seq a d (m+1) - a = 0)
  (h2 : S_n a d (2*m - 1) = 38) : 
  m = 10 := 
sorry

end find_m_l804_80483


namespace new_triangle_area_l804_80447

theorem new_triangle_area (a b : ℝ) (x y : ℝ) (hypotenuse : x = a ∧ y = b ∧ x^2 + y^2 = (a + b)^2) : 
    (3  * (1 / 2) * a * b) = (3 / 2) * a * b :=
by
  sorry

end new_triangle_area_l804_80447


namespace relationship_among_a_b_c_l804_80479

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (5 * Real.pi / 6)
noncomputable def c : ℝ := Real.tan (7 * Real.pi / 5)

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l804_80479


namespace largest_possible_n_l804_80499

theorem largest_possible_n (b g : ℕ) (n : ℕ) (h1 : g = 3 * b)
  (h2 : ∀ (boy : ℕ), boy < b → ∀ (girlfriend : ℕ), girlfriend < g → girlfriend ≤ 2013)
  (h3 : ∀ (girl : ℕ), girl < g → ∀ (boyfriend : ℕ), boyfriend < b → boyfriend ≥ n) :
  n ≤ 671 := by
    sorry

end largest_possible_n_l804_80499


namespace avg_age_of_new_persons_l804_80485

-- We define the given conditions
def initial_persons : ℕ := 12
def initial_avg_age : ℝ := 16
def new_persons : ℕ := 12
def new_avg_age : ℝ := 15.5

-- Define the total initial age
def total_initial_age : ℝ := initial_persons * initial_avg_age

-- Define the total number of persons after new persons join
def total_persons_after_join : ℕ := initial_persons + new_persons

-- Define the total age after new persons join
def total_age_after_join : ℝ := total_persons_after_join * new_avg_age

-- We wish to prove that the average age of the new persons who joined is 15
theorem avg_age_of_new_persons : 
  (total_initial_age + new_persons * 15) = total_age_after_join :=
sorry

end avg_age_of_new_persons_l804_80485


namespace moles_of_HCl_required_l804_80494

noncomputable def numberOfMolesHClRequired (moles_AgNO3 : ℕ) : ℕ :=
  if moles_AgNO3 = 3 then 3 else 0

-- Theorem statement
theorem moles_of_HCl_required : numberOfMolesHClRequired 3 = 3 := by
  sorry

end moles_of_HCl_required_l804_80494


namespace blue_hat_cost_l804_80414

theorem blue_hat_cost :
  ∀ (total_hats green_hats total_price green_hat_price blue_hat_price) 
  (B : ℕ),
  total_hats = 85 →
  green_hats = 30 →
  total_price = 540 →
  green_hat_price = 7 →
  blue_hat_price = B →
  (30 * 7) + (55 * B) = 540 →
  B = 6 := sorry

end blue_hat_cost_l804_80414


namespace expected_turns_formula_l804_80428

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1))))

theorem expected_turns_formula (n : ℕ) (h : n > 1) :
  expected_turns n = n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1)))) :=
by
  unfold expected_turns
  sorry

end expected_turns_formula_l804_80428


namespace red_and_purple_probability_l804_80422

def total_balls : ℕ := 120
def white_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 24
def red_balls : ℕ := 20
def blue_balls : ℕ := 10
def purple_balls : ℕ := 5
def orange_balls : ℕ := 4
def gray_balls : ℕ := 2

def probability_red_purple : ℚ := 5 / 357

theorem red_and_purple_probability :
  ((red_balls / total_balls) * (purple_balls / (total_balls - 1)) +
  (purple_balls / total_balls) * (red_balls / (total_balls - 1))) = probability_red_purple :=
by
  sorry

end red_and_purple_probability_l804_80422


namespace right_triangle_hypotenuse_l804_80434

theorem right_triangle_hypotenuse 
  (shorter_leg longer_leg hypotenuse : ℝ)
  (h1 : longer_leg = 2 * shorter_leg - 1)
  (h2 : 1 / 2 * shorter_leg * longer_leg = 60) :
  hypotenuse = 17 :=
by
  sorry

end right_triangle_hypotenuse_l804_80434


namespace find_y_l804_80460

theorem find_y (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : ∃ C, x * y = C) (hx : x = 4) : y = 50 :=
sorry

end find_y_l804_80460


namespace tiffany_total_lives_l804_80459

-- Define the conditions
def initial_lives : Float := 43.0
def hard_part_won : Float := 14.0
def next_level_won : Float := 27.0

-- State the theorem
theorem tiffany_total_lives : 
  initial_lives + hard_part_won + next_level_won = 84.0 :=
by 
  sorry

end tiffany_total_lives_l804_80459


namespace john_spent_on_sweets_l804_80466

def initial_amount := 7.10
def amount_given_per_friend := 1.00
def amount_left := 4.05
def amount_spent_on_friends := 2 * amount_given_per_friend
def amount_remaining_after_friends := initial_amount - amount_spent_on_friends
def amount_spent_on_sweets := amount_remaining_after_friends - amount_left

theorem john_spent_on_sweets : amount_spent_on_sweets = 1.05 := 
by
  sorry

end john_spent_on_sweets_l804_80466


namespace range_of_a_l804_80448

noncomputable def set_A : Set ℝ := { x | x^2 - 3 * x - 10 < 0 }
noncomputable def set_B : Set ℝ := { x | x^2 + 2 * x - 8 > 0 }
def set_C (a : ℝ) : Set ℝ := { x | 2 * a < x ∧ x < a + 3 }

theorem range_of_a (a : ℝ) :
  (A ∩ B) ∩ set_C a = set_C a → 1 ≤ a := 
sorry

end range_of_a_l804_80448


namespace train_length_l804_80453

theorem train_length (time : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (speed_ms : ℚ) (length : ℚ) :
  time = 50 ∧ speed_kmh = 36 ∧ conversion_factor = 5 / 18 ∧ speed_ms = speed_kmh * conversion_factor ∧ length = speed_ms * time →
  length = 500 :=
by
  sorry

end train_length_l804_80453


namespace correct_multiplication_result_l804_80446

theorem correct_multiplication_result :
  0.08 * 3.25 = 0.26 :=
by
  -- This is to ensure that the theorem is well-formed and logically connected
  sorry

end correct_multiplication_result_l804_80446


namespace min_abs_diff_x1_x2_l804_80401

theorem min_abs_diff_x1_x2 (x1 x2 : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = Real.sin (π * x))
  (Hbounds : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 1 := 
by
  sorry

end min_abs_diff_x1_x2_l804_80401


namespace total_fish_correct_l804_80444

-- Define the number of pufferfish
def num_pufferfish : ℕ := 15

-- Define the number of swordfish as 5 times the number of pufferfish
def num_swordfish : ℕ := 5 * num_pufferfish

-- Define the total number of fish as the sum of pufferfish and swordfish
def total_num_fish : ℕ := num_pufferfish + num_swordfish

-- Theorem stating the total number of fish
theorem total_fish_correct : total_num_fish = 90 := by
  -- Proof is omitted
  sorry

end total_fish_correct_l804_80444


namespace period_length_divisor_l804_80473

theorem period_length_divisor (p d : ℕ) (hp_prime : Nat.Prime p) (hd_period : ∀ n : ℕ, n ≥ 1 → 10^n % p = 1 ↔ n = d) :
  d ∣ (p - 1) :=
sorry

end period_length_divisor_l804_80473


namespace num_possible_sums_l804_80478

theorem num_possible_sums (s : Finset ℕ) (hs : s.card = 80) (hsub: s ⊆ Finset.range 121) : 
  ∃ (n : ℕ), (n = 3201) ∧ ∀ U, U = s.sum id → ∃ (U_min U_max : ℕ), U_min = 3240 ∧ U_max = 6440 ∧ (U_min ≤ U ∧ U ≤ U_max) :=
sorry

end num_possible_sums_l804_80478


namespace custom_op_equality_l804_80481

def custom_op (x y : Int) : Int :=
  x * y - 2 * x

theorem custom_op_equality : custom_op 5 3 - custom_op 3 5 = -4 := by
  sorry

end custom_op_equality_l804_80481


namespace cone_slant_height_l804_80468

theorem cone_slant_height (r l : ℝ) (h1 : r = 1)
  (h2 : 2 * r * Real.pi = (1 / 2) * 2 * l * Real.pi) :
  l = 2 :=
by
  -- Proof steps go here
  sorry

end cone_slant_height_l804_80468


namespace employee_payment_proof_l804_80435

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price as 20 percent above the wholesale cost
def retail_price (C_w : ℝ) : ℝ := C_w + 0.2 * C_w

-- Define the employee discount on the retail price
def employee_discount (C_r : ℝ) : ℝ := 0.15 * C_r

-- Define the amount paid by the employee
def amount_paid_by_employee (C_w : ℝ) : ℝ :=
  let C_r := retail_price C_w
  let D_e := employee_discount C_r
  C_r - D_e

-- Main theorem to prove the employee paid $204
theorem employee_payment_proof : amount_paid_by_employee wholesale_cost = 204 :=
by
  sorry

end employee_payment_proof_l804_80435


namespace arithmetic_sequence_20th_term_l804_80469

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 3
  let n := 20
  (a + (n - 1) * d) = 59 :=
by 
  sorry

end arithmetic_sequence_20th_term_l804_80469


namespace lunch_break_duration_l804_80489

def rate_sandra : ℝ := 0 -- Sandra's painting rate in houses per hour
def rate_helpers : ℝ := 0 -- Combined rate of the three helpers in houses per hour
def lunch_break : ℝ := 0 -- Lunch break duration in hours

axiom monday_condition : (8 - lunch_break) * (rate_sandra + rate_helpers) = 0.6
axiom tuesday_condition : (6 - lunch_break) * rate_helpers = 0.3
axiom wednesday_condition : (2 - lunch_break) * rate_sandra = 0.1

theorem lunch_break_duration : lunch_break = 0.5 :=
by {
  sorry
}

end lunch_break_duration_l804_80489


namespace points_per_vegetable_correct_l804_80465

-- Given conditions
def total_points_needed : ℕ := 200
def number_of_students : ℕ := 25
def number_of_weeks : ℕ := 2
def veggies_per_student_per_week : ℕ := 2

-- Derived values
def total_veggies_eaten_by_class : ℕ :=
  number_of_students * number_of_weeks * veggies_per_student_per_week

def points_per_vegetable : ℕ :=
  total_points_needed / total_veggies_eaten_by_class

-- Theorem to be proven
theorem points_per_vegetable_correct :
  points_per_vegetable = 2 := by
sorry

end points_per_vegetable_correct_l804_80465


namespace ratio_of_trees_l804_80471

theorem ratio_of_trees (plums pears apricots : ℕ) (h_plums : plums = 3) (h_pears : pears = 3) (h_apricots : apricots = 3) :
  plums = pears ∧ pears = apricots :=
by
  sorry

end ratio_of_trees_l804_80471


namespace square_root_properties_l804_80492

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l804_80492


namespace calculation_correctness_l804_80452

theorem calculation_correctness : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end calculation_correctness_l804_80452


namespace max_value_of_y_in_interval_l804_80490

theorem max_value_of_y_in_interval (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : 
  ∃ y_max, ∀ x, 0 < x ∧ x < 1 / 3 → x * (1 - 3 * x) ≤ y_max ∧ y_max = 1 / 12 :=
by sorry

end max_value_of_y_in_interval_l804_80490


namespace amount_subtracted_is_15_l804_80430

theorem amount_subtracted_is_15 (n x : ℕ) (h1 : 7 * n - x = 2 * n + 10) (h2 : n = 5) : x = 15 :=
by 
  sorry

end amount_subtracted_is_15_l804_80430


namespace exist_x_y_satisfy_condition_l804_80470

theorem exist_x_y_satisfy_condition (f g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0) (h2 : ∀ y, 0 ≤ y ∧ y ≤ 1 → g y ≥ 0) :
  ∃ (x : ℝ), ∃ (y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ |f x + g y - x * y| ≥ 1 / 4 :=
by
  sorry

end exist_x_y_satisfy_condition_l804_80470


namespace suff_not_nec_l804_80424

variables (a b : ℝ)
def P := (a = 1) ∧ (b = 1)
def Q := (a + b = 2)

theorem suff_not_nec : P a b → Q a b ∧ ¬ (Q a b → P a b) :=
by
  sorry

end suff_not_nec_l804_80424


namespace problem_a_problem_b_problem_c_problem_d_l804_80456

-- a) Proof problem for \(x^2 + 5x + 6 < 0\)
theorem problem_a (x : ℝ) : x^2 + 5*x + 6 < 0 → -3 < x ∧ x < -2 := by
  sorry

-- b) Proof problem for \(-x^2 + 9x - 20 < 0\)
theorem problem_b (x : ℝ) : -x^2 + 9*x - 20 < 0 → x < 4 ∨ x > 5 := by
  sorry

-- c) Proof problem for \(x^2 + x - 56 < 0\)
theorem problem_c (x : ℝ) : x^2 + x - 56 < 0 → -8 < x ∧ x < 7 := by
  sorry

-- d) Proof problem for \(9x^2 + 4 < 12x\) (No solutions)
theorem problem_d (x : ℝ) : ¬ 9*x^2 + 4 < 12*x := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l804_80456


namespace track_width_l804_80413

theorem track_width (r_1 r_2 : ℝ) (h1 : r_2 = 20) (h2 : 2 * Real.pi * r_1 - 2 * Real.pi * r_2 = 20 * Real.pi) : r_1 - r_2 = 10 :=
sorry

end track_width_l804_80413


namespace number_of_erasers_l804_80431

theorem number_of_erasers (P E : ℕ) (h1 : P + E = 240) (h2 : P = E - 2) : E = 121 := by
  sorry

end number_of_erasers_l804_80431


namespace range_of_k_l804_80426

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 ∧ x ≠ 2) ↔ (k ≤ 3 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l804_80426


namespace least_five_digit_congruent_to_6_mod_17_l804_80439

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l804_80439


namespace transformed_A_coordinates_l804_80404

open Real

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def A : ℝ × ℝ := (-3, 2)

theorem transformed_A_coordinates :
  reflect_over_y_axis (rotate_90_clockwise A) = (-2, 3) :=
by
  sorry

end transformed_A_coordinates_l804_80404


namespace equivalence_condition_l804_80497

theorem equivalence_condition (a b c d : ℝ) (h : (a + b) / (b + c) = (c + d) / (d + a)) : 
  a = c ∨ a + b + c + d = 0 :=
sorry

end equivalence_condition_l804_80497


namespace decreasing_interval_for_function_l804_80451

theorem decreasing_interval_for_function :
  ∀ (f : ℝ → ℝ) (ϕ : ℝ),
  (∀ x, f x = -2 * Real.tan (2 * x + ϕ)) →
  |ϕ| < Real.pi →
  f (Real.pi / 16) = -2 →
  ∃ a b : ℝ, 
  a = 3 * Real.pi / 16 ∧ 
  b = 11 * Real.pi / 16 ∧ 
  ∀ x, a < x ∧ x < b → ∀ y, x < y ∧ y < b → f y < f x :=
by sorry

end decreasing_interval_for_function_l804_80451


namespace compute_expression_l804_80442

theorem compute_expression : 6^2 + 2 * 5 - 4^2 = 30 :=
by sorry

end compute_expression_l804_80442


namespace amount_paid_out_l804_80437

theorem amount_paid_out 
  (amount : ℕ) 
  (h1 : amount % 50 = 0) 
  (h2 : ∃ (n : ℕ), n ≥ 15 ∧ amount = n * 5000 ∨ amount = n * 1000)
  (h3 : ∃ (n : ℕ), n ≥ 35 ∧ amount = n * 1000) : 
  amount = 29950 :=
by 
  sorry

end amount_paid_out_l804_80437


namespace smallest_integer_value_of_x_satisfying_eq_l804_80410

theorem smallest_integer_value_of_x_satisfying_eq (x : ℤ) (h : |x^2 - 5*x + 6| = 14) : 
  ∃ y : ℤ, (y = -1) ∧ ∀ z : ℤ, (|z^2 - 5*z + 6| = 14) → (y ≤ z) :=
sorry

end smallest_integer_value_of_x_satisfying_eq_l804_80410


namespace y_directly_proportional_x_l804_80408

-- Definition for direct proportionality
def directly_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

-- Theorem stating the relationship between y and x given the condition
theorem y_directly_proportional_x (x y : ℝ) (h : directly_proportional x y) :
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x :=
by
  sorry

end y_directly_proportional_x_l804_80408


namespace possible_distances_between_andrey_and_gleb_l804_80445

theorem possible_distances_between_andrey_and_gleb (A B V G : Point) 
  (d_AB : ℝ) (d_VG : ℝ) (d_BV : ℝ) (d_AG : ℝ)
  (h1 : d_AB = 600) 
  (h2 : d_VG = 600) 
  (h3 : d_AG = 3 * d_BV) : 
  d_AG = 900 ∨ d_AG = 1800 :=
by {
  sorry
}

end possible_distances_between_andrey_and_gleb_l804_80445
