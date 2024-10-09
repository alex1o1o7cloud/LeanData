import Mathlib

namespace ellie_needs_25ml_of_oil_l188_18833

theorem ellie_needs_25ml_of_oil 
  (oil_per_wheel : ℕ) 
  (number_of_wheels : ℕ) 
  (other_parts_oil : ℕ) 
  (total_oil : ℕ)
  (h1 : oil_per_wheel = 10)
  (h2 : number_of_wheels = 2)
  (h3 : other_parts_oil = 5)
  (h4 : total_oil = oil_per_wheel * number_of_wheels + other_parts_oil) : 
  total_oil = 25 :=
  sorry

end ellie_needs_25ml_of_oil_l188_18833


namespace periodic_sequence_exists_l188_18890

noncomputable def bounded_sequence (a : ℕ → ℤ) (M : ℤ) :=
  ∀ n, |a n| ≤ M

noncomputable def satisfies_recurrence (a : ℕ → ℤ) :=
  ∀ n, n ≥ 5 → a n = (a (n - 1) + a (n - 2) + a (n - 3) * a (n - 4)) / (a (n - 1) * a (n - 2) + a (n - 3) + a (n - 4))

theorem periodic_sequence_exists (a : ℕ → ℤ) (M : ℤ) 
  (h_bounded : bounded_sequence a M) (h_rec : satisfies_recurrence a) : 
  ∃ l : ℕ, ∀ n : ℕ, a (l + n) = a (l + n + (l + 1) - l) :=
sorry

end periodic_sequence_exists_l188_18890


namespace sum_of_first_100_terms_AP_l188_18881

theorem sum_of_first_100_terms_AP (a d : ℕ) :
  (15 / 2) * (2 * a + 14 * d) = 45 →
  (85 / 2) * (2 * a + 84 * d) = 255 →
  (100 / 2) * (2 * a + 99 * d) = 300 :=
by
  sorry

end sum_of_first_100_terms_AP_l188_18881


namespace total_students_in_class_l188_18814

-- Define the initial conditions
def num_students_in_row (a b: Nat) : Nat := a + 1 + b
def num_lines : Nat := 3
noncomputable def students_in_row : Nat := num_students_in_row 2 5 

-- Theorem to prove the total number of students in the class
theorem total_students_in_class : students_in_row * num_lines = 24 :=
by
  sorry

end total_students_in_class_l188_18814


namespace ratio_equality_l188_18887

variable (a b : ℝ)

theorem ratio_equality (h : a / b = 4 / 3) : (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
sorry

end ratio_equality_l188_18887


namespace james_drove_75_miles_l188_18809

noncomputable def james_total_distance : ℝ :=
  let speed1 := 30  -- mph
  let time1 := 0.5  -- hours
  let speed2 := 2 * speed1
  let time2 := 2 * time1
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  distance1 + distance2

theorem james_drove_75_miles : james_total_distance = 75 := by 
  sorry

end james_drove_75_miles_l188_18809


namespace units_digit_of_G_1000_l188_18886

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_of_G_1000 : (G 1000) % 10 = 2 := 
  sorry

end units_digit_of_G_1000_l188_18886


namespace intersection_P_Q_l188_18855

def P (x : ℝ) : Prop := x^2 - x - 2 ≥ 0

def Q (y : ℝ) : Prop := ∃ x, P x ∧ y = (1/2) * x^2 - 1

theorem intersection_P_Q :
  {m | ∃ (x : ℝ), P x ∧ m = (1/2) * x^2 - 1} = {m | m ≥ 2} := sorry

end intersection_P_Q_l188_18855


namespace hypotenuse_is_18_point_8_l188_18835

def hypotenuse_of_right_triangle (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2) * a * b = 24 ∧ a^2 + b^2 = c^2

theorem hypotenuse_is_18_point_8 (a b c : ℝ) (h : hypotenuse_of_right_triangle a b c) : c = 18.8 :=
  sorry

end hypotenuse_is_18_point_8_l188_18835


namespace find_cube_difference_l188_18837

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l188_18837


namespace longest_possible_height_l188_18838

theorem longest_possible_height (a b c : ℕ) (ha : a = 3 * c) (hb : b * 4 = 12 * c) (h_tri : a - c < b) (h_unequal : ¬(a = c)) :
  ∃ x : ℕ, (4 < x ∧ x < 6) ∧ x = 5 :=
by
  sorry

end longest_possible_height_l188_18838


namespace domain_ln_x_minus_1_l188_18899

def domain_of_log_function (x : ℝ) : Prop := x > 1

theorem domain_ln_x_minus_1 (x : ℝ) : domain_of_log_function x ↔ x > 1 :=
by {
  sorry
}

end domain_ln_x_minus_1_l188_18899


namespace smallest_abundant_number_not_multiple_of_10_l188_18856

-- Definition of proper divisors of a number n
def properDivisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

-- Definition of an abundant number
def isAbundant (n : ℕ) : Prop := 
  (properDivisors n).sum > n

-- Definition of not being a multiple of 10
def notMultipleOf10 (n : ℕ) : Prop := 
  n % 10 ≠ 0

-- Statement to prove
theorem smallest_abundant_number_not_multiple_of_10 :
  ∃ n, isAbundant n ∧ notMultipleOf10 n ∧ ∀ m, (isAbundant m ∧ notMultipleOf10 m) → n ≤ m :=
by
  sorry

end smallest_abundant_number_not_multiple_of_10_l188_18856


namespace domain_of_expression_l188_18843

theorem domain_of_expression (x : ℝ) :
  (1 ≤ x ∧ x < 6) ↔ (∃ y : ℝ, y = (x-1) ∧ y = (6-x) ∧ 0 ≤ y) :=
sorry

end domain_of_expression_l188_18843


namespace find_rate_of_current_l188_18842

-- Define the conditions
def speed_in_still_water (speed : ℝ) : Prop := speed = 15
def distance_downstream (distance : ℝ) : Prop := distance = 7.2
def time_in_hours (time : ℝ) : Prop := time = 0.4

-- Define the effective speed downstream
def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ := boat_speed + current_speed

-- Define rate of current
def rate_of_current (current_speed : ℝ) : Prop :=
  ∃ (c : ℝ), effective_speed_downstream 15 c * 0.4 = 7.2 ∧ c = current_speed

-- The theorem stating the proof problem
theorem find_rate_of_current : rate_of_current 3 :=
by
  sorry

end find_rate_of_current_l188_18842


namespace flower_bed_area_l188_18846

theorem flower_bed_area (total_posts : ℕ) (corner_posts : ℕ) (spacing : ℕ) (long_side_multiplier : ℕ)
  (h1 : total_posts = 24)
  (h2 : corner_posts = 4)
  (h3 : spacing = 3)
  (h4 : long_side_multiplier = 3) :
  ∃ (area : ℕ), area = 144 := 
sorry

end flower_bed_area_l188_18846


namespace number_total_11_l188_18893

theorem number_total_11 (N : ℕ) (S : ℝ)
  (h1 : S = 10.7 * N)
  (h2 : (6 : ℝ) * 10.5 = 63)
  (h3 : (6 : ℝ) * 11.4 = 68.4)
  (h4 : 13.7 = 13.700000000000017)
  (h5 : S = 63 + 68.4 - 13.7) : 
  N = 11 := 
sorry

end number_total_11_l188_18893


namespace centroid_of_triangle_PQR_positions_l188_18815

-- Define the basic setup
def square_side_length : ℕ := 12
def total_points : ℕ := 48

-- Define the centroid calculation condition
def centroid_positions_count : ℕ :=
  let side_segments := square_side_length
  let points_per_edge := total_points / 4
  let possible_positions_per_side := points_per_edge - 1
  (possible_positions_per_side * possible_positions_per_side)

/-- Proof statement: Proving the number of possible positions for the centroid of triangle PQR 
    formed by any three non-collinear points out of the 48 points on the perimeter of the square. --/
theorem centroid_of_triangle_PQR_positions : centroid_positions_count = 121 := 
  sorry

end centroid_of_triangle_PQR_positions_l188_18815


namespace count_right_triangles_with_conditions_l188_18870

theorem count_right_triangles_with_conditions :
  ∃ n : ℕ, n = 10 ∧
    (∀ (a b : ℕ),
      (a ^ 2 + b ^ 2 = (b + 2) ^ 2) →
      (b < 100) →
      (∃ k : ℕ, a = 2 * k ∧ k ^ 2 = b + 1) →
      n = 10) :=
by
  -- The proof goes here
  sorry

end count_right_triangles_with_conditions_l188_18870


namespace servings_correct_l188_18831

-- Define the pieces of popcorn in a serving
def pieces_per_serving := 30

-- Define the pieces of popcorn Jared can eat
def jared_pieces := 90

-- Define the pieces of popcorn each friend can eat
def friend_pieces := 60

-- Define the number of friends
def friends := 3

-- Calculate total pieces eaten by friends
def total_friend_pieces := friends * friend_pieces

-- Calculate total pieces eaten by everyone
def total_pieces := jared_pieces + total_friend_pieces

-- Calculate the number of servings needed
def servings_needed := total_pieces / pieces_per_serving

theorem servings_correct : servings_needed = 9 :=
by
  sorry

end servings_correct_l188_18831


namespace total_weight_of_new_people_l188_18861

theorem total_weight_of_new_people (W W_new : ℝ) :
  (∀ (old_weights : List ℝ), old_weights.length = 25 →
    ((old_weights.sum - (65 + 70 + 75)) + W_new = old_weights.sum + (4 * 25)) →
    W_new = 310) := by
  intros old_weights old_weights_length increase_condition
  -- Proof will be here
  sorry

end total_weight_of_new_people_l188_18861


namespace probability_not_monday_l188_18883

theorem probability_not_monday (P_monday : ℚ) (h : P_monday = 1/7) : P_monday ≠ 1 → ∃ P_not_monday : ℚ, P_not_monday = 6/7 :=
by
  sorry

end probability_not_monday_l188_18883


namespace algebraic_expression_value_l188_18803

-- Define the given condition
def condition (a b : ℝ) : Prop := a + b - 2 = 0

-- State the theorem to prove the algebraic expression value
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : a^2 - b^2 + 4 * b = 4 := by
  sorry

end algebraic_expression_value_l188_18803


namespace beyonce_album_songs_l188_18807

theorem beyonce_album_songs
  (singles : ℕ)
  (album1_songs album2_songs album3_songs total_songs : ℕ)
  (h1 : singles = 5)
  (h2 : album1_songs = 15)
  (h3 : album2_songs = 15)
  (h4 : total_songs = 55) :
  album3_songs = 20 :=
by
  sorry

end beyonce_album_songs_l188_18807


namespace abs_value_expression_l188_18858

theorem abs_value_expression (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 :=
sorry

end abs_value_expression_l188_18858


namespace sum_values_l188_18830

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 2) = -f x
axiom value_at_one : f 1 = 8

theorem sum_values :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end sum_values_l188_18830


namespace compute_fraction_l188_18811

theorem compute_fraction : 
  (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3 / 5 :=
by
  sorry

end compute_fraction_l188_18811


namespace lemonade_glasses_l188_18832

def lemons_total : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_glasses : lemons_total / lemons_per_glass = 9 := by
  sorry

end lemonade_glasses_l188_18832


namespace students_failed_l188_18826

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l188_18826


namespace option_a_is_correct_l188_18817

variable (a b : ℝ)
variable (ha : a < 0)
variable (hb : b < 0)
variable (hab : a < b)

theorem option_a_is_correct : (a < abs (3 * a + 2 * b) / 5) ∧ (abs (3 * a + 2 * b) / 5 < b) :=
by
  sorry

end option_a_is_correct_l188_18817


namespace distance_corresponds_to_additional_charge_l188_18847

-- Define the initial fee
def initial_fee : ℝ := 2.5

-- Define the charge per part of a mile
def charge_per_part_of_mile : ℝ := 0.35

-- Define the total charge for a 3.6 miles trip
def total_charge : ℝ := 5.65

-- Define the correct distance corresponding to the additional charge
def correct_distance : ℝ := 0.9

-- The theorem to prove
theorem distance_corresponds_to_additional_charge :
  (total_charge - initial_fee) / charge_per_part_of_mile * (0.1) = correct_distance :=
by
  sorry

end distance_corresponds_to_additional_charge_l188_18847


namespace witch_votes_is_seven_l188_18849

-- Definitions
def votes_for_witch (W : ℕ) : ℕ := W
def votes_for_unicorn (W : ℕ) : ℕ := 3 * W
def votes_for_dragon (W : ℕ) : ℕ := W + 25
def total_votes (W : ℕ) : ℕ := votes_for_witch W + votes_for_unicorn W + votes_for_dragon W

-- Proof Statement
theorem witch_votes_is_seven (W : ℕ) (h1 : total_votes W = 60) : W = 7 :=
by
  sorry

end witch_votes_is_seven_l188_18849


namespace symmetric_line_eq_l188_18862

theorem symmetric_line_eq (x y : ℝ) : (x - y = 0) → (x = 1) → (y = -x + 2) :=
by
  sorry

end symmetric_line_eq_l188_18862


namespace distance_to_focus_parabola_l188_18875

theorem distance_to_focus_parabola (F P : ℝ × ℝ) (hF : F = (0, -1/2))
  (hP : P = (1, 2)) (C : ℝ × ℝ → Prop)
  (hC : ∀ x, C (x, 2 * x^2)) : dist P F = 17 / 8 := by
sorry

end distance_to_focus_parabola_l188_18875


namespace domain_of_function_l188_18852

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 3 * x + 1 > 0
def condition2 (x : ℝ) : Prop := 2 - x ≠ 0

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > -1 / 3 ∧ x ≠ 2

theorem domain_of_function : 
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ domain x := 
by
  sorry

end domain_of_function_l188_18852


namespace max_sum_42_l188_18859

noncomputable def max_horizontal_vertical_sum (numbers : List ℕ) : ℕ :=
  let a := 14
  let b := 11
  let e := 17
  a + b + e

theorem max_sum_42 : 
  max_horizontal_vertical_sum [2, 5, 8, 11, 14, 17] = 42 := by
  sorry

end max_sum_42_l188_18859


namespace positive_difference_between_numbers_l188_18850

theorem positive_difference_between_numbers:
  ∃ x y : ℤ, x + y = 40 ∧ 3 * y - 4 * x = 7 ∧ |y - x| = 6 := by
  sorry

end positive_difference_between_numbers_l188_18850


namespace expression_evaluation_l188_18834

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l188_18834


namespace minimum_value_f_l188_18889

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f_l188_18889


namespace chloromethane_formation_l188_18829

variable (CH₄ Cl₂ CH₃Cl : Type)
variable (molesCH₄ molesCl₂ molesCH₃Cl : ℕ)

theorem chloromethane_formation 
  (h₁ : molesCH₄ = 3)
  (h₂ : molesCl₂ = 3)
  (reaction : CH₄ → Cl₂ → CH₃Cl)
  (one_to_one : ∀ (x y : ℕ), x = y → x = y): 
  molesCH₃Cl = 3 :=
by
  sorry

end chloromethane_formation_l188_18829


namespace average_speed_of_car_l188_18848

theorem average_speed_of_car (time : ℝ) (distance : ℝ) (h_time : time = 4.5) (h_distance : distance = 360) : 
  distance / time = 80 :=
by
  sorry

end average_speed_of_car_l188_18848


namespace height_of_tree_l188_18857

noncomputable def height_of_flagpole : ℝ := 4
noncomputable def shadow_of_flagpole : ℝ := 6
noncomputable def shadow_of_tree : ℝ := 12

theorem height_of_tree (h : height_of_flagpole / shadow_of_flagpole = x / shadow_of_tree) : x = 8 := by
  sorry

end height_of_tree_l188_18857


namespace find_m_l188_18853

-- Define the condition that the equation has a positive root
def hasPositiveRoot (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (2 / (x - 2) = 1 - (m / (x - 2)))

-- State the theorem
theorem find_m : ∀ m : ℝ, hasPositiveRoot m → m = -2 :=
by
  sorry

end find_m_l188_18853


namespace n_times_s_eq_neg_two_l188_18868

-- Define existence of function g
variable (g : ℝ → ℝ)

-- The given condition for the function g: ℝ -> ℝ
axiom g_cond : ∀ x y : ℝ, g (g x - y) = 2 * g x + g (g y - g (-x)) + y

-- Define n and s as per the conditions mentioned in the problem
def n : ℕ := 1 -- Based on the solution, there's only one possible value
def s : ℝ := -2 -- Sum of all possible values

-- The main statement to prove
theorem n_times_s_eq_neg_two : (n * s) = -2 := by
  sorry

end n_times_s_eq_neg_two_l188_18868


namespace marble_probability_l188_18823

theorem marble_probability :
  let total_ways := (Nat.choose 6 4)
  let favorable_ways := 
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1)
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 4 / 5 := by
  sorry

end marble_probability_l188_18823


namespace theater_ticket_sales_l188_18866

theorem theater_ticket_sales
  (A C : ℕ)
  (h₁ : 8 * A + 5 * C = 236)
  (h₂ : A + C = 34) : A = 22 :=
by
  sorry

end theater_ticket_sales_l188_18866


namespace resulting_polygon_has_30_sides_l188_18805

def polygon_sides : ℕ := 3 + 4 + 5 + 6 + 7 + 8 + 9 - 6 * 2

theorem resulting_polygon_has_30_sides : polygon_sides = 30 := by
  sorry

end resulting_polygon_has_30_sides_l188_18805


namespace dot_product_equilateral_l188_18836

-- Define the conditions for the equilateral triangle ABC
variable {A B C : ℝ}

noncomputable def equilateral_triangle (A B C : ℝ) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ |A - B| = 1 ∧ |B - C| = 1 ∧ |C - A| = 1

-- Define the dot product of the vectors AB and BC
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_equilateral (A B C : ℝ) (h : equilateral_triangle A B C) : 
  dot_product (B - A, 0) (C - B, 0) = -1 / 2 :=
sorry

end dot_product_equilateral_l188_18836


namespace petroleum_crude_oil_problem_l188_18821

variables (x y : ℝ)

theorem petroleum_crude_oil_problem (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 27.5) : y = 30 :=
by
  -- Proof would go here
  sorry

end petroleum_crude_oil_problem_l188_18821


namespace profit_percentage_is_correct_l188_18885

noncomputable def sellingPrice : ℝ := 850
noncomputable def profit : ℝ := 230
noncomputable def costPrice : ℝ := sellingPrice - profit

noncomputable def profitPercentage : ℝ :=
  (profit / costPrice) * 100

theorem profit_percentage_is_correct :
  profitPercentage = 37.10 :=
by
  sorry

end profit_percentage_is_correct_l188_18885


namespace even_integers_diff_digits_200_to_800_l188_18839

theorem even_integers_diff_digits_200_to_800 :
  ∃ n : ℕ, n = 131 ∧ (∀ x : ℕ, 200 ≤ x ∧ x < 800 ∧ (x % 2 = 0) ∧ (∀ i j : ℕ, i ≠ j → (x / 10^i % 10) ≠ (x / 10^j % 10)) ↔ x < n) :=
sorry

end even_integers_diff_digits_200_to_800_l188_18839


namespace arithmetic_sequence_product_l188_18802

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_a4a5 : a 3 * a 4 = 24) :
  a 2 * a 5 = 16 :=
sorry

end arithmetic_sequence_product_l188_18802


namespace calculate_average_fish_caught_l188_18804

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l188_18804


namespace planes_perpendicular_of_line_conditions_l188_18865

variables (a b l : Line) (M N : Plane)

-- Definitions of lines and planes and their relations
def parallel_to_plane (a : Line) (M : Plane) : Prop := sorry
def perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry
def subset_of_plane (a : Line) (M : Plane) : Prop := sorry

-- Statement of the main theorem to be proved
theorem planes_perpendicular_of_line_conditions (a b l : Line) (M N : Plane) :
  (perpendicular_to_plane a M) → (parallel_to_plane a N) → (perpendicular_to_plane N M) :=
  by
  sorry

end planes_perpendicular_of_line_conditions_l188_18865


namespace value_of_x_squared_plus_y_squared_l188_18898

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h : |x - 1/2| + (2*y + 1)^2 = 0) : 
  x^2 + y^2 = 1/2 :=
sorry

end value_of_x_squared_plus_y_squared_l188_18898


namespace money_made_l188_18819

-- Define the conditions
def cost_per_bar := 4
def total_bars := 8
def bars_sold := total_bars - 3

-- We need to show that the money made is $20
theorem money_made :
  bars_sold * cost_per_bar = 20 := 
by
  sorry

end money_made_l188_18819


namespace boat_speed_still_water_l188_18879

theorem boat_speed_still_water (b s : ℝ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := 
by 
  -- Solve the system of equations
  sorry

end boat_speed_still_water_l188_18879


namespace total_players_ground_l188_18813

-- Define the number of players for each type of sport
def c : ℕ := 10
def h : ℕ := 12
def f : ℕ := 16
def s : ℕ := 13

-- Statement of the problem to prove that the total number of players is 51
theorem total_players_ground : c + h + f + s = 51 :=
by
  -- proof will be added later
  sorry

end total_players_ground_l188_18813


namespace marching_band_formations_l188_18851

/-- A marching band of 240 musicians can be arranged in p different rectangular formations 
with s rows and t musicians per row where 8 ≤ t ≤ 30. 
This theorem asserts that there are 8 such different rectangular formations. -/
theorem marching_band_formations (s t : ℕ) (h : s * t = 240) (h_t_bounds : 8 ≤ t ∧ t ≤ 30) : 
  ∃ p : ℕ, p = 8 := 
sorry

end marching_band_formations_l188_18851


namespace abscissa_of_tangent_point_is_2_l188_18878

noncomputable def f (x : ℝ) : ℝ := (x^2) / 4 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := (1/2) * x - 3 / x

theorem abscissa_of_tangent_point_is_2 : 
  ∃ x0 : ℝ, f' x0 = -1/2 ∧ x0 = 2 :=
by
  sorry

end abscissa_of_tangent_point_is_2_l188_18878


namespace problem_1_problem_2_l188_18824

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1
theorem problem_1 (x : ℝ) : (∀ x, f x (-2) > 5) ↔ (x < -4 / 3 ∨ x > 2) :=
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) : (∀ x, f x a ≤ a * |x + 3|) → (a ≥ 1 / 2) :=
  sorry

end problem_1_problem_2_l188_18824


namespace graph_not_pass_first_quadrant_l188_18844

theorem graph_not_pass_first_quadrant (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ¬ (∃ x y : ℝ, y = a^x + b ∧ x > 0 ∧ y > 0) :=
sorry

end graph_not_pass_first_quadrant_l188_18844


namespace denominator_is_five_l188_18863

-- Define the conditions
variables (n d : ℕ)
axiom h1 : d = n - 4
axiom h2 : n + 6 = 3 * d

-- The theorem that needs to be proven
theorem denominator_is_five : d = 5 :=
by
  sorry

end denominator_is_five_l188_18863


namespace erased_digit_is_4_l188_18845

def sum_of_digits (n : ℕ) : ℕ := 
  sorry -- definition of sum of digits

def D (N : ℕ) : ℕ := N - sum_of_digits N

theorem erased_digit_is_4 (N : ℕ) (x : ℕ) 
  (hD : D N % 9 = 0) 
  (h_sum : sum_of_digits (D N) - x = 131) 
  : x = 4 :=
by
  sorry

end erased_digit_is_4_l188_18845


namespace no_real_solutions_l188_18891

theorem no_real_solutions :
  ∀ x y z : ℝ, ¬ (x + y + 2 + 4*x*y = 0 ∧ y + z + 2 + 4*y*z = 0 ∧ z + x + 2 + 4*z*x = 0) :=
by
  sorry

end no_real_solutions_l188_18891


namespace number_of_people_l188_18820

-- Conditions
def cost_oysters : ℤ := 3 * 15
def cost_shrimp : ℤ := 2 * 14
def cost_clams : ℤ := 2 * 135 / 10  -- Using integers for better precision
def total_cost : ℤ := cost_oysters + cost_shrimp + cost_clams
def amount_owed_each_person : ℤ := 25

-- Goal
theorem number_of_people (number_of_people : ℤ) : total_cost = number_of_people * amount_owed_each_person → number_of_people = 4 := by
  -- Proof to be completed here.
  sorry

end number_of_people_l188_18820


namespace julieta_total_spent_l188_18892

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end julieta_total_spent_l188_18892


namespace light_bulbs_circle_l188_18854

theorem light_bulbs_circle : ∀ (f : ℕ → ℕ),
  (f 0 = 1) ∧
  (f 1 = 2) ∧
  (f 2 = 4) ∧
  (f 3 = 8) ∧
  (∀ n, f n = f (n - 1) + f (n - 2) + f (n - 3) + f (n - 4)) →
  (f 9 - 3 * f 3 - 2 * f 2 - f 1 = 367) :=
by
  sorry

end light_bulbs_circle_l188_18854


namespace domain_of_function_l188_18825

def valid_domain (x : ℝ) : Prop :=
  x ≤ 3 ∧ x ≠ 0

theorem domain_of_function (x : ℝ) (h₀ : 3 - x ≥ 0) (h₁ : x ≠ 0) : valid_domain x :=
by
  sorry

end domain_of_function_l188_18825


namespace find_value_of_fraction_l188_18874

theorem find_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > x) (h : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 :=
by
  sorry

end find_value_of_fraction_l188_18874


namespace sum_of_ages_today_l188_18888

variable (RizaWas25WhenSonBorn : ℕ) (SonCurrentAge : ℕ) (SumOfAgesToday : ℕ)

theorem sum_of_ages_today (h1 : RizaWas25WhenSonBorn = 25) (h2 : SonCurrentAge = 40) : SumOfAgesToday = 105 :=
by
  sorry

end sum_of_ages_today_l188_18888


namespace smallest_positive_value_l188_18822

theorem smallest_positive_value (c d : ℤ) (h : c^2 > d^2) : 
  ∃ m > 0, m = (c^2 + d^2) / (c^2 - d^2) + (c^2 - d^2) / (c^2 + d^2) ∧ m = 2 :=
by
  sorry

end smallest_positive_value_l188_18822


namespace more_student_tickets_l188_18895

-- Definitions of given conditions
def student_ticket_price : ℕ := 6
def nonstudent_ticket_price : ℕ := 9
def total_sales : ℕ := 10500
def total_tickets : ℕ := 1700

-- Definitions of the variables for student and nonstudent tickets
variables (S N : ℕ)

-- Lean statement of the problem
theorem more_student_tickets (h1 : student_ticket_price * S + nonstudent_ticket_price * N = total_sales)
                            (h2 : S + N = total_tickets) : S - N = 1500 :=
by
  sorry

end more_student_tickets_l188_18895


namespace simple_interest_rate_l188_18876

-- Definitions based on conditions
def principal : ℝ := 750
def amount : ℝ := 900
def time : ℕ := 10

-- Statement to prove the rate of simple interest
theorem simple_interest_rate : 
  ∃ (R : ℝ), principal * R * time / 100 = amount - principal ∧ R = 2 :=
by
  sorry

end simple_interest_rate_l188_18876


namespace two_point_seven_five_as_fraction_l188_18801

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l188_18801


namespace painters_needed_days_l188_18864

-- Let P be the total work required in painter-work-days
def total_painter_work_days : ℕ := 5

-- Let E be the effective number of workers with advanced tools
def effective_workers : ℕ := 4

-- Define the number of days, we need to prove this equals 1.25
def days_to_complete_work (P E : ℕ) : ℚ := P / E

-- The main theorem to prove: for total_painter_work_days and effective_workers, the days to complete the work is 1.25
theorem painters_needed_days :
  days_to_complete_work total_painter_work_days effective_workers = 5 / 4 :=
by
  sorry

end painters_needed_days_l188_18864


namespace train_crossing_time_l188_18812

-- Definitions of the given conditions
def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 175

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 14.25 := 
sorry

end train_crossing_time_l188_18812


namespace k_league_teams_l188_18867

theorem k_league_teams (n : ℕ) (h : n*(n-1)/2 = 91) : n = 14 := sorry

end k_league_teams_l188_18867


namespace city_miles_count_l188_18882

-- Defining the variables used in the conditions
def miles_per_gallon_city : ℝ := 30
def miles_per_gallon_highway : ℝ := 40
def highway_miles : ℝ := 200
def cost_per_gallon : ℝ := 3
def total_cost : ℝ := 42

-- Required statement for the proof, statement to prove: count of city miles is 270
theorem city_miles_count : ∃ (C : ℝ), C = 270 ∧
  (total_cost / cost_per_gallon) = ((C / miles_per_gallon_city) + (highway_miles / miles_per_gallon_highway)) :=
by
  sorry

end city_miles_count_l188_18882


namespace problem_equiv_l188_18816

theorem problem_equiv {a : ℤ} : (a^2 ≡ 9 [ZMOD 10]) ↔ (a ≡ 3 [ZMOD 10] ∨ a ≡ -3 [ZMOD 10] ∨ a ≡ 7 [ZMOD 10] ∨ a ≡ -7 [ZMOD 10]) :=
sorry

end problem_equiv_l188_18816


namespace initial_notebooks_l188_18800

variable (a n : ℕ)
variable (h1 : n = 13 * a + 8)
variable (h2 : n = 15 * a)

theorem initial_notebooks : n = 60 := by
  -- additional details within the proof
  sorry

end initial_notebooks_l188_18800


namespace relationship_between_M_and_N_l188_18896
   
   variable (x : ℝ)
   def M := 2*x^2 - 12*x + 15
   def N := x^2 - 8*x + 11
   
   theorem relationship_between_M_and_N : M x ≥ N x :=
   by
     sorry
   
end relationship_between_M_and_N_l188_18896


namespace common_root_of_two_equations_l188_18828

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l188_18828


namespace songs_listened_l188_18894

theorem songs_listened (x y : ℕ) 
  (h1 : y = 9) 
  (h2 : y = 2 * (Nat.sqrt x) - 5) 
  : y + x = 58 := 
  sorry

end songs_listened_l188_18894


namespace roots_of_quadratic_l188_18860

theorem roots_of_quadratic (x : ℝ) : (x - 3) ^ 2 = 25 ↔ (x = 8 ∨ x = -2) :=
by sorry

end roots_of_quadratic_l188_18860


namespace complex_magnitude_pow_eight_l188_18818

theorem complex_magnitude_pow_eight :
  (Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I))^8 = 7890481 / 390625 := 
by
  sorry

end complex_magnitude_pow_eight_l188_18818


namespace num_pieces_l188_18810

theorem num_pieces (total_length : ℝ) (piece_length : ℝ) 
  (h1: total_length = 253.75) (h2: piece_length = 0.425) :
  ⌊total_length / piece_length⌋ = 597 :=
by
  rw [h1, h2]
  sorry

end num_pieces_l188_18810


namespace strings_completely_pass_each_other_l188_18869

-- Define the problem parameters
def d : ℝ := 30    -- distance between A and B in cm
def l1 : ℝ := 151  -- length of string A in cm
def l2 : ℝ := 187  -- length of string B in cm
def v1 : ℝ := 2    -- speed of string A in cm/s
def v2 : ℝ := 3    -- speed of string B in cm/s
def r1 : ℝ := 1    -- burn rate of string A in cm/s
def r2 : ℝ := 2    -- burn rate of string B in cm/s

-- The proof problem statement
theorem strings_completely_pass_each_other : ∀ (T : ℝ), T = 40 :=
by
  sorry

end strings_completely_pass_each_other_l188_18869


namespace equilateral_triangle_perimeter_l188_18884

theorem equilateral_triangle_perimeter (s : ℕ) (h1 : 2 * s + 10 = 50) : 3 * s = 60 :=
sorry

end equilateral_triangle_perimeter_l188_18884


namespace new_average_score_l188_18897

theorem new_average_score (average_initial : ℝ) (total_practices : ℕ) (highest_score lowest_score : ℝ) :
  average_initial = 87 → 
  total_practices = 10 → 
  highest_score = 95 → 
  lowest_score = 55 → 
  ((average_initial * total_practices - highest_score - lowest_score) / (total_practices - 2)) = 90 :=
by
  intros h_avg h_total h_high h_low
  sorry

end new_average_score_l188_18897


namespace necessary_and_sufficient_condition_l188_18840

variable {R : Type*} [LinearOrderedField R]
variable (f : R × R → R)
variable (x₀ y₀ : R)

theorem necessary_and_sufficient_condition :
  (f (x₀, y₀) = 0) ↔ ((x₀, y₀) ∈ {p : R × R | f p = 0}) :=
by
  sorry

end necessary_and_sufficient_condition_l188_18840


namespace least_k_9_l188_18880

open Nat

noncomputable def u : ℕ → ℝ
| 0     => 1 / 3
| (n+1) => 3 * u n - 3 * (u n) * (u n)

def M : ℝ := 0.5

def acceptable_error (n : ℕ): Prop := abs (u n - M) ≤ 1 / 2 ^ 500

theorem least_k_9 : ∃ k, 0 ≤ k ∧ acceptable_error k ∧ ∀ j, (0 ≤ j ∧ j < k) → ¬acceptable_error j ∧ k = 9 := by
  sorry

end least_k_9_l188_18880


namespace value_of_x_l188_18827

theorem value_of_x (x : ℝ) (h : (10 - x)^2 = x^2 + 4) : x = 24 / 5 :=
by
  sorry

end value_of_x_l188_18827


namespace interior_angle_of_regular_pentagon_is_108_l188_18871

-- Define the sum of angles in a triangle
def sum_of_triangle_angles : ℕ := 180

-- Define the number of triangles in a convex pentagon
def num_of_triangles_in_pentagon : ℕ := 3

-- Define the total number of interior angles in a pentagon
def num_of_angles_in_pentagon : ℕ := 5

-- Define the total sum of the interior angles of a pentagon
def sum_of_pentagon_interior_angles : ℕ := num_of_triangles_in_pentagon * sum_of_triangle_angles

-- Define the degree measure of an interior angle of a regular pentagon
def interior_angle_of_regular_pentagon : ℕ := sum_of_pentagon_interior_angles / num_of_angles_in_pentagon

theorem interior_angle_of_regular_pentagon_is_108 :
  interior_angle_of_regular_pentagon = 108 :=
by
  -- Proof will be filled in here
  sorry

end interior_angle_of_regular_pentagon_is_108_l188_18871


namespace quadratic_solution_l188_18872

-- Definition of the quadratic function satisfying the given conditions
def quadraticFunc (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (f (-1) = 12 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12)

-- The proof goal: proving the function f(x) is 2x^2 - 10x
theorem quadratic_solution (f : ℝ → ℝ) (h : quadraticFunc f) : ∀ x, f x = 2 * x^2 - 10 * x :=
by
  sorry

end quadratic_solution_l188_18872


namespace q_at_14_l188_18841

noncomputable def q (x : ℝ) : ℝ := - (1 / 2) * x^2 + x + 2

theorem q_at_14 : q 14 = -82 := by
  sorry

end q_at_14_l188_18841


namespace number_one_fourth_more_than_it_is_30_percent_less_than_80_l188_18877

theorem number_one_fourth_more_than_it_is_30_percent_less_than_80 :
    ∃ (n : ℝ), (5 / 4) * n = 56 ∧ n = 45 :=
by
  sorry

end number_one_fourth_more_than_it_is_30_percent_less_than_80_l188_18877


namespace intersection_at_one_point_l188_18806

theorem intersection_at_one_point (b : ℝ) :
  (∃ x₀ : ℝ, bx^2 + 7*x₀ + 4 = 0 ∧ (7)^2 - 4*b*4 = 0) →
  b = 49 / 16 :=
by
  sorry

end intersection_at_one_point_l188_18806


namespace fraction_of_cookies_with_nuts_l188_18808

theorem fraction_of_cookies_with_nuts
  (nuts_per_cookie : ℤ)
  (total_cookies : ℤ)
  (total_nuts : ℤ)
  (h1 : nuts_per_cookie = 2)
  (h2 : total_cookies = 60)
  (h3 : total_nuts = 72) :
  (total_nuts / nuts_per_cookie) / total_cookies = 3 / 5 := by
  sorry

end fraction_of_cookies_with_nuts_l188_18808


namespace find_ordered_triplets_l188_18873

theorem find_ordered_triplets (x y z : ℝ) :
  x^3 = z / y - 2 * y / z ∧
  y^3 = x / z - 2 * z / x ∧
  z^3 = y / x - 2 * x / y →
  (x = 1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) :=
sorry

end find_ordered_triplets_l188_18873
