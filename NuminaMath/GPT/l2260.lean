import Mathlib

namespace NUMINAMATH_GPT_money_spent_l2260_226007

def initial_money (Henry : Type) : ℤ := 11
def birthday_money (Henry : Type) : ℤ := 18
def final_money (Henry : Type) : ℤ := 19

theorem money_spent (Henry : Type) : (initial_money Henry + birthday_money Henry - final_money Henry = 10) := 
by sorry

end NUMINAMATH_GPT_money_spent_l2260_226007


namespace NUMINAMATH_GPT_bobby_initial_blocks_l2260_226015

variable (b : ℕ)

theorem bobby_initial_blocks
  (h : b + 6 = 8) : b = 2 := by
  sorry

end NUMINAMATH_GPT_bobby_initial_blocks_l2260_226015


namespace NUMINAMATH_GPT_ratio_students_sent_home_to_remaining_l2260_226000

theorem ratio_students_sent_home_to_remaining (total_students : ℕ) (students_taken_to_beach : ℕ)
    (students_still_in_school : ℕ) (students_sent_home : ℕ) 
    (h1 : total_students = 1000) (h2 : students_taken_to_beach = total_students / 2)
    (h3 : students_still_in_school = 250) 
    (h4 : students_sent_home = total_students / 2 - students_still_in_school) :
    (students_sent_home / students_still_in_school) = 1 := 
by
    sorry

end NUMINAMATH_GPT_ratio_students_sent_home_to_remaining_l2260_226000


namespace NUMINAMATH_GPT_fraction_students_say_dislike_but_actually_like_is_25_percent_l2260_226088

variable (total_students : Nat) (students_like_dancing : Nat) (students_dislike_dancing : Nat) 
         (students_like_dancing_but_say_dislike : Nat) (students_dislike_dancing_and_say_dislike : Nat) 
         (total_say_dislike : Nat)

def fraction_of_students_who_say_dislike_but_actually_like (total_students students_like_dancing students_dislike_dancing 
         students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike : Nat) : Nat :=
    (students_like_dancing_but_say_dislike * 100) / total_say_dislike

theorem fraction_students_say_dislike_but_actually_like_is_25_percent
  (h1 : total_students = 100)
  (h2 : students_like_dancing = 60)
  (h3 : students_dislike_dancing = 40)
  (h4 : students_like_dancing_but_say_dislike = 12)
  (h5 : students_dislike_dancing_and_say_dislike = 36)
  (h6 : total_say_dislike = 48) :
  fraction_of_students_who_say_dislike_but_actually_like total_students students_like_dancing students_dislike_dancing 
    students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike = 25 :=
by sorry

end NUMINAMATH_GPT_fraction_students_say_dislike_but_actually_like_is_25_percent_l2260_226088


namespace NUMINAMATH_GPT_multiplication_schemes_correct_l2260_226086

theorem multiplication_schemes_correct :
  ∃ A B C D E F G H I K L M N P : ℕ,
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ F = 0 ∧ G = 8 ∧ H = 3 ∧ I = 3 ∧ K = 8 ∧ L = 8 ∧ M = 0 ∧ N = 7 ∧ P = 7 ∧
    (A * 10 + B) * (C * 10 + D) * (A * 10 + B) = E * 100 + F * 10 + G ∧
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C ∧
    E * 100 + F * 10 + G / (H * 1000 + I * 100 + G * 10 + G) = (E * 100 + F * 10 + G) / (H * 1000 + I * 100 + G * 10 + G) ∧
    (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) = (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) :=
sorry

end NUMINAMATH_GPT_multiplication_schemes_correct_l2260_226086


namespace NUMINAMATH_GPT_trajectory_proof_l2260_226063

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  (y + Real.sqrt 2) * (y - Real.sqrt 2) / (x * x) = -2

theorem trajectory_proof :
  ∀ (x y : ℝ), x ≠ 0 → trajectory_eqn x y → (y*y / 2 + x*x = 1) :=
by
  intros x y hx htrajectory
  sorry

end NUMINAMATH_GPT_trajectory_proof_l2260_226063


namespace NUMINAMATH_GPT_probability_no_more_than_10_seconds_l2260_226014

noncomputable def total_cycle_time : ℕ := 80
noncomputable def green_time : ℕ := 30
noncomputable def yellow_time : ℕ := 10
noncomputable def red_time : ℕ := 40
noncomputable def can_proceed : ℕ := green_time + yellow_time + yellow_time

theorem probability_no_more_than_10_seconds : 
  can_proceed / total_cycle_time = 5 / 8 := 
  sorry

end NUMINAMATH_GPT_probability_no_more_than_10_seconds_l2260_226014


namespace NUMINAMATH_GPT_basketball_rim_height_l2260_226020

theorem basketball_rim_height
    (height_in_inches : ℕ)
    (reach_in_inches : ℕ)
    (jump_in_inches : ℕ)
    (above_rim_in_inches : ℕ) :
    height_in_inches = 72
    → reach_in_inches = 22
    → jump_in_inches = 32
    → above_rim_in_inches = 6
    → (height_in_inches + reach_in_inches + jump_in_inches - above_rim_in_inches) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_basketball_rim_height_l2260_226020


namespace NUMINAMATH_GPT_num_ways_to_assign_grades_l2260_226067

theorem num_ways_to_assign_grades : (4 ^ 12) = 16777216 := by
  sorry

end NUMINAMATH_GPT_num_ways_to_assign_grades_l2260_226067


namespace NUMINAMATH_GPT_prism_volume_l2260_226036

theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) : a * b * c = 1470 := by
  sorry

end NUMINAMATH_GPT_prism_volume_l2260_226036


namespace NUMINAMATH_GPT_months_b_after_a_started_business_l2260_226009

theorem months_b_after_a_started_business
  (A_initial : ℝ)
  (B_initial : ℝ)
  (profit_ratio : ℝ)
  (A_investment_time : ℕ)
  (B_investment_time : ℕ)
  (investment_ratio : A_initial * A_investment_time / (B_initial * B_investment_time) = profit_ratio) :
  B_investment_time = 6 :=
by
  -- Given:
  -- A_initial = 3500
  -- B_initial = 10500
  -- profit_ratio = 2 / 3
  -- A_investment_time = 12 months
  -- B_investment_time = 12 - x months
  -- We need to prove that x = 6 months such that investment ratio matches profit ratio.
  sorry

end NUMINAMATH_GPT_months_b_after_a_started_business_l2260_226009


namespace NUMINAMATH_GPT_no_four_nat_satisfy_l2260_226081

theorem no_four_nat_satisfy:
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 :=
by
  sorry

end NUMINAMATH_GPT_no_four_nat_satisfy_l2260_226081


namespace NUMINAMATH_GPT_tangent_line_at_M_l2260_226064

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

theorem tangent_line_at_M :
  let M : ℝ × ℝ := (2, 0)
  ∃ (m n : ℝ), n = f m ∧ m = 4 ∧ n = -2 * Real.exp 4 ∧
    ∀ (x y : ℝ), y = -Real.exp 4 * (x - 2) →
    M.2 = y :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_M_l2260_226064


namespace NUMINAMATH_GPT_number_of_ways_to_choose_water_polo_team_l2260_226006

theorem number_of_ways_to_choose_water_polo_team :
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  ∃ (total_ways : ℕ), 
  total_ways = total_members * Nat.choose (total_members - 1) player_choices ∧ 
  total_ways = 45045 :=
by
  let total_members := 15
  let team_size := 7
  let goalie_positions := 1
  let player_choices := team_size - goalie_positions
  have total_ways : ℕ := total_members * Nat.choose (total_members - 1) player_choices
  use total_ways
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_water_polo_team_l2260_226006


namespace NUMINAMATH_GPT_cookie_calories_l2260_226038

theorem cookie_calories 
  (burger_calories : ℕ)
  (carrot_stick_calories : ℕ)
  (num_carrot_sticks : ℕ)
  (total_lunch_calories : ℕ) :
  burger_calories = 400 ∧ 
  carrot_stick_calories = 20 ∧ 
  num_carrot_sticks = 5 ∧ 
  total_lunch_calories = 750 →
  (total_lunch_calories - (burger_calories + num_carrot_sticks * carrot_stick_calories) = 250) :=
by sorry

end NUMINAMATH_GPT_cookie_calories_l2260_226038


namespace NUMINAMATH_GPT_product_of_square_roots_of_nine_l2260_226023

theorem product_of_square_roots_of_nine (a b : ℝ) (ha : a^2 = 9) (hb : b^2 = 9) : a * b = -9 :=
sorry

end NUMINAMATH_GPT_product_of_square_roots_of_nine_l2260_226023


namespace NUMINAMATH_GPT_expression_evaluation_l2260_226089

theorem expression_evaluation (a b c : ℤ) 
  (h1 : c = a + 8) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23/15 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2260_226089


namespace NUMINAMATH_GPT_length_of_rectangle_l2260_226041

-- Definitions based on conditions:
def side_length_square : ℝ := 4
def width_rectangle : ℝ := 8
def area_square (side : ℝ) : ℝ := side * side
def area_rectangle (width length : ℝ) : ℝ := width * length

-- The goal is to prove the length of the rectangle
theorem length_of_rectangle :
  (area_square side_length_square) = (area_rectangle width_rectangle 2) :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l2260_226041


namespace NUMINAMATH_GPT_range_a_part1_range_a_part2_l2260_226091

def A (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0
def B (x a : ℝ) : Prop := x = x^2 - 4*x + a
def C (x a : ℝ) : Prop := x^2 - a*x - 4 ≤ 0

def p (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a
def q (a : ℝ) : Prop := ∀ x : ℝ, A x → C x a

theorem range_a_part1 : ¬(p a) → a > 6 := sorry

theorem range_a_part2 : p a ∧ q a → 0 ≤ a ∧ a ≤ 6 := sorry

end NUMINAMATH_GPT_range_a_part1_range_a_part2_l2260_226091


namespace NUMINAMATH_GPT_original_data_props_l2260_226032

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {new_x : Fin n → ℝ} 

noncomputable def average (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => data i)) / n

noncomputable def variance (data : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (data i - average data) ^ 2)) / n

-- Conditions
def condition1 (x new_x : Fin n → ℝ) (h : ∀ i, new_x i = x i - 80) : Prop := true

def condition2 (new_x : Fin n → ℝ) : Prop :=
  average new_x = 1.2

def condition3 (new_x : Fin n → ℝ) : Prop :=
  variance new_x = 4.4

theorem original_data_props (h : ∀ i, new_x i = x i - 80)
  (h_avg : average new_x = 1.2) 
  (h_var : variance new_x = 4.4) :
  average x = 81.2 ∧ variance x = 4.4 :=
sorry

end NUMINAMATH_GPT_original_data_props_l2260_226032


namespace NUMINAMATH_GPT_largest_positive_x_l2260_226018

def largest_positive_solution : ℝ := 1

theorem largest_positive_x 
  (x : ℝ) 
  (h : (2 * x^3 - x^2 - x + 1) ^ (1 + 1 / (2 * x + 1)) = 1) : 
  x ≤ largest_positive_solution := 
sorry

end NUMINAMATH_GPT_largest_positive_x_l2260_226018


namespace NUMINAMATH_GPT_find_p_l2260_226072

variables (m n p : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x = y / 3 - 2 / 5

theorem find_p
  (h1 : line_equation m n)
  (h2 : line_equation (m + p) (n + 9)) :
  p = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2260_226072


namespace NUMINAMATH_GPT_perfect_square_tens_digits_l2260_226062

theorem perfect_square_tens_digits
  (a b : ℕ)
  (is_square_a : ∃ k : ℕ, a = k * k)
  (is_square_b : ∃ k : ℕ, b = k * k)
  (units_digit_a : a % 10 = 1)
  (tens_digit_a : ∃ x : ℕ, a / 10 % 10 = x)
  (units_digit_b : b % 10 = 6)
  (tens_digit_b : ∃ y : ℕ, b / 10 % 10 = y) :
  ∃ x y : ℕ, (a / 10 % 10 = x) ∧ (b / 10 % 10 = y) ∧ (x % 2 = 0) ∧ (y % 2 = 1) :=
sorry

end NUMINAMATH_GPT_perfect_square_tens_digits_l2260_226062


namespace NUMINAMATH_GPT_bags_filled_on_saturday_l2260_226050

-- Definitions of the conditions
def bags_sat (S : ℕ) := S
def bags_sun := 4
def cans_per_bag := 9
def total_cans := 63

-- The statement to prove
theorem bags_filled_on_saturday (S : ℕ) 
  (h : total_cans = (bags_sat S + bags_sun) * cans_per_bag) : 
  S = 3 :=
by sorry

end NUMINAMATH_GPT_bags_filled_on_saturday_l2260_226050


namespace NUMINAMATH_GPT_find_k_value_l2260_226016

theorem find_k_value
  (x y k : ℝ)
  (h1 : 4 * x + 3 * y = 1)
  (h2 : k * x + (k - 1) * y = 3)
  (h3 : x = y) :
  k = 11 :=
  sorry

end NUMINAMATH_GPT_find_k_value_l2260_226016


namespace NUMINAMATH_GPT_set_subset_l2260_226085

-- Define the sets M and N
def M := {x : ℝ | abs x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- The mathematical statement to be proved
theorem set_subset : N ⊆ M := sorry

end NUMINAMATH_GPT_set_subset_l2260_226085


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l2260_226068

theorem relationship_between_a_b_c (a b c : ℕ) (h1 : a = 2^40) (h2 : b = 3^32) (h3 : c = 4^24) : a < c ∧ c < b := by
  -- Definitions as per conditions
  have ha : a = 32^8 := by sorry
  have hb : b = 81^8 := by sorry
  have hc : c = 64^8 := by sorry
  -- Comparisons involving the bases
  have h : 32 < 64 := by sorry
  have h' : 64 < 81 := by sorry
  -- Resultant comparison
  exact ⟨by sorry, by sorry⟩

end NUMINAMATH_GPT_relationship_between_a_b_c_l2260_226068


namespace NUMINAMATH_GPT_no_three_distinct_positive_perfect_squares_sum_to_100_l2260_226065

theorem no_three_distinct_positive_perfect_squares_sum_to_100 :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ (m n p : ℕ), a = m^2 ∧ b = n^2 ∧ c = p^2) ∧ a + b + c = 100 :=
by
  sorry

end NUMINAMATH_GPT_no_three_distinct_positive_perfect_squares_sum_to_100_l2260_226065


namespace NUMINAMATH_GPT_cos_equality_l2260_226056

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem cos_equality : ∃ n : ℝ, (0 ≤ n ∧ n ≤ 180) ∧ Real.cos (degrees_to_radians n) = Real.cos (degrees_to_radians 317) :=
by
  use 43
  simp [degrees_to_radians, Real.cos]
  sorry

end NUMINAMATH_GPT_cos_equality_l2260_226056


namespace NUMINAMATH_GPT_candy_division_l2260_226087

theorem candy_division (pieces_of_candy : Nat) (students : Nat) 
  (h1 : pieces_of_candy = 344) (h2 : students = 43) : pieces_of_candy / students = 8 := by
  sorry

end NUMINAMATH_GPT_candy_division_l2260_226087


namespace NUMINAMATH_GPT_value_of_p_l2260_226017

-- Let us assume the conditions given, and the existence of positive values p and q such that p + q = 1,
-- and the second term and fourth term of the polynomial expansion (x + y)^10 are equal when x = p and y = q.

theorem value_of_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h_sum : p + q = 1) (h_eq_terms : 10 * p ^ 9 * q = 120 * p ^ 7 * q ^ 3) :
    p = Real.sqrt (12 / 13) :=
    by sorry

end NUMINAMATH_GPT_value_of_p_l2260_226017


namespace NUMINAMATH_GPT_number_of_zeros_l2260_226002

-- Definitions based on the conditions
def five_thousand := 5 * 10 ^ 3
def one_hundred := 10 ^ 2

-- The main theorem that we want to prove
theorem number_of_zeros : (five_thousand ^ 50) * (one_hundred ^ 2) = 10 ^ 154 * 5 ^ 50 := 
by sorry

end NUMINAMATH_GPT_number_of_zeros_l2260_226002


namespace NUMINAMATH_GPT_sum_of_a_b_l2260_226045

theorem sum_of_a_b (a b : ℝ) (h₁ : a^3 - 3 * a^2 + 5 * a = 1) (h₂ : b^3 - 3 * b^2 + 5 * b = 5) : a + b = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_a_b_l2260_226045


namespace NUMINAMATH_GPT_series_sum_equals_seven_ninths_l2260_226010

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end NUMINAMATH_GPT_series_sum_equals_seven_ninths_l2260_226010


namespace NUMINAMATH_GPT_find_D_l2260_226049

-- Definitions from conditions
def is_different (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- The proof problem
theorem find_D (A B C D : ℕ) (h_diff: is_different A B C D) (h_eq : 700 + 10 * A + 5 + 100 * B + 70 + C = 100 * D + 38) : D = 9 :=
sorry

end NUMINAMATH_GPT_find_D_l2260_226049


namespace NUMINAMATH_GPT_cubic_inequality_solution_l2260_226042

theorem cubic_inequality_solution (x : ℝ) (h : 0 ≤ x) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ 16 < x := 
by 
  sorry

end NUMINAMATH_GPT_cubic_inequality_solution_l2260_226042


namespace NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l2260_226073

-- define the dimensions of the prism
variables (x y z : ℝ)

-- define the areas of the faces as conditions
def top_area := x * y
def back_area := y * z
def lateral_face_area := z * x

-- define the product of the areas of the top, back, and one lateral face
def product_of_areas := (top_area x y) * (back_area y z) * (lateral_face_area z x)

-- define the volume of the prism
def volume := x * y * z

-- theorem to prove: product of areas equals square of the volume
theorem product_of_areas_eq_square_of_volume 
  (ht: top_area x y = x * y)
  (hb: back_area y z = y * z)
  (hl: lateral_face_area z x = z * x) :
  product_of_areas x y z = (volume x y z) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l2260_226073


namespace NUMINAMATH_GPT_sum_of_prime_factors_1320_l2260_226022

theorem sum_of_prime_factors_1320 : 
  let smallest_prime := 2
  let largest_prime := 11
  smallest_prime + largest_prime = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_1320_l2260_226022


namespace NUMINAMATH_GPT_rosa_called_pages_sum_l2260_226069

theorem rosa_called_pages_sum :
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  sorry  -- proof will be done here

end NUMINAMATH_GPT_rosa_called_pages_sum_l2260_226069


namespace NUMINAMATH_GPT_abcd_inequality_l2260_226090

theorem abcd_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) :=
sorry

end NUMINAMATH_GPT_abcd_inequality_l2260_226090


namespace NUMINAMATH_GPT_D_is_largest_l2260_226019

def D := (2008 / 2007) + (2008 / 2009)
def E := (2008 / 2009) + (2010 / 2009)
def F := (2009 / 2008) + (2009 / 2010) - (1 / 2009)

theorem D_is_largest : D > E ∧ D > F := by
  sorry

end NUMINAMATH_GPT_D_is_largest_l2260_226019


namespace NUMINAMATH_GPT_triangle_base_l2260_226054

theorem triangle_base (sum_of_sides : ℕ) (left_side : ℕ) (right_side : ℕ) (base : ℕ)
  (h1 : sum_of_sides = 50)
  (h2 : right_side = left_side + 2)
  (h3 : left_side = 12) :
  base = 24 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_l2260_226054


namespace NUMINAMATH_GPT_exam_cutoff_mark_l2260_226077

theorem exam_cutoff_mark
  (num_students : ℕ)
  (absent_percentage : ℝ)
  (fail_percentage : ℝ)
  (fail_mark_diff : ℝ)
  (just_pass_percentage : ℝ)
  (remaining_avg_mark : ℝ)
  (class_avg_mark : ℝ)
  (absent_students : ℕ)
  (fail_students : ℕ)
  (just_pass_students : ℕ)
  (remaining_students : ℕ)
  (total_marks : ℝ)
  (P : ℝ) :
  absent_percentage = 0.2 →
  fail_percentage = 0.3 →
  fail_mark_diff = 20 →
  just_pass_percentage = 0.1 →
  remaining_avg_mark = 65 →
  class_avg_mark = 36 →
  absent_students = (num_students * absent_percentage) →
  fail_students = (num_students * fail_percentage) →
  just_pass_students = (num_students * just_pass_percentage) →
  remaining_students = num_students - absent_students - fail_students - just_pass_students →
  total_marks = (absent_students * 0) + (fail_students * (P - fail_mark_diff)) + (just_pass_students * P) + (remaining_students * remaining_avg_mark) →
  class_avg_mark = total_marks / num_students →
  P = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_exam_cutoff_mark_l2260_226077


namespace NUMINAMATH_GPT_correct_calculation_is_A_l2260_226003

theorem correct_calculation_is_A : (1 + (-2)) = -1 :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_is_A_l2260_226003


namespace NUMINAMATH_GPT_area_of_sector_equals_13_75_cm2_l2260_226005

noncomputable def radius : ℝ := 5 -- radius in cm
noncomputable def arc_length : ℝ := 5.5 -- arc length in cm
noncomputable def circumference : ℝ := 2 * Real.pi * radius -- circumference of the circle
noncomputable def area_of_circle : ℝ := Real.pi * radius^2 -- area of the entire circle

theorem area_of_sector_equals_13_75_cm2 :
  (arc_length / circumference) * area_of_circle = 13.75 :=
by sorry

end NUMINAMATH_GPT_area_of_sector_equals_13_75_cm2_l2260_226005


namespace NUMINAMATH_GPT_philip_oranges_count_l2260_226074

def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def betty_bill_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * betty_bill_oranges
def seeds_planted := frank_oranges * 2
def orange_trees := seeds_planted
def oranges_per_tree : ℕ := 5
def oranges_for_philip := orange_trees * oranges_per_tree

theorem philip_oranges_count : oranges_for_philip = 810 := by sorry

end NUMINAMATH_GPT_philip_oranges_count_l2260_226074


namespace NUMINAMATH_GPT_ratio_pow_eq_l2260_226033

variable (a b c d e f p q r : ℝ)
variable (n : ℕ)
variable (h : a / b = c / d)
variable (h1 : a / b = e / f)
variable (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)

theorem ratio_pow_eq
  (h : a / b = c / d)
  (h1 : a / b = e / f)
  (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)
  (n_ne_zero : n ≠ 0):
  (a / b) ^ n = (p * a ^ n + q * c ^ n + r * e ^ n) / (p * b ^ n + q * d ^ n + r * f ^ n) :=
by
  sorry

end NUMINAMATH_GPT_ratio_pow_eq_l2260_226033


namespace NUMINAMATH_GPT_fraction_equals_decimal_l2260_226061

theorem fraction_equals_decimal : (5 : ℝ) / 16 = 0.3125 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equals_decimal_l2260_226061


namespace NUMINAMATH_GPT_concert_attendance_l2260_226031

-- Define the given conditions
def buses : ℕ := 8
def students_per_bus : ℕ := 45

-- Statement of the problem
theorem concert_attendance :
  buses * students_per_bus = 360 :=
sorry

end NUMINAMATH_GPT_concert_attendance_l2260_226031


namespace NUMINAMATH_GPT_parabola_intersect_l2260_226037

theorem parabola_intersect (b c m p q x1 x2 : ℝ)
  (h_intersect1 : x1^2 + b * x1 + c = 0)
  (h_intersect2 : x2^2 + b * x2 + c = 0)
  (h_order : m < x1)
  (h_middle : x1 < x2)
  (h_range : x2 < m + 1)
  (h_valm : p = m^2 + b * m + c)
  (h_valm1 : q = (m + 1)^2 + b * (m + 1) + c) :
  p < 1 / 4 ∧ q < 1 / 4 :=
sorry

end NUMINAMATH_GPT_parabola_intersect_l2260_226037


namespace NUMINAMATH_GPT_difference_in_average_speed_l2260_226084

theorem difference_in_average_speed 
  (distance : ℕ) 
  (time_diff : ℕ) 
  (speed_B : ℕ) 
  (time_B : ℕ) 
  (time_A : ℕ) 
  (speed_A : ℕ)
  (h1 : distance = 300)
  (h2 : time_diff = 3)
  (h3 : speed_B = 20)
  (h4 : time_B = distance / speed_B)
  (h5 : time_A = time_B - time_diff)
  (h6 : speed_A = distance / time_A) 
  : speed_A - speed_B = 5 := 
sorry

end NUMINAMATH_GPT_difference_in_average_speed_l2260_226084


namespace NUMINAMATH_GPT_perimeter_of_T_shaped_figure_l2260_226096

theorem perimeter_of_T_shaped_figure :
  let a := 3    -- width of the horizontal rectangle
  let b := 5    -- height of the horizontal rectangle
  let c := 2    -- width of the vertical rectangle
  let d := 4    -- height of the vertical rectangle
  let overlap := 1 -- overlap length
  2 * a + 2 * b + 2 * c + 2 * d - 2 * overlap = 26 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_T_shaped_figure_l2260_226096


namespace NUMINAMATH_GPT_total_foreign_objects_l2260_226055

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_total_foreign_objects_l2260_226055


namespace NUMINAMATH_GPT_trapezoid_rectangle_ratio_l2260_226027

noncomputable def area_ratio (a1 a2 r : ℝ) : ℝ := 
  if a2 = 0 then 0 else a1 / a2

theorem trapezoid_rectangle_ratio 
  (radius : ℝ) (AD BC : ℝ)
  (trapezoid_area rectangle_area : ℝ) :
  radius = 13 →
  AD = 10 →
  BC = 24 →
  area_ratio trapezoid_area rectangle_area = 1 / 2 ∨
  area_ratio trapezoid_area rectangle_area = 289 / 338 :=
  sorry

end NUMINAMATH_GPT_trapezoid_rectangle_ratio_l2260_226027


namespace NUMINAMATH_GPT_smallest_even_integer_l2260_226046

theorem smallest_even_integer :
  ∃ (x : ℤ), |3 * x - 4| ≤ 20 ∧ (∀ (y : ℤ), |3 * y - 4| ≤ 20 → (2 ∣ y) → x ≤ y) ∧ (2 ∣ x) :=
by
  use -4
  sorry

end NUMINAMATH_GPT_smallest_even_integer_l2260_226046


namespace NUMINAMATH_GPT_eval_expression_l2260_226098

theorem eval_expression :
  72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 :=
by sorry

end NUMINAMATH_GPT_eval_expression_l2260_226098


namespace NUMINAMATH_GPT_parabola_line_intersection_l2260_226093

/-- 
Given a parabola \( y^2 = 2x \), a line passing through the focus of 
the parabola intersects the parabola at points \( A \) and \( B \) where 
the sum of the x-coordinates of \( A \) and \( B \) is equal to 2. 
Prove that such a line exists and there are exactly 3 such lines.
--/
theorem parabola_line_intersection :
  ∃ l₁ l₂ l₃ : (ℝ × ℝ) → (ℝ × ℝ), 
    (∀ p, l₁ p = l₂ p ∧ l₁ p = l₃ p → false) ∧
    ∀ (A B : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * A.1) ∧ 
      (B.2 ^ 2 = 2 * B.1) ∧ 
      (A.1 + B.1 = 2) →
      (∃ k : ℝ, 
        ∀ (x : ℝ), 
          ((A.2 = k * (A.1 - 1)) ∧ (B.2 = k * (B.1 - 1))) ∧ 
          (k * (A.1 - 1) = k * (B.1 - 1)) ∧ 
          (k ≠ 0)) :=
sorry

end NUMINAMATH_GPT_parabola_line_intersection_l2260_226093


namespace NUMINAMATH_GPT_money_made_per_minute_l2260_226013

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end NUMINAMATH_GPT_money_made_per_minute_l2260_226013


namespace NUMINAMATH_GPT_infinitely_many_not_sum_of_three_fourth_powers_l2260_226083

theorem infinitely_many_not_sum_of_three_fourth_powers : ∀ n : ℕ, n > 0 → n ≡ 5 [MOD 16] → ¬(∃ a b c : ℤ, n = a^4 + b^4 + c^4) :=
by sorry

end NUMINAMATH_GPT_infinitely_many_not_sum_of_three_fourth_powers_l2260_226083


namespace NUMINAMATH_GPT_tank_length_is_25_l2260_226035

noncomputable def cost_to_paise (cost_in_rupees : ℕ) : ℕ :=
  cost_in_rupees * 100

noncomputable def total_area_plastered (total_cost_in_paise : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  total_cost_in_paise / cost_per_sq_m

noncomputable def length_of_tank (width height cost_in_rupees rate : ℕ) : ℕ :=
  let total_cost_in_paise := cost_to_paise cost_in_rupees
  let total_area := total_area_plastered total_cost_in_paise rate
  let area_eq := total_area = (2 * (height * width) + 2 * (6 * height) + (height * width))
  let simplified_eq := total_area - 144 = 24 * height
  (total_area - 144) / 24

theorem tank_length_is_25 (width height cost_in_rupees rate : ℕ) : 
  width = 12 → height = 6 → cost_in_rupees = 186 → rate = 25 → length_of_tank width height cost_in_rupees rate = 25 :=
  by
    intros hwidth hheight hcost hrate
    unfold length_of_tank
    rw [hwidth, hheight, hcost, hrate]
    simp
    sorry

end NUMINAMATH_GPT_tank_length_is_25_l2260_226035


namespace NUMINAMATH_GPT_dividend_is_5336_l2260_226034

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) : 
  D * Q + R = 5336 := 
by sorry

end NUMINAMATH_GPT_dividend_is_5336_l2260_226034


namespace NUMINAMATH_GPT_geometric_sequence_condition_l2260_226094

-- Definitions based on conditions
def S (n : ℕ) (m : ℤ) : ℤ := 3^(n + 1) + m
def a1 (m : ℤ) : ℤ := S 1 m
def a_n (n : ℕ) : ℤ := if n = 1 then a1 (-3) else 2 * 3^n

-- The proof statement
theorem geometric_sequence_condition (m : ℤ) (h1 : a1 m = 3^2 + m) (h2 : ∀ n, n ≥ 2 → a_n n = 2 * 3^n) :
  m = -3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l2260_226094


namespace NUMINAMATH_GPT_coin_flip_probability_l2260_226044

theorem coin_flip_probability (P : ℕ → ℕ → ℚ) (n : ℕ) :
  (∀ k, P k 0 = 1/2) →
  (∀ k, P k 1 = 1/2) →
  (∀ k m, P k m = 1/2) →
  n = 3 →
  P 0 0 * P 1 1 * P 2 1 = 1/8 :=
by
  intros h0 h1 h_indep hn
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l2260_226044


namespace NUMINAMATH_GPT_moles_required_to_form_2_moles_H2O_l2260_226053

def moles_of_NH4NO3_needed (moles_of_H2O : ℕ) : ℕ := moles_of_H2O

theorem moles_required_to_form_2_moles_H2O :
  moles_of_NH4NO3_needed 2 = 2 := 
by 
  -- From the balanced equation 1 mole of NH4NO3 produces 1 mole of H2O
  -- Therefore, 2 moles of NH4NO3 are needed to produce 2 moles of H2O
  sorry

end NUMINAMATH_GPT_moles_required_to_form_2_moles_H2O_l2260_226053


namespace NUMINAMATH_GPT_leila_armchairs_l2260_226021

theorem leila_armchairs :
  ∀ {sofa_price armchair_price coffee_table_price total_invoice armchairs : ℕ},
  sofa_price = 1250 →
  armchair_price = 425 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  1 * sofa_price + armchairs * armchair_price + 1 * coffee_table_price = total_invoice →
  armchairs = 2 :=
by
  intros sofa_price armchair_price coffee_table_price total_invoice armchairs
  intros h1 h2 h3 h4 h_eq
  sorry

end NUMINAMATH_GPT_leila_armchairs_l2260_226021


namespace NUMINAMATH_GPT_books_read_l2260_226052

-- Definitions
def total_books : ℕ := 13
def unread_books : ℕ := 4

-- Theorem
theorem books_read : total_books - unread_books = 9 :=
by
  sorry

end NUMINAMATH_GPT_books_read_l2260_226052


namespace NUMINAMATH_GPT_total_books_l2260_226011

-- Definitions based on the conditions
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52
def AlexBooks : ℕ := 65

-- Theorem to be proven
theorem total_books : TimBooks + SamBooks + AlexBooks = 161 := by
  sorry

end NUMINAMATH_GPT_total_books_l2260_226011


namespace NUMINAMATH_GPT_persons_in_first_group_l2260_226026

-- Define the given conditions
def first_group_work_done (P : ℕ) : ℕ := P * 12 * 10
def second_group_work_done : ℕ := 30 * 26 * 6

-- Define the proof problem statement
theorem persons_in_first_group (P : ℕ) (h : first_group_work_done P = second_group_work_done) : P = 39 :=
by
  unfold first_group_work_done second_group_work_done at h
  sorry

end NUMINAMATH_GPT_persons_in_first_group_l2260_226026


namespace NUMINAMATH_GPT_Marty_combinations_l2260_226057

theorem Marty_combinations :
  let colors := 5
  let methods := 4
  let patterns := 3
  colors * methods * patterns = 60 :=
by
  sorry

end NUMINAMATH_GPT_Marty_combinations_l2260_226057


namespace NUMINAMATH_GPT_jessie_weight_before_jogging_l2260_226004

-- Definitions: conditions from the problem statement
variables (lost_weight current_weight : ℤ)
-- Conditions
def condition_lost_weight : Prop := lost_weight = 126
def condition_current_weight : Prop := current_weight = 66

-- Proposition to be proved
theorem jessie_weight_before_jogging (W_before_jogging : ℤ) :
  condition_lost_weight lost_weight → condition_current_weight current_weight →
  W_before_jogging = current_weight + lost_weight → W_before_jogging = 192 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jessie_weight_before_jogging_l2260_226004


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l2260_226097

theorem sum_of_digits_of_N :
  (∃ N : ℕ, 3 * N * (N + 1) / 2 = 3825 ∧ (N.digits 10).sum = 5) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l2260_226097


namespace NUMINAMATH_GPT_number_of_students_on_wednesday_l2260_226012

-- Define the problem conditions
variables (W T : ℕ)

-- Define the given conditions
def condition1 : Prop := T = W - 9
def condition2 : Prop := W + T = 65

-- Define the theorem to prove
theorem number_of_students_on_wednesday (h1 : condition1 W T) (h2 : condition2 W T) : W = 37 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_on_wednesday_l2260_226012


namespace NUMINAMATH_GPT_points_on_line_possible_l2260_226025

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_possible_l2260_226025


namespace NUMINAMATH_GPT_x_percent_more_than_y_l2260_226099

theorem x_percent_more_than_y (z : ℝ) (hz : z ≠ 0) (y : ℝ) (x : ℝ)
  (h1 : y = 0.70 * z) (h2 : x = 0.84 * z) :
  x = y + 0.20 * y :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_x_percent_more_than_y_l2260_226099


namespace NUMINAMATH_GPT_find_largest_number_l2260_226059

theorem find_largest_number 
  (a b c : ℕ) 
  (h1 : a + b = 16) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) : 
  c = 19 := 
sorry

end NUMINAMATH_GPT_find_largest_number_l2260_226059


namespace NUMINAMATH_GPT_num_four_digit_integers_with_3_and_6_l2260_226048

theorem num_four_digit_integers_with_3_and_6 : ∃ n, n = 16 ∧
  ∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ ∀ i ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10], (i = 3 ∨ i = 6) → x ∈ [3333, 3336, 3363, 3366, 3633, 3636, 3663, 3666, 6333, 6336, 6363, 6366, 6633, 6636, 6663, 6666] :=
by
  sorry

end NUMINAMATH_GPT_num_four_digit_integers_with_3_and_6_l2260_226048


namespace NUMINAMATH_GPT_freddy_talk_time_dad_l2260_226080

-- Conditions
def localRate : ℝ := 0.05
def internationalRate : ℝ := 0.25
def talkTimeBrother : ℕ := 31
def totalCost : ℝ := 10.0

-- Goal: Prove the duration of Freddy's local call to his dad is 45 minutes
theorem freddy_talk_time_dad : 
  ∃ (talkTimeDad : ℕ), 
    talkTimeDad = 45 ∧
    totalCost = (talkTimeBrother : ℝ) * internationalRate + (talkTimeDad : ℝ) * localRate := 
by
  sorry

end NUMINAMATH_GPT_freddy_talk_time_dad_l2260_226080


namespace NUMINAMATH_GPT_find_abc_l2260_226058

open Real

theorem find_abc {a b c : ℝ}
  (h1 : b + c = 16)
  (h2 : c + a = 17)
  (h3 : a + b = 18) :
  a * b * c = 606.375 :=
sorry

end NUMINAMATH_GPT_find_abc_l2260_226058


namespace NUMINAMATH_GPT_Vishal_investment_percentage_more_than_Trishul_l2260_226008

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end NUMINAMATH_GPT_Vishal_investment_percentage_more_than_Trishul_l2260_226008


namespace NUMINAMATH_GPT_john_sixth_quiz_score_l2260_226028

noncomputable def sixth_quiz_score_needed : ℤ :=
  let scores := [86, 91, 88, 84, 97]
  let desired_average := 95
  let number_of_quizzes := 6
  let total_score_needed := number_of_quizzes * desired_average
  let total_score_so_far := scores.sum
  total_score_needed - total_score_so_far

theorem john_sixth_quiz_score :
  sixth_quiz_score_needed = 124 := 
by
  sorry

end NUMINAMATH_GPT_john_sixth_quiz_score_l2260_226028


namespace NUMINAMATH_GPT_no_six_coins_sum_70_cents_l2260_226078

theorem no_six_coins_sum_70_cents :
  ¬ ∃ (p n d q : ℕ), p + n + d + q = 6 ∧ p + 5 * n + 10 * d + 25 * q = 70 :=
by
  sorry

end NUMINAMATH_GPT_no_six_coins_sum_70_cents_l2260_226078


namespace NUMINAMATH_GPT_bus_driver_total_hours_l2260_226040

def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_compensation : ℝ := 976
def max_regular_hours : ℝ := 40

theorem bus_driver_total_hours :
  ∃ (hours_worked : ℝ), 
  (hours_worked = max_regular_hours + (total_compensation - (regular_rate * max_regular_hours)) / overtime_rate) ∧
  hours_worked = 52 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_total_hours_l2260_226040


namespace NUMINAMATH_GPT_tucker_boxes_l2260_226082

def tissues_per_box := 160
def used_tissues := 210
def left_tissues := 270

def total_tissues := used_tissues + left_tissues

theorem tucker_boxes : total_tissues = tissues_per_box * 3 :=
by
  sorry

end NUMINAMATH_GPT_tucker_boxes_l2260_226082


namespace NUMINAMATH_GPT_simplify_polynomial_l2260_226076

theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2260_226076


namespace NUMINAMATH_GPT_determine_a_l2260_226030

def quadratic_condition (a : ℝ) (x : ℝ) : Prop := 
  abs (x^2 + 2 * a * x + 3 * a) ≤ 2

theorem determine_a : {a : ℝ | ∃! x : ℝ, quadratic_condition a x} = {1, 2} :=
sorry

end NUMINAMATH_GPT_determine_a_l2260_226030


namespace NUMINAMATH_GPT_field_area_l2260_226066

-- Define a rectangular field
structure RectangularField where
  length : ℕ
  width : ℕ
  fencing : ℕ := 2 * width + length
  
-- Given conditions
def field_conditions (L W F : ℕ) : Prop :=
  L = 30 ∧ 2 * W + L = F

-- Theorem stating the required proof
theorem field_area : ∀ (L W F : ℕ), field_conditions L W F → F = 84 → (L * W) = 810 :=
by
  intros L W F h1 h2
  sorry

end NUMINAMATH_GPT_field_area_l2260_226066


namespace NUMINAMATH_GPT_initial_machines_l2260_226001

theorem initial_machines (n x : ℕ) (hx : x > 0) (h : x / (4 * n) = x / 20) : n = 5 :=
by sorry

end NUMINAMATH_GPT_initial_machines_l2260_226001


namespace NUMINAMATH_GPT_parabola_focus_directrix_l2260_226079

-- Definitions and conditions
def parabola (y a x : ℝ) : Prop := y^2 = a * x
def distance_from_focus_to_directrix (d : ℝ) : Prop := d = 2

-- Statement of the problem
theorem parabola_focus_directrix {a : ℝ} (h : parabola y a x) (h2 : distance_from_focus_to_directrix d) : 
  a = 4 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_parabola_focus_directrix_l2260_226079


namespace NUMINAMATH_GPT_line_circle_interaction_l2260_226070

theorem line_circle_interaction (a : ℝ) :
  let r := 10
  let d := |a| / 5
  let intersects := -50 < a ∧ a < 50 
  let tangent := a = 50 ∨ a = -50 
  let separate := a < -50 ∨ a > 50 
  (d < r ↔ intersects) ∧ (d = r ↔ tangent) ∧ (d > r ↔ separate) :=
by sorry

end NUMINAMATH_GPT_line_circle_interaction_l2260_226070


namespace NUMINAMATH_GPT_inner_rectangle_length_is_4_l2260_226047

-- Define the conditions
def inner_rectangle_width : ℝ := 2
def shaded_region_width : ℝ := 2

-- Define the lengths and areas of the respective regions
def inner_rectangle_length (x : ℝ) : ℝ := x
def second_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 4, 6)
def largest_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 8, 10)

def inner_rectangle_area (x : ℝ) : ℝ := inner_rectangle_length x * inner_rectangle_width
def second_rectangle_area (x : ℝ) : ℝ := (second_rectangle_dimensions x).1 * (second_rectangle_dimensions x).2
def largest_rectangle_area (x : ℝ) : ℝ := (largest_rectangle_dimensions x).1 * (largest_rectangle_dimensions x).2

def first_shaded_region_area (x : ℝ) : ℝ := second_rectangle_area x - inner_rectangle_area x
def second_shaded_region_area (x : ℝ) : ℝ := largest_rectangle_area x - second_rectangle_area x

-- Define the arithmetic progression condition
def arithmetic_progression (x : ℝ) : Prop :=
  (first_shaded_region_area x - inner_rectangle_area x) = (second_shaded_region_area x - first_shaded_region_area x)

-- State the theorem
theorem inner_rectangle_length_is_4 :
  ∃ x : ℝ, arithmetic_progression x ∧ inner_rectangle_length x = 4 := 
by
  use 4
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inner_rectangle_length_is_4_l2260_226047


namespace NUMINAMATH_GPT_find_m_n_l2260_226092

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end NUMINAMATH_GPT_find_m_n_l2260_226092


namespace NUMINAMATH_GPT_number_of_roses_now_l2260_226075

-- Given Conditions
def initial_roses : Nat := 7
def initial_orchids : Nat := 12
def current_orchids : Nat := 20
def orchids_more_than_roses : Nat := 9

-- Question to Prove: 
theorem number_of_roses_now :
  ∃ (R : Nat), (current_orchids = R + orchids_more_than_roses) ∧ (R = 11) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_roses_now_l2260_226075


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2260_226043

noncomputable def arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 3

theorem arithmetic_sequence_general_formula
    (a : ℕ → ℤ)
    (h1 : (a 2 + a 6) / 2 = 5)
    (h2 : (a 3 + a 7) / 2 = 7) :
  arithmetic_sequence a :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2260_226043


namespace NUMINAMATH_GPT_sum_first_50_natural_numbers_l2260_226071

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Prove that the sum of the first 50 natural numbers is 1275
theorem sum_first_50_natural_numbers : sum_natural 50 = 1275 := 
by
  -- Skipping proof details
  sorry

end NUMINAMATH_GPT_sum_first_50_natural_numbers_l2260_226071


namespace NUMINAMATH_GPT_domain_of_composite_function_l2260_226039

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 → f x = f x) →
  (∀ (x : ℝ), -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → f (x^2) = f (x^2)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_composite_function_l2260_226039


namespace NUMINAMATH_GPT_original_cost_of_article_l2260_226095

theorem original_cost_of_article (x: ℝ) (h: 0.76 * x = 320) : x = 421.05 :=
sorry

end NUMINAMATH_GPT_original_cost_of_article_l2260_226095


namespace NUMINAMATH_GPT_value_of_function_at_2_l2260_226060

theorem value_of_function_at_2 (q : ℝ → ℝ) : q 2 = 5 :=
by
  -- Condition: The point (2, 5) lies on the graph of q
  have point_on_graph : q 2 = 5 := sorry
  exact point_on_graph

end NUMINAMATH_GPT_value_of_function_at_2_l2260_226060


namespace NUMINAMATH_GPT_sin_alpha_expression_l2260_226051

theorem sin_alpha_expression (α : ℝ) 
  (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 := 
sorry

end NUMINAMATH_GPT_sin_alpha_expression_l2260_226051


namespace NUMINAMATH_GPT_find_numbers_l2260_226029

noncomputable def sum_nat (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem find_numbers : 
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = sum_nat a b} = {14, 26, 37, 48, 59} :=
by {
  sorry
}

end NUMINAMATH_GPT_find_numbers_l2260_226029


namespace NUMINAMATH_GPT_multiplier_is_three_l2260_226024

theorem multiplier_is_three (n m : ℝ) (h₁ : n = 3) (h₂ : 7 * n = m * n + 12) : m = 3 := 
by
  -- Skipping the proof using sorry
  sorry 

end NUMINAMATH_GPT_multiplier_is_three_l2260_226024
