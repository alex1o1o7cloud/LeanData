import Mathlib

namespace check_triangle_345_l721_72154

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem check_triangle_345 : satisfies_triangle_inequality 3 4 5 := by
  sorry

end check_triangle_345_l721_72154


namespace gcd_correct_l721_72138

def gcd_87654321_12345678 : ℕ :=
  gcd 87654321 12345678

theorem gcd_correct : gcd_87654321_12345678 = 75 := by 
  sorry

end gcd_correct_l721_72138


namespace gasoline_price_increase_l721_72179

theorem gasoline_price_increase (high low : ℝ) (high_eq : high = 24) (low_eq : low = 18) : 
  ((high - low) / low) * 100 = 33.33 := 
  sorry

end gasoline_price_increase_l721_72179


namespace coins_remainder_l721_72134

theorem coins_remainder (N : ℕ) (h1 : N % 8 = 5) (h2 : N % 7 = 2) (hN_min : ∀ M : ℕ, (M % 8 = 5 ∧ M % 7 = 2) → N ≤ M) : N % 9 = 1 :=
sorry

end coins_remainder_l721_72134


namespace interest_rate_calculation_l721_72169

theorem interest_rate_calculation (P1 P2 I1 I2 : ℝ) (r1 : ℝ) :
  P2 = 1648 ∧ P1 = 2678 - P2 ∧ I2 = P2 * 0.05 * 3 ∧ I1 = P1 * r1 * 8 ∧ I1 = I2 →
  r1 = 0.03 :=
by sorry

end interest_rate_calculation_l721_72169


namespace find_length_of_chord_AB_l721_72104

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of points A and B
variables (x1 x2 y1 y2 : ℝ)

-- Define the conditions
def conditions : Prop := 
  parabola x1 y1 ∧ parabola x2 y2 ∧ (x1 + x2 = 4 / 3)

-- Define the length of chord AB
def length_of_chord_AB : ℝ := 
  (x1 + 1) + (x2 + 1)

-- Prove the length of chord AB
theorem find_length_of_chord_AB (x1 x2 y1 y2 : ℝ) (h : conditions x1 x2 y1 y2) :
  length_of_chord_AB x1 x2 = 10 / 3 :=
by
  sorry -- Proof is not required

end find_length_of_chord_AB_l721_72104


namespace time_2556_hours_from_now_main_l721_72151

theorem time_2556_hours_from_now (h : ℕ) (mod_res : h % 12 = 0) :
  (3 + h) % 12 = 3 :=
by {
  sorry
}

-- Constants
def current_time : ℕ := 3
def hours_passed : ℕ := 2556
-- Proof input
def modular_result : hours_passed % 12 = 0 := by {
 sorry -- In the real proof, we should show that 2556 is divisible by 12
}

-- Main theorem instance
theorem main : (current_time + hours_passed) % 12 = 3 := 
  time_2556_hours_from_now hours_passed modular_result

end time_2556_hours_from_now_main_l721_72151


namespace Ivy_cupcakes_l721_72128

theorem Ivy_cupcakes (M : ℕ) (h1 : M + (M + 15) = 55) : M = 20 :=
by
  sorry

end Ivy_cupcakes_l721_72128


namespace parabola_distance_l721_72118

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l721_72118


namespace machines_working_time_l721_72197

theorem machines_working_time (y: ℝ) 
  (h1 : y + 8 > 0)  -- condition for time taken by S
  (h2 : y + 2 > 0)  -- condition for time taken by T
  (h3 : 2 * y > 0)  -- condition for time taken by U
  : (1 / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ (y = 3 / 2) := 
by
  have h4 : y ≠ 0 := by linarith [h1, h2, h3]
  sorry

end machines_working_time_l721_72197


namespace sum_of_possible_values_l721_72152

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end sum_of_possible_values_l721_72152


namespace alice_speed_proof_l721_72109

-- Problem definitions
def distance : ℕ := 1000
def abel_speed : ℕ := 50
def abel_arrival_time := distance / abel_speed
def alice_delay : ℕ := 1  -- Alice starts 1 hour later
def earlier_arrival_abel : ℕ := 6  -- Abel arrives 6 hours earlier than Alice

noncomputable def alice_speed : ℕ := (distance / (abel_arrival_time + earlier_arrival_abel))

theorem alice_speed_proof : alice_speed = 200 / 3 := by
  sorry -- proof not required as per instructions

end alice_speed_proof_l721_72109


namespace assign_grades_l721_72108

def num_students : ℕ := 15
def options_per_student : ℕ := 4

theorem assign_grades:
  options_per_student ^ num_students = 1073741824 := by
  sorry

end assign_grades_l721_72108


namespace fraction_is_percent_l721_72161

theorem fraction_is_percent (y : ℝ) (hy : y > 0) : (6 * y / 20 + 3 * y / 10) = (60 / 100) * y :=
by
  sorry

end fraction_is_percent_l721_72161


namespace selling_price_correct_l721_72124

-- Define the conditions
def boxes := 3
def face_masks_per_box := 20
def cost_price := 15  -- in dollars
def profit := 15      -- in dollars

-- Define the total number of face masks
def total_face_masks := boxes * face_masks_per_box

-- Define the total amount he wants after selling all face masks
def total_amount := cost_price + profit

-- Prove that the selling price per face mask is $0.50
noncomputable def selling_price_per_face_mask : ℚ :=
  total_amount / total_face_masks

theorem selling_price_correct : selling_price_per_face_mask = 0.50 := by
  sorry

end selling_price_correct_l721_72124


namespace frequency_distribution_table_understanding_l721_72185

theorem frequency_distribution_table_understanding (size_sample_group : Prop) :
  (∃ (size_proportion : Prop) (corresponding_situation : Prop),
    size_sample_group → size_proportion ∧ corresponding_situation) :=
sorry

end frequency_distribution_table_understanding_l721_72185


namespace find_k_l721_72150

theorem find_k 
  (e1 : ℝ × ℝ) (h_e1 : e1 = (1, 0))
  (e2 : ℝ × ℝ) (h_e2 : e2 = (0, 1))
  (a : ℝ × ℝ) (h_a : a = (1, -2))
  (b : ℝ × ℝ) (h_b : b = (k, 1))
  (parallel : ∃ m : ℝ, a = (m * b.1, m * b.2)) : 
  k = -1/2 :=
sorry

end find_k_l721_72150


namespace min_value_of_quadratic_l721_72156

theorem min_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y^2 - 6 * y + 5 ≥ (x - 3)^2 - 4) ∧ (y^2 - 6 * y + 5 = -4) :=
by sorry

end min_value_of_quadratic_l721_72156


namespace school_club_members_l721_72178

theorem school_club_members :
  ∃ n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 3 ∧
  n % 8 = 5 ∧
  n % 9 = 7 ∧
  n = 269 :=
by
  existsi 269
  sorry

end school_club_members_l721_72178


namespace fabric_length_l721_72113

-- Define the width and area as given in the problem
def width : ℝ := 3
def area : ℝ := 24

-- Prove that the length is 8 cm
theorem fabric_length : (area / width) = 8 :=
by
  sorry

end fabric_length_l721_72113


namespace num_boys_is_22_l721_72189

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end num_boys_is_22_l721_72189


namespace geometric_sequence_general_term_l721_72115

theorem geometric_sequence_general_term (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) 
  (h1 : a 5 = a1 * q^4)
  (h2 : a 10 = a1 * q^9)
  (h3 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h4 : ∀ n, a n = a1 * q^(n - 1))
  (h_inc : q > 1) :
  ∀ n, a n = 2^n :=
by
  sorry

end geometric_sequence_general_term_l721_72115


namespace arithmetic_sequence_sum_l721_72119

theorem arithmetic_sequence_sum :
  let a1 := 1
  let d := 2
  let n := 10
  let an := 19
  let sum := 100
  let general_term := fun (n : ℕ) => a1 + (n - 1) * d
  (general_term n = an) → (n = 10) → (sum = (n * (a1 + an)) / 2) →
  sum = 100 :=
by
  sorry

end arithmetic_sequence_sum_l721_72119


namespace probability_area_less_than_circumference_l721_72137

theorem probability_area_less_than_circumference :
  let probability (d : ℕ) := if d = 2 then (1 / 100 : ℚ)
                             else if d = 3 then (1 / 50 : ℚ)
                             else 0
  let sum_prob (d_s : List ℚ) := d_s.foldl (· + ·) 0
  let outcomes : List ℕ := List.range' 2 19 -- dice sum range from 2 to 20
  let valid_outcomes : List ℕ := outcomes.filter (· < 4)
  sum_prob (valid_outcomes.map probability) = (3 / 100 : ℚ) :=
by
  sorry

end probability_area_less_than_circumference_l721_72137


namespace each_person_paid_45_l721_72120

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l721_72120


namespace solid_could_be_rectangular_prism_or_cylinder_l721_72147

-- Definitions for the conditions
def is_rectangular_prism (solid : Type) : Prop := sorry
def is_cylinder (solid : Type) : Prop := sorry
def front_view_is_rectangle (solid : Type) : Prop := sorry
def side_view_is_rectangle (solid : Type) : Prop := sorry

-- Main statement
theorem solid_could_be_rectangular_prism_or_cylinder
  {solid : Type}
  (h1 : front_view_is_rectangle solid)
  (h2 : side_view_is_rectangle solid) :
  is_rectangular_prism solid ∨ is_cylinder solid :=
sorry

end solid_could_be_rectangular_prism_or_cylinder_l721_72147


namespace expand_expression_l721_72146

theorem expand_expression (x y : ℤ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := 
by
  sorry

end expand_expression_l721_72146


namespace all_numbers_equal_l721_72126

theorem all_numbers_equal (x : Fin 101 → ℝ) 
  (h : ∀ i : Fin 100, x i.val^3 + x ⟨(i.val + 1) % 101, sorry⟩ = (x ⟨(i.val + 1) % 101, sorry⟩)^3 + x ⟨(i.val + 2) % 101, sorry⟩) :
  ∀ i j : Fin 101, x i = x j := 
by 
  sorry

end all_numbers_equal_l721_72126


namespace original_equation_l721_72165

theorem original_equation : 9^2 - 8^2 = 17 := by
  sorry

end original_equation_l721_72165


namespace fibonacci_odd_index_not_divisible_by_4k_plus_3_l721_72160

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_odd_index_not_divisible_by_4k_plus_3 (n k : ℕ) (p : ℕ) (h : p = 4 * k + 3) : ¬ (p ∣ fibonacci (2 * n - 1)) :=
by
  sorry

end fibonacci_odd_index_not_divisible_by_4k_plus_3_l721_72160


namespace equilateral_triangle_side_length_l721_72172

theorem equilateral_triangle_side_length (perimeter : ℝ) (h : perimeter = 2) : abs (perimeter / 3 - 0.67) < 0.01 :=
by
  -- The proof will go here.
  sorry

end equilateral_triangle_side_length_l721_72172


namespace intersection_with_negative_y_axis_max_value_at_x3_l721_72177

theorem intersection_with_negative_y_axis (m : ℝ) (h : 4 - 2 * m < 0) : m > 2 :=
sorry

theorem max_value_at_x3 (m : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 3 * x + 4 - 2 * m ≤ -4) : m = 8.5 :=
sorry

end intersection_with_negative_y_axis_max_value_at_x3_l721_72177


namespace ratio_jake_to_clementine_l721_72132

-- Definitions based on conditions
def ClementineCookies : Nat := 72
def ToryCookies (J : Nat) : Nat := (J + ClementineCookies) / 2
def TotalCookies (J : Nat) : Nat := ClementineCookies + J + ToryCookies J
def TotalRevenue : Nat := 648
def CookiePrice : Nat := 2
def TotalCookiesSold : Nat := TotalRevenue / CookiePrice

-- The main proof statement
theorem ratio_jake_to_clementine : 
  ∃ J : Nat, TotalCookies J = TotalCookiesSold ∧ J / ClementineCookies = 2 :=
by
  sorry

end ratio_jake_to_clementine_l721_72132


namespace harold_shared_with_five_friends_l721_72184

theorem harold_shared_with_five_friends 
  (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) (shared : ℕ) (friends : ℕ)
  (H1 : total_marbles = 100)
  (H2 : kept_marbles = 20)
  (H3 : marbles_per_friend = 16)
  (H4 : shared = total_marbles - kept_marbles)
  (H5 : friends = shared / marbles_per_friend) :
  friends = 5 :=
by
  sorry

end harold_shared_with_five_friends_l721_72184


namespace matrix_corner_sum_eq_l721_72127

theorem matrix_corner_sum_eq (M : Matrix (Fin 2000) (Fin 2000) ℤ)
  (h : ∀ i j : Fin 1999, M i j + M (i+1) (j+1) = M i (j+1) + M (i+1) j) :
  M 0 0 + M 1999 1999 = M 0 1999 + M 1999 0 :=
sorry

end matrix_corner_sum_eq_l721_72127


namespace katrina_cookies_left_l721_72159

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l721_72159


namespace boys_ages_l721_72145

theorem boys_ages (a b : ℕ) (h1 : a = b) (h2 : a + b + 11 = 29) : a = 9 :=
by
  sorry

end boys_ages_l721_72145


namespace solve_for_p_l721_72167

theorem solve_for_p (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end solve_for_p_l721_72167


namespace radius_is_100_div_pi_l721_72162

noncomputable def radius_of_circle (L : ℝ) (θ : ℝ) : ℝ :=
  L * 360 / (θ * 2 * Real.pi)

theorem radius_is_100_div_pi :
  radius_of_circle 25 45 = 100 / Real.pi := 
by
  sorry

end radius_is_100_div_pi_l721_72162


namespace largest_divisor_of_n_squared_sub_n_squared_l721_72175

theorem largest_divisor_of_n_squared_sub_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n_squared_sub_n_squared_l721_72175


namespace probability_at_least_one_male_l721_72117

-- Definitions according to the problem conditions
def total_finalists : ℕ := 8
def female_finalists : ℕ := 5
def male_finalists : ℕ := 3
def num_selected : ℕ := 3

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probabilistic statement
theorem probability_at_least_one_male :
  let total_ways := binom total_finalists num_selected
  let ways_all_females := binom female_finalists num_selected
  let ways_at_least_one_male := total_ways - ways_all_females
  (ways_at_least_one_male : ℚ) / total_ways = 23 / 28 :=
by
  sorry

end probability_at_least_one_male_l721_72117


namespace greatest_integer_less_PS_l721_72101

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l721_72101


namespace value_of_business_l721_72123

variable (business_value : ℝ) -- We are looking for the value of the business
variable (man_ownership_fraction : ℝ := 2/3) -- The fraction of the business the man owns
variable (sale_fraction : ℝ := 3/4) -- The fraction of the man's shares that were sold
variable (sale_amount : ℝ := 6500) -- The amount for which the fraction of the shares were sold

-- The main theorem we are trying to prove
theorem value_of_business (h1 : man_ownership_fraction = 2/3) (h2 : sale_fraction = 3/4) (h3 : sale_amount = 6500) :
    business_value = 39000 := 
sorry

end value_of_business_l721_72123


namespace abs_k_eq_sqrt_19_div_4_l721_72136

theorem abs_k_eq_sqrt_19_div_4
  (k : ℝ)
  (h : ∀ x : ℝ, x^2 - 4 * k * x + 1 = 0 → (x = r ∨ x = s))
  (h₁ : r + s = 4 * k)
  (h₂ : r * s = 1)
  (h₃ : r^2 + s^2 = 17) :
  |k| = (Real.sqrt 19) / 4 := by
sorry

end abs_k_eq_sqrt_19_div_4_l721_72136


namespace area_not_covered_by_smaller_squares_l721_72133

-- Define the conditions given in the problem
def side_length_larger_square : ℕ := 10
def side_length_smaller_square : ℕ := 4
def area_of_larger_square : ℕ := side_length_larger_square * side_length_larger_square
def area_of_each_smaller_square : ℕ := side_length_smaller_square * side_length_smaller_square

-- Define the total area of the two smaller squares
def total_area_smaller_squares : ℕ := area_of_each_smaller_square * 2

-- Define the uncovered area
def uncovered_area : ℕ := area_of_larger_square - total_area_smaller_squares

-- State the theorem to prove
theorem area_not_covered_by_smaller_squares :
  uncovered_area = 68 := by
  -- Placeholder for the actual proof
  sorry

end area_not_covered_by_smaller_squares_l721_72133


namespace cricket_average_l721_72196

theorem cricket_average (x : ℕ) (h : 20 * x + 158 = 21 * (x + 6)) : x = 32 :=
by
  sorry

end cricket_average_l721_72196


namespace trigonometric_identity_l721_72102

theorem trigonometric_identity :
  (Real.cos (12 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.sin (12 * Real.pi / 180) * Real.sin (18 * Real.pi / 180) = 
   Real.cos (30 * Real.pi / 180)) :=
by
  sorry

end trigonometric_identity_l721_72102


namespace total_pawns_left_is_10_l721_72122

noncomputable def total_pawns_left_in_game 
    (initial_pawns : ℕ)
    (sophia_lost : ℕ)
    (chloe_lost : ℕ) : ℕ :=
  initial_pawns - sophia_lost + (initial_pawns - chloe_lost)

theorem total_pawns_left_is_10 :
  total_pawns_left_in_game 8 5 1 = 10 := by
  sorry

end total_pawns_left_is_10_l721_72122


namespace solve_for_m_l721_72141

theorem solve_for_m (m x : ℤ) (h : 4 * x + 2 * m - 14 = 0) (hx : x = 2) : m = 3 :=
by
  -- Proof steps will go here.
  sorry

end solve_for_m_l721_72141


namespace blue_marbles_difference_l721_72100

-- Definitions of the conditions
def total_green_marbles := 95

-- Ratios for Jar 1 and Jar 2
def ratio_blue_green_jar1 := (9, 1)
def ratio_blue_green_jar2 := (8, 1)

-- Total number of green marbles in each jar
def green_marbles_jar1 (a : ℕ) := a
def green_marbles_jar2 (b : ℕ) := b

-- Total number of marbles in each jar
def total_marbles_jar1 (a : ℕ) := 10 * a
def total_marbles_jar2 (b : ℕ) := 9 * b

-- Number of blue marbles in each jar
def blue_marbles_jar1 (a : ℕ) := 9 * a
def blue_marbles_jar2 (b : ℕ) := 8 * b

-- Conditions in terms of Lean definitions
theorem blue_marbles_difference:
  ∀ (a b : ℕ), green_marbles_jar1 a + green_marbles_jar2 b = total_green_marbles →
  total_marbles_jar1 a = total_marbles_jar2 b →
  blue_marbles_jar1 a - blue_marbles_jar2 b = 5 :=
by sorry

end blue_marbles_difference_l721_72100


namespace domain_of_tan_arcsin_xsq_l721_72188

noncomputable def domain_f (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -1 ∧ -1 ≤ x ∧ x ≤ 1

theorem domain_of_tan_arcsin_xsq :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ domain_f x := 
sorry

end domain_of_tan_arcsin_xsq_l721_72188


namespace f_has_two_zeros_iff_l721_72131

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * exp x + a * (x - 1)^2

theorem f_has_two_zeros_iff (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a :=
sorry

end f_has_two_zeros_iff_l721_72131


namespace Annika_hike_time_l721_72163

-- Define the conditions
def hike_rate : ℝ := 12 -- in minutes per kilometer
def initial_distance_east : ℝ := 2.75 -- in kilometers
def total_distance_east : ℝ := 3.041666666666667 -- in kilometers
def total_time_needed : ℝ := 40 -- in minutes

-- The theorem to prove
theorem Annika_hike_time : 
  (initial_distance_east + (total_distance_east - initial_distance_east)) * hike_rate + total_distance_east * hike_rate = total_time_needed := 
by
  sorry

end Annika_hike_time_l721_72163


namespace tree_height_end_of_third_year_l721_72144

theorem tree_height_end_of_third_year (h : ℝ) : 
    (∃ h0 h3 h6 : ℝ, 
      h3 = h0 * 3^3 ∧ 
      h6 = h3 * 2^3 ∧ 
      h6 = 1458) → h3 = 182.25 :=
by sorry

end tree_height_end_of_third_year_l721_72144


namespace original_cube_edge_length_l721_72135

theorem original_cube_edge_length (a : ℕ) (h1 : 6 * (a ^ 3) = 7 * (6 * (a ^ 2))) : a = 7 := 
by 
  sorry

end original_cube_edge_length_l721_72135


namespace gaokun_population_scientific_notation_l721_72139

theorem gaokun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (425000 = a * 10^n) ∧ (a = 4.25) ∧ (n = 5) :=
by
  sorry

end gaokun_population_scientific_notation_l721_72139


namespace max_min_diff_value_l721_72125

noncomputable def max_min_diff_c (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : ℝ :=
  (10 / 3) - (-2)

theorem max_min_diff_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : 
  max_min_diff_c a b c h1 h2 = 16 / 3 := 
by 
  sorry

end max_min_diff_value_l721_72125


namespace grouping_schemes_count_l721_72166

/-- Number of possible grouping schemes where each group consists
    of either 2 or 3 students and the total number of students is 25 is 4.-/
theorem grouping_schemes_count : ∃ (x y : ℕ), 2 * x + 3 * y = 25 ∧ 
  (x = 11 ∧ y = 1 ∨ x = 8 ∧ y = 3 ∨ x = 5 ∧ y = 5 ∨ x = 2 ∧ y = 7) :=
sorry

end grouping_schemes_count_l721_72166


namespace product_sum_even_l721_72105

theorem product_sum_even (m n : ℤ) : Even (m * n * (m + n)) := 
sorry

end product_sum_even_l721_72105


namespace not_enough_space_in_cube_l721_72130

-- Define the edge length of the cube in kilometers.
def cube_edge_length_km : ℝ := 3

-- Define the global population exceeding threshold.
def global_population : ℝ := 7 * 10^9

-- Define the function to calculate the volume of a cube given its edge length in kilometers.
def cube_volume_km (edge_length: ℝ) : ℝ := edge_length^3

-- Define the conversion from kilometers to meters.
def km_to_m (distance_km: ℝ) : ℝ := distance_km * 1000

-- Define the function to calculate the volume of the cube in cubic meters.
def cube_volume_m (edge_length_km: ℝ) : ℝ := (km_to_m edge_length_km)^3

-- Statement: The entire population and all buildings and structures will not fit inside the cube.
theorem not_enough_space_in_cube :
  cube_volume_m cube_edge_length_km < global_population * (some_constant_value_to_account_for_buildings_and_structures) :=
sorry

end not_enough_space_in_cube_l721_72130


namespace find_x_l721_72190

def x : ℕ := 70

theorem find_x :
  x + (5 * 12) / (180 / 3) = 71 :=
by
  sorry

end find_x_l721_72190


namespace length_of_equal_sides_l721_72140

-- Definitions based on conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ a = c)

def is_triangle (a b c : ℝ) : Prop :=
(a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def has_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
a + b + c = P

def one_side_length (a : ℝ) : Prop :=
a = 3

-- The proof statement
theorem length_of_equal_sides (a b c : ℝ) :
isosceles_triangle a b c →
is_triangle a b c →
has_perimeter a b c 7 →
one_side_length a ∨ one_side_length b ∨ one_side_length c →
(b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) :=
by
  intros iso tri per side_length
  sorry

end length_of_equal_sides_l721_72140


namespace thabo_book_ratio_l721_72176

theorem thabo_book_ratio :
  ∃ (P_f P_nf H_nf : ℕ), H_nf = 35 ∧ P_nf = H_nf + 20 ∧ P_f + P_nf + H_nf = 200 ∧ P_f / P_nf = 2 :=
by
  sorry

end thabo_book_ratio_l721_72176


namespace min_diff_two_composite_sum_91_l721_72181

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

-- Minimum positive difference between two composite numbers that sum up to 91
theorem min_diff_two_composite_sum_91 : ∃ a b : ℕ, 
  is_composite a ∧ 
  is_composite b ∧ 
  a + b = 91 ∧ 
  b - a = 1 :=
by
  sorry

end min_diff_two_composite_sum_91_l721_72181


namespace larger_number_l721_72143

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l721_72143


namespace table_sale_price_percentage_l721_72199

theorem table_sale_price_percentage (W : ℝ) : 
  let S := 1.4 * W
  let P := 0.65 * S
  P = 0.91 * W :=
by
  sorry

end table_sale_price_percentage_l721_72199


namespace max_r_value_l721_72149

theorem max_r_value (r : ℕ) (hr : r ≥ 2)
  (m n : Fin r → ℤ)
  (h : ∀ i j : Fin r, i < j → |m i * n j - m j * n i| = 1) :
  r ≤ 3 := 
sorry

end max_r_value_l721_72149


namespace find_divisor_l721_72174

-- Define the conditions
def dividend := 689
def quotient := 19
def remainder := 5

-- Define the division formula
def division_formula (divisor : ℕ) : Prop := 
  dividend = (divisor * quotient) + remainder

-- State the theorem to be proved
theorem find_divisor :
  ∃ divisor : ℕ, division_formula divisor ∧ divisor = 36 :=
by
  sorry

end find_divisor_l721_72174


namespace P_and_Q_together_l721_72164

theorem P_and_Q_together (W : ℝ) (H : W > 0) :
  (1 / (1 / 4 + 1 / (1 / 3 * (1 / 4)))) = 3 :=
by
  sorry

end P_and_Q_together_l721_72164


namespace pears_thrown_away_on_first_day_l721_72110

theorem pears_thrown_away_on_first_day (x : ℝ) (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.8 * P = P * 0.8)
  (total_thrown_percentage : (x / 100) * 0.2 * P + 0.2 * (1 - x / 100) * 0.2 * P = 0.12 * P ) : 
  x = 50 :=
by
  sorry

end pears_thrown_away_on_first_day_l721_72110


namespace symmetric_point_x_axis_l721_72186

theorem symmetric_point_x_axis (x y : ℝ) (p : Prod ℝ ℝ) (hx : p = (x, y)) :
  (x, -y) = (1, -2) ↔ (x, y) = (1, 2) :=
by
  sorry

end symmetric_point_x_axis_l721_72186


namespace park_width_l721_72103

/-- The rectangular park theorem -/
theorem park_width 
  (length : ℕ)
  (lawn_area : ℤ)
  (road_width : ℕ)
  (crossroads : ℕ)
  (W : ℝ) :
  length = 60 →
  lawn_area = 2109 →
  road_width = 3 →
  crossroads = 2 →
  W = (2109 + (2 * 3 * 60) : ℝ) / 60 :=
sorry

end park_width_l721_72103


namespace correlation_non_deterministic_relationship_l721_72129

theorem correlation_non_deterministic_relationship
  (independent_var_fixed : Prop)
  (dependent_var_random : Prop)
  (correlation_def : Prop)
  (correlation_randomness : Prop) :
  (correlation_def → non_deterministic) :=
by
  sorry

end correlation_non_deterministic_relationship_l721_72129


namespace total_rent_payment_l721_72195

def weekly_rent : ℕ := 388
def number_of_weeks : ℕ := 1359

theorem total_rent_payment : weekly_rent * number_of_weeks = 526692 := 
  by 
  sorry

end total_rent_payment_l721_72195


namespace birthday_gift_l721_72106

-- Define the conditions
def friends : Nat := 8
def dollars_per_friend : Nat := 15

-- Formulate the statement to prove
theorem birthday_gift : friends * dollars_per_friend = 120 := by
  -- Proof is skipped using 'sorry'
  sorry

end birthday_gift_l721_72106


namespace unique_sum_of_two_primes_l721_72183

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l721_72183


namespace determine_pairs_l721_72148

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem determine_pairs (n p : ℕ) (hn_pos : 0 < n) (hp_prime : is_prime p) (hn_le_2p : n ≤ 2 * p) (divisibility : n^p - 1 ∣ (p - 1)^n + 1):
  (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
by
  sorry

end determine_pairs_l721_72148


namespace abs_sum_of_roots_l721_72191

theorem abs_sum_of_roots 
  (a b c m : ℤ) 
  (h1 : a + b + c = 0)
  (h2 : ab + bc + ca = -2023)
  : |a| + |b| + |c| = 102 := 
sorry

end abs_sum_of_roots_l721_72191


namespace determine_integer_n_l721_72116

theorem determine_integer_n (n : ℤ) :
  (n + 15 ≥ 16) ∧ (-5 * n < -10) → n = 3 :=
by
  sorry

end determine_integer_n_l721_72116


namespace ordered_triples_lcm_l721_72182

def lcm_equal (a b n : ℕ) : Prop :=
  a * b / (Nat.gcd a b) = n

theorem ordered_triples_lcm :
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z → 
  lcm_equal x y 48 → lcm_equal x z 900 → lcm_equal y z 180 →
  false :=
by sorry

end ordered_triples_lcm_l721_72182


namespace parallel_lines_have_equal_slopes_l721_72193

theorem parallel_lines_have_equal_slopes (a : ℝ) :
  (∃ a : ℝ, (∀ y : ℝ, 2 * a * y - 1 = 0) ∧ (∃ x y : ℝ, (3 * a - 1) * x + y - 1 = 0) 
  → (∃ a : ℝ, (1 / (2 * a)) = - (3 * a - 1))) 
→ a = 1/2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l721_72193


namespace linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l721_72153

-- Defining the first part of the problem
theorem linear_function_increasing_and_composition_eq_implies_values
  (a b : ℝ)
  (H1 : ∀ x y : ℝ, x < y → a * x + b < a * y + b)
  (H2 : ∀ x : ℝ, a * (a * x + b) + b = 16 * x + 5) :
  a = 4 ∧ b = 1 :=
by
  sorry

-- Defining the second part of the problem
theorem monotonic_gx_implies_m_range (m : ℝ)
  (H3 : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → (x2 + m) * (4 * x2 + 1) > (x1 + m) * (4 * x1 + 1)) :
  -9 / 4 ≤ m :=
by
  sorry

end linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l721_72153


namespace arithmetic_sequence_l721_72158

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l721_72158


namespace sales_in_fifth_month_l721_72107

-- Define the sales figures and average target
def s1 : ℕ := 6435
def s2 : ℕ := 6927
def s3 : ℕ := 6855
def s4 : ℕ := 7230
def s6 : ℕ := 6191
def s_target : ℕ := 6700
def n_months : ℕ := 6

-- Define the total sales and the required fifth month sale
def total_sales : ℕ := s_target * n_months
def s5 : ℕ := total_sales - (s1 + s2 + s3 + s4 + s6)

-- The main theorem statement we need to prove
theorem sales_in_fifth_month :
  s5 = 6562 :=
sorry

end sales_in_fifth_month_l721_72107


namespace Miss_Stevie_payment_l721_72187

theorem Miss_Stevie_payment:
  let painting_hours := 8
  let painting_rate := 15
  let painting_earnings := painting_hours * painting_rate
  let mowing_hours := 6
  let mowing_rate := 10
  let mowing_earnings := mowing_hours * mowing_rate
  let plumbing_hours := 4
  let plumbing_rate := 18
  let plumbing_earnings := plumbing_hours * plumbing_rate
  let total_earnings := painting_earnings + mowing_earnings + plumbing_earnings
  let discount := 0.10 * total_earnings
  let amount_paid := total_earnings - discount
  amount_paid = 226.80 :=
by
  sorry

end Miss_Stevie_payment_l721_72187


namespace smallest_possible_value_of_N_l721_72173

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l721_72173


namespace extreme_point_properties_l721_72194

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_point_properties (a x₁ x₂ : ℝ) (h₁ : 0 < a) (h₂ : a < 1 / 4) 
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : x₁ < x₂) :
  f x₁ a < 0 ∧ f x₂ a > (-1 / 2) := 
sorry

end extreme_point_properties_l721_72194


namespace price_reduction_daily_profit_l721_72114

theorem price_reduction_daily_profit
    (profit_per_item : ℕ)
    (avg_daily_sales : ℕ)
    (item_increase_per_unit_price_reduction : ℕ)
    (target_daily_profit : ℕ)
    (x : ℕ) :
    profit_per_item = 40 →
    avg_daily_sales = 20 →
    item_increase_per_unit_price_reduction = 2 →
    target_daily_profit = 1200 →

    ((profit_per_item - x) * (avg_daily_sales + item_increase_per_unit_price_reduction * x) = target_daily_profit) →
    x = 20 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_reduction_daily_profit_l721_72114


namespace randy_fifth_quiz_score_l721_72198

def scores : List ℕ := [90, 98, 92, 94]

def goal_average : ℕ := 94

def total_points (n : ℕ) (avg : ℕ) : ℕ := n * avg

def current_points (l : List ℕ) : ℕ := l.sum

def needed_score (total current : ℕ) : ℕ := total - current

theorem randy_fifth_quiz_score :
  needed_score (total_points 5 goal_average) (current_points scores) = 96 :=
by 
  sorry

end randy_fifth_quiz_score_l721_72198


namespace g_3_2_eq_neg3_l721_72157

noncomputable def f (x y : ℝ) : ℝ := x^3 * y^2 + 4 * x^2 * y - 15 * x

axiom f_symmetric : ∀ x y : ℝ, f x y = f y x
axiom f_2_4_eq_neg2 : f 2 4 = -2

noncomputable def g (x y : ℝ) : ℝ := (x^3 - 3 * x^2 * y + x * y^2) / (x^2 - y^2)

theorem g_3_2_eq_neg3 : g 3 2 = -3 := by
  sorry

end g_3_2_eq_neg3_l721_72157


namespace sequence_sum_l721_72170

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, S n = n^2 * a n) :
  ∀ n : ℕ, S n = 2 * n / (n + 1) := 
by 
  sorry

end sequence_sum_l721_72170


namespace distance_to_place_is_24_l721_72111

-- Definitions of the problem's conditions
def rowing_speed_still_water := 10    -- kmph
def current_velocity := 2             -- kmph
def round_trip_time := 5              -- hours

-- Effective speeds
def effective_speed_with_current := rowing_speed_still_water + current_velocity
def effective_speed_against_current := rowing_speed_still_water - current_velocity

-- Define the unknown distance D
variable (D : ℕ)

-- Define the times for each leg of the trip
def time_with_current := D / effective_speed_with_current
def time_against_current := D / effective_speed_against_current

-- The final theorem stating the round trip distance
theorem distance_to_place_is_24 :
  time_with_current + time_against_current = round_trip_time → D = 24 :=
by sorry

end distance_to_place_is_24_l721_72111


namespace mac_runs_faster_by_120_minutes_l721_72121

theorem mac_runs_faster_by_120_minutes :
  ∀ (D : ℝ), (D / 3 - D / 4 = 2) → 2 * 60 = 120 := by
  -- Definitions matching the conditions
  intro D
  intro h

  -- The proof is not required, hence using sorry
  sorry

end mac_runs_faster_by_120_minutes_l721_72121


namespace radius_of_tangent_circle_l721_72192

theorem radius_of_tangent_circle 
    (side_length : ℝ) 
    (tangent_angle : ℝ) 
    (sin_15 : ℝ)
    (circle_radius : ℝ) :
    side_length = 2 * Real.sqrt 3 →
    tangent_angle = 30 →
    sin_15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2) →
    circle_radius = 2 :=
by sorry

end radius_of_tangent_circle_l721_72192


namespace moles_of_H2O_formed_l721_72180

-- Define the initial conditions
def molesNaOH : ℕ := 2
def molesHCl : ℕ := 2

-- Balanced chemical equation behavior definition
def reaction (x y : ℕ) : ℕ := min x y

-- Statement of the problem to prove
theorem moles_of_H2O_formed :
  reaction molesNaOH molesHCl = 2 := by
  sorry

end moles_of_H2O_formed_l721_72180


namespace range_of_z_l721_72155

variable (x y z : ℝ)

theorem range_of_z (hx : x ≥ 0) (hy : y ≥ x) (hxy : 4*x + 3*y ≤ 12) 
(hz : z = (x + 2 * y + 3) / (x + 1)) : 
2 ≤ z ∧ z ≤ 6 :=
sorry

end range_of_z_l721_72155


namespace find_cd_l721_72112

noncomputable def g (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) :
  g 2 c d = -9 ∧ g (-1) c d = -19 ↔
  (c = 19/3 ∧ d = -7/3) :=
by
  sorry

end find_cd_l721_72112


namespace square_side_length_equals_4_l721_72142

theorem square_side_length_equals_4 (s : ℝ) (h : s^2 = 4 * s) : s = 4 :=
sorry

end square_side_length_equals_4_l721_72142


namespace geometric_sequence_common_ratio_l721_72171

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
    (h1 : a 1 = -1)
    (h2 : a 2 + a 3 = -2) :
    ∃ q : ℝ, (a 2 = a 1 * q) ∧ (a 3 = a 1 * q^2) ∧ (q = -2 ∨ q = 1) :=
sorry

end geometric_sequence_common_ratio_l721_72171


namespace anya_possible_wins_l721_72168

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end anya_possible_wins_l721_72168
