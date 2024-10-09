import Mathlib

namespace polygon_sides_l2220_222069

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 :=
by sorry

end polygon_sides_l2220_222069


namespace sufficient_not_necessary_condition_l2220_222077

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, (x^2 - 2 * x < 0 → 0 < x ∧ x < 4)) ∧ (∃ x : ℝ, (0 < x ∧ x < 4) ∧ ¬ (x^2 - 2 * x < 0)) :=
by
  sorry

end sufficient_not_necessary_condition_l2220_222077


namespace tom_books_problem_l2220_222094

theorem tom_books_problem 
  (original_books : ℕ)
  (books_sold : ℕ)
  (books_bought : ℕ)
  (h1 : original_books = 5)
  (h2 : books_sold = 4)
  (h3 : books_bought = 38) : 
  original_books - books_sold + books_bought = 39 :=
by
  sorry

end tom_books_problem_l2220_222094


namespace measure_angle_R_l2220_222020

-- Given conditions
variables {P Q R : Type}
variable {x : ℝ} -- x represents the measure of angles P and Q

-- Setting up the given conditions
def isosceles_triangle (P Q R : Type) (x : ℝ) : Prop :=
  x + x + (x + 40) = 180

-- Statement we need to prove
theorem measure_angle_R (P Q R : Type) (x : ℝ) (h : isosceles_triangle P Q R x) : ∃ r : ℝ, r = 86.67 :=
by {
  sorry
}

end measure_angle_R_l2220_222020


namespace sum_of_number_and_conjugate_l2220_222066

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l2220_222066


namespace tracy_customers_l2220_222004

theorem tracy_customers
  (total_customers : ℕ)
  (customers_bought_two_each : ℕ)
  (customers_bought_one_each : ℕ)
  (customers_bought_four_each : ℕ)
  (total_paintings_sold : ℕ)
  (h1 : total_customers = 20)
  (h2 : customers_bought_one_each = 12)
  (h3 : customers_bought_four_each = 4)
  (h4 : total_paintings_sold = 36)
  (h5 : 2 * customers_bought_two_each + customers_bought_one_each + 4 * customers_bought_four_each = total_paintings_sold) :
  customers_bought_two_each = 4 :=
by
  sorry

end tracy_customers_l2220_222004


namespace calculate_value_l2220_222049

theorem calculate_value : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end calculate_value_l2220_222049


namespace breakEvenBooks_l2220_222064

theorem breakEvenBooks (FC VC_per_book SP : ℝ) (hFC : FC = 56430) (hVC : VC_per_book = 8.25) (hSP : SP = 21.75) :
  ∃ x : ℕ, FC + (VC_per_book * x) = SP * x ∧ x = 4180 :=
by {
  sorry
}

end breakEvenBooks_l2220_222064


namespace intersection_points_l2220_222002

theorem intersection_points (k : ℝ) : ∃ (P : ℝ × ℝ), P = (1, 0) ∧ ∀ x y : ℝ, (kx - y - k = 0) → (x^2 + y^2 = 2) → ∃ y1 y2 : ℝ, (y = y1 ∨ y = y2) :=
by
  sorry

end intersection_points_l2220_222002


namespace integral_of_quadratic_has_minimum_value_l2220_222043

theorem integral_of_quadratic_has_minimum_value :
  ∃ m : ℝ, (∀ x : ℝ, x^2 + 2 * x + m ≥ -1) ∧ (∫ x in (1:ℝ)..(2:ℝ), x^2 + 2 * x = (16 / 3:ℝ)) :=
by sorry

end integral_of_quadratic_has_minimum_value_l2220_222043


namespace joan_seashells_count_l2220_222078

variable (total_seashells_given_to_sam : ℕ) (seashells_left_with_joan : ℕ)

theorem joan_seashells_count
  (h_given : total_seashells_given_to_sam = 43)
  (h_left : seashells_left_with_joan = 27) :
  total_seashells_given_to_sam + seashells_left_with_joan = 70 :=
sorry

end joan_seashells_count_l2220_222078


namespace enthusiasts_min_max_l2220_222070

-- Define the conditions
def total_students : ℕ := 100
def basketball_enthusiasts : ℕ := 63
def football_enthusiasts : ℕ := 75

-- Define the main proof problem
theorem enthusiasts_min_max :
  ∃ (common_enthusiasts : ℕ), 38 ≤ common_enthusiasts ∧ common_enthusiasts ≤ 63 :=
sorry

end enthusiasts_min_max_l2220_222070


namespace find_sum_of_squares_l2220_222087

-- Definitions for the conditions: a, b, and c are different prime numbers,
-- and their product equals five times their sum.

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def condition (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a * b * c = 5 * (a + b + c)

-- Statement of the proof problem.
theorem find_sum_of_squares (a b c : ℕ) (h : condition a b c) : a^2 + b^2 + c^2 = 78 :=
sorry

end find_sum_of_squares_l2220_222087


namespace solve_for_z_l2220_222026

open Complex

theorem solve_for_z (z : ℂ) (i : ℂ) (h1 : i = Complex.I) (h2 : z * i = 1 + i) : z = 1 - i :=
by sorry

end solve_for_z_l2220_222026


namespace relationship_between_x_y_z_l2220_222001

noncomputable def x := Real.sqrt 0.82
noncomputable def y := Real.sin 1
noncomputable def z := Real.log 7 / Real.log 3

theorem relationship_between_x_y_z : y < z ∧ z < x := 
by sorry

end relationship_between_x_y_z_l2220_222001


namespace average_weight_of_children_l2220_222084

theorem average_weight_of_children :
  let ages := [3, 4, 5, 6, 7]
  let regression_equation (x : ℕ) := 3 * x + 5
  let average l := (l.foldr (· + ·) 0) / l.length
  average (List.map regression_equation ages) = 20 :=
by
  sorry

end average_weight_of_children_l2220_222084


namespace no_real_solution_l2220_222060

theorem no_real_solution :
  ¬ ∃ x : ℝ, 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) :=
by
  sorry

end no_real_solution_l2220_222060


namespace in_proportion_d_value_l2220_222065

noncomputable def d_length (a b c : ℝ) : ℝ := (b * c) / a

theorem in_proportion_d_value :
  let a := 2
  let b := 3
  let c := 6
  d_length a b c = 9 := 
by
  sorry

end in_proportion_d_value_l2220_222065


namespace sum_of_eight_numbers_l2220_222056

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l2220_222056


namespace gold_copper_ratio_l2220_222015

theorem gold_copper_ratio (G C : ℕ) 
  (h1 : 19 * G + 9 * C = 18 * (G + C)) : 
  G = 9 * C :=
by
  sorry

end gold_copper_ratio_l2220_222015


namespace three_hundredth_term_without_squares_l2220_222097

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l2220_222097


namespace base_prime_rep_360_l2220_222045

-- Define the value 360 as n
def n : ℕ := 360

-- Function to compute the base prime representation.
noncomputable def base_prime_representation (n : ℕ) : ℕ :=
  -- Normally you'd implement the actual function to convert n to its base prime representation here
  sorry

-- The theorem statement claiming that the base prime representation of 360 is 213
theorem base_prime_rep_360 : base_prime_representation n = 213 := 
  sorry

end base_prime_rep_360_l2220_222045


namespace neon_sign_blink_interval_l2220_222057

theorem neon_sign_blink_interval :
  ∃ (b : ℕ), (∀ t : ℕ, t > 0 → (t % 9 = 0 ∧ t % b = 0 ↔ t % 45 = 0)) → b = 15 :=
by
  sorry

end neon_sign_blink_interval_l2220_222057


namespace maximum_integer_value_of_fraction_is_12001_l2220_222062

open Real

def max_fraction_value_12001 : Prop :=
  ∃ x : ℝ, (1 + 12 / (4 * x^2 + 12 * x + 8) : ℝ) = 12001

theorem maximum_integer_value_of_fraction_is_12001 :
  ∃ x : ℝ, 1 + (12 / (4 * x^2 + 12 * x + 8)) = 12001 :=
by
  -- Here you should provide the proof steps.
  sorry

end maximum_integer_value_of_fraction_is_12001_l2220_222062


namespace tony_age_in_6_years_l2220_222024

theorem tony_age_in_6_years (jacob_age : ℕ) (tony_age : ℕ) (h : jacob_age = 24) (h_half : tony_age = jacob_age / 2) : (tony_age + 6) = 18 :=
by
  sorry

end tony_age_in_6_years_l2220_222024


namespace solve_for_y_l2220_222082

theorem solve_for_y (y : ℤ) (h : (8 + 12 + 23 + 17 + y) / 5 = 15) : y = 15 :=
by {
  sorry
}

end solve_for_y_l2220_222082


namespace sphere_center_plane_intersection_l2220_222046

theorem sphere_center_plane_intersection
  (d e f : ℝ)
  (O : ℝ × ℝ × ℝ := (0, 0, 0))
  (A B C : ℝ × ℝ × ℝ)
  (p : ℝ)
  (hA : A ≠ O)
  (hB : B ≠ O)
  (hC : C ≠ O)
  (hA_coord : A = (2 * p, 0, 0))
  (hB_coord : B = (0, 2 * p, 0))
  (hC_coord : C = (0, 0, 2 * p))
  (h_sphere : (p, p, p) = (p, p, p)) -- we know that the center is (p, p, p)
  (h_plane : d * (1 / (2 * p)) + e * (1 / (2 * p)) + f * (1 / (2 * p)) = 1) :
  d / p + e / p + f / p = 2 := sorry

end sphere_center_plane_intersection_l2220_222046


namespace valid_number_count_l2220_222063

def is_valid_digit (d: Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def are_adjacent (d1 d2: Nat) : Bool :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 1) ∨
  (d1 = 5 ∧ (d2 = 1 ∨ d2 = 2)) ∨ 
  (d2 = 5 ∧ (d1 = 1 ∨ d1 = 2))

def count_valid_numbers : Nat :=
  sorry -- expression to count numbers according to given conditions.

theorem valid_number_count : count_valid_numbers = 36 :=
  sorry

end valid_number_count_l2220_222063


namespace average_gas_mileage_round_trip_l2220_222032

/-
A student drives 150 miles to university in a sedan that averages 25 miles per gallon.
The same student drives 150 miles back home in a minivan that averages 15 miles per gallon.
Calculate the average gas mileage for the entire round trip.
-/
theorem average_gas_mileage_round_trip (d1 d2 m1 m2 : ℝ) (h1 : d1 = 150) (h2 : m1 = 25) 
  (h3 : d2 = 150) (h4 : m2 = 15) : 
  (2 * d1) / ((d1/m1) + (d2/m2)) = 18.75 := by
  sorry

end average_gas_mileage_round_trip_l2220_222032


namespace leak_empties_tank_in_12_hours_l2220_222042

theorem leak_empties_tank_in_12_hours 
  (capacity : ℕ) (inlet_rate : ℕ) (net_emptying_time : ℕ) (leak_rate : ℤ) (leak_emptying_time : ℕ) :
  capacity = 5760 →
  inlet_rate = 4 →
  net_emptying_time = 8 →
  (inlet_rate - leak_rate : ℤ) = (capacity / (net_emptying_time * 60)) →
  leak_emptying_time = (capacity / leak_rate) →
  leak_emptying_time = 12 * 60 / 60 :=
by sorry

end leak_empties_tank_in_12_hours_l2220_222042


namespace trig_sum_roots_l2220_222036

theorem trig_sum_roots {θ a : Real} (hroots : ∀ x, x^2 - a * x + a = 0 → x = Real.sin θ ∨ x = Real.cos θ) :
  Real.cos (θ - 3 * Real.pi / 2) + Real.sin (3 * Real.pi / 2 + θ) = Real.sqrt 2 - 1 :=
by
  sorry

end trig_sum_roots_l2220_222036


namespace inequality_l2220_222088

noncomputable def x : ℝ := Real.sqrt 3
noncomputable def y : ℝ := Real.log 2 / Real.log 3
noncomputable def z : ℝ := Real.cos 2

theorem inequality : z < y ∧ y < x := by
  sorry

end inequality_l2220_222088


namespace spherical_ball_radius_l2220_222018

noncomputable def largest_spherical_ball_radius (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (table_z : ℝ) : ℝ :=
  let r := 4
  r

theorem spherical_ball_radius
  (inner_radius outer_radius : ℝ)
  (center : ℝ × ℝ × ℝ)
  (table_z : ℝ)
  (h1 : inner_radius = 3)
  (h2 : outer_radius = 5)
  (h3 : center = (4,0,1))
  (h4 : table_z = 0) :
  largest_spherical_ball_radius inner_radius outer_radius center table_z = 4 :=
by sorry

end spherical_ball_radius_l2220_222018


namespace average_comparison_l2220_222005

theorem average_comparison (x : ℝ) : 
    (14 + 32 + 53) / 3 = 3 + (21 + 47 + x) / 3 → 
    x = 22 :=
by 
  sorry

end average_comparison_l2220_222005


namespace sin_double_angle_l2220_222038

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end sin_double_angle_l2220_222038


namespace part_a_part_b_l2220_222031

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem part_a :
  ¬∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^3) :=
sorry

theorem part_b :
  ∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^4) :=
sorry

end part_a_part_b_l2220_222031


namespace ratio_of_only_B_to_both_A_and_B_l2220_222023

theorem ratio_of_only_B_to_both_A_and_B 
  (Total_households : ℕ)
  (Neither_brand : ℕ)
  (Only_A : ℕ)
  (Both_A_and_B : ℕ)
  (Total_households_eq : Total_households = 180)
  (Neither_brand_eq : Neither_brand = 80)
  (Only_A_eq : Only_A = 60)
  (Both_A_and_B_eq : Both_A_and_B = 10) :
  (Total_households = Neither_brand + Only_A + (Total_households - Neither_brand - Only_A - Both_A_and_B) + Both_A_and_B) →
  (Total_households - Neither_brand - Only_A - Both_A_and_B) / Both_A_and_B = 3 :=
by
  intro H
  sorry

end ratio_of_only_B_to_both_A_and_B_l2220_222023


namespace problem_statement_l2220_222009

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem problem_statement : f (g 3) = 120 ∧ f 3 = 8 :=
by sorry

end problem_statement_l2220_222009


namespace num_statements_imply_impl_l2220_222048

variable (p q r : Prop)

def cond1 := p ∧ q ∧ ¬r
def cond2 := ¬p ∧ q ∧ r
def cond3 := p ∧ q ∧ r
def cond4 := ¬p ∧ ¬q ∧ ¬r

def impl := ((p → ¬q) → ¬r)

theorem num_statements_imply_impl : 
  (cond1 p q r → impl p q r) ∧ 
  (cond3 p q r → impl p q r) ∧ 
  (cond4 p q r → impl p q r) ∧ 
  ¬(cond2 p q r → impl p q r) :=
by {
  sorry
}

end num_statements_imply_impl_l2220_222048


namespace matrix_B_cannot_be_obtained_from_matrix_A_l2220_222098

def A : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, -1, -1, 1],
  ![1, 1, -1, 1, -1]
]

def B : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, 1, -1, 1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, 1, -1, 1]
]

theorem matrix_B_cannot_be_obtained_from_matrix_A :
  A.det ≠ B.det := by
  sorry

end matrix_B_cannot_be_obtained_from_matrix_A_l2220_222098


namespace sum_of_numbers_mod_11_l2220_222067

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l2220_222067


namespace kevin_leaves_l2220_222075

theorem kevin_leaves (n : ℕ) (h : n > 1) : ∃ k : ℕ, n = k^3 ∧ n^2 = k^6 ∧ n = 8 := by
  sorry

end kevin_leaves_l2220_222075


namespace expand_product_l2220_222012

theorem expand_product (x : ℝ): (x + 4) * (x - 5 + 2) = x^2 + x - 12 :=
by 
  sorry

end expand_product_l2220_222012


namespace correct_statements_l2220_222080

-- Define the conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := 2 < a ∧ a < 3
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := a > 1

-- Prove the statements
theorem correct_statements (a : ℝ) : (p a → q a) ∧ (r a → q a) :=
by {
    sorry
}

end correct_statements_l2220_222080


namespace unique_zero_point_of_quadratic_l2220_222059

theorem unique_zero_point_of_quadratic (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - x - 1 = 0 → x = -1)) ↔ (a = 0 ∨ a = -1 / 4) :=
by
  sorry

end unique_zero_point_of_quadratic_l2220_222059


namespace max_xy_l2220_222033

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 8 * y = 48) : x * y ≤ 18 :=
sorry

end max_xy_l2220_222033


namespace triplet_divisibility_cond_l2220_222090

theorem triplet_divisibility_cond (a b c : ℤ) (hac : a ≥ 2) (hbc : b ≥ 2) (hcc : c ≥ 2) :
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ 
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 3 ∧ b = 15 ∧ c = 5) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 2 ∧ b = 8 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end triplet_divisibility_cond_l2220_222090


namespace lucas_can_afford_book_l2220_222016

-- Definitions from the conditions
def book_cost : ℝ := 28.50
def two_ten_dollar_bills : ℝ := 2 * 10
def five_one_dollar_bills : ℝ := 5 * 1
def six_quarters : ℝ := 6 * 0.25
def nickel_value : ℝ := 0.05

-- Given the conditions, we need to prove that if Lucas has at least 40 nickels, he can afford the book.
theorem lucas_can_afford_book (m : ℝ) (h : m >= 40) : 
  (two_ten_dollar_bills + five_one_dollar_bills + six_quarters + m * nickel_value) >= book_cost :=
by {
  sorry
}

end lucas_can_afford_book_l2220_222016


namespace purely_imaginary_condition_l2220_222089

theorem purely_imaginary_condition (x : ℝ) :
  (z : ℂ) → (z = (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I) →
  (x = 1 ↔ (∃ y : ℂ, z = y * Complex.I)) :=
by
  sorry

end purely_imaginary_condition_l2220_222089


namespace calvin_weight_after_one_year_l2220_222037

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l2220_222037


namespace intersection_eq_l2220_222028

/-
Define the sets A and B
-/
def setA : Set ℝ := {-1, 0, 1, 2}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

/-
Lean statement to prove the intersection A ∩ B equals {1, 2}
-/
theorem intersection_eq :
  setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_eq_l2220_222028


namespace possible_values_of_a_l2220_222047

def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a :
  {a | M a ⊆ P} = {1, -1, 0} :=
sorry

end possible_values_of_a_l2220_222047


namespace max_possible_acute_angled_triangles_l2220_222021
-- Define the sets of points on lines a and b
def maxAcuteAngledTriangles (n : Nat) : Nat :=
  let sum1 := (n * (n - 1) / 2)  -- Sum of first (n-1) natural numbers
  let sum2 := (sum1 * 50) - (n * (n - 1) * (2 * n - 1) / 6) -- Applying the given formula
  (2 * sum2)  -- Multiply by 2 for both colors of alternating points

-- Define the main theorem
theorem max_possible_acute_angled_triangles : maxAcuteAngledTriangles 50 = 41650 := by
  sorry

end max_possible_acute_angled_triangles_l2220_222021


namespace second_race_distance_l2220_222072

theorem second_race_distance (Va Vb Vc : ℝ) (D : ℝ)
  (h1 : Va / Vb = 10 / 9)
  (h2 : Va / Vc = 80 / 63)
  (h3 : Vb / Vc = D / (D - 100)) :
  D = 800 :=
sorry

end second_race_distance_l2220_222072


namespace angle_A_30_side_b_sqrt2_l2220_222052

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the dot product of vectors AB and AC is 2√3 times the area S, 
    then angle A equals 30 degrees --/
theorem angle_A_30 {a b c S : ℝ} (h : (a * b * Real.sqrt 3 * c * Real.sin (π / 6)) = 2 * Real.sqrt 3 * S) : 
  A = π / 6 :=
sorry

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the tangent of angles A, B, C are in the ratio 1:2:3 and c equals 1, 
    then side b equals √2 --/
theorem side_b_sqrt2 {A B C : ℝ} (a b c : ℝ) (h_tan_ratio : Real.tan A / Real.tan B = 1 / 2 ∧ Real.tan B / Real.tan C = 2 / 3)
  (h_c : c = 1) : b = Real.sqrt 2 :=
sorry

end angle_A_30_side_b_sqrt2_l2220_222052


namespace shelby_drive_rain_minutes_l2220_222085

theorem shelby_drive_rain_minutes
  (total_distance : ℝ)
  (total_time : ℝ)
  (sunny_speed : ℝ)
  (rainy_speed : ℝ)
  (t_sunny : ℝ)
  (t_rainy : ℝ) :
  total_distance = 20 →
  total_time = 50 →
  sunny_speed = 40 →
  rainy_speed = 25 →
  total_time = t_sunny + t_rainy →
  (sunny_speed / 60) * t_sunny + (rainy_speed / 60) * t_rainy = total_distance →
  t_rainy = 30 :=
by
  intros
  sorry

end shelby_drive_rain_minutes_l2220_222085


namespace decimal_equivalent_of_one_half_squared_l2220_222003

theorem decimal_equivalent_of_one_half_squared : (1 / 2 : ℝ) ^ 2 = 0.25 := 
sorry

end decimal_equivalent_of_one_half_squared_l2220_222003


namespace manager_monthly_salary_l2220_222040

theorem manager_monthly_salary :
  let avg_salary := 1200
  let num_employees := 20
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + 100
  let num_people_with_manager := num_employees + 1
  let new_total_salary := num_people_with_manager * new_avg_salary
  let manager_salary := new_total_salary - total_salary
  manager_salary = 3300 := by
  sorry

end manager_monthly_salary_l2220_222040


namespace sum_of_fractions_l2220_222044

theorem sum_of_fractions :
  (3 / 9) + (7 / 12) = (11 / 12) :=
by 
  sorry

end sum_of_fractions_l2220_222044


namespace find_value_of_square_sums_l2220_222091

variable (x y z : ℝ)

-- Define the conditions
def weighted_arithmetic_mean := (2 * x + 2 * y + 3 * z) / 8 = 9
def weighted_geometric_mean := Real.rpow (x^2 * y^2 * z^3) (1 / 7) = 6
def weighted_harmonic_mean := 7 / ((2 / x) + (2 / y) + (3 / z)) = 4

-- State the theorem to be proved
theorem find_value_of_square_sums
  (h1 : weighted_arithmetic_mean x y z)
  (h2 : weighted_geometric_mean x y z)
  (h3 : weighted_harmonic_mean x y z) :
  x^2 + y^2 + z^2 = 351 :=
by sorry

end find_value_of_square_sums_l2220_222091


namespace product_of_18396_and_9999_l2220_222030

theorem product_of_18396_and_9999 : 18396 * 9999 = 183962604 :=
by
  sorry

end product_of_18396_and_9999_l2220_222030


namespace women_in_first_class_equals_22_l2220_222083

def number_of_women (total_passengers : Nat) : Nat :=
  total_passengers * 50 / 100

def number_of_women_in_first_class (number_of_women : Nat) : Nat :=
  number_of_women * 15 / 100

theorem women_in_first_class_equals_22 (total_passengers : Nat) (h1 : total_passengers = 300) : 
  number_of_women_in_first_class (number_of_women total_passengers) = 22 :=
by
  sorry

end women_in_first_class_equals_22_l2220_222083


namespace radius_of_shorter_tank_l2220_222014

theorem radius_of_shorter_tank (h : ℝ) (r : ℝ) 
  (volume_eq : ∀ (π : ℝ), π * (10^2) * (2 * h) = π * (r^2) * h) : 
  r = 10 * Real.sqrt 2 := 
by 
  sorry

end radius_of_shorter_tank_l2220_222014


namespace odds_against_C_winning_l2220_222095

theorem odds_against_C_winning :
  let P_A := 2 / 7
  let P_B := 1 / 5
  let P_C := 1 - (P_A + P_B)
  (1 - P_C) / P_C = 17 / 18 :=
by
  sorry

end odds_against_C_winning_l2220_222095


namespace xy_sum_l2220_222035

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x + y = 2 :=
sorry

end xy_sum_l2220_222035


namespace darius_drive_miles_l2220_222079

theorem darius_drive_miles (total_miles : ℕ) (julia_miles : ℕ) (darius_miles : ℕ) 
  (h1 : total_miles = 1677) (h2 : julia_miles = 998) (h3 : total_miles = darius_miles + julia_miles) : 
  darius_miles = 679 :=
by
  sorry

end darius_drive_miles_l2220_222079


namespace system_of_equations_solutions_l2220_222022

theorem system_of_equations_solutions (x y z : ℝ) :
  (x^2 - y^2 + z = 27 / (x * y)) ∧ 
  (y^2 - z^2 + x = 27 / (y * z)) ∧ 
  (z^2 - x^2 + y = 27 / (z * x)) ↔ 
  (x = 3 ∧ y = 3 ∧ z = 3) ∨
  (x = -3 ∧ y = -3 ∧ z = 3) ∨
  (x = -3 ∧ y = 3 ∧ z = -3) ∨
  (x = 3 ∧ y = -3 ∧ z = -3) :=
by 
  sorry

end system_of_equations_solutions_l2220_222022


namespace angle_BDC_eq_88_l2220_222029

-- Define the problem scenario
variable (A B C : ℝ)
variable (α : ℝ)
variable (B1 B2 B3 C1 C2 C3 : ℝ)

-- Conditions provided
axiom angle_A_eq_42 : α = 42
axiom trisectors_ABC : B = B1 + B2 + B3 ∧ C = C1 + C2 + C3
axiom trisectors_eq : B1 = B2 ∧ B2 = B3 ∧ C1 = C2 ∧ C2 = C3
axiom angle_sum_ABC : α + B + C = 180

-- Proving the measure of ∠BDC
theorem angle_BDC_eq_88 :
  α + (B/3) + (C/3) = 88 :=
by
  sorry

end angle_BDC_eq_88_l2220_222029


namespace impossible_to_form_3x3_in_upper_left_or_right_l2220_222074

noncomputable def initial_positions : List (ℕ × ℕ) := 
  [(6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3)]

def sum_vertical (positions : List (ℕ × ℕ)) : ℕ :=
  positions.foldr (λ pos acc => pos.1 + acc) 0

theorem impossible_to_form_3x3_in_upper_left_or_right
  (initial_positions_set : List (ℕ × ℕ) := initial_positions)
  (initial_sum := sum_vertical initial_positions_set)
  (target_positions_upper_left : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
  (target_positions_upper_right : List (ℕ × ℕ) := [(1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8)])
  (target_sum_upper_left := sum_vertical target_positions_upper_left)
  (target_sum_upper_right := sum_vertical target_positions_upper_right) : 
  ¬ (initial_sum % 2 = 1 ∧ target_sum_upper_left % 2 = 0 ∧ target_sum_upper_right % 2 = 0) := sorry

end impossible_to_form_3x3_in_upper_left_or_right_l2220_222074


namespace min_shift_odd_func_l2220_222053

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem min_shift_odd_func (hφ : ∀ x : ℝ, f (x) = -f (-x + 2 * φ + (Real.pi / 3))) (hφ_positive : φ > 0) :
  φ = Real.pi / 6 :=
sorry

end min_shift_odd_func_l2220_222053


namespace fare_from_midpoint_C_to_B_l2220_222011

noncomputable def taxi_fare (d : ℝ) : ℝ :=
  if d <= 5 then 10.8 else 10.8 + 1.2 * (d - 5)

theorem fare_from_midpoint_C_to_B (x : ℝ) (h1 : taxi_fare x = 24)
    (h2 : taxi_fare (x - 0.46) = 24) :
    taxi_fare (x / 2) = 14.4 :=
by
  sorry

end fare_from_midpoint_C_to_B_l2220_222011


namespace smallest_number_of_blocks_needed_l2220_222000

/--
Given:
  A wall with the following properties:
  1. The wall is 100 feet long and 7 feet high.
  2. Blocks used are 1 foot high and either 1 foot or 2 feet long.
  3. Blocks cannot be cut.
  4. Vertical joins in the blocks must be staggered.
  5. The wall must be even on the ends.
Prove:
  The smallest number of blocks needed to build this wall is 353.
-/
theorem smallest_number_of_blocks_needed :
  let length := 100
  let height := 7
  let block_height := 1
  (∀ b : ℕ, b = 1 ∨ b = 2) →
  ∃ (blocks_needed : ℕ), blocks_needed = 353 :=
by sorry

end smallest_number_of_blocks_needed_l2220_222000


namespace year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l2220_222081

-- Define what it means to be a leap year based on the given conditions.
def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ (y % 4 = 0 ∧ y % 100 ≠ 0)

-- Define the specific years we are examining.
def year_1800 := 1800
def year_1992 := 1992
def year_1994 := 1994
def year_2040 := 2040

-- Assertions about whether each year is a leap year or a common year
theorem year_1800_is_common : ¬ is_leap_year year_1800 :=
  by sorry

theorem year_1992_is_leap : is_leap_year year_1992 :=
  by sorry

theorem year_1994_is_common : ¬ is_leap_year year_1994 :=
  by sorry

theorem year_2040_is_leap : is_leap_year year_2040 :=
  by sorry

end year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l2220_222081


namespace inequality_holds_for_all_x_l2220_222027

theorem inequality_holds_for_all_x (a : ℝ) (h : -1 < a ∧ a < 2) :
  ∀ x : ℝ, -3 < (x^2 + a * x - 2) / (x^2 - x + 1) ∧ (x^2 + a * x - 2) / (x^2 - x + 1) < 2 :=
by
  intro x
  sorry

end inequality_holds_for_all_x_l2220_222027


namespace arithmetic_sequence_15th_term_eq_53_l2220_222099

theorem arithmetic_sequence_15th_term_eq_53 (a1 : ℤ) (d : ℤ) (n : ℕ) (a_15 : ℤ) 
    (h1 : a1 = -3)
    (h2 : d = 4)
    (h3 : n = 15)
    (h4 : a_15 = a1 + (n - 1) * d) : 
    a_15 = 53 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end arithmetic_sequence_15th_term_eq_53_l2220_222099


namespace geometry_problem_z_eq_87_deg_l2220_222034

noncomputable def measure_angle_z (ABC ABD ADB : Real) : Real :=
  43 -- \angle ADB

theorem geometry_problem_z_eq_87_deg
  (ABC : Real)
  (h1 : ABC = 130)
  (ABD : Real)
  (h2 : ABD = 50)
  (ADB : Real)
  (h3 : ADB = 43) :
  measure_angle_z ABC ABD ADB = 87 :=
by
  unfold measure_angle_z
  sorry

end geometry_problem_z_eq_87_deg_l2220_222034


namespace correct_probability_l2220_222007

noncomputable def T : ℕ := 44
noncomputable def num_books : ℕ := T - 35
noncomputable def n : ℕ := 9
noncomputable def favorable_outcomes : ℕ := (Nat.choose n 6) * 2
noncomputable def total_arrangements : ℕ := (Nat.factorial n)
noncomputable def probability : Rat := (favorable_outcomes : ℚ) / (total_arrangements : ℚ)
noncomputable def m : ℕ := 1
noncomputable def p : Nat := Nat.gcd 168 362880
noncomputable def final_prob_form : Rat := 1 / 2160
noncomputable def answer : ℕ := m + 2160

theorem correct_probability : 
  probability = final_prob_form ∧ answer = 2161 := 
by
  sorry

end correct_probability_l2220_222007


namespace four_prime_prime_l2220_222068

-- Define the function based on the given condition
def q' (q : ℕ) : ℕ := 3 * q - 3

-- The statement to prove
theorem four_prime_prime : (q' (q' 4)) = 24 := by
  sorry

end four_prime_prime_l2220_222068


namespace A_inter_B_eq_l2220_222039

def A := {x : ℤ | 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 3}
def B := {x : ℤ | 5 ≤ x ∧ x < 9}

theorem A_inter_B_eq : A ∩ B = {5, 6, 7} :=
by sorry

end A_inter_B_eq_l2220_222039


namespace problem_l2220_222050

def f : ℕ → ℕ → ℕ := sorry

theorem problem (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) :
  2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1) ∧
  (f m 0 = 0) ∧ (f 0 n = 0) → f m n = m * n :=
by sorry

end problem_l2220_222050


namespace seventeen_power_sixty_three_mod_seven_l2220_222013

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l2220_222013


namespace avg_height_trees_l2220_222076

-- Assuming heights are defined as h1, h2, ..., h7 with known h2
noncomputable def avgHeight (h1 h2 h3 h4 h5 h6 h7 : ℝ) : ℝ := 
  (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7

theorem avg_height_trees :
  ∃ (h1 h3 h4 h5 h6 h7 : ℝ), 
    h2 = 15 ∧ 
    (h1 = 2 * h2 ∨ h1 = 3 * h2) ∧
    (h3 = h2 / 3 ∨ h3 = h2 / 2) ∧
    (h4 = 2 * h3 ∨ h4 = 3 * h3 ∨ h4 = h3 / 2 ∨ h4 = h3 / 3) ∧
    (h5 = 2 * h4 ∨ h5 = 3 * h4 ∨ h5 = h4 / 2 ∨ h5 = h4 / 3) ∧
    (h6 = 2 * h5 ∨ h6 = 3 * h5 ∨ h6 = h5 / 2 ∨ h6 = h5 / 3) ∧
    (h7 = 2 * h6 ∨ h7 = 3 * h6 ∨ h7 = h6 / 2 ∨ h7 = h6 / 3) ∧
    avgHeight h1 h2 h3 h4 h5 h6 h7 = 26.4 :=
by
  sorry

end avg_height_trees_l2220_222076


namespace line_passes_through_fixed_point_l2220_222017

theorem line_passes_through_fixed_point (m : ℝ) : 
  (2 + m) * (-1) + (1 - 2 * m) * (-2) + 4 - 3 * m = 0 :=
by
  sorry

end line_passes_through_fixed_point_l2220_222017


namespace common_difference_arithmetic_sequence_l2220_222071

theorem common_difference_arithmetic_sequence :
  ∃ d : ℝ, (d ≠ 0) ∧ (∀ (n : ℕ), a_n = 1 + (n-1) * d) ∧ ((1 + 2 * d)^2 = 1 * (1 + 8 * d)) → d = 1 :=
by
  sorry

end common_difference_arithmetic_sequence_l2220_222071


namespace base6_divisible_by_13_l2220_222008

theorem base6_divisible_by_13 (d : ℕ) (h : d < 6) : 13 ∣ (435 + 42 * d) ↔ d = 5 := 
by
  -- Proof implementation will go here, but is currently omitted
  sorry

end base6_divisible_by_13_l2220_222008


namespace probability_of_D_l2220_222061

theorem probability_of_D (P : Type) (A B C D : P) 
  (pA pB pC pD : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/3) 
  (hC : pC = 1/6) 
  (hSum : pA + pB + pC + pD = 1) :
  pD = 1/4 :=
by 
  sorry

end probability_of_D_l2220_222061


namespace multiples_of_2_correct_multiples_of_3_correct_l2220_222096

def numbers : Set ℕ := {28, 35, 40, 45, 53, 10, 78}

def multiples_of_2_in_numbers : Set ℕ := {n ∈ numbers | n % 2 = 0}
def multiples_of_3_in_numbers : Set ℕ := {n ∈ numbers | n % 3 = 0}

theorem multiples_of_2_correct :
  multiples_of_2_in_numbers = {28, 40, 10, 78} :=
sorry

theorem multiples_of_3_correct :
  multiples_of_3_in_numbers = {45, 78} :=
sorry

end multiples_of_2_correct_multiples_of_3_correct_l2220_222096


namespace basketball_scores_l2220_222055

theorem basketball_scores : ∃ (scores : Finset ℕ), 
  scores = { x | ∃ a b : ℕ, a + b = 7 ∧ x = 2 * a + 3 * b } ∧ scores.card = 8 :=
by
  sorry

end basketball_scores_l2220_222055


namespace calculation_result_l2220_222086

theorem calculation_result :
  3 * 3^3 + 4^7 / 4^5 = 97 :=
by
  sorry

end calculation_result_l2220_222086


namespace quadratic_no_real_solution_l2220_222073

theorem quadratic_no_real_solution 
  (a b c : ℝ) 
  (h1 : (2 * a)^2 - 4 * b^2 > 0) 
  (h2 : (2 * b)^2 - 4 * c^2 > 0) : 
  (2 * c)^2 - 4 * a^2 < 0 :=
sorry

end quadratic_no_real_solution_l2220_222073


namespace solve_inequality_l2220_222058

variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x

-- Prove the main statement
theorem solve_inequality (h : ∀ x : ℝ, f (f x) = x) : ∀ x : ℝ, f (f x) = x := 
by
  sorry

end solve_inequality_l2220_222058


namespace missing_score_find_missing_score_l2220_222092

theorem missing_score
  (score1 score2 score3 score4 mean total : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89)
  (hTotal : total = 445) :
  score1 + score2 + score3 + score4 + x = total :=
by
  sorry

theorem find_missing_score
  (score1 score2 score3 score4 mean : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89) :
  (score1 + score2 + score3 + score4 + x) / 5 = mean
  → x = 90 :=
by
  sorry

end missing_score_find_missing_score_l2220_222092


namespace tile_C_is_TileIV_l2220_222093

-- Define the tiles with their respective sides
structure Tile :=
(top right bottom left : ℕ)

def TileI : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileII : Tile := { top := 6, right := 3, bottom := 1, left := 5 }
def TileIII : Tile := { top := 5, right := 7, bottom := 2, left := 3 }
def TileIV : Tile := { top := 3, right := 5, bottom := 7, left := 2 }

-- Define Rectangles for reasoning
inductive Rectangle
| A
| B
| C
| D

open Rectangle

-- Define the mathematical statement to prove
theorem tile_C_is_TileIV : ∃ tile, tile = TileIV :=
  sorry

end tile_C_is_TileIV_l2220_222093


namespace optimal_direction_l2220_222010

-- Define the conditions as hypotheses
variables (a : ℝ) (V_first V_second : ℝ) (d : ℝ)
variable (speed_rel : V_first = 2 * V_second)
variable (dist : d = a)

-- Create a theorem statement for the problem
theorem optimal_direction (H : d = a) (vel_rel : V_first = 2 * V_second) : true := 
  sorry

end optimal_direction_l2220_222010


namespace sin_cos_of_angle_l2220_222006

theorem sin_cos_of_angle (a : ℝ) (h₀ : a ≠ 0) :
  ∃ (s c : ℝ), (∃ (k : ℝ), s = k * (8 / 17) ∧ c = -k * (15 / 17) ∧ k = if a > 0 then 1 else -1) :=
by
  sorry

end sin_cos_of_angle_l2220_222006


namespace toy_train_produces_5_consecutive_same_tune_l2220_222041

noncomputable def probability_same_tune (plays : ℕ) (p : ℚ) (tunes : ℕ) : ℚ :=
  p ^ plays

theorem toy_train_produces_5_consecutive_same_tune :
  probability_same_tune 5 (1/3) 3 = 1/243 :=
by
  sorry

end toy_train_produces_5_consecutive_same_tune_l2220_222041


namespace bracelet_arrangements_l2220_222054

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def distinct_arrangements : ℕ := factorial 8 / (8 * 2)

theorem bracelet_arrangements : distinct_arrangements = 2520 :=
by
  sorry

end bracelet_arrangements_l2220_222054


namespace quadratic_behavior_l2220_222019

theorem quadratic_behavior (x : ℝ) : x < 3 → ∃ y : ℝ, y = 5 * (x - 3) ^ 2 + 2 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 3 ∧ x2 < 3 → (5 * (x1 - 3) ^ 2 + 2) > (5 * (x2 - 3) ^ 2 + 2) := 
by
  sorry

end quadratic_behavior_l2220_222019


namespace identify_faulty_key_l2220_222025

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l2220_222025


namespace current_bottle_caps_l2220_222051

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem current_bottle_caps : initial_bottle_caps - lost_bottle_caps = 25 :=
by
  -- sorry is used to skip the proof
  sorry

end current_bottle_caps_l2220_222051
