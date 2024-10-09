import Mathlib

namespace sum_ab_system_1_l1337_133744

theorem sum_ab_system_1 {a b : ℝ} 
  (h1 : a^3 - a^2 + a - 5 = 0) 
  (h2 : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := 
by 
  sorry

end sum_ab_system_1_l1337_133744


namespace half_radius_of_circle_y_l1337_133702

theorem half_radius_of_circle_y
  (r_x r_y : ℝ)
  (hx : π * r_x ^ 2 = π * r_y ^ 2)
  (hc : 2 * π * r_x = 10 * π) :
  r_y / 2 = 2.5 :=
by
  sorry

end half_radius_of_circle_y_l1337_133702


namespace smallest_positive_integer_l1337_133793

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 60606 * n = 273 :=
sorry

end smallest_positive_integer_l1337_133793


namespace sum_f_1_to_10_l1337_133761

-- Define the function f with the properties given.

def f (x : ℝ) : ℝ := sorry

-- Specify the conditions of the problem
local notation "R" => ℝ

axiom odd_function : ∀ (x : R), f (-x) = -f (x)
axiom periodicity : ∀ (x : R), f (x + 3) = f (x)
axiom f_neg1 : f (-1) = 1

-- State the theorem to be proved
theorem sum_f_1_to_10 : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry
end sum_f_1_to_10_l1337_133761


namespace maya_total_pages_l1337_133723

def books_first_week : ℕ := 5
def pages_per_book_first_week : ℕ := 300
def books_second_week := books_first_week * 2
def pages_per_book_second_week : ℕ := 350
def books_third_week := books_first_week * 3
def pages_per_book_third_week : ℕ := 400

def total_pages_first_week : ℕ := books_first_week * pages_per_book_first_week
def total_pages_second_week : ℕ := books_second_week * pages_per_book_second_week
def total_pages_third_week : ℕ := books_third_week * pages_per_book_third_week

def total_pages_maya_read : ℕ := total_pages_first_week + total_pages_second_week + total_pages_third_week

theorem maya_total_pages : total_pages_maya_read = 11000 := by
  sorry

end maya_total_pages_l1337_133723


namespace cubic_root_sum_l1337_133747

-- Assume we have three roots a, b, and c of the polynomial x^3 - 3x - 2 = 0
variables {a b c : ℝ}

-- Using Vieta's formulas for the polynomial x^3 - 3x - 2 = 0
axiom Vieta1 : a + b + c = 0
axiom Vieta2 : a * b + a * c + b * c = -3
axiom Vieta3 : a * b * c = -2

-- The proof that the given expression evaluates to 9
theorem cubic_root_sum:
  a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2 = 9 :=
by
  sorry

end cubic_root_sum_l1337_133747


namespace Zhu_Zaiyu_problem_l1337_133737

theorem Zhu_Zaiyu_problem
  (f : ℕ → ℝ) 
  (q : ℝ)
  (h_geom_seq : ∀ n, f (n+1) = q * f n)
  (h_octave : f 13 = 2 * f 1) :
  (f 7) / (f 3) = 2^(1/3) :=
by
  sorry

end Zhu_Zaiyu_problem_l1337_133737


namespace no_valid_bases_l1337_133759

theorem no_valid_bases
  (x y : ℕ)
  (h1 : 4 * x + 9 = 4 * y + 1)
  (h2 : 4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9)
  (hx : x > 1)
  (hy : y > 1)
  : false :=
by
  sorry

end no_valid_bases_l1337_133759


namespace problem_solution_l1337_133730

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := 
by
  sorry

end problem_solution_l1337_133730


namespace odd_function_property_l1337_133714

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by
  -- The proof is omitted as per the instruction
  sorry

end odd_function_property_l1337_133714


namespace sum_inequality_l1337_133727

noncomputable def f (x : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2)

theorem sum_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 1) : 
  f x + f y + f z ≥ 0 :=
by
  sorry

end sum_inequality_l1337_133727


namespace calendar_sum_l1337_133780

theorem calendar_sum (n : ℕ) : 
    n + (n + 7) + (n + 14) = 3 * n + 21 :=
by sorry

end calendar_sum_l1337_133780


namespace chord_length_of_intersection_l1337_133788

theorem chord_length_of_intersection 
  (A B C : ℝ) (x0 y0 r : ℝ)
  (line_eq : A * x0 + B * y0 + C = 0)
  (circle_eq : (x0 - 1)^2 + (y0 - 3)^2 = r^2) 
  (A_line : A = 4) (B_line : B = -3) (C_line : C = 0) 
  (x0_center : x0 = 1) (y0_center : y0 = 3) (r_circle : r^2 = 10) :
  2 * (Real.sqrt (r^2 - ((A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2)))^2)) = 6 :=
by
  sorry

end chord_length_of_intersection_l1337_133788


namespace problem1_problem2_problem3_problem4_l1337_133785

open Set

def M : Set ℝ := { x | x > 3 / 2 }
def N : Set ℝ := { x | x < 1 ∨ x > 3 }
def R := {x : ℝ | 1 ≤ x ∧ x ≤ 3 / 2}

theorem problem1 : M = { x | 2 * x - 3 > 0 } := sorry
theorem problem2 : N = { x | (x - 3) * (x - 1) > 0 } := sorry
theorem problem3 : M ∩ N = { x | x > 3 } := sorry
theorem problem4 : (M ∪ N)ᶜ = R := sorry

end problem1_problem2_problem3_problem4_l1337_133785


namespace gcd_determinant_l1337_133713

theorem gcd_determinant (a b : ℤ) (h : Int.gcd a b = 1) :
  Int.gcd (a + b) (a^2 + b^2 - a * b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a * b) = 3 :=
sorry

end gcd_determinant_l1337_133713


namespace k_gt_4_l1337_133763

theorem k_gt_4 {x y k : ℝ} (h1 : 2 * x + y = 2 * k - 1) (h2 : x + 2 * y = -4) (h3 : x + y > 1) : k > 4 :=
by
  -- This 'sorry' serves as a placeholder for the actual proof steps
  sorry

end k_gt_4_l1337_133763


namespace remainder_9_pow_2023_div_50_l1337_133777

theorem remainder_9_pow_2023_div_50 : (9 ^ 2023) % 50 = 41 := by
  sorry

end remainder_9_pow_2023_div_50_l1337_133777


namespace part_a_part_b_l1337_133707

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end part_a_part_b_l1337_133707


namespace find_x_l1337_133708

theorem find_x (x : ℝ) (h : 2 * x - 3 * x + 5 * x = 80) : x = 20 :=
by 
  -- placeholder for proof
  sorry 

end find_x_l1337_133708


namespace total_students_registered_l1337_133766

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end total_students_registered_l1337_133766


namespace area_of_rectangle_l1337_133742

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l1337_133742


namespace price_is_219_l1337_133749

noncomputable def discount_coupon1 (price : ℝ) : ℝ :=
  if price > 50 then 0.1 * price else 0

noncomputable def discount_coupon2 (price : ℝ) : ℝ :=
  if price > 100 then 20 else 0

noncomputable def discount_coupon3 (price : ℝ) : ℝ :=
  if price > 100 then 0.18 * (price - 100) else 0

noncomputable def more_savings_coupon1 (price : ℝ) : Prop :=
  discount_coupon1 price > discount_coupon2 price ∧ discount_coupon1 price > discount_coupon3 price

theorem price_is_219 (price : ℝ) :
  more_savings_coupon1 price → price = 219 :=
by
  sorry

end price_is_219_l1337_133749


namespace function_relationship_minimize_total_cost_l1337_133758

noncomputable def y (a x : ℕ) : ℕ :=
6400 * x + 50 * a + 100 * a^2 / (x - 1)

theorem function_relationship (a : ℕ) (hx : 2 ≤ x) : 
  y a x = 6400 * x + 50 * a + 100 * a^2 / (x - 1) :=
by sorry

theorem minimize_total_cost (a : ℕ) (hx : 2 ≤ x) (ha : a = 56) : 
  y a x ≥ 1650 * a + 6400 ∧ (x = 8) :=
by sorry

end function_relationship_minimize_total_cost_l1337_133758


namespace parabola_no_intersection_inequality_l1337_133764

-- Definitions for the problem
theorem parabola_no_intersection_inequality
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (a * x^2 + b * x + c ≠ x) ∧ (a * x^2 + b * x + c ≠ -x)) :
  |b^2 - 4 * a * c| > 1 := 
sorry

end parabola_no_intersection_inequality_l1337_133764


namespace unique_two_digit_number_l1337_133797

theorem unique_two_digit_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 99) : 
  (13 * n) % 100 = 42 → n = 34 :=
by
  sorry

end unique_two_digit_number_l1337_133797


namespace max_average_growth_rate_l1337_133728

theorem max_average_growth_rate 
  (P1 P2 : ℝ) (M : ℝ)
  (h1 : P1 + P2 = M) : 
  (1 + (M / 2))^2 ≥ (1 + P1) * (1 + P2) := 
by
  -- AM-GM Inequality application and other mathematical steps go here.
  sorry

end max_average_growth_rate_l1337_133728


namespace part_I_part_II_l1337_133757

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end part_I_part_II_l1337_133757


namespace Jason_age_l1337_133716

theorem Jason_age : ∃ J K : ℕ, (J = 7 * K) ∧ (J + 4 = 3 * (2 * (K + 2))) ∧ (J = 56) :=
by
  sorry

end Jason_age_l1337_133716


namespace find_k_l1337_133701

-- Define the conditions
variables (k : ℝ) -- the variable k
variables (x1 : ℝ) -- x1 coordinate of point A on the graph y = k/x
variable (AREA_ABCD : ℝ := 10) -- the area of the quadrilateral ABCD

-- The statement to be proven
theorem find_k (k : ℝ) (h1 : ∀ x1 : ℝ, (0 < x1 ∧ 2 * abs k = AREA_ABCD → x1 * abs k * 2 = AREA_ABCD)) : k = -5 :=
sorry

end find_k_l1337_133701


namespace proportionality_intersect_calculation_l1337_133738

variables {x1 x2 y1 y2 : ℝ}

/-- Proof that (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15,
    given specific conditions on x1, x2, y1, and y2. -/
theorem proportionality_intersect_calculation
  (h1 : y1 = 5 / x1) 
  (h2 : y2 = 5 / x2)
  (h3 : x1 * y1 = 5)
  (h4 : x2 * y2 = 5)
  (h5 : x1 = -x2)
  (h6 : y1 = -y2) :
  (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15 := 
sorry

end proportionality_intersect_calculation_l1337_133738


namespace total_fraction_inspected_l1337_133705

-- Define the fractions of products inspected by John, Jane, and Roy.
variables (J N R : ℝ)
-- Define the rejection rates for John, Jane, and Roy.
variables (rJ rN rR : ℝ)
-- Define the total rejection rate.
variable (r_total : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  (rJ = 0.007) ∧ (rN = 0.008) ∧ (rR = 0.01) ∧ (r_total = 0.0085) ∧
  (0.007 * J + 0.008 * N + 0.01 * R = 0.0085)

-- The proof statement that the total fraction of products inspected is 1.
theorem total_fraction_inspected (h : conditions J N R rJ rN rR r_total) : J + N + R = 1 :=
sorry

end total_fraction_inspected_l1337_133705


namespace reach_any_composite_from_4_l1337_133726

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

def can_reach (A : ℕ) : Prop :=
  ∀ n : ℕ, is_composite n → ∃ seq : ℕ → ℕ, seq 0 = A ∧ seq (n + 1) - seq n ∣ seq n ∧ seq (n + 1) ≠ seq n ∧ seq (n + 1) ≠ 1 ∧ seq (n + 1) = n

theorem reach_any_composite_from_4 : can_reach 4 :=
  sorry

end reach_any_composite_from_4_l1337_133726


namespace no_solution_for_inequality_l1337_133754

theorem no_solution_for_inequality (x : ℝ) (h : |x| > 2) : ¬ (5 * x^2 + 6 * x + 8 < 0) := 
by
  sorry

end no_solution_for_inequality_l1337_133754


namespace total_students_is_17_l1337_133769

def total_students_in_class (students_liking_both_baseball_football : ℕ)
                             (students_only_baseball : ℕ)
                             (students_only_football : ℕ)
                             (students_liking_basketball_as_well : ℕ)
                             (students_liking_basketball_and_football_only : ℕ)
                             (students_liking_all_three : ℕ)
                             (students_liking_none : ℕ) : ℕ :=
  students_liking_both_baseball_football -
  students_liking_all_three +
  students_only_baseball +
  students_only_football +
  students_liking_basketball_and_football_only +
  students_liking_all_three +
  students_liking_none +
  (students_liking_basketball_as_well -
   (students_liking_all_three +
    students_liking_basketball_and_football_only))

theorem total_students_is_17 :
    total_students_in_class 7 3 4 2 1 2 5 = 17 :=
by sorry

end total_students_is_17_l1337_133769


namespace total_sheets_l1337_133741

-- Define the conditions
def sheets_in_bundle : ℕ := 10
def bundles : ℕ := 3
def additional_sheets : ℕ := 8

-- Theorem to prove the total number of sheets Jungkook has
theorem total_sheets : bundles * sheets_in_bundle + additional_sheets = 38 := by
  sorry

end total_sheets_l1337_133741


namespace intersection_A_B_l1337_133703

def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_B_l1337_133703


namespace maximum_garden_area_l1337_133715

theorem maximum_garden_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 400) : 
  l * w ≤ 10000 :=
by {
  -- proving the theorem
  sorry
}

end maximum_garden_area_l1337_133715


namespace nth_inequality_l1337_133710

theorem nth_inequality (x : ℝ) (n : ℕ) (h_x_pos : 0 < x) : x + (n^n / x^n) ≥ n + 1 := 
sorry

end nth_inequality_l1337_133710


namespace sum_of_coefficients_factors_l1337_133762

theorem sum_of_coefficients_factors :
  ∃ (a b c d e : ℤ), 
    (343 * (x : ℤ)^3 + 125 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 51) :=
sorry

end sum_of_coefficients_factors_l1337_133762


namespace geometric_sequence_problem_l1337_133729

-- Definitions
def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

-- Problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ)
    (h_geom : is_geom_seq a q)
    (h1 : a 3 * a 7 = 8)
    (h2 : a 4 + a 6 = 6) :
    a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l1337_133729


namespace no_solution_to_system_l1337_133735

open Real

theorem no_solution_to_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^(1/3) - y^(1/3) - z^(1/3) = 64) ∧ (x^(1/4) - y^(1/4) - z^(1/4) = 32) ∧ (x^(1/6) - y^(1/6) - z^(1/6) = 8) → False := by
  sorry

end no_solution_to_system_l1337_133735


namespace count_valid_rods_l1337_133736

def isValidRodLength (d : ℕ) : Prop :=
  5 ≤ d ∧ d < 27

def countValidRodLengths (lower upper : ℕ) : ℕ :=
  upper - lower + 1

theorem count_valid_rods :
  let valid_rods_count := countValidRodLengths 5 26
  valid_rods_count = 22 :=
by
  sorry

end count_valid_rods_l1337_133736


namespace minimum_value_l1337_133783

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem minimum_value : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ f (-1) :=
by
  sorry

end minimum_value_l1337_133783


namespace salt_solution_l1337_133767

variable (x : ℝ) (v_water : ℝ) (c_initial : ℝ) (c_final : ℝ)

theorem salt_solution (h1 : v_water = 1) (h2 : c_initial = 0.60) (h3 : c_final = 0.20)
  (h4 : (v_water + x) * c_final = x * c_initial) :
  x = 0.5 :=
by {
  sorry
}

end salt_solution_l1337_133767


namespace luke_trays_l1337_133750

theorem luke_trays 
  (carries_per_trip : ℕ)
  (trips : ℕ)
  (second_table_trays : ℕ)
  (total_trays : carries_per_trip * trips = 36)
  (second_table_value : second_table_trays = 16) : 
  carries_per_trip * trips - second_table_trays = 20 :=
by sorry

end luke_trays_l1337_133750


namespace total_toys_l1337_133778

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l1337_133778


namespace total_turtles_taken_l1337_133771

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l1337_133771


namespace distance_between_A_and_B_l1337_133752

noncomputable def time_from_A_to_B (D : ℝ) : ℝ := D / 200

noncomputable def time_from_B_to_A (D : ℝ) : ℝ := time_from_A_to_B D + 3

def condition (D : ℝ) : Prop := 
  D = 100 * (time_from_B_to_A D)

theorem distance_between_A_and_B :
  ∃ D : ℝ, condition D ∧ D = 600 :=
by
  sorry

end distance_between_A_and_B_l1337_133752


namespace mitzi_money_left_l1337_133721

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end mitzi_money_left_l1337_133721


namespace polynomial_not_factorable_l1337_133740

theorem polynomial_not_factorable :
  ¬ ∃ (A B : Polynomial ℤ), A.degree < 5 ∧ B.degree < 5 ∧ A * B = (Polynomial.C 1 * Polynomial.X ^ 5 - Polynomial.C 3 * Polynomial.X ^ 4 + Polynomial.C 6 * Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 6) :=
by
  sorry

end polynomial_not_factorable_l1337_133740


namespace quadratic_inequality_solution_set_l1337_133772

theorem quadratic_inequality_solution_set (a b c : ℝ) (h₁ : a < 0) (h₂ : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
sorry

end quadratic_inequality_solution_set_l1337_133772


namespace intersection_of_A_and_B_l1337_133709

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_of_A_and_B : (A ∩ B) = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l1337_133709


namespace regular_tetrahedron_of_angle_l1337_133787

-- Definition and condition from the problem
def angle_between_diagonals (shape : Type _) (adj_sides_diag_angle : ℝ) : Prop :=
  adj_sides_diag_angle = 60

-- Theorem stating the problem in Lean 4
theorem regular_tetrahedron_of_angle (shape : Type _) (adj_sides_diag_angle : ℝ) 
  (h : angle_between_diagonals shape adj_sides_diag_angle) : 
  shape = regular_tetrahedron :=
sorry

end regular_tetrahedron_of_angle_l1337_133787


namespace calculation_correct_l1337_133719

theorem calculation_correct :
  (Int.ceil ((15 : ℚ) / 8 * ((-35 : ℚ) / 4)) - 
  Int.floor (((15 : ℚ) / 8) * Int.floor ((-35 : ℚ) / 4 + (1 : ℚ) / 4))) = 1 := by
  sorry

end calculation_correct_l1337_133719


namespace shortest_remaining_side_l1337_133781

theorem shortest_remaining_side (a b : ℝ) (h1 : a = 7) (h2 : b = 24) (right_triangle : ∃ c, c^2 = a^2 + b^2) : a = 7 :=
by
  sorry

end shortest_remaining_side_l1337_133781


namespace relationship_between_x_x2_and_x3_l1337_133720

theorem relationship_between_x_x2_and_x3 (x : ℝ) (h : -1 < x ∧ x < 0) :
  x ^ 3 < x ∧ x < x ^ 2 :=
by
  sorry

end relationship_between_x_x2_and_x3_l1337_133720


namespace cheese_wedge_volume_l1337_133724

theorem cheese_wedge_volume (r h : ℝ) (n : ℕ) (V : ℝ) (π : ℝ) 
: r = 8 → h = 10 → n = 3 → V = π * r^2 * h → V / n = (640 * π) / 3  :=
by
  intros r_eq h_eq n_eq V_eq
  rw [r_eq, h_eq] at V_eq
  rw [V_eq]
  sorry

end cheese_wedge_volume_l1337_133724


namespace simplify_A_minus_B_value_of_A_minus_B_given_condition_l1337_133775

variable (a b : ℝ)

def A := (a + b) ^ 2 - 3 * b ^ 2
def B := 2 * (a + b) * (a - b) - 3 * a * b

theorem simplify_A_minus_B :
  A a b - B a b = -a ^ 2 + 5 * a * b :=
by sorry

theorem value_of_A_minus_B_given_condition :
  (a - 3) ^ 2 + |b - 4| = 0 → A a b - B a b = 51 :=
by sorry

end simplify_A_minus_B_value_of_A_minus_B_given_condition_l1337_133775


namespace combine_like_terms_l1337_133739

theorem combine_like_terms : ∀ (x y : ℝ), -2 * x * y^2 + 2 * x * y^2 = 0 :=
by
  intros
  sorry

end combine_like_terms_l1337_133739


namespace cara_arrangements_l1337_133792

theorem cara_arrangements (n : ℕ) (h : n = 7) : ∃ k : ℕ, k = 6 :=
by
  sorry

end cara_arrangements_l1337_133792


namespace ax_by_n_sum_l1337_133748

theorem ax_by_n_sum {a b x y : ℝ} 
  (h1 : a * x + b * y = 2)
  (h2 : a * x^2 + b * y^2 = 5)
  (h3 : a * x^3 + b * y^3 = 15)
  (h4 : a * x^4 + b * y^4 = 35) :
  a * x^5 + b * y^5 = 10 :=
sorry

end ax_by_n_sum_l1337_133748


namespace closest_number_l1337_133790

theorem closest_number
  (a b c : ℝ)
  (h₀ : a = Real.sqrt 5)
  (h₁ : b = 3)
  (h₂ : b = (a + c) / 2) :
  abs (c - 3.5) ≤ abs (c - 2) ∧ abs (c - 3.5) ≤ abs (c - 2.5) ∧ abs (c - 3.5) ≤ abs (c - 3)  :=
by
  sorry

end closest_number_l1337_133790


namespace sum_of_roots_l1337_133770

theorem sum_of_roots (a b c : ℝ) (h : 6 * a ^ 3 - 7 * a ^ 2 + 2 * a = 0 ∧ 
                                   6 * b ^ 3 - 7 * b ^ 2 + 2 * b = 0 ∧ 
                                   6 * c ^ 3 - 7 * c ^ 2 + 2 * c = 0 ∧ 
                                   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    a + b + c = 7 / 6 :=
sorry

end sum_of_roots_l1337_133770


namespace geometric_seq_a3_equals_3_l1337_133798

variable {a : ℕ → ℝ}
variable (h_geometric : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)
variable (h_pos : ∀ n, n > 0 → a n > 0)
variable (h_cond : a 2 * a 4 = 9)

theorem geometric_seq_a3_equals_3 : a 3 = 3 := by
  sorry

end geometric_seq_a3_equals_3_l1337_133798


namespace intersection_of_sets_l1337_133782

-- Define sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem intersection_of_sets : A ∩ B = {1, 2} := by
  sorry

end intersection_of_sets_l1337_133782


namespace proof_problem_l1337_133796

def number := 432

theorem proof_problem (y : ℕ) (n : ℕ) (h1 : y = 36) (h2 : 6^5 * 2 / n = y) : n = number :=
by 
  -- proof steps would go here
  sorry

end proof_problem_l1337_133796


namespace solve_for_a_l1337_133718

theorem solve_for_a (a : ℝ) (h : ∃ x, x = 2 ∧ a * x - 4 * (x - a) = 1) : a = 3 / 2 :=
sorry

end solve_for_a_l1337_133718


namespace each_person_tip_l1337_133725

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l1337_133725


namespace socks_choice_count_l1337_133791

variable (white_socks : ℕ) (brown_socks : ℕ) (blue_socks : ℕ) (black_socks : ℕ)

theorem socks_choice_count :
  white_socks = 5 →
  brown_socks = 4 →
  blue_socks = 2 →
  black_socks = 2 →
  (white_socks.choose 2) + (brown_socks.choose 2) + (blue_socks.choose 2) + (black_socks.choose 2) = 18 :=
by
  -- Here the proof would be elaborated
  sorry

end socks_choice_count_l1337_133791


namespace Monica_books_read_l1337_133765

theorem Monica_books_read : 
  let books_last_year := 16 
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  books_next_year = 69 :=
by
  let books_last_year := 16
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  sorry

end Monica_books_read_l1337_133765


namespace percent_of_75_of_125_l1337_133704

theorem percent_of_75_of_125 : (75 / 125) * 100 = 60 := by
  sorry

end percent_of_75_of_125_l1337_133704


namespace base_8_add_sub_l1337_133776

-- Definitions of the numbers in base 8
def n1 : ℕ := 4 * 8^2 + 5 * 8^1 + 1 * 8^0
def n2 : ℕ := 1 * 8^2 + 6 * 8^1 + 2 * 8^0
def n3 : ℕ := 1 * 8^2 + 2 * 8^1 + 3 * 8^0

-- Convert the result to base 8
def to_base_8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let rem1 := n % 64
  let d1 := rem1 / 8
  let d0 := rem1 % 8
  d2 * 100 + d1 * 10 + d0

-- Proof statement
theorem base_8_add_sub :
  to_base_8 ((n1 + n2) - n3) = to_base_8 (5 * 8^2 + 1 * 8^1 + 0 * 8^0) :=
by
  sorry

end base_8_add_sub_l1337_133776


namespace smallest_y_for_perfect_fourth_power_l1337_133717

-- Define the conditions
def x : ℕ := 7 * 24 * 48
def y : ℕ := 6174

-- The theorem we need to prove
theorem smallest_y_for_perfect_fourth_power (x y : ℕ) 
  (hx : x = 7 * 24 * 48) 
  (hy : y = 6174) : ∃ k : ℕ, (∃ z : ℕ, z * z * z * z = x * y) :=
sorry

end smallest_y_for_perfect_fourth_power_l1337_133717


namespace length_of_train_l1337_133794

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l1337_133794


namespace inequality_solution_set_range_of_m_l1337_133734

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem inequality_solution_set :
  {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → m > 4 :=
sorry

end inequality_solution_set_range_of_m_l1337_133734


namespace candidate_a_votes_l1337_133733

theorem candidate_a_votes (x : ℕ) (h : 2 * x + x = 21) : 2 * x = 14 :=
by sorry

end candidate_a_votes_l1337_133733


namespace total_legs_l1337_133711

def total_heads : ℕ := 16
def num_cats : ℕ := 7
def cat_legs : ℕ := 4
def captain_legs : ℕ := 1
def human_legs : ℕ := 2

theorem total_legs : (num_cats * cat_legs + (total_heads - num_cats) * human_legs - human_legs + captain_legs) = 45 :=
by 
  -- Proof skipped
  sorry

end total_legs_l1337_133711


namespace smallest_class_size_l1337_133795

theorem smallest_class_size
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 2)
  (total_students : 5 * x + 2 > 40) :
  ∃ (n : ℕ), n = 5 * x + 2 ∧ n = 42 :=
by
  sorry

end smallest_class_size_l1337_133795


namespace product_of_possible_b_values_l1337_133755

theorem product_of_possible_b_values (b : ℝ) :
  (∀ (y1 y2 x1 x2 : ℝ), y1 = -1 ∧ y2 = 3 ∧ x1 = 2 ∧ (x2 = b) ∧ (y2 - y1 = 4) → 
   (b = 2 + 4 ∨ b = 2 - 4)) → 
  (b = 6 ∨ b = -2) → (b = 6) ∧ (b = -2) → 6 * -2 = -12 :=
sorry

end product_of_possible_b_values_l1337_133755


namespace find_slope_intercept_l1337_133706

def line_eqn (x y : ℝ) : Prop :=
  -3 * (x - 5) + 2 * (y + 1) = 0

theorem find_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_eqn x y → y = m * x + b) ∧ (m = 3/2) ∧ (b = -17/2) := sorry

end find_slope_intercept_l1337_133706


namespace minimum_words_to_learn_l1337_133722

-- Definition of the problem
def total_words : ℕ := 600
def required_percentage : ℕ := 90

-- Lean statement of the problem
theorem minimum_words_to_learn : ∃ x : ℕ, (x / total_words : ℚ) = required_percentage / 100 ∧ x = 540 :=
sorry

end minimum_words_to_learn_l1337_133722


namespace sqrt_sq_eq_l1337_133799

theorem sqrt_sq_eq (x : ℝ) : (Real.sqrt x) ^ 2 = x := by
  sorry

end sqrt_sq_eq_l1337_133799


namespace right_triangle_legs_l1337_133779

theorem right_triangle_legs (a b : ℝ) (r R : ℝ) (hypotenuse : ℝ) (h_ab : a + b = 14) (h_c : hypotenuse = 10)
  (h_leg: a * b = a + b + 10) (h_Pythag : a^2 + b^2 = hypotenuse^2) 
  (h_inradius : r = 2) (h_circumradius : R = 5) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
by
  sorry

end right_triangle_legs_l1337_133779


namespace sqrt_sum_inequality_l1337_133784

theorem sqrt_sum_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_sum : a + b + c = 3) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a) :=
by
  sorry

end sqrt_sum_inequality_l1337_133784


namespace boxes_with_neither_l1337_133732

def total_boxes : ℕ := 15
def boxes_with_crayons : ℕ := 9
def boxes_with_markers : ℕ := 6
def boxes_with_both : ℕ := 4

theorem boxes_with_neither : total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

end boxes_with_neither_l1337_133732


namespace polynomial_remainder_division_l1337_133745

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end polynomial_remainder_division_l1337_133745


namespace degree_of_vertex_angle_of_isosceles_triangle_l1337_133768

theorem degree_of_vertex_angle_of_isosceles_triangle (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 40) : 
∃ vertex_angle : ℝ, vertex_angle = 140 :=
by 
  sorry

end degree_of_vertex_angle_of_isosceles_triangle_l1337_133768


namespace algebra_expression_bound_l1337_133789

theorem algebra_expression_bound (x y m : ℝ) 
  (h1 : x + y + m = 6) 
  (h2 : 3 * x - y + m = 4) : 
  (-2 * x * y + 1) ≤ 3 / 2 := 
by 
  sorry

end algebra_expression_bound_l1337_133789


namespace bushes_for_60_zucchinis_l1337_133760

/-- 
Given:
1. Each blueberry bush yields twelve containers of blueberries.
2. Four containers of blueberries can be traded for three pumpkins.
3. Six pumpkins can be traded for five zucchinis.

Prove that eight bushes are needed to harvest 60 zucchinis.
-/
theorem bushes_for_60_zucchinis (bush_to_containers : ℕ) (containers_to_pumpkins : ℕ) (pumpkins_to_zucchinis : ℕ) :
  (bush_to_containers = 12) → (containers_to_pumpkins = 4) → (pumpkins_to_zucchinis = 6) →
  ∃ bushes_needed, bushes_needed = 8 ∧ (60 * pumpkins_to_zucchinis / 5 * containers_to_pumpkins / 3 / bush_to_containers) = bushes_needed :=
by
  intros h1 h2 h3
  sorry

end bushes_for_60_zucchinis_l1337_133760


namespace estimated_watched_students_l1337_133731

-- Definitions for the problem conditions
def total_students : ℕ := 3600
def surveyed_students : ℕ := 200
def watched_students : ℕ := 160

-- Problem statement (proof not included yet)
theorem estimated_watched_students :
  total_students * (watched_students / surveyed_students : ℝ) = 2880 := by
  -- skipping proof step
  sorry

end estimated_watched_students_l1337_133731


namespace coordinates_P_l1337_133700

theorem coordinates_P 
  (P1 P2 P : ℝ × ℝ)
  (hP1 : P1 = (2, -1))
  (hP2 : P2 = (0, 5))
  (h_ext_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)) ∧ t ≠ 1)
  (h_distance : dist P1 P = 2 * dist P P2) :
  P = (-2, 11) := 
by
  sorry

end coordinates_P_l1337_133700


namespace problem_l1337_133743

variable (a b c d : ℕ)

theorem problem (h1 : a + b = 12) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 6 :=
sorry

end problem_l1337_133743


namespace m_ge_1_l1337_133751

open Set

theorem m_ge_1 (m : ℝ) :
  (∀ x, x ∈ {x | x ≤ 1} ∩ {x | ¬ (x ≤ m)} → False) → m ≥ 1 :=
by
  intro h
  sorry

end m_ge_1_l1337_133751


namespace jamies_class_girls_count_l1337_133753

theorem jamies_class_girls_count 
  (g b : ℕ)
  (h_ratio : 4 * g = 3 * b)
  (h_total : g + b = 35) 
  : g = 15 := 
by 
  sorry 

end jamies_class_girls_count_l1337_133753


namespace largest_result_l1337_133756

theorem largest_result :
  let A := (1 / 17 - 1 / 19) / 20
  let B := (1 / 15 - 1 / 21) / 60
  let C := (1 / 13 - 1 / 23) / 100
  let D := (1 / 11 - 1 / 25) / 140
  D > A ∧ D > B ∧ D > C := by
  sorry

end largest_result_l1337_133756


namespace alcohol_concentration_l1337_133786

theorem alcohol_concentration (x : ℝ) (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.35 →
  target_concentration = 0.50 →
  (2.1 + x) / (6 + x) = target_concentration →
  x = 1.8 :=
by
  intros h1 h2 h3 h4
  sorry

end alcohol_concentration_l1337_133786


namespace total_tickets_sold_l1337_133712

def SeniorPrice : Nat := 10
def RegularPrice : Nat := 15
def TotalSales : Nat := 855
def RegularTicketsSold : Nat := 41

theorem total_tickets_sold : ∃ (S R : Nat), R = RegularTicketsSold ∧ 10 * S + 15 * R = TotalSales ∧ S + R = 65 :=
by
  sorry

end total_tickets_sold_l1337_133712


namespace equalize_vertex_values_impossible_l1337_133773

theorem equalize_vertex_values_impossible 
  (n : ℕ) (h₁ : 2 ≤ n) 
  (vertex_values : Fin n → ℤ) 
  (h₂ : ∃! i : Fin n, vertex_values i = 1 ∧ ∀ j ≠ i, vertex_values j = 0) 
  (k : ℕ) (hk : k ∣ n) :
  ¬ (∃ c : ℤ, ∀ v : Fin n, vertex_values v = c) := 
sorry

end equalize_vertex_values_impossible_l1337_133773


namespace amount_paid_l1337_133746

def cost_cat_toy : ℝ := 8.77
def cost_cage : ℝ := 10.97
def change_received : ℝ := 0.26

theorem amount_paid : (cost_cat_toy + cost_cage + change_received) = 20.00 := by
  sorry

end amount_paid_l1337_133746


namespace find_wrongly_noted_mark_l1337_133774

-- Definitions of given conditions
def average_marks := 100
def number_of_students := 25
def reported_correct_mark := 10
def correct_average_marks := 98
def wrongly_noted_mark : ℕ := sorry

-- Computing the sum with the wrong mark
def incorrect_sum := number_of_students * average_marks

-- Sum corrected by replacing wrong mark with correct mark
def sum_with_correct_replacement (wrongly_noted_mark : ℕ) := 
  incorrect_sum - wrongly_noted_mark + reported_correct_mark

-- Correct total sum for correct average
def correct_sum := number_of_students * correct_average_marks

-- The statement to be proven
theorem find_wrongly_noted_mark : wrongly_noted_mark = 60 :=
by sorry

end find_wrongly_noted_mark_l1337_133774
