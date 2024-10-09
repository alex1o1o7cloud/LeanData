import Mathlib

namespace eq_of_frac_sub_l508_50849

theorem eq_of_frac_sub (x : ℝ) (hx : x ≠ 1) : 
  (2 / (x^2 - 1) - 1 / (x - 1)) = - (1 / (x + 1)) := 
by sorry

end eq_of_frac_sub_l508_50849


namespace largest_of_five_consecutive_integers_l508_50899

   theorem largest_of_five_consecutive_integers (n1 n2 n3 n4 n5 : ℕ) 
     (h1: 0 < n1) (h2: n1 + 1 = n2) (h3: n2 + 1 = n3) (h4: n3 + 1 = n4)
     (h5: n4 + 1 = n5) (h6: n1 * n2 * n3 * n4 * n5 = 15120) : n5 = 10 :=
   sorry
   
end largest_of_five_consecutive_integers_l508_50899


namespace find_angle_B_find_max_k_l508_50805

theorem find_angle_B
(A B C a b c : ℝ)
(h_angles : A + B + C = Real.pi)
(h_sides : (2 * a - c) * Real.cos B = b * Real.cos C)
(h_A_pos : 0 < A) (h_B_pos : 0 < B) (h_C_pos : 0 < C) 
(h_Alt_pos : A < Real.pi) (h_Blt_pos : B < Real.pi) 
(h_Clt_pos : C < Real.pi) :
B = Real.pi / 3 := 
sorry

theorem find_max_k
(A : ℝ)
(k : ℝ)
(m : ℝ × ℝ := (Real.sin A, Real.cos (2 * A)))
(n : ℝ × ℝ := (4 * k, 1))
(h_k_cond : 1 < k)
(h_max_dot : (m.1) * (n.1) + (m.2) * (n.2) = 5) :
k = 3 / 2 :=
sorry

end find_angle_B_find_max_k_l508_50805


namespace layla_earnings_l508_50855

def rate_donaldsons : ℕ := 15
def bonus_donaldsons : ℕ := 5
def hours_donaldsons : ℕ := 7
def rate_merck : ℕ := 18
def discount_merck : ℝ := 0.10
def hours_merck : ℕ := 6
def rate_hille : ℕ := 20
def bonus_hille : ℕ := 10
def hours_hille : ℕ := 3
def rate_johnson : ℕ := 22
def flat_rate_johnson : ℕ := 80
def hours_johnson : ℕ := 4
def rate_ramos : ℕ := 25
def bonus_ramos : ℕ := 20
def hours_ramos : ℕ := 2

def donaldsons_earnings := rate_donaldsons * hours_donaldsons + bonus_donaldsons
def merck_earnings := rate_merck * hours_merck - (rate_merck * hours_merck * discount_merck : ℝ)
def hille_earnings := rate_hille * hours_hille + bonus_hille
def johnson_earnings := rate_johnson * hours_johnson
def ramos_earnings := rate_ramos * hours_ramos + bonus_ramos

noncomputable def total_earnings : ℝ :=
  donaldsons_earnings + merck_earnings + hille_earnings + johnson_earnings + ramos_earnings

theorem layla_earnings : total_earnings = 435.2 :=
by
  sorry

end layla_earnings_l508_50855


namespace find_x_when_y_neg_10_l508_50856

def inversely_proportional (x y : ℝ) (k : ℝ) := x * y = k

theorem find_x_when_y_neg_10 (k : ℝ) (h₁ : inversely_proportional 4 (-2) k) (yval : y = -10) 
: ∃ x, inversely_proportional x y k ∧ x = 4 / 5 := by
  sorry

end find_x_when_y_neg_10_l508_50856


namespace flour_quantity_l508_50887

-- Define the recipe ratio of eggs to flour
def recipe_ratio : ℚ := 3 / 2

-- Define the number of eggs needed
def eggs_needed := 9

-- Prove that the number of cups of flour needed is 6
theorem flour_quantity (r : ℚ) (n : ℕ) (F : ℕ) 
  (hr : r = 3 / 2) (hn : n = 9) : F = 6 :=
by
  sorry

end flour_quantity_l508_50887


namespace distinct_real_nums_condition_l508_50833

theorem distinct_real_nums_condition 
  (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 :=
by
  sorry

end distinct_real_nums_condition_l508_50833


namespace sequence_sum_after_6_steps_l508_50886

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 1435 -- would define how numbers sequence works recursively.
  else sorry -- next steps up to 6
  

theorem sequence_sum_after_6_steps : sequence_sum 6 = 191 := 
by
  sorry

end sequence_sum_after_6_steps_l508_50886


namespace brandon_textbooks_weight_l508_50839

-- Define the weights of Jon's textbooks
def weight_jon_book1 := 2
def weight_jon_book2 := 8
def weight_jon_book3 := 5
def weight_jon_book4 := 9

-- Calculate the total weight of Jon's textbooks
def total_weight_jon := weight_jon_book1 + weight_jon_book2 + weight_jon_book3 + weight_jon_book4

-- Define the condition where Jon's textbooks weigh three times as much as Brandon's textbooks
def jon_to_brandon_ratio := 3

-- Define the weight of Brandon's textbooks
def weight_brandon := total_weight_jon / jon_to_brandon_ratio

-- The goal is to prove that the weight of Brandon's textbooks is 8 pounds.
theorem brandon_textbooks_weight : weight_brandon = 8 := by
  sorry

end brandon_textbooks_weight_l508_50839


namespace sin_double_angle_identity_l508_50838

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end sin_double_angle_identity_l508_50838


namespace probability_queen_then_club_l508_50811

-- Define the problem conditions using the definitions
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_clubs : ℕ := 13
def num_club_queens : ℕ := 1

-- Define a function that computes the probability of the given event
def probability_first_queen_second_club : ℚ :=
  let prob_first_club_queen := (num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_club_queen := (num_clubs - 1 : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_1 := prob_first_club_queen * prob_second_club_given_first_club_queen
  let prob_first_non_club_queen := (num_queens - num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_non_club_queen := (num_clubs : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_2 := prob_first_non_club_queen * prob_second_club_given_first_non_club_queen
  prob_case_1 + prob_case_2

-- The statement to be proved
theorem probability_queen_then_club : probability_first_queen_second_club = 1 / 52 := by
  sorry

end probability_queen_then_club_l508_50811


namespace max_tetrahedron_in_cube_l508_50836

open Real

noncomputable def cube_edge_length : ℝ := 6
noncomputable def max_tetrahedron_edge_length (a : ℝ) : Prop :=
  ∃ x : ℝ, x = 2 * sqrt 6 ∧ 
          (∃ R : ℝ, R = (a * sqrt 3) / 2 ∧ x / sqrt (2 / 3) = 4 * R / 3)

theorem max_tetrahedron_in_cube : max_tetrahedron_edge_length cube_edge_length :=
sorry

end max_tetrahedron_in_cube_l508_50836


namespace outcome_transactions_l508_50817

-- Definition of initial property value and profit/loss percentages.
def property_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

-- Calculate selling price after 15% profit.
def selling_price : ℝ := property_value * (1 + profit_percentage)

-- Calculate buying price after 5% loss based on the above selling price.
def buying_price : ℝ := selling_price * (1 - loss_percentage)

-- Calculate the net gain/loss.
def net_gain_or_loss : ℝ := selling_price - buying_price

-- Statement to be proved.
theorem outcome_transactions : net_gain_or_loss = 862.5 := by
  sorry

end outcome_transactions_l508_50817


namespace find_constants_l508_50865

theorem find_constants (P Q R : ℚ) (h : ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) = (P / (x - 1)) + (Q / (x - 4)) + (R / (x - 6))) : 
  (P, Q, R) = (-4/5, -1/2, 23/10) := 
  sorry

end find_constants_l508_50865


namespace min_value_frac_sum_l508_50888

theorem min_value_frac_sum (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  (∃ m, ∀ x y, m = 1 ∧ (
      (1 / (x + y)^2) + (1 / (x - y)^2) ≥ m)) :=
sorry

end min_value_frac_sum_l508_50888


namespace find_angle_A_l508_50812

theorem find_angle_A (BC AC : ℝ) (B : ℝ) (A : ℝ) (h_cond : BC = Real.sqrt 3 ∧ AC = 1 ∧ B = Real.pi / 6) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l508_50812


namespace sphere_diameter_l508_50809

theorem sphere_diameter (r : ℝ) (V : ℝ) (threeV : ℝ) (a b : ℕ) :
  (∀ (r : ℝ), r = 5 →
  V = (4 / 3) * π * r^3 →
  threeV = 3 * V →
  D = 2 * (3 * V * 3 / (4 * π))^(1 / 3) →
  D = a * b^(1 / 3) →
  a = 10 ∧ b = 3) →
  a + b = 13 :=
by
  intros
  sorry

end sphere_diameter_l508_50809


namespace how_many_whole_boxes_did_nathan_eat_l508_50852

-- Define the conditions
def gumballs_per_package := 5
def total_gumballs := 20

-- The problem to prove
theorem how_many_whole_boxes_did_nathan_eat : total_gumballs / gumballs_per_package = 4 :=
by sorry

end how_many_whole_boxes_did_nathan_eat_l508_50852


namespace problem_a_problem_b_unique_solution_l508_50893

-- Problem (a)

theorem problem_a (a b c n : ℤ) (hnat : 0 ≤ n) (h : a * n^2 + b * n + c = 0) : n ∣ c :=
sorry

-- Problem (b)

theorem problem_b_unique_solution : ∀ n : ℕ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = 3 :=
sorry

end problem_a_problem_b_unique_solution_l508_50893


namespace original_denominator_l508_50832

theorem original_denominator (d : ℕ) (h : 11 = 3 * (d + 8)) : d = 25 :=
by
  sorry

end original_denominator_l508_50832


namespace increase_in_length_and_breadth_is_4_l508_50807

-- Define the variables for the original length and breadth of the room
variables (L B x : ℕ)

-- Define the original perimeter
def P_original : ℕ := 2 * (L + B)

-- Define the new perimeter after the increase
def P_new : ℕ := 2 * ((L + x) + (B + x))

-- Define the condition that the perimeter increases by 16 feet
axiom increase_perimeter : P_new L B x - P_original L B = 16

-- State the theorem that \(x = 4\)
theorem increase_in_length_and_breadth_is_4 : x = 4 :=
by
  -- Proof would be filled in here using the axioms and definitions
  sorry

end increase_in_length_and_breadth_is_4_l508_50807


namespace smallest_integer_to_make_perfect_square_l508_50837

-- Define the number y as specified
def y : ℕ := 2^5 * 3^6 * (2^2)^7 * 5^8 * (2 * 3)^9 * 7^10 * (2^3)^11 * (3^2)^12

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The goal statement
theorem smallest_integer_to_make_perfect_square : 
  ∃ z : ℕ, z > 0 ∧ is_perfect_square (y * z) ∧ ∀ w : ℕ, w > 0 → is_perfect_square (y * w) → z ≤ w := by
  sorry

end smallest_integer_to_make_perfect_square_l508_50837


namespace simplify_expression_l508_50891

theorem simplify_expression (b : ℝ) (h : b ≠ 1 / 2) : 1 - (2 / (1 + (b / (1 - 2 * b)))) = (3 * b - 1) / (1 - b) :=
by
    sorry

end simplify_expression_l508_50891


namespace order_of_abc_l508_50867

theorem order_of_abc (a b c : ℝ) (h1 : a = 16 ^ (1 / 3))
                                 (h2 : b = 2 ^ (4 / 5))
                                 (h3 : c = 5 ^ (2 / 3)) :
  c > a ∧ a > b :=
by {
  sorry
}

end order_of_abc_l508_50867


namespace vasya_mushrooms_l508_50823

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end vasya_mushrooms_l508_50823


namespace fishes_per_body_of_water_l508_50831

-- Define the number of bodies of water
def n_b : Nat := 6

-- Define the total number of fishes
def n_f : Nat := 1050

-- Prove the number of fishes per body of water
theorem fishes_per_body_of_water : n_f / n_b = 175 := by 
  sorry

end fishes_per_body_of_water_l508_50831


namespace remainder_divisibility_l508_50835

theorem remainder_divisibility (n : ℕ) (d : ℕ) (r : ℕ) : 
  let n := 1234567
  let d := 256
  let r := n % d
  r = 933 ∧ ¬ (r % 7 = 0) := by
  sorry

end remainder_divisibility_l508_50835


namespace problem1_problem2_problem3_l508_50847

-- (1) Prove 1 - 2(x - y) + (x - y)^2 = (1 - x + y)^2
theorem problem1 (x y : ℝ) : 1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
sorry

-- (2) Prove 25(a - 1)^2 - 10(a - 1) + 1 = (5a - 6)^2
theorem problem2 (a : ℝ) : 25 * (a - 1)^2 - 10 * (a - 1) + 1 = (5 * a - 6)^2 :=
sorry

-- (3) Prove (y^2 - 4y)(y^2 - 4y + 8) + 16 = (y - 2)^4
theorem problem3 (y : ℝ) : (y^2 - 4 * y) * (y^2 - 4 * y + 8) + 16 = (y - 2)^4 :=
sorry

end problem1_problem2_problem3_l508_50847


namespace inequality_proof_l508_50824

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  ¬ (1 / (1 + x + x * y) > 1 / 3 ∧ 
     y / (1 + y + y * z) > 1 / 3 ∧
     (x * z) / (1 + z + x * z) > 1 / 3) :=
by
  sorry

end inequality_proof_l508_50824


namespace abc_inequalities_l508_50897

noncomputable def a : ℝ := Real.log 1 / Real.log 2 - Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 2) ^ 3
noncomputable def c : ℝ := Real.sqrt 3

theorem abc_inequalities :
  a < b ∧ b < c :=
by
  -- Proof omitted
  sorry

end abc_inequalities_l508_50897


namespace percentage_difference_l508_50883

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end percentage_difference_l508_50883


namespace abs_a_lt_abs_b_sub_abs_c_l508_50825

theorem abs_a_lt_abs_b_sub_abs_c (a b c : ℝ) (h : |a + c| < b) : |a| < |b| - |c| :=
sorry

end abs_a_lt_abs_b_sub_abs_c_l508_50825


namespace subtract_real_numbers_l508_50845

theorem subtract_real_numbers : 3.56 - 1.89 = 1.67 :=
by
  sorry

end subtract_real_numbers_l508_50845


namespace exponential_inequality_l508_50866

theorem exponential_inequality (k l m : ℕ) : 2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 :=
by
  sorry

end exponential_inequality_l508_50866


namespace correct_operations_l508_50801

variable (x : ℚ)

def incorrect_equation := ((x - 5) * 3) / 7 = 10

theorem correct_operations :
  incorrect_equation x → (3 * x - 5) / 7 = 80 / 7 :=
by
  intro h
  sorry

end correct_operations_l508_50801


namespace find_three_digit_number_divisible_by_5_l508_50827

theorem find_three_digit_number_divisible_by_5 {n x : ℕ} (hx1 : 100 ≤ x) (hx2 : x < 1000) (hx3 : x % 5 = 0) (hx4 : x = n^3 + n^2) : x = 150 ∨ x = 810 := 
by
  sorry

end find_three_digit_number_divisible_by_5_l508_50827


namespace alloy_cut_weight_l508_50848

variable (a b x : ℝ)
variable (ha : 0 ≤ a ∧ a ≤ 1) -- assuming copper content is a fraction between 0 and 1
variable (hb : 0 ≤ b ∧ b ≤ 1)
variable (h : a ≠ b)
variable (hx : 0 < x ∧ x < 40) -- x is strictly between 0 and 40 (since 0 ≤ x ≤ 40)

theorem alloy_cut_weight (A B : ℝ) (hA : A = 40) (hB : B = 60) (h1 : (a * x + b * (A - x)) / 40 = (b * x + a * (B - x)) / 60) : x = 24 :=
by
  sorry

end alloy_cut_weight_l508_50848


namespace inequal_f_i_sum_mn_ii_l508_50861

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 3 / 2 then -2 
  else if x > -5 / 2 then -x - 1 / 2 
  else 2

theorem inequal_f_i (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
sorry

theorem sum_mn_ii (m n : ℝ) (h1 : f m + f n = 4) (h2 : m < n) : m + n < -5 :=
sorry

end inequal_f_i_sum_mn_ii_l508_50861


namespace Debby_drinks_five_bottles_per_day_l508_50814

theorem Debby_drinks_five_bottles_per_day (total_bottles : ℕ) (days : ℕ) (h1 : total_bottles = 355) (h2 : days = 71) : (total_bottles / days) = 5 :=
by 
  sorry

end Debby_drinks_five_bottles_per_day_l508_50814


namespace henry_initial_money_l508_50800

variable (x : ℤ)

theorem henry_initial_money : (x + 18 - 10 = 19) → x = 11 :=
by
  intro h
  sorry

end henry_initial_money_l508_50800


namespace binary_ternary_product_base_10_l508_50877

theorem binary_ternary_product_base_10 :
  let b2 := 2
  let t3 := 3
  let n1 := 1011 -- binary representation
  let n2 := 122 -- ternary representation
  let a1 := (1 * b2^3) + (0 * b2^2) + (1 * b2^1) + (1 * b2^0)
  let a2 := (1 * t3^2) + (2 * t3^1) + (2 * t3^0)
  a1 * a2 = 187 :=
by
  sorry

end binary_ternary_product_base_10_l508_50877


namespace no_distinct_positive_integers_2007_l508_50878

theorem no_distinct_positive_integers_2007 (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  ¬ (x^2007 + y! = y^2007 + x!) :=
by
  sorry

end no_distinct_positive_integers_2007_l508_50878


namespace find_number_l508_50830

def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_digit_is_three (n : ℕ) : Prop :=
  n / 1000 = 3

def last_digit_is_five (n : ℕ) : Prop :=
  n % 10 = 5

theorem find_number :
  ∃ (x : ℕ), four_digit_number (x^2) ∧ first_digit_is_three (x^2) ∧ last_digit_is_five (x^2) ∧ x = 55 :=
sorry

end find_number_l508_50830


namespace sin_arith_seq_l508_50851

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end sin_arith_seq_l508_50851


namespace rebate_percentage_l508_50804

theorem rebate_percentage (r : ℝ) (h1 : 0 ≤ r) (h2 : r ≤ 1) 
(h3 : (6650 - 6650 * r) * 1.10 = 6876.1) : r = 0.06 :=
sorry

end rebate_percentage_l508_50804


namespace circumscribed_quadrilateral_arc_sum_l508_50876

theorem circumscribed_quadrilateral_arc_sum 
  (a b c d : ℝ) 
  (h : a + b + c + d = 360) : 
  (1/2 * (b + c + d)) + (1/2 * (a + c + d)) + (1/2 * (a + b + d)) + (1/2 * (a + b + c)) = 540 :=
by
  sorry

end circumscribed_quadrilateral_arc_sum_l508_50876


namespace cars_between_15000_and_20000_l508_50895

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l508_50895


namespace find_k_plus_a_l508_50826

theorem find_k_plus_a (k a : ℤ) (h1 : k > a) (h2 : a > 0) 
(h3 : 2 * (Int.natAbs (a - k)) * (Int.natAbs (a + k)) = 32) : k + a = 8 :=
by
  sorry

end find_k_plus_a_l508_50826


namespace sin_cos_pi_over_12_l508_50862

theorem sin_cos_pi_over_12 :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
sorry

end sin_cos_pi_over_12_l508_50862


namespace num_correct_statements_l508_50872

def doubleAbsDiff (a b c d : ℝ) : ℝ :=
  |a - b| - |c - d|

theorem num_correct_statements : 
  (∀ a b c d : ℝ, (a, b, c, d) = (24, 25, 29, 30) → 
    (doubleAbsDiff a b c d = 0) ∨
    (doubleAbsDiff a c b d = 0) ∨
    (doubleAbsDiff a d b c = -0.5) ∨
    (doubleAbsDiff b c a d = 0.5)) → 
  (∀ x : ℝ, x ≥ 2 → 
    doubleAbsDiff (x^2) (2*x) 1 1 = 7 → 
    (x^4 + 2401 / x^4 = 226)) →
  (∀ x : ℝ, x ≥ -2 → 
    (doubleAbsDiff (2*x-5) (3*x-2) (4*x-1) (5*x+3)) ≠ 0) →
  (0 = 0)
:= by
  sorry

end num_correct_statements_l508_50872


namespace max_minus_min_eq_32_l508_50892

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_minus_min_eq_32 : 
  let M := max (f (-3)) (max (f 3) (max (f (-2)) (f 2)))
  let m := min (f (-3)) (min (f 3) (min (f (-2)) (f 2)))
  M - m = 32 :=
by
  sorry

end max_minus_min_eq_32_l508_50892


namespace total_pieces_ten_row_triangle_l508_50857

-- Definitions based on the conditions
def rods (n : ℕ) : ℕ :=
  (n * (2 * 4 + (n - 1) * 5)) / 2

def connectors (n : ℕ) : ℕ :=
  ((n + 1) * (2 * 1 + n * 1)) / 2

def support_sticks (n : ℕ) : ℕ := 
  if n >= 3 then ((n - 2) * (2 * 2 + (n - 3) * 2)) / 2 else 0

-- The theorem stating the total number of pieces is 395 for a ten-row triangle
theorem total_pieces_ten_row_triangle : rods 10 + connectors 10 + support_sticks 10 = 395 :=
by
  sorry

end total_pieces_ten_row_triangle_l508_50857


namespace martian_right_angle_l508_50882

theorem martian_right_angle :
  ∀ (full_circle clerts_per_right_angle : ℕ),
  (full_circle = 600) →
  (clerts_per_right_angle = full_circle / 3) →
  clerts_per_right_angle = 200 :=
by
  intros full_circle clerts_per_right_angle h1 h2
  sorry

end martian_right_angle_l508_50882


namespace total_number_of_coins_is_336_l508_50879

theorem total_number_of_coins_is_336 (N20 : ℕ) (N25 : ℕ) (total_value_rupees : ℚ)
    (h1 : N20 = 260) (h2 : total_value_rupees = 71) (h3 : 20 * N20 + 25 * N25 = 7100) :
    N20 + N25 = 336 :=
by
  sorry

end total_number_of_coins_is_336_l508_50879


namespace product_of_three_numbers_l508_50859

theorem product_of_three_numbers (p q r m : ℝ) (h1 : p + q + r = 180) (h2 : m = 8 * p)
  (h3 : m = q - 10) (h4 : m = r + 10) : p * q * r = 90000 := by
  sorry

end product_of_three_numbers_l508_50859


namespace mean_age_of_children_l508_50819

theorem mean_age_of_children :
  let ages := [8, 8, 12, 12, 10, 14]
  let n := ages.length
  let sum_ages := ages.foldr (· + ·) 0
  let mean_age := sum_ages / n
  mean_age = 10 + 2 / 3 :=
by
  sorry

end mean_age_of_children_l508_50819


namespace darnel_lap_difference_l508_50894

theorem darnel_lap_difference (sprint jog : ℝ) (h_sprint : sprint = 0.88) (h_jog : jog = 0.75) : sprint - jog = 0.13 := 
by 
  rw [h_sprint, h_jog] 
  norm_num

end darnel_lap_difference_l508_50894


namespace find_triangle_sides_l508_50806

theorem find_triangle_sides (a : Fin 7 → ℝ) (h : ∀ i, 1 < a i ∧ a i < 13) : 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 7 ∧ 
           a i + a j > a k ∧ 
           a j + a k > a i ∧ 
           a k + a i > a j :=
sorry

end find_triangle_sides_l508_50806


namespace kids_on_soccer_field_l508_50820

def original_kids : ℕ := 14
def joined_kids : ℕ := 22
def total_kids : ℕ := 36

theorem kids_on_soccer_field : (original_kids + joined_kids) = total_kids :=
by 
  sorry

end kids_on_soccer_field_l508_50820


namespace g_value_at_5_l508_50869

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_5 (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x ^ 2) : g 5 = 1 := 
by 
  sorry

end g_value_at_5_l508_50869


namespace fraction_of_seniors_study_japanese_l508_50889

variable (J S : ℝ)
variable (fraction_seniors fraction_juniors : ℝ)
variable (total_fraction_study_japanese : ℝ)

theorem fraction_of_seniors_study_japanese 
  (h1 : S = 2 * J)
  (h2 : fraction_juniors = 3 / 4)
  (h3 : total_fraction_study_japanese = 1 / 3) :
  fraction_seniors = 1 / 8 :=
by
  -- Here goes the proof.
  sorry

end fraction_of_seniors_study_japanese_l508_50889


namespace not_always_possible_triangle_sides_l508_50871

theorem not_always_possible_triangle_sides (α β γ δ : ℝ) 
  (h1 : α + β + γ + δ = 360) 
  (h2 : α < 180) 
  (h3 : β < 180) 
  (h4 : γ < 180) 
  (h5 : δ < 180) : 
  ¬ (∀ (x y z : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) ∧ (y = α ∨ y = β ∨ y = γ ∨ y = δ) ∧ (z = α ∨ z = β ∨ z = γ ∨ z = δ) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) → x + y > z ∧ x + z > y ∧ y + z > x)
:= sorry

end not_always_possible_triangle_sides_l508_50871


namespace ratio_of_x_to_y_l508_50834

theorem ratio_of_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l508_50834


namespace problem1_simplification_problem2_solve_fraction_l508_50816

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l508_50816


namespace not_divisible_2310_l508_50863

theorem not_divisible_2310 (n : ℕ) (h : n < 2310) : ¬ (2310 ∣ n * (2310 - n)) :=
sorry

end not_divisible_2310_l508_50863


namespace tape_needed_for_large_box_l508_50850

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box_l508_50850


namespace extremum_at_x1_l508_50815

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_x1 (a b : ℝ) (h1 : (3*1^2 + 2*a*1 + b) = 0) (h2 : 1^3 + a*1^2 + b*1 + a^2 = 10) :
  a = 4 :=
by
  sorry

end extremum_at_x1_l508_50815


namespace OH_over_ON_eq_2_no_other_common_points_l508_50828

noncomputable def coordinates (t p : ℝ) : ℝ × ℝ :=
  (t^2 / (2 * p), t)

noncomputable def symmetric_point (M P : ℝ × ℝ) : ℝ × ℝ :=
  let (xM, yM) := M;
  let (xP, yP) := P;
  (2 * xP - xM, 2 * yP - yM)

noncomputable def line_ON (p t : ℝ) : ℝ → ℝ :=
  λ x => (p / t) * x

noncomputable def line_MH (t p : ℝ) : ℝ → ℝ :=
  λ x => (p / (2 * t)) * x + t

noncomputable def point_H (t p : ℝ) : ℝ × ℝ :=
  (2 * t^2 / p, 2 * t)

theorem OH_over_ON_eq_2
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  (H.snd) / (N.snd) = 2 := by
  sorry

theorem no_other_common_points
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  ∀ y, (y ≠ H.snd → ¬ ∃ x, line_MH t p x = y ∧ y^2 = 2 * p * x) := by 
  sorry

end OH_over_ON_eq_2_no_other_common_points_l508_50828


namespace bears_in_shipment_l508_50860

theorem bears_in_shipment (initial_bears shipped_bears bears_per_shelf shelves_used : ℕ) 
  (h1 : initial_bears = 4) 
  (h2 : bears_per_shelf = 7) 
  (h3 : shelves_used = 2) 
  (total_bears_on_shelves : ℕ) 
  (h4 : total_bears_on_shelves = shelves_used * bears_per_shelf) 
  (total_bears_after_shipment : ℕ) 
  (h5 : total_bears_after_shipment = total_bears_on_shelves) 
  : shipped_bears = total_bears_on_shelves - initial_bears := 
sorry

end bears_in_shipment_l508_50860


namespace centroid_y_sum_zero_l508_50880

theorem centroid_y_sum_zero
  (x1 x2 x3 y2 y3 : ℝ)
  (h : y2 + y3 = 0) :
  (x1 + x2 + x3) / 3 = (x1 / 3 + x2 / 3 + x3 / 3) ∧ (y2 + y3) / 3 = 0 :=
by
  sorry

end centroid_y_sum_zero_l508_50880


namespace sandcastle_height_difference_l508_50854

theorem sandcastle_height_difference :
  let Miki_height := 0.8333333333333334
  let Sister_height := 0.5
  Miki_height - Sister_height = 0.3333333333333334 :=
by
  sorry

end sandcastle_height_difference_l508_50854


namespace sequence_general_term_l508_50896

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  ∃ a : ℕ → ℚ, (∀ n, a n = 1 / n) :=
by
  sorry

end sequence_general_term_l508_50896


namespace greatest_possible_gcd_l508_50873

theorem greatest_possible_gcd (d : ℕ) (a : ℕ → ℕ) (h_sum : (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 595)
  (h_gcd : ∀ i, d ∣ a i) : d ≤ 35 :=
sorry

end greatest_possible_gcd_l508_50873


namespace final_expression_in_simplest_form_l508_50898

variable (x : ℝ)

theorem final_expression_in_simplest_form : 
  ((3 * x + 6 - 5 * x + 10) / 5) = (-2 / 5) * x + 16 / 5 :=
by
  sorry

end final_expression_in_simplest_form_l508_50898


namespace speed_of_other_train_l508_50803

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l508_50803


namespace problem_l508_50840

def seq (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem problem (a : ℕ → ℤ) (h₁ : a 1 = 2010) (h₂ : a 2 = 2011) (h₃ : seq a) : a 1000 = 2343 :=
sorry

end problem_l508_50840


namespace find_angle_y_l508_50858

theorem find_angle_y (angle_ABC angle_ABD angle_ADB y : ℝ)
  (h1 : angle_ABC = 115)
  (h2 : angle_ABD = 180 - angle_ABC)
  (h3 : angle_ADB = 30)
  (h4 : angle_ABD + angle_ADB + y = 180) :
  y = 85 := 
sorry

end find_angle_y_l508_50858


namespace sample_size_l508_50841

theorem sample_size 
  (n_A n_B n_C : ℕ)
  (h1 : n_A = 15)
  (h2 : 3 * n_B = 4 * n_A)
  (h3 : 3 * n_C = 7 * n_A) :
  n_A + n_B + n_C = 70 :=
by
sorry

end sample_size_l508_50841


namespace Cubs_home_runs_third_inning_l508_50875

variable (X : ℕ)

theorem Cubs_home_runs_third_inning 
  (h : X + 1 + 2 = 2 + 3) : 
  X = 2 :=
by 
  sorry

end Cubs_home_runs_third_inning_l508_50875


namespace total_balloons_l508_50890

theorem total_balloons (fred_balloons : ℕ) (sam_balloons : ℕ) (mary_balloons : ℕ) :
  fred_balloons = 5 → sam_balloons = 6 → mary_balloons = 7 → fred_balloons + sam_balloons + mary_balloons = 18 :=
by
  intros
  sorry

end total_balloons_l508_50890


namespace percentage_of_seniors_is_90_l508_50884

-- Definitions of the given conditions
def total_students : ℕ := 120
def students_in_statistics : ℕ := total_students / 2
def seniors_in_statistics : ℕ := 54

-- Statement to prove
theorem percentage_of_seniors_is_90 : 
  ( seniors_in_statistics / students_in_statistics : ℚ ) * 100 = 90 := 
by
  sorry  -- Proof will be provided here.

end percentage_of_seniors_is_90_l508_50884


namespace mod_21_solution_l508_50853

theorem mod_21_solution (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n < 21) (h₂ : 47635 ≡ n [MOD 21]) : n = 19 :=
by
  sorry

end mod_21_solution_l508_50853


namespace time_to_fill_bucket_l508_50818

theorem time_to_fill_bucket (t : ℝ) (h : 2/3 = 2 / t) : t = 3 :=
by
  sorry

end time_to_fill_bucket_l508_50818


namespace students_left_is_6_l508_50802

-- Start of the year students
def initial_students : ℕ := 11

-- New students arrived during the year
def new_students : ℕ := 42

-- Students at the end of the year
def final_students : ℕ := 47

-- Definition to calculate the number of students who left
def students_left (initial new final : ℕ) : ℕ := (initial + new) - final

-- Statement to prove
theorem students_left_is_6 : students_left initial_students new_students final_students = 6 :=
by
  -- We skip the proof using sorry
  sorry

end students_left_is_6_l508_50802


namespace no_intersection_with_x_axis_l508_50881

open Real

theorem no_intersection_with_x_axis (m : ℝ) :
  (∀ x : ℝ, 3 ^ (-(|x - 1|)) + m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) :=
by
  sorry

end no_intersection_with_x_axis_l508_50881


namespace odd_function_a_eq_minus_1_l508_50864

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

theorem odd_function_a_eq_minus_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = -1 :=
by
  intros h
  sorry

end odd_function_a_eq_minus_1_l508_50864


namespace fill_in_the_blank_with_flowchart_l508_50821

def methods_to_describe_algorithm := ["Natural language", "Flowchart", "Pseudocode"]

theorem fill_in_the_blank_with_flowchart : 
  methods_to_describe_algorithm[1] = "Flowchart" :=
sorry

end fill_in_the_blank_with_flowchart_l508_50821


namespace train_travel_time_l508_50846

def travel_time (departure arrival : Nat) : Nat :=
  arrival - departure

theorem train_travel_time : travel_time 425 479 = 54 := by
  sorry

end train_travel_time_l508_50846


namespace trig_log_exp_identity_l508_50868

theorem trig_log_exp_identity : 
  (Real.sin (330 * Real.pi / 180) + 
   (Real.sqrt 2 - 1)^0 + 
   3^(Real.log 2 / Real.log 3)) = 5 / 2 :=
by
  -- Proof omitted
  sorry

end trig_log_exp_identity_l508_50868


namespace triangle_is_either_isosceles_or_right_angled_l508_50870

theorem triangle_is_either_isosceles_or_right_angled
  (A B : Real)
  (a b c : Real)
  (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  : a = b ∨ a^2 + b^2 = c^2 :=
sorry

end triangle_is_either_isosceles_or_right_angled_l508_50870


namespace max_plus_shapes_l508_50813

def cover_square (x y : ℕ) : Prop :=
  3 * x + 5 * y = 49

theorem max_plus_shapes (x y : ℕ) (h1 : cover_square x y) (h2 : x ≥ 4) : y ≤ 5 :=
sorry

end max_plus_shapes_l508_50813


namespace sqrt_squared_l508_50843

theorem sqrt_squared (n : ℕ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

example : (Real.sqrt 987654) ^ 2 = 987654 := 
  sqrt_squared 987654 (by norm_num)

end sqrt_squared_l508_50843


namespace relative_positions_of_P_on_AB_l508_50808

theorem relative_positions_of_P_on_AB (A B P : ℝ) : 
  A ≤ B → (A ≤ P ∧ P ≤ B ∨ P = A ∨ P = B ∨ P < A ∨ P > B) :=
by
  intro hAB
  sorry

end relative_positions_of_P_on_AB_l508_50808


namespace find_x_value_l508_50842

theorem find_x_value (x : ℤ)
    (h1 : (5 + 9) / 2 = 7)
    (h2 : (5 + x) / 2 = 10)
    (h3 : (x + 9) / 2 = 12) : 
    x = 15 := 
sorry

end find_x_value_l508_50842


namespace weight_of_b_l508_50810

-- Define the weights of a, b, and c
variables (W_a W_b W_c : ℝ)

-- Define the heights of a, b, and c
variables (h_a h_b h_c : ℝ)

-- Given conditions
axiom average_weight_abc : (W_a + W_b + W_c) / 3 = 45
axiom average_weight_ab : (W_a + W_b) / 2 = 40
axiom average_weight_bc : (W_b + W_c) / 2 = 47
axiom height_condition : h_a + h_c = 2 * h_b
axiom odd_sum_weights : (W_a + W_b + W_c) % 2 = 1

-- Prove that the weight of b is 39 kg
theorem weight_of_b : W_b = 39 :=
by sorry

end weight_of_b_l508_50810


namespace equilateral_triangle_vertex_distance_l508_50822

noncomputable def distance_vertex_to_center (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + (l^2 / 4))

theorem equilateral_triangle_vertex_distance
  (l r : ℝ)
  (h1 : l > 0)
  (h2 : r > 0) :
  distance_vertex_to_center l r = Real.sqrt (r^2 + (l^2 / 4)) :=
sorry

end equilateral_triangle_vertex_distance_l508_50822


namespace amount_paid_after_discount_l508_50874

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l508_50874


namespace methane_combined_l508_50829

def balancedEquation (CH₄ O₂ CO₂ H₂O : ℕ) : Prop :=
  CH₄ = 1 ∧ O₂ = 2 ∧ CO₂ = 1 ∧ H₂O = 2

theorem methane_combined {moles_CH₄ moles_O₂ moles_H₂O : ℕ}
  (h₁ : moles_O₂ = 2)
  (h₂ : moles_H₂O = 2)
  (h_eq : balancedEquation moles_CH₄ moles_O₂ 1 moles_H₂O) : 
  moles_CH₄ = 1 :=
by
  sorry

end methane_combined_l508_50829


namespace distance_from_hotel_l508_50885

def total_distance := 600
def speed1 := 50
def time1 := 3
def speed2 := 80
def time2 := 4

theorem distance_from_hotel :
  total_distance - (speed1 * time1 + speed2 * time2) = 130 := 
by
  sorry

end distance_from_hotel_l508_50885


namespace percentage_failed_in_english_l508_50844

theorem percentage_failed_in_english
  (H_perc : ℝ) (B_perc : ℝ) (Passed_in_English_alone : ℝ) (Total_candidates : ℝ)
  (H_perc_eq : H_perc = 36)
  (B_perc_eq : B_perc = 15)
  (Passed_in_English_alone_eq : Passed_in_English_alone = 630)
  (Total_candidates_eq : Total_candidates = 3000) :
  ∃ E_perc : ℝ, E_perc = 85 := by
  sorry

end percentage_failed_in_english_l508_50844
