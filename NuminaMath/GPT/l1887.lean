import Mathlib

namespace range_of_m_l1887_188789

theorem range_of_m (x m : ℝ) (h1 : x + 3 = 3 * x - m) (h2 : x ≥ 0) : m ≥ -3 := by
  sorry

end range_of_m_l1887_188789


namespace find_k_l1887_188729

def f (n : ℤ) : ℤ :=
if n % 2 = 0 then n / 2 else n + 3

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_f_f_f_k : f (f (f k)) = 27) : k = 105 := by
  sorry

end find_k_l1887_188729


namespace no_perfect_power_l1887_188714

theorem no_perfect_power (n m : ℕ) (hn : 0 < n) (hm : 1 < m) : 102 ^ 1991 + 103 ^ 1991 ≠ n ^ m := 
sorry

end no_perfect_power_l1887_188714


namespace complementary_angle_ratio_l1887_188715

noncomputable def smaller_angle_measure (x : ℝ) : ℝ := 
  3 * (90 / 7)

theorem complementary_angle_ratio :
  ∀ (A B : ℝ), (B = 4 * (90 / 7)) → (A = 3 * (90 / 7)) → 
  (A + B = 90) → A = 38.57142857142857 :=
by
  intros A B hB hA hSum
  sorry

end complementary_angle_ratio_l1887_188715


namespace nathan_ate_total_gumballs_l1887_188721

-- Define the constants and variables based on the conditions
def gumballs_small : Nat := 5
def gumballs_medium : Nat := 12
def gumballs_large : Nat := 20
def small_packages : Nat := 4
def medium_packages : Nat := 3
def large_packages : Nat := 2

-- The total number of gumballs Nathan ate
def total_gumballs : Nat := (small_packages * gumballs_small) + (medium_packages * gumballs_medium) + (large_packages * gumballs_large)

-- The theorem to prove
theorem nathan_ate_total_gumballs : total_gumballs = 96 :=
by
  unfold total_gumballs
  sorry

end nathan_ate_total_gumballs_l1887_188721


namespace total_wood_gathered_l1887_188756

def pieces_per_sack := 20
def number_of_sacks := 4

theorem total_wood_gathered : pieces_per_sack * number_of_sacks = 80 := 
by 
  sorry

end total_wood_gathered_l1887_188756


namespace circle_ellipse_intersect_four_points_l1887_188794

theorem circle_ellipse_intersect_four_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = a^2 → y = x^2 / 2 - a) →
  a > 1 :=
by
  sorry

end circle_ellipse_intersect_four_points_l1887_188794


namespace jorge_acres_l1887_188744

theorem jorge_acres (A : ℕ) (H1 : A = 60) 
    (H2 : ∀ acres, acres / 3 = 60 / 3 ∧ 2 * (acres / 3) = 2 * (60 / 3)) 
    (H3 : ∀ good_yield_per_acre, good_yield_per_acre = 400) 
    (H4 : ∀ clay_yield_per_acre, clay_yield_per_acre = 200) 
    (H5 : ∀ total_yield, total_yield = (2 * (A / 3) * 400 + (A / 3) * 200)) 
    : total_yield = 20000 :=
by 
  sorry

end jorge_acres_l1887_188744


namespace arithmetic_progression_product_l1887_188736

theorem arithmetic_progression_product (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (b : ℕ), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) = b ^ 2008) :=
by
  sorry

end arithmetic_progression_product_l1887_188736


namespace positive_value_of_A_l1887_188708

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l1887_188708


namespace custom_op_evaluation_l1887_188747

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l1887_188747


namespace find_number_l1887_188751

theorem find_number (N : ℝ) (h : (0.47 * N - 0.36 * 1412) + 66 = 6) : N = 953.87 :=
  sorry

end find_number_l1887_188751


namespace no_integer_roots_l1887_188773

theorem no_integer_roots (x : ℤ) : ¬ (x^2 + 2^2018 * x + 2^2019 = 0) :=
sorry

end no_integer_roots_l1887_188773


namespace box_cost_coffee_pods_l1887_188713

theorem box_cost_coffee_pods :
  ∀ (days : ℕ) (cups_per_day : ℕ) (pods_per_box : ℕ) (total_cost : ℕ), 
  days = 40 → cups_per_day = 3 → pods_per_box = 30 → total_cost = 32 → 
  total_cost / ((days * cups_per_day) / pods_per_box) = 8 := 
by
  intros days cups_per_day pods_per_box total_cost hday hcup hpod hcost
  sorry

end box_cost_coffee_pods_l1887_188713


namespace most_likely_units_digit_is_5_l1887_188746

-- Define the problem conditions
def in_range (n : ℕ) := 1 ≤ n ∧ n ≤ 8
def Jack_pick (J : ℕ) := in_range J
def Jill_pick (J K : ℕ) := in_range K ∧ J ≠ K

-- Define the function to get the units digit of the sum
def units_digit (J K : ℕ) := (J + K) % 10

-- Define the proposition stating the most likely units digit is 5
theorem most_likely_units_digit_is_5 :
  ∃ (d : ℕ), d = 5 ∧
    (∃ (J K : ℕ), Jack_pick J → Jill_pick J K → units_digit J K = d) :=
sorry

end most_likely_units_digit_is_5_l1887_188746


namespace distance_traveled_l1887_188703

def velocity (t : ℝ) : ℝ := t^2 + 1

theorem distance_traveled :
  (∫ t in (0:ℝ)..(3:ℝ), velocity t) = 12 :=
by
  simp [velocity]
  sorry

end distance_traveled_l1887_188703


namespace smallest_four_digit_multiple_of_17_l1887_188757

theorem smallest_four_digit_multiple_of_17 : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ 17 ∣ n ∧ ∀ m, m ≥ 1000 ∧ m < 10000 ∧ 17 ∣ m → n ≤ m := 
by
  use 1003
  sorry

end smallest_four_digit_multiple_of_17_l1887_188757


namespace find_k_l1887_188792

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def construct_number (k : ℕ) : ℕ :=
  let n := 1000
  let a := (10^(2000 - k) - 1) / 9
  let b := (10^(1001) - 1) / 9
  a * 10^(1001) + k * 10^(1001 - k) - b

theorem find_k : ∀ k : ℕ, (construct_number k > 0) ∧ (isPerfectSquare (construct_number k) ↔ k = 2) := 
by 
  intro k
  sorry

end find_k_l1887_188792


namespace total_cost_proof_l1887_188710

def tuition_fee : ℕ := 1644
def room_and_board_cost : ℕ := tuition_fee - 704
def total_cost : ℕ := tuition_fee + room_and_board_cost

theorem total_cost_proof : total_cost = 2584 := 
by
  sorry

end total_cost_proof_l1887_188710


namespace find_unknown_numbers_l1887_188762

def satisfies_condition1 (A B : ℚ) : Prop := 
  0.05 * A = 0.20 * 650 + 0.10 * B

def satisfies_condition2 (A B : ℚ) : Prop := 
  A + B = 4000

def satisfies_condition3 (B C : ℚ) : Prop := 
  C = 2 * B

def satisfies_condition4 (A B C D : ℚ) : Prop := 
  A + B + C = 0.40 * D

theorem find_unknown_numbers (A B C D : ℚ) :
  satisfies_condition1 A B → satisfies_condition2 A B →
  satisfies_condition3 B C → satisfies_condition4 A B C D →
  A = 3533 + 1/3 ∧ B = 466 + 2/3 ∧ C = 933 + 1/3 ∧ D = 12333 + 1/3 :=
by
  sorry

end find_unknown_numbers_l1887_188762


namespace stream_speed_l1887_188768

theorem stream_speed (C S : ℝ) 
    (h1 : C - S = 8) 
    (h2 : C + S = 12) : 
    S = 2 :=
sorry

end stream_speed_l1887_188768


namespace four_digit_numbers_neither_5_nor_7_l1887_188799

-- Define the range of four-digit numbers
def four_digit_numbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999}

-- Define the predicates for multiples of 5, 7, and 35
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0
def is_multiple_of_35 (n : ℕ) : Prop := n % 35 = 0

-- Using set notation to define the sets of multiples
def multiples_of_5 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_5 n}
def multiples_of_7 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_7 n}
def multiples_of_35 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_35 n}

-- Total count of 4-digit numbers
def total_four_digit_numbers : ℕ := 9000

-- Count of multiples of 5, 7, and 35 within 4-digit numbers
def count_multiples_of_5 : ℕ := 1800
def count_multiples_of_7 : ℕ := 1286
def count_multiples_of_35 : ℕ := 257

-- Count of multiples of 5 or 7 using the principle of inclusion-exclusion
def count_multiples_of_5_or_7 : ℕ := count_multiples_of_5 + count_multiples_of_7 - count_multiples_of_35

-- Prove that the number of 4-digit numbers which are multiples of neither 5 nor 7 is 6171
theorem four_digit_numbers_neither_5_nor_7 : 
  (total_four_digit_numbers - count_multiples_of_5_or_7) = 6171 := 
by 
  sorry

end four_digit_numbers_neither_5_nor_7_l1887_188799


namespace number_of_integer_solutions_l1887_188704

theorem number_of_integer_solutions : ∃ (n : ℕ), n = 120 ∧ ∀ (x y z : ℤ), x * y * z = 2008 → n = 120 :=
by
  sorry

end number_of_integer_solutions_l1887_188704


namespace wyatt_headmaster_duration_l1887_188796

def duration_of_wyatt_job (start_month end_month total_months : ℕ) : Prop :=
  start_month <= end_month ∧ total_months = end_month - start_month + 1

theorem wyatt_headmaster_duration : duration_of_wyatt_job 3 12 9 :=
by
  sorry

end wyatt_headmaster_duration_l1887_188796


namespace sqrt_14_plus_2_range_l1887_188797

theorem sqrt_14_plus_2_range :
  5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 :=
by
  sorry

end sqrt_14_plus_2_range_l1887_188797


namespace minimum_value_of_sum_of_squares_l1887_188782

noncomputable def minimum_of_sum_of_squares (a b : ℝ) : ℝ :=
  a^2 + b^2

theorem minimum_value_of_sum_of_squares (a b : ℝ) (h : |a * b| = 6) :
  a^2 + b^2 ≥ 12 :=
by {
  sorry
}

end minimum_value_of_sum_of_squares_l1887_188782


namespace tangent_function_range_l1887_188769

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

theorem tangent_function_range {a : ℝ} :
  (∃ (m : ℝ), 4 * m^3 - 3 * a * m^2 + 6 = 0) ↔ a > 2 * Real.sqrt 33 :=
sorry -- proof omitted

end tangent_function_range_l1887_188769


namespace seq_bn_arithmetic_seq_an_formula_sum_an_terms_l1887_188771

-- (1) Prove that the sequence {b_n} is an arithmetic sequence
theorem seq_bn_arithmetic (a : ℕ → ℕ) (b : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, b (n + 1) - b n = 1 := by
  sorry

-- (2) Find the general formula for the sequence {a_n}
theorem seq_an_formula (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, a n = n * 2^(n - 1) := by
  sorry

-- (3) Find the sum of the first n terms of the sequence {a_n}
theorem sum_an_terms (a : ℕ → ℕ) (S : ℕ → ℤ) (h1 : ∀ n, a n = n * 2^(n - 1)) :
  ∀ n, S n = (n - 1) * 2^n + 1 := by
  sorry

end seq_bn_arithmetic_seq_an_formula_sum_an_terms_l1887_188771


namespace geometric_sequence_relation_l1887_188785

theorem geometric_sequence_relation (a b c : ℝ) (r : ℝ)
  (h1 : -2 * r = a)
  (h2 : a * r = b)
  (h3 : b * r = c)
  (h4 : c * r = -8) :
  b = -4 ∧ a * c = 16 := by
  sorry

end geometric_sequence_relation_l1887_188785


namespace intersection_eq_l1887_188772

-- Universal set and its sets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 > 9}
def N : Set ℝ := {x | -1 < x ∧ x < 4}
def complement_N : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}

-- Prove the intersection
theorem intersection_eq :
  M ∩ complement_N = {x | x < -3 ∨ x ≥ 4} :=
by
  sorry

end intersection_eq_l1887_188772


namespace paper_clips_in_morning_l1887_188763

variable (p : ℕ) (used left : ℕ)

theorem paper_clips_in_morning (h1 : left = 26) (h2 : used = 59) (h3 : left = p - used) : p = 85 :=
by
  sorry

end paper_clips_in_morning_l1887_188763


namespace isosceles_triangle_perimeter_l1887_188778

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), is_isosceles a b c ∧ ((a = 3 ∧ b = 3 ∧ c = 4 ∧ a + b + c = 10) ∨ (a = 3 ∧ b = 4 ∧ c = 4 ∧ a + b + c = 11)) :=
by
  sorry

end isosceles_triangle_perimeter_l1887_188778


namespace solution_set_inequality_l1887_188723

theorem solution_set_inequality (x : ℝ) : (x-3) * (x-1) > 0 → (x < 1 ∨ x > 3) :=
by sorry

end solution_set_inequality_l1887_188723


namespace perfect_square_append_100_digits_l1887_188742

-- Define the number X consisting of 99 nines

def X : ℕ := (10^99 - 1)

theorem perfect_square_append_100_digits :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 :=
by 
  sorry

end perfect_square_append_100_digits_l1887_188742


namespace dogs_running_l1887_188770

theorem dogs_running (total_dogs playing_with_toys barking not_doing_anything running : ℕ)
  (h1 : total_dogs = 88)
  (h2 : playing_with_toys = total_dogs / 2)
  (h3 : barking = total_dogs / 4)
  (h4 : not_doing_anything = 10)
  (h5 : running = total_dogs - playing_with_toys - barking - not_doing_anything) :
  running = 12 :=
sorry

end dogs_running_l1887_188770


namespace findNumberOfItemsSoldByStoreA_l1887_188786

variable (P x : ℝ) -- P is the price of the product, x is the number of items Store A sells

-- Total sales amount for Store A (in yuan)
def totalSalesA := P * x = 7200

-- Total sales amount for Store B (in yuan)
def totalSalesB := 0.8 * P * (x + 15) = 7200

-- Same price in both stores
def samePriceInBothStores := (P > 0)

-- Proof Problem Statement
theorem findNumberOfItemsSoldByStoreA (storeASellsAtListedPrice : totalSalesA P x)
  (storeBSells15MoreItemsAndAt80PercentPrice : totalSalesB P x)
  (priceIsPositive : samePriceInBothStores P) :
  x = 60 :=
sorry

end findNumberOfItemsSoldByStoreA_l1887_188786


namespace two_a_sq_minus_six_b_plus_one_l1887_188707

theorem two_a_sq_minus_six_b_plus_one (a b : ℝ) (h : a^2 - 3 * b = 5) : 2 * a^2 - 6 * b + 1 = 11 := by
  sorry

end two_a_sq_minus_six_b_plus_one_l1887_188707


namespace zoe_candy_bars_needed_l1887_188711

def total_cost : ℝ := 485
def grandma_contribution : ℝ := 250
def per_candy_earning : ℝ := 1.25
def required_candy_bars : ℕ := 188

theorem zoe_candy_bars_needed :
  (total_cost - grandma_contribution) / per_candy_earning = required_candy_bars :=
by
  sorry

end zoe_candy_bars_needed_l1887_188711


namespace third_pasture_cows_l1887_188780

theorem third_pasture_cows (x y : ℝ) (H1 : x + 27 * y = 18) (H2 : 2 * x + 84 * y = 51) : 
  10 * x + 10 * 3 * y = 60 -> 60 / 3 = 20 :=
by
  sorry

end third_pasture_cows_l1887_188780


namespace sum_of_x_and_y_l1887_188722

theorem sum_of_x_and_y (x y : ℕ) (hxpos : 0 < x) (hypos : 1 < y) (hxy : x^y < 500) (hmax : ∀ (a b : ℕ), 0 < a → 1 < b → a^b < 500 → a^b ≤ x^y) : x + y = 24 := 
sorry

end sum_of_x_and_y_l1887_188722


namespace range_of_a_l1887_188724

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) := x^3 - x^2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (1/2) 2 → x2 ∈ Set.Icc (1/2) 2 → f x1 a - g x2 ≥ 0) →
  1 ≤ a :=
by
  sorry

end range_of_a_l1887_188724


namespace hyperbola_eccentricity_l1887_188726

-- Define the hyperbola and the condition of the asymptote passing through (2,1)
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧
               (a ≠ 0 ∧ b ≠ 0) ∧
               (x, y) = (2, 1)

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop :=
  a^2 + b^2 = (b * e)^2

theorem hyperbola_eccentricity (a b e : ℝ) 
  (hx : hyperbola a b)
  (ha : a = 2 * b)
  (ggt: (a^2 = 4 * b^2)) :
  eccentricity a b e → e = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_eccentricity_l1887_188726


namespace tennis_tournament_l1887_188706

noncomputable def tennis_tournament_n (k : ℕ) : ℕ := 8 * k + 1

theorem tennis_tournament (n : ℕ) :
  (∃ k : ℕ, n = tennis_tournament_n k) ↔
  (∃ k : ℕ, n = 8 * k + 1) :=
by sorry

end tennis_tournament_l1887_188706


namespace arithmetic_sequence_a5_eq_6_l1887_188730

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a5_eq_6 (h_arith : is_arithmetic_sequence a_n) (h_sum : a_n 2 + a_n 8 = 12) : a_n 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_eq_6_l1887_188730


namespace trapezium_height_l1887_188733

-- Defining the lengths of the parallel sides and the area of the trapezium
def a : ℝ := 28
def b : ℝ := 18
def area : ℝ := 345

-- Defining the distance between the parallel sides to be proven
def h : ℝ := 15

-- The theorem that proves the distance between the parallel sides
theorem trapezium_height :
  (1 / 2) * (a + b) * h = area :=
by
  sorry

end trapezium_height_l1887_188733


namespace work_time_relation_l1887_188752

theorem work_time_relation (m n k x y z : ℝ) 
    (h1 : 1 / x = m / (y + z)) 
    (h2 : 1 / y = n / (x + z)) 
    (h3 : 1 / z = k / (x + y)) : 
    k = (m + n + 2) / (m * n - 1) :=
by
  sorry

end work_time_relation_l1887_188752


namespace point_P_below_line_l1887_188791

def line_equation (x y : ℝ) : ℝ := 2 * x - y + 3

def point_below_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x - y + 3 > 0

theorem point_P_below_line :
  point_below_line (1, -1) :=
by
  sorry

end point_P_below_line_l1887_188791


namespace T_sum_correct_l1887_188716

-- Defining the sequence T_n
def T (n : ℕ) : ℤ := 
(-1)^n * 2 * n + (-1)^(n + 1) * n

-- Values to compute
def n1 : ℕ := 27
def n2 : ℕ := 43
def n3 : ℕ := 60

-- Sum of particular values
def T_sum : ℤ := T n1 + T n2 + T n3

-- Placeholder value until actual calculation
def expected_sum : ℤ := -42 -- Replace with the correct calculated result

theorem T_sum_correct : T_sum = expected_sum := sorry

end T_sum_correct_l1887_188716


namespace solve_inequality_l1887_188737

theorem solve_inequality : { x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 } = { x | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) } :=
by
  sorry

end solve_inequality_l1887_188737


namespace quadratic_rewriting_l1887_188793

theorem quadratic_rewriting (b n : ℝ) (h₁ : 0 < n)
  (h₂ : ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) :
  b = 4 * Real.sqrt 13 :=
by
  sorry

end quadratic_rewriting_l1887_188793


namespace certain_number_value_l1887_188712

variable {t b c x : ℕ}

theorem certain_number_value 
  (h1 : (t + b + c + 14 + x) / 5 = 12) 
  (h2 : (t + b + c + 29) / 4 = 15) : 
  x = 15 := 
by
  sorry

end certain_number_value_l1887_188712


namespace trip_time_is_approximate_l1887_188754

noncomputable def total_distance : ℝ := 620
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def speed1 : ℝ := 70
noncomputable def speed2 : ℝ := 85
noncomputable def time1 : ℝ := half_distance / speed1
noncomputable def time2 : ℝ := half_distance / speed2
noncomputable def total_time : ℝ := time1 + time2

theorem trip_time_is_approximate :
  abs (total_time - 8.0757) < 0.0001 :=
sorry

end trip_time_is_approximate_l1887_188754


namespace find_principal_l1887_188765

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem find_principal
  (A : ℝ) (r : ℝ) (n t : ℕ)
  (hA : A = 4410)
  (hr : r = 0.05)
  (hn : n = 1)
  (ht : t = 2) :
  ∃ (P : ℝ), compound_interest P r n t = A ∧ P = 4000 :=
by
  sorry

end find_principal_l1887_188765


namespace negation_of_existence_l1887_188761

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * a * x + a > 0 :=
by
  sorry

end negation_of_existence_l1887_188761


namespace line_intersection_l1887_188740

-- Definitions for the parametric lines
def line1 (t : ℝ) : ℝ × ℝ := (3 + t, 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3 * u, 4 - u)

-- Statement that expresses the intersection point condition
theorem line_intersection :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (30 / 7, 18 / 7) :=
by
  sorry

end line_intersection_l1887_188740


namespace find_a_l1887_188739

theorem find_a (a : ℝ) :
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ a = 1 ∨ a = 5/3 :=
by
  sorry

end find_a_l1887_188739


namespace angle_z_value_l1887_188748

theorem angle_z_value
  (ABC BAC : ℝ)
  (h1 : ABC = 70)
  (h2 : BAC = 50)
  (h3 : ∀ BCA : ℝ, BCA + ABC + BAC = 180) :
  ∃ z : ℝ, z = 30 :=
by
  sorry

end angle_z_value_l1887_188748


namespace find_k_l1887_188766

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (k * x - 1)

theorem find_k (k : ℝ) : (∀ x : ℝ, f k (f k x) = x) ↔ k = -2 :=
  sorry

end find_k_l1887_188766


namespace consecutive_integers_sum_l1887_188705

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 380) : x + (x + 1) = 39 := by
  sorry

end consecutive_integers_sum_l1887_188705


namespace am_gm_inequality_l1887_188798

-- Definitions of the variables and hypotheses
variables {a b : ℝ}

-- The theorem statement
theorem am_gm_inequality (h : a * b > 0) : a / b + b / a ≥ 2 :=
sorry

end am_gm_inequality_l1887_188798


namespace minimum_value_l1887_188731

noncomputable def minimum_y_over_2x_plus_1_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : ℝ :=
  (y / (2 * x)) + (1 / y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) :
  minimum_y_over_2x_plus_1_over_y x y hx hy h = 2 + Real.sqrt 2 :=
sorry

end minimum_value_l1887_188731


namespace largest_multiple_of_15_less_than_500_l1887_188781

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l1887_188781


namespace cards_dealt_to_people_l1887_188749

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l1887_188749


namespace remainder_of_n_mod_9_eq_5_l1887_188728

-- Definitions of the variables and conditions
variables (a b c n : ℕ)

-- The given conditions as assumptions
def conditions : Prop :=
  a + b + c = 63 ∧
  a = c + 22 ∧
  n = 2 * a + 3 * b + 4 * c

-- The proof statement that needs to be proven
theorem remainder_of_n_mod_9_eq_5 (h : conditions a b c n) : n % 9 = 5 := 
  sorry

end remainder_of_n_mod_9_eq_5_l1887_188728


namespace wall_width_l1887_188776

theorem wall_width (w h l V : ℝ) (h_eq : h = 4 * w) (l_eq : l = 3 * h) (V_eq : V = w * h * l) (v_val : V = 10368) : w = 6 :=
  sorry

end wall_width_l1887_188776


namespace angle_sum_straight_line_l1887_188755

  theorem angle_sum_straight_line (x : ℝ) (h : 90 + x + 20 = 180) : x = 70 :=
  by
    sorry
  
end angle_sum_straight_line_l1887_188755


namespace distance_from_point_to_line_l1887_188701

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def cartesian_distance_to_line (point : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (point.snd - y_line)

theorem distance_from_point_to_line
  (ρ θ : ℝ)
  (h_point : ρ = 2 ∧ θ = Real.pi / 6)
  (h_line : ∀ θ, (3 : ℝ) = ρ * Real.sin θ) :
  cartesian_distance_to_line (polar_to_cartesian ρ θ) 3 = 2 :=
  sorry

end distance_from_point_to_line_l1887_188701


namespace rearrange_marked_cells_below_diagonal_l1887_188779

theorem rearrange_marked_cells_below_diagonal (n : ℕ) (marked_cells : Finset (Fin n × Fin n)) :
  marked_cells.card = n - 1 →
  ∃ row_permutation col_permutation : Equiv (Fin n) (Fin n), ∀ (i j : Fin n),
    (row_permutation i, col_permutation j) ∈ marked_cells → j < i :=
by
  sorry

end rearrange_marked_cells_below_diagonal_l1887_188779


namespace PetesOriginalNumber_l1887_188758

-- Define the context and problem
theorem PetesOriginalNumber (x : ℤ) (h : 3 * (2 * x + 12) = 90) : x = 9 :=
by
  -- proof goes here
  sorry

end PetesOriginalNumber_l1887_188758


namespace problem_statement_l1887_188709

-- Define the problem
theorem problem_statement (a b : ℝ) (h : a - b = 1 / 2) : -3 * (b - a) = 3 / 2 := 
  sorry

end problem_statement_l1887_188709


namespace find_a_10_l1887_188745

-- We define the arithmetic sequence and sum properties
def arithmetic_seq (a_1 d : ℚ) (a_n : ℕ → ℚ) :=
  ∀ n, a_n n = a_1 + d * n

def sum_arithmetic_seq (a : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, S_n n = n * (a 1 + a n) / 2

-- Conditions given in the problem
def given_conditions (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  arithmetic_seq a_1 1 a_n ∧ sum_arithmetic_seq a_n S_n ∧ S_n 6 = 4 * S_n 3

-- The theorem to prove
theorem find_a_10 (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : given_conditions a_1 a_n S_n) : a_n 10 = 19 / 2 :=
by sorry

end find_a_10_l1887_188745


namespace infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l1887_188788

open Nat

theorem infinite_solutions_2n_3n_square :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2 :=
sorry

theorem n_multiple_of_40 :
  ∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → (40 ∣ n) :=
sorry

theorem infinite_solutions_general (m : ℕ) (hm : 0 < m) :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, m * n + 1 = a^2 ∧ (m + 1) * n + 1 = b^2 :=
sorry

end infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l1887_188788


namespace exists_square_divisible_by_12_between_100_and_200_l1887_188734

theorem exists_square_divisible_by_12_between_100_and_200 : 
  ∃ x : ℕ, (∃ y : ℕ, x = y * y) ∧ (12 ∣ x) ∧ (100 ≤ x ∧ x ≤ 200) ∧ x = 144 :=
by
  sorry

end exists_square_divisible_by_12_between_100_and_200_l1887_188734


namespace tangent_line_at_x1_f_nonnegative_iff_l1887_188759

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l1887_188759


namespace meeting_time_l1887_188727

-- Define the conditions
def distance : ℕ := 600  -- distance between A and B
def speed_A_to_B : ℕ := 70  -- speed of the first person
def speed_B_to_A : ℕ := 80  -- speed of the second person
def start_time : ℕ := 10  -- start time in hours

-- State the problem formally in Lean 4
theorem meeting_time : (distance / (speed_A_to_B + speed_B_to_A)) + start_time = 14 := 
by
  sorry

end meeting_time_l1887_188727


namespace range_of_a_l1887_188790

theorem range_of_a (a : ℝ) (x : ℤ) (h1 : ∀ x, x > 0 → ⌊(x + a) / 3⌋ = 2) : a < 8 :=
sorry

end range_of_a_l1887_188790


namespace length_sawed_off_l1887_188741

-- Define the lengths as constants
def original_length : ℝ := 8.9
def final_length : ℝ := 6.6

-- State the property to be proven
theorem length_sawed_off : original_length - final_length = 2.3 := by
  sorry

end length_sawed_off_l1887_188741


namespace best_model_is_model4_l1887_188753

-- Define the R^2 values for each model
def R_squared_model1 : ℝ := 0.25
def R_squared_model2 : ℝ := 0.80
def R_squared_model3 : ℝ := 0.50
def R_squared_model4 : ℝ := 0.98

-- Define the highest R^2 value and which model it belongs to
theorem best_model_is_model4 (R1 R2 R3 R4 : ℝ) (h1 : R1 = R_squared_model1) (h2 : R2 = R_squared_model2) (h3 : R3 = R_squared_model3) (h4 : R4 = R_squared_model4) : 
  (R4 = 0.98) ∧ (R4 > R1) ∧ (R4 > R2) ∧ (R4 > R3) :=
by
  sorry

end best_model_is_model4_l1887_188753


namespace calculate_force_l1887_188775

noncomputable def force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem calculate_force : force_on_dam 1000 10 4.8 7.2 3.0 = 252000 := 
  by 
  sorry

end calculate_force_l1887_188775


namespace dishwasher_manager_wage_ratio_l1887_188767

theorem dishwasher_manager_wage_ratio
  (chef_wage dishwasher_wage manager_wage : ℝ)
  (h1 : chef_wage = 1.22 * dishwasher_wage)
  (h2 : dishwasher_wage = r * manager_wage)
  (h3 : manager_wage = 8.50)
  (h4 : chef_wage = manager_wage - 3.315) :
  r = 0.5 :=
sorry

end dishwasher_manager_wage_ratio_l1887_188767


namespace total_team_cost_correct_l1887_188725

variable (jerseyCost shortsCost socksCost cleatsCost waterBottleCost : ℝ)
variable (numPlayers : ℕ)
variable (discountThreshold discountRate salesTaxRate : ℝ)

noncomputable def totalTeamCost : ℝ :=
  let totalCostPerPlayer := jerseyCost + shortsCost + socksCost + cleatsCost + waterBottleCost
  let totalCost := totalCostPerPlayer * numPlayers
  let discount := if totalCost > discountThreshold then totalCost * discountRate else 0
  let discountedTotal := totalCost - discount
  let tax := discountedTotal * salesTaxRate
  let finalCost := discountedTotal + tax
  finalCost

theorem total_team_cost_correct :
  totalTeamCost 25 15.20 6.80 40 12 25 500 0.10 0.07 = 2383.43 := by
  sorry

end total_team_cost_correct_l1887_188725


namespace roots_of_cubic_eq_l1887_188732

theorem roots_of_cubic_eq (r s t p q : ℝ) (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) 
(h3 : r * s * t = r) : r^2 + s^2 + t^2 = p^2 - 2 * q := 
by 
  sorry

end roots_of_cubic_eq_l1887_188732


namespace ted_worked_hours_l1887_188743

variable (t : ℝ)
variable (julie_rate ted_rate combined_rate : ℝ)
variable (julie_alone_time : ℝ)
variable (job_done : ℝ)

theorem ted_worked_hours :
  julie_rate = 1 / 10 →
  ted_rate = 1 / 8 →
  combined_rate = julie_rate + ted_rate →
  julie_alone_time = 0.9999999999999998 →
  job_done = combined_rate * t + julie_rate * julie_alone_time →
  t = 4 :=
by
  sorry

end ted_worked_hours_l1887_188743


namespace average_nums_correct_l1887_188764

def nums : List ℕ := [55, 48, 507, 2, 684, 42]

theorem average_nums_correct :
  (List.sum nums) / (nums.length) = 223 := by
  sorry

end average_nums_correct_l1887_188764


namespace problem_statement_l1887_188760

def op (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem problem_statement : ((op 7 4) - 12) * 5 = 105 := by
  sorry

end problem_statement_l1887_188760


namespace domain_of_f_l1887_188738

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f : ∀ x : ℝ, x ≠ 8 ↔ ∃ y : ℝ, f x = y :=
  by admit

end domain_of_f_l1887_188738


namespace proposition_not_hold_for_4_l1887_188787

variable (P : ℕ → Prop)

axiom induction_step (k : ℕ) (hk : k > 0) : P k → P (k + 1)
axiom base_case : ¬ P 5

theorem proposition_not_hold_for_4 : ¬ P 4 :=
sorry

end proposition_not_hold_for_4_l1887_188787


namespace height_of_box_l1887_188783

-- Define box dimensions
def box_length := 6
def box_width := 6

-- Define spherical radii
def radius_large := 3
def radius_small := 2

-- Define coordinates
def box_volume (h : ℝ) : Prop :=
  ∃ (z : ℝ), z = 2 + Real.sqrt 23 ∧ 
  z + radius_large = h

theorem height_of_box (h : ℝ) : box_volume h ↔ h = 5 + Real.sqrt 23 := by
  sorry

end height_of_box_l1887_188783


namespace range_of_a_l1887_188750

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | x ≥ a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 :=
by
  sorry

end range_of_a_l1887_188750


namespace age_difference_l1887_188774

theorem age_difference (S M : ℕ) 
  (h1 : S = 35)
  (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 37 :=
by
  sorry

end age_difference_l1887_188774


namespace charge_per_person_on_second_day_l1887_188795

noncomputable def charge_second_day (k : ℕ) (x : ℝ) :=
  let total_revenue := 30 * k + 5 * k * x + 32.5 * k
  let total_visitors := 20 * k
  (total_revenue / total_visitors = 5)

theorem charge_per_person_on_second_day
  (k : ℕ) (hx : charge_second_day k 7.5) :
  7.5 = 7.5 :=
sorry

end charge_per_person_on_second_day_l1887_188795


namespace calculate_area_ADC_l1887_188717

def area_AD (BD DC : ℕ) (area_ABD : ℕ) := 
  area_ABD * DC / BD

theorem calculate_area_ADC
  (BD DC : ℕ) 
  (h_ratio : BD = 5 * DC / 2)
  (area_ABD : ℕ)
  (h_area_ABD : area_ABD = 35) :
  area_AD BD DC area_ABD = 14 := 
by 
  sorry

end calculate_area_ADC_l1887_188717


namespace find_length_of_train_l1887_188719

noncomputable def speed_kmhr : ℝ := 30
noncomputable def time_seconds : ℝ := 9
noncomputable def conversion_factor : ℝ := 5 / 18
noncomputable def speed_ms : ℝ := speed_kmhr * conversion_factor
noncomputable def length_train : ℝ := speed_ms * time_seconds

theorem find_length_of_train : length_train = 74.97 := 
by
  sorry

end find_length_of_train_l1887_188719


namespace smallest_number_of_eggs_l1887_188735

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l1887_188735


namespace mountain_climbing_time_proof_l1887_188702

noncomputable def mountain_climbing_time (x : ℝ) : ℝ := (x + 2) / 4

theorem mountain_climbing_time_proof (x : ℝ) (h1 : (x / 3 + (x + 2) / 4 = 4)) : mountain_climbing_time x = 2 := by
  -- assume the given conditions and proof steps explicitly
  sorry

end mountain_climbing_time_proof_l1887_188702


namespace largest_possible_value_of_s_l1887_188700

theorem largest_possible_value_of_s (r s : Nat) (h1 : r ≥ s) (h2 : s ≥ 3)
  (h3 : (r - 2) * s * 61 = (s - 2) * r * 60) : s = 121 :=
sorry

end largest_possible_value_of_s_l1887_188700


namespace cyclic_sum_inequality_l1887_188718

theorem cyclic_sum_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ( (b + c - a)^2 / (a^2 + (b + c)^2) +
    (c + a - b)^2 / (b^2 + (c + a)^2) +
    (a + b - c)^2 / (c^2 + (a + b)^2) ) ≥ 3 / 5 :=
  sorry

end cyclic_sum_inequality_l1887_188718


namespace sin_3pi_div_2_eq_neg_1_l1887_188777

theorem sin_3pi_div_2_eq_neg_1 : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end sin_3pi_div_2_eq_neg_1_l1887_188777


namespace percentage_who_do_not_have_job_of_choice_have_university_diploma_l1887_188784

theorem percentage_who_do_not_have_job_of_choice_have_university_diploma :
  ∀ (total_population university_diploma job_of_choice no_diploma_job_of_choice : ℝ),
    total_population = 100 →
    job_of_choice = 40 →
    no_diploma_job_of_choice = 10 →
    university_diploma = 48 →
    ((university_diploma - (job_of_choice - no_diploma_job_of_choice)) / (total_population - job_of_choice)) * 100 = 30 :=
by
  intros total_population university_diploma job_of_choice no_diploma_job_of_choice h1 h2 h3 h4
  sorry

end percentage_who_do_not_have_job_of_choice_have_university_diploma_l1887_188784


namespace find_N_mod_inverse_l1887_188720

-- Definitions based on given conditions
def A := 111112
def B := 142858
def M := 1000003
def AB : Nat := (A * B) % M
def N := 513487

-- Statement to prove
theorem find_N_mod_inverse : (711812 * N) % M = 1 := by
  -- Proof skipped as per instruction
  sorry

end find_N_mod_inverse_l1887_188720
