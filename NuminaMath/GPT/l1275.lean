import Mathlib

namespace arithmetic_sequence_sum_zero_l1275_127518

theorem arithmetic_sequence_sum_zero {a1 d n : ℤ} 
(h1 : a1 = 35) 
(h2 : d = -2) 
(h3 : (n * (2 * a1 + (n - 1) * d)) / 2 = 0) : 
n = 36 :=
by sorry

end arithmetic_sequence_sum_zero_l1275_127518


namespace simplify_division_l1275_127581

theorem simplify_division :
  (27 * 10^9) / (9 * 10^5) = 30000 :=
  sorry

end simplify_division_l1275_127581


namespace repeating_decimal_product_l1275_127514

-- Define the repeating decimal 0.\overline{137} as a fraction
def repeating_decimal_137 : ℚ := 137 / 999

-- Define the repeating decimal 0.\overline{6} as a fraction
def repeating_decimal_6 : ℚ := 2 / 3

-- The problem is to prove that the product of these fractions is 274 / 2997
theorem repeating_decimal_product : repeating_decimal_137 * repeating_decimal_6 = 274 / 2997 := by
  sorry

end repeating_decimal_product_l1275_127514


namespace minimum_value_fraction_l1275_127585

theorem minimum_value_fraction (x : ℝ) (h : x > 6) : (∃ c : ℝ, c = 12 ∧ ((x = c) → (x^2 / (x - 6) = 18)))
  ∧ (∀ y : ℝ, y > 6 → y^2 / (y - 6) ≥ 18) :=
by {
  sorry
}

end minimum_value_fraction_l1275_127585


namespace steve_matching_pairs_l1275_127516

/-- Steve's total number of socks -/
def total_socks : ℕ := 25

/-- Number of Steve's mismatching socks -/
def mismatching_socks : ℕ := 17

/-- Number of Steve's matching socks -/
def matching_socks : ℕ := total_socks - mismatching_socks

/-- Number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := matching_socks / 2

/-- Proof that Steve has 4 pairs of matching socks -/
theorem steve_matching_pairs : matching_pairs = 4 := by
  sorry

end steve_matching_pairs_l1275_127516


namespace quad_side_difference_l1275_127542

theorem quad_side_difference (a b c d s x y : ℝ)
  (h1 : a = 80) (h2 : b = 100) (h3 : c = 150) (h4 : d = 120)
  (semiperimeter : s = (a + b + c + d) / 2)
  (h5 : x + y = c) 
  (h6 : (|x - y| = 30)) : 
  |x - y| = 30 :=
sorry

end quad_side_difference_l1275_127542


namespace y_value_l1275_127591

theorem y_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + 1 / y = 8) (h4 : y + 1 / x = 7 / 12) (h5 : x + y = 7) : y = 49 / 103 :=
by
  sorry

end y_value_l1275_127591


namespace fraction_product_eq_l1275_127536
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end fraction_product_eq_l1275_127536


namespace ryan_fish_count_l1275_127538

theorem ryan_fish_count
  (R : ℕ)
  (J : ℕ)
  (Jeffery_fish : ℕ)
  (h1 : Jeffery_fish = 60)
  (h2 : Jeffery_fish = 2 * R)
  (h3 : J + R + Jeffery_fish = 100)
  : R = 30 :=
by
  sorry

end ryan_fish_count_l1275_127538


namespace matthew_ate_8_l1275_127557

variable (M P A K : ℕ)

def kimberly_ate_5 : Prop := K = 5
def alvin_eggs : Prop := A = 2 * K - 1
def patrick_eggs : Prop := P = A / 2
def matthew_eggs : Prop := M = 2 * P

theorem matthew_ate_8 (M P A K : ℕ) (h1 : kimberly_ate_5 K) (h2 : alvin_eggs A K) (h3 : patrick_eggs P A) (h4 : matthew_eggs M P) : M = 8 := by
  sorry

end matthew_ate_8_l1275_127557


namespace maximum_value_l1275_127546

-- Define the variables as positive real numbers
variables (a b c : ℝ)

-- Define the conditions
def condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 2*a*b*c + 1

-- Define the expression
def expr (a b c : ℝ) : ℝ := (a - 2*b*c) * (b - 2*c*a) * (c - 2*a*b)

-- The theorem stating that under the given conditions, the expression has a maximum value of 1/8
theorem maximum_value : ∀ (a b c : ℝ), condition a b c → expr a b c ≤ 1/8 :=
by
  sorry

end maximum_value_l1275_127546


namespace cos_angle_l1275_127599

noncomputable def angle := -19 * Real.pi / 6

theorem cos_angle : Real.cos angle = Real.sqrt 3 / 2 :=
by sorry

end cos_angle_l1275_127599


namespace minimum_boxes_to_eliminate_50_percent_chance_l1275_127502

def total_boxes : Nat := 30
def high_value_boxes : Nat := 6
def minimum_boxes_to_eliminate (total_boxes high_value_boxes : Nat) : Nat :=
  total_boxes - high_value_boxes - high_value_boxes

theorem minimum_boxes_to_eliminate_50_percent_chance :
  minimum_boxes_to_eliminate total_boxes high_value_boxes = 18 :=
by
  sorry

end minimum_boxes_to_eliminate_50_percent_chance_l1275_127502


namespace width_of_smaller_cuboids_is_4_l1275_127552

def length_smaller_cuboid := 5
def height_smaller_cuboid := 3
def length_larger_cuboid := 16
def width_larger_cuboid := 10
def height_larger_cuboid := 12
def num_smaller_cuboids := 32

theorem width_of_smaller_cuboids_is_4 :
  ∃ W : ℝ, W = 4 ∧ (length_smaller_cuboid * W * height_smaller_cuboid) * num_smaller_cuboids = 
            length_larger_cuboid * width_larger_cuboid * height_larger_cuboid :=
by
  sorry

end width_of_smaller_cuboids_is_4_l1275_127552


namespace boys_in_class_l1275_127528

theorem boys_in_class (students : ℕ) (ratio_girls_boys : ℕ → Prop)
  (h1 : students = 56)
  (h2 : ratio_girls_boys 4 ∧ ratio_girls_boys 3) :
  ∃ k : ℕ, 4 * k + 3 * k = students ∧ 3 * k = 24 :=
by
  sorry

end boys_in_class_l1275_127528


namespace number_of_panes_l1275_127530

theorem number_of_panes (length width total_area : ℕ) (h_length : length = 12) (h_width : width = 8) (h_total_area : total_area = 768) :
  total_area / (length * width) = 8 :=
by
  sorry

end number_of_panes_l1275_127530


namespace max_value_fraction_l1275_127577

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∀ z, z = (x / (2 * x + y) + y / (x + 2 * y)) → z ≤ (2 / 3) :=
by
  sorry

end max_value_fraction_l1275_127577


namespace amgm_inequality_abcd_l1275_127529

-- Define the variables and their conditions
variables {a b c d : ℝ}
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)
variable (hd : 0 < d)

-- State the theorem
theorem amgm_inequality_abcd :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end amgm_inequality_abcd_l1275_127529


namespace tan_alpha_third_quadrant_l1275_127560

theorem tan_alpha_third_quadrant (α : ℝ) 
  (h_eq: Real.sin α = Real.cos α) 
  (h_third: π < α ∧ α < 3 * π / 2) : Real.tan α = 1 := 
by 
  sorry

end tan_alpha_third_quadrant_l1275_127560


namespace multiply_polynomials_l1275_127549

variable {x y : ℝ}

theorem multiply_polynomials (x y : ℝ) :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 :=
by
  sorry

end multiply_polynomials_l1275_127549


namespace correct_system_of_equations_l1275_127592

variable (x y : Real)

-- Conditions
def condition1 : Prop := y = x + 4.5
def condition2 : Prop := 0.5 * y = x - 1

-- Main statement representing the correct system of equations
theorem correct_system_of_equations : condition1 x y ∧ condition2 x y :=
  sorry

end correct_system_of_equations_l1275_127592


namespace oscar_leap_difference_in_feet_l1275_127586

theorem oscar_leap_difference_in_feet 
  (strides_per_gap : ℕ) 
  (leaps_per_gap : ℕ) 
  (total_distance : ℕ) 
  (num_poles : ℕ)
  (h1 : strides_per_gap = 54) 
  (h2 : leaps_per_gap = 15) 
  (h3 : total_distance = 5280) 
  (h4 : num_poles = 51) 
  : (total_distance / (strides_per_gap * (num_poles - 1)) -
       total_distance / (leaps_per_gap * (num_poles - 1)) = 5) :=
by
  sorry

end oscar_leap_difference_in_feet_l1275_127586


namespace greatest_x_for_A_is_perfect_square_l1275_127574

theorem greatest_x_for_A_is_perfect_square :
  ∃ x : ℕ, x = 2008 ∧ ∀ y : ℕ, (∃ k : ℕ, 2^182 + 4^y + 8^700 = k^2) → y ≤ 2008 :=
by 
  sorry

end greatest_x_for_A_is_perfect_square_l1275_127574


namespace polygon_angle_pairs_l1275_127556

theorem polygon_angle_pairs
  {r k : ℕ}
  (h_ratio : (180 * r - 360) / r = (4 / 3) * (180 * k - 360) / k)
  (h_k_lt_15 : k < 15)
  (h_r_ge_3 : r ≥ 3) :
  (k = 7 ∧ r = 42) ∨ (k = 6 ∧ r = 18) ∨ (k = 5 ∧ r = 10) ∨ (k = 4 ∧ r = 6) :=
sorry

end polygon_angle_pairs_l1275_127556


namespace smallest_m_integral_roots_l1275_127508

theorem smallest_m_integral_roots (m : ℕ) : 
  (∃ p q : ℤ, (10 * p * p - ↑m * p + 360 = 0) ∧ (p + q = m / 10) ∧ (p * q = 36) ∧ (p % q = 0 ∨ q % p = 0)) → 
  m = 120 :=
by
sorry

end smallest_m_integral_roots_l1275_127508


namespace no_integer_solutions_l1275_127575

theorem no_integer_solutions (P Q : Polynomial ℤ) (a : ℤ) (hP1 : P.eval a = 0) 
  (hP2 : P.eval (a + 1997) = 0) (hQ : Q.eval 1998 = 2000) : 
  ¬ ∃ x : ℤ, Q.eval (P.eval x) = 1 := 
by
  sorry

end no_integer_solutions_l1275_127575


namespace oak_taller_than_shortest_l1275_127521

noncomputable def pine_tree_height : ℚ := 14 + 1 / 2
noncomputable def elm_tree_height : ℚ := 13 + 1 / 3
noncomputable def oak_tree_height : ℚ := 19 + 1 / 2

theorem oak_taller_than_shortest : 
  oak_tree_height - elm_tree_height = 6 + 1 / 6 := 
  sorry

end oak_taller_than_shortest_l1275_127521


namespace integral_sqrt_a_squared_minus_x_squared_l1275_127567

open Real

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) :
  (∫ x in -a..a, sqrt (a^2 - x^2)) = 1/2 * π * a^2 :=
by
  sorry

end integral_sqrt_a_squared_minus_x_squared_l1275_127567


namespace number_of_height_groups_l1275_127579

theorem number_of_height_groups
  (max_height : ℕ) (min_height : ℕ) (class_width : ℕ)
  (h_max : max_height = 186)
  (h_min : min_height = 167)
  (h_class_width : class_width = 3) :
  (max_height - min_height + class_width - 1) / class_width = 7 := by
  sorry

end number_of_height_groups_l1275_127579


namespace sequence_a3_l1275_127543

theorem sequence_a3 (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (recursion : ∀ n, a (n + 1) = a n / (1 + a n)) : 
  a 3 = 1 / 3 :=
by 
  sorry

end sequence_a3_l1275_127543


namespace part1_solution_set_part2_inequality_l1275_127550

-- Part (1)
theorem part1_solution_set (x : ℝ) : |x| < 2 * x - 1 ↔ 1 < x := by
  sorry

-- Part (2)
theorem part2_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + 2 * b + c = 1) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 4 := by
  sorry

end part1_solution_set_part2_inequality_l1275_127550


namespace stones_in_courtyard_l1275_127531

theorem stones_in_courtyard (S T B : ℕ) (h1 : T = S + 3 * S) (h2 : B = 2 * (T + S)) (h3 : B = 400) : S = 40 :=
by
  sorry

end stones_in_courtyard_l1275_127531


namespace borrowed_movie_price_correct_l1275_127503

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def total_paid : ℝ := 20.00
def change_received : ℝ := 1.37
def tickets_cost : ℝ := number_of_tickets * ticket_price
def total_spent : ℝ := total_paid - change_received
def borrowed_movie_cost : ℝ := total_spent - tickets_cost

theorem borrowed_movie_price_correct : borrowed_movie_cost = 6.79 := by
  sorry

end borrowed_movie_price_correct_l1275_127503


namespace option_d_correct_l1275_127547

theorem option_d_correct (a b : ℝ) (h : a > b) : -b > -a :=
sorry

end option_d_correct_l1275_127547


namespace number_of_girls_l1275_127565

theorem number_of_girls (B G : ℕ) (h₁ : B = 6 * G / 5) (h₂ : B + G = 440) : G = 200 :=
by {
  sorry -- Proof steps here
}

end number_of_girls_l1275_127565


namespace zero_squared_sum_l1275_127568

theorem zero_squared_sum (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := 
by 
  sorry

end zero_squared_sum_l1275_127568


namespace find_x_l1275_127598

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end find_x_l1275_127598


namespace find_f_one_l1275_127505

-- Define the function f(x-3) = 2x^2 - 3x + 1
noncomputable def f (x : ℤ) := 2 * (x+3)^2 - 3 * (x+3) + 1

-- Declare the theorem we intend to prove
theorem find_f_one : f 1 = 21 :=
by
  -- The proof goes here (saying "sorry" because the detailed proof is skipped)
  sorry

end find_f_one_l1275_127505


namespace scientists_arrival_probability_l1275_127576

open Real

theorem scientists_arrival_probability (x y z : ℕ) (n : ℝ) (h : z ≠ 0)
  (hz : ¬ ∃ p : ℕ, Nat.Prime p ∧ p ^ 2 ∣ z)
  (h1 : n = x - y * sqrt z)
  (h2 : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 120 ∧ 0 ≤ b ∧ b ≤ 120 ∧
    |a - b| ≤ n)
  (h3 : (120 - n)^2 / (120 ^ 2) = 0.7) :
  x + y + z = 202 := sorry

end scientists_arrival_probability_l1275_127576


namespace length_of_ab_l1275_127526

variable (a b c d e : ℝ)
variable (bc cd de ac ae ab : ℝ)

axiom bc_eq_3cd : bc = 3 * cd
axiom de_eq_7 : de = 7
axiom ac_eq_11 : ac = 11
axiom ae_eq_20 : ae = 20
axiom ac_def : ac = ab + bc -- Definition of ac
axiom ae_def : ae = ab + bc + cd + de -- Definition of ae

theorem length_of_ab : ab = 5 := by
  sorry

end length_of_ab_l1275_127526


namespace number_of_good_numbers_lt_1000_l1275_127525

def is_good_number (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  sum % 10 < 10 ∧
  (sum / 10) % 10 < 10 ∧
  (sum / 100) % 10 < 10 ∧
  (sum < 1000)

theorem number_of_good_numbers_lt_1000 : ∃ n : ℕ, n = 48 ∧
  (forall k, k < 1000 → k < 1000 → is_good_number k → k = 48) := sorry

end number_of_good_numbers_lt_1000_l1275_127525


namespace interesting_seven_digit_numbers_l1275_127593

theorem interesting_seven_digit_numbers :
  ∃ n : Fin 2 → ℕ, (∀ i : Fin 2, n i = 128) :=
by sorry

end interesting_seven_digit_numbers_l1275_127593


namespace algorithm_find_GCD_Song_Yuan_l1275_127522

theorem algorithm_find_GCD_Song_Yuan :
  (∀ method, method = "continuous subtraction" → method_finds_GCD_Song_Yuan) :=
sorry

end algorithm_find_GCD_Song_Yuan_l1275_127522


namespace solve_equation_nat_numbers_l1275_127555

theorem solve_equation_nat_numbers (a b c d e f g : ℕ) 
  (h : a * b * c * d * e * f * g = a + b + c + d + e + f + g) : 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 2 ∧ g = 7) ∨ (f = 7 ∧ g = 2))) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 3 ∧ g = 4) ∨ (f = 4 ∧ g = 3))) :=
sorry

end solve_equation_nat_numbers_l1275_127555


namespace log_diff_lt_one_l1275_127558

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_diff_lt_one
  (b c x : ℝ)
  (h_eq_sym : ∀ (t : ℝ), (t - 2)^2 + b * (t - 2) + c = (t + 2)^2 + b * (t + 2) + c)
  (h_f_zero_pos : (0)^2 + b * (0) + c > 0)
  (m n : ℝ)
  (h_fm_0 : m^2 + b * m + c = 0)
  (h_fn_0 : n^2 + b * n + c = 0)
  (h_m_ne_n : m ≠ n)
  : log_base 4 m - log_base (1/4) n < 1 :=
  sorry

end log_diff_lt_one_l1275_127558


namespace kramer_vote_percentage_l1275_127544

def percentage_of_votes_cast (K : ℕ) (V : ℕ) : ℕ :=
  (K * 100) / V

theorem kramer_vote_percentage (K : ℕ) (V : ℕ) (h1 : K = 942568) 
  (h2 : V = 4 * K) : percentage_of_votes_cast K V = 25 := 
by 
  rw [h1, h2, percentage_of_votes_cast]
  sorry

end kramer_vote_percentage_l1275_127544


namespace base_seven_sum_l1275_127594

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum_l1275_127594


namespace number_of_valid_pairs_l1275_127548

theorem number_of_valid_pairs :
  (∃! S : ℕ, S = 1250 ∧ ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1000) →
  (3^n < 4^m ∧ 4^m < 4^(m+1) ∧ 4^(m+1) < 3^(n+1))) :=
sorry

end number_of_valid_pairs_l1275_127548


namespace combined_sleep_hours_l1275_127504

theorem combined_sleep_hours :
  let connor_sleep_hours := 6
  let luke_sleep_hours := connor_sleep_hours + 2
  let emma_sleep_hours := connor_sleep_hours - 1
  let ava_sleep_hours :=
    2 * 5 + 
    2 * (5 + 1) + 
    2 * (5 + 2) + 
    (5 + 3)
  let puppy_sleep_hours := 2 * luke_sleep_hours
  let cat_sleep_hours := 4 + 7
  7 * connor_sleep_hours +
  7 * luke_sleep_hours +
  7 * emma_sleep_hours +
  ava_sleep_hours +
  7 * puppy_sleep_hours +
  7 * cat_sleep_hours = 366 :=
by
  sorry

end combined_sleep_hours_l1275_127504


namespace sufficient_but_not_necessary_l1275_127520

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary_l1275_127520


namespace complement_union_l1275_127554

variable (U : Set ℤ) (A : Set ℤ) (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := 
by 
  -- Proof is omitted
  sorry

end complement_union_l1275_127554


namespace brie_clothes_washer_l1275_127539

theorem brie_clothes_washer (total_blouses total_skirts total_slacks : ℕ)
  (blouses_pct skirts_pct slacks_pct : ℝ)
  (h_blouses : total_blouses = 12)
  (h_skirts : total_skirts = 6)
  (h_slacks : total_slacks = 8)
  (h_blouses_pct : blouses_pct = 0.75)
  (h_skirts_pct : skirts_pct = 0.5)
  (h_slacks_pct : slacks_pct = 0.25) :
  let blouses_in_hamper := total_blouses * blouses_pct
  let skirts_in_hamper := total_skirts * skirts_pct
  let slacks_in_hamper := total_slacks * slacks_pct
  blouses_in_hamper + skirts_in_hamper + slacks_in_hamper = 14 := 
by
  sorry

end brie_clothes_washer_l1275_127539


namespace angles_with_same_terminal_side_l1275_127501

theorem angles_with_same_terminal_side (k : ℤ) : 
  (∃ (α : ℝ), α = -437 + k * 360) ↔ (∃ (β : ℝ), β = 283 + k * 360) := 
by
  sorry

end angles_with_same_terminal_side_l1275_127501


namespace equations_create_24_l1275_127566

theorem equations_create_24 :
  ∃ (eq1 eq2 : ℤ),
  ((eq1 = 3 * (-6 + 4 + 10) ∧ eq1 = 24) ∧ 
   (eq2 = 4 - (-6 / 3) * 10 ∧ eq2 = 24)) ∧ 
   eq1 ≠ eq2 := 
by
  sorry

end equations_create_24_l1275_127566


namespace maximum_value_expression_l1275_127533

theorem maximum_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ p, p = x * y ∧ (4 * p^3 - 92 * p^2 + 754 * p) = 441 / 2 :=
by {
  sorry
}

end maximum_value_expression_l1275_127533


namespace elroy_more_miles_l1275_127587

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l1275_127587


namespace harmonic_mean_is_54_div_11_l1275_127511

-- Define lengths of sides
def a : ℕ := 3
def b : ℕ := 6
def c : ℕ := 9

-- Define the harmonic mean calculation function
def harmonic_mean (x y z : ℕ) : ℚ :=
  let reciprocals_sum : ℚ := (1 / x + 1 / y + 1 / z)
  let average_reciprocal : ℚ := reciprocals_sum / 3
  1 / average_reciprocal

-- Prove that the harmonic mean of the given lengths is 54/11
theorem harmonic_mean_is_54_div_11 : harmonic_mean a b c = 54 / 11 := by
  sorry

end harmonic_mean_is_54_div_11_l1275_127511


namespace dogwood_trees_total_is_100_l1275_127534

def initial_dogwood_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20
def total_dogwood_trees : ℕ := initial_dogwood_trees + trees_planted_today + trees_planted_tomorrow

theorem dogwood_trees_total_is_100 : total_dogwood_trees = 100 := by
  sorry  -- Proof goes here

end dogwood_trees_total_is_100_l1275_127534


namespace range_of_m_l1275_127512

open Real

noncomputable def complex_modulus_log_condition (m : ℝ) : Prop :=
  Complex.abs (Complex.log (m : ℂ) / Complex.log 2 + Complex.I * 4) ≤ 5

theorem range_of_m (m : ℝ) (h : complex_modulus_log_condition m) : 
  (1 / 8 : ℝ) ≤ m ∧ m ≤ (8 : ℝ) :=
sorry

end range_of_m_l1275_127512


namespace frog_arrangement_count_l1275_127517

theorem frog_arrangement_count :
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let frogs := green_frogs + red_frogs + blue_frogs
  -- Descriptions:
  -- 1. green_frogs refuse to sit next to red_frogs
  -- 2. green_frogs and red_frogs are fine sitting next to blue_frogs
  -- 3. blue_frogs can sit next to each other
  frogs = 7 → 
  ∃ arrangements : ℕ, arrangements = 72 :=
by 
  sorry

end frog_arrangement_count_l1275_127517


namespace jonathan_daily_burn_l1275_127551

-- Conditions
def daily_calories : ℕ := 2500
def extra_saturday_calories : ℕ := 1000
def weekly_deficit : ℕ := 2500

-- Question and Answer
theorem jonathan_daily_burn :
  let weekly_intake := 6 * daily_calories + (daily_calories + extra_saturday_calories)
  let total_weekly_burn := weekly_intake + weekly_deficit
  total_weekly_burn / 7 = 3000 :=
by
  sorry

end jonathan_daily_burn_l1275_127551


namespace joy_remaining_tape_l1275_127519

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end joy_remaining_tape_l1275_127519


namespace sequence_a2018_l1275_127589

theorem sequence_a2018 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 1) 
  (h2 : a 18 = 0) 
  (h3 : a 2017 = 0) :
  a 2018 = 1000 :=
sorry

end sequence_a2018_l1275_127589


namespace graph_inverse_prop_function_quadrants_l1275_127578

theorem graph_inverse_prop_function_quadrants :
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ y = 4 / x → y > 0) ∨ (x < 0 ∧ y = 4 / x → y < 0) := 
sorry

end graph_inverse_prop_function_quadrants_l1275_127578


namespace problem_statement_l1275_127563

variables {p q r s : ℝ}

theorem problem_statement 
  (h : (p - q) * (r - s) / (q - r) * (s - p) = 3 / 7) : 
  (p - r) * (q - s) / (p - q) * (r - s) = -4 / 3 :=
by sorry

end problem_statement_l1275_127563


namespace part_I_part_II_l1275_127513

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x - 1|

theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ 2 - |x - 1|) : 0 ≤ a ∧ a ≤ 4 := 
sorry

theorem part_II (a : ℝ) (h₁ : a < 2) (h₂ : ∀ x : ℝ, f x a ≥ 3) : a = -4 := 
sorry

end part_I_part_II_l1275_127513


namespace probability_2_1_to_2_5_l1275_127553

noncomputable def F (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then (x - 2)^2
else 1

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then 2 * (x - 2)
else 0

theorem probability_2_1_to_2_5 : 
  (F 2.5 - F 2.1 = 0.24) := 
by
  -- calculations and proof go here, but we skip it with sorry
  sorry

end probability_2_1_to_2_5_l1275_127553


namespace find_positive_integer_n_l1275_127561

theorem find_positive_integer_n (n : ℕ) (hpos : 0 < n) : 
  (n + 1) ∣ (2 * n^2 + 5 * n) ↔ n = 2 :=
by
  sorry

end find_positive_integer_n_l1275_127561


namespace find_common_ratio_l1275_127537

variable (a₃ a₂ : ℝ)
variable (S₁ S₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * S₂ = a₃ - 2
def condition2 : Prop := 3 * S₁ = a₂ - 2

-- Theorem statement
theorem find_common_ratio (h1 : condition1 a₃ S₂)
                          (h2 : condition2 a₂ S₁) : 
                          (a₃ / a₂ = 4) :=
by 
  sorry

end find_common_ratio_l1275_127537


namespace original_number_is_10_l1275_127595

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l1275_127595


namespace domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l1275_127571

-- For the function y = sqrt(3 + 2x)
theorem domain_sqrt_3_plus_2x (x : ℝ) : 3 + 2 * x ≥ 0 -> x ∈ Set.Ici (-3 / 2) :=
by
  sorry

-- For the function f(x) = 1 + sqrt(9 - x^2)
theorem domain_1_plus_sqrt_9_minus_x2 (x : ℝ) : 9 - x^2 ≥ 0 -> x ∈ Set.Icc (-3) 3 :=
by
  sorry

-- For the function φ(x) = sqrt(log((5x - x^2) / 4))
theorem domain_sqrt_log_5x_minus_x2_over_4 (x : ℝ) : (5 * x - x^2) / 4 > 0 ∧ (5 * x - x^2) / 4 ≥ 1 -> x ∈ Set.Icc 1 4 :=
by
  sorry

-- For the function y = sqrt(3 - x) + arccos((x - 2) / 3)
theorem domain_sqrt_3_minus_x_plus_arccos (x : ℝ) : 3 - x ≥ 0 ∧ -1 ≤ (x - 2) / 3 ∧ (x - 2) / 3 ≤ 1 -> x ∈ Set.Icc (-1) 3 :=
by
  sorry

end domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l1275_127571


namespace range_of_b_l1275_127572

theorem range_of_b (a b : ℝ) (h₁ : a ≤ -1) (h₂ : a * 2 * b - b - 3 * a ≥ 0) : b ≤ 1 := by
  sorry

end range_of_b_l1275_127572


namespace inequality_proof_l1275_127597

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
    (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := 
by 
  sorry

end inequality_proof_l1275_127597


namespace store_profit_l1275_127532

variable (m n : ℝ)
variable (h_mn : m > n)

theorem store_profit : 10 * (m - n) > 0 :=
by
  sorry

end store_profit_l1275_127532


namespace final_answer_is_correct_l1275_127527

-- Define the chosen number
def chosen_number : ℤ := 1376

-- Define the division by 8
def division_result : ℤ := chosen_number / 8

-- Define the final answer
def final_answer : ℤ := division_result - 160

-- Theorem statement
theorem final_answer_is_correct : final_answer = 12 := by
  sorry

end final_answer_is_correct_l1275_127527


namespace find_n_for_2013_in_expansion_l1275_127545

/-- Define the pattern for the last term of the expansion of n^3 -/
def last_term (n : ℕ) : ℕ :=
  n^2 + n - 1

/-- The main problem statement -/
theorem find_n_for_2013_in_expansion :
  ∃ n : ℕ, last_term (n - 1) ≤ 2013 ∧ 2013 < last_term n ∧ n = 45 :=
by
  sorry

end find_n_for_2013_in_expansion_l1275_127545


namespace constant_sequence_if_and_only_if_arith_geo_progression_l1275_127535

/-- A sequence a_n is both an arithmetic and geometric progression if and only if it is constant --/
theorem constant_sequence_if_and_only_if_arith_geo_progression (a : ℕ → ℝ) :
  (∃ q d : ℝ, (∀ n : ℕ, a (n+1) - a n = d) ∧ (∀ n : ℕ, a n = a 0 * q ^ n)) ↔ (∃ c : ℝ, ∀ n : ℕ, a n = c) := 
sorry

end constant_sequence_if_and_only_if_arith_geo_progression_l1275_127535


namespace negation_of_proposition_l1275_127540

variable (x y : ℝ)

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, (x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) ↔ 
  (∃ x y : ℝ, (x^2 + y^2 ≠ 0) ∧ (x ≠ 0 ∨ y ≠ 0)) :=
sorry

end negation_of_proposition_l1275_127540


namespace eval_power_81_11_over_4_l1275_127564

theorem eval_power_81_11_over_4 : 81^(11/4) = 177147 := by
  sorry

end eval_power_81_11_over_4_l1275_127564


namespace cost_price_per_metre_l1275_127507

theorem cost_price_per_metre (total_metres total_sale total_loss_per_metre total_sell_price : ℕ) (h1: total_metres = 500) (h2: total_sell_price = 15000) (h3: total_loss_per_metre = 10) : total_sell_price + (total_loss_per_metre * total_metres) / total_metres = 40 :=
by
  sorry

end cost_price_per_metre_l1275_127507


namespace solve_x_from_operation_l1275_127562

def operation (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_x_from_operation :
  ∀ x : ℝ, operation (2 * x) 3 3 (-1) = 3 → x = 1 :=
by
  intros x h
  sorry

end solve_x_from_operation_l1275_127562


namespace minimal_total_cost_l1275_127569

def waterway_length : ℝ := 100
def max_speed : ℝ := 50
def other_costs_per_hour : ℝ := 3240
def speed_at_ten_cost : ℝ := 10
def fuel_cost_at_ten : ℝ := 60
def proportionality_constant : ℝ := 0.06

noncomputable def total_cost (v : ℝ) : ℝ :=
  6 * v^2 + 324000 / v

theorem minimal_total_cost :
  (∃ v : ℝ, 0 < v ∧ v ≤ max_speed ∧ total_cost v = 16200) ∧ 
  (∀ v : ℝ, 0 < v ∧ v ≤ max_speed → total_cost v ≥ 16200) :=
sorry

end minimal_total_cost_l1275_127569


namespace walkway_time_stopped_l1275_127590

noncomputable def effective_speed_with_walkway (v_p v_w : ℝ) : ℝ := v_p + v_w
noncomputable def effective_speed_against_walkway (v_p v_w : ℝ) : ℝ := v_p - v_w

theorem walkway_time_stopped (v_p v_w : ℝ) (h1 : effective_speed_with_walkway v_p v_w = 2)
                            (h2 : effective_speed_against_walkway v_p v_w = 2 / 3) :
    (200 / v_p) = 150 :=
by sorry

end walkway_time_stopped_l1275_127590


namespace quadratic_equation_proof_l1275_127541

def is_quadratic_equation (eqn : String) : Prop :=
  eqn = "x^2 + 2x - 1 = 0"

theorem quadratic_equation_proof :
  is_quadratic_equation "x^2 + 2x - 1 = 0" :=
sorry

end quadratic_equation_proof_l1275_127541


namespace fractions_inequality_l1275_127515

variable {a b c d : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0)

theorem fractions_inequality : 
  (a > b) → (b > 0) → (c < d) → (d < 0) → (a / d < b / c) :=
by
  intros h1 h2 h3 h4
  sorry

end fractions_inequality_l1275_127515


namespace find_root_and_m_l1275_127573

theorem find_root_and_m (m x₂ : ℝ) (h₁ : (1 : ℝ) * x₂ = 3) (h₂ : (1 : ℝ) + x₂ = -m) : 
  x₂ = 3 ∧ m = -4 :=
by
  sorry

end find_root_and_m_l1275_127573


namespace tomato_plant_relationship_l1275_127506

theorem tomato_plant_relationship :
  ∃ (T1 T2 T3 : ℕ), T1 = 24 ∧ T3 = T2 + 2 ∧ T1 + T2 + T3 = 60 ∧ T1 - T2 = 7 :=
by
  sorry

end tomato_plant_relationship_l1275_127506


namespace tom_total_expenditure_l1275_127596

noncomputable def tom_spent_total : ℝ :=
  let skateboard_price := 9.46
  let skateboard_discount := 0.10 * skateboard_price
  let discounted_skateboard := skateboard_price - skateboard_discount

  let marbles_price := 9.56
  let marbles_discount := 0.10 * marbles_price
  let discounted_marbles := marbles_price - marbles_discount

  let shorts_price := 14.50

  let figures_price := 12.60
  let figures_discount := 0.20 * figures_price
  let discounted_figures := figures_price - figures_discount

  let puzzle_price := 6.35
  let puzzle_discount := 0.15 * puzzle_price
  let discounted_puzzle := puzzle_price - puzzle_discount

  let game_price_eur := 20.50
  let game_discount_eur := 0.05 * game_price_eur
  let discounted_game_eur := game_price_eur - game_discount_eur
  let exchange_rate := 1.12
  let discounted_game_usd := discounted_game_eur * exchange_rate

  discounted_skateboard + discounted_marbles + shorts_price + discounted_figures + discounted_puzzle + discounted_game_usd

theorem tom_total_expenditure : abs (tom_spent_total - 68.91) < 0.01 :=
by norm_num1; sorry

end tom_total_expenditure_l1275_127596


namespace square_field_area_l1275_127523

theorem square_field_area (s A : ℝ) (h1 : 10 * 4 * s = 9280) (h2 : A = s^2) : A = 53824 :=
by {
  sorry -- The proof goes here
}

end square_field_area_l1275_127523


namespace primes_between_30_and_60_l1275_127510

theorem primes_between_30_and_60 (list_of_primes : List ℕ) 
  (H1 : list_of_primes = [31, 37, 41, 43, 47, 53, 59]) :
  (list_of_primes.headI * list_of_primes.reverse.headI) = 1829 := by
  sorry

end primes_between_30_and_60_l1275_127510


namespace find_s_squared_l1275_127580

-- Define the conditions and entities in Lean
variable (s : ℝ)
def passesThrough (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 / 9) - (x^2 / a^2) = 1

-- State the given conditions as hypotheses
axiom h₀ : passesThrough 0 3 3 1
axiom h₁ : passesThrough 5 (-3) 25 1
axiom h₂ : passesThrough s (-4) 25 1

-- State the theorem we want to prove
theorem find_s_squared : s^2 = 175 / 9 := by
  sorry

end find_s_squared_l1275_127580


namespace multiplicative_inverse_correct_l1275_127588

def A : ℕ := 123456
def B : ℕ := 654321
def m : ℕ := 1234567
def AB_mod : ℕ := (A * B) % m

def N : ℕ := 513629

theorem multiplicative_inverse_correct (h : AB_mod = 470160) : (470160 * N) % m = 1 := 
by 
  have hN : N = 513629 := rfl
  have hAB : AB_mod = 470160 := h
  sorry

end multiplicative_inverse_correct_l1275_127588


namespace M_intersect_N_eq_l1275_127583

def M : Set ℝ := { y | ∃ x, y = x ^ 2 }
def N : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 2 + y^2 ≤ 1) }

theorem M_intersect_N_eq : M ∩ { y | (y ∈ Set.univ) } = { y | 0 ≤ y ∧ y ≤ Real.sqrt 2 } :=
by
  sorry

end M_intersect_N_eq_l1275_127583


namespace sqrt_40000_eq_200_l1275_127500

theorem sqrt_40000_eq_200 : Real.sqrt 40000 = 200 := 
sorry

end sqrt_40000_eq_200_l1275_127500


namespace somu_age_ratio_l1275_127582

theorem somu_age_ratio (S F : ℕ) (h1 : S = 20) (h2 : S - 10 = (F - 10) / 5) : S / F = 1 / 3 :=
by
  sorry

end somu_age_ratio_l1275_127582


namespace max_cables_to_ensure_communication_l1275_127584

theorem max_cables_to_ensure_communication
    (A B : ℕ) (n : ℕ) 
    (hA : A = 16) (hB : B = 12) (hn : n = 28) :
    (A * B ≤ 192) ∧ (A * B = 192) :=
by
  sorry

end max_cables_to_ensure_communication_l1275_127584


namespace dave_initial_video_games_l1275_127570

theorem dave_initial_video_games (non_working_games working_game_price total_earnings : ℕ) 
  (h1 : non_working_games = 2) 
  (h2 : working_game_price = 4) 
  (h3 : total_earnings = 32) : 
  non_working_games + total_earnings / working_game_price = 10 := 
by 
  sorry

end dave_initial_video_games_l1275_127570


namespace sarah_saves_5_dollars_l1275_127509

noncomputable def price_per_pair : ℕ := 40

noncomputable def promotion_A_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n / 2 else price_per_pair

noncomputable def promotion_B_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n - (15 * (n / 2)) else price_per_pair

noncomputable def total_price_promotion_A : ℕ :=
price_per_pair + (price_per_pair / 2)

noncomputable def total_price_promotion_B : ℕ :=
price_per_pair + (price_per_pair - 15)

theorem sarah_saves_5_dollars : total_price_promotion_B - total_price_promotion_A = 5 :=
by
  rw [total_price_promotion_B, total_price_promotion_A]
  norm_num
  sorry

end sarah_saves_5_dollars_l1275_127509


namespace inequality_condition_l1275_127524

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_condition (a b: ℝ) (h : ∀ x : ℝ, f a b x ≥ 0) : (b * (a + 1)) / 2 < 3 / 4 := 
sorry

end inequality_condition_l1275_127524


namespace girls_in_class_l1275_127559

theorem girls_in_class (B G : ℕ) 
  (h1 : G = B + 3) 
  (h2 : G + B = 41) : 
  G = 22 := 
sorry

end girls_in_class_l1275_127559
