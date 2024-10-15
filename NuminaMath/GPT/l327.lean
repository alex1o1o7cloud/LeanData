import Mathlib

namespace NUMINAMATH_GPT_units_digit_8th_group_l327_32782

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_8th_group (t k : ℕ) (ht : t = 7) (hk : k = 8) : 
  units_digit (t + k) = 5 := 
by
  -- Proof step will go here.
  sorry

end NUMINAMATH_GPT_units_digit_8th_group_l327_32782


namespace NUMINAMATH_GPT_solve_for_y_l327_32777

theorem solve_for_y (y : ℤ) : 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) → y = -37 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l327_32777


namespace NUMINAMATH_GPT_floor_equation_solution_l327_32704

theorem floor_equation_solution (a b : ℝ) :
  (∀ x y : ℝ, ⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋) → (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_GPT_floor_equation_solution_l327_32704


namespace NUMINAMATH_GPT_union_complement_A_B_eq_U_l327_32784

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5, 7}
def A : Set ℕ := {4, 7}
def B : Set ℕ := {1, 3, 4, 7}

-- Define the complement of A with respect to U (C_U A)
def C_U_A : Set ℕ := U \ A
-- Define the complement of B with respect to U (C_U B)
def C_U_B : Set ℕ := U \ B

-- The theorem to prove
theorem union_complement_A_B_eq_U : (C_U_A ∪ B) = U := by
  sorry

end NUMINAMATH_GPT_union_complement_A_B_eq_U_l327_32784


namespace NUMINAMATH_GPT_total_carrots_l327_32766

theorem total_carrots (carrots_sandy carrots_mary : ℕ) (h1 : carrots_sandy = 8) (h2 : carrots_mary = 6) :
  carrots_sandy + carrots_mary = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_carrots_l327_32766


namespace NUMINAMATH_GPT_min_value_a_b_c_l327_32733

def A_n (a : ℕ) (n : ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (b : ℕ) (n : ℕ) : ℕ := b * ((10^n - 1) / 9)
def C_n (c : ℕ) (n : ℕ) : ℕ := c * ((10^(2*n) - 1) / 9)

theorem min_value_a_b_c (a b c : ℕ) (Ha : 0 < a ∧ a < 10) (Hb : 0 < b ∧ b < 10) (Hc : 0 < c ∧ c < 10) :
  (∃ n1 n2 : ℕ, (n1 ≠ n2) ∧ (C_n c n1 - A_n a n1 = B_n b n1 ^ 2) ∧ (C_n c n2 - A_n a n2 = B_n b n2 ^ 2)) →
  a + b + c = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_b_c_l327_32733


namespace NUMINAMATH_GPT_verify_statements_l327_32750

noncomputable def f (x : ℝ) : ℝ := 10 ^ x

theorem verify_statements (x1 x2 : ℝ) (h : x1 ≠ x2) :
  (f (x1 + x2) = f x1 * f x2) ∧
  (f x1 - f x2) / (x1 - x2) > 0 :=
by
  sorry

end NUMINAMATH_GPT_verify_statements_l327_32750


namespace NUMINAMATH_GPT_find_number_l327_32770

-- Define the condition
def condition : Prop := ∃ x : ℝ, x / 0.02 = 50

-- State the theorem to prove
theorem find_number (x : ℝ) (h : x / 0.02 = 50) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_number_l327_32770


namespace NUMINAMATH_GPT_votes_cast_proof_l327_32723

variable (V : ℝ)
variable (candidate_votes : ℝ)
variable (rival_votes : ℝ)

noncomputable def total_votes_cast : Prop :=
  candidate_votes = 0.40 * V ∧ 
  rival_votes = candidate_votes + 2000 ∧ 
  rival_votes = 0.60 * V ∧ 
  V = 10000

theorem votes_cast_proof : total_votes_cast V candidate_votes rival_votes :=
by {
  sorry
  }

end NUMINAMATH_GPT_votes_cast_proof_l327_32723


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l327_32729

-- Define the given conditions: Two numbers a and b, their HCF, and their product.
variables (a b : ℕ)
def hcf : ℕ := 55
def product := 82500

-- Define the concept of HCF and LCM, using the provided relationship in the problem
def gcd_ab := hcf
def lcm_ab := (product / gcd_ab)

-- State the main theorem to prove: The LCM of the two numbers is 1500
theorem lcm_of_two_numbers : lcm_ab = 1500 := by
  -- This is the place where the actual proof steps would go
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l327_32729


namespace NUMINAMATH_GPT_sqrt_expression_l327_32743

open Real

theorem sqrt_expression :
  3 * sqrt 12 / (3 * sqrt (1 / 3)) - 2 * sqrt 3 = 6 - 2 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l327_32743


namespace NUMINAMATH_GPT_range_f_in_interval_l327_32788

-- Define the function f and the interval
def f (x : ℝ) (f_deriv_neg1 : ℝ) := x^3 + 2 * x * f_deriv_neg1
def interval := Set.Icc (-2 : ℝ) (3 : ℝ)

-- State the theorem
theorem range_f_in_interval :
  ∃ (f_deriv_neg1 : ℝ),
  (∀ x ∈ interval, f x f_deriv_neg1 ∈ Set.Icc (-4 * Real.sqrt 2) 9) :=
sorry

end NUMINAMATH_GPT_range_f_in_interval_l327_32788


namespace NUMINAMATH_GPT_candy_count_correct_l327_32746

-- Define initial count of candy
def initial_candy : ℕ := 47

-- Define number of pieces of candy eaten
def eaten_candy : ℕ := 25

-- Define number of pieces of candy received
def received_candy : ℕ := 40

-- The final count of candy is what we are proving
theorem candy_count_correct : initial_candy - eaten_candy + received_candy = 62 :=
by
  sorry

end NUMINAMATH_GPT_candy_count_correct_l327_32746


namespace NUMINAMATH_GPT_distance_from_Bangalore_l327_32744

noncomputable def calculate_distance (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) : ℕ :=
  let total_travel_minutes := (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute) - halt_minutes
  let total_travel_hours := total_travel_minutes / 60
  speed * total_travel_hours

theorem distance_from_Bangalore (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) :
  speed = 87 ∧ start_hour = 9 ∧ start_minute = 0 ∧ end_hour = 13 ∧ end_minute = 45 ∧ halt_minutes = 45 →
  calculate_distance speed start_hour start_minute end_hour end_minute halt_minutes = 348 := by
  sorry

end NUMINAMATH_GPT_distance_from_Bangalore_l327_32744


namespace NUMINAMATH_GPT_quadratic_complete_square_l327_32786

theorem quadratic_complete_square (x : ℝ) (m t : ℝ) :
  (4 * x^2 - 16 * x - 448 = 0) → ((x + m) ^ 2 = t) → (t = 116) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l327_32786


namespace NUMINAMATH_GPT_friendly_point_pairs_l327_32709

def friendly_points (k : ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (a, -1 / a) ∧ B = (-a, 1 / a) ∧
  B.2 = k * B.1 + 1 + k

theorem friendly_point_pairs : ∀ (k : ℝ), k ≥ 0 → 
  ∃ n, (n = 1 ∨ n = 2) ∧
  (∀ a : ℝ, a > 0 →
    friendly_points k a (a, -1 / a) (-a, 1 / a))
:= by
  sorry

end NUMINAMATH_GPT_friendly_point_pairs_l327_32709


namespace NUMINAMATH_GPT_last_score_is_71_l327_32710

theorem last_score_is_71 (scores : List ℕ) (h : scores = [71, 74, 79, 85, 88, 92]) (sum_eq: scores.sum = 489) :
  ∃ s : ℕ, s ∈ scores ∧ 
           (∃ avg : ℕ, avg = (scores.sum - s) / 5 ∧ 
           ∀ lst : List ℕ, lst = scores.erase s → (∀ n, n ∈ lst → lst.sum % (lst.length - 1) = 0)) :=
  sorry

end NUMINAMATH_GPT_last_score_is_71_l327_32710


namespace NUMINAMATH_GPT_alexis_sew_skirt_time_l327_32718

theorem alexis_sew_skirt_time : 
  ∀ (S : ℝ), 
  (∀ (C : ℝ), C = 7) → 
  (6 * S + 4 * 7 = 40) → 
  S = 2 := 
by
  intros S _ h
  sorry

end NUMINAMATH_GPT_alexis_sew_skirt_time_l327_32718


namespace NUMINAMATH_GPT_range_of_a_l327_32751

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) ↔ a < 1004 := 
sorry

end NUMINAMATH_GPT_range_of_a_l327_32751


namespace NUMINAMATH_GPT_arithmetic_seq_first_term_l327_32764

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (n : ℕ) (a : ℚ)
  (h₁ : ∀ n, S n = n * (2 * a + (n - 1) * 5) / 2)
  (h₂ : ∀ n, S (3 * n) / S n = 9) :
  a = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_first_term_l327_32764


namespace NUMINAMATH_GPT_fraction_traditionalists_l327_32753

theorem fraction_traditionalists {P T : ℕ} (h1 : ∀ (i : ℕ), i < 5 → T = P / 15) (h2 : T = P / 15) :
  (5 * T : ℚ) / (P + 5 * T : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_traditionalists_l327_32753


namespace NUMINAMATH_GPT_average_speed_third_hour_l327_32785

theorem average_speed_third_hour
  (total_distance : ℝ)
  (total_time : ℝ)
  (speed_first_hour : ℝ)
  (speed_second_hour : ℝ)
  (speed_third_hour : ℝ) :
  total_distance = 150 →
  total_time = 3 →
  speed_first_hour = 45 →
  speed_second_hour = 55 →
  (speed_first_hour + speed_second_hour + speed_third_hour) / total_time = 50 →
  speed_third_hour = 50 :=
sorry

end NUMINAMATH_GPT_average_speed_third_hour_l327_32785


namespace NUMINAMATH_GPT_larry_spent_on_lunch_l327_32769

noncomputable def starting_amount : ℕ := 22
noncomputable def ending_amount : ℕ := 15
noncomputable def amount_given_to_brother : ℕ := 2

theorem larry_spent_on_lunch : 
  (starting_amount - (ending_amount + amount_given_to_brother)) = 5 :=
by
  -- The conditions and the proof structure would be elaborated here
  sorry

end NUMINAMATH_GPT_larry_spent_on_lunch_l327_32769


namespace NUMINAMATH_GPT_smallest_number_of_eggs_l327_32720

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 100) : 102 ≤ 15 * c - 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_eggs_l327_32720


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l327_32715

theorem common_ratio_of_geometric_sequence 
  (a : ℝ) (log2_3 log4_3 log8_3: ℝ)
  (h1: log4_3 = log2_3 / 2)
  (h2: log8_3 = log2_3 / 3) 
  (h_geometric: ∀ i j, 
    i = a + log2_3 → 
    j = a + log4_3 →
    j / i = a + log8_3 / j / i / j
  ) :
  (a + log4_3) / (a + log2_3) = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l327_32715


namespace NUMINAMATH_GPT_find_ordered_pair_l327_32758

theorem find_ordered_pair (s l : ℝ) :
  (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) →
  (s = -19 ∧ l = -7 / 2) :=
by
  intro h
  have : (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) := h
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l327_32758


namespace NUMINAMATH_GPT_solve_a_plus_b_l327_32781

theorem solve_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : 143 * a + 500 * b = 2001) : a + b = 9 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_solve_a_plus_b_l327_32781


namespace NUMINAMATH_GPT_tan_half_angle_inequality_l327_32763

theorem tan_half_angle_inequality (a b c : ℝ) (α β : ℝ)
  (h : a + b < 3 * c)
  (h_tan_identity : Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c)) :
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_half_angle_inequality_l327_32763


namespace NUMINAMATH_GPT_find_n_l327_32707

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end NUMINAMATH_GPT_find_n_l327_32707


namespace NUMINAMATH_GPT_final_point_P_after_transformations_l327_32787

noncomputable def point := (ℝ × ℝ)

def rotate_90_clockwise (p : point) : point :=
  (-p.2, p.1)

def reflect_across_x (p : point) : point :=
  (p.1, -p.2)

def P : point := (3, -5)

def Q : point := (5, -2)

def R : point := (5, -5)

theorem final_point_P_after_transformations : reflect_across_x (rotate_90_clockwise P) = (-5, 3) :=
by 
  sorry

end NUMINAMATH_GPT_final_point_P_after_transformations_l327_32787


namespace NUMINAMATH_GPT_peter_remaining_money_l327_32789

def initial_amount : Float := 500.0 
def sales_tax : Float := 0.05
def discount : Float := 0.10

def calculate_cost_with_tax (price_per_kilo: Float) (quantity: Float) (tax_rate: Float) : Float :=
  quantity * price_per_kilo * (1 + tax_rate)

def calculate_cost_with_discount (price_per_kilo: Float) (quantity: Float) (discount_rate: Float) : Float :=
  quantity * price_per_kilo * (1 - discount_rate)

def total_first_trip : Float :=
  calculate_cost_with_tax 2.0 6 sales_tax +
  calculate_cost_with_tax 3.0 9 sales_tax +
  calculate_cost_with_tax 4.0 5 sales_tax +
  calculate_cost_with_tax 5.0 3 sales_tax +
  calculate_cost_with_tax 3.50 2 sales_tax +
  calculate_cost_with_tax 4.25 7 sales_tax +
  calculate_cost_with_tax 6.0 4 sales_tax +
  calculate_cost_with_tax 5.50 8 sales_tax

def total_second_trip : Float :=
  calculate_cost_with_discount 1.50 2 discount +
  calculate_cost_with_discount 2.75 5 discount

def remaining_money (initial: Float) (first_trip: Float) (second_trip: Float) : Float :=
  initial - first_trip - second_trip

theorem peter_remaining_money : remaining_money initial_amount total_first_trip total_second_trip = 297.24 := 
  by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_peter_remaining_money_l327_32789


namespace NUMINAMATH_GPT_computer_price_increase_l327_32794

theorem computer_price_increase
  (P : ℝ)
  (h1 : 1.30 * P = 351) :
  (P + 1.30 * P) / P = 2.3 := by
  sorry

end NUMINAMATH_GPT_computer_price_increase_l327_32794


namespace NUMINAMATH_GPT_set_of_integers_between_10_and_16_l327_32778

theorem set_of_integers_between_10_and_16 :
  {x : ℤ | 10 < x ∧ x < 16} = {11, 12, 13, 14, 15} :=
by
  sorry

end NUMINAMATH_GPT_set_of_integers_between_10_and_16_l327_32778


namespace NUMINAMATH_GPT_fraction_power_calc_l327_32765

theorem fraction_power_calc : 
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
sorry

end NUMINAMATH_GPT_fraction_power_calc_l327_32765


namespace NUMINAMATH_GPT_smallest_n_digit_sum_l327_32716

theorem smallest_n_digit_sum :
  ∃ n : ℕ, (∃ (arrangements : ℕ), arrangements > 1000000 ∧ arrangements = (1/2 * ((n + 1) * (n + 2)))) ∧ (1 + n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + n % 10 = 9) :=
sorry

end NUMINAMATH_GPT_smallest_n_digit_sum_l327_32716


namespace NUMINAMATH_GPT_emily_eggs_collected_l327_32706

theorem emily_eggs_collected :
  let number_of_baskets := 1525
  let eggs_per_basket := 37.5
  let total_eggs := number_of_baskets * eggs_per_basket
  total_eggs = 57187.5 :=
by
  sorry

end NUMINAMATH_GPT_emily_eggs_collected_l327_32706


namespace NUMINAMATH_GPT_min_x2_y2_z2_l327_32734

theorem min_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 22 :=
by
  sorry

end NUMINAMATH_GPT_min_x2_y2_z2_l327_32734


namespace NUMINAMATH_GPT_mary_cut_roses_l327_32754

-- Definitions from conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- The theorem to prove
theorem mary_cut_roses : (final_roses - initial_roses) = 10 :=
by
  sorry

end NUMINAMATH_GPT_mary_cut_roses_l327_32754


namespace NUMINAMATH_GPT_probability_all_red_or_all_white_l327_32724

theorem probability_all_red_or_all_white :
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 6
  let total_marbles := red_marbles + white_marbles + blue_marbles
  let probability_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let probability_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  (probability_red + probability_white) = (14 / 455) :=
by
  sorry

end NUMINAMATH_GPT_probability_all_red_or_all_white_l327_32724


namespace NUMINAMATH_GPT_solution_set_of_inequality_l327_32774

variable {R : Type*} [LinearOrder R] [OrderedAddCommGroup R]

def odd_function (f : R → R) := ∀ x, f (-x) = -f x

def monotonic_increasing_on (f : R → R) (s : Set R) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : odd_function f)
  (h_mono_inc : monotonic_increasing_on f (Set.Ioi 0))
  (h_f_neg1 : f (-1) = 2) : 
  {x : ℝ | 0 < x ∧ f (x-1) + 2 ≤ 0 } = Set.Ioc 1 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l327_32774


namespace NUMINAMATH_GPT_paintable_fence_l327_32738

theorem paintable_fence :
  ∃ h t u : ℕ,  h > 1 ∧ t > 1 ∧ u > 1 ∧ 
  (∀ n, 4 + (n * h) ≠ 5 + (m * (2 * t))) ∧
  (∀ n, 4 + (n * h) ≠ 6 + (l * (3 * u))) ∧ 
  (∀ m l, 5 + (m * (2 * t)) ≠ 6 + (l * (3 * u))) ∧
  (100 * h + 20 * t + 2 * u = 390) :=
by 
  sorry

end NUMINAMATH_GPT_paintable_fence_l327_32738


namespace NUMINAMATH_GPT_no_integer_solutions_l327_32700

theorem no_integer_solutions (n : ℕ) (h : 2 ≤ n) :
  ¬ ∃ x y z : ℤ, x^2 + y^2 = z^n :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l327_32700


namespace NUMINAMATH_GPT_range_of_m_l327_32735

/-- Given the conditions:
- \( \left|1 - \frac{x - 2}{3}\right| \leq 2 \)
- \( x^2 - 2x + 1 - m^2 \leq 0 \) where \( m > 0 \)
- \( \neg \left( \left|1 - \frac{x - 2}{3}\right| \leq 2 \right) \) is a necessary but not sufficient condition for \( x^2 - 2x + 1 - m^2 \leq 0 \)

Prove that the range of \( m \) is \( m \geq 10 \).
-/
theorem range_of_m (m : ℝ) (x : ℝ)
  (h1 : ∀ x, ¬(abs (1 - (x - 2) / 3) ≤ 2) → x < -1 ∨ x > 11)
  (h2 : ∀ x, ∀ m > 0, x^2 - 2 * x + 1 - m^2 ≤ 0)
  : m ≥ 10 :=
sorry

end NUMINAMATH_GPT_range_of_m_l327_32735


namespace NUMINAMATH_GPT_monotonicity_f_on_interval_l327_32795

def f (x : ℝ) : ℝ := |x + 2|

theorem monotonicity_f_on_interval :
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -4 → x2 < -4 → f x1 ≥ f x2 :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_f_on_interval_l327_32795


namespace NUMINAMATH_GPT_range_of_y_div_x_l327_32736

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + y^2 + 4*x + 3 = 0) :
  - (Real.sqrt 3) / 3 <= y / x ∧ y / x <= (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_range_of_y_div_x_l327_32736


namespace NUMINAMATH_GPT_suitable_sampling_method_l327_32708

-- Conditions given
def num_products : ℕ := 40
def num_top_quality : ℕ := 10
def num_second_quality : ℕ := 25
def num_defective : ℕ := 5
def draw_count : ℕ := 8

-- Possible sampling methods
inductive SamplingMethod
| DrawingLots : SamplingMethod
| RandomNumberTable : SamplingMethod
| Systematic : SamplingMethod
| Stratified : SamplingMethod

-- Problem statement (to be proved)
theorem suitable_sampling_method : 
  (num_products = 40) ∧ 
  (num_top_quality = 10) ∧ 
  (num_second_quality = 25) ∧ 
  (num_defective = 5) ∧ 
  (draw_count = 8) → 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_GPT_suitable_sampling_method_l327_32708


namespace NUMINAMATH_GPT_hallway_width_equals_four_l327_32761

-- Define the conditions: dimensions of the areas and total installed area.
def centralAreaLength : ℝ := 10
def centralAreaWidth : ℝ := 10
def centralArea : ℝ := centralAreaLength * centralAreaWidth

def totalInstalledArea : ℝ := 124
def hallwayLength : ℝ := 6

-- Total area minus central area's area yields hallway's area
def hallwayArea : ℝ := totalInstalledArea - centralArea

-- Statement to prove: the width of the hallway given its area and length.
theorem hallway_width_equals_four :
  (hallwayArea / hallwayLength) = 4 := 
by
  sorry

end NUMINAMATH_GPT_hallway_width_equals_four_l327_32761


namespace NUMINAMATH_GPT_minimum_shots_required_l327_32740

noncomputable def minimum_shots_to_sink_boat : ℕ := 4000

-- Definitions for the problem conditions.
structure Boat :=
(square_side : ℕ)
(base1 : ℕ)
(base2 : ℕ)
(rotatable : Bool)

def boat : Boat := { square_side := 1, base1 := 1, base2 := 3, rotatable := true }

def grid_size : ℕ := 100

def shot_covers_triangular_half : Prop := sorry -- Assumption: Define this appropriately

-- Problem statement in Lean 4
theorem minimum_shots_required (boat_within_grid : Bool) : 
  Boat → grid_size = 100 → boat_within_grid → minimum_shots_to_sink_boat = 4000 :=
by
  -- Here you would do the full proof which we assume is "sorry" for now
  sorry

end NUMINAMATH_GPT_minimum_shots_required_l327_32740


namespace NUMINAMATH_GPT_point_slope_form_of_perpendicular_line_l327_32739

theorem point_slope_form_of_perpendicular_line :
  ∀ (l1 l2 : ℝ → ℝ) (P : ℝ × ℝ),
    (l2 x = x + 1) →
    (P = (2, 1)) →
    (∀ x, l2 x = -1 * l1 x) →
    (∀ x, l1 x = -x + 3) :=
by
  intros l1 l2 P h1 h2 h3
  sorry

end NUMINAMATH_GPT_point_slope_form_of_perpendicular_line_l327_32739


namespace NUMINAMATH_GPT_least_number_to_add_l327_32756

theorem least_number_to_add (n d : ℕ) (h : n = 1024) (h_d : d = 25) :
  ∃ x : ℕ, (n + x) % d = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_GPT_least_number_to_add_l327_32756


namespace NUMINAMATH_GPT_locus_of_centers_l327_32721

set_option pp.notation false -- To ensure nicer looking lean code.

-- Define conditions for circles C_3 and C_4
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C4 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Statement to prove the locus of centers satisfies the equation
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)) →
  (a^2 + 18 * b^2 - 6 * a - 440 = 0) :=
by
  sorry -- Proof not required as per the instructions

end NUMINAMATH_GPT_locus_of_centers_l327_32721


namespace NUMINAMATH_GPT_solve_z_l327_32779

noncomputable def complex_equation (z : ℂ) := (1 + 3 * Complex.I) * z = Complex.I - 3

theorem solve_z (z : ℂ) (h : complex_equation z) : z = Complex.I :=
by
  sorry

end NUMINAMATH_GPT_solve_z_l327_32779


namespace NUMINAMATH_GPT_back_parking_lot_filled_fraction_l327_32768

theorem back_parking_lot_filled_fraction
    (front_spaces : ℕ) (back_spaces : ℕ) (cars_parked : ℕ) (spaces_available : ℕ)
    (h1 : front_spaces = 52)
    (h2 : back_spaces = 38)
    (h3 : cars_parked = 39)
    (h4 : spaces_available = 32) :
    (back_spaces - (front_spaces + back_spaces - cars_parked - spaces_available)) / back_spaces = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_back_parking_lot_filled_fraction_l327_32768


namespace NUMINAMATH_GPT_double_root_values_l327_32780

theorem double_root_values (c : ℝ) :
  (∃ a : ℝ, (a^5 - 5 * a + c = 0) ∧ (5 * a^4 - 5 = 0)) ↔ (c = 4 ∨ c = -4) :=
by
  sorry

end NUMINAMATH_GPT_double_root_values_l327_32780


namespace NUMINAMATH_GPT_typists_initial_group_l327_32749

theorem typists_initial_group
  (T : ℕ) 
  (h1 : 0 < T) 
  (h2 : T * (240 / 40 * 20) = 2400) : T = 10 :=
by
  sorry

end NUMINAMATH_GPT_typists_initial_group_l327_32749


namespace NUMINAMATH_GPT_alex_loan_comparison_l327_32790

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem alex_loan_comparison :
  let P : ℝ := 15000
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let n : ℕ := 12
  let t1_10 : ℝ := 10
  let t1_5 : ℝ := 5
  let t2 : ℝ := 15
  let owed_after_10 := compound_interest P r1 n t1_10
  let payment_after_10 := owed_after_10 / 2
  let remaining_after_10 := owed_after_10 / 2
  let owed_after_15 := compound_interest remaining_after_10 r1 n t1_5
  let total_payment_option1 := payment_after_10 + owed_after_15
  let total_payment_option2 := simple_interest P r2 t2
  total_payment_option1 - total_payment_option2 = 4163 :=
by
  sorry

end NUMINAMATH_GPT_alex_loan_comparison_l327_32790


namespace NUMINAMATH_GPT_average_value_eq_l327_32791

variable (x : ℝ)

theorem average_value_eq :
  ( -4 * x + 0 + 4 * x + 12 * x + 20 * x ) / 5 = 6.4 * x :=
by
  sorry

end NUMINAMATH_GPT_average_value_eq_l327_32791


namespace NUMINAMATH_GPT_john_money_left_l327_32711

variable (q : ℝ) 

def cost_soda := q
def cost_medium_pizza := 3 * q
def cost_small_pizza := 2 * q

def total_cost := 4 * cost_soda q + 2 * cost_medium_pizza q + 3 * cost_small_pizza q

theorem john_money_left (h : total_cost q = 16 * q) : 50 - total_cost q = 50 - 16 * q := by
  simp [total_cost, cost_soda, cost_medium_pizza, cost_small_pizza]
  sorry

end NUMINAMATH_GPT_john_money_left_l327_32711


namespace NUMINAMATH_GPT_complex_power_identity_l327_32730

theorem complex_power_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end NUMINAMATH_GPT_complex_power_identity_l327_32730


namespace NUMINAMATH_GPT_smaller_circle_circumference_l327_32728

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end NUMINAMATH_GPT_smaller_circle_circumference_l327_32728


namespace NUMINAMATH_GPT_hundredth_odd_integer_l327_32792

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end NUMINAMATH_GPT_hundredth_odd_integer_l327_32792


namespace NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_l327_32725

theorem halfway_between_one_eighth_and_one_third : (1/8 + 1/3) / 2 = 11/48 :=
by
  sorry

end NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_l327_32725


namespace NUMINAMATH_GPT_quadratic_solution_l327_32776

theorem quadratic_solution (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l327_32776


namespace NUMINAMATH_GPT_train_passing_platform_time_l327_32752

theorem train_passing_platform_time :
  (500 : ℝ) / (50 : ℝ) > 0 →
  (500 : ℝ) + (500 : ℝ) / ((500 : ℝ) / (50 : ℝ)) = 100 := by
  sorry

end NUMINAMATH_GPT_train_passing_platform_time_l327_32752


namespace NUMINAMATH_GPT_vector_addition_correct_l327_32712

def vec1 : ℤ × ℤ := (5, -9)
def vec2 : ℤ × ℤ := (-8, 14)
def vec_sum (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_addition_correct :
  vec_sum vec1 vec2 = (-3, 5) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_vector_addition_correct_l327_32712


namespace NUMINAMATH_GPT_pearls_problem_l327_32705

theorem pearls_problem :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n = 54) ∧ (n % 9 = 0) :=
by sorry

end NUMINAMATH_GPT_pearls_problem_l327_32705


namespace NUMINAMATH_GPT_painting_time_equation_l327_32745

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end NUMINAMATH_GPT_painting_time_equation_l327_32745


namespace NUMINAMATH_GPT_find_f_value_l327_32722

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_value_l327_32722


namespace NUMINAMATH_GPT_inequality_holds_for_m_l327_32799

theorem inequality_holds_for_m (m : ℝ) :
  (-2 : ℝ) ≤ m ∧ m ≤ (3 : ℝ) ↔ ∀ x : ℝ, x < -1 →
    (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0 :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_m_l327_32799


namespace NUMINAMATH_GPT_scrooge_share_l327_32742

def leftover_pie : ℚ := 8 / 9

def share_each (x : ℚ) : Prop :=
  2 * x + 3 * x = leftover_pie

theorem scrooge_share (x : ℚ):
  share_each x → (2 * x = 16 / 45) := by
  sorry

end NUMINAMATH_GPT_scrooge_share_l327_32742


namespace NUMINAMATH_GPT_log_one_eq_zero_l327_32719

theorem log_one_eq_zero : Real.log 1 = 0 := 
by
  sorry

end NUMINAMATH_GPT_log_one_eq_zero_l327_32719


namespace NUMINAMATH_GPT_card_drawing_ways_l327_32793

theorem card_drawing_ways :
  (30 * 20 = 600) :=
by
  sorry

end NUMINAMATH_GPT_card_drawing_ways_l327_32793


namespace NUMINAMATH_GPT_extra_kilometers_per_hour_l327_32703

theorem extra_kilometers_per_hour (S a : ℝ) (h : a > 2) : 
  (S / (a - 2)) - (S / a) = (S / (a - 2)) - (S / a) :=
by sorry

end NUMINAMATH_GPT_extra_kilometers_per_hour_l327_32703


namespace NUMINAMATH_GPT_gcd_of_sum_of_cubes_and_increment_l327_32726

theorem gcd_of_sum_of_cubes_and_increment {n : ℕ} (h : n > 3) : Nat.gcd (n^3 + 27) (n + 4) = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_of_sum_of_cubes_and_increment_l327_32726


namespace NUMINAMATH_GPT_find_line_equation_l327_32741

theorem find_line_equation : 
  ∃ c : ℝ, (∀ x y : ℝ, 2*x + 4*y + c = 0 ↔ x + 2*y - 8 = 0) ∧ (2*2 + 4*3 + c = 0) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l327_32741


namespace NUMINAMATH_GPT_find_z_coordinate_of_point_on_line_l327_32759

theorem find_z_coordinate_of_point_on_line (x1 y1 z1 x2 y2 z2 x_target : ℝ) 
(h1 : x1 = 1) (h2 : y1 = 3) (h3 : z1 = 2) 
(h4 : x2 = 4) (h5 : y2 = 4) (h6 : z2 = -1)
(h_target : x_target = 7) : 
∃ z_target : ℝ, z_target = -4 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_z_coordinate_of_point_on_line_l327_32759


namespace NUMINAMATH_GPT_find_y_interval_l327_32702

open Real

theorem find_y_interval {y : ℝ}
  (hy_nonzero : y ≠ 0)
  (h_denominator_nonzero : 1 + 3 * y - 4 * y^2 ≠ 0) :
  (y^2 + 9 * y - 1 = 0) →
  (∀ y, y ∈ Set.Icc (-(9 + sqrt 85)/2) (-(9 - sqrt 85)/2) \ {y | y = 0 ∨ 1 + 3 * y - 4 * y^2 = 0} ↔
  (y * (3 - 3 * y))/(1 + 3 * y - 4 * y^2) ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_find_y_interval_l327_32702


namespace NUMINAMATH_GPT_total_tweets_is_correct_l327_32762

-- Define the conditions of Polly's tweeting behavior and durations
def happy_tweets := 18
def hungry_tweets := 4
def mirror_tweets := 45
def duration := 20

-- Define the total tweets calculation
def total_tweets := duration * happy_tweets + duration * hungry_tweets + duration * mirror_tweets

-- Prove that the total number of tweets is 1340
theorem total_tweets_is_correct : total_tweets = 1340 := by
  sorry

end NUMINAMATH_GPT_total_tweets_is_correct_l327_32762


namespace NUMINAMATH_GPT_area_rectangle_around_right_triangle_l327_32772

theorem area_rectangle_around_right_triangle (AB BC : ℕ) (hAB : AB = 5) (hBC : BC = 6) :
    let ADE_area := AB * BC
    ADE_area = 30 := by
  sorry

end NUMINAMATH_GPT_area_rectangle_around_right_triangle_l327_32772


namespace NUMINAMATH_GPT_distinct_ordered_pairs_proof_l327_32796

def num_distinct_ordered_pairs_satisfying_reciprocal_sum : ℕ :=
  List.length [
    (7, 42), (8, 24), (9, 18), (10, 15), 
    (12, 12), (15, 10), (18, 9), (24, 8), 
    (42, 7)
  ]

theorem distinct_ordered_pairs_proof : num_distinct_ordered_pairs_satisfying_reciprocal_sum = 9 := by
  sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_proof_l327_32796


namespace NUMINAMATH_GPT_partition_diff_l327_32732

theorem partition_diff {A : Type} (S : Finset ℕ) (S_card : S.card = 67)
  (P : Finset (Finset ℕ)) (P_card : P.card = 4) :
  ∃ (U : Finset ℕ) (hU : U ∈ P), ∃ (a b c : ℕ) (ha : a ∈ U) (hb : b ∈ U) (hc : c ∈ U),
  a = b - c ∧ (1 ≤ a ∧ a ≤ 67) :=
by sorry

end NUMINAMATH_GPT_partition_diff_l327_32732


namespace NUMINAMATH_GPT_volleyball_team_selection_l327_32798

theorem volleyball_team_selection (total_players starting_players : ℕ) (libero : ℕ) : 
  total_players = 12 → 
  starting_players = 6 → 
  libero = 1 →
  (∃ (ways : ℕ), ways = 5544) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_volleyball_team_selection_l327_32798


namespace NUMINAMATH_GPT_purchase_price_mobile_l327_32747

-- Definitions of the given conditions
def purchase_price_refrigerator : ℝ := 15000
def loss_percent_refrigerator : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10
def overall_profit : ℝ := 50

-- Defining the statement to prove
theorem purchase_price_mobile (P : ℝ)
  (h1 : purchase_price_refrigerator = 15000)
  (h2 : loss_percent_refrigerator = 0.05)
  (h3 : profit_percent_mobile = 0.10)
  (h4 : overall_profit = 50) :
  (15000 * (1 - 0.05) + P * (1 + 0.10)) - (15000 + P) = 50 → P = 8000 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_purchase_price_mobile_l327_32747


namespace NUMINAMATH_GPT_volume_of_right_prism_correct_l327_32748

variables {α β l : ℝ}

noncomputable def volume_of_right_prism (α β l : ℝ) : ℝ :=
  (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α))

theorem volume_of_right_prism_correct
  (α β l : ℝ)
  (α_gt0 : 0 < α) (α_lt90 : α < Real.pi / 2)
  (l_pos : 0 < l)
  : volume_of_right_prism α β l = (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α)) :=
sorry

end NUMINAMATH_GPT_volume_of_right_prism_correct_l327_32748


namespace NUMINAMATH_GPT_shorter_piece_is_20_l327_32714

def shorter_piece_length (total_length : ℕ) (ratio : ℚ) (shorter_piece : ℕ) : Prop :=
    shorter_piece * 7 = 2 * (total_length - shorter_piece)

theorem shorter_piece_is_20 : ∀ (total_length : ℕ) (shorter_piece : ℕ), 
    total_length = 90 ∧
    shorter_piece_length total_length (2/7 : ℚ) shorter_piece ->
    shorter_piece = 20 :=
by
  intro total_length shorter_piece
  intro h
  have h_total_length : total_length = 90 := h.1
  have h_equation : shorter_piece_length total_length (2/7 : ℚ) shorter_piece := h.2
  sorry

end NUMINAMATH_GPT_shorter_piece_is_20_l327_32714


namespace NUMINAMATH_GPT_total_cost_backpacks_l327_32727

theorem total_cost_backpacks:
  let original_price := 20.00
  let discount := 0.20
  let monogram_cost := 12.00
  let coupon := 5.00
  let state_tax : List Real := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discounted_price := original_price * (1 - discount)
  let pre_tax_cost := discounted_price + monogram_cost
  let final_costs := state_tax.map (λ tax_rate => pre_tax_cost * (1 + tax_rate))
  let total_cost_before_coupon := final_costs.sum
  total_cost_before_coupon - coupon = 143.61 := by
    sorry

end NUMINAMATH_GPT_total_cost_backpacks_l327_32727


namespace NUMINAMATH_GPT_find_multiple_of_games_l327_32760

-- declaring the number of video games each person has
def Tory_videos := 6
def Theresa_videos := 11
def Julia_videos := Tory_videos / 3

-- declaring the multiple we need to find
def multiple_of_games := Theresa_videos - Julia_videos * 5

-- Theorem stating the problem
theorem find_multiple_of_games : ∃ m : ℕ, Julia_videos * m + 5 = Theresa_videos :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_games_l327_32760


namespace NUMINAMATH_GPT_number_of_buses_l327_32775

theorem number_of_buses (total_supervisors : ℕ) (supervisors_per_bus : ℕ) (h1 : total_supervisors = 21) (h2 : supervisors_per_bus = 3) : total_supervisors / supervisors_per_bus = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_buses_l327_32775


namespace NUMINAMATH_GPT_Jill_age_l327_32757

variable (J R : ℕ) -- representing Jill's current age and Roger's current age

theorem Jill_age :
  (R = 2 * J + 5) →
  (R - J = 25) →
  J = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Jill_age_l327_32757


namespace NUMINAMATH_GPT_point_P_distance_to_y_axis_l327_32771

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- The distance from point P to the y-axis
def distance_to_y_axis (pt : ℝ × ℝ) : ℝ :=
  abs pt.1

-- Statement to prove
theorem point_P_distance_to_y_axis :
  distance_to_y_axis point_P = 2 :=
by
  sorry

end NUMINAMATH_GPT_point_P_distance_to_y_axis_l327_32771


namespace NUMINAMATH_GPT_seven_solutions_l327_32713

theorem seven_solutions: ∃ (pairs : List (ℕ × ℕ)), 
  (∀ (x y : ℕ), (x < y) → ((1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 2007) ↔ (x, y) ∈ pairs) 
  ∧ pairs.length = 7 :=
sorry

end NUMINAMATH_GPT_seven_solutions_l327_32713


namespace NUMINAMATH_GPT_g_inv_undefined_at_1_l327_32737

noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (5 * x - 3) / (x - 1)

theorem g_inv_undefined_at_1 : ∀ x : ℝ, (g_inv x) = g_inv 1 → x = 1 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_g_inv_undefined_at_1_l327_32737


namespace NUMINAMATH_GPT_probability_participation_on_both_days_l327_32773

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_participation_on_both_days_l327_32773


namespace NUMINAMATH_GPT_prime_numbers_in_list_l327_32755

noncomputable def list_numbers : ℕ → ℕ
| 0       => 43
| (n + 1) => 43 * ((10 ^ (2 * n + 2) - 1) / 99) 

theorem prime_numbers_in_list : ∃ n:ℕ, (∀ m, (m > n) → ¬ Prime (list_numbers m)) ∧ Prime (list_numbers 0) := 
by
  sorry

end NUMINAMATH_GPT_prime_numbers_in_list_l327_32755


namespace NUMINAMATH_GPT_min_value_l327_32767

theorem min_value (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h_sum : x1 + x2 = 1) :
  ∃ m, (∀ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = 1 → (3 * x1 / x2 + 1 / (x1 * x2)) ≥ m) ∧ m = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l327_32767


namespace NUMINAMATH_GPT_part_I_part_II_l327_32731

noncomputable def f (x : ℝ) := (Real.sin x) * (Real.cos x) + (Real.sin x)^2

-- Part I: Prove that f(π / 4) = 1
theorem part_I : f (Real.pi / 4) = 1 := sorry

-- Part II: Prove that the maximum value of f(x) for x ∈ [0, π / 2] is (√2 + 1) / 2
theorem part_II : ∃ x ∈ Set.Icc 0 (Real.pi / 2), (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧ f x = (Real.sqrt 2 + 1) / 2 := sorry

end NUMINAMATH_GPT_part_I_part_II_l327_32731


namespace NUMINAMATH_GPT_cost_per_sqft_is_3_l327_32717

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def extra_cost_per_sqft : ℝ := 1
def total_cost : ℝ := 4800

theorem cost_per_sqft_is_3
    (area : ℝ := deck_length * deck_width)
    (sealant_cost : ℝ := area * extra_cost_per_sqft)
    (deck_construction_cost : ℝ := total_cost - sealant_cost) :
    deck_construction_cost / area = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_sqft_is_3_l327_32717


namespace NUMINAMATH_GPT_xiao_wang_conjecture_incorrect_l327_32797

theorem xiao_wang_conjecture_incorrect : ∃ n : ℕ, n > 0 ∧ (n^2 - 8 * n + 7 > 0) := by
  sorry

end NUMINAMATH_GPT_xiao_wang_conjecture_incorrect_l327_32797


namespace NUMINAMATH_GPT_tan_of_angle_subtraction_l327_32701

theorem tan_of_angle_subtraction (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 7) : Real.tan a = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_angle_subtraction_l327_32701


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l327_32783

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l327_32783
