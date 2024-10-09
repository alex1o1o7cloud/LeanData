import Mathlib

namespace initial_kittens_l2357_235724

-- Define the number of kittens given to Jessica and Sara, and the number of kittens currently Tim has.
def kittens_given_to_Jessica : ℕ := 3
def kittens_given_to_Sara : ℕ := 6
def kittens_left_with_Tim : ℕ := 9

-- Define the theorem to prove the initial number of kittens Tim had.
theorem initial_kittens (kittens_given_to_Jessica kittens_given_to_Sara kittens_left_with_Tim : ℕ) 
    (h1 : kittens_given_to_Jessica = 3)
    (h2 : kittens_given_to_Sara = 6)
    (h3 : kittens_left_with_Tim = 9) :
    (kittens_given_to_Jessica + kittens_given_to_Sara + kittens_left_with_Tim) = 18 := 
    sorry

end initial_kittens_l2357_235724


namespace price_difference_pc_sm_l2357_235761

-- Definitions based on given conditions
def S : ℕ := 300
def x : ℕ := sorry -- This is what we are trying to find
def PC : ℕ := S + x
def AT : ℕ := S + PC
def total_cost : ℕ := S + PC + AT

-- Theorem to be proved
theorem price_difference_pc_sm (h : total_cost = 2200) : x = 500 :=
by
  -- We would prove the theorem here
  sorry

end price_difference_pc_sm_l2357_235761


namespace value_of_a8_l2357_235758

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, ∃ d : α, a (n + 1) = a n + d

variable {a : ℕ → ℝ}

axiom seq_is_arithmetic : arithmetic_sequence a

axiom initial_condition :
  a 1 + 3 * a 8 + a 15 = 120

axiom arithmetic_property :
  a 1 + a 15 = 2 * a 8

theorem value_of_a8 : a 8 = 24 :=
by {
  sorry
}

end value_of_a8_l2357_235758


namespace simplify_expression_l2357_235710

-- Define the hypotheses and the expression.
variables (x : ℚ)
def expr := (1 + 1 / x) * (1 - 2 / (x + 1)) * (1 + 2 / (x - 1))

-- Define the conditions.
def valid_x : Prop := (x ≠ 0) ∧ (x ≠ -1) ∧ (x ≠ 1)

-- State the main theorem.
theorem simplify_expression (h : valid_x x) : expr x = (x + 1) / x := 
sorry

end simplify_expression_l2357_235710


namespace preimage_of_8_is_5_image_of_8_is_64_l2357_235798

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2)

theorem preimage_of_8_is_5 : ∃ x, f x = 8 := by
  use 5
  sorry

theorem image_of_8_is_64 : f 8 = 64 := by
  sorry

end preimage_of_8_is_5_image_of_8_is_64_l2357_235798


namespace find_total_amount_l2357_235738

theorem find_total_amount (x : ℝ) (h₁ : 1.5 * x = 40) : x + 1.5 * x + 0.5 * x = 80.01 :=
by
  sorry

end find_total_amount_l2357_235738


namespace parallel_line_slope_y_intercept_l2357_235748

theorem parallel_line_slope_y_intercept (x y : ℝ) (h : 3 * x - 6 * y = 12) :
  ∃ (m b : ℝ), m = 1 / 2 ∧ b = -2 := 
by { sorry }

end parallel_line_slope_y_intercept_l2357_235748


namespace min_abs_diff_l2357_235731

theorem min_abs_diff (a b c d : ℝ) (h1 : |a - b| = 5) (h2 : |b - c| = 8) (h3 : |c - d| = 10) : 
  ∃ m, m = |a - d| ∧ m = 3 := 
by 
  sorry

end min_abs_diff_l2357_235731


namespace points_3_units_away_from_origin_l2357_235762

theorem points_3_units_away_from_origin (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end points_3_units_away_from_origin_l2357_235762


namespace geometric_sequence_common_ratio_l2357_235739

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (a_mono : ∀ n, a n < a (n+1))
    (a2a5_eq_6 : a 2 * a 5 = 6)
    (a3a4_eq_5 : a 3 + a 4 = 5) 
    (q : ℝ) (hq : ∀ n, a n = a 1 * q ^ (n - 1)) :
    q = 3 / 2 :=
by
    sorry

end geometric_sequence_common_ratio_l2357_235739


namespace arithmetic_sequence_a5_value_l2357_235785

variable {a_n : ℕ → ℝ}

theorem arithmetic_sequence_a5_value
  (h : a_n 2 + a_n 8 = 15 - a_n 5) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l2357_235785


namespace John_surveyed_total_people_l2357_235765

theorem John_surveyed_total_people :
  ∃ P D : ℝ, 
  0 ≤ P ∧ 
  D = 0.868 * P ∧ 
  21 = 0.457 * D ∧ 
  P = 53 :=
by
  sorry

end John_surveyed_total_people_l2357_235765


namespace steak_weight_in_ounces_l2357_235729

-- Definitions from conditions
def pounds : ℕ := 15
def ounces_per_pound : ℕ := 16
def steaks : ℕ := 20

-- The theorem to prove
theorem steak_weight_in_ounces : 
  (pounds * ounces_per_pound) / steaks = 12 := by
  sorry

end steak_weight_in_ounces_l2357_235729


namespace nate_distance_after_resting_l2357_235700

variables (length_of_field total_distance : ℕ)

def distance_before_resting (length_of_field : ℕ) := 4 * length_of_field

def distance_after_resting (total_distance length_of_field : ℕ) : ℕ := 
  total_distance - distance_before_resting length_of_field

theorem nate_distance_after_resting
  (length_of_field_val : length_of_field = 168)
  (total_distance_val : total_distance = 1172) :
  distance_after_resting total_distance length_of_field = 500 :=
by
  -- Proof goes here
  sorry

end nate_distance_after_resting_l2357_235700


namespace find_constants_l2357_235740

theorem find_constants (a b c : ℝ) (h_neq_0_a : a ≠ 0) (h_neq_0_b : b ≠ 0) 
(h_neq_0_c : c ≠ 0) 
(h_eq1 : a * b = 3 * (a + b)) 
(h_eq2 : b * c = 4 * (b + c)) 
(h_eq3 : a * c = 5 * (a + c)) : 
a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 := 
  sorry

end find_constants_l2357_235740


namespace intersection_M_N_l2357_235783

-- Define the sets M and N based on given conditions
def M : Set ℝ := { x : ℝ | x^2 < 4 }
def N : Set ℝ := { x : ℝ | x < 1 }

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l2357_235783


namespace algebraic_expression_equivalence_l2357_235707

theorem algebraic_expression_equivalence (x : ℝ) : 
  x^2 - 6*x + 10 = (x - 3)^2 + 1 := 
by 
  sorry

end algebraic_expression_equivalence_l2357_235707


namespace dima_always_wins_l2357_235723

theorem dima_always_wins (n : ℕ) (P : Prop) : 
  (∀ (gosha dima : ℕ → Prop), 
    (∀ k : ℕ, k < n → (gosha k ∨ dima k))
    ∧ (∀ i : ℕ, i < 14 → (gosha i ∨ dima i))
    ∧ (∃ j : ℕ, j ≤ n ∧ (∃ k ≤ j + 7, dima k))
    ∧ (∃ l : ℕ, l ≤ 14 ∧ (∃ m ≤ l + 7, dima m))
    → P) → P := sorry

end dima_always_wins_l2357_235723


namespace percent_pension_participation_l2357_235751

-- Define the conditions provided
def total_first_shift_members : ℕ := 60
def total_second_shift_members : ℕ := 50
def total_third_shift_members : ℕ := 40

def first_shift_pension_percentage : ℚ := 20 / 100
def second_shift_pension_percentage : ℚ := 40 / 100
def third_shift_pension_percentage : ℚ := 10 / 100

-- Calculate participation in the pension program for each shift
def first_shift_pension_members := total_first_shift_members * first_shift_pension_percentage
def second_shift_pension_members := total_second_shift_members * second_shift_pension_percentage
def third_shift_pension_members := total_third_shift_members * third_shift_pension_percentage

-- Calculate total participation in the pension program and total number of workers
def total_pension_members := first_shift_pension_members + second_shift_pension_members + third_shift_pension_members
def total_workers := total_first_shift_members + total_second_shift_members + total_third_shift_members

-- Lean proof statement
theorem percent_pension_participation : (total_pension_members / total_workers * 100) = 24 := by
  sorry

end percent_pension_participation_l2357_235751


namespace constant_for_odd_m_l2357_235797

theorem constant_for_odd_m (constant : ℝ) (f : ℕ → ℝ)
  (h1 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k + 1) → f m = constant * m)
  (h2 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k) → f m = (1/2 : ℝ) * m)
  (h3 : f 5 * f 6 = 15) : constant = 1 :=
by
  sorry

end constant_for_odd_m_l2357_235797


namespace range_of_g_l2357_235778

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8 - 2 * x ^ 2) / (x + 4)

theorem range_of_g : 
  (∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ y = (3 * x + 8 - 2 * x^2) / (x + 4)) :=
by
  sorry

end range_of_g_l2357_235778


namespace parking_lot_wheels_l2357_235719

noncomputable def total_car_wheels (guest_cars : Nat) (guest_car_wheels : Nat) (parent_cars : Nat) (parent_car_wheels : Nat) : Nat :=
  guest_cars * guest_car_wheels + parent_cars * parent_car_wheels

theorem parking_lot_wheels :
  total_car_wheels 10 4 2 4 = 48 :=
by
  sorry

end parking_lot_wheels_l2357_235719


namespace system_no_solution_l2357_235725

theorem system_no_solution (n : ℝ) :
  ∃ x y z : ℝ, (n * x + y = 1) ∧ (1 / 2 * n * y + z = 1) ∧ (x + 1 / 2 * n * z = 2) ↔ n = -1 := 
sorry

end system_no_solution_l2357_235725


namespace not_perfect_square_l2357_235788

theorem not_perfect_square : ¬ ∃ x : ℝ, x^2 = 7^2025 := by
  sorry

end not_perfect_square_l2357_235788


namespace jills_daily_earnings_first_month_l2357_235728

-- Definitions based on conditions
variable (x : ℕ) -- daily earnings in the first month
def total_earnings_first_month := 30 * x
def total_earnings_second_month := 30 * (2 * x)
def total_earnings_third_month := 15 * (2 * x)
def total_earnings_three_months := total_earnings_first_month x + total_earnings_second_month x + total_earnings_third_month x

-- The theorem we need to prove
theorem jills_daily_earnings_first_month
  (h : total_earnings_three_months x = 1200) : x = 10 :=
sorry

end jills_daily_earnings_first_month_l2357_235728


namespace algebraic_expression_identity_l2357_235712

theorem algebraic_expression_identity (a b x : ℕ) (h : x * 3 * a * b = 3 * a * a * b) : x = a :=
sorry

end algebraic_expression_identity_l2357_235712


namespace rectangular_solid_volume_l2357_235703

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : a * b * c = 60 :=
by
  sorry

end rectangular_solid_volume_l2357_235703


namespace ratio_of_x_to_y_l2357_235705

theorem ratio_of_x_to_y (x y : ℝ) (R : ℝ) (h1 : x = R * y) (h2 : x - y = 0.909090909090909 * x) : R = 11 := by
  sorry

end ratio_of_x_to_y_l2357_235705


namespace subtraction_of_negatives_l2357_235792

theorem subtraction_of_negatives : (-7) - (-5) = -2 := 
by {
  -- sorry replaces the actual proof steps.
  sorry
}

end subtraction_of_negatives_l2357_235792


namespace distance_between_cities_l2357_235769

variable (a b : Nat)

theorem distance_between_cities :
  (a = (10 * a + b) - (10 * b + a)) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → 10 * a + b = 98 := by
  sorry

end distance_between_cities_l2357_235769


namespace same_terminal_side_angle_l2357_235767

theorem same_terminal_side_angle (k : ℤ) : 
  0 ≤ (k * 360 - 35) ∧ (k * 360 - 35) < 360 → (k * 360 - 35) = 325 :=
by
  sorry

end same_terminal_side_angle_l2357_235767


namespace general_term_formula_exponential_seq_l2357_235750

variable (n : ℕ)

def exponential_sequence (a1 r : ℕ) (n : ℕ) : ℕ := a1 * r^(n-1)

theorem general_term_formula_exponential_seq :
  exponential_sequence 2 3 n = 2 * 3^(n-1) :=
by
  sorry

end general_term_formula_exponential_seq_l2357_235750


namespace box_volume_is_correct_l2357_235713

noncomputable def box_volume (length width cut_side : ℝ) : ℝ :=
  (length - 2 * cut_side) * (width - 2 * cut_side) * cut_side

theorem box_volume_is_correct : box_volume 48 36 5 = 9880 := by
  sorry

end box_volume_is_correct_l2357_235713


namespace no_nat_m_n_square_diff_2014_l2357_235744

theorem no_nat_m_n_square_diff_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_m_n_square_diff_2014_l2357_235744


namespace log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l2357_235757

theorem log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6 
  (h1 : 5^0.6 > 1)
  (h2 : 0 < 0.6^5 ∧ 0.6^5 < 1)
  (h3 : Real.logb 0.6 5 < 0) :
  Real.logb 0.6 5 < 0.6^5 ∧ 0.6^5 < 5^0.6 :=
sorry

end log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l2357_235757


namespace dad_caught_more_trouts_l2357_235787

-- Definitions based on conditions
def caleb_trouts : ℕ := 2
def dad_trouts : ℕ := 3 * caleb_trouts

-- The proof problem: proving dad caught 4 more trouts than Caleb
theorem dad_caught_more_trouts : dad_trouts = caleb_trouts + 4 :=
by
  sorry

end dad_caught_more_trouts_l2357_235787


namespace expression_C_eq_seventeen_l2357_235796

theorem expression_C_eq_seventeen : (3 + 4 * 5 - 6) = 17 := 
by 
  sorry

end expression_C_eq_seventeen_l2357_235796


namespace jeremy_age_l2357_235772

theorem jeremy_age
  (A J C : ℕ)
  (h1 : A + J + C = 132)
  (h2 : A = 1 / 3 * J)
  (h3 : C = 2 * A) :
  J = 66 :=
sorry

end jeremy_age_l2357_235772


namespace range_of_a_l2357_235743

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x ^ 2 + 2 * x + 1)

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, a * x ^ 2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l2357_235743


namespace sum_of_fifth_powers_l2357_235746

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l2357_235746


namespace intersection_y_axis_parabola_l2357_235717

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end intersection_y_axis_parabola_l2357_235717


namespace city_growth_rate_order_l2357_235779

theorem city_growth_rate_order 
  (Dover Eden Fairview : Type) 
  (highest lowest : Type)
  (h1 : Dover = highest → ¬(Eden = highest) ∧ (Fairview = lowest))
  (h2 : ¬(Dover = highest) ∧ Eden = highest ∧ Fairview = lowest → Eden = highest ∧ Dover = lowest ∧ Fairview = highest)
  (h3 : ¬(Fairview = lowest) → ¬(Eden = highest) ∧ ¬(Dover = highest)) : 
  Eden = highest ∧ Dover = lowest ∧ Fairview = highest ∧ Eden ≠ lowest :=
by
  sorry

end city_growth_rate_order_l2357_235779


namespace geometric_condition_l2357_235704

def Sn (p : ℤ) (n : ℕ) : ℤ := p * 2^n + 2

def an (p : ℤ) (n : ℕ) : ℤ :=
  if n = 1 then Sn p n
  else Sn p n - Sn p (n - 1)

def is_geometric_progression (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ r : ℤ, an p n = an p (n - 1) * r

theorem geometric_condition (p : ℤ) :
  is_geometric_progression p ↔ p = -2 :=
sorry

end geometric_condition_l2357_235704


namespace divisor_of_p_l2357_235722

-- Define the necessary variables and assumptions
variables (p q r s : ℕ)

-- State the conditions
def conditions := gcd p q = 28 ∧ gcd q r = 45 ∧ gcd r s = 63 ∧ 80 < gcd s p ∧ gcd s p < 120 

-- State the proposition to prove: 11 divides p
theorem divisor_of_p (h : conditions p q r s) : 11 ∣ p := 
sorry

end divisor_of_p_l2357_235722


namespace becky_necklaces_count_l2357_235742

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def new_necklaces := 5
def given_away_necklaces := 15

-- Define the final number of necklaces
def final_necklaces (initial : Nat) (broken : Nat) (bought : Nat) (given_away : Nat) : Nat :=
  initial - broken + bought - given_away

-- The theorem stating that after performing the series of operations,
-- Becky should have 37 necklaces.
theorem becky_necklaces_count :
  final_necklaces initial_necklaces broken_necklaces new_necklaces given_away_necklaces = 37 :=
  by
    -- This proof is just a placeholder to ensure the code can be built successfully.
    -- Actual proof logic needs to be filled in to complete the theorem.
    sorry

end becky_necklaces_count_l2357_235742


namespace smallest_possible_X_l2357_235771

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end smallest_possible_X_l2357_235771


namespace max_side_range_of_triangle_l2357_235760

-- Define the requirement on the sides a and b
def side_condition (a b : ℝ) : Prop :=
  |a - 3| + (b - 7)^2 = 0

-- Prove the range of side c
theorem max_side_range_of_triangle (a b c : ℝ) (h : side_condition a b) (hc : c = max a (max b c)) :
  7 ≤ c ∧ c < 10 :=
sorry

end max_side_range_of_triangle_l2357_235760


namespace mike_payments_total_months_l2357_235790

-- Definitions based on conditions
def lower_rate := 295
def higher_rate := 310
def lower_payments := 5
def higher_payments := 7
def total_paid := 3615

-- The statement to prove
theorem mike_payments_total_months : lower_payments + higher_payments = 12 := by
  -- Proof goes here
  sorry

end mike_payments_total_months_l2357_235790


namespace total_good_vegetables_l2357_235795

theorem total_good_vegetables :
  let carrots_day1 := 23
  let carrots_day2 := 47
  let tomatoes_day1 := 34
  let cucumbers_day1 := 42
  let tomatoes_day2 := 50
  let cucumbers_day2 := 38
  let rotten_carrots_day1 := 10
  let rotten_carrots_day2 := 15
  let rotten_tomatoes_day1 := 5
  let rotten_cucumbers_day1 := 7
  let rotten_tomatoes_day2 := 7
  let rotten_cucumbers_day2 := 12
  let good_carrots := (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2)
  let good_tomatoes := (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2)
  let good_cucumbers := (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2)
  good_carrots + good_tomatoes + good_cucumbers = 178 := 
  sorry

end total_good_vegetables_l2357_235795


namespace largest_three_digit_divisible_and_prime_sum_l2357_235733

theorem largest_three_digit_divisible_and_prime_sum :
  ∃ n : ℕ, 900 ≤ n ∧ n < 1000 ∧
           (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ≠ 0 ∧ n % d = 0) ∧
           Prime (n / 100 + (n / 10) % 10 + n % 10) ∧
           n = 963 ∧
           ∀ m : ℕ, 900 ≤ m ∧ m < 1000 ∧
           (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ≠ 0 ∧ m % d = 0) ∧
           Prime (m / 100 + (m / 10) % 10 + m % 10) →
           m ≤ 963 :=
by
  sorry

end largest_three_digit_divisible_and_prime_sum_l2357_235733


namespace exists_perpendicular_line_l2357_235780

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure DirectionVector :=
  (dx : ℝ)
  (dy : ℝ)
  (dz : ℝ)

noncomputable def parametric_line_through_point 
  (P : Point3D) 
  (d : DirectionVector) : Prop :=
  ∀ t : ℝ, ∃ x y z : ℝ, 
  x = P.x + d.dx * t ∧
  y = P.y + d.dy * t ∧
  z = P.z + d.dz * t

theorem exists_perpendicular_line : 
  ∃ d : DirectionVector, 
    (d.dx * 2 + d.dy * 3 - d.dz = 0) ∧ 
    (d.dx * 4 - d.dy * -1 + d.dz * 3 = 0) ∧ 
    parametric_line_through_point 
      ⟨3, -2, 1⟩ d :=
  sorry

end exists_perpendicular_line_l2357_235780


namespace average_age_new_students_l2357_235727

theorem average_age_new_students (A : ℚ)
    (avg_original_age : ℚ := 48)
    (num_new_students : ℚ := 120)
    (new_avg_age : ℚ := 44)
    (total_students : ℚ := 160) :
    let num_original_students := total_students - num_new_students
    let total_age_original := num_original_students * avg_original_age
    let total_age_all := total_students * new_avg_age
    total_age_original + (num_new_students * A) = total_age_all → A = 42.67 := 
by
  intros
  sorry

end average_age_new_students_l2357_235727


namespace dislike_both_tv_and_video_games_l2357_235759

theorem dislike_both_tv_and_video_games (total_people : ℕ) (percent_dislike_tv : ℝ) (percent_dislike_tv_and_games : ℝ) :
  let people_dislike_tv := percent_dislike_tv * total_people
  let people_dislike_both := percent_dislike_tv_and_games * people_dislike_tv
  total_people = 1800 ∧ percent_dislike_tv = 0.4 ∧ percent_dislike_tv_and_games = 0.25 →
  people_dislike_both = 180 :=
by {
  sorry
}

end dislike_both_tv_and_video_games_l2357_235759


namespace gcd_324_243_l2357_235782

-- Define the numbers involved in the problem.
def a : ℕ := 324
def b : ℕ := 243

-- State the theorem that the GCD of a and b is 81.
theorem gcd_324_243 : Nat.gcd a b = 81 := by
  sorry

end gcd_324_243_l2357_235782


namespace complement_intersection_example_l2357_235777

open Set

theorem complement_intersection_example
  (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 4})
  (hB : B = {2, 3}) :
  (U \ A) ∩ B = {2} :=
by
  sorry

end complement_intersection_example_l2357_235777


namespace gcd_lcm_product_l2357_235774

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l2357_235774


namespace team_A_wins_2_1_team_B_wins_l2357_235718

theorem team_A_wins_2_1 (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (2 * p_a * p_b) * p_a = 0.288 := by
  sorry

theorem team_B_wins (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (p_b * p_b) + (2 * p_a * p_b * p_b) = 0.352 := by
  sorry

end team_A_wins_2_1_team_B_wins_l2357_235718


namespace water_park_children_l2357_235711

theorem water_park_children (cost_adult cost_child total_cost : ℝ) (c : ℕ) 
  (h1 : cost_adult = 1)
  (h2 : cost_child = 0.75)
  (h3 : total_cost = 3.25) :
  c = 3 :=
by
  sorry

end water_park_children_l2357_235711


namespace simplify_expression_l2357_235764

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) :=
by
  sorry

end simplify_expression_l2357_235764


namespace regular_polygon_sides_l2357_235737

theorem regular_polygon_sides (θ : ℝ) (hθ : θ = 45) : 360 / θ = 8 := by
  sorry

end regular_polygon_sides_l2357_235737


namespace number_of_men_in_first_group_l2357_235720

/-
Given the initial conditions:
1. Some men can color a 48 m long cloth in 2 days.
2. 6 men can color a 36 m long cloth in 1 day.

We need to prove that the number of men in the first group is equal to 9.
-/

theorem number_of_men_in_first_group (M : ℕ)
    (h1 : ∃ (x : ℕ), x * 48 = M * 2)
    (h2 : 6 * 36 = 36 * 1) :
    M = 9 :=
by
sorry

end number_of_men_in_first_group_l2357_235720


namespace times_faster_l2357_235745

theorem times_faster (A B W : ℝ) (h1 : A = 3 * B) (h2 : (A + B) * 21 = A * 28) : A = 3 * B :=
by sorry

end times_faster_l2357_235745


namespace amount_C_l2357_235730

theorem amount_C (A_amt B_amt C_amt : ℚ)
  (h1 : A_amt + B_amt + C_amt = 527)
  (h2 : A_amt = (2 / 3) * B_amt)
  (h3 : B_amt = (1 / 4) * C_amt) :
  C_amt = 372 :=
sorry

end amount_C_l2357_235730


namespace unique_polynomial_l2357_235794

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem unique_polynomial 
  (a b c : ℝ) 
  (extremes : f' a b c 1 = 0 ∧ f' a b c (-1) = 0) 
  (tangent_slope : f' a b c 0 = -3)
  : f a b c = f 1 0 (-3) := sorry

end unique_polynomial_l2357_235794


namespace total_amount_spent_l2357_235749

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l2357_235749


namespace cistern_water_breadth_l2357_235799

theorem cistern_water_breadth (length width total_area : ℝ) (h : ℝ) 
  (h_length : length = 10) 
  (h_width : width = 6) 
  (h_area : total_area = 103.2) : 
  (60 + 20*h + 12*h = total_area) → h = 1.35 :=
by
  intros
  sorry

end cistern_water_breadth_l2357_235799


namespace market_value_of_stock_l2357_235753

theorem market_value_of_stock (dividend_rate : ℝ) (yield_rate : ℝ) (face_value : ℝ) :
  dividend_rate = 0.12 → yield_rate = 0.08 → face_value = 100 → (dividend_rate * face_value / yield_rate * 100) = 150 :=
by
  intros h1 h2 h3
  sorry

end market_value_of_stock_l2357_235753


namespace fractional_eq_no_solution_l2357_235734

theorem fractional_eq_no_solution (m : ℝ) :
  ¬ ∃ x, (x - 2) / (x + 2) - (m * x) / (x^2 - 4) = 1 ↔ m = -4 :=
by
  sorry

end fractional_eq_no_solution_l2357_235734


namespace horner_method_V3_correct_when_x_equals_2_l2357_235715

-- Polynomial f(x)
noncomputable def f (x : ℝ) : ℝ :=
  2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

-- Horner's method for evaluating f(x)
noncomputable def V3 (x : ℝ) : ℝ :=
  (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Proof that V3 = 12 when x = 2
theorem horner_method_V3_correct_when_x_equals_2 : V3 2 = 12 := by
  sorry

end horner_method_V3_correct_when_x_equals_2_l2357_235715


namespace inequality_a_cube_less_b_cube_l2357_235773

theorem inequality_a_cube_less_b_cube (a b : ℝ) (ha : a < 0) (hb : b > 0) : a^3 < b^3 :=
by
  sorry

end inequality_a_cube_less_b_cube_l2357_235773


namespace total_hamburger_varieties_l2357_235736

def num_condiments : ℕ := 9
def num_condiment_combinations : ℕ := 2 ^ num_condiments
def num_patties_choices : ℕ := 4
def num_bread_choices : ℕ := 2

theorem total_hamburger_varieties : num_condiment_combinations * num_patties_choices * num_bread_choices = 4096 :=
by
  -- conditions
  have h1 : num_condiments = 9 := rfl
  have h2 : num_condiment_combinations = 2 ^ num_condiments := rfl
  have h3 : num_patties_choices = 4 := rfl
  have h4 : num_bread_choices = 2 := rfl

  -- correct answer
  sorry

end total_hamburger_varieties_l2357_235736


namespace second_markdown_percentage_l2357_235763

theorem second_markdown_percentage (P : ℝ) (h1 : P > 0)
    (h2 : ∃ x : ℝ, x = 0.50 * P) -- First markdown
    (h3 : ∃ y : ℝ, y = 0.45 * P) -- Final price
    : ∃ X : ℝ, X = 10 := 
sorry

end second_markdown_percentage_l2357_235763


namespace range_of_independent_variable_l2357_235781

theorem range_of_independent_variable (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end range_of_independent_variable_l2357_235781


namespace man_l2357_235702

theorem man's_speed_with_stream
  (V_m V_s : ℝ)
  (h1 : V_m = 6)
  (h2 : V_m - V_s = 4) :
  V_m + V_s = 8 :=
sorry

end man_l2357_235702


namespace margaret_spends_on_croissants_l2357_235776

theorem margaret_spends_on_croissants :
  (∀ (people : ℕ) (sandwiches_per_person : ℕ) (croissants_per_sandwich : ℕ) (croissants_per_set : ℕ) (cost_per_set : ℝ),
    people = 24 →
    sandwiches_per_person = 2 →
    croissants_per_sandwich = 1 →
    croissants_per_set = 12 →
    cost_per_set = 8 →
    (people * sandwiches_per_person * croissants_per_sandwich) / croissants_per_set * cost_per_set = 32) := sorry

end margaret_spends_on_croissants_l2357_235776


namespace probability_not_overcoming_is_half_l2357_235784

/-- Define the five elements. -/
inductive Element
| metal | wood | water | fire | earth

open Element

/-- Define the overcoming relation. -/
def overcomes : Element → Element → Prop
| metal, wood => true
| wood, earth => true
| earth, water => true
| water, fire => true
| fire, metal => true
| _, _ => false

/-- Define the probability calculation. -/
def probability_not_overcoming : ℚ :=
  let total_combinations := 10    -- C(5, 2)
  let overcoming_combinations := 5
  let not_overcoming_combinations := total_combinations - overcoming_combinations
  not_overcoming_combinations / total_combinations

/-- The proof problem statement. -/
theorem probability_not_overcoming_is_half : probability_not_overcoming = 1 / 2 :=
by
  sorry

end probability_not_overcoming_is_half_l2357_235784


namespace sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l2357_235747

theorem sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C
  {A B C : ℝ}
  (h : A + B + C = π) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) :=
sorry

end sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l2357_235747


namespace functional_ineq_l2357_235714

noncomputable def f : ℝ → ℝ := sorry

theorem functional_ineq (h1 : ∀ x > 1400^2021, x * f x ≤ 2021) (h2 : ∀ x : ℝ, 0 < x → f x = f (x + 2) + 2 * f (x * (x + 2))) : 
  ∀ x : ℝ, 0 < x → x * f x ≤ 2021 :=
sorry

end functional_ineq_l2357_235714


namespace count_perfect_cubes_l2357_235732

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end count_perfect_cubes_l2357_235732


namespace triangle_rectangle_ratio_l2357_235708

theorem triangle_rectangle_ratio (s b w l : ℕ) 
(h1 : 2 * s + b = 60) 
(h2 : 2 * (w + l) = 60) 
(h3 : 2 * w = l) 
(h4 : b = w) 
: s / w = 5 / 2 := 
by 
  sorry

end triangle_rectangle_ratio_l2357_235708


namespace circle_equation_exists_l2357_235735

noncomputable def point (α : Type*) := {p : α × α // ∃ x y : α, p = (x, y)}

structure Circle (α : Type*) :=
(center : α × α)
(radius : α)

def passes_through (c : Circle ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) ^ 2 + (p.2 - c.center.2) ^ 2 = c.radius ^ 2

theorem circle_equation_exists :
  ∃ (c : Circle ℝ),
    c.center = (-4, 3) ∧ c.radius = 5 ∧ passes_through c (-1, -1) ∧ passes_through c (-8, 0) ∧ passes_through c (0, 6) :=
by { sorry }

end circle_equation_exists_l2357_235735


namespace probability_of_rolling_2_4_6_l2357_235741

open Set Classical

noncomputable def fair_six_sided_die_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def successful_outcomes : Finset ℕ := {2, 4, 6}

theorem probability_of_rolling_2_4_6 : 
  (successful_outcomes.card : ℚ) / (fair_six_sided_die_outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_rolling_2_4_6_l2357_235741


namespace smallest_angle_measure_l2357_235721

-- Define the conditions
def is_spherical_triangle (a b c : ℝ) : Prop :=
  a + b + c > 180 ∧ a + b + c < 540

def angles (k : ℝ) : Prop :=
  let a := 3 * k
  let b := 4 * k
  let c := 5 * k
  is_spherical_triangle a b c ∧ a + b + c = 270

-- Statement of the theorem
theorem smallest_angle_measure (k : ℝ) (h : angles k) : 3 * k = 67.5 :=
sorry

end smallest_angle_measure_l2357_235721


namespace lines_are_perpendicular_l2357_235752

noncomputable def line1 := {x : ℝ | ∃ y : ℝ, x + y - 1 = 0}
noncomputable def line2 := {x : ℝ | ∃ y : ℝ, x - y + 1 = 0}

theorem lines_are_perpendicular : 
  let slope1 := -1
  let slope2 := 1
  slope1 * slope2 = -1 := sorry

end lines_are_perpendicular_l2357_235752


namespace functional_equation_solution_l2357_235726

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * y * f x) :
  ∀ x : ℝ, f x = 0 := 
sorry

end functional_equation_solution_l2357_235726


namespace jessica_total_cost_l2357_235756

def price_of_cat_toy : ℝ := 10.22
def price_of_cage : ℝ := 11.73
def price_of_cat_food : ℝ := 5.65
def price_of_catnip : ℝ := 2.30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.07

def discounted_price_of_cat_toy : ℝ := price_of_cat_toy * (1 - discount_rate)
def total_cost_before_tax : ℝ := discounted_price_of_cat_toy + price_of_cage + price_of_cat_food + price_of_catnip
def sales_tax : ℝ := total_cost_before_tax * tax_rate
def total_cost_after_discount_and_tax : ℝ := total_cost_before_tax + sales_tax

theorem jessica_total_cost : total_cost_after_discount_and_tax = 30.90 := by
  sorry

end jessica_total_cost_l2357_235756


namespace elvis_squares_count_l2357_235716

theorem elvis_squares_count :
  ∀ (total : ℕ) (Elvis_squares Ralph_squares squares_used_by_Ralph matchsticks_left : ℕ)
  (uses_by_Elvis_per_square uses_by_Ralph_per_square : ℕ),
  total = 50 →
  uses_by_Elvis_per_square = 4 →
  uses_by_Ralph_per_square = 8 →
  Ralph_squares = 3 →
  matchsticks_left = 6 →
  squares_used_by_Ralph = Ralph_squares * uses_by_Ralph_per_square →
  total = (Elvis_squares * uses_by_Elvis_per_square) + squares_used_by_Ralph + matchsticks_left →
  Elvis_squares = 5 :=
by
  sorry

end elvis_squares_count_l2357_235716


namespace shaded_square_area_l2357_235754

theorem shaded_square_area (a b s : ℝ) (h : a * b = 40) :
  ∃ s, s^2 = 2500 / 441 :=
by
  sorry

end shaded_square_area_l2357_235754


namespace pizzas_in_park_l2357_235789

-- Define the conditions and the proof problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100  -- in meters
def building_distance : ℕ := 2000  -- in meters
def pizzas_delivered_to_building : ℕ := 2
def total_payment_received : ℕ := 64

-- Prove the number of pizzas delivered in the park
theorem pizzas_in_park : (64 - (pizzas_delivered_to_building * pizza_cost + delivery_charge)) / pizza_cost = 3 :=
by
  sorry -- Proof not required

end pizzas_in_park_l2357_235789


namespace work_completion_time_l2357_235706

theorem work_completion_time (P W : ℕ) (h : P * 8 = W) : 2 * P * 2 = W / 2 := by
  sorry

end work_completion_time_l2357_235706


namespace range_of_a_l2357_235766

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 :=
sorry

end range_of_a_l2357_235766


namespace minimum_value_of_f_l2357_235755

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 4 * x + 3)

theorem minimum_value_of_f : ∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use -16
  sorry

end minimum_value_of_f_l2357_235755


namespace total_chocolate_bar_count_l2357_235709

def large_box_count : ℕ := 150
def small_box_count_per_large_box : ℕ := 45
def chocolate_bar_count_per_small_box : ℕ := 35

theorem total_chocolate_bar_count :
  large_box_count * small_box_count_per_large_box * chocolate_bar_count_per_small_box = 236250 :=
by
  sorry

end total_chocolate_bar_count_l2357_235709


namespace ratio_large_to_small_l2357_235793

-- Definitions of the conditions
def total_fries_sold : ℕ := 24
def small_fries_sold : ℕ := 4
def large_fries_sold : ℕ := total_fries_sold - small_fries_sold

-- The proof goal
theorem ratio_large_to_small : large_fries_sold / small_fries_sold = 5 :=
by
  -- Mathematical steps would go here, but we skip with sorry
  sorry

end ratio_large_to_small_l2357_235793


namespace form_triangle_condition_right_angled_triangle_condition_l2357_235770

def vector (α : Type*) := α × α
noncomputable def oa : vector ℝ := ⟨2, -1⟩
noncomputable def ob : vector ℝ := ⟨3, 2⟩
noncomputable def oc (m : ℝ) : vector ℝ := ⟨m, 2 * m + 1⟩

def vector_sub (v1 v2 : vector ℝ) : vector ℝ := ⟨v1.1 - v2.1, v1.2 - v2.2⟩
def vector_dot (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem form_triangle_condition (m : ℝ) : 
  ¬ ((vector_sub ob oa).1 * (vector_sub (oc m) oa).2 = (vector_sub ob oa).2 * (vector_sub (oc m) oa).1) ↔ m ≠ 8 :=
sorry

theorem right_angled_triangle_condition (m : ℝ) : 
  (vector_dot (vector_sub ob oa) (vector_sub (oc m) oa) = 0 ∨ 
   vector_dot (vector_sub ob oa) (vector_sub (oc m) ob) = 0 ∨ 
   vector_dot (vector_sub (oc m) oa) (vector_sub (oc m) ob) = 0) ↔ 
  (m = -4/7 ∨ m = 6/7) :=
sorry

end form_triangle_condition_right_angled_triangle_condition_l2357_235770


namespace complex_exp_cos_l2357_235701

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l2357_235701


namespace situationD_not_represented_l2357_235775

def situationA := -2 + 10 = 8

def situationB := -2 + 10 = 8

def situationC := 10 - 2 = 8 ∧ -2 + 10 = 8

def situationD := |10 - (-2)| = 12

theorem situationD_not_represented : ¬ (|10 - (-2)| = -2 + 10) := 
by
  sorry

end situationD_not_represented_l2357_235775


namespace find_r_amount_l2357_235786

theorem find_r_amount (p q r : ℝ) (h_total : p + q + r = 8000) (h_r_fraction : r = 2 / 3 * (p + q)) : r = 3200 :=
by 
  -- Proof is not required, hence we use sorry
  sorry

end find_r_amount_l2357_235786


namespace point_in_first_quadrant_l2357_235791

-- Define the system of equations
def equations (x y : ℝ) : Prop :=
  x + y = 2 ∧ x - y = 1

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : equations x y) : x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l2357_235791


namespace anita_smallest_number_of_candies_l2357_235768

theorem anita_smallest_number_of_candies :
  ∃ x : ℕ, x ≡ 5 [MOD 6] ∧ x ≡ 3 [MOD 8] ∧ x ≡ 7 [MOD 9] ∧ ∀ y : ℕ,
  (y ≡ 5 [MOD 6] ∧ y ≡ 3 [MOD 8] ∧ y ≡ 7 [MOD 9]) → x ≤ y :=
  ⟨203, by sorry⟩

end anita_smallest_number_of_candies_l2357_235768
