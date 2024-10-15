import Mathlib

namespace NUMINAMATH_GPT_brenda_cakes_l2224_222437

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end NUMINAMATH_GPT_brenda_cakes_l2224_222437


namespace NUMINAMATH_GPT_jerry_daughters_games_l2224_222470

theorem jerry_daughters_games (x y : ℕ) (h : 4 * x + 2 * x + 4 * y + 2 * y = 96) (hx : x = y) :
  x = 8 ∧ y = 8 :=
by
  have h1 : 6 * x + 6 * y = 96 := by linarith
  have h2 : x = y := hx
  sorry

end NUMINAMATH_GPT_jerry_daughters_games_l2224_222470


namespace NUMINAMATH_GPT_num_convex_pentagons_l2224_222485

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end NUMINAMATH_GPT_num_convex_pentagons_l2224_222485


namespace NUMINAMATH_GPT_radio_price_position_l2224_222406

theorem radio_price_position (n : ℕ) (h₁ : n = 42)
  (h₂ : ∃ m : ℕ, m = 18 ∧ 
    (∀ k : ℕ, k < m → (∃ x : ℕ, x > k))) : 
    ∃ m : ℕ, m = 24 :=
by
  sorry

end NUMINAMATH_GPT_radio_price_position_l2224_222406


namespace NUMINAMATH_GPT_Grisha_owes_correct_l2224_222455

noncomputable def Grisha_owes (dish_cost : ℝ) : ℝ × ℝ :=
  let misha_paid := 3 * dish_cost
  let sasha_paid := 2 * dish_cost
  let friends_contribution := 50
  let equal_payment := 50 / 2
  (misha_paid - equal_payment, sasha_paid - equal_payment)

theorem Grisha_owes_correct :
  ∀ (dish_cost : ℝ), (dish_cost = 30) → Grisha_owes dish_cost = (40, 10) :=
by
  intro dish_cost h
  rw [h]
  unfold Grisha_owes
  simp
  sorry

end NUMINAMATH_GPT_Grisha_owes_correct_l2224_222455


namespace NUMINAMATH_GPT_debby_drinking_days_l2224_222460

def starting_bottles := 264
def daily_consumption := 15
def bottles_left := 99

theorem debby_drinking_days : (starting_bottles - bottles_left) / daily_consumption = 11 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_debby_drinking_days_l2224_222460


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2224_222472

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2224_222472


namespace NUMINAMATH_GPT_perfect_square_of_seq_l2224_222469

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 3, a n = 7 * a (n - 1) - a (n - 2)

theorem perfect_square_of_seq (a : ℕ → ℤ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, k * k = a n + 2 + a (n + 1) :=
sorry

end NUMINAMATH_GPT_perfect_square_of_seq_l2224_222469


namespace NUMINAMATH_GPT_prove_partial_fractions_identity_l2224_222407

def partial_fraction_identity (x : ℚ) (A B C a b c : ℚ) : Prop :=
  a = 0 ∧ b = 1 ∧ c = -1 ∧
  (A / (x - a) + B / (x - b) + C / (x - c) = 4*x - 2 ∧ x^3 - x ≠ 0)

theorem prove_partial_fractions_identity :
  (partial_fraction_identity x 2 1 (-3) 0 1 (-1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_partial_fractions_identity_l2224_222407


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2224_222416

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2224_222416


namespace NUMINAMATH_GPT_ratio_of_numbers_l2224_222474

theorem ratio_of_numbers
  (greater less : ℕ)
  (h1 : greater = 64)
  (h2 : less = 32)
  (h3 : greater + less = 96)
  (h4 : ∃ k : ℕ, greater = k * less) :
  greater / less = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2224_222474


namespace NUMINAMATH_GPT_find_p_q_d_l2224_222440

def f (p q d : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

theorem find_p_q_d :
  ∃ p q d : ℕ, f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 ∧ (p + q + d = 13) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_d_l2224_222440


namespace NUMINAMATH_GPT_probability_of_sum_six_two_dice_l2224_222478

noncomputable def probability_sum_six : ℚ := 5 / 36

theorem probability_of_sum_six_two_dice (dice_faces : ℕ := 6) : 
  ∃ (p : ℚ), p = probability_sum_six :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sum_six_two_dice_l2224_222478


namespace NUMINAMATH_GPT_quarters_count_l2224_222427

noncomputable def num_coins := 12
noncomputable def total_value := 166 -- in cents
noncomputable def min_value := 1 + 5 + 10 + 25 + 50 -- minimum value from one of each type
noncomputable def remaining_value := total_value - min_value
noncomputable def remaining_coins := num_coins - 5

theorem quarters_count :
  ∀ (p n d q h : ℕ), 
  p + n + d + q + h = num_coins ∧
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ h ≥ 1 ∧
  (p + 5*n + 10*d + 25*q + 50*h = total_value) → 
  q = 3 := 
by 
  sorry

end NUMINAMATH_GPT_quarters_count_l2224_222427


namespace NUMINAMATH_GPT_prime_sum_l2224_222403

theorem prime_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : 2 * p + 3 * q = 6 * r) : 
  p + q + r = 7 := 
sorry

end NUMINAMATH_GPT_prime_sum_l2224_222403


namespace NUMINAMATH_GPT_max_fraction_diagonals_sides_cyclic_pentagon_l2224_222482

theorem max_fraction_diagonals_sides_cyclic_pentagon (a b c d e A B C D E : ℝ)
  (h1 : b * e + a * A = C * D)
  (h2 : c * a + b * B = D * E)
  (h3 : d * b + c * C = E * A)
  (h4 : e * c + d * D = A * B)
  (h5 : a * d + e * E = B * C) :
  (a * b * c * d * e) / (A * B * C * D * E) ≤ (5 * Real.sqrt 5 - 11) / 2 :=
sorry

end NUMINAMATH_GPT_max_fraction_diagonals_sides_cyclic_pentagon_l2224_222482


namespace NUMINAMATH_GPT_M_lies_in_third_quadrant_l2224_222497

noncomputable def harmonious_point (a b : ℝ) : Prop :=
  3 * a = 2 * b + 5

noncomputable def point_M_harmonious (m : ℝ) : Prop :=
  harmonious_point (m - 1) (3 * m + 2)

theorem M_lies_in_third_quadrant (m : ℝ) (hM : point_M_harmonious m) : 
  (m - 1 < 0 ∧ 3 * m + 2 < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_M_lies_in_third_quadrant_l2224_222497


namespace NUMINAMATH_GPT_correct_sum_is_132_l2224_222442

-- Let's define the conditions:
-- The ones digit B is mistakenly taken as 1 (when it should be 7)
-- The tens digit C is mistakenly taken as 6 (when it should be 4)
-- The incorrect sum is 146

def correct_ones_digit (mistaken_ones_digit : Nat) : Nat :=
  -- B was mistaken for 1, so B should be 7
  if mistaken_ones_digit = 1 then 7 else mistaken_ones_digit

def correct_tens_digit (mistaken_tens_digit : Nat) : Nat :=
  -- C was mistaken for 6, so C should be 4
  if mistaken_tens_digit = 6 then 4 else mistaken_tens_digit

def correct_sum (incorrect_sum : Nat) : Nat :=
  -- Correcting the sum based on the mistakes
  incorrect_sum + 6 - 20 -- 6 to correct ones mistake, minus 20 to correct tens mistake

theorem correct_sum_is_132 : correct_sum 146 = 132 :=
  by
    -- The theorem is here to check that the corrected sum equals 132
    sorry

end NUMINAMATH_GPT_correct_sum_is_132_l2224_222442


namespace NUMINAMATH_GPT_quadratic_solution_exists_l2224_222468

theorem quadratic_solution_exists (a b : ℝ) : ∃ (x : ℝ), (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_exists_l2224_222468


namespace NUMINAMATH_GPT_area_of_set_R_is_1006point5_l2224_222450

-- Define the set of points R as described in the problem
def isPointInSetR (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y ≤ 2013 ∧ ⌈x⌉ * ⌊y⌋ = ⌊x⌋ * ⌈y⌉

noncomputable def computeAreaOfSetR : ℝ :=
  1006.5

theorem area_of_set_R_is_1006point5 :
  (∃ x y : ℝ, isPointInSetR x y) → computeAreaOfSetR = 1006.5 := by
  sorry

end NUMINAMATH_GPT_area_of_set_R_is_1006point5_l2224_222450


namespace NUMINAMATH_GPT_solve_for_y_l2224_222459

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end NUMINAMATH_GPT_solve_for_y_l2224_222459


namespace NUMINAMATH_GPT_Marcus_ate_more_than_John_l2224_222420

theorem Marcus_ate_more_than_John:
  let John_eaten := 28
  let Marcus_eaten := 40
  Marcus_eaten - John_eaten = 12 :=
by
  sorry

end NUMINAMATH_GPT_Marcus_ate_more_than_John_l2224_222420


namespace NUMINAMATH_GPT_find_k_and_f_min_total_cost_l2224_222494

-- Define the conditions
def construction_cost (x : ℝ) : ℝ := 60 * x
def energy_consumption_cost (x : ℝ) : ℝ := 40 - 4 * x
def total_cost (x : ℝ) : ℝ := construction_cost x + 20 * energy_consumption_cost x

theorem find_k_and_f :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → energy_consumption_cost 0 = 8 → energy_consumption_cost x = 40 - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 10 → total_cost x = 800 - 74 * x) :=
by
  sorry

theorem min_total_cost :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → 800 - 74 * x ≥ 70) ∧
  total_cost 5 = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_f_min_total_cost_l2224_222494


namespace NUMINAMATH_GPT_range_of_a_l2224_222461

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3 * a) ↔ (a ≥ 4 ∨ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2224_222461


namespace NUMINAMATH_GPT_sausage_thickness_correct_l2224_222479

noncomputable def earth_radius := 6000 -- in km
noncomputable def distance_to_sun := 150000000 -- in km
noncomputable def sausage_thickness := 44 -- in km

theorem sausage_thickness_correct :
  let R := earth_radius
  let L := distance_to_sun
  let r := Real.sqrt ((4 * R^3) / (3 * L))
  abs (r - sausage_thickness) < 10 * sausage_thickness :=
by
  sorry

end NUMINAMATH_GPT_sausage_thickness_correct_l2224_222479


namespace NUMINAMATH_GPT_last_three_digits_of_2_pow_6000_l2224_222492

theorem last_three_digits_of_2_pow_6000 (h : 2^200 ≡ 1 [MOD 800]) : (2^6000 ≡ 1 [MOD 800]) :=
sorry

end NUMINAMATH_GPT_last_three_digits_of_2_pow_6000_l2224_222492


namespace NUMINAMATH_GPT_part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l2224_222462

open Set Real

def setA (a : ℝ) : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ a + 5}
def setB : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem part_one_a_two_complement_union (a : ℝ) (h : a = 2) :
  compl (setA a ∪ setB) = Iic 2 ∪ Ici 10 := sorry

theorem part_one_a_two_complement_intersection (a : ℝ) (h : a = 2) :
  compl (setA a) ∩ setB = Ioo 2 3 ∪ Ioo 7 10 := sorry

theorem part_two_subset (a : ℝ) (h : setA a ⊆ setB) :
  a < 5 := sorry

end NUMINAMATH_GPT_part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l2224_222462


namespace NUMINAMATH_GPT_simplify_expr_l2224_222446

noncomputable def expr : ℝ := Real.sqrt 12 - 3 * Real.sqrt (1 / 3) + Real.sqrt 27 + (Real.pi + 1)^0

theorem simplify_expr : expr = 4 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l2224_222446


namespace NUMINAMATH_GPT_total_population_of_city_l2224_222480

theorem total_population_of_city (P : ℝ) (h : 0.85 * P = 85000) : P = 100000 :=
  by
  sorry

end NUMINAMATH_GPT_total_population_of_city_l2224_222480


namespace NUMINAMATH_GPT_max_peaceful_clients_kept_l2224_222489

-- Defining the types for knights, liars, and troublemakers
def Person : Type := ℕ

noncomputable def isKnight : Person → Prop := sorry
noncomputable def isLiar : Person → Prop := sorry
noncomputable def isTroublemaker : Person → Prop := sorry

-- Total number of people in the bar
def totalPeople : ℕ := 30

-- Number of knights, liars, and troublemakers
def numberKnights : ℕ := 10
def numberLiars : ℕ := 10
def numberTroublemakers : ℕ := 10

-- The bartender's goal: get rid of all troublemakers and keep as many peaceful clients as possible
def maxPeacefulClients (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ): ℕ :=
  total - troublemakers

-- Statement to be proved
theorem max_peaceful_clients_kept (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ)
  (h_total : total = 30)
  (h_knights : knights = 10)
  (h_liars : liars = 10)
  (h_troublemakers : troublemakers = 10) :
  maxPeacefulClients total knights liars troublemakers = 19 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_max_peaceful_clients_kept_l2224_222489


namespace NUMINAMATH_GPT_acute_angle_parallel_vectors_l2224_222444

theorem acute_angle_parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
    (h₁ : a = (Real.sin x, 1))
    (h₂ : b = (1 / 2, Real.cos x))
    (h₃ : ∃ k : ℝ, a = k • b ∧ k ≠ 0) :
    x = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_parallel_vectors_l2224_222444


namespace NUMINAMATH_GPT_find_x_plus_y_l2224_222415

theorem find_x_plus_y (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x - y > 0) : x + y = 8 ∨ x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l2224_222415


namespace NUMINAMATH_GPT_simplify_expression_l2224_222438

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end NUMINAMATH_GPT_simplify_expression_l2224_222438


namespace NUMINAMATH_GPT_non_drinkers_count_l2224_222453

-- Define the total number of businessmen and the sets of businessmen drinking each type of beverage.
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def soda_drinkers : ℕ := 8
def coffee_tea_drinkers : ℕ := 7
def tea_soda_drinkers : ℕ := 3
def coffee_soda_drinkers : ℕ := 2
def all_three_drinkers : ℕ := 1

-- Statement to prove:
theorem non_drinkers_count :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers - coffee_tea_drinkers - tea_soda_drinkers - coffee_soda_drinkers + all_three_drinkers) = 6 :=
by
  -- Skip the proof for now.
  sorry

end NUMINAMATH_GPT_non_drinkers_count_l2224_222453


namespace NUMINAMATH_GPT_students_from_other_communities_l2224_222473

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end NUMINAMATH_GPT_students_from_other_communities_l2224_222473


namespace NUMINAMATH_GPT_min_value_fraction_sum_l2224_222432

theorem min_value_fraction_sum : 
  ∀ (n : ℕ), n > 0 → (n / 3 + 27 / n) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l2224_222432


namespace NUMINAMATH_GPT_solve_system_of_equations_l2224_222417

def system_solution : Prop := ∃ x y : ℚ, 4 * x - 6 * y = -14 ∧ 8 * x + 3 * y = -15 ∧ x = -11 / 5 ∧ y = 2.6 / 3

theorem solve_system_of_equations : system_solution := sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2224_222417


namespace NUMINAMATH_GPT_value_of_first_equation_l2224_222434

theorem value_of_first_equation (x y : ℚ) 
  (h1 : 5 * x + 6 * y = 7) 
  (h2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 :=
sorry

end NUMINAMATH_GPT_value_of_first_equation_l2224_222434


namespace NUMINAMATH_GPT_evaluate_expression_l2224_222411

theorem evaluate_expression : (4 - 3) * 2 = 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2224_222411


namespace NUMINAMATH_GPT_tan_theta_eq_neg_2sqrt2_to_expression_l2224_222400

theorem tan_theta_eq_neg_2sqrt2_to_expression (θ : ℝ) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2)) ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_eq_neg_2sqrt2_to_expression_l2224_222400


namespace NUMINAMATH_GPT_gcd_12345_6789_l2224_222465

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_12345_6789_l2224_222465


namespace NUMINAMATH_GPT_no_three_consecutive_geo_prog_l2224_222421

theorem no_three_consecutive_geo_prog (n k m: ℕ) (h: n ≠ k ∧ n ≠ m ∧ k ≠ m) :
  ¬(∃ a b c: ℕ, 
    (a = 2^n + 1 ∧ b = 2^k + 1 ∧ c = 2^m + 1) ∧ 
    (b^2 = a * c)) :=
by sorry

end NUMINAMATH_GPT_no_three_consecutive_geo_prog_l2224_222421


namespace NUMINAMATH_GPT_sin_is_odd_and_has_zero_point_l2224_222426

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem sin_is_odd_and_has_zero_point :
  is_odd_function sin ∧ has_zero_point sin := 
  by sorry

end NUMINAMATH_GPT_sin_is_odd_and_has_zero_point_l2224_222426


namespace NUMINAMATH_GPT_solve_system_l2224_222431

variable {R : Type*} [CommRing R]

-- Given conditions
variables (a b c x y z : R)

-- Assuming the given system of equations
axiom eq1 : x + a*y + a^2*z + a^3 = 0
axiom eq2 : x + b*y + b^2*z + b^3 = 0
axiom eq3 : x + c*y + c^2*z + c^3 = 0

-- The goal is to prove the mathematical equivalence
theorem solve_system : x = -a*b*c ∧ y = a*b + b*c + c*a ∧ z = -(a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2224_222431


namespace NUMINAMATH_GPT_older_brother_age_is_25_l2224_222433

noncomputable def age_of_older_brother (father_age current_n : ℕ) (younger_brother_age : ℕ) : ℕ := 
  (father_age - current_n) / 2

theorem older_brother_age_is_25 
  (father_age : ℕ) 
  (h1 : father_age = 50) 
  (younger_brother_age : ℕ)
  (current_n : ℕ) 
  (h2 : (2 * (younger_brother_age + current_n)) = father_age + current_n) : 
  age_of_older_brother father_age current_n younger_brother_age = 25 := 
by
  sorry

end NUMINAMATH_GPT_older_brother_age_is_25_l2224_222433


namespace NUMINAMATH_GPT_Irja_wins_probability_l2224_222451

noncomputable def probability_irja_wins : ℚ :=
  let X0 : ℚ := 4 / 7
  X0

theorem Irja_wins_probability :
  probability_irja_wins = 4 / 7 :=
sorry

end NUMINAMATH_GPT_Irja_wins_probability_l2224_222451


namespace NUMINAMATH_GPT_garrett_granola_bars_l2224_222452

theorem garrett_granola_bars :
  ∀ (oatmeal_raisin peanut total : ℕ),
  peanut = 8 →
  total = 14 →
  oatmeal_raisin + peanut = total →
  oatmeal_raisin = 6 :=
by
  intros oatmeal_raisin peanut total h_peanut h_total h_sum
  sorry

end NUMINAMATH_GPT_garrett_granola_bars_l2224_222452


namespace NUMINAMATH_GPT_exponential_function_range_l2224_222418

noncomputable def exponential_function (a x : ℝ) : ℝ := a^x

theorem exponential_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : exponential_function a (-2) < exponential_function a (-3)) : 
  0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_range_l2224_222418


namespace NUMINAMATH_GPT_sally_students_are_30_l2224_222487

-- Define the conditions given in the problem
def school_money : ℕ := 320
def book_cost : ℕ := 12
def sally_money : ℕ := 40
def total_students : ℕ := 30

-- Define the total amount Sally can spend on books
def total_amount_available : ℕ := school_money + sally_money

-- The total cost of books for S students
def total_cost (S : ℕ) : ℕ := book_cost * S

-- The main theorem stating that S students will cost the same as the amount Sally can spend
theorem sally_students_are_30 : total_cost 30 = total_amount_available :=
by
  sorry

end NUMINAMATH_GPT_sally_students_are_30_l2224_222487


namespace NUMINAMATH_GPT_parabola_opens_upward_l2224_222410

structure QuadraticFunction :=
  (a b c : ℝ)

def quadratic_y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

def points : List (ℝ × ℝ) :=
  [(-1, 10), (0, 5), (1, 2), (2, 1), (3, 2)]

theorem parabola_opens_upward (f : QuadraticFunction)
  (h_values : ∀ (x : ℝ), (x, quadratic_y f x) ∈ points) :
  f.a > 0 :=
sorry

end NUMINAMATH_GPT_parabola_opens_upward_l2224_222410


namespace NUMINAMATH_GPT_students_like_both_l2224_222481

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_like_mountains : ℕ := 289
def students_like_sea : ℕ := 337
def students_like_neither : ℕ := 56

-- Statement to prove
theorem students_like_both : 
  students_like_mountains + students_like_sea - 182 + students_like_neither = total_students := 
by
  sorry

end NUMINAMATH_GPT_students_like_both_l2224_222481


namespace NUMINAMATH_GPT_water_added_l2224_222404

-- Definitions and constants based on conditions
def initial_volume : ℝ := 80
def initial_jasmine_percentage : ℝ := 0.10
def jasmine_added : ℝ := 5
def final_jasmine_percentage : ℝ := 0.13

-- Problem statement
theorem water_added (W : ℝ) :
  (initial_volume * initial_jasmine_percentage + jasmine_added) / (initial_volume + jasmine_added + W) = final_jasmine_percentage → 
  W = 15 :=
by
  sorry

end NUMINAMATH_GPT_water_added_l2224_222404


namespace NUMINAMATH_GPT_letter_at_position_in_pattern_l2224_222430

/-- Determine the 150th letter in the repeating pattern XYZ is "Z"  -/
theorem letter_at_position_in_pattern :
  ∀ (pattern : List Char) (position : ℕ), pattern = ['X', 'Y', 'Z'] → position = 150 → pattern.get! ((position - 1) % pattern.length) = 'Z' :=
by
  intros pattern position
  intro hPattern hPosition
  rw [hPattern, hPosition]
  -- pattern = ['X', 'Y', 'Z'] and position = 150
  sorry

end NUMINAMATH_GPT_letter_at_position_in_pattern_l2224_222430


namespace NUMINAMATH_GPT_jersey_to_shoes_ratio_l2224_222486

theorem jersey_to_shoes_ratio
  (pairs_shoes: ℕ) (jerseys: ℕ) (total_cost: ℝ) (total_cost_shoes: ℝ) 
  (shoes: pairs_shoes = 6) (jer: jerseys = 4) (total: total_cost = 560) (cost_sh: total_cost_shoes = 480) :
  ((total_cost - total_cost_shoes) / jerseys) / (total_cost_shoes / pairs_shoes) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_jersey_to_shoes_ratio_l2224_222486


namespace NUMINAMATH_GPT_perpendicular_line_through_circle_center_l2224_222423

theorem perpendicular_line_through_circle_center :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (y = m * x + b) → (x = -1 ∧ y = 0) ) ∧ m = 1 ∧ b = 1 ∧ (∀ (x y : ℝ), (y = x + 1) → (x - y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_circle_center_l2224_222423


namespace NUMINAMATH_GPT_n_fifth_plus_4n_mod_5_l2224_222484

theorem n_fifth_plus_4n_mod_5 (n : ℕ) : (n^5 + 4 * n) % 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_n_fifth_plus_4n_mod_5_l2224_222484


namespace NUMINAMATH_GPT_domain_f_log_l2224_222441

noncomputable def domain_f (u : Real) : u ∈ Set.Icc (1 : Real) 2 := sorry

theorem domain_f_log (x : Real) : (x ∈ Set.Icc (4 : Real) 16) :=
by
  have h : ∀ x, (1 : Real) ≤ 2^x ∧ 2^x ≤ 2
  { intro x
    sorry }
  have h_log : ∀ x, 2 ≤ x ∧ x ≤ 4 
  { intro x
    sorry }
  have h_domain : ∀ x, 4 ≤ x ∧ x ≤ 16
  { intro x
    sorry }
  exact sorry

end NUMINAMATH_GPT_domain_f_log_l2224_222441


namespace NUMINAMATH_GPT_bill_bought_60_rats_l2224_222402

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end NUMINAMATH_GPT_bill_bought_60_rats_l2224_222402


namespace NUMINAMATH_GPT_hot_sauce_container_size_l2224_222414

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end NUMINAMATH_GPT_hot_sauce_container_size_l2224_222414


namespace NUMINAMATH_GPT_sam_memorized_digits_l2224_222447

theorem sam_memorized_digits (c s m : ℕ) 
  (h1 : s = c + 6) 
  (h2 : m = 6 * c)
  (h3 : m = 24) : 
  s = 10 :=
by
  sorry

end NUMINAMATH_GPT_sam_memorized_digits_l2224_222447


namespace NUMINAMATH_GPT_constant_expression_l2224_222495

variable {x y m n : ℝ}

theorem constant_expression (hx : x^2 = 25) (hy : ∀ y : ℝ, (x + y) * (x - 2 * y) - m * y * (n * x - y) = 25) :
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end NUMINAMATH_GPT_constant_expression_l2224_222495


namespace NUMINAMATH_GPT_intersection_A_B_l2224_222464

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { x | x < 0 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 0 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l2224_222464


namespace NUMINAMATH_GPT_unique_right_triangle_construction_l2224_222456

noncomputable def right_triangle_condition (c f : ℝ) : Prop :=
  f < c / 2

theorem unique_right_triangle_construction (c f : ℝ) (h_c : 0 < c) (h_f : 0 < f) :
  right_triangle_condition c f :=
  sorry

end NUMINAMATH_GPT_unique_right_triangle_construction_l2224_222456


namespace NUMINAMATH_GPT_remainder_abc_div9_l2224_222401

theorem remainder_abc_div9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) 
    (h1 : a + 2 * b + 3 * c ≡ 0 [MOD 9]) 
    (h2 : 2 * a + 3 * b + c ≡ 5 [MOD 9]) 
    (h3 : 3 * a + b + 2 * c ≡ 5 [MOD 9]) : 
    (a * b * c) % 9 = 0 := 
sorry

end NUMINAMATH_GPT_remainder_abc_div9_l2224_222401


namespace NUMINAMATH_GPT_camille_saw_31_birds_l2224_222475

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end NUMINAMATH_GPT_camille_saw_31_birds_l2224_222475


namespace NUMINAMATH_GPT_sum_h_k_a_b_l2224_222429

def h : ℤ := 3
def k : ℤ := -5
def a : ℤ := 7
def b : ℤ := 4

theorem sum_h_k_a_b : h + k + a + b = 9 := by
  sorry

end NUMINAMATH_GPT_sum_h_k_a_b_l2224_222429


namespace NUMINAMATH_GPT_initial_people_count_l2224_222419

theorem initial_people_count (x : ℕ) (h : (x - 2) + 2 = 10) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_count_l2224_222419


namespace NUMINAMATH_GPT_no_solution_inequalities_l2224_222413

theorem no_solution_inequalities (a : ℝ) : (¬ ∃ x : ℝ, 2 * x - 4 > 0 ∧ x - a < 0) → a ≤ 2 := 
by 
  sorry

end NUMINAMATH_GPT_no_solution_inequalities_l2224_222413


namespace NUMINAMATH_GPT_greatest_integer_function_of_pi_plus_3_l2224_222498

noncomputable def pi_plus_3 : Real := Real.pi + 3

theorem greatest_integer_function_of_pi_plus_3 : Int.floor pi_plus_3 = 6 := 
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_greatest_integer_function_of_pi_plus_3_l2224_222498


namespace NUMINAMATH_GPT_rains_at_least_once_l2224_222408

noncomputable def prob_rains_on_weekend : ℝ :=
  let prob_rain_saturday := 0.60
  let prob_rain_sunday := 0.70
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let independent_events := prob_no_rain_saturday * prob_no_rain_sunday
  1 - independent_events

theorem rains_at_least_once :
  prob_rains_on_weekend = 0.88 :=
by sorry

end NUMINAMATH_GPT_rains_at_least_once_l2224_222408


namespace NUMINAMATH_GPT_average_production_l2224_222493

theorem average_production (n : ℕ) (P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 62) : n = 14 :=
  sorry

end NUMINAMATH_GPT_average_production_l2224_222493


namespace NUMINAMATH_GPT_perimeter_of_park_is_66_l2224_222422

-- Given width and length of the flower bed
variables (w l : ℝ)
-- Given that the length is four times the width
variable (h1 : l = 4 * w)
-- Given the area of the flower bed
variable (h2 : l * w = 100)
-- Given the width of the walkway
variable (walkway_width : ℝ := 2)

-- The total width and length of the park, including the walkway
def w_park := w + 2 * walkway_width
def l_park := l + 2 * walkway_width

-- The proof statement: perimeter of the park equals 66 meters
theorem perimeter_of_park_is_66 :
  2 * (l_park + w_park) = 66 :=
by
  -- The full proof can be filled in here
  sorry

end NUMINAMATH_GPT_perimeter_of_park_is_66_l2224_222422


namespace NUMINAMATH_GPT_fraction_pow_rule_l2224_222499

theorem fraction_pow_rule :
  (5 / 7)^4 = 625 / 2401 :=
by
  sorry

end NUMINAMATH_GPT_fraction_pow_rule_l2224_222499


namespace NUMINAMATH_GPT_BD_range_l2224_222435

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) (BD : ℤ) :=
  AB = 7 ∧ BC = 15 ∧ CD = 7 ∧ DA = 11 ∧ (9 ≤ BD ∧ BD ≤ 17)

theorem BD_range : 
  ∀ (AB BC CD DA : ℕ) (BD : ℤ),
  quadrilateral_BD AB BC CD DA BD → 
  9 ≤ BD ∧ BD ≤ 17 :=
by
  intros AB BC CD DA BD h
  cases h
  -- We would then prove the conditions
  sorry

end NUMINAMATH_GPT_BD_range_l2224_222435


namespace NUMINAMATH_GPT_expand_expression_l2224_222476

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2224_222476


namespace NUMINAMATH_GPT_general_term_sequence_l2224_222488

-- Definition of the sequence conditions
def seq (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n + 1)

-- The main statement to be proved
theorem general_term_sequence (n : ℕ) : seq n = (-1)^(n+1) * (2 * n + 1) :=
sorry

end NUMINAMATH_GPT_general_term_sequence_l2224_222488


namespace NUMINAMATH_GPT_local_minimum_f_eval_integral_part_f_l2224_222428

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sin x * Real.sqrt (1 - Real.cos x))

theorem local_minimum_f :
  (0 < x) -> (x < π) -> f x >= 1 :=
  by sorry

theorem eval_integral_part_f :
  ∫ x in (↑(π / 2))..(↑(2 * π / 3)), f x = sorry :=
  by sorry

end NUMINAMATH_GPT_local_minimum_f_eval_integral_part_f_l2224_222428


namespace NUMINAMATH_GPT_problem_proof_l2224_222496

variable (a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ ((a + 1) * (b + 1) * (c + 1) = 8)

-- The proof problem
theorem problem_proof (h : conditions a b c) : a + b + c ≥ 3 ∧ a * b * c ≤ 1 :=
  sorry

end NUMINAMATH_GPT_problem_proof_l2224_222496


namespace NUMINAMATH_GPT_min_x_plus_y_l2224_222477

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 := by
  sorry

end NUMINAMATH_GPT_min_x_plus_y_l2224_222477


namespace NUMINAMATH_GPT_pizza_problem_l2224_222466

theorem pizza_problem (m d : ℕ) :
  (7 * m + 2 * d > 36) ∧ (8 * m + 4 * d < 48) ↔ (m = 5) ∧ (d = 1) := by
  sorry

end NUMINAMATH_GPT_pizza_problem_l2224_222466


namespace NUMINAMATH_GPT_joshua_additional_cents_needed_l2224_222445

def cost_of_pen_cents : ℕ := 600
def money_joshua_has_cents : ℕ := 500
def money_borrowed_cents : ℕ := 68

def additional_cents_needed (cost money has borrowed : ℕ) : ℕ :=
  cost - (has + borrowed)

theorem joshua_additional_cents_needed :
  additional_cents_needed cost_of_pen_cents money_joshua_has_cents money_borrowed_cents = 32 :=
by
  sorry

end NUMINAMATH_GPT_joshua_additional_cents_needed_l2224_222445


namespace NUMINAMATH_GPT_draw_at_least_one_red_card_l2224_222491

-- Define the deck and properties
def total_cards := 52
def red_cards := 26
def black_cards := 26

-- Define the calculation for drawing three cards sequentially
def total_ways_draw3 := total_cards * (total_cards - 1) * (total_cards - 2)
def black_only_ways_draw3 := black_cards * (black_cards - 1) * (black_cards - 2)

-- Define the main proof statement
theorem draw_at_least_one_red_card : 
    total_ways_draw3 - black_only_ways_draw3 = 117000 := by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_draw_at_least_one_red_card_l2224_222491


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l2224_222412

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l2224_222412


namespace NUMINAMATH_GPT_quadratic_root_3_m_value_l2224_222405

theorem quadratic_root_3_m_value (m : ℝ) : (∃ x : ℝ, 2*x*x - m*x + 3 = 0 ∧ x = 3) → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_3_m_value_l2224_222405


namespace NUMINAMATH_GPT_overall_sale_price_per_kg_l2224_222449

-- Defining the quantities and prices
def tea_A_quantity : ℝ := 80
def tea_A_cost_per_kg : ℝ := 15
def tea_B_quantity : ℝ := 20
def tea_B_cost_per_kg : ℝ := 20
def tea_C_quantity : ℝ := 50
def tea_C_cost_per_kg : ℝ := 25
def tea_D_quantity : ℝ := 40
def tea_D_cost_per_kg : ℝ := 30

-- Defining the profit percentages
def tea_A_profit_percentage : ℝ := 0.30
def tea_B_profit_percentage : ℝ := 0.25
def tea_C_profit_percentage : ℝ := 0.20
def tea_D_profit_percentage : ℝ := 0.15

-- Desired sale price per kg
theorem overall_sale_price_per_kg : 
  (tea_A_quantity * tea_A_cost_per_kg * (1 + tea_A_profit_percentage) +
   tea_B_quantity * tea_B_cost_per_kg * (1 + tea_B_profit_percentage) +
   tea_C_quantity * tea_C_cost_per_kg * (1 + tea_C_profit_percentage) +
   tea_D_quantity * tea_D_cost_per_kg * (1 + tea_D_profit_percentage)) / 
  (tea_A_quantity + tea_B_quantity + tea_C_quantity + tea_D_quantity) = 26 := 
by
  sorry

end NUMINAMATH_GPT_overall_sale_price_per_kg_l2224_222449


namespace NUMINAMATH_GPT_candy_factory_days_l2224_222439

noncomputable def candies_per_hour := 50
noncomputable def total_candies := 4000
noncomputable def working_hours_per_day := 10
noncomputable def total_hours_needed := total_candies / candies_per_hour
noncomputable def total_days_needed := total_hours_needed / working_hours_per_day

theorem candy_factory_days :
  total_days_needed = 8 := 
by
  -- (Proof steps will be filled here)
  sorry

end NUMINAMATH_GPT_candy_factory_days_l2224_222439


namespace NUMINAMATH_GPT_find_m_and_c_l2224_222483

-- Definitions & conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 3 }
def B (m : ℝ) : Point := { x := -6, y := m }

def line (c : ℝ) (p : Point) : Prop := p.x + p.y + c = 0

-- Theorem statement
theorem find_m_and_c (m : ℝ) (c : ℝ) (hc : line c A) (hcB : line c (B m)) :
  m = 3 ∧ c = -2 :=
  by
  sorry

end NUMINAMATH_GPT_find_m_and_c_l2224_222483


namespace NUMINAMATH_GPT_find_vertex_angle_of_cone_l2224_222490

noncomputable def vertexAngleCone (r1 r2 : ℝ) (O1 O2 : ℝ) (touching : Prop) (Ctable : Prop) (equalAngles : Prop) : Prop :=
  -- The given conditions:
  -- r1, r2 are the radii of the spheres, where r1 = 4 and r2 = 1.
  -- O1, O2 are the centers of the spheres.
  -- touching indicates the spheres touch externally.
  -- Ctable indicates that vertex C of the cone is on the segment connecting the points where the spheres touch the table.
  -- equalAngles indicates that the rays CO1 and CO2 form equal angles with the table.
  touching → 
  Ctable → 
  equalAngles →
  -- The target to prove:
  ∃ α : ℝ, 2 * α = 2 * Real.arctan (2 / 5)

theorem find_vertex_angle_of_cone (r1 r2 : ℝ) (O1 O2 : ℝ) :
  let touching : Prop := (r1 = 4 ∧ r2 = 1 ∧ abs (O1 - O2) = r1 + r2)
  let Ctable : Prop := (True)  -- Provided by problem conditions, details can be expanded
  let equalAngles : Prop := (True)  
  vertexAngleCone r1 r2 O1 O2 touching Ctable equalAngles := 
by
  sorry

end NUMINAMATH_GPT_find_vertex_angle_of_cone_l2224_222490


namespace NUMINAMATH_GPT_gcd_lcm_mul_l2224_222424

theorem gcd_lcm_mul (a b : ℤ) : (Int.gcd a b) * (Int.lcm a b) = a * b := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_mul_l2224_222424


namespace NUMINAMATH_GPT_average_of_r_s_t_l2224_222454

theorem average_of_r_s_t
  (r s t : ℝ)
  (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_r_s_t_l2224_222454


namespace NUMINAMATH_GPT_no_solution_exists_l2224_222457

def product_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else (x / 10) * (x % 10)

theorem no_solution_exists :
  ¬ ∃ x : ℕ, product_of_digits x = x^2 - 10 * x - 22 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l2224_222457


namespace NUMINAMATH_GPT_range_of_m_l2224_222471

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → (-2 ≤ x → x ≤ 10))
    (h2 : ∀ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 → (1 - m ≤ x → x ≤ 1 + m))
    (h3 : m > 0)
    (h4 : ∀ x : ℝ, ¬ ((x^2 + 1) * (x^2 - 8*x - 20) ≤ 0) → ¬ (x^2 - 2*x + 1 - m^2 ≤ 0) → (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
  m ≥ 9 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2224_222471


namespace NUMINAMATH_GPT_ab_minus_a_inv_b_l2224_222448

theorem ab_minus_a_inv_b (a : ℝ) (b : ℚ) (h1 : a > 1) (h2 : 0 < (b : ℝ)) (h3 : (a ^ (b : ℝ)) + (a ^ (-(b : ℝ))) = 2 * Real.sqrt 2) :
  (a ^ (b : ℝ)) - (a ^ (-(b : ℝ))) = 2 := 
sorry

end NUMINAMATH_GPT_ab_minus_a_inv_b_l2224_222448


namespace NUMINAMATH_GPT_apples_total_l2224_222458

theorem apples_total :
  ∀ (Marin David Amanda : ℕ),
  Marin = 6 →
  David = 2 * Marin →
  Amanda = David + 5 →
  Marin + David + Amanda = 35 :=
by
  intros Marin David Amanda hMarin hDavid hAmanda
  sorry

end NUMINAMATH_GPT_apples_total_l2224_222458


namespace NUMINAMATH_GPT_cuboid_height_l2224_222443

-- Given conditions
def volume_cuboid : ℝ := 1380 -- cubic meters
def base_area_cuboid : ℝ := 115 -- square meters

-- Prove that the height of the cuboid is 12 meters
theorem cuboid_height : volume_cuboid / base_area_cuboid = 12 := by
  sorry

end NUMINAMATH_GPT_cuboid_height_l2224_222443


namespace NUMINAMATH_GPT_exist_elements_inequality_l2224_222409

open Set

theorem exist_elements_inequality (A : Set ℝ) (a_1 a_2 a_3 a_4 : ℝ)
(hA : A = {a_1, a_2, a_3, a_4})
(h_ineq1 : 0 < a_1 )
(h_ineq2 : a_1 < a_2 )
(h_ineq3 : a_2 < a_3 )
(h_ineq4 : a_3 < a_4 ) :
∃ (x y : ℝ), x ∈ A ∧ y ∈ A ∧ (2 + Real.sqrt 3) * |x - y| < (x + 1) * (y + 1) + x * y := 
sorry

end NUMINAMATH_GPT_exist_elements_inequality_l2224_222409


namespace NUMINAMATH_GPT_part1a_part1b_part2_part3_l2224_222463

-- Definitions for the sequences in columns ①, ②, and ③
def col1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1)
def col2 (n : ℕ) : ℤ := ((-1 : ℤ) ^ n * (2 * n - 1)) - 2
def col3 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1) * 3

-- Problem statements
theorem part1a : col1 10 = 19 :=
sorry

theorem part1b : col2 15 = -31 :=
sorry

theorem part2 : ¬ ∃ n : ℕ, col2 (n - 1) + col2 n + col2 (n + 1) = 1001 :=
sorry

theorem part3 : ∃ k : ℕ, col1 k + col2 k + col3 k = 599 ∧ k = 301 :=
sorry

end NUMINAMATH_GPT_part1a_part1b_part2_part3_l2224_222463


namespace NUMINAMATH_GPT_wrench_force_inv_proportional_l2224_222436

theorem wrench_force_inv_proportional (F₁ : ℝ) (L₁ : ℝ) (F₂ : ℝ) (L₂ : ℝ) (k : ℝ)
  (h₁ : F₁ * L₁ = k) (h₂ : L₁ = 12) (h₃ : F₁ = 300) (h₄ : L₂ = 18) :
  F₂ = 200 :=
by
  sorry

end NUMINAMATH_GPT_wrench_force_inv_proportional_l2224_222436


namespace NUMINAMATH_GPT_sum_abs_coeffs_l2224_222467

theorem sum_abs_coeffs (a : ℝ → ℝ) :
  (∀ x, (1 - 3 * x)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| = 4^9 := by
  sorry

end NUMINAMATH_GPT_sum_abs_coeffs_l2224_222467


namespace NUMINAMATH_GPT_calculate_expression_l2224_222425

theorem calculate_expression : ((-3: ℤ) ^ 3 + (5: ℤ) ^ 2 - ((-2: ℤ) ^ 2)) = -6 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2224_222425
