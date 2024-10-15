import Mathlib

namespace NUMINAMATH_GPT_not_super_lucky_years_l797_79793

def sum_of_month_and_day (m d : ℕ) : ℕ := m + d
def product_of_month_and_day (m d : ℕ) : ℕ := m * d
def sum_of_last_two_digits (y : ℕ) : ℕ :=
  let d1 := y / 10 % 10
  let d2 := y % 10
  d1 + d2

def is_super_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), sum_of_month_and_day m d = 24 ∧
               product_of_month_and_day m d = 2 * sum_of_last_two_digits y

theorem not_super_lucky_years :
  ¬ is_super_lucky_year 2070 ∧
  ¬ is_super_lucky_year 2081 ∧
  ¬ is_super_lucky_year 2092 :=
by {
  sorry
}

end NUMINAMATH_GPT_not_super_lucky_years_l797_79793


namespace NUMINAMATH_GPT_intersection_point_exists_l797_79727

def h : ℝ → ℝ := sorry  -- placeholder for the function h
def j : ℝ → ℝ := sorry  -- placeholder for the function j

-- Conditions
axiom h_3_eq : h 3 = 3
axiom j_3_eq : j 3 = 3
axiom h_6_eq : h 6 = 9
axiom j_6_eq : j 6 = 9
axiom h_9_eq : h 9 = 18
axiom j_9_eq : j 9 = 18

-- Theorem
theorem intersection_point_exists :
  ∃ a b : ℝ, a = 2 ∧ h (3 * a) = 3 * j (a) ∧ h (3 * a) = b ∧ 3 * j (a) = b ∧ a + b = 11 :=
  sorry

end NUMINAMATH_GPT_intersection_point_exists_l797_79727


namespace NUMINAMATH_GPT_part_a_part_b_l797_79745

def good (p q n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def bad (p q n : ℕ) : Prop := 
  ¬ good p q n

theorem part_a (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ A, A = p * q - p - q ∧ ∀ x y, x + y = A → (good p q x ∧ bad p q y) ∨ (bad p q x ∧ good p q y) := by
  sorry

theorem part_b (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ N, N = (p - 1) * (q - 1) / 2 ∧ ∀ n, n < p * q - p - q → bad p q n :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_l797_79745


namespace NUMINAMATH_GPT_kylie_gave_21_coins_to_Laura_l797_79789

def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_left : ℕ := 15

def total_coins_collected : ℕ := coins_from_piggy_bank + coins_from_brother + coins_from_father
def coins_given_to_Laura : ℕ := total_coins_collected - coins_left

theorem kylie_gave_21_coins_to_Laura :
  coins_given_to_Laura = 21 :=
by
  sorry

end NUMINAMATH_GPT_kylie_gave_21_coins_to_Laura_l797_79789


namespace NUMINAMATH_GPT_find_integer_pair_l797_79753

theorem find_integer_pair (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 → (x = -1 ∧ y = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_integer_pair_l797_79753


namespace NUMINAMATH_GPT_average_homework_time_decrease_l797_79725

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end NUMINAMATH_GPT_average_homework_time_decrease_l797_79725


namespace NUMINAMATH_GPT_monotonicity_of_f_on_interval_l797_79702

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x - 2

theorem monotonicity_of_f_on_interval (a b : ℝ) (h1 : a = -3) (h2 : b = 0) :
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → f x1 a b ≥ f x2 a b := 
by
  sorry

end NUMINAMATH_GPT_monotonicity_of_f_on_interval_l797_79702


namespace NUMINAMATH_GPT_count_perfect_squares_between_l797_79755

theorem count_perfect_squares_between :
  let n := 8
  let m := 70
  (m - n + 1) = 64 :=
by
  -- Definitions and step-by-step proof would go here.
  sorry

end NUMINAMATH_GPT_count_perfect_squares_between_l797_79755


namespace NUMINAMATH_GPT_functional_equation_solution_l797_79787

theorem functional_equation_solution:
  (∀ f : ℝ → ℝ, (∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x * y * z = 1 →
  f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)) →
  (∀ x : ℝ, x ≠ 0 → ( (f x = x^2 - 1/x) ∨ (f x = 0)))) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l797_79787


namespace NUMINAMATH_GPT_rectangle_short_side_l797_79780

theorem rectangle_short_side
  (r : ℝ) (a_circle : ℝ) (a_rect : ℝ) (d : ℝ) (other_side : ℝ) :
  r = 6 →
  a_circle = Real.pi * r^2 →
  a_rect = 3 * a_circle →
  d = 2 * r →
  a_rect = d * other_side →
  other_side = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_rectangle_short_side_l797_79780


namespace NUMINAMATH_GPT_highest_possible_rubidium_concentration_l797_79743

noncomputable def max_rubidium_concentration (R C F : ℝ) : Prop :=
  (R + C + F > 0) →
  (0.10 * R + 0.08 * C + 0.05 * F) / (R + C + F) = 0.07 ∧
  (0.05 * F) / (R + C + F) ≤ 0.02 →
  (0.10 * R) / (R + C + F) = 0.01

theorem highest_possible_rubidium_concentration :
  ∃ R C F : ℝ, max_rubidium_concentration R C F :=
sorry

end NUMINAMATH_GPT_highest_possible_rubidium_concentration_l797_79743


namespace NUMINAMATH_GPT_brian_fewer_seashells_l797_79704

-- Define the conditions
def cb_ratio (Craig Brian : ℕ) : Prop := 9 * Brian = 7 * Craig
def craig_seashells (Craig : ℕ) : Prop := Craig = 54

-- Define the main theorem to be proven
theorem brian_fewer_seashells (Craig Brian : ℕ) (h1 : cb_ratio Craig Brian) (h2 : craig_seashells Craig) : Craig - Brian = 12 :=
by
  sorry

end NUMINAMATH_GPT_brian_fewer_seashells_l797_79704


namespace NUMINAMATH_GPT_find_f3_l797_79738

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 4)
  (h2 : f 2 = 10)
  (h3 : ∀ x, f x = a * x^2 + b * x + 2) :
  f 3 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_f3_l797_79738


namespace NUMINAMATH_GPT_lcm_18_30_is_90_l797_79779

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end NUMINAMATH_GPT_lcm_18_30_is_90_l797_79779


namespace NUMINAMATH_GPT_find_ratio_l797_79770

variable {x y k x1 x2 y1 y2 : ℝ}

-- Inverse proportionality
def inverse_proportional (x y k : ℝ) : Prop := x * y = k

-- Given conditions
axiom h1 : inverse_proportional x1 y1 k
axiom h2 : inverse_proportional x2 y2 k
axiom h3 : x1 ≠ 0
axiom h4 : x2 ≠ 0
axiom h5 : y1 ≠ 0
axiom h6 : y2 ≠ 0
axiom h7 : x1 / x2 = 3 / 4

theorem find_ratio : y1 / y2 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l797_79770


namespace NUMINAMATH_GPT_number_of_neither_l797_79720

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def both_drinkers : ℕ := 6

theorem number_of_neither (total_businessmen coffee_drinkers tea_drinkers both_drinkers : ℕ) : 
  coffee_drinkers = 15 ∧ 
  tea_drinkers = 12 ∧ 
  both_drinkers = 6 ∧ 
  total_businessmen = 30 → 
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_neither_l797_79720


namespace NUMINAMATH_GPT_real_part_of_complex_l797_79741

theorem real_part_of_complex (z : ℂ) (h : i * (z + 1) = -3 + 2 * i) : z.re = 1 :=
sorry

end NUMINAMATH_GPT_real_part_of_complex_l797_79741


namespace NUMINAMATH_GPT_arctan_sum_pi_over_two_l797_79747

theorem arctan_sum_pi_over_two : 
  Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
by sorry

end NUMINAMATH_GPT_arctan_sum_pi_over_two_l797_79747


namespace NUMINAMATH_GPT_find_a_minimum_value_at_x_2_l797_79798

def f (x a : ℝ) := x^3 - a * x

theorem find_a_minimum_value_at_x_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y ≠ 2, f y a ≥ f 2 a) → a = 12 :=
by 
  -- Here we should include the proof steps
  sorry

end NUMINAMATH_GPT_find_a_minimum_value_at_x_2_l797_79798


namespace NUMINAMATH_GPT_length_AE_l797_79775

structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt (((Q.x - P.x : ℝ) ^ 2) + ((Q.y - P.y : ℝ) ^ 2))

noncomputable def AE_length : ℝ :=
  (5 * (dist A B)) / 9

theorem length_AE :
  ∃ E : Point, AE_length = (5 * Real.sqrt 65) / 9 := by
  sorry

end NUMINAMATH_GPT_length_AE_l797_79775


namespace NUMINAMATH_GPT_alex_guarantees_victory_with_52_bullseyes_l797_79768

variable (m : ℕ) -- total score of Alex after the first half
variable (opponent_score : ℕ) -- total score of opponent after the first half
variable (remaining_shots : ℕ := 60) -- shots remaining for both players

-- Assume Alex always scores at least 3 points per shot and a bullseye earns 10 points
def min_bullseyes_to_guarantee_victory (m opponent_score : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 52 ∧
  (m + 7 * n + 180) > (opponent_score + 540)

-- Statement: Prove that if Alex leads by 60 points halfway through, then the minimum number of bullseyes he needs to guarantee a win is 52.
theorem alex_guarantees_victory_with_52_bullseyes (m opponent_score : ℕ) :
  m >= opponent_score + 60 → min_bullseyes_to_guarantee_victory m opponent_score :=
  sorry

end NUMINAMATH_GPT_alex_guarantees_victory_with_52_bullseyes_l797_79768


namespace NUMINAMATH_GPT_remainder_of_sum_mod_13_l797_79783

theorem remainder_of_sum_mod_13 {a b c d e : ℕ} 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7) 
  (h4 : d % 13 = 9) 
  (h5 : e % 13 = 11) : 
  (a + b + c + d + e) % 13 = 9 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_13_l797_79783


namespace NUMINAMATH_GPT_find_divisor_l797_79746

theorem find_divisor (x d : ℤ) (h1 : ∃ k : ℤ, x = k * d + 5)
                     (h2 : ∃ n : ℤ, x + 17 = n * 41 + 22) :
    d = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l797_79746


namespace NUMINAMATH_GPT_same_remainder_division_l797_79733

theorem same_remainder_division (k r a b c d : ℕ) 
  (h_k_pos : 0 < k)
  (h_nonzero_r : 0 < r)
  (h_r_lt_k : r < k)
  (a_def : a = 2613)
  (b_def : b = 2243)
  (c_def : c = 1503)
  (d_def : d = 985)
  (h_a : a % k = r)
  (h_b : b % k = r)
  (h_c : c % k = r)
  (h_d : d % k = r) : 
  k = 74 ∧ r = 23 := 
by
  sorry

end NUMINAMATH_GPT_same_remainder_division_l797_79733


namespace NUMINAMATH_GPT_angle_B_in_triangle_l797_79788

theorem angle_B_in_triangle (a b c : ℝ) (B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
  B = 60 ∨ B = 120 := 
sorry

end NUMINAMATH_GPT_angle_B_in_triangle_l797_79788


namespace NUMINAMATH_GPT_fraction_relation_l797_79726

theorem fraction_relation 
  (m n p q : ℚ)
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) : 
  m / q = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_fraction_relation_l797_79726


namespace NUMINAMATH_GPT_rational_solution_system_l797_79760

theorem rational_solution_system (x y z t w : ℚ) :
  (t^2 - w^2 + z^2 = 2 * x * y) →
  (t^2 - y^2 + w^2 = 2 * x * z) →
  (t^2 - w^2 + x^2 = 2 * y * z) →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_rational_solution_system_l797_79760


namespace NUMINAMATH_GPT_sin_double_angle_l797_79769

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.sin (2 * θ) = 3 / 5 := 
by 
sorry

end NUMINAMATH_GPT_sin_double_angle_l797_79769


namespace NUMINAMATH_GPT_money_increase_factor_two_years_l797_79721

theorem money_increase_factor_two_years (P : ℝ) (rate : ℝ) (n : ℕ)
  (h_rate : rate = 0.50) (h_n : n = 2) :
  (P * (1 + rate) ^ n) = 2.25 * P :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_money_increase_factor_two_years_l797_79721


namespace NUMINAMATH_GPT_history_paper_pages_l797_79799

theorem history_paper_pages (days: ℕ) (pages_per_day: ℕ) (h₁: days = 3) (h₂: pages_per_day = 27) : days * pages_per_day = 81 := 
by
  sorry

end NUMINAMATH_GPT_history_paper_pages_l797_79799


namespace NUMINAMATH_GPT_total_people_in_tour_group_l797_79794

noncomputable def tour_group_total_people (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) : Prop :=
  (older_people_percentage = (θ + 9) / 3.6) ∧
  (young_adults_percentage = (θ + 27) / 3.6) ∧
  (N * young_adults_percentage / 100 = N * children_percentage / 100 + 9) ∧
  (children_percentage = θ / 3.6) →
  N = 120

theorem total_people_in_tour_group (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) :
  tour_group_total_people θ N children_percentage young_adults_percentage older_people_percentage :=
sorry

end NUMINAMATH_GPT_total_people_in_tour_group_l797_79794


namespace NUMINAMATH_GPT_Jacob_fill_tank_in_206_days_l797_79737

noncomputable def tank_capacity : ℕ := 350 * 1000
def rain_collection : ℕ := 500
def river_collection : ℕ := 1200
def daily_collection : ℕ := rain_collection + river_collection
def required_days (C R r : ℕ) : ℕ := (C + (R + r) - 1) / (R + r)

theorem Jacob_fill_tank_in_206_days :
  required_days tank_capacity rain_collection river_collection = 206 :=
by 
  sorry

end NUMINAMATH_GPT_Jacob_fill_tank_in_206_days_l797_79737


namespace NUMINAMATH_GPT_finite_negatives_condition_l797_79734

-- Define the sequence terms
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + n * d

-- Define the condition for finite negative terms
def has_finite_negatives (a1 d : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, arithmetic_seq a1 d n ≥ 0

-- Theorem that proves the desired statement
theorem finite_negatives_condition (a1 d : ℝ) (h1 : a1 < 0) (h2 : d > 0) :
  has_finite_negatives a1 d :=
sorry

end NUMINAMATH_GPT_finite_negatives_condition_l797_79734


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l797_79785

-- Given conditions and translated inequalities
variable {x : ℝ}
variable (h_pos : 0 < x) (h_bound : x < π / 2)
variable (h_sin_pos : 0 < Real.sin x) (h_sin_bound : Real.sin x < 1)

-- Define the inequalities we are dealing with
def ineq_1 (x : ℝ) := Real.sqrt x - 1 / Real.sin x < 0
def ineq_2 (x : ℝ) := 1 / Real.sin x - x > 0

-- The main proof statement
theorem necessary_but_not_sufficient_condition 
  (h1 : ineq_1 x) 
  (hx : 0 < x) (hπ : x < π/2) : 
  ineq_2 x → False := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l797_79785


namespace NUMINAMATH_GPT_factor_difference_of_squares_l797_79764

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l797_79764


namespace NUMINAMATH_GPT_packs_per_box_l797_79776

theorem packs_per_box (total_cost : ℝ) (num_boxes : ℕ) (cost_per_pack : ℝ) 
  (num_tissues_per_pack : ℕ) (cost_per_tissue : ℝ) (total_packs : ℕ) :
  total_cost = 1000 ∧ num_boxes = 10 ∧ cost_per_pack = num_tissues_per_pack * cost_per_tissue ∧ 
  num_tissues_per_pack = 100 ∧ cost_per_tissue = 0.05 ∧ total_packs * cost_per_pack = total_cost / num_boxes →
  total_packs = 20 :=
by
  sorry

end NUMINAMATH_GPT_packs_per_box_l797_79776


namespace NUMINAMATH_GPT_length_of_third_side_l797_79709

-- Define the properties and setup for the problem
variables {a b : ℝ} (h1 : a = 4) (h2 : b = 8)

-- Define the condition for an isosceles triangle
def isosceles_triangle (x y z : ℝ) : Prop :=
  (x = y ∧ x ≠ z) ∨ (x = z ∧ x ≠ y) ∨ (y = z ∧ y ≠ x)

-- Define the condition for a valid triangle
def valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- State the theorem to be proved
theorem length_of_third_side (c : ℝ) (h : isosceles_triangle a b c ∧ valid_triangle a b c) : c = 8 :=
sorry

end NUMINAMATH_GPT_length_of_third_side_l797_79709


namespace NUMINAMATH_GPT_range_of_a_for_integer_solutions_l797_79792

theorem range_of_a_for_integer_solutions (a : ℝ) :
  (∃ x : ℤ, (a - 2 < x ∧ x ≤ 3)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_integer_solutions_l797_79792


namespace NUMINAMATH_GPT_line_through_origin_l797_79765

theorem line_through_origin (x y : ℝ) :
  (∃ x0 y0 : ℝ, 4 * x0 + y0 + 6 = 0 ∧ 3 * (-x0) + (- 5) * y0 + 6 = 0)
  → (x + 6 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_origin_l797_79765


namespace NUMINAMATH_GPT_energy_equivalence_l797_79748

def solar_energy_per_sqm := 1.3 * 10^8
def china_land_area := 9.6 * 10^6
def expected_coal_energy := 1.248 * 10^15

theorem energy_equivalence : 
  solar_energy_per_sqm * china_land_area = expected_coal_energy := 
by
  sorry

end NUMINAMATH_GPT_energy_equivalence_l797_79748


namespace NUMINAMATH_GPT_rectangular_to_polar_coordinates_l797_79762

theorem rectangular_to_polar_coordinates :
  ∀ (x y : ℝ) (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) →
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) →
  (r, θ) = (4, Real.pi / 4) :=
by
  intros x y r θ h1 h2 h3
  sorry

end NUMINAMATH_GPT_rectangular_to_polar_coordinates_l797_79762


namespace NUMINAMATH_GPT_sum_of_variables_l797_79716

theorem sum_of_variables (a b c d : ℝ) (h₁ : a * c + a * d + b * c + b * d = 68) (h₂ : c + d = 4) : a + b + c + d = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_variables_l797_79716


namespace NUMINAMATH_GPT_minimum_breaks_l797_79717

-- Definitions based on conditions given in the problem statement
def longitudinal_grooves : ℕ := 2
def transverse_grooves : ℕ := 3

-- The problem statement to be proved
theorem minimum_breaks (l t : ℕ) (hl : l = longitudinal_grooves) (ht : t = transverse_grooves) :
  l + t = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_breaks_l797_79717


namespace NUMINAMATH_GPT_mark_score_is_46_l797_79790

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end NUMINAMATH_GPT_mark_score_is_46_l797_79790


namespace NUMINAMATH_GPT_perimeter_of_triangle_is_13_l797_79744

-- Conditions
noncomputable def perimeter_of_triangle_with_two_sides_and_third_root_of_eq : ℝ :=
  let a := 3
  let b := 6
  let c1 := 2 -- One root of the equation x^2 - 6x + 8 = 0
  let c2 := 4 -- Another root of the equation x^2 - 6x + 8 = 0
  if a + b > c2 ∧ a + c2 > b ∧ b + c2 > a then
    a + b + c2
  else
    0 -- not possible to form a triangle with these sides

-- Assertion
theorem perimeter_of_triangle_is_13 :
  perimeter_of_triangle_with_two_sides_and_third_root_of_eq = 13 := 
sorry

end NUMINAMATH_GPT_perimeter_of_triangle_is_13_l797_79744


namespace NUMINAMATH_GPT_frustum_volume_fraction_l797_79772

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

noncomputable def fraction_of_frustum (base_edge height : ℝ) : ℝ :=
  let original_volume := volume_pyramid base_edge height
  let smaller_volume := volume_pyramid (base_edge / 5) (height / 5)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume

theorem frustum_volume_fraction :
  fraction_of_frustum 40 20 = 63 / 64 :=
by sorry

end NUMINAMATH_GPT_frustum_volume_fraction_l797_79772


namespace NUMINAMATH_GPT_value_of_P_2017_l797_79731

theorem value_of_P_2017 (a b c : ℝ) (h_distinct: a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c)
    (p : ℝ → ℝ) :
    (∀ x, p x = (c * (x - a) * (x - b) / ((c - a) * (c - b))) + (a * (x - b) * (x - c) / ((a - b) * (a - c))) + (b * (x - c) * (x - a) / ((b - c) * (b - a))) + 1) →
    p 2017 = 2 :=
sorry

end NUMINAMATH_GPT_value_of_P_2017_l797_79731


namespace NUMINAMATH_GPT_number_of_intersection_points_l797_79742

theorem number_of_intersection_points (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃ x : Finset ℝ, (∀ y ∈ x, f ((y:ℝ)^2) = f ((y:ℝ)^6)) ∧ x.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_intersection_points_l797_79742


namespace NUMINAMATH_GPT_sorting_five_rounds_l797_79784

def direct_sorting_method (l : List ℕ) : List ℕ := sorry

theorem sorting_five_rounds (initial_seq : List ℕ) :
  initial_seq = [49, 38, 65, 97, 76, 13, 27] →
  (direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method) initial_seq = [97, 76, 65, 49, 38, 13, 27] :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sorting_five_rounds_l797_79784


namespace NUMINAMATH_GPT_remainder_when_divided_82_l797_79781

theorem remainder_when_divided_82 (x : ℤ) (k m : ℤ) (R : ℤ) (h1 : 0 ≤ R) (h2 : R < 82)
    (h3 : x = 82 * k + R) (h4 : x + 7 = 41 * m + 12) : R = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_82_l797_79781


namespace NUMINAMATH_GPT_external_angle_theorem_proof_l797_79766

theorem external_angle_theorem_proof
    (x : ℝ)
    (FAB : ℝ)
    (BCA : ℝ)
    (ABC : ℝ)
    (h1 : FAB = 70)
    (h2 : BCA = 20 + x)
    (h3 : ABC = x + 20)
    (h4 : FAB = ABC + BCA) : 
    x = 15 :=
  by
  sorry

end NUMINAMATH_GPT_external_angle_theorem_proof_l797_79766


namespace NUMINAMATH_GPT_map_to_actual_distance_ratio_l797_79750

def distance_in_meters : ℝ := 250
def distance_on_map_cm : ℝ := 5
def cm_per_meter : ℝ := 100

theorem map_to_actual_distance_ratio :
  distance_on_map_cm / (distance_in_meters * cm_per_meter) = 1 / 5000 :=
by
  sorry

end NUMINAMATH_GPT_map_to_actual_distance_ratio_l797_79750


namespace NUMINAMATH_GPT_jill_age_l797_79751

theorem jill_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : J = 16 :=
by
  sorry

end NUMINAMATH_GPT_jill_age_l797_79751


namespace NUMINAMATH_GPT_car_p_less_hours_l797_79757

theorem car_p_less_hours (distance : ℕ) (speed_r : ℕ) (speed_p : ℕ) (time_r : ℕ) (time_p : ℕ) (h1 : distance = 600) (h2 : speed_r = 50) (h3 : speed_p = speed_r + 10) (h4 : time_r = distance / speed_r) (h5 : time_p = distance / speed_p) : time_r - time_p = 2 := 
by
  sorry

end NUMINAMATH_GPT_car_p_less_hours_l797_79757


namespace NUMINAMATH_GPT_weekly_diesel_spending_l797_79736

-- Conditions
def cost_per_gallon : ℝ := 3
def fuel_used_in_two_weeks : ℝ := 24

-- Question: Prove that Mr. Alvarez spends $36 on diesel fuel each week.
theorem weekly_diesel_spending : (fuel_used_in_two_weeks / 2) * cost_per_gallon = 36 := by
  sorry

end NUMINAMATH_GPT_weekly_diesel_spending_l797_79736


namespace NUMINAMATH_GPT_only_n_equal_1_l797_79732

theorem only_n_equal_1 (n : ℕ) (h : n ≥ 1) : Nat.Prime (9^n - 2^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_only_n_equal_1_l797_79732


namespace NUMINAMATH_GPT_original_price_l797_79715

variable (P SP : ℝ)

axiom condition1 : SP = 0.8 * P
axiom condition2 : SP = 480

theorem original_price : P = 600 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l797_79715


namespace NUMINAMATH_GPT_probability_first_player_takes_card_l797_79713

variable (n : ℕ) (i : ℕ)

-- Conditions
def even_n : Prop := ∃ k, n = 2 * k
def valid_i : Prop := 1 ≤ i ∧ i ≤ n

-- The key function (probability) and theorem to prove
def P (i n : ℕ) : ℚ := (i - 1) / (n - 1)

theorem probability_first_player_takes_card :
  even_n n → valid_i n i → P i n = (i - 1) / (n - 1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_probability_first_player_takes_card_l797_79713


namespace NUMINAMATH_GPT_number_of_cats_l797_79756

def cats_on_ship (C S : ℕ) : Prop :=
  (C + S + 2 = 16) ∧ (4 * C + 2 * S + 3 = 45)

theorem number_of_cats (C S : ℕ) (h : cats_on_ship C S) : C = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cats_l797_79756


namespace NUMINAMATH_GPT_hyperbola_vertex_distance_l797_79706

open Real

/-- The distance between the vertices of the hyperbola represented by the equation
    (y-4)^2 / 32 - (x+3)^2 / 18 = 1 is 8√2. -/
theorem hyperbola_vertex_distance :
  let a := sqrt 32
  2 * a = 8 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_vertex_distance_l797_79706


namespace NUMINAMATH_GPT_coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l797_79701

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l797_79701


namespace NUMINAMATH_GPT_complement_of_intersection_l797_79773

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = -x^2}

theorem complement_of_intersection :
  (Set.compl (A ∩ B) = {x | x < -2 ∨ x > 0 }) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l797_79773


namespace NUMINAMATH_GPT_trig_identity_example_l797_79782

theorem trig_identity_example :
  (Real.cos (47 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - 
   Real.sin (47 * Real.pi / 180) * Real.sin (13 * Real.pi / 180)) = 
  (Real.cos (60 * Real.pi / 180)) := by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l797_79782


namespace NUMINAMATH_GPT_population_increase_l797_79786

theorem population_increase (initial_population final_population: ℝ) (r: ℝ) : 
  initial_population = 14000 →
  final_population = 16940 →
  final_population = initial_population * (1 + r) ^ 2 →
  r = 0.1 :=
by
  intros h_initial h_final h_eq
  sorry

end NUMINAMATH_GPT_population_increase_l797_79786


namespace NUMINAMATH_GPT_equal_cost_number_of_minutes_l797_79722

theorem equal_cost_number_of_minutes :
  ∃ m : ℝ, (8 + 0.25 * m = 12 + 0.20 * m) ∧ m = 80 :=
by
  sorry

end NUMINAMATH_GPT_equal_cost_number_of_minutes_l797_79722


namespace NUMINAMATH_GPT_power_of_two_expression_l797_79711

theorem power_of_two_expression :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = 5 * 2^2006 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_expression_l797_79711


namespace NUMINAMATH_GPT_hundred_times_reciprocal_l797_79700

theorem hundred_times_reciprocal (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := 
by 
  sorry

end NUMINAMATH_GPT_hundred_times_reciprocal_l797_79700


namespace NUMINAMATH_GPT_find_x_l797_79724

theorem find_x (m n k : ℝ) (x z : ℝ) (h1 : x = m * (n / (Real.sqrt z))^3)
  (h2 : x = 3 ∧ z = 12 ∧ 3 * 12 * Real.sqrt 12 = k) :
  (z = 75) → x = 24 / 125 :=
by
  -- Placeholder for proof, these assumptions and conditions would form the basis of the proof.
  sorry

end NUMINAMATH_GPT_find_x_l797_79724


namespace NUMINAMATH_GPT_geometric_sequence_general_formula_no_arithmetic_sequence_l797_79758

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: Sum of the first n terms of the sequence {a_n} is S_n
-- and S_n = 2a_n - n for n \in \mathbb{N}^*.
axiom sum_condition (n : ℕ) (h : n > 0) : S n = 2 * a n - n

-- Question 1: Prove that the sequence {a_n + 1} forms a geometric sequence.
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r, r ≠ 0 ∧ ∀ m, m > 0 → a (m + 1) + 1 = r * (a m + 1) := 
sorry

-- Question 2: Find the general formula for the sequence {a_n}.
theorem general_formula (n : ℕ) (h : n > 0) : a n = 2 ^ n - 1 := 
sorry

-- Question 3: Prove that there do not exist three consecutive terms in the sequence {a_n} that can form an arithmetic sequence.
theorem no_arithmetic_sequence (k : ℕ) (h : k > 0) : ¬ ∃ k, k > 0 ∧ a k = (a (k + 1) + a (k + 2)) / 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_general_formula_no_arithmetic_sequence_l797_79758


namespace NUMINAMATH_GPT_maria_traveled_portion_of_distance_l797_79728

theorem maria_traveled_portion_of_distance (total_distance first_stop remaining_distance_to_destination : ℝ) 
  (h1 : total_distance = 560) 
  (h2 : first_stop = total_distance / 2) 
  (h3 : remaining_distance_to_destination = 210) : 
  ((first_stop - (first_stop - (remaining_distance_to_destination + (first_stop - total_distance / 2)))) / (total_distance - first_stop)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_maria_traveled_portion_of_distance_l797_79728


namespace NUMINAMATH_GPT_marbles_leftover_l797_79795

theorem marbles_leftover (r p : ℤ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 := by
  sorry

end NUMINAMATH_GPT_marbles_leftover_l797_79795


namespace NUMINAMATH_GPT_part1_part2_l797_79714

-- Lean 4 statement for proving A == 2B
theorem part1 (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) 
    (h₃ : 0 < B) (h₄ : B < π / 2) (h₅ : 0 < C) (h₆ : C < π / 2) (h₇ : A + B + C = π)
    (h₈ : c = 2 * b * Real.cos A + b) : A = 2 * B :=
by sorry

-- Lean 4 statement for finding range of area of ∆ABD
theorem part2 (B : ℝ) (c : ℝ) (h₁ : 0 < B) (h₂ : B < π / 2) 
    (h₃ : A = 2 * B) (h₄ : c = 2) : 
    (Real.tan (π / 6) < (1 / 2) * c * (1 / Real.cos B) * Real.sin B) ∧ 
    ((1 / 2) * c * (1 / Real.cos B) * Real.sin B < 1) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l797_79714


namespace NUMINAMATH_GPT_degree_measure_cherry_pie_l797_79759

theorem degree_measure_cherry_pie 
  (total_students : ℕ) 
  (chocolate_pie : ℕ) 
  (apple_pie : ℕ) 
  (blueberry_pie : ℕ) 
  (remaining_students : ℕ)
  (remaining_students_eq_div : remaining_students = (total_students - (chocolate_pie + apple_pie + blueberry_pie))) 
  (equal_division : remaining_students / 2 = 5) 
  : (remaining_students / 2 * 360 / total_students = 45) := 
by 
  sorry

end NUMINAMATH_GPT_degree_measure_cherry_pie_l797_79759


namespace NUMINAMATH_GPT_range_of_f_l797_79774

noncomputable def f (x : ℝ) : ℝ :=
  x + Real.sqrt (x - 2)

theorem range_of_f : Set.range f = {y : ℝ | 2 ≤ y} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l797_79774


namespace NUMINAMATH_GPT_divide_rope_length_l797_79752

-- Definitions of variables based on the problem conditions
def rope_length : ℚ := 8 / 15
def num_parts : ℕ := 3

-- Theorem statement
theorem divide_rope_length :
  (1 / num_parts = (1 : ℚ) / 3) ∧ (rope_length * (1 / num_parts) = 8 / 45) :=
by
  sorry

end NUMINAMATH_GPT_divide_rope_length_l797_79752


namespace NUMINAMATH_GPT_square_implies_increasing_l797_79797

def seq (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n > 1, 
    ((a n - 2 > 0 ∧ ¬(∃ m < n, a m = a n - 2)) → a (n + 1) = a n - 2) ∧
    ((a n - 2 ≤ 0 ∨ ∃ m < n, a m = a n - 2) → a (n + 1) = a n + 3)

theorem square_implies_increasing (a : ℕ → ℤ) (n : ℕ) (h_seq : seq a) 
  (h_square : ∃ k, a n = k^2) (h_n_pos : n > 1) : 
  a n > a (n - 1) :=
sorry

end NUMINAMATH_GPT_square_implies_increasing_l797_79797


namespace NUMINAMATH_GPT_parallel_vectors_l797_79739

theorem parallel_vectors {m : ℝ} 
  (h : (2 * m + 1) / 2 = 3 / m): m = 3 / 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l797_79739


namespace NUMINAMATH_GPT_find_x_l797_79749

noncomputable def inv_cubicroot (y x : ℝ) : ℝ := y * x^(1/3)

theorem find_x (x y : ℝ) (h1 : ∃ k, inv_cubicroot 2 8 = k) (h2 : y = 8) : x = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l797_79749


namespace NUMINAMATH_GPT_solution_is_permutations_of_2_neg2_4_l797_79705

-- Definitions of the conditions
def cond1 (x y z : ℤ) : Prop := x * y + y * z + z * x = -4
def cond2 (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 24
def cond3 (x y z : ℤ) : Prop := x^3 + y^3 + z^3 + 3 * x * y * z = 16

-- The set of all integer solutions as permutations of (2, -2, 4)
def is_solution (x y z : ℤ) : Prop :=
  (x = 2 ∧ y = -2 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = -2) ∨
  (x = -2 ∧ y = 2 ∧ z = 4) ∨ (x = -2 ∧ y = 4 ∧ z = 2) ∨
  (x = 4 ∧ y = 2 ∧ z = -2) ∨ (x = 4 ∧ y = -2 ∧ z = 2)

-- Lean statement for the proof problem
theorem solution_is_permutations_of_2_neg2_4 (x y z : ℤ) :
  cond1 x y z → cond2 x y z → cond3 x y z → is_solution x y z :=
by
  -- sorry, the proof goes here
  sorry

end NUMINAMATH_GPT_solution_is_permutations_of_2_neg2_4_l797_79705


namespace NUMINAMATH_GPT_y_expression_value_l797_79723

theorem y_expression_value
  (y : ℝ)
  (h : y + 2 / y = 2) :
  y^6 + 3 * y^4 - 4 * y^2 + 2 = 2 := sorry

end NUMINAMATH_GPT_y_expression_value_l797_79723


namespace NUMINAMATH_GPT_green_fish_count_l797_79778

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end NUMINAMATH_GPT_green_fish_count_l797_79778


namespace NUMINAMATH_GPT_functional_equation_solution_l797_79740

theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) : 
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l797_79740


namespace NUMINAMATH_GPT_kittens_given_away_l797_79703

-- Conditions
def initial_kittens : ℕ := 8
def remaining_kittens : ℕ := 4

-- Statement to prove
theorem kittens_given_away : initial_kittens - remaining_kittens = 4 :=
by
  sorry

end NUMINAMATH_GPT_kittens_given_away_l797_79703


namespace NUMINAMATH_GPT_exponents_multiplication_l797_79763

variable (a : ℝ)

theorem exponents_multiplication : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_GPT_exponents_multiplication_l797_79763


namespace NUMINAMATH_GPT_range_of_a_l797_79708

theorem range_of_a {a : ℝ} 
  (hA : ∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))
  (hB : ∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))
  (hC : (a^2 + 1) / (2 * a) > 0)
  (hOnlyOneFalse : (¬(∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬(∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬((a^2 + 1) / (2 * a) > 0))):
  0 < a ∧ a < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l797_79708


namespace NUMINAMATH_GPT_total_number_of_elementary_events_is_16_l797_79761

def num_events_three_dice : ℕ := 6 * 6 * 6

theorem total_number_of_elementary_events_is_16 :
  num_events_three_dice = 16 := 
sorry

end NUMINAMATH_GPT_total_number_of_elementary_events_is_16_l797_79761


namespace NUMINAMATH_GPT_john_has_hours_to_spare_l797_79707

def total_wall_area (num_walls : ℕ) (wall_width wall_height : ℕ) : ℕ :=
  num_walls * wall_width * wall_height

def time_to_paint_area (area : ℕ) (rate_per_square_meter_in_minutes : ℕ) : ℕ :=
  area * rate_per_square_meter_in_minutes

def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem john_has_hours_to_spare 
  (num_walls : ℕ) (wall_width wall_height : ℕ)
  (rate_per_square_meter_in_minutes : ℕ) (total_available_hours : ℕ)
  (to_spare_hours : ℕ)
  (h : total_wall_area num_walls wall_width wall_height = num_walls * wall_width * wall_height)
  (h1 : time_to_paint_area (num_walls * wall_width * wall_height) rate_per_square_meter_in_minutes = num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes)
  (h2 : minutes_to_hours (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) = (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) / 60)
  (h3 : total_available_hours = 10) 
  (h4 : to_spare_hours = total_available_hours - (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes / 60)) : 
  to_spare_hours = 5 := 
sorry

end NUMINAMATH_GPT_john_has_hours_to_spare_l797_79707


namespace NUMINAMATH_GPT_number_of_sheets_is_9_l797_79718

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sheets_is_9_l797_79718


namespace NUMINAMATH_GPT_steve_total_money_l797_79777

theorem steve_total_money
    (nickels : ℕ)
    (dimes : ℕ)
    (nickel_value : ℕ := 5)
    (dime_value : ℕ := 10)
    (cond1 : nickels = 2)
    (cond2 : dimes = nickels + 4) 
    : (nickels * nickel_value + dimes * dime_value) = 70 := by
  sorry

end NUMINAMATH_GPT_steve_total_money_l797_79777


namespace NUMINAMATH_GPT_prime_power_sum_l797_79730

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem prime_power_sum (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  is_perfect_square (p^q + p^r) →
  (p = 2 ∧ ((q = 2 ∧ r = 5) ∨ (q = 5 ∧ r = 2) ∨ (q ≥ 3 ∧ is_prime q ∧ q = r)))
  ∨
  (p = 3 ∧ ((q = 2 ∧ r = 3) ∨ (q = 3 ∧ r = 2))) :=
sorry

end NUMINAMATH_GPT_prime_power_sum_l797_79730


namespace NUMINAMATH_GPT_necessary_without_sufficient_for_parallel_lines_l797_79771

noncomputable def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 2 = 0
noncomputable def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 1 = 0

theorem necessary_without_sufficient_for_parallel_lines :
  (∀ (a : ℝ), a = 2 → (∀ (x y : ℝ), line1 a x y → line2 a x y)) ∧ 
  ¬ (∀ (a : ℝ), (∀ (x y : ℝ), line1 a x y → line2 a x y) → a = 2) :=
sorry

end NUMINAMATH_GPT_necessary_without_sufficient_for_parallel_lines_l797_79771


namespace NUMINAMATH_GPT_inequality_solution_l797_79710

theorem inequality_solution (x : ℚ) (hx : x = 3 ∨ x = 2 ∨ x = 1 ∨ x = 0) : 
  (1 / 3) - (x / 3) < -(1 / 2) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l797_79710


namespace NUMINAMATH_GPT_trigonometric_identity_l797_79729

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos (3 * π / 2 - α)) / (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l797_79729


namespace NUMINAMATH_GPT_balls_in_boxes_l797_79791

theorem balls_in_boxes : 
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  total_ways - exclude_one_empty = 537 := 
by
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  have h : total_ways - exclude_one_empty = 537 := sorry
  exact h

end NUMINAMATH_GPT_balls_in_boxes_l797_79791


namespace NUMINAMATH_GPT_total_marbles_l797_79719

-- Define the given conditions 
def bags : ℕ := 20
def marbles_per_bag : ℕ := 156

-- The theorem stating that the total number of marbles is 3120
theorem total_marbles : bags * marbles_per_bag = 3120 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l797_79719


namespace NUMINAMATH_GPT_devin_teaching_years_l797_79735

theorem devin_teaching_years (total_years : ℕ) (tom_years : ℕ) (devin_years : ℕ) 
  (half_tom_years : ℕ)
  (h1 : total_years = 70) 
  (h2 : tom_years = 50)
  (h3 : total_years = tom_years + devin_years) 
  (h4 : half_tom_years = tom_years / 2) : 
  half_tom_years - devin_years = 5 :=
by
  sorry

end NUMINAMATH_GPT_devin_teaching_years_l797_79735


namespace NUMINAMATH_GPT_three_cards_different_suits_probability_l797_79712

-- Define the conditions and problem
noncomputable def prob_three_cards_diff_suits : ℚ :=
  have first_card_options := 52
  have second_card_options := 39
  have third_card_options := 26
  have total_ways_to_pick := (52 : ℕ) * (51 : ℕ) * (50 : ℕ)
  (39 / 51) * (26 / 50)

-- State our proof problem
theorem three_cards_different_suits_probability :
  prob_three_cards_diff_suits = 169 / 425 :=
sorry

end NUMINAMATH_GPT_three_cards_different_suits_probability_l797_79712


namespace NUMINAMATH_GPT_functional_equation_solution_l797_79754

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l797_79754


namespace NUMINAMATH_GPT_part1_solution_set_part2_no_real_x_l797_79767

-- Condition and problem definitions
def f (x a : ℝ) : ℝ := a^2 * x^2 + 2 * a * x - a^2 + 1

theorem part1_solution_set :
  (∀ x : ℝ, f x 2 ≤ 0 ↔ -3 / 2 ≤ x ∧ x ≤ 1 / 2) := sorry

theorem part2_no_real_x :
  ¬ ∃ x : ℝ, ∀ a : ℝ, -2 ≤ a ∧ a ≤ 2 → f x a ≥ 0 := sorry

end NUMINAMATH_GPT_part1_solution_set_part2_no_real_x_l797_79767


namespace NUMINAMATH_GPT_slow_speed_distance_l797_79796

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end NUMINAMATH_GPT_slow_speed_distance_l797_79796
