import Mathlib

namespace find_number_l2018_201803

theorem find_number (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) (number : ℕ) :
  quotient = 9 ∧ remainder = 1 ∧ divisor = 30 → number = 271 := by
  intro h
  sorry

end find_number_l2018_201803


namespace no_triangle_100_sticks_yes_triangle_99_sticks_l2018_201839

-- Definitions for the sums of lengths of sticks
def sum_lengths (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Conditions and questions for the problem
def is_divisible_by_3 (x : ℕ) : Prop := x % 3 = 0

-- Proof problem for n = 100
theorem no_triangle_100_sticks : ¬ (is_divisible_by_3 (sum_lengths 100)) := by
  sorry

-- Proof problem for n = 99
theorem yes_triangle_99_sticks : is_divisible_by_3 (sum_lengths 99) := by
  sorry

end no_triangle_100_sticks_yes_triangle_99_sticks_l2018_201839


namespace homework_done_l2018_201816

theorem homework_done :
  ∃ (D E C Z M : Prop),
    -- Statements of students
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    -- Truth-telling condition
    ((D → D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (E → ¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (C → ¬ D ∧ ¬ E ∧ C ∧ ¬ Z ∧ ¬ M) ∧
    (Z → ¬ D ∧ ¬ E ∧ ¬ C ∧ Z ∧ ¬ M) ∧
    (M → ¬ D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ M)) ∧
    -- Number of students who did their homework condition
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) := 
sorry

end homework_done_l2018_201816


namespace circle_center_and_radius_l2018_201871

theorem circle_center_and_radius (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  (∃ c : ℝ × ℝ, c = (3, 0)) ∧ (∃ r : ℝ, r = 3) := 
sorry

end circle_center_and_radius_l2018_201871


namespace find_x_l2018_201813

theorem find_x (x : ℕ) (h : 27^3 + 27^3 + 27^3 + 27^3 = 3^x) : x = 11 :=
sorry

end find_x_l2018_201813


namespace burger_cost_l2018_201823

theorem burger_cost 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 440)
  (h2 : 3 * b + 2 * s = 330) : b = 110 := 
by 
  sorry

end burger_cost_l2018_201823


namespace remainder_sum_l2018_201869

theorem remainder_sum (n : ℤ) : ((7 - n) + (n + 3)) % 7 = 3 :=
sorry

end remainder_sum_l2018_201869


namespace arithmetic_seq_max_sum_l2018_201825

noncomputable def max_arith_seq_sum_lemma (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_seq_max_sum :
  ∀ (a1 d : ℤ),
    (3 * a1 + 6 * d = 9) →
    (a1 + 5 * d = -9) →
    max_arith_seq_sum_lemma a1 d 3 = 21 :=
by
  sorry

end arithmetic_seq_max_sum_l2018_201825


namespace continuous_zero_point_condition_l2018_201851

theorem continuous_zero_point_condition (f : ℝ → ℝ) {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) :
  (f a * f b < 0) → (∃ c ∈ Set.Ioo a b, f c = 0) ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0 → f a * f b < 0) :=
sorry

end continuous_zero_point_condition_l2018_201851


namespace transformation_identity_l2018_201867

theorem transformation_identity (a b : ℝ) 
    (h1 : ∃ a b : ℝ, ∀ x y : ℝ, (y, -x) = (-7, 3) → (x, y) = (3, 7))
    (h2 : ∃ a b : ℝ, ∀ c d : ℝ, (d, c) = (3, -7) → (c, d) = (-7, 3)) :
    b - a = 4 :=
by
    sorry

end transformation_identity_l2018_201867


namespace points_per_bag_l2018_201801

/-
Wendy had 11 bags but didn't recycle 2 of them. She would have earned 
45 points for recycling all 11 bags. Prove that Wendy earns 5 points 
per bag of cans she recycles.
-/

def total_bags : Nat := 11
def unrecycled_bags : Nat := 2
def recycled_bags : Nat := total_bags - unrecycled_bags
def total_points : Nat := 45

theorem points_per_bag : total_points / recycled_bags = 5 := by
  sorry

end points_per_bag_l2018_201801


namespace find_abc_l2018_201814

theorem find_abc :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 30 ∧
  (1/a + 1/b + 1/c + 450/(a*b*c) = 1) ∧ 
  a*b*c = 1912 :=
sorry

end find_abc_l2018_201814


namespace smallest_number_among_neg2_neg1_0_pi_l2018_201891

/-- The smallest number among -2, -1, 0, and π is -2. -/
theorem smallest_number_among_neg2_neg1_0_pi : min (min (min (-2 : ℝ) (-1)) 0) π = -2 := 
sorry

end smallest_number_among_neg2_neg1_0_pi_l2018_201891


namespace area_of_region_enclosed_by_parabolas_l2018_201854

-- Define the given parabolas
def parabola1 (y : ℝ) : ℝ := -3 * y^2
def parabola2 (y : ℝ) : ℝ := 1 - 4 * y^2

-- Define the integral representing the area between the parabolas
noncomputable def areaBetweenParabolas : ℝ :=
  2 * (∫ y in (0 : ℝ)..1, (parabola2 y - parabola1 y))

-- The statement to be proved
theorem area_of_region_enclosed_by_parabolas :
  areaBetweenParabolas = 4 / 3 := 
sorry

end area_of_region_enclosed_by_parabolas_l2018_201854


namespace octahedron_non_blue_probability_l2018_201873

theorem octahedron_non_blue_probability :
  let total_faces := 8
  let blue_faces := 3
  let red_faces := 3
  let green_faces := 2
  let non_blue_faces := total_faces - blue_faces
  (non_blue_faces / total_faces : ℚ) = (5 / 8 : ℚ) :=
by
  sorry

end octahedron_non_blue_probability_l2018_201873


namespace joseph_cards_percentage_left_l2018_201899

theorem joseph_cards_percentage_left (h1 : ℕ := 16) (h2 : ℚ := 3/8) (h3 : ℕ := 2) :
  ((h1 - (h2 * h1 + h3)) / h1 * 100) = 50 :=
by
  sorry

end joseph_cards_percentage_left_l2018_201899


namespace range_of_xy_l2018_201855

theorem range_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y + x * y = 30) :
  12 < x * y ∧ x * y < 870 :=
by sorry

end range_of_xy_l2018_201855


namespace pool_depth_l2018_201826

theorem pool_depth 
  (length width : ℝ) 
  (chlorine_per_120_cubic_feet chlorine_cost : ℝ) 
  (total_spent volume_per_quart_of_chlorine : ℝ) 
  (H_length : length = 10) 
  (H_width : width = 8)
  (H_chlorine_per_120_cubic_feet : chlorine_per_120_cubic_feet = 1 / 120)
  (H_chlorine_cost : chlorine_cost = 3)
  (H_total_spent : total_spent = 12)
  (H_volume_per_quart_of_chlorine : volume_per_quart_of_chlorine = 120) :
  ∃ depth : ℝ, total_spent / chlorine_cost * volume_per_quart_of_chlorine = length * width * depth ∧ depth = 6 :=
by 
  sorry

end pool_depth_l2018_201826


namespace cylinder_volume_increase_l2018_201827

theorem cylinder_volume_increase {R H : ℕ} (x : ℚ) (C : ℝ) (π : ℝ) 
  (hR : R = 8) (hH : H = 3) (hπ : π = Real.pi)
  (hV : ∃ C > 0, π * (R + x)^2 * (H + x) = π * R^2 * H + C) :
  x = 16 / 3 :=
by
  sorry

end cylinder_volume_increase_l2018_201827


namespace frank_total_points_l2018_201893

def points_defeating_enemies (enemies : ℕ) (points_per_enemy : ℕ) : ℕ :=
  enemies * points_per_enemy

def total_points (points_from_enemies : ℕ) (completion_points : ℕ) : ℕ :=
  points_from_enemies + completion_points

theorem frank_total_points :
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  total_points points_from_enemies completion_points = 62 :=
by
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  -- Placeholder for proof
  sorry

end frank_total_points_l2018_201893


namespace largest_common_term_in_range_1_to_200_l2018_201836

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l2018_201836


namespace solution_set_ineq_l2018_201804

theorem solution_set_ineq (x : ℝ) :
  (x - 1) / (1 - 2 * x) ≥ 0 ↔ (1 / 2 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_ineq_l2018_201804


namespace root_expression_value_l2018_201888

theorem root_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 2021 - 2 * a^2 - 2 * a = 2019 := 
by sorry

end root_expression_value_l2018_201888


namespace domain_of_f_l2018_201886

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.log (x + 2)

theorem domain_of_f :
  {x : ℝ | (x ≠ 0) ∧ (x > -2)} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x)} :=
by
  sorry

end domain_of_f_l2018_201886


namespace line_through_intersections_l2018_201819

-- Conditions
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem statement
theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → x - y - 3 = 0 :=
by
  sorry

end line_through_intersections_l2018_201819


namespace find_x1_l2018_201853

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 :=
  sorry

end find_x1_l2018_201853


namespace salary_reduction_l2018_201859

theorem salary_reduction (S : ℝ) (R : ℝ) 
  (h : (S - (R / 100) * S) * (4 / 3) = S) :
  R = 25 := 
  sorry

end salary_reduction_l2018_201859


namespace john_trip_l2018_201856

theorem john_trip (t : ℝ) (h : t ≥ 0) : 
  ∀ t : ℝ, 60 * t + 90 * ((7 / 2) - t) = 300 :=
by sorry

end john_trip_l2018_201856


namespace profit_from_ad_l2018_201898

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end profit_from_ad_l2018_201898


namespace inequality_proof_l2018_201847

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l2018_201847


namespace part_one_part_two_l2018_201800

-- Part (1)
theorem part_one (x : ℝ) : x - (3 * x - 1) ≤ 2 * x + 3 → x ≥ -1 / 2 :=
by sorry

-- Part (2)
theorem part_two (x : ℝ) : 
  (3 * (x - 1) < 4 * x - 2) ∧ ((1 + 4 * x) / 3 > x - 1) → x > -1 :=
by sorry

end part_one_part_two_l2018_201800


namespace circle_eq_l2018_201881

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l2018_201881


namespace smallest_n_lil_wayne_rain_l2018_201802

noncomputable def probability_rain (n : ℕ) : ℝ := 
  1 / 2 - 1 / 2^(n + 1)

theorem smallest_n_lil_wayne_rain :
  ∃ n : ℕ, probability_rain n > 0.499 ∧ (∀ m : ℕ, m < n → probability_rain m ≤ 0.499) ∧ n = 9 := 
by
  sorry

end smallest_n_lil_wayne_rain_l2018_201802


namespace find_AC_l2018_201832

theorem find_AC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (max_val : A - C = 3) (min_val : -A - C = -1) : 
  A = 2 ∧ C = 1 :=
by
  sorry

end find_AC_l2018_201832


namespace smallest_multiple_1_10_is_2520_l2018_201824

noncomputable def smallest_multiple_1_10 : ℕ :=
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

theorem smallest_multiple_1_10_is_2520 : smallest_multiple_1_10 = 2520 :=
  sorry

end smallest_multiple_1_10_is_2520_l2018_201824


namespace standard_equation_line_standard_equation_circle_intersection_range_a_l2018_201875

theorem standard_equation_line (a t x y : ℝ) (h1 : x = a - 2 * t * y) (h2 : y = -4 * t) : 
    2 * x - y - 2 * a = 0 :=
sorry

theorem standard_equation_circle (θ x y : ℝ) (h1 : x = 4 * Real.cos θ) (h2 : y = 4 * Real.sin θ) : 
    x ^ 2 + y ^ 2 = 16 :=
sorry

theorem intersection_range_a (a : ℝ) (h : ∃ (t θ : ℝ), (a - 2 * t * (-4 * t)) = 4 * (Real.cos θ) ∧ (-4 * t) = 4 * (Real.sin θ)) :
    -4 * Real.sqrt 5 <= a ∧ a <= 4 * Real.sqrt 5 :=
sorry

end standard_equation_line_standard_equation_circle_intersection_range_a_l2018_201875


namespace CA_eq_A_intersection_CB_eq_l2018_201884

-- Definitions as per conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | x > 1 }

-- Proof problems as per questions and answers
theorem CA_eq : (U \ A) = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem A_intersection_CB_eq : (A ∩ (U \ B)) = { x : ℝ | x ≤ 1 } :=
by
  sorry

end CA_eq_A_intersection_CB_eq_l2018_201884


namespace solve_system_l2018_201860

/-- Given the system of equations:
    3 * (x + y) - 4 * (x - y) = 5
    (x + y) / 2 + (x - y) / 6 = 0
  Prove that the solution is x = -1/3 and y = 2/3 
-/
theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x + y) - 4 * (x - y) = 5)
  (h2 : (x + y) / 2 + (x - y) / 6 = 0) : 
  x = -1 / 3 ∧ y = 2 / 3 := 
sorry

end solve_system_l2018_201860


namespace value_of_expression_l2018_201889

theorem value_of_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -1) : -a^2 - b^2 + a * b = -21 := by
  sorry

end value_of_expression_l2018_201889


namespace garden_strawberry_yield_l2018_201848

-- Definitions from the conditions
def garden_length : ℝ := 10
def garden_width : ℝ := 15
def plants_per_sq_ft : ℝ := 5
def strawberries_per_plant : ℝ := 12

-- Expected total number of strawberries
def expected_strawberries : ℝ := 9000

-- Proof statement
theorem garden_strawberry_yield : 
  (garden_length * garden_width * plants_per_sq_ft * strawberries_per_plant) = expected_strawberries :=
by sorry

end garden_strawberry_yield_l2018_201848


namespace gcd_lcm_sum_l2018_201862

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l2018_201862


namespace vacuum_cleaner_cost_l2018_201837

-- Define initial amount collected
def initial_amount : ℕ := 20

-- Define amount added each week
def weekly_addition : ℕ := 10

-- Define number of weeks
def number_of_weeks : ℕ := 10

-- Define the total amount after 10 weeks
def total_amount : ℕ := initial_amount + (weekly_addition * number_of_weeks)

-- Prove that the total amount is equal to the cost of the vacuum cleaner
theorem vacuum_cleaner_cost : total_amount = 120 := by
  sorry

end vacuum_cleaner_cost_l2018_201837


namespace family_member_bites_count_l2018_201849

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end family_member_bites_count_l2018_201849


namespace jack_needs_5_rocks_to_equal_weights_l2018_201866

-- Given Conditions
def WeightJack : ℕ := 60
def WeightAnna : ℕ := 40
def WeightRock : ℕ := 4

-- Theorem Statement
theorem jack_needs_5_rocks_to_equal_weights : (WeightJack - WeightAnna) / WeightRock = 5 :=
by
  sorry

end jack_needs_5_rocks_to_equal_weights_l2018_201866


namespace product_mod5_is_zero_l2018_201861

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l2018_201861


namespace factorize_problem_1_factorize_problem_2_l2018_201852

variables (x y : ℝ)

-- Problem 1: Prove that x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2
theorem factorize_problem_1 : 
  x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2 :=
sorry

-- Problem 2: Prove that x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)
theorem factorize_problem_2 : 
  x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) :=
sorry

end factorize_problem_1_factorize_problem_2_l2018_201852


namespace jenna_less_than_bob_l2018_201892

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end jenna_less_than_bob_l2018_201892


namespace flare_initial_velocity_and_duration_l2018_201897

noncomputable def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

theorem flare_initial_velocity_and_duration (v t : ℝ) :
  (h v 5 = 245) ↔ (v = 73.5) ∧ (5 < t ∧ t < 10) :=
by {
  sorry
}

end flare_initial_velocity_and_duration_l2018_201897


namespace remaining_perimeter_of_square_with_cutouts_l2018_201846

theorem remaining_perimeter_of_square_with_cutouts 
  (square_side : ℝ) (green_square_side : ℝ) (init_perimeter : ℝ) 
  (green_square_perimeter_increase : ℝ) (final_perimeter : ℝ) :
  square_side = 10 → green_square_side = 2 →
  init_perimeter = 4 * square_side → green_square_perimeter_increase = 4 * green_square_side →
  final_perimeter = init_perimeter + green_square_perimeter_increase →
  final_perimeter = 44 :=
by
  intros hsquare_side hgreen_square_side hinit_perimeter hgreen_incr hfinal_perimeter
  -- Proof steps can be added here
  sorry

end remaining_perimeter_of_square_with_cutouts_l2018_201846


namespace tiles_needed_l2018_201818

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end tiles_needed_l2018_201818


namespace find_a_l2018_201822

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_eq (x y a : ℝ) : Prop := x + a * y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq x y a → (x - 1)^2 + (y - 2)^2 = 4) →
  ∃ a, (a = -1) :=
sorry

end find_a_l2018_201822


namespace pair_D_equal_l2018_201833

theorem pair_D_equal: (-1)^3 = (-1)^2023 := by
  sorry

end pair_D_equal_l2018_201833


namespace meaning_of_implication_l2018_201879

theorem meaning_of_implication (p q : Prop) : (p → q) = ((p → q) = True) :=
sorry

end meaning_of_implication_l2018_201879


namespace find_tangent_circles_tangent_circle_at_given_point_l2018_201880

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 4

def is_tangent (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ (u v : ℝ), (u - a)^2 + (v - b)^2 = 1 ∧
  (x - u)^2 + (y - v)^2 = 4 ∧
  (x = u ∧ y = v)

theorem find_tangent_circles (x y a b : ℝ) (hx : circle_C x y)
  (ha_b : is_tangent x y a b) :
  (a = 5 ∧ b = -1) ∨ (a = 3 ∧ b = -1) :=
sorry

theorem tangent_circle_at_given_point (x y : ℝ) (hx : circle_C x y) (y_pos : y = -1)
  : ((x - 5)^2 + (y + 1)^2 = 1) ∨ ((x - 3)^2 + (y + 1)^2 = 1) :=
sorry

end find_tangent_circles_tangent_circle_at_given_point_l2018_201880


namespace max_min_difference_l2018_201895

open Real

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 2 * y = 4) :
  ∃(max min : ℝ), (∀z, z = (|2 * x - y| / (|x| + |y|)) → z ≤ max) ∧ 
                  (∀z, z = (|2 * x - y| / (|x| + |y|)) → min ≤ z) ∧ 
                  (max - min = 5) :=
by
  sorry

end max_min_difference_l2018_201895


namespace find_remainder_l2018_201809

theorem find_remainder (S : Finset ℕ) (h : ∀ n ∈ S, ∃ m, n^2 + 10 * n - 2010 = m^2) :
  (S.sum id) % 1000 = 304 := by
  sorry

end find_remainder_l2018_201809


namespace emma_age_proof_l2018_201821

def is_age_of_emma (age : Nat) : Prop := 
  let guesses := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]
  let at_least_60_percent_low := (guesses.filter (· < age)).length * 10 ≥ 6 * guesses.length
  let exactly_two_off_by_one := (guesses.filter (λ x => x = age - 1 ∨ x = age + 1)).length = 2
  let is_prime := Nat.Prime age
  at_least_60_percent_low ∧ exactly_two_off_by_one ∧ is_prime

theorem emma_age_proof : is_age_of_emma 43 := 
  by sorry

end emma_age_proof_l2018_201821


namespace solve_equation_l2018_201876

theorem solve_equation {x y z : ℝ} (h₁ : x + 95 / 12 * y + 4 * z = 0)
  (h₂ : 4 * x + 95 / 12 * y - 3 * z = 0)
  (h₃ : 3 * x + 5 * y - 4 * z = 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 * z / y^3 = -60 :=
sorry

end solve_equation_l2018_201876


namespace defective_units_shipped_percentage_l2018_201820

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end defective_units_shipped_percentage_l2018_201820


namespace apples_total_l2018_201887

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l2018_201887


namespace solve_system_l2018_201812

theorem solve_system :
  ∃ (x y z : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y - 4 * z = 10 ∧ 2 * x + y + 3 * z = 1.25 :=
by
  sorry

end solve_system_l2018_201812


namespace common_root_l2018_201834

theorem common_root (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔ (p = -3 ∨ p = 9) :=
by
  sorry

end common_root_l2018_201834


namespace number_of_ways_to_buy_three_items_l2018_201835

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l2018_201835


namespace man_total_earnings_l2018_201896

-- Define the conditions
def total_days := 30
def wage_per_day := 10
def fine_per_absence := 2
def days_absent := 7
def days_worked := total_days - days_absent
def earned := days_worked * wage_per_day
def fine := days_absent * fine_per_absence
def total_earnings := earned - fine

-- State the theorem
theorem man_total_earnings : total_earnings = 216 := by
  -- Using the definitions provided, the proof should show that the calculations result in 216
  sorry

end man_total_earnings_l2018_201896


namespace tan_value_of_point_on_graph_l2018_201810

theorem tan_value_of_point_on_graph (a : ℝ) (h : (4 : ℝ) ^ (1/2) = a) : 
  Real.tan ((a / 6) * Real.pi) = Real.sqrt 3 :=
by 
  sorry

end tan_value_of_point_on_graph_l2018_201810


namespace find_k_for_tangent_graph_l2018_201842

theorem find_k_for_tangent_graph (k : ℝ) (h : (∀ x : ℝ, x^2 - 6 * x + k = 0 → (x = 3))) : k = 9 :=
sorry

end find_k_for_tangent_graph_l2018_201842


namespace fraction_denominator_l2018_201865

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end fraction_denominator_l2018_201865


namespace distance_between_intersections_l2018_201807

open Classical
open Real

noncomputable def curve1 (x y : ℝ) : Prop := y^2 = x
noncomputable def curve2 (x y : ℝ) : Prop := x + 2 * y = 10

theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ),
    (curve1 p1.1 p1.2) ∧ (curve2 p1.1 p1.2) ∧
    (curve1 p2.1 p2.2) ∧ (curve2 p2.1 p2.2) ∧
    (dist p1 p2 = 2 * sqrt 55) :=
by
  sorry

end distance_between_intersections_l2018_201807


namespace maximum_sum_of_numbers_in_grid_l2018_201811

theorem maximum_sum_of_numbers_in_grid :
  ∀ (grid : List (List ℕ)) (rect_cover : (ℕ × ℕ) → (ℕ × ℕ) → Prop),
  (∀ x y, rect_cover x y → x ≠ y → x.1 < 6 → x.2 < 6 → y.1 < 6 → y.2 < 6) →
  (∀ x y z w, rect_cover x y ∧ rect_cover z w → 
    (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∨ (x.1 = z.1 ∨ x.2 = z.2) → 
    (x.1 = z.1 ∧ x.2 = y.2 ∨ x.2 = z.2 ∧ x.1 = y.1)) → False) →
  (36 = 6 * 6) →
  18 = 36 / 2 →
  342 = (18 * 19) :=
by
  intro grid rect_cover h_grid h_no_common_edge h_grid_size h_num_rectangles
  sorry

end maximum_sum_of_numbers_in_grid_l2018_201811


namespace total_preparation_and_cooking_time_l2018_201883

def time_to_chop_pepper := 3
def time_to_chop_onion := 4
def time_to_slice_mushroom := 2
def time_to_dice_tomato := 3
def time_to_grate_cheese := 1
def time_to_assemble_and_cook_omelet := 6

def num_peppers := 8
def num_onions := 4
def num_mushrooms := 6
def num_tomatoes := 6
def num_omelets := 10

theorem total_preparation_and_cooking_time :
  (num_peppers * time_to_chop_pepper) +
  (num_onions * time_to_chop_onion) +
  (num_mushrooms * time_to_slice_mushroom) +
  (num_tomatoes * time_to_dice_tomato) +
  (num_omelets * time_to_grate_cheese) +
  (num_omelets * time_to_assemble_and_cook_omelet) = 140 :=
by
  sorry

end total_preparation_and_cooking_time_l2018_201883


namespace triangle_tangent_ratio_l2018_201840

variable {A B C a b c : ℝ}

theorem triangle_tangent_ratio 
  (h : a * Real.cos B - b * Real.cos A = (3 / 5) * c)
  : Real.tan A / Real.tan B = 4 :=
sorry

end triangle_tangent_ratio_l2018_201840


namespace flour_needed_for_bread_l2018_201863

-- Definitions based on conditions
def flour_per_loaf : ℝ := 2.5
def number_of_loaves : ℕ := 2

-- Theorem statement
theorem flour_needed_for_bread : flour_per_loaf * number_of_loaves = 5 :=
by sorry

end flour_needed_for_bread_l2018_201863


namespace chromium_percentage_new_alloy_l2018_201885

-- Define the weights and chromium percentages of the alloys
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 35
def chromium_percent_alloy1 : ℝ := 0.15
def chromium_percent_alloy2 : ℝ := 0.08

-- Define the theorem to calculate the chromium percentage of the new alloy
theorem chromium_percentage_new_alloy :
  ((weight_alloy1 * chromium_percent_alloy1 + weight_alloy2 * chromium_percent_alloy2)
  / (weight_alloy1 + weight_alloy2) * 100) = 10.1 :=
by
  sorry

end chromium_percentage_new_alloy_l2018_201885


namespace peter_completes_remaining_work_in_14_days_l2018_201845

-- Define the conditions and the theorem
variable (W : ℕ) (work_done : ℕ) (remaining_work : ℕ)

theorem peter_completes_remaining_work_in_14_days
  (h1 : Matt_and_Peter_rate = (W/20))
  (h2 : Peter_rate = (W/35))
  (h3 : Work_done_in_12_days = (12 * (W/20)))
  (h4 : Remaining_work = (W - (12 * (W/20))))
  : (remaining_work / Peter_rate)  = 14 := sorry

end peter_completes_remaining_work_in_14_days_l2018_201845


namespace find_m_range_l2018_201844

/--
Given:
1. Proposition \( p \) (p): The equation \(\frac{x^2}{2} + \frac{y^2}{m} = 1\) represents an ellipse with foci on the \( y \)-axis.
2. Proposition \( q \) (q): \( f(x) = \frac{4}{3}x^3 - 2mx^2 + (4m-3)x - m \) is monotonically increasing on \((-\infty, +\infty)\).

Prove:
If \( \neg p \land q \) is true, then the range of values for \( m \) is \( [1, 2] \).
-/

def p (m : ℝ) : Prop :=
  m > 2

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, (4 * x^2 - 4 * m * x + 4 * m - 3) >= 0

theorem find_m_range (m : ℝ) (hpq : ¬ p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end find_m_range_l2018_201844


namespace terminating_decimal_expansion_l2018_201858

theorem terminating_decimal_expansion : (11 / 125 : ℝ) = 0.088 := 
by
  sorry

end terminating_decimal_expansion_l2018_201858


namespace probability_not_in_square_b_l2018_201857

theorem probability_not_in_square_b (area_A : ℝ) (perimeter_B : ℝ) 
  (area_A_eq : area_A = 30) (perimeter_B_eq : perimeter_B = 16) : 
  (14 / 30 : ℝ) = (7 / 15 : ℝ) :=
by
  sorry

end probability_not_in_square_b_l2018_201857


namespace sum_of_fractions_l2018_201877

theorem sum_of_fractions : 
  (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := 
by
  sorry

end sum_of_fractions_l2018_201877


namespace surface_area_circumscribed_sphere_l2018_201830

-- Define the problem
theorem surface_area_circumscribed_sphere (a b c : ℝ)
    (h1 : a^2 + b^2 = 3)
    (h2 : b^2 + c^2 = 5)
    (h3 : c^2 + a^2 = 4) : 
    4 * Real.pi * (a^2 + b^2 + c^2) / 4 = 6 * Real.pi :=
by
  -- The proof is omitted
  sorry

end surface_area_circumscribed_sphere_l2018_201830


namespace find_f_100_l2018_201831

-- Define the function f such that it satisfies the condition f(10^x) = x
noncomputable def f : ℝ → ℝ := sorry

-- Define the main theorem to prove f(100) = 2 given the condition f(10^x) = x
theorem find_f_100 (h : ∀ x : ℝ, f (10^x) = x) : f 100 = 2 :=
by {
  sorry
}

end find_f_100_l2018_201831


namespace y_minus_x_eq_seven_point_five_l2018_201868

theorem y_minus_x_eq_seven_point_five (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) :
  y - x = 7.5 :=
by sorry

end y_minus_x_eq_seven_point_five_l2018_201868


namespace number_of_ways_to_represent_1500_l2018_201815

theorem number_of_ways_to_represent_1500 :
  ∃ (count : ℕ), count = 30 ∧ ∀ (a b c : ℕ), a * b * c = 1500 :=
sorry

end number_of_ways_to_represent_1500_l2018_201815


namespace second_shipment_is_13_l2018_201817

-- Definitions based on the conditions
def first_shipment : ℕ := 7
def third_shipment : ℕ := 45
def total_couscous_used : ℕ := 13 * 5 -- 65
def total_couscous_from_three_shipments (second_shipment : ℕ) : ℕ :=
  first_shipment + second_shipment + third_shipment

-- Statement of the proof problem corresponding to the conditions and question
theorem second_shipment_is_13 (x : ℕ) 
  (h : total_couscous_used = total_couscous_from_three_shipments x) : x = 13 := 
by
  sorry

end second_shipment_is_13_l2018_201817


namespace age_of_seventh_person_l2018_201841

theorem age_of_seventh_person (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ) 
    (h1 : A1 < A2) (h2 : A2 < A3) (h3 : A3 < A4) (h4 : A4 < A5) (h5 : A5 < A6) 
    (h6 : A2 = A1 + D1) (h7 : A3 = A2 + D2) (h8 : A4 = A3 + D3) 
    (h9 : A5 = A4 + D4) (h10 : A6 = A5 + D5)
    (h11 : A1 + A2 + A3 + A4 + A5 + A6 = 246) 
    (h12 : 246 + A7 = 315) : A7 = 69 :=
by
  sorry

end age_of_seventh_person_l2018_201841


namespace arrange_numbers_in_ascending_order_l2018_201806

noncomputable def S := 222 ^ 2
noncomputable def T := 22 ^ 22
noncomputable def U := 2 ^ 222
noncomputable def V := 22 ^ (2 ^ 2)
noncomputable def W := 2 ^ (22 ^ 2)
noncomputable def X := 2 ^ (2 ^ 22)
noncomputable def Y := 2 ^ (2 ^ (2 ^ 2))

theorem arrange_numbers_in_ascending_order :
  S < Y ∧ Y < V ∧ V < T ∧ T < U ∧ U < W ∧ W < X :=
sorry

end arrange_numbers_in_ascending_order_l2018_201806


namespace pizza_sales_calculation_l2018_201829

def pizzas_sold_in_spring (total_sales : ℝ) (summer_sales : ℝ) (fall_percentage : ℝ) (winter_percentage : ℝ) : ℝ :=
  total_sales - summer_sales - (fall_percentage * total_sales) - (winter_percentage * total_sales)

theorem pizza_sales_calculation :
  let summer_sales := 5;
  let fall_percentage := 0.1;
  let winter_percentage := 0.2;
  ∃ (total_sales : ℝ), 0.4 * total_sales = summer_sales ∧
    pizzas_sold_in_spring total_sales summer_sales fall_percentage winter_percentage = 3.75 :=
by
  sorry

end pizza_sales_calculation_l2018_201829


namespace profit_percent_is_correct_l2018_201838

noncomputable def profit_percent : ℝ := 
  let marked_price_per_pen := 1 
  let pens_bought := 56 
  let effective_payment := 46 
  let discount := 0.01
  let cost_price_per_pen := effective_payment / pens_bought
  let selling_price_per_pen := marked_price_per_pen * (1 - discount)
  let total_selling_price := pens_bought * selling_price_per_pen
  let profit := total_selling_price - effective_payment
  (profit / effective_payment) * 100

theorem profit_percent_is_correct : abs (profit_percent - 20.52) < 0.01 :=
by
  sorry

end profit_percent_is_correct_l2018_201838


namespace sum_of_geometric_sequence_l2018_201805

noncomputable def geometric_sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence (a₁ q : ℝ) (n : ℕ) 
  (h1 : a₁ + a₁ * q^3 = 10) 
  (h2 : a₁ * q + a₁ * q^4 = 20) : 
  geometric_sequence_sum a₁ q n = (10 / 9) * (2^n - 1) :=
by 
  sorry

end sum_of_geometric_sequence_l2018_201805


namespace initial_paper_count_l2018_201808

theorem initial_paper_count (used left initial : ℕ) (h_used : used = 156) (h_left : left = 744) :
  initial = used + left :=
sorry

end initial_paper_count_l2018_201808


namespace theme_park_ratio_l2018_201870

theorem theme_park_ratio (a c : ℕ) (h_cost_adult : 20 * a + 15 * c = 1600) (h_eq_ratio : a * 28 = c * 59) :
  a / c = 59 / 28 :=
by
  /-
  Proof steps would go here.
  -/
  sorry

end theme_park_ratio_l2018_201870


namespace miles_on_first_day_l2018_201878

variable (x : ℝ)

/-- The distance traveled on the first day is x miles. -/
noncomputable def second_day_distance := (3/4) * x

/-- The distance traveled on the second day is (3/4)x miles. -/
noncomputable def third_day_distance := (1/2) * (x + second_day_distance x)

theorem miles_on_first_day
    (total_distance : x + second_day_distance x + third_day_distance x = 525)
    : x = 200 :=
sorry

end miles_on_first_day_l2018_201878


namespace colorings_equivalence_l2018_201882

-- Define the problem setup
structure ProblemSetup where
  n : ℕ  -- Number of disks (8)
  blue : ℕ  -- Number of blue disks (3)
  red : ℕ  -- Number of red disks (3)
  green : ℕ  -- Number of green disks (2)
  rotations : ℕ  -- Number of rotations (4: 90°, 180°, 270°, 360°)
  reflections : ℕ  -- Number of reflections (8: 4 through vertices and 4 through midpoints)

def number_of_colorings (setup : ProblemSetup) : ℕ :=
  sorry -- This represents the complex implementation details

def correct_answer : ℕ := 43

theorem colorings_equivalence : ∀ (setup : ProblemSetup),
  setup.n = 8 → setup.blue = 3 → setup.red = 3 → setup.green = 2 → setup.rotations = 4 → setup.reflections = 8 →
  number_of_colorings setup = correct_answer :=
by
  intros setup h1 h2 h3 h4 h5 h6
  sorry

end colorings_equivalence_l2018_201882


namespace find_n_of_sum_of_evens_l2018_201850

-- Definitions based on conditions in part (a)
def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_evens_up_to (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  (k / 2) * (2 + (n - 1))

-- Problem statement in Lean
theorem find_n_of_sum_of_evens : 
  ∃ n : ℕ, is_odd n ∧ sum_of_evens_up_to n = 81 * 82 ∧ n = 163 :=
by
  sorry

end find_n_of_sum_of_evens_l2018_201850


namespace modulo_11_residue_l2018_201843

theorem modulo_11_residue : 
  (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := 
by
  sorry

end modulo_11_residue_l2018_201843


namespace hawks_points_l2018_201872

theorem hawks_points (E H : ℕ) (h₁ : E + H = 82) (h₂ : E = H + 18) (h₃ : H ≥ 9) : H = 32 :=
sorry

end hawks_points_l2018_201872


namespace non_working_games_count_l2018_201864

def total_games : ℕ := 16
def price_each : ℕ := 7
def total_earnings : ℕ := 56

def working_games : ℕ := total_earnings / price_each
def non_working_games : ℕ := total_games - working_games

theorem non_working_games_count : non_working_games = 8 := by
  sorry

end non_working_games_count_l2018_201864


namespace running_time_of_BeastOfWar_is_100_l2018_201890

noncomputable def Millennium := 120  -- minutes
noncomputable def AlphaEpsilon := Millennium - 30  -- minutes
noncomputable def BeastOfWar := AlphaEpsilon + 10  -- minutes
noncomputable def DeltaSquadron := 2 * BeastOfWar  -- minutes

theorem running_time_of_BeastOfWar_is_100 :
  BeastOfWar = 100 :=
by
  -- Proof goes here
  sorry

end running_time_of_BeastOfWar_is_100_l2018_201890


namespace alpha_plus_beta_l2018_201874

theorem alpha_plus_beta (α β : ℝ) (hα_range : -Real.pi / 2 < α ∧ α < Real.pi / 2)
    (hβ_range : -Real.pi / 2 < β ∧ β < Real.pi / 2)
    (h_roots : ∃ (x1 x2 : ℝ), x1 = Real.tan α ∧ x2 = Real.tan β ∧ (x1^2 + 3 * Real.sqrt 3 * x1 + 4 = 0) ∧ (x2^2 + 3 * Real.sqrt 3 * x2 + 4 = 0)) :
    α + β = -2 * Real.pi / 3 :=
sorry

end alpha_plus_beta_l2018_201874


namespace ribbons_left_l2018_201894

theorem ribbons_left {initial_ribbons morning_giveaway afternoon_giveaway ribbons_left : ℕ} 
    (h1 : initial_ribbons = 38) 
    (h2 : morning_giveaway = 14) 
    (h3 : afternoon_giveaway = 16) 
    (h4 : ribbons_left = initial_ribbons - (morning_giveaway + afternoon_giveaway)) : 
  ribbons_left = 8 := 
by 
  sorry

end ribbons_left_l2018_201894


namespace arithmetic_sequence_x_y_sum_l2018_201828

theorem arithmetic_sequence_x_y_sum :
  ∀ (a d x y: ℕ), 
  a = 3 → d = 6 → 
  (∀ (n: ℕ), n ≥ 1 → a + (n-1) * d = 3 + (n-1) * 6) →
  (a + 5 * d = x) → (a + 6 * d = y) → 
  (y = 45 - d) → x + y = 72 :=
by
  intros a d x y h_a h_d h_seq h_x h_y h_y_equals
  sorry

end arithmetic_sequence_x_y_sum_l2018_201828
