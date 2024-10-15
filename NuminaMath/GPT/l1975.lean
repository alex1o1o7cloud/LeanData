import Mathlib

namespace NUMINAMATH_GPT_find_k_l1975_197590

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1975_197590


namespace NUMINAMATH_GPT_quadrilateral_area_l1975_197525

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 30) (hh1 : h1 = 10) (hh2 : h2 = 6) :
  (1 / 2 * d * (h1 + h2) = 240) := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1975_197525


namespace NUMINAMATH_GPT_molly_age_l1975_197540

theorem molly_age
  (avg_age : ℕ)
  (hakimi_age : ℕ)
  (jared_age : ℕ)
  (molly_age : ℕ)
  (h1 : avg_age = 40)
  (h2 : hakimi_age = 40)
  (h3 : jared_age = hakimi_age + 10)
  (h4 : 3 * avg_age = hakimi_age + jared_age + molly_age) :
  molly_age = 30 :=
by
  sorry

end NUMINAMATH_GPT_molly_age_l1975_197540


namespace NUMINAMATH_GPT_proportion_correct_l1975_197523

theorem proportion_correct (m n : ℤ) (h : 6 * m = 7 * n) (hn : n ≠ 0) : (m : ℚ) / 7 = n / 6 :=
by sorry

end NUMINAMATH_GPT_proportion_correct_l1975_197523


namespace NUMINAMATH_GPT_faster_train_speed_l1975_197509

theorem faster_train_speed
  (slower_train_speed : ℝ := 60) -- speed of the slower train in km/h
  (length_train1 : ℝ := 1.10) -- length of the slower train in km
  (length_train2 : ℝ := 0.9) -- length of the faster train in km
  (cross_time_sec : ℝ := 47.99999999999999) -- crossing time in seconds
  (cross_time : ℝ := cross_time_sec / 3600) -- crossing time in hours
  (total_distance : ℝ := length_train1 + length_train2) -- total distance covered
  (relative_speed : ℝ := total_distance / cross_time) -- relative speed
  (faster_train_speed : ℝ := relative_speed - slower_train_speed) -- speed of the faster train
  : faster_train_speed = 90 :=
by
  sorry

end NUMINAMATH_GPT_faster_train_speed_l1975_197509


namespace NUMINAMATH_GPT_initial_noodles_l1975_197587

variable (d w e r : ℕ)

-- Conditions
def gave_to_william (w : ℕ) := w = 15
def gave_to_emily (e : ℕ) := e = 20
def remaining_noodles (r : ℕ) := r = 40

-- The statement to be proven
theorem initial_noodles (h1 : gave_to_william w) (h2 : gave_to_emily e) (h3 : remaining_noodles r) : d = w + e + r := by
  -- Proof will be filled in later.
  sorry

end NUMINAMATH_GPT_initial_noodles_l1975_197587


namespace NUMINAMATH_GPT_cubic_sum_identity_l1975_197566

theorem cubic_sum_identity
  (x y z : ℝ)
  (h1 : x + y + z = 8)
  (h2 : x * y + x * z + y * z = 17)
  (h3 : x * y * z = -14) :
  x^3 + y^3 + z^3 = 62 :=
sorry

end NUMINAMATH_GPT_cubic_sum_identity_l1975_197566


namespace NUMINAMATH_GPT_math_problem_l1975_197556

variables {A B : Type} [Fintype A] [Fintype B]
          (p1 p2 : ℝ) (h1 : 1/2 < p1) (h2 : p1 < p2) (h3 : p2 < 1)
          (nA : ℕ) (hA : nA = 3) (nB : ℕ) (hB : nB = 3)

noncomputable def E_X : ℝ := nA * p1
noncomputable def E_Y : ℝ := nB * p2

noncomputable def D_X : ℝ := nA * p1 * (1 - p1)
noncomputable def D_Y : ℝ := nB * p2 * (1 - p2)

theorem math_problem :
  E_X p1 nA = 3 * p1 →
  E_Y p2 nB = 3 * p2 →
  D_X p1 nA = 3 * p1 * (1 - p1) →
  D_Y p2 nB = 3 * p2 * (1 - p2) →
  E_X p1 nA < E_Y p2 nB ∧ D_X p1 nA > D_Y p2 nB :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1975_197556


namespace NUMINAMATH_GPT_abs_eq_4_reciprocal_eq_self_l1975_197586

namespace RationalProofs

-- Problem 1
theorem abs_eq_4 (x : ℚ) : |x| = 4 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Problem 2
theorem reciprocal_eq_self (x : ℚ) : x ≠ 0 → x⁻¹ = x ↔ x = 1 ∨ x = -1 :=
by sorry

end RationalProofs

end NUMINAMATH_GPT_abs_eq_4_reciprocal_eq_self_l1975_197586


namespace NUMINAMATH_GPT_max_rectangle_area_under_budget_l1975_197578

/-- 
Let L and W be the length and width of a rectangle, respectively, where:
1. The length L is made of materials priced at 3 yuan per meter.
2. The width W is made of materials priced at 5 yuan per meter.
3. Both L and W are integers.
4. The total cost 3L + 5W does not exceed 100 yuan.

Prove that the maximum area of the rectangle that can be made under these constraints is 40 square meters.
--/
theorem max_rectangle_area_under_budget :
  ∃ (L W : ℤ), 3 * L + 5 * W ≤ 100 ∧ 0 ≤ L ∧ 0 ≤ W ∧ L * W = 40 :=
sorry

end NUMINAMATH_GPT_max_rectangle_area_under_budget_l1975_197578


namespace NUMINAMATH_GPT_smallest_solution_l1975_197535

theorem smallest_solution (x : ℕ) (h1 : 6 * x ≡ 17 [MOD 31]) (h2 : x ≡ 3 [MOD 7]) : x = 24 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_solution_l1975_197535


namespace NUMINAMATH_GPT_atomic_number_R_l1975_197545

noncomputable def atomic_number_Pb := 82
def electron_shell_difference := 32

def same_group_atomic_number 
  (atomic_number_Pb : ℕ) 
  (electron_shell_difference : ℕ) : 
  ℕ := 
  atomic_number_Pb + electron_shell_difference

theorem atomic_number_R (R : ℕ) : 
  same_group_atomic_number atomic_number_Pb electron_shell_difference = 114 := 
by
  sorry

end NUMINAMATH_GPT_atomic_number_R_l1975_197545


namespace NUMINAMATH_GPT_lines_intersect_l1975_197526

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
  (2 + 3 * t, 2 - 4 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
  (4 + 5 * u, -6 + 3 * u)

theorem lines_intersect :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = (160 / 29, -160 / 29) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l1975_197526


namespace NUMINAMATH_GPT_infinite_power_tower_equation_l1975_197569

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ x ^ x ^ x ^ x -- continues infinitely

theorem infinite_power_tower_equation (x : ℝ) (h_pos : 0 < x) (h_eq : infinite_power_tower x = 2) : x = Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_infinite_power_tower_equation_l1975_197569


namespace NUMINAMATH_GPT_months_to_save_l1975_197512

/-- The grandfather saves 530 yuan from his pension every month. -/
def savings_per_month : ℕ := 530

/-- The price of the smartphone is 2000 yuan. -/
def smartphone_price : ℕ := 2000

/-- The number of months needed to save enough money to buy the smartphone. -/
def months_needed : ℕ := smartphone_price / savings_per_month

/-- Proof that the number of months needed is 4. -/
theorem months_to_save : months_needed = 4 :=
by
  sorry

end NUMINAMATH_GPT_months_to_save_l1975_197512


namespace NUMINAMATH_GPT_problem_statement_l1975_197522

theorem problem_statement (a b : ℤ) (h : |a + 5| + (b - 2) ^ 2 = 0) : (a + b) ^ 2010 = 3 ^ 2010 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1975_197522


namespace NUMINAMATH_GPT_times_older_l1975_197583

-- Conditions
variables (H S : ℕ)
axiom hold_age : H = 36
axiom hold_son_relation : H = 3 * S

-- Statement of the problem
theorem times_older (H S : ℕ) (h1 : H = 36) (h2 : H = 3 * S) : (H - 8) / (S - 8) = 7 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_times_older_l1975_197583


namespace NUMINAMATH_GPT_rectangle_length_l1975_197567

theorem rectangle_length (b l : ℝ) 
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 5) = l * b + 75) : l = 40 := by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1975_197567


namespace NUMINAMATH_GPT_circle_range_of_m_l1975_197562

theorem circle_range_of_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x + 2 * m * y + 2 * m^2 + m - 1 = 0 → (2 * m^2 + m - 1 = 0)) → (-2 < m) ∧ (m < 2/3) :=
by
  sorry

end NUMINAMATH_GPT_circle_range_of_m_l1975_197562


namespace NUMINAMATH_GPT_points_on_x_axis_circles_intersect_l1975_197589

theorem points_on_x_axis_circles_intersect (a b : ℤ)
  (h1 : 3 * a - b = 9)
  (h2 : 2 * a + 3 * b = -5) : (a : ℝ)^b = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_points_on_x_axis_circles_intersect_l1975_197589


namespace NUMINAMATH_GPT_radius_of_given_circle_is_eight_l1975_197502

noncomputable def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2

theorem radius_of_given_circle_is_eight :
  radius_of_circle 16 = 8 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_given_circle_is_eight_l1975_197502


namespace NUMINAMATH_GPT_first_player_wins_the_game_l1975_197501

-- Define the game state with 1992 stones and rules for taking stones
structure GameState where
  stones : Nat

-- Game rule: Each player can take a number of stones that is a divisor of the number of stones the 
-- opponent took on the previous turn
def isValidMove (prevMove: Nat) (currentMove: Nat) : Prop :=
  currentMove > 0 ∧ prevMove % currentMove = 0

-- The first player can take any number of stones but not all at once on their first move
def isFirstMoveValid (move: Nat) : Prop :=
  move > 0 ∧ move < 1992

-- Define the initial state of the game with 1992 stones
def initialGameState : GameState := { stones := 1992 }

-- Definition of optimal play leading to the first player's victory
def firstPlayerWins (s : GameState) : Prop :=
  s.stones = 1992 →
  ∃ move: Nat, isFirstMoveValid move ∧
  ∃ nextState: GameState, nextState.stones = s.stones - move ∧ 
  -- The first player wins with optimal strategy
  sorry

-- Theorem statement in Lean 4 equivalent to the math problem
theorem first_player_wins_the_game :
  firstPlayerWins initialGameState :=
  sorry

end NUMINAMATH_GPT_first_player_wins_the_game_l1975_197501


namespace NUMINAMATH_GPT_xiao_ming_final_score_l1975_197503

theorem xiao_ming_final_score :
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  (speech_image * weight_speech_image +
   content * weight_content +
   effectiveness * weight_effectiveness) = 8.3 :=
by
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  sorry

end NUMINAMATH_GPT_xiao_ming_final_score_l1975_197503


namespace NUMINAMATH_GPT_eval_expression_l1975_197520

-- Definitions based on the conditions and problem statement
def x (b : ℕ) : ℕ := b + 9

-- The theorem to prove
theorem eval_expression (b : ℕ) : x b - b + 5 = 14 := by
    sorry

end NUMINAMATH_GPT_eval_expression_l1975_197520


namespace NUMINAMATH_GPT_abs_eq_abs_implies_l1975_197598

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end NUMINAMATH_GPT_abs_eq_abs_implies_l1975_197598


namespace NUMINAMATH_GPT_probability_of_8_or_9_ring_l1975_197519

theorem probability_of_8_or_9_ring (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  p9 + p8 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_8_or_9_ring_l1975_197519


namespace NUMINAMATH_GPT_longest_side_range_l1975_197571

-- Definitions and conditions
def is_triangle (x y z : ℝ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

-- Problem statement
theorem longest_side_range (l x y z : ℝ) 
  (h_triangle: is_triangle x y z) 
  (h_perimeter: x + y + z = l / 2) 
  (h_longest: x ≥ y ∧ x ≥ z) : 
  l / 6 ≤ x ∧ x < l / 4 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_range_l1975_197571


namespace NUMINAMATH_GPT_radius_of_circle_l1975_197529

variable {O : Type*} [MetricSpace O]

def distance_near : ℝ := 1
def distance_far : ℝ := 7
def diameter : ℝ := distance_near + distance_far

theorem radius_of_circle (P : O) (r : ℝ) (h1 : distance_near = 1) (h2 : distance_far = 7) :
  r = diameter / 2 :=
by
  -- Proof would go here 
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1975_197529


namespace NUMINAMATH_GPT_remainder_zero_l1975_197570

theorem remainder_zero {n : ℕ} (h : n > 0) : 
  (2013^n - 1803^n - 1781^n + 1774^n) % 203 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_zero_l1975_197570


namespace NUMINAMATH_GPT_more_cabbages_produced_l1975_197564

theorem more_cabbages_produced
  (square_garden : ∀ n : ℕ, ∃ s : ℕ, s ^ 2 = n)
  (area_per_cabbage : ∀ cabbages : ℕ, cabbages = 11236 → ∃ s : ℕ, s ^ 2 = cabbages) :
  11236 - 105 ^ 2 = 211 := by
sorry

end NUMINAMATH_GPT_more_cabbages_produced_l1975_197564


namespace NUMINAMATH_GPT_mass_percentage_iodine_neq_662_l1975_197554

theorem mass_percentage_iodine_neq_662 (atomic_mass_Al : ℝ) (atomic_mass_I : ℝ) (molar_mass_AlI3 : ℝ) :
  atomic_mass_Al = 26.98 ∧ atomic_mass_I = 126.90 ∧ molar_mass_AlI3 = ((1 * atomic_mass_Al) + (3 * atomic_mass_I)) →
  (3 * atomic_mass_I / molar_mass_AlI3 * 100) ≠ 6.62 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_iodine_neq_662_l1975_197554


namespace NUMINAMATH_GPT_student_failed_by_l1975_197579

theorem student_failed_by :
  ∀ (total_marks obtained_marks passing_percentage : ℕ),
  total_marks = 700 →
  obtained_marks = 175 →
  passing_percentage = 33 →
  (passing_percentage * total_marks) / 100 - obtained_marks = 56 :=
by
  intros total_marks obtained_marks passing_percentage h1 h2 h3
  sorry

end NUMINAMATH_GPT_student_failed_by_l1975_197579


namespace NUMINAMATH_GPT_product_of_y_values_l1975_197504

theorem product_of_y_values :
  (∀ (x y : ℤ), x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → (y = 1 ∨ y = 2)) →
  (∀ (x y₁ x' y₂ : ℤ), (x, y₁) ≠ (x', y₂) → x = x' ∨ y₁ ≠ y₂) →
  (∀ (x y : ℤ), (x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → y = 1 ∨ y = 2) →
    (∃ (y₁ y₂ : ℤ), y₁ = 1 ∧ y₂ = 2 ∧ y₁ * y₂ = 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_y_values_l1975_197504


namespace NUMINAMATH_GPT_circle_radius_l1975_197516

theorem circle_radius (x y : ℝ) :
  x^2 + 2 * x + y^2 = 0 → 1 = 1 :=
by sorry

end NUMINAMATH_GPT_circle_radius_l1975_197516


namespace NUMINAMATH_GPT_sahil_selling_price_l1975_197515

noncomputable def sales_tax : ℝ := 0.10 * 18000
noncomputable def initial_cost_with_tax : ℝ := 18000 + sales_tax

noncomputable def broken_part_cost : ℝ := 3000
noncomputable def software_update_cost : ℝ := 4000
noncomputable def total_repair_cost : ℝ := broken_part_cost + software_update_cost
noncomputable def service_tax_on_repair : ℝ := 0.05 * total_repair_cost
noncomputable def total_repair_cost_with_tax : ℝ := total_repair_cost + service_tax_on_repair

noncomputable def transportation_charges : ℝ := 1500
noncomputable def total_cost_before_depreciation : ℝ := initial_cost_with_tax + total_repair_cost_with_tax + transportation_charges

noncomputable def depreciation_first_year : ℝ := 0.15 * total_cost_before_depreciation
noncomputable def value_after_first_year : ℝ := total_cost_before_depreciation - depreciation_first_year

noncomputable def depreciation_second_year : ℝ := 0.15 * value_after_first_year
noncomputable def value_after_second_year : ℝ := value_after_first_year - depreciation_second_year

noncomputable def profit : ℝ := 0.50 * value_after_second_year
noncomputable def selling_price : ℝ := value_after_second_year + profit

theorem sahil_selling_price : selling_price = 31049.44 := by
  sorry

end NUMINAMATH_GPT_sahil_selling_price_l1975_197515


namespace NUMINAMATH_GPT_cheaper_candy_price_l1975_197561

theorem cheaper_candy_price
    (mix_total_weight : ℝ) (mix_price_per_pound : ℝ)
    (cheap_weight : ℝ) (expensive_weight : ℝ) (expensive_price_per_pound : ℝ)
    (cheap_total_value : ℝ) (expensive_total_value : ℝ) (total_mix_value : ℝ) :
    mix_total_weight = 80 →
    mix_price_per_pound = 2.20 →
    cheap_weight = 64 →
    expensive_weight = mix_total_weight - cheap_weight →
    expensive_price_per_pound = 3.00 →
    cheap_total_value = cheap_weight * x →
    expensive_total_value = expensive_weight * expensive_price_per_pound →
    total_mix_value = mix_total_weight * mix_price_per_pound →
    total_mix_value = cheap_total_value + expensive_total_value →
    x = 2 := 
sorry

end NUMINAMATH_GPT_cheaper_candy_price_l1975_197561


namespace NUMINAMATH_GPT_correct_substitution_l1975_197596

theorem correct_substitution (x y : ℝ) 
  (h1 : y = 1 - x) 
  (h2 : x - 2 * y = 4) : x - 2 + 2 * x = 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_substitution_l1975_197596


namespace NUMINAMATH_GPT_problem1_problem2_l1975_197573

-- Problem 1
theorem problem1 : 5*Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 : (2*Real.sqrt 3 - 1)^2 + (Real.sqrt 24) / (Real.sqrt 2) = 13 - 2*Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1975_197573


namespace NUMINAMATH_GPT_repeating_decimal_rational_representation_l1975_197546

theorem repeating_decimal_rational_representation :
  (0.12512512512512514 : ℝ) = (125 / 999 : ℝ) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_rational_representation_l1975_197546


namespace NUMINAMATH_GPT_shaded_area_of_rectangle_l1975_197584

theorem shaded_area_of_rectangle :
  let length := 5   -- Length of the rectangle in cm
  let width := 12   -- Width of the rectangle in cm
  let base := 2     -- Base of each triangle in cm
  let height := 5   -- Height of each triangle in cm
  let rect_area := length * width
  let triangle_area := (1 / 2) * base * height
  let unshaded_area := 2 * triangle_area
  let shaded_area := rect_area - unshaded_area
  shaded_area = 50 :=
by
  -- Calculation follows solution steps.
  sorry

end NUMINAMATH_GPT_shaded_area_of_rectangle_l1975_197584


namespace NUMINAMATH_GPT_simplify_expression_l1975_197592

variable (x y : ℝ)

theorem simplify_expression : (3 * x + 4 * x + 5 * y + 2 * y) = 7 * x + 7 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1975_197592


namespace NUMINAMATH_GPT_johnny_distance_walked_l1975_197530

theorem johnny_distance_walked
  (dist_q_to_y : ℕ) (matthew_rate : ℕ) (johnny_rate : ℕ) (time_diff : ℕ) (johnny_walked : ℕ):
  dist_q_to_y = 45 →
  matthew_rate = 3 →
  johnny_rate = 4 →
  time_diff = 1 →
  (∃ t: ℕ, johnny_walked = johnny_rate * t 
            ∧ dist_q_to_y = matthew_rate * (t + time_diff) + johnny_walked) →
  johnny_walked = 24 := by
  sorry

end NUMINAMATH_GPT_johnny_distance_walked_l1975_197530


namespace NUMINAMATH_GPT_metallic_sheet_dimension_l1975_197553

theorem metallic_sheet_dimension :
  ∃ w : ℝ, (∀ (h := 8) (l := 40) (v := 2688),
    v = (w - 2 * h) * (l - 2 * h) * h) → w = 30 :=
by sorry

end NUMINAMATH_GPT_metallic_sheet_dimension_l1975_197553


namespace NUMINAMATH_GPT_xyz_expression_l1975_197568

theorem xyz_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
    (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
    (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)) = -3 / (2 * (x^2 + y^2 + xy)) :=
by sorry

end NUMINAMATH_GPT_xyz_expression_l1975_197568


namespace NUMINAMATH_GPT_delta_ratio_l1975_197544

theorem delta_ratio 
  (Δx : ℝ) (Δy : ℝ) 
  (y_new : ℝ := (1 + Δx)^2 + 1)
  (y_old : ℝ := 1^2 + 1)
  (Δy_def : Δy = y_new - y_old) :
  Δy / Δx = 2 + Δx :=
by
  sorry

end NUMINAMATH_GPT_delta_ratio_l1975_197544


namespace NUMINAMATH_GPT_minimum_x_value_l1975_197513

theorem minimum_x_value
  (sales_jan_may june_sales x : ℝ)
  (h_sales_jan_may : sales_jan_may = 38.6)
  (h_june_sales : june_sales = 5)
  (h_total_sales_condition : sales_jan_may + june_sales + 2 * june_sales * (1 + x / 100) + 2 * june_sales * (1 + x / 100)^2 ≥ 70) :
  x = 20 := by
  sorry

end NUMINAMATH_GPT_minimum_x_value_l1975_197513


namespace NUMINAMATH_GPT_factorize_expression_l1975_197585

theorem factorize_expression (x : ℝ) : 9 * x^3 - 18 * x^2 + 9 * x = 9 * x * (x - 1)^2 := 
by 
    sorry

end NUMINAMATH_GPT_factorize_expression_l1975_197585


namespace NUMINAMATH_GPT_div_by_30_l1975_197565

theorem div_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end NUMINAMATH_GPT_div_by_30_l1975_197565


namespace NUMINAMATH_GPT_number_of_three_star_reviews_l1975_197538

theorem number_of_three_star_reviews:
  ∀ (x : ℕ),
  (6 * 5 + 7 * 4 + 1 * 2 + x * 3) / 18 = 4 →
  x = 4 :=
by
  intros x H
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_number_of_three_star_reviews_l1975_197538


namespace NUMINAMATH_GPT_points_per_touchdown_l1975_197514

theorem points_per_touchdown (P : ℕ) (games : ℕ) (touchdowns_per_game : ℕ) (two_point_conversions : ℕ) (two_point_conversion_value : ℕ) (total_points : ℕ) :
  touchdowns_per_game = 4 →
  games = 15 →
  two_point_conversions = 6 →
  two_point_conversion_value = 2 →
  total_points = (4 * P * 15 + 6 * two_point_conversion_value) →
  total_points = 372 →
  P = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_points_per_touchdown_l1975_197514


namespace NUMINAMATH_GPT_value_of_A_cos_alpha_plus_beta_l1975_197537

noncomputable def f (A x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem value_of_A {A : ℝ}
  (h1 : f A (Real.pi / 3) = Real.sqrt 2) :
  A = 2 := 
by
  sorry

theorem cos_alpha_plus_beta {α β : ℝ}
  (hαβ1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hαβ2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h2 : f 2 (4*α + 4*Real.pi/3) = -30 / 17)
  (h3 : f 2 (4*β - 2*Real.pi/3) = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
by
  sorry

end NUMINAMATH_GPT_value_of_A_cos_alpha_plus_beta_l1975_197537


namespace NUMINAMATH_GPT_num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l1975_197558

theorem num_shoes_sold (price_shoes : ℕ) (num_shirts : ℕ) (price_shirts : ℕ) (total_earn_per_person : ℕ) : ℕ :=
  let total_earnings_shirts := num_shirts * price_shirts
  let total_earnings := total_earn_per_person * 2
  let earnings_from_shoes := total_earnings - total_earnings_shirts
  let num_shoes_sold := earnings_from_shoes / price_shoes
  num_shoes_sold

theorem sab_dane_sold_6_pairs_of_shoes :
  num_shoes_sold 3 18 2 27 = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l1975_197558


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1975_197563

theorem slope_of_perpendicular_line (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1975_197563


namespace NUMINAMATH_GPT_arithmetic_mean_of_sixty_integers_starting_from_3_l1975_197559

def arithmetic_mean_of_sequence (a d n : ℕ) : ℚ :=
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n / n

theorem arithmetic_mean_of_sixty_integers_starting_from_3 : arithmetic_mean_of_sequence 3 1 60 = 32.5 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_sixty_integers_starting_from_3_l1975_197559


namespace NUMINAMATH_GPT_product_power_conjecture_calculate_expression_l1975_197549

-- Conjecture Proof
theorem product_power_conjecture (a b : ℂ) (n : ℕ) : (a * b)^n = (a^n) * (b^n) :=
sorry

-- Calculation Proof
theorem calculate_expression : 
  ((-0.125 : ℂ)^2022) * ((2 : ℂ)^2021) * ((4 : ℂ)^2020) = (1 / 32 : ℂ) :=
sorry

end NUMINAMATH_GPT_product_power_conjecture_calculate_expression_l1975_197549


namespace NUMINAMATH_GPT_min_area_of_rectangle_with_perimeter_100_l1975_197582

theorem min_area_of_rectangle_with_perimeter_100 :
  ∃ (length width : ℕ), 
    (length + width = 50) ∧ 
    (length * width = 49) := 
by
  sorry

end NUMINAMATH_GPT_min_area_of_rectangle_with_perimeter_100_l1975_197582


namespace NUMINAMATH_GPT_age_of_b_l1975_197541

variable (A B C : ℕ)

-- Conditions
def avg_abc : Prop := A + B + C = 78
def avg_ac : Prop := A + C = 58

-- Question: Prove that B = 20
theorem age_of_b (h1 : avg_abc A B C) (h2 : avg_ac A C) : B = 20 := 
by sorry

end NUMINAMATH_GPT_age_of_b_l1975_197541


namespace NUMINAMATH_GPT_insects_remaining_l1975_197550

-- Define the initial counts of spiders, ants, and ladybugs
def spiders : ℕ := 3
def ants : ℕ := 12
def ladybugs : ℕ := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away : ℕ := 2

-- Prove the total number of remaining insects in the playground
theorem insects_remaining : (spiders + ants + ladybugs - ladybugs_flew_away) = 21 := by
  -- Expand the definitions and compute the result
  sorry

end NUMINAMATH_GPT_insects_remaining_l1975_197550


namespace NUMINAMATH_GPT_total_students_in_halls_l1975_197597

theorem total_students_in_halls :
  let S_g := 30
  let S_b := 2 * S_g
  let S_m := 3 / 5 * (S_g + S_b)
  S_g + S_b + S_m = 144 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_halls_l1975_197597


namespace NUMINAMATH_GPT_max_min_values_f_l1975_197594

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_min_values_f :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ Real.sqrt 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_f_l1975_197594


namespace NUMINAMATH_GPT_square_lawn_area_l1975_197527

theorem square_lawn_area (map_scale : ℝ) (map_edge_length_cm : ℝ) (actual_edge_length_m : ℝ) (actual_area_m2 : ℝ) 
  (h1 : map_scale = 1 / 5000) 
  (h2 : map_edge_length_cm = 4) 
  (h3 : actual_edge_length_m = (map_edge_length_cm / map_scale) / 100)
  (h4 : actual_area_m2 = actual_edge_length_m^2)
  : actual_area_m2 = 400 := 
by 
  sorry

end NUMINAMATH_GPT_square_lawn_area_l1975_197527


namespace NUMINAMATH_GPT_negation_of_exists_l1975_197552

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1975_197552


namespace NUMINAMATH_GPT_convert_base_10_to_base_8_l1975_197500

theorem convert_base_10_to_base_8 (n : ℕ) (n_eq : n = 3275) : 
  n = 3275 → ∃ (a b c d : ℕ), (a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 = 6323) :=
by 
  sorry

end NUMINAMATH_GPT_convert_base_10_to_base_8_l1975_197500


namespace NUMINAMATH_GPT_coffee_cost_l1975_197576

def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def dozens_of_donuts : ℕ := 3
def donuts_per_dozen : ℕ := 12

theorem coffee_cost :
  let total_donuts := dozens_of_donuts * donuts_per_dozen
  let total_ounces := ounces_per_donut * total_donuts
  let total_pots := total_ounces / ounces_per_pot
  let total_cost := total_pots * cost_per_pot
  total_cost = 18 := by
  sorry

end NUMINAMATH_GPT_coffee_cost_l1975_197576


namespace NUMINAMATH_GPT_cos2alpha_plus_sin2alpha_l1975_197531

def point_angle_condition (x y : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  x = -3 ∧ y = 4 ∧ r = 5 ∧ x^2 + y^2 = r^2

theorem cos2alpha_plus_sin2alpha (α : ℝ) (x y r : ℝ)
  (h : point_angle_condition x y r α) : 
  (Real.cos (2 * α) + Real.sin (2 * α)) = -31/25 :=
by
  sorry

end NUMINAMATH_GPT_cos2alpha_plus_sin2alpha_l1975_197531


namespace NUMINAMATH_GPT_four_digit_numbers_count_l1975_197506

theorem four_digit_numbers_count : (3:ℕ) ^ 4 = 81 := by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_count_l1975_197506


namespace NUMINAMATH_GPT_frogs_climbed_onto_logs_l1975_197557

-- Definitions of the conditions
def f_lily : ℕ := 5
def f_rock : ℕ := 24
def f_total : ℕ := 32

-- The final statement we want to prove
theorem frogs_climbed_onto_logs : f_total - (f_lily + f_rock) = 3 :=
by
  sorry

end NUMINAMATH_GPT_frogs_climbed_onto_logs_l1975_197557


namespace NUMINAMATH_GPT_proof_problem_l1975_197581

noncomputable def problem : ℚ :=
  let a := 1
  let b := 2
  let c := 1
  let d := 0
  a + 2 * b + 3 * c + 4 * d

theorem proof_problem : problem = 8 := by
  -- All computations are visible here
  unfold problem
  rfl

end NUMINAMATH_GPT_proof_problem_l1975_197581


namespace NUMINAMATH_GPT_greatest_integer_x_l1975_197599

theorem greatest_integer_x (x : ℤ) : 
  (∃ k : ℤ, (x - 4) = k ∧ x^2 - 3 * x + 4 = k * (x - 4) + 8) →
  x ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_x_l1975_197599


namespace NUMINAMATH_GPT_interval_of_n_l1975_197551

noncomputable def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem interval_of_n (n : ℕ) (hn : 0 < n ∧ n < 2000)
  (h1 : divides n 9999)
  (h2 : divides (n + 4) 999999) :
  801 ≤ n ∧ n ≤ 1200 :=
sorry

end NUMINAMATH_GPT_interval_of_n_l1975_197551


namespace NUMINAMATH_GPT_length_of_bridge_is_80_l1975_197521

-- Define the given constants
def length_of_train : ℕ := 280
def speed_of_train : ℕ := 18
def time_to_cross : ℕ := 20

-- Define the distance traveled by the train in the given time
def distance_traveled : ℕ := speed_of_train * time_to_cross

-- Define the length of the bridge from the given distance traveled
def length_of_bridge := distance_traveled - length_of_train

-- The theorem to prove the length of the bridge is 80 meters
theorem length_of_bridge_is_80 :
  length_of_bridge = 80 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_80_l1975_197521


namespace NUMINAMATH_GPT_intersection_A_B_l1975_197572

def is_log2 (y x : ℝ) : Prop := y = Real.log x / Real.log 2

def set_A (y : ℝ) : Set ℝ := { x | ∃ y, is_log2 y x}
def set_B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : (set_A 1) ∩ set_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1975_197572


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1975_197577

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 1) (h2 : b > 2) :
  (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1975_197577


namespace NUMINAMATH_GPT_parabola_line_intersection_l1975_197560

theorem parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 * x2 = 1) (h2 : x1 + 1 = 4) : x2 + 1 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_l1975_197560


namespace NUMINAMATH_GPT_inequality_solution_l1975_197517

theorem inequality_solution (x : ℝ) : (3 * x + 4 ≥ 4 * x) ∧ (2 * (x - 1) + x > 7) ↔ (3 < x ∧ x ≤ 4) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l1975_197517


namespace NUMINAMATH_GPT_circle_properties_l1975_197555

noncomputable def circle_center_and_radius (x y: ℝ) : Prop :=
  (x^2 + 8*x + y^2 - 10*y = 11)

theorem circle_properties :
  (∃ (a b r : ℝ), (a, b) = (-4, 5) ∧ r = 2 * Real.sqrt 13 ∧ circle_center_and_radius x y → a + b + r = 1 + 2 * Real.sqrt 13) :=
  sorry

end NUMINAMATH_GPT_circle_properties_l1975_197555


namespace NUMINAMATH_GPT_student_correct_numbers_l1975_197528

theorem student_correct_numbers (x y : ℕ) 
  (h1 : (10 * x + 5) * y = 4500)
  (h2 : (10 * x + 3) * y = 4380) : 
  (10 * x + 5 = 75 ∧ y = 60) :=
by 
  sorry

end NUMINAMATH_GPT_student_correct_numbers_l1975_197528


namespace NUMINAMATH_GPT_num_female_fox_terriers_l1975_197543

def total_dogs : Nat := 2012
def total_female_dogs : Nat := 1110
def total_fox_terriers : Nat := 1506
def male_shih_tzus : Nat := 202

theorem num_female_fox_terriers :
    ∃ (female_fox_terriers: Nat), 
        female_fox_terriers = total_fox_terriers - (total_dogs - total_female_dogs - male_shih_tzus) := by
    sorry

end NUMINAMATH_GPT_num_female_fox_terriers_l1975_197543


namespace NUMINAMATH_GPT_largest_n_satisfying_inequality_l1975_197534

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, n ≥ 1 ∧ n^(6033) < 2011^(2011) ∧ ∀ m : ℕ, m > n → m^(6033) ≥ 2011^(2011) :=
sorry

end NUMINAMATH_GPT_largest_n_satisfying_inequality_l1975_197534


namespace NUMINAMATH_GPT_log_diff_decreases_l1975_197575

-- Define the natural number n
variable (n : ℕ)

-- Proof statement
theorem log_diff_decreases (hn : 0 < n) : 
  (Real.log (n + 1) - Real.log n) = Real.log (1 + 1 / n) ∧ 
  ∀ m : ℕ, ∀ hn' : 0 < m, m > n → Real.log (m + 1) - Real.log m < Real.log (n + 1) - Real.log n := by
  sorry

end NUMINAMATH_GPT_log_diff_decreases_l1975_197575


namespace NUMINAMATH_GPT_coefficient_of_y_l1975_197518

theorem coefficient_of_y (x y a : ℝ) (h1 : 7 * x + y = 19) (h2 : x + a * y = 1) (h3 : 2 * x + y = 5) : a = 3 :=
sorry

end NUMINAMATH_GPT_coefficient_of_y_l1975_197518


namespace NUMINAMATH_GPT_find_x_in_terms_of_a_b_l1975_197574

variable (a b x : ℝ)
variable (ha : a > 0) (hb : b > 0) (hx : x > 0) (r : ℝ)
variable (h1 : r = (4 * a)^(3 * b))
variable (h2 : r = a ^ b * x ^ b)

theorem find_x_in_terms_of_a_b 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : (4 * a)^(3 * b) = r)
  (h2 : r = a^b * x^b) :
  x = 64 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_terms_of_a_b_l1975_197574


namespace NUMINAMATH_GPT_entrance_exit_ways_equal_49_l1975_197580

-- Define the number of gates on each side
def south_gates : ℕ := 4
def north_gates : ℕ := 3

-- Define the total number of gates
def total_gates : ℕ := south_gates + north_gates

-- State the theorem and provide the expected proof structure
theorem entrance_exit_ways_equal_49 : (total_gates * total_gates) = 49 := 
by {
  sorry
}

end NUMINAMATH_GPT_entrance_exit_ways_equal_49_l1975_197580


namespace NUMINAMATH_GPT_inequality_always_negative_l1975_197595

theorem inequality_always_negative (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (-3 < k ∧ k ≤ 0) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_inequality_always_negative_l1975_197595


namespace NUMINAMATH_GPT_find_fourth_number_l1975_197542

theorem find_fourth_number (x y : ℝ) (h1 : 0.25 / x = 2 / y) (h2 : x = 0.75) : y = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l1975_197542


namespace NUMINAMATH_GPT_third_median_length_l1975_197588

noncomputable def triangle_median_length (m₁ m₂ : ℝ) (area : ℝ) : ℝ :=
  if m₁ = 5 ∧ m₂ = 4 ∧ area = 6 * Real.sqrt 5 then
    3 * Real.sqrt 7
  else
    0

theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ)
  (h₁ : m₁ = 5) (h₂ : m₂ = 4) (h₃ : area = 6 * Real.sqrt 5) :
  triangle_median_length m₁ m₂ area = 3 * Real.sqrt 7 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_third_median_length_l1975_197588


namespace NUMINAMATH_GPT_arithmetic_seq_solution_l1975_197593

theorem arithmetic_seq_solution (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith : ∀ n ≥ 2, a (n+1) - a n ^ 2 + a (n-1) = 0) 
  (h_sum : ∀ k, S k = (k * (a 1 + a k)) / 2) :
  S (2 * n - 1) - 4 * n = -2 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_solution_l1975_197593


namespace NUMINAMATH_GPT_books_count_l1975_197505

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end NUMINAMATH_GPT_books_count_l1975_197505


namespace NUMINAMATH_GPT_molecular_weight_of_complex_compound_l1975_197511

def molecular_weight (n : ℕ) (N_w : ℝ) (o : ℕ) (O_w : ℝ) (h : ℕ) (H_w : ℝ) (p : ℕ) (P_w : ℝ) : ℝ :=
  (n * N_w) + (o * O_w) + (h * H_w) + (p * P_w)

theorem molecular_weight_of_complex_compound :
  molecular_weight 2 14.01 5 16.00 3 1.01 1 30.97 = 142.02 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_complex_compound_l1975_197511


namespace NUMINAMATH_GPT_pyramid_surface_area_and_volume_l1975_197510

def s := 8
def PF := 15

noncomputable def FM := s / 2
noncomputable def PM := Real.sqrt (PF^2 + FM^2)
noncomputable def baseArea := s^2
noncomputable def lateralAreaTriangle := (1 / 2) * s * PM
noncomputable def totalSurfaceArea := baseArea + 4 * lateralAreaTriangle
noncomputable def volume := (1 / 3) * baseArea * PF

theorem pyramid_surface_area_and_volume :
  totalSurfaceArea = 64 + 16 * Real.sqrt 241 ∧
  volume = 320 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_surface_area_and_volume_l1975_197510


namespace NUMINAMATH_GPT_find_x_range_l1975_197507

-- Given definition for a decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

-- The main theorem to prove
theorem find_x_range (f : ℝ → ℝ) (h_decreasing : is_decreasing f) :
  {x : ℝ | f (|1 / x|) < f 1} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_find_x_range_l1975_197507


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1975_197536

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 54) (h3 : c + a = 58) : 
  a + b + c = 73.5 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_sum_of_three_numbers_l1975_197536


namespace NUMINAMATH_GPT_inequality_abc_l1975_197532

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1975_197532


namespace NUMINAMATH_GPT_abs_neg_2023_l1975_197524

theorem abs_neg_2023 : abs (-2023) = 2023 := 
by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l1975_197524


namespace NUMINAMATH_GPT_expenditure_ratio_l1975_197591

variable {I : ℝ} -- Income in the first year

-- Conditions
def first_year_savings (I : ℝ) : ℝ := 0.5 * I
def first_year_expenditure (I : ℝ) : ℝ := I - first_year_savings I
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (I : ℝ) : ℝ := 2 * first_year_savings I
def second_year_expenditure (I : ℝ) : ℝ := second_year_income I - second_year_savings I

-- Condition statement in Lean
theorem expenditure_ratio (I : ℝ) : 
  let total_expenditure := first_year_expenditure I + second_year_expenditure I
  (total_expenditure / first_year_expenditure I) = 2 :=
  by 
    sorry

end NUMINAMATH_GPT_expenditure_ratio_l1975_197591


namespace NUMINAMATH_GPT_xy_maximum_value_l1975_197547

theorem xy_maximum_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2 * y) : x - 2 * y ≤ 2 / 3 :=
sorry

end NUMINAMATH_GPT_xy_maximum_value_l1975_197547


namespace NUMINAMATH_GPT_six_digit_number_unique_solution_l1975_197539

theorem six_digit_number_unique_solution
    (a b c d e f : ℕ)
    (hN : (N : ℕ) = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)
    (hM : (M : ℕ) = 100000 * d + 10000 * e + 1000 * f + 100 * a + 10 * b + c)
    (h_eq : 7 * N = 6 * M) :
    N = 461538 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_unique_solution_l1975_197539


namespace NUMINAMATH_GPT_inequality_inequality_holds_l1975_197508

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end NUMINAMATH_GPT_inequality_inequality_holds_l1975_197508


namespace NUMINAMATH_GPT_shadow_taller_pot_length_l1975_197533

-- Definitions based on the conditions a)
def height_shorter_pot : ℕ := 20
def shadow_shorter_pot : ℕ := 10
def height_taller_pot : ℕ := 40

-- The proof problem
theorem shadow_taller_pot_length : 
  ∃ (S2 : ℕ), (height_shorter_pot / shadow_shorter_pot = height_taller_pot / S2) ∧ S2 = 20 :=
sorry

end NUMINAMATH_GPT_shadow_taller_pot_length_l1975_197533


namespace NUMINAMATH_GPT_both_fifth_and_ninth_terms_are_20_l1975_197548

def sequence_a (n : ℕ) : ℕ := n^2 - 14 * n + 65

theorem both_fifth_and_ninth_terms_are_20 : sequence_a 5 = 20 ∧ sequence_a 9 = 20 := 
by
  sorry

end NUMINAMATH_GPT_both_fifth_and_ninth_terms_are_20_l1975_197548
