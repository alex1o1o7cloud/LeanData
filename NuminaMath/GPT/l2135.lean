import Mathlib

namespace NUMINAMATH_GPT_solve_system_equations_l2135_213572

variable (x y z : ℝ)

theorem solve_system_equations (h1 : 3 * x = 20 + (20 - x))
    (h2 : y = 2 * x - 5)
    (h3 : z = Real.sqrt (x + 4)) :
  x = 10 ∧ y = 15 ∧ z = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_equations_l2135_213572


namespace NUMINAMATH_GPT_differentiable_difference_constant_l2135_213530

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

theorem differentiable_difference_constant (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) 
  (h : ∀ x, fderiv ℝ f x = fderiv ℝ g x) : 
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end NUMINAMATH_GPT_differentiable_difference_constant_l2135_213530


namespace NUMINAMATH_GPT_r_investment_time_l2135_213573

variables (P Q R Profit_p Profit_q Profit_r Tp Tq Tr : ℕ)
variables (h1 : P / Q = 7 / 5)
variables (h2 : Q / R = 5 / 4)
variables (h3 : Profit_p / Profit_q = 7 / 10)
variables (h4 : Profit_p / Profit_r = 7 / 8)
variables (h5 : Tp = 2)
variables (h6 : Tq = t)

theorem r_investment_time (t : ℕ) :
  ∃ Tr : ℕ, Tr = 4 :=
sorry

end NUMINAMATH_GPT_r_investment_time_l2135_213573


namespace NUMINAMATH_GPT_minimum_value_function_equality_holds_at_two_thirds_l2135_213582

noncomputable def f (x : ℝ) : ℝ := 4 / x + 1 / (1 - x)

theorem minimum_value_function (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 9 := sorry

theorem equality_holds_at_two_thirds : f (2 / 3) = 9 := sorry

end NUMINAMATH_GPT_minimum_value_function_equality_holds_at_two_thirds_l2135_213582


namespace NUMINAMATH_GPT_not_perfect_square_of_divisor_l2135_213534

theorem not_perfect_square_of_divisor (n d : ℕ) (hn : 0 < n) (hd : d ∣ 2 * n^2) :
  ¬ ∃ x : ℕ, n^2 + d = x^2 :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_of_divisor_l2135_213534


namespace NUMINAMATH_GPT_power_of_a_l2135_213515

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end NUMINAMATH_GPT_power_of_a_l2135_213515


namespace NUMINAMATH_GPT_min_sum_areas_of_triangles_l2135_213571

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1 / 4, 0)

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

def O := (0, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def on_opposite_sides_x_axis (p q : ℝ × ℝ) : Prop := p.2 * q.2 < 0

theorem min_sum_areas_of_triangles 
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (hAB : on_opposite_sides_x_axis A B)
  (h_dot : dot_product A B = 2) :
  ∃ m : ℝ, m = 3 := by
  sorry

end NUMINAMATH_GPT_min_sum_areas_of_triangles_l2135_213571


namespace NUMINAMATH_GPT_total_legos_156_l2135_213595

def pyramid_bottom_legos (side_length : Nat) : Nat := side_length * side_length
def pyramid_second_level_legos (length : Nat) (width : Nat) : Nat := length * width
def pyramid_third_level_legos (side_length : Nat) : Nat :=
  let total_legos := (side_length * (side_length + 1)) / 2
  total_legos - 3  -- Subtracting 3 Legos for the corners

def pyramid_fourth_level_legos : Nat := 1

def total_pyramid_legos : Nat :=
  pyramid_bottom_legos 10 +
  pyramid_second_level_legos 8 6 +
  pyramid_third_level_legos 4 +
  pyramid_fourth_level_legos

theorem total_legos_156 : total_pyramid_legos = 156 := by
  sorry

end NUMINAMATH_GPT_total_legos_156_l2135_213595


namespace NUMINAMATH_GPT_negation_equiv_l2135_213558

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 + 1 ≥ 1

-- Negation of the original proposition
def negated_prop : Prop := ∃ x : ℝ, x^2 + 1 < 1

-- Main theorem stating the equivalence
theorem negation_equiv :
  (¬ (∀ x : ℝ, original_prop x)) ↔ negated_prop :=
by sorry

end NUMINAMATH_GPT_negation_equiv_l2135_213558


namespace NUMINAMATH_GPT_solve_equation_l2135_213575

theorem solve_equation (n : ℝ) :
  (3 - 2 * n) / (n + 2) + (3 * n - 9) / (3 - 2 * n) = 2 ↔ 
  n = (25 + Real.sqrt 13) / 18 ∨ n = (25 - Real.sqrt 13) / 18 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2135_213575


namespace NUMINAMATH_GPT_Lexie_age_proof_l2135_213559

variables (L B S : ℕ)

def condition1 : Prop := L = B + 6
def condition2 : Prop := S = 2 * L
def condition3 : Prop := S - B = 14

theorem Lexie_age_proof (h1 : condition1 L B) (h2 : condition2 S L) (h3 : condition3 S B) : L = 8 :=
by
  sorry

end NUMINAMATH_GPT_Lexie_age_proof_l2135_213559


namespace NUMINAMATH_GPT_percentage_of_third_number_l2135_213590

variable (T F S : ℝ)

-- Declare the conditions from step a)
def condition_one : Prop := S = 0.25 * T
def condition_two : Prop := F = 0.20 * S

-- Define the proof problem, proving that F is 5% of T given the conditions
theorem percentage_of_third_number
  (h1 : condition_one T S)
  (h2 : condition_two F S) :
  F = 0.05 * T := by
  sorry

end NUMINAMATH_GPT_percentage_of_third_number_l2135_213590


namespace NUMINAMATH_GPT_typesetter_times_l2135_213594

theorem typesetter_times (α β γ : ℝ) (h1 : 1 / β - 1 / α = 10)
                                        (h2 : 1 / β - 1 / γ = 6)
                                        (h3 : 9 * (α + β) = 10 * (β + γ)) :
    α = 1 / 20 ∧ β = 1 / 30 ∧ γ = 1 / 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_typesetter_times_l2135_213594


namespace NUMINAMATH_GPT_children_got_off_bus_l2135_213516

theorem children_got_off_bus (initial : ℕ) (got_on : ℕ) (after : ℕ) : Prop :=
  initial = 22 ∧ got_on = 40 ∧ after = 2 → initial + got_on - 60 = after


end NUMINAMATH_GPT_children_got_off_bus_l2135_213516


namespace NUMINAMATH_GPT_trigonometric_expression_l2135_213581

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.sin α = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.sin (α + π / 4))) = -7 / 5 := 
sorry

end NUMINAMATH_GPT_trigonometric_expression_l2135_213581


namespace NUMINAMATH_GPT_find_a_share_l2135_213569

noncomputable def total_investment (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def total_profit (b_share total_inv b_inv : ℕ) : ℕ :=
  b_share * total_inv / b_inv

noncomputable def a_share (a_inv total_inv total_pft : ℕ) : ℕ :=
  a_inv * total_pft / total_inv

theorem find_a_share
  (a_inv b_inv c_inv b_share : ℕ)
  (h1 : a_inv = 7000)
  (h2 : b_inv = 11000)
  (h3 : c_inv = 18000)
  (h4 : b_share = 880) :
  a_share a_inv (total_investment a_inv b_inv c_inv) (total_profit b_share (total_investment a_inv b_inv c_inv) b_inv) = 560 := 
by
  sorry

end NUMINAMATH_GPT_find_a_share_l2135_213569


namespace NUMINAMATH_GPT_find_middle_part_length_l2135_213584

theorem find_middle_part_length (a b c : ℝ) 
  (h1 : a + b + c = 28) 
  (h2 : (a - 0.5 * a) + b + 0.5 * c = 16) :
  b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_middle_part_length_l2135_213584


namespace NUMINAMATH_GPT_first_group_number_l2135_213577

theorem first_group_number (x : ℕ) (h1 : x + 120 = 126) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_first_group_number_l2135_213577


namespace NUMINAMATH_GPT_train_crosses_bridge_in_12_4_seconds_l2135_213532

noncomputable def train_crossing_bridge_time (length_train : ℝ) (speed_train_kmph : ℝ) (length_bridge : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (1000 / 3600)
  let total_distance := length_train + length_bridge
  total_distance / speed_train_mps

theorem train_crosses_bridge_in_12_4_seconds :
  train_crossing_bridge_time 110 72 138 = 12.4 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_12_4_seconds_l2135_213532


namespace NUMINAMATH_GPT_xy_not_z_probability_l2135_213503

theorem xy_not_z_probability :
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8
  let P := P_X * P_Y * P_not_Z
  P = (1 : ℝ) / 32 :=
by
  -- Definitions based on problem conditions
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8

  -- Calculate the combined probability
  let P := P_X * P_Y * P_not_Z
  
  -- Check equality with 1/32
  have h : P = (1 : ℝ) / 32 := by sorry
  exact h

end NUMINAMATH_GPT_xy_not_z_probability_l2135_213503


namespace NUMINAMATH_GPT_find_m_l2135_213564

-- Define the conditions
variables {m x1 x2 : ℝ}

-- Given the equation x^2 + mx - 1 = 0 has roots x1 and x2:
-- The sum of the roots x1 + x2 is -m, and the product of the roots x1 * x2 is -1.
-- Furthermore, given that 1/x1 + 1/x2 = -3,
-- Prove that m = -3.

theorem find_m :
  (x1 + x2 = -m) →
  (x1 * x2 = -1) →
  (1 / x1 + 1 / x2 = -3) →
  m = -3 := by
  intros hSum hProd hRecip
  sorry

end NUMINAMATH_GPT_find_m_l2135_213564


namespace NUMINAMATH_GPT_bridge_height_at_distance_l2135_213505

theorem bridge_height_at_distance :
  (∃ (a : ℝ), ∀ (x : ℝ), (x = 25) → (a * x^2 + 25 = 0)) →
  (∀ (x : ℝ), (x = 10) → (-1/25 * x^2 + 25 = 21)) :=
by
  intro h1
  intro x h2
  have h : 625 * (-1 / 25) * (-1 / 25) = -25 := sorry
  sorry

end NUMINAMATH_GPT_bridge_height_at_distance_l2135_213505


namespace NUMINAMATH_GPT_f_is_odd_l2135_213502

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  (1 + sin x)^(2 * n) - (1 - sin x)^(2 * n)

theorem f_is_odd (n : ℕ) (h : n > 0) : ∀ x : ℝ, f (-x) n = -f x n :=
by
  intros x
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_is_odd_l2135_213502


namespace NUMINAMATH_GPT_correct_sequence_l2135_213520

def step1 := "Collect the admission ticket"
def step2 := "Register"
def step3 := "Written and computer-based tests"
def step4 := "Photography"

theorem correct_sequence : [step2, step4, step1, step3] = ["Register", "Photography", "Collect the admission ticket", "Written and computer-based tests"] :=
by
  sorry

end NUMINAMATH_GPT_correct_sequence_l2135_213520


namespace NUMINAMATH_GPT_sum_of_first_seven_primes_mod_eighth_prime_l2135_213529

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_seven_primes_mod_eighth_prime_l2135_213529


namespace NUMINAMATH_GPT_blake_lollipops_count_l2135_213504

theorem blake_lollipops_count (lollipop_cost : ℕ) (choc_cost_per_pack : ℕ) 
  (chocolate_packs : ℕ) (total_paid : ℕ) (change_received : ℕ) 
  (total_spent : ℕ) (total_choc_cost : ℕ) (remaining_amount : ℕ) 
  (lollipop_count : ℕ) : 
  lollipop_cost = 2 →
  choc_cost_per_pack = 4 * lollipop_cost →
  chocolate_packs = 6 →
  total_paid = 6 * 10 →
  change_received = 4 →
  total_spent = total_paid - change_received →
  total_choc_cost = chocolate_packs * choc_cost_per_pack →
  remaining_amount = total_spent - total_choc_cost →
  lollipop_count = remaining_amount / lollipop_cost →
  lollipop_count = 4 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_blake_lollipops_count_l2135_213504


namespace NUMINAMATH_GPT_equilateral_triangle_l2135_213598

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : α + β + γ = π)
  (h8 : a = 2 * Real.sin α)
  (h9 : b = 2 * Real.sin β)
  (h10 : c = 2 * Real.sin γ)
  (h11 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_l2135_213598


namespace NUMINAMATH_GPT_fraction_six_power_l2135_213523

theorem fraction_six_power (n : ℕ) (hyp : n = 6 ^ 2024) : n / 6 = 6 ^ 2023 :=
by sorry

end NUMINAMATH_GPT_fraction_six_power_l2135_213523


namespace NUMINAMATH_GPT_priyas_fathers_age_l2135_213537

-- Define Priya's age P and her father's age F
variables (P F : ℕ)

-- Define the conditions
def conditions : Prop :=
  F - P = 31 ∧ P + F = 53

-- Define the theorem to be proved
theorem priyas_fathers_age (h : conditions P F) : F = 42 :=
sorry

end NUMINAMATH_GPT_priyas_fathers_age_l2135_213537


namespace NUMINAMATH_GPT_smallest_number_is_111111_2_l2135_213555

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end NUMINAMATH_GPT_smallest_number_is_111111_2_l2135_213555


namespace NUMINAMATH_GPT_sandy_painting_area_l2135_213568

theorem sandy_painting_area :
  let wall_height := 10
  let wall_length := 15
  let painting_height := 3
  let painting_length := 5
  let wall_area := wall_height * wall_length
  let painting_area := painting_height * painting_length
  let area_to_paint := wall_area - painting_area
  area_to_paint = 135 := 
by 
  sorry

end NUMINAMATH_GPT_sandy_painting_area_l2135_213568


namespace NUMINAMATH_GPT_loss_percentage_is_20_l2135_213511

-- Define necessary conditions
def CP : ℕ := 2000
def gain_percent : ℕ := 6
def SP_new : ℕ := CP + ((gain_percent * CP) / 100)
def increase : ℕ := 520

-- Define the selling price condition
def SP : ℕ := SP_new - increase

-- Define the loss percentage condition
def loss_percent : ℕ := ((CP - SP) * 100) / CP

-- Prove the loss percentage is 20%
theorem loss_percentage_is_20 : loss_percent = 20 :=
by sorry

end NUMINAMATH_GPT_loss_percentage_is_20_l2135_213511


namespace NUMINAMATH_GPT_reduced_price_per_dozen_bananas_l2135_213525

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price_one_banana (P : ℝ) := 0.60 * P
noncomputable def number_bananas_original (P : ℝ) := 40 / P
noncomputable def number_bananas_reduced (P : ℝ) := 40 / (0.60 * P)
noncomputable def difference_bananas (P : ℝ) := (number_bananas_reduced P) - (number_bananas_original P)

theorem reduced_price_per_dozen_bananas 
  (P : ℝ) 
  (h1 : difference_bananas P = 67) 
  (h2 : P = 16 / 40.2) :
  12 * reduced_price_one_banana P = 2.856 :=
sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_bananas_l2135_213525


namespace NUMINAMATH_GPT_batsman_total_score_eq_120_l2135_213549

/-- A batsman's runs calculation including boundaries, sixes, and running between wickets. -/
def batsman_runs_calculation (T : ℝ) : Prop :=
  let runs_from_boundaries := 5 * 4
  let runs_from_sixes := 5 * 6
  let runs_from_total := runs_from_boundaries + runs_from_sixes
  let runs_from_running := 0.5833333333333334 * T
  T = runs_from_total + runs_from_running

theorem batsman_total_score_eq_120 :
  ∃ T : ℝ, batsman_runs_calculation T ∧ T = 120 :=
sorry

end NUMINAMATH_GPT_batsman_total_score_eq_120_l2135_213549


namespace NUMINAMATH_GPT_fraction_zero_l2135_213546

theorem fraction_zero (x : ℝ) (h : x ≠ 1) (h₁ : (x + 1) / (x - 1) = 0) : x = -1 :=
sorry

end NUMINAMATH_GPT_fraction_zero_l2135_213546


namespace NUMINAMATH_GPT_min_rectangles_needed_l2135_213588

theorem min_rectangles_needed 
  (type1_corners type2_corners : ℕ)
  (rectangles_cover : ℕ → ℕ)
  (h1 : type1_corners = 12)
  (h2 : type2_corners = 12)
  (h3 : ∀ n, rectangles_cover (3 * n) = n) : 
  (rectangles_cover type2_corners) + (rectangles_cover type1_corners) = 12 := 
sorry

end NUMINAMATH_GPT_min_rectangles_needed_l2135_213588


namespace NUMINAMATH_GPT_muffin_cost_is_correct_l2135_213545

variable (M : ℝ)

def total_original_cost (muffin_cost : ℝ) : ℝ := 3 * muffin_cost + 1.45

def discounted_cost (original_cost : ℝ) : ℝ := 0.85 * original_cost

def kevin_paid (discounted_price : ℝ) : Prop := discounted_price = 3.70

theorem muffin_cost_is_correct (h : discounted_cost (total_original_cost M) = 3.70) : M = 0.97 :=
  by
  sorry

end NUMINAMATH_GPT_muffin_cost_is_correct_l2135_213545


namespace NUMINAMATH_GPT_jordan_walk_distance_l2135_213518

theorem jordan_walk_distance
  (d t : ℝ)
  (flat_speed uphill_speed walk_speed : ℝ)
  (total_time : ℝ)
  (h1 : flat_speed = 18)
  (h2 : uphill_speed = 6)
  (h3 : walk_speed = 4)
  (h4 : total_time = 3)
  (h5 : d / (3 * 18) + d / (3 * 6) + d / (3 * 4) = total_time) :
  t = 6.6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jordan_walk_distance_l2135_213518


namespace NUMINAMATH_GPT_evaluate_expression_l2135_213509

theorem evaluate_expression (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2135_213509


namespace NUMINAMATH_GPT_initial_necklaces_count_l2135_213535

theorem initial_necklaces_count (N : ℕ) 
  (h1 : N - 13 = 37) : 
  N = 50 := 
by
  sorry

end NUMINAMATH_GPT_initial_necklaces_count_l2135_213535


namespace NUMINAMATH_GPT_tetrahedron_volume_correct_l2135_213544

noncomputable def tetrahedron_volume (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABD_ABC : ℝ) : ℝ :=
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABD_ABC
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct:
  tetrahedron_volume 3 15 12 (Real.pi / 6) = 20 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_correct_l2135_213544


namespace NUMINAMATH_GPT_find_second_number_l2135_213526

theorem find_second_number
  (first_number : ℕ)
  (second_number : ℕ)
  (h1 : first_number = 45)
  (h2 : first_number / second_number = 5) : second_number = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_second_number_l2135_213526


namespace NUMINAMATH_GPT_percent_defective_units_shipped_for_sale_l2135_213589

variable (total_units : ℕ)
variable (defective_units_percentage : ℝ := 0.08)
variable (shipped_defective_units_percentage : ℝ := 0.05)

theorem percent_defective_units_shipped_for_sale :
  defective_units_percentage * shipped_defective_units_percentage * 100 = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_percent_defective_units_shipped_for_sale_l2135_213589


namespace NUMINAMATH_GPT_problem_I_problem_II_l2135_213548

theorem problem_I (a b p : ℝ) (F_2 M : ℝ × ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : p > 0)
(h4 : (F_2.1)^2 / a^2 + (F_2.2)^2 / b^2 = 1)
(h5 : M.2^2 = 2 * p * M.1)
(h6 : M.1 = abs (M.2 - F_2.2) - 1)
(h7 : (|F_2.1 - 1|) = 5 / 2) :
    p = 2 ∧ ∃ f : ℝ × ℝ, (f.1)^2 / 9 + (f.2)^2 / 8 = 1 := sorry

theorem problem_II (k m x_0 : ℝ) 
(h8 : k ≠ 0) 
(h9 : m ≠ 0) 
(h10 : km = 1) 
(h11: ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m) ∧
    ((A.1)^2 / 9 + (A.2)^2 / 8 = 1) ∧
    ((B.1)^2 / 9 + (B.2)^2 / 8 = 1) ∧
    (x_0 = (A.1 + B.1) / 2)) :
  -1 < x_0 ∧ x_0 < 0 := sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2135_213548


namespace NUMINAMATH_GPT_number_of_marbles_removed_and_replaced_l2135_213576

def bag_contains_red_marbles (r : ℕ) : Prop := r = 12
def total_marbles (t : ℕ) : Prop := t = 48
def probability_not_red_twice (r t : ℕ) : Prop := ((t - r) / t : ℝ) * ((t - r) / t) = 9 / 16

theorem number_of_marbles_removed_and_replaced (r t : ℕ)
  (hr : bag_contains_red_marbles r)
  (ht : total_marbles t)
  (hp : probability_not_red_twice r t) :
  2 = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_marbles_removed_and_replaced_l2135_213576


namespace NUMINAMATH_GPT_first_five_terms_series_l2135_213593

theorem first_five_terms_series (a : ℕ → ℚ) (h : ∀ n, a n = 1 / (n * (n + 1))) :
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 6) ∧
  (a 3 = 1 / 12) ∧
  (a 4 = 1 / 20) ∧
  (a 5 = 1 / 30) :=
by
  sorry

end NUMINAMATH_GPT_first_five_terms_series_l2135_213593


namespace NUMINAMATH_GPT_solve_equation_l2135_213522

theorem solve_equation : ∀ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) → x = 1 :=
by
  intro x
  intro h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_solve_equation_l2135_213522


namespace NUMINAMATH_GPT_determine_scores_l2135_213591

variables {M Q S K : ℕ}

theorem determine_scores (h1 : Q > M ∨ K > M) 
                          (h2 : M ≠ K) 
                          (h3 : S ≠ Q) 
                          (h4 : S ≠ M) : 
  (Q, S, M) = (Q, S, M) :=
by
  -- We state the theorem as true
  sorry

end NUMINAMATH_GPT_determine_scores_l2135_213591


namespace NUMINAMATH_GPT_solve_equation_l2135_213596

theorem solve_equation (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60 → x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2135_213596


namespace NUMINAMATH_GPT_induction_step_divisibility_l2135_213574

theorem induction_step_divisibility {x y : ℤ} (k : ℕ) (h : ∀ n, n = 2*k - 1 → (x^n + y^n) % (x+y) = 0) :
  (x^(2*k+1) + y^(2*k+1)) % (x+y) = 0 :=
sorry

end NUMINAMATH_GPT_induction_step_divisibility_l2135_213574


namespace NUMINAMATH_GPT_percentage_increase_l2135_213540

theorem percentage_increase (P : ℝ) (h : 200 * (1 + P/100) * 0.70 = 182) : 
  P = 30 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l2135_213540


namespace NUMINAMATH_GPT_maximum_take_home_pay_l2135_213578

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - ((x + 10) / 100 * 1000 * x)

theorem maximum_take_home_pay : 
  ∃ x : ℝ, (take_home_pay x = 20250) ∧ (45000 = 1000 * x) :=
by
  sorry

end NUMINAMATH_GPT_maximum_take_home_pay_l2135_213578


namespace NUMINAMATH_GPT_min_value_range_l2135_213510

theorem min_value_range:
  ∀ (x m n : ℝ), 
    (y = (3 * x + 2) / (x - 1)) → 
    (∀ x ∈ Set.Ioo m n, y ≥ 3 + 5 / (x - 1)) → 
    (y = 8) → 
    n = 2 → 
    (1 ≤ m ∧ m < 2) := by
  sorry

end NUMINAMATH_GPT_min_value_range_l2135_213510


namespace NUMINAMATH_GPT_cost_per_piece_l2135_213599

-- Definitions based on the problem conditions
def total_cost : ℕ := 80         -- Total cost is $80
def num_pizzas : ℕ := 4          -- Luigi bought 4 pizzas
def pieces_per_pizza : ℕ := 5    -- Each pizza was cut into 5 pieces

-- Main theorem statement proving the cost per piece
theorem cost_per_piece :
  (total_cost / (num_pizzas * pieces_per_pizza)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_piece_l2135_213599


namespace NUMINAMATH_GPT_find_incorrect_option_l2135_213519

-- The given conditions from the problem
def incomes : List ℝ := [2, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 13]
def mean_incorrect : Prop := (incomes.sum / incomes.length) = 4
def option_incorrect : Prop := ¬ mean_incorrect

-- The goal is to prove that the statement about the mean being 4 is incorrect
theorem find_incorrect_option : option_incorrect := by
  sorry

end NUMINAMATH_GPT_find_incorrect_option_l2135_213519


namespace NUMINAMATH_GPT_range_of_a_l2135_213501

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2135_213501


namespace NUMINAMATH_GPT_work_completion_days_l2135_213500

theorem work_completion_days (A B C : ℕ) (A_rate B_rate C_rate : ℚ) :
  A_rate = 1 / 30 → B_rate = 1 / 55 → C_rate = 1 / 45 →
  1 / (A_rate + B_rate + C_rate) = 55 / 4 :=
by
  intro hA hB hC
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_work_completion_days_l2135_213500


namespace NUMINAMATH_GPT_alberto_spent_more_l2135_213524

noncomputable def alberto_total_before_discount : ℝ := 2457 + 374 + 520
noncomputable def alberto_discount : ℝ := 0.05 * alberto_total_before_discount
noncomputable def alberto_total_after_discount : ℝ := alberto_total_before_discount - alberto_discount

noncomputable def samara_total_before_tax : ℝ := 25 + 467 + 79 + 150
noncomputable def samara_tax : ℝ := 0.07 * samara_total_before_tax
noncomputable def samara_total_after_tax : ℝ := samara_total_before_tax + samara_tax

noncomputable def amount_difference : ℝ := alberto_total_after_discount - samara_total_after_tax

theorem alberto_spent_more : amount_difference = 2411.98 :=
by
  sorry

end NUMINAMATH_GPT_alberto_spent_more_l2135_213524


namespace NUMINAMATH_GPT_rectangle_sides_l2135_213580

theorem rectangle_sides (a b : ℝ) (h₁ : a < b) (h₂ : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 :=
sorry

end NUMINAMATH_GPT_rectangle_sides_l2135_213580


namespace NUMINAMATH_GPT_donny_cost_of_apples_l2135_213527

def cost_of_apples (small_cost medium_cost big_cost : ℝ) (n_small n_medium n_big : ℕ) : ℝ := 
  n_small * small_cost + n_medium * medium_cost + n_big * big_cost

theorem donny_cost_of_apples :
  cost_of_apples 1.5 2 3 6 6 8 = 45 :=
by
  sorry

end NUMINAMATH_GPT_donny_cost_of_apples_l2135_213527


namespace NUMINAMATH_GPT_max_projection_area_l2135_213550

noncomputable def maxProjectionArea (a : ℝ) : ℝ :=
  if a > (Real.sqrt 3 / 3) ∧ a <= (Real.sqrt 3 / 2) then
    Real.sqrt 3 / 4
  else if a >= (Real.sqrt 3 / 2) then
    a / 2
  else 
    0  -- if the condition for a is not met, it's an edge case which shouldn't logically occur here

theorem max_projection_area (a : ℝ) (h1 : a > Real.sqrt 3 / 3) (h2 : a <= Real.sqrt 3 / 2 ∨ a >= Real.sqrt 3 / 2) :
  maxProjectionArea a = 
    if a > Real.sqrt 3 / 3 ∧ a <= Real.sqrt 3 / 2 then Real.sqrt 3 / 4
    else if a >= Real.sqrt 3 / 2 then a / 2
    else
      sorry :=
by sorry

end NUMINAMATH_GPT_max_projection_area_l2135_213550


namespace NUMINAMATH_GPT_arithmetic_formula_geometric_formula_comparison_S_T_l2135_213521

noncomputable def a₁ : ℕ := 16
noncomputable def d : ℤ := -3

def a_n (n : ℕ) : ℤ := -3 * (n : ℤ) + 19
def b_n (n : ℕ) : ℤ := 4^(3 - n)

def S_n (n : ℕ) : ℚ := (-3 * (n : ℚ)^2 + 35 * n) / 2
def T_n (n : ℕ) : ℤ := -n^2 + 3 * n

theorem arithmetic_formula (n : ℕ) : a_n n = -3 * n + 19 :=
sorry

theorem geometric_formula (n : ℕ) : b_n n = 4^(3 - n) :=
sorry

theorem comparison_S_T (n : ℕ) :
  if n = 29 then S_n n = (T_n n : ℚ)
  else if n < 29 then S_n n > (T_n n : ℚ)
  else S_n n < (T_n n : ℚ) :=
sorry

end NUMINAMATH_GPT_arithmetic_formula_geometric_formula_comparison_S_T_l2135_213521


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2135_213513

noncomputable def f (x a : ℝ) := x^2 - 2*a*x + 1

def no_real_roots (a : ℝ) : Prop := 4*a^2 - 4 < 0

def non_monotonic_interval (a m : ℝ) : Prop := m < a ∧ a < m + 3

def A := {a : ℝ | -1 < a ∧ a < 1}
def B (m : ℝ) := {a : ℝ | m < a ∧ a < m + 3}

theorem sufficient_not_necessary (x : ℝ) (m : ℝ) :
  (x ∈ A → x ∈ B m) → (A ⊆ B m) ∧ (exists a : ℝ, a ∈ B m ∧ a ∉ A) →
  -2 ≤ m ∧ m ≤ -1 := by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2135_213513


namespace NUMINAMATH_GPT_find_all_pairs_l2135_213551

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end NUMINAMATH_GPT_find_all_pairs_l2135_213551


namespace NUMINAMATH_GPT_max_value_of_a_l2135_213538

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (x + 1) * (1 + Real.log (x + 1)) - a * x

theorem max_value_of_a (a : ℤ) : 
  (∀ x : ℝ, x ≥ -1 → (a : ℝ) * x ≤ (x + 1) * (1 + Real.log (x + 1))) → a ≤ 3 := sorry

end NUMINAMATH_GPT_max_value_of_a_l2135_213538


namespace NUMINAMATH_GPT_calculate_expression_l2135_213512

def f (x : ℕ) : ℕ := x^2 - 3*x + 4
def g (x : ℕ) : ℕ := 2*x + 1

theorem calculate_expression : f (g 3) - g (f 3) = 23 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2135_213512


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l2135_213567

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ) 
    (h1 : total_degrees = 360) 
    (h2 : manufacturing_degrees = 126) : 
    (manufacturing_degrees / total_degrees) * 100 = 35 := by
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l2135_213567


namespace NUMINAMATH_GPT_sum_lent_correct_l2135_213570

noncomputable section

-- Define the principal amount (sum lent)
def P : ℝ := 4464.29

-- Define the interest rate per annum
def R : ℝ := 12.0

-- Define the time period in years
def T : ℝ := 12.0

-- Define the interest after 12 years (using the initial conditions and results)
def I : ℝ := 1.44 * P

-- Define the interest given as "2500 less than double the sum lent" condition
def I_condition : ℝ := 2 * P - 2500

-- Theorem stating the sum lent is the given value P
theorem sum_lent_correct : P = 4464.29 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_lent_correct_l2135_213570


namespace NUMINAMATH_GPT_find_c_l2135_213554

theorem find_c (c : ℝ) (h : ∀ x, 2 < x ∧ x < 6 → -x^2 + c * x + 8 > 0) : c = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_c_l2135_213554


namespace NUMINAMATH_GPT_solve_fraction_l2135_213566

theorem solve_fraction (a b : ℝ) (hab : 3 * a = 2 * b) : (a + b) / b = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_l2135_213566


namespace NUMINAMATH_GPT_number_of_people_liking_at_least_one_activity_l2135_213536

def total_people := 200
def people_like_books := 80
def people_like_songs := 60
def people_like_movies := 30
def people_like_books_and_songs := 25
def people_like_books_and_movies := 15
def people_like_songs_and_movies := 20
def people_like_all_three := 10

theorem number_of_people_liking_at_least_one_activity :
  total_people = 200 →
  people_like_books = 80 →
  people_like_songs = 60 →
  people_like_movies = 30 →
  people_like_books_and_songs = 25 →
  people_like_books_and_movies = 15 →
  people_like_songs_and_movies = 20 →
  people_like_all_three = 10 →
  (people_like_books + people_like_songs + people_like_movies -
   people_like_books_and_songs - people_like_books_and_movies -
   people_like_songs_and_movies + people_like_all_three) = 120 := sorry

end NUMINAMATH_GPT_number_of_people_liking_at_least_one_activity_l2135_213536


namespace NUMINAMATH_GPT_ab_value_l2135_213583

   variable (log2_3 : Real) (b : Real) (a : Real)

   -- Hypotheses
   def log_condition : Prop := log2_3 = 1
   def exp_condition (b : Real) : Prop := (4:Real) ^ b = 3
   
   -- Final statement to prove
   theorem ab_value (h_log2_3 : log_condition log2_3) (h_exp : exp_condition b) 
   (ha : a = 1) : a * b = 1 / 2 := sorry
   
end NUMINAMATH_GPT_ab_value_l2135_213583


namespace NUMINAMATH_GPT_percentage_decrease_l2135_213597

-- Define the condition given in the problem
def is_increase (pct : ℤ) : Prop := pct > 0
def is_decrease (pct : ℤ) : Prop := pct < 0

-- The main proof statement
theorem percentage_decrease (pct : ℤ) (h : pct = -10) : is_decrease pct :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l2135_213597


namespace NUMINAMATH_GPT_expansive_sequence_in_interval_l2135_213557

-- Definition of an expansive sequence
def expansive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), (i < j) → (|a i - a j| ≥ 1 / j)

-- Upper bound condition for C
def upper_bound_C (C : ℝ) : Prop :=
  C ≥ 2 * Real.log 2

-- The main statement combining both definitions into a proof problem
theorem expansive_sequence_in_interval (C : ℝ) (a : ℕ → ℝ) 
  (h_exp : expansive_sequence a) (h_bound : upper_bound_C C) :
  ∀ n, 0 ≤ a n ∧ a n ≤ C :=
sorry

end NUMINAMATH_GPT_expansive_sequence_in_interval_l2135_213557


namespace NUMINAMATH_GPT_quadratic_reciprocal_sum_l2135_213508

theorem quadratic_reciprocal_sum :
  ∃ (x1 x2 : ℝ), (x1^2 - 5 * x1 + 4 = 0) ∧ (x2^2 - 5 * x2 + 4 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 5) ∧ (x1 * x2 = 4) ∧ (1 / x1 + 1 / x2 = 5 / 4) :=
sorry

end NUMINAMATH_GPT_quadratic_reciprocal_sum_l2135_213508


namespace NUMINAMATH_GPT_hungarian_olympiad_problem_l2135_213514

-- Define the function A_n as given in the problem
def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n - 1) + 1

-- State the theorem to be proved
theorem hungarian_olympiad_problem (n : ℕ) (h : 0 < n) : 8 ∣ A n :=
by
  sorry

end NUMINAMATH_GPT_hungarian_olympiad_problem_l2135_213514


namespace NUMINAMATH_GPT_fraction_divisible_by_n_l2135_213565

theorem fraction_divisible_by_n (a b n : ℕ) (h1 : a ≠ b) (h2 : n > 0) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_divisible_by_n_l2135_213565


namespace NUMINAMATH_GPT_does_not_pass_through_third_quadrant_l2135_213547

noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

theorem does_not_pass_through_third_quadrant (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  ¬ ∃ x, f a b x < 0 ∧ x < 0 := sorry

end NUMINAMATH_GPT_does_not_pass_through_third_quadrant_l2135_213547


namespace NUMINAMATH_GPT_bert_money_left_l2135_213556

theorem bert_money_left
  (initial_amount : ℝ)
  (spent_hardware_store_fraction : ℝ)
  (amount_spent_dry_cleaners : ℝ)
  (spent_grocery_store_fraction : ℝ)
  (final_amount : ℝ) :
  initial_amount = 44 →
  spent_hardware_store_fraction = 1/4 →
  amount_spent_dry_cleaners = 9 →
  spent_grocery_store_fraction = 1/2 →
  final_amount = initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners - (spent_grocery_store_fraction * (initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners)) →
  final_amount = 12 :=
by
  sorry

end NUMINAMATH_GPT_bert_money_left_l2135_213556


namespace NUMINAMATH_GPT_sum_of_ages_l2135_213531

-- Define Henry's and Jill's present ages
def Henry_age : ℕ := 23
def Jill_age : ℕ := 17

-- Define the condition that 11 years ago, Henry was twice the age of Jill
def condition_11_years_ago : Prop := (Henry_age - 11) = 2 * (Jill_age - 11)

-- Theorem statement: sum of Henry's and Jill's present ages is 40
theorem sum_of_ages : Henry_age + Jill_age = 40 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_sum_of_ages_l2135_213531


namespace NUMINAMATH_GPT_find_c_l2135_213585

theorem find_c (a b c d : ℕ) (h1 : 8 = 4 * a / 100) (h2 : 4 = d * a / 100) (h3 : 8 = d * b / 100) (h4 : c = b / a) : 
  c = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_c_l2135_213585


namespace NUMINAMATH_GPT_bumper_car_rides_correct_l2135_213552

def tickets_per_ride : ℕ := 7
def total_tickets : ℕ := 63
def ferris_wheel_rides : ℕ := 5

def tickets_for_bumper_cars : ℕ :=
  total_tickets - ferris_wheel_rides * tickets_per_ride

def bumper_car_rides : ℕ :=
  tickets_for_bumper_cars / tickets_per_ride

theorem bumper_car_rides_correct : bumper_car_rides = 4 :=
by
  sorry

end NUMINAMATH_GPT_bumper_car_rides_correct_l2135_213552


namespace NUMINAMATH_GPT_num_solutions_of_system_eq_two_l2135_213528

theorem num_solutions_of_system_eq_two : 
  (∃ n : ℕ, n = 2 ∧ ∀ (x y : ℝ), 
    5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16 ↔ 
    (x, y) = ((-90 + Real.sqrt 31900) / 68, 3 * ((-90 + Real.sqrt 31900) / 68) / 5 + 3) ∨ 
    (x, y) = ((-90 - Real.sqrt 31900) / 68, 3 * ((-90 - Real.sqrt 31900) / 68) / 5 + 3)) :=
sorry

end NUMINAMATH_GPT_num_solutions_of_system_eq_two_l2135_213528


namespace NUMINAMATH_GPT_original_chairs_count_l2135_213563

theorem original_chairs_count (n : ℕ) (m : ℕ) :
  (∀ k : ℕ, (k % 4 = 0 → k * (2 * n / 4) = k * (3 * n / 4) ) ∧ 
  (m = (4 / 2) * 15) ∧ (n = (4 * m / (2 * m)) - ((2 * m) / m)) ∧ 
  n + (n + 9) = 72) → n = 63 :=
by
  sorry

end NUMINAMATH_GPT_original_chairs_count_l2135_213563


namespace NUMINAMATH_GPT_exists_multiple_of_power_of_2_with_non_zero_digits_l2135_213560

theorem exists_multiple_of_power_of_2_with_non_zero_digits (n : ℕ) (hn : n ≥ 1) :
  ∃ a : ℕ, (∀ d ∈ a.digits 10, d = 1 ∨ d = 2) ∧ 2^n ∣ a :=
by
  sorry

end NUMINAMATH_GPT_exists_multiple_of_power_of_2_with_non_zero_digits_l2135_213560


namespace NUMINAMATH_GPT_find_height_of_box_l2135_213561

-- Definitions for the problem conditions
def numCubes : ℕ := 24
def volumeCube : ℕ := 27
def lengthBox : ℕ := 8
def widthBox : ℕ := 9
def totalVolumeBox : ℕ := numCubes * volumeCube

-- Problem statement in Lean 4
theorem find_height_of_box : totalVolumeBox = lengthBox * widthBox * 9 :=
by sorry

end NUMINAMATH_GPT_find_height_of_box_l2135_213561


namespace NUMINAMATH_GPT_mul_value_proof_l2135_213579

theorem mul_value_proof :
  ∃ x : ℝ, (8.9 - x = 3.1) ∧ ((x * 3.1) * 2.5 = 44.95) :=
by
  sorry

end NUMINAMATH_GPT_mul_value_proof_l2135_213579


namespace NUMINAMATH_GPT_a_plus_b_eq_neg1_l2135_213542

theorem a_plus_b_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_neg1_l2135_213542


namespace NUMINAMATH_GPT_cube_surface_area_correct_l2135_213543

noncomputable def total_surface_area_of_reassembled_cube : ℝ :=
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let top_bottom_area := 3 * 1 -- Each slab contributes 1 square foot for the top and bottom
  let side_area := 2 * 1 -- Each side slab contributes 1 square foot
  let front_back_area := 2 * 1 -- Each front and back contributes 1 square foot
  top_bottom_area + side_area + front_back_area

theorem cube_surface_area_correct :
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let total_surface_area := total_surface_area_of_reassembled_cube
  total_surface_area = 10 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_correct_l2135_213543


namespace NUMINAMATH_GPT_expressions_equal_l2135_213533

variable (a b c : ℝ)

theorem expressions_equal (h : a + 2 * b + 2 * c = 0) : a + 2 * b * c = (a + 2 * b) * (a + 2 * c) := 
by 
  sorry

end NUMINAMATH_GPT_expressions_equal_l2135_213533


namespace NUMINAMATH_GPT_water_required_l2135_213517

-- Definitions based on the conditions
def balanced_equation : Prop := ∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl + H2O = NH4OH + HCl

-- New problem with the conditions translated into Lean
theorem water_required 
  (h_eq : balanced_equation)
  (n : ℕ)
  (m : ℕ)
  (mole_NH4Cl : n = 2 * m)
  (mole_H2O : m = 2) :
  n = m :=
by
  sorry

end NUMINAMATH_GPT_water_required_l2135_213517


namespace NUMINAMATH_GPT_proof_x_eq_y_l2135_213592

variable (x y z : ℝ)

theorem proof_x_eq_y (h1 : x = 6 - y) (h2 : z^2 = x * y - 9) : x = y := 
  sorry

end NUMINAMATH_GPT_proof_x_eq_y_l2135_213592


namespace NUMINAMATH_GPT_giant_exponent_modulo_result_l2135_213541

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end NUMINAMATH_GPT_giant_exponent_modulo_result_l2135_213541


namespace NUMINAMATH_GPT_group_sum_180_in_range_1_to_60_l2135_213553

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem group_sum_180_in_range_1_to_60 :
  ∃ (a n : ℕ), 1 ≤ a ∧ a + n - 1 ≤ 60 ∧ sum_of_arithmetic_series a 1 n = 180 :=
by
  sorry

end NUMINAMATH_GPT_group_sum_180_in_range_1_to_60_l2135_213553


namespace NUMINAMATH_GPT_general_term_formula_l2135_213562

def sequence_sum (n : ℕ) : ℕ := 3 * n^2 - 2 * n

def general_term (n : ℕ) : ℕ := if n = 0 then 0 else 6 * n - 5

theorem general_term_formula (n : ℕ) (h : n > 0) :
  general_term n = sequence_sum n - sequence_sum (n - 1) := by
  sorry

end NUMINAMATH_GPT_general_term_formula_l2135_213562


namespace NUMINAMATH_GPT_equation_of_parallel_line_l2135_213507

noncomputable def line_parallel_and_intercept (m : ℝ) : Prop :=
  (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ z : ℝ, 3*z + m = 0)

theorem equation_of_parallel_line {m : ℝ} :
  line_parallel_and_intercept m ↔ (∃ x y : ℝ, x + y + 2 = 0) ∨ (∃ x y : ℝ, x + y - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l2135_213507


namespace NUMINAMATH_GPT_shortest_altitude_l2135_213587

theorem shortest_altitude (a b c : ℕ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) (h4 : a^2 + b^2 = c^2) : ∃ x, x = 9.6 :=
by
  sorry

end NUMINAMATH_GPT_shortest_altitude_l2135_213587


namespace NUMINAMATH_GPT_problem_statement_l2135_213506

variable {A B C D E F H : Point}
variable {a b c : ℝ}

-- Assume the conditions
variable (h_triangle : Triangle A B C)
variable (h_acute : AcuteTriangle h_triangle)
variable (h_altitudes : AltitudesIntersectAt h_triangle H A D B E C F)
variable (h_sides : Sides h_triangle BC a AC b AB c)

-- Statement to prove
theorem problem_statement : AH * AD + BH * BE + CH * CF = 1/2 * (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2135_213506


namespace NUMINAMATH_GPT_inequality_system_solution_l2135_213586

theorem inequality_system_solution (x : ℝ) : x + 1 > 0 → x - 3 > 0 → x > 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l2135_213586


namespace NUMINAMATH_GPT_cube_volume_l2135_213539

variable (V_sphere : ℝ)
variable (V_cube : ℝ)
variable (R : ℝ)
variable (a : ℝ)

theorem cube_volume (h1 : V_sphere = (32 / 3) * Real.pi)
    (h2 : V_sphere = (4 / 3) * Real.pi * R^3)
    (h3 : R = 2)
    (h4 : R = (Real.sqrt 3 / 2) * a)
    (h5 : a = 4 * Real.sqrt 3 / 3) :
    V_cube = (4 * Real.sqrt 3 / 3) ^ 3 :=
  by
    sorry

end NUMINAMATH_GPT_cube_volume_l2135_213539
