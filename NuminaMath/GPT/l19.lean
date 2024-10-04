import Mathlib

namespace calculate_floor_100_p_l19_19447

noncomputable def max_prob_sum_7 : ℝ := 
  let p1 := 0.2
  let p6 := 0.1
  let p2_p5_p3_p4 := 0.7 - p1 - p6
  2 * (p1 * p6 + p2_p5_p3_p4 / 2 ^ 2)

theorem calculate_floor_100_p : ∃ p : ℝ, (⌊100 * max_prob_sum_7⌋ = 28) :=
  by
  sorry

end calculate_floor_100_p_l19_19447


namespace largest_multiple_of_15_less_than_500_l19_19169

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19169


namespace arrangement_proof_l19_19654

/-- The Happy Valley Zoo houses 5 chickens, 3 dogs, and 6 cats in a large exhibit area
    with separate but adjacent enclosures. We need to find the number of ways to place
    the 14 animals in a row of 14 enclosures, ensuring all animals of each type are together,
    and that chickens are always placed before cats, but with no restrictions regarding the
    placement of dogs. -/
def number_of_arrangements : ℕ :=
  let chickens := 5
  let dogs := 3
  let cats := 6
  let chicken_permutations := Nat.factorial chickens
  let dog_permutations := Nat.factorial dogs
  let cat_permutations := Nat.factorial cats
  let group_arrangements := 3 -- Chickens-Dogs-Cats, Dogs-Chickens-Cats, Chickens-Cats-Dogs
  group_arrangements * chicken_permutations * dog_permutations * cat_permutations

theorem arrangement_proof : number_of_arrangements = 1555200 :=
by 
  sorry

end arrangement_proof_l19_19654


namespace part_I_part_II_part_III_l19_19592

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem part_I : 
  ∀ x:ℝ, f x = x^3 - x :=
by sorry

theorem part_II : 
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (-1:ℝ) 1 ∧ x2 ∈ Set.Icc (-1:ℝ) 1 ∧ (3 * x1^2 - 1) * (3 * x2^2 - 1) = -1 :=
by sorry

theorem part_III (x_n y_m : ℝ) (hx : x_n ∈ Set.Icc (-1:ℝ) 1) (hy : y_m ∈ Set.Icc (-1:ℝ) 1) : 
  |f x_n - f y_m| < 1 :=
by sorry

end part_I_part_II_part_III_l19_19592


namespace domain_log_base_4_l19_19498

theorem domain_log_base_4 (x : ℝ) : {x // x + 2 > 0} = {x | x > -2} :=
by
  sorry

end domain_log_base_4_l19_19498


namespace total_paved_1120_l19_19869

-- Definitions based on given problem conditions
def workers_paved_april : ℕ := 480
def less_than_march : ℕ := 160
def workers_paved_march : ℕ := workers_paved_april + less_than_march
def total_paved : ℕ := workers_paved_april + workers_paved_march

-- The statement to prove
theorem total_paved_1120 : total_paved = 1120 := by
  sorry

end total_paved_1120_l19_19869


namespace inequality_proof_l19_19705

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + (33 * b^2 - b) / (33 * b^2 + 1) + (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 :=
sorry

end inequality_proof_l19_19705


namespace complex_number_solution_l19_19761

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end complex_number_solution_l19_19761


namespace correct_log_conclusions_l19_19015

variables {x₁ x₂ : ℝ} (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h_diff : x₁ ≠ x₂)
noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem correct_log_conclusions :
  ¬ (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ¬ ((f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by {
  sorry
}

end correct_log_conclusions_l19_19015


namespace sequence_sum_consecutive_l19_19288

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l19_19288


namespace largest_expression_is_A_l19_19679

def expr_A := 1 - 2 + 3 + 4
def expr_B := 1 + 2 - 3 + 4
def expr_C := 1 + 2 + 3 - 4
def expr_D := 1 + 2 - 3 - 4
def expr_E := 1 - 2 - 3 + 4

theorem largest_expression_is_A : expr_A = 6 ∧ expr_A > expr_B ∧ expr_A > expr_C ∧ expr_A > expr_D ∧ expr_A > expr_E :=
  by sorry

end largest_expression_is_A_l19_19679


namespace max_U_value_l19_19739

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end max_U_value_l19_19739


namespace reach_any_position_l19_19672

/-- We define a configuration of marbles in terms of a finite list of natural numbers, which corresponds to the number of marbles in each hole. A configuration transitions to another by moving marbles from one hole to subsequent holes in a circular manner. -/
def configuration (n : ℕ) := List ℕ 

/-- Define the operation of distributing marbles from one hole to subsequent holes. -/
def redistribute (l : configuration n) (i : ℕ) : configuration n :=
  sorry -- The exact redistribution function would need to be implemented based on the conditions.

theorem reach_any_position (n : ℕ) (m : ℕ) (init_config final_config : configuration n)
  (h_num_marbles : init_config.sum = m)
  (h_final_marbles : final_config.sum = m) :
  ∃ steps, final_config = (steps : List ℕ).foldl redistribute init_config :=
sorry

end reach_any_position_l19_19672


namespace total_wheels_correct_l19_19706

def total_wheels (bicycles cars motorcycles tricycles quads : ℕ) 
(missing_bicycle_wheels broken_car_wheels missing_motorcycle_wheels : ℕ) : ℕ :=
  let bicycles_wheels := (bicycles - missing_bicycle_wheels) * 2 + missing_bicycle_wheels
  let cars_wheels := (cars - broken_car_wheels) * 4 + broken_car_wheels * 3
  let motorcycles_wheels := (motorcycles - missing_motorcycle_wheels) * 2
  let tricycles_wheels := tricycles * 3
  let quads_wheels := quads * 4
  bicycles_wheels + cars_wheels + motorcycles_wheels + tricycles_wheels + quads_wheels

theorem total_wheels_correct : total_wheels 25 15 8 3 2 5 2 1 = 134 := 
  by sorry

end total_wheels_correct_l19_19706


namespace arithmetic_geometric_sequence_ratio_l19_19487

section
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables {d : ℝ}

-- Definition of the arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
-- 1. S is the sum of the first n terms of the arithmetic sequence a
axiom sn_arith_seq : sum_arithmetic_sequence S a

-- 2. a_1, a_3, and a_4 form a geometric sequence
axiom geom_seq : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)

-- Goal is to prove the given ratio equation
theorem arithmetic_geometric_sequence_ratio (h : ∀ n, a n = -4 * d + (n - 1) * d) :
  (S 3 - S 2) / (S 5 - S 3) = 2 :=
sorry
end

end arithmetic_geometric_sequence_ratio_l19_19487


namespace smallest_missing_digit_units_place_cube_l19_19219

theorem smallest_missing_digit_units_place_cube :
  ∀ d : Fin 10, ∃ n : ℕ, (n ^ 3) % 10 = d :=
by
  sorry

end smallest_missing_digit_units_place_cube_l19_19219


namespace algebraic_expression_value_l19_19266

noncomputable def a : ℝ := 2 * Real.sin (Real.pi / 4) + 1
noncomputable def b : ℝ := 2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_value :
  ((a^2 + b^2) / (2 * a * b) - 1) / ((a^2 - b^2) / (a^2 * b + a * b^2)) = 1 :=
by sorry

end algebraic_expression_value_l19_19266


namespace line_intersects_x_axis_at_point_l19_19695

theorem line_intersects_x_axis_at_point :
  ∃ x, (4 * x - 2 * 0 = 6) ∧ (2 - 0 = 2 * (0 - x)) → x = 2 := 
by
  sorry

end line_intersects_x_axis_at_point_l19_19695


namespace man_l19_19992

-- Define the conditions
def speed_downstream : ℕ := 8
def speed_upstream : ℕ := 4

-- Define the man's rate in still water
def rate_in_still_water : ℕ := (speed_downstream + speed_upstream) / 2

-- The target theorem
theorem man's_rate_in_still_water : rate_in_still_water = 6 := by
  -- The statement is set up. Proof to be added later.
  sorry

end man_l19_19992


namespace inverse_proposition_l19_19085

-- Define the variables m, n, and a^2
variables (m n : ℝ) (a : ℝ)

-- State the proof problem
theorem inverse_proposition
  (h1 : m > n)
: m * a^2 > n * a^2 :=
sorry

end inverse_proposition_l19_19085


namespace range_of_a_l19_19031

theorem range_of_a (a x y : ℝ)
  (h1 : x + y = 3 * a + 4)
  (h2 : x - y = 7 * a - 4)
  (h3 : 3 * x - 2 * y < 11) : a < 1 :=
sorry

end range_of_a_l19_19031


namespace train_crosses_pole_in_time_l19_19854

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length / speed_ms

theorem train_crosses_pole_in_time :
  ∀ (length speed_kmh : ℝ), length = 240 → speed_kmh = 126 →
    time_to_cross_pole length speed_kmh = 6.8571 :=
by
  intros length speed_kmh h_length h_speed
  rw [h_length, h_speed, time_to_cross_pole]
  sorry

end train_crosses_pole_in_time_l19_19854


namespace largest_circle_radius_l19_19727

theorem largest_circle_radius (a b c : ℝ) (h : a > b ∧ b > c) :
  ∃ radius : ℝ, radius = b :=
by
  sorry

end largest_circle_radius_l19_19727


namespace initial_number_of_persons_l19_19319

-- Define the conditions and the goal
def weight_increase_due_to_new_person : ℝ := 102 - 75
def average_weight_increase (n : ℝ) : ℝ := 4.5 * n

theorem initial_number_of_persons (n : ℝ) (h1 : average_weight_increase n = weight_increase_due_to_new_person) : n = 6 :=
by
  -- Skip the proof with sorry
  sorry

end initial_number_of_persons_l19_19319


namespace triangle_angles_l19_19455

variable (a b c t : ℝ)

def angle_alpha : ℝ := 43

def area_condition (α β : ℝ) : Prop :=
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β)

theorem triangle_angles (α β γ : ℝ) (hα : α = angle_alpha) (h_area : area_condition a b t α β) :
  α = 43 ∧ β = 17 ∧ γ = 120 := sorry

end triangle_angles_l19_19455


namespace lim_integral_fn_l19_19589

def fn (n : ℕ) (x : ℝ) : ℝ := Real.arctan (⌊x⌋)

theorem lim_integral_fn : ∀ n : ℕ, (fn n) is RiemannIntegrable ∧ (tendsto (λ n, (1 : ℝ) / (n : ℝ) * ∫(0:ℝ)..(n:ℝ), fn n) atTop (𝓝 (Real.pi / 2))) :=
by
  sorry

end lim_integral_fn_l19_19589


namespace cards_total_l19_19440

theorem cards_total (janet_brenda_diff : ℕ) (mara_janet_mult : ℕ) (mara_less_150 : ℕ) (h1 : janet_brenda_diff = 9) (h2 : mara_janet_mult = 2) (h3 : mara_less_150 = 40) : 
  let brenda := (150 - mara_less_150) / 2 - janet_brenda_diff in
  let janet := brenda + janet_brenda_diff in
  let mara := janet * mara_janet_mult in
  brenda + janet + mara = 211 :=
by
  intros
  simp [janet_brenda_diff, mara_janet_mult, mara_less_150]
  sorry

end cards_total_l19_19440


namespace largest_multiple_of_15_less_than_500_l19_19109

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19109


namespace point_C_lies_within_region_l19_19704

def lies_within_region (x y : ℝ) : Prop :=
  (x + y - 1 < 0) ∧ (x - y + 1 > 0)

theorem point_C_lies_within_region : lies_within_region 0 (-2) :=
by {
  -- Proof is omitted as per the instructions
  sorry
}

end point_C_lies_within_region_l19_19704


namespace exists_m_divisible_by_1988_l19_19619

def f (x : ℕ) : ℕ := 3 * x + 2
def iter_function (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iter_function n x)

theorem exists_m_divisible_by_1988 : ∃ m : ℕ, 1988 ∣ iter_function 100 m :=
by sorry

end exists_m_divisible_by_1988_l19_19619


namespace sum_of_three_numbers_l19_19840

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19840


namespace staff_meeting_doughnuts_l19_19250

theorem staff_meeting_doughnuts (n_d n_s n_l : ℕ) (h₁ : n_d = 50) (h₂ : n_s = 19) (h₃ : n_l = 12) :
  (n_d - n_l) / n_s = 2 :=
by
  sorry

end staff_meeting_doughnuts_l19_19250


namespace conversion_base8_to_base10_l19_19554

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l19_19554


namespace sqrt_720_simplified_l19_19649

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l19_19649


namespace parallel_lines_condition_l19_19856

theorem parallel_lines_condition (a : ℝ) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x y : ℝ, l1 x + a * y + 6 = 0) ∧ 
    (∀ x y : ℝ, (a - 2) * x + 3 * y + 2 * a = 0) ∧
    l1 = l2 ↔ a = 3) :=
sorry

end parallel_lines_condition_l19_19856


namespace solve_for_x_l19_19957

theorem solve_for_x (x : ℝ) : 
  5 * x + 9 * x = 420 - 12 * (x - 4) -> 
  x = 18 :=
by
  intro h
  -- derivation will follow here
  sorry

end solve_for_x_l19_19957


namespace available_space_on_usb_l19_19070

theorem available_space_on_usb (total_capacity : ℕ) (used_percentage : ℝ) (total_capacity = 16) (used_percentage = 0.5) : 
  (total_capacity * (1 - used_percentage) = 8) := sorry

end available_space_on_usb_l19_19070


namespace find_counterfeit_l19_19488

-- Definitions based on the conditions
structure Coin :=
(weight : ℝ)
(is_genuine : Bool)

def is_counterfeit (coins : List Coin) : Prop :=
  ∃ (c : Coin) (h : c ∈ coins), ¬c.is_genuine

def weigh (c1 c2 : Coin) : ℝ := c1.weight - c2.weight

def identify_counterfeit (coins : List Coin) : Prop :=
  ∀ (a b c d : Coin), 
    coins = [a, b, c, d] →
    (¬a.is_genuine ∨ ¬b.is_genuine ∨ ¬c.is_genuine ∨ ¬d.is_genuine) →
    (weigh a b = 0 ∧ weigh c d ≠ 0 ∨ weigh a c = 0 ∧ weigh b d ≠ 0 ∨ weigh a d = 0 ∧ weigh b c ≠ 0) →
    (∃ (fake_coin : Coin), fake_coin ∈ coins ∧ ¬fake_coin.is_genuine)

-- Proof statement
theorem find_counterfeit (coins : List Coin) :
  (∃ (c : Coin), c ∈ coins ∧ ¬c.is_genuine) →
  identify_counterfeit coins :=
by
  sorry

end find_counterfeit_l19_19488


namespace equilateral_triangle_l19_19746

variable {a b c : ℝ}

-- Conditions
def condition1 (a b c : ℝ) : Prop :=
  (a + b + c) * (b + c - a) = 3 * b * c

def condition2 (a b c : ℝ) (cos_B cos_C : ℝ) : Prop :=
  c * cos_B = b * cos_C

-- Theorem statement
theorem equilateral_triangle (a b c : ℝ) (cos_B cos_C : ℝ)
  (h1 : condition1 a b c)
  (h2 : condition2 a b c cos_B cos_C) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l19_19746


namespace Z_real_Z_imaginary_Z_pure_imaginary_l19_19590

-- Definitions

def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2 * a - 15 : ℂ)

-- Statement for the proof problems

theorem Z_real (a : ℝ) : 
  (Z a).im = 0 ↔ a = 5 ∨ a = -3 := sorry

theorem Z_imaginary (a : ℝ) : 
  (Z a).re = 0 ↔ a ≠ 5 ∧ a ≠ -3 := sorry

theorem Z_pure_imaginary (a : ℝ) : 
  (Z a).re = 0 ∧ (Z a).im ≠ 0 ↔ a = 3 := sorry

end Z_real_Z_imaginary_Z_pure_imaginary_l19_19590


namespace root_expression_value_l19_19303

variables (a b : ℝ)
noncomputable def quadratic_eq (a b : ℝ) : Prop := (a + b = 1 ∧ a * b = -1)

theorem root_expression_value (h : quadratic_eq a b) : 3 * a ^ 2 + 4 * b + (2 / a ^ 2) = 11 := sorry

end root_expression_value_l19_19303


namespace amount_of_CaO_required_l19_19725

theorem amount_of_CaO_required (n_H2O : ℝ) (n_CaOH2 : ℝ) (n_CaO : ℝ) 
  (h1 : n_H2O = 2) (h2 : n_CaOH2 = 2) :
  n_CaO = 2 :=
by
  sorry

end amount_of_CaO_required_l19_19725


namespace evaporation_period_length_l19_19689

def initial_water_amount : ℝ := 10
def daily_evaporation_rate : ℝ := 0.0008
def percentage_evaporated : ℝ := 0.004  -- 0.4% expressed as a decimal

theorem evaporation_period_length :
  (percentage_evaporated * initial_water_amount) / daily_evaporation_rate = 50 := by
  sorry

end evaporation_period_length_l19_19689


namespace sequence_sum_consecutive_l19_19289

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l19_19289


namespace usb_drive_available_space_l19_19071

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end usb_drive_available_space_l19_19071


namespace least_possible_average_of_integers_l19_19088

theorem least_possible_average_of_integers :
  ∃ (a b c d : ℤ), a < b ∧ b < c ∧ c < d ∧ d = 90 ∧ a ≥ 21 ∧ (a + b + c + d) / 4 = 39 := by
sorry

end least_possible_average_of_integers_l19_19088


namespace curve_not_parabola_l19_19424

theorem curve_not_parabola (k : ℝ) : ¬(∃ a b c : ℝ, a ≠ 0 ∧ x^2 + ky^2 = a*x^2 + b*y + c) :=
sorry

end curve_not_parabola_l19_19424


namespace range_of_x_for_odd_monotonic_function_l19_19029

theorem range_of_x_for_odd_monotonic_function 
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_increasing_on_R : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  ∀ x : ℝ, (0 < x) → ( (|f (Real.log x) - f (Real.log (1 / x))| / 2) < f 1 ) → (Real.exp (-1) < x ∧ x < Real.exp 1) := 
by
  sorry

end range_of_x_for_odd_monotonic_function_l19_19029


namespace total_enemies_l19_19614

theorem total_enemies (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) (enemies_defeated : ℕ) :  
  (3 = points_per_enemy) → 
  (12 = points_earned) → 
  (2 = enemies_left) → 
  (points_earned / points_per_enemy = enemies_defeated) → 
  (enemies_defeated + enemies_left = 6) := 
by
  intros
  sorry

end total_enemies_l19_19614


namespace cereal_discount_l19_19231

theorem cereal_discount (milk_normal_cost milk_discounted_cost total_savings milk_quantity cereal_quantity: ℝ) 
  (total_milk_savings cereal_savings_per_box: ℝ) 
  (h1: milk_normal_cost = 3)
  (h2: milk_discounted_cost = 2)
  (h3: total_savings = 8)
  (h4: milk_quantity = 3)
  (h5: cereal_quantity = 5)
  (h6: total_milk_savings = milk_quantity * (milk_normal_cost - milk_discounted_cost)) 
  (h7: total_milk_savings + cereal_quantity * cereal_savings_per_box = total_savings):
  cereal_savings_per_box = 1 :=
by 
  sorry

end cereal_discount_l19_19231


namespace largest_multiple_of_15_less_than_500_l19_19113

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19113


namespace radius_is_100_div_pi_l19_19256

noncomputable def radius_of_circle (L : ℝ) (θ : ℝ) : ℝ :=
  L * 360 / (θ * 2 * Real.pi)

theorem radius_is_100_div_pi :
  radius_of_circle 25 45 = 100 / Real.pi := 
by
  sorry

end radius_is_100_div_pi_l19_19256


namespace alexandra_magazines_l19_19536

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l19_19536


namespace grid_arrangement_count_l19_19392

def is_valid_grid (grid : Matrix ℕ (Fin 3) (Fin 3)) : Prop :=
  let all_nums := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let rows_sum := List.map (fun i => List.sum (List.map (fun j => grid i j) [0, 1, 2])) [0, 1, 2]
  let cols_sum := List.map (fun j => List.sum (List.map (fun i => grid i j) [0, 1, 2])) [0, 1, 2]
  List.Sort (List.join (List.map (List.map grid) [0, 1, 2])) = all_nums ∧
  rows_sum = [15, 15, 15] ∧ cols_sum = [15, 15, 15]

theorem grid_arrangement_count : 
  let valid_grids := { grid : Matrix ℕ (Fin 3) (Fin 3) | is_valid_grid grid }
  valid_grids.card = 72 :=
sorry

end grid_arrangement_count_l19_19392


namespace money_left_is_41_l19_19801

-- Define the amounts saved by Tanner in each month
def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25

-- Define the amount spent by Tanner on the video game
def spent_video_game : ℕ := 49

-- Total savings after the three months
def total_savings : ℕ := savings_september + savings_october + savings_november

-- Calculate the money left after spending on the video game
def money_left : ℕ := total_savings - spent_video_game

-- The theorem we need to prove
theorem money_left_is_41 : money_left = 41 := by
  sorry

end money_left_is_41_l19_19801


namespace smallest_bdf_l19_19937

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l19_19937


namespace largest_multiple_of_15_less_than_500_is_495_l19_19156

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19156


namespace negation_of_proposition_p_l19_19947

def has_real_root (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

def negation_of_p : Prop := ∀ m : ℝ, ¬ has_real_root m

theorem negation_of_proposition_p : negation_of_p :=
by sorry

end negation_of_proposition_p_l19_19947


namespace average_of_last_four_numbers_l19_19474

theorem average_of_last_four_numbers
  (seven_avg : ℝ)
  (first_three_avg : ℝ)
  (seven_avg_is_62 : seven_avg = 62)
  (first_three_avg_is_58 : first_three_avg = 58) :
  (7 * seven_avg - 3 * first_three_avg) / 4 = 65 :=
by
  rw [seven_avg_is_62, first_three_avg_is_58]
  sorry

end average_of_last_four_numbers_l19_19474


namespace problem1_proof_problem2_proof_l19_19548

-- Problem 1 proof statement
theorem problem1_proof : (-1)^10 * 2 + (-2)^3 / 4 = 0 := 
by
  sorry

-- Problem 2 proof statement
theorem problem2_proof : -24 * (5 / 6 - 4 / 3 + 3 / 8) = 3 :=
by
  sorry

end problem1_proof_problem2_proof_l19_19548


namespace g_property_l19_19621

theorem g_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y) :
  let n := 2
  let s := 14 / 3
  n = 2 ∧ s = 14 / 3 ∧ n * s = 28 / 3 :=
by {
  sorry
}

end g_property_l19_19621


namespace square_side_4_FP_length_l19_19465

theorem square_side_4_FP_length (EF GH EP FP GP : ℝ) :
  EF = 4 ∧ GH = 4 ∧ EP = 4 ∧ GP = 4 ∧
  (1 / 2) * EP * 2 = 4 → FP = 2 * Real.sqrt 5 :=
by
  intro h
  sorry

end square_side_4_FP_length_l19_19465


namespace no_digit_c_make_2C4_multiple_of_5_l19_19397

theorem no_digit_c_make_2C4_multiple_of_5 : ∀ C, ¬ (C ≥ 0 ∧ C ≤ 9 ∧ (20 * 10 + C * 10 + 4) % 5 = 0) :=
by intros C hC; sorry

end no_digit_c_make_2C4_multiple_of_5_l19_19397


namespace complement_A_in_U_l19_19907

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 + x - 2 < 0}

theorem complement_A_in_U :
  (U \ A) = {-2, 1, 2} :=
by 
  -- proof will be done here
  sorry

end complement_A_in_U_l19_19907


namespace total_earrings_l19_19240

-- Definitions based on the given conditions
def bella_earrings : ℕ := 10
def monica_earrings : ℕ := 4 * bella_earrings
def rachel_earrings : ℕ := monica_earrings / 2
def olivia_earrings : ℕ := bella_earrings + monica_earrings + rachel_earrings + 5

-- The theorem to prove the total number of earrings
theorem total_earrings : bella_earrings + monica_earrings + rachel_earrings + olivia_earrings = 145 := by
  sorry

end total_earrings_l19_19240


namespace product_eq_1280_l19_19221

axiom eq1 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48
axiom eq2 (a b c d : ℝ) : 4 * d + 2 * c = 2 * b
axiom eq3 (a b c d : ℝ) : 4 * b + 2 * c = 2 * a
axiom eq4 (a b c d : ℝ) : c - 2 = d
axiom eq5 (a b c d : ℝ) : d + b = 10

theorem product_eq_1280 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48 → 4 * d + 2 * c = 2 * b → 4 * b + 2 * c = 2 * a → c - 2 = d → d + b = 10 → a * b * c * d = 1280 :=
by 
  intro h1 h2 h3 h4 h5
  -- we put the proof here
  sorry

end product_eq_1280_l19_19221


namespace largest_multiple_of_15_less_than_500_l19_19185

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19185


namespace largest_multiple_of_15_less_than_500_l19_19140

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19140


namespace average_height_of_trees_l19_19295

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l19_19295


namespace find_second_divisor_l19_19522

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l19_19522


namespace wrapping_paper_area_l19_19527

variable (l w h : ℝ)
variable (l_gt_w : l > w)

def area_wrapping_paper (l w h : ℝ) : ℝ :=
  3 * (l + w) * h

theorem wrapping_paper_area :
  area_wrapping_paper l w h = 3 * (l + w) * h :=
sorry

end wrapping_paper_area_l19_19527


namespace no_500_good_trinomials_l19_19879

def is_good_quadratic_trinomial (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b^2 - 4 * a * c) > 0

theorem no_500_good_trinomials (S : Finset ℤ) (hS: S.card = 10)
  (hs_pos: ∀ x ∈ S, x > 0) : ¬(∃ T : Finset (ℤ × ℤ × ℤ), 
  T.card = 500 ∧ (∀ (a b c : ℤ), (a, b, c) ∈ T → is_good_quadratic_trinomial a b c)) :=
by
  sorry

end no_500_good_trinomials_l19_19879


namespace largest_multiple_15_under_500_l19_19123

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19123


namespace find_value_of_s_l19_19032

variable {r s : ℝ}

theorem find_value_of_s (hr : r > 1) (hs : s > 1) (h1 : 1/r + 1/s = 1) (h2 : r * s = 9) :
  s = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_value_of_s_l19_19032


namespace largest_multiple_of_15_less_than_500_l19_19163

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19163


namespace number_of_zeros_of_f_l19_19267

noncomputable def f (a x : ℝ) := x * Real.log x - a * x^2 - x

theorem number_of_zeros_of_f (a : ℝ) (h : |a| ≥ 1 / (2 * Real.exp 1)) :
  ∃! x, f a x = 0 :=
sorry

end number_of_zeros_of_f_l19_19267


namespace total_length_of_fence_l19_19531

theorem total_length_of_fence (x : ℝ) (h1 : 2 * x * x = 1250) : 2 * x + 2 * x = 100 :=
by
  sorry

end total_length_of_fence_l19_19531


namespace rectangle_dimensions_l19_19819

-- Define the dimensions and properties of the rectangle
variables {a b : ℕ}

-- Theorem statement
theorem rectangle_dimensions 
  (h1 : b = a + 3)
  (h2 : 2 * a + 2 * b + a = a * b) : 
  (a = 3 ∧ b = 6) :=
by
  sorry

end rectangle_dimensions_l19_19819


namespace largest_multiple_of_15_less_than_500_l19_19129

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19129


namespace count_positive_n_l19_19586

def is_factorable (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = -2) ∧ (a * b = - (n:ℤ))

theorem count_positive_n : 
  (∃ (S : Finset ℕ), S.card = 45 ∧ ∀ n ∈ S, (1 ≤ n ∧ n ≤ 2000) ∧ is_factorable n) :=
by
  -- Placeholder for the proof
  sorry

end count_positive_n_l19_19586


namespace total_weekly_messages_l19_19435

theorem total_weekly_messages (n r1 r2 r3 r4 r5 m1 m2 m3 m4 m5 : ℕ) 
(p1 p2 p3 p4 : ℕ) (h1 : n = 200) (h2 : r1 = 15) (h3 : r2 = 25) (h4 : r3 = 10) 
(h5 : r4 = 20) (h6 : r5 = 5) (h7 : m1 = 40) (h8 : m2 = 60) (h9 : m3 = 50) 
(h10 : m4 = 30) (h11 : m5 = 20) (h12 : p1 = 15) (h13 : p2 = 25) (h14 : p3 = 40) 
(h15 : p4 = 10) : 
  let total_members_removed := r1 + r2 + r3 + r4 + r5
  let remaining_members := n - total_members_removed
  let daily_messages :=
        (25 * remaining_members / 100 * p1) +
        (50 * remaining_members / 100 * p2) +
        (20 * remaining_members / 100 * p3) +
        (5 * remaining_members / 100 * p4)
  let weekly_messages := daily_messages * 7
  weekly_messages = 21663 :=
by
  sorry

end total_weekly_messages_l19_19435


namespace solution_set_of_inequalities_l19_19624

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def is_strictly_decreasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Main Statement
theorem solution_set_of_inequalities
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_periodic : is_periodic f 2)
  (h_decreasing : is_strictly_decreasing f 0 1)
  (h_f_pi : f π = 1)
  (h_f_2pi : f (2 * π) = 2) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | π - 2 ≤ x ∧ x ≤ 8 - 2 * π} :=
  sorry

end solution_set_of_inequalities_l19_19624


namespace sum_of_three_numbers_l19_19829

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19829


namespace quadratic_function_symmetry_l19_19924

theorem quadratic_function_symmetry (a b x_1 x_2: ℝ) (h_roots: x_1^2 + a * x_1 + b = 0 ∧ x_2^2 + a * x_2 + b = 0)
(h_symmetry: ∀ x, (x - 2015)^2 + a * (x - 2015) + b = (x + 2015 - 2016)^2 + a * (x + 2015 - 2016) + b):
  (x_1 + x_2) / 2 = 2015 :=
sorry

end quadratic_function_symmetry_l19_19924


namespace ratio_of_girls_l19_19765

theorem ratio_of_girls (total_julian_friends : ℕ) (percent_julian_girls : ℚ)
  (percent_julian_boys : ℚ) (total_boyd_friends : ℕ) (percent_boyd_boys : ℚ) :
  total_julian_friends = 80 →
  percent_julian_girls = 0.40 →
  percent_julian_boys = 0.60 →
  total_boyd_friends = 100 →
  percent_boyd_boys = 0.36 →
  (0.64 * total_boyd_friends : ℚ) / (0.40 * total_julian_friends : ℚ) = 2 :=
by
  sorry

end ratio_of_girls_l19_19765


namespace curve_is_circle_l19_19568

-- Definition of the curve in polar coordinates
def curve (r θ : ℝ) : Prop :=
  r = 3 * Real.sin θ

-- The theorem to prove
theorem curve_is_circle : ∀ θ : ℝ, ∃ r : ℝ, curve r θ → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ p : ℝ × ℝ, (Real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) = R)) :=
by
  sorry

end curve_is_circle_l19_19568


namespace harmonic_sum_base_case_l19_19637

theorem harmonic_sum_base_case : 1 + 1/2 + 1/3 < 2 := 
sorry

end harmonic_sum_base_case_l19_19637


namespace sufficient_but_not_necessary_l19_19275

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x * (x - 1) = 0) ∧ ¬(x * (x - 1) = 0 → x = 1) := 
by
  sorry

end sufficient_but_not_necessary_l19_19275


namespace rectangle_area_l19_19345

theorem rectangle_area (length : ℝ) (width : ℝ) (increased_width : ℝ) (area : ℝ)
  (h1 : length = 12)
  (h2 : increased_width = width * 1.2)
  (h3 : increased_width = 12)
  (h4 : area = length * width) : 
  area = 120 := 
by
  sorry

end rectangle_area_l19_19345


namespace correct_answers_l19_19760

-- Definitions based on conditions a)1 and a)2
def can_A_red : ℕ := 5
def can_A_white : ℕ := 2
def can_A_black : ℕ := 3
def can_B_red : ℕ := 4
def can_B_white : ℕ := 3
def can_B_black : ℕ := 3

-- Total balls in Can A
def total_A : ℕ := can_A_red + can_A_white + can_A_black

-- Total balls in Can B
def total_B (extra : ℕ) : ℕ := can_B_red + can_B_white + can_B_black + extra

-- Probability events based on conditions a)3
def P_A1 : ℚ := can_A_red / total_A
def P_A2 : ℚ := can_A_white / total_A
def P_A3 : ℚ := can_A_black / total_A
def P_B_given_A1 : ℚ := (can_B_red + 1) / (total_B 1)

theorem correct_answers :
  P_B_given_A1 = 5 / 11 ∧
  (P_A1 ≠ P_A2 ∧ P_A1 ≠ P_A3 ∧ P_A2 ≠ P_A3) :=
by {
  sorry -- Proof to be completed
}

end correct_answers_l19_19760


namespace percentage_expression_l19_19800

variable {A B : ℝ} (hA : A > 0) (hB : B > 0)

theorem percentage_expression (h : A = (x / 100) * B) : x = 100 * (A / B) :=
sorry

end percentage_expression_l19_19800


namespace find_y_l19_19098

variable (h : ℕ) -- integral number of hours

-- Distance between A and B
def distance_AB : ℕ := 60

-- Speed and distance walked by woman starting at A
def speed_A : ℕ := 3
def distance_A (h : ℕ) : ℕ := speed_A * h

-- Speed and distance walked by woman starting at B
def speed_B_1st_hour : ℕ := 2
def distance_B (h : ℕ) : ℕ := (h * (h + 3)) / 2

-- Meeting point equation
def meeting_point_eqn (h : ℕ) : Prop := (distance_A h) + (distance_B h) = distance_AB

-- Requirement: y miles nearer to A whereas y = distance_AB - 2 * distance_B (since B meets closer to A by y miles)
def y_nearer_A (h : ℕ) : ℕ := distance_AB - 2 * (distance_A h)

-- Prove y = 6 for the specific value of h
theorem find_y : ∃ (h : ℕ), meeting_point_eqn h ∧ y_nearer_A h = 6 := by
  sorry

end find_y_l19_19098


namespace sum_of_numbers_l19_19826

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19826


namespace least_non_lucky_multiple_of_7_correct_l19_19232

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

def least_non_lucky_multiple_of_7 : ℕ :=
  14

theorem least_non_lucky_multiple_of_7_correct : 
  ¬ is_lucky 14 ∧ ∀ m, m < 14 → m % 7 = 0 → ¬ ¬ is_lucky m :=
by
  sorry

end least_non_lucky_multiple_of_7_correct_l19_19232


namespace train_journey_time_l19_19656

theorem train_journey_time {X : ℝ} (h1 : 0 < X) (h2 : X < 60) (h3 : ∀ T_A M_A T_B M_B : ℝ, M_A - T_A = X ∧ M_B - T_B = X) :
    X = 360 / 7 :=
by
  sorry

end train_journey_time_l19_19656


namespace geometric_figure_perimeter_l19_19245

theorem geometric_figure_perimeter (A : ℝ) (n : ℝ) (area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  A = 216 ∧ n = 6 ∧ area = A / n ∧ side_length = Real.sqrt area ∧ perimeter = 2 * (3 * side_length + 2 * side_length) + 2 * side_length →
  perimeter = 72 := 
by 
  sorry

end geometric_figure_perimeter_l19_19245


namespace card_distribution_methods_l19_19073

theorem card_distribution_methods :
  let cards := [1, 2, 3, 4, 5, 6]
  let envelopes := {A, B, C}

  -- Total number of different methods (ways) to place 6 cards into 3 envelopes
  (∃ (f : fin 6 → fin 3), 
    ∀ (i j : fin 6), 
    (cards[i] = 1 ∨ cards[i] = 2) ∧ (cards[j] = 1 ∨ cards[j] = 2) → f i = f j) = 18 := sorry

end card_distribution_methods_l19_19073


namespace soldiers_first_side_l19_19615

theorem soldiers_first_side (x : ℤ) (h1 : ∀ s1 : ℤ, s1 = 10)
                           (h2 : ∀ s2 : ℤ, s2 = 8)
                           (h3 : ∀ y : ℤ, y = x - 500)
                           (h4 : (10 * x + 8 * (x - 500)) = 68000) : x = 4000 :=
by
  -- Left blank for Lean to fill in the required proof steps
  sorry

end soldiers_first_side_l19_19615


namespace f_neg_a_eq_neg_2_l19_19416

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

-- Given condition: f(a) = 4
variable (a : ℝ)
axiom h_f_a : f a = 4

-- We need to prove that: f(-a) = -2
theorem f_neg_a_eq_neg_2 (a : ℝ) (h_f_a : f a = 4) : f (-a) = -2 :=
by
  sorry

end f_neg_a_eq_neg_2_l19_19416


namespace probability_5800_in_three_spins_l19_19436

def spinner_labels : List String := ["Bankrupt", "$600", "$1200", "$4000", "$800", "$2000", "$150"]

def total_outcomes (spins : Nat) : Nat :=
  let segments := spinner_labels.length
  segments ^ spins

theorem probability_5800_in_three_spins :
  (6 / total_outcomes 3 : ℚ) = 6 / 343 :=
by
  sorry

end probability_5800_in_three_spins_l19_19436


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l19_19976

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l19_19976


namespace max_integer_value_of_expression_l19_19042

theorem max_integer_value_of_expression (x : ℝ) :
  ∃ M : ℤ, M = 15 ∧ ∀ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ M :=
sorry

end max_integer_value_of_expression_l19_19042


namespace sum_of_numbers_l19_19834

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19834


namespace general_term_sum_formula_l19_19897

-- Conditions for the sequence
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a2_eq_5 : a 2 = 5
axiom S4_eq_28 : S 4 = 28

-- The sequence is an arithmetic sequence
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- Statement 1: Proof that a_n = 4n - 3
theorem general_term (n : ℕ) : a n = 4 * n - 3 :=
by
  sorry

-- Statement 2: Proof that S_n = 2n^2 - n
theorem sum_formula (n : ℕ) : S n = 2 * n^2 - n :=
by
  sorry

end general_term_sum_formula_l19_19897


namespace locus_points_eq_distance_l19_19388

def locus_is_parabola (x y : ℝ) : Prop :=
  (y - 1) ^ 2 = 16 * (x - 2)

theorem locus_points_eq_distance (x y : ℝ) :
  locus_is_parabola x y ↔ (x, y) = (4, 1) ∨
    dist (x, y) (4, 1) = dist (x, y) (0, y) :=
by
  sorry

end locus_points_eq_distance_l19_19388


namespace largest_multiple_15_under_500_l19_19119

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19119


namespace grid_arrangement_count_l19_19391

theorem grid_arrangement_count :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (grid : array 3 (array 3 ℕ)),
  (∀ i j, grid[i][j] ∈ nums) ∧
  ∀ r, (grid[r].sum = 15 ∧ 
        (grid[0][r] + grid[1][r] + grid[2][r]) = 15) →
  (let arrangements := { a | isValidArrangement a } in arrangements.count = 72) :=
sorry

end grid_arrangement_count_l19_19391


namespace MaryNeedingToGrow_l19_19934

/-- Mary's height is 2/3 of her brother's height. --/
def MarysHeight (brothersHeight : ℕ) : ℕ := (2 * brothersHeight) / 3

/-- Mary needs to grow a certain number of centimeters to meet the minimum height
    requirement for riding Kingda Ka. --/
def RequiredGrowth (minimumHeight maryHeight : ℕ) : ℕ := minimumHeight - maryHeight

theorem MaryNeedingToGrow 
  (minimumHeight : ℕ := 140)
  (brothersHeight : ℕ := 180)
  (brothersHeightIs180 : brothersHeight = 180 := rfl)
  (heightRatio : ℕ → ℕ := MarysHeight)
  (maryHeight : ℕ := heightRatio brothersHeight)
  (maryHeightProof : maryHeight = 120 := by simp [MarysHeight, brothersHeightIs180])
  (requiredGrowth : ℕ := RequiredGrowth minimumHeight maryHeight) :
  requiredGrowth = 20 :=
by
  unfold RequiredGrowth MarysHeight
  rw [maryHeightProof]
  exact rfl

end MaryNeedingToGrow_l19_19934


namespace proposition_p_proposition_q_l19_19895

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

-- Prove the propositions p and q
theorem proposition_p : ∀ x : ℝ, f x > 0 :=
sorry

theorem proposition_q : ∃ x : ℝ, 0 < x ∧ g x = 0 :=
sorry

end proposition_p_proposition_q_l19_19895


namespace largest_multiple_of_15_less_than_500_l19_19111

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19111


namespace price_percentage_combined_assets_l19_19994

variable (A B P : ℝ)

-- Conditions
axiom h1 : P = 1.20 * A
axiom h2 : P = 2 * B

-- Statement
theorem price_percentage_combined_assets : (P / (A + B)) * 100 = 75 := by
  sorry

end price_percentage_combined_assets_l19_19994


namespace largest_multiple_of_15_less_than_500_l19_19195

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19195


namespace odd_square_minus_one_divisible_by_eight_l19_19783

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) : ∃ k : ℤ, ((2 * n + 1) ^ 2 - 1) = 8 * k := 
by
  sorry

end odd_square_minus_one_divisible_by_eight_l19_19783


namespace largest_multiple_of_15_less_than_500_l19_19116

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19116


namespace bathroom_area_l19_19518

def tile_size : ℝ := 0.5 -- Each tile is 0.5 feet

structure Section :=
  (width : ℕ)
  (length : ℕ)

def longer_section : Section := ⟨15, 25⟩
def alcove : Section := ⟨10, 8⟩

def area (s : Section) : ℝ := (s.width * tile_size) * (s.length * tile_size)

theorem bathroom_area :
  area longer_section + area alcove = 113.75 := by
  sorry

end bathroom_area_l19_19518


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l19_19053

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0) 
  (h2 : a 1 + a 4 + a 7 = 12) (h3 : a 1 * a 4 * a 7 = 28) :
  ∀ n, a n = n :=
sorry

theorem geometric_sequence_formula (b : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : b 1 = 16) (h2 : a 2 * b 2 = 4) :
  ∀ n, b n = 2^(n + 3) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = 2^(n + 3)) 
  (h3 : ∀ n, c n = a n * b n) :
  ∀ n, T n = 8 * (2^n * (n + 1) - 1) :=
sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l19_19053


namespace bakery_water_requirement_l19_19285

theorem bakery_water_requirement (flour water : ℕ) (total_flour : ℕ) (h : flour = 300) (w : water = 75) (t : total_flour = 900) : 
  225 = (total_flour / flour) * water :=
by
  sorry

end bakery_water_requirement_l19_19285


namespace ordered_pair_solution_l19_19394

theorem ordered_pair_solution :
  ∃ (x y : ℤ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ (x, y) = (2, 4) :=
by
  sorry

end ordered_pair_solution_l19_19394


namespace largest_multiple_of_15_below_500_l19_19175

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19175


namespace prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l19_19096

noncomputable def probability_A_exactly_2_hits :=
  let p_A := 1/2
  let trials := 3
  (trials.choose 2) * (p_A ^ 2) * ((1 - p_A) ^ (trials - 2))

noncomputable def probability_B_at_least_2_hits :=
  let p_B := 2/3
  let trials := 3
  (trials.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ (trials - 2)) + (trials.choose 3) * (p_B ^ 3)

noncomputable def probability_B_exactly_2_more_hits_A :=
  let p_A := 1/2
  let p_B := 2/3
  let trials := 3
  let B_2_A_0 := (trials.choose 2) * (p_B ^ 2) * (1 - p_B) * (trials.choose 0) * (p_A ^ 0) * ((1 - p_A) ^ trials)
  let B_3_A_1 := (trials.choose 3) * (p_B ^ 3) * (trials.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (trials - 1))
  B_2_A_0 + B_3_A_1

theorem prove_A_exactly_2_hits : probability_A_exactly_2_hits = 3/8 := sorry
theorem prove_B_at_least_2_hits : probability_B_at_least_2_hits = 20/27 := sorry
theorem prove_B_exactly_2_more_hits_A : probability_B_exactly_2_more_hits_A = 1/6 := sorry

end prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l19_19096


namespace inverse_of_5_mod_35_l19_19010

theorem inverse_of_5_mod_35 : (5 * 28) % 35 = 1 :=
by
  sorry

end inverse_of_5_mod_35_l19_19010


namespace largest_multiple_of_15_under_500_l19_19207

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19207


namespace b1_value_l19_19666

axiom seq_b (b : ℕ → ℝ) : ∀ n, b 50 = 2 → 
  (∀ n ≥ 2, (∑ i in finset.range n, b (i + 1)) = n^3 * b n) → b 1 = 100

theorem b1_value (b : ℕ → ℝ) (h50 : b 50 = 2)
  (h : ∀ n ≥ 2, (∑ i in finset.range n, b (i + 1)) = n^3 * b n) : b 1 = 100 :=
sorry

end b1_value_l19_19666


namespace trishul_invested_percentage_less_than_raghu_l19_19844

variable {T V R : ℝ}

def vishal_invested_more (T V : ℝ) : Prop :=
  V = 1.10 * T

def total_sum_of_investments (T V : ℝ) : Prop :=
  T + V + 2300 = 6647

def raghu_investment : ℝ := 2300

theorem trishul_invested_percentage_less_than_raghu
  (h1 : vishal_invested_more T V)
  (h2 : total_sum_of_investments T V) :
  ((raghu_investment - T) / raghu_investment) * 100 = 10 :=
  sorry

end trishul_invested_percentage_less_than_raghu_l19_19844


namespace find_a10_l19_19597

def arith_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variables (a : ℕ → ℚ) (d : ℚ)

-- Conditions
def condition1 := a 4 + a 11 = 16  -- translates to a_5 + a_12 = 16
def condition2 := a 6 = 1  -- translates to a_7 = 1
def condition3 := arith_seq a d  -- a is an arithmetic sequence with common difference d

-- The main theorem
theorem find_a10 : condition1 a ∧ condition2 a ∧ condition3 a d → a 9 = 15 := sorry

end find_a10_l19_19597


namespace total_pages_read_l19_19034

def pages_read_yesterday : ℕ := 21
def pages_read_today : ℕ := 17

theorem total_pages_read : pages_read_yesterday + pages_read_today = 38 :=
by
  sorry

end total_pages_read_l19_19034


namespace sum_of_three_numbers_l19_19832

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l19_19832


namespace election_majority_l19_19682

theorem election_majority (V : ℝ) 
  (h1 : ∃ w l : ℝ, w = 0.70 * V ∧ l = 0.30 * V ∧ w - l = 174) : 
  V = 435 :=
by
  sorry

end election_majority_l19_19682


namespace largest_multiple_15_under_500_l19_19121

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19121


namespace find_geometric_sequence_term_l19_19921

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l19_19921


namespace geometric_sequence_k_value_l19_19437

theorem geometric_sequence_k_value :
  ∀ {S : ℕ → ℤ} (a : ℕ → ℤ) (k : ℤ),
    (∀ n, S n = 3 * 2^n + k) → 
    (∀ n ≥ 2, a n = S n - S (n - 1)) → 
    (∀ n ≥ 2, a n ^ 2 = a 1 * a 3) → 
    k = -3 :=
by
  sorry

end geometric_sequence_k_value_l19_19437


namespace speed_of_B_l19_19859

theorem speed_of_B 
    (initial_distance : ℕ)
    (speed_of_A : ℕ)
    (time : ℕ)
    (distance_covered_by_A : ℕ)
    (distance_covered_by_B : ℕ)
    : initial_distance = 24 → speed_of_A = 5 → time = 2 → distance_covered_by_A = speed_of_A * time → distance_covered_by_B = initial_distance - distance_covered_by_A → distance_covered_by_B / time = 7 :=
by
  sorry

end speed_of_B_l19_19859


namespace kids_on_excursions_l19_19490

theorem kids_on_excursions (total_kids : ℕ) (one_fourth_kids_tubing two := total_kids / 4) (half_tubers_rafting : ℕ := one_fourth_kids_tubing / 2) :
  total_kids = 40 → one_fourth_kids_tubing = 10 → half_tubers_rafting = 5 :=
by
  intros
  sorry

end kids_on_excursions_l19_19490


namespace box_volume_correct_l19_19629

-- Define the dimensions of the obelisk
def obelisk_height : ℕ := 15
def base_length : ℕ := 8
def base_width : ℕ := 10

-- Define the dimension and volume goal for the cube-shaped box
def box_side_length : ℕ := obelisk_height
def box_volume : ℕ := box_side_length ^ 3

-- The proof goal
theorem box_volume_correct : box_volume = 3375 := 
by sorry

end box_volume_correct_l19_19629


namespace find_A_salary_l19_19814

theorem find_A_salary (A B : ℝ) (h1 : A + B = 2000) (h2 : 0.05 * A = 0.15 * B) : A = 1500 :=
sorry

end find_A_salary_l19_19814


namespace profit_share_of_B_l19_19873

-- Defining the initial investments
def a : ℕ := 8000
def b : ℕ := 10000
def c : ℕ := 12000

-- Given difference between profit shares of A and C
def diff_AC : ℕ := 680

-- Define total profit P
noncomputable def P : ℕ := (diff_AC * 15) / 2

-- Calculate B's profit share
noncomputable def B_share : ℕ := (5 * P) / 15

-- The theorem stating B's profit share
theorem profit_share_of_B : B_share = 1700 :=
by sorry

end profit_share_of_B_l19_19873


namespace find_pairs_l19_19717

theorem find_pairs (x y : ℕ) (h : x > 0 ∧ y > 0) (d : ℕ) (gcd_cond : d = Nat.gcd x y)
  (eqn_cond : x * y * d = x + y + d ^ 2) : (x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2) :=
by {
  sorry
}

end find_pairs_l19_19717


namespace probability_no_prize_l19_19516

theorem probability_no_prize : (1 : ℚ) - (1 : ℚ) / (50 * 50) = 2499 / 2500 :=
by
  sorry

end probability_no_prize_l19_19516


namespace find_second_divisor_l19_19521

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l19_19521


namespace doughnuts_per_person_l19_19460

theorem doughnuts_per_person :
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  total_doughnuts / total_people = 6 := 
by
  -- Definitions and conditions from the problem
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  -- Goal to prove
  show total_doughnuts / total_people = 6
  sorry

end doughnuts_per_person_l19_19460


namespace fraction_of_married_men_l19_19708

/-- At a social gathering, there are only single women and married men with their wives.
     The probability that a randomly selected woman is single is 3/7.
     The fraction of the people in the gathering that are married men is 4/11. -/
theorem fraction_of_married_men (women : ℕ) (single_women : ℕ) (married_men : ℕ) (total_people : ℕ) 
  (h_women_total : women = 7)
  (h_single_women_probability : single_women = women * 3 / 7)
  (h_married_women : women - single_women = married_men)
  (h_total_people : total_people = women + married_men) :
  married_men / total_people = 4 / 11 := 
by sorry

end fraction_of_married_men_l19_19708


namespace greenfield_academy_math_count_l19_19707

theorem greenfield_academy_math_count (total_players taking_physics both_subjects : ℕ) 
(h_total: total_players = 30) 
(h_physics: taking_physics = 15) 
(h_both: both_subjects = 3) : 
∃ taking_math : ℕ, taking_math = 21 :=
by
  sorry

end greenfield_academy_math_count_l19_19707


namespace skylar_total_donations_l19_19463

-- Define the conditions
def start_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the statement to be proven
theorem skylar_total_donations : 
  (current_age - start_age) * annual_donation = 432000 := by
    sorry

end skylar_total_donations_l19_19463


namespace product_of_two_digit_numbers_5488_has_smaller_number_56_l19_19665

theorem product_of_two_digit_numbers_5488_has_smaller_number_56 (a b : ℕ) (h_a2 : 10 ≤ a) (h_a3 : a < 100) (h_b2 : 10 ≤ b) (h_b3 : b < 100) (h_prod : a * b = 5488) : a = 56 ∨ b = 56 :=
by {
  sorry
}

end product_of_two_digit_numbers_5488_has_smaller_number_56_l19_19665


namespace largest_multiple_15_under_500_l19_19117

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19117


namespace mike_net_spending_l19_19776

-- Definitions for given conditions
def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84

-- Theorem stating the result
theorem mike_net_spending : trumpet_cost - song_book_revenue = 139.32 :=
by 
  sorry

end mike_net_spending_l19_19776


namespace gcd_12012_18018_l19_19570

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l19_19570


namespace equivar_proof_l19_19027

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l19_19027


namespace expected_value_of_twelve_sided_die_l19_19360

theorem expected_value_of_twelve_sided_die : ∃ E : ℝ, E = 6.5 :=
by
  let n := 12
  let sum_faces := n * (n + 1) / 2
  let E := sum_faces / n
  use E
  -- The sum of the first 12 natural numbers should be calculated: 1 + 2 + ... + 12 = 78
  have sum_calculated : sum_faces = 78 := by compute
  rw [sum_calculated]
  -- The expected value is hence 78 / 12
  have E_calculated : E = 78 / 12 := by compute
  rw [E_calculated]
  -- Finally, 78 / 12 simplifies to 6.5
  have E_final : 78 / 12 = 6.5 := by compute
  rw [E_final]

-- sorry is added for place holder to validate the step is proof-skipped.

end expected_value_of_twelve_sided_die_l19_19360


namespace largest_multiple_15_under_500_l19_19124

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19124


namespace problem_statement_l19_19595

-- Given conditions
noncomputable def S : ℕ → ℝ := sorry
axiom S_3_eq_2 : S 3 = 2
axiom S_6_eq_6 : S 6 = 6

-- Prove that a_{13} + a_{14} + a_{15} = 32
theorem problem_statement : (S 15 - S 12) = 32 :=
by sorry

end problem_statement_l19_19595


namespace bouquet_branches_l19_19758

variable (w : ℕ) (b : ℕ)

theorem bouquet_branches :
  (w + b = 7) → 
  (w ≥ 1) → 
  (∀ x y, x ≠ y → (x = w ∨ y = w) → (x = b ∨ y = b)) → 
  (w = 1 ∧ b = 6) :=
by
  intro h1 h2 h3
  sorry

end bouquet_branches_l19_19758


namespace lisa_likes_only_last_digit_zero_l19_19001

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8

def is_divisible_by_5_and_2 (n : ℕ) : Prop :=
  is_divisible_by_5 n ∧ is_divisible_by_2 n

theorem lisa_likes_only_last_digit_zero : ∀ n, is_divisible_by_5_and_2 n → n % 10 = 0 :=
by
  sorry

end lisa_likes_only_last_digit_zero_l19_19001


namespace find_x_angle_l19_19507

-- Define the conditions
def angles_around_point (a b c d : ℝ) : Prop :=
  a + b + c + d = 360

-- The given problem implies:
-- 120 + x + x + 2x = 360
-- We need to find x such that the above equation holds.
theorem find_x_angle :
  angles_around_point 120 x x (2 * x) → x = 60 :=
by
  sorry

end find_x_angle_l19_19507


namespace arithmetic_sequence_n_value_l19_19616

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_n_value :
  ∀ (a1 d n an : ℕ), a1 = 3 → d = 2 → an = 25 → arithmetic_seq_nth_term a1 d n = an → n = 12 :=
by
  intros a1 d n an ha1 hd han h
  sorry

end arithmetic_sequence_n_value_l19_19616


namespace probability_of_heads_at_least_once_l19_19843

theorem probability_of_heads_at_least_once 
  (X : ℕ → ℝ)
  (hX_binom : ∀ n, X n = binomial (n := 3) (p := 0.5) n)
  (indep_tosses : ∀ i j, i ≠ j → indep_fun X (X j))
  (prob_heads : ∀ n, X n = 1/2) :
  (Pr (X ≥ 1) = 7/8) := 
by 
  sorry

end probability_of_heads_at_least_once_l19_19843


namespace smallest_possible_n_l19_19653

theorem smallest_possible_n :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2010 ∧
  (∃ (m n : ℤ), (x! * y! * z! = m * 10^n) ∧ (m % 10 ≠ 0) ∧ n = 492) :=
by
  sorry

end smallest_possible_n_l19_19653


namespace total_oranges_l19_19094

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l19_19094


namespace largest_multiple_of_15_under_500_l19_19210

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19210


namespace insurance_not_covered_percentage_l19_19669

noncomputable def insurance_monthly_cost : ℝ := 20
noncomputable def insurance_months : ℝ := 24
noncomputable def procedure_cost : ℝ := 5000
noncomputable def amount_saved : ℝ := 3520

theorem insurance_not_covered_percentage :
  ((procedure_cost - amount_saved - (insurance_monthly_cost * insurance_months)) / procedure_cost) * 100 = 20 :=
by
  sorry

end insurance_not_covered_percentage_l19_19669


namespace boy_usual_time_to_school_l19_19850

theorem boy_usual_time_to_school
  (S : ℝ) -- Usual speed
  (T : ℝ) -- Usual time
  (D : ℝ) -- Distance, D = S * T
  (hD : D = S * T)
  (h1 : 3/4 * D / (7/6 * S) + 1/4 * D / (5/6 * S) = T - 2) : 
  T = 35 :=
by
  sorry

end boy_usual_time_to_school_l19_19850


namespace fred_balloon_count_l19_19402

variable (Fred_balloons Sam_balloons Mary_balloons total_balloons : ℕ)

/-- 
  Given:
  - Fred has some yellow balloons
  - Sam has 6 yellow balloons
  - Mary has 7 yellow balloons
  - Total number of yellow balloons (Fred's, Sam's, and Mary's balloons) is 18

  Prove: Fred has 5 yellow balloons.
-/
theorem fred_balloon_count :
  Sam_balloons = 6 →
  Mary_balloons = 7 →
  total_balloons = 18 →
  Fred_balloons = total_balloons - (Sam_balloons + Mary_balloons) →
  Fred_balloons = 5 :=
by
  sorry

end fred_balloon_count_l19_19402


namespace cover_square_floor_l19_19325

theorem cover_square_floor (x : ℕ) (h : 2 * x - 1 = 37) : x^2 = 361 :=
by
  sorry

end cover_square_floor_l19_19325


namespace min_elements_in_S_l19_19060

noncomputable def exists_function_with_property (S : Type) [fintype S]
  (f : ℕ → S) : Prop :=
∀ (x y : ℕ), nat.prime (abs (x - y)) → f x ≠ f y

theorem min_elements_in_S (S : Type) [fintype S]
  (h : ∃ f : ℕ → S, exists_function_with_property S f) : 
  fintype.card S ≥ 4 :=
sorry

end min_elements_in_S_l19_19060


namespace unique_solution_l19_19307

def is_valid_func (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 2001 ∨ f (f n) + f n = 2 * n + 2002

theorem unique_solution (f : ℕ → ℕ) (hf : is_valid_func f) :
  ∀ n, f n = n + 667 :=
sorry

end unique_solution_l19_19307


namespace large_pretzel_cost_l19_19471

theorem large_pretzel_cost : 
  ∀ (P S : ℕ), 
  P = 3 * S ∧ 7 * P + 4 * S = 4 * P + 7 * S + 12 → 
  P = 6 :=
by sorry

end large_pretzel_cost_l19_19471


namespace possible_double_roots_l19_19344

theorem possible_double_roots (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  s^2 ∣ 50 →
  (Polynomial.eval s (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4) = 0) →
  (Polynomial.eval s (Polynomial.derivative (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4)) = 0) →
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 :=
by
  sorry

end possible_double_roots_l19_19344


namespace minimal_connections_correct_l19_19754

-- Define a Lean structure to encapsulate the conditions
structure IslandsProblem where
  islands : ℕ
  towns : ℕ
  min_towns_per_island : ℕ
  condition_islands : islands = 13
  condition_towns : towns = 25
  condition_min_towns : min_towns_per_island = 1

-- Define a function to represent the minimal number of ferry connections
def minimalFerryConnections (p : IslandsProblem) : ℕ :=
  222

-- Define the statement to be proved
theorem minimal_connections_correct (p : IslandsProblem) : 
  p.islands = 13 → 
  p.towns = 25 → 
  p.min_towns_per_island = 1 → 
  minimalFerryConnections p = 222 :=
by
  intros
  sorry

end minimal_connections_correct_l19_19754


namespace count_total_kids_in_lawrence_l19_19381

namespace LawrenceCountyKids

/-- Number of kids who went to camp from Lawrence county -/
def kids_went_to_camp : ℕ := 610769

/-- Number of kids who stayed home -/
def kids_stayed_home : ℕ := 590796

/-- Total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := 1201565

/-- Proof statement -/
theorem count_total_kids_in_lawrence :
  kids_went_to_camp + kids_stayed_home = total_kids_in_county :=
sorry

end LawrenceCountyKids

end count_total_kids_in_lawrence_l19_19381


namespace invalid_root_l19_19452

theorem invalid_root (a_1 a_0 : ℤ) : ¬(19 * (1/7 : ℚ)^3 + 98 * (1/7 : ℚ)^2 + a_1 * (1/7 : ℚ) + a_0 = 0) :=
by 
  sorry

end invalid_root_l19_19452


namespace largest_multiple_of_15_less_than_500_l19_19186

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19186


namespace geometric_seq_prod_l19_19431

-- Conditions: Geometric sequence and given value of a_1 * a_7 * a_13
variables {a : ℕ → ℝ}
variable (r : ℝ)

-- Definition of a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- The proof problem
theorem geometric_seq_prod (h_geo : geometric_sequence a r) (h_prod : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 :=
sorry

end geometric_seq_prod_l19_19431


namespace original_ghee_quantity_l19_19336

theorem original_ghee_quantity (x : ℝ) (H1 : 0.60 * x + 10 = ((1 + 0.40 * x) * 0.80)) :
  x = 10 :=
sorry

end original_ghee_quantity_l19_19336


namespace dance_relationship_l19_19543

theorem dance_relationship (b g : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ b → i = 1 → ∃ m, m = 7)
  (h2 : b = g - 6) 
  : 7 + (b - 1) = g := 
by
  sorry

end dance_relationship_l19_19543


namespace geometric_seq_prod_l19_19432

-- Conditions: Geometric sequence and given value of a_1 * a_7 * a_13
variables {a : ℕ → ℝ}
variable (r : ℝ)

-- Definition of a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- The proof problem
theorem geometric_seq_prod (h_geo : geometric_sequence a r) (h_prod : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 :=
sorry

end geometric_seq_prod_l19_19432


namespace add_two_integers_l19_19566

/-- If the difference of two positive integers is 5 and their product is 180,
then their sum is 25. -/
theorem add_two_integers {x y : ℕ} (h1: x > y) (h2: x - y = 5) (h3: x * y = 180) : x + y = 25 :=
sorry

end add_two_integers_l19_19566


namespace equivar_proof_l19_19026

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l19_19026


namespace find_angle_AOD_l19_19293

noncomputable def angleAOD (x : ℝ) : ℝ :=
4 * x

theorem find_angle_AOD (x : ℝ) (h1 : 4 * x = 180) : angleAOD x = 135 :=
by
  -- x = 45
  have h2 : x = 45 := by linarith

  -- angleAOD 45 = 4 * 45 = 135
  rw [angleAOD, h2]
  norm_num
  sorry

end find_angle_AOD_l19_19293


namespace find_line_through_intersection_and_perpendicular_l19_19569

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def perpendicular (x y m : ℝ) : Prop := x + 3 * y + 4 = 0 ∧ 3 * x - y + m = 0

theorem find_line_through_intersection_and_perpendicular :
  ∃ m : ℝ, ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ perpendicular x y m → 3 * x - y + 2 = 0 :=
by
  sorry

end find_line_through_intersection_and_perpendicular_l19_19569


namespace second_fish_length_l19_19865

-- Defining the conditions
def first_fish_length : ℝ := 0.3
def length_difference : ℝ := 0.1

-- Proof statement
theorem second_fish_length : ∀ (second_fish : ℝ), first_fish_length = second_fish + length_difference → second_fish = 0.2 :=
by 
  intro second_fish
  intro h
  sorry

end second_fish_length_l19_19865


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l19_19975

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l19_19975


namespace minimum_groups_l19_19348

theorem minimum_groups (total_students groupsize : ℕ) (h_students : total_students = 30)
    (h_groupsize : 1 ≤ groupsize ∧ groupsize ≤ 12) :
    ∃ k, k = total_students / groupsize ∧ total_students % groupsize = 0 ∧ k ≥ (total_students / 12) :=
by
  simp [h_students, h_groupsize]
  use 3
  sorry

end minimum_groups_l19_19348


namespace dot_product_a_b_l19_19022

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem dot_product_a_b : (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end dot_product_a_b_l19_19022


namespace right_triangle_area_l19_19808

theorem right_triangle_area {a r R : ℝ} (hR : R = (5 / 2) * r) (h_leg : ∃ BC, BC = a) :
  (∃ area, area = (2 * a^2 / 3) ∨ area = (3 * a^2 / 8)) :=
sorry

end right_triangle_area_l19_19808


namespace three_digit_number_cubed_sum_l19_19888

theorem three_digit_number_cubed_sum {n : ℕ} (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 100 * a + 10 * b + c ∧ n = a^3 + b^3 + c^3) ↔
  n = 153 ∨ n = 370 ∨ n = 371 ∨ n = 407 :=
by
  sorry

end three_digit_number_cubed_sum_l19_19888


namespace no_valid_C_for_2C4_multiple_of_5_l19_19398

theorem no_valid_C_for_2C4_multiple_of_5 :
  ¬ (∃ C : ℕ, C < 10 ∧ (2 * 100 + C * 10 + 4) % 5 = 0) :=
by
  sorry

end no_valid_C_for_2C4_multiple_of_5_l19_19398


namespace largest_multiple_of_15_under_500_l19_19213

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19213


namespace sum_of_three_integers_l19_19973

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l19_19973


namespace students_wearing_blue_lipstick_l19_19453

theorem students_wearing_blue_lipstick
  (total_students : ℕ)
  (half_students_wore_lipstick : total_students / 2 = 180)
  (red_fraction : ℚ)
  (pink_fraction : ℚ)
  (purple_fraction : ℚ)
  (green_fraction : ℚ)
  (students_wearing_red : red_fraction * 180 = 45)
  (students_wearing_pink : pink_fraction * 180 = 60)
  (students_wearing_purple : purple_fraction * 180 = 30)
  (students_wearing_green : green_fraction * 180 = 15)
  (total_red_fraction : red_fraction = 1 / 4)
  (total_pink_fraction : pink_fraction = 1 / 3)
  (total_purple_fraction : purple_fraction = 1 / 6)
  (total_green_fraction : green_fraction = 1 / 12) :
  (180 - (45 + 60 + 30 + 15) = 30) :=
by sorry

end students_wearing_blue_lipstick_l19_19453


namespace white_tiles_in_square_l19_19347

theorem white_tiles_in_square (n S : ℕ) (hn : n * n = S) (black_tiles : ℕ) (hblack_tiles : black_tiles = 81) (diagonal_black_tiles : n = 9) :
  S - black_tiles = 72 :=
by
  sorry

end white_tiles_in_square_l19_19347


namespace largest_multiple_of_15_less_than_500_l19_19206

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19206


namespace Linda_original_savings_l19_19305

theorem Linda_original_savings (S : ℝ)
  (H1 : 3/4 * S + 1/4 * S = S)
  (H2 : 1/4 * S = 220) :
  S = 880 :=
sorry

end Linda_original_savings_l19_19305


namespace maximizing_sum_of_arithmetic_sequence_l19_19230

theorem maximizing_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_decreasing : ∀ n, a n > a (n + 1))
  (h_sum : S 5 = S 10) :
  (S 7 >= S n ∧ S 8 >= S n) := sorry

end maximizing_sum_of_arithmetic_sequence_l19_19230


namespace fraction_to_decimal_l19_19009

theorem fraction_to_decimal : (53 : ℚ) / (4 * 5^7) = 1325 / 10^7 := sorry

end fraction_to_decimal_l19_19009


namespace incorrect_option_B_l19_19539

-- Definitions of the given conditions
def optionA (a : ℝ) : Prop := (8 * a = 8 * a)
def optionB (a : ℝ) : Prop := (a - (0.08 * a) = 8 * a)
def optionC (a : ℝ) : Prop := (8 * a = 8 * a)
def optionD (a : ℝ) : Prop := (a * 8 = 8 * a)

-- The statement to be proved
theorem incorrect_option_B (a : ℝ) : 
  optionA a ∧ ¬optionB a ∧ optionC a ∧ optionD a := 
by
  sorry

end incorrect_option_B_l19_19539


namespace quadratic_eq_solutions_l19_19959

theorem quadratic_eq_solutions : ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ x = 3 / 2 ∨ x = 1 :=
by
  sorry

end quadratic_eq_solutions_l19_19959


namespace math_problem_A_B_M_l19_19273

theorem math_problem_A_B_M :
  ∃ M : Set ℝ,
    M = {m | ∃ A B : Set ℝ,
      A = {x | x^2 - 5 * x + 6 = 0} ∧
      B = {x | m * x - 1 = 0} ∧
      A ∩ B = B ∧
      M = {0, (1:ℝ)/2, (1:ℝ)/3}} ∧
    ∃ subsets : Set (Set ℝ),
      subsets = {∅, {0}, {(1:ℝ)/2}, {(1:ℝ)/3}, {0, (1:ℝ)/2}, {(1:ℝ)/2, (1:ℝ)/3}, {0, (1:ℝ)/3}, {0, (1:ℝ)/2, (1:ℝ)/3}} :=
by
  sorry

end math_problem_A_B_M_l19_19273


namespace circle_radius_l19_19635

theorem circle_radius : 
  ∀ (x y : ℝ), x^2 + y^2 + 12 = 10 * x - 6 * y → ∃ r : ℝ, r = Real.sqrt 22 :=
by
  intros x y h
  -- Additional steps to complete the proof will be added here
  sorry

end circle_radius_l19_19635


namespace find_a_b_l19_19728

theorem find_a_b :
  ∃ a b : ℝ, 
    (a = -4) ∧ (b = -9) ∧
    (∀ x : ℝ, |8 * x + 9| < 7 ↔ a * x^2 + b * x - 2 > 0) := 
sorry

end find_a_b_l19_19728


namespace problem_f_2010_l19_19030

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1 / 4
axiom f_eq : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2010 : f 2010 = 1 / 2 :=
sorry

end problem_f_2010_l19_19030


namespace sqrt_720_eq_12_sqrt_5_l19_19643

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l19_19643


namespace largest_multiple_of_15_less_than_500_l19_19105

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19105


namespace at_least_one_did_not_land_stably_l19_19883

-- Define the propositions p and q
variables (p q : Prop)

-- Define the theorem to prove
theorem at_least_one_did_not_land_stably :
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by
  sorry

end at_least_one_did_not_land_stably_l19_19883


namespace conversion_base8_to_base10_l19_19555

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l19_19555


namespace unique_double_digit_in_range_l19_19055

theorem unique_double_digit_in_range (a b : ℕ) (h₁ : a = 10) (h₂ : b = 40) : 
  ∃! n : ℕ, (10 ≤ n ∧ n ≤ 40) ∧ (n % 10 = n / 10) ∧ (n % 10 = 3) :=
by {
  sorry
}

end unique_double_digit_in_range_l19_19055


namespace cost_per_day_additional_weeks_l19_19476

theorem cost_per_day_additional_weeks :
  let first_week_days := 7
  let first_week_cost_per_day := 18.00
  let first_week_cost := first_week_days * first_week_cost_per_day
  let total_days := 23
  let total_cost := 302.00
  let additional_days := total_days - first_week_days
  let additional_cost := total_cost - first_week_cost
  let cost_per_day_additional := additional_cost / additional_days
  cost_per_day_additional = 11.00 :=
by
  sorry

end cost_per_day_additional_weeks_l19_19476


namespace baseball_opponents_score_l19_19226

theorem baseball_opponents_score 
  (team_scores : List ℕ)
  (team_lost_scores : List ℕ)
  (team_won_scores : List ℕ)
  (opponent_lost_scores : List ℕ)
  (opponent_won_scores : List ℕ)
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : team_lost_scores = [1, 3, 5, 7, 9, 11])
  (h3 : team_won_scores = [6, 9, 12])
  (h4 : opponent_lost_scores = [3, 5, 7, 9, 11, 13])
  (h5 : opponent_won_scores = [2, 3, 4]) :
  (List.sum opponent_lost_scores + List.sum opponent_won_scores = 57) :=
sorry

end baseball_opponents_score_l19_19226


namespace largest_multiple_of_15_less_than_500_l19_19107

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19107


namespace polynomial_solution_l19_19246

theorem polynomial_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, 1 + P.eval x = (1 / 2) * (P.eval (x - 1) + P.eval (x + 1))) →
  ∃ b c : ℝ, ∀ x : ℝ, P.eval x = x^2 + b * x + c := 
sorry

end polynomial_solution_l19_19246


namespace distribute_books_l19_19720

theorem distribute_books (m n : ℕ) (h1 : m = 3*n + 8) (h2 : ∃k, m = 5*k + r ∧ r < 5 ∧ r > 0) : 
  n = 5 ∨ n = 6 :=
by sorry

end distribute_books_l19_19720


namespace cos_sin_value_l19_19740

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end cos_sin_value_l19_19740


namespace pyramid_rhombus_side_length_l19_19802

theorem pyramid_rhombus_side_length
  (α β S: ℝ) (hα : 0 < α) (hβ : 0 < β) (hS : 0 < S) :
  ∃ a : ℝ, a = 2 * Real.sqrt (2 * S * Real.cos β / Real.sin α) :=
by
  sorry

end pyramid_rhombus_side_length_l19_19802


namespace largest_multiple_of_15_less_than_500_l19_19114

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19114


namespace find_last_number_l19_19475

theorem find_last_number (A B C D : ℕ) 
  (h1 : A + B + C = 18) 
  (h2 : B + C + D = 9) 
  (h3 : A + D = 13) 
  : D = 2 := by 
  sorry

end find_last_number_l19_19475


namespace triangle_angle_condition_l19_19638

theorem triangle_angle_condition (a b h_3 : ℝ) (A C : ℝ) 
  (h : 1/(h_3^2) = 1/(a^2) + 1/(b^2)) :
  C = 90 ∨ |A - C| = 90 := 
sorry

end triangle_angle_condition_l19_19638


namespace sqrt_720_simplified_l19_19650

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l19_19650


namespace twelve_sided_die_expected_value_l19_19370

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end twelve_sided_die_expected_value_l19_19370


namespace red_higher_than_green_l19_19698

open ProbabilityTheory

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  (2:ℝ)^(-k)

noncomputable def prob_red_higher_than_green : ℝ :=
  ∑' (k : ℕ), (prob_bin_k k) * (prob_bin_k (k + 1))

theorem red_higher_than_green :
  (∑' (k : ℕ), (2:ℝ) ^ (-k) * (2:ℝ) ^(-(k + 1))) = 1/3 :=
  by
  sorry

end red_higher_than_green_l19_19698


namespace range_of_m_if_not_p_and_q_l19_19024

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l19_19024


namespace quadratic_zeros_l19_19043

theorem quadratic_zeros (a b : ℝ) (h1 : (4 - 2 * a + b = 0)) (h2 : (9 + 3 * a + b = 0)) : a + b = -7 := 
by
  sorry

end quadratic_zeros_l19_19043


namespace compare_x_y_l19_19747

theorem compare_x_y :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := sorry

end compare_x_y_l19_19747


namespace find_geometric_sequence_term_l19_19920

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l19_19920


namespace mowing_field_time_l19_19510

theorem mowing_field_time (h1 : (1 / 28 : ℝ) = (3 / 84 : ℝ))
                         (h2 : (1 / 84 : ℝ) = (1 / 84 : ℝ))
                         (h3 : (1 / 28 + 1 / 84 : ℝ) = (1 / 21 : ℝ)) :
                         21 = 1 / ((1 / 28) + (1 / 84)) := 
by {
  sorry
}

end mowing_field_time_l19_19510


namespace find_weights_l19_19980

theorem find_weights (x y z : ℕ) (h1 : x + y + z = 11) (h2 : 3 * x + 7 * y + 14 * z = 108) :
  x = 1 ∧ y = 5 ∧ z = 5 :=
by
  sorry

end find_weights_l19_19980


namespace intersection_of_A_and_CU_B_l19_19417

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {-1, 0, 1, 2, 3}
noncomputable def B : Set ℝ := { x : ℝ | x ≥ 2 }
noncomputable def CU_B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_of_A_and_CU_B :
  A ∩ CU_B = {-1, 0, 1} :=
by
  sorry

end intersection_of_A_and_CU_B_l19_19417


namespace volume_rect_prism_l19_19346

variables (a d h : ℝ)
variables (ha : a > 0) (hd : d > 0) (hh : h > 0)

theorem volume_rect_prism : a * d * h = adh :=
by
  sorry

end volume_rect_prism_l19_19346


namespace largest_multiple_of_15_less_than_500_l19_19150

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19150


namespace find_m_l19_19599

theorem find_m (x y m : ℤ) (h1 : 3 * x + 4 * y = 7) (h2 : 5 * x - 4 * y = m) (h3 : x + y = 0) : m = -63 := by
  sorry

end find_m_l19_19599


namespace find_product_in_geometric_sequence_l19_19430

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

theorem find_product_in_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 7 * a 13 = 8) : 
  a 3 * a 11 = 4 :=
by sorry

end find_product_in_geometric_sequence_l19_19430


namespace Xiaohuo_books_l19_19680

def books_proof_problem : Prop :=
  ∃ (X_H X_Y X_Z : ℕ), 
    (X_H + X_Y + X_Z = 1248) ∧ 
    (X_H = X_Y + 64) ∧ 
    (X_Y = X_Z - 32) ∧ 
    (X_H = 448)

theorem Xiaohuo_books : books_proof_problem :=
by
  sorry

end Xiaohuo_books_l19_19680


namespace interest_rate_calculation_l19_19233

theorem interest_rate_calculation
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ)
  (h1 : SI = 2100) (h2 : P = 875) (h3 : T = 20) :
  (SI * 100 = P * R * T) → R = 12 :=
by
  sorry

end interest_rate_calculation_l19_19233


namespace solution_set_of_inequality_l19_19815

theorem solution_set_of_inequality :
  { x : ℝ // (x - 2)^2 ≤ 2 * x + 11 } = { x : ℝ | -1 ≤ x ∧ x ≤ 7 } :=
sorry

end solution_set_of_inequality_l19_19815


namespace largest_multiple_of_15_less_than_500_l19_19198

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19198


namespace octagon_properties_l19_19234

-- Definitions for a regular octagon inscribed in a circle
def regular_octagon (r : ℝ) := ∀ (a b : ℝ), abs (a - b) = r
def side_length := 5
def inscribed_in_circle (r : ℝ) := ∃ (a b : ℝ), a * a + b * b = r * r

-- Main theorem statement
theorem octagon_properties (r : ℝ) (h : r = side_length) (h1 : regular_octagon r) (h2 : inscribed_in_circle r) :
  let arc_length := (5 * π) / 4
  let area_sector := (25 * π) / 8
  arc_length = (5 * π) / 4 ∧ area_sector = (25 * π) / 8 := by
  sorry

end octagon_properties_l19_19234


namespace largest_multiple_of_15_under_500_l19_19209

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19209


namespace no_real_roots_quadratic_l19_19564

theorem no_real_roots_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 8) :
    (a ≠ 0) → (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :=
by
  sorry

end no_real_roots_quadratic_l19_19564


namespace sam_cleaner_meetings_two_times_l19_19075

open Nat

noncomputable def sam_and_cleaner_meetings (sam_rate cleaner_rate cleaner_stop_time bench_distance : ℕ) : ℕ :=
  let cycle_time := (bench_distance / cleaner_rate) + cleaner_stop_time
  let distance_covered_in_cycle_sam := sam_rate * cycle_time
  let distance_covered_in_cycle_cleaner := bench_distance
  let effective_distance_reduction := distance_covered_in_cycle_cleaner - distance_covered_in_cycle_sam
  let number_of_cycles_until_meeting := bench_distance / effective_distance_reduction
  number_of_cycles_until_meeting + 1

theorem sam_cleaner_meetings_two_times :
  sam_and_cleaner_meetings 3 9 40 300 = 2 :=
by sorry

end sam_cleaner_meetings_two_times_l19_19075


namespace ratio_city_XY_l19_19878

variable (popZ popY popX : ℕ)

-- Definition of the conditions
def condition1 := popY = 2 * popZ
def condition2 := popX = 16 * popZ

-- The goal to prove
theorem ratio_city_XY 
  (h1 : condition1 popY popZ)
  (h2 : condition2 popX popZ) :
  popX / popY = 8 := 
  by sorry

end ratio_city_XY_l19_19878


namespace arithmetic_sequence_seventh_term_l19_19087

theorem arithmetic_sequence_seventh_term
  (a d : ℝ)
  (h_sum : 4 * a + 6 * d = 20)
  (h_fifth : a + 4 * d = 8) :
  a + 6 * d = 10.4 :=
by
  sorry -- proof to be provided

end arithmetic_sequence_seventh_term_l19_19087


namespace convert_base_8_to_10_l19_19558

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l19_19558


namespace border_collie_catches_ball_in_32_seconds_l19_19544

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l19_19544


namespace no_valid_C_for_2C4_multiple_of_5_l19_19399

theorem no_valid_C_for_2C4_multiple_of_5 :
  ¬ (∃ C : ℕ, C < 10 ∧ (2 * 100 + C * 10 + 4) % 5 = 0) :=
by
  sorry

end no_valid_C_for_2C4_multiple_of_5_l19_19399


namespace conversion_correct_l19_19560

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l19_19560


namespace sum_of_distinct_FGHJ_values_l19_19480

theorem sum_of_distinct_FGHJ_values (A B C D E F G H I J K : ℕ)
  (h1: 0 ≤ A ∧ A ≤ 9)
  (h2: 0 ≤ B ∧ B ≤ 9)
  (h3: 0 ≤ C ∧ C ≤ 9)
  (h4: 0 ≤ D ∧ D ≤ 9)
  (h5: 0 ≤ E ∧ E ≤ 9)
  (h6: 0 ≤ F ∧ F ≤ 9)
  (h7: 0 ≤ G ∧ G ≤ 9)
  (h8: 0 ≤ H ∧ H ≤ 9)
  (h9: 0 ≤ I ∧ I ≤ 9)
  (h10: 0 ≤ J ∧ J ≤ 9)
  (h11: 0 ≤ K ∧ K ≤ 9)
  (h_divisibility_16: ∃ x, GHJK = x ∧ x % 16 = 0)
  (h_divisibility_9: (1 + B + C + D + E + F + G + H + I + J + K) % 9 = 0) :
  (F * G * H * J = 12 ∨ F * G * H * J = 120 ∨ F * G * H * J = 448) →
  (12 + 120 + 448 = 580) := 
by sorry

end sum_of_distinct_FGHJ_values_l19_19480


namespace sum_faces_of_cube_l19_19076

-- Conditions in Lean 4
variables (a b c d e f : ℕ)

-- Sum of vertex labels
def vertex_sum := a * b * c + a * e * c + a * b * f + a * e * f +
                  d * b * c + d * e * c + d * b * f + d * e * f

-- Theorem statement
theorem sum_faces_of_cube (h : vertex_sum a b c d e f = 1001) :
  (a + d) + (b + e) + (c + f) = 31 :=
sorry

end sum_faces_of_cube_l19_19076


namespace graduating_class_total_students_l19_19342

theorem graduating_class_total_students (boys girls students : ℕ) (h1 : girls = boys + 69) (h2 : boys = 208) :
  students = boys + girls → students = 485 :=
by
  sorry

end graduating_class_total_students_l19_19342


namespace asian_games_tourists_scientific_notation_l19_19315

theorem asian_games_tourists_scientific_notation : 
  ∀ (n : ℕ), n = 18480000 → 1.848 * (10:ℝ) ^ 7 = (n : ℝ) :=
by
  intro n
  sorry

end asian_games_tourists_scientific_notation_l19_19315


namespace locus_of_tangency_centers_l19_19320

def locus_of_centers (a b : ℝ) : Prop := 8 * a ^ 2 + 9 * b ^ 2 - 16 * a - 64 = 0

theorem locus_of_tangency_centers (a b : ℝ)
  (hx1 : ∃ x y : ℝ, x ^ 2 + y ^ 2 = 1) 
  (hx2 : ∃ x y : ℝ, (x - 2) ^ 2 + y ^ 2 = 25) 
  (hcent : ∃ r : ℝ, a^2 + b^2 = (r + 1)^2 ∧ (a - 2)^2 + b^2 = (5 - r)^2) : 
  locus_of_centers a b :=
sorry

end locus_of_tangency_centers_l19_19320


namespace find_angle_E_l19_19057

-- Defining the trapezoid properties and angles
variables (EF GH : ℝ) (E H G F : ℝ)
variables (trapezoid_EFGH : Prop) (parallel_EF_GH : Prop)
variables (angle_E_eq_3H : Prop) (angle_G_eq_4F : Prop)

-- Conditions
def trapezoid_EFGH : Prop := ∃ E F G H EF GH, EF ≠ GH
def parallel_EF_GH : Prop := EF ∥ GH
def angle_E_eq_3H : Prop := E = 3 * H
def angle_G_eq_4F : Prop := G = 4 * F

-- Theorem statement
theorem find_angle_E (H_value : ℝ) (H_property : H = 45) :
  E = 135 :=
  by
  -- Assume necessary properties from the problem statements
  assume trapezoid_EFGH
  assume parallel_EF_GH : EF ∥ GH
  assume angle_E_eq_3H : E = 3 * H
  have H_value : H = 45 := sorry
  have angle_E_value : E = 135 := sorry
  exact angle_E_value

end find_angle_E_l19_19057


namespace percentage_of_loss_is_25_l19_19864

-- Definitions from conditions
def CP : ℝ := 2800
def SP : ℝ := 2100

-- Proof statement
theorem percentage_of_loss_is_25 : ((CP - SP) / CP) * 100 = 25 := by
  sorry

end percentage_of_loss_is_25_l19_19864


namespace possible_lost_rectangle_area_l19_19443

theorem possible_lost_rectangle_area (areas : Fin 10 → ℕ) (total_area : ℕ) (h_total : total_area = 65) :
  (∃ (i : Fin 10), (64 = total_area - areas i) ∨ (49 = total_area - areas i)) ↔
  (∃ (i : Fin 10), (areas i = 1) ∨ (areas i = 16)) :=
by
  sorry

end possible_lost_rectangle_area_l19_19443


namespace sequence_fill_l19_19290

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l19_19290


namespace goals_scored_by_each_l19_19661

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l19_19661


namespace solve_for_x_l19_19796

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l19_19796


namespace largest_multiple_of_15_below_500_l19_19172

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19172


namespace compute_fraction_at_six_l19_19550

theorem compute_fraction_at_six (x : ℕ) (h : x = 6) : (x^6 - 16 * x^3 + 64) / (x^3 - 8) = 208 := by
  sorry

end compute_fraction_at_six_l19_19550


namespace min_value_l19_19028

/-- Given x and y are positive real numbers such that x + 3y = 2,
    the minimum value of (2x + y) / (xy) is 1/2 * (7 + 2 * sqrt 6). -/
theorem min_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = 2) :
  ∃ c : ℝ, c = (1/2) * (7 + 2 * Real.sqrt 6) ∧ ∀ (x y : ℝ), (0 < x) → (0 < y) → (x + 3 * y = 2) → ((2 * x + y) / (x * y)) ≥ c :=
sorry

end min_value_l19_19028


namespace inequality_l19_19640
-- Import the necessary libraries from Mathlib

-- Define the theorem statement
theorem inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by
  sorry

end inequality_l19_19640


namespace vivian_mail_june_l19_19984

theorem vivian_mail_june :
  ∀ (m_apr m_may m_jul m_aug : ℕ),
  m_apr = 5 →
  m_may = 10 →
  m_jul = 40 →
  ∃ m_jun : ℕ,
  ∃ pattern : ℕ → ℕ,
  (pattern m_apr = m_may) →
  (pattern m_may = m_jun) →
  (pattern m_jun = m_jul) →
  (pattern m_jul = m_aug) →
  (m_aug = 80) →
  pattern m_may = m_may * 2 →
  pattern m_jun = m_jun * 2 →
  pattern m_jun = 20 :=
by
  sorry

end vivian_mail_june_l19_19984


namespace expected_value_of_twelve_sided_die_l19_19352

theorem expected_value_of_twelve_sided_die : 
  let faces := (List.range (12+1)).tail in
  let E := (1 / 12) * (faces.sum) in
  E = 6.5 :=
by
  let faces := (List.range (12+1)).tail
  let E := (1 / 12) * (faces.sum)
  have : faces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] := by
    unfold faces
    simp [List.range, List.tail]
  have : faces.sum = 78 := by
    simp [this]
    norm_num
  have : E = (1 / 12) * 78 := by
    unfold E
    congr
  have : E = 6.5 := by
    simp [this]
    norm_num
  exact this

end expected_value_of_twelve_sided_die_l19_19352


namespace find_sum_pqr_l19_19961

theorem find_sum_pqr (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h : (p + q + r)^3 - p^3 - q^3 - r^3 = 200) : 
  p + q + r = 7 :=
by 
  sorry

end find_sum_pqr_l19_19961


namespace sequence_sum_after_6_steps_l19_19710

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 1435 -- would define how numbers sequence works recursively.
  else sorry -- next steps up to 6
  

theorem sequence_sum_after_6_steps : sequence_sum 6 = 191 := 
by
  sorry

end sequence_sum_after_6_steps_l19_19710


namespace selling_price_30_percent_profit_l19_19871

noncomputable def store_cost : ℝ := 2412.31 / 1.40

theorem selling_price_30_percent_profit : 
  let selling_price_30 := 1.30 * store_cost in
  selling_price_30 = 2240.00 :=
by
  have h : store_cost = 1723.08 := by sorry
  have : 1.30 * store_cost = 2240.00 :=
    by
      calc
        1.30 * store_cost = 1.30 * 1723.08 : by rw [h]
        ... = 2240.00 : by norm_num
  exact this

end selling_price_30_percent_profit_l19_19871


namespace sin_three_pi_over_two_l19_19722

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_over_two_l19_19722


namespace find_variable_value_l19_19911

axiom variable_property (x : ℝ) (h : 4 + 1 / x ≠ 0) : 5 / (4 + 1 / x) = 1 → x = 1

-- Given condition: 5 / (4 + 1 / x) = 1
-- Prove: x = 1
theorem find_variable_value (x : ℝ) (h : 4 + 1 / x ≠ 0) (h1 : 5 / (4 + 1 / x) = 1) : x = 1 :=
variable_property x h h1

end find_variable_value_l19_19911


namespace largest_multiple_of_15_below_500_l19_19173

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19173


namespace expected_value_of_twelve_sided_die_l19_19361

theorem expected_value_of_twelve_sided_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (list.sum faces / list.length faces : ℝ) = 6.5 := 
by
  -- problem defines that there are 12 faces of the die numbered from 1 to 12
  have h_faces_length : list.length faces = 12 := rfl
  -- problem defines the sum of faces computed using the series sum formula
  have h_faces_sum : list.sum faces = 78 := rfl
  exact sorry

end expected_value_of_twelve_sided_die_l19_19361


namespace evaluate_expression_l19_19382

/-
  Define the expressions from the conditions.
  We define the numerator and denominator separately.
-/
def expr_numerator : ℚ := 1 - (1 / 4)
def expr_denominator : ℚ := 1 - (1 / 3)

/-
  Define the original expression to be proven.
  This is our main expression to evaluate.
-/
def expr : ℚ := expr_numerator / expr_denominator

/-
  State the final proof problem that the expression is equal to 9/8.
-/
theorem evaluate_expression : expr = 9 / 8 := sorry

end evaluate_expression_l19_19382


namespace percentage_increase_in_area_is_96_l19_19324

theorem percentage_increase_in_area_is_96 :
  let r₁ := 5
  let r₃ := 7
  let A (r : ℝ) := Real.pi * r^2
  ((A r₃ - A r₁) / A r₁) * 100 = 96 := by
  sorry

end percentage_increase_in_area_is_96_l19_19324


namespace actual_size_of_plot_l19_19084

/-
Theorem: The actual size of the plot of land is 61440 acres.
Given:
- The plot of land is a rectangle.
- The map dimensions are 12 cm by 8 cm.
- 1 cm on the map equals 1 mile in reality.
- One square mile equals 640 acres.
-/

def map_length_cm := 12
def map_width_cm := 8
def cm_to_miles := 1 -- 1 cm equals 1 mile
def mile_to_acres := 640 -- 1 square mile is 640 acres

theorem actual_size_of_plot
  (length_cm : ℕ) (width_cm : ℕ) (cm_to_miles : ℕ → ℕ) (mile_to_acres : ℕ → ℕ) :
  length_cm = 12 → width_cm = 8 →
  (cm_to_miles 1 = 1) →
  (mile_to_acres 1 = 640) →
  (length_cm * width_cm * mile_to_acres (cm_to_miles 1 * cm_to_miles 1) = 61440) :=
by
  intros
  sorry

end actual_size_of_plot_l19_19084


namespace total_wet_surface_area_is_correct_l19_19526

noncomputable def wet_surface_area (cistern_length cistern_width water_depth platform_length platform_width platform_height : ℝ) : ℝ :=
  let two_longer_walls := 2 * (cistern_length * water_depth)
  let two_shorter_walls := 2 * (cistern_width * water_depth)
  let area_walls := two_longer_walls + two_shorter_walls
  let area_bottom := cistern_length * cistern_width
  let submerged_height := water_depth - platform_height
  let two_longer_sides_platform := 2 * (platform_length * submerged_height)
  let two_shorter_sides_platform := 2 * (platform_width * submerged_height)
  let area_platform_sides := two_longer_sides_platform + two_shorter_sides_platform
  area_walls + area_bottom + area_platform_sides

theorem total_wet_surface_area_is_correct :
  wet_surface_area 8 4 1.25 1 0.5 0.75 = 63.5 :=
by
  -- The proof goes here
  sorry

end total_wet_surface_area_is_correct_l19_19526


namespace largest_multiple_of_15_less_than_500_l19_19143

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19143


namespace Joe_time_from_home_to_school_l19_19762

-- Define the parameters
def walking_time := 4 -- minutes
def waiting_time := 2 -- minutes
def running_speed_ratio := 2 -- Joe's running speed is twice his walking speed

-- Define the walking and running times
def running_time (walking_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time / running_speed_ratio

-- Total time it takes Joe to get from home to school
def total_time (walking_time waiting_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time + waiting_time + running_time walking_time running_speed_ratio

-- Conjecture to be proved
theorem Joe_time_from_home_to_school :
  total_time walking_time waiting_time running_speed_ratio = 10 := by
  sorry

end Joe_time_from_home_to_school_l19_19762


namespace initial_people_in_line_l19_19982

theorem initial_people_in_line (X : ℕ) 
  (h1 : X - 6 + 3 = 18) : X = 21 :=
  sorry

end initial_people_in_line_l19_19982


namespace find_function_expression_l19_19914

noncomputable def f (a b x : ℝ) : ℝ := 2 ^ (a * x + b)

theorem find_function_expression
  (a b : ℝ)
  (h1 : f a b 1 = 2)
  (h2 : ∃ g : ℝ → ℝ, (∀ x y : ℝ, f (-a) (-b) x = y ↔ f a b y = x) ∧ g (f a b 1) = 1) :
  ∃ (a b : ℝ), f a b x = 2 ^ (-x + 2) :=
by
  sorry

end find_function_expression_l19_19914


namespace polynomial_evaluation_l19_19889

theorem polynomial_evaluation (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 :=
by
  sorry

end polynomial_evaluation_l19_19889


namespace starling_nests_flying_condition_l19_19047

theorem starling_nests_flying_condition (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → A ≠ B)
  (h3 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → 
  (∃ d1 d2 : ℝ, d1 < d2 ∧ d1 < d2 → d1 > d2)) : n = 3 :=
by
  sorry

end starling_nests_flying_condition_l19_19047


namespace badges_total_l19_19908

theorem badges_total :
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  hermione_badges + luna_badges + celestia_badges = 83 :=
by
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  sorry

end badges_total_l19_19908


namespace difference_between_numbers_l19_19668

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 24365) (h2 : a % 5 = 0) (h3 : (a / 10) = 2 * b) : a - b = 19931 :=
by sorry

end difference_between_numbers_l19_19668


namespace correct_quotient_of_original_division_operation_l19_19612

theorem correct_quotient_of_original_division_operation 
  (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 102)
  (h2 : correct_divisor = 201)
  (h3 : incorrect_quotient = 753)
  (h4 : ∃ k, k = incorrect_quotient * 3) :
  ∃ q, q = 1146 ∧ (correct_divisor * q = incorrect_divisor * (incorrect_quotient * 3)) :=
by
  sorry

end correct_quotient_of_original_division_operation_l19_19612


namespace largest_multiple_of_15_below_500_l19_19178

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19178


namespace max_value_of_expression_l19_19588

noncomputable def expression (x : ℝ) : ℝ :=
  x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 15 * x^4 + 25)

theorem max_value_of_expression : ∃ x : ℝ, (expression x) = 1 / 17 :=
sorry

end max_value_of_expression_l19_19588


namespace division_remainder_correct_l19_19714

theorem division_remainder_correct :
  ∃ q r, 987670 = 128 * q + r ∧ 0 ≤ r ∧ r < 128 ∧ r = 22 :=
by
  sorry

end division_remainder_correct_l19_19714


namespace geometric_sequence_100th_term_l19_19286

theorem geometric_sequence_100th_term :
  ∀ (a₁ a₂ : ℤ) (r : ℤ), a₁ = 5 → a₂ = -15 → r = a₂ / a₁ → 
  (a₁ * r ^ 99 = -5 * 3 ^ 99) :=
by
  intros a₁ a₂ r ha₁ ha₂ hr
  sorry

end geometric_sequence_100th_term_l19_19286


namespace largest_multiple_of_15_less_than_500_l19_19181

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19181


namespace num_bases_for_625_ending_in_1_l19_19891

theorem num_bases_for_625_ending_in_1 :
  (Finset.card (Finset.filter (λ b : ℕ, 624 % b = 0) (Finset.Icc 3 10))) = 4 :=
by
  sorry

end num_bases_for_625_ending_in_1_l19_19891


namespace ratio_d_c_l19_19609

theorem ratio_d_c (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0) 
  (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 16 * x = d) : d / c = -2 :=
by
  sorry

end ratio_d_c_l19_19609


namespace leon_required_score_l19_19925

noncomputable def leon_scores : List ℕ := [72, 68, 75, 81, 79]

theorem leon_required_score (n : ℕ) :
  (List.sum leon_scores + n) / (List.length leon_scores + 1) ≥ 80 ↔ n ≥ 105 :=
by sorry

end leon_required_score_l19_19925


namespace circle_radius_l19_19690

/-
  Given:
  - The area of the circle x = π r^2
  - The circumference of the circle y = 2π r
  - The sum x + y = 72π

  Prove:
  The radius r = 6
-/
theorem circle_radius (r : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : x = π * r ^ 2) 
  (h₂ : y = 2 * π * r) 
  (h₃ : x + y = 72 * π) : 
  r = 6 := 
sorry

end circle_radius_l19_19690


namespace incorrect_population_growth_statement_l19_19238

def population_growth_behavior (p: ℝ → ℝ) : Prop :=
(p 0 < p 1) ∧ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))

def stabilizes_at_K (p: ℝ → ℝ) (K: ℝ) : Prop :=
∃ t₀, ∀ t > t₀, p t = K

def K_value_definition (K: ℝ) (environmental_conditions: ℝ → ℝ) : Prop :=
∀ t, environmental_conditions t = K

theorem incorrect_population_growth_statement (p: ℝ → ℝ) (K: ℝ) (environmental_conditions: ℝ → ℝ)
(h1: population_growth_behavior p)
(h2: stabilizes_at_K p K)
(h3: K_value_definition K environmental_conditions) :
(p 0 > p 1) ∨ (¬ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))) :=
sorry

end incorrect_population_growth_statement_l19_19238


namespace abs_a_gt_abs_c_sub_abs_b_l19_19748

theorem abs_a_gt_abs_c_sub_abs_b (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| :=
sorry

end abs_a_gt_abs_c_sub_abs_b_l19_19748


namespace sum_of_numbers_is_247_l19_19824

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l19_19824


namespace max_arithmetic_sequence_of_primes_less_than_150_l19_19868

theorem max_arithmetic_sequence_of_primes_less_than_150 : 
  ∀ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x) ∧ (∀ x ∈ S, x < 150) ∧ (∃ d, ∀ x ∈ S, ∃ n : ℕ, x = S.min' (by sorry) + n * d) → S.card ≤ 5 := 
by
  sorry

end max_arithmetic_sequence_of_primes_less_than_150_l19_19868


namespace total_molecular_weight_l19_19329

-- Define atomic weights
def atomic_weight (element : String) : Float :=
  match element with
  | "K"  => 39.10
  | "Cr" => 51.996
  | "O"  => 16.00
  | "Fe" => 55.845
  | "S"  => 32.07
  | "Mn" => 54.938
  | _    => 0.0

-- Molecular weights of compounds
def molecular_weight_K2Cr2O7 : Float := 
  2 * atomic_weight "K" + 2 * atomic_weight "Cr" + 7 * atomic_weight "O"

def molecular_weight_Fe2_SO4_3 : Float := 
  2 * atomic_weight "Fe" + 3 * atomic_weight "S" + 12 * atomic_weight "O"

def molecular_weight_KMnO4 : Float := 
  atomic_weight "K" + atomic_weight "Mn" + 4 * atomic_weight "O"

-- Proof statement 
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 + 3 * molecular_weight_Fe2_SO4_3 + 5 * molecular_weight_KMnO4 = 3166.658 :=
by
  sorry

end total_molecular_weight_l19_19329


namespace production_difference_l19_19072

variables (p h : ℕ)

def first_day_production := p * h

def second_day_production := (p + 5) * (h - 3)

-- Given condition
axiom p_eq_3h : p = 3 * h

theorem production_difference : first_day_production p h - second_day_production p h = 4 * h + 15 :=
by
  sorry

end production_difference_l19_19072


namespace doughnuts_per_person_l19_19461

-- Define the number of dozens bought by Samuel
def samuel_dozens : ℕ := 2

-- Define the number of dozens bought by Cathy
def cathy_dozens : ℕ := 3

-- Define the number of doughnuts in one dozen
def dozen : ℕ := 12

-- Define the total number of people
def total_people : ℕ := 10

-- Prove that each person receives 6 doughnuts
theorem doughnuts_per_person :
  ((samuel_dozens * dozen) + (cathy_dozens * dozen)) / total_people = 6 :=
by
  sorry

end doughnuts_per_person_l19_19461


namespace rectangle_dimensions_l19_19972

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : 2 * l + 2 * w = 150) 
  (h2 : l = w + 15) : 
  w = 30 ∧ l = 45 := 
  by 
  sorry

end rectangle_dimensions_l19_19972


namespace frank_ryan_problem_ratio_l19_19546

theorem frank_ryan_problem_ratio 
  (bill_problems : ℕ)
  (h1 : bill_problems = 20)
  (ryan_problems : ℕ)
  (h2 : ryan_problems = 2 * bill_problems)
  (frank_problems_per_type : ℕ)
  (h3 : frank_problems_per_type = 30)
  (types : ℕ)
  (h4 : types = 4) : 
  frank_problems_per_type * types / ryan_problems = 3 := by
  sorry

end frank_ryan_problem_ratio_l19_19546


namespace find_p0_over_q0_l19_19964

-- Definitions

def p (x : ℝ) := 3 * (x - 4) * (x - 2)
def q (x : ℝ) := (x - 4) * (x + 3)

theorem find_p0_over_q0 : (p 0) / (q 0) = -2 :=
by
  -- Prove the equality given the conditions
  sorry

end find_p0_over_q0_l19_19964


namespace jillian_max_apartment_size_l19_19875

theorem jillian_max_apartment_size :
  ∀ s : ℝ, (1.10 * s = 880) → s = 800 :=
by
  intros s h
  sorry

end jillian_max_apartment_size_l19_19875


namespace min_value_of_2a_b_c_l19_19732

-- Given conditions
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (h : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3)

-- Question to prove
theorem min_value_of_2a_b_c : 2 * a + b + c = 2 * Real.sqrt 3 + 2 :=
sorry

end min_value_of_2a_b_c_l19_19732


namespace constant_term_in_expansion_l19_19846

theorem constant_term_in_expansion {α : Type*} [Comm_ring α] (x : α) :
  let term := (10.choose 5) * 4^5 in
  (term : α) = 258048 :=
by
  sorry

end constant_term_in_expansion_l19_19846


namespace sum_of_numbers_is_247_l19_19822

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l19_19822


namespace gcd_of_12012_and_18018_l19_19584

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l19_19584


namespace largest_multiple_of_15_less_than_500_l19_19188

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19188


namespace students_per_group_l19_19090

theorem students_per_group (total_students not_picked_groups groups : ℕ) (h₁ : total_students = 65) (h₂ : not_picked_groups = 17) (h₃ : groups = 8) :
  (total_students - not_picked_groups) / groups = 6 := by
  sorry

end students_per_group_l19_19090


namespace staircase_toothpicks_l19_19540

theorem staircase_toothpicks (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 9 ∧ r = 3 ∧ n = 3 + 4 
  → (a * r ^ 3 + a * r ^ 2 + a * r + a) + (a * r ^ 2 + a * r + a) + (a * r + a) + a = 351 :=
by
  sorry

end staircase_toothpicks_l19_19540


namespace total_coins_is_twenty_l19_19683

def piles_of_quarters := 2
def piles_of_dimes := 3
def coins_per_pile := 4

theorem total_coins_is_twenty : piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 20 :=
by sorry

end total_coins_is_twenty_l19_19683


namespace odd_function_domain_real_l19_19927

theorem odd_function_domain_real
  (a : ℤ)
  (h_condition : a = -1 ∨ a = 1 ∨ a = 3) :
  (∀ x : ℝ, ∃ y : ℝ, x ≠ 0 → y = x^a) →
  (∀ x : ℝ, x ≠ 0 → (x^a = (-x)^a)) →
  (a = 1 ∨ a = 3) :=
sorry

end odd_function_domain_real_l19_19927


namespace solve_for_x_l19_19791

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l19_19791


namespace largest_multiple_15_under_500_l19_19122

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19122


namespace value_of_unknown_number_l19_19686

theorem value_of_unknown_number (x n : ℤ) 
  (h1 : x = 88320) 
  (h2 : x + n + 9211 - 1569 = 11901) : 
  n = -84061 :=
by
  sorry

end value_of_unknown_number_l19_19686


namespace two_coins_heads_probability_l19_19677

theorem two_coins_heads_probability :
  let outcomes := [{fst := true, snd := true}, {fst := true, snd := false}, {fst := false, snd := true}, {fst := false, snd := false}],
      favorable := {fst := true, snd := true}
  in (favorable ∈ outcomes) → (favorable :: (List.erase outcomes favorable).length) / outcomes.length = 1 / 4 :=
by
  sorry

end two_coins_heads_probability_l19_19677


namespace solve_for_x_l19_19797

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l19_19797


namespace find_f_at_2_l19_19035

noncomputable def f (x : ℝ) : ℝ :=
  (x + 2) * (x - 1) * (x - 3) * (x + 4) - x^2

theorem find_f_at_2 :
  (f(-2) = -4) ∧ (f(1) = -1) ∧ (f(3) = -9) ∧ (f(-4) = -16) ∧ (f 2 = -28) :=
by
  have h₁ : f (-2) = (0 : ℝ) := by sorry
  have h₂ : f 1 = (0 : ℝ) := by sorry
  have h₃ : f 3 = (0 : ℝ) := by sorry
  have h₄ : f (-4) = (0 : ℝ) := by sorry
  have h₅ : f 2 = (0 : ℝ) := by sorry
  exact ⟨h₁, h₂, h₃, h₄, h₅⟩

end find_f_at_2_l19_19035


namespace solve_quadratic_eq_l19_19313

theorem solve_quadratic_eq (x : ℝ) : 4 * x ^ 2 - (x - 1) ^ 2 = 0 ↔ x = -1 ∨ x = 1 / 3 :=
by
  sorry

end solve_quadratic_eq_l19_19313


namespace circle_tangent_line_l19_19228

noncomputable def line_eq (x : ℝ) : ℝ := 2 * x + 1
noncomputable def circle_eq (x y b : ℝ) : ℝ := x^2 + (y - b)^2

theorem circle_tangent_line 
  (b : ℝ) 
  (tangency : ∃ b, (1 - b) / (0 - 1) = -(1 / 2)) 
  (center_point : 1^2 + (3 - b)^2 = 5 / 4) : 
  circle_eq 1 3 b = circle_eq 0 b (7/2) :=
sorry

end circle_tangent_line_l19_19228


namespace games_needed_to_declare_winner_l19_19237

def single_elimination_games (T : ℕ) : ℕ :=
  T - 1

theorem games_needed_to_declare_winner (T : ℕ) :
  (single_elimination_games 23 = 22) :=
by
  sorry

end games_needed_to_declare_winner_l19_19237


namespace greatest_value_x_l19_19970

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l19_19970


namespace number_of_people_purchased_only_book_A_l19_19481

-- Definitions based on the conditions
variable (A B x y z w : ℕ)
variable (h1 : z = 500)
variable (h2 : z = 2 * y)
variable (h3 : w = z)
variable (h4 : x + y + z + w = 2500)
variable (h5 : A = x + z)
variable (h6 : B = y + z)
variable (h7 : A = 2 * B)

-- The statement we want to prove
theorem number_of_people_purchased_only_book_A :
  x = 1000 :=
by
  -- The proof steps will be filled here
  sorry

end number_of_people_purchased_only_book_A_l19_19481


namespace smallest_bdf_value_l19_19939

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l19_19939


namespace cos_value_l19_19403

theorem cos_value (α : ℝ) 
  (h1 : Real.sin (α + Real.pi / 12) = 1 / 3) : 
  Real.cos (α + 7 * Real.pi / 12) = -(1 + Real.sqrt 24) / 6 :=
sorry

end cos_value_l19_19403


namespace largest_multiple_of_15_under_500_l19_19214

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19214


namespace number_of_outfits_l19_19078

theorem number_of_outfits : (5 * 4 * 6 * 3) = 360 := by
  sorry

end number_of_outfits_l19_19078


namespace number_of_valid_sequences_l19_19301

-- Define the sequence property
def sequence_property (b : Fin 10 → Fin 10) : Prop :=
  ∀ i : Fin 10, 2 ≤ i → (∃ j : Fin 10, j < i ∧ (b j = b i + 1 ∨ b j = b i - 1 ∨ b j = b i + 2 ∨ b j = b i - 2))

-- Define the set of such sequences
def valid_sequences : Set (Fin 10 → Fin 10) := {b | sequence_property b}

-- Define the number of such sequences
def number_of_sequences : Fin 512 :=
  sorry -- Proof omitted for brevity

-- The final statement
theorem number_of_valid_sequences : number_of_sequences = 512 :=
  sorry  -- Skip proof

end number_of_valid_sequences_l19_19301


namespace candy_bar_sugar_calories_l19_19775

theorem candy_bar_sugar_calories
  (candy_bars : Nat)
  (soft_drink_calories : Nat)
  (soft_drink_sugar_percentage : Float)
  (recommended_sugar_intake : Nat)
  (excess_percentage : Nat)
  (sugar_in_each_bar : Nat) :
  candy_bars = 7 ∧
  soft_drink_calories = 2500 ∧
  soft_drink_sugar_percentage = 0.05 ∧
  recommended_sugar_intake = 150 ∧
  excess_percentage = 100 →
  sugar_in_each_bar = 25 := by
  sorry

end candy_bar_sugar_calories_l19_19775


namespace sally_earnings_l19_19948

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l19_19948


namespace true_false_questions_count_l19_19759

/-- 
 In an answer key for a quiz, there are some true-false questions followed by 3 multiple-choice questions with 4 answer choices each. 
 The correct answers to all true-false questions cannot be the same. 
 There are 384 ways to write the answer key. How many true-false questions are there?
-/
theorem true_false_questions_count : 
  ∃ n : ℕ, 2^n - 2 = 6 ∧ (2^n - 2) * 4^3 = 384 := 
sorry

end true_false_questions_count_l19_19759


namespace largest_multiple_of_15_less_than_500_l19_19126

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19126


namespace roots_equation_l19_19810

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l19_19810


namespace proposition_B_correct_l19_19703

theorem proposition_B_correct : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x :=
by
  sorry

end proposition_B_correct_l19_19703


namespace isosceles_base_l19_19482

theorem isosceles_base (s b : ℕ) 
  (h1 : 3 * s = 45) 
  (h2 : 2 * s + b = 40) 
  (h3 : s = 15): 
  b = 10 :=
  sorry

end isosceles_base_l19_19482


namespace find_base_b_l19_19257

theorem find_base_b (b : ℕ) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 6) = (4 * b^2 + 1 * b + 1) →
  7 < b →
  b = 10 :=
by
  intro h₁ h₂
  sorry

end find_base_b_l19_19257


namespace largest_multiple_of_15_less_than_500_l19_19139

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19139


namespace largest_multiple_of_15_less_than_500_l19_19127

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19127


namespace probability_at_least_one_pair_two_women_correct_l19_19340

noncomputable def probability_at_least_one_pair_two_women :=
  let total_ways := Nat.factorial 12 / (2^6 * Nat.factorial 6)
  let favorable_ways := total_ways - Nat.factorial 6
  let probability := favorable_ways / total_ways
  probability ≈ 0.93

theorem probability_at_least_one_pair_two_women_correct :
  probability_at_least_one_pair_two_women = 0.93 := by
  sorry

end probability_at_least_one_pair_two_women_correct_l19_19340


namespace sequence_length_l19_19033

theorem sequence_length 
  (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) (n : ℕ) 
  (h₁ : a₁ = -4) 
  (h₂ : d = 3) 
  (h₃ : aₙ = 32) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  n = 13 := 
by 
  sorry

end sequence_length_l19_19033


namespace johns_height_in_feet_l19_19300

def initial_height := 66 -- John's initial height in inches
def growth_rate := 2      -- Growth rate in inches per month
def growth_duration := 3  -- Growth duration in months
def inches_per_foot := 12 -- Conversion factor from inches to feet

def total_growth : ℕ := growth_rate * growth_duration

def final_height_in_inches : ℕ := initial_height + total_growth

-- Now, proof that the final height in feet is 6
theorem johns_height_in_feet : (final_height_in_inches / inches_per_foot) = 6 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end johns_height_in_feet_l19_19300


namespace tangent_line_equation_l19_19657

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := 2 * x^2 - x
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Statement of the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (b = 1 - 3 * 1) ∧ 
  (m = 3) ∧ 
  ∀ (x y : ℝ), y = m * x + b → 3 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_equation_l19_19657


namespace correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l19_19989

theorem correct_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

-- Incorrect options for clarity
theorem incorrect_division (a : ℝ) : a^6 / a^2 ≠ a^3 :=
by sorry

theorem incorrect_multiplication (a : ℝ) : a^2 * a^3 ≠ a^6 :=
by sorry

theorem incorrect_addition (a : ℝ) : (a^2 + a^3) ≠ a^5 :=
by sorry

end correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l19_19989


namespace largest_multiple_of_15_under_500_l19_19215

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19215


namespace sum_of_three_numbers_l19_19841

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19841


namespace gcd_12012_18018_l19_19576

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19576


namespace income_of_deceased_is_correct_l19_19993

-- Definitions based on conditions
def family_income_before_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def family_income_after_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def income_of_deceased (total_before: ℝ) (total_after: ℝ) : ℝ := total_before - total_after

-- Given conditions
def avg_income_before : ℝ := 782
def avg_income_after : ℝ := 650
def num_members_before : ℕ := 4
def num_members_after : ℕ := 3

-- Mathematical statement
theorem income_of_deceased_is_correct : 
  income_of_deceased (family_income_before_death avg_income_before num_members_before) 
                     (family_income_after_death avg_income_after num_members_after) = 1178 :=
by
  sorry

end income_of_deceased_is_correct_l19_19993


namespace largest_multiple_of_15_under_500_l19_19211

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19211


namespace convert_base_8_to_10_l19_19556

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l19_19556


namespace negation_of_universal_statement_l19_19971

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_statement_l19_19971


namespace solve_quadratic_equation_l19_19977

theorem solve_quadratic_equation (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
sorry

end solve_quadratic_equation_l19_19977


namespace tv_horizontal_length_l19_19067

noncomputable def rectangleTvLengthRatio (l h : ℝ) : Prop :=
  l / h = 16 / 9

noncomputable def rectangleTvDiagonal (l h d : ℝ) : Prop :=
  l^2 + h^2 = d^2

theorem tv_horizontal_length
  (h : ℝ)
  (h_positive : h > 0)
  (d : ℝ)
  (h_ratio : rectangleTvLengthRatio l h)
  (h_diagonal : rectangleTvDiagonal l h d)
  (h_diagonal_value : d = 36) :
  l = 56.27 :=
by
  sorry

end tv_horizontal_length_l19_19067


namespace share_of_B_in_profit_l19_19872

variable {D : ℝ} (hD_pos : 0 < D)

def investment (D : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let C := 2.5 * D
  let B := 1.25 * D
  let A := 5 * B
  (A, B, C, D)

def totalInvestment (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem share_of_B_in_profit (D : ℝ) (profit : ℝ) (hD : 0 < D)
  (h_profit : profit = 8000) :
  let ⟨A, B, C, D⟩ := investment D
  B / totalInvestment A B C D * profit = 1025.64 :=
by
  sorry

end share_of_B_in_profit_l19_19872


namespace larger_root_of_degree_11_l19_19988

theorem larger_root_of_degree_11 {x : ℝ} :
  (∃ x₁, x₁ > 0 ∧ (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9)) ∧
  (∃ x₂, x₂ > 0 ∧ (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11)) →
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧
    (x₁ + x₁^2 + x₁^3 + x₁^4 + x₁^5 + x₁^6 + x₁^7 + x₁^8 = 8 - 10 * x₁^9) ∧
    (x₂ + x₂^2 + x₂^3 + x₂^4 + x₂^5 + x₂^6 + x₂^7 + x₂^8 + x₂^9 + x₂^10 = 8 - 10 * x₂^11) ∧
    x₁ < x₂) :=
by
  sorry

end larger_root_of_degree_11_l19_19988


namespace average_score_l19_19317

theorem average_score (avg1 avg2 : ℕ) (n1 n2 total_matches : ℕ) (total_avg : ℕ) 
  (h1 : avg1 = 60) 
  (h2 : avg2 = 70) 
  (h3 : n1 = 10) 
  (h4 : n2 = 15) 
  (h5 : total_matches = 25) 
  (h6 : total_avg = 66) :
  (( (avg1 * n1) + (avg2 * n2) ) / total_matches = total_avg) :=
by
  sorry

end average_score_l19_19317


namespace evaluate_cubic_diff_l19_19023

theorem evaluate_cubic_diff (x y : ℝ) (h1 : x + y = 12) (h2 : 2 * x + y = 16) : x^3 - y^3 = -448 := 
by
    sorry

end evaluate_cubic_diff_l19_19023


namespace total_canoes_by_end_of_april_l19_19378

def canoes_built_jan : Nat := 4

def canoes_built_next_month (prev_month : Nat) : Nat := 3 * prev_month

def canoes_built_feb : Nat := canoes_built_next_month canoes_built_jan
def canoes_built_mar : Nat := canoes_built_next_month canoes_built_feb
def canoes_built_apr : Nat := canoes_built_next_month canoes_built_mar

def total_canoes_built : Nat := canoes_built_jan + canoes_built_feb + canoes_built_mar + canoes_built_apr

theorem total_canoes_by_end_of_april : total_canoes_built = 160 :=
by
  sorry

end total_canoes_by_end_of_april_l19_19378


namespace greatest_possible_x_l19_19967

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l19_19967


namespace weight_of_newcomer_l19_19318

theorem weight_of_newcomer (avg_old W_initial : ℝ) 
  (h_weight_range : 400 ≤ W_initial ∧ W_initial ≤ 420)
  (h_avg_increase : avg_old + 3.5 = (W_initial - 47 + W_new) / 6)
  (h_person_replaced : 47 = 47) :
  W_new = 68 := 
sorry

end weight_of_newcomer_l19_19318


namespace largest_whole_number_n_l19_19216

theorem largest_whole_number_n : ∃ (n : ℕ), (frac (n / 7) + 1/3 < 1) ∧ ∀ (m : ℕ), (frac (m / 7) + 1/3 < 1) → m ≤ n :=
begin
  use 4,
  split,
  { norm_num },
  { intros m h,
    norm_num at h,
    sorry
  }
end

end largest_whole_number_n_l19_19216


namespace samantha_routes_l19_19788

-- Definitions of the conditions
def blocks_west_to_sw_corner := 3
def blocks_south_to_sw_corner := 2
def blocks_east_to_school := 4
def blocks_north_to_school := 3
def ways_house_to_sw_corner : ℕ := Nat.choose (blocks_west_to_sw_corner + blocks_south_to_sw_corner) blocks_south_to_sw_corner
def ways_through_park : ℕ := 2
def ways_ne_corner_to_school : ℕ := Nat.choose (blocks_east_to_school + blocks_north_to_school) blocks_north_to_school

-- The proof statement
theorem samantha_routes : (ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school) = 700 :=
by
  -- Using "sorry" as a placeholder for the actual proof
  sorry

end samantha_routes_l19_19788


namespace add_number_l19_19509

theorem add_number (x : ℕ) (h : 43 + x = 81) : x + 25 = 63 :=
by {
  -- Since this is focusing on the structure and statement no proof steps are required
  sorry
}

end add_number_l19_19509


namespace sum_of_cubes_l19_19044

theorem sum_of_cubes (x y : ℂ) (h1 : x + y = 1) (h2 : x * y = 1) : x^3 + y^3 = -2 := 
by 
  sorry

end sum_of_cubes_l19_19044


namespace find_positive_integer_tuples_l19_19251

theorem find_positive_integer_tuples
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hz_prime : Prime z) :
  z ^ x = y ^ 3 + 1 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_positive_integer_tuples_l19_19251


namespace smallest_product_bdf_l19_19943

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l19_19943


namespace min_brilliant_triple_product_l19_19702

theorem min_brilliant_triple_product :
  ∃ a b c : ℕ, a > b ∧ b > c ∧ Prime a ∧ Prime b ∧ Prime c ∧ (a = b + 2 * c) ∧ (∃ k : ℕ, (a + b + c) = k^2) ∧ (a * b * c = 35651) :=
by
  sorry

end min_brilliant_triple_product_l19_19702


namespace triangle_area_l19_19497

noncomputable def area_of_triangle (l1 l2 l3 : ℝ × ℝ → Prop) (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem triangle_area :
  let A := (1, 6)
  let B := (-1, 6)
  let C := (0, 4)
  ∀ x y : ℝ, 
    (y = 6 → l1 (x, y)) ∧ 
    (y = 2 * x + 4 → l2 (x, y)) ∧ 
    (y = -2 * x + 4 → l3 (x, y)) →
  area_of_triangle l1 l2 l3 A B C = 1 :=
by 
  intros
  unfold area_of_triangle
  sorry

end triangle_area_l19_19497


namespace largest_multiple_of_15_less_than_500_is_495_l19_19158

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19158


namespace michael_boxes_l19_19449

theorem michael_boxes (total_blocks boxes_per_box : ℕ) (h1: total_blocks = 16) (h2: boxes_per_box = 2) :
  total_blocks / boxes_per_box = 8 :=
by
  sorry

end michael_boxes_l19_19449


namespace range_of_k_l19_19262

theorem range_of_k (k : ℝ) : (∀ (x : ℝ), k * x ^ 2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) := 
by 
  sorry

end range_of_k_l19_19262


namespace probability_hare_claims_not_hare_then_not_rabbit_l19_19513

noncomputable def probability_hare_given_claims : ℚ := (27 / 59)

theorem probability_hare_claims_not_hare_then_not_rabbit
  (population : ℚ) (hares : ℚ) (rabbits : ℚ)
  (belief_hare_not_hare : ℚ) (belief_hare_not_rabbit : ℚ)
  (belief_rabbit_not_hare : ℚ) (belief_rabbit_not_rabbit : ℚ) :
  population = 1 ∧ hares = 1/2 ∧ rabbits = 1/2 ∧
  belief_hare_not_hare = 1/4 ∧ belief_hare_not_rabbit = 3/4 ∧
  belief_rabbit_not_hare = 2/3 ∧ belief_rabbit_not_rabbit = 1/3 →
  (27 / 59) = probability_hare_given_claims :=
sorry

end probability_hare_claims_not_hare_then_not_rabbit_l19_19513


namespace find_triangles_l19_19284

/-- In a triangle, if the side lengths a, b, c (a ≤ b ≤ c) are integers, form a geometric progression (i.e., b² = ac),
    and at least one of a or c is equal to 100, then the possible values for the triple (a, b, c) are:
    (49, 70, 100), (64, 80, 100), (81, 90, 100), 
    (100, 100, 100), (100, 110, 121), (100, 120, 144),
    (100, 130, 169), (100, 140, 196), (100, 150, 225), (100, 160, 256). 
-/
theorem find_triangles (a b c : ℕ) (h1 : a ≤ b ∧ b ≤ c) 
(h2 : b * b = a * c)
(h3 : a = 100 ∨ c = 100) : 
  (a = 49 ∧ b = 70 ∧ c = 100) ∨ 
  (a = 64 ∧ b = 80 ∧ c = 100) ∨ 
  (a = 81 ∧ b = 90 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 100 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 110 ∧ c = 121) ∨ 
  (a = 100 ∧ b = 120 ∧ c = 144) ∨ 
  (a = 100 ∧ b = 130 ∧ c = 169) ∨ 
  (a = 100 ∧ b = 140 ∧ c = 196) ∨ 
  (a = 100 ∧ b = 150 ∧ c = 225) ∨ 
  (a = 100 ∧ b = 160 ∧ c = 256) := sorry

end find_triangles_l19_19284


namespace largest_multiple_of_15_less_than_500_l19_19104

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19104


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l19_19652

-- Define the solutions to the given quadratic equations

theorem solve_eq1 (x : ℝ) : 2 * x ^ 2 - 8 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : x ^ 2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1 :=
by sorry

theorem solve_eq3 (x : ℝ) : 5 * x ^ 2 - 4 * x - 1 = 0 ↔ x = -1 / 5 ∨ x = 1 :=
by sorry

theorem solve_eq4 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l19_19652


namespace carla_sharpening_time_l19_19715

theorem carla_sharpening_time (x : ℕ) (h : x + 3 * x = 40) : x = 10 :=
by
  sorry

end carla_sharpening_time_l19_19715


namespace largest_multiple_of_15_less_than_500_l19_19151

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19151


namespace line_passing_through_quadrants_l19_19905

theorem line_passing_through_quadrants (a : ℝ) :
  (∀ x : ℝ, (3 * a - 1) * x - 1 ≠ 0) →
  (3 * a - 1 > 0) →
  a > 1 / 3 :=
by
  intro h1 h2
  -- proof to be filled
  sorry

end line_passing_through_quadrants_l19_19905


namespace sum_of_star_tips_l19_19779

theorem sum_of_star_tips :
  let n := 9
  let alpha := 80  -- in degrees
  let total := n * alpha
  total = 720 := by sorry

end sum_of_star_tips_l19_19779


namespace expected_value_twelve_sided_die_l19_19367

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_twelve_sided_die_l19_19367


namespace basketball_score_l19_19517

theorem basketball_score (score_game1 : ℕ) (score_game2 : ℕ) (score_game3 : ℕ) (score_game4 : ℕ) (score_total_games8 : ℕ) (score_total_games9 : ℕ) :
  score_game1 = 18 ∧ score_game2 = 22 ∧ score_game3 = 15 ∧ score_game4 = 20 ∧ 
  (score_game1 + score_game2 + score_game3 + score_game4) / 4 < score_total_games8 / 8 ∧ 
  score_total_games9 / 9 > 19 →
  score_total_games9 - score_total_games8 ≥ 21 :=
by
-- proof steps would be provided here based on the given solution
sorry

end basketball_score_l19_19517


namespace candies_left_is_correct_l19_19249

-- Define the number of candies bought on different days
def candiesBoughtTuesday : ℕ := 3
def candiesBoughtThursday : ℕ := 5
def candiesBoughtFriday : ℕ := 2

-- Define the number of candies eaten
def candiesEaten : ℕ := 6

-- Define the total candies left
def candiesLeft : ℕ := (candiesBoughtTuesday + candiesBoughtThursday + candiesBoughtFriday) - candiesEaten

theorem candies_left_is_correct : candiesLeft = 4 := by
  -- Placeholder proof: replace 'sorry' with the actual proof when necessary
  sorry

end candies_left_is_correct_l19_19249


namespace find_number_l19_19089

theorem find_number (n : ℕ) (h₁ : ∀ x : ℕ, 21 + 7 * x = n ↔ 3 + x = 47):
  n = 329 :=
by
  -- Proof will go here
  sorry

end find_number_l19_19089


namespace minimum_knights_in_tournament_l19_19529

def knights_tournament : Prop :=
  ∃ (N : ℕ), (∀ (x : ℕ), x = N / 4 →
    ∃ (k : ℕ), k = (3 * x - 1) / 7 → N = 20)

theorem minimum_knights_in_tournament : knights_tournament :=
  sorry

end minimum_knights_in_tournament_l19_19529


namespace circle_center_sum_l19_19563

theorem circle_center_sum (x y : ℝ) (h : (x - 2)^2 + (y + 1)^2 = 15) : x + y = 1 :=
sorry

end circle_center_sum_l19_19563


namespace probability_red_higher_than_green_l19_19699

theorem probability_red_higher_than_green :
  let P (k : ℕ) := 2^(-k)
  in (∑' (k : ℕ), P k * P k) = (1 : ℝ) / 3 :=
by
  sorry

end probability_red_higher_than_green_l19_19699


namespace largest_multiple_of_15_less_than_500_l19_19165

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19165


namespace original_number_l19_19696

theorem original_number (x : ℝ) (h : 1.35 * x = 680) : x = 503.70 :=
sorry

end original_number_l19_19696


namespace bus_interval_l19_19316

theorem bus_interval (num_departures : ℕ) (total_duration : ℕ) (interval : ℕ)
  (h1 : num_departures = 11)
  (h2 : total_duration = 60)
  (h3 : interval = total_duration / (num_departures - 1)) :
  interval = 6 :=
by
  sorry

end bus_interval_l19_19316


namespace gcd_of_78_and_36_l19_19983

theorem gcd_of_78_and_36 :
  Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_of_78_and_36_l19_19983


namespace largest_multiple_of_15_less_than_500_l19_19128

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19128


namespace inequality_am_gm_l19_19931

theorem inequality_am_gm (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by
  sorry

end inequality_am_gm_l19_19931


namespace solve_for_x_l19_19795

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l19_19795


namespace A_inter_B_eq_l19_19742

-- Define set A based on the condition for different integer k.
def A (k : ℤ) : Set ℝ := {x | 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

-- Define set B based on its condition.
def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

-- The final proof problem to show A ∩ B equals to the given set.
theorem A_inter_B_eq : 
  (⋃ k : ℤ, A k) ∩ B = {x | (-Real.pi < x ∧ x < 0) ∨ (Real.pi < x ∧ x < 4)} :=
by
  sorry

end A_inter_B_eq_l19_19742


namespace circumference_of_tank_B_l19_19079

noncomputable def radius_of_tank (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def volume_of_tank (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem circumference_of_tank_B 
  (h_A : ℝ) (C_A : ℝ) (h_B : ℝ) (volume_ratio : ℝ)
  (hA_pos : 0 < h_A) (CA_pos : 0 < C_A) (hB_pos : 0 < h_B) (vr_pos : 0 < volume_ratio) :
  2 * Real.pi * (radius_of_tank (volume_of_tank (radius_of_tank C_A) h_A / (volume_ratio * Real.pi * h_B))) = 17.7245 :=
by 
  sorry

end circumference_of_tank_B_l19_19079


namespace largest_multiple_of_15_below_500_l19_19177

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19177


namespace sqrt_720_simplified_l19_19648

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l19_19648


namespace find_diff_eq_l19_19254

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
∃ (C1 C2 : ℝ), ∀ x : ℝ, y x = C1 * x + C2

theorem find_diff_eq (y : ℝ → ℝ) (C1 C2 : ℝ) (h : ∀ x : ℝ, y x = C1 * x + C2) :
  ∀ x : ℝ, (deriv (deriv y)) x = 0 :=
by
  sorry

end find_diff_eq_l19_19254


namespace largest_multiple_of_15_less_than_500_l19_19196

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19196


namespace find_z_add_inv_y_l19_19469

theorem find_z_add_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) : z + 1 / y = 5 / 27 := by
  sorry

end find_z_add_inv_y_l19_19469


namespace solve_equation_l19_19792

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l19_19792


namespace sum_of_three_numbers_l19_19830

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19830


namespace pen_and_notebook_cost_l19_19933

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 17 * p + 5 * n = 200 ∧ p > n ∧ p + n = 16 := 
by
  sorry

end pen_and_notebook_cost_l19_19933


namespace parallel_lines_necessity_parallel_lines_not_sufficiency_l19_19479

theorem parallel_lines_necessity (a b : ℝ) (h : 2 * b = a * 2) : ab = 4 :=
by sorry

theorem parallel_lines_not_sufficiency (a b : ℝ) (h : ab = 4) : 
  ¬ (2 * b = a * 2 ∧ (2 * a - 2 = 0 -> 2 * b - 2 = 0)) :=
by sorry

end parallel_lines_necessity_parallel_lines_not_sufficiency_l19_19479


namespace alexandra_magazines_l19_19534

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l19_19534


namespace pencils_sold_l19_19607

theorem pencils_sold (C S : ℝ) (n : ℝ) 
  (h1 : 12 * C = n * S) (h2 : S = 1.5 * C) : n = 8 := by
  sorry

end pencils_sold_l19_19607


namespace fewer_seats_on_right_side_l19_19611

-- Definitions based on the conditions
def left_seats := 15
def seats_per_seat := 3
def back_seat_capacity := 8
def total_capacity := 89

-- Statement to prove the problem
theorem fewer_seats_on_right_side : left_seats - (total_capacity - back_seat_capacity - (left_seats * seats_per_seat)) / seats_per_seat = 3 := 
by
  -- proof steps go here
  sorry

end fewer_seats_on_right_side_l19_19611


namespace sally_seashells_l19_19951

theorem sally_seashells 
  (seashells_monday : ℕ)
  (seashells_tuesday : ℕ)
  (price_per_seashell : ℝ)
  (h_monday : seashells_monday = 30)
  (h_tuesday : seashells_tuesday = seashells_monday / 2)
  (h_price : price_per_seashell = 1.2) :
  let total_seashells := seashells_monday + seashells_tuesday in
  let total_money := total_seashells * price_per_seashell in
  total_money = 54 := 
by
  sorry

end sally_seashells_l19_19951


namespace amount_c_is_1600_l19_19995

-- Given conditions
def total_money : ℕ := 2000
def ratio_b_c : (ℕ × ℕ) := (4, 16)

-- Define the total_parts based on the ratio
def total_parts := ratio_b_c.fst + ratio_b_c.snd

-- Define the value of each part
def value_per_part := total_money / total_parts

-- Calculate the amount for c
def amount_c_gets := ratio_b_c.snd * value_per_part

-- Main theorem stating the problem
theorem amount_c_is_1600 : amount_c_gets = 1600 := by
  -- Proof would go here
  sorry

end amount_c_is_1600_l19_19995


namespace a6_is_3_l19_19268

noncomputable def a4 := 8 / 2 -- Placeholder for positive root
noncomputable def a8 := 8 / 2 -- Placeholder for the second root (we know they are both the same for now)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) = (a (n + 1))^2

theorem a6_is_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4_a8: a 4 = a4) (h_a4_a8_root : a 8 = a8) : 
  a 6 = 3 :=
by
  sorry

end a6_is_3_l19_19268


namespace largest_multiple_of_15_less_than_500_is_495_l19_19155

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19155


namespace find_product_in_geometric_sequence_l19_19429

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

theorem find_product_in_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 7 * a 13 = 8) : 
  a 3 * a 11 = 4 :=
by sorry

end find_product_in_geometric_sequence_l19_19429


namespace sqrt_720_simplified_l19_19647

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l19_19647


namespace polynomial_divisible_by_squared_root_l19_19782

noncomputable def f (a1 a2 a3 a4 x : ℝ) : ℝ := 
  x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4

noncomputable def f_prime (a1 a2 a3 a4 x : ℝ) : ℝ := 
  4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_squared_root 
  (a1 a2 a3 a4 x0 : ℝ) 
  (h1 : f a1 a2 a3 a4 x0 = 0) 
  (h2 : f_prime a1 a2 a3 a4 x0 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x, f a1 a2 a3 a4 x = (x - x0)^2 * g x := 
sorry

end polynomial_divisible_by_squared_root_l19_19782


namespace largest_multiple_of_15_less_than_500_l19_19194

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19194


namespace price_of_second_variety_l19_19080

-- Define prices and conditions
def price_first : ℝ := 126
def price_third : ℝ := 175.5
def mixture_price : ℝ := 153
def total_weight : ℝ := 4

-- Define unknown price
variable (x : ℝ)

-- Definition of the weighted mixture price
theorem price_of_second_variety :
  (1 * price_first) + (1 * x) + (2 * price_third) = total_weight * mixture_price →
  x = 135 :=
by
  sorry

end price_of_second_variety_l19_19080


namespace arithmetic_sequence_n_l19_19737

theorem arithmetic_sequence_n (a_n : ℕ → ℕ) (S_n : ℕ) (n : ℕ) 
  (h1 : ∀ i, a_n i = 20 + (i - 1) * (54 - 20) / (n - 1)) 
  (h2 : S_n = 37 * n) 
  (h3 : S_n = 999) : 
  n = 27 :=
by sorry

end arithmetic_sequence_n_l19_19737


namespace find_value_l19_19929

variable (x y : ℝ)

def conditions (x y : ℝ) :=
  y > 2 * x ∧ 2 * x > 0 ∧ (x / y + y / x = 8)

theorem find_value (h : conditions x y) : (x + y) / (x - y) = -Real.sqrt (5 / 3) :=
sorry

end find_value_l19_19929


namespace molecular_weight_NaClO_l19_19255

theorem molecular_weight_NaClO :
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  Na + Cl + O = 74.44 :=
by
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  sorry

end molecular_weight_NaClO_l19_19255


namespace correct_multiplier_l19_19694

theorem correct_multiplier
  (x : ℕ)
  (incorrect_multiplier : ℕ := 34)
  (difference : ℕ := 1215)
  (number_to_be_multiplied : ℕ := 135) :
  number_to_be_multiplied * x - number_to_be_multiplied * incorrect_multiplier = difference →
  x = 43 :=
  sorry

end correct_multiplier_l19_19694


namespace arithmetic_sum_S8_l19_19594

theorem arithmetic_sum_S8 (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, S (n + 1) - S n = S 1 - S 0)
  (h_positive : ∀ n, S n > 0)
  (h_S4 : S 4 = 10)
  (h_S12 : S 12 = 130) : 
  S 8 = 40 :=
sorry

end arithmetic_sum_S8_l19_19594


namespace largest_multiple_of_15_less_than_500_l19_19164

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19164


namespace eq_of_operation_l19_19218

theorem eq_of_operation {x : ℝ} (h : 60 + 5 * 12 / (x / 3) = 61) : x = 180 :=
by
  sorry

end eq_of_operation_l19_19218


namespace valid_unique_arrangement_count_l19_19393

def is_valid_grid (grid : list (list ℕ)) : Prop :=
  grid.length = 3 ∧ (∀ row, row ∈ grid -> row.length = 3) ∧
  (∀ n, n ∈ list.join grid → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ list.count (list.join grid) n = 1) ∧
  let row_sums := grid.map list.sum in
  let col_sums := list.map list.sum (list.transpose grid) in
  row_sums = [15, 15, 15] ∧ col_sums = [15, 15, 15]

noncomputable def number_of_valid_arrangements : ℕ :=
  72

theorem valid_unique_arrangement_count :
  ∃ (valid_grids : list (list (list ℕ))), (∀ g, g ∈ valid_grids -> is_valid_grid g) ∧ list.length valid_grids = number_of_valid_arrangements :=
begin
  use _, -- use a placeholder for the list of valid grids
  split,
  { intros g h,
    sorry }, -- proof of validity of each grid
  { sorry } -- proof of the count of valid grids
end

end valid_unique_arrangement_count_l19_19393


namespace number_of_integer_solutions_Q_is_one_l19_19626

def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 13 * x^2 + 3 * x - 19

theorem number_of_integer_solutions_Q_is_one : 
    (∃! x : ℤ, ∃ k : ℤ, Q x = k^2) := 
sorry

end number_of_integer_solutions_Q_is_one_l19_19626


namespace number_of_meetings_l19_19670

-- Definitions based on the given conditions
def track_circumference : ℕ := 300
def boy1_speed : ℕ := 7
def boy2_speed : ℕ := 3
def both_start_simultaneously := true

-- The theorem to prove
theorem number_of_meetings (h1 : track_circumference = 300) (h2 : boy1_speed = 7) (h3 : boy2_speed = 3) (h4 : both_start_simultaneously) : 
  ∃ n : ℕ, n = 1 := 
sorry

end number_of_meetings_l19_19670


namespace probability_odd_and_divisible_by_5_l19_19470

open Finset

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_div_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_odd_and_divisible_by_5 :
  let S := range (16) \ {0},
      odd_numbers := S.filter is_odd,
      num_ways := (odd_numbers.filter is_div_by_5).card * (odd_numbers.card - 1) in
  (num_ways / 2) / (S.card.choose 2) = (2 / 21 : ℚ) :=
by
  sorry

end probability_odd_and_divisible_by_5_l19_19470


namespace book_pages_l19_19852

theorem book_pages (total_pages : ℝ) : 
  (0.1 * total_pages + 0.25 * total_pages + 30 = 0.5 * total_pages) → 
  total_pages = 240 :=
by
  sorry

end book_pages_l19_19852


namespace min_m_n_sum_l19_19960

theorem min_m_n_sum (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_eq : 45 * m = n^3) : m + n = 90 :=
sorry

end min_m_n_sum_l19_19960


namespace largest_multiple_of_15_less_than_500_l19_19147

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19147


namespace volume_to_surface_area_ratio_l19_19870

-- Define the shape as described in the problem
structure Shape :=
(center_cube : ℕ)  -- Center cube
(surrounding_cubes : ℕ)  -- Surrounding cubes
(unit_volume : ℕ)  -- Volume of each unit cube
(unit_face_area : ℕ)  -- Surface area of each face of the unit cube

-- Conditions and definitions
def is_special_shape (s : Shape) : Prop :=
  s.center_cube = 1 ∧ s.surrounding_cubes = 7 ∧ s.unit_volume = 1 ∧ s.unit_face_area = 1

-- Theorem statement
theorem volume_to_surface_area_ratio (s : Shape) (h : is_special_shape s) : (s.center_cube + s.surrounding_cubes) * s.unit_volume / (s.surrounding_cubes * 5 * s.unit_face_area) = 8 / 35 :=
by
  sorry

end volume_to_surface_area_ratio_l19_19870


namespace largest_multiple_of_15_less_than_500_l19_19193

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19193


namespace find_p_q_l19_19604

theorem find_p_q : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x ^ 5 - x ^ 4 + x ^ 3 - p * x ^ 2 + q * x - 8)) → (p = -1 ∧ q = -10) :=
by
  sorry

end find_p_q_l19_19604


namespace solve_for_2a_2d_l19_19423

noncomputable def f (a b c d x : ℝ) : ℝ :=
  (2 * a * x + b) / (c * x + 2 * d)

theorem solve_for_2a_2d (a b c d : ℝ) (habcd_ne_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h : ∀ x, f a b c d (f a b c d x) = x) : 2 * a + 2 * d = 0 :=
sorry

end solve_for_2a_2d_l19_19423


namespace smallest_integer_remainder_conditions_l19_19506

theorem smallest_integer_remainder_conditions :
  ∃ b : ℕ, (b % 3 = 0) ∧ (b % 4 = 2) ∧ (b % 5 = 3) ∧ (∀ n : ℕ, (n % 3 = 0) ∧ (n % 4 = 2) ∧ (n % 5 = 3) → b ≤ n) :=
sorry

end smallest_integer_remainder_conditions_l19_19506


namespace largest_whole_number_satisfying_inequality_l19_19217

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end largest_whole_number_satisfying_inequality_l19_19217


namespace original_cost_of_car_l19_19074

theorem original_cost_of_car (C : ℝ)
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (h1 : repairs_cost = 14000)
  (h2 : selling_price = 72900)
  (h3 : profit_percent = 17.580645161290324)
  (h4 : profit_percent = ((selling_price - (C + repairs_cost)) / C) * 100) :
  C = 50075 := 
sorry

end original_cost_of_car_l19_19074


namespace polynomial_expansion_correct_l19_19008

def polynomial1 (z : ℤ) : ℤ := 3 * z^3 + 4 * z^2 - 5
def polynomial2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2
def expandedPolynomial (z : ℤ) : ℤ := 12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10

theorem polynomial_expansion_correct (z : ℤ) :
  (polynomial1 z) * (polynomial2 z) = expandedPolynomial z :=
by sorry

end polynomial_expansion_correct_l19_19008


namespace largest_divisor_of_10000_not_dividing_9999_l19_19327

theorem largest_divisor_of_10000_not_dividing_9999 : ∃ d, d ∣ 10000 ∧ ¬ (d ∣ 9999) ∧ ∀ y, (y ∣ 10000 ∧ ¬ (y ∣ 9999)) → y ≤ d := 
by
  sorry

end largest_divisor_of_10000_not_dividing_9999_l19_19327


namespace volume_of_pyramid_l19_19639

/--
Rectangle ABCD is the base of pyramid PABCD. Let AB = 10, BC = 6, PA is perpendicular to AB, and PB = 20. 
If PA makes an angle θ = 30° with the diagonal AC of the base, prove the volume of the pyramid PABCD is 200 cubic units.
-/
theorem volume_of_pyramid (AB BC PB : ℝ) (θ : ℝ) (hAB : AB = 10) (hBC : BC = 6)
  (hPB : PB = 20) (hθ : θ = 30) (PA_is_perpendicular_to_AB : true) (PA_makes_angle_with_AC : true) : 
  ∃ V, V = 1 / 3 * (AB * BC) * 10 ∧ V = 200 := 
by
  exists 1 / 3 * (AB * BC) * 10
  sorry

end volume_of_pyramid_l19_19639


namespace total_distance_journey_l19_19763

theorem total_distance_journey :
  let south := 40
  let east := south + 20
  let north := 2 * east
  (south + east + north) = 220 :=
by
  sorry

end total_distance_journey_l19_19763


namespace focus_of_parabola_l19_19726

theorem focus_of_parabola : (∃ p : ℝ × ℝ, p = (-1, 35/12)) :=
by
  sorry

end focus_of_parabola_l19_19726


namespace distribution_of_collection_items_l19_19766

-- Declaring the collections
structure Collection where
  stickers : Nat
  baseball_cards : Nat
  keychains : Nat
  stamps : Nat

-- Defining the individual collections based on the conditions
def Karl : Collection := { stickers := 25, baseball_cards := 15, keychains := 5, stamps := 10 }
def Ryan : Collection := { stickers := Karl.stickers + 20, baseball_cards := Karl.baseball_cards - 10, keychains := Karl.keychains + 2, stamps := Karl.stamps }
def Ben_scenario1 : Collection := { stickers := Ryan.stickers - 10, baseball_cards := (Ryan.baseball_cards / 2), keychains := Karl.keychains * 2, stamps := Karl.stamps + 5 }

-- Total number of items in the collection
def total_items_scenario1 :=
  Karl.stickers + Karl.baseball_cards + Karl.keychains + Karl.stamps +
  Ryan.stickers + Ryan.baseball_cards + Ryan.keychains + Ryan.stamps +
  Ben_scenario1.stickers + Ben_scenario1.baseball_cards + Ben_scenario1.keychains + Ben_scenario1.stamps

-- The proof statement
theorem distribution_of_collection_items :
  total_items_scenario1 = 184 ∧ total_items_scenario1 % 4 = 0 → (184 / 4 = 46) := 
by
  sorry

end distribution_of_collection_items_l19_19766


namespace volume_of_first_bottle_l19_19876

theorem volume_of_first_bottle (V_2 V_3 : ℕ) (V_total : ℕ):
  V_2 = 750 ∧ V_3 = 250 ∧ V_total = 3 * 1000 →
  (V_total - V_2 - V_3) / 1000 = 2 :=
by
  sorry

end volume_of_first_bottle_l19_19876


namespace gcd_of_12012_18018_l19_19580

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l19_19580


namespace smallest_b_for_no_real_root_l19_19674

theorem smallest_b_for_no_real_root :
  ∃ b : ℤ, (b < 8 ∧ b > -8) ∧ (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ -6) ∧ (b = -7) :=
by
  sorry

end smallest_b_for_no_real_root_l19_19674


namespace largest_multiple_of_15_less_than_500_l19_19199

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19199


namespace expected_value_twelve_sided_die_l19_19369

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_twelve_sided_die_l19_19369


namespace sum_three_numbers_is_247_l19_19839

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l19_19839


namespace sum_of_numbers_l19_19827

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19827


namespace tunnel_length_l19_19456

theorem tunnel_length (L L_1 L_2 v v_new t t_new : ℝ) (H1: L_1 = 6) (H2: L_2 = 12) 
  (H3: v_new = 0.8 * v) (H4: t = (L + L_1) / v) (H5: t_new = 1.5 * t)
  (H6: t_new = (L + L_2) / v_new) : 
  L = 24 :=
by
  sorry

end tunnel_length_l19_19456


namespace blackjack_payout_ratio_l19_19520

theorem blackjack_payout_ratio (total_payout original_bet : ℝ) (h1 : total_payout = 60) (h2 : original_bet = 40):
  total_payout - original_bet = (1 / 2) * original_bet :=
by
  sorry

end blackjack_payout_ratio_l19_19520


namespace geometric_sequence_product_l19_19054

-- Defining a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given data
def a := fun n => (4 : ℝ) * (2 : ℝ)^(n-4)

-- Main proof problem
theorem geometric_sequence_product (a : ℕ → ℝ) (h : is_geometric_sequence a) (h₁ : a 4 = 4) :
  a 2 * a 6 = 16 :=
by
  sorry

end geometric_sequence_product_l19_19054


namespace largest_multiple_of_15_less_than_500_l19_19200

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19200


namespace roots_ratio_quadratic_eq_l19_19014

theorem roots_ratio_quadratic_eq {k r s : ℝ} 
(h_eq : ∃ a b : ℝ, a * r = b * s) 
(ratio_3_2 : ∃ t : ℝ, r = 3 * t ∧ s = 2 * t) 
(eqn : r + s = -10 ∧ r * s = k) : 
k = 24 := 
sorry

end roots_ratio_quadratic_eq_l19_19014


namespace largest_multiple_of_15_below_500_l19_19174

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19174


namespace border_collie_catches_ball_in_32_seconds_l19_19545

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l19_19545


namespace sum_of_ratios_l19_19809

theorem sum_of_ratios (a b c : ℤ) (h : (a * a : ℚ) / (b * b) = 32 / 63) : a + b + c = 39 :=
sorry

end sum_of_ratios_l19_19809


namespace expected_value_of_12_sided_die_is_6_5_l19_19355

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_l19_19355


namespace find_cd_l19_19721

theorem find_cd : 
  (∀ x : ℝ, (4 * x - 3) / (x^2 - 3 * x - 18) = ((7 / 3) / (x - 6)) + ((5 / 3) / (x + 3))) :=
by
  intro x
  have h : x^2 - 3 * x - 18 = (x - 6) * (x + 3) := by
    sorry
  rw [h]
  sorry

end find_cd_l19_19721


namespace find_b6b8_l19_19051

-- Define sequences {a_n} (arithmetic sequence) and {b_n} (geometric sequence)
variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Given conditions
axiom h1 : ∀ n m : ℕ, a m = a n + (m - n) * (a (n + 1) - a n) -- Arithmetic sequence property
axiom h2 : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0
axiom h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1 -- Geometric sequence property
axiom h4 : b 7 = a 7
axiom h5 : ∀ n : ℕ, b n > 0                 -- Assuming b_n has positive terms
axiom h6 : ∀ n : ℕ, a n > 0                 -- Positive terms in sequence a_n

-- Proof objective
theorem find_b6b8 : b 6 * b 8 = 16 :=
by sorry

end find_b6b8_l19_19051


namespace arithmetic_progression_root_difference_l19_19253

theorem arithmetic_progression_root_difference (a b c : ℚ) (h : 81 * a * a * a - 225 * a * a + 164 * a - 30 = 0)
  (hb : b = 5/3) (hprog : ∃ d : ℚ, a = b - d ∧ c = b + d) :
  c - a = 5 / 9 :=
sorry

end arithmetic_progression_root_difference_l19_19253


namespace car_R_speed_l19_19223

theorem car_R_speed (v : ℝ) (h1 : ∀ t_R t_P : ℝ, t_R * v = 800 ∧ t_P * (v + 10) = 800) (h2 : ∀ t_R t_P : ℝ, t_P + 2 = t_R) :
  v = 50 := by
  sorry

end car_R_speed_l19_19223


namespace contradiction_method_assumption_l19_19092

-- Definitions for three consecutive positive integers
variables {a b c : ℕ}

-- Definitions for the proposition and its negation
def consecutive_integers (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1
def at_least_one_divisible_by_2 (a b c : ℕ) : Prop := a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0
def all_not_divisible_by_2 (a b c : ℕ) : Prop := a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem contradiction_method_assumption (a b c : ℕ) (h : consecutive_integers a b c) :
  (¬ at_least_one_divisible_by_2 a b c) ↔ all_not_divisible_by_2 a b c :=
by sorry

end contradiction_method_assumption_l19_19092


namespace maximum_value_l19_19773

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y^2 + x^3 * y^3 + x^2 * y^4 + x * y^5

theorem maximum_value (x y : ℝ) (h : x + y = 5) : maxValue x y h ≤ 625 / 4 :=
sorry

end maximum_value_l19_19773


namespace smallest_bdf_l19_19936

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l19_19936


namespace largest_multiple_of_15_less_than_500_l19_19191

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19191


namespace ratio_of_speeds_l19_19511

theorem ratio_of_speeds (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 * D = 2 * (10 * H) :=
by
  sorry

example (D H : ℕ) (h1 : D = 2 * H) (h2 : 10 * D = 20 * H) :
  10 = 10 :=
by
  sorry

end ratio_of_speeds_l19_19511


namespace minimum_positive_temperatures_announced_l19_19377

theorem minimum_positive_temperatures_announced (x y : ℕ) :
  x * (x - 1) = 110 →
  y * (y - 1) + (x - y) * (x - y - 1) = 54 →
  (∀ z : ℕ, z * (z - 1) + (x - z) * (x - z - 1) = 54 → y ≤ z) →
  y = 4 :=
by
  sorry

end minimum_positive_temperatures_announced_l19_19377


namespace greatest_value_x_l19_19969

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l19_19969


namespace complex_quadrant_l19_19021

open Complex

theorem complex_quadrant 
  (z : ℂ) 
  (h : (1 - I) ^ 2 / z = 1 + I) :
  z = -1 - I :=
by
  sorry

end complex_quadrant_l19_19021


namespace sum_of_three_numbers_l19_19831

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l19_19831


namespace megatech_budget_allocation_l19_19691

theorem megatech_budget_allocation :
  let microphotonics := 14
  let food_additives := 10
  let gmo := 24
  let industrial_lubricants := 8
  let basic_astrophysics := 25
  microphotonics + food_additives + gmo + industrial_lubricants + basic_astrophysics = 81 →
  100 - 81 = 19 :=
by
  intros
  -- We are given the sums already, so directly calculate the remaining percentage.
  sorry

end megatech_budget_allocation_l19_19691


namespace monthly_payment_l19_19002

noncomputable def house_price := 280
noncomputable def deposit := 40
noncomputable def mortgage_years := 10
noncomputable def months_per_year := 12

theorem monthly_payment (house_price deposit : ℕ) (mortgage_years months_per_year : ℕ) :
  (house_price - deposit) / mortgage_years / months_per_year = 2 :=
by
  sorry

end monthly_payment_l19_19002


namespace find_sum_of_abcd_l19_19565

theorem find_sum_of_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) :
  a + b + c + d = -26 / 3 :=
sorry

end find_sum_of_abcd_l19_19565


namespace factoring_expression_l19_19804

theorem factoring_expression (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = -(x - y) * (a + b - c) :=
by
  sorry

end factoring_expression_l19_19804


namespace net_difference_in_expenditure_l19_19997

variable (P Q : ℝ)
-- Condition 1: Price increased by 25%
def new_price (P : ℝ) : ℝ := P * 1.25

-- Condition 2: Purchased 72% of the originally required amount
def new_quantity (Q : ℝ) : ℝ := Q * 0.72

-- Definition of original expenditure
def original_expenditure (P Q : ℝ) : ℝ := P * Q

-- Definition of new expenditure
def new_expenditure (P Q : ℝ) : ℝ := new_price P * new_quantity Q

-- Statement of the proof problem.
theorem net_difference_in_expenditure
  (P Q : ℝ) : new_expenditure P Q - original_expenditure P Q = -0.1 * original_expenditure P Q := 
by
  sorry

end net_difference_in_expenditure_l19_19997


namespace men_work_equivalence_l19_19314

theorem men_work_equivalence : 
  ∀ (M : ℕ) (m w : ℕ),
  (3 * w = 2 * m) ∧ 
  (M * 21 * 8 * m = 21 * 60 * 3 * w) →
  M = 15 := by
  intro M m w
  intro h
  sorry

end men_work_equivalence_l19_19314


namespace cos_C_value_l19_19434

theorem cos_C_value (a b c : ℝ) (A B C : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : 
  Real.cos C = 7 / 25 :=
  sorry

end cos_C_value_l19_19434


namespace largest_multiple_of_15_less_than_500_l19_19110

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19110


namespace largest_multiple_of_15_less_than_500_l19_19103

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19103


namespace can_construct_length_one_l19_19408

noncomputable def possible_to_construct_length_one_by_folding (n : ℕ) : Prop :=
  ∃ k ≤ 10, ∃ (segment_constructed : ℝ), segment_constructed = 1

theorem can_construct_length_one : possible_to_construct_length_one_by_folding 2016 :=
by sorry

end can_construct_length_one_l19_19408


namespace solve_for_q_l19_19651

theorem solve_for_q (k r q : ℕ) (h1 : 4 / 5 = k / 90) (h2 : 4 / 5 = (k + r) / 105) (h3 : 4 / 5 = (q - r) / 150) : q = 132 := 
  sorry

end solve_for_q_l19_19651


namespace not_quadratic_eq3_l19_19220

-- Define the equations as functions or premises
def eq1 (x : ℝ) := 9 * x^2 = 7 * x
def eq2 (y : ℝ) := abs (y^2) = 8
def eq3 (y : ℝ) := 3 * y * (y - 1) = y * (3 * y + 1)
def eq4 (x : ℝ) := abs 2 * (x^2 + 1) = abs 10

-- Define what it means to be a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x = (a * x^2 + b * x + c = 0)

-- Prove that eq3 is not a quadratic equation
theorem not_quadratic_eq3 : ¬ is_quadratic eq3 :=
sorry

end not_quadratic_eq3_l19_19220


namespace smallest_value_a1_l19_19063

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n-1) - 2 * n

theorem smallest_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : seq a) : 
  a 1 ≥ 13 / 18 :=
sorry

end smallest_value_a1_l19_19063


namespace polynomial_degree_bound_l19_19625

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) 
  (hm_pos : 0 < m)
  (hn_pos : 0 < n)
  (hk_pos : 2 ≤ k)
  (hP_odd : ∀ i, P.coeff i % 2 = 1) 
  (h_div : (X - 1) ^ m ∣ P)
  (hm_bound : m ≥ 2 ^ k) :
  n ≥ 2 ^ (k + 1) - 1 := sorry

end polynomial_degree_bound_l19_19625


namespace fraction_to_decimal_l19_19851

theorem fraction_to_decimal : (7 : Rat) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l19_19851


namespace largest_multiple_15_under_500_l19_19118

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19118


namespace midpoint_coordinate_sum_l19_19331

theorem midpoint_coordinate_sum
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 10)
  (h2 : y1 = 3)
  (h3 : x2 = 4)
  (h4 : y2 = -3) :
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym =  7 := by
  sorry

end midpoint_coordinate_sum_l19_19331


namespace angle_y_in_triangle_l19_19880

theorem angle_y_in_triangle (y : ℝ) (h1 : ∀ a b c : ℝ, a + b + c = 180) (h2 : 3 * y + y + 40 = 180) : y = 35 :=
sorry

end angle_y_in_triangle_l19_19880


namespace largest_multiple_of_15_less_than_500_l19_19183

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19183


namespace smallest_n_divisible_by_2009_l19_19005

theorem smallest_n_divisible_by_2009 : ∃ n : ℕ, n > 1 ∧ (n^2 * (n - 1)) % 2009 = 0 ∧ (∀ m : ℕ, m > 1 → (m^2 * (m - 1)) % 2009 = 0 → m ≥ n) :=
by
  sorry

end smallest_n_divisible_by_2009_l19_19005


namespace correct_calculation_l19_19857

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := 
by 
  sorry

end correct_calculation_l19_19857


namespace jon_initial_fastball_speed_l19_19764

theorem jon_initial_fastball_speed 
  (S : ℝ) -- Condition: Jon's initial fastball speed \( S \)
  (h1 : ∀ t : ℕ, t = 4 * 4)  -- Condition: Training time is 4 times for 4 weeks each
  (h2 : ∀ w : ℕ, w = 16)  -- Condition: Total weeks of training (4*4=16)
  (h3 : ∀ g : ℝ, g = 1)  -- Condition: Gains 1 mph per week
  (h4 : ∃ S_new : ℝ, S_new = (S + 16) ∧ S_new = 1.2 * S) -- Condition: Speed increases by 20%
  : S = 80 := 
sorry

end jon_initial_fastball_speed_l19_19764


namespace dreamCarCost_l19_19867

-- Definitions based on given conditions
def monthlyEarnings : ℕ := 4000
def monthlySavings : ℕ := 500
def totalEarnings : ℕ := 360000

-- Theorem stating the desired result
theorem dreamCarCost :
  (totalEarnings / monthlyEarnings) * monthlySavings = 45000 :=
by
  sorry

end dreamCarCost_l19_19867


namespace triangle_formation_and_acuteness_l19_19744

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end triangle_formation_and_acuteness_l19_19744


namespace gcd_12012_18018_l19_19573

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19573


namespace geometric_sequence_log_sum_l19_19279

open Real

theorem geometric_sequence_log_sum {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ m n, m + 1 = n → a m * a n = a (m - 1) * a (n + 1) )
  (h3 : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) :
  ( ∑ i in Finset.range 20, log (a (i+1)) ) = 50 :=
by
  -- The detailed proof is omitted here.
  sorry

end geometric_sequence_log_sum_l19_19279


namespace sqrt_720_l19_19646

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l19_19646


namespace largest_multiple_of_15_less_than_500_l19_19108

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19108


namespace conversion_correct_l19_19559

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l19_19559


namespace number_exceeds_fraction_l19_19222

theorem number_exceeds_fraction (x : ℝ) (hx : x = 0.45 * x + 1000) : x = 1818.18 := 
by
  sorry

end number_exceeds_fraction_l19_19222


namespace doughnuts_per_person_l19_19462

-- Define the number of dozens bought by Samuel
def samuel_dozens : ℕ := 2

-- Define the number of dozens bought by Cathy
def cathy_dozens : ℕ := 3

-- Define the number of doughnuts in one dozen
def dozen : ℕ := 12

-- Define the total number of people
def total_people : ℕ := 10

-- Prove that each person receives 6 doughnuts
theorem doughnuts_per_person :
  ((samuel_dozens * dozen) + (cathy_dozens * dozen)) / total_people = 6 :=
by
  sorry

end doughnuts_per_person_l19_19462


namespace michael_total_fish_l19_19633

-- Definitions based on conditions
def michael_original_fish : ℕ := 31
def ben_fish_given : ℕ := 18

-- Theorem to prove the total number of fish Michael has now
theorem michael_total_fish : (michael_original_fish + ben_fish_given) = 49 :=
by sorry

end michael_total_fish_l19_19633


namespace jade_more_transactions_l19_19454

theorem jade_more_transactions 
    (mabel_transactions : ℕ) 
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + (mabel_transactions / 10))
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = 82) :
    jade_transactions - cal_transactions = 16 :=
sorry

end jade_more_transactions_l19_19454


namespace greatest_prime_factor_3_pow_8_add_6_pow_7_l19_19500
noncomputable theory

open Nat

theorem greatest_prime_factor_3_pow_8_add_6_pow_7 : 
  ∃ p : ℕ, prime p ∧ (∀ q : ℕ, q ∣ (3^8 + 6^7) → prime q → q ≤ p) ∧ p = 131 := 
by
  sorry

end greatest_prime_factor_3_pow_8_add_6_pow_7_l19_19500


namespace find_z_add_inv_y_l19_19468

theorem find_z_add_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) : z + 1 / y = 5 / 27 := by
  sorry

end find_z_add_inv_y_l19_19468


namespace card_pair_probability_l19_19339

theorem card_pair_probability (initial_deck_size removed_pairs remaining_deck_size : ℕ)
(numbers_in_deck cards_per_number pairs_removed : ℕ)
(h₁ : numbers_in_deck = 12)
(h₂ : cards_per_number = 4)
(h₃ : pairs_removed = 2)
(h₄ : initial_deck_size = numbers_in_deck * cards_per_number)
(h₅ : removed_pairs = pairs_removed * 2)
(h₆ : remaining_deck_size = initial_deck_size - removed_pairs)
(h₇ : remaining_deck_size = 44) :
  let total_ways := Nat.choose remaining_deck_size 2,
      full_set_ways := 10 * Nat.choose cards_per_number 2,
      partial_set_ways := 2 * Nat.choose pairs_removed 2,
      favorable_ways := full_set_ways + partial_set_ways,
      probability := favorable_ways / total_ways in
  let reduced_prob := probability.num / probability.denom,
      m := reduced_prob.num,
      n := reduced_prob.denom in
  m + n = 504 := sorry

end card_pair_probability_l19_19339


namespace greatest_possible_x_l19_19968

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l19_19968


namespace product_of_2020_numbers_even_l19_19818

theorem product_of_2020_numbers_even (a : ℕ → ℕ) 
  (h : (Finset.sum (Finset.range 2020) a) % 2 = 1) : 
  (Finset.prod (Finset.range 2020) a) % 2 = 0 :=
sorry

end product_of_2020_numbers_even_l19_19818


namespace max_levels_passable_prob_pass_three_levels_l19_19858

-- Define the condition for passing a level
def passes_level (n : ℕ) (sum : ℕ) : Prop :=
  sum > 2^n

-- Define the maximum sum possible for n dice rolls
def max_sum (n : ℕ) : ℕ :=
  6 * n

-- Define the probability of passing the n-th level
def prob_passing_level (n : ℕ) : ℚ :=
  if n = 1 then 2/3
  else if n = 2 then 5/6
  else if n = 3 then 20/27
  else 0 

-- Combine probabilities for passing the first three levels
def prob_passing_three_levels : ℚ :=
  (2/3) * (5/6) * (20/27)

-- Theorem statement for the maximum number of levels passable
theorem max_levels_passable : 4 = 4 :=
sorry

-- Theorem statement for the probability of passing the first three levels
theorem prob_pass_three_levels : prob_passing_three_levels = 100 / 243 :=
sorry

end max_levels_passable_prob_pass_three_levels_l19_19858


namespace solveSystem1_solveFractionalEq_l19_19798

-- Definition: system of linear equations
def system1 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ x - 4 * y = 9

-- Theorem: solution to the system of equations
theorem solveSystem1 : ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = -1 :=
by
  sorry
  
-- Definition: fractional equation
def fractionalEq (x : ℝ) : Prop :=
  (x + 2) / (x^2 - 2 * x + 1) + 3 / (x - 1) = 0

-- Theorem: solution to the fractional equation
theorem solveFractionalEq : ∃ x : ℝ, fractionalEq x ∧ x = 1 / 4 :=
by
  sorry

end solveSystem1_solveFractionalEq_l19_19798


namespace total_flowers_in_vases_l19_19979

theorem total_flowers_in_vases :
  let vase_count := 5
  let flowers_per_vase_4 := 5
  let flowers_per_vase_1 := 6
  let vases_with_5_flowers := 4
  let vases_with_6_flowers := 1
  (4 * 5 + 1 * 6 = 26) := by
  let total_flowers := 4 * 5 + 1 * 6
  show total_flowers = 26
  sorry

end total_flowers_in_vases_l19_19979


namespace profit_ratio_l19_19484

theorem profit_ratio (I_P I_Q : ℝ) (t_P t_Q : ℕ) 
  (h1 : I_P / I_Q = 7 / 5)
  (h2 : t_P = 5)
  (h3 : t_Q = 14) : 
  (I_P * t_P) / (I_Q * t_Q) = 1 / 2 :=
by
  sorry

end profit_ratio_l19_19484


namespace union_A_B_l19_19743

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem union_A_B : A ∪ B = Ico (-1 : ℝ) 2 := by
  sorry

end union_A_B_l19_19743


namespace range_of_a_l19_19004

theorem range_of_a (a : ℝ) (h₀ : a > 0) : (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 :=
sorry

end range_of_a_l19_19004


namespace operation_multiplication_in_P_l19_19486

-- Define the set P
def P : Set ℕ := {n | ∃ k : ℕ, n = k^2}

-- Define the operation "*" as multiplication within the set P
def operation (a b : ℕ) : ℕ := a * b

-- Define the property to be proved
theorem operation_multiplication_in_P (a b : ℕ)
  (ha : a ∈ P) (hb : b ∈ P) : operation a b ∈ P :=
sorry

end operation_multiplication_in_P_l19_19486


namespace largest_multiple_of_15_less_than_500_l19_19203

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19203


namespace min_value_x3_y2_z_w2_l19_19769

theorem min_value_x3_y2_z_w2 (x y z w : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)
  (h : (1/x) + (1/y) + (1/z) + (1/w) = 8) : x^3 * y^2 * z * w^2 ≥ 1/432 :=
by
  sorry

end min_value_x3_y2_z_w2_l19_19769


namespace range_of_function_l19_19601

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  let y := (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin (2 * x)) - 1 / 2
  in y ∈ Set.Ioc (-1 / 2) 1 :=
sorry

end range_of_function_l19_19601


namespace simplify_and_evaluate_l19_19956

variable (a : ℝ)
variable (ha : a = Real.sqrt 3 - 1)

theorem simplify_and_evaluate : 
  (1 + 3 / (a - 2)) / ((a^2 + 2 * a + 1) / (a - 2)) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_l19_19956


namespace minimum_number_of_groups_l19_19350

def total_students : ℕ := 30
def max_students_per_group : ℕ := 12
def largest_divisor (n : ℕ) (m : ℕ) : ℕ := 
  list.maximum (list.filter (λ d, d ∣ n) (list.range_succ m)).get_or_else 1

theorem minimum_number_of_groups : ∃ k : ℕ, k = total_students / (largest_divisor total_students max_students_per_group) ∧ k = 3 :=
by
  sorry

end minimum_number_of_groups_l19_19350


namespace pounds_of_fish_to_ship_l19_19630

theorem pounds_of_fish_to_ship (crates_weight : ℕ) (cost_per_crate : ℝ) (total_cost : ℝ) :
  crates_weight = 30 → cost_per_crate = 1.5 → total_cost = 27 → 
  (total_cost / cost_per_crate) * crates_weight = 540 :=
by
  intros h1 h2 h3
  sorry

end pounds_of_fish_to_ship_l19_19630


namespace larry_spent_on_lunch_l19_19059

noncomputable def starting_amount : ℕ := 22
noncomputable def ending_amount : ℕ := 15
noncomputable def amount_given_to_brother : ℕ := 2

theorem larry_spent_on_lunch : 
  (starting_amount - (ending_amount + amount_given_to_brother)) = 5 :=
by
  -- The conditions and the proof structure would be elaborated here
  sorry

end larry_spent_on_lunch_l19_19059


namespace largest_multiple_of_15_less_than_500_l19_19133

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19133


namespace largest_multiple_of_15_less_than_500_l19_19205

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19205


namespace geometric_progression_first_term_one_l19_19499

theorem geometric_progression_first_term_one (a r : ℝ) (gp : ℕ → ℝ)
  (h_gp : ∀ n, gp n = a * r^(n - 1))
  (h_product_in_gp : ∀ i j, ∃ k, gp i * gp j = gp k) :
  a = 1 := 
sorry

end geometric_progression_first_term_one_l19_19499


namespace expected_value_twelve_sided_die_l19_19368

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_twelve_sided_die_l19_19368


namespace largest_multiple_of_15_under_500_l19_19208

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19208


namespace ramu_repair_cost_l19_19784

theorem ramu_repair_cost
  (initial_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (repair_cost : ℝ)
  (h1 : initial_cost = 42000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017 / 100)
  (h4 : selling_price = initial_cost + repair_cost + profit_percent * (initial_cost + repair_cost)) :
  repair_cost = 15000 :=
by
  sorry

end ramu_repair_cost_l19_19784


namespace non_periodic_decimal_l19_19410

variable {a : ℕ → ℕ}

-- Condition definitions
def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

def constraint (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ 10 * a n

-- Theorem statement
theorem non_periodic_decimal (a : ℕ → ℕ) 
  (h_inc : is_increasing_sequence a) 
  (h_constraint : constraint a) : 
  ¬ (∃ T : ℕ, ∀ n : ℕ, a (n + T) = a n) :=
sorry

end non_periodic_decimal_l19_19410


namespace exponent_division_example_l19_19496

theorem exponent_division_example : ((3^2)^4) / (3^2) = 729 := by
  sorry

end exponent_division_example_l19_19496


namespace valid_paths_count_l19_19551

-- Define the grid and the prohibited segments
def grid (height width : ℕ) : Type :=
  { p : ℕ × ℕ // p.1 ≤ height ∧ p.2 ≤ width }

def isForbiddenSegment1 (p : ℕ × ℕ) : Prop :=
  p.2 = 3 ∧ 1 ≤ p.1 ∧ p.1 ≤ 3

def isForbiddenSegment2 (p : ℕ × ℕ) : Prop :=
  p.2 = 4 ∧ 2 ≤ p.1 ∧ p.1 ≤ 5

-- Statement of the problem
theorem valid_paths_count : 
  let height := 5 
  let width  := 8 in
  let A := (0, 0) 
  let B := (height, width) 
  count_valid_paths A B height width isForbiddenSegment1 isForbiddenSegment2 = 838 := sorry

end valid_paths_count_l19_19551


namespace luggage_between_340_and_420_l19_19260

noncomputable def luggage_normal_distribution : ProbabilityDistribution ℝ :=
  NormalDistr.mk 380 (20 ^ 2)

theorem luggage_between_340_and_420 :
  Pr(open_interval (340 : ℝ) (420 : ℝ)) luggage_normal_distribution = 0.95 :=
sorry

end luggage_between_340_and_420_l19_19260


namespace binomial_coefficient_9_5_l19_19000

theorem binomial_coefficient_9_5 : nat.choose 9 5 = 126 :=
by
  sorry

end binomial_coefficient_9_5_l19_19000


namespace largest_multiple_of_15_less_than_500_l19_19141

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19141


namespace find_f_2_l19_19036

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end find_f_2_l19_19036


namespace price_of_each_rose_l19_19541

def number_of_roses_started (roses : ℕ) : Prop := roses = 9
def number_of_roses_left (roses : ℕ) : Prop := roses = 4
def amount_earned (money : ℕ) : Prop := money = 35
def selling_price_per_rose (price : ℕ) : Prop := price = 7

theorem price_of_each_rose 
  (initial_roses sold_roses left_roses total_money price_per_rose : ℕ)
  (h1 : number_of_roses_started initial_roses)
  (h2 : number_of_roses_left left_roses)
  (h3 : amount_earned total_money)
  (h4 : initial_roses - left_roses = sold_roses)
  (h5 : total_money / sold_roses = price_per_rose) :
  selling_price_per_rose price_per_rose := 
by
  sorry

end price_of_each_rose_l19_19541


namespace inequality_of_fractions_l19_19894

theorem inequality_of_fractions
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c d : ℝ) (h3 : c < d) (h4 : d < 0)
  (e : ℝ) (h5 : e < 0) :
  (e / ((a - c)^2)) > (e / ((b - d)^2)) :=
by
  sorry

end inequality_of_fractions_l19_19894


namespace largest_multiple_of_15_less_than_500_l19_19131

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19131


namespace expected_value_of_twelve_sided_die_l19_19354

theorem expected_value_of_twelve_sided_die : 
  let faces := (List.range (12+1)).tail in
  let E := (1 / 12) * (faces.sum) in
  E = 6.5 :=
by
  let faces := (List.range (12+1)).tail
  let E := (1 / 12) * (faces.sum)
  have : faces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] := by
    unfold faces
    simp [List.range, List.tail]
  have : faces.sum = 78 := by
    simp [this]
    norm_num
  have : E = (1 / 12) * 78 := by
    unfold E
    congr
  have : E = 6.5 := by
    simp [this]
    norm_num
  exact this

end expected_value_of_twelve_sided_die_l19_19354


namespace largest_multiple_of_15_less_than_500_is_495_l19_19160

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19160


namespace angle_C_measure_ratio_inequality_l19_19745

open Real

variables (A B C a b c : ℝ)

-- Assumptions
variable (ABC_is_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variable (sin_condition : sin (2 * C - π / 2) = 1/2)
variable (inequality_condition : a^2 + b^2 < c^2)

theorem angle_C_measure :
  0 < C ∧ C < π ∧ C = 2 * π / 3 := sorry

theorem ratio_inequality :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end angle_C_measure_ratio_inequality_l19_19745


namespace find_m_n_l19_19735

theorem find_m_n (m n x1 x2 : ℕ) (hm : 0 < m) (hn : 0 < n) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (h_eq : x1 * x2 = m + n) (h_sum : x1 + x2 = m * n) :
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) := 
sorry

end find_m_n_l19_19735


namespace largest_multiple_of_15_less_than_500_is_495_l19_19153

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19153


namespace range_of_m_l19_19903

def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

def has_local_extremum_at (a b x : ℝ) : Prop :=
  f_prime a b x = 0 ∧ f a b x = 0

def h (a b m x : ℝ) : ℝ := f a b x - m + 1

theorem range_of_m (a b m : ℝ) :
  (has_local_extremum_at 2 9 (-1) ∧
   ∀ x, f 2 9 x = x^3 + 6 * x^2 + 9 * x + 4) →
  (∀ x, (x^3 + 6 * x^2 + 9 * x + 4 - m + 1 = 0) → 
  1 < m ∧ m < 5) := 
sorry

end range_of_m_l19_19903


namespace bread_carriers_l19_19493

-- Definitions for the number of men, women, and children
variables (m w c : ℕ)

-- Conditions from the problem
def total_people := m + w + c = 12
def total_bread := 8 * m + 2 * w + c = 48

-- Theorem to prove the correct number of men, women, and children
theorem bread_carriers (h1 : total_people m w c) (h2 : total_bread m w c) : 
  m = 5 ∧ w = 1 ∧ c = 6 :=
sorry

end bread_carriers_l19_19493


namespace sum_of_numbers_l19_19835

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19835


namespace total_oranges_l19_19093

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l19_19093


namespace polar_eq_to_cartesian_l19_19379

-- Define the conditions
def polar_to_cartesian_eq (ρ : ℝ) : Prop :=
  ρ = 2 → (∃ x y : ℝ, x^2 + y^2 = ρ^2)

-- State the main theorem/proof problem
theorem polar_eq_to_cartesian : polar_to_cartesian_eq 2 :=
by
  -- Proof sketch:
  --   Given ρ = 2
  --   We have ρ^2 = 4
  --   By converting to Cartesian coordinates: x^2 + y^2 = ρ^2
  --   Result: x^2 + y^2 = 4
  sorry

end polar_eq_to_cartesian_l19_19379


namespace expected_value_of_twelve_sided_die_l19_19374

noncomputable def expected_value_twelve_sided_die : ℝ :=
  let sides := 12
  let sum_faces := finset.sum (finset.range (sides + 1)) (λ k, k)
  let expected_value := sum_faces / sides
  expected_value

theorem expected_value_of_twelve_sided_die : expected_value_twelve_sided_die = 6.5 :=
by sorry

end expected_value_of_twelve_sided_die_l19_19374


namespace intersection_coordinates_l19_19803

theorem intersection_coordinates (x y : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : y = x + 1) : 
  x = 2 ∧ y = 3 := 
by 
  sorry

end intersection_coordinates_l19_19803


namespace triangle_is_obtuse_l19_19304

-- Define the sides of the triangle with the given ratio
def a (x : ℝ) := 3 * x
def b (x : ℝ) := 4 * x
def c (x : ℝ) := 6 * x

-- The theorem statement
theorem triangle_is_obtuse (x : ℝ) (hx : 0 < x) : 
  (a x)^2 + (b x)^2 < (c x)^2 :=
by
  sorry

end triangle_is_obtuse_l19_19304


namespace smallest_c_is_52_l19_19981

def seq (n : ℕ) : ℤ := -103 + (n:ℤ) * 2

theorem smallest_c_is_52 :
  ∃ c : ℕ, 
  (∀ n : ℕ, n < c → (∀ m : ℕ, m < n → seq m < 0) ∧ seq n = 0) ∧
  seq c > 0 ∧
  c = 52 :=
by
  sorry

end smallest_c_is_52_l19_19981


namespace min_value_of_sum_l19_19405

theorem min_value_of_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) : 
  ∃ x : ℝ, x = (1 / (a - 1) + 1 / b) ∧ x = 4 :=
by
  sorry

end min_value_of_sum_l19_19405


namespace sum_of_triangulars_15_to_20_l19_19713

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_triangulars_15_to_20 : 
  (triangular_number 15 + triangular_number 16 + triangular_number 17 + triangular_number 18 + triangular_number 19 + triangular_number 20) = 980 :=
by
  sorry

end sum_of_triangulars_15_to_20_l19_19713


namespace gwen_money_difference_l19_19587

theorem gwen_money_difference:
  let money_from_grandparents : ℕ := 15
  let money_from_uncle : ℕ := 8
  money_from_grandparents - money_from_uncle = 7 :=
by
  sorry

end gwen_money_difference_l19_19587


namespace dilution_problem_l19_19655

theorem dilution_problem
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (desired_concentration : ℝ)
  (initial_alcohol_content : initial_concentration * initial_volume / 100 = 4.8)
  (desired_alcohol_content : 4.8 = desired_concentration * (initial_volume + N) / 100)
  (N : ℝ) :
  N = 11.2 :=
sorry

end dilution_problem_l19_19655


namespace minimum_triangle_perimeter_l19_19628

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem minimum_triangle_perimeter (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4)) 
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
   l + m + n = 3003 := 
sorry

end minimum_triangle_perimeter_l19_19628


namespace percent_singles_l19_19567

theorem percent_singles :
  ∀ (total_hits home_runs triples doubles : ℕ),
  total_hits = 50 →
  home_runs = 2 →
  triples = 4 →
  doubles = 10 →
  (total_hits - (home_runs + triples + doubles)) * 100 / total_hits = 68 :=
by
  sorry

end percent_singles_l19_19567


namespace largest_multiple_of_15_less_than_500_is_495_l19_19154

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19154


namespace rockham_soccer_league_l19_19472

theorem rockham_soccer_league (cost_socks : ℕ) (cost_tshirt : ℕ) (custom_fee : ℕ) (total_cost : ℕ) :
  cost_socks = 6 →
  cost_tshirt = cost_socks + 7 →
  custom_fee = 200 →
  total_cost = 2892 →
  ∃ members : ℕ, total_cost - custom_fee = members * (2 * (cost_socks + cost_tshirt)) ∧ members = 70 :=
by
  intros
  sorry

end rockham_soccer_league_l19_19472


namespace expected_value_of_twelve_sided_die_l19_19373

noncomputable def expected_value_twelve_sided_die : ℝ :=
  let sides := 12
  let sum_faces := finset.sum (finset.range (sides + 1)) (λ k, k)
  let expected_value := sum_faces / sides
  expected_value

theorem expected_value_of_twelve_sided_die : expected_value_twelve_sided_die = 6.5 :=
by sorry

end expected_value_of_twelve_sided_die_l19_19373


namespace common_solution_exists_l19_19400

theorem common_solution_exists (a b : ℝ) :
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧
                         98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0)
  → a^2 + b^2 ≥ 13689 :=
by
  -- Proof omitted
  sorry

end common_solution_exists_l19_19400


namespace puddle_base_area_l19_19457

theorem puddle_base_area (rate depth hours : ℝ) (A : ℝ) 
  (h1 : rate = 10) 
  (h2 : depth = 30) 
  (h3 : hours = 3) 
  (h4 : depth * A = rate * hours) : 
  A = 1 := 
by 
  sorry

end puddle_base_area_l19_19457


namespace largest_multiple_of_15_below_500_l19_19176

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19176


namespace condition_sufficiency_not_necessity_l19_19037

variable {x y : ℝ}

theorem condition_sufficiency_not_necessity (hx : x ≥ 0) (hy : y ≥ 0) :
  (xy > 0 → |x + y| = |x| + |y|) ∧ (|x + y| = |x| + |y| → xy ≥ 0) :=
sorry

end condition_sufficiency_not_necessity_l19_19037


namespace average_speed_train_l19_19533

theorem average_speed_train (x : ℝ) (h1 : x ≠ 0) :
  let t1 := x / 40
  let t2 := 2 * x / 20
  let t3 := 3 * x / 60
  let total_time := t1 + t2 + t3
  let total_distance := 6 * x
  let average_speed := total_distance / total_time
  average_speed = 240 / 7 := by
  sorry

end average_speed_train_l19_19533


namespace find_c_l19_19041

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end find_c_l19_19041


namespace solve_for_x_l19_19789

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l19_19789


namespace expected_value_of_twelve_sided_die_l19_19363

theorem expected_value_of_twelve_sided_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (list.sum faces / list.length faces : ℝ) = 6.5 := 
by
  -- problem defines that there are 12 faces of the die numbered from 1 to 12
  have h_faces_length : list.length faces = 12 := rfl
  -- problem defines the sum of faces computed using the series sum formula
  have h_faces_sum : list.sum faces = 78 := rfl
  exact sorry

end expected_value_of_twelve_sided_die_l19_19363


namespace cube_difference_divisible_by_16_l19_19310

theorem cube_difference_divisible_by_16 (a b : ℤ) : 
  16 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3 + 8) :=
by
  sorry

end cube_difference_divisible_by_16_l19_19310


namespace max_xy_l19_19446

theorem max_xy (x y : ℝ) (hxy_pos : x > 0 ∧ y > 0) (h : 5 * x + 8 * y = 65) : 
  xy ≤ 25 :=
by
  sorry

end max_xy_l19_19446


namespace initial_money_l19_19777

-- Define the conditions
def spent_toy_truck : ℕ := 3
def spent_pencil_case : ℕ := 2
def money_left : ℕ := 5

-- Define the total money spent
def total_spent := spent_toy_truck + spent_pencil_case

-- Theorem statement
theorem initial_money (I : ℕ) (h : total_spent + money_left = I) : I = 10 :=
sorry

end initial_money_l19_19777


namespace average_income_proof_l19_19963

theorem average_income_proof:
  ∀ (A B C : ℝ),
    (A + B) / 2 = 5050 →
    (B + C) / 2 = 6250 →
    A = 4000 →
    (A + C) / 2 = 5200 := by
  sorry

end average_income_proof_l19_19963


namespace train_speed_in_kmph_l19_19532

-- Definitions based on the conditions
def train_length : ℝ := 280 -- in meters
def time_to_pass_tree : ℝ := 28 -- in seconds

-- Conversion factor from meters/second to kilometers/hour
def mps_to_kmph : ℝ := 3.6

-- The speed of the train in kilometers per hour
theorem train_speed_in_kmph : (train_length / time_to_pass_tree) * mps_to_kmph = 36 := 
sorry

end train_speed_in_kmph_l19_19532


namespace group_product_number_l19_19514

theorem group_product_number (a : ℕ) (group_size : ℕ) (interval : ℕ) (fifth_group_product : ℕ) :
  fifth_group_product = a + 4 * interval → fifth_group_product = 94 → group_size = 5 → interval = 20 →
  (a + (1 - 1) * interval + 1 * interval) = 34 :=
by
  intros fifth_group_eq fifth_group_is_94 group_size_is_5 interval_is_20
  -- Missing steps are handled by sorry
  sorry

end group_product_number_l19_19514


namespace expected_value_of_twelve_sided_die_l19_19353

theorem expected_value_of_twelve_sided_die : 
  let faces := (List.range (12+1)).tail in
  let E := (1 / 12) * (faces.sum) in
  E = 6.5 :=
by
  let faces := (List.range (12+1)).tail
  let E := (1 / 12) * (faces.sum)
  have : faces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] := by
    unfold faces
    simp [List.range, List.tail]
  have : faces.sum = 78 := by
    simp [this]
    norm_num
  have : E = (1 / 12) * 78 := by
    unfold E
    congr
  have : E = 6.5 := by
    simp [this]
    norm_num
  exact this

end expected_value_of_twelve_sided_die_l19_19353


namespace value_two_stds_less_than_mean_l19_19855

theorem value_two_stds_less_than_mean (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : (μ - 2 * σ) = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stds_less_than_mean_l19_19855


namespace sally_seashells_l19_19950

theorem sally_seashells 
  (seashells_monday : ℕ)
  (seashells_tuesday : ℕ)
  (price_per_seashell : ℝ)
  (h_monday : seashells_monday = 30)
  (h_tuesday : seashells_tuesday = seashells_monday / 2)
  (h_price : price_per_seashell = 1.2) :
  let total_seashells := seashells_monday + seashells_tuesday in
  let total_money := total_seashells * price_per_seashell in
  total_money = 54 := 
by
  sorry

end sally_seashells_l19_19950


namespace factorize_1_factorize_2_l19_19887

theorem factorize_1 {x : ℝ} : 2*x^2 - 4*x = 2*x*(x - 2) := 
by sorry

theorem factorize_2 {a b x y : ℝ} : a^2*(x - y) + b^2*(y - x) = (x - y) * (a + b) * (a - b) := 
by sorry

end factorize_1_factorize_2_l19_19887


namespace minimum_even_N_for_A_2015_turns_l19_19308

noncomputable def a (n : ℕ) : ℕ :=
  6 * 2^n - 4

def A_minimum_even_moves_needed (k : ℕ) : ℕ :=
  2015 - 1

theorem minimum_even_N_for_A_2015_turns :
  ∃ N : ℕ, 2 ∣ N ∧ A_minimum_even_moves_needed 2015 ≤ N ∧ a 1007 = 6 * 2^1007 - 4 := by
  sorry

end minimum_even_N_for_A_2015_turns_l19_19308


namespace coordinates_of_S_l19_19664

variable (P Q R S : (ℝ × ℝ))
variable (hp : P = (3, -2))
variable (hq : Q = (3, 1))
variable (hr : R = (7, 1))
variable (h : Rectangle P Q R S)

def Rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  let (xS, yS) := S
  (xP = xQ ∧ yR = yS) ∧ (xS = xR ∧ yP = yQ) 

theorem coordinates_of_S : S = (7, -2) := by
  sorry

end coordinates_of_S_l19_19664


namespace six_letter_words_count_l19_19881

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end six_letter_words_count_l19_19881


namespace gcd_of_12012_and_18018_l19_19582

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l19_19582


namespace units_digit_quotient_4_l19_19719

theorem units_digit_quotient_4 (n : ℕ) (h₁ : n ≥ 1) :
  (5^1994 + 6^1994) % 10 = 1 ∧ (5^1994 + 6^1994) % 7 = 5 → 
  (5^1994 + 6^1994) / 7 % 10 = 4 := 
sorry

end units_digit_quotient_4_l19_19719


namespace sally_earnings_l19_19949

-- Definitions based on the conditions
def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20
def total_money : ℝ := total_seashells * price_per_seashell

-- Lean 4 statement to prove the total amount of money is $54
theorem sally_earnings : total_money = 54 := by
  -- Proof will go here
  sorry

end sally_earnings_l19_19949


namespace find_x_l19_19281

theorem find_x (p q x : ℚ) (h1 : p / q = 4 / 5)
    (h2 : 4 / 7 + x / (2 * q + p) = 1) : x = 12 := 
by
  sorry

end find_x_l19_19281


namespace num_of_valid_3x3_grids_l19_19390

theorem num_of_valid_3x3_grids :
  ∃ (arrangements : Finset (Matrix (Fin 3) (Fin 3) ℕ)), 
  ∀ (M ∈ arrangements), 
    {i : Fin 3 // M i 0 + M i 1 + M i 2 = 15} ∧
    {j : Fin 3 // M 0 j + M 1 j + M 2 j = 15} ∧
    (arrangements.card = 72) ∧
    ((∀ (i j : Fin 3), 1 ≤ M i j ∧ M i j ≤ 9) ∧ 
     (M.to_finset.card = 9)) :=
by sorry

end num_of_valid_3x3_grids_l19_19390


namespace number_of_teams_l19_19918

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end number_of_teams_l19_19918


namespace bookshelf_arrangements_l19_19585

theorem bookshelf_arrangements :
  let math_books := 6
  let english_books := 5
  let valid_arrangements := 2400
  (∃ (math_books : Nat) (english_books : Nat) (valid_arrangements : Nat), 
    math_books = 6 ∧ english_books = 5 ∧ valid_arrangements = 2400) :=
by
  sorry

end bookshelf_arrangements_l19_19585


namespace annual_average_growth_rate_estimated_output_value_2006_l19_19946

-- First problem: Prove the annual average growth rate from 2003 to 2005
theorem annual_average_growth_rate (x : ℝ) (h : 6.4 * (1 + x)^2 = 10) : 
  x = 1/4 :=
by
  sorry

-- Second problem: Prove the estimated output value for 2006 given the annual growth rate
theorem estimated_output_value_2006 (x : ℝ) (output_2005 : ℝ) (h_growth : x = 1/4) (h_2005 : output_2005 = 10) : 
  output_2005 * (1 + x) = 12.5 :=
by 
  sorry

end annual_average_growth_rate_estimated_output_value_2006_l19_19946


namespace sequence_of_perfect_squares_l19_19077

theorem sequence_of_perfect_squares (A B C D: ℕ)
(h1: 10 ≤ 10 * A + B) 
(h2 : 10 * A + B < 100) 
(h3 : (10 * A + B) % 3 = 0 ∨ (10 * A + B) % 3 = 1)
(hC : 1 ≤ C ∧ C ≤ 9)
(hD : 1 ≤ D ∧ D ≤ 9)
(hCD : (C + D) % 3 = 0)
(hAB_square : ∃ k₁ : ℕ, k₁^2 = 10 * A + B) 
(hACDB_square : ∃ k₂ : ℕ, k₂^2 = 1000 * A + 100 * C + 10 * D + B) 
(hACCDDB_square : ∃ k₃ : ℕ, k₃^2 = 100000 * A + 10000 * C + 1000 * C + 100 * D + 10 * D + B) :
∀ n: ℕ, ∃ k : ℕ, k^2 = (10^n * A + (10^(n/2) * C) + (10^(n/2) * D) + B) := 
by
  sorry

end sequence_of_perfect_squares_l19_19077


namespace trigonometric_identity_l19_19412

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11 / 12 :=
by
  sorry

end trigonometric_identity_l19_19412


namespace largest_multiple_of_15_less_than_500_l19_19136

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19136


namespace ratio_of_numbers_l19_19483

theorem ratio_of_numbers (A B : ℕ) (hA : A = 45) (hLCM : Nat.lcm A B = 180) : A / Nat.lcm A B = 45 / 4 :=
by
  sorry

end ratio_of_numbers_l19_19483


namespace find_z_plus_one_over_y_l19_19467

variable {x y z : ℝ}

theorem find_z_plus_one_over_y (h1 : x * y * z = 1) 
                                (h2 : x + 1 / z = 7) 
                                (h3 : y + 1 / x = 31) 
                                (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : 
                              z + 1 / y = 5 / 27 := 
by
  sorry

end find_z_plus_one_over_y_l19_19467


namespace gum_ratio_correct_l19_19244

variable (y : ℝ)
variable (cherry_pieces : ℝ := 30)
variable (grape_pieces : ℝ := 40)
variable (pieces_per_pack : ℝ := y)

theorem gum_ratio_correct:
  ((cherry_pieces - 2 * pieces_per_pack) / grape_pieces = cherry_pieces / (grape_pieces + 4 * pieces_per_pack)) ↔ y = 5 :=
by
  sorry

end gum_ratio_correct_l19_19244


namespace right_triangle_of_three_colors_exists_l19_19007

-- Define the type for color
inductive Color
| color1
| color2
| color3

open Color

-- Define the type for integer coordinate points
structure Point :=
(x : ℤ)
(y : ℤ)
(color : Color)

-- Define the conditions
def all_points_colored : Prop :=
∀ (p : Point), p.color = color1 ∨ p.color = color2 ∨ p.color = color3

def all_colors_used : Prop :=
∃ (p1 p2 p3 : Point), p1.color = color1 ∧ p2.color = color2 ∧ p3.color = color3

-- Define the right_triangle_exist problem
def right_triangle_exists : Prop :=
∃ (p1 p2 p3 : Point), 
  p1.color ≠ p2.color ∧ p2.color ≠ p3.color ∧ p3.color ≠ p1.color ∧
  (p1.x = p2.x ∧ p2.y = p3.y ∧ p1.y = p3.y ∨
   p1.y = p2.y ∧ p2.x = p3.x ∧ p1.x = p3.x ∨
   (p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y) = (p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) ∧
   (p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y) = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y))

theorem right_triangle_of_three_colors_exists (h1 : all_points_colored) (h2 : all_colors_used) : right_triangle_exists := 
sorry

end right_triangle_of_three_colors_exists_l19_19007


namespace amount_after_two_years_l19_19383

theorem amount_after_two_years (P : ℝ) (r1 r2 : ℝ) : 
  P = 64000 → 
  r1 = 0.12 → 
  r2 = 0.15 → 
  (P + P * r1) + (P + P * r1) * r2 = 82432 := by
  sorry

end amount_after_two_years_l19_19383


namespace roots_of_abs_exp_eq_b_l19_19882

theorem roots_of_abs_exp_eq_b (b : ℝ) (h : 0 < b ∧ b < 1) : 
  ∃! (x1 x2 : ℝ), x1 ≠ x2 ∧ abs (2^x1 - 1) = b ∧ abs (2^x2 - 1) = b :=
sorry

end roots_of_abs_exp_eq_b_l19_19882


namespace greatest_b_for_no_real_roots_l19_19013

theorem greatest_b_for_no_real_roots :
  ∀ (b : ℤ), (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) ↔ b ≤ 6 := sorry

end greatest_b_for_no_real_roots_l19_19013


namespace option_b_correct_l19_19874

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) (h : a * c^2 > b * c^2) : a > b :=
sorry

end option_b_correct_l19_19874


namespace goals_per_player_is_30_l19_19659

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l19_19659


namespace max_n_l19_19052

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2 ^ a n

theorem max_n (n : ℕ) (h1 : a 2 = 2) (h2 : ∀ n, b n = 2 ^ a n)
  (h3 : b 4 = 4 * b 2) : n ≤ 9 :=
by 
  sorry

end max_n_l19_19052


namespace largest_multiple_of_15_less_than_500_l19_19100

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19100


namespace necessary_but_not_sufficient_l19_19338

variable (a b : ℝ)

theorem necessary_but_not_sufficient : 
  ¬ (a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) ∧ (a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  sorry

end necessary_but_not_sufficient_l19_19338


namespace largest_multiple_of_15_less_than_500_l19_19132

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19132


namespace smallest_x_solution_l19_19380

theorem smallest_x_solution :
  (∃ x : ℝ, (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) ∧ ∀ y : ℝ, (3 * y^2 + 36 * y - 90 = 2 * y * (y + 16)) → x ≤ y) ↔ x = -10 :=
by
  sorry

end smallest_x_solution_l19_19380


namespace find_a_l19_19806

theorem find_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^3 + 3 * x^2 + 2)
  (hf' : ∀ x, f' x = 3 * a * x^2 + 6 * x) 
  (h : f' (-1) = 4) : 
  a = (10 : ℝ) / 3 := 
sorry

end find_a_l19_19806


namespace sequence_fill_l19_19291

theorem sequence_fill (x2 x3 x4 x5 x6 x7: ℕ) : 
  (20 + x2 + x3 = 100) ∧ 
  (x2 + x3 + x4 = 100) ∧ 
  (x3 + x4 + x5 = 100) ∧ 
  (x4 + x5 + x6 = 100) ∧ 
  (x5 + x6 + 16 = 100) →
  [20, x2, x3, x4, x5, x6, 16] = [20, 16, 64, 20, 16, 64, 20, 16] :=
by
  sorry

end sequence_fill_l19_19291


namespace max_candies_takeable_l19_19919

theorem max_candies_takeable : 
  ∃ (max_take : ℕ), max_take = 159 ∧
  ∀ (boxes: Fin 5 → ℕ), 
    boxes 0 = 11 → 
    boxes 1 = 22 → 
    boxes 2 = 33 → 
    boxes 3 = 44 → 
    boxes 4 = 55 →
    (∀ (i : Fin 5), 
      ∀ (new_boxes : Fin 5 → ℕ),
      (new_boxes i = boxes i - 4) ∧ 
      (∀ (j : Fin 5), j ≠ i → new_boxes j = boxes j + 1) →
      boxes i = 0 → max_take = new_boxes i) :=
sorry

end max_candies_takeable_l19_19919


namespace largest_multiple_of_15_less_than_500_l19_19101

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19101


namespace find_angle_E_l19_19056

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end find_angle_E_l19_19056


namespace trader_gain_l19_19711

-- Conditions
def cost_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the cost price of a pen
def selling_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the selling price of a pen
def gain_percentage : ℝ := 0.40 -- 40% gain

-- Statement of the problem to prove
theorem trader_gain (C : ℝ) (N : ℕ) : 
  (100 : ℕ) * C * gain_percentage = N * C → 
  N = 40 :=
by
  sorry

end trader_gain_l19_19711


namespace m_minus_n_is_perfect_square_l19_19771

theorem m_minus_n_is_perfect_square (m n : ℕ) (h : 0 < m) (h1 : 0 < n) (h2 : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, m = n + k^2 :=
by
    sorry

end m_minus_n_is_perfect_square_l19_19771


namespace largest_multiple_of_15_less_than_500_l19_19138

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19138


namespace average_investment_per_km_in_scientific_notation_l19_19515

-- Definitions based on the conditions of the problem
def total_investment : ℝ := 29.6 * 10^9
def upgraded_distance : ℝ := 6000

-- A theorem to be proven
theorem average_investment_per_km_in_scientific_notation :
  (total_investment / upgraded_distance) = 4.9 * 10^6 :=
by
  sorry

end average_investment_per_km_in_scientific_notation_l19_19515


namespace triple_square_side_area_l19_19780

theorem triple_square_side_area (s : ℝ) : (3 * s) ^ 2 ≠ 3 * (s ^ 2) :=
by {
  sorry
}

end triple_square_side_area_l19_19780


namespace inequality_solution_l19_19562

theorem inequality_solution (x : ℝ) :
  (-4 ≤ x ∧ x < -3 / 2) ↔ (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) :=
by
  sorry

end inequality_solution_l19_19562


namespace hyperbola_focus_eq_parabola_focus_l19_19283

theorem hyperbola_focus_eq_parabola_focus (k : ℝ) (hk : k > 0) :
  let parabola_focus : ℝ × ℝ := (2, 0) in
  let hyperbola_focus_distance : ℝ := Real.sqrt (1 + k^2) in
  hyperbola_focus_distance = 2 ↔ k = Real.sqrt 3 :=
by {
  sorry
}

end hyperbola_focus_eq_parabola_focus_l19_19283


namespace total_workers_is_22_l19_19512

-- Define constants and variables based on conditions
def avg_salary_all : ℝ := 850
def avg_salary_technicians : ℝ := 1000
def avg_salary_rest : ℝ := 780
def num_technicians : ℝ := 7

-- Define the necessary proof statement
theorem total_workers_is_22
  (W : ℝ)
  (h1 : W * avg_salary_all = num_technicians * avg_salary_technicians + (W - num_technicians) * avg_salary_rest) :
  W = 22 :=
by
  sorry

end total_workers_is_22_l19_19512


namespace quadratic_solution_l19_19958

theorem quadratic_solution : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 3 = 0 ↔ (x = 1 ∨ x = 3 / 2) :=
by
  intro x
  constructor
  sorry

end quadratic_solution_l19_19958


namespace probability_of_at_least_one_pair_of_women_l19_19341

/--
Theorem: Calculate the probability that at least one pair consists of two young women from a group of 6 young men and 6 young women paired up randomly is 0.93.
-/
theorem probability_of_at_least_one_pair_of_women 
  (men_women_group : Finset (Fin 12))
  (pairs : Finset (Finset (Fin 12)))
  (h_pairs : pairs.card = 6)
  (h_men_women : ∀ pair ∈ pairs, pair.card = 2)
  (h_distinct : ∀ (x y : Finset (Fin 12)), x ≠ y → x ∩ y = ∅):
  ∃ (p : ℝ), p = 0.93 := 
sorry

end probability_of_at_least_one_pair_of_women_l19_19341


namespace goals_scored_by_each_l19_19662

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l19_19662


namespace find_f_one_l19_19265

-- Define the function f(x-3) = 2x^2 - 3x + 1
noncomputable def f (x : ℤ) := 2 * (x+3)^2 - 3 * (x+3) + 1

-- Declare the theorem we intend to prove
theorem find_f_one : f 1 = 21 :=
by
  -- The proof goes here (saying "sorry" because the detailed proof is skipped)
  sorry

end find_f_one_l19_19265


namespace find_constants_monotonicity_l19_19271

noncomputable def f (x a b : ℝ) := (x^2 + a * x) * Real.exp x + b

theorem find_constants (a b : ℝ) (h_tangent : (f 0 a b = 1) ∧ (deriv (f · a b) 0 = -2)) :
  a = -2 ∧ b = 1 := by
  sorry

theorem monotonicity (a b : ℝ) (h_constants : a = -2 ∧ b = 1) :
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) > 0 → x > Real.sqrt 2 ∨ x < -Real.sqrt 2)) ∧
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) < 0 → -Real.sqrt 2 < x ∧ x < Real.sqrt 2)) := by
  sorry

end find_constants_monotonicity_l19_19271


namespace largest_multiple_of_15_below_500_l19_19171

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19171


namespace determine_c15_l19_19552

noncomputable def polynomial_product (c : ℕ → ℕ) : Polynomial ℤ :=
  (List.range 15).foldr (λ k p, p * (1 - Polynomial.C z ^ (k+1)) ^ (c (k+1))) 1

theorem determine_c15 (c : ℕ → ℕ) (h1 : c 15 = 0) :
  polynomial_product c ≡ 1 - 3 * Polynomial.C z [MOD z^20] :=
sorry

end determine_c15_l19_19552


namespace largest_multiple_of_15_less_than_500_l19_19189

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19189


namespace four_digit_divisors_l19_19724

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l19_19724


namespace quadrilateral_perimeter_l19_19330

theorem quadrilateral_perimeter
  (EF FG HG : ℝ)
  (h1 : EF = 7)
  (h2 : FG = 15)
  (h3 : HG = 3)
  (perp1 : EF * FG = 0)
  (perp2 : HG * FG = 0) :
  EF + FG + HG + Real.sqrt (4^2 + 15^2) = 25 + Real.sqrt 241 :=
by
  sorry

end quadrilateral_perimeter_l19_19330


namespace range_of_c_l19_19419

theorem range_of_c (c : ℝ) :
  (c^2 - 5 * c + 7 > 1 ∧ (|2 * c - 1| ≤ 1)) ∨ ((c^2 - 5 * c + 7 ≤ 1) ∧ |2 * c - 1| > 1) ↔ (0 ≤ c ∧ c ≤ 1) ∨ (2 ≤ c ∧ c ≤ 3) :=
sorry

end range_of_c_l19_19419


namespace stratified_sampling_l19_19756

theorem stratified_sampling 
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (H_male_students : male_students = 40)
  (H_female_students : female_students = 30)
  (H_sample_size : sample_size = 7)
  (H_stratified_sample : sample_size = male_students_drawn + female_students_drawn) :
  male_students_drawn = 4 ∧ female_students_drawn = 3  :=
sorry

end stratified_sampling_l19_19756


namespace fifth_friend_paid_40_l19_19259

-- Defining the conditions given in the problem
variables {a b c d e : ℝ}
variables (h1 : a = (1/3) * (b + c + d + e))
variables (h2 : b = (1/4) * (a + c + d + e))
variables (h3 : c = (1/5) * (a + b + d + e))
variables (h4 : d = (1/6) * (a + b + c + e))
variables (h5 : a + b + c + d + e = 120)

-- Proving that the amount paid by the fifth friend is $40
theorem fifth_friend_paid_40 : e = 40 :=
by
  sorry  -- Proof to be provided

end fifth_friend_paid_40_l19_19259


namespace insurance_payment_yearly_l19_19450

noncomputable def quarterly_payment : ℝ := 378
noncomputable def quarters_per_year : ℕ := 12 / 3
noncomputable def annual_payment : ℝ := quarterly_payment * quarters_per_year

theorem insurance_payment_yearly : annual_payment = 1512 := by
  sorry

end insurance_payment_yearly_l19_19450


namespace largest_multiple_of_15_less_than_500_l19_19182

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19182


namespace solve_equation_l19_19793

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l19_19793


namespace find_an_from_sums_l19_19923

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l19_19923


namespace ratio_dvds_to_cds_l19_19610

def total_sold : ℕ := 273
def dvds_sold : ℕ := 168
def cds_sold : ℕ := total_sold - dvds_sold

theorem ratio_dvds_to_cds : (dvds_sold : ℚ) / cds_sold = 8 / 5 := by
  sorry

end ratio_dvds_to_cds_l19_19610


namespace no_digit_c_make_2C4_multiple_of_5_l19_19396

theorem no_digit_c_make_2C4_multiple_of_5 : ∀ C, ¬ (C ≥ 0 ∧ C ≤ 9 ∧ (20 * 10 + C * 10 + 4) % 5 = 0) :=
by intros C hC; sorry

end no_digit_c_make_2C4_multiple_of_5_l19_19396


namespace abs_ineq_subs_ineq_l19_19685

-- Problem 1
theorem abs_ineq (x : ℝ) : -2 ≤ x ∧ x ≤ 2 ↔ |x - 1| + |x + 1| ≤ 4 := 
sorry

-- Problem 2
theorem subs_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a + b + c := 
sorry

end abs_ineq_subs_ineq_l19_19685


namespace smallest_bdf_value_l19_19940

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l19_19940


namespace bacterium_descendants_in_range_l19_19688

theorem bacterium_descendants_in_range (total_bacteria : ℕ) (initial : ℕ) 
  (h_total : total_bacteria = 1000) (h_initial : initial = total_bacteria) 
  (descendants : ℕ → ℕ)
  (h_step : ∀ k, descendants (k+1) ≤ descendants k / 2) :
  ∃ k, 334 ≤ descendants k ∧ descendants k ≤ 667 :=
by
  sorry

end bacterium_descendants_in_range_l19_19688


namespace add_percentages_10_30_15_50_l19_19505

-- Define the problem conditions:
def ten_percent (x : ℝ) : ℝ := 0.10 * x
def fifteen_percent (y : ℝ) : ℝ := 0.15 * y
def add_percentages (x y : ℝ) : ℝ := ten_percent x + fifteen_percent y

theorem add_percentages_10_30_15_50 :
  add_percentages 30 50 = 10.5 :=
by
  sorry

end add_percentages_10_30_15_50_l19_19505


namespace sum_of_three_numbers_l19_19828

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19828


namespace transform_expression_to_product_l19_19602

variables (a b c d s: ℝ)

theorem transform_expression_to_product
  (h1 : d = a + b + c)
  (h2 : s = (a + b + c + d) / 2) :
    2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) -
    (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 16 * (s - a) * (s - b) * (s - c) * (s - d) :=
by
  sorry

end transform_expression_to_product_l19_19602


namespace sqrt_720_l19_19645

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l19_19645


namespace arithmetic_sequence_second_term_l19_19048

theorem arithmetic_sequence_second_term (a1 a5 : ℝ) (h1 : a1 = 2020) (h5 : a5 = 4040) : 
  ∃ d a2 : ℝ, a2 = a1 + d ∧ d = (a5 - a1) / 4 ∧ a2 = 2525 :=
by
  sorry

end arithmetic_sequence_second_term_l19_19048


namespace cos_alpha_value_cos_2alpha_value_l19_19901

noncomputable def x : ℤ := -3
noncomputable def y : ℤ := 4
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def cos_alpha : ℝ := x / r
noncomputable def cos_2alpha : ℝ := 2 * cos_alpha^2 - 1

theorem cos_alpha_value : cos_alpha = -3 / 5 := by
  sorry

theorem cos_2alpha_value : cos_2alpha = -7 / 25 := by
  sorry

end cos_alpha_value_cos_2alpha_value_l19_19901


namespace avg_salary_supervisors_l19_19916

-- Definitions based on the conditions of the problem
def total_workers : Nat := 48
def supervisors : Nat := 6
def laborers : Nat := 42
def avg_salary_total : Real := 1250
def avg_salary_laborers : Real := 950

-- Given the above conditions, we need to prove the average salary of the supervisors.
theorem avg_salary_supervisors :
  (supervisors * (supervisors * total_workers * avg_salary_total - laborers * avg_salary_laborers) / supervisors) = 3350 :=
by
  sorry

end avg_salary_supervisors_l19_19916


namespace domain_log_function_l19_19083

/-- The quadratic expression x^2 - 2x + 3 is always positive. -/
lemma quadratic_positive (x : ℝ) : x^2 - 2*x + 3 > 0 :=
by
  sorry

/-- The domain of the function y = log(x^2 - 2x + 3) is all real numbers. -/
theorem domain_log_function : ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*x + 3) :=
by
  have h := quadratic_positive
  sorry

end domain_log_function_l19_19083


namespace ratio_of_sums_eq_neg_sqrt_2_l19_19930

open Real

theorem ratio_of_sums_eq_neg_sqrt_2
    (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
    (x + y) / (x - y) = -Real.sqrt 2 :=
by sorry

end ratio_of_sums_eq_neg_sqrt_2_l19_19930


namespace find_a_sq_plus_b_sq_l19_19785

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 48
axiom h2 : a * b = 156

theorem find_a_sq_plus_b_sq : a^2 + b^2 = 1992 :=
by sorry

end find_a_sq_plus_b_sq_l19_19785


namespace pump_B_rate_l19_19709

noncomputable def rate_A := 1 / 2
noncomputable def rate_C := 1 / 6

theorem pump_B_rate :
  ∃ B : ℝ, (rate_A + B - rate_C = 4 / 3) ∧ (B = 1) := by
  sorry

end pump_B_rate_l19_19709


namespace calculate_expression_l19_19243

theorem calculate_expression : (2 * Real.sqrt 3 - Real.pi)^0 - abs (1 - Real.sqrt 3) + 3 * Real.tan (Real.pi / 6) + (-1 / 2)^(-2) = 6 :=
by
  sorry

end calculate_expression_l19_19243


namespace ball_distribution_into_drawers_l19_19821

noncomputable def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem ball_distribution_into_drawers :
  comb 7 4 = 35 := 
sorry

end ball_distribution_into_drawers_l19_19821


namespace add_fractions_l19_19242

theorem add_fractions :
  (8:ℚ) / 19 + 5 / 57 = 29 / 57 :=
sorry

end add_fractions_l19_19242


namespace arithmetic_sequence_general_formula_l19_19738

open Finset BigOperators

-- Part (1): General formula for the arithmetic sequence
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 4 * n

-- Part (2): Sum of first 2n terms of sequence b
def special_seq (a b : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ 
  (∀ n : ℕ, n % 2 = 1 → b (n + 1) = a n) ∧ 
  (∀ n : ℕ, n % 2 = 0 → b (n + 1) = - b n + 2^n)

def sum_first_2n_terms (P : ℕ → ℕ → ℕ) : Prop :=
  ∀ S n: ℕ,
  S = (∑ i in range (2 * n), λ i, b i) → 
  S = (4^n - 1) / 3 + 4 * n - 3

-- Main Lean statement
theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (b : ℕ → ℕ) :
  arithmetic_seq a →
  special_seq a b →
  sum_first_2n_terms b :=
by 
  sorry

end arithmetic_sequence_general_formula_l19_19738


namespace beckys_age_ratio_l19_19884

theorem beckys_age_ratio (Eddie_age : ℕ) (Irene_age : ℕ)
  (becky_age: ℕ)
  (H1 : Eddie_age = 92)
  (H2 : Irene_age = 46)
  (H3 : Irene_age = 2 * becky_age) :
  becky_age / Eddie_age = 1 / 4 :=
by
  sorry

end beckys_age_ratio_l19_19884


namespace greatest_whole_number_satisfying_inequality_l19_19003

theorem greatest_whole_number_satisfying_inequality :
  ∀ (x : ℤ), 3 * x + 2 < 5 - 2 * x → x <= 0 :=
by
  sorry

end greatest_whole_number_satisfying_inequality_l19_19003


namespace largest_multiple_of_15_less_than_500_l19_19115

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19115


namespace river_current_speed_l19_19343

/-- A man rows 18 miles upstream in three hours more time than it takes him to row 
the same distance downstream. If he halves his usual rowing rate, the time upstream 
becomes only two hours more than the time downstream. Prove that the speed of 
the river's current is 2 miles per hour. -/
theorem river_current_speed (r w : ℝ) 
    (h1 : 18 / (r - w) - 18 / (r + w) = 3)
    (h2 : 18 / (r / 2 - w) - 18 / (r / 2 + w) = 2) : 
    w = 2 := 
sorry

end river_current_speed_l19_19343


namespace min_value_of_f_on_interval_l19_19928

noncomputable def f (x : ℝ) : ℝ := (1/2)*x^2 - x - 2*Real.log x

theorem min_value_of_f_on_interval : 
  ∃ c ∈ set.Icc (1:ℝ) Real.exp 1, ∀ x ∈ set.Icc (1:ℝ) Real.exp 1, f x ≥ f c ∧ f c = -2 * Real.log 2 := 
by
  sorry

end min_value_of_f_on_interval_l19_19928


namespace car_R_average_speed_l19_19978

theorem car_R_average_speed :
  ∃ (v : ℕ), (600 / v) - 2 = 600 / (v + 10) ∧ v = 50 :=
by sorry

end car_R_average_speed_l19_19978


namespace probability_five_heads_in_six_tosses_is_09375_l19_19528

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_exact_heads (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * (p^k) * ((1-p)^(n-k))
  
theorem probability_five_heads_in_six_tosses_is_09375 :
  probability_exact_heads 6 5 0.5 = 0.09375 :=
by
  sorry

end probability_five_heads_in_six_tosses_is_09375_l19_19528


namespace hyperbola_parabola_focus_l19_19282

theorem hyperbola_parabola_focus (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, (1/k^2) * y^2 = 0 ∧ x^2 - (y^2 / k^2) = 1) ∧ (∃ x : ℝ, y^2 = 8 * x) →
  k = Real.sqrt 3 :=
by sorry

end hyperbola_parabola_focus_l19_19282


namespace animal_legs_l19_19634

theorem animal_legs (dogs chickens spiders octopus : Nat) (legs_dog legs_chicken legs_spider legs_octopus : Nat)
  (h1 : dogs = 3)
  (h2 : chickens = 4)
  (h3 : spiders = 2)
  (h4 : octopus = 1)
  (h5 : legs_dog = 4)
  (h6 : legs_chicken = 2)
  (h7 : legs_spider = 8)
  (h8 : legs_octopus = 8) :
  dogs * legs_dog + chickens * legs_chicken + spiders * legs_spider + octopus * legs_octopus = 44 := by
    sorry

end animal_legs_l19_19634


namespace largest_multiple_of_15_less_than_500_l19_19145

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19145


namespace find_integer_closest_expression_l19_19387

theorem find_integer_closest_expression :
  let a := (7 + Real.sqrt 48) ^ 2023
  let b := (7 - Real.sqrt 48) ^ 2023
  ((a + b) ^ 2 - (a - b) ^ 2) = 4 :=
by
  sorry

end find_integer_closest_expression_l19_19387


namespace expected_value_twelve_sided_die_l19_19366

theorem expected_value_twelve_sided_die : 
  (1 : ℝ)/12 * (∑ k in Finset.range 13, k) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l19_19366


namespace gcd_of_12012_18018_l19_19581

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l19_19581


namespace conversion_correct_l19_19561

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l19_19561


namespace central_angle_is_two_l19_19413

noncomputable def central_angle_of_sector (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : ℝ :=
  l / r

theorem central_angle_is_two (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : central_angle_of_sector r l h1 h2 = 2 :=
by
  sorry

end central_angle_is_two_l19_19413


namespace tank_capacity_l19_19849

theorem tank_capacity (T : ℝ) (h : 0.4 * T = 0.9 * T - 36) : T = 72 := by
  sorry

end tank_capacity_l19_19849


namespace solve_equation_l19_19794

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end solve_equation_l19_19794


namespace smallest_bdf_value_l19_19941

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l19_19941


namespace joe_flight_expense_l19_19298

theorem joe_flight_expense
  (initial_amount : ℕ)
  (hotel_expense : ℕ)
  (food_expense : ℕ)
  (remaining_amount : ℕ)
  (flight_expense : ℕ)
  (h1 : initial_amount = 6000)
  (h2 : hotel_expense = 800)
  (h3 : food_expense = 3000)
  (h4 : remaining_amount = 1000)
  (h5 : flight_expense = initial_amount - remaining_amount - hotel_expense - food_expense) :
  flight_expense = 1200 :=
by
  sorry

end joe_flight_expense_l19_19298


namespace largest_multiple_of_15_less_than_500_l19_19204

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19204


namespace no_real_roots_m_eq_1_range_of_m_l19_19272

-- Definitions of the given functions
def f (x : ℝ) (m : ℝ) := m*x - m/x
def g (x : ℝ) := 2 * Real.log x

-- First proof problem: Proving no real roots
theorem no_real_roots_m_eq_1 (x : ℝ) (h1 : 1 < x) : f x 1 ≠ g x :=
  sorry

-- Second proof problem: Finding the range of m
theorem range_of_m (m : ℝ) 
  (h2 : ∀ x ∈ Set.Ioc 1 Real.exp, f x m - g x < 2) : 
  m < (4 * Real.exp) / (Real.exp^2 - 1) :=
  sorry

end no_real_roots_m_eq_1_range_of_m_l19_19272


namespace diff_only_at_zero_l19_19438

open Complex

noncomputable def w (z : ℂ) : ℂ := z * conj z

theorem diff_only_at_zero :
  (∀ z : ℂ, DifferentiableAt ℂ w z ↔ z = 0) ∧ 
  ¬ Analytic ℂ (fun z => z * conj z) :=
by
  sorry

end diff_only_at_zero_l19_19438


namespace gcd_12012_18018_l19_19571

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l19_19571


namespace largest_multiple_of_15_less_than_500_l19_19167

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19167


namespace mr_green_yield_l19_19068

noncomputable def steps_to_feet (steps : ℕ) : ℝ :=
  steps * 2.5

noncomputable def total_yield (steps_x : ℕ) (steps_y : ℕ) (yield_potato_per_sqft : ℝ) (yield_carrot_per_sqft : ℝ) : ℝ :=
  let width := steps_to_feet steps_x
  let height := steps_to_feet steps_y
  let area := width * height
  (area * yield_potato_per_sqft) + (area * yield_carrot_per_sqft)

theorem mr_green_yield :
  total_yield 20 25 0.5 0.25 = 2343.75 :=
by
  sorry

end mr_green_yield_l19_19068


namespace c_range_l19_19596

open Real

theorem c_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1)
  (h2 : 1 / (a + b) + 1 / c = 1) : 1 < c ∧ c ≤ 4 / 3 := 
sorry

end c_range_l19_19596


namespace expected_value_of_12_sided_die_is_6_5_l19_19357

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_l19_19357


namespace second_divisor_l19_19524

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l19_19524


namespace find_c_l19_19311

/-- Seven unit squares are arranged in a row in the coordinate plane, 
with the lower left corner of the first square at the origin. 
A line extending from (c,0) to (4,4) divides the entire region 
into two regions of equal area. What is the value of c?
-/
theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 7 ∧ y = (4 / (4 - c)) * (x - c)) : c = 2.25 :=
sorry

end find_c_l19_19311


namespace alexandra_magazines_l19_19537

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l19_19537


namespace largest_multiple_of_15_less_than_500_is_495_l19_19157

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19157


namespace steve_nickels_dimes_l19_19799

theorem steve_nickels_dimes (n d : ℕ) (h1 : d = n + 4) (h2 : 5 * n + 10 * d = 70) : n = 2 :=
by
  -- The proof goes here
  sorry

end steve_nickels_dimes_l19_19799


namespace fraction_increase_invariance_l19_19605

theorem fraction_increase_invariance (x y : ℝ) :
  (3 * (2 * y)) / (2 * x + 2 * y) = 3 * y / (x + y) :=
by
  sorry

end fraction_increase_invariance_l19_19605


namespace age_difference_l19_19998

theorem age_difference {A B C : ℕ} (h : A + B = B + C + 15) : A - C = 15 := 
by 
  sorry

end age_difference_l19_19998


namespace grid_square_count_l19_19716

theorem grid_square_count :
  let width := 6
  let height := 6
  let num_1x1 := (width - 1) * (height - 1)
  let num_2x2 := (width - 2) * (height - 2)
  let num_3x3 := (width - 3) * (height - 3)
  let num_4x4 := (width - 4) * (height - 4)
  num_1x1 + num_2x2 + num_3x3 + num_4x4 = 54 :=
by
  sorry

end grid_square_count_l19_19716


namespace range_of_a_l19_19046

theorem range_of_a {a : ℝ} :
  (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 4) ↔ (-2*Real.sqrt 2 < a ∧ a < 2*Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end range_of_a_l19_19046


namespace no_triangular_sides_of_specific_a_b_l19_19542

theorem no_triangular_sides_of_specific_a_b (a b c : ℕ) (h1 : a = 10^100 + 1002) (h2 : b = 1001) (h3 : ∃ n : ℕ, c = n^2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by sorry

end no_triangular_sides_of_specific_a_b_l19_19542


namespace find_c_l19_19749

-- Definitions from the problem conditions
variables (a c : ℕ)
axiom cond1 : 2 ^ a = 8
axiom cond2 : a = 3 * c

-- The goal is to prove c = 1
theorem find_c : c = 1 :=
by
  sorry

end find_c_l19_19749


namespace root_sum_product_eq_l19_19812

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l19_19812


namespace range_of_z_in_parallelogram_l19_19736

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -1, y := 2}
def B : Point := {x := 3, y := 4}
def C : Point := {x := 4, y := -2}

-- Define the condition for point (x, y) to be inside the parallelogram (including boundary)
def isInsideParallelogram (p : Point) : Prop := sorry -- Placeholder for actual geometric condition

-- Statement of the problem
theorem range_of_z_in_parallelogram (p : Point) (h : isInsideParallelogram p) : 
  -14 ≤ 2 * p.x - 5 * p.y ∧ 2 * p.x - 5 * p.y ≤ 20 :=
sorry

end range_of_z_in_parallelogram_l19_19736


namespace largest_multiple_of_15_less_than_500_l19_19135

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19135


namespace envelope_weight_l19_19441

theorem envelope_weight (E : ℝ) :
  (8 * (1 / 5) + E ≤ 2) ∧ (1 < 8 * (1 / 5) + E) ∧ (E ≥ 0) ↔ E = 2 / 5 :=
by
  sorry

end envelope_weight_l19_19441


namespace twelve_sided_die_expected_value_l19_19371

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end twelve_sided_die_expected_value_l19_19371


namespace problem_1_problem_2_l19_19877

-- Problem 1: Prove that sqrt(6) * sqrt(1/3) - sqrt(16) * sqrt(18) = -11 * sqrt(2)
theorem problem_1 : Real.sqrt 6 * Real.sqrt (1 / 3) - Real.sqrt 16 * Real.sqrt 18 = -11 * Real.sqrt 2 := 
by
  sorry

-- Problem 2: Prove that (2 - sqrt(5)) * (2 + sqrt(5)) + (2 - sqrt(2))^2 = 5 - 4 * sqrt(2)
theorem problem_2 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * Real.sqrt 2 := 
by
  sorry

end problem_1_problem_2_l19_19877


namespace race_speeds_l19_19613

theorem race_speeds (x y : ℕ) 
  (h1 : 5 * x + 10 = 5 * y) 
  (h2 : 6 * x = 4 * y) :
  x = 4 ∧ y = 6 :=
by {
  -- Proof will go here, but for now we skip it.
  sorry
}

end race_speeds_l19_19613


namespace expected_value_of_twelve_sided_die_l19_19375

noncomputable def expected_value_twelve_sided_die : ℝ :=
  let sides := 12
  let sum_faces := finset.sum (finset.range (sides + 1)) (λ k, k)
  let expected_value := sum_faces / sides
  expected_value

theorem expected_value_of_twelve_sided_die : expected_value_twelve_sided_die = 6.5 :=
by sorry

end expected_value_of_twelve_sided_die_l19_19375


namespace second_divisor_l19_19523

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l19_19523


namespace twenty_less_waiter_slices_eq_28_l19_19276

noncomputable def slices_of_pizza : ℕ := 78
noncomputable def buzz_ratio : ℕ := 5
noncomputable def waiter_ratio : ℕ := 8

theorem twenty_less_waiter_slices_eq_28:
  let total_slices := slices_of_pizza in
  let total_ratio := buzz_ratio + waiter_ratio in
  let waiter_slices := (waiter_ratio * total_slices) / total_ratio in
  waiter_slices - 20 = 28 := by
  sorry

end twenty_less_waiter_slices_eq_28_l19_19276


namespace ellipse_chord_line_eq_l19_19414

noncomputable def chord_line (x y : ℝ) : ℝ := 2 * x + 4 * y - 3

theorem ellipse_chord_line_eq :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 = 1) ∧ (x + y = 1) → (chord_line x y = 0) :=
by
  intros x y h
  sorry

end ellipse_chord_line_eq_l19_19414


namespace largest_multiple_of_15_less_than_500_l19_19148

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19148


namespace sum_three_numbers_is_247_l19_19838

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l19_19838


namespace pool_length_l19_19962

theorem pool_length (r : ℕ) (t : ℕ) (w : ℕ) (d : ℕ) (L : ℕ) 
  (H1 : r = 60)
  (H2 : t = 2000)
  (H3 : w = 80)
  (H4 : d = 10)
  (H5 : L = (r * t) / (w * d)) : L = 150 :=
by
  rw [H1, H2, H3, H4] at H5
  exact H5


end pool_length_l19_19962


namespace sum_of_three_integers_l19_19974

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l19_19974


namespace minimum_value_of_expression_l19_19062

theorem minimum_value_of_expression (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ (x y z : ℝ), (x^2 + (y - 1)^2 + z^2) = 18 / 7 ∧ y = -2 / 7 :=
sorry

end minimum_value_of_expression_l19_19062


namespace find_x_l19_19473

theorem find_x : ∃ x : ℝ, (1 / 3 * ((2 * x + 5) + (8 * x + 3) + (3 * x + 8)) = 5 * x - 10) ∧ x = 23 :=
by
  sorry

end find_x_l19_19473


namespace probability_of_picking_letter_in_mathematics_l19_19039

theorem probability_of_picking_letter_in_mathematics (total_letters : ℕ) (unique_letters : ℕ) (word : list Char)
  (h_total_letters : total_letters = 26)
  (h_unique_letters : unique_letters = 8)
  (h_word : word = "MATHEMATICS".toList) :
  (↑unique_letters / ↑total_letters : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l19_19039


namespace megatek_employees_in_manufacturing_l19_19337

theorem megatek_employees_in_manufacturing :
  let total_degrees := 360
  let manufacturing_degrees := 108
  (manufacturing_degrees / total_degrees.toFloat) * 100 = 30 := 
by
  sorry

end megatek_employees_in_manufacturing_l19_19337


namespace trapezoid_midsegment_l19_19701

theorem trapezoid_midsegment (h : ℝ) :
  ∃ k : ℝ, (∃ θ : ℝ, θ = 120 ∧ k = 2 * h * Real.cos (θ / 2)) ∧
  (∃ m : ℝ, m = k / 2) ∧
  (∃ midsegment : ℝ, midsegment = m / Real.sqrt 3 ∧ midsegment = h / Real.sqrt 3) :=
by
  -- This is where the proof would go.
  sorry

end trapezoid_midsegment_l19_19701


namespace parallelogram_perimeter_l19_19091

def perimeter_of_parallelogram (a b : ℝ) : ℝ :=
  2 * (a + b)

theorem parallelogram_perimeter
  (side1 side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_side2 : side2 = 12) :
  perimeter_of_parallelogram side1 side2 = 60 := 
by
  sorry

end parallelogram_perimeter_l19_19091


namespace find_number_l19_19038

theorem find_number (x : ℝ) (h1 : 0.35 * x = 0.2 * 700) (h2 : 0.2 * 700 = 140) (h3 : 0.35 * x = 140) : x = 400 :=
by sorry

end find_number_l19_19038


namespace trains_total_distance_l19_19097

theorem trains_total_distance (speed_A speed_B : ℝ) (time_A time_B : ℝ) (dist_A dist_B : ℝ):
  speed_A = 90 ∧ 
  speed_B = 120 ∧ 
  time_A = 1 ∧ 
  time_B = 5/6 ∧ 
  dist_A = speed_A * time_A ∧ 
  dist_B = speed_B * time_B ->
  (dist_A + dist_B) = 190 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  sorry

end trains_total_distance_l19_19097


namespace unique_solution_inequality_l19_19913

theorem unique_solution_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) → a = 2 :=
by
  sorry

end unique_solution_inequality_l19_19913


namespace card_total_l19_19439

theorem card_total (Brenda Janet Mara : ℕ)
  (h1 : Janet = Brenda + 9)
  (h2 : Mara = 2 * Janet)
  (h3 : Mara = 150 - 40) :
  Brenda + Janet + Mara = 211 := by
  sorry

end card_total_l19_19439


namespace average_height_of_trees_l19_19296

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l19_19296


namespace range_of_m_if_not_p_and_q_l19_19025

def p (m : ℝ) : Prop := 2 < m

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem range_of_m_if_not_p_and_q (m : ℝ) : ¬ p m ∧ q m → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_if_not_p_and_q_l19_19025


namespace largest_multiple_of_15_less_than_500_l19_19142

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19142


namespace empty_vessel_mass_l19_19658

theorem empty_vessel_mass
  (m1 : ℝ) (m2 : ℝ) (rho_K : ℝ) (rho_B : ℝ) (V : ℝ) (m_c : ℝ)
  (h1 : m1 = m_c + rho_K * V)
  (h2 : m2 = m_c + rho_B * V)
  (h_mass_kerosene : m1 = 31)
  (h_mass_water : m2 = 33)
  (h_rho_K : rho_K = 800)
  (h_rho_B : rho_B = 1000) :
  m_c = 23 :=
by
  -- Proof skipped
  sorry

end empty_vessel_mass_l19_19658


namespace largest_multiple_of_15_less_than_500_l19_19099

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19099


namespace solution_set_of_inequality_l19_19667

theorem solution_set_of_inequality (x : ℝ) : 
  (|x| * (1 - 2 * x) > 0) ↔ (x ∈ ((Set.Iio 0) ∪ (Set.Ioo 0 (1/2)))) :=
by
  sorry

end solution_set_of_inequality_l19_19667


namespace num_friends_solved_problems_l19_19863

theorem num_friends_solved_problems (x y n : ℕ) (h1 : 24 * x + 28 * y = 256) (h2 : n = x + y) : n = 10 :=
by
  -- Begin the placeholder proof
  sorry

end num_friends_solved_problems_l19_19863


namespace principal_amount_l19_19503

theorem principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = 155) (h2 : R = 4.783950617283951) (h3 : T = 4) :
  SI * 100 / (R * T) = 810.13 := 
  by 
    -- proof omitted
    sorry

end principal_amount_l19_19503


namespace probability_same_gate_l19_19985

open Finset

-- Definitions based on the conditions
def num_gates : ℕ := 3
def total_combinations : ℕ := num_gates * num_gates -- total number of combinations for both persons
def favorable_combinations : ℕ := num_gates         -- favorable combinations (both choose same gate)

-- Problem statement
theorem probability_same_gate : 
  ∃ (p : ℚ), p = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧ p = (1 / 3 : ℚ) := 
by
  sorry

end probability_same_gate_l19_19985


namespace largest_multiple_of_15_less_than_500_l19_19197

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19197


namespace percentage_increase_l19_19426

theorem percentage_increase (x y P : ℚ)
  (h1 : x = 0.9 * y)
  (h2 : x = 123.75)
  (h3 : y = 125 + 1.25 * P) : 
  P = 10 := 
by 
  sorry

end percentage_increase_l19_19426


namespace root_sum_product_eq_l19_19813

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l19_19813


namespace find_n_l19_19966

theorem find_n (n : ℕ) (h1 : Nat.gcd n 180 = 12) (h2 : Nat.lcm n 180 = 720) : n = 48 := 
by
  sorry

end find_n_l19_19966


namespace smallest_product_bdf_l19_19942

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l19_19942


namespace largest_multiple_of_15_less_than_500_l19_19168

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19168


namespace remainder_modulus_l19_19504

theorem remainder_modulus :
  (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 :=
by
  sorry

end remainder_modulus_l19_19504


namespace gcd_12012_18018_l19_19574

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19574


namespace sum_of_digits_eleven_l19_19606

-- Definitions for the problem conditions
def distinct_digits (p q r : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ p < 10 ∧ q < 10 ∧ r < 10

def is_two_digit_prime (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.Prime

def concat_digits (x y : Nat) : Nat :=
  10 * x + y

def problem_conditions (p q r : Nat) : Prop :=
  distinct_digits p q r ∧
  is_two_digit_prime (concat_digits p q) ∧
  is_two_digit_prime (concat_digits p r) ∧
  is_two_digit_prime (concat_digits q r) ∧
  (concat_digits p q) * (concat_digits p r) = 221

-- Lean 4 statement to prove the sum of p, q, r is 11
theorem sum_of_digits_eleven (p q r : Nat) (h : problem_conditions p q r) : p + q + r = 11 :=
sorry

end sum_of_digits_eleven_l19_19606


namespace ned_games_l19_19069

theorem ned_games (F: ℕ) (bought_from_friend garage_sale non_working good total_games: ℕ) 
  (h₁: bought_from_friend = F)
  (h₂: garage_sale = 27)
  (h₃: non_working = 74)
  (h₄: good = 3)
  (h₅: total_games = non_working + good)
  (h₆: total_games = bought_from_friend + garage_sale) :
  F = 50 :=
by
  sorry

end ned_games_l19_19069


namespace sequence_property_l19_19620

theorem sequence_property (a : ℕ → ℝ)
    (h_rec : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
    (h_a1 : a 1 = 1 + Real.sqrt 7)
    (h_1776 : a 1776 = 13 + Real.sqrt 7) :
    a 2009 = -1 + 2 * Real.sqrt 7 := 
    sorry

end sequence_property_l19_19620


namespace infinite_n_gcd_floor_sqrt_D_eq_m_l19_19896

theorem infinite_n_gcd_floor_sqrt_D_eq_m (D m : ℕ) (hD : ∀ k : ℕ, k * k ≠ D) (hm : 0 < m) :
  ∃ᶠ n in filter.at_top, Int.gcd n (Nat.floor (Real.sqrt D * n)) = m :=
sorry

end infinite_n_gcd_floor_sqrt_D_eq_m_l19_19896


namespace travel_period_l19_19778

-- Nina's travel pattern
def travels_in_one_month : ℕ := 400
def travels_in_two_months : ℕ := travels_in_one_month + 2 * travels_in_one_month

-- The total distance Nina wants to travel
def total_distance : ℕ := 14400

-- The period in months during which Nina travels the given total distance 
def required_period_in_months (d_per_2_months : ℕ) (total_d : ℕ) : ℕ := (total_d / d_per_2_months) * 2

-- Statement we need to prove
theorem travel_period : required_period_in_months travels_in_two_months total_distance = 24 := by
  sorry

end travel_period_l19_19778


namespace inverse_function_property_l19_19965

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_property (a : ℝ) (h : g a 2 = 4) : f a 2 = 1 := by
  have g_inverse_f : g a (f a 2) = 2 := by sorry
  have a_value : a = 2 := by sorry
  rw [a_value]
  sorry

end inverse_function_property_l19_19965


namespace vieta_formula_l19_19622

-- Define what it means to be a root of a polynomial
noncomputable def is_root (p : ℝ) (a b c d : ℝ) : Prop :=
  a * p^3 + b * p^2 + c * p + d = 0

-- Setting up the variables and conditions for the polynomial
variables (p q r : ℝ)
variable (a b c d : ℝ)
variable (ha : a = 5)
variable (hb : b = -10)
variable (hc : c = 17)
variable (hd : d = -7)
variable (hp : is_root p a b c d)
variable (hq : is_root q a b c d)
variable (hr : is_root r a b c d)

-- Lean statement to prove the desired equality using Vieta's formulas
theorem vieta_formula : 
  pq + qr + rp = c / a :=
by
  -- Translate the problem into Lean structure
  sorry

end vieta_formula_l19_19622


namespace sqrt_720_simplified_l19_19642

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l19_19642


namespace percentage_problem_l19_19425

theorem percentage_problem (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 :=
by
  sorry

end percentage_problem_l19_19425


namespace convert_base_8_to_10_l19_19557

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l19_19557


namespace min_value_a_decreasing_range_of_a_x1_x2_l19_19415

noncomputable def f (a x : ℝ) := x / Real.log x - a * x

theorem min_value_a_decreasing :
  ∀ (a : ℝ), (∀ (x : ℝ), 1 < x → f a x <= 0) → a ≥ 1 / 4 :=
sorry

theorem range_of_a_x1_x2 :
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧ f a x₁ ≤ f a x₂ + a)
  → a ≥ 1 / 2 - 1 / (4 * e^2) :=
sorry

end min_value_a_decreasing_range_of_a_x1_x2_l19_19415


namespace quadratic_distinct_real_roots_range_l19_19608

open Real

theorem quadratic_distinct_real_roots_range (k : ℝ) :
    (∃ a b c : ℝ, a = k^2 ∧ b = 4 * k - 1 ∧ c = 4 ∧ (b^2 - 4 * a * c > 0) ∧ a ≠ 0) ↔ (k < 1 / 8 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l19_19608


namespace find_FC_l19_19020

theorem find_FC 
  (DC CB AD: ℝ)
  (h1 : DC = 9)
  (h2 : CB = 6)
  (h3 : AB = (1 / 3) * AD)
  (h4 : ED = (2 / 3) * AD) :
  FC = 9 :=
sorry

end find_FC_l19_19020


namespace mcgregor_books_finished_l19_19448

def total_books := 89
def floyd_books := 32
def books_left := 23

theorem mcgregor_books_finished : ∀ mg_books : Nat, mg_books = total_books - floyd_books - books_left → mg_books = 34 := 
by
  intro mg_books
  sorry

end mcgregor_books_finished_l19_19448


namespace solve_fraction_equation_l19_19385

theorem solve_fraction_equation (t : ℝ) (h₀ : t ≠ 6) (h₁ : t ≠ -4) :
  (t = -2 ∨ t = -5) ↔ (t^2 - 3 * t - 18) / (t - 6) = 2 / (t + 4) := 
by
  sorry

end solve_fraction_equation_l19_19385


namespace largest_multiple_of_15_less_than_500_l19_19162

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19162


namespace goals_per_player_is_30_l19_19660

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end goals_per_player_is_30_l19_19660


namespace prove_b_eq_d_and_c_eq_e_l19_19321

variable (a b c d e f : ℕ)

-- Define the expressions for A and B as per the problem statement
def A := 10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f
def B := 10^5 * f + 10^4 * d + 10^3 * e + 10^2 * b + 10 * c + a

-- Define the condition that A - B is divisible by 271
def divisible_by_271 (n : ℕ) : Prop := ∃ k : ℕ, n = 271 * k

-- Define the main theorem to prove b = d and c = e under the given conditions
theorem prove_b_eq_d_and_c_eq_e
    (h1 : divisible_by_271 (A a b c d e f - B a b c d e f)) :
    b = d ∧ c = e :=
sorry

end prove_b_eq_d_and_c_eq_e_l19_19321


namespace twelve_sided_die_expected_value_l19_19372

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end twelve_sided_die_expected_value_l19_19372


namespace toys_per_rabbit_l19_19442

-- Define the conditions
def rabbits : ℕ := 34
def toys_mon : ℕ := 8
def toys_tue : ℕ := 3 * toys_mon
def toys_wed : ℕ := 2 * toys_tue
def toys_thu : ℕ := toys_mon
def toys_fri : ℕ := 5 * toys_mon
def toys_sat : ℕ := toys_wed / 2

-- Define the total number of toys
def total_toys : ℕ := toys_mon + toys_tue + toys_wed + toys_thu + toys_fri + toys_sat

-- Define the proof statement
theorem toys_per_rabbit : total_toys / rabbits = 4 :=
by
  -- Proof will go here
  sorry

end toys_per_rabbit_l19_19442


namespace sum_of_sides_of_regular_pentagon_l19_19847

theorem sum_of_sides_of_regular_pentagon (s : ℝ) (n : ℕ)
    (h : s = 15) (hn : n = 5) : 5 * 15 = 75 :=
sorry

end sum_of_sides_of_regular_pentagon_l19_19847


namespace john_saves_water_l19_19299

-- Define the conditions
def old_water_per_flush : ℕ := 5
def num_flushes_per_day : ℕ := 15
def reduction_percentage : ℕ := 80
def days_in_june : ℕ := 30

-- Define the savings calculation
def water_saved_in_june : ℕ :=
  let old_daily_usage := old_water_per_flush * num_flushes_per_day
  let old_june_usage := old_daily_usage * days_in_june
  let new_water_per_flush := old_water_per_flush * (100 - reduction_percentage) / 100
  let new_daily_usage := new_water_per_flush * num_flushes_per_day
  let new_june_usage := new_daily_usage * days_in_june
  old_june_usage - new_june_usage

-- The proof problem statement
theorem john_saves_water : water_saved_in_june = 1800 := 
by
  -- Proof would go here
  sorry

end john_saves_water_l19_19299


namespace sum_three_numbers_is_247_l19_19837

variables (A B C : ℕ)

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ (nat.digits 10 n)

theorem sum_three_numbers_is_247
  (hA : 100 ≤ A ∧ A < 1000) -- A is a three-digit number
  (hB : 10 ≤ B ∧ B < 100)   -- B is a two-digit number
  (hC : 10 ≤ C ∧ C < 100)   -- C is a two-digit number
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
        (if contains_digit A 7 then A else 0) +
        (if contains_digit B 7 then B else 0) +
        (if contains_digit C 7 then C else 0) = 208) -- Sum of numbers containing digit 7 is 208
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧
        (if contains_digit B 3 then B else 0) +
        (if contains_digit C 3 then C else 0) = 76) -- Sum of numbers containing digit 3 is 76
  : A + B + C = 247 := 
sorry

end sum_three_numbers_is_247_l19_19837


namespace inequality_proof_l19_19902

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (a b c x y z : R) (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  x^2 + y^2 + z^2 < a^2 + b^2 + c^2 ∧ x^3 + y^3 + z^3 < a^3 + b^3 + c^3 :=
by
  sorry

end inequality_proof_l19_19902


namespace minimum_groups_l19_19351

theorem minimum_groups (students : ℕ) (max_group_size : ℕ) (h_students : students = 30) (h_max_group_size : max_group_size = 12) : 
  ∃ least_groups : ℕ, least_groups = 3 :=
by
  sorry

end minimum_groups_l19_19351


namespace largest_multiple_of_15_less_than_500_l19_19170

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19170


namespace quadratic_equation_roots_l19_19593

theorem quadratic_equation_roots (a b c : ℝ) (h_a_nonzero : a ≠ 0) 
  (h_roots : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1) : 
  a + b + c = 0 ∧ b = 0 :=
by
  -- Using Vieta's formulas and the properties given, we should show:
  -- h_roots means the sum of roots = -(b/a) = 0 → b = 0
  -- and the product of roots = (c/a) = -1/a → c = -a
  -- Substituting these into ax^2 + bx + c = 0 should give us:
  -- a + b + c = 0 → we need to show both parts to complete the proof.
  sorry

end quadratic_equation_roots_l19_19593


namespace solve_for_n_l19_19278

theorem solve_for_n (n : ℕ) : 4^8 = 16^n → n = 4 :=
by
  sorry

end solve_for_n_l19_19278


namespace smallest_a_divisible_by_65_l19_19623

theorem smallest_a_divisible_by_65 (a : ℤ) 
  (h : ∀ (n : ℤ), (5 * n ^ 13 + 13 * n ^ 5 + 9 * a * n) % 65 = 0) : 
  a = 63 := 
by {
  sorry
}

end smallest_a_divisible_by_65_l19_19623


namespace candidate_lost_by_2460_votes_l19_19519

noncomputable def total_votes : ℝ := 8199.999999999998
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def rival_percentage : ℝ := 1 - candidate_percentage
noncomputable def candidate_votes := candidate_percentage * total_votes
noncomputable def rival_votes := rival_percentage * total_votes
noncomputable def votes_lost_by := rival_votes - candidate_votes

theorem candidate_lost_by_2460_votes : votes_lost_by = 2460 := by
  sorry

end candidate_lost_by_2460_votes_l19_19519


namespace brother_age_l19_19530

variables (M B : ℕ)

theorem brother_age (h1 : M = B + 12) (h2 : M + 2 = 2 * (B + 2)) : B = 10 := by
  sorry

end brother_age_l19_19530


namespace value_of_D_l19_19322

variable (L E A D : ℤ)

-- given conditions
def LEAD := 41
def DEAL := 45
def ADDED := 53

-- condition that L = 15
axiom hL : L = 15

-- equations from the problem statement
def eq1 := L + E + A + D = 41
def eq2 := D + E + A + L = 45
def eq3 := A + 3 * D + E = 53

-- stating the problem as proving that D = 4 given the conditions
theorem value_of_D : D = 4 :=
by
  sorry

end value_of_D_l19_19322


namespace largest_multiple_of_15_less_than_500_l19_19137

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19137


namespace gcd_12012_18018_l19_19578

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19578


namespace circle_circumference_l19_19494

noncomputable def circumference_of_circle (speed1 speed2 time : ℝ) : ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  distance1 + distance2

theorem circle_circumference
    (speed1 speed2 time : ℝ)
    (h1 : speed1 = 7)
    (h2 : speed2 = 8)
    (h3 : time = 12) :
    circumference_of_circle speed1 speed2 time = 180 := by
  sorry

end circle_circumference_l19_19494


namespace find_z_plus_one_over_y_l19_19466

variable {x y z : ℝ}

theorem find_z_plus_one_over_y (h1 : x * y * z = 1) 
                                (h2 : x + 1 / z = 7) 
                                (h3 : y + 1 / x = 31) 
                                (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : 
                              z + 1 / y = 5 / 27 := 
by
  sorry

end find_z_plus_one_over_y_l19_19466


namespace basketball_team_win_requirement_l19_19861

theorem basketball_team_win_requirement :
  ∀ (initial_wins : ℕ) (initial_games : ℕ) (total_games : ℕ) (target_win_rate : ℚ) (total_wins : ℕ),
    initial_wins = 30 →
    initial_games = 60 →
    total_games = 100 →
    target_win_rate = 65 / 100 →
    total_wins = total_games * target_win_rate →
    total_wins - initial_wins = 35 :=
by
  -- variables and hypotheses declaration are omitted
  sorry

end basketball_team_win_requirement_l19_19861


namespace a_2016_mod_2017_l19_19485

-- Defining the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧
  a 1 = 2 ∧
  ∀ n, a (n + 2) = 2 * a (n + 1) + 41 * a n

theorem a_2016_mod_2017 (a : ℕ → ℕ) (h : seq a) : 
  a 2016 % 2017 = 0 := 
sorry

end a_2016_mod_2017_l19_19485


namespace train_length_is_350_meters_l19_19853

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let time_hr := time_sec / 3600
  speed_kmh * time_hr * 1000

theorem train_length_is_350_meters :
  length_of_train 60 21 = 350 :=
by
  sorry

end train_length_is_350_meters_l19_19853


namespace fish_to_rice_equivalence_l19_19757

variable (f : ℚ) (l : ℚ)

theorem fish_to_rice_equivalence (h1 : 5 * f = 3 * l) (h2 : l = 6) : f = 18 / 5 := by
  sorry

end fish_to_rice_equivalence_l19_19757


namespace total_books_l19_19274

def school_books : ℕ := 19
def sports_books : ℕ := 39

theorem total_books : school_books + sports_books = 58 := by
  sorry

end total_books_l19_19274


namespace find_number_l19_19395

theorem find_number (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 -> x = 9 :=
by
  sorry

end find_number_l19_19395


namespace kids_tubing_and_rafting_l19_19491

theorem kids_tubing_and_rafting 
  (total_kids : ℕ) 
  (one_fourth_tubing : ℕ)
  (half_rafting : ℕ)
  (h1 : total_kids = 40)
  (h2 : one_fourth_tubing = total_kids / 4)
  (h3 : half_rafting = one_fourth_tubing / 2) :
  half_rafting = 5 :=
by
  sorry

end kids_tubing_and_rafting_l19_19491


namespace finite_solutions_l19_19018

variable (a b : ℕ) (h1 : a ≠ b)

theorem finite_solutions (a b : ℕ) (h1 : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
  (x * y + z * w = a) ∧ (x * z + y * w = b) →
  (x, y, z, w) ∈ S :=
sorry

end finite_solutions_l19_19018


namespace min_value_2013_Quanzhou_simulation_l19_19684

theorem min_value_2013_Quanzhou_simulation:
  ∃ (x y : ℝ), (x - y - 1 = 0) ∧ (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
by
  use 2
  use 3
  sorry

end min_value_2013_Quanzhou_simulation_l19_19684


namespace surface_area_of_cone_l19_19753

-- Definitions based solely on conditions
def central_angle (θ : ℝ) := θ = (2 * Real.pi) / 3
def slant_height (l : ℝ) := l = 2
def radius_cone (r : ℝ) := ∃ (θ l : ℝ), central_angle θ ∧ slant_height l ∧ θ * l = 2 * Real.pi * r
def lateral_surface_area (A₁ : ℝ) (r l : ℝ) := A₁ = Real.pi * r * l
def base_area (A₂ : ℝ) (r : ℝ) := A₂ = Real.pi * r^2
def total_surface_area (A A₁ A₂ : ℝ) := A = A₁ + A₂

-- The theorem proving the total surface area is as specified
theorem surface_area_of_cone :
  ∃ (r l A₁ A₂ A : ℝ), central_angle ((2 * Real.pi) / 3) ∧ slant_height 2 ∧ radius_cone r ∧
  lateral_surface_area A₁ r 2 ∧ base_area A₂ r ∧ total_surface_area A A₁ A₂ ∧ A = (16 * Real.pi) / 9 := sorry

end surface_area_of_cone_l19_19753


namespace four_digit_divisors_l19_19723

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l19_19723


namespace independent_events_probability_l19_19996

variables (A B : Type) (P : Set A → ℚ)
-- Conditions
variables (hA : P {a | a = a} = 5/7)
variables (hB : P {b | b = b} = 2/5)
variables (indep : ∀ (A B : Set A), P (A ∩ B) = P A * P B)

-- Statement
theorem independent_events_probability (A B : Set A) (P : Set A → ℚ)
  (hA : P A = 5 / 7)
  (hB : P B = 2 / 5)
  (indep : P (A ∩ B) = P A * P B) :
  P (A ∩ B) = 2 / 7 :=
by sorry

end independent_events_probability_l19_19996


namespace remainder_of_sum_l19_19631

theorem remainder_of_sum (k j : ℤ) (a b : ℤ) (h₁ : a = 60 * k + 53) (h₂ : b = 45 * j + 17) : ((a + b) % 15) = 5 :=
by
  sorry

end remainder_of_sum_l19_19631


namespace ball_first_less_than_25_cm_l19_19860

theorem ball_first_less_than_25_cm (n : ℕ) :
  ∀ n, (200 : ℝ) * (3 / 4) ^ n < 25 ↔ n ≥ 6 := by sorry

end ball_first_less_than_25_cm_l19_19860


namespace flower_counts_l19_19549

theorem flower_counts (R G Y : ℕ) : (R + G = 62) → (R + Y = 49) → (G + Y = 77) → R = 17 ∧ G = 45 ∧ Y = 32 :=
by
  intros h1 h2 h3
  sorry

end flower_counts_l19_19549


namespace largest_multiple_of_15_less_than_500_l19_19134

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19134


namespace set_C_cannot_form_right_triangle_l19_19239

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_C_cannot_form_right_triangle :
  ¬ is_right_triangle 7 8 9 :=
by
  sorry

end set_C_cannot_form_right_triangle_l19_19239


namespace M_inter_N_eq_l19_19906

open Set

def M : Set ℝ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 < n ∧ n ≤ 3 }

theorem M_inter_N_eq : M ∩ (coe '' N) = {0, 1} :=
by sorry

end M_inter_N_eq_l19_19906


namespace maximize_Sn_l19_19409

theorem maximize_Sn (a1 : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : a1 > 0)
  (h2 : a1 + 9 * (a1 + 5 * d) = 0)
  (h_sn : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)) :
  ∃ n_max, ∀ n, S n ≤ S n_max ∧ n_max = 5 :=
by
  sorry

end maximize_Sn_l19_19409


namespace compute_five_fold_application_l19_19065

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -16 :=
by
  sorry

end compute_five_fold_application_l19_19065


namespace number_of_pipes_l19_19547

theorem number_of_pipes (d_large d_small: ℝ) (π : ℝ) (h1: d_large = 4) (h2: d_small = 2) : 
  ((π * (d_large / 2)^2) / (π * (d_small / 2)^2) = 4) := 
by
  sorry

end number_of_pipes_l19_19547


namespace ab_value_l19_19335

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end ab_value_l19_19335


namespace lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l19_19741

noncomputable def lucky_point (m n : ℝ) : Prop := 2 * m = 4 + n ∧ ∃ (x y : ℝ), (x = m - 1) ∧ (y = (n + 2) / 2)

theorem lucky_point_m2 :
  lucky_point 2 0 := sorry

theorem is_lucky_point_A33 :
  lucky_point 4 4 := sorry

theorem point_M_quadrant (a : ℝ) :
  lucky_point (a + 1) (2 * (2 * a - 1) - 2) → (a = 1) := sorry

end lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l19_19741


namespace fraction_of_clerical_staff_is_one_third_l19_19229

-- Defining the conditions
variables (employees clerical_f clerical employees_reduced employees_remaining : ℝ)

def company_conditions (employees clerical_f clerical employees_reduced employees_remaining : ℝ) : Prop :=
  employees = 3600 ∧
  clerical = 3600 * clerical_f ∧
  employees_reduced = clerical * (2 / 3) ∧
  employees_remaining = employees - clerical * (1 / 3) ∧
  employees_reduced = 0.25 * employees_remaining

-- The statement to prove the fraction of clerical employees given the conditions
theorem fraction_of_clerical_staff_is_one_third
  (hc : company_conditions employees clerical_f clerical employees_reduced employees_remaining) :
  clerical_f = 1 / 3 :=
sorry

end fraction_of_clerical_staff_is_one_third_l19_19229


namespace sum_of_x_and_y_l19_19786

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end sum_of_x_and_y_l19_19786


namespace simplify_and_evaluate_expr_l19_19312

theorem simplify_and_evaluate_expr (x : ℤ) (h : x = -2) : 
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -8 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l19_19312


namespace range_of_a_l19_19428

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x_0 : ℝ, x_0^2 + (a - 1) * x_0 + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_l19_19428


namespace max_value_of_t_l19_19064

variable (n r t : ℕ)
variable (A : Finset (Finset (Fin n)))
variable (h₁ : n ≤ 2 * r)
variable (h₂ : ∀ s ∈ A, Finset.card s = r)
variable (h₃ : Finset.card A = t)

theorem max_value_of_t : 
  (n < 2 * r → t ≤ Nat.choose n r) ∧ 
  (n = 2 * r → t ≤ Nat.choose n r / 2) :=
by
  sorry

end max_value_of_t_l19_19064


namespace conversion_base8_to_base10_l19_19553

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l19_19553


namespace expected_value_of_twelve_sided_die_l19_19359

theorem expected_value_of_twelve_sided_die : ∃ E : ℝ, E = 6.5 :=
by
  let n := 12
  let sum_faces := n * (n + 1) / 2
  let E := sum_faces / n
  use E
  -- The sum of the first 12 natural numbers should be calculated: 1 + 2 + ... + 12 = 78
  have sum_calculated : sum_faces = 78 := by compute
  rw [sum_calculated]
  -- The expected value is hence 78 / 12
  have E_calculated : E = 78 / 12 := by compute
  rw [E_calculated]
  -- Finally, 78 / 12 simplifies to 6.5
  have E_final : 78 / 12 = 6.5 := by compute
  rw [E_final]

-- sorry is added for place holder to validate the step is proof-skipped.

end expected_value_of_twelve_sided_die_l19_19359


namespace regression_analysis_notes_l19_19987

-- Define the conditions
def applicable_population (reg_eq: Type) (sample: Type) : Prop := sorry
def temporality (reg_eq: Type) : Prop := sorry
def sample_value_range_influence (reg_eq: Type) (sample: Type) : Prop := sorry
def prediction_precision (reg_eq: Type) : Prop := sorry

-- Define the key points to note
def key_points_to_note (reg_eq: Type) (sample: Type) : Prop :=
  applicable_population reg_eq sample ∧
  temporality reg_eq ∧
  sample_value_range_influence reg_eq sample ∧
  prediction_precision reg_eq

-- The main statement
theorem regression_analysis_notes (reg_eq: Type) (sample: Type) :
  key_points_to_note reg_eq sample := sorry

end regression_analysis_notes_l19_19987


namespace largest_multiple_15_under_500_l19_19125

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19125


namespace find_x_2187_l19_19899

theorem find_x_2187 (x : ℂ) (h : x - 1/x = complex.I * real.sqrt 3) : x^2187 - 1/(x^2187) = 0 :=
sorry

end find_x_2187_l19_19899


namespace tan_alpha_eq_neg_sqrt_15_l19_19420

/-- Given α in the interval (0, π) and the equation tan(2α) = sin(α) / (2 + cos(α)), prove that tan(α) = -√15. -/
theorem tan_alpha_eq_neg_sqrt_15 (α : ℝ) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end tan_alpha_eq_neg_sqrt_15_l19_19420


namespace length_of_platform_is_280_l19_19681

-- Add conditions for speed, times and conversions
def speed_kmph : ℕ := 72
def time_platform : ℕ := 30
def time_man : ℕ := 16

-- Conversion from km/h to m/s
def speed_mps : ℤ := speed_kmph * 1000 / 3600

-- The length of the train when it crosses the man
def length_of_train : ℤ := speed_mps * time_man

-- The length of the platform
def length_of_platform : ℤ := (speed_mps * time_platform) - length_of_train

theorem length_of_platform_is_280 :
  length_of_platform = 280 := by
  sorry

end length_of_platform_is_280_l19_19681


namespace find_x_squared_add_y_squared_l19_19011

noncomputable def x_squared_add_y_squared (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem find_x_squared_add_y_squared (x y : ℝ) 
  (h1 : x + y = 48)
  (h2 : x * y = 168) :
  x_squared_add_y_squared x y = 1968 :=
by
  sorry

end find_x_squared_add_y_squared_l19_19011


namespace two_coins_heads_probability_l19_19678

/-- 
When tossing two coins of uniform density, the probability that both coins land with heads facing up is 1/4.
-/
theorem two_coins_heads_probability : 
  let outcomes := ["HH", "HT", "TH", "TT"]
  let favorable := "HH"
  probability (favorable) = 1/4 :=
by
  sorry

end two_coins_heads_probability_l19_19678


namespace daily_sacks_per_section_l19_19477

theorem daily_sacks_per_section (harvests sections : ℕ) (h_harvests : harvests = 360) (h_sections : sections = 8) : harvests / sections = 45 := by
  sorry

end daily_sacks_per_section_l19_19477


namespace ratio_of_areas_ACP_BQA_l19_19618

open EuclideanGeometry

-- Define the geometric configuration
variables (A B C D P Q : Point)
  (is_square : square A B C D)
  (is_bisector_CAD : is_angle_bisector A C D P)
  (is_bisector_ABD : is_angle_bisector B A D Q)

-- Define the areas of triangles
def area_triangle (X Y Z : Point) : Real := sorry -- Placeholder for the area function

-- Lean statement for the proof problem
theorem ratio_of_areas_ACP_BQA 
  (h_square : is_square) 
  (h_bisector_CAD : is_bisector_CAD) 
  (h_bisector_ABD : is_bisector_ABD) :
  (area_triangle A C P) / (area_triangle B Q A) = 2 :=
sorry

end ratio_of_areas_ACP_BQA_l19_19618


namespace vince_bus_ride_distance_l19_19326

/-- 
  Vince's bus ride to school is 0.625 mile, 
  given that Zachary's bus ride is 0.5 mile 
  and Vince's bus ride is 0.125 mile longer than Zachary's.
--/
theorem vince_bus_ride_distance (zachary_ride : ℝ) (vince_longer : ℝ) 
  (h1 : zachary_ride = 0.5) (h2 : vince_longer = 0.125) 
  : zachary_ride + vince_longer = 0.625 :=
by sorry

end vince_bus_ride_distance_l19_19326


namespace num_bases_ending_in_1_l19_19892

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l19_19892


namespace sqrt_720_eq_12_sqrt_5_l19_19644

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l19_19644


namespace kids_tubing_and_rafting_l19_19492

theorem kids_tubing_and_rafting 
  (total_kids : ℕ) 
  (one_fourth_tubing : ℕ)
  (half_rafting : ℕ)
  (h1 : total_kids = 40)
  (h2 : one_fourth_tubing = total_kids / 4)
  (h3 : half_rafting = one_fourth_tubing / 2) :
  half_rafting = 5 :=
by
  sorry

end kids_tubing_and_rafting_l19_19492


namespace arithmetic_sequence_term_l19_19292

theorem arithmetic_sequence_term (a : ℕ → ℤ) (d : ℤ) (n : ℕ) :
  a 5 = 33 ∧ a 45 = 153 ∧ (∀ n, a n = a 1 + (n - 1) * d) ∧ a n = 201 → n = 61 :=
by
  sorry

end arithmetic_sequence_term_l19_19292


namespace largest_multiple_of_15_less_than_500_l19_19184

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19184


namespace largest_multiple_of_15_less_than_500_l19_19202

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19202


namespace mrs_hilt_total_payment_l19_19451

-- Define the conditions
def number_of_hot_dogs : ℕ := 6
def cost_per_hot_dog : ℝ := 0.50

-- Define the total cost
def total_cost : ℝ := number_of_hot_dogs * cost_per_hot_dog

-- State the theorem to prove the total cost
theorem mrs_hilt_total_payment : total_cost = 3.00 := 
by
  sorry

end mrs_hilt_total_payment_l19_19451


namespace robin_extra_drinks_l19_19458

-- Conditions
def initial_sodas : ℕ := 22
def initial_energy_drinks : ℕ := 15
def initial_smoothies : ℕ := 12
def drank_sodas : ℕ := 6
def drank_energy_drinks : ℕ := 9
def drank_smoothies : ℕ := 2

-- Total drinks bought
def total_drinks_bought : ℕ :=
  initial_sodas + initial_energy_drinks + initial_smoothies
  
-- Total drinks consumed
def total_drinks_consumed : ℕ :=
  drank_sodas + drank_energy_drinks + drank_smoothies

-- Number of extra drinks
def extra_drinks : ℕ :=
  total_drinks_bought - total_drinks_consumed

-- Theorem to prove
theorem robin_extra_drinks : extra_drinks = 32 :=
  by
  -- skipping the proof
  sorry

end robin_extra_drinks_l19_19458


namespace part1_part2_l19_19406

open Complex

-- Define the first proposition p
def p (m : ℝ) : Prop :=
  (m - 1 < 0) ∧ (m + 3 > 0)

-- Define the second proposition q
def q (m : ℝ) : Prop :=
  abs (Complex.mk 1 (m - 2)) ≤ Real.sqrt 10

-- Prove the first part of the problem
theorem part1 (m : ℝ) (hp : p m) : -3 < m ∧ m < 1 :=
sorry

-- Prove the second part of the problem
theorem part2 (m : ℝ) (h : ¬ (p m ∧ q m) ∧ (p m ∨ q m)) : (-3 < m ∧ m < -1) ∨ (1 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_l19_19406


namespace values_of_a_and_b_l19_19767

theorem values_of_a_and_b (a b : ℝ) 
  (hT : (2, 1) ∈ {p : ℝ × ℝ | ∃ (a : ℝ), p.1 * a + p.2 - 3 = 0})
  (hS : (2, 1) ∈ {p : ℝ × ℝ | ∃ (b : ℝ), p.1 - p.2 - b = 0}) :
  a = 1 ∧ b = 1 :=
by
  sorry

end values_of_a_and_b_l19_19767


namespace students_exceed_pets_l19_19006

-- Defining the conditions
def num_students_per_classroom := 25
def num_rabbits_per_classroom := 3
def num_guinea_pigs_per_classroom := 3
def num_classrooms := 5

-- Main theorem to prove
theorem students_exceed_pets:
  let total_students := num_students_per_classroom * num_classrooms
  let total_rabbits := num_rabbits_per_classroom * num_classrooms
  let total_guinea_pigs := num_guinea_pigs_per_classroom * num_classrooms
  let total_pets := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 :=
by 
  sorry

end students_exceed_pets_l19_19006


namespace range_of_phi_l19_19478

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + 2 * φ)

theorem range_of_phi :
  ∀ φ : ℝ,
  (0 < φ) ∧ (φ < π / 2) →
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → g x φ ≤ g (x + π/6) φ) →
  (∃ x : ℝ, -π/6 < x ∧ x < 0 ∧ g x φ = 0) →
  φ ∈ Set.Ioc (π / 4) (π / 3) := 
by
  intros φ h1 h2 h3
  sorry

end range_of_phi_l19_19478


namespace doubled_cost_percent_l19_19082

-- Definitions
variable (t b : ℝ)
def cost (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percent :
  cost t (2 * b) = 16 * cost t b :=
by
  -- To be proved
  sorry

end doubled_cost_percent_l19_19082


namespace triangles_not_necessarily_symmetric_l19_19095

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A1 : Point)
(A2 : Point)
(A3 : Point)

structure Ellipse :=
(a : ℝ) -- semi-major axis
(b : ℝ) -- semi-minor axis

def inscribed_in (T : Triangle) (E : Ellipse) : Prop :=
  -- Assuming the definition of the inscribed, can be encoded based on the ellipse equation: x^2/a^2 + y^2/b^2 <= 1 for each vertex.
  sorry

def symmetric_wrt_axis (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to an axis (to be defined)
  sorry

def symmetric_wrt_center (T₁ T₂ : Triangle) : Prop :=
  -- Definition of symmetry with respect to the center (to be defined)
  sorry

theorem triangles_not_necessarily_symmetric {E : Ellipse} {T₁ T₂ : Triangle}
  (h₁ : inscribed_in T₁ E) (h₂ : inscribed_in T₂ E) (heq : T₁ = T₂) :
  ¬ symmetric_wrt_axis T₁ T₂ ∧ ¬ symmetric_wrt_center T₁ T₂ :=
sorry

end triangles_not_necessarily_symmetric_l19_19095


namespace daily_serving_size_l19_19805

-- Definitions based on problem conditions
def days : ℕ := 180
def capsules_per_bottle : ℕ := 60
def bottles : ℕ := 6
def total_capsules : ℕ := bottles * capsules_per_bottle

-- Theorem statement to prove the daily serving size
theorem daily_serving_size :
  total_capsules / days = 2 := by
  sorry

end daily_serving_size_l19_19805


namespace honey_harvested_correct_l19_19248

def honey_harvested_last_year : ℕ := 2479
def honey_increase_this_year : ℕ := 6085
def honey_harvested_this_year : ℕ := 8564

theorem honey_harvested_correct :
  honey_harvested_last_year + honey_increase_this_year = honey_harvested_this_year :=
sorry

end honey_harvested_correct_l19_19248


namespace largest_multiple_of_15_less_than_500_l19_19201

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l19_19201


namespace gcd_12012_18018_l19_19577

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19577


namespace total_amount_shared_l19_19862

-- Define the initial conditions
def ratioJohn : ℕ := 2
def ratioJose : ℕ := 4
def ratioBinoy : ℕ := 6
def JohnShare : ℕ := 2000
def partValue : ℕ := JohnShare / ratioJohn

-- Define the shares based on the ratio and part value
def JoseShare := ratioJose * partValue
def BinoyShare := ratioBinoy * partValue

-- Prove the total amount shared is Rs. 12000
theorem total_amount_shared : (JohnShare + JoseShare + BinoyShare) = 12000 :=
  by
  sorry

end total_amount_shared_l19_19862


namespace value_of_a_l19_19019

theorem value_of_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
by
  sorry

end value_of_a_l19_19019


namespace inequality_proof_l19_19890

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
    (a * b + b * c + c * a) * (1 / (a + b)^2 + 1 / (b + c)^2 + 1 / (c + a)^2) ≥ 9 / 4 := 
by
  sorry

end inequality_proof_l19_19890


namespace expected_value_of_twelve_sided_die_l19_19362

theorem expected_value_of_twelve_sided_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  (list.sum faces / list.length faces : ℝ) = 6.5 := 
by
  -- problem defines that there are 12 faces of the die numbered from 1 to 12
  have h_faces_length : list.length faces = 12 := rfl
  -- problem defines the sum of faces computed using the series sum formula
  have h_faces_sum : list.sum faces = 78 := rfl
  exact sorry

end expected_value_of_twelve_sided_die_l19_19362


namespace sqrt_720_simplified_l19_19641

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l19_19641


namespace doughnuts_per_person_l19_19459

theorem doughnuts_per_person :
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  total_doughnuts / total_people = 6 := 
by
  -- Definitions and conditions from the problem
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  -- Goal to prove
  show total_doughnuts / total_people = 6
  sorry

end doughnuts_per_person_l19_19459


namespace find_AC_length_l19_19617

theorem find_AC_length (AB BC CD DA : ℕ) 
  (hAB : AB = 10) (hBC : BC = 9) (hCD : CD = 19) (hDA : DA = 5) : 
  14 < AC ∧ AC < 19 → AC = 15 := 
by
  sorry

end find_AC_length_l19_19617


namespace ratio_to_percent_l19_19323

theorem ratio_to_percent (a b : ℕ) (h : a = 6) (h2 : b = 3) :
  ((a / b : ℚ) * 100 = 200) :=
by
  have h3 : a = 6 := h
  have h4 : b = 3 := h2
  sorry

end ratio_to_percent_l19_19323


namespace find_n_l19_19898

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = 1 / 2 * (Real.log n - 2)) :
  n = Real.exp 2 + 2 :=
by
  sorry

end find_n_l19_19898


namespace luke_bought_stickers_l19_19774

theorem luke_bought_stickers :
  ∀ (original birthday given_to_sister used_on_card left total_before_buying stickers_bought : ℕ),
  original = 20 →
  birthday = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  left = 39 →
  total_before_buying = original + birthday →
  stickers_bought = (left + given_to_sister + used_on_card) - total_before_buying →
  stickers_bought = 12 :=
by
  intros
  sorry

end luke_bought_stickers_l19_19774


namespace age_impossibility_l19_19376

/-
Problem statement:
Ann is 5 years older than Kristine.
Their current ages sum up to 24.
Prove that it's impossible for both their ages to be whole numbers.
-/

theorem age_impossibility 
  (K A : ℕ) -- Kristine's and Ann's ages are natural numbers
  (h1 : A = K + 5) -- Ann is 5 years older than Kristine
  (h2 : K + A = 24) -- their combined age is 24
  : false := sorry

end age_impossibility_l19_19376


namespace solution_set_inequality_l19_19816

theorem solution_set_inequality :
  {x : ℝ | (x^2 - 4) * (x - 6)^2 ≤ 0} = {x : ℝ | (-2 ≤ x ∧ x ≤ 2) ∨ x = 6} :=
  sorry

end solution_set_inequality_l19_19816


namespace sqrt_expression_eq_l19_19247

theorem sqrt_expression_eq :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := 
  sorry

end sqrt_expression_eq_l19_19247


namespace largest_multiple_of_15_less_than_500_l19_19106

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19106


namespace gcd_of_12012_18018_l19_19579

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l19_19579


namespace find_overlap_length_l19_19086

-- Definitions of the given conditions
def total_length_of_segments := 98 -- cm
def edge_to_edge_distance := 83 -- cm
def number_of_overlaps := 6

-- Theorem stating the value of x in centimeters
theorem find_overlap_length (x : ℝ) 
  (h1 : total_length_of_segments = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : number_of_overlaps = 6) 
  (h4 : total_length_of_segments = edge_to_edge_distance + number_of_overlaps * x) : 
  x = 2.5 :=
  sorry

end find_overlap_length_l19_19086


namespace circle_equation_exists_l19_19386

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

end circle_equation_exists_l19_19386


namespace sum_of_numbers_l19_19836

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19836


namespace matt_revenue_l19_19935

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def grams_peanuts (area : ℕ) (g_per_sqft : ℕ) : ℕ := area * g_per_sqft

def kg_peanuts (grams : ℕ) : ℕ := grams / 1000

def grams_peanut_butter (grams_peanuts : ℕ) : ℕ := (grams_peanuts / 20) * 5

def kg_peanut_butter (grams_pb : ℕ) : ℕ := grams_pb / 1000

def revenue (kg_pb : ℕ) (price_per_kg : ℕ) : ℕ := kg_pb * price_per_kg

theorem matt_revenue : 
  let length := 500
  let width := 500
  let g_per_sqft := 50
  let conversion_ratio := 20 / 5
  let price_per_kg := 10
  revenue (kg_peanut_butter (grams_peanut_butter (grams_peanuts (area length width) g_per_sqft))) price_per_kg = 31250 := by
  sorry

end matt_revenue_l19_19935


namespace sum_of_numbers_is_247_l19_19823

/-- Definitions of the conditions -/
def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n / 10 ^ k % 10 = d

variable (A B C : ℕ)
variable (hA : 100 ≤ A ∧ A < 1000)
variable (hB : 10 ≤ B ∧ B < 100)
variable (hC : 10 ≤ C ∧ C < 100)
variable (h_sum_7 : if number_contains_digit A 7 
                  then if number_contains_digit B 7 
                  then if number_contains_digit C 7 
                  then A + B + C 
                  else A + B
                  else A
                  else B + C = 208)
variable (h_sum_3 : if number_contains_digit A 3 
                  then if number_contains_digit B 3
                  then if number_contains_digit C 3
                  then A + B + C 
                  else A + B
                  else A 
                  else B + C = 76)

/-- Prove that the sum of all three numbers is 247 -/
theorem sum_of_numbers_is_247 : A + B + C = 247 :=
by
  sorry

end sum_of_numbers_is_247_l19_19823


namespace find_length_of_smaller_rectangle_l19_19225

theorem find_length_of_smaller_rectangle
  (w : ℝ)
  (h_original : 10 * 15 = 150)
  (h_new_rectangle : 2 * w * w = 150)
  (h_z : w = 5 * Real.sqrt 3) :
  z = 5 * Real.sqrt 3 :=
by
  sorry

end find_length_of_smaller_rectangle_l19_19225


namespace find_b_l19_19045

theorem find_b
  (b : ℝ)
  (hx : ∃ y : ℝ, 4 * 3 + 2 * y = b ∧ 3 * 3 + 4 * y = 3 * b) :
  b = -15 :=
sorry

end find_b_l19_19045


namespace number_of_boys_l19_19049

variables (total_girls total_teachers total_people : ℕ)
variables (total_girls_eq : total_girls = 315) (total_teachers_eq : total_teachers = 772) (total_people_eq : total_people = 1396)

theorem number_of_boys (total_boys : ℕ) : total_boys = total_people - total_girls - total_teachers :=
by sorry

end number_of_boys_l19_19049


namespace largest_multiple_15_under_500_l19_19120

-- Define the problem statement
theorem largest_multiple_15_under_500 : ∃ (n : ℕ), n < 500 ∧ n % 15 = 0 ∧ (∀ m, m < 500 ∧ m % 15 = 0 → m ≤ n) :=
by {
  use 495,
  split,
  {
    exact lt_of_le_of_ne (le_refl _) (ne_of_lt (nat.lt_succ_self 499))
  },
  split,
  {
    exact nat.mod_eq_zero_of_dvd ⟨33, rfl⟩,
  },
  {
    intros m hm h,
    have hmn : m ≤ 495,
    {
      have div15 := hm/15,
      suffices : div15 < 33+1, from nat.le_of_lt_add_one this,
      exact this, sorry,
    },
    exact hmn,
  },
}

end largest_multiple_15_under_500_l19_19120


namespace sheepdog_catches_sheep_l19_19306

-- Define the speeds and the time taken
def v_s : ℝ := 12 -- speed of the sheep in feet/second
def v_d : ℝ := 20 -- speed of the sheepdog in feet/second
def t : ℝ := 20 -- time in seconds

-- Define the initial distance between the sheep and the sheepdog
def initial_distance (v_s v_d t : ℝ) : ℝ :=
  v_d * t - v_s * t

theorem sheepdog_catches_sheep :
  initial_distance v_s v_d t = 160 :=
by
  -- The formal proof would go here, but for now we replace it with sorry
  sorry

end sheepdog_catches_sheep_l19_19306


namespace largest_multiple_of_15_less_than_500_l19_19112

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end largest_multiple_of_15_less_than_500_l19_19112


namespace largest_multiple_of_15_less_than_500_l19_19190

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19190


namespace CE_squared_plus_DE_squared_proof_l19_19768

noncomputable def CE_squared_plus_DE_squared (radius : ℝ) (diameter : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  if radius = 10 ∧ diameter = 20 ∧ BE = 4 ∧ angle_AEC = 30 then 200 else sorry

theorem CE_squared_plus_DE_squared_proof : CE_squared_plus_DE_squared 10 20 4 30 = 200 := by
  sorry

end CE_squared_plus_DE_squared_proof_l19_19768


namespace greatest_prime_factor_of_expression_l19_19502

theorem greatest_prime_factor_of_expression :
  ∃ p : ℕ, p.prime ∧ p = 131 ∧ ∀ q : ℕ, q.prime → q ∣ (3^8 + 6^7) → q ≤ 131 :=
by {
  have h : 3^8 + 6^7 = 3^7 * 131,
  { sorry }, -- proving the factorization
  have prime_131 : prime 131,
  { sorry }, -- proving 131 is prime
  use 131,
  refine ⟨prime_131, rfl, _⟩,
  intros q q_prime q_divides,
  rw h at q_divides,
  cases prime_factors.unique _ q_prime q_divides with k hk,
  sorry -- proving q ≤ 131
}

end greatest_prime_factor_of_expression_l19_19502


namespace largest_multiple_of_15_below_500_l19_19179

theorem largest_multiple_of_15_below_500 : ∃ (k : ℕ), 15 * k < 500 ∧ ∀ (m : ℕ), 15 * m < 500 → 15 * m ≤ 15 * k := 
by
  existsi 33
  split
  · norm_num
  · intro m h
    have h1 : m ≤ 33 := Nat.le_of_lt_succ (Nat.div_lt_succ 500 15 m)
    exact (mul_le_mul_left (by norm_num : 0 < 15)).mpr h1
  sorry

end largest_multiple_of_15_below_500_l19_19179


namespace sufficient_condition_for_reciprocal_inequality_l19_19421

variable (a b : ℝ)

theorem sufficient_condition_for_reciprocal_inequality 
  (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 :=
sorry

end sufficient_condition_for_reciprocal_inequality_l19_19421


namespace probability_dice_roll_l19_19636

theorem probability_dice_roll :
  (1 / 2) * (1 / 3) = 1 / 6 :=
by
  -- Here you can add the proof steps if needed
  sorry

end probability_dice_roll_l19_19636


namespace kids_on_excursions_l19_19489

theorem kids_on_excursions (total_kids : ℕ) (one_fourth_kids_tubing two := total_kids / 4) (half_tubers_rafting : ℕ := one_fourth_kids_tubing / 2) :
  total_kids = 40 → one_fourth_kids_tubing = 10 → half_tubers_rafting = 5 :=
by
  intros
  sorry

end kids_on_excursions_l19_19489


namespace combined_probability_l19_19280

-- Definitions:
def number_of_ways_to_get_3_heads_and_1_tail := Nat.choose 4 3
def probability_of_specific_sequence_of_3_heads_and_1_tail := (1/2) ^ 4
def probability_of_3_heads_and_1_tail := number_of_ways_to_get_3_heads_and_1_tail * probability_of_specific_sequence_of_3_heads_and_1_tail

def favorable_outcomes_die := 2
def total_outcomes_die := 6
def probability_of_number_greater_than_4 := favorable_outcomes_die / total_outcomes_die

-- Proof statement:
theorem combined_probability : probability_of_3_heads_and_1_tail * probability_of_number_greater_than_4 = 1/12 := by
  sorry

end combined_probability_l19_19280


namespace initial_blue_balls_l19_19991

theorem initial_blue_balls (B : ℕ) 
  (h1 : 18 - 3 = 15) 
  (h2 : (B - 3) / 15 = 1 / 5) : 
  B = 6 :=
by sorry

end initial_blue_balls_l19_19991


namespace part_I_part_II_l19_19066

-- Let the volume V of the tetrahedron ABCD be given
def V : ℝ := sorry

-- Areas of the faces opposite vertices A, B, C, D
def S_A : ℝ := sorry
def S_B : ℝ := sorry
def S_C : ℝ := sorry
def S_D : ℝ := sorry

-- Definitions of the edge lengths and angles
def a : ℝ := sorry -- BC
def a' : ℝ := sorry -- DA
def b : ℝ := sorry -- CA
def b' : ℝ := sorry -- DB
def c : ℝ := sorry -- AB
def c' : ℝ := sorry -- DC
def alpha : ℝ := sorry -- Angle between BC and DA
def beta : ℝ := sorry -- Angle between CA and DB
def gamma : ℝ := sorry -- Angle between AB and DC

theorem part_I : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 = 
  (1 / 4) * ((a * a' * Real.sin alpha)^2 + (b * b' * Real.sin beta)^2 + (c * c' * Real.sin gamma)^2) := 
  sorry

theorem part_II : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 ≥ 9 * (3 * V^4)^(1/3) :=
  sorry

end part_I_part_II_l19_19066


namespace proof_inequality_l19_19061

theorem proof_inequality (n : ℕ) (a b : ℝ) (c : ℝ) (h_n : 1 ≤ n) (h_a : 1 ≤ a) (h_b : 1 ≤ b) (h_c : 0 < c) : 
  ((ab + c)^n - c) / ((b + c)^n - c) ≤ a^n :=
sorry

end proof_inequality_l19_19061


namespace find_smallest_x_l19_19986

def smallest_x_divisible (y : ℕ) : ℕ :=
  if y = 11 then 257 else 0

theorem find_smallest_x : 
  smallest_x_divisible 11 = 257 ∧ 
  ∃ k : ℕ, 264 * k - 7 = 257 :=
by
  sorry

end find_smallest_x_l19_19986


namespace exponent_fraction_simplification_l19_19675

theorem exponent_fraction_simplification : 
  (2 ^ 2016 + 2 ^ 2014) / (2 ^ 2016 - 2 ^ 2014) = 5 / 3 := 
by {
  -- proof steps would go here
  sorry
}

end exponent_fraction_simplification_l19_19675


namespace profit_function_equation_maximum_profit_l19_19261

noncomputable def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
noncomputable def sales_revenue (x : ℝ) : ℝ := 18*x
noncomputable def production_profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_function_equation (x : ℝ) : production_profit x = -x^3 + 24*x^2 - 45*x - 10 :=
  by
    unfold production_profit sales_revenue production_cost
    sorry

theorem maximum_profit : (production_profit 15 = 1340) ∧ ∀ x, production_profit 15 ≥ production_profit x :=
  by
    sorry

end profit_function_equation_maximum_profit_l19_19261


namespace eight_digit_numbers_count_l19_19909

theorem eight_digit_numbers_count :
  let first_digit_choices := 9
  let remaining_digits_choices := 10 ^ 7
  9 * 10^7 = 90000000 :=
by
  sorry

end eight_digit_numbers_count_l19_19909


namespace min_value_l19_19269

-- Definition of the conditions
def positive (a : ℝ) : Prop := a > 0

theorem min_value (a : ℝ) (h : positive a) : 
  ∃ m : ℝ, (m = 2 * Real.sqrt 6) ∧ (∀ x : ℝ, positive x → (3 / (2 * x) + 4 * x) ≥ m) :=
sorry

end min_value_l19_19269


namespace roots_equation_l19_19811

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l19_19811


namespace exists_permutation_ab_minus_cd_ge_two_l19_19016

theorem exists_permutation_ab_minus_cd_ge_two (p q r s : ℝ) 
  (h1 : p + q + r + s = 9) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 21) :
  ∃ (a b c d : ℝ), (a, b, c, d) = (p, q, r, s) ∨ (a, b, c, d) = (p, q, s, r) ∨ 
  (a, b, c, d) = (p, r, q, s) ∨ (a, b, c, d) = (p, r, s, q) ∨ 
  (a, b, c, d) = (p, s, q, r) ∨ (a, b, c, d) = (p, s, r, q) ∨ 
  (a, b, c, d) = (q, p, r, s) ∨ (a, b, c, d) = (q, p, s, r) ∨ 
  (a, b, c, d) = (q, r, p, s) ∨ (a, b, c, d) = (q, r, s, p) ∨ 
  (a, b, c, d) = (q, s, p, r) ∨ (a, b, c, d) = (q, s, r, p) ∨ 
  (a, b, c, d) = (r, p, q, s) ∨ (a, b, c, d) = (r, p, s, q) ∨ 
  (a, b, c, d) = (r, q, p, s) ∨ (a, b, c, d) = (r, q, s, p) ∨ 
  (a, b, c, d) = (r, s, p, q) ∨ (a, b, c, d) = (r, s, q, p) ∨ 
  (a, b, c, d) = (s, p, q, r) ∨ (a, b, c, d) = (s, p, r, q) ∨ 
  (a, b, c, d) = (s, q, p, r) ∨ (a, b, c, d) = (s, q, r, p) ∨ 
  (a, b, c, d) = (s, r, p, q) ∨ (a, b, c, d) = (s, r, q, p) ∧ ab - cd ≥ 2 :=
sorry

end exists_permutation_ab_minus_cd_ge_two_l19_19016


namespace prop_p_necessary_but_not_sufficient_for_prop_q_l19_19781

theorem prop_p_necessary_but_not_sufficient_for_prop_q (x y : ℕ) :
  (x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4) → ((x+y ≠ 4) → (x ≠ 1 ∨ y ≠ 3)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4)) :=
by
  sorry

end prop_p_necessary_but_not_sufficient_for_prop_q_l19_19781


namespace erica_has_correct_amount_l19_19787

-- Definitions for conditions
def total_money : ℕ := 91
def sam_money : ℕ := 38

-- Definition for the question regarding Erica's money
def erica_money := total_money - sam_money

-- The theorem stating the proof problem
theorem erica_has_correct_amount : erica_money = 53 := sorry

end erica_has_correct_amount_l19_19787


namespace average_height_of_trees_l19_19297

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l19_19297


namespace solve_quadratic_eq_l19_19817

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 → x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l19_19817


namespace stratified_sampling_third_year_students_l19_19525

theorem stratified_sampling_third_year_students 
  (total_students : ℕ)
  (sample_size : ℕ)
  (ratio_1st : ℕ)
  (ratio_2nd : ℕ)
  (ratio_3rd : ℕ)
  (ratio_4th : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 200)
  (h3 : ratio_1st = 4)
  (h4 : ratio_2nd = 3)
  (h5 : ratio_3rd = 2)
  (h6 : ratio_4th = 1) :
  (ratio_3rd : ℚ) / (ratio_1st + ratio_2nd + ratio_3rd + ratio_4th : ℚ) * sample_size = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l19_19525


namespace abc_zero_l19_19663

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) 
  : a * b * c = 0 := by
  sorry

end abc_zero_l19_19663


namespace min_value_of_reciprocal_sum_l19_19264

variables {m n : ℝ}
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (h3 : m + n = 1)

theorem min_value_of_reciprocal_sum : 
  (1 / m + 1 / n) = 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l19_19264


namespace tens_digit_of_72_pow_25_l19_19332

theorem tens_digit_of_72_pow_25 : (72^25 % 100) / 10 = 3 := 
by
  sorry

end tens_digit_of_72_pow_25_l19_19332


namespace number_of_boys_in_school_l19_19287

-- Definition of percentages for Muslims, Hindus, and Sikhs
def percent_muslims : ℝ := 0.46
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10

-- Given number of boys in other communities
def boys_other_communities : ℝ := 136

-- The total number of boys in the school
def total_boys (B : ℝ) : Prop := B = 850

-- Proof statement (with conditions embedded)
theorem number_of_boys_in_school (B : ℝ) :
  percent_muslims * B + percent_hindus * B + percent_sikhs * B + boys_other_communities = B → 
  total_boys B :=
by
  sorry

end number_of_boys_in_school_l19_19287


namespace largest_multiple_of_15_less_than_500_l19_19146

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19146


namespace central_angle_of_sector_l19_19692

theorem central_angle_of_sector (P : ℝ) (x : ℝ) (h : P = 1 / 8) : x = 45 :=
by
  sorry

end central_angle_of_sector_l19_19692


namespace find_a_for_even_function_l19_19734

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = (x + 1) * (x + a) ∧ f (-x) = f x) : a = -1 := by 
  sorry

end find_a_for_even_function_l19_19734


namespace largest_multiple_of_15_less_than_500_l19_19180

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19180


namespace binom_identity_l19_19302

theorem binom_identity (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  Nat.choose (n + 1) (k + 1) = ∑ i in Finset.range (k + 1), Nat.choose (n - i) k := 
sorry

end binom_identity_l19_19302


namespace repeated_root_and_m_value_l19_19729

theorem repeated_root_and_m_value :
  (∃ x m : ℝ, (x = 2 ∨ x = -2) ∧ 
              (m / (x ^ 2 - 4) + 2 / (x + 2) = 1 / (x - 2)) ∧ 
              (m = 4 ∨ m = 8)) :=
sorry

end repeated_root_and_m_value_l19_19729


namespace min_pairs_opponents_statement_l19_19418

-- Problem statement definitions
variables (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2)

-- Required minimum number of pairs of opponents in a parliament
def min_pairs_opponents (h p : ℕ) : ℕ :=
  min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)

-- Proof statement
theorem min_pairs_opponents_statement (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2) :
  ∀ (hp : ℕ), ∃ (pairs : ℕ), 
    pairs = min_pairs_opponents h p :=
  sorry

end min_pairs_opponents_statement_l19_19418


namespace problem_1_l19_19252

noncomputable def derivative_y (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) : ℝ :=
  (2 * a) / (3 * (1 - y^2))

theorem problem_1 (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) :
  derivative_y a x y h = (2 * a) / (3 * (1 - y^2)) :=
sorry

end problem_1_l19_19252


namespace scaled_det_l19_19731

variable (x y z a b c p q r : ℝ)
variable (det_orig : ℝ)
variable (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2)

theorem scaled_det (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2) :
  Matrix.det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 :=
by
  sorry

end scaled_det_l19_19731


namespace train_B_time_to_destination_l19_19671

-- Definitions (conditions)
def speed_train_A := 60  -- Train A travels at 60 kmph
def speed_train_B := 90  -- Train B travels at 90 kmph
def time_train_A_after_meeting := 9 -- Train A takes 9 hours after meeting train B

-- Theorem statement
theorem train_B_time_to_destination 
  (speed_A : ℝ)
  (speed_B : ℝ)
  (time_A_after_meeting : ℝ)
  (time_B_to_destination : ℝ) :
  speed_A = speed_train_A ∧
  speed_B = speed_train_B ∧
  time_A_after_meeting = time_train_A_after_meeting →
  time_B_to_destination = 4.5 :=
by
  sorry

end train_B_time_to_destination_l19_19671


namespace smallest_bdf_l19_19938

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end smallest_bdf_l19_19938


namespace largest_x_value_l19_19673

theorem largest_x_value : ∃ x : ℝ, (x / 7 + 3 / (7 * x) = 1) ∧ (∀ y : ℝ, (y / 7 + 3 / (7 * y) = 1) → y ≤ (7 + Real.sqrt 37) / 2) :=
by
  -- (Proof of the theorem is omitted for this task)
  sorry

end largest_x_value_l19_19673


namespace sally_money_l19_19953

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l19_19953


namespace max_4_element_subsets_of_8_l19_19328

open Finset

def maximum_subsets_condition (S : Finset ℕ) : ℕ :=
  @max {(F : Finset (Finset ℕ)) // ∀ A B C ∈ F, |A ∩ B ∩ C| ≤ 1} Finset.card
  sorry

theorem max_4_element_subsets_of_8 (S : Finset ℕ) (h : S.card = 8) :
  maximum_subsets_condition S = 8 :=
sorry

end max_4_element_subsets_of_8_l19_19328


namespace single_elimination_tournament_l19_19700

theorem single_elimination_tournament (teams : ℕ) (prelim_games : ℕ) (post_prelim_teams : ℕ) :
  teams = 24 →
  prelim_games = 4 →
  post_prelim_teams = teams - prelim_games →
  post_prelim_teams - 1 + prelim_games = 23 :=
by
  intros
  sorry

end single_elimination_tournament_l19_19700


namespace number_of_boxes_l19_19464

-- Definitions based on conditions
def bottles_per_box := 50
def bottle_capacity := 12
def fill_fraction := 3 / 4
def total_water := 4500

-- Question rephrased as a proof problem
theorem number_of_boxes (h1 : bottles_per_box = 50)
                        (h2 : bottle_capacity = 12)
                        (h3 : fill_fraction = 3 / 4)
                        (h4 : total_water = 4500) :
  4500 / ((12 : ℝ) * (3 / 4)) / 50 = 10 := 
by {
  sorry
}

end number_of_boxes_l19_19464


namespace number_of_valid_arrangements_l19_19389

def valid_grid (grid : List (List Nat)) : Prop :=
  grid.length = 3 ∧ (∀ row ∈ grid, row.length = 3) ∧
  (∀ row ∈ grid, row.sum = 15) ∧ 
  (∀ j : Fin 3, (List.sum (List.map (λ row, row[j]) grid) = 15))

theorem number_of_valid_arrangements : {g : List (List Nat) // valid_grid g}.card = 72 := by
  sorry

end number_of_valid_arrangements_l19_19389


namespace solve_for_x_l19_19790

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l19_19790


namespace tan_to_trig_identity_l19_19404

theorem tan_to_trig_identity (α : ℝ) (h : Real.tan α = 3) : (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
by
  sorry

end tan_to_trig_identity_l19_19404


namespace value_of_m_l19_19270

theorem value_of_m (m x : ℝ) (h1 : mx + 1 = 2 * (m - x)) (h2 : |x + 2| = 0) : m = -|3 / 4| :=
by
  sorry

end value_of_m_l19_19270


namespace tank_capacity_l19_19693

theorem tank_capacity (x : ℝ) (h : 0.24 * x = 120) : x = 500 := 
sorry

end tank_capacity_l19_19693


namespace probability_heads_heads_l19_19676

theorem probability_heads_heads (h_uniform_density : ∀ outcome, outcome ∈ {("heads", "heads"), ("heads", "tails"), ("tails", "heads"), ("tails", "tails")} → True) :
  ℙ({("heads", "heads")}) = 1 / 4 :=
sorry

end probability_heads_heads_l19_19676


namespace cookies_left_at_end_of_week_l19_19401

def trays_baked_each_day : List Nat := [2, 3, 4, 5, 3, 4, 4]
def cookies_per_tray : Nat := 12
def cookies_eaten_by_frank : Nat := 2 * 7
def cookies_eaten_by_ted : Nat := 3 + 5
def cookies_eaten_by_jan : Nat := 5
def cookies_eaten_by_tom : Nat := 8
def cookies_eaten_by_neighbours_kids : Nat := 20

def total_cookies_baked : Nat :=
  (trays_baked_each_day.map (λ trays => trays * cookies_per_tray)).sum

def total_cookies_eaten : Nat :=
  cookies_eaten_by_frank + cookies_eaten_by_ted + cookies_eaten_by_jan +
  cookies_eaten_by_tom + cookies_eaten_by_neighbours_kids

def cookies_left : Nat := total_cookies_baked - total_cookies_eaten

theorem cookies_left_at_end_of_week : cookies_left = 245 :=
by
  sorry

end cookies_left_at_end_of_week_l19_19401


namespace fraction_pow_zero_l19_19845

theorem fraction_pow_zero (a b : ℤ) (hb_nonzero : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 :=
by 
  sorry

end fraction_pow_zero_l19_19845


namespace Elizabeth_lost_bottles_l19_19885

theorem Elizabeth_lost_bottles :
  ∃ (L : ℕ), (10 - L - 1) * 3 = 21 ∧ L = 2 := by
  sorry

end Elizabeth_lost_bottles_l19_19885


namespace sum_of_numbers_l19_19825

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), n = k * 10 + d + m * 10 * (10 ^ k)

theorem sum_of_numbers
  (A B C : ℕ)
  (hA : A >= 100 ∧ A < 1000)
  (hB : B >= 10 ∧ B < 100)
  (hC : C >= 10 ∧ C < 100)
  (h7 : (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) ∧
              (if contains_digit A 7 then A else 0) +
              (if contains_digit B 7 then B else 0) +
              (if contains_digit C 7 then C else 0) = 208)
  (h3 : (contains_digit B 3 ∨ contains_digit C 3) ∧ 
              (if contains_digit B 3 then B else 0) +
              (if contains_digit C 3 then C else 0) = 76) :
  A + B + C = 247 :=
sorry

end sum_of_numbers_l19_19825


namespace average_problem_l19_19912

theorem average_problem
  (a b c d : ℚ)
  (h1 : (a + d) / 2 = 40)
  (h2 : (b + d) / 2 = 60)
  (h3 : (a + b) / 2 = 50)
  (h4 : (b + c) / 2 = 70) :
  c - a = 40 :=
begin
  sorry,
end

end average_problem_l19_19912


namespace correct_statement_d_l19_19990

theorem correct_statement_d (x : ℝ) : 2 * (x + 1) = x + 7 → x = 5 :=
by
  sorry

end correct_statement_d_l19_19990


namespace find_an_from_sums_l19_19922

noncomputable def geometric_sequence (a : ℕ → ℝ) (q r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_an_from_sums (a : ℕ → ℝ) (q r : ℝ) (S3 S6 : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 :  a 1 * (1 - q^3) / (1 - q) = S3) 
  (h3 : a 1 * (1 - q^6) / (1 - q) = S6) : 
  ∃ a1 q, a n = a1 * q^(n-1) := 
by
  sorry

end find_an_from_sums_l19_19922


namespace largest_multiple_of_15_less_than_500_l19_19187

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19187


namespace Maria_green_towels_l19_19632

-- Definitions
variable (G : ℕ) -- number of green towels

-- Conditions
def initial_towels := G + 21
def final_towels := initial_towels - 34

-- Theorem statement
theorem Maria_green_towels : final_towels = 22 → G = 35 :=
by
  sorry

end Maria_green_towels_l19_19632


namespace no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l19_19999

variables {a b : ℝ} (spells : list (ℝ × ℝ)) (infinite_spells : ℕ → ℝ × ℝ)

-- Condition: 0 < a < b
def valid_spell (spell : ℝ × ℝ) : Prop := 0 < spell.1 ∧ spell.1 < spell.2

-- Question a: Finite set of spells, prove that no spell set exists such that the second wizard can guarantee a win.
theorem no_finite_spells_guarantee_second_wizard_win :
  (∀ spell ∈ spells, valid_spell spell) →
  ¬(∃ (strategy : ℕ → ℝ × ℝ), ∀ n, valid_spell (strategy n) ∧ ∃ k, n < k ∧ valid_spell (strategy k)) :=
sorry

-- Question b: Infinite set of spells, prove that there exists a spell set such that the second wizard can guarantee a win.
theorem exists_infinite_spells_guarantee_second_wizard_win :
  (∀ n, valid_spell (infinite_spells n)) →
  ∃ (strategy : ℕ → ℝ × ℝ), ∀ n, ∃ k, n < k ∧ valid_spell (strategy k) :=
sorry

end no_finite_spells_guarantee_second_wizard_win_exists_infinite_spells_guarantee_second_wizard_win_l19_19999


namespace tug_of_war_matches_l19_19955

-- Define the number of classes
def num_classes : ℕ := 7

-- Define the number of matches Grade 3 Class 6 competes in
def matches_class6 : ℕ := num_classes - 1

-- Define the total number of matches
def total_matches : ℕ := (num_classes - 1) * num_classes / 2

-- Main theorem stating the problem
theorem tug_of_war_matches :
  matches_class6 = 6 ∧ total_matches = 21 := by
  sorry

end tug_of_war_matches_l19_19955


namespace average_height_of_trees_l19_19294

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l19_19294


namespace sqrt_product_eq_l19_19712

theorem sqrt_product_eq :
  (16 ^ (1 / 4) : ℝ) * (64 ^ (1 / 2)) = 16 := by
  sorry

end sqrt_product_eq_l19_19712


namespace find_value_of_x_l19_19258

theorem find_value_of_x (x : ℕ) (h : (50 + x / 90) * 90 = 4520) : x = 4470 :=
sorry

end find_value_of_x_l19_19258


namespace problem_a_b_c_l19_19733

theorem problem_a_b_c (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ac = 0) (h4 : abc = 1) : |a + b| > |c| := 
by sorry

end problem_a_b_c_l19_19733


namespace solve_logarithmic_equation_l19_19718

noncomputable def equation_holds (x : ℝ) : Prop :=
  log (x^2 + 5*x + 6) = log ((x + 1) * (x + 4)) + log (x - 2)

theorem solve_logarithmic_equation (x : ℝ) (h1 : x > 2) (h2 : x^2 + 5*x + 6 > 0) (h3 : (x + 1) * (x + 4) > 0) : 
  equation_holds x ↔ x^3 - 4*x - 14 = 0 :=
sorry

end solve_logarithmic_equation_l19_19718


namespace rounding_bounds_l19_19697

theorem rounding_bounds:
  ∃ (max min : ℕ), (∀ x : ℕ, (x >= 1305000) → (x < 1305000) -> false) ∧ 
  (max = 1304999) ∧ 
  (min = 1295000) :=
by
  -- Proof steps would go here
  sorry

end rounding_bounds_l19_19697


namespace expected_value_of_12_sided_die_is_6_5_l19_19356

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_l19_19356


namespace gcd_12012_18018_l19_19575

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l19_19575


namespace minute_hand_40_min_angle_l19_19081

noncomputable def minute_hand_rotation_angle (minutes : ℕ): ℝ :=
  if minutes = 60 then -2 * Real.pi 
  else (minutes / 60) * -2 * Real.pi

theorem minute_hand_40_min_angle :
  minute_hand_rotation_angle 40 = - (4 / 3) * Real.pi :=
by
  sorry

end minute_hand_40_min_angle_l19_19081


namespace solve_x_sq_plus_y_sq_l19_19910

theorem solve_x_sq_plus_y_sq (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = 2) : x^2 + y^2 = 5 :=
by
  sorry

end solve_x_sq_plus_y_sq_l19_19910


namespace sum_of_three_numbers_l19_19833

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1: 100 ≤ A ∧ A ≤ 999)
  (h2: 10 ≤ B ∧ B ≤ 99) 
  (h3: 10 ≤ C ∧ C ≤ 99)
  (h4: (contains_digit A 7 → A) + (contains_digit B 7 → B) + (contains_digit C 7 → C) = 208)
  (h5: (contains_digit B 3 → B) + (contains_digit C 3 → C) = 76) :
  A + B + C = 247 := 
by 
  sorry

end sum_of_three_numbers_l19_19833


namespace find_y_l19_19820

variable (x y z : ℚ)

theorem find_y
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) :
  y = 688 / 9 :=
sorry

end find_y_l19_19820


namespace range_of_ab_l19_19444

noncomputable def range_ab : Set ℝ := 
  { x | 4 ≤ x ∧ x ≤ 112 / 9 }

theorem range_of_ab (a b : ℝ) 
  (q : ℝ) (h1 : q ∈ (Set.Icc (1/3) 2)) 
  (h2 : ∃ m : ℝ, ∃ nq : ℕ, 
    (m * q ^ nq) * m ^ (2 - nq) = 1 ∧ 
    (m + m * q ^ nq) = a ∧ 
    (m * q + m * q ^ 2) = b):
  ab = (q + 1/q + q^2 + 1/q^2) → 
  (ab ∈ range_ab) := 
by 
  sorry

end range_of_ab_l19_19444


namespace geometry_problem_l19_19427

theorem geometry_problem
  (A_square : ℝ)
  (A_rectangle : ℝ)
  (A_triangle : ℝ)
  (side_length : ℝ)
  (rectangle_width : ℝ)
  (rectangle_length : ℝ)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (square_area_eq : A_square = side_length ^ 2)
  (rectangle_area_eq : A_rectangle = rectangle_width * rectangle_length)
  (triangle_area_eq : A_triangle = (triangle_base * triangle_height) / 2)
  (side_length_eq : side_length = 4)
  (rectangle_width_eq : rectangle_width = 4)
  (triangle_base_eq : triangle_base = 8)
  (areas_equal : A_square = A_rectangle ∧ A_square = A_triangle) :
  rectangle_length = 4 ∧ triangle_height = 4 :=
by
  sorry

end geometry_problem_l19_19427


namespace vegetarian_count_l19_19917

variables (v_only v_nboth vegan pesc nvboth : ℕ)
variables (hv_only : v_only = 13) (hv_nboth : v_nboth = 8)
          (hvegan_tot : vegan = 5) (hvegan_v : vveg1 = 3)
          (hpesc_tot : pesc = 4) (hpesc_vnboth : nvboth = 2)

theorem vegetarian_count (total_veg : ℕ) 
  (H_total : total_veg = v_only + v_nboth + (vegan - vveg1)) :
  total_veg = 23 :=
sorry

end vegetarian_count_l19_19917


namespace paula_aunt_gave_her_total_money_l19_19309

theorem paula_aunt_gave_her_total_money :
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  total_spent + money_left = 109 :=
by
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  show total_spent + money_left = 109
  sorry

end paula_aunt_gave_her_total_money_l19_19309


namespace number_of_real_solutions_l19_19445

noncomputable def greatest_integer (x: ℝ) : ℤ :=
  ⌊x⌋

def equation (x: ℝ) :=
  4 * x^2 - 40 * (greatest_integer x : ℝ) + 51 = 0

theorem number_of_real_solutions : 
  ∃ (x1 x2 x3 x4: ℝ), 
  equation x1 ∧ equation x2 ∧ equation x3 ∧ equation x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 := 
sorry

end number_of_real_solutions_l19_19445


namespace range_of_a_l19_19407

variable (x a : ℝ)

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem range_of_a (h : ∀ x, q x a → p x)
  (h_not : ∃ x, ¬ q x a ∧ p x) : 1 ≤ a :=
sorry

end range_of_a_l19_19407


namespace least_groups_needed_l19_19349

noncomputable def numberOfGroups (totalStudents : ℕ) (maxStudentsPerGroup : ℕ) : ℕ :=
  totalStudents / maxStudentsPerGroup

theorem least_groups_needed (totalStudents : ℕ) (maxStudentsPerGroup : ℕ) (h1 : totalStudents = 30) (h2 : maxStudentsPerGroup = 12) :
  numberOfGroups totalStudents 10 = 3 :=
by
  rw [h1, h2]
  repeat { sorry }

end least_groups_needed_l19_19349


namespace gcd_12012_18018_l19_19572

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l19_19572


namespace waiter_slices_l19_19277

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end waiter_slices_l19_19277


namespace largest_multiple_of_15_less_than_500_l19_19130

theorem largest_multiple_of_15_less_than_500 : ∃ (x : ℕ), (x < 500) ∧ (x % 15 = 0) ∧ ∀ (y : ℕ), (y < 500) ∧ (y % 15 = 0) → y ≤ x :=
begin
  use 495,
  split,
  { norm_num },
  split,
  { norm_num },
  intros y hy1 hy2,
  sorry,
end

end largest_multiple_of_15_less_than_500_l19_19130


namespace base16_to_base2_bits_l19_19333

theorem base16_to_base2_bits :
  ∀ (n : ℕ), n = 16^4 * 7 + 16^3 * 7 + 16^2 * 7 + 16 * 7 + 7 → (2^18 ≤ n ∧ n < 2^19) → 
  ∃ b : ℕ, b = 19 := 
by
  intros n hn hpow
  sorry

end base16_to_base2_bits_l19_19333


namespace gcd_of_12012_and_18018_l19_19583

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l19_19583


namespace students_play_neither_l19_19915

-- Define the given conditions
def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_sports_players : ℕ := 17

-- The goal is to prove the number of students playing neither sport is 7
theorem students_play_neither :
  total_students - (football_players + long_tennis_players - both_sports_players) = 7 :=
by
  sorry

end students_play_neither_l19_19915


namespace cubic_polynomial_roots_value_l19_19591

theorem cubic_polynomial_roots_value
  (a b c d : ℝ) 
  (h_cond : a ≠ 0 ∧ d ≠ 0)
  (h_equiv : (a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + d) + (a * (-1/2)^3 + b * (-1/2)^2 + c * (-1/2) + d) = 1000 * d)
  (h_roots : ∃ (x1 x2 x3 : ℝ), a * x1^3 + b * x1^2 + c * x1 + d = 0 ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
  : (∃ (x1 x2 x3 : ℝ), (1 / (x1 * x2) + 1 / (x2 * x3) + 1 / (x1 * x3) = 1996)) :=
by
  sorry

end cubic_polynomial_roots_value_l19_19591


namespace sum_of_squares_of_roots_l19_19772

noncomputable def polynomial := Polynomial R

-- Define the polynomial
def p : polynomial := polynomial.monomial 8 1 - 14 * polynomial.monomial 4 1 - 8 * polynomial.monomial 3 1 
                      - polynomial.monomial 2 1 + polynomial.C 1

-- Define the roots as variables
variables r : ℝ
variables r1 r2 r3 r4 : ℝ

-- Assume that r1, r2, r3, r4 are distinct real roots of the polynomial
axiom roots: polynomial.has_roots [r1, r2, r3, r4]

-- The main theorem stating the proof goal
theorem sum_of_squares_of_roots : r1^2 + r2^2 + r3^2 + r4^2 = 8 :=
sorry

end sum_of_squares_of_roots_l19_19772


namespace two_distinct_real_roots_of_modified_quadratic_l19_19770

theorem two_distinct_real_roots_of_modified_quadratic (a b k : ℝ) (h1 : a^2 - b > 0) (h2 : k > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + 2 * a * x₁ + b + k * (x₁ + a)^2 = 0) ∧ (x₂^2 + 2 * a * x₂ + b + k * (x₂ + a)^2 = 0) :=
by
  sorry

end two_distinct_real_roots_of_modified_quadratic_l19_19770


namespace number_cooking_and_weaving_l19_19334

section CurriculumProblem

variables {total_yoga total_cooking total_weaving : ℕ}
variables {cooking_only cooking_and_yoga all_curriculums CW : ℕ}

-- Given conditions
def yoga (total_yoga : ℕ) := total_yoga = 35
def cooking (total_cooking : ℕ) := total_cooking = 20
def weaving (total_weaving : ℕ) := total_weaving = 15
def cookingOnly (cooking_only : ℕ) := cooking_only = 7
def cookingAndYoga (cooking_and_yoga : ℕ) := cooking_and_yoga = 5
def allCurriculums (all_curriculums : ℕ) := all_curriculums = 3

-- Prove that CW (number of people studying both cooking and weaving) is 8
theorem number_cooking_and_weaving : 
  yoga total_yoga → cooking total_cooking → weaving total_weaving → 
  cookingOnly cooking_only → cookingAndYoga cooking_and_yoga → 
  allCurriculums all_curriculums → CW = 8 := 
by 
  intros h_yoga h_cooking h_weaving h_cookingOnly h_cookingAndYoga h_allCurriculums
  -- Placeholder for the actual proof
  sorry

end CurriculumProblem

end number_cooking_and_weaving_l19_19334


namespace largest_multiple_of_15_less_than_500_is_495_l19_19159

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19159


namespace phil_won_more_games_than_charlie_l19_19945

theorem phil_won_more_games_than_charlie :
  ∀ (P D C Ph : ℕ),
  (P = D + 5) → (C = D - 2) → (Ph = 12) → (P = Ph + 4) →
  Ph - C = 3 :=
by
  intros P D C Ph hP hC hPh hPPh
  sorry

end phil_won_more_games_than_charlie_l19_19945


namespace inequality_sqrt_sum_ge_one_l19_19926

variable (a b c : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (prod_abc : a * b * c = 1)

theorem inequality_sqrt_sum_ge_one :
  (Real.sqrt (a / (8 + a)) + Real.sqrt (b / (8 + b)) + Real.sqrt (c / (8 + c)) ≥ 1) :=
by
  sorry

end inequality_sqrt_sum_ge_one_l19_19926


namespace ratio_of_neighborhood_to_gina_l19_19730

variable (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ)

def neighborhood_to_gina_ratio (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ) := 
  (Total_weight_collected - Gina_bags * Weight_per_bag) / (Gina_bags * Weight_per_bag)

theorem ratio_of_neighborhood_to_gina 
  (h₁ : Gina_bags = 2) 
  (h₂ : Weight_per_bag = 4) 
  (h₃ : Total_weight_collected = 664) :
  neighborhood_to_gina_ratio Gina_bags Weight_per_bag Total_weight_collected = 82 := 
by 
  sorry

end ratio_of_neighborhood_to_gina_l19_19730


namespace terminal_side_half_angle_l19_19900

theorem terminal_side_half_angle {k : ℤ} {α : ℝ} 
  (h : 2 * k * π < α ∧ α < 2 * k * π + π / 2) : 
  (k * π < α / 2 ∧ α / 2 < k * π + π / 4) ∨ (k * π + π <= α / 2 ∧ α / 2 < (k + 1) * π + π / 4) :=
sorry

end terminal_side_half_angle_l19_19900


namespace what_percent_of_y_l19_19433

-- Given condition
axiom y_pos : ℝ → Prop

noncomputable def math_problem (y : ℝ) (h : y_pos y) : Prop :=
  (8 * y / 20 + 3 * y / 10 = 0.7 * y)

-- The theorem to be proved
theorem what_percent_of_y (y : ℝ) (h : y > 0) : 8 * y / 20 + 3 * y / 10 = 0.7 * y :=
by
  sorry

end what_percent_of_y_l19_19433


namespace sally_money_l19_19952

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l19_19952


namespace min_value_expression_l19_19932

-- Let x and y be positive integers such that x^2 + y^2 - 2017 * x * y > 0 and it is not a perfect square.
theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_not_square : ¬ ∃ z : ℕ, (x^2 + y^2 - 2017 * x * y) = z^2) :
  x^2 + y^2 - 2017 * x * y > 0 → ∃ k : ℕ, k = 2019 ∧ ∀ m : ℕ, (m > 0 → ¬ ∃ z : ℤ, (x^2 + y^2 - 2017 * x * y) = z^2 ∧ x^2 + y^2 - 2017 * x * y < k) :=
sorry

end min_value_expression_l19_19932


namespace problem_statement_l19_19603

theorem problem_statement (g : ℝ → ℝ) (m k : ℝ) (h₀ : ∀ x, g x = 5 * x - 3)
  (h₁ : 0 < k) (h₂ : 0 < m)
  (h₃ : ∀ x, |g x - 2| < k ↔ |x - 1| < m) : m ≤ k / 5 :=
sorry

end problem_statement_l19_19603


namespace largest_multiple_of_15_less_than_500_l19_19166

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l19_19166


namespace rem_sum_a_b_c_l19_19422

theorem rem_sum_a_b_c (a b c : ℤ) (h1 : a * b * c ≡ 1 [ZMOD 5]) (h2 : 3 * c ≡ 1 [ZMOD 5]) (h3 : 4 * b ≡ 1 + b [ZMOD 5]) : 
  (a + b + c) % 5 = 3 := by 
  sorry

end rem_sum_a_b_c_l19_19422


namespace smallest_product_bdf_l19_19944

theorem smallest_product_bdf 
  (a b c d e f : ℕ) 
  (h1 : (a + 1) * (c * e) = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) : 
  b * d * f = 60 := 
sorry

end smallest_product_bdf_l19_19944


namespace largest_multiple_of_15_under_500_l19_19212

theorem largest_multiple_of_15_under_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ ∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x :=
by 
  sorry

end largest_multiple_of_15_under_500_l19_19212


namespace alexandra_magazines_l19_19535

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l19_19535


namespace cone_surface_area_l19_19752

-- Definitions based on conditions
def sector_angle : Real := 2 * Real.pi / 3
def sector_radius : Real := 2

-- Definition of the radius of the cone's base
def cone_base_radius (sector_angle sector_radius : Real) : Real :=
  sector_radius * sector_angle / (2 * Real.pi)

-- Definition of the lateral surface area of the cone
def lateral_surface_area (r l : Real) : Real :=
  Real.pi * r * l

-- Definition of the base area of the cone
def base_area (r : Real) : Real :=
  Real.pi * r^2

-- Total surface area of the cone
def total_surface_area (sector_angle sector_radius : Real) : Real :=
  let r := cone_base_radius sector_angle sector_radius
  let S1 := lateral_surface_area r sector_radius
  let S2 := base_area r
  S1 + S2

theorem cone_surface_area (h1 : sector_angle = 2 * Real.pi / 3)
                          (h2 : sector_radius = 2) :
  total_surface_area sector_angle sector_radius = 16 * Real.pi / 9 :=
by
  sorry

end cone_surface_area_l19_19752


namespace problem_statement_l19_19411

variable { a b c x y z : ℝ }

theorem problem_statement 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 :=
by 
  sorry

end problem_statement_l19_19411


namespace largest_multiple_of_15_less_than_500_l19_19149

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19149


namespace range_of_a_l19_19600

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∨ a ∈ Set.Ioi 1 :=
by sorry

end range_of_a_l19_19600


namespace scientific_notation_example_l19_19886

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (3650000 : ℝ) = a * 10 ^ n :=
sorry

end scientific_notation_example_l19_19886


namespace french_students_l19_19755

theorem french_students 
  (T : ℕ) (G : ℕ) (B : ℕ) (N : ℕ) (F : ℕ)
  (hT : T = 78) (hG : G = 22) (hB : B = 9) (hN : N = 24)
  (h_eq : F + G - B = T - N) :
  F = 41 :=
by
  sorry

end french_students_l19_19755


namespace constant_function_of_horizontal_tangent_l19_19751

theorem constant_function_of_horizontal_tangent (f : ℝ → ℝ) (h : ∀ x, deriv f x = 0) : ∃ c : ℝ, ∀ x, f x = c :=
sorry

end constant_function_of_horizontal_tangent_l19_19751


namespace sum_of_three_numbers_l19_19842

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l19_19842


namespace partOneCorrectProbability_partTwoCorrectProbability_l19_19954

noncomputable def teachers_same_gender_probability (mA fA mB fB : ℕ) : ℚ :=
  let total_outcomes := mA * mB + mA * fB + fA * mB + fA * fB
  let same_gender := mA * mB + fA * fB
  same_gender / total_outcomes

noncomputable def teachers_same_school_probability (SA SB : ℕ) : ℚ :=
  let total_teachers := SA + SB
  let total_outcomes := (total_teachers * (total_teachers - 1)) / 2
  let same_school := (SA * (SA - 1)) / 2 + (SB * (SB - 1)) / 2
  same_school / total_outcomes

theorem partOneCorrectProbability : teachers_same_gender_probability 2 1 1 2 = 4 / 9 := by
  sorry

theorem partTwoCorrectProbability : teachers_same_school_probability 3 3 = 2 / 5 := by
  sorry

end partOneCorrectProbability_partTwoCorrectProbability_l19_19954


namespace wine_remaining_percentage_l19_19235

theorem wine_remaining_percentage :
  let initial_wine := 250.0 -- initial wine in liters
  let daily_fraction := (249.0 / 250.0)
  let days := 50
  let remaining_wine := (daily_fraction ^ days) * initial_wine
  let percentage_remaining := (remaining_wine / initial_wine) * 100
  percentage_remaining = 81.846 :=
by
  sorry

end wine_remaining_percentage_l19_19235


namespace certain_event_at_least_one_genuine_l19_19017

def products : Finset (Fin 12) := sorry
def genuine : Finset (Fin 12) := sorry
def defective : Finset (Fin 12) := sorry
noncomputable def draw3 : Finset (Finset (Fin 12)) := sorry

-- Condition: 12 identical products, 10 genuine, 2 defective
axiom products_condition_1 : products.card = 12
axiom products_condition_2 : genuine.card = 10
axiom products_condition_3 : defective.card = 2
axiom products_condition_4 : ∀ x ∈ genuine, x ∈ products
axiom products_condition_5 : ∀ x ∈ defective, x ∈ products
axiom products_condition_6 : genuine ∩ defective = ∅

-- The statement to be proved: when drawing 3 products randomly, it is certain that at least 1 is genuine.
theorem certain_event_at_least_one_genuine :
  ∀ s ∈ draw3, ∃ x ∈ s, x ∈ genuine :=
sorry

end certain_event_at_least_one_genuine_l19_19017


namespace expected_value_of_twelve_sided_die_l19_19358

theorem expected_value_of_twelve_sided_die : ∃ E : ℝ, E = 6.5 :=
by
  let n := 12
  let sum_faces := n * (n + 1) / 2
  let E := sum_faces / n
  use E
  -- The sum of the first 12 natural numbers should be calculated: 1 + 2 + ... + 12 = 78
  have sum_calculated : sum_faces = 78 := by compute
  rw [sum_calculated]
  -- The expected value is hence 78 / 12
  have E_calculated : E = 78 / 12 := by compute
  rw [E_calculated]
  -- Finally, 78 / 12 simplifies to 6.5
  have E_final : 78 / 12 = 6.5 := by compute
  rw [E_final]

-- sorry is added for place holder to validate the step is proof-skipped.

end expected_value_of_twelve_sided_die_l19_19358


namespace largest_multiple_of_15_less_than_500_l19_19152

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19152


namespace probability_square_product_is_3_over_20_l19_19384

theorem probability_square_product_is_3_over_20 :
  let total_tiles := 15
  let total_die := 8
  let total_outcomes := total_tiles * total_die
  let favorable_pairs :=
    [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (9, 1), (4, 4), (2, 8), (8, 2), (6, 6), (9, 4), (7, 7), (8, 8)]
  let favorable_outcomes := favorable_pairs.length
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 3 / 20 :=
by
  let total_tiles := 15
  let total_die := 8
  let total_outcomes := total_tiles * total_die
  let favorable_pairs := [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (9, 1), (4, 4), (2, 8), (8, 2), (6, 6), (9, 4), (7, 7), (8, 8)]
  let favorable_outcomes := favorable_pairs.length
  have h_favorable : favorable_outcomes = 13 := rfl
  have h_total : total_outcomes = 120 := rfl
  sorry

end probability_square_product_is_3_over_20_l19_19384


namespace polynomial_inequality_holds_l19_19627

theorem polynomial_inequality_holds (a : ℝ) : (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2 * Real.sqrt 3 := 
by
  sorry

end polynomial_inequality_holds_l19_19627


namespace probability_black_ball_BoxB_higher_l19_19241

def boxA_red_balls : ℕ := 40
def boxA_black_balls : ℕ := 10
def boxB_red_balls : ℕ := 60
def boxB_black_balls : ℕ := 40
def boxB_white_balls : ℕ := 50

theorem probability_black_ball_BoxB_higher :
  (boxA_black_balls : ℚ) / (boxA_red_balls + boxA_black_balls) <
  (boxB_black_balls : ℚ) / (boxB_red_balls + boxB_black_balls + boxB_white_balls) :=
by
  sorry

end probability_black_ball_BoxB_higher_l19_19241


namespace cost_of_one_shirt_l19_19040

theorem cost_of_one_shirt (J S K : ℕ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) (h3 : 3 * J + 2 * S + K = 90) : S = 15 :=
by
  sorry

end cost_of_one_shirt_l19_19040


namespace largest_multiple_of_15_less_than_500_l19_19192

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l19_19192


namespace repeating_decimal_sum_l19_19508

theorem repeating_decimal_sum (x : ℚ) (h : x = 0.47) :
  let f := x.num + x.denom in f = 146 :=
by
  sorry

end repeating_decimal_sum_l19_19508


namespace find_g_2_l19_19807

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2
  (H : ∀ (x : ℝ), x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x ^ 2):
  g 2 = 67 / 14 :=
by
  sorry

end find_g_2_l19_19807


namespace expected_value_twelve_sided_die_l19_19365

theorem expected_value_twelve_sided_die : 
  (1 : ℝ)/12 * (∑ k in Finset.range 13, k) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l19_19365


namespace total_bird_count_correct_l19_19050

-- Define initial counts
def initial_sparrows : ℕ := 89
def initial_pigeons : ℕ := 68
def initial_finches : ℕ := 74

-- Define additional birds
def additional_sparrows : ℕ := 42
def additional_pigeons : ℕ := 51
def additional_finches : ℕ := 27

-- Define total counts
def initial_total : ℕ := 231
def final_total : ℕ := 312

theorem total_bird_count_correct :
  initial_sparrows + initial_pigeons + initial_finches = initial_total ∧
  (initial_sparrows + additional_sparrows) + 
  (initial_pigeons + additional_pigeons) + 
  (initial_finches + additional_finches) = final_total := by
    sorry

end total_bird_count_correct_l19_19050


namespace min_chord_length_eq_l19_19598

-- Define the Circle C with center (1, 2) and radius 5
def isCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

-- Define the Line l parameterized by m
def isLine (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the minimal chord length intercepted by the circle occurs when the line l is 2x - y - 5 = 0
theorem min_chord_length_eq (x y : ℝ) : 
  (∀ m, isLine m x y → isCircle x y) → isLine 0 x y :=
sorry

end min_chord_length_eq_l19_19598


namespace terminal_velocity_steady_speed_l19_19227

variable (g : ℝ) (t₁ t₂ : ℝ) (a₀ a₁ : ℝ) (v_terminal : ℝ)

-- Conditions
def acceleration_due_to_gravity := g = 10 -- m/s²
def initial_time := t₁ = 0 -- s
def intermediate_time := t₂ = 2 -- s
def initial_acceleration := a₀ = 50 -- m/s²
def final_acceleration := a₁ = 10 -- m/s²

-- Question: Prove the terminal velocity
theorem terminal_velocity_steady_speed 
  (h_g : acceleration_due_to_gravity g)
  (h_t1 : initial_time t₁)
  (h_t2 : intermediate_time t₂)
  (h_a0 : initial_acceleration a₀)
  (h_a1 : final_acceleration a₁) :
  v_terminal = 25 :=
  sorry

end terminal_velocity_steady_speed_l19_19227


namespace pear_counts_after_events_l19_19538

theorem pear_counts_after_events (Alyssa_picked Nancy_picked Carlos_picked : ℕ) (give_away : ℕ)
  (eat_fraction : ℚ) (share_fraction : ℚ) :
  Alyssa_picked = 42 →
  Nancy_picked = 17 →
  Carlos_picked = 25 →
  give_away = 5 →
  eat_fraction = 0.20 →
  share_fraction = 0.5 →
  ∃ (Alyssa_picked_final Nancy_picked_final Carlos_picked_final : ℕ),
    Alyssa_picked_final = 30 ∧
    Nancy_picked_final = 14 ∧
    Carlos_picked_final = 18 :=
by
  sorry

end pear_counts_after_events_l19_19538


namespace employee_payment_sum_l19_19495

theorem employee_payment_sum :
  ∀ (A B : ℕ), 
  (A = 3 * B / 2) → 
  (B = 180) → 
  (A + B = 450) :=
by
  intros A B hA hB
  sorry

end employee_payment_sum_l19_19495


namespace find_line_equation_l19_19012

theorem find_line_equation (a b : ℝ) :
  (2 * a + 3 * b = 0 ∧ a * b < 0) ↔ (3 * a - 2 * b = 0 ∨ a - b + 1 = 0) :=
by
  sorry

end find_line_equation_l19_19012


namespace problem_l19_19224

-- Define i as the imaginary unit
def i : ℂ := Complex.I

-- The statement to be proved
theorem problem : i * (1 - i) ^ 2 = 2 := by
  sorry

end problem_l19_19224


namespace initial_men_count_l19_19750

theorem initial_men_count (x : ℕ) (h : x * 25 = 15 * 60) : x = 36 :=
by
  sorry

end initial_men_count_l19_19750


namespace largest_multiple_of_15_less_than_500_l19_19102

theorem largest_multiple_of_15_less_than_500 :
∀ x : ℕ, (∃ k : ℕ, x = 15 * k ∧ 0 < x ∧ x < 500) → x ≤ 495 :=
begin
  sorry
end

end largest_multiple_of_15_less_than_500_l19_19102


namespace greatest_prime_factor_3_8_plus_6_7_l19_19501

theorem greatest_prime_factor_3_8_plus_6_7 : ∃ p, p = 131 ∧ Prime p ∧ ∀ q, Prime q ∧ q ∣ (3^8 + 6^7) → q ≤ 131 :=
by
  sorry


end greatest_prime_factor_3_8_plus_6_7_l19_19501


namespace min_employees_needed_l19_19236

-- Definitions for the problem conditions
def hardware_employees : ℕ := 150
def software_employees : ℕ := 130
def both_employees : ℕ := 50

-- Statement of the proof problem
theorem min_employees_needed : hardware_employees + software_employees - both_employees = 230 := 
by 
  -- Calculation skipped with sorry
  sorry

end min_employees_needed_l19_19236


namespace largest_multiple_of_15_less_than_500_l19_19144

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l19_19144


namespace largest_multiple_of_15_less_than_500_is_495_l19_19161

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end largest_multiple_of_15_less_than_500_is_495_l19_19161


namespace expected_value_twelve_sided_die_l19_19364

theorem expected_value_twelve_sided_die : 
  (1 : ℝ)/12 * (∑ k in Finset.range 13, k) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l19_19364


namespace john_total_cost_l19_19058

theorem john_total_cost :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 4
  let base_video_card_cost := 300
  let upgraded_video_card_cost := 2.5 * base_video_card_cost
  let video_card_discount := 0.12 * upgraded_video_card_cost
  let upgraded_video_card_final_cost := upgraded_video_card_cost - video_card_discount
  let foreign_monitor_cost_local := 200
  let exchange_rate := 1.25
  let foreign_monitor_cost_usd := foreign_monitor_cost_local / exchange_rate
  let peripherals_sales_tax := 0.05 * peripherals_cost
  let subtotal := computer_cost + peripherals_cost + upgraded_video_card_final_cost + peripherals_sales_tax
  let store_loyalty_discount := 0.07 * (computer_cost + peripherals_cost + upgraded_video_card_final_cost)
  let final_cost := subtotal - store_loyalty_discount + foreign_monitor_cost_usd
  final_cost = 2536.30 := sorry

end john_total_cost_l19_19058


namespace find_value_of_a_l19_19904

noncomputable def f (x a : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

theorem find_value_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ 8) ↔ a = -1 :=
by
  sorry

end find_value_of_a_l19_19904


namespace probability_four_coins_l19_19263

-- Define four fair coin flips, having 2 possible outcomes for each coin
def four_coin_flips_outcomes : ℕ := 2 ^ 4

-- Define the favorable outcomes: all heads or all tails
def favorable_outcomes : ℕ := 2

-- The probability of getting all heads or all tails
def probability_all_heads_or_tails : ℚ := favorable_outcomes / four_coin_flips_outcomes

-- The theorem stating the answer to the problem
theorem probability_four_coins:
  probability_all_heads_or_tails = 1 / 8 := by
  sorry

end probability_four_coins_l19_19263


namespace find_n_l19_19866

theorem find_n : ∃ n : ℕ, 50^4 + 43^4 + 36^4 + 6^4 = n^4 := by
  sorry

end find_n_l19_19866


namespace subtraction_result_l19_19687

noncomputable def division_value : ℝ := 1002 / 20.04

theorem subtraction_result : 2500 - division_value = 2450.0499 :=
by
  have division_eq : division_value = 49.9501 := by sorry
  rw [division_eq]
  norm_num

end subtraction_result_l19_19687


namespace bases_with_final_digit_one_l19_19893

theorem bases_with_final_digit_one :
  { b : ℕ | 3 ≤ b ∧ b ≤ 10 ∧ 624 % b = 0 }.card = 4 :=
by
  sorry

end bases_with_final_digit_one_l19_19893


namespace hermans_breakfast_cost_l19_19848

-- Define the conditions
def meals_per_day : Nat := 4
def days_per_week : Nat := 5
def cost_per_meal : Nat := 4
def total_weeks : Nat := 16

-- Define the statement to prove
theorem hermans_breakfast_cost :
  (meals_per_day * days_per_week * cost_per_meal * total_weeks) = 1280 := by
  sorry

end hermans_breakfast_cost_l19_19848
