import Mathlib

namespace fraction_division_l431_43115

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := by
  sorry

end fraction_division_l431_43115


namespace range_of_m_min_of_squares_l431_43123

-- 1. Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 4)

-- 2. State the condition that f(x) ≤ -m^2 + 6m holds for all x
def condition (m : ℝ) : Prop := ∀ x : ℝ, f x ≤ -m^2 + 6 * m

-- 3. State the range of m to be proven
theorem range_of_m : ∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5 := 
sorry

-- 4. Auxiliary condition for part 2
def m_0 : ℝ := 5

-- 5. State the condition 3a + 4b + 5c = m_0
def sum_condition (a b c : ℝ) : Prop := 3 * a + 4 * b + 5 * c = m_0

-- 6. State the minimum value problem
theorem min_of_squares (a b c : ℝ) : sum_condition a b c → a^2 + b^2 + c^2 ≥ 1 / 2 := 
sorry

end range_of_m_min_of_squares_l431_43123


namespace expr_value_l431_43194

variable (x y m n a : ℝ)
variable (hxy : x = -y) (hmn : m * n = 1) (ha : |a| = 3)

theorem expr_value : (a / (m * n) + 2018 * (x + y)) = a := sorry

end expr_value_l431_43194


namespace sum_of_numbers_l431_43127

theorem sum_of_numbers (x y : ℕ) (hx : 100 ≤ x ∧ x < 1000) (hy : 1000 ≤ y ∧ y < 10000) (h : 10000 * x + y = 12 * x * y) :
  x + y = 1083 :=
sorry

end sum_of_numbers_l431_43127


namespace tiffany_cans_at_end_of_week_l431_43185

theorem tiffany_cans_at_end_of_week:
  (4 + 2.5 - 1.25 + 0 + 3.75 - 1.5 + 0 = 7.5) :=
by
  sorry

end tiffany_cans_at_end_of_week_l431_43185


namespace A_not_on_transformed_plane_l431_43168

noncomputable def A : ℝ × ℝ × ℝ := (-3, -2, 4)
noncomputable def k : ℝ := -4/5
noncomputable def original_plane (x y z : ℝ) : Prop := 2 * x - 3 * y + z - 5 = 0

noncomputable def transformed_plane (x y z : ℝ) : Prop := 
  2 * x - 3 * y + z + (k * -5) = 0

theorem A_not_on_transformed_plane :
  ¬ transformed_plane (-3) (-2) 4 :=
by
  sorry

end A_not_on_transformed_plane_l431_43168


namespace remaining_cookies_l431_43131

theorem remaining_cookies : 
  let naomi_cookies := 53
  let oliver_cookies := 67
  let penelope_cookies := 29
  let total_cookies := naomi_cookies + oliver_cookies + penelope_cookies
  let package_size := 15
  total_cookies % package_size = 14 :=
by
  sorry

end remaining_cookies_l431_43131


namespace possible_values_of_N_l431_43169

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l431_43169


namespace Q_coordinates_l431_43137

def P : (ℝ × ℝ) := (2, -6)

def Q (x : ℝ) : (ℝ × ℝ) := (x, -6)

axiom PQ_parallel_to_x_axis : ∀ x, Q x = (x, -6)

axiom PQ_length : dist (Q 0) P = 2 ∨ dist (Q 4) P = 2

theorem Q_coordinates : Q 0 = (0, -6) ∨ Q 4 = (4, -6) :=
by {
  sorry
}

end Q_coordinates_l431_43137


namespace average_points_per_player_l431_43171

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l431_43171


namespace digit_place_value_ratio_l431_43146

theorem digit_place_value_ratio (n : ℚ) (h1 : n = 85247.2048) (h2 : ∃ d1 : ℚ, d1 * 0.1 = 0.2) (h3 : ∃ d2 : ℚ, d2 * 0.001 = 0.004) : 
  100 = 0.1 / 0.001 :=
by
  sorry

end digit_place_value_ratio_l431_43146


namespace positive_number_property_l431_43180

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_eq : (x^2) / 100 = 9) : x = 30 :=
sorry

end positive_number_property_l431_43180


namespace ratio_of_engineers_to_designers_l431_43188

-- Definitions of the variables
variables (e d : ℕ)

-- Conditions:
-- 1. The average age of the group is 45
-- 2. The average age of engineers is 40
-- 3. The average age of designers is 55

theorem ratio_of_engineers_to_designers (h : (40 * e + 55 * d) / (e + d) = 45) : e / d = 2 :=
by
-- Placeholder for the proof
sorry

end ratio_of_engineers_to_designers_l431_43188


namespace cos_frac_less_sin_frac_l431_43191

theorem cos_frac_less_sin_frac : 
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  a < b :=
by
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  sorry -- proof skipped

end cos_frac_less_sin_frac_l431_43191


namespace find_unknown_rate_l431_43108

def blankets_cost (num : ℕ) (rate : ℕ) (discount_tax : ℕ) (is_discount : Bool) : ℕ :=
  if is_discount then rate * (100 - discount_tax) / 100 * num
  else (rate * (100 + discount_tax) / 100) * num

def total_cost := blankets_cost 3 100 10 true +
                  blankets_cost 4 150 0 false +
                  blankets_cost 3 200 20 false

def avg_cost (total : ℕ) (num : ℕ) : ℕ :=
  total / num

theorem find_unknown_rate
  (unknown_rate : ℕ)
  (h1 : total_cost + 2 * unknown_rate = 1800)
  (h2 : avg_cost (total_cost + 2 * unknown_rate) 12 = 150) :
  unknown_rate = 105 :=
by
  sorry

end find_unknown_rate_l431_43108


namespace john_new_weekly_earnings_l431_43199

theorem john_new_weekly_earnings
  (original_earnings : ℕ)
  (percentage_increase : ℕ)
  (raise_amount : ℕ)
  (new_weekly_earnings : ℕ)
  (original_earnings_eq : original_earnings = 50)
  (percentage_increase_eq : percentage_increase = 40)
  (raise_amount_eq : raise_amount = original_earnings * percentage_increase / 100)
  (new_weekly_earnings_eq : new_weekly_earnings = original_earnings + raise_amount) :
  new_weekly_earnings = 70 := by
  sorry

end john_new_weekly_earnings_l431_43199


namespace maximize_squares_l431_43178

theorem maximize_squares (a b : ℕ) (k : ℕ) :
  (a ≠ b) →
  ((∃ (k : ℤ), k ≠ 1 ∧ b = k^2) ↔ 
   (∃ (c₁ c₂ c₃ : ℕ), a * (b + 8) = c₁^2 ∧ b * (a + 8) = c₂^2 ∧ a * b = c₃^2 
     ∧ a = 1)) :=
by { sorry }

end maximize_squares_l431_43178


namespace no_such_function_exists_l431_43176

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = f (n + 1) - f n :=
by
  sorry

end no_such_function_exists_l431_43176


namespace find_f_1_minus_a_l431_43126

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem find_f_1_minus_a 
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_period : periodic_function f 2)
  (h_value : ∃ a : ℝ, f (1 + a) = 1) :
  ∃ a : ℝ, f (1 - a) = -1 :=
by
  sorry

end find_f_1_minus_a_l431_43126


namespace sector_radius_l431_43145

theorem sector_radius (l : ℝ) (a : ℝ) (r : ℝ) (h1 : l = 2) (h2 : a = 4) (h3 : a = (1 / 2) * l * r) : r = 4 := by
  sorry

end sector_radius_l431_43145


namespace solve_for_n_l431_43110

theorem solve_for_n (n : ℕ) : (8 ^ n) * (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 4 → n = 2 :=
by 
  intro h
  sorry

end solve_for_n_l431_43110


namespace area_of_black_parts_l431_43158

theorem area_of_black_parts (x y : ℕ) (h₁ : x + y = 106) (h₂ : x + 2 * y = 170) : y = 64 :=
sorry

end area_of_black_parts_l431_43158


namespace proof_m_div_x_plus_y_l431_43151

variables (a b c x y m : ℝ)

-- 1. The ratio of 'a' to 'b' is 4 to 5
axiom h1 : a / b = 4 / 5

-- 2. 'c' is half of 'a'.
axiom h2 : c = a / 2

-- 3. 'x' equals 'a' increased by 27 percent of 'a'.
axiom h3 : x = 1.27 * a

-- 4. 'y' equals 'b' decreased by 16 percent of 'b'.
axiom h4 : y = 0.84 * b

-- 5. 'm' equals 'c' increased by 14 percent of 'c'.
axiom h5 : m = 1.14 * c

theorem proof_m_div_x_plus_y : m / (x + y) = 0.2457 :=
by
  -- Proof goes here
  sorry

end proof_m_div_x_plus_y_l431_43151


namespace min_a_condition_l431_43138

-- Definitions of the conditions
def real_numbers (x : ℝ) := true

def in_interval (a m n : ℝ) : Prop := 0 < n ∧ n < m ∧ m < 1 / a

def inequality (a m n : ℝ) : Prop :=
  (n^(1/m) / m^(1/n) > (n^a) / (m^a))

-- Lean statement
theorem min_a_condition (a m n : ℝ) (h1 : real_numbers m) (h2 : real_numbers n)
    (h3 : in_interval a m n) : inequality a m n ↔ 1 ≤ a :=
sorry

end min_a_condition_l431_43138


namespace A_and_B_work_together_for_49_days_l431_43196

variable (A B : ℝ)
variable (d : ℝ)
variable (fraction_left : ℝ)

def work_rate_A := 1 / 15
def work_rate_B := 1 / 20
def combined_work_rate := work_rate_A + work_rate_B

def fraction_work_completed (d : ℝ) := combined_work_rate * d

theorem A_and_B_work_together_for_49_days
    (A : ℝ := 1 / 15)
    (B : ℝ := 1 / 20)
    (fraction_left : ℝ := 0.18333333333333335) :
    (d : ℝ) → (fraction_work_completed d = 1 - fraction_left) →
    d = 49 :=
by
  sorry

end A_and_B_work_together_for_49_days_l431_43196


namespace maria_savings_after_purchase_l431_43152

theorem maria_savings_after_purchase
  (cost_sweater : ℕ)
  (cost_scarf : ℕ)
  (cost_mittens : ℕ)
  (num_family_members : ℕ)
  (savings : ℕ)
  (total_cost_one_set : ℕ)
  (total_cost_all_sets : ℕ)
  (amount_left : ℕ)
  (h1 : cost_sweater = 35)
  (h2 : cost_scarf = 25)
  (h3 : cost_mittens = 15)
  (h4 : num_family_members = 10)
  (h5 : savings = 800)
  (h6 : total_cost_one_set = cost_sweater + cost_scarf + cost_mittens)
  (h7 : total_cost_all_sets = total_cost_one_set * num_family_members)
  (h8 : amount_left = savings - total_cost_all_sets)
  : amount_left = 50 :=
sorry

end maria_savings_after_purchase_l431_43152


namespace polynomial_expansion_l431_43165

theorem polynomial_expansion (z : ℤ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  -- Provide a proof here
  sorry

end polynomial_expansion_l431_43165


namespace store_revenue_is_1210_l431_43120

noncomputable def shirt_price : ℕ := 10
noncomputable def jeans_price : ℕ := 2 * shirt_price
noncomputable def jacket_price : ℕ := 3 * jeans_price
noncomputable def discounted_jacket_price : ℕ := jacket_price - (jacket_price / 10)

noncomputable def total_revenue : ℕ :=
  20 * shirt_price + 10 * jeans_price + 15 * discounted_jacket_price

theorem store_revenue_is_1210 :
  total_revenue = 1210 :=
by
  sorry

end store_revenue_is_1210_l431_43120


namespace addition_addends_l431_43181

theorem addition_addends (a b : ℕ) (c₁ c₂ : ℕ) (d : ℕ) : 
  a + b = c₁ ∧ a + (b - d) = c₂ ∧ d = 50 ∧ c₁ = 982 ∧ c₂ = 577 → 
  a = 450 ∧ b = 532 :=
by
  sorry

end addition_addends_l431_43181


namespace larger_number_l431_43118

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l431_43118


namespace min_moves_move_stack_from_A_to_F_l431_43179

theorem min_moves_move_stack_from_A_to_F : 
  ∀ (squares : Fin 6) (stack : Fin 15), 
  (∃ moves : Nat, 
    (moves >= 0) ∧ 
    (moves == 49) ∧
    ∀ (a b : Fin 6), 
        ∃ (piece_from : Fin 15) (piece_to : Fin 15), 
        ((piece_from > piece_to) → (a ≠ b)) ∧
        (a == 0) ∧ 
        (b == 5)) :=
sorry

end min_moves_move_stack_from_A_to_F_l431_43179


namespace rectangle_diagonals_not_perpendicular_l431_43172

-- Definition of a rectangle through its properties
structure Rectangle (α : Type _) [LinearOrderedField α] :=
  (angle_eq : ∀ (a : α), a = 90)
  (diagonals_eq : ∀ (d1 d2 : α), d1 = d2)
  (diagonals_bisect : ∀ (d1 d2 : α), d1 / 2 = d2 / 2)

-- Theorem stating that a rectangle's diagonals are not necessarily perpendicular
theorem rectangle_diagonals_not_perpendicular (α : Type _) [LinearOrderedField α] (R : Rectangle α) : 
  ¬ (∀ (d1 d2 : α), d1 * d2 = 0) :=
sorry

end rectangle_diagonals_not_perpendicular_l431_43172


namespace smallest_five_digit_divisible_by_53_l431_43134

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l431_43134


namespace trapezoid_side_lengths_l431_43148

theorem trapezoid_side_lengths
  (isosceles : ∀ (A B C D : ℝ) (height BE : ℝ), height = 2 → BE = 2 → A = 2 * Real.sqrt 2 → D = A → 12 = 0.5 * (B + C) * BE → A = D)
  (area : ∀ (BC AD : ℝ), 12 = 0.5 * (BC + AD) * 2)
  (height : ∀ (BE : ℝ), BE = 2)
  (intersect_right_angle : ∀ (A B C D : ℝ), 90 = 45 + 45) :
  ∃ A B C D, A = 2 * Real.sqrt 2 ∧ B = 4 ∧ C = 8 ∧ D = 2 * Real.sqrt 2 :=
by
  sorry

end trapezoid_side_lengths_l431_43148


namespace valid_bases_for_625_l431_43161

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end valid_bases_for_625_l431_43161


namespace sheets_per_class_per_day_l431_43143

theorem sheets_per_class_per_day
  (weekly_sheets : ℕ)
  (school_days_per_week : ℕ)
  (num_classes : ℕ)
  (h1 : weekly_sheets = 9000)
  (h2 : school_days_per_week = 5)
  (h3 : num_classes = 9) :
  (weekly_sheets / school_days_per_week) / num_classes = 200 :=
by
  sorry

end sheets_per_class_per_day_l431_43143


namespace dave_trips_l431_43105

/-- Dave can only carry 9 trays at a time. -/
def trays_per_trip := 9

/-- Number of trays Dave has to pick up from one table. -/
def trays_from_table1 := 17

/-- Number of trays Dave has to pick up from another table. -/
def trays_from_table2 := 55

/-- Total number of trays Dave has to pick up. -/
def total_trays := trays_from_table1 + trays_from_table2

/-- The number of trips Dave will make. -/
def number_of_trips := total_trays / trays_per_trip

theorem dave_trips :
  number_of_trips = 8 :=
sorry

end dave_trips_l431_43105


namespace intersection_point_l431_43133

variables (g : ℤ → ℤ) (b a : ℤ)
def g_def := ∀ x : ℤ, g x = 4 * x + b
def inv_def := ∀ y : ℤ, g y = -4 → y = a
def point_intersection := ∀ y : ℤ, (g y = -4) → (y = a) → (a = -16 + b)
def solution : ℤ := -4

theorem intersection_point (b a : ℤ) (h₁ : g_def g b) (h₂ : inv_def g a) (h₃ : point_intersection g a b) :
  a = solution :=
  sorry

end intersection_point_l431_43133


namespace sum_of_squares_and_product_l431_43102

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l431_43102


namespace tangent_line_at_pi_l431_43184

theorem tangent_line_at_pi :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin x) → 
  ∀ x, x = Real.pi →
  ∀ y, (y = -x + Real.pi) ↔
        (∀ x, y = -x + Real.pi) := 
  sorry

end tangent_line_at_pi_l431_43184


namespace cone_radius_l431_43114

theorem cone_radius
  (l : ℝ) (CSA : ℝ) (π : ℝ) (r : ℝ)
  (h_l : l = 15)
  (h_CSA : CSA = 141.3716694115407)
  (h_pi : π = Real.pi) :
  r = 3 :=
by
  sorry

end cone_radius_l431_43114


namespace sum_of_binary_digits_of_315_l431_43198

theorem sum_of_binary_digits_of_315 : 
    (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_binary_digits_of_315_l431_43198


namespace initial_bottle_caps_l431_43128

theorem initial_bottle_caps (end_caps : ℕ) (eaten_caps : ℕ) (start_caps : ℕ) 
  (h1 : end_caps = 61) 
  (h2 : eaten_caps = 4) 
  (h3 : start_caps = end_caps + eaten_caps) : 
  start_caps = 65 := 
by 
  sorry

end initial_bottle_caps_l431_43128


namespace inequality_proof_l431_43195

variable (a : ℝ)

theorem inequality_proof (a : ℝ) : 
  (a^2 + a + 2) / (Real.sqrt (a^2 + a + 1)) ≥ 2 :=
sorry

end inequality_proof_l431_43195


namespace problem_part_I_problem_part_II_l431_43164

-- Problem (I)
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : 
  a + b = 2 * c -> (a + b) = 2 * c :=
by
  intros h
  sorry

-- Problem (II)
theorem problem_part_II (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = Real.pi / 3) 
  (h2 : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) 
  (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
  (h4 : a + b = 2 * c) : c = 4 :=
by
  intros
  sorry

end problem_part_I_problem_part_II_l431_43164


namespace min_value_of_parabola_l431_43190

theorem min_value_of_parabola : ∃ x : ℝ, ∀ y : ℝ, y = 3 * x^2 - 18 * x + 244 → y = 217 := by
  sorry

end min_value_of_parabola_l431_43190


namespace increasing_intervals_l431_43116

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 →
    (f x > f (x - ε) ∧ f x < f (x + ε) ∧ x ∈ Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∪ Set.Icc (-Real.pi / 12) 0) :=
sorry

end increasing_intervals_l431_43116


namespace min_inv_sum_l431_43141

theorem min_inv_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 2 * a * 1 + b * 2 = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ (1/a) + (1/b) = 4 :=
sorry

end min_inv_sum_l431_43141


namespace circle_tangent_independence_l431_43155

noncomputable def e1 (r : ℝ) (β : ℝ) := r * Real.tan β
noncomputable def e2 (r : ℝ) (α : ℝ) := r * Real.tan α
noncomputable def e3 (r : ℝ) (β α : ℝ) := r * Real.tan (β - α)

theorem circle_tangent_independence 
  (O : ℝ) (r β α : ℝ) (hβ : β < π / 2) (hα : 0 < α) (hαβ : α < β) :
  (e1 r β) * (e2 r α) * (e3 r β α) / ((e1 r β) - (e2 r α) - (e3 r β α)) = r^2 :=
by
  sorry

end circle_tangent_independence_l431_43155


namespace find_number_l431_43132

theorem find_number (x : ℝ) (h : (1/2) * x + 7 = 17) : x = 20 :=
sorry

end find_number_l431_43132


namespace max_glows_in_time_range_l431_43186

theorem max_glows_in_time_range (start_time end_time : ℤ) (interval : ℤ) (h1 : start_time = 3600 + 3420 + 58) (h2 : end_time = 10800 + 1200 + 47) (h3 : interval = 21) :
  (end_time - start_time) / interval = 236 := 
  sorry

end max_glows_in_time_range_l431_43186


namespace total_players_l431_43140

-- Definitions of the given conditions
def K : ℕ := 10
def Kho_only : ℕ := 40
def Both : ℕ := 5

-- The lean statement that captures the problem of proving the total number of players equals 50
theorem total_players : (K - Both) + Kho_only + Both = 50 :=
by
  -- Placeholder for the proof
  sorry

end total_players_l431_43140


namespace slope_angle_of_perpendicular_line_l431_43100

theorem slope_angle_of_perpendicular_line (h : ∀ x, x = (π / 3)) : ∀ θ, θ = (π / 2) := 
by 
  -- Placeholder for the proof
  sorry

end slope_angle_of_perpendicular_line_l431_43100


namespace total_reams_l431_43154

theorem total_reams (h_r : ℕ) (s_r : ℕ) : h_r = 2 → s_r = 3 → h_r + s_r = 5 :=
by
  intro h_eq s_eq
  sorry

end total_reams_l431_43154


namespace domain_ln_l431_43182

theorem domain_ln (x : ℝ) : x^2 - x - 2 > 0 ↔ (x < -1 ∨ x > 2) := by
  sorry

end domain_ln_l431_43182


namespace find_three_digit_number_l431_43157

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end find_three_digit_number_l431_43157


namespace inequality_am_gm_l431_43177

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l431_43177


namespace arithmetic_sequence_inequality_l431_43193

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality 
  (h : is_arithmetic_sequence a d)
  (d_pos : d ≠ 0)
  (a_pos : ∀ n, a n > 0) :
  (a 1) * (a 8) < (a 4) * (a 5) := 
by
  sorry

end arithmetic_sequence_inequality_l431_43193


namespace length_of_other_side_l431_43104

-- Defining the conditions
def roofs := 3
def sides_per_roof := 2
def length_of_one_side := 40 -- measured in feet
def shingles_per_square_foot := 8
def total_shingles := 38400

-- The proof statement
theorem length_of_other_side : 
    ∃ (L : ℕ), (total_shingles / shingles_per_square_foot / roofs / sides_per_roof = 40 * L) ∧ L = 20 :=
by
  sorry

end length_of_other_side_l431_43104


namespace train_passes_man_in_correct_time_l431_43163

-- Definitions for the given conditions
def platform_length : ℝ := 270
def train_length : ℝ := 180
def crossing_time : ℝ := 20

-- Theorem to prove the time taken to pass the man is 8 seconds
theorem train_passes_man_in_correct_time
  (p: ℝ) (l: ℝ) (t_cross: ℝ)
  (h1: p = platform_length)
  (h2: l = train_length)
  (h3: t_cross = crossing_time) :
  l / ((l + p) / t_cross) = 8 := by
  -- Proof goes here
  sorry

end train_passes_man_in_correct_time_l431_43163


namespace range_of_m_l431_43117

open Set

noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + m * x - y + 2 = 0} 

noncomputable def B : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → (m ≤ -1 ∨ m ≥ 3) := 
sorry

end range_of_m_l431_43117


namespace necessary_not_sufficient_condition_l431_43135

-- Definitions of conditions
variable (x : ℝ)

-- Statement of the problem in Lean 4
theorem necessary_not_sufficient_condition (h : |x - 1| ≤ 1) : 2 - x ≥ 0 := sorry

end necessary_not_sufficient_condition_l431_43135


namespace fill_trough_time_l431_43130

noncomputable def time_to_fill (T_old T_new T_third : ℕ) : ℝ :=
  let rate_old := (1 : ℝ) / T_old
  let rate_new := (1 : ℝ) / T_new
  let rate_third := (1 : ℝ) / T_third
  let total_rate := rate_old + rate_new + rate_third
  1 / total_rate

theorem fill_trough_time:
  time_to_fill 600 200 400 = 1200 / 11 := 
by
  sorry

end fill_trough_time_l431_43130


namespace smallest_four_digit_multiple_of_8_with_digit_sum_20_l431_43160

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.foldl (· + ·) 0

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20:
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ sum_of_digits n = 20 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 8 = 0 ∧ sum_of_digits m = 20 → n ≤ m :=
by { sorry }

end smallest_four_digit_multiple_of_8_with_digit_sum_20_l431_43160


namespace max_value_sum_seq_l431_43112

theorem max_value_sum_seq : 
  ∃ a1 a2 a3 a4 : ℝ, 
    a1 = 0 ∧ 
    |a2| = |a1 - 1| ∧ 
    |a3| = |a2 - 1| ∧ 
    |a4| = |a3 - 1| ∧ 
    a1 + a2 + a3 + a4 = 2 := 
by 
  sorry

end max_value_sum_seq_l431_43112


namespace intersection_M_N_l431_43174

def set_M : Set ℝ := { x | x * (x - 1) ≤ 0 }
def set_N : Set ℝ := { x | x < 1 }

theorem intersection_M_N : set_M ∩ set_N = { x | 0 ≤ x ∧ x < 1 } := sorry

end intersection_M_N_l431_43174


namespace sum_a5_a8_l431_43122

variable (a : ℕ → ℝ)
variable (r : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_a5_a8 (a1 a2 a3 a4 : ℝ) (q : ℝ)
  (h1 : a1 + a3 = 1)
  (h2 : a2 + a4 = 2)
  (h_seq : is_geometric_sequence a q)
  (a_def : ∀ n : ℕ, a n = a1 * q^n) :
  a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end sum_a5_a8_l431_43122


namespace sammy_pickles_l431_43136

theorem sammy_pickles 
  (T S R : ℕ) 
  (h1 : T = 2 * S) 
  (h2 : R = 8 * T / 10) 
  (h3 : R = 24) : 
  S = 15 :=
by
  sorry

end sammy_pickles_l431_43136


namespace log_comparison_l431_43124

noncomputable def logBase (a x : ℝ) := Real.log x / Real.log a

theorem log_comparison
  (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) 
  (m : ℝ) (hm : m = logBase a (a^2 + 1))
  (n : ℝ) (hn : n = logBase a (a + 1))
  (p : ℝ) (hp : p = logBase a (2 * a)) :
  p > m ∧ m > n :=
by
  sorry

end log_comparison_l431_43124


namespace fraction_value_l431_43129

variable (x y : ℝ)

theorem fraction_value (hx : x = 4) (hy : y = -3) : (x - 2 * y) / (x + y) = 10 := by
  sorry

end fraction_value_l431_43129


namespace problem1_problem2_l431_43139

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l431_43139


namespace polynomial_product_c_l431_43167

theorem polynomial_product_c (b c : ℝ) (h1 : b = 2 * c - 1) (h2 : (x^2 + b * x + c) = 0 → (∃ r : ℝ, x = r)) :
  c = 1 / 2 :=
sorry

end polynomial_product_c_l431_43167


namespace sum_of_a_b_l431_43197

-- Definitions for the given conditions
def geom_series_sum (a : ℤ) (n : ℕ) : ℤ := 2^n + a
def arith_series_sum (b : ℤ) (n : ℕ) : ℤ := n^2 - 2*n + b

-- Theorem statement
theorem sum_of_a_b (a b : ℤ) (h1 : ∀ n, geom_series_sum a n = 2^n + a)
  (h2 : ∀ n, arith_series_sum b n = n^2 - 2*n + b) :
  a + b = -1 :=
sorry

end sum_of_a_b_l431_43197


namespace brenda_sally_track_length_l431_43101

theorem brenda_sally_track_length
  (c d : ℝ) 
  (h1 : c / 4 * 3 = d) 
  (h2 : d - 120 = 0.75 * c - 120) 
  (h3 : 0.75 * c + 60 <= 1.25 * c - 180) 
  (h4 : (c - 120 + 0.25 * c - 60) = 1.25 * c - 180):
  c = 766.67 :=
sorry

end brenda_sally_track_length_l431_43101


namespace x_finishes_in_24_days_l431_43156

variable (x y : Type) [Inhabited x] [Inhabited y]

/-- 
  y can finish the work in 16 days,
  y worked for 10 days and left the job,
  x alone needs 9 days to finish the remaining work,
  How many days does x need to finish the work alone?
-/
theorem x_finishes_in_24_days
  (days_y : ℕ := 16)
  (work_done_y : ℕ := 10)
  (work_left_x : ℕ := 9)
  (D_x : ℕ) :
  (1 / days_y : ℚ) * work_done_y + (1 / D_x) * work_left_x = 1 / D_x :=
by
  sorry

end x_finishes_in_24_days_l431_43156


namespace final_number_is_odd_l431_43183

theorem final_number_is_odd : 
  ∃ (n : ℤ), n % 2 = 1 ∧ n ≥ 1 ∧ n < 1024 := sorry

end final_number_is_odd_l431_43183


namespace distance_to_school_l431_43192

variable (v d : ℝ) -- typical speed (v) and distance (d)

theorem distance_to_school :
  (30 / 60 : ℝ) = 1 / 2 ∧ -- 30 minutes is 1/2 hour
  (18 / 60 : ℝ) = 3 / 10 ∧ -- 18 minutes is 3/10 hour
  d = v * (1 / 2) ∧ -- distance for typical day
  d = (v + 12) * (3 / 10) -- distance for quieter day
  → d = 9 := sorry

end distance_to_school_l431_43192


namespace arithmetic_seq_common_diff_l431_43107

theorem arithmetic_seq_common_diff
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geomet : (a 3) ^ 2 = a 1 * a 13) :
  d = 2 :=
by
  sorry

end arithmetic_seq_common_diff_l431_43107


namespace total_hamburgers_menu_l431_43103

def meat_patties_choices := 4
def condiment_combinations := 2 ^ 9

theorem total_hamburgers_menu :
  meat_patties_choices * condiment_combinations = 2048 :=
by
  sorry

end total_hamburgers_menu_l431_43103


namespace third_side_integer_lengths_l431_43106

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l431_43106


namespace geometric_sequence_a4_value_l431_43125

theorem geometric_sequence_a4_value 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h1 : a 1 + (2 / 3) * a 2 = 3) 
  (h2 : a 4^2 = (1 / 9) * a 3 * a 7) 
  :
  a 4 = 27 :=
sorry

end geometric_sequence_a4_value_l431_43125


namespace remainder_product_mod_5_l431_43111

theorem remainder_product_mod_5 (a b c : ℕ) (h_a : a % 5 = 2) (h_b : b % 5 = 3) (h_c : c % 5 = 4) :
  (a * b * c) % 5 = 4 := 
by
  sorry

end remainder_product_mod_5_l431_43111


namespace find_incomes_l431_43173

theorem find_incomes (M N O P Q : ℝ) 
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (O + P) / 2 = 6800)
  (h4 : (P + Q) / 2 = 7500)
  (h5 : (M + O + Q) / 3 = 6000) :
  M = 300 ∧ N = 9800 ∧ O = 2700 ∧ P = 10900 ∧ Q = 4100 :=
by
  sorry


end find_incomes_l431_43173


namespace marcus_calzones_total_time_l431_43153

/-
Conditions:
1. It takes Marcus 20 minutes to saute the onions.
2. It takes a quarter of the time to saute the garlic and peppers that it takes to saute the onions.
3. It takes 30 minutes to knead the dough.
4. It takes twice as long to let the dough rest as it takes to knead it.
5. It takes 1/10th of the combined kneading and resting time to assemble the calzones.
-/

def time_saute_onions : ℕ := 20
def time_saute_garlic_peppers : ℕ := time_saute_onions / 4
def time_knead : ℕ := 30
def time_rest : ℕ := 2 * time_knead
def time_assemble : ℕ := (time_knead + time_rest) / 10

def total_time_making_calzones : ℕ :=
  time_saute_onions + time_saute_garlic_peppers + time_knead + time_rest + time_assemble

theorem marcus_calzones_total_time : total_time_making_calzones = 124 := by
  -- All steps and proof details to be filled in
  sorry

end marcus_calzones_total_time_l431_43153


namespace vector_eq_to_slope_intercept_form_l431_43142

theorem vector_eq_to_slope_intercept_form :
  ∀ (x y : ℝ), (2 * (x - 4) + 5 * (y - 1)) = 0 → y = -(2 / 5) * x + 13 / 5 := 
by 
  intros x y h
  sorry

end vector_eq_to_slope_intercept_form_l431_43142


namespace james_two_point_shots_l431_43175

-- Definitions based on conditions
def field_goals := 13
def field_goal_points := 3
def total_points := 79

-- Statement to be proven
theorem james_two_point_shots :
  ∃ x : ℕ, 79 = (field_goals * field_goal_points) + (2 * x) ∧ x = 20 :=
by
  sorry

end james_two_point_shots_l431_43175


namespace x_minus_y_eq_neg_200_l431_43150

theorem x_minus_y_eq_neg_200 (x y : ℤ) (h1 : x + y = 290) (h2 : y = 245) : x - y = -200 := by
  sorry

end x_minus_y_eq_neg_200_l431_43150


namespace determinant_of_matrix4x5_2x3_l431_43170

def matrix4x5_2x3 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 5], ![2, 3]]

theorem determinant_of_matrix4x5_2x3 : matrix4x5_2x3.det = 2 := 
by
  sorry

end determinant_of_matrix4x5_2x3_l431_43170


namespace projects_count_minimize_time_l431_43119

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end projects_count_minimize_time_l431_43119


namespace compute_floor_expression_l431_43121

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end compute_floor_expression_l431_43121


namespace find_LCM_of_numbers_l431_43144

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers_l431_43144


namespace difference_of_squares_l431_43187

theorem difference_of_squares : 73^2 - 47^2 = 3120 :=
by sorry

end difference_of_squares_l431_43187


namespace minimize_on_interval_l431_43162

def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

theorem minimize_on_interval (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≥ if a < 0 then -2 else if 0 ≤ a ∧ a ≤ 2 then -a^2 - 2 else 2 - 4*a) :=
by 
  sorry

end minimize_on_interval_l431_43162


namespace minimum_value_of_C_over_D_is_three_l431_43147

variable (x : ℝ) (C D : ℝ)
variables (hxC : x^3 + 1/(x^3) = C) (hxD : x - 1/(x) = D)

theorem minimum_value_of_C_over_D_is_three (hC : C = D^3 + 3 * D) :
  ∃ x : ℝ, x^3 + 1/(x^3) = C ∧ x - 1/(x) = D → C / D ≥ 3 :=
by
  sorry

end minimum_value_of_C_over_D_is_three_l431_43147


namespace find_x_l431_43166

theorem find_x (x : ℕ) (h : 2^x - 2^(x - 2) = 3 * 2^10) : x = 12 :=
by 
  sorry

end find_x_l431_43166


namespace distance_from_center_to_line_of_tangent_circle_l431_43189

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l431_43189


namespace domain_of_function_l431_43149

theorem domain_of_function :
  {x : ℝ | ∀ k : ℤ, 2 * x + (π / 4) ≠ k * π + (π / 2)}
  = {x : ℝ | ∀ k : ℤ, x ≠ (k * π / 2) + (π / 8)} :=
sorry

end domain_of_function_l431_43149


namespace inequality_holds_for_all_x_iff_a_in_interval_l431_43159

theorem inequality_holds_for_all_x_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end inequality_holds_for_all_x_iff_a_in_interval_l431_43159


namespace fraction_sum_l431_43109

theorem fraction_sum : (3 / 8) + (9 / 12) + (5 / 6) = 47 / 24 := by
  sorry

end fraction_sum_l431_43109


namespace arithmetic_sequence_n_15_l431_43113

theorem arithmetic_sequence_n_15 (a : ℕ → ℤ) (n : ℕ)
  (h₁ : a 3 = 5)
  (h₂ : a 2 + a 5 = 12)
  (h₃ : a n = 29) :
  n = 15 :=
sorry

end arithmetic_sequence_n_15_l431_43113
