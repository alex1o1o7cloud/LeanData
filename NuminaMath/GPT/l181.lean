import Mathlib

namespace value_of_expression_l181_18185

theorem value_of_expression (a b c k : ℕ) (h_a : a = 30) (h_b : b = 25) (h_c : c = 4) (h_k : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 :=
by
  rw [h_a, h_b, h_c, h_k]
  simp
  sorry

end value_of_expression_l181_18185


namespace g_at_3_l181_18195

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2) : g 3 = 0 := by
  sorry

end g_at_3_l181_18195


namespace sum_max_min_values_of_g_l181_18156

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_max_min_values_of_g : (∀ x, 1 ≤ x ∧ x ≤ 7 → g x = 15 - 2 * x ∨ g x = 5) ∧ 
      (g 1 = 13 ∧ g 5 = 5)
      → (13 + 5 = 18) :=
by
  sorry

end sum_max_min_values_of_g_l181_18156


namespace melted_mixture_weight_l181_18153

-- Let Zinc and Copper be real numbers representing their respective weights in kilograms.
variables (Zinc Copper: ℝ)
-- Assume the ratio of Zinc to Copper is 9:11.
axiom ratio_zinc_copper : Zinc / Copper = 9 / 11
-- Assume 26.1kg of Zinc has been used.
axiom zinc_value : Zinc = 26.1

-- Define the total weight of the melted mixture.
def total_weight := Zinc + Copper

-- We state the theorem to prove that the total weight of the mixture equals 58kg.
theorem melted_mixture_weight : total_weight Zinc Copper = 58 :=
by
  sorry

end melted_mixture_weight_l181_18153


namespace income_M_l181_18193

variable (M N O : ℝ)

theorem income_M (h1 : (M + N) / 2 = 5050) 
                  (h2 : (N + O) / 2 = 6250) 
                  (h3 : (M + O) / 2 = 5200) : 
                  M = 2666.67 := 
by 
  sorry

end income_M_l181_18193


namespace arithmetic_sequence_sum_l181_18123

variable (a : ℕ → ℕ)
variable (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 2 - a 1)

theorem arithmetic_sequence_sum (h : a 2 + a 8 = 6) : 
  1 / 2 * 9 * (a 1 + a 9) = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l181_18123


namespace tv_sets_sales_decrease_l181_18122

theorem tv_sets_sales_decrease
  (P Q P' Q' R R': ℝ)
  (h1 : P' = 1.6 * P)
  (h2 : R' = 1.28 * R)
  (h3 : R = P * Q)
  (h4 : R' = P' * Q')
  (h5 : Q' = Q * (1 - D / 100)) :
  D = 20 :=
by
  sorry

end tv_sets_sales_decrease_l181_18122


namespace reciprocal_self_eq_one_or_neg_one_l181_18169

theorem reciprocal_self_eq_one_or_neg_one (x : ℝ) (h : x = 1 / x) : x = 1 ∨ x = -1 := sorry

end reciprocal_self_eq_one_or_neg_one_l181_18169


namespace arcsin_cos_arcsin_rel_arccos_sin_arccos_l181_18111

theorem arcsin_cos_arcsin_rel_arccos_sin_arccos (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
    let α := Real.arcsin (Real.cos (Real.arcsin x))
    let β := Real.arccos (Real.sin (Real.arccos x))
    (Real.arcsin x + Real.arccos x = π / 2) → α + β = π / 2 :=
by
  let α := Real.arcsin (Real.cos (Real.arcsin x))
  let β := Real.arccos (Real.sin (Real.arccos x))
  intro h_arcsin_arccos_eq
  sorry

end arcsin_cos_arcsin_rel_arccos_sin_arccos_l181_18111


namespace find_a9_l181_18178

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => a n + n

theorem find_a9 : a 9 = 37 := by
  sorry

end find_a9_l181_18178


namespace quadrilateral_area_sum_l181_18146

theorem quadrilateral_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a^2 * b = 36) : a + b = 4 := 
sorry

end quadrilateral_area_sum_l181_18146


namespace pell_eq_unique_fund_sol_l181_18184

theorem pell_eq_unique_fund_sol (x y x_0 y_0 : ℕ) 
  (h1 : x_0^2 - 2003 * y_0^2 = 1) 
  (h2 : ∀ x y, x > 0 ∧ y > 0 → x^2 - 2003 * y^2 = 1 → ∃ n : ℕ, x + Real.sqrt 2003 * y = (x_0 + Real.sqrt 2003 * y_0)^n)
  (hx_pos : x > 0) 
  (hy_pos : y > 0)
  (h_sol : x^2 - 2003 * y^2 = 1) 
  (hprime : ∀ p : ℕ, Prime p → p ∣ x → p ∣ x_0)
  : x = x_0 ∧ y = y_0 :=
sorry

end pell_eq_unique_fund_sol_l181_18184


namespace total_students_l181_18198

variable (A B AB : ℕ)

-- Conditions
axiom h1 : AB = (1 / 5) * (A + AB)
axiom h2 : AB = (1 / 4) * (B + AB)
axiom h3 : A - B = 75

-- Proof problem
theorem total_students : A + B + AB = 600 :=
by
  sorry

end total_students_l181_18198


namespace max_profit_l181_18125

noncomputable def C (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def L (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then -(1 / 3) * x^2 + 40 * x - 250
  else -(x + 10000 / x) + 1200

theorem max_profit :
  ∃ x : ℝ, (L x) = 1000 ∧ x = 100 :=
by
  sorry

end max_profit_l181_18125


namespace find_k_l181_18189

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem find_k :
  ∃ k : ℕ, digit_sum (5 * (5 * (10 ^ (k - 1) - 1) / 9)) = 600 ∧ k = 87 :=
by
  sorry

end find_k_l181_18189


namespace zoo_total_animals_l181_18149

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l181_18149


namespace sum_of_ages_l181_18162

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end sum_of_ages_l181_18162


namespace gcd_84_210_l181_18144

theorem gcd_84_210 : Nat.gcd 84 210 = 42 :=
by {
  sorry
}

end gcd_84_210_l181_18144


namespace min_value_of_a_plus_b_l181_18188

theorem min_value_of_a_plus_b (a b : ℤ) (h_ab : a * b = 72) (h_even : a % 2 = 0) : a + b ≥ -38 :=
sorry

end min_value_of_a_plus_b_l181_18188


namespace mailman_junk_mail_l181_18132

theorem mailman_junk_mail (total_mail : ℕ) (magazines : ℕ) (junk_mail : ℕ) 
  (h1 : total_mail = 11) (h2 : magazines = 5) (h3 : junk_mail = total_mail - magazines) : junk_mail = 6 := by
  sorry

end mailman_junk_mail_l181_18132


namespace multiple_of_6_is_multiple_of_2_and_3_l181_18190

theorem multiple_of_6_is_multiple_of_2_and_3 (n : ℕ) :
  (∃ k : ℕ, n = 6 * k) → (∃ m1 : ℕ, n = 2 * m1) ∧ (∃ m2 : ℕ, n = 3 * m2) := by
  sorry

end multiple_of_6_is_multiple_of_2_and_3_l181_18190


namespace arithmetic_sequence_sum_l181_18141

theorem arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (c : ℤ) :
  (∀ n : ℕ, 0 < n → S_n n = n^2 + c) →
  a_n 1 = 1 + c →
  (∀ n, 1 < n → a_n n = S_n n - S_n (n - 1)) →
  (∀ n : ℕ, 0 < n → a_n n = 1 + (n - 1) * 2) →
  c = 0 ∧ (∀ n : ℕ, 0 < n → a_n n = 2 * n - 1) :=
by
  sorry

end arithmetic_sequence_sum_l181_18141


namespace solve_for_b_l181_18102

theorem solve_for_b (b : ℝ) (m : ℝ) (h : b > 0)
  (h1 : ∀ x : ℝ, x^2 + b * x + 54 = (x + m) ^ 2 + 18) : b = 12 :=
by
  sorry

end solve_for_b_l181_18102


namespace correlate_height_weight_l181_18170

-- Define the problems as types
def heightWeightCorrelated : Prop := true
def distanceTimeConstantSpeed : Prop := true
def heightVisionCorrelated : Prop := false
def volumeEdgeLengthCorrelated : Prop := true

-- Define the equivalence for the problem
def correlated : Prop := heightWeightCorrelated

-- Now state that correlated == heightWeightCorrelated
theorem correlate_height_weight : correlated = heightWeightCorrelated :=
by sorry

end correlate_height_weight_l181_18170


namespace number_below_267_is_301_l181_18139

-- Define the row number function
def rowNumber (n : ℕ) : ℕ :=
  Nat.sqrt n + 1

-- Define the starting number of a row
def rowStart (k : ℕ) : ℕ :=
  (k - 1) * (k - 1) + 1

-- Define the number in the row below given a number and its position in the row
def numberBelow (n : ℕ) : ℕ :=
  let k := rowNumber n
  let startK := rowStart k
  let position := n - startK
  let startNext := rowStart (k + 1)
  startNext + position

-- Prove that the number below 267 is 301
theorem number_below_267_is_301 : numberBelow 267 = 301 :=
by
  -- skip proof details, just the statement is needed
  sorry

end number_below_267_is_301_l181_18139


namespace percentage_increase_l181_18126

theorem percentage_increase (regular_rate : ℝ) (regular_hours total_compensation total_hours_worked : ℝ)
  (h1 : regular_rate = 20)
  (h2 : regular_hours = 40)
  (h3 : total_compensation = 1000)
  (h4 : total_hours_worked = 45.714285714285715) :
  let overtime_hours := total_hours_worked - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_compensation - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  let percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100
  percentage_increase = 75 := 
by
  sorry

end percentage_increase_l181_18126


namespace inequality_solution_l181_18194

theorem inequality_solution (x : ℝ) :
  (6*x^2 + 24*x - 63) / ((3*x - 4)*(x + 5)) < 4 ↔ x ∈ Set.Ioo (-(5:ℝ)) (4 / 3) ∪ Set.Iio (5) ∪ Set.Ioi (4 / 3) := by
  sorry

end inequality_solution_l181_18194


namespace coordinates_of_A_l181_18166

-- Defining the point A
def point_A : ℤ × ℤ := (1, -4)

-- Statement that needs to be proved
theorem coordinates_of_A :
  point_A = (1, -4) :=
by
  sorry

end coordinates_of_A_l181_18166


namespace magnitude_of_angle_B_value_of_k_l181_18105

-- Define the conditions and corresponding proofs

variable {a b c : ℝ}
variable {A B C : ℝ} -- Angles in the triangle
variable (k : ℝ) -- Define k
variable (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) -- Given condition for part 1
variable (h2 : (A + B + C) = Real.pi) -- Angle sum in triangle
variable (h3 : k > 1) -- Condition for part 2
variable (m_dot_n_max : ∀ (t : ℝ), 4 * k * t + Real.cos (2 * Real.arcsin t) = 5) -- Given condition for part 2

-- Proofs Required

theorem magnitude_of_angle_B (hA : 0 < A ∧ A < Real.pi) : B = Real.pi / 3 :=
by 
  sorry -- proof to be completed

theorem value_of_k : k = 3 / 2 :=
by 
  sorry -- proof to be completed

end magnitude_of_angle_B_value_of_k_l181_18105


namespace arithmetic_sequence_a8_l181_18154

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a8 (h : a 7 + a 9 = 8) : a 8 = 4 := 
by 
  -- proof steps would go here
  sorry

end arithmetic_sequence_a8_l181_18154


namespace find_f4_l181_18158

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def equilibrium_condition (f4 : ℝ × ℝ) : Prop :=
  f1 + f2 + f3 + f4 = (0, 0)

-- Statement that needs to be proven
theorem find_f4 : ∃ (f4 : ℝ × ℝ), equilibrium_condition f4 :=
  by
  use (1, 2)
  sorry

end find_f4_l181_18158


namespace max_value_2x_minus_y_l181_18157

theorem max_value_2x_minus_y 
  (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  2 * x - y ≤ 1 :=
sorry

end max_value_2x_minus_y_l181_18157


namespace simplify_expression_l181_18118

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := 
by 
  sorry

end simplify_expression_l181_18118


namespace number_of_tiles_l181_18182

open Real

noncomputable def room_length : ℝ := 10
noncomputable def room_width : ℝ := 15
noncomputable def tile_length : ℝ := 5 / 12
noncomputable def tile_width : ℝ := 2 / 3

theorem number_of_tiles :
  (room_length * room_width) / (tile_length * tile_width) = 540 := by
  sorry

end number_of_tiles_l181_18182


namespace greatest_term_in_expansion_l181_18177

theorem greatest_term_in_expansion :
  ∃ k : ℕ, k = 63 ∧
  (∀ n : ℕ, n ∈ (Finset.range 101) → n ≠ k → 
    (Nat.choose 100 n * (Real.sqrt 3)^n) < 
    (Nat.choose 100 k * (Real.sqrt 3)^k)) :=
by
  sorry

end greatest_term_in_expansion_l181_18177


namespace unique_solution_for_4_circ_20_l181_18113

def operation (x y : ℝ) : ℝ := 3 * x - 2 * y + 2 * x * y

theorem unique_solution_for_4_circ_20 : ∃! y : ℝ, operation 4 y = 20 :=
by 
  sorry

end unique_solution_for_4_circ_20_l181_18113


namespace num_ordered_triples_l181_18145

theorem num_ordered_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ∣ b ∧ a ∣ c ∧ a + b + c = 100) :=
  sorry

end num_ordered_triples_l181_18145


namespace cube_triangulation_impossible_l181_18106

theorem cube_triangulation_impossible (vertex_sum : ℝ) (triangle_inter_sum : ℝ) (triangle_sum : ℝ) :
  vertex_sum = 270 ∧ triangle_inter_sum = 360 ∧ triangle_sum = 180 → ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), m ≠ 3 → false :=
by
  sorry

end cube_triangulation_impossible_l181_18106


namespace min_abs_y1_minus_4y2_l181_18199

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)

noncomputable def equation_of_line (k y : ℝ) : ℝ := k * y + 1

-- The Lean theorem statement
theorem min_abs_y1_minus_4y2 {x1 y1 x2 y2 : ℝ} (H1 : parabola x1 y1) (H2 : parabola x2 y2)
    (A_in_first_quadrant : 0 < x1 ∧ 0 < y1)
    (line_through_focus : ∃ k : ℝ, x1 = equation_of_line k y1 ∧ x2 = equation_of_line k y2)
    : |y1 - 4 * y2| = 8 :=
sorry

end min_abs_y1_minus_4y2_l181_18199


namespace change_is_24_l181_18197

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end change_is_24_l181_18197


namespace find_segment_XY_length_l181_18104

theorem find_segment_XY_length (A B C D X Y : Type) 
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq X] [DecidableEq Y]
  (line_l : Type) (BX : ℝ) (DY : ℝ) (AB : ℝ) (BC : ℝ) (l : line_l)
  (hBX : BX = 4) (hDY : DY = 10) (hBC : BC = 2 * AB) :
  XY = 13 :=
  sorry

end find_segment_XY_length_l181_18104


namespace range_of_k_l181_18142

noncomputable def quadratic_has_real_roots (k : ℝ): Prop :=
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l181_18142


namespace other_root_of_quadratic_l181_18164

theorem other_root_of_quadratic (a b k : ℝ) (h : 1^2 - (a+b) * 1 + ab * (1 - k) = 0) : 
  ∃ r : ℝ, r = a + b - 1 := 
sorry

end other_root_of_quadratic_l181_18164


namespace find_x_l181_18196

theorem find_x (x : ℝ) (hx : x > 0) (condition : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 :=
by
  sorry

end find_x_l181_18196


namespace find_value_of_a_l181_18187

noncomputable def log_base_four (a : ℝ) : ℝ := Real.log a / Real.log 4

theorem find_value_of_a (a : ℝ) (h : log_base_four a = (1 : ℝ) / (2 : ℝ)) : a = 2 := by
  sorry

end find_value_of_a_l181_18187


namespace find_original_amount_l181_18168

-- Let X be the original amount of money in Christina's account.
variable (X : ℝ)

-- Condition 1: Remaining balance after transferring 20% is $30,000.
def initial_transfer (X : ℝ) : Prop :=
  0.80 * X = 30000

-- Prove that the original amount before the initial transfer was $37,500.
theorem find_original_amount (h : initial_transfer X) : X = 37500 :=
  sorry

end find_original_amount_l181_18168


namespace g_1000_is_1820_l181_18110

-- Definitions and conditions from the problem
def g (n : ℕ) : ℕ := sorry -- exact definition is unknown, we will assume conditions

-- Conditions as given
axiom g_g (n : ℕ) : g (g n) = 3 * n
axiom g_3n_plus_1 (n : ℕ) : g (3 * n + 1) = 3 * n + 2

-- Statement to prove
theorem g_1000_is_1820 : g 1000 = 1820 :=
by
  sorry

end g_1000_is_1820_l181_18110


namespace rectangle_right_triangle_max_area_and_hypotenuse_l181_18108

theorem rectangle_right_triangle_max_area_and_hypotenuse (x y h : ℝ) (h_triangle : h^2 = x^2 + y^2) (h_perimeter : 2 * (x + y) = 60) :
  (x * y ≤ 225) ∧ (x = 15) ∧ (y = 15) ∧ (h = 15 * Real.sqrt 2) :=
by
  sorry

end rectangle_right_triangle_max_area_and_hypotenuse_l181_18108


namespace negation_of_p_l181_18163

variable (p : Prop) (n : ℕ)

def proposition_p := ∃ n : ℕ, n^2 > 2^n

theorem negation_of_p : ¬ proposition_p ↔ ∀ n : ℕ, n^2 <= 2^n :=
by
  sorry

end negation_of_p_l181_18163


namespace unique_solution_l181_18117

theorem unique_solution (n : ℕ) (h1 : n > 0) (h2 : n^2 ∣ 3^n + 1) : n = 1 :=
sorry

end unique_solution_l181_18117


namespace range_f_3_l181_18103

section

variables (a c : ℝ) (f : ℝ → ℝ)
def quadratic_function := ∀ x, f x = a * x^2 - c

-- Define the constraints given in the problem
axiom h1 : -4 ≤ f 1 ∧ f 1 ≤ -1
axiom h2 : -1 ≤ f 2 ∧ f 2 ≤ 5

-- Prove that the correct range for f(3) is -1 ≤ f(3) ≤ 20
theorem range_f_3 (a c : ℝ) (f : ℝ → ℝ) (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5):
  -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end

end range_f_3_l181_18103


namespace zach_fill_time_l181_18174

theorem zach_fill_time : 
  ∀ (t : ℕ), 
  (∀ (max_time max_rate zach_rate popped total : ℕ), 
    max_time = 30 → 
    max_rate = 2 → 
    zach_rate = 3 → 
    popped = 10 → 
    total = 170 → 
    (max_time * max_rate + t * zach_rate - popped = total) → 
    t = 40) := 
sorry

end zach_fill_time_l181_18174


namespace john_total_time_l181_18115

noncomputable def total_time_spent : ℝ :=
  let landscape_pictures := 10
  let landscape_drawing_time := 2
  let landscape_coloring_time := landscape_drawing_time * 0.7
  let landscape_enhancing_time := 0.75
  let total_landscape_time := (landscape_drawing_time + landscape_coloring_time + landscape_enhancing_time) * landscape_pictures
  
  let portrait_pictures := 15
  let portrait_drawing_time := 3
  let portrait_coloring_time := portrait_drawing_time * 0.75
  let portrait_enhancing_time := 1.0
  let total_portrait_time := (portrait_drawing_time + portrait_coloring_time + portrait_enhancing_time) * portrait_pictures
  
  let abstract_pictures := 20
  let abstract_drawing_time := 1.5
  let abstract_coloring_time := abstract_drawing_time * 0.6
  let abstract_enhancing_time := 0.5
  let total_abstract_time := (abstract_drawing_time + abstract_coloring_time + abstract_enhancing_time) * abstract_pictures
  
  total_landscape_time + total_portrait_time + total_abstract_time

theorem john_total_time : total_time_spent = 193.25 :=
by sorry

end john_total_time_l181_18115


namespace tank_ratio_l181_18159

theorem tank_ratio (V1 V2 : ℝ) (h1 : 0 < V1) (h2 : 0 < V2) (h1_full : 3 / 4 * V1 - 7 / 20 * V2 = 0) (h2_full : 1 / 4 * V2 + 7 / 20 * V2 = 3 / 5 * V2) :
  V1 / V2 = 7 / 9 :=
by
  sorry

end tank_ratio_l181_18159


namespace problem_statement_l181_18172

theorem problem_statement (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104 :=
by
  sorry

end problem_statement_l181_18172


namespace speed_in_still_water_l181_18183

-- We define the given conditions for the man's rowing speeds
def upstream_speed : ℕ := 25
def downstream_speed : ℕ := 35

-- We want to prove that the speed in still water is 30 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 := by
  sorry

end speed_in_still_water_l181_18183


namespace vector_parallel_condition_l181_18150

def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (2 * m, m + 1)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_parallel_condition (m : ℝ) (h_parallel : AB OA OB = (3, 1) ∧ 
    (∀ k : ℝ, 2*m = 3*k ∧ m + 1 = k)) : m = -3 :=
by
  sorry

end vector_parallel_condition_l181_18150


namespace compare_exponents_l181_18134

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem compare_exponents : b < a ∧ a < c :=
by
  have h1 : a = 2^(4/3) := rfl
  have h2 : b = 4^(2/5) := rfl
  have h3 : c = 25^(1/3) := rfl
  -- These are used to indicate the definitions, not the proof steps
  sorry

end compare_exponents_l181_18134


namespace point_in_second_quadrant_l181_18181

def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant : is_in_second_quadrant (-2) 3 :=
by
  sorry

end point_in_second_quadrant_l181_18181


namespace find_total_sales_l181_18148

theorem find_total_sales
  (S : ℝ)
  (h_comm1 : ∀ x, x ≤ 5000 → S = 0.9 * x → S = 16666.67 → false)
  (h_comm2 : S > 5000 → S - (500 + 0.05 * (S - 5000)) = 15000):
  S = 16052.63 :=
by
  sorry

end find_total_sales_l181_18148


namespace arithmetic_sequence_contains_term_l181_18128

theorem arithmetic_sequence_contains_term (a1 : ℤ) (d : ℤ) (k : ℕ) (h1 : a1 = 3) (h2 : d = 9) :
  ∃ n : ℕ, (a1 + (n - 1) * d) = 3 * 4 ^ k := by
  sorry

end arithmetic_sequence_contains_term_l181_18128


namespace negation_of_p_equiv_h_l181_18133

variable (p : ∀ x : ℝ, Real.sin x ≤ 1)
variable (h : ∃ x : ℝ, Real.sin x ≥ 1)

theorem negation_of_p_equiv_h : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x ≥ 1) :=
by
  sorry

end negation_of_p_equiv_h_l181_18133


namespace cos_alpha_add_beta_over_two_l181_18101

theorem cos_alpha_add_beta_over_two (
  α β : ℝ) 
  (h1 : 0 < α ∧ α < (Real.pi / 2)) 
  (h2 : - (Real.pi / 2) < β ∧ β < 0) 
  (hcos1 : Real.cos (α + (Real.pi / 4)) = 1 / 3) 
  (hcos2 : Real.cos ((β / 2) - (Real.pi / 4)) = Real.sqrt 3 / 3) : 
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_two_l181_18101


namespace train_speed_l181_18186

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l181_18186


namespace problem1_problem2_l181_18180

-- Problem 1
theorem problem1 (a : ℝ) : 2 * a + 3 * a - 4 * a = a :=
by sorry

-- Problem 2
theorem problem2 : 
  - (1 : ℝ) ^ 2022 + (27 / 4) * (- (1 / 3) - 1) / ((-3) ^ 2) + abs (-1) = -1 :=
by sorry

end problem1_problem2_l181_18180


namespace medium_bed_rows_l181_18161

theorem medium_bed_rows (large_top_beds : ℕ) (large_bed_rows : ℕ) (large_bed_seeds_per_row : ℕ) 
                         (medium_beds : ℕ) (medium_bed_seeds_per_row : ℕ) (total_seeds : ℕ) :
    large_top_beds = 2 ∧ large_bed_rows = 4 ∧ large_bed_seeds_per_row = 25 ∧
    medium_beds = 2 ∧ medium_bed_seeds_per_row = 20 ∧ total_seeds = 320 →
    ((total_seeds - (large_top_beds * large_bed_rows * large_bed_seeds_per_row)) / medium_bed_seeds_per_row) = 6 :=
by
  intro conditions
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := conditions
  sorry

end medium_bed_rows_l181_18161


namespace gcd_m_n_l181_18140

def m : ℕ := 555555555
def n : ℕ := 1111111111

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l181_18140


namespace average_height_is_64_l181_18137

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l181_18137


namespace find_dividend_l181_18155

-- Given conditions as definitions
def divisor : ℕ := 16
def quotient : ℕ := 9
def remainder : ℕ := 5

-- Lean 4 statement to be proven
theorem find_dividend : divisor * quotient + remainder = 149 := by
  sorry

end find_dividend_l181_18155


namespace technician_completion_percentage_l181_18107

noncomputable def percentage_completed (D : ℝ) : ℝ :=
  let total_distance := 2.20 * D
  let completed_distance := 1.12 * D
  (completed_distance / total_distance) * 100

theorem technician_completion_percentage (D : ℝ) (hD : D > 0) :
  percentage_completed D = 50.91 :=
by
  sorry

end technician_completion_percentage_l181_18107


namespace number_of_divisors_of_n_l181_18100

def n : ℕ := 2^3 * 3^4 * 5^3 * 7^2

theorem number_of_divisors_of_n : ∃ d : ℕ, d = 240 ∧ ∀ k : ℕ, k ∣ n ↔ ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 3 ∧ 0 ≤ d ∧ d ≤ 2 := 
sorry

end number_of_divisors_of_n_l181_18100


namespace pyramid_cross_section_distance_l181_18143

theorem pyramid_cross_section_distance
  (area1 area2 : ℝ) (distance : ℝ)
  (h1 : area1 = 100 * Real.sqrt 3) 
  (h2 : area2 = 225 * Real.sqrt 3) 
  (h3 : distance = 5) : 
  ∃ h : ℝ, h = 15 :=
by
  sorry

end pyramid_cross_section_distance_l181_18143


namespace min_distinct_lines_for_polyline_l181_18136

theorem min_distinct_lines_for_polyline (n : ℕ) (h_n : n = 31) : 
  ∃ (k : ℕ), 9 ≤ k ∧ k ≤ 31 ∧ 
  (∀ (s : Fin n → Fin 31), 
     ∀ i j, i ≠ j → s i ≠ s j) := 
sorry

end min_distinct_lines_for_polyline_l181_18136


namespace part1_part2_l181_18167

open Complex

def equation (a z : ℂ) : Prop := z^2 - (a + I) * z - (I + 2) = 0

theorem part1 (m : ℝ) (a : ℝ) : equation a m → a = 1 := by
  sorry

theorem part2 (a : ℝ) : ¬ ∃ n : ℝ, equation a (n * I) := by
  sorry

end part1_part2_l181_18167


namespace sufficient_not_necessary_a_equals_2_l181_18138

theorem sufficient_not_necessary_a_equals_2 {a : ℝ} :
  (∃ a : ℝ, (a = 2 ∧ 15 * a^2 = 60) → (15 * a^2 = 60) ∧ (15 * a^2 = 60 → a = 2)) → 
  (¬∀ a : ℝ, (15 * a^2 = 60) → a = 2) → 
  (a = 2 → 15 * a^2 = 60) ∧ ¬(15 * a^2 = 60 → a = 2) :=
by
  sorry

end sufficient_not_necessary_a_equals_2_l181_18138


namespace revenue_percentage_l181_18129

theorem revenue_percentage (R C : ℝ) (hR_pos : R > 0) (hC_pos : C > 0) :
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 62.5 := by
  sorry

end revenue_percentage_l181_18129


namespace number_of_non_representable_l181_18179

theorem number_of_non_representable :
  ∀ (a b : ℕ), Nat.gcd a b = 1 →
  (∃ n : ℕ, ¬ ∃ x y : ℕ, n = a * x + b * y) :=
sorry

end number_of_non_representable_l181_18179


namespace sphere_surface_area_l181_18160

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : V = 36 * π) 
  (h2 : V = (4 / 3) * π * r^3) 
  (h3 : A = 4 * π * r^2) 
  : A = 36 * π :=
by
  sorry

end sphere_surface_area_l181_18160


namespace min_ab_bound_l181_18130

theorem min_ab_bound (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n) 
                      (h : ∀ i j, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) :
  ∃ c > 0, min a b > c^n * n^(n/2) :=
sorry

end min_ab_bound_l181_18130


namespace range_of_a_l181_18127

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_x : ∀ x, x ≤ 0 → f x = 2 * x + x^2)
  (h_three_solutions : ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 2 * a^2 + a ∧ f x2 = 2 * a^2 + a ∧ f x3 = 2 * a^2 + a) :
  -1 < a ∧ a < 1/2 :=
sorry

end range_of_a_l181_18127


namespace find_x_y_l181_18121

theorem find_x_y (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x + y * Complex.I)^2 = (7 + 24 * Complex.I)) :
  x + y * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end find_x_y_l181_18121


namespace find_angle_BXY_l181_18131

noncomputable def angle_AXE (angle_CYX : ℝ) : ℝ := 3 * angle_CYX - 108

theorem find_angle_BXY
  (AB_parallel_CD : Prop)
  (h_parallel : ∀ (AXE CYX : ℝ), angle_AXE CYX = AXE)
  (x : ℝ) :
  (angle_AXE x = x) → x = 54 :=
by
  intro h₁
  unfold angle_AXE at h₁
  sorry

end find_angle_BXY_l181_18131


namespace sample_size_is_correct_l181_18112

-- Define the conditions
def total_students : ℕ := 40 * 50
def students_selected : ℕ := 150

-- Theorem: The sample size is 150 given that 150 students are selected
theorem sample_size_is_correct : students_selected = 150 := by
  sorry  -- Proof to be completed

end sample_size_is_correct_l181_18112


namespace skirt_price_l181_18147

theorem skirt_price (S : ℝ) 
  (h1 : 2 * 5 = 10) 
  (h2 : 1 * 4 = 4) 
  (h3 : 6 * (5 / 2) = 15) 
  (h4 : 10 + 4 + 15 + 4 * S = 53) 
  : S = 6 :=
sorry

end skirt_price_l181_18147


namespace remaining_cooking_time_l181_18116

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l181_18116


namespace inequality_and_equality_conditions_l181_18165

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ abc ≤ 1 ∧ ((a + b + c = 3) → (a = 1 ∧ b = 1 ∧ c = 1)) := 
by 
  sorry

end inequality_and_equality_conditions_l181_18165


namespace last_two_digits_x_pow_y_add_y_pow_x_l181_18192

noncomputable def proof_problem (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : ℕ :=
  (x^y + y^x) % 100

theorem last_two_digits_x_pow_y_add_y_pow_x {x y : ℕ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : 
  proof_problem x y h1 h2 h3 h4 = 74 :=
sorry

end last_two_digits_x_pow_y_add_y_pow_x_l181_18192


namespace jason_total_amount_l181_18114

def shorts_price : ℝ := 14.28
def jacket_price : ℝ := 4.74
def shoes_price : ℝ := 25.95
def socks_price : ℝ := 6.80
def tshirts_price : ℝ := 18.36
def hat_price : ℝ := 12.50
def swimsuit_price : ℝ := 22.95
def sunglasses_price : ℝ := 45.60
def wristbands_price : ℝ := 9.80

def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price - (price * discount)

def total_discounted_price : ℝ := 
  (discounted_price shorts_price discount1) + 
  (discounted_price jacket_price discount1) + 
  (discounted_price hat_price discount1) + 
  (discounted_price shoes_price discount2) + 
  (discounted_price socks_price discount2) + 
  (discounted_price tshirts_price discount2) + 
  (discounted_price swimsuit_price discount2) + 
  (discounted_price sunglasses_price discount2) + 
  (discounted_price wristbands_price discount2)

def total_with_tax : ℝ := total_discounted_price + (total_discounted_price * sales_tax_rate)

theorem jason_total_amount : total_with_tax = 153.07 := by
  sorry

end jason_total_amount_l181_18114


namespace convert_base_3_to_base_10_l181_18173

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l181_18173


namespace common_root_for_equations_l181_18176

theorem common_root_for_equations : 
  ∃ p x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 ∧ p = 3 ∧ x = 1 :=
by
  sorry

end common_root_for_equations_l181_18176


namespace cookie_price_ratio_l181_18175

theorem cookie_price_ratio (c b : ℝ) (h1 : 6 * c + 5 * b = 3 * (3 * c + 27 * b)) : c = (4 / 5) * b :=
sorry

end cookie_price_ratio_l181_18175


namespace solve_for_r_l181_18124

noncomputable def k (r : ℝ) : ℝ := 5 / (2 ^ r)

theorem solve_for_r (r : ℝ) :
  (5 = k r * 2 ^ r) ∧ (45 = k r * 8 ^ r) → r = (Real.log 9 / Real.log 2) / 2 :=
by
  intro h
  sorry

end solve_for_r_l181_18124


namespace kylie_daisies_l181_18191

theorem kylie_daisies :
  ∀ (initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies : ℕ),
    initial_daisies = 5 →
    sister_daisies = 9 →
    final_daisies = 7 →
    total_daisies = initial_daisies + sister_daisies →
    daisies_given_to_mother = total_daisies - final_daisies →
    daisies_given_to_mother * 2 = total_daisies :=
by
  intros initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies h1 h2 h3 h4 h5
  sorry

end kylie_daisies_l181_18191


namespace machine_production_percentage_difference_l181_18152

theorem machine_production_percentage_difference 
  (X_production_rate : ℕ := 3)
  (widgets_to_produce : ℕ := 1080)
  (difference_in_hours : ℕ := 60) :
  ((widgets_to_produce / (widgets_to_produce / X_production_rate - difference_in_hours) - 
   X_production_rate) / X_production_rate * 100) = 20 := by
  sorry

end machine_production_percentage_difference_l181_18152


namespace grayson_unanswered_l181_18109

noncomputable def unanswered_questions : ℕ :=
  let total_questions := 200
  let first_set_questions := 50
  let first_set_time := first_set_questions * 1 -- 1 minute per question
  let second_set_questions := 50
  let second_set_time := second_set_questions * (90 / 60) -- convert 90 seconds to minutes
  let third_set_questions := 25
  let third_set_time := third_set_questions * 2 -- 2 minutes per question
  let total_answered_time := first_set_time + second_set_time + third_set_time
  let total_time_available := 4 * 60 -- 4 hours in minutes 
  let unanswered := total_questions - (first_set_questions + second_set_questions + third_set_questions)
  unanswered

theorem grayson_unanswered : unanswered_questions = 75 := 
by 
  sorry

end grayson_unanswered_l181_18109


namespace unique_solution_m_n_l181_18119

theorem unique_solution_m_n (m n : ℕ) (h1 : m > 1) (h2 : (n - 1) % (m - 1) = 0) 
  (h3 : ¬ ∃ k : ℕ, n = m ^ k) :
  ∃! (a b c : ℕ), a + m * b = n ∧ a + b = m * c := 
sorry

end unique_solution_m_n_l181_18119


namespace symmetric_points_l181_18135

-- Let points P and Q be symmetric about the origin
variables (m n : ℤ)
axiom symmetry_condition : (m, 4) = (- (-2), -n)

theorem symmetric_points :
  m = 2 ∧ n = -4 := 
  by {
    sorry
  }

end symmetric_points_l181_18135


namespace triangle_angles_l181_18171

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180

theorem triangle_angles (x : ℝ) (hA : A = x) (hB : B = 2 * A) (hC : C + A + B = 180) :
  A = x ∧ B = 2 * x ∧ C = 180 - 3 * x := by
  -- proof goes here
  sorry

end triangle_angles_l181_18171


namespace molecular_weight_of_10_moles_of_Al2S3_l181_18151

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the molecular weight calculation for Al2S3
def molecular_weight_Al2S3 : ℝ :=
  (2 * atomic_weight_Al) + (3 * atomic_weight_S)

-- Define the molecular weight for 10 moles of Al2S3
def molecular_weight_10_moles_Al2S3 : ℝ :=
  10 * molecular_weight_Al2S3

-- The theorem to prove
theorem molecular_weight_of_10_moles_of_Al2S3 :
  molecular_weight_10_moles_Al2S3 = 1501.4 :=
by
  -- skip the proof
  sorry

end molecular_weight_of_10_moles_of_Al2S3_l181_18151


namespace find_number_l181_18120

theorem find_number (x : ℚ) (h : 1 + 1 / x = 5 / 2) : x = 2 / 3 :=
by
  sorry

end find_number_l181_18120
