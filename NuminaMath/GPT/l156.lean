import Mathlib

namespace least_positive_nine_n_square_twelve_n_cube_l156_15650

theorem least_positive_nine_n_square_twelve_n_cube :
  ∃ (n : ℕ), 0 < n ∧ (∃ (k1 k2 : ℕ), 9 * n = k1^2 ∧ 12 * n = k2^3) ∧ n = 144 :=
by
  sorry

end least_positive_nine_n_square_twelve_n_cube_l156_15650


namespace triangle_area_CO_B_l156_15644

-- Define the conditions as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def Q : Point := ⟨0, 15⟩

variable (p : ℝ)
def C : Point := ⟨0, p⟩
def B : Point := ⟨15, 0⟩

-- Prove the area of triangle COB is 15p / 2
theorem triangle_area_CO_B :
  p ≥ 0 → p ≤ 15 → 
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  area = (15 * p) / 2 := 
by
  intros hp0 hp15
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  have : area = (15 * p) / 2 := sorry
  exact this

end triangle_area_CO_B_l156_15644


namespace travel_time_proportion_l156_15637

theorem travel_time_proportion (D V : ℝ) (hV_pos : V > 0) :
  let Time1 := D / (16 * V)
  let Time2 := 3 * D / (4 * V)
  let TimeTotal := Time1 + Time2
  (Time1 / TimeTotal) = 1 / 13 :=
by
  sorry

end travel_time_proportion_l156_15637


namespace james_out_of_pocket_cost_l156_15686

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l156_15686


namespace caitlin_bracelets_l156_15642

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l156_15642


namespace problem1_problem2_l156_15625

-- Problem 1: Prove that (2a^2 b) * a b^2 / 4a^3 = 1/2 b^3
theorem problem1 (a b : ℝ) : (2 * a^2 * b) * (a * b^2) / (4 * a^3) = (1 / 2) * b^3 :=
  sorry

-- Problem 2: Prove that (2x + 5)(x - 3) = 2x^2 - x - 15
theorem problem2 (x : ℝ): (2 * x + 5) * (x - 3) = 2 * x^2 - x - 15 :=
  sorry

end problem1_problem2_l156_15625


namespace deceased_member_income_l156_15653

theorem deceased_member_income (A B C : ℝ) (h1 : (A + B + C) / 3 = 735) (h2 : (A + B) / 2 = 650) : 
  C = 905 :=
by
  sorry

end deceased_member_income_l156_15653


namespace seven_pow_fifty_one_mod_103_l156_15610

theorem seven_pow_fifty_one_mod_103 : (7^51 - 1) % 103 = 0 := 
by
  -- Fermat's Little Theorem: If p is a prime number and a is an integer not divisible by p,
  -- then a^(p-1) ≡ 1 ⧸ p.
  -- 103 is prime, so for 7 which is not divisible by 103, we have 7^102 ≡ 1 ⧸ 103.
sorry

end seven_pow_fifty_one_mod_103_l156_15610


namespace fixed_point_coordinates_l156_15638

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 4) ∧ ∀ x, P = (x, a^(x-1) + 3) :=
by
  use (1, 4)
  sorry

end fixed_point_coordinates_l156_15638


namespace equal_cubic_values_l156_15667

theorem equal_cubic_values (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 3) 
  (h3 : a * b * c + b * c * d + c * d * a + d * a * b = 1) :
  a * (1 - a)^3 = b * (1 - b)^3 ∧ 
  b * (1 - b)^3 = c * (1 - c)^3 ∧ 
  c * (1 - c)^3 = d * (1 - d)^3 :=
sorry

end equal_cubic_values_l156_15667


namespace smallest_x_for_perfect_cube_l156_15639

theorem smallest_x_for_perfect_cube :
  ∃ (x : ℕ) (h : x > 0), x = 36 ∧ (∃ (k : ℕ), 1152 * x = k ^ 3) := by
  sorry

end smallest_x_for_perfect_cube_l156_15639


namespace cubic_inequality_l156_15654

theorem cubic_inequality :
  {x : ℝ | x^3 - 12*x^2 + 47*x - 60 < 0} = {x : ℝ | 3 < x ∧ x < 5} :=
by
  sorry

end cubic_inequality_l156_15654


namespace sum_of_digits_of_smallest_number_l156_15691

noncomputable def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_of_smallest_number :
  (n : Nat) → (h1 : (Nat.ceil (n / 2) - Nat.ceil (n / 3) = 15)) → 
  sum_of_digits n = 9 :=
by
  sorry

end sum_of_digits_of_smallest_number_l156_15691


namespace billiard_angle_correct_l156_15617

-- Definitions for the problem conditions
def center_O : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (0.5, 0)
def radius : ℝ := 1

-- The angle to be proven
def strike_angle (α x : ℝ) := x = (90 - 2 * α)

-- Main theorem statement
theorem billiard_angle_correct :
  ∃ α x : ℝ, (strike_angle α x) ∧ x = 47 + (4 / 60) :=
sorry

end billiard_angle_correct_l156_15617


namespace find_f_2008_l156_15692

noncomputable def f (x : ℝ) : ℝ := Real.cos x

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
match n with
| 0     => f
| (n+1) => (deriv (f_n n))

theorem find_f_2008 (x : ℝ) : (f_n 2008) x = Real.cos x := by
  sorry

end find_f_2008_l156_15692


namespace pizza_ratio_l156_15623

/-- Define a function that represents the ratio calculation -/
def ratio (a b : ℕ) : ℕ × ℕ := (a / (Nat.gcd a b), b / (Nat.gcd a b))

/-- State the main problem to be proved -/
theorem pizza_ratio (total_slices friend_eats james_eats remaining_slices gcd : ℕ)
  (h1 : total_slices = 8)
  (h2 : friend_eats = 2)
  (h3 : james_eats = 3)
  (h4 : remaining_slices = total_slices - friend_eats)
  (h5 : gcd = Nat.gcd james_eats remaining_slices)
  (h6 : ratio james_eats remaining_slices = (1, 2)) :
  ratio james_eats remaining_slices = (1, 2) :=
by
  sorry

end pizza_ratio_l156_15623


namespace negation_of_universal_l156_15661

theorem negation_of_universal (h : ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) : ∃ x : ℝ, x^2 + 2 * x + 5 = 0 :=
sorry

end negation_of_universal_l156_15661


namespace width_of_foil_covered_prism_l156_15651

theorem width_of_foil_covered_prism (L W H : ℝ) 
    (hW1 : W = 2 * L)
    (hW2 : W = 2 * H)
    (hvol : L * W * H = 128) :
    W + 2 = 8 := 
sorry

end width_of_foil_covered_prism_l156_15651


namespace imaginary_roots_iff_l156_15632

theorem imaginary_roots_iff {k m : ℝ} (hk : k ≠ 0) : (exists (x : ℝ), k * x^2 + m * x + k = 0 ∧ ∃ (y : ℝ), y * 0 = 0 ∧ y ≠ 0) ↔ m ^ 2 < 4 * k ^ 2 :=
by
  sorry

end imaginary_roots_iff_l156_15632


namespace muffins_total_is_83_l156_15674

-- Define the given conditions.
def initial_muffins : Nat := 35
def additional_muffins : Nat := 48

-- Define the total number of muffins.
def total_muffins : Nat := initial_muffins + additional_muffins

-- Statement to prove.
theorem muffins_total_is_83 : total_muffins = 83 := by
  -- Proof is omitted.
  sorry

end muffins_total_is_83_l156_15674


namespace cordelia_bleaching_l156_15612

noncomputable def bleaching_time (B : ℝ) : Prop :=
  B + 4 * B + B / 3 = 10

theorem cordelia_bleaching : ∃ B : ℝ, bleaching_time B ∧ B = 1.875 :=
by {
  sorry
}

end cordelia_bleaching_l156_15612


namespace product_of_roots_l156_15683

theorem product_of_roots :
  ∀ a b c : ℚ, (a ≠ 0) → a = 24 → b = 60 → c = -600 → (c / a) = -25 :=
sorry

end product_of_roots_l156_15683


namespace sci_not_218000_l156_15626

theorem sci_not_218000 : 218000 = 2.18 * 10^5 :=
by
  sorry

end sci_not_218000_l156_15626


namespace range_of_a_for_f_ge_a_l156_15604

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a_for_f_ge_a :
  (∀ x : ℝ, (-1 ≤ x → f x a ≥ a)) ↔ (-3 ≤ a ∧ a ≤ 1) :=
  sorry

end range_of_a_for_f_ge_a_l156_15604


namespace height_of_tank_B_l156_15611

noncomputable def height_tank_A : ℝ := 5
noncomputable def circumference_tank_A : ℝ := 4
noncomputable def circumference_tank_B : ℝ := 10
noncomputable def capacity_ratio : ℝ := 0.10000000000000002

theorem height_of_tank_B {h_B : ℝ} 
  (h_tank_A : height_tank_A = 5)
  (c_tank_A : circumference_tank_A = 4)
  (c_tank_B : circumference_tank_B = 10)
  (capacity_percentage : capacity_ratio = 0.10000000000000002)
  (V_A : ℝ := π * (2 / π)^2 * height_tank_A)
  (V_B : ℝ := π * (5 / π)^2 * h_B)
  (capacity_relation : V_A = capacity_ratio * V_B) :
  h_B = 8 :=
sorry

end height_of_tank_B_l156_15611


namespace sum_b_n_l156_15631

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), (∀ n : ℕ, a (n + 1) = q * a n)

theorem sum_b_n (h_geo : is_geometric a) (h_a1 : a 1 = 3) (h_sum_a : ∑' n, a n = 9) (h_bn : ∀ n, b n = a (2 * n)) :
  ∑' n, b n = 18 / 5 :=
sorry

end sum_b_n_l156_15631


namespace parker_daily_earning_l156_15640

-- Definition of conditions
def total_earned : ℕ := 2646
def weeks_worked : ℕ := 6
def days_per_week : ℕ := 7
def total_days (weeks : ℕ) (days_in_week : ℕ) : ℕ := weeks * days_in_week

-- Proof statement
theorem parker_daily_earning (h : total_days weeks_worked days_per_week = 42) : (total_earned / 42) = 63 :=
by
  sorry

end parker_daily_earning_l156_15640


namespace fish_per_person_l156_15641

theorem fish_per_person (eyes_per_fish : ℕ) (fish_caught : ℕ) (total_eyes : ℕ) (dog_eyes : ℕ) (oomyapeck_eyes : ℕ) (n_people : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_eyes = fish_caught * eyes_per_fish →
  n_people = 3 →
  oomyapeck_eyes = 22 →
  dog_eyes = 2 →
  eyes_per_fish = 2 →
  fish_caught / n_people = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end fish_per_person_l156_15641


namespace cost_equivalence_l156_15696

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end cost_equivalence_l156_15696


namespace kanul_spent_on_raw_materials_l156_15697

theorem kanul_spent_on_raw_materials 
    (total_amount : ℝ)
    (spent_machinery : ℝ)
    (spent_cash_percent : ℝ)
    (spent_cash : ℝ)
    (amount_raw_materials : ℝ)
    (h_total : total_amount = 93750)
    (h_machinery : spent_machinery = 40000)
    (h_percent : spent_cash_percent = 20 / 100)
    (h_cash : spent_cash = spent_cash_percent * total_amount)
    (h_sum : total_amount = amount_raw_materials + spent_machinery + spent_cash) : 
    amount_raw_materials = 35000 :=
sorry

end kanul_spent_on_raw_materials_l156_15697


namespace solve_equation_l156_15628

theorem solve_equation : ∀ (x : ℝ), x ≠ 2 → -2 * x^2 = (4 * x + 2) / (x - 2) → x = 1 :=
by
  intros x hx h_eq
  sorry

end solve_equation_l156_15628


namespace basic_full_fare_l156_15676

theorem basic_full_fare 
  (F R : ℝ)
  (h1 : F + R = 216)
  (h2 : (F + R) + (0.5 * F + R) = 327) :
  F = 210 :=
by
  sorry

end basic_full_fare_l156_15676


namespace min_value_expr_l156_15647

theorem min_value_expr (x y : ℝ) : 
  ∃ min_val, min_val = 2 ∧ min_val ≤ (x + y)^2 + (x - 1/y)^2 :=
sorry

end min_value_expr_l156_15647


namespace find_f_neg_one_l156_15684

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd f)
(h_pos : ∀ x, 0 < x → f x = x^2 + 1/x) : f (-1) = -2 := 
sorry

end find_f_neg_one_l156_15684


namespace sally_combinations_l156_15605

theorem sally_combinations :
  let wall_colors := 4
  let flooring_types := 3
  wall_colors * flooring_types = 12 := by
  sorry

end sally_combinations_l156_15605


namespace value_of_m_plus_n_l156_15635

noncomputable def exponential_function (a x m n : ℝ) : ℝ :=
  a^(x - m) + n - 3

theorem value_of_m_plus_n (a x m n y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : exponential_function a 3 m n = 2) : m + n = 7 :=
by
  sorry

end value_of_m_plus_n_l156_15635


namespace tournament_participants_l156_15616

theorem tournament_participants (x : ℕ) (h1 : ∀ g b : ℕ, g = 2 * b)
  (h2 : ∀ p : ℕ, p = 3 * x) 
  (h3 : ∀ G B : ℕ, G + B = (3 * x * (3 * x - 1)) / 2)
  (h4 : ∀ G B : ℕ, G / B = 7 / 9) 
  (h5 : x = 11) :
  3 * x = 33 :=
by
  sorry

end tournament_participants_l156_15616


namespace find_a1_l156_15613

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then n * a 0 else a 0 * (1 - q ^ n) / (1 - q)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definitions from conditions
def S_3_eq_a2_plus_10a1 (a_1 a_2 S_3 : ℝ) : Prop :=
S_3 = a_2 + 10 * a_1

def a_5_eq_9 (a_5 : ℝ) : Prop :=
a_5 = 9

-- Main theorem statement
theorem find_a1 (h1 : S_3_eq_a2_plus_10a1 (a 1) (a 2) (sum_of_geometric_sequence a q 3))
                (h2 : a_5_eq_9 (a 5))
                (h3 : q ≠ 0 ∧ q ≠ 1) :
    a 1 = 1 / 9 :=
sorry

end find_a1_l156_15613


namespace derivative_at_0_l156_15666

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x * Real.sin x - 7 * x

theorem derivative_at_0 : deriv f 0 = -6 := 
by
  sorry

end derivative_at_0_l156_15666


namespace workers_complete_time_l156_15614

theorem workers_complete_time
  (A : ℝ) -- Total work
  (x1 x2 x3 : ℝ) -- Productivities of the workers
  (h1 : x3 = (x1 + x2) / 2)
  (h2 : 10 * x1 = 15 * x2) :
  (A / x1 = 50) ∧ (A / x2 = 75) ∧ (A / x3 = 60) :=
by
  sorry  -- Proof not required

end workers_complete_time_l156_15614


namespace linear_function_not_third_quadrant_l156_15681

theorem linear_function_not_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ¬ (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ y = k * x + 1) :=
sorry

end linear_function_not_third_quadrant_l156_15681


namespace min_squared_distance_l156_15688

open Real

theorem min_squared_distance : ∀ (x y : ℝ), (3 * x + y = 10) → (x^2 + y^2) ≥ 10 :=
by
  intros x y hxy
  -- Insert the necessary steps or key elements here
  sorry

end min_squared_distance_l156_15688


namespace geometric_arithmetic_series_difference_l156_15618

theorem geometric_arithmetic_series_difference :
  let a := 1
  let r := 1 / 2
  let S := a / (1 - r)
  let T := 1 + 2 + 3
  S - T = -4 :=
by
  sorry

end geometric_arithmetic_series_difference_l156_15618


namespace pencils_added_by_Nancy_l156_15678

def original_pencils : ℕ := 27
def total_pencils : ℕ := 72

theorem pencils_added_by_Nancy : ∃ x : ℕ, x = total_pencils - original_pencils := by
  sorry

end pencils_added_by_Nancy_l156_15678


namespace find_initial_money_l156_15664

def initial_money (s1 s2 s3 : ℝ) : ℝ :=
  let after_store_1 := s1 - (0.4 * s1 + 4)
  let after_store_2 := after_store_1 - (0.5 * after_store_1 + 5)
  let after_store_3 := after_store_2 - (0.6 * after_store_2 + 6)
  after_store_3

theorem find_initial_money (s1 s2 s3 : ℝ) (hs3 : initial_money s1 s2 s3 = 2) : s1 = 90 :=
by
  -- Placeholder for the actual proof
  sorry

end find_initial_money_l156_15664


namespace Louie_monthly_payment_l156_15629

noncomputable def monthly_payment (P : ℕ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let A := P * (1 + r) ^ n
  A / 3

theorem Louie_monthly_payment : 
  monthly_payment 1000 0.10 3 (3 / 12) = 444 := 
by
  -- computation and rounding
  sorry

end Louie_monthly_payment_l156_15629


namespace range_m_graph_in_quadrants_l156_15687

theorem range_m_graph_in_quadrants (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (m + 2) / x > 0) ∧ (x < 0 → (m + 2) / x < 0))) ↔ m > -2 :=
by 
  sorry

end range_m_graph_in_quadrants_l156_15687


namespace translate_B_to_origin_l156_15672

structure Point where
  x : ℝ
  y : ℝ

def translate_right (p : Point) (d : ℕ) : Point := 
  { x := p.x + d, y := p.y }

theorem translate_B_to_origin :
  ∀ (A B : Point) (d : ℕ),
  A = { x := -4, y := 0 } →
  B = { x := 0, y := 2 } →
  (translate_right A d).x = 0 →
  translate_right B d = { x := 4, y := 2 } :=
by
  intros A B d hA hB hA'
  sorry

end translate_B_to_origin_l156_15672


namespace six_digit_number_divisible_by_504_l156_15652

theorem six_digit_number_divisible_by_504 : 
  ∃ a b c : ℕ, (523 * 1000 + 100 * a + 10 * b + c) % 504 = 0 := by 
sorry

end six_digit_number_divisible_by_504_l156_15652


namespace range_of_x_when_a_equals_1_range_of_a_l156_15656

variable {a x : ℝ}

-- Definitions for conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part (1): Prove the range of x when a = 1 and p ∨ q is true.
theorem range_of_x_when_a_equals_1 (h : a = 1) (h1 : p 1 x ∨ q x) : 1 < x ∧ x < 3 :=
by sorry

-- Part (2): Prove the range of a when p is a necessary but not sufficient condition for q.
theorem range_of_a (h2 : ∀ x, q x → p a x) (h3 : ¬ ∀ x, p a x → q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_x_when_a_equals_1_range_of_a_l156_15656


namespace length_of_CB_l156_15660

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end length_of_CB_l156_15660


namespace common_ratio_l156_15693

-- Definitions for the geometric sequence
variables {a_n : ℕ → ℝ} {S_n q : ℝ}

-- Conditions provided in the problem
def condition1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  S_n 3 = a_n 1 + a_n 2 + a_n 3

def condition2 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2 + a_n 3) = a_n 4 - 2

def condition3 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2) = a_n 3 - 2

-- The theorem we want to prove
theorem common_ratio (a_n : ℕ → ℝ) (q : ℝ) :
  condition2 a_n S_n ∧ condition3 a_n S_n → q = 4 :=
by
  sorry

end common_ratio_l156_15693


namespace max_product_of_roots_l156_15643

noncomputable def max_prod_roots_m : ℝ :=
  let m := 4.5
  m

theorem max_product_of_roots (m : ℕ) (h : 36 - 8 * m ≥ 0) : m = max_prod_roots_m :=
  sorry

end max_product_of_roots_l156_15643


namespace range_of_a_l156_15665

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0 ↔ a ∈ Set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l156_15665


namespace n_minus_m_l156_15682

variable (m n : ℕ)

def is_congruent_to_5_mod_13 (x : ℕ) : Prop := x % 13 = 5
def is_smallest_three_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 100 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 100 → x ≤ y

def is_smallest_four_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 1000 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 1000 → x ≤ y

theorem n_minus_m
  (h₁ : is_smallest_three_digit_integer_congruent_to_5_mod_13 m)
  (h₂ : is_smallest_four_digit_integer_congruent_to_5_mod_13 n) :
  n - m = 897 := sorry

end n_minus_m_l156_15682


namespace percent_Asian_in_West_l156_15646

noncomputable def NE_Asian := 2
noncomputable def MW_Asian := 2
noncomputable def South_Asian := 2
noncomputable def West_Asian := 6

noncomputable def total_Asian := NE_Asian + MW_Asian + South_Asian + West_Asian

theorem percent_Asian_in_West (h1 : total_Asian = 12) : (West_Asian / total_Asian) * 100 = 50 := 
by sorry

end percent_Asian_in_West_l156_15646


namespace values_of_a_l156_15698

noncomputable def quadratic_eq (a x : ℝ) : ℝ :=
(a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_eq a x = 0 → x ≥ 0) ↔ (a = 3 ∨ (-1 ≤ a ∧ a ≤ 1)) :=
sorry

end values_of_a_l156_15698


namespace potion_combinations_l156_15679

-- Definitions of conditions
def roots : Nat := 3
def minerals : Nat := 5
def incompatible_combinations : Nat := 2

-- Statement of the problem
theorem potion_combinations : (roots * minerals) - incompatible_combinations = 13 := by
  sorry

end potion_combinations_l156_15679


namespace find_X_l156_15620

theorem find_X (X : ℝ) 
  (h : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002) : 
  X = 1.25 := 
sorry

end find_X_l156_15620


namespace evaluate_expression_at_three_l156_15695

theorem evaluate_expression_at_three : 
  (3^2 + 3 * (3^6) = 2196) :=
by
  sorry -- This is where the proof would go

end evaluate_expression_at_three_l156_15695


namespace william_won_more_rounds_than_harry_l156_15677

def rounds_played : ℕ := 15
def william_won_rounds : ℕ := 10
def harry_won_rounds : ℕ := rounds_played - william_won_rounds
def william_won_more_rounds := william_won_rounds > harry_won_rounds

theorem william_won_more_rounds_than_harry : william_won_rounds - harry_won_rounds = 5 := 
by sorry

end william_won_more_rounds_than_harry_l156_15677


namespace no_base_131_cubed_l156_15622

open Nat

theorem no_base_131_cubed (n : ℕ) (k : ℕ) : 
  (4 ≤ n ∧ n ≤ 12) ∧ (1 * n^2 + 3 * n + 1 = k^3) → False := by
  sorry

end no_base_131_cubed_l156_15622


namespace exist_ai_for_xij_l156_15627

theorem exist_ai_for_xij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j :=
by
  sorry

end exist_ai_for_xij_l156_15627


namespace D_coordinates_l156_15662

namespace Parallelogram

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 2 }
def C : Point := { x := 3, y := 1 }

theorem D_coordinates :
  ∃ D : Point, D = { x := 2, y := -1 } ∧ ∀ A B C D : Point, 
    (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) := by
  sorry

end Parallelogram

end D_coordinates_l156_15662


namespace trains_meeting_time_l156_15675

noncomputable def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

noncomputable def time_to_meet (L1 L2 D S1 S2 : ℕ) : ℕ := 
  let S1_mps := kmph_to_mps S1
  let S2_mps := kmph_to_mps S2
  let relative_speed := S1_mps + S2_mps
  let total_distance := L1 + L2 + D
  total_distance / relative_speed

theorem trains_meeting_time : time_to_meet 210 120 160 74 92 = 10620 / 1000 :=
by
  sorry

end trains_meeting_time_l156_15675


namespace number_of_weavers_l156_15685

theorem number_of_weavers (W : ℕ) 
  (h1 : ∀ t : ℕ, t = 4 → 4 = W * (1 * t)) 
  (h2 : ∀ t : ℕ, t = 16 → 64 = 16 * (1 / (W:ℝ) * t)) : 
  W = 4 := 
by {
  sorry
}

end number_of_weavers_l156_15685


namespace minimize_expression_l156_15619

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end minimize_expression_l156_15619


namespace hyperbola_asymptote_l156_15657

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x, - y^2 = - x^2 / a^2 + 1) ∧ 
  (∀ x y, y + 2 * x = 0) → 
  a = 2 :=
by
  sorry

end hyperbola_asymptote_l156_15657


namespace compare_trig_values_l156_15669

noncomputable def a : ℝ := Real.tan (-7 * Real.pi / 6)
noncomputable def b : ℝ := Real.cos (23 * Real.pi / 4)
noncomputable def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem compare_trig_values : c < a ∧ a < b := sorry

end compare_trig_values_l156_15669


namespace base6_arithmetic_l156_15655

def base6_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  let n4 := n3 / 10
  let d4 := n4 % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0

def base10_to_base6 (n : ℕ) : ℕ :=
  let b4 := n / 6^4
  let r4 := n % 6^4
  let b3 := r4 / 6^3
  let r3 := r4 % 6^3
  let b2 := r3 / 6^2
  let r2 := r3 % 6^2
  let b1 := r2 / 6^1
  let b0 := r2 % 6^1
  b4 * 10000 + b3 * 1000 + b2 * 100 + b1 * 10 + b0

theorem base6_arithmetic : 
  base10_to_base6 ((base6_to_base10 45321 - base6_to_base10 23454) + base6_to_base10 14553) = 45550 :=
by
  sorry

end base6_arithmetic_l156_15655


namespace inequality_sum_squares_products_l156_15633

theorem inequality_sum_squares_products {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_sum_squares_products_l156_15633


namespace common_ratio_of_geometric_series_l156_15615

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l156_15615


namespace kathleen_allowance_l156_15648

theorem kathleen_allowance (x : ℝ) (h1 : Kathleen_middleschool_allowance = x + 2)
(h2 : Kathleen_senior_allowance = 5 + 2 * (x + 2))
(h3 : Kathleen_senior_allowance = 2.5 * Kathleen_middleschool_allowance) :
x = 8 :=
by sorry

end kathleen_allowance_l156_15648


namespace find_range_f_l156_15607

noncomputable def greatestIntegerLessEqual (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def f (x y : ℝ) : ℝ :=
  (x + y) / (greatestIntegerLessEqual x * greatestIntegerLessEqual y + greatestIntegerLessEqual x + greatestIntegerLessEqual y + 1)

theorem find_range_f (x y : ℝ) (h1: 0 < x) (h2: 0 < y) (h3: x * y = 1) : 
  ∃ r : ℝ, r = f x y := 
by
  sorry

end find_range_f_l156_15607


namespace shelves_for_coloring_books_l156_15680

theorem shelves_for_coloring_books (initial_stock sold donated per_shelf remaining total_used needed_shelves : ℕ) 
    (h_initial : initial_stock = 150)
    (h_sold : sold = 55)
    (h_donated : donated = 30)
    (h_per_shelf : per_shelf = 12)
    (h_total_used : total_used = sold + donated)
    (h_remaining : remaining = initial_stock - total_used)
    (h_needed_shelves : (remaining + per_shelf - 1) / per_shelf = needed_shelves) :
    needed_shelves = 6 :=
by
  sorry

end shelves_for_coloring_books_l156_15680


namespace stamp_blocks_inequalities_l156_15634

noncomputable def b (n : ℕ) : ℕ := sorry

theorem stamp_blocks_inequalities (n : ℕ) (m : ℕ) (hn : 0 < n) :
  ∃ c d : ℝ, c = 2 / 7 ∧ d = (4 * m^2 + 4 * m + 40) / 5 ∧
    (1 / 7 : ℝ) * n^2 - c * n ≤ b n ∧ 
    b n ≤ (1 / 5 : ℝ) * n^2 + d * n := 
  sorry

end stamp_blocks_inequalities_l156_15634


namespace coprime_divisible_l156_15658

theorem coprime_divisible (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a ∣ b * c) : a ∣ c :=
by
  sorry

end coprime_divisible_l156_15658


namespace retail_price_of_washing_machine_l156_15668

variable (a : ℝ)

theorem retail_price_of_washing_machine :
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price 
  retail_price = 1.04 * a :=
by
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price
  sorry -- Proof skipped

end retail_price_of_washing_machine_l156_15668


namespace prob_at_least_one_head_is_7_over_8_l156_15659

-- Define the event and probability calculation
def probability_of_tails_all_three_tosses : ℚ :=
  (1 / 2) ^ 3

def probability_of_at_least_one_head : ℚ :=
  1 - probability_of_tails_all_three_tosses

-- Prove the probability of at least one head is 7/8
theorem prob_at_least_one_head_is_7_over_8 : probability_of_at_least_one_head = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_is_7_over_8_l156_15659


namespace fraction_simplification_l156_15630

theorem fraction_simplification : 
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1 / 3 :=
by
  sorry

end fraction_simplification_l156_15630


namespace least_n_ge_100_divides_sum_of_powers_l156_15624

theorem least_n_ge_100_divides_sum_of_powers (n : ℕ) (h₁ : n ≥ 100) :
    77 ∣ (Finset.sum (Finset.range (n + 1)) (λ k => 2^k) - 1) ↔ n = 119 :=
by
  sorry

end least_n_ge_100_divides_sum_of_powers_l156_15624


namespace speed_of_stream_l156_15670

theorem speed_of_stream
  (v : ℝ)
  (h1 : ∀ t : ℝ, t = 7)
  (h2 : ∀ d : ℝ, d = 72)
  (h3 : ∀ s : ℝ, s = 21)
  : (72 / (21 - v) + 72 / (21 + v) = 7) → v = 3 :=
by
  intro h
  sorry

end speed_of_stream_l156_15670


namespace find_a_b_and_range_of_c_l156_15673

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem find_a_b_and_range_of_c (c : ℝ) (h1 : ∀ x, 3 * x^2 - 2 * 3 * x - 9 = 0 → x = -1 ∨ x = 3)
    (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 6 → f x 3 (-9) c < c^2 + 4 * c) : 
    (a = 3 ∧ b = -9) ∧ (c > 6 ∨ c < -9) := by
  sorry

end find_a_b_and_range_of_c_l156_15673


namespace team_A_champion_probability_l156_15636

/-- Teams A and B are playing a volleyball match.
Team A needs to win one more game to become the champion, while Team B needs to win two more games to become the champion.
The probability of each team winning each game is 0.5. -/
theorem team_A_champion_probability :
  let p_win := (0.5 : ℝ)
  let prob_A_champion := 1 - p_win * p_win
  prob_A_champion = 0.75 := by
  sorry

end team_A_champion_probability_l156_15636


namespace field_fence_length_l156_15694

theorem field_fence_length (L : ℝ) (A : ℝ) (W : ℝ) (fencing : ℝ) (hL : L = 20) (hA : A = 210) (hW : A = L * W) : 
  fencing = 2 * W + L → fencing = 41 :=
by
  rw [hL, hA] at hW
  sorry

end field_fence_length_l156_15694


namespace most_cost_effective_years_l156_15608

noncomputable def total_cost (x : ℕ) : ℝ := 100000 + 15000 * x + 1000 + 2000 * ((x * (x - 1)) / 2)

noncomputable def average_annual_cost (x : ℕ) : ℝ := total_cost x / x

theorem most_cost_effective_years : ∃ (x : ℕ), x = 10 ∧
  (∀ y : ℕ, y ≠ 10 → average_annual_cost x ≤ average_annual_cost y) :=
by
  sorry

end most_cost_effective_years_l156_15608


namespace donna_pizza_slices_left_l156_15689

def total_slices_initial : ℕ := 12
def slices_eaten_lunch (slices : ℕ) : ℕ := slices / 2
def slices_remaining_after_lunch (slices : ℕ) : ℕ := slices - slices_eaten_lunch slices
def slices_eaten_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices / 3
def slices_remaining_after_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices - slices_eaten_dinner slices
def slices_shared_friend (slices : ℕ) : ℕ := slices_remaining_after_dinner slices / 4
def slices_remaining_final (slices : ℕ) : ℕ := slices_remaining_after_dinner slices - slices_shared_friend slices

theorem donna_pizza_slices_left : slices_remaining_final total_slices_initial = 3 :=
sorry

end donna_pizza_slices_left_l156_15689


namespace smallest_four_digit_number_l156_15663

theorem smallest_four_digit_number : 
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∃ (AB CD : ℕ), 
      n = 1000 * (AB / 10) + 100 * (AB % 10) + CD ∧
      ((AB / 10) * 10 + (AB % 10) + 2) * CD = 100 ∧ 
      n / CD = ((AB / 10) * 10 + (AB % 10) + 1)^2) ∧
    n = 1805 :=
by
  sorry

end smallest_four_digit_number_l156_15663


namespace proof1_proof2_monotonically_increasing_interval_l156_15609

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 0.5 * Real.cos (2 * x)

theorem proof1 : ∀ x : ℝ, f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

theorem proof2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → -0.5 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
∃ lb ub : ℝ, lb = Real.pi / 6 + k * Real.pi ∧ ub = 2 * Real.pi / 3 + k * Real.pi ∧ ∀ x : ℝ, lb ≤ x ∧ x ≤ ub → f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

end proof1_proof2_monotonically_increasing_interval_l156_15609


namespace geometric_series_properties_l156_15645

noncomputable def first_term := (7 : ℚ) / 8
noncomputable def common_ratio := (-1 : ℚ) / 2

theorem geometric_series_properties : 
  common_ratio = -1 / 2 ∧ 
  (first_term * (1 - common_ratio^4) / (1 - common_ratio)) = 35 / 64 := 
by 
  sorry

end geometric_series_properties_l156_15645


namespace solve_for_x_l156_15621

theorem solve_for_x (x : ℝ) : 3^(3 * x) = Real.sqrt 81 -> x = 2 / 3 :=
by
  sorry

end solve_for_x_l156_15621


namespace arithmetic_seq_a5_value_l156_15601

theorem arithmetic_seq_a5_value (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := 
sorry

end arithmetic_seq_a5_value_l156_15601


namespace arithmetic_equation_false_l156_15690

theorem arithmetic_equation_false :
  4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 := by
  sorry

end arithmetic_equation_false_l156_15690


namespace eggs_ordered_l156_15699

theorem eggs_ordered (E : ℕ) (h1 : E > 0) (h_crepes : E * 1 / 4 = E / 4)
                     (h_cupcakes : 2 / 3 * (3 / 4 * E) = 1 / 2 * E)
                     (h_left : (3 / 4 * E - 2 / 3 * (3 / 4 * E)) = 9) :
  E = 18 := by
  sorry

end eggs_ordered_l156_15699


namespace theta_in_fourth_quadrant_l156_15603

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan (θ + Real.pi / 4) = 1 / 3) : 
  (θ > 3 * Real.pi / 2) ∧ (θ < 2 * Real.pi) :=
sorry

end theta_in_fourth_quadrant_l156_15603


namespace ants_need_more_hours_l156_15671

theorem ants_need_more_hours (initial_sugar : ℕ) (removal_rate : ℕ) (hours_spent : ℕ) : 
  initial_sugar = 24 ∧ removal_rate = 4 ∧ hours_spent = 3 → 
  (initial_sugar - removal_rate * hours_spent) / removal_rate = 3 :=
by
  intro h
  sorry

end ants_need_more_hours_l156_15671


namespace sin_double_angle_of_tan_l156_15602

theorem sin_double_angle_of_tan (α : ℝ) (hα1 : Real.tan α = 2) (hα2 : 0 < α ∧ α < Real.pi / 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tan_l156_15602


namespace inequality_sqrt_sum_ge_2_l156_15649
open Real

theorem inequality_sqrt_sum_ge_2 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  sqrt (a^3 / (1 + b * c)) + sqrt (b^3 / (1 + a * c)) + sqrt (c^3 / (1 + a * b)) ≥ 2 :=
by
  sorry

end inequality_sqrt_sum_ge_2_l156_15649


namespace remainder_of_55_pow_55_plus_15_mod_8_l156_15600

theorem remainder_of_55_pow_55_plus_15_mod_8 :
  (55^55 + 15) % 8 = 6 := by
  -- This statement does not include any solution steps.
  sorry

end remainder_of_55_pow_55_plus_15_mod_8_l156_15600


namespace other_acute_angle_is_60_l156_15606

theorem other_acute_angle_is_60 (a b c : ℝ) (h_triangle : a + b + c = 180) (h_right : c = 90) (h_acute : a = 30) : b = 60 :=
by 
  -- inserting proof later
  sorry

end other_acute_angle_is_60_l156_15606
