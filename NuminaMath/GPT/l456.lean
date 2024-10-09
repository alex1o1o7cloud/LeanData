import Mathlib

namespace Elizabeth_lost_bottles_l456_45676

theorem Elizabeth_lost_bottles :
  ∃ (L : ℕ), (10 - L - 1) * 3 = 21 ∧ L = 2 := by
  sorry

end Elizabeth_lost_bottles_l456_45676


namespace sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l456_45623

-- Definition and proof of sqrt(9) = 3
theorem sqrt_of_9 : Real.sqrt 9 = 3 := by
  sorry

-- Definition and proof of -sqrt(0.49) = -0.7
theorem neg_sqrt_of_0_49 : -Real.sqrt 0.49 = -0.7 := by
  sorry

-- Definition and proof of ±sqrt(64/81) = ±(8/9)
theorem pm_sqrt_of_64_div_81 : (Real.sqrt (64 / 81) = 8 / 9) ∧ (Real.sqrt (64 / 81) = -8 / 9) := by
  sorry

end sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l456_45623


namespace system_solution_correct_l456_45699

theorem system_solution_correct (b : ℝ) : (∃ x y : ℝ, (y = 3 * x - 5) ∧ (y = 2 * x + b) ∧ (x = 1) ∧ (y = -2)) ↔ b = -4 :=
by
  sorry

end system_solution_correct_l456_45699


namespace solve_inequality_l456_45637

theorem solve_inequality :
  { x : ℝ | x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 ∧ 
    (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) } = 
  { x : ℝ | (x < -8) ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ (x > 8) } := sorry

end solve_inequality_l456_45637


namespace external_tangent_twice_internal_tangent_l456_45651

noncomputable def distance_between_centers (r R : ℝ) : ℝ :=
  Real.sqrt (R^2 + r^2 + (10/3) * R * r)

theorem external_tangent_twice_internal_tangent 
  (r R O₁O₂ AB CD : ℝ)
  (h₁ : AB = 2 * CD)
  (h₂ : AB^2 = O₁O₂^2 - (R - r)^2)
  (h₃ : CD^2 = O₁O₂^2 - (R + r)^2) :
  O₁O₂ = distance_between_centers r R :=
by
  sorry

end external_tangent_twice_internal_tangent_l456_45651


namespace junk_items_count_l456_45649

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count_l456_45649


namespace problem1_problem2_l456_45636

-- Problem 1
theorem problem1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ≠ 1) : (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) := sorry

end problem1_problem2_l456_45636


namespace symmetric_circle_eq_l456_45602

theorem symmetric_circle_eq (x y : ℝ) :
  (x + 1)^2 + (y - 1)^2 = 1 → x - y = 1 → (x - 2)^2 + (y + 2)^2 = 1 :=
by
  sorry

end symmetric_circle_eq_l456_45602


namespace problem_solution_l456_45617

variable (f : ℝ → ℝ)

-- Let f be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f(x) = f(4 - x) for all x in ℝ
def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (4 - x)

-- f is increasing on [0, 2]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem problem_solution :
  is_odd_function f →
  satisfies_symmetry f →
  is_increasing_on_interval f 0 2 →
  f 6 < f 4 ∧ f 4 < f 1 :=
by
  intros
  sorry

end problem_solution_l456_45617


namespace non_sophomores_is_75_percent_l456_45697

def students_not_sophomores_percentage (total_students : ℕ) 
                                       (percent_juniors : ℚ)
                                       (num_seniors : ℕ)
                                       (freshmen_more_than_sophomores : ℕ) : ℚ :=
  let num_juniors := total_students * percent_juniors 
  let s := (total_students - num_juniors - num_seniors - freshmen_more_than_sophomores) / 2
  let f := s + freshmen_more_than_sophomores
  let non_sophomores := total_students - s
  (non_sophomores / total_students) * 100

theorem non_sophomores_is_75_percent : students_not_sophomores_percentage 800 0.28 160 16 = 75 := by
  sorry

end non_sophomores_is_75_percent_l456_45697


namespace intersection_A_B_l456_45695

-- Conditions
def A : Set ℝ := {1, 2, 0.5}
def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x^2}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1} :=
sorry

end intersection_A_B_l456_45695


namespace emily_has_7_times_more_oranges_than_sandra_l456_45691

theorem emily_has_7_times_more_oranges_than_sandra
  (B S E : ℕ)
  (h1 : S = 3 * B)
  (h2 : B = 12)
  (h3 : E = 252) :
  ∃ k : ℕ, E = k * S ∧ k = 7 :=
by
  use 7
  sorry

end emily_has_7_times_more_oranges_than_sandra_l456_45691


namespace area_of_sector_l456_45648

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l456_45648


namespace problem_A_problem_C_problem_D_problem_E_l456_45692

variable {a b c : ℝ}
variable (ha : a < 0) (hab : a < b) (hb : b < 0) (hc : 0 < c)

theorem problem_A (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * b > a * c :=
by sorry

theorem problem_C (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * c < b * c :=
by sorry

theorem problem_D (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a + c < b + c :=
by sorry

theorem problem_E (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : c / a > 1 :=
by sorry

end problem_A_problem_C_problem_D_problem_E_l456_45692


namespace cylindrical_to_rectangular_point_l456_45618

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_point :
  cylindrical_to_rectangular (Real.sqrt 2) (Real.pi / 4) 1 = (1, 1, 1) :=
by
  sorry

end cylindrical_to_rectangular_point_l456_45618


namespace intersection_point_l456_45604

noncomputable def f (x : ℝ) := (x^2 - 8 * x + 7) / (2 * x - 6)

noncomputable def g (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) :
  (∀ x, 2 * x - 6 = 0 <-> x ≠ 3) →
  ∃ (k : ℝ), (g a b c x = -2 * x - 4 + k / (x - 3)) →
  (f x = g a b c x) ∧ x ≠ -3 → x = 1 ∧ f 1 = 0 :=
by
  intros
  sorry

end intersection_point_l456_45604


namespace llesis_more_rice_l456_45686

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end llesis_more_rice_l456_45686


namespace prop1_prop2_prop3_l456_45672

variables (a b c d : ℝ)

-- Proposition 1: ab > 0 ∧ bc - ad > 0 → (c/a - d/b > 0)
theorem prop1 (h1 : a * b > 0) (h2 : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

-- Proposition 2: ab > 0 ∧ (c/a - d/b > 0) → bc - ad > 0
theorem prop2 (h1 : a * b > 0) (h2 : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

-- Proposition 3: (bc - ad > 0) ∧ (c/a - d/b > 0) → ab > 0
theorem prop3 (h1 : b * c - a * d > 0) (h2 : c / a - d / b > 0) : a * b > 0 :=
sorry

end prop1_prop2_prop3_l456_45672


namespace no_positive_rational_solutions_l456_45657

theorem no_positive_rational_solutions (n : ℕ) (h_pos_n : 0 < n) : 
  ¬ ∃ (x y : ℚ) (h_x_pos : 0 < x) (h_y_pos : 0 < y), x + y + (1/x) + (1/y) = 3 * n :=
by
  sorry

end no_positive_rational_solutions_l456_45657


namespace lines_in_n_by_n_grid_l456_45625

def num_horizontal_lines (n : ℕ) : ℕ := n + 1
def num_vertical_lines (n : ℕ) : ℕ := n + 1
def total_lines (n : ℕ) : ℕ := num_horizontal_lines n + num_vertical_lines n

theorem lines_in_n_by_n_grid (n : ℕ) :
  total_lines n = 2 * (n + 1) := by
  sorry

end lines_in_n_by_n_grid_l456_45625


namespace Jessica_cut_roses_l456_45621

variable (initial_roses final_roses added_roses : Nat)

theorem Jessica_cut_roses
  (h_initial : initial_roses = 10)
  (h_final : final_roses = 18)
  (h_added : final_roses = initial_roses + added_roses) :
  added_roses = 8 := by
  sorry

end Jessica_cut_roses_l456_45621


namespace at_least_5_limit_ups_needed_l456_45622

-- Let's denote the necessary conditions in Lean
variable (a : ℝ) -- the buying price of stock A

-- Initial price after 4 consecutive limit downs
def price_after_limit_downs (a : ℝ) : ℝ := a * (1 - 0.1) ^ 4

-- Condition of no loss after certain limit ups
def no_loss_after_limit_ups (a : ℝ) (x : ℕ) : Prop := 
  price_after_limit_downs a * (1 + 0.1)^x ≥ a
  
theorem at_least_5_limit_ups_needed (a : ℝ) : ∃ x, no_loss_after_limit_ups a x ∧ x ≥ 5 :=
by
  -- We are required to find such x and prove the condition, which has been shown in the mathematical solution
  sorry

end at_least_5_limit_ups_needed_l456_45622


namespace molecular_weight_of_compound_l456_45619

theorem molecular_weight_of_compound :
  let Cu_atoms := 2
  let C_atoms := 3
  let O_atoms := 5
  let N_atoms := 1
  let atomic_weight_Cu := 63.546
  let atomic_weight_C := 12.011
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  Cu_atoms * atomic_weight_Cu +
  C_atoms * atomic_weight_C +
  O_atoms * atomic_weight_O +
  N_atoms * atomic_weight_N = 257.127 :=
by
  sorry

end molecular_weight_of_compound_l456_45619


namespace person_savings_l456_45644

theorem person_savings (income expenditure savings : ℝ) 
  (h1 : income = 18000)
  (h2 : income / expenditure = 5 / 4)
  (h3 : savings = income - expenditure) : 
  savings = 3600 := 
sorry

end person_savings_l456_45644


namespace train_length_is_correct_l456_45667

noncomputable def speed_km_per_hr := 60
noncomputable def time_seconds := 15
noncomputable def speed_m_per_s : ℝ := (60 * 1000) / 3600
noncomputable def expected_length : ℝ := 250.05

theorem train_length_is_correct : (speed_m_per_s * time_seconds) = expected_length := by
  sorry

end train_length_is_correct_l456_45667


namespace sequence_b_l456_45642

theorem sequence_b (b : ℕ → ℕ) 
  (h1 : b 1 = 2) 
  (h2 : ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) : 
  b 10 = 110 :=
sorry

end sequence_b_l456_45642


namespace Annette_Caitlin_total_weight_l456_45650

variable (A C S : ℕ)

-- Conditions
axiom cond1 : C + S = 87
axiom cond2 : A = S + 8

-- Theorem
theorem Annette_Caitlin_total_weight : A + C = 95 := by
  sorry

end Annette_Caitlin_total_weight_l456_45650


namespace coeffs_divisible_by_5_l456_45641

theorem coeffs_divisible_by_5
  (a b c d : ℤ)
  (h1 : a + b + c + d ≡ 0 [ZMOD 5])
  (h2 : -a + b - c + d ≡ 0 [ZMOD 5])
  (h3 : 8 * a + 4 * b + 2 * c + d ≡ 0 [ZMOD 5])
  (h4 : d ≡ 0 [ZMOD 5]) :
  a ≡ 0 [ZMOD 5] ∧ b ≡ 0 [ZMOD 5] ∧ c ≡ 0 [ZMOD 5] ∧ d ≡ 0 [ZMOD 5] :=
sorry

end coeffs_divisible_by_5_l456_45641


namespace coefficient_of_determination_indicates_better_fit_l456_45680

theorem coefficient_of_determination_indicates_better_fit (R_squared : ℝ) (h1 : 0 ≤ R_squared) (h2 : R_squared ≤ 1) :
  R_squared = 1 → better_fitting_effect_of_regression_model :=
by
  sorry

end coefficient_of_determination_indicates_better_fit_l456_45680


namespace pyramid_height_is_correct_l456_45612

noncomputable def pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : ℝ :=
  let side_length := perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)

theorem pyramid_height_is_correct :
  pyramid_height 40 15 = 5 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_is_correct_l456_45612


namespace distinct_real_roots_a1_l456_45684

theorem distinct_real_roots_a1 {x : ℝ} :
  ∀ a : ℝ, a = 1 →
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + (1 - a) * x1 - 1 = 0) ∧ (a * x2^2 + (1 - a) * x2 - 1 = 0) :=
by sorry

end distinct_real_roots_a1_l456_45684


namespace ring_worth_l456_45694

theorem ring_worth (R : ℝ) (h1 : (R + 2000 + 2 * R = 14000)) : R = 4000 :=
by 
  sorry

end ring_worth_l456_45694


namespace no_integer_roots_l456_45661
open Polynomial

theorem no_integer_roots {p : ℤ[X]} (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_pa : p.eval a = 1) (h_pb : p.eval b = 1) (h_pc : p.eval c = 1) : 
  ∀ m : ℤ, p.eval m ≠ 0 :=
by
  sorry

end no_integer_roots_l456_45661


namespace minimum_yellow_marbles_l456_45665

theorem minimum_yellow_marbles :
  ∀ (n y : ℕ), 
  (3 ∣ n) ∧ (4 ∣ n) ∧ 
  (9 + y + 2 * y ≤ n) ∧ 
  (n = n / 3 + n / 4 + 9 + y + 2 * y) → 
  y = 4 :=
by
  sorry

end minimum_yellow_marbles_l456_45665


namespace minimum_value_of_f_l456_45652

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem minimum_value_of_f :
  f 2 = -3 ∧ (∀ x : ℝ, f x ≥ -3) :=
by
  sorry

end minimum_value_of_f_l456_45652


namespace number_of_sheets_in_stack_l456_45629

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l456_45629


namespace range_of_x_l456_45631

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
   abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
  ↔ (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
by
  sorry

end range_of_x_l456_45631


namespace unique_solution_exists_l456_45690

theorem unique_solution_exists (a x y z : ℝ) 
  (h1 : z = a * (x + 2 * y + 5 / 2)) 
  (h2 : x^2 + y^2 + 2 * x - y + a * (x + 2 * y + 5 / 2) = 0) :
  a = 1 → x = -3 / 2 ∧ y = -1 / 2 ∧ z = 0 := 
by
  sorry

end unique_solution_exists_l456_45690


namespace smallest_b_for_factorization_l456_45635

theorem smallest_b_for_factorization : ∃ (b : ℕ), (∀ p q : ℤ, (x^2 + (b * x) + 2352) = (x + p) * (x + q) → p + q = b ∧ p * q = 2352) ∧ b = 112 := 
sorry

end smallest_b_for_factorization_l456_45635


namespace average_large_basket_weight_l456_45696

-- Definitions derived from the conditions
def small_basket_capacity := 25  -- Capacity of each small basket in kilograms
def num_small_baskets := 28      -- Number of small baskets used
def num_large_baskets := 10      -- Number of large baskets used
def leftover_weight := 50        -- Leftover weight in kilograms

-- Statement of the problem
theorem average_large_basket_weight :
  (small_basket_capacity * num_small_baskets - leftover_weight) / num_large_baskets = 65 :=
by
  sorry

end average_large_basket_weight_l456_45696


namespace fractions_product_equals_54_l456_45638

theorem fractions_product_equals_54 :
  (4 / 5) * (9 / 6) * (12 / 4) * (20 / 15) * (14 / 21) * (35 / 28) * (48 / 32) * (24 / 16) = 54 :=
by
  -- Add the proof here
  sorry

end fractions_product_equals_54_l456_45638


namespace find_S12_l456_45658

variable {a : Nat → Int} -- representing the arithmetic sequence {a_n}
variable {S : Nat → Int} -- representing the sums of the first n terms, S_n

-- Condition: a_1 = -9
axiom a1_def : a 1 = -9

-- Condition: (S_n / n) forms an arithmetic sequence
axiom arithmetic_s : ∃ d : Int, ∀ n : Nat, S n / n = -9 + (n - 1) * d

-- Condition: 2 = S9 / 9 - S7 / 7
axiom condition : S 9 / 9 - S 7 / 7 = 2

-- We want to prove: S_12 = 36
theorem find_S12 : S 12 = 36 := 
sorry

end find_S12_l456_45658


namespace minimum_green_sticks_l456_45614

def natasha_sticks (m n : ℕ) : ℕ :=
  if (m = 3 ∧ n = 3) then 5 else 0

theorem minimum_green_sticks (m n : ℕ) (grid : m = 3 ∧ n = 3) :
  natasha_sticks m n = 5 :=
by
  sorry

end minimum_green_sticks_l456_45614


namespace A_beats_B_by_63_l456_45603

variable (A B C : ℕ)

-- Condition: A beats C by 163 meters
def A_beats_C : Prop := A = 1000 - 163
-- Condition: B beats C by 100 meters
def B_beats_C (X : ℕ) : Prop := 1000 - X = 837 + 100
-- Main theorem statement
theorem A_beats_B_by_63 (X : ℕ) (h1 : A_beats_C A) (h2 : B_beats_C X): X = 63 :=
by
  sorry

end A_beats_B_by_63_l456_45603


namespace class_funding_reached_l456_45677

-- Definition of the conditions
def students : ℕ := 45
def goal : ℝ := 3000
def full_payment_students : ℕ := 25
def full_payment_amount : ℝ := 60
def merit_students : ℕ := 10
def merit_payment_per_student_euro : ℝ := 40
def euro_to_usd : ℝ := 1.20
def financial_needs_students : ℕ := 7
def financial_needs_payment_per_student_pound : ℝ := 30
def pound_to_usd : ℝ := 1.35
def discount_students : ℕ := 3
def discount_payment_per_student_cad : ℝ := 68
def cad_to_usd : ℝ := 0.80
def administrative_fee_yen : ℝ := 10000
def yen_to_usd : ℝ := 0.009

-- Definitions of amounts
def full_payment_amount_total : ℝ := full_payment_students * full_payment_amount
def merit_payment_amount_total : ℝ := merit_students * merit_payment_per_student_euro * euro_to_usd
def financial_needs_payment_amount_total : ℝ := financial_needs_students * financial_needs_payment_per_student_pound * pound_to_usd
def discount_payment_amount_total : ℝ := discount_students * discount_payment_per_student_cad * cad_to_usd
def administrative_fee_usd : ℝ := administrative_fee_yen * yen_to_usd

-- Definition of total collected
def total_collected : ℝ := 
  full_payment_amount_total + 
  merit_payment_amount_total + 
  financial_needs_payment_amount_total + 
  discount_payment_amount_total - 
  administrative_fee_usd

-- The final theorem statement
theorem class_funding_reached : total_collected = 2427.70 ∧ goal - total_collected = 572.30 := by
  sorry

end class_funding_reached_l456_45677


namespace compound_interest_calculation_l456_45668

-- Define the variables used in the problem
def principal : ℝ := 8000
def annual_rate : ℝ := 0.05
def compound_frequency : ℕ := 1
def final_amount : ℝ := 9261
def years : ℝ := 3

-- Statement we need to prove
theorem compound_interest_calculation :
  final_amount = principal * (1 + annual_rate / compound_frequency) ^ (compound_frequency * years) :=
by 
  sorry

end compound_interest_calculation_l456_45668


namespace largest_lcm_among_pairs_is_45_l456_45600

theorem largest_lcm_among_pairs_is_45 :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_among_pairs_is_45_l456_45600


namespace DansAgeCalculation_l456_45634

theorem DansAgeCalculation (D x : ℕ) (h1 : D = 8) (h2 : D + 20 = 7 * (D - x)) : x = 4 :=
by
  sorry

end DansAgeCalculation_l456_45634


namespace maria_original_number_25_3_l456_45698

theorem maria_original_number_25_3 (x : ℚ) 
  (h : ((3 * (x + 3) - 4) / 3) = 10) : 
  x = 25 / 3 := 
by 
  sorry

end maria_original_number_25_3_l456_45698


namespace total_pages_to_read_l456_45679

theorem total_pages_to_read 
  (total_books : ℕ)
  (pages_per_book : ℕ)
  (books_read_first_month : ℕ)
  (books_remaining_second_month : ℕ) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  books_remaining_second_month = (total_books - books_read_first_month) / 2 →
  ((total_books * pages_per_book) - ((books_read_first_month + books_remaining_second_month) * pages_per_book) = 1000) :=
by
  sorry

end total_pages_to_read_l456_45679


namespace carol_age_difference_l456_45628

theorem carol_age_difference (bob_age carol_age : ℕ) (h1 : bob_age + carol_age = 66)
  (h2 : carol_age = 3 * bob_age + 2) (h3 : bob_age = 16) (h4 : carol_age = 50) :
  carol_age - 3 * bob_age = 2 :=
by
  sorry

end carol_age_difference_l456_45628


namespace linear_coefficient_is_one_l456_45633

-- Define the given equation and the coefficient of the linear term
variables {x m : ℝ}
def equation := (m - 3) * x + 4 * m^2 - 2 * m - 1 - m * x + 6

-- State the main theorem: the coefficient of the linear term in the equation is 1 given the conditions
theorem linear_coefficient_is_one (m : ℝ) (hm_neq_3 : m ≠ 3) :
  (m - 3) - m = 1 :=
by sorry

end linear_coefficient_is_one_l456_45633


namespace exists_collinear_B_points_l456_45613

noncomputable def intersection (A B C D : Point) : Point :=
sorry

noncomputable def collinearity (P Q R S T : Point) : Prop :=
sorry

def convex_pentagon (A1 A2 A3 A4 A5 : Point) : Prop :=
-- Condition ensuring A1, A2, A3, A4, A5 form a convex pentagon, to be precisely defined
sorry

theorem exists_collinear_B_points :
  ∃ (A1 A2 A3 A4 A5 : Point),
    convex_pentagon A1 A2 A3 A4 A5 ∧
    collinearity
      (intersection A1 A4 A2 A3)
      (intersection A2 A5 A3 A4)
      (intersection A3 A1 A4 A5)
      (intersection A4 A2 A5 A1)
      (intersection A5 A3 A1 A2) :=
sorry

end exists_collinear_B_points_l456_45613


namespace power_multiplication_l456_45624

variable (p : ℝ)  -- Assuming p is a real number

theorem power_multiplication :
  (-p)^2 * (-p)^3 = -p^5 :=
sorry

end power_multiplication_l456_45624


namespace number_of_divisors_not_multiples_of_14_l456_45610

theorem number_of_divisors_not_multiples_of_14 
  (n : ℕ)
  (h1: ∃ k : ℕ, n = 2 * k * k)
  (h2: ∃ k : ℕ, n = 3 * k * k * k)
  (h3: ∃ k : ℕ, n = 5 * k * k * k * k * k)
  (h4: ∃ k : ℕ, n = 7 * k * k * k * k * k * k * k)
  : 
  ∃ num_divisors : ℕ, num_divisors = 19005 ∧ (∀ d : ℕ, d ∣ n → ¬(14 ∣ d)) := sorry

end number_of_divisors_not_multiples_of_14_l456_45610


namespace nina_money_l456_45674

theorem nina_money (W : ℝ) (P: ℝ) (Q : ℝ) 
  (h1 : P = 6 * W)
  (h2 : Q = 8 * (W - 1))
  (h3 : P = Q) 
  : P = 24 := 
by 
  sorry

end nina_money_l456_45674


namespace max_value_in_range_l456_45660

noncomputable def x_range : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}

noncomputable def expression (x : ℝ) : ℝ :=
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem max_value_in_range :
  ∀ x ∈ x_range, expression x ≤ (11 / 6) * Real.sqrt 3 :=
sorry

end max_value_in_range_l456_45660


namespace son_working_alone_l456_45640

theorem son_working_alone (M S : ℝ) (h1: M = 1 / 5) (h2: M + S = 1 / 3) : 1 / S = 7.5 :=
  by
  sorry

end son_working_alone_l456_45640


namespace triangle_inscribed_circle_area_l456_45689

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end triangle_inscribed_circle_area_l456_45689


namespace rabbits_total_distance_l456_45654

theorem rabbits_total_distance :
  let white_speed := 15
  let brown_speed := 12
  let grey_speed := 18
  let black_speed := 10
  let time := 7
  let white_distance := white_speed * time
  let brown_distance := brown_speed * time
  let grey_distance := grey_speed * time
  let black_distance := black_speed * time
  let total_distance := white_distance + brown_distance + grey_distance + black_distance
  total_distance = 385 :=
by
  sorry

end rabbits_total_distance_l456_45654


namespace value_of_trig_expression_l456_45616

theorem value_of_trig_expression (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 :=
by 
  sorry

end value_of_trig_expression_l456_45616


namespace g_of_f_eq_l456_45664

def f (A B x : ℝ) : ℝ := A * x^2 - B^2
def g (B x : ℝ) : ℝ := B * x + B^2

theorem g_of_f_eq (A B : ℝ) (hB : B ≠ 0) : 
  g B (f A B 1) = B * A - B^3 + B^2 := 
by
  sorry

end g_of_f_eq_l456_45664


namespace find_E_equals_2023_l456_45620

noncomputable def proof : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ (a^2 * (b + c) = 2023) ∧ (b^2 * (c + a) = 2023) ∧ (c^2 * (a + b) = 2023)

theorem find_E_equals_2023 : proof :=
by
  sorry

end find_E_equals_2023_l456_45620


namespace boric_acid_solution_l456_45663

theorem boric_acid_solution
  (amount_first_solution: ℝ) (percentage_first_solution: ℝ)
  (amount_second_solution: ℝ) (percentage_second_solution: ℝ)
  (final_amount: ℝ) (final_percentage: ℝ)
  (h1: amount_first_solution = 15)
  (h2: percentage_first_solution = 0.01)
  (h3: amount_second_solution = 15)
  (h4: final_amount = 30)
  (h5: final_percentage = 0.03)
  : percentage_second_solution = 0.05 := 
by
  sorry

end boric_acid_solution_l456_45663


namespace largest_gcd_sum_1089_l456_45639

theorem largest_gcd_sum_1089 (c d : ℕ) (h₁ : 0 < c) (h₂ : 0 < d) (h₃ : c + d = 1089) : ∃ k, k = Nat.gcd c d ∧ k = 363 :=
by
  sorry

end largest_gcd_sum_1089_l456_45639


namespace polygon_sides_diagonals_l456_45673

theorem polygon_sides_diagonals (n : ℕ) 
  (h1 : 4 * (n * (n - 3)) = 14 * n)
  (h2 : (n + (n * (n - 3)) / 2) % 2 = 0)
  (h3 : n + n * (n - 3) / 2 > 50) : n = 12 := 
by 
  sorry

end polygon_sides_diagonals_l456_45673


namespace ratio_between_house_and_park_l456_45675

theorem ratio_between_house_and_park (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0)
    (h : y / w = x / w + (x + y) / (10 * w)) : x / y = 9 / 11 :=
by 
  sorry

end ratio_between_house_and_park_l456_45675


namespace Bruce_Anne_combined_cleaning_time_l456_45608

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l456_45608


namespace men_wages_eq_13_5_l456_45606

-- Definitions based on problem conditions
def wages (men women boys : ℕ) : ℝ :=
  if 9 * men + women + 7 * boys = 216 then
    men
  else 
    0

def equivalent_wage (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage = women_wage ∧
  women_wage = 7 * boy_wage

def total_earning (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage + 7 * boy_wage = 216

-- Theorem statement
theorem men_wages_eq_13_5 (M_wage W_wage B_wage : ℝ) :
  equivalent_wage M_wage W_wage B_wage →
  total_earning M_wage W_wage B_wage →
  M_wage = 13.5 :=
by 
  intros h_equiv h_total
  sorry

end men_wages_eq_13_5_l456_45606


namespace greatest_k_divides_n_l456_45609

theorem greatest_k_divides_n (n : ℕ) (h_pos : 0 < n) (h_divisors_n : Nat.totient n = 72) (h_divisors_5n : Nat.totient (5 * n) = 90) : ∃ k : ℕ, ∀ m : ℕ, (5^k ∣ n) → (5^(k+1) ∣ n) → k = 3 :=
by
  sorry

end greatest_k_divides_n_l456_45609


namespace question1_question2_l456_45685

def energy_cost (units: ℕ) : ℝ :=
  if units <= 100 then
    units * 0.5
  else
    100 * 0.5 + (units - 100) * 0.8

theorem question1 :
  energy_cost 130 = 74 := by
  sorry

theorem question2 (units: ℕ) (H: energy_cost units = 90) :
  units = 150 := by
  sorry

end question1_question2_l456_45685


namespace part_a_part_b_l456_45662

-- Part (a)
theorem part_a (a b : ℕ) (h : Nat.lcm a (a + 5) = Nat.lcm b (b + 5)) : a = b :=
sorry

-- Part (b)
theorem part_b (a b c : ℕ) (gcd_abc : Nat.gcd a (Nat.gcd b c) = 1) :
  Nat.lcm a b = Nat.lcm (a + c) (b + c) → False :=
sorry

end part_a_part_b_l456_45662


namespace shape_with_congruent_views_is_sphere_l456_45615

def is_congruent_views (shape : Type) : Prop :=
  ∀ (front_view left_view top_view : shape), 
  (front_view = left_view) ∧ (left_view = top_view) ∧ (front_view = top_view)

noncomputable def is_sphere (shape : Type) : Prop := 
  ∀ (s : shape), true -- Placeholder definition for a sphere, as recognizing a sphere is outside Lean's scope

theorem shape_with_congruent_views_is_sphere (shape : Type) :
  is_congruent_views shape → is_sphere shape :=
by
  intro h
  sorry

end shape_with_congruent_views_is_sphere_l456_45615


namespace tickets_bought_l456_45630

theorem tickets_bought
  (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (leftover_money : ℕ)
  (total_money : ℕ) (money_spent : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : leftover_money = 83)
  (h5 : total_money = olivia_money + nigel_money)
  (h6 : total_money = 251)
  (h7 : money_spent = total_money - leftover_money)
  (h8 : money_spent = 168)
  : money_spent / ticket_cost = 6 := 
by
  sorry

end tickets_bought_l456_45630


namespace number_of_liars_on_the_island_l456_45681

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end number_of_liars_on_the_island_l456_45681


namespace probability_of_events_l456_45683

noncomputable def total_types : ℕ := 8

noncomputable def fever_reducing_types : ℕ := 3

noncomputable def cough_suppressing_types : ℕ := 5

noncomputable def total_ways_to_choose_two : ℕ := Nat.choose total_types 2

noncomputable def event_A_ways : ℕ := total_ways_to_choose_two - Nat.choose cough_suppressing_types 2

noncomputable def P_A : ℚ := event_A_ways / total_ways_to_choose_two

noncomputable def event_B_ways : ℕ := fever_reducing_types * cough_suppressing_types

noncomputable def P_B_given_A : ℚ := event_B_ways / event_A_ways

theorem probability_of_events :
  P_A = 9 / 14 ∧ P_B_given_A = 5 / 6 := by
  sorry

end probability_of_events_l456_45683


namespace constant_term_in_binomial_expansion_max_coef_sixth_term_l456_45656

theorem constant_term_in_binomial_expansion_max_coef_sixth_term 
  (n : ℕ) (h : n = 10) : 
  (∃ C : ℕ → ℕ → ℕ, C 10 2 * (Nat.sqrt 2) ^ 8 = 720) :=
sorry

end constant_term_in_binomial_expansion_max_coef_sixth_term_l456_45656


namespace dune_buggy_speed_l456_45666

theorem dune_buggy_speed (S : ℝ) :
  (1/3 * S + 1/3 * (S + 12) + 1/3 * (S - 18) = 58) → S = 60 :=
by
  sorry

end dune_buggy_speed_l456_45666


namespace minimum_dot_product_l456_45601

-- Define point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define points A, B, C, D according to the given problem statement
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 2⟩
def D : Point := ⟨0, 2⟩

-- Define the condition for points E and F on the sides BC and CD respectively.
def isOnBC (E : Point) : Prop := E.x = 1 ∧ 0 ≤ E.y ∧ E.y ≤ 2
def isOnCD (F : Point) : Prop := F.y = 2 ∧ 0 ≤ F.x ∧ F.x ≤ 1

-- Define the distance constraint for |EF| = 1
def distEF (E F : Point) : Prop :=
  (F.x - E.x)^2 + (F.y - E.y)^2 = 1

-- Define the dot product between vectors AE and AF
def dotProductAEAF (E F : Point) : ℝ :=
  2 * E.y + F.x

-- Main theorem to prove the minimum dot product value
theorem minimum_dot_product (E F : Point) (hE : isOnBC E) (hF : isOnCD F) (hDistEF : distEF E F) :
  dotProductAEAF E F = 5 - Real.sqrt 5 :=
  sorry

end minimum_dot_product_l456_45601


namespace no_first_quadrant_l456_45626

theorem no_first_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b < 0) (h_am : (a - b) < 0) :
  ¬∃ x : ℝ, (a - b) * x + b > 0 ∧ x > 0 :=
sorry

end no_first_quadrant_l456_45626


namespace house_to_market_distance_l456_45611

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l456_45611


namespace Kolya_walking_speed_l456_45682

theorem Kolya_walking_speed
  (x : ℝ) 
  (h1 : x > 0) 
  (t_closing : ℝ := (3 * x) / 10) 
  (t_travel : ℝ := ((x / 10) + (x / 20))) 
  (remaining_time : ℝ := t_closing - t_travel)
  (walking_speed : ℝ := x / remaining_time)
  (correct_speed : ℝ := 20 / 3) :
  walking_speed = correct_speed := 
by 
  sorry

end Kolya_walking_speed_l456_45682


namespace problem_statement_l456_45646

variable {P : ℕ → Prop}

theorem problem_statement
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬P 4)
  (n : ℕ) (hn : 1 ≤ n → n ≤ 4 → n ∈ Set.Icc 1 4) :
  ¬P n :=
by
  sorry

end problem_statement_l456_45646


namespace total_number_of_apples_l456_45645

namespace Apples

def red_apples : ℕ := 7
def green_apples : ℕ := 2
def total_apples : ℕ := red_apples + green_apples

theorem total_number_of_apples : total_apples = 9 := by
  -- Definition of total_apples is used directly from conditions.
  -- Conditions state there are 7 red apples and 2 green apples.
  -- Therefore, total_apples = 7 + 2 = 9.
  sorry

end Apples

end total_number_of_apples_l456_45645


namespace find_x_axis_intercept_l456_45655

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end find_x_axis_intercept_l456_45655


namespace solve_quadratic_equation_l456_45607

theorem solve_quadratic_equation (x : ℝ) : x^2 = 100 → x = -10 ∨ x = 10 :=
by
  intro h
  sorry

end solve_quadratic_equation_l456_45607


namespace lcm_16_24_l456_45659

/-
  Prove that the least common multiple (LCM) of 16 and 24 is 48.
-/
theorem lcm_16_24 : Nat.lcm 16 24 = 48 :=
by
  sorry

end lcm_16_24_l456_45659


namespace graduate_degree_ratio_l456_45687

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end graduate_degree_ratio_l456_45687


namespace abc_inequality_l456_45647

theorem abc_inequality 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end abc_inequality_l456_45647


namespace general_term_of_arithmetic_sequence_l456_45632

theorem general_term_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = -2)
  (h_a7 : a 7 = -10) :
  ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end general_term_of_arithmetic_sequence_l456_45632


namespace cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l456_45671

noncomputable def cos_135_deg : Real := - (Real.sqrt 2) / 2

theorem cos_135_eq_neg_sqrt_2_div_2 : Real.cos (135 * Real.pi / 180) = cos_135_deg := sorry

noncomputable def point_Q : Real × Real :=
  (- (Real.sqrt 2) / 2, (Real.sqrt 2) / 2)

theorem point_Q_coordinates :
  ∃ (Q : Real × Real), Q = point_Q ∧ Q = (Real.cos (135 * Real.pi / 180), Real.sin (135 * Real.pi / 180)) := sorry

end cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l456_45671


namespace negation_of_p_implies_a_gt_one_half_l456_45693

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + x + 1 / 2 ≤ 0

-- Define the statement that negation of p implies a > 1/2
theorem negation_of_p_implies_a_gt_one_half (a : ℝ) (h : ¬ p a) : a > 1 / 2 :=
by
  sorry

end negation_of_p_implies_a_gt_one_half_l456_45693


namespace smallest_even_natural_number_l456_45670

theorem smallest_even_natural_number (a : ℕ) :
  ( ∃ a, a % 2 = 0 ∧
    (a + 1) % 3 = 0 ∧
    (a + 2) % 5 = 0 ∧
    (a + 3) % 7 = 0 ∧
    (a + 4) % 11 = 0 ∧
    (a + 5) % 13 = 0 ) → 
  a = 788 := by
  sorry

end smallest_even_natural_number_l456_45670


namespace point_to_focus_distance_l456_45688

def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }

def point_P : ℝ × ℝ := (3, 2) -- Since y^2 = 4*3 hence y = ±2 and we choose one of the (3, 2) or (3, -2)

def focus_F : ℝ × ℝ := (1, 0) -- Focus of y^2 = 4x is (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_to_focus_distance : distance point_P focus_F = 4 := by
  sorry -- Proof goes here

end point_to_focus_distance_l456_45688


namespace ratio_of_capital_l456_45643

variable (C A B : ℝ)
variable (h1 : B = 4 * C)
variable (h2 : B / (A + 5 * C) = 6000 / 16500)

theorem ratio_of_capital : A / B = 17 / 4 :=
by
  sorry

end ratio_of_capital_l456_45643


namespace power_of_two_with_nines_l456_45605

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ (n : ℕ), (2^n % 10^k) / 10^((10 * 5^k + k + 2 - k) / 2) = 9 :=
sorry

end power_of_two_with_nines_l456_45605


namespace total_time_naomi_30webs_l456_45669

-- Define the constants based on the given conditions
def time_katherine : ℕ := 20
def factor_naomi : ℚ := 5/4
def websites : ℕ := 30

-- Define the time taken by Naomi to build one website based on the conditions
def time_naomi (time_katherine : ℕ) (factor_naomi : ℚ) : ℚ :=
  factor_naomi * time_katherine

-- Define the total time Naomi took to build all websites
def total_time_naomi (time_naomi : ℚ) (websites : ℕ) : ℚ :=
  time_naomi * websites

-- Statement: Proving that the total number of hours Naomi took to create 30 websites is 750
theorem total_time_naomi_30webs : 
  total_time_naomi (time_naomi time_katherine factor_naomi) websites = 750 := 
sorry

end total_time_naomi_30webs_l456_45669


namespace price_arun_paid_l456_45678

theorem price_arun_paid 
  (original_price : ℝ)
  (standard_concession_rate : ℝ) 
  (additional_concession_rate : ℝ)
  (reduced_price : ℝ)
  (final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : standard_concession_rate = 0.30)
  (h3 : additional_concession_rate = 0.20)
  (h4 : reduced_price = original_price * (1 - standard_concession_rate))
  (h5 : final_price = reduced_price * (1 - additional_concession_rate)) :
  final_price = 1120 :=
by
  sorry

end price_arun_paid_l456_45678


namespace value_of_a_l456_45653

theorem value_of_a (x y a : ℝ) (h1 : x - 2 * y = a - 6) (h2 : 2 * x + 5 * y = 2 * a) (h3 : x + y = 9) : a = 11 := 
by
  sorry

end value_of_a_l456_45653


namespace problem_statement_l456_45627

def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f (g (f (g 2))) = 7189058 := by
  sorry

end problem_statement_l456_45627
