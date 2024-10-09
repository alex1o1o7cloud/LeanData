import Mathlib

namespace proportion_a_value_l265_26545

theorem proportion_a_value (a b c d : ℝ) (h1 : b = 3) (h2 : c = 4) (h3 : d = 6) (h4 : a / b = c / d) : a = 2 :=
by sorry

end proportion_a_value_l265_26545


namespace min_expr_value_min_expr_value_iff_l265_26559

theorem min_expr_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 :=
by {
  sorry
}

theorem min_expr_value_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2) = 4 / 9) ↔ (x = 2.5 ∧ y = 2.5) :=
by {
  sorry
}

end min_expr_value_min_expr_value_iff_l265_26559


namespace number_of_containers_needed_l265_26535

/-
  Define the parameters for the given problem
-/
def bags_suki : ℝ := 6.75
def weight_per_bag_suki : ℝ := 27

def bags_jimmy : ℝ := 4.25
def weight_per_bag_jimmy : ℝ := 23

def bags_natasha : ℝ := 3.80
def weight_per_bag_natasha : ℝ := 31

def container_capacity : ℝ := 17

/-
  The total weight bought by each person and the total combined weight
-/
def total_weight_suki : ℝ := bags_suki * weight_per_bag_suki
def total_weight_jimmy : ℝ := bags_jimmy * weight_per_bag_jimmy
def total_weight_natasha : ℝ := bags_natasha * weight_per_bag_natasha

def total_weight_combined : ℝ := total_weight_suki + total_weight_jimmy + total_weight_natasha

/-
  Prove that number of containers needed is 24
-/
theorem number_of_containers_needed : 
  Nat.ceil (total_weight_combined / container_capacity) = 24 := 
by
  sorry

end number_of_containers_needed_l265_26535


namespace John_is_26_l265_26524

-- Define the variables representing the ages
def John_age : ℕ := 26
def Grandmother_age : ℕ := John_age + 48

-- Conditions
def condition1 : Prop := John_age = Grandmother_age - 48
def condition2 : Prop := John_age + Grandmother_age = 100

-- Main theorem to prove: John is 26 years old
theorem John_is_26 : John_age = 26 :=
by
  have h1 : condition1 := by sorry
  have h2 : condition2 := by sorry
  -- More steps to combine the conditions and prove the theorem would go here
  -- Skipping proof steps with sorry for demonstration
  sorry

end John_is_26_l265_26524


namespace polynomials_with_sum_of_abs_values_and_degree_eq_4_l265_26586

-- We define the general structure and conditions of the problem.
def polynomial_count : ℕ := 
  let count_0 := 1 -- For n = 0
  let count_1 := 6 -- For n = 1
  let count_2 := 9 -- For n = 2
  let count_3 := 1 -- For n = 3
  count_0 + count_1 + count_2 + count_3

theorem polynomials_with_sum_of_abs_values_and_degree_eq_4 : polynomial_count = 17 := 
by
  unfold polynomial_count
  -- The detailed proof steps for the count would go here
  sorry

end polynomials_with_sum_of_abs_values_and_degree_eq_4_l265_26586


namespace fiona_working_hours_l265_26548

theorem fiona_working_hours (F : ℕ) 
  (John_hours_per_week : ℕ := 30) 
  (Jeremy_hours_per_week : ℕ := 25) 
  (pay_rate : ℕ := 20) 
  (monthly_total_pay : ℕ := 7600) : 
  4 * (John_hours_per_week * pay_rate + Jeremy_hours_per_week * pay_rate + F * pay_rate) = monthly_total_pay → 
  F = 40 :=
by sorry

end fiona_working_hours_l265_26548


namespace sequence_neither_arithmetic_nor_geometric_l265_26558

noncomputable def Sn (n : ℕ) : ℕ := 3 * n + 2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 5 else Sn n - Sn (n - 1)

def not_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ d, ∀ n, a (n + 1) = a n + d

def not_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ¬∃ r, ∀ n, a (n + 1) = r * a n

theorem sequence_neither_arithmetic_nor_geometric :
  not_arithmetic_sequence a ∧ not_geometric_sequence a :=
sorry

end sequence_neither_arithmetic_nor_geometric_l265_26558


namespace cylinder_lateral_surface_area_l265_26537

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (radius_eq : r = 2) 
  (height_eq : h = 5) : 
  2 * Real.pi * r * h = 62.8 :=
by
  -- Proof steps go here
  sorry

end cylinder_lateral_surface_area_l265_26537


namespace angle_sum_solution_l265_26556

theorem angle_sum_solution
  (x : ℝ)
  (h : 3 * x + 140 = 360) :
  x = 220 / 3 :=
by
  sorry

end angle_sum_solution_l265_26556


namespace trig_identity_75_30_15_150_l265_26532

theorem trig_identity_75_30_15_150 :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - 
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end trig_identity_75_30_15_150_l265_26532


namespace mabel_visits_helen_l265_26570

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l265_26570


namespace solve_system_exists_l265_26525

theorem solve_system_exists (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : 1 / x + 1 / y + 1 / z = 5 / 12) 
  (h3 : x^3 + y^3 + z^3 = 45) 
  : (x, y, z) = (2, -3, 4) ∨ (x, y, z) = (2, 4, -3) ∨ (x, y, z) = (-3, 2, 4) ∨ (x, y, z) = (-3, 4, 2) ∨ (x, y, z) = (4, 2, -3) ∨ (x, y, z) = (4, -3, 2) := 
sorry

end solve_system_exists_l265_26525


namespace quadratic_polynomial_coefficients_l265_26575

theorem quadratic_polynomial_coefficients (a b : ℝ)
  (h1 : 2 * a - 1 - b = 0)
  (h2 : 5 * a + b - 13 = 0) :
  a^2 + b^2 = 13 := 
by 
  sorry

end quadratic_polynomial_coefficients_l265_26575


namespace solve_trig_system_l265_26529

theorem solve_trig_system
  (k n : ℤ) :
  (∃ x y : ℝ,
    (2 * Real.sin x ^ 2 + 2 * Real.sqrt 2 * Real.sin x * Real.sin (2 * x) ^ 2 + Real.sin (2 * x) ^ 2 = 0 ∧
     Real.cos x = Real.cos y) ∧
    ((x = 2 * Real.pi * k ∧ y = 2 * Real.pi * n) ∨
     (x = Real.pi + 2 * Real.pi * k ∧ y = Real.pi + 2 * Real.pi * n) ∨
     (x = -Real.pi / 4 + 2 * Real.pi * k ∧ (y = Real.pi / 4 + 2 * Real.pi * n ∨ y = -Real.pi / 4 + 2 * Real.pi * n)) ∨
     (x = -3 * Real.pi / 4 + 2 * Real.pi * k ∧ (y = 3 * Real.pi / 4 + 2 * Real.pi * n ∨ y = -3 * Real.pi / 4 + 2 * Real.pi * n)))) :=
sorry

end solve_trig_system_l265_26529


namespace general_term_of_sequence_l265_26510

theorem general_term_of_sequence (a : Nat → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (2 + a (n + 1))) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 2) := 
sorry

end general_term_of_sequence_l265_26510


namespace number_of_dogs_on_tuesday_l265_26538

variable (T : ℕ)
variable (H1 : 7 + T + 7 + 7 + 9 = 42)

theorem number_of_dogs_on_tuesday : T = 12 := by
  sorry

end number_of_dogs_on_tuesday_l265_26538


namespace businessmen_neither_coffee_nor_tea_l265_26571

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (C T : Finset ℕ)
  (hC : C.card = 15)
  (hT : T.card = 14)
  (hCT : (C ∩ T).card = 7)
  (htotal : total = 30) : 
  total - (C ∪ T).card = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l265_26571


namespace geometric_sequence_increasing_iff_q_gt_one_l265_26506

variables {a_n : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n (n + 1) > a_n n

theorem geometric_sequence_increasing_iff_q_gt_one 
  (h1 : ∀ n, 0 < a_n n)
  (h2 : is_geometric_sequence a_n q) :
  is_increasing_sequence a_n ↔ q > 1 :=
by
  sorry

end geometric_sequence_increasing_iff_q_gt_one_l265_26506


namespace john_cards_sum_l265_26589

theorem john_cards_sum :
  ∃ (g : ℕ → ℕ) (y : ℕ → ℕ),
    (∀ n, (g n) ∈ [1, 2, 3, 4, 5]) ∧
    (∀ n, (y n) ∈ [2, 3, 4, 5]) ∧
    (∀ n, (g n < g (n + 1))) ∧
    (∀ n, (y n < y (n + 1))) ∧
    (∀ n, (g n ∣ y (n + 1) ∨ y (n + 1) ∣ g n)) ∧
    (g 0 = 1 ∧ g 2 = 2 ∧ g 4 = 5) ∧
    ( y 1 = 2 ∧ y 3 = 3 ∧ y 5 = 4 ) →
  g 0 + g 2 + g 4 = 8 := by
sorry

end john_cards_sum_l265_26589


namespace simplify_expression_l265_26595

-- Define the variables and the polynomials
variables (y : ℤ)

-- Define the expressions
def expr1 := (2 * y - 1) * (5 * y^12 - 3 * y^11 + y^9 - 4 * y^8)
def expr2 := 10 * y^13 - 11 * y^12 + 3 * y^11 + y^10 - 9 * y^9 + 4 * y^8

-- State the theorem
theorem simplify_expression : expr1 = expr2 := by
  sorry

end simplify_expression_l265_26595


namespace min_value_of_squared_sums_l265_26502

theorem min_value_of_squared_sums (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ B, (B = x^2 + y^2 + z^2) ∧ (B ≥ 4) := 
by {
  sorry -- Proof will be provided here.
}

end min_value_of_squared_sums_l265_26502


namespace A_plus_B_eq_93_l265_26599

-- Definitions and conditions
def gcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)
def lcm (a b c : ℕ) : ℕ := a * b * c / (gcf a b c)

-- Values for A and B
def A := gcf 18 30 45
def B := lcm 18 30 45

-- Proof statement
theorem A_plus_B_eq_93 : A + B = 93 := by
  sorry

end A_plus_B_eq_93_l265_26599


namespace smallest_n_for_three_pairs_l265_26526

theorem smallest_n_for_three_pairs :
  ∃ (n : ℕ), (0 < n) ∧
    (∀ (x y : ℕ), (x^2 - y^2 = n) → (0 < x) ∧ (0 < y)) ∧
    (∃ (a b c : ℕ), 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
      (∃ (x y : ℕ), (x^2 - y^2 = n) ∧
        (((x, y) = (a, b)) ∨ ((x, y) = (b, c)) ∨ ((x, y) = (a, c))))) :=
sorry

end smallest_n_for_three_pairs_l265_26526


namespace expression_odd_if_p_q_odd_l265_26522

variable (p q : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem expression_odd_if_p_q_odd (hp : is_odd p) (hq : is_odd q) : is_odd (5 * p * q) :=
sorry

end expression_odd_if_p_q_odd_l265_26522


namespace infinite_rational_points_in_region_l265_26523

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
  (∀ p ∈ S, (p.1 ^ 2 + p.2 ^ 2 ≤ 16) ∧ (p.1 ≤ 3) ∧ (p.2 ≤ 3) ∧ (p.1 > 0) ∧ (p.2 > 0)) ∧
  Set.Infinite S :=
sorry

end infinite_rational_points_in_region_l265_26523


namespace total_selling_price_correct_l265_26578

noncomputable def calculateSellingPrice (price1 price2 price3 loss1 loss2 loss3 taxRate overheadCost : ℝ) : ℝ :=
  let totalPurchasePrice := price1 + price2 + price3
  let tax := taxRate * totalPurchasePrice
  let sellingPrice1 := price1 - (loss1 * price1)
  let sellingPrice2 := price2 - (loss2 * price2)
  let sellingPrice3 := price3 - (loss3 * price3)
  let totalSellingPrice := sellingPrice1 + sellingPrice2 + sellingPrice3
  totalSellingPrice + overheadCost + tax

theorem total_selling_price_correct :
  calculateSellingPrice 750 1200 500 0.10 0.15 0.05 0.05 300 = 2592.5 :=
by 
  -- The proof of this theorem is skipped.
  sorry

end total_selling_price_correct_l265_26578


namespace inequality_proof_l265_26579

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := 
by
  sorry

end inequality_proof_l265_26579


namespace triangle_angle_inradius_l265_26520

variable (A B C : ℝ) 
variable (a b c R : ℝ)

theorem triangle_angle_inradius 
    (h1: 0 < A ∧ A < Real.pi)
    (h2: a * Real.cos C + (1/2) * c = b)
    (h3: a = 1):

    A = Real.pi / 3 ∧ R ≤ Real.sqrt 3 / 6 := 
by
  sorry

end triangle_angle_inradius_l265_26520


namespace find_m_l265_26565

theorem find_m (m a : ℝ) (h : (2:ℝ) * 1^2 - 3 * 1 + a = 0) 
  (h_roots : ∀ x : ℝ, 2 * x^2 - 3 * x + a = 0 → (x = 1 ∨ x = m)) :
  m = 1 / 2 :=
by
  sorry

end find_m_l265_26565


namespace frank_bags_on_saturday_l265_26562

def bags_filled_on_saturday (total_cans : Nat) (cans_per_bag : Nat) (bags_on_sunday : Nat) : Nat :=
  total_cans / cans_per_bag - bags_on_sunday

theorem frank_bags_on_saturday : 
  let total_cans := 40
  let cans_per_bag := 5
  let bags_on_sunday := 3
  bags_filled_on_saturday total_cans cans_per_bag bags_on_sunday = 5 :=
  by
  -- Proof to be provided
  sorry

end frank_bags_on_saturday_l265_26562


namespace symmetricPointCorrectCount_l265_26561

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the four symmetry conditions
def isSymmetricXaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricYOZplane (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := -P.z }
def isSymmetricYaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricOrigin (P Q : Point3D) : Prop := Q = { x := -P.x, y := -P.y, z := -P.z }

-- Define a theorem to count the valid symmetric conditions
theorem symmetricPointCorrectCount (P : Point3D) :
  (isSymmetricXaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricYOZplane P { x := P.x, y := -P.y, z := -P.z } = true → false) ∧
  (isSymmetricYaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricOrigin P { x := -P.x, y := -P.y, z := -P.z } = true → true) :=
by
  sorry

end symmetricPointCorrectCount_l265_26561


namespace no_such_function_exists_l265_26560

theorem no_such_function_exists :
  ¬ (∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l265_26560


namespace smallest_sum_abc_d_l265_26547

theorem smallest_sum_abc_d (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) : a + b + c + d = 108 :=
sorry

end smallest_sum_abc_d_l265_26547


namespace ratio_of_kits_to_students_l265_26587

theorem ratio_of_kits_to_students (art_kits students : ℕ) (h1 : art_kits = 20) (h2 : students = 10) : art_kits / Nat.gcd art_kits students = 2 ∧ students / Nat.gcd art_kits students = 1 := by
  sorry

end ratio_of_kits_to_students_l265_26587


namespace find_k_l265_26585

theorem find_k (k x y : ℝ) (h1 : x = 2) (h2 : y = -3)
    (h3 : 2 * x^2 + k * x * y = 4) : k = 2 / 3 :=
by
  sorry

end find_k_l265_26585


namespace infinite_points_of_one_color_l265_26553

theorem infinite_points_of_one_color (colors : ℤ → Prop) (red blue : ℤ → Prop)
  (h_colors : ∀ n : ℤ, colors n → (red n ∨ blue n))
  (h_red_blue : ∀ n : ℤ, red n → ¬ blue n)
  (h_blue_red : ∀ n : ℤ, blue n → ¬ red n) :
  ∃ c : ℤ → Prop, (∀ k : ℕ, ∃ infinitely_many p : ℤ, c p ∧ p % k = 0) :=
by
  sorry

end infinite_points_of_one_color_l265_26553


namespace correct_options_l265_26582

open Real

def option_A (x : ℝ) : Prop :=
  x^2 - 2*x + 1 > 0

def option_B : Prop :=
  ∃ (x : ℝ), (0 < x) ∧ (x + 4 / x = 6)

def option_C (a b : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) → (b / a + a / b ≥ 2)

def option_D (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (x + 2*y = 1) → (2 / x + 1 / y ≥ 8)

theorem correct_options :
  ¬(∀ (x : ℝ), option_A x) ∧ (option_B ∧ (∀ (a b : ℝ), option_C a b) = false ∧ 
  (∀ (x y : ℝ), option_D x y)) :=
by sorry

end correct_options_l265_26582


namespace average_weight_of_boys_l265_26516

theorem average_weight_of_boys
  (average_weight_girls : ℕ) 
  (average_weight_students : ℕ) 
  (h_girls : average_weight_girls = 45)
  (h_students : average_weight_students = 50) : 
  ∃ average_weight_boys : ℕ, average_weight_boys = 55 :=
by
  sorry

end average_weight_of_boys_l265_26516


namespace four_digit_even_and_multiple_of_7_sum_l265_26508

def num_four_digit_even_numbers : ℕ := 4500
def num_four_digit_multiples_of_7 : ℕ := 1286
def C : ℕ := num_four_digit_even_numbers
def D : ℕ := num_four_digit_multiples_of_7

theorem four_digit_even_and_multiple_of_7_sum :
  C + D = 5786 := by
  sorry

end four_digit_even_and_multiple_of_7_sum_l265_26508


namespace remaining_wire_length_l265_26515

theorem remaining_wire_length (total_wire_length : ℝ) (square_side_length : ℝ) 
  (h₀ : total_wire_length = 60) (h₁ : square_side_length = 9) : 
  total_wire_length - 4 * square_side_length = 24 :=
by
  sorry

end remaining_wire_length_l265_26515


namespace value_of_nested_expression_l265_26597

def nested_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2

theorem value_of_nested_expression : nested_expression = 1457 := by
  sorry

end value_of_nested_expression_l265_26597


namespace jaime_average_speed_l265_26584

theorem jaime_average_speed :
  let start_time := 10.0 -- 10:00 AM
  let end_time := 15.5 -- 3:30 PM (in 24-hour format)
  let total_distance := 21.0 -- kilometers
  let total_time := end_time - start_time -- time in hours
  total_distance / total_time = 3.82 := 
sorry

end jaime_average_speed_l265_26584


namespace simplify_and_evaluate_expression_l265_26591

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -6) : 
  (1 - a / (a - 3)) / ((a^2 + 3 * a) / (a^2 - 9)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l265_26591


namespace percent_defective_units_l265_26531

theorem percent_defective_units (D : ℝ) (h1 : 0.05 * D = 0.5) : D = 10 := by
  sorry

end percent_defective_units_l265_26531


namespace value_of_expression_l265_26530

theorem value_of_expression : (165^2 - 153^2) / 12 = 318 := by
  sorry

end value_of_expression_l265_26530


namespace tanya_time_proof_l265_26514

noncomputable def time_sakshi : ℝ := 10
noncomputable def efficiency_increase : ℝ := 1.25
noncomputable def time_tanya (time_sakshi : ℝ) (efficiency_increase : ℝ) : ℝ := time_sakshi / efficiency_increase

theorem tanya_time_proof : time_tanya time_sakshi efficiency_increase = 8 := 
by 
  sorry

end tanya_time_proof_l265_26514


namespace flag_arrangement_division_l265_26536

noncomputable def flag_arrangement_modulo : ℕ :=
  let num_blue_flags := 9
  let num_red_flags := 8
  let num_slots := num_blue_flags + 1
  let initial_arrangements := (num_slots.choose num_red_flags) * (num_blue_flags + 1)
  let invalid_cases := (num_blue_flags.choose num_red_flags) * 2
  let M := initial_arrangements - invalid_cases
  M % 1000

theorem flag_arrangement_division (M : ℕ) (num_blue_flags num_red_flags : ℕ) :
  num_blue_flags = 9 → num_red_flags = 8 → M = flag_arrangement_modulo → M % 1000 = 432 :=
by
  intros _ _ hM
  rw [hM]
  trivial

end flag_arrangement_division_l265_26536


namespace find_g_inv_f_3_l265_26543

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq : ∀ x : ℝ, f_inv (g x) = x^4 - x + 2
axiom g_has_inverse : ∀ y : ℝ, g (g_inv y) = y 

theorem find_g_inv_f_3 :
  ∃ α : ℝ, (α^4 - α - 1 = 0) ∧ g_inv (f 3) = α :=
sorry

end find_g_inv_f_3_l265_26543


namespace correct_statement_C_l265_26550

theorem correct_statement_C : (∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5)) :=
by
  sorry

end correct_statement_C_l265_26550


namespace monotonicity_and_inequality_l265_26557

noncomputable def f (x : ℝ) := 2 * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) (hF_nonneg : ∀ x, F a x ≥ 0) (h_lt : x₁ < x₂) :
  (F a x₂ - F a x₁) / (x₂ - x₁) > 2 * (Real.exp x₁ - 1) :=
sorry

end monotonicity_and_inequality_l265_26557


namespace sum_of_two_numbers_l265_26588

theorem sum_of_two_numbers (x y : ℕ) (h1 : 3 * x = 180) (h2 : 4 * x = y) : x + y = 420 := by
  sorry

end sum_of_two_numbers_l265_26588


namespace total_flowers_l265_26503

-- Definition of conditions
def minyoung_flowers : ℕ := 24
def yoojung_flowers (y : ℕ) : Prop := minyoung_flowers = 4 * y

-- Theorem statement
theorem total_flowers (y : ℕ) (h : yoojung_flowers y) : minyoung_flowers + y = 30 :=
by sorry

end total_flowers_l265_26503


namespace solve_system_eq_l265_26528

theorem solve_system_eq (x y z : ℝ) :
  (x * y * z / (x + y) = 6 / 5) ∧
  (x * y * z / (y + z) = 2) ∧
  (x * y * z / (z + x) = 3 / 2) ↔
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) := 
by
  -- proof to be provided
  sorry

end solve_system_eq_l265_26528


namespace kristi_books_proof_l265_26564

variable (Bobby_books Kristi_books : ℕ)

def condition1 : Prop := Bobby_books = 142

def condition2 : Prop := Bobby_books = Kristi_books + 64

theorem kristi_books_proof (h1 : condition1 Bobby_books) (h2 : condition2 Bobby_books Kristi_books) : Kristi_books = 78 := 
by 
  sorry

end kristi_books_proof_l265_26564


namespace evaluate_polynomial_at_3_l265_26594

def f (x : ℕ) : ℕ := 3 * x ^ 3 + x - 3

theorem evaluate_polynomial_at_3 : f 3 = 28 :=
by
  sorry

end evaluate_polynomial_at_3_l265_26594


namespace trailing_zeros_in_square_l265_26542

-- Define x as given in the conditions
def x : ℕ := 10^12 - 4

-- State the theorem which asserts that the number of trailing zeros in x^2 is 11
theorem trailing_zeros_in_square : 
  ∃ n : ℕ, n = 11 ∧ x^2 % 10^12 = 0 :=
by
  -- Placeholder for the proof
  sorry

end trailing_zeros_in_square_l265_26542


namespace lateral_surface_area_of_frustum_l265_26577

theorem lateral_surface_area_of_frustum (slant_height : ℝ) (ratio : ℕ × ℕ) (central_angle_deg : ℝ)
  (h_slant_height : slant_height = 10) 
  (h_ratio : ratio = (2, 5)) 
  (h_central_angle_deg : central_angle_deg = 216) : 
  ∃ (area : ℝ), area = (252 * Real.pi / 5) := 
by 
  sorry

end lateral_surface_area_of_frustum_l265_26577


namespace average_stamps_collected_per_day_l265_26544

theorem average_stamps_collected_per_day :
  let a := 10
  let d := 6
  let n := 6
  let total_sum := (n / 2) * (2 * a + (n - 1) * d)
  let average := total_sum / n
  average = 25 :=
by
  sorry

end average_stamps_collected_per_day_l265_26544


namespace amy_lily_tie_probability_l265_26596

theorem amy_lily_tie_probability (P_Amy P_Lily : ℚ) (hAmy : P_Amy = 4/9) (hLily : P_Lily = 1/3) :
  1 - P_Amy - (↑P_Lily : ℚ) = 2 / 9 := by
  sorry

end amy_lily_tie_probability_l265_26596


namespace train_length_l265_26583

theorem train_length (L : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = L / 15) 
  (h2 : V2 = (L + 800) / 45) 
  (h3 : V1 = V2) : 
  L = 400 := 
sorry

end train_length_l265_26583


namespace buckets_oranges_l265_26541

theorem buckets_oranges :
  ∀ (a b c : ℕ), 
  a = 22 → 
  b = a + 17 → 
  a + b + c = 89 → 
  b - c = 11 := 
by 
  intros a b c h1 h2 h3 
  sorry

end buckets_oranges_l265_26541


namespace xiaolin_final_score_l265_26552

-- Define the conditions
def score_situps : ℕ := 80
def score_800m : ℕ := 90
def weight_situps : ℕ := 4
def weight_800m : ℕ := 6

-- Define the final score based on the given conditions
def final_score : ℕ :=
  (score_situps * weight_situps + score_800m * weight_800m) / (weight_situps + weight_800m)

-- Prove that the final score is 86
theorem xiaolin_final_score : final_score = 86 :=
by sorry

end xiaolin_final_score_l265_26552


namespace anusha_solution_l265_26509

variable (A B E : ℝ) -- Defining the variables for amounts received by Anusha, Babu, and Esha
variable (total_amount : ℝ) (h_division : 12 * A = 8 * B) (h_division2 : 8 * B = 6 * E) (h_total : A + B + E = 378)

theorem anusha_solution : A = 84 :=
by
  -- Using the given conditions and deriving the amount Anusha receives
  sorry

end anusha_solution_l265_26509


namespace rhombus_area_l265_26569

-- Define d1 and d2 as the lengths of the diagonals
def d1 : ℝ := 15
def d2 : ℝ := 17

-- The theorem to prove the area of the rhombus
theorem rhombus_area : (d1 * d2) / 2 = 127.5 := by
  sorry

end rhombus_area_l265_26569


namespace find_number_l265_26551

theorem find_number (number : ℤ) (h : number + 7 = 6) : number = -1 :=
by
  sorry

end find_number_l265_26551


namespace minimum_expression_value_l265_26511

theorem minimum_expression_value (a b c : ℝ) (hbpos : b > 0) (hab : b > a) (hcb : b > c) (hca : c > a) :
  (a + 2 * b) ^ 2 / b ^ 2 + (b - 2 * c) ^ 2 / b ^ 2 + (c - 2 * a) ^ 2 / b ^ 2 ≥ 65 / 16 := 
sorry

end minimum_expression_value_l265_26511


namespace number_of_integers_l265_26521

theorem number_of_integers (n : ℤ) : 
    (100 < n ∧ n < 300) ∧ (n % 7 = n % 9) → 
    (∃ count: ℕ, count = 21) := by
  sorry

end number_of_integers_l265_26521


namespace greg_age_is_18_l265_26501

def diana_age : ℕ := 15
def eduardo_age (c : ℕ) : ℕ := 2 * c
def chad_age (c : ℕ) : ℕ := c
def faye_age (c : ℕ) : ℕ := c - 1
def greg_age (c : ℕ) : ℕ := 2 * (c - 1)
def diana_relation (c : ℕ) : Prop := 15 = (2 * c) - 5

theorem greg_age_is_18 (c : ℕ) (h : diana_relation c) :
  greg_age c = 18 :=
by
  sorry

end greg_age_is_18_l265_26501


namespace sub_one_inequality_l265_26513

theorem sub_one_inequality (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end sub_one_inequality_l265_26513


namespace product_of_consecutive_integers_sqrt_50_l265_26554

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l265_26554


namespace cube_surface_area_l265_26580

theorem cube_surface_area (PQ a b : ℝ) (x : ℝ) 
  (h1 : PQ = a / 2) 
  (h2 : PQ = Real.sqrt (3 * x^2)) : 
  b = 6 * x^2 → b = a^2 / 2 := 
by
  intros h_surface
  -- sorry is added here to skip the proof step and ensure the code builds successfully.
  sorry

end cube_surface_area_l265_26580


namespace incorrect_statement_trajectory_of_P_l265_26592

noncomputable def midpoint_of_points (x1 x2 y1 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

theorem incorrect_statement_trajectory_of_P (p k x0 y0 : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (hA : A.1 * A.1 + 2 * p * A.2 = 0)
    (hB : B.1 * B.1 + 2 * p * B.2 = 0)
    (hMid : (x0, y0) = midpoint_of_points A.1 B.1 A.2 B.2)
    (hLine : A.2 = k * (A.1 - p / 2))
    (hLineIntersection : B.2 = k * (B.1 - p / 2)) : y0 ^ 2 ≠ 4 * p * (x0 - p / 2) :=
by
  sorry

end incorrect_statement_trajectory_of_P_l265_26592


namespace sqrt_abc_sum_eq_162sqrt2_l265_26504

theorem sqrt_abc_sum_eq_162sqrt2 (a b c : ℝ) (h1 : b + c = 15) (h2 : c + a = 18) (h3 : a + b = 21) :
    Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 :=
by
  sorry

end sqrt_abc_sum_eq_162sqrt2_l265_26504


namespace friend_redistribute_l265_26576

-- Definition and total earnings
def earnings : List Int := [30, 45, 15, 10, 60]
def total_earnings := earnings.sum

-- Number of friends
def number_of_friends : Int := 5

-- Calculate the equal share
def equal_share := total_earnings / number_of_friends

-- Calculate the amount to redistribute by the friend who earned 60
def amount_to_give := 60 - equal_share

theorem friend_redistribute :
  earnings.sum = 160 ∧ equal_share = 32 ∧ amount_to_give = 28 :=
by
  -- Proof goes here, skipped with 'sorry'
  sorry

end friend_redistribute_l265_26576


namespace length_AB_l265_26500

theorem length_AB (x : ℝ) (h1 : 0 < x)
  (hG : G = (0 + 1) / 2)
  (hH : H = (0 + G) / 2)
  (hI : I = (0 + H) / 2)
  (hJ : J = (0 + I) / 2)
  (hAJ : J - 0 = 2) :
  x = 32 := by
  sorry

end length_AB_l265_26500


namespace multiples_of_4_between_200_and_500_l265_26549
-- Import the necessary library

open Nat

theorem multiples_of_4_between_200_and_500 : 
  ∃ n, n = (500 / 4 - 200 / 4) :=
by
  sorry

end multiples_of_4_between_200_and_500_l265_26549


namespace yellow_candles_count_l265_26566

def CalebCandles (grandfather_age : ℕ) (red_candles : ℕ) (blue_candles : ℕ) : ℕ :=
    grandfather_age - (red_candles + blue_candles)

theorem yellow_candles_count :
    CalebCandles 79 14 38 = 27 := by
    sorry

end yellow_candles_count_l265_26566


namespace mike_travel_miles_l265_26574

theorem mike_travel_miles
  (toll_fees_mike : ℝ) (toll_fees_annie : ℝ) (mike_start_fee : ℝ) 
  (annie_start_fee : ℝ) (mike_per_mile : ℝ) (annie_per_mile : ℝ) 
  (annie_travel_time : ℝ) (annie_speed : ℝ) (mike_cost : ℝ) 
  (annie_cost : ℝ) 
  (h_mike_cost_eq : mike_cost = mike_start_fee + toll_fees_mike + mike_per_mile * 36)
  (h_annie_cost_eq : annie_cost = annie_start_fee + toll_fees_annie + annie_per_mile * annie_speed * annie_travel_time)
  (h_equal_costs : mike_cost = annie_cost)
  : 36 = 36 :=
by 
  sorry

end mike_travel_miles_l265_26574


namespace turner_oldest_child_age_l265_26518

theorem turner_oldest_child_age (a b c : ℕ) (avg : ℕ) :
  (a = 6) → (b = 8) → (c = 11) → (avg = 9) → 
  (4 * avg = (a + b + c + x) → x = 11) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end turner_oldest_child_age_l265_26518


namespace emma_withdrew_amount_l265_26598

variable (W : ℝ) -- Variable representing the amount Emma withdrew

theorem emma_withdrew_amount:
  (230 - W + 2 * W = 290) →
  W = 60 :=
by
  sorry

end emma_withdrew_amount_l265_26598


namespace sum_of_three_consecutive_odd_integers_l265_26527

theorem sum_of_three_consecutive_odd_integers (n : ℤ) 
  (h1 : n + (n + 4) = 130) 
  (h2 : n % 2 = 1) : 
  n + (n + 2) + (n + 4) = 195 := 
by
  sorry

end sum_of_three_consecutive_odd_integers_l265_26527


namespace sum_mean_median_mode_l265_26519

theorem sum_mean_median_mode : 
  let data := [2, 5, 1, 5, 2, 6, 1, 5, 0, 2]
  let ordered_data := [0, 1, 1, 2, 2, 2, 5, 5, 5, 6]
  let mean := (0 + 1 + 1 + 2 + 2 + 2 + 5 + 5 + 5 + 6) / 10
  let median := (2 + 2) / 2
  let mode := 5
  mean + median + mode = 9.9 := by
  sorry

end sum_mean_median_mode_l265_26519


namespace simplify_expression_l265_26573

variable (a b c d : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d)

theorem simplify_expression :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
  sorry

end simplify_expression_l265_26573


namespace farmer_goats_l265_26568

theorem farmer_goats (cows sheep goats : ℕ) (extra_goats : ℕ) 
(hcows : cows = 7) (hsheep : sheep = 8) (hgoats : goats = 6) 
(h : (goats + extra_goats = (cows + sheep + goats + extra_goats) / 2)) : 
extra_goats = 9 := by
  sorry

end farmer_goats_l265_26568


namespace value_of_five_minus_c_l265_26534

theorem value_of_five_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 7 + d = 10 + c) :
  5 - c = 6 :=
by
  sorry

end value_of_five_minus_c_l265_26534


namespace sequence_formula_l265_26533

theorem sequence_formula (a : ℕ → ℤ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = -3)
  (h₃ : a 3 = 5)
  (h₄ : a 4 = -7)
  (h₅ : a 5 = 9) :
  ∀ n : ℕ, a n = (-1)^(n+1) * (2 * n - 1) :=
by
  sorry

end sequence_formula_l265_26533


namespace find_roots_of_polynomial_l265_26590

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l265_26590


namespace sin_theta_value_l265_26572

theorem sin_theta_value (θ : ℝ) (h1 : 10 * (Real.tan θ) = 4 * (Real.cos θ)) (h2 : 0 < θ ∧ θ < π) : Real.sin θ = 1/2 :=
by
  sorry

end sin_theta_value_l265_26572


namespace coefficient_condition_l265_26567

theorem coefficient_condition (m : ℝ) (h : m^3 * Nat.choose 6 3 = -160) : m = -2 := sorry

end coefficient_condition_l265_26567


namespace ceil_square_eq_four_l265_26555

theorem ceil_square_eq_four : (⌈(-7 / 4: ℚ)^2⌉ : ℤ) = 4 := by
  sorry

end ceil_square_eq_four_l265_26555


namespace Yoojung_total_vehicles_l265_26593

theorem Yoojung_total_vehicles : 
  let motorcycles := 2
  let bicycles := 5
  motorcycles + bicycles = 7 := 
by
  sorry

end Yoojung_total_vehicles_l265_26593


namespace number_of_possible_values_of_a_l265_26563

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ),
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2040 ∧
  a^2 - b^2 + c^2 - d^2 = 2040 ∧
  508 ∈ {a | ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2040 ∧ a^2 - b^2 + c^2 - d^2 = 2040}

theorem number_of_possible_values_of_a : problem_statement :=
  sorry

end number_of_possible_values_of_a_l265_26563


namespace length_of_bridge_l265_26505

theorem length_of_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (crossing_time_seconds : ℕ)
  (h_train_length : train_length = 125)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_crossing_time_seconds : crossing_time_seconds = 30) :
  ∃ (bridge_length : ℕ), bridge_length = 250 :=
by
  sorry

end length_of_bridge_l265_26505


namespace prime_cube_solution_l265_26512

theorem prime_cube_solution (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : p^3 = p^2 + q^2 + r^2) : 
  p = 3 ∧ q = 3 ∧ r = 3 :=
by
  sorry

end prime_cube_solution_l265_26512


namespace mr_desmond_toys_l265_26539

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l265_26539


namespace find_cost_price_l265_26581

variable (C : ℝ)

def profit_10_percent_selling_price := 1.10 * C

def profit_15_percent_with_150_more := 1.10 * C + 150

def profit_15_percent_selling_price := 1.15 * C

theorem find_cost_price
  (h : profit_15_percent_with_150_more C = profit_15_percent_selling_price C) :
  C = 3000 :=
by
  sorry

end find_cost_price_l265_26581


namespace log24_eq_2b_minus_a_l265_26517

variable (a b : ℝ)

-- given conditions
axiom log6_eq : Real.log 6 = a
axiom log12_eq : Real.log 12 = b

-- proof goal statement
theorem log24_eq_2b_minus_a : Real.log 24 = 2 * b - a :=
by
  sorry

end log24_eq_2b_minus_a_l265_26517


namespace odd_function_iff_l265_26540

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_iff (a b : ℝ) : 
  (∀ x, f x a b = -f (-x) a b) ↔ (a ^ 2 + b ^ 2 = 0) :=
by
  sorry

end odd_function_iff_l265_26540


namespace chipmunk_acorns_l265_26507

theorem chipmunk_acorns :
  ∀ (x y : ℕ), (3 * x = 4 * y) → (y = x - 4) → (3 * x = 48) :=
by
  intros x y h1 h2
  sorry

end chipmunk_acorns_l265_26507


namespace triangle_angles_correct_l265_26546

noncomputable def triangle_angles (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
by sorry

theorem triangle_angles_correct :
  triangle_angles 3 (Real.sqrt 8) (2 + Real.sqrt 2) =
    (67.5, 22.5, 90) :=
by sorry

end triangle_angles_correct_l265_26546
