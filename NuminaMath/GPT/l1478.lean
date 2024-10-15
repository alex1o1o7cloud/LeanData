import Mathlib

namespace NUMINAMATH_GPT_loisa_savings_l1478_147887

namespace SavingsProof

def cost_cash : ℤ := 450
def down_payment : ℤ := 100
def payment_first_4_months : ℤ := 4 * 40
def payment_next_4_months : ℤ := 4 * 35
def payment_last_4_months : ℤ := 4 * 30

def total_installment_payment : ℤ :=
  down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

theorem loisa_savings :
  (total_installment_payment - cost_cash) = 70 := by
  sorry

end SavingsProof

end NUMINAMATH_GPT_loisa_savings_l1478_147887


namespace NUMINAMATH_GPT_number_problem_l1478_147883

theorem number_problem (x : ℤ) (h1 : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := by
  sorry

end NUMINAMATH_GPT_number_problem_l1478_147883


namespace NUMINAMATH_GPT_seven_b_value_l1478_147863

theorem seven_b_value (a b : ℚ) (h₁ : 8 * a + 3 * b = 0) (h₂ : a = b - 3) :
  7 * b = 168 / 11 :=
sorry

end NUMINAMATH_GPT_seven_b_value_l1478_147863


namespace NUMINAMATH_GPT_find_deeper_depth_l1478_147861

noncomputable def swimming_pool_depth_proof 
  (width : ℝ) (length : ℝ) (shallow_depth : ℝ) (volume : ℝ) : Prop :=
  volume = (1 / 2) * (shallow_depth + 4) * width * length

theorem find_deeper_depth
  (h : width = 9)
  (l : length = 12)
  (a : shallow_depth = 1)
  (V : volume = 270) :
  swimming_pool_depth_proof 9 12 1 270 := by
  sorry

end NUMINAMATH_GPT_find_deeper_depth_l1478_147861


namespace NUMINAMATH_GPT_tan_squared_sum_geq_three_over_eight_l1478_147823

theorem tan_squared_sum_geq_three_over_eight 
  (α β γ : ℝ) 
  (hα : 0 ≤ α ∧ α < π / 2) 
  (hβ : 0 ≤ β ∧ β < π / 2) 
  (hγ : 0 ≤ γ ∧ γ < π / 2) 
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := 
sorry

end NUMINAMATH_GPT_tan_squared_sum_geq_three_over_eight_l1478_147823


namespace NUMINAMATH_GPT_increasing_function_greater_at_a_squared_plus_one_l1478_147859

variable (f : ℝ → ℝ) (a : ℝ)

def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_greater_at_a_squared_plus_one :
  strictly_increasing f → f (a^2 + 1) > f a :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_greater_at_a_squared_plus_one_l1478_147859


namespace NUMINAMATH_GPT_saree_original_price_l1478_147821

theorem saree_original_price
  (sale_price : ℝ)
  (P : ℝ)
  (h_discount : sale_price = 0.80 * P * 0.95)
  (h_sale_price : sale_price = 266) :
  P = 350 :=
by
  -- Proof to be completed later
  sorry

end NUMINAMATH_GPT_saree_original_price_l1478_147821


namespace NUMINAMATH_GPT_compare_y_l1478_147808

-- Define the points M and N lie on the graph of y = -5/x
def on_inverse_proportion_curve (x y : ℝ) : Prop :=
  y = -5 / x

-- Main theorem to be proven
theorem compare_y (x1 y1 x2 y2 : ℝ) (h1 : on_inverse_proportion_curve x1 y1) (h2 : on_inverse_proportion_curve x2 y2) (hx : x1 > 0 ∧ x2 < 0) : y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_compare_y_l1478_147808


namespace NUMINAMATH_GPT_parameter_for_three_distinct_solutions_l1478_147842

open Polynomial

theorem parameter_for_three_distinct_solutions (a : ℝ) :
  (∀ x : ℝ, x^4 - 40 * x^2 + 144 = a * (x^2 + 4 * x - 12)) →
  (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  (x1^4 - 40 * x1^2 + 144 = a * (x1^2 + 4 * x1 - 12) ∧ 
   x2^4 - 40 * x2^2 + 144 = a * (x2^2 + 4 * x2 - 12) ∧ 
   x3^4 - 40 * x3^2 + 144 = a * (x3^2 + 4 * x3 - 12) ∧
   x4^4 - 40 * x4^2 + 144 = a * (x4^2 + 4 * x4 - 12))) → a = 48 :=
by
  sorry

end NUMINAMATH_GPT_parameter_for_three_distinct_solutions_l1478_147842


namespace NUMINAMATH_GPT_wire_cut_ratio_l1478_147867

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_ratio_l1478_147867


namespace NUMINAMATH_GPT_correct_choice_l1478_147820

theorem correct_choice
  (options : List String)
  (correct : String)
  (is_correct : correct = "that") :
  "The English spoken in the United States is only slightly different from ____ spoken in England." = 
  "The English spoken in the United States is only slightly different from that spoken in England." :=
by
  sorry

end NUMINAMATH_GPT_correct_choice_l1478_147820


namespace NUMINAMATH_GPT_min_value_of_sum_of_sides_proof_l1478_147836

noncomputable def min_value_of_sum_of_sides (a b c : ℝ) (angleC : ℝ) : ℝ :=
  if (angleC = 60 * (Real.pi / 180)) ∧ ((a + b)^2 - c^2 = 4) then 4 * Real.sqrt 3 / 3 
  else 0

theorem min_value_of_sum_of_sides_proof (a b c : ℝ) (angleC : ℝ) 
  (h1 : angleC = 60 * (Real.pi / 180)) 
  (h2 : (a + b)^2 - c^2 = 4) 
  : min_value_of_sum_of_sides a b c angleC = 4 * Real.sqrt 3 / 3 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_sum_of_sides_proof_l1478_147836


namespace NUMINAMATH_GPT_total_crayons_l1478_147838

-- Define relevant conditions
def crayons_per_child : ℕ := 8
def number_of_children : ℕ := 7

-- Define the Lean statement to prove the total number of crayons
theorem total_crayons : crayons_per_child * number_of_children = 56 :=
by
  sorry

end NUMINAMATH_GPT_total_crayons_l1478_147838


namespace NUMINAMATH_GPT_perfect_square_K_l1478_147899

-- Definitions based on the conditions of the problem
variables (Z K : ℕ)
variables (h1 : 1000 < Z ∧ Z < 5000)
variables (h2 : K > 1)
variables (h3 : Z = K^3)

-- The statement we need to prove
theorem perfect_square_K :
  (∃ K : ℕ, 1000 < K^3 ∧ K^3 < 5000 ∧ K^3 = Z ∧ (∃ a : ℕ, K = a^2)) → K = 16 :=
sorry

end NUMINAMATH_GPT_perfect_square_K_l1478_147899


namespace NUMINAMATH_GPT_major_axis_length_of_intersecting_ellipse_l1478_147857

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end NUMINAMATH_GPT_major_axis_length_of_intersecting_ellipse_l1478_147857


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1478_147835

-- Assume {a_n} is a geometric sequence with positive terms
variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Condition: all terms are positive numbers in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 0 * r ^ n

-- Condition: a_1 * a_9 = 16
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = 16

-- Question to prove: a_2 * a_5 * a_8 = 64
theorem geometric_sequence_problem
  (h_geom : is_geometric_sequence a r)
  (h_pos : ∀ n, 0 < a n)
  (h_cond1 : condition1 a) :
  a 2 * a 5 * a 8 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1478_147835


namespace NUMINAMATH_GPT_team_total_points_l1478_147865

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_team_total_points_l1478_147865


namespace NUMINAMATH_GPT_find_triplets_l1478_147832

theorem find_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧ (2 * y^3 + 1 = 3 * x * y) ∧ (2 * z^3 + 1 = 3 * y * z) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 / 2 ∧ y = -1 / 2 ∧ z = -1 / 2) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_triplets_l1478_147832


namespace NUMINAMATH_GPT_find_m_l1478_147815

def vec (α : Type*) := (α × α)
def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) :
  let a : vec ℝ := (1, 3)
  let b : vec ℝ := (-2, m)
  let c : vec ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  dot_product a c = 0 → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1478_147815


namespace NUMINAMATH_GPT_angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l1478_147827

variable (a b c A B C : ℝ)

-- Condition 1
def cond1 : Prop := b / a = (Real.cos B + 1) / (Real.sqrt 3 * Real.sin A)

-- Condition 2
def cond2 : Prop := 2 * b * Real.sin A = a * Real.tan B

-- Condition 3
def cond3 : Prop := (c - a = b * Real.cos A - a * Real.cos B)

-- Angle B and area of the triangle for Condition 1
theorem angle_B_cond1 (h : cond1 a b A B) : B = π / 3 := sorry

theorem area_range_cond1 (h : cond1 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 2
theorem angle_B_cond2 (h : cond2 a b A B) : B = π / 3 := sorry

theorem area_range_cond2 (h : cond2 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 3
theorem angle_B_cond3 (h : cond3 a b c A B) : B = π / 3 := sorry

theorem area_range_cond3 (h : cond3 a b c A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

end NUMINAMATH_GPT_angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l1478_147827


namespace NUMINAMATH_GPT_smallest_digit_to_correct_sum_l1478_147871

theorem smallest_digit_to_correct_sum (x y z w : ℕ) (hx : x = 753) (hy : y = 946) (hz : z = 821) (hw : w = 2420) :
  ∃ d, d = 7 ∧ (753 + 946 + 821 - 100 * d = 2420) :=
by sorry

end NUMINAMATH_GPT_smallest_digit_to_correct_sum_l1478_147871


namespace NUMINAMATH_GPT_total_amount_divided_l1478_147809

theorem total_amount_divided (A B C : ℝ) (h1 : A = (2/3) * (B + C)) (h2 : B = (2/3) * (A + C)) (h3 : A = 200) :
  A + B + C = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_divided_l1478_147809


namespace NUMINAMATH_GPT_books_leftover_l1478_147819

-- Definitions of the conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def books_bought : ℕ := 26

-- The theorem stating the proof problem
theorem books_leftover : (initial_books + books_bought) - (shelves * books_per_shelf) = 2 := by
  sorry

end NUMINAMATH_GPT_books_leftover_l1478_147819


namespace NUMINAMATH_GPT_line_equation_l1478_147828

theorem line_equation (m : ℝ) (x1 y1 : ℝ) (b : ℝ) :
  m = -3 → x1 = -2 → y1 = 0 → 
  (∀ x y, y - y1 = m * (x - x1) ↔ 3 * x + y + 6 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_l1478_147828


namespace NUMINAMATH_GPT_safe_unlockable_by_five_l1478_147889

def min_total_keys (num_locks : ℕ) (num_people : ℕ) (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) : ℕ :=
  num_locks * ((num_people + 1) / 2)

theorem safe_unlockable_by_five (num_locks : ℕ) (num_people : ℕ) 
  (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) :
  (∀ (P : Finset (Fin num_people)), P.card = 5 → (∀ k : Fin num_locks, ∃ p ∈ P, key_distribution k p)) →
  min_total_keys num_locks num_people key_distribution = 20 := 
by
  sorry

end NUMINAMATH_GPT_safe_unlockable_by_five_l1478_147889


namespace NUMINAMATH_GPT_solve_for_y_l1478_147875

theorem solve_for_y (y : ℝ) (h : -3 * y - 9 = 6 * y + 3) : y = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1478_147875


namespace NUMINAMATH_GPT_race_permutations_l1478_147848

-- Define the number of participants
def num_participants : ℕ := 4

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n + 1) * factorial n

-- Theorem: Given 4 participants, the number of different possible orders they can finish the race is 24.
theorem race_permutations : factorial num_participants = 24 := by
  -- sorry added to skip the proof
  sorry

end NUMINAMATH_GPT_race_permutations_l1478_147848


namespace NUMINAMATH_GPT_streamers_for_price_of_confetti_l1478_147802

variable (p q : ℝ) (x y : ℝ)

theorem streamers_for_price_of_confetti (h1 : x * (1 + p / 100) = y) 
                                   (h2 : y * (1 - q / 100) = x)
                                   (h3 : |p - q| = 90) :
  10 * (y * 0.4) = 4 * y :=
sorry

end NUMINAMATH_GPT_streamers_for_price_of_confetti_l1478_147802


namespace NUMINAMATH_GPT_total_oil_volume_l1478_147878

theorem total_oil_volume (total_bottles : ℕ) (bottles_250ml : ℕ) (bottles_300ml : ℕ)
    (volume_250ml : ℕ) (volume_300ml : ℕ) (total_volume_ml : ℚ) 
    (total_volume_l : ℚ) (h1 : total_bottles = 35)
    (h2 : bottles_250ml = 17) (h3 : bottles_300ml = total_bottles - bottles_250ml)
    (h4 : volume_250ml = 250) (h5 : volume_300ml = 300) 
    (h6 : total_volume_ml = bottles_250ml * volume_250ml + bottles_300ml * volume_300ml)
    (h7 : total_volume_l = total_volume_ml / 1000) : 
    total_volume_l = 9.65 := 
by 
  sorry

end NUMINAMATH_GPT_total_oil_volume_l1478_147878


namespace NUMINAMATH_GPT_optionB_unfactorable_l1478_147873

-- Definitions for the conditions
def optionA (a b : ℝ) : ℝ := -a^2 + b^2
def optionB (x y : ℝ) : ℝ := x^2 + y^2
def optionC (z : ℝ) : ℝ := 49 - z^2
def optionD (m : ℝ) : ℝ := 16 - 25 * m^2

-- The proof statement that option B cannot be factored over the real numbers
theorem optionB_unfactorable (x y : ℝ) : ¬ ∃ (p q : ℝ → ℝ), p x * q y = x^2 + y^2 :=
sorry -- Proof to be filled in

end NUMINAMATH_GPT_optionB_unfactorable_l1478_147873


namespace NUMINAMATH_GPT_mrs_mcpherson_percentage_l1478_147822

def total_rent : ℕ := 1200
def mr_mcpherson_amount : ℕ := 840
def mrs_mcpherson_amount : ℕ := total_rent - mr_mcpherson_amount

theorem mrs_mcpherson_percentage : (mrs_mcpherson_amount.toFloat / total_rent.toFloat) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_mrs_mcpherson_percentage_l1478_147822


namespace NUMINAMATH_GPT_fraction_of_sand_is_one_third_l1478_147801

noncomputable def total_weight : ℝ := 24
noncomputable def weight_of_water (total_weight : ℝ) : ℝ := total_weight / 4
noncomputable def weight_of_gravel : ℝ := 10
noncomputable def weight_of_sand (total_weight weight_of_water weight_of_gravel : ℝ) : ℝ :=
  total_weight - weight_of_water - weight_of_gravel
noncomputable def fraction_of_sand (weight_of_sand total_weight : ℝ) : ℝ :=
  weight_of_sand / total_weight

theorem fraction_of_sand_is_one_third :
  fraction_of_sand (weight_of_sand total_weight (weight_of_water total_weight) weight_of_gravel) total_weight
  = 1/3 := by
  sorry

end NUMINAMATH_GPT_fraction_of_sand_is_one_third_l1478_147801


namespace NUMINAMATH_GPT_some_employees_not_managers_l1478_147846

-- Definitions of the conditions
def isEmployee : Type := sorry
def isManager : isEmployee → Prop := sorry
def isShareholder : isEmployee → Prop := sorry
def isPunctual : isEmployee → Prop := sorry

-- Given conditions
axiom some_employees_not_punctual : ∃ e : isEmployee, ¬isPunctual e
axiom all_managers_punctual : ∀ m : isEmployee, isManager m → isPunctual m
axiom some_managers_shareholders : ∃ m : isEmployee, isManager m ∧ isShareholder m

-- The statement to be proved
theorem some_employees_not_managers : ∃ e : isEmployee, ¬isManager e :=
by sorry

end NUMINAMATH_GPT_some_employees_not_managers_l1478_147846


namespace NUMINAMATH_GPT_unique_symmetric_solutions_l1478_147843

theorem unique_symmetric_solutions (a b α β : ℝ) (h_mul : α * β = a) (h_add : α + β = b) :
  ∀ (x y : ℝ), x * y = a ∧ x + y = b → (x = α ∧ y = β) ∨ (x = β ∧ y = α) :=
by
  sorry

end NUMINAMATH_GPT_unique_symmetric_solutions_l1478_147843


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l1478_147818

theorem arithmetic_sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) :
  S 8 = 30 → S 4 = 7 → 
      (∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) → 
      a 4 = a1 + 3 * d → 
      a 4 = 13 / 4 := by
  intros hS8 hS4 hS_formula ha4_formula
  -- Formal proof to be filled in
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_l1478_147818


namespace NUMINAMATH_GPT_part1_part2_l1478_147834

-- Definitions as per the conditions
def A (a b : ℚ) := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) := - a^2 + (1/2) * a * b + 2 / 3

-- Part (1)
theorem part1 (a b : ℚ) (h1 : a = -1) (h2 : b = -2) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3 := 
by 
  sorry

-- Part (2)
theorem part2 (a : ℚ) : 
  (∀ a : ℚ, 4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3) → 
  b = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1478_147834


namespace NUMINAMATH_GPT_train_speed_l1478_147897

theorem train_speed 
  (length_train : ℝ) (length_bridge : ℝ) (time : ℝ) 
  (h_length_train : length_train = 110)
  (h_length_bridge : length_bridge = 138)
  (h_time : time = 12.399008079353651) : 
  (length_train + length_bridge) / time * 3.6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1478_147897


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt_3_l1478_147880

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_is_sqrt_3 (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt_3_l1478_147880


namespace NUMINAMATH_GPT_max_students_l1478_147807

open Nat

theorem max_students (B G : ℕ) (h1 : 11 * B = 7 * G) (h2 : G = B + 72) (h3 : B + G ≤ 550) : B + G = 324 := by
  sorry

end NUMINAMATH_GPT_max_students_l1478_147807


namespace NUMINAMATH_GPT_solve_quadratic_l1478_147829

theorem solve_quadratic :
  (x = 0 ∨ x = 2/5) ↔ (5 * x^2 - 2 * x = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1478_147829


namespace NUMINAMATH_GPT_toms_total_out_of_pocket_is_680_l1478_147892

namespace HealthCosts

def doctor_visit_cost : ℝ := 300
def cast_cost : ℝ := 200
def initial_insurance_coverage : ℝ := 0.60
def therapy_session_cost : ℝ := 100
def number_of_sessions : ℕ := 8
def therapy_insurance_coverage : ℝ := 0.40

def total_initial_cost : ℝ :=
  doctor_visit_cost + cast_cost

def initial_out_of_pocket : ℝ :=
  total_initial_cost * (1 - initial_insurance_coverage)

def total_therapy_cost : ℝ :=
  therapy_session_cost * number_of_sessions

def therapy_out_of_pocket : ℝ :=
  total_therapy_cost * (1 - therapy_insurance_coverage)

def total_out_of_pocket : ℝ :=
  initial_out_of_pocket + therapy_out_of_pocket

theorem toms_total_out_of_pocket_is_680 :
  total_out_of_pocket = 680 := by
  sorry

end HealthCosts

end NUMINAMATH_GPT_toms_total_out_of_pocket_is_680_l1478_147892


namespace NUMINAMATH_GPT_average_speed_over_ride_l1478_147824

theorem average_speed_over_ride :
  let speed1 := 12 -- speed in km/h
  let time1 := 5 / 60 -- time in hours
  
  let speed2 := 15 -- speed in km/h
  let time2 := 10 / 60 -- time in hours
  
  let speed3 := 18 -- speed in km/h
  let time3 := 15 / 60 -- time in hours
  
  let distance1 := speed1 * time1 -- distance for the first segment
  let distance2 := speed2 * time2 -- distance for the second segment
  let distance3 := speed3 * time3 -- distance for the third segment
  
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  let avg_speed := total_distance / total_time
  
  avg_speed = 16 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_over_ride_l1478_147824


namespace NUMINAMATH_GPT_topology_on_X_l1478_147806

-- Define the universal set X
def X : Set ℕ := {1, 2, 3}

-- Sequences of candidate sets v
def v1 : Set (Set ℕ) := {∅, {1}, {3}, {1, 2, 3}}
def v2 : Set (Set ℕ) := {∅, {2}, {3}, {2, 3}, {1, 2, 3}}
def v3 : Set (Set ℕ) := {∅, {1}, {1, 2}, {1, 3}}
def v4 : Set (Set ℕ) := {∅, {1, 3}, {2, 3}, {3}, {1, 2, 3}}

-- Define the conditions that determine a topology
def isTopology (X : Set ℕ) (v : Set (Set ℕ)) : Prop :=
  X ∈ v ∧ ∅ ∈ v ∧ 
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋃₀ s ∈ v) ∧
  (∀ (s : Set (Set ℕ)), s ⊆ v → ⋂₀ s ∈ v)

-- The statement we want to prove
theorem topology_on_X : 
  isTopology X v2 ∧ isTopology X v4 :=
by
  sorry

end NUMINAMATH_GPT_topology_on_X_l1478_147806


namespace NUMINAMATH_GPT_total_number_of_cards_l1478_147831

/-- There are 9 playing cards and 4 ID cards initially.
If you add 6 more playing cards and 3 more ID cards,
then the total number of playing cards and ID cards will be 22. -/
theorem total_number_of_cards :
  let initial_playing_cards := 9
  let initial_id_cards := 4
  let additional_playing_cards := 6
  let additional_id_cards := 3
  let total_playing_cards := initial_playing_cards + additional_playing_cards
  let total_id_cards := initial_id_cards + additional_id_cards
  let total_cards := total_playing_cards + total_id_cards
  total_cards = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_cards_l1478_147831


namespace NUMINAMATH_GPT_fraction_value_l1478_147876

theorem fraction_value : (3 - (-3)) / (2 - 1) = 6 := 
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1478_147876


namespace NUMINAMATH_GPT_william_wins_10_rounds_l1478_147850

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end NUMINAMATH_GPT_william_wins_10_rounds_l1478_147850


namespace NUMINAMATH_GPT_probability_letter_in_mathematics_l1478_147879

/-- 
Given that Lisa picks one letter randomly from the alphabet, 
prove that the probability that Lisa picks a letter in "MATHEMATICS" is 4/13.
-/
theorem probability_letter_in_mathematics :
  (8 : ℚ) / 26 = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_letter_in_mathematics_l1478_147879


namespace NUMINAMATH_GPT_axis_of_symmetry_l1478_147845

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : y = - 1 / (4 * a) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1478_147845


namespace NUMINAMATH_GPT_total_weight_l1478_147853

def weights (M D C : ℕ): Prop :=
  D = 46 ∧ D + C = 60 ∧ C = M / 5

theorem total_weight (M D C : ℕ) (h : weights M D C) : M + D + C = 130 :=
by
  cases h with
  | intro h1 h2 =>
    cases h2 with
    | intro h2_1 h2_2 => 
      sorry

end NUMINAMATH_GPT_total_weight_l1478_147853


namespace NUMINAMATH_GPT_wrapping_paper_area_l1478_147898

variable (w h : ℝ)

theorem wrapping_paper_area : ∃ A, A = 4 * (w + h) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l1478_147898


namespace NUMINAMATH_GPT_number_of_girls_in_group_l1478_147866

open Finset

/-- Given that a tech group consists of 6 students, and 3 people are to be selected to visit an exhibition,
    if there are at least 1 girl among the selected, the number of different selection methods is 16,
    then the number of girls in the group is 2. -/
theorem number_of_girls_in_group :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 6 ∧ 
            (Nat.choose 6 3 - Nat.choose (6 - n) 3 = 16)) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_group_l1478_147866


namespace NUMINAMATH_GPT_move_point_A_l1478_147814

theorem move_point_A :
  let A := (-5, 6)
  let A_right := (A.1 + 5, A.2)
  let A_upwards := (A_right.1, A_right.2 + 6)
  A_upwards = (0, 12) := by
  sorry

end NUMINAMATH_GPT_move_point_A_l1478_147814


namespace NUMINAMATH_GPT_draw_sequence_count_l1478_147805

noncomputable def total_sequences : ℕ :=
  (Nat.choose 4 3) * (Nat.factorial 4) * 5

theorem draw_sequence_count : total_sequences = 480 := by
  sorry

end NUMINAMATH_GPT_draw_sequence_count_l1478_147805


namespace NUMINAMATH_GPT_kopecks_problem_l1478_147881

theorem kopecks_problem (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b :=
sorry

end NUMINAMATH_GPT_kopecks_problem_l1478_147881


namespace NUMINAMATH_GPT_no_integer_triple_exists_for_10_l1478_147833

theorem no_integer_triple_exists_for_10 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 :=
sorry

end NUMINAMATH_GPT_no_integer_triple_exists_for_10_l1478_147833


namespace NUMINAMATH_GPT_coin_stack_l1478_147893

def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75
def stack_height : ℝ := 14

theorem coin_stack (n_penny n_nickel n_dime n_quarter : ℕ) 
  (h : n_penny * penny_thickness + n_nickel * nickel_thickness + n_dime * dime_thickness + n_quarter * quarter_thickness = stack_height) :
  n_penny + n_nickel + n_dime + n_quarter = 8 :=
sorry

end NUMINAMATH_GPT_coin_stack_l1478_147893


namespace NUMINAMATH_GPT_people_not_in_pool_l1478_147895

noncomputable def total_people_karen_donald : ℕ := 2
noncomputable def children_karen_donald : ℕ := 6
noncomputable def total_people_tom_eva : ℕ := 2
noncomputable def children_tom_eva : ℕ := 4
noncomputable def legs_in_pool : ℕ := 16

theorem people_not_in_pool : total_people_karen_donald + children_karen_donald + total_people_tom_eva + children_tom_eva - (legs_in_pool / 2) = 6 := by
  sorry

end NUMINAMATH_GPT_people_not_in_pool_l1478_147895


namespace NUMINAMATH_GPT_jason_home_distance_l1478_147826

theorem jason_home_distance :
  let v1 := 60 -- speed in miles per hour
  let t1 := 0.5 -- time in hours
  let d1 := v1 * t1 -- distance covered in first part of the journey
  let v2 := 90 -- speed in miles per hour for the second part
  let t2 := 1.0 -- remaining time in hours
  let d2 := v2 * t2 -- distance covered in second part of the journey
  let total_distance := d1 + d2 -- total distance to Jason's home
  total_distance = 120 := 
by
  simp only
  sorry

end NUMINAMATH_GPT_jason_home_distance_l1478_147826


namespace NUMINAMATH_GPT_how_many_pens_l1478_147872

theorem how_many_pens
  (total_cost : ℝ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (avg_pen_price : ℝ)
  (total_cost := 510)
  (num_pencils := 75)
  (avg_pencil_price := 2)
  (avg_pen_price := 12)
  : ∃ (num_pens : ℕ), num_pens = 30 :=
by
  sorry

end NUMINAMATH_GPT_how_many_pens_l1478_147872


namespace NUMINAMATH_GPT_problem_statement_l1478_147810

-- Definitions based on conditions
def position_of_3_in_8_063 := "thousandths"
def representation_of_3_in_8_063 : ℝ := 3 * 0.001
def unit_in_0_48 : ℝ := 0.01

theorem problem_statement :
  (position_of_3_in_8_063 = "thousandths") ∧
  (representation_of_3_in_8_063 = 3 * 0.001) ∧
  (unit_in_0_48 = 0.01) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1478_147810


namespace NUMINAMATH_GPT_minimum_sum_l1478_147854

theorem minimum_sum (a b c : ℕ) (h : a * b * c = 3006) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ 105 :=
sorry

end NUMINAMATH_GPT_minimum_sum_l1478_147854


namespace NUMINAMATH_GPT_negation_of_existence_l1478_147896

theorem negation_of_existence :
  ¬ (∃ (x_0 : ℝ), x_0^2 - x_0 + 1 ≤ 0) ↔ ∀ (x : ℝ), x^2 - x + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1478_147896


namespace NUMINAMATH_GPT_tetrahedron_face_area_squared_l1478_147840

variables {S0 S1 S2 S3 α12 α13 α23 : ℝ}

-- State the theorem
theorem tetrahedron_face_area_squared :
  (S0)^2 = (S1)^2 + (S2)^2 + (S3)^2 - 2 * S1 * S2 * (Real.cos α12) - 2 * S1 * S3 * (Real.cos α13) - 2 * S2 * S3 * (Real.cos α23) :=
sorry

end NUMINAMATH_GPT_tetrahedron_face_area_squared_l1478_147840


namespace NUMINAMATH_GPT_weekly_deficit_is_2800_l1478_147841

def daily_intake (day : String) : ℕ :=
  if day = "Monday" then 2500 else 
  if day = "Tuesday" then 2600 else 
  if day = "Wednesday" then 2400 else 
  if day = "Thursday" then 2700 else 
  if day = "Friday" then 2300 else 
  if day = "Saturday" then 3500 else 
  if day = "Sunday" then 2400 else 0

def daily_expenditure (day : String) : ℕ :=
  if day = "Monday" then 3000 else 
  if day = "Tuesday" then 3200 else 
  if day = "Wednesday" then 2900 else 
  if day = "Thursday" then 3100 else 
  if day = "Friday" then 2800 else 
  if day = "Saturday" then 3000 else 
  if day = "Sunday" then 2700 else 0

def daily_deficit (day : String) : ℤ :=
  daily_expenditure day - daily_intake day

def weekly_caloric_deficit : ℤ :=
  daily_deficit "Monday" +
  daily_deficit "Tuesday" +
  daily_deficit "Wednesday" +
  daily_deficit "Thursday" +
  daily_deficit "Friday" +
  daily_deficit "Saturday" +
  daily_deficit "Sunday"

theorem weekly_deficit_is_2800 : weekly_caloric_deficit = 2800 := by
  sorry

end NUMINAMATH_GPT_weekly_deficit_is_2800_l1478_147841


namespace NUMINAMATH_GPT_craftsman_jars_l1478_147864

theorem craftsman_jars (J P : ℕ) 
  (h1 : J = 2 * P)
  (h2 : 5 * J + 15 * P = 200) : 
  J = 16 := by
  sorry

end NUMINAMATH_GPT_craftsman_jars_l1478_147864


namespace NUMINAMATH_GPT_systematic_sampling_draw_l1478_147839

theorem systematic_sampling_draw
  (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 8)
  (h2 : 160 ≥ 8 * 20)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 → 
    160 ≥ ((k - 1) * 8 + 1 + 7))
  (h4 : ∀ y : ℕ, y = 1 + (15 * 8) → y = 126)
: x = 6 := 
sorry

end NUMINAMATH_GPT_systematic_sampling_draw_l1478_147839


namespace NUMINAMATH_GPT_storybooks_sciencebooks_correct_l1478_147817

-- Given conditions
def total_books : ℕ := 144
def ratio_storybooks_sciencebooks := (7, 5)
def fraction_storybooks := 7 / (7 + 5)
def fraction_sciencebooks := 5 / (7 + 5)

-- Prove the number of storybooks and science books
def number_of_storybooks : ℕ := 84
def number_of_sciencebooks : ℕ := 60

theorem storybooks_sciencebooks_correct :
  (fraction_storybooks * total_books = number_of_storybooks) ∧
  (fraction_sciencebooks * total_books = number_of_sciencebooks) :=
by
  sorry

end NUMINAMATH_GPT_storybooks_sciencebooks_correct_l1478_147817


namespace NUMINAMATH_GPT_mapping_image_l1478_147816

theorem mapping_image (x y l m : ℤ) (h1 : x = 4) (h2 : y = 6) (h3 : l = x + y) (h4 : m = x - y) :
  (l, m) = (10, -2) := by
  sorry

end NUMINAMATH_GPT_mapping_image_l1478_147816


namespace NUMINAMATH_GPT_fraction_shaded_l1478_147811

theorem fraction_shaded (s r : ℝ) (h : s^2 = 3 * r^2) :
    (1/2 * π * r^2) / (1/4 * π * s^2) = 2/3 := 
  sorry

end NUMINAMATH_GPT_fraction_shaded_l1478_147811


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1478_147803

def p (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def q (x a : ℝ) : Prop := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬ p x) ↔ a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1478_147803


namespace NUMINAMATH_GPT_gcd_polynomials_l1478_147849

theorem gcd_polynomials (b : ℤ) (h: ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 :=
sorry

end NUMINAMATH_GPT_gcd_polynomials_l1478_147849


namespace NUMINAMATH_GPT_sum_of_lengths_of_square_sides_l1478_147874

theorem sum_of_lengths_of_square_sides (side_length : ℕ) (h1 : side_length = 9) : 
  (4 * side_length) = 36 :=
by
  -- Here we would normally write the proof
  sorry

end NUMINAMATH_GPT_sum_of_lengths_of_square_sides_l1478_147874


namespace NUMINAMATH_GPT_complex_sum_identity_l1478_147856

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_complex_sum_identity_l1478_147856


namespace NUMINAMATH_GPT_sequence_increasing_or_decreasing_l1478_147800

theorem sequence_increasing_or_decreasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) 
  (hrec : ∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∀ n, x n < x (n + 1) ∨ x n > x (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_increasing_or_decreasing_l1478_147800


namespace NUMINAMATH_GPT_x_intercept_of_translated_line_l1478_147869

theorem x_intercept_of_translated_line :
  let line_translation (y : ℝ) := y + 4
  let new_line_eq := fun (x : ℝ) => 2 * x - 2
  new_line_eq 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_translated_line_l1478_147869


namespace NUMINAMATH_GPT_num_expr_div_by_10_l1478_147886

theorem num_expr_div_by_10 : (11^11 + 12^12 + 13^13) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_num_expr_div_by_10_l1478_147886


namespace NUMINAMATH_GPT_driving_time_equation_l1478_147837

theorem driving_time_equation :
  ∀ (t : ℝ), (60 * t + 90 * (3.5 - t) = 300) :=
by
  intro t
  sorry

end NUMINAMATH_GPT_driving_time_equation_l1478_147837


namespace NUMINAMATH_GPT_phone_call_answered_within_first_four_rings_l1478_147884

def P1 := 0.1
def P2 := 0.3
def P3 := 0.4
def P4 := 0.1

theorem phone_call_answered_within_first_four_rings :
  P1 + P2 + P3 + P4 = 0.9 :=
by
  rw [P1, P2, P3, P4]
  norm_num
  sorry -- Proof step skipped

end NUMINAMATH_GPT_phone_call_answered_within_first_four_rings_l1478_147884


namespace NUMINAMATH_GPT_probability_of_sequential_draws_l1478_147877

theorem probability_of_sequential_draws :
  let total_cards := 52
  let num_fours := 4
  let remaining_after_first_draw := total_cards - 1
  let remaining_after_second_draw := remaining_after_first_draw - 1
  num_fours / total_cards * 1 / remaining_after_first_draw * 1 / remaining_after_second_draw = 1 / 33150 :=
by sorry

end NUMINAMATH_GPT_probability_of_sequential_draws_l1478_147877


namespace NUMINAMATH_GPT_milk_required_for_flour_l1478_147858

theorem milk_required_for_flour (flour_ratio milk_ratio total_flour : ℕ) : 
  (milk_ratio * (total_flour / flour_ratio)) = 160 :=
by
  let milk_ratio := 40
  let flour_ratio := 200
  let total_flour := 800
  exact sorry

end NUMINAMATH_GPT_milk_required_for_flour_l1478_147858


namespace NUMINAMATH_GPT_function_increasing_interval_l1478_147813

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x ^ 2) / Real.log 2

def domain (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem function_increasing_interval : 
  ∀ x, domain x → 0 < x ∧ x < 1 → ∀ y, domain y → 0 < y ∧ y < 1 → x < y → f x < f y :=
by 
  intros x hx h0 y hy h1 hxy
  sorry

end NUMINAMATH_GPT_function_increasing_interval_l1478_147813


namespace NUMINAMATH_GPT_number_of_bushes_l1478_147891

theorem number_of_bushes (T B x y : ℕ) (h1 : B = T - 6) (h2 : x ≥ y + 10) (h3 : T * x = 128) (hT_pos : T > 0) (hx_pos : x > 0) : B = 2 :=
sorry

end NUMINAMATH_GPT_number_of_bushes_l1478_147891


namespace NUMINAMATH_GPT_furniture_cost_final_price_l1478_147844

theorem furniture_cost_final_price 
  (table_cost : ℤ := 140)
  (chair_ratio : ℚ := 1/7)
  (sofa_ratio : ℕ := 2)
  (discount : ℚ := 0.10)
  (tax : ℚ := 0.07)
  (exchange_rate : ℚ := 1.2) :
  let chair_cost := table_cost * chair_ratio
  let sofa_cost := table_cost * sofa_ratio
  let total_cost_before_discount := table_cost + 4 * chair_cost + sofa_cost
  let table_discount := discount * table_cost
  let discounted_table_cost := table_cost - table_discount
  let total_cost_after_discount := discounted_table_cost + 4 * chair_cost + sofa_cost
  let sales_tax := tax * total_cost_after_discount
  let final_cost := total_cost_after_discount + sales_tax
  final_cost = 520.02 
:= sorry

end NUMINAMATH_GPT_furniture_cost_final_price_l1478_147844


namespace NUMINAMATH_GPT_calculate_total_cost_l1478_147885

def cost_of_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def cost_of_non_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def total_cost (p_l1 p_l2 np_l1 np_l2 ppf np_pf : ℕ) : ℕ :=
  cost_of_parallel_sides p_l1 p_l2 ppf + cost_of_non_parallel_sides np_l1 np_l2 np_pf

theorem calculate_total_cost :
  total_cost 25 37 20 24 48 60 = 5616 :=
by
  -- Assuming the conditions are correctly applied, the statement aims to validate that the calculated
  -- sum of the costs for the specified fence sides equal Rs 5616.
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1478_147885


namespace NUMINAMATH_GPT_probability_of_both_contracts_l1478_147847

open Classical

variable (P_A P_B' P_A_or_B P_A_and_B : ℚ)

noncomputable def probability_hardware_contract := P_A = 3 / 4
noncomputable def probability_not_software_contract := P_B' = 5 / 9
noncomputable def probability_either_contract := P_A_or_B = 4 / 5
noncomputable def probability_both_contracts := P_A_and_B = 71 / 180

theorem probability_of_both_contracts {P_A P_B' P_A_or_B P_A_and_B : ℚ} :
  probability_hardware_contract P_A →
  probability_not_software_contract P_B' →
  probability_either_contract P_A_or_B →
  probability_both_contracts P_A_and_B :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_both_contracts_l1478_147847


namespace NUMINAMATH_GPT_mrs_wong_initial_valentines_l1478_147862

theorem mrs_wong_initial_valentines (x : ℕ) (given left : ℕ) (h_given : given = 8) (h_left : left = 22) (h_initial : x = left + given) : x = 30 :=
by
  rw [h_left, h_given] at h_initial
  exact h_initial

end NUMINAMATH_GPT_mrs_wong_initial_valentines_l1478_147862


namespace NUMINAMATH_GPT_sin_150_eq_half_l1478_147812

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sin_150_eq_half_l1478_147812


namespace NUMINAMATH_GPT_actual_estate_area_l1478_147894

theorem actual_estate_area (map_scale : ℝ) (length_inches : ℝ) (width_inches : ℝ) 
  (actual_length : ℝ) (actual_width : ℝ) (area_square_miles : ℝ) 
  (h_scale : map_scale = 300)
  (h_length : length_inches = 4)
  (h_width : width_inches = 3)
  (h_actual_length : actual_length = length_inches * map_scale)
  (h_actual_width : actual_width = width_inches * map_scale)
  (h_area : area_square_miles = actual_length * actual_width) :
  area_square_miles = 1080000 :=
sorry

end NUMINAMATH_GPT_actual_estate_area_l1478_147894


namespace NUMINAMATH_GPT_find_A_l1478_147804

/-- Given that the equation Ax + 10y = 100 has two distinct positive integer solutions, prove that A = 10. -/
theorem find_A (A x1 y1 x2 y2 : ℕ) (h1 : A > 0) (h2 : x1 > 0) (h3 : y1 > 0) 
  (h4 : x2 > 0) (h5 : y2 > 0) (distinct_solutions : x1 ≠ x2 ∧ y1 ≠ y2) 
  (eq1 : A * x1 + 10 * y1 = 100) (eq2 : A * x2 + 10 * y2 = 100) : 
  A = 10 := sorry

end NUMINAMATH_GPT_find_A_l1478_147804


namespace NUMINAMATH_GPT_lcm_proof_l1478_147851

theorem lcm_proof (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) : Nat.lcm b c = 540 :=
sorry

end NUMINAMATH_GPT_lcm_proof_l1478_147851


namespace NUMINAMATH_GPT_find_k_value_l1478_147825

-- Definitions based on conditions
variables {k b x y : ℝ} -- k, b, x, and y are real numbers

-- Conditions given in the problem
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Proposition: Given the conditions, prove that k = 2
theorem find_k_value (h₁ : ∀ x y, y = linear_function k b x → y + 6 = linear_function k b (x + 3)) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1478_147825


namespace NUMINAMATH_GPT_sphere_weight_dependence_l1478_147855

theorem sphere_weight_dependence 
  (r1 r2 SA1 SA2 weight1 weight2 : ℝ) 
  (h1 : r1 = 0.15) 
  (h2 : r2 = 2 * r1) 
  (h3 : SA1 = 4 * Real.pi * r1^2) 
  (h4 : SA2 = 4 * Real.pi * r2^2) 
  (h5 : weight1 = 8) 
  (h6 : weight1 / SA1 = weight2 / SA2) : 
  weight2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_sphere_weight_dependence_l1478_147855


namespace NUMINAMATH_GPT_maddie_milk_usage_l1478_147882

-- Define the constants based on the problem conditions
def cups_per_day : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def bag_cost : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def weekly_coffee_expense : ℝ := 18
def gallon_milk_cost : ℝ := 4

-- Define the proof problem
theorem maddie_milk_usage : 
  (0.5 : ℝ) = (weekly_coffee_expense - 2 * ((cups_per_day * ounces_per_cup * 7) / ounces_per_bag * bag_cost)) / gallon_milk_cost :=
by 
  sorry

end NUMINAMATH_GPT_maddie_milk_usage_l1478_147882


namespace NUMINAMATH_GPT_part1_part2_l1478_147870

variable {a : ℝ} (M N : Set ℝ)

theorem part1 (h : a = 1) : M = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (hM : (M = {x : ℝ | 0 < x ∧ x < a + 1}))
              (hN : N = {x : ℝ | -1 ≤ x ∧ x ≤ 3})
              (h_union : M ∪ N = N) : 
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1478_147870


namespace NUMINAMATH_GPT_find_vector_b_l1478_147890

structure Vec2 where
  x : ℝ
  y : ℝ

def is_parallel (a b : Vec2) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b.x = k * a.x ∧ b.y = k * a.y

def vec_a : Vec2 := { x := 2, y := 3 }
def vec_b : Vec2 := { x := -2, y := -3 }

theorem find_vector_b :
  is_parallel vec_a vec_b := 
sorry

end NUMINAMATH_GPT_find_vector_b_l1478_147890


namespace NUMINAMATH_GPT_latest_departure_time_l1478_147852

noncomputable def minutes_in_an_hour : ℕ := 60
noncomputable def departure_time : ℕ := 20 * minutes_in_an_hour -- 8:00 pm in minutes
noncomputable def checkin_time : ℕ := 2 * minutes_in_an_hour -- 2 hours in minutes
noncomputable def drive_time : ℕ := 45 -- 45 minutes
noncomputable def parking_time : ℕ := 15 -- 15 minutes
noncomputable def total_time_needed : ℕ := checkin_time + drive_time + parking_time -- Total time in minutes

theorem latest_departure_time : departure_time - total_time_needed = 17 * minutes_in_an_hour :=
by
  sorry

end NUMINAMATH_GPT_latest_departure_time_l1478_147852


namespace NUMINAMATH_GPT_find_x_l1478_147860

theorem find_x
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h : a = (Real.sqrt 3, 0))
  (h1 : b = (x, -2))
  (h2 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) :
  x = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l1478_147860


namespace NUMINAMATH_GPT_bicycle_count_l1478_147830

theorem bicycle_count (B T : ℕ) (hT : T = 20) (h_wheels : 2 * B + 3 * T = 160) : B = 50 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_count_l1478_147830


namespace NUMINAMATH_GPT_expression_value_l1478_147868

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z))

theorem expression_value
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod_nonzero : x * y + x * z + y * z ≠ 0) :
  expression x y z = -7 :=
by 
  sorry

end NUMINAMATH_GPT_expression_value_l1478_147868


namespace NUMINAMATH_GPT_area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l1478_147888

-- 1. Prove that the area enclosed by x = π/2, x = 3π/2, y = 0 and y = cos x is 2
theorem area_enclosed_by_lines_and_curve : 
  ∫ (x : ℝ) in (Real.pi / 2)..(3 * Real.pi / 2), (-Real.cos x) = 2 := sorry

-- 2. Prove that the cylindrical coordinates (sqrt(2), π/4, 1) correspond to Cartesian coordinates (1, 1, 1)
theorem cylindrical_to_cartesian_coordinates :
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  let z := 1
  (r * Real.cos θ, r * Real.sin θ, z) = (1, 1, 1) := sorry

-- 3. Prove that (3 + 2i) / (2 - 3i) - (3 - 2i) / (2 + 3i) = 2i
theorem complex_number_evaluation : 
  ((3 + 2 * Complex.I) / (2 - 3 * Complex.I)) - ((3 - 2 * Complex.I) / (2 + 3 * Complex.I)) = 2 * Complex.I := sorry

-- 4. Prove that the area of triangle AOB with given polar coordinates is 2
theorem area_of_triangle_AOB :
  let A := (2, Real.pi / 6)
  let B := (4, Real.pi / 3)
  let area := 1 / 2 * (2 * 4 * Real.sin (Real.pi / 3 - Real.pi / 6))
  area = 2 := sorry

end NUMINAMATH_GPT_area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l1478_147888
