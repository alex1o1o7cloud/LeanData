import Mathlib

namespace NUMINAMATH_GPT_find_n_l1134_113414

theorem find_n (n : ℕ) (M N : ℕ) (hM : M = 4 ^ n) (hN : N = 2 ^ n) (h : M - N = 992) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_l1134_113414


namespace NUMINAMATH_GPT_expand_and_simplify_l1134_113438

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1134_113438


namespace NUMINAMATH_GPT_line_cannot_pass_through_third_quadrant_l1134_113443

theorem line_cannot_pass_through_third_quadrant :
  ∀ (x y : ℝ), x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_GPT_line_cannot_pass_through_third_quadrant_l1134_113443


namespace NUMINAMATH_GPT_payal_book_length_l1134_113487

theorem payal_book_length (P : ℕ) 
  (h1 : (2/3 : ℚ) * P = (1/3 : ℚ) * P + 20) : P = 60 :=
sorry

end NUMINAMATH_GPT_payal_book_length_l1134_113487


namespace NUMINAMATH_GPT_total_spent_l1134_113460

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = 0.90 * B
def condition2 : Prop := B = D + 15

-- Question
theorem total_spent : condition1 B D ∧ condition2 B D → B + D = 285 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_total_spent_l1134_113460


namespace NUMINAMATH_GPT_no_real_b_for_inequality_l1134_113468

theorem no_real_b_for_inequality (b : ℝ) :
  (∃ x : ℝ, |x^2 + 3*b*x + 4*b| ≤ 5 ∧ (∀ y : ℝ, |y^2 + 3*b*y + 4*b| ≤ 5 → y = x)) → false :=
by
  sorry

end NUMINAMATH_GPT_no_real_b_for_inequality_l1134_113468


namespace NUMINAMATH_GPT_rons_height_l1134_113479

variable (R : ℝ)

theorem rons_height
  (depth_eq_16_ron_height : 16 * R = 208) :
  R = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_rons_height_l1134_113479


namespace NUMINAMATH_GPT_ratio_of_final_to_initial_l1134_113459

theorem ratio_of_final_to_initial (P : ℝ) (R : ℝ) (T : ℝ) (hR : R = 0.02) (hT : T = 50) :
  let SI := P * R * T
  let A := P + SI
  A / P = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_final_to_initial_l1134_113459


namespace NUMINAMATH_GPT_trig_identity_cos_add_l1134_113462

open Real

theorem trig_identity_cos_add (x : ℝ) (h1 : sin (π / 3 - x) = 3 / 5) (h2 : π / 2 < x ∧ x < π) :
  cos (x + π / 6) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_cos_add_l1134_113462


namespace NUMINAMATH_GPT_calculate_expression_l1134_113404

theorem calculate_expression (a b c : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1134_113404


namespace NUMINAMATH_GPT_spelling_bee_initial_students_l1134_113467

theorem spelling_bee_initial_students (x : ℕ) 
    (h1 : (2 / 3) * x = 2 / 3 * x)
    (h2 : (3 / 4) * ((1 / 3) * x) = 3 / 4 * (1 / 3 * x))
    (h3 : (1 / 3) * x * (1 / 4) = 30) : 
  x = 120 :=
sorry

end NUMINAMATH_GPT_spelling_bee_initial_students_l1134_113467


namespace NUMINAMATH_GPT_mike_gave_4_marbles_l1134_113497

noncomputable def marbles_given (original_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  original_marbles - remaining_marbles

theorem mike_gave_4_marbles (original_marbles remaining_marbles given_marbles : ℕ) 
  (h1 : original_marbles = 8) (h2 : remaining_marbles = 4) (h3 : given_marbles = marbles_given original_marbles remaining_marbles) : given_marbles = 4 :=
by
  sorry

end NUMINAMATH_GPT_mike_gave_4_marbles_l1134_113497


namespace NUMINAMATH_GPT_sum_of_xyz_l1134_113474

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem sum_of_xyz :
  ∃ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 ∧
  x + y + z = 932 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_xyz_l1134_113474


namespace NUMINAMATH_GPT_lucy_sales_is_43_l1134_113485

def total_packs : Nat := 98
def robyn_packs : Nat := 55
def lucy_packs : Nat := total_packs - robyn_packs

theorem lucy_sales_is_43 : lucy_packs = 43 :=
by
  sorry

end NUMINAMATH_GPT_lucy_sales_is_43_l1134_113485


namespace NUMINAMATH_GPT_calculation_is_one_l1134_113446

noncomputable def calc_expression : ℝ :=
  (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (Real.pi / 3) - Real.sqrt 12

theorem calculation_is_one : calc_expression = 1 :=
by
  -- Each of the steps involved in calculating should match the problem's steps
  -- 1. (1/2)⁻¹ = 2
  -- 2. (2021 + π)^0 = 1
  -- 3. 4 * sin(π / 3) = 2√3 with sin(60°) = √3/2
  -- 4. sqrt(12) = 2√3
  -- Hence 2 - 1 + 2√3 - 2√3 = 1
  sorry

end NUMINAMATH_GPT_calculation_is_one_l1134_113446


namespace NUMINAMATH_GPT_num_valid_k_values_l1134_113465

theorem num_valid_k_values :
  ∃ (s : Finset ℕ), s = { 1, 2, 3, 6, 9, 18 } ∧ s.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_valid_k_values_l1134_113465


namespace NUMINAMATH_GPT_total_steps_correct_l1134_113407

/-- Definition of the initial number of steps on the first day --/
def steps_first_day : Nat := 200 + 300

/-- Definition of the number of steps on the second day --/
def steps_second_day : Nat := (3 / 2) * steps_first_day -- 1.5 is expressed as 3/2

/-- Definition of the number of steps on the third day --/
def steps_third_day : Nat := 2 * steps_second_day

/-- The total number of steps Eliana walked during the three days --/
def total_steps : Nat := steps_first_day + steps_second_day + steps_third_day

theorem total_steps_correct : total_steps = 2750 :=
  by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_total_steps_correct_l1134_113407


namespace NUMINAMATH_GPT_product_of_abc_l1134_113435

variable (a b c m : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 200
def condition2 : Prop := 8 * a = m
def condition3 : Prop := m = b - 10
def condition4 : Prop := m = c + 10

-- The theorem to prove
theorem product_of_abc :
  a + b + c = 200 ∧ 8 * a = m ∧ m = b - 10 ∧ m = c + 10 →
  a * b * c = 505860000 / 4913 :=
by
  sorry

end NUMINAMATH_GPT_product_of_abc_l1134_113435


namespace NUMINAMATH_GPT_smallest_n_factorial_l1134_113477

theorem smallest_n_factorial (a b c m n : ℕ) (h1 : a + b + c = 2020)
(h2 : c > a + 100)
(h3 : m * 10^n = a! * b! * c!)
(h4 : ¬ (10 ∣ m)) : 
  n = 499 :=
sorry

end NUMINAMATH_GPT_smallest_n_factorial_l1134_113477


namespace NUMINAMATH_GPT_not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l1134_113450

noncomputable def f (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem not_bounded_on_neg_infty_zero (a x : ℝ) (h : a = 1) : 
  ¬ ∃ M > 0, ∀ x < 0, |f a x| ≤ M :=
by sorry

-- Problem (2)
theorem range_of_a_bounded_on_zero_infty (a : ℝ) : 
  (∀ x ≥ 0, |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_GPT_not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l1134_113450


namespace NUMINAMATH_GPT_correct_option_l1134_113426

-- Definitions representing the conditions
variable (a b c : Line) -- Define the lines a, b, and c

-- Conditions for the problem
def is_parallel (x y : Line) : Prop := -- Define parallel property
  sorry

def is_perpendicular (x y : Line) : Prop := -- Define perpendicular property
  sorry

noncomputable def proof_statement : Prop :=
  is_parallel a b → is_perpendicular a c → is_perpendicular b c

-- Lean statement of the proof problem
theorem correct_option (h1 : is_parallel a b) (h2 : is_perpendicular a c) : is_perpendicular b c :=
  sorry

end NUMINAMATH_GPT_correct_option_l1134_113426


namespace NUMINAMATH_GPT_sin_sides_of_triangle_l1134_113416

theorem sin_sides_of_triangle {a b c : ℝ} 
  (habc: a + b > c) (hbac: a + c > b) (hcbc: b + c > a) (h_sum: a + b + c ≤ 2 * Real.pi) :
  a > 0 ∧ a < Real.pi ∧ b > 0 ∧ b < Real.pi ∧ c > 0 ∧ c < Real.pi ∧ 
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end NUMINAMATH_GPT_sin_sides_of_triangle_l1134_113416


namespace NUMINAMATH_GPT_total_annual_gain_l1134_113476

theorem total_annual_gain (x : ℝ) 
    (Lakshmi_share : ℝ) 
    (Lakshmi_share_eq: Lakshmi_share = 12000) : 
    (3 * Lakshmi_share = 36000) :=
by
  sorry

end NUMINAMATH_GPT_total_annual_gain_l1134_113476


namespace NUMINAMATH_GPT_modulus_of_z_l1134_113409

open Complex

theorem modulus_of_z (z : ℂ) (h : z * ⟨0, 1⟩ = ⟨2, 1⟩) : abs z = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_modulus_of_z_l1134_113409


namespace NUMINAMATH_GPT_negation_of_proposition_true_l1134_113496

theorem negation_of_proposition_true (a b : ℝ) : 
  (¬ ((a > b) → (∀ c : ℝ, c ^ 2 ≠ 0 → a * c ^ 2 > b * c ^ 2)) = true) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_true_l1134_113496


namespace NUMINAMATH_GPT_height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l1134_113431

-- Definitions
def Height (x : ℕ) : Prop := x = 140
def Weight (x : ℕ) : Prop := x = 23
def BookLength (x : ℕ) : Prop := x = 20
def BookThickness (x : ℕ) : Prop := x = 7
def CargoCapacity (x : ℕ) : Prop := x = 4
def SleepTime (x : ℕ) : Prop := x = 9
def TreeHeight (x : ℕ) : Prop := x = 12

-- Propositions
def XiaohongHeightUnit := "centimeters"
def XiaohongWeightUnit := "kilograms"
def MathBookLengthUnit := "centimeters"
def MathBookThicknessUnit := "millimeters"
def TruckCargoCapacityUnit := "tons"
def ChildrenSleepTimeUnit := "hours"
def BigTreeHeightUnit := "meters"

theorem height_is_centimeters (x : ℕ) (h : Height x) : XiaohongHeightUnit = "centimeters" := sorry
theorem weight_is_kilograms (x : ℕ) (w : Weight x) : XiaohongWeightUnit = "kilograms" := sorry
theorem book_length_is_centimeters (x : ℕ) (l : BookLength x) : MathBookLengthUnit = "centimeters" := sorry
theorem book_thickness_is_millimeters (x : ℕ) (t : BookThickness x) : MathBookThicknessUnit = "millimeters" := sorry
theorem cargo_capacity_is_tons (x : ℕ) (c : CargoCapacity x) : TruckCargoCapacityUnit = "tons" := sorry
theorem sleep_time_is_hours (x : ℕ) (s : SleepTime x) : ChildrenSleepTimeUnit = "hours" := sorry
theorem tree_height_is_meters (x : ℕ) (th : TreeHeight x) : BigTreeHeightUnit = "meters" := sorry

end NUMINAMATH_GPT_height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l1134_113431


namespace NUMINAMATH_GPT_tables_left_l1134_113489

theorem tables_left (original_tables number_of_customers_per_table current_customers : ℝ) 
(h1 : original_tables = 44.0)
(h2 : number_of_customers_per_table = 8.0)
(h3 : current_customers = 256) : 
(original_tables - current_customers / number_of_customers_per_table) = 12.0 :=
by
  sorry

end NUMINAMATH_GPT_tables_left_l1134_113489


namespace NUMINAMATH_GPT_quadratic_sums_l1134_113448

variables {α : Type} [CommRing α] {a b c : α}

theorem quadratic_sums 
  (h₁ : ∀ (a b c : α), a + b ≠ 0 ∧ b + c ≠ 0 ∧ c + a ≠ 0)
  (h₂ : ∀ (r₁ r₂ : α), 
    (r₁^2 + a * r₁ + b = 0 ∧ r₂^2 + b * r₂ + c = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₃ : ∀ (r₁ r₂ : α), 
    (r₁^2 + b * r₁ + c = 0 ∧ r₂^2 + c * r₂ + a = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₄ : ∀ (r₁ r₂ : α), 
    (r₁^2 + c * r₁ + a = 0 ∧ r₂^2 + a * r₂ + b = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0) :
  a^2 + b^2 + c^2 = 18 ∧
  a^2 * b + b^2 * c + c^2 * a = 27 ∧
  a^3 * b^2 + b^3 * c^2 + c^3 * a^2 = -162 :=
sorry

end NUMINAMATH_GPT_quadratic_sums_l1134_113448


namespace NUMINAMATH_GPT_vitamin_supplement_problem_l1134_113429

theorem vitamin_supplement_problem :
  let packA := 7
  let packD := 17
  (∀ n : ℕ, n ≠ 0 → (packA * n = packD * n)) → n = 119 :=
by
  sorry

end NUMINAMATH_GPT_vitamin_supplement_problem_l1134_113429


namespace NUMINAMATH_GPT_rebecca_haircuts_l1134_113482

-- Definitions based on the conditions
def charge_per_haircut : ℕ := 30
def charge_per_perm : ℕ := 40
def charge_per_dye_job : ℕ := 60
def dye_cost_per_job : ℕ := 10
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def tips : ℕ := 50
def total_amount : ℕ := 310

-- Define the unknown number of haircuts scheduled
variable (H : ℕ)

-- Statement of the proof problem
theorem rebecca_haircuts :
  charge_per_haircut * H + charge_per_perm * num_perms + charge_per_dye_job * num_dye_jobs
  - dye_cost_per_job * num_dye_jobs + tips = total_amount → H = 4 :=
by
  sorry

end NUMINAMATH_GPT_rebecca_haircuts_l1134_113482


namespace NUMINAMATH_GPT_common_terms_sequence_l1134_113495

-- Definitions of sequences
def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℤ := 2 ^ n
def c (n : ℕ) : ℤ := 2 ^ (2 * n - 1)

-- Theorem stating the conjecture
theorem common_terms_sequence :
  ∀ n : ℕ, ∃ m : ℕ, a m = b (2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_common_terms_sequence_l1134_113495


namespace NUMINAMATH_GPT_value_of_a3_minus_a2_l1134_113439

theorem value_of_a3_minus_a2 : 
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S n = n^2) ∧ (S 3 - S 2 - (S 2 - S 1)) = 2) :=
sorry

end NUMINAMATH_GPT_value_of_a3_minus_a2_l1134_113439


namespace NUMINAMATH_GPT_california_more_license_plates_l1134_113402

theorem california_more_license_plates :
  let CA_format := 26^4 * 10^2
  let NY_format := 26^3 * 10^3
  CA_format - NY_format = 28121600 := by
  let CA_format : Nat := 26^4 * 10^2
  let NY_format : Nat := 26^3 * 10^3
  have CA_plates : CA_format = 45697600 := by sorry
  have NY_plates : NY_format = 17576000 := by sorry
  calc
    CA_format - NY_format = 45697600 - 17576000 := by rw [CA_plates, NY_plates]
                    _ = 28121600 := by norm_num

end NUMINAMATH_GPT_california_more_license_plates_l1134_113402


namespace NUMINAMATH_GPT_rabbit_parent_genotype_l1134_113455

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end NUMINAMATH_GPT_rabbit_parent_genotype_l1134_113455


namespace NUMINAMATH_GPT_calculation_result_l1134_113498

theorem calculation_result :
  (-1) * (-4) + 2^2 / (7 - 5) = 6 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1134_113498


namespace NUMINAMATH_GPT_two_bacteria_fill_time_l1134_113418

-- Define the conditions
def one_bacterium_fills_bottle_in (a : Nat) (t : Nat) : Prop :=
  (2^t = 2^a)

def two_bacteria_fill_bottle_in (a : Nat) (x : Nat) : Prop :=
  (2 * 2^x = 2^a)

-- State the theorem
theorem two_bacteria_fill_time (a : Nat) : ∃ x, two_bacteria_fill_bottle_in a x ∧ x = a - 1 :=
by
  -- Use the given conditions
  sorry

end NUMINAMATH_GPT_two_bacteria_fill_time_l1134_113418


namespace NUMINAMATH_GPT_maximize_profit_l1134_113427

theorem maximize_profit (x : ℤ) (hx : 20 ≤ x ∧ x ≤ 30) :
  (∀ y, 20 ≤ y ∧ y ≤ 30 → ((y - 20) * (30 - y)) ≤ ((25 - 20) * (30 - 25))) := 
sorry

end NUMINAMATH_GPT_maximize_profit_l1134_113427


namespace NUMINAMATH_GPT_train_speed_l1134_113490

/-
Problem Statement:
Prove that the speed of a train is 26.67 meters per second given:
  1. The length of the train is 320 meters.
  2. The time taken to cross the telegraph post is 12 seconds.
-/

theorem train_speed (distance time : ℝ) (h1 : distance = 320) (h2 : time = 12) :
  (distance / time) = 26.67 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_l1134_113490


namespace NUMINAMATH_GPT_students_play_both_sports_l1134_113415

theorem students_play_both_sports 
  (total_students : ℕ) (students_play_football : ℕ) 
  (students_play_cricket : ℕ) (students_play_neither : ℕ) :
  total_students = 470 → students_play_football = 325 → 
  students_play_cricket = 175 → students_play_neither = 50 → 
  (students_play_football + students_play_cricket - 
    (total_students - students_play_neither)) = 80 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end NUMINAMATH_GPT_students_play_both_sports_l1134_113415


namespace NUMINAMATH_GPT_tangent_point_value_l1134_113420

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end NUMINAMATH_GPT_tangent_point_value_l1134_113420


namespace NUMINAMATH_GPT_part_one_part_two_l1134_113454

-- Defining the sequence {a_n} with the sum of the first n terms.
def S (n : ℕ) : ℕ := 3 * n ^ 2 + 10 * n

-- Defining a_n in terms of the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Defining the arithmetic sequence {b_n}
def b (n : ℕ) : ℕ := 3 * n + 2

-- Defining the sequence {c_n}
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n

-- Defining the sum of the first n terms of {c_n}
def T (n : ℕ) : ℕ :=
  (3 * n + 1) * 2^(n + 2) - 4

-- Theorem to prove general term formula for {b_n}
theorem part_one : ∀ n : ℕ, b n = 3 * n + 2 := 
by sorry

-- Theorem to prove the sum of the first n terms of {c_n}
theorem part_two (n : ℕ) : ∀ n : ℕ, T n = (3 * n + 1) * 2^(n + 2) - 4 :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l1134_113454


namespace NUMINAMATH_GPT_coefficient_of_x8y2_l1134_113410

theorem coefficient_of_x8y2 :
  let term1 := (1 / x^2)
  let term2 := (3 / y)
  let expansion := (x^2 - y)^7
  let coeff1 := 21 * (x ^ 10) * (y ^ 2) * (-1)
  let coeff2 := 35 * (3 / y) * (x ^ 8) * (y ^ 3)
  let comb := coeff1 + coeff2
  comb = -84 * x ^ 8 * y ^ 2 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_x8y2_l1134_113410


namespace NUMINAMATH_GPT_other_root_l1134_113461

theorem other_root (m : ℤ) (h : (∀ x : ℤ, x^2 - x + m = 0 → (x = 2))) : (¬ ∃ y : ℤ, (y^2 - y + m = 0 ∧ y ≠ 2 ∧ y ≠ -1) ) := 
by {
  sorry
}

end NUMINAMATH_GPT_other_root_l1134_113461


namespace NUMINAMATH_GPT_survived_more_than_died_l1134_113419

-- Define the given conditions
def total_trees : ℕ := 13
def trees_died : ℕ := 6
def trees_survived : ℕ := total_trees - trees_died

-- The proof statement
theorem survived_more_than_died :
  trees_survived - trees_died = 1 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_survived_more_than_died_l1134_113419


namespace NUMINAMATH_GPT_number_is_eight_l1134_113481

theorem number_is_eight (x : ℤ) (h : x - 2 = 6) : x = 8 := 
sorry

end NUMINAMATH_GPT_number_is_eight_l1134_113481


namespace NUMINAMATH_GPT_discriminant_quadratic_eqn_l1134_113480

def a := 1
def b := 1
def c := -2
def Δ : ℤ := b^2 - 4 * a * c

theorem discriminant_quadratic_eqn : Δ = 9 := by
  sorry

end NUMINAMATH_GPT_discriminant_quadratic_eqn_l1134_113480


namespace NUMINAMATH_GPT_problem_1_problem_2_l1134_113441

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1134_113441


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1134_113466

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_repeating_decimal_sum_l1134_113466


namespace NUMINAMATH_GPT_c_is_younger_l1134_113423

variables (a b c d : ℕ) -- assuming ages as natural numbers

-- Conditions
axiom cond1 : a + b = b + c + 12
axiom cond2 : b + d = c + d + 8
axiom cond3 : d = a + 5

-- Question
theorem c_is_younger : c = a - 12 :=
sorry

end NUMINAMATH_GPT_c_is_younger_l1134_113423


namespace NUMINAMATH_GPT_shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l1134_113486

noncomputable def k : ℝ := (1 / 20) * Real.log (1 / 4)
noncomputable def b : ℝ := Real.log 160
noncomputable def y (x : ℝ) : ℝ := Real.exp (k * x + b)

theorem shelf_life_at_30_degrees : y 30 = 20 := sorry

theorem temperature_condition_for_shelf_life (x : ℝ) : y x ≥ 80 → x ≤ 10 := sorry

end NUMINAMATH_GPT_shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l1134_113486


namespace NUMINAMATH_GPT_neg_of_if_pos_then_real_roots_l1134_113471

variable (m : ℝ)

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b * x + c = 0

theorem neg_of_if_pos_then_real_roots :
  (∀ m : ℝ, m > 0 → has_real_roots 1 1 (-m) )
  → ( ∀ m : ℝ, m ≤ 0 → ¬ has_real_roots 1 1 (-m) ) := 
sorry

end NUMINAMATH_GPT_neg_of_if_pos_then_real_roots_l1134_113471


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1134_113440

theorem other_root_of_quadratic (m : ℝ) (h : ∃ α : ℝ, α = 1 ∧ (3 * α^2 + m * α = 5)) :
  ∃ β : ℝ, β = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1134_113440


namespace NUMINAMATH_GPT_p_iff_q_l1134_113405

variable (a b : ℝ)

def p := a > 2 ∧ b > 3

def q := a + b > 5 ∧ (a - 2) * (b - 3) > 0

theorem p_iff_q : p a b ↔ q a b := by
  sorry

end NUMINAMATH_GPT_p_iff_q_l1134_113405


namespace NUMINAMATH_GPT_find_rate_l1134_113447

-- Definitions of conditions
def Principal : ℝ := 2500
def Amount : ℝ := 3875
def Time : ℝ := 12

-- Main statement we are proving
theorem find_rate (P : ℝ) (A : ℝ) (T : ℝ) (R : ℝ) 
    (hP : P = Principal) 
    (hA : A = Amount) 
    (hT : T = Time) 
    (hR : R = (A - P) * 100 / (P * T)) : R = 55 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_rate_l1134_113447


namespace NUMINAMATH_GPT_find_f3_value_l1134_113422

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

theorem find_f3_value (a b c : ℝ) (h : f (-3) a b c = 7) : f 3 a b c = -13 := 
by 
  sorry

end NUMINAMATH_GPT_find_f3_value_l1134_113422


namespace NUMINAMATH_GPT_expected_value_coin_flip_l1134_113417

-- Define the conditions
def probability_heads := 2 / 3
def probability_tails := 1 / 3
def gain_heads := 5
def loss_tails := -10

-- Define the expected value calculation
def expected_value := (probability_heads * gain_heads) + (probability_tails * loss_tails)

-- Prove that the expected value is 0.00
theorem expected_value_coin_flip : expected_value = 0 := 
by sorry

end NUMINAMATH_GPT_expected_value_coin_flip_l1134_113417


namespace NUMINAMATH_GPT_min_value_of_function_product_inequality_l1134_113463

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end NUMINAMATH_GPT_min_value_of_function_product_inequality_l1134_113463


namespace NUMINAMATH_GPT_groupD_can_form_triangle_l1134_113472

def groupA := (5, 7, 12)
def groupB := (7, 7, 15)
def groupC := (6, 9, 16)
def groupD := (6, 8, 12)

def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem groupD_can_form_triangle : canFormTriangle 6 8 12 :=
by
  -- Proof of the above theorem will follow the example from the solution.
  sorry

end NUMINAMATH_GPT_groupD_can_form_triangle_l1134_113472


namespace NUMINAMATH_GPT_earnings_difference_l1134_113453

noncomputable def investment_ratio_a : ℕ := 3
noncomputable def investment_ratio_b : ℕ := 4
noncomputable def investment_ratio_c : ℕ := 5

noncomputable def return_ratio_a : ℕ := 6
noncomputable def return_ratio_b : ℕ := 5
noncomputable def return_ratio_c : ℕ := 4

noncomputable def total_earnings : ℕ := 2900

noncomputable def earnings_a (x y : ℕ) : ℚ := (investment_ratio_a * return_ratio_a * x * y) / 100
noncomputable def earnings_b (x y : ℕ) : ℚ := (investment_ratio_b * return_ratio_b * x * y) / 100

theorem earnings_difference (x y : ℕ) (h : (investment_ratio_a * return_ratio_a * x * y + investment_ratio_b * return_ratio_b * x * y + investment_ratio_c * return_ratio_c * x * y) / 100 = total_earnings) :
  earnings_b x y - earnings_a x y = 100 := by
  sorry

end NUMINAMATH_GPT_earnings_difference_l1134_113453


namespace NUMINAMATH_GPT_diana_principal_charge_l1134_113458

theorem diana_principal_charge :
  ∃ P : ℝ, P > 0 ∧ (P + P * 0.06 = 63.6) ∧ P = 60 :=
by
  use 60
  sorry

end NUMINAMATH_GPT_diana_principal_charge_l1134_113458


namespace NUMINAMATH_GPT_train_speed_l1134_113493

/-- A train that crosses a pole in a certain time of 7 seconds and is 210 meters long has a speed of 108 kilometers per hour. -/
theorem train_speed (time_to_cross: ℝ) (length_of_train: ℝ) (speed_kmh : ℝ) 
  (H_time: time_to_cross = 7) (H_length: length_of_train = 210) 
  (conversion_factor: ℝ := 3.6) : speed_kmh = 108 :=
by
  have speed_mps : ℝ := length_of_train / time_to_cross
  have speed_kmh_calc : ℝ := speed_mps * conversion_factor
  sorry

end NUMINAMATH_GPT_train_speed_l1134_113493


namespace NUMINAMATH_GPT_abigail_score_l1134_113483

theorem abigail_score (sum_20 : ℕ) (sum_21 : ℕ) (h1 : sum_20 = 1700) (h2 : sum_21 = 1806) : (sum_21 - sum_20) = 106 :=
by
  sorry

end NUMINAMATH_GPT_abigail_score_l1134_113483


namespace NUMINAMATH_GPT_coeff_x20_greater_in_Q_l1134_113406

noncomputable def coeff (f : ℕ → ℕ → ℤ) (p x : ℤ) : ℤ :=
(x ^ 20) * p

noncomputable def P (x : ℤ) := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℤ) := (1 + x^2 - x^3) ^ 1000

theorem coeff_x20_greater_in_Q :
  coeff 20 (Q x) x > coeff 20 (P x) x :=
  sorry

end NUMINAMATH_GPT_coeff_x20_greater_in_Q_l1134_113406


namespace NUMINAMATH_GPT_cr_inequality_l1134_113432

theorem cr_inequality 
  (a b : ℝ) (r : ℝ)
  (cr : ℝ := if r < 1 then 1 else 2^(r - 1)) 
  (h0 : r ≥ 0) : 
  |a + b|^r ≤ cr * (|a|^r + |b|^r) :=
by 
  sorry

end NUMINAMATH_GPT_cr_inequality_l1134_113432


namespace NUMINAMATH_GPT_find_q_l1134_113412

theorem find_q (p : ℝ) (q : ℝ) (h1 : p ≠ 0) (h2 : p = 4) (h3 : q ≠ 0) (avg_speed_eq : (2 * p * 3) / (p + 3) = 24 / q) : q = 7 := 
 by
  sorry

end NUMINAMATH_GPT_find_q_l1134_113412


namespace NUMINAMATH_GPT_bus_trip_distance_l1134_113400

-- Defining the problem variables
variables (x D : ℝ) -- x: speed in mph, D: total distance in miles

-- Main theorem stating the problem
theorem bus_trip_distance
  (h1 : 0 < x) -- speed of the bus is positive
  (h2 : (2 * x + 3 * (D - 2 * x) / (2 / 3 * x) / 2 + 0.75) - 2 - 4 = 0)
  -- The first scenario summarising the travel and delays
  (h3 : ((2 * x + 120) / x + 3 * (D - (2 * x + 120)) / (2 / 3 * x) / 2 + 0.75) - 3 = 0)
  -- The second scenario summarising the travel and delays; accident 120 miles further down
  : D = 720 := sorry

end NUMINAMATH_GPT_bus_trip_distance_l1134_113400


namespace NUMINAMATH_GPT_infinite_prime_set_exists_l1134_113470

noncomputable def P : Set Nat := {p | Prime p ∧ ∃ m : Nat, p ∣ m^2 + 1}

theorem infinite_prime_set_exists :
  ∃ (P : Set Nat), (∀ p ∈ P, Prime p) ∧ (Set.Infinite P) ∧ 
  (∀ (p : Nat) (hp : p ∈ P) (k : ℕ),
    ∃ (m : Nat), p^k ∣ m^2 + 1 ∧ ¬(p^(k+1) ∣ m^2 + 1)) :=
sorry

end NUMINAMATH_GPT_infinite_prime_set_exists_l1134_113470


namespace NUMINAMATH_GPT_permutation_and_combination_results_l1134_113411

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem permutation_and_combination_results :
  A 5 2 = 20 ∧ C 6 3 + C 6 4 = 35 := by
  sorry

end NUMINAMATH_GPT_permutation_and_combination_results_l1134_113411


namespace NUMINAMATH_GPT_negation_proposition_l1134_113499

theorem negation_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1134_113499


namespace NUMINAMATH_GPT_least_number_remainder_seven_exists_l1134_113449

theorem least_number_remainder_seven_exists :
  ∃ x : ℕ, x ≡ 7 [MOD 11] ∧ x ≡ 7 [MOD 17] ∧ x ≡ 7 [MOD 21] ∧ x ≡ 7 [MOD 29] ∧ x ≡ 7 [MOD 35] ∧ 
           x ≡ 1547 [MOD Nat.lcm 11 (Nat.lcm 17 (Nat.lcm 21 (Nat.lcm 29 35)))] :=
  sorry

end NUMINAMATH_GPT_least_number_remainder_seven_exists_l1134_113449


namespace NUMINAMATH_GPT_find_larger_number_l1134_113452

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1000) 
  (h2 : L = 10 * S + 10) : 
  L = 1110 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1134_113452


namespace NUMINAMATH_GPT_sum_of_ages_l1134_113434

variables (M A : ℕ)

def Maria_age_relation : Prop :=
  M = A + 8

def future_age_relation : Prop :=
  M + 10 = 3 * (A - 6)

theorem sum_of_ages (h₁ : Maria_age_relation M A) (h₂ : future_age_relation M A) : M + A = 44 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1134_113434


namespace NUMINAMATH_GPT_linear_function_passes_through_point_l1134_113457

theorem linear_function_passes_through_point :
  ∀ x y : ℝ, y = -2 * x - 6 → (x = -4 → y = 2) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_passes_through_point_l1134_113457


namespace NUMINAMATH_GPT_product_of_all_possible_N_l1134_113436

theorem product_of_all_possible_N (A B N : ℝ) 
  (h1 : A = B + N)
  (h2 : A - 4 = B + N - 4)
  (h3 : B + 5 = B + 5)
  (h4 : |((B + N - 4) - (B + 5))| = 1) :
  ∃ N₁ N₂ : ℝ, (|N₁ - 9| = 1 ∧ |N₂ - 9| = 1) ∧ N₁ * N₂ = 80 :=
by {
  -- We know the absolute value equation leads to two solutions
  -- hence we will consider N₁ and N₂ such that |N - 9| = 1
  -- which eventually yields N = 10 and N = 8, making their product 80.
  sorry
}

end NUMINAMATH_GPT_product_of_all_possible_N_l1134_113436


namespace NUMINAMATH_GPT_factorization_of_M_l1134_113413

theorem factorization_of_M :
  ∀ (x y z : ℝ), x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = 
  (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end NUMINAMATH_GPT_factorization_of_M_l1134_113413


namespace NUMINAMATH_GPT_union_eq_l1134_113494

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end NUMINAMATH_GPT_union_eq_l1134_113494


namespace NUMINAMATH_GPT_cost_of_chicken_l1134_113442

theorem cost_of_chicken (cost_beef_per_pound : ℝ) (quantity_beef : ℝ) (cost_oil : ℝ) (total_grocery_cost : ℝ) (contribution_each : ℝ) :
  cost_beef_per_pound = 4 →
  quantity_beef = 3 →
  cost_oil = 1 →
  total_grocery_cost = 16 →
  contribution_each = 1 →
  ∃ (cost_chicken : ℝ), cost_chicken = 3 :=
by
  intros h1 h2 h3 h4 h5
  -- This line is required to help Lean handle any math operations
  have h6 := h1
  have h7 := h2
  have h8 := h3
  have h9 := h4
  have h10 := h5
  sorry

end NUMINAMATH_GPT_cost_of_chicken_l1134_113442


namespace NUMINAMATH_GPT_infinite_solutions_b_l1134_113491

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_b_l1134_113491


namespace NUMINAMATH_GPT_consecutive_integers_sum_l1134_113428

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l1134_113428


namespace NUMINAMATH_GPT_cube_painting_probability_l1134_113445

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end NUMINAMATH_GPT_cube_painting_probability_l1134_113445


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1134_113478

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h2 : ∀ c : ℝ, c - a^2 / c = 2 * a) :
  e = 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1134_113478


namespace NUMINAMATH_GPT_derivative_of_y_l1134_113401

variable (x : ℝ)

def y := x^3 + 3 * x^2 + 6 * x - 10

theorem derivative_of_y : (deriv y) x = 3 * x^2 + 6 * x + 6 :=
sorry

end NUMINAMATH_GPT_derivative_of_y_l1134_113401


namespace NUMINAMATH_GPT_books_combination_l1134_113433

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end NUMINAMATH_GPT_books_combination_l1134_113433


namespace NUMINAMATH_GPT_victor_total_money_l1134_113408

def initial_amount : ℕ := 10
def allowance : ℕ := 8
def total_amount : ℕ := initial_amount + allowance

theorem victor_total_money : total_amount = 18 := by
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_victor_total_money_l1134_113408


namespace NUMINAMATH_GPT_total_students_l1134_113437

-- Given definitions
def basketball_count : ℕ := 7
def cricket_count : ℕ := 5
def both_count : ℕ := 3

-- The goal to prove
theorem total_students : basketball_count + cricket_count - both_count = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1134_113437


namespace NUMINAMATH_GPT_range_of_a_l1134_113475

-- Define the input conditions and requirements, and then state the theorem.
def is_acute_angle_cos_inequality (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2

theorem range_of_a (a : ℝ) :
  is_acute_angle_cos_inequality a 1 3 ∧ is_acute_angle_cos_inequality 1 3 a ∧
  is_acute_angle_cos_inequality 3 a 1 ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1134_113475


namespace NUMINAMATH_GPT_savings_fraction_l1134_113484

variable (P : ℝ) -- worker's monthly take-home pay, assumed to be a real number
variable (f : ℝ) -- fraction of the take-home pay that she saves each month, assumed to be a real number

-- Condition: 12 times the fraction saved monthly should equal 8 times the amount not saved monthly.
axiom condition : 12 * f * P = 8 * (1 - f) * P

-- Prove: the fraction saved each month is 2/5
theorem savings_fraction : f = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_savings_fraction_l1134_113484


namespace NUMINAMATH_GPT_probability_two_red_or_blue_correct_l1134_113444

noncomputable def probability_two_red_or_blue_sequential : ℚ := 1 / 5

theorem probability_two_red_or_blue_correct :
  let total_marbles := 15
  let red_blue_marbles := 7
  let first_draw_prob := (7 : ℚ) / 15
  let second_draw_prob := (6 : ℚ) / 14
  first_draw_prob * second_draw_prob = probability_two_red_or_blue_sequential :=
by
  sorry

end NUMINAMATH_GPT_probability_two_red_or_blue_correct_l1134_113444


namespace NUMINAMATH_GPT_smallest_X_divisible_by_60_l1134_113488

/-
  Let \( T \) be a positive integer consisting solely of 0s and 1s.
  If \( X = \frac{T}{60} \) and \( X \) is an integer, prove that the smallest possible value of \( X \) is 185.
-/
theorem smallest_X_divisible_by_60 (T X : ℕ) 
  (hT_digit : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) 
  (h1 : X = T / 60) 
  (h2 : T % 60 = 0) : 
  X = 185 :=
sorry

end NUMINAMATH_GPT_smallest_X_divisible_by_60_l1134_113488


namespace NUMINAMATH_GPT_altitude_from_A_to_BC_l1134_113451

theorem altitude_from_A_to_BC (x y : ℝ) : 
  (3 * x + 4 * y + 12 = 0) ∧ 
  (4 * x - 3 * y + 16 = 0) ∧ 
  (2 * x + y - 2 = 0) → 
  (∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1 / 2) ∧ (b = 2 - 8)) :=
by 
  sorry

end NUMINAMATH_GPT_altitude_from_A_to_BC_l1134_113451


namespace NUMINAMATH_GPT_virus_affected_computers_l1134_113403

theorem virus_affected_computers (m n : ℕ) (h1 : 5 * m + 2 * n = 52) : m = 8 :=
by
  sorry

end NUMINAMATH_GPT_virus_affected_computers_l1134_113403


namespace NUMINAMATH_GPT_combined_cost_is_107_l1134_113430

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end NUMINAMATH_GPT_combined_cost_is_107_l1134_113430


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1134_113492

theorem quadratic_inequality_solution (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c) * x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1134_113492


namespace NUMINAMATH_GPT_four_m_plus_one_2013_eq_neg_one_l1134_113421

theorem four_m_plus_one_2013_eq_neg_one (m : ℝ) (h : |m| = m + 1) : (4 * m + 1) ^ 2013 = -1 := 
sorry

end NUMINAMATH_GPT_four_m_plus_one_2013_eq_neg_one_l1134_113421


namespace NUMINAMATH_GPT_how_many_people_in_group_l1134_113469

-- Definition of the conditions
def ratio_likes_football : ℚ := 24 / 60
def ratio_plays_football_given_likes : ℚ := 1 / 2
def expected_to_play_football : ℕ := 50

-- Combining the ratios to get the fraction of total people playing football
def ratio_plays_football : ℚ := ratio_likes_football * ratio_plays_football_given_likes

-- Total number of people in the group
def total_people_in_group : ℕ := 250

-- Proof statement
theorem how_many_people_in_group (expected_to_play_football : ℕ) : 
  ratio_plays_football * total_people_in_group = expected_to_play_football :=
by {
  -- Directly using our definitions
  sorry
}

end NUMINAMATH_GPT_how_many_people_in_group_l1134_113469


namespace NUMINAMATH_GPT_cost_of_camel_l1134_113424

-- Define the cost of each animal as variables
variables (C H O E : ℝ)

-- Assume the given relationships as hypotheses
def ten_camels_eq_twentyfour_horses := (10 * C = 24 * H)
def sixteens_horses_eq_four_oxen := (16 * H = 4 * O)
def six_oxen_eq_four_elephants := (6 * O = 4 * E)
def ten_elephants_eq_140000 := (10 * E = 140000)

-- The theorem that we want to prove
theorem cost_of_camel (h1 : ten_camels_eq_twentyfour_horses C H)
                      (h2 : sixteens_horses_eq_four_oxen H O)
                      (h3 : six_oxen_eq_four_elephants O E)
                      (h4 : ten_elephants_eq_140000 E) :
  C = 5600 := sorry

end NUMINAMATH_GPT_cost_of_camel_l1134_113424


namespace NUMINAMATH_GPT_second_cat_weight_l1134_113425

theorem second_cat_weight :
  ∀ (w1 w2 w3 w_total : ℕ), 
    w1 = 2 ∧ w3 = 4 ∧ w_total = 13 → 
    w_total = w1 + w2 + w3 → 
    w2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_second_cat_weight_l1134_113425


namespace NUMINAMATH_GPT_min_value_quadratic_l1134_113456

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end NUMINAMATH_GPT_min_value_quadratic_l1134_113456


namespace NUMINAMATH_GPT_number_of_white_balls_l1134_113464

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l1134_113464


namespace NUMINAMATH_GPT_minimize_expr_l1134_113473

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b * c = 8)

-- Define the target expression and the proof goal
def expr := (3 * a + b) * (a + 3 * c) * (2 * b * c + 4)

-- Prove the main statement
theorem minimize_expr : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c = 8) ∧ expr a b c = 384 :=
sorry

end NUMINAMATH_GPT_minimize_expr_l1134_113473
