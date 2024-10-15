import Mathlib

namespace NUMINAMATH_GPT_roots_in_interval_l1098_109858

def P (x : ℝ) : ℝ := x^2014 - 100 * x + 1

theorem roots_in_interval : 
  ∀ x : ℝ, P x = 0 → (1/100) ≤ x ∧ x ≤ 100^(1 / 2013) := 
  sorry

end NUMINAMATH_GPT_roots_in_interval_l1098_109858


namespace NUMINAMATH_GPT_positive_n_for_modulus_eq_l1098_109869

theorem positive_n_for_modulus_eq (n : ℕ) (h_pos : 0 < n) (h_eq : Complex.abs (5 + (n : ℂ) * Complex.I) = 5 * Real.sqrt 26) : n = 25 :=
by
  sorry

end NUMINAMATH_GPT_positive_n_for_modulus_eq_l1098_109869


namespace NUMINAMATH_GPT_total_seeds_eaten_l1098_109809

-- Definitions and conditions
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds
def first_four_players_seeds : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds
def average_seeds : ℕ := first_four_players_seeds / 4
def fifth_player_seeds : ℕ := average_seeds

-- Statement to prove
theorem total_seeds_eaten :
  first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds + fifth_player_seeds = 475 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_seeds_eaten_l1098_109809


namespace NUMINAMATH_GPT_integer_pairs_m_n_l1098_109895

theorem integer_pairs_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (cond1 : ∃ k1 : ℕ, k1 * m = 3 * n ^ 2)
  (cond2 : ∃ k2 : ℕ, k2 ^ 2 = n ^ 2 + m) :
  ∃ a : ℕ, m = 3 * a ^ 2 ∧ n = a :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_m_n_l1098_109895


namespace NUMINAMATH_GPT_sum_constants_l1098_109871

def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

theorem sum_constants (a b c : ℝ) (h : ∀ x : ℝ, -4 * x^2 + 20 * x - 88 = a * (x + b)^2 + c) : 
  a + b + c = -70.5 :=
sorry

end NUMINAMATH_GPT_sum_constants_l1098_109871


namespace NUMINAMATH_GPT_lens_discount_l1098_109834

theorem lens_discount :
  ∃ (P : ℚ), ∀ (D : ℚ),
    (300 - D = 240) →
    (P = (D / 300) * 100) →
    P = 20 :=
by
  sorry

end NUMINAMATH_GPT_lens_discount_l1098_109834


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1098_109873

theorem inscribed_circle_radius (r : ℝ) (radius : ℝ) (angle_deg : ℝ): 
  radius = 6 ∧ angle_deg = 120 ∧ (∀ θ : ℝ, θ = 60) → r = 3 := 
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1098_109873


namespace NUMINAMATH_GPT_total_snowfall_yardley_l1098_109890

theorem total_snowfall_yardley (a b c d : ℝ) (ha : a = 0.12) (hb : b = 0.24) (hc : c = 0.5) (hd : d = 0.36) :
  a + b + c + d = 1.22 :=
by
  sorry

end NUMINAMATH_GPT_total_snowfall_yardley_l1098_109890


namespace NUMINAMATH_GPT_intersection_is_correct_l1098_109804

-- Defining sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Target intersection set
def setIntersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Theorem to be proved
theorem intersection_is_correct : (setA ∩ setB) = setIntersection :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1098_109804


namespace NUMINAMATH_GPT_track_length_l1098_109856

variable {x : ℕ}

-- Conditions
def runs_distance_jacob (x : ℕ) := 120
def runs_distance_liz (x : ℕ) := (x / 2 - 120)

def runs_second_meeting_jacob (x : ℕ) := x + 120 -- Jacob's total distance by second meeting
def runs_second_meeting_liz (x : ℕ) := (x / 2 + 60) -- Liz's total distance by second meeting

-- The relationship is simplified into the final correct answer
theorem track_length (h1 : 120 / (x / 2 - 120) = (x / 2 + 60) / 180) :
  x = 340 := 
sorry

end NUMINAMATH_GPT_track_length_l1098_109856


namespace NUMINAMATH_GPT_odd_function_evaluation_l1098_109857

theorem odd_function_evaluation (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (h : f (-3) = -2) : f 3 + f 0 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_odd_function_evaluation_l1098_109857


namespace NUMINAMATH_GPT_numerator_multiple_of_prime_l1098_109892

theorem numerator_multiple_of_prime (n : ℕ) (hp : Nat.Prime (3 * n + 1)) :
  (2 * n - 1) % (3 * n + 1) = 0 :=
sorry

end NUMINAMATH_GPT_numerator_multiple_of_prime_l1098_109892


namespace NUMINAMATH_GPT_minimum_width_l1098_109847

theorem minimum_width (w : ℝ) (h_area : w * (w + 15) ≥ 200) : w ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_minimum_width_l1098_109847


namespace NUMINAMATH_GPT_toy_factory_max_profit_l1098_109842

theorem toy_factory_max_profit :
  ∃ x y : ℕ,    -- x: number of bears, y: number of cats
  15 * x + 10 * y ≤ 450 ∧    -- labor hours constraint
  20 * x + 5 * y ≤ 400 ∧     -- raw materials constraint
  80 * x + 45 * y = 2200 :=  -- total selling price
by
  sorry

end NUMINAMATH_GPT_toy_factory_max_profit_l1098_109842


namespace NUMINAMATH_GPT_perimeter_of_square_l1098_109845

theorem perimeter_of_square (s : ℝ) (h : s^2 = s * Real.sqrt 2) (h_ne_zero : s ≠ 0) :
    4 * s = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l1098_109845


namespace NUMINAMATH_GPT_solve_for_a_l1098_109819

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem solve_for_a : 
  (∃ a : ℝ, f (f 0 a) a = 4 * a) → (a = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1098_109819


namespace NUMINAMATH_GPT_reverse_digits_multiplication_l1098_109810

theorem reverse_digits_multiplication (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) : 
  (10 * a + b) * (10 * b + a) = 101 * a * b + 10 * (a^2 + b^2) :=
by 
  sorry

end NUMINAMATH_GPT_reverse_digits_multiplication_l1098_109810


namespace NUMINAMATH_GPT_generalized_inequality_l1098_109898

theorem generalized_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) : x^n + (n : ℝ) / x > n + 1 := 
sorry

end NUMINAMATH_GPT_generalized_inequality_l1098_109898


namespace NUMINAMATH_GPT_union_complement_l1098_109872

open Set Real

def P : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | x^2 - 4 < 0 }

theorem union_complement :
  P ∪ (compl Q) = (Iic (-2)) ∪ Ici 1 :=
by
  sorry

end NUMINAMATH_GPT_union_complement_l1098_109872


namespace NUMINAMATH_GPT_action_figure_price_l1098_109891

theorem action_figure_price (x : ℝ) (h1 : 2 + 4 * x = 30) : x = 7 :=
by
  -- The proof is provided here
  sorry

end NUMINAMATH_GPT_action_figure_price_l1098_109891


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l1098_109806

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l1098_109806


namespace NUMINAMATH_GPT_repeatable_transformation_l1098_109854

theorem repeatable_transformation (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  (2 * c > a + b) ∧ (2 * a > b + c) ∧ (2 * b > c + a) := 
sorry

end NUMINAMATH_GPT_repeatable_transformation_l1098_109854


namespace NUMINAMATH_GPT_find_a_minus_b_l1098_109818

theorem find_a_minus_b
  (a b : ℝ)
  (f g h h_inv : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = -4 * x + 3)
  (hh : ∀ x, h x = f (g x))
  (hinv : ∀ x, h_inv x = 2 * x + 6)
  (h_comp : ∀ x, h x = (x - 6) / 2) :
  a - b = 5 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l1098_109818


namespace NUMINAMATH_GPT_solve_for_x_l1098_109800

theorem solve_for_x : ∃ x : ℤ, 25 - 7 = 3 + x ∧ x = 15 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1098_109800


namespace NUMINAMATH_GPT_largest_possible_perimeter_l1098_109815

theorem largest_possible_perimeter :
  ∃ (l w : ℕ), 8 * l + 8 * w = l * w - 1 ∧ 2 * l + 2 * w = 164 :=
sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l1098_109815


namespace NUMINAMATH_GPT_sum_of_abs_arithmetic_sequence_l1098_109878

theorem sum_of_abs_arithmetic_sequence {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} 
  (hS3 : S_n 3 = 21) (hS9 : S_n 9 = 9) :
  ∃ (T_n : ℕ → ℤ), 
    (∀ (n : ℕ), n ≤ 5 → T_n n = -n^2 + 10 * n) ∧
    (∀ (n : ℕ), n ≥ 6 → T_n n = n^2 - 10 * n + 50) :=
sorry

end NUMINAMATH_GPT_sum_of_abs_arithmetic_sequence_l1098_109878


namespace NUMINAMATH_GPT_converse_inverse_l1098_109801

-- Define the properties
def is_parallelogram (polygon : Type) : Prop := sorry -- needs definitions about polygons
def has_two_pairs_of_parallel_sides (polygon : Type) : Prop := sorry -- needs definitions about polygons

-- The given condition
axiom parallelogram_implies_parallel_sides (polygon : Type) :
  is_parallelogram polygon → has_two_pairs_of_parallel_sides polygon

-- Proof of the converse:
theorem converse (polygon : Type) :
  has_two_pairs_of_parallel_sides polygon → is_parallelogram polygon := sorry

-- Proof of the inverse:
theorem inverse (polygon : Type) :
  ¬is_parallelogram polygon → ¬has_two_pairs_of_parallel_sides polygon := sorry

end NUMINAMATH_GPT_converse_inverse_l1098_109801


namespace NUMINAMATH_GPT_find_missing_number_l1098_109851

-- Define the known values
def numbers : List ℕ := [1, 22, 24, 25, 26, 27, 2]
def specified_mean : ℕ := 20
def total_counts : ℕ := 8

-- The theorem statement
theorem find_missing_number : (∀ (x : ℕ), (List.sum (x :: numbers) = specified_mean * total_counts) → x = 33) :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1098_109851


namespace NUMINAMATH_GPT_Person3IsTriussian_l1098_109820

def IsTriussian (person : ℕ) : Prop := if person = 3 then True else False

def Person1Statement : Prop := ∀ i j k : ℕ, i = 1 → j = 2 → k = 3 → (IsTriussian i = (IsTriussian j ∧ IsTriussian k) ∨ (¬IsTriussian j ∧ ¬IsTriussian k))

def Person2Statement : Prop := ∀ i j : ℕ, i = 2 → j = 3 → (IsTriussian j = False)

def Person3Statement : Prop := ∀ i j : ℕ, i = 3 → j = 1 → (IsTriussian j = False)

theorem Person3IsTriussian : (Person1Statement ∧ Person2Statement ∧ Person3Statement) → IsTriussian 3 :=
by 
  sorry

end NUMINAMATH_GPT_Person3IsTriussian_l1098_109820


namespace NUMINAMATH_GPT_exposed_surface_area_equals_42_l1098_109883

-- Define the structure and exposed surface area calculations.
def surface_area_of_sculpture (layers : List Nat) : Nat :=
  (layers.headD 0 * 5) +  -- Top layer (5 faces exposed)
  (layers.getD 1 0 * 3 + layers.getD 1 0) +  -- Second layer
  (layers.getD 2 0 * 1 + layers.getD 2 0) +  -- Third layer
  (layers.getD 3 0 * 1) -- Bottom layer

-- Define the conditions
def number_of_layers : List Nat := [1, 4, 9, 6]

-- State the theorem
theorem exposed_surface_area_equals_42 :
  surface_area_of_sculpture number_of_layers = 42 :=
by
  sorry

end NUMINAMATH_GPT_exposed_surface_area_equals_42_l1098_109883


namespace NUMINAMATH_GPT_least_number_div_condition_l1098_109897

theorem least_number_div_condition (m : ℕ) : 
  (∃ k r : ℕ, m = 34 * k + r ∧ m = 5 * (r + 8) ∧ r < 34) → m = 162 := 
by
  sorry

end NUMINAMATH_GPT_least_number_div_condition_l1098_109897


namespace NUMINAMATH_GPT_intersection_complement_l1098_109877

open Set

/-- The universal set U as the set of all real numbers -/
def U : Set ℝ := @univ ℝ

/-- The set M -/
def M : Set ℝ := {-1, 0, 1}

/-- The set N defined by the equation x^2 + x = 0 -/
def N : Set ℝ := {x | x^2 + x = 0}

/-- The complement of set N in the universal set U -/
def C_U_N : Set ℝ := {x ∈ U | x ≠ -1 ∧ x ≠ 0}

theorem intersection_complement :
  M ∩ C_U_N = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1098_109877


namespace NUMINAMATH_GPT_work_hours_l1098_109861

-- Let h be the number of hours worked
def hours_worked (total_paid part_cost hourly_rate : ℕ) : ℕ :=
  (total_paid - part_cost) / hourly_rate

-- Given conditions
def total_paid : ℕ := 300
def part_cost : ℕ := 150
def hourly_rate : ℕ := 75

-- The statement to be proved
theorem work_hours :
  hours_worked total_paid part_cost hourly_rate = 2 :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_work_hours_l1098_109861


namespace NUMINAMATH_GPT_people_left_after_first_stop_l1098_109850

def initial_people_on_train : ℕ := 48
def people_got_off_train : ℕ := 17

theorem people_left_after_first_stop : (initial_people_on_train - people_got_off_train) = 31 := by
  sorry

end NUMINAMATH_GPT_people_left_after_first_stop_l1098_109850


namespace NUMINAMATH_GPT_chinese_carriage_problem_l1098_109846

theorem chinese_carriage_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) :=
sorry

end NUMINAMATH_GPT_chinese_carriage_problem_l1098_109846


namespace NUMINAMATH_GPT_smaller_number_l1098_109816

theorem smaller_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : y = 15 :=
sorry

end NUMINAMATH_GPT_smaller_number_l1098_109816


namespace NUMINAMATH_GPT_orchids_to_roses_ratio_l1098_109888

noncomputable def total_centerpieces : ℕ := 6
noncomputable def roses_per_centerpiece : ℕ := 8
noncomputable def lilies_per_centerpiece : ℕ := 6
noncomputable def total_budget : ℕ := 2700
noncomputable def cost_per_flower : ℕ := 15
noncomputable def total_flowers : ℕ := total_budget / cost_per_flower

noncomputable def total_roses : ℕ := total_centerpieces * roses_per_centerpiece
noncomputable def total_lilies : ℕ := total_centerpieces * lilies_per_centerpiece
noncomputable def total_roses_and_lilies : ℕ := total_roses + total_lilies
noncomputable def total_orchids : ℕ := total_flowers - total_roses_and_lilies
noncomputable def orchids_per_centerpiece : ℕ := total_orchids / total_centerpieces

theorem orchids_to_roses_ratio : orchids_per_centerpiece / roses_per_centerpiece = 2 :=
by
  sorry

end NUMINAMATH_GPT_orchids_to_roses_ratio_l1098_109888


namespace NUMINAMATH_GPT_pattern_equation_l1098_109802

theorem pattern_equation (n : ℕ) (hn : n > 0) : n * (n + 2) + 1 = (n + 1) ^ 2 := 
by sorry

end NUMINAMATH_GPT_pattern_equation_l1098_109802


namespace NUMINAMATH_GPT_inradius_circumradius_le_height_l1098_109870

theorem inradius_circumradius_le_height
    {α β γ : ℝ}
    (hα : 0 < α ∧ α ≤ 90)
    (hβ : 0 < β ∧ β ≤ 90)
    (hγ : 0 < γ ∧ γ ≤ 90)
    (α_ge_β : α ≥ β)
    (β_ge_γ : β ≥ γ)
    {r R h : ℝ} :
  r + R ≤ h := 
sorry

end NUMINAMATH_GPT_inradius_circumradius_le_height_l1098_109870


namespace NUMINAMATH_GPT_find_p_q_l1098_109874

theorem find_p_q (p q : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 + p * x + q)
  (h_min : ∀ x, x = q → f x = (p + q)^2) : 
  (p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l1098_109874


namespace NUMINAMATH_GPT_perpendicular_lines_slope_eq_l1098_109899

theorem perpendicular_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
               2 * x + m * y - 6 = 0 → 
               (1 / 2) * (-2 / m) = -1) →
  m = 1 := 
by sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_eq_l1098_109899


namespace NUMINAMATH_GPT_parallel_lines_l1098_109849

theorem parallel_lines (m : ℝ) 
  (h : 3 * (m - 2) + m * (m + 2) = 0) 
  : m = 1 ∨ m = -6 := 
by 
  sorry

end NUMINAMATH_GPT_parallel_lines_l1098_109849


namespace NUMINAMATH_GPT_toothpicks_in_300th_stage_l1098_109826

/-- 
Prove that the number of toothpicks needed for the 300th stage is 1201, given:
1. The first stage has 5 toothpicks.
2. Each subsequent stage adds 4 toothpicks to the previous stage.
-/
theorem toothpicks_in_300th_stage :
  let a_1 := 5
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 300 = 1201 := by
  sorry

end NUMINAMATH_GPT_toothpicks_in_300th_stage_l1098_109826


namespace NUMINAMATH_GPT_weight_of_sparrow_l1098_109852

variable (a b : ℝ)

-- Define the conditions as Lean statements
-- 1. Six sparrows and seven swallows are balanced
def balanced_initial : Prop :=
  6 * b = 7 * a

-- 2. Sparrows are heavier than swallows
def sparrows_heavier : Prop :=
  b > a

-- 3. If one sparrow and one swallow are exchanged, the balance is maintained
def balanced_after_exchange : Prop :=
  5 * b + a = 6 * a + b

-- The theorem to prove the weight of one sparrow in terms of the weight of one swallow
theorem weight_of_sparrow (h1 : balanced_initial a b) (h2 : sparrows_heavier a b) (h3 : balanced_after_exchange a b) : 
  b = (5 / 4) * a :=
sorry

end NUMINAMATH_GPT_weight_of_sparrow_l1098_109852


namespace NUMINAMATH_GPT_production_rate_l1098_109848

theorem production_rate (minutes: ℕ) (machines1 machines2 paperclips1 paperclips2 : ℕ)
  (h1 : minutes = 1) (h2 : machines1 = 8) (h3 : machines2 = 18) (h4 : paperclips1 = 560) 
  (h5 : paperclips2 = (paperclips1 / machines1) * machines2 * minutes) : 
  paperclips2 = 7560 :=
by
  sorry

end NUMINAMATH_GPT_production_rate_l1098_109848


namespace NUMINAMATH_GPT_greatest_b_value_l1098_109811

def equation_has_integer_solutions (b : ℕ) : Prop :=
  ∃ (x : ℤ), x * (x + b) = -20

theorem greatest_b_value : ∃ (b : ℕ), b = 21 ∧ equation_has_integer_solutions b :=
by
  sorry

end NUMINAMATH_GPT_greatest_b_value_l1098_109811


namespace NUMINAMATH_GPT_puppies_in_each_cage_l1098_109813

theorem puppies_in_each_cage (initial_puppies sold_puppies cages : ℕ)
  (h_initial : initial_puppies = 18)
  (h_sold : sold_puppies = 3)
  (h_cages : cages = 3) :
  (initial_puppies - sold_puppies) / cages = 5 :=
by
  sorry

end NUMINAMATH_GPT_puppies_in_each_cage_l1098_109813


namespace NUMINAMATH_GPT_proof_C_ST_l1098_109840

-- Definitions for sets and their operations
def A1 : Set ℕ := {0, 1}
def A2 : Set ℕ := {1, 2}
def S : Set ℕ := A1 ∪ A2
def T : Set ℕ := A1 ∩ A2
def C_ST : Set ℕ := S \ T

theorem proof_C_ST : 
  C_ST = {0, 2} := 
by 
  sorry

end NUMINAMATH_GPT_proof_C_ST_l1098_109840


namespace NUMINAMATH_GPT_seashell_count_l1098_109831

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end NUMINAMATH_GPT_seashell_count_l1098_109831


namespace NUMINAMATH_GPT_eq_satisfied_for_all_y_l1098_109889

theorem eq_satisfied_for_all_y (x : ℝ) : 
  (∀ y: ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eq_satisfied_for_all_y_l1098_109889


namespace NUMINAMATH_GPT_relationship_of_y_values_l1098_109814

def parabola_y (x : ℝ) (c : ℝ) : ℝ :=
  2 * (x + 1)^2 + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  y1 = parabola_y (-2) c →
  y2 = parabola_y 1 c →
  y3 = parabola_y 2 c →
  y3 > y2 ∧ y2 > y1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l1098_109814


namespace NUMINAMATH_GPT_length_of_brick_proof_l1098_109886

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end NUMINAMATH_GPT_length_of_brick_proof_l1098_109886


namespace NUMINAMATH_GPT_min_value_x_plus_one_over_x_plus_two_l1098_109876

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1 / (x + 2) ∧ y ≥ 0 := 
sorry

end NUMINAMATH_GPT_min_value_x_plus_one_over_x_plus_two_l1098_109876


namespace NUMINAMATH_GPT_percentage_not_drop_l1098_109866

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end NUMINAMATH_GPT_percentage_not_drop_l1098_109866


namespace NUMINAMATH_GPT_total_Pokemon_cards_l1098_109887

def j : Nat := 6
def o : Nat := j + 2
def r : Nat := 3 * o
def t : Nat := j + o + r

theorem total_Pokemon_cards : t = 38 := by 
  sorry

end NUMINAMATH_GPT_total_Pokemon_cards_l1098_109887


namespace NUMINAMATH_GPT_mark_weekly_leftover_l1098_109879

def initial_hourly_wage := 40
def raise_percentage := 5 / 100
def daily_hours := 8
def weekly_days := 5
def old_weekly_bills := 600
def personal_trainer_cost := 100

def new_hourly_wage := initial_hourly_wage * (1 + raise_percentage)
def weekly_hours := daily_hours * weekly_days
def weekly_earnings := new_hourly_wage * weekly_hours
def new_weekly_expenses := old_weekly_bills + personal_trainer_cost
def leftover_per_week := weekly_earnings - new_weekly_expenses

theorem mark_weekly_leftover : leftover_per_week = 980 := by
  sorry

end NUMINAMATH_GPT_mark_weekly_leftover_l1098_109879


namespace NUMINAMATH_GPT_product_of_real_roots_l1098_109855

theorem product_of_real_roots (x1 x2 : ℝ) (h1 : x1^2 - 6 * x1 + 8 = 0) (h2 : x2^2 - 6 * x2 + 8 = 0) :
  x1 * x2 = 8 := 
sorry

end NUMINAMATH_GPT_product_of_real_roots_l1098_109855


namespace NUMINAMATH_GPT_cost_of_student_ticket_l1098_109803

theorem cost_of_student_ticket
  (cost_adult : ℤ)
  (total_tickets : ℤ)
  (total_revenue : ℤ)
  (adult_tickets : ℤ)
  (student_tickets : ℤ)
  (H1 : cost_adult = 6)
  (H2 : total_tickets = 846)
  (H3 : total_revenue = 3846)
  (H4 : adult_tickets = 410)
  (H5 : student_tickets = 436)
  : (total_revenue = adult_tickets * cost_adult + student_tickets * (318 / 100)) :=
by
  -- mathematical proof steps would go here
  sorry

end NUMINAMATH_GPT_cost_of_student_ticket_l1098_109803


namespace NUMINAMATH_GPT_john_safety_percentage_l1098_109843

def bench_max_weight : ℕ := 1000
def john_weight : ℕ := 250
def weight_on_bar : ℕ := 550
def total_weight := john_weight + weight_on_bar
def percentage_of_max_weight := (total_weight * 100) / bench_max_weight
def percentage_under_max_weight := 100 - percentage_of_max_weight

theorem john_safety_percentage : percentage_under_max_weight = 20 := by
  sorry

end NUMINAMATH_GPT_john_safety_percentage_l1098_109843


namespace NUMINAMATH_GPT_directrix_of_parabola_l1098_109880

theorem directrix_of_parabola (y x : ℝ) (h : y = 4 * x^2) : y = - (1 / 16) :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1098_109880


namespace NUMINAMATH_GPT_new_parabola_after_shift_l1098_109885

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the transformation functions for shifting the parabola
def shift_left (x : ℝ) (shift : ℝ) : ℝ := x + shift
def shift_down (y : ℝ) (shift : ℝ) : ℝ := y - shift

-- Prove the transformation yields the correct new parabola equation
theorem new_parabola_after_shift : 
  (∀ x : ℝ, (shift_down (original_parabola (shift_left x 2)) 3) = (x + 2)^2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_new_parabola_after_shift_l1098_109885


namespace NUMINAMATH_GPT_increasing_function_condition_l1098_109864

theorem increasing_function_condition (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 6) * x1 + (2 * k + 1) < (2 * k - 6) * x2 + (2 * k + 1)) ↔ (k > 3) :=
by
  -- To prove the statement, we would need to prove it in both directions.
  sorry

end NUMINAMATH_GPT_increasing_function_condition_l1098_109864


namespace NUMINAMATH_GPT_no_int_solutions_for_cubic_eqn_l1098_109853

theorem no_int_solutions_for_cubic_eqn :
  ¬ ∃ (m n : ℤ), m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end NUMINAMATH_GPT_no_int_solutions_for_cubic_eqn_l1098_109853


namespace NUMINAMATH_GPT_a_2_pow_100_value_l1098_109828

theorem a_2_pow_100_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (2 * n) = 3 * n * a n) :
  a (2^100) = 2^4852 * 3^4950 :=
by
  sorry

end NUMINAMATH_GPT_a_2_pow_100_value_l1098_109828


namespace NUMINAMATH_GPT_part_one_part_two_part_three_l1098_109868

def numberOfWaysToPlaceBallsInBoxes : ℕ :=
  4 ^ 4

def numberOfWaysOneBoxEmpty : ℕ :=
  Nat.choose 4 2 * (Nat.factorial 4 / Nat.factorial 1)

def numberOfWaysTwoBoxesEmpty : ℕ :=
  (Nat.choose 4 1 * (Nat.factorial 4 / Nat.factorial 2)) + (Nat.choose 4 2 * (Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)))

theorem part_one : numberOfWaysToPlaceBallsInBoxes = 256 := by
  sorry

theorem part_two : numberOfWaysOneBoxEmpty = 144 := by
  sorry

theorem part_three : numberOfWaysTwoBoxesEmpty = 120 := by
  sorry

end NUMINAMATH_GPT_part_one_part_two_part_three_l1098_109868


namespace NUMINAMATH_GPT_balls_sold_eq_13_l1098_109812

-- Let SP be the selling price, CP be the cost price per ball, and loss be the loss incurred.
def SP : ℕ := 720
def CP : ℕ := 90
def loss : ℕ := 5 * CP
def total_CP (n : ℕ) : ℕ := n * CP

-- Given the conditions:
axiom loss_eq : loss = 5 * CP
axiom ball_CP_value : CP = 90
axiom selling_price_value : SP = 720

-- Loss is defined as total cost price minus selling price
def calculated_loss (n : ℕ) : ℕ := total_CP n - SP

-- The proof statement:
theorem balls_sold_eq_13 (n : ℕ) (h1 : calculated_loss n = loss) : n = 13 :=
by sorry

end NUMINAMATH_GPT_balls_sold_eq_13_l1098_109812


namespace NUMINAMATH_GPT_volume_after_increase_l1098_109867

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end NUMINAMATH_GPT_volume_after_increase_l1098_109867


namespace NUMINAMATH_GPT_trig_function_value_l1098_109823

noncomputable def f : ℝ → ℝ := sorry

theorem trig_function_value:
  (∀ x, f (Real.cos x) = Real.cos (3 * x)) →
  f (Real.sin (Real.pi / 6)) = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trig_function_value_l1098_109823


namespace NUMINAMATH_GPT_ram_salary_percentage_more_l1098_109825

theorem ram_salary_percentage_more (R r : ℝ) (h : r = 0.8 * R) :
  ((R - r) / r) * 100 = 25 := 
sorry

end NUMINAMATH_GPT_ram_salary_percentage_more_l1098_109825


namespace NUMINAMATH_GPT_total_amount_shared_l1098_109882

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1098_109882


namespace NUMINAMATH_GPT_three_consecutive_odd_numbers_l1098_109808

theorem three_consecutive_odd_numbers (x : ℤ) (h : x - 2 + x + x + 2 = 27) : 
  (x + 2, x, x - 2) = (11, 9, 7) :=
by
  sorry

end NUMINAMATH_GPT_three_consecutive_odd_numbers_l1098_109808


namespace NUMINAMATH_GPT_sun_volume_exceeds_moon_volume_by_387_cubed_l1098_109875

/-- Given Sun's distance to Earth is 387 times greater than Moon's distance to Earth. 
Given diameters:
- Sun's diameter: D_s
- Moon's diameter: D_m
Formula for volume of a sphere: V = (4/3) * pi * R^3
Derive that the Sun's volume exceeds the Moon's volume by 387^3 times. -/
theorem sun_volume_exceeds_moon_volume_by_387_cubed
  (D_s D_m : ℝ)
  (h : D_s = 387 * D_m) :
  (4/3) * Real.pi * (D_s / 2)^3 = 387^3 * (4/3) * Real.pi * (D_m / 2)^3 := by
  sorry

end NUMINAMATH_GPT_sun_volume_exceeds_moon_volume_by_387_cubed_l1098_109875


namespace NUMINAMATH_GPT_jason_current_money_l1098_109884

/-- Definition of initial amounts and earnings -/
def fred_initial : ℕ := 49
def jason_initial : ℕ := 3
def fred_current : ℕ := 112
def jason_earned : ℕ := 60

/-- The main theorem -/
theorem jason_current_money : jason_initial + jason_earned = 63 := 
by
  -- proof omitted for this example
  sorry

end NUMINAMATH_GPT_jason_current_money_l1098_109884


namespace NUMINAMATH_GPT_female_rainbow_trout_l1098_109844

-- Define the conditions given in the problem
variables (F_s M_s M_r F_r T : ℕ)
variables (h1 : F_s + M_s = 645)
variables (h2 : M_s = 2 * F_s + 45)
variables (h3 : 4 * M_r = 3 * F_s)
variables (h4 : 20 * M_r = 3 * T)
variables (h5 : T = 645 + F_r + M_r)

theorem female_rainbow_trout :
  F_r = 205 :=
by
  sorry

end NUMINAMATH_GPT_female_rainbow_trout_l1098_109844


namespace NUMINAMATH_GPT_find_z_plus_1_over_y_l1098_109827

theorem find_z_plus_1_over_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 20) : 
  z + 1 / y = 29 / 139 := 
by 
  sorry

end NUMINAMATH_GPT_find_z_plus_1_over_y_l1098_109827


namespace NUMINAMATH_GPT_profit_percentage_l1098_109822

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 75) : 
  ((selling_price - cost_price) / cost_price) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1098_109822


namespace NUMINAMATH_GPT_T_description_l1098_109837

def is_single_point {x y : ℝ} : Prop := (x = 2) ∧ (y = 11)

theorem T_description :
  ∀ (T : Set (ℝ × ℝ)),
  (∀ x y : ℝ, 
    (T (x, y) ↔ 
    ((5 = x + 3 ∧ 5 = y - 6) ∨ 
     (5 = x + 3 ∧ x + 3 = y - 6) ∨ 
     (5 = y - 6 ∧ x + 3 = y - 6)) ∧ 
    ((x = 2) ∧ (y = 11))
    )
  ) →
  (T = { (2, 11) }) :=
by
  sorry

end NUMINAMATH_GPT_T_description_l1098_109837


namespace NUMINAMATH_GPT_class_average_weight_l1098_109862

theorem class_average_weight :
  (24 * 40 + 16 * 35 + 18 * 42 + 22 * 38) / (24 + 16 + 18 + 22) = 38.9 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_class_average_weight_l1098_109862


namespace NUMINAMATH_GPT_ninth_observation_l1098_109841

theorem ninth_observation (avg1 : ℝ) (avg2 : ℝ) (n1 n2 : ℝ) 
  (sum1 : n1 * avg1 = 120) 
  (sum2 : n2 * avg2 = 117) 
  (avg_decrease : avg1 - avg2 = 2) 
  (obs_count_change : n1 + 1 = n2) 
  : n2 * avg2 - n1 * avg1 = -3 :=
by
  sorry

end NUMINAMATH_GPT_ninth_observation_l1098_109841


namespace NUMINAMATH_GPT_circumscribed_circle_area_l1098_109838

noncomputable def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area (s : ℝ) (hs : s = 15) : circle_area (circumradius s) = 75 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_area_l1098_109838


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1098_109824

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = x * y) : x + y ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1098_109824


namespace NUMINAMATH_GPT_mailing_ways_l1098_109859

-- Definitions based on the problem conditions
def countWays (letters mailboxes : ℕ) : ℕ := mailboxes^letters

-- The theorem to prove the mathematically equivalent proof problem
theorem mailing_ways (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) : countWays letters mailboxes = 4^3 := 
by
  rw [h_letters, h_mailboxes]
  rfl

end NUMINAMATH_GPT_mailing_ways_l1098_109859


namespace NUMINAMATH_GPT_sampling_method_is_stratified_l1098_109865

-- Given conditions
def unit_population : ℕ := 500 + 1000 + 800
def elderly_ratio : ℕ := 5
def middle_aged_ratio : ℕ := 10
def young_ratio : ℕ := 8
def total_selected : ℕ := 230

-- Prove that the sampling method used is stratified sampling
theorem sampling_method_is_stratified :
  (500 + 1000 + 800 = unit_population) ∧
  (total_selected = 230) ∧
  (500 * 230 / unit_population = elderly_ratio) ∧
  (1000 * 230 / unit_population = middle_aged_ratio) ∧
  (800 * 230 / unit_population = young_ratio) →
  sampling_method = stratified_sampling :=
by
  sorry

end NUMINAMATH_GPT_sampling_method_is_stratified_l1098_109865


namespace NUMINAMATH_GPT_remainder_correct_l1098_109817

noncomputable def p : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^5 + Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X + Polynomial.C 8
noncomputable def d : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) ^ 2
noncomputable def r : Polynomial ℝ := Polynomial.C 16 * Polynomial.X - Polynomial.C 8 

theorem remainder_correct : (p % d) = r := by sorry

end NUMINAMATH_GPT_remainder_correct_l1098_109817


namespace NUMINAMATH_GPT_geom_seq_min_value_l1098_109839

noncomputable def minimum_sum (m n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if (a 7 = a 6 + 2 * a 5) ∧ (a m * a n = 16 * (a 1) ^ 2) ∧ (m > 0) ∧ (n > 0) then
    (1 / m) + (4 / n)
  else
    0

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n, a m * a n = 16 * (a 1) ^ 2 ∧ m > 0 ∧ n > 0) →
  (minimum_sum m n a = 3 / 2) := sorry

end NUMINAMATH_GPT_geom_seq_min_value_l1098_109839


namespace NUMINAMATH_GPT_exists_int_less_than_sqrt_twenty_three_l1098_109821

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end NUMINAMATH_GPT_exists_int_less_than_sqrt_twenty_three_l1098_109821


namespace NUMINAMATH_GPT_not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l1098_109881

theorem not_right_triangle_sqrt_3_sqrt_4_sqrt_5 :
  ¬ (Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2 :=
by
  -- Start constructing the proof here
  sorry

end NUMINAMATH_GPT_not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l1098_109881


namespace NUMINAMATH_GPT_no_infinite_non_constant_arithmetic_progression_with_powers_l1098_109893

theorem no_infinite_non_constant_arithmetic_progression_with_powers (a b : ℕ) (b_ge_2 : b ≥ 2) : 
  ¬ ∃ (f : ℕ → ℕ) (d : ℕ), (∀ n : ℕ, f n = (a^(b + n*d)) ∧ b ≥ 2) := sorry

end NUMINAMATH_GPT_no_infinite_non_constant_arithmetic_progression_with_powers_l1098_109893


namespace NUMINAMATH_GPT_parallel_lines_slope_l1098_109836

theorem parallel_lines_slope (a : ℝ) :
  (∃ b : ℝ, ( ∀ x y : ℝ, a*x - 5*y - 9 = 0 → b*x - 3*y - 10 = 0) → a = 10/3) :=
sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1098_109836


namespace NUMINAMATH_GPT_car_total_travel_time_l1098_109832

def T_NZ : ℕ := 60

def T_NR : ℕ := 8 / 10 * T_NZ -- 80% of T_NZ

def T_ZV : ℕ := 3 / 4 * T_NR -- 75% of T_NR

theorem car_total_travel_time :
  T_NZ + T_NR + T_ZV = 144 := by
  sorry

end NUMINAMATH_GPT_car_total_travel_time_l1098_109832


namespace NUMINAMATH_GPT_smallest_n_l1098_109805

theorem smallest_n (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 2012) :
  ∃ m n : ℕ, a.factorial * b.factorial * c.factorial = m * 10 ^ n ∧ ¬ (10 ∣ m) ∧ n = 501 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1098_109805


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1098_109829

noncomputable def f (m x : ℝ) := Real.log (m * x) - x + 1
noncomputable def g (m x : ℝ) := (x - 1) * Real.exp x - m * x

theorem problem_part1 (m : ℝ) (h : m > 0) (hf : ∀ x, f m x ≤ 0) : m = 1 :=
sorry

theorem problem_part2 (m : ℝ) (h : m > 0) :
  ∃ x₀, (∀ x, g m x ≤ g m x₀) ∧ (1 / 2 * Real.log (m + 1) < x₀ ∧ x₀ < m) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1098_109829


namespace NUMINAMATH_GPT_initial_student_count_l1098_109830

theorem initial_student_count
  (n : ℕ)
  (T : ℝ)
  (h1 : T = 60.5 * (n : ℝ))
  (h2 : T - 8 = 64 * ((n - 1) : ℝ))
  : n = 16 :=
sorry

end NUMINAMATH_GPT_initial_student_count_l1098_109830


namespace NUMINAMATH_GPT_range_of_a_l1098_109896

theorem range_of_a (x a : ℝ) (p : (x + 1)^2 > 4) (q : x > a) 
  (h : (¬((x + 1)^2 > 4)) → (¬(x > a)))
  (sufficient_but_not_necessary : (¬((x + 1)^2 > 4)) → (¬(x > a))) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1098_109896


namespace NUMINAMATH_GPT_find_d_l1098_109835

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def less_than_10_primes (n : Nat) : Prop :=
  n < 10 ∧ is_prime n

theorem find_d (d e f : Nat) (hd : less_than_10_primes d) (he : less_than_10_primes e) (hf : less_than_10_primes f) :
  d + e = f → d < e → d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1098_109835


namespace NUMINAMATH_GPT_find_m_and_other_root_l1098_109863

theorem find_m_and_other_root (m x_2 : ℝ) :
  (∃ (x_1 : ℝ), x_1 = -1 ∧ x_1^2 + m * x_1 - 5 = 0) →
  m = -4 ∧ ∃ (x_2 : ℝ), x_2 = 5 ∧ x_2^2 + m * x_2 - 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_other_root_l1098_109863


namespace NUMINAMATH_GPT_missing_water_calculation_l1098_109807

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end NUMINAMATH_GPT_missing_water_calculation_l1098_109807


namespace NUMINAMATH_GPT_y_intercept_of_line_l1098_109833

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1098_109833


namespace NUMINAMATH_GPT_remainder_of_sum_of_5_consecutive_numbers_mod_9_l1098_109860

theorem remainder_of_sum_of_5_consecutive_numbers_mod_9 :
  (9154 + 9155 + 9156 + 9157 + 9158) % 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_5_consecutive_numbers_mod_9_l1098_109860


namespace NUMINAMATH_GPT_prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l1098_109894

theorem prop_P_subset_q_when_m_eq_1 :
  ∀ x : ℝ, ∀ m : ℝ, m = 1 → (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) ↔ (x ∈ {x | 0 ≤ x ∧ x ≤ 2}) := 
by sorry

theorem range_m_for_necessity_and_not_sufficiency :
  ∀ m : ℝ, (∀ x : ℝ, (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) → (x ∈ {x | 1 - m ≤ x ∧ x ≤ 1 + m})) ↔ (m ≥ 9) := 
by sorry

end NUMINAMATH_GPT_prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l1098_109894
