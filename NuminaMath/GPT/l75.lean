import Mathlib

namespace coefficient_of_x_eq_neg_6_l75_75873

noncomputable def coefficient_of_x_in_expansion : ℤ :=
  let expanded_expr := (1 + x) * (x - 2 / x) ^ 3
  in collect (x : ℝ) expanded_expr

theorem coefficient_of_x_eq_neg_6 : coefficient_of_x_in_expansion = -6 :=
  sorry

end coefficient_of_x_eq_neg_6_l75_75873


namespace ceiling_plus_floor_eq_zero_l75_75477

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l75_75477


namespace trigonometric_identity_l75_75950

theorem trigonometric_identity :
  Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) +
  Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l75_75950


namespace problem_solution_l75_75075

theorem problem_solution (x1 x2 x3 : ℝ) (h1: x1 < x2) (h2: x2 < x3)
(h3 : 10 * x1^3 - 201 * x1^2 + 3 = 0)
(h4 : 10 * x2^3 - 201 * x2^2 + 3 = 0)
(h5 : 10 * x3^3 - 201 * x3^2 + 3 = 0) :
x2 * (x1 + x3) = 398 :=
sorry

end problem_solution_l75_75075


namespace inequality_1_inequality_2_l75_75696

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2_l75_75696


namespace find_a7_l75_75684

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75684


namespace inequality_proof_l75_75169

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  sorry

end inequality_proof_l75_75169


namespace vasya_result_correct_l75_75991

def num : ℕ := 10^1990 + (10^1989 * 6 - 1)
def denom : ℕ := 10 * (10^1989 * 6 - 1) + 4

theorem vasya_result_correct : (num / denom) = (1 / 4) := 
  sorry

end vasya_result_correct_l75_75991


namespace problem_equivalent_proof_l75_75689

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75689


namespace verify_compound_interest_rate_l75_75147

noncomputable def compound_interest_rate
  (P A : ℝ) (t n : ℕ) : ℝ :=
  let r := (A / P) ^ (1 / (n * t)) - 1
  n * r

theorem verify_compound_interest_rate :
  let P := 5000
  let A := 6800
  let t := 4
  let n := 1
  compound_interest_rate P A t n = 8.02 / 100 :=
by
  sorry

end verify_compound_interest_rate_l75_75147


namespace acute_angle_at_3_15_l75_75260

/-- The hour and minute hands' angles and movements are defined as follows. -/
def hour_hand_angle (h m : Nat) : Real := (h % 12) * 30 + m * 0.5
def minute_hand_angle (m : Nat) : Real := (m % 60) * 6

/-- The condition that an acute angle is the smaller angle between hands. -/
def acute_angle (angle1 angle2 : Real) : Real := abs (angle1 - angle2)

/-- At 3:15, the acute angle between the hour and minute hands should be 7.5 degrees. -/
theorem acute_angle_at_3_15
    : acute_angle (hour_hand_angle 3 15) (minute_hand_angle 15) = 7.5 :=
by
    sorry

end acute_angle_at_3_15_l75_75260


namespace fraction_multiplication_division_l75_75633

-- We will define the fractions and state the equivalence
def fraction_1 : ℚ := 145 / 273
def fraction_2 : ℚ := 2 * (173 / 245) -- equivalent to 2 173/245
def fraction_3 : ℚ := 21 * (13 / 15) -- equivalent to 21 13/15

theorem fraction_multiplication_division :
  (frac1 * frac2 / frac3) = 7395 / 112504 := 
by sorry

end fraction_multiplication_division_l75_75633


namespace value_of_A_l75_75467

theorem value_of_A
  (A B C D E F G H I J : ℕ)
  (h_diff : ∀ x y : ℕ, x ≠ y → x ≠ y)
  (h_decreasing_ABC : A > B ∧ B > C)
  (h_decreasing_DEF : D > E ∧ E > F)
  (h_decreasing_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_odd_DEF : D % 2 = 1 ∧ E % 2 = 1 ∧ F % 2 = 1 ∧ E = D - 2 ∧ F = E - 2)
  (h_consecutive_even_GHIJ : G % 2 = 0 ∧ H % 2 = 0 ∧ I % 2 = 0 ∧ J % 2 = 0 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) : 
  A = 8 :=
sorry

end value_of_A_l75_75467


namespace sum_of_interior_angles_of_hexagon_l75_75580

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l75_75580


namespace range_of_real_number_l75_75981

theorem range_of_real_number (a : ℝ) : (a > 0) ∧ (a - 1 > 0) → a > 1 :=
by
  sorry

end range_of_real_number_l75_75981


namespace horner_rule_v3_is_36_l75_75960

def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_rule_v3_is_36 :
  let v0 := 1;
  let v1 := v0 * 3 + 0;
  let v2 := v1 * 3 + 2;
  let v3 := v2 * 3 + 3;
  v3 = 36 := 
by
  sorry

end horner_rule_v3_is_36_l75_75960


namespace mats_weaved_by_mat_weavers_l75_75922

variable (M : ℕ)

theorem mats_weaved_by_mat_weavers :
  -- 10 mat-weavers can weave 25 mats in 10 days
  (10 * 10) * M / (4 * 4) = 25 / (10 / 4)  →
  -- number of mats woven by 4 mat-weavers in 4 days
  M = 4 :=
sorry

end mats_weaved_by_mat_weavers_l75_75922


namespace smallest_number_l75_75112

-- Define the conditions
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def conditions (n : ℕ) : Prop := 
  (n > 12) ∧ 
  is_divisible_by (n - 12) 12 ∧ 
  is_divisible_by (n - 12) 24 ∧
  is_divisible_by (n - 12) 36 ∧
  is_divisible_by (n - 12) 48 ∧
  is_divisible_by (n - 12) 56

-- State the theorem
theorem smallest_number : ∃ n : ℕ, conditions n ∧ n = 1020 :=
by
  sorry

end smallest_number_l75_75112


namespace correct_calculation_l75_75264

variable (a : ℝ)

theorem correct_calculation : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_l75_75264


namespace solve_inequality_l75_75516

variables (a b c x α β : ℝ)

theorem solve_inequality 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (ha : a < 0)
  (h3 : α + β = -b / a)
  (h4 : α * β = c / a) :
  ∀ x, (c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α) := 
  by
    -- A detailed proof would follow here.
    sorry

end solve_inequality_l75_75516


namespace dice_surface_dots_l75_75745

def total_dots_on_die := 1 + 2 + 3 + 4 + 5 + 6

def total_dots_on_seven_dice := 7 * total_dots_on_die

def hidden_dots_on_central_die := total_dots_on_die

def visible_dots_on_surface := total_dots_on_seven_dice - hidden_dots_on_central_die

theorem dice_surface_dots : visible_dots_on_surface = 105 := by
  sorry

end dice_surface_dots_l75_75745


namespace factorization_of_polynomial_l75_75449

noncomputable def p (x : ℤ) : ℤ := x^15 + x^10 + x^5 + 1
noncomputable def f (x : ℤ) : ℤ := x^3 + x^2 + x + 1
noncomputable def g (x : ℤ) : ℤ := x^12 - x^11 + x^9 - x^8 + x^6 - x^5 + x^3 - x^2 + x - 1

theorem factorization_of_polynomial : ∀ x : ℤ, p x = f x * g x :=
by sorry

end factorization_of_polynomial_l75_75449


namespace price_of_cork_l75_75457

theorem price_of_cork (C : ℝ) 
  (h₁ : ∃ (bottle_with_cork bottle_without_cork : ℝ), bottle_with_cork = 2.10 ∧ bottle_without_cork = C + 2.00 ∧ bottle_with_cork = C + bottle_without_cork) :
  C = 0.05 :=
by
  obtain ⟨bottle_with_cork, bottle_without_cork, hwc, hwoc, ht⟩ := h₁
  sorry

end price_of_cork_l75_75457


namespace max_notebooks_with_budget_l75_75550

/-- Define the prices and quantities of notebooks -/
def notebook_price : ℕ := 2
def four_pack_price : ℕ := 6
def seven_pack_price : ℕ := 9
def max_budget : ℕ := 15

def total_notebooks (single_packs four_packs seven_packs : ℕ) : ℕ :=
  single_packs + 4 * four_packs + 7 * seven_packs

theorem max_notebooks_with_budget : 
  ∃ (single_packs four_packs seven_packs : ℕ), 
    notebook_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs ≤ max_budget ∧ 
    booklet_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs + total_notebooks single_packs four_packs seven_packs = 11 := 
by
  sorry

end max_notebooks_with_budget_l75_75550


namespace range_of_m_exists_l75_75166

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Proof problem statement
theorem range_of_m_exists (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = m := 
by
  sorry

end range_of_m_exists_l75_75166


namespace a_2016_value_l75_75035

def S (n : ℕ) : ℕ := n^2 - 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_2016_value : a 2016 = 4031 := by
  sorry

end a_2016_value_l75_75035


namespace negation_of_at_most_one_obtuse_l75_75573

-- Defining a predicate to express the concept of an obtuse angle
def is_obtuse (θ : ℝ) : Prop := θ > 90

-- Defining a triangle with three interior angles α, β, and γ
structure Triangle :=
  (α β γ : ℝ)
  (sum_angles : α + β + γ = 180)

-- Defining the condition that "At most, only one interior angle of a triangle is obtuse"
def at_most_one_obtuse (T : Triangle) : Prop :=
  (is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ)

-- The theorem we want to prove: Negation of "At most one obtuse angle" is "At least two obtuse angles"
theorem negation_of_at_most_one_obtuse (T : Triangle) :
  ¬ at_most_one_obtuse T ↔ (is_obtuse T.α ∧ is_obtuse T.β) ∨ (is_obtuse T.α ∧ is_obtuse T.γ) ∨ (is_obtuse T.β ∧ is_obtuse T.γ) := by
  sorry

end negation_of_at_most_one_obtuse_l75_75573


namespace inequality_solution_set_l75_75984

variable {f : ℝ → ℝ}

-- Conditions
def neg_domain : Set ℝ := {x | x < 0}
def pos_domain : Set ℝ := {x | x > 0}
def f_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f_property_P (f : ℝ → ℝ) := ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

-- Translate question and correct answer into a proposition in Lean
theorem inequality_solution_set (h1 : ∀ x, f (-x) = -f x)
                                (h2 : ∀ x1 x2, (0 < x1) → (0 < x2) → (x1 ≠ x1) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)) :
  {x | f (x - 2) < f (x^2 - 4) / (x + 2)} = {x | x < -3} ∪ {x | -1 < x ∧ x < 2} := 
sorry

end inequality_solution_set_l75_75984


namespace find_a7_l75_75656

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75656


namespace total_value_of_treats_l75_75301

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l75_75301


namespace sin_alpha_value_l75_75335

theorem sin_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < Real.pi)
  (h₂ : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

end sin_alpha_value_l75_75335


namespace minimum_value_of_phi_l75_75904

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def minimum_positive_period (ω : ℝ) := 2 * Real.pi / ω

theorem minimum_value_of_phi {A ω φ : ℝ} (hA : A > 0) (hω : ω > 0) 
  (h_period : minimum_positive_period ω = Real.pi) 
  (h_symmetry : ∀ x, f A ω φ x = f A ω φ (2 * Real.pi / ω - x)) : 
  ∃ k : ℤ, |φ| = |k * Real.pi - Real.pi / 6| → |φ| = Real.pi / 6 :=
by
  sorry

end minimum_value_of_phi_l75_75904


namespace augmented_matrix_solution_l75_75043

theorem augmented_matrix_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), (m * x = 6 ∧ 3 * y = n) ∧ (x = -3 ∧ y = 4)) →
  m + n = 10 :=
by
  intros m n h
  sorry

end augmented_matrix_solution_l75_75043


namespace inverse_proposition_equivalence_l75_75403

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l75_75403


namespace find_a7_l75_75688

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75688


namespace english_teachers_count_l75_75572

theorem english_teachers_count (E : ℕ) 
    (h_prob : 6 / ((E + 6) * (E + 5) / 2) = 1 / 12) : 
    E = 3 :=
by
  sorry

end english_teachers_count_l75_75572


namespace service_station_location_l75_75239

/-- The first exit is at milepost 35. -/
def first_exit_milepost : ℕ := 35

/-- The eighth exit is at milepost 275. -/
def eighth_exit_milepost : ℕ := 275

/-- The expected milepost of the service station built halfway between the first exit and the eighth exit is 155. -/
theorem service_station_location : (first_exit_milepost + (eighth_exit_milepost - first_exit_milepost) / 2) = 155 := by
  sorry

end service_station_location_l75_75239


namespace unique_solution_quadratic_l75_75237

theorem unique_solution_quadratic {a : ℚ} (h : ∃ x : ℚ, 2 * a * x^2 + 15 * x + 9 = 0) : 
  a = 25 / 8 ∧ (∃ x : ℚ, 2 * (25 / 8) * x^2 + 15 * x + 9 = 0 ∧ x = -12 / 5) := 
by
  sorry

end unique_solution_quadratic_l75_75237


namespace value_of_a_minus_2b_l75_75717

theorem value_of_a_minus_2b 
  (a b : ℚ) 
  (h : ∀ y : ℚ, y > 0 → y ≠ 2 → y ≠ -3 → (a / (y-2) + b / (y+3) = (2 * y + 5) / ((y-2)*(y+3)))) 
  : a - 2 * b = 7 / 5 :=
sorry

end value_of_a_minus_2b_l75_75717


namespace participation_plans_count_l75_75651

theorem participation_plans_count :
  ∃ (A B C D : Type), 
  fintype A ∧ fintype B ∧ fintype C ∧ fintype D ∧
  (∀ (set : finset (A ⊕ B ⊕ C ⊕ D)), set.card = 4) ∧
  (∀ (select : finset (A ⊕ B ⊕ C ⊕ D)), 
    select.card = 3 ∧
    (A ∈ select) → 
    (finset.filter (λ x, x ≠ A) select).card = 2 ∧
    (finset.permutations select).card = 6) →
    3 * 6 = 18 := 
by
  sorry

end participation_plans_count_l75_75651


namespace length_DC_of_ABCD_l75_75830

open Real

structure Trapezoid (ABCD : Type) :=
  (AB DC : ℝ)
  (BC : ℝ := 0)
  (angleBCD angleCDA : ℝ)

noncomputable def given_trapezoid : Trapezoid ℝ :=
{ AB := 5,
  DC := 8 + sqrt 3, -- this is from the answer
  BC := 3 * sqrt 2,
  angleBCD := π / 4,   -- 45 degrees in radians
  angleCDA := π / 3 }  -- 60 degrees in radians

variable (ABCD : Trapezoid ℝ)

theorem length_DC_of_ABCD :
  ABCD.AB = 5 ∧
  ABCD.BC = 3 * sqrt 2 ∧
  ABCD.angleBCD = π / 4 ∧
  ABCD.angleCDA = π / 3 →
  ABCD.DC = 8 + sqrt 3 :=
sorry

end length_DC_of_ABCD_l75_75830


namespace sixth_day_is_wednesday_l75_75369

noncomputable def day_of_week : Type := 
  { d // d ∈ ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] }

def five_fridays_sum_correct (x : ℤ) : Prop :=
  x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75

def first_is_friday (x : ℤ) : Prop :=
  x = 1

def day_of_6th_is_wednesday (d : day_of_week) : Prop :=
  d.1 = "Wednesday"

theorem sixth_day_is_wednesday (x : ℤ) (d : day_of_week) :
  five_fridays_sum_correct x → first_is_friday x → day_of_6th_is_wednesday d :=
by
  sorry

end sixth_day_is_wednesday_l75_75369


namespace find_value_of_x_l75_75055
-- Import the broader Mathlib to bring in the entirety of the necessary library

-- Definitions for the conditions
variables {x y z : ℝ}

-- Assume the given conditions
axiom h1 : x = y
axiom h2 : y = 2 * z
axiom h3 : x * y * z = 256

-- Statement to prove
theorem find_value_of_x : x = 8 :=
by {
  -- Proof goes here
  sorry
}

end find_value_of_x_l75_75055


namespace evaluate_expression_l75_75954

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end evaluate_expression_l75_75954


namespace two_a_minus_b_l75_75518

-- Definitions of vector components and parallelism condition
def is_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0
def vector_a : ℝ × ℝ := (1, -2)

-- Given assumptions
variable (m : ℝ)
def vector_b : ℝ × ℝ := (m, 4)

-- Theorem statement
theorem two_a_minus_b (h : is_parallel vector_a (vector_b m)) : 2 • vector_a - vector_b m = (4, -8) :=
sorry

end two_a_minus_b_l75_75518


namespace rabbit_clearing_10_square_yards_per_day_l75_75254

noncomputable def area_cleared_by_one_rabbit_per_day (length width : ℕ) (rabbits : ℕ) (days : ℕ) : ℕ :=
  (length * width) / (3 * 3 * rabbits * days)

theorem rabbit_clearing_10_square_yards_per_day :
  area_cleared_by_one_rabbit_per_day 200 900 100 20 = 10 :=
by sorry

end rabbit_clearing_10_square_yards_per_day_l75_75254


namespace infinite_very_good_pairs_l75_75462

-- Defining what it means for a pair to be "good"
def is_good (m n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ m ↔ p ∣ n)

-- Defining what it means for a pair to be "very good"
def is_very_good (m n : ℕ) : Prop :=
  is_good m n ∧ is_good (m + 1) (n + 1)

-- The theorem to prove: infiniteness of very good pairs
theorem infinite_very_good_pairs : Infinite {p : ℕ × ℕ | is_very_good p.1 p.2} :=
  sorry

end infinite_very_good_pairs_l75_75462


namespace billy_music_book_songs_l75_75148

theorem billy_music_book_songs (can_play : ℕ) (needs_to_learn : ℕ) (total_songs : ℕ) 
  (h1 : can_play = 24) (h2 : needs_to_learn = 28) : 
  total_songs = can_play + needs_to_learn ↔ total_songs = 52 :=
by
  sorry

end billy_music_book_songs_l75_75148


namespace loss_percentage_is_26_l75_75568

/--
Given the cost price of a radio is Rs. 1500 and the selling price is Rs. 1110, 
prove that the loss percentage is 26%
-/
theorem loss_percentage_is_26 (cost_price selling_price : ℝ)
  (h₀ : cost_price = 1500)
  (h₁ : selling_price = 1110) :
  ((cost_price - selling_price) / cost_price) * 100 = 26 := 
by 
  sorry

end loss_percentage_is_26_l75_75568


namespace find_a_l75_75631

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote : ∀ x : ℝ, x = π/2 ∨ x = 3*π/2 ∨ x = -π/2 ∨ x = -3*π/2 → b*x = π/2 ∨ b*x = 3*π/2 ∨ b*x = -π/2 ∨ b*x = -3*π/2)
  (h_amplitude : ∀ x : ℝ, |a * (1 / Real.cos (b * x))| ≤ 3): 
  a = 3 := 
sorry

end find_a_l75_75631


namespace speaker_discounted_price_correct_l75_75048

-- Define the initial price and the discount
def initial_price : ℝ := 475.00
def discount : ℝ := 276.00

-- Define the discounted price
def discounted_price : ℝ := initial_price - discount

-- The theorem to prove that the discounted price is 199.00
theorem speaker_discounted_price_correct : discounted_price = 199.00 :=
by
  -- Proof is omitted here, adding sorry to indicate it.
  sorry

end speaker_discounted_price_correct_l75_75048


namespace quadratic_root_condition_l75_75865

theorem quadratic_root_condition (m n : ℝ) (h : m * (-1)^2 - n * (-1) - 2023 = 0) :
  m + n = 2023 :=
sorry

end quadratic_root_condition_l75_75865


namespace min_sum_log_geq_four_l75_75190

theorem min_sum_log_geq_four (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (hlog : Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4) : 
  m + n ≥ 18 :=
sorry

end min_sum_log_geq_four_l75_75190


namespace turtles_remaining_on_log_l75_75121

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l75_75121


namespace expression_evaluates_to_one_l75_75296

theorem expression_evaluates_to_one :
  (1 / 3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) + (Real.pi - 2016)^0 - (8:ℝ)^(1/3) = 1 :=
by
  -- step-by-step simplification skipped, as per requirements
  sorry

end expression_evaluates_to_one_l75_75296


namespace solve_fraction_eq_l75_75643

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 3) :
  (x = 0 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) ∨ 
  (x = 2 / 3 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) :=
sorry

end solve_fraction_eq_l75_75643


namespace projection_of_a_in_direction_of_b_l75_75350

noncomputable def vector_projection_in_direction (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_in_direction_of_b :
  vector_projection_in_direction (3, 2) (-2, 1) = -4 * Real.sqrt 5 / 5 := 
by
  sorry

end projection_of_a_in_direction_of_b_l75_75350


namespace total_payment_l75_75586

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l75_75586


namespace count_seating_arrangements_l75_75870

/-
  Definition of the seating problem at the round table:
  - The committee has six members from each of three species: Martians (M), Venusians (V), and Earthlings (E).
  - The table has 18 seats numbered from 1 to 18.
  - Seat 1 is occupied by a Martian, and seat 18 is occupied by an Earthling.
  - Martians cannot sit immediately to the left of Venusians.
  - Venusians cannot sit immediately to the left of Earthlings.
  - Earthlings cannot sit immediately to the left of Martians.
-/
def num_arrangements_valid_seating : ℕ := -- the number of valid seating arrangements
  sorry

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def N : ℕ := 347

theorem count_seating_arrangements :
  num_arrangements_valid_seating = N * (factorial 6)^3 :=
sorry

end count_seating_arrangements_l75_75870


namespace angle_C_magnitude_area_triangle_l75_75986

variable {a b c A B C : ℝ}

namespace triangle

-- Conditions and variable declarations
axiom condition1 : 2 * b * Real.cos C = a * Real.cos C + c * Real.cos A
axiom triangle_sides : a = 3 ∧ b = 2 ∧ c = Real.sqrt 7

-- Prove the magnitude of angle C is π/3
theorem angle_C_magnitude : C = Real.pi / 3 :=
by sorry

-- Prove that given b = 2 and c = sqrt(7), a = 3 and the area of triangle ABC is 3*sqrt(3)/2
theorem area_triangle :
  (b = 2 ∧ c = Real.sqrt 7 ∧ C = Real.pi / 3) → 
  (a = 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2)) :=
by sorry

end triangle

end angle_C_magnitude_area_triangle_l75_75986


namespace money_problem_l75_75924

-- Define the conditions and the required proof
theorem money_problem (B S : ℕ) 
  (h1 : B = 2 * S) -- Condition 1: Brother brought twice as much money as the sister
  (h2 : B - 180 = S - 30) -- Condition 3: Remaining money of brother and sister are equal
  : B = 300 ∧ S = 150 := -- Correct answer to prove
  
sorry -- Placeholder for proof

end money_problem_l75_75924


namespace red_cars_count_l75_75111

variable (R B : ℕ)
variable (h1 : R * 8 = 3 * B)
variable (h2 : B = 90)

theorem red_cars_count : R = 33 :=
by
  -- here we would provide the proof
  sorry

end red_cars_count_l75_75111


namespace hyperbola_asymptote_value_l75_75853

theorem hyperbola_asymptote_value {b : ℝ} (h : b > 0) 
  (asymptote_eq : ∀ x : ℝ, y = x * (1 / 2) ∨ y = -x * (1 / 2)) :
  b = 1 :=
sorry

end hyperbola_asymptote_value_l75_75853


namespace triangle_right_angled_l75_75099

-- Define the variables and the condition of the problem
variables {a b c : ℝ}

-- Given condition of the problem
def triangle_condition (a b c : ℝ) : Prop :=
  2 * (a ^ 8 + b ^ 8 + c ^ 8) = (a ^ 4 + b ^ 4 + c ^ 4) ^ 2

-- The theorem to prove the triangle is right-angled
theorem triangle_right_angled (h : triangle_condition a b c) : a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 :=
sorry

end triangle_right_angled_l75_75099


namespace ratio_of_u_to_v_l75_75775

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l75_75775


namespace victor_won_games_l75_75598

theorem victor_won_games (V : ℕ) (ratio_victor_friend : 9 * 20 = 5 * V) : V = 36 :=
sorry

end victor_won_games_l75_75598


namespace larger_number_is_23_l75_75433

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75433


namespace complement_intersect_eq_l75_75349

-- Define Universal Set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define Set P
def P : Set ℕ := {2, 3, 4}

-- Define Set Q
def Q : Set ℕ := {1, 2}

-- Complement of P in U
def complement_U_P : Set ℕ := U \ P

-- Goal Statement
theorem complement_intersect_eq {U P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4}) 
  (hP : P = {2, 3, 4}) 
  (hQ : Q = {1, 2}) : 
  (complement_U_P ∩ Q) = {1} := 
by
  sorry

end complement_intersect_eq_l75_75349


namespace amy_balloons_l75_75544

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 232) (h2 : james_balloons = amy_balloons + 131) :
  amy_balloons = 101 :=
by
  sorry

end amy_balloons_l75_75544


namespace number_of_players_l75_75144

theorem number_of_players (S : ℕ) (h1 : S = 22) (h2 : ∀ (n : ℕ), S = n * 2) : ∃ n, n = 11 :=
by
  sorry

end number_of_players_l75_75144


namespace distance_between_foci_correct_l75_75647

/-- Define the given conditions for the ellipse -/
def ellipse_center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 7
def semi_minor_axis : ℝ := 3

/-- Define the distance between the foci of the ellipse -/
noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)

theorem distance_between_foci_correct :
  distance_between_foci = 4 * Real.sqrt 10 := by
  sorry

end distance_between_foci_correct_l75_75647


namespace probability_purple_between_green_and_twice_l75_75933

noncomputable def prob_purple_greater_green_but_less_than_twice (x_dist : measure ℝ) (y_dist : measure ℝ) : ℝ := 
  ∫⁻ x in Icc (0 : ℝ) 2, ∫⁻ y in Icc (0 : ℝ) 1, 
  if (x < y ∧ y < 2 * x) then 1 else 0 ∂y_dist ∂x_dist

theorem probability_purple_between_green_and_twice :
  let x_dist := measure.Uniform (Icc (0 : ℝ) 2),
      y_dist := measure.Uniform (Icc (0 : ℝ) 1) in
  prob_purple_greater_green_but_less_than_twice x_dist y_dist = 1 / 8 :=
by
  sorry

end probability_purple_between_green_and_twice_l75_75933


namespace total_cost_for_photos_l75_75802

def total_cost (n : ℕ) (f : ℝ) (c : ℝ) : ℝ :=
  f + (n - 4) * c

theorem total_cost_for_photos :
  total_cost 54 24.5 2.3 = 139.5 :=
by
  sorry

end total_cost_for_photos_l75_75802


namespace percentage_decrease_equivalent_l75_75909

theorem percentage_decrease_equivalent :
  ∀ (P D : ℝ), 
    (D = 10) →
    ((1.25 * P) - (D / 100) * (1.25 * P) = 1.125 * P) :=
by
  intros P D h
  rw [h]
  sorry

end percentage_decrease_equivalent_l75_75909


namespace scientific_notation_of_one_point_six_million_l75_75228

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l75_75228


namespace ratio_largest_middle_l75_75920

-- Definitions based on given conditions
def A : ℕ := 24  -- smallest number
def B : ℕ := 40  -- middle number
def C : ℕ := 56  -- largest number

theorem ratio_largest_middle (h1 : C = 56) (h2 : A = C - 32) (h3 : A = 24) (h4 : B = 40) :
  C / B = 7 / 5 := by
  sorry

end ratio_largest_middle_l75_75920


namespace turtles_remaining_on_log_l75_75124
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l75_75124


namespace max_value_expr_l75_75980

theorem max_value_expr (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
sorry

end max_value_expr_l75_75980


namespace sequence_monotonically_increasing_l75_75347

noncomputable def a (n : ℕ) : ℝ := (n - 1 : ℝ) / (n + 1 : ℝ)

theorem sequence_monotonically_increasing : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end sequence_monotonically_increasing_l75_75347


namespace least_possible_number_l75_75998

theorem least_possible_number {x : ℕ} (h1 : x % 6 = 2) (h2 : x % 4 = 3) : x = 50 :=
sorry

end least_possible_number_l75_75998


namespace geometric_product_l75_75875

theorem geometric_product (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 10) 
  (h2 : 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 = 5) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end geometric_product_l75_75875


namespace earnings_percentage_difference_l75_75452

-- Defining the conditions
def MikeEarnings : ℕ := 12
def PhilEarnings : ℕ := 6

-- Proving the percentage difference
theorem earnings_percentage_difference :
  ((MikeEarnings - PhilEarnings: ℕ) * 100 / MikeEarnings = 50) :=
by 
  sorry

end earnings_percentage_difference_l75_75452


namespace quadratic_equation_nonzero_coefficient_l75_75343

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end quadratic_equation_nonzero_coefficient_l75_75343


namespace number_of_people_with_cards_greater_than_0p3_l75_75548

theorem number_of_people_with_cards_greater_than_0p3 :
  (∃ (number_of_people : ℕ),
     number_of_people = (if 0.3 < 0.8 then 1 else 0) +
                        (if 0.3 < (1 / 2) then 1 else 0) +
                        (if 0.3 < 0.9 then 1 else 0) +
                        (if 0.3 < (1 / 3) then 1 else 0)) →
  number_of_people = 4 :=
by
  sorry

end number_of_people_with_cards_greater_than_0p3_l75_75548


namespace min_colors_rect_condition_l75_75040

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l75_75040


namespace least_integer_value_l75_75443

theorem least_integer_value 
  (x : ℤ) (h : |3 * x - 5| ≤ 22) : x = -5 ↔ ∃ (k : ℤ), k = -5 ∧ |3 * k - 5| ≤ 22 :=
by
  sorry

end least_integer_value_l75_75443


namespace hexagon_side_equality_l75_75869

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

def parallel (x y : A) : Prop := ∀ r : ℝ, x = r • y
noncomputable def length_eq (x y : A) : Prop := ∃ r : ℝ, r • x = y

variables (AB DE BC EF CD FA : A)
variables (h1 : parallel AB DE)
variables (h2 : parallel BC EF)
variables (h3 : parallel CD FA)
variables (h4 : length_eq AB DE)

theorem hexagon_side_equality :
  length_eq BC EF ∧ length_eq CD FA :=
by
  sorry

end hexagon_side_equality_l75_75869


namespace ball_hits_ground_at_10_over_7_l75_75754

def ball_hits_ground (t : ℚ) : Prop :=
  -4.9 * t^2 + 3.5 * t + 5 = 0

theorem ball_hits_ground_at_10_over_7 : ball_hits_ground (10 / 7) :=
by
  sorry

end ball_hits_ground_at_10_over_7_l75_75754


namespace xiaogang_xiaoqiang_speeds_and_time_l75_75269

theorem xiaogang_xiaoqiang_speeds_and_time
  (x y : ℕ)
  (distance_meeting : 2 * x = 2 * y + 24)
  (xiaogang_time_after_meeting : 0.5 * x = d_x)
  (total_distance : 2 * x + (2 * x - 24) = D)
  (xiaogang_time_total : D / x = meeting_time + 0.5)
  (xiaoqiang_time_total : D / y = meeting_time + time_xiaoqiang_to_A) :
  x = 16 ∧ y = 4 ∧ time_xiaoqiang_to_A = 8 := by
sorry

end xiaogang_xiaoqiang_speeds_and_time_l75_75269


namespace unit_digit_smaller_by_four_l75_75000

theorem unit_digit_smaller_by_four (x : ℤ) : x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4 :=
by
  sorry

end unit_digit_smaller_by_four_l75_75000


namespace product_two_digit_numbers_l75_75574

theorem product_two_digit_numbers (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 777) : (a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21) := 
  sorry

end product_two_digit_numbers_l75_75574


namespace exists_pair_distinct_integers_l75_75951

theorem exists_pair_distinct_integers :
  ∃ (a b : ℤ), a ≠ b ∧ (a / 2015 + b / 2016 = (2015 + 2016) / (2015 * 2016)) :=
by
  -- Constructing the proof or using sorry to skip it if not needed here
  sorry

end exists_pair_distinct_integers_l75_75951


namespace eval_ceil_floor_l75_75489

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l75_75489


namespace katya_notebooks_l75_75625

theorem katya_notebooks (rubles : ℕ) (cost_per_notebook : ℕ) (stickers_per_notebook : ℕ) (sticker_rate : ℕ):
  rubles = 150 -> cost_per_notebook = 4 -> stickers_per_notebook = 1 -> sticker_rate = 5 ->
  let initial_notebooks := rubles / cost_per_notebook in
  let remaining_rubles := rubles % cost_per_notebook in
  let initial_stickers := initial_notebooks * stickers_per_notebook in
  ∃ final_notebooks : ℕ, 
    (initial_notebooks = 37 ∧ remaining_rubles = 2 ∧ initial_stickers = 37 ∧ 
    (final_notebooks = initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate) + (((initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate)) * stickers_per_notebook) / sticker_rate)) = 46) :=
begin
  intros,
  sorry
end

end katya_notebooks_l75_75625


namespace find_w_l75_75188

variables {x y z w : ℝ}

theorem find_w (h : (1 / x) + (1 / y) + (1 / z) = 1 / w) :
  w = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end find_w_l75_75188


namespace odd_square_minus_one_multiple_of_eight_l75_75221

theorem odd_square_minus_one_multiple_of_eight (a : ℤ) 
  (h₁ : a > 0) 
  (h₂ : a % 2 = 1) : 
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_multiple_of_eight_l75_75221


namespace sqrt_of_square_is_identity_l75_75861

variable {a : ℝ} (h : a > 0)

theorem sqrt_of_square_is_identity (h : a > 0) : Real.sqrt (a^2) = a := 
  sorry

end sqrt_of_square_is_identity_l75_75861


namespace domain_of_h_l75_75637

open Real

def h (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 3) / (|x^2 - 9| + |x - 2|^2)

theorem domain_of_h : ∀ x : ℝ, |x^2 - 9| + |x - 2|^2 ≠ 0 :=
by {
  intro x,
  have h1 : 0 ≤ |x^2 - 9|,
  { exact abs_nonneg _ },
  have h2 : 0 ≤ |x - 2|^2,
  { exact pow_two_nonneg _ },
  by_contradiction H,
  have : |x^2 - 9| = 0 ∧ |x - 2|^2 = 0,
  { rw← add_eq_zero_iff at H,
    exact H, },
  cases this with h3 h4,
  {
    rw abs_eq_zero at h3,
    rw pow_eq_zero_iff at h4,
    cases h4 with h4_1 h4_2,
    rw h3 at h4_1,
    rw h3 at h4_2,
    interval_cases x;
      norm_num at h1 h2 h3 h4_1 h4_2 *,
  }
}
sorry

end domain_of_h_l75_75637


namespace factor_72x3_minus_252x7_l75_75640

theorem factor_72x3_minus_252x7 (x : ℝ) : (72 * x^3 - 252 * x^7) = (36 * x^3 * (2 - 7 * x^4)) :=
by
  sorry

end factor_72x3_minus_252x7_l75_75640


namespace find_f_4_l75_75399

-- Lean code to encapsulate the conditions and the goal
theorem find_f_4 (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x * f y = y * f x)
  (h2 : f 12 = 24) : 
  f 4 = 8 :=
sorry

end find_f_4_l75_75399


namespace number_of_students_third_l75_75575

-- Define the ratio and the total number of samples.
def ratio_first : ℕ := 3
def ratio_second : ℕ := 3
def ratio_third : ℕ := 4
def total_sample : ℕ := 50

-- Define the condition that the sum of ratios equals the total proportion numerator.
def sum_ratios : ℕ := ratio_first + ratio_second + ratio_third

-- Final proposition: the number of students to be sampled from the third grade.
theorem number_of_students_third :
  (ratio_third * total_sample) / sum_ratios = 20 := by
  sorry

end number_of_students_third_l75_75575


namespace equation_of_line_l75_75849

theorem equation_of_line {M : ℝ × ℝ} {a b : ℝ} (hM : M = (4,2)) 
  (hAB : ∃ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ 
    A ≠ B ∧ ∀ x y : ℝ, 
    (x^2 + 4 * y^2 = 36 → (∃ k : ℝ, y - 2 = k * (x - 4) ) )):
  (x + 2 * y - 8 = 0) :=
sorry

end equation_of_line_l75_75849


namespace find_a7_l75_75665

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75665


namespace larger_number_is_23_l75_75410

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75410


namespace find_a7_l75_75666

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75666


namespace total_fishes_caught_l75_75891

def melanieCatches : ℕ := 8
def tomCatches : ℕ := 3 * melanieCatches
def totalFishes : ℕ := melanieCatches + tomCatches

theorem total_fishes_caught : totalFishes = 32 := by
  sorry

end total_fishes_caught_l75_75891


namespace total_pizza_pieces_l75_75303

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l75_75303


namespace question_2_question_3_l75_75706

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 1/x - 1 else 1 - 1/x

theorem question_2 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  1/a + 1/b = 2 :=
sorry

theorem question_3 (a b m : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : ∀ x, a ≤ x ∧ x ≤ b → f x ∈ Set.Icc (m * a) (m * b)) (h4 : m ≠ 0) :
  0 < m ∧ m < 1/4 :=
sorry

end question_2_question_3_l75_75706


namespace sequence_a10_l75_75365

theorem sequence_a10 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n+1) - a n = 1 / (4 * ↑n^2 - 1)) :
  a 10 = 28 / 19 :=
by
  sorry

end sequence_a10_l75_75365


namespace quadratic_condition_l75_75342

theorem quadratic_condition (m : ℝ) (h : (m - 1) ≠ 0) : m ≠ 1 :=
by {
  intro h1,
  rw h1 at h,
  apply h,
  ring,
}

end quadratic_condition_l75_75342


namespace johns_initial_bench_press_weight_l75_75214

noncomputable def initialBenchPressWeight (currentWeight: ℝ) (injuryPercentage: ℝ) (trainingFactor: ℝ) :=
  (currentWeight / (injuryPercentage / 100 * trainingFactor))

theorem johns_initial_bench_press_weight:
  (initialBenchPressWeight 300 80 3) = 500 :=
by
  sorry

end johns_initial_bench_press_weight_l75_75214


namespace moles_of_NaOH_l75_75189

-- Statement of the problem conditions and desired conclusion
theorem moles_of_NaOH (moles_H2SO4 moles_NaHSO4 : ℕ) (h : moles_H2SO4 = 3) (h_eq : moles_H2SO4 = moles_NaHSO4) : moles_NaHSO4 = 3 := by
  sorry

end moles_of_NaOH_l75_75189


namespace find_g_seven_l75_75906

noncomputable def g : ℝ → ℝ :=
  sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_six : g 6 = 7

theorem find_g_seven : g 7 = 49 / 6 :=
by
  -- Proof omitted here
  sorry

end find_g_seven_l75_75906


namespace count_even_n_factorial_condition_l75_75318

def is_integer (x : ℚ) : Prop := ∃ z : ℤ, x = z

theorem count_even_n_factorial_condition :
  (finset.card (finset.filter (λ n : ℕ, n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 100 ∧ is_integer (rat.of_int ((factorial (n^2 - 1)) / (rat.of_int (factorial n)^n)))) (finset.range 101))) = 5 := 
by
  sorry

end count_even_n_factorial_condition_l75_75318


namespace cubed_expression_l75_75859

theorem cubed_expression (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 :=
sorry

end cubed_expression_l75_75859


namespace production_rate_l75_75353

theorem production_rate (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (H : x * x * 2 * x = 2 * x^3) :
  y * y * 3 * y = 3 * y^3 := by
  sorry

end production_rate_l75_75353


namespace ellipse_semi_focal_range_l75_75338

-- Definitions and conditions from the problem
variables (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : a^2 = b^2 + c^2)

-- Statement of the theorem
theorem ellipse_semi_focal_range : 1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 :=
by 
  sorry

end ellipse_semi_focal_range_l75_75338


namespace Lynne_bought_3_magazines_l75_75078

open Nat

def books_about_cats : Nat := 7
def books_about_solar_system : Nat := 2
def book_cost : Nat := 7
def magazine_cost : Nat := 4
def total_spent : Nat := 75

theorem Lynne_bought_3_magazines:
  let total_books := books_about_cats + books_about_solar_system
  let total_cost_books := total_books * book_cost
  let total_cost_magazines := total_spent - total_cost_books
  total_cost_magazines / magazine_cost = 3 :=
by sorry

end Lynne_bought_3_magazines_l75_75078


namespace no_odd_total_given_ratio_l75_75538

theorem no_odd_total_given_ratio (T : ℕ) (hT1 : 50 < T) (hT2 : T < 150) (hT3 : T % 2 = 1) : 
  ∀ (B : ℕ), T ≠ 8 * B + B / 4 :=
sorry

end no_odd_total_given_ratio_l75_75538


namespace magician_decks_l75_75930

theorem magician_decks :
  ∀ (initial_decks price_per_deck earnings decks_sold decks_left_unsold : ℕ),
  initial_decks = 5 →
  price_per_deck = 2 →
  earnings = 4 →
  decks_sold = earnings / price_per_deck →
  decks_left_unsold = initial_decks - decks_sold →
  decks_left_unsold = 3 :=
by
  intros initial_decks price_per_deck earnings decks_sold decks_left_unsold
  intros h_initial h_price h_earnings h_sold h_left
  rw [h_initial, h_price, h_earnings] at *
  sorry

end magician_decks_l75_75930


namespace andrew_permit_rate_l75_75290

def permits_per_hour (a h_a H T : ℕ) : ℕ :=
  T / (H - (a * h_a))

theorem andrew_permit_rate :
  permits_per_hour 2 3 8 100 = 50 := by
  sorry

end andrew_permit_rate_l75_75290


namespace sum_x_y_m_l75_75224

theorem sum_x_y_m (x y m : ℕ) (h1 : x >= 10 ∧ x < 100) (h2 : y >= 10 ∧ y < 100) 
  (h3 : ∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) 
  (h4 : x^2 - y^2 = 4 * m^2) : 
  x + y + m = 105 := 
sorry

end sum_x_y_m_l75_75224


namespace integer_solutions_of_linear_diophantine_eq_l75_75799

theorem integer_solutions_of_linear_diophantine_eq 
  (a b c : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (x₀ y₀ : ℤ)
  (h_particular_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, (a * x + b * y = c) → ∃ (k : ℤ), (x = x₀ + k * b) ∧ (y = y₀ - k * a) := 
by
  sorry

end integer_solutions_of_linear_diophantine_eq_l75_75799


namespace reflect_triangle_final_position_l75_75582

variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Definition of reflection in x-axis and y-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

theorem reflect_triangle_final_position (x1 x2 x3 y1 y2 y3 : ℝ) :
  (reflect_y (reflect_x x1 y1).1 (reflect_x x1 y1).2) = (-x1, -y1) ∧
  (reflect_y (reflect_x x2 y2).1 (reflect_x x2 y2).2) = (-x2, -y2) ∧
  (reflect_y (reflect_x x3 y3).1 (reflect_x x3 y3).2) = (-x3, -y3) :=
by
  sorry

end reflect_triangle_final_position_l75_75582


namespace increase_in_area_l75_75117

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width
noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
noncomputable def radius_of_circle (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)
noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * (radius ^ 2)

theorem increase_in_area :
  let rectangle_length := 60
  let rectangle_width := 20
  let rectangle_area := area_of_rectangle rectangle_length rectangle_width
  let fence_length := perimeter_of_rectangle rectangle_length rectangle_width
  let circle_radius := radius_of_circle fence_length
  let circle_area := area_of_circle circle_radius
  let area_increase := circle_area - rectangle_area
  837.99 ≤ area_increase :=
by
  sorry

end increase_in_area_l75_75117


namespace sequence_divisibility_condition_l75_75319

theorem sequence_divisibility_condition (t a b x1 : ℕ) (x : ℕ → ℕ)
  (h1 : a = 1) (h2 : b = t) (h3 : x1 = t) (h4 : x 1 = x1)
  (h5 : ∀ n, n ≥ 2 → x n = a * x (n - 1) + b) :
  (∀ m n, m ∣ n → x m ∣ x n) ↔ (a = 1 ∧ b = t ∧ x1 = t) := sorry

end sequence_divisibility_condition_l75_75319


namespace isosceles_triangle_congruent_side_length_l75_75900

theorem isosceles_triangle_congruent_side_length 
  (base : ℝ) (area : ℝ) (a b c : ℝ) 
  (h1 : a = c)
  (h2 : a = base / 2)
  (h3 : (base * a) / 2 = area)
  : b = 5 * Real.sqrt 10 := 
by sorry

end isosceles_triangle_congruent_side_length_l75_75900


namespace sum_of_arithmetic_progression_l75_75198

theorem sum_of_arithmetic_progression 
  (a d : ℚ) 
  (S : ℕ → ℚ)
  (h_sum_15 : S 15 = 150)
  (h_sum_75 : S 75 = 30)
  (h_arith_sum : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  S 90 = -180 :=
by
  sorry

end sum_of_arithmetic_progression_l75_75198


namespace baseball_wins_l75_75808

-- Define the constants and conditions
def total_games : ℕ := 130
def won_more_than_lost (L W : ℕ) : Prop := W = 3 * L + 14
def total_games_played (L W : ℕ) : Prop := W + L = total_games

-- Define the theorem statement
theorem baseball_wins (L W : ℕ) 
  (h1 : won_more_than_lost L W)
  (h2 : total_games_played L W) : 
  W = 101 :=
  sorry

end baseball_wins_l75_75808


namespace max_three_digit_sum_l75_75527

theorem max_three_digit_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 101 * A + 11 * B + 11 * C = 986 := 
sorry

end max_three_digit_sum_l75_75527


namespace monotonicity_and_k_range_l75_75887

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1 / 2 : ℝ) * x^2 - x

theorem monotonicity_and_k_range :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) ↔ k ∈ Set.Iic (-2) := sorry

end monotonicity_and_k_range_l75_75887


namespace circle_eq_of_given_center_and_radius_l75_75962

theorem circle_eq_of_given_center_and_radius :
  (∀ (x y : ℝ),
    let C := (-1, 2)
    let r := 4
    (x + 1) ^ 2 + (y - 2) ^ 2 = 16) :=
by
  sorry

end circle_eq_of_given_center_and_radius_l75_75962


namespace negation_of_existence_l75_75242

theorem negation_of_existence : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) = (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by
  sorry

end negation_of_existence_l75_75242


namespace inequality_for_real_numbers_l75_75648

theorem inequality_for_real_numbers (x y z : ℝ) : 
  - (3 / 2) * (x^2 + y^2 + 2 * z^2) ≤ 3 * x * y + y * z + z * x ∧ 
  3 * x * y + y * z + z * x ≤ (3 + Real.sqrt 13) / 4 * (x^2 + y^2 + 2 * z^2) :=
by
  sorry

end inequality_for_real_numbers_l75_75648


namespace prove_inequality_l75_75762

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l75_75762


namespace evaluation_of_expression_l75_75603

theorem evaluation_of_expression : (3^2 - 2^2 + 1^2) = 6 :=
by
  sorry

end evaluation_of_expression_l75_75603


namespace distance_rowed_downstream_l75_75129

-- Define the conditions
def speed_in_still_water (b s: ℝ) := b - s = 60 / 4
def speed_of_stream (s: ℝ) := s = 3
def time_downstream (t: ℝ) := t = 4

-- Define the function that computes the downstream speed
def downstream_speed (b s t: ℝ) := (b + s) * t

-- The theorem we want to prove
theorem distance_rowed_downstream (b s t : ℝ) 
    (h1 : speed_in_still_water b s)
    (h2 : speed_of_stream s)
    (h3 : time_downstream t) : 
    downstream_speed b s t = 84 := by
    sorry

end distance_rowed_downstream_l75_75129


namespace rectangle_dimensions_l75_75095

theorem rectangle_dimensions (x y : ℝ) (h1 : y = 2 * x) (h2 : 2 * (x + y) = 2 * (x * y)) :
  (x = 3 / 2) ∧ (y = 3) := by
  sorry

end rectangle_dimensions_l75_75095


namespace find_a7_l75_75675

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75675


namespace Tim_total_payment_l75_75591

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l75_75591


namespace DanteSoldCoconuts_l75_75559

variable (Paolo_coconuts : ℕ) (Dante_coconuts : ℕ) (coconuts_left : ℕ)

def PaoloHasCoconuts := Paolo_coconuts = 14

def DanteHasThriceCoconuts := Dante_coconuts = 3 * Paolo_coconuts

def DanteLeftCoconuts := coconuts_left = 32

theorem DanteSoldCoconuts 
  (h1 : PaoloHasCoconuts Paolo_coconuts) 
  (h2 : DanteHasThriceCoconuts Paolo_coconuts Dante_coconuts) 
  (h3 : DanteLeftCoconuts coconuts_left) : 
  Dante_coconuts - coconuts_left = 10 := 
by
  rw [PaoloHasCoconuts, DanteHasThriceCoconuts, DanteLeftCoconuts] at *
  sorry

end DanteSoldCoconuts_l75_75559


namespace eval_expression_l75_75019

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l75_75019


namespace luncheon_cost_l75_75396

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + 2 * p = 3.50)
  (h2 : 3 * s + 7 * c + 2 * p = 4.90) :
  s + c + p = 1.00 :=
  sorry

end luncheon_cost_l75_75396


namespace base_6_to_10_conversion_l75_75601

theorem base_6_to_10_conversion : 
  ∀ (n : ℕ), n = 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0 → n = 1295 :=
by
  intro n h
  sorry

end base_6_to_10_conversion_l75_75601


namespace solve_for_x_l75_75602

theorem solve_for_x (x : ℝ) : (2010 + 2 * x) ^ 2 = x ^ 2 → x = -2010 ∨ x = -670 := by
  sorry

end solve_for_x_l75_75602


namespace expression_value_l75_75813

-- Define the difference of squares identity
lemma diff_of_squares (x y : ℤ) : x^2 - y^2 = (x + y) * (x - y) :=
by sorry

-- Define the specific values for x and y
def x := 7
def y := 3

-- State the theorem to be proven
theorem expression_value : ((x^2 - y^2)^2) = 1600 :=
by sorry

end expression_value_l75_75813


namespace max_area_of_cone_l75_75178

noncomputable def max_cross_sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

theorem max_area_of_cone :
  (∀ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) → max_cross_sectional_area 3 θ ≤ (9 / 2))
  ∧ (∃ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) ∧ max_cross_sectional_area 3 θ = (9 / 2)) := 
by
  sorry

end max_area_of_cone_l75_75178


namespace speed_of_stream_l75_75451

theorem speed_of_stream (b s : ℝ) (h1 : 75 = 5 * (b + s)) (h2 : 45 = 5 * (b - s)) : s = 3 :=
by
  have eq1 : b + s = 15 := by linarith [h1]
  have eq2 : b - s = 9 := by linarith [h2]
  have b_val : b = 12 := by linarith [eq1, eq2]
  linarith 

end speed_of_stream_l75_75451


namespace SamaraSpentOnDetailing_l75_75002

def costSamara (D : ℝ) : ℝ := 25 + 467 + D
def costAlberto : ℝ := 2457
def difference : ℝ := 1886

theorem SamaraSpentOnDetailing : 
  ∃ (D : ℝ), costAlberto = costSamara D + difference ∧ D = 79 := 
sorry

end SamaraSpentOnDetailing_l75_75002


namespace domain_of_g_l75_75257

noncomputable def g : ℝ → ℝ := λ x, (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_g_l75_75257


namespace binkie_gemstones_l75_75159

variable (Binkie Frankie Spaatz : ℕ)

-- Define the given conditions
def condition1 : Binkie = 4 * Frankie := by sorry
def condition2 : Spaatz = (1 / 2) * Frankie - 2 := by sorry
def condition3 : Spaatz = 1 := by sorry

-- State the theorem to be proved
theorem binkie_gemstones : Binkie = 24 := by
  have h_Frankie : Frankie = 6 := by
    sorry
  rw [←condition3, ←condition2] at h_Frankie
  have h_Binkie : Binkie = 4 * 6 := by
    rw [condition1]
    sorry
  rw [h_Binkie]
  exact
    show 4 * 6 = 24 from rfl

end binkie_gemstones_l75_75159


namespace solution_set_inequality_l75_75837

theorem solution_set_inequality (x : ℝ) : 4 * x^2 - 3 * x > 5 ↔ x < -5/4 ∨ x > 1 :=
by
  sorry

end solution_set_inequality_l75_75837


namespace juice_profit_eq_l75_75307

theorem juice_profit_eq (x : ℝ) :
  (70 - x) * (160 + 8 * x) = 16000 :=
sorry

end juice_profit_eq_l75_75307


namespace problem_equivalent_proof_l75_75692

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75692


namespace albert_brother_younger_l75_75937

variables (A B Y F M : ℕ)
variables (h1 : F = 48)
variables (h2 : M = 46)
variables (h3 : F - M = 4)
variables (h4 : Y = A - B)

theorem albert_brother_younger (h_cond : (F - M = 4) ∧ (F = 48) ∧ (M = 46) ∧ (Y = A - B)) : Y = 2 :=
by
  rcases h_cond with ⟨h_diff, h_father, h_mother, h_ages⟩
  -- Assuming that each step provided has correct assertive logic.
  sorry

end albert_brother_younger_l75_75937


namespace baseball_cards_per_pack_l75_75082

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end baseball_cards_per_pack_l75_75082


namespace line_intersects_circle_l75_75921

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x - 1) ∧ x^2 + y^2 = 1 :=
by
  sorry

end line_intersects_circle_l75_75921


namespace ink_cartridge_15th_month_l75_75027

def months_in_year : ℕ := 12
def first_change_month : ℕ := 1   -- January is the first month

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + (3 * (n - 1))) % months_in_year

theorem ink_cartridge_15th_month : nth_change_month 15 = 7 := by
  -- This is where the proof would go
  sorry

end ink_cartridge_15th_month_l75_75027


namespace larger_number_is_23_l75_75421

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75421


namespace chess_tournament_games_count_l75_75866

/-- In a chess tournament with 16 participants, each playing exactly one game with each other, the total number of games played is 120.-/
theorem chess_tournament_games_count : ∑ k in range 15, 16 - k = 120 :=
by
  sorry

end chess_tournament_games_count_l75_75866


namespace max_pieces_l75_75501

theorem max_pieces (plywood_width plywood_height piece_width piece_height : ℕ)
  (h_plywood : plywood_width = 22) (h_plywood_height : plywood_height = 15)
  (h_piece : piece_width = 3) (h_piece_height : piece_height = 5) :
  (plywood_width * plywood_height) / (piece_width * piece_height) = 22 := by
  sorry

end max_pieces_l75_75501


namespace eleven_step_paths_l75_75947

def H : (ℕ × ℕ) := (0, 0)
def K : (ℕ × ℕ) := (4, 3)
def J : (ℕ × ℕ) := (6, 5)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem eleven_step_paths (H K J : (ℕ × ℕ)) (H_coords : H = (0, 0)) (K_coords : K = (4, 3)) (J_coords : J = (6, 5)) : 
  (binomial 7 4) * (binomial 4 2) = 210 := by 
  sorry

end eleven_step_paths_l75_75947


namespace simplify_expr_l75_75387

theorem simplify_expr (a : ℝ) : 2 * a * (3 * a ^ 2 - 4 * a + 3) - 3 * a ^ 2 * (2 * a - 4) = 4 * a ^ 2 + 6 * a :=
by
  sorry

end simplify_expr_l75_75387


namespace bricks_required_to_pave_courtyard_l75_75458

theorem bricks_required_to_pave_courtyard :
  let courtyard_length_m := 24
  let courtyard_width_m := 14
  let brick_length_cm := 25
  let brick_width_cm := 15
  let courtyard_area_m2 := courtyard_length_m * courtyard_width_m
  let courtyard_area_cm2 := courtyard_area_m2 * 10000
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  let num_bricks := courtyard_area_cm2 / brick_area_cm2
  num_bricks = 8960 := by
  {
    -- Additional context not needed for theorem statement, mock proof omitted
    sorry
  }

end bricks_required_to_pave_courtyard_l75_75458


namespace find_a2_l75_75345

theorem find_a2 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + 2)
  (h_geom : (a 1) * (a 5) = (a 2) * (a 2)) : a 2 = 3 :=
by
  -- We are given the conditions and need to prove the statement.
  sorry

end find_a2_l75_75345


namespace find_correction_time_l75_75287

-- Define the conditions
def loses_minutes_per_day : ℚ := 2 + 1/2
def initial_time_set : ℚ := 1 * 60 -- 1 PM in minutes
def time_on_march_21 : ℚ := 9 * 60 -- 9 AM in minutes on March 21
def total_minutes_per_day : ℚ := 24 * 60
def days_between : ℚ := 6 - 4/24 -- 6 days minus 4 hours

-- Calculate effective functioning minutes per day
def effective_minutes_per_day : ℚ := total_minutes_per_day - loses_minutes_per_day

-- Calculate the ratio of actual time to the watch's time
def time_ratio : ℚ := total_minutes_per_day / effective_minutes_per_day

-- Calculate the total actual time in minutes between initial set time and the given time showing on the watch
def total_actual_time : ℚ := days_between * total_minutes_per_day + initial_time_set

-- Calculate the actual time according to the ratio
def actual_time_according_to_ratio : ℚ := total_actual_time * time_ratio

-- Calculate the correction required 'n'
def required_minutes_correction : ℚ := actual_time_according_to_ratio - total_actual_time

-- The theorem stating that the required correction is as calculated
theorem find_correction_time : required_minutes_correction = (14 + 14/23) := by
  sorry

end find_correction_time_l75_75287


namespace rubber_band_problem_l75_75138

noncomputable def a : ℤ := 4
noncomputable def b : ℤ := 12
noncomputable def c : ℤ := 3
noncomputable def band_length := a * Real.pi + b * Real.sqrt c

theorem rubber_band_problem (r1 r2 d : ℝ) (h1 : r1 = 3) (h2 : r2 = 9) (h3 : d = 12) :
  let a := 4
  let b := 12
  let c := 3
  let band_length := a * Real.pi + b * Real.sqrt c
  a + b + c = 19 :=
by
  sorry

end rubber_band_problem_l75_75138


namespace probability_of_three_positive_answers_l75_75068

noncomputable def probability_exactly_three_positive_answers : ℚ :=
  (7.choose 3) * (3/7)^3 * (4/7)^4

theorem probability_of_three_positive_answers :
  probability_exactly_three_positive_answers = 242520 / 823543 :=
by
  unfold probability_exactly_three_positive_answers
  sorry

end probability_of_three_positive_answers_l75_75068


namespace find_a_in_triangle_l75_75368

variable (a b c : ℝ) (A B C : ℝ)

-- Given conditions
def condition_c : c = 3 := sorry
def condition_C : C = Real.pi / 3 := sorry
def condition_sinB : Real.sin B = 2 * Real.sin A := sorry

-- Theorem to prove
theorem find_a_in_triangle (hC : condition_C) (hc : condition_c) (hsinB : condition_sinB) :
  a = Real.sqrt 3 :=
sorry

end find_a_in_triangle_l75_75368


namespace number_of_laborers_l75_75289

-- Definitions based on conditions in the problem
def hpd := 140   -- Earnings per day for heavy equipment operators
def gpd := 90    -- Earnings per day for general laborers
def totalPeople := 35  -- Total number of people hired
def totalPayroll := 3950  -- Total payroll in dollars

-- Variables H and L for the number of operators and laborers
variables (H L : ℕ)

-- Conditions provided in mathematical problem
axiom equation1 : H + L = totalPeople
axiom equation2 : hpd * H + gpd * L = totalPayroll

-- Theorem statement: we want to prove that L = 19
theorem number_of_laborers : L = 19 :=
sorry

end number_of_laborers_l75_75289


namespace sequence_a3_l75_75064

theorem sequence_a3 (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (recursion : ∀ n, a (n + 1) = a n / (1 + a n)) : 
  a 3 = 1 / 3 :=
by 
  sorry

end sequence_a3_l75_75064


namespace largest_divisor_of_product_of_consecutive_evens_l75_75780

theorem largest_divisor_of_product_of_consecutive_evens (n : ℤ) : 
  ∃ d, d = 8 ∧ ∀ n, d ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end largest_divisor_of_product_of_consecutive_evens_l75_75780


namespace problem_statement_l75_75505

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a + b + c + 2 = a * b * c) :
  (a+1) * (b+1) * (c+1) ≥ 27 ∧ ((a+1) * (b+1) * (c+1) = 27 → a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end problem_statement_l75_75505


namespace simplify_expression_l75_75323

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l75_75323


namespace radar_placement_coverage_l75_75320

noncomputable def max_distance_radars (r : ℝ) (n : ℕ) : ℝ :=
  r / Real.sin (Real.pi / n)

noncomputable def coverage_ring_area (r : ℝ) (width : ℝ) (n : ℕ) : ℝ :=
  (1440 * Real.pi) / Real.tan (Real.pi / n)

theorem radar_placement_coverage :
  let r := 41
  let width := 18
  let n := 7
  max_distance_radars r n = 40 / Real.sin (Real.pi / 7) ∧
  coverage_ring_area r width n = (1440 * Real.pi) / Real.tan (Real.pi / 7) :=
by
  sorry

end radar_placement_coverage_l75_75320


namespace inequality_proof_l75_75761

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l75_75761


namespace line_intersect_yaxis_at_l75_75282

theorem line_intersect_yaxis_at
  (x1 y1 x2 y2 : ℝ) : (x1 = 3) → (y1 = 19) → (x2 = -7) → (y2 = -1) →
  ∃ y : ℝ, (0, y) = (0, 13) :=
by
  intros h1 h2 h3 h4
  sorry

end line_intersect_yaxis_at_l75_75282


namespace find_a_l75_75948

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a (a : ℚ) : F a 2 3 = F a 3 8 → a = -5 / 19 :=
by
  sorry

end find_a_l75_75948


namespace central_projection_intersect_l75_75152

def central_projection (lines : Set (Set Point)) : Prop :=
  ∃ point : Point, ∀ line ∈ lines, line (point)

theorem central_projection_intersect :
  ∀ lines : Set (Set Point), central_projection lines → ∃ point : Point, ∀ line ∈ lines, line (point) :=
by
  sorry

end central_projection_intersect_l75_75152


namespace express_in_scientific_notation_l75_75753

theorem express_in_scientific_notation : (0.0000028 = 2.8 * 10^(-6)) :=
sorry

end express_in_scientific_notation_l75_75753


namespace ceiling_floor_sum_l75_75481

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l75_75481


namespace smallest_result_l75_75265

theorem smallest_result :
  let a := (-2)^3
  let b := (-2) + 3
  let c := (-2) * 3
  let d := (-2) - 3
  a < b ∧ a < c ∧ a < d :=
by
  -- Lean proof steps would go here
  sorry

end smallest_result_l75_75265


namespace gcd_m_n_15_lcm_m_n_45_l75_75073

-- Let m and n be integers greater than 0, and 3m + 2n = 225.
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225)

-- First part: If the greatest common divisor of m and n is 15, then m + n = 105.
theorem gcd_m_n_15 (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Second part: If the least common multiple of m and n is 45, then m + n = 90.
theorem lcm_m_n_45 (h5 : Int.lcm m n = 45) : m + n = 90 :=
sorry

end gcd_m_n_15_lcm_m_n_45_l75_75073


namespace complete_work_in_12_days_l75_75385

def Ravi_rate_per_day : ℚ := 1 / 24
def Prakash_rate_per_day : ℚ := 1 / 40
def Suresh_rate_per_day : ℚ := 1 / 60
def combined_rate_per_day : ℚ := Ravi_rate_per_day + Prakash_rate_per_day + Suresh_rate_per_day

theorem complete_work_in_12_days : 
  (1 / combined_rate_per_day) = 12 := 
by
  sorry

end complete_work_in_12_days_l75_75385


namespace find_B_inter_complement_U_A_l75_75046

-- Define Universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define Set A
def A : Set ℤ := {2, 3}

-- Define complement of A relative to U
def complement_U_A : Set ℤ := U \ A

-- Define set B
def B : Set ℤ := {1, 4}

-- The goal to prove
theorem find_B_inter_complement_U_A : B ∩ complement_U_A = {1, 4} :=
by 
  have h1 : A = {2, 3} := rfl
  have h2 : U = {-1, 0, 1, 2, 3, 4} := rfl
  have h3 : B = {1, 4} := rfl
  sorry

end find_B_inter_complement_U_A_l75_75046


namespace calculate_polynomial_value_l75_75473

theorem calculate_polynomial_value :
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := 
by 
  sorry

end calculate_polynomial_value_l75_75473


namespace area_of_the_region_l75_75944

noncomputable def region_area (C D : ℝ×ℝ) (rC rD : ℝ) (y : ℝ) : ℝ :=
  let rect_area := (D.1 - C.1) * y
  let sector_areaC := (1 / 2) * Real.pi * rC^2
  let sector_areaD := (1 / 2) * Real.pi * rD^2
  rect_area - (sector_areaC + sector_areaD)

theorem area_of_the_region :
  region_area (3, 5) (10, 5) 3 5 5 = 35 - 17 * Real.pi := by
  sorry

end area_of_the_region_l75_75944


namespace commutative_star_l75_75297

def star (a b : ℤ) : ℤ := a^2 + b^2

theorem commutative_star (a b : ℤ) : star a b = star b a :=
by sorry

end commutative_star_l75_75297


namespace units_digit_of_3_pow_1987_l75_75785

theorem units_digit_of_3_pow_1987 : 3 ^ 1987 % 10 = 7 := by
  sorry

end units_digit_of_3_pow_1987_l75_75785


namespace total_pieces_of_pizza_l75_75306

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l75_75306


namespace set_inter_complement_U_B_l75_75855

-- Define sets U, A, B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- Statement to prove
theorem set_inter_complement_U_B :
  A ∩ (Uᶜ \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end set_inter_complement_U_B_l75_75855


namespace stream_speed_l75_75409

def boat_speed_still : ℝ := 30
def distance_downstream : ℝ := 80
def distance_upstream : ℝ := 40

theorem stream_speed (v : ℝ) (h : (distance_downstream / (boat_speed_still + v) = distance_upstream / (boat_speed_still - v))) :
  v = 10 :=
sorry

end stream_speed_l75_75409


namespace goods_train_speed_l75_75270

theorem goods_train_speed (man_train_speed_kmh : Float) 
    (goods_train_length_m : Float) 
    (passing_time_s : Float) 
    (kmh_to_ms : Float := 1000 / 3600) : 
    man_train_speed_kmh = 50 → 
    goods_train_length_m = 280 → 
    passing_time_s = 9 → 
    Float.round ((goods_train_length_m / passing_time_s + man_train_speed_kmh * kmh_to_ms) * 3600 / 1000) = 61.99
:= by
  sorry

end goods_train_speed_l75_75270


namespace mans_speed_upstream_l75_75135

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l75_75135


namespace contrapositive_example_l75_75902

theorem contrapositive_example (x : ℝ) : (x > 1 → x^2 > 1) → (x^2 ≤ 1 → x ≤ 1) :=
sorry

end contrapositive_example_l75_75902


namespace max_sum_of_multiplication_table_l75_75390

-- Define primes and their sums
def primes : List ℕ := [2, 3, 5, 7, 17, 19]

noncomputable def sum_primes := primes.sum -- 2 + 3 + 5 + 7 + 17 + 19 = 53

-- Define two groups of primes to maximize the product of their sums
def group1 : List ℕ := [2, 3, 17]
def group2 : List ℕ := [5, 7, 19]

noncomputable def sum_group1 := group1.sum -- 2 + 3 + 17 = 22
noncomputable def sum_group2 := group2.sum -- 5 + 7 + 19 = 31

-- Formulate the proof problem
theorem max_sum_of_multiplication_table : 
  ∃ a b c d e f : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
    (a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes) ∧ 
    (a + b + c = sum_group1 ∨ a + b + c = sum_group2) ∧ 
    (d + e + f = sum_group1 ∨ d + e + f = sum_group2) ∧ 
    (a + b + c) ≠ (d + e + f) ∧ 
    ((a + b + c) * (d + e + f) = 682) := 
by
  use 2, 3, 17, 5, 7, 19
  sorry

end max_sum_of_multiplication_table_l75_75390


namespace A_more_likely_than_B_l75_75142

-- Define the conditions
variables (n : ℕ) (k : ℕ)
-- n is the total number of programs, k is the chosen number of programs
def total_programs : ℕ := 10
def selected_programs : ℕ := 3
-- Probability of person B correctly completing each program
def probability_B_correct : ℚ := 3/5
-- Person A can correctly complete 6 out of 10 programs
def person_A_correct : ℕ := 6

-- The probability of person B successfully completing the challenge
def probability_B_success : ℚ := (3 * (9/25) * (2/5)) + (27/125)

-- Define binomial coefficient function for easier combination calculations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probabilities for the number of correct programs for person A
def P_X_0 : ℚ := (choose 4 3 : ℕ) / (choose 10 3 : ℕ)
def P_X_1 : ℚ := (choose 6 1 * choose 4 2 : ℕ) / (choose 10 3 : ℕ)
def P_X_2 : ℚ := (choose 6 2 * choose 4 1 : ℕ) / (choose 10 3 : ℕ)
def P_X_3 : ℚ := (choose 6 3 : ℕ) / (choose 10 3 : ℕ)

-- The distribution and expectation of X for person A
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- The probability of person A successfully completing the challenge
def P_A_success : ℚ := P_X_2 + P_X_3

-- Final comparisons to determine who is more likely to succeed
def compare_success : Prop := P_A_success > probability_B_success

-- Lean statement
theorem A_more_likely_than_B : compare_success := by
  sorry

end A_more_likely_than_B_l75_75142


namespace expected_value_of_winnings_l75_75281

theorem expected_value_of_winnings : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probabilities := [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8] -- uniform distribution over an 8-sided die
  let winnings := λ (x : ℕ), if x == 3 then 3
                             else if x == 6 then 6
                             else if x == 2 then -2
                             else if x == 4 then -2
                             else 0 in 
  let expected_value := ∑ i in (finset.range 8).map (function.embedding.mk (λ i, i + 1) sorry),
                           (probabilities[i] * (winnings outcomes[i])) in
  expected_value = 5 / 4 := 
begin
  sorry
end

end expected_value_of_winnings_l75_75281


namespace minimum_black_edges_5x5_l75_75615

noncomputable def minimum_black_edges_on_border (n : ℕ) : ℕ :=
if n = 5 then 5 else 0

theorem minimum_black_edges_5x5 : 
  minimum_black_edges_on_border 5 = 5 :=
by sorry

end minimum_black_edges_5x5_l75_75615


namespace largest_n_divisible_l75_75442

theorem largest_n_divisible (n : ℕ) (h : (n : ℤ) > 0) : 
  (n^3 + 105) % (n + 12) = 0 ↔ n = 93 :=
sorry

end largest_n_divisible_l75_75442


namespace nonoverlapping_unit_squares_in_figure_50_l75_75060

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem nonoverlapping_unit_squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_50_l75_75060


namespace nested_sqrt_eq_two_l75_75354

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by {
    -- Proof skipped
    sorry
}

end nested_sqrt_eq_two_l75_75354


namespace proof_calculate_expr_l75_75471

def calculate_expr : Prop :=
  (4 + 4 + 6) / 3 - 2 / 3 = 4

theorem proof_calculate_expr : calculate_expr := 
by 
  sorry

end proof_calculate_expr_l75_75471


namespace num_foxes_l75_75539

structure Creature :=
  (is_squirrel : Bool)
  (is_fox : Bool)
  (is_salamander : Bool)

def Anna : Creature := sorry
def Bob : Creature := sorry
def Cara : Creature := sorry
def Daniel : Creature := sorry

def tells_truth (c : Creature) : Bool :=
  c.is_squirrel || (c.is_salamander && ¬c.is_fox)

def Anna_statement : Prop := Anna.is_fox ≠ Daniel.is_fox
def Bob_statement : Prop := tells_truth Bob ↔ Cara.is_salamander
def Cara_statement : Prop := tells_truth Cara ↔ Bob.is_fox
def Daniel_statement : Prop := tells_truth Daniel ↔ (Anna.is_squirrel ∧ Bob.is_squirrel ∧ Cara.is_squirrel ∨ Daniel.is_squirrel)

theorem num_foxes :
  (Anna.is_fox + Bob.is_fox + Cara.is_fox + Daniel.is_fox = 2) :=
  sorry

end num_foxes_l75_75539


namespace mary_total_nickels_l75_75552

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l75_75552


namespace supermarkets_in_us_l75_75584

noncomputable def number_of_supermarkets_in_canada : ℕ := 35
noncomputable def number_of_supermarkets_total : ℕ := 84
noncomputable def diff_us_canada : ℕ := 14
noncomputable def number_of_supermarkets_in_us : ℕ := number_of_supermarkets_in_canada + diff_us_canada

theorem supermarkets_in_us : number_of_supermarkets_in_us = 49 := by
  sorry

end supermarkets_in_us_l75_75584


namespace eval_expr_eq_zero_l75_75486

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l75_75486


namespace inradius_inequality_l75_75274

/-- Given a point P inside the triangle ABC, where da, db, and dc are the distances from P to the sides BC, CA, and AB respectively,
 and r is the inradius of the triangle ABC, prove the inequality -/
theorem inradius_inequality (a b c da db dc : ℝ) (r : ℝ) 
  (h1 : 0 < da) (h2 : 0 < db) (h3 : 0 < dc)
  (h4 : r = (a * da + b * db + c * dc) / (a + b + c)) :
  2 / (1 / da + 1 / db + 1 / dc) < r ∧ r < (da + db + dc) / 2 :=
  sorry

end inradius_inequality_l75_75274


namespace n_times_s_l75_75372

noncomputable def f (x : ℝ) : ℝ := sorry

theorem n_times_s : (f 0 = 0 ∨ f 0 = 1) ∧
  (∀ (y : ℝ), f 0 = 0 → False) ∧
  (∀ (x y : ℝ), f x * f y - f (x * y) = x^2 + y^2) → 
  let n : ℕ := if f 0 = 0 then 1 else 1
  let s : ℝ := if f 0 = 0 then 0 else 1
  n * s = 1 :=
by
  sorry

end n_times_s_l75_75372


namespace count_N_less_than_2000_l75_75523

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l75_75523


namespace remainder_of_12111_div_3_l75_75113

theorem remainder_of_12111_div_3 : 12111 % 3 = 0 := by
  sorry

end remainder_of_12111_div_3_l75_75113


namespace values_for_a_l75_75045

def has_two (A : Set ℤ) : Prop :=
  2 ∈ A

def candidate_values (a : ℤ) : Set ℤ :=
  {-2, 2 * a, a * a - a}

theorem values_for_a (a : ℤ) :
  has_two (candidate_values a) ↔ a = 1 ∨ a = 2 :=
by
  sorry

end values_for_a_l75_75045


namespace total_days_stayed_l75_75901

-- Definitions of given conditions as variables
def cost_first_week := 18
def days_first_week := 7
def cost_additional_week := 13
def total_cost := 334

-- Formulation of the target statement in Lean
theorem total_days_stayed :
  (days_first_week + 
  ((total_cost - (days_first_week * cost_first_week)) / cost_additional_week)) = 23 :=
by
  sorry

end total_days_stayed_l75_75901


namespace students_answered_both_questions_correctly_l75_75528

theorem students_answered_both_questions_correctly (P_A P_B P_A'_B' : ℝ) (h_P_A : P_A = 0.75) (h_P_B : P_B = 0.7) (h_P_A'_B' : P_A'_B' = 0.2) :
  ∃ P_A_B : ℝ, P_A_B = 0.65 := 
by
  sorry

end students_answered_both_questions_correctly_l75_75528


namespace four_cells_different_colors_l75_75037

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l75_75037


namespace eval_expression_l75_75023

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l75_75023


namespace fixed_point_of_exponential_function_l75_75965

-- The function definition and conditions are given as hypotheses
theorem fixed_point_of_exponential_function
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, (∀ x : ℝ, (x = 1) → P = (x, a^(x-1) - 2)) → P = (1, -1) :=
by
  sorry

end fixed_point_of_exponential_function_l75_75965


namespace eval_expression_l75_75024

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l75_75024


namespace correct_statements_about_opposite_numbers_l75_75571

/-- Definition of opposite numbers: two numbers are opposite if one is the negative of the other --/
def is_opposite (a b : ℝ) : Prop := a = -b

theorem correct_statements_about_opposite_numbers (a b : ℝ) :
  (is_opposite a b ↔ a + b = 0) ∧
  (a + b = 0 ↔ is_opposite a b) ∧
  ((is_opposite a b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (a / b = -1)) ∧
  ((a / b = -1 ∧ b ≠ 0) ↔ is_opposite a b) :=
by {
  sorry -- Proof is omitted
}

end correct_statements_about_opposite_numbers_l75_75571


namespace even_and_increasing_on_0_inf_l75_75469

noncomputable def fA (x : ℝ) : ℝ := x^(2/3)
noncomputable def fB (x : ℝ) : ℝ := (1/2)^x
noncomputable def fC (x : ℝ) : ℝ := Real.log x
noncomputable def fD (x : ℝ) : ℝ := -x^2 + 1

theorem even_and_increasing_on_0_inf (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) ∧ (∀ a b, (0 < a ∧ a < b) → f a < f b) ↔ f = fA :=
sorry

end even_and_increasing_on_0_inf_l75_75469


namespace Binkie_gemstones_l75_75157

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l75_75157


namespace f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l75_75455

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_2_2_eq_7 : f 2 2 = 7 :=
sorry

theorem f_3_3_eq_61 : f 3 3 = 61 :=
sorry

theorem f_4_4_can_be_evaluated : ∃ n, f 4 4 = n :=
sorry

end f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l75_75455


namespace simplify_fraction_l75_75266

theorem simplify_fraction (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := 
sorry

end simplify_fraction_l75_75266


namespace petya_points_l75_75058

noncomputable def points_after_disqualification : ℕ :=
4

theorem petya_points (players: ℕ) (initial_points: ℕ) (disqualified: ℕ) (new_points: ℕ) : 
  players = 10 → 
  initial_points < (players * (players - 1) / 2) / players → 
  disqualified = 2 → 
  (players - disqualified) * (players - disqualified - 1) / 2 = new_points →
  new_points / (players - disqualified) < points_after_disqualification →
  points_after_disqualification > new_points / (players - disqualified) →
  points_after_disqualification = 4 :=
by 
  intros 
  exact sorry

end petya_points_l75_75058


namespace incorrect_inequality_l75_75860

theorem incorrect_inequality (a b : ℝ) (h : a > b ∧ b > 0) :
  ¬ (1 / a > 1 / b) :=
by
  sorry

end incorrect_inequality_l75_75860


namespace four_cells_different_colors_l75_75036

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l75_75036


namespace fourth_term_geometric_sequence_l75_75241

theorem fourth_term_geometric_sequence (x : ℝ) :
  ∃ r : ℝ, (r > 0) ∧ 
  x ≠ 0 ∧
  (3 * x + 3)^2 = x * (6 * x + 6) →
  x = -3 →
  6 * x + 6 ≠ 0 →
  4 * (6 * x + 6) * (3 * x + 3) = -24 :=
by
  -- Placeholder for the proof steps
  sorry

end fourth_term_geometric_sequence_l75_75241


namespace max_problems_to_miss_to_pass_l75_75146

theorem max_problems_to_miss_to_pass (total_problems : ℕ) (pass_percentage : ℝ) :
  total_problems = 50 → pass_percentage = 0.85 → 7 = ↑total_problems * (1 - pass_percentage) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end max_problems_to_miss_to_pass_l75_75146


namespace ball_beyond_hole_l75_75003

theorem ball_beyond_hole
  (first_turn_distance : ℕ)
  (second_turn_distance : ℕ)
  (total_distance_to_hole : ℕ) :
  first_turn_distance = 180 →
  second_turn_distance = first_turn_distance / 2 →
  total_distance_to_hole = 250 →
  second_turn_distance - (total_distance_to_hole - first_turn_distance) = 20 :=
by
  intros
  -- Proof omitted
  sorry

end ball_beyond_hole_l75_75003


namespace monogram_count_is_correct_l75_75739

def count_possible_monograms : ℕ :=
  Nat.choose 23 2

theorem monogram_count_is_correct : 
  count_possible_monograms = 253 := 
by 
  -- The proof will show this matches the combination formula calculation
  -- The final proof is left incomplete as per the instructions
  sorry

end monogram_count_is_correct_l75_75739


namespace ceil_floor_sum_l75_75480

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l75_75480


namespace polygon_sides_l75_75096

theorem polygon_sides (n : ℕ) (z : ℕ) (h1 : z = n * (n - 3) / 2) (h2 : z = 3 * n) : n = 9 := by
  sorry

end polygon_sides_l75_75096


namespace find_a7_l75_75676

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75676


namespace smallest_positive_omega_l75_75184

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * (x + Real.pi / 4) - Real.pi / 6)

theorem smallest_positive_omega (ω : ℝ) :
  (∀ x : ℝ, g (ω) x = g (ω) (-x)) → (ω = 4 / 3) := sorry

end smallest_positive_omega_l75_75184


namespace winning_candidate_percentage_l75_75987

/-- 
In an election, a candidate won by a majority of 1040 votes out of a total of 5200 votes.
Prove that the winning candidate received 60% of the votes.
-/
theorem winning_candidate_percentage {P : ℝ} (h_majority : (P * 5200) - ((1 - P) * 5200) = 1040) : P = 0.60 := 
by
  sorry

end winning_candidate_percentage_l75_75987


namespace cost_prices_three_watches_l75_75141

theorem cost_prices_three_watches :
  ∃ (C1 C2 C3 : ℝ), 
    (0.9 * C1 + 210 = 1.04 * C1) ∧ 
    (0.85 * C2 + 180 = 1.03 * C2) ∧ 
    (0.95 * C3 + 250 = 1.06 * C3) ∧ 
    C1 = 1500 ∧ 
    C2 = 1000 ∧ 
    C3 = (25000 / 11) :=
by 
  sorry

end cost_prices_three_watches_l75_75141


namespace problem_statement_l75_75716

theorem problem_statement (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : x * (z + t + y) = z * (x + y + t) :=
sorry

end problem_statement_l75_75716


namespace selling_price_of_cycle_l75_75791

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 18

theorem selling_price_of_cycle : 
    (cost_price - (loss_percentage / 100) * cost_price) = 1148 := 
by
  sorry

end selling_price_of_cycle_l75_75791


namespace hyperbola_equation_focus_and_eccentricity_l75_75341

theorem hyperbola_equation_focus_and_eccentricity (a b : ℝ)
  (h_focus : ∃ c : ℝ, c = 1 ∧ (∃ c_squared : ℝ, c_squared = c ^ 2))
  (h_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 ∧ e = c / a)
  (h_b : b ^ 2 = c ^ 2 - a ^ 2) :
  5 * x^2 - (5 / 4) * y^2 = 1 :=
sorry

end hyperbola_equation_focus_and_eccentricity_l75_75341


namespace LittleRedHeightCorrect_l75_75377

noncomputable def LittleRedHeight : ℝ :=
let LittleMingHeight := 1.3 
let HeightDifference := 0.2 
LittleMingHeight - HeightDifference

theorem LittleRedHeightCorrect : LittleRedHeight = 1.1 := by
  sorry

end LittleRedHeightCorrect_l75_75377


namespace pavan_distance_travelled_l75_75610

theorem pavan_distance_travelled (D : ℝ) (h1 : D / 60 + D / 50 = 11) : D = 300 :=
sorry

end pavan_distance_travelled_l75_75610


namespace evaluate_exponent_sum_l75_75953

theorem evaluate_exponent_sum : 
  let i : ℂ := Complex.I in 
  i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end evaluate_exponent_sum_l75_75953


namespace expected_value_is_150_l75_75381

noncomputable def expected_value_of_winnings : ℝ :=
  let p := (1:ℝ)/8
  let winnings := [0, 2, 3, 5, 7]
  let losses := [4, 6]
  let extra := 5
  let win_sum := (winnings.sum : ℝ)
  let loss_sum := (losses.sum : ℝ)
  let E := p * 0 + p * win_sum - p * loss_sum + p * extra
  E

theorem expected_value_is_150 : expected_value_of_winnings = 1.5 := 
by sorry

end expected_value_is_150_l75_75381


namespace true_discount_is_36_l75_75091

noncomputable def calc_true_discount (BD SD : ℝ) : ℝ := BD / (1 + BD / SD)

theorem true_discount_is_36 :
  let BD := 42
  let SD := 252
  calc_true_discount BD SD = 36 := 
by
  -- proof here
  sorry

end true_discount_is_36_l75_75091


namespace sum_of_diffs_is_10_l75_75234

-- Define the number of fruits each person has
def Sharon_plums : ℕ := 7
def Allan_plums : ℕ := 10
def Dave_oranges : ℕ := 12

-- Define the differences in the number of fruits
def diff_Sharon_Allan : ℕ := Allan_plums - Sharon_plums
def diff_Sharon_Dave : ℕ := Dave_oranges - Sharon_plums
def diff_Allan_Dave : ℕ := Dave_oranges - Allan_plums

-- Define the sum of these differences
def sum_of_diffs : ℕ := diff_Sharon_Allan + diff_Sharon_Dave + diff_Allan_Dave

-- State the theorem to be proved
theorem sum_of_diffs_is_10 : sum_of_diffs = 10 := by
  sorry

end sum_of_diffs_is_10_l75_75234


namespace circle_equation_l75_75702

/-- Given that point C is above the x-axis and
    the circle C with center C is tangent to the x-axis at point A(1,0) and
    intersects with circle O: x² + y² = 4 at points P and Q such that
    the length of PQ is sqrt(14)/2, the standard equation of circle C
    is (x - 1)² + (y - 1)² = 1. -/
theorem circle_equation {C : ℝ × ℝ} (hC : C.2 > 0) (tangent_at_A : C = (1, C.2))
  (intersect_with_O : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ ((P.1 - 1)^2 + (P.2 - C.2)^2 = C.2^2) ∧ 
  ((Q.1 - 1)^2 + (Q.2 - C.2)^2 = C.2^2) ∧ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4)) :
  (C.2 = 1) ∧ ((x - 1)^2 + (y - 1)^2 = 1) :=
by
  sorry

end circle_equation_l75_75702


namespace ceiling_plus_floor_eq_zero_l75_75478

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l75_75478


namespace failed_by_35_l75_75925

variables (M S P : ℝ)
variables (hM : M = 153.84615384615384)
variables (hS : S = 45)
variables (hP : P = 0.52 * M)

theorem failed_by_35 (hM : M = 153.84615384615384) (hS : S = 45) (hP : P = 0.52 * M) : P - S = 35 :=
by
  sorry

end failed_by_35_l75_75925


namespace angle_E_in_quadrilateral_l75_75061

theorem angle_E_in_quadrilateral (E F G H : ℝ) 
  (h1 : E = 5 * H)
  (h2 : E = 4 * G)
  (h3 : E = (5/3) * F)
  (h_sum : E + F + G + H = 360) : 
  E = 131 := by 
  sorry

end angle_E_in_quadrilateral_l75_75061


namespace ln_n_lt_8m_l75_75549

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 
  Real.log x - m * x^2 + 2 * n * x

theorem ln_n_lt_8m (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : ∀ x > 0, f x m n ≤ f 1 m n) : 
  Real.log n < 8 * m := 
sorry

end ln_n_lt_8m_l75_75549


namespace ratio_of_u_to_v_l75_75778

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l75_75778


namespace restore_original_salary_l75_75607

theorem restore_original_salary (orig_salary : ℝ) (reducing_percent : ℝ) (increasing_percent : ℝ) :
  reducing_percent = 20 → increasing_percent = 25 →
  (orig_salary * (1 - reducing_percent / 100)) * (1 + increasing_percent / 100 / (1 - reducing_percent / 100)) = orig_salary
:= by
  intros
  sorry

end restore_original_salary_l75_75607


namespace solve_for_c_l75_75181

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

theorem solve_for_c {c : ℝ} (hc : ∀ x ≠ (-3/2), f c (f c x) = x) : c = -3 :=
by
  intros
  -- The proof steps will go here
  sorry

end solve_for_c_l75_75181


namespace largest_positive_integer_solution_l75_75404

theorem largest_positive_integer_solution (x : ℕ) (h₁ : 1 ≤ x) (h₂ : x + 3 ≤ 6) : 
  x = 3 := by
  sorry

end largest_positive_integer_solution_l75_75404


namespace units_digit_sum_base8_l75_75497

theorem units_digit_sum_base8 : 
  let n1 := 53 
  let n2 := 64 
  let sum_base8 := n1 + n2 
  (sum_base8 % 8) = 7 := 
by 
  sorry

end units_digit_sum_base8_l75_75497


namespace laura_park_time_l75_75732

theorem laura_park_time
  (T : ℝ) -- Time spent at the park each trip in hours
  (walk_time : ℝ := 0.5) -- Time spent walking to and from the park each trip in hours
  (trips : ℕ := 6) -- Total number of trips
  (park_time_percentage : ℝ := 0.80) -- Percentage of total time spent at the park
  (total_park_time_eq : trips * T = park_time_percentage * (trips * (T + walk_time))) :
  T = 2 :=
by
  sorry

end laura_park_time_l75_75732


namespace min_value_function_l75_75757

theorem min_value_function (x : ℝ) (h : 1 < x) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y ≥ 3) :=
sorry

end min_value_function_l75_75757


namespace tim_score_in_math_l75_75796

def even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def sum_even_numbers (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem tim_score_in_math : sum_even_numbers even_numbers = 56 := by
  -- Proof steps would be here
  sorry

end tim_score_in_math_l75_75796


namespace simplify_expression_l75_75322

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l75_75322


namespace problem_conditions_l75_75336

theorem problem_conditions (a1 : ℤ) (d : ℚ) (n : ℕ) :
  (a1 + 4 * d = -3) ∧ (5 * (2 * a1 + 4 * d) = 0) → 
  (∀ n, a1 + (n - 1) * d = (3 * (3 - n)) / 2) ∧ 
  (∀ n, (a1 + (n - 1) * d) * (n * (2 * a1 + (n - 1) * d)) / 2 < 0 → n = 4) :=
by
  sorry  -- proof details excluded

end problem_conditions_l75_75336


namespace bisection_method_applies_l75_75915

noncomputable def f (x : ℝ) : ℝ := x^3 + 1.1*x^2 + 0.9*x - 1.4

theorem bisection_method_applies : 
  ∃ (c : ℝ), c ∈ set.Ioo 0 1 ∧ |c - 0.6875| < 0.1 ∧ f c = 0 :=
by
  have h_interval : 0 ∈ set.Icc (0: ℝ) 1 := by norm_num
  have h_f0 : f 0 = -1.4 := by norm_num [f]
  have h_f1 : f 1 = 1.6 := by norm_num [f]
  have h_sign_change : f 0 < 0 ∧ f 1 > 0 := by norm_num [h_f0, h_f1]
  have h_continuous : continuous f := by continuity
  have h_zero_exists : ∃ (c : ℝ), c ∈ set.Ioo 0 1 ∧ f c = 0 := 
    intermediate_value_Icc (set.mem_Icc_of_Ioo h_interval) h_f0 h_f1 h_continuous
  obtain ⟨c, hc₁, hc₂⟩ := h_zero_exists
  use c
  split
  · exact hc₁
  · norm_num at hc₁
    linarith [hc₁]
  · exact hc₂

end bisection_method_applies_l75_75915


namespace distance_is_3_l75_75738

-- define the distance between Masha's and Misha's homes
def distance_between_homes (d : ℝ) : Prop :=
  -- Masha and Misha meet 1 kilometer from Masha's home in the first occasion
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / v_m = (d - 1) / v_i) ∧

  -- On the second occasion, Masha walked at twice her original speed,
  -- and Misha walked at half his original speed, and they met 1 kilometer away from Misha's home.
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / (2 * v_m) = 2 * (d - 1) / (0.5 * v_i))

-- The theorem to prove the distance is 3
theorem distance_is_3 : distance_between_homes 3 :=
  sorry

end distance_is_3_l75_75738


namespace khalil_dogs_l75_75081

theorem khalil_dogs (D : ℕ) (cost_dog cost_cat : ℕ) (num_cats total_cost : ℕ) 
  (h1 : cost_dog = 60)
  (h2 : cost_cat = 40)
  (h3 : num_cats = 60)
  (h4 : total_cost = 3600) :
  (num_cats * cost_cat + D * cost_dog = total_cost) → D = 20 :=
by
  intros h
  sorry

end khalil_dogs_l75_75081


namespace find_a7_l75_75670

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75670


namespace min_seats_to_occupy_l75_75585

theorem min_seats_to_occupy (n : ℕ) (h_n : n = 150) : 
  ∃ (k : ℕ), k = 90 ∧ ∀ m : ℕ, m ≥ k → ∀ i : ℕ, i < n → ∃ j : ℕ, (j < n) ∧ ((j = i + 1) ∨ (j = i - 1)) :=
sorry

end min_seats_to_occupy_l75_75585


namespace find_a_l75_75966

-- Define the constants b and the asymptote equation
def asymptote_eq (x y : ℝ) := 3 * x + 2 * y = 0

-- Define the hyperbola equation and the condition
def hyperbola_eq (x y a : ℝ) := x^2 / a^2 - y^2 / 9 = 1
def hyperbola_condition (a : ℝ) := a > 0

-- Theorem stating the value of a given the conditions
theorem find_a (a : ℝ) (hcond : hyperbola_condition a) 
  (h_asymp : ∀ x y : ℝ, asymptote_eq x y → y = -(3/2) * x) :
  a = 2 := 
sorry

end find_a_l75_75966


namespace simplify_expression_l75_75328

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l75_75328


namespace chocolates_cost_l75_75118

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost_l75_75118


namespace divisor_of_635_l75_75231

theorem divisor_of_635 (p : ℕ) (h1 : Nat.Prime p) (k : ℕ) (h2 : 635 = 7 * k * p + 11) : p = 89 :=
sorry

end divisor_of_635_l75_75231


namespace problem_equivalent_proof_l75_75693

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75693


namespace larger_number_is_23_l75_75418

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75418


namespace necessary_not_sufficient_l75_75961

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end necessary_not_sufficient_l75_75961


namespace expected_value_of_winnings_after_one_flip_l75_75119

-- Definitions based on conditions from part a)
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def win_heads : ℚ := 3
def lose_tails : ℚ := -2

-- The statement to prove:
theorem expected_value_of_winnings_after_one_flip :
  prob_heads * win_heads + prob_tails * lose_tails = -1 / 3 :=
by
  sorry

end expected_value_of_winnings_after_one_flip_l75_75119


namespace triangle_perimeter_l75_75723

theorem triangle_perimeter (x : ℕ) (hx1 : x % 2 = 1) (hx2 : 5 < x) (hx3 : x < 11) : 
  (3 + 8 + x = 18) ∨ (3 + 8 + x = 20) :=
sorry

end triangle_perimeter_l75_75723


namespace find_area_triangle_boc_l75_75065

noncomputable def area_ΔBOC := 21

theorem find_area_triangle_boc (A B C K O : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup K] [NormedAddCommGroup O]
  (AC : ℝ) (AB : ℝ) (h1 : AC = 14) (h2 : AB = 6)
  (circle_centered_on_AC : Prop)
  (K_on_BC : Prop)
  (angle_BAK_eq_angle_ACB : Prop)
  (midpoint_O_AC : Prop)
  (angle_AKC_eq_90 : Prop)
  (area_ABC : Prop) : 
  area_ΔBOC = 21 := 
sorry

end find_area_triangle_boc_l75_75065


namespace find_m_l75_75863

theorem find_m (m : ℝ) (h : (1 : ℝ) ^ 2 - m * (1 : ℝ) + 2 = 0) : m = 3 :=
by
  sorry

end find_m_l75_75863


namespace people_who_own_neither_l75_75247

theorem people_who_own_neither (total_people cat_owners cat_and_dog_owners dog_owners non_cat_dog_owners: ℕ)
        (h1: total_people = 522)
        (h2: 20 * cat_and_dog_owners = cat_owners)
        (h3: 7 * dog_owners = 10 * (dog_owners + cat_and_dog_owners))
        (h4: 2 * non_cat_dog_owners = (non_cat_dog_owners + dog_owners)):
    non_cat_dog_owners = 126 := 
by
  sorry

end people_who_own_neither_l75_75247


namespace find_a7_l75_75655

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75655


namespace find_line_equation_l75_75100

theorem find_line_equation (k m b : ℝ) :
  (∃ k, |(k^2 + 7*k + 10) - (m*k + b)| = 8) ∧ (8 = 2*m + b) ∧ (b ≠ 0) → (m = 5 ∧ b = 3) := 
by
  intro h
  sorry

end find_line_equation_l75_75100


namespace number_of_odd_blue_faces_cubes_l75_75288

/-
A wooden block is 5 inches long, 5 inches wide, and 1 inch high.
The block is painted blue on all six sides and then cut into twenty-five 1 inch cubes.
Prove that the number of cubes each have a total number of blue faces that is an odd number is 9.
-/

def cubes_with_odd_blue_faces : ℕ :=
  let corner_cubes := 4
  let edge_cubes_not_corners := 16
  let center_cubes := 5
  corner_cubes + center_cubes

theorem number_of_odd_blue_faces_cubes : cubes_with_odd_blue_faces = 9 := by
  have h1 : cubes_with_odd_blue_faces = 4 + 5 := sorry
  have h2 : 4 + 5 = 9 := by norm_num
  exact Eq.trans h1 h2

end number_of_odd_blue_faces_cubes_l75_75288


namespace speeds_and_time_l75_75268

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end speeds_and_time_l75_75268


namespace dice_sum_not_18_l75_75252

theorem dice_sum_not_18 (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) 
    (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (h_prod : d1 * d2 * d3 * d4 = 144) : 
    d1 + d2 + d3 + d4 ≠ 18 := 
sorry

end dice_sum_not_18_l75_75252


namespace determine_s_l75_75825

theorem determine_s 
  (s : ℝ) 
  (h : (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
       6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30) : 
  s = 4 :=
by
  sorry

end determine_s_l75_75825


namespace gcd_90_270_l75_75834

theorem gcd_90_270 : Int.gcd 90 270 = 90 :=
by
  sorry

end gcd_90_270_l75_75834


namespace determinant_nonnegative_of_skew_symmetric_matrix_l75_75562

theorem determinant_nonnegative_of_skew_symmetric_matrix
  (a b c d e f : ℝ)
  (A : Matrix (Fin 4) (Fin 4) ℝ)
  (hA : A = ![
    ![0, a, b, c],
    ![-a, 0, d, e],
    ![-b, -d, 0, f],
    ![-c, -e, -f, 0]]) :
  0 ≤ Matrix.det A := by
  sorry

end determinant_nonnegative_of_skew_symmetric_matrix_l75_75562


namespace inequality_proof_l75_75171

def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

theorem inequality_proof : c > b ∧ b > a := by
  -- Proof omitted
  sorry

end inequality_proof_l75_75171


namespace system_of_equations_solution_l75_75644

theorem system_of_equations_solution (x y z : ℝ) 
  (h : ∀ (n : ℕ), x * (1 - 1 / 2^(n : ℝ)) + y * (1 - 1 / 2^(n+1 : ℝ)) + z * (1 - 1 / 2^(n+2 : ℝ)) = 0) : 
  y = -3 * x ∧ z = 2 * x :=
sorry

end system_of_equations_solution_l75_75644


namespace inequality_proof_l75_75760

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l75_75760


namespace det_example_1_simplified_form_det_at_4_l75_75441

-- Definition for second-order determinant
def second_order_determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Part (1)
theorem det_example_1 :
  second_order_determinant 3 (-2) 4 (-3) = -1 :=
by
  sorry

-- Part (2) simplified determinant
def simplified_det (x : ℤ) : ℤ :=
  second_order_determinant (2 * x - 3) (x + 2) 2 4

-- Proving simplified determinant form
theorem simplified_form :
  ∀ x : ℤ, simplified_det x = 6 * x - 16 :=
by
  sorry

-- Proving specific case when x = 4
theorem det_at_4 :
  simplified_det 4 = 8 :=
by 
  sorry

end det_example_1_simplified_form_det_at_4_l75_75441


namespace y_divides_x_squared_l75_75734

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared_l75_75734


namespace ratio_of_u_to_v_l75_75776

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l75_75776


namespace transistors_in_2010_l75_75226

theorem transistors_in_2010 (initial_transistors: ℕ) 
    (doubling_period_years: ℕ) (start_year: ℕ) (end_year: ℕ) 
    (h_initial: initial_transistors = 500000)
    (h_period: doubling_period_years = 2) 
    (h_start: start_year = 1992) 
    (h_end: end_year = 2010) :
  let years_passed := end_year - start_year
  let number_of_doublings := years_passed / doubling_period_years
  let transistors_in_end_year := initial_transistors * 2^number_of_doublings
  transistors_in_end_year = 256000000 := by
    sorry

end transistors_in_2010_l75_75226


namespace trigonometric_identity_l75_75850

noncomputable def point_on_terminal_side (x y : ℝ) : Prop :=
    ∃ α : ℝ, x = Real.cos α ∧ y = Real.sin α

theorem trigonometric_identity (x y : ℝ) (h : point_on_terminal_side 1 3) :
    (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by
  sorry

end trigonometric_identity_l75_75850


namespace total_value_of_treats_l75_75302

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l75_75302


namespace turtles_remaining_on_log_l75_75125
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l75_75125


namespace ellipse_properties_l75_75510

-- Define the ellipse E with its given properties
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define properties related to the intersection points and lines
def intersects (l : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  l (-1) = 0 ∧ 
  is_ellipse x₁ (l x₁) ∧ 
  is_ellipse x₂ (l x₂) ∧ 
  y₁ = l x₁ ∧ 
  y₂ = l x₂

def perpendicular_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∀ x, l1 x * l2 x = -1

-- Define the main theorem
theorem ellipse_properties :
  (∀ (x y : ℝ), is_ellipse x y) →
  (∀ (l1 l2 : ℝ → ℝ) 
     (A B C D : ℝ × ℝ),
      intersects l1 A.1 A.2 B.1 B.2 → 
      intersects l2 C.1 C.2 D.1 D.2 → 
      perpendicular_lines l1 l2 → 
      12 * (|A.1 - B.1| + |C.1 - D.1|) = 7 * |A.1 - B.1| * |C.1 - D.1|) :=
by 
  sorry

end ellipse_properties_l75_75510


namespace tan_monotonic_increasing_interval_l75_75406

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * Real.pi - (5 * Real.pi) / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3 }

theorem tan_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (y = Real.tan ((x / 2) + (Real.pi / 3))) → 
           x ∈ monotonic_increasing_interval k :=
sorry

end tan_monotonic_increasing_interval_l75_75406


namespace employed_females_percentage_l75_75206

theorem employed_females_percentage (total_population_percent employed_population_percent employed_males_percent : ℝ) :
  employed_population_percent = 70 → employed_males_percent = 21 →
  (employed_population_percent - employed_males_percent) / employed_population_percent * 100 = 70 :=
by
  -- Assume the total population percentage is 100%, which allows us to work directly with percentages.
  let employed_population_percent := 70
  let employed_males_percent := 21
  sorry

end employed_females_percentage_l75_75206


namespace sum_of_nine_consecutive_even_integers_mod_10_l75_75444

theorem sum_of_nine_consecutive_even_integers_mod_10 : 
  (10112 + 10114 + 10116 + 10118 + 10120 + 10122 + 10124 + 10126 + 10128) % 10 = 0 := by
  sorry

end sum_of_nine_consecutive_even_integers_mod_10_l75_75444


namespace gym_distance_l75_75569

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end gym_distance_l75_75569


namespace rectangle_area_l75_75976

variable (a b : ℝ)

-- Given conditions
axiom h1 : (a + b)^2 = 16 
axiom h2 : (a - b)^2 = 4

-- Objective: Prove that the area of the rectangle ab equals 3
theorem rectangle_area : a * b = 3 := by
  sorry

end rectangle_area_l75_75976


namespace no_nat_numbers_satisfy_lcm_eq_l75_75750

theorem no_nat_numbers_satisfy_lcm_eq (n m : ℕ) :
  ¬ (Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019) :=
sorry

end no_nat_numbers_satisfy_lcm_eq_l75_75750


namespace g_is_zero_l75_75472

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.sin x)^4 + (Real.cos x)^2) - 
  Real.sqrt (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem g_is_zero (x : ℝ) : g x = 0 := 
  sorry

end g_is_zero_l75_75472


namespace solve_for_x_l75_75391

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.8) : x = 71.7647 := 
by 
  sorry

end solve_for_x_l75_75391


namespace find_omega_l75_75707

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem find_omega (ω : ℝ) (h₁ : ∀ x₁ x₂, (-ω < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * ω) → f ω x₁ < f ω x₂)
  (h₂ : ∀ x, f ω x = f ω (-2 * ω - x)) :
  ω = Real.sqrt (3 * Real.pi) / 3 :=
by
  sorry

end find_omega_l75_75707


namespace arithmetic_sequence_sum_l75_75363

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : a 1 = -2012)
  (h₂ : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1)))
  (h₃ : (S 12) / 12 - (S 10) / 10 = 2) :
  S 2012 = -2012 := by
  sorry

end arithmetic_sequence_sum_l75_75363


namespace product_of_abcd_l75_75789

theorem product_of_abcd :
  ∃ (a b c d : ℚ), 
    3 * a + 4 * b + 6 * c + 8 * d = 42 ∧ 
    4 * (d + c) = b ∧ 
    4 * b + 2 * c = a ∧ 
    c - 2 = d ∧ 
    a * b * c * d = (367 * 76 * 93 * -55) / (37^2 * 74^2) :=
sorry

end product_of_abcd_l75_75789


namespace trigonometric_identity_proof_l75_75747

theorem trigonometric_identity_proof 
  (α β γ : ℝ) (a b c : ℝ)
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (hc : 0 < c)
  (hb : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (ha : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 :=
by
  sorry

end trigonometric_identity_proof_l75_75747


namespace Jessie_lost_7_kilograms_l75_75545

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end Jessie_lost_7_kilograms_l75_75545


namespace find_a7_l75_75686

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75686


namespace sqrt_product_simplification_l75_75821

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end sqrt_product_simplification_l75_75821


namespace odd_indexed_convergents_same_for_3_and_4_l75_75155

-- Definitions used in Lean 4 Statement
noncomputable def continued_fraction (x: ℝ) : Stream ℚ := sorry

def convergent (cf: Stream ℚ) (n: ℕ) : ℚ := sorry

def sqrt_continued_fraction (c: ℕ) : Stream ℚ :=
  continued_fraction (Real.sqrt (c^2 + c))

-- Assertions based on given conditions
axiom nat_c : c ∈ ℕ

-- Lean 4 statement of the mathematically equivalent problem
theorem odd_indexed_convergents_same_for_3_and_4 :
  (∀ n, n % 2 = 1 → convergent (sqrt_continued_fraction 3) n = convergent (sqrt_continued_fraction 4) n) :=
sorry

end odd_indexed_convergents_same_for_3_and_4_l75_75155


namespace max_good_diagonals_l75_75868

def is_good_diagonal (n : ℕ) (d : ℕ) : Prop := ∀ (P : Fin n → Prop), ∃! (i j : Fin n), P i ∧ P j ∧ (d = i + j)

theorem max_good_diagonals (n : ℕ) (h : 2 ≤ n) :
  (∃ (m : ℕ), is_good_diagonal n m ∧ (m = n - 2 ↔ Even n) ∧ (m = n - 3 ↔ Odd n)) :=
by
  sorry

end max_good_diagonals_l75_75868


namespace number_of_chickens_free_ranging_l75_75768

-- Defining the conditions
def chickens_in_coop : ℕ := 14
def chickens_in_run (coop_chickens : ℕ) : ℕ := 2 * coop_chickens
def chickens_free_ranging (run_chickens : ℕ) : ℕ := 2 * run_chickens - 4

-- Proving the number of chickens free ranging
theorem number_of_chickens_free_ranging : chickens_free_ranging (chickens_in_run chickens_in_coop) = 52 := by
  -- Lean will be able to infer
  sorry  -- proof is not required

end number_of_chickens_free_ranging_l75_75768


namespace pq_iff_cond_l75_75334

def p (a : ℝ) := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem pq_iff_cond (a : ℝ) : (p a ∧ q a) ↔ (a ≤ -2 ∨ a = 1) := 
by
  sorry

end pq_iff_cond_l75_75334


namespace balance_balls_l75_75230

variable (R O B P : ℝ)

-- Conditions based on the problem statement
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 7.5 * B
axiom h3 : 8 * B = 6 * P

-- The theorem we need to prove
theorem balance_balls : 5 * R + 3 * O + 3 * P = 21.5 * B :=
by 
  sorry

end balance_balls_l75_75230


namespace solve_problem_l75_75733

-- Let S be the set of words W=w_1w_2…w_n of length n from {x,y,z}
-- Define two words U and V to be similar if we can insert a string "xyz", "yzx", or "zxy"
-- into U to obtain V or into V to obtain U.
-- A word W is trivial if there is a sequence W0=λ, W1, ..., Wm such that W_i and W_(i+1) are similar.
def is_trivial (w : List Char) : Prop :=
  ∃ (m : ℕ) (words : Fin (m + 1) → List Char), words 0 = [] ∧ words m = w ∧
    ∀ i : Fin m, (words (Fin.mk (i + 1) sorry)).isSimilarTo (words i) 

-- The function f(n) specifies the number of trivial words of length 3n.
def f (n : ℕ) : ℕ := if n = 0 then 1 else sorry

-- Define the series sum which equals to p / q for relatively prime p and q.
def series_sum := ∑' (n : ℕ), (f n) * (225 / 8192 : ℚ) ^ n 

noncomputable def p := 32
noncomputable def q := 29

theorem solve_problem : series_sum = p / q ∧ Nat.coprime p q ∧ (p + q) = 61 :=
by
  sorry

end solve_problem_l75_75733


namespace money_left_after_expenses_l75_75461

theorem money_left_after_expenses :
  let salary := 8123.08
  let food_expense := (1:ℝ) / 3 * salary
  let rent_expense := (1:ℝ) / 4 * salary
  let clothes_expense := (1:ℝ) / 5 * salary
  let total_expense := food_expense + rent_expense + clothes_expense
  let money_left := salary - total_expense
  money_left = 1759.00 :=
sorry

end money_left_after_expenses_l75_75461


namespace angle_CDE_gt_45_l75_75083

open Triangle

theorem angle_CDE_gt_45 
  (A B C D E : Point)
  (hABC_acute : isAcuteTriangle A B C)
  (hBE_bisector : isInternalAngleBisector A B C E)
  (hAD_altitude : isAltitude A D B C) :
  ∠CDE > 45 :=
sorry

end angle_CDE_gt_45_l75_75083


namespace no_intersection_points_l75_75646

theorem no_intersection_points : ¬ ∃ x y : ℝ, y = x ∧ y = x - 2 := by
  sorry

end no_intersection_points_l75_75646


namespace eval_expression_l75_75021

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l75_75021


namespace min_value_of_expression_l75_75511

theorem min_value_of_expression (x y : ℝ) (hposx : x > 0) (hposy : y > 0) (heq : 2 / x + 1 / y = 1) : 
  x + 2 * y ≥ 8 :=
sorry

end min_value_of_expression_l75_75511


namespace stratified_sampling_l75_75867

-- We are defining the data given in the problem
def numStudents : ℕ := 50
def numFemales : ℕ := 20
def sampledFemales : ℕ := 4
def genderRatio := (numFemales : ℚ) / (numStudents : ℚ)

-- The theorem stating the given problem and its conclusion
theorem stratified_sampling : ∀ (n : ℕ), (sampledFemales : ℚ) / (n : ℚ) = genderRatio → n = 10 :=
by
  intro n
  intro h
  sorry

end stratified_sampling_l75_75867


namespace left_handed_classical_music_lovers_l75_75912

-- Define the conditions
variables (total_people left_handed classical_music right_handed_dislike : ℕ)
variables (x : ℕ) -- x will represent the number of left-handed classical music lovers

-- State the assumptions based on conditions
axiom h1 : total_people = 30
axiom h2 : left_handed = 12
axiom h3 : classical_music = 20
axiom h4 : right_handed_dislike = 3
axiom h5 : 30 = x + (12 - x) + (20 - x) + 3

-- State the theorem to prove
theorem left_handed_classical_music_lovers : x = 5 :=
by {
  -- Skip the proof using sorry
  sorry
}

end left_handed_classical_music_lovers_l75_75912


namespace inverse_of_original_l75_75400

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l75_75400


namespace length_LL1_l75_75725

theorem length_LL1 (XZ : ℝ) (XY : ℝ) (YZ : ℝ) (X1Y : ℝ) (X1Z : ℝ) (LM : ℝ) (LN : ℝ) (MN : ℝ) (L1N : ℝ) (LL1 : ℝ) : 
  XZ = 13 → XY = 5 → 
  YZ = Real.sqrt (XZ^2 - XY^2) → 
  X1Y = 60 / 17 → 
  X1Z = 84 / 17 → 
  LM = X1Z → LN = X1Y → 
  MN = Real.sqrt (LM^2 - LN^2) → 
  (∀ k, L1N = 5 * k ∧ (7 * k + 5 * k) = MN → LL1 = 5 * k) →
  LL1 = 20 / 17 :=
by sorry

end length_LL1_l75_75725


namespace arithmetic_sequence_a10_l75_75507

variable {a : ℕ → ℝ}

-- Given the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

-- Conditions
theorem arithmetic_sequence_a10 (h_arith : is_arithmetic_sequence a) 
                                (h1 : a 6 + a 8 = 16)
                                (h2 : a 4 = 1) :
  a 10 = 15 :=
sorry

end arithmetic_sequence_a10_l75_75507


namespace peanut_cluster_percentage_l75_75010

def chocolates_total : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := chocolates_total - caramels - nougats - truffles

theorem peanut_cluster_percentage : 
  (peanut_clusters / chocolates_total.to_real * 100) = 64 := 
by 
  sorry

end peanut_cluster_percentage_l75_75010


namespace find_fraction_l75_75795

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction_l75_75795


namespace simplify_expr_l75_75563

-- Define the terms
def a : ℕ := 2 ^ 10
def b : ℕ := 5 ^ 6

-- Define the expression we need to simplify
def expr := (a * b : ℝ)^(1/3)

-- Define the simplified form
def c : ℕ := 200
def d : ℕ := 2
def simplified_expr := (c : ℝ) * (d : ℝ)^(1/3)

-- The statement we need to prove
theorem simplify_expr : expr = simplified_expr ∧ (c + d = 202) := by
  sorry

end simplify_expr_l75_75563


namespace complex_pure_imaginary_l75_75567

theorem complex_pure_imaginary (m : ℝ) :
  ((m^2 - 2 * m - 3) = 0) ∧ ((m^2 - 4 * m + 3) ≠ 0) → m = -1 :=
by
  sorry

end complex_pure_imaginary_l75_75567


namespace Tahir_contribution_l75_75898

theorem Tahir_contribution
  (headphone_cost : ℕ := 200)
  (kenji_yen : ℕ := 15000)
  (exchange_rate : ℕ := 100)
  (kenji_contribution : ℕ := kenji_yen / exchange_rate)
  (tahir_contribution : ℕ := headphone_cost - kenji_contribution) :
  tahir_contribution = 50 := 
  by sorry

end Tahir_contribution_l75_75898


namespace cost_per_page_l75_75210

theorem cost_per_page
  (num_notebooks : ℕ)
  (pages_per_notebook : ℕ)
  (total_dollars_paid : ℕ)
  (h1 : num_notebooks = 2)
  (h2 : pages_per_notebook = 50)
  (h3 : total_dollars_paid = 5) :
  (total_dollars_paid * 100) / (num_notebooks * pages_per_notebook) = 5 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cost_per_page_l75_75210


namespace equivalent_modulo_l75_75896

theorem equivalent_modulo:
  123^2 * 947 % 60 = 3 :=
by
  sorry

end equivalent_modulo_l75_75896


namespace proof_problem_l75_75967

def U : Set ℤ := {x | x^2 - x - 12 ≤ 0}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {0, 1, 3, 4}

theorem proof_problem : (U \ A) ∩ B = {0, 1, 4} := 
by sorry

end proof_problem_l75_75967


namespace opponents_team_points_l75_75993

theorem opponents_team_points (M D V O : ℕ) (hM : M = 5) (hD : D = 3) 
    (hV : V = 2 * (M + D)) (hO : O = (M + D + V) + 16) : O = 40 := by
  sorry

end opponents_team_points_l75_75993


namespace value_of_f_m_minus_1_pos_l75_75375

variable (a m : ℝ)
variable (f : ℝ → ℝ)
variable (a_pos : a > 0)
variable (fm_neg : f m < 0)
variable (f_def : ∀ x, f x = x^2 - x + a)

theorem value_of_f_m_minus_1_pos : f (m - 1) > 0 :=
by
  sorry

end value_of_f_m_minus_1_pos_l75_75375


namespace selected_numbers_count_l75_75728

noncomputable def check_num_of_selected_numbers : ℕ := 
  let n := 2015
  let max_num := n * n
  let common_difference := 15
  let starting_number := 14
  let count := (max_num - starting_number) / common_difference + 1
  count

theorem selected_numbers_count : check_num_of_selected_numbers = 270681 := by
  -- Skipping the actual proof
  sorry

end selected_numbers_count_l75_75728


namespace moles_of_CO2_formed_l75_75314

-- Definitions based on the conditions provided
def moles_HNO3 := 2
def moles_NaHCO3 := 2
def balanced_eq (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = NaHCO3 ∧ NaNO3 = NaHCO3 ∧ CO2 = NaHCO3 ∧ H2O = NaHCO3

-- Lean Proposition: Prove that 2 moles of CO2 are formed
theorem moles_of_CO2_formed :
  balanced_eq moles_HNO3 moles_NaHCO3 moles_HNO3 moles_HNO3 moles_HNO3 →
  ∃ CO2, CO2 = 2 :=
by
  sorry

end moles_of_CO2_formed_l75_75314


namespace integer_solution_unique_l75_75718

theorem integer_solution_unique (n : ℤ) : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 3) ↔ n = 5 :=
by
  sorry

end integer_solution_unique_l75_75718


namespace ratio_S7_S3_l75_75840

variable {a_n : ℕ → ℕ} -- Arithmetic sequence {a_n}
variable (S_n : ℕ → ℕ) -- Sum of the first n terms of the arithmetic sequence

-- Conditions
def ratio_a2_a4 (a_2 a_4 : ℕ) : Prop := a_2 = 7 * (a_4 / 6)
def sum_formula (n a_1 d : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Proof goal
theorem ratio_S7_S3 (a_1 d : ℕ) (h : ratio_a2_a4 (a_1 + d) (a_1 + 3 * d)): 
  (S_n 7 = sum_formula 7 a_1 d) ∧ (S_n 3 = sum_formula 3 a_1 d) →
  (S_n 7 / S_n 3 = 2) :=
by
  sorry

end ratio_S7_S3_l75_75840


namespace time_addition_correct_l75_75876

theorem time_addition_correct :
  let current_time := (3, 0, 0)  -- Representing 3:00:00 PM as a tuple (hours, minutes, seconds)
  let duration := (313, 45, 56)  -- Duration to be added: 313 hours, 45 minutes, and 56 seconds
  let new_time := ((3 + (313 % 12) + 45 / 60 + (56 / 3600)), (0 + 45 % 60), (0 + 56 % 60))
  let A := (4 : ℕ)  -- Extracted hour part of new_time
  let B := (45 : ℕ)  -- Extracted minute part of new_time
  let C := (56 : ℕ)  -- Extracted second part of new_time
  A + B + C = 105 := 
by
  -- Placeholder for the actual proof.
  sorry

end time_addition_correct_l75_75876


namespace geometric_sequence_problem_l75_75179

theorem geometric_sequence_problem
  (q : ℝ) (h_q : |q| ≠ 1) (m : ℕ)
  (a : ℕ → ℝ)
  (h_a1 : a 1 = -1)
  (h_am : a m = a 1 * a 2 * a 3 * a 4 * a 5) 
  (h_gseq : ∀ n, a (n + 1) = a n * q) :
  m = 11 :=
by
  sorry

end geometric_sequence_problem_l75_75179


namespace trey_more_turtles_than_kristen_l75_75914

theorem trey_more_turtles_than_kristen (kristen_turtles : ℕ) 
  (H1 : kristen_turtles = 12) 
  (H2 : ∀ kris_turtles, kris_turtles = (1 / 4) * kristen_turtles)
  (H3 : ∀ kris_turtles trey_turtles, trey_turtles = 7 * kris_turtles) :
  ∃ trey_turtles, trey_turtles - kristen_turtles = 9 :=
by {
  sorry
}

end trey_more_turtles_than_kristen_l75_75914


namespace find_c_plus_d_l75_75742

-- Conditions as definitions
variables {P A C : Point }
variables {O₁ O₂ : Point}
variables {AB AP CP : ℝ}
variables {c d : ℕ}

-- Given conditions
def Point_on_diagonal (P A C : Point) : Prop := true -- We need to code the detailed properties of being on the diagonal
def circumcenter_of_triangle (P Q R O : Point) : Prop := true -- We need to code the properties of being a circumcenter
def AP_greater_than_CP (AP CP : ℝ) : Prop := AP > CP
def angle_right (A B O : Point) : Prop := true -- Define the right angle property

-- Main statement to prove
theorem find_c_plus_d : 
  Point_on_diagonal P A C ∧
  circumcenter_of_triangle A B P O₁ ∧ 
  circumcenter_of_triangle C D P O₂ ∧ 
  AP_greater_than_CP AP CP ∧
  AB = 10 ∧
  angle_right O₁ P O₂ ∧
  (AP = Real.sqrt c + Real.sqrt d) →
  (c + d = 100) :=
by
  sorry

end find_c_plus_d_l75_75742


namespace cost_per_remaining_ticket_is_seven_l75_75751

def total_tickets : ℕ := 29
def nine_dollar_tickets : ℕ := 11
def total_cost : ℕ := 225
def nine_dollar_ticket_cost : ℕ := 9
def remaining_tickets : ℕ := total_tickets - nine_dollar_tickets

theorem cost_per_remaining_ticket_is_seven :
  (total_cost - nine_dollar_tickets * nine_dollar_ticket_cost) / remaining_tickets = 7 :=
  sorry

end cost_per_remaining_ticket_is_seven_l75_75751


namespace length_of_CB_l75_75721

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end length_of_CB_l75_75721


namespace combination_problem_l75_75357

theorem combination_problem (x : ℕ) (hx_pos : 0 < x) (h_comb : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 :=
by {
  sorry
}

end combination_problem_l75_75357


namespace sqrt_16_l75_75578

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} :=
by
  sorry

end sqrt_16_l75_75578


namespace solve_for_y_l75_75340

theorem solve_for_y (y : ℤ) (h : (8 + 12 + 23 + 17 + y) / 5 = 15) : y = 15 :=
by {
  sorry
}

end solve_for_y_l75_75340


namespace turtles_remaining_on_log_l75_75126

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l75_75126


namespace cube_without_lid_configurations_l75_75803

-- Introduce assumption for cube without a lid
structure CubeWithoutLid

-- Define the proof statement
theorem cube_without_lid_configurations : 
  ∃ (configs : Nat), (configs = 8) :=
by
  sorry

end cube_without_lid_configurations_l75_75803


namespace ceil_floor_sum_l75_75479

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l75_75479


namespace find_speed_of_car_y_l75_75634

noncomputable def average_speed_of_car_y (sₓ : ℝ) (delay : ℝ) (d_afterₓ_started : ℝ) : ℝ :=
  let tₓ_before := delay
  let dₓ_before := sₓ * tₓ_before
  let total_dₓ := dₓ_before + d_afterₓ_started
  let tₓ_after := d_afterₓ_started / sₓ
  let total_time_y := tₓ_after
  d_afterₓ_started / total_time_y

theorem find_speed_of_car_y (h₁ : ∀ t, t = 1.2) (h₂ : ∀ sₓ, sₓ = 35) (h₃ : ∀ d_afterₓ_started, d_afterₓ_started = 42) : 
  average_speed_of_car_y 35 1.2 42 = 35 := by
  unfold average_speed_of_car_y
  simp
  sorry

end find_speed_of_car_y_l75_75634


namespace instantaneous_velocity_at_t2_l75_75964

noncomputable def displacement (t : ℝ) : ℝ := t^2 * Real.exp (t - 2)

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2 = 8) :=
by
  sorry

end instantaneous_velocity_at_t2_l75_75964


namespace games_lost_l75_75744

theorem games_lost (total_games won_games : ℕ) (h_total : total_games = 12) (h_won : won_games = 8) :
  (total_games - won_games) = 4 :=
by
  -- Placeholder for the proof
  sorry

end games_lost_l75_75744


namespace find_a7_l75_75685

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75685


namespace larger_number_is_23_l75_75414

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75414


namespace find_a7_l75_75678

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75678


namespace problem1_correct_problem2_correct_l75_75151

noncomputable def problem1 : Real :=
  2 * Real.sqrt (2 / 3) - 3 * Real.sqrt (3 / 2) + Real.sqrt 24

theorem problem1_correct : problem1 = (7 * Real.sqrt 6) / 6 := by
  sorry

noncomputable def problem2 : Real :=
  Real.sqrt (25 / 2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2

theorem problem2_correct : problem2 = (11 * Real.sqrt 2) / 2 - 3 := by
  sorry

end problem1_correct_problem2_correct_l75_75151


namespace problem_1_problem_2_problem_3_l75_75943

open Real

theorem problem_1 : (1 * (-12)) - (-20) + (-8) - 15 = -15 := by
  sorry

theorem problem_2 : -3^2 + ((2/3) - (1/2) + (5/8)) * (-24) = -28 := by
  sorry

theorem problem_3 : -1^(2023) + 3 * (-2)^2 - (-6) / ((-1/3)^2) = 65 := by
  sorry

end problem_1_problem_2_problem_3_l75_75943


namespace cubic_polynomial_roots_product_l75_75072

theorem cubic_polynomial_roots_product :
  (∃ a b c : ℝ, (3*a^3 - 9*a^2 + 5*a - 15 = 0) ∧
               (3*b^3 - 9*b^2 + 5*b - 15 = 0) ∧
               (3*c^3 - 9*c^2 + 5*c - 15 = 0) ∧
               a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  ∃ a b c : ℝ, (3*a*b*c = 5) := 
sorry

end cubic_polynomial_roots_product_l75_75072


namespace necklace_count_17_beads_l75_75351

theorem necklace_count_17_beads :
  let n := 17 in
  let circular_permutations := (n - 1)! in
  let reflection_symmetry := 2 in
  circular_permutations / reflection_symmetry = 8 * 15! := 
by
  sorry

end necklace_count_17_beads_l75_75351


namespace yellow_ball_percentage_l75_75779

theorem yellow_ball_percentage
  (yellow_balls : ℕ)
  (brown_balls : ℕ)
  (blue_balls : ℕ)
  (green_balls : ℕ)
  (total_balls : ℕ := yellow_balls + brown_balls + blue_balls + green_balls)
  (h_yellow : yellow_balls = 75)
  (h_brown : brown_balls = 120)
  (h_blue : blue_balls = 45)
  (h_green : green_balls = 60) :
  (yellow_balls * 100) / total_balls = 25 := 
by
  sorry

end yellow_ball_percentage_l75_75779


namespace paisa_per_rupee_z_gets_l75_75812

theorem paisa_per_rupee_z_gets
  (y_share : ℝ)
  (y_per_x_paisa : ℝ)
  (total_amount : ℝ)
  (x_share : ℝ)
  (z_share : ℝ)
  (paisa_per_rupee : ℝ)
  (h1 : y_share = 36)
  (h2 : y_per_x_paisa = 0.45)
  (h3 : total_amount = 140)
  (h4 : x_share = y_share / y_per_x_paisa)
  (h5 : z_share = total_amount - (x_share + y_share))
  (h6 : paisa_per_rupee = (z_share / x_share) * 100) :
  paisa_per_rupee = 30 :=
by
  sorry

end paisa_per_rupee_z_gets_l75_75812


namespace ai_eq_i_l75_75253

namespace Problem

def gcd (m n : ℕ) : ℕ := Nat.gcd m n

def sequence_satisfies (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j

theorem ai_eq_i (a : ℕ → ℕ) (h : sequence_satisfies a) : ∀ i : ℕ, a i = i :=
by
  sorry

end Problem

end ai_eq_i_l75_75253


namespace evaluate_expression_l75_75262

theorem evaluate_expression (a b : ℤ) (h_a : a = 4) (h_b : b = -3) : -a - b^3 + a * b = 11 :=
by
  rw [h_a, h_b]
  sorry

end evaluate_expression_l75_75262


namespace sequence_term_a_1000_eq_2340_l75_75057

theorem sequence_term_a_1000_eq_2340
  (a : ℕ → ℤ)
  (h1 : a 1 = 2007)
  (h2 : a 2 = 2008)
  (h_rec : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n) :
  a 1000 = 2340 :=
sorry

end sequence_term_a_1000_eq_2340_l75_75057


namespace quadratic_other_root_l75_75196

theorem quadratic_other_root (a : ℝ) (h1 : ∃ (x : ℝ), x^2 - 2 * x + a = 0 ∧ x = -1) :
  ∃ (x2 : ℝ), x2^2 - 2 * x2 + a = 0 ∧ x2 = 3 :=
sorry

end quadratic_other_root_l75_75196


namespace num_men_in_second_group_l75_75053

-- Define the conditions
def numMen1 := 4
def hoursPerDay1 := 10
def daysPerWeek := 7
def earningsPerWeek1 := 1200

def hoursPerDay2 := 6
def earningsPerWeek2 := 1620

-- Define the earning per man-hour
def earningPerManHour := earningsPerWeek1 / (numMen1 * hoursPerDay1 * daysPerWeek)

-- Define the total man-hours required for the second amount of earnings
def totalManHours2 := earningsPerWeek2 / earningPerManHour

-- Define the number of men in the second group
def numMen2 := totalManHours2 / (hoursPerDay2 * daysPerWeek)

-- Theorem stating the number of men in the second group 
theorem num_men_in_second_group : numMen2 = 9 := by
  sorry

end num_men_in_second_group_l75_75053


namespace min_value_of_quadratic_l75_75313

theorem min_value_of_quadratic :
  ∃ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 = -3 :=
sorry

end min_value_of_quadratic_l75_75313


namespace probability_increasing_function_l75_75509

open ProbabilityTheory

variable {a b : ℝ}

theorem probability_increasing_function :
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) →
  (probability_space.probability {ab : ℝ × ℝ | ab.1 ∈ Set.Ioo 0 1 ∧ ab.2 ∈ Set.Ioo 0 1 ∧ ab.1 ≥ 2 * ab.2}) = 1 / 4 :=
by
  sorry

end probability_increasing_function_l75_75509


namespace solution_set_of_inequality_l75_75316

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x^2 - 3*x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
by sorry

end solution_set_of_inequality_l75_75316


namespace correct_choice_option_D_l75_75815

theorem correct_choice_option_D : (500 - 9 * 7 = 437) := by sorry

end correct_choice_option_D_l75_75815


namespace sum_of_first_10_terms_l75_75736

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_first_10_terms (h_seq : arithmetic_sequence a d) (h_d_nonzero : d ≠ 0)
  (h_eq : (a 4) ^ 2 + (a 5) ^ 2 = (a 6) ^ 2 + (a 7) ^ 2) :
  (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * a 4 = 0 :=
by
  sorry

end sum_of_first_10_terms_l75_75736


namespace largest_prime_factor_sum_divisors_144_l75_75880

theorem largest_prime_factor_sum_divisors_144 :
  ∃ p : ℕ, Nat.Prime p ∧ p = 31 ∧ 
  (∀ q : ℕ, Nat.Prime q ∧ q ∣ (Finset.sum (Finset.filter (λ x, (Nat.gcd x 144) = 1) (Finset.range 145))) → q ≤ p) :=
begin
  sorry
end

end largest_prime_factor_sum_divisors_144_l75_75880


namespace equal_roots_B_value_l75_75817

theorem equal_roots_B_value (B : ℝ) :
  (∀ k : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (k = 1 → (B^2 - 4 * (2 * 1) * 2 = 0))) → B = 4 ∨ B = -4 :=
by
  sorry

end equal_roots_B_value_l75_75817


namespace find_a7_l75_75672

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75672


namespace total_spent_is_49_l75_75929

-- Define the prices of items
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define Paula's purchases
def paula_bracelets := 3
def paula_keychains := 2
def paula_coloring_book := 1
def paula_stickers := 4

-- Define Olive's purchases
def olive_bracelets := 2
def olive_coloring_book := 1
def olive_toy_car := 1
def olive_stickers := 3

-- Calculate total expenses
def paula_total := paula_bracelets * price_bracelet + paula_keychains * price_keychain + paula_coloring_book * price_coloring_book + paula_stickers * price_sticker
def olive_total := olive_coloring_book * price_coloring_book + olive_bracelets * price_bracelet + olive_toy_car * price_toy_car + olive_stickers * price_sticker
def total_expense := paula_total + olive_total

-- Prove the total expenses amount to $49
theorem total_spent_is_49 : total_expense = 49 :=
by
  have : paula_total = (3 * 4) + (2 * 5) + (1 * 3) + (4 * 1) := rfl
  have : olive_total = (1 * 3) + (2 * 4) + (1 *6) + (3 * 1) := rfl
  have : paula_total = 29 := rfl
  have : olive_total = 20 := rfl
  have : total_expense = 29 + 20 := rfl
  exact rfl

end total_spent_is_49_l75_75929


namespace ratio_man_to_son_in_two_years_l75_75931

-- Define the conditions
def son_current_age : ℕ := 32
def man_current_age : ℕ := son_current_age + 34

-- Define the ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem to prove the ratio in two years
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / son_age_in_two_years = 2 :=
by
  -- Skip the proof
  sorry

end ratio_man_to_son_in_two_years_l75_75931


namespace find_angle_C_find_area_l75_75841

open Real

-- Definition of the problem conditions and questions

-- Condition: Given a triangle and the trigonometric relationship
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Trigonometric identity provided in the problem
axiom trig_identity : (sqrt 3) * c / (cos C) = a / (cos (3 * π / 2 + A))

-- First part of the problem
theorem find_angle_C (h1 : sqrt 3 * c / cos C = a / cos (3 * π / 2 + A)) : C = π / 6 :=
sorry

-- Second part of the problem
noncomputable def area_of_triangle (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

variables {c' b' : ℝ}
-- Given conditions for the second question 
axiom condition_c_a : c' / a = 2
axiom condition_b : b' = 4 * sqrt 3

-- Definitions to align with the given problem
def c_from_a (a : ℝ) : ℝ := 2 * a

-- The final theorem for the second part
theorem find_area (hC : C = π / 6) (hc : c_from_a a = c') (hb : b' = 4 * sqrt 3) :
  area_of_triangle a b' C = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end find_angle_C_find_area_l75_75841


namespace column_of_2008_l75_75705

theorem column_of_2008:
  (∃ k, 2008 = 2 * k) ∧
  ((2 % 8) = 2) ∧ ((4 % 8) = 4) ∧ ((6 % 8) = 6) ∧ ((8 % 8) = 0) ∧
  ((16 % 8) = 0) ∧ ((14 % 8) = 6) ∧ ((12 % 8) = 4) ∧ ((10 % 8) = 2) →
  (2008 % 8 = 4) :=
by
  sorry

end column_of_2008_l75_75705


namespace victor_weekly_earnings_l75_75440

def wage_per_hour : ℕ := 12
def hours_monday : ℕ := 5
def hours_tuesday : ℕ := 6
def hours_wednesday : ℕ := 7
def hours_thursday : ℕ := 4
def hours_friday : ℕ := 8

def earnings_monday := hours_monday * wage_per_hour
def earnings_tuesday := hours_tuesday * wage_per_hour
def earnings_wednesday := hours_wednesday * wage_per_hour
def earnings_thursday := hours_thursday * wage_per_hour
def earnings_friday := hours_friday * wage_per_hour

def total_earnings := earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday

theorem victor_weekly_earnings : total_earnings = 360 := by
  sorry

end victor_weekly_earnings_l75_75440


namespace inequality_solution_set_l75_75577

theorem inequality_solution_set :
  {x : ℝ | (x - 3) / (x + 2) ≤ 0} = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by
  sorry

end inequality_solution_set_l75_75577


namespace polygon_sides_twice_diagonals_l75_75407

theorem polygon_sides_twice_diagonals (n : ℕ) (h1 : n ≥ 3) (h2 : n * (n - 3) / 2 = 2 * n) : n = 7 :=
sorry

end polygon_sides_twice_diagonals_l75_75407


namespace chord_count_l75_75599

theorem chord_count {n : ℕ} (h : n = 2024) : 
  ∃ k : ℕ, k ≥ 1024732 ∧ ∀ (i j : ℕ), (i < n → j < n → i ≠ j → true) := sorry

end chord_count_l75_75599


namespace find_a7_l75_75674

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75674


namespace solve_problem_l75_75945

def problem_statement (x y : ℕ) : Prop :=
  (x = 3) ∧ (y = 2) → (x^8 + 2 * x^4 * y^2 + y^4) / (x^4 + y^2) = 85

theorem solve_problem : problem_statement 3 2 :=
  by sorry

end solve_problem_l75_75945


namespace num_students_basketball_l75_75537

-- Definitions for conditions
def num_students_cricket : ℕ := 8
def num_students_both : ℕ := 5
def num_students_either : ℕ := 10

-- statement to be proven
theorem num_students_basketball : ∃ B : ℕ, B = 7 ∧ (num_students_either = B + num_students_cricket - num_students_both) := sorry

end num_students_basketball_l75_75537


namespace total_amount_paid_l75_75592

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l75_75592


namespace mart_income_percentage_l75_75272

variables (T J M : ℝ)

theorem mart_income_percentage (h1 : M = 1.60 * T) (h2 : T = 0.50 * J) :
  M = 0.80 * J :=
by
  sorry

end mart_income_percentage_l75_75272


namespace problem1_problem2_l75_75942

-- Problem 1 Statement
theorem problem1 : (3 * Real.sqrt 48 - 2 * Real.sqrt 27) / Real.sqrt 3 = 6 :=
by sorry

-- Problem 2 Statement
theorem problem2 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5) = -3 - Real.sqrt 5 :=
by sorry

end problem1_problem2_l75_75942


namespace sum_of_first_8_terms_l75_75727

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Given conditions
def c1 (a : ℕ → ℝ) : Prop := geometric_sequence a 2
def c2 (a : ℕ → ℝ) : Prop := sum_of_first_n_terms a 4 = 1

-- The statement to prove
theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : c1 a) (h2 : c2 a) : sum_of_first_n_terms a 8 = 17 :=
by
  sorry

end sum_of_first_8_terms_l75_75727


namespace jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l75_75959

theorem jia_can_formulate_quadratic :
  ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem yi_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem bing_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem ding_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

end jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l75_75959


namespace sequence_initial_value_l75_75161

theorem sequence_initial_value (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : a 1 = 0 ∨ a 1 = 2 :=
sorry

end sequence_initial_value_l75_75161


namespace total_expenses_l75_75542

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l75_75542


namespace total_amount_paid_l75_75593

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l75_75593


namespace no_angle_sat_sin_cos_eq_sin_40_l75_75016

open Real

theorem no_angle_sat_sin_cos_eq_sin_40 :
  ¬∃ α : ℝ, sin α * cos α = sin (40 * π / 180) := 
by 
  sorry

end no_angle_sat_sin_cos_eq_sin_40_l75_75016


namespace amy_hours_per_week_l75_75897

theorem amy_hours_per_week {h w summer_salary school_weeks school_salary} 
  (hours_per_week_summer : h = 45)
  (weeks_summer : w = 8)
  (summer_salary_h : summer_salary = 3600)
  (school_weeks_h : school_weeks = 24)
  (school_salary_h : school_salary = 3600) :
  ∃ hours_per_week_school, hours_per_week_school = 15 :=
by
  sorry

end amy_hours_per_week_l75_75897


namespace base5_minus_base8_to_base10_l75_75639

def base5_to_base10 (n : Nat) : Nat :=
  5 * 5^5 + 4 * 5^4 + 3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 0 * 5^0

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0

theorem base5_minus_base8_to_base10 :
  (base5_to_base10 543210 - base8_to_base10 43210) = 499 :=
by
  sorry

end base5_minus_base8_to_base10_l75_75639


namespace harry_apples_l75_75437

theorem harry_apples (martha_apples : ℕ) (tim_apples : ℕ) (harry_apples : ℕ)
  (h1 : martha_apples = 68)
  (h2 : tim_apples = martha_apples - 30)
  (h3 : harry_apples = tim_apples / 2) :
  harry_apples = 19 := 
by sorry

end harry_apples_l75_75437


namespace find_a7_l75_75669

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75669


namespace turtles_remaining_on_log_l75_75127

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l75_75127


namespace perimeter_of_ghost_l75_75202
open Real

def radius := 2
def angle_degrees := 90
def full_circle_degrees := 360

noncomputable def missing_angle := angle_degrees
noncomputable def remaining_angle := full_circle_degrees - missing_angle
noncomputable def fraction_of_circle := remaining_angle / full_circle_degrees
noncomputable def full_circumference := 2 * π * radius
noncomputable def arc_length := fraction_of_circle * full_circumference
noncomputable def radii_length := 2 * radius

theorem perimeter_of_ghost : arc_length + radii_length = 3 * π + 4 :=
by
  sorry

end perimeter_of_ghost_l75_75202


namespace vacant_seats_calculation_l75_75361

noncomputable def seats_vacant (total_seats : ℕ) (percentage_filled : ℚ) : ℚ := 
  total_seats * (1 - percentage_filled)

theorem vacant_seats_calculation: 
  seats_vacant 600 0.45 = 330 := 
by 
    -- sorry to skip the proof.
    sorry

end vacant_seats_calculation_l75_75361


namespace number_difference_l75_75595

theorem number_difference (a b : ℕ) (h1 : a + b = 44) (h2 : 8 * a = 3 * b) : b - a = 20 := by
  sorry

end number_difference_l75_75595


namespace eval_expression_l75_75025

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l75_75025


namespace count_valid_N_l75_75522

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l75_75522


namespace sequence_ineq_l75_75854

theorem sequence_ineq (a : ℕ → ℝ) (h1 : a 1 = 15) 
  (h2 : ∀ n, a (n + 1) = a n - 2 / 3) 
  (hk : a k * a (k + 1) < 0) : k = 23 :=
sorry

end sequence_ineq_l75_75854


namespace quadratic_equal_roots_l75_75911

theorem quadratic_equal_roots :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 → (0 ≤ 0) ∧ 
  (∀ a b : ℝ, 0 = b^2 - 4 * a * 1 → (x = -b / (2 * a))) :=
by
  sorry

end quadratic_equal_roots_l75_75911


namespace melanie_average_speed_l75_75892

theorem melanie_average_speed
  (bike_distance run_distance total_time : ℝ)
  (h_bike : bike_distance = 15)
  (h_run : run_distance = 5)
  (h_time : total_time = 4) :
  (bike_distance + run_distance) / total_time = 5 :=
by
  sorry

end melanie_average_speed_l75_75892


namespace class_B_has_more_stable_grades_l75_75153

-- Definitions based on conditions
def avg_score_class_A : ℝ := 85
def avg_score_class_B : ℝ := 85
def var_score_class_A : ℝ := 120
def var_score_class_B : ℝ := 90

-- Proving which class has more stable grades (lower variance indicates more stability)
theorem class_B_has_more_stable_grades :
  var_score_class_B < var_score_class_A :=
by
  -- The proof will need to show the given condition and establish the inequality
  sorry

end class_B_has_more_stable_grades_l75_75153


namespace find_b_l75_75949

noncomputable def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

theorem find_b (d b e : ℝ) (h1 : -d / 3 = -e) (h2 : -e = 1 + d + b + e) (h3 : e = 6) : b = -31 :=
by sorry

end find_b_l75_75949


namespace remaining_money_l75_75382

def initial_amount : ℕ := 10
def spent_on_toy_truck : ℕ := 3
def spent_on_pencil_case : ℕ := 2

theorem remaining_money (initial_amount spent_on_toy_truck spent_on_pencil_case : ℕ) : 
  initial_amount - (spent_on_toy_truck + spent_on_pencil_case) = 5 :=
by
  sorry

end remaining_money_l75_75382


namespace solution_set_f_pos_min_a2_b2_c2_l75_75373

def f (x : ℝ) : ℝ := |2 * x + 3| - |x - 1|

theorem solution_set_f_pos : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -3 / 2 ∨ -2 / 3 < x } := 
sorry

theorem min_a2_b2_c2 (a b c : ℝ) (h : a + 2 * b + 3 * c = 5) : 
  a^2 + b^2 + c^2 ≥ 25 / 14 :=
sorry

end solution_set_f_pos_min_a2_b2_c2_l75_75373


namespace find_total_money_l75_75476

theorem find_total_money
  (d x T : ℝ)
  (h1 : d = 5 / 17)
  (h2 : x = 35)
  (h3 : d * T = x) :
  T = 119 :=
by sorry

end find_total_money_l75_75476


namespace find_f_37_5_l75_75339

noncomputable def f (x : ℝ) : ℝ := sorry

/--
Given that \( f \) is an odd function defined on \( \mathbb{R} \) and satisfies
\( f(x+2) = -f(x) \). When \( 0 \leqslant x \leqslant 1 \), \( f(x) = x \),
prove that \( f(37.5) = 0.5 \).
-/
theorem find_f_37_5 (f : ℝ → ℝ) (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (periodic_f : ∀ x : ℝ, f (x + 2) = -f x)
  (interval_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) : f 37.5 = 0.5 :=
sorry

end find_f_37_5_l75_75339


namespace min_colors_rect_condition_l75_75041

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l75_75041


namespace find_number_l75_75720

theorem find_number (x y a : ℝ) (h₁ : x * y = 1) (h₂ : (a ^ ((x + y) ^ 2)) / (a ^ ((x - y) ^ 2)) = 1296) : a = 6 :=
sorry

end find_number_l75_75720


namespace f_three_l75_75851

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_succ : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom f_one : f 1 = 1 

-- Goal
theorem f_three : f 3 = -1 :=
by
  -- The proof will be provided here
  sorry

end f_three_l75_75851


namespace roots_equal_when_m_l75_75063

noncomputable def equal_roots_condition (k n m : ℝ) : Prop :=
  1 + 4 * m^2 * k + 4 * m * n = 0

theorem roots_equal_when_m :
  equal_roots_condition 1 3 (-1.5 + Real.sqrt 2) ∧ 
  equal_roots_condition 1 3 (-1.5 - Real.sqrt 2) :=
by 
  sorry

end roots_equal_when_m_l75_75063


namespace nonneg_int_solutions_eq_l75_75273

theorem nonneg_int_solutions_eq (a b : ℕ) : a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by {
  sorry -- Proof omitted
}

end nonneg_int_solutions_eq_l75_75273


namespace integer_between_squares_l75_75219

theorem integer_between_squares (a b c d: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) (h₃: 0 < d) (h₄: c * d = 1) : 
  ∃ n : ℤ, ab ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) := 
by 
  sorry

end integer_between_squares_l75_75219


namespace ceiling_floor_sum_l75_75482

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l75_75482


namespace swans_count_l75_75871

def numberOfSwans : Nat := 12

theorem swans_count (y : Nat) (x : Nat) (h1 : y = 5) (h2 : ∃ n m : Nat, x = 2 * n + 2 ∧ x = 3 * m - 3) : x = numberOfSwans := 
  by 
    sorry

end swans_count_l75_75871


namespace negate_exists_statement_l75_75243

theorem negate_exists_statement : 
  (∃ x : ℝ, x^2 + x - 2 < 0) ↔ ¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0) :=
by sorry

end negate_exists_statement_l75_75243


namespace problem_statement_l75_75170

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l75_75170


namespace gcd_lcm_mul_l75_75232

theorem gcd_lcm_mul (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
by
  sorry

end gcd_lcm_mul_l75_75232


namespace bird_wings_l75_75212

theorem bird_wings (P Pi C : ℕ) (h_total_money : 4 * 50 = 200)
  (h_total_cost : 30 * P + 20 * Pi + 15 * C = 200)
  (h_P_ge : P ≥ 1) (h_Pi_ge : Pi ≥ 1) (h_C_ge : C ≥ 1) :
  2 * (P + Pi + C) = 24 :=
sorry

end bird_wings_l75_75212


namespace group_sum_180_in_range_1_to_60_l75_75992

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem group_sum_180_in_range_1_to_60 :
  ∃ (a n : ℕ), 1 ≤ a ∧ a + n - 1 ≤ 60 ∧ sum_of_arithmetic_series a 1 n = 180 :=
by
  sorry

end group_sum_180_in_range_1_to_60_l75_75992


namespace ceil_floor_eq_zero_implies_sum_l75_75974

theorem ceil_floor_eq_zero_implies_sum (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ + ⌊x⌋ = 2 * x :=
by
  sorry

end ceil_floor_eq_zero_implies_sum_l75_75974


namespace xiao_hong_mistake_l75_75605

theorem xiao_hong_mistake (a : ℕ) (h : 31 - a = 12) : 31 + a = 50 :=
by
  sorry

end xiao_hong_mistake_l75_75605


namespace centroid_y_sum_zero_l75_75383

theorem centroid_y_sum_zero
  (x1 x2 x3 y2 y3 : ℝ)
  (h : y2 + y3 = 0) :
  (x1 + x2 + x3) / 3 = (x1 / 3 + x2 / 3 + x3 / 3) ∧ (y2 + y3) / 3 = 0 :=
by
  sorry

end centroid_y_sum_zero_l75_75383


namespace triangle_angle_sum_l75_75994

theorem triangle_angle_sum (BAC ACB ABC : Real) (H1 : BAC = 50) (H2 : ACB = 40) : ABC = 90 :=
by
  -- Using the angle sum property of a triangle
  have angle_sum : BAC + ACB + ABC = 180 := sorry
  -- Substituting the given angles
  rw [H1, H2] at angle_sum
  -- Performing the calculation to obtain the result
  linarith

end triangle_angle_sum_l75_75994


namespace problem_solution_l75_75513

def BinomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

def PropositionsCorrect : ℕ :=
  let X_var : ℝ := BinomialVariance 4 0.1
  let prop1 : Prop := X_var = 0.36
  let prop2 : Prop := ∀ (x : ℝ) (data : List ℝ), (data.mean - x) ≠ data.mean ∨ (data.variance - 0) = data.variance
  let prop3 : Prop := ∀ (students : List ℤ), students = [5, 16, 27, 38, 49] → list.prod (list.map (λ (x : ℤ), x - students.head!) students) = 55
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0)

theorem problem_solution : PropositionsCorrect = 1 := by
  sorry

end problem_solution_l75_75513


namespace eval_expr_eq_zero_l75_75485

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l75_75485


namespace sector_area_proof_l75_75622

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_proof
  (r : ℝ) (l : ℝ) (perimeter : ℝ) (theta : ℝ) (h1 : perimeter = 2 * r + l)
  (h2 : l = r * theta) (h3 : perimeter = 16) (h4 : theta = 2) :
  sector_area r theta = 16 := by
  sorry

end sector_area_proof_l75_75622


namespace simplify_expression_l75_75327

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l75_75327


namespace problem_equivalent_proof_l75_75694

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75694


namespace larger_number_l75_75426

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75426


namespace least_N_l75_75558

theorem least_N :
  ∃ N : ℕ, 
    (N % 2 = 1) ∧ 
    (N % 3 = 2) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 4) ∧ 
    (∀ M : ℕ, 
      (M % 2 = 1) ∧ 
      (M % 3 = 2) ∧ 
      (M % 5 = 3) ∧ 
      (M % 7 = 4) → 
      N ≤ M) :=
  sorry

end least_N_l75_75558


namespace solve_for_m_l75_75978

theorem solve_for_m (m x : ℤ) (h : 4 * x + 2 * m - 14 = 0) (hx : x = 2) : m = 3 :=
by
  -- Proof steps will go here.
  sorry

end solve_for_m_l75_75978


namespace part1_equation_part2_equation_l75_75114

-- Part (Ⅰ)
theorem part1_equation :
  (- ((-1) ^ 1000) - 2.45 * 8 + 2.55 * (-8) = -41) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_equation :
  ((1 / 6 - 1 / 3 + 0.25) / (- (1 / 12)) = -1) :=
by
  sorry

end part1_equation_part2_equation_l75_75114


namespace game_ends_after_63_rounds_l75_75168

-- Define tokens for players A, B, C, and D at the start
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16
def initial_tokens_D := 14

-- Define the rules of the game
def game_rounds_to_end (A B C D : ℕ) : ℕ :=
  -- This function calculates the number of rounds after which any player runs out of tokens
  if (A, B, C, D) = (20, 18, 16, 14) then 63 else 0

-- Statement to prove
theorem game_ends_after_63_rounds :
  game_rounds_to_end initial_tokens_A initial_tokens_B initial_tokens_C initial_tokens_D = 63 :=
by sorry

end game_ends_after_63_rounds_l75_75168


namespace parabola_intersections_l75_75251

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections_l75_75251


namespace find_a_plus_b_l75_75051

variables (a b c d x : ℝ)

def conditions (a b c d x : ℝ) : Prop :=
  (a + b = x) ∧
  (b + c = 9) ∧
  (c + d = 3) ∧
  (a + d = 5)

theorem find_a_plus_b (a b c d x : ℝ) (h : conditions a b c d x) : a + b = 11 :=
by
  have h1 : a + b = x := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : a + d = 5 := h.2.2.2
  sorry

end find_a_plus_b_l75_75051


namespace power_function_properties_l75_75703

def power_function (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : Prop :=
  f x = x ^ a

theorem power_function_properties :
  ∃ (f : ℝ → ℝ) (a : ℝ), power_function f 2 a ∧ f 2 = 1/2 ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → 
    (f x1 + f x2) / 2 > f ((x1 + x2) / 2)) :=
sorry

end power_function_properties_l75_75703


namespace speed_of_stream_l75_75106

theorem speed_of_stream (v c : ℝ) (h1 : c - v = 6) (h2 : c + v = 10) : v = 2 :=
by
  sorry

end speed_of_stream_l75_75106


namespace true_propositions_l75_75816

theorem true_propositions :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3 : ℚ) * x^2 + (1/2 : ℚ) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) :=
by {
  sorry
}

end true_propositions_l75_75816


namespace percentage_peanut_clusters_is_64_l75_75009

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end percentage_peanut_clusters_is_64_l75_75009


namespace find_a7_l75_75654

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75654


namespace problem_1_problem_2_problem_3_l75_75187

def vec_a : ℝ × ℝ := (1, real.sqrt 3)
def vec_b : ℝ × ℝ := (-2, 0)

def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

def vec_norm (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 * u.1 + u.2 * u.2)

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def calc_angle (u v : ℝ × ℝ) : ℝ :=
  real.arccos (vec_dot u v / (vec_norm u * vec_norm v))

theorem problem_1 : vec_norm (vec_sub vec_a vec_b) = 2 * real.sqrt 3 := 
by sorry

theorem problem_2 : calc_angle (vec_sub vec_a vec_b) vec_a = real.pi / 6 := 
by sorry

theorem problem_3 {t : ℝ} : 
  ∃ l, ∀ t, vec_norm (vec_sub vec_a (t • vec_b)) ≥ l ∧ l = real.sqrt 3 := 
by sorry

end problem_1_problem_2_problem_3_l75_75187


namespace integer_pairs_l75_75636

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem integer_pairs (a b : ℤ) :
  (is_perfect_square (a^2 + 4 * b) ∧ is_perfect_square (b^2 + 4 * a)) ↔ 
  (a = 0 ∧ b = 0) ∨ (a = -4 ∧ b = -4) ∨ (a = 4 ∧ b = -4) ∨
  (∃ (k : ℕ), a = k^2 ∧ b = 0) ∨ (∃ (k : ℕ), a = 0 ∧ b = k^2) ∨
  (a = -6 ∧ b = -5) ∨ (a = -5 ∧ b = -6) ∨
  (∃ (t : ℕ), a = t ∧ b = 1 - t) ∨ (∃ (t : ℕ), a = 1 - t ∧ b = t) :=
sorry

end integer_pairs_l75_75636


namespace min_value_of_expression_l75_75074

theorem min_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) :
  x + 3 * y + 6 * z >= 27 :=
by
  sorry

end min_value_of_expression_l75_75074


namespace odd_function_a_increasing_function_a_l75_75515

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem odd_function_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - (f x a)) → a = -1 :=
by sorry

theorem increasing_function_a (a : ℝ) :
  (∀ x : ℝ, (Real.exp x - a * Real.exp (-x)) ≥ 0) → a ∈ Set.Iic 0 :=
by sorry

end odd_function_a_increasing_function_a_l75_75515


namespace find_number_l75_75109

theorem find_number (x : ℝ) 
  (h1 : 0.15 * 40 = 6) 
  (h2 : 6 = 0.25 * x + 2) : 
  x = 16 := 
sorry

end find_number_l75_75109


namespace find_value_l75_75344

noncomputable def f : ℝ → ℝ := sorry

def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

def has_tangent_at (f : ℝ → ℝ) (x0 : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L x0 (f x0)

theorem find_value (h : has_tangent_at f 2 tangent_line) :
  f 2 - 2 * (deriv f 2) = -1/2 :=
sorry

end find_value_l75_75344


namespace volume_tetrahedron_ABCD_l75_75205

noncomputable def volume_of_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  (1 / 3) * ((1 / 2) * AB * CD * Real.sin angle) * distance

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (Real.sqrt 3) 2 (Real.pi / 3) = 1 / 2 :=
by
  unfold volume_of_tetrahedron
  sorry

end volume_tetrahedron_ABCD_l75_75205


namespace jackson_saving_l75_75730

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end jackson_saving_l75_75730


namespace henry_games_total_l75_75519

theorem henry_games_total
    (wins : ℕ)
    (losses : ℕ)
    (draws : ℕ)
    (hw : wins = 2)
    (hl : losses = 2)
    (hd : draws = 10) :
  wins + losses + draws = 14 :=
by
  -- The proof is omitted.
  sorry

end henry_games_total_l75_75519


namespace percentage_of_students_70_79_l75_75056

-- Defining basic conditions
def students_in_range_90_100 := 5
def students_in_range_80_89 := 9
def students_in_range_70_79 := 7
def students_in_range_60_69 := 4
def students_below_60 := 3

-- Total number of students
def total_students := students_in_range_90_100 + students_in_range_80_89 + students_in_range_70_79 + students_in_range_60_69 + students_below_60

-- Percentage of students in the 70%-79% range
def percent_students_70_79 := (students_in_range_70_79 / total_students) * 100

theorem percentage_of_students_70_79 : percent_students_70_79 = 25 := by
  sorry

end percentage_of_students_70_79_l75_75056


namespace sin_four_arcsin_eq_l75_75652

theorem sin_four_arcsin_eq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) :=
by
  sorry

end sin_four_arcsin_eq_l75_75652


namespace vector_subtraction_l75_75709

variables (a b : ℝ × ℝ)

-- Definitions based on conditions
def vector_a : ℝ × ℝ := (1, -2)
def m : ℝ := 2
def vector_b : ℝ × ℝ := (4, m)

-- Prove given question equals answer
theorem vector_subtraction :
  vector_a = (1, -2) →
  vector_b = (4, m) →
  (1 * 4 + (-2) * m = 0) →
  5 • vector_a - vector_b = (1, -12) := by
  intros h1 h2 h3
  sorry

end vector_subtraction_l75_75709


namespace bread_cooling_time_l75_75995

theorem bread_cooling_time 
  (dough_room_temp : ℕ := 60)   -- 1 hour in minutes
  (shape_dough : ℕ := 15)       -- 15 minutes
  (proof_dough : ℕ := 120)      -- 2 hours in minutes
  (bake_bread : ℕ := 30)        -- 30 minutes
  (start_time : ℕ := 2 * 60)    -- 2:00 am in minutes
  (end_time : ℕ := 6 * 60)      -- 6:00 am in minutes
  : (end_time - start_time) - (dough_room_temp + shape_dough + proof_dough + bake_bread) = 15 := 
  by
  sorry

end bread_cooling_time_l75_75995


namespace no_valid_x_l75_75903

theorem no_valid_x (x y : ℝ) (h : y = 2 * x) : ¬(3 * y ^ 2 - 2 * y + 5 = 2 * (6 * x ^ 2 - 3 * y + 3)) :=
by
  sorry

end no_valid_x_l75_75903


namespace value_of_expression_l75_75979

theorem value_of_expression (a b : ℤ) (h : 2 * a - b = 10) : 2023 - 2 * a + b = 2013 :=
by
  sorry

end value_of_expression_l75_75979


namespace water_percentage_in_fresh_grapes_l75_75838

theorem water_percentage_in_fresh_grapes 
  (P : ℝ) -- the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 40) -- weight of fresh grapes in kg
  (dry_grapes_weight : ℝ := 5) -- weight of dry grapes in kg
  (dried_grapes_water_percentage : ℝ := 20) -- percentage of water in dried grapes
  (solid_content : ℝ := 4) -- solid content in both fresh and dried grapes in kg
  : P = 90 :=
by
  sorry

end water_percentage_in_fresh_grapes_l75_75838


namespace lowest_possible_price_l75_75621

theorem lowest_possible_price
  (MSRP : ℝ)
  (D1 : ℝ)
  (D2 : ℝ)
  (P_final : ℝ)
  (h1 : MSRP = 45.00)
  (h2 : 0.10 ≤ D1 ∧ D1 ≤ 0.30)
  (h3 : D2 = 0.20) :
  P_final = 25.20 :=
by
  sorry

end lowest_possible_price_l75_75621


namespace problem_statement_l75_75612

theorem problem_statement (a b : ℤ) (h1 : b = 7) (h2: a * b = 2 * (a + b) + 1) :
  b - a = 4 := by
  sorry

end problem_statement_l75_75612


namespace find_x_coordinate_l75_75366

theorem find_x_coordinate 
  (x : ℝ)
  (h1 : (0, 0) = (0, 0))
  (h2 : (0, 4) = (0, 4))
  (h3 : (x, 4) = (x, 4))
  (h4 : (x, 0) = (x, 0))
  (h5 : 0.4 * (4 * x) = 8)
  : x = 5 := 
sorry

end find_x_coordinate_l75_75366


namespace initial_total_toys_l75_75769

-- Definitions based on the conditions
def initial_red_toys (R : ℕ) : Prop := R - 2 = 88
def twice_as_many_red_toys (R W : ℕ) : Prop := R - 2 = 2 * W

-- The proof statement: show that initially there were 134 toys in the box
theorem initial_total_toys (R W : ℕ) (hR : initial_red_toys R) (hW : twice_as_many_red_toys R W) : R + W = 134 := 
by sorry

end initial_total_toys_l75_75769


namespace mary_initial_borrowed_books_l75_75379

-- We first define the initial number of books B.
variable (B : ℕ)

-- Next, we encode the conditions into a final condition of having 12 books.
def final_books (B : ℕ) : ℕ := (B - 3 + 5) - 2 + 7

-- The proof problem is to show that B must be 5.
theorem mary_initial_borrowed_books (B : ℕ) (h : final_books B = 12) : B = 5 :=
by
  sorry

end mary_initial_borrowed_books_l75_75379


namespace triangle_perimeter_is_49_l75_75910

theorem triangle_perimeter_is_49 (a b c : ℕ) (a_gt_b_gt_c : a > b ∧ b > c) 
  (T1 : ℕ := 4 * b * c)
  (T2 : ℕ := 4 * b * c * (6 * a^2 + 2 * b^2 + 2 * c^2))
  (eqn : (T2 / (2 * T1)) = 2023) : a + b + c = 49 :=
by
  have h1 : T1 = 4 * b * c := rfl
  have h2 : T2 = 4 * b * c * (6 * a^2 + 2 * b^2 + 2 * c^2) := rfl
  have h_eq : 3 * a^2 + b^2 + c^2 = 2023 := sorry
  have sol : (a = 23 ∧ b = 20 ∧ c = 6) := sorry
  exact sol.1 + sol.2.1 + sol.2.2

end triangle_perimeter_is_49_l75_75910


namespace AQI_data_median_is_184_5_l75_75393

open List

/--
The Hefei Environmental Protection Central Station released the Air Quality Index (AQI) data from January 11 to January 20, 2014.
The data are as follows: 153, 203, 268, 166, 157, 164, 268, 407, 335, 119.
-/
def AQI_data : List ℝ := [153, 203, 268, 166, 157, 164, 268, 407, 335, 119]

/--
To find the median of the AQI data set.
-/
def median (l : List ℝ) : ℝ :=
  let sorted_l := sort l
  if (length sorted_l) % 2 = 1 then
    -- If the length of the list is odd, take the middle element
    nth_le sorted_l ((length sorted_l) / 2) (by sorry)
  else
    -- If the length of the list is even, take the average of the two middle elements
    (nth_le sorted_l ((length sorted_l) / 2 - 1) (by sorry) + nth_le sorted_l ((length sorted_l) / 2) (by sorry)) / 2

/--
Proof that the median of the AQI data set is 184.5.
-/
theorem AQI_data_median_is_184_5 : median AQI_data = 184.5 :=
by sorry

end AQI_data_median_is_184_5_l75_75393


namespace determine_c_l75_75049

theorem determine_c (a c : ℝ) (h : (2 * a - 1) / -3 < - (c + 1) / -4) : c ≠ -1 ∧ (c > 0 ∨ c < 0) :=
by sorry

end determine_c_l75_75049


namespace total_school_population_220_l75_75275

theorem total_school_population_220 (x B : ℕ) 
  (h1 : 242 = (x * B) / 100) 
  (h2 : B = (50 * x) / 100) : x = 220 := by
  sorry

end total_school_population_220_l75_75275


namespace exists_int_x_l75_75711

theorem exists_int_x (K M N : ℤ) (h1 : K ≠ 0) (h2 : M ≠ 0) (h3 : N ≠ 0) (h_coprime : Int.gcd K M = 1) :
  ∃ x : ℤ, K ∣ (M * x + N) :=
by
  sorry

end exists_int_x_l75_75711


namespace total_stock_worth_is_15000_l75_75623

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the conditions
def stock_condition_1 := 0.20 * X -- Worth of 20% of the stock
def stock_condition_2 := 0.10 * (0.20 * X) -- Profit from 20% of the stock
def stock_condition_3 := 0.80 * X -- Worth of 80% of the stock
def stock_condition_4 := 0.05 * (0.80 * X) -- Loss from 80% of the stock
def overall_loss := 0.04 * X - 0.02 * X

-- The question rewritten as a theorem statement
theorem total_stock_worth_is_15000 (h1 : overall_loss X = 300) : X = 15000 :=
by sorry

end total_stock_worth_is_15000_l75_75623


namespace rank_A_second_l75_75001

-- We define the conditions provided in the problem
variables (a b c : ℕ) -- defining the scores of A, B, and C as natural numbers

-- Conditions given
def A_said (a b c : ℕ) := b < a ∧ c < a
def B_said (b c : ℕ) := b > c
def C_said (a b c : ℕ) := a > c ∧ b > c

-- Conditions as hypotheses
variable (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) -- the scores are different
variable (h2 : A_said a b c ∨ B_said b c ∨ C_said a b c) -- exactly one of the statements is incorrect

-- The theorem to prove
theorem rank_A_second : ∃ (rankA : ℕ), rankA = 2 := by
  sorry

end rank_A_second_l75_75001


namespace only_n_divides_2_pow_n_minus_1_l75_75642

theorem only_n_divides_2_pow_n_minus_1 : ∀ (n : ℕ), n > 0 ∧ n ∣ (2^n - 1) ↔ n = 1 := by
  sorry

end only_n_divides_2_pow_n_minus_1_l75_75642


namespace tablet_battery_life_l75_75889

theorem tablet_battery_life :
  ∀ (active_usage_hours idle_usage_hours : ℕ),
  active_usage_hours + idle_usage_hours = 12 →
  active_usage_hours = 3 →
  ((active_usage_hours / 2) + (idle_usage_hours / 10)) > 1 →
  idle_usage_hours = 9 →
  0 = 0 := 
by
  intros active_usage_hours idle_usage_hours h1 h2 h3 h4
  sorry

end tablet_battery_life_l75_75889


namespace sequence_count_even_odd_l75_75150

/-- The number of 8-digit sequences such that no two adjacent digits have the same parity
    and the sequence starts with an even number. -/
theorem sequence_count_even_odd : 
  let choices_for_even := 5
  let choices_for_odd := 5
  let total_positions := 8
  (choices_for_even * (choices_for_odd * choices_for_even) ^ (total_positions / 2 - 1)) = 390625 :=
by
  sorry

end sequence_count_even_odd_l75_75150


namespace fraction_meaningful_iff_l75_75054

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = x / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l75_75054


namespace eval_ceil_floor_sum_l75_75483

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l75_75483


namespace trapezoid_area_no_solutions_l75_75394

noncomputable def no_solutions_to_trapezoid_problem : Prop :=
  ∀ (b1 b2 : ℕ), 
    (∃ (m n : ℕ), b1 = 10 * m ∧ b2 = 10 * n) →
    (b1 + b2 = 72) → false

theorem trapezoid_area_no_solutions : no_solutions_to_trapezoid_problem :=
by
  sorry

end trapezoid_area_no_solutions_l75_75394


namespace books_bought_l75_75080

theorem books_bought (initial_books bought_books total_books : ℕ) 
    (h_initial : initial_books = 35)
    (h_total : total_books = 56) :
    bought_books = total_books - initial_books → bought_books = 21 := 
by
  sorry

end books_bought_l75_75080


namespace geometric_seq_a7_l75_75664

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75664


namespace total_distance_correct_l75_75556

def d1 : ℕ := 350
def d2 : ℕ := 375
def d3 : ℕ := 275
def total_distance : ℕ := 1000

theorem total_distance_correct : d1 + d2 + d3 = total_distance := by
  sorry

end total_distance_correct_l75_75556


namespace triangle_right_triangle_of_consecutive_integers_sum_l75_75826

theorem triangle_right_triangle_of_consecutive_integers_sum (
  m n : ℕ
) (h1 : 0 < m) (h2 : n^2 = 2*m + 1) : 
  n * n + m * m = (m + 1) * (m + 1) := 
sorry

end triangle_right_triangle_of_consecutive_integers_sum_l75_75826


namespace relationship_between_abc_l75_75193

theorem relationship_between_abc (a b c k : ℝ) 
  (hA : -3 = - (k^2 + 1) / a)
  (hB : -2 = - (k^2 + 1) / b)
  (hC : 1 = - (k^2 + 1) / c)
  (hk : 0 < k^2 + 1) : c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l75_75193


namespace evaluate_expression_l75_75488

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l75_75488


namespace smallest_positive_period_one_increasing_interval_l75_75504

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

def is_periodic_with_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem smallest_positive_period :
  is_periodic_with_period f Real.pi :=
sorry

theorem one_increasing_interval :
  is_increasing_on f (-(Real.pi / 8)) (3 * Real.pi / 8) :=
sorry

end smallest_positive_period_one_increasing_interval_l75_75504


namespace larger_number_l75_75425

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75425


namespace pancakes_needed_l75_75630

def short_stack_pancakes : ℕ := 3
def big_stack_pancakes : ℕ := 5
def short_stack_customers : ℕ := 9
def big_stack_customers : ℕ := 6

theorem pancakes_needed : (short_stack_customers * short_stack_pancakes + big_stack_customers * big_stack_pancakes) = 57 :=
by
  sorry

end pancakes_needed_l75_75630


namespace larger_number_is_23_l75_75428

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75428


namespace inequality_proof_l75_75759

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l75_75759


namespace CombinedHeightOfTowersIsCorrect_l75_75012

-- Define the heights as non-negative reals for clarity.
noncomputable def ClydeTowerHeight : ℝ := 5.0625
noncomputable def GraceTowerHeight : ℝ := 40.5
noncomputable def SarahTowerHeight : ℝ := 2 * ClydeTowerHeight
noncomputable def LindaTowerHeight : ℝ := (ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight) / 3
noncomputable def CombinedHeight : ℝ := ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight + LindaTowerHeight

-- State the theorem to be proven
theorem CombinedHeightOfTowersIsCorrect : CombinedHeight = 74.25 := 
by
  sorry

end CombinedHeightOfTowersIsCorrect_l75_75012


namespace origin_moves_3sqrt5_under_dilation_l75_75279

/--
Given:
1. The original circle has radius 3 centered at point B(3, 3).
2. The dilated circle has radius 6 centered at point B'(9, 12).

Prove that the distance moved by the origin O(0, 0) under this dilation is 3 * sqrt(5).
-/
theorem origin_moves_3sqrt5_under_dilation:
  let B := (3, 3)
  let B' := (9, 12)
  let radius_B := 3
  let radius_B' := 6
  let dilation_center := (-3, -6)
  let origin := (0, 0)
  let k := radius_B' / radius_B
  let d_0 := Real.sqrt ((-3 : ℝ)^2 + (-6 : ℝ)^2)
  let d_1 := k * d_0
  d_1 - d_0 = 3 * Real.sqrt (5 : ℝ) := by sorry

end origin_moves_3sqrt5_under_dilation_l75_75279


namespace rhombus_area_l75_75092

-- Define the lengths of the diagonals
def d1 : ℝ := 25
def d2 : ℝ := 30

-- Statement to prove that the area of the rhombus is 375 square centimeters
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 375 := by
  -- Proof to be provided
  sorry

end rhombus_area_l75_75092


namespace dot_product_of_vectors_l75_75346

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1) - vector_a

theorem dot_product_of_vectors :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = -4 :=
by
  sorry

end dot_product_of_vectors_l75_75346


namespace min_value_range_l75_75173

noncomputable def f (a x : ℝ) := x^2 + a * x

theorem min_value_range (a : ℝ) :
  (∃x : ℝ, ∀y : ℝ, f a (f a x) ≥ f a (f a y)) ∧ (∀x : ℝ, f a x ≥ f a (-a / 2)) →
  a ≤ 0 ∨ a ≥ 2 := sorry

end min_value_range_l75_75173


namespace train_length_l75_75935

-- Define the given conditions
def train_cross_time : ℕ := 40 -- time in seconds
def train_speed_kmh : ℕ := 144 -- speed in km/h

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 5) / 18 

def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh

-- Theorem statement
theorem train_length :
  train_speed_ms * train_cross_time = 1600 :=
by
  sorry

end train_length_l75_75935


namespace larger_number_l75_75424

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75424


namespace unique_solutions_l75_75493

noncomputable def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ∣ (b^4 + 1) ∧ b ∣ (a^4 + 1) ∧ (Nat.floor (Real.sqrt a) = Nat.floor (Real.sqrt b))

theorem unique_solutions :
  ∀ (a b : ℕ), is_solution a b → (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by 
  sorry

end unique_solutions_l75_75493


namespace simplify_expression_l75_75389

theorem simplify_expression : 
  (1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1) :=
by
  sorry

end simplify_expression_l75_75389


namespace fraction_of_single_men_l75_75819

theorem fraction_of_single_men (total_faculty : ℕ) (percent_women percent_married percent_men_married : ℚ)
  (H1 : percent_women = 60 / 100) (H2 : percent_married = 60 / 100) (H3 : percent_men_married = 1 / 4) :
  let total_men := total_faculty * (1 - percent_women)
  let married_men := total_men * percent_men_married
  let single_men := total_men - married_men
  single_men / total_men = 3 / 4 :=
by {
  let total_men := total_faculty * (1 - percent_women),
  let married_men := total_men * percent_men_married,
  let single_men := total_men - married_men,
  have H_total_men : total_men = total_faculty * (1 - percent_women), from rfl,
  rw H_total_men,
  have H_married_men : married_men = total_men * percent_men_married, from rfl,
  rw H_married_men,
  have H_single_men : single_men = total_men - married_men, from rfl,
  rw H_single_men,
  have H_fraction : (total_men - married_men) / total_men = 3 / 4,
  { sorry }, -- detailed proof can be filled in as needed
  exact H_fraction,
}

end fraction_of_single_men_l75_75819


namespace parallelogram_area_l75_75843

-- Define a plane rectangular coordinate system
structure PlaneRectangularCoordinateSystem :=
(axis : ℝ)

-- Define the properties of a square
structure Square :=
(side_length : ℝ)

-- Define the properties of a parallelogram in a perspective drawing
structure Parallelogram :=
(side_length: ℝ)

-- Define the conditions of the problem
def problem_conditions (s : Square) (p : Parallelogram) :=
  s.side_length = 4 ∨ s.side_length = 8 ∧ 
  p.side_length = 4

-- Statement of the problem
theorem parallelogram_area (s : Square) (p : Parallelogram)
  (h : problem_conditions s p) :
  p.side_length * p.side_length = 16 ∨ p.side_length * p.side_length = 64 :=
by {
  sorry
}

end parallelogram_area_l75_75843


namespace find_k_l75_75756

theorem find_k (x y k : ℝ) (h₁ : 3 * x + y = k) (h₂ : -1.2 * x + y = -20) (hx : x = 7) : k = 9.4 :=
by
  sorry

end find_k_l75_75756


namespace Tim_total_payment_l75_75589

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l75_75589


namespace furthest_distance_l75_75097

-- Definitions of point distances as given conditions
def PQ : ℝ := 13
def QR : ℝ := 11
def RS : ℝ := 14
def SP : ℝ := 12

-- Statement of the problem in Lean
theorem furthest_distance :
  ∃ (P Q R S : ℝ),
    |P - Q| = PQ ∧
    |Q - R| = QR ∧
    |R - S| = RS ∧
    |S - P| = SP ∧
    ∀ (a b : ℝ), a ≠ b →
      |a - b| ≤ 25 :=
sorry

end furthest_distance_l75_75097


namespace isosceles_triangle_perimeter_l75_75963

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 10) :
∃ c : ℝ, is_isosceles_triangle a b c ∧ perimeter a b c = 25 :=
by {
  sorry
}

end isosceles_triangle_perimeter_l75_75963


namespace perpendicular_lines_solve_a_l75_75968

theorem perpendicular_lines_solve_a (a : ℝ) :
  (3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0 → a = 0 ∨ a = 12 / 11 :=
by 
  sorry

end perpendicular_lines_solve_a_l75_75968


namespace area_of_L_shape_l75_75629

theorem area_of_L_shape (a : ℝ) (h_pos : a > 0) (h_eq : 4 * ((a + 3)^2 - a^2) = 5 * a^2) : 
  (a + 3)^2 - a^2 = 45 :=
by
  sorry

end area_of_L_shape_l75_75629


namespace determine_k_l75_75160

noncomputable def k_value (k : ℤ) : Prop :=
  let m := (-2 - 2) / (3 - 1)
  let b := 2 - m * 1
  let y := m * 4 + b
  let point := (4, k / 3)
  point.2 = y

theorem determine_k :
  ∃ k : ℤ, k_value k ∧ k = -12 :=
by
  use -12
  sorry

end determine_k_l75_75160


namespace total_pizza_pieces_l75_75304

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l75_75304


namespace birdseed_weekly_consumption_l75_75084

def parakeets := 3
def parakeet_consumption := 2
def parrots := 2
def parrot_consumption := 14
def finches := 4
def finch_consumption := parakeet_consumption / 2
def canaries := 5
def canary_consumption := 3
def african_grey_parrots := 2
def african_grey_parrot_consumption := 18
def toucans := 3
def toucan_consumption := 25

noncomputable def daily_consumption := 
  parakeets * parakeet_consumption +
  parrots * parrot_consumption +
  finches * finch_consumption +
  canaries * canary_consumption +
  african_grey_parrots * african_grey_parrot_consumption +
  toucans * toucan_consumption

noncomputable def weekly_consumption := 7 * daily_consumption

theorem birdseed_weekly_consumption : weekly_consumption = 1148 := by
  sorry

end birdseed_weekly_consumption_l75_75084


namespace boat_distance_downstream_l75_75276

-- Definitions
def boat_speed_in_still_water : ℝ := 24
def stream_speed : ℝ := 4
def time_downstream : ℝ := 3

-- Effective speed downstream
def speed_downstream := boat_speed_in_still_water + stream_speed

-- Distance calculation
def distance_downstream := speed_downstream * time_downstream

-- Proof statement
theorem boat_distance_downstream : distance_downstream = 84 := 
by
  -- This is where the proof would go, but we use sorry for now
  sorry

end boat_distance_downstream_l75_75276


namespace domain_g_l75_75256

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l75_75256


namespace minimum_value_function_l75_75836

theorem minimum_value_function (x : ℝ) (h : x > -1) : 
  (∃ y, y = (x^2 + 7 * x + 10) / (x + 1) ∧ y ≥ 9) :=
sorry

end minimum_value_function_l75_75836


namespace right_triangle_altitude_l75_75988

theorem right_triangle_altitude {DE DF EF altitude : ℝ} (h_right_triangle : DE^2 = DF^2 + EF^2)
  (h_DE : DE = 15) (h_DF : DF = 9) (h_EF : EF = 12) (h_area : (DF * EF) / 2 = 54) :
  altitude = 7.2 := 
  sorry

end right_triangle_altitude_l75_75988


namespace inequality_subtraction_l75_75975

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l75_75975


namespace simplify_expression_l75_75326

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l75_75326


namespace initial_population_l75_75116

theorem initial_population (P : ℝ) (h1 : 0.76 * P = 3553) : P = 4678 :=
by
  sorry

end initial_population_l75_75116


namespace trig_identity_l75_75845

theorem trig_identity (α : ℝ) (h1 : (-Real.pi / 2) < α ∧ α < 0)
  (h2 : Real.sin α + Real.cos α = 1 / 5) :
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25 / 7 := 
by 
  sorry

end trig_identity_l75_75845


namespace larger_number_is_23_l75_75432

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75432


namespace positive_factors_of_450_are_perfect_squares_eq_8_l75_75520

theorem positive_factors_of_450_are_perfect_squares_eq_8 :
  let n := 450 in
  let is_factor (m k : ℕ) : Prop := k ≠ 0 ∧ k ≤ m ∧ m % k = 0 in
  let is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n in
  let factors (m : ℕ) : List ℕ := List.filter (λ k, is_factor m k) (List.range (m + 1)) in
  let perfect_square_factors := List.filter is_perfect_square (factors n) in
  List.length perfect_square_factors = 8 :=
by 
  have h1 : n = 2 * 3^2 * 5^2 := by sorry
  have h2 : ∀ m, m ∈ factors n ↔ is_factor n m := by sorry
  have h3 : ∀ m, is_perfect_square m ↔ (∃ k, k * k = m) := by sorry
  have h4 : List.length perfect_square_factors = 8 := by sorry
  exact h4

end positive_factors_of_450_are_perfect_squares_eq_8_l75_75520


namespace binkie_gemstones_l75_75158

variable (Binkie Frankie Spaatz : ℕ)

-- Define the given conditions
def condition1 : Binkie = 4 * Frankie := by sorry
def condition2 : Spaatz = (1 / 2) * Frankie - 2 := by sorry
def condition3 : Spaatz = 1 := by sorry

-- State the theorem to be proved
theorem binkie_gemstones : Binkie = 24 := by
  have h_Frankie : Frankie = 6 := by
    sorry
  rw [←condition3, ←condition2] at h_Frankie
  have h_Binkie : Binkie = 4 * 6 := by
    rw [condition1]
    sorry
  rw [h_Binkie]
  exact
    show 4 * 6 = 24 from rfl

end binkie_gemstones_l75_75158


namespace larger_number_is_23_l75_75416

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75416


namespace euclidean_division_l75_75884

theorem euclidean_division (a b : ℕ) (hb : b ≠ 0) : ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ a = b * q + r :=
by sorry

end euclidean_division_l75_75884


namespace inverse_proposition_equivalence_l75_75402

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l75_75402


namespace ratio_of_areas_l75_75619

theorem ratio_of_areas (T A B : ℝ) (hT : T = 900) (hB : B = 405) (hSum : A + B = T) :
  (A - B) / ((A + B) / 2) = 1 / 5 :=
by
  sorry

end ratio_of_areas_l75_75619


namespace joe_paint_left_after_third_week_l75_75547

def initial_paint : ℕ := 360

def paint_used_first_week (initial_paint : ℕ) : ℕ := initial_paint / 4

def paint_left_after_first_week (initial_paint : ℕ) : ℕ := initial_paint - paint_used_first_week initial_paint

def paint_used_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week / 2

def paint_left_after_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week - paint_used_second_week paint_left_after_first_week

def paint_used_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week * 2 / 3

def paint_left_after_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week - paint_used_third_week paint_left_after_second_week

theorem joe_paint_left_after_third_week : 
  paint_left_after_third_week (paint_left_after_second_week (paint_left_after_first_week initial_paint)) = 45 :=
by 
  sorry

end joe_paint_left_after_third_week_l75_75547


namespace problem_statement_l75_75560

open Nat

noncomputable def no_rational_roots (n : ℕ) (hn : n > 1) : Prop :=
  ∀ x : ℚ, (bigOperators.sum (range (n + 1)) (λ (k : ℕ), x^(n - k) / (k! : ℚ)) = -1) → False

theorem problem_statement (n : ℕ) (hn : n > 1) : no_rational_roots n hn :=
sorry

end problem_statement_l75_75560


namespace material_needed_l75_75996

-- Define the required conditions
def feet_per_tee_shirt : ℕ := 4
def number_of_tee_shirts : ℕ := 15

-- State the theorem and the proof obligation
theorem material_needed : feet_per_tee_shirt * number_of_tee_shirts = 60 := 
by 
  sorry

end material_needed_l75_75996


namespace arithmetic_geometric_fraction_l75_75848

theorem arithmetic_geometric_fraction (a x₁ x₂ b y₁ y₂ : ℝ) 
  (h₁ : x₁ + x₂ = a + b) 
  (h₂ : y₁ * y₂ = ab) : 
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (ab) := 
by
  sorry

end arithmetic_geometric_fraction_l75_75848


namespace son_l75_75460

theorem son's_age (S F : ℕ) (h₁ : F = 7 * (S - 8)) (h₂ : F / 4 = 14) : S = 16 :=
by {
  sorry
}

end son_l75_75460


namespace common_divisor_seven_l75_75311

-- Definition of numbers A, B, and C based on given conditions
def A (m n : ℤ) : ℤ := n^2 + 2 * m * n + 3 * m^2 + 2
def B (m n : ℤ) : ℤ := 2 * n^2 + 3 * m * n + m^2 + 2
def C (m n : ℤ) : ℤ := 3 * n^2 + m * n + 2 * m^2 + 1

-- The proof statement ensuring A, B and C have a common divisor of 7
theorem common_divisor_seven (m n : ℤ) : ∃ d : ℤ, d > 1 ∧ d ∣ A m n ∧ d ∣ B m n ∧ d ∣ C m n → d = 7 :=
by
  sorry

end common_divisor_seven_l75_75311


namespace solution_set_equiv_l75_75310

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv_l75_75310


namespace tan_A_in_right_triangle_l75_75208

theorem tan_A_in_right_triangle (AC : ℝ) (AB : ℝ) (BC : ℝ) (hAC : AC = Real.sqrt 20) (hAB : AB = 4) (h_right_triangle : AC^2 = AB^2 + BC^2) :
  Real.tan (Real.arcsin (AB / AC)) = 1 / 2 :=
by
  sorry

end tan_A_in_right_triangle_l75_75208


namespace semerka_connected_l75_75990

open Combinatorics

def semerka_graph : SimpleGraph (Fin 15) := sorry

theorem semerka_connected (G : SimpleGraph (Fin 15)) 
  (H : ∀ v : (Fin 15), 7 ≤ (G.degree v)) : G.IsConnected :=
begin
  -- proof steps here
  sorry
end

end semerka_connected_l75_75990


namespace pages_of_shorter_book_is_10_l75_75908

theorem pages_of_shorter_book_is_10
  (x : ℕ) 
  (h_diff : ∀ (y : ℕ), x = y - 10)
  (h_divide : (x + 10) / 2 = x) 
  : x = 10 :=
by
  sorry

end pages_of_shorter_book_is_10_l75_75908


namespace smallest_multiple_l75_75782

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l75_75782


namespace larger_number_l75_75422

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75422


namespace only_integer_triplet_solution_l75_75832

theorem only_integer_triplet_solution 
  (a b c : ℤ) : 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by 
  intro h
  sorry

end only_integer_triplet_solution_l75_75832


namespace new_average_income_l75_75566

theorem new_average_income (old_avg_income : ℝ) (num_members : ℕ) (deceased_income : ℝ) 
  (old_avg_income_eq : old_avg_income = 735) (num_members_eq : num_members = 4) 
  (deceased_income_eq : deceased_income = 990) : 
  ((old_avg_income * num_members) - deceased_income) / (num_members - 1) = 650 := 
by sorry

end new_average_income_l75_75566


namespace tangent_line_slope_l75_75535

theorem tangent_line_slope (h : ℝ → ℝ) (a : ℝ) (P : ℝ × ℝ) 
  (tangent_eq : ∀ x y, 2 * x + y + 1 = 0 ↔ (x, y) = (a, h a)) : 
  deriv h a < 0 :=
sorry

end tangent_line_slope_l75_75535


namespace focal_distance_of_ellipse_l75_75240

theorem focal_distance_of_ellipse : 
  ∀ (θ : ℝ), (∃ (c : ℝ), (x = 5 * Real.cos θ ∧ y = 4 * Real.sin θ) → 2 * c = 6) :=
by
  sorry

end focal_distance_of_ellipse_l75_75240


namespace least_clock_equivalent_l75_75741

theorem least_clock_equivalent (t : ℕ) (h : t > 5) : 
  (t^2 - t) % 24 = 0 → t = 9 :=
by
  sorry

end least_clock_equivalent_l75_75741


namespace jackson_savings_per_paycheck_l75_75729

-- Jackson wants to save $3,000
def total_savings : ℝ := 3000

-- The vacation is 15 months away
def months_to_save : ℝ := 15

-- Jackson is paid twice a month
def pay_times_per_month : ℝ := 2

-- Jackson's required savings per paycheck to have $3,000 saved in 15 months
theorem jackson_savings_per_paycheck :
  (total_savings / (months_to_save * pay_times_per_month)) = 100 :=
by simp [total_savings, months_to_save, pay_times_per_month]; norm_num; sorry

end jackson_savings_per_paycheck_l75_75729


namespace shaded_region_area_l75_75492

theorem shaded_region_area (a b : ℕ) (H : a = 2) (K : b = 4) :
  let s := a + b
  let area_square_EFGH := s * s
  let area_smaller_square_FG := a * a
  let area_smaller_square_EF := b * b
  let shaded_area := area_square_EFGH - (area_smaller_square_FG + area_smaller_square_EF)
  shaded_area = 16 := 
by
  sorry

end shaded_region_area_l75_75492


namespace marvin_substitute_correct_l75_75890

theorem marvin_substitute_correct {a b c d f : ℤ} (ha : a = 3) (hb : b = 4) (hc : c = 7) (hd : d = 5) :
  (a + (b - (c + (d - f))) = 5 - f) → f = 5 :=
sorry

end marvin_substitute_correct_l75_75890


namespace enchilada_taco_cost_l75_75384

variables (e t : ℝ)

theorem enchilada_taco_cost 
  (h1 : 4 * e + 5 * t = 4.00) 
  (h2 : 5 * e + 3 * t = 3.80) 
  (h3 : 7 * e + 6 * t = 6.10) : 
  4 * e + 7 * t = 4.75 := 
sorry

end enchilada_taco_cost_l75_75384


namespace sum_of_coefficients_l75_75699

-- Define a namespace to encapsulate the problem
namespace PolynomialCoefficients

-- Problem statement as a Lean theorem
theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  α^2005 + β^2005 = 1 :=
sorry -- Placeholder for the proof

end PolynomialCoefficients

end sum_of_coefficients_l75_75699


namespace find_a7_l75_75683

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75683


namespace sum_of_all_potential_real_values_of_x_l75_75015

/-- Determine the sum of all potential real values of x such that when the mean, median, 
and mode of the list [12, 3, 6, 3, 8, 3, x, 15] are arranged in increasing order, they 
form a non-constant arithmetic progression. -/
def sum_potential_x_values : ℚ :=
    let values := [12, 3, 6, 3, 8, 3, 15]
    let mean (x : ℚ) : ℚ := (50 + x) / 8
    let mode : ℚ := 3
    let median (x : ℚ) : ℚ := 
      if x ≤ 3 then 3.5 else if x < 6 then (x + 6) / 2 else 6
    let is_arithmetic_seq (a b c : ℚ) : Prop := 2 * b = a + c
    let valid_x_values : List ℚ := 
      (if is_arithmetic_seq mode 3.5 (mean (3.5)) then [] else []) ++
      (if is_arithmetic_seq mode 6 (mean 6) then [22] else []) ++
      (if is_arithmetic_seq mode (median (50 / 7)) (mean (50 / 7)) then [50 / 7] else [])
    (valid_x_values.sum)
theorem sum_of_all_potential_real_values_of_x :
  sum_potential_x_values = 204 / 7 :=
  sorry

end sum_of_all_potential_real_values_of_x_l75_75015


namespace unique_solution_of_system_l75_75392

noncomputable def solve_system_of_equations (x1 x2 x3 x4 x5 x6 x7 : ℝ) : Prop :=
  10 * x1 + 3 * x2 + 4 * x3 + x4 + x5 = 0 ∧
  11 * x2 + 2 * x3 + 2 * x4 + 3 * x5 + x6 = 0 ∧
  15 * x3 + 4 * x4 + 5 * x5 + 4 * x6 + x7 = 0 ∧
  2 * x1 + x2 - 3 * x3 + 12 * x4 - 3 * x5 + x6 + x7 = 0 ∧
  6 * x1 - 5 * x2 + 3 * x3 - x4 + 17 * x5 + x6 = 0 ∧
  3 * x1 + 2 * x2 - 3 * x3 + 4 * x4 + x5 - 16 * x6 + 2 * x7 = 0 ∧
  4 * x1 - 8 * x2 + x3 + x4 - 3 * x5 + 19 * x7 = 0

theorem unique_solution_of_system :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ),
    solve_system_of_equations x1 x2 x3 x4 x5 x6 x7 →
    x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0 ∧ x6 = 0 ∧ x7 = 0 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h
  sorry

end unique_solution_of_system_l75_75392


namespace find_a_l75_75195

theorem find_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0 → (3 * x + y + a = 0))) → a = 1 :=
sorry

end find_a_l75_75195


namespace smallest_positive_integer_l75_75783

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l75_75783


namespace frog_arrangements_l75_75248

theorem frog_arrangements :
  let total_frogs := 7
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let valid_sequences := 4
  let green_permutations := Nat.factorial green_frogs
  let red_permutations := Nat.factorial red_frogs
  let blue_permutations := Nat.factorial blue_frogs
  let total_permutations := valid_sequences * (green_permutations * red_permutations * blue_permutations)
  total_frogs = green_frogs + red_frogs + blue_frogs → 
  green_frogs = 2 ∧ red_frogs = 3 ∧ blue_frogs = 2 →
  valid_sequences = 4 →
  total_permutations = 96 := 
by
  -- Given conditions lead to the calculation of total permutations 
  sorry

end frog_arrangements_l75_75248


namespace take_home_pay_correct_l75_75829

noncomputable def faith_take_home_pay : Float :=
  let regular_hourly_rate := 13.50
  let regular_hours_per_day := 8
  let days_per_week := 5
  let regular_hours_per_week := regular_hours_per_day * days_per_week
  let regular_earnings_per_week := regular_hours_per_week * regular_hourly_rate

  let overtime_rate_multiplier := 1.5
  let overtime_hourly_rate := regular_hourly_rate * overtime_rate_multiplier
  let overtime_hours_per_day := 2
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week
  let overtime_earnings_per_week := overtime_hours_per_week * overtime_hourly_rate

  let total_sales := 3200.0
  let commission_rate := 0.10
  let commission := total_sales * commission_rate

  let total_earnings_before_deductions := regular_earnings_per_week + overtime_earnings_per_week + commission

  let deduction_rate := 0.25
  let amount_withheld := total_earnings_before_deductions * deduction_rate
  let amount_withheld_rounded := (amount_withheld * 100).round / 100

  let take_home_pay := total_earnings_before_deductions - amount_withheld_rounded
  take_home_pay

theorem take_home_pay_correct : faith_take_home_pay = 796.87 :=
by
  /- Proof omitted -/
  sorry

end take_home_pay_correct_l75_75829


namespace hall_reunion_attendees_l75_75102

noncomputable def Oates : ℕ := 40
noncomputable def both : ℕ := 10
noncomputable def total : ℕ := 100
noncomputable def onlyOates := Oates - both
noncomputable def onlyHall := total - onlyOates - both
noncomputable def Hall := onlyHall + both

theorem hall_reunion_attendees : Hall = 70 := by {
  sorry
}

end hall_reunion_attendees_l75_75102


namespace sin_lt_alpha_lt_tan_l75_75447

open Real

theorem sin_lt_alpha_lt_tan {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2) : sin α < α ∧ α < tan α := by
  sorry

end sin_lt_alpha_lt_tan_l75_75447


namespace eval_expr_l75_75446

theorem eval_expr : 4 * (8 - 3 + 2) / 2 = 14 := 
by
  sorry

end eval_expr_l75_75446


namespace image_of_3_5_pre_image_of_3_5_l75_75180

def f (x y : ℤ) : ℤ × ℤ := (x - y, x + y)

theorem image_of_3_5 : f 3 5 = (-2, 8) :=
by
  sorry

theorem pre_image_of_3_5 : ∃ (x y : ℤ), f x y = (3, 5) ∧ x = 4 ∧ y = 1 :=
by
  sorry

end image_of_3_5_pre_image_of_3_5_l75_75180


namespace determine_b2050_l75_75881

theorem determine_b2050 (b : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h₁ : b 1 = 3 + Real.sqrt 2)
  (h₂ : b 2021 = 7 + 2 * Real.sqrt 2) :
  b 2050 = (7 - 2 * Real.sqrt 2) / 41 := 
sorry

end determine_b2050_l75_75881


namespace max_brownies_l75_75971

theorem max_brownies (m n : ℕ) (h : (m - 2) * (n - 2) = 2 * m + 2 * n - 4) : m * n ≤ 60 :=
sorry

end max_brownies_l75_75971


namespace zeros_of_f_is_pm3_l75_75434

def f (x : ℝ) : ℝ := x^2 - 9

theorem zeros_of_f_is_pm3 :
  ∃ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by sorry

end zeros_of_f_is_pm3_l75_75434


namespace dog_food_packages_l75_75468

theorem dog_food_packages
  (packages_cat_food : Nat := 9)
  (cans_per_package_cat_food : Nat := 10)
  (cans_per_package_dog_food : Nat := 5)
  (more_cans_cat_food : Nat := 55)
  (total_cans_cat_food : Nat := packages_cat_food * cans_per_package_cat_food)
  (total_cans_dog_food : Nat := d * cans_per_package_dog_food)
  (h : total_cans_cat_food = total_cans_dog_food + more_cans_cat_food) :
  d = 7 :=
by
  sorry

end dog_food_packages_l75_75468


namespace triangle_sides_arithmetic_progression_l75_75475

theorem triangle_sides_arithmetic_progression (a d : ℤ) (h : 3 * a = 15) (h1 : a > 0) (h2 : d ≥ 0) :
  (a - d = 5 ∨ a - d = 4 ∨ a - d = 3) ∧ 
  (a = 5) ∧ 
  (a + d = 5 ∨ a + d = 6 ∨ a + d = 7) := 
  sorry

end triangle_sides_arithmetic_progression_l75_75475


namespace turtles_remaining_on_log_l75_75122

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l75_75122


namespace rhombus_diagonal_sum_maximum_l75_75770

theorem rhombus_diagonal_sum_maximum 
    (x y : ℝ) 
    (h1 : x^2 + y^2 = 100) 
    (h2 : x ≥ 6) 
    (h3 : y ≤ 6) : 
    x + y = 14 :=
sorry

end rhombus_diagonal_sum_maximum_l75_75770


namespace sequence_bound_l75_75076

-- Define the conditions
variables (a : ℕ → ℝ)
variables (h_nonneg : ∀ n, 0 ≤ a n)
variables (h_additive : ∀ m n, a (n + m) ≤ a n + a m)

-- The theorem statement
theorem sequence_bound (n m : ℕ) (h : n ≥ m) : 
  a n ≤ m * a 1 + ((n.to_real / m.to_real) - 1) * a m :=
by
  sorry

end sequence_bound_l75_75076


namespace find_principal_sum_l75_75793

theorem find_principal_sum (R P : ℝ) 
  (h1 : (3 * P * (R + 1) / 100 - 3 * P * R / 100) = 72) : 
  P = 2400 := 
by 
  sorry

end find_principal_sum_l75_75793


namespace players_on_team_are_4_l75_75330

noncomputable def number_of_players (score_old_record : ℕ) (rounds : ℕ) (score_first_9_rounds : ℕ) (final_round_diff : ℕ) :=
  let points_needed := score_old_record * rounds
  let points_final_needed := score_old_record - final_round_diff
  let total_points_needed := points_needed * 1
  let final_round_points_needed := total_points_needed - score_first_9_rounds
  let P := final_round_points_needed / points_final_needed
  P

theorem players_on_team_are_4 :
  number_of_players 287 10 10440 27 = 4 :=
by
  sorry

end players_on_team_are_4_l75_75330


namespace smallest_side_for_table_rotation_l75_75145

theorem smallest_side_for_table_rotation (S : ℕ) : (S ≥ Int.ofNat (Nat.sqrt (8^2 + 12^2) + 1)) → S = 15 := 
by
  sorry

end smallest_side_for_table_rotation_l75_75145


namespace intersection_P_Q_l75_75371

open Set

-- Definitions for the conditions
def P : Set ℝ := {x | x + 2 ≥ x^2}
def Q : Set ℕ := {x | x ≤ 3}

-- Proof problem statement: Prove P ∩ Q = {0, 1, 2}
theorem intersection_P_Q : P ∩ Q = {0, 1, 2} :=
  sorry

end intersection_P_Q_l75_75371


namespace geometric_seq_a7_l75_75661

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75661


namespace sequence_a8_value_l75_75185

theorem sequence_a8_value :
  ∃ a : ℕ → ℚ, a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) / a n = n / (n + 1)) ∧ a 8 = 1 / 8 :=
by
  -- To be proved
  sorry

end sequence_a8_value_l75_75185


namespace simplify_sqrt_expression_l75_75317

theorem simplify_sqrt_expression (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by
  sorry

end simplify_sqrt_expression_l75_75317


namespace evaluate_composite_function_l75_75332

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_composite_function : g (h 2) = 5288 := by
  sorry

end evaluate_composite_function_l75_75332


namespace larger_number_l75_75423

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75423


namespace irrational_number_problem_l75_75786

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_problem :
  ∀ x ∈ ({(0.4 : ℝ), (2 / 3 : ℝ), (2 : ℝ), - (Real.sqrt 5)} : Set ℝ), 
  is_irrational x ↔ x = - (Real.sqrt 5) :=
by
  intros x hx
  -- Other proof steps can go here
  sorry

end irrational_number_problem_l75_75786


namespace arithmetic_expression_l75_75600

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end arithmetic_expression_l75_75600


namespace range_of_a_l75_75985

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3) ∧ (x - a > 0)) ↔ (a ≤ -1) :=
sorry

end range_of_a_l75_75985


namespace find_number_l75_75115

-- Define the conditions: 0.80 * x - 20 = 60
variables (x : ℝ)
axiom condition : 0.80 * x - 20 = 60

-- State the theorem that x = 100 given the condition
theorem find_number : x = 100 :=
by
  sorry

end find_number_l75_75115


namespace checker_moves_10_cells_l75_75617

theorem checker_moves_10_cells :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) ∧ a 10 = 89 :=
by
  -- mathematical proof goes here
  sorry

end checker_moves_10_cells_l75_75617


namespace ferris_wheel_seat_capacity_l75_75899

theorem ferris_wheel_seat_capacity
  (total_seats : ℕ)
  (broken_seats : ℕ)
  (total_people : ℕ)
  (seats_available : ℕ)
  (people_per_seat : ℕ)
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : total_people = 120)
  (h4 : seats_available = total_seats - broken_seats)
  (h5 : people_per_seat = total_people / seats_available) :
  people_per_seat = 15 := 
by sorry

end ferris_wheel_seat_capacity_l75_75899


namespace rect_side_ratio_square_l75_75464

theorem rect_side_ratio_square (a b d : ℝ) (h1 : b = 2 * a) (h2 : d = a * Real.sqrt 5) : (b / a) ^ 2 = 4 := 
by sorry

end rect_side_ratio_square_l75_75464


namespace candy_given_l75_75167

theorem candy_given (A R G : ℕ) (h1 : A = 15) (h2 : R = 9) : G = 6 :=
by
  sorry

end candy_given_l75_75167


namespace min_value_eq_l75_75042

open Real
open Classical

noncomputable def min_value (x y : ℝ) : ℝ := x + 4 * y

theorem min_value_eq :
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / x + 1 / (2 * y) = 1) → (min_value x y) = 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_eq_l75_75042


namespace range_of_t_l75_75355

variable {f : ℝ → ℝ}

theorem range_of_t (h₁ : ∀ x y : ℝ, x < y → f x ≥ f y) (h₂ : ∀ t : ℝ, f (t^2) < f t) : 
  ∀ t : ℝ, f (t^2) < f t ↔ (t < 0 ∨ t > 1) := 
by 
  sorry

end range_of_t_l75_75355


namespace find_x_l75_75034

theorem find_x (y : ℝ) (x : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  x = (-19 + Real.sqrt 329) / 16 ∨ x = (-19 - Real.sqrt 329) / 16 :=
by
  sorry

end find_x_l75_75034


namespace division_correct_l75_75293

theorem division_correct :
  250 / (15 + 13 * 3^2) = 125 / 66 :=
by
  -- The proof steps can be filled in here.
  sorry

end division_correct_l75_75293


namespace hexagon_piece_area_l75_75284

theorem hexagon_piece_area (A : ℝ) (n : ℕ) (h1 : A = 21.12) (h2 : n = 6) : 
  A / n = 3.52 :=
by
  -- The proof will go here
  sorry

end hexagon_piece_area_l75_75284


namespace ice_cream_sandwiches_each_l75_75439

theorem ice_cream_sandwiches_each (total_ice_cream_sandwiches : ℕ) (number_of_nieces : ℕ) 
  (h1 : total_ice_cream_sandwiches = 143) (h2 : number_of_nieces = 11) : 
  total_ice_cream_sandwiches / number_of_nieces = 13 :=
by
  sorry

end ice_cream_sandwiches_each_l75_75439


namespace ranges_of_a_and_m_l75_75710

open Set Real

def A : Set Real := {x | x^2 - 3*x + 2 = 0}
def B (a : Real) : Set Real := {x | x^2 - a*x + a - 1 = 0}
def C (m : Real) : Set Real := {x | x^2 - m*x + 2 = 0}

theorem ranges_of_a_and_m (a m : Real) :
  A ∪ B a = A → A ∩ C m = C m → (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2*sqrt 2 < m ∧ m < 2*sqrt 2)) :=
by
  have hA : A = {1, 2} := sorry
  sorry

end ranges_of_a_and_m_l75_75710


namespace ratio_of_u_to_v_l75_75777

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l75_75777


namespace factor_expression_l75_75028

theorem factor_expression (b : ℤ) : 52 * b ^ 2 + 208 * b = 52 * b * (b + 4) := 
by {
  sorry
}

end factor_expression_l75_75028


namespace intersection_complement_l75_75502

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := { x | x ≥ 0 }

def B : Set ℝ := { y | y ≤ 0 }

theorem intersection_complement (U : Type) [TopologicalSpace U] : 
  A ∩ (compl B) = { x | x > 0 } :=
by
  sorry

end intersection_complement_l75_75502


namespace work_completion_days_l75_75606

theorem work_completion_days (x : ℕ) 
  (h1 : (1 : ℚ) / x + 1 / 9 = 1 / 6) :
  x = 18 := 
sorry

end work_completion_days_l75_75606


namespace small_square_perimeter_l75_75939

-- Condition Definitions
def perimeter_difference := 17
def side_length_of_square (x : ℝ) := 2 * x = perimeter_difference

-- Theorem Statement
theorem small_square_perimeter (x : ℝ) (h : side_length_of_square x) : 4 * x = 34 :=
by
  sorry

end small_square_perimeter_l75_75939


namespace shorter_tree_height_l75_75246

theorem shorter_tree_height
  (s : ℝ)
  (h₁ : ∀ s, s > 0 )
  (h₂ : s + (s + 20) = 240)
  (h₃ : s / (s + 20) = 5 / 7) :
  s = 110 :=
by
sorry

end shorter_tree_height_l75_75246


namespace probability_of_three_positive_answers_l75_75067

noncomputable def probability_exactly_three_positive_answers : ℚ :=
  (7.choose 3) * (3/7)^3 * (4/7)^4

theorem probability_of_three_positive_answers :
  probability_exactly_three_positive_answers = 242520 / 823543 :=
by
  unfold probability_exactly_three_positive_answers
  sorry

end probability_of_three_positive_answers_l75_75067


namespace sequence_inequality_l75_75077

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end sequence_inequality_l75_75077


namespace arcsin_eq_pi_div_two_solve_l75_75748

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l75_75748


namespace ladder_base_distance_l75_75938

noncomputable def length_of_ladder : ℝ := 8.5
noncomputable def height_on_wall : ℝ := 7.5

theorem ladder_base_distance (x : ℝ) (h : x ^ 2 + height_on_wall ^ 2 = length_of_ladder ^ 2) :
  x = 4 :=
by sorry

end ladder_base_distance_l75_75938


namespace max_expression_value_l75_75312

theorem max_expression_value (a b c : ℝ) (hb : b > a) (ha : a > c) (hb_ne : b ≠ 0) :
  ∃ M, M = 27 ∧ (∀ a b c, b > a → a > c → b ≠ 0 → (∃ M, (2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2 ≤ M * b^2) → M ≤ 27) :=
  sorry

end max_expression_value_l75_75312


namespace analytical_expression_satisfies_conditions_l75_75842

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

theorem analytical_expression_satisfies_conditions :
  is_increasing f ∧ (∀ x : ℝ, f x > 1) :=
by
  sorry

end analytical_expression_satisfies_conditions_l75_75842


namespace larger_number_is_23_l75_75430

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75430


namespace derivative_at_0_5_l75_75695

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := -2

-- State the theorem
theorem derivative_at_0_5 : f' 0.5 = -2 :=
by {
  -- Proof placeholder
  sorry
}

end derivative_at_0_5_l75_75695


namespace prove_inequality_l75_75763

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l75_75763


namespace f_log₂_20_l75_75032

noncomputable def f (x : ℝ) : ℝ := sorry -- This is a placeholder for the function f.

lemma f_neg (x : ℝ) : f (-x) = -f (x) := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (1 - x) := sorry
lemma f_special (x : ℝ) (hx : -1 < x ∧ x < 0) : f (x) = 2^x + 6 / 5 := sorry

theorem f_log₂_20 : f (Real.log 20 / Real.log 2) = -2 := by
  -- Proof details would go here.
  sorry

end f_log₂_20_l75_75032


namespace coin_problem_l75_75616

theorem coin_problem :
  ∃ n : ℕ, (n % 8 = 5) ∧ (n % 7 = 2) ∧ (n % 9 = 1) := 
sorry

end coin_problem_l75_75616


namespace possible_vertex_angles_of_isosceles_triangle_l75_75089

def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (β = γ) ∨ (γ = α)

def altitude_half_side (α β γ a b c : ℝ) : Prop :=
  (a = α / 2) ∨ (b = β / 2) ∨ (c = γ / 2)

theorem possible_vertex_angles_of_isosceles_triangle (α β γ a b c : ℝ) :
  isosceles_triangle α β γ →
  altitude_half_side α β γ a b c →
  α = 30 ∨ α = 120 ∨ α = 150 :=
by
  sorry

end possible_vertex_angles_of_isosceles_triangle_l75_75089


namespace sum_of_three_squares_power_l75_75352

theorem sum_of_three_squares_power (n a b c k : ℕ) (h : n = a^2 + b^2 + c^2) (h_pos : n > 0) (k_pos : k > 0) :
  ∃ A B C : ℕ, n^(2*k) = A^2 + B^2 + C^2 :=
by
  sorry

end sum_of_three_squares_power_l75_75352


namespace village_population_l75_75724

theorem village_population (initial_population: ℕ) (died_percent left_percent: ℕ) (remaining_population current_population: ℕ)
    (h1: initial_population = 6324)
    (h2: died_percent = 10)
    (h3: left_percent = 20)
    (h4: remaining_population = initial_population - (initial_population * died_percent / 100))
    (h5: current_population = remaining_population - (remaining_population * left_percent / 100)):
  current_population = 4554 :=
  by
    sorry

end village_population_l75_75724


namespace mary_total_nickels_l75_75553

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l75_75553


namespace simplify_expression_l75_75324

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l75_75324


namespace F_eq_arithmetic_mean_l75_75220

open Finset

variable {n r : ℕ}

-- Definition of F(n, r)
noncomputable def F (n r : ℕ) : ℚ :=
  let subsets := (powerset (range n.succ)).filter (λ s => card s = r ∧ s.nonempty) in
  (subsets.sum (λ s, (s.min' (by simp) : ℚ))) / (subsets.card : ℚ)

-- Theorem we want to prove
theorem F_eq_arithmetic_mean {n r : ℕ} (h₁ : 1 ≤ r) (h₂ : r ≤ n) : 
    F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := 
  sorry

end F_eq_arithmetic_mean_l75_75220


namespace min_distance_between_curves_l75_75506

theorem min_distance_between_curves :
  let P := (xP : ℝ) → (xP, xP^2 + 2)
  let Q := (xQ : ℝ) → (xQ, Real.sqrt(xQ - 2))
  ∃ xq, ∃ xp, (xq >= 2) → (xp ≥ 0) → 
    (|xP - xQ| + |xP^2 + 2 - Real.sqrt(xQ - 2)|) / Real.sqrt(2) = (7 * Real.sqrt(2)) / 4 :=
sorry

end min_distance_between_curves_l75_75506


namespace evaluate_powers_of_i_l75_75952

-- Definitions based on the given conditions
noncomputable def i_power (n : ℤ) := 
  if n % 4 = 0 then (1 : ℂ)
  else if n % 4 = 1 then complex.I
  else if n % 4 = 2 then -1
  else -complex.I

-- Statement of the problem
theorem evaluate_powers_of_i :
  i_power 14760 + i_power 14761 + i_power 14762 + i_power 14763 = 0 :=
by
  sorry

end evaluate_powers_of_i_l75_75952


namespace problem_equivalent_proof_l75_75690

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75690


namespace average_salary_of_employees_l75_75752

theorem average_salary_of_employees (A : ℝ) 
  (h1 : (20 : ℝ) * A + 3400 = 21 * (A + 100)) : 
  A = 1300 := 
by 
  -- proof goes here 
  sorry

end average_salary_of_employees_l75_75752


namespace man_speed_against_current_proof_l75_75132

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l75_75132


namespace combined_time_to_finish_cereal_l75_75893

theorem combined_time_to_finish_cereal : 
  let rate_fat := 1 / 15
  let rate_thin := 1 / 45
  let combined_rate := rate_fat + rate_thin
  let time_needed := 4 / combined_rate
  time_needed = 45 := 
by 
  sorry

end combined_time_to_finish_cereal_l75_75893


namespace proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l75_75162

noncomputable def problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : Prop :=
  y ≤ 4.5

noncomputable def problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : Prop :=
  y ≥ -8

noncomputable def problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : Prop :=
  -1 ≤ y ∧ y ≤ 1

noncomputable def problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : Prop :=
  y ≤ 1/3

-- Proving that the properties hold:
theorem proof_of_problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : problem1 x y h :=
  sorry

theorem proof_of_problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : problem2 x y h :=
  sorry

theorem proof_of_problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : problem3 x y h :=
  sorry

theorem proof_of_problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : problem4 x y h :=
  sorry

end proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l75_75162


namespace ratio_of_red_to_black_l75_75201

theorem ratio_of_red_to_black (r b : ℕ) (h_r : r = 26) (h_b : b = 70) :
  r / Nat.gcd r b = 13 ∧ b / Nat.gcd r b = 35 :=
by
  sorry

end ratio_of_red_to_black_l75_75201


namespace num_perfect_square_factors_of_450_l75_75521

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end num_perfect_square_factors_of_450_l75_75521


namespace correct_line_equation_l75_75397

theorem correct_line_equation :
  ∃ (c : ℝ), (∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0 → 2 * x - 3 * y + c = 0 ∧ 2 * (-1) - 3 * 2 + c = 0) ∧ c = 8 :=
by
  use 8
  sorry

end correct_line_equation_l75_75397


namespace trajectory_of_P_l75_75245

theorem trajectory_of_P (M P : ℝ × ℝ) (OM OP : ℝ) (x y : ℝ) :
  (M = (4, y)) →
  (P = (x, y)) →
  (OM = Real.sqrt (4^2 + y^2)) →
  (OP = Real.sqrt ((x - 4)^2 + y^2)) →
  (OM * OP = 16) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end trajectory_of_P_l75_75245


namespace eval_expression_l75_75026

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l75_75026


namespace total_people_in_church_l75_75218

def c : ℕ := 80
def m : ℕ := 60
def f : ℕ := 60

theorem total_people_in_church : c + m + f = 200 :=
by
  sorry

end total_people_in_church_l75_75218


namespace calculate_minus_one_minus_two_l75_75295

theorem calculate_minus_one_minus_two : -1 - 2 = -3 := by
  sorry

end calculate_minus_one_minus_two_l75_75295


namespace ratio_of_logs_l75_75565

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem ratio_of_logs (a b: ℝ) (h1 : log_base 8 a = log_base 18 b) 
    (h2 : log_base 18 b = log_base 32 (a + b)) 
    (hpos : 0 < a ∧ 0 < b) :
    b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) :=
by 
    sorry

end ratio_of_logs_l75_75565


namespace sum_of_numbers_l75_75774

variable {R : Type*} [LinearOrderedField R]

theorem sum_of_numbers (x y : R) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 :=
by
  sorry

end sum_of_numbers_l75_75774


namespace thirteen_y_minus_x_l75_75454

theorem thirteen_y_minus_x (x y : ℤ) (hx1 : x = 11 * y + 4) (hx2 : 2 * x = 8 * (3 * y) + 3) : 13 * y - x = 1 :=
by
  sorry

end thirteen_y_minus_x_l75_75454


namespace least_possible_number_of_straight_lines_l75_75923

theorem least_possible_number_of_straight_lines :
  ∀ (segments : Fin 31 → (Fin 2 → ℝ)), 
  (∀ i j, i ≠ j → (segments i 0 = segments j 0) ∧ (segments i 1 = segments j 1) → false) →
  ∃ (lines_count : ℕ), lines_count = 16 :=
by
  sorry

end least_possible_number_of_straight_lines_l75_75923


namespace XiaoMingAgeWhenFathersAgeIsFiveTimes_l75_75267

-- Define the conditions
def XiaoMingAgeCurrent : ℕ := 12
def FatherAgeCurrent : ℕ := 40

-- Prove the question given the conditions
theorem XiaoMingAgeWhenFathersAgeIsFiveTimes : 
  ∃ (x : ℕ), (FatherAgeCurrent - x) = 5 * x - XiaoMingAgeCurrent ∧ x = 7 := 
by
  use 7
  sorry

end XiaoMingAgeWhenFathersAgeIsFiveTimes_l75_75267


namespace max_g_on_interval_l75_75496

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 :=
by
  sorry

end max_g_on_interval_l75_75496


namespace simplify_expression_l75_75321

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l75_75321


namespace simplify_expression_l75_75325

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l75_75325


namespace outlet_pipe_empties_2_over_3_in_16_min_l75_75004

def outlet_pipe_part_empty_in_t (t : ℕ) (part_per_8_min : ℚ) : ℚ :=
  (part_per_8_min / 8) * t

theorem outlet_pipe_empties_2_over_3_in_16_min (
  part_per_8_min : ℚ := 1/3
) : outlet_pipe_part_empty_in_t 16 part_per_8_min = 2/3 :=
by
  sorry

end outlet_pipe_empties_2_over_3_in_16_min_l75_75004


namespace heidi_and_karl_painting_l75_75529

-- Given conditions
def heidi_paint_rate := 1 / 60 -- Rate at which Heidi paints, in walls per minute
def karl_paint_rate := 2 * heidi_paint_rate -- Rate at which Karl paints, in walls per minute
def painting_time := 20 -- Time spent painting, in minutes

-- Prove the amount of each wall painted
theorem heidi_and_karl_painting :
  (heidi_paint_rate * painting_time = 1 / 3) ∧ (karl_paint_rate * painting_time = 2 / 3) :=
sorry

end heidi_and_karl_painting_l75_75529


namespace weight_of_new_person_l75_75453

-- Define the given conditions
variables (avg_increase : ℝ) (num_people : ℕ) (replaced_weight : ℝ)
variable (new_weight : ℝ)

-- These are the conditions directly from the problem
axiom avg_weight_increase : avg_increase = 4.5
axiom number_of_people : num_people = 6
axiom person_to_replace_weight : replaced_weight = 75

-- Mathematical equivalent of the proof problem
theorem weight_of_new_person :
  new_weight = replaced_weight + avg_increase * num_people := 
sorry

end weight_of_new_person_l75_75453


namespace max_fridays_in_year_l75_75713

theorem max_fridays_in_year (days_in_common_year days_in_leap_year : ℕ) 
  (h_common_year : days_in_common_year = 365)
  (h_leap_year : days_in_leap_year = 366) : 
  ∃ (max_fridays : ℕ), max_fridays = 53 := 
by
  existsi 53
  sorry

end max_fridays_in_year_l75_75713


namespace number_of_persons_l75_75395

-- Definitions of the given conditions
def average : ℕ := 15
def average_5 : ℕ := 14
def sum_5 : ℕ := 5 * average_5
def average_9 : ℕ := 16
def sum_9 : ℕ := 9 * average_9
def age_15th : ℕ := 41
def total_sum : ℕ := sum_5 + sum_9 + age_15th

-- The main theorem stating the equivalence
theorem number_of_persons (N : ℕ) (h_average : average * N = total_sum) : N = 17 :=
by
  -- Proof goes here
  sorry

end number_of_persons_l75_75395


namespace problem_equivalent_proof_l75_75691

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l75_75691


namespace mike_earnings_l75_75557

def prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mike_earnings :
  List.sum prices = 75 :=
by
  sorry

end mike_earnings_l75_75557


namespace jill_has_1_more_peach_than_jake_l75_75877

theorem jill_has_1_more_peach_than_jake
    (jill_peaches : ℕ)
    (steven_peaches : ℕ)
    (jake_peaches : ℕ)
    (h1 : jake_peaches = steven_peaches - 16)
    (h2 : steven_peaches = jill_peaches + 15)
    (h3 : jill_peaches = 12) :
    12 - (steven_peaches - 16) = 1 := 
sorry

end jill_has_1_more_peach_than_jake_l75_75877


namespace correct_division_result_l75_75731

theorem correct_division_result (x : ℝ) 
  (h : (x - 14) / 5 = 11) : (x - 5) / 7 = 64 / 7 :=
by
  sorry

end correct_division_result_l75_75731


namespace jo_thinking_number_l75_75997

theorem jo_thinking_number 
  (n : ℕ) 
  (h1 : n < 100) 
  (h2 : n % 8 = 7) 
  (h3 : n % 7 = 4) 
  : n = 95 :=
sorry

end jo_thinking_number_l75_75997


namespace k_2_sufficient_but_not_necessary_l75_75517

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (1, k^2 - 1) - (2, 1)

def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

theorem k_2_sufficient_but_not_necessary (k : ℝ) :
  k = 2 → perpendicular vector_a (vector_b k) ∧ ∃ k, not (k = 2) ∧ perpendicular vector_a (vector_b k) :=
by
  sorry

end k_2_sufficient_but_not_necessary_l75_75517


namespace mary_total_nickels_l75_75555

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l75_75555


namespace exists_i_with_α_close_to_60_l75_75698

noncomputable def α : ℕ → ℝ := sorry  -- Placeholder for the function α

theorem exists_i_with_α_close_to_60 :
  ∃ i : ℕ, abs (α i - 60) < 1
:= sorry

end exists_i_with_α_close_to_60_l75_75698


namespace task1_task2_l75_75957

/-- Given conditions -/
def cost_A : Nat := 30
def cost_B : Nat := 40
def sell_A : Nat := 35
def sell_B : Nat := 50
def max_cost : Nat := 1550
def min_profit : Nat := 365
def total_cars : Nat := 40

/-- Task 1: Prove maximum B-type cars produced if 10 A-type cars are produced -/
theorem task1 (A: Nat) (B: Nat) (hA: A = 10) (hC: cost_A * A + cost_B * B ≤ max_cost) : B ≤ 31 :=
by sorry

/-- Task 2: Prove the possible production plans producing 40 cars meeting profit and cost constraints -/
theorem task2 (A: Nat) (B: Nat) (hTotal: A + B = total_cars)
(hCost: cost_A * A + cost_B * B ≤ max_cost) 
(hProfit: (sell_A - cost_A) * A + (sell_B - cost_B) * B ≥ min_profit) : 
  (A = 5 ∧ B = 35) ∨ (A = 6 ∧ B = 34) ∨ (A = 7 ∧ B = 33) 
∧ (375 ≤ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35 ∧ 375 ≥ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35) :=
by sorry

end task1_task2_l75_75957


namespace find_a_l75_75530

theorem find_a (a : ℝ) (h₁ : a > 1) (h₂ : (∀ x : ℝ, a^3 = 8)) : a = 2 :=
by
  sorry

end find_a_l75_75530


namespace total_pieces_of_pizza_l75_75305

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l75_75305


namespace no_real_solution_l75_75973

theorem no_real_solution (x : ℝ) : ¬ ∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -|x| := 
sorry

end no_real_solution_l75_75973


namespace train_passes_pole_in_10_seconds_l75_75608

theorem train_passes_pole_in_10_seconds :
  let L := 150 -- length of the train in meters
  let S_kmhr := 54 -- speed in kilometers per hour
  let S_ms := S_kmhr * 1000 / 3600 -- speed in meters per second
  (L / S_ms = 10) := 
by
  sorry

end train_passes_pole_in_10_seconds_l75_75608


namespace find_f8_l75_75977

theorem find_f8 (f : ℕ → ℕ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : f 8 = 26 :=
by
  sorry

end find_f8_l75_75977


namespace total_balls_l75_75017

theorem total_balls {balls_per_box boxes : ℕ} (h1 : balls_per_box = 3) (h2 : boxes = 2) : balls_per_box * boxes = 6 :=
by
  sorry

end total_balls_l75_75017


namespace eval_expression_l75_75022

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l75_75022


namespace range_of_a_l75_75983

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ (-9 < a ∧ a < 5/3) :=
by
  sorry

end range_of_a_l75_75983


namespace product_mod_equiv_one_l75_75879

open Nat

theorem product_mod_equiv_one :
  let A := {n : ℕ | 1 ≤ n ∧ n ≤ 2009^2009}
  let S := {n ∈ A | gcd n (2009^2009) = 1}
  let P := ∏ x in S, x
  in P ≡ 1 [MOD 2009^2009] := 
by
  sorry

end product_mod_equiv_one_l75_75879


namespace karen_bonus_problem_l75_75215

theorem karen_bonus_problem (n already_graded last_two target : ℕ) (h_already_graded : already_graded = 8)
  (h_last_two : last_two = 290) (h_target : target = 600) (max_score : ℕ)
  (h_max_score : max_score = 150) (required_avg : ℕ) (h_required_avg : required_avg = 75) :
  ∃ A : ℕ, (A = 70) ∧ (target = 600) ∧ (last_two = 290) ∧ (already_graded = 8) ∧
  (required_avg = 75) := by
  sorry

end karen_bonus_problem_l75_75215


namespace systematic_sampling_interval_l75_75772

-- Definition of the population size and sample size
def populationSize : Nat := 800
def sampleSize : Nat := 40

-- The main theorem stating that the interval k in systematic sampling is 20
theorem systematic_sampling_interval : populationSize / sampleSize = 20 := by
  sorry

end systematic_sampling_interval_l75_75772


namespace maximum_value_of_expression_l75_75223

noncomputable def max_value (x y z w : ℝ) : ℝ := 2 * x + 3 * y + 5 * z - 4 * w

theorem maximum_value_of_expression 
  (x y z w : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 + 16 * w^2 = 4) : 
  max_value x y z w ≤ 6 * Real.sqrt 6 :=
sorry

end maximum_value_of_expression_l75_75223


namespace inequality_proof_equality_condition_l75_75798

variables {a b c x y z : ℕ}

theorem inequality_proof (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 ≤ (c + z) ^ 2 :=
sorry

theorem equality_condition (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 = (c + z) ^ 2 ↔ a * z = c * x ∧ a * y = b * x :=
sorry

end inequality_proof_equality_condition_l75_75798


namespace larger_number_is_23_l75_75419

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75419


namespace probability_of_grid_being_black_l75_75456

noncomputable def probability_grid_black_after_rotation : ℚ := sorry

theorem probability_of_grid_being_black:
  probability_grid_black_after_rotation = 429 / 21845 :=
sorry

end probability_of_grid_being_black_l75_75456


namespace company_a_percentage_l75_75154

theorem company_a_percentage (total_profits: ℝ) (p_b: ℝ) (profit_b: ℝ) (profit_a: ℝ) :
  p_b = 0.40 →
  profit_b = 60000 →
  profit_a = 90000 →
  total_profits = profit_b / p_b →
  (profit_a / total_profits) * 100 = 60 :=
by
  intros h_pb h_profit_b h_profit_a h_total_profits
  sorry

end company_a_percentage_l75_75154


namespace value_for_real_value_for_pure_imaginary_l75_75649

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def value_conditions (k : ℝ) : ℂ := ⟨k^2 - 3*k - 4, k^2 - 5*k - 6⟩

theorem value_for_real (k : ℝ) : is_real (value_conditions k) ↔ (k = 6 ∨ k = -1) :=
by
  sorry

theorem value_for_pure_imaginary (k : ℝ) : is_pure_imaginary (value_conditions k) ↔ (k = 4) :=
by
  sorry

end value_for_real_value_for_pure_imaginary_l75_75649


namespace smallest_valid_N_l75_75809

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (fun d => d > 1 ∧ d < n ∧ n % d = 0)

def two_largest (l : List ℕ) : ℕ × ℕ :=
  match l.reverse with
  | a :: b :: _ => (a, b)
  | _           => (0, 0)

def two_smallest (l : List ℕ) : ℕ × ℕ :=
  match l with
  | a :: b :: _ => (a, b)
  | _           => (0, 0)

def valid_number (N : ℕ) : Prop :=
  N % 10 = 5 ∧
  let divisors := proper_divisors N in
  let (d_max1, d_max2) := two_largest divisors in
  let (d_min1, d_min2) := two_smallest divisors in
  (d_max1 + d_max2) % (d_min1 + d_min2) ≠ 0

theorem smallest_valid_N : ∃ N : ℕ, valid_number N ∧ N = 725 :=
by
  existsi 725
  unfold valid_number
  unfold proper_divisors
  unfold two_largest
  unfold two_smallest
  sorry

end smallest_valid_N_l75_75809


namespace min_students_wearing_both_l75_75359

theorem min_students_wearing_both (n : ℕ) (H1 : n % 3 = 0) (H2 : n % 6 = 0) (H3 : n = 6) :
  ∃ x : ℕ, x = 1 ∧ 
           (∃ b : ℕ, b = n / 3) ∧
           (∃ r : ℕ, r = 5 * n / 6) ∧
           6 = b + r - x :=
by sorry

end min_students_wearing_both_l75_75359


namespace eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l75_75972

theorem eight_digit_numbers_with_012 :
  let total_sequences := 3^8 
  let invalid_sequences := 3^7 
  total_sequences - invalid_sequences = 4374 :=
by sorry

theorem eight_digit_numbers_with_00012222 :
  let total_sequences := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)
  let invalid_sequences := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 4)
  total_sequences - invalid_sequences = 175 :=
by sorry

theorem eight_digit_numbers_starting_with_1_0002222 :
  let number_starting_with_1 := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 4)
  number_starting_with_1 = 35 :=
by sorry

end eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l75_75972


namespace total_expenses_l75_75543

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l75_75543


namespace mary_total_nickels_l75_75554

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l75_75554


namespace common_integer_solutions_l75_75857

theorem common_integer_solutions
    (y : ℤ)
    (h1 : -4 * y ≥ 2 * y + 10)
    (h2 : -3 * y ≤ 15)
    (h3 : -5 * y ≥ 3 * y + 24)
    (h4 : y ≤ -1) :
  y = -3 ∨ y = -4 ∨ y = -5 :=
by 
  sorry

end common_integer_solutions_l75_75857


namespace iterative_average_difference_l75_75818

theorem iterative_average_difference :
  let numbers : List ℕ := [2, 4, 6, 8, 10] 
  let avg2 (a b : ℝ) := (a + b) / 2
  let avg (init : ℝ) (lst : List ℕ) := lst.foldl (λ acc x => avg2 acc x) init
  let max_avg := avg 2 [4, 6, 8, 10]
  let min_avg := avg 10 [8, 6, 4, 2] 
  max_avg - min_avg = 4.25 := 
by
  sorry

end iterative_average_difference_l75_75818


namespace clerical_percentage_after_reduction_l75_75740

theorem clerical_percentage_after_reduction
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (reduction_fraction : ℚ)
  (h1 : total_employees = 3600)
  (h2 : clerical_fraction = 1/4)
  (h3 : reduction_fraction = 1/4) : 
  let initial_clerical := clerical_fraction * total_employees
  let reduced_clerical := (1 - reduction_fraction) * initial_clerical
  let let_go := initial_clerical - reduced_clerical
  let new_total := total_employees - let_go
  let clerical_percentage := (reduced_clerical / new_total) * 100
  clerical_percentage = 20 :=
by sorry

end clerical_percentage_after_reduction_l75_75740


namespace zoe_bought_bottles_l75_75450

theorem zoe_bought_bottles
  (initial_bottles : ℕ)
  (drank_bottles : ℕ)
  (current_bottles : ℕ)
  (initial_bottles_eq : initial_bottles = 42)
  (drank_bottles_eq : drank_bottles = 25)
  (current_bottles_eq : current_bottles = 47) :
  ∃ bought_bottles : ℕ, bought_bottles = 30 :=
by
  sorry

end zoe_bought_bottles_l75_75450


namespace parallel_vectors_l75_75186

variable {k m : ℝ}

theorem parallel_vectors (h₁ : (2 : ℝ) = k * m) (h₂ : m = 2 * k) : m = 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l75_75186


namespace count_positive_integers_N_number_of_N_l75_75526

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l75_75526


namespace sugar_cheaper_than_apples_l75_75828

/-- Given conditions about the prices and quantities of items that Fabian wants to buy,
    prove the price difference between one pack of sugar and one kilogram of apples. --/
theorem sugar_cheaper_than_apples
  (price_kg_apples : ℝ)
  (price_kg_walnuts : ℝ)
  (total_cost : ℝ)
  (cost_diff : ℝ)
  (num_kg_apples : ℕ := 5)
  (num_packs_sugar : ℕ := 3)
  (num_kg_walnuts : ℝ := 0.5)
  (price_kg_apples_val : price_kg_apples = 2)
  (price_kg_walnuts_val : price_kg_walnuts = 6)
  (total_cost_val : total_cost = 16) :
  cost_diff = price_kg_apples - (total_cost - (num_kg_apples * price_kg_apples + num_kg_walnuts * price_kg_walnuts))/num_packs_sugar → 
  cost_diff = 1 :=
by
  sorry

end sugar_cheaper_than_apples_l75_75828


namespace actual_cost_of_article_l75_75143

theorem actual_cost_of_article {x : ℝ} (h : 0.76 * x = 760) : x = 1000 :=
by
  sorry

end actual_cost_of_article_l75_75143


namespace remainder_of_N_l75_75823

-- Definition of the sequence constraints
def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ (∀ i, a i < 512) ∧ (∀ k, 1 ≤ k → k ≤ 9 → ∃ m, 0 ≤ m ∧ m ≤ k - 1 ∧ ((a k - 2 * a m) * (a k - 2 * a m - 1) = 0))

-- Defining N as the number of sequences that are valid.
noncomputable def N : ℕ :=
  Nat.factorial 10 - 2^9

-- The goal is to prove that N mod 1000 is 288
theorem remainder_of_N : N % 1000 = 288 :=
  sorry

end remainder_of_N_l75_75823


namespace maximum_n_for_positive_sum_l75_75204

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :=
  S n > 0

-- Definition of the arithmetic sequence properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d
  
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
variable (h₁ : a 1 > 0)
variable (h₅ : a 2016 + a 2017 > 0)
variable (h₆ : a 2016 * a 2017 < 0)

-- Add the definition of the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Prove the final statement
theorem maximum_n_for_positive_sum : max_n_for_positive_sum a S 4032 :=
by
  -- conditions to use in the proof
  have h₁ : a 1 > 0 := sorry
  have h₅ : a 2016 + a 2017 > 0 := sorry
  have h₆ : a 2016 * a 2017 < 0 := sorry
  -- positively bounded sum
  let Sn := sum_of_first_n_terms a
  -- proof to utilize Lean's capabilities, replace with actual proof later
  sorry

end maximum_n_for_positive_sum_l75_75204


namespace boats_solution_l75_75229

theorem boats_solution (x y : ℕ) (h1 : x + y = 42) (h2 : 6 * x = 8 * y) : x = 24 ∧ y = 18 :=
by
  sorry

end boats_solution_l75_75229


namespace circulation_ratio_l75_75238

variable (A : ℕ) -- Assuming A to be a natural number for simplicity

theorem circulation_ratio (h : ∀ t : ℕ, t = 1971 → t = 4 * A) : 4 / 13 = 4 / 13 := 
by
  sorry

end circulation_ratio_l75_75238


namespace Binkie_gemstones_l75_75156

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l75_75156


namespace geometric_sequence_common_ratio_l75_75512

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) (S_n : ℕ → ℝ)
  (h₁ : S_n 3 = a₁ + a₁ * q + a₁ * q ^ 2)
  (h₂ : S_n 2 = a₁ + a₁ * q)
  (h₃ : S_n 3 / S_n 2 = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l75_75512


namespace average_speed_of_participant_l75_75792

noncomputable def average_speed (d : ℝ) : ℝ :=
  let total_distance := 4 * d
  let total_time := (d / 6) + (d / 12) + (d / 18) + (d / 24)
  total_distance / total_time

theorem average_speed_of_participant :
  ∀ (d : ℝ), d > 0 → average_speed d = 11.52 :=
by
  intros d hd
  unfold average_speed
  sorry

end average_speed_of_participant_l75_75792


namespace parallel_vectors_l75_75712

noncomputable def vector_a : (ℤ × ℤ) := (1, 3)
noncomputable def vector_b (m : ℤ) : (ℤ × ℤ) := (-2, m)

theorem parallel_vectors (m : ℤ) (h : vector_a = (1, 3) ∧ vector_b m = (-2, m))
  (hp: ∃ k : ℤ, ∀ (a1 a2 b1 b2 : ℤ), (a1, a2) = vector_a ∧ (b1, b2) = (1 + k * (-2), 3 + k * m)):
  m = -6 :=
by
  sorry

end parallel_vectors_l75_75712


namespace fraction_value_l75_75331

variable (x y : ℝ)

theorem fraction_value (h : 1/x - 1/y = 3) : (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := 
by sorry

end fraction_value_l75_75331


namespace find_side_length_a_l75_75367

variable {a b c : ℝ}
variable {B : ℝ}

theorem find_side_length_a (h_b : b = 7) (h_c : c = 5) (h_B : B = 2 * Real.pi / 3) :
  a = 3 :=
sorry

end find_side_length_a_l75_75367


namespace no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l75_75110

theorem no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49 :
  ∀ n : ℕ, ¬ (∃ k : ℤ, (n^2 + 5 * n + 1) = 49 * k) :=
by
  sorry

end no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l75_75110


namespace riley_mistakes_l75_75360

theorem riley_mistakes :
  ∃ R O : ℕ, R + O = 17 ∧ O = 35 - ((35 - R) / 2 + 5) ∧ R = 3 := by
  sorry

end riley_mistakes_l75_75360


namespace number_of_perfect_square_factors_l75_75715

theorem number_of_perfect_square_factors :
  let n := (2^14) * (3^9) * (5^20)
  ∃ (count : ℕ), 
  (∀ (a : ℕ) (h : a ∣ n), (∃ k, a = k^2) → true) →
  count = 440 :=
by
  sorry

end number_of_perfect_square_factors_l75_75715


namespace parallelogram_height_l75_75165

theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 336) 
  (h_base : base = 14) 
  (h_formula : area = base * height) : 
  height = 24 := 
by 
  sorry

end parallelogram_height_l75_75165


namespace find_a7_l75_75658

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75658


namespace chromium_percentage_alloy_l75_75203

theorem chromium_percentage_alloy 
  (w1 w2 w3 w4 : ℝ)
  (p1 p2 p3 p4 : ℝ)
  (h_w1 : w1 = 15)
  (h_w2 : w2 = 30)
  (h_w3 : w3 = 10)
  (h_w4 : w4 = 5)
  (h_p1 : p1 = 0.12)
  (h_p2 : p2 = 0.08)
  (h_p3 : p3 = 0.15)
  (h_p4 : p4 = 0.20) :
  (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / (w1 + w2 + w3 + w4) * 100 = 11.17 := 
  sorry

end chromium_percentage_alloy_l75_75203


namespace combine_square_roots_l75_75787

def can_be_combined (x y: ℝ) : Prop :=
  ∃ k: ℝ, y = k * x

theorem combine_square_roots :
  let sqrt12 := 2 * Real.sqrt 3
  let sqrt1_3 := Real.sqrt 1 / Real.sqrt 3
  let sqrt18 := 3 * Real.sqrt 2
  let sqrt27 := 6 * Real.sqrt 3
  can_be_combined (Real.sqrt 3) sqrt12 ∧
  can_be_combined (Real.sqrt 3) sqrt1_3 ∧
  ¬ can_be_combined (Real.sqrt 3) sqrt18 ∧
  can_be_combined (Real.sqrt 3) sqrt27 :=
by
  sorry

end combine_square_roots_l75_75787


namespace number_of_days_l75_75066

theorem number_of_days (m1 d1 m2 d2 : ℕ) (h1 : m1 * d1 = m2 * d2) (k : ℕ) 
(h2 : m1 = 10) (h3 : d1 = 6) (h4 : m2 = 15) (h5 : k = 60) : 
d2 = 4 :=
by
  have : 10 * 6 = 60 := by sorry
  have : 15 * d2 = 60 := by sorry
  exact sorry

end number_of_days_l75_75066


namespace tangent_line_through_point_l75_75194

noncomputable def circle_tangent_line : Prop :=
  let C := fun x y : ℝ => (x - 2) ^ 2 + (y + 3) ^ 2 = 4 in
  let tangent : ℝ → ℝ → Prop := fun x y => (y = -1) ∨ (12 * x + 5 * y + 17 = 0) in
  ∀ x y : ℝ, (x = -1) → (y = -1) → (tangent x y)

theorem tangent_line_through_point (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  circle_tangent_line :=
by
  sorry

end tangent_line_through_point_l75_75194


namespace cubic_sum_identity_l75_75236

theorem cubic_sum_identity
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + ac + bc = -3)
  (h3 : abc = 9) :
  a^3 + b^3 + c^3 = 22 :=
by
  sorry

end cubic_sum_identity_l75_75236


namespace kevin_correct_answer_l75_75999

theorem kevin_correct_answer (k : ℝ) (h : (20 + 1) * (6 + k) = 126 + 21 * k) :
  (20 + 1 * 6 + k) = 21 := by
sorry

end kevin_correct_answer_l75_75999


namespace minimum_polynomial_degree_for_separation_l75_75088

open Polynomial

theorem minimum_polynomial_degree_for_separation {a : ℕ → ℝ} (h : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ (P : Polynomial ℝ), degree P ≤ 12 ∧
    (∀ i j, i ∈ {0, 1, 2, 3, 4, 5} → j ∈ {6, 7, 8, 9, 10, 11, 12} → eval a i P > eval a j P) :=
begin
  sorry
end

end minimum_polynomial_degree_for_separation_l75_75088


namespace pascal_identity_l75_75386

noncomputable def pascal_sequence (n : ℕ) (i : ℕ) : ℕ :=
  Nat.choose n i

theorem pascal_identity (n : ℕ) (h : n = 3004) :
  2 * ∑ i in Finset.range (n + 1), (pascal_sequence n i : ℝ) / (pascal_sequence (n + 1) i) -
  ∑ i in Finset.range n, (pascal_sequence (n - 1) i : ℝ) / (pascal_sequence n i) = 1503.5 :=
by
  rw h
  sorry

end pascal_identity_l75_75386


namespace annual_interest_correct_l75_75233

-- Define the conditions
def Rs_total : ℝ := 3400
def P1 : ℝ := 1300
def P2 : ℝ := Rs_total - P1
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

-- Define the interests
def Interest1 : ℝ := P1 * Rate1
def Interest2 : ℝ := P2 * Rate2

-- The total interest
def Total_Interest : ℝ := Interest1 + Interest2

-- The theorem to prove
theorem annual_interest_correct :
  Total_Interest = 144 :=
by
  sorry

end annual_interest_correct_l75_75233


namespace problem_B_false_l75_75014

def diamondsuit (x y : ℝ) : ℝ := abs (x + y - 1)

theorem problem_B_false : ∀ x y : ℝ, 2 * (diamondsuit x y) ≠ diamondsuit (2 * x) (2 * y) :=
by
  intro x y
  dsimp [diamondsuit]
  sorry

end problem_B_false_l75_75014


namespace morning_rowers_count_l75_75564

def number_afternoon_rowers : ℕ := 7
def total_rowers : ℕ := 60

def number_morning_rowers : ℕ :=
  total_rowers - number_afternoon_rowers

theorem morning_rowers_count :
  number_morning_rowers = 53 := by
  sorry

end morning_rowers_count_l75_75564


namespace scientific_notation_of_one_point_six_million_l75_75227

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l75_75227


namespace total_payment_l75_75588

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l75_75588


namespace roy_consumes_tablets_in_225_minutes_l75_75743

variables 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ)

def total_time_to_consume_all_tablets 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ) : ℕ :=
  (total_tablets - 1) * time_per_tablet

theorem roy_consumes_tablets_in_225_minutes 
  (h1 : total_tablets = 10) 
  (h2 : time_per_tablet = 25) : 
  total_time_to_consume_all_tablets total_tablets time_per_tablet = 225 :=
by
  -- Proof goes here
  sorry

end roy_consumes_tablets_in_225_minutes_l75_75743


namespace larger_number_is_23_l75_75431

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75431


namespace sum_of_fractions_l75_75149

theorem sum_of_fractions :
  (7:ℚ) / 12 + (11:ℚ) / 15 = 79 / 60 :=
by
  sorry

end sum_of_fractions_l75_75149


namespace inequality_solution_set_range_of_a_l75_75514

section
variable {x a : ℝ}

def f (x a : ℝ) := |2 * x - 5 * a| + |2 * x + 1|
def g (x : ℝ) := |x - 1| + 3

theorem inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} :=
sorry

theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 :=
sorry
end

end inequality_solution_set_range_of_a_l75_75514


namespace remainder_b_91_mod_49_l75_75222

def b (n : ℕ) := 12^n + 14^n

theorem remainder_b_91_mod_49 : (b 91) % 49 = 38 := by
  sorry

end remainder_b_91_mod_49_l75_75222


namespace function_unique_l75_75641

open Function

-- Define the domain and codomain
def NatPos : Type := {n : ℕ // n > 0}

-- Define the function f from positive integers to positive integers
noncomputable def f : NatPos → NatPos := sorry

-- Provide the main theorem
theorem function_unique (f : NatPos → NatPos) :
  (∀ (m n : NatPos), (m.val ^ 2 + (f n).val) ∣ ((m.val * (f m).val) + n.val)) →
  (∀ n : NatPos, f n = n) :=
by
  sorry

end function_unique_l75_75641


namespace measure_AB_l75_75364

noncomputable def segment_measure (a b : ℝ) : ℝ :=
  a + (2 / 3) * b

theorem measure_AB (a b : ℝ) (parallel_AB_CD : true) (angle_B_three_times_angle_D : true) (measure_AD_eq_a : true) (measure_CD_eq_b : true) :
  segment_measure a b = a + (2 / 3) * b :=
by
  sorry

end measure_AB_l75_75364


namespace block_of_flats_l75_75989

theorem block_of_flats :
  let total_floors := 12
  let half_floors := total_floors / 2
  let apartments_per_half_floor := 6
  let max_residents_per_apartment := 4
  let total_max_residents := 264
  let apartments_on_half_floors := half_floors * apartments_per_half_floor
  ∃ (x : ℝ), 
    4 * (apartments_on_half_floors + half_floors * x) = total_max_residents ->
    x = 5 :=
sorry

end block_of_flats_l75_75989


namespace probability_at_least_75_cents_l75_75277

def total_coins : ℕ := 3 + 5 + 4 + 3 -- total number of coins

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 3

def successful_outcomes_case1 : ℕ := (Nat.choose 3 3) * (Nat.choose 12 3)
def successful_outcomes_case2 : ℕ := (Nat.choose 3 2) * (Nat.choose 4 2) * (Nat.choose 5 2)

def total_outcomes : ℕ := Nat.choose 15 6
def successful_outcomes : ℕ := successful_outcomes_case1 + successful_outcomes_case2

def probability : ℚ := successful_outcomes / total_outcomes

theorem probability_at_least_75_cents :
  probability = 400 / 5005 := by
  sorry

end probability_at_least_75_cents_l75_75277


namespace shanmukham_total_payment_l75_75561

noncomputable def total_price_shanmukham_pays : Real :=
  let itemA_price : Real := 6650
  let itemA_rebate : Real := 6 -- percentage
  let itemA_tax : Real := 10 -- percentage

  let itemB_price : Real := 8350
  let itemB_rebate : Real := 4 -- percentage
  let itemB_tax : Real := 12 -- percentage

  let itemC_price : Real := 9450
  let itemC_rebate : Real := 8 -- percentage
  let itemC_tax : Real := 15 -- percentage

  let final_price (price : Real) (rebate : Real) (tax : Real) : Real :=
    let rebate_amt := (rebate / 100) * price
    let price_after_rebate := price - rebate_amt
    let tax_amt := (tax / 100) * price_after_rebate
    price_after_rebate + tax_amt

  final_price itemA_price itemA_rebate itemA_tax +
  final_price itemB_price itemB_rebate itemB_tax +
  final_price itemC_price itemC_rebate itemC_tax

theorem shanmukham_total_payment :
  total_price_shanmukham_pays = 25852.12 := by
  sorry

end shanmukham_total_payment_l75_75561


namespace flagpole_height_l75_75804

/-
A flagpole is of certain height. It breaks, folding over in half, such that what was the tip of the flagpole is now dangling two feet above the ground. 
The flagpole broke 7 feet from the base. Prove that the height of the flagpole is 16 feet.
-/

theorem flagpole_height (H : ℝ) (h1 : H > 0) (h2 : H - 7 > 0) (h3 : H - 9 = 7) : H = 16 :=
by
  /- the proof is omitted -/
  sorry

end flagpole_height_l75_75804


namespace star_eq_122_l75_75192

noncomputable def solveForStar (star : ℕ) : Prop :=
  45 - (28 - (37 - (15 - star))) = 56

theorem star_eq_122 : solveForStar 122 :=
by
  -- proof
  sorry

end star_eq_122_l75_75192


namespace num_valid_Ns_less_2000_l75_75524

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l75_75524


namespace range_of_m_l75_75348

def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

theorem range_of_m (m : ℝ) (h : (A m) ∩ B ≠ ∅) : m ≤ -1 :=
sorry

end range_of_m_l75_75348


namespace housewife_left_money_l75_75805

def initial_amount : ℝ := 150
def spent_fraction : ℝ := 2 / 3
def remaining_fraction : ℝ := 1 - spent_fraction

theorem housewife_left_money :
  initial_amount * remaining_fraction = 50 := by
  sorry

end housewife_left_money_l75_75805


namespace find_a7_l75_75668

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75668


namespace max_students_can_be_equally_distributed_l75_75611

def num_pens : ℕ := 2730
def num_pencils : ℕ := 1890

theorem max_students_can_be_equally_distributed : Nat.gcd num_pens num_pencils = 210 := by
  sorry

end max_students_can_be_equally_distributed_l75_75611


namespace find_a7_l75_75681

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75681


namespace man_speed_against_current_proof_l75_75131

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l75_75131


namespace strange_die_expected_winnings_l75_75139

noncomputable def probabilities : List ℚ := [1/4, 1/4, 1/6, 1/6, 1/6, 1/12]
noncomputable def winnings : List ℚ := [2, 2, 4, 4, -6, -12]

def expected_value (p : List ℚ) (w : List ℚ) : ℚ :=
  List.sum (List.zipWith (λ pi wi => pi * wi) p w)

theorem strange_die_expected_winnings :
  expected_value probabilities winnings = 0.17 :=
by
  sorry

end strange_die_expected_winnings_l75_75139


namespace amplitude_of_cosine_wave_l75_75940

theorem amplitude_of_cosine_wave 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_max_min : ∀ x : ℝ, d + a = 5 ∧ d - a = 1) 
  : a = 2 :=
by
  sorry

end amplitude_of_cosine_wave_l75_75940


namespace sqrt_square_l75_75941

theorem sqrt_square (n : ℝ) : (Real.sqrt 2023) ^ 2 = 2023 :=
by
  sorry

end sqrt_square_l75_75941


namespace unique_rs_exists_l75_75883

theorem unique_rs_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (gcd_ab : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), (0 < r ∧ r < b) ∧ (0 < s ∧ s < a) ∧ (a * r - b * s = 1) :=
  sorry

end unique_rs_exists_l75_75883


namespace inverse_proportion_l75_75766

theorem inverse_proportion (a : ℝ) (b : ℝ) (k : ℝ) : 
  (a = k / b^2) → 
  (40 = k / 12^2) → 
  (a = 10) → 
  b = 24 := 
by
  sorry

end inverse_proportion_l75_75766


namespace mans_speed_against_current_l75_75134

variable (V_downstream V_current : ℝ)
variable (V_downstream_eq : V_downstream = 15)
variable (V_current_eq : V_current = 2.5)

theorem mans_speed_against_current : V_downstream - 2 * V_current = 10 :=
by
  rw [V_downstream_eq, V_current_eq]
  exact (15 - 2 * 2.5)

end mans_speed_against_current_l75_75134


namespace find_least_positive_x_l75_75029

theorem find_least_positive_x :
  ∃ x : ℕ, x + 5419 ≡ 3789 [MOD 15] ∧ x = 5 :=
by
  use 5
  constructor
  · sorry
  · rfl

end find_least_positive_x_l75_75029


namespace coloring_four_cells_with_diff_colors_l75_75038

theorem coloring_four_cells_with_diff_colors {n k : ℕ} (h : n ≥ 2) 
    (hk : k = 2 * n) 
    (color : fin n × fin n → fin k) 
    (hcolor : ∀ c, ∃ r c : fin k, ∃ a b : fin k, color (r, c) = a ∧ color (r, c) = b) :
    ∃ r1 r2 c1 c2, color (r1, c1) ≠ color (r1, c2) ∧ color (r1, c1) ≠ color (r2, c1) ∧
                    color (r2, c1) ≠ color (r2, c2) ∧ color (r1, c2) ≠ color (r2, c2) :=
by
  sorry

end coloring_four_cells_with_diff_colors_l75_75038


namespace no_solution_iff_discriminant_l75_75197

theorem no_solution_iff_discriminant (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) ↔ -2 ≤ k ∧ k ≤ 2 := by
  sorry

end no_solution_iff_discriminant_l75_75197


namespace estevan_initial_blankets_l75_75018

theorem estevan_initial_blankets (B : ℕ) 
  (polka_dot_initial : ℕ) 
  (polka_dot_total : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = polka_dot_initial) 
  (h2 : polka_dot_initial + 2 = polka_dot_total) 
  (h3 : polka_dot_total = 10) : 
  B = 24 := 
by 
  sorry

end estevan_initial_blankets_l75_75018


namespace quadratic_equation_standard_form_quadratic_equation_coefficients_l75_75755

theorem quadratic_equation_standard_form : 
  ∀ (x : ℝ), (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (a = 2 ∧ b = -6 ∧ c = -1) :=
by
  sorry

end quadratic_equation_standard_form_quadratic_equation_coefficients_l75_75755


namespace find_number_l75_75137

theorem find_number 
    (x : ℝ)
    (h1 : 3 < x) 
    (h2 : x < 8) 
    (h3 : 6 < x) 
    (h4 : x < 10) : 
    x = 7 :=
sorry

end find_number_l75_75137


namespace acute_angle_at_315_equals_7_5_l75_75261

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l75_75261


namespace find_a_l75_75905

noncomputable def f (x a : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem find_a : (∃ a : ℝ, ((∀ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a ≤ -3) ∧ (∃ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a = -3)) ↔ a = Real.sqrt 6 + 2) :=
by
  sorry

end find_a_l75_75905


namespace martha_cakes_required_l75_75551

-- Conditions
def number_of_children : ℝ := 3.0
def cakes_per_child : ℝ := 18.0

-- The main statement to prove
theorem martha_cakes_required:
  (number_of_children * cakes_per_child) = 54.0 := 
by
  sorry

end martha_cakes_required_l75_75551


namespace find_a7_l75_75680

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75680


namespace eleven_power_five_mod_nine_l75_75235

theorem eleven_power_five_mod_nine : ∃ n : ℕ, (11^5 ≡ n [MOD 9]) ∧ (0 ≤ n ∧ n < 9) ∧ (n = 5) := 
  by 
    sorry

end eleven_power_five_mod_nine_l75_75235


namespace find_a7_l75_75653

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75653


namespace addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l75_75831

theorem addition_comm (a b : ℕ) : a + b = b + a :=
by sorry

theorem subtraction_compare {a b c : ℕ} (h1 : a < b) (h2 : c = 28) : 56 - c < 65 - c :=
by sorry

theorem multiplication_comm (a b : ℕ) : a * b = b * a :=
by sorry

theorem subtraction_greater {a b c : ℕ} (h1 : a - b = 18) (h2 : a - c = 27) (h3 : 32 = b) (h4 : 23 = c) : a - b > a - c :=
by sorry

end addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l75_75831


namespace cost_per_page_of_notebooks_l75_75209

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end cost_per_page_of_notebooks_l75_75209


namespace sector_area_l75_75465

noncomputable def area_of_sector (r : ℝ) (theta : ℝ) : ℝ :=
  1 / 2 * r * r * theta

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = Real.pi) (h_theta : theta = 2 * Real.pi / 3) :
  area_of_sector r theta = Real.pi^3 / 6 :=
by
  sorry

end sector_area_l75_75465


namespace disjoint_range_probability_l75_75735

def A : Finset ℕ := {1, 2, 3, 4}

theorem disjoint_range_probability :
  let funcs := Finset.pi_finset (λ _ : A, A)
  let total_pairs := finset.card funcs * finset.card funcs
  let disjoint_pairs := 1740
  let gcd := nat.gcd 435 16384
  m = 435 := by
  let prob := (disjoint_pairs, total_pairs)
  let simplest_form := (prob.1 / gcd, prob.2 / gcd)
  have : simplest_form.1 = m := rfl
  have : simplest_form.1 = 435 := sorry
  sorry

end disjoint_range_probability_l75_75735


namespace normal_dist_probability_l75_75093

noncomputable def normalDist (μ σ : ℝ) : ℝ → ℝ := sorry -- placeholder for actual normal distribution function

theorem normal_dist_probability (μ σ : ℝ) (a : ℝ)
  (hμ : μ = 4.5) (hσ : σ = 0.05) (ha : a = 0.1) :
  normalDist μ σ (|4.5 - ha|) = 0.9544 :=
sorry

end normal_dist_probability_l75_75093


namespace larger_number_is_23_l75_75417

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75417


namespace no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l75_75714

theorem no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 10000 ∧ (n % 10 = 0) ∧ (Prime n) → False :=
by
  sorry

end no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l75_75714


namespace tan_div_sin_cos_sin_mul_cos_l75_75503

theorem tan_div_sin_cos (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 :=
by
  sorry

theorem sin_mul_cos (α : ℝ) (h : Real.tan α = 7) :
  Real.sin α * Real.cos α = 7 / 50 :=
by
  sorry

end tan_div_sin_cos_sin_mul_cos_l75_75503


namespace peter_present_age_l75_75800

def age_problem (P J : ℕ) : Prop :=
  J = P + 12 ∧ P - 10 = (1 / 3 : ℚ) * (J - 10)

theorem peter_present_age : ∃ (P : ℕ), ∃ (J : ℕ), age_problem P J ∧ P = 16 :=
by {
  -- Add the proof here, which is not required
  sorry
}

end peter_present_age_l75_75800


namespace man_is_older_by_22_l75_75620

/-- 
Given the present age of the son is 20 years and in two years the man's age will be 
twice the age of his son, prove that the man is 22 years older than his son.
-/
theorem man_is_older_by_22 (S M : ℕ) (h1 : S = 20) (h2 : M + 2 = 2 * (S + 2)) : M - S = 22 :=
by
  sorry  -- Proof will be provided here

end man_is_older_by_22_l75_75620


namespace number_of_students_in_class_l75_75090

theorem number_of_students_in_class :
  ∃ n : ℕ, n > 0 ∧ (∀ avg_age teacher_age total_avg_age, avg_age = 26 ∧ teacher_age = 52 ∧ total_avg_age = 27 →
    (∃ total_student_age total_age_with_teacher, 
      total_student_age = n * avg_age ∧ 
      total_age_with_teacher = total_student_age + teacher_age ∧ 
      (total_age_with_teacher / (n + 1) = total_avg_age) → n = 25)) :=
sorry

end number_of_students_in_class_l75_75090


namespace find_m_l75_75969

open Real

namespace VectorPerpendicular

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := (v₁.1 * v₂.1 + v₁.2 * v₂.2) = 0

theorem find_m (m : ℝ) (h : perpendicular a (b m)) : m = 1 / 2 :=
by
  sorry -- Proof is omitted

end VectorPerpendicular

end find_m_l75_75969


namespace gardener_total_expenses_l75_75541

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l75_75541


namespace cost_of_book_sold_at_loss_l75_75794

theorem cost_of_book_sold_at_loss:
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = 490 ∧ 
    C1 * 0.85 = C2 * 1.19 ∧ 
    C1 = 285.93 :=
by
  sorry

end cost_of_book_sold_at_loss_l75_75794


namespace toby_steps_l75_75101

theorem toby_steps (sunday tuesday wednesday thursday friday_saturday monday : ℕ) :
    sunday = 9400 →
    tuesday = 8300 →
    wednesday = 9200 →
    thursday = 8900 →
    friday_saturday = 9050 →
    7 * 9000 = 63000 →
    monday = 63000 - (sunday + tuesday + wednesday + thursday + 2 * friday_saturday) → monday = 9100 :=
by
  intros hs ht hw hth hfs htc hnm
  sorry

end toby_steps_l75_75101


namespace find_nm_l75_75701

theorem find_nm (h : 62^2 + 122^2 = 18728) : 
  ∃ (n m : ℕ), (n = 92 ∧ m = 30) ∨ (n = 30 ∧ m = 92) ∧ n^2 + m^2 = 9364 := 
by 
  sorry

end find_nm_l75_75701


namespace value_of_expression_at_x_eq_2_l75_75263

theorem value_of_expression_at_x_eq_2 :
  (2 * (2: ℕ)^2 - 3 * 2 + 4 = 6) := 
by sorry

end value_of_expression_at_x_eq_2_l75_75263


namespace carly_cooks_in_72_minutes_l75_75011

def total_time_to_cook_burgers (total_guests : ℕ) (cook_time_per_side : ℕ) (burgers_per_grill : ℕ) : ℕ :=
  let guests_who_want_two_burgers := total_guests / 2
  let guests_who_want_one_burger := total_guests - guests_who_want_two_burgers
  let total_burgers := (guests_who_want_two_burgers * 2) + guests_who_want_one_burger
  let total_batches := (total_burgers + burgers_per_grill - 1) / burgers_per_grill  -- ceil division for total batches
  total_batches * (2 * cook_time_per_side)  -- total time

theorem carly_cooks_in_72_minutes : 
  total_time_to_cook_burgers 30 4 5 = 72 :=
by 
  sorry

end carly_cooks_in_72_minutes_l75_75011


namespace miles_to_burger_restaurant_l75_75216

-- Definitions and conditions
def miles_per_gallon : ℕ := 19
def gallons_of_gas : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_friend_house : ℕ := 4
def miles_to_home : ℕ := 11
def total_gas_distance := miles_per_gallon * gallons_of_gas
def total_known_distances := miles_to_school + miles_to_softball_park + miles_to_friend_house + miles_to_home

-- Problem statement to prove
theorem miles_to_burger_restaurant :
  ∃ (miles_to_burger_restaurant : ℕ), 
  total_gas_distance = total_known_distances + miles_to_burger_restaurant ∧ miles_to_burger_restaurant = 2 := 
by
  sorry

end miles_to_burger_restaurant_l75_75216


namespace total_treats_value_l75_75300

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l75_75300


namespace smallest_integer_value_of_x_l75_75917

theorem smallest_integer_value_of_x (x : ℤ) (h : 7 + 3 * x < 26) : x = 6 :=
sorry

end smallest_integer_value_of_x_l75_75917


namespace trajectory_of_Q_l75_75844

/-- Let P(m, n) be a point moving on the circle x^2 + y^2 = 2.
     The trajectory of the point Q(m+n, 2mn) is y = x^2 - 2. -/
theorem trajectory_of_Q (m n : ℝ) (hyp : m^2 + n^2 = 2) : 
  ∃ x y : ℝ, x = m + n ∧ y = 2 * m * n ∧ y = x^2 - 2 :=
by
  sorry

end trajectory_of_Q_l75_75844


namespace base_of_isosceles_triangle_l75_75408

namespace TriangleProblem

def equilateral_triangle_perimeter (s : ℕ) : ℕ := 3 * s
def isosceles_triangle_perimeter (s b : ℕ) : ℕ := 2 * s + b

theorem base_of_isosceles_triangle (s b : ℕ) (h1 : equilateral_triangle_perimeter s = 45) 
    (h2 : isosceles_triangle_perimeter s b = 40) : b = 10 :=
by
  sorry

end TriangleProblem

end base_of_isosceles_triangle_l75_75408


namespace ratio_is_one_half_l75_75291

noncomputable def ratio_of_females_to_males (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ) : ℚ :=
  (f : ℚ) / (m : ℚ)

theorem ratio_is_one_half (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ)
  (h_female_age : avg_female_age = 45)
  (h_male_age : avg_male_age = 30)
  (h_total_age : avg_total_age = 35)
  (h_total_avg : (45 * f + 30 * m) / (f + m) = 35) :
  ratio_of_females_to_males f m avg_female_age avg_male_age avg_total_age = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l75_75291


namespace problem_probability_l75_75006

theorem problem_probability :
  let p_arthur := (1 : ℚ) / 4
  let p_bella := (3 : ℚ) / 10
  let p_xavier := (1 : ℚ) / 6
  let p_yvonne := (1 : ℚ) / 2
  let p_zelda := (5 : ℚ) / 8
  let p_zelda_failure := 1 - p_zelda
  let result := p_arthur * p_bella * p_xavier * p_yvonne * p_zelda_failure
  result = 9 / 3840 := by
  sorry

end problem_probability_l75_75006


namespace original_price_of_computer_l75_75864

theorem original_price_of_computer
  (P : ℝ)
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 :=
by
  sorry

end original_price_of_computer_l75_75864


namespace coeff_of_x_in_expansion_l75_75874

theorem coeff_of_x_in_expansion :
  let f := (1 : ℚ) + (Polynomial.X : Polynomial ℚ)
  let g := (Polynomial.X : Polynomial ℚ) - Polynomial.C (2 / Polynomial.X)
  Polynomial.coeff ((f * g ^ 3).expand ℚ) 1 = -6 := sorry

end coeff_of_x_in_expansion_l75_75874


namespace parallel_lines_l75_75907

theorem parallel_lines (a : ℝ) : (2 * a = a * (a + 4)) → a = -2 :=
by
  intro h
  sorry

end parallel_lines_l75_75907


namespace fraction_geq_81_l75_75746

theorem fraction_geq_81 {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 :=
by
  sorry

end fraction_geq_81_l75_75746


namespace product_of_solutions_of_abs_eq_l75_75726

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x - 5| - 4 = 3) : x * (if x = 12 then -2 else if x = -2 then 12 else 1) = -24 :=
by
  sorry

end product_of_solutions_of_abs_eq_l75_75726


namespace john_works_30_hours_per_week_l75_75199

/-- Conditions --/
def hours_per_week_fiona : ℕ := 40
def hours_per_week_jeremy : ℕ := 25
def hourly_wage : ℕ := 20
def monthly_total_payment : ℕ := 7600
def weeks_in_month : ℕ := 4

/-- Derived Definitions --/
def monthly_hours_fiona_jeremy : ℕ :=
  (hours_per_week_fiona + hours_per_week_jeremy) * weeks_in_month

def monthly_payment_fiona_jeremy : ℕ :=
  hourly_wage * monthly_hours_fiona_jeremy

def monthly_payment_john : ℕ :=
  monthly_total_payment - monthly_payment_fiona_jeremy

def hours_per_month_john : ℕ :=
  monthly_payment_john / hourly_wage

def hours_per_week_john : ℕ :=
  hours_per_month_john / weeks_in_month

/-- Theorem stating that John works 30 hours per week --/
theorem john_works_30_hours_per_week :
  hours_per_week_john = 30 := by
  sorry

end john_works_30_hours_per_week_l75_75199


namespace members_play_both_l75_75918

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end members_play_both_l75_75918


namespace calc_expression_l75_75635

theorem calc_expression :
  (- (2 / 5) : ℝ)^0 - (0.064 : ℝ)^(1/3) + 3^(Real.log (2 / 5) / Real.log 3) + Real.log 2 / Real.log 10 - Real.log (1 / 5) / Real.log 10 = 2 := 
by
  sorry

end calc_expression_l75_75635


namespace megan_works_per_day_hours_l75_75079

theorem megan_works_per_day_hours
  (h : ℝ)
  (earnings_per_hour : ℝ)
  (days_per_month : ℝ)
  (total_earnings_two_months : ℝ) :
  earnings_per_hour = 7.50 →
  days_per_month = 20 →
  total_earnings_two_months = 2400 →
  2 * days_per_month * earnings_per_hour * h = total_earnings_two_months →
  h = 8 :=
by {
  sorry
}

end megan_works_per_day_hours_l75_75079


namespace geometric_seq_a7_l75_75660

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75660


namespace geometric_seq_a7_l75_75663

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75663


namespace inverse_of_original_l75_75401

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l75_75401


namespace larger_number_is_23_l75_75412

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75412


namespace smallest_n_l75_75030

def power_tower (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => a
  | (n+1) => a ^ (power_tower a n)

def pow3_cubed : ℕ := 3 ^ (3 ^ (3 ^ 3))

theorem smallest_n : ∃ n, (∃ k : ℕ, (power_tower 2 n) = k ∧ k > pow3_cubed) ∧ ∀ m, (∃ k : ℕ, (power_tower 2 m) = k ∧ k > pow3_cubed) → m ≥ n :=
  by
  sorry

end smallest_n_l75_75030


namespace sum_of_x_y_l75_75191

theorem sum_of_x_y (m x y : ℝ) (h₁ : x + m = 4) (h₂ : y - 3 = m) : x + y = 7 :=
sorry

end sum_of_x_y_l75_75191


namespace eval_expression_l75_75020

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l75_75020


namespace prove_inequality_l75_75764

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l75_75764


namespace find_a7_l75_75679

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75679


namespace find_phi_l75_75852

open Real

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := cos (2 * x - π/2 + φ)

theorem find_phi 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (symmetry_condition : ∀ x, g (π/2 - x) φ = g (π/2 + x) φ) 
  : φ = π / 2 
:= by 
  sorry

end find_phi_l75_75852


namespace inequality_l75_75885

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 3) :
  1 / (4 - a^2) + 1 / (4 - b^2) + 1 / (4 - c^2) ≤ 9 / (a + b + c)^2 :=
by
  sorry

end inequality_l75_75885


namespace coloring_four_cells_with_diff_colors_l75_75039

theorem coloring_four_cells_with_diff_colors {n k : ℕ} (h : n ≥ 2) 
    (hk : k = 2 * n) 
    (color : fin n × fin n → fin k) 
    (hcolor : ∀ c, ∃ r c : fin k, ∃ a b : fin k, color (r, c) = a ∧ color (r, c) = b) :
    ∃ r1 r2 c1 c2, color (r1, c1) ≠ color (r1, c2) ∧ color (r1, c1) ≠ color (r2, c1) ∧
                    color (r2, c1) ≠ color (r2, c2) ∧ color (r1, c2) ≠ color (r2, c2) :=
by
  sorry

end coloring_four_cells_with_diff_colors_l75_75039


namespace relationship_l75_75172

-- Given conditions
def a : ℝ := 31 / 32
def b : ℝ := Real.cos (1 / 4)
def c : ℝ := 4 * Real.sin (1 / 4)

-- The theorem to be proven
theorem relationship : c > b ∧ b > a := by
  sorry

end relationship_l75_75172


namespace number_of_teams_l75_75358

theorem number_of_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end number_of_teams_l75_75358


namespace cube_identity_l75_75050

theorem cube_identity (a : ℝ) (h : (a + 1/a) ^ 2 = 3) : a^3 + 1/a^3 = 0 := 
by
  sorry

end cube_identity_l75_75050


namespace probability_x_y_le_5_l75_75932

noncomputable section

open MeasureTheory

namespace probability

def region (x y : ℝ) : Prop := (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 4)

noncomputable def probability_of_event : ℝ :=
  (volume {p : ℝ × ℝ | region p.1 p.2 ∧ p.1 + p.2 ≤ 5}) / (volume {p : ℝ × ℝ | region p.1 p.2})

theorem probability_x_y_le_5 : probability_of_event = 17 / 32 := 
 by 
  sorry

end probability

end probability_x_y_le_5_l75_75932


namespace find_a7_l75_75673

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75673


namespace probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l75_75811

namespace ProbabilityKeys

-- Define the problem conditions and the probability computations
def keys : ℕ := 4
def successful_keys : ℕ := 2
def unsuccessful_keys : ℕ := 2

def probability_first_fail (k : ℕ) (s : ℕ) : ℚ := (s : ℚ) / (k : ℚ)
def probability_second_success_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (s + 1 - 1: ℚ) 
def probability_second_success_not_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (k : ℚ)

-- The statements to be proved
theorem probability_door_opened_second_attempt_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_discarded unsuccessful_keys keys) = (1 : ℚ) / (3 : ℚ) :=
by sorry

theorem probability_door_opened_second_attempt_not_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_not_discarded successful_keys keys) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end ProbabilityKeys

end probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l75_75811


namespace min_distance_exists_l75_75175

open Real

-- Define the distance formula function
noncomputable def distance (x : ℝ) : ℝ :=
sqrt ((x - 1) ^ 2 + (3 - 2 * x) ^ 2 + (3 * x - 3) ^ 2)

theorem min_distance_exists :
  ∃ (x : ℝ), distance x = sqrt (14 * x^2 - 32 * x + 19) ∧
               ∀ y, distance y ≥ (sqrt 35) / 7 :=
sorry

end min_distance_exists_l75_75175


namespace circle_center_l75_75164

theorem circle_center :
  ∃ c : ℝ × ℝ, c = (-1, 3) ∧ ∀ (x y : ℝ), (4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 96 = 0 ↔ (x + 1)^2 + (y - 3)^2 = 14) :=
by
  sorry

end circle_center_l75_75164


namespace inequality_system_solution_exists_l75_75531

theorem inequality_system_solution_exists (a : ℝ) : (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := 
sorry

end inequality_system_solution_exists_l75_75531


namespace sufficient_but_not_necessary_condition_l75_75700

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h : |b| + a < 0) : b^2 < a^2 :=
  sorry

end sufficient_but_not_necessary_condition_l75_75700


namespace geometric_seq_a7_l75_75662

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75662


namespace days_in_month_find_days_in_month_l75_75618

noncomputable def computers_per_thirty_minutes : ℕ := 225 / 100 -- representing 2.25
def monthly_computers : ℕ := 3024
def hours_per_day : ℕ := 24

theorem days_in_month (computers_per_hour : ℕ) (daily_production : ℕ) : ℕ :=
  let computers_per_hour := (2 * computers_per_thirty_minutes)
  let daily_production := (computers_per_hour * hours_per_day)
  (monthly_computers / daily_production)

theorem find_days_in_month :
  days_in_month (2 * computers_per_thirty_minutes) ((2 * computers_per_thirty_minutes) * hours_per_day) = 28 :=
by
  sorry

end days_in_month_find_days_in_month_l75_75618


namespace geometric_seq_a7_l75_75659

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l75_75659


namespace sum_of_ammeter_readings_l75_75031

def I1 := 4 
def I2 := 4
def I3 := 2 * I2
def I5 := I3 + I2
def I4 := (5 / 3) * I5

theorem sum_of_ammeter_readings : I1 + I2 + I3 + I4 + I5 = 48 := by
  sorry

end sum_of_ammeter_readings_l75_75031


namespace mans_speed_against_current_l75_75133

variable (V_downstream V_current : ℝ)
variable (V_downstream_eq : V_downstream = 15)
variable (V_current_eq : V_current = 2.5)

theorem mans_speed_against_current : V_downstream - 2 * V_current = 10 :=
by
  rw [V_downstream_eq, V_current_eq]
  exact (15 - 2 * 2.5)

end mans_speed_against_current_l75_75133


namespace sum_of_interior_angles_of_hexagon_l75_75579

theorem sum_of_interior_angles_of_hexagon : 
  let n := 6 in (n - 2) * 180 = 720 := 
by
  let n := 6
  show (n - 2) * 180 = 720
  sorry

end sum_of_interior_angles_of_hexagon_l75_75579


namespace cricket_innings_l75_75278

theorem cricket_innings (n : ℕ) 
  (avg_run_inn : n * 36 = n * 36)  -- average runs is 36 (initially true for any n)
  (increase_avg_by_4 : (36 * n + 120) / (n + 1) = 40) : 
  n = 20 := 
sorry

end cricket_innings_l75_75278


namespace solve_for_x_l75_75536

theorem solve_for_x (x : ℤ) (h : 3 * x + 7 = -2) : x = -3 :=
by
  sorry

end solve_for_x_l75_75536


namespace parallel_lines_m_values_l75_75532

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ mx + 3 * y - 2 = 0) → (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_values_l75_75532


namespace future_years_l75_75130

theorem future_years (P A F : ℝ) (Y : ℝ) 
  (h1 : P = 50)
  (h2 : P = 1.25 * A)
  (h3 : P = 5 / 6 * F)
  (h4 : A + 10 + Y = F) : 
  Y = 10 := sorry

end future_years_l75_75130


namespace evaluate_expression_l75_75487

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l75_75487


namespace smallest_multiple_l75_75781

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l75_75781


namespace candy_probability_l75_75913

/-
Question: 
Given three bags with the following compositions:
  - Bag 1: Three green candies, one red candy.
  - Bag 2: Two green candies, two red candies.
  - Bag 3: One green candy, three red candies.
A child randomly selects one of the bags and then randomly picks a first candy:
  - If the first candy is green, the second candy is chosen from one of the other two bags.
  - If the first candy is red, the second candy is chosen from the same bag.
Prove that the probability that the second candy is green can be expressed as the fraction \( \frac{73}{144} \), and thus \( m + n = 217 \).
-/

theorem candy_probability :
  let m := 73
  let n := 144
  m + n = 217 :=
by {
  sorry -- proof is omitted
}

end candy_probability_l75_75913


namespace nuts_needed_for_cookies_l75_75378

-- Given conditions
def total_cookies : Nat := 120
def fraction_nuts : Rat := 1 / 3
def fraction_chocolate : Rat := 0.25
def nuts_per_cookie : Nat := 3

-- Translated conditions as helpful functions
def cookies_with_nuts : Nat := Nat.floor (fraction_nuts * total_cookies)
def cookies_with_chocolate : Nat := Nat.floor (fraction_chocolate * total_cookies)
def cookies_with_both : Nat := total_cookies - cookies_with_nuts - cookies_with_chocolate
def total_cookies_with_nuts : Nat := cookies_with_nuts + cookies_with_both
def total_nuts_needed : Nat := total_cookies_with_nuts * nuts_per_cookie

-- Proof problem: proving that total nuts needed is 270
theorem nuts_needed_for_cookies : total_nuts_needed = 270 :=
by
  sorry

end nuts_needed_for_cookies_l75_75378


namespace total_pieces_of_gum_l75_75086

def packages : ℕ := 12
def pieces_per_package : ℕ := 20

theorem total_pieces_of_gum : packages * pieces_per_package = 240 :=
by
  -- proof is skipped
  sorry

end total_pieces_of_gum_l75_75086


namespace pancake_cut_l75_75098

theorem pancake_cut (n : ℕ) (h : 3 ≤ n) :
  ∃ (cut_piece : ℝ), cut_piece > 0 :=
sorry

end pancake_cut_l75_75098


namespace turtles_remaining_on_log_l75_75120

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end turtles_remaining_on_log_l75_75120


namespace hyperbola_focus_exists_l75_75824

-- Define the basic premises of the problem
def is_hyperbola (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

-- Define a condition for the focusing property of the hyperbola.
def is_focus (x y : ℝ) : Prop :=
  (x = -2) ∧ (y = 4 + (10 * Real.sqrt 3 / 3))

-- The theorem to be proved
theorem hyperbola_focus_exists : ∃ x y : ℝ, is_hyperbola x y ∧ is_focus x y :=
by
  -- Proof to be filled in
  sorry

end hyperbola_focus_exists_l75_75824


namespace number_of_ways_to_choose_teams_l75_75499

theorem number_of_ways_to_choose_teams : 
  ∃ (n : ℕ), n = Nat.choose 5 2 ∧ n = 10 :=
by
  have h : Nat.choose 5 2 = 10 := by sorry
  use 10
  exact ⟨h, rfl⟩

end number_of_ways_to_choose_teams_l75_75499


namespace pie_cost_correct_l75_75737

-- Define the initial and final amounts of money Mary had.
def initial_amount : ℕ := 58
def final_amount : ℕ := 52

-- Define the cost of the pie as the difference between initial and final amounts.
def pie_cost : ℕ := initial_amount - final_amount

-- State the theorem that given the initial and final amounts, the cost of the pie is 6.
theorem pie_cost_correct : pie_cost = 6 := by 
  sorry

end pie_cost_correct_l75_75737


namespace larger_number_is_23_l75_75415

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75415


namespace square_area_and_diagonal_ratio_l75_75758

theorem square_area_and_diagonal_ratio
    (a b : ℕ)
    (h_perimeter : 4 * a = 16 * b) :
    (a = 4 * b) ∧ ((a^2) / (b^2) = 16) ∧ ((a * Real.sqrt 2) / (b * Real.sqrt 2) = 4) :=
  by
  sorry

end square_area_and_diagonal_ratio_l75_75758


namespace pond_volume_l75_75271

theorem pond_volume (L W H : ℝ) (hL : L = 20) (hW : W = 10) (hH : H = 5) : 
  L * W * H = 1000 :=
by
  rw [hL, hW, hH]
  norm_num

end pond_volume_l75_75271


namespace angle_between_vectors_l75_75970

def vector (α : Type) [Field α] := (α × α)

theorem angle_between_vectors
    (a : vector ℝ)
    (b : vector ℝ)
    (ha : a = (4, 0))
    (hb : b = (-1, Real.sqrt 3)) :
  let dot_product (v w : vector ℝ) : ℝ := (v.1 * w.1 + v.2 * w.2)
  let norm (v : vector ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  let cos_theta := dot_product a b / (norm a * norm b)
  ∀ theta, Real.cos theta = cos_theta → theta = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l75_75970


namespace tied_part_length_l75_75771

theorem tied_part_length (length_of_each_string : ℕ) (num_strings : ℕ) (total_tied_length : ℕ) 
  (H1 : length_of_each_string = 217) (H2 : num_strings = 3) (H3 : total_tied_length = 627) : 
  (length_of_each_string * num_strings - total_tied_length) / (num_strings - 1) = 12 :=
by
  sorry

end tied_part_length_l75_75771


namespace prob_point_closer_to_six_than_zero_l75_75463

theorem prob_point_closer_to_six_than_zero : 
  let interval_start := 0
  let interval_end := 7
  let closer_to_six := fun x => x > ((interval_start + 6) / 2)
  let total_length := interval_end - interval_start
  let length_closer_to_six := interval_end - (interval_start + 6) / 2
  total_length > 0 -> length_closer_to_six / total_length = 4 / 7 :=
by
  sorry

end prob_point_closer_to_six_than_zero_l75_75463


namespace instantaneous_velocity_at_3_l75_75094

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The main statement we need to prove
theorem instantaneous_velocity_at_3 : (deriv s 3) = 5 :=
by 
  -- The theorem requires a proof which we mark as sorry for now.
  sorry

end instantaneous_velocity_at_3_l75_75094


namespace larger_number_is_23_l75_75420

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l75_75420


namespace time_to_see_slow_train_l75_75103

noncomputable def time_to_pass (length_fast_train length_slow_train relative_time_fast seconds_observed_by_slow : ℕ) : ℕ := 
  length_slow_train * seconds_observed_by_slow / length_fast_train

theorem time_to_see_slow_train :
  let length_fast_train := 150
  let length_slow_train := 200
  let seconds_observed_by_slow := 6
  let expected_time := 8
  time_to_pass length_fast_train length_slow_train length_fast_train seconds_observed_by_slow = expected_time :=
by sorry

end time_to_see_slow_train_l75_75103


namespace find_a7_l75_75682

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75682


namespace platform_length_150_l75_75627

def speed_kmph : ℕ := 54  -- Speed in km/hr

def speed_mps : ℚ := speed_kmph * 1000 / 3600  -- Speed in m/s

def time_pass_man : ℕ := 20  -- Time to pass a man in seconds
def time_pass_platform : ℕ := 30  -- Time to pass a platform in seconds

def length_train : ℚ := speed_mps * time_pass_man  -- Length of the train in meters

def length_platform (P : ℚ) : Prop :=
  length_train + P = speed_mps * time_pass_platform  -- The condition involving platform length

theorem platform_length_150 :
  length_platform 150 := by
  -- We would provide a proof here.
  sorry

end platform_length_150_l75_75627


namespace gcd_63_84_l75_75495

theorem gcd_63_84 : Nat.gcd 63 84 = 21 := by
  -- The proof will go here.
  sorry

end gcd_63_84_l75_75495


namespace transistors_2004_l75_75628

-- Definition of Moore's law specifying the initial amount and the doubling period
def moores_law (initial : ℕ) (years : ℕ) (doubling_period : ℕ) : ℕ :=
  initial * 2 ^ (years / doubling_period)

-- Condition: The number of transistors in 1992
def initial_1992 : ℕ := 2000000

-- Condition: The number of years between 1992 and 2004
def years_between : ℕ := 2004 - 1992

-- Condition: Doubling period every 2 years
def doubling_period : ℕ := 2

-- Goal: Prove the number of transistors in 2004 using the conditions above
theorem transistors_2004 : moores_law initial_1992 years_between doubling_period = 128000000 :=
by
  sorry

end transistors_2004_l75_75628


namespace geometric_sequence_a3_l75_75958

theorem geometric_sequence_a3 (a : ℕ → ℝ)
  (h : ∀ n m : ℕ, a (n + m) = a n * a m)
  (pos : ∀ n, 0 < a n)
  (a1 : a 1 = 1)
  (a5 : a 5 = 9) :
  a 3 = 3 := by
  sorry

end geometric_sequence_a3_l75_75958


namespace dodecagon_diagonals_l75_75638

def D (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals : D 12 = 54 :=
by
  sorry

end dodecagon_diagonals_l75_75638


namespace cost_per_meter_l75_75405

-- Defining the parameters and their relationships
def length : ℝ := 58
def breadth : ℝ := length - 16
def total_cost : ℝ := 5300
def perimeter : ℝ := 2 * (length + breadth)

-- Proving the cost per meter of fencing
theorem cost_per_meter : total_cost / perimeter = 26.50 := 
by
  sorry

end cost_per_meter_l75_75405


namespace find_five_digit_number_l75_75928

theorem find_five_digit_number (a b c d e : ℕ) 
  (h : [ (10 * a + a), (10 * a + b), (10 * a + b), (10 * a + b), (10 * a + c), 
         (10 * b + c), (10 * b + b), (10 * b + c), (10 * c + b), (10 * c + b)] = 
         [33, 37, 37, 37, 38, 73, 77, 78, 83, 87]) :
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 37837 :=
sorry

end find_five_digit_number_l75_75928


namespace tallest_building_model_height_l75_75308

def height_campus : ℝ := 120
def volume_campus : ℝ := 30000
def volume_model : ℝ := 0.03
def height_model : ℝ := 1.2

theorem tallest_building_model_height :
  (volume_campus / volume_model)^(1/3) = (height_campus / height_model) :=
by
  sorry

end tallest_building_model_height_l75_75308


namespace false_disjunction_implies_both_false_l75_75533

theorem false_disjunction_implies_both_false (p q : Prop) (h : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
sorry

end false_disjunction_implies_both_false_l75_75533


namespace find_a7_l75_75687

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l75_75687


namespace specific_five_card_order_probability_l75_75498

open Classical

noncomputable def prob_five_cards_specified_order : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49) * (9 / 48)

theorem specific_five_card_order_probability :
  prob_five_cards_specified_order = 2304 / 31187500 :=
by
  sorry

end specific_five_card_order_probability_l75_75498


namespace smallest_five_digit_multiple_of_9_starting_with_7_l75_75107

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∃ (n : ℕ), (70000 ≤ n ∧ n < 80000) ∧ (n % 9 = 0) ∧ n = 70002 :=
sorry

end smallest_five_digit_multiple_of_9_starting_with_7_l75_75107


namespace find_a7_l75_75667

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l75_75667


namespace mans_speed_upstream_l75_75136

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l75_75136


namespace total_flowers_eaten_l75_75894

-- Definitions based on conditions
def num_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Statement asserting the total number of flowers eaten
theorem total_flowers_eaten : num_bugs * flowers_per_bug = 6 := by
  sorry

end total_flowers_eaten_l75_75894


namespace triangle_tangent_limit_l75_75207

theorem triangle_tangent_limit
  (A B C D : Point)
  (BC_len : BC.length = 6)
  (angle_C : ∠ A B C = Real.pi / 4)
  (midpoint_D : Midpoint D B C) :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = (x - 3 * Real.sqrt 2) / (x + 3 * Real.sqrt 2)) ∧ 
  Tendsto f atTop (𝓝 1) :=
by
  sorry

end triangle_tangent_limit_l75_75207


namespace child_ticket_cost_l75_75814

-- Define the conditions
def adult_ticket_cost : ℕ := 11
def total_people : ℕ := 23
def total_revenue : ℕ := 246
def children_count : ℕ := 7
def adults_count := total_people - children_count

-- Define the target to prove that the child ticket cost is 10
theorem child_ticket_cost (child_ticket_cost : ℕ) :
  16 * adult_ticket_cost + 7 * child_ticket_cost = total_revenue → 
  child_ticket_cost = 10 := by
  -- The proof is omitted
  sorry

end child_ticket_cost_l75_75814


namespace find_a7_l75_75671

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l75_75671


namespace train_length_is_300_l75_75936

theorem train_length_is_300 (L V : ℝ)
    (h1 : L = V * 20)
    (h2 : L + 285 = V * 39) :
    L = 300 := by
  sorry

end train_length_is_300_l75_75936


namespace initial_money_l75_75105

/-
We had $3500 left after spending 30% of our money on clothing, 
25% on electronics, and saving 15% in a bank account. 
How much money (X) did we start with before shopping and saving?
-/

theorem initial_money (M : ℝ) 
  (h_clothing : 0.30 * M ≠ 0) 
  (h_electronics : 0.25 * M ≠ 0) 
  (h_savings : 0.15 * M ≠ 0) 
  (remaining_money : 0.30 * M = 3500) : 
  M = 11666.67 := 
sorry

end initial_money_l75_75105


namespace eval_ceil_floor_l75_75490

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l75_75490


namespace probability_of_two_boys_three_girls_l75_75916

noncomputable def probability_family_five_children_two_boys_three_girls 
  (n k : ℕ) (p : ℝ) (h : n = 5) (h1 : k = 2) (h2 : p = 0.5) : ℝ :=
  if h ∧ h1 ∧ h2 then (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) else 0

theorem probability_of_two_boys_three_girls : probability_family_five_children_two_boys_three_girls 5 2 0.5 5 rfl 2 rfl 0.5 rfl = 0.3125 := 
by {  sorry }

end probability_of_two_boys_three_girls_l75_75916


namespace min_c_value_l75_75862

theorem min_c_value 
  (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + 1 = b)
  (h6 : b + 1 = c)
  (h7 : c + 1 = d)
  (h8 : d + 1 = e)
  (h9 : ∃ k : ℕ, k ^ 2 = b + c + d)
  (h10 : ∃ m : ℕ, m ^ 3 = a + b + c + d + e) : 
  c = 675 := 
sorry

end min_c_value_l75_75862


namespace cistern_fill_time_l75_75790

-- Let F be the rate at which the first tap fills the cistern (cisterns per hour)
def F : ℚ := 1 / 4

-- Let E be the rate at which the second tap empties the cistern (cisterns per hour)
def E : ℚ := 1 / 5

-- Prove that the time it takes to fill the cistern is 20 hours given the rates F and E
theorem cistern_fill_time : (1 / (F - E)) = 20 := 
by
  -- Insert necessary proofs here
  sorry

end cistern_fill_time_l75_75790


namespace find_a_for_polynomial_identity_l75_75249

theorem find_a_for_polynomial_identity : 
  ∃ (a : ℤ), ∀ (b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) → a = 5 :=
begin
  sorry
end

end find_a_for_polynomial_identity_l75_75249


namespace largest_percentage_increase_l75_75007

def students_2003 := 80
def students_2004 := 88
def students_2005 := 94
def students_2006 := 106
def students_2007 := 130

theorem largest_percentage_increase :
  let incr_03_04 := (students_2004 - students_2003) / students_2003 * 100
  let incr_04_05 := (students_2005 - students_2004) / students_2004 * 100
  let incr_05_06 := (students_2006 - students_2005) / students_2005 * 100
  let incr_06_07 := (students_2007 - students_2006) / students_2006 * 100
  incr_06_07 > incr_03_04 ∧
  incr_06_07 > incr_04_05 ∧
  incr_06_07 > incr_05_06 :=
by
  -- Proof goes here
  sorry

end largest_percentage_increase_l75_75007


namespace part1_part2_part3_l75_75044

-- Define the function f(x) and its derivative
def f (a x : ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- The value of a such that the slope of the tangent line to y = f(x) at x = 0 is 3
theorem part1 (a : ℝ) (h : deriv (λ x, f a x) 0 = 3) : a = 1/2 :=
by
  sorry

-- The range of a such that f(x) + f(-x) ≥ 12 * log x for any x ∈ (0, +∞)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, 0 < x → f a x + f a (-x) ≥ 12 * log x) : a ≤ -1 - 1/Real.e :=
by
  sorry

-- If a > 1, the minimum value of h(a) = M(a) - m(a) on [1, 2] is 8/27
theorem part3 (a : ℝ) (ha : 1 < a) :
  let M := λ a, max (f a 1) (f a 2)
      m := λ a, min (f a 1) (f a 2)
      h := λ a, M a - m a
  in h a = 8/27 :=
by
  sorry

end part1_part2_part3_l75_75044


namespace train_length_l75_75807

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def distance_ahead : ℝ := 270
noncomputable def time_to_pass : ℝ := 39

noncomputable def jogger_speed_mps := jogger_speed_kmph * (1000 / 1) * (1 / 3600)
noncomputable def train_speed_mps := train_speed_kmph * (1000 / 1) * (1 / 3600)

noncomputable def relative_speed_mps := train_speed_mps - jogger_speed_mps

theorem train_length :
  let jogger_speed := 9 * (1000 / 3600)
  let train_speed := 45 * (1000 / 3600)
  let relative_speed := train_speed - jogger_speed
  let distance := 270
  let time := 39
  distance + relative_speed * time = 390 → relative_speed * time = 120 := by
  sorry

end train_length_l75_75807


namespace largest_sum_fraction_l75_75474

open Rat

theorem largest_sum_fraction :
  let a := (2:ℚ) / 5
  let c1 := (1:ℚ) / 6
  let c2 := (1:ℚ) / 3
  let c3 := (1:ℚ) / 7
  let c4 := (1:ℚ) / 8
  let c5 := (1:ℚ) / 9
  max (a + c1) (max (a + c2) (max (a + c3) (max (a + c4) (a + c5)))) = a + c2
  ∧ a + c2 = (11:ℚ) / 15 := by
  sorry

end largest_sum_fraction_l75_75474


namespace solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l75_75765

theorem solution_of_inequality (a b x : ℝ) :
    (b - a * x > 0) ↔
    (a > 0 ∧ x < b / a ∨ 
     a < 0 ∧ x > b / a ∨ 
     a = 0 ∧ false) :=
by sorry

-- Additional theorems to rule out incorrect answers
theorem answer_A_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a|) → false :=
by sorry

theorem answer_B_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x < |b| / |a|) → false :=
by sorry

theorem answer_C_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > -|b| / |a|) → false :=
by sorry

theorem D_is_correct (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a| ∨ x < |b| / |a| ∨ x > -|b| / |a|) → false :=
by sorry

end solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l75_75765


namespace ring_worth_l75_75370

theorem ring_worth (R : ℝ) (h1 : (R + 2000 + 2 * R = 14000)) : R = 4000 :=
by 
  sorry

end ring_worth_l75_75370


namespace kirsty_initial_models_l75_75071

theorem kirsty_initial_models 
  (x : ℕ)
  (initial_price : ℝ)
  (increased_price : ℝ)
  (models_bought : ℕ)
  (h_initial_price : initial_price = 0.45)
  (h_increased_price : increased_price = 0.5)
  (h_models_bought : models_bought = 27) 
  (h_total_saved : x * initial_price = models_bought * increased_price) :
  x = 30 :=
by 
  sorry

end kirsty_initial_models_l75_75071


namespace trainB_destination_time_l75_75438

def trainA_speed : ℕ := 90
def trainB_speed : ℕ := 135
def trainA_time_after_meeting : ℕ := 9
def trainB_time_after_meeting (x : ℕ) : ℕ := 18 - 3 * x

theorem trainB_destination_time : (trainA_time_after_meeting, trainA_speed) = (9, 90) → 
  (trainB_speed, trainB_time_after_meeting 3) = (135, 3) := by
  sorry

end trainB_destination_time_l75_75438


namespace number_of_valid_Ns_l75_75525

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l75_75525


namespace amusement_park_ticket_price_l75_75927

-- Conditions as definitions in Lean
def weekday_adult_ticket_cost : ℕ := 22
def weekday_children_ticket_cost : ℕ := 7
def weekend_adult_ticket_cost : ℕ := 25
def weekend_children_ticket_cost : ℕ := 10
def adult_discount_rate : ℕ := 20
def sales_tax_rate : ℕ := 10
def num_of_adults : ℕ := 2
def num_of_children : ℕ := 2

-- Correct Answer to be proved equivalent:
def expected_total_price := 66

-- Statement translating the problem to Lean proof obligation
theorem amusement_park_ticket_price :
  let cost_before_discount := (num_of_adults * weekend_adult_ticket_cost) + (num_of_children * weekend_children_ticket_cost)
  let discount := (num_of_adults * weekend_adult_ticket_cost) * adult_discount_rate / 100
  let subtotal := cost_before_discount - discount
  let sales_tax := subtotal * sales_tax_rate / 100
  let total_cost := subtotal + sales_tax
  total_cost = expected_total_price :=
by
  sorry

end amusement_park_ticket_price_l75_75927


namespace smallest_positive_integer_l75_75784

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l75_75784


namespace not_divisible_l75_75895

-- Defining the necessary conditions
variable (m : ℕ)

theorem not_divisible (m : ℕ) : ¬ (1000^m - 1 ∣ 1978^m - 1) :=
sorry

end not_divisible_l75_75895


namespace find_a5_l75_75697

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2 + 1) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h3 : S 1 = 2) :
  a 5 = 9 :=
sorry

end find_a5_l75_75697


namespace oranges_after_selling_l75_75546

-- Definitions derived from the conditions
def oranges_picked := 37
def oranges_sold := 10
def oranges_left := 27

-- The theorem to prove that Joan is left with 27 oranges
theorem oranges_after_selling (h : oranges_picked - oranges_sold = oranges_left) : oranges_left = 27 :=
by
  -- Proof omitted
  sorry

end oranges_after_selling_l75_75546


namespace faster_ship_speed_l75_75596

theorem faster_ship_speed :
  ∀ (x y : ℕ),
    (200 + 100 = 300) → -- Total distance covered for both directions
    (x + y) * 10 = 300 → -- Opposite direction equation
    (x - y) * 25 = 300 → -- Same direction equation
    x = 21 := 
by
  intros x y _ eq1 eq2
  sorry

end faster_ship_speed_l75_75596


namespace amoeba_after_ten_days_l75_75934

def amoeba_count (n : ℕ) : ℕ := 
  3^n

theorem amoeba_after_ten_days : amoeba_count 10 = 59049 := 
by
  -- proof omitted
  sorry

end amoeba_after_ten_days_l75_75934


namespace larger_number_is_23_l75_75413

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75413


namespace number_of_tables_l75_75286

-- Define the total number of customers the waiter is serving
def total_customers := 90

-- Define the number of women per table
def women_per_table := 7

-- Define the number of men per table
def men_per_table := 3

-- Define the total number of people per table
def people_per_table : ℕ := women_per_table + men_per_table

-- Statement to prove the number of tables
theorem number_of_tables (T : ℕ) (h : T * people_per_table = total_customers) : T = 9 := by
  sorry

end number_of_tables_l75_75286


namespace quadratic_trinomial_neg_values_l75_75500

theorem quadratic_trinomial_neg_values (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by
sorry

end quadratic_trinomial_neg_values_l75_75500


namespace smallest_n_fig2_valid_fig4_impossible_49_fig4_impossible_33_smallest_n_fig4_valid_l75_75613

noncomputable def smallest_n_fig2 : ℕ :=
  4

theorem smallest_n_fig2_valid : ∃ n : ℕ, n = smallest_n_fig2 ∧
  ∀ a b : ℕ, a ≠ b ∧ a, b ≤ n =
  (connected a b → gcd (a - b) n = 1) ∧
  (¬connected a b → gcd (a - b) n > 1) :=
sorry

theorem fig4_impossible_49 : ¬∃ (f : fin 5 → ℕ),
  (∀ (i j : fin 5), i ≠ j → gcd (f i - f j) 49 = 1 = connected i j ∧
    gcd (f i - f j) 49 > 1 = ¬connected i j) :=
sorry

theorem fig4_impossible_33 : ¬∃ (f : fin 5 → ℕ),
  (∀ (i j : fin 5), i ≠ j → gcd (f i - f j) 33 = 1 = connected i j ∧
    gcd (f i - f j) 33 > 1 = ¬connected i j) :=
sorry

noncomputable def smallest_n_fig4 : ℕ :=
  105

theorem smallest_n_fig4_valid : ∃ n : ℕ, n = smallest_n_fig4 ∧
  ∀ a b : ℕ, a ≠ b ∧ a, b ≤ n =
  (connected a b → gcd (a - b) n = 1) ∧
  (¬connected a b → gcd (a - b) n > 1) :=
sorry

end smallest_n_fig2_valid_fig4_impossible_49_fig4_impossible_33_smallest_n_fig4_valid_l75_75613


namespace find_a7_l75_75657

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l75_75657


namespace graph_of_cubic_equation_is_three_lines_l75_75298

theorem graph_of_cubic_equation_is_three_lines (x y : ℝ) :
  (x + y) ^ 3 = x ^ 3 + y ^ 3 →
  (y = -x ∨ x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_cubic_equation_is_three_lines_l75_75298


namespace katya_notebooks_l75_75624

theorem katya_notebooks (rubles: ℕ) (cost_per_notebook: ℕ) (stickers_per_exchange: ℕ) 
  (initial_rubles: ℕ) (initial_notebooks: ℕ) :
  (initial_notebooks = initial_rubles / cost_per_notebook) →
  (rubles = initial_notebooks * cost_per_notebook) →
  (initial_notebooks = 37) →
  (initial_rubles = 150) →
  (cost_per_notebook = 4) →
  (stickers_per_exchange = 5) →
  (rubles = 148) →
  let rec total_notebooks (notebooks stickers : ℕ) : ℕ :=
      if stickers < stickers_per_exchange then notebooks
      else let new_notebooks := stickers / stickers_per_exchange in
           total_notebooks (notebooks + new_notebooks) 
                           (stickers % stickers_per_exchange + new_notebooks) in
  total_notebooks initial_notebooks initial_notebooks = 46 :=
begin
  sorry
end

end katya_notebooks_l75_75624


namespace exponential_equality_l75_75719

theorem exponential_equality (n : ℕ) (h : 4 ^ n = 64 ^ 2) : n = 6 :=
  sorry

end exponential_equality_l75_75719


namespace good_numbers_l75_75810

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → (d + 1) ∣ (n + 1)

theorem good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n) :=
by
  sorry

end good_numbers_l75_75810


namespace conditional_prob_eventA_given_eventB_l75_75087

noncomputable def eventA (d1 d2 d3 : ℕ) : Prop :=
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

noncomputable def eventB (d1 d2 d3 : ℕ) : Prop :=
  d1 = 2 ∨ d2 = 2 ∨ d3 = 2

theorem conditional_prob_eventA_given_eventB :
  -- Total number of outcomes for three dice
  let total_outcomes := 6^3 in
  -- Number of outcomes where no die shows a 2
  let no_two_outcomes := 5^3 in
  -- Number of outcomes for event B
  let B_outcomes := total_outcomes - no_two_outcomes in
  -- Number of favorable outcomes for both A and B
  let favorable_outcomes := 3 * 5 * 4 in
  -- Conditional probability P(A|B)
  let P_A_given_B := (favorable_outcomes : ℝ) / (B_outcomes : ℝ) in
  P_A_given_B = 60 / 91 :=
begin
  sorry
end

end conditional_prob_eventA_given_eventB_l75_75087


namespace angle_at_3_15_is_7_point_5_degrees_l75_75258

-- Definitions for the positions of the hour and minute hands
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60.0) * 360.0
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

-- The time 3:15
def time_3_15 := (3, 15)

-- The acute angle calculation
def acute_angle_between_hands (hour: ℕ) (minute: ℕ) : ℝ :=
  let minute_angle := minute_hand_position minute
  let hour_angle := hour_hand_position hour minute
  abs (minute_angle - hour_angle)

-- The theorem statement
theorem angle_at_3_15_is_7_point_5_degrees : 
  acute_angle_between_hands 3 15 = 7.5 := 
  sorry

end angle_at_3_15_is_7_point_5_degrees_l75_75258


namespace coefficient_of_x_in_expansion_l75_75872

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l75_75872


namespace solve_for_x_l75_75052

-- Definitions based on provided conditions
variables (x : ℝ) -- defining x as a real number
def condition : Prop := 0.25 * x = 0.15 * 1600 - 15

-- The theorem stating that x equals 900 given the condition
theorem solve_for_x (h : condition x) : x = 900 :=
by
  sorry

end solve_for_x_l75_75052


namespace no_perfect_squares_xy_zt_l75_75176

theorem no_perfect_squares_xy_zt
    (x y z t : ℕ) 
    (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < t)
    (h_eq1 : x + y = z + t) 
    (h_eq2 : xy - zt = x + y) : ¬(∃ a b : ℕ, xy = a^2 ∧ zt = b^2) :=
by
  sorry

end no_perfect_squares_xy_zt_l75_75176


namespace geometric_seq_ad_eq_2_l75_75704

open Real

def geometric_sequence (a b c d : ℝ) : Prop :=
∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r 

def is_max_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
f x = y ∧ ∀ z : ℝ, z ≠ x → f x ≥ f z

theorem geometric_seq_ad_eq_2 (a b c d : ℝ) :
  geometric_sequence a b c d →
  is_max_point (λ x => 3 * x - x ^ 3) b c →
  a * d = 2 :=
by
  sorry

end geometric_seq_ad_eq_2_l75_75704


namespace intersection_point_in_AB_l75_75839

def A (p : ℝ × ℝ) : Prop := p.snd = 2 * p.fst - 1
def B (p : ℝ × ℝ) : Prop := p.snd = p.fst + 3

theorem intersection_point_in_AB : (4, 7) ∈ {p : ℝ × ℝ | A p} ∩ {p : ℝ × ℝ | B p} :=
by
  sorry

end intersection_point_in_AB_l75_75839


namespace expression_evaluation_l75_75956

open Rat

theorem expression_evaluation :
  ∀ (a b c : ℚ),
  c = b - 4 →
  b = a + 4 →
  a = 3 →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 :=
by
  intros a b c hc hb ha h1 h2 h3
  simp [hc, hb, ha]
  have h1 : 3 + 1 ≠ 0 := by norm_num
  have h2 : 7 - 3 ≠ 0 := by norm_num
  have h3 : 3 + 7 ≠ 0 := by norm_num
  -- Placeholder for the simplified expression computation
  sorry

end expression_evaluation_l75_75956


namespace probability_of_red_card_l75_75356

theorem probability_of_red_card (successful_attempts not_successful_attempts : ℕ) (h : successful_attempts = 5) (h2 : not_successful_attempts = 8) : (successful_attempts / (successful_attempts + not_successful_attempts) : ℚ) = 5 / 13 := by
  sorry

end probability_of_red_card_l75_75356


namespace parabola_hyperbola_tangent_l75_75244

-- Definitions of the parabola and hyperbola
def parabola (x : ℝ) : ℝ := x^2 + 4
def hyperbola (x y : ℝ) (m : ℝ) : Prop := y^2 - m*x^2 = 1

-- Tangency condition stating that the parabola and hyperbola are tangent implies m = 8 + 2*sqrt(15)
theorem parabola_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, parabola x = y → hyperbola x y m) → m = 8 + 2 * Real.sqrt 15 :=
by
  sorry

end parabola_hyperbola_tangent_l75_75244


namespace John_bought_new_socks_l75_75213

theorem John_bought_new_socks (initial_socks : ℕ) (thrown_away_socks : ℕ) (current_socks : ℕ) :
    initial_socks = 33 → thrown_away_socks = 19 → current_socks = 27 → 
    current_socks = (initial_socks - thrown_away_socks) + 13 :=
by
  sorry

end John_bought_new_socks_l75_75213


namespace total_stamps_l75_75820

-- Definitions based on the conditions
def AJ := 370
def KJ := AJ / 2
def CJ := 2 * KJ + 5

-- Proof Statement
theorem total_stamps : AJ + KJ + CJ = 930 := by
  sorry

end total_stamps_l75_75820


namespace sum_modulo_seven_l75_75822

theorem sum_modulo_seven :
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  s % 7 = 2 :=
by
  sorry

end sum_modulo_seven_l75_75822


namespace turtles_remaining_on_log_l75_75128

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l75_75128


namespace jason_spent_on_shorts_l75_75211

def total_spent : ℝ := 14.28
def jacket_spent : ℝ := 4.74
def shorts_spent : ℝ := total_spent - jacket_spent

theorem jason_spent_on_shorts :
  shorts_spent = 9.54 :=
by
  -- Placeholder for the proof. The statement is correct as it matches the given problem data.
  sorry

end jason_spent_on_shorts_l75_75211


namespace zoe_distance_more_than_leo_l75_75788

theorem zoe_distance_more_than_leo (d t s : ℝ)
  (maria_driving_time : ℝ := t + 2)
  (maria_speed : ℝ := s + 15)
  (zoe_driving_time : ℝ := t + 3)
  (zoe_speed : ℝ := s + 20)
  (leo_distance : ℝ := s * t)
  (maria_distance : ℝ := (s + 15) * (t + 2))
  (zoe_distance : ℝ := (s + 20) * (t + 3))
  (maria_leo_distance_diff : ℝ := 110)
  (h1 : maria_distance = leo_distance + maria_leo_distance_diff)
  : zoe_distance - leo_distance = 180 :=
by
  sorry

end zoe_distance_more_than_leo_l75_75788


namespace hyperbola_equation_l75_75494

theorem hyperbola_equation :
  ∃ (b : ℝ), (∀ (x y : ℝ), ((x = 2) ∧ (y = 2)) →
    ((x^2 / 5) - (y^2 / b^2) = 1)) ∧
    (∀ x y, (y = (2 / Real.sqrt 5) * x) ∨ (y = -(2 / Real.sqrt 5) * x) → 
    (∀ (a b : ℝ), (a = 2) → (b = 2) →
      (b^2 = 4) → ((5 * y^2 / 4) - x^2 = 1))) :=
sorry

end hyperbola_equation_l75_75494


namespace find_parallel_line_l75_75645

def line1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y + 2 = 0
def parallelLine : ℝ → ℝ → Prop := λ x y => 4 * x + y - 4 = 0

theorem find_parallel_line (x y : ℝ) (hx : line1 x y) (hy : line2 x y) : 
  ∃ c : ℝ, (λ x y => 4 * x + y + c = 0) (2:ℝ) (2:ℝ) ∧ 
          ∀ x' y', (λ x' y' => 4 * x' + y' + c = 0) x' y' ↔ 4 * x' + y' - 10 = 0 := 
sorry

end find_parallel_line_l75_75645


namespace sum_of_cubes_of_roots_l75_75604

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) (h₀ : 3 * x₁ ^ 2 - 5 * x₁ - 2 = 0)
  (h₁ : 3 * x₂ ^ 2 - 5 * x₂ - 2 = 0) :
  x₁^3 + x₂^3 = 215 / 27 :=
by sorry

end sum_of_cubes_of_roots_l75_75604


namespace sum_of_number_and_reverse_divisible_by_11_l75_75085

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) (hA : 0 ≤ A) (hA9 : A ≤ 9) (hB : 0 ≤ B) (hB9 : B ≤ 9) :
  11 ∣ ((10 * A + B) + (10 * B + A)) :=
by
  sorry

end sum_of_number_and_reverse_divisible_by_11_l75_75085


namespace find_a7_l75_75677

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l75_75677


namespace Tim_total_payment_l75_75590

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l75_75590


namespace smallest_solution_l75_75315

open Int

theorem smallest_solution (x : ℝ) (h : ⌊x⌋ = 3 + 50 * (x - ⌊x⌋)) : x = 3.00 :=
by
  sorry

end smallest_solution_l75_75315


namespace find_f_g_3_l75_75882

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_f_g_3 :
  f (g 3) = -2 := by
  sorry

end find_f_g_3_l75_75882


namespace students_activity_details_l75_75062

open Finset

-- Definitions for sets and their cardinalities
variable (S G P : Finset ℕ)
variable (total_students spanish_students german_students sports_students : ℕ)
variable (intersection_SP intersection_SG intersection_GP intersection_SGP : ℕ)

def only_one_activity_students : ℕ :=
  (spanish_students - intersection_SG - intersection_SP + intersection_SGP) +
  (german_students - intersection_SG - intersection_GP + intersection_SGP) +
  (sports_students - intersection_SP - intersection_GP + intersection_SGP)

def no_activity_students : ℕ :=
  total_students - (spanish_students + german_students + sports_students -
                     intersection_SP - intersection_SG - intersection_GP + intersection_SGP)

theorem students_activity_details
  (h1 : total_students = 94)
  (h2 : spanish_students = 40)
  (h3 : german_students = 27)
  (h4 : sports_students = 60)
  (h5 : intersection_SP = 24)
  (h6 : intersection_SG = 10)
  (h7 : intersection_GP = 12)
  (h8 : intersection_SGP = 4) :
  only_one_activity_students total_students spanish_students german_students sports_students intersection_SP intersection_SG intersection_GP intersection_SGP = 47 ∧
  no_activity_students total_students spanish_students german_students sports_students intersection_SP intersection_SG intersection_GP intersection_SGP = 9 := by
  sorry

end students_activity_details_l75_75062


namespace turtles_remaining_on_log_l75_75123
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l75_75123


namespace coffee_last_days_l75_75005

theorem coffee_last_days (weight : ℕ) (cups_per_lb : ℕ) (cups_per_day : ℕ) 
  (h_weight : weight = 3) 
  (h_cups_per_lb : cups_per_lb = 40) 
  (h_cups_per_day : cups_per_day = 3) : 
  (weight * cups_per_lb) / cups_per_day = 40 := 
by 
  sorry

end coffee_last_days_l75_75005


namespace find_width_fabric_width_is_3_l75_75491

variable (Area Length : ℝ)
variable (Width : ℝ)

theorem find_width (h1 : Area = 24) (h2 : Length = 8) :
  Width = Area / Length :=
sorry

theorem fabric_width_is_3 (h1 : Area = 24) (h2 : Length = 8) :
  (Area / Length) = 3 :=
by
  have h : Area / Length = 3 := by sorry
  exact h

end find_width_fabric_width_is_3_l75_75491


namespace total_cost_one_each_l75_75435

theorem total_cost_one_each (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 6.3)
  (h2 : 4 * x + 10 * y + z = 8.4) :
  x + y + z = 2.1 :=
  sorry

end total_cost_one_each_l75_75435


namespace housewife_money_left_l75_75806

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end housewife_money_left_l75_75806


namespace point_distance_l75_75946

theorem point_distance (x y n : ℝ) 
    (h1 : abs x = 8) 
    (h2 : (x - 3)^2 + (y - 10)^2 = 225) 
    (h3 : y > 10) 
    (hn : n = Real.sqrt (x^2 + y^2)) : 
    n = Real.sqrt (364 + 200 * Real.sqrt 2) := 
sorry

end point_distance_l75_75946


namespace larger_number_l75_75427

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l75_75427


namespace evaluate_expression_l75_75955

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end evaluate_expression_l75_75955


namespace total_nails_used_l75_75801

-- Given definitions from the conditions
def square_side_length : ℕ := 36
def nails_per_side : ℕ := 40
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

-- Statement of the problem proof
theorem total_nails_used : nails_per_side * sides_of_square - corners_of_square = 156 := by
  sorry

end total_nails_used_l75_75801


namespace cube_volume_l75_75926

theorem cube_volume (s : ℝ) (h : s ^ 2 = 64) : s ^ 3 = 512 :=
sorry

end cube_volume_l75_75926


namespace driver_travel_distance_per_week_l75_75280

noncomputable def daily_distance := 30 * 3 + 25 * 4 + 40 * 2

noncomputable def total_weekly_distance := daily_distance * 6 + 35 * 5

theorem driver_travel_distance_per_week : total_weekly_distance = 1795 := by
  simp [daily_distance, total_weekly_distance]
  done

end driver_travel_distance_per_week_l75_75280


namespace simplify_expression_l75_75374

theorem simplify_expression (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) 
  (h : x^2 + y^2 + z^2 = xy + yz + zx) : 
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) = 3 / x^2 := 
by
  sorry

end simplify_expression_l75_75374


namespace eval_ceil_floor_sum_l75_75484

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l75_75484


namespace larger_number_is_23_l75_75429

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75429


namespace clock_angle_at_315_l75_75259

theorem clock_angle_at_315 : 
  (angle_between_hour_and_minute_at (hours := 3) (minutes := 15)) = 7.5 :=
sorry

end clock_angle_at_315_l75_75259


namespace determine_ω_and_φ_l75_75182

noncomputable def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) := (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ d > 0, d < T ∧ ∀ m n : ℤ, m ≠ n → f (m * d) ≠ f (n * d))

theorem determine_ω_and_φ :
  ∃ ω φ : ℝ,
    (0 < ω) ∧
    (|φ| < Real.pi / 2) ∧
    (smallest_positive_period (f ω φ) Real.pi) ∧
    (f 0 ω φ = Real.sqrt 3) ∧
    (ω = 2 ∧ φ = Real.pi / 3) :=
by
  sorry

end determine_ω_and_φ_l75_75182


namespace toms_score_l75_75773

theorem toms_score (T J : ℝ) (h1 : T = J + 30) (h2 : (T + J) / 2 = 90) : T = 105 := by
  sorry

end toms_score_l75_75773


namespace distance_between_stations_l75_75797

/-- Two trains start at the same time from two stations and proceed towards each other.
    Train 1 travels at 20 km/hr.
    Train 2 travels at 25 km/hr.
    When they meet, Train 2 has traveled 55 km more than Train 1.
    Prove that the distance between the two stations is 495 km. -/
theorem distance_between_stations : ∃ x t : ℕ, 20 * t = x ∧ 25 * t = x + 55 ∧ 2 * x + 55 = 495 :=
by {
  sorry
}

end distance_between_stations_l75_75797


namespace distance_between_points_l75_75255

theorem distance_between_points :
  let (x1, y1) := (1, 2)
  let (x2, y2) := (6, 5)
  let d := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  d = Real.sqrt 34 :=
by
  sorry

end distance_between_points_l75_75255


namespace C_investment_l75_75108

theorem C_investment (A B C_profit total_profit : ℝ) (hA : A = 24000) (hB : B = 32000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 92000) (x : ℝ) (h : x / (A + B + x) = C_profit / total_profit) : x = 36000 := 
by
  sorry

end C_investment_l75_75108


namespace brick_wall_problem_l75_75292

theorem brick_wall_problem
  (b : ℕ)
  (rate_ben rate_arya : ℕ → ℕ)
  (combined_rate : ℕ → ℕ → ℕ)
  (work_duration : ℕ)
  (effective_combined_rate : ℕ → ℕ × ℕ → ℕ)
  (rate_ben_def : ∀ (b : ℕ), rate_ben b = b / 12)
  (rate_arya_def : ∀ (b : ℕ), rate_arya b = b / 15)
  (combined_rate_def : ∀ (b : ℕ), combined_rate (rate_ben b) (rate_arya b) = rate_ben b + rate_arya b)
  (effective_combined_rate_def : ∀ (b : ℕ), effective_combined_rate b (rate_ben b, rate_arya b) = combined_rate (rate_ben b) (rate_arya b) - 15)
  (work_duration_def : work_duration = 6)
  (completion_condition : ∀ (b : ℕ), work_duration * effective_combined_rate b (rate_ben b, rate_arya b) = b) :
  b = 900 :=
by
  -- Proof would go here
  sorry

end brick_wall_problem_l75_75292


namespace union_of_M_and_N_l75_75856

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x ≥ -1}
  let N := {x : ℝ | -real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2}
  M ∪ N = {x : ℝ | x ≥ -real.sqrt 2} :=
by
  sorry

end union_of_M_and_N_l75_75856


namespace product_of_numbers_l75_75609

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 :=
by
  sorry

end product_of_numbers_l75_75609


namespace range_of_a_l75_75534

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := (a-2)*x^2 + 2*(a-2)*x < 4

-- The main theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, inequality a x) ↔ (-2 : ℝ) < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l75_75534


namespace bigger_part_is_45_l75_75614

variable (x y : ℕ)

theorem bigger_part_is_45
  (h1 : x + y = 60)
  (h2 : 10 * x + 22 * y = 780) :
  max x y = 45 := by
  sorry

end bigger_part_is_45_l75_75614


namespace larger_number_is_23_l75_75411

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l75_75411


namespace geometric_sequence_first_term_l75_75570

theorem geometric_sequence_first_term 
  (T : ℕ → ℝ) 
  (h1 : T 5 = 243) 
  (h2 : T 6 = 729) 
  (hr : ∃ r : ℝ, ∀ n : ℕ, T n = T 1 * r^(n - 1)) :
  T 1 = 3 :=
by
  sorry

end geometric_sequence_first_term_l75_75570


namespace transmission_time_is_128_l75_75309

def total_time (blocks chunks_per_block rate : ℕ) : ℕ :=
  (blocks * chunks_per_block) / rate

theorem transmission_time_is_128 :
  total_time 80 256 160 = 128 :=
  by
  sorry

end transmission_time_is_128_l75_75309


namespace area_of_enclosed_region_l75_75632

theorem area_of_enclosed_region :
  let region := { p : ℝ × ℝ | abs (2 * p.1 + p.2) + abs (p.1 - 2 * p.2) ≤ 6 }
  area region = 6 * real.sqrt 3 :=
by sorry

end area_of_enclosed_region_l75_75632


namespace max_elevation_reached_l75_75283

theorem max_elevation_reached 
  (t : ℝ) 
  (s : ℝ) 
  (h : s = 200 * t - 20 * t^2) : 
  ∃ t_max : ℝ, ∃ s_max : ℝ, t_max = 5 ∧ s_max = 500 ∧ s_max = 200 * t_max - 20 * t_max^2 := sorry

end max_elevation_reached_l75_75283


namespace optimal_fence_area_l75_75436

variables {l w : ℝ}

theorem optimal_fence_area
  (h1 : 2 * l + 2 * w = 400) -- Tiffany must use exactly 400 feet of fencing.
  (h2 : l ≥ 100) -- The length must be at least 100 feet.
  (h3 : w ≥ 50) -- The width must be at least 50 feet.
  : l * w ≤ 10000 :=      -- We need to prove that the area is at most 10000 square feet.
by
  sorry

end optimal_fence_area_l75_75436


namespace side_length_of_square_l75_75445

-- Mathematical definitions and conditions
def square_area (side : ℕ) : ℕ := side * side

theorem side_length_of_square {s : ℕ} (h : square_area s = 289) : s = 17 :=
sorry

end side_length_of_square_l75_75445


namespace total_treats_value_l75_75299

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l75_75299


namespace paint_liters_needed_l75_75225

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed_l75_75225


namespace no_sum_of_19_l75_75104

theorem no_sum_of_19 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6)
  (hprod : a * b * c * d = 180) : a + b + c + d ≠ 19 :=
sorry

end no_sum_of_19_l75_75104


namespace calculate_rent_l75_75217

def monthly_income : ℝ := 3200
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def car_payment : ℝ := 350
def gas_maintenance : ℝ := 350

def total_expenses : ℝ := utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance
def rent : ℝ := monthly_income - total_expenses

theorem calculate_rent : rent = 1250 := by
  -- condition proof here
  sorry

end calculate_rent_l75_75217


namespace arcsin_eq_pi_div_two_solve_l75_75749

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l75_75749


namespace trigonometric_identity_l75_75919

theorem trigonometric_identity :
  8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 :=
by
  sorry

end trigonometric_identity_l75_75919


namespace question1_solution_question2_solution_l75_75708

-- Definitions of the problem conditions
def f (x a : ℝ) : ℝ := abs (x - a)

-- First proof problem (Question 1)
theorem question1_solution (x : ℝ) : (f x 2) ≥ (4 - abs (x - 4)) ↔ (x ≥ 5 ∨ x ≤ 1) :=
by sorry

-- Second proof problem (Question 2)
theorem question2_solution (x : ℝ) (a : ℝ) (h_sol : 1 ≤ x ∧ x ≤ 2) 
  (h_ineq : abs (f (2 * x + a) a - 2 * f x a) ≤ 2) : a = 3 :=
by sorry

end question1_solution_question2_solution_l75_75708


namespace sum_of_fractions_l75_75294

theorem sum_of_fractions :
  (3 / 9) + (6 / 12) = 5 / 6 := by
  sorry

end sum_of_fractions_l75_75294


namespace gardener_total_expenses_l75_75540

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l75_75540


namespace number_of_camels_l75_75200

theorem number_of_camels (hens goats keepers camel_feet heads total_feet : ℕ)
  (h_hens : hens = 50) (h_goats : goats = 45) (h_keepers : keepers = 15)
  (h_feet_diff : total_feet = heads + 224)
  (h_heads : heads = hens + goats + keepers)
  (h_hens_feet : hens * 2 = 100)
  (h_goats_feet : goats * 4 = 180)
  (h_keepers_feet : keepers * 2 = 30)
  (h_camels_feet : camel_feet = 24)
  (h_total_feet : total_feet = 334)
  (h_feet_without_camels : 100 + 180 + 30 = 310) :
  camel_feet / 4 = 6 := sorry

end number_of_camels_l75_75200


namespace percentage_of_blue_flowers_l75_75888

theorem percentage_of_blue_flowers 
  (total_flowers : Nat)
  (red_flowers : Nat)
  (white_flowers : Nat)
  (total_flowers_eq : total_flowers = 10)
  (red_flowers_eq : red_flowers = 4)
  (white_flowers_eq : white_flowers = 2)
  :
  ( (total_flowers - (red_flowers + white_flowers)) * 100 ) / total_flowers = 40 :=
by
  sorry

end percentage_of_blue_flowers_l75_75888


namespace find_speed_ratio_l75_75597

noncomputable def circular_track_speed_ratio (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0) : Prop :=
  let t_1 := C / (v_V + v_P)
  let t_2 := (C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r

theorem find_speed_ratio
  (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0)
  (meeting1 : v_V * (C / (v_V + v_P)) + v_P * (C / (v_V + v_P)) = C)
  (lap_vasya : v_V * ((C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))) = C + v_V * (C / (v_V + v_P)))
  (lap_petya : v_P * ((C * (2 * v_P + v_V)) / (v_P * (v_V + v_P))) = C + v_P * (C / (v_V + v_P))) :
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r :=
  sorry

end find_speed_ratio_l75_75597


namespace length_of_train_l75_75140

-- Definitions based on the conditions in the problem
def time_to_cross_signal_pole : ℝ := 18
def time_to_cross_platform : ℝ := 54
def length_of_platform : ℝ := 600.0000000000001

-- Prove that the length of the train is 300.00000000000005 meters
theorem length_of_train
    (L V : ℝ)
    (h1 : L = V * time_to_cross_signal_pole)
    (h2 : L + length_of_platform = V * time_to_cross_platform) :
    L = 300.00000000000005 :=
by
  sorry

end length_of_train_l75_75140


namespace arith_seq_general_formula_geom_seq_sum_l75_75337

-- Problem 1
theorem arith_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (h_d : d = 3) (h_a1 : a 1 = 4) :
  a n = 3 * n + 1 :=
sorry

-- Problem 2
theorem geom_seq_sum (b : ℕ → ℚ) (S : ℕ → ℚ) (h_b1 : b 1 = 1 / 3) (r : ℚ) (h_r : r = 1 / 3) :
  S n = (1 / 2) * (1 - (1 / 3 ^ n)) :=
sorry

end arith_seq_general_formula_geom_seq_sum_l75_75337


namespace right_triangle_legs_l75_75362

theorem right_triangle_legs (a b : ℕ) (hypotenuse : ℕ) (h : hypotenuse = 39) : a^2 + b^2 = 39^2 → (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l75_75362


namespace area_of_triangle_is_correct_l75_75833

def point := ℚ × ℚ

def A : point := (4, -4)
def B : point := (-1, 1)
def C : point := (2, -7)

def vector_sub (p1 p2 : point) : point :=
(p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℚ :=
v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℚ :=
(abs (determinant (vector_sub C A) (vector_sub C B))) / 2

theorem area_of_triangle_is_correct :
  area_of_triangle A B C = 12.5 :=
by sorry

end area_of_triangle_is_correct_l75_75833


namespace intersection_A_B_subsets_C_l75_75508

-- Definition of sets A and B
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | 0 ≤ x}

-- Definition of intersection C
def C : Set ℤ := A ∩ B

-- The proof statements
theorem intersection_A_B : C = {1, 2} := 
by sorry

theorem subsets_C : {s | s ⊆ C} = {∅, {1}, {2}, {1, 2}} := 
by sorry

end intersection_A_B_subsets_C_l75_75508


namespace arithmetic_sequence_common_difference_l75_75033

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 4) (hS4 : S 4 = 20)
  (hS_formula : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)) : 
  d = 3 :=
by sorry

end arithmetic_sequence_common_difference_l75_75033


namespace cylinder_height_l75_75459

theorem cylinder_height
  (V : ℝ → ℝ → ℝ) 
  (π : ℝ)
  (r h : ℝ)
  (vol_increase_height : ℝ)
  (vol_increase_radius : ℝ)
  (h_increase : ℝ)
  (r_increase : ℝ)
  (original_radius : ℝ) :
  V r h = π * r^2 * h → 
  vol_increase_height = π * r^2 * h_increase →
  vol_increase_radius = π * ((r + r_increase)^2 - r^2) * h →
  r = original_radius →
  vol_increase_height = 72 * π →
  vol_increase_radius = 72 * π →
  original_radius = 3 →
  r_increase = 2 →
  h_increase = 2 →
  h = 4.5 :=
by
  sorry

end cylinder_height_l75_75459


namespace largest_divisor_n4_minus_5n2_plus_6_l75_75835

theorem largest_divisor_n4_minus_5n2_plus_6 :
  ∀ (n : ℤ), (n^4 - 5 * n^2 + 6) % 1 = 0 :=
by
  sorry

end largest_divisor_n4_minus_5n2_plus_6_l75_75835


namespace opposite_and_reciprocal_numbers_l75_75847

theorem opposite_and_reciprocal_numbers (a b c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1) :
  2019 * a + (7 / (c * d)) + 2019 * b = 7 :=
sorry

end opposite_and_reciprocal_numbers_l75_75847


namespace hexagon_interior_angles_sum_l75_75581

theorem hexagon_interior_angles_sum :
  let n := 6 in
  (n - 2) * 180 = 720 :=
by
  sorry

end hexagon_interior_angles_sum_l75_75581


namespace sequence_sum_zero_l75_75827

-- Define the sequence as a function
def seq (n : ℕ) : ℤ :=
  if (n-1) % 8 < 4
  then (n+1) / 2
  else - (n / 2)

-- Define the sum of the sequence up to a given number
def seq_sum (m : ℕ) : ℤ :=
  (Finset.range (m+1)).sum (λ n => seq n)

-- The actual problem statement
theorem sequence_sum_zero : seq_sum 2012 = 0 :=
  sorry

end sequence_sum_zero_l75_75827


namespace find_x_collinear_l75_75047

theorem find_x_collinear (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (x, 1)) 
  (h_collinear : ∃ k : ℝ, (2 * 2 + x) = k * x ∧ (2 * -1 + 1) = k * 1) : x = -2 :=
by
  sorry

end find_x_collinear_l75_75047


namespace exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l75_75183

noncomputable def f (x a : ℝ) : ℝ :=
if x > a then (x - 1)^3 else abs (x - 1)

theorem exists_no_minimum_value :
  ∃ a : ℝ, ¬ ∃ m : ℝ, ∀ x : ℝ, f x a ≥ m :=
sorry

theorem has_zeros_for_any_a (a : ℝ) : ∃ x : ℝ, f x a = 0 :=
sorry

theorem not_monotonically_increasing_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  ¬ ∀ x y : ℝ, 1 < x → x < y → y < a → f x a ≤ f y a :=
sorry

theorem exists_m_for_3_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ m : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m :=
sorry

end exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l75_75183


namespace select_people_with_boys_and_girls_l75_75650

theorem select_people_with_boys_and_girls :
  let boys := 5
  let girls := 4
  ∃ (ways : ℕ), ways = (Nat.choose (boys + girls) 4 - Nat.choose boys 4 - Nat.choose girls 4) ∧ ways = 120 :=
by
  let boys := 5
  let girls := 4
  use (Nat.choose (boys + girls) 4 - Nat.choose boys 4 - Nat.choose girls 4)
  sorry

end select_people_with_boys_and_girls_l75_75650


namespace simplify_expression_l75_75388

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l75_75388


namespace tim_pencils_l75_75250

-- Problem statement: If x = 2 and z = 5, then y = z - x where y is the number of pencils Tim placed.
def pencils_problem (x y z : Nat) : Prop :=
  x = 2 ∧ z = 5 → y = z - x

theorem tim_pencils : pencils_problem 2 3 5 :=
by
  sorry

end tim_pencils_l75_75250


namespace distance_between_parallel_lines_l75_75174

theorem distance_between_parallel_lines
  (l1 : ∀ (x y : ℝ), 2*x + y + 1 = 0)
  (l2 : ∀ (x y : ℝ), 4*x + 2*y - 1 = 0) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 := by
  sorry

end distance_between_parallel_lines_l75_75174


namespace find_pairs_l75_75163

theorem find_pairs (m n : ℕ) : 
  ∃ x : ℤ, x * x = 2^m * 3^n + 1 ↔ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
by
  sorry

end find_pairs_l75_75163


namespace liam_birthday_next_monday_2018_l75_75376

-- Define year advancement rules
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define function to calculate next weekday
def next_weekday (current_day : ℕ) (years_elapsed : ℕ) : ℕ :=
  let advance := (years_elapsed / 4) * 2 + (years_elapsed % 4)
  (current_day + advance) % 7

theorem liam_birthday_next_monday_2018 :
  (next_weekday 4 3 = 0) :=
sorry

end liam_birthday_next_monday_2018_l75_75376


namespace sqrt_difference_inequality_l75_75013

noncomputable def sqrt10 := Real.sqrt 10
noncomputable def sqrt6 := Real.sqrt 6
noncomputable def sqrt7 := Real.sqrt 7
noncomputable def sqrt3 := Real.sqrt 3

theorem sqrt_difference_inequality : sqrt10 - sqrt6 < sqrt7 - sqrt3 :=
by 
  sorry

end sqrt_difference_inequality_l75_75013


namespace total_volume_structure_l75_75008

theorem total_volume_structure (d : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) 
  (r := d / 2) 
  (V_cone := (1 / 3) * π * r^2 * h_cone) 
  (V_cylinder := π * r^2 * h_cylinder) 
  (V_total := V_cone + V_cylinder) :
  d = 8 → h_cone = 9 → h_cylinder = 4 → V_total = 112 * π :=
by
  intros
  sorry

end total_volume_structure_l75_75008


namespace circuit_disconnected_scenarios_l75_75470

def num_scenarios_solder_points_fall_off (n : Nat) : Nat :=
  2 ^ n - 1

theorem circuit_disconnected_scenarios : num_scenarios_solder_points_fall_off 6 = 63 :=
by
  sorry

end circuit_disconnected_scenarios_l75_75470


namespace total_amount_paid_l75_75594

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l75_75594


namespace average_age_when_youngest_born_l75_75583

theorem average_age_when_youngest_born (n : ℕ) (current_average_age youngest age_difference total_ages : ℝ)
  (hc1 : n = 7)
  (hc2 : current_average_age = 30)
  (hc3 : youngest = 6)
  (hc4 : age_difference = youngest * 6)
  (hc5 : total_ages = n * current_average_age - age_difference) :
  total_ages / n = 24.857
:= sorry

end average_age_when_youngest_born_l75_75583


namespace exists_real_k_l75_75576

theorem exists_real_k (c : Fin 1998 → ℕ)
  (h1 : 0 ≤ c 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → m + n < 1998 → c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1) :
  ∃ k : ℝ, ∀ n : Fin 1998, 1 ≤ n → c n = Int.floor (n * k) :=
by
  sorry

end exists_real_k_l75_75576


namespace arithmetic_sequence_100th_term_l75_75982

theorem arithmetic_sequence_100th_term (a b : ℤ)
  (h1 : 2 * a - a = a) -- definition of common difference d where d = a
  (h2 : b - 2 * a = a) -- b = 3a
  (h3 : a - 6 - b = -2 * a - 6) -- consistency of fourth term
  (h4 : 6 * a = -6) -- equation to solve for a
  : (a + 99 * (2 * a - a)) = -100 := 
sorry

end arithmetic_sequence_100th_term_l75_75982


namespace average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l75_75722

theorem average_children_per_grade (G3_girls G3_boys G3_club : ℕ) 
                                  (G4_girls G4_boys G4_club : ℕ) 
                                  (G5_girls G5_boys G5_club : ℕ) 
                                  (H1 : G3_girls = 28) 
                                  (H2 : G3_boys = 35) 
                                  (H3 : G3_club = 12) 
                                  (H4 : G4_girls = 45) 
                                  (H5 : G4_boys = 42) 
                                  (H6 : G4_club = 15) 
                                  (H7 : G5_girls = 38) 
                                  (H8 : G5_boys = 51) 
                                  (H9 : G5_club = 10) :
   (63 + 87 + 89) / 3 = 79.67 :=
by sorry

theorem average_girls_per_grade (G3_girls G4_girls G5_girls : ℕ) 
                                (H1 : G3_girls = 28) 
                                (H2 : G4_girls = 45) 
                                (H3 : G5_girls = 38) :
   (28 + 45 + 38) / 3 = 37 :=
by sorry

theorem average_boys_per_grade (G3_boys G4_boys G5_boys : ℕ)
                               (H1 : G3_boys = 35) 
                               (H2 : G4_boys = 42) 
                               (H3 : G5_boys = 51) :
   (35 + 42 + 51) / 3 = 42.67 :=
by sorry

theorem average_club_members_per_grade (G3_club G4_club G5_club : ℕ) 
                                       (H1 : G3_club = 12)
                                       (H2 : G4_club = 15)
                                       (H3 : G5_club = 10) :
   (12 + 15 + 10) / 3 = 12.33 :=
by sorry

end average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l75_75722


namespace total_payment_l75_75587

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l75_75587


namespace parallelogram_area_example_l75_75858

def point := (ℚ × ℚ)
def parallelogram_area (A B C D : point) : ℚ :=
  let base := B.1 - A.1
  let height := C.2 - A.2
  base * height

theorem parallelogram_area_example : 
  parallelogram_area (1, 1) (7, 1) (4, 9) (10, 9) = 48 := by
  sorry

end parallelogram_area_example_l75_75858


namespace inequality_proof_l75_75333

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a / (b^2 * (c + 1))) + (b / (c^2 * (a + 1))) + (c / (a^2 * (b + 1))) ≥ 3 / 2 :=
sorry

end inequality_proof_l75_75333


namespace tangent_line_at_point_l75_75398

noncomputable def f : ℝ → ℝ := λ x => 2 * Real.log x + x^2 

def tangent_line_equation (x y : ℝ) : Prop :=
  4 * x - y - 3 = 0 

theorem tangent_line_at_point {x y : ℝ} (h : f 1 = 1) : 
  tangent_line_equation 1 1 ∧
  y = 4 * (x - 1) + 1 := 
sorry

end tangent_line_at_point_l75_75398


namespace mike_total_cost_self_correct_l75_75380

-- Definition of the given conditions
def cost_per_rose_bush : ℕ := 75
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def cost_per_tiger_tooth_aloes : ℕ := 100
def total_tiger_tooth_aloes : ℕ := 2

-- Calculate the total cost for Mike's plants
def total_cost_mike_self: ℕ := 
  (total_rose_bushes - friend_rose_bushes) * cost_per_rose_bush + total_tiger_tooth_aloes * cost_per_tiger_tooth_aloes

-- The main proposition to be proved
theorem mike_total_cost_self_correct : total_cost_mike_self = 500 := by
  sorry

end mike_total_cost_self_correct_l75_75380


namespace total_snakes_among_pet_owners_l75_75767

theorem total_snakes_among_pet_owners :
  let owns_only_snakes := 15
  let owns_cats_and_snakes := 7
  let owns_dogs_and_snakes := 10
  let owns_birds_and_snakes := 2
  let owns_snakes_and_hamsters := 3
  let owns_cats_dogs_and_snakes := 4
  let owns_cats_snakes_and_hamsters := 2
  let owns_all_categories := 1
  owns_only_snakes + owns_cats_and_snakes + owns_dogs_and_snakes + owns_birds_and_snakes + owns_snakes_and_hamsters + owns_cats_dogs_and_snakes + owns_cats_snakes_and_hamsters + owns_all_categories = 44 :=
by
  sorry

end total_snakes_among_pet_owners_l75_75767


namespace problem_solution_l75_75886

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x. Prove
    that the number of real solutions to the equation x² - 2⌊x⌋ - 3 = 0 is 3. -/
theorem problem_solution : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^2 - 2 * ⌊x⌋ - 3 = 0 := 
sorry

end problem_solution_l75_75886


namespace inverse_function_passes_through_point_a_l75_75846

theorem inverse_function_passes_through_point_a
  (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ (∀ x, (a^(x-3) + 1) = 2 ↔ x = 3) → (2 - 1)/(3-3) = 0 :=
by
  sorry

end inverse_function_passes_through_point_a_l75_75846


namespace tank_capacity_l75_75285

variable (x : ℝ) -- Total capacity of the tank

theorem tank_capacity (h1 : x / 8 = 120 / (1 / 2 - 1 / 8)) :
  x = 320 :=
by
  sorry

end tank_capacity_l75_75285


namespace laura_total_owed_l75_75878

-- Define the principal amounts charged each month
def january_charge : ℝ := 35
def february_charge : ℝ := 45
def march_charge : ℝ := 55
def april_charge : ℝ := 25

-- Define the respective interest rates for each month, as decimals
def january_interest_rate : ℝ := 0.05
def february_interest_rate : ℝ := 0.07
def march_interest_rate : ℝ := 0.04
def april_interest_rate : ℝ := 0.06

-- Define the interests accrued for each month's charges
def january_interest : ℝ := january_charge * january_interest_rate
def february_interest : ℝ := february_charge * february_interest_rate
def march_interest : ℝ := march_charge * march_interest_rate
def april_interest : ℝ := april_charge * april_interest_rate

-- Define the totals including original charges and their respective interests
def january_total : ℝ := january_charge + january_interest
def february_total : ℝ := february_charge + february_interest
def march_total : ℝ := march_charge + march_interest
def april_total : ℝ := april_charge + april_interest

-- Define the total amount owed a year later
def total_owed : ℝ := january_total + february_total + march_total + april_total

-- Prove that the total amount owed a year later is $168.60
theorem laura_total_owed :
  total_owed = 168.60 := by
  sorry

end laura_total_owed_l75_75878


namespace katya_total_notebooks_l75_75626

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end katya_total_notebooks_l75_75626


namespace teacher_zhang_friends_l75_75059

-- Define the conditions
def num_students : ℕ := 50
def both_friends : ℕ := 30
def neither_friend : ℕ := 1
def diff_in_friends : ℕ := 7

-- Prove that Teacher Zhang has 43 friends on social media
theorem teacher_zhang_friends : ∃ x : ℕ, 
  x + (x - diff_in_friends) - both_friends + neither_friend = num_students ∧ x = 43 := 
by
  sorry

end teacher_zhang_friends_l75_75059


namespace soda_cost_original_l75_75069

theorem soda_cost_original 
  (x : ℚ) -- note: x in rational numbers to capture fractional cost accurately
  (h1 : 3 * (0.90 * x) = 6) :
  x = 20 / 9 :=
by
  sorry

end soda_cost_original_l75_75069


namespace smallest_integer_n_l75_75448

theorem smallest_integer_n (n : ℕ) (h : Nat.lcm 60 n / Nat.gcd 60 n = 75) : n = 500 :=
sorry

end smallest_integer_n_l75_75448


namespace jungkook_seokjin_books_l75_75070

/-- Given the number of books Jungkook and Seokjin originally had and the number of books they 
   bought, prove that Jungkook has 7 more books than Seokjin. -/
theorem jungkook_seokjin_books
  (jungkook_initial : ℕ)
  (seokjin_initial : ℕ)
  (jungkook_bought : ℕ)
  (seokjin_bought : ℕ)
  (h1 : jungkook_initial = 28)
  (h2 : seokjin_initial = 28)
  (h3 : jungkook_bought = 18)
  (h4 : seokjin_bought = 11) :
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 :=
by
  sorry

end jungkook_seokjin_books_l75_75070


namespace smallest_possible_value_l75_75177

theorem smallest_possible_value (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) : 
  ∃ (m : ℝ), m = -1/12 ∧ (∀ x y : ℝ, (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → (x + y) / (x^2) ≥ m) :=
sorry

end smallest_possible_value_l75_75177


namespace simplify_expression_l75_75329

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l75_75329


namespace ratio_of_boys_l75_75466

variables {b g o : ℝ}

theorem ratio_of_boys (h1 : b = (1/2) * o)
  (h2 : g = o - b)
  (h3 : b + g + o = 1) :
  b = 1 / 4 :=
by
  sorry

end ratio_of_boys_l75_75466
