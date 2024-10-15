import Mathlib

namespace NUMINAMATH_GPT_find_y1_l94_9488

noncomputable def y1_proof : Prop :=
∃ (y1 y2 y3 : ℝ), 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1 ∧
(1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1 / 9 ∧
y1 = 1 / 2

-- Statement to be proven:
theorem find_y1 : y1_proof :=
sorry

end NUMINAMATH_GPT_find_y1_l94_9488


namespace NUMINAMATH_GPT_sail_pressure_l94_9419

theorem sail_pressure (k : ℝ) :
  (forall (V A : ℝ), P = k * A * (V : ℝ)^2) 
  → (P = 1.25) → (V = 20) → (A = 1)
  → (A = 4) → (V = 40)
  → (P = 20) :=
by
  sorry

end NUMINAMATH_GPT_sail_pressure_l94_9419


namespace NUMINAMATH_GPT_value_of_expression_l94_9490

-- defining the conditions
def in_interval (a : ℝ) : Prop := 1 < a ∧ a < 2

-- defining the algebraic expression
def algebraic_expression (a : ℝ) : ℝ := abs (a - 2) + abs (1 - a)

-- theorem to be proved
theorem value_of_expression (a : ℝ) (h : in_interval a) : algebraic_expression a = 1 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_value_of_expression_l94_9490


namespace NUMINAMATH_GPT_iPhone_savings_l94_9437

theorem iPhone_savings
  (costX costY : ℕ)
  (discount_same_model discount_mixed : ℝ)
  (h1 : costX = 600)
  (h2 : costY = 800)
  (h3 : discount_same_model = 0.05)
  (h4 : discount_mixed = 0.03) :
  (costX + costX + costY) - ((costX * (1 - discount_same_model)) * 2 + costY * (1 - discount_mixed)) = 84 :=
by
  sorry

end NUMINAMATH_GPT_iPhone_savings_l94_9437


namespace NUMINAMATH_GPT_solution_set_inequality_l94_9477

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- Statement to prove
theorem solution_set_inequality :
  is_odd_function f ∧ (∀ x > 0, f x = 2^x - 3)
  → {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l94_9477


namespace NUMINAMATH_GPT_inequality_proof_l94_9471

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l94_9471


namespace NUMINAMATH_GPT_paper_thickness_after_folds_l94_9426

def folded_thickness (initial_thickness : ℝ) (folds : ℕ) : ℝ :=
  initial_thickness * 2^folds

theorem paper_thickness_after_folds :
  folded_thickness 0.1 4 = 1.6 :=
by
  sorry

end NUMINAMATH_GPT_paper_thickness_after_folds_l94_9426


namespace NUMINAMATH_GPT_pencil_and_eraser_cost_l94_9411

theorem pencil_and_eraser_cost (p e : ℕ) :
  2 * p + e = 40 →
  p > e →
  e ≥ 3 →
  p + e = 22 :=
by
  sorry

end NUMINAMATH_GPT_pencil_and_eraser_cost_l94_9411


namespace NUMINAMATH_GPT_painting_time_l94_9404

theorem painting_time (t₁₂ : ℕ) (h : t₁₂ = 6) (r : ℝ) (hr : r = t₁₂ / 12) (n : ℕ) (hn : n = 20) : 
  t₁₂ + n * r = 16 := by
  sorry

end NUMINAMATH_GPT_painting_time_l94_9404


namespace NUMINAMATH_GPT_find_people_who_own_only_cats_l94_9464

variable (C : ℕ)

theorem find_people_who_own_only_cats
  (ownsOnlyDogs : ℕ)
  (ownsCatsAndDogs : ℕ)
  (ownsCatsDogsSnakes : ℕ)
  (totalPetOwners : ℕ)
  (h1 : ownsOnlyDogs = 15)
  (h2 : ownsCatsAndDogs = 5)
  (h3 : ownsCatsDogsSnakes = 3)
  (h4 : totalPetOwners = 59) :
  C = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_people_who_own_only_cats_l94_9464


namespace NUMINAMATH_GPT_smallest_nonneg_integer_l94_9458

theorem smallest_nonneg_integer (n : ℕ) (h : 0 ≤ n ∧ n < 53) :
  50 * n ≡ 47 [MOD 53] → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_nonneg_integer_l94_9458


namespace NUMINAMATH_GPT_solve_for_y_l94_9427

theorem solve_for_y (y : ℕ) (h : 9 / y^2 = 3 * y / 81) : y = 9 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l94_9427


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l94_9417

-- Define the quadratic function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

-- The problem statement to prove that "m < 1" is a sufficient condition
-- but not a necessary condition for the function f(x) to have a root.
theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 1 → ∃ x : ℝ, f x m = 0) ∧ ¬(¬(m < 1) → ∃ x : ℝ, f x m = 0) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l94_9417


namespace NUMINAMATH_GPT_part1_part2_l94_9474

-- Given Definitions
variable (p : ℕ) [hp : Fact (p > 3)] [prime : Fact (Nat.Prime p)]
variable (A_l : ℕ → ℕ)

-- Assertions to Prove
theorem part1 (l : ℕ) (hl : 1 ≤ l ∧ l ≤ p - 2) : A_l l % p = 0 :=
sorry

theorem part2 (l : ℕ) (hl : 1 < l ∧ l < p ∧ l % 2 = 1) : A_l l % (p * p) = 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l94_9474


namespace NUMINAMATH_GPT_combined_resistance_parallel_l94_9431

theorem combined_resistance_parallel (R1 R2 : ℝ) (r : ℝ) 
  (hR1 : R1 = 8) (hR2 : R2 = 9) (h_parallel : (1 / r) = (1 / R1) + (1 / R2)) : 
  r = 72 / 17 :=
by
  sorry

end NUMINAMATH_GPT_combined_resistance_parallel_l94_9431


namespace NUMINAMATH_GPT_analytical_expression_satisfies_conditions_l94_9400

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

theorem analytical_expression_satisfies_conditions :
  is_increasing f ∧ (∀ x : ℝ, f x > 1) :=
by
  sorry

end NUMINAMATH_GPT_analytical_expression_satisfies_conditions_l94_9400


namespace NUMINAMATH_GPT_functional_equation_solution_l94_9416

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y, f (x ^ 2) - f (y ^ 2) + 2 * x + 1 = f (x + y) * f (x - y)) :
  (∀ x, f x = x + 1) ∨ (∀ x, f x = -x - 1) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l94_9416


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l94_9456

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- The terms of the arithmetic sequence

theorem arithmetic_sequence_a5 :
  (∀ (n : ℕ), a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  a_n 1 = 1 →
  a_n 1 + a_n 3 = 16 →
  a_n 4 = 15 :=
by {
  -- Proof omission, ensure these statements are correct with sorry
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_a5_l94_9456


namespace NUMINAMATH_GPT_find_k_l94_9407

theorem find_k (x k : ℝ) (h : ((x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) ∧ k ≠ 0) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l94_9407


namespace NUMINAMATH_GPT_part_I_part_II_l94_9487

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (x a : ℝ) : ℝ := (f x a) + (g x)

theorem part_I (a : ℝ) :
  (∀ x > 0, f x a ≥ g x) → a ≤ 0.5 :=
by
  sorry

theorem part_II (a x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (hx1_lt_half : x1 < 0.5) :
  (h x1 a = 2 * x1^2 + Real.log x1) →
  (h x2 a = 2 * x2^2 + Real.log x2) →
  (x1 * x2 = 0.5) →
  h x1 a - h x2 a > (3 / 4) - Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l94_9487


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_solve_equation4_l94_9480

theorem solve_equation1 (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by sorry

theorem solve_equation2 (x : ℝ) : (x - 2)^2 = 5 * (x - 2) ↔ (x = 2 ∨ x = 7) :=
by sorry

theorem solve_equation3 (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) :=
by sorry

theorem solve_equation4 (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_solve_equation3_solve_equation4_l94_9480


namespace NUMINAMATH_GPT_coin_collection_problem_l94_9401

variable (n d q : ℚ)

theorem coin_collection_problem 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 20 * q = 340)
  (h3 : d = 2 * n) :
  q - n = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_coin_collection_problem_l94_9401


namespace NUMINAMATH_GPT_one_liter_fills_five_cups_l94_9472

-- Define the problem conditions and question in Lean 4
def one_liter_milliliters : ℕ := 1000
def cup_volume_milliliters : ℕ := 200

theorem one_liter_fills_five_cups : one_liter_milliliters / cup_volume_milliliters = 5 := 
by 
  sorry -- proof skipped

end NUMINAMATH_GPT_one_liter_fills_five_cups_l94_9472


namespace NUMINAMATH_GPT_range_of_m_l94_9420

theorem range_of_m (x m : ℝ) : (|x - 3| ≤ 2) → ((x - m + 1) * (x - m - 1) ≤ 0) → 
  (¬(|x - 3| ≤ 2) → ¬((x - m + 1) * (x - m - 1) ≤ 0)) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_range_of_m_l94_9420


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l94_9446

variables (a b c e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
          (c_eq : c = 4) (b_eq : b = 2 * Real.sqrt 3)
          (hyperbola_eq : c ^ 2 = a ^ 2 + b ^ 2)
          (projection_cond : 2 < (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ∧ (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ≤ 4)

theorem hyperbola_eccentricity : e = c / a := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l94_9446


namespace NUMINAMATH_GPT_ellipse_condition_sufficient_not_necessary_l94_9463

theorem ellipse_condition_sufficient_not_necessary (n : ℝ) :
  (-1 < n) ∧ (n < 2) → 
  (2 - n > 0) ∧ (n + 1 > 0) ∧ (2 - n > n + 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ellipse_condition_sufficient_not_necessary_l94_9463


namespace NUMINAMATH_GPT_solved_just_B_is_six_l94_9494

variables (a b c d e f g : ℕ)

-- Conditions given
axiom total_competitors : a + b + c + d + e + f + g = 25
axiom twice_as_many_solved_B : b + d = 2 * (c + d)
axiom only_A_one_more : a = 1 + (e + f + g)
axiom A_equals_B_plus_C : a = b + c

-- Prove that the number of competitors solving just problem B is 6.
theorem solved_just_B_is_six : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_solved_just_B_is_six_l94_9494


namespace NUMINAMATH_GPT_monomial_same_type_l94_9414

-- Define a structure for monomials
structure Monomial where
  coeff : ℕ
  vars : List String

-- Monomials definitions based on the given conditions
def m1 := Monomial.mk 3 ["a"]
def m2 := Monomial.mk 2 ["b"]
def m3 := Monomial.mk 1 ["a", "b"]
def m4 := Monomial.mk 3 ["a", "c"]
def target := Monomial.mk 2 ["a", "b"]

-- Define a predicate to check if two monomials are of the same type
def sameType (m n : Monomial) : Prop :=
  m.vars = n.vars

theorem monomial_same_type :
  sameType m3 target := sorry

end NUMINAMATH_GPT_monomial_same_type_l94_9414


namespace NUMINAMATH_GPT_sin_theta_value_l94_9461

theorem sin_theta_value (θ : ℝ) (h₁ : θ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 :=
sorry

end NUMINAMATH_GPT_sin_theta_value_l94_9461


namespace NUMINAMATH_GPT_secant_length_problem_l94_9475

theorem secant_length_problem (tangent_length : ℝ) (internal_segment_length : ℝ) (external_segment_length : ℝ) 
    (h1 : tangent_length = 18) (h2 : internal_segment_length = 27) : external_segment_length = 9 :=
by
  sorry

end NUMINAMATH_GPT_secant_length_problem_l94_9475


namespace NUMINAMATH_GPT_carla_water_drank_l94_9466

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end NUMINAMATH_GPT_carla_water_drank_l94_9466


namespace NUMINAMATH_GPT_sandra_socks_l94_9451

variables (x y z : ℕ)

theorem sandra_socks :
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≤ 6 →
  y ≤ 6 →
  z ≤ 6 →
  x = 11 :=
by
  sorry

end NUMINAMATH_GPT_sandra_socks_l94_9451


namespace NUMINAMATH_GPT_product_correlation_function_l94_9442

open ProbabilityTheory

/-
Theorem: Given two centered and uncorrelated random functions \( \dot{X}(t) \) and \( \dot{Y}(t) \),
the correlation function of their product \( Z(t) = \dot{X}(t) \dot{Y}(t) \) is the product of their correlation functions.
-/
theorem product_correlation_function 
  (X Y : ℝ → ℝ)
  (hX_centered : ∀ t, (∫ x, X t ∂x) = 0) 
  (hY_centered : ∀ t, (∫ y, Y t ∂y) = 0)
  (h_uncorrelated : ∀ t1 t2, ∫ x, X t1 * Y t2 ∂x = (∫ x, X t1 ∂x) * (∫ y, Y t2 ∂y)) :
  ∀ t1 t2, 
  (∫ x, (X t1 * Y t1) * (X t2 * Y t2) ∂x) = 
  (∫ x, X t1 * X t2 ∂x) * (∫ y, Y t1 * Y t2 ∂y) :=
by
  sorry

end NUMINAMATH_GPT_product_correlation_function_l94_9442


namespace NUMINAMATH_GPT_xy_addition_l94_9441

theorem xy_addition (x y : ℕ) (h1 : x * y = 24) (h2 : x - y = 5) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 11 := 
sorry

end NUMINAMATH_GPT_xy_addition_l94_9441


namespace NUMINAMATH_GPT_num_distinct_remainders_of_prime_squared_mod_120_l94_9452

theorem num_distinct_remainders_of_prime_squared_mod_120:
  ∀ p : ℕ, Prime p → p > 5 → (p^2 % 120 = 1 ∨ p^2 % 120 = 49) := 
sorry

end NUMINAMATH_GPT_num_distinct_remainders_of_prime_squared_mod_120_l94_9452


namespace NUMINAMATH_GPT_mrs_awesome_class_l94_9449

def num_students (b g : ℕ) : ℕ := b + g

theorem mrs_awesome_class (b g : ℕ) (h1 : b = g + 3) (h2 : 480 - (b * b + g * g) = 5) : num_students b g = 31 :=
by
  sorry

end NUMINAMATH_GPT_mrs_awesome_class_l94_9449


namespace NUMINAMATH_GPT_triangle_cosine_rule_c_triangle_tangent_C_l94_9499

-- Define a proof statement for the cosine rule-based proof of c = 4.
theorem triangle_cosine_rule_c (a b : ℝ) (angleB : ℝ) (ha : a = 2)
                              (hb : b = 2 * Real.sqrt 3) (hB : angleB = π / 3) :
  ∃ (c : ℝ), c = 4 := by
  sorry

-- Define a proof statement for the tangent identity-based proof of tan C = 3 * sqrt 3 / 5.
theorem triangle_tangent_C (tanA : ℝ) (tanB : ℝ) (htA : tanA = 2 * Real.sqrt 3)
                           (htB : tanB = Real.sqrt 3) :
  ∃ (tanC : ℝ), tanC = 3 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_GPT_triangle_cosine_rule_c_triangle_tangent_C_l94_9499


namespace NUMINAMATH_GPT_power_product_rule_l94_9402

theorem power_product_rule (a : ℤ) : (-a^2)^3 = -a^6 := 
by 
  sorry

end NUMINAMATH_GPT_power_product_rule_l94_9402


namespace NUMINAMATH_GPT_unique_n0_exists_l94_9455

open Set

theorem unique_n0_exists 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h3 : ∀ n : ℕ, S 0 = a 0) :
  ∃! n_0 : ℕ, (S (n_0 + 1)) / n_0 > a (n_0 + 1)
             ∧ (S (n_0 + 1)) / n_0 ≤ a (n_0 + 2) := 
sorry

end NUMINAMATH_GPT_unique_n0_exists_l94_9455


namespace NUMINAMATH_GPT_ram_balance_speed_l94_9435

theorem ram_balance_speed
  (part_speed : ℝ)
  (balance_distance : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (part_time : ℝ)
  (balance_speed : ℝ)
  (h1 : part_speed = 20)
  (h2 : total_distance = 400)
  (h3 : total_time = 8)
  (h4 : part_time = 3.2)
  (h5 : balance_distance = total_distance - part_speed * part_time)
  (h6 : balance_speed = balance_distance / (total_time - part_time)) :
  balance_speed = 70 :=
by
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_ram_balance_speed_l94_9435


namespace NUMINAMATH_GPT_more_children_got_off_than_got_on_l94_9438

-- Define the initial number of children on the bus
def initial_children : ℕ := 36

-- Define the number of children who got off the bus
def children_got_off : ℕ := 68

-- Define the total number of children on the bus after changes
def final_children : ℕ := 12

-- Define the unknown number of children who got on the bus
def children_got_on : ℕ := sorry -- We will use the conditions to solve for this in the proof

-- The main proof statement
theorem more_children_got_off_than_got_on : (children_got_off - children_got_on = 24) :=
by
  -- Write the equation describing the total number of children after changes
  have h1 : initial_children - children_got_off + children_got_on = final_children := sorry
  -- Solve for the number of children who got on the bus (children_got_on)
  have h2 : children_got_on = final_children + (children_got_off - initial_children) := sorry
  -- Substitute to find the required difference
  have h3 : children_got_off - final_children - (children_got_off - initial_children) = 24 := sorry
  -- Conclude the proof
  exact sorry


end NUMINAMATH_GPT_more_children_got_off_than_got_on_l94_9438


namespace NUMINAMATH_GPT_number_is_correct_l94_9405

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_number_is_correct_l94_9405


namespace NUMINAMATH_GPT_find_number_l94_9448

theorem find_number (x : ℝ) (h : 0.30 * x = 90 + 120) : x = 700 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l94_9448


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l94_9421
open Classical

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 > 0) → ∃ x : ℝ, ¬(x^2 > 0) :=
by
  intro h
  have := (not_forall.mp h)
  exact this

end NUMINAMATH_GPT_negation_of_universal_proposition_l94_9421


namespace NUMINAMATH_GPT_mother_sold_rings_correct_l94_9425

noncomputable def motherSellsRings (initial_bought_rings mother_bought_rings remaining_rings final_stock : ℤ) : ℤ :=
  let initial_stock := initial_bought_rings / 2
  let total_stock := initial_bought_rings + initial_stock
  let sold_by_eliza := (3 * total_stock) / 4
  let remaining_after_eliza := total_stock - sold_by_eliza
  let new_total_stock := remaining_after_eliza + mother_bought_rings
  new_total_stock - final_stock

theorem mother_sold_rings_correct :
  motherSellsRings 200 300 225 300 = 150 :=
by
  sorry

end NUMINAMATH_GPT_mother_sold_rings_correct_l94_9425


namespace NUMINAMATH_GPT_Toby_second_part_distance_l94_9459

noncomputable def total_time_journey (distance_unloaded_second: ℝ) : ℝ :=
  18 + (distance_unloaded_second / 20) + 8 + 7

theorem Toby_second_part_distance:
  ∃ d : ℝ, total_time_journey d = 39 ∧ d = 120 :=
by
  use 120
  unfold total_time_journey
  sorry

end NUMINAMATH_GPT_Toby_second_part_distance_l94_9459


namespace NUMINAMATH_GPT_triangle_external_angle_properties_l94_9479

theorem triangle_external_angle_properties (A B C : ℝ) (hA : 0 < A ∧ A < 180) (hB : 0 < B ∧ B < 180) (hC : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) :
  (∃ E1 E2 E3, E1 + E2 + E3 = 360 ∧ E1 > 90 ∧ E2 > 90 ∧ E3 <= 90) :=
by
  sorry

end NUMINAMATH_GPT_triangle_external_angle_properties_l94_9479


namespace NUMINAMATH_GPT_range_of_a_l94_9462

variable (a : ℝ)

theorem range_of_a
  (h : ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) :
  a < -1 ∨ a > 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l94_9462


namespace NUMINAMATH_GPT_fraction_of_fritz_money_l94_9450

theorem fraction_of_fritz_money
  (Fritz_money : ℕ)
  (total_amount : ℕ)
  (fraction : ℚ)
  (Sean_money : ℚ)
  (Rick_money : ℚ)
  (h1 : Fritz_money = 40)
  (h2 : total_amount = 96)
  (h3 : Sean_money = fraction * Fritz_money + 4)
  (h4 : Rick_money = 3 * Sean_money)
  (h5 : Rick_money + Sean_money = total_amount) :
  fraction = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_fritz_money_l94_9450


namespace NUMINAMATH_GPT_kyunghwan_spent_the_most_l94_9413

-- Define initial pocket money for everyone
def initial_money : ℕ := 20000

-- Define remaining money
def remaining_S : ℕ := initial_money / 4
def remaining_K : ℕ := initial_money / 8
def remaining_D : ℕ := initial_money / 5

-- Calculate spent money
def spent_S : ℕ := initial_money - remaining_S
def spent_K : ℕ := initial_money - remaining_K
def spent_D : ℕ := initial_money - remaining_D

theorem kyunghwan_spent_the_most 
  (h1 : remaining_S = initial_money / 4)
  (h2 : remaining_K = initial_money / 8)
  (h3 : remaining_D = initial_money / 5) :
  spent_K > spent_S ∧ spent_K > spent_D :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_kyunghwan_spent_the_most_l94_9413


namespace NUMINAMATH_GPT_ral_current_age_l94_9484

-- Definitions according to the conditions
def ral_three_times_suri (ral suri : ℕ) : Prop := ral = 3 * suri
def suri_in_6_years (suri : ℕ) : Prop := suri + 6 = 25

-- The proof problem statement
theorem ral_current_age (ral suri : ℕ) (h1 : ral_three_times_suri ral suri) (h2 : suri_in_6_years suri) : ral = 57 :=
by sorry

end NUMINAMATH_GPT_ral_current_age_l94_9484


namespace NUMINAMATH_GPT_dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l94_9476

-- Define a regular dodecagon
def dodecagon_sides : ℕ := 12

-- Prove that the number of diagonals in a regular dodecagon is 54
theorem dodecagon_diagonals_eq_54 : (dodecagon_sides * (dodecagon_sides - 3)) / 2 = 54 :=
by sorry

-- Prove that the number of possible triangles formed from a regular dodecagon vertices is 220
theorem dodecagon_triangles_eq_220 : Nat.choose dodecagon_sides 3 = 220 :=
by sorry

end NUMINAMATH_GPT_dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l94_9476


namespace NUMINAMATH_GPT_remaining_distance_l94_9439

-- Definitions for the given conditions
def total_distance : ℕ := 436
def first_stopover_distance : ℕ := 132
def second_stopover_distance : ℕ := 236

-- Prove that the remaining distance from the second stopover to the island is 68 miles.
theorem remaining_distance : total_distance - (first_stopover_distance + second_stopover_distance) = 68 := by
  -- The proof (details) will go here
  sorry

end NUMINAMATH_GPT_remaining_distance_l94_9439


namespace NUMINAMATH_GPT_sum_of_solutions_l94_9412

theorem sum_of_solutions (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := 
by {
  -- missing proof part
  sorry
}

end NUMINAMATH_GPT_sum_of_solutions_l94_9412


namespace NUMINAMATH_GPT_yura_picture_dimensions_l94_9497

theorem yura_picture_dimensions (l w : ℕ) (h_frame : (l + 2) * (w + 2) - l * w = l * w) :
    (l = 3 ∧ w = 10) ∨ (l = 4 ∧ w = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_yura_picture_dimensions_l94_9497


namespace NUMINAMATH_GPT_find_n_l94_9447

theorem find_n (x n : ℝ) (h_x : x = 0.5) : (9 / (1 + n / x) = 1) → n = 4 := 
by
  intro h
  have h_x_eq : x = 0.5 := h_x
  -- Proof content here covering the intermediary steps
  sorry

end NUMINAMATH_GPT_find_n_l94_9447


namespace NUMINAMATH_GPT_f_is_odd_l94_9433

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 2) * x ^ α

theorem f_is_odd (α : ℝ) (hα : α = 3) : ∀ x : ℝ, f α (-x) = -f α x :=
by sorry

end NUMINAMATH_GPT_f_is_odd_l94_9433


namespace NUMINAMATH_GPT_residue_of_neg_1237_mod_37_l94_9418

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_residue_of_neg_1237_mod_37_l94_9418


namespace NUMINAMATH_GPT_total_age_of_siblings_l94_9495

def age_total (Susan Arthur Tom Bob : ℕ) : ℕ := Susan + Arthur + Tom + Bob

theorem total_age_of_siblings :
  ∀ (Susan Arthur Tom Bob : ℕ),
    (Arthur = Susan + 2) →
    (Tom = Bob - 3) →
    (Bob = 11) →
    (Susan = 15) →
    age_total Susan Arthur Tom Bob = 51 :=
by
  intros Susan Arthur Tom Bob h1 h2 h3 h4
  rw [h4, h1, h3, h2]    -- Use the conditions
  norm_num               -- Simplify numerical expressions
  sorry                  -- Placeholder for the proof

end NUMINAMATH_GPT_total_age_of_siblings_l94_9495


namespace NUMINAMATH_GPT_dilation_translation_correct_l94_9430

def transformation_matrix (d: ℝ) (tx: ℝ) (ty: ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![d, 0, tx],
    ![0, d, ty],
    ![0, 0, 1]
  ]

theorem dilation_translation_correct :
  transformation_matrix 4 2 3 =
  ![
    ![4, 0, 2],
    ![0, 4, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end NUMINAMATH_GPT_dilation_translation_correct_l94_9430


namespace NUMINAMATH_GPT_area_of_trapezium_is_105_l94_9406

-- Define points in the coordinate plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨14, 3⟩
def C : Point := ⟨18, 10⟩
def D : Point := ⟨0, 10⟩

noncomputable def length (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)
noncomputable def height (p1 p2 : Point) : ℝ := abs (p2.y - p1.y)

-- Calculate lengths of parallel sides AB and CD, and height
noncomputable def AB := length A B
noncomputable def CD := length C D
noncomputable def heightAC := height A C

-- Define the area of trapezium
noncomputable def area_trapezium (AB CD height : ℝ) : ℝ := (1/2) * (AB + CD) * height

-- The proof problem statement
theorem area_of_trapezium_is_105 :
  area_trapezium AB CD heightAC = 105 := by
  sorry

end NUMINAMATH_GPT_area_of_trapezium_is_105_l94_9406


namespace NUMINAMATH_GPT_problem_statement_l94_9469

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2004 = 2005 :=
sorry

end NUMINAMATH_GPT_problem_statement_l94_9469


namespace NUMINAMATH_GPT_solve_inequality_l94_9483

theorem solve_inequality {a x : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (x : ℝ), (a > 1 ∧ (a^(2/3) ≤ x ∧ x < a^(3/4) ∨ x > a)) ∨ (0 < a ∧ a < 1 ∧ (a^(3/4) < x ∧ x ≤ a^(2/3) ∨ 0 < x ∧ x < a))) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l94_9483


namespace NUMINAMATH_GPT_min_2a_b_c_l94_9489

theorem min_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * b * c = 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 := sorry

end NUMINAMATH_GPT_min_2a_b_c_l94_9489


namespace NUMINAMATH_GPT_josie_total_animals_is_correct_l94_9493

noncomputable def totalAnimals : Nat :=
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

theorem josie_total_animals_is_correct : totalAnimals = 1308 := by
  sorry

end NUMINAMATH_GPT_josie_total_animals_is_correct_l94_9493


namespace NUMINAMATH_GPT_income_increase_percentage_l94_9440

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end NUMINAMATH_GPT_income_increase_percentage_l94_9440


namespace NUMINAMATH_GPT_arrangement_of_digits_11250_l94_9408

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_of_digits_11250_l94_9408


namespace NUMINAMATH_GPT_power_expression_l94_9436

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end NUMINAMATH_GPT_power_expression_l94_9436


namespace NUMINAMATH_GPT_number_of_ways_to_arrange_matches_l94_9496

open Nat

theorem number_of_ways_to_arrange_matches :
  (factorial 7) * (2 ^ 3) = 40320 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_arrange_matches_l94_9496


namespace NUMINAMATH_GPT_shortest_side_of_right_triangle_l94_9457

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  ∀ c, (c = 5 ∨ c = 12 ∨ c = (Real.sqrt (a^2 + b^2))) → c = 5 :=
by
  intros c h
  sorry

end NUMINAMATH_GPT_shortest_side_of_right_triangle_l94_9457


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l94_9443

-- Define the function based on the given equation
noncomputable def f (x : ℝ) : ℝ := (4 * x - 16) / (3 * x - 4)

-- Statement to be proved
theorem greatest_possible_value_of_x : 
  (∀ x : ℝ, (f x)^2 + (f x) = 20) → 
  ∃ x : ℝ, (f x)^2 + (f x) = 20 ∧ x = 36 / 19 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l94_9443


namespace NUMINAMATH_GPT_necessarily_positive_y_plus_z_l94_9473

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end NUMINAMATH_GPT_necessarily_positive_y_plus_z_l94_9473


namespace NUMINAMATH_GPT_evaluate_expression_l94_9432

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l94_9432


namespace NUMINAMATH_GPT_perpendicular_tangent_line_exists_and_correct_l94_9434

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end NUMINAMATH_GPT_perpendicular_tangent_line_exists_and_correct_l94_9434


namespace NUMINAMATH_GPT_evaluate_expression_l94_9410

theorem evaluate_expression : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l94_9410


namespace NUMINAMATH_GPT_larger_number_l94_9470

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end NUMINAMATH_GPT_larger_number_l94_9470


namespace NUMINAMATH_GPT_least_multiple_36_sum_digits_l94_9460

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_multiple_36_sum_digits :
  ∃ n : ℕ, n = 36 ∧ (36 ∣ n) ∧ (9 ∣ digit_sum n) ∧ (∀ m : ℕ, (36 ∣ m) ∧ (9 ∣ digit_sum m) → 36 ≤ m) :=
by sorry

end NUMINAMATH_GPT_least_multiple_36_sum_digits_l94_9460


namespace NUMINAMATH_GPT_manager_salary_l94_9498

theorem manager_salary
    (average_salary_employees : ℝ)
    (num_employees : ℕ)
    (increase_in_average_due_to_manager : ℝ)
    (total_salary_20_employees : ℝ)
    (new_average_salary : ℝ)
    (total_salary_with_manager : ℝ) :
  average_salary_employees = 1300 →
  num_employees = 20 →
  increase_in_average_due_to_manager = 100 →
  total_salary_20_employees = average_salary_employees * num_employees →
  new_average_salary = average_salary_employees + increase_in_average_due_to_manager →
  total_salary_with_manager = new_average_salary * (num_employees + 1) →
  total_salary_with_manager - total_salary_20_employees = 3400 :=
by 
  sorry

end NUMINAMATH_GPT_manager_salary_l94_9498


namespace NUMINAMATH_GPT_find_ab_integer_l94_9467

theorem find_ab_integer (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ b) :
    ∃ n : ℤ, (a^b + b^a) = n * (a^a - b^b) ↔ (a = 2 ∧ b = 1) ∨ (a = 1 ∧ b = 2) := 
sorry

end NUMINAMATH_GPT_find_ab_integer_l94_9467


namespace NUMINAMATH_GPT_find_13th_result_l94_9423

theorem find_13th_result (avg25 : ℕ) (avg12_first : ℕ) (avg12_last : ℕ)
  (h_avg25 : avg25 = 18) (h_avg12_first : avg12_first = 10) (h_avg12_last : avg12_last = 20) :
  ∃ r13 : ℕ, r13 = 90 := by
  sorry

end NUMINAMATH_GPT_find_13th_result_l94_9423


namespace NUMINAMATH_GPT_smallest_x_y_sum_299_l94_9424

theorem smallest_x_y_sum_299 : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x < y ∧ (100 + (x / y : ℚ) = 2 * (100 * x / y : ℚ)) ∧ (x + y = 299) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_y_sum_299_l94_9424


namespace NUMINAMATH_GPT_ratio_a_over_3_to_b_over_2_l94_9486

theorem ratio_a_over_3_to_b_over_2 (a b c : ℝ) (h1 : 2 * a = 3 * b) (h2 : c ≠ 0) (h3 : 3 * a + 2 * b = c) :
  (a / 3) / (b / 2) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_a_over_3_to_b_over_2_l94_9486


namespace NUMINAMATH_GPT_quadratic_roots_r12_s12_l94_9481

theorem quadratic_roots_r12_s12 (r s : ℝ) (h1 : r + s = 2 * Real.sqrt 3) (h2 : r * s = 1) :
  r^12 + s^12 = 940802 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_r12_s12_l94_9481


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l94_9478

theorem isosceles_triangle_base_angle (A B C : ℝ) (h_sum : A + B + C = 180) (h_iso : B = C) (h_one_angle : A = 80) : B = 50 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l94_9478


namespace NUMINAMATH_GPT_initial_necklaces_15_l94_9429

variable (N E : ℕ)
variable (initial_necklaces : ℕ) (initial_earrings : ℕ) (store_necklaces : ℕ) (store_earrings : ℕ) (mother_earrings : ℕ) (total_jewelry : ℕ)

axiom necklaces_eq_initial : N = initial_necklaces
axiom earrings_eq_15 : E = initial_earrings
axiom initial_earrings_15 : initial_earrings = 15
axiom store_necklaces_eq_initial : store_necklaces = initial_necklaces
axiom store_earrings_eq_23_initial : store_earrings = 2 * initial_earrings / 3
axiom mother_earrings_eq_115_store : mother_earrings = 1 * store_earrings / 5
axiom total_jewelry_is_57 : total_jewelry = 57
axiom jewelry_pieces_eq : 2 * initial_necklaces + initial_earrings + store_earrings + mother_earrings = total_jewelry

theorem initial_necklaces_15 : initial_necklaces = 15 := by
  sorry

end NUMINAMATH_GPT_initial_necklaces_15_l94_9429


namespace NUMINAMATH_GPT_asthma_distribution_l94_9485

noncomputable def total_children := 490
noncomputable def boys := 280
noncomputable def general_asthma_ratio := 2 / 7
noncomputable def boys_asthma_ratio := 1 / 9

noncomputable def total_children_with_asthma := general_asthma_ratio * total_children
noncomputable def boys_with_asthma := boys_asthma_ratio * boys
noncomputable def girls_with_asthma := total_children_with_asthma - boys_with_asthma

theorem asthma_distribution
  (h_general_asthma: general_asthma_ratio = 2 / 7)
  (h_total_children: total_children = 490)
  (h_boys: boys = 280)
  (h_boys_asthma: boys_asthma_ratio = 1 / 9):
  boys_with_asthma = 31 ∧ girls_with_asthma = 109 :=
by
  sorry

end NUMINAMATH_GPT_asthma_distribution_l94_9485


namespace NUMINAMATH_GPT_find_base_l94_9409

theorem find_base (r : ℕ) (h1 : 5 * r^2 + 3 * r + 4 + 3 * r^2 + 6 * r + 6 = r^3) : r = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_base_l94_9409


namespace NUMINAMATH_GPT_sufficient_drivers_and_correct_time_l94_9445

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end NUMINAMATH_GPT_sufficient_drivers_and_correct_time_l94_9445


namespace NUMINAMATH_GPT_value_of_expression_l94_9403

theorem value_of_expression (x : ℝ) (h : x ^ 2 - 3 * x + 1 = 0) : 
  x ≠ 0 → (x ^ 2) / (x ^ 4 + x ^ 2 + 1) = 1 / 8 :=
by 
  intros h1 
  sorry

end NUMINAMATH_GPT_value_of_expression_l94_9403


namespace NUMINAMATH_GPT_solve_inequality_group_l94_9428

theorem solve_inequality_group (x : ℝ) (h1 : -9 < 2 * x - 1) (h2 : 2 * x - 1 ≤ 6) :
  -4 < x ∧ x ≤ 3.5 := 
sorry

end NUMINAMATH_GPT_solve_inequality_group_l94_9428


namespace NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l94_9465

theorem businessmen_neither_coffee_nor_tea :
  ∀ (total_count coffee tea both neither : ℕ),
    total_count = 30 →
    coffee = 15 →
    tea = 13 →
    both = 6 →
    neither = total_count - (coffee + tea - both) →
    neither = 8 := 
by
  intros total_count coffee tea both neither ht hc ht2 hb hn
  rw [ht, hc, ht2, hb] at hn
  simp at hn
  exact hn

end NUMINAMATH_GPT_businessmen_neither_coffee_nor_tea_l94_9465


namespace NUMINAMATH_GPT_polynomial_value_l94_9453

theorem polynomial_value (x : ℝ) :
  let a := 2009 * x + 2008
  let b := 2009 * x + 2009
  let c := 2009 * x + 2010
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 3 := by
  sorry

end NUMINAMATH_GPT_polynomial_value_l94_9453


namespace NUMINAMATH_GPT_logarithmic_function_through_point_l94_9491

noncomputable def log_function_expression (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem logarithmic_function_through_point (f : ℝ → ℝ) :
  (∀ x a : ℝ, a > 0 ∧ a ≠ 1 → f x = log_function_expression a x) ∧ f 4 = 2 →
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g x = log_function_expression 2 x :=
by {
  sorry
}

end NUMINAMATH_GPT_logarithmic_function_through_point_l94_9491


namespace NUMINAMATH_GPT_find_twentieth_special_number_l94_9482

theorem find_twentieth_special_number :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 5 [MOD 8]) ∧ (∀ k < 20, ∃ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 5 [MOD 8]) ∧ m < n) ∧ (n = 461) := 
sorry

end NUMINAMATH_GPT_find_twentieth_special_number_l94_9482


namespace NUMINAMATH_GPT_units_digit_2_pow_2015_minus_1_l94_9415

theorem units_digit_2_pow_2015_minus_1 : (2^2015 - 1) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_2_pow_2015_minus_1_l94_9415


namespace NUMINAMATH_GPT_fraction_ratio_l94_9422

theorem fraction_ratio (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_ratio_l94_9422


namespace NUMINAMATH_GPT_proper_divisors_increased_by_one_l94_9492

theorem proper_divisors_increased_by_one
  (n : ℕ)
  (hn1 : 2 ≤ n)
  (exists_m : ∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ d + 1 ≠ m)
  : n = 4 ∨ n = 8 :=
  sorry

end NUMINAMATH_GPT_proper_divisors_increased_by_one_l94_9492


namespace NUMINAMATH_GPT_three_digit_problem_l94_9454

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_problem_l94_9454


namespace NUMINAMATH_GPT_calculate_sum_and_double_l94_9444

theorem calculate_sum_and_double :
  2 * (1324 + 4231 + 3124 + 2413) = 22184 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sum_and_double_l94_9444


namespace NUMINAMATH_GPT_remainder_div_power10_l94_9468

theorem remainder_div_power10 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, (10^n - 1) % 37 = k^2 := by
  sorry

end NUMINAMATH_GPT_remainder_div_power10_l94_9468
