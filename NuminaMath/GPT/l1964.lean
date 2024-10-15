import Mathlib

namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l1964_196400

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℤ, (∀ m : ℤ, m * m ≤ 500 → m < n) ∧ n = 23 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l1964_196400


namespace NUMINAMATH_GPT_simplify_expression_l1964_196448

theorem simplify_expression (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1964_196448


namespace NUMINAMATH_GPT_integer_subset_property_l1964_196466

theorem integer_subset_property (M : Set ℤ) (h1 : ∃ a ∈ M, a > 0) (h2 : ∃ b ∈ M, b < 0)
(h3 : ∀ {a b : ℤ}, a ∈ M → b ∈ M → 2 * a ∈ M ∧ a + b ∈ M)
: ∀ a b : ℤ, a ∈ M → b ∈ M → a - b ∈ M :=
by
  sorry

end NUMINAMATH_GPT_integer_subset_property_l1964_196466


namespace NUMINAMATH_GPT_john_buys_spools_l1964_196476

theorem john_buys_spools (spool_length necklace_length : ℕ) 
  (necklaces : ℕ) 
  (total_length := necklaces * necklace_length) 
  (spools := total_length / spool_length) :
  spool_length = 20 → 
  necklace_length = 4 → 
  necklaces = 15 → 
  spools = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_john_buys_spools_l1964_196476


namespace NUMINAMATH_GPT_rewrite_equation_to_function_l1964_196457

theorem rewrite_equation_to_function (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_rewrite_equation_to_function_l1964_196457


namespace NUMINAMATH_GPT_toby_steps_needed_l1964_196473

noncomputable def total_steps_needed : ℕ := 10000 * 9

noncomputable def first_sunday_steps : ℕ := 10200
noncomputable def first_monday_steps : ℕ := 10400
noncomputable def tuesday_steps : ℕ := 9400
noncomputable def wednesday_steps : ℕ := 9100
noncomputable def thursday_steps : ℕ := 8300
noncomputable def friday_steps : ℕ := 9200
noncomputable def saturday_steps : ℕ := 8900
noncomputable def second_sunday_steps : ℕ := 9500

noncomputable def total_steps_walked := 
  first_sunday_steps + 
  first_monday_steps + 
  tuesday_steps + 
  wednesday_steps + 
  thursday_steps + 
  friday_steps + 
  saturday_steps + 
  second_sunday_steps

noncomputable def remaining_steps_needed := total_steps_needed - total_steps_walked

noncomputable def days_left : ℕ := 3

noncomputable def average_steps_needed := remaining_steps_needed / days_left

theorem toby_steps_needed : average_steps_needed = 5000 := by
  sorry

end NUMINAMATH_GPT_toby_steps_needed_l1964_196473


namespace NUMINAMATH_GPT_convex_polygon_sides_l1964_196460

theorem convex_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) - 90 = 2790) : n = 18 :=
sorry

end NUMINAMATH_GPT_convex_polygon_sides_l1964_196460


namespace NUMINAMATH_GPT_integer_part_divisible_by_112_l1964_196443

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem integer_part_divisible_by_112
  (m : ℕ) (hm_pos : 0 < m) (hm_odd : is_odd m) (hm_not_div3 : not_divisible_by_3 m) :
  ∃ n : ℤ, 112 * n = 4^m - (2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m :=
by
  sorry

end NUMINAMATH_GPT_integer_part_divisible_by_112_l1964_196443


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_x_gt_2_l1964_196428

theorem necessary_but_not_sufficient_condition_for_x_gt_2 :
  ∀ (x : ℝ), (2 / x < 1 → x > 2) ∧ (x > 2 → 2 / x < 1) → (¬ (x > 2 → 2 / x < 1) ∨ ¬ (2 / x < 1 → x > 2)) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_x_gt_2_l1964_196428


namespace NUMINAMATH_GPT_solve_for_B_l1964_196415

theorem solve_for_B (B : ℕ) (h : 3 * B + 2 = 20) : B = 6 :=
by 
  -- This is just a placeholder, the proof will go here
  sorry

end NUMINAMATH_GPT_solve_for_B_l1964_196415


namespace NUMINAMATH_GPT_total_bags_l1964_196425

-- Definitions based on the conditions
def bags_on_monday : ℕ := 4
def bags_next_day : ℕ := 8

-- Theorem statement
theorem total_bags : bags_on_monday + bags_next_day = 12 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_total_bags_l1964_196425


namespace NUMINAMATH_GPT_brothers_work_rate_l1964_196495

theorem brothers_work_rate (A B C : ℝ) :
  (1 / A + 1 / B = 1 / 8) ∧ (1 / A + 1 / C = 1 / 9) ∧ (1 / B + 1 / C = 1 / 10) →
  A = 160 / 19 ∧ B = 160 / 9 ∧ C = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_brothers_work_rate_l1964_196495


namespace NUMINAMATH_GPT_total_spent_proof_l1964_196478

noncomputable def total_spent (cost_pen cost_pencil cost_notebook : ℝ) 
  (pens_robert pencils_robert notebooks_dorothy : ℕ) 
  (julia_pens_ratio robert_pens_ratio dorothy_pens_ratio : ℝ) 
  (julia_pencils_diff notebooks_julia_diff : ℕ) 
  (robert_notebooks_ratio dorothy_pencils_ratio : ℝ) : ℝ :=
    let pens_julia := robert_pens_ratio * pens_robert
    let pens_dorothy := dorothy_pens_ratio * pens_julia
    let total_pens := pens_robert + pens_julia + pens_dorothy
    let cost_pens := total_pens * cost_pen 
    
    let pencils_julia := pencils_robert - julia_pencils_diff
    let pencils_dorothy := dorothy_pencils_ratio * pencils_julia
    let total_pencils := pencils_robert + pencils_julia + pencils_dorothy
    let cost_pencils := total_pencils * cost_pencil 
        
    let notebooks_julia := notebooks_dorothy + notebooks_julia_diff
    let notebooks_robert := robert_notebooks_ratio * notebooks_julia
    let total_notebooks := notebooks_dorothy + notebooks_julia + notebooks_robert
    let cost_notebooks := total_notebooks * cost_notebook
        
    cost_pens + cost_pencils + cost_notebooks

theorem total_spent_proof 
  (cost_pen : ℝ := 1.50)
  (cost_pencil : ℝ := 0.75)
  (cost_notebook : ℝ := 4.00)
  (pens_robert : ℕ := 4)
  (pencils_robert : ℕ := 12)
  (notebooks_dorothy : ℕ := 3)
  (julia_pens_ratio : ℝ := 3)
  (robert_pens_ratio : ℝ := 3)
  (dorothy_pens_ratio : ℝ := 0.5)
  (julia_pencils_diff : ℕ := 5)
  (notebooks_julia_diff : ℕ := 1)
  (robert_notebooks_ratio : ℝ := 0.5)
  (dorothy_pencils_ratio : ℝ := 2) : 
  total_spent cost_pen cost_pencil cost_notebook pens_robert pencils_robert notebooks_dorothy 
    julia_pens_ratio robert_pens_ratio dorothy_pens_ratio julia_pencils_diff notebooks_julia_diff robert_notebooks_ratio dorothy_pencils_ratio 
    = 93.75 := 
by 
  sorry

end NUMINAMATH_GPT_total_spent_proof_l1964_196478


namespace NUMINAMATH_GPT_find_k_l1964_196465

theorem find_k (k : ℕ) (h : (64 : ℕ) / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_GPT_find_k_l1964_196465


namespace NUMINAMATH_GPT_find_prime_p_l1964_196483

noncomputable def concatenate (q r : ℕ) : ℕ :=
q * 10 ^ (r.digits 10).length + r

theorem find_prime_p (q r p : ℕ) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp : Nat.Prime p)
  (h : concatenate q r + 3 = p^2) : p = 5 :=
sorry

end NUMINAMATH_GPT_find_prime_p_l1964_196483


namespace NUMINAMATH_GPT_sum_of_f_values_l1964_196494

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f_values :
  (∀ x : ℝ, f x + f (-x) = 0) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = 2^x - 1) →
  f (1/2) + f 1 + f (3/2) + f 2 + f (5/2) = Real.sqrt 2 - 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_f_values_l1964_196494


namespace NUMINAMATH_GPT_cosine_of_acute_angle_l1964_196489

theorem cosine_of_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) : Real.cos α = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cosine_of_acute_angle_l1964_196489


namespace NUMINAMATH_GPT_tangent_subtraction_identity_l1964_196407

theorem tangent_subtraction_identity (α β : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : Real.tan (Real.pi - β) = 1/2) : 
  Real.tan (α - β) = -2/11 := 
sorry

end NUMINAMATH_GPT_tangent_subtraction_identity_l1964_196407


namespace NUMINAMATH_GPT_sqrt_quartic_equiv_l1964_196453

-- Define x as a positive real number
variable (x : ℝ)
variable (hx : 0 < x)

-- Statement of the problem to prove
theorem sqrt_quartic_equiv (x : ℝ) (hx : 0 < x) : (x^2 * x^(1/2))^(1/4) = x^(5/8) :=
sorry

end NUMINAMATH_GPT_sqrt_quartic_equiv_l1964_196453


namespace NUMINAMATH_GPT_average_problem_l1964_196408

theorem average_problem
  (h : (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_problem_l1964_196408


namespace NUMINAMATH_GPT_N_positive_l1964_196436

def N (a b : ℝ) : ℝ :=
  4 * a^2 - 12 * a * b + 13 * b^2 - 6 * a + 4 * b + 13

theorem N_positive (a b : ℝ) : N a b > 0 :=
by
  sorry

end NUMINAMATH_GPT_N_positive_l1964_196436


namespace NUMINAMATH_GPT_shaded_region_occupies_32_percent_of_total_area_l1964_196480

-- Conditions
def angle_sector := 90
def r_small := 1
def r_large := 3
def r_sector := 4

-- Question: Prove the shaded region occupies 32% of the total area given the conditions
theorem shaded_region_occupies_32_percent_of_total_area :
  let area_large_sector := (1 / 4) * Real.pi * (r_sector ^ 2)
  let area_small_sector := (1 / 4) * Real.pi * (r_large ^ 2)
  let total_area := area_large_sector + area_small_sector
  let shaded_area := (1 / 4) * Real.pi * (r_large ^ 2) - (1 / 4) * Real.pi * (r_small ^ 2)
  let shaded_percent := (shaded_area / total_area) * 100
  shaded_percent = 32 := by
  sorry

end NUMINAMATH_GPT_shaded_region_occupies_32_percent_of_total_area_l1964_196480


namespace NUMINAMATH_GPT_dawns_earnings_per_hour_l1964_196490

variable (hours_per_painting : ℕ) (num_paintings : ℕ) (total_earnings : ℕ)

def total_hours (hours_per_painting num_paintings : ℕ) : ℕ :=
  hours_per_painting * num_paintings

def earnings_per_hour (total_earnings total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem dawns_earnings_per_hour :
  hours_per_painting = 2 →
  num_paintings = 12 →
  total_earnings = 3600 →
  earnings_per_hour total_earnings (total_hours hours_per_painting num_paintings) = 150 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_dawns_earnings_per_hour_l1964_196490


namespace NUMINAMATH_GPT_maria_money_difference_l1964_196419

-- Defining constants for Maria's money when she arrived and left the fair
def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

-- Calculating the expected difference
def expected_difference : ℕ := 71

-- Statement: proving that the difference between money_at_arrival and money_at_departure is expected_difference
theorem maria_money_difference : money_at_arrival - money_at_departure = expected_difference := by
  sorry

end NUMINAMATH_GPT_maria_money_difference_l1964_196419


namespace NUMINAMATH_GPT_sqrt_108_eq_6_sqrt_3_l1964_196442

theorem sqrt_108_eq_6_sqrt_3 : Real.sqrt 108 = 6 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_sqrt_108_eq_6_sqrt_3_l1964_196442


namespace NUMINAMATH_GPT_find_a10_l1964_196455

theorem find_a10 (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a n - a (n+1) = a n * a (n+1)) : 
  a 10 = 1 / 10 :=
sorry

end NUMINAMATH_GPT_find_a10_l1964_196455


namespace NUMINAMATH_GPT_octagon_diagonals_l1964_196418

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end NUMINAMATH_GPT_octagon_diagonals_l1964_196418


namespace NUMINAMATH_GPT_price_decrease_l1964_196426

theorem price_decrease (current_price original_price : ℝ) (h1 : current_price = 684) (h2 : original_price = 900) :
  ((original_price - current_price) / original_price) * 100 = 24 :=
by
  sorry

end NUMINAMATH_GPT_price_decrease_l1964_196426


namespace NUMINAMATH_GPT_certain_number_divided_by_10_l1964_196434
-- Broad import to bring in necessary libraries

-- Define the constants and hypotheses
variable (x : ℝ)
axiom condition : 5 * x = 100

-- Theorem to prove the required equality
theorem certain_number_divided_by_10 : (x / 10) = 2 :=
by
  -- The proof is skipped by sorry
  sorry

end NUMINAMATH_GPT_certain_number_divided_by_10_l1964_196434


namespace NUMINAMATH_GPT_gamma_lt_delta_l1964_196499

open Real

variables (α β γ δ : ℝ)

-- Hypotheses as given in the problem
axiom h1 : 0 < α 
axiom h2 : α < β
axiom h3 : β < π / 2
axiom hg1 : 0 < γ
axiom hg2 : γ < π / 2
axiom htan_gamma_eq : tan γ = (tan α + tan β) / 2
axiom hd1 : 0 < δ
axiom hd2 : δ < π / 2
axiom hcos_delta_eq : (1 / cos δ) = (1 / 2) * (1 / cos α + 1 / cos β)

-- Goal to prove
theorem gamma_lt_delta : γ < δ := 
by 
sorry

end NUMINAMATH_GPT_gamma_lt_delta_l1964_196499


namespace NUMINAMATH_GPT_find_angle_C_max_area_triangle_l1964_196491

-- Part I: Proving angle C
theorem find_angle_C (a b c : ℝ) (A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
    C = Real.pi / 3 :=
sorry

-- Part II: Finding maximum area of triangle ABC
theorem max_area_triangle (a b : ℝ) (c : ℝ) (h_c : c = 2 * Real.sqrt 3) (A B C : ℝ)
    (h_A : A > 0) (h_B : B > 0) (h_C : C = Real.pi / 3)
    (h : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
    0.5 * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angle_C_max_area_triangle_l1964_196491


namespace NUMINAMATH_GPT_identity_element_exists_identity_element_self_commutativity_associativity_l1964_196467

noncomputable def star_op (a b : ℤ) : ℤ := a + b + a * b

theorem identity_element_exists : ∃ E : ℤ, ∀ a : ℤ, star_op a E = a :=
by sorry

theorem identity_element_self (E : ℤ) (h1 : ∀ a : ℤ, star_op a E = a) : star_op E E = E :=
by sorry

theorem commutativity (a b : ℤ) : star_op a b = star_op b a :=
by sorry

theorem associativity (a b c : ℤ) : star_op (star_op a b) c = star_op a (star_op b c) :=
by sorry

end NUMINAMATH_GPT_identity_element_exists_identity_element_self_commutativity_associativity_l1964_196467


namespace NUMINAMATH_GPT_physics_class_size_l1964_196474

variable (students : ℕ)
variable (physics math both : ℕ)

-- Conditions
def conditions := students = 75 ∧ physics = 2 * (math - both) + both ∧ both = 9

-- The proof goal
theorem physics_class_size : conditions students physics math both → physics = 56 := 
by 
  sorry

end NUMINAMATH_GPT_physics_class_size_l1964_196474


namespace NUMINAMATH_GPT_probability_first_4_second_club_third_2_l1964_196446

theorem probability_first_4_second_club_third_2 :
  let deck_size := 52
  let prob_4_first := 4 / deck_size
  let deck_minus_first_card := deck_size - 1
  let prob_club_second := 13 / deck_minus_first_card
  let deck_minus_two_cards := deck_minus_first_card - 1
  let prob_2_third := 4 / deck_minus_two_cards
  prob_4_first * prob_club_second * prob_2_third = 1 / 663 :=
by
  sorry

end NUMINAMATH_GPT_probability_first_4_second_club_third_2_l1964_196446


namespace NUMINAMATH_GPT_wine_division_l1964_196411

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end NUMINAMATH_GPT_wine_division_l1964_196411


namespace NUMINAMATH_GPT_algebraic_expression_value_l1964_196464

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1964_196464


namespace NUMINAMATH_GPT_max_integer_value_of_x_l1964_196435

theorem max_integer_value_of_x (x : ℤ) : 3 * x - (1 / 4 : ℚ) ≤ (1 / 3 : ℚ) * x - 2 → x ≤ -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_max_integer_value_of_x_l1964_196435


namespace NUMINAMATH_GPT_birthday_money_l1964_196484

theorem birthday_money (x : ℤ) (h₀ : 16 + x - 25 = 19) : x = 28 :=
by
  sorry

end NUMINAMATH_GPT_birthday_money_l1964_196484


namespace NUMINAMATH_GPT_return_time_possibilities_l1964_196456

variables (d v w : ℝ) (t_return : ℝ)

-- Condition 1: Flight against wind takes 84 minutes
axiom flight_against_wind : d / (v - w) = 84

-- Condition 2: Return trip with wind takes 9 minutes less than without wind
axiom return_wind_condition : d / (v + w) = d / v - 9

-- Problem Statement: Find the possible return times
theorem return_time_possibilities :
  t_return = d / (v + w) → t_return = 63 ∨ t_return = 12 :=
sorry

end NUMINAMATH_GPT_return_time_possibilities_l1964_196456


namespace NUMINAMATH_GPT_correct_answer_l1964_196406

variables (x y : ℝ)

def cost_equations (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 120) ∧ (2 * x - y = 20)

theorem correct_answer : cost_equations x y :=
sorry

end NUMINAMATH_GPT_correct_answer_l1964_196406


namespace NUMINAMATH_GPT_noncongruent_integer_tris_l1964_196452

theorem noncongruent_integer_tris : 
  ∃ S : Finset (ℕ × ℕ × ℕ), S.card = 18 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ S → 
      (a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2) :=
sorry

end NUMINAMATH_GPT_noncongruent_integer_tris_l1964_196452


namespace NUMINAMATH_GPT_find_k_l1964_196450

theorem find_k (k : ℕ) : (∃ n : ℕ, 2^k + 8*k + 5 = n^2) ↔ k = 2 := by
  sorry

end NUMINAMATH_GPT_find_k_l1964_196450


namespace NUMINAMATH_GPT_range_of_m_value_of_m_l1964_196463

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end NUMINAMATH_GPT_range_of_m_value_of_m_l1964_196463


namespace NUMINAMATH_GPT_find_e_l1964_196440

theorem find_e (a e : ℕ) (h1: a = 105) (h2: a ^ 3 = 21 * 25 * 45 * e) : e = 49 :=
sorry

end NUMINAMATH_GPT_find_e_l1964_196440


namespace NUMINAMATH_GPT_plywood_width_is_5_l1964_196461

theorem plywood_width_is_5 (length width perimeter : ℕ) (h1 : length = 6) (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 22) : width = 5 :=
by {
  -- proof steps would go here, but are omitted per instructions
  sorry
}

end NUMINAMATH_GPT_plywood_width_is_5_l1964_196461


namespace NUMINAMATH_GPT_four_digit_divisors_l1964_196449

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end NUMINAMATH_GPT_four_digit_divisors_l1964_196449


namespace NUMINAMATH_GPT_triangle_inequality_1_triangle_inequality_2_l1964_196472

variable (a b c : ℝ)

theorem triangle_inequality_1 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b * c + 28 / 27 ≥ a * b + b * c + c * a :=
by
  sorry

theorem triangle_inequality_2 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b + b * c + c * a ≥ a * b * c + 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_1_triangle_inequality_2_l1964_196472


namespace NUMINAMATH_GPT_no_valid_abc_l1964_196441

theorem no_valid_abc : 
  ∀ (a b c : ℕ), (100 * a + 10 * b + c) % 15 = 0 → (10 * b + c) % 4 = 0 → a > b → b > c → false :=
by
  intros a b c habc_mod15 hbc_mod4 h_ab_gt h_bc_gt
  sorry

end NUMINAMATH_GPT_no_valid_abc_l1964_196441


namespace NUMINAMATH_GPT_min_seats_occupied_l1964_196445

theorem min_seats_occupied (total_seats : ℕ) (h_total_seats : total_seats = 180) : 
  ∃ min_occupied : ℕ, 
    min_occupied = 90 ∧ 
    (∀ num_occupied : ℕ, num_occupied < min_occupied -> 
      ∃ next_seat : ℕ, (next_seat ≤ total_seats ∧ 
      num_occupied + next_seat < total_seats ∧ 
      (next_seat + 1 ≤ total_seats → ∃ a b: ℕ, a = next_seat ∧ b = next_seat + 1 ∧ 
      num_occupied + 1 < min_occupied ∧ 
      (a = b ∨ b = a + 1)))) :=
sorry

end NUMINAMATH_GPT_min_seats_occupied_l1964_196445


namespace NUMINAMATH_GPT_arrange_books_l1964_196481

open Nat

theorem arrange_books :
    let german_books := 3
    let spanish_books := 4
    let french_books := 3
    let total_books := german_books + spanish_books + french_books
    (total_books == 10) →
    let units := 2
    let items_to_arrange := units + german_books
    factorial items_to_arrange * factorial spanish_books * factorial french_books = 17280 :=
by 
    intros
    sorry

end NUMINAMATH_GPT_arrange_books_l1964_196481


namespace NUMINAMATH_GPT_complex_problem_l1964_196429

def is_imaginary_unit (x : ℂ) : Prop := x^2 = -1

theorem complex_problem (a b : ℝ) (i : ℂ) (h1 : (a - 2 * i) / i = (b : ℂ) + i) (h2 : is_imaginary_unit i) :
  a - b = 1 := 
sorry

end NUMINAMATH_GPT_complex_problem_l1964_196429


namespace NUMINAMATH_GPT_vegetarian_count_l1964_196403

variables (v_only v_nboth vegan pesc nvboth : ℕ)
variables (hv_only : v_only = 13) (hv_nboth : v_nboth = 8)
          (hvegan_tot : vegan = 5) (hvegan_v : vveg1 = 3)
          (hpesc_tot : pesc = 4) (hpesc_vnboth : nvboth = 2)

theorem vegetarian_count (total_veg : ℕ) 
  (H_total : total_veg = v_only + v_nboth + (vegan - vveg1)) :
  total_veg = 23 :=
sorry

end NUMINAMATH_GPT_vegetarian_count_l1964_196403


namespace NUMINAMATH_GPT_coloring_15_segments_impossible_l1964_196468

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end NUMINAMATH_GPT_coloring_15_segments_impossible_l1964_196468


namespace NUMINAMATH_GPT_volume_of_pyramid_l1964_196420

noncomputable def volume_pyramid : ℝ :=
  let a := 9
  let b := 12
  let s := 15
  let base_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let half_diagonal := diagonal / 2
  let height := Real.sqrt (s^2 - half_diagonal^2)
  (1 / 3) * base_area * height

theorem volume_of_pyramid :
  volume_pyramid = 36 * Real.sqrt 168.75 := by
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1964_196420


namespace NUMINAMATH_GPT_fraction_of_field_planted_l1964_196479

theorem fraction_of_field_planted : 
  let field_area := 5 * 6
  let triangle_area := (5 * 6) / 2
  let a := (41 * 3) / 33  -- derived from the given conditions
  let square_area := a^2
  let planted_area := triangle_area - square_area
  (planted_area / field_area) = (404 / 841) := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_field_planted_l1964_196479


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_83_l1964_196444

theorem last_three_digits_of_7_pow_83 :
  (7 ^ 83) % 1000 = 886 := sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_83_l1964_196444


namespace NUMINAMATH_GPT_other_x_intercept_l1964_196459

theorem other_x_intercept (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = y) 
  (h_vertex: (5, 10) = ((-b / (2 * a)), (4 * a * 10 / (4 * a)))) 
  (h_intercept : ∃ x, a * x * 0 + b * 0 + c = 0) : ∃ x, x = 10 :=
by
  sorry

end NUMINAMATH_GPT_other_x_intercept_l1964_196459


namespace NUMINAMATH_GPT_initial_number_18_l1964_196458

theorem initial_number_18 (N : ℤ) (h : ∃ k : ℤ, N + 5 = 23 * k) : N = 18 := 
sorry

end NUMINAMATH_GPT_initial_number_18_l1964_196458


namespace NUMINAMATH_GPT_pythagorean_triple_correct_l1964_196477

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct :
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 7 9 11 ∧
  ¬ is_pythagorean_triple 6 9 12 ∧
  ¬ is_pythagorean_triple (3/10) (4/10) (5/10) :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_correct_l1964_196477


namespace NUMINAMATH_GPT_domain_of_function_l1964_196433

noncomputable def function_domain := {x : ℝ | 1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0}

theorem domain_of_function : function_domain = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1964_196433


namespace NUMINAMATH_GPT_unique_function_f_l1964_196410

theorem unique_function_f (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x = -f (-x))
    (h2 : ∀ x : ℝ, f (x + 1) = f x + 1)
    (h3 : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / x^2 * f x) :
    ∀ x : ℝ, f x = x := 
sorry

end NUMINAMATH_GPT_unique_function_f_l1964_196410


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1964_196462

variable (x y z : ℝ)

theorem sum_of_three_numbers :
  y = 5 → 
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 :=
by
  intros hy h1 h2
  rw [hy] at h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1964_196462


namespace NUMINAMATH_GPT_intersection_when_m_eq_2_range_of_m_l1964_196492

open Set

variables (m x : ℝ)

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}
def intersection (m : ℝ) : Set ℝ := A m ∩ B

-- First proof: When m = 2, the intersection of A and B is [1,2].
theorem intersection_when_m_eq_2 : intersection 2 = {x | 1 ≤ x ∧ x ≤ 2} :=
sorry

-- Second proof: The range of m such that A ⊆ A ∩ B
theorem range_of_m : {m | A m ⊆ B} = {m | -2 ≤ m ∧ m ≤ 1 / 2} :=
sorry

end NUMINAMATH_GPT_intersection_when_m_eq_2_range_of_m_l1964_196492


namespace NUMINAMATH_GPT_subtraction_correct_l1964_196402

theorem subtraction_correct :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end NUMINAMATH_GPT_subtraction_correct_l1964_196402


namespace NUMINAMATH_GPT_count_arithmetic_sequence_l1964_196470

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end NUMINAMATH_GPT_count_arithmetic_sequence_l1964_196470


namespace NUMINAMATH_GPT_find_solutions_l1964_196493

theorem find_solutions :
  ∀ (x n : ℕ), 0 < x → 0 < n → x^(n+1) - (x + 1)^n = 2001 → (x, n) = (13, 2) :=
by
  intros x n hx hn heq
  sorry

end NUMINAMATH_GPT_find_solutions_l1964_196493


namespace NUMINAMATH_GPT_rectangular_plot_dimensions_l1964_196488

theorem rectangular_plot_dimensions (a b : ℝ) 
  (h_area : a * b = 800) 
  (h_perimeter_fencing : 2 * a + b = 100) :
  (a = 40 ∧ b = 20) ∨ (a = 10 ∧ b = 80) := 
sorry

end NUMINAMATH_GPT_rectangular_plot_dimensions_l1964_196488


namespace NUMINAMATH_GPT_triangle_angle_D_l1964_196417

theorem triangle_angle_D (F E D : ℝ) (hF : F = 15) (hE : E = 3 * F) (h_triangle : D + E + F = 180) : D = 120 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_D_l1964_196417


namespace NUMINAMATH_GPT_office_distance_eq_10_l1964_196497

noncomputable def distance_to_office (D T : ℝ) : Prop :=
  D = 10 * (T + 10 / 60) ∧ D = 15 * (T - 10 / 60)

theorem office_distance_eq_10 (D T : ℝ) (h : distance_to_office D T) : D = 10 :=
by
  sorry

end NUMINAMATH_GPT_office_distance_eq_10_l1964_196497


namespace NUMINAMATH_GPT_transformations_map_onto_self_l1964_196423

/-- Define the transformations -/
def T1 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a 90 degree rotation around the center of a square
  sorry

def T2 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a translation parallel to line ℓ
  sorry

def T3 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across line ℓ
  sorry

def T4 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across a line perpendicular to line ℓ
  sorry

/-- Define the pattern -/
def pattern (p : ℝ × ℝ) : Type :=
  -- Representation of alternating right triangles and squares along line ℓ
  sorry

/-- The main theorem:
    Prove that there are exactly 3 transformations (T1, T2, T3) that will map the pattern onto itself. -/
theorem transformations_map_onto_self : (∃ pattern : ℝ × ℝ → Type,
  (T1 pattern = pattern) ∧
  (T2 pattern = pattern) ∧
  (T3 pattern = pattern) ∧
  ¬ (T4 pattern = pattern)) → (3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_transformations_map_onto_self_l1964_196423


namespace NUMINAMATH_GPT_find_n_value_l1964_196438

theorem find_n_value (n a b : ℕ) 
    (h1 : n = 12 * b + a)
    (h2 : n = 10 * a + b)
    (h3 : 0 ≤ a ∧ a ≤ 11)
    (h4 : 0 ≤ b ∧ b ≤ 9) : 
    n = 119 :=
by
  sorry

end NUMINAMATH_GPT_find_n_value_l1964_196438


namespace NUMINAMATH_GPT_parallel_planes_imply_l1964_196432

variable {Point Line Plane : Type}

-- Definitions of parallelism and perpendicularity between lines and planes
variables {parallel_perpendicular : Line → Plane → Prop}
variables {parallel_lines : Line → Line → Prop}
variables {parallel_planes : Plane → Plane → Prop}

-- Given conditions
variable {m n : Line}
variable {α β : Plane}

-- Conditions
axiom m_parallel_n : parallel_lines m n
axiom m_perpendicular_α : parallel_perpendicular m α
axiom n_perpendicular_β : parallel_perpendicular n β

-- The statement to be proven
theorem parallel_planes_imply (m_parallel_n : parallel_lines m n)
  (m_perpendicular_α : parallel_perpendicular m α)
  (n_perpendicular_β : parallel_perpendicular n β) :
  parallel_planes α β :=
sorry

end NUMINAMATH_GPT_parallel_planes_imply_l1964_196432


namespace NUMINAMATH_GPT_coeff_x2_product_l1964_196496

open Polynomial

noncomputable def poly1 : Polynomial ℤ := -5 * X^3 - 5 * X^2 - 7 * X + 1
noncomputable def poly2 : Polynomial ℤ := -X^2 - 6 * X + 1

theorem coeff_x2_product : (poly1 * poly2).coeff 2 = 36 := by
  sorry

end NUMINAMATH_GPT_coeff_x2_product_l1964_196496


namespace NUMINAMATH_GPT_probability_no_three_consecutive_1s_l1964_196422

theorem probability_no_three_consecutive_1s (m n : ℕ) (h_relatively_prime : Nat.gcd m n = 1) (h_eq : 2^12 = 4096) :
  let b₁ := 2
  let b₂ := 4
  let b₃ := 7
  let b₄ := b₃ + b₂ + b₁
  let b₅ := b₄ + b₃ + b₂
  let b₆ := b₅ + b₄ + b₃
  let b₇ := b₆ + b₅ + b₄
  let b₈ := b₇ + b₆ + b₅
  let b₉ := b₈ + b₇ + b₆
  let b₁₀ := b₉ + b₈ + b₇
  let b₁₁ := b₁₀ + b₉ + b₈
  let b₁₂ := b₁₁ + b₁₀ + b₉
  (m = 1705 ∧ n = 4096 ∧ b₁₂ = m) →
  m + n = 5801 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_no_three_consecutive_1s_l1964_196422


namespace NUMINAMATH_GPT_correct_exp_identity_l1964_196437

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end NUMINAMATH_GPT_correct_exp_identity_l1964_196437


namespace NUMINAMATH_GPT_find_num_female_students_l1964_196439

noncomputable def numFemaleStudents (totalAvg maleAvg femaleAvg : ℕ) (numMales : ℕ) : ℕ :=
  let numFemales := (totalAvg * (numMales + (totalAvg * 0)) - (maleAvg * numMales)) / femaleAvg
  numFemales

theorem find_num_female_students :
  (totalAvg maleAvg femaleAvg : ℕ) →
  (numMales : ℕ) →
  totalAvg = 90 →
  maleAvg = 83 →
  femaleAvg = 92 →
  numMales = 8 →
  numFemaleStudents totalAvg maleAvg femaleAvg numMales = 28 := by
    intros
    sorry

end NUMINAMATH_GPT_find_num_female_students_l1964_196439


namespace NUMINAMATH_GPT_taxi_fare_function_l1964_196405

theorem taxi_fare_function (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, y = 2 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_function_l1964_196405


namespace NUMINAMATH_GPT_set_representation_listing_method_l1964_196414

def is_in_set (a : ℤ) : Prop := 0 < 2 * a - 1 ∧ 2 * a - 1 ≤ 5

def M : Set ℤ := {a | is_in_set a}

theorem set_representation_listing_method :
  M = {1, 2, 3} :=
sorry

end NUMINAMATH_GPT_set_representation_listing_method_l1964_196414


namespace NUMINAMATH_GPT_gwen_science_problems_l1964_196469

theorem gwen_science_problems (math_problems : ℕ) (finished_problems : ℕ) (remaining_problems : ℕ)
  (h1 : math_problems = 18) (h2 : finished_problems = 24) (h3 : remaining_problems = 5) :
  (finished_problems + remaining_problems - math_problems = 11) :=
by
  sorry

end NUMINAMATH_GPT_gwen_science_problems_l1964_196469


namespace NUMINAMATH_GPT_exists_fraction_equal_to_d_minus_1_l1964_196404

theorem exists_fraction_equal_to_d_minus_1 (n d : ℕ) (hdiv : d > 0 ∧ n % d = 0) :
  ∃ k : ℕ, k < n ∧ (n - k) / (n - (n - k)) = d - 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_fraction_equal_to_d_minus_1_l1964_196404


namespace NUMINAMATH_GPT_find_positive_number_l1964_196424

theorem find_positive_number (x n : ℝ) (h₁ : (x + 1) ^ 2 = n) (h₂ : (x - 5) ^ 2 = n) : n = 9 := 
sorry

end NUMINAMATH_GPT_find_positive_number_l1964_196424


namespace NUMINAMATH_GPT_stratified_sampling_difference_l1964_196430

theorem stratified_sampling_difference
  (male_athletes : ℕ := 56)
  (female_athletes : ℕ := 42)
  (sample_size : ℕ := 28)
  (H_total : male_athletes + female_athletes = 98)
  (H_sample_frac : sample_size = 28)
  : (56 * (sample_size / 98) - 42 * (sample_size / 98) = 4) :=
sorry

end NUMINAMATH_GPT_stratified_sampling_difference_l1964_196430


namespace NUMINAMATH_GPT_find_radius_l1964_196485

noncomputable def radius_from_tangent_circles (AB : ℝ) (r : ℝ) : ℝ :=
  let O1O2 := 2 * r
  let proportion := AB / O1O2
  r + r * proportion

theorem find_radius
  (AB : ℝ) (r : ℝ)
  (hAB : AB = 11) (hr : r = 5) :
  radius_from_tangent_circles AB r = 55 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l1964_196485


namespace NUMINAMATH_GPT_math_problem_l1964_196486

theorem math_problem (a b c d x : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |x| = 2) :
  x^4 + c * d * x^2 - a - b = 20 :=
sorry

end NUMINAMATH_GPT_math_problem_l1964_196486


namespace NUMINAMATH_GPT_fraction_difference_l1964_196427

theorem fraction_difference (a b : ℝ) : 
  (a / (a + 1)) - (b / (b + 1)) = (a - b) / ((a + 1) * (b + 1)) :=
sorry

end NUMINAMATH_GPT_fraction_difference_l1964_196427


namespace NUMINAMATH_GPT_monotonically_decreasing_iff_l1964_196413

noncomputable def f (a x : ℝ) : ℝ := (x^2 - 2 * a * x) * Real.exp x

theorem monotonically_decreasing_iff (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a (-1) ∧ f a x ≤ f a 1) ↔ (a ≥ 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_monotonically_decreasing_iff_l1964_196413


namespace NUMINAMATH_GPT_cos_triple_angle_l1964_196401

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l1964_196401


namespace NUMINAMATH_GPT_distance_difference_l1964_196451

-- Definitions related to the problem conditions
variables (v D_AB D_BC D_AC : ℝ)

-- Conditions
axiom h1 : D_AB = v * 7
axiom h2 : D_BC = v * 5
axiom h3 : D_AC = 6
axiom h4 : D_AC = D_AB + D_BC

-- Theorem for proof problem
theorem distance_difference : D_AB - D_BC = 1 :=
by sorry

end NUMINAMATH_GPT_distance_difference_l1964_196451


namespace NUMINAMATH_GPT_a_3_equals_35_l1964_196416

noncomputable def S (n : ℕ) : ℕ := 5 * n ^ 2 + 10 * n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_3_equals_35 : a 3 = 35 := by
  sorry

end NUMINAMATH_GPT_a_3_equals_35_l1964_196416


namespace NUMINAMATH_GPT_ben_initial_marbles_l1964_196454

theorem ben_initial_marbles (B : ℕ) (John_initial_marbles : ℕ) (H1 : John_initial_marbles = 17) (H2 : John_initial_marbles + B / 2 = B / 2 + B / 2 + 17) : B = 34 := by
  sorry

end NUMINAMATH_GPT_ben_initial_marbles_l1964_196454


namespace NUMINAMATH_GPT_cost_of_large_tubs_l1964_196471

theorem cost_of_large_tubs (L : ℝ) (h1 : 3 * L + 6 * 5 = 48) : L = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_large_tubs_l1964_196471


namespace NUMINAMATH_GPT_hiring_probabilities_l1964_196412

-- Define the candidates and their abilities
inductive Candidate : Type
| Strong
| Moderate
| Weak

open Candidate

-- Define the ordering rule and hiring rule
def interviewOrders : List (Candidate × Candidate × Candidate) :=
  [(Strong, Moderate, Weak), (Strong, Weak, Moderate), 
   (Moderate, Strong, Weak), (Moderate, Weak, Strong),
   (Weak, Strong, Moderate), (Weak, Moderate, Strong)]

def hiresStrong (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Moderate, Strong, Weak) => true
  | (Moderate, Weak, Strong) => true
  | (Weak, Strong, Moderate) => true
  | _ => false

def hiresModerate (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Strong, Weak, Moderate) => true
  | (Weak, Moderate, Strong) => true
  | _ => false

-- The main theorem to be proved
theorem hiring_probabilities :
  let orders := interviewOrders
  let p := (orders.filter hiresStrong).length / orders.length
  let q := (orders.filter hiresModerate).length / orders.length
  p = 1 / 2 ∧ q = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_hiring_probabilities_l1964_196412


namespace NUMINAMATH_GPT_fraction_habitable_surface_l1964_196475

noncomputable def fraction_land_not_covered_by_water : ℚ := 1 / 3
noncomputable def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_land_not_covered_by_water * fraction_inhabitable_land = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_habitable_surface_l1964_196475


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1964_196498

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 2 → a (k + 1) - a k^2 + a (k - 1) = 0) (h2 : ∀ k, a k ≠ 0) (h3 : ∀ k ≥ 2, a (k + 1) + a (k - 1) = 2 * a k) :
  S (2 * n - 1) - 4 * n = -2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1964_196498


namespace NUMINAMATH_GPT_sets_tossed_per_show_l1964_196421

-- Definitions
def sets_used_per_show : ℕ := 5
def number_of_shows : ℕ := 30
def total_sets_used : ℕ := 330

-- Statement to prove
theorem sets_tossed_per_show : 
  (total_sets_used - (sets_used_per_show * number_of_shows)) / number_of_shows = 6 := 
by
  sorry

end NUMINAMATH_GPT_sets_tossed_per_show_l1964_196421


namespace NUMINAMATH_GPT_minimize_quadratic_l1964_196431

theorem minimize_quadratic : ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12*x + 28 ≤ y^2 - 12*y + 28) :=
by
  use 6
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l1964_196431


namespace NUMINAMATH_GPT_max_card_count_sum_l1964_196487

theorem max_card_count_sum (W B R : ℕ) (total_cards : ℕ) 
  (white_cards black_cards red_cards : ℕ) : 
  total_cards = 300 ∧ white_cards = 100 ∧ black_cards = 100 ∧ red_cards = 100 ∧
  (∀ w, w < white_cards → ∃ b, b < black_cards) ∧ 
  (∀ b, b < black_cards → ∃ r, r < red_cards) ∧ 
  (∀ r, r < red_cards → ∃ w, w < white_cards) →
  ∃ max_sum, max_sum = 20000 :=
by
  sorry

end NUMINAMATH_GPT_max_card_count_sum_l1964_196487


namespace NUMINAMATH_GPT_distinct_roots_iff_l1964_196447

theorem distinct_roots_iff (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + m + 3 = 0 ∧ x2^2 + m * x2 + m + 3 = 0) ↔ (m < -2 ∨ m > 6) := 
sorry

end NUMINAMATH_GPT_distinct_roots_iff_l1964_196447


namespace NUMINAMATH_GPT_vector_expression_eval_l1964_196482

open Real

noncomputable def v1 : ℝ × ℝ := (3, -8)
noncomputable def v2 : ℝ × ℝ := (2, -4)
noncomputable def k : ℝ := 5

theorem vector_expression_eval : (v1.1 - k * v2.1, v1.2 - k * v2.2) = (-7, 12) :=
  by sorry

end NUMINAMATH_GPT_vector_expression_eval_l1964_196482


namespace NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1964_196409

-- Define the basic events and conditions
structure Pocket :=
(red : ℕ)
(black : ℕ)

-- Define the event type
inductive Event
| atleast_one_black : Event
| both_black : Event
| atleast_one_red : Event
| both_red : Event
| exactly_one_black : Event
| exactly_two_black : Event
| none_black : Event

def is_mutually_exclusive (e1 e2 : Event) : Prop :=
  match e1, e2 with
  | Event.exactly_one_black, Event.exactly_two_black => true
  | Event.exactly_two_black, Event.exactly_one_black => true
  | _, _ => false

def is_complementary (e1 e2 : Event) : Prop :=
  e1 = Event.none_black ∧ e2 = Event.both_red ∨
  e1 = Event.both_red ∧ e2 = Event.none_black

-- Given conditions
def pocket : Pocket := { red := 2, black := 2 }

-- Proof problem setup
theorem mutually_exclusive_not_complementary : 
  is_mutually_exclusive Event.exactly_one_black Event.exactly_two_black ∧
  ¬ is_complementary Event.exactly_one_black Event.exactly_two_black :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1964_196409
