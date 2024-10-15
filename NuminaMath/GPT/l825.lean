import Mathlib

namespace NUMINAMATH_GPT_find_value_of_a_l825_82512

theorem find_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≤ 24) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 24) ∧
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 3) → 
  a = 2 ∨ a = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l825_82512


namespace NUMINAMATH_GPT_smallest_b_value_l825_82535

def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

def not_triangle (x y z : ℝ) : Prop :=
  ¬triangle_inequality x y z

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
    (h3 : not_triangle 2 a b) (h4 : not_triangle (1 / b) (1 / a) 1) :
    b >= 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_value_l825_82535


namespace NUMINAMATH_GPT_find_f_neg1_l825_82550

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_f1 : f 1 = 2)

-- Theorem stating the necessary proof
theorem find_f_neg1 : f (-1) = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg1_l825_82550


namespace NUMINAMATH_GPT_january_31_is_friday_l825_82572

theorem january_31_is_friday (h : ∀ (d : ℕ), (d % 7 = 0 → d = 1)) : ∀ d, (d = 31) → (d % 7 = 3) :=
by
  sorry

end NUMINAMATH_GPT_january_31_is_friday_l825_82572


namespace NUMINAMATH_GPT_probability_of_mathematics_letter_l825_82513

-- Definitions for the problem
def english_alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Set the total number of letters in the English alphabet
def total_letters := english_alphabet.card

-- Set the number of unique letters in 'MATHEMATICS'
def mathematics_unique_letters := mathematics_letters.card

-- Statement of the Lean theorem
theorem probability_of_mathematics_letter : (mathematics_unique_letters : ℚ) / total_letters = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_mathematics_letter_l825_82513


namespace NUMINAMATH_GPT_value_range_of_f_l825_82521

def f (x : ℝ) := 2 * x ^ 2 + 4 * x + 1

theorem value_range_of_f :
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 4 → (∃ y ∈ Set.Icc (-1 : ℝ) 49, f x = y) :=
by sorry

end NUMINAMATH_GPT_value_range_of_f_l825_82521


namespace NUMINAMATH_GPT_cost_per_topping_is_2_l825_82561

theorem cost_per_topping_is_2 : 
  ∃ (x : ℝ), 
    let large_pizza_cost := 14 
    let num_large_pizzas := 2 
    let num_toppings_per_pizza := 3 
    let tip_rate := 0.25 
    let total_cost := 50 
    let cost_pizzas := num_large_pizzas * large_pizza_cost 
    let num_toppings := num_large_pizzas * num_toppings_per_pizza 
    let cost_toppings := num_toppings * x 
    let before_tip_cost := cost_pizzas + cost_toppings 
    let tip := tip_rate * before_tip_cost 
    let final_cost := before_tip_cost + tip 
    final_cost = total_cost ∧ x = 2 := 
by
  simp
  sorry

end NUMINAMATH_GPT_cost_per_topping_is_2_l825_82561


namespace NUMINAMATH_GPT_lines_through_point_l825_82548

theorem lines_through_point (k : ℝ) : ∀ x y : ℝ, (y = k * (x - 1)) ↔ (x = 1 ∧ y = 0) ∨ (x ≠ 1 ∧ y / (x - 1) = k) :=
by
  sorry

end NUMINAMATH_GPT_lines_through_point_l825_82548


namespace NUMINAMATH_GPT_three_integers_desc_order_l825_82589

theorem three_integers_desc_order (a b c : ℤ) : ∃ a' b' c' : ℤ, 
  (a = a' ∨ a = b' ∨ a = c') ∧
  (b = a' ∨ b = b' ∨ b = c') ∧
  (c = a' ∨ c = b' ∨ c = c') ∧ 
  (a' ≠ b' ∨ a' ≠ c' ∨ b' ≠ c') ∧
  a' ≥ b' ∧ b' ≥ c' :=
sorry

end NUMINAMATH_GPT_three_integers_desc_order_l825_82589


namespace NUMINAMATH_GPT_quarters_total_l825_82581

variable (q1 q2 S: Nat)

def original_quarters := 760
def additional_quarters := 418

theorem quarters_total : S = original_quarters + additional_quarters :=
sorry

end NUMINAMATH_GPT_quarters_total_l825_82581


namespace NUMINAMATH_GPT_range_of_x_l825_82558

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + (a - 4) * x + 4 - 2 * a

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  ∀ x : ℝ, (f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_range_of_x_l825_82558


namespace NUMINAMATH_GPT_david_remaining_money_l825_82584

noncomputable def initial_funds : ℝ := 1500
noncomputable def spent_on_accommodations : ℝ := 400
noncomputable def spent_on_food_eur : ℝ := 300
noncomputable def eur_to_usd : ℝ := 1.10
noncomputable def spent_on_souvenirs_yen : ℝ := 5000
noncomputable def yen_to_usd : ℝ := 0.009
noncomputable def loan_to_friend : ℝ := 200
noncomputable def difference : ℝ := 500

noncomputable def spent_on_food_usd : ℝ := spent_on_food_eur * eur_to_usd
noncomputable def spent_on_souvenirs_usd : ℝ := spent_on_souvenirs_yen * yen_to_usd
noncomputable def total_spent_excluding_loan : ℝ := spent_on_accommodations + spent_on_food_usd + spent_on_souvenirs_usd

theorem david_remaining_money : 
  initial_funds - total_spent_excluding_loan - difference = 275 :=
by
  sorry

end NUMINAMATH_GPT_david_remaining_money_l825_82584


namespace NUMINAMATH_GPT_four_digit_integer_transformation_l825_82578

theorem four_digit_integer_transformation (a b c d n : ℕ) (A : ℕ)
  (hA : A = 1000 * a + 100 * b + 10 * c + d)
  (ha : a + 2 < 10)
  (hc : c + 2 < 10)
  (hb : b ≥ 2)
  (hd : d ≥ 2)
  (hA4 : 1000 ≤ A ∧ A < 10000) :
  (1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n)) = n * A → n = 2 → A = 1818 :=
by sorry

end NUMINAMATH_GPT_four_digit_integer_transformation_l825_82578


namespace NUMINAMATH_GPT_range_of_x_l825_82594

variable {f : ℝ → ℝ}

-- Define the function is_increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_x (h_inc : is_increasing f) (h_ineq : ∀ x : ℝ, f x < f (2 * x - 3)) :
  ∀ x : ℝ, 3 < x → f x < f (2 * x - 3) := 
sorry

end NUMINAMATH_GPT_range_of_x_l825_82594


namespace NUMINAMATH_GPT_rectangle_width_l825_82562

theorem rectangle_width (w : ℝ)
    (h₁ : 5 > 0) (h₂ : 6 > 0) (h₃ : 3 > 0) 
    (area_relation : w * 5 = 3 * 6 + 2) : w = 4 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l825_82562


namespace NUMINAMATH_GPT_total_sides_tom_tim_l825_82530

def sides_per_die : Nat := 6

def tom_dice_count : Nat := 4
def tim_dice_count : Nat := 4

theorem total_sides_tom_tim : tom_dice_count * sides_per_die + tim_dice_count * sides_per_die = 48 := by
  sorry

end NUMINAMATH_GPT_total_sides_tom_tim_l825_82530


namespace NUMINAMATH_GPT_diagonals_perpendicular_l825_82564

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 3 }
def B : Point := { x := 2, y := 6 }
def C : Point := { x := 6, y := -1 }
def D : Point := { x := -3, y := -4 }

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem diagonals_perpendicular :
  let AC := vector A C
  let BD := vector B D
  dot_product AC BD = 0 :=
by
  let AC := vector A C
  let BD := vector B D
  sorry

end NUMINAMATH_GPT_diagonals_perpendicular_l825_82564


namespace NUMINAMATH_GPT_first_discount_calculation_l825_82515

-- Define the given conditions and final statement
theorem first_discount_calculation (P : ℝ) (D : ℝ) :
  (1.35 * (1 - D / 100) * 0.85 = 1.03275) → (D = 10.022) :=
by
  -- Proof is not provided, to be done.
  sorry

end NUMINAMATH_GPT_first_discount_calculation_l825_82515


namespace NUMINAMATH_GPT_find_deductive_reasoning_l825_82509

noncomputable def is_deductive_reasoning (reasoning : String) : Prop :=
  match reasoning with
  | "B" => true
  | _ => false

theorem find_deductive_reasoning : is_deductive_reasoning "B" = true :=
  sorry

end NUMINAMATH_GPT_find_deductive_reasoning_l825_82509


namespace NUMINAMATH_GPT_find_d_l825_82542

theorem find_d (a d : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d * x + 12) :
  d = 7 :=
sorry

end NUMINAMATH_GPT_find_d_l825_82542


namespace NUMINAMATH_GPT_sector_radius_cone_l825_82579

theorem sector_radius_cone {θ R r : ℝ} (sector_angle : θ = 120) (cone_base_radius : r = 2) :
  (R * θ / 360) * 2 * π = 2 * π * r → R = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sector_radius_cone_l825_82579


namespace NUMINAMATH_GPT_marta_hours_worked_l825_82526

-- Definitions of the conditions in Lean 4
def total_collected : ℕ := 240
def hourly_rate : ℕ := 10
def tips_collected : ℕ := 50
def work_earned : ℕ := total_collected - tips_collected

-- Goal: To prove the number of hours worked by Marta
theorem marta_hours_worked : work_earned / hourly_rate = 19 := by
  sorry

end NUMINAMATH_GPT_marta_hours_worked_l825_82526


namespace NUMINAMATH_GPT_sum_abc_equals_16_l825_82580

theorem sum_abc_equals_16 (a b c : ℝ) (h : (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0) : 
  a + b + c = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_abc_equals_16_l825_82580


namespace NUMINAMATH_GPT_travelers_on_liner_l825_82597

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end NUMINAMATH_GPT_travelers_on_liner_l825_82597


namespace NUMINAMATH_GPT_joker_then_spade_probability_correct_l825_82546

-- Defining the conditions of the deck
def deck_size : ℕ := 60
def joker_count : ℕ := 4
def suit_count : ℕ := 4
def cards_per_suit : ℕ := 15

-- The probability of drawing a Joker first and then a spade
def prob_joker_then_spade : ℚ :=
  (joker_count * (cards_per_suit - 1) + (deck_size - joker_count) * cards_per_suit) /
  (deck_size * (deck_size - 1))

-- The expected probability according to the solution
def expected_prob : ℚ := 224 / 885

theorem joker_then_spade_probability_correct :
  prob_joker_then_spade = expected_prob :=
by
  -- Skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_joker_then_spade_probability_correct_l825_82546


namespace NUMINAMATH_GPT_quadratic_eq_has_equal_roots_l825_82516

theorem quadratic_eq_has_equal_roots (q : ℚ) :
  (∃ x : ℚ, x^2 - 3 * x + q = 0 ∧ (x^2 - 3 * x + q = 0)) → q = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_has_equal_roots_l825_82516


namespace NUMINAMATH_GPT_determinant_not_sufficient_nor_necessary_l825_82507

-- Definitions of the initial conditions
variables {a1 b1 a2 b2 c1 c2 : ℝ}

-- Conditions given: neither line coefficients form the zero vector
axiom non_zero_1 : a1^2 + b1^2 ≠ 0
axiom non_zero_2 : a2^2 + b2^2 ≠ 0

-- The matrix determinant condition and line parallelism
def determinant_condition (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 ≠ 0

def lines_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 = 0 ∧ a1 * c2 ≠ a2 * c1

-- Proof problem statement: proving equivalence
theorem determinant_not_sufficient_nor_necessary :
  ¬ (∀ a1 b1 a2 b2 c1 c2, (determinant_condition a1 b1 a2 b2 → lines_parallel a1 b1 c1 a2 b2 c2) ∧
                          (lines_parallel a1 b1 c1 a2 b2 c2 → determinant_condition a1 b1 a2 b2)) :=
sorry

end NUMINAMATH_GPT_determinant_not_sufficient_nor_necessary_l825_82507


namespace NUMINAMATH_GPT_geometric_progressions_common_ratio_l825_82573

theorem geometric_progressions_common_ratio (a b p q : ℝ) :
  (∀ n : ℕ, (a * p^n + b * q^n) = (a * b) * ((p^n + q^n)/a)) →
  p = q := by
  sorry

end NUMINAMATH_GPT_geometric_progressions_common_ratio_l825_82573


namespace NUMINAMATH_GPT_max_d_n_l825_82537

open Int

def a_n (n : ℕ) : ℤ := 80 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (a_n n) (a_n (n + 1))

theorem max_d_n : ∃ n : ℕ, d_n n = 5 ∧ ∀ m : ℕ, d_n m ≤ 5 := by
  sorry

end NUMINAMATH_GPT_max_d_n_l825_82537


namespace NUMINAMATH_GPT_part_a_part_b_l825_82567

theorem part_a (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * (Real.sqrt n + 1) / (n - 1))) :
  μ < (2 * (Real.sqrt n + 1) / (n - 1)) :=
by 
  exact h_μ.2

theorem part_b (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1)))) :
  μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1))) :=
by
  exact h_μ.2

end NUMINAMATH_GPT_part_a_part_b_l825_82567


namespace NUMINAMATH_GPT_coin_toss_dice_roll_l825_82544

theorem coin_toss_dice_roll :
  let coin_toss := 2 -- two outcomes for same side coin toss
  let dice_roll := 2 -- two outcomes for multiple of 3 on dice roll
  coin_toss * dice_roll = 4 :=
by
  sorry

end NUMINAMATH_GPT_coin_toss_dice_roll_l825_82544


namespace NUMINAMATH_GPT_razorback_tshirt_revenue_l825_82538

theorem razorback_tshirt_revenue 
    (total_tshirts : ℕ) (total_money : ℕ) 
    (h1 : total_tshirts = 245) 
    (h2 : total_money = 2205) : 
    (total_money / total_tshirts = 9) := 
by 
    sorry

end NUMINAMATH_GPT_razorback_tshirt_revenue_l825_82538


namespace NUMINAMATH_GPT_root_magnitude_conditions_l825_82500

theorem root_magnitude_conditions (p : ℝ) (h : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = -p) ∧ (r1 * r2 = -12)) :
  (∃ r1 r2 : ℝ, (r1 ≠ r2) ∧ |r1| > 2 ∨ |r2| > 2) ∧ (∀ r1 r2 : ℝ, (r1 + r2 = -p) ∧ (r1 * r2 = -12) → |r1| * |r2| ≤ 14) :=
by
  -- Proof of the theorem goes here
  sorry

end NUMINAMATH_GPT_root_magnitude_conditions_l825_82500


namespace NUMINAMATH_GPT_domain_of_function_l825_82529

theorem domain_of_function :
  (∀ x : ℝ, 2 + x ≥ 0 ∧ 3 - x ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_GPT_domain_of_function_l825_82529


namespace NUMINAMATH_GPT_common_root_poly_identity_l825_82570

theorem common_root_poly_identity
  (α p p' q q' : ℝ)
  (h1 : α^3 + p*α + q = 0)
  (h2 : α^3 + p'*α + q' = 0) : 
  (p * q' - q * p') * (p - p')^2 = (q - q')^3 := 
by
  sorry

end NUMINAMATH_GPT_common_root_poly_identity_l825_82570


namespace NUMINAMATH_GPT_candice_spending_l825_82591

variable (total_budget : ℕ) (remaining_money : ℕ) (mildred_spending : ℕ)

theorem candice_spending 
  (h1 : total_budget = 100)
  (h2 : remaining_money = 40)
  (h3 : mildred_spending = 25) :
  (total_budget - remaining_money) - mildred_spending = 35 := 
by
  sorry

end NUMINAMATH_GPT_candice_spending_l825_82591


namespace NUMINAMATH_GPT_negation_universal_proposition_l825_82511

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by sorry

end NUMINAMATH_GPT_negation_universal_proposition_l825_82511


namespace NUMINAMATH_GPT_operation_addition_x_l825_82508

theorem operation_addition_x (x : ℕ) (h : 106 + 106 + x + x = 19872) : x = 9830 :=
sorry

end NUMINAMATH_GPT_operation_addition_x_l825_82508


namespace NUMINAMATH_GPT_y_is_multiple_of_3_and_6_l825_82566

-- Define y as a sum of given numbers
def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_is_multiple_of_3_and_6 :
  (y % 3 = 0) ∧ (y % 6 = 0) :=
by
  -- Proof would go here, but we will end with sorry
  sorry

end NUMINAMATH_GPT_y_is_multiple_of_3_and_6_l825_82566


namespace NUMINAMATH_GPT_product_evaluation_l825_82518

theorem product_evaluation :
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 *
  (1 / 512) * 1024 * (1 / 2048) * 4096 = 64 :=
by
  sorry

end NUMINAMATH_GPT_product_evaluation_l825_82518


namespace NUMINAMATH_GPT_magnitude_of_2a_plus_b_l825_82555

open Real

variables (a b : ℝ × ℝ) (angle : ℝ)

-- Conditions
axiom angle_between_a_b (a b : ℝ × ℝ) : angle = π / 3 -- 60 degrees in radians
axiom norm_a_eq_1 (a : ℝ × ℝ) : ‖a‖ = 1
axiom b_eq (b : ℝ × ℝ) : b = (3, 0)

-- Theorem
theorem magnitude_of_2a_plus_b (h1 : angle = π / 3) (h2 : ‖a‖ = 1) (h3 : b = (3, 0)) :
  ‖2 • a + b‖ = sqrt 19 :=
sorry

end NUMINAMATH_GPT_magnitude_of_2a_plus_b_l825_82555


namespace NUMINAMATH_GPT_problem1_problem2_l825_82510

-- Definitions of vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 3)
def vec_c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Problem Part 1: Prove m = -1 given a ⊥ (b + c)
theorem problem1 (m : ℝ) (h : vec_a.1 * (vec_b + vec_c m).1 + vec_a.2 * (vec_b + vec_c m).2 = 0) : m = -1 :=
sorry

-- Problem Part 2: Prove k = -2 given k*a + b is collinear with 2*a - b
theorem problem2 (k : ℝ) (h : (k * vec_a.1 + vec_b.1) / (2 * vec_a.1 - vec_b.1) = (k * vec_a.2 + vec_b.2) / (2 * vec_a.2 - vec_b.2)) : k = -2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l825_82510


namespace NUMINAMATH_GPT_lincoln_high_students_club_overlap_l825_82552

theorem lincoln_high_students_club_overlap (total_students : ℕ)
  (drama_club_students science_club_students both_or_either_club_students : ℕ)
  (h1 : total_students = 500)
  (h2 : drama_club_students = 150)
  (h3 : science_club_students = 200)
  (h4 : both_or_either_club_students = 300) :
  drama_club_students + science_club_students - both_or_either_club_students = 50 :=
by
  sorry

end NUMINAMATH_GPT_lincoln_high_students_club_overlap_l825_82552


namespace NUMINAMATH_GPT_problem_1_problem_2_l825_82596

open Real

theorem problem_1 : sqrt 3 * cos (π / 12) - sin (π / 12) = sqrt 2 := 
sorry

theorem problem_2 : ∀ θ : ℝ, sqrt 3 * cos θ - sin θ ≤ 2 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l825_82596


namespace NUMINAMATH_GPT_tangent_line_at_e_range_of_a_l825_82517

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x - 2 * a * x

theorem tangent_line_at_e (a : ℝ) :
  a = 0 →
  ∃ m b : ℝ, (∀ x, y = m * x + b) ∧ 
             y = (2 / Real.exp 1 - 2 * Real.exp 1) * x + (Real.exp 1)^2 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 → g a x < 0) →
  a ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_e_range_of_a_l825_82517


namespace NUMINAMATH_GPT_root_of_f_l825_82593

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

theorem root_of_f (h_inv : f_inv 0 = 2) (h_interval : 1 ≤ (f_inv 0) ∧ (f_inv 0) ≤ 4) : f 2 = 0 := 
sorry

end NUMINAMATH_GPT_root_of_f_l825_82593


namespace NUMINAMATH_GPT_find_marks_in_physics_l825_82545

theorem find_marks_in_physics (P C M : ℕ) (h1 : P + C + M = 225) (h2 : P + M = 180) (h3 : P + C = 140) : 
    P = 95 :=
sorry

end NUMINAMATH_GPT_find_marks_in_physics_l825_82545


namespace NUMINAMATH_GPT_bob_fencing_needed_l825_82528

-- Problem conditions
def length : ℕ := 225
def width : ℕ := 125
def small_gate : ℕ := 3
def large_gate : ℕ := 10

-- Definition of perimeter
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Total width of the gates
def total_gate_width (g1 g2 : ℕ) : ℕ := g1 + g2

-- Amount of fencing needed
def fencing_needed (p gw : ℕ) : ℕ := p - gw

-- Theorem statement
theorem bob_fencing_needed :
  fencing_needed (perimeter length width) (total_gate_width small_gate large_gate) = 687 :=
by 
  sorry

end NUMINAMATH_GPT_bob_fencing_needed_l825_82528


namespace NUMINAMATH_GPT_linear_func_passing_point_l825_82505

theorem linear_func_passing_point :
  ∃ k : ℝ, ∀ x y : ℝ, (y = k * x + 1) → (x = -1 ∧ y = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_linear_func_passing_point_l825_82505


namespace NUMINAMATH_GPT_part1_part2_l825_82534

def A (x : ℝ) : Prop := x < -3 ∨ x > 7
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def complement_R_A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 7

theorem part1 (m : ℝ) :
  (∀ x, complement_R_A x ∨ B m x → complement_R_A x) →
  m ≤ 4 :=
by
  sorry

theorem part2 (m : ℝ) (a b : ℝ) :
  (∀ x, complement_R_A x ∧ B m x ↔ (a ≤ x ∧ x ≤ b)) ∧ (b - a ≥ 1) →
  3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l825_82534


namespace NUMINAMATH_GPT_caterpillars_left_on_tree_l825_82543

-- Definitions based on conditions
def initialCaterpillars : ℕ := 14
def hatchedCaterpillars : ℕ := 4
def caterpillarsLeftToCocoon : ℕ := 8

-- The proof problem statement in Lean
theorem caterpillars_left_on_tree : initialCaterpillars + hatchedCaterpillars - caterpillarsLeftToCocoon = 10 :=
by
  -- solution steps will go here eventually
  sorry

end NUMINAMATH_GPT_caterpillars_left_on_tree_l825_82543


namespace NUMINAMATH_GPT_workshop_worker_count_l825_82522

theorem workshop_worker_count (W T N : ℕ) (h1 : T = 7) (h2 : 8000 * W = 7 * 14000 + 6000 * N) (h3 : W = T + N) : W = 28 :=
by
  sorry

end NUMINAMATH_GPT_workshop_worker_count_l825_82522


namespace NUMINAMATH_GPT_find_constants_u_v_l825_82531

theorem find_constants_u_v : 
  ∃ u v : ℝ, (∀ x : ℝ, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 :=
sorry

end NUMINAMATH_GPT_find_constants_u_v_l825_82531


namespace NUMINAMATH_GPT_problem_16_l825_82588

-- Definitions of the problem conditions
def trapezoid_inscribed_in_circle (r : ℝ) (a b : ℝ) : Prop :=
  r = 25 ∧ a = 14 ∧ b = 30 

def average_leg_length_of_trapezoid (a b : ℝ) (m : ℝ) : Prop :=
  a = 14 ∧ b = 30 ∧ m = 2000 

-- Using Lean to state the problem
theorem problem_16 (r a b m : ℝ) 
  (h1 : trapezoid_inscribed_in_circle r a b) 
  (h2 : average_leg_length_of_trapezoid a b m) : 
  m = 2000 := by
  sorry

end NUMINAMATH_GPT_problem_16_l825_82588


namespace NUMINAMATH_GPT_workEfficiencyRatioProof_is_2_1_l825_82559

noncomputable def workEfficiencyRatioProof : Prop :=
  ∃ (A B : ℝ), 
  (1 / B = 21) ∧ 
  (1 / (A + B) = 7) ∧
  (A / B = 2)

theorem workEfficiencyRatioProof_is_2_1 : workEfficiencyRatioProof :=
  sorry

end NUMINAMATH_GPT_workEfficiencyRatioProof_is_2_1_l825_82559


namespace NUMINAMATH_GPT_bucket_capacity_l825_82586

theorem bucket_capacity (x : ℕ) (h₁ : 12 * x = 132 * 5) : x = 55 := by
  sorry

end NUMINAMATH_GPT_bucket_capacity_l825_82586


namespace NUMINAMATH_GPT_num_five_digit_palindromes_with_even_middle_l825_82554

theorem num_five_digit_palindromes_with_even_middle :
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ∃ c', c = 2 * c' ∧ 0 ≤ c' ∧ c' ≤ 4 ∧ 10000 * a + 1000 * b + 100 * c + 10 * b + a ≤ 99999) →
  9 * 10 * 5 = 450 :=
by
  sorry

end NUMINAMATH_GPT_num_five_digit_palindromes_with_even_middle_l825_82554


namespace NUMINAMATH_GPT_units_digit_3_pow_2005_l825_82560

theorem units_digit_3_pow_2005 : 
  let units_digit (n : ℕ) : ℕ := n % 10
  units_digit (3^2005) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_3_pow_2005_l825_82560


namespace NUMINAMATH_GPT_shaded_area_is_correct_l825_82536

def area_of_rectangle (l w : ℕ) : ℕ := l * w

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

def area_of_shaded_region : ℕ :=
  let length := 8
  let width := 4
  let area_rectangle := area_of_rectangle length width
  let area_triangle := area_of_triangle length width
  area_rectangle - area_triangle

theorem shaded_area_is_correct : area_of_shaded_region = 16 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l825_82536


namespace NUMINAMATH_GPT_purchasing_power_increase_l825_82576

theorem purchasing_power_increase (P M : ℝ) (h : 0 < P ∧ 0 < M) :
  let new_price := 0.80 * P
  let original_quantity := M / P
  let new_quantity := M / new_price
  new_quantity = 1.25 * original_quantity :=
by
  sorry

end NUMINAMATH_GPT_purchasing_power_increase_l825_82576


namespace NUMINAMATH_GPT_arithmetic_progression_probability_l825_82539

theorem arithmetic_progression_probability (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_outcomes = 6^4 ∧ favorable_outcomes = 3 →
  favorable_outcomes / total_outcomes = 1 / 432 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_probability_l825_82539


namespace NUMINAMATH_GPT_f_zero_f_odd_f_not_decreasing_f_increasing_l825_82565

noncomputable def f (x : ℝ) : ℝ := sorry -- The function definition is abstract.

-- Functional equation condition
axiom functional_eq (x y : ℝ) (h1 : -1 < x) (h2 : x < 1) (h3 : -1 < y) (h4 : y < 1) : 
  f x + f y = f ((x + y) / (1 + x * y))

-- Condition for negative interval
axiom neg_interval (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : f x < 0

-- Statements to prove

-- a): f(0) = 0
theorem f_zero : f 0 = 0 := 
by
  sorry

-- b): f(x) is an odd function
theorem f_odd (x : ℝ) (h1 : -1 < x) (h2 : x < 1) : f (-x) = -f x := 
by
  sorry

-- c): f(x) is not a decreasing function
theorem f_not_decreasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : ¬(f x1 > f x2) :=
by
  sorry

-- d): f(x) is an increasing function
theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : f x1 < f x2 :=
by
  sorry

end NUMINAMATH_GPT_f_zero_f_odd_f_not_decreasing_f_increasing_l825_82565


namespace NUMINAMATH_GPT_tenth_term_is_correct_l825_82599

-- Conditions and calculation
variable (a l : ℚ)
variable (d : ℚ)
variable (a10 : ℚ)

-- Setting the given values:
noncomputable def first_term : ℚ := 2 / 3
noncomputable def seventeenth_term : ℚ := 3 / 2
noncomputable def common_difference : ℚ := (seventeenth_term - first_term) / 16

-- Calculate the tenth term using the common difference
noncomputable def tenth_term : ℚ := first_term + 9 * common_difference

-- Statement to prove
theorem tenth_term_is_correct : 
  first_term = 2 / 3 →
  seventeenth_term = 3 / 2 →
  common_difference = (3 / 2 - 2 / 3) / 16 →
  tenth_term = 2 / 3 + 9 * ((3 / 2 - 2 / 3) / 16) →
  tenth_term = 109 / 96 :=
  by
    sorry

end NUMINAMATH_GPT_tenth_term_is_correct_l825_82599


namespace NUMINAMATH_GPT_ice_cream_ordering_ways_l825_82524

def number_of_cone_choices : ℕ := 2
def number_of_flavor_choices : ℕ := 4

theorem ice_cream_ordering_ways : number_of_cone_choices * number_of_flavor_choices = 8 := by
  sorry

end NUMINAMATH_GPT_ice_cream_ordering_ways_l825_82524


namespace NUMINAMATH_GPT_baker_bought_131_new_cakes_l825_82549

def number_of_new_cakes_bought (initial_cakes: ℕ) (cakes_sold: ℕ) (excess_sold: ℕ): ℕ :=
    cakes_sold - excess_sold - initial_cakes

theorem baker_bought_131_new_cakes :
    number_of_new_cakes_bought 8 145 6 = 131 :=
by
  -- This is where the proof would normally go
  sorry

end NUMINAMATH_GPT_baker_bought_131_new_cakes_l825_82549


namespace NUMINAMATH_GPT_scientific_notation_of_neg_0_000008691_l825_82553

theorem scientific_notation_of_neg_0_000008691:
  -0.000008691 = -8.691 * 10^(-6) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_neg_0_000008691_l825_82553


namespace NUMINAMATH_GPT_count_non_decreasing_digits_of_12022_l825_82575

/-- Proof that the number of digits left in the number 12022 that form a non-decreasing sequence is 3. -/
theorem count_non_decreasing_digits_of_12022 : 
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2] -- non-decreasing sequence from 12022
  List.length remaining = 3 :=
by
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2]
  have h : List.length remaining = 3 := rfl
  exact h

end NUMINAMATH_GPT_count_non_decreasing_digits_of_12022_l825_82575


namespace NUMINAMATH_GPT_extrema_range_l825_82503

noncomputable def hasExtrema (a : ℝ) : Prop :=
  (4 * a^2 + 12 * a > 0)

theorem extrema_range (a : ℝ) : hasExtrema a ↔ (a < -3 ∨ a > 0) := sorry

end NUMINAMATH_GPT_extrema_range_l825_82503


namespace NUMINAMATH_GPT_determine_parabola_l825_82527

-- Define the parabola passing through point P(1,1)
def parabola_passing_through (a b c : ℝ) :=
  (1:ℝ)^2 * a + 1 * b + c = 1

-- Define the condition that the tangent line at Q(2, -1) has a slope parallel to y = x - 3, which means slope = 1
def tangent_slope_at_Q (a b : ℝ) :=
  4 * a + b = 1

-- Define the parabola passing through point Q(2, -1)
def parabola_passing_through_Q (a b c : ℝ) :=
  (2:ℝ)^2 * a + (2:ℝ) * b + c = -1

-- The proof statement
theorem determine_parabola (a b c : ℝ):
  parabola_passing_through a b c ∧ 
  tangent_slope_at_Q a b ∧ 
  parabola_passing_through_Q a b c → 
  a = 3 ∧ b = -11 ∧ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_determine_parabola_l825_82527


namespace NUMINAMATH_GPT_tan_product_pi_nine_l825_82568

theorem tan_product_pi_nine (t1 t2 t4 : ℝ) 
  (h1 : t1 = Real.tan (Real.pi / 9)) 
  (h2 : t2 = Real.tan (2 * Real.pi / 9))
  (h4 : t4 = Real.tan (4 * Real.pi / 9))
  (hpoly : ∀ t : ℝ, 9 * t - 84 * t^3 + 126 * t^5 - 36 * t^7 + t^9 = 0) :
  t1 * t2 * t4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_pi_nine_l825_82568


namespace NUMINAMATH_GPT_sum_of_first_cards_l825_82533

variables (a b c d : ℕ)

theorem sum_of_first_cards (a b c d : ℕ) : 
  ∃ x, x = b * (c + 1) + d - a :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_cards_l825_82533


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l825_82592

theorem problem1 : (5 / 16) - (3 / 16) + (7 / 16) = 9 / 16 := by
  sorry

theorem problem2 : (3 / 12) - (4 / 12) + (6 / 12) = 5 / 12 := by
  sorry

theorem problem3 : 64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 := by
  sorry

theorem problem4 : (2 : ℚ) - (8 / 9) - (1 / 9) + (1 + 98 / 99) = 2 + 98 / 99 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l825_82592


namespace NUMINAMATH_GPT_pool_cleaning_l825_82577

theorem pool_cleaning (full_capacity_liters : ℕ) (percent_full : ℕ) (loss_per_jump_ml : ℕ) 
    (full_capacity : full_capacity_liters = 2000) (trigger_clean : percent_full = 80) 
    (loss_per_jump : loss_per_jump_ml = 400) : 
    let trigger_capacity_liters := (full_capacity_liters * percent_full) / 100
    let splash_out_capacity_liters := full_capacity_liters - trigger_capacity_liters
    let splash_out_capacity_ml := splash_out_capacity_liters * 1000
    (splash_out_capacity_ml / loss_per_jump_ml) = 1000 :=
by {
    sorry
}

end NUMINAMATH_GPT_pool_cleaning_l825_82577


namespace NUMINAMATH_GPT_box_problem_l825_82520

theorem box_problem 
    (x y : ℕ) 
    (h1 : 10 * x + 20 * y = 18 * (x + y)) 
    (h2 : 10 * x + 20 * (y - 10) = 16 * (x + y - 10)) :
    x + y = 20 :=
sorry

end NUMINAMATH_GPT_box_problem_l825_82520


namespace NUMINAMATH_GPT_probability_two_faces_no_faces_l825_82501

theorem probability_two_faces_no_faces :
  let side_length := 5
  let total_cubes := side_length ^ 3
  let painted_faces := 2 * (side_length ^ 2)
  let two_painted_faces := 16
  let no_painted_faces := total_cubes - painted_faces + two_painted_faces
  (two_painted_faces = 16) →
  (no_painted_faces = 91) →
  -- Total ways to choose 2 cubes from 125
  let total_ways := (total_cubes * (total_cubes - 1)) / 2
  -- Ways to choose 1 cube with 2 painted faces and 1 with no painted faces
  let successful_ways := two_painted_faces * no_painted_faces
  (successful_ways = 1456) →
  (total_ways = 7750) →
  -- The desired probability
  let probability := successful_ways / (total_ways : ℝ)
  probability = 4 / 21 :=
by
  intros side_length total_cubes painted_faces two_painted_faces no_painted_faces h1 h2 total_ways successful_ways h3 h4 probability
  sorry

end NUMINAMATH_GPT_probability_two_faces_no_faces_l825_82501


namespace NUMINAMATH_GPT_largest_integer_solution_l825_82582

theorem largest_integer_solution (x : ℤ) : 
  (x - 3 * (x - 2) ≥ 4) → (2 * x + 1 < x - 1) → (x = -3) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_solution_l825_82582


namespace NUMINAMATH_GPT_total_seeds_grace_can_plant_l825_82583

theorem total_seeds_grace_can_plant :
  let lettuce_seeds_per_row := 25
  let carrot_seeds_per_row := 20
  let radish_seeds_per_row := 30
  let large_bed_rows_limit := 5
  let medium_bed_rows_limit := 3
  let small_bed_rows_limit := 2
  let large_beds := 2
  let medium_beds := 2
  let small_bed := 1
  let large_bed_planting := 
    [(3, lettuce_seeds_per_row), (2, carrot_seeds_per_row)]  -- 3 rows of lettuce, 2 rows of carrots in large beds
  let medium_bed_planting := 
    [(1, lettuce_seeds_per_row), (1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in medium beds
  let small_bed_planting := 
    [(1, carrot_seeds_per_row), (1, radish_seeds_per_row)] --in small beds
  (3 * lettuce_seeds_per_row + 2 * carrot_seeds_per_row) * large_beds +
  (1 * lettuce_seeds_per_row + 1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * medium_beds +
  (1 * carrot_seeds_per_row + 1 * radish_seeds_per_row) * small_bed = 430 :=
by
  sorry

end NUMINAMATH_GPT_total_seeds_grace_can_plant_l825_82583


namespace NUMINAMATH_GPT_range_of_a_l825_82540

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (2^x - 2^(-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (2^x + 2^(-x))

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) ↔ a ≥ -17 / 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l825_82540


namespace NUMINAMATH_GPT_monthly_interest_payment_l825_82585

theorem monthly_interest_payment (principal : ℝ) (annual_rate : ℝ) (months_in_year : ℝ) : 
  principal = 31200 → 
  annual_rate = 0.09 → 
  months_in_year = 12 → 
  (principal * annual_rate) / months_in_year = 234 := 
by 
  intros h_principal h_rate h_months
  rw [h_principal, h_rate, h_months]
  sorry

end NUMINAMATH_GPT_monthly_interest_payment_l825_82585


namespace NUMINAMATH_GPT_space_filled_with_rhombic_dodecahedra_l825_82547

/-
  Given: Space can be filled completely using cubic cells (cubic lattice).
  To Prove: Space can be filled completely using rhombic dodecahedron cells.
-/

theorem space_filled_with_rhombic_dodecahedra :
  (∀ (cubic_lattice : Type), (∃ fill_space_with_cubes : (cubic_lattice → Prop), 
    ∀ x : cubic_lattice, fill_space_with_cubes x)) →
  (∃ (rhombic_dodecahedra_lattice : Type), 
      (∀ fill_space_with_rhombic_dodecahedra : rhombic_dodecahedra_lattice → Prop, 
        ∀ y : rhombic_dodecahedra_lattice, fill_space_with_rhombic_dodecahedra y)) :=
by {
  sorry
}

end NUMINAMATH_GPT_space_filled_with_rhombic_dodecahedra_l825_82547


namespace NUMINAMATH_GPT_LCM_is_4199_l825_82587

theorem LCM_is_4199 :
  let beats_of_cymbals := 13
  let beats_of_triangle := 17
  let beats_of_tambourine := 19
  Nat.lcm (Nat.lcm beats_of_cymbals beats_of_triangle) beats_of_tambourine = 4199 := 
by 
  sorry 

end NUMINAMATH_GPT_LCM_is_4199_l825_82587


namespace NUMINAMATH_GPT_new_phone_plan_cost_l825_82504

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.30
def new_plan_cost := old_plan_cost + (increase_percentage * old_plan_cost)

theorem new_phone_plan_cost : new_plan_cost = 195 := by
  -- From the condition that the old plan cost is $150 and the increase percentage is 30%
  -- We should prove that the new plan cost is $195
  sorry

end NUMINAMATH_GPT_new_phone_plan_cost_l825_82504


namespace NUMINAMATH_GPT_gcd_of_36_between_70_and_85_is_81_l825_82523

theorem gcd_of_36_between_70_and_85_is_81 {n : ℕ} (h1 : n ≥ 70) (h2 : n ≤ 85) (h3 : Nat.gcd 36 n = 9) : n = 81 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_gcd_of_36_between_70_and_85_is_81_l825_82523


namespace NUMINAMATH_GPT_inradius_of_right_triangle_l825_82541

variable (a b c : ℕ) -- Define the sides
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

noncomputable def area (a b : ℕ) : ℝ :=
  0.5 * (a : ℝ) * (b : ℝ)

noncomputable def semiperimeter (a b c : ℕ) : ℝ :=
  ((a + b + c) : ℝ) / 2

noncomputable def inradius (a b c : ℕ) : ℝ :=
  let s := semiperimeter a b c
  let A := area a b
  A / s

theorem inradius_of_right_triangle (h : right_triangle 7 24 25) : inradius 7 24 25 = 3 := by
  sorry

end NUMINAMATH_GPT_inradius_of_right_triangle_l825_82541


namespace NUMINAMATH_GPT_sugar_needed_for_40_cookies_l825_82525

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_sugar_needed_for_40_cookies_l825_82525


namespace NUMINAMATH_GPT_count_3_digit_numbers_divisible_by_13_l825_82569

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end NUMINAMATH_GPT_count_3_digit_numbers_divisible_by_13_l825_82569


namespace NUMINAMATH_GPT_h_comp_h_3_l825_82519

def h (x : ℕ) : ℕ := 3 * x * x + 5 * x - 3

theorem h_comp_h_3 : h (h 3) = 4755 := by
  sorry

end NUMINAMATH_GPT_h_comp_h_3_l825_82519


namespace NUMINAMATH_GPT_line_symmetric_to_itself_l825_82590

theorem line_symmetric_to_itself :
  ∀ x y : ℝ, y = 3 * x + 3 ↔ ∃ (m b : ℝ), y = m * x + b ∧ m = 3 ∧ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_line_symmetric_to_itself_l825_82590


namespace NUMINAMATH_GPT_num_valid_k_l825_82595

/--
The number of natural numbers \( k \), not exceeding 485000, 
such that \( k^2 - 1 \) is divisible by 485 is 4000.
-/
theorem num_valid_k (n : ℕ) (h₁ : n ≤ 485000) (h₂ : 485 ∣ (n^2 - 1)) : 
  (∃ k : ℕ, k = 4000) :=
sorry

end NUMINAMATH_GPT_num_valid_k_l825_82595


namespace NUMINAMATH_GPT_difference_of_squares_l825_82532

theorem difference_of_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x y : ℤ, a = x^2 - y^2) ∨ 
  (∃ x y : ℤ, b = x^2 - y^2) ∨ 
  (∃ x y : ℤ, a + b = x^2 - y^2) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l825_82532


namespace NUMINAMATH_GPT_num_boys_l825_82598

theorem num_boys (total_students : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) (r : girls_ratio = 4) (b : boys_ratio = 3) (o : others_ratio = 2) (total_eq : girls_ratio * k + boys_ratio * k + others_ratio * k = total_students) (total_given : total_students = 63) : 
  boys_ratio * k = 21 :=
by
  sorry

end NUMINAMATH_GPT_num_boys_l825_82598


namespace NUMINAMATH_GPT_max_triangle_area_l825_82502

noncomputable def max_area_of_triangle (a b c S : ℝ) : ℝ := 
if h : 4 * S = a^2 - (b - c)^2 ∧ b + c = 4 then 
  2 
else
  sorry

-- The statement we want to prove
theorem max_triangle_area : ∀ (a b c S : ℝ),
  (4 * S = a^2 - (b - c)^2) →
  (b + c = 4) →
  S ≤ max_area_of_triangle a b c S ∧ max_area_of_triangle a b c S = 2 :=
by sorry

end NUMINAMATH_GPT_max_triangle_area_l825_82502


namespace NUMINAMATH_GPT_problem_statement_l825_82571

-- Define the set of numbers
def num_set := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_multiple (a b : ℕ) : Prop := b ∣ a

-- Problem statement
theorem problem_statement (al bill cal : ℕ) (h_al : al ∈ num_set) (h_bill : bill ∈ num_set) (h_cal : cal ∈ num_set) (h_distinct: distinct al bill cal) : 
  (is_multiple al bill) ∧ (is_multiple bill cal) →
  ∃ (p : ℚ), p = 1 / 190 :=
sorry

end NUMINAMATH_GPT_problem_statement_l825_82571


namespace NUMINAMATH_GPT_endpoint_coordinates_l825_82574

theorem endpoint_coordinates (x y : ℝ) (h : y > 0) :
  let slope_condition := (y - 2) / (x - 2) = 3 / 4
  let distance_condition := (x - 2) ^ 2 + (y - 2) ^ 2 = 64
  slope_condition → distance_condition → 
    (x = 2 + (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 + (4 * Real.sqrt 5475) / 25) + 1 / 2) ∨
    (x = 2 - (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 - (4 * Real.sqrt 5475) / 25) + 1 / 2) :=
by
  intros slope_condition distance_condition
  sorry

end NUMINAMATH_GPT_endpoint_coordinates_l825_82574


namespace NUMINAMATH_GPT_correct_relationships_l825_82557

open Real

theorem correct_relationships (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1/a < 1/b) := by
    sorry

end NUMINAMATH_GPT_correct_relationships_l825_82557


namespace NUMINAMATH_GPT_denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l825_82563

variable (DenyMotion : Prop) (AcknowledgeStillness : Prop) (LeadsToRelativism : Prop)
variable (LeadsToSophistry : Prop)

theorem denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry
  (h1 : DenyMotion)
  (h2 : AcknowledgeStillness)
  (h3 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToRelativism)
  (h4 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToSophistry):
  ¬ (DenyMotion ∧ AcknowledgeStillness → LeadsToRelativism ∧ LeadsToSophistry) :=
by sorry

end NUMINAMATH_GPT_denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l825_82563


namespace NUMINAMATH_GPT_number_of_possible_triples_l825_82556

-- Given conditions
variables (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)

-- Revenue equation
def revenue_equation : Prop := 10 * x + 5 * y + z = 120

-- Proving the solution
theorem number_of_possible_triples (h : revenue_equation x y z) : 
  ∃ (n : ℕ), n = 121 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_triples_l825_82556


namespace NUMINAMATH_GPT_simplify_expr_l825_82514

-- Define the condition
def y : ℕ := 77

-- Define the expression and the expected result
def expr := (7 * y + 77) / 77

-- The theorem statement
theorem simplify_expr : expr = 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l825_82514


namespace NUMINAMATH_GPT_pq_true_l825_82506

open Real

def p : Prop := ∃ x0 : ℝ, tan x0 = sqrt 3

def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_true : p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_pq_true_l825_82506


namespace NUMINAMATH_GPT_certainEvent_l825_82551

def scoopingTheMoonOutOfTheWaterMeansCertain : Prop :=
  ∀ (e : String), e = "scooping the moon out of the water" → (∀ (b : Bool), b = true)

theorem certainEvent (e : String) (h : e = "scooping the moon out of the water") : ∀ (b : Bool), b = true :=
  by
  sorry

end NUMINAMATH_GPT_certainEvent_l825_82551
