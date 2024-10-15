import Mathlib

namespace NUMINAMATH_GPT_brown_loss_percentage_is_10_l1384_138472

-- Define the initial conditions
def initialHousePrice : ℝ := 100000
def profitPercentage : ℝ := 0.10
def sellingPriceBrown : ℝ := 99000

-- Compute the price Mr. Brown bought the house
def priceBrownBought := initialHousePrice * (1 + profitPercentage)

-- Define the loss percentage as a goal to prove
theorem brown_loss_percentage_is_10 :
  ((priceBrownBought - sellingPriceBrown) / priceBrownBought) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_brown_loss_percentage_is_10_l1384_138472


namespace NUMINAMATH_GPT_inequality_a2_b2_c2_geq_abc_l1384_138483

theorem inequality_a2_b2_c2_geq_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_cond: a + b + c ≥ a * b * c) :
  a^2 + b^2 + c^2 ≥ a * b * c := 
sorry

end NUMINAMATH_GPT_inequality_a2_b2_c2_geq_abc_l1384_138483


namespace NUMINAMATH_GPT_dealership_sales_l1384_138468

theorem dealership_sales (sports_cars : ℕ) (sedans : ℕ) (trucks : ℕ) 
  (h1 : sports_cars = 36)
  (h2 : (3 : ℤ) * sedans = 5 * sports_cars)
  (h3 : (3 : ℤ) * trucks = 4 * sports_cars) :
  sedans = 60 ∧ trucks = 48 := 
sorry

end NUMINAMATH_GPT_dealership_sales_l1384_138468


namespace NUMINAMATH_GPT_smallest_solution_floor_eq_l1384_138412

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end NUMINAMATH_GPT_smallest_solution_floor_eq_l1384_138412


namespace NUMINAMATH_GPT_sequence_inequality_l1384_138478

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (m n : ℕ)
  (h1 : a 1 = 21/16)
  (h2 : ∀ n ≥ 2, 2 * a n - 3 * a (n - 1) = 3 / 2^(n + 1))
  (h3 : m ≥ 2)
  (h4 : n ≤ m) :
  (a n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^(n * (m - 1) / m)) < (m^2 - 1) / (m - n + 1) :=
sorry

end NUMINAMATH_GPT_sequence_inequality_l1384_138478


namespace NUMINAMATH_GPT_original_cost_l1384_138462

theorem original_cost (P : ℝ) (h : 0.85 * 0.76 * P = 988) : P = 1529.41 := by
  sorry

end NUMINAMATH_GPT_original_cost_l1384_138462


namespace NUMINAMATH_GPT_circle_equation_l1384_138408

theorem circle_equation (a : ℝ) (h : a = 1) :
  (∀ (C : ℝ × ℝ), C = (a, a) →
  (∀ (r : ℝ), r = dist C (1, 0) →
  r = 1 → ((x - a) ^ 2 + (y - a) ^ 2 = r ^ 2))) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1384_138408


namespace NUMINAMATH_GPT_new_paint_intensity_l1384_138439

variable (V : ℝ)  -- V is the volume of the original 50% intensity red paint.
variable (I₁ I₂ : ℝ)  -- I₁ is the intensity of the original paint, I₂ is the intensity of the replaced paint.
variable (f : ℝ)  -- f is the fraction of the original paint being replaced.

-- Assume given conditions
axiom intensity_original : I₁ = 0.5
axiom intensity_new : I₂ = 0.25
axiom fraction_replaced : f = 0.8

-- Prove that the new intensity is 30%
theorem new_paint_intensity :
  (f * I₂ + (1 - f) * I₁) = 0.3 := 
by 
  -- This is the main theorem we want to prove
  sorry

end NUMINAMATH_GPT_new_paint_intensity_l1384_138439


namespace NUMINAMATH_GPT_transaction_mistake_in_cents_l1384_138498

theorem transaction_mistake_in_cents
  (x y : ℕ)
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (error_cents : 100 * y + x - (100 * x + y) = 5616) :
  y = x + 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_transaction_mistake_in_cents_l1384_138498


namespace NUMINAMATH_GPT_triangle_internal_angle_A_l1384_138454

theorem triangle_internal_angle_A {B C A : ℝ} (hB : Real.tan B = -2) (hC : Real.tan C = 1 / 3) (h_sum: A = π - B - C) : A = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_internal_angle_A_l1384_138454


namespace NUMINAMATH_GPT_scrabble_middle_letter_value_l1384_138437

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end NUMINAMATH_GPT_scrabble_middle_letter_value_l1384_138437


namespace NUMINAMATH_GPT_mathematician_daily_questions_l1384_138411

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end NUMINAMATH_GPT_mathematician_daily_questions_l1384_138411


namespace NUMINAMATH_GPT_total_amount_spent_l1384_138404

-- Definitions for problem conditions
def mall_spent_before_discount : ℝ := 250
def clothes_discount_percent : ℝ := 0.15
def mall_tax_percent : ℝ := 0.08

def movie_ticket_price : ℝ := 24
def num_movies : ℝ := 3
def ticket_discount_percent : ℝ := 0.10
def movie_tax_percent : ℝ := 0.05

def beans_price : ℝ := 1.25
def num_beans : ℝ := 20
def cucumber_price : ℝ := 2.50
def num_cucumbers : ℝ := 5
def tomato_price : ℝ := 5.00
def num_tomatoes : ℝ := 3
def pineapple_price : ℝ := 6.50
def num_pineapples : ℝ := 2
def market_tax_percent : ℝ := 0.07

-- Proof statement
theorem total_amount_spent :
  let mall_spent_after_discount := mall_spent_before_discount * (1 - clothes_discount_percent)
  let mall_tax := mall_spent_after_discount * mall_tax_percent
  let total_mall_spent := mall_spent_after_discount + mall_tax

  let total_ticket_cost_before_discount := num_movies * movie_ticket_price
  let ticket_cost_after_discount := total_ticket_cost_before_discount * (1 - ticket_discount_percent)
  let movie_tax := ticket_cost_after_discount * movie_tax_percent
  let total_movie_spent := ticket_cost_after_discount + movie_tax

  let total_beans_cost := num_beans * beans_price
  let total_cucumbers_cost := num_cucumbers * cucumber_price
  let total_tomatoes_cost := num_tomatoes * tomato_price
  let total_pineapples_cost := num_pineapples * pineapple_price
  let total_market_spent_before_tax := total_beans_cost + total_cucumbers_cost + total_tomatoes_cost + total_pineapples_cost
  let market_tax := total_market_spent_before_tax * market_tax_percent
  let total_market_spent := total_market_spent_before_tax + market_tax
  
  let total_spent := total_mall_spent + total_movie_spent + total_market_spent
  total_spent = 367.63 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1384_138404


namespace NUMINAMATH_GPT_vertex_coordinates_l1384_138402

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := (x + 3) ^ 2 - 1

-- Define the statement for the coordinates of the vertex of the parabola
theorem vertex_coordinates : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x + 3) ^ 2 - 1) ∧ h = -3 ∧ k = -1 := 
  sorry

end NUMINAMATH_GPT_vertex_coordinates_l1384_138402


namespace NUMINAMATH_GPT_volume_of_intersecting_octahedra_l1384_138438

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_volume_of_intersecting_octahedra_l1384_138438


namespace NUMINAMATH_GPT_problem_l1384_138443

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def quadratic_roots (a₃ a₁₀ : ℝ) : Prop :=
a₃^2 - 3 * a₃ - 5 = 0 ∧ a₁₀^2 - 3 * a₁₀ - 5 = 0

theorem problem (a : ℕ → ℝ) (h1 : is_arithmetic_seq a)
  (h2 : quadratic_roots (a 3) (a 10)) :
  a 5 + a 8 = 3 :=
sorry

end NUMINAMATH_GPT_problem_l1384_138443


namespace NUMINAMATH_GPT_A_wins_when_n_is_9_l1384_138433

-- Definition of the game conditions and the strategy
def game (n : ℕ) (A_first : Bool) :=
  ∃ strategy : ℕ → ℕ,
    ∀ taken balls_left : ℕ,
      balls_left - taken > 0 →
      taken ≥ 1 → taken ≤ 3 →
      if A_first then
        (balls_left - taken = 0 → strategy (balls_left - taken) = 1) ∧
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)
      else
        (balls_left - taken = 0 → strategy (balls_left - taken) = 0) ∨
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)

-- Prove that for n = 9 A has a winning strategy
theorem A_wins_when_n_is_9 : game 9 true :=
sorry

end NUMINAMATH_GPT_A_wins_when_n_is_9_l1384_138433


namespace NUMINAMATH_GPT_correct_calculation_l1384_138423

theorem correct_calculation (x y a b : ℝ) :
  (3*x + 3*y ≠ 6*x*y) ∧
  (x + x ≠ x^2) ∧
  (-9*y^2 + 16*y^2 ≠ 7) ∧
  (9*a^2*b - 9*a^2*b = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1384_138423


namespace NUMINAMATH_GPT_problem1_l1384_138405

theorem problem1 :
  let total_products := 10
  let defective_products := 4
  let first_def_pos := 5
  let last_def_pos := 10
  ∃ (num_methods : Nat), num_methods = 103680 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1384_138405


namespace NUMINAMATH_GPT_find_urn_yellow_balls_l1384_138459

theorem find_urn_yellow_balls :
  ∃ (M : ℝ), 
    (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
    M = 111 := 
sorry

end NUMINAMATH_GPT_find_urn_yellow_balls_l1384_138459


namespace NUMINAMATH_GPT_trigonometric_identity_l1384_138474

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) + Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 * Real.tan (10 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1384_138474


namespace NUMINAMATH_GPT_problem_1_problem_2_l1384_138436

-- Define propositions
def prop_p (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - m) + y^2 / m = 1)

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

def prop_s (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

-- Problems
theorem problem_1 (m : ℝ) (h : prop_s m) : m < 0 ∨ m ≥ 1 := 
  sorry

theorem problem_2 {m : ℝ} (h1 : prop_p m ∨ prop_q m) (h2 : ¬ prop_q m) : 1 ≤ m ∧ m < 2 :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1384_138436


namespace NUMINAMATH_GPT_no_real_roots_quadratic_l1384_138445

theorem no_real_roots_quadratic (k : ℝ) : 
  ∀ (x : ℝ), k * x^2 - 2 * x + 1 / 2 ≠ 0 → k > 2 :=
by 
  intro x h
  have h1 : (-2)^2 - 4 * k * (1/2) < 0 := sorry
  have h2 : 4 - 2 * k < 0 := sorry
  have h3 : 2 < k := sorry
  exact h3

end NUMINAMATH_GPT_no_real_roots_quadratic_l1384_138445


namespace NUMINAMATH_GPT_negation_of_symmetry_about_y_eq_x_l1384_138420

theorem negation_of_symmetry_about_y_eq_x :
  ¬ (∀ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x) ↔ ∃ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x :=
by sorry

end NUMINAMATH_GPT_negation_of_symmetry_about_y_eq_x_l1384_138420


namespace NUMINAMATH_GPT_find_range_a_l1384_138453

noncomputable def f (a x : ℝ) : ℝ := abs (2 * x * a + abs (x - 1))

theorem find_range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 5) ↔ a ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_find_range_a_l1384_138453


namespace NUMINAMATH_GPT_geometric_sequence_product_l1384_138409

-- Define the geometric sequence sum and the initial conditions
variables {S : ℕ → ℚ} {a : ℕ → ℚ}
variables (q : ℚ) (h1 : a 1 = -1/2)
variables (h2 : S 6 / S 3 = 7 / 8)

-- The main proof problem statement
theorem geometric_sequence_product (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 2 * a 4 = 1 / 64 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1384_138409


namespace NUMINAMATH_GPT_calories_in_300g_l1384_138470

/-
Define the conditions of the problem.
-/

def lemon_juice_grams := 150
def sugar_grams := 200
def lime_juice_grams := 50
def water_grams := 500

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 390
def lime_juice_calories_per_100g := 20
def water_calories := 0

/-
Define the total weight of the beverage.
-/
def total_weight := lemon_juice_grams + sugar_grams + lime_juice_grams + water_grams

/-
Define the total calories of the beverage.
-/
def total_calories := 
  (lemon_juice_calories_per_100g * lemon_juice_grams / 100) + 
  (sugar_calories_per_100g * sugar_grams / 100) + 
  (lime_juice_calories_per_100g * lime_juice_grams / 100) + 
  water_calories

/-
Prove the number of calories in 300 grams of the beverage.
-/
theorem calories_in_300g : (total_calories / total_weight) * 300 = 278 := by
  sorry

end NUMINAMATH_GPT_calories_in_300g_l1384_138470


namespace NUMINAMATH_GPT_find_y_l1384_138400

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 8) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1384_138400


namespace NUMINAMATH_GPT_problem_statement_l1384_138426

/-- Let x, y, z be nonzero real numbers such that x + y + z = 0.
    Prove that ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → x + y + z = 0 → (x^3 + y^3 + z^3) / (x * y * z) = 3. -/
theorem problem_statement (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1384_138426


namespace NUMINAMATH_GPT_quadratic_equation_solution_l1384_138428

theorem quadratic_equation_solution :
  ∃ x1 x2 : ℝ, (x1 = (-1 + Real.sqrt 13) / 2 ∧ x2 = (-1 - Real.sqrt 13) / 2 
  ∧ (∀ x : ℝ, x^2 + x - 3 = 0 → x = x1 ∨ x = x2)) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l1384_138428


namespace NUMINAMATH_GPT_units_digit_specified_expression_l1384_138480

theorem units_digit_specified_expression :
  let numerator := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11)
  let denominator := 8000
  let product := numerator * 20
  (∃ d, product / denominator = d ∧ (d % 10 = 6)) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_specified_expression_l1384_138480


namespace NUMINAMATH_GPT_meaningful_sqrt_l1384_138463

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end NUMINAMATH_GPT_meaningful_sqrt_l1384_138463


namespace NUMINAMATH_GPT_rahim_average_price_l1384_138486

/-- 
Rahim bought 40 books for Rs. 600 from one shop and 20 books for Rs. 240 from another.
What is the average price he paid per book?
-/
def books1 : ℕ := 40
def cost1 : ℕ := 600
def books2 : ℕ := 20
def cost2 : ℕ := 240
def totalBooks : ℕ := books1 + books2
def totalCost : ℕ := cost1 + cost2
def averagePricePerBook : ℕ := totalCost / totalBooks

theorem rahim_average_price :
  averagePricePerBook = 14 :=
by
  sorry

end NUMINAMATH_GPT_rahim_average_price_l1384_138486


namespace NUMINAMATH_GPT_carrots_picked_by_Carol_l1384_138488

theorem carrots_picked_by_Carol (total_carrots mom_carrots : ℕ) (h1 : total_carrots = 38 + 7) (h2 : mom_carrots = 16) :
  total_carrots - mom_carrots = 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_carrots_picked_by_Carol_l1384_138488


namespace NUMINAMATH_GPT_entrance_exit_plans_l1384_138416

-- Definitions as per the conditions in the problem
def south_gates : Nat := 4
def north_gates : Nat := 3
def west_gates : Nat := 2

-- Conditions translated into Lean definitions
def ways_to_enter := south_gates + north_gates
def ways_to_exit := west_gates + north_gates

-- The theorem to be proved: the number of entrance and exit plans
theorem entrance_exit_plans : ways_to_enter * ways_to_exit = 35 := by
  sorry

end NUMINAMATH_GPT_entrance_exit_plans_l1384_138416


namespace NUMINAMATH_GPT_friends_with_john_l1384_138469

def total_slices (pizzas slices_per_pizza : Nat) : Nat := pizzas * slices_per_pizza

def total_people (total_slices slices_per_person : Nat) : Nat := total_slices / slices_per_person

def number_of_friends (total_people john : Nat) : Nat := total_people - john

theorem friends_with_john (pizzas slices_per_pizza slices_per_person john friends : Nat) (h_pizzas : pizzas = 3) 
                          (h_slices_per_pizza : slices_per_pizza = 8) (h_slices_per_person : slices_per_person = 4)
                          (h_john : john = 1) (h_friends : friends = 5) :
  number_of_friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) john = friends := by
  sorry

end NUMINAMATH_GPT_friends_with_john_l1384_138469


namespace NUMINAMATH_GPT_regions_formula_l1384_138419

-- Define the number of regions R(n) created by n lines
def regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

-- Theorem statement: for n lines, no two parallel, no three concurrent, the regions are defined by the formula
theorem regions_formula (n : ℕ) : regions n = 1 + (n * (n + 1)) / 2 := 
by sorry

end NUMINAMATH_GPT_regions_formula_l1384_138419


namespace NUMINAMATH_GPT_area_of_triangle_is_11_25_l1384_138424

noncomputable def area_of_triangle : ℝ :=
  let A := (1 / 2, 2)
  let B := (8, 2)
  let C := (2, 5)
  let base := (B.1 - A.1 : ℝ)
  let height := (C.2 - A.2 : ℝ)
  0.5 * base * height

theorem area_of_triangle_is_11_25 :
  area_of_triangle = 11.25 := sorry

end NUMINAMATH_GPT_area_of_triangle_is_11_25_l1384_138424


namespace NUMINAMATH_GPT_product_of_three_greater_than_product_of_two_or_four_l1384_138475

theorem product_of_three_greater_than_product_of_two_or_four
  (nums : Fin 10 → ℝ)
  (h_positive : ∀ i, 0 < nums i)
  (h_distinct : Function.Injective nums) :
  ∃ (a b c : Fin 10),
    (∃ (d e : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (c ≠ d) ∧ (c ≠ e) ∧ nums a * nums b * nums c > nums d * nums e) ∨
    (∃ (d e f g : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ nums a * nums b * nums c > nums d * nums e * nums f * nums g) :=
sorry

end NUMINAMATH_GPT_product_of_three_greater_than_product_of_two_or_four_l1384_138475


namespace NUMINAMATH_GPT_polygon_diagonals_l1384_138413

-- Lean statement of the problem

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 2018) : n = 2021 :=
  by sorry

end NUMINAMATH_GPT_polygon_diagonals_l1384_138413


namespace NUMINAMATH_GPT_derivative_of_f_domain_of_f_range_of_f_l1384_138441

open Real

noncomputable def f (x : ℝ) := 1 / (x + sqrt (1 + 2 * x^2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = - ((sqrt (1 + 2 * x^2) + 2 * x) / (sqrt (1 + 2 * x^2) * (x + sqrt (1 + 2 * x^2))^2)) :=
by
  sorry

theorem domain_of_f : ∀ x : ℝ, f x ≠ 0 :=
by
  sorry

theorem range_of_f : 
  ∀ y : ℝ, 0 < y ∧ y ≤ sqrt 2 → ∃ x : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_f_domain_of_f_range_of_f_l1384_138441


namespace NUMINAMATH_GPT_find_dividing_line_l1384_138477

/--
A line passing through point P(1,1) divides the circular region \{(x, y) \mid x^2 + y^2 \leq 4\} into two parts,
making the difference in area between these two parts the largest. Prove that the equation of this line is x + y - 2 = 0.
-/
theorem find_dividing_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ (A B C : ℝ), A * 1 + B * 1 + C = 0 ∧
                 (∀ x y, x^2 + y^2 ≤ 4 → A * x + B * y + C = 0 → (x + y - 2) = 0) :=
sorry

end NUMINAMATH_GPT_find_dividing_line_l1384_138477


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_23_mod_89_is_805_l1384_138497

theorem smallest_positive_multiple_of_23_mod_89_is_805 : 
  ∃ a : ℕ, 23 * a ≡ 4 [MOD 89] ∧ 23 * a = 805 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_23_mod_89_is_805_l1384_138497


namespace NUMINAMATH_GPT_percentage_increase_in_rent_l1384_138457

theorem percentage_increase_in_rent
  (avg_rent_per_person_before : ℝ)
  (num_friends : ℕ)
  (friend_original_rent : ℝ)
  (avg_rent_per_person_after : ℝ)
  (total_rent_before : ℝ := num_friends * avg_rent_per_person_before)
  (total_rent_after : ℝ := num_friends * avg_rent_per_person_after)
  (rent_increase : ℝ := total_rent_after - total_rent_before)
  (percentage_increase : ℝ := (rent_increase / friend_original_rent) * 100)
  (h1 : avg_rent_per_person_before = 800)
  (h2 : num_friends = 4)
  (h3 : friend_original_rent = 1400)
  (h4 : avg_rent_per_person_after = 870) :
  percentage_increase = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_rent_l1384_138457


namespace NUMINAMATH_GPT_initial_amount_of_water_l1384_138422

theorem initial_amount_of_water 
  (W : ℚ) 
  (h1 : W - (7/15) * W - (5/8) * (W - (7/15) * W) - (2/3) * (W - (7/15) * W - (5/8) * (W - (7/15) * W)) = 2.6) 
  : W = 39 := 
sorry

end NUMINAMATH_GPT_initial_amount_of_water_l1384_138422


namespace NUMINAMATH_GPT_roots_of_quadratic_l1384_138447

theorem roots_of_quadratic (x : ℝ) : (5 * x^2 = 4 * x) → (x = 0 ∨ x = 4 / 5) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1384_138447


namespace NUMINAMATH_GPT_find_x_average_l1384_138499

theorem find_x_average :
  ∃ x : ℝ, (x + 8 + (7 * x - 3) + (3 * x + 10) + (-x + 6)) / 4 = 5 * x - 4 ∧ x = 3.7 :=
  by
  use 3.7
  sorry

end NUMINAMATH_GPT_find_x_average_l1384_138499


namespace NUMINAMATH_GPT_percentage_of_blue_flowers_l1384_138473

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

end NUMINAMATH_GPT_percentage_of_blue_flowers_l1384_138473


namespace NUMINAMATH_GPT_trigonometric_relationship_l1384_138435

theorem trigonometric_relationship :
  let a := [10, 9, 8, 7, 6, 4, 3, 2, 1]
  let sum_of_a := a.sum
  let x := Real.sin sum_of_a
  let y := Real.cos sum_of_a
  let z := Real.tan sum_of_a
  sum_of_a = 50 →
  z < x ∧ x < y :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_relationship_l1384_138435


namespace NUMINAMATH_GPT_contradiction_method_l1384_138414

theorem contradiction_method (x y : ℝ) (h : x + y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end NUMINAMATH_GPT_contradiction_method_l1384_138414


namespace NUMINAMATH_GPT_coplanar_lines_l1384_138493

def vector3 := ℝ × ℝ × ℝ

def vec1 : vector3 := (2, -1, 3)
def vec2 (k : ℝ) : vector3 := (3 * k, 1, 2)
def pointVec : vector3 := (-3, 2, -3)

def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem coplanar_lines (k : ℝ) : det3x3 2 (-1) 3 (3 * k) 1 2 (-3) 2 (-3) = 0 → k = -29 / 9 :=
  sorry

end NUMINAMATH_GPT_coplanar_lines_l1384_138493


namespace NUMINAMATH_GPT_range_of_x_l1384_138406

variable (x y : ℝ)

theorem range_of_x (h1 : 2 * x - y = 4) (h2 : -2 < y ∧ y ≤ 3) :
  1 < x ∧ x ≤ 7 / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_x_l1384_138406


namespace NUMINAMATH_GPT_min_value_of_f_l1384_138452

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (2 * x / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 1 + 2 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_min_value_of_f_l1384_138452


namespace NUMINAMATH_GPT_washing_machines_removed_per_box_l1384_138458

theorem washing_machines_removed_per_box 
  (crates : ℕ) (boxes_per_crate : ℕ) (washing_machines_per_box : ℕ) 
  (total_removed : ℕ) (total_crates : ℕ) (total_boxes_per_crate : ℕ) 
  (total_washing_machines_per_box : ℕ) 
  (h1 : crates = total_crates) (h2 : boxes_per_crate = total_boxes_per_crate) 
  (h3 : washing_machines_per_box = total_washing_machines_per_box) 
  (h4 : total_removed = 60) (h5 : total_crates = 10) 
  (h6 : total_boxes_per_crate = 6) 
  (h7 : total_washing_machines_per_box = 4):
  total_removed / (total_crates * total_boxes_per_crate) = 1 :=
by
  sorry

end NUMINAMATH_GPT_washing_machines_removed_per_box_l1384_138458


namespace NUMINAMATH_GPT_num_pos_int_x_l1384_138496

theorem num_pos_int_x (x : ℕ) : 
  (30 < x^2 + 5 * x + 10) ∧ (x^2 + 5 * x + 10 < 60) ↔ x = 3 ∨ x = 4 ∨ x = 5 := 
sorry

end NUMINAMATH_GPT_num_pos_int_x_l1384_138496


namespace NUMINAMATH_GPT_alloy_price_per_kg_l1384_138450

theorem alloy_price_per_kg (cost_A cost_B ratio_A_B total_cost total_weight price_per_kg : ℤ)
  (hA : cost_A = 68) 
  (hB : cost_B = 96) 
  (hRatio : ratio_A_B = 3) 
  (hTotalCost : total_cost = 3 * cost_A + cost_B) 
  (hTotalWeight : total_weight = 3 + 1)
  (hPricePerKg : price_per_kg = total_cost / total_weight) : 
  price_per_kg = 75 := 
by
  sorry

end NUMINAMATH_GPT_alloy_price_per_kg_l1384_138450


namespace NUMINAMATH_GPT_eiffel_tower_scale_l1384_138401

theorem eiffel_tower_scale (height_tower_m : ℝ) (height_model_cm : ℝ) :
    height_tower_m = 324 →
    height_model_cm = 50 →
    (height_tower_m * 100) / height_model_cm = 648 →
    (648 / 100) = 6.48 :=
by
  intro h_tower h_model h_ratio
  rw [h_tower, h_model] at h_ratio
  sorry

end NUMINAMATH_GPT_eiffel_tower_scale_l1384_138401


namespace NUMINAMATH_GPT_continuity_at_2_l1384_138491

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem continuity_at_2 (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) → b = 9 :=
by
  sorry  

end NUMINAMATH_GPT_continuity_at_2_l1384_138491


namespace NUMINAMATH_GPT_water_needed_l1384_138456

theorem water_needed (nutrient_concentrate : ℝ) (distilled_water : ℝ) (total_volume : ℝ) 
    (h1 : nutrient_concentrate = 0.08) (h2 : distilled_water = 0.04) (h3 : total_volume = 1) :
    total_volume * (distilled_water / (nutrient_concentrate + distilled_water)) = 0.333 :=
by
  sorry

end NUMINAMATH_GPT_water_needed_l1384_138456


namespace NUMINAMATH_GPT_solve_equation_l1384_138417

theorem solve_equation (x : ℚ) (h1 : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1384_138417


namespace NUMINAMATH_GPT_two_digit_number_ratio_l1384_138487

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b
def swapped_two_digit_number (a b : ℕ) : ℕ := 10 * b + a

theorem two_digit_number_ratio (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h_ratio : 6 * two_digit_number a b = 5 * swapped_two_digit_number a b) : 
  two_digit_number a b = 45 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_ratio_l1384_138487


namespace NUMINAMATH_GPT_calculate_value_of_A_plus_C_l1384_138492

theorem calculate_value_of_A_plus_C (A B C : ℕ) (hA : A = 238) (hAB : A = B + 143) (hBC : C = B + 304) : A + C = 637 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_of_A_plus_C_l1384_138492


namespace NUMINAMATH_GPT_cheburashkas_erased_l1384_138425

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end NUMINAMATH_GPT_cheburashkas_erased_l1384_138425


namespace NUMINAMATH_GPT_tan_alpha_cos2alpha_plus_2sin2alpha_l1384_138471

theorem tan_alpha_cos2alpha_plus_2sin2alpha (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_cos2alpha_plus_2sin2alpha_l1384_138471


namespace NUMINAMATH_GPT_tetrahedron_painting_l1384_138481

theorem tetrahedron_painting (unique_coloring_per_face : ∀ f : Fin 4, ∃ c : Fin 4, True)
  (rotation_identity : ∀ f g : Fin 4, (f = g → unique_coloring_per_face f = unique_coloring_per_face g))
  : (number_of_distinct_paintings : ℕ) = 2 :=
sorry

end NUMINAMATH_GPT_tetrahedron_painting_l1384_138481


namespace NUMINAMATH_GPT_max_4x3_y3_l1384_138466

theorem max_4x3_y3 (x y : ℝ) (h1 : x ≤ 2) (h2 : y ≤ 3) (h3 : x + y = 3) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : 
  4 * x^3 + y^3 ≤ 33 :=
sorry

end NUMINAMATH_GPT_max_4x3_y3_l1384_138466


namespace NUMINAMATH_GPT_gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l1384_138427

def a : ℕ := 2^1025 - 1
def b : ℕ := 2^1056 - 1
def answer : ℕ := 2147483647

theorem gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1 :
  Int.gcd a b = answer := by
  sorry

end NUMINAMATH_GPT_gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l1384_138427


namespace NUMINAMATH_GPT_total_writing_instruments_l1384_138495

theorem total_writing_instruments 
 (bags : ℕ) (compartments_per_bag : ℕ) (empty_compartments : ℕ) (one_compartment : ℕ) (remaining_compartments : ℕ) 
 (writing_instruments_per_compartment : ℕ) (writing_instruments_in_one : ℕ) : 
 bags = 16 → 
 compartments_per_bag = 6 → 
 empty_compartments = 5 → 
 one_compartment = 1 → 
 remaining_compartments = 90 →
 writing_instruments_per_compartment = 8 → 
 writing_instruments_in_one = 6 → 
 (remaining_compartments * writing_instruments_per_compartment + one_compartment * writing_instruments_in_one) = 726 := 
  by
   sorry

end NUMINAMATH_GPT_total_writing_instruments_l1384_138495


namespace NUMINAMATH_GPT_problem_statement_l1384_138430

noncomputable def k_value (k : ℝ) : Prop :=
  (∀ (x y : ℝ), x + y = k → x^2 + y^2 = 4) ∧ (∀ (A B : ℝ × ℝ), (∃ (x y : ℝ), A = (x, y) ∧ x^2 + y^2 = 4) ∧ (∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4) ∧ 
  (∃ (xa ya xb yb : ℝ), A = (xa, ya) ∧ B = (xb, yb) ∧ |(xa - xb, ya - yb)| = |(xa, ya)| + |(xb, yb)|)) → k = 2

theorem problem_statement (k : ℝ) (h : k > 0) : k_value k :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1384_138430


namespace NUMINAMATH_GPT_total_cost_3m3_topsoil_l1384_138410

def topsoil_cost (V C : ℕ) : ℕ :=
  V * C

theorem total_cost_3m3_topsoil : topsoil_cost 3 12 = 36 :=
by
  unfold topsoil_cost
  exact rfl

end NUMINAMATH_GPT_total_cost_3m3_topsoil_l1384_138410


namespace NUMINAMATH_GPT_remainder_of_polynomial_l1384_138460

   def polynomial_division_remainder (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

   theorem remainder_of_polynomial : polynomial_division_remainder 1 = 4 :=
   by
     -- This placeholder indicates that the proof is omitted.
     sorry
   
end NUMINAMATH_GPT_remainder_of_polynomial_l1384_138460


namespace NUMINAMATH_GPT_student_exchanges_l1384_138442

theorem student_exchanges (x : ℕ) : x * (x - 1) = 72 :=
sorry

end NUMINAMATH_GPT_student_exchanges_l1384_138442


namespace NUMINAMATH_GPT_exists_polynomials_Q_R_l1384_138415

noncomputable def polynomial_with_integer_coeff (P : Polynomial ℤ) : Prop :=
  true

theorem exists_polynomials_Q_R (P : Polynomial ℤ) (hP : polynomial_with_integer_coeff P) :
  ∃ (Q R : Polynomial ℤ), 
    (∃ g : Polynomial ℤ, P * Q = Polynomial.comp g (Polynomial.X ^ 2)) ∧ 
    (∃ h : Polynomial ℤ, P * R = Polynomial.comp h (Polynomial.X ^ 3)) :=
by
  sorry

end NUMINAMATH_GPT_exists_polynomials_Q_R_l1384_138415


namespace NUMINAMATH_GPT_power_function_solution_l1384_138465

theorem power_function_solution (m : ℤ)
  (h1 : ∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = x^(-m^2 + 2 * m + 3) ∧ ∀ x, f x = f (-x))
  (h2 : ∀ x : ℝ, x > 0 → (x^(-m^2 + 2 * m + 3)) < x^(-m^2 + 2 * m + 3 + x)) :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^4 :=
by
  sorry

end NUMINAMATH_GPT_power_function_solution_l1384_138465


namespace NUMINAMATH_GPT_one_div_abs_z_eq_sqrt_two_l1384_138451

open Complex

theorem one_div_abs_z_eq_sqrt_two (z : ℂ) (h : z = i / (1 - i)) : 1 / Complex.abs z = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_one_div_abs_z_eq_sqrt_two_l1384_138451


namespace NUMINAMATH_GPT_solve_for_x_l1384_138482

theorem solve_for_x : ∀ (x : ℝ), 
  (x + 2 * x + 3 * x + 4 * x = 5) → (x = 1 / 2) :=
by 
  intros x H
  sorry

end NUMINAMATH_GPT_solve_for_x_l1384_138482


namespace NUMINAMATH_GPT_initial_professors_l1384_138434

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_professors_l1384_138434


namespace NUMINAMATH_GPT_find_x_l1384_138485

def op (a b : ℤ) : ℤ := -2 * a + b

theorem find_x (x : ℤ) (h : op x (-5) = 3) : x = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1384_138485


namespace NUMINAMATH_GPT_last_four_digits_5_pow_2011_l1384_138449

theorem last_four_digits_5_pow_2011 : 
  (5^2011 % 10000) = 8125 :=
by
  -- Definitions based on conditions in the problem
  have h5 : 5^5 % 10000 = 3125 := sorry
  have h6 : 5^6 % 10000 = 5625 := sorry
  have h7 : 5^7 % 10000 = 8125 := sorry
  
  -- Prove using periodicity and modular arithmetic
  sorry

end NUMINAMATH_GPT_last_four_digits_5_pow_2011_l1384_138449


namespace NUMINAMATH_GPT_factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l1384_138490

-- Factorization of 4a^2 - 9 as (2a + 3)(2a - 3)
theorem factorize_4a2_minus_9 (a : ℝ) : 4 * a^2 - 9 = (2 * a + 3) * (2 * a - 3) :=
by 
  sorry

-- Factorization of 2x^2 y - 8xy + 8y as 2y(x-2)^2
theorem factorize_2x2y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2) ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l1384_138490


namespace NUMINAMATH_GPT_Calvin_insect_count_l1384_138407

theorem Calvin_insect_count:
  ∀ (roaches scorpions crickets caterpillars : ℕ), 
    roaches = 12 →
    scorpions = 3 →
    crickets = roaches / 2 →
    caterpillars = scorpions * 2 →
    roaches + scorpions + crickets + caterpillars = 27 := 
by
  intros roaches scorpions crickets caterpillars h_roaches h_scorpions h_crickets h_caterpillars
  rw [h_roaches, h_scorpions, h_crickets, h_caterpillars]
  norm_num
  sorry

end NUMINAMATH_GPT_Calvin_insect_count_l1384_138407


namespace NUMINAMATH_GPT_common_ratio_of_geom_seq_l1384_138421

-- Define the conditions: geometric sequence and the given equation
def is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_geom_seq
  (a : ℕ → ℝ)
  (h_geom : is_geom_seq a)
  (h_eq : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, is_geom_seq a ∧ q = 3 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_geom_seq_l1384_138421


namespace NUMINAMATH_GPT_roots_range_l1384_138418

theorem roots_range (b : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + b = 0 → 0 < x) ↔ 0 < b ∧ b ≤ 1 :=
sorry

end NUMINAMATH_GPT_roots_range_l1384_138418


namespace NUMINAMATH_GPT_values_for_a_l1384_138461

def has_two (A : Set ℤ) : Prop :=
  2 ∈ A

def candidate_values (a : ℤ) : Set ℤ :=
  {-2, 2 * a, a * a - a}

theorem values_for_a (a : ℤ) :
  has_two (candidate_values a) ↔ a = 1 ∨ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_values_for_a_l1384_138461


namespace NUMINAMATH_GPT_ribbon_used_l1384_138446

def total_ribbon : ℕ := 84
def leftover_ribbon : ℕ := 38
def used_ribbon : ℕ := 46

theorem ribbon_used : total_ribbon - leftover_ribbon = used_ribbon := sorry

end NUMINAMATH_GPT_ribbon_used_l1384_138446


namespace NUMINAMATH_GPT_quadrilateral_area_correct_l1384_138489

open Real
open Function
open Classical

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0)
  let B := (2, 3)
  let C := (5, 0)
  let D := (3, -2)
  let vector_cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0.5 * abs (vector_cross_product (p2 - p1) (p3 - p1))
  area_triangle A B D + area_triangle B C D

theorem quadrilateral_area_correct : quadrilateral_area = 17 / 2 :=
  sorry

end NUMINAMATH_GPT_quadrilateral_area_correct_l1384_138489


namespace NUMINAMATH_GPT_bus_patrons_correct_l1384_138444

-- Definitions corresponding to conditions
def number_of_golf_carts : ℕ := 13
def patrons_per_cart : ℕ := 3
def car_patrons : ℕ := 12

-- Multiply to get total patrons transported by golf carts
def total_patrons := number_of_golf_carts * patrons_per_cart

-- Calculate bus patrons
def bus_patrons := total_patrons - car_patrons

-- The statement to prove
theorem bus_patrons_correct : bus_patrons = 27 :=
by
  sorry

end NUMINAMATH_GPT_bus_patrons_correct_l1384_138444


namespace NUMINAMATH_GPT_simplify_expression_l1384_138464

variable (b c : ℝ)

theorem simplify_expression :
  3 * b * (3 * b ^ 3 + 2 * b) - 2 * b ^ 2 + c * (3 * b ^ 2 - c) = 9 * b ^ 4 + 4 * b ^ 2 + 3 * b ^ 2 * c - c ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1384_138464


namespace NUMINAMATH_GPT_Elberta_has_21_dollars_l1384_138429

theorem Elberta_has_21_dollars
  (Granny_Smith : ℕ)
  (Anjou : ℕ)
  (Elberta : ℕ)
  (h1 : Granny_Smith = 72)
  (h2 : Anjou = Granny_Smith / 4)
  (h3 : Elberta = Anjou + 3) :
  Elberta = 21 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_Elberta_has_21_dollars_l1384_138429


namespace NUMINAMATH_GPT_snickers_bars_needed_l1384_138448

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_snickers_bars_needed_l1384_138448


namespace NUMINAMATH_GPT_sum_of_all_angles_l1384_138403

-- Defining the three triangles and their properties
structure Triangle :=
  (a1 a2 a3 : ℝ)
  (sum : a1 + a2 + a3 = 180)

def triangle_ABC : Triangle := {a1 := 1, a2 := 2, a3 := 3, sum := sorry}
def triangle_DEF : Triangle := {a1 := 4, a2 := 5, a3 := 6, sum := sorry}
def triangle_GHI : Triangle := {a1 := 7, a2 := 8, a3 := 9, sum := sorry}

theorem sum_of_all_angles :
  triangle_ABC.a1 + triangle_ABC.a2 + triangle_ABC.a3 +
  triangle_DEF.a1 + triangle_DEF.a2 + triangle_DEF.a3 +
  triangle_GHI.a1 + triangle_GHI.a2 + triangle_GHI.a3 = 540 := by
  sorry

end NUMINAMATH_GPT_sum_of_all_angles_l1384_138403


namespace NUMINAMATH_GPT_arctan_tan_sub_eq_l1384_138484

noncomputable def arctan_tan_sub (a b : ℝ) : ℝ := Real.arctan (Real.tan a - 3 * Real.tan b)

theorem arctan_tan_sub_eq (a b : ℝ) (ha : a = 75) (hb : b = 15) :
  arctan_tan_sub a b = 75 :=
by
  sorry

end NUMINAMATH_GPT_arctan_tan_sub_eq_l1384_138484


namespace NUMINAMATH_GPT_nat_divides_2_pow_n_minus_1_l1384_138431

theorem nat_divides_2_pow_n_minus_1 (n : ℕ) (hn : 0 < n) : n ∣ 2^n - 1 ↔ n = 1 :=
  sorry

end NUMINAMATH_GPT_nat_divides_2_pow_n_minus_1_l1384_138431


namespace NUMINAMATH_GPT_estimate_undetected_typos_l1384_138479

variables (a b c : ℕ)
-- a, b, c ≥ 0 are non-negative integers representing discovered errors by proofreader A, B, and common errors respectively.

theorem estimate_undetected_typos (h : c ≤ a ∧ c ≤ b) :
  ∃ n : ℕ, n = a * b / c - a - b + c :=
sorry

end NUMINAMATH_GPT_estimate_undetected_typos_l1384_138479


namespace NUMINAMATH_GPT_probability_of_woman_lawyer_is_54_percent_l1384_138494

variable (total_members : ℕ) (women_percentage lawyers_percentage : ℕ)
variable (H_total_members_pos : total_members > 0) 
variable (H_women_percentage : women_percentage = 90)
variable (H_lawyers_percentage : lawyers_percentage = 60)

def probability_woman_lawyer : ℕ :=
  (women_percentage * lawyers_percentage * total_members) / (100 * 100)

theorem probability_of_woman_lawyer_is_54_percent (H_total_members_pos : total_members > 0)
  (H_women_percentage : women_percentage = 90)
  (H_lawyers_percentage : lawyers_percentage = 60) :
  probability_woman_lawyer total_members women_percentage lawyers_percentage = 54 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_woman_lawyer_is_54_percent_l1384_138494


namespace NUMINAMATH_GPT_operation_1_2010_l1384_138440

def operation (m n : ℕ) : ℕ := sorry

axiom operation_initial : operation 1 1 = 2
axiom operation_step (m n : ℕ) : operation m (n + 1) = operation m n + 3

theorem operation_1_2010 : operation 1 2010 = 6029 := sorry

end NUMINAMATH_GPT_operation_1_2010_l1384_138440


namespace NUMINAMATH_GPT_sum_of_coordinates_reflection_l1384_138455

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_reflection_l1384_138455


namespace NUMINAMATH_GPT_sequence_general_term_l1384_138467

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end NUMINAMATH_GPT_sequence_general_term_l1384_138467


namespace NUMINAMATH_GPT_derivative_of_f_l1384_138476

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + (1 / 3) * ((3 * x - 1) / (3 * x^2 - 2 * x + 1))

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
by intros; sorry

end NUMINAMATH_GPT_derivative_of_f_l1384_138476


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1384_138432

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1384_138432
