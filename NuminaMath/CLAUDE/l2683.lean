import Mathlib

namespace NUMINAMATH_CALUDE_trig_identity_proof_l2683_268361

theorem trig_identity_proof : 
  4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2683_268361


namespace NUMINAMATH_CALUDE_x_value_proof_l2683_268336

theorem x_value_proof (x : ℝ) : (-1 : ℝ) * 2 * x * 4 = 24 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2683_268336


namespace NUMINAMATH_CALUDE_cody_tickets_l2683_268353

theorem cody_tickets (initial : ℝ) (lost_bet : ℝ) (spent_beanie : ℝ) (won_game : ℝ) (dropped : ℝ)
  (h1 : initial = 56.5)
  (h2 : lost_bet = 6.3)
  (h3 : spent_beanie = 25.75)
  (h4 : won_game = 10.25)
  (h5 : dropped = 3.1) :
  initial - lost_bet - spent_beanie + won_game - dropped = 31.6 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l2683_268353


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l2683_268322

theorem sum_of_fourth_powers_of_roots (P : ℝ → ℝ) (r₁ r₂ : ℝ) : 
  P = (fun x ↦ x^2 + 2*x + 3) →
  P r₁ = 0 →
  P r₂ = 0 →
  r₁^4 + r₂^4 = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l2683_268322


namespace NUMINAMATH_CALUDE_mark_fruit_theorem_l2683_268358

/-- The number of fruit pieces Mark kept for next week -/
def fruit_kept_for_next_week (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) : ℕ :=
  initial_fruit - pieces_eaten_four_days - pieces_for_friday

theorem mark_fruit_theorem (initial_fruit pieces_eaten_four_days pieces_for_friday : ℕ) 
  (h1 : initial_fruit = 10)
  (h2 : pieces_eaten_four_days = 5)
  (h3 : pieces_for_friday = 3) :
  fruit_kept_for_next_week initial_fruit pieces_eaten_four_days pieces_for_friday = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_theorem_l2683_268358


namespace NUMINAMATH_CALUDE_pig_cure_probability_l2683_268384

theorem pig_cure_probability (p : ℝ) (n k : ℕ) (h_p : p = 0.9) (h_n : n = 5) (h_k : k = 3) :
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = (Nat.choose 5 3 : ℝ) * 0.9^3 * 0.1^2 :=
sorry

end NUMINAMATH_CALUDE_pig_cure_probability_l2683_268384


namespace NUMINAMATH_CALUDE_best_fit_model_l2683_268321

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |1 - model.r_squared| ≤ |1 - m.r_squared|

theorem best_fit_model :
  let models : List RegressionModel := [
    ⟨"Model 1", 0.25⟩,
    ⟨"Model 2", 0.50⟩,
    ⟨"Model 3", 0.98⟩,
    ⟨"Model 4", 0.80⟩
  ]
  let model3 : RegressionModel := ⟨"Model 3", 0.98⟩
  has_best_fit model3 models := by sorry

end NUMINAMATH_CALUDE_best_fit_model_l2683_268321


namespace NUMINAMATH_CALUDE_set_equality_l2683_268366

theorem set_equality (A : Set ℕ) : 
  ({1, 3} : Set ℕ) ⊆ A ∧ ({1, 3} : Set ℕ) ∪ A = {1, 3, 5} → A = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2683_268366


namespace NUMINAMATH_CALUDE_equality_of_four_reals_l2683_268369

theorem equality_of_four_reals (a b c d : ℝ) :
  a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a → a = b ∧ b = c ∧ c = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_four_reals_l2683_268369


namespace NUMINAMATH_CALUDE_solution_value_l2683_268390

theorem solution_value (a : ℚ) : 
  (∃ x : ℚ, x = -2 ∧ 2 * x + 3 * a = 0) → a = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2683_268390


namespace NUMINAMATH_CALUDE_john_total_spent_l2683_268332

def silver_amount : ℝ := 1.5
def gold_amount : ℝ := 2 * silver_amount
def silver_price_per_ounce : ℝ := 20
def gold_price_per_ounce : ℝ := 50 * silver_price_per_ounce

def total_spent : ℝ := silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

theorem john_total_spent :
  total_spent = 3030 := by sorry

end NUMINAMATH_CALUDE_john_total_spent_l2683_268332


namespace NUMINAMATH_CALUDE_dawson_group_size_l2683_268345

/-- The number of people in a group given the total cost and cost per person -/
def group_size (total_cost : ℕ) (cost_per_person : ℕ) : ℕ :=
  total_cost / cost_per_person

/-- Proof that the group size is 15 given the specific costs -/
theorem dawson_group_size :
  group_size 13500 900 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dawson_group_size_l2683_268345


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2683_268327

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2683_268327


namespace NUMINAMATH_CALUDE_sin_1050_degrees_l2683_268305

theorem sin_1050_degrees : Real.sin (1050 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1050_degrees_l2683_268305


namespace NUMINAMATH_CALUDE_bisection_uses_all_structures_l2683_268337

/-- Represents the basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- The bisection method algorithm -/
def bisectionMethod : Algorithm := sorry

/-- Every algorithm has a sequential structure -/
axiom sequential_in_all (a : Algorithm) : 
  AlgorithmStructure.Sequential ∈ a.structures

/-- Loop structure implies conditional structure -/
axiom loop_implies_conditional (a : Algorithm) :
  AlgorithmStructure.Loop ∈ a.structures → 
  AlgorithmStructure.Conditional ∈ a.structures

/-- Bisection method involves a loop structure -/
axiom bisection_has_loop : 
  AlgorithmStructure.Loop ∈ bisectionMethod.structures

/-- Theorem: The bisection method algorithm requires all three basic structures -/
theorem bisection_uses_all_structures : 
  AlgorithmStructure.Sequential ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Conditional ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Loop ∈ bisectionMethod.structures := by
  sorry


end NUMINAMATH_CALUDE_bisection_uses_all_structures_l2683_268337


namespace NUMINAMATH_CALUDE_three_fractions_l2683_268376

-- Define the list of expressions
def expressions : List String := [
  "3/a",
  "(a+b)/7",
  "x^2 + (1/2)y^2",
  "5",
  "1/(x-1)",
  "x/(8π)",
  "x^2/x"
]

-- Define what constitutes a fraction
def is_fraction (expr : String) : Prop :=
  ∃ (num denom : String), 
    expr = num ++ "/" ++ denom ∧ 
    denom ≠ "1" ∧
    ¬∃ (simplified : String), simplified ≠ expr ∧ ¬(∃ (n d : String), simplified = n ++ "/" ++ d)

-- Theorem stating that exactly 3 expressions are fractions
theorem three_fractions : 
  ∃ (fracs : List String), 
    fracs.length = 3 ∧ 
    (∀ expr ∈ fracs, expr ∈ expressions ∧ is_fraction expr) ∧
    (∀ expr ∈ expressions, is_fraction expr → expr ∈ fracs) :=
sorry

end NUMINAMATH_CALUDE_three_fractions_l2683_268376


namespace NUMINAMATH_CALUDE_alonzo_tomato_harvest_l2683_268347

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mrs. Maxwell -/
def sold_to_maxwell : ℝ := 125.5

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mr. Wilson -/
def sold_to_wilson : ℝ := 78

/-- The amount of tomatoes (in kg) that Mr. Alonzo has not sold -/
def not_sold : ℝ := 42

/-- The total amount of tomatoes (in kg) that Mr. Alonzo harvested -/
def total_harvested : ℝ := sold_to_maxwell + sold_to_wilson + not_sold

theorem alonzo_tomato_harvest : total_harvested = 245.5 := by
  sorry

end NUMINAMATH_CALUDE_alonzo_tomato_harvest_l2683_268347


namespace NUMINAMATH_CALUDE_intersection_condition_l2683_268383

/-- Given a line y = kx + 2k and a circle x^2 + y^2 + mx + 4 = 0,
    if the line has at least one intersection point with the circle, then m > 4 -/
theorem intersection_condition (k m : ℝ) : 
  (∃ x y : ℝ, y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2683_268383


namespace NUMINAMATH_CALUDE_sum_cube_over_power_of_three_l2683_268355

open Real BigOperators

/-- The sum of the infinite series $\sum_{k=1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{16}$. -/
theorem sum_cube_over_power_of_three :
  ∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ) = 39 / 16 := by sorry

end NUMINAMATH_CALUDE_sum_cube_over_power_of_three_l2683_268355


namespace NUMINAMATH_CALUDE_circle_area_increase_l2683_268349

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2683_268349


namespace NUMINAMATH_CALUDE_net_change_is_correct_l2683_268325

/-- Calculates the final price after applying two percentage changes -/
def apply_price_changes (original_price : ℚ) (change1 : ℚ) (change2 : ℚ) : ℚ :=
  original_price * (1 + change1) * (1 + change2)

/-- Represents the store inventory with original prices and price changes -/
structure Inventory where
  tv_price : ℚ
  tv_change1 : ℚ
  tv_change2 : ℚ
  fridge_price : ℚ
  fridge_change1 : ℚ
  fridge_change2 : ℚ
  washer_price : ℚ
  washer_change1 : ℚ
  washer_change2 : ℚ

/-- Calculates the net change in total prices -/
def net_change (inv : Inventory) : ℚ :=
  let final_tv_price := apply_price_changes inv.tv_price inv.tv_change1 inv.tv_change2
  let final_fridge_price := apply_price_changes inv.fridge_price inv.fridge_change1 inv.fridge_change2
  let final_washer_price := apply_price_changes inv.washer_price inv.washer_change1 inv.washer_change2
  let total_final_price := final_tv_price + final_fridge_price + final_washer_price
  let total_original_price := inv.tv_price + inv.fridge_price + inv.washer_price
  total_final_price - total_original_price

theorem net_change_is_correct (inv : Inventory) : 
  inv.tv_price = 500 ∧ 
  inv.tv_change1 = -1/5 ∧ 
  inv.tv_change2 = 9/20 ∧
  inv.fridge_price = 1000 ∧ 
  inv.fridge_change1 = 7/20 ∧ 
  inv.fridge_change2 = -3/20 ∧
  inv.washer_price = 750 ∧ 
  inv.washer_change1 = 1/10 ∧ 
  inv.washer_change2 = -1/5 
  → net_change inv = 275/2 := by
  sorry

#eval net_change { 
  tv_price := 500, tv_change1 := -1/5, tv_change2 := 9/20,
  fridge_price := 1000, fridge_change1 := 7/20, fridge_change2 := -3/20,
  washer_price := 750, washer_change1 := 1/10, washer_change2 := -1/5
}

end NUMINAMATH_CALUDE_net_change_is_correct_l2683_268325


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2683_268365

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop := ∃ m : ℝ, m * m = n

-- Define a function to check if a number is in its simplest quadratic radical form
def isSimplestQuadraticRadical (n : ℝ) : Prop :=
  n > 0 ∧ ¬(isPerfectSquare n) ∧ ∀ m : ℝ, m > 1 → ¬(isPerfectSquare (n / (m * m)))

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical 6 ∧
  ¬(isSimplestQuadraticRadical 4) ∧
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 12) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2683_268365


namespace NUMINAMATH_CALUDE_cube_root_sum_equation_l2683_268356

theorem cube_root_sum_equation (x : ℝ) :
  x = (11 + Real.sqrt 337) ^ (1/3 : ℝ) + (11 - Real.sqrt 337) ^ (1/3 : ℝ) →
  x^3 + 18*x = 22 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_equation_l2683_268356


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l2683_268370

def ones_digit_cycle : List Nat := [8, 4, 2, 6]

theorem ones_digit_of_8_to_47 (h : ones_digit_cycle = [8, 4, 2, 6]) :
  (8^47 : ℕ) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l2683_268370


namespace NUMINAMATH_CALUDE_bracelet_pairing_impossibility_l2683_268326

theorem bracelet_pairing_impossibility (n : ℕ) (h : n = 100) :
  ¬ ∃ (arrangement : List (Finset (Fin n))),
    (∀ s ∈ arrangement, s.card = 3) ∧
    (∀ i j : Fin n, i ≠ j → 
      (arrangement.filter (λ s => i ∈ s ∧ j ∈ s)).length = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_bracelet_pairing_impossibility_l2683_268326


namespace NUMINAMATH_CALUDE_minimum_value_of_f_minimum_value_case1_minimum_value_case2_l2683_268394

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the theorem
theorem minimum_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, f x ≥ f a ∧ -2 < a ∧ a ≤ 1) ∨
  (∀ x ∈ Set.Icc (-2) a, f x ≥ -1 ∧ a > 1) := by
  sorry

-- Define helper theorems for each case
theorem minimum_value_case1 (a : ℝ) (h1 : -2 < a) (h2 : a ≤ 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ f a := by
  sorry

theorem minimum_value_case2 (a : ℝ) (h : a > 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_minimum_value_case1_minimum_value_case2_l2683_268394


namespace NUMINAMATH_CALUDE_tractor_finance_l2683_268378

/-- Calculates the total amount financed given monthly payment and number of years -/
def total_financed (monthly_payment : ℚ) (years : ℕ) : ℚ :=
  monthly_payment * (years * 12)

/-- Proves that financing $150 per month for 5 years results in a total of $9000 -/
theorem tractor_finance : total_financed 150 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_tractor_finance_l2683_268378


namespace NUMINAMATH_CALUDE_part_one_part_two_l2683_268382

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x - b * x^2

-- Part 1
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2683_268382


namespace NUMINAMATH_CALUDE_blue_balls_removal_l2683_268380

theorem blue_balls_removal (total : ℕ) (red_percent : ℚ) (target_red_percent : ℚ) 
  (h1 : total = 120) 
  (h2 : red_percent = 2/5) 
  (h3 : target_red_percent = 3/4) : 
  ∃ (removed : ℕ), 
    removed = 56 ∧ 
    (red_percent * total : ℚ) / (total - removed : ℚ) = target_red_percent := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_removal_l2683_268380


namespace NUMINAMATH_CALUDE_union_A_B_intersect_complement_A_B_l2683_268372

/-- The set A -/
def A : Set ℝ := {x | x < -5 ∨ x > 1}

/-- The set B -/
def B : Set ℝ := {x | -4 < x ∧ x < 3}

/-- Theorem: The union of A and B -/
theorem union_A_B : A ∪ B = {x : ℝ | x < -5 ∨ x > -4} := by sorry

/-- Theorem: The intersection of the complement of A and B -/
theorem intersect_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | -4 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersect_complement_A_B_l2683_268372


namespace NUMINAMATH_CALUDE_magic_square_y_zero_l2683_268399

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  g : ℤ
  h : ℤ
  i : ℤ
  is_magic : 
    a + b + c = d + e + f ∧
    a + b + c = g + h + i ∧
    a + b + c = a + d + g ∧
    a + b + c = b + e + h ∧
    a + b + c = c + f + i ∧
    a + b + c = a + e + i ∧
    a + b + c = c + e + g

/-- The theorem stating that y must be 0 in the given magic square configuration -/
theorem magic_square_y_zero (ms : MagicSquare) 
  (h1 : ms.a = y)
  (h2 : ms.b = 17)
  (h3 : ms.c = 124)
  (h4 : ms.d = 9) :
  y = 0 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_y_zero_l2683_268399


namespace NUMINAMATH_CALUDE_power_function_above_identity_l2683_268393

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) (hα : α < 1) : x^α > x := by
  sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l2683_268393


namespace NUMINAMATH_CALUDE_nested_square_root_value_l2683_268309

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l2683_268309


namespace NUMINAMATH_CALUDE_logarithm_identity_l2683_268359

theorem logarithm_identity : Real.log 5 ^ 2 + Real.log 2 * Real.log 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l2683_268359


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2683_268348

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) (h_eq : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : α + β = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2683_268348


namespace NUMINAMATH_CALUDE_broken_seashells_l2683_268331

/-- Given the total number of seashells and the number of unbroken seashells,
    calculate the number of broken seashells. -/
theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h : unbroken ≤ total) :
  total - unbroken = total - unbroken :=
by sorry

end NUMINAMATH_CALUDE_broken_seashells_l2683_268331


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l2683_268301

theorem right_triangle_consecutive_sides (a c : ℕ) (h1 : c = a + 1) :
  ∃ b : ℕ, b * b = c + a ∧ c * c = a * a + b * b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l2683_268301


namespace NUMINAMATH_CALUDE_cards_distribution_l2683_268363

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2683_268363


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l2683_268346

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) :
  total_caps / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l2683_268346


namespace NUMINAMATH_CALUDE_common_material_choices_eq_120_l2683_268323

/-- The number of ways to choose r items from n items --/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items --/
def arrange (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common --/
def commonMaterialChoices : ℕ :=
  choose 6 1 * arrange 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_common_material_choices_eq_120_l2683_268323


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2683_268338

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2683_268338


namespace NUMINAMATH_CALUDE_derivative_cosine_at_pi_half_l2683_268315

theorem derivative_cosine_at_pi_half (f : ℝ → ℝ) (h : ∀ x, f x = 5 * Real.cos x) :
  deriv f (Real.pi / 2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cosine_at_pi_half_l2683_268315


namespace NUMINAMATH_CALUDE_divisibility_by_12321_l2683_268319

theorem divisibility_by_12321 (a : ℤ) : 
  (∃ k : ℕ, 12321 ∣ (a^k + 1)) ↔ 
  (∃ n : ℤ, a ≡ 11 [ZMOD 111] ∨ 
            a ≡ 41 [ZMOD 111] ∨ 
            a ≡ 62 [ZMOD 111] ∨ 
            a ≡ 65 [ZMOD 111] ∨ 
            a ≡ 77 [ZMOD 111] ∨ 
            a ≡ 95 [ZMOD 111] ∨ 
            a ≡ 101 [ZMOD 111] ∨ 
            a ≡ 104 [ZMOD 111] ∨ 
            a ≡ 110 [ZMOD 111]) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_12321_l2683_268319


namespace NUMINAMATH_CALUDE_inequality_solution_l2683_268341

theorem inequality_solution (x : ℝ) : 
  (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0 ↔ -3 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2683_268341


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l2683_268308

/-- Given that (-1,2) and (2,a) lie on the graph of y = k/x, prove that a = -1 -/
theorem inverse_proportion_points (k a : ℝ) : 
  (2 = k / (-1)) → (a = k / 2) → a = -1 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l2683_268308


namespace NUMINAMATH_CALUDE_mary_has_ten_marbles_l2683_268398

/-- The number of blue marbles Dan has -/
def dan_marbles : ℕ := 5

/-- The factor by which Mary has more marbles than Dan -/
def mary_factor : ℕ := 2

/-- The number of blue marbles Mary has -/
def mary_marbles : ℕ := mary_factor * dan_marbles

theorem mary_has_ten_marbles : mary_marbles = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_marbles_l2683_268398


namespace NUMINAMATH_CALUDE_F_36_72_equals_48_max_F_happy_pair_equals_58_l2683_268343

/-- Function F calculates the sum of products of digits in two-digit numbers -/
def F (m n : ℕ) : ℕ :=
  (m / 10) * (n % 10) + (m % 10) * (n / 10)

/-- Swaps the digits of a two-digit number -/
def swapDigits (m : ℕ) : ℕ :=
  (m % 10) * 10 + (m / 10)

/-- Checks if two numbers form a "happy pair" -/
def isHappyPair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 5 ∧
  m = 21 * a + b ∧ n = 53 + b ∧
  (swapDigits m + 5 * (n % 10)) % 11 = 0

theorem F_36_72_equals_48 : F 36 72 = 48 := by sorry

theorem max_F_happy_pair_equals_58 :
  (∃ (m n : ℕ), isHappyPair m n ∧ F m n = 58) ∧
  (∀ (m n : ℕ), isHappyPair m n → F m n ≤ 58) := by sorry

end NUMINAMATH_CALUDE_F_36_72_equals_48_max_F_happy_pair_equals_58_l2683_268343


namespace NUMINAMATH_CALUDE_smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l2683_268335

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def containsAllDigitsOnce (n : ℕ) : Prop := sorry

/-- The smallest perfect square containing all digits from 0 to 9 exactly once -/
def smallestSquare : ℕ := 1026753849

/-- The largest perfect square containing all digits from 0 to 9 exactly once -/
def largestSquare : ℕ := 9814072356

/-- Theorem stating that smallestSquare is a perfect square -/
theorem smallestSquare_is_square : ∃ k : ℕ, k * k = smallestSquare := sorry

/-- Theorem stating that largestSquare is a perfect square -/
theorem largestSquare_is_square : ∃ k : ℕ, k * k = largestSquare := sorry

/-- Theorem stating that smallestSquare contains all digits from 0 to 9 exactly once -/
theorem smallestSquare_contains_all_digits : containsAllDigitsOnce smallestSquare := sorry

/-- Theorem stating that largestSquare contains all digits from 0 to 9 exactly once -/
theorem largestSquare_contains_all_digits : containsAllDigitsOnce largestSquare := sorry

/-- Theorem stating that smallestSquare is the smallest such square -/
theorem smallestSquare_is_smallest :
  ∀ n : ℕ, n < smallestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

/-- Theorem stating that largestSquare is the largest such square -/
theorem largestSquare_is_largest :
  ∀ n : ℕ, n > largestSquare → ¬(∃ k : ℕ, k * k = n ∧ containsAllDigitsOnce n) := sorry

end NUMINAMATH_CALUDE_smallestSquare_is_square_largestSquare_is_square_smallestSquare_contains_all_digits_largestSquare_contains_all_digits_smallestSquare_is_smallest_largestSquare_is_largest_l2683_268335


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_equivalent_l2683_268357

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem for the first part of the problem
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the second part of the problem
theorem subset_condition_equivalent :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_equivalent_l2683_268357


namespace NUMINAMATH_CALUDE_water_fountain_trips_l2683_268395

/-- The number of trips to the water fountain -/
def number_of_trips (total_distance : ℕ) (distance_to_fountain : ℕ) : ℕ :=
  total_distance / distance_to_fountain

theorem water_fountain_trips : 
  number_of_trips 120 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_fountain_trips_l2683_268395


namespace NUMINAMATH_CALUDE_sector_central_angle_l2683_268373

/-- Given a circular sector with area 4 and arc length 4, its central angle is 2 radians. -/
theorem sector_central_angle (area : ℝ) (arc_length : ℝ) (radius : ℝ) (angle : ℝ) :
  area = 4 →
  arc_length = 4 →
  area = (1 / 2) * radius * arc_length →
  arc_length = radius * angle →
  angle = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2683_268373


namespace NUMINAMATH_CALUDE_doris_earnings_l2683_268391

/-- Calculates the number of weeks needed for Doris to earn enough to cover her monthly expenses --/
def weeks_to_earn_expenses (hourly_rate : ℚ) (weekday_hours : ℚ) (saturday_hours : ℚ) (monthly_expense : ℚ) : ℚ :=
  let weekly_hours := 5 * weekday_hours + saturday_hours
  let weekly_earnings := weekly_hours * hourly_rate
  monthly_expense / weekly_earnings

theorem doris_earnings : 
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  weeks_to_earn_expenses hourly_rate weekday_hours saturday_hours monthly_expense = 3 := by
  sorry

end NUMINAMATH_CALUDE_doris_earnings_l2683_268391


namespace NUMINAMATH_CALUDE_noah_holidays_per_month_l2683_268303

/-- The number of holidays Noah takes in a year -/
def total_holidays : ℕ := 36

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Noah takes each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem noah_holidays_per_month :
  holidays_per_month = 3 := by sorry

end NUMINAMATH_CALUDE_noah_holidays_per_month_l2683_268303


namespace NUMINAMATH_CALUDE_negative_sixty_four_two_thirds_power_l2683_268313

theorem negative_sixty_four_two_thirds_power : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_two_thirds_power_l2683_268313


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l2683_268314

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-3, 0)

-- Define the tangent line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k * (x + 3)

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (y_intercept : ℝ),
    -- The center of M
    center = (-2, 1) ∧
    -- The radius of M
    radius = Real.sqrt 2 ∧
    -- The y-intercept of line l
    y_intercept = -3 ∧
    -- Line l is tangent to circle M at point P
    (∀ x y, circle_M x y → line_l x y → (x, y) = point_P) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l2683_268314


namespace NUMINAMATH_CALUDE_walking_scenario_theorem_l2683_268397

/-- Represents the walking scenario with Yolanda, Bob, and Jim -/
structure WalkingScenario where
  total_distance : ℝ
  yolanda_speed : ℝ
  bob_speed_difference : ℝ
  jim_speed : ℝ
  yolanda_head_start : ℝ

/-- Calculates the distance Bob walked when he met Yolanda -/
def bob_distance_walked (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Calculates the point where Jim and Yolanda met, measured from point X -/
def jim_yolanda_meeting_point (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct distances for Bob and Jim -/
theorem walking_scenario_theorem (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 80)
  (h2 : scenario.yolanda_speed = 4)
  (h3 : scenario.bob_speed_difference = 2)
  (h4 : scenario.jim_speed = 5)
  (h5 : scenario.yolanda_head_start = 1) :
  bob_distance_walked scenario = 45.6 ∧ 
  jim_yolanda_meeting_point scenario = 38 :=
by sorry

end NUMINAMATH_CALUDE_walking_scenario_theorem_l2683_268397


namespace NUMINAMATH_CALUDE_line_equation_correct_l2683_268392

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a point (x, y) satisfies the equation of the line -/
def Line.satisfiesEquation (l : Line) (x y : ℝ) : Prop :=
  2 * x - y - 5 = 0

theorem line_equation_correct (l : Line) :
  l.slope = 2 ∧ l.point = (3, 1) →
  ∀ x y : ℝ, l.satisfiesEquation x y ↔ y - 1 = l.slope * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2683_268392


namespace NUMINAMATH_CALUDE_biathlon_run_distance_l2683_268302

/-- Biathlon problem -/
theorem biathlon_run_distance
  (total_distance : ℝ)
  (bicycle_distance : ℝ)
  (bicycle_velocity : ℝ)
  (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : bicycle_distance = 145)
  (h3 : bicycle_velocity = 29)
  (h4 : total_time = 6)
  : total_distance - bicycle_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_run_distance_l2683_268302


namespace NUMINAMATH_CALUDE_points_needed_for_average_increase_l2683_268381

/-- Represents a basketball player's scoring history -/
structure PlayerStats where
  gamesPlayed : ℕ
  totalPoints : ℕ

/-- Calculates the average points per game -/
def averagePoints (stats : PlayerStats) : ℚ :=
  stats.totalPoints / stats.gamesPlayed

/-- Updates player stats after a game -/
def updateStats (stats : PlayerStats) (points : ℕ) : PlayerStats :=
  { gamesPlayed := stats.gamesPlayed + 1
  , totalPoints := stats.totalPoints + points }

/-- Theorem: A player who raised their average from 20 to 21 by scoring 36 points
    must score 38 points to raise their average to 22 -/
theorem points_needed_for_average_increase 
  (initialStats : PlayerStats)
  (h1 : averagePoints initialStats = 20)
  (h2 : averagePoints (updateStats initialStats 36) = 21) :
  averagePoints (updateStats (updateStats initialStats 36) 38) = 22 := by
  sorry


end NUMINAMATH_CALUDE_points_needed_for_average_increase_l2683_268381


namespace NUMINAMATH_CALUDE_sandy_mall_change_l2683_268306

/-- The change Sandy received after buying clothes at the mall -/
def sandys_change (pants_cost shirt_cost bill_amount : ℚ) : ℚ :=
  bill_amount - (pants_cost + shirt_cost)

/-- Theorem stating that Sandy's change is $2.51 given the problem conditions -/
theorem sandy_mall_change :
  sandys_change 9.24 8.25 20 = 2.51 := by
  sorry

end NUMINAMATH_CALUDE_sandy_mall_change_l2683_268306


namespace NUMINAMATH_CALUDE_liters_to_pints_conversion_l2683_268387

/-- Given that 0.75 liters is approximately 1.575 pints, prove that 3 liters is equal to 6.3 pints. -/
theorem liters_to_pints_conversion (liter_to_pint : ℝ → ℝ) 
  (h : liter_to_pint 0.75 = 1.575) : liter_to_pint 3 = 6.3 := by
  sorry

end NUMINAMATH_CALUDE_liters_to_pints_conversion_l2683_268387


namespace NUMINAMATH_CALUDE_cosine_equality_l2683_268329

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (331 * π / 180) → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2683_268329


namespace NUMINAMATH_CALUDE_contacts_per_dollar_theorem_l2683_268330

/-- Represents a box of contacts with quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box -/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.quantity / box.price

/-- Theorem stating that the number of contacts equal to $1 worth in the box 
    with the lower cost per contact is 3 -/
theorem contacts_per_dollar_theorem (box1 box2 : ContactBox) 
  (h1 : box1.quantity = 50 ∧ box1.price = 25)
  (h2 : box2.quantity = 99 ∧ box2.price = 33) :
  let betterBox := if contactsPerDollar box1 > contactsPerDollar box2 then box1 else box2
  contactsPerDollar betterBox = 3 := by
  sorry

end NUMINAMATH_CALUDE_contacts_per_dollar_theorem_l2683_268330


namespace NUMINAMATH_CALUDE_sum_of_roots_range_l2683_268371

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 else Real.exp x

-- Define the function F as [f(x)]^2
def F (x : ℝ) : ℝ := (f x)^2

-- Define the property that F(x) = a has exactly two roots
def has_two_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ = a ∧ F x₂ = a ∧ ∀ x, F x = a → x = x₁ ∨ x = x₂

-- Theorem statement
theorem sum_of_roots_range (a : ℝ) (h : has_two_roots a) :
  ∃ x₁ x₂, F x₁ = a ∧ F x₂ = a ∧ x₁ + x₂ > -1 ∧ ∀ M, ∃ b > a, 
  ∃ y₁ y₂, F y₁ = b ∧ F y₂ = b ∧ y₁ + y₂ > M :=
sorry

end

end NUMINAMATH_CALUDE_sum_of_roots_range_l2683_268371


namespace NUMINAMATH_CALUDE_g_lower_bound_l2683_268300

theorem g_lower_bound (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m < 1) :
  Real.exp (m * x - 1) - (Real.log x + 1) / m > m^(1/m) - m^(-1/m) := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l2683_268300


namespace NUMINAMATH_CALUDE_linear_equation_solve_l2683_268320

theorem linear_equation_solve (x y : ℝ) : 
  x + 2 * y = 6 → y = (-x + 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l2683_268320


namespace NUMINAMATH_CALUDE_three_squares_representation_l2683_268379

theorem three_squares_representation (N : ℕ) :
  (∃ a b c : ℤ, N = (3*a)^2 + (3*b)^2 + (3*c)^2) →
  (∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_representation_l2683_268379


namespace NUMINAMATH_CALUDE_corner_sum_9x9_l2683_268340

def checkerboard_size : ℕ := 9

def corner_sum (n : ℕ) : ℕ :=
  1 + n + (n^2 - n + 1) + n^2

theorem corner_sum_9x9 :
  corner_sum checkerboard_size = 164 :=
by sorry

end NUMINAMATH_CALUDE_corner_sum_9x9_l2683_268340


namespace NUMINAMATH_CALUDE_alyssa_cookie_count_l2683_268352

/-- The number of cookies Alyanna has -/
def aiyanna_cookies : ℕ := 140

/-- The difference between Aiyanna's and Alyssa's cookies -/
def cookie_difference : ℕ := 11

/-- The number of cookies Alyssa has -/
def alyssa_cookies : ℕ := aiyanna_cookies - cookie_difference

theorem alyssa_cookie_count : alyssa_cookies = 129 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookie_count_l2683_268352


namespace NUMINAMATH_CALUDE_smallest_odd_four_primes_with_13_l2683_268350

def is_prime (n : ℕ) : Prop := sorry

def prime_factors (n : ℕ) : Finset ℕ := sorry

theorem smallest_odd_four_primes_with_13 :
  ∀ n : ℕ,
  n % 2 = 1 →
  (prime_factors n).card = 4 →
  13 ∈ prime_factors n →
  n ≥ 1365 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_four_primes_with_13_l2683_268350


namespace NUMINAMATH_CALUDE_negation_of_square_non_negative_l2683_268333

theorem negation_of_square_non_negative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_non_negative_l2683_268333


namespace NUMINAMATH_CALUDE_rectangle_triangle_count_l2683_268312

theorem rectangle_triangle_count (n m : ℕ) (hn : n = 6) (hm : m = 7) :
  n.choose 2 * m + m.choose 2 * n = 231 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_count_l2683_268312


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2683_268385

/-- The constant term in the expansion of x(1 - 2/√x)^6 is 60 -/
theorem constant_term_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2683_268385


namespace NUMINAMATH_CALUDE_carbonic_acid_molecular_weight_l2683_268368

/-- The molecular weight of carbonic acid in grams per mole. -/
def molecular_weight_carbonic_acid : ℝ := 62

/-- The number of moles of carbonic acid in the given sample. -/
def moles_carbonic_acid : ℝ := 8

/-- The total weight of the given sample of carbonic acid in grams. -/
def total_weight_carbonic_acid : ℝ := 496

/-- Theorem stating that the molecular weight of carbonic acid is 62 grams/mole,
    given that 8 moles of carbonic acid weigh 496 grams. -/
theorem carbonic_acid_molecular_weight :
  molecular_weight_carbonic_acid = total_weight_carbonic_acid / moles_carbonic_acid :=
by sorry

end NUMINAMATH_CALUDE_carbonic_acid_molecular_weight_l2683_268368


namespace NUMINAMATH_CALUDE_complex_fraction_product_l2683_268375

theorem complex_fraction_product (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 : ℂ) + 7 * Complex.I = (a + b * Complex.I) * (2 - Complex.I) →
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l2683_268375


namespace NUMINAMATH_CALUDE_apartment_number_l2683_268316

theorem apartment_number : ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 17 * (n % 10) := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_l2683_268316


namespace NUMINAMATH_CALUDE_tonys_correct_score_l2683_268388

def class_size : ℕ := 20
def initial_average : ℚ := 73
def final_average : ℚ := 74
def score_increase : ℕ := 16

theorem tonys_correct_score :
  ∀ (initial_score final_score : ℕ),
  (class_size - 1 : ℚ) * initial_average + (initial_score : ℚ) / class_size = initial_average →
  (class_size - 1 : ℚ) * initial_average + (final_score : ℚ) / class_size = final_average →
  final_score = initial_score + score_increase →
  final_score = 36 := by
sorry

end NUMINAMATH_CALUDE_tonys_correct_score_l2683_268388


namespace NUMINAMATH_CALUDE_coffee_expense_theorem_l2683_268328

/-- Calculates the weekly coffee expense for a household -/
def weekly_coffee_expense (
  num_people : ℕ
) (cups_per_person_per_day : ℕ
) (ounces_per_cup : ℚ
) (price_per_ounce : ℚ
) : ℚ :=
  (num_people * cups_per_person_per_day : ℚ) *
  ounces_per_cup *
  price_per_ounce *
  7

/-- Proves that the weekly coffee expense for the given conditions is $35 -/
theorem coffee_expense_theorem :
  weekly_coffee_expense 4 2 (1/2) (5/4) = 35 := by
  sorry

end NUMINAMATH_CALUDE_coffee_expense_theorem_l2683_268328


namespace NUMINAMATH_CALUDE_barbell_cost_is_270_l2683_268334

/-- The cost of each barbell given the total amount paid, change received, and number of barbells purchased. -/
def barbell_cost (total_paid : ℕ) (change : ℕ) (num_barbells : ℕ) : ℕ :=
  (total_paid - change) / num_barbells

/-- Theorem stating that the cost of each barbell is $270 under the given conditions. -/
theorem barbell_cost_is_270 :
  barbell_cost 850 40 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_is_270_l2683_268334


namespace NUMINAMATH_CALUDE_loss_per_metre_is_five_l2683_268386

/-- Calculates the loss per metre of cloth given the total metres sold, 
    total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_metres

/-- Proves that the loss per metre is 5 given the specified conditions. -/
theorem loss_per_metre_is_five : 
  loss_per_metre 500 18000 41 = 5 := by
  sorry

#eval loss_per_metre 500 18000 41

end NUMINAMATH_CALUDE_loss_per_metre_is_five_l2683_268386


namespace NUMINAMATH_CALUDE_gas_used_for_appointments_l2683_268360

def distance_to_dermatologist : ℝ := 30
def distance_to_gynecologist : ℝ := 50
def car_efficiency : ℝ := 20

theorem gas_used_for_appointments : 
  (2 * distance_to_dermatologist + 2 * distance_to_gynecologist) / car_efficiency = 8 := by
  sorry

end NUMINAMATH_CALUDE_gas_used_for_appointments_l2683_268360


namespace NUMINAMATH_CALUDE_parametric_elimination_l2683_268362

theorem parametric_elimination (x y t : ℝ) 
  (hx : x = 1 + 2 * t - 2 * t^2) 
  (hy : y = 2 * (1 + t) * Real.sqrt (1 - t^2)) : 
  y^4 + 2 * y^2 * (x^2 - 12 * x + 9) + x^4 + 8 * x^3 + 18 * x^2 - 27 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_elimination_l2683_268362


namespace NUMINAMATH_CALUDE_probability_of_defective_product_l2683_268318

/-- Given a set of products with some defective ones, calculate the probability of selecting a defective product -/
theorem probability_of_defective_product 
  (total : ℕ) 
  (defective : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : defective ≤ total) : 
  (defective : ℚ) / total = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_defective_product_l2683_268318


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_negative_six_l2683_268307

def S : Finset Int := {0, 5, -2, 18, -4, 3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y ∧ y ≠ z ∧ x ≠ z → 
  a + b + c ≤ x + y + z :=
by sorry

theorem smallest_sum_is_negative_six :
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = -6 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y ∧ y ≠ z ∧ x ≠ z → 
   a + b + c ≤ x + y + z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_smallest_sum_is_negative_six_l2683_268307


namespace NUMINAMATH_CALUDE_tan_sum_identity_l2683_268354

theorem tan_sum_identity : 
  Real.tan (25 * π / 180) + Real.tan (35 * π / 180) + 
  Real.sqrt 3 * Real.tan (25 * π / 180) * Real.tan (35 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l2683_268354


namespace NUMINAMATH_CALUDE_hall_wallpaper_expenditure_l2683_268310

/-- Calculates the total expenditure for covering the walls and ceiling of a rectangular hall with wallpaper. -/
def total_expenditure (length width height cost_per_sqm : ℚ) : ℚ :=
  let wall_area := 2 * (length * height + width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for covering a 30m x 25m x 10m hall with wallpaper costing Rs. 75 per square meter is Rs. 138,750. -/
theorem hall_wallpaper_expenditure :
  total_expenditure 30 25 10 75 = 138750 := by
  sorry

end NUMINAMATH_CALUDE_hall_wallpaper_expenditure_l2683_268310


namespace NUMINAMATH_CALUDE_license_plate_difference_l2683_268351

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters at the beginning of a California license plate -/
def ca_prefix_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def ca_digits : ℕ := 3

/-- The number of letters at the end of a California license plate -/
def ca_suffix_letters : ℕ := 2

/-- The number of letters in a Texas license plate -/
def tx_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def tx_digits : ℕ := 4

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : 
  (num_letters ^ (ca_prefix_letters + ca_suffix_letters) * num_digits ^ ca_digits) - 
  (num_letters ^ tx_letters * num_digits ^ tx_digits) = 301093376000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2683_268351


namespace NUMINAMATH_CALUDE_point_C_x_value_l2683_268377

def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C (x : ℝ) : ℝ × ℝ := (5, x)

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

theorem point_C_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_C_x_value_l2683_268377


namespace NUMINAMATH_CALUDE_total_weight_of_tickets_l2683_268339

-- Define the given conditions
def loose_boxes : ℕ := 9
def tickets_per_box : ℕ := 5
def weight_per_box : ℝ := 1.2
def boxes_per_case : ℕ := 10
def cases : ℕ := 2

-- Define the theorem
theorem total_weight_of_tickets :
  (loose_boxes + cases * boxes_per_case) * weight_per_box = 34.8 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_tickets_l2683_268339


namespace NUMINAMATH_CALUDE_brick_length_calculation_l2683_268364

/-- Calculates the length of a brick given wall and brick specifications --/
theorem brick_length_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_count : ℕ) (brick_width brick_height : ℝ) :
  wall_length = 10 ∧ wall_width = 4 ∧ wall_height = 5 ∧
  mortar_percentage = 0.1 ∧ brick_count = 6000 ∧
  brick_width = 15 ∧ brick_height = 8 →
  ∃ (brick_length : ℝ),
    brick_length = 250 ∧
    (wall_length * wall_width * wall_height * (1 - mortar_percentage) * 1000000) =
    (brick_length * brick_width * brick_height * brick_count) :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l2683_268364


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2683_268342

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) + 2*i * (3 - 2*i) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2683_268342


namespace NUMINAMATH_CALUDE_seashells_given_to_mike_l2683_268374

theorem seashells_given_to_mike (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 79)
  (h2 : remaining_seashells = 16) :
  initial_seashells - remaining_seashells = 63 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_mike_l2683_268374


namespace NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2683_268344

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- Add necessary fields here
  convex : Bool

/-- Represents an orientation-preserving movement (rotation or translation). -/
structure OrientationPreservingMovement where
  -- Add necessary fields here

/-- Represents a division of a polygon into two parts. -/
structure PolygonDivision (P : ConvexPolygon) where
  part1 : Set (ℝ × ℝ)
  part2 : Set (ℝ × ℝ)
  is_valid : part1 ∪ part2 = Set.univ -- The union of parts equals the whole polygon

/-- Predicate to check if a division is by a broken line. -/
def is_broken_line_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of broken line division

/-- Predicate to check if a division is by a straight line segment. -/
def is_segment_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of straight line segment division

/-- Predicate to check if two parts of a division can be transformed into each other
    by an orientation-preserving movement. -/
def parts_transformable (P : ConvexPolygon) (d : PolygonDivision P) 
    (m : OrientationPreservingMovement) : Prop :=
  sorry -- Definition of transformability

/-- Main theorem statement -/
theorem convex_polygon_division_theorem (P : ConvexPolygon) 
    (h_convex : P.convex = true) :
    (∃ (d : PolygonDivision P) (m : OrientationPreservingMovement), 
      is_broken_line_division P d ∧ parts_transformable P d m) →
    (∃ (d' : PolygonDivision P) (m' : OrientationPreservingMovement),
      is_segment_division P d' ∧ parts_transformable P d' m') :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2683_268344


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l2683_268324

/-- The sum of squares of roots of x^2 - (m+1)x + m - 1 = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let sum_squares := (m + 1)^2 - 2*(m - 1)
  ∀ k : ℝ, sum_squares ≤ (k + 1)^2 - 2*(k - 1) → m = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l2683_268324


namespace NUMINAMATH_CALUDE_T_properties_l2683_268389

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2) / (x + 1)}

theorem T_properties :
  ∃ (n N : ℝ),
    n ∈ T ∧
    (∀ y ∈ T, n ≤ y) ∧
    (∀ y ∈ T, y < N) ∧
    N ∉ T ∧
    (∀ ε > 0, ∃ y ∈ T, N - ε < y) :=
  sorry

end NUMINAMATH_CALUDE_T_properties_l2683_268389


namespace NUMINAMATH_CALUDE_dot_product_CA_CB_l2683_268317

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the center of the circle C
def center_C : ℝ × ℝ := (0, 2)

-- Define a point B on the circle C
def point_B : ℝ × ℝ := sorry

-- State that line l is tangent to circle C at point B
axiom tangent_line : (point_B.1 - point_A.1) * (point_B.1 - center_C.1) + 
                     (point_B.2 - point_A.2) * (point_B.2 - center_C.2) = 0

-- The main theorem
theorem dot_product_CA_CB : 
  (point_A.1 - center_C.1) * (point_B.1 - center_C.1) + 
  (point_A.2 - center_C.2) * (point_B.2 - center_C.2) = 5 :=
sorry

end NUMINAMATH_CALUDE_dot_product_CA_CB_l2683_268317


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l2683_268396

/-- Proves that the ratio of sugar to flour is 3:8 given the conditions in the bakery storage room --/
theorem bakery_storage_ratio : ∀ (flour baking_soda : ℕ),
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  (900 : ℕ) / flour = 3 / 8 :=
by
  sorry

#check bakery_storage_ratio

end NUMINAMATH_CALUDE_bakery_storage_ratio_l2683_268396


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2683_268304

theorem sqrt_sum_equality : Real.sqrt 18 + Real.sqrt 24 / Real.sqrt 3 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2683_268304


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2683_268367

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + i)^2 / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2683_268367


namespace NUMINAMATH_CALUDE_upstream_journey_distance_l2683_268311

/-- Calculates the effective speed of a boat traveling upstream -/
def effectiveSpeed (boatSpeed currentSpeed : ℝ) : ℝ :=
  boatSpeed - currentSpeed

/-- Calculates the distance traveled in one hour given the effective speed -/
def distanceTraveled (effectiveSpeed : ℝ) : ℝ :=
  effectiveSpeed * 1

theorem upstream_journey_distance 
  (boatSpeed : ℝ) 
  (currentSpeed1 currentSpeed2 currentSpeed3 : ℝ) 
  (h1 : boatSpeed = 50)
  (h2 : currentSpeed1 = 10)
  (h3 : currentSpeed2 = 20)
  (h4 : currentSpeed3 = 15) :
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed1) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed2) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed3) = 105 := by
  sorry

end NUMINAMATH_CALUDE_upstream_journey_distance_l2683_268311
