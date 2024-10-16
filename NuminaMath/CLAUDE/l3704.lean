import Mathlib

namespace NUMINAMATH_CALUDE_linda_total_sales_eq_366_9_l3704_370475

/-- Calculates the total sales for Linda's store given the following conditions:
  * Jeans are sold at $22 each
  * Tees are sold at $15 each
  * Jackets are sold at $37 each
  * 10% discount on jackets during the first half of the day
  * 7 tees sold
  * 4 jeans sold
  * 5 jackets sold in total
  * 3 jackets sold during the discount period
-/
def lindaTotalSales : ℝ :=
  let jeanPrice : ℝ := 22
  let teePrice : ℝ := 15
  let jacketPrice : ℝ := 37
  let jacketDiscount : ℝ := 0.1
  let teesSold : ℕ := 7
  let jeansSold : ℕ := 4
  let jacketsSold : ℕ := 5
  let discountedJackets : ℕ := 3
  let fullPriceJackets : ℕ := jacketsSold - discountedJackets
  let discountedJacketPrice : ℝ := jacketPrice * (1 - jacketDiscount)
  
  jeanPrice * jeansSold +
  teePrice * teesSold +
  jacketPrice * fullPriceJackets +
  discountedJacketPrice * discountedJackets

/-- Theorem stating that Linda's total sales at the end of the day equal $366.9 -/
theorem linda_total_sales_eq_366_9 : lindaTotalSales = 366.9 := by
  sorry

end NUMINAMATH_CALUDE_linda_total_sales_eq_366_9_l3704_370475


namespace NUMINAMATH_CALUDE_sandy_comic_books_l3704_370429

/-- Proves that Sandy bought 6 comic books given the initial conditions -/
theorem sandy_comic_books :
  let initial_books : ℕ := 14
  let sold_books : ℕ := initial_books / 2
  let current_books : ℕ := 13
  let bought_books : ℕ := current_books - (initial_books - sold_books)
  bought_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l3704_370429


namespace NUMINAMATH_CALUDE_tennis_tournament_player_count_l3704_370449

/-- Represents a valid number of players in a tennis tournament with 2 vs 2 matches -/
def ValidPlayerCount (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 8 * k + 1

/-- Each player plays against every other player exactly once -/
def EachPlayerPlaysAllOthers (n : ℕ) : Prop :=
  (n - 1) % 2 = 0

/-- The total number of games is an integer -/
def TotalGamesInteger (n : ℕ) : Prop :=
  (n * (n - 1)) % 8 = 0

/-- Main theorem: Characterization of valid player counts in the tennis tournament -/
theorem tennis_tournament_player_count (n : ℕ) :
  (EachPlayerPlaysAllOthers n ∧ TotalGamesInteger n) ↔ ValidPlayerCount n :=
sorry

end NUMINAMATH_CALUDE_tennis_tournament_player_count_l3704_370449


namespace NUMINAMATH_CALUDE_specific_child_group_size_l3704_370465

/-- Represents a group of children with specific age characteristics -/
structure ChildGroup where
  sum_of_ages : ℕ
  age_difference : ℕ
  eldest_age : ℕ

/-- Calculates the number of children in a ChildGroup -/
def number_of_children (group : ChildGroup) : ℕ :=
  sorry

/-- Theorem stating that for a specific ChildGroup, the number of children is 10 -/
theorem specific_child_group_size :
  let group : ChildGroup := {
    sum_of_ages := 50,
    age_difference := 2,
    eldest_age := 14
  }
  number_of_children group = 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_child_group_size_l3704_370465


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3704_370405

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella combinations. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and salami combinations. -/
def rye_salami_combos : ℕ := num_cheeses

/-- Represents the number of sandwiches with white bread and chicken combinations. -/
def white_chicken_combos : ℕ := num_cheeses

/-- Theorem stating the number of possible sandwich combinations. -/
theorem sandwich_combinations :
  num_breads * num_meats * num_cheeses - 
  (turkey_mozzarella_combos + rye_salami_combos + white_chicken_combos) = 193 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3704_370405


namespace NUMINAMATH_CALUDE_largest_y_value_l3704_370415

theorem largest_y_value (y : ℝ) : 
  5 * (4 * y^2 + 12 * y + 15) = y * (4 * y - 25) →
  y ≤ (-85 + 5 * Real.sqrt 97) / 32 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l3704_370415


namespace NUMINAMATH_CALUDE_cell_plan_comparison_l3704_370444

/-- Represents a cell phone plan with a flat fee and per-minute rate -/
structure CellPlan where
  flatFee : ℕ  -- Flat fee in cents
  perMinRate : ℕ  -- Per-minute rate in cents
  
/-- Calculates the cost of a plan for a given number of minutes -/
def planCost (plan : CellPlan) (minutes : ℕ) : ℕ :=
  plan.flatFee + plan.perMinRate * minutes

/-- The three cell phone plans -/
def planX : CellPlan := { flatFee := 0, perMinRate := 15 }
def planY : CellPlan := { flatFee := 2500, perMinRate := 7 }
def planZ : CellPlan := { flatFee := 3000, perMinRate := 6 }

theorem cell_plan_comparison :
  (∀ m : ℕ, m < 313 → planCost planX m ≤ planCost planY m) ∧
  (planCost planY 313 < planCost planX 313) ∧
  (∀ m : ℕ, m < 334 → planCost planX m ≤ planCost planZ m) ∧
  (planCost planZ 334 < planCost planX 334) :=
by sorry


end NUMINAMATH_CALUDE_cell_plan_comparison_l3704_370444


namespace NUMINAMATH_CALUDE_incorrect_statement_l3704_370472

theorem incorrect_statement : ¬ (∀ x : ℝ, |x| = x ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3704_370472


namespace NUMINAMATH_CALUDE_aiNK_probability_l3704_370471

/-- The number of distinct cards labeled with the letters of "NanKai" -/
def total_cards : ℕ := 6

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The number of ways to form "aiNK" from the drawn cards -/
def successful_outcomes : ℕ := 1

/-- The total number of ways to draw 4 cards from 6 -/
def total_outcomes : ℕ := Nat.choose total_cards drawn_cards

/-- The probability of drawing four cards that can form "aiNK" -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem aiNK_probability : probability = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_aiNK_probability_l3704_370471


namespace NUMINAMATH_CALUDE_output_increase_l3704_370446

theorem output_increase (production_increase : Real) (hours_decrease : Real) : 
  production_increase = 0.8 →
  hours_decrease = 0.1 →
  ((1 + production_increase) / (1 - hours_decrease) - 1) * 100 = 100 := by
sorry

end NUMINAMATH_CALUDE_output_increase_l3704_370446


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3704_370478

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  (a = b + 2) →  -- a is two years older than b
  (a + b + c = 27) →  -- The total of the ages of a, b, and c is 27
  (b = 10) →  -- b is 10 years old
  (b : ℚ) / c = 2 / 1 :=  -- The ratio of b's age to c's age is 2:1
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3704_370478


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3704_370448

/-- Represents a parallelogram with given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 22 cm and height 14 cm is 308 square centimeters -/
theorem parallelogram_area_example : 
  let p : Parallelogram := { base := 22, height := 14 }
  area p = 308 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3704_370448


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3704_370457

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ f c ∧
  f c = 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3704_370457


namespace NUMINAMATH_CALUDE_min_value_when_a_2_range_of_a_l3704_370437

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 1|

-- Theorem for the minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = 3 ∧ ∀ x, f 2 x ≥ min :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∃ x, f a x < 2) ↔ -3 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_2_range_of_a_l3704_370437


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l3704_370488

def roundness (n : ℕ) : ℕ := sorry

theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l3704_370488


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3704_370460

theorem rationalize_denominator :
  (Real.sqrt 2) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) = 
  (3 + Real.sqrt 6 + Real.sqrt 15) / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3704_370460


namespace NUMINAMATH_CALUDE_reflection_of_P_l3704_370427

/-- Given a point P in a Cartesian coordinate system, 
    return its coordinates with respect to the origin -/
def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

/-- Theorem: The reflection of point P(2,1) across the origin is (-2,-1) -/
theorem reflection_of_P : reflect_point (2, 1) = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_l3704_370427


namespace NUMINAMATH_CALUDE_fresh_fruit_weight_l3704_370422

theorem fresh_fruit_weight (total_fruit : ℕ) (fresh_ratio frozen_ratio : ℕ) 
  (h_total : total_fruit = 15000)
  (h_ratio : fresh_ratio = 7 ∧ frozen_ratio = 3) :
  (fresh_ratio * total_fruit) / (fresh_ratio + frozen_ratio) = 10500 :=
by sorry

end NUMINAMATH_CALUDE_fresh_fruit_weight_l3704_370422


namespace NUMINAMATH_CALUDE_unique_solution_l3704_370482

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

-- State the theorem
theorem unique_solution :
  ∃! (p q r : ℝ),
    (A p ∪ B q r = {-2, 1, 5}) ∧
    (A p ∩ B q r = {-2}) ∧
    p = -1 ∧ q = -3 ∧ r = -10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3704_370482


namespace NUMINAMATH_CALUDE_polynomial_roots_l3704_370464

/-- The polynomial x^3 - 7x^2 + 11x + 13 -/
def f (x : ℝ) := x^3 - 7*x^2 + 11*x + 13

/-- The set of roots of the polynomial -/
def roots : Set ℝ := {2, 6, -1}

theorem polynomial_roots :
  (∀ x ∈ roots, f x = 0) ∧
  (∀ x : ℝ, f x = 0 → x ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3704_370464


namespace NUMINAMATH_CALUDE_clara_three_times_anna_age_l3704_370489

/-- Proves that Clara was three times Anna's age 41 years ago -/
theorem clara_three_times_anna_age : ∃ x : ℕ, x = 41 ∧ 
  (80 : ℝ) - x = 3 * ((54 : ℝ) - x) := by
  sorry

end NUMINAMATH_CALUDE_clara_three_times_anna_age_l3704_370489


namespace NUMINAMATH_CALUDE_internship_arrangement_l3704_370479

theorem internship_arrangement (n : Nat) (k : Nat) (m : Nat) : 
  n = 5 → k = 4 → m = 2 →
  (Nat.choose k m / 2) * (Nat.factorial n / (Nat.factorial (n - m))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_internship_arrangement_l3704_370479


namespace NUMINAMATH_CALUDE_electricity_scientific_notation_l3704_370474

-- Define the number of kilowatt-hours
def electricity_delivered : ℝ := 105.9e9

-- Theorem to prove the scientific notation
theorem electricity_scientific_notation :
  electricity_delivered = 1.059 * (10 : ℝ)^10 := by
  sorry

end NUMINAMATH_CALUDE_electricity_scientific_notation_l3704_370474


namespace NUMINAMATH_CALUDE_power_function_through_point_l3704_370461

/-- A power function passing through (2, √2/2) has f(9) = 1/3 -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 / 2 →            -- f passes through (2, √2/2)
  f 9 = 1 / 3 :=                     -- f(9) = 1/3
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3704_370461


namespace NUMINAMATH_CALUDE_otimes_example_l3704_370430

-- Define the ⊗ operation
def otimes (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem otimes_example : otimes 4 (otimes 2 (-1)) = 7 := by sorry

end NUMINAMATH_CALUDE_otimes_example_l3704_370430


namespace NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l3704_370435

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 2 * (a (n + 2) + a (n + 1)) - a n

/-- Theorem: All terms in the sequence a_n are perfect squares -/
theorem all_terms_are_perfect_squares :
  ∃ x : ℕ → ℤ, ∀ n : ℕ, a n = (x n)^2 := by
  sorry

end NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l3704_370435


namespace NUMINAMATH_CALUDE_opposite_of_two_l3704_370468

theorem opposite_of_two : -(2 : ℝ) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l3704_370468


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3704_370413

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 75 / 16 ∧ B = 21 / 16 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -4 →
    (6 * x + 3) / (x^2 - 8*x - 48) = A / (x - 12) + B / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3704_370413


namespace NUMINAMATH_CALUDE_units_digit_sum_series_l3704_370456

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_series : ℕ := 
  (units_digit (factorial 1)) + 
  (units_digit ((factorial 2)^2)) + 
  (units_digit (factorial 3)) + 
  (units_digit ((factorial 4)^2)) + 
  (units_digit (factorial 5)) + 
  (units_digit ((factorial 6)^2)) + 
  (units_digit (factorial 7)) + 
  (units_digit ((factorial 8)^2)) + 
  (units_digit (factorial 9)) + 
  (units_digit ((factorial 10)^2))

theorem units_digit_sum_series : units_digit sum_series = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_series_l3704_370456


namespace NUMINAMATH_CALUDE_khali_snow_volume_l3704_370401

/-- Calculates the total volume of snow to be shoveled given sidewalk dimensions and snow depths -/
def total_snow_volume (length width initial_depth additional_depth : ℚ) : ℚ :=
  length * width * (initial_depth + additional_depth)

/-- Proves that the total snow volume for Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let initial_depth : ℚ := 3/4
  let additional_depth : ℚ := 1/4
  total_snow_volume length width initial_depth additional_depth = 90 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l3704_370401


namespace NUMINAMATH_CALUDE_dvd_packs_theorem_l3704_370418

/-- The number of DVD packs that can be bought with a given amount of money -/
def dvd_packs (total_money : ℚ) (pack_cost : ℚ) : ℚ :=
  total_money / pack_cost

/-- Theorem: Given 110 dollars and a pack cost of 11 dollars, 10 DVD packs can be bought -/
theorem dvd_packs_theorem : dvd_packs 110 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_theorem_l3704_370418


namespace NUMINAMATH_CALUDE_intersection_line_through_origin_l3704_370493

/-- Given two lines l₁ and l₂ in the plane, prove that the line passing through
    their intersection point and the origin has the equation x - 10y = 0. -/
theorem intersection_line_through_origin
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {(x, y) | 2 * x + y = 3})
  (h₂ : l₂ = {(x, y) | x + 4 * y = 2})
  (P : ℝ × ℝ)
  (hP : P ∈ l₁ ∧ P ∈ l₂)
  (l : Set (ℝ × ℝ))
  (hl : l = {(x, y) | ∃ t : ℝ, x = t * P.1 ∧ y = t * P.2}) :
  l = {(x, y) | x - 10 * y = 0} :=
sorry

end NUMINAMATH_CALUDE_intersection_line_through_origin_l3704_370493


namespace NUMINAMATH_CALUDE_parabola_equation_l3704_370404

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ x^2 = 2*p*y
  h_focus : focus = (0, p/2)

/-- Theorem: If there exists a point M on parabola C such that |OM| = |MF| = 3,
    then the equation of parabola C is x^2 = 8y -/
theorem parabola_equation (C : Parabola) :
  (∃ M : ℝ × ℝ, C.equation M.1 M.2 ∧ 
    Real.sqrt (M.1^2 + M.2^2) = 3 ∧
    Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) →
  C.p = 4 ∧ ∀ x y, C.equation x y ↔ x^2 = 8*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3704_370404


namespace NUMINAMATH_CALUDE_linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l3704_370421

/-- A function is a bounded multiple function if there exists a constant M > 0 
    such that |f(x)| ≤ M|x| for all real x. -/
def BoundedMultipleFunction (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

/-- The function f(x) = 2x is a bounded multiple function. -/
theorem linear_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ 2 * x) := by
  sorry

/-- The function f(x) = x/(x^2 - x + 3) is a bounded multiple function. -/
theorem rational_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ x / (x^2 - x + 3)) := by
  sorry

/-- An odd function f(x) defined on ℝ that satisfies |f(x₁) - f(x₂)| ≤ 2|x₁ - x₂| 
    for all x₁, x₂ ∈ ℝ is a bounded multiple function. -/
theorem odd_lipschitz_is_bounded_multiple 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_lipschitz : ∀ x₁ x₂, |f x₁ - f x₂| ≤ 2 * |x₁ - x₂|) : 
  BoundedMultipleFunction f := by
  sorry

end NUMINAMATH_CALUDE_linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l3704_370421


namespace NUMINAMATH_CALUDE_cosine_six_arccos_one_third_l3704_370443

theorem cosine_six_arccos_one_third : 
  Real.cos (6 * Real.arccos (1/3)) = 329/729 := by
  sorry

end NUMINAMATH_CALUDE_cosine_six_arccos_one_third_l3704_370443


namespace NUMINAMATH_CALUDE_three_numbers_product_l3704_370469

theorem three_numbers_product (x y z : ℤ) : 
  x + y + z = 165 ∧ 
  7 * x = y - 9 ∧ 
  7 * x = z + 9 → 
  x * y * z = 64328 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_product_l3704_370469


namespace NUMINAMATH_CALUDE_negative_rational_function_interval_l3704_370417

theorem negative_rational_function_interval (x : ℝ) :
  x ≠ 3 →
  ((x - 5) / ((x - 3)^2) < 0) ↔ (3 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_negative_rational_function_interval_l3704_370417


namespace NUMINAMATH_CALUDE_main_theorem_l3704_370484

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The sequence a_n -/
noncomputable def a : Sequence := sorry

/-- The sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := a n + n

/-- The sum of the first n terms of b_n -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem main_theorem :
  (∀ n : ℕ, a n < 0) ∧
  (∀ n : ℕ, a (n + 1) = 2/3 * a n) ∧
  (a 2 * a 5 = 8/27) →
  (∀ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2/3) ∧
  (∀ n : ℕ, a n = -(2/3)^(n-1)) ∧
  (∀ n : ℕ, S n = (n^2 + n + 6)/2 - 3 * (2/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l3704_370484


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3704_370439

/-- Three lines intersect at a single point if and only if m = 22/7 -/
theorem three_lines_intersection (x y m : ℚ) : 
  (y = 3 * x + 2) ∧ 
  (y = -4 * x + 10) ∧ 
  (y = 2 * x + m) → 
  m = 22 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3704_370439


namespace NUMINAMATH_CALUDE_largest_quantity_l3704_370473

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e + 1) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
sorry

end NUMINAMATH_CALUDE_largest_quantity_l3704_370473


namespace NUMINAMATH_CALUDE_problem_solution_l3704_370424

theorem problem_solution (p q r s : ℝ) 
  (h : p^2 + q^2 + r^2 + 4 = s + Real.sqrt (p + q + r - s)) : 
  s = 5/4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3704_370424


namespace NUMINAMATH_CALUDE_my_matrix_is_projection_l3704_370486

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![9/25, 18/45],
    ![12/25, 27/45]]

theorem my_matrix_is_projection : projection_matrix my_matrix := by
  sorry

end NUMINAMATH_CALUDE_my_matrix_is_projection_l3704_370486


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3704_370423

/-- The radius of the inscribed circle of a triangle with sides 6, 8, and 10 is 2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3704_370423


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3704_370455

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - (num_pencils : ℝ) * pencil_price) / (num_pens : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3704_370455


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3704_370492

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Finds the point symmetric to a given point with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The theorem stating that the point symmetric to (-3, -4, 5) with respect to the xOz plane is (-3, 4, 5) -/
theorem symmetric_point_theorem :
  let A : Point3D := { x := -3, y := -4, z := 5 }
  symmetricPointXOZ A = { x := -3, y := 4, z := 5 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l3704_370492


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l3704_370467

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 64 ∧ b = 64 ∧ c < a ∧ a + b + c = 180 → a - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l3704_370467


namespace NUMINAMATH_CALUDE_intersection_range_l3704_370436

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}
def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

-- State the theorem
theorem intersection_range (b : ℝ) : 
  (M ∩ N b).Nonempty → b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l3704_370436


namespace NUMINAMATH_CALUDE_minimum_cost_for_boxes_l3704_370452

/-- The dimensions of a box in inches -/
def box_dimensions : Fin 3 → ℕ
  | 0 => 20
  | 1 => 20
  | 2 => 15
  | _ => 0

/-- The volume of a single box in cubic inches -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The cost of a single box in cents -/
def box_cost : ℕ := 50

/-- The total volume of the collection in cubic inches -/
def collection_volume : ℕ := 3060000

/-- The number of boxes needed to package the collection -/
def boxes_needed : ℕ := (collection_volume + box_volume - 1) / box_volume

theorem minimum_cost_for_boxes : 
  boxes_needed * box_cost = 25500 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_for_boxes_l3704_370452


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l3704_370454

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ :=
  0.50 + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on each other axle -/
def numAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_calculation :
  let x := numAxles 18 2 4
  toll x = 2 :=
by sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l3704_370454


namespace NUMINAMATH_CALUDE_positive_real_properties_l3704_370410

theorem positive_real_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b - a * b = 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4 * y - x * y = 0 ∧ x + 4 * y < a + 4 * b) →
  (a + 2 * b ≥ 6 + 4 * Real.sqrt 2) ∧
  (16 / a^2 + 1 / b^2 ≥ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_properties_l3704_370410


namespace NUMINAMATH_CALUDE_solution_value_l3704_370403

-- Define the equations
def equation1 (m x : ℝ) : Prop := (m + 3) * x^(|m| - 2) + 6 * m = 0
def equation2 (n x : ℝ) : Prop := n * x - 5 = x * (3 - n)

-- Define the linearity condition for equation1
def equation1_is_linear (m : ℝ) : Prop := |m| - 2 = 0

-- Define the main theorem
theorem solution_value (m n x : ℝ) :
  (∀ y : ℝ, equation1 m y ↔ equation2 n y) →
  equation1_is_linear m →
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_value_l3704_370403


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3704_370407

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem: There are 10 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball. -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 10 := by
  sorry

#eval distribute_balls 6 3

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3704_370407


namespace NUMINAMATH_CALUDE_fourth_month_sale_l3704_370440

/-- Given the sales for 5 out of 6 months and the average sale for 6 months, 
    prove that the sale in the fourth month must be 8230. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7920)
  (h3 : sale3 = 7855)
  (h5 : sale5 = 7560)
  (h6 : sale6 = 6000)
  (h_avg : avg_sale = 7500)
  : ∃ (sale4 : ℕ), sale4 = 8230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = avg_sale :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l3704_370440


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l3704_370400

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of Type A televisions -/
def typeA : ℕ := 4

/-- The number of Type B televisions -/
def typeB : ℕ := 5

/-- The total number of televisions to be chosen -/
def totalChosen : ℕ := 3

/-- The number of ways to choose the televisions -/
def waysToChoose : ℕ := choose typeA 2 * choose typeB 1 + choose typeA 1 * choose typeB 2

theorem tv_selection_theorem : waysToChoose = 70 := by sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l3704_370400


namespace NUMINAMATH_CALUDE_children_tickets_count_l3704_370483

/-- Proves that the number of children's tickets is 21 given the ticket prices, total amount paid, and total number of tickets. -/
theorem children_tickets_count (adult_price child_price total_amount total_tickets : ℕ)
  (h_adult_price : adult_price = 8)
  (h_child_price : child_price = 5)
  (h_total_amount : total_amount = 201)
  (h_total_tickets : total_tickets = 33) :
  ∃ (adult_count child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_amount ∧
    child_count = 21 :=
by sorry

end NUMINAMATH_CALUDE_children_tickets_count_l3704_370483


namespace NUMINAMATH_CALUDE_second_bucket_capacity_l3704_370496

/-- Proves that given a tank of 48 liters and two buckets, where one bucket has a capacity of 4 liters
    and is used 4 times less than the other bucket to fill the tank, the capacity of the second bucket is 3 liters. -/
theorem second_bucket_capacity
  (tank_capacity : ℕ)
  (first_bucket_capacity : ℕ)
  (usage_difference : ℕ)
  (h1 : tank_capacity = 48)
  (h2 : first_bucket_capacity = 4)
  (h3 : usage_difference = 4)
  (h4 : ∃ (second_bucket_capacity : ℕ),
    tank_capacity / first_bucket_capacity = tank_capacity / second_bucket_capacity - usage_difference) :
  ∃ (second_bucket_capacity : ℕ), second_bucket_capacity = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_bucket_capacity_l3704_370496


namespace NUMINAMATH_CALUDE_dog_food_duration_aunt_gemma_dog_food_duration_l3704_370480

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, number of sacks, and weight of each sack. -/
theorem dog_food_duration (num_dogs : ℕ) (feedings_per_day : ℕ) (food_per_meal : ℕ)
                          (num_sacks : ℕ) (sack_weight_kg : ℕ) : ℕ :=
  let total_food_grams : ℕ := num_sacks * sack_weight_kg * 1000
  let daily_consumption : ℕ := num_dogs * food_per_meal * feedings_per_day
  total_food_grams / daily_consumption

/-- Proves that given Aunt Gemma's specific conditions, the dog food will last for 50 days. -/
theorem aunt_gemma_dog_food_duration : 
  dog_food_duration 4 2 250 2 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_duration_aunt_gemma_dog_food_duration_l3704_370480


namespace NUMINAMATH_CALUDE_problem_solution_l3704_370453

theorem problem_solution : 9 - (3 / (1 / 3) + 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3704_370453


namespace NUMINAMATH_CALUDE_mrs_blue_tomato_yield_l3704_370434

/-- Represents the dimensions of a rectangular vegetable patch in steps -/
structure PatchDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a vegetable patch -/
def expected_tomato_yield (dimensions : PatchDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (dimensions.length : ℝ) * step_length * (dimensions.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected tomato yield for Mrs. Blue's vegetable patch -/
theorem mrs_blue_tomato_yield :
  let dimensions : PatchDimensions := ⟨18, 25⟩
  let step_length : ℝ := 3
  let yield_per_sqft : ℝ := 3 / 4
  expected_tomato_yield dimensions step_length yield_per_sqft = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_tomato_yield_l3704_370434


namespace NUMINAMATH_CALUDE_base_five_digits_of_1837_l3704_370470

theorem base_five_digits_of_1837 (n : Nat) (h : n = 1837) :
  (Nat.log 5 n + 1 : Nat) = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1837_l3704_370470


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3704_370442

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point
def given_point : Point := (2, -6)

-- Theorem stating that the coordinates of the given point with respect to the origin are (2, -6)
theorem coordinates_wrt_origin (p : Point) : p = given_point → p = (2, -6) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3704_370442


namespace NUMINAMATH_CALUDE_power_product_evaluation_l3704_370451

theorem power_product_evaluation : 
  let a : ℕ := 2
  (a^3 * a^4 : ℕ) = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l3704_370451


namespace NUMINAMATH_CALUDE_nods_per_kilometer_l3704_370485

/-- Given the relationships between winks, nods, leaps, and kilometers,
    prove that the number of nods in one kilometer is equal to qts / (pru) -/
theorem nods_per_kilometer
  (p q r s t u : ℚ)
  (h1 : p * 1 = q)  -- p winks equal q nods
  (h2 : r * 1 = s)  -- r leaps equal s winks
  (h3 : t * 1 = u)  -- t leaps are equivalent to u kilometers
  : 1 = q * t * s / (p * r * u) :=
sorry

end NUMINAMATH_CALUDE_nods_per_kilometer_l3704_370485


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l3704_370431

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  relative_speed_ms * passing_time

/-- Proof that a train with speed 60 km/hr passing a man running at 6 km/hr in the opposite direction
    in approximately 29.997600191984645 seconds has a length of approximately 550 meters. -/
theorem train_length_proof : 
  ∃ ε > 0, |train_length 60 6 29.997600191984645 - 550| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l3704_370431


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3704_370494

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
  (∀ x, 9 * x^2 = 4 * (3 * x - 1)) →
  (∀ x, a * x^2 + b * x + c = 0) →
  a = 9 ∧ b = -12 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3704_370494


namespace NUMINAMATH_CALUDE_bisection_method_termination_condition_l3704_370495

/-- The bisection method termination condition -/
def bisection_termination (x₁ x₂ ε : ℝ) : Prop :=
  |x₁ - x₂| < ε

/-- Theorem stating the correct termination condition for the bisection method -/
theorem bisection_method_termination_condition 
  (f : ℝ → ℝ) (a b x₁ x₂ ε : ℝ) 
  (hf : Continuous f) 
  (ha : f a < 0) 
  (hb : f b > 0) 
  (hε : ε > 0) 
  (hx₁ : x₁ ∈ Set.Icc a b) 
  (hx₂ : x₂ ∈ Set.Icc a b) :
  bisection_termination x₁ x₂ ε ↔ 
    (∃ x ∈ Set.Icc x₁ x₂, f x = 0) ∧ 
    (∀ y ∈ Set.Icc a b, f y = 0 → y ∈ Set.Icc x₁ x₂) := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_termination_condition_l3704_370495


namespace NUMINAMATH_CALUDE_bbq_ice_per_person_l3704_370411

/-- Given the conditions of Chad's BBQ, prove that the amount of ice needed per person is 2 pounds. -/
theorem bbq_ice_per_person (people : ℕ) (pack_price : ℚ) (pack_size : ℕ) (total_spent : ℚ) :
  people = 15 →
  pack_price = 3 →
  pack_size = 10 →
  total_spent = 9 →
  (total_spent / pack_price * pack_size) / people = 2 := by
  sorry

#check bbq_ice_per_person

end NUMINAMATH_CALUDE_bbq_ice_per_person_l3704_370411


namespace NUMINAMATH_CALUDE_negative_cube_squared_l3704_370441

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l3704_370441


namespace NUMINAMATH_CALUDE_game_ends_in_25_rounds_l3704_370416

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13,
    round := 0 }

/-- Determines if the game has ended (i.e., if any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The theorem to prove -/
theorem game_ends_in_25_rounds :
  ∃ finalState : GameState,
    finalState.round = 25 ∧
    gameEnded finalState ∧
    (∀ prevState : GameState, prevState.round < 25 → ¬gameEnded prevState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_25_rounds_l3704_370416


namespace NUMINAMATH_CALUDE_bob_always_wins_l3704_370490

/-- The game described in the problem -/
def Game (n : ℕ) : Prop :=
  ∀ (A : Fin (n + 1) → Finset (Fin (2^n))),
    (∀ i, (A i).card = 2^(n-1)) →
    ∃ (a : Fin (n + 1) → Fin (2^n)),
      ∀ t : Fin (2^n),
        ∃ i s, s ∈ A i ∧ (s + a i : Fin (2^n)) = t

/-- Bob always has a winning strategy for any positive n -/
theorem bob_always_wins :
  ∀ n : ℕ, n > 0 → Game n :=
sorry

end NUMINAMATH_CALUDE_bob_always_wins_l3704_370490


namespace NUMINAMATH_CALUDE_solution_set_l3704_370406

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
def condition1 : Prop := -Real.sqrt (1 / ((a - b)^2)) * (b - a) = 1
def condition2 : Prop := 3*x - 4*a ≤ a - 2*x
def condition3 : Prop := (3*x + 2*b) / 5 > b

-- State the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a x) (h3 : condition3 b x) :
  b < x ∧ x ≤ a :=
sorry

end NUMINAMATH_CALUDE_solution_set_l3704_370406


namespace NUMINAMATH_CALUDE_min_value_of_function_l3704_370466

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 3 / (4 * x) ≥ Real.sqrt 3 ∧ ∃ y > 0, y + 3 / (4 * y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3704_370466


namespace NUMINAMATH_CALUDE_min_cards_for_four_of_a_kind_standard_deck_l3704_370426

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (num_ranks : Nat)
  (cards_per_rank : Nat)
  (num_jokers : Nat)

/-- Calculates the minimum number of cards needed to guarantee "a four of a kind" -/
def min_cards_for_four_of_a_kind (d : Deck) : Nat :=
  d.num_jokers + (d.num_ranks * (d.cards_per_rank - 1)) + 1

/-- Theorem stating the minimum number of cards needed for "a four of a kind" in a standard deck -/
theorem min_cards_for_four_of_a_kind_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    num_ranks := 13,
    cards_per_rank := 4,
    num_jokers := 2
  }
  min_cards_for_four_of_a_kind standard_deck = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_for_four_of_a_kind_standard_deck_l3704_370426


namespace NUMINAMATH_CALUDE_positive_A_value_l3704_370433

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 4 = 65) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3704_370433


namespace NUMINAMATH_CALUDE_height_ratio_of_isosceles_triangles_l3704_370463

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  side_length : ℝ
  base_length : ℝ
  height : ℝ

/-- The problem statement -/
theorem height_ratio_of_isosceles_triangles
  (triangle_A triangle_B : IsoscelesTriangle)
  (h_vertical_angle : 180 - 2 * triangle_A.base_angle = 180 - 2 * triangle_B.base_angle)
  (h_base_angle_A : triangle_A.base_angle = 40)
  (h_base_angle_B : triangle_B.base_angle = 50)
  (h_side_ratio : triangle_B.side_length / triangle_A.side_length = 5 / 3)
  (h_area_ratio : (triangle_B.base_length * triangle_B.height) / (triangle_A.base_length * triangle_A.height) = 25 / 9) :
  triangle_B.height / triangle_A.height = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_height_ratio_of_isosceles_triangles_l3704_370463


namespace NUMINAMATH_CALUDE_sum_of_powers_lower_bound_l3704_370447

theorem sum_of_powers_lower_bound 
  (x y z : ℝ) 
  (n : ℕ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) 
  (pos_n : 0 < n) : 
  x^n + y^n + z^n ≥ 1 / (3^(n-1)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_lower_bound_l3704_370447


namespace NUMINAMATH_CALUDE_journey_length_l3704_370428

theorem journey_length (first_part second_part third_part total : ℝ) 
  (h1 : first_part = (1/4) * total)
  (h2 : second_part = 30)
  (h3 : third_part = (1/3) * total)
  (h4 : total = first_part + second_part + third_part) :
  total = 72 := by
  sorry

end NUMINAMATH_CALUDE_journey_length_l3704_370428


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l3704_370409

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 12 * x
  distance_to_focus : 6 = |x + 3| -- The focus is at (3, 0)

/-- The coordinates of a point on the parabola y² = 12x with distance 6 to the focus -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  (p.x = 3 ∧ p.y = 6) ∨ (p.x = 3 ∧ p.y = -6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l3704_370409


namespace NUMINAMATH_CALUDE_base_nine_solution_l3704_370477

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_nine_solution :
  ∃! b : Nat, b > 0 ∧ 
    to_decimal [1, 7, 2] b + to_decimal [1, 4, 5] b = to_decimal [3, 2, 7] b :=
by sorry

end NUMINAMATH_CALUDE_base_nine_solution_l3704_370477


namespace NUMINAMATH_CALUDE_dads_toothpaste_usage_l3704_370432

/-- Represents the amount of toothpaste used by Anne's dad at each brushing -/
def dads_toothpaste_use : ℝ := 3

/-- Theorem stating that Anne's dad uses 3 grams of toothpaste at each brushing -/
theorem dads_toothpaste_usage 
  (total_toothpaste : ℝ) 
  (moms_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_to_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : moms_usage = 2)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_to_empty = 5)
  : dads_toothpaste_use = 3 := by
  sorry

#check dads_toothpaste_usage

end NUMINAMATH_CALUDE_dads_toothpaste_usage_l3704_370432


namespace NUMINAMATH_CALUDE_equilateral_triangle_properties_l3704_370438

/-- Proves properties of an equilateral triangle with given area and side length -/
theorem equilateral_triangle_properties :
  ∀ (area base altitude perimeter : ℝ),
  area = 450 →
  base = 25 →
  area = (1/2) * base * altitude →
  perimeter = 3 * base →
  altitude = 36 ∧ perimeter = 75 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_properties_l3704_370438


namespace NUMINAMATH_CALUDE_carols_rectangle_length_l3704_370445

theorem carols_rectangle_length (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 30)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 12 := by
  sorry

#check carols_rectangle_length

end NUMINAMATH_CALUDE_carols_rectangle_length_l3704_370445


namespace NUMINAMATH_CALUDE_sum_reciprocal_bound_l3704_370408

theorem sum_reciprocal_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a + b = 2) :
  c / a + c / b ≥ 2 * c ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ c / a' + c / b' > ε := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_bound_l3704_370408


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3704_370402

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point (m, n) on the line ax + by + 2c = 0, 
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c m n : ℝ) : 
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a * m + b * n + 2 * c = 0 →  -- Point (m, n) lies on the line
  ∃ (m₀ n₀ : ℝ), a * m₀ + b * n₀ + 2 * c = 0 ∧ 
    ∀ (m' n' : ℝ), a * m' + b * n' + 2 * c = 0 → m₀^2 + n₀^2 ≤ m'^2 + n'^2 ∧
    m₀^2 + n₀^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3704_370402


namespace NUMINAMATH_CALUDE_square_sum_equals_34_l3704_370412

theorem square_sum_equals_34 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 4.5) : a^2 + b^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_34_l3704_370412


namespace NUMINAMATH_CALUDE_m_range_l3704_370462

/-- Proposition p: The quadratic equation with real coefficients x^2 + mx + 2 = 0 has imaginary roots -/
def prop_p (m : ℝ) : Prop := m^2 - 8 < 0

/-- Proposition q: For the equation 2x^2 - 4(m-1)x + m^2 + 7 = 0 (m ∈ ℝ), 
    the sum of the moduli of its two imaginary roots does not exceed 4√2 -/
def prop_q (m : ℝ) : Prop := 16*(m-1)^2 - 8*(m^2 + 7) < 0

/-- The range of m when both propositions p and q are true -/
theorem m_range (m : ℝ) : prop_p m ∧ prop_q m ↔ -1 < m ∧ m < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3704_370462


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l3704_370499

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (Real.sin θ + Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ),
  polar_equation r θ →
  x = r * Real.cos θ →
  y = r * Real.sin θ →
  line_equation x y :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l3704_370499


namespace NUMINAMATH_CALUDE_range_of_m_l3704_370450

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, ((x - 4) / 3)^2 > 4 → x^2 - 2*x + 1 - m^2 > 0) →
  (∃ x, ((x - 4) / 3)^2 > 4 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3704_370450


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l3704_370481

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line2D where
  slope : ℝ
  point : ℝ × ℝ

/-- The symmetric point of a given point with respect to a center point -/
def symmetricPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * (p.1 - l.point.1) + l.point.2

/-- Two lines are symmetric about a point if the reflection of any point on one line
    through the center point lies on the other line -/
def symmetricLines (l1 l2 : Line2D) (center : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ, pointOnLine l1 p → pointOnLine l2 (symmetricPoint p center)

theorem symmetric_line_passes_through_fixed_point :
  ∀ (k : ℝ) (l1 l2 : Line2D),
    l1.slope = k ∧
    l1.point = (4, 0) ∧
    symmetricLines l1 l2 (2, 1) →
    pointOnLine l2 (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l3704_370481


namespace NUMINAMATH_CALUDE_unique_student_count_l3704_370491

theorem unique_student_count : ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 18 ∧ n = 424 := by
  sorry

end NUMINAMATH_CALUDE_unique_student_count_l3704_370491


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l3704_370425

def committee_size : ℕ := 7
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_selected : ℕ := 2

theorem probability_at_least_one_girl :
  let total_combinations := Nat.choose committee_size num_selected
  let combinations_with_no_girls := Nat.choose num_boys num_selected
  let favorable_combinations := total_combinations - combinations_with_no_girls
  (favorable_combinations : ℚ) / total_combinations = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l3704_370425


namespace NUMINAMATH_CALUDE_yoga_studio_men_count_l3704_370497

theorem yoga_studio_men_count :
  ∀ (num_men : ℕ) (avg_weight_men avg_weight_women avg_weight_all : ℝ),
    avg_weight_men = 190 →
    avg_weight_women = 120 →
    num_men + 6 = 14 →
    (num_men * avg_weight_men + 6 * avg_weight_women) / 14 = avg_weight_all →
    avg_weight_all = 160 →
    num_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_men_count_l3704_370497


namespace NUMINAMATH_CALUDE_largest_gcd_of_four_integers_l3704_370414

theorem largest_gcd_of_four_integers (a b c d : ℕ+) : 
  a + b + c + d = 1105 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ c ∧ k ∣ d) →
  (∀ m : ℕ, m ∣ a ∧ m ∣ b ∧ m ∣ c ∧ m ∣ d → m ≤ 221) ∧
  (∃ n : ℕ, n = 221 ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_four_integers_l3704_370414


namespace NUMINAMATH_CALUDE_eggs_per_box_l3704_370476

/-- Given that Maria has 3 boxes of eggs and a total of 21 eggs, 
    prove that each box contains 7 eggs. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) 
  (h1 : total_eggs = 21) (h2 : num_boxes = 3) : 
  total_eggs / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3704_370476


namespace NUMINAMATH_CALUDE_always_quadratic_in_x_l3704_370487

/-- A quadratic equation in x is of the form ax² + bx + c = 0 where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (m²+1)x² - mx - 3 = 0 is quadratic in x for all real m -/
theorem always_quadratic_in_x (m : ℝ) : 
  is_quadratic_in_x (m^2 + 1) (-m) (-3) := by sorry

end NUMINAMATH_CALUDE_always_quadratic_in_x_l3704_370487


namespace NUMINAMATH_CALUDE_partial_fraction_A_value_l3704_370459

-- Define the polynomial in the denominator
def p (x : ℝ) : ℝ := x^4 - 2*x^3 - 29*x^2 + 70*x + 120

-- Define the partial fraction decomposition
def partial_fraction (x A B C D : ℝ) : Prop :=
  1 / p x = A / (x + 4) + B / (x - 2) + C / (x - 2)^2 + D / (x - 3)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction x A B C D) → A = -1/252 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_A_value_l3704_370459


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3704_370458

theorem election_votes_theorem (candidates : ℕ) (winner_percentage : ℝ) (majority : ℕ) 
  (h1 : candidates = 4)
  (h2 : winner_percentage = 0.7)
  (h3 : majority = 3000) :
  ∃ total_votes : ℕ, 
    (↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = majority) ∧ 
    total_votes = 7500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3704_370458


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l3704_370420

-- Define the lines and points
def line1 (x y : ℝ) : Prop := x + 3 * y - 2 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 6 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x_p y_p x_q y_q : ℝ) : Prop :=
  x₀ = (x_p + x_q) / 2 ∧ y₀ = (y_p + y_q) / 2

-- State the theorem
theorem midpoint_ratio_range (x₀ y₀ x_p y_p x_q y_q : ℝ) :
  line1 x_p y_p →
  line2 x_q y_q →
  is_midpoint x₀ y₀ x_p y_p x_q y_q →
  y₀ < x₀ + 2 →
  (y₀ / x₀ < -1/3 ∨ y₀ / x₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l3704_370420


namespace NUMINAMATH_CALUDE_total_swim_distance_l3704_370419

/-- The total distance Molly swam on Saturday in meters -/
def saturday_distance : ℕ := 400

/-- The total distance Molly swam on Sunday in meters -/
def sunday_distance : ℕ := 300

/-- The theorem states that the total distance Molly swam in all four pools
    is equal to the sum of the distances she swam on Saturday and Sunday -/
theorem total_swim_distance :
  saturday_distance + sunday_distance = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_swim_distance_l3704_370419


namespace NUMINAMATH_CALUDE_factor_polynomial_l3704_370498

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 300 * x^13 = 75 * x^7 * (1 - 4 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3704_370498
