import Mathlib

namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l3860_386022

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (hp_markup : markup_percentage = 50) 
  (hp_discount : discount_percentage = 10) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discounted_price := marked_price * (1 - discount_percentage / 100)
  let profit := discounted_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 35 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l3860_386022


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3860_386077

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3860_386077


namespace NUMINAMATH_CALUDE_largest_divisor_fifth_largest_divisor_l3860_386065

def n : ℕ := 1516000000

-- Define a function to get the kth largest divisor
def kthLargestDivisor (k : ℕ) : ℕ := sorry

-- The largest divisor of n is itself
theorem largest_divisor : kthLargestDivisor 1 = n := sorry

-- The fifth-largest divisor of n is 94,750,000
theorem fifth_largest_divisor : kthLargestDivisor 5 = 94750000 := sorry

end NUMINAMATH_CALUDE_largest_divisor_fifth_largest_divisor_l3860_386065


namespace NUMINAMATH_CALUDE_bills_initial_money_l3860_386076

def total_initial_money : ℕ := 42
def num_pizzas : ℕ := 3
def pizza_cost : ℕ := 11
def bill_final_money : ℕ := 39

theorem bills_initial_money :
  let frank_spent := num_pizzas * pizza_cost
  let frank_leftover := total_initial_money - frank_spent
  let bill_initial := bill_final_money - frank_leftover
  bill_initial = 30 := by
sorry

end NUMINAMATH_CALUDE_bills_initial_money_l3860_386076


namespace NUMINAMATH_CALUDE_ladder_rungs_count_ladder_rungs_count_proof_l3860_386079

theorem ladder_rungs_count : ℕ → Prop :=
  fun n =>
    let middle_rung := n / 2
    let final_position := middle_rung + 5 - 7 + 8 + 7
    (n % 2 = 1) ∧ (final_position = n) → n = 27

-- The proof is omitted
theorem ladder_rungs_count_proof : ladder_rungs_count 27 := by sorry

end NUMINAMATH_CALUDE_ladder_rungs_count_ladder_rungs_count_proof_l3860_386079


namespace NUMINAMATH_CALUDE_range_of_a_l3860_386014

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1) ^ x₁ > (2 * a - 1) ^ x₂) →
  1/2 < a ∧ a ≤ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3860_386014


namespace NUMINAMATH_CALUDE_intersection_A_B_C_subset_intersection_A_B_l3860_386058

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 3*a*x + 2*a^2 < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

-- Theorem for the range of a such that C is a subset of A ∩ B
theorem C_subset_intersection_A_B (a : ℝ) : 
  C a ⊆ (A ∩ B) ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_C_subset_intersection_A_B_l3860_386058


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l3860_386069

/-- Given Isabella's hair length at the end of the year and the amount it grew,
    prove that her initial hair length is equal to the final length minus the growth. -/
theorem isabellas_hair_length (final_length growth : ℕ) (h : final_length = 24 ∧ growth = 6) :
  final_length - growth = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l3860_386069


namespace NUMINAMATH_CALUDE_unique_element_condition_l3860_386045

def A (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x - 1 = 0}

theorem unique_element_condition (a : ℝ) : (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_element_condition_l3860_386045


namespace NUMINAMATH_CALUDE_equalize_volume_l3860_386008

-- Define the volumes in milliliters
def transparent_volume : ℚ := 12400
def opaque_volume : ℚ := 7600

-- Define the conversion factor from milliliters to liters
def ml_to_l : ℚ := 1000

-- Define the function to calculate the volume to be transferred
def volume_to_transfer : ℚ :=
  (transparent_volume - opaque_volume) / 2

-- Theorem statement
theorem equalize_volume :
  volume_to_transfer = 2400 ∧
  volume_to_transfer / ml_to_l = (12 / 5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equalize_volume_l3860_386008


namespace NUMINAMATH_CALUDE_problem_solution_l3860_386000

theorem problem_solution (x : ℝ) : 
  (7/11) * (5/13) * x = 48 → (315/100) * x = 617.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3860_386000


namespace NUMINAMATH_CALUDE_equation_solution_l3860_386025

theorem equation_solution : 
  {x : ℝ | x + 36 / (x - 5) = -12} = {-8, 3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3860_386025


namespace NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l3860_386090

theorem dice_sum_divisibility_probability (n : ℕ) (p q r : ℝ) : 
  p ≥ 0 → q ≥ 0 → r ≥ 0 → p + q + r = 1 → 
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l3860_386090


namespace NUMINAMATH_CALUDE_ellipse_properties_l3860_386049

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Properties of a specific ellipse -/
theorem ellipse_properties (C : Ellipse) (P : Point) :
  P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1 →  -- P is on the ellipse
  P.x = 1 →                           -- P's x-coordinate is 1
  P.y = Real.sqrt 2 / 2 →             -- P's y-coordinate is √2/2
  (∃ F₁ F₂ : Point, |P.x - F₁.x| + |P.y - F₁.y| + |P.x - F₂.x| + |P.y - F₂.y| = 2 * Real.sqrt 2) →  -- Distance sum to foci is 2√2
  (C.a^2 = 2 ∧ C.b^2 = 1) ∧           -- Standard equation of C is x²/2 + y² = 1
  (∃ (A B O : Point) (l : Set Point),
    O = ⟨0, 0⟩ ∧                      -- O is the origin
    F₂ ∈ l ∧ A ∈ l ∧ B ∈ l ∧          -- l passes through F₂, A, and B
    A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1 ∧  -- A is on the ellipse
    B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1 ∧  -- B is on the ellipse
    (∀ A' B' : Point,
      A' ∈ l → B' ∈ l →
      A'.x^2 / C.a^2 + A'.y^2 / C.b^2 = 1 →
      B'.x^2 / C.a^2 + B'.y^2 / C.b^2 = 1 →
      abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 ≥
      abs ((A'.x - O.x) * (B'.y - O.y) - (B'.x - O.x) * (A'.y - O.y)) / 2) ∧
    abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = Real.sqrt 2 / 2) -- Max area of AOB is √2/2
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3860_386049


namespace NUMINAMATH_CALUDE_baker_cakes_remaining_l3860_386094

/-- Given the initial number of cakes, additional cakes made, and cakes sold,
    prove that the number of cakes remaining is equal to 67. -/
theorem baker_cakes_remaining 
  (initial_cakes : ℕ) 
  (additional_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (h1 : initial_cakes = 62)
  (h2 : additional_cakes = 149)
  (h3 : sold_cakes = 144) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_remaining_l3860_386094


namespace NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l3860_386060

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (Nat.factorial n + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l3860_386060


namespace NUMINAMATH_CALUDE_team_a_two_projects_probability_l3860_386009

/-- The number of ways to distribute n identical objects into k distinct boxes,
    where each box must contain at least one object. -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The probability of team A contracting exactly two projects out of five projects
    distributed among four teams, where each team must contract at least one project. -/
theorem team_a_two_projects_probability :
  let total_distributions := stars_and_bars 5 4
  let favorable_distributions := stars_and_bars 3 3
  (favorable_distributions : ℚ) / total_distributions = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_team_a_two_projects_probability_l3860_386009


namespace NUMINAMATH_CALUDE_angle_c_90_sufficient_not_necessary_l3860_386061

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Theorem stating that in a triangle ABC, angle C = 90° is a sufficient 
    but not necessary condition for cos A + sin A = cos B + sin B -/
theorem angle_c_90_sufficient_not_necessary (t : Triangle) :
  (t.C = π / 2 → Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B) ∧
  ∃ t' : Triangle, Real.cos t'.A + Real.sin t'.A = Real.cos t'.B + Real.sin t'.B ∧ t'.C ≠ π / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_c_90_sufficient_not_necessary_l3860_386061


namespace NUMINAMATH_CALUDE_price_reduction_profit_l3860_386039

/-- Represents the daily sales and profit model for a product in a shopping mall -/
structure SalesModel where
  baseItems : ℕ  -- Base number of items sold per day
  baseProfit : ℕ  -- Base profit per item in yuan
  salesIncrease : ℕ  -- Additional items sold per yuan of price reduction
  priceReduction : ℕ  -- Amount of price reduction in yuan

/-- Calculates the daily profit given a SalesModel -/
def dailyProfit (model : SalesModel) : ℕ :=
  let newItems := model.baseItems + model.salesIncrease * model.priceReduction
  let newProfit := model.baseProfit - model.priceReduction
  newItems * newProfit

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (model : SalesModel) 
  (h1 : model.baseItems = 30)
  (h2 : model.baseProfit = 50)
  (h3 : model.salesIncrease = 2)
  (h4 : model.priceReduction = 20) :
  dailyProfit model = 2100 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_profit_l3860_386039


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3860_386016

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 = (1 : ℝ) / 4 →
  a 2 * a 8 = 4 * (a 5 - 1) →
  a 4 + a 5 + a 6 + a 7 + a 8 = 31 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3860_386016


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3860_386007

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 53 ∧ ∀ (s : ℝ), -3 * s^2 + 24 * s + 5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3860_386007


namespace NUMINAMATH_CALUDE_paint_house_theorem_l3860_386048

/-- Represents the time taken to paint a house given the number of people -/
def paint_time (people : ℝ) (hours : ℝ) : Prop :=
  people * hours = 5 * 10

theorem paint_house_theorem :
  paint_time 5 10 → paint_time 4 12.5 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_house_theorem_l3860_386048


namespace NUMINAMATH_CALUDE_cos_315_degrees_l3860_386080

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l3860_386080


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l3860_386019

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a * Real.log x

theorem monotonicity_and_range :
  ∀ (a : ℝ), a ≤ 0 →
  (∀ (x : ℝ), x > 0 → f (-2) x < f (-2) (1 + Real.sqrt 2) → x < 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f (-2) x > f (-2) (1 + Real.sqrt 2) → x > 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f a x > (1/2)*(2*Real.exp 1 + 1)*a ↔ a ∈ Set.Ioo (-2*(Real.exp 1)^2/(2*Real.exp 1 + 1)) 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l3860_386019


namespace NUMINAMATH_CALUDE_factorization_implies_c_value_l3860_386026

theorem factorization_implies_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 3*x + c = (x + 1)*(x + 2)) → c = 2 := by
sorry

end NUMINAMATH_CALUDE_factorization_implies_c_value_l3860_386026


namespace NUMINAMATH_CALUDE_line_x_intercept_l3860_386004

/-- Given a line passing through the point (3, 4) with slope 2, its x-intercept is 1. -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 2 * x + (4 - 2 * 3)) →  -- Line equation derived from point-slope form
  f 4 = 3 →                           -- Line passes through (3, 4)
  f 0 = 1 :=                          -- x-intercept is at (1, 0)
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l3860_386004


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l3860_386018

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in billions of yuan -/
def gdp_billions : ℝ := 53100

/-- Theorem stating that the GDP in billions is equal to its scientific notation -/
theorem gdp_scientific_notation : 
  to_scientific_notation gdp_billions = ScientificNotation.mk 5.31 12 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l3860_386018


namespace NUMINAMATH_CALUDE_complex_magnitude_reciprocal_one_plus_i_l3860_386020

theorem complex_magnitude_reciprocal_one_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 1 / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_complex_magnitude_reciprocal_one_plus_i_l3860_386020


namespace NUMINAMATH_CALUDE_travel_options_count_l3860_386034

/-- The number of train services from location A to location B -/
def num_train_services : ℕ := 3

/-- The number of ferry services from location B to location C -/
def num_ferry_services : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_travel_options : ℕ := num_train_services * num_ferry_services

theorem travel_options_count : total_travel_options = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_options_count_l3860_386034


namespace NUMINAMATH_CALUDE_zhou_yu_age_equation_l3860_386036

/-- Represents the age of Zhou Yu at death -/
def age (x : ℕ) : ℕ := 10 * x + (x + 3)

/-- Theorem stating the relationship between Zhou Yu's age digits -/
theorem zhou_yu_age_equation (x : ℕ) : 
  age x = (x + 3)^2 :=
sorry

end NUMINAMATH_CALUDE_zhou_yu_age_equation_l3860_386036


namespace NUMINAMATH_CALUDE_sales_model_results_l3860_386072

/-- Represents the weekly sales and profit model for a children's clothing store --/
structure SalesModel where
  originalPrice : ℝ
  originalSales : ℝ
  priceReduction : ℝ
  salesIncrease : ℝ
  costPrice : ℝ

/-- Calculates the weekly sales volume based on the selling price --/
def weeklySales (model : SalesModel) (x : ℝ) : ℝ :=
  model.originalSales + model.salesIncrease * (model.originalPrice - x)

/-- Calculates the weekly profit based on the selling price --/
def weeklyProfit (model : SalesModel) (x : ℝ) : ℝ :=
  (x - model.costPrice) * (weeklySales model x)

/-- Theorem stating the main results of the problem --/
theorem sales_model_results (model : SalesModel)
  (h1 : model.originalPrice = 60)
  (h2 : model.originalSales = 300)
  (h3 : model.priceReduction = 1)
  (h4 : model.salesIncrease = 30)
  (h5 : model.costPrice = 40) :
  (∀ x, weeklySales model x = -30 * x + 2100) ∧
  (∃ x_max, x_max = 55 ∧ ∀ x, weeklyProfit model x ≤ weeklyProfit model x_max) ∧
  (weeklyProfit model 55 = 6750) ∧
  (∀ x, 52 ≤ x ∧ x ≤ 58 ↔ weeklyProfit model x ≥ 6480) := by
  sorry

end NUMINAMATH_CALUDE_sales_model_results_l3860_386072


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_l3860_386005

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sum_of_digits_range (start : ℕ) (count : ℕ) : ℕ :=
  List.range count |>.map (fun i => sum_of_digits (start + i)) |>.sum

theorem consecutive_numbers_digit_sum :
  ∃! start : ℕ, sum_of_digits_range start 10 = 145 ∧ start ≥ 100 ∧ start < 1000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_l3860_386005


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3860_386082

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℝ := 5

/-- The cost of popcorn in dollars -/
def popcorn_cost : ℝ := 0.8 * ticket_cost

/-- The cost of soda in dollars -/
def soda_cost : ℝ := 0.5 * popcorn_cost

/-- Theorem stating that the given conditions result in a ticket cost of $5 -/
theorem movie_ticket_cost : 
  4 * ticket_cost + 2 * popcorn_cost + 4 * soda_cost = 36 := by
  sorry


end NUMINAMATH_CALUDE_movie_ticket_cost_l3860_386082


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3860_386066

theorem sum_of_x_and_y (x y : ℝ) (hxy : x ≠ y)
  (det1 : Matrix.det ![![2, 5, 10], ![4, x, y], ![4, y, x]] = 0)
  (det2 : Matrix.det ![![x, y], ![y, x]] = 16) :
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3860_386066


namespace NUMINAMATH_CALUDE_problem_solution_l3860_386023

-- Define the propositions
def p : Prop := ∃ k : ℤ, 0 = 2 * k
def q : Prop := ∃ k : ℤ, 3 = 2 * k

-- Theorem to prove
theorem problem_solution : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3860_386023


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3860_386089

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 10) →
  ((a + 4) / (1 - r) = 15) →
  r = 1/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3860_386089


namespace NUMINAMATH_CALUDE_abs_sum_greater_than_abs_l3860_386044

theorem abs_sum_greater_than_abs (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + a * c = 0) 
  (h4 : a * b * c = 1) : 
  |a + b| > |c| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_greater_than_abs_l3860_386044


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l3860_386006

theorem walnut_trees_in_park (initial_trees new_trees : ℕ) : 
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l3860_386006


namespace NUMINAMATH_CALUDE_inner_square_probability_l3860_386088

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Checks if a square is on the perimeter or center lines -/
def is_on_perimeter_or_center (b : Board) (row col : ℕ) : Prop :=
  row = 1 ∨ row = b.size ∨ col = 1 ∨ col = b.size ∨
  row = b.size / 2 ∨ row = b.size / 2 + 1 ∨
  col = b.size / 2 ∨ col = b.size / 2 + 1

/-- Counts squares not on perimeter or center lines -/
def count_inner_squares (b : Board) : ℕ :=
  (b.size - 4) * (b.size - 4)

/-- The main theorem -/
theorem inner_square_probability (b : Board) (h : b.size = 10) :
  (count_inner_squares b : ℚ) / (b.size * b.size : ℚ) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_inner_square_probability_l3860_386088


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3860_386046

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^2 - a * abs x + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3860_386046


namespace NUMINAMATH_CALUDE_exists_triangle_with_large_inner_triangle_l3860_386037

-- Define the structure of a triangle
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute (t : Triangle) : Prop := sorry

-- Define the line segments
def median (t : Triangle) : Point → Point := sorry
def angle_bisector (t : Triangle) : Point → Point := sorry
def altitude (t : Triangle) : Point → Point := sorry

-- Define the intersection points
def intersection_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- The main theorem
theorem exists_triangle_with_large_inner_triangle :
  ∃ (t : Triangle),
    is_acute t ∧
    area (intersection_points t) > 0.499 * area t :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_with_large_inner_triangle_l3860_386037


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_z_l3860_386055

theorem inequality_holds_for_all_z (x y : ℝ) (hx : x > 0) :
  ∀ z : ℝ, y - z < Real.sqrt (z^2 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_z_l3860_386055


namespace NUMINAMATH_CALUDE_noahs_yearly_call_cost_l3860_386067

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (weeks_per_year : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) : ℚ :=
  (weeks_per_year * minutes_per_call : ℕ) * cost_per_minute

/-- Theorem: Noah's yearly call cost to Grammy is $78 -/
theorem noahs_yearly_call_cost :
  total_cost 52 30 (5/100) = 78 := by
  sorry

end NUMINAMATH_CALUDE_noahs_yearly_call_cost_l3860_386067


namespace NUMINAMATH_CALUDE_segment_length_on_ellipse_l3860_386097

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem segment_length_on_ellipse :
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  (∃ t : ℝ, A = F₁ + t • (B - F₁)) →  -- A, B, and F₁ are collinear
  distance F₂ A + distance F₂ B = 12 →
  distance A B = 8 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_on_ellipse_l3860_386097


namespace NUMINAMATH_CALUDE_basketball_probability_l3860_386059

theorem basketball_probability (free_throw high_school pro : ℝ) 
  (h1 : free_throw = 4/5)
  (h2 : high_school = 1/2)
  (h3 : pro = 1/3) :
  1 - (1 - free_throw) * (1 - high_school) * (1 - pro) = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l3860_386059


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3860_386087

/-- Given a group of children with various emotional states, prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3860_386087


namespace NUMINAMATH_CALUDE_casting_theorem_l3860_386021

def men : ℕ := 7
def women : ℕ := 5
def male_roles : ℕ := 3
def either_gender_roles : ℕ := 2
def total_roles : ℕ := male_roles + either_gender_roles

def casting_combinations : ℕ := (men.choose male_roles) * ((men + women - male_roles).choose either_gender_roles)

theorem casting_theorem : casting_combinations = 15120 := by sorry

end NUMINAMATH_CALUDE_casting_theorem_l3860_386021


namespace NUMINAMATH_CALUDE_sum_of_digits_after_addition_l3860_386040

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of carries in addition -/
def carries_in_addition (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : carries_in_addition A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_after_addition_l3860_386040


namespace NUMINAMATH_CALUDE_distance_between_centers_l3860_386092

/-- An isosceles triangle with its circumcircle and inscribed circle -/
structure IsoscelesTriangleWithCircles where
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- r is less than R (as the inscribed circle must fit inside the circumcircle) -/
  r_lt_R : r < R

/-- The distance between the centers of the circumcircle and inscribed circle
    of an isosceles triangle is √(R(R - 2r)) -/
theorem distance_between_centers (t : IsoscelesTriangleWithCircles) :
  ∃ d : ℝ, d = Real.sqrt (t.R * (t.R - 2 * t.r)) ∧ d > 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3860_386092


namespace NUMINAMATH_CALUDE_unique_prime_digit_l3860_386070

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (B : ℕ) : ℕ := 303160 + B

theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ is_prime (number B) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_digit_l3860_386070


namespace NUMINAMATH_CALUDE_minimum_students_l3860_386041

theorem minimum_students (b g : ℕ) : 
  (3 * b = 5 * g) →  -- Same number of boys and girls passed
  (b ≥ 5) →          -- At least 5 boys (for 3/5 to be meaningful)
  (g ≥ 6) →          -- At least 6 girls (for 5/6 to be meaningful)
  (∀ b' g', (3 * b' = 5 * g') → (b' ≥ 5) → (g' ≥ 6) → (b' + g' ≥ b + g)) →
  b + g = 43 :=
by sorry

#check minimum_students

end NUMINAMATH_CALUDE_minimum_students_l3860_386041


namespace NUMINAMATH_CALUDE_smallest_candy_count_l3860_386071

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m + 6) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l3860_386071


namespace NUMINAMATH_CALUDE_value_of_y_l3860_386024

theorem value_of_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1/y) (eq2 : y = 1 + 1/x) : y = (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3860_386024


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3860_386012

theorem min_value_of_expression (a b : ℝ) (h : 2*a + 3*b = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 2*x + 3*y = 4 → 4^x + 8^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3860_386012


namespace NUMINAMATH_CALUDE_expanded_dining_area_total_l3860_386053

/-- The total area of an expanded outdoor dining area consisting of a rectangular section
    with an area of 35 square feet and a semi-circular section with a radius of 4 feet
    is equal to 35 + 8π square feet. -/
theorem expanded_dining_area_total (rectangular_area : ℝ) (semi_circle_radius : ℝ) :
  rectangular_area = 35 ∧ semi_circle_radius = 4 →
  rectangular_area + (1/2 * π * semi_circle_radius^2) = 35 + 8*π := by
  sorry

end NUMINAMATH_CALUDE_expanded_dining_area_total_l3860_386053


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l3860_386002

-- Define Mr. Kishore's expenses and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500
def savings : ℕ := 2000

-- Define total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define total salary
def total_salary : ℕ := total_expenses + savings

-- Theorem to prove
theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by
  sorry


end NUMINAMATH_CALUDE_kishore_savings_percentage_l3860_386002


namespace NUMINAMATH_CALUDE_inequality_range_l3860_386064

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3860_386064


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3860_386029

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℤ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3860_386029


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3860_386051

theorem cube_root_simplification :
  ∃ (a b : ℕ+), (a.val : ℝ) * (b.val : ℝ)^(1/3 : ℝ) = (2^11 * 3^8 : ℝ)^(1/3 : ℝ) ∧ 
  a.val = 72 ∧ b.val = 36 := by
sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3860_386051


namespace NUMINAMATH_CALUDE_square_root_approximation_l3860_386074

theorem square_root_approximation : ∃ ε > 0, ε < 0.0001 ∧ 
  |Real.sqrt ((16^10 + 32^10) / (16^6 + 32^11)) - 0.1768| < ε :=
by sorry

end NUMINAMATH_CALUDE_square_root_approximation_l3860_386074


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3860_386052

-- Define the quadratic function f
def f (x : ℝ) := 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (f 1 = 1) ∧
  (∀ x, f (x + 1) - f x = 4 * x - 2) ∧
  (∀ a, (∃ x y, 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)
    ↔ (0 < a ∧ a < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3860_386052


namespace NUMINAMATH_CALUDE_chess_tournament_rounds_l3860_386091

/-- The number of rounds needed for a chess tournament -/
theorem chess_tournament_rounds (n : ℕ) (games_per_round : ℕ) 
  (h1 : n = 20) 
  (h2 : games_per_round = 10) : 
  (n * (n - 1)) / games_per_round = 38 := by
  sorry

#check chess_tournament_rounds

end NUMINAMATH_CALUDE_chess_tournament_rounds_l3860_386091


namespace NUMINAMATH_CALUDE_second_place_wins_l3860_386063

/-- Represents a hockey team's performance --/
structure TeamPerformance where
  wins : ℕ
  ties : ℕ

/-- Calculates points for a team based on wins and ties --/
def calculatePoints (team : TeamPerformance) : ℕ := 2 * team.wins + team.ties

/-- Represents the hockey league --/
structure HockeyLeague where
  firstPlace : TeamPerformance
  secondPlace : TeamPerformance
  elsasTeam : TeamPerformance

theorem second_place_wins (league : HockeyLeague) : 
  league.firstPlace = ⟨12, 4⟩ →
  league.elsasTeam = ⟨8, 10⟩ →
  league.secondPlace.ties = 1 →
  (calculatePoints league.firstPlace + calculatePoints league.secondPlace + calculatePoints league.elsasTeam) / 3 = 27 →
  league.secondPlace.wins = 13 := by
  sorry

#eval calculatePoints ⟨13, 1⟩  -- Expected output: 27

end NUMINAMATH_CALUDE_second_place_wins_l3860_386063


namespace NUMINAMATH_CALUDE_root_between_roots_l3860_386054

theorem root_between_roots (a b c r s : ℝ) 
  (hr : a * r^2 + b * r + c = 0)
  (hs : -a * s^2 + b * s + c = 0) :
  ∃ t : ℝ, (t > r ∧ t < s ∨ t > s ∧ t < r) ∧ a/2 * t^2 + b * t + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_between_roots_l3860_386054


namespace NUMINAMATH_CALUDE_remainder_set_different_l3860_386075

theorem remainder_set_different (a b c : ℤ) 
  (ha : 0 < a ∧ a < c - 1) 
  (hb : 1 < b ∧ b < c) : 
  let r : ℤ → ℤ := λ k => (k * b) % c
  (∀ k, 0 ≤ k ∧ k ≤ a → 0 ≤ r k ∧ r k < c) →
  {k : ℤ | 0 ≤ k ∧ k ≤ a}.image r ≠ {k : ℤ | 0 ≤ k ∧ k ≤ a} := by
  sorry

end NUMINAMATH_CALUDE_remainder_set_different_l3860_386075


namespace NUMINAMATH_CALUDE_articles_produced_l3860_386043

/-- Given that x men working x hours a day for 2x days produce 2x³ articles,
    prove that y men working 2y hours a day for y days produce 2y³ articles. -/
theorem articles_produced (x y : ℕ) (h : x * x * (2 * x) = 2 * x^3) :
  y * (2 * y) * y = 2 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_articles_produced_l3860_386043


namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l3860_386084

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l3860_386084


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3860_386015

theorem inequality_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3860_386015


namespace NUMINAMATH_CALUDE_largest_possible_z_value_l3860_386030

theorem largest_possible_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = 2 * Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 := by sorry

end NUMINAMATH_CALUDE_largest_possible_z_value_l3860_386030


namespace NUMINAMATH_CALUDE_remainder_theorem_l3860_386093

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3860_386093


namespace NUMINAMATH_CALUDE_banana_permutations_l3860_386032

-- Define the word BANANA
def word : String := "BANANA"

-- Define the total number of letters
def total_letters : Nat := word.length

-- Define the number of As
def num_A : Nat := 3

-- Define the number of Ns
def num_N : Nat := 2

-- Theorem statement
theorem banana_permutations : 
  (Nat.factorial total_letters) / (Nat.factorial num_A * Nat.factorial num_N) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_permutations_l3860_386032


namespace NUMINAMATH_CALUDE_bake_sale_donation_l3860_386038

/-- The total donation to the homeless shelter given the bake sale earnings and additional personal donation -/
def total_donation_to_shelter (total_earnings : ℕ) (ingredients_cost : ℕ) (personal_donation : ℕ) : ℕ :=
  let remaining_total := total_earnings - ingredients_cost
  let shelter_donation := remaining_total / 2 + personal_donation
  shelter_donation

/-- Theorem stating that the total donation to the homeless shelter is $160 -/
theorem bake_sale_donation :
  total_donation_to_shelter 400 100 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_l3860_386038


namespace NUMINAMATH_CALUDE_certain_number_proof_l3860_386042

theorem certain_number_proof : 
  ∃ (x : ℝ), x / 1.45 = 17.5 → x = 25.375 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3860_386042


namespace NUMINAMATH_CALUDE_largest_room_length_l3860_386099

theorem largest_room_length 
  (largest_width : ℝ) 
  (smallest_width smallest_length : ℝ) 
  (area_difference : ℝ) 
  (h1 : largest_width = 45)
  (h2 : smallest_width = 15)
  (h3 : smallest_length = 8)
  (h4 : area_difference = 1230)
  (h5 : largest_width * largest_length - smallest_width * smallest_length = area_difference) :
  largest_length = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_room_length_l3860_386099


namespace NUMINAMATH_CALUDE_product_of_12_and_3460_l3860_386010

theorem product_of_12_and_3460 : ∃ x : ℕ, x * 12 = x * 240 → 12 * 3460 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_of_12_and_3460_l3860_386010


namespace NUMINAMATH_CALUDE_f_properties_l3860_386033

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x - Real.sqrt x + 1
  else if x = 0 then 0
  else x + Real.sqrt (-x) - 1

-- State the properties of f
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x - Real.sqrt x + 1) ∧  -- given definition for x > 0
  (Set.range f = {y | y ≥ 3/4 ∨ y ≤ -3/4 ∨ y = 0}) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3860_386033


namespace NUMINAMATH_CALUDE_local_max_derivative_range_l3860_386096

/-- Given a function f with derivative f'(x) = a(x + 1)(x - a) and a local maximum at x = a, 
    prove that a is in the open interval (-1, 0) -/
theorem local_max_derivative_range (f : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h₂ : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_derivative_range_l3860_386096


namespace NUMINAMATH_CALUDE_sales_tax_calculation_l3860_386095

-- Define the total spent
def total_spent : ℝ := 40

-- Define the tax rate
def tax_rate : ℝ := 0.06

-- Define the cost of tax-free items
def tax_free_cost : ℝ := 34.7

-- Theorem to prove
theorem sales_tax_calculation :
  let taxable_cost := total_spent - tax_free_cost
  let sales_tax := taxable_cost * tax_rate / (1 + tax_rate)
  sales_tax = 0.3 := by sorry

end NUMINAMATH_CALUDE_sales_tax_calculation_l3860_386095


namespace NUMINAMATH_CALUDE_sector_central_angle_l3860_386050

/-- Given a sector with circumference 6 and area 2, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2*r + l = 6) (h2 : (1/2)*l*r = 2) :
  l/r = 1 ∨ l/r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3860_386050


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l3860_386057

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l3860_386057


namespace NUMINAMATH_CALUDE_kittens_remaining_l3860_386056

theorem kittens_remaining (initial_kittens given_away : ℕ) : 
  initial_kittens = 8 → given_away = 2 → initial_kittens - given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_kittens_remaining_l3860_386056


namespace NUMINAMATH_CALUDE_square_is_quadratic_and_power_l3860_386003

/-- A function f: ℝ → ℝ is a power function if there exists a real number a such that f(x) = x^a for all x in the domain of f. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x, f x = x ^ a

/-- A function f: ℝ → ℝ is a quadratic function if there exist real numbers a, b, and c with a ≠ 0 such that f(x) = ax^2 + bx + c for all x in ℝ. -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is both a quadratic function and a power function. -/
theorem square_is_quadratic_and_power :
  let f : ℝ → ℝ := fun x ↦ x^2
  IsQuadraticFunction f ∧ IsPowerFunction f := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_and_power_l3860_386003


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l3860_386028

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount Sasha has in cents -/
def total_amount : ℕ := 480

/-- Theorem stating the maximum number of quarters Sasha can have -/
theorem max_quarters_sasha : 
  ∀ (quarters nickels dimes : ℕ),
  quarters = nickels →
  dimes = 4 * nickels →
  quarters * coin_value "quarter" + 
  nickels * coin_value "nickel" + 
  dimes * coin_value "dime" ≤ total_amount →
  quarters ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l3860_386028


namespace NUMINAMATH_CALUDE_tabithas_age_l3860_386062

/-- Tabitha's hair color problem -/
theorem tabithas_age :
  ∀ (current_year : ℕ) (start_year : ℕ) (start_colors : ℕ) (future_year : ℕ) (future_colors : ℕ),
  start_year = 15 →
  start_colors = 2 →
  future_year = current_year + 3 →
  future_colors = 8 →
  future_colors = start_colors + (future_year - start_year) →
  current_year = 18 :=
by sorry

end NUMINAMATH_CALUDE_tabithas_age_l3860_386062


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l3860_386017

theorem gcd_of_powers_of_101 : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l3860_386017


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l3860_386035

theorem rectangle_area_sum : 
  let width : ℕ := 3
  let lengths : List ℕ := [2^2, 4^2, 6^2, 8^2, 10^2]
  let areas : List ℕ := lengths.map (λ l => width * l)
  areas.sum = 660 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l3860_386035


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3860_386085

theorem fraction_sum_simplification (x : ℝ) (h : x + 1 ≠ 0) :
  x / ((x + 1)^2) + 1 / ((x + 1)^2) = 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3860_386085


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3860_386027

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a - 21 = (a - 3) * (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3860_386027


namespace NUMINAMATH_CALUDE_rupert_ronald_jumps_l3860_386083

theorem rupert_ronald_jumps 
  (ronald_jumps : ℕ) 
  (total_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : ronald_jumps < total_jumps - ronald_jumps) :
  total_jumps - ronald_jumps - ronald_jumps = 86 := by
  sorry

end NUMINAMATH_CALUDE_rupert_ronald_jumps_l3860_386083


namespace NUMINAMATH_CALUDE_rubble_short_by_8_75_l3860_386073

def initial_amount : ℚ := 45
def notebook_cost : ℚ := 4
def pen_cost : ℚ := 1.5
def eraser_cost : ℚ := 2.25
def pencil_case_cost : ℚ := 7.5
def notebook_count : ℕ := 5
def pen_count : ℕ := 8
def eraser_count : ℕ := 3
def pencil_case_count : ℕ := 2

def total_cost : ℚ :=
  notebook_cost * notebook_count +
  pen_cost * pen_count +
  eraser_cost * eraser_count +
  pencil_case_cost * pencil_case_count

theorem rubble_short_by_8_75 :
  initial_amount - total_cost = -8.75 := by
  sorry

end NUMINAMATH_CALUDE_rubble_short_by_8_75_l3860_386073


namespace NUMINAMATH_CALUDE_euler_family_mean_age_is_11_l3860_386078

def euler_family_mean_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem euler_family_mean_age_is_11 :
  let ages := [8, 8, 8, 13, 13, 16]
  euler_family_mean_age ages = 11 := by
sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_is_11_l3860_386078


namespace NUMINAMATH_CALUDE_det_matrix1_det_matrix2_l3860_386081

-- Define the determinant function for 2x2 matrices
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem for the first matrix
theorem det_matrix1 : det2 2 5 (-3) (-4) = 7 := by sorry

-- Theorem for the second matrix
theorem det_matrix2 (a b : ℝ) : det2 (a^2) (a*b) (a*b) (b^2) = 0 := by sorry

end NUMINAMATH_CALUDE_det_matrix1_det_matrix2_l3860_386081


namespace NUMINAMATH_CALUDE_tensor_A_B_l3860_386001

-- Define the ⊗ operation
def tensor (M N : Set ℝ) : Set ℝ := (M ∪ N) \ (M ∩ N)

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem tensor_A_B : tensor A B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_tensor_A_B_l3860_386001


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3860_386031

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 6*a + 5 = 0 →
  b^2 - 6*b + 5 = 0 →
  a ≠ b →
  (a + a + b = 11 ∨ b + b + a = 11) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3860_386031


namespace NUMINAMATH_CALUDE_total_mowings_is_30_l3860_386013

/-- Represents the number of times Ned mowed a lawn in each season -/
structure SeasonalMowing where
  spring : Nat
  summer : Nat
  fall : Nat

/-- Calculates the total number of mowings for a lawn across all seasons -/
def totalMowings (s : SeasonalMowing) : Nat :=
  s.spring + s.summer + s.fall

/-- The number of times Ned mowed his front lawn in each season -/
def frontLawnMowings : SeasonalMowing :=
  { spring := 6, summer := 5, fall := 4 }

/-- The number of times Ned mowed his backyard lawn in each season -/
def backyardLawnMowings : SeasonalMowing :=
  { spring := 5, summer := 7, fall := 3 }

/-- Theorem: The total number of times Ned mowed his lawns is 30 -/
theorem total_mowings_is_30 :
  totalMowings frontLawnMowings + totalMowings backyardLawnMowings = 30 := by
  sorry


end NUMINAMATH_CALUDE_total_mowings_is_30_l3860_386013


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_power_of_two_l3860_386068

theorem exists_number_divisible_by_power_of_two (n : ℕ) :
  ∃ k : ℕ, (∀ d : ℕ, d < n → (k / 10^d % 10 = 1 ∨ k / 10^d % 10 = 2)) ∧ k % 2^n = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_power_of_two_l3860_386068


namespace NUMINAMATH_CALUDE_dog_treat_cost_l3860_386086

-- Define the given conditions
def treats_per_day : ℕ := 2
def cost_per_treat : ℚ := 1/10
def days_in_month : ℕ := 30

-- Define the theorem to prove
theorem dog_treat_cost :
  (treats_per_day * days_in_month : ℚ) * cost_per_treat = 6 := by sorry

end NUMINAMATH_CALUDE_dog_treat_cost_l3860_386086


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l3860_386047

theorem combined_teaching_experience (james_experience partner_experience : ℕ) 
  (h1 : james_experience = 40)
  (h2 : partner_experience = james_experience - 10) :
  james_experience + partner_experience = 70 := by
sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l3860_386047


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3860_386098

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3860_386098


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3860_386011

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3860_386011
