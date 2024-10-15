import Mathlib

namespace NUMINAMATH_CALUDE_total_cost_of_eggs_l1998_199805

def dozen : ℕ := 12
def egg_cost : ℚ := 0.50
def num_dozens : ℕ := 3

theorem total_cost_of_eggs :
  (↑num_dozens * ↑dozen) * egg_cost = 18 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_eggs_l1998_199805


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1998_199858

/-- A linear function y = (k+1)x + 3 where k > -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + 3

/-- The constraint on k -/
def k_constraint (k : ℝ) : Prop := k > -1

theorem point_not_on_graph (k : ℝ) (h : k_constraint k) :
  ¬ (linear_function k 5 = -1) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1998_199858


namespace NUMINAMATH_CALUDE_continuous_function_property_P_l1998_199848

open Function Set Real

theorem continuous_function_property_P 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_dom : ∀ x, x ∈ (Set.Ioc 0 1) → f x ≠ 0) 
  (hf_eq : f 0 = f 1) :
  ∀ k : ℕ, k ≥ 2 → ∃ x₀ ∈ Set.Icc 0 (1 - 1/k), f x₀ = f (x₀ + 1/k) :=
sorry

end NUMINAMATH_CALUDE_continuous_function_property_P_l1998_199848


namespace NUMINAMATH_CALUDE_final_quantities_correct_l1998_199831

/-- Represents the inventory and transactions of a stationery shop -/
structure StationeryShop where
  x : ℝ  -- initial number of pencils
  y : ℝ  -- initial number of pens
  z : ℝ  -- initial number of rulers

/-- Calculates the final quantities after transactions -/
def finalQuantities (shop : StationeryShop) : ℝ × ℝ × ℝ :=
  let remainingPencils := shop.x * 0.75
  let remainingPens := shop.y * 0.60
  let remainingRulers := shop.z * 0.80
  let finalPencils := remainingPencils + remainingPencils * 2.50
  let finalPens := remainingPens + 100
  let finalRulers := remainingRulers + remainingRulers * 5
  (finalPencils, finalPens, finalRulers)

/-- Theorem stating the correctness of the final quantities calculation -/
theorem final_quantities_correct (shop : StationeryShop) :
  finalQuantities shop = (2.625 * shop.x, 0.60 * shop.y + 100, 4.80 * shop.z) := by
  sorry

end NUMINAMATH_CALUDE_final_quantities_correct_l1998_199831


namespace NUMINAMATH_CALUDE_boat_accident_proof_l1998_199835

/-- The number of sheep that drowned in a boat accident -/
def drowned_sheep : ℕ := 3

theorem boat_accident_proof :
  let initial_sheep : ℕ := 20
  let initial_cows : ℕ := 10
  let initial_dogs : ℕ := 14
  let drowned_cows : ℕ := 2 * drowned_sheep
  let survived_dogs : ℕ := initial_dogs
  let total_survived : ℕ := 35
  total_survived = (initial_sheep - drowned_sheep) + (initial_cows - drowned_cows) + survived_dogs :=
by sorry

end NUMINAMATH_CALUDE_boat_accident_proof_l1998_199835


namespace NUMINAMATH_CALUDE_kevins_tshirts_l1998_199806

/-- Calculates the number of T-shirts Kevin can buy given the following conditions:
  * T-shirt price is $8
  * Sweater price is $18
  * Jacket original price is $80
  * Jacket discount is 10%
  * Sales tax is 5%
  * Kevin buys 4 sweaters and 5 jackets
  * Total payment including tax is $504
-/
theorem kevins_tshirts :
  let tshirt_price : ℚ := 8
  let sweater_price : ℚ := 18
  let jacket_original_price : ℚ := 80
  let jacket_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05
  let num_sweaters : ℕ := 4
  let num_jackets : ℕ := 5
  let total_payment : ℚ := 504

  let jacket_discounted_price := jacket_original_price * (1 - jacket_discount)
  let sweaters_cost := sweater_price * num_sweaters
  let jackets_cost := jacket_discounted_price * num_jackets
  let subtotal := sweaters_cost + jackets_cost
  let tax_amount := subtotal * sales_tax
  let total_without_tshirts := subtotal + tax_amount
  let amount_for_tshirts := total_payment - total_without_tshirts
  let num_tshirts := ⌊amount_for_tshirts / tshirt_price⌋

  num_tshirts = 6 := by sorry

end NUMINAMATH_CALUDE_kevins_tshirts_l1998_199806


namespace NUMINAMATH_CALUDE_bulk_warehouse_case_price_l1998_199811

/-- The price of a case at the bulk warehouse -/
def bulk_case_price (cans_per_case : ℕ) (grocery_cans : ℕ) (grocery_price : ℚ) (price_difference : ℚ) : ℚ :=
  let grocery_price_per_can : ℚ := grocery_price / grocery_cans
  let bulk_price_per_can : ℚ := grocery_price_per_can - price_difference
  cans_per_case * bulk_price_per_can

/-- Theorem stating that the price of a case at the bulk warehouse is $12.00 -/
theorem bulk_warehouse_case_price :
  bulk_case_price 48 12 6 (25/100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bulk_warehouse_case_price_l1998_199811


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l1998_199813

/-- A regular tetrahedron with painted stripes on each face -/
structure StripedTetrahedron :=
  (faces : Fin 4 → Fin 2)

/-- The probability of a specific stripe configuration -/
def stripe_probability : ℚ := 1 / 16

/-- A continuous stripe encircles the tetrahedron -/
def has_continuous_stripe (t : StripedTetrahedron) : Prop := sorry

/-- The number of stripe configurations that result in a continuous stripe -/
def continuous_stripe_count : ℕ := 2

theorem continuous_stripe_probability :
  (continuous_stripe_count : ℚ) * stripe_probability = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l1998_199813


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l1998_199807

/-- Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
    prove that a, b, and c form a geometric progression. -/
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l1998_199807


namespace NUMINAMATH_CALUDE_max_value_function_l1998_199860

theorem max_value_function (x y : ℝ) :
  (2*x + 3*y + 4) / Real.sqrt (x^2 + 2*y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l1998_199860


namespace NUMINAMATH_CALUDE_sum_of_even_integers_102_to_200_l1998_199888

theorem sum_of_even_integers_102_to_200 :
  let first_term : ℕ := 102
  let last_term : ℕ := 200
  let num_terms : ℕ := 50
  (num_terms : ℚ) / 2 * (first_term + last_term) = 7550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_102_to_200_l1998_199888


namespace NUMINAMATH_CALUDE_notebook_word_count_l1998_199853

theorem notebook_word_count (total_pages : Nat) (max_words_per_page : Nat) 
  (h1 : total_pages = 150)
  (h2 : max_words_per_page = 90)
  (h3 : ∃ (words_per_page : Nat), words_per_page ≤ max_words_per_page ∧ 
        (total_pages * words_per_page) % 221 = 210) :
  ∃ (words_per_page : Nat), words_per_page = 90 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % 221 = 210 := by
  sorry

end NUMINAMATH_CALUDE_notebook_word_count_l1998_199853


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1998_199814

/-- A square with perimeter 36 cm has an area of 81 cm² -/
theorem square_area_from_perimeter : 
  ∀ (s : ℝ), s > 0 → 4 * s = 36 → s^2 = 81 :=
by
  sorry


end NUMINAMATH_CALUDE_square_area_from_perimeter_l1998_199814


namespace NUMINAMATH_CALUDE_value_of_a_l1998_199873

theorem value_of_a (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6) :
  a = (384 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1998_199873


namespace NUMINAMATH_CALUDE_rachels_winter_clothing_l1998_199868

theorem rachels_winter_clothing (boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) 
  (h1 : boxes = 7)
  (h2 : scarves_per_box = 3)
  (h3 : mittens_per_box = 4) :
  boxes * (scarves_per_box + mittens_per_box) = 49 :=
by sorry

end NUMINAMATH_CALUDE_rachels_winter_clothing_l1998_199868


namespace NUMINAMATH_CALUDE_sons_age_l1998_199856

theorem sons_age (father son : ℕ) : 
  (father + 6 + (son + 6) = 68) →  -- After 6 years, sum of ages is 68
  (father = 6 * son) →             -- Father's age is 6 times son's age
  son = 8 :=                       -- Son's age is 8
by sorry

end NUMINAMATH_CALUDE_sons_age_l1998_199856


namespace NUMINAMATH_CALUDE_sum_of_squares_l1998_199842

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 8)
  (eq2 : b^2 + 4*c + 1 = -6)
  (eq3 : c^2 + 6*a = -15) :
  a^2 + b^2 + c^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1998_199842


namespace NUMINAMATH_CALUDE_profit_loss_percentage_l1998_199821

/-- 
Given an article with a cost price and two selling prices:
1. An original selling price that yields a 27.5% profit
2. A new selling price that is 2/3 of the original price

This theorem proves that the loss percentage at the new selling price is 15%.
-/
theorem profit_loss_percentage (cost_price : ℝ) (original_price : ℝ) (new_price : ℝ) : 
  original_price = cost_price * (1 + 0.275) →
  new_price = (2/3) * original_price →
  (cost_price - new_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_loss_percentage_l1998_199821


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l1998_199884

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![0, 1]
def vector_c (k : ℝ) : Fin 2 → ℝ := ![-2, k]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i, v i = t * w i

theorem parallel_vectors_k (k : ℝ) :
  parallel (λ i => vector_a i + 2 * vector_b i) (vector_c k) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l1998_199884


namespace NUMINAMATH_CALUDE_sequence_distinct_terms_l1998_199865

theorem sequence_distinct_terms (n m : ℕ) (hn : n ≥ 1) (hm : m ≥ 1) (hnm : n ≠ m) :
  n / (n + 1 : ℚ) ≠ m / (m + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sequence_distinct_terms_l1998_199865


namespace NUMINAMATH_CALUDE_prob_at_least_one_event_l1998_199823

/-- The probability that at least one of two independent events occurs -/
theorem prob_at_least_one_event (A B : ℝ) (hA : 0 ≤ A ∧ A ≤ 1) (hB : 0 ≤ B ∧ B ≤ 1) 
  (hAval : A = 0.9) (hBval : B = 0.8) :
  1 - (1 - A) * (1 - B) = 0.98 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_event_l1998_199823


namespace NUMINAMATH_CALUDE_subset_X_l1998_199824

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_subset_X_l1998_199824


namespace NUMINAMATH_CALUDE_julian_celine_ratio_is_one_l1998_199896

/-- The number of erasers collected by Celine -/
def celine_erasers : ℕ := 10

/-- The number of erasers collected by Julian -/
def julian_erasers : ℕ := celine_erasers

/-- The total number of erasers collected -/
def total_erasers : ℕ := 35

/-- The ratio of erasers collected by Julian to Celine -/
def julian_to_celine_ratio : ℚ := julian_erasers / celine_erasers

theorem julian_celine_ratio_is_one : julian_to_celine_ratio = 1 := by
  sorry

end NUMINAMATH_CALUDE_julian_celine_ratio_is_one_l1998_199896


namespace NUMINAMATH_CALUDE_arrangements_count_l1998_199879

/-- The number of ways to arrange 5 distinct objects in a row, 
    where two specific objects are not allowed to be adjacent -/
def arrangements_with_restriction : ℕ := 72

/-- Theorem stating that the number of arrangements with the given restriction is 72 -/
theorem arrangements_count : arrangements_with_restriction = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1998_199879


namespace NUMINAMATH_CALUDE_coordinate_sum_of_A_l1998_199872

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the coordinate plane -/
structure Line where
  m : ℝ
  b : ℝ

/-- The theorem statement -/
theorem coordinate_sum_of_A (A B C : Point) (l₁ l₂ l₃ : Line) (a b : ℝ) :
  B.y = 0 →  -- B is on Ox axis
  C.x = 0 →  -- C is on Oy axis
  (l₁.m = a ∧ l₁.b = 4) ∨ (l₁.m = 2 ∧ l₁.b = b) ∨ (l₁.m = a/2 ∧ l₁.b = 8) →  -- l₁ is one of the given lines
  (l₂.m = a ∧ l₂.b = 4) ∨ (l₂.m = 2 ∧ l₂.b = b) ∨ (l₂.m = a/2 ∧ l₂.b = 8) →  -- l₂ is one of the given lines
  (l₃.m = a ∧ l₃.b = 4) ∨ (l₃.m = 2 ∧ l₃.b = b) ∨ (l₃.m = a/2 ∧ l₃.b = 8) →  -- l₃ is one of the given lines
  l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ →  -- All lines are different
  (A.y = l₁.m * A.x + l₁.b) ∧ (B.y = l₁.m * B.x + l₁.b) →  -- A and B are on l₁
  (B.y = l₂.m * B.x + l₂.b) ∧ (C.y = l₂.m * C.x + l₂.b) →  -- B and C are on l₂
  (A.y = l₃.m * A.x + l₃.b) ∧ (C.y = l₃.m * C.x + l₃.b) →  -- A and C are on l₃
  A.x + A.y = 13 ∨ A.x + A.y = 20 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_A_l1998_199872


namespace NUMINAMATH_CALUDE_inequality_may_not_hold_l1998_199861

theorem inequality_may_not_hold (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, a / c ≤ b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_may_not_hold_l1998_199861


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l1998_199800

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem sheridan_fish_problem :
  fish_given 47.0 25 = 22 :=
by sorry

end NUMINAMATH_CALUDE_sheridan_fish_problem_l1998_199800


namespace NUMINAMATH_CALUDE_probability_three_non_defective_pencils_l1998_199841

theorem probability_three_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_non_defective_pencils_l1998_199841


namespace NUMINAMATH_CALUDE_rebecca_earrings_l1998_199854

theorem rebecca_earrings (magnets : ℕ) : 
  magnets > 0 → 
  (4 * (3 * (magnets / 2))) = 24 → 
  magnets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_earrings_l1998_199854


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l1998_199881

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 3

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ (a : ℝ) : Prop := 4 * a = 12 * Real.sqrt 2

-- Define points P and Q on the ellipse
def point_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint of PQ
def midpoint_PQ (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = 1

-- Define the theorem
theorem ellipse_and_line_equation 
  (a b c : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : focal_distance c) 
  (h₄ : perimeter_ABF₂ a) 
  (h₅ : point_on_ellipse x₁ y₁ a b) 
  (h₆ : point_on_ellipse x₂ y₂ a b) 
  (h₇ : x₁ ≠ x₂ ∨ y₁ ≠ y₂) 
  (h₈ : midpoint_PQ x₁ y₁ x₂ y₂) : 
  (ellipse_C 3 (Real.sqrt 2) = ellipse_C 3 3) ∧ 
  (∀ (x y : ℝ), y = -(x - 2) + 1 ↔ x + y = 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l1998_199881


namespace NUMINAMATH_CALUDE_bug_position_after_2012_jumps_l1998_199891

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is even -/
def Point.isEven : Point → Bool
  | .two => true
  | .four => true
  | _ => false

/-- Calculates the next point after a jump -/
def nextPoint (p : Point) : Point :=
  match p with
  | .one => .three
  | .two => .five
  | .three => .five
  | .four => .two
  | .five => .two

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2012_jumps :
  jumpNTimes Point.five 2012 = Point.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2012_jumps_l1998_199891


namespace NUMINAMATH_CALUDE_tomatoes_on_tuesday_eq_2500_l1998_199830

/-- Calculates the amount of tomatoes ready for sale on Tuesday given the initial shipment,
    sales, rotting, and new shipment. -/
def tomatoesOnTuesday (initialShipment sales rotted : ℕ) : ℕ :=
  let remainingAfterSales := initialShipment - sales
  let remainingAfterRotting := remainingAfterSales - rotted
  let newShipment := 2 * initialShipment
  remainingAfterRotting + newShipment

/-- Theorem stating that given the specific conditions, the amount of tomatoes
    ready for sale on Tuesday is 2500 kg. -/
theorem tomatoes_on_tuesday_eq_2500 :
  tomatoesOnTuesday 1000 300 200 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_on_tuesday_eq_2500_l1998_199830


namespace NUMINAMATH_CALUDE_xiaoming_mother_retirement_year_l1998_199864

/-- Calculates the retirement year based on the given retirement plan --/
def calculate_retirement_year (birth_year : ℕ) : ℕ :=
  let original_retirement_year := birth_year + 55
  if original_retirement_year ≥ 2018 ∧ original_retirement_year < 2021
  then original_retirement_year + 1
  else original_retirement_year

/-- Theorem stating that Xiaoming's mother's retirement year is 2020 --/
theorem xiaoming_mother_retirement_year :
  calculate_retirement_year 1964 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_mother_retirement_year_l1998_199864


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l1998_199850

/-- Given a hyperbola with the following properties:
  * Asymptotes: y = 3x + 2 and y = -3x + 8
  * Passes through the point (2, 10)
  * Standard form: (y-k)^2/a^2 - (x-h)^2/b^2 = 1
  * a, b > 0
  Prove that a + h = 6√2 + 1 -/
theorem hyperbola_a_plus_h (a b h k : ℝ) : 
  (∀ x y : ℝ, (y = 3*x + 2 ∨ y = -3*x + 8) → 
    ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) →
  ((10 - k)^2 / a^2 - (2 - h)^2 / b^2 = 1) →
  (a > 0 ∧ b > 0) →
  a + h = 6 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l1998_199850


namespace NUMINAMATH_CALUDE_gcd_problem_l1998_199887

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1193 * k ∧ Odd k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1998_199887


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1998_199803

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
  (x + c)^2 / 4 + y^2 / 4 = b^2) → 
  c^2 / a^2 = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1998_199803


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1998_199875

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  let S := (a^2 - a*b + b^2) * (b^2 - b*c + c^2) * (c^2 - c*a + a^2)
  ∃ (max_value : ℝ), max_value = 12 ∧ S ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1998_199875


namespace NUMINAMATH_CALUDE_manager_average_salary_l1998_199812

/-- Proves that the average salary of managers is $90,000 given the conditions of the company. -/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℚ) 
  (company_avg_salary : ℚ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary)) / 
   (num_managers * (num_managers + num_associates)) = 90000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l1998_199812


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l1998_199897

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def line_through_point (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem ellipse_constant_product :
  ∀ k : ℝ,
  ∃ x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 ∧ ellipse x2 y2 ∧
  line_through_point k 1 x1 y1 ∧
  line_through_point k 1 x2 y2 ∧
  x1 ≠ x2 →
  dot_product (17/8 - x1) (-y1) (17/8 - x2) (-y2) = 33/64 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l1998_199897


namespace NUMINAMATH_CALUDE_factorization_equality_l1998_199867

theorem factorization_equality (x : ℝ) : 
  x^2 * (x + 3) + 2 * (x + 3) - 5 * (x + 3) = (x + 3) * (x^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1998_199867


namespace NUMINAMATH_CALUDE_kennel_arrangement_l1998_199862

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  (Nat.factorial 3) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats)

/-- Theorem: The number of ways to arrange 3 chickens, 3 dogs, and 4 cats
    in a row of 10 cages, with animals of each type in adjacent cages,
    is 5184 -/
theorem kennel_arrangement : arrange_animals 3 3 4 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_kennel_arrangement_l1998_199862


namespace NUMINAMATH_CALUDE_unique_numbers_problem_l1998_199851

theorem unique_numbers_problem (a b : ℕ) : 
  a ≠ b → 
  a > 11 → 
  b > 11 → 
  (∃ (s : ℕ), s = a + b) → 
  (a % 2 = 0 ∨ b % 2 = 0) →
  (∀ (x y : ℕ), x ≠ y → x > 11 → y > 11 → x + y = a + b → 
    (x % 2 = 0 ∨ y % 2 = 0) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_unique_numbers_problem_l1998_199851


namespace NUMINAMATH_CALUDE_total_children_l1998_199843

/-- The number of happy children -/
def happy_children : ℕ := 30

/-- The number of sad children -/
def sad_children : ℕ := 10

/-- The number of children who are neither happy nor sad -/
def neutral_children : ℕ := 20

/-- The number of boys -/
def boys : ℕ := 17

/-- The number of girls -/
def girls : ℕ := 43

/-- The number of happy boys -/
def happy_boys : ℕ := 6

/-- The number of sad girls -/
def sad_girls : ℕ := 4

/-- The number of boys who are neither happy nor sad -/
def neutral_boys : ℕ := 5

/-- Theorem stating that the total number of children is 60 -/
theorem total_children : boys + girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l1998_199843


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l1998_199899

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l1998_199899


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1998_199871

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x| = x) → (x^2 + x ≥ 0)) ∧
  (∃ x : ℝ, (x^2 + x ≥ 0) ∧ (|x| ≠ x)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1998_199871


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l1998_199822

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the distance from its foci to the asymptote is 3,
    then the length of its imaginary axis is 6. -/
theorem hyperbola_imaginary_axis_length
  (b : ℝ)
  (h_b_pos : b > 0)
  (h_distance : b * Real.sqrt (4 + b^2) / Real.sqrt (4 + b^2) = 3) :
  2 * b = 6 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l1998_199822


namespace NUMINAMATH_CALUDE_grandmas_age_l1998_199878

theorem grandmas_age :
  ∀ x : ℕ, (x : ℝ) - (x : ℝ) / 7 = 84 → x = 98 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_age_l1998_199878


namespace NUMINAMATH_CALUDE_two_decimals_sum_and_difference_l1998_199840

theorem two_decimals_sum_and_difference (x y : ℝ) : 
  (0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10) → -- x and y are single-digit decimals
  (x + y = 10) →                     -- their sum is 10
  (|x - y| = 0.4) →                  -- their difference is 0.4
  ((x = 4.8 ∧ y = 5.2) ∨ (x = 5.2 ∧ y = 4.8)) := by
sorry

end NUMINAMATH_CALUDE_two_decimals_sum_and_difference_l1998_199840


namespace NUMINAMATH_CALUDE_ellipse_equation_and_intersection_range_l1998_199882

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line x - y + 2√2 = 0
def Line := {p : ℝ × ℝ | p.1 - p.2 + 2 * Real.sqrt 2 = 0}

theorem ellipse_equation_and_intersection_range :
  ∃ (a b c : ℝ),
    -- Conditions
    (0, -1) ∈ Ellipse a b ∧  -- One vertex at (0, -1)
    (c, 0) ∈ Ellipse a b ∧   -- Right focus on x-axis
    (∀ (x y : ℝ), (x, y) ∈ Line → ((x - c)^2 + y^2).sqrt = 3) ∧  -- Distance from right focus to line is 3
    -- Conclusions
    (Ellipse a b = Ellipse (Real.sqrt 3) 1) ∧  -- Equation of ellipse
    (∀ m : ℝ, (∃ (p q : ℝ × ℝ), p ≠ q ∧ p ∈ Ellipse (Real.sqrt 3) 1 ∧ q ∈ Ellipse (Real.sqrt 3) 1 ∧
                p.2 = p.1 + m ∧ q.2 = q.1 + m) ↔ -2 < m ∧ m < 2)  -- Intersection range
    := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_intersection_range_l1998_199882


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1998_199829

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) 
  (h_monic : MonicQuarticPolynomial p)
  (h_neg_two : p (-2) = -4)
  (h_one : p 1 = -1)
  (h_three : p 3 = -9)
  (h_five : p 5 = -25) :
  p 0 = -30 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1998_199829


namespace NUMINAMATH_CALUDE_prob_at_least_one_odd_prob_outside_or_on_circle_l1998_199869

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the sample space for two die rolls
def TwoRolls : Type := Die × Die

-- Define the probability measure
def P : Set TwoRolls → ℚ := sorry

-- Define the event of at least one odd number
def AtLeastOneOdd : Set TwoRolls := sorry

-- Define the event of the point lying outside or on the circle
def OutsideOrOnCircle : Set TwoRolls := sorry

-- Theorem for the first probability
theorem prob_at_least_one_odd : P AtLeastOneOdd = 3/4 := sorry

-- Theorem for the second probability
theorem prob_outside_or_on_circle : P OutsideOrOnCircle = 7/9 := sorry

end NUMINAMATH_CALUDE_prob_at_least_one_odd_prob_outside_or_on_circle_l1998_199869


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1998_199810

/-- Given a fixed perimeter, the area of a rectangle is maximized when it is a square -/
theorem rectangle_max_area (P : ℝ) (h : P > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = P / 2 →
  x * y ≤ (P / 4) ^ 2 ∧ 
  (x * y = (P / 4) ^ 2 ↔ x = y) := by
sorry


end NUMINAMATH_CALUDE_rectangle_max_area_l1998_199810


namespace NUMINAMATH_CALUDE_sin_cos_sum_squared_l1998_199802

theorem sin_cos_sum_squared (x : Real) : 
  (Real.sin x + Real.cos x = Real.sqrt 2 / 2) → 
  (Real.sin x)^4 + (Real.cos x)^4 = 7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_squared_l1998_199802


namespace NUMINAMATH_CALUDE_beatrice_book_count_l1998_199866

/-- The cost of each of the first 5 books -/
def initial_book_cost : ℕ := 20

/-- The number of books at the initial price -/
def initial_book_count : ℕ := 5

/-- The discount applied to each book after the initial count -/
def discount : ℕ := 2

/-- The total amount Beatrice paid -/
def total_paid : ℕ := 370

/-- Function to calculate the total cost for a given number of books -/
def total_cost (num_books : ℕ) : ℕ :=
  if num_books ≤ initial_book_count then
    num_books * initial_book_cost
  else
    initial_book_count * initial_book_cost +
    (num_books - initial_book_count) * (initial_book_cost - discount)

/-- Theorem stating that Beatrice bought 20 books -/
theorem beatrice_book_count : ∃ (n : ℕ), n = 20 ∧ total_cost n = total_paid := by
  sorry

end NUMINAMATH_CALUDE_beatrice_book_count_l1998_199866


namespace NUMINAMATH_CALUDE_remainder_sum_mod_60_l1998_199898

theorem remainder_sum_mod_60 (c d : ℤ) 
  (h1 : c % 120 = 114)
  (h2 : d % 180 = 174) : 
  (c + d) % 60 = 48 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_60_l1998_199898


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_inequality_three_l1998_199852

-- 1. 3x + 1 ≥ -2 if and only if x ≥ -1
theorem inequality_one (x : ℝ) : 3 * x + 1 ≥ -2 ↔ x ≥ -1 := by sorry

-- 2. (y ≥ 1 and -2y ≥ -2) if and only if y = 1
theorem inequality_two (y : ℝ) : (y ≥ 1 ∧ -2 * y ≥ -2) ↔ y = 1 := by sorry

-- 3. y²(x² + 1) - 1 ≤ x² if and only if -1 ≤ y ≤ 1
theorem inequality_three (x y : ℝ) : y^2 * (x^2 + 1) - 1 ≤ x^2 ↔ -1 ≤ y ∧ y ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_inequality_three_l1998_199852


namespace NUMINAMATH_CALUDE_correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l1998_199804

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem for the correct representation (Option D)
theorem correct_line_representation (n : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∃ (l : Line), l.slope = k ∧ pointOnLine l ⟨n, 0⟩ ∧
    ∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x = k * y + n :=
sorry

-- Theorem for the incorrectness of Option A
theorem incorrect_representation_A (x₀ y₀ : ℝ) :
  ¬ (∀ (l : Line), ∃ (k : ℝ), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ y - y₀ = k * (x - x₀)) :=
sorry

-- Theorem for the incorrectness of Option B
theorem incorrect_representation_B (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  ¬ (∀ (l : Line), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁)) :=
sorry

-- Theorem for the incorrectness of Option C
theorem incorrect_representation_C :
  ¬ (∀ (l : Line) (a b : ℝ), (¬ pointOnLine l ⟨0, 0⟩) →
    (∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x / a + y / b = 1)) :=
sorry

end NUMINAMATH_CALUDE_correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l1998_199804


namespace NUMINAMATH_CALUDE_unique_prime_base_1021_l1998_199808

theorem unique_prime_base_1021 : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^3 + 2*n + 1) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_base_1021_l1998_199808


namespace NUMINAMATH_CALUDE_curly_bracket_calculation_l1998_199893

-- Define the ceiling function for rational numbers
def ceiling (a : ℚ) : ℤ := Int.ceil a

-- Define the curly bracket notation
def curly_bracket (a : ℚ) : ℤ := ceiling a

-- Theorem statement
theorem curly_bracket_calculation :
  (curly_bracket (-6 + 5/6) : ℚ) - 
  (curly_bracket 5 : ℚ) * (curly_bracket (-1 - 3/4) : ℚ) / (curly_bracket (59/10) : ℚ) = -5 := by
  sorry


end NUMINAMATH_CALUDE_curly_bracket_calculation_l1998_199893


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1998_199863

theorem quadratic_roots_relation (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ 
  (x₂^2 + p*x₂ + q = 0) ∧ 
  ((x₁ + 1)^2 + q*(x₁ + 1) + p = 0) ∧ 
  ((x₂ + 1)^2 + q*(x₂ + 1) + p = 0) →
  p = -1 ∧ q = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1998_199863


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1998_199827

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := a / b
  let perpendicular_slope := -1 / original_slope
  (5 : ℝ) * x - (4 : ℝ) * y = (20 : ℝ) → perpendicular_slope = -(4 : ℝ) / (5 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1998_199827


namespace NUMINAMATH_CALUDE_four_greater_than_sqrt_fifteen_l1998_199801

theorem four_greater_than_sqrt_fifteen : 4 > Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_four_greater_than_sqrt_fifteen_l1998_199801


namespace NUMINAMATH_CALUDE_binary_division_remainder_l1998_199809

theorem binary_division_remainder (n : ℕ) (h : n = 0b111001011110) : n % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_remainder_l1998_199809


namespace NUMINAMATH_CALUDE_fraction_equality_l1998_199874

theorem fraction_equality (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 0.4166666666666667) 
  (h3 : f * x = (25/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1998_199874


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1998_199817

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  a 1 = 1 →  -- a_1 = 1
  a 7 = a 6 + 2 * a 5 →  -- a_7 = a_6 + 2a_5
  (∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = q * a k) →  -- geometric sequence
  a m * a n = 16 →  -- a_m * a_n = 16
  m > 0 ∧ n > 0 →  -- m and n are positive
  1 / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1998_199817


namespace NUMINAMATH_CALUDE_pentagon_angle_Q_measure_l1998_199844

-- Define the sum of angles in a pentagon
def pentagon_angle_sum : ℝ := 540

-- Define the known angles
def angle1 : ℝ := 130
def angle2 : ℝ := 90
def angle3 : ℝ := 110
def angle4 : ℝ := 115

-- Define the relation between Q and R
def Q_R_relation (Q R : ℝ) : Prop := Q = 2 * R

-- Theorem statement
theorem pentagon_angle_Q_measure :
  ∀ Q R : ℝ,
  Q_R_relation Q R →
  angle1 + angle2 + angle3 + angle4 + Q + R = pentagon_angle_sum →
  Q = 63.33 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_angle_Q_measure_l1998_199844


namespace NUMINAMATH_CALUDE_function_properties_l1998_199870

open Function

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x, HasDerivAt f (f' x) x)
variable (h2 : ∀ x, HasDerivAt g (g' x) x)
variable (h3 : ∀ x, f x = g ((x + 1) / 2) + x)
variable (h4 : Even f)
variable (h5 : Odd (fun x ↦ g' (x + 1)))

-- State the theorem
theorem function_properties :
  f' 1 = 1 ∧ g' (3/2) = 2 ∧ g' 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1998_199870


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1998_199855

/-- Represents a high school with stratified sampling -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  total_sample : ℕ
  first_year_sample : ℕ
  third_year_sample : ℕ

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- The sampling ratio for the second year -/
def sampling_ratio (hs : HighSchool) : ℚ :=
  (hs.total_sample - hs.first_year_sample - hs.third_year_sample : ℚ) / hs.second_year

theorem stratified_sampling_theorem (hs : HighSchool) 
  (h1 : hs.second_year = 900)
  (h2 : hs.total_sample = 370)
  (h3 : hs.first_year_sample = 120)
  (h4 : hs.third_year_sample = 100)
  (h5 : sampling_ratio hs = 1 / 6) :
  total_students hs = 2220 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1998_199855


namespace NUMINAMATH_CALUDE_min_value_theorem_l1998_199845

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → (a + 1) * (b + 1) / (a * b) ≥ (x + 1) * (y + 1) / (x * y)) ∧
  (x + 1) * (y + 1) / (x * y) = 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1998_199845


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1998_199818

/-- Given a triangle ABC with angles α, β, γ and sides a, b,
    if a + b = tan(γ/2) * (a * tan(α) + b * tan(β)),
    then the triangle ABC is isosceles. -/
theorem triangle_isosceles (α β γ a b : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  0 < a ∧ 0 < b ∧
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  a = b ∨ α = β := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1998_199818


namespace NUMINAMATH_CALUDE_no_integer_solution_l1998_199857

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 23 * x^2 - 92 * y^2 = 3128 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1998_199857


namespace NUMINAMATH_CALUDE_bike_route_length_l1998_199849

/-- Represents a rectangular bike route in a park -/
structure BikeRoute where
  upper_horizontal : List Float
  left_vertical : List Float

/-- Calculates the total length of the bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.upper_horizontal.sum + route.left_vertical.sum)

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_length :
  let route : BikeRoute := {
    upper_horizontal := [4, 7, 2],
    left_vertical := [6, 7]
  }
  total_length route = 52 := by sorry

end NUMINAMATH_CALUDE_bike_route_length_l1998_199849


namespace NUMINAMATH_CALUDE_min_non_parallel_lines_l1998_199883

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Function to create a line passing through two points -/
def line_through_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.x * q.y - q.x * p.y }

/-- The main theorem -/
theorem min_non_parallel_lines (n : ℕ) (points : Fin n → Point) 
  (h_n : n ≥ 3)
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line),
    (∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) ∧
    (∀ lines' : Fin n' → Line, n' < n →
      ¬(∀ i j, i ≠ j → ¬parallel (lines' i) (lines' j))) :=
sorry

end NUMINAMATH_CALUDE_min_non_parallel_lines_l1998_199883


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1998_199839

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1998_199839


namespace NUMINAMATH_CALUDE_parallelogram_base_l1998_199837

/-- Given a parallelogram with area 462 square centimeters and height 21 cm, its base is 22 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 462 → height = 21 → area = base * height → base = 22 := by sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1998_199837


namespace NUMINAMATH_CALUDE_tennis_tournament_balls_l1998_199815

theorem tennis_tournament_balls (total_balls : ℕ) (balls_per_can : ℕ) : 
  total_balls = 225 →
  balls_per_can = 3 →
  (8 + 4 + 2 + 1 : ℕ) * (total_balls / balls_per_can / (8 + 4 + 2 + 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_balls_l1998_199815


namespace NUMINAMATH_CALUDE_square_difference_solutions_l1998_199825

theorem square_difference_solutions :
  (∀ x y : ℕ, x^2 - y^2 = 31 ↔ (x = 16 ∧ y = 15)) ∧
  (∀ x y : ℕ, x^2 - y^2 = 303 ↔ (x = 152 ∧ y = 151) ∨ (x = 52 ∧ y = 49)) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_solutions_l1998_199825


namespace NUMINAMATH_CALUDE_triangle_inequality_from_seven_numbers_l1998_199836

theorem triangle_inequality_from_seven_numbers
  (a : Fin 7 → ℝ)
  (h : ∀ i, 1 < a i ∧ a i < 13) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i + a j > a k ∧
    a j + a k > a i ∧
    a k + a i > a j :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_seven_numbers_l1998_199836


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l1998_199876

theorem multiply_and_simplify (x : ℝ) : (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^4 + 32*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l1998_199876


namespace NUMINAMATH_CALUDE_gilled_mushroom_count_l1998_199820

/-- Represents the types of mushrooms --/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom --/
structure Mushroom where
  type : MushroomType

/-- Represents a collection of mushrooms on a log --/
structure MushroomLog where
  mushrooms : Finset Mushroom
  total_count : Nat
  spotted_count : Nat
  gilled_count : Nat
  h_total : total_count = mushrooms.card
  h_partition : total_count = spotted_count + gilled_count
  h_types : ∀ m ∈ mushrooms, m.type = MushroomType.Spotted ∨ m.type = MushroomType.Gilled
  h_ratio : spotted_count = 9 * gilled_count

theorem gilled_mushroom_count (log : MushroomLog) (h : log.total_count = 30) :
  log.gilled_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_gilled_mushroom_count_l1998_199820


namespace NUMINAMATH_CALUDE_factors_of_12_correct_ratio_exists_in_factors_l1998_199847

def is_factor (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def factors_of_12 : Set ℕ := {1, 2, 3, 4, 6, 12}

theorem factors_of_12_correct :
  ∀ n : ℕ, n ∈ factors_of_12 ↔ is_factor 12 n := by sorry

theorem ratio_exists_in_factors :
  ∃ a b c d : ℕ, a ∈ factors_of_12 ∧ b ∈ factors_of_12 ∧ c ∈ factors_of_12 ∧ d ∈ factors_of_12 ∧
  a * d = b * c ∧ a ≠ 0 ∧ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_factors_of_12_correct_ratio_exists_in_factors_l1998_199847


namespace NUMINAMATH_CALUDE_root_product_equals_27_l1998_199846

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l1998_199846


namespace NUMINAMATH_CALUDE_reflected_light_equation_l1998_199880

/-- Given points A, B, and P in a plane, and a line l passing through P parallel to AB,
    prove that the equation of the reflected light line from B to A via l is 11x + 27y + 74 = 0 -/
theorem reflected_light_equation (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (8, -6) →
  B = (2, 2) →
  P = (2, -3) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y + 1 = 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y - P.2 = k * (x - P.1)) →
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ l ∧ 
    ((y₀ - B.2) / (x₀ - B.1) = -(x₀ - A.1) / (y₀ - A.2))) →
  ∃ (x y : ℝ), 11*x + 27*y + 74 = 0 ↔ 
    (y - A.2) / (x - A.1) = (A.2 - y₀) / (A.1 - x₀) :=
by sorry

end NUMINAMATH_CALUDE_reflected_light_equation_l1998_199880


namespace NUMINAMATH_CALUDE_smallest_addend_for_divisibility_problem_solution_l1998_199826

theorem smallest_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 913475821
  let d := 13
  ∃ (k : ℕ), k = 2 ∧ k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_addend_for_divisibility_problem_solution_l1998_199826


namespace NUMINAMATH_CALUDE_smallest_S_for_equal_probability_l1998_199885

-- Define the number of sides on a standard die
def standardDieSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 2000

-- Define the function to calculate the minimum number of dice needed to reach the target sum
def minDiceNeeded (target : ℕ) (sides : ℕ) : ℕ :=
  (target + sides - 1) / sides

-- Define the function to calculate S given n dice
def calculateS (n : ℕ) (target : ℕ) : ℕ :=
  7 * n - target

-- Theorem statement
theorem smallest_S_for_equal_probability :
  let n := minDiceNeeded targetSum standardDieSides
  calculateS n targetSum = 338 := by sorry

end NUMINAMATH_CALUDE_smallest_S_for_equal_probability_l1998_199885


namespace NUMINAMATH_CALUDE_f_properties_l1998_199832

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x + 1

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x ∈ Set.Icc (2 - Real.exp 1) 1) ∧
  (∀ a ≤ 0, ∃! x, f a x = 0) ∧
  (∀ a > 0, ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z, f a z = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1998_199832


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l1998_199819

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∨ t.angle1 = t.angle3 ∨ t.angle2 = t.angle3

-- Define the sum of angles in a triangle
def angleSum (t : Triangle) : ℝ :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem isosceles_triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 40)
  (h2 : t.angle2 = 70)
  (h3 : angleSum t = 180) :
  isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l1998_199819


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1998_199828

-- Define the function f'(x)
def f' (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1998_199828


namespace NUMINAMATH_CALUDE_vacation_cost_l1998_199859

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 30) → cost = 360 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l1998_199859


namespace NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l1998_199892

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_level : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank when both pipes are open -/
def time_to_empty_or_fill (tank : WaterTank) : ℚ :=
  tank.initial_level / (tank.empty_rate - tank.fill_rate)

/-- Theorem stating that the tank will be emptied in 3 minutes under given conditions -/
theorem tank_emptied_in_three_minutes :
  let tank : WaterTank := {
    initial_level := 1/5,
    fill_rate := 1/10,
    empty_rate := 1/6
  }
  time_to_empty_or_fill tank = 3 := by
  sorry

#eval time_to_empty_or_fill {
  initial_level := 1/5,
  fill_rate := 1/10,
  empty_rate := 1/6
}

end NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l1998_199892


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1998_199886

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  ∃ (pq pr qr : ℝ), pq = 7 ∧ pr = 8 ∧ qr = 5 ∧ 
  pq + pr > qr ∧ pq + qr > pr ∧ pr + qr > pq

-- State the theorem
theorem triangle_trigonometric_identity (P Q R : ℝ) 
  (h : Triangle P Q R) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1998_199886


namespace NUMINAMATH_CALUDE_a_max_at_6_l1998_199834

-- Define the sequence a_n
def a (n : ℤ) : ℚ := (10 / 11) ^ n * (3 * n + 13)

-- Theorem stating that a_n is maximized when n = 6
theorem a_max_at_6 : ∀ (k : ℤ), a 6 ≥ a k := by sorry

end NUMINAMATH_CALUDE_a_max_at_6_l1998_199834


namespace NUMINAMATH_CALUDE_point_on_line_l1998_199890

/-- Given a line y = mx + b where m is the slope and b is the y-intercept,
    if m + b = 3, then the point (1, 3) lies on this line. -/
theorem point_on_line (m b : ℝ) (h : m + b = 3) :
  let f : ℝ → ℝ := fun x ↦ m * x + b
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1998_199890


namespace NUMINAMATH_CALUDE_kolya_speed_increase_l1998_199838

theorem kolya_speed_increase (N : ℕ) (x : ℕ) : 
  -- Total problems for each student
  N = (3 * x) / 2 →
  -- Kolya has solved 1/3 of what Seryozha has left
  x / 6 = (x / 2) / 3 →
  -- Seryozha has solved half of his problems
  x = N / 2 →
  -- The factor by which Kolya needs to increase his speed
  (((3 * x) / 2 - x / 6) / (x / 2)) / (x / 6 / x) = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_kolya_speed_increase_l1998_199838


namespace NUMINAMATH_CALUDE_company_valuation_l1998_199889

theorem company_valuation (P A B : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 2 * B) : 
  P / (A + B) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_company_valuation_l1998_199889


namespace NUMINAMATH_CALUDE_base_5_sum_l1998_199895

/-- Represents a digit in base 5 -/
def Base5Digit := { n : ℕ // n > 0 ∧ n < 5 }

/-- Converts a three-digit number in base 5 to its decimal representation -/
def toDecimal (a b c : Base5Digit) : ℕ := 25 * a.val + 5 * b.val + c.val

theorem base_5_sum (A B C : Base5Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : toDecimal A B C + toDecimal B C A + toDecimal C A B = 25 * 31 * A.val) :
  B.val + C.val = 4 :=
sorry

end NUMINAMATH_CALUDE_base_5_sum_l1998_199895


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1998_199877

/-- Given two lines l₁ and l₂ in the real plane, prove that if l₁ with equation x + 2y - 1 = 0 
    is parallel to l₂ with equation mx - y = 0, then m = -1/2. -/
theorem parallel_lines_slope (m : ℝ) : 
  (∀ x y : ℝ, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1998_199877


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l1998_199816

theorem snow_probability_first_week (p1 p2 : ℝ) : 
  p1 = 1/3 → p2 = 1/4 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l1998_199816


namespace NUMINAMATH_CALUDE_tail_cut_divisibility_by_7_l1998_199833

def tail_cut (n : ℕ) : ℕ :=
  (n / 10) - 2 * (n % 10)

def is_divisible_by_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem tail_cut_divisibility_by_7 (A : ℕ) :
  (A > 0) →
  (is_divisible_by_7 A ↔ 
    ∃ (k : ℕ), is_divisible_by_7 (Nat.iterate tail_cut k A)) :=
by sorry

end NUMINAMATH_CALUDE_tail_cut_divisibility_by_7_l1998_199833


namespace NUMINAMATH_CALUDE_b_fourth_plus_inverse_l1998_199894

theorem b_fourth_plus_inverse (b : ℝ) (h : (b + 1/b)^2 = 5) : b^4 + 1/b^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_plus_inverse_l1998_199894
