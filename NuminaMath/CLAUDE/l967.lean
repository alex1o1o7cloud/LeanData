import Mathlib

namespace NUMINAMATH_CALUDE_timothy_land_cost_l967_96772

/-- Represents the cost breakdown of Timothy's farm --/
structure FarmCosts where
  land_acres : ℕ
  house_cost : ℕ
  cow_count : ℕ
  cow_cost : ℕ
  chicken_count : ℕ
  chicken_cost : ℕ
  solar_install_hours : ℕ
  solar_install_rate : ℕ
  solar_equipment_cost : ℕ
  total_cost : ℕ

/-- Calculates the cost per acre of land given the farm costs --/
def land_cost_per_acre (costs : FarmCosts) : ℕ :=
  (costs.total_cost - 
   (costs.house_cost + 
    costs.cow_count * costs.cow_cost + 
    costs.chicken_count * costs.chicken_cost + 
    costs.solar_install_hours * costs.solar_install_rate + 
    costs.solar_equipment_cost)) / costs.land_acres

/-- Theorem stating that the cost per acre of Timothy's land is $20 --/
theorem timothy_land_cost (costs : FarmCosts) 
  (h1 : costs.land_acres = 30)
  (h2 : costs.house_cost = 120000)
  (h3 : costs.cow_count = 20)
  (h4 : costs.cow_cost = 1000)
  (h5 : costs.chicken_count = 100)
  (h6 : costs.chicken_cost = 5)
  (h7 : costs.solar_install_hours = 6)
  (h8 : costs.solar_install_rate = 100)
  (h9 : costs.solar_equipment_cost = 6000)
  (h10 : costs.total_cost = 147700) :
  land_cost_per_acre costs = 20 := by
  sorry


end NUMINAMATH_CALUDE_timothy_land_cost_l967_96772


namespace NUMINAMATH_CALUDE_solution_product_l967_96763

theorem solution_product (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l967_96763


namespace NUMINAMATH_CALUDE_f_min_at_three_l967_96755

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 has a minimum value when x = 3 -/
theorem f_min_at_three : 
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_three_l967_96755


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l967_96769

def damage_in_euros : ℝ := 45000000
def exchange_rate : ℝ := 0.9

theorem hurricane_damage_conversion :
  damage_in_euros * (1 / exchange_rate) = 49995000 := by
  sorry

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l967_96769


namespace NUMINAMATH_CALUDE_tourist_growth_rate_l967_96702

theorem tourist_growth_rate (feb mar apr may x : ℝ) : 
  mar = feb * (1 - 0.4) →
  apr = mar * (1 - 0.5) →
  may = 2 * feb →
  may = apr * (1 + x) →
  (1 - 0.4) * (1 - 0.5) * (1 + x) = 2 := by
sorry

end NUMINAMATH_CALUDE_tourist_growth_rate_l967_96702


namespace NUMINAMATH_CALUDE_angle_equivalence_l967_96790

/-- Given α = 2022°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 37π/30 radians. -/
theorem angle_equivalence (α β : Real) : 
  α = 2022 * (π / 180) →  -- Convert 2022° to radians
  (∃ k : ℤ, β = α + 2 * π * k) →  -- Same terminal side
  0 < β ∧ β < 2 * π →  -- β ∈ (0, 2π)
  β = 37 * π / 30 := by
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l967_96790


namespace NUMINAMATH_CALUDE_stadium_empty_seats_l967_96737

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: In a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats :
  empty_seats 92 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_stadium_empty_seats_l967_96737


namespace NUMINAMATH_CALUDE_total_highlighters_l967_96740

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 15

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 12

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 9

/-- The number of green highlighters in the teacher's desk -/
def green_highlighters : ℕ := 7

/-- The number of purple highlighters in the teacher's desk -/
def purple_highlighters : ℕ := 6

/-- Theorem stating that the total number of highlighters is 49 -/
theorem total_highlighters : 
  pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l967_96740


namespace NUMINAMATH_CALUDE_garden_vegetables_l967_96732

theorem garden_vegetables (potatoes cucumbers peppers : ℕ) : 
  cucumbers = potatoes - 60 →
  peppers = 2 * cucumbers →
  potatoes + cucumbers + peppers = 768 →
  potatoes = 237 := by
sorry

end NUMINAMATH_CALUDE_garden_vegetables_l967_96732


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l967_96747

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Calculates the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeWays (n : Nat) (k : Nat) : Nat :=
  sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeWays 6 3 = 122 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l967_96747


namespace NUMINAMATH_CALUDE_regular_polygon_inscribed_circle_l967_96717

theorem regular_polygon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_inscribed_circle_l967_96717


namespace NUMINAMATH_CALUDE_equal_area_floors_width_l967_96792

/-- Represents the dimensions of a rectangular floor -/
structure Floor :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.length * f.width

theorem equal_area_floors_width :
  ∀ (X Y : Floor),
  area X = area Y →
  X.length = 18 →
  X.width = 10 →
  Y.length = 20 →
  Y.width = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_floors_width_l967_96792


namespace NUMINAMATH_CALUDE_shirt_store_profit_optimization_l967_96765

/-- Represents the daily profit function for a shirt store -/
def daily_profit (x : ℝ) : ℝ := (20 + 2*x) * (40 - x)

/-- Represents the price reduction that achieves a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20 -- The actual value should be solved from the equation, but we're using the known result

/-- Represents the price reduction that maximizes daily profit -/
def optimal_price_reduction : ℝ := 15

/-- The maximum daily profit achieved at the optimal price reduction -/
def max_daily_profit : ℝ := 1250

theorem shirt_store_profit_optimization :
  (daily_profit (price_reduction_for_profit 1200) = 1200) ∧
  (∀ x : ℝ, daily_profit x ≤ max_daily_profit) ∧
  (daily_profit optimal_price_reduction = max_daily_profit) := by
  sorry


end NUMINAMATH_CALUDE_shirt_store_profit_optimization_l967_96765


namespace NUMINAMATH_CALUDE_f_properties_l967_96784

noncomputable def f (x : ℝ) := (2 * x - x^2) * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo 0 2, f x > 0) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-Real.sqrt 2 - ε) (-Real.sqrt 2 + ε), f (-Real.sqrt 2) ≤ f x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (Real.sqrt 2 - ε) (Real.sqrt 2 + ε), f x ≤ f (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l967_96784


namespace NUMINAMATH_CALUDE_roots_product_l967_96752

/-- Given that x₁ = ∛(17 - (27/4)√6) and x₂ = ∛(17 + (27/4)√6) are roots of x² - ax + b = 0, prove that ab = 10 -/
theorem roots_product (a b : ℝ) : 
  let x₁ : ℝ := (17 - (27/4) * Real.sqrt 6) ^ (1/3)
  let x₂ : ℝ := (17 + (27/4) * Real.sqrt 6) ^ (1/3)
  (x₁ ^ 2 - a * x₁ + b = 0) ∧ (x₂ ^ 2 - a * x₂ + b = 0) → a * b = 10 := by
  sorry


end NUMINAMATH_CALUDE_roots_product_l967_96752


namespace NUMINAMATH_CALUDE_units_digit_problem_l967_96734

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 4 ∧ 8 * 14 * 1955 - 6^4 ≡ n [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_problem_l967_96734


namespace NUMINAMATH_CALUDE_triangle_ratio_l967_96780

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  p : ℝ
  q : ℝ
  equation : ∀ x y : ℝ, x^2 / p^2 + y^2 / q^2 = 1

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                   (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (A.1 - C.1)^2 + (A.2 - C.2)^2

/-- The configuration described in the problem -/
structure Configuration where
  E : Ellipse
  T : EquilateralTriangle
  B_on_ellipse : T.B = (0, E.q)
  AC_parallel_x : T.A.2 = T.C.2
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  F₁_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₁ = (t * T.B.1 + (1 - t) * T.C.1, t * T.B.2 + (1 - t) * T.C.2)
  F₂_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₂ = (t * T.A.1 + (1 - t) * T.B.1, t * T.A.2 + (1 - t) * T.B.2)
  focal_distance : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4

theorem triangle_ratio (c : Configuration) : 
  let AB := ((c.T.A.1 - c.T.B.1)^2 + (c.T.A.2 - c.T.B.2)^2).sqrt
  let F₁F₂ := ((c.F₁.1 - c.F₂.1)^2 + (c.F₁.2 - c.F₂.2)^2).sqrt
  AB / F₁F₂ = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l967_96780


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l967_96766

theorem complex_in_second_quadrant (θ : Real) (h : θ ∈ Set.Ioo (3*Real.pi/4) (5*Real.pi/4)) :
  let z : ℂ := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l967_96766


namespace NUMINAMATH_CALUDE_min_transportation_cost_l967_96745

/-- Represents the transportation problem with two production locations and two delivery venues -/
structure TransportationProblem where
  unitsJ : ℕ  -- Units produced in location J
  unitsY : ℕ  -- Units produced in location Y
  unitsA : ℕ  -- Units delivered to venue A
  unitsB : ℕ  -- Units delivered to venue B
  costJB : ℕ  -- Transportation cost from J to B per unit
  fixedCost : ℕ  -- Fixed overhead cost

/-- Calculates the total transportation cost given the number of units transported from J to A -/
def totalCost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costJB * (p.unitsJ - x) + p.fixedCost

/-- Theorem stating the minimum transportation cost -/
theorem min_transportation_cost (p : TransportationProblem) 
    (h1 : p.unitsJ = 17) (h2 : p.unitsY = 15) (h3 : p.unitsA = 18) (h4 : p.unitsB = 14)
    (h5 : p.costJB = 200) (h6 : p.fixedCost = 19300) :
    ∃ (x : ℕ), x ≥ 3 ∧ 
    (∀ (y : ℕ), y ≥ 3 → totalCost p x ≤ totalCost p y) ∧
    totalCost p x = 19900 := by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l967_96745


namespace NUMINAMATH_CALUDE_additional_investment_rate_l967_96786

theorem additional_investment_rate
  (initial_investment : ℝ)
  (initial_rate : ℝ)
  (additional_investment : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 1400)
  (h4 : total_rate = 0.06)
  : (initial_investment * initial_rate + additional_investment * (112 / 1400)) / (initial_investment + additional_investment) = total_rate :=
by sorry

end NUMINAMATH_CALUDE_additional_investment_rate_l967_96786


namespace NUMINAMATH_CALUDE_cat_difference_l967_96764

theorem cat_difference (sheridan_cats garrett_cats : ℕ) 
  (h1 : sheridan_cats = 11) 
  (h2 : garrett_cats = 24) : 
  garrett_cats - sheridan_cats = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_difference_l967_96764


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l967_96781

theorem systematic_sampling_smallest_number
  (n : ℕ) -- Total number of products
  (k : ℕ) -- Sample size
  (x : ℕ) -- A number in the sample
  (h1 : n = 80) -- Total number of products is 80
  (h2 : k = 5) -- Sample size is 5
  (h3 : x = 42) -- The number 42 is in the sample
  (h4 : x < n) -- The number in the sample is less than the total number of products
  : ∃ (interval : ℕ) (smallest : ℕ),
    interval = n / k ∧
    x = interval * 2 + smallest ∧
    smallest = 10 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l967_96781


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l967_96760

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ+, n < 101 → ¬(Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.05) ∧ 
  (Real.sqrt 101 - Real.sqrt 100 < 0.05) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l967_96760


namespace NUMINAMATH_CALUDE_min_gcd_of_primes_squared_minus_one_l967_96795

theorem min_gcd_of_primes_squared_minus_one (p q : ℕ) : 
  Prime p → Prime q → p > 100 → q > 100 → 
  8 ≤ Nat.gcd (p^2 - 1) (q^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_gcd_of_primes_squared_minus_one_l967_96795


namespace NUMINAMATH_CALUDE_books_not_shared_l967_96756

/-- The number of books that are in either Emily's or Olivia's collection, but not both -/
def books_in_either_not_both (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) : ℕ :=
  (emily_total - shared_books) + olivia_unique

/-- Theorem stating the number of books in either Emily's or Olivia's collection, but not both -/
theorem books_not_shared (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) 
  (h1 : shared_books = 15)
  (h2 : emily_total = 23)
  (h3 : olivia_unique = 8) :
  books_in_either_not_both shared_books emily_total olivia_unique = 16 := by
  sorry

end NUMINAMATH_CALUDE_books_not_shared_l967_96756


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l967_96757

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 64 ways to distribute 6 distinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes :
  distribute_balls 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l967_96757


namespace NUMINAMATH_CALUDE_square_root_domain_only_five_satisfies_l967_96739

theorem square_root_domain (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) ↔ x ≥ 4 :=
sorry

theorem only_five_satisfies : 
  (∃ y : ℝ, y^2 = 5 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 0 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 1 - 4) ∧ 
  ¬(∃ y : ℝ, y^2 = 2 - 4) :=
sorry

end NUMINAMATH_CALUDE_square_root_domain_only_five_satisfies_l967_96739


namespace NUMINAMATH_CALUDE_min_b_for_q_half_or_more_l967_96711

def q (b : ℕ) : ℚ :=
  (Nat.choose (40 - b) 2 + Nat.choose (b - 1) 2) / 1225

theorem min_b_for_q_half_or_more : 
  ∀ b : ℕ, 1 ≤ b ∧ b ≤ 41 → (q b ≥ 1/2 ↔ b ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_b_for_q_half_or_more_l967_96711


namespace NUMINAMATH_CALUDE_min_stamps_proof_l967_96728

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  three_cent : ℕ
  four_cent : ℕ
  five_cent : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.three_cent + 4 * s.four_cent + 5 * s.five_cent

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ :=
  s.three_cent + s.four_cent + s.five_cent

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- The minimum number of stamps needed -/
def min_stamps : ℕ := 10

theorem min_stamps_proof :
  (∀ s : StampCombination, is_valid s → total_stamps s ≥ min_stamps) ∧
  (∃ s : StampCombination, is_valid s ∧ total_stamps s = min_stamps) := by
  sorry

#check min_stamps_proof

end NUMINAMATH_CALUDE_min_stamps_proof_l967_96728


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l967_96718

/-- Given an unfair coin with the following properties:
  * Probability of heads: 2/3
  * Probability of tails: 1/3
  * Gain on heads: $5
  * Loss on tails: $12
  This theorem proves that the expected value of a single coin flip
  is -2/3 dollars. -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 12
  p_heads * gain_heads + p_tails * (-loss_tails) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l967_96718


namespace NUMINAMATH_CALUDE_smallest_M_for_inequality_l967_96787

open Real

/-- The smallest real number M such that |∑ab(a²-b²)| ≤ M(∑a²)² holds for all real a, b, c -/
theorem smallest_M_for_inequality : 
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧ 
  (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') :=
sorry


end NUMINAMATH_CALUDE_smallest_M_for_inequality_l967_96787


namespace NUMINAMATH_CALUDE_square_garden_area_perimeter_difference_l967_96782

theorem square_garden_area_perimeter_difference :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 28 →
    s^2 - 4 * s = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_perimeter_difference_l967_96782


namespace NUMINAMATH_CALUDE_investment_growth_l967_96723

/-- Calculates the final amount after simple interest --/
def finalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 5 years is $350 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  finalAmount principal ((amount_after_2_years - principal) / (principal * 2)) 5 = 350 :=
by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l967_96723


namespace NUMINAMATH_CALUDE_four_students_three_communities_l967_96771

/-- The number of ways to distribute n students among k communities,
    where each student goes to exactly one community and each community
    receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students among 3 communities
    results in 36 different arrangements. -/
theorem four_students_three_communities :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l967_96771


namespace NUMINAMATH_CALUDE_right_triangle_area_l967_96704

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 3 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4) * Real.sin angle
  area = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l967_96704


namespace NUMINAMATH_CALUDE_opposites_imply_x_equals_one_l967_96713

theorem opposites_imply_x_equals_one : 
  ∀ x : ℝ, (-2 * x) = -(3 * x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_imply_x_equals_one_l967_96713


namespace NUMINAMATH_CALUDE_age_difference_l967_96762

/-- Given three people x, y, and z, where z is 10 decades younger than x,
    prove that the combined age of x and y is 100 years greater than
    the combined age of y and z. -/
theorem age_difference (x y z : ℕ) (h : z = x - 100) :
  (x + y) - (y + z) = 100 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l967_96762


namespace NUMINAMATH_CALUDE_cos_theta_plus_5pi_6_l967_96715

theorem cos_theta_plus_5pi_6 (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ / 2 + π / 6) = 2 / 3) : 
  Real.cos (θ + 5 * π / 6) = -4 * Real.sqrt 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_5pi_6_l967_96715


namespace NUMINAMATH_CALUDE_quadratic_roots_squared_difference_l967_96758

theorem quadratic_roots_squared_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (x₁^2 - x₂^2 = c^2 / a^2) ↔ (b^4 - c^4 = 4*a*b^2*c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_squared_difference_l967_96758


namespace NUMINAMATH_CALUDE_absent_days_calculation_l967_96783

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days based on contract details -/
def absentDays (contract : ContractDetails) : ℚ :=
  (contract.totalDays * contract.paymentPerDay - contract.totalReceived) / (contract.paymentPerDay + contract.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 8 -/
theorem absent_days_calculation (contract : ContractDetails) 
  (h1 : contract.totalDays = 30)
  (h2 : contract.paymentPerDay = 25)
  (h3 : contract.finePerDay = 7.5)
  (h4 : contract.totalReceived = 490) :
  absentDays contract = 8 := by
  sorry

#eval absentDays { totalDays := 30, paymentPerDay := 25, finePerDay := 7.5, totalReceived := 490 }

end NUMINAMATH_CALUDE_absent_days_calculation_l967_96783


namespace NUMINAMATH_CALUDE_arithmetic_progression_logarithm_l967_96743

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the arithmetic progression property
def isArithmeticProgression (a b c : ℝ) : Prop := 2 * b = a + c

-- Theorem statement
theorem arithmetic_progression_logarithm :
  isArithmeticProgression (lg 3) (lg 6) (lg x) → x = 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_logarithm_l967_96743


namespace NUMINAMATH_CALUDE_position_3_1_is_B_l967_96796

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Char

-- Define the valid letters
def ValidLetter (c : Char) : Prop := c ∈ ['A', 'B', 'C', 'D', 'E']

-- Define the property of a valid grid
def ValidGrid (g : Grid) : Prop :=
  (∀ r c, ValidLetter (g r c)) ∧
  (∀ r, ∀ c₁ c₂, c₁ ≠ c₂ → g r c₁ ≠ g r c₂) ∧
  (∀ c, ∀ r₁ r₂, r₁ ≠ r₂ → g r₁ c ≠ g r₂ c) ∧
  (∀ i j, i ≠ j → g i i ≠ g j j) ∧
  (∀ i j, i ≠ j → g i (4 - i) ≠ g j (4 - j))

-- Define the theorem
theorem position_3_1_is_B (g : Grid) (h : ValidGrid g)
  (h1 : g 0 0 = 'A') (h2 : g 3 0 = 'D') (h3 : g 4 0 = 'E') :
  g 2 0 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_position_3_1_is_B_l967_96796


namespace NUMINAMATH_CALUDE_gold_checkpoint_problem_l967_96751

theorem gold_checkpoint_problem (x : ℝ) : 
  x > 0 →
  x - x * (1/2 + 1/3 * 1/2 + 1/4 * 2/3 + 1/5 * 3/4 + 1/6 * 4/5) = 1 →
  x = 1.2 := by
sorry

end NUMINAMATH_CALUDE_gold_checkpoint_problem_l967_96751


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l967_96712

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/b = 2 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 2 ∧ x₀ + 2*y₀ = (3 + 2*Real.sqrt 2)/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l967_96712


namespace NUMINAMATH_CALUDE_expression_evaluation_l967_96721

theorem expression_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  5 * x^2 - 2 * (3 * y^2 + 6 * x * y) - (2 * x^2 - 6 * y^2) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l967_96721


namespace NUMINAMATH_CALUDE_boat_speed_l967_96741

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 38) 
  (h2 : against_stream = 16) : 
  (along_stream + against_stream) / 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l967_96741


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l967_96761

theorem fixed_point_of_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l967_96761


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_irrational_product_of_roots_l967_96719

theorem cubic_polynomial_with_irrational_product_of_roots :
  ∃ (a b c : ℚ) (u v : ℝ),
    (u^3 + a*u^2 + b*u + c = 0) ∧
    (v^3 + a*v^2 + b*v + c = 0) ∧
    ((u*v)^3 + a*(u*v)^2 + b*(u*v) + c = 0) ∧
    ¬(∃ (q : ℚ), u*v = q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_irrational_product_of_roots_l967_96719


namespace NUMINAMATH_CALUDE_max_value_of_f_l967_96716

-- Define the function f(x) = -3x^2 + 9
def f (x : ℝ) : ℝ := -3 * x^2 + 9

-- Theorem stating that the maximum value of f(x) is 9
theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l967_96716


namespace NUMINAMATH_CALUDE_line_y_intercept_l967_96726

/-- A line with slope 6 and x-intercept (8, 0) has y-intercept (0, -48) -/
theorem line_y_intercept (f : ℝ → ℝ) (h_slope : ∀ x y, f y - f x = 6 * (y - x)) 
  (h_x_intercept : f 8 = 0) : f 0 = -48 := by
  sorry

end NUMINAMATH_CALUDE_line_y_intercept_l967_96726


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_108_l967_96799

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem f_g_f_3_equals_108 : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_108_l967_96799


namespace NUMINAMATH_CALUDE_qin_jiushao_algorithm_l967_96750

theorem qin_jiushao_algorithm (n : ℕ) (x : ℝ) (h1 : n = 5) (h2 : x = 2) :
  (Finset.range (n + 1)).sum (fun i => x ^ i) = 63 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_algorithm_l967_96750


namespace NUMINAMATH_CALUDE_area_perimeter_product_l967_96777

/-- A square on a grid with vertices at (1,5), (5,5), (5,1), and (1,1) -/
structure GridSquare where
  v1 : (ℕ × ℕ) := (1, 5)
  v2 : (ℕ × ℕ) := (5, 5)
  v3 : (ℕ × ℕ) := (5, 1)
  v4 : (ℕ × ℕ) := (1, 1)

/-- Calculate the side length of the GridSquare -/
def sideLength (s : GridSquare) : ℕ :=
  (s.v2.1 - s.v1.1)

/-- Calculate the area of the GridSquare -/
def area (s : GridSquare) : ℕ :=
  (sideLength s) ^ 2

/-- Calculate the perimeter of the GridSquare -/
def perimeter (s : GridSquare) : ℕ :=
  4 * (sideLength s)

/-- Theorem: The product of the area and perimeter of the GridSquare is 256 -/
theorem area_perimeter_product (s : GridSquare) : 
  area s * perimeter s = 256 := by
  sorry


end NUMINAMATH_CALUDE_area_perimeter_product_l967_96777


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l967_96746

def z : ℂ := Complex.I * (1 + Complex.I)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l967_96746


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l967_96727

theorem smallest_positive_omega : ∃ (ω : ℝ), 
  (ω > 0) ∧ 
  (∀ x, Real.sin (ω * (x - Real.pi / 6)) = Real.cos (ω * x)) ∧
  (∀ ω' > 0, (∀ x, Real.sin (ω' * (x - Real.pi / 6)) = Real.cos (ω' * x)) → ω ≤ ω') ∧
  ω = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l967_96727


namespace NUMINAMATH_CALUDE_inequality_proof_l967_96753

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (a / (b + 1)) + (b / (a + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l967_96753


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l967_96749

theorem quadratic_solution_sum (x : ℝ) (m n p : ℕ) : 
  x * (4 * x - 9) = -4 ∧ 
  (∃ (r : ℝ), r * r = n ∧ 
    (x = (m + r) / p ∨ x = (m - r) / p)) ∧
  Nat.gcd m (Nat.gcd n p) = 1 →
  m + n + p = 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l967_96749


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l967_96785

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l967_96785


namespace NUMINAMATH_CALUDE_book_selection_problem_l967_96733

theorem book_selection_problem (total_books math_books physics_books selected_books selected_math selected_physics : ℕ) :
  total_books = 20 →
  math_books = 6 →
  physics_books = 4 →
  selected_books = 8 →
  selected_math = 4 →
  selected_physics = 2 →
  (Nat.choose math_books selected_math) * (Nat.choose physics_books selected_physics) *
  (Nat.choose (total_books - math_books - physics_books) (selected_books - selected_math - selected_physics)) = 4050 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_problem_l967_96733


namespace NUMINAMATH_CALUDE_max_value_of_f_l967_96770

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l967_96770


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l967_96754

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are the same -/
def symmetricAboutYAxis (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

theorem symmetric_points_sum (m n : ℝ) :
  let A : Point := ⟨m, 2⟩
  let B : Point := ⟨3, n⟩
  symmetricAboutYAxis A B → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l967_96754


namespace NUMINAMATH_CALUDE_no_win_prob_at_least_two_no_win_prob_l967_96788

-- Define the probability of winning for a single bottle
def win_prob : ℚ := 1/6

-- Define the number of students
def num_students : ℕ := 3

-- Theorem 1: Probability that none of the three students win a prize
theorem no_win_prob : 
  (1 - win_prob) ^ num_students = 125/216 := by sorry

-- Theorem 2: Probability that at least two of the three students do not win a prize
theorem at_least_two_no_win_prob : 
  1 - (Nat.choose num_students 2 * win_prob^2 * (1 - win_prob) + win_prob^num_students) = 25/27 := by sorry

end NUMINAMATH_CALUDE_no_win_prob_at_least_two_no_win_prob_l967_96788


namespace NUMINAMATH_CALUDE_expression_value_l967_96791

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l967_96791


namespace NUMINAMATH_CALUDE_cloth_loss_problem_l967_96793

/-- Calculates the loss per metre of cloth given the total quantity sold,
    total selling price, and cost price per metre. -/
def loss_per_metre (quantity : ℕ) (selling_price total_cost_price : ℚ) : ℚ :=
  (total_cost_price - selling_price) / quantity

theorem cloth_loss_problem (quantity : ℕ) (selling_price cost_price_per_metre : ℚ) 
  (h1 : quantity = 200)
  (h2 : selling_price = 12000)
  (h3 : cost_price_per_metre = 66) :
  loss_per_metre quantity selling_price (quantity * cost_price_per_metre) = 6 := by
sorry

end NUMINAMATH_CALUDE_cloth_loss_problem_l967_96793


namespace NUMINAMATH_CALUDE_union_complement_equality_l967_96773

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 3, 4}

theorem union_complement_equality : A ∪ (U \ B) = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l967_96773


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l967_96778

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2*y + 7| ≤ 16)) ∧ (|2*x + 7| ≤ 16) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l967_96778


namespace NUMINAMATH_CALUDE_min_value_and_max_t_l967_96779

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), m = 1 ∧ ∀ x, f a b x ≥ m) →
  (2*a + b = 2) ∧
  (∀ t, (a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t₀, t₀ = 9/2 ∧ a + 2*b ≥ t₀*a*b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_t_l967_96779


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l967_96775

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (1 + Complex.I) / (1 - Complex.I) → 
  z = Complex.mk a b → 
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l967_96775


namespace NUMINAMATH_CALUDE_fredrickson_chickens_l967_96794

/-- Represents the total number of chickens Mrs. Fredrickson has -/
def total_chickens : ℕ := 80

/-- Represents the number of chickens that do not lay eggs -/
def non_laying_chickens : ℕ := 35

/-- The proportion of chickens that are roosters -/
def rooster_ratio : ℚ := 1/4

/-- The proportion of hens that lay eggs -/
def laying_hen_ratio : ℚ := 3/4

theorem fredrickson_chickens :
  (rooster_ratio * total_chickens : ℚ) +
  ((1 - rooster_ratio) * (1 - laying_hen_ratio) * total_chickens : ℚ) =
  non_laying_chickens :=
sorry

end NUMINAMATH_CALUDE_fredrickson_chickens_l967_96794


namespace NUMINAMATH_CALUDE_sin_cos_difference_identity_l967_96735

theorem sin_cos_difference_identity :
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_identity_l967_96735


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l967_96744

/-- Represents the amount of pure water added in liters -/
def W : ℝ := 1

/-- The initial volume of salt solution in liters -/
def initial_volume : ℝ := 1

/-- The initial concentration of salt in the solution -/
def initial_concentration : ℝ := 0.40

/-- The final concentration of salt in the mixture -/
def final_concentration : ℝ := 0.20

theorem salt_solution_mixture :
  initial_volume * initial_concentration = 
  (initial_volume + W) * final_concentration := by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l967_96744


namespace NUMINAMATH_CALUDE_prime_divides_or_coprime_l967_96768

theorem prime_divides_or_coprime (p n : ℕ) (hp : Prime p) :
  p ∣ n ∨ Nat.gcd p n = 1 := by sorry

end NUMINAMATH_CALUDE_prime_divides_or_coprime_l967_96768


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l967_96708

/-- A quadratic inequality with parameter m has exactly 3 integer solutions -/
def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃! (a b c : ℤ), (m : ℝ) * (a : ℝ)^2 + (2 - m) * (a : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (b : ℝ)^2 + (2 - m) * (b : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (c : ℝ)^2 + (2 - m) * (c : ℝ) - 2 > 0 ∧
                   ∀ (x : ℤ), (m : ℝ) * (x : ℝ)^2 + (2 - m) * (x : ℝ) - 2 > 0 → (x = a ∨ x = b ∨ x = c)

/-- The main theorem -/
theorem quadratic_inequality_solution_range (m : ℝ) :
  has_three_integer_solutions m → -1/2 < m ∧ m ≤ -2/5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l967_96708


namespace NUMINAMATH_CALUDE_sphere_radius_is_6_l967_96738

/-- The radius of a sphere whose surface area is equal to the curved surface area of a right circular cylinder with height and diameter both 12 cm. -/
def sphere_radius : ℝ := 6

/-- The height of the cylinder. -/
def cylinder_height : ℝ := 12

/-- The diameter of the cylinder. -/
def cylinder_diameter : ℝ := 12

/-- The theorem stating that the radius of the sphere is 6 cm. -/
theorem sphere_radius_is_6 :
  sphere_radius = 6 ∧
  4 * Real.pi * sphere_radius ^ 2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_is_6_l967_96738


namespace NUMINAMATH_CALUDE_two_distinct_roots_l967_96776

/-- The custom operation ⊗ for real numbers -/
def otimes (a b : ℝ) : ℝ := b^2 - a*b

/-- Theorem stating that the equation (k-3) ⊗ x = k-1 has two distinct real roots for any real k -/
theorem two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (k-3) x₁ = k-1 ∧ otimes (k-3) x₂ = k-1 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l967_96776


namespace NUMINAMATH_CALUDE_comic_books_liked_by_females_l967_96703

/-- Given a comic store with the following properties:
  - There are 300 comic books in total
  - Males like 120 comic books
  - 30% of comic books are disliked by both males and females
  Prove that the percentage of comic books liked by females is 30% -/
theorem comic_books_liked_by_females 
  (total_comics : ℕ) 
  (liked_by_males : ℕ) 
  (disliked_percentage : ℚ) :
  total_comics = 300 →
  liked_by_males = 120 →
  disliked_percentage = 30 / 100 →
  (total_comics - (disliked_percentage * total_comics).num - liked_by_males) / total_comics = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_comic_books_liked_by_females_l967_96703


namespace NUMINAMATH_CALUDE_budget_allocation_l967_96700

theorem budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) (gm_microorganisms : ℝ) (industrial_lubricants : ℝ) (astrophysics_degrees : ℝ) :
  microphotonics = 12 ∧
  home_electronics = 24 ∧
  gm_microorganisms = 29 ∧
  industrial_lubricants = 8 ∧
  astrophysics_degrees = 43.2 →
  100 - (microphotonics + home_electronics + gm_microorganisms + industrial_lubricants + (astrophysics_degrees / 360 * 100)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l967_96700


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l967_96720

-- Define the vectors a and b
variable (a b : ℝ × ℝ)

-- Define the conditions
axiom unit_a : ‖a‖ = 1
axiom unit_b : ‖b‖ = 1
axiom angle_ab : a • b = -1/2  -- cos(120°) = -1/2

-- Theorem to prove
theorem vector_difference_magnitude : ‖a - 3 • b‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l967_96720


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l967_96701

theorem complex_fraction_simplification :
  (1 + 2 * Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l967_96701


namespace NUMINAMATH_CALUDE_intersection_line_equation_l967_96722

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (A B : ℝ × ℝ),
    (A ≠ B) ∧
    (circle1 A.1 A.2) ∧ (circle1 B.1 B.2) ∧
    (circle2 A.1 A.2) ∧ (circle2 B.1 B.2) →
    (∀ (x y : ℝ), line x y ↔ ∃ (t : ℝ), x = (1-t)*A.1 + t*B.1 ∧ y = (1-t)*A.2 + t*B.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l967_96722


namespace NUMINAMATH_CALUDE_max_m_inequality_l967_96707

theorem max_m_inequality (m : ℝ) : 
  (∀ a b : ℝ, (a / Real.exp a - b)^2 ≥ m - (a - b + 3)^2) → m ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l967_96707


namespace NUMINAMATH_CALUDE_square_area_equals_one_l967_96789

theorem square_area_equals_one (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l = 8 / 9) :
  ∃ s : ℝ, s > 0 ∧ 4 * s = 6 * w ∧ s^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_one_l967_96789


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l967_96724

/-- Given two cylinders A and B, where A's radius is r and height is h,
    and B's radius is h and height is r, prove that if A's volume is
    three times B's volume, then A's volume is 9πh^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 3 * (π * h^2 * r) →
  π * r^2 * h = 9 * π * h^3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l967_96724


namespace NUMINAMATH_CALUDE_candy_bar_sales_earnings_candy_bar_sales_proof_l967_96748

/-- Calculates the total amount earned from candy bar sales given the specified conditions --/
theorem candy_bar_sales_earnings (num_members : ℕ) (type_a_price type_b_price : ℚ) 
  (avg_total_bars avg_type_a avg_type_b : ℕ) : ℚ :=
  let total_bars := num_members * avg_total_bars
  let total_type_a := num_members * avg_type_a
  let total_type_b := num_members * avg_type_b
  let earnings_type_a := total_type_a * type_a_price
  let earnings_type_b := total_type_b * type_b_price
  earnings_type_a + earnings_type_b

/-- Proves that the group earned $95 from their candy bar sales --/
theorem candy_bar_sales_proof :
  candy_bar_sales_earnings 20 (1/2) (3/4) 8 5 3 = 95 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_earnings_candy_bar_sales_proof_l967_96748


namespace NUMINAMATH_CALUDE_cycle_reappearance_l967_96706

theorem cycle_reappearance (letter_seq_length digit_seq_length : ℕ) 
  (h1 : letter_seq_length = 9)
  (h2 : digit_seq_length = 4) : 
  Nat.lcm letter_seq_length digit_seq_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_cycle_reappearance_l967_96706


namespace NUMINAMATH_CALUDE_existence_of_finite_set_with_1993_unit_distance_neighbors_l967_96731

theorem existence_of_finite_set_with_1993_unit_distance_neighbors :
  ∃ (A : Set (ℝ × ℝ)), Set.Finite A ∧
    ∀ X ∈ A, ∃ (Y : Fin 1993 → ℝ × ℝ),
      (∀ i, Y i ∈ A) ∧
      (∀ i j, i ≠ j → Y i ≠ Y j) ∧
      (∀ i, dist X (Y i) = 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_finite_set_with_1993_unit_distance_neighbors_l967_96731


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l967_96729

/-- The probability of getting 4 consecutive heads before 3 consecutive tails
    when repeatedly flipping a fair coin -/
def q : ℚ := 31 / 63

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1 / 2)

/-- The event of getting 4 consecutive heads -/
def four_heads : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 4 → n + i = 1

/-- The event of getting 3 consecutive tails -/
def three_tails : ℕ → Prop := λ n => ∀ i, i ∈ Finset.range 3 → n + i = 0

/-- The probability of an event occurring before another event
    when repeatedly performing an experiment -/
def prob_before (p : ℚ) (event1 event2 : ℕ → Prop) : Prop :=
  ∃ n : ℕ, (∀ k < n, ¬event1 k ∧ ¬event2 k) ∧ event1 n ∧ (∀ k ≤ n, ¬event2 k)

theorem four_heads_before_three_tails :
  fair_coin (λ p => prob_before q four_heads three_tails) :=
sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l967_96729


namespace NUMINAMATH_CALUDE_sum_of_squares_204_l967_96736

theorem sum_of_squares_204 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ+) 
  (h : a₁^2 + (2*a₂)^2 + (3*a₃)^2 + (4*a₄)^2 + (5*a₅)^2 + (6*a₆)^2 + (7*a₇)^2 + (8*a₈)^2 = 204) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_204_l967_96736


namespace NUMINAMATH_CALUDE_sqrt_sum_implies_product_l967_96742

theorem sqrt_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_implies_product_l967_96742


namespace NUMINAMATH_CALUDE_parallelogram_count_formula_l967_96730

/-- An equilateral triangle with side length n, tiled with n^2 smaller equilateral triangles -/
structure TiledTriangle (n : ℕ) where
  side_length : ℕ := n
  num_small_triangles : ℕ := n^2

/-- The number of parallelograms in a tiled equilateral triangle -/
def count_parallelograms (t : TiledTriangle n) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating that the number of parallelograms in a tiled equilateral triangle
    is equal to 3 * (n+2 choose 4) -/
theorem parallelogram_count_formula (n : ℕ) (t : TiledTriangle n) :
  count_parallelograms t = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_formula_l967_96730


namespace NUMINAMATH_CALUDE_circle_radius_a_values_l967_96797

theorem circle_radius_a_values (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + 4*a*y = 0) →
  (∃ x₀ y₀ : ℝ, (x₀ + a)^2 + (y₀ + 2*a)^2 = 5) →
  (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_a_values_l967_96797


namespace NUMINAMATH_CALUDE_polygon_sides_count_l967_96774

theorem polygon_sides_count (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l967_96774


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l967_96705

/-- Given that a, b, and c are the roots of x^3 - 3x - 2 = 0,
    prove that a(b+c)^2 + b(c+a)^2 + c(a+b)^2 = -6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 3*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b+c)^2 + b*(c+a)^2 + c*(a+b)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l967_96705


namespace NUMINAMATH_CALUDE_fraction_less_than_mode_l967_96710

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_mode_l967_96710


namespace NUMINAMATH_CALUDE_lollipop_distribution_theorem_l967_96709

/-- Represents the lollipop distribution rules and class attendance --/
structure LollipopDistribution where
  mainTeacherRatio : ℕ  -- Students per lollipop for main teacher
  assistantRatio : ℕ    -- Students per lollipop for assistant
  assistantThreshold : ℕ -- Threshold for assistant to start giving lollipops
  initialStudents : ℕ   -- Initial number of students
  lateStudents : List ℕ  -- List of additional students joining later

/-- Calculates the total number of lollipops given away --/
def totalLollipops (d : LollipopDistribution) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, 21 lollipops will be given away --/
theorem lollipop_distribution_theorem :
  let d : LollipopDistribution := {
    mainTeacherRatio := 5,
    assistantRatio := 7,
    assistantThreshold := 30,
    initialStudents := 45,
    lateStudents := [10, 5, 5]
  }
  totalLollipops d = 21 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_distribution_theorem_l967_96709


namespace NUMINAMATH_CALUDE_apple_weight_average_l967_96759

theorem apple_weight_average (standard_weight : ℝ) (deviations : List ℝ) : 
  standard_weight = 30 →
  deviations = [0.4, -0.2, -0.8, -0.4, 1, 0.3, 0.5, -2, 0.5, -0.1] →
  (standard_weight + (deviations.sum / deviations.length)) = 29.92 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_average_l967_96759


namespace NUMINAMATH_CALUDE_circle_circumference_ratio_l967_96714

/-- The ratio of the new circumference to the original circumference when the radius is increased by 2 units -/
theorem circle_circumference_ratio (r : ℝ) (h : r > 0) :
  (2 * Real.pi * (r + 2)) / (2 * Real.pi * r) = 1 + 2 / r := by
  sorry

#check circle_circumference_ratio

end NUMINAMATH_CALUDE_circle_circumference_ratio_l967_96714


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l967_96725

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l967_96725


namespace NUMINAMATH_CALUDE_contrapositive_equality_l967_96767

theorem contrapositive_equality (a b : ℝ) : 
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l967_96767


namespace NUMINAMATH_CALUDE_gregs_shopping_expenditure_l967_96798

/-- Greg's shopping expenditure theorem -/
theorem gregs_shopping_expenditure (shirt_cost shoes_cost : ℕ) : 
  shirt_cost + shoes_cost = 300 →
  shoes_cost = 2 * shirt_cost + 9 →
  shirt_cost = 97 := by
  sorry

end NUMINAMATH_CALUDE_gregs_shopping_expenditure_l967_96798
