import Mathlib

namespace NUMINAMATH_CALUDE_cats_adoption_proof_l624_62458

def adopt_cats (initial_cats : ℕ) (added_cats : ℕ) (cats_per_adopter : ℕ) (final_cats : ℕ) : ℕ :=
  ((initial_cats + added_cats) - final_cats) / cats_per_adopter

theorem cats_adoption_proof :
  adopt_cats 20 3 2 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cats_adoption_proof_l624_62458


namespace NUMINAMATH_CALUDE_binomial_gcd_divisibility_l624_62417

theorem binomial_gcd_divisibility (n k : ℕ+) :
  ∃ m : ℕ, m * n = Nat.choose n.val k.val * Nat.gcd n.val k.val := by
  sorry

end NUMINAMATH_CALUDE_binomial_gcd_divisibility_l624_62417


namespace NUMINAMATH_CALUDE_arithmetic_progression_cosine_squared_l624_62411

open Real

theorem arithmetic_progression_cosine_squared (x y z : ℝ) (α : ℝ) :
  (∃ k : ℝ, y = x + α ∧ z = y + α) →  -- x, y, z form an arithmetic progression
  α = arccos (-1/3) →                 -- common difference
  (∃ m : ℝ, 3 / cos y = 1 / cos x + m ∧ 1 / cos z = 3 / cos y + m) →  -- 1/cos(x), 3/cos(y), 1/cos(z) form an AP
  cos y ^ 2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cosine_squared_l624_62411


namespace NUMINAMATH_CALUDE_super_soup_expansion_l624_62435

/-- The number of new stores opened by Super Soup in 2020 -/
def new_stores_2020 (initial_2018 : ℕ) (opened_2019 closed_2019 closed_2020 final_2020 : ℕ) : ℕ :=
  final_2020 - (initial_2018 + opened_2019 - closed_2019 - closed_2020)

/-- Theorem stating that Super Soup opened 10 new stores in 2020 -/
theorem super_soup_expansion :
  new_stores_2020 23 5 2 6 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_super_soup_expansion_l624_62435


namespace NUMINAMATH_CALUDE_unique_coin_expected_value_l624_62491

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * (-loss_tails)

theorem unique_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 4
  let loss_tails : ℚ := 3
  coin_flip_expected_value p_heads p_tails win_heads loss_tails = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_unique_coin_expected_value_l624_62491


namespace NUMINAMATH_CALUDE_kitchen_broken_fraction_l624_62416

theorem kitchen_broken_fraction :
  let foyer_broken : ℕ := 10
  let kitchen_total : ℕ := 35
  let total_not_broken : ℕ := 34
  let foyer_total : ℕ := foyer_broken * 3
  let total_bulbs : ℕ := foyer_total + kitchen_total
  let total_broken : ℕ := total_bulbs - total_not_broken
  let kitchen_broken : ℕ := total_broken - foyer_broken
  (kitchen_broken : ℚ) / kitchen_total = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_kitchen_broken_fraction_l624_62416


namespace NUMINAMATH_CALUDE_three_digit_with_three_without_five_seven_l624_62468

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n = k * 10 + d ∨ n = k * 100 + d ∨ ∃ m, n = k * 100 + m * 10 + d

def not_contains_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ¬(contains_digit n d₁) ∧ ¬(contains_digit n d₂)

theorem three_digit_with_three_without_five_seven (n : ℕ) :
  (is_three_digit n ∧ contains_digit n 3 ∧ not_contains_digits n 5 7) →
  ∃ S : Finset ℕ, S.card = 154 ∧ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_three_digit_with_three_without_five_seven_l624_62468


namespace NUMINAMATH_CALUDE_bank_comparison_l624_62431

/-- Calculates the annual yield given a quarterly interest rate -/
def annual_yield_quarterly (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

/-- Calculates the annual yield given an annual interest rate -/
def annual_yield_annual (annual_rate : ℝ) : ℝ :=
  annual_rate

theorem bank_comparison (bank1_quarterly_rate : ℝ) (bank2_annual_rate : ℝ)
    (h1 : bank1_quarterly_rate = 0.8)
    (h2 : bank2_annual_rate = -9) :
    annual_yield_quarterly bank1_quarterly_rate > annual_yield_annual bank2_annual_rate := by
  sorry

#eval annual_yield_quarterly 0.8
#eval annual_yield_annual (-9)

end NUMINAMATH_CALUDE_bank_comparison_l624_62431


namespace NUMINAMATH_CALUDE_percent_of_itself_l624_62460

theorem percent_of_itself (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 4) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_itself_l624_62460


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l624_62450

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    (∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) (heven : IsEven f) (hroots : HasFourRoots f) :
  ∃ (a b c d : ℝ), f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l624_62450


namespace NUMINAMATH_CALUDE_gumballs_per_pair_is_9_l624_62432

/-- The number of gumballs Kim gets for each pair of earrings --/
def gumballs_per_pair : ℕ :=
  let earrings_day1 : ℕ := 3
  let earrings_day2 : ℕ := 2 * earrings_day1
  let earrings_day3 : ℕ := earrings_day2 - 1
  let total_earrings : ℕ := earrings_day1 + earrings_day2 + earrings_day3
  let gumballs_per_day : ℕ := 3
  let total_days : ℕ := 42
  let total_gumballs : ℕ := gumballs_per_day * total_days
  total_gumballs / total_earrings

theorem gumballs_per_pair_is_9 : gumballs_per_pair = 9 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_per_pair_is_9_l624_62432


namespace NUMINAMATH_CALUDE_student_sister_weight_l624_62445

theorem student_sister_weight (student_weight : ℝ) (weight_loss : ℝ) :
  student_weight = 90 ∧
  (student_weight - weight_loss) = 2 * ((student_weight - weight_loss) / 2) ∧
  weight_loss = 6 →
  student_weight + ((student_weight - weight_loss) / 2) = 132 :=
by sorry

end NUMINAMATH_CALUDE_student_sister_weight_l624_62445


namespace NUMINAMATH_CALUDE_min_sum_of_determinant_condition_l624_62495

theorem min_sum_of_determinant_condition (x y : ℤ) 
  (h : 1 < 6 - x * y ∧ 6 - x * y < 3) : 
  ∃ (a b : ℤ), a + b = -5 ∧ 
    (∀ (c d : ℤ), 1 < 6 - c * d ∧ 6 - c * d < 3 → a + b ≤ c + d) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_determinant_condition_l624_62495


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l624_62467

theorem tax_reduction_theorem (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_consumption := original_consumption * 1.05
  let new_revenue := original_tax * original_consumption * 0.84
  let new_tax := new_revenue / new_consumption
  (original_tax - new_tax) / original_tax = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l624_62467


namespace NUMINAMATH_CALUDE_grouping_theorem_l624_62429

/-- The number of ways to distribute 4 men and 5 women into groups -/
def grouping_ways : ℕ := 
  let men : ℕ := 4
  let women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let num_small_groups : ℕ := 2
  100

/-- Theorem stating that the number of ways to distribute 4 men and 5 women
    into two groups of two people and one group of five people, 
    with at least one man and one woman in each group, is 100 -/
theorem grouping_theorem : grouping_ways = 100 := by
  sorry

end NUMINAMATH_CALUDE_grouping_theorem_l624_62429


namespace NUMINAMATH_CALUDE_milk_cost_per_liter_l624_62497

/-- Represents the milkman's milk mixture problem -/
def MilkProblem (total_milk pure_milk water_added mixture_price profit : ℝ) : Prop :=
  total_milk = 30 ∧
  pure_milk = 20 ∧
  water_added = 5 ∧
  (pure_milk + water_added) * mixture_price - pure_milk * mixture_price = profit ∧
  profit = 35

/-- The cost of pure milk per liter is 7 rupees -/
theorem milk_cost_per_liter (total_milk pure_milk water_added mixture_price profit : ℝ) 
  (h : MilkProblem total_milk pure_milk water_added mixture_price profit) : 
  mixture_price = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_cost_per_liter_l624_62497


namespace NUMINAMATH_CALUDE_reasoning_is_deductive_l624_62464

-- Define the set of all substances
variable (Substance : Type)

-- Define the property of being a metal
variable (is_metal : Substance → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Substance → Prop)

-- Define iron as a specific substance
variable (iron : Substance)

-- Theorem stating that the given reasoning is deductive
theorem reasoning_is_deductive 
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals can conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  (h3 : conducts_electricity iron)                 -- Iron can conduct electricity
  : Prop :=
sorry

end NUMINAMATH_CALUDE_reasoning_is_deductive_l624_62464


namespace NUMINAMATH_CALUDE_toy_production_rate_l624_62479

/-- Represents the toy production in a factory --/
structure ToyFactory where
  weekly_production : ℕ
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ

/-- Calculates the hourly toy production rate --/
def hourly_production_rate (factory : ToyFactory) : ℚ :=
  let total_hours := factory.monday_hours + factory.tuesday_hours + factory.wednesday_hours + factory.thursday_hours
  factory.weekly_production / total_hours

/-- Theorem stating the hourly production rate for the given factory --/
theorem toy_production_rate (factory : ToyFactory) 
  (h1 : factory.weekly_production = 20500)
  (h2 : factory.monday_hours = 8)
  (h3 : factory.tuesday_hours = 7)
  (h4 : factory.wednesday_hours = 9)
  (h5 : factory.thursday_hours = 6) :
  ∃ (ε : ℚ), abs (hourly_production_rate factory - 683.33) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_toy_production_rate_l624_62479


namespace NUMINAMATH_CALUDE_solve_linear_equation_l624_62436

theorem solve_linear_equation (n : ℚ) (h : 2 * n + 5 = 16) : 2 * n - 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l624_62436


namespace NUMINAMATH_CALUDE_one_fifth_equals_point_two_l624_62439

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = 0.200000 := by sorry

end NUMINAMATH_CALUDE_one_fifth_equals_point_two_l624_62439


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l624_62478

/-- Given a square with side length 2 and four congruent isosceles triangles
    constructed on its sides, if the sum of the triangles' areas equals
    the square's area, then each triangle's congruent side length is √2. -/
theorem isosceles_triangle_side_length :
  let square_side : ℝ := 2
  let square_area : ℝ := square_side ^ 2
  let triangle_area : ℝ := square_area / 4
  let triangle_base : ℝ := square_side
  let triangle_height : ℝ := 2 * triangle_area / triangle_base
  let triangle_side : ℝ := Real.sqrt (triangle_height ^ 2 + (triangle_base / 2) ^ 2)
  triangle_side = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l624_62478


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l624_62420

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 ^ k.val : ℤ) % 3 ≠ (k.val ^ 7 : ℤ) % 3) ∧ 
  (7 ^ n.val : ℤ) % 3 = (n.val ^ 7 : ℤ) % 3 → 
  n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l624_62420


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l624_62484

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 120) (h₃ : a₃ = 27/16) :
  ∃ b : ℝ, b > 0 ∧ b * b = a₁ * a₃ ∧ b = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l624_62484


namespace NUMINAMATH_CALUDE_probability_heart_spade_club_standard_deck_l624_62496

/-- A standard deck of cards. -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (spades : Nat)
  (clubs : Nat)
  (diamonds : Nat)

/-- The probability of drawing a heart, then a spade, then a club from a standard deck. -/
def probability_heart_spade_club (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total *
  (d.spades : ℚ) / (d.total - 1) *
  (d.clubs : ℚ) / (d.total - 2)

/-- Theorem stating the probability of drawing a heart, then a spade, then a club
    from a standard 52-card deck. -/
theorem probability_heart_spade_club_standard_deck :
  let standard_deck : Deck := ⟨52, 13, 13, 13, 13⟩
  probability_heart_spade_club standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_spade_club_standard_deck_l624_62496


namespace NUMINAMATH_CALUDE_find_M_l624_62449

theorem find_M : ∃ M : ℚ, (25 / 100) * M = (35 / 100) * 4025 ∧ M = 5635 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l624_62449


namespace NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l624_62473

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum square side length needed to fit two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r2.width) (r1.height + r2.height)

/-- The theorem stating the smallest possible square area -/
theorem smallest_square_area_for_rectangles :
  let r1 : Rectangle := ⟨2, 5⟩
  let r2 : Rectangle := ⟨4, 3⟩
  (minSquareSide r1 r2) ^ 2 = 36 := by sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_rectangles_l624_62473


namespace NUMINAMATH_CALUDE_base_conversion_l624_62493

/-- Given that 26 in decimal is equal to 32 in base k, prove that k = 8 -/
theorem base_conversion (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l624_62493


namespace NUMINAMATH_CALUDE_subset_implies_a_bound_l624_62454

theorem subset_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 ≤ 0 → 1/(x-3) < a) → a > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_bound_l624_62454


namespace NUMINAMATH_CALUDE_joey_age_digit_sum_l624_62446

def joey_age_sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem joey_age_digit_sum :
  ∃ (chloe_age : ℕ) (joey_age : ℕ),
    joey_age = chloe_age + 2 ∧
    chloe_age > 2 ∧
    chloe_age % 5 = 0 ∧
    joey_age % 5 = 0 ∧
    ∀ k : ℕ, k < chloe_age → k % 5 ≠ 0 ∧
    joey_age_sum_of_digits joey_age = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_joey_age_digit_sum_l624_62446


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l624_62465

/-- Given a canoe rowing upstream at 9 km/hr and a stream speed of 1.5 km/hr,
    the speed of the canoe when rowing downstream is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed : ℝ := 9
  let stream_speed : ℝ := 1.5
  let canoe_speed : ℝ := upstream_speed + stream_speed
  let downstream_speed : ℝ := canoe_speed + stream_speed
  downstream_speed = 12 := by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l624_62465


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l624_62425

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  /-- The side length of the equilateral triangle -/
  t : ℝ
  /-- The distance from vertex D to point Q -/
  DQ : ℝ
  /-- The distance from vertex E to point Q -/
  EQ : ℝ
  /-- The distance from vertex F to point Q -/
  FQ : ℝ
  /-- The triangle is equilateral -/
  equilateral : t > 0
  /-- The point Q is inside the triangle -/
  interior : DQ > 0 ∧ EQ > 0 ∧ FQ > 0
  /-- The distances from Q to the vertices -/
  distances : DQ = 2 ∧ EQ = Real.sqrt 5 ∧ FQ = 3

/-- The theorem stating that the side length of the special triangle is 2√3 -/
theorem special_triangle_side_length (T : SpecialTriangle) : T.t = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l624_62425


namespace NUMINAMATH_CALUDE_range_of_m_l624_62483

/-- The proposition p: x^2 - 7x + 10 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 ≤ 0

/-- The proposition q: m ≤ x ≤ m + 1 -/
def q (m x : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

/-- q is a sufficient condition for p -/
def q_sufficient_for_p (m : ℝ) : Prop := ∀ x, q m x → p x

theorem range_of_m (m : ℝ) : 
  q_sufficient_for_p m → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l624_62483


namespace NUMINAMATH_CALUDE_james_muffins_count_l624_62401

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The factor by which James baked more muffins than Arthur -/
def james_factor : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_factor

theorem james_muffins_count : james_muffins = 1380 := by
  sorry

end NUMINAMATH_CALUDE_james_muffins_count_l624_62401


namespace NUMINAMATH_CALUDE_four_vertex_cycle_exists_l624_62414

/-- A graph with n ≥ 4 vertices where each vertex has degree between 1 and n-2 (inclusive) --/
structure CompanyGraph (n : ℕ) where
  (vertices : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (h1 : n ≥ 4)
  (h2 : ∀ v ∈ vertices, 1 ≤ (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card)
  (h3 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ n - 2)
  (h4 : ∀ e ∈ edges, e.1 ∈ vertices ∧ e.2 ∈ vertices)
  (h5 : ∀ e ∈ edges, (e.2, e.1) ∈ edges)  -- Knowledge is mutual

/-- A cycle of four vertices in the graph --/
structure FourVertexCycle (n : ℕ) (G : CompanyGraph n) where
  (v1 v2 v3 v4 : Fin n)
  (h1 : v1 ∈ G.vertices ∧ v2 ∈ G.vertices ∧ v3 ∈ G.vertices ∧ v4 ∈ G.vertices)
  (h2 : (v1, v2) ∈ G.edges ∧ (v2, v3) ∈ G.edges ∧ (v3, v4) ∈ G.edges ∧ (v4, v1) ∈ G.edges)
  (h3 : (v1, v3) ∉ G.edges ∧ (v2, v4) ∉ G.edges)

/-- The main theorem --/
theorem four_vertex_cycle_exists (n : ℕ) (G : CompanyGraph n) : 
  ∃ c : FourVertexCycle n G, True :=
sorry

end NUMINAMATH_CALUDE_four_vertex_cycle_exists_l624_62414


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l624_62486

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + k ≠ 0) → k > 25/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l624_62486


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l624_62499

/-- Given two complementary angles with measures in the ratio of 3:1, 
    their positive difference is 45 degrees. -/
theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45 degrees
:= by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l624_62499


namespace NUMINAMATH_CALUDE_project_hours_difference_l624_62498

/-- Given a project with three contributors (Pat, Kate, and Mark) with specific charging ratios,
    prove the difference in hours charged between Mark and Kate. -/
theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 198 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 110 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l624_62498


namespace NUMINAMATH_CALUDE_circle_radius_proof_l624_62438

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 5^2) →
  (A₂ = (A₁ + A₂) / 2) →
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = 5 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l624_62438


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l624_62453

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line point.1 (f point.1) ∧
  ∀ x : ℝ, (tangent_line x (f point.1 + (x - point.1) * (3 * point.1^2 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l624_62453


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l624_62410

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - m := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l624_62410


namespace NUMINAMATH_CALUDE_john_max_books_l624_62463

def john_money : ℕ := 2545  -- in cents
def initial_book_price : ℕ := 285  -- in cents
def discounted_book_price : ℕ := 250  -- in cents
def discount_threshold : ℕ := 10

def max_books_buyable (money : ℕ) (price : ℕ) (discount_price : ℕ) (threshold : ℕ) : ℕ :=
  if money < threshold * price then
    money / price
  else
    threshold + (money - threshold * price) / discount_price

theorem john_max_books :
  max_books_buyable john_money initial_book_price discounted_book_price discount_threshold = 8 :=
sorry

end NUMINAMATH_CALUDE_john_max_books_l624_62463


namespace NUMINAMATH_CALUDE_g_one_half_l624_62406

/-- Given a function g : ℝ → ℝ satisfying certain properties, prove that g(1/2) = 1/2 -/
theorem g_one_half (g : ℝ → ℝ) 
  (h1 : g 2 = 2)
  (h2 : ∀ x y : ℝ, g (x * y + g x) = y * g x + g x) : 
  g (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_g_one_half_l624_62406


namespace NUMINAMATH_CALUDE_graces_nickels_l624_62400

theorem graces_nickels (dimes : ℕ) (nickels : ℕ) : 
  dimes = 10 →
  dimes * 10 + nickels * 5 = 150 →
  nickels = 10 := by
sorry

end NUMINAMATH_CALUDE_graces_nickels_l624_62400


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l624_62423

def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def K_count : ℕ := 1
def Br_count : ℕ := 1
def O_count : ℕ := 3

def molecular_weight : ℝ :=
  K_count * atomic_weight_K +
  Br_count * atomic_weight_Br +
  O_count * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 167.00 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l624_62423


namespace NUMINAMATH_CALUDE_sine_equality_solution_l624_62426

theorem sine_equality_solution (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
sorry

end NUMINAMATH_CALUDE_sine_equality_solution_l624_62426


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l624_62457

def garden_problem (initial_flowering : ℕ) (initial_fruiting_multiplier : ℕ)
  (bought_flowering : ℕ) (bought_fruiting : ℕ)
  (given_flowering : ℕ) (given_fruiting : ℕ) : ℕ :=
  let initial_fruiting := initial_flowering * initial_fruiting_multiplier
  let after_buying_flowering := initial_flowering + bought_flowering
  let after_buying_fruiting := initial_fruiting + bought_fruiting
  let final_flowering := after_buying_flowering - given_flowering
  let final_fruiting := after_buying_fruiting - given_fruiting
  final_flowering + final_fruiting

theorem roxy_garden_plants :
  garden_problem 7 2 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l624_62457


namespace NUMINAMATH_CALUDE_death_rate_is_three_per_two_seconds_l624_62482

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per second -/
def birth_rate : ℚ := 3

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 129600

/-- Calculates the death rate in people per second -/
def death_rate : ℚ := birth_rate - (net_increase_per_day : ℚ) / seconds_per_day

/-- Theorem stating that the death rate is 3 people every two seconds -/
theorem death_rate_is_three_per_two_seconds : 
  death_rate * 2 = 3 := by sorry

end NUMINAMATH_CALUDE_death_rate_is_three_per_two_seconds_l624_62482


namespace NUMINAMATH_CALUDE_square_side_length_l624_62413

theorem square_side_length (diagonal_inches : ℝ) (h : diagonal_inches = 2 * Real.sqrt 2) :
  let diagonal_feet := diagonal_inches / 12
  let side_feet := diagonal_feet / Real.sqrt 2
  side_feet = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l624_62413


namespace NUMINAMATH_CALUDE_friday_increase_is_forty_percent_l624_62492

/-- Represents the library borrowing scenario for Krystian --/
structure LibraryBorrowing where
  dailyAverage : ℕ
  weeklyTotal : ℕ
  workdays : ℕ

/-- Calculates the percentage increase of Friday's borrowing compared to the daily average --/
def fridayPercentageIncrease (lb : LibraryBorrowing) : ℚ :=
  let fridayBorrowing := lb.weeklyTotal - (lb.workdays - 1) * lb.dailyAverage
  let increase := fridayBorrowing - lb.dailyAverage
  (increase : ℚ) / lb.dailyAverage * 100

/-- Theorem stating that the percentage increase on Friday is 40% --/
theorem friday_increase_is_forty_percent (lb : LibraryBorrowing) 
    (h1 : lb.dailyAverage = 40)
    (h2 : lb.weeklyTotal = 216)
    (h3 : lb.workdays = 5) : 
  fridayPercentageIncrease lb = 40 := by
  sorry

end NUMINAMATH_CALUDE_friday_increase_is_forty_percent_l624_62492


namespace NUMINAMATH_CALUDE_unique_integer_solution_l624_62461

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2*n^2 + m^2 + n^2 + 6*m*n → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l624_62461


namespace NUMINAMATH_CALUDE_problem_solution_l624_62485

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 5 + 1520 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l624_62485


namespace NUMINAMATH_CALUDE_power_calculation_l624_62494

theorem power_calculation : 
  ((18^13 * 18^11)^2 / 6^8) * 3^4 = 2^40 * 3^92 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l624_62494


namespace NUMINAMATH_CALUDE_stratified_sampling_difference_l624_62475

theorem stratified_sampling_difference (total_male : Nat) (total_female : Nat) (sample_size : Nat) : 
  total_male = 56 → 
  total_female = 42 → 
  sample_size = 28 → 
  (sample_size : ℚ) / ((total_male + total_female) : ℚ) = 2 / 7 → 
  (total_male : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) - 
  (total_female : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) = 4 := by
  sorry

#check stratified_sampling_difference

end NUMINAMATH_CALUDE_stratified_sampling_difference_l624_62475


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l624_62476

/-- The probability of picking two red balls from a bag containing 7 red, 5 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 7) (h2 : blue = 5) (h3 : green = 4) :
  let total := red + blue + green
  (red / total) * ((red - 1) / (total - 1)) = 7 / 40 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l624_62476


namespace NUMINAMATH_CALUDE_ln_power_rational_l624_62466

theorem ln_power_rational (f : ℝ) (r : ℚ) (hf : f > 0) :
  Real.log (f ^ (r : ℝ)) = r * Real.log f := by
  sorry

end NUMINAMATH_CALUDE_ln_power_rational_l624_62466


namespace NUMINAMATH_CALUDE_non_sunday_average_is_240_l624_62487

/-- Represents the average number of visitors to a library on different days. -/
structure LibraryVisitors where
  sunday : ℕ
  otherDays : ℕ
  monthlyAverage : ℕ

/-- Calculates the average number of visitors on non-Sunday days given the conditions. -/
def calculateNonSundayAverage (v : LibraryVisitors) : ℕ :=
  ((v.monthlyAverage * 30) - (v.sunday * 5)) / 25

/-- Theorem stating that under the given conditions, the average number of visitors
    on non-Sunday days is 240. -/
theorem non_sunday_average_is_240 (v : LibraryVisitors)
  (h1 : v.sunday = 600)
  (h2 : v.monthlyAverage = 300) :
  calculateNonSundayAverage v = 240 := by
  sorry

#eval calculateNonSundayAverage ⟨600, 0, 300⟩

end NUMINAMATH_CALUDE_non_sunday_average_is_240_l624_62487


namespace NUMINAMATH_CALUDE_solution_sum_l624_62489

theorem solution_sum (p q : ℝ) : 
  (2^2 - 2*p + 6 = 0) → 
  (2^2 + 6*2 - q = 0) → 
  p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l624_62489


namespace NUMINAMATH_CALUDE_relationship_a_ab_ab_squared_l624_62404

theorem relationship_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_relationship_a_ab_ab_squared_l624_62404


namespace NUMINAMATH_CALUDE_zhukov_birth_year_l624_62419

theorem zhukov_birth_year (total_years : ℕ) (years_diff : ℕ) (birth_year : ℕ) :
  total_years = 78 →
  years_diff = 70 →
  birth_year = 1900 - (total_years - years_diff) / 2 →
  birth_year = 1896 :=
by sorry

end NUMINAMATH_CALUDE_zhukov_birth_year_l624_62419


namespace NUMINAMATH_CALUDE_sara_sister_notebooks_l624_62462

def calculate_notebooks (initial : ℕ) (increase_percent : ℕ) (lost : ℕ) : ℕ :=
  let increased : ℕ := initial + initial * increase_percent / 100
  increased - lost

theorem sara_sister_notebooks : calculate_notebooks 4 150 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sara_sister_notebooks_l624_62462


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l624_62481

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l624_62481


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l624_62443

theorem square_minus_product_equals_one : 2002^2 - 2001 * 2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l624_62443


namespace NUMINAMATH_CALUDE_existence_of_n_consecutive_representable_l624_62440

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part 1: Existence of n such that n + S(n) = 1980
theorem existence_of_n : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part 2: For any m, either m or m+1 can be expressed as n + S(n)
theorem consecutive_representable (m : ℕ) : 
  (∃ n : ℕ, n + S n = m) ∨ (∃ n : ℕ, n + S n = m + 1) := by sorry

end NUMINAMATH_CALUDE_existence_of_n_consecutive_representable_l624_62440


namespace NUMINAMATH_CALUDE_negation_equivalence_l624_62474

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l624_62474


namespace NUMINAMATH_CALUDE_volume_of_five_cubes_l624_62469

/-- The volume of a solid formed by adjacent cubes -/
def volume_of_adjacent_cubes (n : ℕ) (side_length : ℝ) : ℝ :=
  n * (side_length ^ 3)

/-- Theorem: The volume of a solid formed by five adjacent cubes with side length 5 cm is 625 cm³ -/
theorem volume_of_five_cubes : volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_five_cubes_l624_62469


namespace NUMINAMATH_CALUDE_ending_number_proof_l624_62430

theorem ending_number_proof (n : ℕ) : 
  (n > 100) ∧ 
  (∃ (count : ℕ), count = 33 ∧ 
    (∀ k : ℕ, 100 < k ∧ k ≤ n ∧ k % 3 = 0 → 
      ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (count : ℕ), count = 33 ∧ 
      (∀ k : ℕ, 100 < k ∧ k ≤ m ∧ k % 3 = 0 → 
        ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i))) →
  n = 198 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l624_62430


namespace NUMINAMATH_CALUDE_revenue_loss_l624_62442

/-- Represents the types of tickets sold in the theater. -/
inductive TicketType
  | GeneralRegular
  | GeneralVIP
  | ChildRegular
  | ChildVIP
  | SeniorRegular
  | SeniorVIP
  | VeteranRegular
  | VeteranVIP

/-- Calculates the revenue for a given ticket type. -/
def ticketRevenue (t : TicketType) : ℚ :=
  match t with
  | .GeneralRegular => 10
  | .GeneralVIP => 15
  | .ChildRegular => 6
  | .ChildVIP => 11
  | .SeniorRegular => 8
  | .SeniorVIP => 13
  | .VeteranRegular => 8
  | .VeteranVIP => 13

/-- Represents the theater's seating and pricing structure. -/
structure Theater where
  regularSeats : ℕ
  vipSeats : ℕ
  regularPrice : ℚ
  vipSurcharge : ℚ

/-- Calculates the potential revenue if all seats were sold at full price. -/
def potentialRevenue (t : Theater) : ℚ :=
  t.regularSeats * t.regularPrice + t.vipSeats * (t.regularPrice + t.vipSurcharge)

/-- Represents the actual sales for the night. -/
structure ActualSales where
  generalRegular : ℕ
  generalVIP : ℕ
  childRegular : ℕ
  childVIP : ℕ
  seniorRegular : ℕ
  seniorVIP : ℕ
  veteranRegular : ℕ
  veteranVIP : ℕ

/-- Calculates the actual revenue from the given sales. -/
def actualRevenue (s : ActualSales) : ℚ :=
  s.generalRegular * ticketRevenue .GeneralRegular +
  s.generalVIP * ticketRevenue .GeneralVIP +
  s.childRegular * ticketRevenue .ChildRegular +
  s.childVIP * ticketRevenue .ChildVIP +
  s.seniorRegular * ticketRevenue .SeniorRegular +
  s.seniorVIP * ticketRevenue .SeniorVIP +
  s.veteranRegular * ticketRevenue .VeteranRegular +
  s.veteranVIP * ticketRevenue .VeteranVIP

theorem revenue_loss (t : Theater) (s : ActualSales) :
    t.regularSeats = 40 ∧
    t.vipSeats = 10 ∧
    t.regularPrice = 10 ∧
    t.vipSurcharge = 5 ∧
    s.generalRegular = 12 ∧
    s.generalVIP = 6 ∧
    s.childRegular = 3 ∧
    s.childVIP = 1 ∧
    s.seniorRegular = 4 ∧
    s.seniorVIP = 2 ∧
    s.veteranRegular = 2 ∧
    s.veteranVIP = 1 →
    potentialRevenue t - actualRevenue s = 224 := by
  sorry

#eval potentialRevenue { regularSeats := 40, vipSeats := 10, regularPrice := 10, vipSurcharge := 5 }
#eval actualRevenue { generalRegular := 12, generalVIP := 6, childRegular := 3, childVIP := 1,
                      seniorRegular := 4, seniorVIP := 2, veteranRegular := 2, veteranVIP := 1 }

end NUMINAMATH_CALUDE_revenue_loss_l624_62442


namespace NUMINAMATH_CALUDE_iced_tea_consumption_iced_tea_consumption_is_198_l624_62433

theorem iced_tea_consumption : ℝ → Prop :=
  fun total_consumption =>
    ∃ (rob_size : ℝ),
      let mary_size : ℝ := 1.75 * rob_size
      let rob_remaining : ℝ := (1/3) * rob_size
      let mary_remaining : ℝ := (1/3) * mary_size
      let mary_share : ℝ := (1/4) * mary_remaining + 3
      let rob_total : ℝ := (2/3) * rob_size + mary_share
      let mary_total : ℝ := (2/3) * mary_size - mary_share
      rob_total = mary_total ∧
      total_consumption = rob_size + mary_size ∧
      total_consumption = 198

theorem iced_tea_consumption_is_198 : iced_tea_consumption 198 := by
  sorry

end NUMINAMATH_CALUDE_iced_tea_consumption_iced_tea_consumption_is_198_l624_62433


namespace NUMINAMATH_CALUDE_gcd_282_470_l624_62412

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l624_62412


namespace NUMINAMATH_CALUDE_flag_distance_l624_62434

theorem flag_distance (road_length : ℝ) (total_flags : ℕ) (h1 : road_length = 191.8) (h2 : total_flags = 58) :
  let intervals := total_flags / 2 - 1
  road_length / intervals = 6.85 := by
sorry

end NUMINAMATH_CALUDE_flag_distance_l624_62434


namespace NUMINAMATH_CALUDE_rationalize_denominator_l624_62428

theorem rationalize_denominator : 
  (36 : ℝ) / (12 : ℝ)^(1/3) = 3 * (144 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l624_62428


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l624_62409

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (8 - 5 * x > 22) → x ≤ -3 ∧ 8 - 5 * (-3) > 22 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l624_62409


namespace NUMINAMATH_CALUDE_mean_height_of_basketball_team_l624_62490

def heights : List ℝ := [48, 50, 51, 54, 56, 57, 57, 59, 60, 63, 64, 65, 67, 69, 69, 71, 72, 74]

theorem mean_height_of_basketball_team : 
  (heights.sum / heights.length : ℝ) = 61.444444444444445 := by sorry

end NUMINAMATH_CALUDE_mean_height_of_basketball_team_l624_62490


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l624_62472

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 12*n + 28 ≤ 0 ∧ 
  ∀ (m : ℤ), m^2 - 12*m + 28 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l624_62472


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l624_62452

-- Define the quadratic function
def f (x b c : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  f 1 b c = 0 →
  f 3 b c = 0 →
  f (-1) b c = 8 ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = 3) ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≥ -1) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l624_62452


namespace NUMINAMATH_CALUDE_range_of_a_l624_62427

theorem range_of_a (a x y z : ℝ) 
  (h1 : |a - 2| ≤ x^2 + 2*y^2 + 3*z^2)
  (h2 : x + y + z = 1) :
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l624_62427


namespace NUMINAMATH_CALUDE_max_disks_in_rectangle_l624_62459

/-- The maximum number of circular disks that can be cut from a rectangular sheet. -/
def max_disks (rect_width rect_height disk_diameter : ℝ) : ℕ :=
  32

/-- Theorem stating the maximum number of 5 cm diameter circular disks 
    that can be cut from a 9 × 100 cm rectangular sheet. -/
theorem max_disks_in_rectangle : 
  max_disks 9 100 5 = 32 := by sorry

end NUMINAMATH_CALUDE_max_disks_in_rectangle_l624_62459


namespace NUMINAMATH_CALUDE_complement_of_S_in_U_l624_62488

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define the set S
def S : Set Nat := {1, 3}

-- Theorem statement
theorem complement_of_S_in_U :
  U \ S = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_S_in_U_l624_62488


namespace NUMINAMATH_CALUDE_range_of_a_l624_62455

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (1 - a)^x < (1 - a)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (0 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l624_62455


namespace NUMINAMATH_CALUDE_third_score_calculation_l624_62402

theorem third_score_calculation (score1 score2 score4 : ℕ) (average : ℚ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  average = 75 →
  ∃ score3 : ℕ, (score1 + score2 + score3 + score4) / 4 = average ∧ score3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_third_score_calculation_l624_62402


namespace NUMINAMATH_CALUDE_largest_root_is_two_l624_62470

/-- A polynomial of degree 6 with specific coefficients and three parameters -/
def P (a b c : ℝ) (x : ℝ) : ℝ :=
  x^6 - 6*x^5 + 17*x^4 + 6*x^3 + a*x^2 - b*x - c

/-- The theorem stating that if P has exactly three distinct double roots, the largest root is 2 -/
theorem largest_root_is_two (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    (∀ x : ℝ, P a b c x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    (∀ x : ℝ, (x - r₁)^2 * (x - r₂)^2 * (x - r₃)^2 = P a b c x)) →
  (∃ r : ℝ, r = 2 ∧ P a b c r = 0 ∧ ∀ s : ℝ, P a b c s = 0 → s ≤ r) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_two_l624_62470


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l624_62407

theorem no_solution_in_interval : ¬∃ x : ℝ, 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l624_62407


namespace NUMINAMATH_CALUDE_house_numbering_proof_l624_62422

theorem house_numbering_proof :
  (2 * 169^2 - 1 = 239^2) ∧ (2 * (288^2 + 288) = 408^2) := by
  sorry

end NUMINAMATH_CALUDE_house_numbering_proof_l624_62422


namespace NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l624_62471

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem area_regular_octagon_in_circle (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 256 * Real.pi → 
  octagon_area = 8 * (1/2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) → 
  octagon_area = 512 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l624_62471


namespace NUMINAMATH_CALUDE_female_population_l624_62480

theorem female_population (total_population : ℕ) (num_parts : ℕ) (female_parts : ℕ) : 
  total_population = 720 →
  num_parts = 4 →
  female_parts = 2 →
  (total_population / num_parts) * female_parts = 360 :=
by sorry

end NUMINAMATH_CALUDE_female_population_l624_62480


namespace NUMINAMATH_CALUDE_emily_earnings_is_twenty_l624_62408

/-- The number of chocolate bars in a box -/
def total_bars : ℕ := 8

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℕ := 4

/-- The number of unsold bars -/
def unsold_bars : ℕ := 3

/-- Emily's earnings from selling chocolate bars -/
def emily_earnings : ℕ := (total_bars - unsold_bars) * cost_per_bar

/-- Theorem stating that Emily's earnings are $20 -/
theorem emily_earnings_is_twenty : emily_earnings = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_earnings_is_twenty_l624_62408


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l624_62437

def scores : List Nat := [91, 68, 59, 99, 82, 88, 86, 79, 72, 60, 87, 85, 83, 76, 81, 93, 65, 89, 78, 74]

def is_grade_b (score : Nat) : Bool :=
  83 ≤ score ∧ score ≤ 92

def count_grade_b (scores : List Nat) : Nat :=
  scores.filter is_grade_b |>.length

theorem percentage_of_b_grades (scores : List Nat) :
  scores.length = 20 →
  (count_grade_b scores : Rat) / scores.length * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l624_62437


namespace NUMINAMATH_CALUDE_can_form_triangle_l624_62415

theorem can_form_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 12) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l624_62415


namespace NUMINAMATH_CALUDE_k_h_symmetry_l624_62448

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_symmetry (h_def : ∀ x, h x = 4 * x^2 - 12) 
                     (k_h_3 : k (h 3) = 16) : 
  k (h (-3)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_k_h_symmetry_l624_62448


namespace NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l624_62447

theorem inverse_false_implies_negation_false (p : Prop) :
  (p → False) → ¬p = False :=
by sorry

end NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l624_62447


namespace NUMINAMATH_CALUDE_problem_solution_l624_62418

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = x^2 + 9)
  (h3 : ∀ x, g x = x^2 - 3)
  (h4 : f (g a) = 9) :
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l624_62418


namespace NUMINAMATH_CALUDE_college_students_count_l624_62405

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l624_62405


namespace NUMINAMATH_CALUDE_dolphin_count_dolphin_count_proof_l624_62403

theorem dolphin_count : ℕ → Prop :=
  fun total_dolphins =>
    let fully_trained := total_dolphins / 4
    let remaining := total_dolphins - fully_trained
    let in_training := (2 * remaining) / 3
    let untrained := remaining - in_training
    (fully_trained = total_dolphins / 4) ∧
    (in_training = (2 * remaining) / 3) ∧
    (untrained = 5) →
    total_dolphins = 20

-- The proof goes here
theorem dolphin_count_proof : dolphin_count 20 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_dolphin_count_proof_l624_62403


namespace NUMINAMATH_CALUDE_price_difference_pants_belt_l624_62451

/-- Given the total cost of pants and belt, and the price of pants, 
    calculate the difference in price between the belt and the pants. -/
theorem price_difference_pants_belt 
  (total_cost : ℝ) 
  (pants_price : ℝ) 
  (h1 : total_cost = 70.93)
  (h2 : pants_price = 34.00)
  (h3 : pants_price < total_cost - pants_price) :
  total_cost - pants_price - pants_price = 2.93 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_pants_belt_l624_62451


namespace NUMINAMATH_CALUDE_no_root_in_interval_l624_62477

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  ∀ x ∈ Set.Ioo 2 3, f x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_root_in_interval_l624_62477


namespace NUMINAMATH_CALUDE_rope_remaining_lengths_l624_62441

/-- Calculates the remaining lengths of two ropes after giving away portions. -/
theorem rope_remaining_lengths (x y : ℝ) (p q : ℝ) : 
  p = 0.40 * x ∧ q = 0.5625 * y := by
  sorry

#check rope_remaining_lengths

end NUMINAMATH_CALUDE_rope_remaining_lengths_l624_62441


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l624_62444

/-- A quadratic polynomial with two distinct real roots -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, P x = a * x^2 + b * x + c
  has_distinct_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

/-- The inequality condition for the polynomial -/
def SatisfiesInequality (P : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

/-- The main theorem -/
theorem quadratic_polynomial_negative_root (p : QuadraticPolynomial) 
    (h : SatisfiesInequality p.P) : 
    ∃ (x : ℝ), x < 0 ∧ p.P x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l624_62444


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l624_62456

open Real

-- Define the function f and its properties
theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (h1 : ∀ x > 0, HasDerivAt f (f' x) x)
  (h2 : ∀ x > 0, x * f' x + f x = (log x) / x)
  (h3 : f (exp 1) = (exp 1)⁻¹) :
  {x : ℝ | f (x + 1) - f ((exp 1) + 1) > x - (exp 1)} = Set.Ioo (-1) (exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l624_62456


namespace NUMINAMATH_CALUDE_check_problem_l624_62421

/-- The check problem -/
theorem check_problem (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 2058 →
  (10 ≤ x ∧ x ≤ 78) ∧ y = x + 21 :=
by sorry

end NUMINAMATH_CALUDE_check_problem_l624_62421


namespace NUMINAMATH_CALUDE_rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l624_62424

/-- Calculate the number of eggs needed for the Rotary Club Omelet Breakfast -/
theorem rotary_club_omelet_eggs : ℕ :=
  let small_children := 53
  let older_children := 35
  let adults := 75
  let seniors := 37
  let small_children_omelets := 0.5
  let older_children_omelets := 1
  let adults_omelets := 2
  let seniors_omelets := 1.5
  let extra_omelets := 25
  let eggs_per_omelet := 2

  let total_omelets := small_children * small_children_omelets +
                       older_children * older_children_omelets +
                       adults * adults_omelets +
                       seniors * seniors_omelets +
                       extra_omelets

  let total_eggs := total_omelets * eggs_per_omelet

  584

theorem rotary_club_omelet_eggs_proof : rotary_club_omelet_eggs = 584 := by
  sorry

end NUMINAMATH_CALUDE_rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l624_62424
