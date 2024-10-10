import Mathlib

namespace apples_per_pie_l3960_396095

theorem apples_per_pie (total_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  total_apples = 51 →
  handed_out = 41 →
  num_pies = 2 →
  (total_apples - handed_out) / num_pies = 5 := by
sorry

end apples_per_pie_l3960_396095


namespace quadratic_roots_product_l3960_396020

theorem quadratic_roots_product (x : ℝ) : 
  (x - 4) * (2 * x + 10) = x^2 - 15 * x + 56 → 
  ∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ (c / a) + 6 = -90 :=
by sorry

end quadratic_roots_product_l3960_396020


namespace abs_greater_than_two_necessary_not_sufficient_l3960_396067

theorem abs_greater_than_two_necessary_not_sufficient :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end abs_greater_than_two_necessary_not_sufficient_l3960_396067


namespace arithmetic_sequence_sum_l3960_396042

/-- Given an arithmetic sequence {a_n} where a_5 + a_7 = 2, 
    prove that a_4 + 2a_6 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 5 + a 7 = 2) →                                -- given condition
  (a 4 + 2 * a 6 + a 8 = 4) :=                     -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l3960_396042


namespace profit_division_time_l3960_396052

/-- Represents the partnership problem with given conditions -/
def PartnershipProblem (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) : Prop :=
  -- Initial ratio of shares
  initial_ratio_p + initial_ratio_q + initial_ratio_r = 1 ∧
  -- p withdraws half of the capital after two months
  withdrawal_time = 2 ∧
  withdrawal_fraction = 1/2 ∧
  -- Given total profit and r's share
  total_profit > 0 ∧
  r_profit > 0 ∧
  r_profit < total_profit

/-- Theorem stating the number of months after which the profit was divided -/
theorem profit_division_time (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) :
  PartnershipProblem initial_ratio_p initial_ratio_q initial_ratio_r
    withdrawal_time withdrawal_fraction total_profit r_profit →
  ∃ (n : ℕ), n = 12 := by
  sorry

end profit_division_time_l3960_396052


namespace pool_capacity_l3960_396064

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 :=
by
  sorry

end pool_capacity_l3960_396064


namespace symmetry_yoz_plane_l3960_396018

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yoz plane in 3D space -/
def yozPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yoz plane -/
def symmetricPointYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

theorem symmetry_yoz_plane :
  let p : Point3D := ⟨2, 3, 5⟩
  symmetricPointYOZ p = ⟨-2, 3, 5⟩ := by
  sorry

end symmetry_yoz_plane_l3960_396018


namespace toys_in_box_time_l3960_396009

/-- The time in minutes required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (mom_puts_in : ℕ) (mia_takes_out : ℕ) (cycle_time : ℕ) : ℕ :=
  let net_gain_per_cycle := mom_puts_in - mia_takes_out
  let cycles_after_first_minute := (total_toys - 2 * mom_puts_in) / net_gain_per_cycle
  let total_seconds := (cycles_after_first_minute + 2) * cycle_time
  total_seconds / 60

/-- Theorem stating that under the given conditions, it takes 22 minutes to put all toys in the box -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 4 3 30 = 22 := by
  sorry

end toys_in_box_time_l3960_396009


namespace puzzle_completion_time_l3960_396005

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_set : ℕ) (minutes_per_set : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let num_sets := total_pieces / pieces_per_set
  num_sets * minutes_per_set

/-- Theorem stating that completing 2 puzzles of 2000 pieces each, at a rate of 100 pieces per 10 minutes, takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

#eval time_to_complete_puzzles 2 2000 100 10

end puzzle_completion_time_l3960_396005


namespace equation_condition_for_x_equals_4_l3960_396003

theorem equation_condition_for_x_equals_4 :
  (∃ x : ℝ, x^2 - 3*x - 4 = 0) ∧
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) :=
by sorry

end equation_condition_for_x_equals_4_l3960_396003


namespace exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l3960_396023

/-- The set A defined by the quadratic equation (a-1)x^2 + 3x - 2 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | (a - 1) * x^2 + 3 * x - 2 = 0}

/-- The theorem stating the existence of 'a' values for which A has exactly two subsets -/
theorem exist_a_with_two_subsets :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  (∀ (S : Set ℝ), S ⊆ A a₁ → (S = ∅ ∨ S = A a₁)) ∧
  (∀ (S : Set ℝ), S ⊆ A a₂ → (S = ∅ ∨ S = A a₂)) ∧
  a₁ = 1 ∧ a₂ = -1/8 := by
  sorry

/-- The theorem stating that for a = 1, A = {2/3} -/
theorem a_eq_one_implies_A_eq_two_thirds :
  A 1 = {2/3} := by
  sorry

/-- The theorem stating that for a = -1/8, A = {4/3} -/
theorem a_eq_neg_one_eighth_implies_A_eq_four_thirds :
  A (-1/8) = {4/3} := by
  sorry

end exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l3960_396023


namespace lcm_inequality_l3960_396038

theorem lcm_inequality (k m n : ℕ) : 
  (Nat.lcm k m) * (Nat.lcm m n) * (Nat.lcm n k) ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
sorry

end lcm_inequality_l3960_396038


namespace zoe_calorie_intake_l3960_396081

-- Define the quantities
def strawberries : ℕ := 12
def yogurt_ounces : ℕ := 6
def calories_per_strawberry : ℕ := 4
def calories_per_yogurt_ounce : ℕ := 17

-- Define the total calories
def total_calories : ℕ := strawberries * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

-- Theorem statement
theorem zoe_calorie_intake : total_calories = 150 := by
  sorry

end zoe_calorie_intake_l3960_396081


namespace dog_distribution_theorem_l3960_396062

/-- The number of ways to distribute 12 dogs into three groups -/
def dog_distribution_ways : ℕ :=
  (Nat.choose 11 3) * (Nat.choose 7 4)

/-- Theorem stating the number of ways to distribute the dogs -/
theorem dog_distribution_theorem : dog_distribution_ways = 5775 := by
  sorry

end dog_distribution_theorem_l3960_396062


namespace jen_current_age_l3960_396054

/-- Jen's age when her son was born -/
def jen_age_at_birth : ℕ := 25

/-- Relationship between Jen's age and her son's age -/
def jen_age_relation (son_age : ℕ) : ℕ := 3 * son_age - 7

/-- Theorem stating Jen's current age -/
theorem jen_current_age :
  ∃ (son_age : ℕ), jen_age_at_birth + son_age = jen_age_relation son_age ∧
                   jen_age_at_birth + son_age = 41 := by
  sorry

end jen_current_age_l3960_396054


namespace compare_powers_l3960_396080

theorem compare_powers : (4 ^ 12 : ℕ) < 9 ^ 8 ∧ 9 ^ 8 = 3 ^ 16 := by sorry

end compare_powers_l3960_396080


namespace cyclic_sum_inequality_l3960_396091

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end cyclic_sum_inequality_l3960_396091


namespace abc_product_l3960_396032

theorem abc_product (a b c : ℝ) 
  (eq1 : b + c = 16) 
  (eq2 : c + a = 17) 
  (eq3 : a + b = 18) : 
  a * b * c = 606.375 := by
sorry

end abc_product_l3960_396032


namespace bowTie_solution_l3960_396024

noncomputable def bowTie (a b : ℝ) : ℝ := a^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

theorem bowTie_solution (y : ℝ) : bowTie 3 y = 18 → y = 6 * Real.sqrt 2 ∨ y = -6 * Real.sqrt 2 := by
  sorry

end bowTie_solution_l3960_396024


namespace isosceles_triangle_condition_l3960_396069

theorem isosceles_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_condition : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end isosceles_triangle_condition_l3960_396069


namespace trapezium_side_length_l3960_396044

theorem trapezium_side_length 
  (a b h area : ℝ) 
  (h1 : b = 28) 
  (h2 : h = 21) 
  (h3 : area = 504) 
  (h4 : area = (a + b) * h / 2) : 
  a = 20 := by sorry

end trapezium_side_length_l3960_396044


namespace quadratic_sum_abc_l3960_396075

/-- Given a quadratic polynomial 12x^2 - 72x + 432, prove that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 333. -/
theorem quadratic_sum_abc (x : ℝ) : 
  ∃ (a b c : ℝ), (12 * x^2 - 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 333) := by
  sorry

end quadratic_sum_abc_l3960_396075


namespace jason_after_school_rate_l3960_396010

/-- Calculates Jason's hourly rate for after-school work --/
def after_school_rate (total_earnings weekly_hours saturday_hours saturday_rate : ℚ) : ℚ :=
  let saturday_earnings := saturday_hours * saturday_rate
  let after_school_earnings := total_earnings - saturday_earnings
  let after_school_hours := weekly_hours - saturday_hours
  after_school_earnings / after_school_hours

/-- Theorem stating Jason's after-school hourly rate --/
theorem jason_after_school_rate :
  after_school_rate 88 18 8 6 = 4 := by
  sorry

end jason_after_school_rate_l3960_396010


namespace equation_solutions_l3960_396056

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end equation_solutions_l3960_396056


namespace card_collection_solution_l3960_396088

/-- Represents the card collection problem --/
structure CardCollection where
  total_cards : Nat
  damaged_cards : Nat
  full_box_capacity : Nat
  damaged_box_capacity : Nat

/-- Calculates the number of cards in the last partially filled box of undamaged cards --/
def last_box_count (cc : CardCollection) : Nat :=
  (cc.total_cards - cc.damaged_cards) % cc.full_box_capacity

/-- Theorem stating the solution to the card collection problem --/
theorem card_collection_solution (cc : CardCollection) 
  (h1 : cc.total_cards = 120)
  (h2 : cc.damaged_cards = 18)
  (h3 : cc.full_box_capacity = 10)
  (h4 : cc.damaged_box_capacity = 5) :
  last_box_count cc = 2 := by
  sorry

#eval last_box_count { total_cards := 120, damaged_cards := 18, full_box_capacity := 10, damaged_box_capacity := 5 }

end card_collection_solution_l3960_396088


namespace expected_elderly_in_sample_l3960_396087

/-- Calculates the expected number of elderly individuals in a stratified sample -/
def expectedElderlyInSample (totalPopulation : ℕ) (elderlyPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (elderlyPopulation * sampleSize) / totalPopulation

/-- Theorem: Expected number of elderly individuals in the sample -/
theorem expected_elderly_in_sample :
  expectedElderlyInSample 165 22 15 = 2 := by
  sorry

end expected_elderly_in_sample_l3960_396087


namespace triangle_existence_l3960_396082

theorem triangle_existence (q : ℝ) (α β γ : ℝ) 
  (h_positive : q > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum : α + β + γ = Real.pi) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * Real.sin γ) / 2 = q^2 ∧
    Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = α ∧
    Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) = β ∧
    Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = γ :=
by sorry


end triangle_existence_l3960_396082


namespace als_initial_investment_l3960_396034

theorem als_initial_investment (a b c : ℝ) : 
  a + b + c = 2000 →
  3*a + 2*b + 2*c = 3500 →
  a = 500 := by
sorry

end als_initial_investment_l3960_396034


namespace parabola_focus_directrix_distance_l3960_396007

/-- The distance from the focus to the directrix of the parabola y = 1/2 * x^2 is 1 -/
theorem parabola_focus_directrix_distance : 
  let p : ℝ → ℝ := fun x ↦ (1/2) * x^2
  ∃ f d : ℝ, 
    (∀ x, p x = (1/4) * (x^2 + 1)) ∧  -- Standard form of parabola
    (f = 1/2) ∧                       -- y-coordinate of focus
    (d = -1/2) ∧                      -- y-coordinate of directrix
    (f - d = 1) :=                    -- Distance between focus and directrix
by sorry

end parabola_focus_directrix_distance_l3960_396007


namespace max_brownies_144_l3960_396033

/-- Represents the dimensions of a rectangular pan -/
structure PanDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in the pan -/
def interiorPieces (d : PanDimensions) : ℕ := (d.m - 2) * (d.n - 2)

/-- Calculates the number of perimeter pieces in the pan -/
def perimeterPieces (d : PanDimensions) : ℕ := 2 * d.m + 2 * d.n - 4

/-- Represents the condition that interior pieces are twice the perimeter pieces -/
def interiorTwicePerimeter (d : PanDimensions) : Prop :=
  interiorPieces d = 2 * perimeterPieces d

/-- The theorem stating that the maximum number of brownies is 144 -/
theorem max_brownies_144 :
  ∃ (d : PanDimensions), interiorTwicePerimeter d ∧
  (∀ (d' : PanDimensions), interiorTwicePerimeter d' → d.m * d.n ≥ d'.m * d'.n) ∧
  d.m * d.n = 144 := by
  sorry

end max_brownies_144_l3960_396033


namespace system_solvability_l3960_396013

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x = 6 / a - |y - a| ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ := {a | a ≤ -2/3 ∨ a > 0}

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ b x y, system a b x y) ↔ a ∈ valid_a_set :=
sorry

end system_solvability_l3960_396013


namespace fraction_difference_to_fifth_power_l3960_396079

theorem fraction_difference_to_fifth_power :
  (3/4 - 1/8)^5 = 3125/32768 := by sorry

end fraction_difference_to_fifth_power_l3960_396079


namespace quadratic_root_meaningful_l3960_396068

theorem quadratic_root_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y = 2 / Real.sqrt (3 + x)) ↔ x > -3 := by
sorry

end quadratic_root_meaningful_l3960_396068


namespace solution_set_not_empty_or_specific_interval_l3960_396099

theorem solution_set_not_empty_or_specific_interval (a : ℝ) :
  ∃ x : ℝ, a * (x - a) * (a * x + a) ≥ 0 ∧
  ¬(∀ x : ℝ, a * (x - a) * (a * x + a) < 0) ∧
  ¬(∀ x : ℝ, (a * (x - a) * (a * x + a) ≥ 0) ↔ (a ≤ x ∧ x ≤ -1)) :=
by sorry

end solution_set_not_empty_or_specific_interval_l3960_396099


namespace cookout_bun_packs_l3960_396035

/-- Calculate the number of bun packs needed for a cookout --/
theorem cookout_bun_packs 
  (total_friends : ℕ) 
  (burgers_per_guest : ℕ) 
  (non_meat_eaters : ℕ) 
  (no_bread_eaters : ℕ) 
  (gluten_free_friends : ℕ) 
  (nut_allergy_friends : ℕ)
  (regular_buns_per_pack : ℕ) 
  (gluten_free_buns_per_pack : ℕ) 
  (nut_free_buns_per_pack : ℕ)
  (h1 : total_friends = 35)
  (h2 : burgers_per_guest = 3)
  (h3 : non_meat_eaters = 7)
  (h4 : no_bread_eaters = 4)
  (h5 : gluten_free_friends = 3)
  (h6 : nut_allergy_friends = 1)
  (h7 : regular_buns_per_pack = 15)
  (h8 : gluten_free_buns_per_pack = 6)
  (h9 : nut_free_buns_per_pack = 5) :
  (((total_friends - non_meat_eaters) * burgers_per_guest - no_bread_eaters * burgers_per_guest + regular_buns_per_pack - 1) / regular_buns_per_pack = 5) ∧ 
  ((gluten_free_friends * burgers_per_guest + gluten_free_buns_per_pack - 1) / gluten_free_buns_per_pack = 2) ∧
  ((nut_allergy_friends * burgers_per_guest + nut_free_buns_per_pack - 1) / nut_free_buns_per_pack = 1) :=
by sorry

end cookout_bun_packs_l3960_396035


namespace opposite_and_reciprocal_sum_l3960_396014

theorem opposite_and_reciprocal_sum (a b x y : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : x * y = 1)  -- x and y are reciprocals
  : 2 * (a + b) + (7 / 4) * x * y = 7 / 4 := by
  sorry

end opposite_and_reciprocal_sum_l3960_396014


namespace charlyn_visible_area_l3960_396092

/-- The area of the region visible to Charlyn during her walk around a square -/
def visible_area (square_side : ℝ) (visibility_range : ℝ) : ℝ :=
  let inner_square_side := square_side - 2 * visibility_range
  let inner_area := inner_square_side ^ 2
  let outer_rectangles_area := 4 * (square_side * visibility_range)
  let corner_squares_area := 4 * (visibility_range ^ 2)
  (square_side ^ 2 - inner_area) + outer_rectangles_area + corner_squares_area

/-- Theorem stating that the visible area for Charlyn's walk is 160 km² -/
theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

#eval visible_area 10 2

end charlyn_visible_area_l3960_396092


namespace multiply_powers_same_base_l3960_396090

theorem multiply_powers_same_base (x : ℝ) : x * x^2 = x^3 := by
  sorry

end multiply_powers_same_base_l3960_396090


namespace bacon_suggestion_count_l3960_396070

theorem bacon_suggestion_count (mashed_potatoes : ℕ) (bacon : ℕ) : 
  mashed_potatoes = 479 → 
  bacon = mashed_potatoes + 10 → 
  bacon = 489 := by
sorry

end bacon_suggestion_count_l3960_396070


namespace arccos_sqrt2_over_2_l3960_396019

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by sorry

end arccos_sqrt2_over_2_l3960_396019


namespace article_count_l3960_396000

theorem article_count (cost_price selling_price : ℝ) (gain_percentage : ℝ) : 
  gain_percentage = 42.857142857142854 →
  50 * cost_price = 35 * selling_price →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  35 = 50 * (100 / (100 + gain_percentage)) :=
by sorry

end article_count_l3960_396000


namespace odd_decreasing_function_theorem_l3960_396089

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is decreasing if f(x₁) > f(x₂) for all x₁ < x₂ in its domain -/
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ > f x₂

theorem odd_decreasing_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : IsOdd f)
    (h_decreasing : IsDecreasing f (-1) 1)
    (h_condition : f (1 + a) + f (1 - a^2) < 0) :
    a ∈ Set.Ioo (-1) 0 := by
  sorry


end odd_decreasing_function_theorem_l3960_396089


namespace solution_count_is_49_l3960_396026

/-- The number of positive integer pairs (x, y) satisfying xy / (x + y) = 1000 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x * y / (x + y) = 1000
  ) (Finset.product (Finset.range 2001) (Finset.range 2001))).card

theorem solution_count_is_49 : solution_count = 49 := by
  sorry

end solution_count_is_49_l3960_396026


namespace kim_cherry_difference_l3960_396027

/-- The number of questions Nicole answered correctly -/
def nicole_correct : ℕ := 22

/-- The number of questions Cherry answered correctly -/
def cherry_correct : ℕ := 17

/-- The number of questions Kim answered correctly -/
def kim_correct : ℕ := nicole_correct + 3

theorem kim_cherry_difference : kim_correct - cherry_correct = 8 := by
  sorry

end kim_cherry_difference_l3960_396027


namespace solution_to_equation_l3960_396022

theorem solution_to_equation (y : ℝ) (h1 : y ≠ 3) (h2 : y ≠ 3/2) :
  (y^2 - 11*y + 24)/(y - 3) + (2*y^2 + 7*y - 18)/(2*y - 3) = -10 ↔ y = -4 :=
by sorry

end solution_to_equation_l3960_396022


namespace combined_female_average_score_l3960_396077

theorem combined_female_average_score 
  (a b c d : ℕ) 
  (adam_avg : (71 * a + 76 * b) / (a + b) = 74)
  (baker_avg : (81 * c + 90 * d) / (c + d) = 84)
  (male_avg : (71 * a + 81 * c) / (a + c) = 79) :
  (76 * b + 90 * d) / (b + d) = 84 :=
sorry

end combined_female_average_score_l3960_396077


namespace range_of_b_over_a_l3960_396001

def quadratic_equation (a b x : ℝ) : ℝ := x^2 + (a+1)*x + a + b + 1

theorem range_of_b_over_a (a b : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, quadratic_equation a b x = 0) →
  (x₁ ≠ x₂) →
  (quadratic_equation a b x₁ = 0) →
  (quadratic_equation a b x₂ = 0) →
  (0 < x₁ ∧ x₁ < 1) →
  (x₂ > 1) →
  (-2 < b/a ∧ b/a < -1/2) :=
sorry

end range_of_b_over_a_l3960_396001


namespace rotate_minus_two_zero_l3960_396048

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_minus_two_zero :
  rotate90Clockwise (-2, 0) = (0, 2) := by
  sorry

end rotate_minus_two_zero_l3960_396048


namespace orange_count_proof_l3960_396046

/-- The number of apples in the basket -/
def num_apples : ℕ := 10

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The initial number of oranges in the basket -/
def initial_oranges : ℕ := 5

theorem orange_count_proof :
  (num_apples : ℚ) = (1 / 2 : ℚ) * ((num_apples : ℚ) + (initial_oranges : ℚ) + (added_oranges : ℚ)) :=
by sorry

end orange_count_proof_l3960_396046


namespace biased_coin_probability_l3960_396076

def coin_prob (n : Nat) : ℚ :=
  match n with
  | 1 => 3/4
  | 2 => 1/2
  | 3 => 1/4
  | 4 => 1/3
  | 5 => 2/3
  | 6 => 3/5
  | 7 => 4/7
  | _ => 0

theorem biased_coin_probability :
  (coin_prob 1 * coin_prob 2 * (1 - coin_prob 3) * (1 - coin_prob 4) *
   (1 - coin_prob 5) * (1 - coin_prob 6) * (1 - coin_prob 7)) = 3/560 := by
  sorry

end biased_coin_probability_l3960_396076


namespace twoPointThreeFive_equals_fraction_l3960_396029

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 2.35̄ -/
def twoPointThreeFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 35 }

theorem twoPointThreeFive_equals_fraction :
  toRational twoPointThreeFive = 233 / 99 := by
  sorry

end twoPointThreeFive_equals_fraction_l3960_396029


namespace honey_balance_l3960_396041

/-- The initial amount of honey produced by a bee colony -/
def initial_honey : ℝ := 0.36

/-- The amount of honey eaten by bears -/
def eaten_honey : ℝ := 0.05

/-- The amount of honey that remains -/
def remaining_honey : ℝ := 0.31

/-- Theorem stating that the initial amount of honey is equal to the sum of eaten and remaining honey -/
theorem honey_balance : initial_honey = eaten_honey + remaining_honey := by
  sorry

end honey_balance_l3960_396041


namespace mixed_rectangles_count_even_l3960_396074

/-- Represents a tiling of an m × n grid using 2×2 and 1×3 mosaics -/
def GridTiling (m n : ℕ) : Type := Unit

/-- Counts the number of 1×2 rectangles with one cell from a 2×2 mosaic and one from a 1×3 mosaic -/
def countMixedRectangles (tiling : GridTiling m n) : ℕ := sorry

/-- Theorem stating that the count of mixed rectangles is even -/
theorem mixed_rectangles_count_even (m n : ℕ) (tiling : GridTiling m n) :
  Even (countMixedRectangles tiling) := by sorry

end mixed_rectangles_count_even_l3960_396074


namespace ac_squared_gt_bc_squared_implies_a_gt_b_l3960_396078

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b :=
by sorry

end ac_squared_gt_bc_squared_implies_a_gt_b_l3960_396078


namespace f_inequalities_l3960_396059

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities :
  (∀ x, f 3 x < 0 ↔ -1 < x ∧ x < 3) ∧
  (∀ x, f (-1) x > 0 ↔ x ≠ -1) ∧
  (∀ a, a > -1 → ∀ x, f a x > 0 ↔ x < -1 ∨ x > a) ∧
  (∀ a, a < -1 → ∀ x, f a x > 0 ↔ x < a ∨ x > -1) :=
by sorry

end f_inequalities_l3960_396059


namespace expansion_coefficient_sum_l3960_396083

theorem expansion_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℚ) : 
  (∀ x, (2*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  2^8 - 1 = 255 →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 81 := by
  sorry

end expansion_coefficient_sum_l3960_396083


namespace symmetric_seven_zeros_sum_l3960_396071

/-- A function representing |(1-x^2)(x^2+ax+b)| - c -/
def f (a b c x : ℝ) : ℝ := |(1 - x^2) * (x^2 + a*x + b)| - c

/-- Symmetry condition: f is symmetric about x = -2 -/
def is_symmetric (a b c : ℝ) : Prop :=
  ∀ x, f a b c (x + 2) = f a b c (-x - 2)

/-- The function has exactly 7 zeros -/
def has_seven_zeros (a b c : ℝ) : Prop :=
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f a b c x = 0

theorem symmetric_seven_zeros_sum (a b c : ℝ) :
  is_symmetric a b c →
  has_seven_zeros a b c →
  c ≠ 0 →
  a + b + c = 32 := by sorry

end symmetric_seven_zeros_sum_l3960_396071


namespace negation_of_existence_inequality_l3960_396055

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_inequality_l3960_396055


namespace correct_date_l3960_396094

-- Define a type for days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a type for months
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

-- Define a structure for a date
structure Date where
  day : Nat
  month : Month
  dayOfWeek : DayOfWeek

def nextDay (d : Date) : Date := sorry
def addDays (d : Date) (n : Nat) : Date := sorry

-- The main theorem
theorem correct_date (d : Date) : 
  (nextDay d).month ≠ Month.September ∧ 
  (addDays d 7).month = Month.September ∧
  (addDays d 2).dayOfWeek ≠ DayOfWeek.Wednesday →
  d = Date.mk 25 Month.August DayOfWeek.Wednesday :=
by sorry

end correct_date_l3960_396094


namespace happy_children_count_l3960_396061

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) :
  total = 60 →
  sad = 10 →
  neither = 20 →
  boys = 17 →
  girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  neither_boys = 5 →
  ∃ (happy : ℕ), happy = 30 ∧ happy + sad + neither = total :=
by sorry

end happy_children_count_l3960_396061


namespace geometric_sequence_ratio_l3960_396017

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  ratio : ℝ
  ratio_pos : ratio > 0
  geom : ∀ n, a (n + 1) = a n * ratio

/-- Theorem: In a geometric sequence with positive terms, if a₁a₃ = 4 and a₂ + a₄ = 10, then the common ratio is 2 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 2 + seq.a 4 = 10) :
  seq.ratio = 2 := by
  sorry

end geometric_sequence_ratio_l3960_396017


namespace temperature_proof_l3960_396049

-- Define the temperatures for each day
def monday : ℝ := sorry
def tuesday : ℝ := sorry
def wednesday : ℝ := sorry
def thursday : ℝ := sorry
def friday : ℝ := 31

-- Define the conditions
theorem temperature_proof :
  (monday + tuesday + wednesday + thursday) / 4 = 48 →
  (tuesday + wednesday + thursday + friday) / 4 = 46 →
  friday = 31 →
  monday = 39 :=
by sorry

end temperature_proof_l3960_396049


namespace greatest_length_of_rope_pieces_l3960_396004

theorem greatest_length_of_rope_pieces : Nat.gcd 28 (Nat.gcd 42 70) = 7 := by sorry

end greatest_length_of_rope_pieces_l3960_396004


namespace shirt_cost_proof_l3960_396063

/-- The cost of the shirt Macey wants to buy -/
def shirt_cost : ℚ := 3

/-- The amount Macey has already saved -/
def saved_amount : ℚ := 3/2

/-- The number of weeks Macey needs to save -/
def weeks_to_save : ℕ := 3

/-- The amount Macey saves per week -/
def weekly_savings : ℚ := 1/2

theorem shirt_cost_proof : 
  shirt_cost = saved_amount + weeks_to_save * weekly_savings := by
  sorry

end shirt_cost_proof_l3960_396063


namespace gold_copper_alloy_ratio_l3960_396073

theorem gold_copper_alloy_ratio 
  (G : ℝ) 
  (h_G : G > 9) : 
  let x := 9 / (G - 9)
  x * G + (1 - x) * 9 = 18 := by
  sorry

end gold_copper_alloy_ratio_l3960_396073


namespace complex_fraction_equality_l3960_396015

theorem complex_fraction_equality : (5 / 2 / (1 / 2) * (5 / 2)) / (5 / 2 * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end complex_fraction_equality_l3960_396015


namespace intersection_value_l3960_396002

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}

theorem intersection_value (m : ℕ) : A ∩ B m = {1, m} → m = 3 ∨ m = 5 := by
  sorry

end intersection_value_l3960_396002


namespace linen_tablecloth_cost_is_25_l3960_396037

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecorations where
  num_tables : ℕ
  place_settings_per_table : ℕ
  place_setting_cost : ℕ
  roses_per_centerpiece : ℕ
  lilies_per_centerpiece : ℕ
  rose_cost : ℕ
  lily_cost : ℕ
  total_decoration_cost : ℕ

/-- Calculates the cost of a single linen tablecloth --/
def linen_tablecloth_cost (d : WeddingDecorations) : ℕ :=
  let place_settings_cost := d.num_tables * d.place_settings_per_table * d.place_setting_cost
  let centerpiece_cost := d.num_tables * (d.roses_per_centerpiece * d.rose_cost + d.lilies_per_centerpiece * d.lily_cost)
  let tablecloth_total_cost := d.total_decoration_cost - (place_settings_cost + centerpiece_cost)
  tablecloth_total_cost / d.num_tables

/-- Theorem stating that the cost of a single linen tablecloth is $25 --/
theorem linen_tablecloth_cost_is_25 (d : WeddingDecorations)
  (h1 : d.num_tables = 20)
  (h2 : d.place_settings_per_table = 4)
  (h3 : d.place_setting_cost = 10)
  (h4 : d.roses_per_centerpiece = 10)
  (h5 : d.lilies_per_centerpiece = 15)
  (h6 : d.rose_cost = 5)
  (h7 : d.lily_cost = 4)
  (h8 : d.total_decoration_cost = 3500) :
  linen_tablecloth_cost d = 25 := by
  sorry

end linen_tablecloth_cost_is_25_l3960_396037


namespace m_range_l3960_396047

-- Define the condition function
def condition (x : ℝ) (m : ℝ) : Prop := 0 ≤ x ∧ x ≤ m

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0

-- Define the necessary but not sufficient relationship
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, quadratic_inequality x → condition x m) ∧
  (∃ x, condition x m ∧ ¬quadratic_inequality x)

-- Theorem statement
theorem m_range (m : ℝ) :
  necessary_not_sufficient m ↔ m ∈ Set.Ici 2 :=
sorry

end m_range_l3960_396047


namespace extreme_values_and_tangent_line_l3960_396085

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem extreme_values_and_tangent_line :
  -- Local minimum at x = 5/3
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (5/3 - δ₁) (5/3 + δ₁), f x ≥ f (5/3)) ∧
  f (5/3) = -58/27 ∧
  -- Local maximum at x = 1
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f 1 = -2 ∧
  -- Tangent line equation at x = 2
  (∀ x : ℝ, f' 2 * (x - 2) + f 2 = x - 4) := by
  sorry

end extreme_values_and_tangent_line_l3960_396085


namespace floor_times_self_equals_54_l3960_396065

theorem floor_times_self_equals_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ x = 54 / 7 := by
  sorry

end floor_times_self_equals_54_l3960_396065


namespace smallest_consecutive_digit_sum_divisible_by_7_l3960_396031

-- Define a function to calculate the digit sum of a natural number
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

-- Define a predicate for consecutive numbers with digit sums divisible by 7
def consecutiveDigitSumDivisibleBy7 (n : ℕ) : Prop :=
  (digitSum n) % 7 = 0 ∧ (digitSum (n + 1)) % 7 = 0

-- Theorem statement
theorem smallest_consecutive_digit_sum_divisible_by_7 :
  ∀ n : ℕ, n < 69999 → ¬(consecutiveDigitSumDivisibleBy7 n) ∧
  consecutiveDigitSumDivisibleBy7 69999 :=
by sorry

end smallest_consecutive_digit_sum_divisible_by_7_l3960_396031


namespace dusty_paid_hundred_l3960_396021

/-- Represents the cost and quantity of cake slices --/
structure CakeOrder where
  single_layer_cost : ℕ
  double_layer_cost : ℕ
  single_layer_quantity : ℕ
  double_layer_quantity : ℕ

/-- Calculates the total cost of the cake order --/
def total_cost (order : CakeOrder) : ℕ :=
  order.single_layer_cost * order.single_layer_quantity +
  order.double_layer_cost * order.double_layer_quantity

/-- Represents Dusty's cake purchase and change received --/
structure DustysPurchase where
  order : CakeOrder
  change_received : ℕ

/-- Theorem: Given Dusty's cake purchase and change received, prove that he paid $100 --/
theorem dusty_paid_hundred (purchase : DustysPurchase)
  (h1 : purchase.order.single_layer_cost = 4)
  (h2 : purchase.order.double_layer_cost = 7)
  (h3 : purchase.order.single_layer_quantity = 7)
  (h4 : purchase.order.double_layer_quantity = 5)
  (h5 : purchase.change_received = 37) :
  total_cost purchase.order + purchase.change_received = 100 := by
  sorry


end dusty_paid_hundred_l3960_396021


namespace simplify_expression_l3960_396008

theorem simplify_expression : (6^6 * 12^6 * 6^12 * 12^12 : ℕ) = 72^18 := by
  sorry

end simplify_expression_l3960_396008


namespace min_distance_exp_to_line_l3960_396057

/-- The minimum distance from a point on y = e^x to y = x is √2/2 -/
theorem min_distance_exp_to_line :
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let g : ℝ → ℝ := fun x ↦ x
  ∃ (x₀ : ℝ), ∀ (x : ℝ),
    Real.sqrt ((x - g x)^2 + (f x - g x)^2) ≥ Real.sqrt 2 / 2 :=
by sorry

end min_distance_exp_to_line_l3960_396057


namespace hyperbola_range_l3960_396066

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m + 1) = 1 ∧ (m + 2) * (m + 1) < 0

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end hyperbola_range_l3960_396066


namespace equal_volumes_of_modified_cylinders_l3960_396039

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders :
  let initial_radius : ℝ := 5
  let initial_height : ℝ := 10
  let radius_increase : ℝ := 4
  let volume1 := π * (initial_radius + radius_increase)^2 * initial_height
  let volume2 (x : ℝ) := π * initial_radius^2 * (initial_height + x)
  ∀ x : ℝ, volume1 = volume2 x ↔ x = 112 / 5 :=
by sorry

end equal_volumes_of_modified_cylinders_l3960_396039


namespace geometric_series_terms_l3960_396053

theorem geometric_series_terms (r : ℝ) (sum : ℝ) (h_r : r = 1/4) (h_sum : sum = 40) :
  let a := sum * (1 - r)
  (a * r = 7.5) ∧ (a * r^2 = 1.875) := by
sorry

end geometric_series_terms_l3960_396053


namespace vectors_in_same_plane_l3960_396011

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b c : V)

def is_basis (a b c : V) : Prop :=
  LinearIndependent ℝ ![a, b, c] ∧ Submodule.span ℝ {a, b, c} = ⊤

def coplanar (u v w : V) : Prop :=
  ∃ (x y z : ℝ), x • u + y • v + z • w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

theorem vectors_in_same_plane (h : is_basis V a b c) :
  coplanar V (2 • a + b) (a + b + c) (7 • a + 5 • b + 3 • c) :=
sorry

end vectors_in_same_plane_l3960_396011


namespace nonnegative_solutions_count_l3960_396097

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 18 := by
  sorry

end nonnegative_solutions_count_l3960_396097


namespace circle_equation_l3960_396093

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle is either (x+1)² + y² = 2 or (x+3)² + y² = 2 -/
theorem circle_equation (c : Circle) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ∨ (x + 3)^2 + y^2 = 2 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end circle_equation_l3960_396093


namespace complex_square_equality_l3960_396006

theorem complex_square_equality (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = (3 : ℂ) + 4 * Complex.I →
  a^2 + b^2 = 5 ∧ a * b = 2 := by
sorry

end complex_square_equality_l3960_396006


namespace paint_remaining_l3960_396028

theorem paint_remaining (num_statues : ℕ) (paint_per_statue : ℚ) (h1 : num_statues = 3) (h2 : paint_per_statue = 1/6) : 
  num_statues * paint_per_statue = 1/2 := by
  sorry

end paint_remaining_l3960_396028


namespace tournament_games_count_l3960_396086

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def total_games (ratio_won ratio_lost ratio_tied : ℕ) (games_won : ℕ) : ℕ :=
  let games_per_ratio := games_won / ratio_won
  let games_lost := ratio_lost * games_per_ratio
  let games_tied := ratio_tied * games_per_ratio
  games_won + games_lost + games_tied

/-- Theorem stating that given a ratio of 7:4:5 for games won:lost:tied and 42 games won, the total number of games played is 96. -/
theorem tournament_games_count :
  total_games 7 4 5 42 = 96 := by
  sorry

end tournament_games_count_l3960_396086


namespace squad_selection_ways_l3960_396025

/-- The number of ways to choose a squad of 8 players (including one dedicated setter) from a team of 12 members -/
def choose_squad (team_size : ℕ) (squad_size : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (squad_size - 1))

/-- Theorem stating that choosing a squad of 8 players (including one dedicated setter) from a team of 12 members can be done in 3960 ways -/
theorem squad_selection_ways :
  choose_squad 12 8 = 3960 := by
  sorry

end squad_selection_ways_l3960_396025


namespace increasing_interval_of_sine_function_l3960_396012

open Real

theorem increasing_interval_of_sine_function 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (ω : ℝ) :
  (ω > 0) →
  (∀ x, f x = 2 * sin (ω * x + π / 4)) →
  (∀ x, g x = 2 * cos (2 * x - π / 4)) →
  (∀ x, f (x + π / ω) = f x) →
  (∀ x, g (x + π) = g x) →
  (Set.Icc 0 (π / 8) : Set ℝ) = {x | x ∈ Set.Icc 0 π ∧ ∀ y ∈ Set.Icc 0 x, f y ≤ f x} :=
sorry

end increasing_interval_of_sine_function_l3960_396012


namespace bake_sale_donation_ratio_is_one_to_one_l3960_396098

/-- Represents the financial details of Andrew's bake sale fundraiser. -/
structure BakeSale where
  total_earnings : ℕ
  ingredient_cost : ℕ
  personal_donation : ℕ
  total_homeless_donation : ℕ

/-- Calculates the ratio of homeless shelter donation to food bank donation. -/
def donation_ratio (sale : BakeSale) : ℚ :=
  let available_for_donation := sale.total_earnings - sale.ingredient_cost
  let homeless_donation := sale.total_homeless_donation - sale.personal_donation
  let food_bank_donation := available_for_donation - homeless_donation
  homeless_donation / food_bank_donation

/-- Theorem stating that the donation ratio is 1:1 for the given bake sale. -/
theorem bake_sale_donation_ratio_is_one_to_one 
  (sale : BakeSale) 
  (h1 : sale.total_earnings = 400)
  (h2 : sale.ingredient_cost = 100)
  (h3 : sale.personal_donation = 10)
  (h4 : sale.total_homeless_donation = 160) : 
  donation_ratio sale = 1 := by
  sorry

end bake_sale_donation_ratio_is_one_to_one_l3960_396098


namespace fraction_equality_l3960_396072

theorem fraction_equality : 2 / 3 = (2 + 4) / (3 + 6) := by sorry

end fraction_equality_l3960_396072


namespace abhinav_bhupathi_total_money_l3960_396030

/-- The problem of calculating the total amount of money Abhinav and Bhupathi have together. -/
theorem abhinav_bhupathi_total_money (abhinav_amount bhupathi_amount : ℚ) : 
  (4 : ℚ) / 15 * abhinav_amount = (2 : ℚ) / 5 * bhupathi_amount →
  bhupathi_amount = 484 →
  abhinav_amount + bhupathi_amount = 1210 := by
  sorry

#check abhinav_bhupathi_total_money

end abhinav_bhupathi_total_money_l3960_396030


namespace gigi_initial_flour_l3960_396060

/-- The amount of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- The number of batches Gigi has already baked -/
def baked_batches : ℕ := 3

/-- The number of additional batches Gigi can make with the remaining flour -/
def future_batches : ℕ := 7

/-- The total amount of flour in Gigi's bag initially -/
def initial_flour : ℕ := flour_per_batch * (baked_batches + future_batches)

theorem gigi_initial_flour :
  initial_flour = 20 := by sorry

end gigi_initial_flour_l3960_396060


namespace largest_inexpressible_integer_l3960_396084

/-- 
Given positive integers a, b, and c with no pairwise common divisors greater than 1,
the largest integer that cannot be expressed as xbc + yca + zab 
(where x, y, z are non-negative integers) is 2abc - ab - bc - ca.
-/
theorem largest_inexpressible_integer 
  (a b c : ℕ+) 
  (h_coprime : ∀ (p : ℕ+), p ∣ a → p ∣ b → p = 1) 
  (h_coprime' : ∀ (p : ℕ+), p ∣ b → p ∣ c → p = 1) 
  (h_coprime'' : ∀ (p : ℕ+), p ∣ a → p ∣ c → p = 1) :
  ∀ n : ℤ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = x*b*c + y*c*a + z*a*b :=
by sorry

end largest_inexpressible_integer_l3960_396084


namespace lcm_18_24_l3960_396036

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l3960_396036


namespace evaluate_g_l3960_396016

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end evaluate_g_l3960_396016


namespace two_and_three_digit_problem_l3960_396096

theorem two_and_three_digit_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 7 * x * y ∧
    x + y = 1074 := by
  sorry

end two_and_three_digit_problem_l3960_396096


namespace marble_difference_is_seventeen_l3960_396051

/-- Calculates the difference in marbles between John and Ben after Ben gives half his marbles to John -/
def marbleDifference (benInitial : ℕ) (johnInitial : ℕ) : ℕ :=
  let benFinal := benInitial - benInitial / 2
  let johnFinal := johnInitial + benInitial / 2
  johnFinal - benFinal

/-- Proves that the difference in marbles between John and Ben is 17 after the transfer -/
theorem marble_difference_is_seventeen :
  marbleDifference 18 17 = 17 := by
  sorry

#eval marbleDifference 18 17

end marble_difference_is_seventeen_l3960_396051


namespace family_gathering_handshakes_l3960_396050

/-- The number of handshakes at a family gathering --/
def total_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twin_count := twin_sets * 2
  let triplet_count := triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_triplet_handshakes := twin_count * (triplet_count / 3) + triplet_count * (twin_count / 4)
  twin_handshakes + triplet_handshakes + twin_triplet_handshakes

/-- Theorem stating the total number of handshakes at the family gathering --/
theorem family_gathering_handshakes :
  total_handshakes 10 7 = 614 := by
  sorry

end family_gathering_handshakes_l3960_396050


namespace white_marbles_in_basket_c_l3960_396058

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- The greatest difference between marble counts in any basket -/
def greatestDifference : ℕ := 6

/-- Basket A containing red and yellow marbles -/
def basketA : Basket := ⟨"red", 4, "yellow", 2⟩

/-- Basket B containing green and yellow marbles -/
def basketB : Basket := ⟨"green", 6, "yellow", 1⟩

/-- Basket C containing white and yellow marbles -/
def basketC : Basket := ⟨"white", 15, "yellow", 9⟩

/-- Theorem stating that the number of white marbles in Basket C is 15 -/
theorem white_marbles_in_basket_c :
  basketC.color1 = "white" ∧ basketC.count1 = 15 :=
by sorry

end white_marbles_in_basket_c_l3960_396058


namespace roots_sum_abs_l3960_396045

theorem roots_sum_abs (a b c m : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end roots_sum_abs_l3960_396045


namespace smaller_tetrahedron_volume_ratio_l3960_396043

-- Define a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  is_positive : edge_length > 0

-- Define the division of edges
def divide_edges (t : RegularTetrahedron) : ℕ := 3

-- Define the smaller tetrahedron
structure SmallerTetrahedron (t : RegularTetrahedron) where
  division_points : divide_edges t = 3

-- Define the volume ratio
def volume_ratio (t : RegularTetrahedron) (s : SmallerTetrahedron t) : ℚ := 1 / 27

-- Theorem statement
theorem smaller_tetrahedron_volume_ratio 
  (t : RegularTetrahedron) 
  (s : SmallerTetrahedron t) : 
  volume_ratio t s = 1 / 27 := by
  sorry


end smaller_tetrahedron_volume_ratio_l3960_396043


namespace unique_rectangle_exists_restore_coordinate_system_l3960_396040

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop :=
  let AB := (r.B.x - r.A.x)^2 + (r.B.y - r.A.y)^2
  let BC := (r.C.x - r.B.x)^2 + (r.C.y - r.B.y)^2
  let CD := (r.D.x - r.C.x)^2 + (r.D.y - r.C.y)^2
  let DA := (r.A.x - r.D.x)^2 + (r.A.y - r.D.y)^2
  AB = CD ∧ BC = DA ∧ 
  (r.B.x - r.A.x) * (r.C.x - r.B.x) + (r.B.y - r.A.y) * (r.C.y - r.B.y) = 0

/-- Theorem: Given two points A and B, there exists a unique rectangle with A and B as diagonal endpoints -/
theorem unique_rectangle_exists (A B : Point) : 
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

/-- Main theorem: Given points A(1,2) and B(3,1), a unique rectangle can be constructed 
    with A and B as diagonal endpoints, which is sufficient to restore the coordinate system -/
theorem restore_coordinate_system : 
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨3, 1⟩
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

end unique_rectangle_exists_restore_coordinate_system_l3960_396040
