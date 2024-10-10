import Mathlib

namespace daisy_crown_problem_l3620_362014

theorem daisy_crown_problem (white pink red : ℕ) : 
  white = 6 →
  pink = 9 * white →
  white + pink + red = 273 →
  4 * pink - red = 3 :=
by
  sorry

end daisy_crown_problem_l3620_362014


namespace floor_power_divisibility_l3620_362097

theorem floor_power_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ := by
  sorry

end floor_power_divisibility_l3620_362097


namespace max_sum_of_pairwise_sums_l3620_362040

theorem max_sum_of_pairwise_sums (a b c d e : ℝ) 
  (h : (a + b) + (a + c) + (b + c) + (d + e) = 1096) :
  (a + d) + (a + e) + (b + d) + (b + e) + (c + d) + (c + e) ≤ 4384 := by
  sorry

end max_sum_of_pairwise_sums_l3620_362040


namespace daughter_least_intelligent_l3620_362036

-- Define the types for people and intelligence levels
inductive Person : Type
| Father : Person
| Sister : Person
| Son : Person
| Daughter : Person

inductive IntelligenceLevel : Type
| Least : IntelligenceLevel
| Smartest : IntelligenceLevel

-- Define the properties
def isTwin (p1 p2 : Person) : Prop := sorry

def sex (p : Person) : Bool := sorry

def age (p : Person) : ℕ := sorry

def intelligenceLevel (p : Person) : IntelligenceLevel := sorry

-- Define the theorem
theorem daughter_least_intelligent 
  (h1 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        (∃ p3 : Person, isTwin p1 p3 ∧ sex p3 ≠ sex p2))
  (h2 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        age p1 = age p2)
  : intelligenceLevel Person.Daughter = IntelligenceLevel.Least := by
  sorry

end daughter_least_intelligent_l3620_362036


namespace no_natural_square_diff_2014_l3620_362076

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_square_diff_2014_l3620_362076


namespace myrtle_eggs_theorem_l3620_362008

/-- The number of eggs Myrtle has after all collections and drops -/
def eggs_remaining (num_hens : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (dropped_eggs : List ℕ)
  (daily_eggs : List ℕ) : ℕ :=
  let total_eggs := (List.sum daily_eggs) * days_gone
  let remaining_after_neighbor := total_eggs - neighbor_took
  remaining_after_neighbor - (List.sum dropped_eggs)

/-- Theorem stating the number of eggs Myrtle has after all collections and drops -/
theorem myrtle_eggs_theorem : 
  eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3] = 162 := by
  sorry

#eval eggs_remaining 5 12 32 [3, 5, 2] [3, 4, 2, 5, 3]

end myrtle_eggs_theorem_l3620_362008


namespace visitor_growth_and_optimal_price_l3620_362020

def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000
def cost_price : ℚ := 6
def initial_price : ℚ := 25
def initial_sales : ℕ := 300
def sales_increase : ℕ := 30
def target_profit : ℚ := 6300

theorem visitor_growth_and_optimal_price :
  -- Part 1: Average annual growth rate
  ∃ (growth_rate : ℚ),
    (1 + growth_rate) ^ 2 * visitors_2022 = visitors_2024 ∧
    growth_rate = 1/5 ∧
  -- Part 2: Optimal selling price
  ∃ (optimal_price : ℚ),
    optimal_price ≤ initial_price ∧
    (optimal_price - cost_price) *
      (initial_sales + sales_increase * (initial_price - optimal_price)) =
      target_profit ∧
    optimal_price = 20 :=
  sorry

end visitor_growth_and_optimal_price_l3620_362020


namespace game_cost_l3620_362075

/-- 
Given:
- Will made 104 dollars mowing lawns
- He spent 41 dollars on new mower blades
- He bought 7 games with the remaining money
Prove that each game cost 9 dollars
-/
theorem game_cost (total_earned : ℕ) (spent_on_blades : ℕ) (num_games : ℕ) :
  total_earned = 104 →
  spent_on_blades = 41 →
  num_games = 7 →
  (total_earned - spent_on_blades) / num_games = 9 :=
by sorry

end game_cost_l3620_362075


namespace brick_fence_height_l3620_362003

/-- Calculates the height of a brick fence given its specifications -/
theorem brick_fence_height 
  (wall_length : ℕ) 
  (wall_depth : ℕ) 
  (num_walls : ℕ) 
  (total_bricks : ℕ) 
  (h1 : wall_length = 20)
  (h2 : wall_depth = 2)
  (h3 : num_walls = 4)
  (h4 : total_bricks = 800) : 
  total_bricks / (wall_length * num_walls * wall_depth) = 5 := by
sorry

end brick_fence_height_l3620_362003


namespace m_le_n_l3620_362089

theorem m_le_n (a b : ℝ) : 
  let m := (6^a) / (36^(a+1) + 1)
  let n := (1/3) * b^2 - b + 5/6
  m ≤ n := by sorry

end m_le_n_l3620_362089


namespace fraction_sum_equality_l3620_362000

theorem fraction_sum_equality : (7 : ℚ) / 10 + (3 : ℚ) / 100 + (9 : ℚ) / 1000 = 739 / 1000 := by
  sorry

end fraction_sum_equality_l3620_362000


namespace intersection_forms_hyperbola_l3620_362096

/-- The equation of the first line -/
def line1 (t x y : ℝ) : Prop := t * x - 2 * y - 3 * t = 0

/-- The equation of the second line -/
def line2 (t x y : ℝ) : Prop := x - 2 * t * y + 3 = 0

/-- The equation of a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / (9/4) = 1

/-- Theorem stating that the intersection points form a hyperbola -/
theorem intersection_forms_hyperbola :
  ∀ t x y : ℝ, line1 t x y → line2 t x y → is_hyperbola x y :=
sorry

end intersection_forms_hyperbola_l3620_362096


namespace power_of_three_equality_l3620_362023

theorem power_of_three_equality (m : ℕ) : 3^m = 27 * 81^4 * 243^3 → m = 34 := by
  sorry

end power_of_three_equality_l3620_362023


namespace expected_black_pairs_60_30_l3620_362022

/-- The expected number of adjacent black card pairs in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

/-- Theorem: Expected number of adjacent black pairs in a 60-card deck with 30 black cards -/
theorem expected_black_pairs_60_30 :
  expected_black_pairs 60 30 = 870 / 59 := by
  sorry

end expected_black_pairs_60_30_l3620_362022


namespace parabola_tangent_to_line_l3620_362043

/-- The parabola y = ax^2 + 10 is tangent to the line y = 2x + 3 if and only if a = 1/7 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 = 2 * x + 3) ↔ a = 1/7 := by
  sorry

end parabola_tangent_to_line_l3620_362043


namespace polynomial_coefficient_properties_l3620_362038

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₄ = 60 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63) := by
  sorry

end polynomial_coefficient_properties_l3620_362038


namespace expression_simplification_l3620_362098

theorem expression_simplification (x : ℤ) (hx : x ≠ 0) :
  (x^3 - 3*x^2*(x+2) + 4*x*(x+2)^2 - (x+2)^3 + 2) / (x * (x+2)) = 2 / (x * (x+2)) := by
  sorry

end expression_simplification_l3620_362098


namespace number_ratio_l3620_362027

theorem number_ratio : 
  ∀ (s l : ℕ), 
  s > 0 → 
  l > s → 
  l - s = 16 → 
  s = 28 → 
  (l : ℚ) / s = 11 / 7 := by
sorry

end number_ratio_l3620_362027


namespace only_subtraction_negative_positive_l3620_362002

theorem only_subtraction_negative_positive : 
  (1 + (-2) ≤ 0) ∧ 
  (1 - (-2) > 0) ∧ 
  (1 * (-2) ≤ 0) ∧ 
  (1 / (-2) < 0) :=
by sorry

end only_subtraction_negative_positive_l3620_362002


namespace min_students_in_class_l3620_362016

theorem min_students_in_class (boys girls : ℕ) : 
  (boys / 2 = girls * 2 / 3) →  -- Equal number of boys and girls passed
  (boys > 0) →                  -- There is at least one boy
  (girls > 0) →                 -- There is at least one girl
  (boys + girls ≥ 7) →          -- Total number of students is at least 7
  ∃ (min_students : ℕ), 
    min_students = boys + girls ∧ 
    min_students = 7 :=
by sorry

end min_students_in_class_l3620_362016


namespace car_efficiency_problem_l3620_362032

/-- The combined fuel efficiency of two cars -/
def combined_efficiency (e1 e2 : ℚ) : ℚ :=
  2 / (1 / e1 + 1 / e2)

theorem car_efficiency_problem :
  let ray_efficiency : ℚ := 50
  let tom_efficiency : ℚ := 15
  combined_efficiency ray_efficiency tom_efficiency = 300 / 13 := by
sorry

end car_efficiency_problem_l3620_362032


namespace students_liking_both_desserts_l3620_362046

theorem students_liking_both_desserts
  (total_students : ℕ)
  (ice_cream_fans : ℕ)
  (cookie_fans : ℕ)
  (neither_fans : ℕ)
  (h1 : total_students = 50)
  (h2 : ice_cream_fans = 28)
  (h3 : cookie_fans = 20)
  (h4 : neither_fans = 14) :
  total_students - neither_fans - (ice_cream_fans + cookie_fans - total_students + neither_fans) = 12 :=
by
  sorry

end students_liking_both_desserts_l3620_362046


namespace cos_x_plus_7pi_12_l3620_362047

theorem cos_x_plus_7pi_12 (x : ℝ) (h : Real.sin (x + π / 12) = 1 / 3) :
  Real.cos (x + 7 * π / 12) = - 1 / 3 := by
  sorry

end cos_x_plus_7pi_12_l3620_362047


namespace minyoung_position_l3620_362081

/-- Given a line of people, calculates the position from the front given the total number of people and the position from the back. -/
def position_from_front (total : ℕ) (from_back : ℕ) : ℕ :=
  total - from_back + 1

/-- Proves that in a line of 27 people, if a person is 13th from the back, they are 15th from the front. -/
theorem minyoung_position :
  position_from_front 27 13 = 15 := by
  sorry

end minyoung_position_l3620_362081


namespace unique_parallel_line_l3620_362021

/-- Two planes are parallel -/
def parallel_planes (α β : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Set (Fin 3 → ℝ)) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A point is in a plane -/
def point_in_plane (x : Fin 3 → ℝ) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Two lines are parallel -/
def parallel_lines (l₁ l₂ : Set (Fin 3 → ℝ)) : Prop := sorry

/-- The set of all lines in a plane passing through a point -/
def lines_through_point (p : Set (Fin 3 → ℝ)) (x : Fin 3 → ℝ) : Set (Set (Fin 3 → ℝ)) := sorry

theorem unique_parallel_line 
  (α β : Set (Fin 3 → ℝ)) 
  (a : Set (Fin 3 → ℝ)) 
  (B : Fin 3 → ℝ) 
  (h₁ : parallel_planes α β) 
  (h₂ : line_in_plane a α) 
  (h₃ : point_in_plane B β) : 
  ∃! l, l ∈ lines_through_point β B ∧ parallel_lines l a := by
  sorry

end unique_parallel_line_l3620_362021


namespace shaded_area_is_twelve_l3620_362092

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problemSetup (rect : Rectangle) (tri : IsoscelesTriangle) : Prop :=
  rect.height = tri.height ∧
  rect.base = 12 ∧
  rect.height = 8 ∧
  tri.base = 12

-- Define the intersection point
def intersectionPoint : Point :=
  { x := 18, y := 2 }

-- Theorem statement
theorem shaded_area_is_twelve (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h : problemSetup rect tri) : 
  (1/2 : ℝ) * tri.base * intersectionPoint.y = 12 := by
  sorry

end shaded_area_is_twelve_l3620_362092


namespace total_balloons_l3620_362065

theorem total_balloons (allan_balloons jake_balloons maria_balloons : ℕ) 
  (h1 : allan_balloons = 5)
  (h2 : jake_balloons = 7)
  (h3 : maria_balloons = 3) :
  allan_balloons + jake_balloons + maria_balloons = 15 := by
  sorry

end total_balloons_l3620_362065


namespace adult_ticket_cost_l3620_362060

/-- Proves that the cost of an adult ticket is 8 dollars given the specified conditions -/
theorem adult_ticket_cost
  (total_attendees : ℕ)
  (num_children : ℕ)
  (child_ticket_cost : ℕ)
  (total_revenue : ℕ)
  (h1 : total_attendees = 22)
  (h2 : num_children = 18)
  (h3 : child_ticket_cost = 1)
  (h4 : total_revenue = 50) :
  (total_revenue - num_children * child_ticket_cost) / (total_attendees - num_children) = 8 :=
by sorry

end adult_ticket_cost_l3620_362060


namespace xsin2x_necessary_not_sufficient_l3620_362072

theorem xsin2x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (∀ x, (0 < x ∧ x < π/2) → (x * Real.sin x < 1 → x * Real.sin x * Real.sin x < 1)) ∧
  (∃ x, (0 < x ∧ x < π/2) ∧ x * Real.sin x * Real.sin x < 1 ∧ x * Real.sin x ≥ 1) :=
by sorry

end xsin2x_necessary_not_sufficient_l3620_362072


namespace fraction_calculation_l3620_362011

theorem fraction_calculation : 
  (((4 : ℚ) / 9 + (1 : ℚ) / 9) / ((5 : ℚ) / 8 - (1 : ℚ) / 8)) = (10 : ℚ) / 9 := by
  sorry

end fraction_calculation_l3620_362011


namespace count_non_negative_rationals_l3620_362042

def rational_list : List ℚ := [-15, 5 + 1/3, -23/100, 0, 76/10, 2, -1/3, 314/100]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end count_non_negative_rationals_l3620_362042


namespace largest_digit_divisible_by_six_l3620_362054

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3456 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 ∧ (3456 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end largest_digit_divisible_by_six_l3620_362054


namespace complement_of_A_relative_to_U_l3620_362079

-- Define the universal set U
def U : Set ℝ := {x | x < 3}

-- Define the subset A
def A : Set ℝ := {x | x < 1}

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A_relative_to_U :
  complement_U_A = {x | 1 ≤ x ∧ x < 3} :=
sorry

end complement_of_A_relative_to_U_l3620_362079


namespace percentage_increase_proof_l3620_362025

def initial_earnings : ℝ := 65
def new_earnings : ℝ := 72

theorem percentage_increase_proof :
  let difference := new_earnings - initial_earnings
  let percentage_increase := (difference / initial_earnings) * 100
  ∀ ε > 0, |percentage_increase - 10.77| < ε :=
by
  sorry

end percentage_increase_proof_l3620_362025


namespace eraser_pencil_price_ratio_l3620_362048

/-- Represents the store's sales and pricing structure -/
structure StoreSales where
  pencils_sold : ℕ
  total_earnings : ℕ
  eraser_price : ℕ
  pencil_eraser_ratio : ℕ

/-- Theorem stating the ratio of eraser price to pencil price -/
theorem eraser_pencil_price_ratio 
  (s : StoreSales) 
  (h1 : s.pencils_sold = 20)
  (h2 : s.total_earnings = 80)
  (h3 : s.eraser_price = 1)
  (h4 : s.pencil_eraser_ratio = 2) : 
  (s.eraser_price : ℚ) / ((s.total_earnings - s.eraser_price * s.pencils_sold * s.pencil_eraser_ratio) / s.pencils_sold : ℚ) = 1 / 2 := by
  sorry


end eraser_pencil_price_ratio_l3620_362048


namespace algebraic_expression_value_l3620_362090

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) : a^2 - a*b = 3 := by
  sorry

end algebraic_expression_value_l3620_362090


namespace f_sum_symmetric_l3620_362039

/-- For the function f(x) = x + sin(x) + 1, f(x) + f(-x) = 2 for all real x -/
theorem f_sum_symmetric (x : ℝ) : let f : ℝ → ℝ := λ x ↦ x + Real.sin x + 1
  f x + f (-x) = 2 := by
  sorry

end f_sum_symmetric_l3620_362039


namespace total_smoothie_ingredients_l3620_362024

def strawberries : ℝ := 0.2
def yogurt : ℝ := 0.1
def orange_juice : ℝ := 0.2
def spinach : ℝ := 0.15
def protein_powder : ℝ := 0.05

theorem total_smoothie_ingredients :
  strawberries + yogurt + orange_juice + spinach + protein_powder = 0.7 := by
  sorry

end total_smoothie_ingredients_l3620_362024


namespace triangle_area_16_triangle_AED_area_l3620_362086

/-- The area of a triangle with base 8 and height 4 is 16 square units. -/
theorem triangle_area_16 (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (1 / 2) * base * height = 16 := by sorry

/-- Given a triangle AED where AE = 8, height = 4, and ED = DA = 5,
    the area of triangle AED is 16 square units. -/
theorem triangle_AED_area (AE ED DA height : ℝ)
  (h1 : AE = 8)
  (h2 : height = 4)
  (h3 : ED = 5)
  (h4 : DA = 5) :
  (1 / 2) * AE * height = 16 := by sorry

end triangle_area_16_triangle_AED_area_l3620_362086


namespace camping_match_ratio_l3620_362015

def match_ratio (initial matches_dropped final : ℕ) : ℚ :=
  let matches_lost := initial - final
  let matches_eaten := matches_lost - matches_dropped
  matches_eaten / matches_dropped

theorem camping_match_ratio :
  match_ratio 70 10 40 = 2 := by sorry

end camping_match_ratio_l3620_362015


namespace smallest_n_congruence_l3620_362082

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 678 [ZMOD 11] ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(19 * m ≡ 678 [ZMOD 11])) ↔ n = 5 := by
  sorry

end smallest_n_congruence_l3620_362082


namespace min_text_length_for_symbol_occurrence_l3620_362001

theorem min_text_length_for_symbol_occurrence : 
  ∃ (x : ℕ), (19 : ℝ) * (21 : ℝ) / 200 < (x : ℝ) ∧ (x : ℝ) < (19 : ℝ) * (11 : ℝ) / 100 ∧
  ∀ (L : ℕ), L < 19 → ¬∃ (y : ℕ), (L : ℝ) * (21 : ℝ) / 200 < (y : ℝ) ∧ (y : ℝ) < (L : ℝ) * (11 : ℝ) / 100 :=
by sorry

end min_text_length_for_symbol_occurrence_l3620_362001


namespace evaluate_expression_l3620_362028

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end evaluate_expression_l3620_362028


namespace tangent_implies_a_equals_two_l3620_362041

noncomputable section

-- Define the line and curve equations
def line (x : ℝ) : ℝ := x + 1
def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    line x = curve a x ∧ 
    (deriv (curve a)) x = (deriv line) x

-- Theorem statement
theorem tangent_implies_a_equals_two :
  ∀ a : ℝ, is_tangent a → a = 2 :=
sorry

end

end tangent_implies_a_equals_two_l3620_362041


namespace expression_value_l3620_362049

theorem expression_value : 
  let a : ℝ := 5
  let b : ℝ := 7
  let c : ℝ := 3
  (2*a - (3*b - 4*c)) - ((2*a - 3*b) - 4*c) = 24 := by
sorry

end expression_value_l3620_362049


namespace product_integer_part_l3620_362026

theorem product_integer_part : 
  ⌊(1.1 : ℝ) * 1.2 * 1.3 * 1.4 * 1.5 * 1.6⌋ = 1 := by sorry

end product_integer_part_l3620_362026


namespace b_range_l3620_362053

-- Define the quadratic equation
def quadratic (x b c : ℝ) : Prop := x^2 + 2*b*x + c = 0

-- Define the condition for roots in [-1, 1]
def roots_in_range (b c : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ quadratic x b c

-- Define the inequality condition
def inequality_condition (b c : ℝ) : Prop :=
  0 ≤ 4*b + c ∧ 4*b + c ≤ 3

-- Theorem statement
theorem b_range (b c : ℝ) :
  roots_in_range b c → inequality_condition b c → b ∈ Set.Icc (-1) 2 := by
  sorry


end b_range_l3620_362053


namespace tan_alpha_value_l3620_362095

theorem tan_alpha_value (α : Real) (h : Real.tan α = -1/2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end tan_alpha_value_l3620_362095


namespace magic_square_x_value_l3620_362037

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℤ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧ 
    (entries 0 j + entries 1 j + entries 2 j = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 2 + entries 1 1 + entries 2 0 = entries 0 0 + entries 0 1 + entries 0 2)

/-- The main theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.entries 0 0 = x)
  (h2 : ms.entries 0 1 = 21)
  (h3 : ms.entries 0 2 = 70)
  (h4 : ms.entries 1 0 = 7) :
  x = 133 := by
  sorry

end magic_square_x_value_l3620_362037


namespace product_of_complements_bound_l3620_362091

theorem product_of_complements_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end product_of_complements_bound_l3620_362091


namespace man_double_son_age_l3620_362094

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 35

/-- Represents the son's present age -/
def son_present_age : ℕ := 33

/-- Calculates the number of years until the man's age is twice his son's age -/
def years_until_double_age : ℕ := 2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age :
  (son_present_age + years_until_double_age) * 2 = 
  (son_present_age + age_difference + years_until_double_age) :=
by sorry

end man_double_son_age_l3620_362094


namespace max_value_of_expression_l3620_362064

theorem max_value_of_expression (x y : ℝ) : 
  2 * x^2 + 3 * y^2 = 22 * x + 18 * y + 20 →
  4 * x + 5 * y ≤ 110 :=
by sorry

end max_value_of_expression_l3620_362064


namespace m_range_for_g_l3620_362006

/-- Definition of an (a, b) type function -/
def is_ab_type_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = b

/-- Definition of the function g -/
def g (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - m * (x - 1) + 1

/-- Main theorem -/
theorem m_range_for_g :
  ∀ m : ℝ,
  (m > 0) →
  (is_ab_type_function (g m) 1 4) →
  (∀ x ∈ Set.Icc 0 2, 1 ≤ g m x ∧ g m x ≤ 3) →
  (2 - 2 * Real.sqrt 6 / 3 ≤ m ∧ m ≤ 2) :=
by sorry

end m_range_for_g_l3620_362006


namespace china_coal_production_2003_l3620_362074

/-- Represents a large number in different formats -/
structure LargeNumber where
  value : Nat
  word_representation : String
  billion_representation : String

/-- Converts a natural number to its word representation -/
def nat_to_words (n : Nat) : String :=
  sorry

/-- Converts a natural number to its billion representation -/
def nat_to_billions (n : Nat) : String :=
  sorry

/-- Theorem stating the correct representations of China's coal production in 2003 -/
theorem china_coal_production_2003 :
  let production : LargeNumber := {
    value := 15500000000,
    word_representation := nat_to_words 15500000000,
    billion_representation := nat_to_billions 15500000000
  }
  production.word_representation = "one hundred and fifty-five billion" ∧
  production.billion_representation = "155 billion" :=
sorry

end china_coal_production_2003_l3620_362074


namespace leftover_coin_value_l3620_362055

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 30

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters James has -/
def james_quarters : ℕ := 77

/-- The number of dimes James has -/
def james_dimes : ℕ := 138

/-- The number of quarters Lindsay has -/
def lindsay_quarters : ℕ := 112

/-- The number of dimes Lindsay has -/
def lindsay_dimes : ℕ := 244

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The theorem stating the value of leftover coins -/
theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 2.45 := by
  sorry


end leftover_coin_value_l3620_362055


namespace complex_number_in_first_quadrant_l3620_362009

theorem complex_number_in_first_quadrant (m : ℝ) (h : m > 1) :
  let z : ℂ := m * (3 + Complex.I) - (2 + Complex.I)
  z.re > 0 ∧ z.im > 0 :=
by sorry

end complex_number_in_first_quadrant_l3620_362009


namespace muffin_combinations_l3620_362069

/-- Given four kinds of muffins, when purchasing eight muffins with at least one of each kind,
    there are 23 different possible combinations. -/
theorem muffin_combinations : ℕ :=
  let num_muffin_types : ℕ := 4
  let total_muffins : ℕ := 8
  let min_of_each_type : ℕ := 1
  23

#check muffin_combinations

end muffin_combinations_l3620_362069


namespace max_value_of_x_plus_y_l3620_362073

theorem max_value_of_x_plus_y : ∃ (max : ℤ),
  (max = 13) ∧
  (∀ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 → x + y ≤ max) ∧
  (∃ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 ∧ x + y = max) := by
  sorry

end max_value_of_x_plus_y_l3620_362073


namespace triangle_area_upper_bound_l3620_362099

/-- Given a triangle ABC with BC = 2 and AB · AC = 1, prove that its area is at most √2 -/
theorem triangle_area_upper_bound (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
  let triangle_area := Real.sqrt (((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))^2 / 4)
  Real.sqrt ((BC.1^2 + BC.2^2) / 4) = 1 →
  dot_product AB AC = 1 →
  triangle_area ≤ Real.sqrt 2 := by
sorry

end triangle_area_upper_bound_l3620_362099


namespace largest_perfect_square_factor_of_3780_l3620_362029

/-- The largest perfect square factor of a natural number -/
def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The largest perfect square factor of 3780 is 36 -/
theorem largest_perfect_square_factor_of_3780 :
  largest_perfect_square_factor 3780 = 36 := by
  sorry

end largest_perfect_square_factor_of_3780_l3620_362029


namespace unique_solution_quadratic_system_l3620_362035

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) ∧ (x = 1/8) :=
by sorry

end unique_solution_quadratic_system_l3620_362035


namespace student_score_l3620_362063

theorem student_score (max_score : ℕ) (pass_threshold : ℚ) (fail_margin : ℕ) (student_score : ℕ) : 
  max_score = 500 →
  pass_threshold = 33 / 100 →
  fail_margin = 40 →
  student_score = ⌊max_score * pass_threshold⌋ - fail_margin →
  student_score = 125 := by
sorry

end student_score_l3620_362063


namespace haunted_house_entry_exit_l3620_362013

theorem haunted_house_entry_exit (total_windows : ℕ) (magical_barrier : ℕ) : 
  total_windows = 8 →
  magical_barrier = 1 →
  (total_windows - magical_barrier - 1) * (total_windows - 2) + 
  magical_barrier * (total_windows - 1) = 49 :=
by
  sorry

end haunted_house_entry_exit_l3620_362013


namespace g_monotonically_decreasing_l3620_362071

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotonically_decreasing (a : ℝ) :
  (∀ x < a / 3, g_derivative a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end g_monotonically_decreasing_l3620_362071


namespace min_tan_product_l3620_362050

theorem min_tan_product (α β γ : Real) (h_acute : α ∈ Set.Ioo 0 (π/2) ∧ β ∈ Set.Ioo 0 (π/2) ∧ γ ∈ Set.Ioo 0 (π/2)) 
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  ∃ (min : Real), 
    (∀ (α' β' γ' : Real), 
      α' ∈ Set.Ioo 0 (π/2) → β' ∈ Set.Ioo 0 (π/2) → γ' ∈ Set.Ioo 0 (π/2) →
      Real.cos α' ^ 2 + Real.cos β' ^ 2 + Real.cos γ' ^ 2 = 1 →
      Real.tan α' * Real.tan β' * Real.tan γ' ≥ min) ∧
    Real.tan α * Real.tan β * Real.tan γ = min ∧
    min = 2 * Real.sqrt 2 :=
by sorry

end min_tan_product_l3620_362050


namespace min_area_at_one_eighth_l3620_362034

-- Define the lines l₁ and l₂
def l₁ (k x y : ℝ) : Prop := k * x - 2 * y - 2 * k + 8 = 0
def l₂ (k x y : ℝ) : Prop := 2 * x + k^2 * y - 4 * k^2 - 4 = 0

-- Define the area of the quadrilateral as a function of k
noncomputable def quadrilateral_area (k : ℝ) : ℝ := 
  let x₁ := (2 * k - 8) / k
  let y₁ := 4 - k
  let x₂ := 2 * k^2 + 2
  let y₂ := 4 + 4 / k^2
  (x₁ * y₁) / 2 + (x₂ * y₂) / 2

-- State the theorem
theorem min_area_at_one_eighth (k : ℝ) (h : 0 < k ∧ k < 4) :
  ∃ (min_k : ℝ), min_k = 1/8 ∧ 
  ∀ k', 0 < k' ∧ k' < 4 → quadrilateral_area min_k ≤ quadrilateral_area k' :=
sorry

end min_area_at_one_eighth_l3620_362034


namespace square_area_and_diagonal_l3620_362033

/-- Given a square with perimeter 40 feet, prove its area and diagonal length -/
theorem square_area_and_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  (side ^ 2 = 100) ∧ (side * Real.sqrt 2 = 10 * Real.sqrt 2) := by
  sorry

end square_area_and_diagonal_l3620_362033


namespace vins_bike_trips_l3620_362012

theorem vins_bike_trips (distance_to_school : ℕ) (distance_from_school : ℕ) (total_distance : ℕ) :
  distance_to_school = 6 →
  distance_from_school = 7 →
  total_distance = 65 →
  total_distance / (distance_to_school + distance_from_school) = 5 := by
sorry

end vins_bike_trips_l3620_362012


namespace purely_imaginary_complex_l3620_362018

theorem purely_imaginary_complex (m : ℝ) : 
  (Complex.mk (m^2 - m) m).im ≠ 0 ∧ (Complex.mk (m^2 - m) m).re = 0 → m = 1 := by
  sorry

end purely_imaginary_complex_l3620_362018


namespace average_book_width_l3620_362017

def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]

theorem average_book_width :
  (book_widths.sum / book_widths.length : ℚ) = 241/49 := by
  sorry

end average_book_width_l3620_362017


namespace larger_cube_volume_l3620_362085

theorem larger_cube_volume (n : ℕ) (small_cube_volume : ℝ) (surface_area_diff : ℝ) :
  n = 216 →
  small_cube_volume = 1 →
  surface_area_diff = 1080 →
  (n : ℝ) * 6 * small_cube_volume^(2/3) - 6 * ((n : ℝ) * small_cube_volume)^(2/3) = surface_area_diff →
  (n : ℝ) * small_cube_volume = 216 :=
by sorry

end larger_cube_volume_l3620_362085


namespace quinary_444_equals_octal_174_l3620_362087

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (d : ℕ) : ℕ := sorry

/-- Theorem: The quinary number 444₅ is equal to the octal number 174₈ --/
theorem quinary_444_equals_octal_174 : 
  decimal_to_octal (quinary_to_decimal 444) = 174 := by sorry

end quinary_444_equals_octal_174_l3620_362087


namespace evenAdjacentCellsCount_l3620_362030

/-- The number of cells with an even number of adjacent cells in an equilateral triangle -/
def evenAdjacentCells (sideLength : ℕ) : ℕ :=
  sideLength * sideLength - (sideLength - 3) * (sideLength - 3) - 3

/-- The side length of the large equilateral triangle -/
def largeSideLength : ℕ := 2022

theorem evenAdjacentCellsCount :
  evenAdjacentCells largeSideLength = 12120 := by
  sorry

end evenAdjacentCellsCount_l3620_362030


namespace polyhedron_inequality_l3620_362019

/-- A convex polyhedron is represented by its number of vertices, edges, and maximum number of triangular faces sharing a common vertex. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  T : ℕ  -- maximum number of triangular faces sharing a common vertex

/-- The inequality V ≤ √E + T holds for any convex polyhedron. -/
theorem polyhedron_inequality (P : ConvexPolyhedron) : P.V ≤ Real.sqrt (P.E : ℝ) + P.T := by
  sorry

end polyhedron_inequality_l3620_362019


namespace variance_implies_fluctuation_l3620_362070

-- Define a type for our data set
def DataSet := List ℝ

-- Define variance
def variance (data : DataSet) : ℝ := sorry

-- Define a measure of fluctuation
def fluctuation (data : DataSet) : ℝ := sorry

-- Theorem statement
theorem variance_implies_fluctuation (data1 data2 : DataSet) :
  variance data1 > variance data2 → fluctuation data1 > fluctuation data2 := by
  sorry

end variance_implies_fluctuation_l3620_362070


namespace complement_of_N_in_M_l3620_362045

def M : Set Nat := {1, 2, 3, 4, 5}
def N : Set Nat := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end complement_of_N_in_M_l3620_362045


namespace geometric_sequence_minimum_l3620_362031

/-- Given a positive geometric sequence {a_n} where a_7 = a_6 + 2a_5, 
    and there exist two terms a_m and a_n such that √(a_m * a_n) = 4a_1,
    the minimum value of 1/m + 4/n is 3/2. -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- Positive sequence
  (∃ q > 0, ∀ k, a (k + 1) = q * a k) →  -- Geometric sequence
  a 7 = a 6 + 2 * a 5 →  -- Given condition
  Real.sqrt (a m * a n) = 4 * a 1 →  -- Given condition
  (∀ i j : ℕ, 1 / i + 4 / j ≥ 3 / 2) ∧  -- Minimum value is at least 3/2
  (∃ i j : ℕ, 1 / i + 4 / j = 3 / 2) :=  -- Minimum value of 3/2 is achievable
by sorry

end geometric_sequence_minimum_l3620_362031


namespace smallest_number_l3620_362066

def number_set : Set ℤ := {-1, 0, 1, 2}

theorem smallest_number : ∀ x ∈ number_set, -1 ≤ x := by sorry

end smallest_number_l3620_362066


namespace john_needs_two_sets_l3620_362084

/-- The number of metal bars in each set -/
def bars_per_set : ℕ := 7

/-- The total number of metal bars -/
def total_bars : ℕ := 14

/-- The number of sets of metal bars John needs -/
def sets_needed : ℕ := total_bars / bars_per_set

theorem john_needs_two_sets : sets_needed = 2 := by
  sorry

end john_needs_two_sets_l3620_362084


namespace intersection_length_l3620_362062

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the circle C in polar form
def circle_C (ρ θ : ℝ) : Prop := ρ^2 + 2*ρ*(Real.sin θ) = 3

-- Define the intersection points
def intersection_points (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ → Prop) : Set (ℝ × ℝ) :=
  {p | ∃ t, l t = p ∧ ∃ ρ θ, C ρ θ ∧ p.1 = ρ * (Real.cos θ) ∧ p.2 = ρ * (Real.sin θ)}

-- Theorem statement
theorem intersection_length :
  let points := intersection_points line_l circle_C
  ∃ M N : ℝ × ℝ, M ∈ points ∧ N ∈ points ∧ M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end intersection_length_l3620_362062


namespace swimmers_pass_23_times_l3620_362080

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed_a speed_b : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_pass_23_times :
  swimmers_passing_count 120 4 3 (15 * 60) = 23 := by
  sorry

end swimmers_pass_23_times_l3620_362080


namespace complex_in_second_quadrant_range_l3620_362004

theorem complex_in_second_quadrant_range (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 > 0) → 2 < x ∧ x < 5 := by
  sorry

end complex_in_second_quadrant_range_l3620_362004


namespace quadratic_inequality_implies_a_range_l3620_362051

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end quadratic_inequality_implies_a_range_l3620_362051


namespace squares_different_areas_l3620_362007

-- Define what a square is
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define properties of squares
def Square.isEquiangular (s : Square) : Prop := true
def Square.isRectangle (s : Square) : Prop := true
def Square.isRegularPolygon (s : Square) : Prop := true
def Square.isSimilarTo (s1 s2 : Square) : Prop := true

-- Define the area of a square
def Square.area (s : Square) : ℝ := s.side * s.side

-- Theorem: There exist squares with different areas
theorem squares_different_areas :
  ∃ (s1 s2 : Square), 
    Square.isEquiangular s1 ∧ 
    Square.isEquiangular s2 ∧
    Square.isRectangle s1 ∧ 
    Square.isRectangle s2 ∧
    Square.isRegularPolygon s1 ∧ 
    Square.isRegularPolygon s2 ∧
    Square.isSimilarTo s1 s2 ∧
    Square.area s1 ≠ Square.area s2 :=
by
  sorry

end squares_different_areas_l3620_362007


namespace Ba_atomic_weight_l3620_362083

def atomic_weight_Ba (molecular_weight : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  molecular_weight - atomic_weight_O

theorem Ba_atomic_weight :
  let molecular_weight : ℝ := 153
  let atomic_weight_O : ℝ := 16
  atomic_weight_Ba molecular_weight atomic_weight_O = 137 := by
sorry

end Ba_atomic_weight_l3620_362083


namespace shed_length_calculation_l3620_362078

theorem shed_length_calculation (backyard_length backyard_width shed_width sod_area : ℝ) :
  backyard_length = 20 ∧
  backyard_width = 13 ∧
  shed_width = 5 ∧
  sod_area = 245 →
  backyard_length * backyard_width - sod_area = shed_width * 3 :=
by
  sorry

end shed_length_calculation_l3620_362078


namespace triangle_isosceles_from_equation_l3620_362052

/-- A triangle with sides a, b, and c is isosceles if two of its sides are equal. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- 
  Theorem: If the three sides a, b, c of a triangle ABC satisfy a²-ac-b²+bc=0, 
  then the triangle is isosceles.
-/
theorem triangle_isosceles_from_equation 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_eq : a^2 - a*c - b^2 + b*c = 0) : 
  IsIsosceles a b c := by
  sorry


end triangle_isosceles_from_equation_l3620_362052


namespace valentine_treats_l3620_362077

/-- Represents the number of heart biscuits Mrs. Heine buys for each dog -/
def heart_biscuits_per_dog : ℕ := sorry

/-- Represents the total number of items Mrs. Heine buys -/
def total_items : ℕ := 12

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the number of sets of puppy boots per dog -/
def puppy_boots_per_dog : ℕ := 1

theorem valentine_treats :
  heart_biscuits_per_dog * num_dogs + puppy_boots_per_dog * num_dogs = total_items ∧
  heart_biscuits_per_dog = 4 := by sorry

end valentine_treats_l3620_362077


namespace parabola_vertex_x_coordinate_l3620_362068

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : f a b c 0 = 0)
  (h2 : f a b c 4 = 0)
  (h3 : f a b c 3 = 9) :
  -b / (2 * a) = 2 := by
sorry

end parabola_vertex_x_coordinate_l3620_362068


namespace nearest_integer_to_cube_root_five_sixth_power_l3620_362005

theorem nearest_integer_to_cube_root_five_sixth_power :
  ∃ (n : ℕ), n = 74608 ∧ ∀ (m : ℕ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end nearest_integer_to_cube_root_five_sixth_power_l3620_362005


namespace f_15_equals_227_l3620_362058

/-- Given a function f(n) = n^2 - n + 17, prove that f(15) = 227 -/
theorem f_15_equals_227 (f : ℕ → ℕ) (h : ∀ n, f n = n^2 - n + 17) : f 15 = 227 := by
  sorry

end f_15_equals_227_l3620_362058


namespace sqrt_calculation_l3620_362067

theorem sqrt_calculation : Real.sqrt (1/2) * Real.sqrt 8 - (Real.sqrt 3)^2 = -1 := by
  sorry

end sqrt_calculation_l3620_362067


namespace trailing_zeroes_1500_factorial_l3620_362057

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem trailing_zeroes_1500_factorial :
  trailingZeroes 1500 = 374 := by
  sorry

end trailing_zeroes_1500_factorial_l3620_362057


namespace square_filling_theorem_l3620_362010

def is_valid_permutation (p : Fin 5 → Fin 5) : Prop :=
  Function.Injective p ∧ Function.Surjective p

theorem square_filling_theorem :
  ∃ (p : Fin 5 → Fin 5), is_valid_permutation p ∧
    (p 0).val + 1 + (p 1).val + 1 = ((p 2).val + 1) * ((p 3).val + 1 - ((p 4).val + 1)) :=
by sorry

end square_filling_theorem_l3620_362010


namespace shobha_current_age_l3620_362093

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobha_current_age 
  (shekhar_age shobha_age : ℕ) 
  (age_ratio : shekhar_age / shobha_age = 4 / 3) 
  (shekhar_future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
sorry

end shobha_current_age_l3620_362093


namespace inequality_system_solution_set_l3620_362061

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 6 ≤ 8 ∧ x - 7 < 2 * (x - 3)}
  S = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end inequality_system_solution_set_l3620_362061


namespace puffy_muffy_weight_l3620_362044

/-- The weight of Scruffy in ounces -/
def scruffy_weight : ℕ := 12

/-- The weight difference between Scruffy and Muffy in ounces -/
def muffy_scruffy_diff : ℕ := 3

/-- The weight difference between Puffy and Muffy in ounces -/
def puffy_muffy_diff : ℕ := 5

/-- The combined weight of Puffy and Muffy in ounces -/
def combined_weight : ℕ := scruffy_weight - muffy_scruffy_diff + (scruffy_weight - muffy_scruffy_diff + puffy_muffy_diff)

theorem puffy_muffy_weight : combined_weight = 23 := by
  sorry

end puffy_muffy_weight_l3620_362044


namespace max_min_diff_abs_sum_ratio_l3620_362088

/-- The difference between the maximum and minimum values of |a + b| / (|a| + |b|) for nonzero real numbers a and b is 1. -/
theorem max_min_diff_abs_sum_ratio : ∃ (m' M' : ℝ),
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → m' ≤ |a + b| / (|a| + |b|) ∧ |a + b| / (|a| + |b|) ≤ M') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = m') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = M') ∧
  M' - m' = 1 := by
sorry

end max_min_diff_abs_sum_ratio_l3620_362088


namespace line_intersects_ellipse_l3620_362056

/-- The line kx+y+k+1=0 intersects the ellipse x^2/25 + y^2/16 = 1 for all real values of k -/
theorem line_intersects_ellipse (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + k + 1 = 0) ∧ (x^2 / 25 + y^2 / 16 = 1) := by sorry

end line_intersects_ellipse_l3620_362056


namespace remainder_97_pow_50_mod_100_l3620_362059

theorem remainder_97_pow_50_mod_100 : 97^50 % 100 = 49 := by
  sorry

end remainder_97_pow_50_mod_100_l3620_362059
