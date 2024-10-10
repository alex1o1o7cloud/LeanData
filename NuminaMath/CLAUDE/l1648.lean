import Mathlib

namespace hyperbola_equation_l1648_164863

/-- A hyperbola sharing a focus with a parabola and having a specific eccentricity -/
def HyperbolaWithSharedFocus (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (x₀ y₀ : ℝ), (x₀ = 2 ∧ y₀ = 0) ∧  -- Focus of parabola y² = 8x
  ∃ (c : ℝ), c = 2 ∧  -- Distance from center to focus
  ∃ (e : ℝ), e = 2 ∧ e = c / a  -- Eccentricity

/-- Theorem stating the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) 
  (h : HyperbolaWithSharedFocus a b) : 
  a = 1 ∧ b^2 = 3 := by sorry

end hyperbola_equation_l1648_164863


namespace suitcase_problem_l1648_164865

structure SuitcaseScenario where
  total_suitcases : ℕ
  business_suitcases : ℕ
  placement_interval : ℕ

def scenario : SuitcaseScenario :=
  { total_suitcases := 200
  , business_suitcases := 10
  , placement_interval := 2 }

def probability_last_suitcase_at_two_minutes (s : SuitcaseScenario) : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose s.total_suitcases s.business_suitcases : ℚ)

def expected_waiting_time (s : SuitcaseScenario) : ℚ :=
  4020 / 11

theorem suitcase_problem (s : SuitcaseScenario) 
  (h1 : s.total_suitcases = 200) 
  (h2 : s.business_suitcases = 10) 
  (h3 : s.placement_interval = 2) :
  probability_last_suitcase_at_two_minutes s = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) ∧ 
  expected_waiting_time s = 4020 / 11 := by
  sorry

#eval probability_last_suitcase_at_two_minutes scenario
#eval expected_waiting_time scenario

end suitcase_problem_l1648_164865


namespace range_of_f_on_interval_range_of_a_l1648_164846

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Part 1: Range of f on [0, 4]
theorem range_of_f_on_interval :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 0 4, f x = y ∧
  ∀ x ∈ Set.Icc 0 4, 1 ≤ f x ∧ f x ≤ 10 :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f x ≤ 5) ↔ a ∈ Set.Icc (-1) 1 :=
sorry

end range_of_f_on_interval_range_of_a_l1648_164846


namespace circle_diameter_endpoints_l1648_164857

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Checks if two points are endpoints of a diameter in a given circle -/
def areDiameterEndpoints (c : Circle) (p1 p2 : Point2D) : Prop :=
  (p1.x - c.center.x)^2 + (p1.y - c.center.y)^2 = c.radius^2 ∧
  (p2.x - c.center.x)^2 + (p2.y - c.center.y)^2 = c.radius^2 ∧
  (p1.x + p2.x) / 2 = c.center.x ∧
  (p1.y + p2.y) / 2 = c.center.y

theorem circle_diameter_endpoints :
  let c : Circle := { center := { x := 1, y := 2 }, radius := Real.sqrt 13 }
  let p1 : Point2D := { x := 3, y := -1 }
  let p2 : Point2D := { x := -1, y := 5 }
  areDiameterEndpoints c p1 p2 := by
  sorry

end circle_diameter_endpoints_l1648_164857


namespace arithmetic_sequence_ratio_property_l1648_164886

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n - 1)) / 2

/-- Theorem: If S_6 / S_3 = 3 for an arithmetic sequence, then S_12 / S_9 = 5/3 -/
theorem arithmetic_sequence_ratio_property (seq : ArithmeticSequence) 
  (h : seq.S 6 / seq.S 3 = 3) : seq.S 12 / seq.S 9 = 5/3 := by
  sorry

end arithmetic_sequence_ratio_property_l1648_164886


namespace arthurs_dinner_cost_l1648_164885

/-- Calculate the total cost of Arthur's dinner --/
theorem arthurs_dinner_cost :
  let appetizer_cost : ℚ := 8
  let steak_cost : ℚ := 20
  let wine_cost : ℚ := 3
  let dessert_cost : ℚ := 6
  let wine_glasses : ℕ := 2
  let voucher_discount : ℚ := 1/2
  let tip_percentage : ℚ := 1/5

  let full_meal_cost : ℚ := appetizer_cost + steak_cost + wine_cost * wine_glasses + dessert_cost
  let discounted_meal_cost : ℚ := full_meal_cost - steak_cost * voucher_discount
  let tip : ℚ := full_meal_cost * tip_percentage

  discounted_meal_cost + tip = 38 := by sorry

end arthurs_dinner_cost_l1648_164885


namespace paint_house_time_l1648_164812

/-- Given that five people can paint a house in seven hours and everyone works at the same rate,
    proves that two people would take 17.5 hours to paint the same house. -/
theorem paint_house_time (people_rate : ℝ → ℝ → ℝ) :
  (people_rate 5 7 = 1) →  -- Five people can paint the house in seven hours
  (∀ n t, people_rate n t = people_rate 1 1 * n * t) →  -- Everyone works at the same rate
  (people_rate 2 17.5 = 1) :=  -- Two people take 17.5 hours
by sorry

end paint_house_time_l1648_164812


namespace dodecahedron_face_centers_form_icosahedron_l1648_164828

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  -- Add necessary properties

/-- A regular icosahedron -/
structure RegularIcosahedron where
  -- Add necessary properties

/-- Function that connects centers of faces of a regular dodecahedron -/
def connectFaceCenters (d : RegularDodecahedron) : RegularIcosahedron :=
  sorry

/-- Theorem stating that connecting face centers of a regular dodecahedron results in a regular icosahedron -/
theorem dodecahedron_face_centers_form_icosahedron (d : RegularDodecahedron) :
  ∃ (i : RegularIcosahedron), connectFaceCenters d = i :=
sorry

end dodecahedron_face_centers_form_icosahedron_l1648_164828


namespace problem_statement_l1648_164807

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end problem_statement_l1648_164807


namespace dividend_divisible_by_divisor_l1648_164837

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^55 + x^44 + x^33 + x^22 + x^11 + 1

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- Theorem stating that the dividend is divisible by the divisor -/
theorem dividend_divisible_by_divisor :
  ∃ q : ℂ → ℂ, ∀ x, dividend x = (divisor x) * (q x) := by
  sorry

end dividend_divisible_by_divisor_l1648_164837


namespace sequence_expression_l1648_164833

theorem sequence_expression (a : ℕ → ℕ) (h1 : a 1 = 33) 
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 := by
sorry

end sequence_expression_l1648_164833


namespace quadratic_equation_roots_range_l1648_164873

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 4 = 0 ∧ x₂^2 + m*x₂ + 4 = 0) → 
  m ≤ -4 :=
by sorry

end quadratic_equation_roots_range_l1648_164873


namespace sum_of_mixed_numbers_l1648_164854

theorem sum_of_mixed_numbers : 
  (3 + 1/6 : ℚ) + (4 + 2/3 : ℚ) + (6 + 1/18 : ℚ) = 13 + 8/9 := by sorry

end sum_of_mixed_numbers_l1648_164854


namespace three_turns_sufficient_l1648_164891

/-- Represents a five-digit number with distinct digits -/
structure FiveDigitNumber where
  digits : Fin 5 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- Represents a turn where positions are selected and digits are revealed -/
structure Turn where
  positions : Set (Fin 5)
  revealed_digits : Set (Fin 10)

/-- Represents the process of guessing the number -/
def guess_number (n : FiveDigitNumber) (turns : List Turn) : Prop :=
  ∀ m : FiveDigitNumber, 
    (∀ t ∈ turns, {n.digits i | i ∈ t.positions} = t.revealed_digits) →
    (∀ t ∈ turns, {m.digits i | i ∈ t.positions} = t.revealed_digits) →
    n = m

/-- The main theorem stating that 3 turns are sufficient -/
theorem three_turns_sufficient :
  ∃ strategy : List Turn, 
    strategy.length ≤ 3 ∧ 
    ∀ n : FiveDigitNumber, guess_number n strategy :=
sorry

end three_turns_sufficient_l1648_164891


namespace sequence_sum_formula_l1648_164896

theorem sequence_sum_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = 2 * a n - 2) →
  ∀ n : ℕ, a n = 2^n :=
by sorry

end sequence_sum_formula_l1648_164896


namespace total_cost_calculation_l1648_164813

def total_cost (total_bricks : ℕ) (discount1_percent : ℚ) (discount2_percent : ℚ) 
                (full_price : ℚ) (discount1_fraction : ℚ) (discount2_fraction : ℚ)
                (full_price_fraction : ℚ) (additional_cost : ℚ) : ℚ :=
  let discounted_price1 := full_price * (1 - discount1_percent)
  let discounted_price2 := full_price * (1 - discount2_percent)
  let cost1 := (total_bricks : ℚ) * discount1_fraction * discounted_price1
  let cost2 := (total_bricks : ℚ) * discount2_fraction * discounted_price2
  let cost3 := (total_bricks : ℚ) * full_price_fraction * full_price
  cost1 + cost2 + cost3 + additional_cost

theorem total_cost_calculation :
  total_cost 1000 (1/2) (1/5) (1/2) (3/10) (2/5) (3/10) 200 = 585 := by
  sorry

end total_cost_calculation_l1648_164813


namespace man_son_age_difference_man_son_age_difference_proof_l1648_164884

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun son_age man_age =>
    (son_age = 22) →
    (man_age + 2 = 2 * (son_age + 2)) →
    (man_age - son_age = 24)

-- The proof is omitted
theorem man_son_age_difference_proof : ∃ (son_age man_age : ℕ), man_son_age_difference son_age man_age :=
  sorry

end man_son_age_difference_man_son_age_difference_proof_l1648_164884


namespace vocabulary_increase_l1648_164845

def words_per_day : ℕ := 10
def years : ℕ := 2
def days_per_year : ℕ := 365
def initial_vocabulary : ℕ := 14600

theorem vocabulary_increase :
  let total_new_words := words_per_day * years * days_per_year
  let final_vocabulary := initial_vocabulary + total_new_words
  let percentage_increase := (total_new_words : ℚ) / (initial_vocabulary : ℚ) * 100
  percentage_increase = 50 := by sorry

end vocabulary_increase_l1648_164845


namespace quadratic_equation_one_solution_l1648_164827

theorem quadratic_equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + x + 1 = 0) ↔ a = 0 ∨ a = 1/4 := by
  sorry

end quadratic_equation_one_solution_l1648_164827


namespace right_triangle_third_side_l1648_164816

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ (a = 3 ∨ b = 3) :=
sorry

end right_triangle_third_side_l1648_164816


namespace arrangement_count_l1648_164838

def arrange_people (n : ℕ) (k : ℕ) (m : ℕ) : Prop :=
  (n = 6) ∧ (k = 2) ∧ (m = 4)

theorem arrangement_count (n k m : ℕ) (h : arrange_people n k m) : 
  (Nat.choose n 2 * 2) + (Nat.choose n 3) = 50 := by
  sorry

end arrangement_count_l1648_164838


namespace plant_arrangements_eq_144_l1648_164801

/-- The number of ways to arrange 3 distinct vegetable plants and 3 distinct flower plants in a row,
    with all flower plants next to each other -/
def plant_arrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 3)

/-- Theorem stating that the number of plant arrangements is 144 -/
theorem plant_arrangements_eq_144 : plant_arrangements = 144 := by
  sorry

end plant_arrangements_eq_144_l1648_164801


namespace optimal_viewpoint_for_scenery_l1648_164802

/-- The problem setup -/
structure ScenerySetup where
  A : ℝ × ℝ
  B : ℝ × ℝ
  distance_AB : ℝ

/-- The viewing angle between two points from a given viewpoint -/
def viewing_angle (viewpoint : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The optimal viewpoint maximizes the viewing angle -/
def is_optimal_viewpoint (setup : ScenerySetup) (viewpoint : ℝ × ℝ) : Prop :=
  ∀ other : ℝ × ℝ, viewing_angle viewpoint setup.A setup.B ≥ viewing_angle other setup.A setup.B

/-- The main theorem -/
theorem optimal_viewpoint_for_scenery (setup : ScenerySetup) 
    (h1 : setup.A = (Real.sqrt 2, Real.sqrt 2))
    (h2 : setup.B = (0, 2 * Real.sqrt 2))
    (h3 : setup.distance_AB = 2)
    (h4 : setup.A.2 > 0 ∧ setup.B.2 > 0) : -- Ensuring A and B are on the same side of x-axis
  is_optimal_viewpoint setup (0, 0) := by sorry

end optimal_viewpoint_for_scenery_l1648_164802


namespace infinite_geometric_series_sum_specific_series_sum_l1648_164847

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^n

theorem infinite_geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem specific_series_sum :
  ∑' n, geometric_series (1/4) (1/2) n = 1/2 :=
sorry

end infinite_geometric_series_sum_specific_series_sum_l1648_164847


namespace survivor_probability_l1648_164874

/-- The number of contestants in the game -/
def total_contestants : ℕ := 18

/-- The number of tribes in the game -/
def number_of_tribes : ℕ := 3

/-- The number of contestants in each tribe -/
def contestants_per_tribe : ℕ := 6

/-- The number of contestants who quit the game -/
def quitters : ℕ := 2

/-- Probability that both quitters are from the same tribe -/
def prob_same_tribe : ℚ := 5/17

theorem survivor_probability :
  (total_contestants = number_of_tribes * contestants_per_tribe) →
  (total_contestants ≥ quitters) →
  (prob_same_tribe = (number_of_tribes * (contestants_per_tribe.choose quitters)) / 
                     (total_contestants.choose quitters)) :=
by sorry

end survivor_probability_l1648_164874


namespace faye_age_l1648_164841

/-- Represents the ages of the people in the problem -/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.diana + 5 = ages.eduardo ∧
  ages.eduardo = ages.chad + 6 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 17

/-- The theorem statement -/
theorem faye_age (ages : Ages) : age_conditions ages → ages.faye = 20 := by
  sorry

end faye_age_l1648_164841


namespace function_difference_inequality_l1648_164820

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h1 : ∀ x > 1, deriv f x > deriv g x)
  (h2 : ∀ x < 1, deriv f x < deriv g x) :
  f 2 - f 1 > g 2 - g 1 :=
by sorry

end function_difference_inequality_l1648_164820


namespace distance_to_x_axis_distance_M_to_x_axis_l1648_164809

/-- The distance from a point to the x-axis in a Cartesian coordinate system
    is equal to the absolute value of its y-coordinate. -/
theorem distance_to_x_axis (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  abs y = dist M (x, 0) :=
by sorry

/-- The distance from the point M(-9,12) to the x-axis is 12. -/
theorem distance_M_to_x_axis :
  let M : ℝ × ℝ := (-9, 12)
  dist M (-9, 0) = 12 :=
by sorry

end distance_to_x_axis_distance_M_to_x_axis_l1648_164809


namespace board_numbers_proof_l1648_164880

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_proof :
  ∃ (a b c d e : ℤ),
    pairwise_sums a b c d e = {5, 9, 10, 11, 12, 16, 16, 17, 21, 23} ∧
    Finset.toList {a, b, c, d, e} = [2, 3, 7, 9, 14] ∧
    a * b * c * d * e = 5292 :=
by sorry

end board_numbers_proof_l1648_164880


namespace intersection_locus_is_hyperbola_l1648_164855

/-- The locus of points (x, y) satisfying the given system of equations is a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∀ (x y s : ℝ), 
    (s * x - 3 * y - 4 * s = 0) → 
    (x - 3 * s * y + 4 = 0) → 
    ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end intersection_locus_is_hyperbola_l1648_164855


namespace arithmetic_calculation_l1648_164823

theorem arithmetic_calculation : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end arithmetic_calculation_l1648_164823


namespace abc_sum_mod_seven_l1648_164836

theorem abc_sum_mod_seven (a b c : ℤ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℤ) →
  (a * b * c) % 7 = 1 →
  (2 * c) % 7 = 5 →
  (3 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 6 := by
sorry

end abc_sum_mod_seven_l1648_164836


namespace power_of_power_l1648_164800

theorem power_of_power : (2^2)^3 = 64 := by
  sorry

end power_of_power_l1648_164800


namespace jacket_final_price_l1648_164834

/-- The final price of a jacket after two successive discounts -/
theorem jacket_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 9 := by
  sorry

end jacket_final_price_l1648_164834


namespace jerseys_sold_l1648_164872

def jersey_profit : ℕ := 165
def total_jersey_sales : ℕ := 25740

theorem jerseys_sold : 
  (total_jersey_sales / jersey_profit : ℕ) = 156 :=
by sorry

end jerseys_sold_l1648_164872


namespace not_p_sufficient_not_necessary_for_not_q_l1648_164879

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 2| > 3
def q (x : ℝ) : Prop := x > 5

-- Statement to prove
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l1648_164879


namespace sqrt_equation_solution_l1648_164835

theorem sqrt_equation_solution (x : ℝ) :
  x ≥ 1 →
  (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1) ↔
  (5 ≤ x ∧ x ≤ 10) :=
by sorry

end sqrt_equation_solution_l1648_164835


namespace student_correct_answers_l1648_164868

theorem student_correct_answers 
  (total_questions : ℕ) 
  (score : ℤ) 
  (correct_answers : ℕ) 
  (incorrect_answers : ℕ) :
  total_questions = 100 →
  score = correct_answers - 2 * incorrect_answers →
  correct_answers + incorrect_answers = total_questions →
  score = 70 →
  correct_answers = 90 := by
sorry

end student_correct_answers_l1648_164868


namespace unique_right_triangle_existence_l1648_164826

/-- A right triangle with leg lengths a and b, and hypotenuse c. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c

/-- The difference between the sum of legs and hypotenuse. -/
def leg_hyp_diff (t : RightTriangle) : ℝ := t.a + t.b - t.c

/-- Theorem: A unique right triangle exists given one leg a and the difference d
    between the sum of the legs and the hypotenuse, if and only if d < a. -/
theorem unique_right_triangle_existence (a d : ℝ) (ha : 0 < a) :
  (∃! t : RightTriangle, t.a = a ∧ leg_hyp_diff t = d) ↔ d < a := by
  sorry

end unique_right_triangle_existence_l1648_164826


namespace grocery_store_inventory_l1648_164856

/-- The total number of bottles and fruits in a grocery store -/
def total_items (regular_soda diet_soda sparkling_water orange_juice cranberry_juice apples oranges bananas pears : ℕ) : ℕ :=
  regular_soda + diet_soda + sparkling_water + orange_juice + cranberry_juice + apples + oranges + bananas + pears

/-- Theorem stating the total number of items in the grocery store -/
theorem grocery_store_inventory : 
  total_items 130 88 65 47 27 102 88 74 45 = 666 := by
  sorry

end grocery_store_inventory_l1648_164856


namespace hcd_6432_132_minus_8_l1648_164883

theorem hcd_6432_132_minus_8 : Nat.gcd 6432 132 - 8 = 4 := by
  sorry

end hcd_6432_132_minus_8_l1648_164883


namespace certain_number_proof_l1648_164899

theorem certain_number_proof (n : ℝ) : n / 1.25 = 5700 → n = 7125 := by
  sorry

end certain_number_proof_l1648_164899


namespace quadratic_equation_solution_l1648_164806

theorem quadratic_equation_solution (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 := by
  sorry

end quadratic_equation_solution_l1648_164806


namespace distance_from_negative_one_l1648_164867

theorem distance_from_negative_one : 
  {x : ℝ | |x - (-1)| = 5} = {4, -6} := by
sorry

end distance_from_negative_one_l1648_164867


namespace chicken_count_l1648_164825

/-- The number of chickens in different locations and their relationships --/
theorem chicken_count :
  ∀ (coop run free_range barn : ℕ),
  coop = 14 →
  run = 2 * coop →
  5 * (coop + run) = 2 * free_range →
  2 * barn = coop →
  free_range = 105 := by
  sorry

end chicken_count_l1648_164825


namespace overall_average_calculation_l1648_164829

theorem overall_average_calculation (math_score history_score third_score : ℚ) 
  (h1 : math_score = 74/100)
  (h2 : history_score = 81/100)
  (h3 : third_score = 70/100) :
  (math_score + history_score + third_score) / 3 = 75/100 := by
  sorry

end overall_average_calculation_l1648_164829


namespace gaochun_population_scientific_notation_l1648_164818

theorem gaochun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    (1 ≤ |a| ∧ |a| < 10) ∧ 
    425000 = a * (10 : ℝ) ^ n ∧
    a = 4.25 ∧ n = 5 := by
  sorry

end gaochun_population_scientific_notation_l1648_164818


namespace congruence_solution_solution_properties_sum_of_solution_l1648_164811

theorem congruence_solution : ∃ (y : ℤ), (10 * y + 3) % 18 = 7 % 18 ∧ y % 9 = 4 % 9 := by
  sorry

theorem solution_properties : 4 < 9 ∧ 9 ≥ 2 := by
  sorry

theorem sum_of_solution : ∃ (a m : ℤ), (10 * a + 3) % 18 = 7 % 18 ∧ a % m = a ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := by
  sorry

end congruence_solution_solution_properties_sum_of_solution_l1648_164811


namespace third_term_of_geometric_sequence_l1648_164892

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 3 → q = -2 → a 3 = 12 := by
  sorry

end third_term_of_geometric_sequence_l1648_164892


namespace brick_length_is_25cm_l1648_164890

/-- Proof that the length of each brick is 25 cm -/
theorem brick_length_is_25cm (wall_length : ℝ) (wall_height : ℝ) (wall_thickness : ℝ)
  (brick_width : ℝ) (brick_height : ℝ) (num_bricks : ℕ) :
  wall_length = 900 →
  wall_height = 600 →
  wall_thickness = 22.5 →
  brick_width = 11.25 →
  brick_height = 6 →
  num_bricks = 7200 →
  ∃ brick_length : ℝ,
    wall_length * wall_height * wall_thickness =
    num_bricks * (brick_length * brick_width * brick_height) ∧
    brick_length = 25 := by
  sorry

end brick_length_is_25cm_l1648_164890


namespace adults_eaten_correct_l1648_164832

/-- Represents the number of adults who had their meal -/
def adults_eaten : ℕ := 42

/-- Represents the total number of adults in the group -/
def total_adults : ℕ := 55

/-- Represents the total number of children in the group -/
def total_children : ℕ := 70

/-- Represents the meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- Represents the meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- Represents the number of children that can be catered with the remaining food -/
def remaining_children : ℕ := 36

theorem adults_eaten_correct : 
  adults_eaten = 42 ∧
  total_adults = 55 ∧
  total_children = 70 ∧
  meal_capacity_adults = 70 ∧
  meal_capacity_children = 90 ∧
  remaining_children = 36 ∧
  meal_capacity_children - (adults_eaten * meal_capacity_children / meal_capacity_adults) = remaining_children :=
by sorry

end adults_eaten_correct_l1648_164832


namespace find_divisor_l1648_164859

theorem find_divisor (n : ℕ) (s : ℕ) (d : ℕ) : 
  n = 724946 →
  s = 6 →
  d ∣ (n - s) →
  (∀ k < s, ¬(d ∣ (n - k))) →
  d = 2 :=
by
  sorry

end find_divisor_l1648_164859


namespace central_angle_regular_octagon_l1648_164840

/-- The central angle of a regular octagon is 45 degrees. -/
theorem central_angle_regular_octagon :
  let total_angle : ℝ := 360
  let num_sides : ℕ := 8
  let central_angle := total_angle / num_sides
  central_angle = 45 := by sorry

end central_angle_regular_octagon_l1648_164840


namespace quadratic_inequality_properties_l1648_164889

/-- Given that the solution set of ax² + bx + c > 0 is {x | -3 < x < 2} -/
def solution_set (a b c : ℝ) : Set ℝ :=
  {x | -3 < x ∧ x < 2 ∧ a * x^2 + b * x + c > 0}

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x | -3 < x ∧ x < 2}) :
  a < 0 ∧
  a + b + c > 0 ∧
  {x | c * x^2 + b * x + a < 0} = {x | -1/3 < x ∧ x < 1/2} := by
  sorry

end quadratic_inequality_properties_l1648_164889


namespace doug_had_22_marbles_l1648_164821

/-- Calculates the initial number of marbles Doug had -/
def dougs_initial_marbles (eds_marbles : ℕ) (difference : ℕ) : ℕ :=
  eds_marbles - difference

theorem doug_had_22_marbles (eds_marbles : ℕ) (difference : ℕ) 
  (h1 : eds_marbles = 27) 
  (h2 : difference = 5) : 
  dougs_initial_marbles eds_marbles difference = 22 := by
sorry

end doug_had_22_marbles_l1648_164821


namespace rectangle_ratio_problem_l1648_164831

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents a regular pentagon with side length a -/
structure Pentagon where
  a : ℝ

/-- Theorem statement for the rectangle ratio problem -/
theorem rectangle_ratio_problem (p : Pentagon) (r : Rectangle) : 
  -- The pentagon is regular and has side length a
  p.a > 0 →
  -- Five congruent rectangles are placed around the pentagon
  -- The shorter side of each rectangle lies against a side of the inner pentagon
  r.y = p.a →
  -- The area of the outer pentagon is 5 times that of the inner pentagon
  -- (We use this as an assumption without deriving it geometrically)
  r.x + r.y = Real.sqrt 5 * p.a →
  -- The ratio of the longer side to the shorter side of each rectangle is √5 - 1
  r.x / r.y = Real.sqrt 5 - 1 := by
  sorry

end rectangle_ratio_problem_l1648_164831


namespace onesDigitOfComplexExpression_l1648_164862

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The given complex expression -/
def complexExpression : ℕ :=
  onesDigit ((73 ^ 1253) * (44 ^ 987) + (47 ^ 123) / (39 ^ 654) * (86 ^ 1484) - (32 ^ 1987) % 10)

/-- Theorem stating that the ones digit of the complex expression is 2 -/
theorem onesDigitOfComplexExpression : complexExpression = 2 := by
  sorry

end onesDigitOfComplexExpression_l1648_164862


namespace congruence_problem_l1648_164875

theorem congruence_problem (x : ℤ) :
  (4 * x + 5) % 20 = 3 → (3 * x + 8) % 10 = 2 := by
  sorry

end congruence_problem_l1648_164875


namespace julio_orange_bottles_l1648_164839

-- Define the number of bottles for each person and soda type
def julio_grape_bottles : ℕ := 7
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3

-- Define the volume of soda per bottle
def liters_per_bottle : ℕ := 2

-- Define the additional amount of soda Julio has compared to Mateo
def julio_extra_liters : ℕ := 14

-- Define a function to calculate the total liters of soda
def total_liters (orange_bottles grape_bottles : ℕ) : ℕ :=
  (orange_bottles + grape_bottles) * liters_per_bottle

-- State the theorem
theorem julio_orange_bottles : 
  ∃ (julio_orange_bottles : ℕ),
    total_liters julio_orange_bottles julio_grape_bottles = 
    total_liters mateo_orange_bottles mateo_grape_bottles + julio_extra_liters ∧
    julio_orange_bottles = 4 := by
  sorry

end julio_orange_bottles_l1648_164839


namespace mans_speed_with_current_l1648_164860

/-- Given a current speed and a speed against the current, calculates the speed with the current -/
def speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem: Given the specified conditions, the man's speed with the current is 15 km/hr -/
theorem mans_speed_with_current :
  let current_speed : ℝ := 2.8
  let speed_against_current : ℝ := 9.4
  speed_with_current current_speed speed_against_current = 15 := by
  sorry

end mans_speed_with_current_l1648_164860


namespace f_is_even_l1648_164876

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_odd g) : ∀ x, f g (-x) = f g x := by
  sorry

end f_is_even_l1648_164876


namespace cook_is_innocent_l1648_164853

-- Define the type for individuals
def Individual : Type := String

-- Define the property of stealing pepper
def stole_pepper (x : Individual) : Prop := sorry

-- Define the property of lying
def always_lies (x : Individual) : Prop := sorry

-- Define the property of knowing who stole the pepper
def knows_thief (x : Individual) : Prop := sorry

-- The cook
def cook : Individual := "Cook"

-- Axiom: Individuals who steal pepper always lie
axiom pepper_thieves_lie : ∀ x : Individual, stole_pepper x → always_lies x

-- Axiom: The cook stated they know who stole the pepper
axiom cook_statement : knows_thief cook

-- Theorem: The cook is innocent (did not steal the pepper)
theorem cook_is_innocent : ¬(stole_pepper cook) := by sorry

end cook_is_innocent_l1648_164853


namespace max_annual_profit_l1648_164852

/-- Represents the annual profit function in million yuan -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    50 * x - (1/3 * x^2 + 10 * x) / 100 - 250
  else
    50 * x - (51 * x + 10000 / x - 1450) / 100 - 250

/-- The maximum annual profit is 1000 million yuan -/
theorem max_annual_profit :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → annual_profit x ≥ annual_profit y ∧ annual_profit x = 1000 :=
sorry

end max_annual_profit_l1648_164852


namespace gem_stone_necklaces_count_l1648_164804

/-- Proves that the number of gem stone necklaces sold is 3, given the conditions of the problem -/
theorem gem_stone_necklaces_count :
  let bead_necklaces : ℕ := 4
  let price_per_necklace : ℕ := 3
  let total_earnings : ℕ := 21
  let gem_stone_necklaces : ℕ := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces = 3 := by sorry

end gem_stone_necklaces_count_l1648_164804


namespace three_digit_number_property_l1648_164817

theorem three_digit_number_property : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ)^2 + ((n / 10) % 10 : ℕ)^2 + (n % 10 : ℕ)^2 ∧
  n = 550 := by
  sorry

end three_digit_number_property_l1648_164817


namespace intermediate_value_theorem_l1648_164887

theorem intermediate_value_theorem 
  {f : ℝ → ℝ} {a b : ℝ} (h₁ : a < b) (h₂ : Continuous f) (h₃ : f a * f b < 0) :
  ∃ c ∈ Set.Ioo a b, f c = 0 :=
sorry

end intermediate_value_theorem_l1648_164887


namespace power_equation_solution_l1648_164878

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^5 * 81^3 → n = 23 := by
  sorry

end power_equation_solution_l1648_164878


namespace angle_product_theorem_l1648_164815

theorem angle_product_theorem (α β : Real) (m : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ y = Real.sqrt 3 * x ∧ x < 0 ∧ y < 0) →  -- condition 1
  ((1/2)^2 + m^2 = 1) →  -- condition 2
  (Real.sin α * Real.cos β < 0) →  -- condition 3
  (Real.cos α * Real.sin β = Real.sqrt 3 / 4 ∨ Real.cos α * Real.sin β = -Real.sqrt 3 / 4) :=
by sorry


end angle_product_theorem_l1648_164815


namespace max_blocks_is_twelve_l1648_164822

/-- A block covers exactly two cells -/
structure Block where
  cells : Fin 16 → Fin 16
  covers_two : ∃ (c1 c2 : Fin 16), c1 ≠ c2 ∧ (∀ c, cells c = c1 ∨ cells c = c2)

/-- Configuration of blocks on a 4x4 grid -/
structure Configuration where
  blocks : List Block
  all_cells_covered : ∀ c : Fin 16, ∃ b ∈ blocks, ∃ c', b.cells c' = c
  removal_uncovers : ∀ b ∈ blocks, ∃ c : Fin 16, (∀ b' ∈ blocks, b' ≠ b → ∀ c', b'.cells c' ≠ c)

/-- The maximum number of blocks in a valid configuration -/
def max_blocks : ℕ := 12

/-- The theorem stating that 12 is the maximum number of blocks -/
theorem max_blocks_is_twelve :
  ∀ cfg : Configuration, cfg.blocks.length ≤ max_blocks :=
sorry

end max_blocks_is_twelve_l1648_164822


namespace japanese_students_count_l1648_164824

theorem japanese_students_count (chinese : ℕ) (korean : ℕ) (japanese : ℕ) 
  (h1 : korean = (6 * chinese) / 11)
  (h2 : japanese = chinese / 8)
  (h3 : korean = 48) : 
  japanese = 11 := by
sorry

end japanese_students_count_l1648_164824


namespace cos_2alpha_plus_4pi_3_l1648_164882

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) :
  Real.cos (2 * α + 4 * Real.pi / 3) = -7/8 := by
  sorry

end cos_2alpha_plus_4pi_3_l1648_164882


namespace solve_system_l1648_164844

theorem solve_system (x y : ℝ) :
  (x / 6) * 12 = 10 ∧ (y / 4) * 8 = x → x = 5 ∧ y = (5 / 2) :=
by sorry

end solve_system_l1648_164844


namespace good_apples_count_l1648_164881

theorem good_apples_count (total : ℕ) (unripe : ℕ) (h1 : total = 14) (h2 : unripe = 6) :
  total - unripe = 8 := by
sorry

end good_apples_count_l1648_164881


namespace chess_tournament_games_l1648_164895

/-- The number of games in a chess tournament --/
def num_games (n : ℕ) : ℕ :=
  3 * (n * (n - 1) / 2)

/-- Theorem: In a chess tournament with 35 players, where each player plays
    three times with every other player, the total number of games is 1785 --/
theorem chess_tournament_games :
  num_games 35 = 1785 := by
  sorry


end chess_tournament_games_l1648_164895


namespace value_difference_reciprocals_l1648_164888

theorem value_difference_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x - y = x / y) : 1 / x - 1 / y = -1 / y^2 := by
  sorry

end value_difference_reciprocals_l1648_164888


namespace real_equal_roots_condition_l1648_164819

theorem real_equal_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 12 = 0 → y = x) ↔ 
  (k = -10 ∨ k = 14) := by sorry

end real_equal_roots_condition_l1648_164819


namespace amount_ratio_l1648_164851

theorem amount_ratio (total : ℚ) (b_amt : ℚ) (a_fraction : ℚ) :
  total = 1440 →
  b_amt = 270 →
  a_fraction = 1/3 →
  ∃ (c_amt : ℚ),
    total = a_fraction * b_amt + b_amt + c_amt ∧
    b_amt / c_amt = 1/4 :=
by sorry

end amount_ratio_l1648_164851


namespace jawbreakers_eaten_l1648_164830

def package_size : ℕ := 8
def jawbreakers_left : ℕ := 4

theorem jawbreakers_eaten : ℕ := by
  sorry

end jawbreakers_eaten_l1648_164830


namespace correct_match_probability_l1648_164877

/-- The probability of correctly matching all items when one match is known --/
theorem correct_match_probability (n : ℕ) (h : n = 4) : 
  (1 : ℚ) / Nat.factorial (n - 1) = (1 : ℚ) / 6 :=
sorry

end correct_match_probability_l1648_164877


namespace arithmetic_sequences_ratio_l1648_164842

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio :
  let seq1_sum := arithmetic_sum 4 4 68
  let seq2_sum := arithmetic_sum 5 5 85
  seq1_sum / seq2_sum = 4 / 5 := by
sorry

end arithmetic_sequences_ratio_l1648_164842


namespace fraction_of_66_l1648_164848

theorem fraction_of_66 (x : ℚ) (h : x = 22.142857142857142) : 
  ((((x + 5) * 7) / 5) - 5) = 66 * (1 / 2) := by
  sorry

end fraction_of_66_l1648_164848


namespace triangle_abc_properties_l1648_164803

theorem triangle_abc_properties (a b c A B C : Real) :
  -- Given conditions
  (2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C) →
  (b = 2 * Real.sqrt 3) →
  (A = π / 4) →
  -- Conclusions
  (B = 2 * π / 3) ∧
  (1/2 * b * c * Real.sin A = (3 - Real.sqrt 3) / 2) :=
by sorry

end triangle_abc_properties_l1648_164803


namespace hexagon_area_fraction_l1648_164850

/-- Represents a tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller units in one side of a large square -/
  units_per_side : ℕ
  /-- The number of units occupied by hexagons in a large square -/
  hexagon_units : ℕ

/-- The fraction of the plane enclosed by hexagons -/
def hexagon_fraction (pattern : TilingPattern) : ℚ :=
  pattern.hexagon_units / (pattern.units_per_side ^ 2 : ℚ)

/-- The specific tiling pattern described in the problem -/
def problem_pattern : TilingPattern :=
  { units_per_side := 4
  , hexagon_units := 8 }

theorem hexagon_area_fraction :
  hexagon_fraction problem_pattern = 1/2 := by sorry

end hexagon_area_fraction_l1648_164850


namespace imaginary_part_reciprocal_l1648_164805

theorem imaginary_part_reciprocal (a : ℝ) (h1 : a > 0) :
  let z : ℂ := a + Complex.I
  (Complex.abs z = Real.sqrt 5) →
  Complex.im (z⁻¹) = -1/5 :=
by sorry

end imaginary_part_reciprocal_l1648_164805


namespace largest_prime_factor_of_expression_l1648_164858

theorem largest_prime_factor_of_expression : 
  let n : ℤ := 20^3 + 15^4 - 10^5 + 5^6
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ p = 103 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p :=
by sorry

end largest_prime_factor_of_expression_l1648_164858


namespace last_three_digits_of_7_to_103_l1648_164814

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 327 := by
  sorry

end last_three_digits_of_7_to_103_l1648_164814


namespace susna_class_f_fraction_l1648_164898

/-- Represents the fractions of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ

/-- The conditions of the problem -/
def susna_class : GradeDistribution where
  a := 1/4
  b := 1/2
  c := 1/8
  d := 1/12
  f := 0 -- We'll prove this is actually 1/24

theorem susna_class_f_fraction :
  let g := susna_class
  (g.a + g.b + g.c = 7/8) →
  (g.a + g.b + g.c = 0.875) →
  (g.a + g.b + g.c + g.d + g.f = 1) →
  g.f = 1/24 := by sorry

end susna_class_f_fraction_l1648_164898


namespace triangle_property_l1648_164810

open Real

theorem triangle_property (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) 
  (hABC : A + B + C = π) (hSin : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * π / 3 ∧ 
  (∃ (a b c : ℝ), a = 3 ∧ 
    sin A / a = sin B / b ∧ 
    sin A / a = sin C / c ∧ 
    a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

#check triangle_property

end triangle_property_l1648_164810


namespace relationship_uvwt_l1648_164869

theorem relationship_uvwt (m p r s : ℝ) (u v w t : ℝ) 
  (h1 : m^u = p^v) (h2 : p^v = r) (h3 : p^w = m^t) (h4 : m^t = s) :
  u * v = w * t := by
  sorry

end relationship_uvwt_l1648_164869


namespace function_inequality_proof_l1648_164894

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

def solution_set_A : Set ℝ := {x | x < -1 ∨ x > 2}

def solution_set_B (a : ℝ) : Set ℝ := {x | f x > |a - 1|}

theorem function_inequality_proof :
  (∀ x, f x > 6 ↔ x ∈ solution_set_A) ∧
  (∀ a, solution_set_B a ⊆ solution_set_A ↔ a ≤ -5 ∨ a ≥ 7) :=
sorry

end function_inequality_proof_l1648_164894


namespace smallest_number_of_cubes_is_56_l1648_164861

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 56 -/
theorem smallest_number_of_cubes_is_56 :
  smallestNumberOfCubes ⟨35, 20, 10⟩ = 56 := by
  sorry

#eval smallestNumberOfCubes ⟨35, 20, 10⟩

end smallest_number_of_cubes_is_56_l1648_164861


namespace jim_ran_16_miles_l1648_164849

/-- The number of miles Jim ran in 2 hours -/
def jim_miles : ℝ := 16

/-- The number of hours Jim ran -/
def jim_hours : ℝ := 2

/-- The number of miles Frank ran in 2 hours -/
def frank_miles : ℝ := 20

/-- The difference in miles between Frank and Jim in one hour -/
def miles_difference : ℝ := 2

theorem jim_ran_16_miles :
  jim_miles = 16 ∧
  jim_hours = 2 ∧
  frank_miles = 20 ∧
  miles_difference = 2 →
  jim_miles = 16 :=
by sorry

end jim_ran_16_miles_l1648_164849


namespace smallest_n_congruence_l1648_164866

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k : ℕ, 0 < k ∧ k < 5 → ¬(1031 * k ≡ 1067 * k [ZMOD 30])) ∧ 
  (1031 * 5 ≡ 1067 * 5 [ZMOD 30]) := by
  sorry

end smallest_n_congruence_l1648_164866


namespace ninety_nine_in_third_column_l1648_164870

/-- Represents the column number (1 to 5) in the arrangement -/
inductive Column
  | one
  | two
  | three
  | four
  | five

/-- Function that determines the column for a given odd number -/
def columnForOddNumber (n : ℕ) : Column :=
  match n % 5 with
  | 1 => Column.one
  | 2 => Column.four
  | 3 => Column.two
  | 4 => Column.five
  | 0 => Column.three
  | _ => Column.one  -- This case should never occur for odd numbers

theorem ninety_nine_in_third_column :
  columnForOddNumber 99 = Column.three :=
sorry

end ninety_nine_in_third_column_l1648_164870


namespace lindsey_owns_four_more_cars_l1648_164843

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  carol : ℕ
  susan : ℕ
  lindsey : ℕ

/-- The conditions of the car ownership problem --/
def carProblemConditions (co : CarOwnership) : Prop :=
  co.cathy = 5 ∧
  co.carol = 2 * co.cathy ∧
  co.susan = co.carol - 2 ∧
  co.lindsey > co.cathy ∧
  co.cathy + co.carol + co.susan + co.lindsey = 32

/-- The theorem stating that Lindsey owns 4 more cars than Cathy --/
theorem lindsey_owns_four_more_cars (co : CarOwnership) 
  (h : carProblemConditions co) : co.lindsey - co.cathy = 4 := by
  sorry

end lindsey_owns_four_more_cars_l1648_164843


namespace determinant_special_matrix_l1648_164808

theorem determinant_special_matrix (a y : ℝ) : 
  Matrix.det !![a, y, y; y, a, y; y, y, a] = a^3 - 2*a*y^2 + 2*y^3 := by
  sorry

end determinant_special_matrix_l1648_164808


namespace min_digit_sum_of_sum_l1648_164871

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the sum of digits of a natural number -/
def digitSum (n : Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem min_digit_sum_of_sum (a b : ThreeDigitNumber) 
  (h1 : a.hundreds < 5)
  (h2 : a.hundreds ≠ b.hundreds ∧ a.hundreds ≠ b.tens ∧ a.hundreds ≠ b.ones ∧
        a.tens ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.tens ≠ b.ones ∧
        a.ones ≠ b.hundreds ∧ a.ones ≠ b.tens ∧ a.ones ≠ b.ones)
  (h3 : (a.value + b.value) < 1000) :
  15 ≤ digitSum (a.value + b.value) :=
sorry

end min_digit_sum_of_sum_l1648_164871


namespace tim_keys_needed_l1648_164893

/-- Calculates the total number of keys needed for apartment complexes -/
def total_keys (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_lock : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_lock

/-- Proves that for Tim's specific case, the total number of keys needed is 72 -/
theorem tim_keys_needed :
  total_keys 2 12 3 = 72 := by
  sorry

end tim_keys_needed_l1648_164893


namespace compound_has_one_Al_l1648_164864

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- A compound with Aluminium and Iodine -/
structure Compound where
  Al_count : ℕ
  I_count : ℕ
  molecular_weight : ℝ

/-- The compound in question -/
def our_compound : Compound where
  Al_count := 1
  I_count := 3
  molecular_weight := 408

/-- Theorem stating that our compound has exactly 1 Aluminium atom -/
theorem compound_has_one_Al : 
  our_compound.Al_count = 1 ∧
  our_compound.I_count = 3 ∧
  our_compound.molecular_weight = 408 ∧
  (our_compound.Al_count : ℝ) * atomic_weight_Al + (our_compound.I_count : ℝ) * atomic_weight_I = our_compound.molecular_weight :=
by sorry

end compound_has_one_Al_l1648_164864


namespace fold_paper_l1648_164897

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is perpendicular bisector of two points -/
def isPerpBisector (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.y = l.slope * midpoint.x + l.intercept) ∧
  (l.slope * (p2.x - p1.x) = -(p2.y - p1.y))

/-- Function to check if two points are symmetric about a line -/
def areSymmetric (l : Line) (p1 p2 : Point) : Prop :=
  isPerpBisector l p1 p2

/-- Main theorem -/
theorem fold_paper (l : Line) (p1 p2 p3 : Point) (p q : ℝ) :
  areSymmetric l ⟨1, 3⟩ ⟨5, 1⟩ →
  areSymmetric l ⟨8, 4⟩ ⟨p, q⟩ →
  p + q = 8 := by
  sorry

end fold_paper_l1648_164897
