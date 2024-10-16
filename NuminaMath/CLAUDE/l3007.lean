import Mathlib

namespace NUMINAMATH_CALUDE_andy_twice_rahims_age_l3007_300790

def rahims_current_age : ℕ := 6
def andys_age_difference : ℕ := 1

theorem andy_twice_rahims_age (x : ℕ) : 
  (rahims_current_age + andys_age_difference + x = 2 * rahims_current_age) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_andy_twice_rahims_age_l3007_300790


namespace NUMINAMATH_CALUDE_gold_coin_count_l3007_300706

theorem gold_coin_count (c n : ℕ) (h1 : n = 8 * (c - 3))
  (h2 : n = 5 * c + 4) (h3 : c ≥ 10) : n = 54 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_count_l3007_300706


namespace NUMINAMATH_CALUDE_natural_number_pairs_l3007_300791

theorem natural_number_pairs : 
  ∀ a b : ℕ, 
    (90 < a + b ∧ a + b < 100) ∧ 
    (0.9 < (a : ℝ) / (b : ℝ) ∧ (a : ℝ) / (b : ℝ) < 0.91) → 
    ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l3007_300791


namespace NUMINAMATH_CALUDE_roots_problem_l3007_300701

theorem roots_problem :
  (∀ x : ℝ, x > 0 → x^2 = 1/16 → x = 1/4) ∧
  (∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -8 → x = -2) := by
sorry

end NUMINAMATH_CALUDE_roots_problem_l3007_300701


namespace NUMINAMATH_CALUDE_pie_to_bar_representation_l3007_300772

-- Define the structure of a pie chart
structure PieChart :=
  (section1 : ℝ)
  (section2 : ℝ)
  (section3 : ℝ)

-- Define the structure of a bar graph
structure BarGraph :=
  (bar1 : ℝ)
  (bar2 : ℝ)
  (bar3 : ℝ)

-- Define the conditions of the pie chart
def validPieChart (p : PieChart) : Prop :=
  p.section1 = p.section2 ∧ p.section3 = p.section1 + p.section2

-- Define the correct bar graph representation
def correctBarGraph (p : PieChart) (b : BarGraph) : Prop :=
  b.bar1 = b.bar2 ∧ b.bar3 = b.bar1 + b.bar2

-- Theorem: For a valid pie chart, there exists a correct bar graph representation
theorem pie_to_bar_representation (p : PieChart) (h : validPieChart p) :
  ∃ b : BarGraph, correctBarGraph p b :=
sorry

end NUMINAMATH_CALUDE_pie_to_bar_representation_l3007_300772


namespace NUMINAMATH_CALUDE_chocolate_cost_l3007_300750

theorem chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) :
  candies_per_box = 25 →
  cost_per_box = 6 →
  total_candies = 600 →
  (total_candies / candies_per_box) * cost_per_box = 144 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_cost_l3007_300750


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3007_300736

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(2*x + 2*y) = 2) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → (2 : ℝ)^(2*a + 2*b) = 2 → 1/a + 1/b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3007_300736


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_and_range_l3007_300740

noncomputable section

open Real

-- Define f(x) = ln x
def f (x : ℝ) : ℝ := log x

-- Define g(x) = f(x) + f''(x)
def g (x : ℝ) : ℝ := f x + (deriv^[2] f) x

theorem tangent_line_and_monotonicity_and_range :
  -- 1. The tangent line to y = f(x) at (1, f(1)) is y = x - 1
  (∀ y, y = deriv f 1 * (x - 1) + f 1 ↔ y = x - 1) ∧
  -- 2. g(x) is decreasing on (0, 1) and increasing on (1, +∞)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → g x₁ > g x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → g x₁ < g x₂) ∧
  -- 3. For any x > 0, g(a) - g(x) < 1/a holds if and only if 0 < a < e
  (∀ a, (0 < a ∧ a < ℯ) ↔ (∀ x, x > 0 → g a - g x < 1 / a)) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_and_range_l3007_300740


namespace NUMINAMATH_CALUDE_valid_pairs_count_l3007_300737

def is_valid_pair (x y : ℕ) : Prop :=
  x ≠ y ∧
  x < 10 ∧ y < 10 ∧
  let product := (x * 1111) * (y * 1111)
  product ≥ 1000000 ∧ product < 10000000 ∧
  product % 10 = x ∧ (product / 1000000) % 10 = x

theorem valid_pairs_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, is_valid_pair p.1 p.2) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l3007_300737


namespace NUMINAMATH_CALUDE_solution_equality_l3007_300710

theorem solution_equality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_solution_equality_l3007_300710


namespace NUMINAMATH_CALUDE_pens_bought_theorem_l3007_300727

/-- The number of pens bought at the cost price -/
def num_pens_bought : ℕ := 17

/-- The number of pens sold to equal the cost price of the bought pens -/
def num_pens_sold : ℕ := 12

/-- The gain percentage -/
def gain_percentage : ℚ := 40/100

theorem pens_bought_theorem :
  ∀ (cost_price selling_price : ℚ),
  cost_price > 0 →
  selling_price > 0 →
  (num_pens_bought : ℚ) * cost_price = (num_pens_sold : ℚ) * selling_price →
  (selling_price - cost_price) / cost_price = gain_percentage →
  num_pens_bought = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_bought_theorem_l3007_300727


namespace NUMINAMATH_CALUDE_tribe_organization_ways_l3007_300726

/-- The number of members in the tribe -/
def tribeSize : ℕ := 13

/-- The number of supporting chiefs -/
def numSupportingChiefs : ℕ := 3

/-- The number of inferiors for each supporting chief -/
def numInferiors : ℕ := 2

/-- Calculate the number of ways to organize the tribe's leadership -/
def organizationWays : ℕ := 
  tribeSize * (tribeSize - 1) * (tribeSize - 2) * (tribeSize - 3) * 
  Nat.choose (tribeSize - 4) 2 * 
  Nat.choose (tribeSize - 6) 2 * 
  Nat.choose (tribeSize - 8) 2

/-- Theorem stating that the number of ways to organize the leadership is 12355200 -/
theorem tribe_organization_ways : organizationWays = 12355200 := by
  sorry

end NUMINAMATH_CALUDE_tribe_organization_ways_l3007_300726


namespace NUMINAMATH_CALUDE_mask_usage_duration_l3007_300702

theorem mask_usage_duration (total_masks : ℕ) (family_members : ℕ) (total_days : ℕ) 
  (h1 : total_masks = 100)
  (h2 : family_members = 5)
  (h3 : total_days = 80) :
  (total_masks : ℚ) / total_days / family_members = 1 / 4 := by
  sorry

#check mask_usage_duration

end NUMINAMATH_CALUDE_mask_usage_duration_l3007_300702


namespace NUMINAMATH_CALUDE_parabola_points_condition_l3007_300709

/-- The parabola equation -/
def parabola (x y k : ℝ) : Prop := y = -2 * (x - 1)^2 + k

theorem parabola_points_condition (m y₁ y₂ k : ℝ) :
  parabola (m - 1) y₁ k →
  parabola m y₂ k →
  y₁ > y₂ →
  m > 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_points_condition_l3007_300709


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3007_300756

theorem only_negative_three_less_than_negative_two :
  let numbers : List ℚ := [-3, -1/2, 0, 2]
  ∀ x ∈ numbers, x < -2 ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3007_300756


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l3007_300738

theorem no_root_greater_than_three : 
  ¬∃ x : ℝ, (x > 3 ∧ 
    ((3 * x^2 - 2 = 25) ∨ 
     ((2*x-1)^2 = (x-1)^2) ∨ 
     (x^2 - 7 = x - 1 ∧ x ≥ 1))) := by
  sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l3007_300738


namespace NUMINAMATH_CALUDE_fenced_area_with_cutouts_l3007_300734

/-- The area of a fenced region with cutouts -/
theorem fenced_area_with_cutouts :
  let rectangle_length : ℝ := 20
  let rectangle_width : ℝ := 18
  let square_side : ℝ := 4
  let triangle_leg : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let square_cutout_area := square_side * square_side
  let triangle_cutout_area := (1 / 2) * triangle_leg * triangle_leg
  rectangle_area - square_cutout_area - triangle_cutout_area = 339.5 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_with_cutouts_l3007_300734


namespace NUMINAMATH_CALUDE_S_is_empty_l3007_300754

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ r : ℝ, (2 + 5*i)*z = r ∧ z.re = 2*z.im}

-- Theorem statement
theorem S_is_empty : S = ∅ := by sorry

end NUMINAMATH_CALUDE_S_is_empty_l3007_300754


namespace NUMINAMATH_CALUDE_tony_haircut_distance_l3007_300711

theorem tony_haircut_distance (total_distance halfway_distance groceries_distance doctor_distance : ℕ)
  (h1 : total_distance = 2 * halfway_distance)
  (h2 : halfway_distance = 15)
  (h3 : groceries_distance = 10)
  (h4 : doctor_distance = 5) :
  total_distance - (groceries_distance + doctor_distance) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tony_haircut_distance_l3007_300711


namespace NUMINAMATH_CALUDE_chen_pushups_l3007_300735

/-- The number of push-ups done by Chen -/
def chen : ℕ := sorry

/-- The number of push-ups done by Ruan -/
def ruan : ℕ := sorry

/-- The number of push-ups done by Lu -/
def lu : ℕ := sorry

/-- The number of push-ups done by Tao -/
def tao : ℕ := sorry

/-- The number of push-ups done by Yang -/
def yang : ℕ := sorry

/-- Chen, Lu, and Yang together averaged 40 push-ups per person -/
axiom condition1 : chen + lu + yang = 40 * 3

/-- Ruan, Tao, and Chen together averaged 28 push-ups per person -/
axiom condition2 : ruan + tao + chen = 28 * 3

/-- Ruan, Lu, Tao, and Yang together averaged 33 push-ups per person -/
axiom condition3 : ruan + lu + tao + yang = 33 * 4

theorem chen_pushups : chen = 36 := by
  sorry

end NUMINAMATH_CALUDE_chen_pushups_l3007_300735


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l3007_300749

theorem min_distance_circle_to_line : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 15 = 0}
  ∃ d : ℝ, d = 2 ∧ 
    ∀ p ∈ circle, ∀ q ∈ line, 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ circle, ∃ q' ∈ line, 
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_circle_to_line_l3007_300749


namespace NUMINAMATH_CALUDE_fibonacci_closed_form_l3007_300755

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_closed_form_l3007_300755


namespace NUMINAMATH_CALUDE_average_value_theorem_l3007_300770

/-- The average value of the set {0, 2z, 4z, 8z, 16z} is 6z -/
theorem average_value_theorem (z : ℝ) : 
  (0 + 2*z + 4*z + 8*z + 16*z) / 5 = 6*z := by
  sorry

end NUMINAMATH_CALUDE_average_value_theorem_l3007_300770


namespace NUMINAMATH_CALUDE_seven_dots_max_regions_l3007_300748

/-- The maximum number of regions formed by connecting n dots on a circle's circumference --/
def max_regions (n : ℕ) : ℕ :=
  1 + (n.choose 2) + (n.choose 4)

/-- Theorem: For 7 dots on a circle's circumference, the maximum number of regions is 57 --/
theorem seven_dots_max_regions :
  max_regions 7 = 57 := by
  sorry

end NUMINAMATH_CALUDE_seven_dots_max_regions_l3007_300748


namespace NUMINAMATH_CALUDE_five_digit_multiplication_reversal_l3007_300752

theorem five_digit_multiplication_reversal :
  ∃! (a b c d e : ℕ),
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    0 ≤ e ∧ e ≤ 9 ∧
    a ≠ 0 ∧
    (10000 * a + 1000 * b + 100 * c + 10 * d + e) * 9 =
    10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
    a = 1 ∧ b = 0 ∧ c = 9 ∧ d = 8 ∧ e = 9 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiplication_reversal_l3007_300752


namespace NUMINAMATH_CALUDE_archer_probabilities_l3007_300712

/-- Represents the probability of an archer hitting a target -/
def hit_probability : ℚ := 2/3

/-- Represents the number of shots taken -/
def num_shots : ℕ := 5

/-- Calculates the probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- Calculates the probability of hitting the target k times in a row and missing n-k times in n shots -/
def prob_consecutive_hits (n k : ℕ) : ℚ :=
  (n - k + 1 : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

theorem archer_probabilities :
  (prob_exact_hits num_shots 2 = 40/243) ∧
  (prob_consecutive_hits num_shots 3 = 8/81) := by
  sorry

end NUMINAMATH_CALUDE_archer_probabilities_l3007_300712


namespace NUMINAMATH_CALUDE_system_1_solution_l3007_300763

theorem system_1_solution (x y : ℝ) :
  x = y + 1 ∧ 4 * x - 3 * y = 5 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_l3007_300763


namespace NUMINAMATH_CALUDE_can_collection_difference_l3007_300720

/-- Theorem: Difference in can collection between two days -/
theorem can_collection_difference
  (sarah_yesterday : ℝ)
  (lara_yesterday : ℝ)
  (alex_yesterday : ℝ)
  (sarah_today : ℝ)
  (lara_today : ℝ)
  (alex_today : ℝ)
  (h1 : sarah_yesterday = 50.5)
  (h2 : lara_yesterday = sarah_yesterday + 30.3)
  (h3 : alex_yesterday = 90.2)
  (h4 : sarah_today = 40.7)
  (h5 : lara_today = 70.5)
  (h6 : alex_today = 55.3) :
  (sarah_yesterday + lara_yesterday + alex_yesterday) -
  (sarah_today + lara_today + alex_today) = 55 := by
  sorry

end NUMINAMATH_CALUDE_can_collection_difference_l3007_300720


namespace NUMINAMATH_CALUDE_equal_book_distribution_l3007_300713

theorem equal_book_distribution (total_students : ℕ) (girls : ℕ) (boys : ℕ) 
  (total_books : ℕ) (girls_books : ℕ) :
  total_students = girls + boys →
  total_books = 375 →
  girls = 15 →
  boys = 10 →
  girls_books = 225 →
  ∃ (books_per_student : ℕ), 
    books_per_student = 15 ∧
    girls_books = girls * books_per_student ∧
    total_books = total_students * books_per_student :=
by sorry

end NUMINAMATH_CALUDE_equal_book_distribution_l3007_300713


namespace NUMINAMATH_CALUDE_system_solution_in_first_quadrant_l3007_300766

theorem system_solution_in_first_quadrant (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 2 ∧ c * x + y = 3) ↔ -1 < c ∧ c < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_in_first_quadrant_l3007_300766


namespace NUMINAMATH_CALUDE_horner_rule_example_l3007_300798

def horner_polynomial (x : ℝ) : ℝ :=
  (((((x + 2) * x) * x - 3) * x + 7) * x - 2)

theorem horner_rule_example :
  horner_polynomial 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_example_l3007_300798


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l3007_300731

theorem tens_digit_of_3_to_2023 : ∃ n : ℕ, 3^2023 ≡ 20 + n [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l3007_300731


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3007_300707

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  S = if a > 0 then
        {x : ℝ | x < -a/4 ∨ x > a/3}
      else if a = 0 then
        {x : ℝ | x ≠ 0}
      else
        {x : ℝ | x < a/3 ∨ x > -a/4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3007_300707


namespace NUMINAMATH_CALUDE_simplify_expressions_l3007_300774

theorem simplify_expressions :
  ((-4 : ℝ)^2023 * (-0.25)^2024 = -0.25) ∧
  (23 * (-4/11 : ℝ) + (-5/11) * 23 - 23 * (2/11) = -23) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3007_300774


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3007_300725

/-- A polygon with interior angle sum of 1800 degrees has 9 diagonals from one vertex -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3007_300725


namespace NUMINAMATH_CALUDE_only_f₁_is_quadratic_l3007_300757

-- Define the four functions
def f₁ (x : ℝ) : ℝ := -3 * x^2
def f₂ (x : ℝ) : ℝ := 2 * x
def f₃ (x : ℝ) : ℝ := x + 1
def f₄ (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- State the theorem
theorem only_f₁_is_quadratic :
  is_quadratic f₁ ∧ ¬is_quadratic f₂ ∧ ¬is_quadratic f₃ ∧ ¬is_quadratic f₄ :=
sorry

end NUMINAMATH_CALUDE_only_f₁_is_quadratic_l3007_300757


namespace NUMINAMATH_CALUDE_carpet_square_cost_l3007_300723

/-- Calculates the cost of each carpet square given the floor dimensions, carpet square dimensions, and total cost. -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : square_side = 8)
  (h4 : total_cost = 576)
  : (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 24 :=
by
  sorry

#check carpet_square_cost

end NUMINAMATH_CALUDE_carpet_square_cost_l3007_300723


namespace NUMINAMATH_CALUDE_triangle_law_of_sines_l3007_300795

theorem triangle_law_of_sines (A B C : Real) (a b c : Real) :
  A = π / 6 →
  a = Real.sqrt 2 →
  b / Real.sin B = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_law_of_sines_l3007_300795


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l3007_300778

def arrange_books (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

theorem book_arrangement_proof :
  arrange_books 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l3007_300778


namespace NUMINAMATH_CALUDE_distance_to_x_axis_for_point_p_l3007_300704

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- Theorem: The distance from point P(3, -2) to the x-axis is 2 --/
theorem distance_to_x_axis_for_point_p :
  distanceToXAxis 3 (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_for_point_p_l3007_300704


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_of_2_52_l3007_300759

def decimal_to_fraction (d : ℚ) : ℤ × ℤ :=
  let n := d.num
  let d := d.den
  let g := n.gcd d
  (n / g, d / g)

theorem sum_of_fraction_parts_of_2_52 :
  let (n, d) := decimal_to_fraction (252 / 100)
  n + d = 88 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_of_2_52_l3007_300759


namespace NUMINAMATH_CALUDE_floor_sqrt_75_l3007_300729

theorem floor_sqrt_75 : ⌊Real.sqrt 75⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_75_l3007_300729


namespace NUMINAMATH_CALUDE_square_sum_value_l3007_300730

theorem square_sum_value (m n : ℝ) :
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0 →
  m^2 + 3*n^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3007_300730


namespace NUMINAMATH_CALUDE_sum_of_palindromic_primes_less_than_70_l3007_300781

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def isPalindromicPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 70 ∧ isPrime n ∧ isPrime (reverseDigits n) ∧ reverseDigits n < 70

def sumOfPalindromicPrimes : ℕ := sorry

theorem sum_of_palindromic_primes_less_than_70 :
  sumOfPalindromicPrimes = 92 := by sorry

end NUMINAMATH_CALUDE_sum_of_palindromic_primes_less_than_70_l3007_300781


namespace NUMINAMATH_CALUDE_quadratic_roots_root_of_two_two_as_only_root_l3007_300794

/-- The quadratic equation x^2 - 2px + q = 0 -/
def quadratic_equation (p q x : ℝ) : Prop :=
  x^2 - 2*p*x + q = 0

/-- The discriminant of the quadratic equation -/
def discriminant (p q : ℝ) : ℝ :=
  4*p^2 - 4*q

theorem quadratic_roots (p q : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation p q x ∧ quadratic_equation p q y) ↔ q < p^2 :=
sorry

theorem root_of_two (p q : ℝ) :
  quadratic_equation p q 2 ↔ q = 4*p - 4 :=
sorry

theorem two_as_only_root (p q : ℝ) :
  (∀ x : ℝ, quadratic_equation p q x ↔ x = 2) ↔ (p = 2 ∧ q = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_root_of_two_two_as_only_root_l3007_300794


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_triangle_l3007_300747

/-- Given a quadrilateral inscribed in a circle of radius R, with points P, Q, and M as described,
    and distances a, b, and c from these points to the circle's center,
    prove that the sides of triangle PQM have the given lengths. -/
theorem inscribed_quadrilateral_triangle (R a b c : ℝ) (h_pos : R > 0) :
  ∃ (PQ QM PM : ℝ),
    PQ = Real.sqrt (a^2 + b^2 - 2*R^2) ∧
    QM = Real.sqrt (b^2 + c^2 - 2*R^2) ∧
    PM = Real.sqrt (c^2 + a^2 - 2*R^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_triangle_l3007_300747


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l3007_300784

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l3007_300784


namespace NUMINAMATH_CALUDE_max_value_ab_l3007_300796

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 / ((2 * a + b) * b)) + (2 / ((2 * b + a) * a)) = 1) :
  ab ≤ 2 - (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (1 / ((2 * a₀ + b₀) * b₀)) + (2 / ((2 * b₀ + a₀) * a₀)) = 1 ∧
    a₀ * b₀ = 2 - (2 * Real.sqrt 2) / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_ab_l3007_300796


namespace NUMINAMATH_CALUDE_car_distance_theorem_l3007_300780

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours. -/
def total_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + (initial_speed + h * speed_increase)) 0

/-- Theorem stating that a car with initial speed 50 km/h, increasing by 2 km/h each hour, travels 732 km in 12 hours. -/
theorem car_distance_theorem : total_distance 50 2 12 = 732 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l3007_300780


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l3007_300724

theorem largest_x_satisfying_equation : ∃ (x : ℝ), 
  (∀ y : ℝ, ⌊y⌋ / y = 8 / 9 → y ≤ x) ∧ 
  ⌊x⌋ / x = 8 / 9 ∧ 
  x = 63 / 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l3007_300724


namespace NUMINAMATH_CALUDE_adam_katie_miles_difference_l3007_300732

/-- Proves that Adam ran 25 miles more than Katie -/
theorem adam_katie_miles_difference :
  let adam_miles : ℕ := 35
  let katie_miles : ℕ := 10
  adam_miles - katie_miles = 25 := by
  sorry

end NUMINAMATH_CALUDE_adam_katie_miles_difference_l3007_300732


namespace NUMINAMATH_CALUDE_shara_borrowed_time_l3007_300714

/-- Represents the problem of determining when Shara borrowed money. -/
theorem shara_borrowed_time (monthly_repayment : ℕ) (total_borrowed : ℕ) : 
  monthly_repayment = 10 →
  total_borrowed / 2 = monthly_repayment * 6 →
  total_borrowed / 2 - monthly_repayment * 4 = 20 →
  6 = total_borrowed / (2 * monthly_repayment) := by
  sorry

end NUMINAMATH_CALUDE_shara_borrowed_time_l3007_300714


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l3007_300762

theorem imaginary_part_of_complex (z : ℂ) : z = 1 - 2*I → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l3007_300762


namespace NUMINAMATH_CALUDE_five_percent_of_255_l3007_300765

theorem five_percent_of_255 : 
  let percent_5 : ℝ := 0.05
  255 * percent_5 = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_five_percent_of_255_l3007_300765


namespace NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l3007_300741

/-- Given an equation (x+B)(Ax+40) / ((x+C)(x+8)) = 3, this theorem proves that
    for specific values of A, B, and C, the equation has infinitely many solutions,
    and provides the sum of x values that do not satisfy the equation. -/
theorem infinite_solutions_and_sum_of_exceptions :
  ∃ (A B C : ℚ),
    (A = 3 ∧ B = 8 ∧ C = 40/3) ∧
    (∀ x : ℚ, x ≠ -C → x ≠ -8 →
      (x + B) * (A * x + 40) / ((x + C) * (x + 8)) = 3) ∧
    ((-8) + (-40/3) = -64/3) := by
  sorry


end NUMINAMATH_CALUDE_infinite_solutions_and_sum_of_exceptions_l3007_300741


namespace NUMINAMATH_CALUDE_square_difference_pattern_l3007_300722

theorem square_difference_pattern (n : ℕ) (h : n ≥ 1) :
  (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l3007_300722


namespace NUMINAMATH_CALUDE_points_to_win_match_jeff_tennis_points_l3007_300787

/-- Calculates the number of points needed to win a tennis match given the total playing time,
    point scoring interval, and number of games won. -/
theorem points_to_win_match 
  (total_time : ℕ) 
  (point_interval : ℕ) 
  (games_won : ℕ) : ℕ :=
  let total_minutes := total_time * 60
  let total_points := total_minutes / point_interval
  total_points / games_won

/-- Proves that 8 points are needed to win a match given the specific conditions. -/
theorem jeff_tennis_points : points_to_win_match 2 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_points_to_win_match_jeff_tennis_points_l3007_300787


namespace NUMINAMATH_CALUDE_curve_tangent_parallel_l3007_300716

/-- The curve C: y = ax^3 + bx^2 + d -/
def C (a b d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + d

/-- The derivative of C with respect to x -/
def C' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem curve_tangent_parallel (a b d : ℝ) :
  C a b d 1 = 1 →  -- Point A(1,1) lies on the curve
  C a b d (-1) = -3 →  -- Point B(-1,-3) lies on the curve
  C' a b 1 = C' a b (-1) →  -- Tangents at A and B are parallel
  a^3 + b^2 + d = 7 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_parallel_l3007_300716


namespace NUMINAMATH_CALUDE_total_money_found_l3007_300717

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_money_found_l3007_300717


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3007_300753

/-- Represents the profit function for a product sale scenario. -/
def profit_function (x : ℝ) : ℝ := -160 * x^2 + 560 * x + 3120

/-- Represents the factory price of the product. -/
def factory_price : ℝ := 3

/-- Represents the initial retail price. -/
def initial_retail_price : ℝ := 4

/-- Represents the initial monthly sales volume. -/
def initial_sales_volume : ℝ := 400

/-- Represents the change in sales volume for every 0.5 CNY price change. -/
def sales_volume_change : ℝ := 40

/-- Theorem stating the maximum profit and the corresponding selling prices. -/
theorem max_profit_theorem :
  (∃ (x : ℝ), x = 1.5 ∨ x = 2) ∧
  (∀ (y : ℝ), y ≤ 3600 → ∃ (x : ℝ), profit_function x = y) ∧
  profit_function 1.5 = 3600 ∧
  profit_function 2 = 3600 := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3007_300753


namespace NUMINAMATH_CALUDE_factor_polynomial_l3007_300799

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 250 * x^13 = 25 * x^7 * (3 - 10 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3007_300799


namespace NUMINAMATH_CALUDE_skee_ball_tickets_value_l3007_300743

/-- The number of tickets Kaleb won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 8

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 5

/-- The number of candies Kaleb can buy -/
def candies_bought : ℕ := 3

/-- The number of tickets Kaleb won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candy_cost * candies_bought - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 7 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_value_l3007_300743


namespace NUMINAMATH_CALUDE_movie_day_points_l3007_300797

theorem movie_day_points (num_students : ℕ) (num_weeks : ℕ) (veg_per_week : ℕ) (points_per_veg : ℕ)
  (h1 : num_students = 25)
  (h2 : num_weeks = 2)
  (h3 : veg_per_week = 2)
  (h4 : points_per_veg = 2) :
  num_students * num_weeks * veg_per_week * points_per_veg = 200 := by
  sorry

#check movie_day_points

end NUMINAMATH_CALUDE_movie_day_points_l3007_300797


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3007_300775

theorem binomial_coefficient_divisibility (p k : ℕ) : 
  Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3007_300775


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l3007_300733

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p + 21) :
  p = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l3007_300733


namespace NUMINAMATH_CALUDE_power_function_properties_l3007_300777

def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧
  (∀ x, f m (-x) = f m x) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3007_300777


namespace NUMINAMATH_CALUDE_electric_bike_survey_sample_size_l3007_300746

/-- Represents the survey about middle school students riding electric bikes to school -/
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  negative_attitudes : ℕ

/-- The specific survey conducted -/
def electric_bike_survey : Survey where
  total_population := 823
  sample_size := 150
  negative_attitudes := 136

/-- Theorem stating that the sample size of the electric bike survey is 150 -/
theorem electric_bike_survey_sample_size :
  electric_bike_survey.sample_size = 150 := by
  sorry

end NUMINAMATH_CALUDE_electric_bike_survey_sample_size_l3007_300746


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3007_300739

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3007_300739


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3007_300744

theorem candy_bar_cost (marvin_sales : ℕ) (tina_sales : ℕ) (price : ℚ) : 
  marvin_sales = 35 →
  tina_sales = 3 * marvin_sales →
  tina_sales * price = marvin_sales * price + 140 →
  price = 2 := by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3007_300744


namespace NUMINAMATH_CALUDE_gcd_problem_l3007_300751

theorem gcd_problem (p : Nat) (h : Prime p) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_problem_l3007_300751


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3007_300771

theorem shopkeeper_profit_percentage
  (selling_price_profit : ℝ)
  (selling_price_loss : ℝ)
  (loss_percentage : ℝ)
  (h1 : selling_price_profit = 900)
  (h2 : selling_price_loss = 540)
  (h3 : loss_percentage = 25) :
  let cost_price := selling_price_loss / (1 - loss_percentage / 100)
  let profit := selling_price_profit - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 25 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3007_300771


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l3007_300779

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 85)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 85)
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 87 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l3007_300779


namespace NUMINAMATH_CALUDE_car_average_speed_l3007_300767

/-- Proves that the average speed of a car is 72 km/h given specific travel conditions -/
theorem car_average_speed (s : ℝ) (h : s > 0) : 
  let t1 := s / 2 / 60
  let t2 := s / 6 / 120
  let t3 := s / 3 / 80
  s / (t1 + t2 + t3) = 72 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l3007_300767


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3007_300788

/-- A rectangular prism is a three-dimensional shape with six faces. -/
structure RectangularPrism where
  faces : Fin 6 → Rectangle

/-- The number of edges in a rectangular prism -/
def edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, corners, and faces in a rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) : 
  edges p + corners p + faces p = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l3007_300788


namespace NUMINAMATH_CALUDE_jerry_debt_payment_l3007_300705

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℝ) (remaining_debt : ℝ) (extra_payment : ℝ) :
  total_debt = 50 ∧ 
  remaining_debt = 23 ∧ 
  extra_payment = 3 →
  ∃ (payment_two_months_ago : ℝ),
    payment_two_months_ago = 12 ∧
    total_debt = remaining_debt + payment_two_months_ago + (payment_two_months_ago + extra_payment) :=
by sorry

end NUMINAMATH_CALUDE_jerry_debt_payment_l3007_300705


namespace NUMINAMATH_CALUDE_same_last_three_digits_l3007_300776

theorem same_last_three_digits (N : ℕ) (h1 : N > 0) :
  (∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 
   N % 1000 = 100 * a + 10 * b + c ∧
   (N^2) % 1000 = 100 * a + 10 * b + c) →
  N % 1000 = 873 :=
by sorry

end NUMINAMATH_CALUDE_same_last_three_digits_l3007_300776


namespace NUMINAMATH_CALUDE_solve_equation_l3007_300768

theorem solve_equation (k l q : ℚ) : 
  (3/4 : ℚ) = k/108 ∧ 
  (3/4 : ℚ) = (l+k)/126 ∧ 
  (3/4 : ℚ) = (q-l)/180 → 
  q = 148.5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3007_300768


namespace NUMINAMATH_CALUDE_largest_number_l3007_300700

theorem largest_number (S : Set ℤ) (h : S = {0, 2, -1, -2}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3007_300700


namespace NUMINAMATH_CALUDE_honey_servings_l3007_300783

/-- Calculates the number of full servings of honey in a container -/
def fullServings (containerAmount : Rat) (servingSize : Rat) : Rat :=
  containerAmount / servingSize

/-- Proves that a container with 47 2/3 tablespoons of honey provides 14 1/5 full servings when each serving is 3 1/3 tablespoons -/
theorem honey_servings :
  let containerAmount : Rat := 47 + 2/3
  let servingSize : Rat := 3 + 1/3
  fullServings containerAmount servingSize = 14 + 1/5 := by
sorry

#eval fullServings (47 + 2/3) (3 + 1/3)

end NUMINAMATH_CALUDE_honey_servings_l3007_300783


namespace NUMINAMATH_CALUDE_joy_tape_problem_l3007_300728

/-- The initial amount of tape given field dimensions and leftover tape -/
def initial_tape (width length leftover : ℕ) : ℕ :=
  2 * (width + length) + leftover

/-- Theorem: Given a field 20 feet wide and 60 feet long, with 90 feet of tape left over after wrapping once, the initial amount of tape is 250 feet -/
theorem joy_tape_problem :
  initial_tape 20 60 90 = 250 := by
  sorry

end NUMINAMATH_CALUDE_joy_tape_problem_l3007_300728


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3007_300789

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) ≤ 3 / 2 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) = 3 / 2 ↔
  x = y ∧ y = z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3007_300789


namespace NUMINAMATH_CALUDE_unique_trig_value_l3007_300742

open Real

theorem unique_trig_value (x : ℝ) (h1 : 0 < x) (h2 : x < π / 3) 
  (h3 : cos x = tan x) : sin x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_trig_value_l3007_300742


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_six_l3007_300782

theorem ceiling_of_negative_three_point_six :
  ⌈(-3.6 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_six_l3007_300782


namespace NUMINAMATH_CALUDE_sqrt_three_comparison_l3007_300708

theorem sqrt_three_comparison : 2 * Real.sqrt 3 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_comparison_l3007_300708


namespace NUMINAMATH_CALUDE_zoo_animals_l3007_300761

theorem zoo_animals (lions : ℕ) (penguins : ℕ) : 
  lions = 30 →
  11 * lions = 3 * penguins →
  penguins - lions = 80 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l3007_300761


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3007_300718

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 2) ↔ 
  (∀ x : ℝ, x = 2 → x^2 - 3*x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3007_300718


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3007_300745

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (h3 : α > 0)
  (h4 : a < 0)
  (h5 : α + β = -b / a)
  (h6 : α * β = c / a) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3007_300745


namespace NUMINAMATH_CALUDE_solve_system_l3007_300792

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3007_300792


namespace NUMINAMATH_CALUDE_power_of_one_fourth_l3007_300715

theorem power_of_one_fourth (a b : ℕ) : 
  (2^a : ℕ) ∣ 180 ∧ 
  (∀ k : ℕ, k > a → ¬((2^k : ℕ) ∣ 180)) ∧ 
  (3^b : ℕ) ∣ 180 ∧ 
  (∀ k : ℕ, k > b → ¬((3^k : ℕ) ∣ 180)) → 
  (1/4 : ℚ)^(b - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_of_one_fourth_l3007_300715


namespace NUMINAMATH_CALUDE_parabola_vertex_l3007_300721

/-- The equation of a parabola is y^2 + 8y + 2x + 1 = 0. 
    This theorem proves that the vertex of the parabola is (7.5, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  (y^2 + 8*y + 2*x + 1 = 0) → (x = 7.5 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3007_300721


namespace NUMINAMATH_CALUDE_expression_value_l3007_300703

theorem expression_value : 
  (1 - 2/7) / (0.25 + 3 * (1/4)) + (2 * 0.3) / (1.3 - 0.4) = 29/21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3007_300703


namespace NUMINAMATH_CALUDE_meeting_attendance_l3007_300760

/-- The number of people attending a meeting where each person receives two copies of a contract --/
def number_of_people (pages_per_contract : ℕ) (copies_per_person : ℕ) (total_pages_copied : ℕ) : ℕ :=
  total_pages_copied / (pages_per_contract * copies_per_person)

/-- Theorem stating that the number of people in the meeting is 9 --/
theorem meeting_attendance : number_of_people 20 2 360 = 9 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendance_l3007_300760


namespace NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l3007_300769

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l3007_300769


namespace NUMINAMATH_CALUDE_dress_pocket_ratio_l3007_300758

/-- Proves that the ratio of dresses with 2 pockets to the total number of dresses with pockets is 1:3 --/
theorem dress_pocket_ratio :
  ∀ (x y : ℕ),
  -- Total number of dresses
  24 = x + y + (24 / 2) →
  -- Total number of pockets
  2 * x + 3 * y = 32 →
  -- Ratio of dresses with 2 pockets to total dresses with pockets
  x / (x + y) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_dress_pocket_ratio_l3007_300758


namespace NUMINAMATH_CALUDE_line_segment_length_l3007_300719

/-- Given points A, B, C, and D on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (B - A = 2) →                  -- AB = 2 cm
  (C - A = 5) →                  -- AC = 5 cm
  (D - B = 6) →                  -- BD = 6 cm
  (D - C = 3) :=                 -- CD = 3 cm (to be proved)
by sorry

end NUMINAMATH_CALUDE_line_segment_length_l3007_300719


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3007_300786

theorem modular_inverse_28_mod_29 : ∃ x : ℤ, (28 * x) % 29 = 1 :=
by
  use 28
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3007_300786


namespace NUMINAMATH_CALUDE_complex_power_difference_l3007_300785

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3007_300785


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3007_300773

/-- Given a point and its image under reflection across a line, prove the sum of the line's slope and y-intercept. -/
theorem reflection_line_sum (x₁ y₁ x₂ y₂ : ℝ) (m b : ℝ) 
  (h₁ : (x₁, y₁) = (2, 3))  -- Original point
  (h₂ : (x₂, y₂) = (10, 7))  -- Image point
  (h₃ : ∀ x y, y = m * x + b →  -- Reflection line equation
              (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0) :
  m + b = 15 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3007_300773


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3007_300793

theorem not_all_perfect_squares (k : ℕ) : ¬(∃ a b c : ℤ, (2 * k - 1 = a^2) ∧ (5 * k - 1 = b^2) ∧ (13 * k - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3007_300793


namespace NUMINAMATH_CALUDE_function_property_implies_zero_l3007_300764

open Set
open Function

theorem function_property_implies_zero (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Ioo a b, f x + f (-x) = 0) : f (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_zero_l3007_300764
