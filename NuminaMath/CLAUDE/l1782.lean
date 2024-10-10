import Mathlib

namespace square_side_length_l1782_178277

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s * s = d * d / 2 ∧ s = 2 := by
  sorry

end square_side_length_l1782_178277


namespace factorial_calculation_l1782_178246

theorem factorial_calculation : 
  Nat.factorial 8 - 7 * Nat.factorial 7 - 2 * Nat.factorial 6 = 3600 := by
  sorry

end factorial_calculation_l1782_178246


namespace sum_in_base_b_l1782_178297

/-- Given a base b, converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, converts a number from base 10 to base b -/
def fromBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a given base b satisfies the condition (14)(17)(18) = 5404 in base b -/
def isValidBase (b : ℕ) : Prop :=
  (toBase10 14 b) * (toBase10 17 b) * (toBase10 18 b) = toBase10 5404 b

theorem sum_in_base_b (b : ℕ) (h : isValidBase b) :
  fromBase10 ((toBase10 14 b) + (toBase10 17 b) + (toBase10 18 b)) b = 49 := by
  sorry

end sum_in_base_b_l1782_178297


namespace contrapositive_equivalence_l1782_178201

theorem contrapositive_equivalence (p q : Prop) :
  (q → p) → (¬p → ¬q) := by sorry

end contrapositive_equivalence_l1782_178201


namespace percentage_comparison_l1782_178200

theorem percentage_comparison (base : ℝ) (first second : ℝ) 
  (h1 : first = base * 1.71)
  (h2 : second = base * 1.80) :
  first / second * 100 = 95 := by
  sorry

end percentage_comparison_l1782_178200


namespace cone_volume_from_slant_and_height_l1782_178211

/-- The volume of a cone given its slant height and height --/
theorem cone_volume_from_slant_and_height 
  (slant_height : ℝ) 
  (height : ℝ) 
  (h_slant : slant_height = 15) 
  (h_height : height = 9) : 
  (1/3 : ℝ) * Real.pi * (slant_height^2 - height^2) * height = 432 * Real.pi := by
  sorry

end cone_volume_from_slant_and_height_l1782_178211


namespace acid_mixing_problem_l1782_178224

/-- Represents the acid mixing problem -/
theorem acid_mixing_problem 
  (volume_first : ℝ) 
  (percentage_second : ℝ) 
  (volume_final : ℝ) 
  (percentage_final : ℝ) 
  (h1 : volume_first = 4)
  (h2 : percentage_second = 75)
  (h3 : volume_final = 20)
  (h4 : percentage_final = 72) :
  ∃ (percentage_first : ℝ),
    percentage_first = 60 ∧
    volume_first * (percentage_first / 100) + 
    (volume_final - volume_first) * (percentage_second / 100) = 
    volume_final * (percentage_final / 100) :=
sorry

end acid_mixing_problem_l1782_178224


namespace problem_statement_l1782_178285

theorem problem_statement (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := by
  sorry

end problem_statement_l1782_178285


namespace race_probability_l1782_178247

theorem race_probability (total_cars : ℕ) (prob_X : ℚ) (prob_Z : ℚ) (prob_XYZ : ℚ) :
  total_cars = 8 →
  prob_X = 1/2 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ (prob_Y : ℚ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_Y = 1/4 := by
  sorry

end race_probability_l1782_178247


namespace sin_negative_135_degrees_l1782_178272

theorem sin_negative_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end sin_negative_135_degrees_l1782_178272


namespace closest_fraction_l1782_178228

def actual_fraction : ℚ := 24 / 150

def candidate_fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ candidate_fractions ∧
  (∀ (f : ℚ), f ∈ candidate_fractions → |f - actual_fraction| ≥ |closest - actual_fraction|) ∧
  closest = 1/6 := by
  sorry

end closest_fraction_l1782_178228


namespace factory_shift_cost_l1782_178265

/-- The cost to employ all workers for one 8-hour shift -/
def total_cost (total_employees : ℕ) (low_wage_employees : ℕ) (mid_wage_employees : ℕ) 
  (low_wage : ℕ) (mid_wage : ℕ) (high_wage : ℕ) (shift_hours : ℕ) : ℕ :=
  let high_wage_employees := total_employees - low_wage_employees - mid_wage_employees
  low_wage_employees * low_wage * shift_hours + 
  mid_wage_employees * mid_wage * shift_hours + 
  high_wage_employees * high_wage * shift_hours

/-- Theorem stating the total cost for the given scenario -/
theorem factory_shift_cost : 
  total_cost 300 200 40 12 14 17 8 = 31840 := by
  sorry

end factory_shift_cost_l1782_178265


namespace unique_point_on_circle_l1782_178261

-- Define the points A and B
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (2, 1)

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 16

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem unique_point_on_circle (a : ℝ) : 
  (∃! P : ℝ × ℝ, C a P.1 P.2 ∧ distanceSquared P A + 2 * distanceSquared P B = 24) →
  a = -1 ∨ a = 3 := by
sorry


end unique_point_on_circle_l1782_178261


namespace waysToSelectIs186_l1782_178256

/-- The number of ways to select 5 balls from a bag containing 4 red balls and 6 white balls,
    such that the total score is at least 7 points (where red balls score 2 points and white balls score 1 point). -/
def waysToSelect : ℕ :=
  Nat.choose 4 4 * Nat.choose 6 1 +
  Nat.choose 4 3 * Nat.choose 6 2 +
  Nat.choose 4 2 * Nat.choose 6 3

/-- The theorem stating that the number of ways to select the balls is 186. -/
theorem waysToSelectIs186 : waysToSelect = 186 := by
  sorry

end waysToSelectIs186_l1782_178256


namespace simplify_expression_l1782_178253

theorem simplify_expression : 
  (9 * 10^12) / (3 * 10^4) + (2 * 10^8) / (4 * 10^2) = 300500000 := by
  sorry

end simplify_expression_l1782_178253


namespace equation_solution_l1782_178282

theorem equation_solution : ∃! x : ℝ, (x + 4) / (x - 2) = 3 ∧ x = 5 := by
  sorry

end equation_solution_l1782_178282


namespace x_x_minus_one_sufficient_not_necessary_l1782_178258

theorem x_x_minus_one_sufficient_not_necessary (x : ℝ) :
  (∀ x, x * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ x * (x - 1) ≥ 0) :=
by sorry

end x_x_minus_one_sufficient_not_necessary_l1782_178258


namespace cubic_root_reciprocal_sum_l1782_178249

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (r s t : ℂ) 
  (ha : a ≠ 0) (hd : d ≠ 0)
  (h_cubic : ∀ x : ℂ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) :
  1 / r^2 + 1 / s^2 + 1 / t^2 = (c^2 - 2 * b * d) / d^2 := by
  sorry

end cubic_root_reciprocal_sum_l1782_178249


namespace ellipse_left_right_vertices_l1782_178203

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/7 = 1

-- Define the left and right vertices
def left_right_vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem statement
theorem ellipse_left_right_vertices :
  ∀ (p : ℝ × ℝ), p ∈ left_right_vertices ↔ 
    (ellipse_equation p.1 p.2 ∧ 
     ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 → abs q.1 ≤ abs p.1) :=
by sorry

end ellipse_left_right_vertices_l1782_178203


namespace coffee_shop_spending_l1782_178266

theorem coffee_shop_spending (ryan_spent : ℝ) (sarah_spent : ℝ) : 
  (sarah_spent = 0.60 * ryan_spent) →
  (ryan_spent = sarah_spent + 12.50) →
  (ryan_spent + sarah_spent = 50.00) :=
by
  sorry

end coffee_shop_spending_l1782_178266


namespace product_equals_half_l1782_178292

/-- Given that a * b * c * d = (√((a + 2) * (b + 3))) / (c + 1) * sin(d) for any a, b, c, and d,
    prove that 6 * 15 * 11 * 30 = 0.5 -/
theorem product_equals_half :
  (∀ a b c d : ℝ, a * b * c * d = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) * Real.sin d) →
  6 * 15 * 11 * 30 = 0.5 := by
  sorry

end product_equals_half_l1782_178292


namespace banana_bread_recipe_l1782_178244

/-- Given a banana bread recipe and baking requirements, determine the number of loaves the recipe can make. -/
theorem banana_bread_recipe (total_loaves : ℕ) (total_bananas : ℕ) (bananas_per_recipe : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : total_bananas = 33)
  (h3 : bananas_per_recipe = 1)
  (h4 : total_bananas > 0) :
  total_loaves / total_bananas = 3 := by
  sorry

#check banana_bread_recipe

end banana_bread_recipe_l1782_178244


namespace ball_max_height_l1782_178250

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 10

/-- The maximum height reached by the ball -/
def max_height : ℝ := 135

/-- Theorem stating that the maximum height reached by the ball is 135 meters -/
theorem ball_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end ball_max_height_l1782_178250


namespace chord_equation_l1782_178216

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define a chord that is bisected by P
def bisected_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
  (x₁ + x₂) / 2 = P.1 ∧ (y₁ + y₂) / 2 = P.2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 5 * x - 3 * y - 13 = 0

theorem chord_equation :
  ∀ x₁ y₁ x₂ y₂ : ℝ, bisected_chord x₁ y₁ x₂ y₂ →
  ∀ x y : ℝ, line_equation x y ↔ (y - P.2) = (5/3) * (x - P.1) :=
by sorry

end chord_equation_l1782_178216


namespace baby_grasshoppers_count_l1782_178202

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := 31

/-- The number of baby grasshoppers under the plant -/
def baby_grasshoppers : ℕ := total_grasshoppers - grasshoppers_on_plant

theorem baby_grasshoppers_count : baby_grasshoppers = 24 := by
  sorry

end baby_grasshoppers_count_l1782_178202


namespace complex_real_condition_l1782_178273

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * Complex.I
  (Z.im = 0 ∧ (a^2 + 4*a - 5) ≠ 0) → a = 3 :=
by sorry

end complex_real_condition_l1782_178273


namespace square_area_l1782_178279

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The line y = 7 -/
def line : ℝ := 7

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    parabola x₁ = line ∧ 
    parabola x₂ = line ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 28 :=
sorry

end square_area_l1782_178279


namespace sum_mod_9_equals_5_l1782_178231

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a list of natural numbers, compute the sum of their modulo 9 values
    after reducing each number to the sum of its digits -/
def sum_mod_9_of_digit_sums (numbers : List ℕ) : ℕ :=
  (numbers.map (fun n => sum_of_digits n % 9)).sum % 9

/-- The main theorem stating that the sum of modulo 9 values of the given numbers
    after reducing each to the sum of its digits is 5 -/
theorem sum_mod_9_equals_5 :
  sum_mod_9_of_digit_sums [1, 21, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999] = 5 := by
  sorry

end sum_mod_9_equals_5_l1782_178231


namespace additional_class_choices_l1782_178275

def total_classes : ℕ := 10
def compulsory_classes : ℕ := 1
def total_classes_to_take : ℕ := 4

theorem additional_class_choices : 
  Nat.choose (total_classes - compulsory_classes) (total_classes_to_take - compulsory_classes) = 84 := by
  sorry

end additional_class_choices_l1782_178275


namespace hyperbola_point_range_l1782_178259

theorem hyperbola_point_range (x₀ y₀ : ℝ) : 
  (x₀^2 / 2 - y₀^2 = 1) →  -- Point on hyperbola
  (((-Real.sqrt 3 - x₀) * (Real.sqrt 3 - x₀) + (-y₀) * (-y₀)) ≤ 0) →  -- Dot product condition
  (-Real.sqrt 3 / 3 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 3) :=
by sorry

end hyperbola_point_range_l1782_178259


namespace two_valid_permutations_l1782_178274

def S : Finset ℕ := Finset.range 2022

def is_valid_permutation (A : Fin 2022 → ℕ) : Prop :=
  Function.Injective A ∧ (∀ i, A i ∈ S) ∧
  (∀ n m : Fin 2022, (A n + A m) % (Nat.gcd n.val m.val) = 0)

theorem two_valid_permutations :
  ∃! (p : Finset (Fin 2022 → ℕ)), p.card = 2 ∧ ∀ A ∈ p, is_valid_permutation A :=
sorry

end two_valid_permutations_l1782_178274


namespace olympic_torch_relay_l1782_178218

/-- The total number of cities -/
def total_cities : ℕ := 8

/-- The number of cities to be selected for the relay route -/
def selected_cities : ℕ := 6

/-- The number of ways to select exactly one city from two cities -/
def select_one_from_two : ℕ := 2

/-- The number of ways to select 5 cities from 6 cities -/
def select_five_from_six : ℕ := 6

/-- The number of ways to select 4 cities from 6 cities -/
def select_four_from_six : ℕ := 15

/-- The number of permutations of 6 cities -/
def permutations_of_six : ℕ := 720

theorem olympic_torch_relay :
  (
    /- Condition 1 -/
    (select_one_from_two * select_five_from_six = 12) ∧
    (12 * permutations_of_six = 8640)
  ) ∧
  (
    /- Condition 2 -/
    (select_one_from_two * select_five_from_six + select_four_from_six = 27) ∧
    (27 * permutations_of_six = 19440)
  ) := by sorry

end olympic_torch_relay_l1782_178218


namespace bird_count_l1782_178295

theorem bird_count (N t : ℚ) : 
  (3 / 5 * N + 1 / 4 * N + 10 * t = N) → 
  (3 / 5 * N = 40 * t) :=
by sorry

end bird_count_l1782_178295


namespace otimes_nested_l1782_178217

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 + y

/-- Theorem: a ⊗ (a ⊗ a) = 2a^2 + a -/
theorem otimes_nested (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a := by
  sorry

end otimes_nested_l1782_178217


namespace book_sale_fraction_l1782_178243

theorem book_sale_fraction (price : ℝ) (remaining : ℕ) (total_received : ℝ) :
  price = 3.5 →
  remaining = 36 →
  total_received = 252 →
  ∃ (total : ℕ) (sold : ℕ),
    total > 0 ∧
    sold = total - remaining ∧
    (sold : ℝ) / total = 2 / 3 ∧
    price * sold = total_received :=
by sorry

end book_sale_fraction_l1782_178243


namespace no_valid_arrangement_l1782_178234

/-- A type representing a circular arrangement of 60 numbers -/
def CircularArrangement := Fin 60 → ℕ

/-- Predicate checking if a given arrangement satisfies all conditions -/
def SatisfiesConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, (arr i + arr ((i + 2) % 60)) % 2 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 3) % 60)) % 3 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 7) % 60)) % 7 = 0)

/-- Predicate checking if an arrangement is a permutation of 1 to 60 -/
def IsValidArrangement (arr : CircularArrangement) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 60 → ∃ i : Fin 60, arr i = n + 1) ∧
  (∀ i j : Fin 60, arr i = arr j → i = j)

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_arrangement :
  ¬ ∃ arr : CircularArrangement, IsValidArrangement arr ∧ SatisfiesConditions arr :=
sorry

end no_valid_arrangement_l1782_178234


namespace rectangle_x_value_l1782_178269

/-- A rectangle with specified side lengths -/
structure Rectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X must be 7 in the given rectangle -/
theorem rectangle_x_value (r : Rectangle) 
    (h1 : r.top_left = 1)
    (h2 : r.top_middle = 2)
    (h3 : r.top_right = 3)
    (h4 : r.bottom_left = 4)
    (h5 : r.bottom_middle = 2)
    (h6 : r.bottom_right = 7)
    (h_rect : r.top_left + r.top_middle + X + r.top_right = 
              r.bottom_left + r.bottom_middle + r.bottom_right) : 
  X = 7 := by
  sorry


end rectangle_x_value_l1782_178269


namespace valid_triangle_constructions_l1782_178291

-- Define the basic structure
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the given points
variable (A A₀ D E : ℝ × ℝ)

-- Define the midpoint property
def is_midpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the median property
def is_median (M : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  is_midpoint M B C

-- Define the angle bisector property
def is_angle_bisector (D : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := sorry

-- Define the perpendicular bisector property
def is_perpendicular_bisector (E : ℝ × ℝ) (A C : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem valid_triangle_constructions 
  (h1 : is_midpoint A₀ (Triangle.B t) (Triangle.C t))
  (h2 : is_median A₀ (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h3 : is_angle_bisector D (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h4 : is_perpendicular_bisector E (Triangle.A t) (Triangle.C t)) :
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧ 
    (∃ (t₁ t₂ : Triangle), 
      (t₁.A = A ∧ t₁.C = C₁) ∧ 
      (t₂.A = A ∧ t₂.C = C₂) ∧
      (is_midpoint A₀ t₁.B t₁.C) ∧
      (is_midpoint A₀ t₂.B t₂.C) ∧
      (is_median A₀ t₁.A t₁.B t₁.C) ∧
      (is_median A₀ t₂.A t₂.B t₂.C) ∧
      (is_angle_bisector D t₁.A t₁.B t₁.C) ∧
      (is_angle_bisector D t₂.A t₂.B t₂.C) ∧
      (is_perpendicular_bisector E t₁.A t₁.C) ∧
      (is_perpendicular_bisector E t₂.A t₂.C)) :=
sorry


end valid_triangle_constructions_l1782_178291


namespace percent_less_than_l1782_178254

theorem percent_less_than (P Q : ℝ) (h : P < Q) :
  (Q - P) / Q * 100 = 100 * (Q - P) / Q :=
by sorry

end percent_less_than_l1782_178254


namespace ratio_problem_l1782_178287

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : 
  x / y = 7 / 4 := by
sorry

end ratio_problem_l1782_178287


namespace least_subtraction_for_divisibility_l1782_178215

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (9671 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (9671 - m) % 5 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l1782_178215


namespace sum_of_repeating_decimals_l1782_178280

/-- The sum of the repeating decimals 0.666... and 0.333... is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (10 * x - x = 6 ∧ 10 * y - y = 3) → x + y = 1) := by
  sorry

end sum_of_repeating_decimals_l1782_178280


namespace quadratic_inequality_solution_set_l1782_178223

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} := by
  sorry

end quadratic_inequality_solution_set_l1782_178223


namespace solution_set_f_greater_than_two_range_of_t_l1782_178229

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1} ∪ {x : ℝ | x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5) := by sorry

end solution_set_f_greater_than_two_range_of_t_l1782_178229


namespace triangle_perimeter_l1782_178209

/-- Given a triangle with inradius 3 cm and area 30 cm², its perimeter is 20 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 3 → A = 30 → A = r * (p / 2) → p = 20 := by sorry

end triangle_perimeter_l1782_178209


namespace sum_of_squares_of_roots_l1782_178264

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (10 * x₁^2 + 15 * x₁ - 17 = 0) →
  (10 * x₂^2 + 15 * x₂ - 17 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 113/20 := by
sorry

end sum_of_squares_of_roots_l1782_178264


namespace pig_farm_fence_length_l1782_178252

/-- Represents a rectangular pig farm with specific dimensions -/
structure PigFarm where
  /-- Length of the shorter sides of the rectangle -/
  short_side : ℝ
  /-- Ensures the short side is positive -/
  short_side_pos : short_side > 0

/-- Calculates the area of the pig farm -/
def PigFarm.area (farm : PigFarm) : ℝ :=
  2 * farm.short_side * farm.short_side

/-- Calculates the total fence length of the pig farm -/
def PigFarm.fence_length (farm : PigFarm) : ℝ :=
  4 * farm.short_side

/-- Theorem stating the fence length for a pig farm with area 1250 sq ft -/
theorem pig_farm_fence_length :
  ∃ (farm : PigFarm), farm.area = 1250 ∧ farm.fence_length = 100 := by
  sorry

end pig_farm_fence_length_l1782_178252


namespace fixed_points_for_specific_values_two_distinct_fixed_points_condition_l1782_178245

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point (f 1 (-2)) x ↔ (x = 3 ∨ x = -1) := by sorry

theorem two_distinct_fixed_points_condition :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔ (0 < a ∧ a < 1) := by sorry

end fixed_points_for_specific_values_two_distinct_fixed_points_condition_l1782_178245


namespace fraction_comparison_l1782_178240

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
sorry

end fraction_comparison_l1782_178240


namespace sum_binary_digits_310_l1782_178205

def sum_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem sum_binary_digits_310 : sum_binary_digits 310 = 5 := by
  sorry

end sum_binary_digits_310_l1782_178205


namespace temperature_conversion_l1782_178251

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 68 → t = 20 := by
  sorry

end temperature_conversion_l1782_178251


namespace technician_round_trip_completion_l1782_178238

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let total_distance := 2 * distance
  let completed_distance := distance + 0.1 * distance
  (completed_distance / total_distance) * 100 = 55 := by
sorry

end technician_round_trip_completion_l1782_178238


namespace max_routes_in_network_l1782_178226

/-- A bus route network -/
structure BusNetwork where
  stops : Nat
  routes : Nat
  stops_per_route : Nat
  route_intersection : Nat

/-- The condition that any two routes either have no common stops or have exactly one common stop -/
def valid_intersection (network : BusNetwork) : Prop :=
  network.route_intersection = 0 ∨ network.route_intersection = 1

/-- The maximum number of routes possible given the constraints -/
def max_routes (network : BusNetwork) : Prop :=
  network.routes ≤ (network.stops * 4) / 3 ∧
  network.routes = 12

/-- Theorem stating the maximum number of routes in the given network -/
theorem max_routes_in_network (network : BusNetwork) 
  (h1 : network.stops = 9)
  (h2 : network.stops_per_route = 3)
  (h3 : valid_intersection network) :
  max_routes network :=
sorry

end max_routes_in_network_l1782_178226


namespace sum_of_f_at_lg2_and_lg_half_l1782_178262

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

theorem sum_of_f_at_lg2_and_lg_half (a : ℝ) :
  f a 0 = 1 → f a (Real.log 2 / Real.log 10) + f a (Real.log (1/2) / Real.log 10) = 2 := by
  sorry

end sum_of_f_at_lg2_and_lg_half_l1782_178262


namespace tank_capacity_l1782_178239

/-- Represents the capacity of a tank and its inlet/outlet properties -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions -/
theorem tank_capacity (t : Tank)
  (h1 : t.outlet_time = 10)
  (h2 : t.inlet_rate = 8 * 60)
  (h3 : t.combined_time = 16)
  : t.capacity = 1280 := by
  sorry

end tank_capacity_l1782_178239


namespace L8_2_7_exponent_is_columns_l1782_178233

/-- Represents an orthogonal array -/
structure OrthogonalArray where
  experiments : ℕ
  levels : ℕ
  columns : ℕ

/-- The specific orthogonal array L₈(2⁷) -/
def L8_2_7 : OrthogonalArray :=
  { experiments := 8
  , levels := 2
  , columns := 7 }

theorem L8_2_7_exponent_is_columns : L8_2_7.columns = 7 := by
  sorry

end L8_2_7_exponent_is_columns_l1782_178233


namespace hospital_baby_probability_l1782_178221

/-- The probability of success for a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 6

/-- The number of successes we're interested in -/
def k : ℕ := 3

/-- The probability of at least k successes in n trials with probability p -/
def prob_at_least (p : ℚ) (n k : ℕ) : ℚ :=
  1 - (Finset.range k).sum (λ i => Nat.choose n i * p^i * (1-p)^(n-i))

theorem hospital_baby_probability :
  prob_at_least p n k = 233/729 := by sorry

end hospital_baby_probability_l1782_178221


namespace supermarket_spending_l1782_178276

theorem supermarket_spending (total : ℝ) :
  (1/2 : ℝ) * total +  -- Spent on fresh fruits and vegetables
  (1/3 : ℝ) * total +  -- Spent on meat products
  (1/10 : ℝ) * total + -- Spent on bakery products
  10 = total           -- Remaining spent on candy
  →
  total = 150 := by
sorry

end supermarket_spending_l1782_178276


namespace tim_balloon_count_l1782_178299

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim has more balloons than Dan -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

theorem tim_balloon_count : tim_balloons = 203 := by
  sorry

end tim_balloon_count_l1782_178299


namespace gcd_of_three_numbers_l1782_178283

theorem gcd_of_three_numbers : Nat.gcd 9486 (Nat.gcd 13524 36582) = 6 := by
  sorry

end gcd_of_three_numbers_l1782_178283


namespace simplify_trigonometric_expressions_l1782_178206

theorem simplify_trigonometric_expressions :
  (∀ α : ℝ, (1 + Real.tan α ^ 2) * Real.cos α ^ 2 = 1) ∧
  (Real.sin (7 * π / 6) + Real.tan (5 * π / 4) = 1 / 2) := by
  sorry

end simplify_trigonometric_expressions_l1782_178206


namespace partial_fraction_decomposition_l1782_178260

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
  (6 * x - 3) / (x^2 - 9*x - 36) = (23/5) / (x - 12) + (7/5) / (x + 3) := by
  sorry

end partial_fraction_decomposition_l1782_178260


namespace roots_product_theorem_l1782_178278

theorem roots_product_theorem (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) → 
  (b^3 - 15*b^2 + 25*b - 10 = 0) → 
  (c^3 - 15*c^2 + 25*c - 10 = 0) → 
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end roots_product_theorem_l1782_178278


namespace expected_value_Z_l1782_178298

/-- The probability mass function for the random variable Z --/
def pmf_Z (P : ℝ) (k : ℕ) : ℝ :=
  if k ≥ 2 then P * (1 - P)^(k - 1) + (1 - P) * P^(k - 1) else 0

/-- The expected value of Z --/
noncomputable def E_Z (P : ℝ) : ℝ :=
  ∑' k, k * pmf_Z P k

/-- Theorem stating the expected value of Z --/
theorem expected_value_Z (P : ℝ) (hP : 0 < P ∧ P < 1) :
  E_Z P = 1 / (P * (1 - P)) - 1 := by
  sorry

end expected_value_Z_l1782_178298


namespace total_stops_theorem_l1782_178257

def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

theorem total_stops_theorem : yoojeong_stops + namjoon_stops = 5 := by
  sorry

end total_stops_theorem_l1782_178257


namespace function_value_comparison_l1782_178214

theorem function_value_comparison (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end function_value_comparison_l1782_178214


namespace greatest_k_value_l1782_178255

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) → 
  k ≤ Real.sqrt 117 :=
by sorry

end greatest_k_value_l1782_178255


namespace angle_between_vectors_not_necessarily_alpha_minus_beta_l1782_178293

theorem angle_between_vectors_not_necessarily_alpha_minus_beta 
  (α β : ℝ) (a b : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  a ≠ b →
  ∃ θ, Real.cos θ = Real.cos α * Real.cos β + Real.sin α * Real.sin β ∧ θ ≠ α - β :=
by sorry

end angle_between_vectors_not_necessarily_alpha_minus_beta_l1782_178293


namespace greatest_integer_less_than_AD_l1782_178235

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  AD : ℝ
  E : ℝ
  ac_be_perp : Bool

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.AB = 80 ∧
  rect.E = (1/3) * rect.AD ∧
  rect.ac_be_perp = true

/-- Theorem statement -/
theorem greatest_integer_less_than_AD (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  ⌊rect.AD⌋ = 138 :=
sorry

end greatest_integer_less_than_AD_l1782_178235


namespace inequality_solution_l1782_178213

theorem inequality_solution (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
  (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
by sorry

end inequality_solution_l1782_178213


namespace decimal_to_fraction_l1782_178294

theorem decimal_to_fraction : 
  (3.76 : ℚ) = 94 / 25 := by sorry

end decimal_to_fraction_l1782_178294


namespace circle_sum_l1782_178290

def Circle := Fin 12 → ℝ

def is_valid_circle (c : Circle) : Prop :=
  (∀ i, c i ≠ 0) ∧
  (∀ i, i % 2 = 0 → c i = c ((i + 11) % 12) + c ((i + 1) % 12)) ∧
  (∀ i, i % 2 = 1 → c i = c ((i + 11) % 12) * c ((i + 1) % 12))

theorem circle_sum (c : Circle) (h : is_valid_circle c) :
  (Finset.sum Finset.univ c) = 4.5 := by
  sorry

end circle_sum_l1782_178290


namespace milk_container_problem_l1782_178230

theorem milk_container_problem (A B C : ℝ) : 
  A > 0 →  -- A is positive (container capacity)
  B = 0.375 * A →  -- B is 62.5% less than A
  C = A - B →  -- C contains the rest of the milk
  C - 152 = B + 152 →  -- After transfer, B and C are equal
  A = 608 :=
by
  sorry

end milk_container_problem_l1782_178230


namespace intersection_x_coordinate_l1782_178289

-- Define the lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 14
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 40

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -68 := by sorry

end intersection_x_coordinate_l1782_178289


namespace problem_body_surface_area_l1782_178208

/-- Represents a three-dimensional geometric body -/
structure GeometricBody where
  -- Add necessary fields to represent the geometric body
  -- This is a placeholder as we don't have specific information about the structure

/-- Calculates the surface area of a geometric body -/
noncomputable def surfaceArea (body : GeometricBody) : ℝ :=
  sorry -- Actual calculation would go here

/-- Represents the specific geometric body from the problem -/
def problemBody : GeometricBody :=
  sorry -- Construction of the specific body would go here

/-- Theorem stating that the surface area of the problem's geometric body is 40 -/
theorem problem_body_surface_area :
    surfaceArea problemBody = 40 := by
  sorry

end problem_body_surface_area_l1782_178208


namespace complex_equation_solution_l1782_178270

theorem complex_equation_solution (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 + 5 * Complex.I) →
  (t > 0) →
  (t = 3 + 10 * Real.sqrt 2) := by
sorry

end complex_equation_solution_l1782_178270


namespace work_completion_days_l1782_178281

theorem work_completion_days (total_men : ℕ) (absent_men : ℕ) (reduced_days : ℕ) 
  (h1 : total_men = 60)
  (h2 : absent_men = 10)
  (h3 : reduced_days = 60) :
  let remaining_men := total_men - absent_men
  let original_days := (remaining_men * reduced_days) / total_men
  original_days = 50 := by
  sorry

end work_completion_days_l1782_178281


namespace transform_sine_function_l1782_178236

/-- Given a function f and its transformed version g, 
    where g is obtained by shortening the abscissas of f to half their original length 
    and then shifting the resulting curve to the right by π/3 units,
    prove that f(x) = sin(x/2 + π/12) if g(x) = sin(x - π/4) -/
theorem transform_sine_function (f g : ℝ → ℝ) :
  (∀ x, g x = f ((x - π/3) / 2)) →
  (∀ x, g x = Real.sin (x - π/4)) →
  ∀ x, f x = Real.sin (x/2 + π/12) := by
sorry

end transform_sine_function_l1782_178236


namespace paris_visits_l1782_178204

/-- Represents the attractions in Paris --/
inductive Attraction
  | EiffelTower
  | ArcDeTriomphe
  | Montparnasse
  | Playground

/-- Represents a nephew's statement about visiting an attraction --/
structure Statement where
  attraction : Attraction
  visited : Bool

/-- Represents a nephew's set of statements --/
structure NephewStatements where
  statements : List Statement

/-- The statements made by the three nephews --/
def nephewsStatements : List NephewStatements := [
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := true },
    { attraction := Attraction.Montparnasse, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.Montparnasse, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := false },
    { attraction := Attraction.Playground, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := false },
    { attraction := Attraction.ArcDeTriomphe, visited := true }
  ] }
]

/-- The theorem to prove --/
theorem paris_visits (statements : List NephewStatements) 
  (h : statements = nephewsStatements) : 
  ∃ (visits : List Attraction),
    visits = [Attraction.EiffelTower, Attraction.ArcDeTriomphe, Attraction.Montparnasse] ∧
    Attraction.Playground ∉ visits :=
sorry

end paris_visits_l1782_178204


namespace tens_digit_of_6_to_2050_l1782_178284

theorem tens_digit_of_6_to_2050 : 6^2050 % 100 = 56 := by sorry

end tens_digit_of_6_to_2050_l1782_178284


namespace scale_length_l1782_178222

-- Define the number of parts in the scale
def num_parts : ℕ := 5

-- Define the length of each part in inches
def part_length : ℕ := 16

-- Theorem stating the total length of the scale
theorem scale_length : num_parts * part_length = 80 := by
  sorry

end scale_length_l1782_178222


namespace population_size_l1782_178225

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (P : ℝ) 
  (birth_rate : ℝ) (death_rate : ℝ) (net_growth_rate : ℝ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate = 2.1)
  (h4 : (birth_rate - death_rate) / P * 100 = net_growth_rate) :
  P = 1000 := by
  sorry

end population_size_l1782_178225


namespace bagel_savings_l1782_178237

/-- The cost of an individual bagel in cents -/
def individual_cost : ℕ := 225

/-- The cost of a dozen bagels in dollars -/
def dozen_cost : ℕ := 24

/-- The number of bagels in a dozen -/
def bagels_per_dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
theorem bagel_savings : ℕ := by
  -- Convert individual cost to cents
  -- Calculate cost per bagel when buying a dozen
  -- Convert dozen cost to cents
  -- Calculate the difference
  sorry

end bagel_savings_l1782_178237


namespace fixed_point_of_logarithmic_function_l1782_178212

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = 1 + log_a(x-1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

-- State the theorem
theorem fixed_point_of_logarithmic_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end fixed_point_of_logarithmic_function_l1782_178212


namespace triangle_with_angle_ratio_1_2_3_is_right_triangle_l1782_178267

/-- A triangle with interior angles in the ratio 1:2:3 is a right triangle. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (α β γ : ℝ) :
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  α + β + γ = 180 →        -- Sum of angles in a triangle is 180°
  β = 2 * α ∧ γ = 3 * α →  -- Angles are in the ratio 1:2:3
  γ = 90                   -- The largest angle is 90°
  := by sorry

end triangle_with_angle_ratio_1_2_3_is_right_triangle_l1782_178267


namespace max_area_is_1406_l1782_178241

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  width : ℕ
  length : ℕ
  perimeter_constraint : width * 2 + length * 2 = 150

/-- The area of a rectangular garden. -/
def garden_area (g : RectangularGarden) : ℕ :=
  g.width * g.length

/-- The maximum area of a rectangular garden with a perimeter of 150 feet. -/
def max_garden_area : ℕ := 1406

/-- Theorem stating that the maximum area of a rectangular garden with
    a perimeter of 150 feet and integer side lengths is 1406 square feet. -/
theorem max_area_is_1406 :
  ∀ g : RectangularGarden, garden_area g ≤ max_garden_area :=
by sorry

end max_area_is_1406_l1782_178241


namespace perpendicular_bisector_equation_l1782_178286

/-- The line that is the perpendicular bisector of two points -/
def perpendicular_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P A = dist P B}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point satisfies a line equation -/
def satisfies_equation (P : ℝ × ℝ) (L : LineEquation) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

theorem perpendicular_bisector_equation 
  (A B : ℝ × ℝ) 
  (hA : A = (7, -4)) 
  (hB : B = (-5, 6)) :
  ∃ L : LineEquation, 
    L.a = 6 ∧ L.b = -5 ∧ L.c = -1 ∧
    ∀ P, P ∈ perpendicular_bisector A B ↔ satisfies_equation P L :=
by sorry

end perpendicular_bisector_equation_l1782_178286


namespace no_perfect_square_with_conditions_l1782_178242

/-- A function that checks if a natural number is a nine-digit number -/
def isNineDigit (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n < 1000000000

/-- A function that checks if a natural number ends with 5 -/
def endsWithFive (n : ℕ) : Prop :=
  n % 10 = 5

/-- A function that checks if a natural number contains each of the digits 1-9 exactly once -/
def containsEachDigitOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] →
    (∃! p : ℕ, p < 9 ∧ (n / 10^p) % 10 = d)

/-- The main theorem stating that no number satisfying the given conditions is a perfect square -/
theorem no_perfect_square_with_conditions :
  ¬∃ n : ℕ, isNineDigit n ∧ endsWithFive n ∧ containsEachDigitOnce n ∧ ∃ m : ℕ, n = m^2 := by
  sorry


end no_perfect_square_with_conditions_l1782_178242


namespace car_fuel_efficiency_l1782_178248

/-- Proves that a car can travel approximately 56.01 kilometers on a liter of fuel given specific conditions. -/
theorem car_fuel_efficiency (travel_time : Real) (fuel_used_gallons : Real) (speed_mph : Real)
    (gallons_to_liters : Real) (miles_to_km : Real)
    (h1 : travel_time = 5.7)
    (h2 : fuel_used_gallons = 3.9)
    (h3 : speed_mph = 91)
    (h4 : gallons_to_liters = 3.8)
    (h5 : miles_to_km = 1.6) :
    ∃ km_per_liter : Real, abs (km_per_liter - 56.01) < 0.01 ∧
    km_per_liter = (speed_mph * travel_time * miles_to_km) / (fuel_used_gallons * gallons_to_liters) :=
by
  sorry


end car_fuel_efficiency_l1782_178248


namespace mitch_savings_l1782_178263

/-- Represents the total amount of money Mitch has saved for his boating hobby -/
def total_saved : ℕ := 20000

/-- Cost of a new boat per foot in length -/
def boat_cost_per_foot : ℕ := 1500

/-- Amount Mitch needs to keep for license and registration -/
def license_registration_cost : ℕ := 500

/-- Maximum length of boat Mitch can buy -/
def max_boat_length : ℕ := 12

/-- Docking fee multiplier (relative to license and registration cost) -/
def docking_fee_multiplier : ℕ := 3

theorem mitch_savings :
  total_saved = 
    boat_cost_per_foot * max_boat_length + 
    license_registration_cost + 
    docking_fee_multiplier * license_registration_cost :=
by sorry

end mitch_savings_l1782_178263


namespace quadratic_transformation_l1782_178220

/-- The transformation from y = -2x^2 + 4x + 1 to y = -2x^2 -/
theorem quadratic_transformation (f g : ℝ → ℝ) (h_f : f = λ x => -2*x^2 + 4*x + 1) (h_g : g = λ x => -2*x^2) : 
  (∃ (a b : ℝ), ∀ x, f x = g (x + a) + b) ∧ 
  (∃ (vertex_f vertex_g : ℝ × ℝ), 
    vertex_f = (1, 3) ∧ 
    vertex_g = (0, 0) ∧ 
    vertex_f.1 - vertex_g.1 = 1 ∧ 
    vertex_f.2 - vertex_g.2 = 3) :=
by sorry

end quadratic_transformation_l1782_178220


namespace quadratic_factorization_l1782_178227

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l1782_178227


namespace consecutive_sum_at_least_17_l1782_178210

theorem consecutive_sum_at_least_17 (a : Fin 10 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 11 \ {0}) : 
  ∃ i : Fin 10, a i + a (i + 1) + a (i + 2) ≥ 17 :=
sorry

end consecutive_sum_at_least_17_l1782_178210


namespace three_men_three_women_arrangements_l1782_178207

/-- The number of ways to arrange n men and n women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

theorem three_men_three_women_arrangements :
  alternating_arrangements 3 = 72 := by
  sorry

end three_men_three_women_arrangements_l1782_178207


namespace davids_mowing_hours_l1782_178296

theorem davids_mowing_hours (rate : ℝ) (days : ℕ) (remaining : ℝ) : 
  rate = 14 → days = 7 → remaining = 49 → 
  ∃ (hours : ℝ), 
    hours * rate * days / 2 / 2 = remaining ∧ 
    hours = 2 := by
  sorry

end davids_mowing_hours_l1782_178296


namespace stuffed_animal_cost_l1782_178268

theorem stuffed_animal_cost (coloring_books_cost peanuts_cost total_spent : ℚ) : 
  coloring_books_cost = 8 →
  peanuts_cost = 6 →
  total_spent = 25 →
  total_spent - (coloring_books_cost + peanuts_cost) = 11 :=
by
  sorry

end stuffed_animal_cost_l1782_178268


namespace smallest_cover_count_l1782_178288

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered -/
structure Region where
  width : ℕ
  height : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

def Region.area (r : Region) : ℕ := r.width * r.height

/-- The number of rectangles needed to cover a region -/
def coverCount (r : Rectangle) (reg : Region) : ℕ :=
  Region.area reg / Rectangle.area r

theorem smallest_cover_count (r : Rectangle) (reg : Region) :
  r.width = 3 ∧ r.height = 4 ∧ reg.width = 12 ∧ reg.height = 12 →
  coverCount r reg = 12 ∧
  ∀ (r' : Rectangle) (reg' : Region),
    r'.width * r'.height ≤ r.width * r.height →
    reg'.width = 12 →
    coverCount r' reg' ≥ 12 :=
sorry

end smallest_cover_count_l1782_178288


namespace function_range_theorem_l1782_178219

open Real

theorem function_range_theorem (a : ℝ) (m n p : ℝ) : 
  let f := fun (x : ℝ) => -x^3 + 3*x + a
  (m ≠ n ∧ n ≠ p ∧ m ≠ p) →
  (f m = 2022 ∧ f n = 2022 ∧ f p = 2022) →
  (2020 < a ∧ a < 2024) := by
sorry

end function_range_theorem_l1782_178219


namespace problem_solution_l1782_178271

def A (a : ℚ) : Set ℚ := {a^2, a+1, -3}
def B (a : ℚ) : Set ℚ := {a-3, 3*a-1, a^2+1}
def C (m : ℚ) : Set ℚ := {x | m*x = 1}

theorem problem_solution (a m : ℚ) 
  (h1 : A a ∩ B a = {-3}) 
  (h2 : C m ⊆ A a ∩ B a) : 
  a = -2/3 ∧ (m = 0 ∨ m = -1/3) := by
  sorry


end problem_solution_l1782_178271


namespace train_speed_on_time_l1782_178232

/-- Proves that the speed at which the train arrives on time is 84 km/h, given the conditions -/
theorem train_speed_on_time (d : ℝ) (t : ℝ) :
  (d = 80 * (t + 24/60)) →
  (d = 90 * (t - 32/60)) →
  (d / t = 84) :=
by sorry

end train_speed_on_time_l1782_178232
