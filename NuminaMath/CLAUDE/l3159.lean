import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3159_315965

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3159_315965


namespace NUMINAMATH_CALUDE_roberts_spending_l3159_315984

theorem roberts_spending (total : ℝ) : 
  total = 100 + 125 + 0.1 * total → total = 250 :=
by sorry

end NUMINAMATH_CALUDE_roberts_spending_l3159_315984


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3159_315928

/-- The value of k for which vectors a and b are perpendicular --/
theorem perpendicular_vectors (i j a b : ℝ × ℝ) (k : ℝ) : 
  i = (1, 0) →
  j = (0, 1) →
  a = (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2) →
  b = (k * i.1 + 0 * i.2, 0 * j.1 + (-4) * j.2) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3159_315928


namespace NUMINAMATH_CALUDE_smallest_k_carboxylic_for_8002_l3159_315934

/-- A function that checks if a number has all digits the same --/
def allDigitsSame (n : ℕ) : Prop := sorry

/-- A function that checks if a list of natural numbers are all distinct --/
def allDistinct (list : List ℕ) : Prop := sorry

/-- A function that checks if all numbers in a list are greater than 9 --/
def allGreaterThan9 (list : List ℕ) : Prop := sorry

/-- A function that checks if a number is k-carboxylic --/
def isKCarboxylic (n k : ℕ) : Prop :=
  ∃ (list : List ℕ), 
    list.length = k ∧ 
    list.sum = n ∧ 
    allDistinct list ∧ 
    allGreaterThan9 list ∧ 
    ∀ m ∈ list, allDigitsSame m

/-- The main theorem --/
theorem smallest_k_carboxylic_for_8002 :
  (isKCarboxylic 8002 14) ∧ ∀ k < 14, ¬(isKCarboxylic 8002 k) := by sorry

end NUMINAMATH_CALUDE_smallest_k_carboxylic_for_8002_l3159_315934


namespace NUMINAMATH_CALUDE_bagel_count_l3159_315999

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the cost of a bagel in cents -/
def bagel_cost : ℕ := 65

/-- Represents the cost of a muffin in cents -/
def muffin_cost : ℕ := 40

/-- Represents the number of days in the week -/
def days_in_week : ℕ := 7

/-- 
Given a 7-day period where either a 40-cent muffin or a 65-cent bagel is bought each day, 
and the total spending is a whole number of dollars, the number of bagels bought must be 4.
-/
theorem bagel_count : 
  ∀ (b : ℕ), 
  b ≤ days_in_week → 
  (bagel_cost * b + muffin_cost * (days_in_week - b)) % cents_per_dollar = 0 → 
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_bagel_count_l3159_315999


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3159_315937

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  3 * x^2 + 5 * y^2 - 9 * x + 10 * y + 15 = 0

/-- Definition of an ellipse -/
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y, f x y ↔ A * (x - h)^2 + B * (y - k)^2 = 1

/-- Theorem: The given conic equation represents an ellipse -/
theorem conic_is_ellipse : is_ellipse conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3159_315937


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l3159_315987

theorem triangle_square_side_ratio : 
  ∀ (triangle_side square_side : ℚ),
    triangle_side * 3 = 60 →
    square_side * 4 = 60 →
    triangle_side / square_side = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l3159_315987


namespace NUMINAMATH_CALUDE_computer_printer_price_l3159_315906

/-- The total price of a basic computer and printer, given specific conditions -/
theorem computer_printer_price (basic_price enhanced_price printer_price total_price : ℝ) : 
  basic_price = 2125 →
  enhanced_price = basic_price + 500 →
  printer_price = (1 / 8) * (enhanced_price + printer_price) →
  total_price = basic_price + printer_price →
  total_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_printer_price_l3159_315906


namespace NUMINAMATH_CALUDE_polynomial_equality_l3159_315951

theorem polynomial_equality : 110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 161051000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3159_315951


namespace NUMINAMATH_CALUDE_pet_store_birds_l3159_315933

theorem pet_store_birds (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ)
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 2) :
  num_cages * (parrots_per_cage + parakeets_per_cage) = 36 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3159_315933


namespace NUMINAMATH_CALUDE_homework_problem_l3159_315948

theorem homework_problem (p t : ℕ) (h1 : p > 12) (h2 : t > 0) 
  (h3 : p * t = (p + 6) * (t - 3)) : p * t = 140 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l3159_315948


namespace NUMINAMATH_CALUDE_ratio_closest_to_nine_l3159_315920

theorem ratio_closest_to_nine : 
  ∀ n : ℕ, |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - 9| ≤ 
           |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - n| :=
by sorry

end NUMINAMATH_CALUDE_ratio_closest_to_nine_l3159_315920


namespace NUMINAMATH_CALUDE_trig_roots_theorem_l3159_315944

theorem trig_roots_theorem (θ : ℝ) (m : ℝ) 
  (h1 : (Real.sin θ)^2 - (Real.sqrt 3 - 1) * (Real.sin θ) + m = 0)
  (h2 : (Real.cos θ)^2 - (Real.sqrt 3 - 1) * (Real.cos θ) + m = 0) :
  (m = (3 - 2 * Real.sqrt 3) / 2) ∧
  ((Real.cos θ - Real.sin θ * Real.tan θ) / (1 - Real.tan θ) = Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_trig_roots_theorem_l3159_315944


namespace NUMINAMATH_CALUDE_remainder_sum_l3159_315926

theorem remainder_sum (x y : ℤ) 
  (hx : x ≡ 47 [ZMOD 60])
  (hy : y ≡ 26 [ZMOD 45]) :
  x + y ≡ 13 [ZMOD 15] := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3159_315926


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l3159_315916

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_product_of_factors : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    is_factor a 60 ∧ 
    is_factor b 60 ∧ 
    ¬(is_factor (a * b) 60) ∧
    a * b = 8 ∧
    (∀ (c d : ℕ), 
      c ≠ d → 
      c > 0 → 
      d > 0 → 
      is_factor c 60 → 
      is_factor d 60 → 
      ¬(is_factor (c * d) 60) → 
      c * d ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l3159_315916


namespace NUMINAMATH_CALUDE_part_one_part_two_l3159_315952

noncomputable section

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Define the specific triangle with the given condition
def SpecialTriangle (t : Triangle) : Prop :=
  3 * t.a = 2 * t.b

-- Part I
theorem part_one (t : Triangle) (h : SpecialTriangle t) (hB : t.B = π/3) :
  Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6 :=
sorry

-- Part II
theorem part_two (t : Triangle) (h : SpecialTriangle t) (hC : Real.cos t.C = 2/3) :
  Real.sin (t.A - t.B) = -Real.sqrt 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3159_315952


namespace NUMINAMATH_CALUDE_point_symmetric_to_origin_l3159_315900

theorem point_symmetric_to_origin (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, 3 * a + 6)
  (|2 - a| = |3 * a + 6|) → 
  (∃ (x y : ℝ), (x = -3 ∧ y = -3) ∨ (x = -6 ∧ y = 6)) ∧ 
  ((-(2 - a), -(3 * a + 6)) = (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_point_symmetric_to_origin_l3159_315900


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l3159_315967

theorem min_value_quadratic (x : ℝ) : x^2 - 6*x + 10 ≥ 1 := by sorry

theorem min_value_achieved : ∃ x : ℝ, x^2 - 6*x + 10 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_achieved_l3159_315967


namespace NUMINAMATH_CALUDE_point_on_line_l3159_315922

/-- A point is on a line if it satisfies the equation of the line formed by two other points. -/
def is_on_line (x₁ y₁ x₂ y₂ x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The point (5, 56/3) is on the line formed by (8, 16) and (2, 0). -/
theorem point_on_line : is_on_line 8 16 2 0 5 (56/3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3159_315922


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l3159_315956

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_powers : units_digit ((35 ^ 7) + (93 ^ 45)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l3159_315956


namespace NUMINAMATH_CALUDE_sum_in_Q_l3159_315994

-- Define the sets P, Q, and M
def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k - 1}
def M : Set Int := {x | ∃ k, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_Q (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_in_Q_l3159_315994


namespace NUMINAMATH_CALUDE_eight_person_handshakes_l3159_315923

/-- The number of handshakes in a group where each person shakes hands with every other person once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 28 -/
theorem eight_person_handshakes : num_handshakes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_person_handshakes_l3159_315923


namespace NUMINAMATH_CALUDE_k_range_theorem_l3159_315963

def f (x : ℝ) : ℝ := x * abs x

theorem k_range_theorem (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ f (x - 2*k) < k) → k ∈ Set.Ioi (1/4) :=
by sorry

end NUMINAMATH_CALUDE_k_range_theorem_l3159_315963


namespace NUMINAMATH_CALUDE_min_value_theorem_l3159_315945

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3159_315945


namespace NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l3159_315909

/-- Represents a house in the square yard -/
inductive House
| NorthEast
| NorthWest
| SouthEast
| SouthWest

/-- Represents a quarrel between two friends -/
structure Quarrel where
  house1 : House
  house2 : House
  day : Nat

/-- Define what it means for two houses to be neighbors -/
def are_neighbors (h1 h2 : House) : Bool :=
  match h1, h2 with
  | House.NorthEast, House.NorthWest => true
  | House.NorthEast, House.SouthEast => true
  | House.NorthWest, House.SouthWest => true
  | House.SouthEast, House.SouthWest => true
  | House.NorthWest, House.NorthEast => true
  | House.SouthEast, House.NorthEast => true
  | House.SouthWest, House.NorthWest => true
  | House.SouthWest, House.SouthEast => true
  | _, _ => false

/-- The main theorem to prove -/
theorem quarrel_between_opposite_houses 
  (total_friends : Nat) 
  (quarrels : List Quarrel) 
  (h1 : total_friends = 77)
  (h2 : quarrels.length = 365)
  (h3 : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h4 : ∀ h1 h2 : House, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1)) :
  ∃ q ∈ quarrels, ¬are_neighbors q.house1 q.house2 := by
sorry

end NUMINAMATH_CALUDE_quarrel_between_opposite_houses_l3159_315909


namespace NUMINAMATH_CALUDE_abc_is_cube_l3159_315986

theorem abc_is_cube (a b c : ℤ) (h : (a / b) + (b / c) + (c / a) = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_abc_is_cube_l3159_315986


namespace NUMINAMATH_CALUDE_intersection_A_B_l3159_315910

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set A
def A : Set ℕ := {x ∈ PositiveInt | x ≤ Real.exp 1}

-- Define set B
def B : Set ℕ := {0, 1, 2, 3}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3159_315910


namespace NUMINAMATH_CALUDE_only_two_digit_cyclic_divisible_number_l3159_315950

theorem only_two_digit_cyclic_divisible_number : ∀ (a b : ℕ),
  a ≠ 0 → b ≠ 0 →
  (10 * a + b) % (10 * b + a) = 0 →
  a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_two_digit_cyclic_divisible_number_l3159_315950


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l3159_315930

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 3 * y - 3 = 0
def l2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 ↔ 
      (∃ x0 y0 : ℝ, l1 x0 y0 ∧ l2 x0 y0 ∧ 
        (y - y0 = -(a/b) * (x - x0))) ∧
      (∃ k : ℝ, a/b = -2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l3159_315930


namespace NUMINAMATH_CALUDE_charles_vowel_learning_time_l3159_315979

/-- The number of days Charles takes to learn one alphabet -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end NUMINAMATH_CALUDE_charles_vowel_learning_time_l3159_315979


namespace NUMINAMATH_CALUDE_book_price_changes_l3159_315988

theorem book_price_changes (initial_price : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) (final_price : ℝ) : 
  initial_price = 400 →
  decrease_percent = 15 →
  increase_percent = 40 →
  final_price = 476 →
  initial_price * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = final_price := by
sorry

end NUMINAMATH_CALUDE_book_price_changes_l3159_315988


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3159_315929

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c :=  -- The ratio of b's age to c's age is 2:1
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3159_315929


namespace NUMINAMATH_CALUDE_problem_statement_l3159_315983

theorem problem_statement (a b : ℝ) (h : 2*a + b + 1 = 0) : 1 + 4*a + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3159_315983


namespace NUMINAMATH_CALUDE_special_square_area_l3159_315955

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The length of BR
  br : ℝ
  -- The length of PR
  pr : ℝ
  -- Assumption that BR = 9
  br_eq : br = 9
  -- Assumption that PR = 12
  pr_eq : pr = 12
  -- Assumption that BP and CQ intersect at right angles
  right_angle : True

/-- The theorem stating that the area of the special square is 324 -/
theorem special_square_area (s : SpecialSquare) : s.side ^ 2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_l3159_315955


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3159_315924

theorem inequality_solution_set :
  {x : ℝ | 3 * x - 4 > 2} = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3159_315924


namespace NUMINAMATH_CALUDE_min_neighbors_2005_points_l3159_315943

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for its endpoints to be considered neighbors -/
def max_neighbor_angle : ℝ := 10

/-- A function that calculates the minimum number of neighbor pairs given the number of points and maximum neighbor angle -/
noncomputable def min_neighbor_pairs (n : ℕ) (max_angle : ℝ) : ℕ := sorry

/-- Theorem stating that the minimum number of neighbor pairs for 2005 points with a 10° maximum angle is 56430 -/
theorem min_neighbors_2005_points :
  min_neighbor_pairs num_points max_neighbor_angle = 56430 := by sorry

end NUMINAMATH_CALUDE_min_neighbors_2005_points_l3159_315943


namespace NUMINAMATH_CALUDE_library_visitors_l3159_315992

theorem library_visitors (non_sunday_avg : ℕ) (monthly_avg : ℕ) (month_days : ℕ) (sundays : ℕ) :
  non_sunday_avg = 240 →
  monthly_avg = 285 →
  month_days = 30 →
  sundays = 5 →
  (sundays * (monthly_avg * month_days - non_sunday_avg * (month_days - sundays))) / sundays = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l3159_315992


namespace NUMINAMATH_CALUDE_shaded_area_is_24_5_l3159_315996

/-- Represents the structure of the grid --/
structure Grid :=
  (rect1 : Int × Int)
  (rect2 : Int × Int)
  (rect3 : Int × Int)

/-- Calculates the area of a rectangle --/
def rectangleArea (dims : Int × Int) : Int :=
  dims.1 * dims.2

/-- Calculates the total area of the grid --/
def totalGridArea (g : Grid) : Int :=
  rectangleArea g.rect1 + rectangleArea g.rect2 + rectangleArea g.rect3

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : Int) : Rat :=
  (base * height) / 2

/-- The main theorem stating the area of the shaded region --/
theorem shaded_area_is_24_5 (g : Grid) 
    (h1 : g.rect1 = (3, 4))
    (h2 : g.rect2 = (4, 5))
    (h3 : g.rect3 = (5, 6))
    (h4 : totalGridArea g = 62)
    (h5 : triangleArea 15 5 = 37.5) :
  totalGridArea g - triangleArea 15 5 = 24.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_24_5_l3159_315996


namespace NUMINAMATH_CALUDE_evaluate_expression_l3159_315927

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3159_315927


namespace NUMINAMATH_CALUDE_stating_tour_cost_is_correct_l3159_315958

/-- Represents the cost of a tour at an aqua park -/
def tour_cost : ℝ := 6

/-- Represents the admission fee for the aqua park -/
def admission_fee : ℝ := 12

/-- Represents the total number of people in the first group -/
def group1_size : ℕ := 10

/-- Represents the total number of people in the second group -/
def group2_size : ℕ := 5

/-- Represents the total earnings of the aqua park -/
def total_earnings : ℝ := 240

/-- 
Theorem stating that the tour cost is correct given the problem conditions
-/
theorem tour_cost_is_correct : 
  group1_size * (admission_fee + tour_cost) + group2_size * admission_fee = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_stating_tour_cost_is_correct_l3159_315958


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_value_for_minimum_four_l3159_315941

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

theorem a_value_for_minimum_four :
  ∀ a : ℝ, 0 < a → a < 5 → (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m ∧ (∃ y : ℝ, f a y = m) ∧ m = 4) → a = 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_value_for_minimum_four_l3159_315941


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3159_315982

theorem difference_of_squares_factorization (y : ℝ) :
  49 - 16 * y^2 = (7 - 4*y) * (7 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3159_315982


namespace NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l3159_315995

theorem four_integers_product_2002_sum_less_40 :
  ∀ (a b c d : ℕ+),
    a * b * c * d = 2002 →
    (a : ℕ) + b + c + d < 40 →
    ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
     (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
     (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
     (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
     (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
     (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
     (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
     (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) ∨
     (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 2 ∧ c = 7 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 2 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 13 ∧ c = 2 ∧ d = 7) ∨
     (a = 11 ∧ b = 13 ∧ c = 7 ∧ d = 2) ∨
     (a = 13 ∧ b = 2 ∧ c = 7 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 2 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 11 ∧ d = 2) ∨
     (a = 13 ∧ b = 11 ∧ c = 2 ∧ d = 7) ∨
     (a = 13 ∧ b = 11 ∧ c = 7 ∧ d = 2) ∨
     (a = 14 ∧ b = 1 ∧ c = 11 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 1 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 1 ∧ c = 14 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 1 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 13 ∧ c = 1 ∧ d = 14) ∨
     (a = 11 ∧ b = 13 ∧ c = 14 ∧ d = 1) ∨
     (a = 13 ∧ b = 1 ∧ c = 14 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 1 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 11 ∧ d = 1) ∨
     (a = 13 ∧ b = 11 ∧ c = 1 ∧ d = 14) ∨
     (a = 13 ∧ b = 11 ∧ c = 14 ∧ d = 1)) :=
by sorry

end NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l3159_315995


namespace NUMINAMATH_CALUDE_solution_difference_l3159_315925

theorem solution_difference (a b : ℝ) : 
  ((a - 4) * (a + 4) = 28 * a - 112) → 
  ((b - 4) * (b + 4) = 28 * b - 112) → 
  a ≠ b →
  a > b →
  a - b = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3159_315925


namespace NUMINAMATH_CALUDE_first_month_sale_l3159_315978

theorem first_month_sale
  (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale2 = 6927)
  (h2 : sale3 = 6855)
  (h3 : sale4 = 7230)
  (h4 : sale5 = 6562)
  (h5 : sale6 = 6191)
  (h6 : avg_sale = 6700) :
  6 * avg_sale - (sale2 + sale3 + sale4 + sale5 + sale6) = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l3159_315978


namespace NUMINAMATH_CALUDE_min_value_problem_l3159_315968

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3159_315968


namespace NUMINAMATH_CALUDE_adult_admission_price_l3159_315911

/-- Proves that the adult admission price was 60 cents given the conditions -/
theorem adult_admission_price (total_attendance : ℕ) (child_ticket_price : ℕ) 
  (children_attended : ℕ) (total_revenue : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_adult_admission_price_l3159_315911


namespace NUMINAMATH_CALUDE_tangent_chord_equation_l3159_315971

/-- Given a point P(4, -5) outside the circle x^2 + y^2 = 4, 
    the equation of the line containing the points of tangency 
    of the tangents from P to the circle is 4x - 5y = 4. -/
theorem tangent_chord_equation :
  let P : ℝ × ℝ := (4, -5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  ∀ (x y : ℝ), (x, y) ∈ circle → (4 * x - 5 * y = 4) ↔ 
    ∃ (A B : ℝ × ℝ), A ∈ circle ∧ B ∈ circle ∧
    (∀ t : ℝ, (1 - t) • P + t • A ∉ circle) ∧
    (∀ t : ℝ, (1 - t) • P + t • B ∉ circle) ∧
    (x, y) = (1 - t) • A + t • B :=
by sorry

end NUMINAMATH_CALUDE_tangent_chord_equation_l3159_315971


namespace NUMINAMATH_CALUDE_log_5_12_in_terms_of_m_n_l3159_315938

theorem log_5_12_in_terms_of_m_n (m n : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = m) 
  (h2 : Real.log 3 / Real.log 10 = n) : 
  Real.log 12 / Real.log 5 = (2*m + n) / (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_log_5_12_in_terms_of_m_n_l3159_315938


namespace NUMINAMATH_CALUDE_net_weekly_increase_is_five_l3159_315914

/-- Calculates the net weekly increase in earnings given a raise, work hours, and housing benefit reduction -/
def netWeeklyIncrease (raise : ℚ) (workHours : ℕ) (housingBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := raise * workHours
  let weeklyHousingBenefitReduction := housingBenefitReduction / 4
  weeklyRaise - weeklyHousingBenefitReduction

/-- Theorem stating that given the specified conditions, the net weekly increase is $5 -/
theorem net_weekly_increase_is_five :
  netWeeklyIncrease (1/2) 40 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_net_weekly_increase_is_five_l3159_315914


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3159_315931

theorem minimum_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 : ℝ) = Real.sqrt (2^a * 2^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ) = Real.sqrt (2^x * 2^y) → 1/a + 1/b ≤ 1/x + 1/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2 : ℝ) = Real.sqrt (2^x * 2^y) ∧ 1/a + 1/b = 1/x + 1/y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3159_315931


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l3159_315993

/-- A type representing digits from 1 to 9 -/
inductive Digit : Type
  | one : Digit
  | two : Digit
  | three : Digit
  | four : Digit
  | five : Digit
  | six : Digit
  | seven : Digit
  | eight : Digit
  | nine : Digit

/-- Convert a Digit to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- Convert three Digits to a natural number -/
def three_digit_to_nat (e f g : Digit) : ℕ :=
  100 * (digit_to_nat e) + 10 * (digit_to_nat f) + (digit_to_nat g)

/-- The main theorem -/
theorem unique_digit_arrangement :
  ∃! (a b c d e f g h : Digit),
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ h) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ h) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ h) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧ (d ≠ h) ∧
    (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ h) ∧
    (f ≠ g) ∧ (f ≠ h) ∧
    (g ≠ h) ∧
    ((digit_to_nat a) / (digit_to_nat b) = (digit_to_nat c) / (digit_to_nat d)) ∧
    ((digit_to_nat c) / (digit_to_nat d) = (three_digit_to_nat e f g) / (10 * (digit_to_nat h) + 9)) :=
by sorry


end NUMINAMATH_CALUDE_unique_digit_arrangement_l3159_315993


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l3159_315901

/-- The maximum area of a rectangular garden with 150 feet of fencing and natural number side lengths -/
theorem max_area_rectangular_garden :
  ∃ (l w : ℕ), 
    (2 * l + 2 * w = 150) ∧ 
    (∀ (a b : ℕ), (2 * a + 2 * b = 150) → (a * b ≤ l * w)) ∧ 
    (l * w = 1406) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l3159_315901


namespace NUMINAMATH_CALUDE_real_numbers_closed_closed_set_contains_zero_l3159_315919

-- Definition of a closed set
def is_closed_set (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Theorem 1: The set of real numbers is a closed set
theorem real_numbers_closed : is_closed_set Set.univ := by sorry

-- Theorem 2: If S is a closed set, then 0 is an element of S
theorem closed_set_contains_zero (S : Set ℝ) (h : is_closed_set S) (h_nonempty : S.Nonempty) : 
  (0 : ℝ) ∈ S := by sorry

end NUMINAMATH_CALUDE_real_numbers_closed_closed_set_contains_zero_l3159_315919


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_4_l3159_315976

theorem factorization_of_4x_squared_minus_4 (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_4_l3159_315976


namespace NUMINAMATH_CALUDE_duck_cow_problem_l3159_315962

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 24 → C = 12 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l3159_315962


namespace NUMINAMATH_CALUDE_ratio_equality_l3159_315921

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ c ≠ 0) :
  (a + b) / c = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3159_315921


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l3159_315907

theorem theater_ticket_cost (adult_price : ℝ) : 
  (5 * adult_price + 4 * (adult_price / 2) = 24.50) → 
  (8 * adult_price + 6 * (adult_price / 2) = 38.50) := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_l3159_315907


namespace NUMINAMATH_CALUDE_sum_irrational_implies_component_irrational_l3159_315990

theorem sum_irrational_implies_component_irrational (a b c : ℝ) : 
  ¬ (∃ (q : ℚ), (a + b + c : ℝ) = q) → 
  ¬ (∃ (q₁ q₂ q₃ : ℚ), (a = q₁ ∧ b = q₂ ∧ c = q₃)) :=
by sorry

end NUMINAMATH_CALUDE_sum_irrational_implies_component_irrational_l3159_315990


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_l3159_315913

theorem complex_fourth_quadrant (m : ℝ) :
  (∃ z : ℂ, z = (m + Complex.I) / (1 + Complex.I) ∧ 
   z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_l3159_315913


namespace NUMINAMATH_CALUDE_expression_inequality_l3159_315961

theorem expression_inequality : 
  let x : ℚ := 3 + 1/10 + 4/100
  let y : ℚ := 3 + 5/110
  x ≠ y :=
by
  sorry

end NUMINAMATH_CALUDE_expression_inequality_l3159_315961


namespace NUMINAMATH_CALUDE_no_nonzero_triple_sum_equals_third_l3159_315969

theorem no_nonzero_triple_sum_equals_third : 
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    (a + b = c) ∧ (b + c = a) ∧ (c + a = b) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_triple_sum_equals_third_l3159_315969


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l3159_315973

theorem complex_exp_thirteen_pi_over_two : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l3159_315973


namespace NUMINAMATH_CALUDE_symmetry_sum_theorem_l3159_315939

/-- Properties of a regular 25-gon -/
structure RegularPolygon25 where
  /-- Number of lines of symmetry -/
  L : ℕ
  /-- Smallest positive angle for rotational symmetry in degrees -/
  R : ℝ
  /-- The polygon has 25 sides -/
  sides_eq : L = 25
  /-- The smallest rotational symmetry angle is 360/25 degrees -/
  angle_eq : R = 360 / 25

/-- Theorem about the sum of symmetry lines and half the rotational angle -/
theorem symmetry_sum_theorem (p : RegularPolygon25) :
  p.L + p.R / 2 = 32.2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_theorem_l3159_315939


namespace NUMINAMATH_CALUDE_orchestra_only_females_l3159_315912

theorem orchestra_only_females (
  band_females : ℕ) (band_males : ℕ) 
  (orchestra_females : ℕ) (orchestra_males : ℕ)
  (both_females : ℕ) (both_males : ℕ)
  (total_students : ℕ) :
  band_females = 120 →
  band_males = 110 →
  orchestra_females = 100 →
  orchestra_males = 130 →
  both_females = 90 →
  both_males = 80 →
  total_students = 280 →
  total_students = band_females + band_males + orchestra_females + orchestra_males - both_females - both_males →
  orchestra_females - both_females = 10 := by
sorry

end NUMINAMATH_CALUDE_orchestra_only_females_l3159_315912


namespace NUMINAMATH_CALUDE_ten_men_absent_l3159_315981

/-- Represents the work scenario with men and days -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ

/-- Calculates the number of absent men given a work scenario -/
def absentMen (w : WorkScenario) : ℕ :=
  w.totalMen - (w.totalMen * w.originalDays) / w.actualDays

/-- The theorem stating that 10 men became absent in the given scenario -/
theorem ten_men_absent : absentMen ⟨60, 50, 60⟩ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_men_absent_l3159_315981


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3159_315949

theorem other_root_of_complex_quadratic (z : ℂ) :
  z = 4 + 7*I ∧ z^2 = -73 + 24*I → (-z)^2 = -73 + 24*I := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3159_315949


namespace NUMINAMATH_CALUDE_die_roll_probability_l3159_315974

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling a different number from the previous roll -/
def p : ℚ := (n - 1) / n

/-- The probability of rolling the same number as the previous roll -/
def q : ℚ := 1 / n

theorem die_roll_probability : 
  p^(r - 2) * q = 282475249 / 8589934592 := by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3159_315974


namespace NUMINAMATH_CALUDE_multiple_of_seven_square_gt_200_lt_30_l3159_315935

theorem multiple_of_seven_square_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 7 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 21 ∨ x = 28 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_seven_square_gt_200_lt_30_l3159_315935


namespace NUMINAMATH_CALUDE_grilled_cheese_slices_l3159_315959

/-- The number of ham sandwiches made -/
def num_ham_sandwiches : ℕ := 10

/-- The number of grilled cheese sandwiches made -/
def num_grilled_cheese : ℕ := 10

/-- The number of cheese slices used in one ham sandwich -/
def cheese_per_ham : ℕ := 2

/-- The total number of cheese slices used -/
def total_cheese : ℕ := 50

/-- The number of cheese slices in one grilled cheese sandwich -/
def cheese_per_grilled_cheese : ℕ := (total_cheese - num_ham_sandwiches * cheese_per_ham) / num_grilled_cheese

theorem grilled_cheese_slices :
  cheese_per_grilled_cheese = 3 :=
sorry

end NUMINAMATH_CALUDE_grilled_cheese_slices_l3159_315959


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3159_315977

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop := sorry

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  Midpoint E B C →
  PointOnSegment D A C →
  Length A C = 1 →
  Angle B A C = π / 3 →  -- 60°
  Angle A B C = 5 * π / 9 →  -- 100°
  Angle A C B = π / 9 →  -- 20°
  Angle D E C = 4 * π / 9 →  -- 80°
  TriangleArea A B C + 2 * TriangleArea C D E = Real.sqrt 3 / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3159_315977


namespace NUMINAMATH_CALUDE_students_not_in_program_x_l3159_315905

/-- Represents a grade level in the school -/
inductive GradeLevel
  | Elementary
  | Middle
  | High

/-- Represents the gender of students -/
inductive Gender
  | Girl
  | Boy

/-- The number of students in each grade level and gender -/
def studentCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 192
  | GradeLevel.Elementary, Gender.Boy => 135
  | GradeLevel.Middle, Gender.Girl => 233
  | GradeLevel.Middle, Gender.Boy => 163
  | GradeLevel.High, Gender.Girl => 117
  | GradeLevel.High, Gender.Boy => 89

/-- The number of students in Program X for each grade level and gender -/
def programXCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 48
  | GradeLevel.Elementary, Gender.Boy => 28
  | GradeLevel.Middle, Gender.Girl => 98
  | GradeLevel.Middle, Gender.Boy => 51
  | GradeLevel.High, Gender.Girl => 40
  | GradeLevel.High, Gender.Boy => 25

/-- The total number of students not participating in Program X -/
def studentsNotInProgramX : ℕ :=
  (studentCount GradeLevel.Elementary Gender.Girl - programXCount GradeLevel.Elementary Gender.Girl) +
  (studentCount GradeLevel.Elementary Gender.Boy - programXCount GradeLevel.Elementary Gender.Boy) +
  (studentCount GradeLevel.Middle Gender.Girl - programXCount GradeLevel.Middle Gender.Girl) +
  (studentCount GradeLevel.Middle Gender.Boy - programXCount GradeLevel.Middle Gender.Boy) +
  (studentCount GradeLevel.High Gender.Girl - programXCount GradeLevel.High Gender.Girl) +
  (studentCount GradeLevel.High Gender.Boy - programXCount GradeLevel.High Gender.Boy)

theorem students_not_in_program_x :
  studentsNotInProgramX = 639 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_program_x_l3159_315905


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3159_315936

theorem purely_imaginary_complex_number (a : ℝ) :
  (a^2 - 4 : ℂ) + (a - 2 : ℂ) * Complex.I = (0 : ℂ) + (b : ℂ) * Complex.I ∧ b ≠ 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3159_315936


namespace NUMINAMATH_CALUDE_distance_to_hypotenuse_l3159_315985

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The distance from the intersection point of the medians to one leg -/
  dist1 : ℝ
  /-- The distance from the intersection point of the medians to the other leg -/
  dist2 : ℝ
  /-- Ensure the triangle is not degenerate -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  /-- The distances from the intersection point to the legs are positive -/
  dist1_pos : 0 < dist1
  dist2_pos : 0 < dist2
  /-- The given distances from the intersection point to the legs -/
  dist1_eq : dist1 = 3
  dist2_eq : dist2 = 4

/-- The theorem to be proved -/
theorem distance_to_hypotenuse (t : RightTriangle) : 
  let hypotenuse := Real.sqrt (t.leg1^2 + t.leg2^2)
  let area := t.leg1 * t.leg2 / 2
  area / hypotenuse = 12/5 := by sorry

end NUMINAMATH_CALUDE_distance_to_hypotenuse_l3159_315985


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3159_315954

theorem algebraic_expression_value
  (m n p q x : ℝ)
  (h1 : m = -n)
  (h2 : p * q = 1)
  (h3 : |x| = 2) :
  (m + n) / 2022 + 2023 * p * q + x^2 = 2027 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3159_315954


namespace NUMINAMATH_CALUDE_bailey_towel_cost_l3159_315932

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.20

theorem bailey_towel_cost : 
  let total_before_discount := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_rate * total_before_discount
  let final_cost := total_before_discount - discount_amount
  final_cost = 224 := by sorry

end NUMINAMATH_CALUDE_bailey_towel_cost_l3159_315932


namespace NUMINAMATH_CALUDE_volume_error_percentage_l3159_315940

theorem volume_error_percentage (L W H : ℝ) (L_meas W_meas H_meas : ℝ)
  (h_L : L_meas = 1.08 * L)
  (h_W : W_meas = 1.12 * W)
  (h_H : H_meas = 1.05 * H) :
  let V_true := L * W * H
  let V_calc := L_meas * W_meas * H_meas
  (V_calc - V_true) / V_true * 100 = 25.424 := by
sorry

end NUMINAMATH_CALUDE_volume_error_percentage_l3159_315940


namespace NUMINAMATH_CALUDE_sara_book_cost_l3159_315903

/-- The cost of Sara's first book -/
def first_book_cost : ℝ := sorry

/-- The cost of Sara's second book -/
def second_book_cost : ℝ := 6.5

/-- The amount Sara paid -/
def amount_paid : ℝ := 20

/-- The change Sara received -/
def change_received : ℝ := 8

theorem sara_book_cost : first_book_cost = 5.5 := by sorry

end NUMINAMATH_CALUDE_sara_book_cost_l3159_315903


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l3159_315966

theorem enclosing_polygons_sides (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 / m : ℚ) / 2 = 360 / n →
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l3159_315966


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l3159_315918

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 8

/-- The mass of the first alloy in kg -/
def mass_first : ℝ := 15

/-- The mass of the second alloy in kg -/
def mass_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 9.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first / 100) * mass_first + 
  (chromium_percentage_second / 100) * mass_second = 
  (chromium_percentage_result / 100) * (mass_first + mass_second) :=
by sorry

#check chromium_percentage_calculation

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l3159_315918


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3159_315957

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 265 →
  divisor = 22 →
  quotient = 12 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3159_315957


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3159_315972

/-- Proves that for a rectangle with length-to-width ratio of 5:2 and diagonal length d,
    the area A can be expressed as A = (10/29)d² -/
theorem rectangle_area_diagonal (l w d A : ℝ) (h1 : l / w = 5 / 2) (h2 : l ^ 2 + w ^ 2 = d ^ 2) 
    (h3 : A = l * w) : A = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3159_315972


namespace NUMINAMATH_CALUDE_parabola_point_y_coord_l3159_315947

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y
  distance_to_focus : (x - 0)^2 + (y - 1)^2 = 2^2

/-- Theorem: The y-coordinate of a point on the parabola x^2 = 4y that is 2 units away from the focus (0, 1) is 1 -/
theorem parabola_point_y_coord (P : ParabolaPoint) : P.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_y_coord_l3159_315947


namespace NUMINAMATH_CALUDE_game_choices_l3159_315975

theorem game_choices (p : ℝ) (n : ℕ) 
  (h1 : p = 0.9375) 
  (h2 : p = 1 - 1 / n) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_game_choices_l3159_315975


namespace NUMINAMATH_CALUDE_annual_growth_rate_for_doubling_l3159_315915

theorem annual_growth_rate_for_doubling (x : ℝ) (y : ℝ) (h : x > 0) :
  x * (1 + y)^2 = 2*x → y = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_for_doubling_l3159_315915


namespace NUMINAMATH_CALUDE_candy_bar_sales_theorem_l3159_315946

/-- Calculates the total money earned from candy bar sales given the number of members,
    average number of candy bars sold per member, and the cost per candy bar. -/
def total_money_earned (num_members : ℕ) (avg_bars_per_member : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (num_members * avg_bars_per_member : ℚ) * cost_per_bar

/-- Proves that a group of 20 members selling an average of 8 candy bars at $0.50 each
    earns a total of $80 from their sales. -/
theorem candy_bar_sales_theorem :
  total_money_earned 20 8 (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_theorem_l3159_315946


namespace NUMINAMATH_CALUDE_sams_german_shepherds_l3159_315904

theorem sams_german_shepherds (sam_french_bulldogs peter_total : ℕ) 
  (h1 : sam_french_bulldogs = 4)
  (h2 : peter_total = 17)
  (h3 : ∃ (sam_german_shepherds : ℕ), 
    3 * sam_german_shepherds + 2 * sam_french_bulldogs = peter_total) :
  ∃ (sam_german_shepherds : ℕ), sam_german_shepherds = 3 :=
by sorry

end NUMINAMATH_CALUDE_sams_german_shepherds_l3159_315904


namespace NUMINAMATH_CALUDE_CD_length_approx_l3159_315997

/-- A quadrilateral with intersecting diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)
  (BO : ℝ)
  (OD : ℝ)
  (AO : ℝ)
  (OC : ℝ)
  (AB : ℝ)

/-- The length of CD in the quadrilateral -/
def CD_length (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the length of CD in the given quadrilateral -/
theorem CD_length_approx (q : Quadrilateral) 
  (h1 : q.BO = 3)
  (h2 : q.OD = 5)
  (h3 : q.AO = 7)
  (h4 : q.OC = 4)
  (h5 : q.AB = 5) :
  ∃ ε > 0, |CD_length q - 8.51| < ε :=
sorry

end NUMINAMATH_CALUDE_CD_length_approx_l3159_315997


namespace NUMINAMATH_CALUDE_rainfall_second_week_l3159_315998

/-- Proves that given a total rainfall of 40 inches over two weeks, 
    where the second week's rainfall is 1.5 times the first week's, 
    the rainfall in the second week is 24 inches. -/
theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) : 
  total_rainfall = 40 ∧ ratio = 1.5 → 
  ∃ (first_week : ℝ), 
    first_week + ratio * first_week = total_rainfall ∧ 
    ratio * first_week = 24 := by
  sorry

#check rainfall_second_week

end NUMINAMATH_CALUDE_rainfall_second_week_l3159_315998


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l3159_315989

/-- Represents a quilt block as described in the problem -/
structure QuiltBlock where
  size : Nat
  fully_shaded : Nat
  half_shaded : Nat
  quarter_shaded : Nat

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2) / (q.size * q.size)

/-- The specific quilt block described in the problem -/
def problem_quilt : QuiltBlock :=
  { size := 4
    fully_shaded := 4
    half_shaded := 8
    quarter_shaded := 4 }

theorem quilt_shaded_fraction :
  shaded_fraction problem_quilt = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l3159_315989


namespace NUMINAMATH_CALUDE_minimum_guests_l3159_315964

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 406) (h2 : max_per_guest = 2.5) :
  ∃ n : ℕ, n * max_per_guest ≥ total_food ∧ ∀ m : ℕ, m * max_per_guest ≥ total_food → m ≥ n ∧ n = 163 :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3159_315964


namespace NUMINAMATH_CALUDE_box_surface_area_is_288_l3159_315917

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_side : ℕ) : ℕ :=
  let new_length := sheet_length - 2 * corner_side
  let new_width := sheet_width - 2 * corner_side
  new_length * new_width

/-- Theorem: The surface area of the interior of the open box is 288 square units. -/
theorem box_surface_area_is_288 :
  interior_surface_area 36 24 6 = 288 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_is_288_l3159_315917


namespace NUMINAMATH_CALUDE_complex_solutions_count_l3159_315908

theorem complex_solutions_count : 
  ∃! (s : Finset ℂ), s.card = 2 ∧ 
  (∀ z ∈ s, (z^4 - 1) / (z^3 + z^2 - 2*z) = 0) ∧
  (∀ z : ℂ, (z^4 - 1) / (z^3 + z^2 - 2*z) = 0 → z ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l3159_315908


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l3159_315980

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  bike : ℕ
  jog : ℕ
  swim : ℕ

/-- The problem statement -/
theorem rates_sum_of_squares (r : Rates) : r.bike^2 + r.jog^2 + r.swim^2 = 314 :=
  by
  have h1 : 2 * r.bike + 3 * r.jog + 4 * r.swim = 74 := by sorry
  have h2 : 4 * r.bike + 2 * r.jog + 3 * r.swim = 91 := by sorry
  sorry

#check rates_sum_of_squares

end NUMINAMATH_CALUDE_rates_sum_of_squares_l3159_315980


namespace NUMINAMATH_CALUDE_f_difference_l3159_315942

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 5 * x - 4

theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3159_315942


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l3159_315991

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) 
  (total : v + a + d = 93)
  (v_more_than_a : v = a + 9)
  (v_less_than_d : v = d - 9) :
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l3159_315991


namespace NUMINAMATH_CALUDE_shirts_per_pants_l3159_315960

/-- 
Given:
- Mr. Jones has 40 pants.
- The total number of pieces of clothes he owns is 280.
- Mr. Jones has a certain number of shirts for every pair of pants.

Prove that Mr. Jones has 6 shirts for every pair of pants.
-/
theorem shirts_per_pants (num_pants : ℕ) (total_clothes : ℕ) (shirts_per_pants : ℕ) : 
  num_pants = 40 → total_clothes = 280 → shirts_per_pants * num_pants + num_pants = total_clothes → 
  shirts_per_pants = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_pants_l3159_315960


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l3159_315953

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (a : ℝ) : ℝ :=
  (-a) * binomial_coefficient 9 1

theorem binomial_expansion_theorem (a : ℝ) :
  binomial_expansion_coefficient a = 36 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l3159_315953


namespace NUMINAMATH_CALUDE_base6_multiplication_l3159_315970

/-- Converts a base 6 number to base 10 --/
def toBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem base6_multiplication :
  let a := [2, 3, 1]  -- 132₆ in reverse order
  let b := [4, 1]     -- 14₆ in reverse order
  toBase6 (toBase10 a * toBase10 b) = [2, 3, 3, 2] := by
  sorry


end NUMINAMATH_CALUDE_base6_multiplication_l3159_315970


namespace NUMINAMATH_CALUDE_largest_divisible_by_eight_l3159_315902

theorem largest_divisible_by_eight (A B C : ℕ) : 
  A = 8 * B + C → 
  B = C → 
  C < 8 → 
  (∃ k : ℕ, A = 8 * k) → 
  A ≤ 63 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_eight_l3159_315902
