import Mathlib

namespace parallel_vectors_m_value_l893_89354

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then m = -3 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  parallel a b → m = -3 :=
by sorry

end parallel_vectors_m_value_l893_89354


namespace max_value_of_expression_l893_89322

/-- An arithmetic sequence with positive first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  a₁_pos : 0 < a₁
  d_pos : 0 < d

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem max_value_of_expression (seq : ArithmeticSequence)
    (h1 : seq.nthTerm 1 + seq.nthTerm 2 ≤ 60)
    (h2 : seq.nthTerm 2 + seq.nthTerm 3 ≤ 100) :
    5 * seq.nthTerm 1 + seq.nthTerm 5 ≤ 200 := by
  sorry

end max_value_of_expression_l893_89322


namespace sin_300_degrees_l893_89318

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l893_89318


namespace scientific_notation_equivalence_l893_89311

theorem scientific_notation_equivalence :
  216000 = 2.16 * (10 ^ 5) :=
by sorry

end scientific_notation_equivalence_l893_89311


namespace g_zero_at_neg_three_iff_s_eq_neg_192_l893_89388

/-- The function g(x) defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

/-- Theorem stating that g(-3) = 0 if and only if s = -192 -/
theorem g_zero_at_neg_three_iff_s_eq_neg_192 :
  ∀ s : ℝ, g (-3) s = 0 ↔ s = -192 := by sorry

end g_zero_at_neg_three_iff_s_eq_neg_192_l893_89388


namespace myrtle_absence_duration_l893_89387

/-- Proves that Myrtle was gone for 21 days given the conditions of the problem -/
theorem myrtle_absence_duration (daily_production neighbor_took dropped remaining : ℕ) 
  (h1 : daily_production = 3)
  (h2 : neighbor_took = 12)
  (h3 : dropped = 5)
  (h4 : remaining = 46) :
  ∃ d : ℕ, d * daily_production - neighbor_took - dropped = remaining ∧ d = 21 := by
  sorry

end myrtle_absence_duration_l893_89387


namespace exact_calculation_equals_rounded_l893_89309

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  if x % 100 < 50 then x - (x % 100) else x + (100 - (x % 100))

theorem exact_calculation_equals_rounded : round_to_nearest_hundred (63 + 48 - 21) = 100 := by
  sorry

end exact_calculation_equals_rounded_l893_89309


namespace sum_of_tangent_slopes_l893_89302

/-- The parabola P with equation y = x^2 + 10x -/
def P (x y : ℝ) : Prop := y = x^2 + 10 * x

/-- The point Q (10, 5) -/
def Q : ℝ × ℝ := (10, 5)

/-- A line through Q with slope m -/
def line_through_Q (m : ℝ) (x y : ℝ) : Prop :=
  y - Q.2 = m * (x - Q.1)

/-- The sum of slopes of tangent lines to P passing through Q is 60 -/
theorem sum_of_tangent_slopes :
  ∃ r s : ℝ,
    (∀ m : ℝ, r < m ∧ m < s ↔
      ¬∃ x y : ℝ, P x y ∧ line_through_Q m x y) ∧
    r + s = 60 := by
  sorry

end sum_of_tangent_slopes_l893_89302


namespace equation_solution_l893_89383

theorem equation_solution : 
  ∃ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end equation_solution_l893_89383


namespace marias_cookies_l893_89375

theorem marias_cookies (x : ℕ) : 
  x ≥ 5 →
  (x - 5) % 2 = 0 →
  ((x - 5) / 2 - 2 = 5) →
  x = 19 := by
sorry

end marias_cookies_l893_89375


namespace periodic_odd_function_at_six_l893_89357

/-- An odd function that satisfies f(x+2) = -f(x) for all x -/
def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = -f x)

/-- For a periodic odd function f, f(6) = 0 -/
theorem periodic_odd_function_at_six (f : ℝ → ℝ) (h : periodic_odd_function f) : f 6 = 0 := by
  sorry

end periodic_odd_function_at_six_l893_89357


namespace sum_of_distances_l893_89331

theorem sum_of_distances (saham_distance mother_distance : ℝ) 
  (h1 : saham_distance = 2.6)
  (h2 : mother_distance = 5.98) :
  saham_distance + mother_distance = 8.58 := by
sorry

end sum_of_distances_l893_89331


namespace correct_calculation_l893_89355

theorem correct_calculation (a : ℝ) : 3 * a^2 - 2 * a^2 = a^2 := by
  sorry

end correct_calculation_l893_89355


namespace right_angle_vector_coord_l893_89330

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    if they form a right angle at B, then the y-coordinate of A is 5. -/
theorem right_angle_vector_coord (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (OB.1 * AB.1 + OB.2 * AB.2 = 0) → t = 5 := by
  sorry

end right_angle_vector_coord_l893_89330


namespace unique_number_property_l893_89314

theorem unique_number_property : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n.div 100 + n.mod 100 / 10 + n.mod 10 = 328 - n) :=
by
  -- The proof goes here
  sorry

end unique_number_property_l893_89314


namespace opposite_of_negative_one_l893_89384

theorem opposite_of_negative_one :
  (∀ x : ℤ, x + (-x) = 0) →
  -(-1) = 1 := by sorry

end opposite_of_negative_one_l893_89384


namespace upgraded_fraction_is_one_fourth_l893_89368

/-- Represents a satellite with modular units and sensors -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  upgraded_total : ℕ

/-- The fraction of upgraded sensors on a satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

/-- Theorem: The fraction of upgraded sensors is 1/4 under given conditions -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.upgraded_total / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

#eval upgraded_fraction { units := 24, non_upgraded_per_unit := 1, upgraded_total := 8 }

end upgraded_fraction_is_one_fourth_l893_89368


namespace chord_length_is_2_sqrt_2_l893_89376

/-- The circle with center (1, 1) and radius 2 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

/-- The line x - y + 2 = 0 -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 + 2 = 0}

/-- The length of the chord intercepted by line l on circle C -/
def chord_length : ℝ := sorry

/-- Theorem stating that the chord length is 2√2 -/
theorem chord_length_is_2_sqrt_2 : chord_length = 2 * Real.sqrt 2 := by sorry

end chord_length_is_2_sqrt_2_l893_89376


namespace sales_discount_effect_l893_89390

theorem sales_discount_effect (discount : ℝ) 
  (h1 : discount = 10)
  (h2 : (1 - discount / 100) * 1.12 = 1.008) : 
  discount = 10 := by
sorry

end sales_discount_effect_l893_89390


namespace wood_per_chair_l893_89305

def total_wood : ℕ := 672
def wood_per_table : ℕ := 12
def num_tables : ℕ := 24
def num_chairs : ℕ := 48

theorem wood_per_chair :
  (total_wood - num_tables * wood_per_table) / num_chairs = 8 := by
  sorry

end wood_per_chair_l893_89305


namespace triangle_perimeter_l893_89304

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 7) (hc : c = 5) :
  a + b + c = 22 := by
sorry

end triangle_perimeter_l893_89304


namespace mikes_typing_speed_reduction_l893_89310

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Mike's typing speed reduction is 20 words per minute. -/
theorem mikes_typing_speed_reduction :
  typing_speed_reduction 65 810 18 = 20 := by
  sorry

end mikes_typing_speed_reduction_l893_89310


namespace arithmetic_sequence_constant_ratio_l893_89326

/-- The sum of the first n terms of an arithmetic sequence -/
def S (a d : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- Theorem: If the ratio of S_{4n} to S_n is constant for an arithmetic sequence 
    with common difference 5, then the first term is 5/2 -/
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ c : ℚ, S a 5 (4 * n) / S a 5 n = c) →
  a = 5 / 2 := by
  sorry

end arithmetic_sequence_constant_ratio_l893_89326


namespace sin_alpha_plus_beta_l893_89346

theorem sin_alpha_plus_beta (α β t : ℝ) : 
  (Real.exp (α + π/6) - Real.exp (-α - π/6) + Real.cos (5*π/3 + α) = t) →
  (Real.exp (β - π/4) - Real.exp (π/4 - β) + Real.cos (5*π/4 + β) = -t) →
  Real.sin (α + β) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end sin_alpha_plus_beta_l893_89346


namespace students_behind_yoongi_l893_89386

theorem students_behind_yoongi (total_students : ℕ) (students_in_front : ℕ) : 
  total_students = 20 → students_in_front = 11 → total_students - (students_in_front + 1) = 8 := by
  sorry

end students_behind_yoongi_l893_89386


namespace intersection_of_A_and_B_l893_89317

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

end intersection_of_A_and_B_l893_89317


namespace parabola_line_intersection_l893_89344

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l with slope k passing through (-1, 0) -/
def line_l (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point on parabola C -/
def on_parabola_C (x y : ℝ) : Prop := parabola_C x y

/-- Point on line l -/
def on_line_l (k x y : ℝ) : Prop := line_l k x y

/-- Intersection ratio condition -/
def intersection_ratio (y₁ y₂ : ℝ) : Prop := y₁/y₂ + y₂/y₁ = 18

theorem parabola_line_intersection (k : ℝ) 
  (hk : k > 0)
  (hA : ∃ x₁ y₁, on_parabola_C x₁ y₁ ∧ on_line_l k x₁ y₁)
  (hB : ∃ x₂ y₂, on_parabola_C x₂ y₂ ∧ on_line_l k x₂ y₂)
  (hM : ∃ xₘ yₘ, on_parabola_C xₘ yₘ)
  (hN : ∃ xₙ yₙ, on_parabola_C xₙ yₙ)
  (h_ratio : ∀ y₁ y₂, intersection_ratio y₁ y₂) :
  k = Real.sqrt 5 / 5 := by
  sorry

end parabola_line_intersection_l893_89344


namespace parabola_and_line_problem_l893_89313

-- Define the parabola and directrix
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := y = x
def l₂ (x y : ℝ) : Prop := y = -x

-- Define the point E
def E : ℝ × ℝ := (4, 1)

-- Define the circle N
def circle_N (center : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = r^2

-- Theorem statement
theorem parabola_and_line_problem :
  -- Part 1: The coordinates of N are (2, 0)
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x y : ℝ), parabola p x y → 
    (∀ (x' : ℝ), directrix p x' → 
      ((x - x')^2 + y^2 = (x - 2)^2 + y^2))) ∧
  -- Part 2: No line l exists satisfying all conditions
  ¬∃ (m b : ℝ), 
    -- Define line l: y = mx + b
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      -- l intersects l₁ and l₂
      (y₁ = m*x₁ + b ∧ l₁ x₁ y₁) ∧
      (y₂ = m*x₂ + b ∧ l₂ x₂ y₂) ∧
      -- Midpoint of intersection points is E
      ((x₁ + x₂)/2 = E.1 ∧ (y₁ + y₂)/2 = E.2) ∧
      -- Chord length on circle N is 2
      (∃ (r : ℝ), 
        circle_N (2, 0) r 2 2 ∧ 
        circle_N (2, 0) r 2 (-2) ∧
        ∃ (x₃ y₃ x₄ y₄ : ℝ),
          y₃ = m*x₃ + b ∧ y₄ = m*x₄ + b ∧
          circle_N (2, 0) r x₃ y₃ ∧
          circle_N (2, 0) r x₄ y₄ ∧
          (x₃ - x₄)^2 + (y₃ - y₄)^2 = 4)) :=
sorry

end parabola_and_line_problem_l893_89313


namespace planting_probabilities_l893_89367

structure CropPlanting where
  transition_A : Fin 2 → ℚ
  transition_B : Fin 2 → ℚ
  transition_C : Fin 2 → ℚ

def planting : CropPlanting :=
  { transition_A := ![1/3, 2/3],
    transition_B := ![1/4, 3/4],
    transition_C := ![2/5, 3/5] }

def probability_A_third_given_B_first (p : CropPlanting) : ℚ :=
  p.transition_B 1 * p.transition_C 0

def distribution_X_given_A_first (p : CropPlanting) : Fin 2 → ℚ
  | 0 => p.transition_A 1 * p.transition_C 1 + p.transition_A 0 * p.transition_B 1
  | 1 => p.transition_A 1 * p.transition_C 0 + p.transition_A 0 * p.transition_B 0

def expectation_X_given_A_first (p : CropPlanting) : ℚ :=
  1 * distribution_X_given_A_first p 0 + 2 * distribution_X_given_A_first p 1

theorem planting_probabilities :
  probability_A_third_given_B_first planting = 3/10 ∧
  distribution_X_given_A_first planting 0 = 13/20 ∧
  distribution_X_given_A_first planting 1 = 7/20 ∧
  expectation_X_given_A_first planting = 27/20 := by
  sorry

end planting_probabilities_l893_89367


namespace inequality_range_l893_89378

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
sorry

end inequality_range_l893_89378


namespace absolute_value_inequality_l893_89377

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 6 ↔ -7/2 < x ∧ x < 5/2 := by
  sorry

end absolute_value_inequality_l893_89377


namespace rational_absolute_value_equation_l893_89328

theorem rational_absolute_value_equation (a : ℚ) : 
  |a - 1| = 4 → (a = 5 ∨ a = -3) := by
  sorry

end rational_absolute_value_equation_l893_89328


namespace complex_calculation_l893_89381

theorem complex_calculation (p q : ℂ) (hp : p = 3 + 2*I) (hq : q = 2 - 3*I) :
  3*p + 4*q = 17 - 6*I := by
  sorry

end complex_calculation_l893_89381


namespace special_right_triangle_sides_l893_89321

/-- A right triangle with a special inscribed circle -/
structure SpecialRightTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first leg of the triangle -/
  x : ℝ
  /-- The second leg of the triangle -/
  y : ℝ
  /-- The hypotenuse of the triangle -/
  z : ℝ
  /-- The area of the triangle is 2r^2/3 -/
  area_eq : x * y / 2 = 2 * r^2 / 3
  /-- The triangle is right-angled -/
  pythagoras : x^2 + y^2 = z^2
  /-- The circle touches one leg, the extension of the other leg, and the hypotenuse -/
  circle_property : z = 2*r + x - y

/-- The sides of a special right triangle are r, 4r/3, and 5r/3 -/
theorem special_right_triangle_sides (t : SpecialRightTriangle) : 
  t.x = t.r ∧ t.y = 4 * t.r / 3 ∧ t.z = 5 * t.r / 3 := by
  sorry

end special_right_triangle_sides_l893_89321


namespace adams_book_purchase_l893_89362

/-- Represents a bookcase with a given number of shelves and average books per shelf. -/
structure Bookcase where
  shelves : ℕ
  avgBooksPerShelf : ℕ

/-- Calculates the total capacity of a bookcase. -/
def Bookcase.capacity (b : Bookcase) : ℕ := b.shelves * b.avgBooksPerShelf

theorem adams_book_purchase (
  adam_bookcase : Bookcase
  ) (adam_bookcase_shelves : adam_bookcase.shelves = 4)
    (adam_bookcase_avg : adam_bookcase.avgBooksPerShelf = 20)
    (initial_books : ℕ) (initial_books_count : initial_books = 56)
    (books_left_over : ℕ) (books_left_over_count : books_left_over = 2) :
  adam_bookcase.capacity + books_left_over - initial_books = 26 := by
  sorry

end adams_book_purchase_l893_89362


namespace probability_three_divisible_by_3_l893_89339

/-- The probability of a single 12-sided die showing a number divisible by 3 -/
def p_divisible_by_3 : ℚ := 1 / 3

/-- The probability of a single 12-sided die not showing a number divisible by 3 -/
def p_not_divisible_by_3 : ℚ := 2 / 3

/-- The number of dice rolled -/
def total_dice : ℕ := 7

/-- The number of dice that should show a number divisible by 3 -/
def target_dice : ℕ := 3

/-- The theorem stating the probability of exactly three out of seven fair 12-sided dice 
    showing a number divisible by 3 -/
theorem probability_three_divisible_by_3 : 
  (Nat.choose total_dice target_dice : ℚ) * 
  p_divisible_by_3 ^ target_dice * 
  p_not_divisible_by_3 ^ (total_dice - target_dice) = 560 / 2187 := by
  sorry

end probability_three_divisible_by_3_l893_89339


namespace alcohol_percentage_in_second_vessel_l893_89365

theorem alcohol_percentage_in_second_vessel
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 5)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_percentage = 27.5)
  : ∃ (vessel2_alcohol_percentage : ℝ),
    vessel2_alcohol_percentage = 40 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) +
    vessel2_capacity * (vessel2_alcohol_percentage / 100) =
    final_vessel_capacity * (final_mixture_percentage / 100) :=
by sorry

end alcohol_percentage_in_second_vessel_l893_89365


namespace square_side_length_l893_89338

theorem square_side_length (diagonal : ℝ) (h : diagonal = 4) : 
  ∃ side : ℝ, side = 2 * Real.sqrt 2 ∧ side ^ 2 + side ^ 2 = diagonal ^ 2 :=
by sorry

end square_side_length_l893_89338


namespace smallest_m_correct_l893_89366

/-- The smallest positive value of m for which the equation 10x^2 - mx + 600 = 0 has consecutive integer solutions -/
def smallest_m : ℕ := 170

/-- Predicate to check if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1 ∨ a = b + 1

theorem smallest_m_correct :
  ∀ m : ℕ,
  (∃ x y : ℤ, consecutive x y ∧ 10 * x^2 - m * x + 600 = 0 ∧ 10 * y^2 - m * y + 600 = 0) →
  m ≥ smallest_m :=
by sorry

end smallest_m_correct_l893_89366


namespace lcm_sum_inequality_l893_89364

theorem lcm_sum_inequality (a b c d e : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (1 : ℚ) / Nat.lcm a b + 1 / Nat.lcm b c + 1 / Nat.lcm c d + 1 / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end lcm_sum_inequality_l893_89364


namespace lucky_number_in_13_consecutive_l893_89391

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def isLucky (n : ℕ) : Prop := sumOfDigits n % 7 = 0

/-- Any sequence of 13 consecutive numbers contains at least one lucky number -/
theorem lucky_number_in_13_consecutive (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 12 ∧ isLucky k := by sorry

end lucky_number_in_13_consecutive_l893_89391


namespace triangle_square_perimeter_l893_89345

theorem triangle_square_perimeter (d : ℕ) : 
  let triangle_side := s + d
  let square_side := s
  (∃ s : ℚ, s > 0 ∧ 3 * triangle_side - 4 * square_side = 1989) →
  d > 663 :=
by
  sorry

end triangle_square_perimeter_l893_89345


namespace specific_parallelogram_area_and_height_l893_89315

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  angle : ℝ  -- One angle of the parallelogram in degrees
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of the adjacent side
  extension : ℝ  -- Length of extension beyond the vertex

/-- Calculates the area and height of a parallelogram with specific properties -/
def parallelogram_area_and_height (p : Parallelogram) : ℝ × ℝ :=
  sorry

/-- Theorem stating the area and height of a specific parallelogram -/
theorem specific_parallelogram_area_and_height :
  let p : Parallelogram := ⟨150, 10, 18, 2⟩
  let (area, height) := parallelogram_area_and_height p
  area = 36 * Real.sqrt 3 ∧ height = 2 * Real.sqrt 3 := by
  sorry

end specific_parallelogram_area_and_height_l893_89315


namespace intersection_of_A_and_B_l893_89369

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end intersection_of_A_and_B_l893_89369


namespace expansion_coefficient_l893_89348

/-- The coefficient of x^3y^7 in the expansion of (2/3x - 3/4y)^10 -/
def coefficient_x3y7 : ℚ :=
  let a : ℚ := 2/3
  let b : ℚ := -3/4
  let n : ℕ := 10
  let k : ℕ := 7
  (n.choose k) * a^(n-k) * b^k

theorem expansion_coefficient :
  coefficient_x3y7 = -4374/921 := by
  sorry

end expansion_coefficient_l893_89348


namespace exact_time_l893_89371

/-- Represents the time in minutes after 4:00 --/
def t : ℝ := by sorry

/-- The angle of the minute hand at time t --/
def minute_hand (t : ℝ) : ℝ := 6 * t

/-- The angle of the hour hand at time t --/
def hour_hand (t : ℝ) : ℝ := 120 + 0.5 * t

/-- The condition that the time is between 4:00 and 5:00 --/
axiom time_range : 0 ≤ t ∧ t < 60

/-- The condition that the minute hand is opposite to where the hour hand was 5 minutes ago --/
axiom opposite_hands : 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 180 ∨ 
  |minute_hand (t + 10) - hour_hand (t - 5)| = 540

theorem exact_time : t = 25 := by sorry

end exact_time_l893_89371


namespace campground_distance_l893_89342

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end campground_distance_l893_89342


namespace tan_sum_squared_l893_89301

theorem tan_sum_squared (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) + Real.tan (b / 2))^2 = 6 ∨ (Real.tan (a / 2) + Real.tan (b / 2))^2 = 26 := by
  sorry

end tan_sum_squared_l893_89301


namespace waste_processing_growth_equation_l893_89393

/-- Represents the growth of processing capacity over two months -/
def processing_capacity_growth (initial_capacity : ℝ) (final_capacity : ℝ) (growth_rate : ℝ) : Prop :=
  initial_capacity * (1 + growth_rate)^2 = final_capacity

/-- The equation correctly models the company's waste processing capacity growth -/
theorem waste_processing_growth_equation :
  processing_capacity_growth 1000 1200 x ↔ 1000 * (1 + x)^2 = 1200 :=
by
  sorry

end waste_processing_growth_equation_l893_89393


namespace max_profit_l893_89303

-- Define the types of products
inductive Product
| A
| B

-- Define the profit function
def profit (x y : ℕ) : ℕ := 300 * x + 400 * y

-- Define the material constraints
def material_constraint (x y : ℕ) : Prop :=
  x + 2 * y ≤ 12 ∧ 2 * x + y ≤ 12

-- State the theorem
theorem max_profit :
  ∃ x y : ℕ,
    material_constraint x y ∧
    profit x y = 2800 ∧
    ∀ a b : ℕ, material_constraint a b → profit a b ≤ 2800 :=
sorry

end max_profit_l893_89303


namespace product_equals_nine_l893_89320

theorem product_equals_nine : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * 
  (1 + 1/5) * (1 + 1/6) * (1 + 1/7) * (1 + 1/8) = 9 := by
  sorry

end product_equals_nine_l893_89320


namespace root_implies_c_value_l893_89379

theorem root_implies_c_value (b c : ℝ) :
  (∃ (x : ℂ), x^2 + b*x + c = 0 ∧ x = 1 - Complex.I * Real.sqrt 2) →
  c = 3 := by
  sorry

end root_implies_c_value_l893_89379


namespace intersection_of_lines_l893_89340

/-- Given four points in 3D space, this theorem proves that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (8, -5, 5) →
  B = (18, -15, 10) →
  C = (1, 5, -7) →
  D = (3, -3, 13) →
  ∃ t s : ℝ, 
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (1 + 2*s, 5 - 8*s, -7 + 20*s) ∧
    (8 + 10*t, -5 - 10*t, 5 + 5*t) = (-16, 7, -7) := by
  sorry

end intersection_of_lines_l893_89340


namespace pascal_cycling_trip_l893_89356

theorem pascal_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_increase : ℝ) 
  (h1 : current_speed = 8)
  (h2 : speed_reduction = 4)
  (h3 : time_increase = 16)
  (h4 : current_speed * (time_increase + t) = (current_speed - speed_reduction) * (time_increase + t + time_increase))
  (h5 : current_speed * t = (current_speed + current_speed / 2) * (time_increase + t - time_increase)) :
  current_speed * t = 256 := by
  sorry

end pascal_cycling_trip_l893_89356


namespace paul_clothing_expense_l893_89332

def shirt_price : ℝ := 15
def pants_price : ℝ := 40
def suit_price : ℝ := 150
def sweater_price : ℝ := 30

def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.2
def coupon_discount : ℝ := 0.1

def total_before_discount : ℝ := 
  shirt_price * num_shirts + 
  pants_price * num_pants + 
  suit_price * num_suits + 
  sweater_price * num_sweaters

def total_after_store_discount : ℝ :=
  total_before_discount * (1 - store_discount)

def final_total : ℝ :=
  total_after_store_discount * (1 - coupon_discount)

theorem paul_clothing_expense : final_total = 252 :=
sorry

end paul_clothing_expense_l893_89332


namespace proton_origin_probability_proton_max_probability_at_six_l893_89395

/-- Represents the probability of a proton being at a specific position after n moves --/
def ProtonProbability (n : ℕ) (position : ℤ) : ℚ :=
  sorry

/-- The probability of the proton being at the origin after 4 moves --/
theorem proton_origin_probability : ProtonProbability 4 0 = 3/8 :=
  sorry

/-- The number of moves that maximizes the probability of the proton being at position 6 --/
def MaxProbabilityMoves : Finset ℕ :=
  sorry

/-- The probability of the proton being at position 6 is maximized when the number of moves is either 34 or 36 --/
theorem proton_max_probability_at_six :
  MaxProbabilityMoves = {34, 36} :=
  sorry

end proton_origin_probability_proton_max_probability_at_six_l893_89395


namespace coefficient_of_c_l893_89385

theorem coefficient_of_c (A : ℝ) (c d : ℝ) : 
  (∀ c', c' ≤ 47) → 
  (A * 47 + (d - 12)^2 = 235) → 
  A = 5 := by
sorry

end coefficient_of_c_l893_89385


namespace sets_equality_implies_sum_l893_89334

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {0, |x|, y}
def B (x y : ℝ) : Set ℝ := {x, x*y, Real.sqrt (x-y)}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) : A x y = B x y → x + y = -2 := by
  sorry

end sets_equality_implies_sum_l893_89334


namespace original_manufacturing_cost_l893_89306

/-- 
Given a fixed selling price and information about profit changes,
prove that the original manufacturing cost was $70.
-/
theorem original_manufacturing_cost
  (P : ℝ) -- Selling price
  (h1 : P - P * 0.5 = 50) -- New manufacturing cost is $50
  : P * 0.7 = 70 := by
  sorry

end original_manufacturing_cost_l893_89306


namespace inequality_solution_l893_89324

theorem inequality_solution : ∃! x : ℝ, 
  (Real.sqrt (x^3 + x - 90) + 7) * |x^3 - 10*x^2 + 31*x - 28| ≤ 0 ∧
  x = 3 + Real.sqrt 2 := by
  sorry

end inequality_solution_l893_89324


namespace largest_number_l893_89333

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def a : Nat := base_to_decimal [5, 8] 9
def b : Nat := base_to_decimal [1, 0, 3] 5
def c : Nat := base_to_decimal [1, 0, 0, 1] 2

theorem largest_number : a > b ∧ a > c := by
  sorry

end largest_number_l893_89333


namespace quadratic_sequence_l893_89329

theorem quadratic_sequence (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end quadratic_sequence_l893_89329


namespace problem_solution_l893_89352

def f (k : ℝ) (x : ℝ) : ℝ := |3*x - 1| + |3*x + k|
def g (x : ℝ) : ℝ := x + 4

theorem problem_solution :
  (∀ x : ℝ, f (-3) x ≥ 4 ↔ (x ≤ 0 ∨ x ≥ 4/3)) ∧
  (∀ k : ℝ, k > -1 → 
    (∀ x : ℝ, x ∈ Set.Icc (-k/3) (1/3) → f k x ≤ g x) →
    k ∈ Set.Ioo (-1) (9/4)) :=
sorry

end problem_solution_l893_89352


namespace shenzhen_revenue_precision_l893_89327

/-- Represents a large monetary amount in yuan -/
structure LargeAmount where
  value : ℝ
  unit : String

/-- Defines the precision of a number -/
inductive Precision
  | HundredBillion
  | TenBillion
  | Billion
  | HundredMillion
  | TenMillion
  | Million

/-- Returns the precision of a given LargeAmount -/
def getPrecision (amount : LargeAmount) : Precision :=
  sorry

theorem shenzhen_revenue_precision :
  let revenue : LargeAmount := { value := 21.658, unit := "billion yuan" }
  getPrecision revenue = Precision.HundredMillion := by sorry

end shenzhen_revenue_precision_l893_89327


namespace intersection_A_B_l893_89398

-- Define the sets A and B
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioc 1 2 := by sorry

end intersection_A_B_l893_89398


namespace large_data_logarithm_l893_89382

theorem large_data_logarithm (m : ℝ) (n : ℕ+) :
  (1 < m) ∧ (m < 10) ∧
  (0.4771 < Real.log 3 / Real.log 10) ∧ (Real.log 3 / Real.log 10 < 0.4772) ∧
  (3 ^ 2000 : ℝ) = m * 10 ^ (n : ℝ) →
  n = 954 := by
  sorry

end large_data_logarithm_l893_89382


namespace sally_picked_seven_lemons_l893_89336

/-- The number of lemons Mary picked -/
def mary_lemons : ℕ := 9

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons Sally picked -/
def sally_lemons : ℕ := total_lemons - mary_lemons

theorem sally_picked_seven_lemons : sally_lemons = 7 := by
  sorry

end sally_picked_seven_lemons_l893_89336


namespace mans_rowing_speed_l893_89325

/-- The rowing speed of a man in still water, given his speeds with and against a stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16)
  (h2 : speed_against_stream = 4) : 
  (speed_with_stream + speed_against_stream) / 2 = 10 := by
  sorry

#check mans_rowing_speed

end mans_rowing_speed_l893_89325


namespace largest_average_l893_89316

def multiples_average (m n : ℕ) : ℚ :=
  (m + n * (n.div m) * m) / (2 * n.div m)

theorem largest_average : 
  let avg3 := multiples_average 3 101
  let avg4 := multiples_average 4 102
  let avg5 := multiples_average 5 100
  let avg7 := multiples_average 7 101
  avg5 = 52.5 ∧ avg7 = 52.5 ∧ avg5 > avg3 ∧ avg5 > avg4 :=
by sorry

end largest_average_l893_89316


namespace special_triangle_sum_l893_89319

/-- A triangle with an incircle that evenly trisects a median -/
structure SpecialTriangle where
  -- The side length BC
  a : ℝ
  -- The area of the triangle
  area : ℝ
  -- k and p, where area = k√p
  k : ℕ
  p : ℕ
  -- Conditions
  side_length : a = 24
  area_form : area = k * Real.sqrt p
  p_not_square_divisible : ∀ (q : ℕ), Prime q → ¬(q^2 ∣ p)
  incircle_trisects_median : True  -- This condition is implicit in the structure

/-- The sum of k and p for the special triangle is 51 -/
theorem special_triangle_sum (t : SpecialTriangle) : t.k + t.p = 51 := by
  sorry

end special_triangle_sum_l893_89319


namespace golden_raisin_cost_l893_89360

/-- The cost per scoop of natural seedless raisins -/
def natural_cost : ℝ := 3.45

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The cost per scoop of the mixture -/
def mixture_cost : ℝ := 3

/-- The cost per scoop of golden seedless raisins -/
def golden_cost : ℝ := 2.55

theorem golden_raisin_cost :
  (natural_cost * natural_scoops + golden_cost * golden_scoops) / (natural_scoops + golden_scoops) = mixture_cost :=
sorry

end golden_raisin_cost_l893_89360


namespace coffee_consumption_ratio_l893_89323

/-- Given that Brayan drinks 4 cups of coffee per hour and they drink a total of 30 cups of coffee
    together in 5 hours, prove that the ratio of the amount of coffee Brayan drinks to the amount
    Ivory drinks is 2:1. -/
theorem coffee_consumption_ratio :
  let brayan_per_hour : ℚ := 4
  let total_cups : ℚ := 30
  let total_hours : ℚ := 5
  let ivory_per_hour : ℚ := total_cups / total_hours - brayan_per_hour
  brayan_per_hour / ivory_per_hour = 2 := by
  sorry

end coffee_consumption_ratio_l893_89323


namespace expand_and_simplify_l893_89308

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (x - 2*y) - y * (3 - 4*y) = x^2 - 3*y := by
  sorry

end expand_and_simplify_l893_89308


namespace complex_modulus_problem_l893_89312

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end complex_modulus_problem_l893_89312


namespace richmond_tigers_ticket_sales_l893_89359

theorem richmond_tigers_ticket_sales (first_half_sales second_half_sales : ℕ) 
  (h1 : first_half_sales = 3867)
  (h2 : second_half_sales = 5703) :
  first_half_sales + second_half_sales = 9570 := by
  sorry

end richmond_tigers_ticket_sales_l893_89359


namespace hotel_guests_count_l893_89389

theorem hotel_guests_count (oates_count hall_count both_count : ℕ) 
  (ho : oates_count = 40)
  (hh : hall_count = 70)
  (hb : both_count = 10) :
  oates_count + hall_count - both_count = 100 := by
  sorry

end hotel_guests_count_l893_89389


namespace shifted_line_equation_and_intercept_l893_89373

/-- A line obtained by shifting a direct proportion function -/
structure ShiftedLine where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_one_two : k * 1 + b = 2 + 5
  shifted_up_five : b = 5

theorem shifted_line_equation_and_intercept (l : ShiftedLine) :
  (l.k = 2 ∧ l.b = 5) ∧ 
  (∃ (x : ℝ), x = -2.5 ∧ l.k * x + l.b = 0) := by
  sorry

end shifted_line_equation_and_intercept_l893_89373


namespace classroom_area_less_than_hectare_l893_89396

-- Define the area of 1 hectare in square meters
def hectare_area : ℝ := 10000

-- Define the typical area of a classroom in square meters
def typical_classroom_area : ℝ := 60

-- Theorem stating that a typical classroom area is much less than a hectare
theorem classroom_area_less_than_hectare :
  typical_classroom_area < hectare_area ∧ typical_classroom_area / hectare_area < 0.01 :=
by sorry

end classroom_area_less_than_hectare_l893_89396


namespace winter_clothing_count_l893_89349

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) :
  num_boxes = 6 →
  scarves_per_box = 5 →
  mittens_per_box = 5 →
  num_boxes * (scarves_per_box + mittens_per_box) = 60 :=
by
  sorry

end winter_clothing_count_l893_89349


namespace performance_arrangements_l893_89300

def original_programs : ℕ := 6
def added_programs : ℕ := 3
def available_spaces : ℕ := original_programs - 1

theorem performance_arrangements : 
  (Nat.descFactorial available_spaces added_programs) + 
  (Nat.descFactorial 3 2 * Nat.descFactorial available_spaces 2) + 
  (5 * Nat.descFactorial 3 3) = 210 := by sorry

end performance_arrangements_l893_89300


namespace trajectory_of_B_l893_89337

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the ellipse equation
def satisfiesEllipseEquation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Main theorem
theorem trajectory_of_B (ABC : Triangle) :
  ABC.A = (-1, 0) →
  ABC.C = (1, 0) →
  isArithmeticSequence (dist ABC.B ABC.C) (dist ABC.C ABC.A) (dist ABC.A ABC.B) →
  ∀ x y, x ≠ 2 ∧ x ≠ -2 →
  ABC.B = (x, y) →
  satisfiesEllipseEquation x y :=
sorry

end trajectory_of_B_l893_89337


namespace quadratic_roots_problem_l893_89397

theorem quadratic_roots_problem (x₁ x₂ k : ℝ) : 
  (x₁^2 - 3*x₁ + k = 0) →
  (x₂^2 - 3*x₂ + k = 0) →
  (x₁ = 2*x₂) →
  k = 2 := by
sorry

end quadratic_roots_problem_l893_89397


namespace angle_through_point_l893_89361

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) →
  α = 5 * Real.pi / 3 := by
sorry

end angle_through_point_l893_89361


namespace gcd_168_54_264_l893_89399

theorem gcd_168_54_264 : Nat.gcd 168 (Nat.gcd 54 264) = 6 := by
  sorry

end gcd_168_54_264_l893_89399


namespace quadratic_monotonicity_l893_89351

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity (m : ℝ) :
  let f := fun (x : ℝ) ↦ -2 * x^2 + m * x + 1
  IsMonotonic f (-1) 4 ↔ m ∈ Set.Iic (-4) ∪ Set.Ici 16 := by
  sorry

end quadratic_monotonicity_l893_89351


namespace geometric_series_problem_l893_89341

/-- Given two infinite geometric series with the specified properties, prove that n = 195 --/
theorem geometric_series_problem (n : ℝ) : 
  let first_series_a1 : ℝ := 15
  let first_series_a2 : ℝ := 5
  let second_series_a1 : ℝ := 15
  let second_series_a2 : ℝ := 5 + n
  let first_series_sum := first_series_a1 / (1 - (first_series_a2 / first_series_a1))
  let second_series_sum := second_series_a1 / (1 - (second_series_a2 / second_series_a1))
  second_series_sum = 5 * first_series_sum →
  n = 195 := by
sorry

end geometric_series_problem_l893_89341


namespace square_perimeter_from_rectangle_perimeter_l893_89358

/-- Given a square divided into six congruent rectangles, if each rectangle has a perimeter of 30 inches, then the perimeter of the square is 360/7 inches. -/
theorem square_perimeter_from_rectangle_perimeter (s : ℝ) : 
  s > 0 → 
  (2 * s + 2 * (s / 6) = 30) → 
  (4 * s = 360 / 7) := by
  sorry

#check square_perimeter_from_rectangle_perimeter

end square_perimeter_from_rectangle_perimeter_l893_89358


namespace inscribed_cube_surface_area_l893_89392

theorem inscribed_cube_surface_area :
  let outer_cube_edge : ℝ := 12
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_surface_area : ℝ := 6 * inner_cube_edge ^ 2
  inner_cube_surface_area = 288 := by sorry

end inscribed_cube_surface_area_l893_89392


namespace ellipse_symmetric_points_range_l893_89394

/-- Given an ellipse and two symmetric points on it, prove the range of m -/
theorem ellipse_symmetric_points_range (x₁ y₁ x₂ y₂ m : ℝ) : 
  (x₁^2 / 4 + y₁^2 / 3 = 1) →  -- Point A on ellipse
  (x₂^2 / 4 + y₂^2 / 3 = 1) →  -- Point B on ellipse
  ((y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m) →  -- A and B symmetric about y = 4x + m
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- A and B are distinct
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end ellipse_symmetric_points_range_l893_89394


namespace total_tax_percentage_calculation_l893_89370

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingSpendPercentage : ℝ) (foodSpendPercentage : ℝ) 
  (electronicsSpendPercentage : ℝ) (otherSpendPercentage : ℝ)
  (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (electronicsTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingSpendPercentage * clothingTaxRate + 
  foodSpendPercentage * foodTaxRate + 
  electronicsSpendPercentage * electronicsTaxRate + 
  otherSpendPercentage * otherTaxRate

theorem total_tax_percentage_calculation :
  totalTaxPercentage 0.585 0.12 0.225 0.07 0.052 0 0.073 0.095 = 0.053495 := by
  sorry

end total_tax_percentage_calculation_l893_89370


namespace prop_p_or_q_l893_89343

theorem prop_p_or_q : 
  (∀ x : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x : ℝ, Real.sin x + Real.cos x = 2) :=
sorry

end prop_p_or_q_l893_89343


namespace intersection_of_A_and_B_l893_89363

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 10}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l893_89363


namespace sample_size_is_70_l893_89307

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes for products A, B, and C -/
structure SampleSize where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the sample sizes for each product -/
def totalSampleSize (s : SampleSize) : ℕ := s.a + s.b + s.c

/-- Theorem stating that given the product ratio and the sample size of product A, 
    the total sample size is 70 -/
theorem sample_size_is_70 (ratio : ProductRatio) (sample : SampleSize) :
  ratio = ⟨3, 4, 7⟩ → sample.a = 15 → totalSampleSize sample = 70 := by
  sorry

#check sample_size_is_70

end sample_size_is_70_l893_89307


namespace vector_subtraction_l893_89350

/-- Given two plane vectors a and b, prove that a - 2b equals (3, 7) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (5, 3)) (hb : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end vector_subtraction_l893_89350


namespace max_cows_is_correct_l893_89335

/-- Represents the maximum number of cows a rancher can buy given specific constraints. -/
def max_cows : ℕ :=
  let budget : ℕ := 1300
  let steer_cost : ℕ := 30
  let cow_cost : ℕ := 33
  30

/-- Theorem stating that max_cows is indeed the maximum number of cows the rancher can buy. -/
theorem max_cows_is_correct :
  ∀ s c : ℕ,
  s > 0 →
  c > 0 →
  c > 2 * s →
  s * 30 + c * 33 ≤ 1300 →
  c ≤ max_cows :=
by sorry

#eval max_cows  -- Should output 30

end max_cows_is_correct_l893_89335


namespace fraction_simplification_l893_89353

theorem fraction_simplification : (2 : ℚ) / (1 - 2 / 3) = 6 := by sorry

end fraction_simplification_l893_89353


namespace triangle_solution_l893_89372

/-- Given a triangle with sides a, b, c, angle γ, and circumscribed circle diameter d,
    if a² - b² = 19, γ = 126°52'12", and d = 21.25,
    then a ≈ 10, b ≈ 9, and c ≈ 17 -/
theorem triangle_solution (a b c : ℝ) (γ : Real) (d : ℝ) : 
  a^2 - b^2 = 19 →
  γ = 126 * π / 180 + 52 * π / (180 * 60) + 12 * π / (180 * 60 * 60) →
  d = 21.25 →
  (abs (a - 10) < 0.5 ∧ abs (b - 9) < 0.5 ∧ abs (c - 17) < 0.5) :=
by sorry

end triangle_solution_l893_89372


namespace system_solution_square_difference_l893_89374

theorem system_solution_square_difference (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 1) 
  (eq2 : x + y = 2) : 
  x^2 - 2 * y^2 = -1 := by
sorry

end system_solution_square_difference_l893_89374


namespace no_goal_scored_l893_89347

def football_play (play1 play2 play3 play4 : ℝ) : Prop :=
  play1 = -5 ∧ 
  play2 = 13 ∧ 
  play3 = -(play1^2) ∧ 
  play4 = -play3 / 2

def total_progress (play1 play2 play3 play4 : ℝ) : ℝ :=
  play1 + play2 + play3 + play4

def score_goal (progress : ℝ) : Prop :=
  progress ≥ 30

theorem no_goal_scored (play1 play2 play3 play4 : ℝ) :
  football_play play1 play2 play3 play4 →
  ¬(score_goal (total_progress play1 play2 play3 play4)) :=
by sorry

end no_goal_scored_l893_89347


namespace tangent_line_at_point_one_two_l893_89380

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 1 :=
by sorry

end tangent_line_at_point_one_two_l893_89380
