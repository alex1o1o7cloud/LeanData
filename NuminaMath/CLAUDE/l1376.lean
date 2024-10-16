import Mathlib

namespace NUMINAMATH_CALUDE_parallel_line_length_l1376_137630

theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 18 → 
  (parallel_line / base)^2 = 1/2 → 
  parallel_line = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l1376_137630


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1376_137656

theorem quadratic_unique_solution :
  ∃! (k x : ℚ), 5 * k * x^2 + 30 * x + 10 = 0 ∧ k = 9/2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1376_137656


namespace NUMINAMATH_CALUDE_squares_in_6x4_rectangle_l1376_137664

/-- The number of unit squares that can fit in a rectangle -/
def squaresInRectangle (length width : ℕ) : ℕ := length * width

/-- Theorem: A 6x4 rectangle can fit 24 unit squares -/
theorem squares_in_6x4_rectangle :
  squaresInRectangle 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_6x4_rectangle_l1376_137664


namespace NUMINAMATH_CALUDE_arrangements_with_pair_together_eq_48_l1376_137689

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange five people in a row, with two specific people always standing together -/
def arrangements_with_pair_together : ℕ :=
  factorial 4 * factorial 2

theorem arrangements_with_pair_together_eq_48 :
  arrangements_with_pair_together = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_pair_together_eq_48_l1376_137689


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1376_137675

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l1376_137675


namespace NUMINAMATH_CALUDE_max_books_with_200_dollars_l1376_137632

/-- The maximum number of books that can be purchased with a given budget and book price -/
def maxBooks (budget : ℕ) (bookPrice : ℕ) : ℕ :=
  (budget * 100) / bookPrice

/-- Theorem: Given a book price of $45 and a budget of $200, the maximum number of books that can be purchased is 444 -/
theorem max_books_with_200_dollars : maxBooks 200 45 = 444 := by
  sorry

end NUMINAMATH_CALUDE_max_books_with_200_dollars_l1376_137632


namespace NUMINAMATH_CALUDE_crayon_production_l1376_137620

theorem crayon_production (num_colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) :
  num_colors = 4 →
  crayons_per_color = 2 →
  boxes_per_hour = 5 →
  hours = 4 →
  (num_colors * crayons_per_color * boxes_per_hour * hours) = 160 :=
by sorry

end NUMINAMATH_CALUDE_crayon_production_l1376_137620


namespace NUMINAMATH_CALUDE_chord_length_l1376_137695

/-- The length of the chord cut by the line y = 3x on the circle (x+1)^2 + (y-2)^2 = 25 is 3√10. -/
theorem chord_length (x y : ℝ) : 
  y = 3 * x →
  (x + 1)^2 + (y - 2)^2 = 25 →
  ∃ (x1 y1 x2 y2 : ℝ), 
    y1 = 3 * x1 ∧
    (x1 + 1)^2 + (y1 - 2)^2 = 25 ∧
    y2 = 3 * x2 ∧
    (x2 + 1)^2 + (y2 - 2)^2 = 25 ∧
    ((x2 - x1)^2 + (y2 - y1)^2) = 90 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1376_137695


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1376_137601

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 22) → C = 11 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l1376_137601


namespace NUMINAMATH_CALUDE_membership_condition_l1376_137608

def is_necessary_but_not_sufficient {α : Type*} (A B : Set α) : Prop :=
  (A ∩ B = B) ∧ (A ≠ B) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B)

theorem membership_condition {α : Type*} (A B : Set α) 
  (h1 : A ∩ B = B) (h2 : A ≠ B) :
  is_necessary_but_not_sufficient A B :=
sorry

end NUMINAMATH_CALUDE_membership_condition_l1376_137608


namespace NUMINAMATH_CALUDE_a_14_mod_7_l1376_137668

/-- Sequence defined recursively -/
def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1  -- We assume a₁ = 1 based on the solution
  | 2 => 2
  | (n + 3) => a (n + 1) + (a (n + 2))^2

/-- The 14th term of the sequence is congruent to 5 modulo 7 -/
theorem a_14_mod_7 : a 14 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_a_14_mod_7_l1376_137668


namespace NUMINAMATH_CALUDE_total_people_in_program_l1376_137641

theorem total_people_in_program (parents pupils : ℕ) 
  (h1 : parents = 105) 
  (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l1376_137641


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1376_137642

theorem infinite_series_sum : 
  (∑' n : ℕ, (n : ℝ) / (5 ^ n)) = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1376_137642


namespace NUMINAMATH_CALUDE_neighbor_oranges_correct_l1376_137653

/-- The number of kilograms of oranges added for the neighbor -/
def neighbor_oranges : ℕ := 25

/-- The initial purchase of oranges in kilograms -/
def initial_purchase : ℕ := 10

/-- The total quantity of oranges bought over three weeks in kilograms -/
def total_quantity : ℕ := 75

/-- The quantity of oranges bought in each of the next two weeks -/
def next_weeks_purchase : ℕ := 2 * initial_purchase

theorem neighbor_oranges_correct :
  (initial_purchase + neighbor_oranges) + next_weeks_purchase + next_weeks_purchase = total_quantity :=
by sorry

end NUMINAMATH_CALUDE_neighbor_oranges_correct_l1376_137653


namespace NUMINAMATH_CALUDE_teal_greenish_count_teal_greenish_proof_l1376_137645

def total_surveyed : ℕ := 120
def kinda_blue : ℕ := 70
def both : ℕ := 35
def neither : ℕ := 20

theorem teal_greenish_count : ℕ :=
  total_surveyed - (kinda_blue - both) - both - neither
  
theorem teal_greenish_proof : teal_greenish_count = 65 := by
  sorry

end NUMINAMATH_CALUDE_teal_greenish_count_teal_greenish_proof_l1376_137645


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l1376_137613

/-- The equation of a circle given two points on its diameter -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (0, 3) →
  B = (-4, 0) →
  ∀ (x y : ℝ),
    (x - (-2))^2 + (y - (3/2))^2 = (5/2)^2 ↔ x^2 + y^2 + 4*x - 3*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l1376_137613


namespace NUMINAMATH_CALUDE_climb_down_distance_is_6_l1376_137685

-- Define the climb up speed
def climb_up_speed : ℝ := 2

-- Define the climb down speed
def climb_down_speed : ℝ := 3

-- Define the total time
def total_time : ℝ := 4

-- Define the additional distance on the way down
def additional_distance : ℝ := 2

-- Theorem statement
theorem climb_down_distance_is_6 :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x / climb_up_speed + (x + additional_distance) / climb_down_speed = total_time ∧
    x + additional_distance = 6 :=
sorry

end NUMINAMATH_CALUDE_climb_down_distance_is_6_l1376_137685


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1376_137605

def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5*x + 6 > 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1376_137605


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1376_137600

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 / y^2) + (y^2 / z^2) + (z^2 / x^2) ≥ (x / y) + (y / z) + (z / x) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1376_137600


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1376_137696

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 1) * (x - 2) < 0 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1376_137696


namespace NUMINAMATH_CALUDE_train_length_calculation_l1376_137650

/-- Calculates the length of a train given the speeds of two trains, time to cross, and length of the other train -/
theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (l2 : ℝ) (h1 : v1 = 120) (h2 : v2 = 80) (h3 : t = 9) (h4 : l2 = 410.04) :
  let relative_speed := (v1 + v2) * 1000 / 3600
  let total_length := relative_speed * t
  let l1 := total_length - l2
  l1 = 90 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1376_137650


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1376_137677

theorem pythagorean_theorem_3_4_5 :
  let a : ℝ := 30
  let b : ℝ := 40
  let c : ℝ := 50
  a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l1376_137677


namespace NUMINAMATH_CALUDE_point_inside_iff_odd_intersections_l1376_137661

/-- A closed, non-self-intersecting path in a plane. -/
structure ClosedPath :=
  (path : Set (ℝ × ℝ))
  (closed : IsClosed path)
  (non_self_intersecting : ∀ x y : ℝ × ℝ, x ∈ path → y ∈ path → x ≠ y → (∃ t : ℝ, 0 < t ∧ t < 1 ∧ (1 - t) • x + t • y ∉ path))

/-- A point in the plane. -/
def Point := ℝ × ℝ

/-- The number of intersections between a line segment and a path. -/
def intersectionCount (p q : Point) (path : ClosedPath) : ℕ :=
  sorry

/-- A point is known to be outside the region bounded by the path. -/
def isOutside (p : Point) (path : ClosedPath) : Prop :=
  sorry

/-- A point is inside the region bounded by the path. -/
def isInside (p : Point) (path : ClosedPath) : Prop :=
  ∀ q : Point, isOutside q path → Odd (intersectionCount p q path)

theorem point_inside_iff_odd_intersections (p : Point) (path : ClosedPath) :
  isInside p path ↔ ∀ q : Point, isOutside q path → Odd (intersectionCount p q path) :=
sorry

end NUMINAMATH_CALUDE_point_inside_iff_odd_intersections_l1376_137661


namespace NUMINAMATH_CALUDE_ellipse_equation_l1376_137663

theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : c / a = Real.sqrt 3 / 2)
  (h5 : a - c = 2 - Real.sqrt 3)
  (h6 : b^2 = a^2 - c^2) :
  ∃ (x y : ℝ), y^2 / 4 + x^2 = 1 ∧ y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1376_137663


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1376_137606

theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  s * Real.sqrt 3 = 10 * Real.sqrt 3 →
  s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1376_137606


namespace NUMINAMATH_CALUDE_cookie_distribution_l1376_137669

theorem cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 120)
  (h2 : num_adults = 2)
  (h3 : num_children = 4) :
  (total_cookies - (total_cookies / 3)) / num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1376_137669


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1376_137624

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1376_137624


namespace NUMINAMATH_CALUDE_subtracted_number_l1376_137621

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem subtracted_number (x : Nat) : 
  (sum_of_digits (10^38 - x) = 330) → 
  (x = 10^37 + 3 * 10^36) :=
by sorry

end NUMINAMATH_CALUDE_subtracted_number_l1376_137621


namespace NUMINAMATH_CALUDE_four_color_theorem_l1376_137655

/-- Represents a country on the map -/
structure Country where
  borders : ℕ
  border_divisible_by_three : borders % 3 = 0

/-- Represents a map of countries -/
structure Map where
  countries : List Country

/-- Represents a coloring of the map -/
def Coloring := Map → Country → Fin 4

/-- A coloring is proper if no adjacent countries have the same color -/
def is_proper_coloring (m : Map) (c : Coloring) : Prop := sorry

/-- Volynsky's theorem -/
axiom volynsky_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c

/-- Main theorem: If the number of borders of each country on a normal map
    is divisible by 3, then the map can be properly colored with four colors -/
theorem four_color_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c :=
by
  sorry

end NUMINAMATH_CALUDE_four_color_theorem_l1376_137655


namespace NUMINAMATH_CALUDE_square_neq_iff_neq_and_neq_neg_l1376_137637

theorem square_neq_iff_neq_and_neq_neg (x y : ℝ) :
  x^2 ≠ y^2 ↔ x ≠ y ∧ x ≠ -y := by
  sorry

end NUMINAMATH_CALUDE_square_neq_iff_neq_and_neq_neg_l1376_137637


namespace NUMINAMATH_CALUDE_remainder_3_pow_210_mod_17_l1376_137609

theorem remainder_3_pow_210_mod_17 : (3^210 : ℕ) % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_210_mod_17_l1376_137609


namespace NUMINAMATH_CALUDE_cosine_squared_sum_equality_l1376_137680

theorem cosine_squared_sum_equality (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2 * x))^2 + (Real.cos (3 * x))^2 = 1 ↔ 
  (∃ k : ℤ, x = (k * Real.pi / 2 + Real.pi / 4) ∨ x = (k * Real.pi / 3 + Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_squared_sum_equality_l1376_137680


namespace NUMINAMATH_CALUDE_lower_average_price_l1376_137635

theorem lower_average_price (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check lower_average_price

end NUMINAMATH_CALUDE_lower_average_price_l1376_137635


namespace NUMINAMATH_CALUDE_third_day_income_l1376_137611

def cab_driver_income (day1 day2 day4 day5 : ℕ) (average : ℚ) : Prop :=
  ∃ day3 : ℕ,
    (day1 + day2 + day3 + day4 + day5 : ℚ) / 5 = average ∧
    day3 = 60

theorem third_day_income :
  cab_driver_income 45 50 65 70 58 :=
sorry

end NUMINAMATH_CALUDE_third_day_income_l1376_137611


namespace NUMINAMATH_CALUDE_min_a_for_probability_half_or_more_l1376_137692

/-- Represents a deck of cards numbered from 1 to 60 -/
def Deck := Finset (Fin 60)

/-- Represents the probability function p(a,b) -/
noncomputable def p (a b : ℕ) : ℚ :=
  let remaining_cards := 58
  let total_ways := Nat.choose remaining_cards 2
  let lower_team_ways := Nat.choose (a - 1) 2
  let higher_team_ways := Nat.choose (48 - a) 2
  (lower_team_ways + higher_team_ways : ℚ) / total_ways

/-- The main theorem to prove -/
theorem min_a_for_probability_half_or_more (deck : Deck) :
  (∀ a < 13, p a (a + 10) < 1/2) ∧ 
  p 13 23 = 473/551 ∧
  p 13 23 ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_min_a_for_probability_half_or_more_l1376_137692


namespace NUMINAMATH_CALUDE_expression_value_l1376_137687

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  10 - 4 * a + 6 * b = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1376_137687


namespace NUMINAMATH_CALUDE_sum_first_44_is_116_l1376_137604

/-- Represents the sequence where the nth 1 is followed by n 3s -/
def specialSequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| k + 1 => if k < (n * (n + 1)) / 2 then
             if k = (n * (n - 1)) / 2 then 1 else 3
           else specialSequence (n + 1) k

/-- The sum of the first 44 terms of the special sequence -/
def sumFirst44 : ℕ := (List.range 44).map (specialSequence 1) |>.sum

/-- Theorem stating that the sum of the first 44 terms is 116 -/
theorem sum_first_44_is_116 : sumFirst44 = 116 := by sorry

end NUMINAMATH_CALUDE_sum_first_44_is_116_l1376_137604


namespace NUMINAMATH_CALUDE_train_speed_is_60_mph_l1376_137634

/-- The speed of a train given its length and the time it takes to pass another train --/
def train_speed (train_length : ℚ) (passing_time : ℚ) : ℚ :=
  (2 * train_length) / (passing_time / 3600)

/-- Theorem stating that the speed of each train is 60 mph --/
theorem train_speed_is_60_mph (train_length : ℚ) (passing_time : ℚ)
  (h1 : train_length = 1/6)
  (h2 : passing_time = 10) :
  train_speed train_length passing_time = 60 := by
  sorry

#eval train_speed (1/6) 10

end NUMINAMATH_CALUDE_train_speed_is_60_mph_l1376_137634


namespace NUMINAMATH_CALUDE_triangle_value_l1376_137644

theorem triangle_value (triangle p : ℤ) 
  (eq1 : triangle + p = 73)
  (eq2 : (triangle + p) + 2*p = 157) : 
  triangle = 31 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l1376_137644


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l1376_137694

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), (p ∧ q) → (p ∨ q)) ∧
  (∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l1376_137694


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_difference_l1376_137691

theorem arithmetic_sequence_sum_difference : 
  let seq1 := List.range 93
  let seq2 := List.range 93
  let sum1 := (List.sum (seq1.map (fun i => 2001 + i)))
  let sum2 := (List.sum (seq2.map (fun i => 201 + i)))
  sum1 - sum2 = 167400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_difference_l1376_137691


namespace NUMINAMATH_CALUDE_modulus_of_z_l1376_137614

-- Define the complex number z
def z : ℂ := 2 + Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1376_137614


namespace NUMINAMATH_CALUDE_root_equation_implies_value_l1376_137603

theorem root_equation_implies_value (m : ℝ) : 
  m^2 - 2*m - 2019 = 0 → 2*m^2 - 4*m = 4038 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_value_l1376_137603


namespace NUMINAMATH_CALUDE_limit_proof_l1376_137602

/-- The limit of (2 - e^(arcsin^2(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
  |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_proof_l1376_137602


namespace NUMINAMATH_CALUDE_a_lt_c_lt_b_l1376_137623

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Conditions
axiom derivative_f : ∀ x, HasDerivAt f (f' x) x

axiom symmetry_f' : ∀ x, f' (x - 1) = f' (1 - x)

axiom symmetry_f : ∀ x, f x = f (2 - x)

axiom monotone_f : MonotoneOn f (Set.Icc (-7) (-6))

-- Define a, b, and c
def a : ℝ := f (Real.log (6 * Real.exp 1 / 5))
def b : ℝ := f (Real.exp 0.2 - 1)
def c : ℝ := f (2 / 9)

-- Theorem to prove
theorem a_lt_c_lt_b : a < c ∧ c < b :=
  sorry

end NUMINAMATH_CALUDE_a_lt_c_lt_b_l1376_137623


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1376_137646

theorem min_value_of_expression (a b : ℤ) (h1 : a > b) (h2 : a ≠ b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a b : ℤ), a > b ∧ a ≠ b ∧ (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1376_137646


namespace NUMINAMATH_CALUDE_partition_existence_l1376_137636

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1) ∧ 0 < a n

/-- A partition of ℕ into infinitely many subsets -/
def Partition (A : ℕ → Set ℕ) : Prop :=
  (∀ i j : ℕ, i ≠ j → A i ∩ A j = ∅) ∧
  (∀ n : ℕ, ∃ i : ℕ, n ∈ A i) ∧
  (∀ i : ℕ, Set.Infinite (A i))

/-- The condition on consecutive elements in each subset -/
def SatisfiesCondition (A : ℕ → Set ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i k : ℕ, ∀ b : ℕ → ℕ,
    (∀ n : ℕ, b n ∈ A i ∧ b n < b (n + 1)) →
    (∀ n : ℕ, n + 1 ≤ a k → b (n + 1) - b n ≤ k)

theorem partition_existence :
  ∀ a : ℕ → ℕ, StrictlyIncreasingSeq a →
  ∃ A : ℕ → Set ℕ, Partition A ∧ SatisfiesCondition A a :=
sorry

end NUMINAMATH_CALUDE_partition_existence_l1376_137636


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l1376_137690

theorem sqrt_720_simplification : Real.sqrt 720 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l1376_137690


namespace NUMINAMATH_CALUDE_root_equation_problem_l1376_137616

theorem root_equation_problem (a b : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ((x + a) * (x + b) * (x + 10) = 0 ∧ x + 2 ≠ 0) ∧
    ((y + a) * (y + b) * (y + 10) = 0 ∧ y + 2 ≠ 0) ∧
    ((z + a) * (z + b) * (z + 10) = 0 ∧ z + 2 ≠ 0)) →
  (∃! w : ℝ, (w + 2*a) * (w + 4) * (w + 8) = 0 ∧ 
    (w + b) * (w + 10) ≠ 0) →
  100 * a + b = 208 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l1376_137616


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1376_137684

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def milkshakes_bought : ℕ := 6

theorem money_left_after_purchase : 
  initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought) = 70 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1376_137684


namespace NUMINAMATH_CALUDE_largest_927_triple_l1376_137625

/-- Converts a base 10 number to its base 9 representation as a list of digits -/
def toBase9 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Interprets a list of digits as a base 10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 9-27 triple -/
def is927Triple (n : ℕ) : Prop :=
  fromDigits (toBase9 n) = 3 * n

/-- States that 108 is the largest 9-27 triple -/
theorem largest_927_triple :
  (∀ m : ℕ, m > 108 → ¬(is927Triple m)) ∧ is927Triple 108 := by
  sorry

end NUMINAMATH_CALUDE_largest_927_triple_l1376_137625


namespace NUMINAMATH_CALUDE_sequence_bound_l1376_137619

theorem sequence_bound (x : ℕ → ℝ) (b : ℝ) : 
  (∀ n : ℕ, x (n + 1) = x n ^ 2 - 4 * x n) →
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃ k : ℕ, x k ≥ b) →
  b = (3 + Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_bound_l1376_137619


namespace NUMINAMATH_CALUDE_talent_show_participants_l1376_137652

theorem talent_show_participants (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 →
  difference = 22 →
  girls = (total + difference) / 2 →
  girls = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_talent_show_participants_l1376_137652


namespace NUMINAMATH_CALUDE_ellens_age_l1376_137615

/-- Proves Ellen's age given Martha's age and the relationship between their ages -/
theorem ellens_age (martha_age : ℕ) (h : martha_age = 32) :
  ∃ (ellen_age : ℕ), martha_age = 2 * (ellen_age + 6) ∧ ellen_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellens_age_l1376_137615


namespace NUMINAMATH_CALUDE_kiwis_for_18_apples_l1376_137670

-- Define the costs of fruits in terms of an arbitrary unit
variable (apple_cost banana_cost cucumber_cost kiwi_cost : ℚ)

-- Define the conditions
axiom apple_banana_ratio : 9 * apple_cost = 3 * banana_cost
axiom banana_cucumber_ratio : banana_cost = 2 * cucumber_cost
axiom cucumber_kiwi_ratio : 3 * cucumber_cost = 4 * kiwi_cost

-- Define the theorem
theorem kiwis_for_18_apples : 
  ∃ n : ℕ, (18 * apple_cost = n * kiwi_cost) ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_kiwis_for_18_apples_l1376_137670


namespace NUMINAMATH_CALUDE_good_games_count_l1376_137662

def games_from_friend : ℕ := 50
def games_from_garage_sale : ℕ := 27
def non_working_games : ℕ := 74

def total_games : ℕ := games_from_friend + games_from_garage_sale

theorem good_games_count : total_games - non_working_games = 3 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l1376_137662


namespace NUMINAMATH_CALUDE_prob_fifth_six_given_two_sixes_l1376_137698

/-- Represents a six-sided die -/
inductive Die
| Fair
| Biased

/-- Probability of rolling a six for a given die -/
def prob_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 1/2

/-- Probability of rolling a number other than six for a given die -/
def prob_not_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/10

/-- Probability of rolling at least two sixes in four rolls for a given die -/
def prob_at_least_two_sixes (d : Die) : ℚ :=
  match d with
  | Die.Fair => 11/1296
  | Die.Biased => 11/16

/-- The main theorem -/
theorem prob_fifth_six_given_two_sixes (d : Die) : 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) *
  (prob_six d * prob_at_least_two_sixes d) / 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) = 325/656 :=
sorry

end NUMINAMATH_CALUDE_prob_fifth_six_given_two_sixes_l1376_137698


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1376_137643

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a < b ∧ b < c →          -- Ordering of numbers
  a = 28 :=                -- Smallest number is 28
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1376_137643


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l1376_137640

theorem simplify_fourth_root (x y : ℕ+) :
  (2^6 * 3^5 * 5^2 : ℝ)^(1/4) = x * y^(1/4) →
  x + y = 306 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l1376_137640


namespace NUMINAMATH_CALUDE_annulus_equal_area_division_l1376_137678

theorem annulus_equal_area_division (r : ℝ) : 
  r > 0 ∧ r < 14 ∧ 
  (π * (14^2 - r^2) = π * (r^2 - 2^2)) → 
  r = 10 := by sorry

end NUMINAMATH_CALUDE_annulus_equal_area_division_l1376_137678


namespace NUMINAMATH_CALUDE_gcf_60_90_l1376_137693

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_90_l1376_137693


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l1376_137683

/-- The game that predicts marriage age based on name, age, birth month, and siblings' ages -/
def marriage_age_prediction 
  (name_length : ℕ) 
  (age : ℕ) 
  (birth_month : ℕ) 
  (sibling_ages : List ℕ) : ℕ :=
  let step1 := name_length + 2 * age
  let step2 := step1 * (sibling_ages.sum)
  let step3 := step2 / (sibling_ages.length)
  step3 * birth_month

/-- Theorem stating that Sarah's predicted marriage age is 966 -/
theorem sarah_marriage_age : 
  marriage_age_prediction 5 9 7 [5, 7] = 966 := by
  sorry

#eval marriage_age_prediction 5 9 7 [5, 7]

end NUMINAMATH_CALUDE_sarah_marriage_age_l1376_137683


namespace NUMINAMATH_CALUDE_constant_k_value_l1376_137697

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) ↔ k = -18 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l1376_137697


namespace NUMINAMATH_CALUDE_number_exists_l1376_137629

theorem number_exists : ∃ x : ℝ, 0.6667 * x - 10 = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1376_137629


namespace NUMINAMATH_CALUDE_cos_negative_45_degrees_l1376_137671

theorem cos_negative_45_degrees : Real.cos (-(Real.pi / 4)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_45_degrees_l1376_137671


namespace NUMINAMATH_CALUDE_sum_of_single_digit_numbers_l1376_137676

theorem sum_of_single_digit_numbers (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → B = A - 2 → A = 5 + 3 → A + B = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_single_digit_numbers_l1376_137676


namespace NUMINAMATH_CALUDE_charlie_calculation_l1376_137658

theorem charlie_calculation (x : ℝ) : 
  (x / 7 + 20 = 21) → (x * 7 - 20 = 29) := by
sorry

end NUMINAMATH_CALUDE_charlie_calculation_l1376_137658


namespace NUMINAMATH_CALUDE_special_function_at_one_seventh_l1376_137688

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1) ∧
  f 0 = 0 ∧ f 1 = 1 ∧
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧
    ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≤ y →
      f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem special_function_at_one_seventh (f : ℝ → ℝ) (h : special_function f) :
  f (1/7) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_seventh_l1376_137688


namespace NUMINAMATH_CALUDE_sum_powers_i_2047_l1376_137660

def imaginary_unit_sum (i : ℂ) : ℕ → ℂ
  | 0 => 1
  | n + 1 => i^(n + 1) + imaginary_unit_sum i n

theorem sum_powers_i_2047 (i : ℂ) (h : i^2 = -1) :
  imaginary_unit_sum i 2047 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_i_2047_l1376_137660


namespace NUMINAMATH_CALUDE_domain_shift_l1376_137617

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_shift (h : ∀ x, f (x + 1) ∈ domain_f_shifted ↔ x ∈ domain_f_shifted) :
  (∀ x, f x ∈ Set.Icc (-1) 4 ↔ x ∈ Set.Icc (-1) 4) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l1376_137617


namespace NUMINAMATH_CALUDE_parallelogram_area_l1376_137628

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 22 → 
    height = 21 → 
    area = base * height → 
    area = 462 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1376_137628


namespace NUMINAMATH_CALUDE_sin_pi_eight_squared_l1376_137654

theorem sin_pi_eight_squared : 1 - 2 * Real.sin (π / 8) ^ 2 = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_pi_eight_squared_l1376_137654


namespace NUMINAMATH_CALUDE_walter_school_allocation_l1376_137633

/-- Represents Walter's work schedule and earnings --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation_ratio : ℚ

/-- Calculates the amount Walter allocates for school given his work schedule --/
def school_allocation (schedule : WorkSchedule) : ℚ :=
  schedule.days_per_week * schedule.hours_per_day * schedule.hourly_rate * schedule.school_allocation_ratio

/-- Theorem stating that Walter allocates $75 for school each week --/
theorem walter_school_allocation :
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation_ratio := 3/4
  }
  school_allocation walter_schedule = 75 := by
  sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l1376_137633


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1376_137674

theorem exam_score_calculation (total_questions : ℕ) (answered_questions : ℕ) (correct_answers : ℕ) (raw_score : ℚ) :
  total_questions = 85 →
  answered_questions = 82 →
  correct_answers = 70 →
  raw_score = 67 →
  ∃ (points_per_correct : ℚ),
    points_per_correct * correct_answers - (answered_questions - correct_answers) * (1/4 : ℚ) = raw_score ∧
    points_per_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1376_137674


namespace NUMINAMATH_CALUDE_martin_wasted_time_l1376_137626

def traffic_time : ℝ := 2
def freeway_time_multiplier : ℝ := 4

theorem martin_wasted_time : 
  traffic_time + freeway_time_multiplier * traffic_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_martin_wasted_time_l1376_137626


namespace NUMINAMATH_CALUDE_notebook_pages_calculation_l1376_137610

theorem notebook_pages_calculation (num_notebooks : ℕ) (pages_per_day : ℕ) (days_lasted : ℕ) : 
  num_notebooks > 0 → 
  pages_per_day > 0 → 
  days_lasted > 0 → 
  (pages_per_day * days_lasted) % num_notebooks = 0 → 
  (pages_per_day * days_lasted) / num_notebooks = 40 :=
by
  sorry

#check notebook_pages_calculation 5 4 50

end NUMINAMATH_CALUDE_notebook_pages_calculation_l1376_137610


namespace NUMINAMATH_CALUDE_power_equation_solution_l1376_137622

theorem power_equation_solution : ∃ x : ℕ, 27^3 + 27^3 + 27^3 + 27^3 = 3^x :=
by
  use 11
  have h1 : 27 = 3^3 := by sorry
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1376_137622


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l1376_137665

/-- Represents the types of land in the farm. -/
inductive LandType
  | Flat
  | Ditch
  | Sloped

/-- Represents the total acreage for each land type. -/
def total_acreage : LandType → ℕ
  | LandType.Flat => 150
  | LandType.Ditch => 30
  | LandType.Sloped => 90

/-- The total acreage of all land types. -/
def total_land : ℕ := 270

/-- The size of the sample to be taken. -/
def sample_size : ℕ := 18

/-- Calculates the stratified sample size for a given land type. -/
def stratified_sample (land : LandType) : ℕ :=
  (total_acreage land * sample_size) / total_land

/-- Theorem stating the correct stratified sampling allocation. -/
theorem correct_stratified_sampling :
  (stratified_sample LandType.Flat = 10) ∧
  (stratified_sample LandType.Ditch = 2) ∧
  (stratified_sample LandType.Sloped = 6) :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l1376_137665


namespace NUMINAMATH_CALUDE_spherical_coordinate_negation_l1376_137631

/-- Given a point with rectangular coordinates (-3, 5, -2) and corresponding
    spherical coordinates (r, θ, φ), prove that the point with spherical
    coordinates (r, -θ, φ) has rectangular coordinates (-3, -5, -2). -/
theorem spherical_coordinate_negation (r θ φ : ℝ) :
  (r * Real.sin φ * Real.cos θ = -3 ∧
   r * Real.sin φ * Real.sin θ = 5 ∧
   r * Real.cos φ = -2) →
  (r * Real.sin φ * Real.cos (-θ) = -3 ∧
   r * Real.sin φ * Real.sin (-θ) = -5 ∧
   r * Real.cos φ = -2) := by
  sorry


end NUMINAMATH_CALUDE_spherical_coordinate_negation_l1376_137631


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l1376_137638

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := 16

/-- The number of movies watched -/
def movies_watched : ℕ := 19

/-- The difference between movies watched and books read -/
def movie_book_difference : ℕ := 3

theorem crazy_silly_school_books :
  num_books = movies_watched - movie_book_difference :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l1376_137638


namespace NUMINAMATH_CALUDE_count_numbers_greater_than_three_l1376_137627

theorem count_numbers_greater_than_three : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.9, 1/3}
  (numbers.filter (λ x => x > 3)).card = 0 := by
sorry

end NUMINAMATH_CALUDE_count_numbers_greater_than_three_l1376_137627


namespace NUMINAMATH_CALUDE_train_length_l1376_137682

/-- Calculates the length of a train given the bridge length, train speed, and time to pass the bridge. -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  bridge_length = 160 ∧ 
  train_speed_kmh = 40 ∧ 
  time_to_pass = 25.2 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - bridge_length = 120 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1376_137682


namespace NUMINAMATH_CALUDE_man_walking_speed_percentage_l1376_137612

/-- Proves that if a man's usual time to cover a distance is 72.00000000000001 minutes,
    and he takes 24 minutes more when walking at a slower speed,
    then he is walking at 75% of his usual speed. -/
theorem man_walking_speed_percentage : 
  let usual_time : ℝ := 72.00000000000001
  let additional_time : ℝ := 24
  let new_time : ℝ := usual_time + additional_time
  let speed_ratio : ℝ := usual_time / new_time
  speed_ratio = 0.75 := by sorry

end NUMINAMATH_CALUDE_man_walking_speed_percentage_l1376_137612


namespace NUMINAMATH_CALUDE_joan_seashells_l1376_137659

/-- Given 245 initial seashells, prove that after giving 3/5 to Mike and 2/5 of the remainder to Lisa, Joan is left with 59 seashells. -/
theorem joan_seashells (initial_seashells : ℕ) (mike_fraction : ℚ) (lisa_fraction : ℚ) :
  initial_seashells = 245 →
  mike_fraction = 3 / 5 →
  lisa_fraction = 2 / 5 →
  initial_seashells - (initial_seashells * mike_fraction).floor -
    ((initial_seashells - (initial_seashells * mike_fraction).floor) * lisa_fraction).floor = 59 := by
  sorry


end NUMINAMATH_CALUDE_joan_seashells_l1376_137659


namespace NUMINAMATH_CALUDE_gcd_228_1995_decimal_to_ternary_l1376_137681

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Convert 104 to base 3
theorem decimal_to_ternary :
  ∃ (a b c d e : Nat),
    104 = a * 3^4 + b * 3^3 + c * 3^2 + d * 3^1 + e * 3^0 ∧
    a = 1 ∧ b = 0 ∧ c = 2 ∧ d = 1 ∧ e = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_decimal_to_ternary_l1376_137681


namespace NUMINAMATH_CALUDE_basketball_players_count_l1376_137607

/-- The number of boys playing basketball in a group with given conditions -/
def boys_playing_basketball (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither

theorem basketball_players_count :
  boys_playing_basketball 22 15 3 18 = 19 :=
by sorry

end NUMINAMATH_CALUDE_basketball_players_count_l1376_137607


namespace NUMINAMATH_CALUDE_highlighters_count_l1376_137666

/-- The total number of highlighters in Kaya's teacher's desk -/
def total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ :=
  pink + yellow + blue

/-- Theorem stating that the total number of highlighters is 22 -/
theorem highlighters_count :
  total_highlighters 9 8 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l1376_137666


namespace NUMINAMATH_CALUDE_apple_packing_problem_l1376_137686

theorem apple_packing_problem (apples_per_crate : ℕ) (num_crates : ℕ) 
  (rotten_percentage : ℚ) (apples_per_box : ℕ) (available_boxes : ℕ) :
  apples_per_crate = 400 →
  num_crates = 35 →
  rotten_percentage = 11/100 →
  apples_per_box = 30 →
  available_boxes = 1000 →
  ∃ (boxes_needed : ℕ), 
    boxes_needed = 416 ∧ 
    boxes_needed * apples_per_box ≥ 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    (boxes_needed - 1) * apples_per_box < 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    boxes_needed ≤ available_boxes :=
by sorry

end NUMINAMATH_CALUDE_apple_packing_problem_l1376_137686


namespace NUMINAMATH_CALUDE_g_zero_l1376_137648

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_def : h = f * g

-- Define the constant terms of f and h
axiom f_const : f.coeff 0 = -6
axiom h_const : h.coeff 0 = 12

-- Theorem to prove
theorem g_zero : g.eval 0 = -2 := by sorry

end NUMINAMATH_CALUDE_g_zero_l1376_137648


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_minus_one_range_l1376_137649

theorem cos_squared_plus_sin_minus_one_range :
  ∀ x : ℝ, -2 ≤ (Real.cos x)^2 + Real.sin x - 1 ∧ (Real.cos x)^2 + Real.sin x - 1 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_minus_one_range_l1376_137649


namespace NUMINAMATH_CALUDE_radical_equation_solution_l1376_137657

theorem radical_equation_solution (a b c : ℕ) (N : ℝ) : 
  a > 1 → b > 1 → c > 1 → N ≠ 1 →
  (N^((1:ℝ)/a + 1/(a*b) + 2/(a*b*c)) = N^(17/24) ↔ b = 4) :=
sorry

end NUMINAMATH_CALUDE_radical_equation_solution_l1376_137657


namespace NUMINAMATH_CALUDE_circular_seating_pairs_l1376_137673

/-- The number of adjacent pairs in a circular seating arrangement --/
def adjacentPairs (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with n people,
    the number of different sets of two people sitting next to each other is n --/
theorem circular_seating_pairs (n : ℕ) (h : n > 0) :
  adjacentPairs n = n := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_pairs_l1376_137673


namespace NUMINAMATH_CALUDE_kittens_given_to_friends_l1376_137679

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa has left -/
def remaining_kittens : ℕ := 4

/-- The number of kittens Alyssa gave to her friends -/
def kittens_given_away : ℕ := initial_kittens - remaining_kittens

theorem kittens_given_to_friends :
  kittens_given_away = 4 := by sorry

end NUMINAMATH_CALUDE_kittens_given_to_friends_l1376_137679


namespace NUMINAMATH_CALUDE_sum_of_ages_seven_years_hence_l1376_137651

-- Define X's current age
def X_current : ℕ := 45

-- Define Y's current age as a function of X's current age
def Y_current : ℕ := X_current - 21

-- Theorem to prove
theorem sum_of_ages_seven_years_hence : 
  X_current + Y_current + 14 = 83 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_seven_years_hence_l1376_137651


namespace NUMINAMATH_CALUDE_wheel_speed_proof_l1376_137667

/-- Proves that the original speed of a wheel is 20 mph given specific conditions -/
theorem wheel_speed_proof (circumference : Real) (speed_increase : Real) (time_decrease : Real) :
  circumference = 50 / 5280 → -- circumference in miles
  speed_increase = 10 → -- speed increase in mph
  time_decrease = 1 / (2 * 3600) → -- time decrease in hours
  ∃ (r : Real),
    r > 0 ∧
    r * (50 * 3600 / (5280 * r)) = 50 / 5280 * 3600 ∧
    (r + speed_increase) * (50 * 3600 / (5280 * r) - time_decrease) = 50 / 5280 * 3600 ∧
    r = 20 :=
by sorry


end NUMINAMATH_CALUDE_wheel_speed_proof_l1376_137667


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_25_l1376_137672

-- Define functions to convert numbers from different bases to base 10
def base8ToBase10 (n : ℕ) : ℕ := sorry

def base4ToBase10 (n : ℕ) : ℕ := sorry

def base5ToBase10 (n : ℕ) : ℕ := sorry

def base3ToBase10 (n : ℕ) : ℕ := sorry

-- Define the numbers in their respective bases
def num1 : ℕ := 254  -- in base 8
def den1 : ℕ := 14   -- in base 4
def num2 : ℕ := 132  -- in base 5
def den2 : ℕ := 26   -- in base 3

-- Theorem to prove
theorem sum_of_fractions_equals_25 :
  (base8ToBase10 num1 / base4ToBase10 den1) + (base5ToBase10 num2 / base3ToBase10 den2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_25_l1376_137672


namespace NUMINAMATH_CALUDE_f_monotone_increasing_on_interval_l1376_137618

/-- The function f(x) = (1/2)^(x^2 - x - 1) is monotonically increasing on (-∞, 1/2) -/
theorem f_monotone_increasing_on_interval :
  ∀ x y : ℝ, x < y → x < (1/2 : ℝ) → y < (1/2 : ℝ) →
  ((1/2 : ℝ) ^ (x^2 - x - 1)) < ((1/2 : ℝ) ^ (y^2 - y - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_on_interval_l1376_137618


namespace NUMINAMATH_CALUDE_walking_speed_l1376_137699

theorem walking_speed (distance : Real) (time_minutes : Real) (speed : Real) : 
  distance = 500 ∧ time_minutes = 6 → speed = 5000 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_l1376_137699


namespace NUMINAMATH_CALUDE_max_min_f_l1376_137639

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 0 ∧ min = -Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_l1376_137639


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l1376_137647

/-- The number of sandwiches Michelle has left to give to her other co-workers -/
def sandwiches_left (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - first - second - (2 * first) - (3 * second)

/-- Proof that Michelle has 26 sandwiches left -/
theorem michelle_sandwiches : sandwiches_left 50 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l1376_137647
