import Mathlib

namespace NUMINAMATH_CALUDE_integral_sin_over_one_minus_cos_squared_l1614_161498

theorem integral_sin_over_one_minus_cos_squared (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π / 2) π, (2 * Real.sin x) / ((1 - Real.cos x)^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_over_one_minus_cos_squared_l1614_161498


namespace NUMINAMATH_CALUDE_train_speed_l1614_161484

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) :
  length = 125.01 →
  time = 5 →
  let speed := (length / 1000) / (time / 3600)
  ∃ ε > 0, abs (speed - 90.0072) < ε := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1614_161484


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l1614_161450

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 2 / 3 → 
  b / c = 3 / 4 → 
  a^2 + c^2 = 180 → 
  b = 9 := by
sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l1614_161450


namespace NUMINAMATH_CALUDE_triangle_side_length_l1614_161429

/-- 
Given a triangle XYZ where:
- y = 7
- z = 3
- cos(Y - Z) = 40/41
Prove that x² = 56.1951
-/
theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 40 / 41 →
  x ^ 2 = 56.1951 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1614_161429


namespace NUMINAMATH_CALUDE_mean_median_difference_l1614_161495

/-- Represents the absence data for a class of students -/
structure AbsenceData where
  students : ℕ
  absences : List (ℕ × ℕ)  -- (days missed, number of students)

/-- Calculates the median number of days missed -/
def median (data : AbsenceData) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (data : AbsenceData) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) : 
  data.students = 20 ∧ 
  data.absences = [(0, 4), (1, 3), (2, 7), (3, 2), (4, 2), (5, 1), (6, 1)] →
  mean data - median data = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1614_161495


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l1614_161499

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 1824) (hb : b = 5435) (hc : c = 80525) : 
  (a * b * c) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l1614_161499


namespace NUMINAMATH_CALUDE_sqrt_a_minus_b_is_natural_l1614_161407

theorem sqrt_a_minus_b_is_natural (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) :
  ∃ k : ℕ, a - b = k^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_b_is_natural_l1614_161407


namespace NUMINAMATH_CALUDE_triangle_theorem_l1614_161403

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 3)
  (hcosB : Real.cos t.angleB = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.angleC) = (3 * Real.sqrt 15) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1614_161403


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1614_161465

theorem positive_integer_solutions_of_equation : 
  ∀ x y : ℕ+, 
    (x : ℚ) - (y : ℚ) = (x : ℚ) / (y : ℚ) + (x : ℚ)^2 / (y : ℚ)^2 + (x : ℚ)^3 / (y : ℚ)^3 
    ↔ (x = 28 ∧ y = 14) ∨ (x = 112 ∧ y = 28) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1614_161465


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_l1614_161468

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → 
    (m < 100 ∨ m ≥ 1000 ∨
    (∀ k : ℕ, m ≠ 3 * k + 1) ∨
    (∀ k : ℕ, m ≠ 4 * k + 1) ∨
    (∀ k : ℕ, m ≠ 5 * k + 1) ∨
    (∀ k : ℕ, m ≠ 7 * k + 1))) ∧
  n = 421 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_l1614_161468


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l1614_161418

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem two_digit_number_theorem (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ 
  (10 * x + y) - (10 * y + x) = 81 ∧ 
  is_prime (x + y) → 
  x - y = 7 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l1614_161418


namespace NUMINAMATH_CALUDE_park_visitors_difference_l1614_161441

theorem park_visitors_difference (total : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → hikers = 427 → total = hikers + bikers → hikers - bikers = 178 := by
  sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l1614_161441


namespace NUMINAMATH_CALUDE_mail_difference_l1614_161477

theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday < tuesday →
  thursday = wednesday + 15 →
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - wednesday = 5 := by
sorry

end NUMINAMATH_CALUDE_mail_difference_l1614_161477


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1614_161481

theorem trigonometric_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1614_161481


namespace NUMINAMATH_CALUDE_intersection_range_of_b_l1614_161471

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_range_of_b :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_of_b_l1614_161471


namespace NUMINAMATH_CALUDE_max_books_read_l1614_161425

def reading_speed : ℕ := 120
def pages_per_book : ℕ := 360
def reading_time : ℕ := 8

theorem max_books_read : 
  (reading_speed * reading_time) / pages_per_book = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_books_read_l1614_161425


namespace NUMINAMATH_CALUDE_expression_evaluation_l1614_161464

theorem expression_evaluation :
  ((2^2009)^2 - (2^2007)^2) / ((2^2008)^2 - (2^2006)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1614_161464


namespace NUMINAMATH_CALUDE_probability_white_or_black_l1614_161436

-- Define the total number of balls and the number of balls to be drawn
def total_balls : ℕ := 5
def drawn_balls : ℕ := 3

-- Define the number of favorable outcomes (combinations including white or black)
def favorable_outcomes : ℕ := 9

-- Define the total number of possible outcomes
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

-- State the theorem
theorem probability_white_or_black :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_white_or_black_l1614_161436


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1614_161485

/-- Represents the configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  small_square_side : ℝ
  large_square_side : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The theorem stating the ratio of the rectangle's length to its width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h1 : config.large_square_side = 4 * config.small_square_side)
  (h2 : config.rectangle_length = config.large_square_side)
  (h3 : config.rectangle_width = config.large_square_side - 3 * config.small_square_side) :
  config.rectangle_length / config.rectangle_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1614_161485


namespace NUMINAMATH_CALUDE_largest_square_from_string_l1614_161449

theorem largest_square_from_string (string_length : ℝ) (side_length : ℝ) : 
  string_length = 32 →
  side_length * 4 = string_length →
  side_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_from_string_l1614_161449


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1614_161482

/-- Represents the number of students in each grade and the total sample size -/
structure SchoolPopulation where
  total : Nat
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat
  sampleSize : Nat

/-- Represents the number of students sampled from each grade -/
structure StratifiedSample where
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat

/-- Function to calculate the stratified sample given a school population -/
def calculateStratifiedSample (pop : SchoolPopulation) : StratifiedSample :=
  { firstYear := pop.firstYear * pop.sampleSize / pop.total,
    secondYear := pop.secondYear * pop.sampleSize / pop.total,
    thirdYear := pop.thirdYear * pop.sampleSize / pop.total }

theorem stratified_sampling_theorem (pop : SchoolPopulation)
    (h1 : pop.total = 1000)
    (h2 : pop.firstYear = 500)
    (h3 : pop.secondYear = 300)
    (h4 : pop.thirdYear = 200)
    (h5 : pop.sampleSize = 100)
    (h6 : pop.total = pop.firstYear + pop.secondYear + pop.thirdYear) :
    let sample := calculateStratifiedSample pop
    sample.firstYear = 50 ∧ sample.secondYear = 30 ∧ sample.thirdYear = 20 :=
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1614_161482


namespace NUMINAMATH_CALUDE_glass_pane_impact_l1614_161456

/-- Represents a point inside a rectangle --/
structure ImpactPoint (width height : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 < x ∧ x < width
  y_bound : 0 < y ∧ y < height

/-- The glass pane problem --/
theorem glass_pane_impact
  (width : ℝ)
  (height : ℝ)
  (p : ImpactPoint width height)
  (h_width : width = 8)
  (h_height : height = 6)
  (h_right_area : p.x * height = 3 * (width - p.x) * height)
  (h_bottom_area : p.y * width = 2 * (height - p.y) * p.x) :
  p.x = 2 ∧ (width - p.x) = 6 ∧ p.y = 3 ∧ (height - p.y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_glass_pane_impact_l1614_161456


namespace NUMINAMATH_CALUDE_equation_solution_l1614_161445

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ∧ x = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1614_161445


namespace NUMINAMATH_CALUDE_min_sum_abc_def_l1614_161408

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def are_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem min_sum_abc_def :
  ∀ a b c d e f : ℕ,
    is_valid_digit a → is_valid_digit b → is_valid_digit c →
    is_valid_digit d → is_valid_digit e → is_valid_digit f →
    are_distinct a b c d e f →
    459 ≤ to_number a b c + to_number d e f :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_def_l1614_161408


namespace NUMINAMATH_CALUDE_joans_kittens_l1614_161496

theorem joans_kittens (initial_kittens given_away_kittens : ℕ) 
  (h1 : initial_kittens = 15)
  (h2 : given_away_kittens = 7) :
  initial_kittens - given_away_kittens = 8 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l1614_161496


namespace NUMINAMATH_CALUDE_tangent_length_specific_circle_l1614_161461

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle passing through three given points -/
def circleThrough (A B C : ℝ × ℝ) : Circle :=
  sorry

/-- The length of a tangent segment from a point to a circle -/
def tangentLength (P : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry

/-- The theorem stating the length of the tangent segment -/
theorem tangent_length_specific_circle :
  let A : ℝ × ℝ := (4, 5)
  let B : ℝ × ℝ := (7, 9)
  let C : ℝ × ℝ := (6, 14)
  let P : ℝ × ℝ := (1, 1)
  let c := circleThrough A B C
  tangentLength P c = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_specific_circle_l1614_161461


namespace NUMINAMATH_CALUDE_max_valid_domains_l1614_161453

def f (x : ℝ) : ℝ := x^2 - 1

def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x = 0 ∨ f x = 1) ∧
  (∃ x ∈ D, f x = 0) ∧
  (∃ x ∈ D, f x = 1)

theorem max_valid_domains :
  ∃ (domains : Finset (Set ℝ)),
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D, is_valid_domain D → D ∈ domains) ∧
    domains.card = 9 :=
sorry

end NUMINAMATH_CALUDE_max_valid_domains_l1614_161453


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1614_161467

theorem simplify_and_rationalize (a b c d e f g h i : ℝ) 
  (ha : a = 3) (hb : b = 7) (hc : c = 5) (hd : d = 8) (he : e = 6) (hf : f = 9) :
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f) = 
  Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1614_161467


namespace NUMINAMATH_CALUDE_expression_evaluation_l1614_161406

theorem expression_evaluation : 
  let c : ℝ := 2
  let d : ℝ := 1/4
  (Real.sqrt (c - d) / (c^2 * Real.sqrt (2*c))) * 
  (Real.sqrt ((c - d)/(c + d)) + Real.sqrt ((c^2 + c*d)/(c^2 - c*d))) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1614_161406


namespace NUMINAMATH_CALUDE_sin_660_deg_l1614_161447

theorem sin_660_deg : Real.sin (660 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_660_deg_l1614_161447


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1614_161476

theorem triangle_trigonometric_identity (A B C : Real) : 
  C = Real.pi / 3 →
  Real.tan (A / 2) + Real.tan (B / 2) = 1 →
  A + B + C = Real.pi →
  Real.sin (A / 2) * Real.sin (B / 2) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identity_l1614_161476


namespace NUMINAMATH_CALUDE_reservoir_capacity_proof_l1614_161466

theorem reservoir_capacity_proof (current_level : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_level = 30)
  (h2 : current_level = 2 * normal_level)
  (h3 : current_level = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_proof_l1614_161466


namespace NUMINAMATH_CALUDE_candy_bar_consumption_l1614_161440

/-- Given that a candy bar contains 31 calories and a person consumed 341 calories,
    prove that the number of candy bars eaten is 11. -/
theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) : 
  calories_per_bar = 31 → total_calories = 341 → total_calories / calories_per_bar = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_consumption_l1614_161440


namespace NUMINAMATH_CALUDE_goat_cost_is_400_l1614_161494

/-- The cost of a single goat in dollars -/
def goat_cost : ℝ := sorry

/-- The number of goats purchased -/
def num_goats : ℕ := 3

/-- The number of llamas purchased -/
def num_llamas : ℕ := 6

/-- The cost of a single llama in terms of goat cost -/
def llama_cost : ℝ := 1.5 * goat_cost

/-- The total amount spent on all animals -/
def total_spent : ℝ := 4800

theorem goat_cost_is_400 : goat_cost = 400 :=
  sorry

end NUMINAMATH_CALUDE_goat_cost_is_400_l1614_161494


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_6_l1614_161405

/-- A function f(x) = x^2 - 2(a-1)x + 2 is decreasing on the interval (-∞, 5] -/
def is_decreasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 5 → y ≤ 5 → f x ≥ f y

/-- The quadratic function f(x) = x^2 - 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a-1)*x + 2

theorem decreasing_quadratic_implies_a_geq_6 :
  ∀ a : ℝ, is_decreasing_on_interval (f a) a → a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_6_l1614_161405


namespace NUMINAMATH_CALUDE_number_divisibility_problem_l1614_161448

theorem number_divisibility_problem :
  ∃ (N : ℕ), N > 0 ∧ N % 44 = 0 ∧ N % 35 = 3 ∧ N / 44 = 12 :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_problem_l1614_161448


namespace NUMINAMATH_CALUDE_ageOfReplacedManIs42_l1614_161401

/-- Given a group of 6 men where:
    - The average age increases by 3 years when two women replace two men
    - One of the men is 26 years old
    - The average age of the women is 34
    This function calculates the age of the other man who was replaced. -/
def ageOfReplacedMan (averageIncrease : ℕ) (knownManAge : ℕ) (womenAverageAge : ℕ) : ℕ :=
  2 * womenAverageAge - knownManAge

/-- Theorem stating that under the given conditions, 
    the age of the other replaced man is 42 years. -/
theorem ageOfReplacedManIs42 :
  ageOfReplacedMan 3 26 34 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ageOfReplacedManIs42_l1614_161401


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1614_161416

/-- An ellipse with focal points (-2, 0) and (2, 0) that intersects the line x + y + 4 = 0 at exactly one point has a major axis of length 8. -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) = 8) →
  (∃! (P : ℝ × ℝ), P ∈ E ∧ P.1 + P.2 + 4 = 0) →
  8 = 8 := by
sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1614_161416


namespace NUMINAMATH_CALUDE_recycle_243_cans_l1614_161462

/-- The number of new cans that can be made from recycling a given number of aluminum cans. -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 3 then 0
  else (initial_cans / 3) + recycle_cans (initial_cans / 3)

/-- Theorem stating that recycling 243 aluminum cans results in 121 new cans. -/
theorem recycle_243_cans :
  recycle_cans 243 = 121 := by sorry

end NUMINAMATH_CALUDE_recycle_243_cans_l1614_161462


namespace NUMINAMATH_CALUDE_problem_solution_l1614_161475

-- Define the conditions
def is_square_root_of_same_number (x y : ℝ) : Prop := ∃ z : ℝ, z > 0 ∧ x^2 = z ∧ y^2 = z

-- Main theorem
theorem problem_solution :
  ∀ (a b c : ℝ),
  (is_square_root_of_same_number (a + 3) (2*a - 15)) →
  (b^(1/3 : ℝ) = -2) →
  (c ≥ 0 ∧ c^(1/2 : ℝ) = c) →
  ((c = 0 → a + b - 2*c = -4) ∧ (c = 1 → a + b - 2*c = -6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1614_161475


namespace NUMINAMATH_CALUDE_triangle_properties_l1614_161410

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  (t.a / t.b = (1 + Real.cos t.A) / Real.cos t.C) →
  (t.A = π / 2) ∧
  (t.a = 1 → ∃ S : ℝ, S ≤ 1/4 ∧ 
    ∀ S' : ℝ, (∃ t' : Triangle, t'.a = 1 ∧ t'.A = π/2 ∧ S' = 1/2 * t'.b * t'.c) → 
      S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1614_161410


namespace NUMINAMATH_CALUDE_power_equation_solution_l1614_161473

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1614_161473


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l1614_161444

/-- Given c and d are real numbers with d ≠ 0, and s and y are defined such that
    s = (3c)^(3d) and s = c^d * y^(3d), prove that y = 3c. -/
theorem tripled_base_and_exponent (c d : ℝ) (s y : ℝ) (h1 : d ≠ 0) 
    (h2 : s = (3 * c) ^ (3 * d)) (h3 : s = c^d * y^(3*d)) : y = 3 * c := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l1614_161444


namespace NUMINAMATH_CALUDE_little_john_sweets_expenditure_l1614_161427

/-- Proof of the amount spent on sweets by Little John --/
theorem little_john_sweets_expenditure 
  (initial_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (final_amount : ℚ)
  (h1 : initial_amount = 20.10)
  (h2 : amount_per_friend = 1)
  (h3 : num_friends = 2)
  (h4 : final_amount = 17.05) :
  initial_amount - (↑num_friends * amount_per_friend) - final_amount = 1.05 := by
  sorry

#check little_john_sweets_expenditure

end NUMINAMATH_CALUDE_little_john_sweets_expenditure_l1614_161427


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_l1614_161487

/-- The circumference of a semicircle given rectangle dimensions --/
theorem semicircle_circumference_from_rectangle (l b : ℝ) (h1 : l = 24) (h2 : b = 16) :
  let rectangle_perimeter := 2 * (l + b)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := π * square_side / 2 + square_side
  ‖semicircle_circumference - 51.40‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_l1614_161487


namespace NUMINAMATH_CALUDE_new_average_income_l1614_161497

/-- Given a family's initial average income, number of earning members, and the income of a deceased member,
    calculate the new average income after the member's death. -/
theorem new_average_income
  (initial_average : ℚ)
  (initial_members : ℕ)
  (deceased_income : ℚ)
  (new_members : ℕ)
  (h1 : initial_average = 735)
  (h2 : initial_members = 4)
  (h3 : deceased_income = 1170)
  (h4 : new_members = initial_members - 1) :
  let initial_total := initial_average * initial_members
  let new_total := initial_total - deceased_income
  new_total / new_members = 590 := by
sorry


end NUMINAMATH_CALUDE_new_average_income_l1614_161497


namespace NUMINAMATH_CALUDE_money_distribution_l1614_161483

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 700)
  (ac_sum : A + C = 300)
  (bc_sum : B + C = 600) :
  C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1614_161483


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1614_161491

theorem quadratic_sum_of_constants (x : ℝ) : 
  ∃ (b c : ℝ), x^2 - 20*x + 49 = (x + b)^2 + c ∧ b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1614_161491


namespace NUMINAMATH_CALUDE_julian_needs_80_more_legos_l1614_161419

/-- The number of additional legos Julian needs to complete two identical airplane models -/
def additional_legos_needed (total_legos : ℕ) (legos_per_model : ℕ) (num_models : ℕ) : ℕ :=
  max 0 (legos_per_model * num_models - total_legos)

/-- Proof that Julian needs 80 more legos -/
theorem julian_needs_80_more_legos :
  additional_legos_needed 400 240 2 = 80 := by
  sorry

#eval additional_legos_needed 400 240 2

end NUMINAMATH_CALUDE_julian_needs_80_more_legos_l1614_161419


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_l1614_161439

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def expression (n : ℕ) : ℕ := (n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4

theorem expression_perfect_square_iff (n : ℕ) (hn : n > 0) :
  is_perfect_square (expression n) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_l1614_161439


namespace NUMINAMATH_CALUDE_parrots_per_cage_l1614_161457

/-- Given a pet store with bird cages, prove the number of parrots in each cage. -/
theorem parrots_per_cage 
  (num_cages : ℕ) 
  (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parakeets_per_cage = 6)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 := by
sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l1614_161457


namespace NUMINAMATH_CALUDE_geometric_series_sum_127_128_l1614_161438

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_127_128 : 
  geometric_series_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_127_128_l1614_161438


namespace NUMINAMATH_CALUDE_alice_bob_meeting_l1614_161446

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's movement (clockwise) -/
def a : ℕ := 7

/-- Bob's movement (counterclockwise) -/
def b : ℕ := 13

/-- The number of turns it takes for Alice and Bob to meet again -/
def meetingTurns : ℕ := 9

/-- Theorem stating that Alice and Bob meet after 'meetingTurns' turns -/
theorem alice_bob_meeting :
  (meetingTurns * (a + n - b)) % n = 0 := by sorry

end NUMINAMATH_CALUDE_alice_bob_meeting_l1614_161446


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l1614_161474

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon, excluding the sides -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of sets of intersecting diagonals in a regular decagon -/
def num_intersecting_diagonals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon intersect inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) : 
  (num_intersecting_diagonals : ℚ) / num_diagonal_pairs = 42 / 119 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l1614_161474


namespace NUMINAMATH_CALUDE_value_of_x_l1614_161486

theorem value_of_x (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1614_161486


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1614_161469

-- Define the hyperbola and its properties
def hyperbola_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-Real.sqrt 10, 0) ∧ F₂ = (Real.sqrt 10, 0)

def point_on_hyperbola (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  let MF₁ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ := (M.2 - F₂.1, M.2 - F₂.2)
  MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0 ∧
  Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2

-- Theorem statement
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) :
  hyperbola_foci F₁ F₂ →
  point_on_hyperbola M F₁ F₂ →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 / 9 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1614_161469


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1614_161490

theorem linear_equation_condition (m : ℝ) : 
  (|m - 1| = 1 ∧ m - 2 ≠ 0) ↔ m = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1614_161490


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1614_161459

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 1) % (x - 2) = 41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1614_161459


namespace NUMINAMATH_CALUDE_greyson_fuel_expense_l1614_161488

/-- Calculates the total fuel expense for a week given the number of refills and cost per refill -/
def total_fuel_expense (num_refills : ℕ) (cost_per_refill : ℕ) : ℕ :=
  num_refills * cost_per_refill

/-- Proves that Greyson's total fuel expense for the week is $40 -/
theorem greyson_fuel_expense :
  total_fuel_expense 4 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_greyson_fuel_expense_l1614_161488


namespace NUMINAMATH_CALUDE_solution_composition_l1614_161415

theorem solution_composition (solution1_percent : Real) (solution1_carbonated : Real) 
  (solution2_carbonated : Real) (mixture_carbonated : Real) :
  solution1_percent = 0.4 →
  solution2_carbonated = 0.55 →
  mixture_carbonated = 0.65 →
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated = mixture_carbonated →
  solution1_carbonated = 0.8 := by
sorry

end NUMINAMATH_CALUDE_solution_composition_l1614_161415


namespace NUMINAMATH_CALUDE_sally_mcqueen_cost_l1614_161493

/-- The cost of Sally McQueen given the costs of Lightning McQueen and Mater -/
theorem sally_mcqueen_cost 
  (lightning_cost : ℝ) 
  (mater_cost_percentage : ℝ) 
  (sally_cost_multiplier : ℝ) 
  (h1 : lightning_cost = 140000)
  (h2 : mater_cost_percentage = 0.1)
  (h3 : sally_cost_multiplier = 3) : 
  sally_cost_multiplier * (mater_cost_percentage * lightning_cost) = 42000 := by
  sorry

end NUMINAMATH_CALUDE_sally_mcqueen_cost_l1614_161493


namespace NUMINAMATH_CALUDE_propositions_correctness_l1614_161422

-- Proposition ①
def proposition_1 : Prop :=
  (¬∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 < 3*x)

-- Proposition ②
def proposition_2 : Prop :=
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Proposition ③
def proposition_3 : Prop :=
  ∀ a : ℝ, (a > 3 → a > Real.pi) ∧ ¬(a > Real.pi → a > 3)

-- Proposition ④
def proposition_4 : Prop :=
  ∀ a : ℝ, (∀ x : ℝ, (x + 2) * (x + a) = (-x + 2) * (-x + a)) → a = -2

theorem propositions_correctness :
  ¬proposition_1 ∧ proposition_2 ∧ ¬proposition_3 ∧ proposition_4 :=
sorry

end NUMINAMATH_CALUDE_propositions_correctness_l1614_161422


namespace NUMINAMATH_CALUDE_min_value_theorem_l1614_161409

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (m : ℝ), m = 16/7 ∧ ∀ (z : ℝ), z ≥ m ↔ z ≥ x^2/(x+1) + y^2/(y+2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1614_161409


namespace NUMINAMATH_CALUDE_g_of_3_equals_4_l1614_161413

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem statement
theorem g_of_3_equals_4 : g 3 = 4 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_4_l1614_161413


namespace NUMINAMATH_CALUDE_circle_equation_l1614_161424

/-- A circle C in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- Circle C satisfies the given conditions -/
def satisfies_conditions (C : Circle) : Prop :=
  let (a, b) := C.center
  line a b ∧
  (1 - a)^2 + (3 - b)^2 = C.radius^2 ∧
  (3 - a)^2 + (5 - b)^2 = C.radius^2

/-- The standard equation of circle C -/
def standard_equation (C : Circle) (x y : ℝ) : Prop :=
  let (a, b) := C.center
  (x - a)^2 + (y - b)^2 = C.radius^2

theorem circle_equation :
  ∃ C : Circle, satisfies_conditions C ∧
    ∀ x y : ℝ, standard_equation C x y ↔ (x - 1)^2 + (y - 5)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1614_161424


namespace NUMINAMATH_CALUDE_happy_valley_theorem_l1614_161417

/-- The number of ways to arrange animals in the Happy Valley Kennel -/
def happy_valley_arrangements : ℕ :=
  let num_chickens : ℕ := 3
  let num_dogs : ℕ := 4
  let num_cats : ℕ := 6
  let total_animals : ℕ := num_chickens + num_dogs + num_cats
  let group_arrangements : ℕ := 2  -- chicken-dog or dog-chicken around cats
  let chicken_arrangements : ℕ := Nat.factorial num_chickens
  let dog_arrangements : ℕ := Nat.factorial num_dogs
  let cat_arrangements : ℕ := Nat.factorial num_cats
  group_arrangements * chicken_arrangements * dog_arrangements * cat_arrangements

/-- Theorem stating the correct number of arrangements for the Happy Valley Kennel problem -/
theorem happy_valley_theorem : happy_valley_arrangements = 69120 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_theorem_l1614_161417


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1614_161472

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 36 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  y = 3*x ∨ y = -3*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equations x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1614_161472


namespace NUMINAMATH_CALUDE_shekar_science_marks_l1614_161434

/-- Represents a student's marks in different subjects -/
structure StudentMarks where
  mathematics : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ
  science : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Shekar's marks in other subjects and his average, 
    his science marks must be 65 -/
theorem shekar_science_marks (marks : StudentMarks) 
  (h1 : marks.mathematics = 76)
  (h2 : marks.social_studies = 82)
  (h3 : marks.english = 47)
  (h4 : marks.biology = 85)
  (h5 : marks.average = 71)
  (h6 : marks.total_subjects = 5)
  : marks.science = 65 := by
  sorry

#check shekar_science_marks

end NUMINAMATH_CALUDE_shekar_science_marks_l1614_161434


namespace NUMINAMATH_CALUDE_quadratic_sum_l1614_161426

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → 
  a + h + k = -23.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1614_161426


namespace NUMINAMATH_CALUDE_equation_solution_l1614_161412

theorem equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1614_161412


namespace NUMINAMATH_CALUDE_sticker_distribution_l1614_161451

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical stickers onto 5 sheets of paper,
    with each sheet receiving at least one sticker -/
theorem sticker_distribution : distribute 10 5 = 7 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1614_161451


namespace NUMINAMATH_CALUDE_A_oxen_count_l1614_161400

/-- Represents the number of oxen A put for grazing -/
def X : ℕ := sorry

/-- Total rent of the pasture in Rs -/
def total_rent : ℕ := 175

/-- Number of months A's oxen grazed -/
def A_months : ℕ := 7

/-- Number of oxen B put for grazing -/
def B_oxen : ℕ := 12

/-- Number of months B's oxen grazed -/
def B_months : ℕ := 5

/-- Number of oxen C put for grazing -/
def C_oxen : ℕ := 15

/-- Number of months C's oxen grazed -/
def C_months : ℕ := 3

/-- C's share of rent in Rs -/
def C_share : ℕ := 45

/-- Theorem stating that A put 10 oxen for grazing -/
theorem A_oxen_count : X = 10 := by sorry

end NUMINAMATH_CALUDE_A_oxen_count_l1614_161400


namespace NUMINAMATH_CALUDE_find_other_number_l1614_161435

theorem find_other_number (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 14) (hLCM : Nat.lcm A B = 312) : B = 182 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l1614_161435


namespace NUMINAMATH_CALUDE_classroom_gpa_problem_l1614_161430

theorem classroom_gpa_problem (class_size : ℝ) (h_class_size_pos : class_size > 0) :
  let third_size := class_size / 3
  let rest_size := class_size - third_size
  let third_gpa := 60
  let overall_gpa := 64
  let rest_gpa := (overall_gpa * class_size - third_gpa * third_size) / rest_size
  rest_gpa = 66 := by sorry

end NUMINAMATH_CALUDE_classroom_gpa_problem_l1614_161430


namespace NUMINAMATH_CALUDE_shower_usage_solution_l1614_161460

/-- The water usage for Roman and Remy's showers -/
def shower_usage (R : ℝ) : Prop :=
  let remy_usage := 3 * R + 1
  R + remy_usage = 33 ∧ remy_usage = 25

/-- Theorem stating that there exists a value for Roman's usage satisfying the conditions -/
theorem shower_usage_solution : ∃ R : ℝ, shower_usage R := by
  sorry

end NUMINAMATH_CALUDE_shower_usage_solution_l1614_161460


namespace NUMINAMATH_CALUDE_total_flowers_collected_l1614_161414

/-- The maximum number of flowers each person can pick --/
def max_flowers : ℕ := 50

/-- The number of tulips Arwen picked --/
def arwen_tulips : ℕ := 20

/-- The number of roses Arwen picked --/
def arwen_roses : ℕ := 18

/-- The number of sunflowers Arwen picked --/
def arwen_sunflowers : ℕ := 6

/-- The number of tulips Elrond picked --/
def elrond_tulips : ℕ := 2 * arwen_tulips

/-- The number of roses Elrond picked --/
def elrond_roses : ℕ := min (3 * arwen_roses) (max_flowers - elrond_tulips)

/-- The number of tulips Galadriel picked --/
def galadriel_tulips : ℕ := min (3 * elrond_tulips) max_flowers

/-- The number of roses Galadriel picked --/
def galadriel_roses : ℕ := min (2 * arwen_roses) (max_flowers - galadriel_tulips)

/-- The number of sunflowers Legolas picked --/
def legolas_sunflowers : ℕ := arwen_sunflowers

/-- The number of roses Legolas picked --/
def legolas_roses : ℕ := (max_flowers - legolas_sunflowers) / 2

/-- The number of tulips Legolas picked --/
def legolas_tulips : ℕ := (max_flowers - legolas_sunflowers) / 2

theorem total_flowers_collected :
  arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips = 194 := by
  sorry

#eval arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips

end NUMINAMATH_CALUDE_total_flowers_collected_l1614_161414


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1614_161431

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x ^ 6 + Real.cos x ^ 6 + Real.sin x ^ 2 = 2 * Real.sin x ^ 4 + Real.cos x ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1614_161431


namespace NUMINAMATH_CALUDE_f_sum_zero_l1614_161443

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_property (x : ℝ) : f (2 - x) + f x = 0

-- State the theorem
theorem f_sum_zero : f 2022 + f 2023 = 0 := by sorry

end NUMINAMATH_CALUDE_f_sum_zero_l1614_161443


namespace NUMINAMATH_CALUDE_percentage_problem_l1614_161463

theorem percentage_problem (number : ℝ) (P : ℝ) : number = 15 → P = 20 / 100 * number + 47 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1614_161463


namespace NUMINAMATH_CALUDE_twelve_percent_of_700_is_84_l1614_161458

theorem twelve_percent_of_700_is_84 : ∃ x : ℝ, (12 / 100) * x = 84 ∧ x = 700 := by
  sorry

end NUMINAMATH_CALUDE_twelve_percent_of_700_is_84_l1614_161458


namespace NUMINAMATH_CALUDE_value_of_b_l1614_161489

theorem value_of_b (a b : ℝ) (h1 : a = 5) (h2 : a^2 + a*b = 60) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1614_161489


namespace NUMINAMATH_CALUDE_remainder_problem_l1614_161480

theorem remainder_problem : Int.mod (179 + 231 - 359) 37 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1614_161480


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_relationship_range_of_m_l1614_161452

def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(m-4) = 1 ∧ (m-1)*(m-4) < 0

def Q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-2) + y^2/(4-m) = 1 ∧ m-2 > 0 ∧ 4-m > 0 ∧ m-2 ≠ 4-m

theorem hyperbola_ellipse_relationship (m : ℝ) :
  (P m → Q m) ∧ ¬(Q m → P m) :=
sorry

theorem range_of_m (m : ℝ) :
  (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → ((1 < m ∧ m ≤ 2) ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_relationship_range_of_m_l1614_161452


namespace NUMINAMATH_CALUDE_possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l1614_161454

/-- A type representing a door or a key --/
def DoorKey := Fin 100

/-- A function representing the mapping of keys to doors --/
def KeyToDoor := DoorKey → DoorKey

/-- Predicate to check if a key-to-door mapping is valid --/
def IsValidMapping (f : KeyToDoor) : Prop :=
  ∀ k : DoorKey, (f k).val = k.val ∨ (f k).val = k.val + 1 ∨ (f k).val = k.val - 1

/-- Theorem stating that it's possible to determine the key-door mapping in 99 attempts --/
theorem possible_in_99_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 99 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 99 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's possible to determine the key-door mapping in 75 attempts --/
theorem possible_in_75_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 75 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 75 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's impossible to determine the key-door mapping in 74 attempts --/
theorem impossible_in_74_attempts :
  ∃ f : KeyToDoor, IsValidMapping f ∧
    ∀ (algorithm : ℕ → DoorKey × DoorKey),
      (∀ n : ℕ, n < 74 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
      ∃ k : DoorKey, ∀ n : ℕ, n < 74 → (algorithm n).1 ≠ k ∨ (algorithm n).2 ≠ f k :=
  sorry

end NUMINAMATH_CALUDE_possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l1614_161454


namespace NUMINAMATH_CALUDE_division_in_base5_l1614_161470

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem division_in_base5 :
  let dividend := base5ToBase10 [2, 3, 2, 3]  -- 3232 in base 5
  let divisor := base5ToBase10 [1, 2]         -- 21 in base 5
  let quotient := base5ToBase10 [0, 3, 1]     -- 130 in base 5
  let remainder := 2
  dividend = divisor * quotient + remainder ∧
  remainder < divisor ∧
  base10ToBase5 (dividend / divisor) = [0, 3, 1] ∧
  base10ToBase5 (dividend % divisor) = [2] :=
by sorry


end NUMINAMATH_CALUDE_division_in_base5_l1614_161470


namespace NUMINAMATH_CALUDE_bezout_identity_solutions_l1614_161402

theorem bezout_identity_solutions (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (u₀ v₀ : ℤ), ∀ (u v : ℤ),
    (a * u + b * v = Int.gcd a b) ↔ ∃ (k : ℤ), u = u₀ - k * b ∧ v = v₀ + k * a :=
by sorry

end NUMINAMATH_CALUDE_bezout_identity_solutions_l1614_161402


namespace NUMINAMATH_CALUDE_smallest_multiple_l1614_161442

theorem smallest_multiple (x : ℕ) : x = 40 ↔ 
  (x > 0 ∧ 
   800 ∣ (360 * x) ∧ 
   ∀ y : ℕ, y > 0 → 800 ∣ (360 * y) → x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1614_161442


namespace NUMINAMATH_CALUDE_roots_sum_cube_plus_linear_l1614_161492

theorem roots_sum_cube_plus_linear (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^3 + 5*β + 10 = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_cube_plus_linear_l1614_161492


namespace NUMINAMATH_CALUDE_overall_percentage_correct_l1614_161455

theorem overall_percentage_correct (score1 score2 score3 : ℚ)
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 75 / 100 ∧ 
  score2 = 85 / 100 ∧ 
  score3 = 60 / 100 ∧
  problems1 = 20 ∧
  problems2 = 50 ∧
  problems3 = 15 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3) = 79 / 100 := by
sorry

#eval (15 + 43 + 9) / (20 + 50 + 15)  -- Should evaluate to approximately 0.7882

end NUMINAMATH_CALUDE_overall_percentage_correct_l1614_161455


namespace NUMINAMATH_CALUDE_total_basketballs_l1614_161437

/-- Calculates the total number of basketballs for three basketball teams -/
theorem total_basketballs (spurs_players spurs_balls dynamos_players dynamos_balls lions_players lions_balls : ℕ) :
  spurs_players = 22 →
  spurs_balls = 11 →
  dynamos_players = 18 →
  dynamos_balls = 9 →
  lions_players = 26 →
  lions_balls = 7 →
  spurs_players * spurs_balls + dynamos_players * dynamos_balls + lions_players * lions_balls = 586 :=
by
  sorry

#check total_basketballs

end NUMINAMATH_CALUDE_total_basketballs_l1614_161437


namespace NUMINAMATH_CALUDE_ana_driving_problem_l1614_161423

theorem ana_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (target_average_speed : ℝ) (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  additional_speed = 70 →
  target_average_speed = 60 →
  additional_distance = 70 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / additional_speed)) = target_average_speed :=
by
  sorry

end NUMINAMATH_CALUDE_ana_driving_problem_l1614_161423


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1614_161479

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (squares_diff_eq : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1614_161479


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1614_161420

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (2 - |x|) / (x + 2) = 0 → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1614_161420


namespace NUMINAMATH_CALUDE_upstream_downstream_time_relation_stream_speed_is_twelve_l1614_161428

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 36

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 12

/-- The time taken to row upstream is twice the time taken to row downstream -/
theorem upstream_downstream_time_relation (d : ℝ) (h : d > 0) :
  d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed)) :=
by sorry

/-- Proves that the stream speed is 12 kmph given the conditions -/
theorem stream_speed_is_twelve :
  stream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_relation_stream_speed_is_twelve_l1614_161428


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1614_161433

theorem quadratic_inequality_solution_set :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x : ℝ | -x^2 + 3*x - 2 ≥ 0} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1614_161433


namespace NUMINAMATH_CALUDE_bird_watching_average_l1614_161432

theorem bird_watching_average :
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds
  let num_people : ℕ := 3
  (total_birds : ℚ) / num_people = 9 := by sorry

end NUMINAMATH_CALUDE_bird_watching_average_l1614_161432


namespace NUMINAMATH_CALUDE_M_subset_N_l1614_161478

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem stating that M is a subset of N
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1614_161478


namespace NUMINAMATH_CALUDE_vector_dot_product_collinear_l1614_161421

/-- Given two vectors a and b in ℝ², prove that if they are collinear and have specific components, their dot product satisfies a certain equation. -/
theorem vector_dot_product_collinear (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3/2, 1]
  let b : Fin 2 → ℝ := ![3, k]
  (∃ (t : ℝ), b = t • a) →   -- Collinearity condition
  (a - b) • (2 • a + b) = -13 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_collinear_l1614_161421


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1614_161404

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1614_161404


namespace NUMINAMATH_CALUDE_brendas_mice_problem_l1614_161411

theorem brendas_mice_problem (total_litters : Nat) (mice_per_litter : Nat) 
  (fraction_to_robbie : Rat) (multiplier_to_pet_store : Nat) (fraction_to_feeder : Rat) :
  total_litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1 / 6 →
  multiplier_to_pet_store = 3 →
  fraction_to_feeder = 1 / 2 →
  (total_litters * mice_per_litter 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie * multiplier_to_pet_store) 
    * (1 - fraction_to_feeder) = 4 := by
  sorry

end NUMINAMATH_CALUDE_brendas_mice_problem_l1614_161411
