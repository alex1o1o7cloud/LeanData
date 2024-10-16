import Mathlib

namespace NUMINAMATH_CALUDE_nine_point_zero_one_closest_l3244_324424

def options : List ℝ := [10.01, 9.998, 9.9, 9.01]

def closest_to_nine (x : ℝ) : Prop :=
  ∀ y ∈ options, |x - 9| ≤ |y - 9|

theorem nine_point_zero_one_closest :
  closest_to_nine 9.01 := by sorry

end NUMINAMATH_CALUDE_nine_point_zero_one_closest_l3244_324424


namespace NUMINAMATH_CALUDE_side_e_length_l3244_324460

-- Define the triangle DEF
structure Triangle where
  D : Real
  E : Real
  F : Real
  d : Real
  e : Real
  f : Real

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.E = 4 * t.D ∧ t.d = 18 ∧ t.f = 27

-- State the theorem
theorem side_e_length (t : Triangle) 
  (h : triangle_conditions t) : t.e = 27 := by
  sorry

end NUMINAMATH_CALUDE_side_e_length_l3244_324460


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3244_324403

/-- Circle C₁ with center (4, 0) and radius 3 -/
def C₁ : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + p.2^2 = 9}

/-- Circle C₂ with center (0, 3) and radius 2 -/
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 4}

/-- The center of C₁ -/
def center₁ : ℝ × ℝ := (4, 0)

/-- The center of C₂ -/
def center₂ : ℝ × ℝ := (0, 3)

/-- The radius of C₁ -/
def radius₁ : ℝ := 3

/-- The radius of C₂ -/
def radius₂ : ℝ := 2

/-- Theorem: C₁ and C₂ are externally tangent -/
theorem circles_externally_tangent :
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 = (radius₁ + radius₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3244_324403


namespace NUMINAMATH_CALUDE_linda_money_l3244_324477

/-- Represents the price of a single jean in dollars -/
def jean_price : ℕ := 11

/-- Represents the price of a single tee in dollars -/
def tee_price : ℕ := 8

/-- Represents the number of tees sold in a day -/
def tees_sold : ℕ := 7

/-- Represents the number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- Calculates the total money Linda had at the end of the day -/
def total_money : ℕ := jean_price * jeans_sold + tee_price * tees_sold

theorem linda_money : total_money = 100 := by
  sorry

end NUMINAMATH_CALUDE_linda_money_l3244_324477


namespace NUMINAMATH_CALUDE_laura_weekly_driving_distance_l3244_324457

/-- Calculates the total miles driven by Laura per week -/
def total_miles_per_week (
  house_to_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (school_trips_per_week : ℕ)
  (supermarket_trips_per_week : ℕ) : ℕ :=
  let school_miles := house_to_school_round_trip * school_trips_per_week
  let supermarket_round_trip := house_to_school_round_trip + 2 * supermarket_extra_distance
  let supermarket_miles := supermarket_round_trip * supermarket_trips_per_week
  school_miles + supermarket_miles

/-- Laura's weekly driving distance theorem -/
theorem laura_weekly_driving_distance :
  total_miles_per_week 20 10 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_driving_distance_l3244_324457


namespace NUMINAMATH_CALUDE_rectangle_uncovered_area_l3244_324479

/-- The area of the portion of a rectangle not covered by four circles --/
theorem rectangle_uncovered_area (rectangle_length : ℝ) (rectangle_width : ℝ) (circle_radius : ℝ) :
  rectangle_length = 4 →
  rectangle_width = 8 →
  circle_radius = 1 →
  (rectangle_length * rectangle_width) - (4 * Real.pi * circle_radius ^ 2) = 32 - 4 * Real.pi := by
  sorry

#check rectangle_uncovered_area

end NUMINAMATH_CALUDE_rectangle_uncovered_area_l3244_324479


namespace NUMINAMATH_CALUDE_rectangle_area_l3244_324482

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3244_324482


namespace NUMINAMATH_CALUDE_complement_of_union_l3244_324488

open Set

def U : Set Nat := {1,2,3,4,5,6}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_of_union : (U \ (S ∪ T)) = {2,4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3244_324488


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_and_intersecting_l3244_324483

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- A point lies on a parametric line if there exists a t satisfying both equations -/
def lies_on (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- The main theorem -/
theorem vector_to_line_parallel_and_intersecting :
  let l : ParametricLine := { x := λ t => 5 * t + 1, y := λ t => 2 * t + 1 }
  let v : ℝ × ℝ := (12.5, 5)
  let w : ℝ × ℝ := (5, 2)
  parallel v w ∧ lies_on v l := by sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_and_intersecting_l3244_324483


namespace NUMINAMATH_CALUDE_complex_number_properties_l3244_324406

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3244_324406


namespace NUMINAMATH_CALUDE_min_value_A_over_C_l3244_324414

theorem min_value_A_over_C (x : ℝ) (A C : ℝ) (h1 : x^2 + 1/x^2 = A) (h2 : x + 1/x = C)
  (h3 : A > 0) (h4 : C > 0) (h5 : ∀ y : ℝ, y > 0 → y + 1/y ≥ 2) :
  A / C ≥ 1 ∧ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀^2 + 1/x₀^2) / (x₀ + 1/x₀) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_A_over_C_l3244_324414


namespace NUMINAMATH_CALUDE_percentage_calculation_l3244_324420

theorem percentage_calculation : (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3244_324420


namespace NUMINAMATH_CALUDE_expression_simplification_l3244_324470

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 3) (h3 : a ≠ -3) :
  (3 / (a - 3) - a / (a + 3)) * ((a^2 - 9) / a) = (-a^2 + 6*a + 9) / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3244_324470


namespace NUMINAMATH_CALUDE_green_tile_probability_l3244_324448

theorem green_tile_probability :
  let tiles := Finset.range 100
  let green_tiles := tiles.filter (fun n => (n + 1) % 7 = 3)
  (green_tiles.card : ℚ) / tiles.card = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_green_tile_probability_l3244_324448


namespace NUMINAMATH_CALUDE_remainder_equality_l3244_324469

theorem remainder_equality (a b : ℕ) (h1 : a ≠ b) (h2 : a > b) :
  ∃ (q1 q2 r : ℕ), a = (a - b) * q1 + r ∧ b = (a - b) * q2 + r ∧ r < a - b :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l3244_324469


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3244_324421

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3244_324421


namespace NUMINAMATH_CALUDE_complex_simplification_l3244_324402

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  5 * (1 + i^3) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3244_324402


namespace NUMINAMATH_CALUDE_problem_statement_l3244_324484

theorem problem_statement :
  let p := ∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x
  let q := ∃ x : ℝ, x^2 = 2 - x
  (¬p ∧ q) → (∃ x : ℝ, x^2 = 2 - x ∧ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3244_324484


namespace NUMINAMATH_CALUDE_gwen_bookcase_total_l3244_324478

def mystery_books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 6
def picture_books_per_shelf : ℕ := 5
def picture_shelves : ℕ := 4
def biography_books_per_shelf : ℕ := 3
def biography_shelves : ℕ := 3
def scifi_books_per_shelf : ℕ := 9
def scifi_shelves : ℕ := 2

def total_books : ℕ := 
  mystery_books_per_shelf * mystery_shelves +
  picture_books_per_shelf * picture_shelves +
  biography_books_per_shelf * biography_shelves +
  scifi_books_per_shelf * scifi_shelves

theorem gwen_bookcase_total : total_books = 89 := by
  sorry

end NUMINAMATH_CALUDE_gwen_bookcase_total_l3244_324478


namespace NUMINAMATH_CALUDE_water_height_after_transfer_l3244_324438

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of water in a rectangular tank given its dimensions and water height -/
def waterVolume (tank : TankDimensions) (waterHeight : ℝ) : ℝ :=
  tank.length * tank.width * waterHeight

/-- Theorem: The height of water in Tank A after transfer -/
theorem water_height_after_transfer (tankA : TankDimensions) (transferredVolume : ℝ) :
  tankA.length = 3 →
  tankA.width = 2 →
  tankA.height = 4 →
  transferredVolume = 12 →
  (waterVolume tankA (transferredVolume / (tankA.length * tankA.width))) = transferredVolume :=
by sorry

end NUMINAMATH_CALUDE_water_height_after_transfer_l3244_324438


namespace NUMINAMATH_CALUDE_perfect_square_solution_l3244_324409

theorem perfect_square_solution (t n : ℤ) : 
  (n > 0) → (n^2 + (4*t - 1)*n + 4*t^2 = 0) → ∃ k : ℤ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_solution_l3244_324409


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l3244_324498

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a b d : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib b = r * fib a ∧ fib d = r * fib b) →  -- Fₐ, Fᵦ, Fᵈ form an increasing geometric sequence
  a + b + d = 3000 →  -- Sum of indices is 3000
  b = a + 2 →  -- b - a = 2
  d = b + 2 →  -- d = b + 2
  a = 998 := by  -- Conclusion: a = 998
sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l3244_324498


namespace NUMINAMATH_CALUDE_propositions_truth_l3244_324473

-- Proposition 1
def proposition1 : Prop := ∃ a b : ℝ, a ≤ b ∧ a^2 > b^2

-- Proposition 2
def proposition2 : Prop := ∀ x y : ℝ, x = -y → x + y = 0

-- Proposition 3
def proposition3 : Prop := ∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4

theorem propositions_truth : ¬proposition1 ∧ proposition2 ∧ proposition3 := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l3244_324473


namespace NUMINAMATH_CALUDE_tina_book_expense_l3244_324446

def savings_june : ℤ := 27
def savings_july : ℤ := 14
def savings_august : ℤ := 21
def spent_on_shoes : ℤ := 17
def money_left : ℤ := 40

theorem tina_book_expense :
  ∃ (book_expense : ℤ),
    savings_june + savings_july + savings_august - book_expense - spent_on_shoes = money_left ∧
    book_expense = 5 := by
  sorry

end NUMINAMATH_CALUDE_tina_book_expense_l3244_324446


namespace NUMINAMATH_CALUDE_minimal_moves_l3244_324492

/-- Represents a permutation of 2n numbers -/
def Permutation (n : ℕ) := Fin (2 * n) → Fin (2 * n)

/-- Represents a move that can be applied to a permutation -/
inductive Move (n : ℕ)
  | swap : Fin (2 * n) → Fin (2 * n) → Move n
  | cyclic : Fin (2 * n) → Fin (2 * n) → Fin (2 * n) → Move n

/-- Applies a move to a permutation -/
def applyMove (n : ℕ) (p : Permutation n) (m : Move n) : Permutation n :=
  sorry

/-- Checks if a permutation is in increasing order -/
def isIncreasing (n : ℕ) (p : Permutation n) : Prop :=
  sorry

/-- The main theorem: n moves are necessary and sufficient -/
theorem minimal_moves (n : ℕ) :
  (∃ (moves : List (Move n)), moves.length = n ∧
    ∀ (p : Permutation n), ∃ (appliedMoves : List (Move n)),
      appliedMoves.length ≤ n ∧
      isIncreasing n (appliedMoves.foldl (applyMove n) p)) ∧
  (∀ (k : ℕ), k < n →
    ∃ (p : Permutation n), ∀ (moves : List (Move n)),
      moves.length ≤ k → ¬isIncreasing n (moves.foldl (applyMove n) p)) :=
  sorry

end NUMINAMATH_CALUDE_minimal_moves_l3244_324492


namespace NUMINAMATH_CALUDE_greatest_n_no_substring_divisible_by_9_l3244_324432

-- Define a function to check if a number is divisible by 9
def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

-- Define a function to get all integer substrings of a number
def integer_substrings (n : ℕ) : List ℕ := sorry

-- Define the property that no integer substring is divisible by 9
def no_substring_divisible_by_9 (n : ℕ) : Prop :=
  ∀ m ∈ integer_substrings n, ¬(divisible_by_9 m)

-- State the theorem
theorem greatest_n_no_substring_divisible_by_9 :
  (∀ k > 88888888, ¬(no_substring_divisible_by_9 k)) ∧
  (no_substring_divisible_by_9 88888888) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_no_substring_divisible_by_9_l3244_324432


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3244_324487

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3244_324487


namespace NUMINAMATH_CALUDE_sqrt_11_between_3_and_4_l3244_324472

theorem sqrt_11_between_3_and_4 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_between_3_and_4_l3244_324472


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l3244_324480

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 := by
  sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 := by
  sorry

theorem is_145_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l3244_324480


namespace NUMINAMATH_CALUDE_female_officers_count_l3244_324426

theorem female_officers_count (total_on_duty : ℕ) (female_duty_percent : ℚ) 
  (h1 : total_on_duty = 160)
  (h2 : female_duty_percent = 16 / 100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
    (female_duty_percent * total_female : ℚ) = total_on_duty := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3244_324426


namespace NUMINAMATH_CALUDE_dans_age_l3244_324464

theorem dans_age : 
  ∀ x : ℕ, (x + 18 = 5 * (x - 6)) → x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_age_l3244_324464


namespace NUMINAMATH_CALUDE_triangle_side_expression_simplification_l3244_324458

theorem triangle_side_expression_simplification
  (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |a - b - c| + |b - c + a| + |c - a - b| = a + 3*b - c :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_expression_simplification_l3244_324458


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3244_324437

theorem sine_cosine_inequality (α β : ℝ) (h : 0 < α + β ∧ α + β ≤ π) :
  (Real.sin α - Real.sin β) * (Real.cos α - Real.cos β) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3244_324437


namespace NUMINAMATH_CALUDE_f_g_inequality_l3244_324433

open Set
open Function
open Topology

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x > deriv g x)

-- State the theorem
theorem f_g_inequality (x : ℝ) (hx : x ∈ Ioo a b) :
  f x + g a > g x + f a := by sorry

end NUMINAMATH_CALUDE_f_g_inequality_l3244_324433


namespace NUMINAMATH_CALUDE_total_pears_picked_l3244_324495

def sara_pears : ℕ := 6
def tim_pears : ℕ := 5

theorem total_pears_picked : sara_pears + tim_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3244_324495


namespace NUMINAMATH_CALUDE_swimming_problem_l3244_324412

/-- Represents the daily swimming distances of Jamir, Sarah, and Julien -/
structure SwimmingDistances where
  julien : ℕ
  sarah : ℕ
  jamir : ℕ

/-- Calculates the total distance swam by all three swimmers in a week -/
def weeklyTotalDistance (d : SwimmingDistances) : ℕ :=
  7 * (d.julien + d.sarah + d.jamir)

/-- The swimming problem statement -/
theorem swimming_problem (d : SwimmingDistances) : 
  d.sarah = 2 * d.julien →
  d.jamir = d.sarah + 20 →
  weeklyTotalDistance d = 1890 →
  d.julien = 50 := by
  sorry

end NUMINAMATH_CALUDE_swimming_problem_l3244_324412


namespace NUMINAMATH_CALUDE_coordinate_sum_with_slope_l3244_324445

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 4,
    if the slope of segment AB is 3/4, then the sum of B's coordinates is 28/3. -/
theorem coordinate_sum_with_slope (x : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 4)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 4 = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_with_slope_l3244_324445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3244_324465

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 + a 12 = 90) →
  (a 10 - (1/3) * a 14 = 12) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3244_324465


namespace NUMINAMATH_CALUDE_balls_in_boxes_l3244_324468

/-- The number of ways to place 3 different balls into 4 boxes. -/
def total_ways : ℕ := 4^3

/-- The number of ways to place 3 different balls into the first 3 boxes. -/
def ways_without_fourth : ℕ := 3^3

/-- The number of ways to place 3 different balls into 4 boxes,
    such that the 4th box contains at least one ball. -/
def ways_with_fourth : ℕ := total_ways - ways_without_fourth

theorem balls_in_boxes : ways_with_fourth = 37 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l3244_324468


namespace NUMINAMATH_CALUDE_corrected_mean_l3244_324401

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong1 wrong2 correct1 correct2 : ℝ) 
  (h1 : n = 100)
  (h2 : original_mean = 56)
  (h3 : wrong1 = 38)
  (h4 : wrong2 = 27)
  (h5 : correct1 = 89)
  (h6 : correct2 = 73) :
  let incorrect_sum := n * original_mean
  let difference := (correct1 + correct2) - (wrong1 + wrong2)
  let corrected_sum := incorrect_sum + difference
  corrected_sum / n = 56.97 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3244_324401


namespace NUMINAMATH_CALUDE_wheel_rotation_l3244_324474

/-- Given three wheels A, B, and C with radii 35 cm, 20 cm, and 8 cm respectively,
    where wheel A rotates through an angle of 72°, and all wheels rotate without slipping,
    prove that wheel C rotates through an angle of 315°. -/
theorem wheel_rotation (r_A r_B r_C : ℝ) (θ_A θ_C : ℝ) : 
  r_A = 35 →
  r_B = 20 →
  r_C = 8 →
  θ_A = 72 →
  r_A * θ_A = r_C * θ_C →
  θ_C = 315 := by
  sorry

#check wheel_rotation

end NUMINAMATH_CALUDE_wheel_rotation_l3244_324474


namespace NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l3244_324447

def sum_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  (last - first) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l3244_324447


namespace NUMINAMATH_CALUDE_pizza_slices_l3244_324485

-- Define the number of slices in each pizza
def slices_per_pizza : ℕ := sorry

-- Define the total number of pizzas
def total_pizzas : ℕ := 2

-- Define the fractions eaten by each person
def bob_fraction : ℚ := 1/2
def tom_fraction : ℚ := 1/3
def sally_fraction : ℚ := 1/6
def jerry_fraction : ℚ := 1/4

-- Define the number of slices left over
def slices_left : ℕ := 9

theorem pizza_slices : 
  slices_per_pizza = 12 ∧
  (bob_fraction + tom_fraction + sally_fraction + jerry_fraction) * slices_per_pizza * total_pizzas = 
    slices_per_pizza * total_pizzas - slices_left :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_l3244_324485


namespace NUMINAMATH_CALUDE_bijective_function_property_l3244_324499

variable {V : Type*} [Fintype V]
variable (f g : V → V)
variable (S T : Set V)

def is_bijective (h : V → V) : Prop :=
  Function.Injective h ∧ Function.Surjective h

theorem bijective_function_property
  (hf : is_bijective f)
  (hg : is_bijective g)
  (hS : S = {w : V | f (f w) = g (g w)})
  (hT : T = {w : V | f (g w) = g (f w)})
  (hST : S ∪ T = Set.univ) :
  ∀ w : V, f w ∈ S ↔ g w ∈ S :=
by sorry

end NUMINAMATH_CALUDE_bijective_function_property_l3244_324499


namespace NUMINAMATH_CALUDE_lemon_ratio_l3244_324451

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  jayden = eli / 3 ∧
  levi + jayden + eli + ian = 115 ∧
  eli * 2 = ian

theorem lemon_ratio :
  ∀ levi jayden eli ian : ℕ,
    lemon_problem levi jayden eli ian →
    eli * 2 = ian :=
by
  sorry

end NUMINAMATH_CALUDE_lemon_ratio_l3244_324451


namespace NUMINAMATH_CALUDE_initial_red_marbles_l3244_324490

/-- Given a bag of red and green marbles with the following properties:
    1. The initial ratio of red to green marbles is 5:3
    2. After adding 15 red marbles and removing 9 green marbles, the new ratio is 3:1
    This theorem proves that the initial number of red marbles is 52.5 -/
theorem initial_red_marbles (r g : ℚ) : 
  r / g = 5 / 3 →
  (r + 15) / (g - 9) = 3 / 1 →
  r = 52.5 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l3244_324490


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3244_324455

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + b

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ), f a b (-1) = 0 →
  (b = 2*a - 1) ∧
  (a = -1 → ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f (-1) (-3) x ≤ f (-1) (-3) y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3244_324455


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3244_324434

theorem quadratic_form_ratio (j : ℝ) (c p q : ℝ) : 
  8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q → q / p = -119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3244_324434


namespace NUMINAMATH_CALUDE_probability_three_white_two_black_l3244_324491

/-- The number of white balls in the box -/
def white_balls : ℕ := 8

/-- The number of black balls in the box -/
def black_balls : ℕ := 9

/-- The total number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls drawn -/
def white_drawn : ℕ := 3

/-- The number of black balls drawn -/
def black_drawn : ℕ := 2

/-- The probability of drawing 3 white balls and 2 black balls -/
theorem probability_three_white_two_black :
  (Nat.choose white_balls white_drawn * Nat.choose black_balls black_drawn : ℚ) /
  (Nat.choose (white_balls + black_balls) drawn_balls : ℚ) = 672 / 2063 :=
sorry

end NUMINAMATH_CALUDE_probability_three_white_two_black_l3244_324491


namespace NUMINAMATH_CALUDE_segments_in_proportion_l3244_324423

/-- Four line segments are in proportion if the product of the outer segments
    equals the product of the inner segments -/
def are_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (4, 8, 5, 10) -/
def segment_set : Vector ℝ 4 := ⟨[4, 8, 5, 10], rfl⟩

/-- Theorem: The set of line segments (4, 8, 5, 10) is in proportion -/
theorem segments_in_proportion :
  are_in_proportion (segment_set.get 0) (segment_set.get 1) (segment_set.get 2) (segment_set.get 3) :=
by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l3244_324423


namespace NUMINAMATH_CALUDE_current_speed_l3244_324486

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current --/
theorem current_speed (v_upstream v_downstream : ℝ) (h1 : v_upstream = 2) (h2 : v_downstream = 5) :
  (v_downstream - v_upstream) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3244_324486


namespace NUMINAMATH_CALUDE_unique_root_formula_l3244_324463

/-- A quadratic polynomial with exactly one root -/
class UniqueRootQuadratic (g : ℝ → ℝ) : Prop where
  is_quadratic : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
  unique_root : ∃! x : ℝ, g x = 0

/-- The property that g(ax + b) + g(cx + d) has exactly one root -/
def has_unique_combined_root (g : ℝ → ℝ) (a b c d : ℝ) : Prop :=
  ∃! x : ℝ, g (a * x + b) + g (c * x + d) = 0

theorem unique_root_formula 
  (g : ℝ → ℝ) (a b c d : ℝ) 
  [UniqueRootQuadratic g] 
  (h₁ : has_unique_combined_root g a b c d) 
  (h₂ : a ≠ c) : 
  ∃ x₀ : ℝ, (∀ x, g x = 0 ↔ x = x₀) ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_formula_l3244_324463


namespace NUMINAMATH_CALUDE_sum_of_factors_60_l3244_324453

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_60 : sum_of_factors 60 = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_60_l3244_324453


namespace NUMINAMATH_CALUDE_ellipse_min_area_l3244_324471

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
    ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  ∃ k : ℝ, k = 1 ∧ π * a * b ≥ k * π :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_area_l3244_324471


namespace NUMINAMATH_CALUDE_parallel_slope_relation_l3244_324497

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define parallel relation for lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem parallel_slope_relation :
  ∀ (l₁ l₂ : Line),
    (parallel l₁ l₂ → l₁.slope = l₂.slope) ∧
    ∃ (l₃ l₄ : Line), l₃.slope = l₄.slope ∧ ¬parallel l₃ l₄ := by
  sorry

end NUMINAMATH_CALUDE_parallel_slope_relation_l3244_324497


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l3244_324476

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 700
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  let platform_crossing_time : ℝ := total_distance / train_speed
  platform_crossing_time = 190 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l3244_324476


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3244_324439

/-- The set of available digits --/
def available_digits : Finset Nat := {0, 2, 4, 6}

/-- A function to check if a number is a valid three-digit number formed from the available digits --/
def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : Nat), a ∈ available_digits ∧ b ∈ available_digits ∧ c ∈ available_digits ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 100 * a + 10 * b + c

/-- The largest valid number --/
def largest_number : Nat := 642

/-- The smallest valid number --/
def smallest_number : Nat := 204

/-- Theorem: The sum of the largest and smallest valid numbers is 846 --/
theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 846 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3244_324439


namespace NUMINAMATH_CALUDE_projection_bound_implies_coverage_l3244_324493

/-- A figure in a metric space -/
class Figure (α : Type*) [MetricSpace α]

/-- The projection of a figure onto a line -/
def projection (α : Type*) [MetricSpace α] (Φ : Figure α) (l : Set α) : ℝ := sorry

/-- A figure Φ is covered by a circle of diameter d -/
def covered_by_circle (α : Type*) [MetricSpace α] (Φ : Figure α) (d : ℝ) : Prop := sorry

theorem projection_bound_implies_coverage 
  (α : Type*) [MetricSpace α] (Φ : Figure α) :
  (∀ l : Set α, projection α Φ l ≤ 1) →
  (¬ covered_by_circle α Φ 1) ∧ (covered_by_circle α Φ 1.5) := by sorry

end NUMINAMATH_CALUDE_projection_bound_implies_coverage_l3244_324493


namespace NUMINAMATH_CALUDE_ellipse_properties_l3244_324415

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Properties of the ellipse and related points -/
structure EllipseProperties (E : Ellipse) where
  O : Point
  A : Point
  B : Point
  M : Point
  C : Point
  N : Point
  h_O : O.x = 0 ∧ O.y = 0
  h_A : A.x = E.a ∧ A.y = 0
  h_B : B.x = 0 ∧ B.y = E.b
  h_M : M.x = 2 * E.a / 3 ∧ M.y = E.b / 3
  h_OM_slope : (M.y - O.y) / (M.x - O.x) = Real.sqrt 5 / 10
  h_C : C.x = -E.a ∧ C.y = 0
  h_N : N.x = (B.x + C.x) / 2 ∧ N.y = (B.y + C.y) / 2
  h_symmetric : ∃ (S : Point), S.y = 13 / 2 ∧
    (S.x - N.x) * (E.a / E.b + E.b / E.a) = S.y + N.y

/-- The main theorem to prove -/
theorem ellipse_properties (E : Ellipse) (props : EllipseProperties E) :
  (Real.sqrt (E.a^2 - E.b^2) / E.a = 2 * Real.sqrt 5 / 5) ∧
  (E.a^2 = 45 ∧ E.b^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3244_324415


namespace NUMINAMATH_CALUDE_boris_clock_theorem_l3244_324494

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_valid_time (h m : ℕ) : Prop :=
  h ≤ 23 ∧ m ≤ 59

def satisfies_clock_conditions (h m : ℕ) : Prop :=
  is_valid_time h m ∧ digit_sum h + digit_sum m = 6 ∧ h + m = 15

def possible_times : Set (ℕ × ℕ) :=
  {(0,15), (1,14), (2,13), (3,12), (4,11), (5,10), (10,5), (11,4), (12,3), (13,2), (14,1), (15,0)}

theorem boris_clock_theorem :
  {(h, m) | satisfies_clock_conditions h m} = possible_times :=
sorry

end NUMINAMATH_CALUDE_boris_clock_theorem_l3244_324494


namespace NUMINAMATH_CALUDE_packets_in_box_l3244_324441

/-- The number of packets in a box of sugar substitute -/
def packets_per_box : ℕ := sorry

/-- The daily usage of sugar substitute packets -/
def daily_usage : ℕ := 2

/-- The number of days for which sugar substitute is needed -/
def duration : ℕ := 90

/-- The total cost of sugar substitute for the given duration -/
def total_cost : ℚ := 24

/-- The cost of one box of sugar substitute -/
def cost_per_box : ℚ := 4

/-- Theorem stating that the number of packets in a box is 30 -/
theorem packets_in_box :
  packets_per_box = 30 :=
by sorry

end NUMINAMATH_CALUDE_packets_in_box_l3244_324441


namespace NUMINAMATH_CALUDE_combined_population_l3244_324429

def wellington_population : ℕ := 900

def port_perry_population : ℕ := 7 * wellington_population

def lazy_harbor_population : ℕ := 2 * wellington_population + 600

def newbridge_population : ℕ := 3 * (port_perry_population - wellington_population)

theorem combined_population :
  port_perry_population + lazy_harbor_population + newbridge_population = 24900 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_l3244_324429


namespace NUMINAMATH_CALUDE_conditional_prob_one_jiuzhaigou_l3244_324440

/-- The number of attractions available for choice. -/
def num_attractions : ℕ := 5

/-- The probability that two people choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that exactly one person chooses Jiuzhaigou and they choose different attractions. -/
def prob_one_jiuzhaigou_different : ℚ := 8 / 25

/-- The conditional probability that exactly one person chooses Jiuzhaigou given that they choose different attractions. -/
theorem conditional_prob_one_jiuzhaigou (h : num_attractions = 5) :
  prob_one_jiuzhaigou_different / prob_different_attractions = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_one_jiuzhaigou_l3244_324440


namespace NUMINAMATH_CALUDE_jerry_cans_count_l3244_324435

/-- The number of cans Jerry can carry at once -/
def cans_per_trip : ℕ := 4

/-- The time in seconds it takes to drain 4 cans -/
def drain_time : ℕ := 30

/-- The time in seconds for a round trip to the sink/recycling bin -/
def round_trip_time : ℕ := 20

/-- The total time in seconds to throw all cans away -/
def total_time : ℕ := 350

/-- The time in seconds for one complete cycle (draining and round trip) -/
def cycle_time : ℕ := drain_time + round_trip_time

theorem jerry_cans_count : 
  (total_time / cycle_time) * cans_per_trip = 28 := by sorry

end NUMINAMATH_CALUDE_jerry_cans_count_l3244_324435


namespace NUMINAMATH_CALUDE_max_value_theorem_l3244_324456

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 46 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3244_324456


namespace NUMINAMATH_CALUDE_problem_solution_l3244_324410

open Real

noncomputable def f (x : ℝ) : ℝ :=
  ((1 + cos (2*x))^2 - 2*cos (2*x) - 1) / (sin (π/4 + x) * sin (π/4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  (1/2) * f x + sin (2*x)

theorem problem_solution :
  (f (-11*π/12) = Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≤ Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), g x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/4), g x = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3244_324410


namespace NUMINAMATH_CALUDE_equation_solution_l3244_324407

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / 6 = 4 / 3 - x ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3244_324407


namespace NUMINAMATH_CALUDE_leftover_value_l3244_324467

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 50

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters Kim has -/
def kim_quarters : ℕ := 95

/-- The number of dimes Kim has -/
def kim_dimes : ℕ := 183

/-- The number of quarters Mark has -/
def mark_quarters : ℕ := 157

/-- The number of dimes Mark has -/
def mark_dimes : ℕ := 328

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1 / 4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total value of leftover coins after making complete rolls -/
theorem leftover_value : 
  let total_quarters := kim_quarters + mark_quarters
  let total_dimes := kim_dimes + mark_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_l3244_324467


namespace NUMINAMATH_CALUDE_path_count_is_210_l3244_324459

/-- Number of paths on a grid from C to D -/
def num_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose total_steps up_steps

/-- Theorem: The number of different paths from C to D is 210 -/
theorem path_count_is_210 :
  num_paths 10 6 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_path_count_is_210_l3244_324459


namespace NUMINAMATH_CALUDE_pascals_triangle_sum_l3244_324431

theorem pascals_triangle_sum (n : ℕ) : 
  n = 51 → Nat.choose n 4 + Nat.choose n 6 = 18249360 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_sum_l3244_324431


namespace NUMINAMATH_CALUDE_perfect_squares_between_a_and_2a_l3244_324489

theorem perfect_squares_between_a_and_2a (a : ℕ) : 
  (a > 0) → 
  (∃ x : ℕ, x^2 > a ∧ (x+9)^2 < 2*a ∧ 
    ∀ y : ℕ, (y^2 > a ∧ y^2 < 2*a) → (x ≤ y ∧ y ≤ x+9)) →
  (481 ≤ a ∧ a ≤ 684) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_a_and_2a_l3244_324489


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l3244_324462

-- Define the solution set of |2x-3| < x
def solution_set : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define m and n based on the quadratic equation x^2 - mx + n = 0 with roots 1 and 3
def m : ℝ := 4
def n : ℝ := 3

-- Define the constraint for a, b, c
def abc_constraint (a b c : ℝ) : Prop := 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ a * b + b * c + a * c = 1

theorem solution_and_minimum_value :
  (m - n = 1) ∧
  (∀ a b c : ℝ, abc_constraint a b c → a + b + c ≥ Real.sqrt 3) ∧
  (∃ a b c : ℝ, abc_constraint a b c ∧ a + b + c = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l3244_324462


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l3244_324413

/-- Proves that the walking speed is 4 km/hr given the problem conditions -/
theorem walking_speed_calculation (total_distance : ℝ) (total_time : ℝ) (running_speed : ℝ) :
  total_distance = 8 →
  total_time = 1.5 →
  running_speed = 8 →
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time ∧
    walking_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l3244_324413


namespace NUMINAMATH_CALUDE_salon_customers_count_l3244_324418

/-- Represents the number of customers who made only one visit -/
def single_visit_customers : ℕ := 44

/-- Represents the number of customers who made two visits -/
def double_visit_customers : ℕ := 30

/-- Represents the number of customers who made three visits -/
def triple_visit_customers : ℕ := 10

/-- The cost of the first visit in a calendar month -/
def first_visit_cost : ℕ := 10

/-- The cost of each subsequent visit in the same calendar month -/
def subsequent_visit_cost : ℕ := 8

/-- The total revenue for the calendar month -/
def total_revenue : ℕ := 1240

theorem salon_customers_count :
  single_visit_customers + double_visit_customers + triple_visit_customers = 84 ∧
  first_visit_cost * (single_visit_customers + double_visit_customers + triple_visit_customers) +
  subsequent_visit_cost * (double_visit_customers + 2 * triple_visit_customers) = total_revenue :=
sorry

end NUMINAMATH_CALUDE_salon_customers_count_l3244_324418


namespace NUMINAMATH_CALUDE_complex_computation_l3244_324427

theorem complex_computation :
  let A : ℂ := 3 + 2*I
  let B : ℂ := -1 - 2*I
  let C : ℂ := 5*I
  let D : ℂ := 3 + I
  2 * (A - B + C + D) = 8 + 20*I :=
by sorry

end NUMINAMATH_CALUDE_complex_computation_l3244_324427


namespace NUMINAMATH_CALUDE_cookie_count_l3244_324400

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3244_324400


namespace NUMINAMATH_CALUDE_log_xy_value_l3244_324496

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^5) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 6/7 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l3244_324496


namespace NUMINAMATH_CALUDE_intersection_point_d_l3244_324454

/-- The function g(x) = 2x + c -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

theorem intersection_point_d (c d : ℤ) :
  g c (-4) = d ∧ g c d = -4 → d = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_d_l3244_324454


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3244_324422

/-- The minimum value of a quadratic function f(x) = ax^2 + (b + 5)x + c where a > 0 -/
theorem quadratic_minimum_value (a b c : ℝ) (ha : a > 0) :
  let f := fun x => a * x^2 + (b + 5) * x + c
  ∃ m, ∀ x, f x ≥ m ∧ ∃ x₀, f x₀ = m :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3244_324422


namespace NUMINAMATH_CALUDE_range_of_a_l3244_324481

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ↔ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3244_324481


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l3244_324425

-- Define the set of geometric solids
inductive GeometricSolid
  | Cone
  | Cylinder
  | TriangularPyramid
  | QuadrangularPrism

-- Define a predicate for having a quadrilateral front view
def hasQuadrilateralFrontView (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.QuadrangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ (solid : GeometricSolid),
    hasQuadrilateralFrontView solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.QuadrangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l3244_324425


namespace NUMINAMATH_CALUDE_solution_verification_l3244_324452

theorem solution_verification (x : ℝ) : 
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧ 
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧ 
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) := by
sorry

end NUMINAMATH_CALUDE_solution_verification_l3244_324452


namespace NUMINAMATH_CALUDE_determine_coins_in_38_bags_l3244_324444

/-- Represents a bag of coins -/
structure Bag where
  coins : ℕ
  inv : coins ≥ 1000

/-- Represents the state of all bags -/
def BagState := Fin 40 → Bag

/-- An operation that checks two bags and potentially removes a coin from one of them -/
def CheckOperation (state : BagState) (i j : Fin 40) : BagState := sorry

/-- Predicate to check if we know the exact number of coins in a bag -/
def KnowExactCoins (state : BagState) (i : Fin 40) : Prop := sorry

/-- The main theorem stating that it's possible to determine the number of coins in 38 out of 40 bags -/
theorem determine_coins_in_38_bags :
  ∃ (operations : List (Fin 40 × Fin 40)),
    operations.length ≤ 100 ∧
    ∀ (initial_state : BagState),
      let final_state := operations.foldl (fun state (i, j) => CheckOperation state i j) initial_state
      (∃ (unknown1 unknown2 : Fin 40), ∀ (i : Fin 40),
        i ≠ unknown1 → i ≠ unknown2 → KnowExactCoins final_state i) :=
sorry

end NUMINAMATH_CALUDE_determine_coins_in_38_bags_l3244_324444


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l3244_324442

def nth_odd_multiple_of_five (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_five : 
  nth_odd_multiple_of_five 15 = 145 := by sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l3244_324442


namespace NUMINAMATH_CALUDE_equal_fish_time_l3244_324461

def brent_fish (n : ℕ) : ℕ := 9 * 4^n

def gretel_fish (n : ℕ) : ℕ := 243 * 3^n

theorem equal_fish_time : ∃ (n : ℕ), n > 0 ∧ brent_fish n = gretel_fish n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_fish_time_l3244_324461


namespace NUMINAMATH_CALUDE_bicyclist_effective_speed_l3244_324411

/-- Calculates the effective speed of a bicyclist considering headwind -/
def effective_speed (initial_speed_ms : ℝ) (headwind_kmh : ℝ) : ℝ :=
  initial_speed_ms * 3.6 - headwind_kmh

/-- Proves that the effective speed of a bicyclist with an initial speed of 18 m/s
    and a headwind of 10 km/h is 54.8 km/h -/
theorem bicyclist_effective_speed :
  effective_speed 18 10 = 54.8 := by sorry

end NUMINAMATH_CALUDE_bicyclist_effective_speed_l3244_324411


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3244_324443

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {5, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3244_324443


namespace NUMINAMATH_CALUDE_log_eight_three_equals_five_twelve_l3244_324408

theorem log_eight_three_equals_five_twelve (x : ℝ) : 
  Real.log x / Real.log 8 = 3 → x = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_three_equals_five_twelve_l3244_324408


namespace NUMINAMATH_CALUDE_ball_max_height_l3244_324449

/-- The height function of the ball's path -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum value of f is equal to max_height -/
theorem ball_max_height : ∀ t : ℝ, f t ≤ max_height := by sorry

end NUMINAMATH_CALUDE_ball_max_height_l3244_324449


namespace NUMINAMATH_CALUDE_balls_sold_l3244_324450

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 → 
  cost_price = 60 → 
  loss = 5 * cost_price → 
  ∃ n : ℕ, n * cost_price - selling_price = loss ∧ n = 17 :=
by sorry

end NUMINAMATH_CALUDE_balls_sold_l3244_324450


namespace NUMINAMATH_CALUDE_tank_capacity_l3244_324466

/-- Represents a water tank with a given capacity --/
structure WaterTank where
  capacity : ℚ
  initial_fill : ℚ
  final_fill : ℚ
  added_water : ℚ

/-- Theorem stating the capacity of the tank given the conditions --/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initial_fill = 3 / 4)
  (h2 : tank.final_fill = 7 / 8)
  (h3 : tank.added_water = 5)
  (h4 : tank.initial_fill * tank.capacity + tank.added_water = tank.final_fill * tank.capacity) :
  tank.capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3244_324466


namespace NUMINAMATH_CALUDE_valid_selection_probability_l3244_324475

/-- Represents a glove with a color and handedness -/
structure Glove :=
  (color : Nat)
  (isLeft : Bool)

/-- Represents a pair of gloves -/
structure GlovePair :=
  (left : Glove)
  (right : Glove)

/-- The set of all glove pairs in the cabinet -/
def glovePairs : Finset GlovePair := sorry

/-- The total number of ways to select two gloves -/
def totalSelections : Nat := sorry

/-- The number of valid selections (one left, one right, different pairs) -/
def validSelections : Nat := sorry

/-- The probability of a valid selection -/
def probabilityValidSelection : Rat := sorry

theorem valid_selection_probability :
  glovePairs.card = 3 →
  (∀ p : GlovePair, p ∈ glovePairs → p.left.color = p.right.color) →
  (∀ p q : GlovePair, p ∈ glovePairs → q ∈ glovePairs → p ≠ q → p.left.color ≠ q.left.color) →
  probabilityValidSelection = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_valid_selection_probability_l3244_324475


namespace NUMINAMATH_CALUDE_chessboard_square_rectangle_ratio_l3244_324436

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles expressed as a fraction with relatively prime positive integers -/
def square_rectangle_ratio : ℚ := 19 / 135

theorem chessboard_square_rectangle_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = square_rectangle_ratio := by
  sorry

end NUMINAMATH_CALUDE_chessboard_square_rectangle_ratio_l3244_324436


namespace NUMINAMATH_CALUDE_regular_decagon_perimeter_l3244_324405

/-- A regular decagon is a polygon with 10 sides of equal length -/
def RegularDecagon := Nat

/-- The side length of a regular decagon -/
def sideLength (d : RegularDecagon) : ℝ := 3

/-- The perimeter of a regular decagon -/
def perimeter (d : RegularDecagon) : ℝ := 10 * sideLength d

theorem regular_decagon_perimeter (d : RegularDecagon) : 
  perimeter d = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_perimeter_l3244_324405


namespace NUMINAMATH_CALUDE_total_tiles_to_replace_l3244_324404

/-- Represents the layout of paths in the park -/
structure ParkPaths where
  horizontalLengths : List Nat
  verticalLengths : List Nat

/-- Calculates the total number of tiles needed for replacement -/
def totalTiles (paths : ParkPaths) : Nat :=
  let horizontalSum := paths.horizontalLengths.sum
  let verticalSum := paths.verticalLengths.sum
  let intersections := 16  -- This value is derived from the problem description
  horizontalSum + verticalSum - intersections

/-- The main theorem stating the total number of tiles to be replaced -/
theorem total_tiles_to_replace :
  ∃ (paths : ParkPaths),
    paths.horizontalLengths = [30, 50, 30, 20, 20, 50] ∧
    paths.verticalLengths = [20, 50, 20, 50, 50] ∧
    totalTiles paths = 374 :=
  sorry

end NUMINAMATH_CALUDE_total_tiles_to_replace_l3244_324404


namespace NUMINAMATH_CALUDE_chord_length_l3244_324419

theorem chord_length (r : ℝ) (h : r = 15) : 
  let chord_length : ℝ := 2 * (r^2 - (r/3)^2).sqrt
  chord_length = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3244_324419


namespace NUMINAMATH_CALUDE_white_square_arc_length_bound_l3244_324428

/-- Represents a circle on a chessboard --/
structure ChessboardCircle where
  center : ℝ × ℝ
  radius : ℝ
  encloses_white_square : Bool

/-- Represents the portion of a circle's circumference passing through white squares --/
def white_square_arc_length (c : ChessboardCircle) : ℝ := sorry

/-- The theorem to be proved --/
theorem white_square_arc_length_bound 
  (c : ChessboardCircle) 
  (h1 : c.radius = 1) 
  (h2 : c.encloses_white_square = true) : 
  white_square_arc_length c ≤ (1/3) * (2 * Real.pi * c.radius) := by
  sorry

end NUMINAMATH_CALUDE_white_square_arc_length_bound_l3244_324428


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3244_324416

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3244_324416


namespace NUMINAMATH_CALUDE_rehabilitation_centers_count_l3244_324417

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane. -/
def total_rehabilitation_centers (lisa jude han jane : ℕ) : ℕ :=
  lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited. -/
theorem rehabilitation_centers_count :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_rehabilitation_centers lisa jude han jane = 27 := by
  sorry

end NUMINAMATH_CALUDE_rehabilitation_centers_count_l3244_324417


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l3244_324430

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y ≥ 1/a + 4/b) → 1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l3244_324430
